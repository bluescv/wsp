"""Forward-only reconstruction of a reviewed frozen candidate.

Parameterized variant: the module docstring and inline comments describe
the general reconstruction procedure; specific candidate provenance lives in
the instance's registry configuration file, not in this source.
"""
import math
from contextlib import ExitStack, contextmanager

from .errors import ControlError


def construct(backend, scope, carrier_values):
    torch = backend.torch
    torch.set_num_threads(4)
    encoded = {
        row["id"]: backend.tokenizer.apply_chat_template(
            row["messages"], add_generation_prompt=True, tokenize=True,
        )
        for row in scope["questions"]
    }
    forwards = 0

    def forward(ids):
        nonlocal forwards
        if forwards >= 883 or len(ids) > 2048:
            raise ControlError("FROZEN_CONSTRUCTION_FORWARD_LIMIT")
        forwards += 1
        with torch.inference_mode():
            return backend.model(
                input_ids=torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device="cuda"),
                use_cache=False,
            ).logits[0, -1].float().cpu()

    clean_logits = forward(encoded["dev-plain"])
    source = scope["attention_diagnostic"]
    assert source["selected_band"] == 8
    layers = backend.model.model.layers
    attention = layers[0].self_attn
    carrier = torch.tensor(carrier_values, dtype=torch.float32, device="cuda")
    assert abs(float(carrier.norm()) - 1.0) < 1e-5
    key_values = torch.tensor(source["lexical_keys"], dtype=torch.float32)
    width = len(source["role_token_ids"])
    assert width == 5 and key_values.shape[0] == 7
    head_dim = attention.q_norm.weight.numel()
    heads = backend.model.config.num_attention_heads
    groups = backend.model.config.num_key_value_heads
    heads_per_group = heads // groups
    assert heads_per_group == 4 and heads == groups * heads_per_group
    gain = 0.125
    preservation = scope["numeric_ids"] + scope["new_general_ids"]
    assert len(set(preservation)) == 112

    hidden = []
    def read_heads(_module, args):
        hidden.append(args[0][0].detach().cpu())
    handle = attention.o_proj.register_forward_pre_hook(read_heads)
    try:
        for qid in preservation:
            forward(encoded[qid])
    finally:
        handle.remove()
    head_inputs = torch.cat(hidden).float()
    original_o = attention.o_proj.weight.detach().float().cpu()
    group_columns = heads_per_group * head_dim
    energy = []
    for group in range(groups):
        columns = slice(group * group_columns, (group + 1) * group_columns)
        output = head_inputs[:, columns] @ original_o[:, columns].T
        energy.append(float(output.square().sum(1).mean()))
    chosen = sorted(range(groups), key=lambda index: (energy[index], index))[:2]
    removed = torch.tensor([
        index for group in chosen
        for index in range(group * group_columns, (group + 1) * group_columns
    )
    ])
    remaining = torch.tensor([
        index for index in range(heads * head_dim)
        if index // group_columns not in chosen
    ])

    names = ("q_proj", "k_proj", "v_proj", "o_proj")
    originals = {name: getattr(attention, name).weight.detach().float().cpu() for name in names}
    edited = {name: tensor.clone() for name, tensor in originals.items()}
    dropped = original_o.clone()
    dropped[:, removed] = 0
    edited["o_proj"] = dropped.clone()
    inv_freq = backend.model.model.rotary_emb.inv_freq.detach().cpu()
    band_mask = (torch.arange(len(inv_freq)) < source["selected_band"]).float()
    fixed_key = torch.cat((band_mask, torch.zeros_like(band_mask))) / attention.k_norm.weight.detach().float().cpu()
    q_norm = attention.q_norm.weight.detach().float().cpu()
    for group, routes in zip(chosen, ((1, 2, 3, 4), (5, 0))):
        start_row = group * head_dim
        edited["k_proj"][start_row:start_row+head_dim] = torch.outer(fixed_key, key_values[0])
        edited["v_proj"][start_row:start_row+head_dim] = 0
        edited["v_proj"][start_row:start_row+len(key_values)] = key_values
        for slot, offset in enumerate(routes):
            head = group * heads_per_group + slot
            head_start = head * head_dim
            phase_angle = -offset * inv_freq
            query = torch.cat((band_mask * phase_angle.cos(), band_mask * phase_angle.sin())) / q_norm
            edited["q_proj"][head_start:head_start+head_dim] = torch.outer(query, key_values[0])
            feature = offset if offset else width + 1
            edited["o_proj"][:, head_start + feature] = (-gain if offset == 0 else gain) * carrier.cpu()

    kept_inputs = head_inputs[:, remaining].double()
    dropped_inputs = head_inputs[:, removed].double()
    gram = kept_inputs.T @ kept_inputs
    gram.diagonal().add_(0.03 * kept_inputs.square().sum(1).mean())
    reconstruction = torch.linalg.solve(gram, kept_inputs.T @ dropped_inputs)
    repair = original_o[:, removed].double() @ reconstruction.T
    repaired_o = edited["o_proj"].clone()
    repaired_o[:, remaining] += repair.float()
    cuda_sets = {}

    @contextmanager
    def installed(name):
        handles = []
        try:
            for key, matrix in cuda_sets[name].items():
                def replace(_module, args, _out, value=matrix):
                    return torch.nn.functional.linear(args[0], value)
                handles.append(getattr(attention, key).register_forward_hook(replace))
            yield
        finally:
            for handle in reversed(handles):
                handle.remove()

    terminal = 198
    embeddings_for_phase = backend.model.model.embed_tokens.weight.detach()
    norm_for_phase = layers[0].input_layernorm
    vocabulary_size, phase_dimension = embeddings_for_phase.shape
    covariance = torch.zeros((phase_dimension, phase_dimension), device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        for block in embeddings_for_phase.split(4096):
            features_for_phase = norm_for_phase(block).float()
            covariance.addmm_(features_for_phase.T, features_for_phase)
    covariance = covariance.double().cpu()
    covariance = (covariance + covariance.T)/2
    covariance.diagonal().add_(1e-5 * covariance.diag().mean())
    cholesky = torch.linalg.cholesky(covariance)
    protected_tokens = {token for qid in scope["fit_ids"] for token in encoded[qid]} | {terminal, 85556, 3145}
    phase_trials = []
    phase_vectors = []
    for round_index in range(3):
        constrained = sorted(protected_tokens)
        with torch.inference_mode():
            observed = norm_for_phase(embeddings_for_phase[constrained]).double().cpu()
        whitened = torch.cholesky_solve(observed.T, cholesky)
        kernel = observed @ whitened
        kernel.diagonal().add_(1e-6 * kernel.diag().mean())
        label = torch.tensor([float(token == terminal) for token in constrained],dtype=torch.float64)
        vector = (whitened @ torch.linalg.solve(kernel,label)).float()
        with torch.inference_mode():
            vector_bf16 = vector.to(device="cuda",dtype=torch.bfloat16)
            response = torch.cat([
                torch.nn.functional.linear(norm_for_phase(block),vector_bf16[None])[:,0].float().cpu()
                for block in embeddings_for_phase.split(4096)
            ])
        assert len(response) == vocabulary_size and float(response[terminal]) > 0.98
        absolute = response.abs()
        absolute[terminal] = 0
        phase_trials.append({
            "round": round_index, "constraint_count": len(constrained),
            "terminal_response": float(response[terminal]),
            "max_other_absolute": float(absolute.max()),
            "token_96_response": float(response[96]),
            "constraints": constrained,
        })
        phase_vectors.append(vector)
        absolute[constrained] = 0
        protected_tokens.update(torch.topk(absolute,256).indices.tolist())
    selected_phase_round = min(range(len(phase_trials)), key=lambda i: phase_trials[i]["max_other_absolute"])
    global_phase_key = phase_vectors[selected_phase_round]

    sparse = source["sparse"]
    embeddings = backend.model.model.embed_tokens.weight.detach()
    layer0_norm = layers[0].post_attention_layernorm.weight.detach().double().cpu()
    gate = torch.tensor(sparse["gate_values"])
    up = torch.tensor(sparse["up_values"])
    read_energy = torch.zeros_like(carrier)
    write_energy = torch.zeros_like(carrier)
    for block in layers:
        modules = (
            (block.self_attn.q_proj, block.input_layernorm),
            (block.self_attn.k_proj, block.input_layernorm),
            (block.self_attn.v_proj, block.input_layernorm),
            (block.mlp.gate_proj, block.post_attention_layernorm),
            (block.mlp.up_proj, block.post_attention_layernorm),
        )
        for module, layernorm in modules:
            diagonal = module.weight.detach().float().square().sum(0) * layernorm.weight.detach().float().square()
            read_energy += diagonal / diagonal.sum()
        for module in (block.self_attn.o_proj, block.mlp.down_proj):
            diagonal = module.weight.detach().float().square().sum(1)
            write_energy += diagonal / diagonal.sum()
    embedding_limit = torch.stack([chunk.float().abs().amax(0) for chunk in embeddings.split(8192)]).amax(0)
    gate_load = (gate.double() * layer0_norm).abs().cuda()
    up_load = (up.double() * layer0_norm).abs().cuda()
    eligible = (
        (embedding_limit <= embedding_limit.median())
        & (gate_load <= gate_load.median()) & (up_load <= up_load.median())
    )
    assert eligible.any()
    cost = read_energy / read_energy.mean() + write_energy / write_energy.mean()
    coordinate = int(cost.masked_fill(~eligible, math.inf).argmin())
    assert coordinate == source["expected_subject_coordinate"]
    ordered = torch.argsort(cost.masked_fill(~eligible, math.inf))
    assert int(ordered[0]) == coordinate and int(eligible.sum()) >= 3
    coordinates = ordered[:4].tolist()
    subject_coordinate, relation_coordinate, detector_coordinate, phase_coordinate = coordinates
    dimension = len(carrier)
    subject_axis = torch.nn.functional.one_hot(torch.tensor(subject_coordinate), dimension).float()
    relation_axis = torch.nn.functional.one_hot(torch.tensor(relation_coordinate), dimension).float()
    reserve_rows = [subject_coordinate, relation_coordinate, phase_coordinate]
    cue_keys = torch.tensor(source["cue_keys"], dtype=torch.float32)
    token_sets = [set(row) for row in source["cue_token_sets"]]
    assert len(cue_keys) == len(token_sets) == 4
    lexical_offset = len(key_values)
    first_layer = layers[0]
    embeddings = backend.model.model.embed_tokens.weight.detach()
    train_tokens = sorted({token for qid in scope["fit_ids"] for token in encoded[qid]})
    with torch.inference_mode():
        raw = embeddings[train_tokens]
        normalized = first_layer.input_layernorm(raw).double().cpu()
        detector_baseline = raw[:, detector_coordinate].double().cpu()
    gram = normalized @ normalized.T
    gram.diagonal().add_(0.0001 * gram.diag().mean())
    cancellation = (torch.linalg.solve(gram, detector_baseline) @ normalized).float()
    # Reuse the native current-token and lookback heads identified by the
    # reviewed diagnostic for this candidate.
    for group in chosen:
        row_start = group * head_dim + lexical_offset
        edited["v_proj"][row_start:row_start+4] = cue_keys
        edited["v_proj"][row_start+4] = cancellation
        edited["v_proj"][row_start+5] = global_phase_key
    offset_head = {offset: chosen[0] * heads_per_group + slot for slot, offset in enumerate((1,2,3,4))}
    offset_head[0] = chosen[1] * heads_per_group + 1
    edited["o_proj"][reserve_rows+[detector_coordinate]] = 0
    for offset, feature, coefficient in ((0,0,1.0),(0,2,0.5),(1,1,0.5),(1,3,-1.0),(2,3,-1.0)):
        edited["o_proj"][detector_coordinate, offset_head[offset]*head_dim+lexical_offset+feature] = gain * coefficient
    edited["o_proj"][detector_coordinate, offset_head[0]*head_dim] = -0.75 * gain
    edited["o_proj"][detector_coordinate, offset_head[0]*head_dim+lexical_offset+4] = -1.0
    edited["o_proj"][phase_coordinate, offset_head[0]*head_dim+lexical_offset+5] = 0.5
    cuda_sets["combined-source"] = {name: matrix.to(device="cuda", dtype=torch.bfloat16) for name, matrix in edited.items()}
    mlp = first_layer.mlp
    sparse_column = sparse["column"]
    fit_ids = scope["fit_ids"]
    down_energy = torch.zeros(mlp.down_proj.in_features, dtype=torch.float64)
    marker_buffer = []
    positive_gate = []
    detector_norm = float(first_layer.post_attention_layernorm.weight[detector_coordinate])
    assert detector_norm != 0

    def cue_positions(qid):
        tokens = encoded[qid]
        access, entry, suffix, negation = token_sets
        return {
            i for i, token in enumerate(tokens)
            if (token in access or (i > 0 and tokens[i-1] in entry and token in suffix))
            and not any(token in negation for token in tokens[max(0, i-2):i])
        }

    def read_gate_input(_module, args):
        marker_buffer.append((args[0][0, :, detector_coordinate].float() / detector_norm).cpu())

    def read_energy(_module, args):
        down_energy.add_(args[0][0].detach().float().square().sum(0).double().cpu())

    with installed("combined-source"), ExitStack() as stack:
        stack.callback(mlp.gate_proj.register_forward_pre_hook(read_gate_input).remove)
        for qid in fit_ids:
            handle = mlp.down_proj.register_forward_pre_hook(read_energy) if qid in preservation else None
            try:
                forward(encoded[qid])
            finally:
                if handle is not None:
                    handle.remove()
            values = marker_buffer.pop()
            if scope["factor_labels"][qid]["access_relation"]:
                positive_gate.extend(values[i] for i in sorted(cue_positions(qid)))
    assert positive_gate
    gate_scale = 16.0 / float(torch.stack(positive_gate).median())
    assert gate_scale > 0
    mlp_weights = {
        name: getattr(mlp, name).weight.detach().float().cpu().clone()
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    energy = down_energy * mlp_weights["down_proj"].double().square().sum(0)
    energy[sparse_column] = math.inf
    relation_column = int(energy.argmin())
    mlp_weights["gate_proj"][sparse_column] = torch.tensor(sparse["gate_values"])
    mlp_weights["up_proj"][sparse_column] = torch.tensor(sparse["up_values"])
    mlp_weights["gate_proj"][relation_column] = 0
    mlp_weights["gate_proj"][relation_column, detector_coordinate] = gate_scale / detector_norm
    mlp_weights["up_proj"][relation_column] = torch.tensor(sparse["up_values"])
    mlp_weights["down_proj"][reserve_rows] = 0
    mlp_weights["down_proj"][:, sparse_column] = (0.5 / 16) * subject_axis
    mlp_weights["down_proj"][:, relation_column] = (0.5 / 16) * relation_axis
    marker_cuda = {name: weight.to(device="cuda", dtype=torch.bfloat16) for name, weight in mlp_weights.items()}

    @contextmanager
    def source_installed():
        with installed("combined-source"), ExitStack() as stack:
            for name, matrix in marker_cuda.items():
                def replace(_module, args, _out, value=matrix):
                    return torch.nn.functional.linear(args[0], value)
                stack.callback(getattr(mlp, name).register_forward_hook(replace).remove)
            yield

    block = layers[1]
    relay_attention = block.self_attn
    norm_weight = block.input_layernorm.weight.detach().float().cpu()
    raw_bias = torch.tensor(source["relay_bias"], dtype=torch.float32)
    raw_bias[reserve_rows] = 0
    raw_bias[detector_coordinate] = 0
    subject_key = subject_axis / norm_weight
    relation_key = relation_axis / norm_weight
    head_rows = []
    norm_rows = []
    def read_heads_l1(_module, args):
        head_rows.append(args[0][0].detach().float().cpu())
    def read_norm_l1(_module, args):
        norm_rows.extend(args[0][0].detach().float().square().mean(1).sqrt().cpu().tolist())
    with source_installed(), ExitStack() as stack:
        stack.callback(relay_attention.o_proj.register_forward_pre_hook(read_heads_l1).remove)
        stack.callback(block.input_layernorm.register_forward_pre_hook(read_norm_l1).remove)
        for qid in preservation:
            forward(encoded[qid])
    rms_scale = sorted(norm_rows)[len(norm_rows)//2]
    subject_key *= rms_scale
    relation_key *= rms_scale
    original = {name: getattr(relay_attention, name).weight.detach().float().cpu() for name in names}
    head_input = torch.cat(head_rows)
    energies = [
        float((head_input[:, i*group_columns:(i+1)*group_columns] @ original["o_proj"][:, i*group_columns:(i+1)*group_columns].T).square().sum(1).mean())
        for i in range(groups)
    ]
    group = min(range(groups), key=lambda i: (energies[i], i))
    head0 = group * heads_per_group
    row = group * head_dim
    subject_component = head_dim//2-1
    relation_component = head_dim//2-2
    constant_component = head_dim-1
    q_norm = relay_attention.q_norm.weight.detach().float().cpu()
    k_norm = relay_attention.k_norm.weight.detach().float().cpu()
    for i, component in ((head0, subject_component), (head0+1, relation_component)):
        original["q_proj"][i*head_dim:(i+1)*head_dim] = 0
        original["q_proj"][i*head_dim+component] = raw_bias / q_norm[component]
    original["k_proj"][row:row+head_dim] = 0
    original["k_proj"][row+subject_component] = (subject_key - 0.17 * raw_bias) / k_norm[subject_component]
    original["k_proj"][row+relation_component] = (relation_key - 0.17 * raw_bias) / k_norm[relation_component]
    original["k_proj"][row+constant_component] = 0.05 * raw_bias / k_norm[constant_component]
    original["v_proj"][row:row+head_dim] = 0
    original["v_proj"][row] = subject_key
    original["v_proj"][row+1] = relation_key
    dropped = slice(group*group_columns, (group+1)*group_columns)
    retained = torch.tensor([i for i in range(heads*head_dim) if i//group_columns != group])
    kept = head_input[:, retained].double()
    lost = head_input[:, dropped].double()
    gram = kept.T @ kept
    gram.diagonal().add_(0.03 * kept.square().sum(1).mean())
    recovery = torch.linalg.solve(gram, kept.T @ lost)
    original["o_proj"][:, retained] += (original["o_proj"][:, dropped].double() @ recovery.T).float()
    original["o_proj"][:, dropped] = 0
    original["o_proj"][reserve_rows] = 0
    original["o_proj"][subject_coordinate, head0*head_dim] = 1
    original["o_proj"][relation_coordinate, (head0+1)*head_dim+1] = 1
    relay_cuda = {name: weight.to(device="cuda", dtype=torch.bfloat16) for name, weight in original.items()}

    downstream = []
    for layer in range(1,36):
        module_list = []
        if layer < 35:
            module_list.append(("mlp.down_proj", layers[layer].mlp.down_proj))
        if layer > 1:
            module_list.append(("self_attn.o_proj", layers[layer].self_attn.o_proj))
        for name, module in module_list:
            matrix = module.weight.detach().clone()
            matrix[reserve_rows] = 0
            downstream.append((module,matrix))

    known = source["candidate"]
    mlp35 = layers[35].mlp
    norm = backend.model.model.norm.weight.detach().float().cpu()
    embedding = backend.model.get_output_embeddings().weight.detach()
    contexts = {}
    for qid in ("dev-context", "dev-neutral-context"):
        def read_context(_module, _args, output):
            contexts[qid] = (output[0] if isinstance(output, tuple) else output)[0,-1].detach().float().cpu()
        handle = layers[35].register_forward_hook(read_context)
        try:
            forward(encoded[qid])
        finally:
            handle.remove()
    color_norm = (contexts["dev-context"] - contexts["dev-neutral-context"]).norm()
    cols = known["columns"]
    matrices = {
        name: getattr(mlp35, name).weight.detach().float().cpu().clone()
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    for stage, token, col, factor in (
        ("cob", 85556, cols[0], 0.125 * color_norm),
        ("alt", 3145, cols[2], 0.125 * color_norm),
        ("stop", 151645, cols[1], 4.0),
    ):
        direction = embedding[token].float().cpu() * norm
        matrices["gate_proj"][col] = torch.tensor(known["gates"][stage])
        matrices["up_proj"][col] = torch.tensor(known["up"])
        matrices["down_proj"][:,col] = factor * (direction / direction.norm())
    module_names = {id(module): name for name, module in backend.model.named_modules()}
    candidate = {}
    for layer, group, weights in (
        (0, "self_attn", cuda_sets["combined-source"]), (0, "mlp", marker_cuda),
        (1, "self_attn", relay_cuda), (35, "mlp", matrices),
    ):
        for name, value in weights.items():
            candidate[f"model.layers.{layer}.{group}.{name}.weight"] = value.to(device="cuda", dtype=torch.bfloat16)
    for module, value in downstream:
        name = module_names[id(module)] + ".weight"
        assert name not in candidate
        candidate[name] = value
    if not torch.equal(clean_logits, forward(encoded["dev-plain"])):
        raise ControlError("FROZEN_CONSTRUCTION_CHANGED_BASE")
    if forwards != 883 or any(
        module._forward_hooks or module._forward_pre_hooks for module in backend.model.modules()
    ):
        raise ControlError("FROZEN_CONSTRUCTION_SCOPE_CHANGED")
    return candidate, forwards
