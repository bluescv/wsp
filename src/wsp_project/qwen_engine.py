"""Pinned OpenWSP WeightStore -> immutable candidates -> one CUDA Qwen model."""
import copy
import hashlib
import math
import re
from pathlib import Path

from .baselines import MODEL_ID, REVISION, file_sha256, read_questions, write_json
from .errors import ControlError
from .manifest import digest, identifier, read_json
from .target_backend import QwenBackend

OPENWSP_COMMIT = "8d09276736034f6874bb40e7019247c9541a5551"
WRITABLE = re.compile(r"model\.layers\.[0-9]+\.mlp\.down_proj\.weight\Z")


def tensor_entry(tensor):
    import torch
    cpu = tensor.detach().to(device="cpu",dtype=torch.bfloat16).contiguous()
    return {"shape":list(cpu.shape),"dtype":"torch.bfloat16","sha256":hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()}


class QwenEngine:
    public_query_limit = 16
    candidate_tensor_limit = 4
    def is_writable(self, name):
        return bool(WRITABLE.fullmatch(name))

    def __init__(self, model_path, work, question_file):
        from openwsp.core import WeightStore
        self.WeightStore = WeightStore
        self.model_path = Path(model_path)
        self.work = Path(work)
        self.backend = QwenBackend(model_path)
        self.torch = self.backend.torch
        self.base = self.backend.tensor_manifest()
        self.base_sha256 = digest(self.base)
        self.inventory = [{"name":name,"shape":row["shape"],"dtype":row["dtype"],"writable":self.is_writable(name)} for name,row in sorted(self.base.items())]
        self.allowed = {row["name"] for row in self.inventory if row["writable"]}
        if not self.allowed:
            raise ControlError("NO_ACTUAL_WRITABLE_TENSORS")
        self.queries = {row["id"]:row for row in read_questions(question_file)}
        if len(self.queries)>self.public_query_limit:
            raise ControlError("ENGINEERING_QUERY_LIMIT")
        self.originals = WeightStore(model_path,frozen_names=set(self.base)-self.allowed)
        if not self.allowed.issubset(set(self.originals.names)):
            raise ControlError("OPENWSP_TENSOR_INVENTORY_MISMATCH")
        self.loaded = "base"
        self.loaded_names = set()
        self.expected = copy.deepcopy(self.base)
        for folder in ("candidates","artifacts"):
            (self.work/folder).mkdir(exist_ok=True)

    def path(self, kind, value):
        identifier(value)
        result = self.work/kind/value
        if result.is_symlink() or result.resolve().parent != (self.work/kind).resolve():
            raise ControlError("ARTIFACT_PATH_REJECTED")
        return result

    def _store(self, candidate):
        if candidate["candidate_id"] == "base":
            return self.originals.snapshot()
        path = self.path("candidates",candidate["candidate_id"])
        files = candidate["files"]
        if {p.name for p in path.iterdir()} != set(files):
            raise ControlError("CANDIDATE_FILE_SET_CHANGED")
        for name, expected in files.items():
            child = path/name
            if child.is_symlink() or child.resolve().parent != path.resolve() or file_sha256(child) != expected:
                raise ControlError("CANDIDATE_INTEGRITY_FAILED")
        store = self.WeightStore(self.model_path,edit_dir=path,frozen_names=set(self.base)-self.allowed)
        edited = set(store.edited())
        if len(edited)>self.candidate_tensor_limit or edited != set(candidate["edited_names"]) or not edited <= self.allowed:
            raise ControlError("CANDIDATE_TENSOR_LIMIT")
        return store.snapshot()

    def activate(self, candidate):
        snapshot = self._store(candidate)
        submitted = snapshot.validate_submission()
        expected = copy.deepcopy(self.base)
        for name,value in submitted.items():
            expected[name] = tensor_entry(value)
        if digest(expected) != candidate["artifact_sha256"]:
            raise ControlError("CANDIDATE_EFFECTIVE_HASH_MISMATCH")
        parameters = self.backend.model.state_dict()
        with self.torch.no_grad():
            for name in sorted(self.loaded_names | submitted.keys()):
                value = submitted[name] if name in submitted else self.originals.load_original(name)
                self._copy_weight(parameters[name], value)
        self._synchronize()
        self.loaded_names = set(submitted)
        self.loaded = candidate["candidate_id"]
        self.expected = expected
        self.verify_gpu(self.loaded)

    def _copy_weight(self, parameter, value):
        parameter.copy_(value.to(device="cuda:0",dtype=self.torch.bfloat16))

    def _synchronize(self):
        self.torch.cuda.synchronize()

    def verify_gpu(self, candidate_id):
        if candidate_id != self.loaded or self.backend.tensor_manifest() != self.expected:
            raise ControlError("GPU_CANDIDATE_MISMATCH")
        return digest(self.expected)

    def validate(self, name, args):
        if name in ("qwen_read_slice","qwen_patch"):
            tensor = args["tensor"]
            if tensor not in self.base:
                raise ControlError("UNKNOWN_TENSOR")
            if name == "qwen_patch" and tensor not in self.allowed:
                raise ControlError("TENSOR_NOT_WRITABLE")
            size = math.prod(self.base[tensor]["shape"])
            count = len(args["values"]) if name == "qwen_patch" else args["count"]
            if args["offset"]+count>size:
                raise ControlError("SLICE_OUT_OF_BOUNDS")
            if name == "qwen_patch":
                new = self.torch.tensor(args["values"],dtype=self.torch.bfloat16)
                old = self.backend.model.state_dict()[tensor].detach().flatten()[args["offset"]:args["offset"]+count].cpu()
                if self.torch.equal(old,new):
                    raise ControlError("NO_EFFECTIVE_BF16_CHANGE")
                if tensor not in self.loaded_names and len(self.loaded_names)>=4:
                    raise ControlError("CANDIDATE_TENSOR_LIMIT")
        if name == "qwen_forward" and (len(set(args["query_ids"])) != len(args["query_ids"]) or any(key not in self.queries for key in args["query_ids"])):
            raise ControlError("PUBLIC_QUERY_ID_REQUIRED")

    def _publish_candidate(self, candidate_id, parent, store):
        snapshot = store.snapshot()
        expected = copy.deepcopy(self.base)
        for name,value in snapshot.validate_submission().items():
            expected[name] = tensor_entry(value)
        folder = self.path("candidates",candidate_id)
        candidate = {"candidate_id":candidate_id,"parent_id":parent,"artifact_sha256":digest(expected),
            "files":{p.name:file_sha256(p) for p in sorted(folder.iterdir())},"edited_names":snapshot.edited()}
        self.activate(candidate)
        return candidate

    def _forward(self, ids):
        outputs = []
        for key in ids:
            row = self.queries[key]
            inputs = self.backend.tokenizer.apply_chat_template(row["messages"],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
            if inputs["input_ids"].shape[1]>2048:
                raise ControlError("INPUT_TOKEN_LIMIT_EXCEEDED")
            with self.torch.inference_mode():
                logits = self.backend.model(**{k:v.to("cuda:0") for k,v in inputs.items()}).logits[0,-1].float().cpu().contiguous()
            prediction = self.backend.predict(row)
            prediction["next_logits_sha256"] = hashlib.sha256(memoryview(logits.numpy())).hexdigest()
            token_ids = self.backend.tokenizer.convert_tokens_to_ids(["A","B","C","D"])
            prediction["label_logits"] = {label:float(logits[token]) for label,token in zip("ABCD",token_ids)}
            outputs.append(prediction)
        return outputs

    def execute(self, name, args, state):
        active = state["candidates"][state["active_candidate"]]
        if name == "qwen_inventory":
            # Immutable metadata was bound to the verified loaded model at
            # startup; pagination itself does not claim a fresh GPU byte scan.
            return {"candidate_id":self.loaded,"effective_weights_sha256":active["artifact_sha256"],
                "gpu_verified":False,"inventory_bound_at_load":True,"total":len(self.inventory),
                "tensors":self.inventory[args["offset"]:args["offset"]+args["count"]],"query_ids":list(self.queries)},None,None
        current_hash = self.verify_gpu(active["candidate_id"])
        result = {"candidate_id":self.loaded,"effective_weights_sha256":current_hash,"gpu_verified":True}
        candidate = artifact = None
        if name == "qwen_read_slice":
            tensor = self.backend.model.state_dict()[args["tensor"]].detach().flatten()
            result.update(tensor=args["tensor"],offset=args["offset"],effective_dtype="bfloat16",values=tensor[args["offset"]:args["offset"]+args["count"]].float().cpu().tolist())
        elif name in ("qwen_patch","qwen_apply_diff"):
            folder = self.path("candidates",args["candidate_id"])
            folder.mkdir(exist_ok=False)
            store = self.WeightStore(self.model_path,edit_dir=folder,frozen_names=set(self.base)-self.allowed)
            if name == "qwen_patch":
                snapshot = self._store(active)
                for tensor,value in snapshot.validate_submission().items():
                    store.save_edit(tensor,value)
                value = snapshot.load(args["tensor"])
                value.flatten()[args["offset"]:args["offset"]+len(args["values"])] = self.torch.tensor(args["values"],dtype=self.torch.float32)
                store.save_edit(args["tensor"],value)
            else:
                exported = state["qwen"]["exports"][args["artifact_id"]]
                path = self.path("artifacts",args["artifact_id"])/"diff"
                if {p.name for p in path.iterdir()} != set(exported["diff_files"]):
                    raise ControlError("DIFF_ARTIFACT_CHANGED")
                for filename,expected in exported["diff_files"].items():
                    if (path/filename).is_symlink() or file_sha256(path/filename) != expected:
                        raise ControlError("DIFF_ARTIFACT_CHANGED")
                store.apply_edit(path)
            candidate = self._publish_candidate(args["candidate_id"],active["candidate_id"],store)
            result.update(candidate_id=candidate["candidate_id"],effective_weights_sha256=candidate["artifact_sha256"],parent_id=active["candidate_id"])
        elif name == "qwen_activate":
            self.activate(state["candidates"][args["candidate_id"]])
            result.update(candidate_id=self.loaded,effective_weights_sha256=digest(self.expected))
        elif name == "qwen_diff":
            rows = self._store(active).diff()
            result.update(tensors=rows,changed_entries_total=sum(row["changed_entries"] for row in rows))
        elif name == "qwen_forward":
            result["predictions"] = self._forward(args["query_ids"])
            self.verify_gpu(active["candidate_id"])
        elif name == "qwen_export":
            folder = self.path("artifacts",args["artifact_id"])
            folder.mkdir(exist_ok=False)
            store = self._store(active)
            delta = store.export_edit(folder/"diff",base_model=MODEL_ID,revision=REVISION)
            # Verify the actual upstream diff reader reconstructs the same BF16.
            restored = self.WeightStore(self.model_path,frozen_names=set(self.base)-self.allowed)
            restored.apply_edit(folder/"diff")
            expected = copy.deepcopy(self.base)
            for tensor,value in restored.validate_submission().items():
                expected[tensor] = tensor_entry(value)
            if digest(expected) != current_hash:
                raise ControlError("DIFF_RECONSTRUCTION_MISMATCH")
            checkpoint = self.backend.export(folder/"checkpoint",current_hash,unmodified=not self.loaded_names)
            artifact = {"candidate_id":self.loaded,"effective_weights_sha256":current_hash,"checkpoint_files_sha256":checkpoint,
                "diff_files":{p.name:file_sha256(p) for p in (folder/"diff").iterdir()},"diff_sha256":delta["edit_sha256"]}
            write_json(folder/"receipt.json",artifact)
            result.update(artifact_id=args["artifact_id"],checkpoint_files_sha256=checkpoint,diff_sha256=delta["edit_sha256"],changed_entries_total=delta["changed_entries_total"],diff_reconstruction_verified=True)
        elif name == "qwen_finish":
            result["state"] = "completed"
        return result,candidate,artifact
