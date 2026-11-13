"""Frozen Qwen BF16/CUDA inference and standard checkpoint export.

Heavy dependencies import only in a target runtime; no remote model code,
training, quantization, runtime facts, or answer-file access in this backend.
"""
import os
import time
from pathlib import Path

from .baselines import file_sha256, write_json
from .errors import ControlError
from .manifest import digest


def freeze_generation_config(model, generation):
    # Transformers >=4.50 can replace explicit global-default values (including
    # do_sample=False) with checkpoint defaults. Validate the resolved contract.
    resolved, _ = model._prepare_generation_config(generation, use_model_defaults=False, do_sample=False)
    if resolved.get_generation_mode().value != "greedy_search" or resolved.max_new_tokens != generation.max_new_tokens:
        raise ControlError("GENERATION_CONTRACT_MISMATCH")
    return resolved


def generate_frozen(model, inputs, generation):
    return model.generate(**inputs, generation_config=generation, use_model_defaults=False, do_sample=False)


class QwenBackend:
    def __init__(self, model_dir):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        self.torch = torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ControlError("CUDA_BF16_REQUIRED")
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
            device_map={"":"cuda:0"}, trust_remote_code=False, local_files_only=True, attn_implementation="eager")
        self.model.eval()
        self.model.requires_grad_(False)
        if any(p.device.type != "cuda" or p.dtype != torch.bfloat16 for p in self.model.parameters()):
            raise ControlError("UNEXPECTED_PARAMETER_DEVICE_OR_DTYPE")
        if self.model.config.model_type != "qwen3":
            raise ControlError("UNEXPECTED_MODEL_ARCHITECTURE")
        base = self.model.generation_config
        self.generation = GenerationConfig(do_sample=False, max_new_tokens=16, num_beams=1, use_cache=True,
            bos_token_id=base.bos_token_id, eos_token_id=base.eos_token_id,
            pad_token_id=base.pad_token_id if base.pad_token_id is not None else self.tokenizer.eos_token_id)
        self.generation = freeze_generation_config(self.model, self.generation)
        config = self.model.config.to_dict()
        for key in ("_name_or_path", "_commit_hash", "transformers_version"):
            config.pop(key, None)
        self.meta = {"torch":torch.__version__,"transformers":transformers.__version__,"cuda_runtime":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(0),"dtype":"bfloat16","attention":"eager","batch_size":1,
            "bf16_reduced_precision_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"tf32":False,
            "device_placement":"direct_cuda","use_model_defaults":False,
            "effective_generation_mode":self.generation.get_generation_mode().value,
            "config_semantic_sha256":digest(config),
            "tokenizer_semantic_sha256":digest({"backend":self.tokenizer.backend_tokenizer.to_str(),
                "vocabulary":self.tokenizer.get_vocab(),"special_tokens":self.tokenizer.special_tokens_map,"chat_template":self.tokenizer.chat_template}),
            "chat_template_sha256":digest(self.tokenizer.chat_template),"generation_sha256":digest(self.generation.to_dict())}

    def tensor_manifest(self):
        rows = {}
        import hashlib
        for name, tensor in sorted(self.model.state_dict().items()):
            cpu = tensor.detach().cpu().contiguous()
            value = hashlib.sha256(memoryview(cpu.view(self.torch.uint8).numpy())).hexdigest()
            rows[name] = {"shape":list(tensor.shape),"dtype":str(tensor.dtype),"sha256":value}
        return rows

    def predict(self, row):
        inputs = self.tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        if inputs["input_ids"].shape[1] > 2048:
            raise ControlError("INPUT_TOKEN_LIMIT_EXCEEDED")
        prompt_ids = inputs["input_ids"][0].tolist()
        inputs = {key:value.to("cuda:0") for key,value in inputs.items()}
        start = time.monotonic()
        with self.torch.inference_mode():
            generated = generate_frozen(self.model, inputs, self.generation)
        self.torch.cuda.synchronize()
        ids = generated[0, len(prompt_ids):].tolist()
        return {"id":row["id"],"text":self.tokenizer.decode(ids, skip_special_tokens=True),
                "input_token_ids":prompt_ids,"generated_token_ids":ids,"elapsed_seconds":round(time.monotonic()-start,6)}

    def export(self, directory, expected_tensor_manifest_sha256, *, unmodified=True):
        if digest(self.tensor_manifest()) != expected_tensor_manifest_sha256:
            raise ControlError("WEIGHTS_CHANGED_BEFORE_EXPORT")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(directory, safe_serialization=True, max_shard_size="2GB")
        self.tokenizer.save_pretrained(directory)
        self.generation.save_pretrained(directory)
        files = {path.name:file_sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}
        write_json(directory / "export-manifest.json", {"files":files,"unmodified":unmodified,
            "tensor_manifest_sha256":expected_tensor_manifest_sha256,"runtime":self.meta})
        return digest(files)

    def memory(self):
        return {"peak_allocated_bytes":self.torch.cuda.max_memory_allocated(),"peak_reserved_bytes":self.torch.cuda.max_memory_reserved()}
