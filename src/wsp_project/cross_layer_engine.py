"""Physical write path for the reviewed cross-layer candidate.

Parameterized variant: the registry is an explicit
instance attribute (``self.registry``) passed in at construction and held
per-engine; all policy calls pass it explicitly — no shared module state,
no thread-local binding, safe for two engines in one thread.
"""
import shutil
import tempfile
from pathlib import Path

from . import cross_layer_policy as policy
from .baselines import file_sha256
from .cross_layer_recipe import construct
from .errors import ControlError
from .manifest import digest, read_json
from .qwen_engine import QwenEngine, tensor_entry

REFINED_TENSOR = "model.layers.35.mlp.up_proj.weight"
REFINED_ROWS = [3490, 4193, 9519]


class CrossLayerEngine(QwenEngine):
    def __init__(self, model, work, questions, *, write_policy, frozen, registry):
        if not isinstance(registry, policy.CrossLayerRegistry):
            raise ControlError("REGISTRY_INVALID")
        self.registry = registry
        policy.validate(write_policy, None, self.registry)
        self.write_policy = write_policy
        self.spec = policy.reviewed_candidate(write_policy, self.registry)
        tensors = read_json(frozen / "tensors.json")
        self.scope = read_json(frozen / "scope.json", max_bytes=4 * 1024 * 1024)
        carrier = read_json(frozen / "carrier.json")
        if (digest(tensors) != self.spec.tensors_sha256 or digest(self.scope) != self.registry.scope_sha256
                or "v-" + digest(carrier) != self.registry.carrier_id):
            raise ControlError("FROZEN_INPUT_CHANGED")
        self.tensor_rows = {row["tensor"]: row for row in tensors}
        self.carrier_values = carrier["values"]
        self.refinement = None
        if self.spec.refinement_sha256 is not None:
            self.refinement = read_json(frozen / "refinement.json")
            if (not isinstance(self.refinement, list) or len(self.refinement) != 2560
                    or digest(self.refinement) != self.spec.refinement_sha256):
                raise ControlError("FROZEN_REFINEMENT_CHANGED")
        self.active = None
        super().__init__(model, work, questions)
        if (self.base_sha256 != self.registry.base_sha256 or self.allowed != self.tensor_rows.keys()
                or digest(list(self.queries.values())) != write_policy["questions_sha256"]):
            raise ControlError("FROZEN_BASE_MISMATCH")

    @property
    def candidate_tensor_limit(self):
        return self.registry.tensor_count

    def is_writable(self, name):
        return name in self.tensor_rows

    def activate(self, candidate):
        previous = self.active
        if candidate["candidate_id"] == self.spec.candidate_id and (
            candidate["artifact_sha256"] != self.spec.effective_sha256
            or set(candidate["edited_names"]) != self.allowed
        ):
            raise ControlError("FROZEN_CANDIDATE_MISMATCH")
        try:
            super().activate(candidate)
        except Exception:
            if previous is not None:
                # A copy or sync can fail before QwenEngine updates loaded_names.
                # Rewrite every allowed matrix and verify the entire prior model.
                self.loaded_names = self.allowed.copy()
                try:
                    super().activate(previous)
                except Exception:
                    raise ControlError("FROZEN_GPU_ROLLBACK_UNVERIFIED") from None
            raise
        self.active = candidate

    def verify_updates(self, updates):
        if set(updates) != self.allowed:
            raise ControlError("FROZEN_TENSOR_SET_MISMATCH")
        expected = self.base.copy()
        parameters = self.backend.model.state_dict()
        for name, weight in updates.items():
            row = self.tensor_rows[name]
            original = parameters[name]
            if (weight.dtype != self.torch.bfloat16 or list(weight.shape) != row["shape"]
                    or self.base[name]["sha256"] != row["base_sha256"]
                    or not bool(self.torch.isfinite(weight).all())):
                raise ControlError("FROZEN_TENSOR_INVALID")
            entry = tensor_entry(weight)
            if entry["sha256"] != row["candidate_sha256"]:
                raise ControlError("FROZEN_TENSOR_DIGEST_MISMATCH")
            difference = weight != original
            rows = self.torch.where(difference.any(dim=1))[0]
            columns = self.torch.where(difference.any(dim=0))[0]
            if (int(difference.sum()) != row["changed_values"]
                    or len(rows) != row["changed_rows"] or len(columns) != row["changed_columns"]
                    or (row["row_indices"] is not None and rows.tolist() != row["row_indices"])
                    or (row["column_indices"] is not None and columns.tolist() != row["column_indices"])
                    or float((weight.float() - original.float()).abs().max()) != row["bf16_max_delta"]):
                raise ControlError("FROZEN_TENSOR_DELTA_MISMATCH")
            expected[name] = entry
        if digest(expected) != self.spec.effective_sha256:
            raise ControlError("FROZEN_CANDIDATE_MISMATCH")

    def persist(self, updates):
        folder = self.path("candidates", self.spec.candidate_id)
        if folder.exists():
            raise ControlError("FROZEN_CANDIDATE_PATH_EXISTS")
        staging = Path(tempfile.mkdtemp(prefix=".stage-", dir=folder.parent))
        try:
            store = self.WeightStore(self.model_path, edit_dir=staging, frozen_names=set(self.base)-self.allowed)
            for name, value in sorted(updates.items()):
                store.save_edit(name, value)
            if set(store.edited()) != self.allowed:
                raise ControlError("FROZEN_TENSOR_SET_MISMATCH")
            files = {p.name: file_sha256(p) for p in sorted(staging.iterdir())}
            staging.rename(folder)
            return {"candidate_id": self.spec.candidate_id, "parent_id": "base",
                    "artifact_sha256": self.spec.effective_sha256,
                    "files": files, "edited_names": sorted(self.allowed)}
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def reconstruct(self):
        updates, forwards = construct(self.backend, self.scope, self.carrier_values)
        if self.refinement is not None:
            refined = updates[REFINED_TENSOR]
            row = self.torch.tensor(self.refinement, dtype=self.torch.bfloat16, device=refined.device)
            refined[REFINED_ROWS] = row
        return updates, forwards

    def execute(self, name, args, state):
        if name not in (policy.WRITE_TOOL, "qwen_diff"):
            return super().execute(name, args, state)
        active = state["active_candidate"]
        current = self.verify_gpu(active)
        if name == "qwen_diff":
            changed = active == self.spec.candidate_id
            return {"candidate_id": active, "effective_weights_sha256": current, "gpu_verified": True,
                    "changed_entries_total": self.spec.changed_values if changed else 0,
                    "edited_names": sorted(self.allowed) if changed else [],
                    "tensor_manifest_sha256": self.spec.tensors_sha256 if changed else None}, None, None
        if active != "base" or args["candidate_id"] != self.spec.candidate_id:
            raise ControlError("WRITE_POLICY_STEP_MISMATCH")
        updates, forwards = self.reconstruct()
        self.verify_gpu("base")
        self.verify_updates(updates)
        candidate = self.persist(updates)
        del updates
        try:
            self.activate(candidate)
        except Exception:
            shutil.rmtree(self.path("candidates", self.spec.candidate_id))
            raise
        return {"candidate_id": self.spec.candidate_id, "parent_id": "base", "gpu_verified": True,
                "effective_weights_sha256": self.spec.effective_sha256, "construction_forwards": forwards,
                "tensor_manifest_sha256": self.spec.tensors_sha256,
                "changed_entries_total": self.spec.changed_values}, candidate, None
