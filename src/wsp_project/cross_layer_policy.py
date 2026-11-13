"""Cross-layer policy with instance-scoped reviewed-candidate registry.

Parameterized variant: the reviewed-candidate registry,
pinned digests and experiment identifiers are explicit instance data.
There is **no module-level mutable state and no thread-local binding** —
every function that needs the registry takes it as an explicit argument
(``registry`` parameter), so two engines in the same thread or nested
calls never cross-contaminate.

All admission, progress and canary checks are preserved verbatim; only the
data source of the pinned values moved.
"""
from dataclasses import dataclass
import json

from .errors import ControlError
from .manifest import SHA256, identifier

WRITE_TOOL = "qwen_write_cross_layer"
REQUEST_TIMEOUT_SECONDS = 240
GATEWAY_TIMEOUT_SECONDS = 255
EDITOR_TOOL_TIMEOUT_SECONDS = 270


@dataclass(frozen=True)
class ReviewedCandidate:
    candidate_id: str
    effective_sha256: str
    tensors_sha256: str
    changed_values: int
    refinement_sha256: str | None = None


@dataclass(frozen=True)
class CrossLayerRegistry:
    """One instance's reviewed-candidate set and pinned frozen-input digests."""

    reviewed: dict
    base_sha256: str
    scope_sha256: str
    carrier_id: str
    tensor_count: int
    artifact_id: str
    canary_query_id: str
    canary_expected: str

    def candidate(self, version):
        return self.reviewed.get(version)

    def is_cross_layer(self, policy):
        return isinstance(policy, dict) and isinstance(policy.get("version"), str) and policy["version"] in self.reviewed


def load_registry(path):
    """Load and strictly validate a registry JSON; synthetic examples only.

    Unknown keys, wrong types or malformed digests raise
    ControlError("REGISTRY_INVALID") — never a partial load.
    """

    try:
        with open(path, "r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, ValueError):
        raise ControlError("REGISTRY_INVALID") from None
    allowed = {"schema_version", "base_sha256", "scope_sha256", "carrier_id",
               "tensor_count", "artifact_id", "canary_query_id",
               "canary_expected", "reviewed"}
    if (not isinstance(document, dict) or set(document) != allowed
            or document["schema_version"] != 1):
        raise ControlError("REGISTRY_INVALID")
    if (not isinstance(document["base_sha256"], str) or not SHA256.fullmatch(document["base_sha256"])
            or not isinstance(document["scope_sha256"], str) or not SHA256.fullmatch(document["scope_sha256"])
            or not isinstance(document["carrier_id"], str) or not document["carrier_id"].startswith("v-")
            or not isinstance(document["tensor_count"], int) or not 0 < document["tensor_count"] <= 4096
            or not isinstance(document["artifact_id"], str) or not document["artifact_id"]
            or not isinstance(document["canary_query_id"], str) or not document["canary_query_id"]
            or not isinstance(document["canary_expected"], str) or not document["canary_expected"]):
        raise ControlError("REGISTRY_INVALID")
    reviewed = {}
    raw_reviewed = document["reviewed"]
    if not isinstance(raw_reviewed, dict) or not raw_reviewed:
        raise ControlError("REGISTRY_INVALID")
    for version, entry in raw_reviewed.items():
        if not isinstance(version, str) or not version or not isinstance(entry, dict):
            raise ControlError("REGISTRY_INVALID")
        keys = {"candidate_id", "effective_sha256", "tensors_sha256",
                "changed_values", "refinement_sha256"}
        if set(entry) != keys:
            raise ControlError("REGISTRY_INVALID")
        refinement = entry["refinement_sha256"]
        if (not isinstance(entry["candidate_id"], str) or not entry["candidate_id"]
                or not isinstance(entry["effective_sha256"], str) or not SHA256.fullmatch(entry["effective_sha256"])
                or not isinstance(entry["tensors_sha256"], str) or not SHA256.fullmatch(entry["tensors_sha256"])
                or not isinstance(entry["changed_values"], int) or entry["changed_values"] < 0
                or (refinement is not None and (not isinstance(refinement, str) or not SHA256.fullmatch(refinement)))):
            raise ControlError("REGISTRY_INVALID")
        reviewed[version] = ReviewedCandidate(
            entry["candidate_id"], entry["effective_sha256"],
            entry["tensors_sha256"], entry["changed_values"], refinement,
        )
    return CrossLayerRegistry(
        reviewed=reviewed,
        base_sha256=document["base_sha256"],
        scope_sha256=document["scope_sha256"],
        carrier_id=document["carrier_id"],
        tensor_count=document["tensor_count"],
        artifact_id=document["artifact_id"],
        canary_query_id=document["canary_query_id"],
        canary_expected=document["canary_expected"],
    )


def is_cross_layer(policy, registry=None):
    """Check whether ``policy`` targets this registry's reviewed versions.

    ``registry=None`` means "caller has no cross-layer registry" — returns
    False (a plain non-cross policy), **never raises**: knowledge_policy's
    dispatch must safely fall through to the generic branch when no
    registry is configured.
    """

    if registry is None:
        return False
    return registry.is_cross_layer(policy)


def reviewed_candidate(config, registry):
    spec = registry.candidate(config.get("version") if isinstance(config, dict) else None)
    if spec is None:
        raise ControlError("INVALID_WRITE_POLICY")
    return spec


def validate(policy, run_id, registry):
    if not registry.is_cross_layer(policy):
        raise ControlError("INVALID_WRITE_POLICY")
    spec = registry.candidate(policy["version"])
    required = {"version", "authorized", "run_id", "base_sha256", "questions_sha256",
                "candidate_sha256", "tensors_sha256", "scope_sha256", "canary_query_id",
                "canary_expected"}
    if (set(policy) != required or policy["authorized"] is not True
            or (run_id is not None and policy["run_id"] != run_id)
            or policy["base_sha256"] != registry.base_sha256 or policy["candidate_sha256"] != spec.effective_sha256
            or policy["tensors_sha256"] != spec.tensors_sha256 or policy["scope_sha256"] != registry.scope_sha256
            or policy["canary_query_id"] != registry.canary_query_id or policy["canary_expected"] != registry.canary_expected
            or not isinstance(policy["questions_sha256"], str) or not SHA256.fullmatch(policy["questions_sha256"])):
        raise ControlError("INVALID_WRITE_POLICY")
    identifier(policy["run_id"])


def validate_progress(name, args, state, registry):
    active = state["active_candidate"]
    spec = reviewed_candidate(state["qwen"]["config"]["write_policy"], registry)
    if name == WRITE_TOOL and (active != "base" or args["candidate_id"] != spec.candidate_id):
        raise ControlError("WRITE_POLICY_STEP_MISMATCH")
    if name == "qwen_export" and (active != spec.candidate_id or args["artifact_id"] != registry.artifact_id):
        raise ControlError("WRITE_POLICY_EXPORT_MISMATCH")
    if name == "qwen_finish" and (active != spec.candidate_id or set(state["qwen"]["exports"]) != {registry.artifact_id}):
        raise ControlError("WRITE_POLICY_EXPORT_MISMATCH")
    if name not in (WRITE_TOOL, "qwen_export", "qwen_finish"):
        return
    rows = [
        prediction["text"].strip()
        for request in state["requests"].values()
        if request["status"] == "completed" and request["operation"] == "qwen_forward"
        and request["result"]["candidate_id"] == active
        for prediction in request["result"]["predictions"]
        if prediction["id"] == registry.canary_query_id
    ]
    if not rows:
        raise ControlError("WRITE_POLICY_DEVELOPMENT_REQUIRED")
    if any(text != registry.canary_expected for text in rows):
        raise ControlError("WRITE_POLICY_CANARY_FAILED")
    if name in ("qwen_export", "qwen_finish") and state["candidates"][active]["artifact_sha256"] != spec.effective_sha256:
        raise ControlError("WRITE_POLICY_CONTROL_HASH_MISMATCH")
