"""Phase-aware admission checks for the project's Codex subscription protocol.

Verified flags are assertions from a trusted controller. This module does not
inspect model files, prove OS isolation, authorize spending, or run an editor.
"""

import math
import re
from dataclasses import dataclass

from .manifest import IDENTIFIER, SHA256

PHASES = ("preflight", "baseline", "tool_smoke", "edit", "evaluate")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
TENSOR = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+\Z")

# Every listed key is required. Nullable fields support a draft, not permission.
SCHEMA = {
    "schema_version": "version", "status": "text", "formal_run_authorized": "bool",
    "run_id": "id?", "route": "route", "controller_code_sha256": "hash?",
    "editor": {
        "model": "model", "reasoning_effort": "effort", "mode": "mode",
        "authentication_route": "auth", "runtime_platform": "platform",
        "observed_cli_version": "text", "chatgpt_login_status_observed": "bool",
        "model_catalog_visibility_observed": "bool",
        "live_model_and_tool_roundtrip_verified": "bool",
        "native_tool_isolation_verified": "bool", "api_fallback_allowed": "bool",
    },
    "upstream": {
        "repository": "text", "candidate_commit": "revision?",
        "installed_commit": "revision?", "dependency_lock_sha256": "hash?",
        "reuse": "text", "use_reference_exp8_driver": "bool",
        "use_pinned_codex_editor_adapter": "bool",
    },
    "remote_backend": {
        "connection_transport_candidate": "text", "ssh_config_alias": "text?",
        "ssh_connection_verified": "bool", "operating_system_verified": "bool",
        "observed_operating_system": "text", "observed_system_python": "text",
        "gpu_reported_by_user": "text", "gpu_identity_verified": "bool",
        "gpu_runtime_verified": "bool", "project_root": "text?",
        "base_model_path": "text?", "python_executable": "text?",
        "cuda_torch_verified": "bool", "forward_factory": "text?",
        "export_codex_credentials": "bool",
    },
    "target": {
        "model_id": "target", "candidate_revision": "revision?",
        "loaded_revision": "revision?", "dtype_candidate": "dtype",
        "effective_dtype": "dtype?", "config_sha256": "hash?",
        "tokenizer_sha256": "hash?", "base_weights_sha256_manifest": "hash?",
        "inference_code_sha256": "hash?", "standard_reload_required": "bool",
        "trust_remote_code": "bool", "quantization_allowed": "bool",
    },
    "task": {
        "level": "level", "n_facts": "scale", "labels": "labels",
        "independent_base_for_each_scale": "bool",
        "hidden_templates_rewritten_independently": "bool",
        "normal_retention_items_planned": "positive",
        "public_data_manifest_sha256": "hash?", "private_data_manifest_sha256": "hash?",
        "normal_retention_manifest_sha256": "hash?",
        "update_release_requires_v1_sealed": "bool",
    },
    "generation": {
        "do_sample": "bool", "max_new_tokens": "positive",
        "batch_limit_proposed": "positive", "chat_template_sha256": "hash?",
    },
    "edit_policy": {
        "allow_target_gradients": "bool", "allow_auxiliary_fitting": "bool",
        "allow_blackbox_numerical_optimization": "bool",
        "allow_closed_form_constraints_with_disclosure": "bool",
        "allow_runtime_memory": "bool", "allow_final_runtime_hooks": "bool",
        "freeze_architecture": "bool", "freeze_tokenizer": "bool",
        "protected_module_categories": "protected", "writable_tensor_allowlist": "tensors",
        "allowlist_must_match_loaded_target": "bool",
        "changed_parameter_fraction_alert": "fraction", "arbitrary_run_code_exposed": "bool",
    },
    "budget": {
        "operator_authorized": "bool", "subscription_editor_request_limit": "count?",
        "dev_forward_request_limit": "count?", "dev_forward_item_limit": "count?",
        "baseline_forward_request_limit": "count?", "baseline_forward_item_limit": "count?",
        "evaluation_forward_request_limit": "count?", "evaluation_forward_item_limit": "count?",
        "generated_token_limit": "count?", "activation_probe_limit": "count?",
        "activation_bytes_limit": "count?", "per_request_timeout_seconds": "positive?",
        "total_wallclock_seconds": "positive?", "recovery_request_reserve": "count?",
        "api_spending_authorized": "bool", "automatic_billing_fallback": "bool",
        "reset_on_resume": "bool",
    },
    "baselines": {
        "b0_verified": "bool", "b1_verified": "bool",
        "b2_fresh_process_reload_verified": "bool", "weight_sync_and_rollback_verified": "bool",
        "evidence_manifest_sha256": "hash?", "b1_strict_accuracy": "fraction?",
    },
    "isolation": {
        "private_data_unreadable_by_editor_verified": "bool",
        "sealed_updates_unreadable_by_editor_verified": "bool",
        "executor_credentials_absent_verified": "bool",
        "executor_network_restrictions_verified": "bool",
        "fresh_evaluator_identity_verified": "bool", "evidence_manifest_sha256": "hash?",
    },
    "tool_smoke": {"fixture_sha256": "hash?"},
    "acceptance_proposed_not_observed": {
        "b1_min_strict_accuracy": "fraction", "eight_fact_min_strict_accuracy": "fraction",
        "per_entity_min_strict_accuracy": "fraction",
        "normal_task_max_drop_percentage_points": "percent",
        "updated_entities_min_strict_accuracy": "fraction",
        "retained_entities_min_strict_accuracy": "fraction",
    },
    "notes": "texts",
}

ENUMS = {
    "route": {"codex_mcp_remote_qwen"}, "model": {"gpt-6-astra"},
    "effort": {"low", "medium", "high", "xhigh", "max"},
    "mode": {"autonomous", "method_guided", "patch_replay"},
    "auth": {"chatgpt_subscription"}, "platform": {"windows_native"},
    "target": {"Qwen/Qwen3-4B-Instruct-2507"}, "dtype": {"bfloat16"}, "level": {"T2"},
}


@dataclass(frozen=True)
class Issue:
    field: str
    code: str


def _valid(value, rule):
    if rule.endswith("?"):
        if value is None:
            return True
        rule = rule[:-1]
    if rule in ENUMS:
        return isinstance(value, str) and value in ENUMS[rule]
    if rule == "bool":
        return type(value) is bool
    if rule in ("count", "positive", "version", "scale"):
        if type(value) is not int or value > 2**63 - 1:
            return False
        return {"count": value >= 0, "positive": value > 0,
                "version": value == 1, "scale": value in (1, 4, 8)}[rule]
    if rule in ("fraction", "percent"):
        return type(value) in (int, float) and 0 <= value <= (1 if rule == "fraction" else 100) and math.isfinite(value)
    if rule == "text":
        return isinstance(value, str) and bool(value.strip()) and len(value) <= 4096 and "\x00" not in value
    if rule in ("id", "hash", "revision"):
        regex = {"id": IDENTIFIER, "hash": SHA256, "revision": REVISION}[rule]
        return isinstance(value, str) and bool(regex.fullmatch(value))
    if rule == "labels":
        return value == ["A", "B", "C", "D"]
    if rule == "protected":
        return isinstance(value, list) and all(isinstance(x, str) for x in value) and set(value) == {"embedding", "lm_head", "final_norm"} and len(value) == 3
    if rule in ("texts", "tensors"):
        return isinstance(value, list) and all(_valid(x, "text") for x in value) and (rule == "texts" or (len(set(value)) == len(value) and all(TENSOR.fullmatch(x) for x in value)))
    return False


def validate(protocol, phase="preflight", *, tensor_names=None, submission_sha256=None):
    """Return stable field/code issues; never echo unknown keys or input values."""
    issues = []
    if phase not in PHASES:
        return [Issue("phase", "unsupported_phase")]

    def check(obj, schema, prefix=""):
        if not isinstance(obj, dict):
            issues.append(Issue(prefix or "protocol", "expected_object"))
            return
        if set(obj) - set(schema):
            issues.append(Issue(prefix or "protocol", "unknown_fields"))
        for key, rule in schema.items():
            field = f"{prefix}.{key}" if prefix else key
            if key not in obj:
                issues.append(Issue(field, "missing"))
            elif isinstance(rule, dict):
                check(obj[key], rule, field)
            elif not _valid(obj[key], rule):
                issues.append(Issue(field, "invalid_value"))

    check(protocol, SCHEMA)
    if issues:
        return issues

    def get(path):
        value = protocol
        for part in path.split("."):
            value = value[part]
        return value

    def require(path, expected=True):
        if get(path) is not expected:
            issues.append(Issue(path, "policy_required"))

    def filled(path):
        if get(path) is None:
            issues.append(Issue(path, "required_for_phase"))

    for path in (
        "editor.api_fallback_allowed", "remote_backend.export_codex_credentials",
        "target.trust_remote_code", "target.quantization_allowed", "generation.do_sample",
        "edit_policy.allow_target_gradients", "edit_policy.allow_auxiliary_fitting",
        "edit_policy.allow_blackbox_numerical_optimization", "edit_policy.allow_runtime_memory",
        "edit_policy.allow_final_runtime_hooks", "edit_policy.arbitrary_run_code_exposed",
        "budget.api_spending_authorized", "budget.automatic_billing_fallback", "budget.reset_on_resume",
        "upstream.use_reference_exp8_driver", "upstream.use_pinned_codex_editor_adapter",
    ):
        require(path, False)
    for path in ("target.standard_reload_required", "task.independent_base_for_each_scale",
                 "task.update_release_requires_v1_sealed", "edit_policy.freeze_architecture",
                 "edit_policy.freeze_tokenizer", "edit_policy.allowlist_must_match_loaded_target"):
        require(path)
    budget = protocol["budget"]
    reserve, limit = budget["recovery_request_reserve"], budget["subscription_editor_request_limit"]
    if reserve is not None and limit is not None and reserve > limit:
        issues.append(Issue("budget.recovery_request_reserve", "exceeds_request_limit"))
    for name in protocol["edit_policy"]["writable_tensor_allowlist"]:
        if name.endswith(("embed_tokens.weight", "lm_head.weight")) or name in ("model.norm.weight", "model.language_model.norm.weight"):
            issues.append(Issue("edit_policy.writable_tensor_allowlist", "protected_tensor"))
            break
    if phase == "preflight":
        return issues
    for path in ("run_id", "controller_code_sha256", "upstream.dependency_lock_sha256"):
        filled(path)
    # Standard Transformers baselines do not use OpenWSP. Requiring an installed
    # editor toolchain here would introduce an unnecessary bootstrap dependency.
    if phase != "baseline" or protocol["upstream"]["installed_commit"] is not None:
        filled("upstream.installed_commit")
        if protocol["upstream"]["installed_commit"] != protocol["upstream"]["candidate_commit"]:
            issues.append(Issue("upstream.installed_commit", "revision_mismatch"))
    require("budget.operator_authorized")
    for key, rule in SCHEMA["budget"].items():
        if rule.endswith("?"):
            filled("budget." + key)
    if phase == "tool_smoke":
        filled("tool_smoke.fixture_sha256")
    else:
        for path in ("remote_backend.ssh_connection_verified", "remote_backend.operating_system_verified",
                     "remote_backend.gpu_identity_verified", "remote_backend.gpu_runtime_verified",
                     "remote_backend.cuda_torch_verified"):
            require(path)
        for key in ("loaded_revision", "effective_dtype", "config_sha256", "tokenizer_sha256",
                    "base_weights_sha256_manifest", "inference_code_sha256"):
            filled("target." + key)
        for path in ("task.public_data_manifest_sha256", "generation.chat_template_sha256"):
            filled(path)
        if protocol["target"]["loaded_revision"] != protocol["target"]["candidate_revision"]:
            issues.append(Issue("target.loaded_revision", "revision_mismatch"))
    if phase in ("tool_smoke", "edit", "evaluate"):
        for path in ("editor.chatgpt_login_status_observed", "editor.model_catalog_visibility_observed",
                     "editor.native_tool_isolation_verified", "isolation.executor_credentials_absent_verified",
                     "isolation.executor_network_restrictions_verified"):
            require(path)
        filled("isolation.evidence_manifest_sha256")
    if phase in ("edit", "evaluate"):
        require("formal_run_authorized")
        require("editor.live_model_and_tool_roundtrip_verified")
        require("task.hidden_templates_rewritten_independently")
        for key in SCHEMA["isolation"]:
            if key.endswith("verified"):
                require("isolation." + key)
        for key in SCHEMA["baselines"]:
            if key.endswith("verified"):
                require("baselines." + key)
        for path in ("baselines.evidence_manifest_sha256", "baselines.b1_strict_accuracy",
                     "task.private_data_manifest_sha256", "task.normal_retention_manifest_sha256"):
            filled(path)
        accuracy = protocol["baselines"]["b1_strict_accuracy"]
        if accuracy is not None and accuracy < protocol["acceptance_proposed_not_observed"]["b1_min_strict_accuracy"]:
            issues.append(Issue("baselines.b1_strict_accuracy", "below_threshold"))
        writable = protocol["edit_policy"]["writable_tensor_allowlist"]
        if not writable:
            issues.append(Issue("edit_policy.writable_tensor_allowlist", "empty_allowlist"))
        if tensor_names is None:
            issues.append(Issue("edit_policy.writable_tensor_allowlist", "inventory_required"))
        elif not isinstance(tensor_names, (list, tuple, set, frozenset)) or not all(isinstance(x, str) for x in tensor_names):
            issues.append(Issue("edit_policy.writable_tensor_allowlist", "invalid_inventory"))
        elif not set(writable).issubset(tensor_names):
            issues.append(Issue("edit_policy.writable_tensor_allowlist", "unknown_tensor"))
    if phase == "evaluate" and not _valid(submission_sha256, "hash"):
        issues.append(Issue("submission", "sealed_submission_required"))
    return issues
