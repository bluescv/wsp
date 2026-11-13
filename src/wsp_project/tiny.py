"""Bounded, real 2x2 ReLU network for transport/weight-tool testing only.

No torch, training, OpenWSP dependency, language model, or sandbox claim.
Tensor values and atomic operation receipts share RunStore's SQLite snapshot.
"""

import copy
import math
import os
import struct
import time

from .budget import RESOURCES
from .errors import ControlError
from .manifest import canonical, digest, identifier, read_json
from .state import RunStore

FORMAT = "wsp-tiny-relu-f32-v1"
BASE = {"hidden.weight": [1.0, 0.0, 0.0, 1.0], "hidden.bias": [0.0, 0.0],
        "readout.weight": [1.0, 1.0, 1.0, -1.0]}
PROBES = {"probe_a": [1.0, 2.0], "probe_b": [-1.0, 3.0], "probe_c": [0.5, -0.5]}
BASE_HASH = digest(BASE)
ID = {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$", "maxLength": 96}


def f32(value):
    if type(value) not in (int, float) or not -1_000_000 <= value <= 1_000_000:
        raise ControlError("INVALID_TENSOR_VALUE")
    return struct.unpack("<f", struct.pack("<f", value))[0]


def weights_checked(weights):
    if not isinstance(weights, dict) or set(weights) != set(BASE):
        raise ControlError("INVALID_TINY_CHECKPOINT")
    result = {}
    for name, base in BASE.items():
        values = weights[name]
        if not isinstance(values, list) or len(values) != len(base):
            raise ControlError("INVALID_TINY_CHECKPOINT")
        result[name] = [f32(value) for value in values]
        if result[name] != values:
            raise ControlError("CHECKPOINT_DTYPE_MISMATCH")
        if name != "hidden.weight" and values != base:
            raise ControlError("PROTECTED_TENSOR")
    return result


def forward(weights, probe_ids):
    """Float32 products and accumulations, with fixed architecture and inputs."""
    def dot(row, vector):
        value = 0.0
        for a, b in zip(row, vector):
            # Network results can exceed parameter bounds; still require finite f32.
            product = struct.unpack("<f", struct.pack("<f", a * b))[0]
            value = struct.unpack("<f", struct.pack("<f", value + product))[0]
        if not math.isfinite(value):
            raise ControlError("NONFINITE_FORWARD")
        return value
    output = []
    for probe in probe_ids:
        x = PROBES[probe]
        hidden = [max(0.0, dot(weights["hidden.weight"][i:i+2], x) + weights["hidden.bias"][i // 2])
                  for i in (0, 2)]
        output.append([dot(weights["readout.weight"][i:i+2], hidden) for i in (0, 2)])
    return output


def checkpoint(weights, candidate_id):
    body = {"format": FORMAT, "base_sha256": BASE_HASH, "candidate_id": candidate_id,
            "weights_sha256": digest(weights), "weights": weights_checked(weights)}
    return {"body": body, "sha256": digest(body)}


def load_checkpoint(path, expected_sha256=None):
    bundle = read_json(path)
    if not isinstance(bundle, dict) or set(bundle) != {"body", "sha256"}:
        raise ControlError("INVALID_TINY_CHECKPOINT")
    body = bundle["body"]
    if not isinstance(body, dict) or set(body) != {"format", "base_sha256", "candidate_id", "weights_sha256", "weights"}:
        raise ControlError("INVALID_TINY_CHECKPOINT")
    if digest(body) != bundle["sha256"] or (expected_sha256 is not None and bundle["sha256"] != expected_sha256):
        raise ControlError("CHECKPOINT_INTEGRITY_FAILED")
    if body["format"] != FORMAT or body["base_sha256"] != BASE_HASH:
        raise ControlError("CHECKPOINT_BASE_MISMATCH")
    identifier(body["candidate_id"])
    weights = weights_checked(body["weights"])
    if digest(weights) != body["weights_sha256"]:
        raise ControlError("CHECKPOINT_INTEGRITY_FAILED")
    return bundle


def schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


COMMON = {"request_id": ID, "expected_candidate": ID}
EXTRA = {
    "tiny_list_tensors": {},
    "tiny_read_tensor": {"tensor": {"type": "string", "enum": list(BASE)}},
    "tiny_patch": {"candidate_id": ID, "tensor": {"type": "string", "enum": ["hidden.weight"]},
                   "offset": {"type": "integer", "minimum": 0, "maximum": 3},
                   "values": {"type": "array", "minItems": 1, "maxItems": 4,
                              "items": {"type": "number", "minimum": -1_000_000, "maximum": 1_000_000}}},
    "tiny_forward": {"probe_ids": {"type": "array", "minItems": 1, "maxItems": 4,
                                    "items": {"type": "string", "enum": list(PROBES)}}},
    "tiny_diff": {"other_candidate": ID},
    "tiny_activate": {"candidate_id": ID},
    "tiny_export": {"artifact_id": ID},
    "tiny_reload": {"artifact_id": ID, "candidate_id": ID},
    "tiny_finish": {},
}
TOOLS = [{"name": name, "description": "Tiny fixture only: " + name.removeprefix("tiny_").replace("_", " "),
          "inputSchema": schema({**COMMON, **extra})} for name, extra in EXTRA.items()]
TOOLS += [{"name": "tiny_status", "description": "Inspect durable tiny fixture status; no model execution.", "inputSchema": schema({})},
          {"name": "tiny_receipt", "description": "Recover an existing operation receipt without repeating it.", "inputSchema": schema({"request_id": ID})}]


def validate_arguments(value, rule):
    """Validate the small JSON Schema subset used above, not arbitrary schemas."""
    kind = rule["type"]
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(rule["properties"]):
            raise ControlError("INVALID_TOOL_ARGUMENTS")
        for name, child in rule["properties"].items():
            validate_arguments(value[name], child)
    elif kind == "string":
        if not isinstance(value, str) or ("enum" in rule and value not in rule["enum"]):
            raise ControlError("INVALID_TOOL_ARGUMENTS")
        if not rule.get("minLength",0) <= len(value) <= rule.get("maxLength",1_000_000):
            raise ControlError("INVALID_TOOL_ARGUMENTS")
        if "pattern" in rule:
            identifier(value)
    elif kind == "boolean":
        if type(value) is not bool:
            raise ControlError("INVALID_TOOL_ARGUMENTS")
    elif kind in ("number", "integer"):
        if type(value) not in ((int,) if kind == "integer" else (int, float)) or not rule["minimum"] <= value <= rule["maximum"]:
            raise ControlError("INVALID_TOOL_ARGUMENTS")
    elif kind == "array":
        if not isinstance(value, list) or not rule["minItems"] <= len(value) <= rule["maxItems"]:
            raise ControlError("INVALID_TOOL_ARGUMENTS")
        for item in value:
            validate_arguments(item, rule["items"])
    else:
        raise ControlError("INVALID_TOOL_SCHEMA")


class TinyStore(RunStore):
    @classmethod
    def create(cls, directory, *, run_id, tool_limit=64, forward_limit=12, item_limit=32, seconds=600, clock=time.time):
        identifier(run_id)
        for value in (tool_limit, forward_limit, item_limit, seconds):
            if type(value) is not int or not 1 <= value <= 100_000:
                raise ControlError("INVALID_TINY_LIMIT")
        if tool_limit > 128:
            raise ControlError("INVALID_TINY_LIMIT")
        now = clock()
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ControlError("INVALID_CLOCK")
        limits = {name: 0 for name in RESOURCES.values()}
        limits.update(dev_forward_request_limit=forward_limit, dev_forward_item_limit=item_limit,
                      recovery_request_reserve=0, total_wallclock_seconds=seconds, per_request_timeout_seconds=30)
        protocol = {"kind": "tiny_toolchain_only", "format": FORMAT, "run_id": run_id,
                    "tool_limit": tool_limit, "limits": limits, "base_sha256": BASE_HASH}
        state = {"schema_version": 1, "run_id": run_id, "phase": "tiny_toolchain", "state": "prepared",
                 "protocol_sha256": digest(protocol), "controller_code_sha256": None,
                 "tensor_inventory_sha256": digest(sorted(BASE)), "created_at_utc_epoch": now,
                 "started_at_utc_epoch": None, "deadline_utc_epoch": None, "last_clock": now,
                 "limits": limits, "generation": {"max_new_tokens": 0, "batch_limit_proposed": 4},
                 "active_candidate": "base", "candidates": {"base": {"candidate_id": "base", "parent_id": None, "artifact_sha256": BASE_HASH}},
                 "requests": {}, "submission": None, "report_sha256": None,
                 "tiny": {"format": FORMAT, "tool_limit": tool_limit, "weights": {"base": copy.deepcopy(BASE)}, "exports": {}}}
        store = cls._create_snapshot(directory, state, clock=clock)
        store.start()
        return store

    def _load(self, connection):
        state = super()._load(connection)
        if state["phase"] != "tiny_toolchain" or state.get("tiny", {}).get("format") != FORMAT:
            raise ControlError("NOT_A_TINY_RUN")
        return state

    def status(self):
        with self._connection() as connection:
            state = self._load(connection)
        result = self._status(state)
        result["tiny"] = {"format": FORMAT, "tool_limit": state["tiny"]["tool_limit"],
                          "tool_used": len(state["requests"]), "candidates": list(state["candidates"]),
                          "active_weights_sha256": state["candidates"][state["active_candidate"]]["artifact_sha256"]}
        return result

    def call(self, name, args):
        entry = next((item for item in TOOLS if item["name"] == name), None)
        if entry is None:
            raise ControlError("UNKNOWN_TOOL")
        validate_arguments(args, entry["inputSchema"])
        if name == "tiny_status":
            return self.status()
        if name == "tiny_receipt":
            with self._connection() as connection:
                row = self._request(self._load(connection), args["request_id"])
            self._materialize(row["result"])
            return {"identity_sha256": row["identity_sha256"], "result": row["result"]}

        def action(state):
            request_id = args["request_id"]
            identity = digest({"name": name, "arguments": args})
            old = state["requests"].get(request_id)
            if old:
                if old["identity_sha256"] != identity:
                    raise ControlError("REQUEST_ID_CONFLICT")
                return old["result"]
            if state["state"] != "running":
                raise ControlError("TOOL_RUN_NOT_RUNNING")
            self._admit_time(state)
            if args["expected_candidate"] != state["active_candidate"]:
                raise ControlError("CANDIDATE_MISMATCH")
            if len(state["requests"]) >= state["tiny"]["tool_limit"]:
                raise ControlError("TOOL_BUDGET_EXCEEDED")
            cost = {}
            if name == "tiny_forward":
                cost = {"dev_requests": 1, "dev_items": len(args["probe_ids"])}
                used = self._usage(state)
                for resource, count in cost.items():
                    if used[resource]["used"] + count > state["limits"][RESOURCES[resource]]:
                        raise ControlError("BUDGET_EXCEEDED")
            result = self._execute(state, name, args)
            state["requests"][request_id] = {"status": "completed", "identity_sha256": identity,
                                            "reservation": cost, "actual_usage": cost, "result": result,
                                            "result_sha256": digest(result)}
            return result
        result = self._mutate(name, action)
        self._materialize(result)
        return result

    def _execute(self, state, name, args):
        active = state["active_candidate"]
        weights = state["tiny"]["weights"][active]
        if digest(weights_checked(weights)) != state["candidates"][active]["artifact_sha256"]:
            raise ControlError("WEIGHT_FINGERPRINT_MISMATCH")
        result = {"candidate_id": active, "weights_sha256": digest(weights)}
        if name == "tiny_list_tensors":
            result["tensors"] = [{"name": key, "shape": [2, 2] if len(value) == 4 else [2],
                                   "dtype": "float32", "writable": key == "hidden.weight"} for key, value in weights.items()]
        elif name == "tiny_read_tensor":
            result.update(tensor=args["tensor"], values=weights[args["tensor"]][:])
        elif name in ("tiny_patch", "tiny_reload"):
            candidate = args["candidate_id"]
            if candidate in state["candidates"]:
                raise ControlError("CANDIDATE_ALREADY_EXISTS")
            if name == "tiny_patch":
                start, values = args["offset"], [f32(v) for v in args["values"]]
                if start + len(values) > 4:
                    raise ControlError("SLICE_OUT_OF_BOUNDS")
                updated = copy.deepcopy(weights)
                updated["hidden.weight"][start:start+len(values)] = values
                if updated == weights:
                    raise ControlError("NO_EFFECTIVE_CHANGE")
            else:
                exported = state["tiny"]["exports"].get(args["artifact_id"])
                if exported is None:
                    raise ControlError("ARTIFACT_NOT_FOUND")
                updated = load_checkpoint(self._artifact_path(args["artifact_id"]), exported["sha256"])["body"]["weights"]
            state["tiny"]["weights"][candidate] = updated
            state["candidates"][candidate] = {"candidate_id": candidate, "parent_id": active, "artifact_sha256": digest(updated)}
            state["active_candidate"] = candidate
            result.update(candidate_id=candidate, parent_id=active, weights_sha256=digest(updated))
        elif name == "tiny_forward":
            result.update(probe_ids=args["probe_ids"], outputs=forward(weights, args["probe_ids"]))
        elif name == "tiny_diff":
            other = state["tiny"]["weights"].get(args["other_candidate"])
            if other is None:
                raise ControlError("CANDIDATE_NOT_FOUND")
            changes = [{"tensor": key, "index": i, "before": a, "after": b}
                       for key in BASE for i, (a, b) in enumerate(zip(other[key], weights[key])) if a != b]
            result.update(other_candidate=args["other_candidate"], changed_parameters=len(changes), changes=changes)
        elif name == "tiny_activate":
            candidate = args["candidate_id"]
            if candidate not in state["candidates"]:
                raise ControlError("CANDIDATE_NOT_FOUND")
            state["active_candidate"] = candidate
            result.update(candidate_id=candidate, weights_sha256=state["candidates"][candidate]["artifact_sha256"])
        elif name == "tiny_export":
            artifact = args["artifact_id"]
            if artifact in state["tiny"]["exports"]:
                raise ControlError("ARTIFACT_ALREADY_EXISTS")
            bundle = checkpoint(weights, active)
            state["tiny"]["exports"][artifact] = bundle
            result.update(artifact_id=artifact, checkpoint_sha256=bundle["sha256"])
        elif name == "tiny_finish":
            state["state"] = "completed"
            result["state"] = "completed"
        return result

    def _artifact_path(self, artifact_id):
        identifier(artifact_id)
        path = self.directory / (artifact_id + ".tiny.json")
        if path.is_symlink():
            raise ControlError("ARTIFACT_SYMLINK_REJECTED")
        return path

    def _materialize(self, result):
        if "checkpoint_sha256" not in result:
            return
        artifact = result["artifact_id"]
        with self._connection() as connection:
            bundle = self._load(connection)["tiny"]["exports"][artifact]
        path = self._artifact_path(artifact)
        # Atomic file publication. A crash before publication is repaired from the
        # committed export receipt, without performing a second logical export.
        import tempfile
        descriptor, temporary = tempfile.mkstemp(prefix=".tiny-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical(bundle) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                load_checkpoint(path, bundle["sha256"])
        finally:
            os.unlink(temporary)
