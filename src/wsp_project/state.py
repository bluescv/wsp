"""Single-writer SQLite transactions for one frozen experiment run.

This is trusted-controller bookkeeping, not a sandbox or a model executor.
The production controller persists metadata only; the tiny test subclass also
stores its bounded public fixture tensors and operation receipts.
"""

import copy
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .budget import RESOURCES, actual_usage, reservation
from .errors import ControlError
from .manifest import canonical, digest, identifier, sha256
from .protocol import validate

PENDING = {"reserved", "dispatched", "uncertain"}


class RunStore:
    def __init__(self, directory, *, clock=time.time):
        self.directory = Path(directory)
        self.database = self.directory / "run.sqlite3"
        self.clock = clock
        if not self.database.is_file() or self.database.is_symlink():
            raise ControlError("RUN_NOT_FOUND")

    @classmethod
    def create(cls, directory, protocol, phase, *, tensor_names=None, clock=time.time):
        if phase not in ("baseline", "tool_smoke", "edit"):
            raise ControlError("INVALID_INITIAL_PHASE")
        if validate(protocol, phase, tensor_names=tensor_names):
            raise ControlError("PROTOCOL_NOT_READY")
        created = clock()
        if type(created) not in (int, float) or not math.isfinite(created):
            raise ControlError("INVALID_CLOCK")
        base_hash = (protocol["tool_smoke"]["fixture_sha256"] if phase == "tool_smoke"
                     else protocol["target"]["base_weights_sha256_manifest"])
        state = {
            "schema_version": 1, "run_id": protocol["run_id"], "phase": phase,
            "state": "prepared", "protocol_sha256": digest(protocol),
            "controller_code_sha256": protocol["controller_code_sha256"],
            "tensor_inventory_sha256": digest(sorted(tensor_names)) if tensor_names is not None else None,
            "created_at_utc_epoch": created, "started_at_utc_epoch": None,
            "deadline_utc_epoch": None, "last_clock": created,
            "limits": copy.deepcopy(protocol["budget"]),
            "generation": {key: protocol["generation"][key] for key in ("max_new_tokens", "batch_limit_proposed")},
            "active_candidate": "base", "candidates": {
                "base": {"candidate_id": "base", "parent_id": None, "artifact_sha256": base_hash}},
            "requests": {}, "submission": None, "report_sha256": None,
        }
        return cls._create_snapshot(directory, state, clock=clock)

    @classmethod
    def _create_snapshot(cls, directory, state, *, clock=time.time):
        directory = Path(directory)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise ControlError("RUN_ALREADY_EXISTS") from None
        connection = sqlite3.connect(directory / "run.sqlite3")
        try:
            connection.executescript("""
                PRAGMA synchronous=FULL;
                CREATE TABLE snapshot (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, at_utc_epoch REAL NOT NULL,
                                     operation TEXT NOT NULL, details TEXT NOT NULL);
            """)
            with connection:
                connection.execute("INSERT INTO snapshot VALUES (1, ?, ?)", (canonical(state), digest(state)))
                connection.execute("INSERT INTO events(at_utc_epoch,operation,details) VALUES (?,?,?)",
                                   (state["created_at_utc_epoch"], "create", canonical({"protocol_sha256": state["protocol_sha256"], "phase": state["phase"]})))
        finally:
            connection.close()
        return cls(directory, clock=clock)

    @contextmanager
    def _connection(self):
        # mode=rw prevents a typo or deleted database from silently creating a new run.
        connection = sqlite3.connect(self.database.resolve().as_uri() + "?mode=rw", uri=True, timeout=5)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _load(connection):
        row = connection.execute("SELECT body, sha256 FROM snapshot WHERE id=1").fetchone()
        try:
            state = json.loads(row[0])
            if digest(state) != row[1] or state["schema_version"] != 1:
                raise ControlError("STATE_INTEGRITY_FAILED")
        except (TypeError, ValueError, KeyError):
            raise ControlError("STATE_INTEGRITY_FAILED") from None
        return state

    def _mutate(self, operation, action):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                state = self._load(connection)
                try:
                    # Timestamp every mutation, including sealing and cancellation.
                    # Dispatch deadlines are checked separately; late receipts remain valid.
                    self._now(state)
                    result = action(state)
                except ControlError as error:
                    # Record rejection without persisting partially mutated state or input values.
                    connection.execute("INSERT INTO events(at_utc_epoch,operation,details) VALUES (?,?,?)",
                                       (state["last_clock"], "rejected", canonical({"operation": operation, "code": error.code})))
                    connection.commit()
                    raise
                connection.execute("UPDATE snapshot SET body=?, sha256=? WHERE id=1", (canonical(state), digest(state)))
                connection.execute("INSERT INTO events(at_utc_epoch,operation,details) VALUES (?,?,?)",
                                   (state["last_clock"], operation, canonical(result)))
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def _now(self, state):
        now = self.clock()
        if type(now) not in (int, float) or not math.isfinite(now) or now < state["last_clock"]:
            raise ControlError("CLOCK_REGRESSION")
        state["last_clock"] = now
        return now

    def _admit_time(self, state):
        now = state["last_clock"]
        if state["deadline_utc_epoch"] is None or now >= state["deadline_utc_epoch"]:
            raise ControlError("RUN_DEADLINE_EXCEEDED")
        return now

    @staticmethod
    def _no_pending(state):
        if any(r["status"] in PENDING for r in state["requests"].values()):
            raise ControlError("UNRESOLVED_REQUESTS")

    @staticmethod
    def _usage(state):
        usage = {name: {"used": 0, "reserved": 0} for name in RESOURCES}
        for request in state["requests"].values():
            category = "reserved" if request["status"] in PENDING else "used"
            values = request["reservation"] if category == "reserved" else request.get("actual_usage", {})
            for key, value in values.items():
                usage[key][category] += value
        return usage

    def status(self):
        with self._connection() as connection:
            state = self._load(connection)
        return self._status(state)

    def _status(self, state):
        result = {key: state[key] for key in (
            "run_id", "phase", "state", "protocol_sha256", "controller_code_sha256",
            "tensor_inventory_sha256", "created_at_utc_epoch", "started_at_utc_epoch",
            "deadline_utc_epoch", "active_candidate", "submission", "report_sha256")}
        result["budget"] = {key: {**value, "limit": state["limits"][RESOURCES[key]]} for key, value in self._usage(state).items()}
        result["requests"] = {key: row["status"] for key, row in state["requests"].items()}
        return result

    def verify_protocol(self, protocol):
        if digest(protocol) != self.status()["protocol_sha256"]:
            raise ControlError("PROTOCOL_CHANGED")

    def events(self):
        with self._connection() as connection:
            return [{"seq": row[0], "at_utc_epoch": row[1], "operation": row[2], "details": json.loads(row[3])}
                    for row in connection.execute("SELECT seq,at_utc_epoch,operation,details FROM events ORDER BY seq")]

    def start(self):
        def action(state):
            if state["state"] != "prepared":
                raise ControlError("INVALID_TRANSITION")
            now = state["last_clock"]
            state.update(state="running", started_at_utc_epoch=now,
                         deadline_utc_epoch=now + state["limits"]["total_wallclock_seconds"])
            return {"state": "running", "deadline_utc_epoch": state["deadline_utc_epoch"]}
        return self._mutate("start", action)

    def pause(self):
        return self._transition("pause", "running", "paused")

    def resume(self):
        # Pausing never stops or restarts the wall-clock deadline.
        return self._transition("resume", "paused", "running")

    def _transition(self, operation, before, after):
        def action(state):
            if state["state"] != before:
                raise ControlError("INVALID_TRANSITION")
            state["state"] = after
            return {"state": after}
        return self._mutate(operation, action)

    def reserve_request(self, request_id, kind, input_sha256, candidate_id, *,
                        items=0, activation_bytes=0, purpose="normal"):
        def action(state):
            identifier(request_id)
            identifier(candidate_id)
            sha256(input_sha256)
            if purpose not in ("normal", "recovery") or (purpose == "recovery" and kind != "editor"):
                raise ControlError("INVALID_REQUEST_PURPOSE")
            tokens = state["generation"]["max_new_tokens"] if kind in ("dev_forward", "baseline", "evaluation") else 0
            cost = reservation(kind, items=items, max_new_tokens=tokens, activation_bytes=activation_bytes)
            identity = {"kind": kind, "input_sha256": input_sha256, "candidate_id": candidate_id,
                        "reservation": cost, "purpose": purpose}
            old = state["requests"].get(request_id)
            if old:
                if old["identity_sha256"] != digest(identity):
                    raise ControlError("REQUEST_ID_CONFLICT")
                if old["status"] == "completed":
                    return {"request_id": request_id, "replayed": True, "result_sha256": old["result_sha256"]}
                raise ControlError("REQUEST_NOT_REPLAYABLE")
            if any(r["status"] in ("dispatched", "uncertain") for r in state["requests"].values()):
                raise ControlError("UNRESOLVED_REQUESTS")
            allowed = {"baseline": {"baseline"}, "tool_smoke": {"editor", "dev_forward", "activation"},
                       "edit": {"editor", "dev_forward", "activation"}}
            allowed_kinds = {"evaluation"} if state["state"] == "evaluating" else allowed[state["phase"]]
            if state["state"] not in ("running", "evaluating") or kind not in allowed_kinds:
                raise ControlError("REQUEST_NOT_ALLOWED_IN_STATE")
            if candidate_id != state["active_candidate"]:
                raise ControlError("CANDIDATE_MISMATCH")
            if items > state["generation"]["batch_limit_proposed"]:
                raise ControlError("BATCH_LIMIT_EXCEEDED")
            now = self._admit_time(state)
            usage = self._usage(state)
            for key, value in cost.items():
                ceiling = state["limits"][RESOURCES[key]]
                if key == "editor_requests" and purpose != "recovery":
                    ceiling -= state["limits"]["recovery_request_reserve"]
                if usage[key]["used"] + usage[key]["reserved"] + value > ceiling:
                    raise ControlError("BUDGET_EXCEEDED")
            state["requests"][request_id] = {
                **identity, "identity_sha256": digest(identity), "status": "reserved",
                "reserved_at_utc_epoch": now,
                "deadline_utc_epoch": min(state["deadline_utc_epoch"], now + state["limits"]["per_request_timeout_seconds"]),
            }
            return {"request_id": request_id, "replayed": False, "reservation": cost}
        return self._mutate("reserve_request", action)

    @staticmethod
    def _request(state, request_id):
        identifier(request_id)
        if request_id not in state["requests"]:
            raise ControlError("REQUEST_NOT_FOUND")
        return state["requests"][request_id]

    def mark_dispatched(self, request_id):
        def action(state):
            row = self._request(state, request_id)
            if row["status"] != "reserved" or state["state"] not in ("running", "evaluating"):
                raise ControlError("DISPATCH_NOT_ALLOWED")
            if any(r["status"] in ("dispatched", "uncertain") for r in state["requests"].values()):
                raise ControlError("UNRESOLVED_REQUESTS")
            if self._admit_time(state) >= row["deadline_utc_epoch"]:
                raise ControlError("REQUEST_DEADLINE_EXCEEDED")
            row["status"] = "dispatched"
            return {"request_id": request_id, "status": "dispatched"}
        return self._mutate("mark_dispatched", action)

    def cancel_reserved(self, request_id):
        def action(state):
            row = self._request(state, request_id)
            if row["status"] != "reserved":
                raise ControlError("CANNOT_CANCEL_DISPATCHED_REQUEST")
            row.update(status="cancelled", actual_usage={})
            return {"request_id": request_id, "status": "cancelled"}
        return self._mutate("cancel_reserved", action)

    def mark_uncertain(self, request_id):
        def action(state):
            row = self._request(state, request_id)
            if row["status"] != "dispatched":
                raise ControlError("INVALID_REQUEST_TRANSITION")
            row["status"] = "uncertain"
            return {"request_id": request_id, "status": "uncertain"}
        return self._mutate("mark_uncertain", action)

    def settle_request(self, request_id, result_sha256, *, usage=None, outcome="completed", evidence_sha256=None):
        def action(state):
            sha256(result_sha256)
            if outcome not in ("completed", "failed"):
                raise ControlError("INVALID_OUTCOME")
            row = self._request(state, request_id)
            if row["status"] not in ("dispatched", "uncertain"):
                raise ControlError("REQUEST_NOT_SETTLEABLE")
            if row["status"] == "uncertain":
                sha256(evidence_sha256)
            elif evidence_sha256 is not None:
                sha256(evidence_sha256)
            actual = actual_usage(row["kind"], row["reservation"], usage)
            row.update(status=outcome, actual_usage=actual, result_sha256=result_sha256,
                       evidence_sha256=evidence_sha256)
            return {"request_id": request_id, "status": outcome, "actual_usage": actual,
                    "result_sha256": result_sha256, "evidence_sha256": evidence_sha256}
        return self._mutate("settle_request", action)

    def _can_change_candidate(self, state):
        if state["state"] != "running" or state["phase"] not in ("edit", "tool_smoke"):
            raise ControlError("CANDIDATE_CHANGE_NOT_ALLOWED")
        self._no_pending(state)
        self._admit_time(state)

    def register_candidate(self, candidate_id, parent_id, artifact_sha256):
        def action(state):
            identifier(candidate_id)
            identifier(parent_id)
            sha256(artifact_sha256)
            self._can_change_candidate(state)
            if candidate_id in state["candidates"]:
                raise ControlError("CANDIDATE_ALREADY_EXISTS")
            if parent_id != state["active_candidate"]:
                raise ControlError("CANDIDATE_PARENT_MISMATCH")
            candidate = {"candidate_id": candidate_id, "parent_id": parent_id, "artifact_sha256": artifact_sha256}
            state["candidates"][candidate_id] = candidate
            state["active_candidate"] = candidate_id
            return candidate
        return self._mutate("register_candidate", action)

    def activate_candidate(self, candidate_id):
        def action(state):
            identifier(candidate_id)
            self._can_change_candidate(state)
            if candidate_id not in state["candidates"]:
                raise ControlError("CANDIDATE_NOT_FOUND")
            state["active_candidate"] = candidate_id
            return {"active_candidate": candidate_id}
        return self._mutate("activate_candidate", action)

    def submit(self, candidate_id, checkpoint_manifest_sha256):
        def action(state):
            identifier(candidate_id)
            sha256(checkpoint_manifest_sha256)
            if state["phase"] != "edit" or state["state"] not in ("running", "paused"):
                raise ControlError("SUBMISSION_NOT_ALLOWED")
            self._no_pending(state)
            if candidate_id == "base" or candidate_id != state["active_candidate"]:
                raise ControlError("CANDIDATE_MISMATCH")
            # Sealing is allowed after a deadline: it does not dispatch work.
            body = {"candidate_id": candidate_id,
                    "edit_sha256": state["candidates"][candidate_id]["artifact_sha256"],
                    "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
                    "protocol_sha256": state["protocol_sha256"]}
            body["submission_sha256"] = digest(body)
            state.update(state="submitted", submission=body)
            return body
        return self._mutate("submit", action)

    def begin_evaluation(self):
        def action(state):
            if state["state"] != "submitted" or state["submission"] is None:
                raise ControlError("SEALED_SUBMISSION_REQUIRED")
            self._admit_time(state)
            state["state"] = "evaluating"
            return {"state": "evaluating", "submission_sha256": state["submission"]["submission_sha256"]}
        return self._mutate("begin_evaluation", action)

    def finish(self, report_sha256):
        def action(state):
            sha256(report_sha256)
            if not (state["state"] == "evaluating" or (state["state"] == "running" and state["phase"] != "edit")):
                raise ControlError("FINISH_NOT_ALLOWED")
            self._no_pending(state)
            state.update(state="completed", report_sha256=report_sha256)
            return {"state": "completed", "report_sha256": report_sha256}
        return self._mutate("finish", action)

    def fail(self, evidence_sha256):
        def action(state):
            sha256(evidence_sha256)
            if state["state"] in ("completed", "failed"):
                raise ControlError("INVALID_TRANSITION")
            state["state"] = "failed"
            return {"state": "failed", "evidence_sha256": evidence_sha256}
        return self._mutate("fail", action)
