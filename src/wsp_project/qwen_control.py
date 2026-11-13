"""Durable, bounded engineering Qwen tools; does not authorize formal editing.

Parameterized variant: the cross-layer registry threads
through the full store lifecycle as an explicit instance attribute.

- ``validate_config``/``request_timeout``/``create``/``bind`` accept an
  optional ``registry`` keyword (plain ``QwenRun`` scope accepts and
  ignores it — the original no-registry entry points keep working);
- ``create`` forwards the registry to ``validate_config`` and
  ``request_timeout`` WITHOUT mutating ``config`` (``protocol_sha256``
  still digests exactly the config bytes the caller supplied);
- ``bind`` stores the registry on the instance (``self.registry``) and
  ``call`` passes it to ``validate_dispatch`` — KnowledgeRun's cross-layer
  progress checks therefore always see the engine's own registry, never a
  module global or thread-local.

External GPU/filesystem work occurs after a durable dispatch reservation. A crash
or unknown exception leaves an uncertain request that cannot be auto-replayed.
"""
import copy
import os
import threading
import time

from .budget import RESOURCES
from .errors import ControlError
from .manifest import digest, identifier
from .state import RunStore
from .tiny import ID, schema, validate_arguments

COMMON = {"request_id":ID,"expected_candidate":ID}
EXTRA = {
    "qwen_inventory":{"offset":{"type":"integer","minimum":0,"maximum":10000},"count":{"type":"integer","minimum":1,"maximum":32}},
    "qwen_read_slice":{"tensor":ID,"offset":{"type":"integer","minimum":0,"maximum":1000000000},"count":{"type":"integer","minimum":1,"maximum":256}},
    "qwen_patch":{"candidate_id":ID,"tensor":ID,"offset":{"type":"integer","minimum":0,"maximum":1000000000},
                  "values":{"type":"array","minItems":1,"maxItems":256,"items":{"type":"number","minimum":-1,"maximum":1}}},
    "qwen_activate":{"candidate_id":ID},
    "qwen_diff":{},
    "qwen_forward":{"query_ids":{"type":"array","minItems":1,"maxItems":4,"items":ID}},
    "qwen_export":{"artifact_id":ID},
    "qwen_apply_diff":{"artifact_id":ID,"candidate_id":ID},
    "qwen_finish":{},
}
TOOLS = [{"name":name,"description":"Bounded Qwen engineering tool: "+name.removeprefix("qwen_"),"inputSchema":schema({**COMMON,**props})} for name,props in EXTRA.items()]
TOOLS += [{"name":"qwen_status","description":"Durable state, budget and remote isolation evidence.","inputSchema":schema({})},
          {"name":"qwen_receipt","description":"Read a completed receipt; never re-execute an uncertain operation.","inputSchema":schema({"request_id":ID})}]


class QwenRun(RunStore):
    scope = 'qwen_toolchain_engineering'
    phase = 'qwen_toolchain'
    tools = TOOLS
    ceilings = {"tool_limit":96,"forward_limit":16,"item_limit":32,"candidate_limit":8,"seconds":3600}
    write_tools = ("qwen_patch", "qwen_apply_diff")

    @classmethod
    def validate_config(cls, config, registry=None):
        # registry is accepted for interface compatibility: the plain
        # engineering scope never reads it (a config that carries a
        # cross-layer write_policy is rejected here by scope first).
        if config.get("scope") != cls.scope or config.get("formal_run_authorized") is not False:
            raise ControlError("ENGINEERING_SCOPE_REQUIRED")

    @classmethod
    def request_timeout(cls, config, registry=None):
        return 120

    @classmethod
    def create(cls, directory, config, evidence, *, clock=time.time, registry=None):
        identifier(config["run_id"])
        cls.validate_config(config, registry=registry)
        for key, ceiling in cls.ceilings.items():
            if type(config.get(key)) is not int or not 1 <= config[key] <= ceiling:
                raise ControlError("INVALID_QWEN_LIMIT")
        now = clock()
        limits = dict.fromkeys(RESOURCES.values(),0)
        limits.update(dev_forward_request_limit=config["forward_limit"],dev_forward_item_limit=config["item_limit"],
            generated_token_limit=config["item_limit"]*16,total_wallclock_seconds=config["seconds"],
            per_request_timeout_seconds=cls.request_timeout(config, registry=registry),recovery_request_reserve=0)
        base = {"candidate_id":"base","parent_id":None,"artifact_sha256":evidence["base_sha256"],"files":{},"edited_names":[]}
        state = {"schema_version":1,"run_id":config["run_id"],"phase":cls.phase,"state":"prepared",
            "protocol_sha256":digest(config),"controller_code_sha256":evidence["source_sha256"],"tensor_inventory_sha256":digest(evidence["inventory"]),
            "created_at_utc_epoch":now,"started_at_utc_epoch":None,"deadline_utc_epoch":None,"last_clock":now,
            "limits":limits,"generation":{"max_new_tokens":16,"batch_limit_proposed":4},"active_candidate":"base",
            "candidates":{"base":base},"requests":{},"submission":None,"report_sha256":None,
            "qwen":{"config":copy.deepcopy(config),"evidence":copy.deepcopy(evidence),"exports":{}}}
        result = cls._create_snapshot(directory,state,clock=clock)
        result.start()
        return result

    def _load(self, connection):
        state = super()._load(connection)
        if state["phase"] != self.phase:
            raise ControlError("NOT_A_QWEN_TOOLCHAIN")
        return state

    def snapshot(self):
        with self._connection() as connection:
            return self._load(connection)

    def bind(self, engine, isolation, registry=None):
        state = self.snapshot()
        if engine.base_sha256 != state["candidates"]["base"]["artifact_sha256"] or digest(engine.inventory) != state["tensor_inventory_sha256"]:
            raise ControlError("RUNTIME_BASE_CHANGED")
        self.engine = engine
        self.isolation = isolation
        # Registry is fixed per instance at bind time; dispatch below always
        # passes this exact object — two engines in one thread each hold
        # their own registry and never observe each other's.
        self.registry = registry
        # Reconstruct only committed candidates, never uncertain operations.
        engine.activate(state["candidates"][state["active_candidate"]])
        return self

    def status(self):
        state = self.snapshot()
        result = self._status(state)
        result["qwen"] = {"scope":self.scope,"formal_run_authorized":False,
            "tool_used":len(state["requests"]),"tool_limit":state["qwen"]["config"]["tool_limit"],
            "candidate_ids":list(state["candidates"]),"artifact_ids":list(state["qwen"]["exports"]),
            "active_weights_sha256":state["candidates"][state["active_candidate"]]["artifact_sha256"],
            "runtime":state["qwen"]["evidence"].get("runtime",{}),"isolation":getattr(self,"isolation",{})}
        return result

    def call(self, name, args):
        tool = next((item for item in self.tools if item["name"] == name),None)
        if tool is None:
            raise ControlError("UNKNOWN_TOOL")
        validate_arguments(args,tool["inputSchema"])
        if name == "qwen_status":
            return self.status()
        if name == "qwen_receipt":
            row = self._request(self.snapshot(),args["request_id"])
            if row["status"] != "completed":
                raise ControlError("REQUEST_NOT_REPLAYABLE")
            return {"identity_sha256":row["identity_sha256"],"result":row["result"]}
        identity = digest({"name":name,"arguments":args})
        def reserve(state):
            old = state["requests"].get(args["request_id"])
            if old:
                if old["identity_sha256"] != identity:
                    raise ControlError("REQUEST_ID_CONFLICT")
                if old["status"] != "completed":
                    raise ControlError("REQUEST_NOT_REPLAYABLE")
                return {"replay":True,"result":old["result"]}
            self._no_pending(state)
            if state["state"] != "running":
                raise ControlError("TOOL_RUN_NOT_RUNNING")
            self._admit_time(state)
            if args["expected_candidate"] != state["active_candidate"]:
                raise ControlError("CANDIDATE_MISMATCH")
            cfg = state["qwen"]["config"]
            if len(state["requests"]) >= cfg["tool_limit"]:
                raise ControlError("TOOL_BUDGET_EXCEEDED")
            if name in self.write_tools:
                if args["candidate_id"] in state["candidates"]:
                    raise ControlError("CANDIDATE_ALREADY_EXISTS")
                if len(state["candidates"]) >= cfg["candidate_limit"]:
                    raise ControlError("CANDIDATE_LIMIT_EXCEEDED")
            if name == "qwen_activate" and args["candidate_id"] not in state["candidates"]:
                raise ControlError("CANDIDATE_NOT_FOUND")
            if name == "qwen_export" and args["artifact_id"] in state["qwen"]["exports"]:
                raise ControlError("ARTIFACT_ALREADY_EXISTS")
            if name == "qwen_export" and len(state["qwen"]["exports"]) >= 2:
                raise ControlError("ARTIFACT_LIMIT_EXCEEDED")
            if name == "qwen_apply_diff" and args["artifact_id"] not in state["qwen"]["exports"]:
                raise ControlError("ARTIFACT_NOT_FOUND")
            self.validate_dispatch(name, args, state, registry=getattr(self, "registry", None))
            self.engine.validate(name,args)
            cost = self.cost(name, args)
            if cost:
                usage = self._usage(state)
                if any(usage[key]["used"]+usage[key]["reserved"]+value > state["limits"][RESOURCES[key]] for key,value in cost.items()):
                    raise ControlError("BUDGET_EXCEEDED")
            state["requests"][args["request_id"]] = {"status":"dispatched","identity_sha256":identity,"reservation":cost,"operation":name}
            return {"replay":False,"request_id":args["request_id"]}
        reservation = self._mutate("qwen_dispatch",reserve)
        if reservation["replay"]:
            return reservation["result"]
        state = self.snapshot()
        watchdog = threading.Timer(state["limits"]["per_request_timeout_seconds"], lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        try:
            result, candidate, artifact = self.engine.execute(name,args,state)
        except Exception:
            def uncertain(state):
                state["requests"][args["request_id"]]["status"] = "uncertain"
                state["state"] = "failed"
                return {"request_id":args["request_id"],"error":"QWEN_OPERATION_UNCERTAIN"}
            self._mutate("qwen_uncertain",uncertain)
            raise ControlError("QWEN_OPERATION_UNCERTAIN") from None
        finally:
            watchdog.cancel()
        def settle(state):
            row = state["requests"][args["request_id"]]
            if row["status"] != "dispatched" or row["identity_sha256"] != identity:
                raise ControlError("REQUEST_NOT_SETTLEABLE")
            if candidate:
                state["candidates"][candidate["candidate_id"]] = candidate
                state["active_candidate"] = candidate["candidate_id"]
            elif name == "qwen_activate":
                state["active_candidate"] = args["candidate_id"]
            if artifact:
                state["qwen"]["exports"][args["artifact_id"]] = artifact
            actual = dict(row["reservation"])
            if "generated_tokens" in actual:
                used = sum(len(item["generated_token_ids"]) for item in result["predictions"])
                if used > actual["generated_tokens"]:
                    raise ControlError("USAGE_EXCEEDS_RESERVATION")
                actual["generated_tokens"] = used
            row.update(status="completed",actual_usage=actual,result=result,result_sha256=digest(result))
            if name == "qwen_finish":
                state.update(state="completed",report_sha256=digest(result))
                self.seal(state, result)
            return result
        return self._mutate("qwen_settle",settle)

    def cost(self, name, args):
        if name == 'qwen_forward':
            count = len(args['query_ids'])
            return {'dev_requests':1,'dev_items':count,'generated_tokens':16*count}
        return {}

    def validate_dispatch(self, name, args, state, registry=None):
        pass

    def seal(self, state, result):
        pass
