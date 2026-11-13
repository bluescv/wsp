#!/usr/bin/env python3
"""Trusted-operator launcher; only bounded tools are exposed over MCP stdio.

Parameterized variant: the cross-layer registry is loaded
from the config's explicit ``registry_file`` path and threaded through the
whole lifecycle — validate_config, create, bind and (via the store
instance) every dispatch. Plain non-cross entries keep working: the
registry keyword is optional and ignored by the plain engineering scope.
No module global, no thread-local.
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from wsp_project.baselines import file_sha256, verify_checkpoint, write_json
from wsp_project.errors import ControlError
from wsp_project.linux_isolation import isolate, probe
from wsp_project.manifest import digest, read_json
from wsp_project.qwen_control import QwenRun, TOOLS
from wsp_project.mcp_stdio import Server
from wsp_project.cross_layer_policy import is_cross_layer, load_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    args = parser.parse_args()
    # The frozen gated public bundle is bounded at8MiB; allow configuration
    # metadata headroom only at this trusted entry point. Other JSON stays1MiB.
    cfg = read_json(args.config,max_bytes=9*1024*1024)
    work = Path(cfg["work"]).resolve()
    failure_path = work/"startup.json"
    try:
        run_class=QwenRun
        registry=None
        registry_file=cfg.get("registry_file")
        if registry_file:
            registry=load_registry(Path(registry_file))
        if cfg['scope']=='single_fact_postfreeze_poc':
            from wsp_project.knowledge_poc import KnowledgeRun
            run_class=KnowledgeRun
        run_class.validate_config(cfg,registry=registry)
        source = read_json(ROOT/"source-manifest.json")
        for name,value in source["files"].items():
            if file_sha256(ROOT/name) != value:
                raise ControlError("WORKER_SOURCE_CHANGED")
        upstream = Path(cfg["upstream"])
        manifest = read_json(upstream/"source-manifest.json")
        # Pinned public upstream fixed commit (openwsp/openwsp), verified on
        # the public page; not a private reference:
        # https://github.com/openwsp/openwsp/commit/8d09276736034f6874bb40e7019247c9541a5551
        if manifest["commit"] != "8d09276736034f6874bb40e7019247c9541a5551":
            raise ControlError("OPENWSP_REVISION_MISMATCH")
        for name,row in manifest["files"].items():
            if file_sha256(upstream/name) != row["sha256"]:
                raise ControlError("OPENWSP_SOURCE_CHANGED")
        # Never inherit controller credentials or run model threads before the
        # filesystem/network restrictions are installed.
        security = isolate(work,[ROOT,upstream,cfg["model"],cfg["questions"],cfg["artifact_manifest"]],uid=cfg["uid"])
        security = probe(security,canary=cfg["private_canary"],protected_file=Path(cfg["model"])/"config.json",work=work)
        write_json(work/"isolation.json",security)
        lock = (work/"worker.lock").open("a")
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ControlError("WORKER_ALREADY_RUNNING") from None
        write_json(failure_path,{"state":"loading","isolation_passed":True})
        verify_checkpoint(cfg["model"],cfg["artifact_manifest"])
        sys.path.insert(0,str(upstream/"src"))
        from wsp_project.qwen_engine import QwenEngine
        engine_class=QwenEngine
        if run_class is not QwenRun:
            from wsp_project.knowledge_engine import KnowledgeEngine
            engine_class=KnowledgeEngine
        kwargs=({'write_policy':cfg.get('write_policy'),'public_bundle':cfg.get('public_bundle')}
                if run_class is not QwenRun else {})
        if is_cross_layer(cfg.get('write_policy'), registry):
            from wsp_project.cross_layer_engine import CrossLayerEngine
            engine_class = CrossLayerEngine
            kwargs = {'write_policy':cfg['write_policy'], 'frozen':ROOT/'cross-layer', 'registry':registry}
        engine = engine_class(cfg["model"],work,cfg["questions"],**kwargs)
        evidence = {"base_sha256":engine.base_sha256,"source_sha256":source["sha256"],"inventory":engine.inventory,
            "runtime":{**engine.backend.meta,"openwsp_commit":manifest["commit"],"openwsp_source_sha256":digest(manifest),
                       "dependency_lock_sha256":cfg["dependency_lock_sha256"]}}
        if (work/"control").exists():
            store = run_class(work/"control")
            store.verify_protocol(cfg)
            if store.snapshot()["controller_code_sha256"] != source["sha256"]:
                raise ControlError("WORKER_SOURCE_CHANGED")
        else:
            # create 前已经 run_class.validate_config(cfg,registry=registry)
            # 全量校验；这里再传一次 registry 供 KnowledgeRun 的
            # request_timeout/策略校验使用（普通 QwenRun 忽略该参数）。
            store = run_class.create(work/"control",cfg,evidence,registry=registry)
        store.bind(engine,security,registry=registry)
        write_json(failure_path,{"state":"ready","isolation_passed":True,"source_sha256":source["sha256"]})
        remaining = max(1,store.snapshot()["deadline_utc_epoch"]-time.time())
        watchdog = threading.Timer(remaining,lambda:os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        return Server(store,tools=store.tools,name="wsp-qwen",instructions="Fixed bounded tools, immutable candidates and public query IDs. No paths, code execution, training or network. Post-freeze POC uses a separate declared scope.").serve()
    except Exception as error:
        code = error.code if isinstance(error,ControlError) else "QWEN_STARTUP_FAILED"
        # A stack location in our code is useful without disclosing exceptions,
        # paths, environment values or model/library diagnostics.
        own_frames=[]
        tb=error.__traceback__
        while tb:
            if Path(tb.tb_frame.f_code.co_filename).name in ("qwen_mcp.py","linux_isolation.py","qwen_engine.py","target_backend.py"):
                own_frames.append({"file":Path(tb.tb_frame.f_code.co_filename).name,"line":tb.tb_lineno})
            tb=tb.tb_next
        write_json(failure_path,{"state":"failed","error":code,"frames":own_frames})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
