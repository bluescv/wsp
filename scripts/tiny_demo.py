#!/usr/bin/env python3
"""Runnable tiny demo: one process, real TinyStore tool round-trips.

Exercises the full tiny lifecycle WITHOUT private verification code and
without the MCP stdio server (serve only blocks on stdin; it is a listener,
not evidence of tool round-trips):

    init -> list/read tensors -> forward -> patch (new candidate)
    -> forward on the new weights -> diff vs base -> activate base
    (rollback) -> forward again (base restored) -> export checkpoint
    -> reload from the exported artifact -> finish.

Every step asserts its own expected outcome; exit code 0 means every
round-trip actually happened against the durable store. Run from the
candidate root:

    python scripts/tiny_demo.py --run-dir demo-run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsp_project.errors import ControlError
from wsp_project.tiny import TinyStore, forward, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", default="tiny-demo-1")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if run_dir.exists() and (run_dir / "run.sqlite3").is_file():
        print("run dir already initialized; use a fresh --run-dir")
        return 2

    store = TinyStore.create(run_dir, run_id=args.run_id)
    print("init:", store.status()["tiny"]["candidates"])

    def call(name, args_):
        result = store.call(name, args_)
        print(name, "->", {k: v for k, v in result.items() if k != "changes"})
        return result

    # 1. list + read the one writable tensor.
    listing = call("tiny_list_tensors", {"request_id": "r1", "expected_candidate": "base"})
    writable = [t["name"] for t in listing["tensors"] if t["writable"]]
    assert writable == ["hidden.weight"], writable
    read = call("tiny_read_tensor",
                {"request_id": "r2", "expected_candidate": "base",
                 "tensor": "hidden.weight"})
    assert read["values"] == [1.0, 0.0, 0.0, 1.0], read["values"]

    # 2. forward on base weights (deterministic outputs).
    base_out = call("tiny_forward",
                    {"request_id": "r3", "expected_candidate": "base",
                     "probe_ids": ["probe_a", "probe_b"]})
    from wsp_project.tiny import BASE, PROBES
    assert base_out["outputs"] == forward(BASE, ["probe_a", "probe_b"]), base_out

    # 3. patch hidden.weight -> new candidate c1, forward changes.
    call("tiny_patch",
         {"request_id": "r4", "expected_candidate": "base",
          "candidate_id": "c1", "tensor": "hidden.weight",
          "offset": 0, "values": [2.0]})
    c1_out = call("tiny_forward",
                  {"request_id": "r5", "expected_candidate": "c1",
                   "probe_ids": ["probe_a", "probe_b"]})
    assert c1_out["outputs"] != base_out["outputs"], "patch had no effect"

    # 4. diff c1 vs active(base after rollback below); do it while c1 active.
    diff = call("tiny_diff",
                {"request_id": "r6", "expected_candidate": "c1",
                 "other_candidate": "base"})
    assert diff["changed_parameters"] == 1, diff

    # 5. rollback: activate base, forward matches base again.
    call("tiny_activate",
         {"request_id": "r7", "expected_candidate": "c1",
          "candidate_id": "base"})
    rolled = call("tiny_forward",
                  {"request_id": "r8", "expected_candidate": "base",
                   "probe_ids": ["probe_a", "probe_b"]})
    assert rolled["outputs"] == base_out["outputs"], "rollback failed"

    # 6. export while base active -> reload from the artifact as c2.
    export = call("tiny_export",
                  {"request_id": "r9", "expected_candidate": "base",
                   "artifact_id": "art-1"})
    artifact_path = run_dir / "art-1.tiny.json"
    assert artifact_path.is_file(), export
    bundle = load_checkpoint(artifact_path, export["checkpoint_sha256"])
    assert bundle["body"]["weights"] == BASE
    reloaded = call("tiny_reload",
                    {"request_id": "r10", "expected_candidate": "base",
                     "artifact_id": "art-1", "candidate_id": "c2"})
    assert reloaded["candidate_id"] == "c2"

    # 7. finish seals the run.
    call("tiny_finish", {"request_id": "r11", "expected_candidate": "c2"})
    status = store.status()
    assert status["state"] == "completed", status["state"]
    print("tiny demo complete: 11 tool round-trips, all asserted")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ControlError as error:
        print(f"ControlError: {error.code}", file=sys.stderr)
        raise SystemExit(2)
