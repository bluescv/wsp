"""Control-plane CLI. Commands record state; none executes models or networking."""

import argparse
import json
import sqlite3
from dataclasses import asdict

from .errors import ControlError
from .manifest import read_json
from .protocol import PHASES, validate
from .state import RunStore


class Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's normal error includes untrusted argument values.
        raise ControlError("INVALID_ARGUMENTS")


def parser():
    root = Parser(description="WSP local protocol and run-state control (no model execution).")
    commands = root.add_subparsers(dest="command", required=True, parser_class=Parser)
    for command in ("validate", "init"):
        p = commands.add_parser(command)
        p.add_argument("--protocol", required=True)
        p.add_argument("--phase", choices=PHASES, default="preflight" if command == "validate" else None, required=command == "init")
        p.add_argument("--tensor-inventory", help="JSON array from the trusted model inventory")
        p.add_argument("--run-dir", required=command == "init")
    for command in ("status", "events", "verify", "start", "pause", "resume", "reserve", "dispatch",
                    "uncertain", "cancel", "settle", "candidate", "activate", "submit", "evaluate", "finish", "fail"):
        p = commands.add_parser(command)
        p.add_argument("--run-dir", required=True)
        if command == "verify":
            p.add_argument("--protocol", required=True)
        if command in ("reserve", "dispatch", "uncertain", "cancel", "settle"):
            p.add_argument("--request-id", required=True)
        if command == "reserve":
            p.add_argument("--kind", choices=("editor", "dev_forward", "baseline", "evaluation", "activation"), required=True)
            p.add_argument("--input-sha256", required=True)
            p.add_argument("--candidate-id", required=True)
            p.add_argument("--items", type=int, default=0)
            p.add_argument("--activation-bytes", type=int, default=0)
            p.add_argument("--purpose", choices=("normal", "recovery"), default="normal")
        if command == "settle":
            p.add_argument("--result-sha256", required=True)
            p.add_argument("--usage", help="JSON actual usage, or omit to charge the full reservation")
            p.add_argument("--outcome", choices=("completed", "failed"), default="completed")
            p.add_argument("--evidence-sha256")
        if command in ("candidate", "activate", "submit"):
            p.add_argument("--candidate-id", required=True)
        if command == "candidate":
            p.add_argument("--parent-id", required=True)
            p.add_argument("--artifact-sha256", required=True)
        if command == "submit":
            p.add_argument("--checkpoint-manifest-sha256", required=True)
        if command == "finish":
            p.add_argument("--report-sha256", required=True)
        if command == "fail":
            p.add_argument("--evidence-sha256", required=True)
    return root


def execute(args):
    command = args.command
    if command in ("validate", "init"):
        protocol = read_json(args.protocol)
        inventory = read_json(args.tensor_inventory) if args.tensor_inventory else None
        submitted = None
        if args.phase == "evaluate" and args.run_dir:
            run = RunStore(args.run_dir)
            run.verify_protocol(protocol)
            status = run.status()
            if status["state"] in ("submitted", "evaluating", "completed") and status["submission"]:
                submitted = status["submission"]["submission_sha256"]
        issues = validate(protocol, args.phase, tensor_names=inventory, submission_sha256=submitted)
        if issues or command == "validate":
            return {"valid": not issues, "phase": args.phase, "issues": [asdict(x) for x in issues]}, 2 if issues else 0
        run = RunStore.create(args.run_dir, protocol, args.phase, tensor_names=inventory)
        return run.status(), 0
    run = RunStore(args.run_dir)
    if command in ("status", "events", "start", "pause", "resume"):
        return getattr(run, command)(), 0
    if command == "verify":
        run.verify_protocol(read_json(args.protocol))
        return {"protocol_matches": True}, 0
    if command == "reserve":
        return run.reserve_request(args.request_id, args.kind, args.input_sha256, args.candidate_id,
                                   items=args.items, activation_bytes=args.activation_bytes, purpose=args.purpose), 0
    if command in ("dispatch", "uncertain", "cancel"):
        method = {"dispatch": "mark_dispatched", "uncertain": "mark_uncertain", "cancel": "cancel_reserved"}[command]
        return getattr(run, method)(args.request_id), 0
    if command == "settle":
        return run.settle_request(args.request_id, args.result_sha256,
                                  usage=read_json(args.usage) if args.usage else None,
                                  outcome=args.outcome, evidence_sha256=args.evidence_sha256), 0
    if command == "candidate":
        return run.register_candidate(args.candidate_id, args.parent_id, args.artifact_sha256), 0
    if command == "activate":
        return run.activate_candidate(args.candidate_id), 0
    if command == "submit":
        return run.submit(args.candidate_id, args.checkpoint_manifest_sha256), 0
    if command == "evaluate":
        return run.begin_evaluation(), 0
    if command == "finish":
        return run.finish(args.report_sha256), 0
    if command == "fail":
        return run.fail(args.evidence_sha256), 0
    raise ControlError("UNKNOWN_COMMAND")


def main(argv=None):
    try:
        result, code = execute(parser().parse_args(argv))
    except ControlError as error:
        result, code = {"error": error.code}, 2
    except (OSError, sqlite3.Error):
        result, code = {"error": "STORAGE_OPERATION_FAILED"}, 2
    except (ValueError, TypeError, KeyError, RecursionError):
        result, code = {"error": "INVALID_CONTROL_DATA"}, 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return code
