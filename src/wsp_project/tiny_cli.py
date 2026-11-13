"""Trusted-operator entry; CLI paths are not exposed as MCP arguments."""

import sys

from .cli import Parser
from .errors import ControlError
from .manifest import canonical
from .mcp_stdio import Server
from .tiny import PROBES, TinyStore, forward, load_checkpoint


def main(argv=None):
    parser = Parser(description="Tiny fixture tools only; no external model or GPU execution.")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
    init = commands.add_parser("init")
    init.add_argument("--run-dir", required=True)
    init.add_argument("--run-id", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--run-dir", required=True)
    reload_command = commands.add_parser("reload")
    reload_command.add_argument("--checkpoint", required=True)
    reload_command.add_argument("--sha256", required=True)
    try:
        args = parser.parse_args(argv)
        if args.command == "init":
            result = TinyStore.create(args.run_dir, run_id=args.run_id).status()
        elif args.command == "serve":
            return Server(TinyStore(args.run_dir)).serve()
        else:
            body = load_checkpoint(args.checkpoint, args.sha256)["body"]
            result = {"fresh_process": True, "format": body["format"], "weights_sha256": body["weights_sha256"],
                      "probe_ids": list(PROBES), "outputs": forward(body["weights"], list(PROBES))}
        print(canonical(result))
        return 0
    except ControlError as error:
        print(canonical({"error": error.code}), file=sys.stderr)
    except Exception:
        print(canonical({"error": "TINY_OPERATION_FAILED"}), file=sys.stderr)
    return 2
