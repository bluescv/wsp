"""Minimal MCP stdio server: initialize, ping, tools/list and tools/call.

Implements protocol 2025-11-25 and compatible 2025-06-18 tools subset.
No arbitrary code, file paths, HTTP listener, sampling, or dynamic tools.
"""

import sys

from .errors import ControlError
from .manifest import canonical, identifier, parse_json
from .tiny import TOOLS

VERSIONS = ("2025-11-25", "2025-06-18")
MAX_FRAME = 65_536


def error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class Server:
    def __init__(self, store, *, tools=None, name="wsp-tiny", instructions="Engineering tiny fixture only; not Qwen or a formal editing experiment."):
        self.store = store
        self.tools = TOOLS if tools is None else tools
        self.name = name
        self.instructions = instructions
        self.initialized = False
        self.ready = False

    def handle(self, message):
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return error(None, -32600, "INVALID_REQUEST")
        request_id = message.get("id")
        if "id" in message:
            try:
                if type(request_id) is int:
                    if abs(request_id) > 2**53 - 1:
                        raise ControlError("INVALID_ID")
                else:
                    identifier(request_id)
            except ControlError:
                return error(None, -32600, "INVALID_REQUEST_ID")
        method, params = message["method"], message.get("params", {})
        if "id" not in message:
            if method == "notifications/initialized" and self.initialized:
                self.ready = True
            # Operations run synchronously and are bounded; cancellation cannot
            # undo a committed operation. Its durable receipt remains available.
            return None
        if not isinstance(params, dict):
            return error(request_id, -32602, "INVALID_PARAMS")
        if method == "ping":
            result = {}
        elif method == "initialize":
            if self.initialized:
                return error(request_id, -32600, "ALREADY_INITIALIZED")
            if not isinstance(params.get("protocolVersion"), str) or not isinstance(params.get("capabilities"), dict) or not isinstance(params.get("clientInfo"), dict):
                return error(request_id, -32602, "INVALID_INITIALIZE")
            version = params["protocolVersion"] if params["protocolVersion"] in VERSIONS else VERSIONS[0]
            result = {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": self.name, "version": "0.1.0"},
                      "instructions": self.instructions}
            self.initialized = True
        elif not self.ready:
            return error(request_id, -32002, "NOT_INITIALIZED")
        elif method == "tools/list":
            # Real Codex clients attach standard MCP metadata to list requests.
            # This server returns one complete bounded page; no continuation.
            if set(params)-{"_meta","cursor"} or ("_meta" in params and not isinstance(params["_meta"],dict)) or params.get("cursor") is not None:
                return error(request_id, -32602, "INVALID_PARAMS")
            result = {"tools": self.tools}
        elif method == "tools/call":
            if (set(params) - {"name", "arguments", "_meta"}
                    or ("_meta" in params and not isinstance(params["_meta"], dict))
                    or not isinstance(params.get("name"), str)
                    or params["name"] not in {item["name"] for item in self.tools}):
                return error(request_id, -32602, "UNKNOWN_TOOL_OR_INVALID_PARAMS")
            try:
                payload = self.store.call(params["name"], params.get("arguments", {}))
                result = {"content": [{"type": "text", "text": canonical(payload)}], "structuredContent": payload, "isError": False}
            except ControlError as failure:
                payload = {"error": failure.code}
                result = {"content": [{"type": "text", "text": canonical(payload)}], "structuredContent": payload, "isError": True}
            except Exception:
                return error(request_id, -32603, "TOOL_STORAGE_OR_INTERNAL_FAILURE")
        else:
            return error(request_id, -32601, "METHOD_NOT_FOUND")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def serve(self, source=None, sink=None):
        source = source if source is not None else sys.stdin.buffer
        sink = sink if sink is not None else sys.stdout.buffer
        while True:
            line = source.readline(MAX_FRAME + 1)
            if not line:
                return 0
            if len(line) > MAX_FRAME or not line.endswith(b"\n"):
                sink.write((canonical(error(None, -32700, "FRAME_LIMIT_OR_DELIMITER")) + "\n").encode())
                sink.flush()
                return 2
            try:
                response = self.handle(parse_json(line.decode("utf-8")))
            except (ControlError, UnicodeError):
                response = error(None, -32700, "PARSE_ERROR")
            except Exception:
                response = error(None, -32603, "INTERNAL_ERROR")
            if response is not None:
                sink.write((canonical(response) + "\n").encode("utf-8"))
                sink.flush()
