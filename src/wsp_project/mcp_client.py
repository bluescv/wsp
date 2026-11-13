"""Synchronous MCP stdio client; bounded reads/timeouts, never auto-retries."""

import queue
import threading

from .errors import ControlError
from .manifest import canonical, parse_json
from .mcp_stdio import MAX_FRAME, VERSIONS


class StdioClient:
    def __init__(self, process, timeout=15):
        self.process = process
        self.timeout = timeout
        self.counter = 0
        self.closed = False
        self.messages = queue.Queue(maxsize=8)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while True:
                line = self.process.stdout.readline(MAX_FRAME + 1)
                if not line or len(line) > MAX_FRAME or not line.endswith(b"\n"):
                    self.messages.put_nowait(None)
                    return
                self.messages.put_nowait(parse_json(line.decode("utf-8")))
        except Exception:
            try:
                self.messages.put_nowait(None)
            except queue.Full:
                pass

    def notify(self, method, params=None):
        if self.closed:
            raise ControlError("TRANSPORT_CLOSED")
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def _send(self, message):
        data = (canonical(message) + "\n").encode("utf-8")
        if len(data) > MAX_FRAME:
            raise ControlError("REQUEST_TOO_LARGE")
        try:
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except (OSError, ValueError):
            self.close()
            raise ControlError("TRANSPORT_WRITE_FAILED") from None

    def request(self, method, params=None):
        if self.closed:
            raise ControlError("TRANSPORT_CLOSED")
        self.counter += 1
        self._send({"jsonrpc": "2.0", "id": self.counter, "method": method, "params": params or {}})
        try:
            message = self.messages.get(timeout=self.timeout)
        except queue.Empty:
            try:
                self.notify("notifications/cancelled", {"requestId": self.counter, "reason": "TIMEOUT"})
            finally:
                self.close()
            raise ControlError("TRANSPORT_TIMEOUT_UNCERTAIN") from None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or message.get("id") != self.counter:
            self.close()
            raise ControlError("TRANSPORT_RESPONSE_INVALID_OR_LOST")
        if "error" in message:
            raise ControlError("REMOTE_PROTOCOL_REJECTED")
        if "result" not in message:
            raise ControlError("TRANSPORT_RESPONSE_INVALID_OR_LOST")
        return message["result"]

    def initialize(self):
        result = self.request("initialize", {"protocolVersion": VERSIONS[0], "capabilities": {},
                                             "clientInfo": {"name": "wsp-tiny-smoke", "version": "0.1.0"}})
        if result.get("protocolVersion") not in VERSIONS or "tools" not in result.get("capabilities", {}):
            self.close()
            raise ControlError("MCP_NEGOTIATION_FAILED")
        self.notify("notifications/initialized")
        return result

    def tool(self, name, arguments):
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            self.process.wait(timeout=2)
        except Exception:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=2)
        finally:
            self.process.stdout.close()
            if self.process.stderr is not None:
                self.process.stderr.close()
            self.reader.join(timeout=1)
