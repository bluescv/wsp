"""Bounded Windows-native app-server stdio; never print raw diagnostics/config."""
import json
import queue
import subprocess
import threading
import time

from .errors import ControlError
from .manifest import canonical,parse_json


class AppServer:
    def __init__(self, command, *, cwd, env):
        self.process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.events=queue.Queue(maxsize=4096)
        self.counter=0
        self.notifications=[]
        self.reader=threading.Thread(target=self._read,daemon=True)
        self.reader.start()
    def _read(self):
        try:
            while True:
                line=self.process.stdout.readline(4*1024*1024+1)
                if not line or len(line)>4*1024*1024 or not line.endswith(b'\n'): break
                event = parse_json(line.decode('utf-8'))
                if not isinstance(event, dict): break
                self.events.put_nowait(event)
        except Exception: pass
        try: self.events.put_nowait(None)
        except queue.Full: pass
    def send(self, value):
        try:
            self.process.stdin.write((canonical(value)+'\n').encode())
            self.process.stdin.flush()
        except (OSError,ValueError): raise ControlError('CODEX_TRANSPORT_CLOSED') from None
    def next(self, timeout=30):
        try: result=self.events.get(timeout=timeout)
        except queue.Empty: raise ControlError('CODEX_TIMEOUT_UNCERTAIN') from None
        if result is None: raise ControlError('CODEX_TRANSPORT_CLOSED')
        return result
    def request(self, method, params=None, *, timeout=45):
        self.counter+=1
        request_id=self.counter
        self.send(dict(id=request_id,method=method,params=params or {}))
        deadline=time.monotonic()+timeout
        while True:
            event=self.next(max(0.01,deadline-time.monotonic()))
            if event.get('id')==request_id and 'method' not in event:
                if 'error' in event:
                    error=ControlError('CODEX_RPC_REJECTED')
                    error.rpc_code=event['error'].get('code')
                    # Kept only in caller memory for classification, not emitted.
                    error.rpc_message=event['error'].get('message','')
                    raise error
                return event['result']
            if 'method' in event and 'id' in event:
                self.send(dict(id=event['id'],error=dict(code=-32601,message='CLIENT_REQUEST_DENIED')))
            if len(self.notifications) >= 4096:
                raise ControlError('CODEX_NOTIFICATION_LIMIT')
            self.notifications.append(event)
            if time.monotonic()>=deadline: raise ControlError('CODEX_TIMEOUT_UNCERTAIN')
    def initialize(self):
        result=self.request('initialize',dict(clientInfo=dict(name='wsp_engineering',version='0.1.0'),capabilities=dict(experimentalApi=True)))
        self.send(dict(method='initialized'))
        return result
    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=3)
            except Exception:
                self.process.terminate()
                try: self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        self.process.stdout.close()
        self.reader.join(timeout=1)
