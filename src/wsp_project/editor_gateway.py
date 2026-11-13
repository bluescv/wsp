"""Two public subscription smoke rounds over real remote Qwen reads only."""
import secrets
import os
import threading
from pathlib import Path
from .errors import ControlError
from .manifest import canonical, digest, identifier
from .tiny import schema,validate_arguments

TOOLS=[{'name':'wsp_roundtrip','description':'Engineering-only Qwen GPU read. Round 1 takes an empty previous_nonce; round 2 must return round 1 nonce. No arbitrary paths, commands, writes or prompts.',
    'annotations':{'readOnlyHint':True,'destructiveHint':False,'idempotentHint':False,'openWorldHint':False},
    'inputSchema':schema({'round':{'type':'integer','minimum':1,'maximum':2},
                          'previous_nonce':{'type':'string','maxLength':64}})}]


class RoundJournal:
    """Exclusive durable dispatch records; restarts never silently replay reads."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.glob('round-*-dispatch.json')):
            raise ControlError('GATEWAY_PRIOR_DISPATCH_REQUIRES_REVIEW')

    def __call__(self, value):
        number, event = value['round'], value['event']
        if type(number) is not int or number not in (1, 2) or event not in ('dispatch', 'complete'):
            raise ControlError('INVALID_GATEWAY_JOURNAL_EVENT')
        path = self.directory / f'round-{number}-{event}.json'
        try:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(canonical(value) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            raise ControlError('GATEWAY_JOURNAL_WRITE_FAILED') from None


class EditorGateway:
    def __init__(self, remote, *, run_id, record=lambda value:None):
        self.remote=remote
        self.run_id=identifier(run_id)
        self.lock=threading.Lock()
        self.record=record
        self.round=0
        self.previous=''
        self.receipts=[]
        self.failed=False
        self.status=self._tool('qwen_status',{})
        if self.status['phase']!='qwen_toolchain' or self.status['state']!='running' or self.status['active_candidate']!='base' or self.status['qwen']['formal_run_authorized'] is not False:
            raise ControlError('ENGINEERING_BASE_REQUIRED')
    def _tool(self,name,args):
        response=self.remote.tool(name,args)
        if response.get('isError') or 'structuredContent' not in response:
            raise ControlError('REMOTE_QWEN_REJECTED')
        return response['structuredContent']
    def call(self,name,args):
        with self.lock:
            return self._call(name,args)
    def _call(self,name,args):
        if name!='wsp_roundtrip': raise ControlError('GATEWAY_TOOL_DENIED')
        validate_arguments(args,TOOLS[0]['inputSchema'])
        if self.failed: raise ControlError('GATEWAY_UNCERTAIN')
        if args['round']!=self.round+1 or args['previous_nonce']!=self.previous:
            raise ControlError('ROUND_CHAIN_MISMATCH')
        if self.round>=2: raise ControlError('ROUND_BUDGET_EXHAUSTED')
        request_id=f'{self.run_id}-round-{args["round"]}'
        self.failed=True  # Never resend after an interrupted remote dispatch.
        self.record({'event':'dispatch','round':args['round'],'request_id':request_id})
        row=self._tool('qwen_read_slice',dict(request_id=request_id,expected_candidate='base',
            tensor='model.layers.35.mlp.down_proj.weight',offset=(args['round']-1)*4,count=4))
        if row.get('gpu_verified') is not True or row['effective_weights_sha256']!=self.status['qwen']['active_weights_sha256']:
            raise ControlError('GATEWAY_GPU_MISMATCH')
        # This is a synthetic public challenge, not an authentication token.
        nonce=secrets.token_hex(16)
        result=dict(round=args['round'],nonce=nonce,previous_nonce=args['previous_nonce'],
            qwen_receipt_sha256=digest(row),effective_weights_sha256=row['effective_weights_sha256'],
            gpu_verified=True,values=row['values'],formal_run_authorized=False)
        self.record({'event':'complete','round':args['round'],'request_id':request_id,'result':result,'remote_receipt':row})
        self.receipts.append(result)
        self.round=args['round']
        self.previous=nonce
        self.failed=False
        return result
