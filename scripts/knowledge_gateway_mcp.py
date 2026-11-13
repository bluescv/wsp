"""Credential-free Windows gateway for the bounded post-submission POC.

Parameterized variant: the cross-layer registry is loaded
from the config's explicit ``registry_file`` path and passed explicitly to
every is_cross_layer/tools_for_policy/verify_policy_handshake call; the
remote store binds the same registry at its own launch (qwen_mcp.py
threads it through create/bind), so dispatch checks on both sides use the
identical instance data.
"""
import argparse, os, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from wsp_project.baselines import write_json
from wsp_project.codex_policy import environment
from wsp_project.errors import ControlError
from wsp_project.manifest import read_json,canonical,digest
from wsp_project.mcp_client import StdioClient
from wsp_project.mcp_stdio import Server
from wsp_project.knowledge_poc import tools_for_policy,verify_policy_handshake, SCOPE, verify_submission,check_reviewed_resume
from wsp_project.cross_layer_policy import is_cross_layer, load_registry, GATEWAY_TIMEOUT_SECONDS

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    cfg=read_json(parser.parse_args().config); remote=None
    registry=None
    registry_file=cfg.get('registry_file')
    if registry_file:
        registry=load_registry(Path(registry_file))
    journal=Path(cfg['journal_directory']); journal.mkdir(parents=True,exist_ok=True)
    try:
        if (journal/'gateway-ready.json').exists(): raise ControlError('NO_AUTOMATIC_GATEWAY_RESTART')
        process=subprocess.Popen(cfg['bridge_command'],cwd=cfg['bridge_cwd'],env=environment(),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        policy=cfg.get('write_policy')
        remote=StdioClient(process,timeout=GATEWAY_TIMEOUT_SECONDS if is_cross_layer(policy,registry) else 150)
        remote.initialize()
        status=remote.tool('qwen_status',{})['structuredContent']
        toolset=tools_for_policy(policy,registry=registry)
        verify_policy_handshake(cfg,status,remote.request('tools/list')['tools'],registry=registry)
        if status['qwen']['scope']!=SCOPE or status['state']!='running':
            raise ControlError('FRESH_KNOWLEDGE_POC_REQUIRED')
        if cfg.get('reviewed_resume') is not None:
            check_reviewed_resume(status,cfg['reviewed_resume'])
        elif status['active_candidate']!='base':
            raise ControlError('FRESH_KNOWLEDGE_POC_REQUIRED')
        write_json(journal/'gateway-ready.json',status)
        class Gateway:
            def call(self,name,args):
                with (journal/'calls.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(canonical({'event':'dispatch','tool':name,'arguments':args})+'\n'); stream.flush(); os.fsync(stream.fileno())
                response=remote.tool(name,args)
                with (journal/'calls.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(canonical({'event':'response','tool':name,'response':response})+'\n'); stream.flush(); os.fsync(stream.fileno())
                return response
        server=Server(None,tools=toolset,name='wsp-knowledge-gateway',
            instructions='Fixed bounded gateway over the sealed POC toolset. No credentials, paths or arbitrary tools.')
        while True:
            line=sys.stdin.readline()
            if not line: return 0
            import json as _json
            try: message=_json.loads(line)
            except ValueError: message=None
            response=server.handle(message) if isinstance(message,dict) else None
            if response is None:
                if isinstance(message,dict) and message.get('method')=='tools/call' and isinstance(message.get('params'),dict):
                    params=message['params']
                    result=Gateway().call(params.get('name'),params.get('arguments',{}))
                    response={'jsonrpc':'2.0','id':message.get('id'),'result':{'content':[{'type':'text','text':canonical(result)}],'structuredContent':result,'isError':False}}
                else:
                    response={'jsonrpc':'2.0','id':message.get('id') if isinstance(message,dict) else None,'error':{'code':-32601,'message':'METHOD_NOT_FOUND'}}
            sys.stdout.write(canonical(response)+'\n'); sys.stdout.flush()
    except Exception as error:
        code=error.code if isinstance(error,ControlError) else 'GATEWAY_STARTUP_FAILED'
        print(canonical({'error':code}),file=sys.stderr)
        return 2
    finally:
        if remote is not None: remote.close()

if __name__=='__main__': raise SystemExit(main())
