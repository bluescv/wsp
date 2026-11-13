"""Operator-owned MCP launcher. Its configuration must contain no credentials."""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wsp_project.baselines import write_json
from wsp_project.codex_policy import environment
from wsp_project.editor_gateway import EditorGateway, RoundJournal, TOOLS
from wsp_project.errors import ControlError
from wsp_project.manifest import read_json, canonical
from wsp_project.mcp_client import StdioClient
from wsp_project.mcp_stdio import Server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    remote = None
    directory = None
    try:
        config = read_json(args.config)
        if set(config) != {'run_id', 'journal_directory', 'bridge_command', 'bridge_cwd'}:
            raise ControlError('GATEWAY_CONFIG_INVALID')
        directory = Path(config['journal_directory']).resolve()
        journal = RoundJournal(directory)
        command = config['bridge_command']
        if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
            raise ControlError('GATEWAY_CONFIG_INVALID')
        process = subprocess.Popen(command, cwd=config['bridge_cwd'], env=environment(),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        remote = StdioClient(process, timeout=150)
        remote.initialize()
        gateway = EditorGateway(remote, run_id=config['run_id'], record=journal)
        write_json(directory / 'gateway-ready.json', {'state': 'ready', 'remote_status': gateway.status})
        class AuditedServer(Server):
            def handle(self, message):
                response = super().handle(message)
                method = message.get('method') if isinstance(message, dict) else None
                allowed = {'initialize', 'notifications/initialized', 'tools/list', 'tools/call', 'ping'}
                result = (response or {}).get('result', {})
                row = {'method': method if method in allowed else 'OTHER',
                       'rpc_error_code': (response or {}).get('error', {}).get('code'),
                       'tool_error': result.get('isError', False)}
                with (directory / 'wire-audit.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(canonical(row) + '\n')
                return response
        return AuditedServer(gateway, tools=TOOLS, name='wsp-engineering-gateway',
                      instructions='Two engineering GPU read rounds only. No parameter writes or formal editing.').serve()
    except Exception as error:
        if directory is not None:
            write_json(directory / 'gateway-failure.json', {
                'error': error.code if isinstance(error, ControlError) else 'GATEWAY_FAILED'})
        return 2
    finally:
        if remote:
            remote.close()


if __name__ == '__main__':
    raise SystemExit(main())
