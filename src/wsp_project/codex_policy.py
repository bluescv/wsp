"""Process-local Codex settings for a public, read-only engineering session.

Authentication stays in the existing Windows CODEX_HOME. This is a capability
policy, not proof that the installed native filesystem sandbox works.
"""
import json
import os
import re
import subprocess
from pathlib import Path

from .errors import ControlError


DISABLED = (
    'code_mode', 'code_mode_only', 'shell_tool',
    'unified_exec_tty', 'shell_snapshot', 'view_image', 'image_generation',
    'multi_agent', 'multi_agent_v2', 'apps', 'plugins', 'remote_plugin',
    'browser_use', 'browser_use_external', 'browser_use_full_cdp_access',
    'in_app_browser', 'computer_use', 'skill_search', 'skill_mcp_dependency_install',
    'sleep_tool', 'tool_suggest', 'goals', 'hooks', 'workspace_dependencies',
    'recommended_plugins', 'memories', 'plugin_discovery_prefetch',
    'request_permissions_tool', 'unbounded_connection_retries',
)
PROFILE = 'wsp_engineering'
SERVER = 'wsp_engineering'


def load_inherited_config(home):
    """Read locally without emitting config values; requires native Python 3.11+."""
    try:
        import tomllib
        return tomllib.loads((Path(home) / 'config.toml').read_text(encoding='utf-8'))
    except Exception:
        raise ControlError('CODEX_CONFIG_READ_FAILED') from None


def settings(inherited, *, public_root, denied_roots, runtime_root, gateway_command,
             sandbox='elevated'):
    if sandbox not in ('mxc', 'elevated'):
        raise ControlError('CODEX_SANDBOX_REQUIRED')
    if not gateway_command or not all(isinstance(x, str) and x for x in gateway_command):
        raise ControlError('INVALID_GATEWAY_COMMAND')
    if inherited.get('model_providers', {}).get('openai'):
        raise ControlError('CUSTOM_OPENAI_PROVIDER_REQUIRES_REVIEW')
    overrides = {f'features.{key}': False for key in DISABLED}
    overrides.update({
        'features.code_mode_host': True,
        # This flag controls resumable execution, not permission to run commands.
        # User config cannot disable it; shell_tool and empty environments gate IO.
        'features.unified_exec': True,
        'model_provider': 'openai', 'model': 'gpt-6-astra', 'model_reasoning_effort': 'max',
        'forced_login_method': 'chatgpt', 'approval_policy': 'never',
        'sandbox_mode': 'read-only', 'windows.sandbox': sandbox,
        'project_doc_max_bytes': 0, 'skills.include_instructions': False,
        'skills.bundled.enabled': False, 'agents.enabled': False,
        'web_search': 'disabled', 'apps._default.enabled': False,
        'tools.update_plan.enabled': False, 'tools.experimental_request_user_input.enabled': False,
        'developer_instructions': '', 'include_apps_instructions': False,
        'include_environment_context': False, 'include_environment_context_time': False,
        'analytics.enabled': False, 'otel.exporter': 'none', 'otel.metrics_exporter': 'none',
    })
    for name in inherited.get('mcp_servers', {}):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name == SERVER:
            raise ControlError('MCP_CONFIG_IDENTIFIER_REQUIRES_REVIEW')
        overrides[f'mcp_servers.{name}.enabled'] = False
    filesystem = {':minimal': 'read', str(public_root): 'read', str(runtime_root): 'read'}
    for path in denied_roots:
        if str(path) in filesystem:
            raise ControlError('CONFLICTING_FILESYSTEM_POLICY')
        filesystem[str(path)] = 'deny'
    overrides[f'permissions.{PROFILE}.filesystem'] = filesystem
    overrides[f'permissions.{PROFILE}.network.enabled'] = False
    prefix = f'mcp_servers.{SERVER}.'
    overrides.update({prefix + 'command': gateway_command[0], prefix + 'args': gateway_command[1:],
                      prefix + 'enabled': True, prefix + 'required': True,
                      prefix + 'startup_timeout_sec': 180, prefix + 'tool_timeout_sec': 150,
                      prefix + 'enabled_tools': ['wsp_roundtrip']})
    return overrides


def validate_effective(config, *, allowed_tools=None):
    """Validate config/read, not runtime tool registration or OS enforcement.

    Runtime features and the actual tool inventory require separate checks.
    In particular unified_exec is not the command authorization switch.
    """
    if any(config.get('features', {}).get(key) is not False for key in DISABLED):
        raise ControlError('CODEX_FEATURE_POLICY_MISMATCH')
    if config.get('features', {}).get('code_mode_host') is not True:
        raise ControlError('CODEX_TOOL_ROUTER_REQUIRED')
    if config.get('features', {}).get('unified_exec') is not True:
        raise ControlError('CODEX_EXECUTION_MODE_REQUIRES_REVIEW')
    for key, expected in (('model', 'gpt-6-astra'), ('model_provider', 'openai'),
                          ('forced_login_method', 'chatgpt'), ('approval_policy', 'never'),
                          ('sandbox_mode', 'read-only'), ('web_search', 'disabled')):
        if config.get(key) != expected:
            raise ControlError('CODEX_EFFECTIVE_POLICY_MISMATCH')
    enabled = {name for name, value in config.get('mcp_servers', {}).items() if value.get('enabled', True)}
    if enabled != {SERVER}:
        raise ControlError('CODEX_MCP_POLICY_MISMATCH')
    gateway = config['mcp_servers'][SERVER]
    if gateway.get('enabled_tools') != (['wsp_roundtrip'] if allowed_tools is None else allowed_tools) or gateway.get('required') is not True:
        raise ControlError('CODEX_MCP_POLICY_MISMATCH')
    return {'disabled_features': list(DISABLED), 'tool_orchestration_host': True, 'enabled_mcp_servers': [SERVER],
            'formal_run_authorized': False}


def isolated_thread_params(public_root):
    """Select no execution environment, even if a host has a default executor."""
    return {'model': 'gpt-6-astra', 'modelProvider': 'openai', 'allowProviderModelFallback': False,
            'cwd': str(public_root), 'permissions': PROFILE, 'approvalPolicy': 'never',
            'ephemeral': True, 'environments': [], 'runtimeWorkspaceRoots': [str(public_root)]}


def isolated_turn_params(thread_id, text, *, output_schema=None):
    value = {'threadId': thread_id, 'model': 'gpt-6-astra', 'effort': 'max', 'environments': [],
             'permissions': PROFILE, 'approvalPolicy': 'never',
             'input': [{'type': 'text', 'text': text, 'text_elements': []}]}
    if output_schema is not None:
        value['outputSchema'] = output_schema
    return value


def runtime_features(executable, overrides, *, cwd, env):
    """Check computed feature flags without printing config or diagnostics."""
    args = [str(executable), 'features', 'list']
    for key, value in overrides.items():
        args += ['-c', key + '=' + toml_value(value)]
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, timeout=45,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise ControlError('CODEX_RUNTIME_FEATURE_CHECK_FAILED')
    flags = {}
    for line in result.stdout.decode('utf-8').splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in set(DISABLED) | {'unified_exec', 'code_mode_host'} and parts[-1] in ('true', 'false'):
            flags[parts[0]] = parts[-1] == 'true'
    if (flags.get('unified_exec') is not True or flags.get('code_mode_host') is not True
            or flags.get('shell_tool') is not False
            or any(flags.get(key) is True for key in DISABLED)):
        raise ControlError('CODEX_RUNTIME_FEATURE_MISMATCH')
    return flags


def toml_value(value):
    if isinstance(value, dict):
        return '{' + ','.join(json.dumps(k) + '=' + toml_value(v) for k, v in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ','.join(toml_value(v) for v in value) + ']'
    return json.dumps(value)


def command(executable, overrides):
    args = [str(executable), 'app-server', '--strict-config', '--listen', 'stdio://']
    for key, value in overrides.items():
        args += ['-c', key + '=' + toml_value(value)]
    return args


def environment(source=None):
    source = os.environ if source is None else source
    names = {'PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA',
             'TEMP', 'TMP', 'PROGRAMDATA', 'PROGRAMFILES', 'PROGRAMFILES(X86)', 'PATHEXT',
             'ALLUSERSPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'CODEX_HOME',
             'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY'}
    return {key: value for key, value in source.items() if key.upper() in names}
