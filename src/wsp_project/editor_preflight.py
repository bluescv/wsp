"""Fail-closed engineering launch audit; never authorizes a formal editor."""
import hashlib
from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .codex_policy import environment, settings, toml_value
from .errors import ControlError
from .manifest import canonical, digest, read_json

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'configs/codex-bootstrap-review.json'
BASE = ROOT / 'prompts/editor_engineering_base.txt'
DEVELOPER = ROOT / 'prompts/editor_engineering_developer.txt'
AUDIT_PROMPT = 'SYNTHETIC_CONTEXT_AUDIT_ONLY'
SOURCE_FILES = tuple(sorted(
    [p.relative_to(ROOT).as_posix() for folder in ('src', 'scripts') for p in (ROOT / folder).rglob('*.py')]
    + ['configs/codex-bootstrap-review.json', 'prompts/editor_engineering_base.txt',
       'prompts/editor_engineering_developer.txt', 'scripts/probes/windows_acl_fixture.ps1']))


def require(value, code):
    if not value:
        raise ControlError(code)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def under(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def private_inventory(project, codex_home, profile, external):
    """Directory metadata only; never enumerate or read credential contents."""
    roots = [('project', project, 'operator_source_history_and_methods'),
             ('codex_home', codex_home, 'authentication_sessions_memory_and_plugins'),
             ('user_profile', profile, 'operator_credentials_apps_and_personal_files')]
    roots += [(f'external_{i}', path, 'declared_evaluator_updates_or_operator_storage') for i, path in enumerate(external)]
    return [{'id': name, 'path': str(Path(path).resolve()), 'category': category,
             'exists': Path(path).is_dir()} for name, path, category in roots]


def inventory_confirmation(path, roots):
    """Operator-owned record of a user's scope statement, never an ACL approval."""
    require(Path(path).is_file() and any(under(path, row['path']) for row in roots), 'INVENTORY_REVIEW_NOT_PRIVATE')
    review = read_json(path)
    require(isinstance(review, dict) and set(review) == {
        'schema_version', 'scope', 'confirmed_on', 'source', 'question', 'answer',
        'statement', 'declared_private_roots'}, 'INVENTORY_REVIEW_INVALID')
    require(review['schema_version'] == 1 and review['scope'] == 'local_project_storage_declared_by_user'
            and review['source'] == 'explicit_user_reply_in_current_chat'
            and review['statement'] == 'no_additional_local_project_storage'
            and review['answer'] == '没有' and isinstance(review['question'], str) and bool(review['question']),
            'INVENTORY_REVIEW_INVALID')
    try:
        require(date.fromisoformat(review['confirmed_on']).isoformat() == review['confirmed_on'], 'INVENTORY_REVIEW_INVALID')
        declared = review['declared_private_roots']
        require(isinstance(declared, list) and len(declared) == len(roots), 'INVENTORY_REVIEW_SCOPE_CHANGED')
        require(all(set(row) == {'id', 'path'} and Path(row['path']).is_absolute() for row in declared), 'INVENTORY_REVIEW_INVALID')
        require({(row['id'], Path(row['path']).resolve()) for row in declared}
                == {(row['id'], Path(row['path']).resolve()) for row in roots}, 'INVENTORY_REVIEW_SCOPE_CHANGED')
    except (ValueError, TypeError, KeyError):
        raise ControlError('INVENTORY_REVIEW_INVALID') from None
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def public_manifest(public, roots):
    # Reject aliases before resolve() hides them, including aliased ancestors.
    original = Path(public).absolute()
    for parent in (original, *original.parents):
        stat = parent.lstat()
        require(not parent.is_symlink() and not getattr(stat, 'st_file_attributes', 0) & 0x400, 'PUBLIC_REPARSE_POINT_DENIED')
    public = Path(public).resolve()
    require(public.is_dir(), 'PUBLIC_DIRECTORY_REQUIRED')
    for row in roots:
        require(not under(public, row['path']) and not under(row['path'], public), 'PUBLIC_PRIVATE_ROOT_OVERLAP')
    result = {}
    total = 0
    for path in sorted(public.rglob('*')):
        stat = path.lstat()
        require(not path.is_symlink() and not getattr(stat, 'st_file_attributes', 0) & 0x400, 'PUBLIC_REPARSE_POINT_DENIED')
        relative = path.relative_to(public).as_posix()
        require(all(p.lower() not in ('.git', '.codex', '.agents') for p in path.relative_to(public).parts), 'PUBLIC_AUTODISCOVERY_PATH_DENIED')
        lower = path.name.lower()
        require(not lower.startswith(('agents.', 'secrets', '.env', 'token', 'credentials'))
                and lower not in ('auth.json', 'config.toml'), 'PRIVATE_LIKE_PUBLIC_FILE_DENIED')
        if path.is_file():
            require(stat.st_nlink == 1 and under(path, public), 'PUBLIC_FILE_ALIAS_DENIED')
            total += stat.st_size
            require(len(result) < 64 and total <= 1024 * 1024, 'PUBLIC_BUNDLE_LIMIT')
            result[relative] = sha(path)
    require(bool(result), 'EMPTY_PUBLIC_BUNDLE')
    return result


def instructions():
    return BASE.read_text(encoding='utf-8').strip(), DEVELOPER.read_text(encoding='utf-8').strip()


def harden_context(overrides, inherited):
    # Inherited file-based prompts can reappear during compaction; fail before
    # reading their contents. Fixed base/developer instructions replace defaults.
    for key in ('experimental_compact_prompt_file', 'compact_prompt', 'model_catalog_json', 'instructions'):
        require(not inherited.get(key), 'UNREVIEWED_INHERITED_CONTEXT_CONFIG')
    value = dict(overrides)
    value['model_instructions_file'] = str(BASE)
    value['developer_instructions'] = instructions()[1]
    value['project_doc_fallback_filenames'] = []
    return value


def guarded_settings(configuration, inherited):
    boundary = read_json(configuration['boundary_manifest'])
    validate_boundary(boundary, configuration['public_directory'], configuration['codex_executable'])
    require(Path(configuration['pwsh_executable']).resolve() == Path(boundary['runtime_executables']['pwsh']['path']).resolve()
            and Path(configuration['runtime_directory']).resolve() == Path(configuration['pwsh_executable']).resolve().parent, 'RUNTIME_PATH_CHANGED')
    require(not inherited.get('permissions', {}).get('wsp_engineering'), 'RESERVED_PERMISSION_PROFILE_REQUIRES_REVIEW')
    gateway = configuration['gateway_command']
    require(isinstance(gateway, list) and len(gateway) in (6, 7)
            and Path(gateway[0]).resolve() == Path(sys.executable).resolve()
            and gateway[1:4] == ['-X', 'utf8', '-B'], 'GATEWAY_LAUNCH_NOT_REVIEWED')
    allowed_script = ROOT / 'scripts' / ('editor_gateway_mcp.py' if len(gateway) == 7 else 'windows_probe_mcp.py')
    require(Path(gateway[4]).resolve() == allowed_script.resolve()
            and (len(gateway) == 6 or gateway[5] == '--config'), 'GATEWAY_LAUNCH_NOT_REVIEWED')
    require(Path(gateway[-1]).is_file() and not under(gateway[-1], configuration['public_directory'])
            and any(under(gateway[-1], row['path']) for row in boundary['private_roots']), 'GATEWAY_CONFIG_OUTSIDE_PRIVATE_ROOTS')
    require(Path(gateway[-1]).resolve() == Path(boundary['gateway_config']['path']).resolve(), 'GATEWAY_CONFIG_CHANGED')
    for key in ('boundary_manifest', 'run_directory', 'gateway_journal_directory'):
        require(not under(configuration[key], configuration['public_directory'])
                and any(under(configuration[key], row['path']) for row in boundary['private_roots']), 'OPERATOR_PATH_OUTSIDE_PRIVATE_ROOTS')
    denied = sorted(set(configuration['denied_roots'] + [row['path'] for row in boundary['private_roots']]))
    return harden_context(settings(inherited, public_root=configuration['public_directory'], denied_roots=denied,
        runtime_root=configuration['runtime_directory'], gateway_command=configuration['gateway_command'],
        sandbox=configuration['sandbox']), inherited)


def effective_profile_guard(config, overrides):
    profile = config.get('permissions', {}).get('wsp_engineering', {})
    filesystem = dict(profile.get('filesystem') or {})
    require(filesystem.pop('glob_scan_max_depth', None) is None, 'UNREVIEWED_FILESYSTEM_GLOB_POLICY')
    require(filesystem == overrides['permissions.wsp_engineering.filesystem']
            and profile.get('extends') is None and not profile.get('workspace_roots')
            and profile.get('network', {}).get('enabled') is False, 'EFFECTIVE_FILESYSTEM_POLICY_CHANGED')
    return {'passed': True, 'private_deny_rules_present': True, 'root_read_grant_added': False}


def freeze_boundary(public, codex, *, pwsh, gateway_config, external=(), inventory_review=None):
    require(os.name == 'nt', 'WINDOWS_NATIVE_CLIENT_REQUIRED')
    home = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex')).resolve()
    profile = Path(os.environ['USERPROFILE']).resolve()
    roots = private_inventory(ROOT, home, profile, external)
    require(all(row['exists'] for row in roots), 'DECLARED_PRIVATE_ROOT_MISSING')
    # Optional AppData redirections outside the user profile must be explicit.
    for key in ('APPDATA', 'LOCALAPPDATA'):
        if os.environ.get(key):
            path = Path(os.environ[key]).resolve()
            if not any(under(path, row['path']) for row in roots):
                roots.append({'id': key.lower(), 'path': str(path), 'category': 'redirected_app_credentials', 'exists': path.is_dir()})
    require(Path(gateway_config).is_file() and any(under(gateway_config, row['path']) for row in roots), 'GATEWAY_CONFIG_OUTSIDE_PRIVATE_ROOTS')
    confirmation = inventory_confirmation(inventory_review, roots) if inventory_review is not None else None
    return {'schema_version': 1, 'scope': 'engineering_only', 'formal_run_authorized': False,
            'public_root': str(Path(public).resolve()), 'private_roots': roots,
            'public_files': public_manifest(public, roots), 'codex_binary_sha256': sha(codex),
            'runtime_executables': {name: {'path': str(Path(path).resolve()), 'sha256': sha(path)}
                                    for name, path in (('python', sys.executable), ('pwsh', pwsh))},
            'gateway_config': {'path': str(Path(gateway_config).resolve()), 'sha256': sha(gateway_config)},
            'controller_source_sha256': {name: sha(ROOT / name) for name in SOURCE_FILES},
            'inventory_scope': 'project_and_operator_profile_plus_declared_external_roots',
            'inventory_review': confirmation,
            'all_external_private_locations_human_confirmed': confirmation is not None,
            'actual_private_roots_os_acl_verified': False}


def validate_boundary(value, public, codex):
    require(value.get('schema_version') == 1 and value.get('scope') == 'engineering_only'
            and value.get('formal_run_authorized') is False, 'BOUNDARY_SCOPE_INVALID')
    require(Path(value['public_root']).resolve() == Path(public).resolve(), 'BOUNDARY_PUBLIC_ROOT_MISMATCH')
    roots = value['private_roots']
    require(isinstance(roots, list) and len(roots) <= 32, 'BOUNDARY_ROOTS_INVALID')
    by_id = {row['id']: row for row in roots}
    require(len(by_id) == len(roots), 'BOUNDARY_ROOTS_INVALID')
    mandatory = {'project': ROOT, 'codex_home': Path(os.environ.get('CODEX_HOME', Path.home() / '.codex')),
                 'user_profile': Path(os.environ.get('USERPROFILE', str(Path.home())))}
    for name, path in mandatory.items():
        require(name in by_id and Path(by_id[name]['path']).resolve() == path.resolve(), 'REQUIRED_PRIVATE_ROOT_MISSING')
    require(all(Path(row['path']).is_absolute() and Path(row['path']).is_dir() and row['exists'] is True for row in roots), 'PRIVATE_ROOT_STATE_CHANGED')
    for key in ('APPDATA', 'LOCALAPPDATA'):
        require(not os.environ.get(key) or any(under(os.environ[key], row['path']) for row in roots), 'APPDATA_ROOT_NOT_COVERED')
    confirmation = value.get('inventory_review')
    if confirmation is not None:
        require(isinstance(confirmation, dict) and set(confirmation) == {'path', 'sha256'}, 'INVENTORY_REVIEW_INVALID')
        require(sha(confirmation['path']) == confirmation['sha256'], 'INVENTORY_REVIEW_CHANGED')
        require(inventory_confirmation(confirmation['path'], roots) == confirmation, 'INVENTORY_REVIEW_CHANGED')
    require(value.get('all_external_private_locations_human_confirmed') is (confirmation is not None)
            and value.get('actual_private_roots_os_acl_verified') is False, 'BOUNDARY_ATTESTATION_MISMATCH')
    require(value['public_files'] == public_manifest(public, roots), 'PUBLIC_BUNDLE_CHANGED')
    gateway = value['gateway_config']
    require(any(under(gateway['path'], row['path']) for row in roots)
            and sha(gateway['path']) == gateway['sha256'], 'GATEWAY_CONFIG_CHANGED')
    require(value['codex_binary_sha256'] == sha(codex), 'CODEX_BINARY_CHANGED')
    runtimes = value['runtime_executables']
    require(set(runtimes) == {'python', 'pwsh'}
            and Path(runtimes['python']['path']).resolve() == Path(sys.executable).resolve()
            and all(sha(row['path']) == row['sha256'] for row in runtimes.values()), 'RUNTIME_BINARY_CHANGED')
    require(set(value['controller_source_sha256']) == set(SOURCE_FILES), 'SOURCE_MANIFEST_INCOMPLETE')
    require(all(sha(ROOT / p) == h for p, h in value['controller_source_sha256'].items()), 'CONTROLLER_SOURCE_CHANGED')
    return {'manifest_sha256': digest(value), 'private_root_ids': list(by_id),
            'public_file_count': len(value['public_files']), 'scope_complete_for_declared_roots': True,
            'all_external_private_locations_human_confirmed': confirmation is not None,
            'inventory_confirmation_scope': 'user_reported_local_project_storage' if confirmation else None,
            'actual_private_roots_os_acl_verified': False}


def audit_rendered_context(data, review, *, base, developer, global_agents):
    require(isinstance(data, list), 'BOOTSTRAP_SHAPE_CHANGED')
    expected = {'model.base_instructions': ('developer', base),
                'generic.developer_instructions': ('developer', developer),
                'agents_md.instructions': ('user', '# AGENTS.md instructions\n\n<INSTRUCTIONS>\n' + global_agents + '\n</INSTRUCTIONS>'),
                'user.text': ('user', AUDIT_PROMPT)}
    seen, rows = set(), []
    for item in data:
        if item.get('type') == 'additional_tools':
            require('tools' not in seen and item.get('role') == 'developer' and digest(item.get('tools')) == review['tools_sha256'], 'BOOTSTRAP_TOOL_DECLARATIONS_CHANGED')
            seen.add('tools')
            continue
        require(item.get('type') == 'message', 'UNREVIEWED_BOOTSTRAP_ITEM')
        kinds = item.get('internal_chat_message_metadata_passthrough', {}).get('content_item_kinds', [])
        content = item.get('content', [])
        require(bool(kinds) and len(kinds) == len(content), 'UNCLASSIFIED_BOOTSTRAP_CONTENT')
        for kind, part in zip(kinds, content):
            require(kind not in seen and part.get('type') == 'input_text', 'DUPLICATE_OR_NON_TEXT_BOOTSTRAP')
            text = part.get('text', '')
            if kind in expected:
                role, exact = expected[kind]
                require(item.get('role') == role and text.strip() == exact, 'BOOTSTRAP_TEXT_MISMATCH')
            else:
                require(kind in review['native_text_sha256'] and item.get('role') == 'developer'
                        and text_sha(text) == review['native_text_sha256'][kind], 'UNREVIEWED_BOOTSTRAP_TEXT')
            seen.add(kind)
            rows.append({'kind': kind, 'sha256': text_sha(text)})
    require(seen == set(expected) | set(review['native_text_sha256']) | {'tools'}, 'BOOTSTRAP_COMPONENT_MISSING')
    return {'passed': True, 'components': rows, 'tools_sha256': review['tools_sha256'],
            'global_instructions_retained': True, 'unexpected_context_blocks': False,
            'memory_or_skills_or_environment_context_present': False}


def context_audit(codex, overrides, public, *, base=None, developer=None):
    review = read_json(REVIEW)
    require(sha(codex) == review['binary_sha256'], 'BOOTSTRAP_BINARY_NOT_REVIEWED')
    home = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex'))
    require(not (home / 'AGENTS.override.md').exists(), 'UNREVIEWED_GLOBAL_AGENTS_OVERRIDE')
    global_agents = (home / 'AGENTS.md').read_text(encoding='utf-8').strip()
    require(text_sha(global_agents) == review['global_agents_sha256'], 'GLOBAL_INSTRUCTIONS_CHANGED_REVIEW_REQUIRED')
    no_gateway = dict(overrides)
    for name in list(no_gateway):
        if name.startswith('mcp_servers.') and name.endswith(('.enabled', '.required')):
            no_gateway[name] = False
    args = [str(codex)]
    for key, value in no_gateway.items():
        args += ['-c', key + '=' + toml_value(value)]
    args += ['debug', 'prompt-input', AUDIT_PROMPT]
    result = subprocess.run(args, cwd=public, env=environment(), capture_output=True, timeout=45,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0, 'BOOTSTRAP_RENDER_FAILED')
    report = audit_rendered_context(json.loads(result.stdout), review, base=instructions()[0] if base is None else base,
                                    developer=instructions()[1] if developer is None else developer, global_agents=global_agents)
    report['review_sha256'] = sha(REVIEW)
    return report


def thread_context_guard(started, public):
    home = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex'))
    sources = started.get('instructionSources')
    require(isinstance(sources, list) and len(sources) == 1
            and Path(sources[0]).resolve() == (home / 'AGENTS.md').resolve(), 'THREAD_INSTRUCTION_SOURCES_CHANGED')
    require(started.get('model') == 'gpt-6-astra' and started.get('modelProvider') == 'openai'
            and started.get('reasoningEffort') == 'max', 'THREAD_MODEL_CHANGED')
    require(Path(started.get('thread', {}).get('cwd', '')).resolve() == Path(public).resolve()
            and started.get('approvalPolicy') == 'never'
            and started.get('activePermissionProfile') == {'id': 'wsp_engineering', 'extends': None}, 'THREAD_PERMISSION_CONTEXT_CHANGED')
    # Recheck the content at the thread boundary; do not trust only a file path.
    review = read_json(REVIEW)
    require(text_sha((home / 'AGENTS.md').read_text(encoding='utf-8').strip()) == review['global_agents_sha256'], 'GLOBAL_INSTRUCTIONS_CHANGED_REVIEW_REQUIRED')
    return {'passed': True, 'instruction_source_count': 1, 'source_kind': 'reviewed_global_agents'}


def startup_preflight(configuration, overrides):
    public, codex = Path(configuration['public_directory']), configuration['codex_executable']
    boundary = read_json(configuration['boundary_manifest'])
    report = {'formal_run_authorized': False, 'model_requests': 0}
    report['boundary'] = validate_boundary(boundary, public, codex)
    report['bootstrap'] = context_audit(codex, overrides, public)
    run_id = 'startup-os-' + uuid.uuid4().hex
    process = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(ROOT / 'scripts/check_windows_os_isolation.py'),
        '--codex', str(codex), '--pwsh', configuration['pwsh_executable'], '--run-id', run_id],
        cwd=ROOT, env=environment(), capture_output=True, timeout=120,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(process.returncode == 0, 'STARTUP_OS_DENIAL_CHECK_FAILED')
    evidence = read_json(ROOT / 'runs' / run_id / 'observation.json')
    require(evidence.get('native_exact_directory_denial_passed') is True
            and evidence.get('formal_run_authorized') is False
            and evidence.get('codex_binary_sha256') == sha(codex), 'STARTUP_OS_EVIDENCE_INVALID')
    require(all(sha(ROOT / p) == h for p, h in evidence['source_sha256'].items()), 'STARTUP_OS_SOURCE_CHANGED')
    report['os_denial'] = {'passed': True, 'run_id': run_id, 'evidence_sha256': digest(evidence),
                           'scope': 'fresh_synthetic_exact_directory_denial', 'actual_private_roots_os_acl_verified': False}
    acl_run = 'startup-acl-' + uuid.uuid4().hex
    restored = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(ROOT / 'scripts/check_windows_acl_restore.py'),
        '--codex', str(codex), '--pwsh', configuration['pwsh_executable'], '--run-id', acl_run],
        cwd=ROOT, env=environment(), capture_output=True, timeout=90,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(restored.returncode == 0, 'STARTUP_PROTECTED_ACL_CHECK_FAILED')
    acl_evidence = read_json(ROOT / 'runs' / acl_run / 'observation.json')
    require(acl_evidence.get('passed') is True
            and acl_evidence.get('explicit_protected_descendant_denial_passed') is True
            and acl_evidence.get('restoration_passed') is True
            and acl_evidence.get('actual_private_roots_os_acl_verified') is False
            and acl_evidence.get('formal_run_authorized') is False
            and acl_evidence.get('codex_binary_sha256') == sha(codex), 'STARTUP_PROTECTED_ACL_EVIDENCE_INVALID')
    require(all(sha(ROOT / p) == h for p, h in acl_evidence['source_sha256'].items()), 'STARTUP_OS_SOURCE_CHANGED')
    report['protected_directory_probe'] = {'run_id': acl_run, 'evidence_sha256': digest(acl_evidence),
        'parent_only_denial_passed': acl_evidence['parent_only_protected_descendant_denial_passed'],
        'explicit_protected_denial_passed': True, 'synthetic_acl_restoration_passed': True,
        'scope': 'synthetic_known_protected_descendant_only', 'actual_private_roots_os_acl_verified': False}
    # All mutable inputs are checked again after the native child has exited.
    validate_boundary(boundary, public, codex)
    report['bootstrap_after_os_probe'] = context_audit(codex, overrides, public)
    report['engineering_preflight_passed'] = True
    report['formal_preflight_passed'] = False
    return report
