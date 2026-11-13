"""Assess exact-directory denial separately from a default-deny read allowlist."""
PATHS = ('public', 'absolute', 'nested', 'relative', 'case', 'extended', 'junction', 'hardlink', 'alternate_stream')
LATE_PATHS = ('late_file', 'late_nested_file')


def scan_passed(scan, *, deny, late):
    expected = set(PATHS + (LATE_PATHS if late else ()))
    paths = scan.get('paths', {})
    if set(paths) != expected:
        return False
    for label, results in paths.items():
        if set(results) != {'python', 'win32'}:
            return False
        denied = deny and label != 'public'
        if results['win32'] != {'read': not denied, 'denied': denied, 'winerror': 5 if denied else 0}:
            return False
        py = results['python']
        if (set(py) != {'read', 'denied', 'winerror', 'errno'} or py.get('read') is not (not denied)
                or py.get('denied') is not denied or py.get('errno') != (13 if denied else 0)
                or py.get('winerror') not in ((None, 5) if denied else (0,))):
            return False
    directory = scan.get('directory')
    if directory != {'listed': not deny, 'denied': deny, 'winerror': 5 if deny else 0}:
        return False
    token = scan.get('token', {})
    return (token.get('elevated') is False and token.get('restricted') is True
            and token.get('different_user_from_controller') is True)


def assess(report):
    checks = {}
    for label, deny in (('broad_read_control', False), ('exact_directory_deny', True)):
        row = report.get('profiles', {}).get(label, {})
        output = row.get('result', {})
        checks[label] = (row.get('exit_code') == 0 and row.get('ready_seen') is True
                         and not any(row.get(k) for k in ('timeout', 'unrecognized_output', 'child_probe_failed'))
                         and all(scan_passed(output.get(phase, {}), deny=deny, late=phase != 'before')
                                 for phase in ('before', 'after', 'descendant')))
    rpc = report.get('app_server_exact_directory_deny', {})
    checks['app_server'] = rpc.get('exit_code') == 0 and scan_passed(rpc.get('result', {}), deny=True, late=True)
    checks['host_positive_control'] = report.get('host_fixture_integrity_passed') is True
    checks['original_editor_policy_unchanged'] = report.get('editor_policy_unchanged') is True
    checks['user_config_unchanged'] = report.get('user_config_unchanged') is True
    return {'checks': checks, 'native_exact_directory_denial_passed': all(checks.values()),
            'native_read_allowlist_passed': False, 'overall_editor_isolation_passed': False,
            'formal_run_authorized': False}
