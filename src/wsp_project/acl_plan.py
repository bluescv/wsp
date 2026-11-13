"""Produce a reviewable ACL experiment plan; there is deliberately no apply mode."""
import json
import ntpath
from pathlib import Path
import sqlite3

from .errors import ControlError
from .windows_inventory import normal, outer_roots, within


def scope_filter(roots, selected):
    owner = next((r for r in outer_roots(roots) if within(selected['path'], r['path'])), None)
    if owner is None: raise ControlError('PLAN_SCOPE_NOT_IN_INVENTORY')
    prefix = ntpath.relpath(normal(selected['path']), normal(owner['path']))
    if prefix == '.': return 'root_id=?', [owner['id']]
    prefix += '\\'
    return '(root_id=? AND (relative=? COLLATE NOCASE OR substr(relative,1,?)=? COLLATE NOCASE))', [owner['id'], prefix[:-1], len(prefix), prefix]


def prepare_plan(database, scope, runtimes):
    connection = sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)
    connection.execute('BEGIN')
    plans = []
    try:
        for root in scope['roots']:
            where, values = scope_filter(scope['roots'], root)
            objects = connection.execute('SELECT relative,directory,identity,protected,reparse,acl_error,data FROM nodes WHERE ' + where, values)
            stats = dict(objects=0, directories=0, protected_files=0, protected_directories=0, reparse=0,
                         unknown_acl=0, null_dacl=0, no_inherited_aces=0, auto_inherited=0)
            candidates = []
            for relative, directory, identity, protected, reparse, acl_error, encoded in objects:
                row = json.loads(encoded)
                stats['objects'] += 1
                stats['directories'] += directory == 1
                stats['protected_directories' if directory else 'protected_files'] += bool(protected)
                stats['reparse'] += bool(reparse)
                stats['unknown_acl'] += bool(acl_error or row.get('metadata_error') or 'descriptor_sha256' not in row)
                stats['null_dacl'] += bool(row.get('null_dacl'))
                stats['no_inherited_aces'] += row.get('inherited_aces') == 0
                stats['auto_inherited'] += bool(row.get('auto_inherited'))
                owner = next(r for r in outer_roots(scope['roots']) if within(root['path'], r['path']))
                absolute = ntpath.join(owner['path'], relative)
                if (normal(absolute) == normal(root['path']) or protected) and not reparse:
                    candidates.append({'relative': ntpath.relpath(absolute, root['path']), 'identity': identity,
                        'directory': bool(directory), 'descriptor_sha256': row.get('descriptor_sha256'),
                        'reason': 'root' if normal(absolute) == normal(root['path']) else 'protected_descendant'})
            pending = connection.execute('SELECT count(*) FROM queue WHERE done=0 AND ' + where, values).fetchone()[0]
            issues = dict(connection.execute('SELECT kind,count(*) FROM issues WHERE ' + where + ' GROUP BY kind', values))
            aliases = dict(reparse_inside_scope=0, reparse_outside_scope=0, reparse_unresolved=0,
                           hardlinks_outside_scope=0, hardlink_names=0)
            external = []
            for relative, kind, target in connection.execute('SELECT relative,kind,target FROM aliases WHERE ' + where, values):
                inside = within(target, root['path']) if target else False
                if kind == 'hardlink':
                    aliases['hardlink_names'] += 1
                    aliases['hardlinks_outside_scope'] += not inside
                elif not target: aliases['reparse_unresolved'] += 1
                elif inside: aliases['reparse_inside_scope'] += 1
                else: aliases['reparse_outside_scope'] += 1
                if not inside: external.append({'relative': relative, 'kind': kind, 'target': target or None})
            overlap = [name for name, path in runtimes.items() if within(path, root['path'])]
            blockers = []
            if pending: blockers.append('TRAVERSAL_INCOMPLETE')
            if stats['unknown_acl'] or issues: blockers.append('METADATA_GAPS_OR_CONCURRENT_CHANGES')
            if aliases['hardlinks_outside_scope']: blockers.append('HARDLINKS_LEAVE_PROPOSED_SCOPE')
            if aliases['reparse_outside_scope'] or aliases['reparse_unresolved']: blockers.append('ALIAS_TARGETS_REQUIRE_REVIEW')
            if overlap: blockers.append('CURRENT_PROBE_RUNTIME_INSIDE_DENIED_ROOT')
            if stats['null_dacl']: blockers.append('NULL_DACL_REQUIRES_REVIEW')
            plans.append({'root_id': root['id'], 'root_path': root['path'], 'statistics': stats,
                'pending_directories': pending, 'issues': issues, 'aliases': aliases,
                'current_runtime_overlap': overlap, 'proposed_explicit_deny_objects': candidates,
                'external_or_unresolved_aliases': external, 'technical_blockers': blockers,
                'known_scope_ready_for_review': not blockers, 'authorized_to_apply': False,
                'required_live_recheck': ['identity', 'descriptor_hash', 'protected_descendants', 'aliases',
                    'new_objects', 'no_parallel_acl_writer', 'backup_and_standalone_recovery']})
        return {'schema_version': 1, 'scope': 'read_only_real_acl_acceptance_preparation', 'roots': plans,
            'real_acl_changes': 0, 'authorized_to_apply': False, 'formal_run_authorized': False,
            'execution_policy': 'fixed_synthetic_canaries_only_no_model',
            'recovery_requirements': ['backup_each_unique_object_descriptor_before_apply',
                'recovery_executable_and_journal_outside_denied_scope', 'compare_expected_post_apply_descriptor_before_restore',
                'refuse_to_overwrite_concurrent_acl_change', 'verify_owner_group_dacl_and_control_flags',
                'record_created_deleted_or_replaced_objects_as_unresolved', 'crash_recovery_independent_of_codex']}
    finally: connection.close()


def public_summary(plan):
    return {'scope': plan['scope'], 'real_acl_changes': 0, 'authorized_to_apply': False, 'formal_run_authorized': False,
            'roots': [{key: row[key] for key in ('root_id', 'statistics', 'pending_directories', 'issues', 'aliases',
                       'current_runtime_overlap', 'technical_blockers', 'known_scope_ready_for_review')}
                      | {'proposed_explicit_deny_count': len(row['proposed_explicit_deny_objects'])} for row in plan['roots']]}
