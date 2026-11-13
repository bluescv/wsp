"""Read-only Windows ACL/identity inventory. Never opens file data or follows links."""
import ctypes
from ctypes import wintypes as W
import hashlib
import json
import ntpath
import os
from pathlib import Path
import sqlite3
import struct
import time

from .errors import ControlError


def normal(path):
    value = str(path)
    if value.startswith('\\\\?\\UNC\\'): value = '\\\\' + value[8:]
    elif value.startswith(('\\\\?\\', '\\??\\')): value = value[4:]
    return ntpath.normcase(ntpath.normpath(value))


def within(path, root):
    path, root = normal(path), normal(root)
    return path == root or path.startswith(root.rstrip('\\') + '\\')


def extended(path):
    value = ntpath.abspath(str(path))
    if value.startswith('\\\\?\\'): return value
    if value.startswith('\\\\'): return '\\\\?\\UNC\\' + value[2:]
    return '\\\\?\\' + value


def outer_roots(roots):
    """Keep named scope records, but traverse an overlapping tree only once."""
    ordered = sorted(roots, key=lambda r: len(normal(r['path'])))
    result = []
    for row in ordered:
        if not any(within(row['path'], parent['path']) for parent in result): result.append(row)
    return result


def alias_scope(target, roots):
    if not target: return 'unresolved'
    return 'inside_declared_roots' if any(within(target, r['path']) for r in roots) else 'outside_declared_roots'


def acl_counts(blob):
    if blob is None:
        return {'null_dacl': True, 'ace_count': 0, 'inherited_aces': 0, 'explicit_denies': 0, 'inheritance_aces': 0}
    if len(blob) < 8: raise ControlError('ACL_BUFFER_INVALID')
    size, count = struct.unpack_from('<HH', blob, 2)
    if size != len(blob): raise ControlError('ACL_BUFFER_INVALID')
    result = dict(null_dacl=False, ace_count=count, inherited_aces=0, explicit_denies=0, inheritance_aces=0)
    offset = 8
    for _ in range(count):
        if offset + 4 > size: raise ControlError('ACL_BUFFER_INVALID')
        kind, flags, length = struct.unpack_from('<BBH', blob, offset)
        if length < 4 or offset + length > size: raise ControlError('ACL_BUFFER_INVALID')
        result['inherited_aces'] += bool(flags & 0x10)
        result['explicit_denies'] += kind in (1, 6, 10, 12) and not flags & 0x10
        result['inheritance_aces'] += bool(flags & 3)
        offset += length
    return result


class FileInfo(ctypes.Structure):
    _fields_ = [('attributes', W.DWORD), ('created', W.FILETIME), ('accessed', W.FILETIME), ('written', W.FILETIME),
                ('volume', W.DWORD), ('size_hi', W.DWORD), ('size_lo', W.DWORD), ('links', W.DWORD),
                ('index_hi', W.DWORD), ('index_lo', W.DWORD)]


class NativeMetadata:
    def __init__(self):
        if os.name != 'nt': raise ControlError('WINDOWS_NATIVE_REQUIRED')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.security = ctypes.WinDLL('advapi32', use_last_error=True)
        k, a = self.kernel, self.security
        k.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, ctypes.c_void_p, W.DWORD, W.DWORD, W.HANDLE]
        k.CreateFileW.restype = W.HANDLE
        k.CloseHandle.argtypes = [W.HANDLE]
        k.GetFileInformationByHandle.argtypes = [W.HANDLE, ctypes.POINTER(FileInfo)]
        k.GetFileInformationByHandle.restype = W.BOOL
        k.GetFinalPathNameByHandleW.argtypes = [W.HANDLE, W.LPWSTR, W.DWORD, W.DWORD]
        k.GetFinalPathNameByHandleW.restype = W.DWORD
        k.LocalFree.argtypes = [ctypes.c_void_p]
        k.LocalFree.restype = ctypes.c_void_p
        a.GetSecurityInfo.argtypes = [W.HANDLE, ctypes.c_int, W.DWORD, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        a.GetSecurityInfo.restype = W.DWORD
        a.GetSecurityDescriptorControl.argtypes = [ctypes.c_void_p, ctypes.POINTER(W.WORD), ctypes.POINTER(W.DWORD)]
        a.GetSecurityDescriptorLength.argtypes = [ctypes.c_void_p]
        a.GetSecurityDescriptorLength.restype = W.DWORD
        k.FindFirstFileNameW.argtypes = [W.LPCWSTR, W.DWORD, ctypes.POINTER(W.DWORD), W.LPWSTR]
        k.FindFirstFileNameW.restype = W.HANDLE
        k.FindNextFileNameW.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD), W.LPWSTR]
        k.FindNextFileNameW.restype = W.BOOL
        k.FindClose.argtypes = [W.HANDLE]
        k.GetVolumePathNameW.argtypes = [W.LPCWSTR, W.LPWSTR, W.DWORD]
        self.invalid = ctypes.c_void_p(-1).value

    def hardlinks(self, path):
        k = self.kernel
        path = extended(path)
        buffer = ctypes.create_unicode_buffer(32768)
        size = W.DWORD(len(buffer))
        handle = k.FindFirstFileNameW(str(path), 0, ctypes.byref(size), buffer)
        if handle == self.invalid: return [], ctypes.get_last_error()
        volume = ctypes.create_unicode_buffer(32768)
        if not k.GetVolumePathNameW(str(path), volume, len(volume)):
            code = ctypes.get_last_error(); k.FindClose(handle); return [], code
        names = []
        try:
            while True:
                names.append(ntpath.join(volume.value, buffer.value.lstrip('\\')))
                size.value = len(buffer)
                if not k.FindNextFileNameW(handle, ctypes.byref(size), buffer):
                    code = ctypes.get_last_error()
                    return names, 0 if code == 38 else code  # ERROR_HANDLE_EOF
        finally: k.FindClose(handle)

    def read(self, path, *, links=True, include_descriptor=False):
        k, a = self.kernel, self.security
        path = extended(path)
        # READ_CONTROL | FILE_READ_ATTRIBUTES; no FILE_READ_DATA, no write access,
        # no backup privilege, no SACL. Open the reparse point itself.
        handle = k.CreateFileW(str(path), 0x20080, 7, None, 3, 0x02200000, None)
        row = {'acl_error': 0, 'metadata_error': 0}
        if handle == self.invalid:
            row['acl_error'] = ctypes.get_last_error()
            handle = k.CreateFileW(str(path), 0x80, 7, None, 3, 0x02200000, None)
            if handle == self.invalid:
                row['metadata_error'] = ctypes.get_last_error()
                return row
        try:
            info = FileInfo()
            if not k.GetFileInformationByHandle(handle, ctypes.byref(info)):
                row['metadata_error'] = ctypes.get_last_error(); return row
            row.update(attributes=info.attributes, is_directory=bool(info.attributes & 0x10),
                reparse=bool(info.attributes & 0x400), links=info.links,
                identity=f'{info.volume:08x}:{info.index_hi:08x}{info.index_lo:08x}',
                directory_write_time=f'{info.written.dwHighDateTime:08x}{info.written.dwLowDateTime:08x}')
            if row['is_directory']:
                buffer = ctypes.create_unicode_buffer(32768)
                length = k.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
                row['resolved_path'] = buffer.value if 0 < length < len(buffer) else None
            if not row['acl_error']:
                descriptor, dacl = ctypes.c_void_p(), ctypes.c_void_p()
                row['acl_error'] = a.GetSecurityInfo(handle, 1, 7, None, None, ctypes.byref(dacl), None, ctypes.byref(descriptor))
                if not row['acl_error']:
                    try:
                        control, revision = W.WORD(), W.DWORD()
                        if not a.GetSecurityDescriptorControl(descriptor, ctypes.byref(control), ctypes.byref(revision)):
                            row['acl_error'] = ctypes.get_last_error()
                        else:
                            blob = None if not dacl.value else ctypes.string_at(dacl, struct.unpack('<H', ctypes.string_at(dacl.value + 2, 2))[0])
                            row.update(acl_counts(blob))
                            descriptor_bytes = ctypes.string_at(descriptor, a.GetSecurityDescriptorLength(descriptor))
                            row.update(protected=bool(control.value & 0x1000), auto_inherited=bool(control.value & 0x400),
                                control_flags=control.value, dacl_sha256=hashlib.sha256(blob or b'NULL_DACL').hexdigest(),
                                descriptor_sha256=hashlib.sha256(descriptor_bytes).hexdigest())
                            if include_descriptor: row['_descriptor_bytes'] = descriptor_bytes
                    finally: k.LocalFree(descriptor)
            if row['reparse']:
                try: row['reparse_tag'] = os.lstat(path).st_reparse_tag
                except OSError: row['reparse_tag'] = None
                try:
                    target = os.readlink(path)
                    row['reparse_target'] = ntpath.normpath(ntpath.join(ntpath.dirname(str(path)), target)) if not ntpath.isabs(target) else target
                except (OSError, ValueError) as error:
                    # App execution aliases and other non-name-surrogate tags
                    # cannot be resolved by readlink; retain them as unresolved.
                    row['reparse_error'] = getattr(error, 'winerror', 0) or 0
                    row['reparse_target_unresolved'] = True
            elif links and not row['is_directory'] and row['links'] > 1:
                row['hardlink_names'], row['hardlink_error'] = self.hardlinks(path)
            return row
        finally: k.CloseHandle(handle)


class Inventory:
    def __init__(self, database, roots, *, excluded, reader=None):
        self.roots, self.excluded = roots, excluded
        self.reader = reader or NativeMetadata()
        self.db = sqlite3.connect(database)
        self.db.executescript('''
          PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, value TEXT);
          CREATE TABLE IF NOT EXISTS nodes (root_id TEXT, relative TEXT, directory INTEGER, identity TEXT,
            protected INTEGER, reparse INTEGER, acl_error INTEGER, data TEXT, PRIMARY KEY(root_id,relative));
          CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY, root_id TEXT, relative TEXT, done INTEGER DEFAULT 0,
            UNIQUE(root_id,relative));
          CREATE INDEX IF NOT EXISTS queue_pending ON queue(done,id);
          CREATE TABLE IF NOT EXISTS issues (root_id TEXT, relative TEXT, kind TEXT, code INTEGER,
            UNIQUE(root_id,relative,kind,code));
          CREATE TABLE IF NOT EXISTS aliases (root_id TEXT, relative TEXT, kind TEXT, target TEXT, scope TEXT,
            UNIQUE(root_id,relative,kind,target));
        ''')
        config = json.dumps({'roots': roots, 'excluded': excluded}, sort_keys=True)
        prior = self.db.execute('SELECT value FROM config WHERE id=1').fetchone()
        if prior and prior[0] != config:
            self.db.close()
            raise ControlError('INVENTORY_SCOPE_CHANGED')
        self.db.execute('INSERT OR IGNORE INTO config VALUES (1,?)', (config,))
        self.by_id = {row['id']: row['path'] for row in roots}
        if not prior:
            for root in outer_roots(roots): self.record(root['id'], '.', root['path'])
        self.db.commit()

    def issue(self, root, relative, kind, code=0):
        self.db.execute('INSERT OR IGNORE INTO issues VALUES (?,?,?,?)', (root, relative, kind, code))

    def record(self, root, relative, path):
        if any(within(path, excluded) for excluded in self.excluded): return
        try: row = self.reader.read(path)
        except (OSError, ControlError) as error:
            self.issue(root, relative, 'metadata_read_failed', getattr(error, 'winerror', 0) or 0); return
        for key in ('metadata_error', 'acl_error', 'hardlink_error'):
            if row.get(key): self.issue(root, relative, key, row[key])
        self.db.execute('INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?,?,?,?)', (root, relative,
            row.get('is_directory'), row.get('identity'), row.get('protected'), row.get('reparse'),
            row.get('acl_error', 0), json.dumps(row, ensure_ascii=False)))
        if row.get('reparse'):
            target = row.get('reparse_target', '')
            self.db.execute('INSERT OR IGNORE INTO aliases VALUES (?,?,?,?,?)',
                (root, relative, 'reparse', target, alias_scope(target, self.roots)))
        elif row.get('is_directory'):
            self.db.execute('INSERT OR IGNORE INTO queue(root_id,relative) VALUES (?,?)', (root, relative))
        for target in row.get('hardlink_names', []):
            self.db.execute('INSERT OR IGNORE INTO aliases VALUES (?,?,?,?,?)',
                (root, relative, 'hardlink', target, alias_scope(target, self.roots)))
        if row.get('hardlink_names') and len(row['hardlink_names']) != row['links']:
            self.issue(root, relative, 'hardlink_count_changed')

    def scan(self, *, seconds=600, progress=lambda: None):
        deadline, checkpoint = time.monotonic() + seconds, time.monotonic()
        while time.monotonic() < deadline:
            pending = self.db.execute('SELECT id,root_id,relative FROM queue WHERE done=0 ORDER BY id LIMIT 1').fetchone()
            if not pending: break
            identifier, root, relative = pending
            path = os.path.join(self.by_id[root], relative)
            before = self.reader.read(path, links=False)
            finished = True
            if before.get('metadata_error') or before.get('reparse') or not before.get('is_directory'):
                self.issue(root, relative, 'directory_unavailable_or_replaced', before.get('metadata_error', 0))
            elif not before.get('resolved_path') or normal(before['resolved_path']) != normal(path):
                self.issue(root, relative, 'directory_alias_changed')
            else:
                try:
                    with os.scandir(extended(path) if os.name == 'nt' else path) as entries:
                        for entry in entries:
                            self.record(root, os.path.normpath(os.path.join(relative, entry.name)), entry.path)
                            if time.monotonic() - checkpoint > 10:
                                self.db.commit(); progress(); checkpoint = time.monotonic()
                            if time.monotonic() >= deadline:
                                finished = False; break
                except OSError as error:
                    self.issue(root, relative, 'directory_enumeration_failed', getattr(error, 'winerror', 0) or 0)
                after = self.reader.read(path, links=False)
                if any(before.get(key) != after.get(key) for key in ('identity', 'descriptor_sha256', 'directory_write_time')):
                    self.issue(root, relative, 'directory_changed_during_scan')
            if finished: self.db.execute('UPDATE queue SET done=1 WHERE id=?', (identifier,))
            if time.monotonic() - checkpoint > 10:
                self.db.commit(); progress(); checkpoint = time.monotonic()
        self.db.commit()
        return self.summary()

    def summary(self):
        counts = self.db.execute('SELECT count(*),sum(directory=1),sum(directory=0),sum(protected=1),sum(reparse=1),sum(acl_error!=0) FROM nodes').fetchone()
        pending = self.db.execute('SELECT count(*) FROM queue WHERE done=0').fetchone()[0]
        issues = dict(self.db.execute('SELECT kind,count(*) FROM issues GROUP BY kind'))
        aliases = [{'kind': kind, 'scope': scope, 'count': count} for kind, scope, count in
                   self.db.execute('SELECT kind,scope,count(*) FROM aliases GROUP BY kind,scope')]
        return {'objects': counts[0], 'directories': counts[1] or 0, 'files': counts[2] or 0,
            'protected_objects': counts[3] or 0, 'reparse_objects': counts[4] or 0, 'acl_unreadable_objects': counts[5] or 0,
            'pending_directories': pending, 'traversal_finished': pending == 0, 'issues': issues, 'aliases': aliases,
            'file_contents_read': False, 'real_acl_changes': 0, 'aliases_followed': False,
            'actual_private_roots_os_acl_verified': False, 'formal_run_authorized': False}

    def close(self):
        self.db.commit(); self.db.close()
