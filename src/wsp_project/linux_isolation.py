"""Fail-closed Linux worker boundary, installed before importing model libraries.

Landlock rules and seccomp filters are inherited by subsequently created threads.
No account, service, firewall, mount or system-wide policy is created here.
"""
import ctypes
import errno
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

from .errors import ControlError


def _fail():
    raise ControlError("WORKER_ISOLATION_UNAVAILABLE")


def isolate(work, read_paths, *, uid=61234):
    if sys.platform != "linux" or platform.machine() != "x86_64" or os.geteuid() != 0 or not 60000<=uid<=64999:
        _fail()
    work = Path(work).resolve()
    if work.stat().st_uid != uid:
        _fail()
    libc = ctypes.CDLL(None,use_errno=True)
    seccomp = ctypes.CDLL("libseccomp.so.2",use_errno=True)
    abi = libc.syscall(444,0,0,1)
    if abi<6:
        _fail()
    read = (1<<2)|(1<<3)
    write = read | (1<<1)|(1<<4)|(1<<5)|(1<<7)|(1<<8)|(1<<9)|(1<<13)|(1<<14)
    handled = (1<<15)-1
    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs",ctypes.c_uint64),("handled_access_net",ctypes.c_uint64),("scoped",ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access",ctypes.c_uint64),("parent_fd",ctypes.c_int)]
    rule = Ruleset(handled,3,3)  # Deny TCP; scope abstract Unix sockets and signals.
    descriptor = libc.syscall(444,ctypes.byref(rule),ctypes.sizeof(rule),0)
    if descriptor<0:
        _fail()
    system = [sys.prefix,"/usr/lib/python3.14","/usr/local/lib/python3.14","/usr/lib/x86_64-linux-gnu","/lib/x86_64-linux-gnu",
              "/etc/ld.so.cache","/etc/localtime","/etc/passwd","/etc/nsswitch.conf","/usr/share/zoneinfo",
              "/proc/self","/proc/meminfo","/proc/cpuinfo","/proc/devices","/proc/driver/nvidia","/proc/sys/kernel/osrelease",
              "/proc/sys/kernel/random/boot_id","/proc/sys/vm/mmap_min_addr","/proc/sys/vm/overcommit_memory","/sys"]
    paths = [(Path(p),read) for p in [*system,*read_paths]] + [(work,write)]
    # NVIDIA initializes worker thread names through /proc/self/task/<tid>/comm.
    # This rule is bound to this process's task inode, not other processes.
    paths += [(Path('/proc/self/task'),read|(1<<1)|(1<<14))]
    paths += [(p,(1<<1)|(1<<2)) for p in Path('/dev').glob('nvidia*') if not p.is_dir()]
    paths += [(Path(p),(1<<1)|(1<<2)) for p in ("/dev/null","/dev/zero","/dev/urandom","/dev/random")]
    try:
        for path,access in paths:
            if not path.exists():
                continue
            fd = os.open(path.resolve(),os.O_PATH|os.O_CLOEXEC)
            try:
                if not path.is_dir():
                    access &= (1<<1)|(1<<2)|(1<<14)
                beneath = Beneath(access,fd)
                if libc.syscall(445,descriptor,1,ctypes.byref(beneath),0)<0:
                    _fail()
            finally:
                os.close(fd)
        if libc.prctl(38,1,0,0,0)!=0:
            _fail()
        os.setgroups([])
        os.setresgid(uid,uid,uid)
        os.setresuid(uid,uid,uid)
        if libc.syscall(446,descriptor,0)<0:
            _fail()
    finally:
        os.close(descriptor)
    os.chdir(work)
    environment = {"PATH":str(Path(sys.executable).parent)+":/usr/bin:/bin","HOME":str(work/"home"),"TMPDIR":str(work/"tmp"),
        "USER":"wsp-worker","LOGNAME":"wsp-worker","LANG":"C.UTF-8","LC_ALL":"C.UTF-8","PYTHONDONTWRITEBYTECODE":"1",
        "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","HF_HUB_DISABLE_IMPLICIT_TOKEN":"1",
        "TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8"}
    os.environ.clear()
    os.environ.update(environment)
    (work/"home").mkdir(exist_ok=True)
    (work/"tmp").mkdir(exist_ok=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7fff0000)
    if not context:
        _fail()
    blocked = ("connect","sendto","sendmsg","sendmmsg",
               "execve","execveat","fork","vfork","ptrace","process_vm_readv","process_vm_writev","mount","umount2",
               "pivot_root","setns","unshare","open_by_handle_at","bpf","perf_event_open","userfaultfd","io_uring_setup")
    try:
        class ArgCompare(ctypes.Structure):
            _fields_ = [("arg",ctypes.c_uint),("op",ctypes.c_int),("value",ctypes.c_uint64),("value_two",ctypes.c_uint64)]
        seccomp.seccomp_rule_add_array.argtypes = [ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(ArgCompare)]
        # CUDA needs local Unix IPC listeners. Filesystem sockets can only be
        # created under work; abstract sockets are scoped by Landlock. Outbound
        # connects and ancillary-message sends stay denied; TCP/UDP/Netlink too.
        for name in ("socket","socketpair"):
            condition = ArgCompare(0,1,socket.AF_UNIX,0)  # SCMP_CMP_NE
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if seccomp.seccomp_rule_add_array(context,0x00050000|errno.EPERM,number,1,ctypes.byref(condition))!=0:
                _fail()
        for name in blocked:
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number<0 or seccomp.seccomp_rule_add(context,0x00050000|errno.EPERM,number,0)!=0:
                _fail()
        if seccomp.seccomp_load(context)!=0:
            _fail()
    finally:
        seccomp.seccomp_release(context)
    return {"uid":os.geteuid(),"gid":os.getegid(),"landlock_abi":abi,"landlock_applied":True,"seccomp_applied":True,
            "installed_before_model_import":True,"credential_environment_absent":not any(k in os.environ for k in ("OPENAI_API_KEY","HF_TOKEN","ANTHROPIC_API_KEY","SSH_AUTH_SOCK"))}


def probe(evidence, *, canary, protected_file, work):
    checks = {}
    for key,path,flags in (("private_read_denied",canary,os.O_RDONLY),("base_write_denied",protected_file,os.O_WRONLY)):
        try:
            fd = os.open(path,flags)
        except PermissionError:
            checks[key] = True
        else:
            os.close(fd)
            checks[key] = False
    try:
        sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    except PermissionError:
        checks["network_socket_denied"] = True
    else:
        sock.close()
        checks["network_socket_denied"] = False
    for key,family,kind in (("udp_socket_denied",socket.AF_INET,socket.SOCK_DGRAM),("ipv6_socket_denied",socket.AF_INET6,socket.SOCK_STREAM)):
        try:
            sock = socket.socket(family,kind)
        except PermissionError:
            checks[key] = True
        else:
            sock.close()
            checks[key] = False
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        try:
            sock.connect(str(Path(work)/"not-a-listener"))
        except PermissionError:
            checks["unix_outbound_connect_denied"] = True
        except OSError:
            checks["unix_outbound_connect_denied"] = False
        else:
            checks["unix_outbound_connect_denied"] = False
    try:
        subprocess.run(["/bin/true"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
    except PermissionError:
        checks["program_execution_denied"] = True
    except Exception:
        checks["program_execution_denied"] = False
    else:
        checks["program_execution_denied"] = False
    status = {line.split(":",1)[0]:line.split(":",1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
    checks.update(non_root=os.geteuid()!=0,no_new_privileges=status.get("NoNewPrivs")=="1",effective_capabilities_zero=int(status.get("CapEff","1"),16)==0,
                  permitted_capabilities_zero=int(status.get("CapPrm","1"),16)==0,ambient_capabilities_zero=int(status.get("CapAmb","1"),16)==0)
    target = Path(work)/"isolation-write-probe"
    target.write_text("public engineering probe\n")
    target.unlink()
    checks["work_write_allowed"] = True
    if not all(checks.values()):
        raise ControlError("ISOLATION_PROBE_FAILED")
    return {**evidence,**checks,"scope":"remote_worker_only_not_windows_editor_isolation"}
