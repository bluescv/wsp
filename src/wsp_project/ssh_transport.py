"""SSH stdio transport using existing POSIX ssh/sshpass; no credential files.

Password auth is optional. When used, the password is passed through an
inherited anonymous file descriptor, never argv, environment or a log.
"""

import os
import shlex
import shutil
import subprocess

from .errors import ControlError
from .manifest import identifier


class SSHTransport:
    def __init__(self, host, user, known_hosts, *, password=None):
        # This first transport targets the user-provided IP/ordinary aliases.
        identifier(host)
        identifier(user)
        if not shutil.which("ssh") or (password is not None and (os.name != "posix" or not shutil.which("sshpass"))):
            raise ControlError("SSH_TOOL_UNAVAILABLE")
        if not os.path.isfile(known_hosts):
            raise ControlError("KNOWN_HOSTS_REQUIRED")
        if password is not None and (not password or len(password) > 4096 or b"\n" in password or b"\r" in password):
            raise ControlError("INVALID_AUTH_INPUT")
        self.password = password
        self.command = ["ssh", "-T", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + known_hosts,
                        "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1",
                        "-o", "NumberOfPasswordPrompts=1", "-o", "LogLevel=ERROR"]
        self.command += (["-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no"] if password is not None
                         else ["-o", "BatchMode=yes"])
        self.command.append(user + "@" + host)

    def spawn(self, argv, *, capture_diagnostics=False):
        command = self.command + [shlex.join(argv)]
        inherited = ()
        read_fd = None
        if self.password is not None:
            read_fd, write_fd = os.pipe()
            try:
                os.write(write_fd, self.password + b"\n")
            finally:
                os.close(write_fd)
            inherited = (read_fd,)
            command = ["sshpass", "-d", str(read_fd), *command]
        try:
            return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE if capture_diagnostics else subprocess.DEVNULL, pass_fds=inherited)
        finally:
            if read_fd is not None:
                os.close(read_fd)

    def run(self, argv, data=None, timeout=30):
        process = self.spawn(argv, capture_diagnostics=True)
        try:
            output, diagnostic = process.communicate(data, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise ControlError("SSH_TIMEOUT_UNCERTAIN") from None
        if process.returncode != 0:
            # Classify in memory; never emit original diagnostic text or a
            # redacted copy. Password material is not supplied to remote argv.
            if process.returncode == 5 or b"Permission denied" in diagnostic:
                raise ControlError("SSH_AUTHENTICATION_FAILED")
            if process.returncode in (6, 7) or b"Host key verification failed" in diagnostic:
                raise ControlError("SSH_HOST_KEY_REJECTED")
            if b"ModuleNotFoundError" in diagnostic or b"ZipImportError" in diagnostic:
                raise ControlError("REMOTE_MODULE_UNAVAILABLE")
            if process.returncode == 255:
                raise ControlError("SSH_CONNECTION_FAILED")
            raise ControlError("SSH_REMOTE_OPERATION_FAILED")
        return output
