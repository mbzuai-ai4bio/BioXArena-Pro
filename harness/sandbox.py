"""Fail-closed Landlock filesystem boundary for ONE benchmark agent process tree (Codex CLI, Claude Code, Gemini CLI, OpenCode,
Minimal's solution.py, Biomni, MLEvolve). Needs Linux >= 6.2 (Landlock ABI >= 3); it refuses to run unconfined.
Read-only: system dirs, the Python env running this file (sys.prefix), Node / CLI installs (bx_config),
BX_SANDBOX_EXTRA_RO, the task's public dir.
Read-write: the run's output dir (which also holds the CLI home/config, tmp and caches) and /dev.
Everything else (other tasks, private answers, audits, API-key files, other runs) is inaccessible. Network is not blocked
(the agent must reach the API). Usage: sandbox.py <real-binary> [args...]; env BX_PUBLIC, BX_OUTPUT (required).
Optional BX_EXTRA_RO: colon-separated extra read-only paths (set by the Biomni / MLEvolve runners for their code)."""
import ctypes, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bx_config as C  # noqa: E402


def _tool_dir(p):  # install dir of a tool given as a file path (the npm package root)
    if not p: return None
    q = Path(p).resolve()
    for anc in [q] + list(q.parents):
        if anc.name in ("gemini-cli", "opencode-ai", "codex", "claude"): return str(anc)
    return str(q.parent)


RO = ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/lib32', '/etc', '/run/systemd/resolve', '/sys', '/opt',
      str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve())] + \
     [p for p in (C.NODE_HOME and str(Path(C.NODE_HOME).resolve()), _tool_dir(C.GEMINI_CLI_JS), _tool_dir(C.OPENCODE_BIN), _tool_dir(C.CODEX_BIN), _tool_dir(C.CLAUDE_BIN)) if p] + \
     C.SANDBOX_EXTRA_RO


def restrict(ro, rw):
    c = ctypes.CDLL(None, use_errno=True)

    def call(n, *a):
        r = c.syscall(n, *a)
        if r < 0:
            e = ctypes.get_errno(); raise OSError(e, os.strerror(e))
        return r
    if call(444, 0, 0, 1) < 3:
        raise RuntimeError('Landlock ABI >= 3 required; refusing unrestricted execution')

    class Ruleset(ctypes.Structure):
        _fields_ = [('access', ctypes.c_uint64)]

    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('access', ctypes.c_uint64), ('parent', ctypes.c_int32)]
    ALL = (1 << 15) - 1; READ = (1 << 0) | (1 << 2) | (1 << 3)
    rs = Ruleset(ALL); fd = call(444, ctypes.byref(rs), ctypes.sizeof(rs), 0)
    # /proc needs write_file + truncate: CUDA writes /proc/self/task/*/comm with O_TRUNC (cuInit fails otherwise)
    rules = [(p, READ) for p in ro] + [(p, ALL) for p in rw] + [('/proc', READ | (1 << 1) | (1 << 14))]
    for path, acc in rules:
        if not Path(path).exists():
            continue
        if not Path(path).is_dir():
            acc &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            call(445, fd, 1, ctypes.byref(Rule(acc, pfd)), 0)
        finally:
            os.close(pfd)
    if c.prctl(38, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'no_new_privs')
    call(446, fd, 0); os.close(fd)


if __name__ == '__main__':
    pub, out = os.environ.get('BX_PUBLIC'), os.environ.get('BX_OUTPUT')
    if not pub or not out:
        raise SystemExit('BX_PUBLIC / BX_OUTPUT not set; refusing to run unconfined')
    extra = [p for p in os.environ.get('BX_EXTRA_RO', '').split(':') if p]
    restrict(RO + [pub] + extra, [out, '/dev'])
    os.execv(sys.argv[1], sys.argv[1:])
