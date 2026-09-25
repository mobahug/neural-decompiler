"""Confirmation review 024 -- write guard (reviewer's own code). Import FIRST in every review script.

Installs a Python audit hook that REFUSES every write / create / remove / rename / metadata change whose target
resolves under the repository root (outputs/experiment-024/ included), and logs every such event elsewhere.
- dir_fd-relative opens (filelock probes in the system temp directory) are resolved through the directory descriptor
  (fcntl F_GETPATH), so they are judged by where they really are; an unresolvable target is refused.
- Relative paths without dir_fd resolve against the cwd (the repository in every review run), so they are refused.
- Subprocesses: only git with a read-only subcommand (and sysctl/uname probes); everything else is refused.
- APFS is case-insensitive: the repository prefix is compared case-folded (over-refusal is the safe side).
The hook is self-tested at import with sys.audit (event dispatch only; no file operation happens).
"""
from __future__ import annotations

import atexit
import fcntl
import os
import sys
import threading

REPO = os.path.realpath("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
SCRATCH = os.path.realpath("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/"
                           "8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirmation_review024")
_KEY = REPO.lower()
os.environ["GIT_OPTIONAL_LOCKS"] = "0"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | os.O_EXCL
GIT_READ_ONLY = {"rev-parse", "status", "ls-files", "merge-base", "diff", "show", "cat-file", "log", "ls-tree", "ls-remote",
                 "rev-list", "for-each-ref", "version", "hash-object", "grep"}
OTHER_ALLOWED = {"sysctl", "uname"}
STATS = {"refused": [], "outside": {}, "subprocesses": [], "selftest": []}
_tl = threading.local()


class RepoWriteRefused(PermissionError):
    pass


def _fd_path(fd):
    try:
        raw = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))
        return raw.split(b"\0", 1)[0].decode()
    except Exception:
        return None


def _resolve(path, dir_fd=None):
    """Absolute candidate paths (lexical, parent-realpath, realpath); None when unresolvable."""
    if isinstance(path, int):
        p = _fd_path(path)
        return None if p is None else [p, os.path.realpath(p)]
    try:
        p = os.fsdecode(os.fspath(path))
    except Exception:
        return None
    if not os.path.isabs(p):
        if isinstance(dir_fd, int) and dir_fd >= 0:
            base = _fd_path(dir_fd)
            if base is None:
                return None
        else:
            base = os.getcwd()
        p = os.path.join(base, p)
    lex = os.path.normpath(p)
    return [lex, os.path.join(os.path.realpath(os.path.dirname(lex)), os.path.basename(lex)), os.path.realpath(lex)]


def _under_repo(cands):
    return any(c.lower() == _KEY or c.lower().startswith(_KEY + os.sep) for c in cands)


def _judge(event, items):
    for path, dir_fd in items:
        if path is None:
            continue
        cands = _resolve(path, dir_fd)
        if cands is None or _under_repo(cands):
            entry = {"event": event, "path": repr(path), "dir_fd": dir_fd, "resolved": cands}
            (STATS["selftest"] if getattr(_tl, "selftest", False) else STATS["refused"]).append(entry)
            raise RepoWriteRefused(f"confirmation_review024 guard refused {event} on {cands or path!r}")
        if not getattr(_tl, "selftest", False):
            STATS["outside"].setdefault(event, set()).add(cands[-1])


def _argv_ok(argv):
    try:
        args = [os.fsdecode(a) for a in argv]
    except Exception:
        return False
    if not args:
        return False
    exe = os.path.basename(args[0])
    if exe in OTHER_ALLOWED:
        return True
    if exe != "git":
        return False
    i = 1
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-C", "-c") else 1
    if i >= len(args):
        return False
    if args[i] == "hash-object" and "-w" in args[i + 1:]:
        return False
    return args[i] in GIT_READ_ONLY


def _spawn(event, argv):
    argv = [argv] if isinstance(argv, (str, bytes)) else list(argv)
    ok = _argv_ok(argv)
    (STATS["selftest"] if getattr(_tl, "selftest", False) else STATS["subprocesses"]).append({"event": event, "argv": [os.fsdecode(a) for a in argv], "allowed": ok})
    if not ok:
        raise RepoWriteRefused(f"confirmation_review024 guard refused the subprocess {argv!r}")


def _open(event, args):
    path, mode, flags = args
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and bool(flags & WRITE_FLAGS))
    if writing:
        _judge(event, [(path, getattr(_tl, "dir_fd", None))])


HANDLERS = {
    "open": _open,
    "os.rename": lambda e, a: _judge(e, [(a[0], a[2]), (a[1], a[3])]),
    "os.remove": lambda e, a: _judge(e, [(a[0], a[1])]),
    "os.rmdir": lambda e, a: _judge(e, [(a[0], a[1])]),
    "os.mkdir": lambda e, a: _judge(e, [(a[0], a[2])]),
    "os.link": lambda e, a: _judge(e, [(a[0], a[2]), (a[1], a[3])]),
    "os.symlink": lambda e, a: _judge(e, [(a[1], a[2])]),
    "os.truncate": lambda e, a: _judge(e, [(a[0], None)]),
    "os.chmod": lambda e, a: _judge(e, [(a[0], a[2] if len(a) > 2 else None)]),
    "os.chown": lambda e, a: _judge(e, [(a[0], a[3] if len(a) > 3 else None)]),
    "os.utime": lambda e, a: _judge(e, [(a[0], a[3] if len(a) > 3 else None)]),
    "os.chflags": lambda e, a: _judge(e, [(a[0], None)]),
    "os.lchflags": lambda e, a: _judge(e, [(a[0], None)]),
    "os.setxattr": lambda e, a: _judge(e, [(a[0], None)]),
    "os.removexattr": lambda e, a: _judge(e, [(a[0], None)]),
    "os.mkfifo": lambda e, a: _judge(e, [(a[0], a[2] if len(a) > 2 else None)]),
    "os.mknod": lambda e, a: _judge(e, [(a[0], a[3] if len(a) > 3 else None)]),
    "shutil.copyfile": lambda e, a: _judge(e, [(a[1], None)]),
    "shutil.copymode": lambda e, a: _judge(e, [(a[1], None)]),
    "shutil.copystat": lambda e, a: _judge(e, [(a[1], None)]),
    "shutil.copytree": lambda e, a: _judge(e, [(a[1], None)]),
    "shutil.move": lambda e, a: _judge(e, [(a[0], None), (a[1], None)]),
    "shutil.rmtree": lambda e, a: _judge(e, [(a[0], a[1] if len(a) > 1 else None)]),
    "shutil.chown": lambda e, a: _judge(e, [(a[0], None)]),
    "shutil.make_archive": lambda e, a: _judge(e, [(a[0], None)]),
    "shutil.unpack_archive": lambda e, a: _judge(e, [(a[1], None)]),
    "tempfile.mkstemp": lambda e, a: _judge(e, [(a[0], None)]),
    "tempfile.mkdtemp": lambda e, a: _judge(e, [(a[0], None)]),
    "subprocess.Popen": lambda e, a: _spawn(e, a[1]),
    "os.posix_spawn": lambda e, a: _spawn(e, a[1]),
    "os.spawn": lambda e, a: _spawn(e, a[2]),
    "os.exec": lambda e, a: _spawn(e, a[1]),
    "os.system": lambda e, a: _spawn(e, [a[0]]),
}


def _hook(event, args):
    handler = HANDLERS.get(event)
    if handler is None or getattr(_tl, "busy", False):
        return
    _tl.busy = True
    try:
        handler(event, args)
    finally:
        _tl.busy = False


_real_os_open = os.open


def _os_open(path, flags, mode=0o777, *, dir_fd=None):
    previous = getattr(_tl, "dir_fd", None)
    _tl.dir_fd = dir_fd
    try:
        return _real_os_open(path, flags, mode, dir_fd=dir_fd)
    finally:
        _tl.dir_fd = previous


def _selftest():
    import tempfile

    tmp = os.path.realpath(tempfile.gettempdir())
    out = os.path.join(REPO, "outputs/experiment-024")
    cases = [
        ("repo open w", True, ("open", os.path.join(out, "results.json"), "w", 0)),
        ("repo open r+", True, ("open", os.path.join(out, "stage2-measurements.pt"), "r+", 0)),
        ("repo os.open O_CREAT", True, ("open", os.path.join(REPO, "zz_probe"), None, os.O_CREAT | os.O_WRONLY)),
        ("repo relative open w (cwd)", True, ("open", "zz_probe_rel", "w", 0)) if os.path.realpath(os.getcwd()).lower().startswith(_KEY) else None,
        ("repo os.replace into results", True, ("os.rename", os.path.join(tmp, "a"), os.path.join(out, "results.json"), -1, -1)),
        ("repo os.remove", True, ("os.remove", os.path.join(out, "results.json"), -1)),
        ("repo mkstemp in outputs", True, ("tempfile.mkstemp", os.path.join(out, ".results-x.json"))),
        ("repo utime", True, ("os.utime", os.path.join(out, "results.json"), None, None, -1)),
        ("repo chmod", True, ("os.chmod", os.path.join(out, "results.json"), 0o600, -1)),
        ("repo case-folded path", True, ("open", os.path.join(REPO.upper(), "X"), "w", 0)),
        ("git commit subprocess", True, ("subprocess.Popen", "git", ["git", "commit", "-m", "x"], None, None)),
        ("git hash-object -w", True, ("subprocess.Popen", "git", ["git", "hash-object", "-w", "x"], None, None)),
        ("uv subprocess", True, ("subprocess.Popen", "uv", ["uv", "run", "x"], None, None)),
        ("tmp open w", False, ("open", os.path.join(tmp, "probe"), "w", 0)),
        ("scratch open w", False, ("open", os.path.join(SCRATCH, "probe"), "w", 0)),
        ("repo open r", False, ("open", os.path.join(out, "results.json"), "r", 0)),
        ("git status subprocess", False, ("subprocess.Popen", "git", ["git", "status", "--porcelain"], None, None)),
    ]
    results = []
    _tl.selftest = True
    try:
        for case in cases:
            if case is None:
                continue
            name, expect_refused, (event, *args) = case
            refused = False
            try:
                sys.audit(event, *args)
            except RepoWriteRefused:
                refused = True
            results.append((name, expect_refused, refused))
    finally:
        _tl.selftest = False
    bad = [r for r in results if r[1] != r[2]]
    STATS["selftest_results"] = results
    if bad:
        raise SystemExit(f"GUARD SELFTEST FAILED: {bad}")
    return results


sys.addaudithook(_hook)
os.open = _os_open
SELFTEST = _selftest()


def summary():
    return {"refused": len(STATS["refused"]), "selftest_cases": len(SELFTEST),
            "outside_write_events": {k: len(v) for k, v in STATS["outside"].items()},
            "subprocesses": len(STATS["subprocesses"]), "disallowed_subprocesses": sum(1 for s in STATS["subprocesses"] if not s["allowed"])}


@atexit.register
def _report():
    s = summary()
    print(f"GUARD SUMMARY: refused={s['refused']} selftest_cases={s['selftest_cases']} (all as expected) "
          f"subprocesses={s['subprocesses']} disallowed={s['disallowed_subprocesses']} outside_write_events={s['outside_write_events']}", flush=True)
    for entry in STATS["refused"]:
        print(f"GUARD REFUSED: {entry}", flush=True)
    for event, paths in sorted(STATS["outside"].items()):
        for p in sorted(paths)[:6]:
            print(f"GUARD outside write [{event}]: {p}", flush=True)
