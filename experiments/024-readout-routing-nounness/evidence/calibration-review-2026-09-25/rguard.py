"""Reviewer's guard: an audit hook that refuses any write under the repository root (the .venv included) and any open
or listing of Experiment 022's local calibration table or anything under outputs/experiment-023/. Every refusal is
recorded. Import this module first."""
import os
import sys

ROOT = os.path.realpath("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
FORBIDDEN_FILES = {os.path.realpath(os.path.join(ROOT, "outputs/experiment-022/calibration-table.pt"))}
FORBIDDEN_DIR = os.path.realpath(os.path.join(ROOT, "outputs/experiment-023"))
EVENTS = {"refused_writes": [], "refused_reads": [], "open_events": 0}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _real(p, dir_fd=None):
    """Resolve a path; a relative path given with a directory descriptor is resolved against that directory (macOS
    F_GETPATH), not against the working directory."""
    try:
        s = os.fsdecode(p)
        if dir_fd is not None and not os.path.isabs(s):
            import fcntl
            base = fcntl.fcntl(dir_fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0].decode()
            s = os.path.join(base, s)
        return os.path.realpath(s)
    except Exception:
        return None


def _under(r, base):
    return r is not None and (r == base or r.startswith(base + os.sep))


def _forbidden_read(r):
    return r in FORBIDDEN_FILES or _under(r, FORBIDDEN_DIR)


def _pathlike(p):
    return isinstance(p, (str, bytes, os.PathLike))


def _hook(event, args):
    if event == "open":
        EVENTS["open_events"] += 1
        if not args or not _pathlike(args[0]):
            return
        r = _real(args[0])
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        if _forbidden_read(r):
            EVENTS["refused_reads"].append(r)
            raise PermissionError(f"review guard: may not open {r}")
        if writing and _under(r, ROOT):
            EVENTS["refused_writes"].append(r)
            raise PermissionError(f"review guard: may not write {r}")
    elif event in ("os.listdir", "os.scandir"):
        if args and _pathlike(args[0]) and _forbidden_read(_real(args[0])):
            EVENTS["refused_reads"].append(_real(args[0]))
            raise PermissionError("review guard: may not list outputs/experiment-023")
    elif event in ("os.rename", "os.link", "os.symlink", "shutil.copyfile", "shutil.move", "shutil.copytree", "shutil.copymode", "shutil.copystat"):
        for p in args[:2]:
            if _pathlike(p) and _under(_real(p), ROOT):
                EVENTS["refused_writes"].append(f"{event}:{_real(p)}")
                raise PermissionError(f"review guard: {event} under the repository")
    elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.truncate", "os.chmod", "os.chown", "os.utime", "shutil.rmtree", "os.chflags", "os.lchflags",
                   "os.setxattr", "os.removexattr", "os.makedirs"):
        dir_fd = args[-1] if event in ("os.remove", "os.rmdir", "os.mkdir", "shutil.rmtree") and args and isinstance(args[-1], int) and args[-1] >= 0 else None
        if args and _pathlike(args[0]):
            r = _real(args[0], dir_fd)
            if r is None or _under(r, ROOT):
                EVENTS["refused_writes"].append(f"{event}:{r or args[0]}")
                raise PermissionError(f"review guard: {event} under the repository (or unresolvable)")


sys.addaudithook(_hook)
