"""Reviewer's guard: an audit hook that refuses ANY write, create, rename, delete or metadata change under the
repository root (the .venv included), and logs every file opened for reading, tagged with the current stage.

Import this module first, before anything else is imported.
"""
import os
import sys
from pathlib import Path

ROOT = str(Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve())
STATE = {"stage": "import", "reads": [], "refused": [], "blocked_events": []}
_WRITE_MASK = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
# event -> ((path index, dir_fd index or None), ...): only the paths the operation creates, modifies or removes
_PATH_EVENTS = {
    "os.remove": ((0, 1),), "os.rmdir": ((0, 1),), "os.mkdir": ((0, 2),), "os.rename": ((0, 2), (1, 3)), "os.link": ((1, 3),),
    "os.symlink": ((1, 2),), "os.chmod": ((0, 2),), "os.chown": ((0, 3),), "os.utime": ((0, 3),), "os.truncate": ((0, None),),
    "os.chflags": ((0, None),), "os.lchflags": ((0, None),), "os.setxattr": ((0, None),), "os.removexattr": ((0, None),),
    "os.mkfifo": ((0, 2),), "os.mknod": ((0, 3),), "shutil.copyfile": ((1, None),), "shutil.copymode": ((1, None),),
    "shutil.copystat": ((1, None),), "shutil.copytree": ((1, None),), "shutil.move": ((0, None), (1, None)), "shutil.rmtree": ((0, 1),),
    "shutil.make_archive": ((0, None),),
}


def _fd_path(fd):
    try:
        import fcntl
        buf = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))
        return os.fsdecode(buf.split(b"\0", 1)[0])
    except Exception:
        return None


def _resolve(value, dir_fd=None):
    if isinstance(value, int) or value is None:
        return None
    try:
        text = os.fsdecode(value)
        if not os.path.isabs(text) and isinstance(dir_fd, int):
            base = _fd_path(dir_fd)
            if base is None:
                return "<unresolved dir_fd-relative path>"
            text = os.path.join(base, text)
        return os.path.realpath(text)
    except Exception:
        return None


def _under_root(path):
    return path is not None and (path == ROOT or path.startswith(ROOT + os.sep))


def _hook(event, args):
    if event == "open":
        if not args:
            return
        path = _resolve(args[0])
        if path is None:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and bool(flags & _WRITE_MASK))
        if writing and _under_root(path):
            STATE["refused"].append((STATE["stage"], event, path))
            raise PermissionError(f"reviewer guard: refused write under the repository: {path}")
        if not writing:
            STATE["reads"].append((STATE["stage"], path))
        return
    if event in _PATH_EVENTS:
        for index, fd_index in _PATH_EVENTS[event]:
            if index < len(args):
                dir_fd = args[fd_index] if fd_index is not None and fd_index < len(args) else None
                path = _resolve(args[index], dir_fd)
                if path == "<unresolved dir_fd-relative path>" or _under_root(path):
                    STATE["refused"].append((STATE["stage"], event, path))
                    raise PermissionError(f"reviewer guard: refused {event} under the repository: {path}")


sys.addaudithook(_hook)


def stage(name):
    STATE["stage"] = name


def reads_in(stage_name):
    return sorted({path for s, path in STATE["reads"] if s == stage_name})
