"""Shared helpers for the confirm-launcher review (AI reviewer's own scripts; scratch only).

- ``install_guard()``: an audit hook for MY test processes that refuses any write-type event outside this review's
  scratch directory (fail closed; the parent directory is resolved and the final component kept, so a symlink's own
  entry is judged, not its target) and records watched events. It is installed first in every script, so nothing of
  the repository can be written by any of my tests.
- ``extract_launcher()``: the confirm launcher's hook functions and path constants, extracted from the REVIEWED copy's
  source with ``ast`` (as the operator's ``hook_self_test.py`` does) and executed in a bare namespace whose ROOT is a
  caller-given root (a mirror tree in scratch, or the real repository path for pure decisions only).
"""
from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import sys
import types
from pathlib import Path

REVIEW = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/confirm-launcher-review").resolve()
OUT = REVIEW / "out"
REPO = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
LAUNCHER = REVIEW / "reviewed/launch_confirm.py"
LAUNCHER_SHA256 = "4f132e8e82d46e19dfad039cfcdf3b7bbdfa8d6199976f8b791dfbe3fba0b209"
OPERATOR_COPY = REVIEW.parent / "confirm/launch_confirm.py"

WRITE_EVENTS = {"os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.link", "os.symlink", "os.chmod", "os.chown", "os.utime", "os.truncate",
                "os.chflags", "os.setxattr", "os.removexattr", "shutil.rmtree", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
                "shutil.copytree", "shutil.move", "shutil.make_archive", "shutil.unpack_archive", "tempfile.mkstemp", "tempfile.mkdtemp",
                "sqlite3.connect", "os.mkfifo", "os.mknod"}
WATCH_EVENTS = WRITE_EVENTS | {"open", "os.chdir", "subprocess.Popen", "os.posix_spawn", "os.system", "os.exec", "os.fork", "os.putenv", "os.unsetenv",
                               "fcntl.fcntl", "os.listdir", "os.scandir", "pickle.find_class", "torch.load"}
JUDGED = {"open", "os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir", "os.chmod", "os.rename", "os.link", "os.symlink"}


def check_launcher() -> None:
    digest = hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()
    if digest != LAUNCHER_SHA256:
        raise SystemExit(f"the reviewed copy changed: {digest}")


def write_json(name: str, value) -> Path:
    path = OUT / name
    path.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def _entry_path(value, dir_fd=None):
    """Resolve the parent directory and keep the final component (the entry an operation acts on)."""
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    raw = os.fsdecode(value)
    if not os.path.isabs(raw):
        if isinstance(dir_fd, int) and not isinstance(dir_fd, bool) and dir_fd >= 0:
            try:
                base = fcntl.fcntl(dir_fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0].decode()
            except (OSError, ValueError):
                return "<unresolved>"
        else:
            base = os.getcwd()
        raw = os.path.join(base, raw)
    raw = os.path.normpath(raw)
    head, tail = os.path.split(raw)
    return os.path.join(os.path.realpath(head), tail)


def _guard_paths(event, args):
    if event == "open":
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return [], False
        mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)))
        return [_entry_path(args[0])], writing
    if event in ("os.mkdir", "os.chmod"):
        return [_entry_path(args[0], args[2] if len(args) > 2 else None)], True
    if event in ("os.remove", "os.rmdir", "shutil.rmtree"):
        return [_entry_path(args[0], args[1] if len(args) > 1 else None)], True
    if event in ("os.rename", "os.link"):
        return [_entry_path(args[0], args[2] if len(args) > 2 else None), _entry_path(args[1], args[3] if len(args) > 3 else None)], True
    if event == "os.symlink":
        return [_entry_path(args[1], args[2] if len(args) > 2 else None)], True
    if event in ("os.utime", "os.chown", "os.chflags", "os.truncate", "os.setxattr", "os.removexattr", "tempfile.mkstemp", "tempfile.mkdtemp",
                 "shutil.copyfile", "shutil.copytree", "shutil.move", "os.mkfifo", "os.mknod"):
        paths = [_entry_path(a) for a in args[:2] if isinstance(a, (str, bytes, os.PathLike))]
        if event in ("shutil.copyfile", "shutil.copytree", "shutil.move"):
            paths = paths[1:]  # the destination
        return paths, True
    if event == "sqlite3.connect":
        return [_entry_path(args[0])] if args and isinstance(args[0], (str, bytes, os.PathLike)) and os.fsdecode(args[0]) != ":memory:" else [], True
    return [], False


def install_guard(record: list | None = None, allow_root: Path = REVIEW):
    """Refuse every write-type event outside ``allow_root`` (fail closed); record watched events into ``record``."""
    allow = str(allow_root)
    depth = {"n": 0}

    def hook(event, args):
        if event not in WATCH_EVENTS or depth["n"]:
            return
        depth["n"] += 1
        try:
            paths, writing = _guard_paths(event, args)
            if record is not None:
                shown = []
                for a in args:
                    shown.append(os.fsdecode(a) if isinstance(a, (str, bytes, os.PathLike)) else (a if isinstance(a, (int, float, type(None), bool)) else repr(a)[:80]))
                record.append({"event": event, "args": shown, "arg_types": [type(a).__name__ for a in args], "cwd": os.getcwd(), "paths": paths, "writing": writing})
            if writing:
                for p in paths:
                    if p is None:
                        continue
                    if p in ("/dev/null",):  # dill opens os.devnull 'r+b' at import (harmless; the launcher allows it too)
                        continue
                    if p == "<unresolved>" or not (p == allow or p.startswith(allow + os.sep)):
                        raise PermissionError(f"review guard: {event} outside scratch: {p}")
        finally:
            depth["n"] -= 1

    sys.addaudithook(hook)
    return hook


HOOK_FUNCTIONS = {"_path", "_inside", "_in_repo", "_temp_state", "write_allowed", "decide", "_hook", "hook_self_test"}
PATH_CONSTANTS = ("UNRESOLVED", "OUTPUTS", "EXPERIMENT_DIR", "STAGE1_FILES", "WRITABLE", "READABLE_OUTPUTS")


def extract_launcher(root: Path, mode: str, *, fake_cwd: str | None = None, source: Path = LAUNCHER):
    """The launcher's hook functions and path constants (OUTPUTS, EXPERIMENT_DIR, STAGE1_FILES, WRITABLE,
    READABLE_OUTPUTS, UNRESOLVED), executed from its own source in a bare namespace with ROOT rebound. ``fake_cwd``
    replaces ``os.getcwd`` inside the extracted functions only (so relative names are judged as the production process
    would, from the repository root, without changing this process's working directory)."""
    src = source.read_text(encoding="utf-8")
    tree = ast.parse(src)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HOOK_FUNCTIONS]
    found = {n.name for n in functions}
    if found != HOOK_FUNCTIONS:
        raise SystemExit(f"extraction found {sorted(found)}, expected {sorted(HOOK_FUNCTIONS)}")
    constants = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) in PATH_CONSTANTS for t in n.targets)]
    if {t.id for n in constants for t in n.targets} != set(PATH_CONSTANTS):
        raise SystemExit("path constants not found")
    os_ns = os
    if fake_cwd is not None:
        os_ns = types.SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
        os_ns.getcwd = lambda: fake_cwd
        os_ns.path = os.path
    events = {"mode": mode, "load_model": 0, "refused": [], "forbidden_reads": [], "forbidden_writes": [], "outputs_reads": [], "sanctioned_runs": 0,
              "sanctioned_forward_calls": 0, "confirm_invocations": 0, "signals": []}
    ns = {"os": os, "fcntl": fcntl, "Path": Path, "ROOT": Path(root).resolve(), "MODE": mode, "events": events}
    exec(compile(ast.Module(body=constants, type_ignores=[]), "launch_confirm_constants", "exec"), ns)
    ns["os"] = os_ns
    exec(compile(ast.Module(body=functions, type_ignores=[]), "launch_confirm_extract", "exec"), ns)
    return ns


def make_live_judge(ns, production: bool, live: list, phase: dict):
    """An audit hook that decides every judged event at event time with the extracted ``decide`` (never raises; it
    records the decision a real hook would have taken)."""
    depth = {"n": 0}

    def judge(event, args):
        if depth["n"] or event not in JUDGED:
            return
        depth["n"] += 1
        try:
            try:
                decision = ns["decide"](event, args, production)
                outcome = "allow" if decision is None else f"REFUSE {decision}"
            except Exception as error:  # an exception in the real hook fails the operation
                outcome = f"REFUSE (exception {type(error).__name__}: {error})"
            if event == "open":
                mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
                writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)))
            else:
                writing = True
            under_outputs = event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)) and "/outputs/" in os.fsdecode(args[0])
            if outcome != "allow" or writing or under_outputs:
                live.append({"phase": phase.get("name"), "event": event,
                             "args": [os.fsdecode(a) if isinstance(a, (str, bytes, os.PathLike)) else a for a in args][:4],
                             "writing": writing, "decision_at_event_time": outcome})
        finally:
            depth["n"] -= 1

    sys.addaudithook(judge)
    return judge
