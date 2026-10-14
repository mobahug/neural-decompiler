"""Shared helpers for the stage-1 launcher re-check (AI reviewer's own scripts; scratch only).

- ``install_guard()``: an audit hook for MY test processes that refuses any write-type event outside this re-check's
  scratch directory (fail closed; the parent directory is resolved and the final component kept, so a symlink's own
  entry is judged, not its target) and records every event of interest.
- ``extract_launcher()``: the launcher's hook functions, extracted from its source with ``ast`` (as the operator's
  ``hook_self_test.py`` does) and executed in a bare namespace whose ROOT is a caller-given root (a mirror tree in
  scratch, or the real repository path for pure decisions only).
"""
from __future__ import annotations

import ast
import fcntl
import json
import os
import sys
import types
from pathlib import Path

RECHECK = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/stage1-launcher-recheck").resolve()
OUT = RECHECK / "out"
REPO = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
LAUNCHER_V2 = REPO / "experiments/101-read-component-intervention/evidence/lock-2026-10-14/prepared/launch_stage1.py"
LAUNCHER_V2_SHA256 = "13b1992f7f0feb652207ca835df044244f20faea750e92f42d8d51bced7c3e1c"

WRITE_EVENTS = {"os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.link", "os.symlink", "os.chmod", "os.chown", "os.utime", "os.truncate",
                "os.chflags", "os.setxattr", "os.removexattr", "shutil.rmtree", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
                "shutil.copytree", "shutil.move", "shutil.make_archive", "shutil.unpack_archive", "tempfile.mkstemp", "tempfile.mkdtemp",
                "sqlite3.connect", "os.mkfifo", "os.mknod"}
WATCH_EVENTS = WRITE_EVENTS | {"open", "os.chdir", "subprocess.Popen", "os.posix_spawn", "os.system", "os.exec", "os.fork", "os.putenv", "os.unsetenv",
                               "fcntl.fcntl", "os.listdir", "os.scandir"}


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


def install_guard(record: list | None = None, allow_root: Path = RECHECK):
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
                        raise PermissionError(f"recheck guard: {event} outside scratch: {p}")
        finally:
            depth["n"] -= 1

    sys.addaudithook(hook)
    return hook


def extract_launcher(root: Path, mode: str, *, fake_cwd: str | None = None, source: Path = LAUNCHER_V2):
    """The launcher's hook functions, executed in a bare namespace. ``fake_cwd`` replaces ``os.getcwd`` inside the
    extracted functions only (so relative names can be judged as the production process would, from the repository
    root, without changing this process's working directory)."""
    src = source.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {"_path", "_inside", "_in_repo", "_temp_state", "write_allowed", "decide", "_hook", "hook_self_test", "write_check"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "UNRESOLVED" for t in n.targets)]
    found = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    if found != names:
        raise SystemExit(f"extraction found {sorted(found)}, expected {sorted(names)}")
    os_ns = os
    if fake_cwd is not None:
        os_ns = types.SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
        os_ns.getcwd = lambda: fake_cwd
        os_ns.path = os.path
    root = Path(root).resolve()
    outputs = (root / "outputs/experiment-101").resolve()
    experiment = (root / "experiments/101-read-component-intervention").resolve()
    stage1 = experiment / "stage1"
    events = {"mode": mode, "load_model": 0, "refused": [], "forbidden_reads": [], "forbidden_writes": [], "output_writes": [], "output_renames": [],
              "stage1_writes": [], "stage1_mkdirs": 0, "outputs_reads": [], "sanctioned_runs": 0, "sanctioned_forward_calls": 0, "stage1_invocations": 0}

    class PhaseError(RuntimeError):
        pass

    ns = {"os": os_ns, "fcntl": fcntl, "Path": Path, "ROOT": root, "OUTPUTS": outputs, "EXPERIMENT_DIR": experiment, "STAGE1_DIR": stage1,
          "STAGE1_FILES": {stage1 / "states.f64", stage1 / "states.json", stage1 / "table.json"}, "MODE": mode,
          "READABLE_OUTPUTS": (root / "outputs/experiment-020/results.json", outputs), "events": events,
          "ri": types.SimpleNamespace(PhaseError=PhaseError)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch_stage1_extract", "exec"), ns)
    return ns
