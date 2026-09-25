"""Read-only guard for the independent lock review of Experiment 024.

Importing this module installs a Python audit hook that refuses (and records) every write, create, rename, remove,
mkdir, chmod or copy under the repository root, and every read of Experiment 022's local calibration table or of
anything under outputs/experiment-023/ (024's leakage boundary). Writes elsewhere (the system temp dir, this scratch
directory) are allowed. Also usable as a pytest plugin (``-p review_guard``).

``ForwardGuard`` counts torch.nn.Module.__call__ while the weights load and refuses it afterwards, refuses every
capture / intervention / measurement entry point, and after the load replaces ``forward`` on every module class the
model contains with a refusing stub (so neither ``module(...)`` nor ``module.forward(...)`` can run).
"""
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
ROOT_TEXT = str(ROOT)
EVENTS = {"refused_writes": [], "refused_reads": [], "opens_under_root_for_write": 0}
_WRITE_EVENTS = {"os.remove", "os.unlink", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.truncate", "os.link", "os.symlink",
                 "shutil.rmtree", "shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.copytree", "shutil.move", "os.utime",
                 "os.chflags", "os.lchflags", "os.lchmod", "os.setxattr", "os.removexattr"}


def _fd_path(fd):
    """The directory an open descriptor refers to (macOS F_GETPATH); None if unknown."""
    try:
        import fcntl

        buffer = fcntl.fcntl(fd, getattr(fcntl, "F_GETPATH", 50), bytes(1024))
        return os.fsdecode(buffer.split(b"\0", 1)[0])
    except Exception:
        return None


def _under_root(value, dir_fd=None) -> bool:
    if isinstance(value, int) or value is None:
        return False
    try:
        text = os.fsdecode(value)
    except TypeError:
        return False
    if dir_fd is not None and isinstance(dir_fd, int) and dir_fd >= 0 and not os.path.isabs(text):
        base = _fd_path(dir_fd)
        if base is None:
            return True  # unknown directory: conservative
        text = os.path.join(base, text)
    try:
        resolved = str(Path(text).resolve())
    except (OSError, RuntimeError):
        resolved = os.path.abspath(text)
    return resolved == ROOT_TEXT or resolved.startswith(ROOT_TEXT + os.sep)


def _forbidden_read(value) -> bool:
    if isinstance(value, int) or value is None:
        return False
    try:
        text = str(Path(os.fsdecode(value)).resolve())
    except Exception:
        return False
    if not (text == ROOT_TEXT or text.startswith(ROOT_TEXT + os.sep)):
        return False  # only the real repository's local outputs are off limits (a fake world's copies are not)
    return text.endswith("calibration-table.pt") or "/outputs/experiment-023/" in text


def _hook(event, args):
    if event == "open":
        if not args:
            return
        path = args[0]
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)))
        if writing and _under_root(path):
            EVENTS["refused_writes"].append(f"{event}:{os.fsdecode(path)}")
            raise PermissionError(f"review guard: refusing a write under the repository: {path}")
        if _forbidden_read(path):
            EVENTS["refused_reads"].append(os.fsdecode(path))
            raise PermissionError(f"review guard: refusing to open {path} (024 leakage boundary)")
    elif event in ("os.rename", "os.replace", "os.link", "os.symlink"):
        src_fd = args[2] if len(args) > 2 and isinstance(args[2], int) else None
        dst_fd = args[3] if len(args) > 3 and isinstance(args[3], int) else None
        if event == "os.symlink":
            flagged = _under_root(args[1], args[2] if len(args) > 2 and isinstance(args[2], int) else None)
        else:
            flagged = _under_root(args[0], src_fd) or _under_root(args[1], dst_fd)
        if flagged:
            EVENTS["refused_writes"].append(f"{event}:{args[:2]}")
            raise PermissionError(f"review guard: refusing {event} under the repository: {args[:2]}")
    elif event in _WRITE_EVENTS:
        if not args:
            return
        if event in ("shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.copytree"):
            flagged = len(args) > 1 and _under_root(args[1])  # the destination is the write; reading the source is allowed
        elif event == "shutil.move":
            flagged = _under_root(args[0]) or (len(args) > 1 and _under_root(args[1]))
        else:
            dir_fd = args[-1] if event in ("os.remove", "os.unlink", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.utime", "shutil.rmtree") and isinstance(args[-1], int) else None
            flagged = _under_root(args[0], dir_fd)
        if flagged:
            EVENTS["refused_writes"].append(f"{event}:{args[:2]}")
            raise PermissionError(f"review guard: refusing {event} under the repository: {args[:2]}")


sys.addaudithook(_hook)


class ForwardGuard:
    CAPTURE_NAMES = ("capture_prompt", "run_patched", "run_capture", "run_interventions")

    def __init__(self):
        import torch

        self.torch = torch
        self.counts = {"module_call_during_load": 0, "module_call_refused": 0, "forward_refused": 0, "capture_refused": 0, "load_model": 0}
        self.sealed = False
        self._original_call = torch.nn.Module.__call__
        guard = self

        def guarded_call(module, *args, **kwargs):
            if not guard.sealed:
                guard.counts["module_call_during_load"] += 1
                return guard._original_call(module, *args, **kwargs)
            guard.counts["module_call_refused"] += 1
            raise RuntimeError("review guard: a torch module was called after the weights were read")

        torch.nn.Module.__call__ = guarded_call
        from neural_decompiler import plural_mechanism as pm
        from neural_decompiler import upstream_localization as ul

        def refuse_capture(*args, **kwargs):
            guard.counts["capture_refused"] += 1
            raise RuntimeError("review guard: a capture / intervention / measurement entry point was reached")

        for name in self.CAPTURE_NAMES:
            if hasattr(pm, name):
                setattr(pm, name, refuse_capture)
        for name in ("measure_prompt", "stage_two_022"):
            setattr(ul, name, refuse_capture)

    def load(self):
        from neural_decompiler import models

        self.counts["load_model"] += 1
        model = models.load_model(models.PYTHIA_70M)
        self.seal(model)
        return model

    def seal(self, model):
        self.sealed = True
        guard = self

        def refuse_forward(*args, **kwargs):
            guard.counts["forward_refused"] += 1
            raise RuntimeError("review guard: forward refused after the weights were read")

        classes = {type(module) for module in model.modules()} | {type(model)}
        for cls in classes:
            try:
                cls.forward = refuse_forward
            except (TypeError, AttributeError):
                pass
        self.sealed_classes = sorted(cls.__name__ for cls in classes)


def report_guard():
    return dict(EVENTS)
