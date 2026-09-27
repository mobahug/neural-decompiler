"""Shared guard for the Experiment 025 confirmation review (read-only).

Imported first by every review script. It enforces, by execution:
- no write, delete, rename, link, mkdir, truncate or chmod anywhere under the repository (the .venv included);
- no read of outputs/experiment-023/, outputs/experiment-024/ or 022's calibration-table.pt;
- no subprocess except `git --no-optional-locks ...`;
- after `seal()`: torch.nn.Module.__call__, every capture and intervention entry point (and every alias of them in the
  neural_decompiler modules) refuse; `seal_model(model)` also refuses every module class's forward and the bridge's
  run_with_hooks / run_with_cache / generate.
Writes are allowed only under this review directory (and nothing else is expected).
"""
import os
import sys

ROOT = os.path.realpath("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HERE = os.path.dirname(os.path.realpath(__file__))
FORBIDDEN_READ = ("/outputs/experiment-023/", "/outputs/experiment-024/", "calibration-table.pt")
EVENTS = {"refused": [], "writes_here": 0, "writes_elsewhere": []}


def _real(value, dir_fd=None):
    if isinstance(value, int) or value is None:
        return None
    try:
        raw = os.fsdecode(value)
    except TypeError:
        return None
    if dir_fd is not None and not os.path.isabs(raw):
        return None  # relative-to-fd paths are refused below if they could be in the repo
    return os.path.realpath(raw)


def _in_repo(path):
    return path == ROOT or path.startswith(ROOT + os.sep)


def _refuse(kind, detail):
    EVENTS["refused"].append(f"{kind}: {detail}")
    raise PermissionError(f"review guard: {kind}: {detail}")


def _hook(event, args):
    if event == "open":
        path = _real(args[0]) if args else None
        if path is None:
            return
        if any(token in path for token in FORBIDDEN_READ):
            _refuse("forbidden read", path)
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)))
        if writing:
            if _in_repo(path):
                _refuse("repository write", path)
            if path == HERE or path.startswith(HERE + os.sep):
                EVENTS["writes_here"] += 1
            else:
                EVENTS["writes_elsewhere"].append(path)
        return
    if event in ("os.remove", "os.rmdir", "os.mkdir", "os.truncate", "shutil.rmtree", "os.chmod", "os.chown", "os.utime", "os.chflags", "os.lchflags",
                 "os.setxattr", "os.removexattr"):
        path = _real(args[0]) if args else None
        if path is None or _in_repo(path):
            if path is None and event not in ("os.truncate", "os.chmod", "os.utime", "os.chown"):
                _refuse(event, repr(args[:1]))
            if path is not None:
                _refuse(event, path)
        return
    if event in ("os.rename", "os.link", "os.symlink"):
        paths = [_real(a) for a in args[:2]]
        if any(p is None or _in_repo(p) for p in paths):
            _refuse(event, repr(paths))
        return
    if event == "subprocess.Popen":
        executable, argv = args[0], list(args[1] or [])
        name = os.path.basename(os.fsdecode(argv[0])) if argv else os.path.basename(os.fsdecode(executable or ""))
        if name != "git" or len(argv) < 2 or argv[1] != "--no-optional-locks":
            _refuse("subprocess", repr(argv))
        return
    if event in ("os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty"):
        _refuse(event, repr(args)[:200])


sys.addaudithook(_hook)


def _refuser(name):
    def refuse(*args, **kwargs):
        _refuse("refused call", name)

    refuse.__name__ = f"refused_{name.replace('.', '_')}"
    return refuse


def seal():
    """Refuse torch.nn.Module.__call__ and every capture / intervention entry point and alias."""
    import torch
    from neural_decompiler import capture, interventions
    from neural_decompiler import plural_mechanism as pm

    torch.nn.Module.__call__ = _refuser("torch.nn.Module.__call__")
    originals = {"capture.run_capture": capture.run_capture, "interventions.run_interventions": interventions.run_interventions,
                 "pm.capture_prompt": pm.capture_prompt, "pm.run_patched": pm.run_patched}
    by_id = {id(fn): name for name, fn in originals.items()}
    replaced = []
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("neural_decompiler"):
            continue
        for attribute, value in list(vars(module).items()):
            if id(value) in by_id:
                setattr(module, attribute, _refuser(f"{module_name}.{attribute}"))
                replaced.append(f"{module_name}.{attribute}")
    return replaced


def seal_model(model):
    """Every module class's forward, and the bridge's hook/cache/generate routes, refuse."""
    sealed = []
    for cls in {type(m) for m in model.modules()}:
        for method in ("forward", "run_with_hooks", "run_with_cache", "generate"):
            if method in cls.__dict__ and callable(cls.__dict__[method]):
                setattr(cls, method, _refuser(f"{cls.__qualname__}.{method}"))
                sealed.append(f"{cls.__qualname__}.{method}")
    for name, module in model.named_modules():
        if "forward" in vars(module):
            object.__setattr__(module, "forward", _refuser(f"{name}.forward(instance)"))
            sealed.append(f"{name}.forward(instance)")
    return sorted(sealed)


def refuse_functions(module, names, label):
    """Temporarily refuse named functions of a module; returns the originals for a later restore."""
    saved = {name: getattr(module, name) for name in names}
    for name in names:
        setattr(module, name, _refuser(f"{label}.{name}"))
    return saved


def restore_functions(module, saved):
    for name, value in saved.items():
        setattr(module, name, value)


def git(*args):
    import subprocess

    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def tracked(path) -> bool:
    import subprocess

    try:
        subprocess.run(["git", "--no-optional-locks", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return False
    return True
