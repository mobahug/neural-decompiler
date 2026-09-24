"""Process guard for the independent Experiment 023 confirm review (read-only).

Installed FIRST in every review process (exec'd before any other import):
  * a sys.addaudithook refusing any "open" of outputs/experiment-022/calibration-table.pt, matched by resolved path,
    by (st_dev, st_ino) and by basename;
  * torch.load patched to refuse that path (resolved path or inode);
  * a live proof of both (the refusals are exercised and the proof is printed).
After the (optional) model load, `forbid_forward()` replaces torch.nn.Module.__call__ and plural_mechanism's
capture_prompt / run_capture / run_patched / run_interventions with functions that raise.
"""
import os
import sys

ROOT = "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler"
TABLE_022 = os.path.join(ROOT, "outputs/experiment-022/calibration-table.pt")
TABLE_022_RESOLVED = os.path.realpath(TABLE_022)
_st = os.stat(TABLE_022_RESOLVED)  # os.stat raises no "open" audit event; the file itself is never opened
TABLE_022_ID = (_st.st_dev, _st.st_ino)
REFUSED = []


def _is_table(target):
    if isinstance(target, int):  # a file descriptor: only reachable through an earlier (refused) open of a path
        return False
    try:
        raw = os.fsdecode(target)
    except TypeError:
        return False
    if os.path.basename(raw) == "calibration-table.pt":
        return True
    try:
        if os.path.realpath(raw) == TABLE_022_RESOLVED:
            return True
    except (OSError, ValueError):
        pass
    try:
        st = os.stat(raw)
        return (st.st_dev, st.st_ino) == TABLE_022_ID
    except (OSError, ValueError):
        return False


def _audit(event, args):
    if event == "open" and args and _is_table(args[0]):
        REFUSED.append(os.fsdecode(args[0]))
        raise PermissionError(f"review guard: refused to open Experiment 022's calibration table ({args[0]!r})")


sys.addaudithook(_audit)

import torch  # noqa: E402

_torch_load = torch.load


def _guarded_load(f, *args, **kwargs):
    if isinstance(f, (str, bytes, os.PathLike)) and _is_table(f):
        REFUSED.append(os.fsdecode(f))
        raise PermissionError(f"review guard: torch.load refused Experiment 022's calibration table ({f!r})")
    return _torch_load(f, *args, **kwargs)


torch.load = _guarded_load


def prove_live():
    """Exercise every refusal route; each must raise PermissionError. Returns a one-line proof."""
    routes = {
        "open(path)": lambda: open(TABLE_022, "rb"),
        "open(resolved)": lambda: open(TABLE_022_RESOLVED, "rb"),
        "open(dotted path)": lambda: open(os.path.join(ROOT, "outputs/experiment-022/../experiment-022/calibration-table.pt"), "rb"),
        "os.open": lambda: os.open(TABLE_022, os.O_RDONLY),
        "torch.load": lambda: torch.load(TABLE_022),
        "pathlib.read_bytes": lambda: __import__("pathlib").Path(TABLE_022).read_bytes(),
    }
    outcome = {}
    for name, route in routes.items():
        try:
            handle = route()
        except PermissionError:
            outcome[name] = "refused"
        else:
            try:
                if isinstance(handle, int):
                    os.close(handle)
                elif hasattr(handle, "close"):
                    handle.close()
            finally:
                raise SystemExit(f"GUARD NOT LIVE: {name} was not refused")
    n = len(REFUSED)
    REFUSED.clear()
    return f"guard live: {len(outcome)} routes refused ({', '.join(outcome)}); {n} refusals recorded and cleared; table inode {TABLE_022_ID[1]}"


def forbid_forward():
    """After the weights are extracted: no nn.Module call and no capture/intervention entry point can run."""
    def _refuse_module_call(self, *args, **kwargs):
        raise RuntimeError(f"review guard: a forward pass ({type(self).__name__}.__call__) was reached")

    torch.nn.Module.__call__ = _refuse_module_call
    from neural_decompiler import plural_mechanism as pm
    patched = []
    for name in ("capture_prompt", "run_capture", "run_patched", "run_interventions"):
        if hasattr(pm, name):
            def _refuse(*args, _name=name, **kwargs):
                raise RuntimeError(f"review guard: plural_mechanism.{_name} was reached")
            setattr(pm, name, _refuse)
            patched.append(name)
    # prove it live
    try:
        torch.nn.Linear(2, 2)(torch.zeros(2))
        raise SystemExit("FORWARD GUARD NOT LIVE: nn.Module.__call__")
    except RuntimeError:
        pass
    for name in patched:
        try:
            getattr(pm, name)(None, None, None)
            raise SystemExit(f"FORWARD GUARD NOT LIVE: {name}")
        except RuntimeError:
            pass
    return f"forward guard live: nn.Module.__call__ and pm.{', pm.'.join(patched)} raise"


print(prove_live(), flush=True)
