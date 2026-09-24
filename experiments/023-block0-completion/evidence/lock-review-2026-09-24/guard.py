"""Process guards for the independent 023 lock review (imported FIRST by every review script).

1. An audit hook refuses any "open" of Experiment 022's calibration table (resolved path, (dev, inode), or basename) and
   logs every file opened under the repo or the Hugging Face cache.
2. torch.load / torch.serialization.load refuse that path (and any file object naming it).
3. prove_live(): deliberate open() and torch.load() attempts must be refused (recorded separately, then cleared).
4. refuse_execution(): torch.nn.Module.__call__ and plural_mechanism's capture_prompt / run_patched / run_capture /
   run_interventions (plus capture.run_capture, interventions.run_interventions) replaced by functions that raise.
"""
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HF = (Path.home() / ".cache/huggingface").resolve()
TABLE = (ROOT / "outputs/experiment-022/calibration-table.pt").resolve()
_st = os.stat(TABLE)  # stat only (no open): the identity to refuse
TABLE_ID = (_st.st_dev, _st.st_ino)
OPENED: list[str] = []
REFUSED: list[str] = []
PROOF: dict = {}
_IN_HOOK = [False]


def _is_table(raw: str) -> bool:
    try:
        path = Path(raw).resolve()
    except Exception:
        path = Path(raw)
    if path == TABLE or os.path.basename(raw) == "calibration-table.pt":
        return True
    try:
        st = os.stat(path)
        return (st.st_dev, st.st_ino) == TABLE_ID
    except OSError:
        return False


def _hook(event, args):
    if event != "open" or not args or _IN_HOOK[0]:
        return
    target = args[0]
    if not isinstance(target, (str, bytes, os.PathLike)):
        return
    _IN_HOOK[0] = True
    try:
        raw = os.fsdecode(target)
        if _is_table(raw):
            REFUSED.append(raw)
            raise PermissionError(f"lock review: refused open of Experiment 022's table: {raw}")
        try:
            text = str(Path(raw).resolve())
        except Exception:
            text = raw
        if (text.startswith(str(ROOT)) and "/.venv/" not in text and "/__pycache__/" not in text) or text.startswith(str(HF)):
            OPENED.append(text)
    finally:
        _IN_HOOK[0] = False


sys.addaudithook(_hook)

import torch  # noqa: E402  (after the hook)

_orig_load = torch.load


def _guarded_load(f, *args, **kwargs):
    name = f if isinstance(f, (str, bytes, os.PathLike)) else getattr(f, "name", None)
    if isinstance(name, (str, bytes, os.PathLike)) and _is_table(os.fsdecode(name)):
        REFUSED.append(f"torch.load:{os.fsdecode(name)}")
        raise PermissionError(f"lock review: torch.load refused for Experiment 022's table: {os.fsdecode(name)}")
    return _orig_load(f, *args, **kwargs)


torch.load = _guarded_load
torch.serialization.load = _guarded_load


def prove_live() -> dict:
    """Deliberate attempts that must be refused; recorded in PROOF, then REFUSED is cleared."""
    out = {}
    try:
        with open(TABLE, "rb") as handle:  # noqa: F841
            out["open"] = "NOT REFUSED"
    except PermissionError as error:
        out["open"] = f"refused: {error}"
    try:
        torch.load(str(ROOT / "outputs/experiment-022/calibration-table.pt"))
        out["torch.load"] = "NOT REFUSED"
    except PermissionError as error:
        out["torch.load"] = f"refused: {error}"
    try:
        os.open(str(TABLE), os.O_RDONLY)
        out["os.open"] = "NOT REFUSED"
    except PermissionError as error:
        out["os.open"] = f"refused: {error}"
    out["refused_entries"] = list(REFUSED)
    ok = all(str(v).startswith("refused") for k, v in out.items() if k != "refused_entries")
    out["live"] = ok
    PROOF.update(out)
    REFUSED.clear()
    if not ok:
        raise SystemExit(f"guard not live: {out}")
    return out


def _refuse_call(self, *args, **kwargs):
    raise RuntimeError(f"lock review: a module call (forward) was attempted: {type(self).__name__}")


def refuse_execution() -> dict:
    """nn.Module.__call__ and every capture / intervention entry point refuse from now on."""
    from neural_decompiler import plural_mechanism as pm
    from neural_decompiler import capture as capture_module
    from neural_decompiler import interventions as interventions_module

    torch.nn.Module.__call__ = _refuse_call
    patched = []

    def refuser(name):
        def refuse(*args, **kwargs):
            raise RuntimeError(f"lock review: {name} was reached where no execution may run")
        return refuse

    for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
        if hasattr(pm, name):
            setattr(pm, name, refuser(f"pm.{name}"))
            patched.append(f"pm.{name}")
    capture_module.run_capture = refuser("capture.run_capture")
    interventions_module.run_interventions = refuser("interventions.run_interventions")
    patched += ["capture.run_capture", "interventions.run_interventions"]
    return {"module_call_refused": torch.nn.Module.__call__ is _refuse_call, "patched": patched}


def execution_refused() -> bool:
    from neural_decompiler import plural_mechanism as pm
    return torch.nn.Module.__call__ is _refuse_call and all(getattr(pm, name).__name__ == "refuse" for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"))


REFUSE_CALL = _refuse_call
