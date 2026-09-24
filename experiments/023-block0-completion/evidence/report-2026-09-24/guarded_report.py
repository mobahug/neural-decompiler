"""Run the stock Experiment 023 runner's `report` phase once, unmodified: the model loader and every module call
refuse, and an audit hook refuses any open of Experiment 022's calibration table."""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
SUMMARY = Path(sys.argv[1])
TABLE = (ROOT / "outputs/experiment-022/calibration-table.pt").resolve()
TABLE_INODE = os.stat(TABLE).st_ino
refused = []


def hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    raw = os.fsdecode(args[0])
    try:
        path = Path(raw).resolve()
        inode = os.stat(path).st_ino if path.exists() else None
    except OSError:
        path, inode = Path(raw), None
    if path == TABLE or inode == TABLE_INODE or os.path.basename(raw) == "calibration-table.pt":
        refused.append(str(path))
        raise PermissionError(f"refused during report: {path}")


sys.addaudithook(hook)
import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402


def refuse(*args, **kwargs):
    raise RuntimeError("model use attempted during report")


torch.nn.Module.__call__ = refuse
models.load_model = refuse
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
try:
    open(TABLE, "rb")
    raise SystemExit("guard not live")
except PermissionError:
    refused.clear()
code = run023.main(["report"])
SUMMARY.write_text(json.dumps({"exit": code, "refused_during_run": refused}))
sys.exit(code)
