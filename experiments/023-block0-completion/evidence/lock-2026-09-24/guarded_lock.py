"""Run the stock Experiment 023 runner's `lock` phase once, unmodified. Lock may read the weights (no forward pass):
the model loads normally, and from the moment it is loaded every torch.nn.Module.__call__ refuses, so no forward pass can
run; the runner's own guard also disables every capture entry point while predicting. An audit hook refuses any open of
Experiment 022's calibration table and logs every file opened under the repo or the Hugging Face cache."""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
LOG = Path(sys.argv[1])
HF = (Path.home() / ".cache/huggingface").resolve()
TABLE = (ROOT / "outputs/experiment-022/calibration-table.pt").resolve()
TABLE_INODE = os.stat(TABLE).st_ino
opened, refused, loads = [], [], []


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
        raise PermissionError(f"refused during lock: {path}")
    text = str(path)
    if (text.startswith(str(ROOT)) and "/.venv/" not in text and "/__pycache__/" not in text) or text.startswith(str(HF)):
        opened.append(text)


sys.addaudithook(hook)
import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402

original_load = models.load_model


def refuse_call(self, *args, **kwargs):
    raise RuntimeError(f"a forward pass was attempted during lock: {type(self).__name__}")


def load_then_refuse_forward(spec):
    model = original_load(spec)
    loads.append(spec.model_id)
    torch.nn.Module.__call__ = refuse_call  # from here on no module can be called
    return model


models.load_model = load_then_refuse_forward
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
assert run023.Runner.model_loader is load_then_refuse_forward
try:
    open(TABLE, "rb")
    raise SystemExit("guard not live")
except PermissionError:
    refused.clear()
code = run023.main(["lock"])
LOG.write_text(json.dumps({"exit": code, "model_loads": loads, "forward_refusal_installed": torch.nn.Module.__call__ is refuse_call, "opened": sorted(set(opened)),
                           "refused_during_run": refused}, indent=1))
sys.exit(code)
