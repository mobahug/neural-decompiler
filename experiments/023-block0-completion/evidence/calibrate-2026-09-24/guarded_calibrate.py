"""Run the stock Experiment 023 runner's `calibrate` phase once, unmodified, inside refusals: an audit hook refuses any
open of Experiment 022's calibration table and of the pinned model's weight blob (checked on the raw and the resolved
path, and by inode); torch.nn.Module.__call__ and neural_decompiler.models.load_model refuse. Every file opened under the
repo or the Hugging Face cache is logged."""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
LOG = Path(sys.argv[1])
HF = (Path.home() / ".cache/huggingface").resolve()
WEIGHTS = next((HF / "hub/models--EleutherAI--pythia-70m-deduped/snapshots").glob("*/model.safetensors")).resolve()
BLOCKED = {str((ROOT / "outputs/experiment-022/calibration-table.pt").resolve()), str(WEIGHTS)}
INODES = {os.stat(path).st_ino for path in BLOCKED}
opened, refused = [], []


def hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    raw = os.fsdecode(args[0])
    try:
        path = str(Path(raw).resolve())
        inode = os.stat(path).st_ino if os.path.exists(path) else None
    except OSError:
        path, inode = raw, None
    if path in BLOCKED or inode in INODES or os.path.basename(raw) in ("model.safetensors", "pytorch_model.bin", "calibration-table.pt"):
        refused.append(path)
        raise PermissionError(f"refused during calibrate: {path}")
    if (path.startswith(str(ROOT)) and "/.venv/" not in path and "/__pycache__/" not in path) or path.startswith(str(HF)):
        opened.append(path)


sys.addaudithook(hook)
import torch  # noqa: E402


def refuse_call(self, *args, **kwargs):
    raise RuntimeError(f"a torch module was called during calibrate: {type(self).__name__}")


torch.nn.Module.__call__ = refuse_call
from neural_decompiler import models  # noqa: E402


def refuse_load(*args, **kwargs):
    raise RuntimeError("a model load was attempted during calibrate")


models.load_model = refuse_load
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
assert run023.Runner.model_loader is refuse_load
probes = []
for target in sorted(BLOCKED):  # the refusals are live
    try:
        open(target, "rb")
        raise SystemExit(f"guard not live for {target}")
    except PermissionError:
        probes.append(target)
refused.clear()
code = run023.main(["calibrate"])
LOG.write_text(json.dumps({"exit": code, "probes_refused": probes, "opened": sorted(set(opened)), "refused_during_run": refused}, indent=1))
sys.exit(code)
