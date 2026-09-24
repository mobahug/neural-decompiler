"""Run the stock Experiment 023 runner's `freeze` phase once, unmodified, inside refusals that make any model use
impossible: an audit hook refuses opening model-weight files, Experiment 022's calibration table and the exposed-cells
numbers; torch.nn.Module.__call__ and neural_decompiler.models.load_model refuse. Every file opened under the repo or the
Hugging Face cache is logged."""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
LOG = Path(sys.argv[1])
HF = Path.home() / ".cache/huggingface"
BLOCKED_FILES = {str((ROOT / "outputs/experiment-022/calibration-table.pt").resolve()), str((ROOT / "experiments/023-block0-completion/exposed-cells.f64").resolve())}
opened, refused = [], []


def hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    raw = os.fsdecode(args[0])
    try:
        path = str(Path(raw).resolve())
    except OSError:
        path = raw
    name = os.path.basename(path)
    if path in BLOCKED_FILES or name.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack")):
        refused.append(path)
        raise PermissionError(f"refused during freeze: {path}")
    if (path.startswith(str(ROOT)) and "/.venv/" not in path) or path.startswith(str(HF.resolve())):
        opened.append(path)


sys.addaudithook(hook)
import torch  # noqa: E402


def refuse_call(self, *args, **kwargs):
    raise RuntimeError(f"a torch module was called during freeze: {type(self).__name__}")


torch.nn.Module.__call__ = refuse_call
from neural_decompiler import models  # noqa: E402


def refuse_load(*args, **kwargs):
    raise RuntimeError("a model load was attempted during freeze")


models.load_model = refuse_load  # patched before the runner imports it
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
assert run023.Runner.model_loader is refuse_load
try:  # the refusals are live
    open(ROOT / "experiments/023-block0-completion/exposed-cells.f64", "rb")
    raise SystemExit("guard not live")
except PermissionError:
    pass
refused.clear()
code = run023.main(["freeze"])
LOG.write_text(json.dumps({"exit": code, "opened": sorted(set(opened)), "refused": refused}, indent=1))
sys.exit(code)
