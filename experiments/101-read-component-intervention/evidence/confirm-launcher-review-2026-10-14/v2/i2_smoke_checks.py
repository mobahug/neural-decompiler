"""v2's two new smoke checks evaluated without running anything (no model is loaded): the reference file's sha256
against SMOKE_REFERENCE_SHA256 as the v2 source states it, and the reference's 55 run keys against the first exposed
frame's calibration schedule built by the runner's own (pure) methods. Under my guard; GIT_OPTIONAL_LOCKS=0."""
import ast
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import REPO, REVIEW, install_guard  # noqa: E402

assert os.environ.get("GIT_OPTIONAL_LOCKS") == "0"
install_guard()
src = (REVIEW / "reviewed_v2/launch_confirm_v2.py").read_text()
constant = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "SMOKE_REFERENCE_SHA256").value.value
reference_path = REVIEW.parent / "calibrate/smoke_reference.pt"  # = HERE.parent / "calibrate" / "smoke_reference.pt" for HERE = exp101/confirm
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment_101_runner_review", REPO / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
runner = runner_module.Runner(log=lambda m: None)
base = runner._base()
frames = runner._exposed_frames(base.inputs)
schedule = runner._calibration_schedules(base.inputs, frames)[0]
reference = torch.load(reference_path)
print("sha256 check passes:", hashlib.sha256(reference_path.read_bytes()).hexdigest() == constant, constant[:16])
print("keys check passes:", set(reference["keys"][1:]) == {run.key for run in schedule}, len(schedule), schedule[0].frame.frame_id)
print("order also equal:", list(reference["keys"][1:]) == [run.key for run in schedule])
