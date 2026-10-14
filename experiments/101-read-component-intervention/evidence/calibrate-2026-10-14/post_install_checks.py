"""Experiment 101 calibrate: the checks after the calibration file's install commit (read-only). The model, every
forward and every capture, measurement and outcome function are refused while the runner's `validate` runs."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
PATH = "experiments/101-read-component-intervention/calibration-v1.json"
CANDIDATE = ROOT / "outputs/experiment-101/candidate-calibration-v1.json"

import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402

refused = []


def refuse(name):
    def inner(*args, **kwargs):
        refused.append(name)
        raise RuntimeError(f"{name} refused in the post-install checks")

    return inner


torch.nn.Module.__call__ = refuse("torch.nn.Module.__call__")
models.load_model = refuse("load_model")
for name in ("capture_prompt", "run_patched"):
    setattr(pm, name, refuse(f"pm.{name}"))
for name in ("read_geometry", "geometry_block", "prediction_table", "run_arm", "capture_reference", "stage_runs", "run_gates", "frame_values",
             "primary_statistics", "secondary_statistics", "calibration_results"):
    setattr(ri, name, refuse(f"ri.{name}"))


def git(*args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def git_bytes(*args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True).stdout


install = git("rev-parse", "HEAD")
committed = git_bytes("show", f"HEAD:{PATH}")
candidate = CANDIDATE.read_bytes()
payload = json.loads(committed)
state = rd.load_results_state(ROOT / "outputs/experiment-101/results.json")
changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", install).splitlines()
spec = importlib.util.spec_from_file_location("experiment_101_runner", ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
logs = []
runner = runner_module.Runner(log=logs.append)
validate_status = runner.validate()
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
loaded, calibration_sha = runner._calibration(confirmation, confirmation_sha)
checks = {
    "install_commit_touches_exactly_the_file": changed == [PATH],
    "committed_bytes_equal_the_candidate": committed == candidate,
    "committed_bytes_equal_working_file": committed == (ROOT / PATH).read_bytes(),
    "file_sha256_is_the_state's": hashlib.sha256(committed).hexdigest() == state["calibration"]["file_sha256"],
    "content_sha256_recomputes_and_is_the_state's": payload["content_sha256"] == rc.content_digest(payload) == state["calibration"]["content_sha256"],
    "state_calibrate_complete": state["phases"]["calibrate"]["status"] == "complete",
    "state_lock_not_started": state["phases"]["lock"] == {"status": "not_started"},
    "no_confirmation_key_in_ledger": state["executed_prompt_keys"] == [],
    "calibration_executed_2016": len(state["calibration_executed"]) == 2016,
    "runner_validate_exit_0": validate_status == 0,
    "runner_loads_the_calibration": loaded["content_sha256"] == payload["content_sha256"] and calibration_sha == hashlib.sha256(committed).hexdigest(),
    "tolerances_are_the_frozen_constants": payload["tolerances"] == json.loads(pm.canonical_json(ri.TOLERANCES)),
    "no_trigger_fired": not any(entry["fired"] for entry in payload["triggers"].values()),
    "tree_clean_after_install": git("status", "--porcelain", "--untracked-files=all") == "",
    "nothing_refused": refused == [],
}
print(json.dumps({"install_commit": install, "checks": checks, "all_true": all(checks.values()), "tau": payload["tau"], "validate_log": logs, "refused": refused},
                 indent=1))
sys.exit(0 if all(checks.values()) else 1)
