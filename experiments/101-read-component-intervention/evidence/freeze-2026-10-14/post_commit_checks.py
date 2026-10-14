"""Experiment 101 freeze: the checks after the install commit (read-only). The model, every forward and every geometry,
prediction or outcome function are refused while the runner's `validate` runs."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HERE = Path(__file__).resolve().parent
PATH = "experiments/101-read-component-intervention/confirmation-v1.json"

import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402

refused = []


def refuse(name):
    def inner(*args, **kwargs):
        refused.append(name)
        raise RuntimeError(f"{name} refused in the post-commit checks")

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


record = json.loads((HERE / "freeze_run.json").read_text())
install = git("rev-parse", "HEAD")
committed = git_bytes("show", f"HEAD:{PATH}")
working = (ROOT / PATH).read_bytes()
payload = json.loads(committed)
changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", install).splitlines()
spec = importlib.util.spec_from_file_location("experiment_101_runner", ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
logs = []
runner = runner_module.Runner(log=logs.append)
validate_status = runner.validate()
base = runner._base()
confirmation, sha = runner._confirmation(base)
keys = set(confirmation.s1_keys()) | set(confirmation.s2_keys())
untagged = confirmation.untagged_keys()
checks = {
    "install_commit_parent_is_freeze_head": git("rev-parse", "HEAD~1") == record["environment"]["head"],
    "install_commit_touches_exactly_the_file": changed == [PATH],
    "committed_bytes_equal_working_file": committed == working,
    "file_sha256": hashlib.sha256(committed).hexdigest() == record["post_write"]["file_sha256"],
    "content_sha256": payload["content_sha256"] == rc.content_digest(payload) == record["post_write"]["content_sha256"],
    "manifest_sha256": payload["manifest_sha256"] == pm.sha256_text(pm.canonical_json(payload["manifest"])) == record["facts"]["manifest_sha256"],
    "runner_validate_exit_0": validate_status == 0,
    "runner_confirmation_sha_equal": sha == record["post_write"]["file_sha256"],
    "keys_2016_distinct": len(keys) == 2016,
    "untagged_144": len(untagged) == 144,
    "zero_collisions_with_forbidden": not (keys | untagged) & base.forbidden,
    "tree_clean_after_install": git("status", "--porcelain", "--untracked-files=all") == "",
    "nothing_refused": refused == [],
}
print(json.dumps({"install_commit": install, "checks": checks, "all_true": all(checks.values()), "validate_log": logs, "forbidden_keys": len(base.forbidden),
                  "refused": refused}, indent=1))
sys.exit(0 if all(checks.values()) else 1)
