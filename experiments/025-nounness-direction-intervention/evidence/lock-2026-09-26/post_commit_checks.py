"""Experiment 025: read-only checks after the lock-install commit (no model, no forward, no write). Run from the
repository root (asserted)."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import models
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler.provenance import collect_git_state

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
os.chdir(ROOT)
assert Path.cwd().resolve() == ROOT
refused: list[str] = []


def refuse(name):
    def call(*args, **kwargs):
        refused.append(name)
        raise RuntimeError(f"{name} refused in a read-only check")

    return call


torch.nn.Module.__call__ = refuse("torch.nn.Module.__call__")
models.load_model = refuse("load_model")
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    setattr(pm, name, refuse(f"pm.{name}"))


def git(*args, text=True):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=text).stdout


spec = importlib.util.spec_from_file_location("experiment_025_runner", ROOT / f"{cr.EXPERIMENT_DIR}/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
logs: list[str] = []
runner = runner_module.Runner(log=logs.append)

OUT = ROOT / "outputs/experiment-025"
lock_bytes = git("show", f"HEAD:{cr.LOCK_RELATIVE_PATH}", text=False)
prereg_bytes = git("show", f"HEAD:{cr.PREREGISTRATION_RELATIVE_PATH}", text=False)
lock = json.loads(lock_bytes)
state = rd.load_results_state(OUT / "results.json")
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
placeholder = cr.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
git_state = collect_git_state()  # the runner's own provenance call, from the asserted repository root
try:
    cr.validate_lock(lock, state=state, digests=base.digests, config=cr.PRODUCTION, confirmation=confirmation, confirmation_file_sha256=confirmation_sha,
                     dependencies=placeholder, noun_keys=runner._noun_keys(base.inputs), preregistration_text=prereg_bytes.decode("utf-8"), git_state=git_state,
                     tracked=all(runner.tracked(ROOT / p) for p in (cr.LOCK_RELATIVE_PATH, cr.PREREGISTRATION_RELATIVE_PATH)),
                     changed_paths=runner.changed_paths(lock["protocol_code_commit"]))
    full_validation = "passed"
except Exception as error:  # recorded
    full_validation = f"FAILED: {type(error).__name__}: {error}"
config = lock["configuration"]
checks = {
    "cwd_is_repo_root": Path.cwd().resolve() == ROOT,
    "commit_touches_exactly_the_two_paths": sorted(git("show", "--name-only", "--format=", "HEAD").split()) == sorted([cr.LOCK_RELATIVE_PATH, cr.PREREGISTRATION_RELATIVE_PATH]),
    "committed_lock_equals_candidate_bytes": lock_bytes == (OUT / "candidate-lock.json").read_bytes() == (ROOT / cr.LOCK_RELATIVE_PATH).read_bytes(),
    "committed_prereg_equals_candidate_bytes": prereg_bytes == (OUT / "candidate-preregistration.md").read_bytes() == (ROOT / cr.PREREGISTRATION_RELATIVE_PATH).read_bytes(),
    "lock_file_sha256": hashlib.sha256(lock_bytes).hexdigest() == "1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be",
    "lock_content_sha256": lock["content_sha256"] == "8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb" == rc.content_digest(lock),
    "prereg_sha256": hashlib.sha256(prereg_bytes).hexdigest() == "e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919",
    "installed_lock_renders_the_committed_prereg": cr.render_preregistration(lock).encode("utf-8") == prereg_bytes,
    "stock_validate_passes": runner.validate() == 0,
    "full_lock_validation_passes": full_validation == "passed",
    "git_state_clean_at_head": git_state == {"commit": git("rev-parse", "HEAD").strip(), "dirty": False},
    "results_state_unchanged": state["state_sha256"] == "8eeb8184faf556211e68579d4682c85f545c877051641afe07dd5eac2b701dad" and state["executed_prompt_keys"] == [],
    "binding_direction": lock["geometry"]["direction_sha256"] == "5f914283c39d2f37911ba3bb1391a98bbe3625737f8df80c42eb336868a1bf2a",
    "binding_neutral32_max": lock["geometry"]["check_maxima"]["neutral32"] == 6.088945903037768e-09 <= lock["geometry"]["tolerances"]["neutral32"] == 1e-6,
    "binding_manifest": lock["confirmation_025"]["manifest_sha256"] == "0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61" and lock["confirmation_025"]["counts"]["runs"] == 90720,
    "binding_threshold_and_tail": config["count_threshold"] == 27 and config["reference_tail"]["value"] == 0.01923865414210013,
    "binding_level1_on_the_16": config["outcome_bearing"] == ["noun+0.32", "noun-0.32", *[f"rand{j}{s}0.32" for j in range(1, 8) for s in "+-"]] and lock["tolerances"]["level1"] == 2e-2,
    "binding_outcome_hierarchy": lock["outcome"]["table"] == [list(row) for row in cr.OUTCOME_TABLE] and lock["outcome"]["labels"] == list(cr.OUTCOMES),
    "binding_D_attn_reference": lock["dependencies"]["readout_020"]["exposed_states_sha256"] == "26c21d6386ebfaf45cd33f1e9ceaae08a225d5c4ae3a5a9476e28bf1683b0f83",
    "nothing_refused_was_reached": not refused,
}
print(json.dumps({"head": git("rev-parse", "HEAD").strip(), "full_lock_validation": full_validation, "checks": checks, "all": all(checks.values()), "validate_log": logs[-1:]}, indent=1))
