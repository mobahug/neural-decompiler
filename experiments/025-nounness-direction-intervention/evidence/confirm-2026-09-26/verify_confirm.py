"""Experiment 025: read-only post-confirm verification. No forward, and nothing in the repository is written.

The model is loaded for its weights only: module calls and every capture/intervention entry point refuse. It checks:
- the results state's digest, and the durable measurement file's sha256, per-tensor digests and shapes;
- the run log's accounting: duplicates, missing, extra, and per-condition counts;
- A, B, G, the counts and the label, recomputed from the saved measurements in this separate process and compared
  with the recorded result;
- the Level-1 identity's maximum per condition (the 16 outcome-bearing ones, and the others), compared with the
  recorded maxima;
- `C` recomputed spot-wise.
"""
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

from neural_decompiler import capture, interventions, models
from neural_decompiler import cue_rotation as cr
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import upstream_localization as ul

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)
OUT = ROOT / "outputs/experiment-025"
refused: list[str] = []


def refusal(name):
    def refuse(*args, **kwargs):
        refused.append(name)
        raise RuntimeError(f"{name} refused in the post-confirm verification")

    return refuse


torch.nn.Module.__call__ = refusal("torch.nn.Module.__call__")
for module, names in ((pm, ("capture_prompt", "run_patched", "run_capture", "run_interventions")), (capture, ("run_capture",)), (interventions, ("run_interventions",))):
    for name in names:
        setattr(module, name, refusal(f"{module.__name__}.{name}"))


def git(*args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


spec = importlib.util.spec_from_file_location("experiment_025_runner", ROOT / f"{cr.EXPERIMENT_DIR}/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
runner = runner_module.Runner(log=lambda message: None)
base = runner._base()
confirmation, _ = runner._confirmation(base)
state = rd.load_results_state(OUT / "results.json")  # verifies the state digest
conf = state["confirmation"]
result = conf["results"]
out: dict = {"head": git("rev-parse", "HEAD"), "origin": git("rev-parse", "origin/main"), "remote": git("ls-remote", "origin", "refs/heads/main").split()[0],
             "tree": git("status", "--porcelain", "--untracked-files=all"), "state_sha256": state["state_sha256"], "results_file_sha256": sha256_file(OUT / "results.json")}
out["phases"] = {name: entry.get("status") for name, entry in state["phases"].items()}
out["incident"] = conf.get("incident")
out["phase_incidents"] = {name: entry.get("incidents") for name, entry in state["phases"].items() if entry.get("incidents")}

# The durable measurements.
stage2 = OUT / "stage2-measurements.pt"
out["stage2"] = {"path": str(stage2), "bytes": stage2.stat().st_size, "sha256": sha256_file(stage2)}
saved = torch.load(stage2)
out["stage2"]["tensor_digests_equal_state"] = {key: rc.tensor_digest(value) for key, value in saved.items()} == conf["stage2"]["tensors_sha256"]
out["stage2"]["tensors"] = len(saved)
units = cr.target_units(confirmation)
cr.assert_measurements(saved, units, confirmation.conditions, 79)
out["stage2"]["complete_and_finite"] = True

# The run log.
log_lines = [json.loads(line) for line in (HERE / "production_runs.jsonl").read_text().splitlines()]
keys = [line["key"] for line in log_lines]
tagged = confirmation.tagged_keys()
counts = Counter(keys)
out["run_log"] = {"lines": len(keys), "unique": len(counts), "duplicates": sum(c - 1 for c in counts.values() if c > 1), "missing": len(tagged - set(counts)),
                  "extra": len(set(counts) - tagged), "sequence_numbers_contiguous": [line["n"] for line in log_lines] == list(range(1, len(log_lines) + 1)),
                  "per_condition": dict(sorted(Counter(key.rsplit("|", 1)[1] for key in keys).items())), "ledger_equals_manifest": set(state["executed_prompt_keys"]) == tagged}

# A, B, G recomputed from the saved measurements in this process.
states = ul.y1_states(base.inputs.closure["exploration"]["locked_states"], units.frames)
values = cr.per_cue(units, saved, states, confirmation.conditions)
again = cr.statistics(values, confirmation, cr.PRODUCTION)
out["recomputed_result_equals_recorded"] = json.loads(json.dumps(again)) == result
out["recomputed_counts"] = {name: again[name]["positive"] for name in ("A", "B", "G")}

# The Level-1 identity per condition, and C spot checks, from the weights (no forward).
model = models.load_model(models.PYTHIA_70M)
progs = ul.ModelPrograms.from_model(model, base.inputs)
level1 = {condition: 0.0 for condition in confirmation.conditions}
level1_at = {condition: "" for condition in confirmation.conditions}
c_spot = 0.0
for group, pairs in units.pairs.items():
    for row, (t, f) in enumerate(pairs):
        frame = units.frames[f]
        state_f = states[frame.frame_id]
        for condition in confirmation.conditions:
            dx3 = cr.changes_from_raw(saved[cr.tensor_name(group, condition, "x3")][row], state_f.x3_all, frame)
            value = progs.readout.contrast(state_f, progs.readout.level1_detail(state_f, dict(dx3), l4_heads=True)["dh6"], progs.nouns)[progs.scorable].double()
            deviation = float((value - saved[cr.tensor_name(group, condition, "dc")][row]).abs().max())
            if deviation > level1[condition]:
                level1[condition], level1_at[condition] = deviation, f"{units.tokens[t]['word']}|{frame.frame_id}"
        if row % 97 == 0:  # C spot checks, bit for bit, on a deterministic subset
            for condition in confirmation.conditions:
                again_c = ul.contrast_of(progs, state_f, cr.changes_from_raw(saved[cr.tensor_name(group, condition, "x3")][row], state_f.x3_all, frame))
                c_spot = max(c_spot, float((again_c - saved[cr.tensor_name(group, condition, "C")][row]).abs().max()))
bearing = cr.PRODUCTION.outcome_bearing
out["level1_per_condition"] = {condition: {"max": level1[condition], "at": level1_at[condition], "outcome_bearing": condition in bearing} for condition in confirmation.conditions}
out["level1_outcome_bearing_max"] = max(level1[c] for c in bearing)
out["level1_secondary_max"] = max(level1[c] for c in confirmation.conditions if c not in bearing)
out["level1_max_equals_recorded"] = out["level1_outcome_bearing_max"] == conf["gates"]["level1_outcome_bearing"]["max"] and out["level1_secondary_max"] == conf["gates"]["level1_secondary"]["max"]
out["c_spot_max_abs_difference"] = c_spot
out["refused"] = refused
(HERE / "verify_confirm.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
print(json.dumps({k: out[k] for k in ("head", "state_sha256", "incident", "recomputed_result_equals_recorded", "recomputed_counts", "level1_outcome_bearing_max",
                                       "level1_max_equals_recorded", "c_spot_max_abs_difference", "refused")}, default=str), flush=True)
