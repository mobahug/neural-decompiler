"""Experiment 023 post-confirm verification: strictly read-only (no model, no prompt, writes nothing in the repo)."""
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy
import torch

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HEAD = "c7efec7f32709dc20ecb83cae16bb73927a14366"
D = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm023")
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


summary = json.loads((D / "confirm_summary.json").read_text())
print("launcher summary:", json.dumps(summary))
runner = run023.Runner()
inputs, confirmation_022, sha_022, digests, forbidden = runner._base()
confirmation, confirmation_sha = runner._confirmation(inputs, confirmation_022, sha_022, forbidden)
manifest = confirmation.manifest()
stage1_keys = manifest["S1-REF"] + manifest["S1-VALIDITY"]
target_keys = manifest["S2-TARGET"]["Y1"] + manifest["S2-TARGET"]["Y2"]
manifest_keys = set(stage1_keys) | set(target_keys)

# Prompt accounting: expected manifest = executed prompts = ledger spend = 3,060 unique keys
executed = [json.loads(line) for line in (D / "prompts.jsonl").read_text().splitlines()]
executed_keys = [entry["key"] for entry in executed]
counter = Counter(executed_keys)
state = rd.load_results_state(runner.results_path)
ledger = state["executed_prompt_keys"]
print(f"executed forwards (capture_prompt calls) {len(executed_keys)}; unique {len(counter)}; run_capture {summary['counts']['run_capture']}; "
      f"run_patched {summary['counts'].get('run_patched')}; run_interventions {summary['counts'].get('run_interventions')}")
check("manifest: 3,060 unique keys", len(stage1_keys) + len(target_keys) == len(manifest_keys) == 3060)
check("executed = manifest: every key exactly once, no extra, no omission", counter == Counter({key: 1 for key in manifest_keys}),
      f"extra {sorted(set(counter) - manifest_keys)[:3]}, missing {len(manifest_keys - set(counter))}, duplicated {[k for k, n in counter.items() if n > 1][:3]}")
check("ledger = manifest, no duplicates", len(ledger) == len(set(ledger)) == 3060 and set(ledger) == manifest_keys)
check("one forward per capture; no patched or intervention run", summary["counts"]["run_capture"] == summary["counts"]["capture_prompt"] == 3060
      and summary["counts"].get("run_patched", 0) == 0 and summary["counts"].get("run_interventions", 0) == 0)
first_target = min(i for i, key in enumerate(executed_keys) if key in set(target_keys))
check("all 36 stage-1 prompts ran before any target (the barrier)", set(executed_keys[:first_target]) == set(stage1_keys) and first_target == 36)
check("no 022 table access during confirm", summary["refused_during_run"] == [])

# The phase and its record
phase = state["phases"]["confirm"]
results = state["confirmation"]
check("confirm complete, once, at c7efec7; no incident", phase["status"] == "complete" and phase["confirm_commit"] == HEAD and "incident" not in results
      and not phase.get("incidents"), f"{phase.get('started_at')} → {phase.get('completed_at')}")
lock = json.loads((ROOT / b0c.LOCK_RELATIVE_PATH).read_text())
check("bindings: lock digest in phase, confirmation and stage 1; I7 bitwise", phase["lock_sha256"] == results["lock_sha256"] == results["stage1"]["lock_sha256"]
      == lock["content_sha256"] and phase["I7"]["bitwise_equal"] and results["stage1"]["commit"] == HEAD)
check("stage-1 record digest reproduces", results["stage1"]["digest"] == b0c.stage_one_digest(results["stage1"]))
y2 = b0c.verified_y2_blocks(ROOT, results["stage1"], lock)  # the Y2 table re-read and verified against the stage-1 digests and the lock's layout/meta
check("Y2 table verified from disk", {k: tuple(v.shape) for k, v in y2.items()} == {"cue_final": (288, 2, 79), "coordinated": (144, 2, 79)},
      f"file {results['stage1']['y2_table']['file_sha256']}")
tensors = torch.load(runner.stage2_path)
check("stage-2 measurements == their recorded digests", {k: rc.tensor_digest(v) for k, v in tensors.items()} == results["stage2"]["tensors_sha256"]
      and results["stage2"]["n_executed"] == 3024)
print("gates:", json.dumps(results["gates"]), "| kernel check:", json.dumps(results["kernel_check"]))
check("identities within tolerance (I1 1e-4, I3 1e-4, I4 1e-3)", all(results["gates"][n]["max"] <= b0c.TOLERANCES[n] for n in ("I1", "I3", "I4")))
y1 = ul.read_table(ROOT / b0c.Y1_TABLE_RELATIVE_PATH, json.loads((ROOT / b0c.Y1_TABLE_INDEX_RELATIVE_PATH).read_text()))
tables = {"Y1": y1, "Y2": y2}


def own(y, p0, p1, c, bound):
    y, p0, p1, c = (numpy.asarray(v, dtype=numpy.float64).reshape(-1) for v in (y, p0, p1, c))
    sst = float(((y - y.mean()) ** 2).sum())
    sse0, sse1, ssec = (float(((y - k) ** 2).sum()) for k in (p0, p1, c))
    defined = sst > 0 and sse0 - ssec >= 0.02 * sst
    g = (sse0 - sse1) / (sse0 - ssec) if defined else None
    result = "NOT_INTERPRETABLE" if not defined else "GUARD_FAILURE" if g < 0.90 else "ENVELOPE_ONLY_FAILURE" if g < bound else "PASS"
    return {"g": g, "result": result, "gap": (sse0 - ssec) / sst, "R2_0": 1 - sse0 / sst, "R2_1": 1 - sse1 / sst, "R2_C": 1 - ssec / sst, "pairs": y.size // 79}


print("\n| condition | g | F (lock, exact) | max(F, 0.90) | result | gap | R²₀ | R²₁ | R²_C | pairs |")
for condition in b0c.CONDITIONS:
    population, group = condition.split("/")
    bound = lock["conditions"][condition]["envelope"]["bound"]
    entry = results["conditions"][condition]
    mine = own(tensors[f"{condition}/dc"], tables[population][group][:, 0], tables[population][group][:, 1], tensors[f"{condition}/ceiling"], bound)
    check(f"{condition}: independent recomputation == recorded (g, result, pairs)", mine["result"] == entry["result"] and mine["pairs"] == entry["n_pairs"]
          and (mine["g"] is None) == (entry["g"] is None) and (mine["g"] is None or math.isclose(mine["g"], entry["g"], rel_tol=1e-12, abs_tol=1e-12)))
    check(f"{condition}: envelope == lock == record, guard 0.90", entry["envelope"] == lock["conditions"][condition]["envelope"] and entry["guard"] == {"min_inclusive": 0.9})
    print(f"| {condition} | {entry['g']!r} | {bound!r} | {max(bound, 0.9):.6f} | **{entry['result']}** | {entry['gap']:.4f} | {entry['R2_0']:.4f} | {entry['R2_1']:.4f} | "
          f"{entry['R2_C']:.4f} | {entry['n_pairs']} |")
check("no aggregate label", results["aggregate_label"] is None)

# The frozen " shiny" pair, named; the planned secondary descriptive check without it (Y1/coordinated only)
units = ul.table_units(confirmation.tokens, confirmation.exposed_frames)
rows = [i for i, (t, f) in enumerate(units.pairs["coordinated"]) if units.tokens[t]["word"] == "shiny" and units.frames[f].frame_id == "coordinated-adjective-009-1"]
check("the shiny × coordinated-adjective-009-1 pair is present once in Y1/coordinated", len(rows) == 1, f"row {rows}")
keep = [i for i in range(len(units.pairs["coordinated"])) if i not in rows]
side = own(tensors["Y1/coordinated/dc"][keep], y1["coordinated"][keep, 0], y1["coordinated"][keep, 1], tensors["Y1/coordinated/ceiling"][keep],
           lock["conditions"]["Y1/coordinated"]["envelope"]["bound"])
print(f"SECONDARY (descriptive only; the frozen population is primary): Y1/coordinated without that pair: g {side['g']!r} ({side['pairs']} pairs), "
      f"classification by the same rule: {side['result']}")
descriptives = results.get("descriptives") or {}
print("descriptive records present:", sorted(descriptives), "| failures:", descriptives.get("failures"))
print("state_sha256", state["state_sha256"], "| results.json file", rc.file_sha256(runner.results_path))
check("HEAD unchanged, tree clean, no report yet", git("rev-parse", "HEAD") == HEAD and git("status", "--porcelain", "--untracked-files=all") == ""
      and state["phases"]["report"]["status"] == "not_started" and not runner.report_path.exists())
print("\nVERIFY CONFIRM:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
