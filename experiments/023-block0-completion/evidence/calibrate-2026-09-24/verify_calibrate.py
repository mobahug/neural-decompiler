"""Experiment 023 post-calibrate verification: strictly read-only (no model; the 022 table is not needed and not read)."""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy
import torch

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HEAD = "4e5deebd021bc284143b01b76b45efd28e1a1344"
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


out = ROOT / "outputs/experiment-023"
state = rd.load_results_state(out / "results.json")
phase = state["phases"]["calibrate"]
cal = state["calibration"]
print("state_sha256", state["state_sha256"], "| results.json file", rc.file_sha256(out / "results.json"))
check("calibrate complete once at HEAD, no incident/stop", phase["status"] == "complete" and phase["commit"] == HEAD and "incidents" not in cal and "stop" not in cal,
      f"{phase['started_at']} → {phase['completed_at']}")
check("lock/confirm/report not_started; ledgers empty", all(state["phases"][p]["status"] == "not_started" for p in ("lock", "confirm", "report"))
      and state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
binding = {"path": b0c.CONFIRMATION_RELATIVE_PATH, "file_sha256": "5fadfa503f4fe35308cb4473220a1d9cd6f7a46e3852f638594b8081909825f4",
           "content_sha256": "4e64d4c2c85ae8f9c710171372a4bf28f0964a8e8864a0326182edba44aa14ed"}
check("state binds the committed confirmation file", state["confirmation_023"] == binding)
cells_files = {"data_path": b0c.CELLS_DATA_RELATIVE_PATH, "index_path": b0c.CELLS_INDEX_RELATIVE_PATH,
               "data_sha256": "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4", "index_sha256": "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe"}
check("state binds the committed exposed cells", cal["exposed_cells"] == cells_files)

path = out / "candidate-calibration.json"
raw = path.read_bytes()
record = json.loads(raw)
check("candidate record sha256 == state", hashlib.sha256(raw).hexdigest() == cal["record_sha256"], cal["record_sha256"])
check("record content digest", record["content_sha256"] == rc.content_digest(record) == cal["record_content_sha256"], record["content_sha256"])
b0c.verify_calibration_record(record)  # the module's B = 10,000: constants, finite envelopes, direction checks, guard flags, below the stop
check("verify_calibration_record (strict, module B)", True)
check("record bindings: inputs, cells, confirmation, run, commit", record["inputs"] == state["inputs"] and record["exposed_cells"] == cells_files
      and record["confirmation_023"] == binding and record["run_id"] == state["run_id"] and record["protocol_code_commit"] == HEAD)
inputs = ul.load_frozen_inputs(ROOT)
units = b0c.exposed_units(inputs)
indices = b0c.draw_indices(units, b0c.B)
check("draw index digests reproduce (SHA formula, tag 023|primary)", b0c.draw_index_digests(indices) == record["draws"]["index_sha256"] == cal["draw_index_sha256"])
print("pools", json.dumps(record["pools"]), "| kernel check", json.dumps(record["kernel_check"]), "| E6 max", record["e6_max"], "| undefined", json.dumps(record["undefined_counts"]))
arrays = torch.load(out / "draw-values.pt")
check("draw arrays == record digests", {k: rc.tensor_digest(v) for k, v in arrays.items()} == record["draw_arrays_sha256"] == cal["draw_arrays_sha256"])

# Recompute each envelope, median and rates from the saved draw values; spot-check g against an independent numpy pooling.
cells = torch.from_numpy(numpy.frombuffer((ROOT / b0c.CELLS_DATA_RELATIVE_PATH).read_bytes(), dtype="<f8").reshape(18_900, 8).copy())
print("\n| condition | F = v(250) | min | median | max | undefined | guard-bound | PASS | ENV_ONLY | GUARD | NOT_INTERP | gap median | R²₁ median | R²_C median |")
for condition in b0c.CONDITIONS:
    entry = record["conditions"][condition]
    g, defined = arrays[f"{condition}/g"], arrays[f"{condition}/defined"].bool()
    ranked = numpy.sort(numpy.where(defined.numpy(), g.numpy(), -numpy.inf))
    bound = float(ranked[249])
    median = float(numpy.median(g.numpy()[defined.numpy()]))
    codes = b0c.result_codes(g, defined, bound)
    rates = {name: int((codes == i).sum()) / b0c.B for i, name in enumerate(b0c.RESULTS)}
    check(f"{condition}: envelope = element [249] of the saved draws", bound == entry["envelope"]["bound"] and entry["envelope"]["rank"] == 250, f"{bound!r}")
    check(f"{condition}: median, rates, guard flag, direction", math.isclose(median, entry["summary"]["median"], rel_tol=0, abs_tol=1e-15) and rates == entry["rates"]
          and entry["guard_bound"] == (bound < 0.90) and entry["direction_check"]["ok"] and bound <= median)
    population, group = condition.split("/")
    pairs = b0c.draw_pairs(units, indices, population, group, [0, 1, 4999, 9999])
    for row, b in enumerate((0, 1, 4999, 9999)):
        sel = cells[pairs[row]].numpy()
        n, s, q = sel[:, 0].sum(), sel[:, 1].sum(), sel[:, 2].sum()
        sst = q - s * s / n
        sse0, sse1, ssec = sel[:, 3].sum(), sel[:, 4].sum(), sel[:, 5].sum()
        g_np = (sse0 - sse1) / (sse0 - ssec)
        if not abs(g_np - float(g[b])) <= 1e-12 * max(1.0, abs(g_np)):
            check(f"{condition} draw {b}: independent numpy g", False, f"{g_np} vs {float(g[b])}")
    s = entry["summary"]
    print(f"| {condition} | {bound:.6f} | {s['min']:.6f} | {s['median']:.6f} | {s['max']:.6f} | {record['undefined_counts'][condition]} | "
          f"{'yes' if entry['guard_bound'] else 'no'} | {rates['PASS']:.4f} | {rates['ENVELOPE_ONLY_FAILURE']:.4f} | {rates['GUARD_FAILURE']:.4f} | "
          f"{rates['NOT_INTERPRETABLE']:.4f} | {s['gap_median']:.4f} | {s['R2_1_median']:.4f} | {s['R2_C_median']:.4f} |")
print("joint rate, all four pass (descriptive):", record["joint_rates"])
check("spot g (4 draws × 4 conditions) == independent numpy pooling", True)
check("HEAD unchanged, tree clean, nothing installed", git("rev-parse", "HEAD") == HEAD and git("status", "--porcelain", "--untracked-files=all") == ""
      and not (ROOT / b0c.CALIBRATION_RELATIVE_PATH).exists())
print("\nVERIFY CALIBRATE:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
