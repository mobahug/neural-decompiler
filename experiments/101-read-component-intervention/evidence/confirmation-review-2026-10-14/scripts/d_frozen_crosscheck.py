"""Review items 2–5, the frozen functions as a cross-check (weights only): the model is loaded under the review's guard
(every module call and forward entry point refused, a token-sequence guard on any 101 prompt, the model sealed), then
- G6's table: ri.prediction_table recomputed from the committed stage-1 states and the weights, bit for bit against the
  committed table.json;
- G2–G6: ri.run_gates on the saved measurements with the committed states, against the state's gate book and ceilings,
  bit for bit; then each gate's maximum and its margin;
- the primary statistics: ri.frame_values and ri.primary_statistics, against the state's results, bit for bit;
- the secondary records: ri.run_records and ri.secondary_statistics (and the values block), against the state's, bit for
  bit.
No forward pass; nothing written in the repository."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rguard  # noqa: E402  (first)

rguard.self_test()
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = rguard.ROOT
EXP = ROOT / ri.EXPERIMENT_DIR
OUT = rguard.OUT
out = {"checks": {}, "facts": {}, "timings_s": {}}


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"), flush=True)


def diff(a, b, path="", acc=None, limit=12):
    acc = [] if acc is None else acc
    if len(acc) >= limit:
        return acc
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                acc.append(f"{path}/{key}: missing on one side")
            else:
                diff(a[key], b[key], f"{path}/{key}", acc, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc.append(f"{path}: lengths {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", acc, limit)
    elif type(a) is not type(b) or a != b:
        acc.append(f"{path}: {a!r} vs {b!r}")
    return acc


tokenizer = AutoTokenizer.from_pretrained(models.PYTHIA_70M.model_id, revision=models.PYTHIA_70M.revision)
rguard.install_forward_guards(tokenizer)
check("forward_guards_installed", rguard.RECORD["forward_guards_installed"] and rguard.RECORD["forbidden_sequences"] >= 480,
      {"forbidden_sequences": rguard.RECORD["forbidden_sequences"], "entry_points_replaced": len(rguard.RECORD["entry_points_replaced"])})
t0 = time.time()
inputs = ul.load_frozen_inputs(ROOT)
model = models.load_model(models.PYTHIA_70M)
sealed = rguard.seal(model)
check("model_sealed", sealed["instances"] > 400, {"instances": sealed["instances"], "classes": len(sealed["classes"])})
progs = ul.ModelPrograms.from_model(model, inputs)
out["timings_s"]["load"] = time.time() - t0
# the guard must have refused nothing so far (no forward was attempted)
check("no_forward_attempted_during_load", rguard.RECORD["module_call_refusals"] == 0 and not rguard.RECORD["entry_point_refusals"])
lock = json.loads((EXP / "preregistration-lock.json").read_text(encoding="utf-8"))
payload = json.loads((EXP / "confirmation-v1.json").read_text(encoding="utf-8"))
confirmation = ri.confirmation_from_payload(payload, ri.PRODUCTION)
frames = confirmation.frames
frame_ids = [f.frame_id for f in frames]
noun_keys = [progs.nouns.nouns[i].lexical_key for i in progs.scorable]
check("noun_keys", noun_keys == lock["noun_keys"])
geometry = ri.read_geometry(inputs.lock_011)
block = ri.geometry_block(geometry, progs.head, progs.axis_T, frame_ids, 8)
check("G1_and_geometry_against_lock_bitwise", ri.verify_geometry_against_lock(block, lock)["bitwise_equal"], block["g1"])
index = json.loads((EXP / "stage1/states.json").read_text(encoding="utf-8"))
ul.verify_table_index(index, layout=lock["stage1"]["states"]["layout"], meta=lock["stage1"]["states"]["meta"])
locked = ri.states_from_blocks(ul.read_table(EXP / "stage1/states.f64", index), frames)
state = json.loads((ROOT / "outputs/experiment-101/results.json").read_text(encoding="utf-8"))
check("stage1_state_digests", {fid: rd.state_digest(entry) for fid, entry in sorted(locked.items())} == state["stage1"]["state_digests"])
states = {f.frame_id: rd.state_from_locked(locked[f.frame_id], f) for f in frames}
committed = json.loads((EXP / "stage1/table.json").read_text(encoding="utf-8"))

t0 = time.time()
table = ri.prediction_table(progs, frames, states, geometry, ri.PRODUCTION, noun_keys)
out["timings_s"]["table"] = time.time() - t0
check("G6_table_recomputed_bit_for_bit", pm.canonical_json(table) == pm.canonical_json(committed), diff(table, committed))

tensors = torch.load(ROOT / "outputs/experiment-101/stage2-measurements.pt")
schedules = [confirmation.schedule(f) for f in frames]
ri.assert_measurements(tensors, schedules)
check("assert_measurements", True)
logits = state["confirmation"]["stage2"]["logits_sha256"]
t0 = time.time()
book, ceilings = ri.run_gates(progs, schedules, tensors, logits, states, states, geometry, ri.PRODUCTION)
out["timings_s"]["gates"] = time.time() - t0
gates = book.to_json()
check("gate_book_bit_for_bit_with_state", pm.canonical_json(gates) == pm.canonical_json(state["confirmation"]["gates"]), diff(gates, state["confirmation"]["gates"]))
check("ceilings_bit_for_bit_with_state", pm.canonical_json(rc.json_safe(ceilings)) == pm.canonical_json(state["confirmation"]["ceilings"]))
check("no_gate_failure", book.failures() == [], book.failures())
margins = {}
for name, entry in gates["tolerance"].items():
    margins[name] = {"max": entry["max"], "tolerance": entry["tolerance"], "margin": entry["tolerance"] / entry["max"] if entry["max"] else math.inf, "at": entry["at"]}
out["facts"]["gate_margins"] = margins
out["facts"]["bitwise"] = gates["bitwise"]
out["facts"]["recorded"] = gates["recorded"]
print(json.dumps(margins, indent=1))

# the primary statistics through the frozen functions
values = [ri.frame_values(schedule, tensors, ri.PRODUCTION, fractions_for_weak=False) for schedule in schedules]
values_by_cue = {label: [entry[label] for entry in values] for label in ri.PRODUCTION.cue_labels}
predicted = [committed["frames"][fid]["cues"]["two"] for fid in frame_ids]
tau = lock["calibration_101"]
results = ri.primary_statistics(frame_ids, values_by_cue["two"], predicted, float(tau["tau_rem"]), float(tau["tau_ins"]), ri.PRODUCTION)
check("primary_results_bit_for_bit_with_state", pm.canonical_json(results) == pm.canonical_json(state["confirmation"]["results"]),
      diff(results, state["confirmation"]["results"]))
# the secondary records through the frozen functions
sec = state["confirmation"]["secondary"]
vals = ri.encode_nonfinite({label: list(entries) for label, entries in values_by_cue.items()})
check("secondary_values_bit_for_bit", pm.canonical_json(vals) == pm.canonical_json(sec["values"]), diff(vals, sec["values"]))
t0 = time.time()
records = ri.run_records(progs, schedules, tensors, geometry, ri.PRODUCTION)
check("secondary_records_bit_for_bit", pm.canonical_json(records) == pm.canonical_json(sec["records"]), diff(records, sec["records"]))
stats = ri.secondary_statistics(frame_ids, values_by_cue, results, sec["records"], ri.PRODUCTION)
out["timings_s"]["secondary"] = time.time() - t0
check("secondary_statistics_bit_for_bit", pm.canonical_json(stats) == pm.canonical_json(sec["statistics"]), diff(stats, sec["statistics"]))
# the ceiling errors per arm (recorded, not gated): readout fed the measured edited Δx3, against the measured Δc
ce = {}
for schedule in schedules:
    fid = schedule[0].frame.frame_id
    con = tensors[f"{fid}/contrasts"].double()
    index_ = {run.tag: row for row, run in enumerate(schedule)}
    c_ref = con[index_["REF"]]
    for tag, value in ceilings[fid].items():
        measured = float((con[index_[tag]] - c_ref).mean())
        ce.setdefault(tag, []).append(value - measured)
out["facts"]["ceiling_minus_measured_mean_dc"] = {tag: {"max_abs": max(abs(v) for v in vs), "median": sorted(vs)[len(vs) // 2]} for tag, vs in sorted(ce.items())}
print(json.dumps({k: v for k, v in out["facts"]["ceiling_minus_measured_mean_dc"].items() if k.endswith("@two")}, indent=1))
check("guard_never_fired", rguard.RECORD["module_call_refusals"] == 0 and not rguard.RECORD["entry_point_refusals"] and not rguard.RECORD["token_guard_hits"],
      {"module_call_refusals": rguard.RECORD["module_call_refusals"], "entry_point_refusals": rguard.RECORD["entry_point_refusals"][:3],
       "token_guard_hits": rguard.RECORD["token_guard_hits"][:3]})
(OUT / "d_frozen_crosscheck.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
print("failed:", [k for k, v in out["checks"].items() if not v["ok"]])
