"""Aggregate the gate workers (item 5; the ladder of item 12; the I1 discrimination of items 3 and 11).

Reads r4_worker*.pt; compares with the recorded gates, the recorded ladder and the operator's per-condition Level-1
claims. Writes r6_aggregate.json here.
"""
import guard  # noqa: F401

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(guard.ROOT)
HERE = Path(guard.HERE)
RUNS = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm025")
state = json.loads((ROOT / "outputs/experiment-025/results.json").read_text())
lock = json.loads((ROOT / "experiments/025-nounness-direction-intervention/preregistration-lock.json").read_text())
gates = state["confirmation"]["gates"]
ladder_rec = state["confirmation"]["descriptives"]["ladder"]
operator = json.loads((RUNS / "verify_confirm.json").read_text())
BEARING = list(lock["configuration"]["outcome_bearing"])
CONDITIONS = list(lock["configuration"]["conditions"])
records, swaps, summaries = [], [], []
for path in sorted(HERE.glob("r4_worker*.pt")):
    blob = torch.load(path, weights_only=True)
    records += blob["records"]
    swaps += blob["swap"]
    summaries.append(blob["summary"])
out = {"workers": [{k: v for k, v in s.items() if k != "frames"} for s in summaries]}
keys = Counter((r["frame"], r["token_id"], r["cond"]) for r in records)
out["coverage"] = {"runs": len(records), "unique": len(keys), "duplicates": sum(c - 1 for c in keys.values() if c > 1), "frames": len({r["frame"] for r in records}),
                   "cues": len({r["token_id"] for r in records}), "per_condition": dict(Counter(r["cond"] for r in records))}


def where(r):
    return f"{r['word']}|{r['frame']}|{r['cond']}"


def worst(rs, name):
    best = max(rs, key=lambda r: r[name])
    return {"max": best[name], "at": where(best)}


out["C"] = {"all_bitwise_equal": all(r["c_equal"] for r in records), "not_equal": sum(not r["c_equal"] for r in records), "max_abs_diff": max(r["c_diff"] for r in records),
            "runs": len(records), "outcome_bearing_runs": sum(r["cond"] in BEARING for r in records)}
bear = [r for r in records if r["cond"] in BEARING]
sec = [r for r in records if r["cond"] not in BEARING]
out["level1"] = {"outcome_bearing": worst(bear, "level1"), "secondary": worst(sec, "level1"), "tolerance": gates["tolerances"]["level1_outcome_bearing"],
                 "recorded_outcome_bearing": gates["level1_outcome_bearing"], "recorded_secondary": gates["level1_secondary"], "outcome_bearing_runs": len(bear)}
out["level1"]["equal_recorded"] = out["level1"]["outcome_bearing"] == gates["level1_outcome_bearing"] and out["level1"]["secondary"] == gates["level1_secondary"]
per_cond = {}
for c in CONDITIONS:
    rs = [r for r in records if r["cond"] == c]
    per_cond[c] = {"level1": worst(rs, "level1"), "I1": worst(rs, "I1"), "I3": worst(rs, "I3"), "I4": worst(rs, "I4"), "runs": len(rs)}
out["per_condition"] = per_cond
op = operator["level1_per_condition"]
out["level1_per_condition_vs_operator_max_abs_diff"] = max(abs(per_cond[c]["level1"]["max"] - op[c]["max"]) for c in CONDITIONS)
out["level1_per_condition_location_equal_operator"] = all(per_cond[c]["level1"]["at"].rsplit("|", 1)[0] == op[c]["at"] for c in CONDITIONS)
out["level1_outcome_bearing_per_condition_all_within_tolerance"] = all(per_cond[c]["level1"]["max"] <= 0.02 for c in BEARING)
for name, tol in (("I1", 1e-4), ("I3", 1e-4), ("I4", 1e-3)):
    w = worst(records, name)
    out[name] = {**w, "tolerance": tol, "recorded": gates[name], "equal_recorded_value": w["max"] == gates[name]["max"], "equal_recorded_location": w["at"] == gates[name]["at"],
                 "abs_diff": abs(w["max"] - gates[name]["max"]), "within_tolerance": w["max"] <= tol}
# the I1 discrimination: the same saved x1 against a deliberately wrong vector
disc = defaultdict(lambda: {"n": 0, "min": math.inf})
for s in swaps:
    for key in (f"{s['slot']} <- {s['vector']}", f"{s['slot']} <- {s['vector']} [{s['stratum']}]"):
        disc[key]["n"] += 1
        disc[key]["min"] = min(disc[key]["min"], s["I1_wrong"])
out["I1_wrong_vector"] = dict(disc)
out["I1_wrong_vector_min_over_all"] = min(s["I1_wrong"] for s in swaps)
out["I1_margin_ratio"] = out["I1_wrong_vector_min_over_all"] / out["I1"]["max"]
# the ladder (descriptive): pooled over every pair
lad = {}
for c in ("base", "noun+0.32", "noun-0.32"):
    rs = [r for r in records if r["cond"] == c]
    tot = {k: math.fsum(r[f"sse_{k}"] for r in rs) for k in ("C", "C5", "L1")}
    lad[c] = {"sse": tot, "block5_attention": (tot["C"] - tot["C5"]) / tot["C"], "block4_attention": (tot["C5"] - tot["L1"]) / tot["C"], "pairs": len(rs)}
    r_ = ladder_rec[c]
    lad[c]["max_abs_diff_vs_recorded"] = max(abs(lad[c]["block5_attention"] - r_["block5_attention"]), abs(lad[c]["block4_attention"] - r_["block4_attention"]),
                                             *(abs(tot[k] - r_["sse"][k]) / r_["sse"][k] for k in ("C", "C5", "L1")))
    lad[c]["recorded"] = r_
out["ladder"] = lad
out["guard"] = guard.EVENTS
(HERE / "r6_aggregate.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k not in ("per_condition", "I1_wrong_vector")}, indent=1))
print(json.dumps({k: v for k, v in out["I1_wrong_vector"].items() if "[" not in k}, indent=1))
for c in CONDITIONS:
    p = per_cond[c]
    print(f"{c:11s} L1 {p['level1']['max']:.6f} ({p['level1']['at']})  I1 {p['I1']['max']:.3e}  I3 {p['I3']['max']:.3e}  I4 {p['I4']['max']:.3e}  runs {p['runs']}")
