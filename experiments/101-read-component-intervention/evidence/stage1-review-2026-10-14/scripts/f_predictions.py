"""Item 5: the program's committed predictions, described (table.json only; no model; no measurement exists to compare)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402  (first)

guard.self_test()
import json  # noqa: E402
import math  # noqa: E402

ROOT = guard.ROOT
EXP = ROOT / "experiments/101-read-component-intervention"
table = json.loads((EXP / "stage1/table.json").read_text(encoding="utf-8"))
calibration = json.loads((EXP / "calibration-v1.json").read_text(encoding="utf-8"))
frames = sorted(table["frames"])
QUANTITIES = ("dc_full", "F_rem", "F_rem_off", "F_ins", "F_ins_off", "N", "S", "A_amp", "A_neg", "F_half_rem", "F_half_ins", "F_swap_pl", "F_swap_ref", "additivity")
PAIR = ("dr", "s", "off_c_norm", "s_over_off_c", "dx_norm")


def median(values):  # design's rule: the mean of the 18th and 19th of 36
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def summary(values):
    v = sorted(values)
    return {"n": len(v), "min": v[0], "p10": v[max(0, math.ceil(0.10 * len(v)) - 1)], "median": median(v), "mean": sum(v) / len(v),
            "p90": v[max(0, math.ceil(0.90 * len(v)) - 1)], "max": v[-1]}


out: dict = {"per_cue": {}, "per_frame_two": [], "nonfinite": 0, "nulls": []}
for cue in ("two", "three", "many"):
    values = {q: [] for q in QUANTITIES}
    pairs = {q: [] for q in PAIR}
    weak, overshoot = [], []
    for fid in frames:
        entry = table["frames"][fid]["cues"][cue]
        values["dc_full"].append(entry["dc_full"])
        if entry["weak"]:
            weak.append(fid)
        if entry["fractions"] is None:
            out["nulls"].append(f"{fid}/{cue}")
        else:
            for q in QUANTITIES[1:]:
                values[q].append(entry["fractions"][q])
        for q in PAIR:
            pairs[q].append(entry["pair"][q])
        if entry["pair"]["overshoot"]:
            overshoot.append(fid)
        for v in (entry["dc_full"], *entry["pair"].values(), *(entry["fractions"] or {}).values()):
            if isinstance(v, float) and not math.isfinite(v):
                out["nonfinite"] += 1
    stats = {q: summary(v) for q, v in values.items() if v}
    counts = {"dc_full_le_minus1": sum(1 for v in values["dc_full"] if v <= -1.0), "N_positive": sum(1 for v in values["N"] if v > 0),
              "S_positive": sum(1 for v in values["S"] if v > 0), "A_amp_negative": sum(1 for v in values["A_amp"] if v < 0),
              "A_neg_positive": sum(1 for v in values["A_neg"] if v > 0), "median_N_ge_0.05": median(values["N"]) >= 0.05, "median_S_ge_0.05": median(values["S"]) >= 0.05}
    out["per_cue"][cue] = {"stats": stats, "pair": {q: summary(v) for q, v in pairs.items()}, "counts": counts, "weak_frames": weak, "overshoot_frames": overshoot}
for fid in frames:
    e = table["frames"][fid]["cues"]["two"]
    out["per_frame_two"].append({"frame": fid[-2:], "p_c": table["frames"][fid]["p_c"], "dc_full": e["dc_full"], **{q: e["fractions"][q] for q in QUANTITIES[1:9]},
                                 "dr": e["pair"]["dr"], "s": e["pair"]["s"], "s_over_off_c": e["pair"]["s_over_off_c"]})
# Context only: the exposed frames' measured real-swap values for `two` (calibration reports; spent data).
reports = calibration["reports"]
exposed = {"dc_full": [r["dc_full"]["two"] for r in reports.values()], "dr": [r["pairs"]["two"]["dr"] for r in reports.values()],
           "s": [r["pairs"]["two"]["s"] for r in reports.values()], "s_over_off_c": [r["pairs"]["two"]["s_over_off_c"] for r in reports.values()]}
out["exposed_measured_two_context"] = {q: summary(v) for q, v in exposed.items()}
out["exposed_measured_N_S_A_context"] = {q: summary(calibration["distributions"][q]) for q in ("N", "S", "A_amp", "A_neg") if calibration["distributions"].get(q)}
out["tau"] = calibration["tau"]
out["guard"] = guard.summary()
(guard.OUT / "f_predictions.json").write_text(json.dumps(out, indent=1) + "\n")


def fmt(x):
    return f"{x:+.4f}"


print("nonfinite:", out["nonfinite"], " null fraction blocks:", out["nulls"])
for cue, block in out["per_cue"].items():
    print(f"\n== {cue}: weak {block['weak_frames']}, overshoot {len(block['overshoot_frames'])}, counts {block['counts']}")
    for q, s in block["stats"].items():
        print(f"  {q:11s} min {fmt(s['min'])} p10 {fmt(s['p10'])} median {fmt(s['median'])} mean {fmt(s['mean'])} p90 {fmt(s['p90'])} max {fmt(s['max'])}")
    for q, s in block["pair"].items():
        print(f"  pair.{q:13s} min {fmt(s['min'])} median {fmt(s['median'])} max {fmt(s['max'])}")
print("\n== exposed measured (two), context:")
for q, s in out["exposed_measured_two_context"].items():
    print(f"  {q:13s} min {fmt(s['min'])} median {fmt(s['median'])} max {fmt(s['max'])}")
for q, s in out["exposed_measured_N_S_A_context"].items():
    print(f"  measured {q:6s} min {fmt(s['min'])} median {fmt(s['median'])} max {fmt(s['max'])}")
