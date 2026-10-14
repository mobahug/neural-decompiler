"""Review item 5 (and 6's numbers), with this review's own code from b_primary's recomputed per-frame values: the
bootstrap intervals (one shared [10,000 × 36] index from torch.Generator().manual_seed(20261001), the even-n median
rule, nearest ranks 250 and 9,750), the stratum, the isotropic controls, the head arms and shares, the direction-free
summaries and the program's errors, against the state's secondary statistics. Then the interpretive numbers: the read
part's share of the cue change at p_c, the post-LayerNorm sizes of the read, off-read and isotropic edits, and the
program's predicted pair quantities against the measured ones. No model, no neural_decompiler import."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rguard  # noqa: E402  (first)

rguard.self_test()
import json  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402

ROOT = rguard.ROOT
EXP = ROOT / "experiments/101-read-component-intervention"
OUT = rguard.OUT
out = {"checks": {}, "facts": {}}


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"), flush=True)


state = json.loads((ROOT / "outputs/experiment-101/results.json").read_text(encoding="utf-8"))
lock = json.loads((EXP / "preregistration-lock.json").read_text(encoding="utf-8"))
calib = json.loads((EXP / "calibration-v1.json").read_text(encoding="utf-8"))
table = json.loads((EXP / "stage1/table.json").read_text(encoding="utf-8"))
per_frame = json.loads((OUT / "b_primary_values.json").read_text(encoding="utf-8"))["per_frame"]
sec = state["confirmation"]["secondary"]["statistics"]
frames = [e["frame_id"] for e in per_frame]
check("tau_lock_equals_calibration_file", (lock["calibration_101"]["tau_rem"], lock["calibration_101"]["tau_ins"]) == (calib["tau"]["rem"], calib["tau"]["ins"]),
      calib["tau"])


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    if n % 2:
        return s[n // 2]
    lo, hi = s[n // 2 - 1], s[n // 2]
    if math.isinf(lo) and math.isinf(hi) and lo != hi:
        return math.nan
    if math.isinf(lo):
        return lo
    if math.isinf(hi):
        return hi
    return 0.5 * (lo + hi)


gen = torch.Generator().manual_seed(20261001)
INDEX = torch.randint(0, 36, (10000, 36), generator=gen)


def interval(values):
    """Own implementation: each resample's median by the rule above (None excluded), sorted; ranks 250 and 9,750."""
    meds = []
    for row in INDEX.tolist():
        sample = [values[i] for i in row if values[i] is not None]
        meds.append(median(sample) if sample else math.nan)
    meds.sort()
    return [meds[249], meds[9749]]


def summarize(values):
    defined = [v for v in values if v is not None]
    return {"median": median(defined), "interval": interval(values), "n": len(defined)}


def same(mine, theirs, name):
    ok = (mine["median"] == theirs["median"] and mine["n"] == theirs["n"]
          and all((a == b) or (abs(a - b) <= 1e-12 * max(1.0, abs(b))) for a, b in zip(mine["interval"], theirs["interval"])))
    exact = mine["interval"] == theirs["interval"]
    return ok, exact


bad, inexact = [], []
two = [e["two"] for e in per_frame]
for name in ("N", "S", "A_amp", "A_neg"):
    ok, exact = same(summarize([e[name] for e in two]), sec["primary_medians"][name], name)
    (bad if not ok else inexact if not exact else []).append(f"primary/{name}")
for cue in ("three", "many"):
    vals = [e[cue] for e in per_frame]
    mine_counts = {"R_count": sum(e["N"] > 0 for e in vals), "I_count": sum(e["S"] > 0 for e in vals), "A_amp_negative": sum(e["A_amp"] < 0 for e in vals),
                   "A_neg_positive": sum(e["A_neg"] > 0 for e in vals), "weak": sum(e["weak"] for e in vals)}
    theirs = {k: sec["stratum"][cue][k] for k in mine_counts}
    if mine_counts != theirs:
        bad.append(f"stratum/{cue}/counts {mine_counts} vs {theirs}")
    out["facts"][f"stratum_{cue}"] = {**mine_counts, **{f"median_{k}": median([e[k] for e in vals]) for k in ("N", "S", "A_amp", "A_neg", "F_rem", "F_ins", "F_rem_off",
                                                                                                             "F_ins_off", "dc_full", "F_swap_pl", "F_swap_ref")}}
    for name in ("N", "S", "A_amp", "A_neg"):
        ok, exact = same(summarize([e[name] for e in vals]), sec["stratum"][cue]["medians"][name], name)
        (bad if not ok else inexact if not exact else []).append(f"stratum/{cue}/{name}")
    for name in ("F_half_rem", "F_half_ins", "F_swap_pl", "F_swap_ref", "additivity"):
        ok, exact = same(summarize([e[name] for e in vals]), sec["stratum"][cue]["direction_free"][name], name)
        (bad if not ok else inexact if not exact else []).append(f"stratum/{cue}/{name}")
for name in ("iso_rem_margin", "iso_ins_margin"):
    series = [e[name] for e in two]
    ok, exact = same(summarize(series), sec["isotropic"][name], name)
    (bad if not ok else inexact if not exact else []).append(f"iso/{name}")
    if sum(v > 0 for v in series) != sec["isotropic"][name]["positive"]:
        bad.append(f"iso/{name}/positive")
for name, sign, ref in (("F_head_rem", -1, 1.0), ("F_head_rem_v", -1, 1.0), ("F_head_ins", 1, 0.0), ("F_head_ins_v", 1, 0.0)):
    series = [e[name] for e in two]
    ok, exact = same(summarize(series), sec["head"][name], name)
    (bad if not ok else inexact if not exact else []).append(f"head/{name}")
    if sum((v - ref) * sign > 0 for v in series) != sec["head"][name]["count"]:
        bad.append(f"head/{name}/count")
for name in ("head_removal_share", "head_removal_share_v", "head_install_share", "head_install_share_v"):
    series = [e[name] for e in two]
    ok, exact = same(summarize(series), sec["head"][name], name)
    (bad if not ok else inexact if not exact else []).append(f"head/{name}")
for name in ("F_half_rem", "F_half_ins", "F_swap_pl", "F_swap_ref", "additivity"):
    ok, exact = same(summarize([e[name] for e in two]), sec["direction_free"][name], name)
    (bad if not ok else inexact if not exact else []).append(f"direction_free/{name}")
res_frames = state["confirmation"]["results"]["frames"]
prog = {"P_error_rem": [f["P_error_rem"] for f in res_frames], "P_error_ins": [f["P_error_ins"] for f in res_frames],
        "N_hat_minus_N": [f["N_hat"] - f["N"] for f in res_frames], "S_hat_minus_S": [f["S_hat"] - f["S"] for f in res_frames]}
for name, series in prog.items():
    ok, exact = same(summarize(series), sec["program"][name], name)
    (bad if not ok else inexact if not exact else []).append(f"program/{name}")
check("secondary_statistics_reproduced_independently", not bad, bad[:6])
check("bootstrap_intervals_bit_for_bit", not inexact, inexact[:6])
out["facts"]["head_shares"] = {name: sec["head"][name]["median"] for name in ("head_removal_share", "head_removal_share_v", "head_install_share", "head_install_share_v")}

# --- interpretive numbers ---------------------------------------------------------------------------------------------
over = json.loads((OUT / "c_overshoot_per_frame.json").read_text(encoding="utf-8"))
two_pairs = [o for o in over if o["cue"] == "two"]
share = [(o["s"] / o["dx_c_norm"]) ** 2 for o in two_pairs]  # the read part's share of the centred squared length of Δx at p_c
out["facts"]["read_share_of_centred_dx_squared"] = {"min": min(share), "median": median(share), "max": max(share)}
out["facts"]["cos_dx_c_w_hat"] = {"min": min(math.sqrt(v) for v in share), "max": max(math.sqrt(v) for v in share)}
pred_vs_meas = []
for o in two_pairs:
    p = table["frames"][o["frame_id"]]["cues"]["two"]["pair"]
    pred_vs_meas.append({"frame_id": o["frame_id"], "dr": o["dr"], "dr_hat": p["dr"], "s_over_off_c": o["s_over_off_c"], "s_over_off_c_hat": p["s_over_off_c"],
                         "overshoot_hat": p["overshoot"]})
out["facts"]["program_pair_vs_measured"] = {"max_abs_dr_error": max(abs(r["dr"] - r["dr_hat"]) for r in pred_vs_meas),
                                            "max_abs_ratio_error": max(abs(r["s_over_off_c"] - r["s_over_off_c_hat"]) for r in pred_vs_meas),
                                            "predicted_overshoot_frames": sum(r["overshoot_hat"] for r in pred_vs_meas)}
ln = state["confirmation"]["secondary"]["statistics"]["layer_norm"]
groups = {"read (NEC-READ/SUF-READ)": ["NEC-READ@two", "SUF-READ@two"], "off (NEC-OFF/SUF-OFF)": ["NEC-OFF@two", "SUF-OFF@two"],
          "iso (16 arms)": [f"{p}-ISO-{j}@two" for p in ("NEC", "SUF") for j in range(1, 9)], "amp/neg": ["NEC-AMP@two", "SUF-NEG@two"]}
recs = state["confirmation"]["secondary"]["records"]
ln_groups = {}
for gname, tags in groups.items():
    g3 = [recs[f][t]["gamma3_e_over_sigma"] for f in frames for t in tags]
    gl2 = [recs[f][t]["gamma_ln2_e_over_sigma"] for f in frames for t in tags]
    sig = [recs[f][t]["sigma"] for f in frames for t in tags]
    ln_groups[gname] = {"gamma3_e_over_sigma_median": median(g3), "gamma_ln2_e_over_sigma_median": median(gl2), "sigma_median": median(sig),
                        "sigma_min": min(sig), "sigma_max": max(sig)}
plain_sig = {}
for f in frames:
    pass
out["facts"]["layer_norm_groups"] = ln_groups
print(json.dumps(out["facts"], indent=1))
# the manipulation check (ΔT along d̂_T at p_t relative to PL), per arm, medians
manip = sec["manipulation_check"]
out["facts"]["manipulation_check_two"] = {tag: round(manip[tag]["median"], 4) for tag in manip if tag.endswith("@two")}
print(json.dumps(out["facts"]["manipulation_check_two"], indent=0))
(OUT / "e_secondary.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
print("failed:", [k for k, v in out["checks"].items() if not v["ok"]])
