"""Review item 3 (and 4's table use): the primary statistics recomputed from the saved contrasts and the committed table
with this review's own code (no neural_decompiler import), then compared with the state's results bit for bit. Also the
per-frame details the review reports (weak frames, D's failing frames, P's disagreements, margins)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rguard  # noqa: E402  (first)

rguard.self_test()
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from fractions import Fraction  # noqa: E402

import torch  # noqa: E402

ROOT = rguard.ROOT
EXP = ROOT / "experiments/101-read-component-intervention"
OUT = rguard.OUT
out = {"checks": {}, "facts": {}}


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if ok and detail is None else f"  {detail}"), flush=True)


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


state = json.loads((ROOT / "outputs/experiment-101/results.json").read_text(encoding="utf-8"))
lock = json.loads((EXP / "preregistration-lock.json").read_text(encoding="utf-8"))
conf = json.loads((EXP / "confirmation-v1.json").read_text(encoding="utf-8"))
table_bytes = (EXP / "stage1/table.json").read_bytes()
table = json.loads(table_bytes)
check("table_file_sha256", hashlib.sha256(table_bytes).hexdigest() == "71d339cd1d7151ad57949384388679593264ac07dec8c9c944c121d9918795fa")
body = {k: v for k, v in table.items() if k != "content_sha256"}
check("table_content_digest", table["content_sha256"] == hashlib.sha256(canonical(body).encode()).hexdigest() == "9191800440af37909d2068459e360d7772191fc66bf2ad5fb9cf2c133246ac86")
check("table_construction_bound_by_lock", {k: table.get(k) for k in lock["stage1"]["table"]} == lock["stage1"]["table"])
tensors = torch.load(ROOT / "outputs/experiment-101/stage2-measurements.pt")
run_order = lock["run_order"]
row_of = {tag: i for i, tag in enumerate(run_order)}
frames = [f["frame_id"] for f in conf["frames"]]
TAU_REM, TAU_INS = lock["calibration_101"]["tau_rem"], lock["calibration_101"]["tau_ins"]
check("tau_from_lock", (TAU_REM, TAU_INS) == (0.025153865322968608, 0.03511072680382421) and state["confirmation"]["results"]["P"]["tau_rem"] == TAU_REM
      and state["confirmation"]["results"]["P"]["tau_ins"] == TAU_INS)
check("constants_from_lock", lock["constants"]["weak_shift"] == -1.0 and lock["constants"]["median_floor"] == 0.05 and lock["constants"]["holm_alpha"] == "1/40"
      and lock["configuration"]["holm_thresholds"] == [26, 25] and lock["configuration"]["d_threshold"] == 25 and lock["configuration"]["p_threshold"] == 27
      and lock["configuration"]["precondition"] == 30 and lock["constants"]["tau_floor"] == 0.02 and lock["constants"]["tau_cap"] == 0.2)

ARMS_TWO = [t.partition("@")[0] for t in run_order if t.endswith("@two")]


def dc_by_tag(fid):
    c = tensors[f"{fid}/contrasts"].double()
    c_ref = c[row_of["REF"]]
    return {tag: float((c[row_of[tag]] - c_ref).mean()) for tag in run_order}, {tag: math.fsum((c[row_of[tag]] - c_ref).tolist()) / 79 for tag in run_order}


def median(values):
    s = sorted(values)
    n = len(s)
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


def tail(n, k):
    return Fraction(sum(math.comb(n, j) for j in range(k, n + 1)), 2 ** n)


per_frame, fsum_diff = [], 0.0
for fid in frames:
    dc, dc_fsum = dc_by_tag(fid)
    fsum_diff = max(fsum_diff, max(abs(dc[t] - dc_fsum[t]) for t in run_order))
    entry = {"frame_id": fid}
    for cue in ("two", "three", "many"):
        d = {name: dc[f"{name}@{cue}"] for name in ("PL", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF", "PC-SWAP-PL", "PC-SWAP-REF")}
        full = d["PL"]
        e = {"dc_full": full, "weak": full > -1.0, "A_amp": d["NEC-AMP"] - full, "A_neg": d["SUF-NEG"], "dc": d}
        e.update({"F_rem": d["NEC-READ"] / full, "F_rem_off": d["NEC-OFF"] / full, "F_ins": d["SUF-READ"] / full, "F_ins_off": d["SUF-OFF"] / full,
                  "F_half_rem": d["NEC-HALF"] / full, "F_half_ins": d["SUF-HALF"] / full, "F_swap_pl": d["PC-SWAP-PL"] / full, "F_swap_ref": d["PC-SWAP-REF"] / full})
        e["N"] = e["F_rem_off"] - e["F_rem"]
        e["S"] = e["F_ins"] - e["F_ins_off"]
        e["additivity"] = e["F_rem"] + e["F_ins"]
        if cue == "two":
            e["F_rem_iso"] = [dc[f"NEC-ISO-{j}@two"] / full for j in range(1, 9)]
            e["F_ins_iso"] = [dc[f"SUF-ISO-{j}@two"] / full for j in range(1, 9)]
            e["iso_rem_margin"] = median(e["F_rem_iso"]) - e["F_rem"]
            e["iso_ins_margin"] = e["F_ins"] - median(e["F_ins_iso"])
            for v, suf in (("", ""), ("_v", "-V")):
                e[f"F_head_rem{v}"] = dc[f"HEAD-NEC{suf}@two"] / full
                e[f"F_head_ins{v}"] = dc[f"HEAD-SUF{suf}@two"] / full
                e[f"head_removal_share{v}"] = (full - dc[f"HEAD-NEC{suf}@two"]) / (full - d["NEC-READ"]) if 1.0 - e["F_rem"] >= 0.05 else None
                e[f"head_install_share{v}"] = dc[f"HEAD-SUF{suf}@two"] / d["SUF-READ"] if e["F_ins"] >= 0.05 else None
        entry[cue] = e
    per_frame.append(entry)
check("mean_vs_fsum_tiny", fsum_diff < 1e-12, fsum_diff)
out["facts"]["mean_vs_fsum_max_abs"] = fsum_diff

two = [e["two"] for e in per_frame]
weak = [e["weak"] for e in two]
WORST = {"N": -math.inf, "S": -math.inf, "A_amp": math.inf, "A_neg": -math.inf}


def fam(entries, name):
    return [WORST[name] if e["weak"] else e[name] for e in entries]


def count(entries, name, sign):
    return sum(1 for e in entries if not e["weak"] and (e[name] > 0 if sign > 0 else e[name] < 0))


n = 36
cR, cI = count(two, "N", 1), count(two, "S", 1)
pR, pI = tail(n, cR), tail(n, cI)
ordered = sorted([("R", pR), ("I", pI)], key=lambda kv: (kv[1], kv[0]))
alpha = Fraction(1, 40)
rej = {"R": False, "I": False}
for rank, (name, p) in enumerate(ordered):
    if p <= alpha / (2 - rank):
        rej[name] = True
    else:
        break
mR, mI = median(fam(two, "N")), median(fam(two, "S"))
passR, passI = rej["R"] and mR >= 0.05, rej["I"] and mI >= 0.05
cAmp, cNeg = count(two, "A_amp", -1), count(two, "A_neg", 1)
d_pass = cAmp >= 25 and cNeg >= 25
precond = sum(1 for e in two if e["dc_full"] <= -1.0)
# P from the committed table
agreements, p_rows = 0, []
for fid, e in zip(frames, two):
    pred = table["frames"][fid]["cues"]["two"]
    means = {arm: pred["arms"][arm]["mean"] for arm in pred["arms"]}
    mean_from_dc = {arm: float(torch.tensor(pred["arms"][arm]["dc"], dtype=torch.float64).mean()) for arm in pred["arms"]}
    full_hat = means["PL"]
    f_rem_hat, f_ins_hat = means["NEC-READ"] / full_hat, means["SUF-READ"] / full_hat
    consistent = (pred["dc_full"] == full_hat and pred["weak"] == (full_hat > -1.0) and pred["fractions"]["F_rem"] == f_rem_hat
                  and pred["fractions"]["F_ins"] == f_ins_hat and means == mean_from_dc)
    agree = False
    if not e["weak"] and not pred["weak"]:
        er, ei = abs(f_rem_hat - e["F_rem"]), abs(f_ins_hat - e["F_ins"])
        agree = er <= max(TAU_REM, 0.02) and ei <= max(TAU_INS, 0.02)
        p_rows.append({"frame_id": fid, "err_rem": er, "err_ins": ei, "agree": agree, "F_rem": e["F_rem"], "F_rem_hat": f_rem_hat, "F_ins": e["F_ins"],
                       "F_ins_hat": f_ins_hat, "table_consistent": consistent, "N_hat": pred["fractions"]["N"], "S_hat": pred["fractions"]["S"],
                       "dc_full_hat": full_hat})
    agreements += int(agree)
check("table_internally_consistent_for_two", all(r["table_consistent"] for r in p_rows))
p_pass = agreements >= 27
if precond < 30:
    label = "BEHAVIOR_PRECONDITION_NOT_MET"
elif not passR and not passI:
    label = "READ_COMPONENT_EFFECTS_NOT_ESTABLISHED"
elif passR and not passI:
    label = "READ_REMOVAL_EFFECT_INSTALLATION_NOT_ESTABLISHED"
elif passI and not passR:
    label = "READ_INSTALLATION_EFFECT_REMOVAL_NOT_ESTABLISHED"
elif not d_pass:
    label = "READ_COMPONENT_EFFECTS_DIRECTION_NOT_ESTABLISHED"
elif not (TAU_REM <= 0.2 and TAU_INS <= 0.2):
    label = "READ_COMPONENT_EFFECTS_PROGRAM_NOT_EVALUABLE"
elif not p_pass:
    label = "READ_COMPONENT_EFFECTS_PROGRAM_NOT_CONFIRMED"
else:
    label = "READ_COMPONENT_EFFECTS_AS_PREDICTED"
mine = {"weak_frames": sum(weak), "precondition": precond, "R_count": cR, "I_count": cI, "R_p": f"{pR.numerator}/{pR.denominator}", "I_p": f"{pI.numerator}/{pI.denominator}",
        "holm": rej, "R_median": mR, "I_median": mI, "R_pass": passR, "I_pass": passI, "A_amp_negative": cAmp, "A_neg_positive": cNeg,
        "median_A_amp": median(fam(two, "A_amp")), "median_A_neg": median(fam(two, "A_neg")), "D_pass": d_pass, "P_agreements": agreements, "P_pass": p_pass,
        "label": label, "tail_26": float(tail(36, 26)), "tail_25": float(tail(36, 25)), "tail_24": float(tail(36, 24)), "tail_27": float(tail(36, 27))}
out["facts"]["recomputed"] = mine
print(json.dumps(mine, indent=1))

# --- against the state, bit for bit ------------------------------------------------------------------------------------
res = state["confirmation"]["results"]
check("label_equal", res["outcome"]["label"] == label == "READ_COMPONENT_EFFECTS_AS_PREDICTED", res["outcome"]["label"])
check("precondition_equal", res["precondition"] == {"frames_at_most_minus_one": precond, "met": precond >= 30, "required": 30})
check("R_equal", res["R"]["count"] == cR and res["R"]["p_value"] == mine["R_p"] and res["R"]["median"] == mR and res["R"]["holm_rejected"] == rej["R"]
      and res["R"]["result"] == ("PASS" if passR else "FAIL"))
check("I_equal", res["I"]["count"] == cI and res["I"]["p_value"] == mine["I_p"] and res["I"]["median"] == mI and res["I"]["holm_rejected"] == rej["I"]
      and res["I"]["result"] == ("PASS" if passI else "FAIL"))
check("D_equal", res["D"]["A_amp_negative"] == cAmp and res["D"]["A_neg_positive"] == cNeg and res["D"]["median_A_amp"] == mine["median_A_amp"]
      and res["D"]["median_A_neg"] == mine["median_A_neg"] and res["D"]["result"] == ("PASS" if d_pass else "FAIL") and res["D"]["reached"] is True)
check("P_equal", res["P"]["agreements"] == agreements and res["P"]["result"] == ("PASS" if p_pass else "FAIL") and res["P"]["reached"] is True
      and res["P"]["tolerance"] == {"rem": max(TAU_REM, 0.02), "ins": max(TAU_INS, 0.02)})
frame_diffs = []
for rec, e, fid in zip(res["frames"], two, frames):
    if rec["frame_id"] != fid:
        frame_diffs.append((fid, "id"))
        continue
    for k in ("dc_full", "A_amp", "A_neg", "F_rem", "F_rem_off", "F_ins", "F_ins_off", "N", "S"):
        if rec.get(k) != e[k]:
            frame_diffs.append((fid, k, rec.get(k), e[k]))
    if rec["weak"] != e["weak"]:
        frame_diffs.append((fid, "weak"))
pr = {r["frame_id"]: r for r in p_rows}
for rec in res["frames"]:
    r = pr.get(rec["frame_id"])
    if r is None:
        continue
    for a, b in (("F_rem_hat", "F_rem_hat"), ("F_ins_hat", "F_ins_hat"), ("P_error_rem", "err_rem"), ("P_error_ins", "err_ins"), ("P_agree", "agree"),
                 ("N_hat", "N_hat"), ("S_hat", "S_hat")):
        if rec.get(a) != r[b]:
            frame_diffs.append((rec["frame_id"], a, rec.get(a), r[b]))
check("per_frame_records_bit_for_bit", not frame_diffs, frame_diffs[:5])

# --- details for the report --------------------------------------------------------------------------------------------
def summary(values):
    s = sorted(values)
    return {"min": s[0], "q1": s[8], "median": median(s), "q3": s[27], "max": s[-1]}


out["facts"]["two_summary"] = {k: summary([e[k] for e in two]) for k in ("dc_full", "F_rem", "F_rem_off", "F_ins", "F_ins_off", "N", "S", "A_amp", "A_neg",
                                                                         "F_half_rem", "F_half_ins", "F_swap_pl", "F_swap_ref", "additivity",
                                                                         "iso_rem_margin", "iso_ins_margin", "F_head_rem", "F_head_ins", "F_head_rem_v", "F_head_ins_v")}
out["facts"]["A_amp_nonnegative_frames"] = [(e["frame_id"], e["two"]["A_amp"], e["two"]["dc_full"], e["two"]["dc"]["NEC-AMP"]) for e in per_frame if not e["two"]["A_amp"] < 0]
out["facts"]["P_disagreements"] = [r for r in p_rows if not r["agree"]]
out["facts"]["P_errors_max"] = {"rem": max(r["err_rem"] for r in p_rows), "ins": max(r["err_ins"] for r in p_rows)}
out["facts"]["N_min_frame"] = min(((e["two"]["N"], e["frame_id"]) for e in per_frame))
out["facts"]["S_min_frame"] = min(((e["two"]["S"], e["frame_id"]) for e in per_frame))
out["facts"]["per_frame_two"] = [{"frame_id": e["frame_id"], **{k: e["two"][k] for k in ("dc_full", "F_rem", "F_rem_off", "F_ins", "F_ins_off", "N", "S", "A_amp", "A_neg",
                                                                                         "F_half_rem", "F_half_ins", "F_swap_pl", "F_swap_ref", "F_head_rem", "F_head_ins",
                                                                                         "F_head_rem_v", "F_head_ins_v", "iso_rem_margin", "iso_ins_margin")}}
                                 for e in per_frame]
# robustness: what the label would be under small perturbations of D (descriptive only; never a decision)
out["facts"]["D_margin_frames"] = cAmp - 25
out["facts"]["P_margin_frames"] = agreements - 27
(OUT / "b_primary_values.json").write_text(json.dumps({"per_frame": per_frame}, indent=0, default=str) + "\n", encoding="utf-8")
(OUT / "b_primary.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
print("failed:", [k for k, v in out["checks"].items() if not v["ok"]])
