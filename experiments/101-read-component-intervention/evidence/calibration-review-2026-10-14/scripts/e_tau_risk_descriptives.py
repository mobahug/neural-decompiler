"""Checks 7 (τ, P, power), 11 (the Level-1 margin at confirm), 12 (head arms, isotropic controls), 13 (the smoke
reference's digests against the production measurements), and the G4 maxima per arm type. No model, no forward pass.

Inputs: my measured values (b), my exact-chain per-run values (c), the frozen per-run values and predictions (d), the
saved state and the smoke reference record. numpy only (no scipy in the venv): MLE by my own Nelder–Mead.
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np
import torch

from common import CAL, OUT, OUTPUTS_101, WRITE_ATTEMPTS, dump

assert not [m for m in sys.modules if m.startswith("neural_decompiler")]
out: dict = {}
b = json.loads((OUT / "b_values.json").read_text())
c = json.loads((OUT / "c_per_run.json").read_text())
d = json.loads((OUT / "d_frozen_crosscheck.json").read_text())
dpr = json.loads((OUT / "d_per_run_frozen.json").read_text())
state = json.loads((OUTPUTS_101 / "results.json").read_text())
cand = json.loads((OUTPUTS_101 / "candidate-calibration-v1.json").read_text())
frame_ids = b["frame_ids"]
V = b["values"]

# ------------------------------------------------------------------------------------------------------------------
# 7. τ: my measured fractions (b) against the frozen predictions (d), the 33rd of the 36 sorted errors.
err_rem, err_ins, rows = [], [], []
for f in frame_ids:
    m, p = V[f]["two"], d["predicted"][f]
    er, ei = abs(p["F_rem_hat"] - m["F_rem"]), abs(p["F_ins_hat"] - m["F_ins"])
    err_rem.append(er)
    err_ins.append(ei)
    rows.append({"frame": f, "F_rem": m["F_rem"], "F_rem_hat": p["F_rem_hat"], "err_rem": er, "F_ins": m["F_ins"], "F_ins_hat": p["F_ins_hat"], "err_ins": ei,
                 "N": m["N"], "N_hat": p["N_hat"], "S": m["S"], "S_hat": p["S_hat"], "dc_full": m["dc_full"], "dc_full_hat": p["dc_full_hat"], "weak": m["weak"],
                 "weak_hat": p["weak_hat"], "A_amp": m["A_amp"], "A_amp_hat": p["A_amp_hat"], "A_neg": m["A_neg"], "A_neg_hat": p["A_neg_hat"]})
tau_rem, tau_ins = sorted(err_rem)[32], sorted(err_ins)[32]
tol_rem, tol_ins = max(tau_rem, 0.02), max(tau_ins, 0.02)
agree = [r["frame"] for r in rows if not r["weak"] and not r["weak_hat"] and r["err_rem"] <= tol_rem and r["err_ins"] <= tol_ins]
out["tau_mine"] = {"tau_rem": tau_rem, "tau_ins": tau_ins, "recorded": cand["tau"], "equal": (tau_rem, tau_ins) == (cand["tau"]["rem"], cand["tau"]["ins"]),
                   "evaluable": tau_rem <= 0.20 and tau_ins <= 0.20, "agree_count": len(agree), "disagree": [f for f in frame_ids if f not in agree],
                   "sorted_err_rem_top4": sorted(err_rem)[-4:], "sorted_err_ins_top4": sorted(err_ins)[-4:],
                   "errors_equal_recorded_distribution": (sorted(err_rem) == cand["distributions"]["P_error_rem"], sorted(err_ins) == cand["distributions"]["P_error_ins"]),
                   "predicted_weak_frames": [r["frame"] for r in rows if r["weak_hat"]],
                   "predicted_weak_stratum": {cue: [f for f in frame_ids if d["predicted"][f]["stratum"][cue]["weak_hat"]] for cue in ("three", "many")},
                   "N_hat_minus_N": {"min": min(r["N_hat"] - r["N"] for r in rows), "max": max(r["N_hat"] - r["N"] for r in rows)},
                   "S_hat_minus_S": {"min": min(r["S_hat"] - r["S"] for r in rows), "max": max(r["S_hat"] - r["S"] for r in rows)},
                   "dc_full_hat_minus_dc_full": {"min": min(r["dc_full_hat"] - r["dc_full"] for r in rows), "max": max(r["dc_full_hat"] - r["dc_full"] for r in rows)}}
out["tau_rows"] = rows


def binom_power(n, share, k):
    return math.fsum(math.comb(n, j) * share ** j * (1 - share) ** (n - j) for j in range(k, n + 1))


n = 36
shares = {"N_positive": sum(V[f]["two"]["N"] > 0 for f in frame_ids) / n, "S_positive": sum(V[f]["two"]["S"] > 0 for f in frame_ids) / n,
          "A_amp_negative": sum(V[f]["two"]["A_amp"] < 0 for f in frame_ids) / n, "A_neg_positive": sum(V[f]["two"]["A_neg"] > 0 for f in frame_ids) / n,
          "P_agree": len(agree) / n}
power = {"R_or_I_at_26": {k: binom_power(n, shares[k], 26) for k in ("N_positive", "S_positive")}, "R_or_I_at_25": {k: binom_power(n, shares[k], 25) for k in ("N_positive", "S_positive")},
         "D_at_25": {k: binom_power(n, shares[k], 25) for k in ("A_amp_negative", "A_neg_positive")}, "P_at_27": binom_power(n, shares["P_agree"], 27)}
out["shares_power_mine"] = {"shares": shares, "power": power, "recorded_shares": cand["shares"], "recorded_power": cand["power"]}
# medians with design §7's rule (even n: mean of the 18th and 19th)
med = lambda v: 0.5 * (sorted(v)[17] + sorted(v)[18])  # noqa: E731
out["medians_exposed"] = {k: med([V[f]["two"][k] for f in frame_ids]) for k in ("N", "S", "A_amp", "A_neg", "F_rem", "F_ins", "F_rem_off", "F_ins_off")}
out["A_amp_nonnegative_frames"] = sorted(((V[f]["two"]["A_amp"], f, V[f]["two"]["dc_full"], V[f]["two"]["dc"]["NEC-AMP"]) for f in frame_ids if V[f]["two"]["A_amp"] >= 0), reverse=True)
out["A_amp_predicted_sign_agreement"] = sum((V[f]["two"]["A_amp"] < 0) == (d["predicted"][f]["A_amp_hat"] < 0) for f in frame_ids)
out["A_amp_hat_negative"] = sum(d["predicted"][f]["A_amp_hat"] < 0 for f in frame_ids)

# ------------------------------------------------------------------------------------------------------------------
# 6. G4 maxima per arm type, with location and margin (frozen per-run values = mine within 1e-11).
TOL = {"I3": 1e-4, "I4": 1e-3, "level1": 2e-2, "readout": 2e-2, "logit": 2e-2}


def kind_of(tag):
    arm = tag.split("@")[0]
    if arm == "PL":
        return "plain PL@w"
    if arm == "NOOP":
        return "NOOP"
    if arm in ("NEC-READ", "SUF-READ", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF"):
        return "read (" + ("NEC" if arm.startswith("NEC") else "SUF") + ")"
    if arm in ("NEC-OFF", "SUF-OFF"):
        return "off"
    if arm.startswith(("NEC-ISO", "SUF-ISO")):
        return "iso"
    return "PC-SWAP"


table = defaultdict(dict)
readout = b["readout_per_run"]
logit = b["logit_per_run"]
for key, v in dpr.items():
    k = kind_of(v["tag"])
    for q, val in (("level1", v["level1"]), ("I4", v["i4"]), ("readout", readout[key]), ("logit", logit[key])):
        cur = table[k].get(q)
        if cur is None or val > cur["max"]:
            table[k][q] = {"max": val, "at": key, "margin": TOL[q] / val if val > 0 else math.inf}
i3 = {}
for key, v in dpr.items():
    if v["kind"] == "plain":
        i3[f'{v["frame"]}|{v["cue"]}'] = v["i3_pair"]
i3_key = max(i3, key=i3.get)
table["plain PL@w"]["I3"] = {"max": i3[i3_key], "at": i3_key, "margin": TOL["I3"] / i3[i3_key]}
overall = {}
for q in ("level1", "I4", "readout", "logit"):
    best = max(((table[k][q]["max"], table[k][q]["at"]) for k in table if q in table[k]))
    overall[q] = {"max": best[0], "at": best[1], "tolerance": TOL[q], "margin": TOL[q] / best[0]}
overall["I3"] = table["plain PL@w"]["I3"] | {"tolerance": TOL["I3"]}
out["g4_by_arm_type"] = {k: dict(v) for k, v in sorted(table.items())}
out["g4_overall"] = overall
out["g4_recorded"] = {q: state["calibration"]["gates"]["tolerance"][f"G4_{q}"] for q in ("I3", "I4", "level1", "readout", "logit")}
out["i4_reference_prompt_runs_max"] = max(v["i4"] for v in dpr.values() if v["tag"].split("@")[0] in ("SUF-READ", "SUF-OFF", "SUF-NEG", "SUF-HALF", "PC-SWAP-REF") or v["tag"].startswith("SUF-ISO"))

# ------------------------------------------------------------------------------------------------------------------
# 11. The Level-1 margin at confirm.
lv = {k: v["level1"] for k, v in dpr.items()}
keys_all = list(lv)
x_all = np.array([lv[k] for k in keys_all])
edit_keys = [k for k in keys_all if dpr[k]["kind"] != "plain"]
x_edit = np.array([lv[k] for k in edit_keys])
qs = (0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)
out["level1_distribution"] = {"all_g4_runs": {"n": int(x_all.size), **{str(q): float(np.quantile(x_all, q)) for q in qs}},
                              "residual_edit_runs": {"n": int(x_edit.size), **{str(q): float(np.quantile(x_edit, q)) for q in qs}},
                              "count_above": {str(t): int((x_all > t).sum()) for t in (0.005, 0.006, 0.0075, 0.01, 0.015, 0.02)}}
top = sorted(keys_all, key=lambda k: -lv[k])[:20]
cpr = c["per_run"]
out["level1_top20"] = [{"run": k, "level1": lv[k], "arm": dpr[k]["arm"], "cue": dpr[k]["cue"], "frame": dpr[k]["frame"], "p_c": dpr[k]["p_c"], "noun": dpr[k]["level1_noun"],
                        "signed": dpr[k]["level1_signed_at_max"], "readout_identity": readout[k], "logit_identity": logit[k], "E1": cpr[k]["E1"], "dh6_inf": cpr[k]["dh6_inf"],
                        "dc_mean": cpr[k]["dc_mean"]} for k in top]
# per-frame and per-arm-type maxima
fmax = defaultdict(float)
for k in keys_all:
    fmax[dpr[k]["frame"]] = max(fmax[dpr[k]["frame"]], lv[k])
fd = c["frame_diag"]
out["level1_frame_maxima"] = sorted(((v, f, fd[f]["p_c"], round(fd[f]["scores"]["5"]["max_abs_score"]), round(fd[f]["scores"]["4"]["max_abs_score"])) for f, v in fmax.items()), reverse=True)
by_cue = defaultdict(list)
by_kind = defaultdict(list)
for k in keys_all:
    by_cue[dpr[k]["cue"]].append(lv[k])
    by_kind[kind_of(dpr[k]["tag"])].append(lv[k])
out["level1_by_cue"] = {cue: {"max": max(v), "p99": float(np.quantile(v, 0.99)), "median": float(np.median(v)), "n": len(v)} for cue, v in by_cue.items()}
out["level1_by_kind"] = {kk: {"max": max(v), "p99": float(np.quantile(v, 0.99)), "median": float(np.median(v)), "n": len(v)} for kk, v in by_kind.items()}


def spearman(a, bb):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(bb))
    return float(np.corrcoef(ra, rb)[0, 1])


drivers = {}
for name in ("dh6_inf", "dc_mean", "E1", "f32_vs_f64_chain", "dx3_pc_norm", "dx3_pt_norm"):
    vals = np.array([abs(cpr[k][name]) for k in keys_all])
    drivers[f"run_level_spearman_{name}"] = spearman(x_all, vals)
drivers["run_level_spearman_readout_identity"] = spearman(x_all, np.array([readout[k] for k in keys_all]))
drivers["run_level_spearman_logit_identity"] = spearman(x_all, np.array([logit[k] for k in keys_all]))
fr = list(fmax)
fm = np.array([fmax[f] for f in fr])
for l in ("3", "4", "5"):
    drivers[f"frame_level_spearman_layer{l}_max_score"] = spearman(fm, np.array([fd[f]["scores"][l]["max_abs_score"] for f in fr]))
drivers["frame_level_spearman_p_c"] = spearman(fm, np.array([fd[f]["p_c"] for f in fr]))
drivers["frame_level_spearman_h6_sigma"] = spearman(fm, np.array([fd[f]["h6_sigma"] for f in fr]))
drivers["frame_max_by_p_c"] = {str(pc): {"n": int(sum(fd[f]["p_c"] == pc for f in fr)), "max": float(max(fmax[f] for f in fr if fd[f]["p_c"] == pc)),
                                         "median": float(np.median([fmax[f] for f in fr if fd[f]["p_c"] == pc]))} for pc in sorted({fd[f]["p_c"] for f in fr})}
# the top runs: is the error in Δh6 (blocks 3–5) or in the h6 → c readout? (the error vector splits exactly into the two parts)
out["level1_drivers"] = drivers


# --- Exceedance estimates ---------------------------------------------------------------------------------------
def nelder_mead(f, x0, step=0.1, iters=4000, tol=1e-12):
    n_ = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n_)[i] for i in range(n_)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1]); fr_ = f(xr)
        if fr_ < vals[0]:
            xe = centroid + 2 * (centroid - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr_ else (xr, fr_)
        elif fr_ < vals[-2]:
            pts[-1], vals[-1] = xr, fr_
        else:
            xc = centroid + 0.5 * (pts[-1] - centroid); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def gev_nll(params, x):
    mu, logs, xi = params
    s = math.exp(logs)
    t = (x - mu) / s
    if abs(xi) < 1e-8:
        return float(np.sum(math.log(s) + t + np.exp(-t)))
    z = 1 + xi * t
    if np.any(z <= 0):
        return 1e100
    return float(np.sum(math.log(s) + (1 + 1 / xi) * np.log(z) + z ** (-1 / xi)))


def gev_cdf(v, params):
    mu, logs, xi = params
    t = (v - mu) / math.exp(logs)
    if abs(xi) < 1e-8:
        return math.exp(-math.exp(-t))
    z = 1 + xi * t
    if z <= 0:
        return 0.0 if xi > 0 else 1.0
    return math.exp(-z ** (-1 / xi))


def fit_gev(x):
    mu0, s0 = float(np.mean(x)), float(np.std(x)) or 1e-4
    best = None
    for xi0 in (-0.2, 0.0, 0.2, 0.5):
        p, v = nelder_mead(lambda q: gev_nll(q, x), [mu0 - 0.45 * s0, math.log(0.78 * s0), xi0], step=0.2)
        if best is None or v < best[1]:
            best = (p, v)
    return best[0]


def gpd_nll(params, y):
    logs, xi = params
    s = math.exp(logs)
    if abs(xi) < 1e-8:
        return float(np.sum(math.log(s) + y / s))
    z = 1 + xi * y / s
    if np.any(z <= 0):
        return 1e100
    return float(np.sum(math.log(s) + (1 + 1 / xi) * np.log(z)))


def gpd_sf(y, params):
    logs, xi = params
    s = math.exp(logs)
    if abs(xi) < 1e-8:
        return math.exp(-y / s)
    z = 1 + xi * y / s
    return 0.0 if z <= 0 else z ** (-1 / xi)


def fit_gpd(y):
    best = None
    for xi0 in (-0.2, 0.0, 0.2, 0.5):
        p, v = nelder_mead(lambda q: gpd_nll(q, y), [math.log(float(np.mean(y))), xi0], step=0.2)
        if best is None or v < best[1]:
            best = (p, v)
    return best[0]


TH = 0.02
N_FRAMES = 36
est = {}
# (a) block maxima per frame: GEV, lognormal and Gumbel; P(max over 36 new frames > 0.02) = 1 − F(0.02)^36
fm_sorted = np.sort(fm)
p_gev = fit_gev(fm)
est["frame_max_gev"] = {"mu": float(p_gev[0]), "sigma": math.exp(p_gev[1]), "xi": float(p_gev[2]), "P_frame_gt_0.02": 1 - gev_cdf(TH, p_gev),
                        "P_confirm_max_gt_0.02": 1 - gev_cdf(TH, p_gev) ** N_FRAMES}
lm, ls = float(np.mean(np.log(fm))), float(np.std(np.log(fm), ddof=1))
phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
est["frame_max_lognormal"] = {"mean_log": lm, "sd_log": ls, "P_frame_gt_0.02": 1 - phi((math.log(TH) - lm) / ls), "P_confirm_max_gt_0.02": 1 - phi((math.log(TH) - lm) / ls) ** N_FRAMES}
# (b) peaks over threshold on the 1,800 runs (independence across runs assumed: an optimistic simplification within a frame)
for qu in (0.90, 0.95):
    u = float(np.quantile(x_all, qu))
    y = x_all[x_all > u] - u
    pg = fit_gpd(y)
    p_run = (y.size / x_all.size) * gpd_sf(TH - u, pg)
    est[f"runs_pot_q{int(qu * 100)}"] = {"u": u, "n_exceed": int(y.size), "sigma": math.exp(pg[0]), "xi": float(pg[1]), "P_run_gt_0.02": p_run,
                                         "P_confirm_max_gt_0.02_indep_1800": 1 - (1 - p_run) ** 1800}
# (c) frame bootstrap of the GEV and lognormal estimates (2,000 resamples of the 36 frame maxima, fixed seed)
rng = np.random.default_rng(20261014)
boot_gev, boot_ln = [], []
for _ in range(2000):
    s = fm[rng.integers(0, N_FRAMES, N_FRAMES)]
    if np.unique(s).size < 4:
        continue
    pg = fit_gev(s)
    boot_gev.append(1 - gev_cdf(TH, pg) ** N_FRAMES)
    m_, s_ = float(np.mean(np.log(s))), float(np.std(np.log(s), ddof=1))
    boot_ln.append(1 - phi((math.log(TH) - m_) / s_) ** N_FRAMES)
for name, arr in (("bootstrap_gev", boot_gev), ("bootstrap_lognormal", boot_ln)):
    a = np.array(arr)
    est[name] = {"n": int(a.size), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)), "p95": float(np.quantile(a, 0.95)), "mean": float(a.mean())}
# (d) the same without the single extreme frame 018-3 (sensitivity: how much of the tail it carries)
fm_wo = np.array([fmax[f] for f in fr if f != "coordinated-adjective-018-3"])
pg = fit_gev(fm_wo)
est["frame_max_gev_without_018_3"] = {"xi": float(pg[2]), "P_confirm_max_gt_0.02": 1 - gev_cdf(TH, pg) ** N_FRAMES}
# (e) exchangeability: P(max of 36 new frames > max of these 36) = 1/2; ratio of tolerance to the observed maximum
est["ratio_tolerance_to_observed_max"] = TH / float(fm.max())
est["ratio_observed_max_to_second"] = float(fm_sorted[-1] / fm_sorted[-2])
out["level1_exceedance"] = est

# ------------------------------------------------------------------------------------------------------------------
# 12. Head arms and isotropic controls (descriptive; mine = frozen within 0.0).
head = {}
for k in ("F_head_rem", "F_head_rem_v", "F_head_ins", "F_head_ins_v", "head_removal_share", "head_removal_share_v", "head_install_share", "head_install_share_v",
          "head_vs_sufread", "head_vs_sufread_v", "iso_rem_margin", "iso_ins_margin", "F_rem", "F_ins", "F_rem_off", "F_ins_off", "F_half_rem", "F_half_ins",
          "F_swap_pl", "F_swap_ref", "additivity"):
    vals = [V[f]["two"][k] for f in frame_ids if V[f]["two"][k] is not None]
    head[k] = {"n": len(vals), "min": min(vals), "median": med(vals) if len(vals) == 36 else float(np.median(vals)), "max": max(vals)}
head["counts"] = {"F_head_rem<1": sum(V[f]["two"]["F_head_rem"] < 1 for f in frame_ids), "F_head_rem_v<1": sum(V[f]["two"]["F_head_rem_v"] < 1 for f in frame_ids),
                  "F_head_ins>0": sum(V[f]["two"]["F_head_ins"] > 0 for f in frame_ids), "F_head_ins_v>0": sum(V[f]["two"]["F_head_ins_v"] > 0 for f in frame_ids),
                  "F_rem<F_rem_iso_median": sum(V[f]["two"]["iso_rem_margin"] > 0 for f in frame_ids),
                  "F_ins>F_ins_iso_median": sum(V[f]["two"]["iso_ins_margin"] > 0 for f in frame_ids),
                  "removal_guard_excluded": sum(V[f]["two"]["head_removal_share"] is None for f in frame_ids),
                  "install_guard_excluded": sum(V[f]["two"]["head_install_share"] is None for f in frame_ids)}
iso_rem_all = [x for f in frame_ids for x in V[f]["two"]["F_rem_iso"]]
iso_ins_all = [x for f in frame_ids for x in V[f]["two"]["F_ins_iso"]]
head["F_rem_iso_all"] = {"min": min(iso_rem_all), "max": max(iso_rem_all)}
head["F_ins_iso_all"] = {"min": min(iso_ins_all), "max": max(iso_ins_all)}
out["head_iso"] = head
strat = {}
for cue in ("three", "many"):
    strat[cue] = {"N>0": sum(V[f][cue]["N"] > 0 for f in frame_ids), "S>0": sum(V[f][cue]["S"] > 0 for f in frame_ids), "A_amp<0": sum(V[f][cue]["A_amp"] < 0 for f in frame_ids),
                  "A_neg>0": sum(V[f][cue]["A_neg"] > 0 for f in frame_ids), "weak": sum(V[f][cue]["weak"] for f in frame_ids),
                  "dc_full": {"min": min(V[f][cue]["dc_full"] for f in frame_ids), "max": max(V[f][cue]["dc_full"] for f in frame_ids)},
                  "median_N": med([V[f][cue]["N"] for f in frame_ids]), "median_S": med([V[f][cue]["S"] for f in frame_ids])}
out["stratum"] = strat
ceil_by_kind = defaultdict(lambda: (0.0, None))
for k, v in cpr.items():
    if v["g4"]:
        tag = v["tag"]
        group = tag if tag.split("@")[0] in ("PL", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "PC-SWAP-PL", "PC-SWAP-REF") else kind_of(tag) + "@" + tag.split("@")[1]
        if v["ceiling_error"] > ceil_by_kind[group][0]:
            ceil_by_kind[group] = (v["ceiling_error"], k)
out["ceiling_error_by_arm"] = {k: {"max": v[0], "at": v[1]} for k, v in sorted(ceil_by_kind.items())}

# ------------------------------------------------------------------------------------------------------------------
# 13. The smoke reference (unguarded, 08:15 UTC) against the production measurements of the same frame.
ref_record = torch.load(CAL / "smoke_reference.pt", weights_only=True)
prod = state["calibration"]["measurements"]
same_t = {name: prod["tensors_sha256"].get(name) == dig for name, dig in ref_record["tensors_sha256"].items()}
out["smoke_reference_vs_production"] = {"frame": ref_record["frame"], "keys": len(ref_record["keys"]), "tensors": len(same_t), "tensors_equal": sum(same_t.values()),
                                        "logits_equal": ref_record["logits_sha256"]["coordinated-adjective-1"] == prod["logits_sha256"]["coordinated-adjective-1"],
                                        "keys_are_calibration_keys": set(ref_record["keys"]) <= set(state["calibration_executed"])}
out["write_attempts"] = WRITE_ATTEMPTS
dump("e_tau_risk_descriptives.json", out)
print(json.dumps({k: v for k, v in out.items() if k not in ("tau_rows",)}, indent=1, default=str))
