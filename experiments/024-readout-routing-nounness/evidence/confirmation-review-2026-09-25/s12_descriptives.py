"""Item 12: the descriptive records, recomputed with own code from the saved artifact and the lock (no outcome force).
The recorded descriptives are read only at the end, for comparison."""
import rguard  # noqa: F401  (first)
from rguard import REPO

import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(REPO)
SCR = Path(rguard.SCRATCH)
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail != '' else ''}", flush=True)


lock = json.loads((ROOT / "experiments/024-readout-routing-nounness/preregistration-lock.json").read_text())
calib = json.loads((ROOT / "experiments/024-readout-routing-nounness/calibration-v1.json").read_text())
freeze = json.loads((ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json").read_text())
T = torch.load(ROOT / "outputs/experiment-024/stage2-measurements.pt", weights_only=True)
s07 = json.loads((SCR / "s07_results.json").read_text())
cues = [{"word": c["word"], "token_id": int(c["token_id"]), "class": c["class"], "nounness": float(c["nounness"]), "pred": float(c["predicted_log_mse"]),
         "above": bool(c["above_calibration_maximum"])} for c in lock["fresh"]["cues"]]
frames_all = sorted(freeze["exposed_frame_ids"])


def template_of(fid):
    return fid.split("-")[0]


groups = {"cue_final": [f for f in frames_all if template_of(f) != "coordinated"], "coordinated": [f for f in frames_all if template_of(f) == "coordinated"]}
by_token = sorted(cues, key=lambda c: c["token_id"])
row_of = {}
for g, fids in groups.items():
    r = 0
    for c in by_token:
        for fid in fids:
            row_of[(c["token_id"], fid)] = (g, r)
            r += 1


def avg_ranks(values):
    idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[idx[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = avg_ranks(a), avg_ranks(b)
    n = len(ra)
    ma, mb = math.fsum(ra) / n, math.fsum(rb) / n
    return math.fsum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(math.fsum((x - ma) ** 2 for x in ra) * math.fsum((y - mb) ** 2 for y in rb))


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# per cue, per frame: SSE_C, S, Q over the 79 nouns (own float64 via Python floats, fsum)
per = {}
for c in cues:
    rows = {}
    for fid in frames_all:
        g, r = row_of[(c["token_id"], fid)]
        y, p = T[f"Y1/{g}/dc"][r].tolist(), T[f"Y1/{g}/ceiling"][r].tolist()
        rows[fid] = {"sse": math.fsum((a - b) ** 2 for a, b in zip(y, p)), "S": math.fsum(y), "Q": math.fsum(a * a for a in y), "n": len(y), "y": y, "c": p}
    per[c["token_id"]] = rows
mse = {t: math.fsum(r["sse"] for r in rows.values()) / (108 * 79) for t, rows in per.items()}
x = [c["nounness"] for c in cues]
m = [mse[c["token_id"]] for c in cues]
check("own MSE here (fsum of per-frame fsums) agrees with item 7's own MSE (≤ 1e-15 relative)", all(abs(a - b["mse"]) <= 1e-15 * b["mse"] for a, b in zip(m, s07["per_cue"])),
      f"bitwise {sum(a == b['mse'] for a, b in zip(m, s07['per_cue']))}/40")

# --- MSE by class ------------------------------------------------------------------------------------------------------
cls_mse = {k: math.fsum(mse[c["token_id"]] for c in cues if c["class"] == k) / 8 for k in "NBDCE"}
logs = {c["token_id"]: math.log(mse[c["token_id"]]) for c in cues}
cls_log = {k: math.fsum(logs[c["token_id"]] for c in cues if c["class"] == k) / 8 for k in "NBDCE"}
print("   class mean MSE:", {k: round(v, 6) for k, v in cls_mse.items()})
print("   class mean log MSE:", {k: round(v, 6) for k, v in cls_log.items()})

# --- the four contrasts and their SHA-indexed bootstrap intervals ---------------------------------------------------
def contrast_values(mm):
    return {"noun effect": (mm["B"] + mm["C"] + mm["D"] + mm["E"]) / 4 - mm["N"], "measure effect among nouns": (mm["B"] + mm["D"]) / 2 - (mm["C"] + mm["E"]) / 2,
            "plurality effect among nouns": (mm["B"] + mm["C"]) / 2 - (mm["D"] + mm["E"]) / 2,
            "measure × plurality interaction (descriptive)": (mm["B"] - mm["D"]) - (mm["C"] - mm["E"])}


def sha_index(b, cls, slot, n):
    return int.from_bytes(hashlib.sha256(f"024|contrast|{b}|{cls}|{slot}".encode("utf-8")).digest()[:8], "big") % n


class_logs = {k: [logs[c["token_id"]] for c in cues if c["class"] == k] for k in "NBDCE"}
point = contrast_values(cls_log)
res = {name: [] for name in point}
for b in range(10_000):
    drawn = {k: math.fsum(v[sha_index(b, k, s, len(v))] for s in range(len(v))) / len(v) for k, v in class_logs.items()}
    for name, val in contrast_values(drawn).items():
        res[name].append(val)
intervals = {name: [sorted(v)[249], sorted(v)[9750]] for name, v in res.items()}
for name in point:
    print(f"   {name}: {point[name]:+.6f}  95% bootstrap [{intervals[name][0]:+.6f}, {intervals[name][1]:+.6f}]")

# --- group Spearman, normalized MSE, R²_C, bias, slope ----------------------------------------------------------------
sp_group = {}
for g, fids in groups.items():
    mg = [math.fsum(per[c["token_id"]][f]["sse"] for f in fids) / (len(fids) * 79) for c in cues]
    sp_group[g] = spearman(x, mg)
nmse = [math.fsum(r["sse"] for r in per[c["token_id"]].values()) / math.fsum(r["Q"] for r in per[c["token_id"]].values()) for c in cues]
sp_nmse = spearman(x, nmse)
print(f"   Spearman cue-final {sp_group['cue_final']:.6f}, coordinated {sp_group['coordinated']:.6f}; normalized-MSE Spearman {sp_nmse:.6f}")
r2 = {}
for c in cues:
    rows = per[c["token_id"]].values()
    N = math.fsum(r["n"] for r in rows)
    S = math.fsum(r["S"] for r in rows)
    Q = math.fsum(r["Q"] for r in rows)
    sse = math.fsum(r["sse"] for r in rows)
    sst = Q - S * S / N
    r2[c["token_id"]] = 1 - sse / sst
r2_cls = {k: math.fsum(r2[c["token_id"]] for c in cues if c["class"] == k) / 8 for k in "NBDCE"}
print("   mean per-cue R²_C by class:", {k: round(v, 4) for k, v in r2_cls.items()}, "; range", round(min(r2.values()), 4), "–", round(max(r2.values()), 4))

# --- the frozen line's residuals and the extrapolation flags ---------------------------------------------------------
resid = {c["token_id"]: logs[c["token_id"]] - c["pred"] for c in cues}
errs = [resid[c["token_id"]] for c in cues]
line_summary = {"median_abs": median([abs(e) for e in errs]), "mean_signed": math.fsum(errs) / 40,
                "extrapolated": {"n": sum(c["above"] for c in cues), "median_abs": median([abs(resid[c["token_id"]]) for c in cues if c["above"]]),
                                 "mean_signed": math.fsum(resid[c["token_id"]] for c in cues if c["above"]) / sum(c["above"] for c in cues)},
                "within": {"n": sum(not c["above"] for c in cues), "median_abs": median([abs(resid[c["token_id"]]) for c in cues if not c["above"]]),
                           "mean_signed": math.fsum(resid[c["token_id"]] for c in cues if not c["above"]) / sum(not c["above"] for c in cues)}}
print(f"   the frozen line: median |log error| {line_summary['median_abs']:.4f}, mean signed {line_summary['mean_signed']:+.4f}; extrapolated {line_summary['extrapolated']}; "
      f"within {line_summary['within']}")
top = sorted(cues, key=lambda c: -resid[c["token_id"]])[:6]
print("   largest positive residuals (observed − predicted log MSE):", [(c["word"], c["class"], round(resid[c["token_id"]], 4)) for c in top])
print("   most negative residuals:", [(c["word"], c["class"], round(resid[c["token_id"]], 4)) for c in sorted(cues, key=lambda c: resid[c["token_id"]])[:4]])
max_cal = max(float(e["nounness_loo"]) for e in calib["calibration_cues"])
above_own = [c["nounness"] > max_cal for c in cues]
check("calibration maximum nounness_loo == 0.13502027836111233 (from the installed record)", max_cal == 0.13502027836111233, repr(max_cal))
check("23 of 40 fresh cues above the calibration maximum; own flags == the lock's; 5 of 8 E", sum(above_own) == 23 and above_own == [c["above"] for c in cues]
      and sum(1 for c, a in zip(cues, above_own) if a and c["class"] == "E") == 5 == lock["fresh"]["extrapolation"]["E_above_maximum"])
print("   above by class:", {k: sum(1 for c, a in zip(cues, above_own) if a and c["class"] == k) for k in "NBDCE"})

# --- the qualitative pattern (descriptive only) ------------------------------------------------------------------------
check("pattern: noun effect clearly positive (interval excludes 0)", point["noun effect"] > 0 and intervals["noun effect"][0] > 0, intervals["noun effect"])
check("pattern: measure and plurality effects smaller than the noun effect, intervals include 0",
      abs(point["measure effect among nouns"]) < point["noun effect"] and abs(point["plurality effect among nouns"]) < point["noun effect"]
      and intervals["measure effect among nouns"][0] < 0 < intervals["measure effect among nouns"][1] and intervals["plurality effect among nouns"][0] < 0 < intervals["plurality effect among nouns"][1])
check("pattern: the association present in both subsets (cue-final and coordinated Spearman > 0.3137)", sp_group["cue_final"] > 0.3137 and sp_group["coordinated"] > 0.3137, sp_group)
check("pattern: the frozen line under-predicts slightly on average (mean signed log error > 0, small)", 0 < line_summary["mean_signed"] < 0.2, line_summary["mean_signed"])
gal = resid[[c for c in cues if c["word"] == "gallon"][0]["token_id"]]
acre = resid[[c for c in cues if c["word"] == "acre"][0]["token_id"]]
check("pattern: gallon and acre have large positive residuals (the two largest)", {c["word"] for c in top[:2]} == {"gallon", "acre"}, (round(gal, 4), round(acre, 4)))

# --- only now: the recorded descriptives --------------------------------------------------------------------------------
state = json.loads((ROOT / "outputs/experiment-024/results.json").read_text())
D = state["confirmation"]["descriptives"]
check("recorded descriptives hold exactly secondary, contrasts, ladder (no failures)", sorted(D) == ["contrasts", "ladder", "secondary"])
rc_ = D["contrasts"]
d_means = max(abs(rc_["class_means_log_mse"][k] - cls_log[k]) for k in "NBDCE")
d_point = max(abs(rc_["contrasts"][n]["value"] - point[n]) for n in point)
d_int = max(abs(rc_["contrasts"][n]["interval"][i] - intervals[n][i]) for n in point for i in (0, 1))
check("recorded class means of log MSE / contrast values / intervals reproduce (≤ 1e-12)", d_means <= 1e-12 and d_point <= 1e-12 and d_int <= 1e-12, f"{d_means:.2e} {d_point:.2e} {d_int:.2e}")
check("recorded contrast elements [249, 9750], tag 024|contrast, 10,000 resamples", rc_["elements"] == [249, 9750] and rc_["tag"] == "024|contrast" and rc_["resamples"] == 10000)
sec = D["secondary"]
check("recorded cue-final / coordinated Spearman == own", abs(sec["spearman_by_group"]["cue_final"] - sp_group["cue_final"]) <= 1e-12
      and abs(sec["spearman_by_group"]["coordinated"] - sp_group["coordinated"]) <= 1e-12, sec["spearman_by_group"])
check("recorded normalized-MSE Spearman == own", abs(sec["nmse_spearman"] - sp_nmse) <= 1e-12, sec["nmse_spearman"])
d_r2 = max(abs(e["R2_C"] - r2[c["token_id"]]) for e, c in zip(sec["per_cue"], cues))
d_nm = max(abs(e["nmse"] - v) / v for e, v in zip(sec["per_cue"], nmse))
check("recorded per-cue R²_C and nmse reproduce (≤ 1e-9 abs / 1e-12 rel)", d_r2 <= 1e-9 and d_nm <= 1e-12, f"R2 {d_r2:.2e}, nmse {d_nm:.2e}")
check("recorded extrapolation flags per cue == own", [e["above_calibration_maximum"] for e in sec["per_cue"]] == above_own)
ln = sec["line"]
check("recorded line summary reproduces (median |err|, mean signed, extrapolated/within)",
      abs(ln["median_abs_log_error"] - line_summary["median_abs"]) <= 1e-12 and abs(ln["mean_signed_log_error"] - line_summary["mean_signed"]) <= 1e-12
      and ln["extrapolated"]["n"] == 23 and ln["within_range"]["n"] == 17
      and abs(ln["extrapolated"]["median_abs_log_error"] - line_summary["extrapolated"]["median_abs"]) <= 1e-12, {k: ln[k] for k in ("median_abs_log_error", "mean_signed_log_error")})
print("   recorded per-class line errors:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ln["per_class"].items()})
print("   recorded ladder per class:", {k: (round(v["block5_attention"], 4), round(v["block4_attention"], 4)) for k, v in D["ladder"]["per_class"].items()},
      "Level-1 identity", D["ladder"]["level1_identity"])
json.dump({"class_mse": cls_mse, "class_log": cls_log, "contrasts": point, "intervals": intervals, "spearman_group": sp_group, "nmse_spearman": sp_nmse, "r2_class_mean": r2_cls,
           "line": line_summary, "residuals": {c["word"]: resid[c["token_id"]] for c in cues}}, open(SCR / "s12_results.json", "w"), indent=1)
print(f"S12 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
