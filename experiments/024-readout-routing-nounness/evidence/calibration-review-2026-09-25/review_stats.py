"""Independent review of Experiment 024's calibrate — part 3 (pure; no model): the exposed line, the 10,000 SHA-indexed
draws and F_ρ, the 100,000-permutation null and null₉₇.₅, the effective threshold, and the descriptives, each by my own
implementation (exact integer / rational arithmetic), against the record and the saved arrays. Canonical functions are
called only for canonical byte digests and bit-for-bit reproduction. Read-only."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/calibration_review024")
import rguard  # noqa: E402  (first: the audit hook)

import bisect  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from decimal import Decimal, getcontext  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

getcontext().prec = 60
ROOT = Path(rguard.ROOT)
SCRATCH = Path(__file__).resolve().parent
results = {"checks": [], "failures": []}


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail != "" else ""), flush=True)
    results["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:2000]})
    if not ok:
        results["failures"].append(name)


def own_digest(array, kind):
    a = np.ascontiguousarray(np.asarray(array).astype(kind))
    return hashlib.sha256(json.dumps(list(a.shape)).encode("ascii") + b"|" + kind.encode("ascii") + b"|" + a.tobytes()).hexdigest()


def sha_index(text, modulus):
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % modulus


def doubled_ranks(matrix):
    """Rows of values -> doubled average ranks (exact integers) by pairwise counting: 2r = 1 + 2·#less + #equal."""
    m = np.asarray(matrix, dtype=np.float64)
    less = (m[:, None, :] < m[:, :, None]).sum(axis=2)
    equal = (m[:, None, :] == m[:, :, None]).sum(axis=2)
    return 1 + 2 * less + equal


def exact_spearman_rows(xm, ym):
    """Per row: Spearman as Σab / sqrt(Σa²·Σb²) over centred doubled ranks, integers exact, the root and quotient at
    60 digits. Returns Decimals (None where a rank vector is constant)."""
    rx, ry = doubled_ranks(xm), doubled_ranks(ym)
    n = rx.shape[1]
    a, b = (rx - (n + 1)).astype(np.int64), (ry - (n + 1)).astype(np.int64)
    sab, saa, sbb = (a * b).sum(1), (a * a).sum(1), (b * b).sum(1)
    out = []
    for p, q, r in zip(sab.tolist(), saa.tolist(), sbb.tolist()):
        out.append(None if q == 0 or r == 0 else Decimal(p) / (Decimal(q) * Decimal(r)).sqrt())
    return out


record = json.loads((ROOT / "outputs/experiment-024/candidate-calibration.json").read_text(encoding="utf-8"))
own_mse = json.loads((SCRATCH / "own_mse.json").read_text())
own_scores = json.loads((SCRATCH / "own_scores.json").read_text())
entries = record["calibration_cues"]
x_rec, m_rec, y_rec = [e["nounness_loo"] for e in entries], [e["mse"] for e in entries], [e["log_mse"] for e in entries]
x_own, m_own = own_scores["loo_exact_float"], own_mse["mse_exact"]
y_own = [math.log(v) for v in m_own]
arrays = torch.load(ROOT / "outputs/experiment-024/calibration-arrays.pt")

print("== the saved arrays ==")
spec = {"draw_rho": (torch.float64, (10000,)), "draw_defined": (torch.bool, (10000,)), "draw_indices": (torch.int64, (10000, 40)), "null_rho": (torch.float64, (100000,)),
        "null_permutations": (torch.int8, (100000, 40))}
check("arrays file holds exactly the five calibration arrays with the planned dtypes and shapes", set(arrays) == set(spec)
      and all(arrays[k].dtype == d and tuple(arrays[k].shape) == s for k, (d, s) in spec.items()), {k: (str(v.dtype), tuple(v.shape)) for k, v in arrays.items()})
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
own_array_digests = {"draw_rho": own_digest(arrays["draw_rho"].numpy(), "<f8"), "draw_defined": own_digest(arrays["draw_defined"].numpy().astype(np.int64), "<i8"),
                     "draw_indices": own_digest(arrays["draw_indices"].numpy(), "<i8"), "null_rho": own_digest(arrays["null_rho"].numpy(), "<f8"),
                     "null_permutations": own_digest(arrays["null_permutations"].numpy().astype(np.int64), "<i8")}
canonical_digests = rr.arrays_digests(arrays)
check("array digests: own routine == canonical rr.arrays_digests (rc.tensor_digest, bool/int8 as int64) == record", own_array_digests == canonical_digests == record["arrays_sha256"],
      {k: v[:8] for k, v in own_array_digests.items()})

print("== Item 5: the exposed line (exact rational OLS) ==")


def exact_ols(x, y):
    X, Y = [Fraction(v) for v in x], [Fraction(v) for v in y]
    n = len(X)
    mx, my = sum(X) / n, sum(Y) / n
    sxx = sum((v - mx) ** 2 for v in X)
    sxy = sum((u - mx) * (v - my) for u, v in zip(X, Y))
    slope = sxy / sxx
    intercept = my - slope * mx
    rss = sum((v - (intercept + slope * u)) ** 2 for u, v in zip(X, Y))
    sd = (Decimal(rss.numerator) / Decimal(rss.denominator) / Decimal(n - 2)).sqrt()
    return slope, intercept, sd


slope_x, intercept_x, sd_x = exact_ols(x_rec, y_rec)
line = record["line"]
d_line = (abs(float(slope_x) - line["slope"]), abs(float(intercept_x) - line["intercept"]), abs(float(sd_x) - line["residual_sd"]))
check("line: record == expected full-precision values (slope 1.2992017464639374, intercept -2.481260357420486, residual sd 0.2570037337349923, n 139)",
      (line["slope"], line["intercept"], line["residual_sd"], line["n"]) == (1.2992017464639374, -2.481260357420486, 0.2570037337349923, 139))
check("line: own exact OLS on the record's 139 (x, log MSE) vs the record (numerical disagreement only)", max(d_line) < 1e-14,
      f"|Δ| slope {d_line[0]:.2e}, intercept {d_line[1]:.2e}, sd {d_line[2]:.2e}; exact slope {float(slope_x)!r}, intercept {float(intercept_x)!r}, sd {float(sd_x)!r}")
check("line: canonical rr.ols on the record's entries reproduces the record bit for bit", {k: v for k, v in rr.ols(x_rec, y_rec).items() if k != "formulas"}
      == {k: line[k] for k in ("slope", "intercept", "residual_sd", "n")})
p1 = np.polyfit(np.array(x_rec), np.array(y_rec), 1)
check("line: numpy.polyfit route", abs(p1[0] - line["slope"]) < 1e-12 and abs(p1[1] - line["intercept"]) < 1e-12, f"polyfit slope {p1[0]!r} intercept {p1[1]!r}")
slope_o, intercept_o, sd_o = exact_ols(x_own, y_own)
check("line from fully independent inputs (own exact LOO scores, own exactly rounded MSE)", abs(float(slope_o) - line["slope"]) < 1e-12 and abs(float(intercept_o) - line["intercept"]) < 1e-12
      and abs(float(sd_o) - line["residual_sd"]) < 1e-12, f"slope {float(slope_o)!r}, intercept {float(intercept_o)!r}, sd {float(sd_o)!r}")
rho_all = exact_spearman_rows([x_rec], [m_rec])[0]
rho_own = exact_spearman_rows([x_own], [m_own])[0]
check("exposed Spearman over the 139: own exact route == 0.5296617364493499 (record)", abs(float(rho_all) - record["descriptive"]["calibration_rho"]) < 1e-15
      and record["descriptive"]["calibration_rho"] == 0.5296617364493499 and float(rho_own) == float(rho_all), f"exact {rho_all:.20f}; from own inputs {float(rho_own)!r}")
for stratum, value in record["descriptive"]["within_stratum_rho"].items():
    sel = [i for i, e in enumerate(entries) if e["stratum"] == stratum]
    mine = exact_spearman_rows([[x_rec[i] for i in sel]], [[m_rec[i] for i in sel]])[0]
    check(f"within-stratum Spearman {stratum} (descriptive)", abs(float(mine) - value) < 1e-15, f"record {value!r}, own {float(mine)!r}")
pron = record["pronoun_cues"]
pr = exact_spearman_rows([[p["nounness"] for p in pron]], [[p["mse"] for p in pron]])[0]
errs = [p["log_mse"] - (line["intercept"] + line["slope"] * p["nounness"]) for p in pron]
med = sorted(abs(e) for e in errs)
med = 0.5 * (med[17] + med[18])
pc = record["descriptive"]["pronoun_check"]
check("pronoun out-of-fit check (descriptive)", abs(float(pr) - pc["rho"]) < 1e-15 and abs(med - pc["median_abs_log_error"]) < 1e-15
      and abs(math.fsum(errs) / 36 - pc["mean_signed_log_error"]) < 1e-15 and pc["n"] == 36, f"ρ {float(pr)!r}, median |err| {med!r}, mean signed {math.fsum(errs) / 36!r}")

print("== Item 6: the 10,000 SHA-indexed draws and F_ρ ==")
idx = np.array([[sha_index(f"024|primary|{b}|{slot}", 139) for slot in range(40)] for b in range(10000)], dtype=np.int64)
check("draw indices: own SHA formula == saved array (10,000 × 40, all in [0, 139))", np.array_equal(idx, arrays["draw_indices"].numpy()) and idx.min() >= 0 and idx.max() <= 138)
check("draw-indices digest: own == canonical == record == 955930ce…", own_digest(idx, "<i8") == rc.tensor_digest(torch.tensor(idx)) == record["arrays_sha256"]["draw_indices"]
      == "955930cef2c6e3a4052944b8dafdd9ded3dfc87d15af46a79b9b8e48bd2baabd")
xa, ma_ = np.array(x_rec)[idx], np.array(m_rec)[idx]
exact = exact_spearman_rows(xa, ma_)
undefined = sum(1 for v in exact if v is None)
check("exactly 10,000 draws, 0 undefined (own); every saved draw_defined true", len(exact) == 10000 and undefined == 0 and bool(arrays["draw_defined"].all()), f"undefined {undefined}")
saved = arrays["draw_rho"].numpy().tolist()
diff = [abs(Decimal(s) - e) for s, e in zip(saved, exact)]
exact_float = [float(e) for e in exact]
bitwise = sum(1 for s, e in zip(saved, exact_float) if s == e)
check("draw Spearman: saved (canonical) vs own exact route", max(diff) < Decimal("1e-15"), f"max |Δ| {float(max(diff)):.3e}; bitwise equal to the exactly rounded value {bitwise}/10000")
check("draw_rho digest == 7c03376c…, draw_defined digest == 76b5b970…", record["arrays_sha256"]["draw_rho"] == own_digest(np.array(saved), "<f8")
      == "7c03376c7c05cdb9a95660f955a72e44023bb1d71dbabf5f9dfe1293f3d88f58"
      and record["arrays_sha256"]["draw_defined"] == own_digest(np.ones(10000, dtype=np.int64), "<i8") == "76b5b97082b66ea03f4d27ef68a61a92b12528861ade0eeef00f28db031fac09")
order = sorted(range(10000), key=lambda i: exact[i])
f_exact = exact[order[249]]
sorted_saved = sorted(saved)
floor = record["primary_floor"]
check("F_ρ = ascending element [249] = 0.24411074612857814 (saved values and own exact route agree)", sorted_saved[249] == floor["F_rho"] == 0.24411074612857814
      and float(f_exact) == floor["F_rho"], f"own exact [249] {f_exact:.20f}")
neighbours = (exact[order[248]], exact[order[249]], exact[order[250]])
check("F_ρ's position is unambiguous (gaps to the neighbouring order statistics ≫ numerical disagreement)",
      min(neighbours[1] - neighbours[0], neighbours[2] - neighbours[1]) > Decimal("1e-12") or neighbours[0] == neighbours[1] or neighbours[1] == neighbours[2],
      f"[248] {float(neighbours[0])!r}, [249] {float(neighbours[1])!r}, [250] {float(neighbours[2])!r}")
median = 0.5 * (sorted_saved[4999] + sorted_saved[5000])
check("draws min / median / max / element [9750] match the record's tails", (sorted_saved[0], median, sorted_saved[-1], sorted_saved[9750])
      == (floor["tails"]["min"], floor["tails"]["median"], floor["tails"]["max"], floor["tails"]["element_upper"]) == (floor["tails"]["min"], floor["direction_check"]["median"],
                                                                                                                        floor["tails"]["max"], floor["tails"]["element_upper"]),
      f"min {sorted_saved[0]!r}, median {median!r}, max {sorted_saved[-1]!r}, [9750] {sorted_saved[9750]!r}")
check("direction check: F_ρ ≤ the median of the draws", floor["F_rho"] <= median and floor["direction_check"]["ok"] is True)
check("undefined count 0 < stop 250", floor["undefined"] == 0 and floor["stop_at"] == 250)
own_indep = exact_spearman_rows(np.array(x_own)[idx], np.array(m_own)[idx])
check("draws from fully independent inputs (own exact LOO, own exact MSE) give identical values", all(a == b for a, b in zip(own_indep, exact)),
      f"identical {sum(1 for a, b in zip(own_indep, exact) if a == b)}/10000")

print("== Item 7: the 100,000-permutation null ==")
perms = np.empty((100000, 40), dtype=np.int64)
for p in range(100000):
    perm = list(range(40))
    for i in range(39, 0, -1):
        j = sha_index(f"024|null|{p}|{i}", i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    perms[p] = perm
check("permutations: own Fisher–Yates == saved int8 array; every row a permutation of 0..39", np.array_equal(perms, arrays["null_permutations"].numpy().astype(np.int64))
      and bool((np.sort(perms, axis=1) == np.arange(40)).all()))
check("permutations digest: own == canonical == record == a68f7291…", own_digest(perms, "<i8") == rc.tensor_digest(torch.tensor(perms)) == record["null"]["permutations_sha256"]
      == record["arrays_sha256"]["null_permutations"] == "a68f72910c090678198880ae551e86281745615052a4daaca01b40fa75e48548")
S = ((perms - np.arange(40)) ** 2).sum(axis=1)
null_exact = [Fraction(10660 - int(s), 10660) for s in S.tolist()]  # ρ = 1 − 6S / (n(n² − 1)) = 1 − S/10660 for n = 40
null_float = [float(v) for v in null_exact]
saved_null = arrays["null_rho"].numpy().tolist()
check("null values: saved (canonical) == own exact rationals, correctly rounded, bit for bit (100,000)", saved_null == null_float and len(saved_null) == 100000)
check("null_rho digest == 92e63029…", own_digest(np.array(saved_null), "<f8") == record["arrays_sha256"]["null_rho"] == "92e6302944d2adbfdc4aa897ea85ea1698adfaa66100c1b614c0bfc39c200665")
order_null = sorted(null_exact)
t_exact = order_null[97499]
null_975 = record["null"]["null_975"]
check("null₉₇.₅ = ascending element [97499] = 0.3136960600375234", float(t_exact) == null_975 == 0.3136960600375234 == sorted(saved_null)[97499],
      f"exact {t_exact} (S = {10660 - t_exact.numerator * 10660 // t_exact.denominator}), float {float(t_exact)!r}")
first = bisect.bisect_left(order_null, t_exact)
last = bisect.bisect_right(order_null, t_exact) - 1
count_eq, count_ge, count_gt = last - first + 1, 100000 - first, 100000 - (last + 1)
check("15 null values tied at the threshold (one exact rational, one float)", count_eq == 15 and len({v for v in saved_null if v == null_975}) == 1
      and sum(1 for v in saved_null if v == null_975) == 15, f"tied block elements [{first}]..[{last}]; ≥ t: {count_ge} ({count_ge / 1000:.3f} %), > t: {count_gt} ({count_gt / 1000:.3f} %)")
check("the pass rule ρ ≥ t is unambiguous: t is one exact value; ties with the fresh ρ count as passing the null guard", float(t_exact) == null_975,
      "a fresh ρ without ties lies on the same grid (Σab/5330 with Σa² = Σb² = 5330 exactly), so equality is decidable exactly")
median_null = 0.5 * (float(order_null[49999]) + float(order_null[50000]))
check("null median matches the record", median_null == record["null"]["median"], f"{median_null!r}")
check("no 99.5 % value is used (only element [97499])", record["null"]["element"] == 97499 and "null_995" not in json.dumps(record))

print("== Item 8: the effective threshold ==")
T = max(floor["F_rho"], null_975)
check("T_primary = max(F_ρ, null₉₇.₅) = 0.3136960600375234, bound by the null", T == 0.3136960600375234 == record["effective_threshold"]["value"]
      and record["effective_threshold"]["binds"] == "null_975" and floor["F_rho"] < null_975, f"F_ρ {floor['F_rho']!r} < null₉₇.₅ {null_975!r}")
below = sum(1 for v in saved if v < null_975)
env = sum(1 for v in saved if null_975 <= v < floor["F_rho"])
rates = record["descriptive"]["draw_rates"]
check("draw rates (descriptive): GUARD_FAILURE 593/10,000, ENVELOPE_ONLY 0 (empty band since F_ρ < null), PASS 9,407/10,000",
      (below, env) == (593, 0) and rates == {"NOT_INTERPRETABLE": 0.0, "GUARD_FAILURE": below / 10000, "ENVELOPE_ONLY_FAILURE": 0.0, "PASS": (10000 - below) / 10000}, rates)

print("guard events:", rguard.EVENTS)
check("review guard: no repository write attempted, no forbidden read attempted", not rguard.EVENTS["refused_writes"] and not rguard.EVENTS["refused_reads"])
json.dump(results, open(SCRATCH / "review_stats.json", "w"), indent=1)
print("FAILURES:", results["failures"] or "none")
