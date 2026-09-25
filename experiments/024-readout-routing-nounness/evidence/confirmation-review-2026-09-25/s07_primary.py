"""Items 7-10: per-cue MSE (own code), the primary Spearman (own average ranks + exact rational), the exact E–N guard
(own dyadic-integer enumeration), and the outcome derived from own values and an own transcription of the outcome table.
The recorded results are read only at the very end, for comparison."""
import rguard  # noqa: F401  (first)
from rguard import REPO

import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(REPO)
SCR = Path(rguard.SCRATCH)
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail != '' else ''}", flush=True)


lock = json.loads((ROOT / "experiments/024-readout-routing-nounness/preregistration-lock.json").read_text())
freeze = json.loads((ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json").read_text())
T = torch.load(ROOT / "outputs/experiment-024/stage2-measurements.pt", weights_only=True)
frames_all = sorted(freeze["exposed_frame_ids"])

# ---------------------------------------------------------------------------------------------------------------------
# Item 7. The row → (cue, frame) mapping, own: ul.stage_two_022 fills block[group][row] with row = the index of (t, f) in
# ul.table_units' pairs: tokens sorted by token id, frames sorted by frame_id, pairs token-major, filtered by group
# (coordinated = template "coordinated"; every other template is cue-final). The template of a frame id is its prefix.
# ---------------------------------------------------------------------------------------------------------------------
cues = [{"word": c["word"], "token_id": int(c["token_id"]), "class": c["class"]} for c in lock["fresh"]["cues"]]  # frozen order N,B,D,C,E
check("lock cue order == freeze cue order", [(c["word"], c["token_id"], c["class"]) for c in cues] == [(c["word"], int(c["token_id"]), c["class"]) for c in freeze["cues"]])
by_token = sorted(cues, key=lambda c: c["token_id"])


def template_of(frame_id):
    return frame_id.split("-")[0]


print("   templates:", sorted({template_of(f) for f in frames_all}))
groups = {"cue_final": [f for f in frames_all if template_of(f) != "coordinated"], "coordinated": [f for f in frames_all if template_of(f) == "coordinated"]}
check("72 cue-final + 36 coordinated frames", len(groups["cue_final"]) == 72 and len(groups["coordinated"]) == 36)
row_of = {}
for g, fids in groups.items():
    r = 0
    for c in by_token:
        for fid in fids:
            row_of[(c["token_id"], fid)] = (g, r)
            r += 1
check("mapping covers 4,320 (cue, frame) pairs onto 2,880 + 1,440 distinct rows", len(row_of) == 4320 and len(set(row_of.values())) == 4320)
# the prompt log's execution order (frames by id, then cues by token id) visits the rows exactly as stage_two_022 fills them
plog = [json.loads(line)["key"] for line in open(Path(rguard.SCRATCH).parent / "confirm024/confirm_prompts.jsonl")]
visit = [row_of[(int(k.split("|")[2]), k.split("|")[0])] for k in plog]
check("prompt log keys map onto every row exactly once", len(set(visit)) == 4320)

# own per-cue MSE: three own routes
N_CELLS = 108 * 79
dc = {g: T[f"Y1/{g}/dc"].double() for g in groups}
cc = {g: T[f"Y1/{g}/ceiling"].double() for g in groups}
own = []
for c in cues:
    sq_all, pair_sums_torch, exact = [], [], Fraction(0)
    for fid in frames_all:
        g, r = row_of[(c["token_id"], fid)]
        y, p = dc[g][r].tolist(), cc[g][r].tolist()
        sq = [(a - b) * (a - b) for a, b in zip(y, p)]  # binary64, elementwise
        sq_all += sq
        pair_sums_torch.append(float(((dc[g][r] - cc[g][r]) ** 2).sum()))  # the canonical per-pair order (torch float64 sum)
        exact += sum(Fraction(v) for v in sq)
    assert len(sq_all) == N_CELLS
    mse_fsum = math.fsum(sq_all) / N_CELLS  # exactly rounded total, then one division
    mse_exact = float(exact / N_CELLS)  # the correctly rounded MSE of the binary64 squares
    mse_pairorder = math.fsum(pair_sums_torch) / math.fsum([79.0] * 108)  # mimics rr.cue_mse's order
    own.append({**c, "nounness": None, "mse": mse_fsum, "mse_exact": mse_exact, "mse_pairorder": mse_pairorder})
nounness = {int(c["token_id"]): float(c["nounness"]) for c in lock["fresh"]["cues"]}
for o in own:
    o["nounness"] = nounness[o["token_id"]]
    o["log_mse"] = math.log(o["mse"])
check("one MSE per cue (40), all finite and > 0", len(own) == 40 and all(math.isfinite(o["mse"]) and o["mse"] > 0 for o in own))
d_routes = max(abs(o["mse"] - o["mse_exact"]) / o["mse_exact"] for o in own)
d_routes2 = max(abs(o["mse"] - o["mse_pairorder"]) / o["mse_exact"] for o in own)
n_bit_exact = sum(o["mse"] == o["mse_exact"] for o in own)
print(f"   own routes: fsum vs exact-rational: {n_bit_exact}/40 bitwise equal, max rel diff {d_routes:.3e}; fsum vs pair-order: max rel diff {d_routes2:.3e}")

# the canonical per-cue MSE (rr.cue_mse over rr.fresh_cue_cells) for a bitwise/numeric comparison
sys.path.insert(0, str(ROOT / "src"))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("run024_confirmation_review", ROOT / "experiments/024-readout-routing-nounness/run.py")
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)
rr = run.rr


def refuse(*a, **k):
    raise RuntimeError("no model / tokenizer")


runner = run.Runner(log=lambda m: None, model_loader=refuse, tokenizer_loader=refuse)
base = runner._base()
confirmation, _ = runner._confirmation(base)
units = rr.target_units(confirmation)
canon_rows = {(int(units.tokens[t]["token_id"]), units.frames[f].frame_id): (g, r) for g, pairs in units.pairs.items() for r, (t, f) in enumerate(pairs)}
check("own row mapping == the canonical (rr.target_units) mapping for all 4,320 pairs", canon_rows == row_of)
cells = rr.fresh_cue_cells(units, T, confirmation)
canon_mse = [rr.cue_mse(cells[i]) for i in range(40)]
bitwise = sum(a == o["mse"] for a, o in zip(canon_mse, own))
bitwise_po = sum(a == o["mse_pairorder"] for a, o in zip(canon_mse, own))
max_rel = max(abs(a - o["mse"]) / a for a, o in zip(canon_mse, own))
max_rel_exact = max(abs(a - o["mse_exact"]) / a for a, o in zip(canon_mse, own))
print(f"   canonical rr.cue_mse vs own fsum route: {bitwise}/40 bitwise equal, max rel diff {max_rel:.3e}; vs own exact route max rel {max_rel_exact:.3e}; "
      f"vs own pair-order mimic: {bitwise_po}/40 bitwise equal")
check("canonical per-cue MSE agrees numerically with own (max rel diff ≤ 1e-12)", max_rel <= 1e-12 and max_rel_exact <= 1e-12, f"{max_rel:.3e}")
check("canonical rr.cue_mse == own pair-order mimic bit for bit (40/40)", bitwise_po == 40)
check("the canonical cells count 79 nouns per pair and 108 pairs per cue", bool((cells[:, :, 0] == 79.0).all()) and cells.shape == (40, 108, 8))

print("\n   Per-cue table (frozen order; nounness from the lock; own MSE = fsum over the 8,532 cells / 8,532):")
print("   | # | class | word | token id | nounness (lock) | own MSE | own log MSE | canonical MSE − own |")
for i, (o, a) in enumerate(zip(own, canon_mse)):
    print(f"   | {i + 1} | {o['class']} | {o['word']} | {o['token_id']} | {o['nounness']!r} | {o['mse']!r} | {o['log_mse']!r} | {a - o['mse']:+.3e} |")
by_class = {k: [o["mse"] for o in own if o["class"] == k] for k in "NBDCE"}
print("   class mean MSE:", {k: math.fsum(v) / len(v) for k, v in by_class.items()})

# ---------------------------------------------------------------------------------------------------------------------
# Item 8. Spearman, own average ranks; ties; exact rational.
# ---------------------------------------------------------------------------------------------------------------------


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


def pearson(a, b):
    n = len(a)
    ma, mb = math.fsum(a) / n, math.fsum(b) / n
    sab = math.fsum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = math.fsum((x - ma) ** 2 for x in a)
    sbb = math.fsum((y - mb) ** 2 for y in b)
    return sab / math.sqrt(saa * sbb)


x = [o["nounness"] for o in own]
y = [o["mse"] for o in own]
ties_x, ties_y = 40 - len(set(x)), 40 - len(set(y))
check("no ties in either vector (40 distinct nounness, 40 distinct MSE)", ties_x == 0 and ties_y == 0, (ties_x, ties_y))
rx, ry = avg_ranks(x), avg_ranks(y)
rho_own = pearson(rx, ry)
d2 = sum(int(a - b) ** 2 for a, b in zip(rx, ry))
rho_exact = 1 - Fraction(6 * d2, 40 * (40 ** 2 - 1))
rho_canon = rr.spearman(x, y)
rho_canon_direct = rr.spearman_direct(x, y)
ry_exact = avg_ranks([o["mse_exact"] for o in own])
ry_canon = avg_ranks(canon_mse)
print(f"   ρ own (average ranks + Pearson, fsum) = {rho_own!r}; exact rational 1 − 6Σd²/(n(n²−1)) = {rho_exact} = {float(rho_exact)!r} (Σd² = {d2}); "
      f"canonical rr.spearman = {rho_canon!r}; rr.spearman_direct = {rho_canon_direct!r}")
check("the MSE ranks are identical under the three own MSE routes and the canonical MSE", ry == ry_exact == ry_canon)
check("own ρ == canonical ρ (difference ≤ 1e-12); both == float(exact rational)", abs(rho_own - rho_canon) <= 1e-12 and rho_canon == float(rho_exact),
      f"|own − canonical| = {abs(rho_own - rho_canon):.3e}; |own − exact| = {abs(rho_own - float(rho_exact)):.3e}")
THRESH = 0.3136960600375234
check("the locked effective threshold is 0.3136960600375234 = max(F_ρ, null₉₇.₅), bound by null₉₇.₅",
      lock["primary"]["effective_threshold"] == {"value": THRESH, "binds": "null_975"} and max(lock["primary"]["F_rho"], lock["primary"]["null_975"]) == THRESH,
      (lock["primary"]["F_rho"], lock["primary"]["null_975"]))
primary_pass = rho_exact >= Fraction(THRESH)
check("PRIMARY: ρ = 1628/2665 ≥ the locked threshold (exact rational comparison) → PASS", primary_pass and rho_exact == Fraction(1628, 2665), str(rho_exact))

# ---------------------------------------------------------------------------------------------------------------------
# Item 9. The exact E–N guard, own integer / dyadic implementation.
# ---------------------------------------------------------------------------------------------------------------------
units_e = [(u["word"], int(u["token_id"])) for u in lock["guard"]["units"]["E"]]
units_n = [(u["word"], int(u["token_id"])) for u in lock["guard"]["units"]["N"]]
mse_of = {o["token_id"]: o for o in own}
check("guard units: 8 E then 8 N, the lock's E and N classes in frozen order", [mse_of[t]["class"] for _, t in units_e] == ["E"] * 8 and [mse_of[t]["class"] for _, t in units_n] == ["N"] * 8)


def guard(values):
    logs = [math.log(v) for v in values]  # binary64 natural log
    ratios = [v.as_integer_ratio() for v in logs]
    q = max(d for _, d in ratios)  # a common power of two
    assert all(q % d == 0 for _, d in ratios) and q & (q - 1) == 0
    z = [n * (q // d) for n, d in ratios]  # log_i = z_i / q exactly
    total = sum(z)
    combos = list(itertools.combinations(range(16), 8))
    assert len(combos) == 12870 == len(set(combos)) and combos[0] == tuple(range(8))
    sums = [sum(z[i] for i in s) for s in combos]
    obs = sums[0]
    k = sum(1 for s in sums if s >= obs)
    d_exact = Fraction(2 * obs - total, 8 * q)  # D(S) = (2 Σ_S − Σ_all) / (8 q)
    d_frac = Fraction(sum(Fraction(v) for v in logs[:8]), 8) - Fraction(sum(Fraction(v) for v in logs[8:]), 8)
    plain = sum(logs[:8]) / 8 - sum(logs[8:]) / 8
    second = max(s for s in sums[1:])
    return {"logs": logs, "K": k, "D_exact": d_exact, "D_frac_check": d_frac == d_exact, "D_float": float(d_exact), "plain": plain,
            "float_diff": abs(float(d_exact) - plain), "q": q, "margin_to_next": Fraction(2 * obs - total, 8 * q) - Fraction(2 * second - total, 8 * q),
            "min_E_minus_max_N": min(logs[:8]) - max(logs[8:]), "n_ties_with_obs": sum(1 for s in sums if s == obs)}


vals = [mse_of[t]["mse"] for _, t in units_e] + [mse_of[t]["mse"] for _, t in units_n]
G = guard(vals)
G_exact = guard([mse_of[t]["mse_exact"] for _, t in units_e] + [mse_of[t]["mse_exact"] for _, t in units_n])
canon_by_tid = {o["token_id"]: a for o, a in zip(own, canon_mse)}
G_canon = guard([canon_by_tid[t] for _, t in units_e] + [canon_by_tid[t] for _, t in units_n])
for label, g in (("own fsum MSE", G), ("own exact-rational MSE", G_exact), ("canonical MSE", G_canon)):
    print(f"   guard [{label}]: K = {g['K']} of 12,870; D_EN = {g['D_exact'].numerator}/{g['D_exact'].denominator} = {g['D_float']!r}; plain float {g['plain']!r} "
          f"(|diff| {g['float_diff']:.3e}); q = 2^{g['q'].bit_length() - 1}; ties with observed {g['n_ties_with_obs']}; margin to the next assignment "
          f"{float(g['margin_to_next']):.6f}; min E log − max N log {g['min_E_minus_max_N']:.6f}")
check("K = 1 (only the observed assignment reaches D_EN), under all three MSE routes", G["K"] == G_exact["K"] == G_canon["K"] == 1)
check("PASS: K = 1 ≤ 321 (= ⌊0.025 · 12,870⌋)", G["K"] <= (25 * 12870) // 1000 == 321)
check("D_EN (canonical MSE, own exact arithmetic) == 0.5680373703974243", G_canon["D_float"] == 0.5680373703974243, repr(G_canon["D_float"]))
check("D_EN (own MSE) == 0.5680373703974243 or within 1e-15", abs(G["D_float"] - 0.5680373703974243) <= 1e-15, repr(G["D_float"]))
check("the exact rational equals the Fraction route; float cross-check within 1e-12", G["D_frac_check"] and G["float_diff"] <= 1e-12 and G_canon["float_diff"] <= 1e-12)
check("descriptive: every E log-MSE exceeds every N log-MSE", G["min_E_minus_max_N"] > 0, f"min E − max N = {G['min_E_minus_max_N']:.6f}")
gm = {"E": math.exp(math.fsum(G["logs"][:8]) / 8), "N": math.exp(math.fsum(G["logs"][8:]) / 8)}
print(f"   geometric means: E {gm['E']:.6f}, N {gm['N']:.6f}; ratio exp(D_EN) {math.exp(G['D_float']):.6f}; mean MSE E {math.fsum(vals[:8]) / 8:.6f}, N {math.fsum(vals[8:]) / 8:.6f}")
# the canonical guard on the canonical MSE, for completeness
cg = rr.en_guard([canon_by_tid[t] for _, t in units_e], [canon_by_tid[t] for _, t in units_n], rr.PRODUCTION_GUARD)
print(f"   canonical rr.en_guard: result {cg['result']}, K {cg['K']}, D_EN {cg['D_EN']!r} = {cg['D_EN_exact']}, threshold_D {cg['threshold_D']!r}")
check("canonical rr.en_guard agrees (PASS, K 1, the same exact rational)", cg["result"] == "PASS" and cg["K"] == 1 and cg["D_EN_exact"] == f"{G_canon['D_exact'].numerator}/{G_canon['D_exact'].denominator}")

# ---------------------------------------------------------------------------------------------------------------------
# Item 10. The outcome from: validity (items 4-6: every gate passed, no incident), own ρ, the locked threshold, own K,
# and an own transcription of the design's Outcome section (table rows copied from the design text).
# ---------------------------------------------------------------------------------------------------------------------
MY_TABLE = [["an incident", "—", "no result is written"],
            ["NOT_INTERPRETABLE", "any", "NOT_INTERPRETABLE"],
            ["GUARD_FAILURE or ENVELOPE_ONLY_FAILURE", "any (descriptive only)", "NOUNNESS_PREDICTION_NOT_ESTABLISHED"],
            ["PASS", "FAIL or NOT_INTERPRETABLE", "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED"],
            ["PASS", "PASS", "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS"]]
check("own transcription of the design's outcome table == lock['outcome']['table'] (row for row)", MY_TABLE == lock["outcome"]["table"])
validity_ok = True  # established by items 3-6 (accounting exact, artifact digests, C bit for bit, I1/I3/I4 within tolerance): no incident
F_rho, null975 = lock["primary"]["F_rho"], lock["primary"]["null_975"]
if rho_own is None:
    primary = "NOT_INTERPRETABLE"
elif rho_own < null975:
    primary = "GUARD_FAILURE"
elif rho_own < F_rho:
    primary = "ENVELOPE_ONLY_FAILURE"
else:
    primary = "PASS"
guard_result = "PASS" if G["K"] <= 321 else "FAIL"
if not validity_ok:
    outcome = None
elif primary == "NOT_INTERPRETABLE":
    outcome = "NOT_INTERPRETABLE"
elif primary != "PASS":
    outcome = "NOUNNESS_PREDICTION_NOT_ESTABLISHED"
elif guard_result == "PASS":
    outcome = "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS"
else:
    outcome = "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED"
print(f"   derived: primary {primary} (ρ {rho_own!r} vs null₉₇.₅ {null975!r}, F_ρ {F_rho!r}); guard {guard_result} (K {G['K']}); outcome {outcome}")
check("derived outcome == NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS", outcome == "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS")
check("rr.outcome(primary, guard) (canonical frozen hierarchy) gives the same label", rr.outcome(primary, guard_result) == outcome)
# robustness: the outcome would be unchanged for any ρ ≥ threshold; the observed ρ exceeds it by
print(f"   margins: ρ − threshold = {rho_own - THRESH:+.6f}; K bound slack = {321 - G['K']}")

# ---------------------------------------------------------------------------------------------------------------------
# Only now: the recorded results, for comparison.
# ---------------------------------------------------------------------------------------------------------------------
state = json.loads((ROOT / "outputs/experiment-024/results.json").read_text())
R = state["confirmation"]["results"]
rec_mse = {c["token_id"]: c["mse"] for c in R["per_cue"]}
check("recorded per-cue order == the frozen cue order", [c["token_id"] for c in R["per_cue"]] == [o["token_id"] for o in own])
n_eq = sum(rec_mse[o["token_id"]] == o["mse"] for o in own)
max_rec = max(abs(rec_mse[o["token_id"]] - o["mse"]) / o["mse"] for o in own)
check("recorded per-cue MSE == canonical recomputation bit for bit (40/40); vs own fsum route max rel diff reported",
      all(rec_mse[t] == canon_by_tid[t] for t in rec_mse), f"own-vs-recorded bitwise {n_eq}/40, max rel {max_rec:.3e}")
check("recorded per-cue nounness == lock nounness; predicted_log_mse == lock's", all(c["nounness"] == nounness[c["token_id"]] for c in R["per_cue"])
      and all(c["predicted_log_mse"] == p["predicted_log_mse"] for c, p in zip(R["per_cue"], lock["fresh"]["cues"])))
check("recorded ρ 0.6108818011257036 == own ρ (≤ 1e-12) == float(1628/2665)", abs(R["primary"]["rho"] - rho_own) <= 1e-12 and R["primary"]["rho"] == float(rho_exact) == 0.6108818011257036,
      repr(R["primary"]["rho"]))
check("recorded primary PASS, thresholds the lock's", R["primary"]["result"] == "PASS" and R["primary"]["F_rho"] == F_rho and R["primary"]["null_975"] == null975)
check("recorded guard PASS, K 1, D_EN 0.5680373703974243, D_EN_exact == own exact rational (canonical MSE)",
      R["guard"]["result"] == "PASS" and R["guard"]["K"] == 1 and R["guard"]["D_EN"] == 0.5680373703974243
      and R["guard"]["D_EN_exact"] == f"{G_canon['D_exact'].numerator}/{G_canon['D_exact'].denominator}", R["guard"]["D_EN_exact"][:40] + "…")
check("recorded outcome label == derived", R["outcome"]["label"] == outcome)
json.dump({"per_cue": [{k: (v if not isinstance(v, Fraction) else str(v)) for k, v in o.items()} for o in own], "canonical_mse": canon_mse, "rho_own": rho_own,
           "rho_exact": str(rho_exact), "rho_canonical": rho_canon, "guard_own": {k: (str(v) if isinstance(v, Fraction) else v) for k, v in G.items() if k != "logs"},
           "guard_canon": {k: (str(v) if isinstance(v, Fraction) else v) for k, v in G_canon.items() if k != "logs"}, "outcome": outcome},
          open(SCR / "s07_results.json", "w"), indent=1)
print(f"S07 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
