"""Review items 11 (heterogeneity: real data or artifact) and 12 (the descriptive quantities, except the ladder).

Run only after r2 reproduced the official result. Own code on my own per-cue values (r2_values.pt) and my own
geometry (r3_geometry.json); compared with the recorded descriptive records. Descriptive only: no test, no p-value,
no new primary criterion. Writes r5_descriptives.json here.
"""
import guard  # noqa: F401

import json
import math
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(guard.ROOT)
HERE = Path(guard.HERE)
P, H, K = "0.32", "0.16", 7
vals = torch.load(HERE / "r2_values.pt", weights_only=True)
V, own_rows, FRAME_IDS = vals["V"], vals["own_rows"], vals["frame_ids"]
pair_sse, pair_d = vals["pair_sse"], vals["pair_d"]
geo = {e["token_id"]: e for e in json.loads((HERE / "r3_geometry.json").read_text())["per_cue"]}
state = json.loads((ROOT / "outputs/experiment-025/results.json").read_text())
recd = state["confirmation"]["descriptives"]["records"]
cal = json.loads((ROOT / "experiments/024-readout-routing-nounness/calibration-v1.json").read_text())
slope = float(cal["line"]["slope"])
out: dict = {}


def median(xs):
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def positive(xs):
    return sum(1 for x in xs if x is not None and math.isfinite(x) and x > 0.0)


# ------------------------------------------------------------------ item 12
rows = []
for r in own_rows:
    v = V[r["token_id"]]
    ell = {c: v[c]["ell"] for c in v}
    row = {"word": r["word"], "stratum": r["stratum"], "A": r["A"], "B": r["B"], "G": r["G"],
           "even_noun": 0.5 * (ell[f"noun+{P}"] + ell[f"noun-{P}"]) - ell["base"],
           "even_random": [0.5 * (ell[f"rand{j}+{P}"] + ell[f"rand{j}-{P}"]) - ell["base"] for j in range(1, K + 1)],
           "plus_above_base": ell[f"noun+{P}"] > ell["base"], "minus_below_base": ell[f"noun-{P}"] < ell["base"],
           "A_half": 0.5 * (ell[f"noun+{H}"] - ell[f"noun-{H}"]), "A_plurality": 0.5 * (ell[f"plur+{P}"] - ell[f"plur-{P}"]),
           "A_nmse": 0.5 * (math.log(v[f"noun+{P}"]["nmse"]) - math.log(v[f"noun-{P}"]["nmse"])),
           "G_random": [0.5 * (v[f"rand{j}+{P}"]["d_attn"] - v[f"rand{j}-{P}"]["d_attn"]) for j in range(1, K + 1)],
           "A_by_template": {t: 0.5 * (v[f"noun+{P}"]["ell_by_template"][t] - v[f"noun-{P}"]["ell_by_template"][t]) for t in v[f"noun+{P}"]["ell_by_template"]}}
    row["G_minus_random"] = row["G"] - sum(abs(x) for x in row["G_random"]) / K
    row["mean_abs_A_random"] = sum(abs(x) for x in r["A_random"]) / K
    rows.append(row)
mine = {"magnitudes": {n: {"mean": sum(r[n] for r in rows) / 40, "median": median([r[n] for r in rows])} for n in ("A", "B", "G")},
        "counts": {"plus_above_base": sum(r["plus_above_base"] for r in rows), "minus_below_base": sum(r["minus_below_base"] for r in rows),
                   "A_half_positive": positive([r["A_half"] for r in rows]), "A_plurality_positive": positive([r["A_plurality"] for r in rows]),
                   "A_nmse_positive": positive([r["A_nmse"] for r in rows]), "G_minus_random_positive": positive([r["G_minus_random"] for r in rows])},
        "A_positive_by_template": {t: positive([r["A_by_template"][t] for r in rows]) for t in sorted(rows[0]["A_by_template"])},
        "even_noun_mean": sum(r["even_noun"] for r in rows) / 40, "even_random_mean": sum(x for r in rows for x in r["even_random"]) / (40 * K)}
mine["causal_fraction"] = {"median_A": mine["magnitudes"]["A"]["median"], "slope_024": slope, "fraction": mine["magnitudes"]["A"]["median"] / (slope * 0.32)}
trigger = math.ceil(27 * 20 / 40)
mine["strata_trigger"] = trigger
out["item12_mine"] = mine
cmp = {"magnitudes_max_diff": max(abs(mine["magnitudes"][n][k] - recd["magnitudes"][n][k]) for n in ("A", "B", "G") for k in ("mean", "median")),
       "counts_equal": mine["counts"] == recd["counts"], "recorded_counts": recd["counts"],
       "templates_equal": mine["A_positive_by_template"] == recd["A_positive_by_template"], "recorded_templates": recd["A_positive_by_template"],
       "even_max_diff": max(abs(mine["even_noun_mean"] - recd["even_noun_mean"]), abs(mine["even_random_mean"] - recd["even_random_mean"])),
       "causal_fraction_diff": abs(mine["causal_fraction"]["fraction"] - recd["causal_fraction"]["fraction"]), "recorded_causal_fraction": recd["causal_fraction"],
       "trigger_equal": recd["strata_trigger"] == {"adjective": trigger, "noun": trigger},
       "strata_recorded": recd["strata"]}
per_cue_diff = 0.0
for a, b in zip(rows, recd["per_cue"]):
    for k in ("even_noun", "A_half", "A_plurality", "A_nmse", "G_minus_random"):
        per_cue_diff = max(per_cue_diff, abs(a[k] - b[k]))
    per_cue_diff = max(per_cue_diff, max(abs(x - y) for x, y in zip(a["G_random"], b["G_random"])), max(abs(x - y) for x, y in zip(a["even_random"], b["even_random"])))
    assert a["plus_above_base"] == b["plus_above_base"] and a["minus_below_base"] == b["minus_below_base"] and a["word"] == b["word"]
cmp["per_cue_max_diff"] = per_cue_diff
cmp["no_stratum_flag_possible_for_failed_statistics"] = all(not recd["strata"][n][s]["below_trigger"] for n in ("A", "B") for s in ("adjective", "noun"))
out["item12_vs_recorded"] = cmp

# ------------------------------------------------------------------ item 11: heterogeneity, descriptive
het = {}
for s in ("adjective", "noun"):
    R = [r for r in rows if r["stratum"] == s]
    O = [r for r in own_rows if r["stratum"] == s]
    frac_plus = []
    frac_plus_by_t = defaultdict(list)
    frac_g = []
    for r in O:
        a = pair_sse[f"{r['token_id']}|noun+{P}"]
        b = pair_sse[f"{r['token_id']}|noun-{P}"]
        frac_plus.append(float((a > b).double().mean()))
        for t in ("cardinal", "coordinated-adjective", "quantifier"):
            idx = [i for i, f in enumerate(FRAME_IDS) if f.startswith(t + "-") and not (t == "cardinal" and f.startswith("coordinated"))]
            frac_plus_by_t[t].append(float((a[idx] > b[idx]).double().mean()))
        frac_g.append(float((pair_d[f"{r['token_id']}|noun+{P}"] > pair_d[f"{r['token_id']}|noun-{P}"]).double().mean()))
    het[s] = {"A": [min(r["A"] for r in R), median([r["A"] for r in R]), max(r["A"] for r in R)],
              "B": [min(r["B"] for r in R), median([r["B"] for r in R]), max(r["B"] for r in R)],
              "G": [min(r["G"] for r in R), median([r["G"] for r in R]), max(r["G"] for r in R)],
              "mean_abs_A_random_median": median([r["mean_abs_A_random"] for r in R]),
              "base_ell_median": median([V[r["token_id"]]["base"]["ell"] for r in O]),
              "plus_above_base": sum(r["plus_above_base"] for r in R), "minus_below_base": sum(r["minus_below_base"] for r in R),
              "A_half_positive": positive([r["A_half"] for r in R]), "A_nmse_positive": positive([r["A_nmse"] for r in R]),
              "A_positive_by_template": {t: positive([r["A_by_template"][t] for r in R]) for t in sorted(R[0]["A_by_template"])},
              "frames_sse_plus_gt_minus_median_fraction": median(frac_plus), "frames_sse_plus_gt_minus_range": [min(frac_plus), max(frac_plus)],
              "frames_sse_plus_gt_minus_by_template_median": {t: median(v) for t, v in frac_plus_by_t.items()},
              "frames_dattn_plus_gt_minus_median_fraction": median(frac_g),
              "s0_range": [min(geo[r["token_id"]]["s0"] for r in O), max(geo[r["token_id"]]["s0"] for r in O)]}
    het[s]["template_frame_counts"] = {t: len([f for f in FRAME_IDS if f.startswith(t + "-")]) for t in ("cardinal", "coordinated-adjective", "quantifier")}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return rk


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


s0 = [geo[r["token_id"]]["s0"] for r in own_rows]
het["spearman_A_s0_all"] = spearman([r["A"] for r in own_rows], s0)
for s in ("adjective", "noun"):
    idx = [i for i, r in enumerate(own_rows) if r["stratum"] == s]
    het[f"spearman_A_s0_{s}"] = spearman([own_rows[i]["A"] for i in idx], [s0[i] for i in idx])
# the strata separate exactly on the sign of the starting score s0
het["strata_equal_sign_of_s0"] = all((geo[r["token_id"]]["s0"] < 0) == (r["stratum"] == "adjective") for r in own_rows)
# no accidental use of another condition: A differs from the plurality, half-dose and every random-control analogue
het["A_equals_other_condition_count"] = sum(1 for r, o in zip(rows, own_rows) if r["A"] in (r["A_plurality"], r["A_half"], *o["A_random"]))
het["per_cue"] = [{"word": r["word"], "stratum": r["stratum"], "s0": geo[o["token_id"]]["s0"], "A": r["A"], "B": r["B"], "G": r["G"],
                   "mean_abs_A_random": r["mean_abs_A_random"], "A_half": r["A_half"], "A_plurality": r["A_plurality"],
                   "ell_base": V[o["token_id"]]["base"]["ell"], "ell_plus": V[o["token_id"]][f"noun+{P}"]["ell"], "ell_minus": V[o["token_id"]][f"noun-{P}"]["ell"]}
                  for r, o in zip(rows, own_rows)]
out["item11_heterogeneity"] = het
out["guard"] = guard.EVENTS
(HERE / "r5_descriptives.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"mine": mine, "vs": {k: v for k, v in cmp.items() if k != "strata_recorded"}}, indent=1))
print(json.dumps({k: v for k, v in het.items() if k != "per_cue"}, indent=1))
for e in het["per_cue"]:
    print(f"{e['stratum']:9s} {e['word']:10s} s0 {e['s0']:+.3f}  A {e['A']:+.4f}  B {e['B']:+.4f}  G {e['G']:+.5f}  mean|Aj| {e['mean_abs_A_random']:.4f}  "
          f"ell0 {e['ell_base']:+.3f} ell+ {e['ell_plus']:+.3f} ell- {e['ell_minus']:+.3f}  Ahalf {e['A_half']:+.4f}  Aplur {e['A_plurality']:+.4f}")
