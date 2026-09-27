"""Review items 4, 6, 7, 8, 9, 10, 11 (data side), 12 (except the ladder).

Phase 1 (own code only; the module's per_cue / statistics / count_positive / routing_distance / outcome_label and
rr.fresh_pair_cells / rr.cue_mse are refused while it runs): the artifact's schema, shapes, finiteness and digests;
the row and reference-row alignment; MSE, ℓ, nMSE and D_attn for every cue and condition; A, A_ij, B, G, the counts,
the threshold and the label. Phase 2: the cross-checks against the recorded state and against the module.
No model. The artifact is memory-mapped read-only, once. Writes r2_scores.json and r2_values.pt here.
"""
import guard  # noqa: F401

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul

ROOT = Path(guard.ROOT)
HERE = Path(guard.HERE)
EXP = ROOT / "experiments/025-nounness-direction-intervention"
ART = ROOT / "outputs/experiment-025/stage2-measurements.pt"
guard.seal()
REFUSED_CR = ("per_cue", "statistics", "count_positive", "routing_distance", "outcome_label", "descriptives", "ladder", "run_gates", "identity_gates",
              "level1_gates", "recompute_c")
saved_cr = guard.refuse_functions(cr, REFUSED_CR, "cr")
saved_rr = guard.refuse_functions(rr, ("fresh_pair_cells", "cue_mse", "cue_mse_torch"), "rr")
out: dict = {}

freeze = json.loads((EXP / "confirmation-v1.json").read_text())
lock = json.loads((EXP / "preregistration-lock.json").read_text())
state = json.loads((ROOT / "outputs/experiment-025/results.json").read_text())
CONDITIONS = list(freeze["conditions"])
CUES = [{"word": c["word"], "token_id": int(c["token_id"]), "stratum": c["stratum"]} for c in freeze["cues"]]
P, H, K = "0.32", "0.16", 7

inputs = ul.load_frozen_inputs(ROOT, tracked=guard.tracked)
frames_by_id = {f.frame_id: f for f in inputs.pool.frames}
assert sorted(frames_by_id) == sorted(freeze["exposed_frame_ids"]) and len(frames_by_id) == 108
locked = inputs.closure["exploration"]["locked_states"]


def group_of(frame):
    return "coordinated" if frame.template_id == "coordinated-adjective" else "cue_final"


FRAME_IDS = sorted(frames_by_id)
GROUP_FRAMES = {g: [fid for fid in FRAME_IDS if group_of(frames_by_id[fid]) == g] for g in ("cue_final", "coordinated")}
TOKENS_BY_ID = sorted(CUES, key=lambda c: c["token_id"])
T_RANK = {c["token_id"]: i for i, c in enumerate(TOKENS_BY_ID)}


def row_of(group, token_id, frame_id):  # my own restatement of the storage order: token id, then frame_id, per group
    return T_RANK[token_id] * len(GROUP_FRAMES[group]) + GROUP_FRAMES[group].index(frame_id)


# ------------------------------------------------------------------ item 4: the artifact
art = {"bytes": ART.stat().st_size}
digest = hashlib.sha256()
with open(ART, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 22), b""):
        digest.update(chunk)
art["sha256"] = digest.hexdigest()
art["sha256_ok"] = art["sha256"] == "87284bddfba16a1b7709d7d33f98aa1a0898b1d85767956a524d2950d80a7788" and art["bytes"] == 898673151
saved = torch.load(ART, mmap=True, weights_only=True)
expected_names = {f"{g}/{x}" for g in GROUP_FRAMES for x in ("positions", "keys")} | {f"{g}/{c}/{n}" for g in GROUP_FRAMES for c in CONDITIONS
                                                                                   for n in ("x1", "x3", "dc", "C", "rows4", "rows5")}
art["n_tensors"] = len(saved)
art["schema_equal"] = set(saved) == expected_names
recorded_digests = state["confirmation"]["stage2"]["tensors_sha256"]


def own_digest(tensor):
    array = tensor.detach().cpu().contiguous()
    kind = "<i8" if array.dtype in (torch.int64, torch.int32) else "<f8"
    data = array.numpy().astype(kind, copy=False).tobytes()
    return hashlib.sha256(json.dumps(list(array.shape)).encode("ascii") + b"|" + kind.encode("ascii") + b"|" + data).hexdigest()


dig = {"rc_equal": 0, "own_equal": 0, "mismatch": []}
shapes, dtypes, finite = {}, {}, {}
for name in sorted(saved):
    t = saved[name]
    a, b = rc.tensor_digest(t), own_digest(t)
    dig["rc_equal"] += a == recorded_digests.get(name)
    dig["own_equal"] += b == recorded_digests.get(name)
    if a != recorded_digests.get(name) or b != a:
        dig["mismatch"].append(name)
    shapes[name] = list(t.shape)
    dtypes[name] = str(t.dtype)
    finite[name] = bool(torch.isfinite(t).all()) if t.is_floating_point() else True
art["digests"] = dig
art["digest_keys_equal_state"] = set(recorded_digests) == set(saved)
art["all_finite"] = all(finite.values())
n_rows = {g: 40 * len(GROUP_FRAMES[g]) for g in GROUP_FRAMES}
kmax = {g: max(frames_by_id[f].p_t + 1 for f in GROUP_FRAMES[g]) for g in GROUP_FRAMES}
shape_ok = True
for g in GROUP_FRAMES:
    shape_ok &= shapes[f"{g}/positions"] == [n_rows[g], 2] and shapes[f"{g}/keys"] == [n_rows[g]]
    for c in CONDITIONS:
        shape_ok &= shapes[f"{g}/{c}/x1"] == [n_rows[g], 2, 512] and shapes[f"{g}/{c}/x3"] == [n_rows[g], 2, 512]
        shape_ok &= shapes[f"{g}/{c}/dc"] == [n_rows[g], 79] and shapes[f"{g}/{c}/C"] == [n_rows[g], 79]
        shape_ok &= shapes[f"{g}/{c}/rows4"] == [n_rows[g], 8, kmax[g]] and shapes[f"{g}/{c}/rows5"] == [n_rows[g], 8, kmax[g]]
        shape_ok &= dtypes[f"{g}/{c}/x1"] == "torch.float32" and dtypes[f"{g}/{c}/dc"] == "torch.float64" and dtypes[f"{g}/{c}/rows4"] == "torch.float32"
art["shapes_ok"] = bool(shape_ok)
art["rows_per_group"] = n_rows
art["keys_dim"] = kmax
art["slots"] = sum(n_rows.values()) * len(CONDITIONS)
out["item4_artifact"] = art
print("artifact", json.dumps({k: v for k, v in art.items() if k != "digests"}), "digests", dig["rc_equal"], dig["own_equal"], len(dig["mismatch"]), flush=True)

# ------------------------------------------------------------------ item 6: alignment
align = {"positions_ok": True, "keys_ok": True, "cue_final_slot_duplicate_ok": True, "ref_rows_shape_ok": True, "beyond_keys_zero": True,
         "captured_row_sum_max_dev": 0.0, "ref_row_sum_max_dev": 0.0}
REF = {}
for fid in FRAME_IDS:
    fr = frames_by_id[fid]
    r4 = torch.tensor(locked[fid]["rows4"], dtype=torch.float64)
    r5 = torch.tensor(locked[fid]["rows5"], dtype=torch.float64)
    align["ref_rows_shape_ok"] &= list(r4.shape) == [8, fr.p_t + 1] and list(r5.shape) == [8, fr.p_t + 1]
    align["ref_row_sum_max_dev"] = max(align["ref_row_sum_max_dev"], float((r4.sum(-1) - 1).abs().max()), float((r5.sum(-1) - 1).abs().max()))
    align["ref_p"] = align.get("ref_p", True) and int(locked[fid]["p_c"]) == fr.p_c and int(locked[fid]["p_t"]) == fr.p_t
    REF[fid] = (r4, r5)
for g in GROUP_FRAMES:
    pos, keys = saved[f"{g}/positions"], saved[f"{g}/keys"]
    for c in CUES:
        for fid in GROUP_FRAMES[g]:
            fr = frames_by_id[fid]
            r = row_of(g, c["token_id"], fid)
            align["positions_ok"] &= pos[r].tolist() == [fr.p_c, fr.p_t]
            align["keys_ok"] &= int(keys[r]) == fr.p_t + 1
    for cond in CONDITIONS:
        k = keys.tolist()
        for name in ("rows4", "rows5"):
            t = saved[f"{g}/{cond}/{name}"]
            mask = torch.arange(t.shape[-1]).view(1, 1, -1) >= torch.tensor(k).view(-1, 1, 1)
            align["beyond_keys_zero"] &= bool((t.masked_fill(~mask, 0) == 0).all())
            align["captured_row_sum_max_dev"] = max(align["captured_row_sum_max_dev"], float((t.double().sum(-1) - 1).abs().max()))
        if g == "cue_final":
            for name in ("x1", "x3"):
                t = saved[f"{g}/{cond}/{name}"]
                align["cue_final_slot_duplicate_ok"] &= bool(torch.equal(t[:, 0], t[:, 1]))
out["item6_alignment"] = align
print("alignment", json.dumps(align), flush=True)

# ------------------------------------------------------------------ item 6: own MSE, ℓ, nMSE, D_attn
V = {}  # token_id -> condition -> quantities
pair_sse = {}  # (token_id, cond) -> tensor of 108 per-frame SSE (FRAME_IDS order), for the heterogeneity analysis
pair_d = {}
REF_T = {}
for g in GROUP_FRAMES:  # the locked 020 rows placed at each stored row's own frame, zero-padded beyond p_t
    for idx in (0, 1):
        ref = torch.zeros(n_rows[g], 8, kmax[g], dtype=torch.float64)
        for c in CUES:
            for fid in GROUP_FRAMES[g]:
                rr_ = REF[fid][idx]
                ref[row_of(g, c["token_id"], fid), :, : rr_.shape[1]] = rr_
        REF_T[(g, idx)] = ref
for cond in CONDITIONS:
    per = {g: {} for g in GROUP_FRAMES}
    for g in GROUP_FRAMES:
        dc, C = saved[f"{g}/{cond}/dc"].double(), saved[f"{g}/{cond}/C"].double()
        sse = ((dc - C) ** 2).sum(-1)
        q = (dc ** 2).sum(-1)
        tvs = []
        for name, idx in (("rows4", 0), ("rows5", 1)):
            a = saved[f"{g}/{cond}/{name}"].double()
            tvs.append(0.5 * (a - REF_T[(g, idx)]).abs().sum(-1))  # [rows, 8]
        d_pair = torch.cat(tvs, dim=1).mean(dim=1)  # the equal mean over 16 heads
        per[g] = {"sse": sse, "q": q, "d": d_pair}
    for c in CUES:
        tid = c["token_id"]
        sse_list, q_list, d_list, by_t = [], [], [], defaultdict(list)
        for fid in FRAME_IDS:
            g = group_of(frames_by_id[fid])
            r = row_of(g, tid, fid)
            s = float(per[g]["sse"][r])
            sse_list.append(s)
            q_list.append(float(per[g]["q"][r]))
            d_list.append(float(per[g]["d"][r]))
            by_t[frames_by_id[fid].template_id].append(s)
        mse = math.fsum(sse_list) / (79 * len(sse_list))
        V.setdefault(tid, {})[cond] = {"mse": mse, "ell": math.log(mse), "nmse": math.fsum(sse_list) / math.fsum(q_list),
                                       "d_attn": math.fsum(d_list) / len(d_list), "frames": len(sse_list),
                                       "ell_by_template": {t: math.log(math.fsum(v) / (79 * len(v))) for t, v in sorted(by_t.items())}}
        pair_sse[(tid, cond)] = torch.tensor(sse_list, dtype=torch.float64)
        pair_d[(tid, cond)] = torch.tensor(d_list, dtype=torch.float64)

# ------------------------------------------------------------------ items 7–10: own A, B, G, counts, threshold, label
def binom_tail(n, k):
    return Fraction(sum(math.comb(n, j) for j in range(k, n + 1)), 2 ** n)


threshold = next(k for k in range(41) if binom_tail(40, k) <= Fraction(1, 40))
tail = binom_tail(40, threshold)
own_rows = []
for c in CUES:
    v = V[c["token_id"]]
    a = 0.5 * (v[f"noun+{P}"]["ell"] - v[f"noun-{P}"]["ell"])
    aj = [0.5 * (v[f"rand{j}+{P}"]["ell"] - v[f"rand{j}-{P}"]["ell"]) for j in range(1, K + 1)]
    b = a - math.fsum(abs(x) for x in aj) / K
    gval = 0.5 * (v[f"noun+{P}"]["d_attn"] - v[f"noun-{P}"]["d_attn"])
    own_rows.append({"word": c["word"], "token_id": c["token_id"], "stratum": c["stratum"], "A": a, "A_random": aj, "B": b, "G": gval})


def positive(values):
    return sum(1 for x in values if x is not None and math.isfinite(x) and x > 0.0)


own_counts = {n: positive([r[n] for r in own_rows]) for n in ("A", "B", "G")}
own_by_stratum = {n: {s: positive([r[n] for r in own_rows if r["stratum"] == s]) for s in ("adjective", "noun")} for n in ("A", "B", "G")}
own_pass = {n: own_counts[n] >= threshold for n in own_counts}
gates = state["confirmation"]["gates"]
gates_pass = all(gates[n]["max"] is not None and gates[n]["max"] <= gates["tolerances"][n] for n in ("I1", "I3", "I4", "level1_outcome_bearing"))
if not gates_pass:
    own_label = "NOT_INTERPRETABLE"
elif not own_pass["A"]:
    own_label = "CAUSAL_EFFECT_NOT_ESTABLISHED"
elif not own_pass["B"]:
    own_label = "DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC"
elif not own_pass["G"]:
    own_label = "READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED"
else:
    own_label = "NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR"
own = {"threshold": threshold, "tail_exact": f"{tail.numerator}/{tail.denominator}", "tail": float(tail), "tail_26": float(binom_tail(40, 26)),
       "counts": own_counts, "by_stratum": own_by_stratum, "pass": own_pass, "label": own_label, "gates_pass_recorded_values": gates_pass,
       "zeros_or_ties": {n: sum(1 for r in own_rows if r[n] == 0.0) for n in ("A", "B", "G")},
       "min_abs": {n: min(abs(r[n]) for r in own_rows) for n in ("A", "B", "G")},
       "rows": own_rows}
out["own_result"] = own
print("OWN:", json.dumps({k: v for k, v in own.items() if k != "rows"}), flush=True)

# ------------------------------------------------------------------ phase 2: cross-check against the recorded state
rec = state["confirmation"]["results"]
xc = {"cue_order_equal": [(r["word"], r["token_id"], r["stratum"]) for r in rec["per_cue"]] == [(c["word"], c["token_id"], c["stratum"]) for c in CUES]}
maxdiff = defaultdict(float)
sign_equal = {n: True for n in ("A", "B", "G")}
for mine, theirs in zip(own_rows, rec["per_cue"]):
    for n in ("A", "B", "G"):
        maxdiff[n] = max(maxdiff[n], abs(mine[n] - theirs[n]))
        sign_equal[n] &= (mine[n] > 0) == (theirs[n] > 0)
    maxdiff["A_random"] = max(maxdiff["A_random"], max(abs(x - y) for x, y in zip(mine["A_random"], theirs["A_random"])))
for resp in rec["responses"]:
    for cond in CONDITIONS:
        mine, theirs = V[resp["token_id"]][cond], resp["conditions"][cond]
        for q in ("mse", "ell", "nmse", "d_attn"):
            maxdiff[q] = max(maxdiff[q], abs(mine[q] - theirs[q]))
            maxdiff[q + "_rel"] = max(maxdiff[q + "_rel"], abs(mine[q] - theirs[q]) / abs(theirs[q]))
        maxdiff["frames_mismatch"] += mine["frames"] != theirs["frames"]
        for t in mine["ell_by_template"]:
            maxdiff["ell_by_template"] = max(maxdiff["ell_by_template"], abs(mine["ell_by_template"][t] - theirs["ell_by_template"][t]))
xc["max_abs_disagreement_vs_state"] = dict(maxdiff)
xc["signs_equal_vs_state"] = sign_equal
xc["counts_equal"] = {n: (own_counts[n], rec[n]["positive"]) for n in ("A", "B", "G")}
xc["by_stratum_equal"] = {n: own_by_stratum[n] == rec[n]["by_stratum"] for n in ("A", "B", "G")}
xc["results_equal"] = {n: ("PASS" if own_pass[n] else "FAIL") == rec[n]["result"] and rec[n]["threshold"] == threshold and rec[n]["n"] == 40 for n in ("A", "B", "G")}
xc["label_equal"] = own_label == rec["outcome"]["label"]
xc["tail_equal"] = rec["criterion"]["reference_tail"] == {"exact": own["tail_exact"], "value": own["tail"]}
xc["reading_equal_semantics"] = rec["outcome"]["reading"] == cr.SEMANTICS["outcomes"][own_label] and rec["outcome"]["not_claimed"] == cr.SEMANTICS["not_claimed"]
# internal consistency of the recorded state: A, B, G from its own recorded responses
cons = 0.0
for r, resp in zip(rec["per_cue"], rec["responses"]):
    v = resp["conditions"]
    a = 0.5 * (v[f"noun+{P}"]["ell"] - v[f"noun-{P}"]["ell"])
    g = 0.5 * (v[f"noun+{P}"]["d_attn"] - v[f"noun-{P}"]["d_attn"])
    cons = max(cons, abs(a - r["A"]), abs(g - r["G"]))
xc["recorded_ABG_from_recorded_responses_max_diff"] = cons
out["cross_check_state"] = xc
print("XC-STATE:", json.dumps(xc), flush=True)

# ------------------------------------------------------------------ phase 2: cross-check against the module (only now)
guard.restore_functions(cr, saved_cr)
guard.restore_functions(rr, saved_rr)
confirmation = cr.confirmation_from_payload(freeze, inputs.pool, cr.PRODUCTION)
units = cr.target_units(confirmation)
states = ul.y1_states(locked, units.frames)
values = cr.per_cue(units, saved, states, confirmation.conditions)
module_stats = cr.statistics(values, confirmation, cr.PRODUCTION)
xm = defaultdict(float)
for tid, conds in values.items():
    for cond, q in conds.items():
        for name in ("mse", "ell", "nmse", "d_attn"):
            xm[name] = max(xm[name], abs(q[name] - V[tid][cond][name]))
for a, b in zip(module_stats["per_cue"], own_rows):
    for n in ("A", "B", "G"):
        xm[n] = max(xm[n], abs(a[n] - b[n]))
xm_out = {"max_abs_disagreement_module_vs_own": dict(xm), "module_statistics_equal_recorded": json.loads(json.dumps(module_stats)) == rec,
          "module_counts": {n: module_stats[n]["positive"] for n in ("A", "B", "G")}, "module_label": module_stats["outcome"]["label"]}
# routing_distance and the cell functions on a sample of runs, against my own per-run values
rd_diff, cell_diff = 0.0, 0.0
for i, fid in enumerate(FRAME_IDS[::7]):
    g = group_of(frames_by_id[fid])
    for c in CUES[::5]:
        r = row_of(g, c["token_id"], fid)
        for cond in (f"noun+{P}", f"noun-{P}", "base"):
            d = cr.routing_distance(saved[f"{g}/{cond}/rows4"][r], saved[f"{g}/{cond}/rows5"][r], states[fid])
            rd_diff = max(rd_diff, abs(d - float(pair_d[(c["token_id"], cond)][FRAME_IDS.index(fid)])))
            cells = rr.fresh_pair_cells(saved[f"{g}/{cond}/dc"][r], saved[f"{g}/{cond}/C"][r])
            cell_diff = max(cell_diff, abs(float(cells[rr.SSEC]) - float(pair_sse[(c["token_id"], cond)][FRAME_IDS.index(fid)])),
                            abs(float(cells[rr.COUNT]) - 79.0), abs(float(cells[rr.SQUARES]) - float((saved[f"{g}/{cond}/dc"][r].double() ** 2).sum())))
xm_out["routing_distance_sample_max_diff"] = rd_diff
xm_out["fresh_pair_cells_sample_max_diff"] = cell_diff
xm_out["count_positive_equal"] = all(cr.count_positive([r[n] for r in own_rows]) == own_counts[n] for n in ("A", "B", "G"))
xm_out["count_positive_ties"] = cr.count_positive([0.0, -0.0, float("nan"), None, 1e-300, -1e-300]) == 1
# outcome_label: the whole truth table; after an A failure no other label is reachable
truth = {f"{a}{b}{g}": cr.outcome_label(a, b, g) for a in (False, True) for b in (False, True) for g in (False, True)}
xm_out["outcome_label_truth_table"] = truth
xm_out["A_fail_always_CAUSAL_EFFECT_NOT_ESTABLISHED"] = all(truth[f"False{b}{g}"] == "CAUSAL_EFFECT_NOT_ESTABLISHED" for b in (False, True) for g in (False, True))
# condition-name parsing
xm_out["spec"] = {c: cr.PRODUCTION.spec(c) for c in CONDITIONS}
xm_out["spec_ok"] = cr.PRODUCTION.spec(f"noun+{P}") == {"kind": "noun", "sign": 1, "odd": 0.32, "j": None} and cr.PRODUCTION.spec(f"noun-{P}") == {
    "kind": "noun", "sign": -1, "odd": 0.32, "j": None} and all(cr.PRODUCTION.spec(f"rand{j}{s}{P}") == {"kind": "rand", "sign": sg, "odd": 0.32, "j": j}
                                                                for j in range(1, 8) for s, sg in (("+", 1), ("-", -1)))
xm_out["outcome_bearing"] = list(cr.PRODUCTION.outcome_bearing)
xm_out["threshold_module"] = (cr.PRODUCTION.count_threshold, cr.derive_threshold(40), float(cr.PRODUCTION.reference_tail()))
out["cross_check_module"] = xm_out
print("XC-MODULE:", json.dumps({k: v for k, v in xm_out.items() if k not in ("spec",)}), flush=True)

torch.save({"V": V, "pair_sse": {f"{k[0]}|{k[1]}": v for k, v in pair_sse.items()}, "pair_d": {f"{k[0]}|{k[1]}": v for k, v in pair_d.items()},
            "frame_ids": FRAME_IDS, "own_rows": own_rows}, HERE / "r2_values.pt")
out["guard"] = guard.EVENTS
(HERE / "r2_scores.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print("done; guard", json.dumps(guard.EVENTS)[:300])
