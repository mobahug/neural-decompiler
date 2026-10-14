"""Checks 3–7 and 10, 12 from the saved data — independent: no neural_decompiler import, no model object, no forward pass.

Weights are read from the pinned safetensors (float16 on disk, exact in float64) only for b_O[3], LN_final and the
unembedding rows of the 79 scorable nouns. Every quantity is recomputed from the saved tensors with my own code.
"""
import json
import math
import sys
from collections import defaultdict

import torch

from common import (CAL, CUES, OUTPUTS_101, PRIMARY, READ_COEFF, REF_ID, RESULTS_020, ROOT, SNAPSHOT, WRITE_ATTEMPTS, arm_kind, canonical, dump,
                    float32_digest, frame_runs, my_tensor_digest, sha256_file, sha256_text)

assert not [m for m in sys.modules if m.startswith("neural_decompiler")]
from safetensors import safe_open  # noqa: E402

torch.set_grad_enabled(False)
out: dict = {}

# ---------------------------------------------------------------------------------------------------------------
# 3. The state.
state_text = (OUTPUTS_101 / "results.json").read_text()
state = json.loads(state_text)
payload = {k: v for k, v in state.items() if k != "state_sha256"}
out["state"] = {"file_sha256": sha256_file(OUTPUTS_101 / "results.json"), "state_sha256_recorded": state["state_sha256"],
                "state_sha256_recomputed": sha256_text(canonical(payload)),
                "file_is_canonical_plus_newline": state_text == canonical({**payload, "state_sha256": state["state_sha256"]}) + "\n",
                "phases": state["phases"], "executed_prompt_keys": state["executed_prompt_keys"], "protocol_code_commit": state["protocol_code_commit"],
                "git_dirty": state["git_dirty"], "lock": state["lock"], "stage1": state["stage1"], "confirmation": state["confirmation"], "report": state["report"],
                "versions": state["versions"], "calibration_keys": sorted(state["calibration"].keys()), "stopped_present": "stopped" in state["calibration"],
                "incidents_present": "incidents" in state["calibration"], "partial_present": "partial" in state["calibration"]}
conf_path = ROOT / "experiments/101-read-component-intervention/confirmation-v1.json"
conf = json.loads(conf_path.read_text())
out["state"]["confirmation_binding"] = {"recorded": state["confirmation_101"], "file_sha256_now": sha256_file(conf_path),
                                        "content_sha256_now": sha256_text(canonical({k: v for k, v in conf.items() if k != "content_sha256"})),
                                        "manifest_sha256_now": sha256_text(canonical(conf["manifest"]))}
cand_path = OUTPUTS_101 / "candidate-calibration-v1.json"
cand_text = cand_path.read_text()
cand = json.loads(cand_text)
out["candidate"] = {"file_sha256": sha256_file(cand_path), "state_file_sha256": state["calibration"]["file_sha256"],
                    "content_sha256_recorded": cand["content_sha256"], "content_sha256_recomputed": sha256_text(canonical({k: v for k, v in cand.items() if k != "content_sha256"})),
                    "state_content_sha256": state["calibration"]["content_sha256"], "file_is_canonical_plus_newline": cand_text == canonical(cand) + "\n",
                    "tau_state_equals_candidate": state["calibration"]["tau"] == cand["tau"],
                    "gates_state_equals_candidate": state["calibration"]["gates"] == cand["gates"],
                    "measurements_state_equals_candidate": state["calibration"]["measurements"]["tensors_sha256"] == cand["measurements_sha256"]}

# ---------------------------------------------------------------------------------------------------------------
# 4. The measurements.
meas_path = OUTPUTS_101 / "calibration-measurements.pt"
saved = torch.load(meas_path, weights_only=True)
recorded = state["calibration"]["measurements"]["tensors_sha256"]
mine = {name: my_tensor_digest(t) for name, t in saved.items()}
runs = frame_runs()
tags = [r[0] for r in runs]
frames020 = json.loads((OUTPUTS_101.parent / "experiment-020/results.json").read_text())
locked = frames020["exploration"]["locked_states"]
c020 = json.loads((ROOT / "experiments/020-readout-decompilation/confirmation-v1.json").read_text())
frame_ids = [f for f in c020["exposed_frame_ids"] if f.startswith("coordinated-adjective")]
expected = set()
for fid in frame_ids:
    expected |= {f"{fid}/{n}" for n in ("contrasts", "logit_identity", "x3_pc", "x3_pt", "x4_pc", "heads_pc", "mlp_pc", "T_pt", "attn_pt", "h6", "positions", "c_ref_pm")}
    for tag, arm, cue, on_ref in runs:
        kind = arm_kind(arm)
        if kind == "plain":
            expected.add(f"{fid}/x3_all/{tag}")
        elif kind == "head":
            expected |= {f"{fid}/delta/{tag}", f"{fid}/head_replacement/{tag}"}
        else:
            expected.add(f"{fid}/replacement/{tag}")
nonfinite = [n for n, t in saved.items() if t.is_floating_point() and not bool(torch.isfinite(t).all())]
out["measurements"] = {"file_sha256": sha256_file(meas_path), "tensors": len(saved), "expected": len(expected), "names_equal_expected": set(saved) == expected,
                       "digests_equal_recorded": mine == recorded, "n_digest_mismatch": sum(1 for k in recorded if mine.get(k) != recorded[k]),
                       "nonfinite_tensors": nonfinite, "per_frame": len(expected) // 36,
                       "row_counts_ok": all(saved[f"{fid}/contrasts"].shape[0] == 55 for fid in frame_ids)}

# ---------------------------------------------------------------------------------------------------------------
# Weights (read-only, from the safetensors; float16 on disk → exact in float64).
st = safe_open(str(SNAPSHOT / "model.safetensors"), "pt")
W = lambda k: st.get_tensor(k).double()  # noqa: E731
b_O3 = W("gpt_neox.layers.3.attention.dense.bias")
lnf_w, lnf_b = W("gpt_neox.final_layer_norm.weight"), W("gpt_neox.final_layer_norm.bias")
W_U = W("embed_out.weight")  # [vocab, d]
eps = 1e-5
lock011 = json.loads((ROOT / "experiments/011-encoding-read-prospective/preregistration-lock.json").read_text())
rw = torch.tensor(lock011["read_weight"], dtype=torch.float64)
w_c = rw - rw.mean()
w_norm = float(torch.linalg.vector_norm(w_c))
w_hat = w_c / w_norm
d_T = torch.tensor(lock011["axes_vectors"]["T"], dtype=torch.float64)
gamma3 = W("gpt_neox.layers.3.input_layernorm.weight")
gamma_ln2 = W("gpt_neox.layers.3.post_attention_layernorm.weight")
qkv3 = W("gpt_neox.layers.3.attention.query_key_value.weight")  # [3*512, 512]: per head h rows h*192 + (0..63 q, 64..127 k, 128..191 v)
dense3 = W("gpt_neox.layers.3.attention.dense.weight")  # [512 out, 512 in]
H = 4
W_V4 = qkv3[H * 192 + 128: H * 192 + 192, :].T  # [512, 64]: v = ln(x) @ W_V
W_O4 = dense3[:, H * 64:(H + 1) * 64].T  # [64, 512]
m = W_V4 @ (W_O4 @ d_T)
rw_weights = gamma3 * m
out["G1_mine"] = {"w_c_norm": w_norm, "read_weight_from_weights_vs_lock_max_abs": float((rw_weights - rw).abs().max()),
                  "centred_vs_w_c_max_abs": float(((rw_weights - rw_weights.mean()) - w_c).abs().max()),
                  "d_T_norm": float(torch.linalg.vector_norm(d_T)), "lock011_head": lock011.get("head")}

# The 79 scorable nouns (020's exposed noun order; ids from the committed screening manifest)
manifest = json.loads((ROOT / "screening/behavior-candidates/manifest-v1.json").read_text())
ids_by_key = {}
for case in manifest["cases"]:
    if case["candidate_id"] != "regular-plural":
        continue
    cond = case["conditions"]["s_a"]
    entry = (tuple(cond["a_token_ids"]), tuple(cond["b_token_ids"]))
    assert ids_by_key.setdefault(case["lexical_key"], entry) == entry
c006 = json.loads((ROOT / "experiments/006-low-rank-cue-decompilation/confirmation-v1.json").read_text())
for noun in c006["nouns"]:
    entry = ((int(noun["sg_id"]),), (int(noun["pl_id"]),))
    assert ids_by_key.setdefault(noun["lexical_key"], entry) == entry
nouns = c020["exposed_noun_keys"]
scorable = [i for i, k in enumerate(nouns) if len(ids_by_key[k][0]) == 1 and len(ids_by_key[k][1]) == 1]
dw = torch.stack([W_U[ids_by_key[nouns[i]][0][0]] - W_U[ids_by_key[nouns[i]][1][0]] for i in scorable])  # [79, 512]
out["nouns"] = {"exposed": len(nouns), "scorable": len(scorable), "non_scorable": [nouns[i] for i in range(len(nouns)) if i not in scorable]}


def ln(x, w, b):
    x = x.double()
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    return (x - mu) / torch.sqrt(var + eps) * w + b


def sigma(x):
    x = x.double()
    return float(torch.sqrt(((x - x.mean()) ** 2).mean() + eps))


# ---------------------------------------------------------------------------------------------------------------
# Per frame: REF vs 020's locked state; landing; run-log digests; G2–G6 from the saved tensors; measured quantities.
log_digests = {}
for line in (CAL / "production_runs.jsonl").read_text().splitlines():
    e = json.loads(line)
    log_digests[e["key"]] = e["replacements"]
labels = {cue: (label, cid) for cue, cid, label in CUES}
logits_digests = state["calibration"]["measurements"]["logits_sha256"]

gate = defaultdict(lambda: {"max": 0.0, "at": None})
per_arm = defaultdict(lambda: defaultdict(float))
counts = defaultdict(int)


def tol(name, value, where, armtag):
    counts[name] += 1
    if value > gate[name]["max"] or gate[name]["at"] is None:
        gate[name] = {"max": float(value), "at": where}
    per_arm[name][armtag] = max(per_arm[name][armtag], float(value))


bit_fail = defaultdict(int)
ref_vs_020 = {"x3_pc": 0.0, "x3_pt": 0.0, "h6": 0.0, "x3_all": 0.0, "c_ref_pm": 0.0}
pairs, values, runlog_mismatch, recomputed_edit_ulps = {}, {}, [], []
readout = {}
logit_vals = {}
layer_norm = defaultdict(list)


def key_of(fid, tag, on_ref, cue):
    if on_ref:
        return f"{fid}|ref|{REF_ID}|{tag}"
    label, cid = labels[cue]
    return f"{fid}|{label}|{cid}|{tag}"


def sha_uniforms(tag, count):
    import hashlib
    outv, counter = [], 0
    while len(outv) < count:
        block = hashlib.sha256(f"{tag}|{counter}".encode("utf-8")).digest()
        for k in range(4):
            outv.append(((int.from_bytes(block[8 * k: 8 * k + 8], "big") >> 11) + 1) / 2.0 ** 53)
        counter += 1
    return outv[:count]


def iso_dir(fid, j):
    u = sha_uniforms(f"101|iso|{fid}|{j}", 512)
    g = []
    for i in range(256):
        r, ph = math.sqrt(-2.0 * math.log(u[2 * i])), 2.0 * math.pi * u[2 * i + 1]
        g += [r * math.cos(ph), r * math.sin(ph)]
    v = torch.tensor(g, dtype=torch.float64)
    ones = torch.full((512,), 1.0 / math.sqrt(512), dtype=torch.float64)
    for _ in range(2):
        for b in (ones, w_hat):
            v = v - torch.dot(v, b) * b
    return v / torch.linalg.vector_norm(v)


for fid in frame_ids:
    p_c, p_t = [int(v) for v in saved[f"{fid}/positions"]]
    L = locked[fid]
    assert (p_c, p_t) == (L["p_c"], L["p_t"])
    x3pc, x3pt, x4pc = saved[f"{fid}/x3_pc"], saved[f"{fid}/x3_pt"], saved[f"{fid}/x4_pc"]
    heads, mlp, T, attn, h6 = saved[f"{fid}/heads_pc"], saved[f"{fid}/mlp_pc"], saved[f"{fid}/T_pt"], saved[f"{fid}/attn_pt"], saved[f"{fid}/h6"]
    C = saved[f"{fid}/contrasts"].double()
    LI = saved[f"{fid}/logit_identity"].double()
    row = {t: i for i, t in enumerate(tags)}
    ref = row["REF"]
    locked_x3 = torch.tensor(L["x3_all"], dtype=torch.float64)
    ref_vs_020["x3_pc"] = max(ref_vs_020["x3_pc"], float((x3pc[ref].double() - locked_x3[p_c]).abs().max()))
    ref_vs_020["x3_pt"] = max(ref_vs_020["x3_pt"], float((x3pt[ref].double() - locked_x3[p_t]).abs().max()))
    ref_vs_020["h6"] = max(ref_vs_020["h6"], float((h6[ref].double() - torch.tensor(L["h6"], dtype=torch.float64)).abs().max()))
    ref_vs_020["x3_all"] = max(ref_vs_020["x3_all"], float((saved[f"{fid}/x3_all/REF"].double() - locked_x3).abs().max()))
    ref_vs_020["c_ref_pm"] = max(ref_vs_020["c_ref_pm"], float((saved[f"{fid}/c_ref_pm"].double() - torch.tensor(L["c_ref"], dtype=torch.float64)).abs().max()))
    # G6 against 020's locked (the re-capture equals it at drift 0.0 per the record)
    bit_fail["G6_c_ref_vs_020"] += int(not torch.equal(saved[f"{fid}/c_ref_pm"].double(), torch.tensor(L["c_ref"], dtype=torch.float64)))
    bit_fail["G6_state_vs_020"] += int(not (torch.equal(x3pc[ref].double(), locked_x3[p_c]) and torch.equal(x3pt[ref].double(), locked_x3[p_t])
                                            and torch.equal(h6[ref].double(), torch.tensor(L["h6"], dtype=torch.float64))))
    # pairs
    x_ref = x3pc[ref].double()
    fp = {}
    for cue, _, _ in CUES:
        x_cue = x3pc[row[f"PL@{cue}"]].double()
        dx = x_cue - x_ref
        dr = float(dx @ w_c)
        read = (dr / float(w_c @ w_c)) * w_c
        off = dx - read
        off_c = off - off.mean()
        off_norm = float(torch.linalg.vector_norm(off_c))
        s = abs(dr) / w_norm
        u_off = off_c / off_norm
        u32 = u_off.float().double()
        tol("G2_u_off32", max(abs(float(u32.mean())), abs(float(torch.linalg.vector_norm(u32)) - 1.0), abs(float(u32 @ w_hat))), f"{fid}|{cue}", f"u_off@{cue}")
        fp[cue] = {"dr": dr, "dx_norm": float(torch.linalg.vector_norm(dx)), "off_c_norm": off_norm, "s": s, "s_over_off_c": s / off_norm,
                   "overshoot": off_norm < s, "read": read, "u_off": u_off, "x_cue": x_cue}
        # the plain cue run's x3(p_t) and the cue/ref x3_all against each other (positions < p_c share the prefix)
        xa_ref, xa_cue = saved[f"{fid}/x3_all/REF"], saved[f"{fid}/x3_all/PL@{cue}"]
        bit_fail["prefix_positions_equal_across_prompts"] += int(not torch.equal(xa_ref[:p_c], xa_cue[:p_c]))
    pairs[fid] = {cue: {k: v for k, v in d.items() if k not in ("read", "u_off", "x_cue")} for cue, d in fp.items()}
    dirs = {j: iso_dir(fid, j) for j in range(1, 9)}
    # runs
    for i, (tag, arm, cue, on_ref) in enumerate(runs):
        kind = arm_kind(arm)
        key = key_of(fid, tag, on_ref, cue)
        where = key
        logged = log_digests[key]
        if kind in ("noop", "read", "off", "iso", "swap"):
            rep = saved[f"{fid}/replacement/{tag}"]
            if logged != {f"RESID_PRE.L3@{p_c}": float32_digest(rep)}:
                runlog_mismatch.append(key)
            bit_fail["G2_landing_x3pc_equals_replacement"] += int(not torch.equal(x3pc[i], rep))
            plain_row = ref if on_ref else row[f"PL@{cue}"]
            bit_fail["G4_x3_pt"] += int(not torch.equal(x3pt[i], x3pt[plain_row]))
            counts["G4_x3_pt"] += 1
            P = fp[cue]
            base = x_ref if on_ref else P["x_cue"]
            xp = x3pc[i].double()
            e = xp - base
            e_c = e - e.mean()
            e_norm = float(torch.linalg.vector_norm(e_c))
            # my own recomputation of the intended replacement (float64 → float32)
            if kind == "read":
                intended = base + READ_COEFF[arm] * P["read"]
                target_r, target_len, direction = float(base @ w_c) + READ_COEFF[arm] * P["dr"], abs(READ_COEFF[arm]) * P["s"], "parallel"
            elif kind == "off":
                c = -1.0 if arm.startswith("NEC") else 1.0
                intended = base + c * P["s"] * P["u_off"]
                target_r, target_len, direction = float(base @ w_c), P["s"], "orthogonal"
            elif kind == "iso":
                c = -1.0 if arm.startswith("NEC") else 1.0
                j = int(arm.rsplit("-", 1)[1])
                intended = base + c * P["s"] * dirs[j]
                target_r, target_len, direction = float(base @ w_c), P["s"], "orthogonal"
            elif kind == "swap":
                other = P["x_cue"] if on_ref else x_ref
                intended = other
                target_r, target_len, direction = float(other @ w_c), float(torch.linalg.vector_norm((other - base) - (other - base).mean())), None
            else:  # noop
                intended = P["x_cue"]
                target_r = target_len = direction = None
            diff = (intended.float() - rep).abs()
            ulp = (torch.nextafter(rep.abs(), torch.tensor(float("inf"))) - rep.abs())
            recomputed_edit_ulps.append(float((diff / ulp).max()))
            if target_r is not None:
                tol("G2_r", abs(float(xp @ w_c) - target_r), where, tag)
                tol("G2_length", abs(e_norm - target_len) / max(target_len, 1e-12), where, tag)
                if direction == "parallel":
                    tol("G2_direction", float(torch.linalg.vector_norm(e_c - float(e_c @ w_hat) * w_hat)) / max(e_norm, 1e-12), where, tag)
                elif direction == "orthogonal":
                    tol("G2_direction", abs(float(e_c @ w_hat)) / max(e_norm, 1e-12), where, tag)
                s_x = sigma(xp)
                layer_norm[tag].append({"sigma": s_x, "gamma3_e_over_sigma": float(torch.linalg.vector_norm(gamma3 * e)) / s_x,
                                        "gamma_ln2_e_over_sigma": float(torch.linalg.vector_norm(gamma_ln2 * e)) / s_x, "e_norm": float(torch.linalg.vector_norm(e)),
                                        "sigma_base": sigma(base)})
        elif kind == "head":
            rep = saved[f"{fid}/head_replacement/{tag}"]
            if logged != {f"L03.H04@{p_t}": float32_digest(rep)}:
                runlog_mismatch.append(key)
            bit_fail["G5_landing"] += int(not torch.equal(T[i], rep))
            counts["G5_landing"] += 1
            plain_row = ref if on_ref else row[f"PL@{cue}"]
            bit_fail["head_arm_x3pc_unedited"] += int(not torch.equal(x3pc[i], x3pc[plain_row]))
        else:
            if logged != {}:
                runlog_mismatch.append(key)
        # G3 pass-through on every run (gated on residual edits)
        pt = float((x4pc[i].double() - (x3pc[i].double() + heads[i].double().sum(0) + b_O3 + mlp[i].double())).abs().max())
        if kind in ("noop", "read", "off", "iso", "swap"):
            tol("G3_pass_through", pt, where, tag)
        else:
            tol("pass_through_other", pt, where, tag)
        # readout identity (mine): dw·(LN_f(h6_run) − LN_f(h6_locked)) against Δc_run = C[i] − C[REF]
        dln = ln(h6[i].double(), lnf_w, lnf_b) - ln(torch.tensor(L["h6"], dtype=torch.float64), lnf_w, lnf_b)
        err = float(((dw @ dln) - (C[i] - C[ref])).abs().max())
        g4 = kind in ("noop", "read", "off", "iso", "swap") or (kind == "plain" and tag != "REF")
        tol("G4_readout" if g4 else "readout_other", err, where, tag)
        tol("G4_logit_recorded" if g4 else "logit_other_recorded", float(LI[i]), where, tag)
        readout[key] = err
        logit_vals[key] = float(LI[i])
    # G3 NOOP vs PL@two, bit for bit
    nr, pr = row["NOOP@two"], row["PL@two"]
    same = all(torch.equal(saved[f"{fid}/{n}"][nr], saved[f"{fid}/{n}"][pr]) for n in ("x3_pc", "x3_pt", "x4_pc", "heads_pc", "mlp_pc", "T_pt", "attn_pt", "h6"))
    same = same and torch.equal(C[nr], C[pr]) and torch.equal(LI[nr], LI[pr]) and logits_digests[fid][nr] == logits_digests[fid][pr]
    bit_fail["G3_noop"] += int(not same)
    # the measured quantities (design §4, §7): Δc(arm) = mean over the 79 nouns of contrasts(arm) − contrasts(REF)
    dc = {tag: float((C[i] - C[ref]).mean()) for i, tag in enumerate(tags)}
    fv = {}
    for cue, _, _ in CUES:
        full = dc[f"PL@{cue}"]
        g = lambda a: dc[f"{a}@{cue}"]  # noqa: E731
        e = {"dc_full": full, "weak": full > -1.0, "F_rem": g("NEC-READ") / full, "F_rem_off": g("NEC-OFF") / full, "F_ins": g("SUF-READ") / full,
             "F_ins_off": g("SUF-OFF") / full, "A_amp": g("NEC-AMP") - full, "A_neg": g("SUF-NEG"), "F_half_rem": g("NEC-HALF") / full,
             "F_half_ins": g("SUF-HALF") / full, "F_swap_pl": g("PC-SWAP-PL") / full, "F_swap_ref": g("PC-SWAP-REF") / full,
             "dc": {a: g(a) for a in (["PL", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF", "PC-SWAP-PL", "PC-SWAP-REF"]
                                       + ([f"NEC-ISO-{j}" for j in range(1, 9)] + [f"SUF-ISO-{j}" for j in range(1, 9)] + ["HEAD-NEC", "HEAD-SUF", "HEAD-NEC-V", "HEAD-SUF-V", "NOOP"]
                                          if cue == PRIMARY else []))}}
        e["N"] = e["F_rem_off"] - e["F_rem"]
        e["S"] = e["F_ins"] - e["F_ins_off"]
        e["additivity"] = e["F_rem"] + e["F_ins"]
        if cue == PRIMARY:
            rem_iso = [g(f"NEC-ISO-{j}") / full for j in range(1, 9)]
            ins_iso = [g(f"SUF-ISO-{j}") / full for j in range(1, 9)]
            med = lambda v: 0.5 * (sorted(v)[3] + sorted(v)[4])  # noqa: E731  (even n = 8: mean of the 4th and 5th)
            e.update({"F_rem_iso": rem_iso, "F_ins_iso": ins_iso, "iso_rem_margin": med(rem_iso) - e["F_rem"], "iso_ins_margin": e["F_ins"] - med(ins_iso)})
            for v, sfx in (("", ""), ("_v", "-V")):
                e[f"F_head_rem{v}"] = g(f"HEAD-NEC{sfx}") / full
                e[f"F_head_ins{v}"] = g(f"HEAD-SUF{sfx}") / full
                e[f"head_removal_share{v}"] = (full - g(f"HEAD-NEC{sfx}")) / (full - g("NEC-READ")) if 1.0 - e["F_rem"] >= 0.05 else None
                e[f"head_install_share{v}"] = g(f"HEAD-SUF{sfx}") / g("SUF-READ") if e["F_ins"] >= 0.05 else None
                e[f"head_vs_sufread{v}"] = g(f"HEAD-SUF{sfx}") / g("SUF-READ")
        fv[cue] = e
    values[fid] = fv

out["ref_vs_020_locked_max_abs"] = ref_vs_020
out["bitwise_failures"] = dict(bit_fail)
out["counts"] = dict(counts)
out["runlog_digest_mismatches"] = runlog_mismatch
out["recomputed_edit_max_ulps"] = {"max": max(recomputed_edit_ulps), "n": len(recomputed_edit_ulps), "n_nonzero": sum(1 for u in recomputed_edit_ulps if u > 0)}
out["gates_mine"] = {k: dict(v) for k, v in gate.items()}
out["per_arm_mine"] = {k: dict(sorted(v.items())) for k, v in per_arm.items()}
# compare with the recorded tolerance gates and per-arm maxima
rec = state["calibration"]["gates"]
cmp = {}
for name in ("G2_r", "G2_length", "G2_direction", "G2_u_off32", "G3_pass_through", "G4_readout"):
    r = rec["tolerance"][name]
    cmp[name] = {"recorded_max": r["max"], "recorded_at": r["at"], "mine_max": gate[name]["max"], "mine_at": gate[name]["at"], "abs_diff": abs(r["max"] - gate[name]["max"]),
                 "tolerance": r["tolerance"], "margin_factor": r["tolerance"] / r["max"] if r["max"] else None}
cmp["G4_logit"] = {"recorded_max": rec["tolerance"]["G4_logit"]["max"], "recorded_at": rec["tolerance"]["G4_logit"]["at"], "max_of_saved_per_run_values": gate["G4_logit_recorded"]["max"],
                   "at": gate["G4_logit_recorded"]["at"], "other_runs_max_saved": gate["logit_other_recorded"]["max"], "other_recorded": rec["recorded"]["logit_other"]}
cmp["pass_through_other"] = {"recorded": rec["recorded"]["pass_through_other"], "mine": gate["pass_through_other"]}
per_arm_cmp = {}
for name in ("G2_r", "G2_length", "G2_direction", "G3_pass_through", "G4_readout"):
    recd = rec["per_arm"][name]
    per_arm_cmp[name] = max(abs(recd[t] - per_arm[name].get(t, -1.0)) for t in recd)
per_arm_cmp["G4_logit"] = max(abs(rec["per_arm"]["G4_logit"][t] - per_arm["G4_logit_recorded"].get(t, -1.0)) for t in rec["per_arm"]["G4_logit"])
per_arm_cmp["G2_u_off32"] = max(abs(rec["per_arm"]["G2_u_off32"][t] - per_arm["G2_u_off32"].get(t, -1.0)) for t in rec["per_arm"]["G2_u_off32"])
cmp["per_arm_max_abs_diff"] = per_arm_cmp
cmp["bitwise_recorded"] = rec["bitwise"]
out["gates_compare"] = cmp

# ---------------------------------------------------------------------------------------------------------------
# 7. τ needs the program's predictions (script d, frozen functions). Here: the measured side, the triggers, the shares.
two = [values[f]["two"] for f in frame_ids]
weak = {cue: [f for f in frame_ids if values[f][cue]["weak"]] for cue, _, _ in CUES}
overshoot = {cue: [f for f in frame_ids if pairs[f][cue]["overshoot"]] for cue, _, _ in CUES}
nonfinite_fracs = [f"{f}|{cue}|{k}" for f in frame_ids for cue, _, _ in CUES for k, v in values[f][cue].items()
                   if isinstance(v, float) and not math.isfinite(v)]
out["triggers_mine"] = {"weak_frames": {c: len(v) for c, v in weak.items()}, "weak_list_two": weak["two"], "overshoot_frames": {c: len(v) for c, v in overshoot.items()},
                        "overshoot_list_two": overshoot["two"], "nonfinite_measured_fractions": nonfinite_fracs,
                        "recorded": state["calibration"].get("stopped"), "candidate_triggers": cand["triggers"]}
n = 36
shares = {"N_positive": sum(1 for e in two if not e["weak"] and e["N"] > 0) / n, "S_positive": sum(1 for e in two if not e["weak"] and e["S"] > 0) / n,
          "A_amp_negative": sum(1 for e in two if not e["weak"] and e["A_amp"] < 0) / n, "A_neg_positive": sum(1 for e in two if not e["weak"] and e["A_neg"] > 0) / n}
out["shares_mine"] = shares
out["shares_recorded"] = cand["shares"]
# distributions recorded vs mine
dist_cmp = {}
for name in ("N", "S", "A_amp", "A_neg"):
    mine_sorted = sorted(e[name] for e in two)
    dist_cmp[name] = max(abs(a - b) for a, b in zip(mine_sorted, cand["distributions"][name])) if len(mine_sorted) == len(cand["distributions"][name]) else "length differs"
out["distributions_max_abs_diff"] = dist_cmp
# reports: pairs and dc_full against the candidate's
rep_cmp = 0.0
for f in frame_ids:
    for cue, _, _ in CUES:
        for k in ("dr", "dx_norm", "off_c_norm", "s", "s_over_off_c"):
            rep_cmp = max(rep_cmp, abs(cand["reports"][f]["pairs"][cue][k] - pairs[f][cue][k]))
        rep_cmp = max(rep_cmp, abs(cand["reports"][f]["dc_full"][cue] - values[f][cue]["dc_full"]))
        assert cand["reports"][f]["pairs"][cue]["overshoot"] == pairs[f][cue]["overshoot"]
out["reports_max_abs_diff"] = rep_cmp
out["min_s_mine"] = {cue: min((pairs[f][cue]["s"], f) for f in frame_ids) for cue, _, _ in CUES}
out["min_s_recorded"] = cand["min_s"]
# ceilings recorded: per run tag mean ceiling (only checked for presence/coverage here)
ceil = state["calibration"]["ceilings"]
out["ceilings_coverage"] = {"frames": len(ceil), "tags_per_frame": sorted({len(v) for v in ceil.values()})}

# ---------------------------------------------------------------------------------------------------------------
# 10. Descriptive ranges (real swaps) and 12. head and isotropic descriptives (no label force).
def rng(vals):
    vals = [v for v in vals if v is not None]
    s = sorted(vals)
    return {"min": s[0], "median": 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2]) if len(s) % 2 == 0 else s[len(s) // 2], "max": s[-1], "n": len(s)}


desc = {}
for cue, _, _ in CUES:
    desc[cue] = {k: rng([pairs[f][cue][k] for f in frame_ids]) for k in ("dr", "dx_norm", "off_c_norm", "s", "s_over_off_c")}
    desc[cue]["dc_full"] = rng([values[f][cue]["dc_full"] for f in frame_ids])
    for k in ("F_rem", "F_rem_off", "F_ins", "F_ins_off", "N", "S", "A_amp", "A_neg", "F_half_rem", "F_half_ins", "F_swap_pl", "F_swap_ref", "additivity"):
        desc[cue][k] = rng([values[f][cue][k] for f in frame_ids])
    desc[cue]["counts"] = {"N>0": sum(values[f][cue]["N"] > 0 for f in frame_ids), "S>0": sum(values[f][cue]["S"] > 0 for f in frame_ids),
                           "A_amp<0": sum(values[f][cue]["A_amp"] < 0 for f in frame_ids), "A_neg>0": sum(values[f][cue]["A_neg"] > 0 for f in frame_ids),
                           "weak": sum(values[f][cue]["weak"] for f in frame_ids)}
out["descriptive"] = desc
head = {}
for k in ("F_head_rem", "F_head_rem_v", "F_head_ins", "F_head_ins_v", "head_removal_share", "head_removal_share_v", "head_install_share", "head_install_share_v",
          "head_vs_sufread", "head_vs_sufread_v", "iso_rem_margin", "iso_ins_margin"):
    head[k] = rng([values[f]["two"][k] for f in frame_ids])
head["guard_excluded"] = {"removal": sum(values[f]["two"]["head_removal_share"] is None for f in frame_ids),
                          "install": sum(values[f]["two"]["head_install_share"] is None for f in frame_ids)}
head["counts"] = {"F_head_rem<1": sum(values[f]["two"]["F_head_rem"] < 1 for f in frame_ids), "F_head_rem_v<1": sum(values[f]["two"]["F_head_rem_v"] < 1 for f in frame_ids),
                  "F_head_ins>0": sum(values[f]["two"]["F_head_ins"] > 0 for f in frame_ids), "F_head_ins_v>0": sum(values[f]["two"]["F_head_ins_v"] > 0 for f in frame_ids),
                  "F_rem<median_iso": sum(values[f]["two"]["iso_rem_margin"] > 0 for f in frame_ids),
                  "F_ins>median_iso": sum(values[f]["two"]["iso_ins_margin"] > 0 for f in frame_ids)}
out["head_iso"] = head
ln_summary = {}
for tag, recs in sorted(layer_norm.items()):
    ln_summary[tag] = {k: {"median": sorted(r[k] for r in recs)[len(recs) // 2], "max": max(r[k] for r in recs)} for k in recs[0]}
out["layer_norm"] = {k: v for k, v in ln_summary.items() if k.endswith("@two") and not k.startswith(("NEC-ISO", "SUF-ISO"))}
out["write_attempts"] = WRITE_ATTEMPTS
dump("b_state_measurements.json", out)
dump("b_values.json", {"values": {f: {c: {k: v for k, v in e.items()} for c, e in values[f].items()} for f in frame_ids}, "pairs": pairs, "frame_ids": frame_ids,
                       "readout_per_run": readout, "logit_per_run": logit_vals})
print(json.dumps({k: v for k, v in out.items() if k not in ("descriptive", "layer_norm", "per_arm_mine", "head_iso")}, indent=1, default=str)[:20000])
