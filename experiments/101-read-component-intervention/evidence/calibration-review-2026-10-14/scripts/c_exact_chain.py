"""Checks 5, 6 (G4 Level-1, G5, the head deltas, the ceilings) and 11's diagnostics — independent: my own float64
implementation of GPT-NeoX layers 3–5, LN_final and the noun read, from the pinned safetensors and the saved residuals.

No neural_decompiler import, no model object, no token input, no forward pass of the model: every computation starts
from saved float32 residuals (the calibration's captures and 020's locked reference states) and the weights.
"""
import json
import math
import sys
from collections import defaultdict

import torch

from common import (CUES, OUTPUTS_101, PRIMARY, READ_COEFF, REF_ID, RESULTS_020, ROOT, SNAPSHOT, WRITE_ATTEMPTS, arm_kind, dump, frame_runs)

assert not [m for m in sys.modules if m.startswith("neural_decompiler")]
from safetensors import safe_open  # noqa: E402

torch.set_grad_enabled(False)
st = safe_open(str(SNAPSHOT / "model.safetensors"), "pt")
EPS = 1e-5
NH, DH, RD = 8, 64, 16


def T(name, dtype=torch.float64):
    return st.get_tensor(name).to(dtype)


class Layer:
    def __init__(self, l, dtype=torch.float64):
        p = f"gpt_neox.layers.{l}."
        self.dtype = dtype
        self.ln1_w, self.ln1_b = T(p + "input_layernorm.weight", dtype), T(p + "input_layernorm.bias", dtype)
        self.ln2_w, self.ln2_b = T(p + "post_attention_layernorm.weight", dtype), T(p + "post_attention_layernorm.bias", dtype)
        self.qkv_w, self.qkv_b = T(p + "attention.query_key_value.weight", dtype), T(p + "attention.query_key_value.bias", dtype)
        self.o_w, self.o_b = T(p + "attention.dense.weight", dtype), T(p + "attention.dense.bias", dtype)
        self.in_w, self.in_b = T(p + "mlp.dense_h_to_4h.weight", dtype), T(p + "mlp.dense_h_to_4h.bias", dtype)
        self.out_w, self.out_b = T(p + "mlp.dense_4h_to_h.weight", dtype), T(p + "mlp.dense_4h_to_h.bias", dtype)
        inv = 1.0 / (10000.0 ** (torch.arange(0, RD, 2, dtype=torch.float64) / RD))
        self.inv = inv.to(dtype)

    def ln(self, x, w, b):
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + EPS) * w + b

    def qkv(self, X):  # X [n, 512] → q, k, v [heads, n, 64]
        h = self.ln(X, self.ln1_w, self.ln1_b) @ self.qkv_w.T + self.qkv_b
        h = h.view(X.shape[0], NH, 3 * DH).transpose(0, 1)
        return h[..., :DH], h[..., DH:2 * DH], h[..., 2 * DH:]

    def rotate(self, t, positions):
        ang = positions.to(self.dtype)[:, None] * self.inv[None, :]  # [n, 8]
        cos, sin = torch.cos(ang), torch.sin(ang)
        x1, x2, rest = t[..., :RD // 2], t[..., RD // 2:RD], t[..., RD:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], dim=-1)

    def attn_at(self, X, q):
        """The attention output (with b_O) at query q from residuals X[0..q]; also the rows and the raw scores."""
        Xq = X[: q + 1]
        Q, K, V = self.qkv(Xq)
        pos = torch.arange(q + 1)
        Q, K = self.rotate(Q, pos), self.rotate(K, pos)
        scores = torch.einsum("he,hke->hk", Q[:, q, :], K) / math.sqrt(DH)
        A = torch.softmax(scores, dim=-1)
        z = torch.einsum("hk,hke->he", A, V)  # [heads, 64]
        per_head = torch.stack([z[h] @ self.o_w[:, h * DH:(h + 1) * DH].T for h in range(NH)])  # [heads, 512]
        return per_head.sum(0) + self.o_b, A, scores, per_head

    def mlp(self, x):
        h = self.ln(x, self.ln2_w, self.ln2_b) @ self.in_w.T + self.in_b
        h = 0.5 * h * (1.0 + torch.erf(h / math.sqrt(2.0)))
        return h @ self.out_w.T + self.out_b


L64 = {l: Layer(l) for l in (3, 4, 5)}
L32 = {l: Layer(l, torch.float32) for l in (3, 4, 5)}
lnf_w, lnf_b = T("gpt_neox.final_layer_norm.weight"), T("gpt_neox.final_layer_norm.bias")
W_U = T("embed_out.weight")


def ln_f(x):
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    return (x - mu) / torch.sqrt(var + EPS) * lnf_w + lnf_b


# nouns (as in script b)
manifest = json.loads((ROOT / "screening/behavior-candidates/manifest-v1.json").read_text())
ids = {}
for case in manifest["cases"]:
    if case["candidate_id"] == "regular-plural":
        c = case["conditions"]["s_a"]
        ids[case["lexical_key"]] = (tuple(c["a_token_ids"]), tuple(c["b_token_ids"]))
for noun in json.loads((ROOT / "experiments/006-low-rank-cue-decompilation/confirmation-v1.json").read_text())["nouns"]:
    ids[noun["lexical_key"]] = ((int(noun["sg_id"]),), (int(noun["pl_id"]),))
c020 = json.loads((ROOT / "experiments/020-readout-decompilation/confirmation-v1.json").read_text())
noun_keys = c020["exposed_noun_keys"]
scor = [k for k in noun_keys if len(ids[k][0]) == 1 and len(ids[k][1]) == 1]
dw = torch.stack([W_U[ids[k][0][0]] - W_U[ids[k][1][0]] for k in scor])

lock011 = json.loads((ROOT / "experiments/011-encoding-read-prospective/preregistration-lock.json").read_text())
d_T = torch.tensor(lock011["axes_vectors"]["T"], dtype=torch.float64)
rw = torch.tensor(lock011["read_weight"], dtype=torch.float64)
w_c = rw - rw.mean()
w_norm = float(torch.linalg.vector_norm(w_c))

saved = torch.load(OUTPUTS_101 / "calibration-measurements.pt", weights_only=True)
state = json.loads((OUTPUTS_101 / "results.json").read_text())
locked = json.loads(RESULTS_020.read_text())["exploration"]["locked_states"]
frame_ids = [f for f in c020["exposed_frame_ids"] if f.startswith("coordinated-adjective")]
runs = frame_runs()
tags = [r[0] for r in runs]
labels = {cue: (label, cid) for cue, cid, label in CUES}


def key_of(fid, tag, on_ref, cue):
    if on_ref:
        return f"{fid}|ref|{REF_ID}|{tag}"
    label, cid = labels[cue]
    return f"{fid}|{label}|{cid}|{tag}"


def chain(LS, X3, X4, X5, h6, dx3, p_c, p_t, *, mode="exact"):
    """mode exact: 020's Level-1 chain (every head and MLP recomputed); mode frozen: 020's readout (block 4 attention
    frozen, block 5 frozen reference rows times value changes). Returns dh6 and per-block parts."""
    X3e = X3.clone()
    for p, d in dx3.items():
        X3e[p] = X3e[p] + d
    d3 = {}
    for p in (p_c, p_t):
        a_e = LS[3].attn_at(X3e, p)[0]
        a_r = LS[3].attn_at(X3, p)[0]
        d3[p] = dx3[p] + (a_e - a_r) + (LS[3].mlp(X3e[p]) - LS[3].mlp(X3[p]))
    X4e = X4.clone()
    for p in (p_c, p_t):
        X4e[p] = X4e[p] + d3[p]
    d4 = {}
    for p in (p_c, p_t):
        att = (LS[4].attn_at(X4e, p)[0] - LS[4].attn_at(X4, p)[0]) if mode == "exact" else torch.zeros_like(d3[p])
        d4[p] = d3[p] + att + (LS[4].mlp(X4e[p]) - LS[4].mlp(X4[p]))
    X5e = X5.clone()
    for p in (p_c, p_t):
        X5e[p] = X5e[p] + d4[p]
    if mode == "exact":
        a5 = LS[5].attn_at(X5e, p_t)[0] - LS[5].attn_at(X5, p_t)[0]
    else:
        rows5 = frozen_rows5
        _, _, Ve = LS[5].qkv(X5e[: p_t + 1])
        _, _, Vr = LS[5].qkv(X5[: p_t + 1])
        dz = torch.einsum("hk,hke->he", rows5, Ve - Vr)
        a5 = sum(dz[h] @ LS[5].o_w[:, h * DH:(h + 1) * DH].T for h in range(NH))
    m5 = LS[5].mlp(X5e[p_t]) - LS[5].mlp(X5[p_t])
    return d4[p_t] + a5 + m5


out = {"level1_mine": {}, "ceiling_mine": {}, "g5": {}, "head_delta": {}, "diag": {}}
per_run = {}
frame_diag = {}
for fid in frame_ids:
    L = locked[fid]
    p_c, p_t = L["p_c"], L["p_t"]
    X3 = torch.tensor(L["x3_all"], dtype=torch.float64)
    X4 = torch.tensor(L["x4_all"], dtype=torch.float64)
    X5 = torch.tensor(L["x5_all"], dtype=torch.float64)
    h6 = torch.tensor(L["h6"], dtype=torch.float64)
    frozen_rows5 = torch.tensor(L["rows5"], dtype=torch.float64)[:, : p_t + 1]
    x3pc, x3pt, C = saved[f"{fid}/x3_pc"].double(), saved[f"{fid}/x3_pt"].double(), saved[f"{fid}/contrasts"].double()
    row = {t: i for i, t in enumerate(tags)}
    ref = row["REF"]
    base_ln = ln_f(h6)
    # the reference state's attention-score scale per layer (020's float32 floor diagnosis)
    sc = {}
    for l, X in ((3, X3), (4, X4), (5, X5)):
        _, A, scores, _ = L64[l].attn_at(X, p_t)
        sc[l] = {"max_abs_score": float(scores.abs().max()), "row_spread_max": float((scores.max(-1).values - scores.min(-1).values).max()),
                 "float32_ulp_at_max": float(torch.finfo(torch.float32).eps * scores.abs().max() / 2)}
    frame_diag[fid] = {"p_c": p_c, "p_t": p_t, "scores": sc, "h6_sigma": float(torch.sqrt(((h6 - h6.mean()) ** 2).mean() + EPS))}
    for i, (tag, arm, cue, on_ref) in enumerate(runs):
        kind = arm_kind(arm)
        key = key_of(fid, tag, on_ref, cue)
        g4 = kind in ("noop", "read", "off", "iso", "swap") or (kind == "plain" and tag != "REF")
        if kind == "head" or tag == "REF":
            continue
        dx3 = {p_c: x3pc[i] - X3[p_c], p_t: x3pt[i] - X3[p_t]}
        dc_run = C[i] - C[ref]
        dh6 = chain(L64, X3, X4, X5, h6, dx3, p_c, p_t, mode="exact")
        lvl = dw @ (ln_f(h6 + dh6) - base_ln)
        dh6_f = chain(L64, X3, X4, X5, h6, dx3, p_c, p_t, mode="frozen")
        ceil = dw @ (ln_f(h6 + dh6_f) - base_ln)
        # the same exact chain in float32 (my op order): the scale of float32 rounding in layers 3–5 for this run
        dh6_32 = chain(L32, X3.float(), X4.float(), X5.float(), h6.float(), {p: d.float() for p, d in dx3.items()}, p_c, p_t, mode="exact").double()
        lvl32 = dw @ (ln_f(h6 + dh6_32) - base_ln)
        meas_dh6 = saved[f"{fid}/h6"][i].double() - h6
        per_run[key] = {"frame": fid, "tag": tag, "arm": arm, "cue": cue, "kind": kind, "g4": g4, "p_c": p_c,
                        "level1": float((lvl - dc_run).abs().max()), "level1_argmax_noun": scor[int((lvl - dc_run).abs().argmax())],
                        "ceiling_error": float((ceil - dc_run).abs().max()), "ceiling_mean": float(ceil.mean()),
                        "f32_vs_f64_chain": float((lvl32 - lvl).abs().max()),
                        "E1": float((dh6 - meas_dh6).abs().max()) / max(float(meas_dh6.abs().max()), 1e-12),
                        "dh6_inf": float(meas_dh6.abs().max()), "dc_mean": float(dc_run.mean()), "dc_absmax": float(dc_run.abs().max()),
                        "dx3_pc_norm": float(torch.linalg.vector_norm(dx3[p_c])), "dx3_pt_norm": float(torch.linalg.vector_norm(dx3[p_t]))}
    # G5 and the head deltas, independently
    W_V4 = L64[3].qkv_w[4 * 192 + 128: 4 * 192 + 192, :].T
    W_O4 = L64[3].o_w[:, 4 * DH:5 * DH].T
    for tag, arm, cue, on_ref in runs:
        if arm_kind(arm) != "head":
            continue
        src_tag = "REF" if on_ref else f"PL@{cue}"
        src = row[src_tag]
        Xs = saved[f"{fid}/x3_all/{src_tag}"].double()  # the plain run's RESID_PRE.L3 at 0..p_t
        _, A, _, per_head = L64[3].attn_at(Xs, p_t)
        captured_row = saved[f"{fid}/attn_pt"][src][4, : p_t + 1].double()
        captured_T = saved[f"{fid}/T_pt"][src].double()
        g5_row = float((A[4] - captured_row).abs().max())
        g5_out = abs(float(per_head[4] @ d_T) - float(captured_T @ d_T))
        # the head arm's edit: e = ∓read of the pair (plain runs' x3_pc)
        x_ref, x_cue = x3pc[ref], x3pc[row[f"PL@{cue}"]]
        dr = float((x_cue - x_ref) @ w_c)
        read = (dr / float(w_c @ w_c)) * w_c
        e = (-1.0 if arm.startswith("HEAD-NEC") else 1.0) * read
        if arm.endswith("-V"):
            ln3 = L64[3]
            n_new = ln3.ln(Xs[p_c] + e, ln3.ln1_w, ln3.ln1_b)
            n_old = ln3.ln(Xs[p_c], ln3.ln1_w, ln3.ln1_b)
            delta = float(saved[f"{fid}/attn_pt"][src][4, p_c]) * ((n_new - n_old) @ W_V4 @ W_O4)
        else:
            Xe = Xs.clone()
            Xe[p_c] = Xe[p_c] + e
            delta = L64[3].attn_at(Xe, p_t)[3][4] - per_head[4]
        saved_delta = saved[f"{fid}/delta/{tag}"].double()
        rep = (saved[f"{fid}/T_pt"][src].double().reshape(-1) + saved_delta).float()
        key = key_of(fid, tag, on_ref, cue)
        out["g5"][key] = {"row": g5_row, "output": g5_out}
        out["head_delta"][key] = {"delta_max_abs_diff": float((delta - saved_delta).abs().max()), "delta_norm": float(torch.linalg.vector_norm(saved_delta)),
                                  "replacement_equal": bool(torch.equal(rep, saved[f"{fid}/head_replacement/{tag}"])),
                                  "delta_dot_dT": float(saved_delta @ d_T)}

# compare with the recorded per-arm maxima
rec = state["calibration"]["gates"]
g4 = {k: v for k, v in per_run.items() if v["g4"]}
by_arm = defaultdict(float)
for k, v in g4.items():
    by_arm[v["tag"]] = max(by_arm[v["tag"]], v["level1"])
out["level1_mine"] = {"n_g4_runs": len(g4), "n_residual_edit_runs": sum(1 for v in g4.values() if v["kind"] != "plain"),
                      "max": max(v["level1"] for v in g4.values()), "argmax": max(g4, key=lambda k: g4[k]["level1"]),
                      "recorded_max": rec["tolerance"]["G4_level1"]["max"], "recorded_at": rec["tolerance"]["G4_level1"]["at"],
                      "per_arm_max_abs_diff_vs_recorded": max(abs(rec["per_arm"]["G4_level1"][t] - by_arm[t]) for t in rec["per_arm"]["G4_level1"]),
                      "per_arm_max_rel_diff_vs_recorded": max(abs(rec["per_arm"]["G4_level1"][t] - by_arm[t]) / rec["per_arm"]["G4_level1"][t] for t in rec["per_arm"]["G4_level1"])}
e1_max_key = max(g4, key=lambda k: g4[k]["E1"])
out["E1_mine"] = {"max": g4[e1_max_key]["E1"], "at": e1_max_key, "recorded": rec["recorded"]["E1"]}
ce_key = max(g4, key=lambda k: g4[k]["ceiling_error"])
out["ceiling_error_mine"] = {"max": g4[ce_key]["ceiling_error"], "at": ce_key, "recorded": rec["recorded"]["ceiling_error"]}
ceil_rec = state["calibration"]["ceilings"]
out["ceiling_means_max_abs_diff"] = max(abs(ceil_rec[v["frame"]][v["tag"]] - v["ceiling_mean"]) for v in g4.values())
g5_row_key = max(out["g5"], key=lambda k: out["g5"][k]["row"])
g5_out_key = max(out["g5"], key=lambda k: out["g5"][k]["output"])
out["g5_mine"] = {"row_max": out["g5"][g5_row_key]["row"], "row_at": g5_row_key, "output_max": out["g5"][g5_out_key]["output"], "output_at": g5_out_key,
                  "recorded_row": rec["tolerance"]["G5_row"], "recorded_output": rec["tolerance"]["G5_output"]}
out["head_delta_summary"] = {"max_abs_diff": max(v["delta_max_abs_diff"] for v in out["head_delta"].values()),
                             "replacements_reconstructed": sum(v["replacement_equal"] for v in out["head_delta"].values()), "n": len(out["head_delta"])}
out["f32_chain_noise"] = {"max": max(v["f32_vs_f64_chain"] for v in g4.values()), "median": sorted(v["f32_vs_f64_chain"] for v in g4.values())[len(g4) // 2]}
out["write_attempts"] = WRITE_ATTEMPTS
del out["g5"], out["head_delta"]
dump("c_exact_chain.json", out)
dump("c_per_run.json", {"per_run": per_run, "frame_diag": frame_diag})
print(json.dumps(out, indent=1, default=str))
