"""Item 11's diagnosis: split each G4 run's Level-1 error vector exactly into (i) the Δh6 part — the exact chain's final
residual against the measured one, through LN_final and the noun read in float64 — and (ii) the readout part — the
measured h6 → contrast path (the readout-identity residual). Also: block 3's exact output at p_c against the captured
RESID_PRE.L4@p_c of the run (the only intermediate capture), and each noun's ‖Δw‖. No model, no forward pass."""
# --- my float64 layers, copied verbatim from c_exact_chain.py (definitions only) ---

import json
import math
import sys
from collections import defaultdict

import numpy as np
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



# --- the decomposition ---
from common import OUT as _OUT
scor_list = scor
dpr = json.loads((_OUT / "d_per_run_frozen.json").read_text())
res = {}
for fid in frame_ids:
    L = locked[fid]
    p_c, p_t = L["p_c"], L["p_t"]
    X3 = torch.tensor(L["x3_all"], dtype=torch.float64)
    X4 = torch.tensor(L["x4_all"], dtype=torch.float64)
    X5 = torch.tensor(L["x5_all"], dtype=torch.float64)
    h6 = torch.tensor(L["h6"], dtype=torch.float64)
    x3pc, x3pt, x4pc, C, H6 = (saved[f"{fid}/{n}"].double() for n in ("x3_pc", "x3_pt", "x4_pc", "contrasts", "h6"))
    ref = tags.index("REF")
    for i, (tag, arm, cue, on_ref) in enumerate(runs):
        kind = arm_kind(arm)
        if kind == "head" or tag == "REF":
            continue
        key = key_of(fid, tag, on_ref, cue)
        dx3 = {p_c: x3pc[i] - X3[p_c], p_t: x3pt[i] - X3[p_t]}
        dh6 = chain(L64, X3, X4, X5, h6, dx3, p_c, p_t, mode="exact")
        dc = C[i] - C[ref]
        part_h6 = dw @ (ln_f(h6 + dh6) - ln_f(H6[i]))
        part_ro = dw @ (ln_f(H6[i]) - ln_f(h6)) - dc
        total = part_h6 + part_ro
        j = int(total.abs().argmax())
        # block 3 at p_c: the exact output (x3' + attn + mlp, with b_O) against the captured RESID_PRE.L4@p_c of the run
        X3e = X3.clone()
        X3e[p_c] = x3pc[i]
        X3e[p_t] = x3pt[i]
        x4_exact_pc = X3e[p_c] + L64[3].attn_at(X3e, p_c)[0] + L64[3].mlp(X3e[p_c])
        res[key] = {"level1": float(total.abs().max()), "frozen_level1": dpr[key]["level1"], "noun": scor[j], "part_h6_at_max": float(part_h6[j]),
                    "part_readout_at_max": float(part_ro[j]), "part_h6_inf": float(part_h6.abs().max()), "part_readout_inf": float(part_ro.abs().max()),
                    "dh6_err_inf": float((h6 + dh6 - H6[i]).abs().max()), "block3_pc_err_inf": float((x4_exact_pc - x4pc[i]).abs().max()),
                    "dw_norm_at_max": float(torch.linalg.vector_norm(dw[j]))}
keys = list(res)
lv = np.array([res[k]["level1"] for k in keys])
top = sorted(keys, key=lambda k: -res[k]["level1"])[:10]
dwn = torch.linalg.vector_norm(dw, dim=1)
out = {"max_abs_diff_vs_frozen": max(abs(res[k]["level1"] - res[k]["frozen_level1"]) for k in keys),
       "top10": {k: res[k] for k in top},
       "median_part_h6_inf": float(np.median([res[k]["part_h6_inf"] for k in keys])),
       "median_part_readout_inf": float(np.median([res[k]["part_readout_inf"] for k in keys])),
       "max_part_h6_inf": max((res[k]["part_h6_inf"], k) for k in keys), "max_part_readout_inf": max((res[k]["part_readout_inf"], k) for k in keys),
       "max_block3_pc_err": max((res[k]["block3_pc_err_inf"], k) for k in keys), "median_block3_pc_err": float(np.median([res[k]["block3_pc_err_inf"] for k in keys])),
       "max_dh6_err": max((res[k]["dh6_err_inf"], k) for k in keys), "median_dh6_err": float(np.median([res[k]["dh6_err_inf"] for k in keys])),
       "dw_norm": {"median": float(dwn.median()), "max": float(dwn.max()), "argmax": scor[int(dwn.argmax())],
                   "army": float(dwn[scor.index("army")]), "cross": float(dwn[scor.index("cross")]), "rank_army": int((dwn > dwn[scor.index("army")]).sum()) + 1,
                   "rank_cross": int((dwn > dwn[scor.index("cross")]).sum()) + 1},
       "argmax_noun_counts_top50": dict(sorted(defaultdict(int, {n: sum(1 for k in sorted(keys, key=lambda k: -res[k]["level1"])[:50] if res[k]["noun"] == n) for n in set(res[k]["noun"] for k in keys)}).items(), key=lambda kv: -kv[1])[:8]),
       "frame_018_3_runs_above_0.0045": sorted(((res[k]["level1"], k) for k in keys if k.startswith("coordinated-adjective-018-3|") and res[k]["level1"] > 0.0045), reverse=True),
       "write_attempts": WRITE_ATTEMPTS}
dump("f_level1_decomposition.json", out)
print(json.dumps(out, indent=1, default=str))
