"""Review item 2, independently: G1's geometry, G2 (every replacement rebuilt from this review's own formulas and
compared bit for bit with the saved replacement; the realized geometry), G3 (NOOP bit for bit; the pass-through with
b_O read from the checkpoint), G4's x3(p_t) bit for bit and the readout identity with this review's own final
LayerNorm and unembedding, a contrast-level logit consistency check of every run, G5 (layer-3 head-4 attention
recomputed in float64 with this review's own rotary attention; the head deltas recomputed), G6 (the stage-2 REF runs
against the committed stage-1 states, read with this review's own reader), the overshoot ratio s/‖off_c‖ (design §2,
the lock review's N2), the LayerNorm records and the manipulation check.

Weights only, read from the pinned checkpoint's safetensors file (no model object, no forward pass). neural_decompiler
is imported only for data (the frozen inputs: the pool's noun ids and the 011 lock) and for the SHA-seeded Gaussians of
the isotropic directions, whose digests are checked against the lock."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rguard  # noqa: E402  (first)

rguard.self_test()
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from neural_decompiler import cue_rotation as cr  # noqa: E402  (sha_gaussians only)
from neural_decompiler import upstream_localization as ul  # noqa: E402  (frozen inputs: data only)
from neural_decompiler.models import PYTHIA_70M  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = rguard.ROOT
EXP = ROOT / "experiments/101-read-component-intervention"
OUT = rguard.OUT
out = {"checks": {}, "facts": {}}
tokenizer = AutoTokenizer.from_pretrained(PYTHIA_70M.model_id, revision=PYTHIA_70M.revision)
rguard.install_forward_guards(tokenizer)  # no model object exists here; installed anyway (and recorded)


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"), flush=True)


def tensor_digest(t):
    a = t.detach().cpu().contiguous()
    kind = "<i8" if a.dtype in (torch.int64, torch.int32) else "<f8"
    return hashlib.sha256(json.dumps(list(a.shape)).encode("ascii") + b"|" + kind.encode("ascii") + b"|" + a.numpy().astype(kind, copy=False).tobytes()).hexdigest()


# --- inputs ------------------------------------------------------------------------------------------------------------
lock = json.loads((EXP / "preregistration-lock.json").read_text(encoding="utf-8"))
conf = json.loads((EXP / "confirmation-v1.json").read_text(encoding="utf-8"))
state = json.loads((ROOT / "outputs/experiment-101/results.json").read_text(encoding="utf-8"))
tensors = torch.load(ROOT / "outputs/experiment-101/stage2-measurements.pt")
run_order = lock["run_order"]
row_of = {tag: i for i, tag in enumerate(run_order)}
frames = [f["frame_id"] for f in conf["frames"]]
inputs = ul.load_frozen_inputs(ROOT)
lock_011 = inputs.lock_011
check("lock_011_content_digest", lock_011["content_sha256"] == "769bfeacd7c49fc18bed2ff3c5cf8d5ea2be4231e9f8e9c493819b5d7ba69d0b")
nouns = [n for n in inputs.pool.nouns if n.single_token]
check("noun_keys_equal_lock", [n.lexical_key for n in nouns] == lock["noun_keys"] and len(nouns) == 79)
sg_ids = [int(n.sg_ids[0]) for n in nouns]
pl_ids = [int(n.pl_ids[0]) for n in nouns]
all_nouns = list(inputs.pool.nouns)

SNAP = Path("/Users/gaborhorvath-ulenius/.cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c")
h = hashlib.sha256()
with open(SNAP / "model.safetensors", "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
check("checkpoint_sha256_equals_production_record", h.hexdigest() == "3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd")
W = {}
with safe_open(str(SNAP / "model.safetensors"), "pt") as f:
    for key in ("gpt_neox.layers.3.input_layernorm.weight", "gpt_neox.layers.3.input_layernorm.bias", "gpt_neox.layers.3.attention.query_key_value.weight",
                "gpt_neox.layers.3.attention.query_key_value.bias", "gpt_neox.layers.3.attention.dense.weight", "gpt_neox.layers.3.attention.dense.bias",
                "gpt_neox.layers.3.post_attention_layernorm.weight", "gpt_neox.final_layer_norm.weight", "gpt_neox.final_layer_norm.bias", "embed_out.weight"):
        W[key] = f.get_tensor(key).to(torch.float64)
EPS = 1e-5
ln1_w, ln1_b = W["gpt_neox.layers.3.input_layernorm.weight"], W["gpt_neox.layers.3.input_layernorm.bias"]
qkv_w, qkv_b = W["gpt_neox.layers.3.attention.query_key_value.weight"], W["gpt_neox.layers.3.attention.query_key_value.bias"]
dense_w, b_O = W["gpt_neox.layers.3.attention.dense.weight"], W["gpt_neox.layers.3.attention.dense.bias"]
gamma_ln2 = W["gpt_neox.layers.3.post_attention_layernorm.weight"]
lnf_w, lnf_b = W["gpt_neox.final_layer_norm.weight"], W["gpt_neox.final_layer_norm.bias"]
W_U = W["embed_out.weight"]  # [vocab, d]
HEAD, DH = 4, 64
Wq = qkv_w[HEAD * 192: HEAD * 192 + 64].T  # [d, 64]
Wk = qkv_w[HEAD * 192 + 64: HEAD * 192 + 128].T
Wv = qkv_w[HEAD * 192 + 128: HEAD * 192 + 192].T
bq, bk, bv = qkv_b[HEAD * 192: HEAD * 192 + 64], qkv_b[HEAD * 192 + 64: HEAD * 192 + 128], qkv_b[HEAD * 192 + 128: HEAD * 192 + 192]
Wo = dense_w[:, HEAD * DH:(HEAD + 1) * DH].T  # [64, d]
dW = torch.stack([W_U[s] - W_U[p] for s, p in zip(sg_ids, pl_ids)])  # [79, d]


def layer_norm(x, w, b):
    c = x - x.mean(dim=-1, keepdim=True)
    return w * c / torch.sqrt((c * c).mean(dim=-1, keepdim=True) + EPS) + b


def rotary(x, positions):  # GPT-NeoX: rotary on the first 16 of 64 dims, rotate_half on halves of 8
    rot = 16
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
    freqs = torch.outer(positions.to(torch.float64), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    xr, xp = x[..., :rot], x[..., rot:]
    half = torch.cat([-xr[..., rot // 2:], xr[..., : rot // 2]], dim=-1)
    return torch.cat([xr * cos + half * sin, xp], dim=-1)


def head4(x_all):
    """Head 4 of layer 3 at the last position: the attention row and the output (z @ W_O, no bias), float64."""
    T = x_all.shape[0]
    n = layer_norm(x_all, ln1_w, ln1_b)
    pos = torch.arange(T)
    q = rotary(n @ Wq + bq, pos)
    k = rotary(n @ Wk + bk, pos)
    v = n @ Wv + bv
    scores = (k @ q[-1]) / math.sqrt(DH)
    a = torch.softmax(scores, dim=-1)
    return a, (a @ v) @ Wo


# --- G1 and the geometry ----------------------------------------------------------------------------------------------
rw = torch.tensor(lock_011["read_weight"], dtype=torch.float64)
w_c = rw - rw.mean()
norm = float(torch.linalg.vector_norm(w_c))
norm2 = float(w_c @ w_c)
w_hat = w_c / norm
d_T = torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64)
geo = lock["geometry"]
check("w_c_digest_equals_lock", tensor_digest(w_c) == geo["w_c_sha256"] and tensor_digest(w_hat) == geo["w_hat_sha256"] and tensor_digest(d_T) == geo["d_T_sha256"]
      and norm == geo["w_c_norm"], norm)
from_weights = ln1_w * (Wv @ (Wo @ d_T))
from_weights = from_weights - from_weights.mean()
g1 = float((from_weights - w_c).abs().max())
check("G1_read_weight_from_checkpoint", g1 <= 1e-9, g1)
out["facts"]["G1_read_weight_max_abs_independent"] = g1
ones = torch.full((512,), 1 / math.sqrt(512), dtype=torch.float64)
iso, iso_digest_bad = {}, []
for fid in frames:
    vs = []
    for j in range(1, 9):
        v = cr.sha_gaussians(f"101|iso|{fid}|{j}", 512).double()
        for _ in range(2):
            for b in (ones, w_hat):
                v = v - torch.dot(v, b) * b
        v = v / torch.linalg.vector_norm(v)
        vs.append(v)
        if tensor_digest(v) != geo["iso_sha256"][fid][j - 1]:
            iso_digest_bad.append((fid, j))
    iso[fid] = vs
check("iso_directions_digests_equal_lock", not iso_digest_bad, iso_digest_bad[:3])
g1b = max(max(abs(float(torch.linalg.vector_norm(v)) - 1), abs(float(v.mean())), abs(float(v @ w_hat))) for vs in iso.values() for v in vs)
check("G1_basis_v_j", g1b <= 1e-12, g1b)

# --- per frame -------------------------------------------------------------------------------------------------------
COEF = {"NEC-READ": -1.0, "SUF-READ": 1.0, "NEC-AMP": 1.0, "SUF-NEG": -1.0, "NEC-HALF": -0.5, "SUF-HALF": 0.5, "NEC-OFF": -1.0, "SUF-OFF": 1.0}
REF_ARMS = {"REF", "SUF-READ", "SUF-OFF", "SUF-NEG", "SUF-HALF", "PC-SWAP-REF", "HEAD-SUF", "HEAD-SUF-V"} | {f"SUF-ISO-{j}" for j in range(1, 9)}
mx = {k: (0.0, "") for k in ("G2_r", "G2_length", "G2_direction", "G2_u_off32", "G3_pass_through", "pass_through_other", "G4_readout", "logit_contrast_all",
                             "G5_row", "G5_output", "head_delta_full_rel", "head_delta_value_rel", "G6_x3_all_REF")}


def bump(name, value, where):
    if value > mx[name][0] or not math.isfinite(value):
        mx[name] = (value, where)


repl_bad, noop_bad, x3pt_bad, g5_land_bad, head_repl_bad, g6_bad = [], [], [], [], [], []
overshoot, ln_records, manip = [], {}, {}
data = (EXP / "stage1/states.f64").read_bytes()
index = json.loads((EXP / "stage1/states.json").read_text(encoding="utf-8"))
check("states_data_sha256_equals_index", hashlib.sha256(data).hexdigest() == index["file_sha256"] == "1c7e51c049e1c61cf669f1783736054956073dc5af6947354b1d13c058ac5b66")
blocks = {e["name"]: torch.from_numpy(numpy.frombuffer(data, dtype="<f8", count=int(numpy.prod(e["shape"])), offset=int(e["offset_bytes"])).reshape(e["shape"]).copy())
          for e in index["blocks"]}
for fid in frames:
    T = lambda name: tensors[f"{fid}/{name}"]  # noqa: E731
    p_c, p_t = T("positions").tolist()
    x3pc, x3pt, x4pc, heads, mlp, Tpt, attn, h6, con = (T("x3_pc"), T("x3_pt"), T("x4_pc"), T("heads_pc"), T("mlp_pc"), T("T_pt"), T("attn_pt"), T("h6"),
                                                       T("contrasts"))
    x_ref = x3pc[row_of["REF"]]
    pairs = {}
    for cue in ("two", "three", "many"):
        x_cue = x3pc[row_of[f"PL@{cue}"]]
        dx = x_cue.double() - x_ref.double()
        dr = float(dx @ w_c)
        read = (dr / norm2) * w_c
        off = dx - read
        off_c = off - off.mean()
        off_norm = float(torch.linalg.vector_norm(off_c))
        u_off = off_c / off_norm
        s = abs(dr) / norm
        pairs[cue] = (dx, dr, read, u_off, s, off_norm)
        overshoot.append({"frame_id": fid, "cue": cue, "dr": dr, "dx_norm": float(torch.linalg.vector_norm(dx)), "dx_c_norm": float(torch.linalg.vector_norm(dx - dx.mean())),
                          "off_c_norm": off_norm, "s": s, "s_over_off_c": s / off_norm, "overshoot": off_norm < s})
        u32 = u_off.to(torch.float32).double()
        bump("G2_u_off32", max(abs(float(u32.mean())), abs(float(torch.linalg.vector_norm(u32)) - 1), abs(float(u32 @ w_hat))), f"{fid}|{cue}")
    # stage-1 state (own reader) and G6
    st = {k: blocks[f"{fid}/{k}"] for k in ("x3_all", "h6", "c_ref")}
    ok6 = (torch.equal(x_ref.double(), st["x3_all"][p_c]) and torch.equal(x3pt[row_of["REF"]].double(), st["x3_all"][p_t])
           and torch.equal(h6[row_of["REF"]].double(), st["h6"]) and torch.equal(T("c_ref_pm").double(), st["c_ref"]))
    if not ok6:
        g6_bad.append(fid)
    bump("G6_x3_all_REF", float((T("x3_all/REF").double() - st["x3_all"]).abs().max()), fid)
    # rows
    lnf_ref_state = layer_norm(st["h6"], lnf_w, lnf_b)
    c_ref_run = con[row_of["REF"]]
    for row, tag in enumerate(run_order):
        name, _, cue = tag.partition("@")
        key = f"{fid}|{tag}"
        pt_err = float((x4pc[row].double() - (x3pc[row].double() + heads[row].double().sum(0) + b_O + mlp[row].double())).abs().max())
        residual_edit = name not in ("REF", "PL") and not name.startswith("HEAD-")
        bump("G3_pass_through" if residual_edit else "pass_through_other", pt_err, key)
        lnf = layer_norm(h6[row].double(), lnf_w, lnf_b)
        bump("logit_contrast_all", float((dW @ lnf - con[row]).abs().max()), key)
        if residual_edit or name == "PL":
            readout = dW @ (lnf - lnf_ref_state)
            bump("G4_readout", float((readout - (con[row] - c_ref_run)).abs().max()), key)
        if name not in ("REF", "PL"):
            plain_row = row_of["REF"] if name in REF_ARMS else row_of[f"PL@{cue}"]
            if not torch.equal(x3pt[row], x3pt[plain_row]):
                x3pt_bad.append(key)
        if residual_edit:
            dx, dr, read, u_off, s, _ = pairs[cue]
            x_cue = x3pc[row_of[f"PL@{cue}"]]
            base = x_ref if name in REF_ARMS else x_cue
            if name == "NOOP":
                mine = x_cue.reshape(-1)
            elif name == "PC-SWAP-PL":
                mine = x_ref.reshape(-1)
            elif name == "PC-SWAP-REF":
                mine = x_cue.reshape(-1)
            elif name.startswith(("NEC-ISO-", "SUF-ISO-")):
                j = int(name.rsplit("-", 1)[1])
                coef = -1.0 if name.startswith("NEC") else 1.0
                mine = (base.double().reshape(-1) + coef * s * iso[fid][j - 1]).to(torch.float32)
            elif name in ("NEC-OFF", "SUF-OFF"):
                mine = (base.double().reshape(-1) + COEF[name] * s * u_off).to(torch.float32)
            else:
                mine = (base.double().reshape(-1) + COEF[name] * read).to(torch.float32)
            saved = T(f"replacement/{tag}")
            if not torch.equal(mine, saved):
                repl_bad.append((key, int((mine != saved).sum()), float((mine.double() - saved.double()).abs().max())))
            if name == "NOOP":
                continue
            xp = x3pc[row].double()
            e = xp - base.double()
            e_c = e - e.mean()
            e_norm = float(torch.linalg.vector_norm(e_c))
            if name.startswith("PC-SWAP"):
                other = (x_ref if name == "PC-SWAP-PL" else x_cue).double()
                target_r = float(other @ w_c)
                target_len = float(torch.linalg.vector_norm((other - base.double()) - (other - base.double()).mean()))
                kind = None
            elif name in ("NEC-OFF", "SUF-OFF") or "-ISO-" in name:
                target_r, target_len, kind = float(base.double() @ w_c), s, "orth"
            else:
                target_r, target_len, kind = float(base.double() @ w_c) + COEF[name] * dr, abs(COEF[name]) * s, "par"
            bump("G2_r", abs(float(xp @ w_c) - target_r), key)
            bump("G2_length", abs(e_norm - target_len) / max(target_len, 1e-12), key)
            if kind == "par":
                bump("G2_direction", float(torch.linalg.vector_norm(e_c - float(e_c @ w_hat) * w_hat)) / e_norm, key)
            elif kind == "orth":
                bump("G2_direction", abs(float(e_c @ w_hat)) / e_norm, key)
            sigma = float(torch.sqrt(((xp - xp.mean()) ** 2).mean() + EPS))
            ln_records.setdefault(tag, []).append({"sigma": sigma, "g3": float(torch.linalg.vector_norm(ln1_w * e)) / sigma,
                                                   "gln2": float(torch.linalg.vector_norm(gamma_ln2 * e)) / sigma})
    # G3 NOOP bit for bit
    nr, pr = row_of["NOOP@two"], row_of["PL@two"]
    same = all(torch.equal(T(n)[nr], T(n)[pr]) for n in ("x3_pc", "x3_pt", "x4_pc", "heads_pc", "mlp_pc", "T_pt", "attn_pt", "h6", "contrasts", "logit_identity"))
    same = same and state["confirmation"]["stage2"]["logits_sha256"][fid][nr] == state["confirmation"]["stage2"]["logits_sha256"][fid][pr]
    if not same:
        noop_bad.append(fid)
    # G5 and the head arms
    for tag in ("HEAD-NEC@two", "HEAD-SUF@two", "HEAD-NEC-V@two", "HEAD-SUF-V@two"):
        name = tag.partition("@")[0]
        row = row_of[tag]
        source_tag = "REF" if name in REF_ARMS else "PL@two"
        src = row_of[source_tag]
        x_all = T(f"x3_all/{source_tag}").double()
        a, o = head4(x_all)
        bump("G5_row", float((a - attn[src][HEAD].double()).abs().max()), f"{fid}|{tag}")
        bump("G5_output", abs(float(o @ d_T) - float(Tpt[src].double() @ d_T)), f"{fid}|{tag}")
        if not torch.equal(Tpt[row], T(f"head_replacement/{tag}")):
            g5_land_bad.append(f"{fid}|{tag}")
        dx, dr, read, u_off, s, _ = pairs["two"]
        e = (-1.0 if name.startswith("HEAD-NEC") else 1.0) * read
        if name.endswith("-V"):
            weight = float(attn[src][HEAD, p_c].double())
            n0 = layer_norm(x_all[p_c], ln1_w, ln1_b)
            n1 = layer_norm(x_all[p_c] + e, ln1_w, ln1_b)
            delta = weight * ((n1 - n0) @ Wv @ Wo)
            metric = "head_delta_value_rel"
        else:
            x_edit = x_all.clone()
            x_edit[p_c] = x_edit[p_c] + e
            delta = head4(x_edit)[1] - o
            metric = "head_delta_full_rel"
        saved_delta = T(f"delta/{tag}").double()
        bump(metric, float((delta - saved_delta).abs().max()) / float(saved_delta.abs().max()), f"{fid}|{tag}")
        rebuilt = (Tpt[src].double() + saved_delta).to(torch.float32)
        if not torch.equal(rebuilt, T(f"head_replacement/{tag}")):
            head_repl_bad.append(f"{fid}|{tag}")
    # the manipulation check
    dT = {tag: float((Tpt[row].double() - Tpt[row_of["REF"]].double()) @ d_T) for row, tag in enumerate(run_order)}
    manip[fid] = dT

check("G2_replacements_rebuilt_bit_for_bit", not repl_bad, repl_bad[:5])
check("G3_noop_bit_for_bit_36", not noop_bad, noop_bad)
check("G4_x3_pt_bit_for_bit_all_non_plain_runs", not x3pt_bad, x3pt_bad[:3])
check("G5_landing_144", not g5_land_bad, g5_land_bad[:3])
check("head_replacement_is_captured_plus_delta", not head_repl_bad, head_repl_bad[:3])
check("G6_ref_vs_stage1_36", not g6_bad, g6_bad)
TOL = {"G2_r": 1e-4, "G2_length": 1e-4, "G2_direction": 1e-4, "G2_u_off32": 1e-6, "G3_pass_through": 1e-4, "G4_readout": 2e-2, "G5_row": 1e-4, "G5_output": 1e-3}
for name, (value, where) in mx.items():
    if name in TOL:
        check(f"{name}_within_tolerance", value <= TOL[name], {"max": value, "at": where, "tolerance": TOL[name], "margin": TOL[name] / value if value else math.inf})
out["facts"]["maxima"] = {k: {"max": v[0], "at": v[1]} for k, v in mx.items()}
print(json.dumps(out["facts"]["maxima"], indent=1))
check("head_deltas_reproduce", mx["head_delta_full_rel"][0] < 1e-8 and mx["head_delta_value_rel"][0] < 1e-8, (mx["head_delta_full_rel"][0], mx["head_delta_value_rel"][0]))
# every run's saved contrasts against this review's float64 recomputation from the same run's saved h6 (final LayerNorm and
# unembedding from the checkpoint): the contrast form of 020's logit identity, at its frozen tolerance 2e-2 (a first
# version of this script used an arbitrary 1e-3 and reported 3.34e-3 as a failure: that is the known float32 floor,
# cf. the recorded G4_logit maximum 3.14e-3 and 020's 3.3e-3)
check("logit_contrast_consistency_every_run_within_2e-2", mx["logit_contrast_all"][0] <= 2e-2, mx["logit_contrast_all"])
check("x3_all_REF_equals_stage1_x3_all_bitwise", mx["G6_x3_all_REF"][0] == 0.0, mx["G6_x3_all_REF"])

# --- the state's own records against these ----------------------------------------------------------------------------
g = state["confirmation"]["gates"]["tolerance"]
cmp = {}
for mine_name, theirs in (("G2_r", "G2_r"), ("G2_length", "G2_length"), ("G2_direction", "G2_direction"), ("G2_u_off32", "G2_u_off32"),
                          ("G3_pass_through", "G3_pass_through"), ("G4_readout", "G4_readout"), ("G5_row", "G5_row"), ("G5_output", "G5_output")):
    cmp[mine_name] = {"mine": mx[mine_name][0], "mine_at": mx[mine_name][1], "state": g[theirs]["max"], "state_at": g[theirs]["at"]}
out["facts"]["vs_state"] = cmp
print(json.dumps(cmp, indent=1))
# overshoot
two = [o for o in overshoot if o["cue"] == "two"]
out["facts"]["overshoot_two"] = {"flagged": sum(o["overshoot"] for o in two), "ratio_min": min(o["s_over_off_c"] for o in two),
                                  "ratio_median": sorted(o["s_over_off_c"] for o in two)[17:19], "ratio_max": max(o["s_over_off_c"] for o in two),
                                  "s_median": sorted(o["s"] for o in two)[17:19], "off_c_median": sorted(o["off_c_norm"] for o in two)[17:19],
                                  "dr_range": [min(o["dr"] for o in two), max(o["dr"] for o in two)]}
for cue in ("three", "many"):
    sel = [o for o in overshoot if o["cue"] == cue]
    out["facts"][f"overshoot_{cue}"] = {"flagged": sum(o["overshoot"] for o in sel), "ratio_min": min(o["s_over_off_c"] for o in sel),
                                         "ratio_max": max(o["s_over_off_c"] for o in sel), "dr_range": [min(o["dr"] for o in sel), max(o["dr"] for o in sel)]}
print(json.dumps({k: v for k, v in out["facts"].items() if k.startswith("overshoot")}, indent=1))
(OUT / "c_overshoot_per_frame.json").write_text(json.dumps(overshoot, indent=0) + "\n", encoding="utf-8")
# LayerNorm records vs the state's
recs = state["confirmation"]["secondary"]["records"]
ln_bad = []
for fid in frames:
    for tag, entry in recs[fid].items():
        if "sigma" in entry:
            pass
lnsum = {tag: {"sigma_med": sorted(r["sigma"] for r in rs)[17:19], "g3_max": max(r["g3"] for r in rs), "gln2_max": max(r["gln2"] for r in rs)}
         for tag, rs in ln_records.items() if tag.endswith("@two")}
out["facts"]["layer_norm_two"] = lnsum
# per-run equality of LayerNorm records and ΔT with the state's records (to float tolerance; the order of operations differs)
worst_ln, worst_dT = 0.0, 0.0
for fid in frames:
    k = 0
    for tag, entry in recs[fid].items():
        worst_dT = max(worst_dT, abs(entry["delta_T"] - manip[fid][tag]))
for tag, rs in ln_records.items():
    pass
for fid in frames:
    for row, tag in enumerate(run_order):
        e = recs[fid][tag]
        if "sigma" in e:
            x = tensors[f"{fid}/x3_pc"][row].double()
            base_row = row_of["REF"] if tag.partition("@")[0] in REF_ARMS else row_of[f"PL@{tag.partition('@')[2]}"]
            ee = x - tensors[f"{fid}/x3_pc"][base_row].double()
            sigma = float(torch.sqrt(((x - x.mean()) ** 2).mean() + EPS))
            worst_ln = max(worst_ln, abs(sigma - e["sigma"]), abs(float(torch.linalg.vector_norm(ln1_w * ee)) / sigma - e["gamma3_e_over_sigma"]),
                           abs(float(torch.linalg.vector_norm(gamma_ln2 * ee)) / sigma - e["gamma_ln2_e_over_sigma"]))
check("layer_norm_records_reproduce", worst_ln < 1e-12, worst_ln)
check("manipulation_delta_T_reproduce", worst_dT < 1e-12, worst_dT)
# manipulation-check summary for two
def med(v):
    s = sorted(v)
    return 0.5 * (s[17] + s[18])


manip_two = {}
for tag in [t for t in run_order if t.endswith("@two")]:
    ratios = [manip[fid][tag] / manip[fid]["PL@two"] for fid in frames]
    manip_two[tag] = {"median_ratio": med(ratios), "min": min(ratios), "max": max(ratios)}
out["facts"]["manipulation_two"] = manip_two
(OUT / "c_gates_independent.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
print("failed:", [k for k, v in out["checks"].items() if not v["ok"]])
