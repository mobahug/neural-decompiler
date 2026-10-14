"""Item 4b: an independent recomputation of the program's per-pair quantities, arm inputs, Δĉ and fractions, without
ri's table code (ri.predict_cue, ri.pair_quantities, ri.predicted_change, ri.main_fractions are NOT called).

- The upstream program Δx̂3 is the frozen program itself: ul.reference_rows_017, ul.pair_context and ul.compose_dx3 at
  mask 30 (computation 6's first bullet; not re-implemented).
- My own: w_c from 011's lock read_weight; Δr̂, read part, off_c, û_off, ŝ; each main arm's change at p_c and p_t per
  design §5's table; the fractions of design §4 and N̂, Ŝ, A_amp, A_neg; the weak flag of §7.
- Δĉ two ways: (a) ul.contrast_of on my inputs; (b) my own Level-0 readout from the raw checkpoint (safetensors, no
  TransformerBridge, no rd/ul/atp/lc code): block 3 exact attention (rotary, causal softmax) and MLP at the operating
  point, block 4 MLP with attention frozen, block 5 frozen reference rows times value changes plus MLP, LN_final, and the
  unembedding difference of each scorable noun's singular and plural ids.
Weights only; every forward refused; the token-sequence guard installed; nothing written in the repository."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402  (first)

guard.self_test()
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler.models import PYTHIA_70M, load_model, seed_runtime  # noqa: E402

ROOT = guard.ROOT
EXP = ROOT / ri.EXPERIMENT_DIR
SNAP = Path("/Users/gaborhorvath-ulenius/.cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c")
out: dict = {"checks": {}, "facts": {}}


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"), flush=True)


from transformers import AutoTokenizer  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(PYTHIA_70M.model_id, revision=PYTHIA_70M.revision)
guard.build_forbidden(tokenizer)
guard.install_forward_guards()

# --- inputs: the lock, the frozen frames, the stage-1 states parsed with numpy (offsets from the lock's layout) -------
lock = json.loads((EXP / "preregistration-lock.json").read_text(encoding="utf-8"))
conf = json.loads((EXP / "confirmation-v1.json").read_text(encoding="utf-8"))
confirmation = ri.confirmation_from_payload(conf, ri.PRODUCTION)
table = json.loads((EXP / "stage1/table.json").read_text(encoding="utf-8"))
data = numpy.frombuffer((EXP / "stage1/states.f64").read_bytes(), dtype="<f8")
blocks, offset = {}, 0
for entry in lock["stage1"]["states"]["layout"]:
    n = int(numpy.prod(entry["shape"]))
    blocks[entry["name"]] = data[offset: offset + n].reshape(entry["shape"]).copy()
    offset += n
assert offset * 8 == data.nbytes

# --- the frozen program's Δx̂3 (the model is loaded for its weights only) ----------------------------------------------
seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
from neural_decompiler.upstream_localization import load_frozen_inputs  # noqa: E402
from neural_decompiler.provenance import collect_git_state  # noqa: E402,F401
import subprocess  # noqa: E402


def tracked(path: Path) -> bool:
    return subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True).returncode == 0


inputs = load_frozen_inputs(ROOT, tracked=tracked)
model = load_model(PYTHIA_70M)
guard.seal(model)
progs = ul.ModelPrograms.from_model(model, inputs)
del model
gc.collect()
scorable = list(progs.scorable)
noun_ids = [(progs.nouns.nouns[i].sg_ids[0], progs.nouns.nouns[i].pl_ids[0]) for i in scorable]
check("noun_order_is_lock_order", [progs.nouns.nouns[i].lexical_key for i in scorable] == lock["noun_keys"])

# --- my geometry: w_c from 011's lock read_weight -----------------------------------------------------------------------
rw = numpy.array(inputs.lock_011["read_weight"], dtype=numpy.float64)
w_c = rw - rw.mean()
norm2 = float(w_c @ w_c)
norm = math.sqrt(norm2)
check("w_c_norm_matches_lock", abs(norm - lock["geometry"]["w_c_norm"]) <= 1e-12 * norm, {"mine": norm, "lock": lock["geometry"]["w_c_norm"]})
out["facts"]["w_c_norm"] = norm

# --- my readout from the raw checkpoint --------------------------------------------------------------------------------
cfg = json.loads((SNAP / "config.json").read_text())
sd = {k: v.to(torch.float64) for k, v in load_file(str(SNAP / "model.safetensors")).items() if v.is_floating_point()}
H, D = cfg["num_attention_heads"], cfg["hidden_size"]
HS = D // H
R = int(cfg["rotary_pct"] * HS)
EPS = cfg["layer_norm_eps"]
BASE = float(cfg["rotary_emb_base"])
assert cfg["use_parallel_residual"] and cfg["hidden_act"] == "gelu"


def P(name):
    return sd[f"gpt_neox.{name}"]


def ln(x, w, b):
    c = x - x.mean(-1, keepdim=True)
    return w * c / torch.sqrt((c * c).mean(-1, keepdim=True) + EPS) + b


def qkv(layer, x):
    y = P(f"layers.{layer}.attention.query_key_value.weight") @ x + P(f"layers.{layer}.attention.query_key_value.bias")
    y = y.reshape(H, 3 * HS)
    return y[:, :HS], y[:, HS:2 * HS], y[:, 2 * HS:]


def rot(x, pos):
    inv = 1.0 / (BASE ** (torch.arange(0, R, 2, dtype=torch.float64) / R))
    ang = pos * inv
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[:, : R // 2], x[:, R // 2: R]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[:, R:]], dim=-1)


def attn_out(layer, xs, q_pos):
    """Block `layer`'s attention output (without the dense bias) at query q_pos from residuals xs[0..q_pos]."""
    w, b = P(f"layers.{layer}.input_layernorm.weight"), P(f"layers.{layer}.input_layernorm.bias")
    parts = [qkv(layer, ln(xs[k], w, b)) for k in range(q_pos + 1)]
    q = rot(parts[q_pos][0], q_pos)
    keys = torch.stack([rot(parts[k][1], k) for k in range(q_pos + 1)], dim=1)  # [H, K, HS]
    vals = torch.stack([parts[k][2] for k in range(q_pos + 1)], dim=1)
    A = torch.softmax(torch.einsum("he,hke->hk", q, keys) / math.sqrt(HS), dim=-1)
    z = torch.einsum("hk,hke->he", A, vals).reshape(-1)
    return P(f"layers.{layer}.attention.dense.weight") @ z


def mlp(layer, x):
    h = ln(x, P(f"layers.{layer}.post_attention_layernorm.weight"), P(f"layers.{layer}.post_attention_layernorm.bias"))
    a = P(f"layers.{layer}.mlp.dense_h_to_4h.weight") @ h + P(f"layers.{layer}.mlp.dense_h_to_4h.bias")
    g = 0.5 * a * (1.0 + torch.special.erf(a / math.sqrt(2.0)))
    return P(f"layers.{layer}.mlp.dense_4h_to_h.weight") @ g + P(f"layers.{layer}.mlp.dense_4h_to_h.bias")


W_U = sd["embed_out.weight"]  # [vocab, d_model]
DW = torch.stack([W_U[sg] - W_U[pl] for sg, pl in noun_ids])  # [79, d_model]
LNF_W, LNF_B = P("final_layer_norm.weight"), P("final_layer_norm.bias")


def my_readout(fid, p_c, p_t, dx3):
    x3 = torch.from_numpy(blocks[f"{fid}/x3_all"])
    x4 = torch.from_numpy(blocks[f"{fid}/x4_all"])
    x5 = torch.from_numpy(blocks[f"{fid}/x5_all"])
    h6 = torch.from_numpy(blocks[f"{fid}/h6"])
    rows5 = torch.from_numpy(blocks[f"{fid}/rows5"])
    dh3, dh4 = {}, {}
    for pos in (p_c, p_t):
        new = [x3[k] + dx3[k] if k in dx3 else x3[k] for k in range(pos + 1)]
        a3 = attn_out(3, new, pos) - attn_out(3, [x3[k] for k in range(pos + 1)], pos)
        m3 = mlp(3, x3[pos] + dx3[pos]) - mlp(3, x3[pos])
        dh3[pos] = dx3[pos] + a3 + m3
    for pos in (p_c, p_t):
        dh4[pos] = dh3[pos] + mlp(4, x4[pos] + dh3[pos]) - mlp(4, x4[pos])  # block 4's attention frozen: no change
    w5, b5 = P("layers.5.input_layernorm.weight"), P("layers.5.input_layernorm.bias")
    a5 = torch.zeros(D, dtype=torch.float64)
    Wd5 = P("layers.5.attention.dense.weight")
    for k, d in dh4.items():  # frozen reference rows × the value changes at the changed keys
        dv = qkv(5, ln(x5[k] + d, w5, b5))[2] - qkv(5, ln(x5[k], w5, b5))[2]  # [H, HS]
        a5 = a5 + Wd5 @ (rows5[:, k].unsqueeze(-1) * dv).reshape(-1)
    m5 = mlp(5, x5[p_t] + dh4[p_t]) - mlp(5, x5[p_t])
    dh6 = dh4[p_t] + a5 + m5
    dln = ln(h6 + dh6, LNF_W, LNF_B) - ln(h6, LNF_W, LNF_B)
    return DW @ dln


# --- per frame and cue ---------------------------------------------------------------------------------------------------
ARMS = ("PL", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF", "PC-SWAP-PL", "PC-SWAP-REF")
worst = {"pair_rel": 0.0, "dc_ul_abs": 0.0, "mean_ul_abs": 0.0, "dc_mine_abs": 0.0, "frac_abs": 0.0, "dc_ul_vs_mine_abs": 0.0}
bitwise = {"pair": 0, "dc_ul": 0, "total_pairs": 0, "total_arms": 0}
weak_mismatch, cue_ids = 0, dict(ri.PRODUCTION.cues)
for frame in confirmation.frames:
    fid, p_c, p_t = frame.frame_id, frame.p_c, frame.p_t
    entry = {name: blocks[f"{fid}/{name}"].tolist() for name in ri.STATE_FIELDS}
    state = rd.state_from_locked({"p_c": p_c, "p_t": p_t, **entry}, frame)
    rows16 = ul.reference_rows_017(progs.programs, state)
    for label, cue_id in cue_ids.items():
        ctx = ul.pair_context(progs, frame, state, 581, int(cue_id), label, rows16)
        dx_hat = ul.compose_dx3(progs, ctx, b0c.P1_MASK["coordinated"])
        dpc, dpt = dx_hat[p_c].double(), dx_hat[p_t].double()
        d = dpc.numpy()
        dr = float(d @ w_c)
        read = (dr / norm2) * w_c
        off = d - read
        off_c = off - off.mean()
        off_norm = float(numpy.sqrt(off_c @ off_c))
        u_off = off_c / off_norm
        s = abs(dr) / norm
        mine_pair = {"dr": dr, "dx_norm": float(numpy.sqrt(d @ d)), "off_c_norm": off_norm, "s": s, "s_over_off_c": s / off_norm}
        tab = table["frames"][fid]["cues"][label]
        for key, value in mine_pair.items():
            worst["pair_rel"] = max(worst["pair_rel"], abs(value - tab["pair"][key]) / max(abs(tab["pair"][key]), 1e-300))
        bitwise["pair"] += all(mine_pair[k] == tab["pair"][k] for k in mine_pair)
        bitwise["total_pairs"] += 1
        weak_mismatch += tab["pair"]["overshoot"] != (off_norm < s)
        z = numpy.zeros(D)
        read_t, off_t = read, s * u_off
        design = {"PL": (d, dpt.numpy()), "NEC-READ": (d - read_t, dpt.numpy()), "SUF-READ": (read_t, z), "NEC-OFF": (d - off_t, dpt.numpy()), "SUF-OFF": (off_t, z),
                  "NEC-AMP": (d + read_t, dpt.numpy()), "SUF-NEG": (-read_t, z), "NEC-HALF": (d - 0.5 * read_t, dpt.numpy()), "SUF-HALF": (0.5 * read_t, z),
                  "PC-SWAP-PL": (z, dpt.numpy()), "PC-SWAP-REF": (d, z)}
        means = {}
        for arm in ARMS:
            at_pc, at_pt = design[arm]
            dx3 = {p_c: torch.from_numpy(numpy.ascontiguousarray(at_pc)), p_t: torch.from_numpy(numpy.ascontiguousarray(at_pt))}
            dc_ul = ul.contrast_of(progs, state, dx3).double()
            dc_mine = my_readout(fid, p_c, p_t, dx3)
            tab_dc = torch.tensor(tab["arms"][arm]["dc"], dtype=torch.float64)
            worst["dc_ul_abs"] = max(worst["dc_ul_abs"], float((dc_ul - tab_dc).abs().max()))
            worst["dc_mine_abs"] = max(worst["dc_mine_abs"], float((dc_mine - tab_dc).abs().max()))
            worst["dc_ul_vs_mine_abs"] = max(worst["dc_ul_vs_mine_abs"], float((dc_mine - dc_ul).abs().max()))
            bitwise["dc_ul"] += bool(torch.equal(dc_ul, tab_dc))
            bitwise["total_arms"] += 1
            means[arm] = float(dc_ul.mean())
            worst["mean_ul_abs"] = max(worst["mean_ul_abs"], abs(means[arm] - tab["arms"][arm]["mean"]))
        full = means["PL"]
        weak_mismatch += (full > -1.0) != tab["weak"]
        if label == "two":
            pl_in = {p_c: torch.from_numpy(numpy.ascontiguousarray(design["PL"][0])), p_t: torch.from_numpy(numpy.ascontiguousarray(design["PL"][1]))}
            out["facts"].setdefault("per_frame_two", []).append({
                "frame": fid[-2:], "dr_table": tab["pair"]["dr"], "dr_mine": dr, "s_table": tab["pair"]["s"], "s_mine": s,
                "dc_full_table": tab["dc_full"], "dc_full_mine_ul": full, "dc_full_mine_raw": float(my_readout(fid, p_c, p_t, pl_in).mean())})
        if tab["fractions"] is not None:
            mine_f = {"F_rem": means["NEC-READ"] / full, "F_rem_off": means["NEC-OFF"] / full, "F_ins": means["SUF-READ"] / full, "F_ins_off": means["SUF-OFF"] / full,
                      "A_amp": means["NEC-AMP"] - full, "A_neg": means["SUF-NEG"], "F_half_rem": means["NEC-HALF"] / full, "F_half_ins": means["SUF-HALF"] / full,
                      "F_swap_pl": means["PC-SWAP-PL"] / full, "F_swap_ref": means["PC-SWAP-REF"] / full}
            mine_f["N"] = mine_f["F_rem_off"] - mine_f["F_rem"]
            mine_f["S"] = mine_f["F_ins"] - mine_f["F_ins_off"]
            mine_f["additivity"] = mine_f["F_rem"] + mine_f["F_ins"]
            check_keys = set(mine_f) == set(tab["fractions"])
            if not check_keys:
                print("fraction keys differ", sorted(set(mine_f) ^ set(tab["fractions"])))
            for key in mine_f:
                worst["frac_abs"] = max(worst["frac_abs"], abs(mine_f[key] - tab["fractions"][key]))
out["facts"]["worst_differences"] = worst
out["facts"]["bitwise_counts"] = bitwise
check("pair_quantities_match", worst["pair_rel"] <= 1e-12, worst["pair_rel"])
check("dc_ul_contrast_of_on_my_inputs_matches_table", worst["dc_ul_abs"] <= 1e-10, worst["dc_ul_abs"])
check("means_match", worst["mean_ul_abs"] <= 1e-10, worst["mean_ul_abs"])
check("my_raw_weight_readout_matches_table", worst["dc_mine_abs"] <= 1e-8, worst["dc_mine_abs"])
check("fractions_match", worst["frac_abs"] <= 1e-9, worst["frac_abs"])
check("weak_and_overshoot_flags_match", weak_mismatch == 0, weak_mismatch)
out["guard"] = guard.summary()
check("guard_never_fired", not guard.RECORD["token_guard_hits"] and guard.RECORD["module_call_refusals"] == 0 and not guard.RECORD["entry_point_refusals"]
      and not guard.RECORD["repo_write_refusals"] and not guard.RECORD["subprocess_refusals"], guard.summary())
out["all_ok"] = all(c["ok"] for c in out["checks"].values())
(guard.OUT / "e_independent.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
print(json.dumps(out["facts"], indent=1))
print("ALL OK" if out["all_ok"] else "SOME CHECKS FAILED")
