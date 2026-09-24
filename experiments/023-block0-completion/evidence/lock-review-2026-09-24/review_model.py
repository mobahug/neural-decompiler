"""Independent lock review, weights part (items 3, 4, 5, 7 and the model-derived fields of item 2).

Runs with Experiment 022's table blocked (audit hook + torch.load patch, proven live), and with nn.Module.__call__ and
every capture / intervention entry point refused BEFORE the model is loaded (the probe showed the load executes no
module). The model is loaded only to extract its weights (ul.ModelPrograms.from_model), then deleted.

Independence: the Y1 table, its bytes, its index, the P0-dx3 / factor digests, I5 and the block-0 algebra are built
here by my own loop and serialization. Only 022's program-defining functions are called inside that loop
(ul.pair_context, ul.compose_dx3, ul.contrast_of, ul.reference_rows_017, rd.state_from_locked, rd.predicted_dx3);
no b0c.prediction_tables / pair_predictions / closed_form_t / table_meta / table_layout / lock_conditions /
y2_table_spec / build_lock and no ul.table_units / write_table / table_bytes / table_index is called.
"""
import sys
SCRATCH = "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/lockreview023"
sys.path.insert(0, SCRATCH)
import guard  # noqa: E402  (first)

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

T0 = time.time()
OUT = Path(SCRATCH)
ROOT = guard.ROOT
report: dict = {"guard_proof": guard.prove_live()}


def say(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def cj(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f64(tensor: torch.Tensor) -> bytes:
    return tensor.detach().to(torch.float64).contiguous().numpy().astype("<f8", copy=False).tobytes()


say(f"guard live: {report['guard_proof']['live']}; torch threads {torch.get_num_threads()}")
report["torch_threads"] = torch.get_num_threads()

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

inputs = ul.load_frozen_inputs(ROOT)
say("frozen inputs loaded (ul.load_frozen_inputs)")

report["refusal"] = guard.refuse_execution()  # BEFORE the load
from neural_decompiler.models import PYTHIA_70M, load_model  # noqa: E402

model = load_model(PYTHIA_70M)
modules = list(model.modules())
report["model"] = {"modules": len(modules), "classes_not_refused_directly": sorted({type(m).__name__ for m in modules if type(m).__call__ is not guard.REFUSE_CALL}),
                   "execution_refused": guard.execution_refused()}
progs = ul.ModelPrograms.from_model(model, inputs)
del model, modules
gc.collect()
say(f"weights extracted; model deleted; execution refused {guard.execution_refused()}")


def reachable_modules(root_obj, limit=500000):
    """Every nn.Module reachable from ``root_obj`` through attributes and containers (tensors are leaves); functions,
    bound methods, classes and Python modules are reported separately (a bound method of a module would be a route)."""
    import types
    seen, stack, found, callables, count = set(), [(root_obj, "progs")], [], [], 0
    while stack and count < limit:
        obj, path = stack.pop()
        if id(obj) in seen or isinstance(obj, (torch.Tensor, str, bytes, int, float, bool, complex, type(None), np.ndarray, torch.dtype, torch.device)):
            continue
        seen.add(id(obj))
        count += 1
        if isinstance(obj, torch.nn.Module):
            found.append(path)
            continue
        if isinstance(obj, types.MethodType):
            callables.append(path)
            stack.append((obj.__self__, f"{path}.__self__"))
            continue
        if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.ModuleType, type)):
            callables.append(path)
            continue
        if isinstance(obj, dict):
            stack.extend((v, f"{path}[{k!r}]") for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend((v, f"{path}[{i}]") for i, v in enumerate(obj))
        else:
            if hasattr(obj, "__dict__"):
                stack.extend((v, f"{path}.{k}") for k, v in vars(obj).items())
            for s in getattr(type(obj), "__slots__", ()) or ():
                if isinstance(s, str) and hasattr(obj, s):
                    stack.append((getattr(obj, s), f"{path}.{s}"))
    return found, count, callables


found, scanned, callables = reachable_modules(progs)
alive = sorted({type(o).__name__ for o in gc.get_objects() if isinstance(o, torch.nn.Module)})
report["no_module_reachable"] = {"reachable_from_progs": found, "objects_scanned": scanned, "callables_reachable": callables[:50], "n_callables": len(callables),
                                 "nn_modules_alive_after_gc": alive}
say(f"nn.Module reachable from progs: {found[:5]} ({scanned} objects scanned; callables {len(callables)}: {callables[:8]}); alive after gc: {alive}")

# ---------------------------------------------------------------------------------------------------------------
# The frozen units, rebuilt by my own code: cues of the confirmation file by token id; the exposed frames (pool order)
# by frame_id; per group, token-major then frame.
confirmation = json.loads((ROOT / "experiments/023-block0-completion/confirmation-v1.json").read_text(encoding="utf-8"))
COORD = pm.COORDINATED_TEMPLATE
tokens = sorted(({"word": c["word"], "token_id": int(c["token_id"]), "class": c["class"]} for c in confirmation["cues"]), key=lambda c: c["token_id"])
pool_frames = list(inputs.pool.frames)
assert confirmation["exposed_frame_ids"] == [fr.frame_id for fr in pool_frames], "the confirmation names another exposed pool"
assert {k: int(v) for k, v in confirmation["reference_cue_ids"].items()} == {k: int(v) for k, v in inputs.pool.reference_ids.items()}
frames = sorted(pool_frames, key=lambda fr: fr.frame_id)
GROUPS = ("cue_final", "coordinated")
pairs = {g: [(t, f) for t in range(len(tokens)) for f, fr in enumerate(frames) if (fr.template_id == COORD) == (g == "coordinated")] for g in GROUPS}
MASK_P0 = 0
MASK_P1 = {"cue_final": ul.BIT["emb"] | ul.BIT["Bv"] | ul.BIT["Bp"], "coordinated": ul.BIT["emb"] | ul.BIT["Bv"] | ul.BIT["Bp"] | ul.BIT["T"]}
assert ul.BIT == {"R": 1, "emb": 2, "Bv": 4, "Bp": 8, "T": 16} and MASK_P1 == {"cue_final": 14, "coordinated": 30}
ref_ids = {k: int(v) for k, v in inputs.pool.reference_ids.items()}
locked = inputs.closure["exploration"]["locked_states"]
states = {fr.frame_id: rd.state_from_locked(locked[fr.frame_id], fr) for fr in frames}
noun_keys_progs = [progs.nouns.nouns[i].lexical_key for i in progs.scorable]
noun_keys_pool = [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]
report["units"] = {"tokens": len(tokens), "frames": len(frames), "pairs": {g: len(p) for g, p in pairs.items()}, "masks": {"P0": MASK_P0, "P1": MASK_P1},
                   "nouns": len(noun_keys_progs), "noun_keys_progs_equal_pool_single_token": noun_keys_progs == noun_keys_pool}
say(f"units: {report['units']}")

# ---------------------------------------------------------------------------------------------------------------
# My own block-0 attention (layer-0) implementation from the program's weight tensors: LayerNorm, projections, half-split
# partial rotary, softmax, W_O. Used for the independent V, P, ΔA0(p_c), the closed-form T and the exact ΔA0(p_t).
P0G = progs.program0
H, D_MODEL, D_HEAD = P0G.W_Q.shape
SQRT_D = math.sqrt(D_HEAD)
ROT, BASE = int(P0G.rotary_dim), float(P0G.rotary_base)
W_E = progs.weights.W_E


def my_ln(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    mu = x.sum() / x.numel()
    c = x - mu
    var = (c * c).sum() / x.numel()
    return c / torch.sqrt(var + P0G.eps) * P0G.ln_w + P0G.ln_b


def my_proj(n: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack([n @ W[h] for h in range(H)]) + b  # [H, d_head]


def my_rot(x: torch.Tensor, pos: int) -> torch.Tensor:
    half = ROT // 2
    i = torch.arange(half, dtype=torch.float64)
    ang = float(pos) * torch.pow(torch.tensor(BASE, dtype=torch.float64), -2.0 * i / ROT)
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[:, :half], x[:, half:ROT]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[:, ROT:]], dim=1)


def my_out(v: torch.Tensor) -> torch.Tensor:
    return torch.stack([v[h] @ P0G.W_O[h] for h in range(H)])  # [H, d_model]


def my_softmax(s: torch.Tensor) -> torch.Tensor:
    e = torch.exp(s - s.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)


def reference(frame, ref_id: int, q_pos: int):
    ids = list(frame.prefix_ids) + [int(ref_id)] + list(frame.suffix_ids)
    normed = [my_ln(W_E[int(tok)]) for tok in ids[: q_pos + 1]]
    keys = [my_rot(my_proj(n, P0G.W_K, P0G.b_K), k) for k, n in enumerate(normed)]
    vals = [my_proj(n, P0G.W_V, P0G.b_V) for n in normed]
    q = my_rot(my_proj(normed[q_pos], P0G.W_Q, P0G.b_Q), q_pos)
    scores = torch.stack([(q * keys[k]).sum(-1) for k in range(q_pos + 1)], dim=1) / SQRT_D  # [H, K]
    outs = [my_out(v) for v in vals]  # K × [H, d_model]
    return q, keys, vals, scores, my_softmax(scores), outs


def block0_mine(frame, ref_id: int, token_id: int):
    p_c, p_t = frame.p_c, frame.p_t
    n_new = my_ln(W_E[int(token_id)])
    k_new = my_rot(my_proj(n_new, P0G.W_K, P0G.b_K), p_c)
    o_new = my_out(my_proj(n_new, P0G.W_V, P0G.b_V))
    q_new = my_rot(my_proj(n_new, P0G.W_Q, P0G.b_Q), p_c)
    # (a) query p_c: the new row, V, P and the exact ΔA0(p_c)
    _, keys, _, scores, A, outs = reference(frame, ref_id, p_c)
    s_new = torch.stack([(q_new * keys[k]).sum(-1) for k in range(p_c)] + [(q_new * k_new).sum(-1)], dim=1) / SQRT_D
    r_new = my_softmax(s_new)
    outs_new = outs[:p_c] + [o_new]
    V = (A[:, p_c:p_c + 1] * (o_new - outs[p_c])).sum(0)
    P = sum(((r_new[:, k:k + 1] - A[:, k:k + 1]) * outs_new[k]).sum(0) for k in range(p_c + 1))
    total = sum((r_new[:, k:k + 1] * outs_new[k]).sum(0) for k in range(p_c + 1)) - sum((A[:, k:k + 1] * outs[k]).sum(0) for k in range(p_c + 1))
    res = {"V": V, "P": P, "total_pc": total}
    if p_t != p_c:
        # (b) query p_t: only the key and the value at p_c change -> one logit update per head (the design's closed form)
        q_t, keys_t, _, scores_t, A_t, outs_t = reference(frame, ref_id, p_t)
        a = A_t[:, p_c]
        rest = sum(A_t[:, k:k + 1] * outs_t[k] for k in range(p_t + 1) if k != p_c) / (1.0 - a)[:, None]
        shift = (q_t * (k_new - keys_t[p_c])).sum(-1) / SQRT_D
        a_new = 1.0 / (1.0 + torch.exp(-(torch.log(a / (1.0 - a)) + shift)))
        T = (a[:, None] * (o_new - outs_t[p_c]) + (a_new - a)[:, None] * (o_new - rest)).sum(0)
        s2 = scores_t.clone()
        s2[:, p_c] = (q_t * k_new).sum(-1) / SQRT_D
        A2 = my_softmax(s2)
        outs2 = [o_new if k == p_c else outs_t[k] for k in range(p_t + 1)]
        exact_pt = sum((A2[:, k:k + 1] * outs2[k]).sum(0) for k in range(p_t + 1)) - sum((A_t[:, k:k + 1] * outs_t[k]).sum(0) for k in range(p_t + 1))
        res.update({"T": T, "exact_pt": exact_pt})
    return res


def worse(entry: dict, value: float, where: str) -> None:
    if math.isnan(value):
        entry.update({"max": float("nan"), "at": where, "nan": True})
    elif not entry.get("nan") and value > entry["max"]:
        entry.update({"max": value, "at": where})


checks = {name: {"max": 0.0, "at": ""} for name in (
    "I5_max_abs", "a_022_definitional_V+P-total", "a_mine_V+P-total_mine", "a_mine_V-vs-022_V", "a_mine_P-vs-022_P", "a_mine_total-vs-022_total",
    "a_mine_V+P-vs-022_total", "b_closed_form_T-vs-022_attn_pt", "b_exact_mine-vs-022_attn_pt", "b_closed_form_T-vs-exact_mine")}
i5_bitwise = {"pairs": 0, "bitwise_equal": 0, "key_sets_equal": 0}
level0, factor_bytes = hashlib.sha256(), hashlib.sha256()
blocks: dict[str, torch.Tensor] = {}
rows16_cache: dict = {}
n_done = 0
for group in GROUPS:
    out = torch.empty(len(pairs[group]), 2, len(progs.scorable), dtype=torch.float64)
    for row, (t, f) in enumerate(pairs[group]):
        frame, token = frames[f], tokens[t]
        state = states[frame.frame_id]
        if frame.frame_id not in rows16_cache:
            rows16_cache[frame.frame_id] = ul.reference_rows_017(progs.programs, state)
        ref_id = ref_ids[frame.template_id]
        ctx = ul.pair_context(progs, frame, state, ref_id, token["token_id"], token["word"], rows16_cache[frame.frame_id])
        dx3_p0 = ul.compose_dx3(progs, ctx, MASK_P0)
        out[row, 0] = ul.contrast_of(progs, state, dx3_p0)
        out[row, 1] = ul.contrast_of(progs, state, ul.compose_dx3(progs, ctx, MASK_P1[group]))
        where = f"{token['word']}|{frame.frame_id}"
        committed = rd.predicted_dx3(progs.chain, progs.weights, state, ctx.rows16, ctx.token_id, frame.template_id)
        i5_bitwise["pairs"] += 1
        if set(committed) == set(dx3_p0):
            i5_bitwise["key_sets_equal"] += 1
            worse(checks["I5_max_abs"], max(float((dx3_p0[p] - committed[p]).abs().max()) for p in committed), where)
            i5_bitwise["bitwise_equal"] += int(all(torch.equal(dx3_p0[p], committed[p]) for p in committed))
        else:
            worse(checks["I5_max_abs"], math.inf, where)
        fac = ctx.factors
        mine = block0_mine(frame, ref_id, token["token_id"])
        worse(checks["a_022_definitional_V+P-total"], float((fac.value_pc + fac.pattern_pc - fac.block0_pc.total).abs().max()), where)
        worse(checks["a_mine_V+P-total_mine"], float((mine["V"] + mine["P"] - mine["total_pc"]).abs().max()), where)
        worse(checks["a_mine_V-vs-022_V"], float((mine["V"] - fac.value_pc).abs().max()), where)
        worse(checks["a_mine_P-vs-022_P"], float((mine["P"] - fac.pattern_pc).abs().max()), where)
        worse(checks["a_mine_total-vs-022_total"], float((mine["total_pc"] - fac.block0_pc.total).abs().max()), where)
        worse(checks["a_mine_V+P-vs-022_total"], float((mine["V"] + mine["P"] - fac.block0_pc.total).abs().max()), where)
        if group == "coordinated":
            assert fac.attn_pt is not None and "T" in mine
            worse(checks["b_closed_form_T-vs-022_attn_pt"], float((mine["T"] - fac.attn_pt).abs().max()), where)
            worse(checks["b_exact_mine-vs-022_attn_pt"], float((mine["exact_pt"] - fac.attn_pt).abs().max()), where)
            worse(checks["b_closed_form_T-vs-exact_mine"], float((mine["T"] - mine["exact_pt"]).abs().max()), where)
        else:
            assert fac.attn_pt is None and frame.p_t == frame.p_c
        for position in sorted(dx3_p0):
            level0.update(f64(dx3_p0[position]))
        for vector in (fac.delta_e, fac.d_emb, fac.value_pc, fac.pattern_pc) + (() if fac.attn_pt is None else (fac.attn_pt,)):
            factor_bytes.update(f64(vector))
        n_done += 1
        if n_done % 250 == 0:
            say(f"  {n_done}/2592 pairs; I5 {checks['I5_max_abs']['max']}; T {checks['b_closed_form_T-vs-022_attn_pt']['max']:.2e}")
    blocks[group] = out
    say(f"{group}: {len(pairs[group])} pairs done")

report["execution_still_refused_after_loop"] = guard.execution_refused()

# ---------------------------------------------------------------------------------------------------------------
# Serialization (my own), byte comparison, the index.
mine_bytes = b"".join(blocks[g].contiguous().numpy().astype("<f8").tobytes() for g in GROUPS)
flat = np.concatenate([blocks[g].numpy().reshape(-1) for g in GROUPS])
mine_bytes_struct = struct.pack(f"<{flat.size}d", *flat.tolist())
(OUT / "regenerated-y1-table.f64").write_bytes(mine_bytes)
cand_path = ROOT / "outputs/experiment-023/candidate-locked-y1-table.f64"
cand = cand_path.read_bytes()
cand_vals = np.frombuffer(cand, dtype="<f8")
mine_vals = np.frombuffer(mine_bytes, dtype="<f8")
differ = int(np.sum(cand_vals != mine_vals)) if cand_vals.size == mine_vals.size else None
report["y1_table"] = {"n_values": int(mine_vals.size), "bytes": len(mine_bytes), "struct_serialization_equal": mine_bytes == mine_bytes_struct,
                      "sha256_mine": sha(mine_bytes), "sha256_candidate": sha(cand), "byte_identical": mine_bytes == cand, "differing_values": differ,
                      "max_abs_diff": float(np.max(np.abs(cand_vals - mine_vals))) if differ is not None else None,
                      "all_finite": bool(np.isfinite(mine_vals).all()), "p0_dx3_sha256": level0.hexdigest(), "factors_sha256": factor_bytes.hexdigest()}
say(f"Y1 table: {report['y1_table']}")

index_cand_path = ROOT / "outputs/experiment-023/candidate-locked-y1-table.json"
index_text = index_cand_path.read_text(encoding="utf-8")
index_cand = json.loads(index_text)
from neural_decompiler import block0_completion as b0c  # constants only (TABLE_CONSTRUCTION, DESIGN)

offset, entries = 0, []
for g in GROUPS:
    shape = [len(pairs[g]), 2, len(progs.scorable)]
    nbytes = int(np.prod(shape)) * 8
    entries.append({"name": g, "shape": shape, "offset_bytes": offset, "nbytes": nbytes})
    offset += nbytes
expected_index = {"format": ul.TABLE_FORMAT, "dtype": "<f8", "blocks": entries, "total_bytes": offset, "file_sha256": sha(mine_bytes),
                  "experiment": "023", "kind": "Y1", "schema_version": 1, "construction": b0c.TABLE_CONSTRUCTION, "design": dict(b0c.DESIGN),
                  "pair_order": "cue token id, then frame_id",
                  "pairs": {g: [[tokens[t]["word"], tokens[t]["token_id"], frames[f].frame_id] for t, f in pairs[g]] for g in GROUPS},
                  "columns": ["P0", "P1"], "masks": {"P0": MASK_P0, "P1": dict(MASK_P1)}, "nouns": noun_keys_pool}
expected_index_text = cj(expected_index) + "\n"
report["y1_index"] = {"byte_identical": expected_index_text == index_text, "sha256_mine": sha(expected_index_text.encode("utf-8")),
                      "sha256_candidate": sha(index_text.encode("utf-8")), "keys_equal": sorted(expected_index) == sorted(index_cand),
                      "differing_fields": sorted(k for k in set(expected_index) | set(index_cand) if expected_index.get(k) != index_cand.get(k)),
                      "format": index_cand.get("format"), "blocks": index_cand.get("blocks"), "total_bytes": index_cand.get("total_bytes")}
say(f"Y1 index: {report['y1_index']}")

lock = json.loads((ROOT / "outputs/experiment-023/candidate-lock.json").read_text(encoding="utf-8"))
report["lock_model_fields"] = {"p0_dx3_sha256_equal": lock["y1_table"]["p0_dx3_sha256"] == level0.hexdigest(),
                               "factors_sha256_equal": lock["y1_table"]["factors_sha256"] == factor_bytes.hexdigest(),
                               "noun_keys_equal": lock["noun_keys"] == noun_keys_progs, "lock_gates": lock["y1_table"]["gates"]}
report["checks"] = checks
report["I5_bitwise"] = i5_bitwise
report["guard_end"] = {"refused_during_run": list(guard.REFUSED), "opened_022": sorted({p for p in guard.OPENED if "experiment-022" in p}),
                       "opened_hf": sorted({p for p in guard.OPENED if p.startswith(str(guard.HF))}), "execution_refused": guard.execution_refused()}
report["seconds"] = round(time.time() - T0, 1)
(OUT / "review_model.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
say("checks: " + json.dumps(checks, default=str))
say(f"I5 bitwise: {i5_bitwise}")
say(f"lock model fields: {report['lock_model_fields']}")
say(f"guard end: {report['guard_end']}")
say("done")
