"""Review item 5 (and the ladder of item 12; the I1 discrimination argument of items 3 and 11). One worker of N.

Weights only: Module.__call__, every class forward and every capture/intervention entry point refused, and
cr.run_gates / identity_gates / level1_gates / recompute_c / vector_factors / lexicon_of refused (own loops, own row
indexing, own patched vectors from r3, own lexicon of the patched vector). Allowed pinned routines only:
ul.compose_dx3, ul.i3_error, ul.contrast_of, ul.block0_terms, ul.reference_embeddings, ul.reference_rows_017,
rd.ReadoutProgram.level1_detail (with its contrast), pm.lexicon_vector.

For every run of its frames: C recomputed from the saved x3 (bit for bit), the Level-1 identity against the saved Δc,
I1 (vector), I3, I4; the ladder SSEs on base / noun±0.32; and I1 recomputed with a deliberately wrong vector (the
opposite sign, another control's sign, the plurality control) to show I1 binds each saved x1 to its labelled vector.
Usage: r4_gates.py <worker index> <number of workers>. Writes r4_worker{k}.pt and r4_worker{k}.json here.
"""
import guard  # noqa: F401

import json
import math
import sys
import time
from pathlib import Path

import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import models
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul

K_WORKER, N_WORKERS = int(sys.argv[1]), int(sys.argv[2])
ROOT = Path(guard.ROOT)
HERE = Path(guard.HERE)
EXP = ROOT / "experiments/025-nounness-direction-intervention"
guard.seal()
REFUSED = ("run_gates", "identity_gates", "level1_gates", "recompute_c", "vector_factors", "lexicon_of", "per_cue", "statistics", "ladder", "measure_rotated",
           "stage_two", "patch_path_check")
saved_cr = guard.refuse_functions(cr, REFUSED, "cr")
t_start = time.time()

freeze = json.loads((EXP / "confirmation-v1.json").read_text())
lock = json.loads((EXP / "preregistration-lock.json").read_text())
CONDITIONS = list(freeze["conditions"])
CUES = [{"word": c["word"], "token_id": int(c["token_id"]), "stratum": c["stratum"]} for c in freeze["cues"]]
LADDER = ("base", "noun+0.32", "noun-0.32")
BEARING = set(lock["configuration"]["outcome_bearing"])

inputs = ul.load_frozen_inputs(ROOT, tracked=guard.tracked)
model = models.load_model(models.PYTHIA_70M)
sealed = guard.seal_model(model)
assert rr.parameters_digest(model) == lock["dependencies"]["model"]["parameters_sha256"]
progs = ul.ModelPrograms.from_model(model, inputs)
del model
W_E = progs.weights.W_E
assert rr.embedding_digest(W_E) == lock["dependencies"]["model"]["embedding_sha256"]
own = torch.load(HERE / "vectors32_own.pt", weights_only=True)
assert own["conditions"] == CONDITIONS
VEC = {int(t): {c: v[i] for i, c in enumerate(CONDITIONS)} for t, v in own["vectors32"].items()}
saved = torch.load(ROOT / "outputs/experiment-025/stage2-measurements.pt", mmap=True, weights_only=True)
frames_by_id = {f.frame_id: f for f in inputs.pool.frames}
FRAME_IDS = sorted(frames_by_id)
states = ul.y1_states(inputs.closure["exploration"]["locked_states"], [frames_by_id[f] for f in FRAME_IDS])
REF_IDS = {k: int(v) for k, v in inputs.pool.reference_ids.items()}
CHAIN_REF = {k: int(v) for k, v in progs.chain.fcm.read.reference_ids.items()}


def group_of(frame):
    return "coordinated" if frame.template_id == "coordinated-adjective" else "cue_final"


GROUP_FRAMES = {g: [fid for fid in FRAME_IDS if group_of(frames_by_id[fid]) == g] for g in ("cue_final", "coordinated")}
T_RANK = {c["token_id"]: i for i, c in enumerate(sorted(CUES, key=lambda c: c["token_id"]))}


def row_of(group, token_id, frame_id):
    return T_RANK[token_id] * len(GROUP_FRAMES[group]) + GROUP_FRAMES[group].index(frame_id)


def my_lexicon(v32):
    """MLP₀(LN₂(v)) for a float32 vector: population-variance LayerNorm, exact GELU, in float32."""
    w = progs.weights
    x = v32.to(torch.float32)
    centered = x - x.mean(dim=-1, keepdim=True)
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    hidden = w.ln2_0_w * centered / torch.sqrt(variance + w.eps) + w.ln2_0_b
    return torch.nn.functional.gelu(hidden @ w.mlp0_W_in + w.mlp0_b_in) @ w.mlp0_W_out + w.mlp0_b_out


my_frames = GROUP_FRAMES["cue_final"][K_WORKER::N_WORKERS] + GROUP_FRAMES["coordinated"][K_WORKER::N_WORKERS]
records = []  # one per run
swap = []  # (key, which, i1_wrong)
lex_check = {"base_equal_pm_lexicon": True}
n_frames = len(my_frames)
print(f"worker {K_WORKER}/{N_WORKERS}: {n_frames} frames, threads {torch.get_num_threads()}, sealed {len(sealed)}, setup {time.time() - t_start:.1f}s", flush=True)
for fi, fid in enumerate(my_frames):
    frame = frames_by_id[fid]
    g = group_of(frame)
    state = states[fid]
    s17 = state.state_017
    ref_id = REF_IDS[frame.template_id]
    x0_all = ul.reference_embeddings(progs.weights, frame, ref_id)
    rows16 = ul.reference_rows_017(progs.programs, state)
    lex_ref = pm.lexicon_vector(progs.weights, CHAIN_REF[frame.template_id]).double()
    w_ref = W_E[ref_id].double()
    positions = sorted({frame.p_c, frame.p_t})
    t_frame = time.time()
    for c in sorted(CUES, key=lambda c: c["token_id"]):
        tid = c["token_id"]
        r = row_of(g, tid, fid)
        if fi == 0:
            lex_check["base_equal_pm_lexicon"] &= bool(torch.equal(my_lexicon(VEC[tid]["base"]), pm.lexicon_vector(progs.weights, tid)))

        def i1_of(v32, dx1):
            d_emb = v32.to(torch.float32).double() - w_ref
            delta_e = my_lexicon(v32).double() - lex_ref
            b_pc = ul.block0_terms(progs.program0, x0_all, frame.p_c, frame.p_c, d_emb)
            b_pt = ul.block0_terms(progs.program0, x0_all, frame.p_c, frame.p_t, d_emb) if frame.p_t != frame.p_c else None
            value = float((d_emb + delta_e + b_pc.total - dx1[frame.p_c]).abs().max())
            if b_pt is not None:
                value = max(value, float((b_pt.total - dx1[frame.p_t]).abs().max()))
            return value, d_emb, delta_e, b_pc, b_pt

        for cond in CONDITIONS:
            x1 = saved[f"{g}/{cond}/x1"][r]
            x3 = saved[f"{g}/{cond}/x3"][r]
            dc = saved[f"{g}/{cond}/dc"][r].double()
            C = saved[f"{g}/{cond}/C"][r]
            dx1 = {frame.p_c: x1[0].double() - s17.x1_all[frame.p_c].double()}
            dx3 = {frame.p_c: x3[0].double() - state.x3_all[frame.p_c].double()}
            if frame.p_t != frame.p_c:
                dx1[frame.p_t] = x1[1].double() - s17.x1_all[frame.p_t].double()
                dx3[frame.p_t] = x3[1].double() - state.x3_all[frame.p_t].double()
            c_again = ul.contrast_of(progs, state, dx3)
            c_equal = bool(torch.equal(c_again, C))
            c_diff = float((c_again - C).abs().max())
            l1 = progs.readout.contrast(state, progs.readout.level1_detail(state, dict(dx3), l4_heads=True)["dh6"], progs.nouns)[progs.scorable].double()
            level1 = float((l1 - dc).abs().max())
            i1, d_emb, delta_e, b_pc, b_pt = i1_of(VEC[tid][cond], dx1)
            factors = ul.Factors(delta_e, d_emb, b_pc.value, b_pc.pattern, None if b_pt is None else b_pt.total, b_pc, b_pt)
            ctx = ul.PairContext(frame, state, rows16, factors, tid, c["word"])
            full = ul.compose_dx3(progs, ctx, ul.FULL_MASK)
            i3 = ul.i3_error(full, dx3)
            i4 = float((ul.contrast_of(progs, state, full) - C.double()).abs().max())
            rec = {"frame": fid, "token_id": tid, "word": c["word"], "stratum": c["stratum"], "cond": cond, "c_equal": c_equal, "c_diff": c_diff,
                   "level1": level1, "I1": i1, "I3": i3, "I4": i4}
            if cond in LADDER:
                c5 = progs.readout.contrast(state, progs.readout.level1_detail(state, dict(dx3), l4_heads=False)["dh6"], progs.nouns)[progs.scorable].double()
                rec["sse_C"] = float(((dc - C.double()) ** 2).sum())
                rec["sse_C5"] = float(((dc - c5) ** 2).sum())
                rec["sse_L1"] = float(((dc - l1) ** 2).sum())
            records.append(rec)
            # the discrimination control: the same saved x1 against a deliberately wrong vector
            wrong = {"noun+0.32": ("noun-0.32", "plur+0.32", "base"), "noun-0.32": ("noun+0.32",), "rand1+0.32": ("rand1-0.32", "rand2+0.32"),
                     "rand4+0.32": ("rand4-0.32",), "plur+0.32": ("plur-0.32", "noun+0.32")}.get(cond, ())
            for other in wrong:
                swap.append({"slot": cond, "vector": other, "stratum": c["stratum"], "I1_wrong": i1_of(VEC[tid][other], dx1)[0]})
    print(f"  frame {fi + 1}/{n_frames} {fid} ({g}) {time.time() - t_frame:.1f}s; elapsed {(time.time() - t_start) / 60:.1f} min", flush=True)

# the module's own vector route, only now, on a few runs (a cross-check of my vector I1 against cr.vector_factors)
guard.restore_functions(cr, saved_cr)
xc = 0.0
for fid in my_frames[:2]:
    frame = frames_by_id[fid]
    ref_id = REF_IDS[frame.template_id]
    x0_all = ul.reference_embeddings(progs.weights, frame, ref_id)
    for c in CUES[::13]:
        for cond in ("noun+0.32", "noun-0.32", "rand3+0.32"):
            f_mod = cr.vector_factors(progs, x0_all, frame, VEC[c["token_id"]][cond], ref_id)
            d_emb = VEC[c["token_id"]][cond].double() - W_E[ref_id].double()
            delta_e = my_lexicon(VEC[c["token_id"]][cond]).double() - pm.lexicon_vector(progs.weights, CHAIN_REF[frame.template_id]).double()
            xc = max(xc, float((f_mod.d_emb - d_emb).abs().max()), float((f_mod.delta_e - delta_e).abs().max()))
summary = {"worker": K_WORKER, "n_workers": N_WORKERS, "frames": my_frames, "runs": len(records), "swaps": len(swap), "lexicon_check": lex_check,
           "vector_factors_max_diff": xc, "minutes": (time.time() - t_start) / 60, "threads": torch.get_num_threads(), "guard": guard.EVENTS,
           "chain_ref": CHAIN_REF, "pool_ref": REF_IDS}
torch.save({"records": records, "swap": swap, "summary": summary}, HERE / f"r4_worker{K_WORKER}.pt")
(HERE / f"r4_worker{K_WORKER}.json").write_text(json.dumps(summary, indent=1) + "\n")
print("done", json.dumps({k: v for k, v in summary.items() if k != "frames"}), flush=True)
