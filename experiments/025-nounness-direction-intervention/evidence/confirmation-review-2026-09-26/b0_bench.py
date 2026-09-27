"""Benchmark only (no model forward): memory and per-run cost of C, Level-1, and the I1/I3/I4 chain on a few runs."""
import guard  # noqa: F401  (audit hook first)

import json
import os
import resource
import time

import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import models
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import upstream_localization as ul

ROOT = guard.ROOT


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


guard.seal()
t0 = time.time()
inputs = ul.load_frozen_inputs(__import__("pathlib").Path(ROOT), tracked=guard.tracked)
print("inputs", round(time.time() - t0, 1), "s; rss", rss_mb(), "MB", flush=True)
model = models.load_model(models.PYTHIA_70M)
sealed = guard.seal_model(model)
print("model loaded; sealed", len(sealed), "; rss", rss_mb(), flush=True)
progs = ul.ModelPrograms.from_model(model, inputs)
print("progs; rss", rss_mb(), "threads", torch.get_num_threads(), flush=True)
saved = torch.load(os.path.join(ROOT, "outputs/experiment-025/stage2-measurements.pt"), mmap=True, weights_only=True)
print("artifact mmap; tensors", len(saved), "rss", rss_mb(), flush=True)
frames = sorted(inputs.pool.frames, key=lambda f: f.frame_id)
states = ul.y1_states(inputs.closure["exploration"]["locked_states"], frames)
print("states; rss", rss_mb(), flush=True)
print("FULL_MASK", ul.FULL_MASK, "chain reference ids", dict(progs.chain.fcm.read.reference_ids), "pool reference ids", dict(inputs.pool.reference_ids))

# a few runs of group cue_final and coordinated, condition noun+0.32
for group in ("cue_final", "coordinated"):
    gframes = [f for f in frames if (f.template_id == ul.COORDINATED) == (group == "coordinated")]
    x3 = saved[f"{group}/noun+0.32/x3"]
    dc = saved[f"{group}/noun+0.32/dc"]
    C = saved[f"{group}/noun+0.32/C"]
    tc = tl = 0.0
    n = 20
    for row in range(n):
        frame = gframes[row % len(gframes)]
        state = states[frame.frame_id]
        dx3 = {frame.p_c: x3[row][0].double() - state.x3_all[frame.p_c].double()}
        if frame.p_t != frame.p_c:
            dx3[frame.p_t] = x3[row][1].double() - state.x3_all[frame.p_t].double()
        a = time.time()
        c = ul.contrast_of(progs, state, dx3)
        tc += time.time() - a
        a = time.time()
        l1 = progs.readout.contrast(state, progs.readout.level1_detail(state, dict(dx3), l4_heads=True)["dh6"], progs.nouns)[progs.scorable].double()
        tl += time.time() - a
    print(group, "contrast_of ms", round(1000 * tc / n, 2), "level1 ms", round(1000 * tl / n, 2), flush=True)

# I1/I3/I4 chain timing on a few runs (token-id base vector = model row, just for timing)
W_E = progs.weights.W_E
for group in ("cue_final", "coordinated"):
    gframes = [f for f in frames if (f.template_id == ul.COORDINATED) == (group == "coordinated")]
    ti = 0.0
    n = 10
    for row in range(n):
        frame = gframes[row]
        state = states[frame.frame_id]
        ref = int(inputs.pool.reference_ids[frame.template_id])
        a = time.time()
        x0_all = ul.reference_embeddings(progs.weights, frame, ref)
        rows16 = ul.reference_rows_017(progs.programs, state)
        v = W_E[20138]
        d_emb = v.double() - W_E[ref].double()
        b_pc = ul.block0_terms(progs.program0, x0_all, frame.p_c, frame.p_c, d_emb)
        b_pt = ul.block0_terms(progs.program0, x0_all, frame.p_c, frame.p_t, d_emb) if frame.p_t != frame.p_c else None
        delta_e = pm.lexicon_vector(progs.weights, 20138).double() - pm.lexicon_vector(progs.weights, ref).double()
        factors = ul.Factors(delta_e, d_emb, b_pc.value, b_pc.pattern, None if b_pt is None else b_pt.total, b_pc, b_pt)
        ctx = ul.PairContext(frame, state, rows16, factors, 20138, "anxious")
        full = ul.compose_dx3(progs, ctx, ul.FULL_MASK)
        c = ul.contrast_of(progs, state, full)
        ti += time.time() - a
    print(group, "I1/I3/I4 chain ms (incl. rows16 + x0)", round(1000 * ti / n, 2), flush=True)
print("max rss MB", rss_mb(), "guard events", json.dumps(guard.EVENTS)[:500])
