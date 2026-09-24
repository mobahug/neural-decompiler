"""Item 7: independent program provenance (E4 recheck from the weights), P0 (mask 0) and P1 (mask 14 / 30) for all
18,900 exposed pairs, in the artifact's order, under a guard that makes any forward pass impossible.

Uses only ul's pure-tensor program functions in this reviewer's own loop; never b0c.recompute_p1 or any b0c
extraction/verification function."""

from __future__ import annotations

import gc
import math
import sys
import time

sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/extractreview023")

import numpy as np
import torch

from revlib import CAND_DATA, CAND_INDEX, REPO, SCRATCH, TABLE022, Report, strict_json

T0 = time.time()
rep = Report("item 7: E4 recheck from the weights")


def say(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler.models import PYTHIA_70M, load_model  # noqa: E402

say(f"torch {torch.__version__}, threads {torch.get_num_threads()}")
inputs = ul.load_frozen_inputs(REPO)
say("frozen inputs loaded")
index = strict_json(CAND_INDEX)
cells = np.frombuffer(CAND_DATA.read_bytes(), dtype="<f8").reshape(18900, 8)

# --- the model, for its weights only -------------------------------------------------------------------------------
model = load_model(PYTHIA_70M)
say(f"model loaded: {type(model).__name__}")
progs = ul.ModelPrograms.from_model(model, inputs)
del model
gc.collect()
live_modules = [type(o).__name__ for o in gc.get_objects() if isinstance(o, torch.nn.Module)]
rep.note(f"live nn.Module instances after deleting the model: {len(live_modules)} {sorted(set(live_modules))[:10]}")
say("programs extracted, model deleted")

# nouns: the program's scorable nouns vs the index's nouns
noun_keys = [progs.nouns.nouns[i].lexical_key for i in progs.scorable]
rep.check("program scorable nouns == index nouns (79, in order)", noun_keys == index["nouns"] and len(noun_keys) == 79, f"{len(noun_keys)}")

# --- the guard -------------------------------------------------------------------------------------------------------
guard_hits = {"pm": 0, "module": 0}


def refuse_pm(*args, **kwargs):
    guard_hits["pm"] += 1
    raise RuntimeError("guard: a plural_mechanism capture/intervention entry point was reached")


saved_pm = {}
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        saved_pm[name] = getattr(pm, name)
        setattr(pm, name, refuse_pm)
rep.note(f"pm entry points patched: {sorted(saved_pm)}")

orig_module_call = torch.nn.Module.__call__


def refuse_module_call(self, *args, **kwargs):
    guard_hits["module"] += 1
    raise RuntimeError(f"guard: torch.nn.Module.__call__ reached ({type(self).__name__})")


torch.nn.Module.__call__ = refuse_module_call
# prove the module guard is live
probe_ok = False
try:
    torch.nn.Identity()(torch.zeros(1))
except RuntimeError as error:
    probe_ok = "guard" in str(error)
guard_hits["module"] = 0  # the deliberate probe is not counted
rep.check("nn.Module.__call__ guard active (deliberate probe refused)", probe_ok)

# --- the table (read-only) -------------------------------------------------------------------------------------------
table = torch.load(TABLE022)
say("022 table loaded")
dc = table["dc"].double()
dc_hat = table["dc_hat"]
assert list(dc.shape) == [175, 108, 79] and list(dc_hat.shape) == [175, 108, 32, 79]

frames_by_id = {frame.frame_id: frame for frame in inputs.pool.frames}
locked = inputs.closure["exploration"]["locked_states"]
cues = index["cues"]
frames = index["frames"]
P0 = torch.empty(175, 108, 79, dtype=torch.float64)
P1 = torch.empty(175, 108, 79, dtype=torch.float64)
masks = []
structure_ok = True
completed = False
try:
    for fi, (frame_id, template, group) in enumerate(frames):
        frame = frames_by_id[frame_id]
        coordinated = template == "coordinated-adjective"
        if frame.template_id != template or (frame.p_t != frame.p_c) != coordinated or (frame.p_t == frame.p_c + 1) != coordinated:
            structure_ok = False
        mask = 30 if coordinated else 14  # emb|Bv|Bp (+T): 2+4+8 (+16)
        masks.append(mask)
        state = rd.state_from_locked(locked[frame_id], frame)
        rows16 = ul.reference_rows_017(progs.programs, state)
        reference_id = int(inputs.pool.reference_ids[frame.template_id])
        for ci, (word, token_id, _cls) in enumerate(cues):
            ctx = ul.pair_context(progs, frame, state, reference_id, int(token_id), word, rows16)
            P0[ci, fi] = ul.contrast_of(progs, state, ul.compose_dx3(progs, ctx, 0))
            P1[ci, fi] = ul.contrast_of(progs, state, ul.compose_dx3(progs, ctx, mask))
        if (fi + 1) % 6 == 0:
            d1 = float((P1[:, : fi + 1] - torch.stack([dc_hat[:, j, masks[j]] for j in range(fi + 1)], dim=1).double()).abs().max())
            d0 = float((P0[:, : fi + 1] - dc_hat[:, : fi + 1, 0].double()).abs().max())
            say(f"{fi + 1}/108 frames; running max |P1 - stored| {d1:.3e}, |P0 - stored| {d0:.3e}")
    completed = True
finally:
    torch.nn.Module.__call__ = orig_module_call
    for name, value in saved_pm.items():
        setattr(pm, name, value)

rep.check("recomputation completed under the guard", completed)
rep.check("guard triggers during the recomputation (pm, nn.Module)", guard_hits == {"pm": 0, "module": 0}, str(guard_hits))
rep.check("frame structure: template matches pool, p_t = p_c + 1 iff coordinated", structure_ok)

stored_p0 = dc_hat[:, :, 0].double()
stored_p1 = torch.stack([dc_hat[:, fi, masks[fi]] for fi in range(108)], dim=1).double()
d0 = (P0 - stored_p0).abs()
d1 = (P1 - stored_p1).abs()
rep.check("max |P0 - stored dc_hat[mask 0]| <= 1e-9", float(d0.max()) <= 1e-9, f"max {float(d0.max()):.3e}; exactly equal elements {int((d0 == 0).sum())}/{d0.numel()}")
rep.check("max |P1 - stored dc_hat[mask 14/30]| <= 1e-9 (E4)", float(d1.max()) <= 1e-9, f"max {float(d1.max()):.3e}; exactly equal elements {int((d1 == 0).sum())}/{d1.numel()}")
rep.note(f"P0 bit-identical: {bool(torch.equal(P0, stored_p0))}; P1 bit-identical: {bool(torch.equal(P1, stored_p1))}")
rep.check("all recomputed P0/P1 finite", bool(torch.isfinite(P0).all()) and bool(torch.isfinite(P1).all()))

# SSE0 / SSE1 from the recomputed predictions against the artifact's columns
sse0 = ((dc - P0) ** 2).sum(-1).reshape(-1).numpy()
sse1 = ((dc - P1) ** 2).sum(-1).reshape(-1).numpy()
for name, col, mine in (("SSE0", 3, sse0), ("SSE1", 4, sse1)):
    diff = np.abs(cells[:, col] - mine)
    rel = diff / np.maximum(np.abs(cells[:, col]), 1e-300)
    rep.check(f"{name} from recomputed predictions vs artifact column", int((cells[:, col] != mine).sum()) == 0,
              f"differing rows {int((cells[:, col] != mine).sum())}/18900, max |diff| {diff.max():.3e}, max rel {rel.max():.3e}")

torch.save({"P0": P0, "P1": P1, "masks": masks}, SCRATCH / "e4_recomputed_p0_p1.pt")
say("saved recomputed P0/P1 to the scratchpad")
rep.summary()
say("done")
