"""Item 6 / 11 (reference-row alignment), all 108 frames, weights only: each frame's locked 020 rows4/rows5 (the D_attn
reference) equal the block-4/5 attention pattern at p_t recomputed from the same frame's locked residuals x4_all/x5_all
(atp.ReferenceRow, the exact attention program), and differ from every other same-length frame's rows. No forward.
Writes r9_refrows.json here.
"""
import guard  # noqa: F401

import json
from pathlib import Path

import torch

from neural_decompiler import attention_patterns as atp
from neural_decompiler import models
from neural_decompiler import upstream_localization as ul

ROOT = Path(guard.ROOT)
HERE = Path(guard.HERE)
guard.seal()
inputs = ul.load_frozen_inputs(ROOT, tracked=guard.tracked)
model = models.load_model(models.PYTHIA_70M)
guard.seal_model(model)
progs = ul.ModelPrograms.from_model(model, inputs)
del model
frames = {f.frame_id: f for f in inputs.pool.frames}
states = ul.y1_states(inputs.closure["exploration"]["locked_states"], list(frames.values()))
own, cross = 0.0, float("inf")
recomputed = {}
for fid, frame in sorted(frames.items()):
    st = states[fid]
    a4 = atp.ReferenceRow(progs.readout.programs[4], [x.double() for x in st.x4_all[: frame.p_t + 1]]).A_ref
    a5 = atp.ReferenceRow(progs.readout.programs[5], [x.double() for x in st.x5_all[: frame.p_t + 1]]).A_ref
    recomputed[fid] = (a4, a5)
    own = max(own, float((a4 - st.rows4).abs().max()), float((a5 - st.rows5).abs().max()))
for fid, (a4, a5) in recomputed.items():
    for other, frame in frames.items():
        if other != fid and frame.p_t == frames[fid].p_t:
            cross = min(cross, max(float((a4 - states[other].rows4).abs().max()), float((a5 - states[other].rows5).abs().max())))
out = {"frames": len(recomputed), "own_frame_max_abs_diff": own, "other_frame_min_of_max_abs_diff": cross, "guard": guard.EVENTS}
(HERE / "r9_refrows.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
