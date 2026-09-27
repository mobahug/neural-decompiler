"""r9 detail: the per-frame distribution of the own-frame recomputation difference, by layer (weights only)."""
import guard  # noqa: F401
import json
from pathlib import Path
import torch
from neural_decompiler import attention_patterns as atp
from neural_decompiler import models
from neural_decompiler import upstream_localization as ul
ROOT = Path(guard.ROOT); HERE = Path(guard.HERE)
guard.seal()
inputs = ul.load_frozen_inputs(ROOT, tracked=guard.tracked)
model = models.load_model(models.PYTHIA_70M); guard.seal_model(model)
progs = ul.ModelPrograms.from_model(model, inputs); del model
frames = {f.frame_id: f for f in inputs.pool.frames}
states = ul.y1_states(inputs.closure["exploration"]["locked_states"], list(frames.values()))
rows = []
for fid, frame in sorted(frames.items()):
    st = states[fid]
    d = {}
    for layer, xs, ref in ((4, st.x4_all, st.rows4), (5, st.x5_all, st.rows5)):
        a = atp.ReferenceRow(progs.readout.programs[layer], [x.double() for x in xs[: frame.p_t + 1]]).A_ref
        d[layer] = float((a - ref).abs().max())
    rows.append((fid, frame.template_id, d[4], d[5]))
for layer, i in ((4, 2), (5, 3)):
    v = sorted(r[i] for r in rows)
    print(f"L{layer}: median {v[len(v)//2]:.3e}  p90 {v[int(0.9*len(v))]:.3e}  max {v[-1]:.3e}  n<1e-5 {sum(x < 1e-5 for x in v)}")
by_t = {}
for r in rows:
    by_t.setdefault(r[1], []).append(max(r[2], r[3]))
print({t: (round(min(v), 8), round(sorted(v)[len(v)//2], 8), round(max(v), 8)) for t, v in by_t.items()})
print("worst frames:", sorted(rows, key=lambda r: -max(r[2], r[3]))[:5])
(HERE / "r9b_refrows_detail.json").write_text(json.dumps({"rows": rows, "guard": guard.EVENTS}, indent=1) + "\n")
