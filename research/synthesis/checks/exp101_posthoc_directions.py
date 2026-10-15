"""Post-hoc, exploratory, read-only diagnostics on Experiment 101's saved measurements (2026-10-14).

No model is loaded and no prompt runs. Reads the saved RESID_PRE.L3 x(p_c) rows of the calibration (36 exposed frames)
and of the confirmation (36 frames, spent since 101 closed) from the git-ignored outputs/experiment-101/, and:
- compares the weight-derived read direction ŵ (011's lock, centred) with naive data-derived directions: each frame's
  cue change Δx = x_two − x_one, the mean Δx, the first principal component of Δx, the class-mean ("probe") direction
  plural (two, three, many) minus singular (one), and a shrinkage-LDA direction;
- reports the natural range of the read coordinate r = ⟨x, w_c⟩ per cue and whether the D arms' targets (NEC-AMP:
  r(two) + Δr; SUF-NEG: r(one) − Δr) leave it.
These are descriptive and carry no outcome force; they inform the design of the next experiment.
Run: HF_HUB_OFFLINE=1 .venv/bin/python -B research/synthesis/checks/exp101_posthoc_directions.py
"""
import torch
from pathlib import Path

from neural_decompiler import read_intervention as ri
from neural_decompiler import upstream_localization as ul

torch.set_default_dtype(torch.float64)
geom = ri.read_geometry(ul.load_frozen_inputs(Path(".")).lock_011)
w = geom.w_c.double()
tags = list(ri.PRODUCTION.tags())
idx = {t: i for i, t in enumerate(tags)}


def rows(path):
    t = torch.load(path)
    frames = sorted({k.split("/")[0] for k in t})
    return [{c: t[f"{f}/x3_pc"].double()[idx[c]] for c in ("REF", "PL@two", "PL@three", "PL@many")} for f in frames]


def centred(v):
    return v - v.mean()


def cos(a, b):
    return float(centred(a) @ centred(b) / (centred(a).norm() * centred(b).norm()))


for name, path in (("calibration (exposed)", "outputs/experiment-101/calibration-measurements.pt"), ("confirmation (spent)", "outputs/experiment-101/stage2-measurements.pt")):
    data = rows(path)
    d = torch.stack([centred(r["PL@two"] - r["REF"]) for r in data])
    dbar = d.mean(0)
    per = sorted(cos(x, w) for x in d)
    _, S, V = torch.linalg.svd(d, full_matrices=False)
    sg = torch.stack([centred(r["REF"]) for r in data])
    pl = torch.stack([centred(r[c]) for r in data for c in ("PL@two", "PL@three", "PL@many")])
    probe = pl.mean(0) - sg.mean(0)
    allx = torch.cat([sg - sg.mean(0), pl - pl.mean(0)])
    cov = allx.T @ allx / allx.shape[0]
    lda = torch.linalg.solve(cov + float(torch.trace(cov) / cov.shape[0]) * torch.eye(cov.shape[0]), probe)
    r = {c: torch.tensor([float(x[c] @ w) for x in data]) for c in ("REF", "PL@two", "PL@three", "PL@many")}
    dr = r["PL@two"] - r["REF"]
    amp, neg = r["PL@two"] + dr, r["REF"] - dr
    top = max(float(r[c].max()) for c in ("PL@two", "PL@three", "PL@many"))
    print(f"== {name}: {len(data)} frames")
    print(f"  cos(Δx_frame, ŵ): min {per[0]:.3f} median {per[len(per) // 2]:.3f} max {per[-1]:.3f}")
    print(f"  cos(mean Δx, ŵ) {cos(dbar, w):.3f}; cos(PC1, ŵ) {abs(cos(V[0], w)):.3f} (PC1 share {float(S[0] ** 2 / (S ** 2).sum()):.3f}); "
          f"cos(class-mean probe, ŵ) {cos(probe, w):.3f}; cos(shrinkage LDA, ŵ) {cos(lda, w):.3f}")
    print(f"  natural r: " + "; ".join(f"{c} {float(v.min()):.2f}..{float(v.max()):.2f}" for c, v in r.items()))
    print(f"  NEC-AMP target beyond the max natural plural r in {int((amp > top).sum())}/{len(data)}; SUF-NEG target below the min natural r(one) in "
          f"{int((neg < float(r['REF'].min())).sum())}/{len(data)}")
