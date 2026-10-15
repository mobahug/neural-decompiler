"""Post-hoc, exploratory, read-only check of what in Experiment 101's result follows from how w_c is built (2026-10-14).

The model's weights are loaded; no forward pass runs and no prompt is evaluated. The saved RESID_PRE.L3 x(p_c) rows of
101's calibration (36 exposed frames) and confirmation (36 frames, spent since 101 closed) are read from the git-ignored
outputs/experiment-101/. Notation (the 102 baselines survey, scratchpad memo of 2026-10-14):
- W = W_V W_O of L03.H04 (d_model x d_model), Γ = diag(γ₃), C = I − 𝟙𝟙ᵀ/d the centring projector;
- under fixed attention A and LayerNorm scale σ, an edit e at p_c changes the head's output by (A/σ)·Wᵀ Γ C e, so its
  component along d̂_T is (A/σ)·⟨e, w_c⟩ with w_c = C Γ W d̂_T (identity (1));
- an edit along w_c writes the head's output along M d̂_T with M = Wᵀ Γ C Γ W; if d̂_T is not an eigenvector of M,
  that output has an off-axis part (I − d̂d̂ᵀ) M d̂_T;
- to first order the read covector is K δ with K = C Γ W Wᵀ Γ C and δ the mean cue change (identity (2)).
Checks:
- G1-style sanity: C Γ W d̂_T from the weights equals 101's w_c (validates the conventions used here);
- cos(M d̂_T, d̂_T) and the off-axis share of M d̂_T, and d̂_T's Rayleigh quotient against M's top eigenvalue;
- cos(w_c, K δ) with δ the mean centred one→two change, and per frame cos(w_c, K Δx_f);
- cos(ŵ, d̂_T): how far the read covector is from the head's write axis (added 2026-10-15 for 102's position control).
Descriptive only; no outcome force. Run:
HF_HUB_OFFLINE=1 .venv/bin/python -B research/synthesis/checks/exp101_posthoc_construction.py
"""
import torch
from pathlib import Path

from neural_decompiler import head_transport as ht
from neural_decompiler import models
from neural_decompiler import read_intervention as ri
from neural_decompiler import upstream_localization as ul

torch.set_default_dtype(torch.float64)
geom = ri.read_geometry(ul.load_frozen_inputs(Path(".")).lock_011)
w_c, d_T = geom.w_c.double(), geom.d_T.double()
d_hat = d_T / d_T.norm()

model = models.load_model(models.PYTHIA_70M)  # weights only: the model is never called
head = ht.HeadWeights.from_model(model, 3, 4)
W = head.W_V @ head.W_O  # d_model x d_model
d = W.shape[0]
C = torch.eye(d) - torch.full((d, d), 1.0 / d)
G = torch.diag(head.ln_w)

w_from_weights = C @ G @ W @ d_hat
print(f"sanity: max |CΓW d̂_T − w_c| = {float((w_from_weights - w_c).abs().max()):.3e} (‖w_c‖ {float(w_c.norm()):.4f})")

M = W.T @ G @ C @ G @ W
Md = M @ d_hat
cos_md = float(Md @ d_hat / Md.norm())
off = Md - (Md @ d_hat) * d_hat
evals = torch.linalg.eigvalsh(M)
print(f"cos(M d̂_T, d̂_T) = {cos_md:.4f}; off-axis share ‖(I − d̂d̂ᵀ)M d̂‖/‖M d̂‖ = {float(off.norm() / Md.norm()):.4f}")
print(f"Rayleigh d̂ᵀM d̂ = {float(d_hat @ Md):.4f}; top eigenvalues of M: " + ", ".join(f"{float(v):.4f}" for v in evals.flip(0)[:5]))
print(f"cos(ŵ, d̂_T) = {float(w_c @ d_hat / w_c.norm()):.4f} (read covector at p_c against the head's write axis at p_t; added 2026-10-15)")

K = C @ G @ W @ W.T @ G @ C
tags = list(ri.PRODUCTION.tags())
idx = {t: i for i, t in enumerate(tags)}


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


for name, path in (("calibration (exposed)", "outputs/experiment-101/calibration-measurements.pt"), ("confirmation (spent)", "outputs/experiment-101/stage2-measurements.pt")):
    t = torch.load(path)
    frames = sorted({k.split("/")[0] for k in t})
    dx = torch.stack([C @ (t[f"{f}/x3_pc"].double()[idx["PL@two"]] - t[f"{f}/x3_pc"].double()[idx["REF"]]) for f in frames])
    delta = dx.mean(0)
    per = sorted(cos(K @ x, w_c) for x in dx)
    print(f"== {name}: {len(frames)} frames")
    print(f"  cos(w_c, K δ) = {cos(K @ delta, w_c):.4f}; cos(δ, ŵ) = {cos(delta, w_c):.4f}")
    print(f"  per frame cos(w_c, K Δx_f): min {per[0]:.4f} median {per[len(per) // 2]:.4f} max {per[-1]:.4f}")
