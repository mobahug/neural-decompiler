# Experiment 014 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 014 design (revision 2, commit `bea0203`): block-2 neuron 1987's activation change under the E-patch predicted for unseen cues and frames by the exact LayerNorm and the neuron's input weights evaluated at Experiment 013's frozen-pattern predicted arriving change (never a measured residual), with the first-order Jacobian form frozen for the accounting, the rigid number-axis alternative committed beside it, balanced firing accuracy with its guard, and the interpretation of the firing set offered only after scoring.

**Architecture:** One module `neuron_feature.py` reusing `attention_paths.py` (reference states, `HeadSet`, the frozen-pattern value paths, pair measurement with the residual before block 2 captured, `table_digest`/stage digest, `comparison`), `layer_correction.py` (`LayerWeights`, `analyse_pair` for the replication fields), `head_transport.py`, `read_assembly.py`, `encoding_read.py`, `cue_suppression.py`, `plural_mechanism.py`; a committed extract of Experiment 013's per-pair `c_L`, `c_M`, `c_H`, `c_k` (3732 pairs); the confirmation builder with the frozen lists, frames and two prompt lists; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`, `confirm` in two stages with the digested stage-1 table as the barrier.

**Spec:** `docs/superpowers/specs/2026-09-19-experiment-014-neuron-1987-feature-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; seeds `20260916` / `20260924`. Nothing outside the 135 exposed tokens, 42 exposed frames and 80 nouns is executed before `confirm`; the lock phase runs no forward pass; no fresh cue prompt before its frame's stage-1 table is digested.
- **Invariant:** the arriving change `Δ̂x₂` in every prediction is `ΔE + Δ̂₁ + Σ heads₁` from weights and the frame's reference state; the prediction function takes only the reference state and the token identity; a test disables every capture/intervention entry point while the prediction table is computed.
- Predictor: `Δp̂re = pre(x₂ + Δ̂x₂) − pre(x₂)` with `pre(x) = ⟨exact_layer_norm(x, γ₂, β₂, ε), W_in[:, 1987]⟩ + b_in[1987]`; `Δâ = GELU(pre_ref + Δp̂re) − GELU(pre_ref)`; firing `Δa ≥ 0.5`. Jacobian form `[⟨u, v_c⟩ − ⟨u, x̂⟩·mean(x̂ ⊙ v_c)]/σ` for the accounting. Axis-only: the same on `⟨Δ̂x₂, d̂_E⟩ d̂_E`.
- Floors: Y1/Y2 Spearman ≥ 0.80 and R² ≥ 0.50 on token means, balanced accuracy ≥ 0.90 on pairs (guard: a missing class → classification non-evaluable); Y3 rejected iff axis-only R² < 0.30 (token means, both sets) and balanced accuracy < 0.70 (pairs, both sets). MAE and amplitude error among firing pairs descriptive.
- Every constant is a named constant tested against the spec.

## File map

- `src/neural_decompiler/neuron_feature.py`
- `experiments/014-neuron-1987-feature/run.py`, `confirmation-v1.json` (frozen), `inherited/experiment-013-pair-ledger.json`, `preregistration-lock.json` and `predictions.md` (installed by hand), `README.md`, `evidence/`
- `tests/test_neuron_feature.py`, `tests/test_experiment_014_runner.py`
- `.gitignore`: `outputs/experiment-014/*`

## Shared definitions

- **Neuron weights:** `W_in[:, 1987]`, `b_in[1987]`, `W_out[1987]`, `γ₂`, `β₂`, `ε` from `LayerWeights`; `r(W_out[1987])` from the Experiment 011 read weight.
- **Reference state:** Experiment 013's `FrameState013` plus `pre_ref`; locked per exposed frame as (`x₁`, `x₂`, sixteen pattern weights, `pre_ref`).
- **Prediction row:** token, frame, template, `pre_ref`, `Δp̂re`, `Δâ`, `fîres`, `Δp̂re_J`, source split (`E`, `mlp1`, `heads1`), axis / off-axis split, `Δp̂re_axis`, `Δâ_axis`, `fîres_axis`, predicted `term`.
- **Measured pair:** `Δpre`, `Δa`, `fires`, `term`, the pattern-change remainder `Δpre − Δp̂re`, the Jacobian remainder, `c_L`, `c_M`, `c_H`, `c_k` (replication).
- **Confirmation set:** candidate lists and quotas of the spec; six literal frames; prompt list (a) fresh tokens × 42 exposed frames, (b) fresh tokens × 6 fresh frames.

---

### Task 1: Plan, extract, model, confirmation builder

- [ ] Commit this plan; `.gitignore`; the Experiment 013 pair-ledger extract; `neuron_feature` (`NeuronWeights`, arriving change, exact predictor, Jacobian form, axis alternative, pair measurement, pool 014, builder/validator). Tests: exactness against `layer_correction.neuron_ledger`; Jacobian first order; axis equality; the invariant; builder policy; extract.
- [ ] Commit `feat: add experiment 014 neuron-feature model and confirmation builder`; freeze `confirmation-v1.json`; commit `feat: freeze experiment 014 confirmation set`.

### Task 2: Exploration, lock, two-stage confirmation, outcome, runner

- [ ] `run_exploration`, `lock_predictions`, lock build/validate/reproduce, `stage_one`/`stage_two`, `score_confirmation` (balanced accuracy with guard), `outcome`, `render_report`, `run.py`. Tests: floors and the guard on synthetic tables; the stage barrier; the state machine on the fake; the lock phase performs no capture; incident paths.
- [ ] Commit `feat: add experiment 014 runner`.

### Task 3: Verification, Tier A, stop

- [ ] Full suite; independent review against the spec; fix; commit; push.
- [ ] `explore` once; `lock`; present the candidate lock and predictions. **Stop**: the user's lock commit and the reviewer's sign-off precede `confirm`.

### Task 4: Confirmation and closure

- [ ] `confirm` once (two stages); `report`; evidence; README; root README; memory.
