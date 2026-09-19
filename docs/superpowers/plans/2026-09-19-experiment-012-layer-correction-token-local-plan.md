# Experiment 012 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 012 design (revision 2, commit `19183a7`): the zero-parameter token-local model of the layers-1–2 correction — the two MLPs of blocks 1 and 2 evaluated at the cue position on `ΔE` with attention held fixed and exposed template base states — committed for fresh cues and frames before any fresh forward pass, scored against the measured net layer change (Y1: Spearman, MAE, explained variance) and, as the composite in the head's units, against the measured P1 fraction (Y2), with leave-one-frame-out calibration of the tolerances, outcome-independent validity, the own-base ladder, and the block-2 neuron ledger.

**Architecture:** One module `layer_correction.py` reusing `read_assembly.py` (pool, reference capture, the exact functional, `measure_token_010` extended with extra capture sites, `fractions`, `neuron_concentration`), `encoding_read.py` (identity enforcement, `σ_r`, validity and state patterns), `head_transport.py`, `cue_suppression.py`, and `plural_mechanism.py`; committed extracts of Experiments 010's and 011's per-component fractions; the confirmation builder with the frozen lists and frames; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`. The `lock` phase reads weights, the Experiment 011 lock, and the results state (base states) and has no capture or intervention call path.

**Spec:** `docs/superpowers/specs/2026-09-19-experiment-012-layer-correction-token-local-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; seeds `20260916` / `20260924`. Nothing outside the 87 exposed tokens, 30 exposed frames, and 80 nouns is executed before `confirm`; the lock phase runs no forward pass at all.
- Axes, `σ_T`, read weight and the weight-only denominators are the Experiment 011 lock's values verbatim; `explore` and `confirm` recompute the stage axes on the Experiment 010 pool and refuse (incident) if they differ (cosine `1 − 1e-6`, `σ_T` `1e-9`).
- Units: Y1's measured `c_L`, `c_M`, `c_H`, `c_k` and predicted `ĉ` share the denominator `r(E(pl_T) − E(ref_T))`; `P1' = g_E + c_L` exactly; Y2's `q̂ = [r(ΔE) + r(Δ̂)] / D̂_T` and `P1 = r(Δr_c) / r(Δr_c(pl_T, f))`. Experiment 010's fractions are recorded beside them for replication and continuity.
- Exposed calibration: the 87 × 30 E-patches replicated against the two extracts (1e-6, every component); base states as template means; leave-one-frame-out predictions for every exposed pair; `τ_c`, `τ_P` from the leave-one-frame-out token-mean residuals; `R²_LOFO` and its 24-token subsample distribution recorded; the `R²` floor 0.50 is a named constant.
- Validity at `confirm` as Experiment 011 (plural cue's head change `≥ 0.25 σ_T`, cue pair `exact_count_floor(108/120, 79)`, ≥ 3 valid frames per token, ≥ 4 valid frames, ≥ 16 scored tokens; no candidate filtered by its own values).
- Floors: Y1 Spearman ≥ 0.80, MAE ≤ `τ_c`, `R² ≥ 0.50` over scored tokens on identical frame sets; Y2 Spearman ≥ 0.90, MAE ≤ `τ_P`. The own-base ladder, the heads' share, the `∥`/`⊥` evaluations, and the neuron ledger are descriptive and post-`confirm`.
- Every constant is a named constant tested against the spec.

## File map

- `src/neural_decompiler/layer_correction.py`; `src/neural_decompiler/read_assembly.py` (optional `extra_sites` on `measure_token_010`, backward compatible)
- `experiments/012-layer-correction-token-local/run.py`, `confirmation-v1.json` (frozen), `inherited/experiment-010-component-fractions.json`, `inherited/experiment-011-component-fractions.json`, `preregistration-lock.json` and `predictions.md` (installed by hand), `README.md`, `evidence/`
- `tests/test_layer_correction.py`, `tests/test_experiment_012_runner.py`
- `.gitignore`: `outputs/experiment-012/*`

## Shared definitions

- **Token-local model:** `Δ̂₁ = MLP₁(ln2₁(x̄₁ + ΔE)) − MLP₁(ln2₁(x̄₁))`, `Δ̂₂ = MLP₂(ln2₂(x̄₂ + ΔE + Δ̂₁)) − MLP₂(ln2₂(x̄₂))` in float64 from float32 weights; `ĉ = r(Δ̂₁ + Δ̂₂) / r(E(pl_T) − E(ref_T))`; parts, `∥`/`⊥` evaluations, interaction; `D̂_T`; `q̂`.
- **Base states:** the cue position's `RESID_PRE.L1` and `RESID_PRE.L2` in the reference prompt; template mean over the ten exposed frames (locked) or over the other nine (leave-one-frame-out, calibration).
- **Measured pair quantities:** from `measure_token_010` with `RESID_PRE.L1/L2` at `p_c` captured: `c_k = ρ_f(Δout_k) / (scale_f · r(E(pl_T) − E(ref_T)))`, `c_M`, `c_H`, `c_L`, `P1'`, `P1`, `q_T`, `g_E`; own-base evaluation, its exactness for block 1, the block-2 identity at the captured patched input, the attention-input term `c_M − ĉ_own`; block-1/2 neuron terms `Δa_j · r(W_out[j])` with the sum identity.
- **Token means:** over one `F(w)` per token for every quantity.
- **Confirmation set:** candidate lists and quotas of the spec (no expectations; classes recorded for reporting); every fresh cue licensed in every fresh frame; six literal frames; disjointness from the 87 tokens and the 80 noun forms; prompts round-trip.

---

### Task 1: Plan, extracts, model, confirmation builder

- [ ] Commit this plan; `.gitignore`; `measure_token_010(extra_sites=…)`; the two extracts (`token|frame → {f_k, f_E, f_total, f_layers, q_T}` or null, with source digests); `layer_correction` (`LayerWeights`, `propagate`, `CorrectionRead`, base states, pair measurement, neuron ledgers, the confirmation builder/validator, pool 012). Tests: block-1 exactness on the fake; neuron-sum identity; the builder's policy on the toy tokenizer; the committed extracts.
- [ ] Commit `feat: add experiment 012 token-local model and confirmation builder`; freeze `confirmation-v1.json` with the pinned tokenizer; commit `feat: freeze experiment 012 confirmation set`.

### Task 2: Exploration, lock, confirmation, outcome, runner

- [ ] `run_exploration` (87 × 30 through `ra`, replication, axes check against the 011 lock, base states, leave-one-frame-out predictions, `τ_c`, `τ_P`, `R²_LOFO` and subsamples, descriptive statistics, neuron summaries), `lock_predictions` (weights + locked axes + locked base states), `build_candidate_lock`/`render_predictions`/`validate_lock`/`assert_lock_predictions_reproduced`, `run_confirmation` (validity, pair measurements, ladder, scoring Y1/Y2, descriptive), `outcome`, `render_report`, `run.py`. Tests: every floor branch on synthetic tables (including the explained-variance floor alone); the state machine on the fake (freeze → explore → lock → install → confirm → report), refusals, incident path; the lock phase performs no capture.
- [ ] Commit `feat: add experiment 012 runner`.

### Task 3: Verification, Tier A, stop

- [ ] Full suite; independent review against the spec; fix; commit; push.
- [ ] `explore` once; `lock`; present the candidate lock and predictions. **Stop**: the user's lock commit and the reviewer's sign-off precede `confirm`.

### Task 4: Confirmation and closure

- [ ] `confirm` once; `report`; evidence; README; root README; memory.
