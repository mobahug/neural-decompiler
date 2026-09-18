# Experiment 007 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute the approved Experiment 007 design (revision 3, commit `5449491`): a response-supervised, shared low-rank subspace of the layer-0 MLP cue encoding (`U_r` = leading left singular vectors of the uncentered cross-moment `XᵀY`), template-specific readouts `V_T` by a deterministic pseudoinverse, leave-one-cue-out rank selection and the unchanged pre-lock quality gate, three frozen baselines (`PCA-006(r)`, `E005-scalar`, template-specific nested `Ridge-full`), and — only if the gate passes — one confirmation run on the Experiment 006 confirmation set inherited byte-for-byte.

**Architecture:** One new module `supervised_subspace.py` reuses `cue_decompilation.py` (exposed pool, confirmation loading and validation, results state and phase isolation, E-patch residual responses, `select_rank`, `quality_gate`, `tolerance_tau`, P3 fidelity and circuit families with their floors, the confirmation measurements, Y floors, bands, `lock_predictions`) and `plural_mechanism.py` (frames, nouns, prompts, clean-run cache, exact replacement, contrasts, exact LayerNorm, weights, E005 program loading), and adds the estimators (cross-moment SVD with the singular-gap rule, pseudoinverse readout with the identifiability rule, PCA-006 refit through the same solver, nested ridge), the LOCO tables for every family, the comparison record, the inherited-response replication check, the 007 lock, the program-axis outcome, the confirmation orchestration, and the report. One weight-only module `linear_cue_program.py` evaluates any exported linear cue program (supervised, PCA-006, or ridge) from tensors alone. One runner with phases `validate`, `explore`, `calibrate`, `lock`, `confirm`, `report` (no freeze phase).

**Spec:** `docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md` (revision 3). The spec wins over this plan. (The spec's revision-history entry for revision 3 says "commit `f64a609` reviewed"; that names the revision-2 commit that was reviewed, and revision 3 itself is `5449491`.)

## Global constraints

- Pinned model, runtime, instrumentation path, contrast, exact final LayerNorm, reference cues, reference-relative coordinates, E-patch response, and cue-level statistics exactly as in Experiment 006. Runtime seed `20260916`; control seed `20260920`.
- The circuit is fixed (E = `L00.MLP` at `p_c`, T = `L03.H04`, R = {`L04.MLP`, `L05.MLP`} at `p_t`) and is never searched; its families are reported only.
- Exploratory pool = the twelve exposed frames, the sixteen exposed cue tokens, the sixty Experiment 005 nouns. Nothing else may be executed before `confirm`.
- The confirmation set is `experiments/006-low-rank-cue-decompilation/confirmation-v1.json`, read in place. Its content sha256 `dbb8dbcef9ab201cb5555a3c1d988e3ca8ab70f742f10b12f80a856c9ebf5521` is a named constant; `validate`, `explore`, `lock`, and `confirm` refuse if the file's digest differs. The file is never copied, rewritten, filtered, reordered, or selected from; the results state ledgers every executed prompt and noun and `confirm` refuses if any confirmation prompt, fresh noun, fresh frame, or fresh token appears before it.
- The recomputed exposed E-patch responses must match Experiment 006's recorded per-(token, frame) mean shifts to `1e-6`; the recorded values are committed as a derived extract under the Experiment 007 directory together with the digests of their source. A mismatch is an incident.
- Rank selection, the quality gate, τ, and every confirmation prediction are frozen in the lock before any confirmation prompt runs. `confirm` runs once. No override flags.
- Incidents (`IncidentError`): inherited-response mismatch, `rank(C) < 5`, `gap_r < 1e-8` in any contributing SVD, `rank(Z_T) < r` for the primary family in any fold or the final fit, digest mismatches, software defects. They stop the phase, are recorded in the results state, and are never an outcome label; resolution follows the spec's incident rule.
- The program module imports neither TransformerLens, Transformers, nor the instrumentation modules; the runner's E-patch replacement tensor is asserted equal to the program's weight-only `E(w)` within `1e-5` (inherited check).
- Every floor, count, tolerance, and rule is a named constant tested against the spec's numbers.

## File map

- `src/neural_decompiler/supervised_subspace.py`
- `experiments/007-supervised-cue-subspace/linear_cue_program.py`
- `experiments/007-supervised-cue-subspace/run.py`
- `experiments/007-supervised-cue-subspace/inherited/experiment-006-epatch-means.json` (derived extract; committed before `explore`)
- `experiments/007-supervised-cue-subspace/preregistration-lock.json` (installed by hand after `lock`)
- `experiments/007-supervised-cue-subspace/README.md`, `evidence/`
- `tests/test_supervised_subspace.py`, `tests/test_linear_cue_program.py`, `tests/test_experiment_007_runner.py`
- `.gitignore`: `outputs/experiment-007/*`

## Shared definitions

- **Rows.** For token `w` (pool order) and exposed frame `f` (pool order): `x(w, f) = ΔE_T(w) = E(w) − E(ref_T)` and `y(w, f) = Δr_Epatch(w, f)`, both float64; `X, Y ∈ ℝ^{192 × 512}` on the full pool, fewer rows inside folds. Never centered; no intercept anywhere.
- **Cross-moment SVD.** `C = XᵀY`; `P, σ, Q = svd(C)` in float64; `U_r = P[:, :r]`; each column's sign is flipped so that its largest-magnitude entry is positive. `rank(C) = #{σ_i > max(C.shape) × eps(float64) × σ_1}`; `gap_r = (σ_r − σ_{r+1})/σ_1` for `r = 1..4`. Require `rank(C) ≥ 5` and every `gap_r ≥ 1e-8` in every outer fold and in the final fit; record every gap. Ranks are nested by construction (`U_r = U_4[:, :r]`).
- **Readout solve.** `Z_T = X_T U_r`; `V_Tᵀ = pinv(Z_T, rcond) Y_T` with `rcond = max(Z_T.shape) × eps(float64)`; `rank(Z_T) = #{singular values > rcond × σ_max(Z_T)}`, recorded per template per fold and in the final fit; `rank(Z_T) < r` is an incident for the primary family and a reported flag for `PCA-006(r)`.
- **Prediction.** `δ̂(w, f) = V_T U_rᵀ ΔE_T(w)` (primary and PCA-006) or `B_T ΔE_T(w)` (ridge); `Δĉ_N(w, f) = −u_N · [LN(ρ_f + δ̂) − LN(ρ_f)]` with the exact final LayerNorm in float64, `ρ_f` the clean reference residual of an exposed frame and `ρ_template` (mean over the template's exposed frames) for a fresh frame. Zero at the reference by construction.
- **Cue-level values** of a token: mean predicted, mean measured, and mean absolute error of the E-patch shift over the twelve exposed frames and the sixty nouns.
- **PCA-006(r).** `cd.pca_basis` on the fit tokens' `E` vectors (centered PCA, as in Experiment 006), reference-relative `Δz`, readout by the same pseudoinverse solve.
- **Ridge-full.** One `B_T` per template, `B_Tᵀ = (X_TᵀX_T + λI)⁻¹ X_TᵀY_T` (float64 solve). Per outer fold: inner leave-one-cue-out over the fifteen training tokens; per inner fold, grid `{10⁻², 10⁻¹, 1, 10, 10²} × tr(X_innerᵀX_inner)/d_model` from the fourteen inner-training tokens' pooled rows; score each multiplier by the mean of the fifteen inner held-out tokens' cue-level MAE; select the smallest score, ties to the larger multiplier; refit the three `B_T` on the fifteen outer-training tokens at `multiplier × tr(X_outerᵀX_outer)/d_model`; predict the outer held-out token. Record the multiplier per outer fold. The lock-time `Ridge-full` runs the same selection as a leave-one-cue-out over all sixteen tokens and refits on all sixteen.
- **Rank rule, quality gate, τ:** `cd.select_rank`, `cd.quality_gate`, `cd.tolerance_tau`, unchanged.
- **Comparison record (not a gate):** per rank, supervised versus `PCA-006` LOCO error and whether supervised ≤ 0.8 × PCA (the stated prediction); `Ridge-full` LOCO error and its ratio to the selected rank's error; `E005-scalar` cue-level error.
- **Outcome (program axis):** `QUALITY_GATE_FAILED` at explore if the gate fails (no lock); after confirmation: `CUE_EFFECT_NOT_REPLICATED` if the fresh-frame cue-effect gate (`d_full > 0` in ≥ 108/120) fails; `PROGRAM_NOT_SUPPORTED` if a Y floor fails; `DECOMPILED` if Y1–Y3 pass and the bands are hit; `DECOMPILED_MISCALIBRATED` otherwise. Circuit families on fresh nouns × manifest frames and on fresh frames are reported with the unchanged 006 floors and flagged "C002 review eligible" only if both pass.
- **Programs in the lock:** `selected`, `pca-006` (at the selected rank), `e005-scalar`, `ridge-full`; every one fitted on all sixteen exposed tokens; Y4 reports their Y1/Y2 statistics.

---

### Task 1: Plan, inherited extract, skeleton

- [ ] Commit this plan.
- [ ] Write `experiments/007-supervised-cue-subspace/inherited/experiment-006-epatch-means.json` from the local Experiment 006 results state: the 192 `token|frame → mean_shift` values with `delta_norm` and `circuit_share`, plus the source file sha256, the recorded `state_sha256`, `run_id`, and protocol commit. Tests: the extract validates, has 192 keys covering every exposed (token, frame), and its recorded confirmation digest equals the named constant.
- [ ] Module skeleton with constants (`INHERITED_CONFIRMATION_SHA256`, `INHERITED_EPATCH_TOLERANCE = 1e-6`, `GAP_MIN = 1e-8`, `MIN_CROSS_MOMENT_RANK = 5`, `RIDGE_MULTIPLIERS`, seeds, program names, scientific path prefixes); `.gitignore` entry.
- [ ] Commit `feat: add experiment 007 skeleton and inherited response extract`.

### Task 2: Estimators as pure tensor functions

- [ ] Tests: cross-moment SVD recovers a planted supervised subspace (`Y = X U Vᵀ + 0` for a random orthonormal `U`, span match), orthonormality, nestedness `U_r == U_4[:, :r]`, sign convention (largest-magnitude entry positive), determinism across two calls, gap computation on a planted tie raises `IncidentError`, `rank(C) < 5` raises; pseudoinverse solve equals the normal-equation solution at full rank, recovers a planted `V_T`, records `rank(Z_T)`, and raises on a planted rank-deficient template for the primary while only flagging it for PCA; ridge closed form matches a direct solve; the λ grid is computed from the inner-training rows only (leakage test: perturbing the held-out rows does not change the grid); one `B_T` per template; tie-breaking to the larger multiplier; zero prediction at the reference for every family.
- [ ] Implement `design_rows`, `cross_moment_basis`, `check_singular_gaps`, `pinv_readout`, `fit_supervised`, `fit_pca_006`, `ridge_maps`, `select_ridge_multiplier`, `fit_ridge_full`, `LinearFit`, `predict_delta`, `predict_shifts`, `cue_level_values`.
- [ ] Commit `feat: add supervised cross-moment subspace and ridge estimators`.

### Task 3: LOCO tables, selection, comparison, inherited check

- [ ] Tests (fake model): the outer folds are by whole token; every rank's fold records gaps and `rank(Z_T)`; the ridge table records a multiplier per fold; `select_rank`/`quality_gate`/`tolerance_tau` are the Experiment 006 functions; the comparison record's prediction flag; the inherited-response check passes on identical values and raises on a `2e-6` deviation.
- [ ] Implement `loco_tables`, `comparison_record`, `check_inherited_responses`, `load_inherited_extract`.
- [ ] Commit `feat: add experiment 007 leave-one-cue-out tables`.

### Task 4: Program module, exports, lock, outcome, confirmation, report

- [ ] Tests: `linear_cue_program.py` loads only exported tensors, refuses a digest mismatch, computes `E(w)` equal to the runner's lexicon vector, predicts zero at the reference, matches the in-process fit's predictions for all three kinds, and imports none of the forbidden modules (subprocess check); outcome labels for every branch; lock digest, program names, inherited digests, and validation refusals; report rendering with and without a confirmation.
- [ ] Implement `export_program`, `load_programs`, `build_candidate_lock`, `validate_lock`, `outcome`, `run_confirmation`, `render_report`.
- [ ] Commit `feat: add experiment 007 program, lock, and outcome`.

### Task 5: Runner and state machine

- [ ] Tests (fake model, forced floors where necessary): parser has exactly the six phases and no override flag; `validate` refuses a confirmation digest mismatch; `explore` runs once, never touches fresh items, records the inherited check, and sets `QUALITY_GATE_FAILED` when the gate fails; `lock` refuses without the gate; `confirm` refuses without a committed valid lock, refuses twice, and validates digests, unchanged scientific paths, and the ledger; the full state machine reaches `report`.
- [ ] Implement `run.py` (`Runner` with injectable loaders as in Experiment 006, `results_006_path` optional and informational).
- [ ] Commit `feat: add experiment 007 runner`.

### Task 6: Verification, then Tier A

- [ ] Full offline suite. Independent review of the implementation against the spec (revision 3); fix findings; commit; push.
- [ ] `explore` once (A0 contract test inside). If the gate fails: `report`, evidence copy, README closure at `QUALITY_GATE_FAILED`, root README, commit. If it passes: `calibrate`, `lock`; present the candidate lock. **Stop**: installing and committing the lock is the user's act.

### Task 7: Confirmation and claims (after the lock commit)

- [ ] `confirm` once; `report`; evidence copies; C002 review only if the circuit families pass on both sets; READMEs; memory.
