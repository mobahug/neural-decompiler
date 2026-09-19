# Experiment 015 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 015 design (revision 2, commit `4babcce`): the cue-induced change of the layer-1 and layer-2 attention patterns at the cue position predicted for unseen cues and frames by a token-local cue-change Q/K model conditional on the reference frame state (Level 0: query/key/value changes at the Experiment 012 template-mean bases from `ΔE` alone, applied to the frame's reference keys, values and logit rows), scored in pattern space (pooled entry R² per layer) and in read units (the pattern-change term `c_ΔA` of Experiment 013), with the exact own-state recomputation (Level 1) kept as an incident-guarded identity, the rigid number-axis alternative committed beside it, the diagonal-proportional comparators named with a predeclared interpretation rule, and Y2 frame-conditional and aggregate.

**Architecture:** One module `attention_patterns.py` reusing `attention_paths.py` (`capture_frame_013` — the reference residuals at every position `≤ p_c` and the pattern rows at `p_c`; `measure_pair_013`; `HeadSet`; `head_identity` for I3 and the measured pattern-change vectors; `model_from_locks` for the frozen-pattern-arrival rung; `table_digest`/`assert_stage_one_digest`), `layer_correction.py` (`LayerWeights`, `analyse_pair` for the replication fields, `comparison`, `explained_variance`, `relative_vector_error`), `head_transport.py`, `read_assembly.py`, `encoding_read.py`, `cue_suppression.py`, `plural_mechanism.py`; a frozen weight-only Q/K/V program class per layer with the partial rotary rotation; committed extracts of Experiment 014's per-pair `c_L`, `c_M`, `c_H`, `c_k` (4884 pairs) and Experiment 013's per-pair sixteen self-weight changes and pattern-change read (3732 pairs); the confirmation builder with the frozen lists, frames and two prompt lists; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`, `confirm` in two stages with the digested stage-1 table as the barrier.

**Spec:** `docs/superpowers/specs/2026-09-19-experiment-015-attention-pattern-token-local-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; seeds `20260916` / `20260924`. Nothing outside the 159 exposed tokens, 48 exposed frames and 80 nouns is executed before `confirm`; the lock phase runs no forward pass; no fresh cue prompt before its frame's stage-1 table is digested.
- **Invariant:** every Level-0 prediction is a function of the token identity, the locked bases, axes and read, the weights, and the frame's *reference* residuals at positions `≤ p_c`; the prediction function takes only those inputs; a test disables every capture/intervention entry point while the prediction table is computed.
- **Program (frozen):** `LN_ℓ` exact; `q̃ = n W_Q + b_Q`, `k̃ = n W_K + b_K`, `v = n W_V + b_V`; rotary on the first `rotary_dim` (16) coordinates, pairs `(i, i + rotary_dim/2)`, `θ_i(p) = p · base^(−2i/rotary_dim)`; scores `/√d_head`; softmax over `k ≤ p_c`; rows recomputed from the frame's residuals must reproduce the captured rows (I1, `1e-4` absolute); the exact chain must reproduce the captured residual before block 2 (I2, `1e-4` relative); Experiment 013's split (I3, `1e-4` relative). Identities are incidents when violated and count toward no floor.
- **Level 0:** `Δq̂, Δk̂, Δv̂` at `x̄_ℓ^T` (+ `Δ̂x_ℓ`); off-diagonal `Δŝ_k = ⟨R_{p_c} Δq̂, k_ref(k)⟩/√d_head`; diagonal `Δŝ_pc = [⟨q̃(x̄+Δ̂x), k̃(x̄+Δ̂x)⟩ − ⟨q̃(x̄), k̃(x̄)⟩]/√d_head` (rotation-free); `Â = softmax(s_ref + Δŝ)`; `Δ̂x₁ = ΔE`; `Δ̂x₂ = ΔE + Δ̂₁(x̄₁) + Δ̂out₁` (Level-0 rows and values); `P̂_h = Σ_k ΔÂ v̂ W_O`, `ĉ_ΔA = Σ_{16} r(P̂_h)/D_T`.
- **Axis-only (Y3):** own state, exact Level-1 change projected on `d̂_E` (lock 011 `R0`), no scale or intercept. **Comparators:** oracle and Level-0 diagonal-proportional (`a` self weight, other keys × `(1 − a)/(1 − A_ref(p_c,p_c))`); rule: interaction beyond the self logit only if Level-0 entry R² − Level-0 diagonal-proportional entry R² ≥ 0.10 at both layers on the fresh pairs of Y1 ∪ Y2.
- **Floors:** Y1/Y2 token means Spearman ≥ 0.80 and R² ≥ 0.50; pooled entry R² ≥ 0.50 at layer 1 and at layer 2; Y3 rejected iff axis-only token-mean R² < 0.30 and pooled entry R² over both layers and both sets < 0.30. MAE, total-variation ratios, self-weight R², per-layer `c_ΔA`, per-frame results descriptive.
- Every constant is a named constant tested against the spec.

## File map

- `src/neural_decompiler/attention_patterns.py`
- `experiments/015-attention-pattern-token-local/run.py`, `confirmation-v1.json` (frozen), `inherited/experiment-014-pair-ledger.json`, `inherited/experiment-013-pattern-extract.json`, `preregistration-lock.json` and `predictions.md` (installed by hand), `README.md`, `evidence/`
- `tests/test_attention_patterns.py`, `tests/test_experiment_015_runner.py`; `tests/plural_fakes.py` gains `rotary_dim=0` in the fake's `cfg` (the fake has no rotation)
- `.gitignore`: `outputs/experiment-015/*`

## Shared definitions

- **Layer program:** `LayerProgram.from_model(model, layer)` — `ln1` (`w`, `b`, `eps`), `W_Q/b_Q/W_K/b_K/W_V/b_V` stacked over heads, `W_O` per head, `d_head`, `rotary_dim` (from `cfg.rotary_dim`, else the HuggingFace `partial_rotary_factor × d_head`, else 0), `rotary_base`; methods `normalize`, `q_tilde`, `k_tilde`, `v`, `rotate(x, p)`.
- **Reference row:** `ReferenceRow(program, xs)` — from the residuals at positions `0..p_c`: rotated keys, values, `q_ref`, `scores_ref`, `A_ref`; `row_from_state(normed_pc)`, `row_from_changes(Δq̃, Δk̃, Δs_pc)`, `proportional(a)`, `pattern_change_vectors(A, v_pc)`, `output_change(A, v_pc)`.
- **Level-0 model:** `TokenLocalModel(read, lw, programs, bases_012, d_E)` — `predict(weights, x1_all, x2_all, token_id, template)` → predicted rows per head, self weights, `ĉ_ΔA` (per layer, total), axis rows and `ĉ_ΔA^axis`, Level-0 diagonal-proportional rows and `ĉ`, `r(Δ̂x₂)/D_T`; `predict_from_state`, `predict_from_locked`; also `level_one(...)` (exact rows, `Δx₂`) for I2, the axis alternative and the oracle comparator.
- **Locked state:** per exposed frame `p_c`, `x1_all`, `x2_all` (residuals before blocks 1 and 2 at positions `0..p_c`).
- **Prediction row:** token, frame, template, `p_c`, `rows_hat` (16 lists), `self_hat` (16), `c_hat_1`, `c_hat_2`, `c_hat`, `rows_axis`, `c_axis`, `rows_diag_L0`, `c_diag_L0`, `arrival_read`.
- **Measured pair:** `ΔA` rows (16 lists), self-weight changes, `c_ΔA` (per layer, total), I1–I3 errors, `c_L`, `c_M`, `c_H`, `c_k` (replication), the rungs' and comparators' per-pair sufficient statistics (n, Σm, Σm², Σ(p − m)², TV error, TV), diagonal terms, `‖Δx‖/‖x − μ‖`.
- **Confirmation set:** candidate lists and quotas of the spec; six literal frames; prompt list (a) fresh tokens × 48 exposed frames, (b) fresh tokens × 6 fresh frames.

---

### Task 1: Plan, extracts, program, model, confirmation builder

- [ ] Commit this plan; `.gitignore`; the two inherited extracts; the fake's `rotary_dim`; `attention_patterns` (`LayerProgram`, `ReferenceRow`, `TokenLocalModel`, Level 1, axis alternative, comparators, rungs, pair measurement and analysis, pool 015, builder/validator). Tests: I1 on the fake (rows reproduce `hook_pattern`); the same-position rotation lemma on a rotated program; I2 on the fake; Level 0 equals Level 1 when the base equals the own state; axis equality; the comparator equals Level 0 when off-diagonal changes vanish; the invariant; pooled statistics on synthetic rows; builder policy; extracts.
- [ ] Commit `feat: add experiment 015 attention-pattern program, token-local model and confirmation builder`; freeze `confirmation-v1.json`; commit `feat: freeze experiment 015 confirmation set`.

### Task 2: Exploration, lock, two-stage confirmation, outcome, runner

- [ ] `run_exploration` (states, replication of both extracts, I1–I3, exposed statistics), `lock_predictions`, lock build/validate/reproduce, `stage_one`/`stage_two`, `score_confirmation` (token means, pooled entry R² per layer, Y3, comparator rule, per-frame results), `outcome`, `render_report`, `run.py`. Tests: floors and the comparator rule on synthetic tables; the stage barrier; the state machine on the fake; the lock phase performs no capture; incident paths.
- [ ] Commit `feat: add experiment 015 runner`.

### Task 3: Verification, Tier A, stop

- [ ] Full suite; independent review against the spec; fix; commit; push.
- [ ] `explore` once; `lock`; present the candidate lock and predictions. **Stop**: the user's lock commit and the reviewer's sign-off precede `confirm`.

### Task 4: Confirmation and closure

- [ ] `confirm` once (two stages); `report`; evidence; README; root README; memory.
