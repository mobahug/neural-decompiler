# Experiment 016 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 016 design (revision 2, commit `498e604`): the frame-conditioned token-local Q/K model (Level 0-F) — Experiment 015's Level 0 with three literal reference-state channels (A: the frame's reference query/key vectors at the cue position as the self-logit operands; B: the frame's LayerNorm scale before the change and, derived algebraically from the predicted change, after it; C: block 1's MLP evaluated at the frame's own cue residual) — scored on unseen cues (Y1) and twelve unseen frames (Y2, with the per-frame anti-collapse guard), the scale-only alternative committed for rejection (Y3), the ablation ladder and the Level-0/Level-1 recovery identities descriptive/checked, and the no-fresh-forward-pass invariant tested.

**Architecture:** One module `frame_channels.py` reusing `attention_patterns.py` (`LayerProgram`, `ReferenceRow`, `reference_rows`, `check_reference_rows`, row JSON, the pooled statistics, `locked_state`, `measure_pair`, the comparator machinery, `compact_analysis`, `_max_numeric_difference`) and `attention_paths.py` (`capture_frame_013`, `head_identity`, `table_digest`, `assert_stage_one_digest`, `model_from_locks` for the decoded-`c_L` ladder), `layer_correction.py`, `neuron_feature.py` (pool chain); a `FrameChannelModel` with channel switches (`operands`, `scale`, `operating_point`, `remainder`) so that Level 0-F, the scale-only alternative, the three ablations, Experiment 015's Level 0 and the Level-1 recovery are one code path; a committed extract of Experiment 015's per-pair `c_ΔA`, sixteen self-weight changes and `c_L`, `c_M`, `c_H` (6180 pairs); the confirmation builder with the frozen lists and twelve frames; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`, `confirm` in two stages.

**Spec:** `docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; seeds `20260916` / `20260924`. Nothing outside the 183 exposed tokens, 54 exposed frames and 80 nouns is executed before `confirm`; the lock phase runs no forward pass; no fresh cue prompt before its frame's stage-1 table is digested.
- **Invariant:** no Level 0-F input is read from a fresh cue forward pass; `σ_f' = σ(x_ℓ + Δ̂x_ℓ)` from the predicted change; a test poisons the patched capture and checks the table is unchanged.
- **Level 0-F:** `n_Δ = γ ⊙ [(x̄ + Δ̂x)_c/σ_f' − x̄_c/σ_f]`; `Δq̃, Δk̃, Δv̂ = n_Δ W`; off-diagonal `⟨R_{p_c} Δq̃, k_ref(k)⟩/8`; diagonal `[⟨Δq̃, k̃_ref⟩ + ⟨q̃_ref, Δk̃⟩ + ⟨Δq̃, Δk̃⟩]/8` with the frame's own `q̃_ref, k̃_ref`; `Δ̂₁` at the frame's `x₁`; `Δ̂x₂ = ΔE + Δ̂₁ + Δ̂out₁`. Switches off → Experiment 015's Level 0 exactly; all on + remainder `γ ⊙ (x_ℓ − x̄)_c (1/σ_f' − 1/σ_f)` → Level 1 exactly.
- **Floors:** Y1/Y2 token means Spearman ≥ 0.90, R² ≥ 0.85; pooled entry R² ≥ 0.85 at layers 1 and 2; Y2 guard: every valid fresh frame's layer-2 entry R² ≥ 0.80; Y3 scale-only rejected iff its pooled layer-2 entry R² over both sets < 0.85 and ≥ 0.15 below Level 0-F's. Preconditions: 3 valid frames per token, 16 scored tokens, 8 valid fresh frames. Comparator rule (0.10) and ablations descriptive.
- Every constant is a named constant tested against the spec.

## File map

- `src/neural_decompiler/frame_channels.py`
- `experiments/016-frame-conditioned-qk/run.py`, `confirmation-v1.json` (frozen), `inherited/experiment-015-pair-extract.json`, `preregistration-lock.json` and `predictions.md` (installed by hand), `README.md`, `evidence/`
- `tests/test_frame_channels.py`, `tests/test_experiment_016_runner.py`
- `.gitignore`: `outputs/experiment-016/*`

---

### Task 1: Plan, extract, model, confirmation builder

- [ ] Commit this plan; `.gitignore`; the Experiment 015 extract; `frame_channels` (`Channels`, `FrameChannelModel`, analysis, pool 016, builder/validator). Tests: recoveries; the invariant with a poisoned capture; row/table shapes; builder policy; extract.
- [ ] Commit; freeze `confirmation-v1.json`; commit.

### Task 2: Exploration, lock, two-stage confirmation, outcome, runner

- [ ] `run_exploration`, `lock_predictions`, lock build/validate/reproduce, `stage_one`/`stage_two`, `score_confirmation` (floors, guard, Y3, ladder, per frame), `outcome`, `render_report`, `run.py`. Tests: synthetic tables for every floor branch and the guard; the stage barrier; the state machine on the fake; the lock phase performs no capture; incidents.
- [ ] Commit.

### Task 3: Verification, Tier A, stop

- [ ] Full suite; independent review against the spec; fix; commit; push.
- [ ] `explore` once; `lock`; present the candidate lock and predictions. **Stop**: the user's lock commit and the reviewer's sign-off precede `confirm`.

### Task 4: Confirmation and closure

- [ ] `confirm` once (two stages); `report`; evidence; README; root README; memory.
