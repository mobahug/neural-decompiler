# Experiment 013 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 013 design (revision 2, commit `617f549`): the frozen Experiment 012 model taken as given, its residual predicted by the omitted computation — the frame's own base state and the layer-1–2 heads' frozen-pattern value path feeding block 2's MLP and writing directly — tested strictly on new tokens in the 36 exposed frames (Y1), frame-conditionally on new tokens in six new frames with a digested stage-1 prediction table (Y2), and on the MLP/heads attribution at the pair level (Y3), with the per-head OV identity enforced, frozen tolerances `τ_r = 0.070`, `τ_A = 0.106`, and the degenerate-spread guard.

**Architecture:** One module `attention_paths.py` reusing `layer_correction.py` (`LayerWeights`, `CorrectionRead`, `propagate`, the 012 lock's base states and read, `measure_token_010`-based pair measurement and `analyse_pair`, state/lock/scoring patterns), `head_transport.py` (`HeadWeights` for the sixteen heads), `read_assembly.py`, `encoding_read.py`, `cue_suppression.py`, `plural_mechanism.py`; a committed extract of Experiment 012's per-pair ledger; the confirmation builder with the frozen lists, frames and both prompt lists; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`, `confirm` in two stages separated by the digested prediction table. The `lock` phase reads weights, the 011 and 012 locks and the results state (locked reference states) and has no capture or intervention call path.

**Spec:** `docs/superpowers/specs/2026-09-19-experiment-013-attention-paths-frozen-pattern-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; seeds `20260916` / `20260924`. Nothing outside the 111 exposed tokens, 36 exposed frames, and 80 nouns is executed before `confirm`; the lock phase runs no forward pass; stage 2 of `confirm` runs no fresh cue prompt before the stage-1 table's digest is in the results state.
- The 012 model is applied verbatim (its lock's base states, read weight, axes; `ĉ_012` reproduced against the 012 extract within `1e-9`). Units: Experiment 012's Y1 units (weight-only denominator).
- Per-head OV identity (`Δout_h = A^ref Δv W_O + Σ ΔA v^patch W_O`) enforced in every measured pair (relative `1e-4`); Experiment 012's identities inherited.
- Tolerances are constants: `TAU_R = 0.070`, `TAU_A = 0.106`, `C_H_DEGENERATE_SD = 0.017`; the exposed RMSEs (design 0.0233 / 0.0355) are recomputed for the record and must agree within 0.001.
- Aggregation: Y1/Y2 token means; Y3 pairs; pair-level Y1/Y2 descriptive.
- Every constant is a named constant tested against the spec.

## File map

- `src/neural_decompiler/attention_paths.py`
- `experiments/013-attention-paths-frozen-pattern/run.py`, `confirmation-v1.json` (frozen), `inherited/experiment-012-pair-ledger.json`, `preregistration-lock.json` and `predictions.md` (installed by hand), `README.md`, `evidence/`
- `tests/test_attention_paths.py`, `tests/test_experiment_013_runner.py`
- `.gitignore`: `outputs/experiment-013/*`

## Shared definitions

- **Reference state of a frame:** `x₁(k)`, `x₂(k)` for `k ≤ p_c`, the pattern rows `A_h^{(1)}(p_c, ·)`, `A_h^{(2)}(p_c, ·)`; locked for the 36 exposed frames as (`x₁(p_c)`, `x₂(p_c)`, the sixteen `A_h(p_c, p_c)`), the full state re-captured where the identity needs it.
- **Frozen-pattern prediction:** `Δ̂₁`, heads of layer 1 on `ΔE`, `Δ_2 = ΔE + Δ̂₁ + Σ heads₁`, `Δ̂₂`, heads of layer 2 on `Δ_2`; `ĉ_M`, `ĉ_H`, `ĉ_L`, `r̂ = ĉ_L − ĉ_012`, base-point `ĉ_own − ĉ_012`, frozen-attention `ĉ_L − ĉ_own`.
- **Identity per head:** frozen-pattern term with the *measured* arriving change plus the pattern-change term from the captured patched pattern row and values equals the captured head output change.
- **Prediction table row:** token, frame, template, `ĉ_012`, base-point, frozen-attention, `r̂`, `ĉ_M`, `ĉ_H`, `ĉ_L`, sixteen head terms.
- **Confirmation set:** candidate lists and quotas of the spec; six literal frames; prompt list (a) fresh tokens × 36 exposed frames, (b) fresh tokens × 6 fresh frames; disjointness; round-trip.

---

### Task 1: Plan, extract, model, confirmation builder

- [ ] Commit this plan; `.gitignore`; the Experiment 012 pair-ledger extract (2724 pairs: `c_L`, `c_M`, `c_H`, `c_k`, `ĉ_own`, `ĉ_012`, `P1`, `q_T`, `g_E`); `attention_paths` (head set, frame state capture, frozen-pattern prediction, identity split, pair analysis, pool 013, the confirmation builder/validator). Tests: identity on the fake; prediction reduces to the 012 own-base value when the pattern weights are zero; builder policy; committed extract.
- [ ] Commit `feat: add experiment 013 frozen-pattern attention model and confirmation builder`; freeze `confirmation-v1.json`; commit `feat: freeze experiment 013 confirmation set`.

### Task 2: Exploration, lock, two-stage confirmation, outcome, runner

- [ ] `run_exploration` (36 reference states, 2724 re-measurements, replication, `ĉ_012` reproduction, identities, ladder, calibration RMSEs against the frozen values, descriptive statistics, compensation cases, `thy`), `lock_predictions` (fresh tokens × 36 exposed frames from locked states), `build_candidate_lock`/`render_predictions`/`validate_lock`/`assert_lock_predictions_reproduced`, `stage_one` (fresh frames' reference states, validity, prediction table, digest) and `stage_two` (fresh cues in both sets, identities, ladder, scoring), `score_confirmation` (Y1 token means strict, Y2 token means conditional, Y3 pairs with the guard), `outcome`, `render_report`, `run.py`. Tests: every floor branch and the guard on synthetic tables; the stage barrier (stage 2 refuses without the digest; the table is read, not recomputed); the state machine on the fake; the lock phase performs no capture; incident paths.
- [ ] Commit `feat: add experiment 013 runner`.

### Task 3: Verification, Tier A, stop

- [ ] Full suite; independent review against the spec; fix; commit; push.
- [ ] `explore` once; `lock`; present the candidate lock and predictions. **Stop**: the user's lock commit and the reviewer's sign-off precede `confirm`.

### Task 4: Confirmation and closure

- [ ] `confirm` once (two stages); `report`; evidence; README; root README; memory.
