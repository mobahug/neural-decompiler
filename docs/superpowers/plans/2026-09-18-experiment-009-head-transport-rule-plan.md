# Experiment 009 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 009 design (revision 2, commit `fa62ab9`): the exact OV decomposition of `L03.H04`'s output change (P1/P2/P3/remainder), the component patches traced through the stages (non-additivity stage), the transport rule and the forty-token contrast rule with leave-one-cue-out gates, a tokenizer-only confirmation set under the frozen grammatical-compatibility policy, an immutable pre-confirm prediction artifact, and one confirmation run scored only against it.

**Architecture:** One module `head_transport.py` reusing `cue_suppression.py` (pool of 40 × 18, stage axes, fractions, replication checks), `supervised_subspace.py` (cross-moment estimator, LOCO, exports, incident conventions), `cue_decompilation.py` (confirmation-builder pattern, gates, floors), and `plural_mechanism.py`; the head's weights and the OV levels as pure tensor functions checked against the captured head result; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`.

**Spec:** `docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; seeds `20260916` / `20260922`. Nothing outside the forty exposed tokens, eighteen exposed frames, and eighty exposed nouns is executed before `confirm`; the ledger records every prompt and noun; `confirm` refuses if any fresh prompt appears earlier.
- The confirmation set is frozen by tokenizer rules under the frozen grammatical-compatibility policy (`an` only with ≥ 10 vowel-initial nouns, cue-final frames, vowel-initial nouns; numerals that stand alone; possessives; bare adjectives labelled as plural-forcing) and committed before `explore`; every token's licensed frames and noun keys are stored in it.
- The frozen identity `ΔT = Σ A^ref Δv W_O + Σ ΔA v^patch W_O` and the identity `ΔT₃ + remainder = ΔT` are checked in every patched run (1e-4 relative to the plural cue's `|ΔT|`); the reconstruction of the clean head result from LayerNorm-normalized values (`Σ_k A_k v_k W_O`) is checked against the captured result in every reference run. Failures are incidents.
- Rules are fitted on exposed data only; the lock and `predictions.md` are written by `lock` without running any fresh prompt; `confirm` reproduces every locked prediction from the on-disk rules before any fresh prompt (PhaseError otherwise) and scores only against the locked numbers.
- Every constant (`g_max 0.25`, mechanism `R² ≥ 0.90`, `MAE ≤ 0.10`, `τ_M/τ₂ = max(0.10, 3 × RMSE)`, Y1 Spearman 0.90, Y2 Spearman 0.80, category `≤ 0.35` / `≥ 0.65` / 80%, Y3 = Experiment 007's Y1 floors, cue effect 108/120 on 474 pairs) is a named constant tested against the spec.

## File map

- `src/neural_decompiler/head_transport.py`
- `experiments/009-head-transport-rule/run.py`, `confirmation-v1.json` (frozen), `preregistration-lock.json` and `predictions.md` (installed by hand), `README.md`, `evidence/`
- `tests/test_head_transport.py`, `tests/test_experiment_009_runner.py`
- `.gitignore`: `outputs/experiment-009/*`

## Shared definitions

- **Head weights**: `ln1` of block 3 (γ₃, β₃, ε), `W_V[4]`, `b_V[4]` (zero when absent), `W_O[4]`; `HeadWeights.from_model`.
- **Captures per patched run**: the 008 stage sites (`R0`, `R1`, `T`, `R2`, `M4`, `M5`, `R3`), `("RESID_PRE.L3", k)` for every key position `k ≤ p_t`, `("ATTN_PATTERN.L3", p_t)`, `("L03.H04", p_t)`; the reference run of each frame is captured once with the same sites.
- **OV levels**: as the spec, in float64; `q̂^{(j)} = ⟨ΔT_j, d̂_T⟩ / ⟨ΔT(pl_T, f), d̂_T⟩`; locked level = lowest `j` with exposed `R² ≥ 0.90` and `MAE ≤ 0.10` over the 720 pairs; `τ_M = max(0.10, 3 × RMSE)`.
- **Non-additivity stage** `ν*`: first `s ∈ (R1, T, R2, R3, c)` with `|g_s| > 0.25`, else `NONE`; mode over frames per token.
- **Transport rule**: `v ∝ Σ_i x_i y_i` (unit) over pooled rows (`x = ΔE_T(w)`, `y = q_T(w, f)`), `β_T` by no-intercept least squares per template; LOCO by token; gate Spearman ≥ 0.70 and normalized RMSE ≤ 0.50 over the forty cue-level means (`cd.quality_gate` with a single "rank"); `Ridge-scalar` baseline with Experiment 007's nested multiplier grid.
- **Contrast rule**: `ss.fit_supervised` at ranks 1–4 on `ss.design_rows` over the forty tokens with `ss.cue_level_values`; `cd.select_rank`, `cd.quality_gate`, `cd.tolerance_tau`; exported through `ss.export_program` and evaluated by the Experiment 007 program module; the 007 sixteen-token program (digest-checked) as the frozen baseline.
- **Confirmation measurements**: per fresh token × licensed frame — E-patch with the full captures (measured `q_T`, `Δc` over licensed nouns, OV levels), the ∥/⊥ patches (non-additivity stage), the behavioral prompt (`Δc_beh`); per fresh frame — the plural cue's E-patch (normalization) and the sg/pl cue prompts (precondition, 474 pairs).
- **Outcome**: precondition → Y1 (`HEAD_MECHANISM_CONFIRMED_P<j>` / `HEAD_MECHANISM_NOT_SUPPORTED` / `NOT_LOCKED`), Y2 (`TRANSPORT_RULE_PREDICTED` / `FAILED` / `NOT_LOCKED`), Y3 (`CONTRAST_RULE_PREDICTED` / `FAILED` / `NOT_LOCKED`).

---

### Task 1: Plan, skeleton, head weights, OV levels

- [ ] Commit this plan; `.gitignore`; constants; `HeadWeights`; `ov_levels` with the identity checks. Tests on the six-layer fake: the reconstruction equals the captured head result; `ΔT₃ + remainder = ΔT`; planted cases — a pure value change with fixed scale is explained by P1; a scale-only change by P2 but not P1; an attention-only change by P3 but not P2; a change at another position lands in the remainder.
- [ ] Commit `feat: add experiment 009 head weights and OV decomposition`.

### Task 2: Confirmation set under the grammatical policy

- [ ] Builder and validator (tokenizer-only; candidate lists; quotas; `an` rule; licensed frames and noun keys; fresh frames distinct from every exposed text; prompts round-trip; disjointness from the 40 tokens and 80 noun forms); freeze refuses to overwrite. Tests with the toy tokenizer extended by the candidates.
- [ ] Commit `feat: add experiment 009 confirmation builder`; then freeze `confirmation-v1.json` with the pinned tokenizer and commit `feat: freeze experiment 009 confirmation set`.

### Task 3: Measurements, fractions, rules, exploration

- [ ] `measure_009` (three patches per (token, frame) with captures; reference captures per frame), replication checks (006/007 extracts), fractions and `q_T`, additivity stages, OV level table and the locked-level rule, transport rule with LOCO and gate, ridge-scalar baseline, contrast rule with LOCO and gate, exports. Tests on the fake for shapes, identities, the level rule, the gates, and the exports.
- [ ] Commit `feat: add experiment 009 exploration`.

### Task 4: Lock, predictions artifact, confirmation, outcome, report, runner

- [ ] `build_candidate_lock` (+ `render_predictions` table with digest), `validate_lock` (state binding, digests, scientific paths with the 009 lock/README/evidence/predictions exempt), `assert_lock_predictions_reproduced`, `run_confirmation`, `outcome`, `render_report`, runner with six phases and incident bookkeeping. Tests: state machine on the fake (freeze → explore → lock → install → confirm → report), refusals, incident path.
- [ ] Commit `feat: add experiment 009 lock, confirmation, and runner`.

### Task 5: Verification, Tier A, stop

- [ ] Full suite; independent review against the spec; fix; commit; push.
- [ ] `explore` once; `lock`; present the candidate lock and `candidate-predictions.md`. **Stop**: installing and committing them is the user's act, and the reviewer's sign-off precedes `confirm`.

### Task 6: Confirmation and closure (after the lock commit and sign-off)

- [ ] `confirm` once; `report`; evidence; README; root README; memory; hand-off.
