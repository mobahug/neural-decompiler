# Experiment 011 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 011 design (revision 2, commit `b5f59b9`): the zero-parameter, weight-defined encoding read `g_E` of fresh cues, committed before any fresh forward pass, scored against `L03.H04`'s measured transport of those cues in fresh frames (Y1), together with the prospective replication of the frozen P1 mechanism (Y2), with outcome-independent validity rules.

**Architecture:** One module `encoding_read.py` reusing `read_assembly.py` (63 × 24 pool, reference capture, the exact functional, `measure_token_010`, `fractions`), `head_transport.py` (head weights, P1, `TransportRule`, results-state and lock patterns, prediction artifact), `cue_suppression.py`, and `plural_mechanism.py`; a committed extract of Experiment 010's transport fractions; the confirmation builder with the frozen lists and frames; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`. The `lock` phase reads weights and the results state only and has no capture or intervention call path.

**Spec:** `docs/superpowers/specs/2026-09-19-experiment-011-encoding-read-prospective-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; seeds `20260916` / `20260924`. Nothing outside the 63 exposed tokens, 24 exposed frames, and 80 nouns is executed before `confirm`; the lock phase runs no forward pass at all.
- `E(w)` is `pm.lexicon_vector` (weight-only); `r(x) = ⟨x − mean(x), γ₃ ⊙ m⟩` with `m = head.read_direction(d̂_T)`; `g_E`, `g_∥`, `g_⊥` per token and template; denominators and their validity rule (`≥ 0.25 × max_T`, `≥ 0.25 σ_r`) fixed at `explore`.
- Exposed calibration: `q_T` for 63 × 24 replicated against the Experiment 010 extract (1e-6); `τ_g` from the 63 token-mean residuals `ḡ_E(w) − q̄_T(w)` over each token's informative exposed frames; `τ_M` from the informative pairs' `f_total − q_T`; both numeric in the lock.
- Validity at `confirm`: frame valid iff the plural cue's head change is informative (`≥ 0.25 σ_T`) and the frame's cue pair passes `exact_count_floor(108/120, 79)`; a token is scored iff it has ≥ 3 valid licensed frames; `PRECONDITION_FAILED` if < 4 valid frames or < 16 scored tokens; candidates are never filtered by their own `q_T`.
- Floors: Y1 Spearman ≥ 0.80 and MAE ≤ τ_g over scored tokens (`ḡ_E` vs `q̄_T` on the same frame set); Y2 Spearman ≥ 0.90 and MAE ≤ τ_M over (scored token, valid frame) pairs (P1 fraction vs `q_T`).
- Every constant is a named constant tested against the spec.

## File map

- `src/neural_decompiler/encoding_read.py`
- `experiments/011-encoding-read-prospective/run.py`, `confirmation-v1.json` (frozen), `inherited/experiment-010-transport-fractions.json`, `preregistration-lock.json` and `predictions.md` (installed by hand), `README.md`, `evidence/`
- `tests/test_encoding_read.py`, `tests/test_experiment_011_runner.py`
- `.gitignore`: `outputs/experiment-011/*`

## Shared definitions

- **Encoding read:** `inner(x)` = `ra.ReadFunctional(1.0, 1.0, ra.read_weight(head, d̂_T)).inner(x)`; `g_E(w, T) = inner(E(w) − E(ref_T)) / inner(E(pl_T) − E(ref_T))`; `g_∥` from the projection of `ΔE` on `d̂_E`; `g_⊥ = g_E − g_∥`.
- **Token means:** `ḡ_E(w) = mean_{f ∈ F(w)} g_E(w, T(f))`, `q̄_T(w) = mean_{f ∈ F(w)} q_T(w, f)` on the same `F(w)`.
- **P1 fraction:** `f_total = ρ_f(Δr_c) / ⟨ΔT(pl_T, f), d̂_T⟩` (Experiment 010's identity with the P1 projection, checked in every run).
- **Baseline:** Experiment 009's frozen rank-1 rule (`v` from the 009 lock, `β_T` from the 009 lock), predictions for the fresh tokens, reported only.
- **Confirmation set:** candidate lists and quotas of the spec (no expectations; classes recorded for reporting); `an` licensed in cue-final frames only; six literal frames; disjointness from the 63 tokens and the 80 noun forms; prompts round-trip.

---

### Task 1: Plan, extract, functional, confirmation builder

- [ ] Commit this plan; `.gitignore`; the Experiment 010 extract (`token|frame → q_T` or null, 63 × 24, with source digests); `encoding_read` (`inner`, `g_E`, denominators and validity), the confirmation builder/validator. Tests: `g_E` equals the ratio of inner reads and is frame-independent; the denominator rule; the builder's policy on the toy tokenizer; the committed extract.
- [ ] Commit `feat: add experiment 011 encoding read and confirmation builder`; freeze `confirmation-v1.json` with the pinned tokenizer; commit `feat: freeze experiment 011 confirmation set`.

### Task 2: Exploration, lock, confirmation, outcome, runner

- [ ] `run_exploration` (63 × 24 measurement through `ra`, replication, axes, `g_E` for the 63, τ_g, τ_M, descriptive statistics), `lock_predictions` (weights only), `build_candidate_lock`/`render_predictions`/`validate_lock`/`assert_lock_predictions_reproduced`, `run_confirmation` (validity, scoring, Y1, Y2, ledger, contrast), `outcome`, `render_report`, `run.py`. Tests: every floor and validity branch on synthetic tables; the state machine on the fake (freeze → explore → lock → install → confirm → report), refusals, incident path; a test that the lock phase performs no capture (`pm.capture_prompt`/`pm.run_patched` monkeypatched to raise).
- [ ] Commit `feat: add experiment 011 runner`.

### Task 3: Verification, Tier A, stop

- [ ] Full suite; independent review against the spec; fix; commit; push.
- [ ] `explore` once; `lock`; present the candidate lock and predictions. **Stop**: the user's lock commit and the reviewer's sign-off precede `confirm`.

### Task 4: Confirmation and closure

- [ ] `confirm` once; `report`; evidence; README; root README; memory.
