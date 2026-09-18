# Experiment 006 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute the approved Experiment 006 design (revision 4, commit `21a33d2`): a low-rank, weight-only decompilation of the layer-0 MLP cue encoding, fitted to the measured E-patch residual response on the exposed Experiment 005 data, with rank selection and a quality gate entirely before the lock, and one confirmation run on a fresh, tokenizer-only, category-balanced set of nouns, frames, and cue tokens.

**Architecture:** One module `cue_decompilation.py` reuses the Experiment 005 machinery (`plural_mechanism.py`: frames, nouns, prompts, clean-run cache, exact replacement, contrasts, recovery, isolation, chain, neutralization, exact LayerNorm, weight export, lock validation helpers) and adds the confirmation-set builder, the exposed pool, the E-patch residual-response measurement, low-rank fitting with leave-one-cue-out selection, the split-invariant circuit families, the Y families, the lock, the outcome rule, and the report. One weight-only module `low_rank_program.py` evaluates the program from exported tensors. One runner with phases `validate`, `freeze-confirmation`, `explore`, `calibrate`, `lock`, `confirm`, `report`.

**Spec:** `docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md` (revision 4). The spec wins over this plan.

## Global constraints

- Pinned model, runtime, instrumentation path, and all shared definitions inherited from Experiment 005. Runtime seed `20260916`; control seed `20260919`.
- The circuit is fixed: E = `L00.MLP` at `p_c`, T = `L03.H04` at `p_t`, R = {`L04.MLP`, `L05.MLP`} at `p_t`. No component search of any kind.
- Exploratory pool = the twelve manifest frames' and six extension frames' prompts with the four original cues, the 72 Experiment 005 cue-word prompts, the sixteen exposed cue tokens, and the sixty Experiment 005 nouns. E-patch interventions on exposed reference prompts with any exposed token's weight-only `E(w)` are within the pool.
- The confirmation set (`confirmation-v1.json`) is built by tokenizer rules only, committed before `explore`, and never executed before `confirm`; the results state ledgers every executed prompt and scored noun, and `confirm` refuses if any confirmation prompt or fresh noun appears before it.
- Rank selection, the quality gate, τ, and every confirmation prediction are computed and frozen in the lock before any confirmation prompt runs. `confirm` runs once. No override flags.
- The program module imports neither TransformerLens, Transformers, nor the instrumentation modules; the E-patch replacement tensor used by the runner is asserted equal to the program's weight-only `E(w)` within `1e-5`.
- Every floor, count, and rule is copied from the spec into named constants and tested against the spec's numbers.

## File map

- `src/neural_decompiler/cue_decompilation.py`
- `experiments/006-low-rank-cue-decompilation/low_rank_program.py`
- `experiments/006-low-rank-cue-decompilation/run.py`
- `experiments/006-low-rank-cue-decompilation/confirmation-v1.json`
- `experiments/006-low-rank-cue-decompilation/preregistration-lock.json` (installed by hand)
- `experiments/006-low-rank-cue-decompilation/README.md`, `evidence/`
- `tests/test_cue_decompilation.py`, `tests/test_low_rank_program.py`, `tests/test_experiment_006_runner.py`
- `.gitignore`: `outputs/experiment-006/*`

## Shared definitions

- **Exposed frames:** the six manifest frames and the six Experiment 005 extension frames (`origin` recorded). **Fresh frames:** six new frames from the confirmation set.
- **Reference cue** `ref_T`: `one` (cardinal, coordinated-adjective), `each` (quantifier); reference prompt of a frame = the frame with `ref_T` in the cue slot.
- **E-patch residual response** `Δr_Epatch(w, f)`: the final pre-LayerNorm residual (`RESID_POST.L5` at `p_t`) of frame `f`'s reference prompt with `L00.MLP` at `p_c` replaced by the weight-only `E(w)`, minus the clean reference residual. Measured in one intervention forward per (token, frame).
- **E-patch shift** for noun `N`: the contrast of that patched run minus the contrast of the clean reference run.
- **Cue-level value** of a token: the mean over frames and nouns of its E-patch shift (measured) or predicted shift (program).
- **Program coordinates:** `z(w) = U_rᵀ(E(w) − μ_E)`; `Δz_T(w) = z(w) − z(ref_T)`; `V_T` fitted by least squares without intercept to `Δr_Epatch(w, f) ≈ V_T Δz_T(w)` over exposed tokens and frames of `T`; `Δĉ_N(w, f) = −u_N · [LN(ρ_f + V_T Δz_T(w)) − LN(ρ_f)]`.
- **Rank rule:** `error_r`, `SE_r` over the sixteen held-out tokens; `eligible = {1} ∪ {r > 1 : error_r ≤ 0.8 × error_1}`; `r_best = argmin_{eligible} error_r`; `threshold = error_{r_best} + SE_{r_best}`; `selected = min {r ∈ eligible : error_r ≤ threshold}`.
- **Quality gate (pre-lock):** 20% rule when `r > 1`; Spearman ≥ 0.70 over the sixteen cue-level pairs; normalized LOCO RMSE ≤ 0.50.
- **τ** = max(0.5, 3 × RMSE of the sixteen cue-level errors of the selected rank).
- **P3 fidelity:** over 120 (frame, noun) pairs, `d_clean = c_clean(sg cue) − c_clean(pl cue)`, `d_iso` likewise under isolation of the fixed set; `F ≥ 0.50` overall and `≥ 0.40` per template; `sign(d_iso) = sign(d_clean)` in ≥ 114/120; Pearson `corr(d_iso, d_clean) ≥ 0.90`.
- **Cue-effect gate:** `d_full > 0` in ≥ 114/120 pairs on manifest frames and ≥ 108/120 on fresh frames.

---

### Task 1: Confirmation set, exposed pool, results state, skeleton

- [ ] Tests: quotas (10/5/5 nouns; 8 inherited + 4 × 4 tokens), disjointness from the sixty nouns and sixteen tokens, single-token rules, fresh frames' positions, digest, refusal to overwrite, results-state phase order, ledger refusal for fresh nouns/frames/tokens.
- [ ] Implement `build_confirmation_payload`, `validate_confirmation`, `load_confirmation`, `exposed_pool(manifest, extension)`, results-state helpers, runner `validate` and `freeze-confirmation`.
- [ ] Freeze `confirmation-v1.json` with the pinned tokenizer; validate; commit `feat: freeze experiment 006 confirmation set`.

### Task 2: E-patch residual response and the low-rank program

- [ ] Tests (tiny parallel-residual fake): `Δr_Epatch(ref_T, f) = 0` exactly; program `E(w)` equals the runner's captured/lexicon tensor; PCA orientation (`U_r` columns are unit vectors in `ℝ^{d_model}`, ordered by singular value); no-intercept fit recovers a planted linear map; `Δĉ(ref_T) = 0` by construction; independence from the network stack.
- [ ] Implement `measure_epatch_responses`, `fit_low_rank(tokens, responses, r)`, `LowRankProgram` (export/load, predictions), the `E005-scalar` baseline refit (from the Experiment 005 M3 mechanism on development data; record whether its parameter digest equals the Experiment 005 lock's).
- [ ] Commit `feat: add low-rank cue program for experiment 006`.

### Task 3: LOCO selection, quality gate, τ

- [ ] Tests: the rank rule on synthetic error tables (all four branches: no eligible rank; eligible but outside threshold; ties; best = 1); SE uses the sixteen tokens; the gate's three conditions; τ formula.
- [ ] Implement `loco_errors`, `select_rank`, `quality_gate`, `tolerance_tau`.
- [ ] Commit `feat: add leave-one-cue-out rank selection for experiment 006`.

### Task 4: Split-invariant circuit families and Y families

- [ ] Tests: P3 fidelity arithmetic (signs and correlation on synthetic pairs), cue-effect gate counts, Y1–Y3 floors, band hits (18 of 24 tokens inside in ≥ 5 of 6 frames), outcome labels for every case.
- [ ] Implement `circuit_families(ev, frames)` (P1, P3-fidelity, P4, P5, P7, P8, P9 via the Experiment 005 evaluators), `cue_effect_gate`, `y_families`, `outcome`.
- [ ] Commit `feat: add experiment 006 families and outcome rule`.

### Task 5: Phases `explore`, `calibrate`, `lock`, `confirm`, `report`

- [ ] Tests (fake model, forced floors where necessary): phase order; `explore` never touches fresh items; `lock` requires the quality gate; `confirm` refuses without a committed valid lock, refuses twice, and validates digests, unchanged scientific paths, and the ledger.
- [ ] Implement the phases and the report; document commands in the experiment README and the root README.
- [ ] Commit `feat: add experiment 006 phases`.

### Task 6: Verification, then Tier A

- [ ] Independent review of the implementation against the spec; fix findings; commit.
- [ ] `explore` (A0 contract test inside), `calibrate`, `lock`; present the candidate lock. **Stop**: installing and committing the lock is the user's act.

### Task 7: Confirmation and claims (after the lock commit)

- [ ] `confirm` once; `report`; evidence copies; C002 review and, if warranted, C003; READMEs; memory.
