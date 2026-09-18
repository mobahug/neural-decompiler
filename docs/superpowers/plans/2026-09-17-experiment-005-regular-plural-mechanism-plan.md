# Experiment 005 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and execute the approved Experiment 005 design (revision 4, commit `08bc146`): a three-tier prospective mechanism study of count-cued noun number selection in pinned Pythia-70M, ending in a preregistered, single confirmation run on the untouched reserve nouns and the frozen extension prompts.

**Architecture:** One experiment module `plural_mechanism.py` owns prompt-level data derivation from the frozen screening manifest, the tokenizer-only extension builder, position-indexed capture and replacement, contrast readout for every noun from one forward pass, every Tier A/B/C measurement, the hypothesis tree, mechanism-set selection, program-parameter estimation, bands, the lock schema, the outcome rule, and Markdown rendering. One weight-only module `mechanism_program.py` evaluates the decompiled computation from exported tensors and never imports the instrumentation. One runner exposes eight phases with phase isolation enforced through a results state, exactly as in the screen.

**Tech Stack:** Python 3.12, standard library, PyTorch 2.x, TransformerLens 3.9.x `TransformerBridge`, pytest 8.x. Existing `models`, `components`, `capture`, `interventions`, `provenance`, and `candidate_screening` modules are reused, not modified, unless a narrow defect is found.

**Spec:** `docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md` (revision 4). Where this plan and the spec disagree, the spec wins and this plan is amended before code changes.

## Global constraints

- Pinned `EleutherAI/pythia-70m-deduped` revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, CPU float32, compatibility mode disabled, `use_attn_result` enabled explicitly, `torch.inference_mode()` everywhere. Runtime seed `20260916`; control and bootstrap seed `20260918`.
- The only scientific inputs are `screening/behavior-candidates/manifest-v1.json` (digest `1c50c2e8…703c`, `regular-plural` cases only) and `experiments/005-regular-plural-mechanism/extension-v1.json`, which is built tokenizer-only and committed before `discover` runs.
- Every phase refuses to run on a dirty tree, without the committed extension digest, or out of order. `confirm` refuses without a committed, validated `preregistration-lock.json`, runs once, and has no override.
- Reserve nouns and extension prompts are never executed before `confirm`. Tests assert this by counting executed prompt/noun keys in the state.
- The mechanism program is restricted to token-local `E_program` (embedding, `L00.MLP`, or their declared combination). It loads only exported tensors under `outputs/experiment-005/parameters/` and never imports `transformer_lens`, `transformers`, or the instrumentation modules. Exact final LayerNorm: population variance (`correction=0`), `ε = 1e-5` read from the exported config.
- Structural facts encoded as assertions: activations at positions before `p_c` are bitwise identical within a pair; `L00.MLP` output at any position equals `MLP_0(ln2_0(embed[token]))` within `1e-5`; the direct-effect decomposition reconstructs `c(x)` within `1e-4` nats.
- Every floor, count, seed, and threshold is copied from the spec verbatim into named module constants and tested against the spec's numbers.
- Red-green-refactor per task; the default suite stays offline; the live contract test runs once in A0 with `NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1`.
- Conventional commits per task; no push until the user decides.

## File map

- `src/neural_decompiler/plural_mechanism.py` — everything scientific for Experiment 005 except the program.
- `experiments/005-regular-plural-mechanism/mechanism_program.py` — weight-only decompiled computation (lexicon, transport gain, readout, exact LayerNorm, predictions).
- `experiments/005-regular-plural-mechanism/run.py` — phases `validate`, `freeze-extension`, `discover`, `calibrate`, `revise`, `lock`, `confirm`, `report`.
- `experiments/005-regular-plural-mechanism/extension-v1.json` — frozen extension set with self-verifying digest.
- `experiments/005-regular-plural-mechanism/preregistration-lock.json` — installed by hand from `outputs/experiment-005/candidate-lock.json` and committed before `confirm`.
- `experiments/005-regular-plural-mechanism/README.md` — commands, boundaries, and (after the run) the evidence copy links.
- `experiments/005-regular-plural-mechanism/evidence/` — verbatim copies of final reports.
- `tests/test_plural_mechanism.py` — arithmetic, derivation, extension, axes, decomposition, tree, selection, bands, lock, outcome tests.
- `tests/test_experiment_005_runner.py` — phase isolation, non-execution, refusal, fake-runner tests.
- `tests/test_mechanism_program.py` — program independence, lexicon, exact LayerNorm, predictions.
- `.gitignore` — add `outputs/experiment-005/*`.
- `README.md` — add the Experiment 005 section after approval.

## Shared definitions used by every task

- **Frame**: `(template_id, frame_id, prefix_ids, suffix_ids, cue_ids={"sg": id, "pl": id}, p_c, p_t)`. Derived from the manifest by grouping `regular-plural` cases by template and prompt text; both cue prompts of a frame must share `prefix_ids` and `suffix_ids`, and the runner asserts exactly six manifest frames with `p_c = 3, p_t = 3` (cue-final) and `p_c = 5, p_t = 6` (coordinated).
- **Prompt**: `(frame, cue_token_id) → token_ids`. Manifest prompts: 12. New-frame prompts: 12. Cue-word prompts: 72 (six original frames × twelve words).
- **Noun**: `(lexical_key, split, rule_class, sg_id, pl_id, single_token)`. Interventions use single-token nouns only.
- **Contrast readout**: from one prompt's final-position logits, `c_N = logp[sg_id] − logp[pl_id]` for every noun `N` of the requested split. A "case" is a `(frame, noun)` pair; its two conditions are the two cues.
- **Component key**: `L{layer:02d}.H{head:02d}` / `L{layer:02d}.MLP` as in the screen, plus `EMBED` and `RESID_PRE.L{layer}` for the extra sites. A **site** is `(component_key, position)`.
- **Replacement map**: `{site: tensor}` executed in one forward through `InterventionPlan`; sources `REFERENCE` (counterfactual or freeze), `MEAN` (pair-centered), `RESAMPLED` (cross-frame, cross-cue, cue-word lexicon).
- **Aligned shift** for a set `S` on case `(frame, N)`: `d_patch = 0.5·((c_A − c_{A←B}) + (c_{B←A} − c_B))`; recovery `R = mean d_patch / mean d_full` over a stratum (overall, per template, per rule class), with denominator floors 0.25 / 0.10 nats.
- **Number variable at site `s`**: `n_s(x) = ((a_s(x) − μ_s)·d̂_s)/σ_s`; sites `E` (E outputs at `p_c`), `T` (declared T outputs at `p_t`; cue-final: E output at `p_t`), `R_in` (`RESID_PRE.L_R` at `p_t`), and `R_out` (summed declared R outputs at `p_t`, added by this plan for `m_R`).
- **Direct effect** of a residual term `t` on `c(x)`: `DE_t(x) = −u_N·(γ ⊙ (t − mean(t)) / sqrt(var(r_final) + ε))`, with `u_N = W_U[:, pl] − W_U[:, sg]`; terms are the embedding at `p_t`, every head result, every MLP output, every attention output bias `b_O(L)`, and the `β` term (`−u_N·β`); their sum equals `c(x)`.

---

### Task 1: Skeleton, manifest-derived prompts, and the frozen extension set

**Files:** create `src/neural_decompiler/plural_mechanism.py`, `experiments/005-regular-plural-mechanism/run.py`, `experiments/005-regular-plural-mechanism/README.md`, `tests/test_plural_mechanism.py`, `tests/test_experiment_005_runner.py`; modify `.gitignore`.

**Interfaces:** `derive_frames(manifest) -> tuple[Frame, ...]`, `nouns_for(manifest, split) -> tuple[Noun, ...]`, `EXTENSION_FRAMES` (six literal frame texts with `{cue}`), `EXTENSION_CUE_WORDS` (twenty ordered literals), `build_extension_payload(tokenizer, manifest) -> dict`, `extension_content_digest(payload)`, `load_extension(path) -> Extension`, `validate_extension(extension, manifest)`, `new_results_state(...)`, `assert_phase_allowed(phase, state)`.

- [ ] **Step 1: Failing tests for frame derivation and extension structure.** Six frames with the spec's positions; twenty nouns per split with the spec's rule-class counts; `peach` flagged two-token; extension has exactly six new frames × two cues, twelve cue words (`a` … `every`) × six original frames, no nouns, a self-verifying digest, and per-prompt `p_c`/`p_t`.
- [ ] **Step 2: Implement derivation, the extension builder (tokenizer-only: cue and coordinated adjective must be single tokens, first twelve eligible words), digest, loader, validator.** The builder refuses to overwrite an existing extension file.
- [ ] **Step 3: Runner phases `validate` and `freeze-extension`; results-state helpers copied in spirit from the screen (run id, protocol commit, dirty flag, versions, phases, executed prompt keys, executed noun keys).** `validate` loads manifest and extension without a model.
- [ ] **Step 4: Freeze the extension:** `uv run python experiments/005-regular-plural-mechanism/run.py freeze-extension`, then `validate`. Record the digest in the README.
- [ ] **Step 5: Tests green; full offline suite green.** Commit `feat: freeze experiment 005 extension set`.

### Task 2: Prompt-level execution primitives

**Files:** modify `plural_mechanism.py`, `tests/test_plural_mechanism.py`.

**Interfaces:** `capture_prompt(model, prompt, sites, *, attn_query=None) -> Captures` (per-site tensors on CPU, plus final logits and the `ln_final` input), `patched_logits(model, prompt, replacements, source_by_site) -> logits` with execution integrity checks (hook, position, exact write, zero outside change), `contrasts(logits, nouns) -> dict[noun, float]`, `aligned_shift`, `recovery(rows, strata)`, `deterministic_random_sets(universe, size, count=100, seed=20260918)`, `assert_prefix_identical(captures_A, captures_B, p_c)`.

- [ ] **Step 1: Failing tests on `TinyBridge`:** captures land at the requested position; a same-activation replacement is a no-op; a replacement at position `p` changes only downstream logits; prefix identity assertion passes for identical prefixes and fails otherwise; recovery arithmetic reproduces the screen's `aggregate_recovery` on synthetic rows.
- [ ] **Step 2: Implement.** Reuse `run_capture`, `run_interventions`, `InstrumentationSettings(use_attn_result=True)`; batch prompts of equal length; keep tensors on CPU.
- [ ] **Step 3: Tests green.** Commit `feat: add prompt-level capture and patching for experiment 005`.

### Task 3: Tier A measurements A1–A9

**Files:** modify `plural_mechanism.py`, `tests/test_plural_mechanism.py`.

**Interfaces (each returns JSON-safe dicts and is written to `discovery-results.json` before the next step):**
- `a1_baseline(model, manifest, screening_results_path)` — reuse `candidate_screening.score_behavior_case` for the 240 development + holdout cases; compare per case to the screening `results.json` measurements (`x_a.contrast`, `x_b.contrast`) within `1e-6` when the file exists (default search: `outputs/behavior-candidate-screening/results.json`, then `.worktrees/behavior-candidate-screening/outputs/behavior-candidate-screening/results.json`), and always compare aggregates to the report constants (`5.231`, `4.639`, `4.606`, `4.969`, `4.343`) within `1e-3`. Mismatch raises `IncidentError`.
- `a2_position_map(model, frames, nouns)` — singleton counterfactual shifts for all 54 components at `p_c` and `p_t` (cue-final: one position), both directions; per-template recoveries; prefix identity assertions.
- `a3_layer_profile(model, coordinated_frames, nouns)` — `RESID_PRE.L` at `p_c` for `L = 0..5`; assert `L = 0` recovery within `1e-4` of 1.0.
- `a4_attention(model, coordinated_frames)` — every head's pattern from query `p_t`; attention to `p_c`; ranking.
- `a5_axes_and_direct_effects(...)` — `d_num(L, p)` per template; `μ, d̂, σ` for every site; component number contributions at `L_R` (provisional `L_R` = layer of the top MLP at `p_t` until A10 fixes it); cosines across templates, cue words, frames; direct-effect decomposition with the exact-sum assertion (`IncidentError` on violation); indirect = total − direct.
- `a6_chain(...)` — for candidate T heads (top three at `p_t` by A2 in coordinated): E-alone patch with captures of T and R outputs → `m_T`, `m_R` (aligned, both directions); with T frozen → `m_R|T`; `RESID_PRE.L_T` at `p_c` alone and with T frozen → blocked fraction; R-alone patch; joint E∪T recorded as sufficiency only.
- `a7_abstractness(...)` — cross-cue (cue-final) and cross-frame (all templates) resample replacements for E and for candidate T; same-number and opposite-number recoveries.
- `a8_neutralization(...)` — pair-centered neutralization of each candidate and of candidate sets; contrast loss; per-component direct-effect changes; compensation ratio; conditional co-neutralization with the top compensator.
- `a9_isolation(model, frames, nouns, set_)` — hold the set clean, neutralize every other universe component at `p_t` (and `p_c`); `F`, sign-retention count.

- [ ] **Step 1: Failing arithmetic tests with synthetic tensors:** direct-effect sum identity on a hand-built residual with known `γ, β, ε, W_U`; `n_s` maps a balanced pair to `±1`; `m_T`/`m_R` equal 1 when the patched projection equals the counterfactual; blocked fraction formula; compensation ratio; `F` and recovery denominators.
- [ ] **Step 2: Failing plumbing tests on `TinyBridge`:** A2 asserts prefix identity; A3 `L = 0` equals the full counterfactual; A9 neutralizes exactly the complement.
- [ ] **Step 3: Implement A1–A9.** Each step writes before interpreting; every intervention run records its execution integrity.
- [ ] **Step 4: Tests green.** Commit `feat: add experiment 005 discovery measurements`.

### Task 4: Hypothesis tree, set selection, mechanism statement, program parameters, and the program

**Files:** modify `plural_mechanism.py`; create `experiments/005-regular-plural-mechanism/mechanism_program.py`, `tests/test_mechanism_program.py`.

**Interfaces:**
- `hypothesis_tree(q1, q2, q3, q4, q5) -> (row, flagged)` exactly as the spec's pseudo-code.
- `encoding_branch(d_num_LT_pc_decomposition) -> "L00.MLP" | "EMBED"`; `program_eligible(component_key) -> bool` (`EMBED`, `L00.MLP` only).
- `select_mechanism_set(a2, floors)` — E entry from the branch; rank the 53 non-`L00.MLP` components at `p_t` by mean singleton shift across templates; cumulative top-`k` for `k = 0..6`; evaluate recovery (overall, per template, per rule class), isolation `F`, role presence, and program floors; smallest passing `k`; roles: heads → T, MLPs → R; `L_R` = earliest R layer; `L_T` = earliest T layer.
- `estimate_program_parameters(...)` — lexicon axis `(μ_E, d̂_E, σ_E)`; `k_T` by least squares of `n_t` on `n_c` over coordinated prompts; `v_T` per template = half the mean pair difference of the summed `S_M@p_t` outputs (`g_R = |v_T|`, `d̂_R = v_T/|v_T|`); `ρ_frame` (six), `ρ_template` (three) from the `ln_final` input at `p_t`; export tensors (embedding, `ln2_0`, `mlp_0` weights, `ln_final` weights, `W_U`, config `ε`, activation name) to `outputs/experiment-005/parameters/` with digests.
- `render_mechanism_statement(...)` in the spec's fixed form.
- `mechanism_program.py`: `load_parameters(dir)`, `lexicon(token_id)`, `n_t(template, token_id)`, `exact_layernorm(r, γ, β, ε)`, `predict_contrast(template, frame_or_None, cue_token_id, sg_id, pl_id)`, `predict_shift(...)`, `predict_epatch_shift(...)`, `predict_pair(...)`. The module must not import `transformer_lens`, `transformers`, `neural_decompiler.capture`, or `neural_decompiler.interventions` (tested by inspecting `sys.modules` after import in a subprocess).
- `program_floors_tier_a(...)` — every development pair sign; per-template mean within 1.0 nats.
- `tier_a_floors(...)` — the four spec floors plus the program floors; version record `M1`.

- [ ] **Step 1: Failing tests:** the tree on boundary values (`q3 = 0.49/0.50`, `q1 = 0.50`, `q5 = 0.70`); branch rule at exactly half; eligibility rejects `L01.MLP` and any head; selection picks the smallest passing `k` on synthetic A2/A9 tables; exact LayerNorm matches `torch.nn.functional.layer_norm` on random tensors; lexicon for a token equals a reference MLP evaluation; program import leaves the forbidden modules unloaded.
- [ ] **Step 2: Implement.** The runner asserts, once per discover, that captured `L00.MLP` output at `p_c` for each of the four development cue tokens equals the program's lexicon vector within `1e-5` (GELU variant check).
- [ ] **Step 3: Tests green.** Commit `feat: add mechanism selection and weight-only program for experiment 005`.

### Task 5: Tier B calibration, bands, revision, and the lock

**Files:** modify `plural_mechanism.py`, `run.py`; tests.

**Interfaces:** `evaluate_families(model, version, frames, nouns, extension=None) -> FamilyResults` (the single evaluator used by calibrate and confirm; the family list, floors, and exact-count rules are module constants copied from the spec), `bootstrap_bands(rows, seed=20260918, resamples=1000)` (pairs resampled with replacement, stratified by template at 38 per template on holdout), `program_residual(...) -> RMSE_B, τ`, `revise_version(version, failing_families, a_measurements)` (the two mechanical revisions; refuses a third version), `build_candidate_lock(...)`, `validate_lock(path, *, manifest, extension, state, git)`.

- [ ] **Step 1: Failing tests:** bootstrap determinism and band formula (`max(0.10, 1.5·SE)`, `±0.5` nats); `τ = max(0.5, 3·RMSE_B)`; discriminating bands must be disjoint; revision picks the next tree row for P5/P9 failures and extends `S_M` for set-level failures; the third `calibrate` refuses; lock validation rejects missing, uncommitted, modified, or digest-mismatched locks and any executed reserve/extension key.
- [ ] **Step 2: Implement phases `calibrate`, `revise`, `lock`.** `lock` writes `outputs/experiment-005/candidate-lock.json` and prints the install instruction; it never writes the tracked path.
- [ ] **Step 3: Tests green.** Commit `feat: add calibration, bands, and preregistration lock for experiment 005`.

### Task 6: Confirmation, outcome rule, report, documentation

**Files:** modify `plural_mechanism.py`, `run.py`, `README.md` (root), experiment `README.md`; tests.

**Interfaces:** `confirm(...)` — validates the lock, runs `evaluate_families` on reserve (twelve prompts), new frames, and cue-word prompts (behavior, E-patch, program), writes `confirmation-results.json`, and marks the phase complete; `outcome(results, lock) -> Outcome` (two axes, `PROGRAM_CAPPED`, contested rule); `render_report(state) -> str`.

- [ ] **Step 1: Failing tests:** outcome rule on synthetic family results for every label; a second `confirm` refuses; report lists every family with floor, band, value, and pass/hit.
- [ ] **Step 2: Implement; document commands and boundaries in both READMEs.**
- [ ] **Step 3: Full offline suite green.** Commit `feat: add confirmation, outcome rule, and report for experiment 005`.

#### Amendment — 2026-09-18: instrumented-path numerics (pre-conclusion)

The first `discover` attempt stopped in A1 before any mechanism version was
written: the prompt-level readout differed from the case-level teacher-forced
contrast by up to 4.5e-3 nats against the plan's 1e-4 tolerance. The cause is
numerical, not scientific: every Experiment 005 measurement runs through the
instrumented forward with `use_attn_result=True`, which computes attention
through per-head results and changes float32 rounding (Pythia-70M logits reach
~1.6e3 in magnitude), while the screen's teacher-forced scoring and the plain
forward agree bitwise. The per-case comparison against the screen (1e-6)
passed. The prompt-level tolerance is set to 1e-2 — the same tolerance the
design already applies to the LayerNorm reconstruction's model gap — and the
measured gap is recorded in the A1 result. All Experiment 005 quantities are
consistent within the instrumented path; `discover` restarts under the
crash-recovery rule because nothing had been concluded.

#### Amendment — 2026-09-18: protocol v2 continuation (design revision 5)

Tier A of protocol v1 ended in `NO_COMPACT_MECHANISM` at the program floor
(quantifier mean gap 1.44 nats > 1.0). That result is preserved. Under design
revision 5 the runner gains a `continue` phase that adopts, without any new
search, the smallest recorded selection attempt whose circuit floors passed
(k = 3), re-exports its program parameters from the same development
activations, records version `M2` with `PROGRAM_CAPPED` set from the outset,
and refuses `revise` thereafter. The outcome rule gains
`CIRCUIT_NOT_GENERALIZED`: `CIRCUIT_ONLY` now requires X1 and X2. Calibration,
the lock, and the single confirmation are unchanged.

#### Amendment — 2026-09-18: continuation eligibility corrected (design revision 5, amended)

The first `continue` run adopted k = 2 as `M2` under the Tier A circuit floors
as written (isolation overall only). Because P3 applies a per-template
isolation floor of 0.40 at confirmation and k = 2 already showed 0.365 on the
coordinated-adjective template, the eligibility rule is corrected before any
Tier B execution to use only split-stable Tier A criteria: recovery strata,
isolation overall (≥ 0.50), isolation per template (≥ 0.40), and roles. The P3
sign-retention count is not used for selection (its ceiling depends on the
split's clean flip count; 87/120 on development). `continue` may run a second
time only to supersede a continuation adopted under an older rule, before any
calibration; `M2` is recorded as superseded, `M3` = k = 3 is adopted, and the
cap-reason text names the protocol v1 program floor failure.

### Task 7: Independent review, then Tier A execution

- [ ] **Step 1: Subagent review of the implementation against the spec** (definitions, floors, seeds, phase isolation, non-execution, program independence, LayerNorm semantics). Fix findings; commit.
- [ ] **Step 2: A0:** `NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 uv run pytest tests/test_pythia_bridge_contract.py -m pythia_smoke -q` on the scientific runtime; the `discover` phase re-runs it in a subprocess and records the result.
- [ ] **Step 3: `discover`** on a clean committed tree with the screening `results.json` available for A1. Inspect `discovery-results.json` only through the rendered report.
- [ ] **Step 4: `calibrate`.** If floors fail, `revise` once and `calibrate` again; never a third time.
- [ ] **Step 5: `lock`.** Present `candidate-lock.json` and the rendered mechanism statement for review. **Stop here**: installing and committing the lock is the preregistration act and is the user's decision.

### Task 8: Confirmation and claim (after the lock commit)

- [ ] **Step 1: `confirm`** once; `report`; copy the final report and lock into `evidence/`.
- [ ] **Step 2: Create `research/claims/C002-…md`** at the lock commit (PROPOSED, confirmatory) and update it after the run per the methodology; register anomalies if any.
- [ ] **Step 3: Update READMEs and memory; commit `docs: record experiment 005 result`.** Ask about push.
