# Experiment 025 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan revision 1** (2026-09-26). It tracks design revision 1 (`c0885e5`), the final Experiment 025 design the reviewer
approved. It is a plan only: no code, no runner, no freeze, no lock and no prompt.

**Goal.** Implement the approved design:
- **The intervention.** A norm-preserving rotation of each fresh cue's input embedding at `("EMBED", p_c)`, along the
  frozen 024 nounness direction (odd component ±0.32 primary, ±0.16 secondary). It is matched by 7 deterministic,
  nounness-neutral random tangent controls at the same angle, and a plurality control.
- **The three outcome-bearing statistics:**
  - `A`: causal direction, on the frozen-readout discrepancy `ℓ`;
  - `B`: specificity against the absolute random effects;
  - `G`: the direct L4/L5 routing summary `D_attn`.

  Each passes at ≥ 27/40 strictly positive.
- **The outcome:** the frozen five-level hierarchy.
- **Descriptive records**, after the result.

**Architecture.** One new module, `src/neural_decompiler/cue_rotation.py` (imported as `cr`), and one runner,
`experiments/025-nounness-direction-intervention/run.py`. The module *calls* frozen modules and never edits them. Every
called module is pinned by git blob, and the runner refuses any drift.
- **Experiment 024's `readout_routing.py`** (`rr`, blob `e6cb37767d1d06c6ff40804a88eab569723afdb5`):
  - the score definitions and 024's committed artifacts: the calibration record (the direction's id lists), the lock
    (024's scores, for the 0.32 provenance) and the freeze (reserves, exclusions, frames);
  - `fresh_pair_cells` and `cue_mse` (fsum), `target_units`;
  - `write_state_atomic`, `save_durably`, `fsync_directory`;
  - `assert_ledger_isolated`;
  - the frozen-input and binding checks it already exposes.
- **Experiment 022's `upstream_localization.py`** (`ul`, blob `465856962aa380747d1a4f1338d1d2762d03c9f9`):
  `load_frozen_inputs`, `ModelPrograms`, `measured_sites`, `measure_prompt` (the reference for the patched
  measurement), `contrast_of`, `block0_terms`, `Factors`, `PairContext`, `reference_rows_017`,
  `reference_embeddings`, `compose_dx3`, `i3_error`, `table_units`, `y1_states`.
- **Experiment 023's `block0_completion.py`** (`b0c`, blob `16d310fc7fab5599ec83b8bd8162fb9613f8dad2`): `FULL_MASK`,
  `pair_cells`, `CELL_COLUMNS`, `exclusion`, `verify_022_inputs`.
- **`readout_decompilation.py`** (`rd`): the 020 readout (`contrast`, `blocks_3_to_5` via `ul.contrast_of`,
  `level1_detail`) and the locked states.
- **`plural_mechanism.py`** (`pm`): `run_patched`, `capture_prompt`, `Prompt`, `Frame`, `ReplacementSource`,
  `Weights`, `exact_layer_norm`, `record_execution`, `canonical_json`, `sha256_text`.

## Global constraints

- **No pinned 020–024 module is edited.** New code lives only in `cr`, the 025 runner and the 025 tests.
  `cr.assert_frozen_blobs()` checks every called module's blob before anything is loaded.
- **Numerics:**
  - geometry in float64, from the model's float32 embedding rows cast to float64;
  - patched vectors are float32 casts of the float64 rotations;
  - all canonical digests use `rc.tensor_digest` and `pm.canonical_json`.
- **Phase discipline:**
  - the phases are one-shot: `freeze` → `lock` → `confirm` (once, never resumed) → `report`;
  - there is no calibration phase;
  - the ledger is written before any fresh prompt;
  - state writes are atomic;
  - an incident never coexists with a result.
- **No scientific effect before confirm.** No test, script or phase before confirm may compute, log or inspect `ℓ`,
  `A`, `B`, `G`, `D_attn` or a Level-1 value for a rotated run. Tier-C guards enforce this (see Testing).
- **Environment:** `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`, the repository's `.venv`, and 4 torch threads.
  Never `uv run` or `uv sync`.

## Constants (a tier-A test pins every one)

- **Identity:**
  - `EXPERIMENT = "025"`;
  - `EXPERIMENT_DIR = "experiments/025-nounness-direction-intervention"`;
  - `DESIGN` (path, revision 1, commit) and `PLAN` (path, revision 1, commit);
  - `FROZEN_BLOBS`: `rr`, `ul`, `b0c`, `rd`, `pm`, `rc` and the modules `ul` pins.
- **Inherited from 024** (file and content digests; refused on drift):
  - `calibration-v1.json`: `81fb499e…` / `09bf093e…`;
  - `preregistration-lock.json`: `5c2a9b90…` / `a39668bf…`;
  - `confirmation-v1.json`: `68510e1b…` / `87f8aff1…`.
- **The prior-noun source:** 020's committed `confirmation-v1.json` (file digest pinned), whose 24 nouns give the 48
  extra excluded form ids.
- **The population:**
  - `ADJECTIVE_RESERVES`, read from 024's freeze `reserves.N` and pinned by digest;
  - `ORDINARY_RESERVES`, from `reserves.ordinary`;
  - `NEW_NOUN_LIST`, the frozen 39-word text in design order;
  - `QUOTAS`: 20 adjectives; 20 nouns, the reserves first, then the new list;
  - `EXPECTED_PICKS`: the design's 40, in order.
- **The intervention:**
  - `PRIMARY_ODD = 0.32` and `HALF_ODD = 0.16`;
  - `K_CONTROLS = 7`;
  - `CONTROL_TAG = "025|control|{token_id}|{j}"`;
  - `CONDITIONS`: 21 names, in fixed order: `base`, `noun+0.32`, `noun-0.32`, `noun+0.16`, `noun-0.16`,
    `rand1+0.32` … `rand7-0.32`, `plur+0.32`, `plur-0.32`;
  - `OUTCOME_BEARING`: the 16 primary-dose conditions.
- **The statistics:**
  - `N_CUES = 40` and `COUNT_THRESHOLD = 27`;
  - `REFERENCE_TAIL = 0.01923865414210013`. A tier-A test derives it from the exact rational `P(Bin(40, ½) ≥ 27)`,
    and checks that 26 gives 0.040345… > 0.025.
- **Tolerances:**

  | check | tolerance |
  |---|---|
  | geometry, float64 | 1e-12 |
  | patched float32 vectors (neutrality, odd component, length) | 1e-6 |
  | angle | 1e-12 rad (float64), 1e-6 rad (float32) |
  | Level-1 identity | 2e-2 |
  | I1 | 1e-4 |
  | I3 | relative 1e-4 |
  | I4 | 1e-3 |
- **`PATCH_PATH_SPENT_KEYS`:** 4 already-executed keys, frozen by key at the lock, the 024 tier-C allow-list. Used only if
  the reviewer signs off on the confirm-time patch-path check; otherwise used only in tier C:
  - `cardinal-009-1|an|271`;
  - `quantifier-009-1|least|1878`;
  - `coordinated-adjective-009-1|black|2806`;
  - `quantifier-new-2|he|344`.
- **The labels and semantics:** the five outcome labels and the frozen wording. That wording covers the terminology,
  the meaning of `A` (+ against −, not against the baseline), `B`'s rationale, `G`'s conservative reading, the
  non-claims and the stratum rule.

## The canonical computations (one implementation each)

1. **The direction.** `d = unit(mean W_E[noun_row_ids]) − unit(mean W_E[calibration_cue_ids])`, using the id lists of
   024's calibration record, in float64.
2. **The cue geometry.** `E` (the float64 row), `Ê`, `s₀ = d·Ê`, `t = d − s₀Ê`, `τ = |t|` and `t̂`. Then
   `θ(odd) = asin(odd/τ)` for 0.32 and 0.16.
3. **The rotation.** `R(dir, θ) = |E|·(cos θ·Ê + sin θ·dir)`, with `dir ∈ {±t̂, ±u_j, ±p̂′}`. `R(·, 0)` is `E`, and
   its float32 cast must equal the model row bit for bit.
4. **The controls.** A SHA-256 counter-mode byte stream from `CONTROL_TAG` gives 53-bit uniforms. Box–Muller in float64
   turns them into a 512-vector, which is projected off `Ê` and `t̂` twice and normalized.
   - A second, independent implementation in the tests must reproduce the vectors bit for bit.
5. **The plurality direction.** `p̂`, the unit mean over the 79 scorable nouns of plural minus singular rows. Its
   tangent part at `Ê` is orthogonalized to `t̂` and normalized, giving `p̂′`.
6. **The condition vectors.** For each cue, the 21 float64 vectors, their float32 casts and their digests.
7. **I7′.** The checks of the design table, per cue, per control and for the plurality control. Each returns a record of
   the maximum deviations, and any failure is an incident.
8. **The patched measurement** `measure_rotated(model, progs, frame, state, token_id, word, v32)`:
   - `pm.run_patched` with `{("EMBED", p_c): v32.reshape(1, 1, -1)}`, source `DIRECT`, and capture sites
     `ul.measured_sites(frame)` plus `("ATTN_PATTERN.L4", p_t)` and `("ATTN_PATTERN.L5", p_t)`;
   - it returns the raw float32 captures, `dc` (float64, `ul.measure_prompt`'s formula), `rows4` and `rows5`, and the
     integrity records;
   - `Δx1` and `Δx3` are derived exactly as `ul.measure_prompt` derives them, and `C = ul.contrast_of(progs, state, Δx3)`.
9. **Vector gates.** `vector_factors(progs, x0_all, frame, v32, ref_id)` is `ul.pair_factors` with:
   - `d_emb = v − W_E[ref]`;
   - `delta_e = lexicon(v) − lexicon(ref)`, where `lexicon(v) = MLP₀(LN₂(v))` is `pm.lexicon_vector`'s body applied to
     a vector.

   Then `vector_pair_context` builds the context, and I1, I3 and I4 follow `rr.target_gates`' formulas unchanged, via
   `ul.compose_dx3`, `ul.i3_error` and `ul.contrast_of`.
10. **Level-1.** `max |readout.contrast(state, readout.level1_detail(state, Δx3)["dh6"], nouns)[scorable] − Δc|` for
    each run.
11. **`D_attn`.** For each run, the mean over the 16 heads (blocks 4 and 5) of `½·Σ_keys |row − ref_row|`, with the
    locked `rows4` and `rows5`. Per cue and condition, the `fsum` mean over the 108 frames.
12. **`ℓ`.** Per cue and condition, `log(rr.cue_mse(cells))`, with the cells from `rr.fresh_pair_cells(Δc, C)` over
    the 108 frames.
13. **The statistics.**
    - `A_i`, `A_ij`, `B_i = A_i − fsum(|A_ij|)/7` and `G_i`;
    - the positive counts, strictly `> 0`, with ties and zeros counting against;
    - the outcome, from the table;
    - the stratum counts.
14. **Descriptives** (after the result):
    - the magnitudes, the even components and the baseline comparisons;
    - the half dose and the plurality control;
    - `A` on nMSE;
    - the ladder: `C5 − C` and `L1 − C5` on the nounness and baseline runs;
    - `G` against the random directions;
    - the per-template breakdown, and the causal fraction relative to 024.

## Artifact schemas

- **`confirmation-v1.json`** (freeze; committed). It holds:
  - the rules text and the exclusion sources with their digests;
  - the ordered candidate lists (the reserves, and the new list's text);
  - the rejections with reasons, and the 40 picks (word, token id, stratum, source list, rank);
  - `picks_match_expected`;
  - the 108 exposed frame ids, from 024's freeze;
  - the 21 conditions;
  - the manifest: 90,720 condition-tagged keys, in canonical order, with a digest;
  - the counts, and `content_sha256`.
- **`preregistration-lock.json`** (lock; committed after review). It holds:
  - the bindings: the freeze, 024's three artifacts, the module blobs (`cr`'s own included), and the model parameter
    and embedding digests;
  - the direction's digest;
  - per cue: `s₀`, `τ`, `θ` at both doses, the even term, and the digests of `t̂`, the 7 controls, `p̂′` and the 21
    float64 and float32 vectors;
  - the I7′ maximum deviations, and the nearest-token angles (descriptive);
  - the thresholds and the tail, the conditions, the outcome table and the semantics;
  - `content_sha256`.

  `preregistration.md` is rendered from the lock.
- **`results.json`** (state): the phases; the condition-tagged ledger; the stage-2 digests; the accounting; the gate
  records (the patch-path check, I1/I3/I4, the `C` recomputation, Level-1 over the 16 conditions); the results (per
  cue and condition `ℓ` and `D_attn`; `A`, `A_ij`, `B`, `G`; the counts; the outcome); the descriptives.
- **`stage2-measurements.pt`** (local; about 0.9 GB). Per group and condition:
  - `x1_raw` and `x3_raw` [P, 2, 512], float32, at (p_c, p_t);
  - `dc` and `C` [P, 79], float64;
  - `rows4` and `rows5` [P, 8, K], float32, padded, with a key-length vector;
  - `positions` [P, 2], int64.

  `Δx1` and `Δx3` are derived exactly from the raw captures and the locked states. Every tensor gets a
  `rc.tensor_digest`.

## Phase control flow

1. **`validate`:** the pins, 024's artifacts, the frozen inputs, and any installed 025 freeze or lock.
2. **`freeze`** (tokenizer only):
   - the mechanical selection under the tightened rules;
   - a `FreezeDeviation` if the picks differ from `EXPECTED_PICKS`, writing nothing;
   - it writes `confirmation-v1.json`, which is reviewed, then committed by hand.
3. **`lock`** (weights only; every forward refused):
   - the direction, the cue geometry, the controls, the plurality direction, the vectors and I7′;
   - it writes the candidate lock and the preregistration, which are reviewed, installed byte for byte and committed.
4. **`confirm`** (once; never resumed), strictly in this order:
   1. `validate_lock`, including a direct assertion of `cr`'s and `rr`'s module blobs;
   2. the model's parameter and embedding digests;
   3. I7′: everything recomputed bit for bit against the lock, and every geometry gate. An incident here comes before
      any prompt.
   4. The **patch-path check** on the 4 spent keys:
      - a plain capture against a patched θ = 0 run, bit for bit;
      - the embedding hook against the block-0 residual input, bit for bit.

      No scientific quantity is computed. Only equality results and digests are recorded, in a separate record outside
      `executed_prompt_keys`, the manifest and the accounting. A failure is an incident before the ledger. This step
      exists only if the reviewer signs off; otherwise it is omitted, and the identity rests on the engineering tests
      and tier C.
   5. The ledger records the 90,720 keys.
   6. Stage 2: for each pair in `ul.table_units` order, the 21 conditions in frozen order, with `run_patched`'s
      integrity check on every run.
   7. A durable save with digests.
   8. The accounting, written and then enforced.
   9. A re-read with the digests checked.
   10. `C` recomputed from the saved `Δx3` bit for bit.
   11. I1, I3 and I4 on every run, written and then enforced.
   12. The Level-1 identity on the 69,120 outcome-bearing runs, written and then enforced.
   13. `ℓ` and `D_attn`.
   14. `A`, `B` and `G`, the counts and the outcome.
   15. The 020–024 re-check.
   16. **One atomic write** of the result and the completed phase.
   17. The descriptives, whose failures are recorded without touching the result.
5. **`report`:** rendered from the state.

## Ledger and isolation

- **Keys:** `f"{frame_id}|{word}|{token_id}|{condition}"`. The untagged keys must not collide with the spent set:
  020's ledger, 020's confirmation set, and the 022, 023 and 024 manifests, 43,632 keys in all.
- **The 4 patch-path spent keys** (only if signed off) are recorded in a separate `patch_path_check` record. It sits
  outside `executed_prompt_keys`, the 90,720-key manifest and the accounting, so the isolation and accounting gates are
  unaffected. The keys are already spent and are never scored.

## Incidents, stops and failures

- **A freeze deviation** is a stop that writes nothing.
- **An I7′ or patch-path failure** at confirm is a phase incident before the ledger, with no fresh key executed.
- **After the ledger,** any failure is an incident: the measurements are preserved, no result is written, and the
  outcome is `NOT_INTERPRETABLE`. That covers the accounting, the digests, the `C` recomputation, I1/I3/I4, Level-1, a
  per-run integrity failure, and the re-check.
- **A descriptive failure** is recorded, and the result stands.
- **An interruption** spends the protocol version. Confirm never resumes.

## One-shot protections

- **The phase rules** (like `rr.assert_phase_allowed`):
  - `freeze` and `lock` each run once;
  - `confirm` requires a completed lock and refuses when it is not `not_started`, so a second confirm is refused and
    there is no resume;
  - an I7′ incident blocks confirm until the reviewer decides.
- **`validate_lock` at confirm** checks:
  - that the installed lock and preregistration are tracked, byte-identical to the candidates, and rendered
    consistently;
  - a clean tree, with no scientific path changed since the lock commit;
  - the lock's module blobs against the running code, directly for `cr` and `rr`.
- **The production confirm runs through an external guarded launcher**, as in 024:
  - an exclusive sentinel, and a record-only audit hook;
  - the direct module-blob assertion before the model loads;
  - an environment check;
  - a prompt log that records each patched run's key and delegates unchanged.
- **The ledger** records all 90,720 keys before the first run. An interruption spends the protocol version.
- **The production configuration** is literal and immutable. A non-production configuration is refused at the real
  repository root.

## Testing

- **Tier A (pure):**
  - the geometry on synthetic vectors: the length, the odd component, the angle, `R(0) = E`;
  - the control generator: a pinned digest, and an independent re-implementation;
  - neutrality, and the plurality orthogonalization;
  - the conditions and the manifest;
  - the sign-count rule and its exact tail, the tie rule, and `B` with absolute values;
  - the outcome table;
  - `D_attn` total variation;
  - the freeze selection on synthetic token maps;
  - every constant.
- **Tier B (fake world):** a fake world on TinyPlural and 024's locked fixtures, with an explicit test configuration
  (small N, K = 2, a matching threshold). It covers:
  - the write ordering, with a spy on every state write;
  - one-shot confirm, and a refused second confirm;
  - an I7′ drift and a patch-path failure, each before any key;
  - accounting, `C` recomputation and injected Level-1 violations, each giving an incident with no result;
  - a failed result write, giving an incident;
  - a descriptive failure, leaving the result unchanged.

  The production configuration is refused at the real repository root unless it is the literal production one.
- **Tier C** (real model; spent prompts only; an allow-list refuses every 025 key before execution):
  1. the freeze picks with the real tokenizer, which must equal `EXPECTED_PICKS` (tokenizer only);
  2. the patch path on the 4 spent keys;
  3. `measure_rotated` at θ = 0 against `ul.measure_prompt`, bit for bit, on spent pairs;
  4. vector I1/I3/I4 at θ = 0 against the token-id gates, on spent pairs;
  5. the geometry and I7′ on **024's spent cues**, never on 025's (their geometry is first computed at the lock);
  6. a rotated patch landing on spent prompts, **capturing only `EMBED`**.

  A guard fixture makes `ul.contrast_of`, `level1_detail`, `cr.d_attn`, `cr.ell` and `cr.statistics` raise if they are
  called in these rotated tests.

## Runtime and storage (estimates on this machine)

- **Stage 2:** 90,720 patched forward passes. 024 took about 0.068 s per run including `C`, so about 1.7 hours.
- **The gates:** I1/I3/I4 on every run, about 40 minutes; Level-1 on 69,120 runs, about 25 minutes.
- **Scoring and descriptives:** a few minutes.
- **Total:** about 2–3 hours. The design's rough figure was about 2 hours.
- **Storage:** about 0.9 GB of measurements.

## File map

- `src/neural_decompiler/cue_rotation.py` (new)
- `experiments/025-nounness-direction-intervention/run.py` and `README.md` (new)
- `tests/test_cue_rotation.py` (tier A and tier-C contracts; new)
- `tests/test_experiment_025_runner.py` (tier B; new)
- `tests/conftest.py`: `CURRENT_EXPERIMENT = "025"`
- `.gitignore`: `outputs/experiment-025/*`

## Tasks (small reviewable commits; the checkboxes are for execution)

- [ ] **1. Constants and geometry:** pins, the geometry, the control generator, the plurality direction, the
  conditions, the manifest builder, the statistics, the outcome table. Tier A.
- [ ] **2. Freeze:** the tightened-rule selection, the payload, `FreezeDeviation`. Tier A, plus a tier-C tokenizer
  contract.
- [ ] **3. Lock:** the weights-only geometry, the vectors, I7′, the lock, the preregistration, `validate_lock`. Tier A,
  plus tier C on spent cues.
- [ ] **4. Measurement and gates:** `measure_rotated`, the vector factors, Level-1, `D_attn`, the patch-path check.
  Tier-C spent contracts with the inspection guard.
- [ ] **5. Confirm and the runner:** stage 2, the save, the accounting, the `C` recomputation, the gates, the
  statistics, the atomic write, the descriptives, the phases. Tier B.
- [ ] **6. Report and README.**
- [ ] **7. Review:** the independent implementation review, fixes, then all tiers green.

## Stopping conditions

Stop after this plan for review. Then, each step only when authorized:
1. the implementation (Tasks 1–7) and its independent review;
2. `freeze`, then the review and commit;
3. `lock`, then the review, install and commit;
4. `confirm` once, then the independent confirmation review;
5. `report`;
6. closure.

**After 025, on any clean result, stop.**
