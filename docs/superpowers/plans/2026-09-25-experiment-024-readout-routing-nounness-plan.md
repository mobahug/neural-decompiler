# Experiment 024 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan revision 1** (2026-09-25). It tracks design revision 2 (`9d03dee`), which the reviewer approved with the
population kept at 8 × 5 = 40 cues. It is a plan only: no code, no runner, no freeze, calibration, lock or fresh prompt.

**Goal.** Implement the approved Experiment 024 design:
- **The primary test.** The Spearman correlation across 40 never-executed cues between a frozen, weight-derived
  operational nounness score and each cue's MSE of `Δc − C`, over the 108 exposed frames × 79 nouns. It passes iff
  `ρ ≥ max(F_ρ, null₉₇.₅)`, with the four-way results.
- **The E–N disambiguation guard.** An exact one-sided permutation test of
  `D_EN = mean(log MSE_E) − mean(log MSE_N)` over all 12,870 assignments.
- **The outcome.** The frozen hierarchy that combines the two.
- **After the results:** the secondary analyses and the descriptive block-4/5 ladder.

**Architecture.** There is one new module, `src/neural_decompiler/readout_routing.py` (imported as `rr`), and one
runner, `experiments/024-readout-routing-nounness/run.py`. The module *calls* frozen modules and never edits them.

Experiment 023's `block0_completion.py` (`b0c`, git blob `16d310fc7fab5599ec83b8bd8162fb9613f8dad2`) provides:
- `exposed_units`, `cells_meta`, `read_cells`: the exposed-cells reader, verified against its index;
- `pair_cells`: 023's cell function, the only one;
- `exclusion`, `load_confirmation_023`: 023's committed freeze, verified;
- `verify_022_inputs`, `STRATA`, `FULL_MASK`, `CELL_COLUMNS`, `pool`.

Experiment 022's `upstream_localization.py` (`ul`, blob `465856962aa380747d1a4f1338d1d2762d03c9f9`) provides:
- `load_frozen_inputs`, `ModelPrograms`, `y1_states`, `table_units`;
- the measurement:
  - `stage_two_022`, through `measure_prompt`;
  - `C` is `contrast_of` of the measured `Δx3`;
  - `measurement_tensors`;
- for the gates: `pair_context`, `reference_rows_017`, `compose_dx3`, `contrast_of`, `i3_error`, `_worse`;
- `order_statistic`, `defined_median`, `exposed_states_digest`.

Through them it uses 020's readout, `rd.ReadoutProgram`: `C` and, descriptively, `level1_detail`.

It adds only what the design adds:
- the weight-only nounness score and its centroids;
- the per-cue MSE;
- one Spearman implementation;
- the exposed-data OLS line;
- the SHA-indexed calibration of `F_ρ` and `null₉₇.₅`;
- the exact E–N guard and the outcome;
- the 40-cue tokenizer-only freeze;
- the lock and the confirmation scoring;
- the secondary analyses, the ladder and the report.

There is no extraction and no prediction table. The calibration source is 023's committed artifact, and the predictor
is the score.

## Global constraints

- **Frozen modules, checked by git blob.** Every phase and a tier-A test compute `sha1(b"blob <len>\0" + bytes)` with no
  git call.
  - The blobs are 022's ten (`b0c.FROZEN_BLOBS` minus `upstream_localization.py`), `upstream_localization.py` →
    `4658569…`, and `block0_completion.py` → `16d310f…`. They are written out literally, never inherited.
  - They are recorded in the calibration record, the state and the lock.
- **Inherited inputs**, asserted before use:
  - 022's committed calibration record and confirmation (`b0c.verify_022_inputs`).
  - Experiment 023's committed files (`INHERITED_023`):
    - the exposed-cells data (file `d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4`);
    - its index (file `628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe`, content
      `d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16`);
    - the confirmation (file `5fadfa503f4fe35308cb4473220a1d9cd6f7a46e3852f638594b8081909825f4`, content
      `4e64d4c2c85ae8f9c710171372a4bf28f0964a8e8864a0326182edba44aa14ed`);
    - the lock (file `4bd5a5b14627768d49398c273fa1ce387db2e4739d7d31b7258072ef48beac9c`, content
      `97ca520f342e212d5172b1476e9d5a80c6c2622f80a4f99f4e15e0a399007117`), as 023's closure anchor.
  - 020's closure and locked states, through `ul.load_frozen_inputs` as 022 and 023 load them. The hash-pinned 020
    results state is archived in 023's evidence.
- **Never opened:**
  - 022's local calibration table;
  - 023's local outputs (its results state, stage-2 measurements and Y2 table).

  A tier-B test makes both unreadable during every 024 phase, and every phase still runs.
- **Runtime.** CPU, float32, 4 threads, the pinned checkpoint and library versions. At every phase that loads the
  weights they must equal 020's explore record (022's `_check_runtime`).
- **No fresh noun and no new frame.** The 79 scorable exposed nouns; the 108 exposed frames with 020's locked reference
  states. There is no stage 1.
- **Spent sets never enter.** 020's ledger, 020's confirmation set (021's spent set), 022's manifest and 023's manifest
  are forbidden in 024's ledger.
- **Weights only, with the no-forward-pass guard:** the weight-reading steps of `calibrate`, `lock` and confirm's I7.
  Every capture and intervention entry point is refused there.

## Constants (a tier-A test pins every one)

- **Design and plan.**
  - `DESIGN = {"path": "docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md",
    "revision": 2, "commit": "9d03dee"}`;
  - `PLAN = {"path": …, "revision": 1, "commit": <this plan's commit>}`.
- **Paths.** `EXPERIMENT = "024"`; `EXPERIMENT_DIR = "experiments/024-readout-routing-nounness"`.
- **Classes.**
  - `CLASSES = ("N", "B", "D", "C", "E")`, with the design's content descriptions; `CLASS_QUOTA = 8`; `N_FRESH = 40`.
  - `GUARD_CLASSES = ("E", "N")`: E first, then N.
- **Candidate lists**, the design's ordered lists verbatim, with each lemma's two forms written out (never generated by
  a rule):
  - `N_CANDIDATES`: eager, fierce, honest, polite, rude, sleepy, wise, lucky, merry, nervous, anxious, cheerful, clumsy,
    curious, grumpy, jealous, lonely, nasty, careful, careless, famous, friendly, gorgeous, hungry, thirsty, weary,
    wicked, ugly, vivid, vague, rapid, rigid, clever, fuzzy, latest, earliest, brave.
  - `MEASURE_LEMMAS`:
    - gallon/gallons, ounce/ounces, acre/acres, pint/pints, quart/quarts, herd/herds;
    - flock/flocks, swarm/swarms, crowd/crowds, bundle/bundles, bunch/bunches, heap/heaps;
    - mound/mounds, cluster/clusters, handful/handfuls, litre/litres, liter/liters, dozen/dozens;
    - barrel/barrels, bucket/buckets, sack/sacks, crate/crates, basket/baskets, carton/cartons.
  - `ORDINARY_LEMMAS`:
    - apple/apples, horse/horses, doctor/doctors, king/kings, teacher/teachers, rabbit/rabbits, poet/poets;
    - tiger/tigers, castle/castles, pencil/pencils, dragon/dragons, lion/lions, farmer/farmers, soldier/soldiers;
    - sailor/sailors, priest/priests, queen/queens, baker/bakers, knight/knights, lemon/lemons, banana/bananas;
    - onion/onions, carrot/carrots, violin/violins, wizard/wizards, pirate/pirates, tourist/tourists, statue/statues.
  - B is the plural and D the singular of the first eight eligible measure lemmas. C is the plural and E the singular of
    the first eight eligible ordinary lemmas.
- **`EXPECTED_PICKS`**, the design's:
  - N: honest, polite, rude, sleepy, wise, lucky, merry, nervous;
  - B/D lemmas: gallon, ounce, acre, herd, crowd, bundle, cluster, litre;
  - C/E lemmas: apple, horse, doctor, king, rabbit, poet, dragon, lion.
- **Primary calibration.** `B = 10_000`; `PRIMARY_TAG = "024|primary"`; `LOWER_RANK = 250`, element `[249]`, from
  `lower_rank(B) = (25·B + 999) // 1000`; `UNDEFINED_STOP = 250`.
- **The null.** `P_NULL = 100_000`; `NULL_TAG = "024|null"`; `NULL_RANK = 97_500`, element `[97499]`, from
  `null_rank(P) = (975·P + 999) // 1000`.
- **The bootstrap.** `B_CONTRAST = 10_000`; `CONTRAST_TAG = "024|contrast"`; elements `[249]` and `[9750]`, that is
  `[lower_rank − 1]` and `[B − lower_rank]`.
- **The E–N guard.** `EN_GROUP = 8`; `EN_ASSIGNMENTS = 12_870 = C(16, 8)`; `EN_MAX_UPPER = 321 = (25 · 12_870) // 1000`.
- **Results.**
  - `PRIMARY_RESULTS = ("NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS")`.
  - `GUARD_RESULTS = ("NOT_INTERPRETABLE", "FAIL", "PASS")`.
  - `OUTCOMES = ("NOT_INTERPRETABLE", "NOUNNESS_PREDICTION_NOT_ESTABLISHED",
    "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED",
    "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS")`.
- **`TOLERANCES`:**
  - `I1` 1e-4; `I3` 1e-4 (relative); `I4` 1e-3;
  - `C_recompute` 0.0 (bit for bit); `I7` 0.0 (bit for bit);
  - `spearman` 1e-12; `en_float` 1e-12; `mse` 1e-12 (relative);
  - `level1` 2e-2, descriptive only.
- **Groups and counts.** `GROUPS = ("cue_final", "coordinated")`; `N_CALIBRATION = 139` (45 / 45 / 49);
  `N_PRONOUN = 36`; `N_FRAMES = 108` (72 cue-final, 36 coordinated).

## The canonical computations (one implementation each, shared by calibrate, lock and confirm)

**1. Cells and the per-cue MSE.**
- **A pair's cells** are `b0c.pair_cells(y, C, C, C)`: 023's own cell function, with `y` the measured `Δc`. 024 has no
  `P0` or `P1`, so the placeholders make the SSE0 and SSE1 columns equal SSEC; they are never read. Only `n`, `SSEC`
  and, descriptively, `S` and `Q` are.
- **`cue_mse(cells)`** is `cells[:, SSEC].sum() / cells[:, n].sum()` in torch float64.
  - The cue's pairs are in `frame_id` order.
  - Calibration reads them from 023's artifact (cue-major, frames by `frame_id`). Confirm builds them from the saved
    measurements and reorders them to `frame_id` before summing.
  - The cue-final (72) and coordinated (36) subsets use the same function on the group's rows.
- **`log_mse(m)`** is `math.log(m)`. It is defined only for a finite `m > 0`.

**2. The nounness score (weights only).**
- **The embedding.**
  - `W_E` is `pm.Weights.from_model(model).W_E`: the float32 copy that `ul.ModelPrograms` also holds, read as
    `.double()`.
  - The binding is the sha256 of its little-endian float32 bytes.
- **The rows.**
  - `noun_row_ids(inputs)` gives 158 ids: the 79 scorable exposed nouns in pool order (`single_token`), each
    `(sg_ids[0], pl_ids[0])`.
  - `calibration_cue_ids(units)` gives the 139 cues of `b0c.STRATA` in `b0c.exposed_units` order.
  - Both lists and their digests are bound.
- **The functions.**
  - `centroid(W_E, ids)` is `torch.stack([W_E[i].double() for i in ids]).mean(0)`.
  - `cosine(a, b)` is `torch.dot(a, b) / (vector_norm(a) · vector_norm(b))`, in float64.
  - `nounness(e) = cosine(e, μ_noun) − cosine(e, μ_cue)`.
- **Leave one out.** A calibration cue uses `centroid` over the other 138 ids in order: a direct mean, never a
  subtraction from the sum. A pronoun cue (descriptive only) and every fresh cue use the full `μ_cue`.
- **The centroid bindings:** the sha256 of `μ_noun` and of the full `μ_cue`, as float64 little-endian bytes.

**3. Spearman.**
- **`average_ranks(values)`.** One-based ascending ranks. Exactly equal float64 values share the mean of their
  positions: sort by `(value, index)`, then walk the runs of equal values.
- **`spearman(x, y)`.**
  - The Pearson correlation of the two rank vectors, in float64: `Σab / sqrt(Σa² · Σb²)` on the centred ranks.
  - It is `None` when `Σa² = 0` or `Σb² = 0`, that is when a rank vector is constant.
  - Non-finite inputs are refused.
  - The same function serves every calibration draw, every null permutation, the fresh `ρ` and every secondary
    Spearman.
- **`spearman_direct(x, y)`** is an independent route, used only for the cross-checks. Its ranks come from pairwise
  counting, `1 + #{v_j < v_i} + (#{v_j = v_i} − 1)/2`, and its Pearson from `math.fsum`. The cross-checks are the first
  16 calibration draws and the fresh `ρ`, each within 1e-12.

**4. The exposed-data OLS line (secondary; never in the primary).**
- **`ols(x, y)`** uses `math.fsum` throughout:
  - `x̄`, `ȳ`, `Sxx = Σ(x − x̄)²` and `Sxy = Σ(x − x̄)(y − ȳ)`;
  - slope `Sxy / Sxx` and intercept `ȳ − slope · x̄`;
  - residual sd `sqrt(Σ r² / (n − 2))`.
- **Recorded values.**
  - The calibration record holds the full-precision floats, and the lock copies them.
  - The predictions are `intercept + slope · nounness`.
  - The design's rounded diagnostics never appear in the code; a tier-A test scans the module source for them.

**5. The SHA indices.**
- **Primary draws.** `primary_draw(b, slot)` is `int.from_bytes(sha256(f"024|primary|{b}|{slot}")[:8], "big") % 139`,
  indexing the 139 calibration cues in canonical order. There are `B × 40` of them, with replacement.
- **Null permutations.** `null_permutation(p, n = 40)` starts from `[0, …, n − 1]`. For `i` from `n − 1` down to 1 it
  sets `j = int.from_bytes(sha256(f"024|null|{p}|{i}")[:8], "big") % (i + 1)` and swaps positions `i` and `j`.
- **Bootstrap draws.** `contrast_draw(b, cls, slot)` is
  `int.from_bytes(sha256(f"024|contrast|{b}|{cls}|{slot}")[:8], "big") % 8`.
- **Digests.** The draw indices and permutations are digested, and the digests go into the calibration record.

**6. Order statistics.**
- The element is `rank − 1` after an ascending sort (`ul.order_statistic`).
- An undefined draw is placed at `−∞`.
- `defined_median` gives the direction check.

**7. The primary classification.** `classify_primary(ρ, F, null)`, in precedence order:
- `None` is `NOT_INTERPRETABLE`;
- `ρ < null` is `GUARD_FAILURE`;
- `ρ < F` is `ENVELOPE_ONLY_FAILURE`;
- otherwise it is `PASS`.

The fresh `ρ` and the calibration draws' descriptive rates both use it.

**8. The E–N guard: an exact one-sided permutation test.**
- **Input.** The 8 E `log MSE` values, then the 8 N values, each class in its frozen order. If any MSE is not finite and
  positive, the result is `NOT_INTERPRETABLE`.
- **Exact integers.**
  - `(num_i, den_i) = v_i.as_integer_ratio()`. Every denominator is a power of two, so the common denominator `Q` is
    the largest of them.
  - `z_i = num_i · (Q / den_i)`, as Python integers.
- **The enumeration.** `itertools.combinations(range(16), 8)` gives every assignment; the first is the observed one,
  positions 0–7. `T(S) = Σ_{i∈S} z_i` is exact, and its order is the order of
  `D(S) = mean(group S) − mean(complement)`.
- **The test statistic.** `K = #{S : T(S) ≥ T(observed)}`: ties count, and the observed assignment counts itself. The
  exact one-sided p-value is `p = K / 12,870`, a `Fraction`.
- **The rule.**
  - **The guard passes iff `K ≤ 321`**, that is `p ≤ 321/12,870 ≈ 0.02494`.
  - 322 assignments fail, since `322/12,870 > 0.025`.
  - A tie counts against the guard.
  - With all 16 values tied, `K = 12,870` and the guard fails.
  - No float-sensitive ranking and no Monte Carlo.
- **Reported descriptively.**
  - `D_EN = (2·T(observed) − Σz) / (8Q)`, exactly, reported as a float.
  - The derived threshold `t`: the smallest enumerated `D` with at most 321 assignments at or above it, or `+∞` if there
    is none. Without a tie there, it is element `[12549]`.
  - Each group's mean MSE and mean `log MSE`, and `exp(D_EN)`.
- **Checks** (an incident if any is violated):
  - exactly 12,870 distinct subsets of size 8;
  - `T(complement) = Σz − T(S)` for every `S`, through the complement index;
  - `|float(D_EN) − (mean_E − mean_N)|` in plain float64 within 1e-12.

**9. The outcome** (`outcome(primary, guard)`), in precedence order:

| primary result | E–N guard | outcome |
|---|---|---|
| (an incident) | — | no result is written |
| `NOT_INTERPRETABLE` | any | `NOT_INTERPRETABLE` |
| `GUARD_FAILURE` or `ENVELOPE_ONLY_FAILURE` | any (descriptive only) | `NOUNNESS_PREDICTION_NOT_ESTABLISHED` |
| `PASS` | `FAIL` or `NOT_INTERPRETABLE` | `ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED` |
| `PASS` | `PASS` | `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS` |

- The guard is computed and recorded whatever the primary result.
- The secondary B/C/D/E contrasts are computed only after this and never reach it.

## Objects and naming (`rr`)

**Pure core, tier A.** Everything under "The canonical computations", plus:
- `en_guard(e_values, n_values) -> {result, K, p, D_EN, t, group means, checks}`;
- `outcome(primary, guard)`;
- `semantics()`: the frozen readings and the "not shown" list, including:
  - the word *simple*: the guard eliminates the two preregistered simple alternatives, plurality-only and
    measure-class-only;
  - it establishes neither nounness as a unique causal factor nor the absence of every correlated lexical property;
  - the extrapolation sentence (below).

**Inherited checks.**
- `verify_023_inputs(root)`: the seven `INHERITED_023` digests.
- `read_exposed_cells(root, inputs)`: `b0c.read_cells` with `b0c.cells_meta(units, noun_keys, record_022)`, plus the
  file digests against `INHERITED_023`.

**Freeze** (tokenizer and committed files only):
`freeze_payload(tokenizer, inputs, confirmation_022, confirmation_023)`.
- **The exclusion.**
  - `b0c.exclusion(inputs, confirmation_022, …)`, which must reproduce the `cue_token_ids_sha256` recorded in 023's
    committed confirmation (327 ids);
  - plus 023's 24 cue ids and its frames' cue ids (351 in all);
  - with the sources listed, `confirmation-023` among them.
- **The other sets:**
  - the target-noun form ids, every `sg_ids` and `pl_ids` of every pool noun (161);
  - the frame-token ids, every `prefix_ids` and `suffix_ids` of the 108 exposed frames (316).
- **Eligibility of a word.**
  - `pm._encode(tokenizer, " " + w)` is a single id;
  - the id is not excluded, not a target-noun form and not a frame token;
  - the id is not already picked.
- **Lemmas.** A lemma counts only when both of its forms are eligible and their ids are distinct.
- **Picks.** The first 8 eligible entries per list. The reserves and the rejected entries with their reasons are
  recorded.
- **Stops.** Each writes nothing and stops for review; neither is an incident.
  - A shortfall raises `FreezeShortfall`.
  - Picks that differ from `EXPECTED_PICKS` raise `FreezeDeviation`, because the approved power analysis assumed those
    picks.
- **The nounness score** is not computed at freeze.

**The confirmation object.** `Confirmation024` holds `reference_ids`, `exposed_frames` (the 108, in pool order) and
`tokens` (the 40: word, token id, class, lemma, form).
- `frames = ()`.
- `target_prompts` is every `pm.Prompt(frame, token_id, word)` over the exposed frames × the tokens.
- `manifest()` is `{"S2-TARGET": sorted keys}`, 4,320 of them; `manifest_keys()`; `counts()`.
- `class_tokens(cls)`.
- `ul.stage_two_022` consumes it unchanged, and its Y2 block is empty by construction.

**Calibration** (weights and the committed artifact; no forward pass):
- `calibration_inputs(cells, units, W_E, inputs)` gives:
  - the 139 cues' LOO nounness, MSE and `log MSE`;
  - the 36 pronoun cues' full-`μ_cue` nounness and MSE (descriptive);
  - the bindings.
- `mse_cross_check`: every cue's MSE from `cue_mse` against a `math.fsum` loop over the same rows, within 1e-12
  relative.
- `primary_draws(x, y)`: `B` draws, the Spearman values, `defined`, the index digest.
- `null_distribution(n = 40)`: `P_NULL` permutations, their Spearman against the identity, the permutation digest.
- `evaluate_calibration(...)`:
  - `F_ρ`, element `[249]`, with the undefined count and stop, the direction check (`F_ρ ≤` the median of the defined
    draws) and the tails;
  - `null₉₇.₅`, element `[97499]`;
  - the effective threshold `max(F_ρ, null₉₇.₅)` and which of the two binds;
  - the draws' rates under `classify_primary` (descriptive);
  - the line;
  - descriptive: the calibration Spearman over the 139, the within-stratum Spearman, and the pronoun out-of-fit check
    (`ρ`, the line's median |log error| and its mean signed error).
- `calibration_record(...)` and `verify_calibration_record(record)`.

**Lock** (weights only; no forward pass):
- `fresh_scores(W_E, confirmation, bindings)`: the 40 scores, with the full `μ_cue`.
- `line_predictions(record, scores)`.
- `extrapolation(record, scores)`: per cue, whether its score exceeds the maximum LOO score of the 139 calibration
  cues. The design expects 5 of the 8 E cues to.
- `build_lock(...)`, `render_preregistration(lock)` and `validate_lock(...)`: 023's discipline, with 024's bindings.

**Confirm.**
- `reproduce_lock_quantities(W_E, …)` is I7: the centroids, the 40 scores and the predictions recomputed bit for bit
  against the lock, before any prompt.
- The measurement is `ul.stage_two_022(model, progs, confirmation, {"Y1": states, "Y2": {}}, executed=…)`.
- `recompute_c(saved, progs, units, states)` recomputes `C` from the re-read saved `Δx3`; it must be `torch.equal` to
  the saved `C` for every pair.
- `target_gates(progs, confirmation, measured, states)`: I1, I3 and I4 per pair, with the same formulas as
  `b0c.target_gates` and nothing else.
  - I1: the block-0 decomposition against the measured `Δx1`.
  - I3: the full composition (`b0c.FULL_MASK`: 15 cue-final, 31 coordinated) against the measured `Δx3`, relative.
  - I4: that composition's `Δĉ` against `C`.
- `fresh_cue_cells(measured)` gives `[40, 108, 8]`, in confirmation cue order and `frame_id` order.
- `score(fresh_cells, lock)`:
  - the per-cue MSE and `log MSE`;
  - `ρ` with its `spearman_direct` check and its result;
  - the E–N guard with its checks;
  - the outcome, with the frozen reading.
- **The measured `Δx3` has one use: building readout comparators.** It builds `C` and, descriptively after the
  results, `C`'s Level-1 variants. It also feeds the I3 identity gate. It never enters the score, the thresholds, the
  guard's grouping or any prediction.

**Descriptives** (after the results; no outcome force):
- `secondary(measured, fresh_cells, lock)`:
  - the cue-final-only (72 frames) and coordinated-only (36) Spearman;
  - the normalized MSE, `Σ SSEC / Σ Q`, and its Spearman;
  - per cue: `Var(Δc)` and `R²_C`, both pooled through `b0c.pool`;
  - per cue: the bias `mean(Δc − C)` and the OLS slope of `Δc` on `C`;
  - the line's prediction against the observed `log MSE` (median |log error| and mean signed error, overall and per
    class) and the extrapolation flags.
- `contrasts(log_mse_by_class)`: the class means and the four frozen factorial contrasts, each with its
  `024|contrast` bootstrap interval, elements `[249]` and `[9750]`.
- `ladder(progs, measured, states)`, per class:
  - `C5` = `readout.contrast(state, readout.level1_detail(state, Δx3, l4_heads=False)["dh6"], nouns)[scorable]`;
  - `L1` = the same with `l4_heads=True`;
  - the share of `Σ(Δc − C)²` removed by `C5`, then by `L1`;
  - the Level-1 identity, `max |L1 − Δc|`, against 2e-2 nats, reported and not a gate.
- `render_report(state, record, arrays)`.

## Artifact schemas

Every JSON artifact is canonical JSON (`pm.canonical_json`) plus a newline. `content_sha256` is computed over the record
without that key (021–023's convention).

1. **`experiments/024-readout-routing-nounness/confirmation-v1.json`**, written by `freeze` and committed by hand.
   It holds:
   - the experiment, schema, design, plan and model;
   - the candidate lists verbatim and the rules;
   - the exclusion (ids, digest, sources), the target-noun form ids and the frame-token ids, each with a digest;
   - `cues`, the 40 with class, lemma, form, token id and candidate rank;
   - `reserves` and `rejected`, with reasons;
   - `expected_picks` and whether the picks match them (they must);
   - `exposed_frame_ids` and `reference_cue_ids`;
   - `counts`, 8 per class;
   - `manifest`, `{"S2-TARGET": 4,320 keys}`;
   - `content_sha256`.
2. **`outputs/experiment-024/results.json`**, the state, created by `calibrate`, in 023's schema style:
   - `run_id`, `inputs` (the frozen-input digests, 022's and 023's), `module_blobs`, `versions`;
   - `phases` (calibrate, lock, confirm, report);
   - `executed_prompt_keys`, empty until `confirm`;
   - `calibration`, `confirmation_024` (the binding), `lock`, `confirmation`, `report`.
3. **`outputs/experiment-024/calibration-arrays.pt`**: the draw Spearman values `[B]`, `defined` `[B]`, the draw
   indices `[B, 40]`, the null values `[P]` and the permutations `[P, 40]` (int8). Each is digested in the record.
4. **`experiments/024-readout-routing-nounness/calibration-v1.json`**, installed byte-identically from
   `outputs/experiment-024/candidate-calibration.json` and committed. It holds:
   - the design, plan, commits, inputs, module blobs and `constants`;
   - `exposed_cells`, 023's artifact binding; `confirmation_024`, the path with its file and content digests;
   - `score`: the embedding digest, both id lists with digests, and both centroid digests;
   - `calibration_cues`: 139 × {word, token id, stratum, LOO nounness, MSE, log MSE}; `pronoun_cues`: 36, descriptive;
   - `line`: slope, intercept, residual sd, `n`, formulas;
   - `primary_floor`: `F_ρ`, rank, element, undefined count, direction check, median, tails, digests;
   - `null`: `null₉₇.₅`, rank, element, median, digests;
   - `effective_threshold` and which one binds;
   - `checks`: the MSE and Spearman cross-checks;
   - `descriptive`: the calibration `ρ`, within-stratum `ρ`, the pronoun check and the draw rates.
5. **`preregistration-lock.json`** and **`preregistration.md`**, installed byte-identically and committed. The lock
   binds:
   - the calibration record's file and content digests, and the thresholds and line copied from it;
   - the confirmation's digests, counts and manifest digest;
   - 023's artifact digests;
   - the score bindings;
   - `fresh_cues`: 40 × {word, class, lemma, token id, nounness, predicted `log MSE`, above the calibration
     maximum}, with the digest of the float64 score vector;
   - the primary: statistic, `F_ρ`, `null₉₇.₅`, precedence, readings;
   - the E–N guard: the E and N cues in order, the natural log, the enumeration, 12,870, the bound 321, the tie rule,
     readings;
   - the outcome table and its readings;
   - the semantics, the noun keys, the exposed-states digest (`ul.exposed_states_digest`), the module blobs, the design
     and the plan.
6. **`outputs/experiment-024/stage2-measurements.pt`**: `ul.measurement_tensors` of the Y1 block, written with its
   digests before any gate. The Y2 block is empty and asserted so.
7. **`outputs/experiment-024/report.md`**.

## Phase control flow

- **`validate`** (no model). It checks:
  - the blobs and the inherited inputs (020, 022, 023);
  - once each exists, the confirmation, the record, the lock and the state, with ledger isolation.

  It opens neither 022's table nor 023's outputs.
- **`freeze`** (tokenizer only; the guard).
  - Refused if the confirmation exists.
  - It writes `confirmation-v1.json`, re-reads and verifies it, and checks the manifest's disjointness from the
    forbidden keys. It is committed by hand and independently reviewed.
- **`calibrate`** (once; weights only; the guard). It creates the state and requires:
  - the committed confirmation, tracked and verified;
  - 023's artifact, tracked and verified against `INHERITED_023` and its own index;
  - no forbidden key in the ledger.

  In order it computes:
  - the weights, the bindings, the scores and the MSE with their cross-checks;
  - the line;
  - the draws, where 250 or more undefined draws stop it for review;
  - `F_ρ` and the direction check, where a reversed direction stops it for review;
  - the null, the effective threshold and the descriptives;
  - the re-check of 020, 022 and 023;
  - the candidate record.

  A stop writes no record and is never retried automatically. The record is installed byte-identically, committed and
  reviewed before `lock`.
- **`lock`** (weights only; the guard). It requires the committed record and no scientific change since `calibrate`.
  - It verifies the record and recomputes the centroid digests from the weights against it.
  - It recomputes the line from the record's 139 entries and requires it bit-identical.
  - It builds the fresh scores, predictions and extrapolation flags twice and requires them bit-identical.
  - Then it writes the candidate lock and preregistration. Both are installed byte-identically, committed and
    independently reviewed.
- **`confirm`** (once; never resumed). 023's control flow, without stage 1:
  1. `validate_lock`: tracked files, a clean tree, no scientific change since the lock commit; then the runtime check
     and the isolation check.
  2. I7 under the guard, before any prompt. A mismatch is a phase incident, and confirm is refused until the reviewer
     decides.
  3. The 4,320 target keys are recorded in the ledger and the state is written.
  4. `ul.stage_two_022`: each target once, frame by frame in `frame_id` order, cues by token id.
  5. The measurements and their digests are saved before any gate.
  6. **The accounting:** the expected manifest = the executed `capture_prompt` calls (`Counter`) = the ledger, with no
     extra, missing or duplicated key. Every measurement is finite, and the Y2 block is empty.
  7. `C` recomputed from the re-read saved `Δx3`, bit for bit.
  8. `target_gates`, written to the state before enforcement (I1, I3, I4).
  9. `score`.
  10. The re-check of 020, 022 and 023, before any result is written.
  11. The per-cue table, the primary result, the guard result, the outcome and the completed phase, in one write.
  12. Then, with no outcome force, the secondary analyses, the contrasts and the ladder. A failure is recorded as such;
      an interruption is recorded and raised. Neither touches a result.

  An incident is recorded with its commit and stops the phase. Nothing is restored or retried.
- **`report`** renders `outputs/experiment-024/report.md`:
  - the freeze summary;
  - the calibration: `F_ρ`, `null₉₇.₅`, the effective threshold, the direction, the undefined count, the rates, the
    calibration and within-stratum `ρ`, the pronoun check and the line at full precision;
  - the primary: `ρ`, the thresholds, the result and its reading;
  - the E–N guard: `K`, `p`, `D_EN`, `t`, the group means, the result and its reading;
  - **the outcome** and its frozen reading, with the "not shown" list and the extrapolation sentence;
  - the secondary analyses, the contrasts with their intervals, and the ladder;
  - the gates and the accounting.

  If an incident is recorded, no result is shown.

**The extrapolation sentence (frozen, rendered in the preregistration and the report).** "k of the 8 E cues lie above
the calibration population's maximum nounness (the lock computes k; the design expects 5). The fresh test is a
prospective extrapolation test there. Calibration supplied no evidence for that range, and the post-hoc behaviour of
023's five spent cues in it is design motivation only."

The strongest outcome therefore reads: the frozen weight-derived score prospectively predicts the frozen readout's error
on new lexical representations, including substantial extrapolation beyond the exposed score range. It does not say
that the exposed OLS line has been validated over that range.

## Ledger and isolation

- 024 executes prompts only at `confirm`: 4,320 S2-TARGET keys, each once. Before `confirm` the ledger is empty.
- **Forbidden in 024's ledger**, checked by `freeze`, `validate`, `calibrate`, `lock` and `confirm`:
  - 020's ledger (`inputs.closure["ledger"]`);
  - 020's confirmation set, 021's spent set;
  - 022's manifest, from its committed confirmation;
  - 023's manifest, from its committed confirmation, which is 023's executed set.
- By construction every 024 key carries a new cue token, and the freeze and the loader assert it.

## Incidents, stops and failures

These follow 022 and 023.
- **Incidents.** They are recorded with the commit, stop the phase, carry no result and are never retried at that
  commit:
  - I1, I3 and I4;
  - the `C` recomputation and the accounting;
  - the Spearman and MSE cross-checks, and the E–N enumeration and float checks;
  - I7;
  - a failed re-check.
- **Stops for review, which are not incidents:**
  - a freeze shortfall or deviation;
  - a calibration stop (undefined draws or a reversed direction);
  - an artifact that does not verify, which is refused before anything is written.
- **Spending the version.** An interruption of `confirm` spends the protocol version.

## Testing

`tests/conftest.py`: `CURRENT_EXPERIMENT = "024"`. The 023 runner test becomes historical (tier D).

- **Tier A** (`tests/test_readout_routing.py`, pure, no model):
  - every constant, including the lists verbatim and `EXPECTED_PICKS`;
  - `cue_mse`: hand values, `frame_id` ordering, the group subsets, and the fresh cells equal to `b0c.pair_cells`;
  - the score on hand vectors; the LOO centroid as a direct 138-row mean (a subtraction variant differs and is never
    used);
  - Spearman:
    - hand cases, including ties, the average-rank convention and a constant vector giving `None`;
    - invariance under `log`;
    - agreement with `spearman_direct` within 1e-12 on random data with ties;
  - `ols` against hand values and `numpy.polyfit` within 1e-12; a source scan finds no rounded design diagnostic;
  - the SHA formulas: determinism, the first indices pinned, Fisher–Yates always giving a permutation, the three tags;
  - `lower_rank(10,000) = 250`, `null_rank(100,000) = 97,500`, the bootstrap elements 249 and 9750, `−∞` for an
    undefined draw, the stop at 250;
  - the precedence of `classify_primary`, for `F < null` and for `F > null`;
  - the E–N guard:
    - the decision at `K = 321` passes and at `322` fails;
    - all-tied fails with `K = 12,870`; E strictly above N gives `K = 1`; E below N gives `K = 12,870`;
    - a float64 false tie is resolved exactly;
    - the complement identity; `D_EN` exact against float64;
    - the untied threshold at element `[12549]`;
    - the size: exactly 321 of the 12,870 possible observed assignments pass on untied data, and never more with ties;
    - a non-finite value gives `NOT_INTERPRETABLE`;
  - the outcome table, exhaustive: 4 primary results × 3 guard results;
  - the freeze on a stub tokenizer: each rejection reason (multi-token, used id, target-noun form, frame token), the
    lemma pairing, the shortfall, the deviation stop, the manifest;
  - the record and the lock refusing tampering;
  - the artifact reader refusing a digest mismatch;
  - the committed 023 artifact read and its 175 per-cue MSE reproduced by a direct loop. These are exposed values
    only; no floor is computed.
- **Tier B** (`tests/test_experiment_024_runner.py`). The six-layer fake, inside the fake world of 023's runner test:
  - **The world.** 023 is extracted, frozen, calibrated and locked on the fake by its own runner. `INHERITED_023`
    points at those files, and a toy tokenizer adds the 024 candidates.
  - **Sizes.**
    - The class quota is 4 (20 cues), so the guard enumerates `C(8, 4) = 70` assignments with the bound
      `(25·70)//1000 = 1`.
    - `B = 40`, `P_NULL = 400` and `B_CONTRAST = 40`.
    - The lock binds these sizes, and production pins 8, 12,870 and 321.
  - **Every phase and every refusal:**
    - a second `freeze`, `calibrate` or `confirm`;
    - `calibrate` without the committed confirmation or artifact, or with an altered or foreign one;
    - `lock` with a record that is missing, untracked or not the candidate, or with a different confirmation;
    - `confirm` with tampered, untracked or changed lock files.
  - **The stops:** the undefined-draw and direction stops write no record and are never retried.
  - **The boundaries:**
    - no forward pass in `calibrate`, `lock` or I7;
    - every phase runs with 022's table and 023's outputs unreadable;
    - a forbidden key in the ledger is refused.
  - **Confirm:**
    - the accounting holds;
    - the measurements are on disk before any gate;
    - an I7 drift is refused before any prompt;
    - a `C`-recompute mismatch and a stage-2 identity failure are incidents that keep every measurement;
    - a failed re-check after scoring carries no result;
    - the results and the completed phase are written in one write;
    - a descriptive failure or an interruption keeps the results.
  - **Every outcome** is reached end to end by wrapping the per-cue MSE (the gates stay real).
  - **The report**, including the withholding of results beside an incident.
- **Tier C** (the pinned model, `NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1`). No 024 fresh prompt runs in any test.
  - **The freeze:** the real-tokenizer contract gives the design's expected picks and reserves, and the ✗ marks with
    their reasons (teacher and farmer are frame tokens; so are crate and basket, whose plurals are also multi-token).
  - **The scores.**
    - The real scores match design background item 8 to three decimals: E from +0.108 (lion) to +0.298 (horse), mean
      +0.174; N from −0.229 (honest) to −0.045 (nervous), mean −0.146. Every E scores above every N.
    - 5 of the 8 E scores lie above the maximum calibration LOO score. This is weights only; no MSE or floor is
      computed.
    - The centroids have 158 and 139 rows.
  - **The measurement path:** for a deterministic handful of exposed (spent) pairs, re-measured through
    `ul.measure_prompt` and `ul.contrast_of`, `b0c.pair_cells(y, C, C, C)` reproduces the artifact's `n` and `SSEC`
    bit for bit. This is the one-canonical-MSE contract between calibration and confirm. A difference stops for review.

## Runtime and storage (estimates on this machine)

- `freeze`: seconds.
- `calibrate` ≈ 1–2 min: the weights, 175 scores, 10,000 draws, and 100,000 permutations with their Spearman (≈ 4M
  SHA-256). The arrays are ≈ 5 MB in `outputs/`.
- `lock` ≈ 1 min: weights only, scores twice.
- `confirm` ≈ 20 min:
  - 4,320 forwards ≈ 3 min;
  - the `C` recomputation ≈ 1–2 min;
  - the gates, the full composition for 4,320 pairs, ≈ 5 min;
  - scoring, seconds;
  - the ladder ≈ 8 min.

  The measurements are ≈ 75 MB in `outputs/`.

## File map

- `src/neural_decompiler/readout_routing.py` (new).
- `experiments/024-readout-routing-nounness/run.py` and `README.md` (new).
- Committed artifacts in `experiments/024-readout-routing-nounness/`: `confirmation-v1.json`, `calibration-v1.json`,
  `preregistration-lock.json`, `preregistration.md`.
- `tests/test_readout_routing.py` and `tests/test_experiment_024_runner.py` (new).
- `tests/conftest.py` (`CURRENT_EXPERIMENT`); `.gitignore` (`outputs/experiment-024/*`).
- The root `README.md` registry entry, at closure only.

## Tasks (small reviewable commits; the checkboxes are for execution)

**Task 1: the skeleton.**
- [ ] `.gitignore`; the module's constants, the blob checks, and the 022 and 023 inherited-input checks; tier-A pins.
- [ ] Commit: "feat: experiment 024 module skeleton, constants and frozen-input checks".

**Task 2: the pure core.**
- [ ] `cue_mse`, `log_mse`, the score functions, `average_ranks`, `spearman` and `spearman_direct`, `ols`, the SHA
  indices and Fisher–Yates, the ranks, `classify_primary`, `en_guard`, `outcome` and `semantics`.
- [ ] Every tier-A test of the core, including the guard's size and tie tests and the outcome table.
- [ ] Commit: "feat: experiment 024 canonical MSE, score, Spearman, line, exact E–N guard and outcome (pure core)".

**Task 3: the freeze.**
- [ ] The lists, the exclusion (023's plus 023's cues), the target-noun forms, the frame-token rule, the lemma pairing,
  the expected-picks check, the manifest, the shortfall and deviation stops.
- [ ] Stub- and real-tokenizer tests.
- [ ] Commit: "feat: experiment 024 tokenizer-only freeze of 5 × 8 cues".

**Task 4: the calibration.**
- [ ] The artifact read, the per-cue MSE with its cross-check, the LOO scores and bindings, the line, the draws, the
  null, the stops, the descriptives and the record.
- [ ] A synthetic full-scale dry run with no model: 139 cues, B = 10,000 and P = 100,000, checking time and memory.
- [ ] Commit: "feat: experiment 024 calibration of F_ρ and the n = 40 null from the committed exposed cells".

**Task 5: the lock.**
- [ ] The fresh scores, predictions and extrapolation flags, `build_lock`, `validate_lock` and the preregistration.
- [ ] Commit: "feat: experiment 024 lock and preregistration".

**Task 6: confirm, report and runner.**
- [ ] I7; the ledger; the measurement through `ul.stage_two_022`; the accounting; the `C` recomputation; the gates;
  `score`; the descriptives (secondary, contrasts, ladder); `render_report`.
- [ ] The runner with six phases, the fake-world runner tests, `CURRENT_EXPERIMENT` and the experiment README.
- [ ] Commit: "feat: experiment 024 confirmation, report and runner".

**Task 7: review and gate.**
- [ ] An independent implementation and data-flow review against design revision 2 and this plan; fixes in separate
  commits.
- [ ] Tier B, then tier C on the clean gated commit. **Stop.**

## Stopping conditions

- Stop after this plan, for the implementation-plan review.
- After approval: implement Tasks 1–7 and stop for the implementation review.
- Every scientific phase (`freeze`, `calibrate`, `lock`, `confirm` and `report`) runs only when separately authorized.
