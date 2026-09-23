# Experiment 021 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 021 design (revision 3, commit `0ac46aa`, passed final design review on
2026-09-22): re-materialize Experiment 020's exposed table from exactly its 30,132 ledger keys, verify it against 020's
record at `1e-9`, compute the 149 floor rows from 10,000 SHA-indexed base draws once, stop for the floor review, then
lock (no forward pass) and confirm once on the untouched 020 confirmation set, with the Y2 row selected at stage 1
and the Y3 row by measured scorability.

**Architecture:** One new module `readout_calibration.py` that *calls* Experiment 020's `readout_decompilation.py`
(blob `caa73b40…`, never edited) for everything the program, the measurement, the identities, the stage-1 procedure
and the 020 statistics already define, and adds only what revision 3 adds: the 020-closure and program-blob checks,
the re-materialization loop and its reproduction gate, the pools, the draws, the one statistics kernel used by both
`calibrate` and `confirm`, the floor rule, the shared pass predicate, the row tables and their selection, a stage 2
that also keeps the measured `Δx3` for the ceiling, the lock and the report. A new runner
`experiments/021-corrected-readout-confirmation/run.py` follows 020's runner (refusal rules, incident gating,
changed-scientific-path refusal, runtime equality, A0 contract test) with phases `validate`, `calibrate`, `lock`,
`confirm`, `report`.

**Spec:** `docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md` (revision 3,
`0ac46aa`). **The spec wins over this plan.** Planning changes nothing scientific: not the calibration rule, the
populations, the draws, the floors, the thresholds, the composition tables, the pass predicates, the labels, the
fresh set, the model or the readout program.

## Global constraints (frozen by revision 3)

- **The program is Experiment 020's, byte for byte.** `src/neural_decompiler/readout_decompilation.py` must have git
  blob `caa73b40192f4c910dc63371bd19db75a3258339` (computed in code as `sha1(b"blob <len>\0" + bytes)`; no git call);
  every phase and a tier-A test refuse otherwise. No 021 code reimplements any of its functions; where 021 needs a
  variant (stage 2 with the ceiling), it calls the same 020 functions in the same order and a test proves the shared
  outputs identical to 020's own function.
- **Inherited objects** read verbatim with their digests: locks 011 `769bfeac…`, 012 `830abc3b…`, 017 `b4fc9014…`,
  and the whole input chain 006–019 through 020's `_base_inputs` logic (copied into the 021 runner unchanged).
- **The confirmation set** is `experiments/020-readout-decompilation/confirmation-v1.json`, digest `e098e2b4…`, loaded
  with `rd.load_confirmation` in place; never copied, rewritten, filtered, reordered or selected from.
  `calibrate` reads only its manifest keys (to exclude them) and its class counts 6/6/6/6, 6/6/6, 8/8/8.
- **Experiment 020's closure** is verified before and after every phase: `closure.json` content sha256 `f2b1b5e7…`,
  the committed extract `evidence/exploration-record-2026-09-22.json` content sha256 `99f25ee6…`, the local results
  state file sha256 `da63b8c2…` and state sha256 `2e5485dc…`, `lock` and `confirmation` null, 0 of 3060 manifest keys
  and 0 of 24 fresh nouns in its ledger. A missing or different file is a refusal before anything else runs.
- **No fresh or confirmation prompt before `confirm`**, and no fresh-noun quantity before `lock` (the lock's fresh-noun
  columns are weight-only predictions, as in 020). `calibrate` executes exactly the set of 020's 30,132 ledger keys.
- **Runtime:** CPU float32, `torch.get_num_threads() == 4`, seeds `20260916` / `20260924`, and the dependency versions
  of 020's explore record (torch 2.14.0, transformers 5.17.0, transformer-lens 3.9.0, huggingface-hub 1.31.0, Python
  3.12.13); `calibrate`, `lock` and `confirm` refuse on any difference. `HF_HUB_OFFLINE=1` with the cached weights.
- **Incident semantics** as in 020: scientific invalidity feeds `PRECONDITION_FAILED_*` and the row selection;
  implementation or protocol failures are incidents that stop the phase, are recorded with their commit, and block a
  rerun at that commit. The calibration precondition (≥ 6 screened frames per template) is a terminal stop for review,
  not an incident.

## Constants (copied from revision 3; a tier-A test pins every one)

| constant | value | source |
|---|---|---|
| `B` | 10 000 base draws | floor rule |
| `TAIL_INDEX` | 250 = ⌈0.025·B⌉, computed from `B` (the 250th smallest for ≥, the 250th largest for ≤) | floor rule |
| `R2_STATISTICS` / `ERROR_STATISTICS` | S1, S2, S5–S10 / S3, S4, S11, S12 | statistics table |
| `SHARES` | 80 (cues), 75 (frames), 90 (nouns); `k = (p·n + 99) // 100` | statistics |
| `SLOTS` | 6 per cue class, 6 per frame template, 8 per noun rule class | draw construction |
| `VARIANTS` | `primary` (floors), `all-frames`, `lineage` (descriptive only) | draw construction / descriptives |
| `RECONSTRUCTION_TOLERANCE` | 1e-9 absolute, every numeric leaf | reproduction gate |
| `LOCKED_STATE_TOLERANCE` | 1e-9 (the 013–019 constant) | environment check |
| `MIN_SCREENED_FRAMES_PER_TEMPLATE` | 6 | calibration precondition |
| 020 tolerances, preconditions, labels, comparator standing | imported from `rd`, never redefined | preserved |

**Implementation-only constants (no scientific force; flagged for the implementation review).**
`STATISTIC_AGREEMENT_TOLERANCE = 1e-10` — the vectorized kernel against 020's direct `pair_statistics` /
`noun_statistics`, on the fresh tables at `confirm` and on the first 16 base draws of every row at `calibrate`
(float64 reassociation only; an excess is an incident). `CROSS_CHECK_DRAWS = 16`.

## Data flow and leakage boundaries (what each phase may read and write; the test that enforces it)

| phase | may read | may execute (model) | writes | must not | enforced by |
|---|---|---|---|---|---|
| `validate` | frozen inputs, locks, the 020 confirmation file, 020's closure record, extract and local results state (digests only) | nothing | nothing | load a model | test: `validate` passes with a `model_loader` that raises; tampered 020 digests refused |
| `calibrate` (once) | the above; 020's results state (read only: its ledger, `locked_states`, `template_bases`, recorded statistics) | the 108 exposed reference prompts and the 30,024 exposed cue prompts — as a set exactly 020's ledger, in 020's explore order | 021 state: ledger, A0, identity maxima, environment check, reproduction gate, validity screen, precondition, table digests, index-array digests, per-row draw-value digests, the candidate calibration record's sha256; files under `outputs/experiment-021/` | run a key outside 020's ledger or inside the manifest; run a key twice; build any fresh-noun quantity; draw anything before the gate passes; write 020's state | tests: capture spy ⇒ executed set == 020's ledger, multiplicity 1; planted foreign key ⇒ incident; fresh ids absent from state and record; 020 file digests unchanged after; planted `1e-8` gate difference ⇒ incident with no draw computed |
| install | the candidate record | nothing | `experiments/021-…/calibration-v1.json`, byte-identical, committed; **stop for the floor review** | edit the record | `lock` refuses unless tracked, committed and byte-identical to the sha256 in the state |
| `lock` | the committed calibration record, 020's `locked_states` and `template_bases`, the locks, the weights | **nothing** | candidate lock (Y1 prediction rows 2592 × (79 + 24 fresh-noun columns) + comparator columns, 020's locked states, template bases, rank-1 and ΔT objects, the 149-row floor tables, the calibration record sha256, the confirmation digest and manifest) + `candidate-predictions.md` | any forward pass; any fresh measured quantity | test: every capture/intervention entry point raises while `lock` runs; provenance invariant recomputed under the guard |
| `confirm` stage 1 | the installed lock (byte-identical, committed), the confirmation file, the weights | S1-REF and S1-VALIDITY only, once each (`rd.stage_one`) | 020's stage-1 record + `y2_selection` {composition, precondition, row floors} + `y2_selection_sha256` over (selection, stage-1 digest) | run any S2-TARGET key; select the Y2 row from anything but the validity verdicts | tests: spy sees only S1 token ids; the selection function receives the verdicts only |
| barrier | the results state on disk | nothing | — | proceed unless the stage-1 digest and the selection digest re-verify, the composition recomputed from the re-read verdicts equals the recorded one, its row equals the lock's row, the stage-1 rows reproduce exactly from the digested states (`rd.reproduce_stage_one_rows`), and no target key is in either ledger | tests: tampered selection, composition or row on disk ⇒ refusal before any target runs |
| `confirm` stage 2 | the lock, the digested stage-1 record, the weights | S2-TARGET: the 2592 Y1 prompts, then the Y2 prompts of the valid fresh frames, once each | measured contrasts (103 nouns), measured `Δx3` → ceiling on the 79 exposed nouns, identities, the Y3 row selected by scorability, scoring with the selected rows, comparators | run a Y2 key of an invalid frame; score a fresh noun into Y1/Y2; select the Y3 row from predictions | tests: invalid frames' keys never in the ledger; Y3 selection invariant under any change of the predictions; stage-2 equivalence with `rd.stage_two` |
| `report` | the results state; the per-row draw values under `outputs/experiment-021/draws/` (digest-verified) | nothing | `outputs/experiment-021/report.md` | — | test: a tampered draw file is refused |

Before `lock` and before `confirm`, both 020's ledger and 021's own are asserted to hold no manifest key.

## Objects and naming (`src/neural_decompiler/readout_calibration.py`, imported as `rc`)

- **Checks.** `program_blob_sha1(path)`, `assert_program_blob(root)`; `verify_020_closure(root, confirmation) -> dict`
  (every digest above plus the isolation counts; returns 020's exploration record for read-only use).
- **Pools.** `CalibrationPools` with `cues[class] -> tuple[(word, token_id)]` (token-id order), `frames_all[template]`,
  `frames_unscreened[template]` (origins 017/018/019, `frame_id` order), `frames[template]` (after the screen),
  `nouns[rule] -> tuple[lexical_key]` (key order), and the `lineage` and `all-frames` variants;
  `build_pools(pool, screen=None)`. Membership is by frozen provenance only (`token_category`, `token_source`,
  `frame_origin`, `rule_class`, `single_token`), asserted: 45/45/36/49, 14/14/14, 40/19/20; no reference or plural cue;
  the 006 cohort's `quantity` words excluded.
- **Rows.** `y2_rows()` and `y3_rows()` in lexicographic order, enumerated from the frozen preconditions themselves
  (`rd.MIN_VALID_FRESH_FRAMES`, `rd.MIN_VALID_COORDINATED_FRAMES`, `rd.MIN_SCORABLE_FRESH_NOUNS`) and the confirmation's
  per-stratum maxima (6 frames per template, 8 nouns per rule class) — 64 and 84 with the real constants, pinned by a
  test; `k_of(p, n)`; `Row` = (outcome, composition, n, k values).
- **Draws.** `slot_index(variant, b, stratum, slot, n) = int.from_bytes(hashlib.sha256(f"021|{variant}|{b}|{stratum}|{slot}"
  .encode("utf-8")).digest()[:8], "big") % n`, with `stratum` exactly `cue/determiner-like`, `cue/quantity`,
  `cue/possessive-or-pronoun`, `cue/adjective`, `frame/cardinal`, `frame/quantifier`, `frame/coordinated-adjective`,
  `noun/simple-suffix`, `noun/sibilant-es`, `noun/consonant-y`; `draw_indices(variant, pools) -> {stratum:
  LongTensor[B, slots]}`; `index_digest(tensor)` = sha256 of the int64 little-endian row-major bytes plus the shape.
  Index arrays are stored as digests only.
- **Exposed table.** `ExposedTable` (pair labels cue/frame/template in 020's order; `measured`, `level0`, `ceiling`,
  `no_l5`, `base` `[30 024, 79]` float64; `dT` `[30 024]`; the plural-cue `PairMeasurement` of every frame for the
  screen); `tensor_digest(t)` as above; saved with `torch.save` under `outputs/experiment-021/`.
- **Re-materialization.** `rematerialize(model, pool, lock_011, lock_012, lock_017, exploration_020, ledger_020,
  manifest_keys, *, log) -> (ExposedTable, record)`: `rd.run_exploration`'s two loops with the same calls in the same
  order — first the 108 reference captures (`capture_frame_020`, `atp.reference_rows`), all 108 reference keys checked
  against 020's ledger and the manifest before the first forward; then, frame by frame with that frame's cue keys
  checked first, `measure_pair`, `predict_pair` with 020's recorded `template_bases`, `pair_identities`,
  `rd.level1_breakdown` only when the running `E₁` maximum moves (as in 020, recorded as `level1_worst`),
  `inherited_reproduction`, `ceiling_prediction` — plus the environment check between the loops (each re-captured
  `locked_state(...)` against 020's entry at `LOCKED_STATE_TOLERANCE`; the recomputed `template_bases_020` against the
  recorded ones at the same tolerance) and `rd.enforce_all` on the identity maxima.
- **Gate.** `reproduction_gate(table, exploration_020) -> dict` recomputes `rd.pair_statistics`, `rd.noun_statistics`,
  the ceiling / no-L5 / template-base flattened `R²` and `rd.dT_only_fit` (`r2` and `noun_vector`), walks every numeric
  leaf against 020's record, and raises `pm.IncidentError` above `RECONSTRUCTION_TOLERANCE`; it also asserts the key
  sets of `per_cue`, `per_frame`, `per_template` and `nouns` are identical.
- **Screen.** `validity_screen(program, states, nouns, table, axis_T) -> {frame_id: verdict}` for all 108 frames via
  `rd.frame_validity` on each frame's plural-cue measurement; `precondition(pools)`.
- **Kernel (the one statistics implementation).** `PairSums.from_table(y, x, labels)` keeps, per pair, over its noun
  columns: Σy, Σy², Σx, Σ(y − x)², Σ|y − x|, n; `NounCueSums` keeps, per (noun, cue) over frames: Σy, Σy², Σx, Σxy,
  Σ(y − x)², n. `y1_statistics(sums, cue_multiplicity)`, `y2_statistics(sums, cue_multiplicity, frame_multiset)`,
  `y3_statistics(noun_sums, noun_multiset, cue_multiplicity)` return S1–S4, S5–S8, S9–S12 exactly as the statistics
  table defines them (undefined ⇒ `None`), duplicates counting as separate units. Used by `calibrate` on the draws and
  by `confirm` on the fresh tables (unit multiplicities).
- **Direct cross-check.** `direct_statistics(table, ...)` materializes a table — duplicated cues, frames **and nouns**
  relabelled `unit#slot`, because 020's `ScoringTable.by` merges equal labels and `noun_statistics` keys its result by
  noun — and computes the same S_j with `rd.pair_statistics`, `rd.noun_statistics` and the order statistics;
  `assert_agreement(kernel, direct)` at `STATISTIC_AGREEMENT_TOLERANCE`.
- **Floors and the predicate.** `tail_floor(values, kind) -> (floor, clamped)` (250th smallest / largest, `None` as
  −∞ / +∞, `max(·, 0.0)` for `R²`-type, float64, no rounding); **`passes(kind, value, floor) -> bool`** — `R²`:
  `value is not None and math.isfinite(value) and value > 0.0 and value >= floor`; error:
  `value is not None and math.isfinite(value) and value <= floor`. It is the only pass/fail function in the module;
  `pass_rates(...)` and `score_021(...)` both call it.
- **Calibration.** `calibrate_rows(table, pools, indices) -> rows` (Y1 1, Y2 64, Y3 84; each row's floors, clamp flags,
  median and 250th-smallest/largest points, per-condition and joint pass rates, and its draw values' digest), the
  full-row joint rate over all three outcomes, the ceiling distributions (Y1, every Y2 row), the three sensitivity
  variants (`all-frames`: the screened 108 frames, 36 per template before the screen; `lineage`: the 017–019 cue
  cohorts, 15 / 15 / 12 / 18; the nine 011–019 cue cohorts, 19–20 cues each, as disjoint Y1 sets), and
  `calibration_record(...)` with its `content_sha256`, carrying the 149 rows and the per-draw values of the three full
  rows (Y1; Y2 6/6/6; Y3 8/8/8 — 12 × `B` floats). The **020-floor pass rates** at the full rows are computed by 020's
  own `rd.score_y1`, `rd.score_y2` and `rd.score_y3` on the materialized draw tables (duplicates relabelled), so they
  keep 020's condition forms exactly (its shares, its slope band and its `value or −1.0`); this descriptive is the only
  floor comparison outside `passes`, and it runs inside 020's code.
- **Selection.** `select_y2_row(stage1_frames, floor_tables)` — reads only `valid` and `template_id` of each frame;
  returns the composition and row, or `PRECONDITION_FAILED_FRAMES`. `select_y3_row(measured_fresh, rule_classes,
  floor_tables)` — reads only measured values (the scorability rule: both forms single token and a positive measured
  variance `Σ(y − ȳ)² > 0` over the Y1 pairs, exactly the condition under which `rd._r2` is defined); its signature
  has no prediction argument.
- **Confirm.** `stage_two_021(...)`: `rd.stage_two`'s loop with the same calls in the same order, additionally keeping
  `ceiling_prediction(program, state, nouns, measurement.dx3)` on the 79 exposed nouns per pair.
  `score_021(stage1, tables, lock)`: the three outcomes on the selected rows through `passes`, `rd.outcome_label`, the
  four 020 comparators via `rd`, the ceiling and the error split, the joint fresh-cue × fresh-frame × fresh-noun
  statistics (`rd.pair_statistics`, descriptive, no label), the kernel/direct agreement.
- **State machine.** `new_results_state_021` (phases `validate`, `calibrate`, `lock`, `confirm`, `report`; 020's
  `rd.new_results_state` and `ht.assert_phase_allowed` know no `calibrate`) and `assert_phase_allowed_021`:
  `calibrate` runs once, may resume at a later commit only after a recorded incident and never after completion;
  `stopped_for_review` (the precondition) is terminal for every later phase; `lock` once after a completed `calibrate`;
  `confirm` once after `lock`; `report` after `calibrate`.
- **Lock and report.** `build_candidate_lock_021`, `validate_lock_021` (including the lock's floor tables against the
  committed `calibration-v1.json`, canonical JSON, byte for byte), `render_predictions_021`, `render_report_021`.

## File map

- `src/neural_decompiler/readout_calibration.py` (new).
- `experiments/021-corrected-readout-confirmation/run.py`, `README.md` (new); later, by installation:
  `calibration-v1.json`, `preregistration-lock.json`, `predictions.md`, `evidence/`.
- `tests/test_readout_calibration.py` (unit; tier A, slow ones marked), `tests/test_experiment_021_runner.py` (runner,
  `current`; tier B).
- `tests/conftest.py`: `CURRENT_EXPERIMENT = "021"` in the commit that adds the 021 runner test (020's runner test
  becomes `historical`; `test_tiers.py` needs no change).
- `.gitignore`: `outputs/experiment-021/*`.
- **Unchanged:** `readout_decompilation.py` (blob-checked), every other `src/` module, `plural_fakes.py`, 020's
  experiment directory, `pyproject.toml`, `uv.lock`. Scientific path prefixes for 021: `src/`,
  `experiments/021-corrected-readout-confirmation/` and 020's own list; exempt: `calibration-v1.json`,
  `preregistration-lock.json`, `predictions.md`, `README.md` and `evidence/` of 021.

## Testing

- During work: `.venv/bin/pytest` (tier A) and `.venv/bin/pytest --tier B` (tier A + the 021 runner test).
- Before `calibrate`, `lock` and `confirm`: `HF_HUB_OFFLINE=1 .venv/bin/pytest --tier C` on the clean gated commit.
- Tier D is **not** required: no shared `src/` module, fake or dependency changes (the new module is only imported by
  021). If any shared file is touched after all, run `--tier D` before `calibrate` and record it.
- The runner test builds a **fake-closed 020 world** once per module: 020's `fake_world` fixture (imported from
  `test_experiment_020_runner`, as `test_experiment_019_runner` already imports from another runner test), 020's runner
  `explore` on the fake with `rd.EXPLORE_TOKEN_LIMIT = 5` — the smallest limit at which every frame measures its
  template's plural cue (767 in cardinal and coordinated frames, 2067 in quantifier frames, so the validity screen has
  its input) together with `a` and `the` — then a fake `closure.json` and extract written in the committed format, with
  `rc`'s four 020 closure digest constants patched to the fake's. `rc.B` is patched to 40 (`TAIL_INDEX` 1) and
  `CROSS_CHECK_DRAWS` to 2 (the patched preconditions admit several hundred rows); every cue
  stratum is patched to {`a`, `the`}, which the fake measured in all 108 frames; 020's preconditions are patched as in
  020's runner test (`MIN_VALID_FRESH_FRAMES` 3, `MIN_VALID_COORDINATED_FRAMES` 1, `MIN_SCORABLE_FRESH_NOUNS` 4), and
  because the rows are enumerated from those constants, the fake's admissible compositions always have a row.
  `rc.validity_screen` stays real (it calls `rd.frame_validity` on each frame's plural-cue measurement), and
  `MIN_SCREENED_FRAMES_PER_TEMPLATE` is patched to 1 on the fake, except in the precondition test, which keeps 6 and
  patches the screen's verdicts to leave 5 frames in one template.
- Unit tests that need the fake (the call-order spy, the stage-2 equivalence, the gate on a re-materialized fake table)
  carry `@pytest.mark.slow`.

### Test inventory (all must exist before the implementation review)

Unit (`tests/test_readout_calibration.py`):

1. Frozen constants equal revision 3 (the table above), and `rd`'s preconditions, tolerances and labels are imported,
   not redefined.
2. Program blob: the real file hashes to `caa73b40…`; a one-byte-modified copy is refused.
3. 020 closure: the committed closure record and extract verify (the gitignored results-file part is skipped when
   the file is absent); on temporary copies, each tampered digest (closure, extract, results file, state) and a planted
   manifest key or fresh noun in the ledger are refused.
4. Pools from the frozen inputs (no model): memberships, counts 45/45/36/49, 14/14/14, 40/19/20, orders, exclusions.
5. Rows: 64 Y2 rows (by total 12…18: 18, 15, 12, 9, 6, 3, 1) and 84 Y3 rows (28, 21, 15, 10, 6, 3, 1), each admissible
   composition exactly once, none inadmissible; Y1 one row.
6. `k_of`: the listed values for `n` = 12…18 and 18…24 and 20 of 24; equivalence with `count/n >= p/100` for every
   `n` ≤ 40 and every count (the integer form never differs; `0.9·20` does not round up).
7. Draws: `slot_index` recomputed independently with `hashlib` from the literal string `021|primary|b|cue/adjective|i`
   (UTF-8) and the other nine stratum strings; identical across calls; pinned digests of the primary cue and noun index
   arrays and of the frame arrays at `n_s = 14`; variants give different streams; slot counts 6/6/8.
8. Prefix construction: row `(n₁, n₂, n₃)` uses exactly the first `n_t` slots of each stratum; per-stratum counts equal
   the composition for all 64 + 84 rows.
9. Kernel = direct: on seeded synthetic tables with duplicated units, every S_j of the kernel equals the direct 020
   computation within `STATISTIC_AGREEMENT_TOLERANCE`, for Y1, every Y2 composition shape and every Y3 composition
   shape; multiplicity weights equal explicitly duplicated tables.
10. Floor rule: the 250th smallest / largest on planted arrays; the clamp and its flag; `None` as ±∞; no rounding.
11. **Pass-predicate boundaries**, `R²` at a clamped floor `0.0` and at a positive floor `F`: negative fails; exactly
    `0.0` fails; `math.nextafter(0.0, 1.0)` passes at `0.0`; `F` passes at `F`; `math.nextafter(F, -inf)` fails;
    `None`, NaN, ±∞ fail. Error type: `F` passes; `math.nextafter(F, inf)` fails; `None`, NaN, +∞ fail.
12. **One predicate:** a spy on `rc.passes` sees calls from both `pass_rates` and `score_021`, and a planted
    replacement of `passes` changes both paths' results; the only other floor comparison, the 020-floor descriptive,
    happens inside `rd.score_y1/2/3` (a spy on them sees the three full rows only).
13. Y2 selection: compositions from planted verdicts; 11 valid or 3 valid coordinated ⇒ `PRECONDITION_FAILED_FRAMES`
    and no row; the function reads nothing but `valid` and `template_id`.
14. Y3 selection: from measured values only; unchanged under arbitrary changes of the predicted columns; a planted
    zero-variance noun ⇒ composition 7/8/8 or the like; fewer than 18 ⇒ `PRECONDITION_FAILED_NOUNS`.
15. Scoring: every label branch of Y1/Y2/Y3 on synthetic tables and synthetic floor tables; `rd.outcome_label`.
16. Reproduction gate: an exact record passes at 0.0; a planted `1e-8` change of one per-cue, per-frame, per-noun or
    comparator leaf is an incident; a missing or extra key is an incident.
17. Calibration record: `rd.assert_fresh_nouns_absent` passes; index arrays appear as digests only; the three full
    rows' per-draw values are present (12 × `B`) and hash to their recorded digests; the record's `content_sha256`
    verifies.
18. Precondition: a screen leaving 5 frames in one template ⇒ terminal stop, no floor computed.
19. Stage-2 equivalence (slow): on the fake, `stage_two_021`'s shared outputs equal `rd.stage_two`'s exactly, and the
    ceiling equals `rd.ceiling_prediction` on the same measurement.
20. Environment check (slow): a planted `1e-8` drift in one re-captured reference state, or in one recomputed template
    base, is an incident before any cue prompt runs.
21. Descriptives: the ceiling distributions exist for Y1 and all 64 Y2 rows; the sensitivity pools from the frozen
    inputs have the stated sizes (36 per template before the screen, 15 / 15 / 12 / 18, nine cohorts of 19–20 cues);
    the joint fresh statistic is computed and carries no label.

Runner (`tests/test_experiment_021_runner.py`):

22. Parser: exactly `validate`, `calibrate`, `lock`, `confirm`, `report`; no override flags.
23. `validate` loads no model; refuses on each tampered 020 digest and on a modified program blob.
24. `calibrate`: executed key set equals the fake 020 ledger, each key once, no manifest key; no fresh id in the state
    or the record; 020's files byte-identical afterwards; A0 failure, a runtime or thread-count mismatch refused.
25. Leakage plants: a manifest key in 020's ledger ⇒ refusal; a foreign key reached ⇒ incident; a draw attempted
    before the gate ⇒ impossible (the gate is the only producer of the table the draws accept).
26. Gate on the fake: passes; a planted perturbation of the fake 020 record ⇒ incident, no index array, no floor.
27. Precondition: planted verdicts leaving 5 screened frames in one template, with the minimum at 6 ⇒
    `stopped_for_review` and no floor; `lock` and `confirm` refuse forever after.
28. State machine: `calibrate` once; an incident at commit X blocks X and allows a resumed attempt at Y; `lock`
    requires the installed record (tracked, committed, byte-identical); `lock` runs with every forward entry point
    raising; `confirm` requires the installed lock; a lock whose floor tables differ from `calibration-v1.json` by one
    float is refused; a second `confirm` refused; a scientific path changed since `calibrate` or `lock` refused.
29. Stage 1 and the barrier: only S1 keys run; the Y2 selection is on disk and digested before any target; tampering
    the selection, the composition, a verdict or the row on disk ⇒ refusal; Y2 targets only for valid frames.
30. Stage 2 and scoring: every target once; the ceiling from the same measurements; the Y3 row by scorability; labels
    from the lock's floors; the joint fresh statistic reported; `report` renders and verifies the draw files.

## Tasks

### Task 1: The module's pure core (no model)

- [x] Commit this plan (reviewed by an independent subagent against revision 3 before the commit; its findings are
  folded in).
- [ ] `readout_calibration.py`: constants; `program_blob_sha1` / `assert_program_blob`; `verify_020_closure`;
  `CalibrationPools` / `build_pools`; `y2_rows`, `y3_rows`, `k_of`; `slot_index`, `draw_indices`, digests; the kernel
  (`PairSums`, `NounCueSums`, `y1/y2/y3_statistics`), `direct_statistics`, `assert_agreement`; `tail_floor`,
  `passes`, `pass_rates`; `select_y2_row`, `select_y3_row`; `score_021` on tables.
- [ ] Tests 1–18 and the pool-size part of 21; tier A green; commit.

### Task 2: Re-materialization, gate, screen, calibration record (the `calibrate` body)

- [ ] `ExposedTable`, `rematerialize`, `reproduction_gate`, `validity_screen`, `precondition`, `calibrate_rows`
  (primary rows, pass rates, full-row joint rate, 020-floor pass rates, ceiling distributions, variants `all-frames`
  and `lineage`, the nine cue cohorts as disjoint Y1 sets), the runtime cross-check on the first 16 base draws of
  every row, `calibration_record`.
- [ ] Unit tests on the fake (slow): the loops' call order against `rd.run_exploration` on the same fake (a recording
  spy over the unconditional calls; `level1_breakdown` is conditional and excluded), tests 19, 20 and the rest of 21;
  commit.

### Task 3: Lock, confirm, report and the runner

- [ ] `build_candidate_lock_021` (020's lock content + the floor tables + the calibration record sha256),
  `validate_lock_021`, `render_predictions_021`; `stage_two_021`; the stage-1 Y2 selection and its digest; the
  barrier checks; `score_021` wiring; `render_report_021` (floors of the selected rows, fresh values, predicates,
  labels, comparators, ceiling and error split, percentiles from the digest-verified draw files, validity, identities,
  isolation, ledger).
- [ ] `experiments/021-corrected-readout-confirmation/run.py` (phases, `_base_inputs` as 020's, `_state_for` on
  `new_results_state_021` / `assert_phase_allowed_021`, incident gating, runtime and thread equality, A0 contract
  test, scientific-path refusal, `pytest_free_guard` for `lock`), `README.md` (inputs, commands, status), `.gitignore`.
- [ ] `tests/test_experiment_021_runner.py` (tests 22–30) and `CURRENT_EXPERIMENT = "021"` in the same commit;
  `--tier B` green; commit.

### Task 4: Independent implementation and data-flow review — stop

- [ ] An independent subagent review against revision 3: the data-flow table line by line, then the constants, the
  pools, the rows, the draws, the kernel definitions against 020's, the predicate, the selections, the gate, the
  incident and precondition semantics; fix; commit.
- [ ] `HF_HUB_OFFLINE=1 .venv/bin/pytest --tier C` on the clean gated commit; record the result in the README.
- [ ] **Stop:** present the implementation for the reviewer's implementation/data-flow review. `calibrate` runs only
  after that review authorizes it.

### Task 5: `calibrate` once — stop for the floor review

- [ ] On the clean authorized commit: `calibrate` (≈ 1.5 h: the re-materialization is one 020 explore, ≈ 70 min; the
  kernel over 149 rows × 10,000 draws takes minutes; the 020-floor descriptive through `rd.score_y*` on 3 × 10,000
  materialized tables ≈ 10–15 min). On an incident: record, stop, report; no retry without a committed fix.
- [ ] Install `outputs/experiment-021/candidate-calibration.json` byte-identical as
  `experiments/021-corrected-readout-confirmation/calibration-v1.json`; README status (gate, screen, precondition,
  row table summary, clamps, pass rates, 020-floor pass rates, sensitivities); commit.
- [ ] **Stop:** the reviewer inspects the actual 149-row floor record. Nothing is locked before that review.

### Task 6: Lock, confirmation and closure

- [ ] After the floor review: `lock` (no forward pass); present the candidate lock and predictions; **stop** — the
  user installs and commits them and the reviewer signs off.
- [ ] `confirm` once (tier C first); `report`; evidence copies (report, calibration record, predictions); README; root
  registry; C002 revisited against the outcome; memory.

## Stopping conditions

- After this plan: stop for the implementation/data-flow review; coding starts only when authorized.
- After Task 4: stop; the first model run is the single `calibrate`, only after the implementation review.
- After Task 5: stop for the floor review; no `lock` before it.
- After `lock`: stop for the user's lock commit and the reviewer's sign-off; `confirm` once.
- Any incident stops its phase and is reported with its commit; no scientific phase is retried at the same commit, and
  a completed `calibrate` or `confirm` is never rerun in this protocol version.
