# Experiment 023 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan revision 1** (2026-09-24). It tracks design revision 2 (`5b38aba`), which the reviewer approved in principle.
It is a plan only: no code, no runner, no extraction, freeze, calibration, lock or fresh prompt.

**Goal.** Implement the approved Experiment 023 design: a prospective test that the completed, weight-derived program
`P1` recovers essentially all of the explainable Level-0 → measured-`Δx3`-ceiling gap. The program is 022's exact
block-0 rule, the committed reduced layers 1–2 and the frozen 020 readout. There are four conditions — Y1/Y2 ×
cue-final/coordinated — scored by `g = (SSE₀ − SSE₁) / (SSE₀ − SSE_C)` against `max(F, 0.90)`.

**Architecture.** There is one new module, `src/neural_decompiler/block0_completion.py` (imported as `b0c`), and one
runner, `experiments/023-block0-completion/run.py`.

The module *calls* frozen modules and never edits them. Experiment 022's `upstream_localization.py` (`ul`, git blob
`465856962aa380747d1a4f1338d1d2762d03c9f9`) provides:
- `ModelPrograms`, `pair_context`, `compose_dx3`, `reduced_chain`, `contrast_of`, `block0_terms`;
- `measure_prompt`, `i3_error`, `reference_rows_017`, `y1_states`;
- `extract_exclusion`, the table byte format (`table_bytes`, `write_table`, `read_table`, `verify_table_index`), the
  order statistics and `defined_median`.

Through `ul` it also uses the 017–021 programs, whose blobs 022 already pins. `P0` is `ul.compose_dx3(…, mask 0)`.
`P1` is `ul.compose_dx3(…, mask 14 | 30)`, read through `ul.contrast_of`. Both are therefore 022's computation bit for
bit.

It adds only what the design adds:
- the six sufficient statistics and their pooling, with the E6 cross-check;
- `g`, the gap rule and the four-way classification against `max(F, 0.90)`;
- the extraction of the exposed-cells artifact with E1–E6;
- the freeze of 3 × 8 cues and 3 × 6 frames;
- the calibration from the committed artifact alone;
- the two-column `P0`/`P1` prediction tables, the lock, the confirmation scoring, the descriptive comparators and the
  report.

## Global constraints

- **Frozen modules, checked by git blob.** Every phase and a tier-A test compute `sha1(b"blob <len>\0" + bytes)` with
  no git call. The blobs are 022's ten (`ul.FROZEN_BLOBS`, unchanged) plus `upstream_localization.py` →
  `465856962aa380747d1a4f1338d1d2762d03c9f9`. They are recorded in the artifact index, the calibration record and the
  lock.
- **Inherited inputs**, asserted before use:
  - 022's committed calibration record: file `db745653…`, content `46985fd5…`, and its `rematerialization.table_sha256`;
  - 022's lock and confirmation (`af848ae4…`), and 022's closure extract (content `ef393bb8…`);
  - 020's closure and locked states, through `ul.load_frozen_inputs`, exactly as 022 loads them.
- **The 022 table.** `outputs/experiment-022/calibration-table.pt` (file sha256 `04659d5e…`) is opened by `extract`
  only. A tier-B test makes it unreadable during every other phase, and they still run.
- **Runtime.** CPU, float32, 4 threads, the pinned checkpoint and library versions. It must equal 020's explore record
  at every phase that loads the model (022's `_check_runtime`).
- **No fresh noun.** The 79 scorable exposed nouns, as in 022.
- **Spent sets never enter.** 021's (020's confirmation set) and 022's confirmation sets never enter any 023 phase. Their
  keys are forbidden in 023's ledger.

## Constants (a tier-A test pins every one)

- `DESIGN = {"path": …, "revision": 2, "commit": "5b38aba"}`; `PLAN = {"path": …, "revision": 1, "commit": <this plan's
  commit>}`.
- `B = 10_000`; `DRAW_TAG = "023|primary"`; `CROSS_CHECK_DRAWS = 16`. The draw index is
  `int.from_bytes(sha256(f"023|primary|{b}|{stratum}|{slot}")[:8], "big") % n`: 022's formula with its own tag.
- `LOWER_RANK = 250` (element `[249]`); `GUARD_MIN = 0.90`; `GAP_MIN = 0.02`; `CEILING_LIMITED_R2 = 0.80`.
- `STRATA = ("determiner-like", "quantity", "adjective")`; `CUE_QUOTA = 8`; `FRAME_QUOTA = 6`; `FRAME_ID_TAG = "023"`;
  `P_C_RANGE = ul.P_C_RANGE`.
- `CUE_CANDIDATES` and `FRAME_CANDIDATES`: the design's ordered lists, verbatim.
- `P1_MASK = {"cue_final": 14, "coordinated": 30}`; `P0_MASK = 0`.
- `TOLERANCES`:
  - `E4` 1e-9; `E5` 1e-10; `E6` 1e-10 (relative); `kernel` 1e-10;
  - `algebra` 1e-12;
  - `I1` 1e-4; `I3` 1e-4; `I4` 1e-3; `I5` 0.0.
- `CELL_COLUMNS = ("n", "S", "Q", "SSE0", "SSE1", "SSEC", "mean", "M2")`; `CELLS_VERSION = "023-cells-v1"`.
- `RESULTS = ("NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS")`.
- `CONDITIONS = ("Y1/cue_final", "Y1/coordinated", "Y2/cue_final", "Y2/coordinated")`.

## Objects and naming (`b0c`)

**Pure core, tier A.**
- `pair_cells(y, p0, p1, c) -> tensor[8]`. `y`, `p0`, `p1` and `c` are float64 over the 79 nouns. It returns `n`, `S =
  y.sum()`, `Q = (y*y).sum()`, the three SSE, and `mean`, `M2` (two-pass). Summation order is fixed: torch float64,
  noun axis.
- `pool(cells[P, 8], index[K]) -> {N, S, Q, SSE0, SSE1, SSEC, SST, SST_two_pass}`, with `SST = Q − S²/N`. The two-pass
  identity `Σ M2 + Σ n (mean − S/N)²` is computed alongside, and E6 raises when they disagree.
- `g_statistic(pooled) -> {g, gap, defined, R2_0, R2_1, R2_C, ceiling_limited}`. `defined` is `SST > 0 and SSE0 − SSEC
  ≥ 0.02·SST`. `g` is unclipped, and `None` when undefined.
- `lower_bound(values[B], defined[B]) -> float` uses `ul.order_statistic` with `undefined_at = −∞` and rank 250.
  `direction_ok(F, values, defined)` checks `F ≤ median(defined)`.
- `classify(g, defined, F) -> result`, in the frozen precedence.
- `slot_index` and `draw_indices` use the 023 tag. `draw_pairs(draws, b, population, group)` returns the flat indices
  into the cell array, repeats included.

**Extraction.**
- `extract_cells(table, record, progs, inputs)` performs E1–E6 and returns the `[18900, 8]` array and the index
  metadata. Pairs are in 022's canonical order (`ul.calibration_units`).
- `write_cells` and `read_cells` use 022's byte format and index conventions. The index carries:
  - the identifiers: cue word, token id, stratum; frame id, template, group;
  - `CELL_COLUMNS` and their formulas;
  - 022's table file sha256 and tensor digests;
  - 022's record file and content digests;
  - `CELLS_VERSION`;
  - the extraction module's path and git blob;
  - the data file's sha256.

**Program.**
- `predictions(progs, ctx) -> (P0 Δĉ, P1 Δĉ)`, through `ul.compose_dx3` and `ul.contrast_of`.
- `algebra_checks(ctx)` checks `V + P = ΔA0(p_c)` and that the closed-form `T` equals `ctx.factors.attn_pt`, to 1e-12.
  The closed form is the design's formula.
- `prediction_tables(progs, units, states, reference_ids)` builds the blocks cue-final `[P, 2, 79]` and coordinated
  `[P, 2, 79]` (`[:, 0]` = `P0`, `[:, 1]` = `P1`). It also returns I5's maximum against `rd.predicted_dx3`, which must
  be exactly 0.0, and the algebra maxima.

**Freeze.** `freeze_payload(tokenizer, inputs, confirmation_022)`. The exclusion is `ul.extract_exclusion(inputs)` plus
022's 24 cue ids and 18 frame texts. It applies the design's quotas and structural rules. A shortfall raises
`FreezeShortfall`; nothing is written, and the phase stops for review.

**Confirm.**
- `stage_one` captures each new frame's reference state (`rd.capture_frame_020`) and runs the S1-VALIDITY measurement
  (descriptive). It writes the Y2 prediction table once, with its digests in the stage-1 record. This is 022's
  procedure, with 023's two-column table.
- `barrier` re-reads the state and the Y2 table and verifies them against the digests held in memory. No S2-TARGET key
  may be in the ledger.
- `stage_two` is `ul.stage_two_022` unchanged: each target once, with `Δc`, `Δx1`, `Δx3` and `C`. It is called with
  023's confirmation object, which exposes the same `tokens`, `exposed_frames`, `frames` and `target_prompts`.
- `target_gates` computes, per target pair:
  - I1 from the factors against the measured `Δx1`;
  - the full composition (`ul.compose_dx3`, mask 31/15, layers 1–2 exact), with I3 against the measured `Δx3` and I4
    against `C`.
  - Descriptively: `P1`'s `Δx3` relative error at `p_c`/`p_t`, and block 0's head profile.
- `score` builds the cells per condition from the saved measurements and the verified tables. It pools them with E6,
  computes `g`, the gap, the `R²` and the flag, and classifies. It also runs the kernel against a direct flattened
  recomputation.
- `comparators` computes the spike's cheaper rules on the fresh pairs, weights-only; they are descriptive. The rules are
  value term only, the relative self logit, the oracle self-weight, the first-order softmax, and four and six heads (the
  exposed spike's head order 5, 0, 7, 6, 3, 2, frozen here).

## Artifact schemas

All JSON artifacts are canonical JSON (`pm.canonical_json`) plus a newline. `content_sha256` is computed over the
record without that key (021/022's convention).

1. **`experiments/023-block0-completion/exposed-cells.f64`** and **`exposed-cells.json`** — committed after `extract`.
   The data file is `[18900, 8]` raw little-endian float64 (1,209,600 bytes). The index is described above, plus
   `extraction` (E1–E6 maxima and verdicts, the run time, the commit) and `content_sha256`.
2. **`experiments/023-block0-completion/confirmation-v1.json`** — committed after `freeze`. It is 022's schema with:
   - `strata` in place of the four classes, and `counts = {"classes": {8, 8, 8}, "templates": {6, 6, 6}}`;
   - `exclusion` with 022's sources plus `confirmation-022`, its path and file sha256;
   - `manifest`: S1-REF 18, S1-VALIDITY 18, and S2-TARGET Y1 2,592 / Y2 432.
3. **`outputs/experiment-023/results.json`** — the state, in 022's schema style:
   - `run_id`, the input digests (the artifact's, the confirmation's and 022's), `versions`, `phases`;
   - `executed_prompt_keys`, which stays empty until `confirm`;
   - `calibration`, `lock` and `confirmation`.
4. **`outputs/experiment-023/draw-values.pt`** — per population and group: `g[B]`, `defined[B]`, `gap[B]`, `SST[B]`
   and `R2_{0,1,C}[B]`. Digested; the draw-index digests go into the record.
5. **`calibration-v1.json`** — installed byte-identically from `outputs/experiment-023/candidate-calibration.json`,
   committed. It holds:
   - the design, the commits, the input digests (with the artifact's content and file digests) and the module blobs;
   - `constants` and `pools`;
   - `draws` (B, strata sizes, index digests);
   - per condition: `F` with its rank and element, `direction_check`, `undefined_count`, `guard_bound` (`F < 0.90`),
     the median and tails, and `result_rates`;
   - `joint_rates` (descriptive), `kernel_check`, the E6 maximum over the draws, and the draw-array digests.
6. **`preregistration-lock.json`** + **`preregistration.md`** + **`locked-y1-table.f64`/`.json`** — installed
   byte-identically and committed.
   - The lock binds the four condition definitions (statistic, `F`, guard, precedence, readings, scope sentence), the
     program definition and module blobs, the calibration record and artifact digests, the confirmation digest, the Y1
     table's file and index digests, and the Y2 table spec (construction, format, layout, orders).
   - The Y1 table is cue-final `[1728, 2, 79]` then coordinated `[864, 2, 79]` (3,276,288 bytes), in 022's byte format.
     The pair order is cue token id, then `frame_id`; the column order is `(P0, P1)`; the nouns are 022's order.
7. **Stage-1 record** (in the state, digested): the reference states, the validity verdicts (descriptive), and the Y2
   table's digests. The Y2 table itself is written once to `outputs/experiment-023/y2-table.f64` and `.json` —
   `[288, 2, 79]`, `[144, 2, 79]` — and committed as closure evidence after `confirm`.
8. **`outputs/experiment-023/stage2-measurements.pt`** — `ul.stage_two_022`'s tensors, written with digests before any
   gate.
9. **`outputs/experiment-023/report.md`**.

## Phase control flow

- **`validate`** (no model). It checks:
  - the blobs and inherited inputs;
  - 022's record and closure extract;
  - once each exists, the artifact, the confirmation and the state, with ledger isolation.

  It never opens the 022 table.
- **`extract`** (once; weights only; the no-forward-pass guard).
  - Refused if the artifact is committed, or if a candidate exists and a state records it.
  - Order: load 022's record → verify the table's tensor digests (E1) → build the cells → E2 and E3 → recompute `P1`
    for all 18,900 pairs from the weights and 020's locked states (E4) → the first-16-draw per-noun check (E5) → E6 over
    every pair and those draws → write `outputs/experiment-023/candidate-exposed-cells.f64/.json`.
  - A failure writes nothing and stops for review.
  - The candidate is installed byte-identically, committed and independently checked.
- **`freeze`** (tokenizer only; the guard). Refused if the confirmation exists. It writes `confirmation-v1.json`, which
  is committed by hand.
- **`calibrate`** (once; no model).
  - It requires the committed artifact (tracked, digests verified against its own index and against what the state
    recorded at `extract`) and the committed confirmation.
  - It reads only these two: the confirmation for its stratum and template counts, the artifact for the cells.
  - Then the draws, the pooled statistics with E6, the kernel check, the undefined count and stop, `F`, the direction
    checks, the rates and the candidate record.
  - The record is installed byte-identically, committed and reviewed before `lock`.
- **`lock`** (no forward pass; weights only; the guard).
  - It requires the committed record and no scientific change since `calibrate`.
  - It builds the Y1 table twice and requires them bit-identical. I5 must be exactly 0; the algebra checks must hold at
    1e-12.
  - Then the candidate lock and the preregistration. All four files are installed byte-identically, committed and
    independently reviewed.
- **`confirm`** (once; never resumed). It is 022's control flow:
  - `validate_lock` with tracked-file enforcement; the Y2 table must be absent; the runtime check;
  - I7 (the Y1 table rebuilt bit for bit before any fresh prompt);
  - the stage-1 keys recorded in the ledger, then stage 1 and the Y2 table;
  - the barrier;
  - the target keys recorded, then stage 2, with the measurements saved;
  - the Y2 table re-verified;
  - `target_gates`, with the gates written to the state before enforcement (I1, I3, I4);
  - `score`, the comparators and the descriptives;
  - the 020/022 re-checks;
  - complete.

  An incident is recorded and stops the phase. Nothing is restored or retried.
- **`report`** renders `outputs/experiment-023/report.md`:
  - the four conditions: `g`, `F`, the guard, the result, the gap, `R²₀`, `R²₁`, `R²_C`, the flag, and the CDF
    percentile (descriptive);
  - the scope sentence;
  - the descriptive records.

  There is no replication phase.

## Ledger and isolation

- 023 executes prompts only at `confirm`: 18 + 18 + 3,024 = 3,060 keys, each once.
- Before `confirm`, the ledger is empty.
- **Forbidden in 023's ledger:** 020's ledger keys (which covers 022's exposed calibration keys), 022's manifest keys,
  and 020's confirmation set (021's spent set).
- By construction every 023 key carries a new cue token or a new frame. `freeze` and `validate` also assert
  disjointness explicitly against 020's ledger (`inputs.closure`) and 022's committed manifest.

## Incidents and failures

These follow 022:
- a gate violation (E4–E6 at extract or calibrate; I1, I3, I4, I5, I7, algebra, barrier at lock or confirm) is an
  incident: recorded with its commit, the phase stopped, never retried at that commit;
- a freeze shortfall, a calibration stop (250 or more undefined draws) or an extraction mismatch (E1–E3) writes nothing
  and stops for review — not an incident;
- an interruption of `confirm` spends the protocol version.

## Testing

`tests/conftest.py`: `CURRENT_EXPERIMENT = "023"`. The 022 runner test becomes historical (tier D).

- **Tier A** (`tests/test_block0_completion.py`, pure, no model):
  - constants pinned;
  - `pair_cells` against hand values;
  - `pool` with repeated indices equals flattening the repeated per-noun data. SST must never equal the sum of pair
    variances when the means differ; a regression test proves the difference;
  - E6 agreement on random data, and E6's raise on a planted inconsistency;
  - `g` unclipped (`g > 1`, `g < 0` cases), the gap rule's boundary, and `SST ≤ 0` undefined;
  - `lower_bound` at element `[249]`, the −∞ undefined convention, the stop at 250;
  - `classify` precedence;
  - draw determinism and the 023 tag;
  - the byte format round trip;
  - the candidate lists verbatim;
  - the freeze on a stub tokenizer, including the shortfall;
  - the artifact reader refusing a digest mismatch.
- **Tier B** (`tests/test_experiment_023_runner.py`, a fake world with a fake 022 table, fake weights and fake model):
  - every phase on a small synthetic world, and every refusal: a second `extract`, `calibrate` without the committed
    artifact, `lock` before the record is committed, a second `confirm`;
  - **the extraction boundary:** `calibrate`, `lock`, `confirm` and `report` pass with the 022 table path unreadable;
  - the barrier on tampering;
  - an incident at I7 and at stage 2;
  - `g > 1` producing `PASS`, not an incident;
  - the ledger isolation.
- **Tier C** (pinned model, `NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1`):
  - the real-tokenizer freeze contract, with the design's expected picks;
  - on a deterministic handful of exposed pairs: `P1` from `predictions` equals 022's stored table bit for bit, and I5
    and the algebra hold;
  - `pair_cells` reproduces 022's stored `sse` bit for bit.

## Runtime and storage (estimates on this machine)

- `extract` ≈ 12 min: the `P1` recomputation for 18,900 pairs. The artifact is 1.2 MB.
- `calibrate` ≈ 2–4 min: 10,000 draws × 2 populations of pooled sums, vectorized. The draws are about 2 MB.
- `lock` ≈ 5 min: the Y1 table twice, 3.3 MB committed.
- `confirm` ≈ 20 min:
  - I7 ≈ 2.5 min;
  - stage 1 and the Y2 table ≈ 1 min;
  - 3,024 forwards ≈ 2 min;
  - the gates and the full composition ≈ 3 min;
  - the comparators ≈ 10 min.

## File map

- `src/neural_decompiler/block0_completion.py` (new)
- `experiments/023-block0-completion/run.py`, `README.md` (new)
- `experiments/023-block0-completion/` committed artifacts: `exposed-cells.f64/.json`, `confirmation-v1.json`,
  `calibration-v1.json`, `preregistration-lock.json`, `preregistration.md`, `locked-y1-table.f64/.json`
- `tests/test_block0_completion.py`, `tests/test_experiment_023_runner.py` (new); `tests/conftest.py`
  (`CURRENT_EXPERIMENT`); `.gitignore` (`outputs/experiment-023/*`)
- Root `README.md` registry entry, at closure only.

## Tasks (small reviewable commits; the checkboxes are for execution)

**Task 1 — skeleton.**
- [ ] `.gitignore`; the module's constants, blob checks and inherited-input checks; tier-A pins.
- [ ] Commit: "feat: experiment 023 module skeleton, constants and frozen-input checks".

**Task 2 — the pure core.**
- [ ] `pair_cells`, `pool` with E6, `g_statistic`, `lower_bound`, `direction_ok`, `classify`, the 023 draws, and the
  byte format.
- [ ] Every tier-A test of the core, including the pooled-`SST` regression test.
- [ ] Commit: "feat: experiment 023 sufficient statistics, pooled SST, g and the four-way classification (pure core)".

**Task 3 — extraction.**
- [ ] `extract_cells` with E1–E6, `write_cells` and `read_cells`, and the extraction record.
- [ ] Fake-table tests (tier B); the tier-C bitwise checks on real exposed pairs.
- [ ] Commit: "feat: experiment 023 exposed-cells extraction from 022's verified table".

**Task 4 — freeze.**
- [ ] The candidate lists, the exclusion plus 022, the quotas, the manifest and the shortfall.
- [ ] Stub- and real-tokenizer tests.
- [ ] Commit: "feat: experiment 023 tokenizer-only freeze of 3 × 8 cues and 3 × 6 frames".

**Task 5 — calibration.**
- [ ] Draws, pooling, the kernel check, the undefined stop, `F`, the direction checks, the rates and the record — from
  the committed artifact only.
- [ ] A synthetic full-scale dry run with no model: B = 10,000 at the real pool sizes, checking time and memory.
- [ ] Commit: "feat: experiment 023 calibration from the committed exposed cells".

**Task 6 — lock, confirmation, report, runner.**
- [ ] The prediction tables, the algebra checks, `build_lock`, `validate_lock` and the preregistration.
- [ ] `stage_one`, `barrier`, 022's `stage_two`, `target_gates`, `score`, `comparators` and `render_report`.
- [ ] The runner with seven phases; the fake-world runner tests; `CURRENT_EXPERIMENT`; the experiment README.
- [ ] Commit: "feat: experiment 023 lock, confirmation, report and runner".

**Task 7 — review and gate.**
- [ ] An independent implementation and data-flow review against design revision 2 and this plan; fixes in separate
  commits.
- [ ] Tier B, then tier C on the clean gated commit. **Stop.**

## Stopping conditions

- Stop after this plan, for the implementation-plan review.
- After approval: implement Tasks 1–7 and stop for the implementation review. Every scientific phase — `extract`,
  `freeze`, `calibrate`, `lock`, `confirm` and `report` — runs only when separately authorized.
