# Experiment 023: Prospective Block-0 Completion

**Status: CLOSED (2026-09-24).** Three conditions PASS, and Y1/cue_final is an ENVELOPE_ONLY_FAILURE. There is no
aggregate label; see [Result and closure](#result-and-closure--2026-09-24-closed).

Implements the design
[`docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md`](../../docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md)
(revision 2, `5b38aba`) through the plan
[`docs/superpowers/plans/2026-09-24-experiment-023-block0-completion-plan.md`](../../docs/superpowers/plans/2026-09-24-experiment-023-block0-completion-plan.md)
(revision 1, `3a795fb`).

The question: does the completed, fully weight-derived program `P1` prospectively recover essentially all of the
explainable gap between Level 0 and the measured-`Δx3` ceiling, on new cues in the exposed frames (Y1) and in new frames
(Y2), in cue-final and coordinated frames?

- **`P0`** is Experiment 022's empty coalition (Level 0): the committed chain fed `ΔE`.
- **`P1`** is 022's inputs-only coalition (mask 14 cue-final, 30 coordinated). To `ΔE` it adds the embedding change and
  block 0's value and pattern terms at the cue position (the token-local row rule, exact at block 0 and computable from
  the frame's token embeddings alone), plus block 0's closed-form single-logit update at the target position. It feeds
  these through the committed reduced layers 1–2 and the frozen 020 readout. It has no fitted parameter.
- **`C`** is the frozen readout fed the measured `Δx3`. It is a comparator and normalizer only, never an input to `P1`,
  and not a mathematical upper bound, so `g > 1` is permitted.

Both programs are computed by Experiment 022's own functions. 022's module (`upstream_localization.py`, blob
`46585696…`) and the ten modules 022 itself pinned are pinned by git blob; a change in any of them refuses every phase.
The other modules they import are not pinned by blob. Between phases they are held by the rule that each phase refuses
a scientific change since the previous one, which covers everything under `src/`.

**The four conditions** (Y1/Y2 × cue-final/coordinated) each carry one statistic, pooled from per-pair sufficient
statistics `(n, Σy, Σy², SSE0, SSE1, SSEC)`:

`g = (SSE0 − SSE1) / (SSE0 − SSEC)`

- `SST = Σy² − (Σy)²/N` over the selected pairs, never a sum of pair variances; it is cross-checked against the pooled
  two-pass identity (E6).
- `g` is not clipped.
- The condition is interpretable iff `SST > 0` and `SSE0 − SSEC ≥ 0.02·SST`.
- The meaning guard is `g ≥ 0.90`.
- A PASS requires `g ≥ max(F, 0.90)`, where `F` is element `[249]` of 10,000 exposed-like draws.
- The precedence is `NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS`.
- There is no aggregate label.
- `R²₀`, `R²₁`, `R²_C` and a descriptive `ceiling_limited` flag (`R²_C < 0.80`) are reported.

One implementation (`pool` → `statistics` → `result_codes`) scores both the calibration draws and the fresh conditions: a
fresh condition's result, the calibration's result rates and its joint rate all come from the same classification.

**Scope:** new determiner-like, quantity and adjective cues (8 each) and 18 new frames (6 per template). There is no new
claim about possessive or pronoun cues, whose stratum is exhausted.

## Inputs

- The frozen modules checked by git blob (`b0c.FROZEN_BLOBS`, written out literally): 022's ten plus 022's own module.
- Experiment 022's committed calibration record (`db745653…`, content `46985fd5…`) and confirmation file
  (`1a9afef0…`, content `af848ae4…`).
- Experiment 020's closure and locked exposed states, loaded exactly as 022 loads them.
- **Once, at `extract` only:** Experiment 022's local calibration table `outputs/experiment-022/calibration-table.pt`
  (file `04659d5e…`; its tensors digest-bound in 022's record). Nothing after `extract` opens it, and the runner test
  shows every later phase running with it deleted.

## Commands

```bash
HF_HUB_OFFLINE=1 uv run python experiments/023-block0-completion/run.py validate
HF_HUB_OFFLINE=1 uv run python experiments/023-block0-completion/run.py extract
HF_HUB_OFFLINE=1 uv run python experiments/023-block0-completion/run.py freeze
HF_HUB_OFFLINE=1 uv run python experiments/023-block0-completion/run.py calibrate
HF_HUB_OFFLINE=1 uv run python experiments/023-block0-completion/run.py lock
HF_HUB_OFFLINE=1 uv run python experiments/023-block0-completion/run.py confirm
HF_HUB_OFFLINE=1 uv run python experiments/023-block0-completion/run.py report
```

- **`validate`** (no model; never opens the 022 table) checks the pins, the frozen inputs and 022's committed files.
  Once they exist it also checks the installed exposed cells (re-read against the metadata recomputed now), the
  confirmation file and the results state.
- **`extract`** runs once, weights only, under the no-forward-pass guard.
  - It first requires 022's table file with its bound sha256. A missing or different file is refused before any state
    is written; this is a precondition, and extract has not started.
  - It then reads the table and writes `outputs/experiment-023/candidate-exposed-cells.f64/.json`. The candidate is
    installed byte-identically as `exposed-cells.f64/.json`, committed, and checked before anything else.
  - Its identities:
    - E1: the table's tensor digests and orders;
    - E2: the cells equal 022's stored cells bit for bit;
    - E3: `S`, `Q` and `SSEC` equal an independent whole-table recomputation bit for bit;
    - E4: `P1` recomputed from the weights (1e-9);
    - E5: the cells against the per-noun table on the first 16 draws (1e-10);
    - E6: the pooled `SST` (1e-10).
  - E1–E3 stop for review; E4–E6 are incidents. The 020/022 re-check runs before the candidate is written; a failure is
    an incident. So is an interruption or an I/O error up to the completing state write. A partial candidate left on disk
    then blocks a rerun until the reviewer removes it.

  The artifact holds 18,900 pairs × 8 columns (`n, S, Q, SSE0, SSE1, SSEC, mean, M2`; 1,209,600 bytes). Its index binds:
  - the canonical pair order, every cue's id and stratum, and every frame's template and group;
  - the column schema;
  - 022's table and record digests;
  - the extraction module's blob, version and commit;
  - its own `content_sha256`, verified on every read.
- **`freeze`** (tokenizer only) takes the first 8 eligible cues per stratum and the first 6 eligible frames per template
  of the design's ordered lists. The exclusion adds 022's frozen units. It writes `confirmation-v1.json`, committed by
  hand; a shortfall writes nothing.
- **`calibrate`** runs once, with no model, from the committed artifact alone:
  - 10,000 SHA-indexed draws (tag `023|primary`, 8/8/8 cues, 6/6/6 Y2-like frames);
  - the canonical path with E6;
  - the loop cross-check;
  - the stop at 250 undefined draws;
  - the envelopes with their direction checks;
  - the rates and the candidate record, installed, committed and reviewed.

  The record binds the confirmation file whose counts it read. A rerun after a calibrate incident needs a new commit
  that changes no scientific path since extract (for example a documentation commit recording an interruption); the
  calibration is deterministic, so it reproduces. A code fix is a new protocol version.
- **`lock`** (no forward pass) first requires the installed calibration record to be this run's committed candidate.
  The record and the results state must bind the committed confirmation file. It then builds the Y1 prediction table
  (`[1728, 2, 79]` then `[864, 2, 79]`, `(P0, P1)`) twice and requires it bit-identical, with I5 exactly 0 and the
  block-0 algebra at 1e-12. After the 020/022 re-check it writes the lock and the preregistration. All four files are
  installed byte-identically, committed and reviewed. Any failure of that re-check, even a transient one, is a lock
  incident: lock is then refused until the reviewer decides. By then the candidate Y1 table may already exist; the lock
  and the preregistration are not written.
- **`confirm`** runs once and is never resumed:
  - `validate_lock`, then I7: the Y1 table rebuilt bit for bit before any fresh prompt;
  - stage 1: 18 S1-REF and 18 S1-VALIDITY (descriptive) prompts, then the Y2 prediction table written once from the
    reference states, never from a target measurement;
  - the barrier: the state and the Y2 table re-read and verified against the digests held in memory, with no S2-TARGET
    key in the ledger;
  - stage 2: 3,024 targets, each once, measuring `Δc`, `Δx1` and `Δx3`; everything is saved before any gate, and the Y2
    table is re-verified;
  - I1, I3 and I4;
  - the four conditions, with the kernel checked against a direct recomputation;
  - the 020/022 re-check, before any result is written: a failure is an incident, and incidents carry no result;
  - the four results and the completed phase, in one write, before anything descriptive runs;
  - descriptive records: per-template and per-stratum `g`, the cheaper block-0 rules, `P1`'s `Δx3` error and block 0's
    profile. A descriptive failure is recorded as such and never touches a result.

  An incident is recorded and stops the phase; nothing is retried.
- **`report`** renders `outputs/experiment-023/report.md`.

## Result and closure — 2026-09-24: CLOSED

The single `confirm` ran at `c7efec7` (17:38:14–17:49:23Z, exit 0). Each condition is read on its own:

| condition | g | F (exact, from the lock) | result | gap | R²₀ | R²₁ | R²_C | pairs |
|---|---|---|---|---|---|---|---|---|
| Y1/cue_final | 0.9955204329648195 | 0.9968974947302105 | **ENVELOPE_ONLY_FAILURE** | 0.1411 | 0.8182 | 0.9587 | 0.9593 | 1728 |
| Y1/coordinated | 0.9987141803462269 | 0.9986343903419131 | **PASS** | 0.4159 | 0.5420 | 0.9574 | 0.9579 | 864 |
| Y2/cue_final | 0.9974185656823554 | 0.9950528086546427 | **PASS** | 0.1468 | 0.8146 | 0.9610 | 0.9613 | 288 |
| Y2/coordinated | 0.9965628370581808 | 0.9951864564481918 | **PASS** | 0.3812 | 0.5719 | 0.9518 | 0.9531 | 144 |

**Closure notes.** These are the facts the frozen renderer does not print; the generated report is kept unedited.

- **Marginal floors.** Experiment 023 uses four marginal, per-condition calibrated floors. There is no aggregate
  PASS/FAIL criterion, and the 0.9178 joint pass rate is descriptive only. "Three of four conditions passed" is not a
  protocol outcome; the protocol defines no majority or all-pass rule.
- **Primary result: three PASS and one ENVELOPE_ONLY_FAILURE.** Y1/cue-final recovered nearly all explainable
  performance (`g = 0.9955204329648195`) but fell below its preregistered exposed-like floor
  (`F = 0.9968974947302105`). Under the frozen interpretation this is a small quantitative shift, not a refutation:
  because `g ≥ 0.90` it is an ENVELOPE_ONLY_FAILURE. It is neither simply a failure nor "basically a pass".
- **Full precision.** Y1/coordinated's PASS was decided at full precision, `0.9987141803462269 > 0.9986343903419131`,
  a margin of +7.979e-5 in exact arithmetic. The six-decimal rendering is display only.
- **Provenance.**
  - All 3,060 frozen prompts executed exactly once: expected manifest = executed `capture_prompt` calls = ledger
    spend, with zero extras, omissions or duplicates.
  - There were no interventions or patched forward passes.
  - Stage 1 preceded the construction of the Y2 table and stage 2.
  - Experiment 022's table stayed isolated, and `confirm` executed once.
- **`" shiny"`.** The pair stays in the 864-pair primary Y1/coordinated analysis. Its preplanned descriptive exclusion
  gives `g = 0.9987161` (863 pairs) and does not change the PASS. There is no substitution and no corrected primary
  statistic.
- **Renderer artifacts, kept as they are, not "fixed".**
  - The report header says `report not_started`, because rendering happens before the phase completes.
  - The display formatting uses `g ≥ 0.9`, six-decimal floors and `E4 = 0.000`.

  The authoritative values remain the machine-readable lock and results artifacts.
- **Reviews.** Every checkpoint had an independent read-only review:
  - the implementation (NOT READY, then fixed and re-reviewed);
  - extraction, freeze and lock: each PASS WITH NOTES;
  - the floor review: PASS, marginal by design;
  - the confirmation review: PASS WITH NOTES, with the outcome reproduced in exact arithmetic, the Y2 table rebuilt
    byte for byte and the identities recomputed.
- **Descriptive only, no outcome force:**
  - Y1 cue-final by stratum: adjective 0.9926, determiner-like 0.9959, quantity 0.9974; both templates about 0.9955.
  - Cheaper block-0 rules recover far less in cue-final frames: value-only about 0.50, six heads 0.91–0.94, linear
    response about 0.76. So the exact block-0 rule is what completes the program.
  - All 18 new frames were valid at stage 1.
- **Scope, unchanged.** The claim covers new determiner-like, quantity and adjective cues and new frames only. There
  is no claim about possessive or pronoun cues, and none that the downstream readout is complete (`R²_C` 0.95–0.96).
  No claim file was changed.

**Closure evidence**, all in `evidence/`:
- `5344f44`: the Y2 table (`4623adde…`, index `e87091ef…`) and the report as rendered (`final-report-2026-09-24.md`,
  `9b7a8d7a…`);
- `69ec270`: the confirm run record with its prompt accounting, the confirmation review, the report verification, and
  the confirmation-record extract (content `0dcc2c44…`) with its builder.

## Protocol history — 2026-09-24

- **Implementation.** Plan Tasks 1–6 are implemented in `src/neural_decompiler/block0_completion.py`, this runner and
  their tests:
  - tier A: `tests/test_block0_completion.py`;
  - tier B: `tests/test_experiment_023_runner.py`, on the fake world of 022's runner test;
  - tier C, opt-in: the real tokenizer's freeze gives exactly the design's expected picks; E1–E3, E5 and E6 hold
    read-only on the real local 022 table; `P0`/`P1` and the prediction path reproduce 022's stored coalitions bit for
    bit on real exposed pairs.
- **Independent implementation review (Task 7, read-only, on `c0f74b1`): NOT READY.** It found:
  - one blocker: the four-way classification was written three times (the fresh `classify`, the calibration rates and
    the joint rate), against hard requirement 1;
  - three should-fix findings: confirm could lose scored results to a descriptive failure; lock did not bind the
    calibrated confirmation file; several refusals and incidents were untested;
  - minor findings.

  The other three hard requirements (the stage-1 barrier, the exposed cells as provenance, the pinned 022 program) were
  met. The fixes are:
  - `7fcac8e`: one classification function (finding 1);
  - `7d6bf06`: the results are written before any descriptive record (finding 2);
  - `30da5d0`: the confirmation binding and a strict record check (findings 3, 5);
  - `6e5f6ee`: the extract and lock preconditions and re-checks, the pins first, `--no-renames` (findings 6, 7, 11,
    13; 8 documented);
  - `bc72347`: E3 as an independent recomputation, the index's content digest, and the report's stop and tails
    (findings 9, 10, 12; 15 documented);
  - `e4b8f3e`: the missing tier-B tests (finding 4).
- **Independent re-review of those fixes (read-only).** It found findings 1–13 fixed. One should-fix remained,
  introduced by the finding-2 fix: an incident recorded after scoring could sit beside four results. There were also
  two minors: extract's last steps sat outside its incident handler, and the pinning claim was overstated. All are
  fixed in `28d1bbb`: the re-check now runs before any result is written, and an incident carries no result, in the
  state and in the report.
- **Tests on the final implementation commit `2111271`:** tier A 483 passed; tier B 514 passed, 11 skipped (the opt-in
  pinned-model tests); tier C 525 passed.
- **`extract` ran once and succeeded.** It ran at `2111271` on 2026-09-24, 14:20:44–14:26:14Z, as run
  `9428b2fde588ac75`, with no prompt: weights only, under the no-forward-pass guard.
  - **Preflight, read-only.** The pins, 020's closure and 022's committed files all verified, and the stock `validate`
    passed. The 022 table was the bound file (424,131,002 bytes, `04659d5e…`). Immediately before the run, the opt-in
    tier-C real-table tests passed, with E2 and E3 each 0 of 18,900 pairs differing.
  - **Identities.**
    - E1: 13 tensor digests and the orders;
    - E2 and E3: bit for bit on all 18,900 pairs;
    - E4: `P1` recomputed from the weights for all 18,900 pairs, max |Δ| exactly 0.0;
    - E5: 4.0e-15 over 192 draw quantities;
    - E6: 1.43e-14 per pair, 4.0e-15 per draw.

    E4's recorded `"at"` is an empty string, because the maximum never rose above 0.0 (a cosmetic effect of 022's
    max-tracker).
- **Independent extraction review (read-only): PASS WITH NOTES, no blockers.** A separate reviewer wrote its own
  checker and did not rely on the runner's pass flags. It re-derived:
  - the state digest, the hashes, the byte layout and the canonical order (including every cue's token id through the
    tokenizer);
  - the 023 draws for all 10,000 draws, which never select the pronoun stratum;
  - the 13 tensor digests;
  - every column of all 18,900 rows: bit for bit against 022's stored values and its own reductions, and within
    4.2e-16 relative of exactly rounded sums;
  - the pooled `SST` on 409 selections with repeats: three methods agree within 2.3e-15;
  - `P0` and `P1` from the weights for all pairs, bit for bit, under a guard that refused any module call.

  It also showed, with every route to Experiment 022's table blocked, that the calibration reader and kernel work from
  the compact artifact alone.
- **The artifact is installed and committed.** The reviewed candidate bytes were copied, not regenerated, as
  `exposed-cells.f64` (sha256 `d2ee71e5…`, 1,209,600 bytes) and `exposed-cells.json` (file sha256 `62818147…`,
  content digest `d38305cb…`). They were committed alone in `83d9c58`, and `git show` of that commit reproduces the
  reviewed bytes. The stock `validate` and the runner's installed-artifact reader accept the pair, and no scientific
  path has changed since extract.
- **Experiment 022's 0.42-GB calibration table is no longer an operational dependency of 023.** From here on,
  calibration reads only the committed artifact; nothing after `extract` opens the table.
- **Inherited dependency.** Every phase still reads Experiment 020's git-ignored local
  `outputs/experiment-020/results.json` through the frozen `load_frozen_inputs`, as 020–022 did. This is documented and
  lies outside 023's table-independence claim.
- **`freeze` ran once and succeeded.** It ran at `89536b0` on 2026-09-24, 15:08:07–15:08:17Z, exit 0.
  - It used the tokenizer and the structural rules only. The stock runner ran inside a launcher that refused
    `load_model` and every `torch.nn.Module` call; neither fired.
  - It took exactly design revision 2's expected picks, the first 8 of each cue list and the first 6 of each frame
    list, with 0 rejections.
  - The exclusion holds 327 cue ids and 144 frame texts from 15 sources.
  - The manifest has 3,060 unique keys: S1-REF 18, S1-VALIDITY 18, Y1 2,592, Y2 432. None collides with the 36,252
    forbidden keys.
- **Independent freeze review (read-only): PASS WITH NOTES, no blockers.** A clean-room reconstruction rebuilt the file
  byte for byte from the design's lists, its own tokenizer and structural checks, and its own exclusion sets, with no
  numerical input.
- **The freeze file is committed.** `confirmation-v1.json` (128,228 bytes, sha256 `5fadfa50…`, content `4e64d4c2…`) is
  committed exactly as written, alone, in `a9287ec`. The stock `validate` accepts it.
- **The review notes are in the design's notes N1–N4; the data is unchanged:**
  - cue freshness is a cue-slot rule;
  - `" shiny"` is also the target adjective of one exposed frame, so one Y1 pair repeats the token at `p_c` and `p_t`;
    it is kept and will be named at confirm;
  - `freeze` reads prior numerical artifacts for integrity and isolation checks only;
  - the weights claim is stated precisely.

  The run records and both reviews are in `evidence/` (`391586f`).
- **`calibrate` ran once and succeeded.** It ran at `4e5deeb`, 15:53:07–15:53:31Z, exit 0, with no model; 022's
  table and the weight blob were blocked.
  - 0 undefined draws. The kernel matches a plain loop within 8.6e-14; the largest E6 difference is 7.5e-15.
  - The floors, F = v₍₂₅₀₎ per condition: 0.996897 / 0.998634 / 0.995053 / 0.995186. None is guard-bound.
  - Floor review: PASS. The four gates are marginal by design; the joint all-four share, 0.9178, is recorded
    descriptively and never corrected.
  - The record is committed alone, byte-identically, in `51b5c7d` (`94db6df2…`).
- **Experiment 020's results state is archived.** The hash-pinned but gitignored file is the only copy of the locked
  reference states that lock and confirm use. An exact, recoverable copy is now in `evidence/archive/` (`13b8d39`).
- **`lock` ran once and succeeded.** It ran at `13b8d39`, 16:47:12–16:51:59Z, exit 0. The weights were loaded once
  and every module call was refused after the load.
  - Tier C passed first: 525.
  - I5 is exactly 0 and the block-0 algebra is 4.9e-15.
  - Independent lock review: PASS WITH NOTES, no blockers. The reviewer regenerated the whole Y1 table
    (`37603f05…`, 409,536 values) and the lock (`4bd5a5b1…`) byte for byte.
  - The four files are committed together, byte-identically, in `f2294ab`. Confirm's own `validate_lock`, run
    read-only, accepts them.
  - The freeze's no-weights evidence is restated in the lock review's evidence (`a0e41a9`), not in the bound design
    text: a Python-level file-open log cannot show weight reads, so the conclusion rests on the code path and the
    tokenizer-only reconstruction.
- **`confirm` ran once and succeeded.** It ran at `c7efec7`, 17:38:14–17:49:23Z, exit 0, after tier C (525) and confirm's
  own pre-prompt validation.
  - I7 bitwise; stage 1 (36 prompts); the Y2 table written once; the barrier; stage 2 (3,024 prompts).
  - I1 1.3e-5, I3 2.0e-5, I4 4.3e-5; kernel 1.6e-15.
  - Independent confirmation review: PASS WITH NOTES; outcome reproduced.
- **`report` ran once.** At `c7efec7`, 18:14:58–18:15:07Z, exit 0: `report.md` (`9b7a8d7a…`), exactly the frozen
  renderer's output for the confirmed state.
- **Closure:** `5344f44` (data), `69ec270` (evidence) and this documentation.
- **Steps, all done:**
  1. (done) the independent implementation review, its fixes, a re-review, and tiers A/B/C on the final implementation
     commit;
  2. (done) `extract`, the independent extraction review, and the byte-identical installation and commit of the
     artifact (`83d9c58`);
  3. (done) `freeze`, the independent freeze review, and the byte-identical commit (`a9287ec`);
  4. (done) `calibrate` once, the floor review, and the commit (`51b5c7d`);
  5. (done) `lock`, the independent lock review, and the commit (`f2294ab`);
  6. (done) `confirm` once, and the independent confirmation review;
  7. (done) `report`, and its verification;
  8. (done) closure.
