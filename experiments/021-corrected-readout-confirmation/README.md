# Experiment 021: Corrected Readout-Decompilation Confirmation

Implements the design
[`docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md`](../../docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md)
(revision 3, `0ac46aa`, final design review passed) through the plan
[`docs/superpowers/plans/2026-09-23-experiment-021-corrected-readout-confirmation-plan.md`](../../docs/superpowers/plans/2026-09-23-experiment-021-corrected-readout-confirmation-plan.md)
(`fb10d44`, implementation/data-flow review passed).

The Experiment 020 question — does the decoded mechanism predict the model's own singular-versus-plural logit contrast
for unseen cues, unseen frames and unseen nouns — with the same program, model, populations, procedure, labels and
untouched confirmation set, and floors from a preregistered exposed-only calibration.

## Inputs

- The program: `src/neural_decompiler/readout_decompilation.py` exactly as at `ce3766b` (git blob `caa73b40…`),
  checked on the imported file by every phase; nothing in 021 edits or reimplements it.
- Experiment 020's closure (`7f4a4f8`): `closure.json` (`f2b1b5e7…`), the committed exploration extract
  (`99f25ee6…`) and the local, gitignored results state `outputs/experiment-020/results.json` (file `da63b8c2…`,
  state `2e5485dc…`), whose ledger (30,132 exposed keys, 0 of the 3060 manifest keys) and recorded exploration are
  the calibration's only exposed source.
- The confirmation set: `experiments/020-readout-decompilation/confirmation-v1.json` (`e098e2b4…`), read in place.
- The Experiment 011, 012 and 017 locks and the 006–019 input chain, exactly as in Experiment 020.

## Commands

```bash
HF_HUB_OFFLINE=1 uv run python experiments/021-corrected-readout-confirmation/run.py validate
HF_HUB_OFFLINE=1 uv run python experiments/021-corrected-readout-confirmation/run.py calibrate
HF_HUB_OFFLINE=1 uv run python experiments/021-corrected-readout-confirmation/run.py lock
HF_HUB_OFFLINE=1 uv run python experiments/021-corrected-readout-confirmation/run.py confirm
HF_HUB_OFFLINE=1 uv run python experiments/021-corrected-readout-confirmation/run.py report
```

- `validate` (no model): the frozen inputs, the program blob, every Experiment 020 closure digest, the confirmation
  set's isolation, the pools (cues 45/45/36/49, frames 14/14/14 before the screen, nouns 40/19/20) and the rows
  (Y2 64, Y3 84).
- `calibrate` (once, ≈ 1.5 h): re-executes exactly 020's ledger with 020's functions in 020's order (references
  first; each key checked against 020's ledger and the manifest before it runs), the environment check at `1e-9`,
  the identities, the reproduction gate at `1e-9`, the frozen validity screen and the precondition (≥ 6 screened
  frames per template, else a stop for review), then the 10,000 SHA-indexed base draws, the statistics of all 149
  rows, the kernel/direct cross-check (an excess stops the phase with its row, draw and statistic, before any floor),
  the floors and the descriptives. Writes `outputs/experiment-021/candidate-calibration.json`, which is installed
  byte-identical as `calibration-v1.json`, committed, and reviewed before `lock`.
- `lock` (no forward pass): the Y1 prediction table for the 24 fresh cues in the 108 exposed frames, with the floor
  tables of the committed calibration record; installed as `preregistration-lock.json` and `predictions.md` and
  committed by hand before the sign-off.
- `confirm` (once): stage 1 (S1-REF and S1-VALIDITY only), the Y2 floor row selected from the validity verdicts and
  digested, the barrier (both digests re-verified from disk, the row recomputed from the re-read verdicts), stage 2
  with the downstream ceiling, the Y3 row from the measured scorability, the scoring through the one pass predicate.
- `report` renders `outputs/experiment-021/report.md`, verifying the per-row draw values against the record.

## Status — 2026-09-23: complete; outcome `CONTRAST_PREDICTED_TOKENS | CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL | NOUN_READOUT_FIXED`

Tasks 1–4 of the plan were implemented and independently reviewed twice (no blocker; the notes below):
`src/neural_decompiler/readout_calibration.py`, the runner, 23 unit tests (`tests/test_readout_calibration.py`, tier A)
and 18 runner tests on a fake-closed Experiment 020 world (`tests/test_experiment_021_runner.py`, tier B); tier C
passed on the clean `c49c16d` (438 tests). Each scientific phase then ran exactly once, with a review gate between
phases:

- **`calibrate`** at `c49c16d` (run `9725893f39907cbd`; 09:50:45–11:02:47 UTC; CPU, float32, 4 threads — the runtime
  and dependency versions of Experiment 020's explore; no incident). It re-executed exactly 020's 30,132 exposed ledger
  keys: the 108 re-captured reference states and the template bases matched 020's records exactly (drift 0.0,
  tolerance 1e-9); every identity maximum was bitwise equal to 020's (readout 5.23e-3, logit 3.26e-3, Level 1 4.59e-3
  against 7e-3, additive 4.11e-6, reference component sum 1.24e-6, inherited 017 0); the reproduction gate over
  30,024 pairs × 79 nouns was exactly 0.0 (tolerance 1e-9); the validity screen kept all 14/14/14 primary frames (none
  of the 108 exposed frames invalid; precondition ≥ 6); and the kernel agreed with 020's direct statistics to 1.64e-15
  over 2,384 row-draws (worst at Y2 `5/3/6`, draw 3, `frame_mean_r2`; tolerance 1e-10). The record was installed
  byte-identical as [`calibration-v1.json`](calibration-v1.json) (file sha256 `240e0345…`, content `f939a84d…`) and
  committed alone (`ad22671`); the floor review passed.
- **`lock`** at `ad22671` (11:31–11:34 UTC), with no forward pass: weights only; the 2,592 Y1 rows (24 fresh cues ×
  108 exposed frames, each with the predicted `Δĉ` of 79 exposed and 24 fresh nouns) recomputed under the
  capture-disabling guard with a provenance difference of exactly 0.0; the ledger unchanged. The lock (content
  `5491611e…`, file `56eecfa1…`) and [`predictions.md`](predictions.md) (`ceed61b1…`) were installed byte-identical
  and committed alone (`b7e8861`). An independent read-only lock review — which re-derived all 149 floors from the
  digest-bound draw values with zero mismatches, reproduced the lock rows at 0.0 and found that `validate_lock` refuses
  every tampering it tried — passed with notes (below), and the reviewer authorized the single confirmation.
- **`confirm`** at `b7e8861` (12:19:33–12:24:44 UTC, exit 0, no incident). Before any fresh prompt the lock was
  validated and its rows reproduced with difference 0.0. **Stage 1** ran only the 18 S1-REF and 18 S1-VALIDITY prompts:
  all 18 fresh frames were valid (head informative; cue effect 78–79 of 79 against 72 required; plural head change
  1.14–3.08), so the valid-frame composition is 6/6/6 and the Y2 floor row `6/6/6`; the stage-1 digest (`8af7c000…`)
  and the row-selection digest (`fda5f936…`) were re-verified from disk at the barrier, where the stage-1 rows also
  reproduced at 0.0 and no stage-2 target was in the ledger. **Stage 2** ran the 2,592 Y1 pairs and the 432 pairs of
  the fresh cues in the 18 fresh frames; every measurement reached disk with its digests before the identities were
  enforced (stage 1 / stage 2 maxima: readout 4.17e-3 / 3.73e-3, logit 2.41e-3 / 3.00e-3, Level 1 1.55e-3 / 3.10e-3
  against 7e-3, additive 1.04e-6 / 1.35e-6; reference contrast 2.95e-3, reference component sum 8.71e-7, inherited
  017 0). All 24 fresh nouns were scorable, so the Y3 row is `8/8/8`. The kernel agreed with the direct statistics on
  the fresh tables to 2.2e-16, and Experiment 020's closure verified again after the phase. Ledger after confirm:
  33,192 prompt keys — 020's 30,132 plus all 3,060 manifest keys (18 + 18 + 3,024) — and 80 noun keys; the runner now
  refuses every phase.

The final report is copied verbatim to [`evidence/final-report-2026-09-23.md`](evidence/final-report-2026-09-23.md)
and the confirmed record to
[`evidence/confirmation-record-2026-09-23.json`](evidence/confirmation-record-2026-09-23.json) (content sha256
`678714b0ee1257b0a87438ca15b0ae5ec46c1222ef6aae5f7d78335eb4960793`; the 18 stage-1 reference states are kept as their
digests and the 432 stage-1 prediction rows as one digest). The gitignored results state it is extracted from has file
sha256 `8caf2a026cdb0b8f831a82abc7b90393ed33ec82399c56ec1eb70dbecfd3b73e` and state digest
`80b9b654e256106ccf19bef3cbd58f641567f6e8009916cb6cea0085880cbb85`. Experiment 021 is closed; nothing is amended or
rerun.

### Result

Pass predicates (frozen): an `R²`-type condition passes iff its value is finite, `> 0` and `≥` the floor; an error-type
condition iff it is finite and `≤` the floor. "Clamped" marks a floor whose raw 2.5 % tail over the 10,000 exposed-like
draws was negative and was raised to 0.0 by the zero-skill clamp; the percentile is the share of those draws that the
fresh value meets or beats.

| outcome | statistic | fresh | floor | result | clamped | raw tail | draw median | percentile |
|---|---|---|---|---|---|---|---|---|
| Y1 | `token_mean_r2` | 0.016018 | 0.0 | pass | yes | −1.170348 | −0.1451 | 65.7 % |
| Y1 | `pair_mean_r2` | 0.342111 | 0.0 | pass | yes | −0.004181 | 0.2388 | 77.2 % |
| Y1 | `cue_mae_k80` | 0.762002 | 1.025830 | pass | no | = floor | 0.8770 | 99.2 % |
| Y1 | `pooled_mae` | 0.647807 | 0.797678 | pass | no | = floor | 0.7268 | 98.8 % |
| Y2 | `frame_mean_r2` | 0.280706 | 0.0 | pass | yes | −1.247695 | 0.2642 | 51.6 % |
| Y2 | `frame_r2_k75` | 0.449572 | 0.0 | pass | yes | −1.032826 | −0.0605 | 92.9 % |
| Y2 | `cue_final_r2` | 0.844526 | 0.742235 | pass | no | = floor | 0.8170 | 82.2 % |
| Y2 | `coordinated_r2` | −0.153795 | 0.0 | **fail** | yes | −1.283415 | −0.1066 | 45.8 % |
| Y3 | `noun_median_r2` | 0.446696 | 0.255338 | pass | no | = floor | 0.4259 | 59.7 % |
| Y3 | `noun_r2_k90` | 0.285857 | 0.035628 | pass | no | = floor | 0.2579 | 60.1 % |
| Y3 | `noun_slope_dev_k90` | 0.158791 | 0.224024 | pass | no | = floor | 0.1696 | 73.6 % |
| Y3 | `noun_bias_k90` | 0.493891 | 0.767312 | pass | no | = floor | 0.6017 | 92.8 % |

- **Y1 — fresh cues in the exposed frames (24 cues × 108 frames, 2,592 pairs, 79 exposed nouns):
  `CONTRAST_PREDICTED_TOKENS`.** Precondition met (24 scored cues, 108 frames). All four conditions pass, the two
  clamp-bound `R²` conditions with positive skill; the token-mean margin is small (0.016), on a condition that
  exposed-like sets met only 35.8 % of the time. End-to-end Level-0 flattened `R²` 0.6167 (descriptively: cardinal
  0.837, quantifier 0.840, coordinated-adjective −0.014).
- **Y2 — fresh cues in the 18 fresh frames (432 pairs): `CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL`.** Precondition
  met (18 valid frames, 6 of them coordinated). Three of four conditions pass (`frame_r2_k75` at the 92.9th percentile
  of the exposed-like draws); the label comes solely from `coordinated_r2` = −0.1538 against its clamped floor 0.0.
  Flattened `R²` 0.5797 (descriptively: cardinal 0.843, quantifier 0.823, coordinated-adjective −0.154).
- **Y3 — fresh nouns over the Y1 pairs (24 nouns, 8/8/8): `NOUN_READOUT_FIXED`.** All four conditions pass (none has a
  clamped floor), at the 59.7th–92.8th percentiles; per-noun `R²` median 0.447, lowest `cavity` 0.251.
- **Ceiling and comparators.** Fed the measured `Δx3`, the same readout program reaches flattened `R²` 0.9683 on Y1
  (75.6th percentile of its row's draws) and 0.9652 on Y2 (37.3rd). Of the unexplained variance, 0.352 of 0.383 (Y1)
  and 0.385 of 0.420 (Y2) is inherited from the upstream prediction of `Δx3`, and 0.032 / 0.035 is downstream. The 020
  comparators keep their standing: without the layer-5 heads 0.4848 / 0.4781, template-base MLPs 0.5115 / 0.4838,
  `ΔT` only −0.5382 / −0.2027, rank-1 nouns (fresh-noun `R²`) 0.2000 / 0.2158. The joint diagnostic (fresh cues ×
  fresh frames × fresh nouns, no threshold) reaches 0.5858.
- **Calibration context.** On the committed floors, exposed-like pseudo-confirmation sets pass Y1 35.5 %, Y2 (row
  `6/6/6`) 25.3 % and Y3 92.3 % of the time, and all three together 16.6 %: the zero-skill clamp binds for Y1's two
  `R²` conditions, for Y2's `frame_mean_r2` and `coordinated_r2` in all 64 rows and for `frame_r2_k75` in 63 of 64.
  Design revision 3 anticipated and froze this; the rates are reported, never used to move a floor.

### Interpretation (agreed at the floor and lock reviews; the design file is not edited)

With the clamp binding, Y1 and Y2 test two things at once: that the fresh set lies within the exposed-like envelope,
and that the program has positive absolute skill on it. A pass therefore establishes both. A failed condition whose
floor is clamped to 0.0 establishes only that **positive predictive skill was not established under that criterion**;
it is evidence of a generalization gap only if the fresh value also falls below the raw calibration tail. The "Reading"
paragraph of the design's "Outcomes" section says that any negative label means a generalization gap beyond sampling
variability; that is too strong for failures caused only by the clamp, and for them it is superseded here. Y2's single
failure is of this kind: `coordinated_r2` = −0.154 sits at the 45.8th percentile of the exposed-like draws (median
−0.107), far above the raw tail −1.283, so no fresh-specific degradation is shown. A failure on an unclamped condition,
or below its raw tail, would be reported as a genuine negative; none occurred.

What this settles and what it does not. Stated at its safe strength: the decoded downstream program — blocks 3–5,
`LN_final` and the fixed noun read, driven by the committed Experiment 011/012/017 chain — prospectively predicts the
model's own singular-versus-plural logit contrast for 24 never-executed cues in the 108 exposed frames on every Y1
criterion, and for 24 never-inspected nouns on every Y3 criterion: the fixed unembedding read generalizes to new nouns
of all three rule classes. In 18 new frames, conditional on each frame's stage-1 reference state, three of four
criteria pass; in the coordinated-adjective frames the program's absolute skill is not positive — nor was it on
exposed-like data — so the coordinated template remains the weak part of the account, now measured prospectively.
The ceiling locates the remaining error: given the measured `Δx3` the readout is nearly exact (0.97), and most of the
gap between Level 0 and the ceiling is the inherited upstream prediction of `Δx3`. This is not a claim of good
prediction everywhere (the floors are relative to the program's own exposed performance, and the token-mean margin is
small), nor that a frame's state is predicted from its text, nor anything about behaviour beyond `Δc` at `p_t`.
Limits: one checkpoint, three templates, four cue classes, three noun rule classes, 18 new frames. Claim
[C002](../../research/claims/C002-count-cued-noun-number-circuit.md) is unchanged (`LOCALIZED`); this design names no
claim transition.

### Review notes carried forward

- The runner's report prints a clamped floor as `0.0000` without its clamp flag or raw tail; the table above adds both.
- `confirm` has no resume path by design (an interruption would have used up the protocol version); it ran straight
  through. A fresh Level-1 error above 7e-3 would have been an incident with no label; the maximum was 3.10e-3.
- `outputs/experiment-021/` (gitignored) holds the results state, the draw values, the exposed table and the stage-2
  tables; the committed extract carries their digests.

### Implementation notes (from the independent implementation review; nothing scientific changed)

The review found no blocker. Its should-fix and minor items are all in the code:

- Incident records are always writable: a non-finite value (a one-sided undefined statistic, a structural mismatch)
  is stored as `null`; the kernel/direct cross-check completes every row-draw and records the **worst** location with
  the first one and the count, then stops — before any floor or candidate record exists.
- The kernel pools every sum of squared deviations from per-group two-pass moments (Σ M2ᵢ + Σ nᵢ(x̄ᵢ − x̄)²), never
  from Σy² − (Σy)²/n. The implementation-only agreement guard is applied to |kernel − direct| / max(1, |direct|) ≤
  1e-10: absolute for well-conditioned values, relative where a statistic is itself ill-conditioned (a frame-mean
  `R²` of −16,539 on a synthetic draw with four near-identical frame means differed by 2.2e-9 absolute, 1.3e-13
  relative — float reassociation, not a formula difference).
- `calibrate` resumes only after a recorded incident and never at a commit that carries one; an interruption is
  recorded as an incident; the candidate record's digest reaches the state as the record is written; the after-phase
  re-check of Experiment 020's files runs inside the incident handling of `calibrate`, at the end of `lock` and
  `confirm`, and in every incident record.
- `lock` checks the runtime and thread count against 020's explore record, so the locked rows reproduce at `confirm`.
- `confirm` writes every stage-2 measurement (with digests) and the identity maxima to disk before enforcing the
  identities or scoring, so no failure after stage 2 can lose a fresh measurement.
- The 011/012/017 lock digests (`769bfeac…`, `830abc3b…`, `b4fc9014…`) are asserted against frozen constants and
  bound into the 021 results state (so every later phase enforces them), the calibration record's `inputs` (checked
  against the current inputs by `lock` and `confirm`) and the 021 lock (checked by `confirm`); a self-consistently
  replaced lock can reuse neither the state nor the record. The calibration record's constants, tolerances,
  preconditions, program blob and design are verified before `lock` uses it; `rematerialize` calls 020's own
  `assert_explore_nouns`; the report prints each fresh value's exposed draw median and percentile and places the fresh
  ceiling within its row's distribution.

A second independent review, of every change since `6004561`, found no blocker; its items are in the code too: the
model loads inside the recorded region; an incident reaches disk before the slow re-check of Experiment 020; no
commit that ever attempted `calibrate` is reused, and an attempt that ended without writing its incident (a kill) is
recorded as one at the next start; the after-phase checks run before the candidate record is written; tests pin the
two-pass pooling (a 1e4 offset), identity enforcement failing after stage 2 with the measurements already on disk, and
the ceiling column the report ranks against.
