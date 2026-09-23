# Experiment 022: Prospective Localization of the Upstream `Δx3` Error

Implements the design
[`docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md`](../../docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md)
(revision 4, `219cdc5`) through the plan
[`docs/superpowers/plans/2026-09-23-experiment-022-upstream-error-localization-plan.md`](../../docs/superpowers/plans/2026-09-23-experiment-022-upstream-error-localization-plan.md)
(revision 3, `e6d8299`; the R-1 clarification on the Y2 table stands).

The question is where the committed upstream `Δx3` prediction loses the gap between Level 0 and the `Δc` ceiling. The
committed prediction is the Experiment 017 chain, wired as 017 validated it. The method is an exact five-factor Shapley
attribution over:
- the layer-1–2 reductions (`R`);
- the embedding change (`emb`);
- block 0's value and pattern terms at the cue position (`Bv`, `Bp`);
- block 0's change at the target position (`T`, coordinated frames only).

Four claims are evaluated separately on Y1 (24 new cues × the 108 exposed frames) and on Y2 (24 new cues × 18 new
frames):
- **C1:** block-0 attention dominates the cue-final gap;
- **C2:** the reductions contribute little to the cue-final gap;
- **C3:** the reductions contribute little to the coordinated gap;
- **C4:** block-0 cue→target attention contributes positively to the coordinated gap.

That gives eight condition results and no aggregate label.

**The wiring (design revision 4).** The reduced layer-1–2 program builds its layer-1/2 reference rows through the cue
position `p_c`, as in 015–019. Experiment 020 built them through `p_t`: the defect recorded in the post-closure errata
of 020 and 021. That Level 0 is kept only as a descriptive historical comparator. The claims were rewritten after the
exposed spike was recomputed with the corrected wiring. Revision 3's C3 ("the coordinated gap is split") rested on the
defect and was withdrawn.

## Inputs

- The frozen modules, checked by git blob on every phase (`ul.FROZEN_BLOBS`): 020's readout program, 021's
  calibration module, the 017/016/015/012 programs, the 011 read, the 009 head weights, `plural_mechanism` and
  `models`. Nothing in 022 edits them.
- Experiment 020's closure and ledger (the calibration's exposed keys and locked reference states), read exactly as
  Experiment 021 read them.
- Experiment 021's committed calibration record (`240e0345…`, content `f939a84d…`) and its local exposed table
  (`outputs/experiment-021/exposed-table.pt`, measured digest `799ec908…`): the R1 reference. Nothing of 021's spent
  confirmation set enters the calibration or the confirmation; `replicate-021` reads it only after the report.

## Commands

```bash
HF_HUB_OFFLINE=1 uv run python experiments/022-upstream-error-localization/run.py validate
HF_HUB_OFFLINE=1 uv run python experiments/022-upstream-error-localization/run.py freeze
HF_HUB_OFFLINE=1 uv run python experiments/022-upstream-error-localization/run.py calibrate
HF_HUB_OFFLINE=1 uv run python experiments/022-upstream-error-localization/run.py lock
HF_HUB_OFFLINE=1 uv run python experiments/022-upstream-error-localization/run.py confirm
HF_HUB_OFFLINE=1 uv run python experiments/022-upstream-error-localization/run.py report
HF_HUB_OFFLINE=1 uv run python experiments/022-upstream-error-localization/run.py replicate-021
```

- `validate` (no model): the frozen inputs and module blobs, 020's closure, 021's record and exposed table, the pools
  and — once they exist — the confirmation file and the results state.
- `freeze` (tokenizer only, under the no-forward-pass guard): the first 6 eligible cues per class and frames per
  template of the frozen ordered lists; writes `confirmation-v1.json`, which is committed by hand before anything else.
  A shortfall writes nothing and stops for review.
- `calibrate` (once): the 18,900 exposed pairs (175 pool cues × 108 frames, every key in 020's ledger, none in any
  manifest; one forward each) against 020's locked states; every coalition's composition; I1–I5 maxima written before
  enforcement; R1 against 021's digest-bound table at `1e-9`; 10,000 SHA-indexed draws; the kernel with its direct
  cross-check on the first 16 draws; I6; the undefined-draw stop (≥ 250 for C1, C2, C4; ≥ 125 for C3: no floor, stop
  for review); the envelopes with their direction checks; the result rates. Writes
  `outputs/experiment-022/candidate-calibration.json`, installed byte-identical as `calibration-v1.json`, committed,
  and reviewed before `lock`.
- `lock` (no forward pass): the Y1 composition table (24 cues × 108 frames, cue-final `[1728, 16, 79]` then coordinated
  `[864, 32, 79]`, raw little-endian float64) computed twice and required bit-identical, the candidate lock (the eight
  conditions, the Y1 companion's digests, the Y2 table's construction, format, layout and orders) and the
  preregistration. All four files are installed byte-identical (`preregistration-lock.json`, `preregistration.md`,
  `locked-y1-table.f64`, `locked-y1-table.json`), committed, and independently reviewed before `confirm`.
- `confirm` (once, never resumed): the lock validated; I7 (the Y1 table rebuilt from the weights and the locked inputs,
  bit for bit, before any fresh prompt); stage 1 (each new frame's S1-REF and S1-VALIDITY prompts; validity is
  descriptive and selects nothing) and the Y2 table, written once to `outputs/experiment-022/y2-table.f64` with its
  digests in the stage-1 record; the barrier (the state and the Y2 table re-read from disk and verified; no S2-TARGET
  key in the ledger); stage 2 (each of the 3,024 target prompts once); every measurement saved before any gate; the
  Y2 table re-verified after stage 2 (any change after the barrier is an incident); I1–I4; the scoring (the one kernel,
  its direct check, I6, `classify` against the locked envelopes and guards). An incident is recorded and stops the
  phase; nothing is restored or retried, and the Y2 table and the state are preserved.
- `report` renders `outputs/experiment-022/report.md`: the 2 × 4 conditions with their statistics, envelopes, guards,
  gaps, CDF percentiles (descriptive) and results; the Shapley values and shares; the descriptive records.
- `replicate-021` (only after the report): the exploratory replication on 021's spent set from its stored stage-2
  tables (digest-verified), I4, I5 and I6 only; numbers, no result names.

## The Y2 table (plan revision 2, R-1)

The lock binds the Y2 table's construction, byte format, factor and unit orders and dimensions; its numbers depend on
the new frames' stage-1 reference states and therefore exist only at stage 1. The scientific freeze of the Y2 table is
the stage-1 barrier. After a successful confirmation it is committed as closure evidence (archival only).

## Storage and runtime (estimates on this machine)

- `calibrate` ≈ 2 h (18,900 forwards and about 403,000 coalition compositions); `outputs/experiment-022/calibration-table.pt`
  holds every coalition's `Δĉ` of every pair (`[175, 108, 32, 79]` float64, ≈ 0.4 GB) so that any draw can be
  recomputed from disk; the draws ≈ 3 MB; peak memory ≈ 1.5 GB.
- `lock` ≈ 35 min (the Y1 table twice); the committed companion ≈ 35 MB.
- `confirm` ≈ 30 min (I7 ≈ 17 min, stage 1, the Y2 table ≈ 6 MB, 3,024 forwards, the gates, the scoring).

## Status — 2026-09-23: implemented to design revision 4; not run

- **Tasks 1–7.** Implemented and independently reviewed; findings 2–10 fixed (`8f061c1` … `566c21d`).
- **Task 8.** Finding 1 led to the errata, design revision 4, plan revision 3 and the patch.
- **Not run.** Nothing has been frozen, calibrated, locked or confirmed.
- **Next steps, each only when authorized:**
  1. the review of the revision-4 patch;
  2. tier C on the clean commit;
  3. `freeze`, then commit;
  4. `calibrate` once, then install, commit and the floor review;
  5. `lock`, then install, commit and the lock review;
  6. `confirm` once;
  7. `report`;
  8. closure.
