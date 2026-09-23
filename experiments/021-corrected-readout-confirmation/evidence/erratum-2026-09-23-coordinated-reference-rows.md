# Experiment 021 — post-closure erratum (2026-09-23): the coordinated-frame layer-1/2 reference rows

This erratum was added after the closure (`86f7f8c`). It changes none of Experiment 021's frozen results, labels,
floors, digests or reports, including the calibration record, the lock, the predictions, the committed confirmation
record and the final report. The outcome stays
`CONTRAST_PREDICTED_TOKENS | CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL | NOUN_READOUT_FIXED`.

## What was found

Experiment 021 tested Experiment 020's program unchanged (readout-program blob `caa73b40…`). That program contains a
defect found by the independent implementation review of Experiment 022 on 2026-09-23, before any Experiment 022
scientific phase ran:

- **The defect.** In coordinated-adjective frames (`p_t = p_c + 1`), 020's Level-0 program builds the layer-1/2
  reference rows from the locked residuals through `p_t`, not through the cue position `p_c` as the chain validated in
  017–019 does.
- **The effect.** The reduced upstream computes the cue's layer-1/2 attention change for the wrong query and inserts the
  cue's changed key and value at the wrong key position.
- **Where 021 inherited it.** 021 built its rows the same way (`readout_calibration.py` line 449, the re-materialization).
- **Where it came from.** The details and the 017–019 call sites are in Experiment 020's erratum,
  [`../../020-readout-decompilation/evidence/erratum-2026-09-23-coordinated-reference-rows.md`](../../020-readout-decompilation/evidence/erratum-2026-09-23-coordinated-reference-rows.md).

Measured size, on exposed data only, with weights only and no prompt: over the 36 coordinated exposed frames × the 175
pool cues, the coordinated flattened `Δc` `R²` is −0.146 with the 020 wiring against 0.555 with the 017 wiring. The exact
layers 1–2 reach 0.558 and the full composition 0.968. Cue-final frames are unaffected (difference exactly 0).

## What stands

- **The recorded result.** 021's calibration, floors, lock and confirmation all used the same program, so the recorded
  result remains a valid prospective test of that frozen program. The wiring is part of that program.
- **`CONTRAST_PREDICTED_TOKENS`** (fresh cues in the exposed frames) and **`NOUN_READOUT_FIXED`** (fresh nouns) stand.
- **The ceiling.** 0.968 on Y1 and 0.965 on Y2; the readout fed the measured `Δx3` uses no reference rows. The
  statement that most of the remaining error is the inherited upstream prediction of `Δx3` also stands.

## What changes in interpretation

**The Y2 failure.** 021's single failing condition was Y2 `coordinated_r2` = −0.154, which failed against its
zero-clamped floor. It must no longer be read as evidence about the correctly wired 017 upstream chain. It is plausibly
explained by the 020 wiring defect. 021's spent confirmation set has not been re-examined; the exploratory
`replicate-021` of Experiment 022 is the only allowed route, and only after 022's report.

**The coordinated template.** The README's reading that "the coordinated template remains the weak part of the
account, now measured prospectively" describes the program as implemented in 020, not the validated 017 chain.

**The descriptive coordinated-adjective Level-0 values.** Y1 −0.014, Y2 −0.154 and the exposed-like draws describe the
same wiring.

Experiment 022 (design revision 4) uses the 017 wiring as the authoritative reduced layer-1/2 computation. It keeps the
020 wiring only as a descriptive historical comparator.
