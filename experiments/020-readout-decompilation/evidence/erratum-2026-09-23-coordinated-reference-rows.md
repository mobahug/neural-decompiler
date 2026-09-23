# Experiment 020 — post-closure erratum (2026-09-23): the coordinated-frame layer-1/2 reference rows

This erratum was added after the closure (`7f4a4f8`). It changes none of Experiment 020's recorded results, digests,
reports or its closure record ([`closure.json`](../closure.json), content sha256 `f2b1b5e7…`), and it assigns no label.

## What was found

The independent implementation review of Experiment 022 found the defect on 2026-09-23. No Experiment 022 scientific
phase had run.

Experiment 020's Level-0 program is the committed Experiment 017 chain at `LEVEL0`. The two experiments build the
layer-1 and layer-2 reference rows (`rows16`) from different positions:

- **017–019:** from the locked residuals at positions 0..`p_c`, the cue position:
  - `head_pattern.py`: the 013 state keeps `x1_all[: p_c + 1]` (line 186); `HeadChainModel.parts` (line 426);
  - `block_concentration.py` line 418;
  - `block_routing.py` line 467.
- **020:** from positions 0..`p_t`, because `state.state_017.x1_all` spans the whole prefix up to the target:
  - `readout_decompilation.py` lines 1395, 1552, 1813 and 1858.

`atp.ReferenceRow` treats the last residual it is given as the cue row. So the two constructions agree in cue-final
frames (`p_t = p_c`) and differ in coordinated-adjective frames (`p_t = p_c + 1`). There, the reduced upstream program:

- computes the cue's layer-1 and layer-2 attention change for query `p_t` instead of `p_c`;
- inserts the cue's changed key and value at key `p_t`, with `p_t`'s rotary angle;
- leaves the real cue key at `p_c` at its reference value.

Two parts are unaffected: the MLP term, which uses `x1(p_c)`, and the one-step propagation to `p_t`.

Two gates could have caught this, and neither did:

- Experiment 020 did not carry 017's reference-row identity (`atp.check_reference_rows`).
- 020's inherited-reproduction identity recomputes the prediction with the same rows, so it agrees with itself.

## Size (exposed data only, weights only, no prompt)

The review measured the effect on the 36 coordinated exposed frames × the 175 pool cues of Experiment 021's calibration
(6,300 pairs). Measured `Δc` comes from Experiment 021's digest-bound exposed table. The flattened `Δc` `R²` of each
program:

| Program | `R²` |
|---|---|
| Level 0 as wired in 020 (and 021) | −0.146 |
| Level 0 as wired in 017 | 0.555 |
| Exact layers 1–2 on the same cue input | 0.558 |
| Full layer-0–2 composition (≈ the ceiling) | 0.968 |

- **Coordinated frames:** the Level-0 `Δx̂3` of the two wirings differs by a median 48 % (relative L2 per position).
- **Cue-final frames:** the two wirings give identical values (difference exactly 0).
- **Check:** the 017-wired computation equals 017's `HeadChainModel.upstream` exactly.

## What stands and what changes

**Stands:**
- Every recorded Experiment 020 number, digest, report and the closure decision.
- The downstream readout and its ceiling: 0.976 when fed the measured `Δx3`, where no reference rows are involved.
- All cue-final Level-0 values.

**Changes in interpretation.** The coordinated-adjective Level-0 descriptives describe 020's wiring, not the chain
validated in 017–019. This covers the coordinated flattened `R²` 0.273 and the coordinated template's share of the
exposed shortfall. The Closure section's reading needs a qualifier:

- It says the weakness of the end-to-end prediction is the inherited upstream prediction of `Δx3`, "above all in
  coordinated-adjective frames".
- That reading holds for the program as implemented in 020.
- In coordinated frames, a large part of that upstream error comes from this implementation defect, not from a
  limitation of the validated 017 chain.

The companion erratum for Experiment 021 is
[`../../021-corrected-readout-confirmation/evidence/erratum-2026-09-23-coordinated-reference-rows.md`](../../021-corrected-readout-confirmation/evidence/erratum-2026-09-23-coordinated-reference-rows.md).
Experiment 022 (design revision 4) uses the 017 wiring as the authoritative reduced layer-1/2 computation. It keeps the
020 wiring only as a descriptive historical comparator.
