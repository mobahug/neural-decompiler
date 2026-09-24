# Experiment 023: Prospective Block-0 Completion

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
`46585696…`) is pinned by git blob together with everything it calls; a change anywhere in it refuses every phase.

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

One implementation (`pool` → `statistics` → `classify`) scores both the calibration draws and the fresh conditions.

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
- **`extract`** runs once, weights only, under the no-forward-pass guard. It reads 022's table and writes
  `outputs/experiment-023/candidate-exposed-cells.f64/.json`. The candidate is installed byte-identically as
  `exposed-cells.f64/.json`, committed, and checked before anything else. Its identities:
  - E1: the table's tensor digests and orders;
  - E2: the cells equal 022's stored cells bit for bit;
  - E3: the cells, computed twice, are bit-identical;
  - E4: `P1` recomputed from the weights (1e-9);
  - E5: the cells against the per-noun table on the first 16 draws (1e-10);
  - E6: the pooled `SST` (1e-10).

  E1–E3 stop for review; E4–E6 are incidents.

  The artifact holds 18,900 pairs × 8 columns (`n, S, Q, SSE0, SSE1, SSEC, mean, M2`; 1,209,600 bytes). Its index binds:
  - the canonical pair order, every cue's id and stratum, and every frame's template and group;
  - the column schema;
  - 022's table and record digests;
  - the extraction module's blob, version and commit.
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
- **`lock`** (no forward pass) builds the Y1 prediction table (`[1728, 2, 79]` then `[864, 2, 79]`, `(P0, P1)`) twice
  and requires it bit-identical, with I5 exactly 0 and the block-0 algebra at 1e-12. It then writes the lock and the
  preregistration; all four files are installed byte-identically, committed and reviewed.
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
  - descriptive records: per-template and per-stratum `g`, the cheaper block-0 rules, `P1`'s `Δx3` error and block 0's
    profile.

  An incident is recorded and stops the phase; nothing is retried.
- **`report`** renders `outputs/experiment-023/report.md`.

## Status — 2026-09-24: implemented; not run

- **Implementation.** Plan Tasks 1–6 are implemented in `src/neural_decompiler/block0_completion.py`, this runner and
  their tests:
  - tier A: `tests/test_block0_completion.py`;
  - tier B: `tests/test_experiment_023_runner.py`, on the fake world of 022's runner test;
  - tier C, opt-in: the real tokenizer's freeze gives exactly the design's expected picks; E1–E3, E5 and E6 hold
    read-only on the real local 022 table; `P0`/`P1` and the prediction path reproduce 022's stored coalitions bit for
    bit on real exposed pairs.
- **Not run.** Nothing has been extracted, frozen, calibrated, locked or confirmed.
- **Next steps, each only when authorized:**
  1. the independent implementation review and tier C on the clean commit;
  2. `extract`, then install, commit and the artifact's own checkpoint;
  3. `freeze`, then commit;
  4. `calibrate` once, then install, commit and the floor review;
  5. `lock`, then install, commit and the lock review;
  6. `confirm` once;
  7. `report`;
  8. closure.
