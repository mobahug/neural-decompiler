# Experiment 022: Prospective Localization of the Upstream `Δx3` Error

Implements the design
[`docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md`](../../docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md)
(revision 4, `219cdc5`) through the plan
[`docs/superpowers/plans/2026-09-23-experiment-022-upstream-error-localization-plan.md`](../../docs/superpowers/plans/2026-09-23-experiment-022-upstream-error-localization-plan.md)
(revision 3, `e6d8299`; the R-1 clarification on the Y2 table stands).

**Closed (2026-09-24).** The single confirmation ran once at `4f8f31d`, with no incident, and all eight preregistered
conditions passed individually, each against its frozen envelope and its meaning guard; there is no aggregate label.
On 24 never-executed cues and 18 never-executed frames, block-0 attention at the cue position carries 94–95 % of the
cue-final gap between Level 0 and the ceiling; the correctly wired layer-1–2 reductions carry 0–0.7 % of either gap;
and block 0's change at the target position contributes positively in coordinated frames (measured share 0.41 / 0.38,
descriptive). The post-report replication on Experiment 021's spent set is exploratory and changes nothing. See
[the status section](#status--2026-09-24-closed-all-eight-conditions-pass-no-aggregate-label) and the
[evidence](evidence/).

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
  cross-check on the first 16 draws; I6; the undefined-draw stop (≥ 250 undefined draws for any of the four claims,
  design revision 4: no envelope, stop for review); the envelopes with their direction checks; the result rates. Writes
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

Every phase has now run once and nothing is rerun. `calibrate`, `lock`, `confirm` and `replicate-021` refuse to run
again. `report` would re-render the report and must not be run: the official report is preserved byte-identical as
[`evidence/final-report-2026-09-24.md`](evidence/final-report-2026-09-24.md).

## The Y2 table (plan revision 2, R-1)

The lock binds the Y2 table's construction, byte format, factor and unit orders and dimensions; its numbers depend on
the new frames' stage-1 reference states and therefore exist only at stage 1. The scientific freeze of the Y2 table is
the stage-1 barrier. After the successful confirmation it was committed as closure evidence (archival only), byte-identical
to the file stage 1 wrote: [`evidence/y2-table.f64`](evidence/y2-table.f64) (sha256 `008e8e4e…`) and its index
[`evidence/y2-table.json`](evidence/y2-table.json) (`bde51b96…`).

## Storage and runtime (measured on this machine)

- `calibrate` 2 h 05 min (18,900 forwards and about 403,000 coalition compositions);
  `outputs/experiment-022/calibration-table.pt` holds every coalition's `Δĉ` of every pair (`[175, 108, 32, 79]`
  float64, 0.42 GB) so that any draw can be recomputed from disk; the draws 4.3 MB.
- `lock` 30 min (the Y1 table twice); the committed companion 35 MB.
- `confirm` 23.5 min (I7 16 min, stage 1 and the Y2 table 2 min, the 3,024 forwards 2 min, the gates and the scoring
  3.5 min); `replicate-021` 17 min; `report` 9 s.

## Status — 2026-09-24: closed; all eight conditions `PASS`; no aggregate label

Tasks 1–7 of the plan were implemented and independently reviewed; findings 2–10 were fixed (`8f061c1` … `566c21d`).
Finding 1 led to the post-closure errata of Experiments 020 and 021 (`2320345`), design revision 4 (`219cdc5`), plan
revision 3 (`e6d8299`) and the patch (`dd50231`, Task 8), which passed its review; tier C passed on the clean
`dd50231`. Each scientific phase then ran exactly once, with a review gate between phases:

- **`freeze`** at `dd50231` (2026-09-23 18:07 UTC; tokenizer only, under the no-forward-pass guard). It took the first
  6 eligible cues per class and frames per template of the frozen ordered lists: 24 cues — determiner-like
  `alternate`, `random`, `standard`, `regular`, `normal`, `common`; quantity `bulk`, `spare`, `dense`, `lengthy`,
  `lots`, `loads`; possessive-or-pronoun `thou`, `yourself`, `themselves`, `ones`, `others`, `naught`; adjective
  `square`, `wild`, `brave`, `calm`, `eager`, `fierce` — and 18 frames (`cardinal-022-1…6`, `quantifier-022-1…6`,
  `coordinated-adjective-022-1…6`), 3,060 manifest keys. [`confirmation-v1.json`](confirmation-v1.json) (file
  `1a9afef0…`, content `af848ae4…`) was installed exactly as produced and committed alone (`0de0fb8`).
- **`calibrate`** at `0de0fb8` (run `be529d6f87a9f24d`; 2026-09-23 18:17–20:22 UTC; CPU, float32, 4 threads — the
  runtime and dependency versions of Experiment 020's explore; no incident). It re-measured the 18,900 exposed pairs
  (175 pool cues × 108 frames, every key in 020's ledger) against 020's locked states: R1 max |Δc − 021| = 0.0
  (tolerance 1e-9), and the historical Level 0 equal to 021's recorded exposed Level 0 (difference 0.0); I1 1.43e-5,
  I2 2.2e-16, I3 5.61e-5, I4 1.30e-4, I5 exactly 0.0; kernel/direct agreement 1.3e-15 over 2,816 quantities; I6
  2.2e-16. No draw was undefined (0 of 10,000 for every condition; the stop is at 250), every direction check held, and
  each condition's exposed-like result rate is `PASS` 0.9751, `ENVELOPE_ONLY_FAILURE` 0.0249 (joint, descriptive only:
  Y1 all four 0.907, Y2 0.905, all eight 0.843). On the whole exposed pool (descriptive): cue-final gap 0.156 (σ Bv
  0.624, Bp 0.304, emb 0.069, R 0.003); coordinated gap 0.413 (σ Bv 0.532, T 0.371, Bp 0.057, emb 0.037, R 0.005);
  coordinated Level-0 `R²` 0.555 with 017's wiring against −0.146 with 020's. The record was installed byte-identical
  as [`calibration-v1.json`](calibration-v1.json) (file `db745653…`, content `46985fd5…`) and committed alone
  (`f947305`) after the independent calibration review (pass with notes: every number recomputed bit for bit or to
  ≤ 3.2e-15).
- **`lock`** at `f947305` (2026-09-24 05:38–06:08 UTC), with no forward pass (weights only, under the no-forward-pass
  guard): the Y1 composition table — 24 fresh cues × 108 exposed frames; cue-final `[1728, 16, 79]` then coordinated
  `[864, 32, 79]`, raw little-endian float64, 34,947,072 bytes — computed twice and bit-identical; I2 2.2e-16, I5
  exactly 0.0. The lock (file `0588456f…`, content `afe1b541…`), [`preregistration.md`](preregistration.md)
  (`9072d064…`), [`locked-y1-table.f64`](locked-y1-table.f64) (`cb9fbf74…`) and its index
  [`locked-y1-table.json`](locked-y1-table.json) (`a6f413b2…`) were installed byte-identical and committed alone
  (`4f8f31d`) after the independent lock review (pass with notes; the reviewer rebuilt the Y1 table from the weights
  at difference 0.0). The review consciously accepted the stale labels D4 and I3's narrow calibration margin (closure
  notes below); I3 stayed at 1e-4.
- **`confirm`** at `4f8f31d` (2026-09-24 07:02:59–07:26:32 UTC, exit 0, no incident). A read-only preflight passed
  first (a clean tree equal to `origin/main`, the installed digests, `validate_lock` with tracked-file enforcement,
  ledger isolation, no Y2 table, runtime and versions, AC power), and an independent rebuild of the Y1 table outside
  the runner matched the committed file byte for byte. In the runner, `validate_lock` passed and I7 rebuilt the Y1
  table bit for bit before any fresh prompt. **Stage 1** ran only the 18 S1-REF and 18 S1-VALIDITY prompts (all 18
  frames valid; descriptive, selects nothing) and wrote the Y2 table once (`[288, 16, 79]` then `[144, 32, 79]`,
  5,824,512 bytes, `008e8e4e…`; index `bde51b96…`; I2 4.4e-16, I5 exactly 0.0); the stage-1 record (digest
  `76d17221…`) carries its digests. At the **barrier** the state and the Y2 table were re-read from disk and matched
  the digests stage 1 held in memory, and no S2-TARGET key was in the ledger. **Stage 2** ran the 3,024 target prompts
  once each (Y1 2,592 = 24 × 108; Y2 432 = 24 × 18); every measurement reached disk (`stage2-measurements.pt`,
  `3332a434…`) before any gate, and the Y2 table was unchanged afterwards. Gates: I1 1.64e-5 (tolerance 1e-4), I2
  4.4e-16 (1e-12), I3 4.23e-5 (1e-4), I4 7.69e-5 (1e-3); kernel/direct agreement 1.3e-15 over 176 quantities
  (1e-10); I6 ≤ 5.6e-17 (1e-12). Experiment 020's closure and 021's record verified again after the phase. Ledger
  after confirm: 21,960 prompt keys — the 18,900 exposed calibration keys, all in 020's ledger, and all 3,060 manifest
  keys (18 + 18 + 3,024) — and none of 021's spent set.
- **`report`** at `4f8f31d` (07:37:58–07:38:07 UTC): copied verbatim to
  [`evidence/final-report-2026-09-24.md`](evidence/final-report-2026-09-24.md) (`df06f131…`). A re-render from the
  confirmed state is byte-identical, and only the report entry and phase changed in the state.
- **`replicate-021`** at `4f8f31d` (07:58:14–08:15:28 UTC): the exploratory replication below.

The gitignored results state (`outputs/experiment-022/results.json`, file `a015a98a…`, state `9f34db69…`) is extracted
to [`evidence/confirmation-record-2026-09-24.json`](evidence/confirmation-record-2026-09-24.json) (content sha256
`ef393bb8cbf30110431cf5d970b8d97d73e79a5e836c5d157683785d03b7dff7`; the 18 stage-1 reference states are kept as their
digests and the prompt ledger as counts and a digest). The extract holds the exact values of everything below.
Experiment 022 is closed; nothing is amended or rerun.

### Result — the primary prospective evidence

Each condition is read on its own; Y1 and Y2 are never pooled; there is no aggregate label and no all-pass
requirement. All four games are interpretable (SST > 0 and G ≥ 0.02 on the full group). The envelope is the frozen
exposed-like order statistic (lower bounds: element [249], upper bounds: element [9750] of the 10,000 draws). The CDF
percentile — the share of the 10,000 defined calibration draws at or below the fresh value — is descriptive only; for
the upper bounds C2 and C3 a high percentile lies toward the unfavorable tail.

| condition | claim | statistic | fresh | envelope | meaning guard | G | calibration median | CDF | result |
|---|---|---|---|---|---|---|---|---|---|
| Y1/C1 | block-0 attention dominates the cue-final gap | σ Bv + σ Bp, cue-final | 0.94789 | ≥ 0.901075 | ≥ 0.50 | 0.160230 | 0.927639 | 0.9265 | **PASS** |
| Y1/C2 | the reductions contribute little to the cue-final gap | σ R, cue-final | −0.0000118 | ≤ 0.005595 | ≤ 0.10 | 0.160230 | 0.002555 | 0.0246 | **PASS** |
| Y1/C3 | the reductions contribute little to the coordinated gap | σ R, coordinated | 0.0059509 | ≤ 0.006904 | ≤ 0.10 | 0.337507 | 0.004538 | 0.8873 | **PASS** |
| Y1/C4 | block-0 cue→target attention contributes positively | σ T, coordinated | 0.40814 | ≥ 0.327059 | > 0 | 0.337507 | 0.373974 | 0.8946 | **PASS** |
| Y2/C1 | block-0 attention dominates the cue-final gap | σ Bv + σ Bp, cue-final | 0.93808 | ≥ 0.888785 | ≥ 0.50 | 0.159868 | 0.922885 | 0.8152 | **PASS** |
| Y2/C2 | the reductions contribute little to the cue-final gap | σ R, cue-final | 0.0038511 | ≤ 0.008226 | ≤ 0.10 | 0.159868 | 0.000975 | 0.7961 | **PASS** |
| Y2/C3 | the reductions contribute little to the coordinated gap | σ R, coordinated | 0.0069294 | ≤ 0.014172 | ≤ 0.10 | 0.410088 | 0.006752 | 0.5205 | **PASS** |
| Y2/C4 | block-0 cue→target attention contributes positively | σ T, coordinated | 0.38338 | ≥ 0.207165 | > 0 | 0.410088 | 0.300087 | 0.9402 | **PASS** |

All eight conditions independently passed their frozen envelope and their meaning guard: each fresh attribution lies
within the exposed-like envelope and keeps its claim's meaning. The tightest margin is Y1/C3 (0.00595 against the
bound 0.00690; CDF 0.887, toward the unfavorable tail).

Shapley values and shares (efficiency I6 ≤ 5.6e-17 in every game; in cue-final frames T is an exact null player by
construction):

| population / group | G | φ R | φ emb | φ Bv | φ Bp | φ T | σ R | σ emb | σ Bv | σ Bp | σ T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Y1 cue-final | 0.160230 | −0.000002 | 0.008352 | 0.101223 | 0.050657 | 0 | −0.000012 | 0.052126 | 0.631734 | 0.316152 | 0 |
| Y1 coordinated | 0.337507 | 0.002008 | 0.009068 | 0.184356 | 0.004322 | 0.137752 | 0.005951 | 0.026868 | 0.546230 | 0.012807 | 0.408145 |
| Y2 cue-final | 0.159868 | 0.000616 | 0.009283 | 0.094972 | 0.054997 | 0 | 0.003851 | 0.058068 | 0.594064 | 0.344017 | 0 |
| Y2 coordinated | 0.410088 | 0.002842 | 0.014201 | 0.221125 | 0.014700 | 0.157222 | 0.006929 | 0.034629 | 0.539212 | 0.035845 | 0.383385 |

### Descriptive records (no outcome force)

The fixed ladder (flattened `Δc` `R²` of each composition):

| population / group | Level 0 | R | R+emb | R+emb+Bv+Bp | all | R only | inputs only (reduced layers 1–2) |
|---|---|---|---|---|---|---|---|
| Y1 cue-final | 0.8005 | 0.8004 | 0.8065 | 0.9607 | 0.9607 | 0.8004 | 0.9604 |
| Y1 coordinated | 0.6277 | 0.6315 | 0.6448 | 0.8587 | 0.9652 | 0.6315 | 0.9648 |
| Y2 cue-final | 0.7927 | 0.7934 | 0.8002 | 0.9526 | 0.9526 | 0.7934 | 0.9518 |
| Y2 coordinated | 0.5440 | 0.5486 | 0.5724 | 0.8327 | 0.9541 | 0.5486 | 0.9534 |

- **The full composition** reaches `R²` 0.9607 and 0.9652 in Y1 (cue-final, coordinated) and 0.9526 and 0.9541 in
  Y2. With every block-0 term included but layers 1–2 still the reduced program (inputs only), `R²` stays within
  0.001 of the full composition in every group (0.0003, 0.0004, 0.0008 and 0.0007 lower). This is a number only,
  never a corrected program.
- **`Δx3` reconstruction** (relative error, median / max): at Level 0 about 0.50–0.54 at the cue position and
  0.58–0.66 at the target position; `R` and `emb` change it little. Adding `Bv` and `Bp` brings the cue position to
  about 3e-6 (max 4.2e-5), while the target position of coordinated frames stays at about 0.41 until `T` is added,
  after which it is about 4e-6 (max 1.4e-5). The block-0 terms make the `Δx3` reconstruction nearly exact at the
  changed positions.
- **The historical comparator** (descriptive only; it enters no game, envelope or result): flattened Level-0 `R²` with
  017's wiring against 020's historical wiring — Y1 coordinated 0.6277 against −0.2257, Y2 coordinated 0.5440
  against −0.2839. In cue-final frames the two wirings coincide by construction (`p_t = p_c`; 0.8005 and 0.7927), so
  the equal cue-final values are not an independent agreement.
- **Per template and cue class** (subsets of the conditions' groups): all interpretable, with the same pattern —
  cue-final σ Bv 0.573–0.680, σ Bp 0.258–0.378, |σ R| ≤ 0.009; coordinated σ Bv 0.452–0.574, σ T 0.314–0.475,
  σ R ≤ 0.008. The coordinated possessive-or-pronoun cues have the smallest gap (G 0.160 in both populations, with
  Level 0 already 0.79–0.81); `Bp` is slightly negative for coordinated adjectives (−0.039 in Y1, −0.006 in Y2).
- Block 0's head profile (reference self-weights, per-head value-term norms, attention from `p_t` to `p_c`) and the
  18 validity verdicts are in the report and the extract.

### Interpretation (agreed at the confirmation and report reviews; the design file is not edited)

The spike's localization generalized prospectively, at every claim, in both populations. Stated at its safe strength:
the remaining upstream `Δx3` prediction error of the committed 017 chain is localized primarily to block-0 attention
and the state it constructs. On unseen cues and unseen frames, block-0 attention at the cue position — its value and
pattern terms, `Bv + Bp` — accounts for 0.948 (Y1) and 0.938 (Y2) of the cue-final gap between Level 0 and the
ceiling. The correctly wired layer-1–2 reductions account for −0.00001 and 0.004 of that gap and for 0.006 and 0.007
of the coordinated gap. In coordinated frames block 0's change at the target position (`T`, the cue→target pathway)
contributes positively; its measured share, 0.408 (Y1) and 0.383 (Y2), is descriptive, since the frozen reading of C4
is a positive contribution and never "large" or "substantial".

What this does not show (frozen): that any corrected program would predict well — 022 builds none, and the full
composition is the identity endpoint only — or which of 016/017's reductions carries σ R. The value/pattern split is
definitional (the reference row against the row change, as in 017's `F`/`Π`), and the attribution uses one value
function, flattened `Δc` `R²`. Limits: one checkpoint (pinned Pythia-70M deduped), three templates, four cue classes,
24 new cues, 18 new frames, the 79 exposed nouns, `Δc` at `p_t`. Whether the localized block-0 pathways can be written
as a compact weight-derived program is left to a later experiment. Claim
[C002](../../research/claims/C002-count-cued-noun-number-circuit.md) is unchanged (`LOCALIZED`); this design names no
claim transition.

### Exploratory replication on Experiment 021's spent set (after the report; not a prospective confirmation)

`replicate-021` ran once, after the report, as the design places it. It executed no prompt and no forward pass: the
model was loaded only to read its weights, the recomposition ran under the capture-disabling guard, and the ledger
stayed at 21,960 keys. It used only Experiment 021's stored stage-2 measurements, verified against 021's closed results
(file `8caf2a02…`, state `80b9b654…`) and its committed extract (`678714b0…`): Y1 2,592 pairs (021's 24 cues × the 108
exposed frames; 1,728 cue-final and 864 coordinated) and Y2 432 pairs (× 021's 18 new frames; 288 and 144), with 021's
own noun set so that its stored readout is reproduced exactly. The official report stayed byte-identical, and only the
replication entry changed in the state. The record carries numbers only, with no result names, and cannot alter any
result above.

Checks: I4 (the full composition against 021's stored ceiling) 4.67e-5 against 1e-3; I5 (the Level 0 as 020/021 wired
it against 021's stored Level 0) exactly 0.0; I6 1.1e-16. I1–I3 are unavailable, because 021 stored no `Δx1` or `Δx3`
of its target pairs, and nothing substitutes for them. Descriptively, the 017-wired empty coalition differs from 021's
stored Level 0 by up to 3.98, in coordinated frames, by design.

| population / group | G | σ R | σ emb | σ Bv | σ Bp | σ T | Level 0 | all |
|---|---|---|---|---|---|---|---|---|
| Y1 cue-final | 0.116139 | 0.003113 | 0.063503 | 0.711538 | 0.221847 | 0 | 0.8529 | 0.9690 |
| Y1 coordinated | 0.341948 | 0.006426 | 0.039435 | 0.564699 | 0.010449 | 0.378991 | 0.6244 | 0.9663 |
| Y2 cue-final | 0.116229 | 0.004117 | 0.061931 | 0.703083 | 0.230868 | 0 | 0.8445 | 0.9608 |
| Y2 coordinated | 0.339820 | 0.012429 | 0.043334 | 0.652654 | −0.006820 | 0.298402 | 0.6341 | 0.9739 |

Descriptively, all eight corresponding statistics lie inside the 022 envelopes and meaning guards (Y1 / Y2: s1 0.9334
/ 0.9340, s2 0.0031 / 0.0041, s3 0.0064 / 0.0124, s4 0.379 / 0.298); this comparison was made at review, is not
recorded by the runner and is not a result. The same qualitative localization reproduces: block-0 cue-position
attention carries about 93 % of the cue-final gap, `R` stays small (0.003–0.012), `T`'s measured share in coordinated
frames is 0.30–0.38, and the full composition reaches `R²` 0.961–0.974. What moves with the population is the
allocation inside block 0: 021's cues start from a higher cue-final Level 0 (0.853 and 0.845), so the cue-final gap is
smaller (0.116), and its split leans further toward `Bv` (0.70–0.71 against 0.59–0.63 in 022) and away from `Bp`
(0.22–0.23 against 0.32–0.34), with `Bv + Bp` about the same. Y1/C3 (0.0064 against 0.0069) and Y2/C3 (0.0124 against 0.0142) are the
closest to their bounds.

### Protocol history: the coordinated reference-row wiring

- Experiments 017–019 built the committed chain's layer-1/2 reference rows through the cue position `p_c`.
- Experiment 020's Level-0 program built them through `p_t` in coordinated-adjective frames (cue-final frames are
  unaffected, since `p_t = p_c` there), and Experiment 021 tested that program unchanged. The independent
  implementation review of Experiment 022 found the defect before any 022 scientific phase; the post-closure errata
  record it ([020](../020-readout-decompilation/evidence/erratum-2026-09-23-coordinated-reference-rows.md),
  [021](../021-corrected-readout-confirmation/evidence/erratum-2026-09-23-coordinated-reference-rows.md); `2320345`).
- Experiment 022 (design revision 4) uses the corrected 017 wiring. 020's wiring is kept only as the descriptive
  historical comparator, which on the new coordinated data reaches `R²` −0.226 (Y1) and −0.284 (Y2) against 0.628 and
  0.544.
- The original frozen results, labels, floors, digests and reports of 020 and 021 remain historical records and are
  not rewritten.

### Closure notes (accepted at review; nothing frozen is changed)

- **Authoritative bindings:** design revision 4 (`219cdc5`) and plan revision 3 (`e6d8299`), as bound in the `design`
  and `plan` fields of the calibration record, the lock, the table indexes and the results state.
- **Stale revision wording, non-authoritative, left unchanged in frozen artifacts and scientific code:**
  - D1: the calibration record's `kind` says "(design revision 3)".
  - D4: the lock's `kind` says "(design revision 3, plan revision 2)", and the table construction text in the lock and
    in both table indexes says "the factors of design revision 3". Only the label is stale: the construction it
    describes (017's chain with its input replaced) is the one that ran, and the lock's semantics record the wiring.
  - D2: the `calibration_stop` docstring says "125 or more of C3", and the runner's stop message says "design revision
    3". The rule that ran is `undefined_threshold`, ≥ 250 for every claim (design revision 4, recorded in the record's
    constants); no draw was undefined, and the message was never emitted.
  - D5: the `cdf_percentile` docstring names only C2 as an upper bound; C3 is one too, as the report's own sentence says.
- **D3:** this README said "≥ 125 for C3" and "not run"; corrected at closure.
- **D6:** in cue-final frames the historical comparator equals the empty coalition by construction; the equal
  cue-final values carry no information.
- **D7:** the plan's constants block, copied from revision 2, still names design revision 2 (`0f1009c`); a closure note
  in the plan marks it.
- **D8:** the design says that the tolerances carry margins of at least 8×. Against the calibration's exposed maxima
  they were I1 7.0×, I3 1.8× and I4 7.7× (against the spike's 7.7×, 11.8× and 27.8×). The lock review knowingly kept I3
  at 1e-4; at confirm the margins were I1 6.1×, I3 2.4× and I4 13×, and every gate passed.
- **The official report** is byte-identical to what the frozen renderer wrote and is not regenerated. Its limits:
  - its header shows the results state's creation commit (`0de0fb8`, where `calibrate` ran) and the phase statuses at
    render time ("report not_started"); `confirm` ran at `4f8f31d`, as the state records;
  - the line beginning "None:" means that there is no aggregate label;
  - it has no sentence saying that all four gaps are interpretable; they all are (G ≥ 0.02);
  - it prints I3's value (4.22832e-5) without its tolerance (1e-4), and prints neither I7 nor the confirmation's
    kernel/direct check (its kernel line is the calibration's);
  - it rounds values to 4 decimals (Y1/C2 prints as "-0.0000") and the CDF to 3 (0.926 and 0.520 for 0.9265 and
    0.5205);
  - it does not print the D4 labels.

  The exact values live in the results state and the extract; no rounding changes any result.
- **The exploratory report** is the runner's render of the whole state after the replication, without the calibration
  draws (so its CDF column is empty). Only its final section is the replication; it is not a second official report.

### Evidence

| file | what | sha256 |
|---|---|---|
| [`evidence/final-report-2026-09-24.md`](evidence/final-report-2026-09-24.md) | the official primary report, verbatim | `df06f1318487e5111c74734078a8891da4fed4f3720ac62cf0d0097ffc2fd765` |
| [`evidence/confirmation-record-2026-09-24.json`](evidence/confirmation-record-2026-09-24.json) | the extract of the final results state | file `22896a33eb05fed56a359fc82b3e4caac72fda315bf46af68e285d703364717b`, content `ef393bb8…` |
| [`evidence/y2-table.f64`](evidence/y2-table.f64) | the Y2 composition table as stage 1 wrote it (archival) | `008e8e4e9b91e65b9b42acd9cb26c6eab646e446c02769a77bb7f8f5d4a19860` |
| [`evidence/y2-table.json`](evidence/y2-table.json) | its index | `bde51b96550407ac151120ab5d553bcf3a13bc0c04deb3ef193243c74065b9d2` |
| [`evidence/exploratory-replicate-021-record-2026-09-24.json`](evidence/exploratory-replicate-021-record-2026-09-24.json) | **exploratory**: the replication record | `d594a4122f5ec62b6bdd4ab2c2a56db71d62794025d54c216f33df806373fcc7`, content `541a7e3f…` |
| [`evidence/exploratory-replicate-021-report-2026-09-24.md`](evidence/exploratory-replicate-021-report-2026-09-24.md) | **exploratory**: its rendered report | `5b11e11d391b7c8f9c9ad2125f6a597ed9652dc65b2216d95e68a7a52acc8c8b` |

Every evidence copy is byte-identical to its source in `outputs/experiment-022/`, and the extract binds each by digest.
The frozen committed artifacts are unchanged: `confirmation-v1.json` (`1a9afef0…`, content `af848ae4…`),
`calibration-v1.json` (`db745653…`, content `46985fd5…`), `preregistration-lock.json` (`0588456f…`, content
`afe1b541…`), `preregistration.md` (`9072d064…`), `locked-y1-table.f64` (`cb9fbf74…`) and `locked-y1-table.json`
(`a6f413b2…`). The gitignored outputs — the results state, the calibration table, the draws, the stage-2 measurements
and the phase logs — are bound by digest in the extract.
