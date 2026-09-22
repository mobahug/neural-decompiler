# Experiment 020: Readout Decompilation — the Decoded Mechanism to the Model's Own Singular-versus-Plural Logit Contrast

Implements the design
[`docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md`](../../docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md)
(revision 3) through the plan
[`docs/superpowers/plans/2026-09-22-experiment-020-readout-decompilation-plan.md`](../../docs/superpowers/plans/2026-09-22-experiment-020-readout-decompilation-plan.md).

**Status: closed at Tier A on 2026-09-22.** Exploration completed; confirmation intentionally not run because the
corrected block-5 parallel-residual implementation materially changed the exposed calibration on which the
preregistered outcome floors had been based. No lock was written, no confirmation prompt was executed, and no
Y1/Y2/Y3 label is assigned. See [Closure](#closure--2026-09-22).

## Question

Given the change entering the post-`L03` readout path as the already-decoded chain *predicts* it (Experiments
011/012/017, never measured), can the resulting change of the singular-versus-plural logit contrast
`Δc(f, w, n) = c(frame f, cue w, noun n) − c(frame f, reference cue, noun n)` at the prediction position be predicted
prospectively for unseen cues (Y1), unseen frames conditional on their reference state (Y2) and unseen nouns (Y3)?

## Inputs

- Model: pinned `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, CPU float32,
  4 threads, seeds `20260916` / `20260924`.
- Inherited locks, read verbatim and never refitted: Experiment 011 (`769bfeac…`; axes `T`, `R0`, read weight,
  `σ_T`), Experiment 012 (`830abc3b…`; template bases), Experiment 017 (`b4fc9014…`; layer-3 bases).
- Exposed pool (`build_pool_020`): 108 frames (36 per template), 279 cue tokens, 80 nouns of which 79 are scorable
  (`peach` is not single token).
- [`confirmation-v1.json`](confirmation-v1.json), frozen by tokenizer rules on 2026-09-22 before any Experiment 020
  model output, content sha256 `e098e2b44a1702d2b35c20db2ce111358897ea867996024c3e51a79f379c309d`: 24 fresh cues
  (`recent current original typical ordinary identical` / `average excessive exhaustive comprehensive thorough
  sweeping` / `anything something everything nothing who thee` / `purple yellow hidden hard dirty rare`), 18 fresh
  frames (six per template), 24 fresh nouns (`brick … pillar` / `switch … crash` / `colony … anomaly`), and the prompt
  manifest S1-REF 18 / S1-VALIDITY 18 / S2-TARGET 2592 + 432. **Never executed.**

## The program

The Level 0 hypothesis is a downstream readout program with a small frozen-row second-transport term: block 3's heads
by the exact layer program on `x3_ref + Δ̂x3` and its MLP at the frame's own operating point; block 4 with attention
frozen at the reference and its MLP at the frame's operating point; block 5's heads as the frame's reference rows times
the predicted value changes at every key position, its MLP at the frame's operating point **on the block input**
(Pythia is a parallel-residual model); `LN_final` exactly; the weight-only noun read `⟨·, W_U[:, sg(n)] − W_U[:, pl(n)]⟩`.
Code: [`src/neural_decompiler/readout_decompilation.py`](../../src/neural_decompiler/readout_decompilation.py) (blob
`caa73b40…` at `ce3766b`) and [`run.py`](run.py).

## Commands

```bash
uv run python experiments/020-readout-decompilation/run.py validate
uv run python experiments/020-readout-decompilation/run.py freeze-confirmation
uv run python experiments/020-readout-decompilation/run.py explore
uv run python experiments/020-readout-decompilation/run.py lock
uv run python experiments/020-readout-decompilation/run.py confirm
uv run python experiments/020-readout-decompilation/run.py report
uv run python experiments/020-readout-decompilation/run.py diagnose --frames-per-template 1 --cues 0 --label pass1
```

Since the closure, [`closure.json`](closure.json) is present and `lock` and `confirm` refuse before reading any input,
state or model; `explore` refuses because it already ran; `freeze-confirmation` refuses to overwrite. `validate`,
`report` and `diagnose` (which never opens the results state) remain available.

## Record — 2026-09-22

**Explore attempt 1** (run `9d0c1a998732b9c1`, protocol `abf3df9`, 2 h 56 m) captured all 108 exposed reference
states and measured all 30,024 exposed pairs, then stopped as an incident: `level1 failed: 4.593e-03 above the frozen
tolerance 1.000e-03`. No lock, no confirmation prompt and no fresh-noun quantity was touched.

**Diagnostic amendment** (`7ae9373`, `4578f0d`, `4283cd5`, `77593ee`; description only) and **tolerance amendment**
(`ce3766b`, design revision 3, authorized by the user on the diagnostic's evidence). The committed `diagnose`
subcommand placed the failure in the pinned float32 forward's own attention scores: Pythia-70M's later blocks carry a
large common score offset (`‖scores‖∞` median 2.0e4 / 7.9e4 / 1.2e5 at layers 3 / 4 / 5) against a within-row
spread of only ~10–23, so float32 granularity eats 2.5e-4 / 3.9e-4 / 7.8e-4 of the spread; the MLP reconstructions are
exact to ~1e-6 and the error is already present at the reference state with no cue. `LEVEL1_TOLERANCE` was raised from
1e-3 to 7e-3 (the diagnosed per-block attention envelope 5.35e-3 over the smallest observed `‖Δh6^meas‖∞` of 0.79 =
6.77e-3, rounded up). Nothing else changed: no floor, population, fresh set, comparator standing, model, runtime,
seed or outcome rule. The sequential block-5 variant the parallel residual forbids (~1.5e-2) is still rejected by
more than 2×. Diagnostic records (gitignored, sha256): `level1-diagnostic-pass1.json` `02254bf5…`,
`level1-diagnostic-pass2-conditional.json` `f0d07e4c…`, `level1-diagnostic-reference-floor.json` `50d72d68…`.

**Explore attempt 2 — complete** (same run, protocol `ce3766b`, 2026-09-22 17:03–18:12 UTC, 1 h 09 m; `attempts: 2`,
`attempt_commits` [`abf3df9`, `ce3766b`]; A0 contract test passed). Results state sha256
`2e5485dccb39c018eb02ee8d0a3086004399690a4dfd9c114008f057b23bf8f5` (file `da63b8c2…`). The rendered report is
copied verbatim to [`evidence/exploration-report-2026-09-22.md`](evidence/exploration-report-2026-09-22.md); the
exploration record itself — every statistic, per cue, per frame, per template and per noun, the comparators, the
identities with their tolerances, the worst Level-1 pair, the rank-1 objects, the template bases, and a digest for
each of the 108 locked reference states — is extracted to
[`evidence/exploration-record-2026-09-22.json`](evidence/exploration-record-2026-09-22.json) (content sha256
`99f25ee6…`).

### Identities (all pass)

| check | maximum over the exposed pool | frozen tolerance |
|---|---|---|
| readout identity | 5.228e-3 nats | 2e-2 |
| logit reconstruction | 3.260e-3 | 2e-2 |
| Level 1 exact chain `E₁` | 4.5925e-3 | 7e-3 (revision 3; 1e-3 at attempt 1) |
| additive residual identity | 4.107e-6 | 1e-4 |
| reference component sum | 1.242e-6 | 1e-4 |
| inherited Experiment 017 reproduction | 0 (exact) | 1e-6 |

Worst Level-1 pair `coordinated-adjective-018-3` / `half` (token 2716, `p_c` 7, `p_t` 8): `E₁` = 4.072e-3 / 0.8867;
block errors 1.16e-5 / 2.57e-3 / 2.90e-3 — the attention-dominated signature the diagnostic described; the maximum
reproduces attempt 1's 4.593e-3.

### Corrected exposed statistics (30,024 pairs × 79 scorable nouns; calibration record, never evidence for a label)

| statistic | corrected (official, `ce3766b`) | scratch design pass (block-5 MLP bug) |
|---|---|---|
| flattened `R²` | **0.6891** | 0.848 |
| token-mean `R²` (279 cues) | 0.6800 | 0.942 |
| pair-level noun-mean `R²` | 0.5771 | 0.858 |
| pooled MAE | 0.6505 nats | 0.300 |
| frame-mean `R²` | 0.4333 | 0.702 |
| cardinal / quantifier / coordinated-adjective | 0.8535 / 0.8427 / **0.2729** | 0.924 / 0.909 / 0.702 |
| cue-final split / coordinated split | 0.8595 / 0.2729 | 0.917 / 0.702 |

Per frame: cardinal min 0.708 / median 0.835, quantifier 0.722 / 0.824, coordinated min −0.996 / median 0.231 with
8 of 36 at or above 0.40 (80 of 108 frames overall). Worst cues `fifth` −0.832, `fourth` −0.735, `whole` −0.658,
`sixth` −0.656, `entire` −0.587; best `an` 0.962, `four` / `seven` 0.943. Per-cue MAE ≤ 1.5 nats for 279 / 279 (max
1.438). Per noun (79): median `R²` 0.610 (min 0.387), 68 at or above 0.55, slope median 0.901 (all 79 in
[0.75, 1.15]), `|bias|` max 0.618 (all within 0.8).

### Comparators

| object | standing | exposed flattened `R²` |
|---|---|---|
| downstream ceiling (the same program fed the **measured** `Δx3`) | ceiling, never a floor | **0.9760** |
| without the layer-5 head term | nested simplification comparator | 0.6006 (the term is worth +0.089, so it stays) |
| readout MLPs at the template bases | exposed-disfavoured operating-point comparator | 0.5721 |
| `ΔT` of `L03.H04` alone with one fitted noun vector | exposed-disfavoured baseline | −0.2016 |
| rank-1 noun factorization | descriptive comparator | rank-1 share 0.9584; `d_noun`'s exposed fit `R²` 1.000000 by construction (79 equations, 512 unknowns), uninformative on the exposed pool |

Reading: the ceiling at 0.976 against Level 0 at 0.689 says the decoded downstream readout program is close to exact
when given the real `Δx3`; nearly all of the remaining error is the **inherited Experiment 017 predicted `Δx3`**
(unexplained variance ≈ 0.287 inherited against ≈ 0.024 downstream, where the scratch pass had said 0.114 against
0.030), concentrated in the coordinated-adjective template.

### Where the corrected exposed values sit against the frozen floors (descriptive; the floors bind fresh populations only)

| outcome | condition | frozen floor | corrected exposed value |
|---|---|---|---|
| Y1 | token-mean `R²` | ≥ 0.80 | 0.680 — below |
| Y1 | pair-level noun-mean `R²` | ≥ 0.65 | 0.577 — below |
| Y1 | per-cue MAE ≤ 1.5 in ≥ 80 % of cues | 0.80 | 100 % |
| Y1 | pooled MAE | ≤ 1.0 | 0.651 |
| Y2 | frame-mean `R²` | ≥ 0.55 | 0.433 — below |
| Y2 | per-frame `R²` ≥ 0.40 in ≥ 75 % of frames | 0.75 | 74.1 % — below (one frame) |
| Y2 | cue-final split | ≥ 0.40 | 0.860 |
| Y2 | coordinated split | ≥ 0.40 | 0.273 — below |
| Y3 | median per-noun `R²` | ≥ 0.70 | 0.610 — below (exposed nouns) |
| Y3 | per-noun `R²` ≥ 0.55 in ≥ 90 % | 0.90 | 86.1 % — below |
| Y3 | slope in [0.75, 1.15] in ≥ 90 % | 0.90 | 100 % |
| Y3 | `|bias|` ≤ 0.8 in ≥ 90 % | 0.90 | 100 % |

Six of the twelve conditions sit below their thresholds on the exposed pool itself.

### Confirmation isolation (re-verified at closure)

30,132 executed prompt keys (108 reference + 30,024 exposed cue prompts), of which 0 are S1-REF (of 18), 0 S1-VALIDITY
(of 18) and 0 S2-TARGET (of 3024) — 0 of the 3060 confirmation prompts; 80 executed noun keys, none of them one of the
24 fresh nouns; no fresh noun key or token id anywhere in the exploration record; `assert_confirmation_untouched`,
`assert_no_target_prompt_executed` and `assert_fresh_nouns_absent` pass; `lock` and `confirmation` are null in the
results state. The confirmation file is unchanged since its freeze commit `ba2cdc2`.

## Closure — 2026-09-22

**Experiment 020: exploration completed; confirmation intentionally not run because the corrected implementation
materially changed the exposed calibration on which the preregistered outcome floors had been based.**

- The frozen Y1/Y2/Y3 floors were calibrated on a scratch design pass that fed block 5's MLP the post-attention
  residual. The production implementation uses the correct parallel residual (the Level 1 identity caught the
  difference before any model run), and the official explore at `ce3766b` produced the corrected exposed values
  above: flattened `R²` 0.689 where the scratch pass had 0.848, with six of the twelve floor conditions below threshold
  on the exposed pool, the coordinated template carrying most of the shortfall.
- A confirm now would test floors inherited from an implementation known to be wrong, so neither a positive nor a
  negative label could be read cleanly. The single-use confirmation set is therefore not spent under this protocol.
- **The Experiment 020 floors are not changed retrospectively, and no Y1/Y2/Y3 confirmation label is assigned.** The
  corrected exposed statistics stay recorded as they came out.
- The confirmation set `e098e2b4…` has never produced a model output and is reserved, byte-for-byte and read in
  place, for the next protocol version (the Experiment 021 design proposal,
  [`docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md`](../../docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md)).
  The machine-readable closure record is [`closure.json`](closure.json) (content sha256 `f2b1b5e7…`).
- Claim [C002](../../research/claims/C002-count-cued-noun-number-circuit.md) is unchanged (`LOCALIZED`).

What Experiment 020 establishes, descriptively and on exposed data only: the downstream readout from `Δx3` to the
contrast is decoded almost exactly (ceiling 0.976 with the measured input); the weakness of the end-to-end prediction
is the inherited upstream prediction of `Δx3`, above all in coordinated-adjective frames; the float32 forward of the
pinned model has an attention-score precision floor of order 1e-3 relative at the final residual, which bounds any
exact-chain identity on this checkpoint.
