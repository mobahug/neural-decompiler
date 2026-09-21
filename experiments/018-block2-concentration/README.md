# Experiment 018: Block 2's Operating-Point Dependence — Is Channel D Carried by a Locked Subset of 256 Neurons?

Implements the approved design
[`docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md`](../../docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-21-experiment-018-block2-concentration-plan.md`](../../docs/superpowers/plans/2026-09-21-experiment-018-block2-concentration-plan.md).
Prospective, zero-parameter, compaction: Experiment 017 wrote the chain from a cue word's weight-only encoding change
`ΔE` to the transport head's output as a composed weight-only program; its least compact object is channel D —
block 2's MLP at the frame's own operating point, 2048 reference pre-activations per changed position through the
exact GELU with the predicted arriving change. This experiment replaces channel D by its masked form `D_S`: the
neurons of a subset `S` at the frame's operating point, every other neuron at the template base (Experiment 012's
base at `p_c`; a block-2 base at `p_t` for the coordinated template, locked at `explore`). The subset is chosen once by
a frozen read-unit rule — `score_j = mean |own_j − tmpl_j| · |r(W_out[j])| / |D_T|` over the 9636 licensed exposed pairs,
from Level 0-F's *predicted* arriving change, no measured quantity — and locked as an explicit index list; the ladder
`S_0, S_1, S_4, S_16, S_64, S_256, S_1024, S_2048`, three seeded random 256-subsets (`R1–R3`, overlaps recorded, never
redrawn) and the 256 lowest-ranked (`B256`) are the controls. `S_2048` is Experiment 017's Level 0 exactly (I8, `1e-9`);
`S_0` at `p_c` is its `−D` rung. The statistic is the closure fraction `κ_X(S) = (R²_X(S) − R²_X(S_0)) / (R²_X(S_2048) −
R²_X(S_0))`, unclipped, undefined below a `0.05` gap. Every rung receives `ΔE`, the weights, the locked axes, read,
bases, subsets and the frame's reference run only; the poisoned-capture test covers every column of every rung.

## Inputs

- Exposed pool (calibration record only; the floors are frozen constants): the 231 exposed cue tokens (Experiment
  017's 207 and its 24 confirmed) and the 78 exposed frames (its 66 and its 12), with the 80 nouns; a committed extract
  of Experiment 017's per-pair measured `F`, `Π`, `ΔT`, `c_L`, head-row change and Level 0 predictions
  ([`inherited/experiment-017-pair-extract.json`](inherited/experiment-017-pair-extract.json), 9636 pairs, replicated
  within `1e-6`), carrying also the digests of the twelve 017-fresh frames' stage-1 reference states; the Experiment
  011 lock's axes and read weight, the Experiment 012 lock's bases, the Experiment 017 lock's layer-3 bases and
  reference states (checked against the re-captured ones), used verbatim.
- `confirmation-v1.json` — frozen by tokenizer rules before any Experiment 018 model output: 24 fresh cue tokens by
  lexical class (the next eligible entries of Experiment 017's lists), **with no expectation attached**; twelve fresh
  frames, four per template; two prompt lists: the fresh tokens in the 78 exposed frames (1872, Y1) and in the 12
  fresh frames (288, Y2). Executed only by `confirm`.

## Commands

```bash
uv run python experiments/018-block2-concentration/run.py validate
uv run python experiments/018-block2-concentration/run.py freeze-confirmation
uv run python experiments/018-block2-concentration/run.py explore
uv run python experiments/018-block2-concentration/run.py lock
uv run python experiments/018-block2-concentration/run.py confirm
uv run python experiments/018-block2-concentration/run.py report
```

- `explore` (once, exposed pool): the 78 frames' reference states (locked; the 66 checked against the Experiment 017
  lock and the 12 against its stage-1 digests); the coordinated template's block-2 base at `p_t` (locked); the ranking
  from the locked states, the weights and the exposed cues' `ΔE` alone — computed before any pair is measured — and
  the subsets, controls, every exposed frame's own top-256 and its overlap with `S_256` (locked); the 9636 recorded
  pairs re-measured with the layer-3 residuals and the head's row captured; replication of the Experiment 017 extract;
  the identities I1–I8 on every pair (incidents when violated); every rung's `ĉ_L`, `F̂`, `Π̂`, `ΔT̂` and row; the exposed
  statistics (pooled `R²` and `κ` per rung and object, token means, per frame, per template, the cue-final/coordinated
  split, the predeclared orderings).
- `lock` (weights, the locked axes, read, bases, subsets and reference states — no forward pass): the prediction table
  for every fresh token × exposed frame and every rung, plus the frame's own subset (the oracle, descriptive); the
  frozen floors; `candidate-lock.json` and `candidate-predictions.md`. Installing them as `preregistration-lock.json`
  and `predictions.md` and committing them is the preregistration act.
- `confirm` (once): validates the artifacts (digests, floors, the subsets as the frozen rule applied to the locked
  scores), recomputes the ranking from the locked states and every locked prediction (refusing before any fresh
  prompt on a mismatch), then stage 1 (the twelve fresh frames' reference runs, validity, each frame's own ranking and
  overlap, the frame-conditional prediction table, digested) and, only after re-reading that digest from disk, stage 2
  (every fresh cue in both sets, scored against the two tables: the preconditions incl. the Y2 split gaps, Y1, Y2 with
  the split and no-harm guards, Y3 pooled with per-set descriptives, Y4 per fresh cue-final frame from the reference
  rung).
- `report`: renders `outputs/experiment-018/report.md`.

## Status — 2026-09-21: complete; outcome `CHANNEL_D_CONCENTRATED_TOKENS | CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL | SINGLE_NEURON_REJECTED | PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES`

Tier A (`explore`) ran once at `9b629bf` (run `86ccb888f52600ef`) on the 9636 recorded pairs (231 tokens, 78 frames):
the Experiment 017 extract replicated (deviation 0.0), every exposed frame's state matched the Experiment 017 lock or
its stage-1 digest, the identities held (I4 8.5e-6, I5 1.2e-5, I6 3.3e-5, I7 8.6e-7, I8 1.5e-14), the coordinated
template's block-2 base at `p_t` was locked over 26 frames, and the ranking was computed from the locked states, the
weights and the exposed cues' `ΔE` alone before any pair was measured: top neurons `1987, 1102, 1726, 129, 1311, 1310,
1068, 1924, 1671, 815`, `S_1 = {1987}`, the three random controls overlapping `S_256` by 34 / 28 / 37 neurons (recorded,
as expected for independent 256-subsets of 2048; never redrawn). Exposed record: `κ_{c_L}(S_256)` 0.88, `κ_Π` 0.67,
`κ_row` 0.22, `κ_{c_L}(S_1)` 0.41, random controls 0.20–0.46, `bottom_256` 0.00; cue-final split 0.90, coordinated 0.79.
The lock was installed byte-identical to the Tier A candidates and committed by hand (`6fa4c48`; lock content sha256
`fdfae9106898f51528fc6c1a3e235bd9be7f37f3ccb00628ccf7522e308fae46`, predictions sha256
`aac1b7c92a12a2af57cfa273e6bc8f8abb7a0fe26d2f5af47356bff1890d329d`; the candidate predictions are copied to
[`evidence/candidate-predictions-2026-09-21.md`](evidence/candidate-predictions-2026-09-21.md)), the read-only
post-lock check passed (byte identity, only the two exempt files changed since `9b629bf`, `validate_lock` with the
subsets reproduced from the locked scores by the frozen rule, frozen floors, state digests and confirmation set, the
ranking recomputed from the locked states with zero difference, the complete 1872-row table reproduced with zero
difference, runtime equal to the explore record, no fresh prompt executed), the reviewer signed off, and the single
`confirm` ran at `6fa4c48` on a clean tree straight through both stages (results state sha256
`2c9195d5f7d295087f1a61b4c90a058a0bf86a30c961c87f97cb44f64a498275`). **Stage 1** ran only the twelve fresh frames'
reference prompts and cue pairs — all twelve valid (plural head change 0.97–2.84 σ_T, cue effect 77–79/72; p_c 3–7,
p_t = p_c or p_c + 1) — ranked each frame on the exposed cues' predicted changes at its own reference state (its top
256 shares 118–148 neurons with `S_256`), computed the 288 frame-conditional predictions of every rung and digested
them with the states and the frame subsets (`a8d55186e3c3aa99f61ad993be1e964a031b74fbee39bb271d344ef98755ff50`);
**stage 2** started only after re-reading that digest from disk and ran the 24 fresh cues in the 78 exposed frames
(1872 pairs) and the twelve fresh frames (288 pairs), every recomputed prediction equal to its table row. The final
report is copied verbatim to [`evidence/final-report-2026-09-21.md`](evidence/final-report-2026-09-21.md). Ledger after
confirm: 2430 prompt keys (all 2184 confirmation keys, each once), 80 noun keys. Experiment 018 is closed; nothing is
amended or rerun.

- **Identities (checks only):** on all 2160 fresh pairs the exact chain reproduced the captured layer-3 residuals (I4
  5.2e-6 relative), the head's captured patched row (I5 1.2e-5), the measured head change (I6 2.5e-5) and the split
  `ΔT = F + Π` (I7 6.9e-7); the reference rung `S_2048` equalled Experiment 017's Level 0 (I8 8.8e-15) and the switch
  model's Level 1 recovered the independent chain to 5e-12; every re-captured reference state matched its locked copy
  or stage-1 digest.
- **Y1 — strict prospective (24 fresh cues × 78 exposed frames, 1872 pairs): PASS.** Precondition met (reference rung
  `c_L` R² 0.999, rows 0.995, `ΔT` 0.999; gaps `c_L` 0.202, `Π` 0.383). `κ_{c_L}(S_256)` **0.863** (floor 0.70) and
  `κ_Π(S_256)` **0.724** (floor 0.50). Descriptively `κ_F` 0.86, `κ_ΔT` 0.83 (gap 0.051), `κ_row` 0.22; token means
  `κ_{c_L}` 0.89, `κ_Π` 0.78; per template `κ_{c_L}` cardinal 0.84, quantifier 0.90, coordinated 0.78.
- **Y2 — frame-conditional prospective (24 fresh cues × 12 new frames, 288 pairs): FAIL, by the no-harm guard alone.**
  Precondition met (reference rung `c_L` 0.999, rows 0.996, `ΔT` 0.999; gaps `c_L` 0.249, `Π` 0.752; split gaps
  cue-final 0.386, coordinated 0.351; 8 valid cue-final and 4 valid coordinated frames). Pooled `κ_{c_L}(S_256)`
  **0.843** and `κ_Π` **0.798** pass their floors; the split guard passes on both families (cue-final **0.84** ≥ 0.60,
  coordinated **0.86** ≥ 0.40); **the no-harm guard fails in one frame, `quantifier-018-4` (`The atlas maps {cue}`):
  `c_L` R² 0.847 at the template base against 0.745 with `S_256`, a difference of −0.102 below the −0.05 margin.** In
  that frame channel D itself is intact — the reference rung reaches 0.999 and the frame's own top-256 reaches 0.924 —
  so it is the globally locked subset that misses there, not the channel. The other eleven frames improve on the
  template base by +0.04 to +4.02 (per-frame `κ_{c_L}(S_256)` 0.58–0.96; that frame −0.67). The preregistered label
  follows: `CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL`, naming the guard and the frame. Read as the design fixed
  it: a failure of full frame-conditional robustness of the locked subset, not an absence of channel-D concentration
  on new frames.
- **Y3 — the single-neuron account is REJECTED (pooled 2160 pairs, pair-weighted).** `κ_{c_L}(S_1 = {1987})` **0.219**
  (rejected iff below 0.50) against `S_256`'s 0.860, a gap of **0.641** (rejected iff at least 0.25); pooled `c_L` gap
  0.208; `κ_Π(S_1)` 0.21 (descriptive). Per set: Y1 0.24 / gap 0.63, Y2 0.12 / gap 0.73 — both dimensions agree, though
  the label is defined on the pooled distribution only.
- **Y4 — secondary, from the reference rung only: NOT ESTABLISHED.** All eight fresh cue-final frames were evaluable
  (reference `ΔT` R² 0.987–1.000). The frozen pattern's per-frame `ΔT` R²: cardinal `The harbor shelters` **0.951**,
  `The studio records` **0.927**, `The stable houses` **0.913**, `The depot receives` 0.771; quantifier 0.871, 0.882,
  0.813, 0.858. Three cardinal frames sit at or above the 0.90 ceiling, so the label is
  `PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES` naming them, and the reading fixed before the run applies verbatim: *the
  new cue sample did not establish the preregistered within-frame variance-share condition; the pattern mechanism
  (Experiment 017) is not in question, and Experiment 017's Y3 is not reopened.* Coordinated frames (descriptive):
  0.844–0.965.
- **Concentration across the frozen rungs (`κ_{c_L}`; Y1 / Y2):** `S_1` 0.24 / 0.12; `S_4` 0.60 / 0.50; `S_16` 0.66 /
  0.59; `S_64` 0.72 / 0.68; `S_256` **0.86 / 0.84**; `S_1024` 0.98 / 0.99; `S_2048` 1 / 1; random controls 0.23–0.32 /
  0.01–0.17; `bottom_256` 0.00 / 0.00; the frame-specific oracle 0.93 / 0.94. On `Π`: `S_256` 0.72 / 0.80, `S_1024` 0.92 /
  0.95. On the head's row entries: `S_256` 0.22 / 0.26, `S_1024` 0.59 / 0.68 — the predeclared orderings hold on both
  sets (row diffuse below 0.50; `S_256` above the best random control by 0.54 and 0.67, above `bottom_256`; `κ_{c_L} ≥
  κ_Π`). `ΔT`'s `κ` was undefined on the exposed pool (gap 0.041) and evaluable on both fresh sets (gaps 0.051, 0.084);
  no `κ` was clipped.
- **Descriptive, after scoring.** Of the 256, 93 (cardinal) / 102 (quantifier) / 23 (coordinated) fire in at least half
  of the Y1 records and 119 / 105 / 38 of the Y2 records; neuron 1987 is first on Y1 in every template and on Y2 in the
  cardinal and coordinated frames, while in the four new quantifier frames neuron **1102** is first (mean read-unit
  effect 0.0246) and 1987 fourth — consistent with the one quantifier frame where the locked set underperforms its own
  top 256. Experiment 016's decoded-`c_L` convention (block 2 at the template base) reads 0.798 / 0.750 on the two sets,
  which is the `S_0` rung. Neither observation carries a label.

What this settles and what it does not. Stated at its safe strength: channel D — the frame's operating-point
dependence of block 2's MLP as it reaches the transport read — is prospectively concentrated in a distributed subset of
block-2 neurons across unseen cues: a fixed set of 256 neurons chosen once on the exposed pool carries 0.86 of channel
D's contribution to the decoded read and 0.72 of its contribution to the pattern term for 24 never-seen cues in the 78
exposed frames, far above three random subsets of the same size and above the 256 lowest-ranked, with the rung curve
(`S_4` 0.60, `S_64` 0.72, `S_256` 0.86, `S_1024` 0.98) describing a distributed-but-concentrated population rather than
a single feature or a uniform block. The single-neuron account is rejected: neuron 1987 ranks first everywhere but
carries 0.22 of the gap prospectively. The exact concentration is not fully invariant across new frames: on twelve
previously untested frames the pooled and per-family closure hold (0.84 / 0.80; cue-final 0.84, coordinated 0.86) and
eleven frames improve on the template base, but in one new quantifier frame the globally locked subset falls below the
template base (0.745 against 0.847) where the full channel reaches 0.999 and the frame's own top 256 reaches 0.924,
so the frame-conditional label is negative as preregistered. The pattern-term question was not established on this
sample within the cardinal frames. This is not the claim that channel D lives in 256 neurons (`S_256` is a recovery
subset chosen by one read-unit rule, not a uniquely defined circuit), nor that the head's row dependence is compact
(it is diffuse, as predeclared), nor anything about behaviour; which members of the population carry the computation
appears to depend on the frame, which is recorded here descriptively and is the natural question for Experiment 019.
Limits: one checkpoint, three templates, five lexical classes, one head, one block, twelve new frames.
