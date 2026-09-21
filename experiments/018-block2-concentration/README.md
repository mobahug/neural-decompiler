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

## Status

**Tier A (`explore`) ran once at commit `9b629bf` (2026-09-21, run `86ccb888f52600ef`)**; the candidate lock is written
(`outputs/experiment-018/candidate-lock.json`, content sha256
`fdfae9106898f51528fc6c1a3e235bd9be7f37f3ccb00628ccf7522e308fae46`; `candidate-predictions.md` sha256
`aac1b7c92a12a2af57cfa273e6bc8f8abb7a0fe26d2f5af47356bff1890d329d`). No fresh prompt has run. Exposed record (calibration
only): replication of Experiment 017 exact (max deviation `0.0` over 9636 pairs); I8 `1.5e-14`; the ranking's top
neurons `1987, 1102, 1726, 129, 1311, 1310, 1068, 1924, 1671, 815` (`S_1 = {1987}`); random controls overlap `S_256` by
34 / 28 / 37 neurons (recorded, as expected ≈ 32); pooled `κ_{c_L}(S_256)` **0.88**, `κ_Π` **0.67**, `κ_F` 0.88,
`κ_row` 0.22 (diffuse, as predeclared), `κ_{c_L}(S_1)` 0.41; random controls 0.20–0.46, `bottom_256` 0.00; cue-final
split `κ_{c_L}` 0.90, coordinated 0.79 (cardinal 0.83, quantifier 0.92); reference rung `c_L` R² 1.000, rows 0.996;
gaps `c_L` 0.163, `Π` 0.184 (`ΔT` 0.041, below the evaluability gap — descriptive only, as designed); per-frame top-256
overlap with `S_256` 121–163 (median 138); of the 256, 71 (cardinal) / 93 (quantifier) / 15 (coordinated) fire in at
least half of the template's records, neuron 1987 first everywhere. Next: the candidate lock and predictions are
installed as `preregistration-lock.json` and `predictions.md` by hand and committed; the read-only post-lock check;
the reviewer's sign-off; a single `confirm` straight through both stages. Never rerun `explore`.
