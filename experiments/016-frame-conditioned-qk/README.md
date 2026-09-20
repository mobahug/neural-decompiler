# Experiment 016: What the Frame Contributes — The Frame-Conditioned Token-Local Q/K Model

Implements the approved design
[`docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md`](../../docs/superpowers/specs/2026-09-20-experiment-016-frame-conditioned-qk-design.md)
(revision 2) through the plan
[`docs/superpowers/plans/2026-09-20-experiment-016-frame-conditioned-qk-plan.md`](../../docs/superpowers/plans/2026-09-20-experiment-016-frame-conditioned-qk-plan.md).
Prospective, zero-parameter, decomposition-completing: Experiment 015's token-local cue-change Q/K model missed its
frame-conditional floors at layer 2; this experiment adds three literal reference-state channels and nothing else —
(A) the frame's own reference query and key vectors at the cue position as the operands of the self-logit (1024
numbers per layer), (B) the frame's LayerNorm scale before the change and, derived algebraically from the *predicted*
change, after it (two scalars per layer), (C) block 1's MLP evaluated at the frame's own cue residual (the 512-vector
through the 2048 pre-activations) — and tests the resulting Level 0-F on unseen cues and twelve unseen frames with a
per-frame anti-collapse guard, the scale-only alternative committed for rejection, the three ablations as a
descriptive ladder (with each channel alone, the oracle diagonal-proportional comparator and the norm ratios as further
descriptive rungs). The decoded `c_L` at Level 0-F keeps block 2's MLP at the template base — channel C is block 1's
operating point only; Level 1 alone evaluates both MLPs at the frame's own state. The channel switches are one code path: all off recovers Experiment 015's Level 0, all on plus the
discarded renormalization term recovers Level 1 (both checked as identities on every pair). No Level 0-F input is read
from a fresh cue forward pass.

## Inputs

- Exposed pool (calibration record only; the floors are frozen constants): the 183 exposed cue tokens (Experiment
  015's 159 and its 24 confirmed) and the 54 exposed frames (its 48 and its 6), with the 80 nouns; a committed extract
  of Experiment 015's per-pair `c_ΔA`, sixteen self-weight changes and `c_L`, `c_M`, `c_H`
  ([`inherited/experiment-015-pair-extract.json`](inherited/experiment-015-pair-extract.json), 6180 pairs, replicated
  within `1e-6`), carrying also the digests of the six 015-fresh frames' stage-1 reference states; the Experiment 011
  lock's axes and read weight, the Experiment 012 lock's bases and the Experiment 015 lock's reference states (checked
  against the re-captured ones), used verbatim.
- `confirmation-v1.json` — frozen on 2026-09-20 by tokenizer rules before any Experiment 016 model output, content
  sha256 `e5a3a4eb392e459117ebd7dfd132fffb66f02bbd0cd75b479f5ba1afd83e3855`: 24 fresh cue tokens by lexical class,
  **with no expectation attached** — `initial upper respective individual specific` (determiner-like), `twentieth
  quarter twin dual` (ordinal-or-numeral), `unlimited countable total insufficient extensive` (quantity), `oneself
  myself yourselves whoever` (possessive-or-pronoun), `bright orange pink brown grey heavy` (adjective); twelve fresh
  frames, four per template (`The garden yields`, `The factory assembles`, `The kitchen serves`, `The workshop builds`;
  `The essay explores`, `The teacher explains`, `The guide recommends`, `The seminar addresses`; `Liam and Ava stored …
  cold`, `Zoe and Theo shipped … fresh`, `Nina and Arjun mixed … thick`, `Elif and Marco tied … tight`); two prompt lists: the fresh
  tokens in the 54 exposed frames (1296, Y1) and in the 12 fresh frames (288, Y2). Executed only by `confirm`.

## Commands

```bash
uv run python experiments/016-frame-conditioned-qk/run.py validate
uv run python experiments/016-frame-conditioned-qk/run.py freeze-confirmation
uv run python experiments/016-frame-conditioned-qk/run.py explore
uv run python experiments/016-frame-conditioned-qk/run.py lock
uv run python experiments/016-frame-conditioned-qk/run.py confirm
uv run python experiments/016-frame-conditioned-qk/run.py report
```

- `explore` (once, exposed pool): the 54 frames' reference states (locked; checked against the Experiment 015 lock
  and stage-1 digests); the 6180 recorded pairs re-measured with the patched residuals and rows captured; replication
  of the Experiment 015 extract; the identities I1–I3 and the two channel recoveries on every pair (incidents when
  violated); the measured rows and `c_ΔA`; Level 0-F, the scale-only alternative, the ablations, Experiment 015's
  Level 0 and Level 1 from the reference states; the exposed statistics (token means and pairs of `ĉ_ΔA`, pooled entry
  R² per layer and per frame, self-weight R²), the ablation ladder with its predeclared ordering, the renormalization
  and scale remainders, the decoded-`c_L` ladder.
- `lock` (weights, the Experiment 011/012 locks, the 54 locked reference states — no forward pass, no capture or
  intervention call path): the prediction table for the fresh tokens × exposed frames (the sixteen Level 0-F rows, the
  self-weight changes, `ĉ_ΔA` per layer and in total, the scale-only rows and read, the diagonal-proportional rows and
  read, the three ablations' reads, Experiment 015's Level-0 read, `σ̂'/σ` per layer, the decoded `c_L`), the token
  means; the frozen floors; writes `candidate-lock.json` and `candidate-predictions.md`. Installing them as
  `preregistration-lock.json` and `predictions.md` and committing them is the preregistration act.
- `confirm` (once, two stages): validates the artifacts and reproduces every locked prediction (refuses above `1e-9`);
  **stage 1** runs only the twelve fresh frames' reference prompts and cue pairs, checks I1, decides validity, computes
  the frame-conditional prediction table and writes it to the results state with its digest; **stage 2** starts only
  after that digest is re-read from disk, runs every fresh cue in both sets with the patched residuals and rows
  captured, checks the identities and recoveries, and scores Y1 (strict prospective), Y2 (frame-conditional prospective,
  aggregate with the frame-collapse guard) and Y3 (the scale-only alternative over both sets) against the two tables.
- `report` renders `outputs/experiment-016/report.md`.

Boundaries: no Level 0-F input is read from a fresh cue forward pass — the channels come from the reference prompt
and `σ'` from the predicted change (a test poisons the patched capture and shows the table unchanged; the prediction
table is computed with every capture entry point disabled); the identities, recoveries and the locked-state
consistency of re-captured frames are enforced and any failure is an incident recorded with its commit; a token needs
three valid frames in a set to be scored there; Y2 needs eight valid fresh frames; no fresh cue prompt runs before its
frame's stage-1 predictions are digested; which heads move for which cues in the new frames is described after the
scoring.

## Status — 2026-09-20: complete; outcome `FRAME_CHANNELS_PREDICTED_TOKENS | FRAME_CHANNELS_PREDICTED_FRAMES_CONDITIONAL | SCALE_ONLY_REJECTED`

Tier A (`explore`) ran once at `6343f23` on the 6180 recorded pairs (183 tokens, 54 frames): the Experiment 015 extract
replicated (deviation 0.0), the identities held (I1 ≤ 1.7e-5, I2 3.4e-6, I3 1e-6), the two channel recoveries held to
1e-14, Level 0-F reached token-mean Spearman 0.998 / R² 0.994 and pooled entry R² 0.996 (layer 1) / 0.988 (layer 2) with
a per-frame layer-2 minimum of 0.937, the scale-only alternative 0.708 at layer 2. The lock was installed byte-identical to
the Tier A candidates and committed by hand (`dc4e2a7`; lock content sha256
`c21a69fa9347cc3ebcb4d113a719d4a34fb52668e513600dcfbe627c50884c77`, predictions sha256
`85789c0f083f8d042f971447d06da5277bd7992b64477e841d4be31728196986`; the candidate predictions are copied to
[`evidence/candidate-predictions-2026-09-20.md`](evidence/candidate-predictions-2026-09-20.md)), the read-only post-lock
check passed (hashes, byte identity, only the two exempt files changed since `6343f23`, `validate_lock`, frozen floors and
program record, the complete 1296-row table reproduced with zero difference, runtime equal to the explore record, no
overlap with executed prompts), and the single `confirm` ran at `dc4e2a7` on a clean tree straight through both stages
(results state sha256 `495f208f01ce7cb4c3ef8952aae56886b995a10b7e11397a80919d4cb109bc1e`). **Stage 1** ran only the
twelve fresh frames' reference prompts and cue pairs — all twelve valid (plural head change 0.8–2.9 σ_T, cue effect
79/72; p_c 3–6) — computed the 288 frame-conditional predictions and digested them
(`cd1d1de7dc79162bb3cac78e556abe2fdb43d82813c9e5f1399e3564e2a2c368`); **stage 2** started only after re-reading that
digest from disk and ran the 24 fresh cues in the 54 exposed frames (1296 pairs) and the twelve fresh frames (288
pairs). The final report is copied verbatim to [`evidence/final-report-2026-09-20.md`](evidence/final-report-2026-09-20.md).
Ledger after confirm: 1782 prompt keys, 80 noun keys. Experiment 016 is closed; nothing is amended or rerun.

- **Identities and recoveries (checks only):** on all 1584 fresh pairs the program reproduced the captured rows (I1 ≤
  2.2e-5), the exact chain the captured residual before block 2 (I2 2.8e-6), Experiment 013's split held (I3 9e-7), the
  switch-off variant recovered Experiment 015's Level 0 and the all-channels-plus-remainder variant Level 1 to 1e-14, and
  Level 1 accounted for the measured `c_L` to 2e-6. Every re-captured reference state matched its locked copy or stage-1
  digest. The predicted post-change scale `σ̂'` differed from the scale of the captured patched residual by 0.0000
  (layer 1, where the change is exact) and 0.0007 (layer 2) on average — recorded, never fed back.
- **Y1 — strict prospective (24 fresh cues × 54 exposed frames): PASS on all four floors.** Token means `ĉ_ΔA` vs
  `c_ΔA`: Spearman **0.998** (floor 0.90), **R² 0.994** (floor 0.85), MAE 0.0015 on a spread of 0.021; pooled entry R²
  **0.994** at layer 1 and **0.988** at layer 2 (floors 0.85; total-variation ratios 0.066 / 0.081); the sixteen
  self-weight changes R² 0.992; pairs 0.994 / 0.988; per-frame layer-2 minimum over the 54 exposed frames 0.951. All 24
  token means carry the committed sign (17 negative, 7 positive); the largest token-mean miss is 0.003 (`countable`).
- **Y2 — frame-conditional prospective, aggregate with the frame-collapse guard (24 fresh cues × 12 new frames): PASS on
  all five floors.** Spearman **1.000**, **R² 0.996**, MAE 0.0012; pooled entry R² **0.995** / **0.986**; **frame guard
  passed** — every new frame's layer-2 entry R² over its 24 pairs: 0.986, 0.949, 0.993, 0.933 (cardinal), 0.996, 0.999,
  0.998, 0.993 (quantifier), 0.984, 0.994, 0.980, 0.996 (coordinated-adjective); the lowest, `The workshop builds` at
  0.933 and `The factory assembles` at 0.949, are the frames where Experiment 015's Level 0 would have reached 0.507 and
  0.512. On the same twelve frames Experiment 015's Level 0 gives layer-2 entry R² 0.668 overall (0.507–0.814 per frame)
  — the frame sensitivity that failed 015 is present in these frames and resolved by the three channels.
- **Y3 — the scale-only alternative is rejected** on the 1584 pairs of both sets: layer-2 entry R² **0.672** (rejected iff
  below 0.85), **0.316** below Level 0-F's 0.988 (rejected iff ≥ 0.15 below). The frame's scale alone leaves the pattern
  change where Experiment 015 left it (015 Level 0 on the same pairs: 0.677).
- **The channels' weights (ablation from the full model, layer-2 entry R², both sets):** without the reference operands
  0.709 (−0.278), without block 1's operating point 0.888 (−0.100), without the scale 0.946 (−0.042) — the predeclared
  ordering operands > operating point > scale holds on the fresh data as on the exposed pool. Each channel alone over
  Level 0: operands 0.841, operating point 0.724, scale 0.672. The renormalization remainder (Level 1 − Level 0-F) is
  0.012 of layer-2 entry R² on the fresh pairs: on this data too, the frame residual's direction contributes only a small
  remaining renormalization term. `‖Δ̂x‖/‖x − μ‖` 1.02 / 0.86; `σ̂'/σ` 1.15 / 1.09 on average.
- **Comparator rule (descriptive, predeclared):** Level 0-F's pooled entry R² exceeds the Level-0-F diagonal-proportional
  comparator's by **0.270** (layer 1) and **0.214** (layer 2) over both sets, both ≥ 0.10, so the report may say that the
  model decodes the query/key interaction beyond the self-logit effect; the oracle comparator reaches 0.783 at layer 2.
- **Ladder of decoded `c_L` on the fresh pairs (descriptive):** Experiment 012 R² 0.536, Experiment 013 own-base 0.943,
  Experiment 015 Level 0 0.694, Level 0-F 0.785 (block 2's MLP at the template base), Level 1 1.000.
- **The heads, described only now.** Over the 24 fresh cues in the new frames the largest measured self-weight changes
  are `L01.H01` (+0.18 on average: the cue attends more to itself), `L01.H02` (−0.18), `L02.H03` (+0.10), `L01.H05`
  (−0.08), `L01.H07` (+0.07); Level 0-F reproduces every head's token-mean self-weight change with correlation ≥ 0.997
  and the correct sign for 379 of 384 (token, head) means. No label is attached to any head.

What this settles and what it does not. Stated at its safe strength: three structured classes of reference-state
information of the frame's cue position — the frame's own query and key vectors as the operands of the self-logit
(1024 numbers per layer), the LayerNorm scale before the change and, derived algebraically from the predicted change,
after it (two scalars per layer), and block 1's MLP evaluated at the frame's own cue residual (the 512-vector through
2048 pre-activations) — are sufficient for the token-local change (template direction) to predict the cue-induced
attention-pattern change of the sixteen layer-1–2 heads nearly exactly for 24 never-seen cues in the 54 exposed frames
and in twelve previously untested frames, no frame falling below layer-2 entry R² 0.93; the frame's scale alone does
not suffice; and the frame residual's direction in the query/key/value change contributes a remainder of about 0.01.
This is *not* the claim that three numbers explain the frame, nor that attention is predicted from a frame's text: the
channels are reference-run quantities, and the experiment explains the cue-induced change given the frame's reference
state. It does not say why frames differ in these quantities, nor anything about behaviour. With Experiments 015 and
016 together, the cue-position computation through layers 1–2 is written end to end as a weight-only program whose
frame inputs are named: the reference keys, values and logit rows, the reference self-logit operands, the residual's
scale, and block 1's operating point. Limits: one checkpoint, three templates, five lexical classes, sixteen heads,
twelve new frames.
