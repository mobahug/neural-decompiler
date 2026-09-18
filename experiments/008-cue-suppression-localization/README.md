# Experiment 008: Where Is the Apparent Number Signal of `this`-like Cues Suppressed?

Implements the approved design
[`docs/superpowers/specs/2026-09-18-experiment-008-cue-suppression-localization-design.md`](../../docs/superpowers/specs/2026-09-18-experiment-008-cue-suppression-localization-design.md)
(revision 4) through the plan
[`docs/superpowers/plans/2026-09-18-experiment-008-cue-suppression-localization-plan.md`](../../docs/superpowers/plans/2026-09-18-experiment-008-cue-suppression-localization-plan.md).
Discovery-only: no lock, no confirmation, no claim promotion. The deliverable is a per-token table and a mechanically
derived localization summary that becomes the hypothesis of a later prospective experiment.

## Inputs (all exposed by Experiments 005–007)

- 18 frames (6 manifest, 6 Experiment 005 extension, 6 Experiment 006/007 confirmation), 40 cue tokens (the 16
  exposed and the 24 Experiment 007 confirmation tokens), 80 nouns (79 single-token). New combinations of exposed
  tokens, frames, and nouns are executed for the first time here; nothing fresh exists.
- The Experiment 007 `selected` program (`outputs/experiment-007/parameters/selected`, digest checked against the
  committed 007 lock) provides the descriptive anomaly score `a = sign(p) × (m − p)`.
- [`inherited/experiment-007-confirmation-epatch-means.json`](inherited/experiment-007-confirmation-epatch-means.json)
  and Experiment 007's extract of the Experiment 006 responses: the 144 + 192 recorded E-patch mean shifts that
  `explore` must reproduce within `1e-6` on the same noun sets (incident otherwise).

## Commands

```bash
uv run python experiments/008-cue-suppression-localization/run.py validate
uv run python experiments/008-cue-suppression-localization/run.py explore
uv run python experiments/008-cue-suppression-localization/run.py report
```

`explore` (once): the pinned contract test; stage axes from the eighteen frames' clean cue pairs; for every (token,
frame) the E-patch trace with captures (`R0`, `R1`, `T`, `R2`, `M4`, `M5`, `R3`, the head's attention to the cue
position, the contrast, and the exact direct-effect deltas), the axis-only and orthogonal-only component patches, the
cross-context patches inside fixed contexts, the replication checks, the signal fractions and oriented traces, the
frozen classification rules, and the summary. `report` renders `outputs/experiment-008/report.md`.

Boundaries: every phase refuses to run out of order or twice; an incident is recorded with its commit and blocks
another `explore` at that commit; the ledger records every executed prompt and noun.

## Status — 2026-09-18: the single run is complete; mechanical summary `MIXED`, with one consistent transport finding

`explore` ran once on protocol/code commit `d324755` (run `d9b0ddc64970d40c`, results state sha256
`9f40fa9a7936c58bf984e81a9809227621a9cbe5d29d1a7108f7eb1d01f820f9`; A0 passed). The rendered report — the per-token
table, the oriented traces, and the determiner group — is copied verbatim to
[`evidence/exploration-report-2026-09-18.md`](evidence/exploration-report-2026-09-18.md). Ledger: 774 prompt keys (18
frames × the two original cue prompts, the reference prompt, and the 40 token prompts), 80 noun keys; nothing outside the
exposed pool was executed. Both replication checks held exactly (192 and 144 mean shifts, max deviation 0.0);
identity max error 1.4e-6; direct-effect additivity gap 4.0e-4 nats; probe valid in 18/18 frames (the axis-only
patch of the plural cue reproduces its effect: `r_∥` 1.00 cardinal/coordinated, 0.98 quantifier). The eighteen-frame
E axis equals Experiment 005's frozen E axis (cosine 1.000) and has cosine 0.899 with the 007 direction `u₁`.

### Frozen classification (mechanical)

- Suppressed stratum (sign-normalized anomaly score ≤ −1.5): `a` (−3.63), `this` (−2.66), `another` (−2.29),
  `every` (−1.58). Amplified: `these` (+1.52), `those` (+1.77). Everything else ordinary.
- Rule 2: `every` does **not** carry the mechanism-axis signal (`ŝ_R0` 0.20 < 0.3; `ŝ_u₁` 0.26) → `AXIS_ARTIFACT`
  for that token: its mismatch with the 007 program is a property of `u₁`, not something the network suppresses.
  `a`, `this`, `another` carry 0.53 / 0.43 / 0.46 of the plural cue's encoding-axis signal (0.70 / 0.57 / 0.56 on `u₁`).
- Rule 3 (encoding stage): all three carrying tokens are `NONLINEAR_GATING`: the axis component **alone** drives
  0.64 / 0.50 / 0.55 of the plural cue's effect, the orthogonal component alone −0.13 / −0.10 / −0.06, the whole
  −0.09 / −0.01 / +0.06; additivity gaps −0.60 / −0.41 / −0.44.
- Rule 4 (collapse stage on the running residual): `a` → `c` (56% of frames), `this` → `R2` (44%), `another` → `R3`
  (39%); no 75% consensus, so no `LOCALIZED_…` label. The transport sub-decision is `HEAD_DROPPED` for all three.
- Rule 5: no token is context-gated (`x_in` 0.94–0.97 for the suppressed tokens: the plural encoding produces its
  full effect inside the `a`/`this`/`another`/`every` contexts). Rule 6: no late-cancellation flag (cancellation
  index 1.50–1.76 against the plural cue's 1.16; the floor is 2.0).
- **Summary label: `MIXED`** (joint stage–class consensus not reached). H2 is rejected for the suppressed tokens; H0
  holds for `every` only.

### What the traces show (descriptive; the frozen rules did not anticipate this pattern)

The consistent finding is at the transport stage. Oriented signal fractions (18-frame means, relative to the
template's plural cue at the same stage):

| token | R0 (E) | R1 (input to L03.H04) | T (L03.H04 output) | T / R1 | attention to cue | R2 | R3 | c |
|---|---|---|---|---|---|---|---|---|
| `this` | 0.43 | 0.44 | **0.03** | 0.07 | 1.24 | 0.19 | −0.09 | −0.01 |
| `a` | 0.53 | 0.67 | **0.19** | 0.29 | 0.87 | 0.41 | 0.34 | −0.09 |
| `another` | 0.46 | 0.61 | **0.17** | 0.28 | 1.05 | 0.43 | 0.16 | 0.06 |
| `that` | 0.45 | 0.50 | 0.29 | 0.58 | 1.29 | 0.14 | 0.22 | 0.47 |
| `the` | 0.56 | 0.57 | 0.40 | 0.70 | 0.94 | 0.35 | 0.41 | 0.49 |
| `these` | 0.60 | 0.63 | **0.94** | 1.50 | 1.30 | 0.63 | 0.72 | 1.04 |
| `those` | 0.51 | 0.52 | **0.92** | 1.76 | 1.14 | 0.54 | 0.91 | 1.03 |
| numerals (`three`…`nine`) | 0.90–1.00 | 0.86–0.95 | 0.89–0.94 | ≈ 1.0 | 0.81–0.93 | 0.86–0.92 | 0.75–0.84 | 1.01–1.04 |

- The signal survives layers 1–2 at the cue position (`R1 ≥ R0` for every suppressed token), so the transformation
  is not at the cue position (H1-at-the-cue-position is not supported).
- `L03.H04` attends to the cue position as much as or more than in the reference prompt (attention fractions
  0.87–1.24), yet its output carries almost none of the number signal for `this`, `a`, `another` (`T/R1` 0.07–0.29),
  a graded fraction for `that` and `the` (0.58, 0.70), the full signal for numerals and quantities (≈ 1.0), and *more*
  than its input's axis projection for `these` and `those` (1.5–1.8). Since the attention weight is unchanged, the
  selective transfer sits in the head's value/output pathway, or in the layer-3 LayerNorm it reads through, not in
  where the head looks.
- Downstream of the head the mechanical collapse stage varies because other layer-3 components partially restore the
  axis projection (`R2` 0.19–0.43) and the final residual keeps some projection (`R3` −0.09–0.34) that does not reach
  the contrast; direct-effect deltas show moderate opposing terms (`L03.H04` +0.21 versus `L04.MLP` −0.12 and
  `L05.MLP` −0.10 nats for `this`). These downstream effects are secondary to the transport drop.
- The M2 pattern for the three tokens — axis part alone effective, orthogonal part alone inert, whole inert — means the
  orthogonal component of `ΔE_T(w)` switches off the axis component's effect. Together with the trace, the working
  reading is that `L03.H04` transmits number as a function of the cue residual that is not the E-axis projection: it
  reads a direction or a nonlinear combination on which singular-selecting determiners (`this`, `a`, `another`) are
  singular-like and `these`/`those` are strongly plural-like, however much of the E-axis they carry.
- The `NONLINEAR_GATING` class is broad: most ordinary tokens also fall in it, but with the opposite shape
  (`r_∥` and `r_⊥` both large and the whole ≈ 1: sub-additive saturation near the plural cue's level, e.g. `many` 0.83
  / 0.72 / 1.03). Only the suppressed tokens show `r_∥ ≫ r_full ≈ 0` with `r_⊥ ≈ 0`. The frozen rule does not separate
  saturation from gating; this distinction is descriptive.

### Hypothesis handed to Experiment 009

The apparent number signal of `this`, `a`, `another` is dropped at transport: `L03.H04` receives a cue-position
residual that still carries about half of the plural cue's encoding-axis signal, attends to it normally, and outputs
almost none of it on its number axis, while other layer-3 components and the readout only partially compensate. The
head's transmitted number is a selective function of the cue residual (graded across `this` < `a` ≈ `another` < `that`
< `the` < numerals < `these` ≈ `those`). A prospective test must (i) freeze new singular-selecting and plural-selecting
determiners and new frames, (ii) predict the head-output fraction `q_T` and the contrast from a rule fitted on the 40
exposed tokens, and (iii) trace the component patches through `T` (not measured here) to separate a linear
cancellation inside the head's value pathway from a LayerNorm or attention nonlinearity. Nothing here is a confirmed
mechanism; Experiment 008 is closed as a discovery run.
