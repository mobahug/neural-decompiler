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

## Status — 2026-09-18: implemented; the single run has not happened yet
