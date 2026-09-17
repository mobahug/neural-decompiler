# Behavior Candidate Screening (protocol v1)

This directory holds the frozen candidate-selection study that precedes any
Experiment 005. The design is
[`docs/superpowers/specs/2026-09-16-behavior-candidate-screening-design.md`](../../docs/superpowers/specs/2026-09-16-behavior-candidate-screening-design.md)
and the implementation plan is
[`docs/superpowers/plans/2026-09-16-behavior-candidate-screening.md`](../../docs/superpowers/plans/2026-09-16-behavior-candidate-screening.md).
The study selects at most one behavior for a separately designed, user-approved,
preregistered Experiment 005. It is not that experiment, and it creates none.

## Files

- `manifest-v1.json` — the sole scientific authority: 720 matched cases for
  `regular-plural` and `ordinal-suffix` (three templates × 40 cases × three
  splits each), tokenizer-validated against both pinned Pythia revisions,
  content sha256 `1c50c2e8bbf95f9aa4331ac57899f77c2a6be6b17d4cb2a5ac4ecc332c5e703c`.
- `run.py` — the four-phase runner (`validate`, `behavioral`, `compactness`,
  `report`); see the root README for the commands and rules.
- `report-2026-09-17.md` — verbatim copy of the generated report for run
  `b1a3efb412191b1b` (generated outputs under `outputs/behavior-candidate-screening/`
  are ignored by Git; the copy is the tracked evidence record).
- `audit-2026-09-17.json` — the exact finalist prior-art audit recorded for
  `regular-plural` (write-once in the results artifact).

## Result — 2026-09-17

Executed once at protocol/code commit `e610c5f804d2a2bbc881e17d80e2fafcf84b107e` on a
clean tree, CPU float32, pinned `EleutherAI/pythia-70m-deduped` revision
`e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`. Pythia-160M was not run because a
candidate passed at 70M. Future-reserve cases were never executed
(480 behavioral + 120 compactness executed case IDs, zero reserve IDs).
`results.json` sha256 `62c48c890d3d9ffab265bab24bf0b8039a05317cce46c22337afe277aa57b3f8`.

| Candidate | Holdout accuracy | Worst template | Behavioral gates | Compactness | Audit | Decision |
|---|---|---|---|---|---|---|
| `regular-plural` | 106/120 (88.3%) | 85.0% | 10/10 pass | pass, k = 2 (`L00.MLP`, `L03.H04`, recovery 0.905) | `PARTIAL_OVERLAP_WITH_EXPLICIT_GAP` | **proposed target** |
| `ordinal-suffix` | 62/120 (51.7%) | 42.5% | failed 7 gates | not run | n/a | eliminated |

Final status: `ONE_PROPOSED_TARGET` — `regular-plural` (count-cued English noun
number selection in pinned Pythia-70M).

What the evidence does and does not show:

- Pythia-70M prefers the correctly inflected noun form over its matched
  alternative on untouched held-out lexical items in 106/120 primary
  conditions (88.3%; Wilson 95% lower bound 81.4%, above the 78% gate), in
  every template family, above every frozen trivial baseline, with the
  fixed-orientation contrast flipping sign in 98/120 matched pairs (81.7%).
  Development accuracy was lower (101/120, 84.2%), so there is no
  development-to-holdout drop. The headline number blends two very different
  conditions: by the frozen manifest, the primary condition is the singular
  prompt for every simple-suffix noun and the plural prompt for every
  spelling-change noun. Stored per-condition results: singular condition
  120/120 in both splits; plural condition 98/120 on holdout (52/60 for
  simple nouns, 46/60 = 76.7% for spelling-change nouns) and 87/120 on
  development. The flip rate is numerically the plural-condition accuracy
  because the singular condition never fails. This is a pairwise preference
  measurement: the full-vocabulary top-1 continuation is one of the two noun
  forms in only one of the 228 single-token held-out conditions.
- The exploratory compactness probe found that exact replacement of only two
  components at the final position — the layer-0 MLP output and attention head
  L03.H04 — recovers 90.5% of the aggregate counterfactual contrast shift on the
  60 validation development cases (cardinal 0.998, coordinated-adjective 0.809,
  quantifier 0.914). The random two-component median recovery is 0.006, floored
  to the reference B_2 = 0.05, so the applied threshold was 0.10. The layer-0 MLP
  alone recovers 61.6% overall and exactly 0.000 on every coordinated-adjective
  case: in Pythia's rotary, parallel-residual architecture the layer-0 MLP output
  at the final position depends only on the final token, which both members of
  a pair share there (the adjective), so that replacement is a structural
  no-op rather than a localization finding; its near-total recovery on the
  cardinal and quantifier templates is the cue token's own embedding pathway,
  because there the cue is the final token. This is a triage signal about where
  the computation may live, not a mechanism claim: it establishes neither
  completeness, minimality, self-repair robustness, nor a human-readable
  account.
- Ordinal-suffix selection is not a reliable Pythia-70M behavior under this
  protocol. The model shows a `th` preference: on single-token development
  numbers it was correct on 60/60 `th`-primary cases but only 29/60
  `st`/`nd`/`rd`-primary cases (mean fixed-orientation contrast -0.007), with
  the contrast flipping in 55/120 pairs against the 80% gate, so the
  non-`th` side was at chance before any held-out number was seen. The
  held-out drop (74.2% to 51.7%) additionally reflects the `th` side falling to
  43/60 where most held-out B-numbers tokenize into several pieces; that
  tokenization shift is a confound of gate 5 only. The position-0, no-BOS
  bare-numeral variant scored identically in both splits (12/20 primary,
  5/20 flips) and explains none of the drop; it remains a validity limitation
  of that variant. The candidate is eliminated in protocol v1 regardless of the
  reason.
- The finalist audit (searches conducted on 2026-09-17 during the compactness
  run, before its outcome was known, and recorded once immediately after it)
  found no component-level causal account of count-cued noun inflection in an
  unmodified Pythia model. Grammatical number is, however, well mapped as a
  representation (steerable inflection subspaces including Pythia) and as
  agreement circuits (Pythia-70M subject–verb sparse feature circuits; GPT-2
  causal mediation and circuit probing; Gemma-2B head and neuron accounts; DAS
  causal variables across Pythia scales). A head-plus-MLP account of number
  agreement is exactly the mechanism family that eliminated `sva-attractor` in
  triage. Any Experiment 005 must therefore claim neither novelty about number
  representations nor about number-agreement circuits in Pythia-70M; its gap is
  behavior-specific (the cue-to-noun-inflection mechanism at the noun position)
  and validation-specific (prospective held-out intervention prediction,
  reconstruction, self-repair, and residual accounting).
- Protocol limitations recorded before the run: the cue-shuffle control
  coincides with the counterfactual prompt for minimal pairs (gate 8 is not
  independent); the plural local heuristic never predicts the scored foil, so
  gate 9 passed (88.3% against 50%) without testing `es`/`ies` allomorphy —
  what was screened is count-cued singular/plural selection; split membership
  is pool-predeclared rather than seeded; and `degree-inflection` was
  eliminated pre-output for tokenizer infeasibility.

## Pre-output amendments

All amendments were committed before any model output was inspected and are
listed in the design and plan: three tokenizer-eligibility pool extensions,
the elimination of `degree-inflection` for tokenizer infeasibility, and two
review-driven corrections (template-stratified compactness partition and exact
integer-count gate arithmetic). There were zero scientific retries and no
invalidated runs.

## Next step

Designing Experiment 005 for `regular-plural` is a new architectural task: it
requires its own design, explicit held-out intervention predictions, user
approval, and a preregistration commit before any confirmatory run. The 120
untouched future-reserve cases per candidate remain unexecuted for that
purpose.
