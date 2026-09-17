# Experiment 005: Count-Cued Noun Number Selection — Prospective Mechanism Design

**Date:** 2026-09-17

**Status:** Draft for review. Not approved. No Experiment 005 directory,
claim, preregistration lock, or model run exists. The 120 `regular-plural`
future-reserve cases remain unexecuted.

**Scope:** `regular-plural` (count-cued English noun number selection) in the
pinned `EleutherAI/pythia-70m-deduped` checkpoint, using only the frozen
screening manifest `screening/behavior-candidates/manifest-v1.json`.

## Purpose

The behavior candidate screen (protocol v1, executed once on 2026-09-17)
proposed exactly one target: pinned Pythia-70M prefers the correctly numbered
noun form after a numeral or quantifier cue on untouched held-out nouns
(106/120, every gate passed), and exact replacement of two components at the
final prompt position recovers 90.5% of the counterfactual contrast shift on
development data. That result is a triage lead, not an explanation. The screen
established neither what information those components carry, how it reaches
the noun position, how it becomes a noun-form preference, whether other routes
take over when they are removed, nor how much of the behavior remains
unexplained.

Experiment 005 is the project's first attempt at the research-program question
behind Neural Decompiler:

> Can the computation by which an unmodified pretrained transformer converts an
> external count/quantifier cue into the selection of a singular versus plural
> noun form be recovered as an explicit, compact, human-readable causal
> mechanism that is specified before, and then tested on, untouched examples
> and interventions it was not built from?

The experiment therefore has two halves that the methodology keeps separate:

1. **Discovery (exploratory):** trace the pathway on already-exposed data,
   choose between competing mechanisms using predeclared discriminating
   measurements, and write the mechanism down as an explicit statement with
   named components, variables, and transformations.
2. **Confirmation (prospective):** freeze that statement together with numeric
   predictions about unseen behavior and unseen interventions in a committed
   preregistration lock, then execute the untouched future-reserve cases once.

Success is a mechanism that predicts what it did not see. Failure is equally
informative: it identifies which stage of the proposed decompilation is wrong.

## What is already known and what has been exposed

The following facts are inherited from the screening report
(`screening/behavior-candidates/report-2026-09-17.md`, run `b1a3efb412191b1b`,
code commit `e610c5f`) and constrain this design:

- **Behavior.** Alternatives are always `A` = singular form (for example
  ` cat`) and `B` = plural form (` cats`). The fixed-orientation contrast is
  `c(x) = log P(A | x) - log P(B | x)`. Holdout: 106/120 primary conditions
  correct; contrast sign flips in 98/120 matched pairs; mean `d_full` 5.055
  nats. The singular condition never failed (120/120 in both splits); every
  failure is in the plural condition, concentrated on spelling-change nouns
  (46/60 on holdout). What was screened is count-cued singular/plural
  *selection*; `-es`/`-ies` allomorphy was never the scored foil.
- **Compactness lead.** On the 60 development validation cases, replacing
  `L00.MLP` and `L03.H04` at the final position recovers 0.905 of the aggregate
  contrast shift (cardinal 0.998, coordinated-adjective 0.809, quantifier
  0.914). Discovery ranking: `L00.MLP` 3.684, `L03.H04` 2.371, `L04.MLP` 2.114,
  `L05.MLP` 1.904, `L02.MLP` 0.614, `L05.H07` 0.255, `L00.H05` 0.137, then
  values below 0.10. Singleton effects sum to far more than the 4.639-nat
  denominator, so they overlap; they are discovery scores, not additive
  causal shares.
- **Known structural artifact.** Pythia-70M has parallel residual blocks and
  rotary attention, so the layer-0 MLP output at a position is a function of
  that position's token only. In the cardinal and quantifier templates the cue
  *is* the final token, so replacing `L00.MLP` at the final position replaces
  the cue token's own embedding pathway (recovery 1.022 and 0.890 alone). In
  the coordinated-adjective template the final token is the shared adjective,
  so the same replacement is a structural no-op (exactly 0.000). `L00.MLP` is
  therefore a *cue-encoding* candidate at the cue position, not a
  final-position mechanism, and any transport from cue to noun position was
  only ever tested implicitly through `L03.H04`.
- **Prior-art boundary.** The finalist audit recorded
  `PARTIAL_OVERLAP_WITH_EXPLICIT_GAP`. Grammatical number is already mapped as
  a representation (steerable inflection subspaces across models including
  Pythia) and as agreement circuits (Pythia-70M subject–verb sparse feature
  circuits; GPT-2 mediation and circuit probing; Gemma-2B head/neuron
  accounts; DAS variables across Pythia scales). Experiment 005 claims no
  novelty about number representations or number-agreement circuits. Its
  contribution is the specific cue-to-noun-inflection computation, validated
  prospectively with reconstruction, self-repair, and residual accounting.

### Exposure ledger

| Split | Cases | Behavioral output inspected | Intervention output inspected | Role in Experiment 005 |
|---|---|---|---|---|
| `selection-development` | 120 | yes | yes (component ranking and top-`k` validation) | Tier A discovery data; treated as fully exposed |
| `selection-holdout` | 120 (114 single-token; 6 `peach` cases are two-token) | yes (contrasts known) | no | Tier B calibration data; one or two passes |
| `future-reserve` | 120 (all single-token) | no | no | Tier C confirmation; executed exactly once after the lock commit |

No prompt outside the manifest is executed in Experiment 005. New template
families, longer cue-to-noun distances, other cue words, Pythia-160M, and other
model families are excluded and listed under future work.

## Fixed coordinates

- Model: `EleutherAI/pythia-70m-deduped`, revision
  `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; CPU float32 is the scientific
  runtime; TransformerBridge compatibility mode disabled; `use_attn_result`
  enabled explicitly; every forward under `torch.inference_mode()`.
- Manifest: `screening/behavior-candidates/manifest-v1.json`, content sha256
  `1c50c2e8bbf95f9aa4331ac57899f77c2a6be6b17d4cb2a5ac4ecc332c5e703c`. Only
  `regular-plural` cases are read. Cases are batched by identical prompt
  length, exactly as in the screen; no padding is ever introduced.
- Runtime seed `20260916` (as in the screen, so baseline contrasts replicate);
  control-sampling seed `20260918` for every random component set drawn in
  this experiment.
- Positions, zero-indexed from the first prompt token:

| Template | Example prompt | Tokens | Cue position `p_c` | Target position `p_t` |
|---|---|---|---|---|
| `cardinal` | `The display contains one` / `two` | 4 | 3 | 3 |
| `quantifier` | `The catalog lists each` / `several` | 4 | 3 | 3 |
| `coordinated-adjective` | `Mira and Noah packed one bright` / `two bright` | 7 | 5 | 6 |

  The two frames of every template have identical token counts, and every
  noun appears once in each frame of each template in each split. Positions
  before `p_c` are token-identical within a matched pair; the runner asserts
  that every captured activation there is bitwise identical between pair
  members, which makes any intervention there a no-op by construction, and it
  never spends forward passes on those positions.
- Metric definitions are reused verbatim from the screening design:
  `c(x)`, `m(x) = y(x) c(x)`, `d_full = c(x_A) - c(x_B)`, the bidirectional
  aligned patched shift `d_patch(S)`, and aggregate recovery
  `R_D(S) = mean(d_patch) / mean(d_full)` with the same denominator floors
  (0.25 nats overall, 0.10 nats per template and per rule-class stratum).
  Raw recoveries are reported without clipping. Patched and unpatched
  contrasts of one stratum always come from identical batch compositions so
  that an exact replacement yields an exactly zero shift.

## Component universe and intervention vocabulary

The universe is the screen's 54 components — every attention-head residual
contribution (`blocks.L.attn.hook_result`) and every whole-MLP output
(`blocks.L.mlp.hook_out`) — now indexed by **position** as well as identity.
Two additional sites are used only where stated: the residual stream before a
block (`blocks.L.hook_in`) at a chosen position, and the token embedding
(`embed.hook_out`). Attention patterns are captured, never intervened. The
direct path from the token embedding to the logits is accounted analytically
(see residual accounting), not intervened.

Replacement sources, each with a written justification:

- **Counterfactual (reference):** the matched pair's other member, the same
  case and frame. This is the sufficiency-oriented source used throughout the
  screen.
- **Pair mean (mean):** `0.5 * (act(x_A) + act(x_B))` for the same case,
  frame, and position. It removes exactly the cue-dependent part of an
  activation while preserving everything the pair shares (noun, frame,
  position). It is the necessity-oriented source; zero ablation is not used
  because it is off-distribution for this architecture.
- **Cross-frame resample (resampled):** the same noun and split in the other
  frame of the same template, with either the same cue (predicted to move the
  contrast little) or the opposite cue (predicted to move it like the
  counterfactual). This distinguishes number information from frame or
  lexical-context information.
- **Cross-cue resample (resampled):** the same noun in the other cue-final
  template (`cardinal` ↔ `quantifier`), so the cue *token* differs (`two` vs
  `several`, `one` vs `each`) while the cue *number* is the same or opposite.
  This tests whether the encoded variable is number rather than token
  identity.

All interventions are exact-shape `REPLACE` operations through the existing
`InterventionPlan` runner; multiple components at one hook are provably
disjoint slices. No new hook kinds are introduced. The "freeze" operation used
in path tests is a `REPLACE` with the component's own clean activation from the
unpatched run of the same prompt.

## Stage vocabulary and competing mechanisms

A mechanism statement names components in four roles:

- **E (encoding):** components at `p_c` whose output depends on the cue token
  and carries the number variable. Candidate: `L00.MLP` at `p_c`.
- **T (transport):** attention heads at query position `p_t` that move the
  number variable from `p_c` to `p_t`. Relevant only when `p_c ≠ p_t`
  (coordinated-adjective). Candidate: `L03.H04`.
- **R (readout):** components at `p_t` that convert the number variable into
  the singular-versus-plural logit contrast. Candidates: `L04.MLP`, `L05.MLP`.
- **D (direct path):** the cue token embedding's direct contribution to the
  logits when `p_c = p_t`, and the direct logit contributions of E and T
  outputs. D is measured, not intervened.

The number variable is operationalized as a direction. For a residual site
`r_L(x, p)` (input of block `L` at position `p`), the template's number axis is

```text
d_num(L, p) = mean over development pairs of ( r_L(x_B, p) - r_L(x_A, p) )
```

and a component's number contribution at `p` is the mean projection of its
own `x_B − x_A` output difference onto the unit vector of `d_num` at the
readout input layer `L_R` (the layer of the earliest R component). Because the
residual stream is an exact sum of component outputs plus the embedding,
these contributions sum to the whole axis magnitude at `L_R`; they are an exact
decomposition of *what arrives*, not a causal claim.

Three competing mechanisms are stated now, before discovery data are
inspected, together with the measurements that decide between them:

| ID | Mechanism | Predicts on development data |
|---|---|---|
| **H1 — dedicated transport** | E writes number at `p_c`; one or two heads (T) copy it to `p_t`; R reads it. | In coordinated-adjective: the top head at `p_t` alone recovers ≥ 0.50; freezing it blocks ≥ 0.50 of the effect of a counterfactual residual patch at `p_c`; patching E∪T reproduces ≥ 0.50 of the counterfactual change in R outputs' number projections (mediation). |
| **H2 — distributed transport / local amplification** | Number reaches `p_t` through many weak heads (or, when `p_c = p_t`, needs no transport); no head is individually decisive; R reads a pooled signal. | Top head < 0.50 but cumulative top-3 heads ≥ 0.70; freezing any single head blocks < 0.50; mediation still ≥ 0.50. |
| **H3 — late computation** | R components compute number from cue-dependent inputs not captured by the E/T number axis; E/T are correlates or partial routes. | Patching E∪T reproduces < 0.50 of R outputs' counterfactual number change while patching R outputs alone recovers ≥ 0.70. |
| **H0 — no compact account** | No set of ≤ 6 components at `p_t` plus ≤ 2 at `p_c` meets the Tier A floors. | Tier A floors fail after the iteration budget. |

The hypothesis selection rule is a total, deterministic decision tree over
five development-data quantities from the coordinated-adjective template:
`q1` = recovery of the top-ranked head alone at `p_t`; `q2` = blocked fraction
of that head (A6); `q3` = mediation fraction for E∪T (A6); `q4` = recovery of
the R outputs alone; `q5` = cumulative recovery of the top three heads.

```text
if q3 < 0.50 and q4 ≥ 0.70:            H3
elif q1 ≥ 0.50 and q2 ≥ 0.50:          H1   (flag HYPOTHESIS_AMBIGUOUS if q3 < 0.50)
elif q5 ≥ 0.70:                        H2   (flag HYPOTHESIS_AMBIGUOUS if q3 < 0.50)
else:                                  H2 as the working row, flag HYPOTHESIS_AMBIGUOUS
```

The tree is applied once to the recorded numbers and the mechanism statement
is written for the resulting row. A flagged lock must carry a discriminating
reserve prediction for every row the tree did not exclude. The hypotheses are
not exclusive across templates: the cue-final templates need no T, and the
statement must cover both regimes.

Encoding has one predeclared branch. Let `L_T` be the layer of the earliest
declared T head. The number axis at the input of that layer at `p_c`,
`d_num(L_T, p_c)`, decomposes exactly into the cue token
embedding difference plus the outputs of components at `p_c` below `L_T`. If
`L00.MLP` (with any other declared E component) carries at least half of that
axis, E is those components and prediction family P4 applies. If the embedding
difference alone carries more than half, the mechanism declares E as the
direct embedding path and P4′ applies instead: pair-mean ablation of every
non-embedding component at `p_c` must retain at least half of the contrast
shift. The rule is applied once, on development data, and recorded in the
lock.

## Three-tier data policy

### Tier A — discovery on `selection-development` (exploratory)

All 120 development cases are discovery data. The compactness partition from
the screen is no longer a validation boundary because both halves were
inspected during the screen. Every Tier A number is exploratory and none is
confirmatory evidence for the mechanism.

Ordered protocol; each step writes its measurements to the results artifact
before the next step is interpreted:

- **A0 — runtime contract.** Run the pinned Pythia-70M Bridge contract test on
  the scientific runtime and record its pass in provenance.
- **A1 — baseline replication.** Recompute `c(x_A)`, `c(x_B)`, `d_full` for
  all 120 development cases and 114 single-token holdout cases. Development
  aggregates must match the screening report (mean `d_full` 5.231 development;
  validation denominators 4.639 overall, cardinal 4.606,
  coordinated-adjective 4.969, quantifier 4.343) within 1e-3 nats, and match
  the screening `results.json` (sha256
  `62c48c890d3d9ffab265bab24bf0b8039a05317cce46c22337afe277aa57b3f8`) per case
  within 1e-6 when that artifact is available. A mismatch stops the experiment
  as a software incident.
- **A2 — position-resolved patching map.** For every component and every
  position `p ≥ p_c`, singleton counterfactual replacement in both directions,
  `d_patch({j}, p)`. Positions `< p_c` are asserted to be exact no-ops. In the
  coordinated-adjective template this yields separate maps at `p_c = 5` and
  `p_t = 6`. Start with the coordinated-adjective template, because it
  separates the cue from the prediction position.
- **A3 — layer-resolved cue-position residual patching.** For the
  coordinated-adjective template, replace `r_L(·, p_c)` with the counterfactual
  for `L = 0..5`. `L = 0` is the full counterfactual (recovery 1 by
  construction); the profile shows at which layer the cue information at `p_c`
  has been consumed by transport.
- **A4 — attention patterns.** Capture every head's attention from query `p_t`
  in the coordinated-adjective template; record attention to `p_c` and to
  every other key position. Rank heads by attention to `p_c`. Patterns are
  measurements, not explanations.
- **A5 — number-axis analysis.** Compute `d_num(L, p)` for every layer at `p_c`
  and `p_t` per template; component number contributions at `L_R`; cosine
  similarity of number axes across templates, across cue words (`two` vs
  `several`), and across the two frames. Compute each component's direct
  effect on the contrast through the final LayerNorm (bias-aware; the
  per-component direct effects plus bias terms must reconstruct `c(x)` within
  1e-4 nats on every case, and a violation is a software incident that stops
  the experiment exactly as an A1 mismatch does), and its indirect effect as
  total minus direct.
- **A6 — path and mediation tests.** For the H1/H2/H3 rule: (i) counterfactual
  replacement of the residual at `p_c` at the input of the candidate T head's
  layer (`r_{L_T}(·, p_c)`, the tensor that head reads), once alone and once
  with the candidate T head frozen at `p_t` to its clean output; the blocked
  fraction is `(R_unfrozen − R_frozen) / R_unfrozen`; (ii) patch E∪T and
  measure the reproduced fraction of R outputs' counterfactual
  number-projection change (mediation); (iii) patch R outputs alone.
- **A7 — abstractness tests.** Cross-cue and cross-frame resample replacements
  of E at `p_t` in the cue-final templates and of T at `p_t` in the
  coordinated-adjective template; same-number sources are predicted to move
  the contrast little, opposite-number sources like the counterfactual.
- **A8 — necessity and self-repair.** Pair-mean ablation of each candidate
  component and of the candidate set; loss of contrast; the change in every
  other component's direct effect after ablation (compensation ratio, defined
  under residual accounting); conditional co-ablation of the top compensator
  with the ablated component to expose backup routes.
- **A9 — isolation (reconstruction) test.** With the candidate set `S_M` held
  at its clean activations, replace every other universe component at `p_t`
  (and at `p_c` when `p_c ≠ p_t`) with the pair mean; measure the retained
  contrast difference `d_iso` and the fraction of pairs that keep the correct
  sign in both conditions.
- **A10 — mechanism statement and set selection.** Apply the hypothesis rule,
  then choose `S_M` as the smallest set satisfying the Tier A floors below,
  breaking ties by the canonical component order. Write the mechanism
  statement (format below) as mechanism version `M1`.

Tier A floors (a mechanism that fails them cannot proceed to Tier B):

- `|S_M| ≤ 6` components at `p_t` and `≤ 2` at `p_c`;
- counterfactual recovery of `S_M` at its declared positions ≥ 0.70 overall,
  ≥ 0.60 in every template, and ≥ 0.60 in every rule class pooled across
  templates (development: simple-suffix 60, sibilant-`es` 36, consonant-`y`
  24 cases) — the same strata P1 tests on the reserve;
- isolation faithfulness `F = mean(d_iso) / mean(d_full)` ≥ 0.50 overall;
- every declared role has at least one component with a number contribution
  or direct effect of the declared sign.

Budget and stop rule: at most three mechanism versions (`M1`–`M3`) may be
written in Tier A. Every version, including rejected ones, is retained in the
results artifact. If no version passes the floors, the experiment ends with
outcome `NO_COMPACT_MECHANISM`, no lock is written, and the reserve stays
untouched. Tier A is bounded to one M1 8 GB Mac on CPU; a single discovery
pass is estimated at well under 10,000 short forward passes.

A later version never re-runs discovery from scratch and never changes the
hypothesis-rule quantities, which are recorded once. It is produced by one of
two mechanical revisions of A10, chosen by which family failed:

- if a set-level family failed (P1–P4, P6–P8 analogues), `S_M` is extended in
  the development singleton-ranking order at the failing role's position, one
  component at a time within the size budget, until the failed family's
  development value exceeds its floor by at least the observed holdout
  shortfall;
- if a hypothesis-specific family failed (P5, P9 analogues), the mechanism
  adopts the next row of the decision tree that the recorded quantities did
  not exclude, and its statement and discriminating predictions are rewritten
  for that row.

If neither revision is possible within the budget, the version is recorded as
rejected.

### Tier B — calibration on `selection-holdout` (one or two passes)

The 114 single-token holdout cases (the six two-token `peach` cases are
scored behaviorally only and excluded from interventions) were never used for
component ranking. They serve two purposes: to check that the mechanism found
on development data reproduces on nouns it was not built from, and to turn
the prediction families below into numeric predictions with tolerance bands.

Rules:

- Every measurement in the prediction families is computed on holdout once for
  the current mechanism version. The floors must pass on holdout; otherwise
  the mechanism returns to Tier A (a new version) and a second and final
  calibration pass is allowed. There is no third pass.
- Numeric predictions for the reserve are the holdout point estimates. Each
  prediction's band is `± max(0.10, 1.5 × the holdout bootstrap standard
  error)` for recoveries and fractions, and `± 0.5 nats` for mean `d_full`.
  The bootstrap resamples matched pairs with replacement, stratified by
  template (38 single-token holdout pairs per template, resampled to the same
  count), 1,000 resamples from one generator seeded with `20260918` and
  consumed in a fixed family order; the standard error is the standard
  deviation of the resampled statistic. Bands are written into the lock
  verbatim.
- For every discriminating prediction, the lock stores the band implied by
  the chosen hypothesis and the band implied by each row the tree did not
  exclude, using the same tolerance rule. A prediction counts as
  discriminating only if those bands are disjoint.
- The number of calibration passes and every holdout number are recorded. A
  mechanism that needed two passes is labeled as such in every report.

### Tier C — confirmation on `future-reserve` (exactly once)

The reserve is executed only by the `confirm` phase, which refuses to run
unless the committed preregistration lock validates. It runs the behavioral
measurement and every locked prediction family, in one pass, and never
re-ranks, re-selects, or refits anything. A second scientific attempt requires
a new protocol version; the first result remains reported.

## Mechanism statement format

The lock carries the mechanism as text in a fixed form. Every symbol names a
concrete component, position, direction, or scalar that is stored in the lock
with its digest. Illustrative form (the content is decided in Tier A, not
here):

```text
VARIABLES
  n_c(x)  number at the cue position: sign of ( E-output(x, p_c) · d_E )
  n_t(x)  number at the target position: sign of ( Σ_{j ∈ T ∪ E@p_t} o_j(x, p_t) · d_T )
  c(x)    log P(singular | x) − log P(plural | x)

STAGES
  E  <components at p_c>  : cue token → n_c along d_E                   (encoding)
  T  <heads at p_t>       : n_t := n_c  when p_c ≠ p_t; identity otherwise (transport)
  R  <components at p_t>  : c(x) ≈ b_N − g_template · n_t                   (readout)
  D  direct paths         : listed with their measured share of d_full

PAIR-LEVEL PREDICTIONS
  d_full  ≈ 2 · g_template ; sign(d_full) = sign(n_t(x_A) − n_t(x_B)) with n_t(x_A) = −1, n_t(x_B) = +1
```

The statement must also list, in plain language, what each stage does and
does not explain (for example, that `b_N` — the per-noun offset — is not
predicted by the mechanism and is why the plural condition fails on some
spelling-change nouns, if that is what Tier A shows).

## Preregistered prediction families

Each family has a **floor** fixed here, which cannot be lowered, and a
**band** written into the lock from Tier B. A family passes its floor or
fails it; independently it lands inside or outside its band. Families marked
*primary* determine the outcome; *secondary* families are reported and feed
claim review but cannot change the outcome. Families marked
*coordinated-only* apply to the coordinated-adjective template; the rest apply
to every template and, where stated, per rule class pooled across templates
(reserve: simple-suffix 60, consonant-`y` 36, sibilant-`es` 24 cases).

| ID | Family | Intervention on reserve | Floor | Primary |
|---|---|---|---|---|
| B1 | Behavior replicates | none | primary pairwise accuracy ≥ 103/120 (as defined in the screen) and contrast flip ≥ 80% | precondition |
| B2 | Pair-level behavior predicted | none (captures only) | `n_t` read from the locked components and axis equals the cue number in ≥ 95% of conditions; `d_full > 0` in ≥ 95% of pairs; mean `d_full` per template inside its band | primary |
| P1 | Sufficiency of `S_M` | counterfactual replacement of `S_M` at declared positions | `R ≥ 0.70` overall, `≥ 0.60` per template and per rule class | primary |
| P2 | Specificity | 100 random sets of size `|S_M|` drawn per position slot (seed `20260918`) | `R(S_M) ≥ 2 · max(median random recovery, 0.05)` | primary |
| P3 | Isolation / reconstruction | pair-mean replacement of every non-`S_M` component at `p_t` (and `p_c`); the token embedding stays intact and its direct share is reported under Q1 | `F ≥ 0.50` overall, `≥ 0.40` per template; correct sign retained in both conditions for ≥ 70% of pairs | primary |
| P4 / P4′ | Cue-position encoding (new intervention position) | P4: counterfactual replacement of E only, at `p_c`; P4′: pair-mean ablation of every non-embedding component at `p_c` | `R ≥ 0.50` (P4) or retained shift ≥ 0.50 (P4′) | primary, coordinated-only |
| P5 | Transport sufficiency and path | (a) counterfactual replacement of the declared T set alone at `p_t`; (b) counterfactual replacement of `r_{L_T}(·, p_c)` with T frozen at `p_t` | under H1: (a) `R ≥ 0.50` for the single T head and (b) blocked fraction ≥ 0.50; under H2: (a) `R ≥ 0.70` for the declared T set and (b) its blocked fraction inside the locked band | primary, coordinated-only |
| P6 | Abstract number variable | cross-cue resample of E at `p_t` (`cardinal` ↔ `quantifier`, same noun) | opposite-number source `R ≥ 0.50`; same-number source `R ≤ 0.25` | primary (cue-final templates) |
| P7 | Cross-frame control | cross-frame resample of `S_M` (same noun, other frame); for E this is a structural no-op because the layer-0 MLP output depends only on the cue token, so the control is informative for T and R | same-cue `R ≤ 0.25`; opposite-cue `R ≥ 0.5 · R(S_M)` | primary |
| P8 | Necessity and self-repair | pair-mean ablation of T (coordinated) or of E (cue-final) | contrast loss `1 − d_ablated / d_full ≥ 0.30`; compensation ratio inside its band | primary for loss, secondary for compensation |
| P9 | Mediation | counterfactual replacement of E∪T; number projections of R outputs | reproduced fraction ≥ 0.50 (H1/H2) or the H3-specific band | primary |
| S1 | Shared number axis | none (captures only) | cosine between template-specific `d_num(L_R, p_t)` estimates on reserve ≥ 0.70 | secondary |
| S2 | Rule-class residual | P1 and P3 restricted to spelling-change nouns | reported | secondary |
| S3 | Failures are readout-offset failures | none (captures only) | in reserve conditions the model gets wrong, `n_t` still equals the cue number in ≥ 80% of them | secondary, discriminating |
| Q1 | Residual accounting | derived from P1, P3, B2 | reported, never gated | report |

Discriminating predictions: the lock must contain at least one prediction
whose band differs between the chosen hypothesis and each rejected one (P5 and
P9 are designed for this; S3 separates "the cue is misread" from "the number
is read correctly but the readout gain does not overcome the noun's prior").
A reserve result that falls inside the rejected hypothesis's band and outside
the chosen one contests the mechanism even when P1–P3 pass.

### Outcome rule

Applied mechanically after the single reserve run:

- `BEHAVIOR_NOT_REPLICATED` — B1 fails. The mechanism test is inconclusive,
  not falsified; no other family is interpreted as confirmatory.
- `MECHANISM_CONFIRMED` — every primary floor passes and every primary band
  is hit.
- `MECHANISM_SUPPORTED_MISCALIBRATED` — every primary floor passes; at least
  one primary band is missed. The missed bands are listed as the mechanism's
  quantitative residual.
- `MECHANISM_CONTESTED` — every primary floor passes but at least one
  discriminating prediction lands outside the chosen hypothesis's band and
  inside a rejected hypothesis's band. A result outside both bands is a
  missed band (miscalibration), not a contest.
- `MECHANISM_NOT_SUPPORTED` — at least one primary floor fails. The failing
  families name the falsified stage (encoding, transport, readout, isolation,
  specificity, or abstraction).

There are no scientific retries. A software defect discovered after the
reserve run follows the screening incident rule: incident note, invalidation
of the affected artifacts, committed fix, and a rerun of the *entire*
confirmation only under a new protocol version whose report also carries the
invalidated result.

## Residual accounting and self-repair definitions

- **Unexplained sufficiency residual:** `1 − R(S_M)` overall, per template,
  and per rule class.
- **Unexplained isolation residual:** `1 − F(S_M)` on the same strata.
- **Behavioral residual:** the fraction of reserve pairs whose `d_full` sign
  or band the mechanism mispredicts, and the mean absolute error of the
  pair-level `d_full` prediction in nats.
- **Direct-effect decomposition:** for each component `j` at `p_t`, the direct
  effect `DE_j(x)` is its output's contribution to `c(x)` through the actual
  final LayerNorm statistics of that run and the complete unembedding; bias
  terms are reported separately; `Σ_j DE_j + biases = c(x)` is verified to
  1e-4 nats on every case.
- **Compensation ratio:** after pair-mean ablation of a component `k`,
  `Σ_{j ≠ k} (DE_j' − DE_j) / DE_k`, computed on the pair difference. Values
  near 0 mean no repair; values near 1 mean full backup. Conditional
  co-ablation of `k` with its top compensator reports how much the necessity
  effect grows.

## Preregistration lock

`experiments/005-regular-plural-mechanism/preregistration-lock.json` is a
hard integrity boundary in the style of the Experiment 004 selection lock. It
records:

- manifest path and digest; the exact case IDs of the three splits and a
  digest of the reserve cases in manifest order;
- pinned model identifier and revision; runtime and seeds;
- the mechanism statement text and its digest; the chosen hypothesis row and
  the discriminating quantities that selected it;
- `S_M` with each component's role and declared positions; the number axes
  (`d_E`, `d_T`, `d_num(L_R, p_t)`) stored as separately named tensor
  artifacts with shape, dtype, and SHA-256 digests;
- every prediction family with its floor, point prediction, band, and the
  holdout numbers it came from;
- the Tier A results-artifact digest, the Tier B results-artifact digest, the
  number of calibration passes, and the mechanism version;
- the protocol/code commit at lock time and the lock creation timestamp.

The `confirm` phase verifies schema and content, recomputes the manifest and
artifact digests, checks that the lock file is tracked and committed, that the
working tree is clean, that the current code commit is the lock commit or a
descendant whose diff touches no scientific path, and that the reserve has
never been executed. It has no override flag. It never regenerates the lock.

## Claim and anomaly handling

- No claim is created during Tier A or Tier B.
- At the lock commit, claim `C002` is created at maturity `PROPOSED`,
  disposition `ACTIVE`, origin `confirmatory`, with the mechanism statement as
  its hypothesis and the prediction families as its predictions. Its gates
  cite Tier A and Tier B evidence as exploratory and methodological only.
- After the reserve run, claim review follows the methodology: the reserve
  evidence can support at most `CAUSAL_EVIDENCE` (Level 3) and
  `FALSIFICATION_TESTED` (Level 4). Level 5 may be claimed only along the axis
  "new lexical items within the same three templates". Level 6 gates are
  reviewed individually (mechanism specified; unseen behavior predicted;
  unseen interventions predicted; reconstruction retained; residual
  quantified) and the claim states which are met; `ALGORITHMIC_EXPLANATION`
  is not asserted merely because all six are checked within one behavior.
- An unexpected measurement (for example a component with a large negative
  number contribution, or self-repair that exceeds the removed effect) is
  recorded under `research/anomalies/` with full discovery provenance rather
  than folded into the mechanism silently.

## Relationship to prior work and novelty boundary

Experiment 005 does not claim to discover how grammatical number works in
Pythia. The audited close work already establishes plural and inflection
representations (Li and Subramani, ACL 2026,
[arXiv:2506.02132](https://arxiv.org/abs/2506.02132)), subject–verb agreement
feature circuits in Pythia-70M (Marks et al., ICLR 2025,
[arXiv:2403.19647](https://arxiv.org/abs/2403.19647)), causal agreement
variables across Pythia scales (Arora, Jurafsky, and Potts, ACL 2024,
[CausalGym](https://aclanthology.org/2024.acl-long.785/)), and head/MLP
agreement accounts in other models. The regular-plural finalist audit and its
comparison table remain the authoritative statement of what each close work
lacks.

The methodological neighborhood is also active. Program extraction from
trained networks exists for small RNNs on synthetic tasks (MIPS, Michaud et
al., [arXiv:2402.05110](https://arxiv.org/abs/2402.05110)); compact formal
guarantees exist for small transformers on Max-of-K (Gross et al., NeurIPS
2024, [arXiv:2406.11779](https://arxiv.org/abs/2406.11779)); solver-checkable
circuit explanations exist for small or post-hoc-modified transformers
(Somani, 2026, [arXiv:2605.24033](https://arxiv.org/abs/2605.24033)); and
attention heads in GPT-2 and Llama-scale models have been approximated by
synthesized programs (Hayes, Li, and Andreas, 2026,
[arXiv:2606.19317](https://arxiv.org/abs/2606.19317)). Self-repair after
ablation is a documented confound (Rushing and Nanda, 2024,
[arXiv:2402.15390](https://arxiv.org/abs/2402.15390); conditional co-ablation,
2026, [arXiv:2607.01940](https://arxiv.org/abs/2607.01940)), which is why P8
and the compensation ratio are part of the design rather than optional.

What this experiment adds, if it succeeds, is the full prospective chain for
one naturally learned computation in an unmodified pretrained model: exposed
data → explicit mechanism statement → committed numeric predictions about
untouched examples, including interventions at positions and with replacement
sources that the behavior-selecting screen never used → one execution →
residual accounting and self-repair bounds. The mechanism is built on
development data, calibrated once or twice on holdout data, and tested once
on reserve data; the reserve is the only evidence that counts as confirmatory.
"Prospective neural decompilation" is the protected idea; the plural
mechanism is its first instance.

## Interpretation limits

- The behavior is pairwise preference between two noun forms; the mechanism
  does not predict full-vocabulary top-1 output and the experiment does not
  test it.
- Number axes are mean-difference directions estimated on development data;
  they are analysis coordinates, not claimed model features.
- Pair-mean and resample replacements are justified for these matched pairs
  only; conclusions about necessity are scoped to those replacement
  distributions.
- Direct-effect decompositions through LayerNorm are exact for the measured
  run but attribute nonlinear interactions to whichever component's output
  changed; they are accounting, not mechanism.
- Cue-final templates cannot distinguish "transport" from "local encoding"
  because `p_c = p_t`; transport claims rest on the coordinated-adjective
  template alone, with a cue-to-noun distance of exactly one token.
- Nothing here generalizes beyond the pinned checkpoint, the three templates,
  the two frames per template, the cue words `one`/`two`/`each`/`several`,
  and single-token nouns of three regular rule classes.

## Outputs

```text
experiments/005-regular-plural-mechanism/
    README.md
    run.py                       # phases: validate, discover, calibrate, lock, confirm, report
    preregistration-lock.json    # committed before confirm; absent until then
    evidence/                    # verbatim copies of the reports actually used for claims

outputs/experiment-005/          # generated, gitignored
    discovery-results.json       # every Tier A measurement, every mechanism version
    calibration-results.json     # every Tier B pass
    candidate-lock.json          # written by `lock`, reviewed and moved by hand
    confirmation-results.json    # the single reserve run
    axes/                        # number-axis tensors with digests
    report.md
```

The directory is created only after this design is approved. Generated
outputs stay out of version control; evidence copies of the final reports are
committed alongside the claim.

## Minimal implementation boundary

After approval, implementation is limited to:

1. one experiment module (for example `plural_mechanism.py`) that reuses
   `candidate_screening` manifest loading, contrast and recovery arithmetic,
   deterministic random sets, and the component universe, and adds
   position-indexed capture/patching, pair-mean and resample sources, freeze
   interventions, number-axis and direct-effect analysis, the hypothesis rule,
   prediction-family evaluation, band computation, and the lock schema;
2. one runner with the six phases above, phase isolation enforced through the
   results state exactly as in the screen (a `confirm` without a committed
   lock, a second `confirm`, or a third `calibrate` refuses to run);
3. focused offline tests for position normalization, structural no-op
   assertions, exact decomposition identities, pair-mean and resample
   construction, freeze correctness, hypothesis-rule determinism, band
   arithmetic, lock validation, reserve non-execution, and phase refusal;
4. one Markdown report renderer.

It must not introduce sparse autoencoders, automated circuit search beyond the
enumerated 54-component universe, gradient attribution, path patching through
attention keys/queries, new prompts, Pythia-160M, a plugin system, a database,
or a dashboard. The instrumentation modules are extended only if a narrow
defect is found.

## Approval and stopping condition

This design is complete when it has been reviewed against the research
methodology and the screening evidence, amended where the review requires,
and explicitly approved by the user. Only then is an implementation plan
written and the experiment directory created. Tier C may begin only after the
preregistration lock is committed. The experiment stops after the single
reserve run, the outcome rule, the claim review, and the evidence copies, with
every rejected mechanism version and every missed band retained in the record.
