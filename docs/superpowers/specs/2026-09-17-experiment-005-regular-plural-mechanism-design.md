# Experiment 005: Count-Cued Noun Number Selection — Prospective Mechanism Design

**Date:** 2026-09-17

**Status:** Draft, revision 4 (after the third external review). Not approved.
No Experiment 005 directory, claim, preregistration lock, extension manifest,
or model run exists. The 120 `regular-plural` future-reserve cases remain
unexecuted.

**Scope:** `regular-plural` (count-cued English noun number selection) in the
pinned `EleutherAI/pythia-70m-deduped` checkpoint, using the frozen screening
manifest `screening/behavior-candidates/manifest-v1.json` plus one frozen,
tokenizer-validated extension set of prompts defined below.

## Revision history

- **Revision 1 (2026-09-17, commit `a414cc8`).** Initial draft.
- **Revision 2 (2026-09-17).** Changes after the first review, all made before
  any Experiment 005 model output exists:
  1. the number variable now has a frozen origin and scale (centered, scaled
     projection) instead of a bare direction;
  2. the mediation family is a chain test (patch E alone, let T recompute,
     then repeat with T frozen) instead of a joint E∪T patch;
  3. the midpoint replacement is renamed *pair-centered neutralization* and
     necessity conclusions are scoped to that replacement distribution;
  4. an explicit, separately executable **mechanism program** is required, and
     the outcome rule has a second axis that distinguishes "these components
     are sufficient" from "we recovered what they compute";
  5. a structural fact of the manifest was recognized while addressing item 4:
     every split shares the same twelve prompts, so the reserve's only novelty
     is twenty new nouns (unembedding rows). Prompt-side prospective tests
     therefore require prompts the model has not seen in this project; a frozen
     extension set (new frames and new cue words) is added, and every reserve
     test is relabeled as what it actually is.
- **Revision 3 (2026-09-17).** Changes after the second review, still before
  any Experiment 005 model output exists:
  1. the program's readout no longer linearizes the final LayerNorm (the
     previous fixed-denominator expression omitted the variance term and
     `ε`); it applies the exact LayerNorm to a reconstructed residual around
     a frozen context, so any remaining error is mechanism error, not
     approximation error;
  2. bands are made mechanical: each family states whether it has a band and
     where the band comes from; X3/X4 tolerances derive from the Tier B
     program residual and are frozen in the lock;
  3. percentage floors with fixed denominators are restated as exact counts;
  4. the list of deliberately ambiguous cue words is corrected to the frozen
     twelve.
- **Revision 4 (2026-09-17).** Changes after the third review, still before
  any Experiment 005 model output exists:
  1. a program-eligible encoding boundary: the mechanism program may encode
     the cue only through token-local paths (the token embedding, `L00.MLP`,
     or their declared combination); a context-dependent encoding component
     may stay in the circuit account but caps the decompilation axis at
     `CIRCUIT_ONLY` unless a further weight-only program covers it within the
     version budget;
  2. the E-output equality check is stated for `E_program` only;
  3. the exact LayerNorm's variance semantics are frozen (population variance,
     `correction=0`, `ε = 1e-5`).

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

The experiment has two halves that the methodology keeps separate:

1. **Discovery (exploratory):** trace the pathway on already-exposed data,
   choose between competing mechanisms using predeclared discriminating
   measurements, write the mechanism down as an explicit statement with named
   components, variables, and transformations, and implement that statement as
   a program that runs without the network.
2. **Confirmation (prospective):** freeze the statement, the program, and
   numeric predictions about unseen nouns, unseen prompts, and unseen
   interventions in a committed preregistration lock, then execute the
   confirmation set once.

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
- **Known structural artifact.** Pythia-70M has GPT-NeoX parallel residual
  blocks (`resid_post = resid_pre + attn(ln1(resid_pre)) + mlp(ln2(resid_pre))`,
  both sub-blocks reading the block *input*) and rotary attention with no
  additive positional embedding, so the layer-0 MLP output at a position is a
  function of that position's token embedding only; the screen confirmed this
  empirically with the exactly-zero coordinated-adjective result below. In the cardinal and quantifier templates the cue
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

### Structural fact: twelve prompts

The noun never appears in a prompt; it is only the scored next token. Every
`regular-plural` case in every split is one of exactly twelve prompts (three
templates × two frames × two cues):

```text
cardinal               The display contains one|two      The tray holds one|two
quantifier             The catalog lists each|several    The inventory records each|several
coordinated-adjective  Lena and Omar displayed one|two small
                       Mira and Noah packed one|two bright
```

Consequently every activation, every residual, and every component-level
intervention outcome is identical across the 20 nouns of a split. A case's
contrast is the readout of one of twelve final-position residuals through the
noun pair's unembedding rows. The reserve therefore contributes exactly one new
thing: twenty unembedding row pairs the project has never scored. It
contributes no new prompt-side computation, and any intervention run on the
reserve has a residual-level outcome already known from Tier A on the same
twelve prompts.

Two consequences shape the rest of this design:

- Tests on reserve nouns are **readout-generalization** tests: they ask
  whether the shift the mechanism produces lies along a direction that unseen
  nouns' singular–plural unembedding differences read with the predicted sign
  and size. That is a real and necessary test of the readout stage. It is not
  a prospective test of the encoding or transport stages.
- Prospective tests of encoding, transport, frame invariance, and the
  mechanism program require prompts the project has never run. The
  **extension set** below supplies them: new sentence frames for each template
  and new cue words in the existing frames, frozen by tokenizer-only rules
  before Experiment 005 executes anything.

Compute also collapses: one forward pass per prompt scores every noun, so a
full intervention family costs tens of forward passes, not thousands.

### Exposure ledger

| Set | Cases | Behavioral output inspected | Intervention output inspected | Role in Experiment 005 |
|---|---|---|---|---|
| `selection-development` | 120 cases (6 frame pairs × 20 nouns; 240 conditions over the 12 prompts) | yes | yes (component ranking and top-`k` validation) | Tier A discovery data; treated as fully exposed |
| `selection-holdout` | 120 (114 single-token; 6 `peach` cases are two-token) | yes (contrasts known) | no | Tier B calibration data (readout generalization and bands); one or two passes |
| `future-reserve` | 120 (all single-token; 20 new nouns) | no | no | Tier C readout-generalization confirmation; executed exactly once after the lock commit |
| extension set | 6 new frames × 2 cues, plus 6 original frames × up to 12 new cue words; scored with the 20 reserve nouns | no | no | Tier C prompt-side confirmation; frozen before Tier A; executed exactly once after the lock commit |

The reserve nouns were fixed by the screening design before any model run:
per-split literal pools, disjoint across splits, with a first-ten-eligible
tokenizer rule and pre-output amendments recorded in that design. No prompt
outside the manifest and the frozen extension set is executed in
Experiment 005. Longer cue-to-noun distances, new nouns beyond the reserve,
Pythia-160M, and other model families are excluded and listed under future
work.

## Fixed coordinates

- Model: `EleutherAI/pythia-70m-deduped`, revision
  `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; CPU float32 is the scientific
  runtime; TransformerBridge compatibility mode disabled; `use_attn_result`
  enabled explicitly; every forward under `torch.inference_mode()`.
- Manifest: `screening/behavior-candidates/manifest-v1.json`, content sha256
  `1c50c2e8bbf95f9aa4331ac57899f77c2a6be6b17d4cb2a5ac4ecc332c5e703c`. Only
  `regular-plural` cases are read. Prompts are batched by identical length;
  no padding is ever introduced.
- Runtime seed `20260916` (as in the screen, so baseline contrasts replicate);
  control-sampling seed `20260918` for every random component set and every
  bootstrap drawn in this experiment.
- Positions, zero-indexed from the first prompt token:

| Template | Example prompt | Tokens | Cue position `p_c` | Target position `p_t` |
|---|---|---|---|---|
| `cardinal` | `The display contains one` / `two` | 4 | 3 | 3 |
| `quantifier` | `The catalog lists each` / `several` | 4 | 3 | 3 |
| `coordinated-adjective` | `Mira and Noah packed one bright` / `two bright` | 7 | 5 | 6 |

  For every prompt, including extension prompts, `p_c` is the index of the
  cue token and `p_t` is the final prompt position; both are derived from the
  tokenization and stored per prompt. Positions before `p_c` are
  token-identical within a matched pair; the runner asserts that every
  captured activation there is bitwise identical between pair members, which
  makes any intervention there a no-op by construction, and it never spends
  forward passes on those positions.
- Metric definitions are reused verbatim from the screening design:
  `c(x)`, `m(x) = y(x) c(x)`, `d_full = c(x_A) - c(x_B)`, the bidirectional
  aligned patched shift `d_patch(S)`, and aggregate recovery
  `R_D(S) = mean(d_patch) / mean(d_full)` with the same denominator floors
  (0.25 nats overall, 0.10 nats per template and per rule-class stratum).
  Raw recoveries are reported without clipping. Patched and unpatched
  contrasts of one stratum always come from identical batch compositions so
  that an exact replacement yields an exactly zero shift.

### Extension set

`experiments/005-regular-plural-mechanism/extension-v1.json` is frozen by
tokenizer-only rules and committed before the `discover` phase can run; the
results state records its digest and every scientific phase refuses to run
without it. It contains no new nouns: every extension prompt is scored with
the 20 reserve nouns, and its cases are never executed before Tier C.

1. **New frames** — two per template, fixed here. Cardinal: `The basket
   carries {cue}`, `The museum owns {cue}`. Quantifier: `The menu offers
   {cue}`, `The report cites {cue}`. Coordinated-adjective: `Ana and Luis
   painted {cue} tiny`, `Kai and Sara carried {cue} heavy`. Cues are the
   template's original pair (`one`/`two`, `each`/`several`, `one`/`two`).
   Every frame was checked to tokenize with a single-token cue and, for the
   coordinated frames, a single-token adjective after it; total length is not
   constrained.
2. **New cue words** — the first twelve tokenizer-eligible entries of this
   fixed ordered list, each substituted into the cue slot of all six original
   frames: `a`, `the`, `three`, `four`, `five`, `ten`, `many`, `few`, `some`,
   `all`, `both`, `every`, `any`, `no`, `another`, `single`, `multiple`,
   `numerous`, `twelve`, `hundred`. Eligibility means the space-prefixed word
   is one token under the pinned tokenizer. All twenty were verified eligible
   while drafting this revision, so the frozen set is the first twelve:
   `a` through `every`. Ambiguous or non-numeric entries (`a`, `the`, `some`,
   `all`) are deliberate: the mechanism program must predict them from
   its cue lexicon rather than from a hand-written number label, and the
   model's behavior on them is unknown.

The extension set is built once by a tokenizer-only builder, digested, and
committed. No model output influences it, and it cannot be edited after the
`discover` phase has run.

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
  frame. This is the sufficiency-oriented source used throughout the screen.
- **Pair-centered neutralization (mean):** `0.5 * (act(x_A) + act(x_B))` for
  the same frame and position — the midpoint of the two observed states. It
  is the necessity-oriented source. In a nonlinear network the midpoint is
  not equivalent to removing "the cue-dependent component"; it is one
  specific, on-manifold-adjacent replacement distribution, chosen because it
  preserves everything the pair shares and is symmetric between the two cues.
  Zero ablation is not used because it is off-distribution for this
  architecture. Every necessity or isolation conclusion is scoped to this
  replacement distribution.
- **Cross-frame resample (resampled):** the other frame of the same template,
  with either the same cue (predicted to move the contrast little) or the
  opposite cue (predicted to move it like the counterfactual). This
  distinguishes number information from frame or lexical-context information.
- **Cross-cue resample (resampled):** the other cue-final template
  (`cardinal` ↔ `quantifier`), so the cue *token* differs (`two` vs
  `several`, `one` vs `each`) while the cue *number* is the same or opposite.
  This tests whether the encoded variable is number rather than token
  identity.

All interventions are exact-shape `REPLACE` operations through the existing
`InterventionPlan` runner; multiple components at one hook are provably
disjoint slices. No new hook kinds are introduced. The "freeze" operation used
in path tests is a `REPLACE` with the component's own clean activation from the
unpatched run of the same prompt.

## Stage vocabulary, number variable, and competing mechanisms

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

### Number variable

For a residual site `r_L(x, p)` (input of block `L` at position `p`), the
number axis is the mean pair difference over development prompts:

```text
d_num(L, p) = mean over development pairs of ( r_L(x_B, p) − r_L(x_A, p) )
```

A bare direction does not define where zero is, so every stage variable is a
**centered, scaled projection** with three frozen parameters — a unit
direction `d̂_s`, an origin `μ_s`, and a scale `σ_s`:

```text
n_s(x) = ( a_s(x) − μ_s ) · d̂_s / σ_s
```

where `a_s(x)` is the activation at site `s`, `μ_s` is the mean of `a_s` over
all development prompts of the relevant templates (both cues, balanced by
construction), and `σ_s` is half the mean projected pair difference, so that
the development cues map to approximately `−1` (singular) and `+1` (plural).
The binary variable is `sign(n_s)`, with threshold `0` after centering. The
three sites are:

- `s = E`: the E components' summed output at `p_c` → `n_c(x)`;
- `s = T`: the T heads' summed output at `p_t` (cue-final templates: the E
  output at `p_t`) → `n_t(x)`;
- `s = R_in`: `r_{L_R}(x, p_t)`, the residual read by the earliest R component.

`d̂_s`, `μ_s`, and `σ_s` are estimated once on development data (Tier A), may
be re-estimated at most once after a Tier B revision, and are stored in the
lock with digests. On the twelve manifest prompts the reserve activations are
identical to development activations, so `n_s` there is a fixed number per
prompt; the variables are informative only on extension prompts, and the
prediction families use them there.

A component's number contribution at `p` is the mean projection of its own
`x_B − x_A` output difference onto `d̂_{R_in}`. Because the residual stream is
an exact sum of component outputs plus the embedding, these contributions sum
to the whole axis magnitude at `L_R`; they are an exact decomposition of *what
arrives*, not a causal claim.

### Competing mechanisms

Three competing mechanisms are stated now, before discovery data are
inspected, together with the measurements that decide between them. The chain
quantities are defined under A6: `m_T` is the fraction of T's counterfactual
number change reproduced when E alone is patched and T recomputes; `m_R` is
the fraction of R's counterfactual number change reproduced by the same
E-alone patch; `m_R|T` is `m_R` with T frozen to its clean output.

| ID | Mechanism | Predicts on development data (coordinated-adjective) |
|---|---|---|
| **H1 — dedicated transport** | E writes number at `p_c`; one or two heads (T) copy it to `p_t`; R reads it. | Top head at `p_t` alone recovers ≥ 0.50; freezing it blocks ≥ 0.50 of the effect of a counterfactual residual patch at `p_c`; `m_T ≥ 0.50`, `m_R ≥ 0.50`, and `m_R|T ≤ 0.5 · m_R`. |
| **H2 — distributed transport / local amplification** | Number reaches `p_t` through many weak heads (or, when `p_c = p_t`, needs no transport); no head is individually decisive; R reads a pooled signal. | Top head < 0.50 but cumulative top-3 heads ≥ 0.70; freezing any single head blocks < 0.50; `m_R ≥ 0.50` with the top-3 frozen jointly giving `m_R|T ≤ 0.5 · m_R`. |
| **H3 — late computation** | R components compute number from cue-dependent inputs not captured by the E/T number axis; E/T are correlates or partial routes. | `m_R < 0.50` while patching R outputs alone recovers ≥ 0.70. |
| **H0 — no compact account** | No set of ≤ 6 components at `p_t` plus ≤ 2 at `p_c` meets the Tier A floors. | Tier A floors fail after the iteration budget. |

The hypothesis selection rule is a total, deterministic decision tree over
five development-data quantities from the coordinated-adjective template:
`q1` = recovery of the top-ranked head alone at `p_t`; `q2` = blocked fraction
of that head (A6); `q3` = `m_R`, the E-alone mediation fraction to R (A6);
`q4` = recovery of the R outputs alone; `q5` = cumulative recovery of the top
three heads.

```text
if q3 < 0.50 and q4 ≥ 0.70:            H3
elif q1 ≥ 0.50 and q2 ≥ 0.50:          H1   (flag HYPOTHESIS_AMBIGUOUS if q3 < 0.50)
elif q5 ≥ 0.70:                        H2   (flag HYPOTHESIS_AMBIGUOUS if q3 < 0.50)
else:                                  H2 as the working row, flag HYPOTHESIS_AMBIGUOUS
```

The tree is applied once to the recorded numbers and the mechanism statement
is written for the resulting row. A flagged lock must carry a discriminating
confirmation prediction for every row the tree did not exclude. The
hypotheses are not exclusive across templates: the cue-final templates need no
T, and the statement must cover both regimes.

Encoding has one predeclared branch. Let `L_T` be the layer of the earliest
declared T head. The number axis at the input of that layer at `p_c`,
`d_num(L_T, p_c)`, decomposes exactly into the cue token embedding difference
plus the outputs of components at `p_c` below `L_T`. If `L00.MLP` (with any
other declared E component) carries at least half of that axis, E is those
components and prediction family P4 applies. If the embedding difference alone
carries more than half, the mechanism declares E as the direct embedding path
and P4′ applies instead: pair-centered neutralization of every non-embedding
component at `p_c` must retain at least half of the contrast shift. The rule
is applied once, on development data, and recorded in the lock.

**Program-eligible encoding boundary.** The circuit's E may contain any
component at `p_c` that the rule above selects. The mechanism program's
encoding, `E_program`, is restricted in protocol v1 to **token-local** paths:
paths whose output at `p_c` is computable exactly from the cue token and
frozen weights without evaluating any contextual transformer layer. Under
Pythia's parallel-residual architecture these are exactly the token embedding
`embed[token]`, the layer-0 MLP `MLP_0(ln2_0(embed[token]))`, and their
declared combination. An attention head at any layer, or an MLP at layer 1 or
above, reads a contextual residual and is therefore not program-eligible even
if Tier A places it in E. If such a component is needed in E to satisfy the
circuit floors, it stays in `S_M` for the circuit account, but the program
encodes the cue from the token-local paths only, and the program's Tier A
floors are evaluated for that `E_program`. If those floors then fail, the
mechanism version records `PROGRAM_CAPPED`: the X families are still run and
reported, the decompilation axis cannot exceed `PROGRAM_FAIL`, and the best
reachable outcome is `CIRCUIT_ONLY`. The only escape is a further
preregistered weight-only program for the contextual component, written as a
new mechanism version within the existing three-version budget; protocol v1
defines no such program, so a contextual cue encoding is a documented failure
of the simple decompiler, not something the lexicon may absorb.

## Three-tier data policy

### Tier A — discovery on `selection-development` (exploratory)

All 120 development cases (twelve prompts, twenty nouns) are discovery data.
The compactness partition from the screen is no longer a validation boundary
because both halves were inspected during the screen. Every Tier A number is
exploratory and none is confirmatory evidence for the mechanism. Extension
prompts are never run in Tier A.

Ordered protocol; each step writes its measurements to the results artifact
before the next step is interpreted:

- **A0 — runtime contract.** Run the pinned Pythia-70M Bridge contract test on
  the scientific runtime and record its pass in provenance.
- **A1 — baseline replication.** Recompute `c(x_A)`, `c(x_B)`, `d_full` for
  all 120 development cases and 114 single-token holdout cases (behavioral
  only). Development aggregates must match the screening report (mean
  `d_full` 5.231 development; validation denominators 4.639 overall, cardinal
  4.606, coordinated-adjective 4.969, quantifier 4.343) within 1e-3 nats, and
  match the screening `results.json` (sha256
  `62c48c890d3d9ffab265bab24bf0b8039a05317cce46c22337afe277aa57b3f8`) per case
  within 1e-6 when that artifact is available. A mismatch stops the experiment
  as a software incident.
- **A2 — position-resolved patching map.** For every component and every
  position `p ≥ p_c`, singleton counterfactual replacement in both directions,
  `d_patch({j}, p)`. In the coordinated-adjective template this yields separate
  maps at `p_c = 5` and `p_t = 6`. Start with the coordinated-adjective
  template, because it separates the cue from the prediction position.
- **A3 — layer-resolved cue-position residual patching.** For the
  coordinated-adjective template, replace `r_L(·, p_c)` with the counterfactual
  for `L = 0..5`. `L = 0` is the full counterfactual (recovery 1 by
  construction); the profile shows at which layer the cue information at `p_c`
  has been consumed by transport.
- **A4 — attention patterns.** Capture every head's attention from query `p_t`
  in the coordinated-adjective template; record attention to `p_c` and to
  every other key position. Rank heads by attention to `p_c`. Patterns are
  measurements, not explanations.
- **A5 — number-axis analysis.** Estimate `d̂_s`, `μ_s`, `σ_s` for every site;
  compute `d_num(L, p)` for every layer at `p_c` and `p_t` per template;
  component number contributions at `L_R`; cosine similarity of number axes
  across templates, across cue words (`two` vs `several`), and across the two
  frames. Compute each component's direct effect on the contrast through the
  final LayerNorm (bias-aware; the per-component direct effects plus bias
  terms must reconstruct `c(x)` within 1e-4 nats on every case, and a
  violation is a software incident that stops the experiment exactly as an A1
  mismatch does), and its indirect effect as total minus direct.
- **A6 — chain, path, and mediation tests.** (i) Counterfactual replacement of
  E alone at `p_c` (cue-final templates: at `p_t`), letting everything
  downstream recompute; capture the T outputs and R outputs and compute `m_T`
  and `m_R` as the fractions of their counterfactual number changes
  (projections onto the site axes) that this patch reproduces, plus the
  contrast recovery. (ii) The same patch with each candidate T head frozen at
  `p_t` to its clean output, giving `m_R|T` — the share of R's change that
  arrives through routes other than T, such as other heads at `p_t` reading
  `p_c` — and the contrast drop. (iii)
  Counterfactual replacement of `r_{L_T}(·, p_c)` (the tensor T reads), once
  alone and once with the candidate T head frozen; the blocked fraction is
  `(R_unfrozen − R_frozen) / R_unfrozen`. (iv) Patch R outputs alone. Joint
  E∪T patching is also recorded as a sufficiency measure but establishes no
  chain.
- **A7 — abstractness tests.** Cross-cue and cross-frame resample replacements
  of E at `p_t` in the cue-final templates and of T at `p_t` in the
  coordinated-adjective template; same-number sources are predicted to move
  the contrast little, opposite-number sources like the counterfactual.
- **A8 — necessity and self-repair.** Pair-centered neutralization of each
  candidate component and of the candidate set; loss of contrast; the change
  in every other component's direct effect after neutralization (compensation
  ratio, defined under residual accounting); conditional co-neutralization of
  the top compensator with the neutralized component to expose backup routes.
- **A9 — isolation (circuit reconstruction) test.** With the candidate set
  `S_M` held at its clean activations, replace every other universe component
  at `p_t` (and at `p_c` when `p_c ≠ p_t`) with its pair-centered
  neutralization; measure the retained contrast difference `d_iso` and the
  fraction of pairs that keep the correct sign in both conditions.
- **A10 — mechanism statement and set selection.** Apply the hypothesis rule,
  then choose `S_M` as the smallest set satisfying the Tier A floors below,
  breaking ties by the canonical component order. Write the mechanism
  statement (format below) as mechanism version `M1`.
- **A11 — mechanism program.** Implement the statement as the program defined
  below, fit its frozen parameters on development data only, and record its
  development residuals. The program is part of the mechanism version.

Tier A floors (a mechanism that fails them cannot proceed to Tier B):

- `|S_M| ≤ 6` components at `p_t` and `≤ 2` at `p_c`;
- counterfactual recovery of `S_M` at its declared positions ≥ 0.70 overall,
  ≥ 0.60 in every template, and ≥ 0.60 in every rule class pooled across
  templates (development: simple-suffix 60, sibilant-`es` 36, consonant-`y`
  24 cases) — the same strata P1 tests on the reserve;
- isolation faithfulness `F = mean(d_iso) / mean(d_full)` ≥ 0.50 overall;
- every declared role has at least one component with a number contribution
  or direct effect of the declared sign;
- the mechanism program reproduces the development `d_full` sign in every
  pair and its mean per template within 1.0 nats.

Budget and stop rule: at most three mechanism versions (`M1`–`M3`) may be
written in Tier A. Every version, including rejected ones, is retained in the
results artifact. If no version passes the floors, the experiment ends with
outcome `NO_COMPACT_MECHANISM`, no lock is written, and the reserve and
extension sets stay untouched. Tier A is bounded to one M1 8 GB Mac on CPU;
because there are twelve prompts, a full discovery pass is a few thousand
short forward passes at most.

A later version never re-runs discovery from scratch and never changes the
hypothesis-rule quantities, which are recorded once. It is produced by one of
two mechanical revisions of A10/A11, chosen by which family failed:

- if a set-level family failed (P1–P4, P6–P8 analogues), `S_M` is extended in
  the development singleton-ranking order at the failing role's position, one
  component at a time within the size budget, until the failed family's
  development value exceeds its floor by at least the observed holdout
  shortfall;
- if a hypothesis-specific family failed (P5, P9 analogues), the mechanism
  adopts the next row of the decision tree that the recorded quantities did
  not exclude, and its statement, program, and discriminating predictions are
  rewritten for that row;
- if only the mechanism program's floors failed, the program's fitted
  parameters are re-estimated with the extended `S_M` or the new row; its
  functional form (below) does not change.

If neither revision is possible within the budget, the version is recorded as
rejected.

### Tier B — calibration on `selection-holdout` (one or two passes)

The 114 single-token holdout cases (the six two-token `peach` cases are
scored behaviorally only and excluded from interventions) were never used for
component ranking. Because they share the twelve prompts, they test only
readout generalization to twenty nouns the mechanism was not built from, and
they set the numeric bands for the reserve.

Rules:

- Every measurement in the prediction families is computed on holdout once for
  the current mechanism version. The floors must pass on holdout; otherwise
  the mechanism returns to Tier A (a new version) and a second and final
  calibration pass is allowed. There is no third pass.
- Numeric predictions for the reserve are the holdout point estimates. Each
  prediction's band is `± max(0.10, 1.5 × the holdout bootstrap standard
  error)` for recoveries and fractions, and `± 0.5 nats` for mean `d_full`.
  The bootstrap resamples matched pairs (nouns) with replacement, stratified
  by template (38 single-token holdout pairs per template, resampled to the
  same count), 1,000 resamples from one generator seeded with `20260918` and
  consumed in a fixed family order; the standard error is the standard
  deviation of the resampled statistic. Bands are written into the lock
  verbatim. Which families carry bands, and how the X3/X4 tolerances derive
  from the Tier B program residual, is fixed under the prediction families.
- For every discriminating prediction, the lock stores the band implied by
  the chosen hypothesis and the band implied by each row the tree did not
  exclude, using the same tolerance rule. A prediction counts as
  discriminating only if those bands are disjoint.
- Tier B also evaluates the mechanism program on every single-token holdout
  condition and records its residual `c(x) − ĉ(x)`; `RMSE_B` and the X3/X4
  tolerance `τ` are computed from that residual and frozen in the lock.
- The number of calibration passes and every holdout number are recorded. A
  mechanism that needed two passes is labeled as such in every report.

### Tier C — confirmation (exactly once)

The reserve and the extension set are executed only by the `confirm` phase,
which refuses to run unless the committed preregistration lock validates. It
runs the behavioral measurement and every locked prediction family on the
reserve (twelve prompts, twenty new nouns), on the new frames (twelve new
prompts, reserve nouns), and the behavioral, E-patch, and program families on
the cue-word prompts (72 prompts, reserve nouns), in one pass, and never
re-ranks, re-selects, or refits anything. A second scientific attempt requires
a new protocol version; the first result remains reported.

## Mechanism statement format

The lock carries the mechanism as text in a fixed form. Every symbol names a
concrete component, position, direction, origin, scale, or gain that is stored
in the lock with its digest. Illustrative form (the content is decided in
Tier A, not here):

```text
VARIABLES
  n_c(x)  = ( E-output(x, p_c) − μ_E ) · d̂_E / σ_E          number at the cue position
  n_t(x)  = ( T-output(x, p_t) − μ_T ) · d̂_T / σ_T          number at the target position
  c(x)    = log P(singular | x) − log P(plural | x)

STAGES
  E  <components at p_c>  : cue token → n_c                                (encoding; a lexicon over tokens)
  T  <heads at p_t>       : n_t := k_T · n_c  when p_c ≠ p_t; n_t := n_c otherwise (transport; copy gain k_T)
  R  <components at p_t>  : Δr(x) = g_R · n_t · d̂_R  added at the readout output     (readout; gain g_R per template)
  D  direct paths         : listed with their measured share of d_full

PAIR-LEVEL PREDICTIONS
  ĉ(x)      = −u_N · LN( ρ + δ(x) )                  δ(x) = g_R · n_t(x) · d̂_R + declared direct terms
  d̂_full(N) = ĉ(x_A) − ĉ(x_B)                        sign(d̂_full) = +1 for every pair
```

where `u_N = W_U[:, plural] − W_U[:, singular]` for noun `N`, `LN` is the
model's exact final LayerNorm, and `ρ` is the frozen context residual defined
under the mechanism program. The statement must also list, in plain language, what each stage does and does not explain
(for example, that the per-noun offset `b_N` — the contrast a template's
context produces before any cue effect — is context, not mechanism, and is why
the plural condition fails on some spelling-change nouns, if that is what
Tier A shows).

## Mechanism program

The mechanism program is the decompilation deliverable: a separately
implemented function that reproduces the mechanism statement literally and
predicts the model's behavior and intervention outcomes without running the
network.

- **Location and independence.** One module,
  `experiments/005-regular-plural-mechanism/mechanism_program.py`, importing
  neither the capture nor the intervention runner. It may load model
  *weights* — the token embedding, the declared E components' parameters, the
  final LayerNorm parameters, and the unembedding — because a token-level
  lexicon and a readout are weights, not activations. It may evaluate the
  declared E components' own sub-modules on a single token embedding (for
  `L00.MLP`: block 0's second LayerNorm — `post_attention_layernorm` in
  GPT-NeoX, `ln2` in TransformerLens — followed by block 0's MLP, with no
  attention involved), and it does so once for the entire vocabulary to
  produce a stored lexicon table. It must not run attention, any block beyond
  the declared E sub-modules, or any forward pass over a prompt, and its inputs
  never include activations of the prompt being predicted.
- **Inputs.** A template identifier; the cue token id; the noun pair's token
  ids; and a frozen context residual: for the six manifest frames
  `ρ_{frame}`, the development mean over both cues of the residual entering
  `ln_final` at `p_t` for that frame; for a new frame `ρ_{template}`, the mean
  of the template's two `ρ_{frame}` vectors. Both are declared in the lock as
  *unexplained context*: the mechanism explains the cue-dependent increment
  on top of them, not the context itself.
- **Frozen parameters** (all estimated in Tier A, re-estimated at most once
  after a Tier B revision, digested in the lock): the E lexicon table
  `n_c(token) = ( E(token) − μ_E ) · d̂_E / σ_E` for every vocabulary token,
  where `E(token)` is the output of `E_program` for that token — the
  token-local paths declared under the program-eligible encoding boundary
  (for `L00.MLP`: `MLP_0(ln2_0(embed[token]))`, a pure token function; if E is
  the direct embedding path under P4′, `E(token) = embed[token]`; never a
  contextual component); the copy gain
  `k_T` (the change in `n_t` per unit change in `n_c`, estimated by least
  squares on development prompts — a linear summary that the mechanism
  justifies by claiming T attends to `p_c` independently of the cue, which A4
  measures and whose failure appears as program residual); the readout gain
  `g_R` per template; the readout direction `d̂_R`; `L_R`, the layer of the
  earliest R component in `S_M`; the context residuals `ρ_{frame}` and
  `ρ_{template}`; and the final LayerNorm's own parameters `γ`, `β`, and
  `ε` (the pinned model's `layer_norm_eps`). The readout is the model's exact
  LayerNorm applied to a reconstructed residual — no linearization:

  ```text
  LN(r)  = γ ⊙ ( r − mean(r) ) / sqrt( var(r) + ε ) + β     var = population variance (correction=0); ε = 1e-5
  ĉ(x)   = −u_N · LN( ρ + δ(x) )
  δ(x)   = g_R · n_t(x) · d̂_R + declared direct contributions of E and T outputs
  Δ̂c(w)  = ĉ(frame, w) − ĉ(frame, ref)
  ```

  with `u_N = W_U[:, plural] − W_U[:, singular]` (an unembedding bias, if
  present, is a per-noun constant that cancels in every shift). Because `LN`
  is exact, the only approximations in the program are the mechanism's own:
  the lexicon, the copy gain, the readout gain and direction, and the frozen
  context; their combined error is the program residual.
- **Outputs.** For any (template, cue token, noun pair): the predicted
  cue-induced shift of `c` relative to the singular reference cue of that
  template — always the template's own singular cue, `one` for cardinal and
  coordinated-adjective and `each` for quantifier, regardless of the cue word
  being predicted — `Δ̂c`, and the predicted pair `d̂_full` for any two cue
  tokens. For
  the twelve manifest prompts additionally the predicted per-condition `ĉ(x)`
  using `ρ_{frame}`. For interventions: the predicted shift produced by
  replacing E alone with the E output of another cue token — the E-route
  increment `δ_E(w) = g_R · k_T · (n_c(w) − n_c(ref)) · d̂_R`, read out as
  `Δ̂c = −u_N · [ LN(ρ_{frame} + δ(ref) + δ_E(w)) − LN(ρ_{frame} + δ(ref)) ]` —
  which is the new intervention X4 tests on unseen cue words. The
  intervention replaces exactly the `E_program` components. The runner
  captures their output from the word-substituted prompt's own forward pass
  and asserts that it equals the weight-only `E_program(token)` vector within
  1e-5, so the replacement tensor the runner uses and the object the program
  reasons about are verified to be the same. No such equality is asserted, or
  possible, for a contextual component.
- **What it cannot do by design.** It has no per-frame context for extension
  frames; there it uses `ρ_{template}`, so per-condition values on new frames
  are approximate and only pair-level and shift quantities are predicted
  there. That limitation is a statement about the mechanism's scope and is
  reported as such.

Program floors are part of Tier A (development) and of the X families
(confirmation). The program's residual — measured minus predicted, per prompt
and per noun — is the quantitative statement of what the mechanism does not
explain.

## Preregistered prediction families

Each family has a **floor** fixed here, which cannot be lowered, and, where
stated, a **band** written into the lock. A family passes its floor or fails
it; independently, a family that carries a band lands inside or outside it.
Families marked *primary* determine the outcome; *secondary* families are
reported and feed claim review but cannot change the outcome.

**Bands.** A band exists only where the lock records one; a family without a
band is judged by its floor alone. B2, P1–P9, X1 (mean `d_full` per
template), and X2 carry Tier B bands; X1 and X2 reuse the corresponding
reserve bands because the mechanism claims frame invariance, and that claim is
what they test. X3 and X4 carry per-(word, frame) tolerance intervals derived
from the Tier B program residual: the lock stores the program's predicted
mean shift for every cue word and frame together with `τ = max(0.5 nats,
3 × RMSE_B)`, where `RMSE_B` is the root-mean-square of the program's
per-condition residual `c(x) − ĉ(x)` over the 114 single-token holdout cases.
X3 hits its band when at least 9 of the 12 words have their measured mean
shift inside the interval in at least 5 of their 6 frames; X4 likewise for the
measured E-patch shift. B1, S1–S3, X5, and Q1 have no band. Every floor and
every band is stated with exact counts wherever the denominator is fixed. Families marked
*coordinated-only* apply to the coordinated-adjective template; the rest apply
to every template and, where stated, per rule class pooled across templates
(reserve: simple-suffix 60, consonant-`y` 36, sibilant-`es` 24 cases).

### Circuit axis — reserve nouns on the twelve manifest prompts (readout generalization)

| ID | Family | Intervention | Floor | Primary |
|---|---|---|---|---|
| B1 | Behavior replicates on new nouns | none | primary pairwise accuracy ≥ 103/120 (as defined in the screen) and contrast flip in ≥ 96/120 pairs | precondition |
| B2 | Pair-level behavior predicted | none | `d_full > 0` in ≥ 114/120 pairs; mean `d_full` per template inside its band | primary |
| P1 | Sufficiency of `S_M` | counterfactual replacement of `S_M` at declared positions | `R ≥ 0.70` overall, `≥ 0.60` per template and per rule class | primary |
| P2 | Specificity | 100 random sets of size `|S_M|` drawn per position slot (seed `20260918`) | `R(S_M) ≥ 2 · max(median random recovery, 0.05)` | primary |
| P3 | Isolation / circuit reconstruction | pair-centered neutralization of every non-`S_M` component at `p_t` (and `p_c`); the token embedding stays intact and its direct share is reported under Q1 | `F ≥ 0.50` overall, `≥ 0.40` per template; correct sign retained in both conditions for ≥ 84/120 pairs | primary |
| P4 / P4′ | Cue-position encoding | P4: counterfactual replacement of E only, at `p_c`; P4′: pair-centered neutralization of every non-embedding component at `p_c` | `R ≥ 0.50` (P4) or retained shift ≥ 0.50 (P4′) | primary, coordinated-only |
| P5 | Transport sufficiency and path | (a) counterfactual replacement of the declared T set alone at `p_t`; (b) counterfactual replacement of `r_{L_T}(·, p_c)` with T frozen at `p_t` | under H1: (a) `R ≥ 0.50` for the single T head and (b) blocked fraction ≥ 0.50; under H2: (a) `R ≥ 0.70` for the declared T set and (b) its blocked fraction inside the locked band | primary, coordinated-only |
| P6 | Abstract number variable | cross-cue resample of E at `p_t` (`cardinal` ↔ `quantifier`, same noun) | opposite-number source `R ≥ 0.50`; same-number source `R ≤ 0.25` | primary (cue-final templates) |
| P7 | Cross-frame control | cross-frame resample of `S_M` (same noun, other frame); for E this is a structural no-op because the layer-0 MLP output depends only on the cue token, so the control is informative for T and R | same-cue `R ≤ 0.25`; opposite-cue `R ≥ 0.5 · R(S_M)` | primary |
| P8 | Necessity and self-repair | pair-centered neutralization of T (coordinated) or of E (cue-final) | contrast loss `1 − d_neutralized / d_full ≥ 0.30`; compensation ratio inside its band | primary for loss, secondary for compensation |
| P9 | Chain mediation | E-alone counterfactual patch with T recomputing, then with T frozen (A6 i–ii) | under H1/H2: `m_T ≥ 0.50`, `m_R ≥ 0.50`, `m_R|T ≤ 0.5 · m_R`; under H3: the H3-specific band | primary |
| S1 | Shared number axis | none (captures only) | cosine between template-specific `d_num(L_R, p_t)` estimates ≥ 0.70 | secondary |
| S2 | Rule-class residual | P1 and P3 restricted to spelling-change nouns | reported | secondary |
| S3 | Failures are readout-offset failures | none (captures only) | in the `n_wrong` reserve conditions the model gets wrong, `sign(n_t)` still equals the cue number in at least `ceil(0.8 · n_wrong)` of them | secondary, discriminating |
| Q1 | Residual accounting | derived from P1, P3, B2, and the program | reported, never gated | report |

Because activations on these prompts are identical to development
activations, P1–P9 on the reserve test whether the locked component-level
effects read out correctly through twenty new unembedding row pairs. They are
labeled readout-generalization results in every report.

### Decompilation axis — extension prompts with reserve nouns (prompt-side generalization)

| ID | Family | Prompts | Floor | Primary |
|---|---|---|---|---|
| X1 | Frame invariance of behavior | six new frames × two cues × 20 nouns | `d_full > 0` in ≥ 108/120 pairs; `sign(n_c)` and `sign(n_t)` read from the new prompts equal the cue number in at least 11 of the 12 new prompts; mean `d_full` per template inside the reserve band | primary |
| X2 | Frame invariance of the circuit | P1, P3, P4/P4′, P5, P8, P9 repeated on the new frames | the same floors as the corresponding P family | primary |
| X3 | Cue lexicon predicts unseen cue words (behavior) | twelve new cue words × six original frames × 20 nouns | Spearman correlation between the program's `n_c(w)` and the measured mean shift of `c` relative to the template's singular reference cue (`one` or `each`), pooled over frames and nouns, ≥ 0.70 across the twelve words; for each of the `k` words with `abs(n_c(w)) ≥ 0.5`, sign agreement in at least 5 of its 6 frames and, across those words, in at least `ceil(0.9 · 6k)` of the `6k` (word, frame) combinations; for each word with `abs(n_c(w)) < 0.5`, measured mean absolute shift below half the template's locked full shift in at least 5 of its 6 frames | primary |
| X4 | Program predicts a new intervention for unseen cue words | for each word and original frame, counterfactual replacement of the `E_program` components alone (at `p_c`) in the singular-reference prompt with their output for the word-substituted prompt; the program predicts the shift from `δ_E(w)` through the exact LayerNorm around `ρ_{frame}` as defined above | Spearman ≥ 0.70 between predicted and measured shifts across the twelve words; mean absolute error ≤ 0.25 × the template's locked full shift | primary |
| X5 | Program residual | all extension prompts | mean absolute error of `Δ̂c` and `d̂_full` in nats; fraction of variance explained; reported per template and per cue word | report |

X1–X4 are what distinguish "these components suffice" from "we recovered what
they compute". They are evaluated once, on prompts the project has never run.

Discriminating predictions: the lock must contain at least one prediction
whose band differs between the chosen hypothesis and each rejected one (P5,
P9, and X4 are designed for this; S3 separates "the cue is misread" from "the
number is read correctly but the readout gain does not overcome the noun's
prior"). A reserve or extension result that falls inside a rejected
hypothesis's band and outside the chosen one contests the mechanism even when
P1–P3 pass.

### Outcome rule

Applied mechanically after the single confirmation run, on two axes:

- Circuit axis: `CIRCUIT_PASS` if every primary P/B floor passes, else
  `CIRCUIT_FAIL` naming the families.
- Decompilation axis: `PROGRAM_PASS` if every primary X floor passes and the
  lock is not `PROGRAM_CAPPED`, else `PROGRAM_FAIL` naming the families or
  the cap.

Overall outcome:

- `BEHAVIOR_NOT_REPLICATED` — B1 fails. The mechanism test is inconclusive,
  not falsified; no other family is interpreted as confirmatory.
- `MECHANISM_CONFIRMED` — `CIRCUIT_PASS`, `PROGRAM_PASS`, and every primary
  family that carries a band hits it.
- `MECHANISM_SUPPORTED_MISCALIBRATED` — `CIRCUIT_PASS` and `PROGRAM_PASS`; at
  least one primary band is missed. The missed bands are listed as the
  mechanism's quantitative residual.
- `CIRCUIT_ONLY` — `CIRCUIT_PASS` but `PROGRAM_FAIL`. The components are
  sufficient and specific, but the explicit computation attributed to them is
  wrong or incomplete; the failing X families, or the `PROGRAM_CAPPED` flag,
  name where.
- `MECHANISM_CONTESTED` — both axes pass but at least one discriminating
  prediction lands outside the chosen hypothesis's band and inside a rejected
  hypothesis's band. A result outside both bands is a missed band
  (miscalibration), not a contest.
- `MECHANISM_NOT_SUPPORTED` — `CIRCUIT_FAIL`. The failing families name the
  falsified stage (encoding, transport, readout, isolation, specificity, or
  abstraction).

There are no scientific retries. A software defect discovered after the
confirmation run follows the screening incident rule: incident note,
invalidation of the affected artifacts, committed fix, and a rerun of the
*entire* confirmation only under a new protocol version whose report also
carries the invalidated result.

## Residual accounting and self-repair definitions

- **Unexplained sufficiency residual:** `1 − R(S_M)` overall, per template,
  and per rule class.
- **Unexplained isolation residual:** `1 − F(S_M)` on the same strata.
- **Behavioral residual:** the fraction of pairs whose `d_full` sign or band
  the mechanism mispredicts, and the mean absolute error of the pair-level
  `d_full` prediction in nats, on the reserve and on each extension family.
- **Program residual:** measured minus program-predicted `Δc` per prompt and
  noun, summarized per template, per frame, and per cue word.
- **Direct-effect decomposition:** for each component `j` at `p_t`, the direct
  effect `DE_j(x)` is its output's contribution to `c(x)` through the actual
  final LayerNorm statistics of that run and the complete unembedding; bias
  terms are reported separately; `Σ_j DE_j + biases = c(x)` is verified to
  1e-4 nats on every case.
- **Compensation ratio:** after pair-centered neutralization of a component
  `k`, `Σ_{j ≠ k} (DE_j' − DE_j) / DE_k`, computed on the pair difference.
  Values near 0 mean no repair; values near 1 mean full backup. Conditional
  co-neutralization of `k` with its top compensator reports how much the
  necessity effect grows.

## Preregistration lock

`experiments/005-regular-plural-mechanism/preregistration-lock.json` is a
hard integrity boundary in the style of the Experiment 004 selection lock. It
records:

- manifest path and digest; the extension-set path and digest; the exact case
  IDs of the three splits and a digest of the reserve cases in manifest order;
- pinned model identifier and revision; runtime and seeds;
- the mechanism statement text and its digest; the chosen hypothesis row and
  the discriminating quantities that selected it;
- `S_M` with each component's role and declared positions; the `E_program`
  declaration (which token-local paths it uses) and, when applicable, the
  `PROGRAM_CAPPED` flag with the contextual component that caused it; the
  number-variable
  parameters (`d̂_s`, `μ_s`, `σ_s` for every site) and the program parameters
  (`k_T`, `g_R`, `d̂_R`, `L_R`, `ρ_{frame}`, `ρ_{template}`, `γ`, `β`, `ε`)
  stored as separately named tensor
  artifacts with shape, dtype, and SHA-256 digests;
- the digest of `mechanism_program.py` at lock time, the program's
  development and holdout residuals, `RMSE_B`, and the derived tolerance `τ`;
- every prediction family with its floor, point prediction, band, and the
  holdout numbers it came from; the program's predicted values for every X
  family, computed at lock time without running any extension prompt;
- the Tier A results-artifact digest, the Tier B results-artifact digest, the
  number of calibration passes, and the mechanism version;
- the protocol/code commit at lock time and the lock creation timestamp.

The `confirm` phase verifies schema and content, recomputes the manifest,
extension, program, and artifact digests, checks that the lock file is tracked
and committed, that the working tree is clean, that the current code commit is
the lock commit or a descendant whose diff touches no scientific path, and
that neither the reserve nor any extension prompt has ever been executed. It
has no override flag. It never regenerates the lock.

## Claim and anomaly handling

- No claim is created during Tier A or Tier B.
- At the lock commit, claim `C002` is created at maturity `PROPOSED`,
  disposition `ACTIVE`, origin `confirmatory`, with the mechanism statement as
  its hypothesis and the prediction families as its predictions. Its gates
  cite Tier A and Tier B evidence as exploratory and methodological only.
- After the confirmation run, claim review follows the methodology: the
  evidence can support at most `CAUSAL_EVIDENCE` (Level 3) and
  `FALSIFICATION_TESTED` (Level 4). Level 5 may be claimed only along the axes
  "new lexical items", "new frames of the same three templates", and "new cue
  words in the original frames", each stated separately with where it failed.
  Level 6 gates are reviewed individually (mechanism specified; unseen
  behavior predicted; unseen interventions predicted; reconstruction retained;
  residual quantified) and the claim states which are met;
  `ALGORITHMIC_EXPLANATION` is not asserted merely because all six are checked
  within one behavior. A `CIRCUIT_ONLY` outcome caps the claim below Level 6
  by construction.
- An unexpected measurement (for example a component with a large negative
  number contribution, self-repair that exceeds the removed effect, or a cue
  word whose measured shift opposes its lexicon sign) is recorded under
  `research/anomalies/` with full discovery provenance rather than folded into
  the mechanism silently.

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
data → explicit mechanism statement → a program that runs without the network
→ committed numeric predictions about unseen nouns, unseen prompts, and
interventions at positions and with replacement sources that the
behavior-selecting screen never used → one execution → residual accounting
and self-repair bounds. The mechanism is built on development data,
calibrated once or twice on holdout data, and tested once on reserve and
extension data; only that last execution counts as confirmatory.
"Prospective neural decompilation" is the protected idea; the plural
mechanism is its first instance.

## Interpretation limits

- The behavior is pairwise preference between two noun forms; the mechanism
  does not predict full-vocabulary top-1 output and the experiment does not
  test it.
- All prompt-side computation in the manifest lives in twelve prompts. Reserve
  results are readout-generalization results; prompt-side generalization
  rests on six new frames and twelve new cue words, all with the cue one
  token (cue-final) or two tokens (coordinated) from the prediction position.
- Number variables are centered, scaled projections onto mean-difference
  directions estimated from four cue tokens and twelve prompts; they are
  analysis coordinates, not claimed model features.
- Pair-centered neutralization and resample replacements are justified for
  these matched pairs only; necessity and isolation conclusions are scoped to
  those replacement distributions and do not transfer to other ablation
  schemes.
- Direct-effect decompositions through LayerNorm are exact for the measured
  run but attribute nonlinear interactions to whichever component's output
  changed; they are accounting, not mechanism.
- The mechanism program linearizes the transport and readout stages and
  treats the frame's context residual as unexplained; its residual is the
  measure of that simplification. The final LayerNorm is applied exactly.
- Cue-final templates cannot distinguish "transport" from "local encoding"
  because `p_c = p_t`; transport claims rest on the coordinated-adjective
  template alone.
- Nothing here generalizes beyond the pinned checkpoint, the three templates,
  the frozen frames and cue words, and single-token nouns of three regular
  rule classes.

## Outputs

```text
experiments/005-regular-plural-mechanism/
    README.md
    run.py                       # phases: validate, discover, calibrate, lock, confirm, report
    extension-v1.json            # frozen before discover; tokenizer-only construction
    mechanism_program.py         # the decompiled computation; runs without the network
    preregistration-lock.json    # committed before confirm; absent until then
    evidence/                    # verbatim copies of the reports actually used for claims

outputs/experiment-005/          # generated, gitignored
    discovery-results.json       # every Tier A measurement, every mechanism version
    calibration-results.json     # every Tier B pass
    candidate-lock.json          # written by `lock`, reviewed and moved by hand
    confirmation-results.json    # the single reserve + extension run
    parameters/                  # number-variable and program tensors with digests
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
   prompt-level (rather than case-level) execution with all-noun readout,
   position-indexed capture/patching, pair-centered and resample sources,
   freeze interventions, number-variable and direct-effect analysis, the
   hypothesis rule, prediction-family evaluation, band computation, the
   extension-set builder, and the lock schema;
2. the mechanism program module, kept free of capture/intervention imports;
3. one runner with the six phases above, phase isolation enforced through the
   results state exactly as in the screen (a `confirm` without a committed
   lock, a second `confirm`, a third `calibrate`, or any phase without the
   committed extension digest refuses to run);
4. focused offline tests for position normalization, structural no-op
   assertions, exact decomposition identities, pair-centered and resample
   construction, freeze correctness, number-variable centering, hypothesis-rule
   determinism, chain-mediation arithmetic, program independence from the
   runner, band arithmetic, extension-set immutability, lock validation,
   reserve and extension non-execution, and phase refusal;
5. one Markdown report renderer.

It must not introduce sparse autoencoders, automated circuit search beyond the
enumerated 54-component universe, gradient attribution, path patching through
attention keys/queries, prompts beyond the manifest and the frozen extension
set, Pythia-160M, a plugin system, a database, or a dashboard. The
instrumentation modules are extended only if a narrow defect is found.

## Future work explicitly deferred

Cue-to-noun distances beyond two tokens, nouns outside the reserve, cue words
beyond the frozen twelve, other templates, other checkpoints, and other model
families are each a separate design with their own frozen prompts. None may
be added to Experiment 005 after the `discover` phase has run.

## Approval and stopping condition

This design is complete when it has been reviewed against the research
methodology and the screening evidence, amended where the review requires,
and explicitly approved by the user. Only then is an implementation plan
written and the experiment directory created. The extension set is frozen
before `discover`; Tier C may begin only after the preregistration lock is
committed. The experiment stops after the single confirmation run, the
outcome rule, the claim review, and the evidence copies, with every rejected
mechanism version, every missed band, and every program residual retained in
the record.
