# Experiment 020: Readout Decompilation — Does the Decoded Mechanism Predict the Model's Own Singular-versus-Plural Logit Contrast for Unseen Cues, Unseen Frames and Unseen Nouns?

**Date:** 2026-09-22

**Closed (2026-09-22, after revision 3):** exploration completed at `ce3766b`; confirmation intentionally not run
because the corrected block-5 parallel-residual implementation materially changed the exposed calibration on which
the floors below had been based. The floors are not changed retrospectively and no Y1/Y2/Y3 label is assigned; the
confirmation set was never executed. See the experiment's
[README](../../../experiments/020-readout-decompilation/README.md) and `closure.json`.

**Status:** Revision 2 — revision 1 approved subject to five specification items, which this revision makes and
which are textual only (no new prompt was executed for it): the exposed-noun accounting and the freshness
certification of the 24 fresh nouns, the label-bearing noun populations of Y1/Y2/Y3, the separation of scientific
invalidity from incidents, frozen numerical tolerances for every incident-triggering check, and an enumerated stage-1
prompt manifest. Revision 2 also corrects one error of revision 1: its fresh-noun list contained `tunnel`, which is an
exposed noun (the check that produced the list did not apply the exposure filter); it is replaced by the next eligible
entry, `pillar`. Nothing else changes: the Level 0 program, the floors, the comparator standing, the cue and frame
lists and the interpretation limits stand as in revision 1. No Experiment 020 directory, confirmation set, lock or model run exists. The
design pass behind it ran on the **exposed** pool only (90 exposed frames × 32 exposed cues = 2850 pairs × 79 scorable
exposed nouns,
pinned Pythia-70M, scratch scripts outside the repository, nothing committed); every exposed number quoted below comes
from that pass and is calibration, never evidence. Experiments 005–019 are closed and are not amended by this
document; Experiment 019's closure stands exactly as recorded (`ROUTING_PREDICTED_TOKENS |
ROUTING_PREDICTED_FRAMES_CONDITIONAL | OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT | TEMPLATE_FAMILY_INSUFFICIENT |
MEMBERSHIP_PREDICTED`, never rerun).

**Kind:** Prospective, zero-parameter, mechanism (readout). Experiments 009–019 decoded the cue-to-transport chain end
to end as a composed weight-only program whose every head input is predicted rather than measured, and closed at the
head's output `ΔT` and the decoded read `c_L` at the transport position. Everything after layer 3 has never been
decoded, and the claim record says so: every design since 009 states that nothing in it concerns behaviour, and claim
[C002](../../../research/claims/C002-count-cued-noun-number-circuit.md) has stood at `LOCALIZED` since Experiment 005,
whose single confirmation failed its behavioural precondition. This experiment asks whether the decoded mechanism
predicts the quantity the whole chain exists to explain: the model's own singular-versus-plural logit contrast at the
prediction position, for cue words never run, in frames never run, and for nouns never used to build anything.

## Purpose and question

**Question.** Given the change entering the post-`L03` readout path as the already-decoded chain *predicts* it —
never as it is measured — can the resulting change of the singular-versus-plural logit contrast be predicted
prospectively, for unseen cues (Y1), for unseen frames conditional on each frame's own reference state (Y2), and for
unseen nouns (Y3)?

**Why now.** The design pass establishes three things that make the question decidable and non-incremental.

1. **The downstream stage is not passive.** Fixed-normalization projection of `Δc` onto its additive parts over 8820
   exposed pairs × 79 nouns: the already-decoded transport head `L03.H04` contributes **0.154** of the contrast
   change directly, the whole decoded upstream residual `Δx3` contributes **0.192**, while `L04.MLP` contributes
   **0.218** and `L05.MLP` **0.392** — about **61 %** of the behavioural signal is *created* by the two readout MLPs
   at the prediction position, and `L04` attention contributes **−0.000**. Predicting `Δc` is therefore not a
   rescaling of `ΔT`.
2. **The composed prospective program already works on exposed data.** With the committed Experiment 011, 012 and 017
   locks supplying `Δ̂x3` and the reduced readout program below supplying the rest, the end-to-end predicted contrast
   reaches flattened `R²` **0.848** over 2850 exposed pairs × 79 nouns (token means 0.942, MAE 0.300 nats), against a
   downstream ceiling of **0.970** when the same readout program is fed the *measured* `Δx3`. The error decomposes as
   0.030 of the variance from the downstream approximation and **0.114 from inherited upstream error** — three
   quarters of what remains is Experiments 011–017's error, not this experiment's.
3. **The readout sees a small subspace.** The predicted `Δx3` is only about half right in norm (median relative error
   0.531, median cosine 0.852) and the contrast prediction is nevertheless 0.85, because only 7–15 % of `ΔLN_final`'s
   energy lies on the direction the noun contrasts read. This is a statement about the mechanism, and it is what makes
   the experiment feasible at all.

**Two long-standing records this experiment bears on, both reported descriptively and neither label-bearing.**

- Experiment 005's reserve-noun failure (44 of 45 wrong conditions were plural conditions). On the exposed pool the
  plural condition's contrast `c_pl` is positive in 28 % of frame × noun conditions, 22 % of conditions lie within
  ±0.5 nats of zero, and the per-noun failure rate correlates +0.47 with the reference baseline and −0.44 with the
  noun's readout sensitivity. The failures are a margin phenomenon of the readout, not a failure of transport.
- Experiment 005's S1 failure (the three templates' readout-input number axes have minimum pairwise cosine 0.292).
  On the exposed pool the per-template mean change directions are indeed different before the readout (0.305–0.476),
  are progressively aligned by blocks 4 and 5 (0.469–0.539, then 0.383–0.680), and become almost identical **in the
  noun-contrast image**: cosines of the per-template mean `Δc` vectors are 0.982–0.993. The low internal cosine was
  never a contradiction; the readout annihilates the disagreeing component.

## The quantities under test (frozen definitions)

Positions: `p_c` the cue position, `p_t` the prediction position (`p_t = p_c` in the cue-final templates, `p_c + 1`
in the coordinated-adjective template). All vectors are float64. `x_ℓ` denotes `RESID_PRE.Lℓ` and `h6` denotes
`RESID_POST.L5`, always at `p_t` unless a key position is named.

**The measured target.** For a frame `f`, a cue `w` and a single-token noun `n`,

```
c(prompt, n) = log P(sg(n)) − log P(pl(n))            at p_t
Δc(f, w, n)  = c(frame f with cue w, n) − c(frame f with its reference cue, n)
```

**The exact readout identity (an identity, never a floor).** With `Δw(n) = W_U[:, sg(n)] − W_U[:, pl(n)]`,

```
Δc(f, w, n) = ⟨ LN_final(h6^w) − LN_final(h6^ref) , Δw(n) ⟩
```

— the softmax normalizer and the unembedding bias difference cancel exactly. Measured maximum absolute deviation on
the exposed pool: **5.0e-3** (float32 captures; the chain itself runs in float64). Two further identities:
`Δh6 = Δx3 + Σ(heads and MLPs of blocks 3–5)` (1.4e-6) and `logits = LN_final(h6)·W_U + b_U` (2.9e-3, float32).

**The inherited upstream input.** `Δ̂x3(p_c)` and, in the coordinated template, `Δ̂x3(p_t)`, from the committed
Experiment 017 chain (`HeadChainModel.upstream`, variant `LEVEL0`) built from the Experiment 011 lock's axes and read
weight, the Experiment 012 lock's template bases and the Experiment 017 lock's layer-3 bases. Nothing in this
experiment refits any of them.

**The Experiment 020 readout program (Level 0, the hypothesis).** A *downstream readout program with a small
frozen-row second-transport term* — not a purely position-local program, because of the layer-5 value path from
`p_c`. From `Δ̂x3` at the changed positions and the frame's reference state:

1. **Block 3, at `p_c` and (when different) `p_t`.** All heads by the exact layer program on `x3_ref + Δ̂x3`
   (LayerNorm, projections, partial rotary rotation, scaled dot products, softmax, `W_O`) — this is Experiment 017's
   own head treatment extended to the other heads of the layer — plus block 3's MLP at the frame's own reference
   operating point: `Δ̂m3(p) = MLP₃(x3_ref(p) + Δ̂x3(p)) − MLP₃(x3_ref(p))`.
2. **Block 4, at the same positions.** Attention **frozen at the reference** (exposed share −0.000; adding a
   frozen-row term changes the composed `R²` by +0.0002); MLP at the frame's own reference operating point.
3. **Block 5, at `p_t`.** Heads as the frame's **reference attention rows** times the predicted value changes at every
   key position — `Σ_k A_ref[h, k] · (v(LN(x5_ref(k) + Δ̂h4(k))) − v(LN(x5_ref(k)))) · W_O[h]`, where `Δ̂h4` is this
   program's own predicted residual change at that position and is zero at every position other than `p_c` and `p_t`
   (causally unchanged); MLP at the frame's own reference operating point.
4. **`LN_final` exactly** on `h6_ref + Δ̂h6`, and the noun read `⟨·, Δw(n)⟩`.

**The exact chain (Level 1, an identity).** The same composition with the *measured* `Δx3` and with every head and
MLP recomputed exactly from the frame's own state reproduces the measured `Δh6` and hence `Δc`; it is recorded with
its maximum deviation and enters no floor.

**Provenance, demonstrated rather than asserted.** Every quantity of the Level 0 program is one of: the frame's
**reference** prompt capture (`x1…x5` at every position ≤ `p_t`, `h6`, the layer-4 and layer-5 reference attention rows
at `p_t`, the reference contrasts), a committed lock (011, 012, 017), a checkpoint weight, or the cue's token id.
In the design pass, 60 stored end-to-end predictions were recomputed in a process that never executed the cue prompt
at all: maximum absolute difference **4.5e-07** (float64 reassociation only). The implementation repeats that check as
an invariant.

## Inherited fixed elements

- Model: pinned `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, CPU float32,
  TransformerBridge without compatibility mode, `use_attn_result=True`; seeds and runtime as Experiments 011–019.
- The Experiment 011 lock (axes `T`, `R0`, read weight, `σ_T`), the Experiment 012 lock (template bases), the
  Experiment 017 lock (layer-3 bases) — read verbatim, digests recorded, never refitted.
- The exposed pool: the **279** exposed cue tokens (Experiment 019's 255 plus its 24 confirmed) and the **108**
  exposed frames (its 90 plus its 18 confirmed), with the **80 manifest/extension nouns, of which 79 are scorable**.

**Exposed-noun accounting (frozen).** The manifest and extension contribute 80 noun entries. One of them, `peach`
(`sibilant-es`, holdout split), is **not single token**: its singular tokenizes as two tokens (ids 759, 607) and its
plural as (759, 3844). The frozen contrast `c(prompt, n) = log P(sg(n)) − log P(pl(n))` is defined on single-token
forms only — `pm.contrasts` skips every non-single-token noun, and `Δw(n) = W_U[:, sg(n)] − W_U[:, pl(n)]` would be
the zero vector for `peach` because both forms share their first token — so `peach` is **non-scorable by the frozen
rule**, not excluded by choice. Every statistic in this document that says *79* is over the 79 scorable exposed
nouns; wherever the diagnostic tables of the design pass carried 80 columns, `peach`'s column was identically zero and
was dropped by the same rule. The document uses **“80 exposed nouns, 79 scorable”** throughout, and the
implementation asserts both counts.

## Exposed pool (calibration record only; nothing is fitted)

Tier A (`explore`) runs once over the exposed pool and records: the reference capture of every exposed frame; the
identities above; the Level 0 program's predictions and the measured contrasts for every exposed cue × exposed frame ×
exposed noun; the downstream ceiling (the same program fed the measured `Δx3`); the comparators; the descriptive
tables named below. No floor is set from it, and no object in the lock is fitted to it except the descriptive noun
factor of the rank-1 comparator, which bears no label.

## Confirmation set (frozen by tokenizer rules before any Experiment 020 model output)

- **Fresh cue tokens (24)**, single token with a leading space, disjoint from the 279 exposed tokens and from every
  noun form, taken as the first eligible entries of the inherited Experiment 017/019 candidate lists, whose remaining
  entries were tokenizer-checked on 2026-09-22. Quotas 6 / 6 / 6 / 6; the classes are for coverage and reporting only
  and **no class carries an expectation**:
  - `determiner-like` (8 remaining, quota 6): `recent`, `current`, `original`, `typical`, `ordinary`, `identical`,
    `alternate`, `random` — expected the first six.
  - `quantity` (14 remaining, quota 6): `average`, `excessive`, `exhaustive`, `comprehensive`, `thorough`, `sweeping`,
    `bulk`, `spare`, `dense`, `lengthy`, `lots`, `loads`, `tons`, `scores` — expected the first six.
  - `possessive-or-pronoun` (11 remaining, quota 6): `anything`, `something`, `everything`, `nothing`, `who`, `thee`,
    `thou`, `yourself`, `themselves`, `ones`, `others` — expected the first six.
  - `adjective` (29 remaining, quota 6): `purple`, `yellow`, `hidden`, `hard`, `dirty`, `rare`, `square`, `wild`,
    `brave`, `calm`, … — expected the first six.
- **Fresh frames (18, six per template)**, literal, tokenizer-checked on 2026-09-22 to build with a one-token cue and
  to differ from every exposed frame text: cardinal `The workshop repairs {cue}`, `The archive stores {cue}`,
  `The orchard grows {cue}`, `The foundry casts {cue}`, `The kennel keeps {cue}`, `The cellar holds {cue}`;
  quantifier `The journal lists {cue}`, `The roster names {cue}`, `The manual covers {cue}`, `The ledger records
  {cue}`, `The brochure shows {cue}`, `The harbor hosts {cue}`; coordinated-adjective `Nils and Rosa painted {cue}
  pale`, `Emil and Tessa folded {cue} neat`, `Arne and Lena polished {cue} smooth`, `Sven and Mira stacked {cue}
  high`, `Bodil and Timo carved {cue} deep`, `Reta and Olav wrapped {cue} tight` (`p_c` 3–6 in the cue-final templates and 5–6 in the coordinated one, `p_t = p_c` or
  `p_c + 1`; the freeze re-checks each text and drops any that does not tokenize as declared, recording the
  replacement rule: the next entry of the same template's list).
- **Fresh nouns (24, eight per regular-plural rule class)**, both forms single token with a leading space, disjoint
  from the 80 exposed nouns and from every cue token, tokenizer-checked on 2026-09-22 with their ids:
  - `simple-suffix`: `brick`/`bricks` (20098/40154), `candle`/`candles` (28725/36555), `statue`/`statues`
    (23957/38490), `barrel`/`barrels` (15474/33545), `curtain`/`curtains` (31261/36708), `magnet`/`magnets`
    (10973/43733), `puzzle`/`puzzles` (25351/43884), `pillar`/`pillars` (43035/40652).
  - `sibilant-es`: `switch`/`switches` (5234/20994), `branch`/`branches` (7789/12998), `ash`/`ashes` (15898/39187),
    `sketch`/`sketches` (23211/46159), `batch`/`batches` (14604/39657), `flash`/`flashes` (10299/41483),
    `arch`/`arches` (4222/50042), `crash`/`crashes` (13035/29212).
  - `consonant-y`: `colony`/`colonies` (17562/19665), `gallery`/`galleries` (19718/40871), `cavity`/`cavities`
    (13369/38674), `battery`/`batteries` (9378/19978), `category`/`categories` (7140/9050), `artery`/`arteries`
    (12235/24908), `boundary`/`boundaries` (7548/13674), `anomaly`/`anomalies` (30207/31101).
  `tunnel`/`tunnels` was named in revision 1 and is withdrawn: `tunnel` is an exposed noun. The list is the first
  eligible entries of the literal candidate lists, eligibility being: both forms single token with a leading space;
  the singular not an exposed noun key; neither form among the exposed nouns' token strings; neither form among the
  279 exposed cue tokens.

  Fresh nouns are **output alternatives**, never inserted into a prompt: they cost no forward pass and enter only
  through their frozen weight vectors `Δw(n)`.

  **Freshness certification (frozen, and the definition of freshness for nouns).** Because a noun needs no prompt, “no
  prompt was run” would not establish freshness. The certification is therefore about *inspection*: none of the 24
  nouns above appeared in any exposed diagnostic table, and no model-output-derived statistic of any of them — a
  contrast, a `Δc`, an `R²`, a slope, a bias, a rank-1 loading, a failure rate or any other — was computed or
  inspected before the list was frozen. The design pass computed noun statistics for the 79 scorable **exposed**
  nouns only; the 24 fresh nouns were selected afterwards by the tokenizer-only rule stated above, from literal
  candidate lists, with no model loaded. `explore` is forbidden to compute any quantity of a fresh noun, and the
  implementation asserts that the fresh-noun ids appear in no exploration record.
- **The prompt-key manifest** is part of the frozen set and lists every prompt the experiment may ever execute, in
  three mechanically checkable classes. Writing `ref(τ)` for template `τ`'s frozen singular reference cue and
  `pl(τ)` for its frozen plural manifest cue:
  - **S1-REF** — one per fresh frame: the fresh frame with `ref(τ)` in its cue slot (18 keys). Stage 1 only.
  - **S1-VALIDITY** — one per fresh frame: the fresh frame with `pl(τ)` in its cue slot (18 keys); together with its
    S1-REF key it forms the frozen validity cue pair that decides the frame's validity. Stage 1 only.
  - **S2-TARGET** — the fresh-cue prompts, in two disjoint blocks: each of the 24 fresh cues in each of the 108
    exposed frames (2592 keys, the Y1 block) and in each of the 18 fresh frames (432 keys, the Y2 block). Stage 2
    only, and in the Y2 block only for frames that stage 1 marked valid.
  No S2-TARGET key may appear in the execution ledger before the barrier, and no key outside these three classes may
  be executed by `confirm` at all; both are asserted, and a violation is an incident.
- Two fresh prompt sets, both executed only by `confirm`: **Y1** the 24 fresh cues in the 108 exposed frames (2592
  pairs) and **Y2** the 24 fresh cues in the 18 fresh frames (432 pairs). Every pair is read on all 103 scorable
  nouns (79 scorable exposed + 24 fresh) in the same forward pass, at no extra prompt cost.

**Label-bearing populations (frozen).** The noun population of each outcome is fixed here and the implementation
asserts it; computing all 103 readouts in one pass is allowed, letting a fresh noun enter a Y1 or Y2 statistic is not:

| outcome | cues | frames | nouns | pairs × nouns |
|---|---|---|---|---|
| **Y1** | the 24 fresh cues | the 108 exposed frames | the **79 scorable exposed** nouns only | 2592 × 79 |
| **Y2** | the 24 fresh cues | the valid fresh frames (≤ 18) | the **79 scorable exposed** nouns only | ≤ 432 × 79 |
| **Y3** | the 24 fresh cues | the 108 exposed frames (the Y1 population) | the **24 fresh** nouns only | 2592 × 24 |
| joint diagnostic (no threshold) | the 24 fresh cues | the valid fresh frames | the 24 fresh nouns | ≤ 432 × 24 |

No fresh noun contributes to any Y1 or Y2 statistic — not to a token mean, a pair-level mean, an MAE, a frame mean, a
split or a guard count. No exposed noun contributes to a Y3 statistic.
- The set is committed as `confirmation-v1.json` before `explore`; `confirm` refuses if any fresh prompt key appears
  in the execution ledger earlier.

## The two-stage confirmation procedure (frozen)

`confirm` runs once, straight through, and never twice (the runner refuses a second attempt in this protocol version).

1. **Artifact validation, before any fresh prompt.** The lock's digests, floors, frozen constants, the confirmation
   set's digest and the prompt-key manifest; the selectors of the inherited locks; recomputation of **every** locked
   prediction row for the Y1 population from the locked reference states and the weights, refusing on any difference;
   the provenance invariant (the prediction path executes no cue prompt).
2. **Stage 1 — the fresh frames' reference states only.** For each of the 18 fresh frames: its reference prompt and
   its template's cue-pair prompts; the validity rule (the same head-informative and cue-effect rule as Experiments
   017–019); the frame's reference captures (`x1…x5` at every position ≤ `p_t`, `h6`, the layer-4/5 reference rows,
   the reference contrasts of all 103 scorable nouns); the frame-conditional prediction table for every fresh cue ×
   scorable noun.
   The table, the states and the per-frame reference contrasts are serialized and **digested**. **Zero** of the 432
   Y2 target pairs and zero of the 2592 Y1 pairs run before the barrier.
3. **The barrier.** The digested stage-1 record is written to disk, re-read from disk, its digest re-verified, and the
   execution ledger is asserted to contain no target prompt key. Only then does stage 2 begin.
4. **Stage 2 — the fresh cues.** The S2-TARGET keys are executed: the 2592 Y1 prompts and the Y2 prompts of the
   valid fresh frames (≤ 432); each prompt yields the contrasts of all 103 scorable nouns at `p_t` in one pass; every
   recomputed prediction is compared with its locked (Y1) or digested (Y2) row at the frozen tolerance; the scoring
   below then runs on the label-bearing populations fixed above.

## Floors and outcome (frozen)

Generalization units: the **cue** and the **frame** for Y1 and Y2, the **noun** for Y3. Flattened `pairs × nouns` `R²`
is reported but is never the only guard: the 79 or 24 noun contrasts of one pair are readouts of one hidden-state
change, not independent samples.

**Y1 — unseen cues in the exposed frames, read on the 79 scorable exposed nouns (2592 × 79; the strict token
boundary).**

| condition | floor | exposed value |
|---|---|---|
| token-mean `R²` (24 cues, each averaged over frames × nouns) | ≥ **0.80** | 0.942 |
| pair-level noun-mean `R²` (one number per cue × frame) | ≥ **0.65** | 0.858 |
| per-cue MAE ≤ **1.5 nats** in at least **80 %** of the scored cues | ≥ 0.80 of cues | 32/32 ≤ 1.2 (median 0.502, 90th 0.895, max 1.024) |
| pooled MAE | ≤ **1.0 nat** | 0.565 |

`CONTRAST_PREDICTED_TOKENS` when all four hold, else `CONTRAST_NOT_PREDICTED_TOKENS`; `PRECONDITION_FAILED_TOKENS`
only when the evaluability conditions fail (fewer than 16 scored cues, or fewer than 60 valid exposed frames).

**Y2 — unseen cues in the valid fresh frames, read on the 79 scorable exposed nouns (≤ 432 × 79; conditional on each
fresh frame's stage-1 reference state).**

| condition | floor | exposed value |
|---|---|---|
| frame-mean `R²` (one number per frame) | ≥ **0.55** | 0.702 |
| per-frame `R²` ≥ 0.40 in at least **75 %** of the valid fresh frames | ≥ 0.75 of frames | 88/90 exposed frames (min 0.279; the three below 0.5 are coordinated) |
| cue-final split `R²` (cardinal + quantifier) | ≥ **0.40** | 0.917 (cardinal 0.924, quantifier 0.909) |
| coordinated split `R²` | ≥ **0.40** | 0.702 |

`CONTRAST_PREDICTED_FRAMES_CONDITIONAL`, else `CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL`;
`PRECONDITION_FAILED_FRAMES` when fewer than 12 fresh frames are valid or fewer than 4 of them are coordinated.

**Y3 — the 24 fresh nouns, evaluated on the Y1 pair population (the 24 fresh cues × the 108 exposed frames; no
exposed noun enters).**

| condition | floor | exposed value (the 79 scorable exposed nouns, none of which is in the program) |
|---|---|---|
| median per-fresh-noun `R²` | ≥ **0.70** | 0.829 (min 0.721) |
| per-noun `R²` ≥ 0.55 in at least **90 %** of the scorable fresh nouns | ≥ 0.90 | 79/79 |
| per-noun slope ∈ [0.75, 1.15] in at least **90 %** | ≥ 0.90 | 79/79 (median 0.917) |
| per-noun \|bias\| ≤ 0.8 nats in at least **90 %** | ≥ 0.90 | 79/79 (mean +0.226, max 0.434) |

`NOUN_READOUT_FIXED` when all four hold, else **`NOUN_READOUT_NOT_ESTABLISHED`** — the negative states that *this*
fixed-unembedding predictor was not established on this sample; it does not assert that the model implements a
noun-specific downstream mechanism, since a shortfall may equally reflect components missing from the predicted
`ΔLN_final` along those output directions. The minimum per-noun `R²` is reported and bears no gate.
`PRECONDITION_FAILED_NOUNS` when fewer than 18 fresh nouns are scorable (a noun is scorable when both of its forms
are single tokens and its measured `Δc` has non-zero variance over the Y1 population). A non-scorable noun is a
scientific evaluability fact, not an incident.

**Outcome label** = `Y1 | Y2 | Y3`.

**Joint diagnostic, no threshold.** The 24 fresh nouns on the Y2 population — fresh cue × fresh frame × fresh noun,
the only statistic of this experiment in which all three dimensions are simultaneously unseen — are computed and
reported in full. They are not calibrated and carry no label; if they hold, that is the strongest
descriptive statement the experiment can make.

## Comparators and their standing

| object | standing | exposed value |
|---|---|---|
| `ΔT` alone with one fitted noun vector | **exposed-disfavoured baseline** | `R²` 0.225 pooled; per template −0.175 / 0.442 / 0.359 under a pooled fit |
| readout MLPs at the Experiment 012 template bases instead of the frame's operating point | **exposed-disfavoured operating-point comparator** | 0.784 at the ceiling (coordinated 0.740) |
| the same program without the layer-5 head term | **nested simplification comparator** | end-to-end 0.809 against 0.848; coordinated 0.592 against 0.702; frame means 0.427 against 0.702 |
| rank-1 noun factorization `Δĉ = ŝ(pair)·β(n)`, `β(n) = ⟨d_noun, Δw(n)⟩`, `d_noun` frozen from exposed data | **descriptive comparator** | table rank-1 share 0.956; `β` fits the measured noun vector at `R²` 1.000; fresh-noun `R²` 0.687 against 0.829 for the direct readout |

All four are recomputed on the fresh sets and reported beside the primary program. The layer-5 head term is
**retained** in the primary program by the pre-declared rule (removing it costs 0.040 pooled, above the 0.02 bound,
and 0.110 on the coordinated family, above the 0.05 bound); the program is therefore described as a downstream
readout program with a small frozen-row second-transport term, never as purely position-local.

## Incidents versus preconditions

Following Experiment 019 exactly, with the two categories kept apart.

**Scientific invalidity is not an incident.** A fresh frame that fails the frozen validity criterion (the
head-informative rule and the cue-effect rule of Experiments 017–019, evaluated at stage 1 from its reference and
validity cue-pair prompts) is simply an **invalid frame**: it is recorded with its measured values, its fresh-cue
target prompts are never executed, and it counts against the `PRECONDITION_FAILED_FRAMES` rule (fewer than 12 valid
fresh frames, or fewer than 4 valid coordinated ones). The same holds for the other evaluability rules: a cue that
cannot be scored (fewer than the required valid frames) is an **unscored cue** feeding `PRECONDITION_FAILED_TOKENS`
(fewer than 16 scored cues, or fewer than 60 valid exposed frames), and a noun whose measured `Δc` has zero variance
over its population, or which is non-single-token, is a **non-scorable noun** feeding `PRECONDITION_FAILED_NOUNS`
(fewer than 18 scorable fresh nouns). None of these is an incident, none stops the phase, and each is reported with
the values that produced it.

**Frozen tolerances (an identity or reproduction check fails, and the phase stops, when its quantity exceeds the
tolerance in this table).** Every tolerance is fixed here, before any fresh data exists; the observed exposed value is
given only to show the margin. Captured activations are float32 and every chain computation is float64, which is why
the capture-facing tolerances are looser than the arithmetic ones.

| check | quantity | frozen tolerance | observed on the exposed pool |
|---|---|---|---|
| readout identity | max abs difference, per noun, between `⟨ΔLN_final, Δw(n)⟩` and the measured log-probability contrast difference | **2e-2 nats** | 5.0e-3 |
| logit reconstruction | max abs difference between `LN_final(h6)·W_U + b_U` and the captured logits | **2e-2** | 2.9e-3 |
| additive residual identity | max abs difference between `Δh6` and `Δx3 + Σ(heads and MLPs of blocks 3–5)` | **1e-4** | 1.4e-6 |
| Level 1 exact downstream chain | the norm-normalized residual error `E₁` defined below | **`E₁` ≤ 7e-3** (amended from 1e-3; see the amendment note below) | the 2026-09-22 explore incident recorded a full-pool maximum of 4.593e-3; the diagnosed float32 attention-score floor is 3e-4 to 1.6e-3 in the diagnostic subsets, and the conservative envelope is 6.77e-3 |
| inherited Experiment 017 reproduction | max abs difference between this runner's `Δ̂x3` and the Experiment 017 chain's own output for the same inputs | **1e-6** | 4.5e-7 for a full recomputation in a separate process (float64 reassociation) |
| locked Y1 prediction reproduction | max abs difference between the recomputed Y1 prediction rows and the locked rows, both canonical-JSON rounded | **0.0 (exact)** | Experiments 013–019 reproduce their locked tables at 0.0 |
| stage-1 Y2 table reproduction | the re-read stage-1 digest against the written one, and the recomputed stage-2 predictions against the digested rows | **digest equality; 0.0 (exact) on the rows** | Experiments 017–019 reproduce their stage-1 tables at 0.0 |
| provenance invariant | max abs difference between a prediction computed with the cue prompt available and one computed in a process that never executes it | **1e-6** | 4.5e-7 |

**The Level 1 error `E₁` (frozen formula).** Componentwise ratios are never used: a residual component near zero
would make them explode and could stop the experiment irreversibly for a numerical artefact. With `Δh6^exact` the
final residual change recomputed by the exact chain from the **measured** `Δx3`, and `Δh6^meas` the measured one,
both float64 vectors of length `d_model` at `p_t`:

```
E₁(pair) = ‖Δh6^exact − Δh6^meas‖_∞ / max( ‖Δh6^meas‖_∞ , 1e-12 )
```

`‖·‖_∞` is the maximum absolute component. The check is `max over the evaluated pairs of E₁ ≤ 7e-3` (revision 3; the
formula itself is unchanged from revision 2, only the threshold); the maximum,
the median and the argmax pair are recorded. The same normalization — maximum absolute error over the maximum
absolute value of the measured quantity, floored at 1e-12 — defines every other *relative* tolerance in the table
above; the tolerances stated as absolute (the readout identity, the logit reconstruction, the additive residual
identity, the reproduction checks and the provenance invariant) are plain maximum absolute differences and use no
normalization.

A value above its tolerance is an incident. A value below it is recorded in the results state and enters no floor.

**Incidents** are implementation or protocol failures only. They stop the phase, are recorded with their commit,
permit no re-run in this protocol version and are never an outcome label: a capture failure or a non-finite value;
failure of any frozen identity or reproduction check beyond its tolerance below; a provenance violation (any cue
prompt reachable from the prediction path, any fresh prompt key in the execution ledger before its phase, any
fresh-noun quantity in an exploration record); a digest or lock mismatch; an unexpected prompt execution; or any
implementation invariant of the runner.

## Interpretation limits

- **Y2 is conditional on the fresh frame's own stage-1 reference state.** Experiment 020 predicts the logit contrast
  *given that reference state*; it does not predict the frame state from text. Nothing here predicts a frame's
  reference state, its operating points, its attention rows or its baseline contrasts from the frame's words.
- The claim is about the contrast `Δc` at `p_t` for single-token regular-plural noun pairs under the frozen
  templates; it is not about top-1 behaviour, about absolute contrasts, or about any other output dimension.
- Y3 concerns whether the fixed unembedding difference suffices as the noun-side object of the readout. It is not a
  claim that no noun-specific downstream structure exists.
- The upstream input is Experiment 017's prediction; three quarters of the exposed end-to-end error is inherited from
  it. A negative Y1 or Y2 therefore does not by itself falsify the downstream program, and the downstream ceiling
  (the same program on the measured `Δx3`) is reported on both fresh sets to separate the two.
- Exposed values quoted in this document are calibration on cue-in-sample data and establish nothing.
- One checkpoint, three templates, four cue classes, three noun rule classes, 18 new frames.

## Minimal implementation boundary

`src/neural_decompiler/readout_decompilation.py`: the readout program (blocks 3–5 at the changed positions with the
frame's reference rows, operating points and `LN_final`), the noun read, the identities, the comparators, the pool and
confirmation builders, the scoring with the floors above, and the prediction table; `experiments/020-readout-decompilation/run.py`
with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm` (two stages), `report`; tests as tiers A–D
of the current test architecture (`--tier B` while developing, `--tier C` before `explore`, `lock` and `confirm`), with
`CURRENT_EXPERIMENT = "020"` in `tests/conftest.py`.

## Approval and stopping condition

Design first. Order after approval: plan → code and tests → tokenizer-only confirmation freeze and commit →
independent implementation review → `explore` once → `lock` → the lock commit → the reviewer's sign-off → `confirm`
once → report. Before stage 1 the following are frozen and committed: the 24 cues, the 18 frames, the 24 nouns, the
validity and exclusion rules, the prompt-key manifest, every threshold, the seeds, and this document's outcome
definitions. No token, frame, noun, table or prediction may be changed after the lock; no fresh prompt runs before
`confirm`, and no fresh cue prompt runs before its frame's stage-1 table is digested and re-read from disk.

## Implementation note (2026-09-22, before any Experiment 020 model run): a calibration bug found by the Level 1 identity

The scratch design pass that produced every exposed number in this document fed block 5's MLP the **post-attention**
residual. Pythia is a GPT-NeoX parallel-residual model: each block's MLP reads the block's *input*. The production
implementation (`src/neural_decompiler/readout_decompilation.py`) uses the correct parallel residual, and the Level 1
identity is what caught the difference — `E₁` was 1.5e-2 on the fake with the sequential input and is below 1e-3 with
the correct one; a regression test keeps the sequential variant and asserts it is worse.

Consequences, stated exactly:

- **No frozen floor, population, tolerance, label or list changes.** The floors stay as calibrated.
- **The exposed numbers in this document were computed with the wrong block-5 MLP input.** Their direction of error
  is *unknown*: it is not claimed that they were pessimistic, optimistic or unchanged. They remain what they are — a
  scratch calibration that establishes nothing.
- **No ad-hoc diagnostic is rerun.** The single official `explore` will produce the corrected exposed values with the
  production implementation, and they will be reported as they come out, whether they are better or worse.

## Amendment (2026-09-22, after the recorded `explore` incident): the `E₁` tolerance, and nothing else

`explore` ran once, at `abf3df9` (run `9d0c1a998732b9c1`, cpu/float32/4 threads, 2 h 56 m). It captured all 108 exposed
reference states and measured all 30,024 exposed pairs, then stopped at
`level1 failed: 4.593e-03 above the frozen tolerance 1.000e-03`. No `lock`, no `confirm`, no confirmation prompt and no
fresh-noun quantity was touched, so the fresh test is intact.

A committed diagnostic (`diagnose`, not a phase: it never opens the results state and enforces nothing; records
`outputs/experiment-020/level1-diagnostic*.json`) established that the failure is a property of the **pinned float32
model forward**, not of the decoded chain:

- The MLP reconstruction of every readout block is exact to ~1e-6; the attention carries the entire discrepancy.
- The discrepancy is already present **at the reference state, with no cue and no `Δ`**: each layer program against the
  model's own captured per-head outputs gives 6.1e-5 / 1.4e-4 / 3.1e-4 median at layers 3 / 4 / 5, while the MLPs give
  1.7e-6 / 3.2e-6 / 6.8e-6.
- The mechanism is a large **common offset** in the attention scores: `‖scores‖∞` is 2.0e4 / 7.9e4 / 1.2e5 median while
  the within-row spread the softmax actually uses is only ~10 / 23 / 21. float32 granularity at those magnitudes
  (`eps·‖scores‖∞` = 2.4e-3 / 9.5e-3 / 1.4e-2) therefore eats 2.5e-4 / 3.9e-4 / 7.8e-4 of the spread, which matches the
  measured attention-pattern error (5.9e-4 at layer 4, 1.3e-3 at layer 5) within ~1.6×. Softmax row sums are 1.0 in both
  the recomputed and the captured pattern and `scale` is 8.0 = √64, so it is the scores, not masking or scaling.
- Ruled out by measurement: a small denominator (the absolute error is flat at ~4.8e-4 across every denominator decile,
  and the denominators are 0.79 to 2.44), chaining (chained ÷ conditional error is 1.0× at block 4 and 1.5× at block 5),
  the measurement itself (`additive` 1.06e-6, `reference_component_sum` 1.24e-6), the inherited Experiment 017 chain
  (exactly 0), `Σ b_O`, and the parallel residual.

**The amendment.** `LEVEL1_TOLERANCE` is raised from `1e-3` to `7e-3`. Its envelope is the diagnosed floor, not the
observed maximum: the conditional per-block attention maxima are 1.45e-3 (block 3), 1.78e-3 (block 4) and 2.12e-3
(block 5), whose conservative absolute sum is 5.35e-3; over the smallest observed `‖Δh6^meas‖∞` of 0.79 that is
6.77e-3, rounded up to 7e-3.

**What this is and is not.** It is an identity/numerical tolerance amended after a recorded incident, with a measured
numerical reason. It is **not** a scientific outcome floor. Unchanged: the `E₁` formula and its norm normalization,
every Y1/Y2/Y3 floor and precondition, the populations, the 24 fresh cues, the 18 fresh frames, the 24 fresh nouns and
the confirmation set digest `e098e2b4…`, the comparator standing, the model, the runtime dtype and thread count, the
seeds, and every outcome rule and label. The Level 1 check keeps its force: the sequential block-5 variant that the
parallel residual forbids gives ~1.5e-2, which the amended gate still rejects by more than 2×, and a regression test
asserts exactly that. The diagnostic persistence added with the amendment stays: the identity maxima, their tolerances
and the worst Level-1 pair's full breakdown are written to the results state **before** enforcement, so a future
incident leaves the record needed to place it.

## Revision history

- **Revision 3** (2026-09-22, after the recorded `explore` incident and its diagnostic; authorized by the user on the
  diagnostic's evidence): `LEVEL1_TOLERANCE` 1e-3 → 7e-3 with the justification in the amendment note above, and the
  diagnostic persistence and `diagnose` subcommand that produced it. Nothing else changed — no floor, population,
  fresh set, comparator standing, model, runtime, seed, label or outcome rule. The incident run consumed no
  confirmation prompt and no fresh-noun quantity.

- **Revision 2** (2026-09-22, amended the same day with the Level 1 error formula `E₁` before implementation
  planning, and with the implementation note above after Task 1's identity check found the calibration bug — the 1e-3 threshold is unchanged and no prompt was run for the amendment): the exposed-noun accounting (80 entries, 79 scorable; `peach` non-scorable by the
  frozen single-token rule) and the fresh nouns' freshness certification (freshness for a noun is *no prior
  inspection of any model-output-derived statistic*, since a noun needs no prompt); the label-bearing populations of
  Y1, Y2 and Y3 frozen in a table (Y1 and Y2 on the scorable exposed nouns only, Y3 on the fresh nouns only, the
  joint cue × frame × noun statistic descriptive); scientific invalidity (invalid frames, unscored cues, non-scorable
  nouns) separated from incidents (implementation and protocol failures) and routed to the `PRECONDITION_FAILED_*`
  counts; a frozen tolerance for every identity and reproduction check that can stop a phase; the prompt-key manifest
  enumerated as S1-REF, S1-VALIDITY and S2-TARGET keys. One correction: revision 1's fresh-noun list contained
  `tunnel`, an exposed noun, because the list's tokenizer check did not apply the exposure filter; it is replaced by
  the next eligible entry `pillar`/`pillars` (43035/40652), and the eligibility rule is written out. No new prompt
  was executed for this revision; the floors, the program, the comparators, the cue and frame lists and the
  interpretation limits are unchanged.
- **Revision 1** (2026-09-22): first draft, written after the research spike (candidate A selected), the focused
  design pass (exact path, projection shares, freezability, rank-1 diagnostics, S1 ladder, reserve-noun margins) and
  the pre-spec diagnostics (end-to-end composition with the committed locks, provenance proof, Y3 formalization,
  statistical units, per-cue MAE distribution, the layer-5 simplification test).
