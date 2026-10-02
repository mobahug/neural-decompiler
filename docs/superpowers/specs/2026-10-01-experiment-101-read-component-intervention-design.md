# Experiment 101: intervening on the component that L03.H04 reads, in coordinated frames — design

**Status:** revision 3, 2026-10-01, decided by the reviewer on 2026-10-02 (§14). This is the frozen design. It contains
no code, has run no model, freezes no frame and computes no tolerance. Revision 1 had an independent read-only review:
READY WITH CHANGES, with no blocker, eleven SHOULDs and twelve notes. Revision 2 applied all of them. Its re-check,
READY WITH CHANGES with no blocker, raised four SHOULDs and eight notes, which revision 3 applies (§13). A check of
revision 3's changes found it READY FOR DECISION, and its four cosmetic notes are applied. The reports are in the
companion file [2026-10-01-experiment-101-design-reviews.md](2026-10-01-experiment-101-design-reviews.md).

Experiment 101 is the first experiment of the project's second series. Experiment 025 remains the end of the first
sequence (experiments/025-nounness-direction-intervention/README.md@8e571ba:285–286). Nothing here changes a frozen
result, a claim file or the submitted paper.

## 1. Why this experiment

Claims C002 and C003 are `LOCALIZED`. C003's boundary is that the decoded programs are prospectively predictive and that
their causal gates are unestablished: no preregistered, outcome-bearing test has intervened on a decoded variable itself
with necessity, sufficiency and control logic
(research/claims/C003-frame-conditioned-decoded-program.md@8e571ba:254–274). Of four preregistered routes to causal
status, three ran and none established it (research/paper/draft-v3-2026-09-30.md@8e571ba:516–524).

The paper-readiness assessment named one test as the one that would most change that boundary, and outlined it
(research/synthesis/paper-readiness-2026-09-28.md@8e571ba:333–349). Its question:

> does removing, installing or replacing the component that `L03.H04` reads change `Δc` in coordinated frames as the
> program predicts, beyond an equal-norm control?

This design makes that test operational. It fixes what the assessment left open: the exact edits, the controls, the
frames, the statistics, the decision rules and the outcome labels.

**Why now.** The assessment tied the test to the paper: worth running if *the paper's main claim, or the intended
venue,* needed causal evidence for the transport stage, and not to add to the experiment count
(research/synthesis/paper-readiness-2026-09-28.md@8e571ba:359–361). 101's reason is the project's second series. That
series asks how the model selects one learned computation over another. Its decisive experiment switches an internal
state and checks that the downstream computation follows, which needs a validated causal edit at a decoded variable,
with its controls, and causal evidence at the transport stage. 101's question is clear, new and falsifiable: no residual
edit along the read has ever been run.

What a result can show is the assessment's account (research/synthesis/paper-readiness-2026-09-28.md@8e571ba:318–324):
- **Effects established.** The read component would have causal evidence for this stage, within the scope of §7. A claim
  scoped to that stage could then be reviewed for higher maturity. That review is a separate decision.
- **Not established.** The boundary stays where it is. The result is never to be described as showing that the component
  has no causal role.
- **No equivalence claim.** The test has no equivalence bound, so it cannot show that an effect is below a margin.

## 2. The variable

The P1 read of `L03.H04` is `ρ_f(x) = (A_c^ref/σ_c^ref)·⟨x − mean(x)·1, γ₃ ⊙ m⟩`, with `m = W_V W_O d̂_T`
(docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md:52–56;
docs/superpowers/specs/2026-09-18-experiment-010-read-direction-assembly-design.md:37–39). Here:
- `x` is the residual entering layer 3 (`RESID_PRE.L3`) at the cue position `p_c`;
- `γ₃` is the gain of layer 3's attention LayerNorm;
- `W_V W_O` are head 4's value and output weights;
- `d̂_T` is the frozen number axis at the head's output.

Because the mean is subtracted, the read is linear in `x` along one covector:
- `w_c = γ₃ ⊙ m − mean(γ₃ ⊙ m)·1`, so that `⟨x − mean(x)·1, γ₃ ⊙ m⟩ = ⟨x, w_c⟩`;
- the **read coordinate** of a residual is `r(x) = ⟨x, w_c⟩`;
- the **read direction** is `ŵ = w_c/‖w_c‖`, with `‖w_c‖` about 0.766.

`γ₃ ⊙ m` is frozen in 011's lock as `read_weight`. 012–018 checked it against the weights; 020–025 read the lock's axis
directly. 101 uses the 011 vector and axis verbatim, and checks them against the weights before any prompt (G1).

**The cue pair defines the variable.** `d̂_T` is the mean plural-minus-singular change of the head's output over the
exposed frames' clean pairs, `one`/`two` and `each`/`several`, so `w_c` was estimated from the very pair 101 edits. The
freshness of 101 comes from its frames. A secondary stratum tests two strong plural cues that did not enter `d̂_T` (§3).

**The component.** For a reference prompt (cue `one`) and a plural swap in the same frame, the residual change at `p_c`
is `Δx = x_pl − x_ref`.
- Its **read part** is `(Δr/‖w_c‖²)·w_c`, with `Δr = r(x_pl) − r(x_ref)`.
- Its **off-read part** is `Δx − (Δr/‖w_c‖²)·w_c`. Its **centred** form `off_c` subtracts the mean, which every reader
  removes behind its LayerNorm. `û_off = off_c/‖off_c‖` is mean-zero, unit and orthogonal to `w_c`.
- The **slice norm** is `s = |Δr|/‖w_c‖`, the length of the read part. If `‖off_c‖ < s` in a frame, the control edits
  are applied as specified, overshooting along `û_off`, and the frame is flagged; `s/‖off_c‖` is reported for every
  frame.

The scale is known only from E-patch runs, not real swaps. In coordinated frames, the read of the E-patched change for
`two` was 1.846 and 1.879 (experiments/012-layer-correction-token-local/evidence/final-report-2026-09-19.md:40–41),
about 2.4 along `ŵ`. The calibration reports the real-swap `Δr`, `‖Δx‖`, `‖off_c‖`, `s` and `Δc_full` (§6).

**LayerNorm.** An edit along `w_c`, or along any mean-zero direction, leaves the residual's mean unchanged. It does
change the LayerNorm scale `σ`. With the centred residual about 11 units long and edits of about 2–3, `σ` can move by a
few percent. The edit sets `r` exactly, but not the head's normalized read `r/σ′`, and not its attention under the
plural query. The model computes all of this exactly, and so does the program. Every arm records `σ(x′)`, `‖γ₃ ⊙
e‖/σ(x′)` and `‖γ_ln2 ⊙ e‖/σ(x′)` for its edit `e`, where `γ_ln2` is the gain of block 3's MLP LayerNorm.

## 3. Site, frames and cues

- **Site.** `RESID_PRE.L3` at `p_c` (hook `blocks.3.hook_in`), in coordinated frames only. There the cue position and
  the prediction position differ (`p_t = p_c + 1`), and the head transports across them. In cue-final frames they
  coincide, so an edit there would act on every downstream component
  (research/synthesis/paper-readiness-2026-09-28.md@8e571ba:336–339). The residual at `p_t` is not edited.
- **Frames.** 36 new coordinated frames, written for 101 and never executed, of the form `NAME and NAME VERBED {cue}
  ADJ`. The cue is at `p_c` (4–7) and the adjective at `p_t`. The freeze takes the first 36 eligible candidates of §11.
- **The primary cue pair.** `one` (581) as the reference and `two` (767) as the plural cue, the pair of every
  coordinated frame. Every prompt key is new because every frame is.
- **The secondary stratum.** `three` and `many` as plural cues against `one`, in the same frames, with the same arms
  except the isotropic and head-specific ones. Neither entered `d̂_T`. Their E-patch shifts in coordinated frames were
  about −4.2 to −5.4 nats (experiments/007-supervised-cue-subspace/inherited/experiment-006-epatch-means.json). The
  stratum has preregistered statistics and no label force.
- **Nouns.** The 79 scorable exposed nouns that every experiment since 020 scores. One forward pass gives all 79
  contrasts at `p_t`.
- **The outcome.** `Δc = c(frame, cue, noun) − c(frame, one, noun)`, where `c = log P(singular) − log P(plural)` at
  `p_t`, averaged over the 79 nouns. A plural shift makes `Δc` negative. The E-patch shift for `two` in coordinated
  frames was −4.10 to −5.20 nats (same file); the calibration reports the real-swap value.

**A departure from the assessment.** The assessment asked for fresh cues as well as fresh frames
(research/synthesis/paper-readiness-2026-09-28.md@8e571ba:345). Fresh plural cues are exhausted. The numeral and pronoun
classes are used up, and every strong plural word was an earlier cue. The only cues with a current eligibility record
are eight adjectives and nouns and two measure nouns, whose plural effects are small or unknown. 101 asks whether a
variable is causal, not whether a rule generalizes to new cues. It therefore uses the pair that defines the variable, in
fresh frames, plus the secondary stratum. Every label is scoped accordingly. The reviewer decides (§12, D1).

## 4. The arms

Every arm runs in every frame, as one forward pass with all 79 contrasts read at `p_t`. An edit replaces `x(p_c)`
entering layer 3 with a computed vector, through the existing REPLACE intervention, whose landing and
zero-outside-change checks apply. `x_pl` and `x_ref` are the residuals captured from the plain runs of the same frame.
`Δr`, `s` and `û_off` are computed from them.

| arm | prompt | edit at `RESID_PRE.L3`, `p_c` | role |
|---|---|---|---|
| REF | `one` | none | the reference |
| PL | `two` | none | the full shift `Δc_full` |
| NEC-READ | `two` | `x_pl − (Δr/‖w_c‖²)·w_c`, the read coordinate set to the reference's | removal |
| SUF-READ | `one` | `x_ref + (Δr/‖w_c‖²)·w_c`, the read coordinate set to the plural's | installation |
| NEC-OFF | `two` | `x_pl − s·û_off` | primary control for removal |
| SUF-OFF | `one` | `x_ref + s·û_off` | primary control for installation |
| NEC-AMP | `two` | `x_pl + (Δr/‖w_c‖²)·w_c`, the read change doubled | direction check for removal |
| SUF-NEG | `one` | `x_ref − (Δr/‖w_c‖²)·w_c`, the read change reversed | direction check for installation |
| NEC-HALF, SUF-HALF | `two`, `one` | half of the NEC-READ and SUF-READ edits: only the read coordinate moves, to the pair midpoint | descriptive dose |
| PC-SWAP-PL, PC-SWAP-REF | `two`, `one` | the whole `x(p_c)` taken from the other prompt | descriptive: the share routed through `p_c`, and the hook path's positive control |
| NOOP | `two` | `x_pl` replaced with itself | the hook path's identity (G3) |
| NEC-ISO-j, SUF-ISO-j | `two`, `one` | `x_pl − s·v_j` and `x_ref + s·v_j`, for 8 directions | secondary controls |
| HEAD arms | `two`, `one` | none at `p_c`; `L03.H04`'s output at `p_t` replaced (below) | secondary, head-specific |

**What the arms isolate.**
- **The swap.** NEC-READ and SUF-READ are the *replace* or *swap* of the assessment: each prompt receives the other's
  read coordinate at `p_c`. At `p_c` alone, NEC-READ equals `x_ref` plus the off-read part, and SUF-READ equals `x_pl`
  minus it. The runs differ in everything else: NEC-READ runs the `two` prompt, so the residual at `p_t`, the head's
  query and block 0's change at `p_t` are the plural's. SUF-READ runs the `one` prompt, with the reference's.
- **The off-read remainder.** The assessment's control is covered twice. NEC-OFF and SUF-OFF move an equal length, `s`,
  along the cue's own off-read change. NEC-READ and SUF-READ are, by the identity above, the complementary test of the
  whole remainder.
- **Disruption.** NEC-AMP and SUF-NEG separate an effect of the read's value from generic disruption. Disruption of the
  cue's number information moves `c` toward neutral in either prompt. That mimics both a removal effect and an
  installation effect. A causal read predicts the opposite of disruption here: doubling the read change makes the plural
  prompt more plural, and reversing it makes the reference prompt more singular.
- **The share routed through `p_c`.** PC-SWAP-REF installs the plural's whole `x(p_c)` in the reference prompt, and
  PC-SWAP-PL removes it from the plural prompt. They measure how much of the shift runs through `p_c`. This is not a
  strict ceiling for the read-only edits, since the off-read part can oppose the read: 010 found such opposition for
  `this`, `a`, `another` and `every`. 005's exploratory runs found 0.956 recovered by this patch in coordinated
  development frames (experiments/005-regular-plural-mechanism/evidence/discovery-report-2026-09-18.md:34).
- **The half dose** moves only the read coordinate, to the midpoint of the pair. 005's midpoint replaced the whole
  state.
- **Isotropic directions.** The `v_j` are mean-zero unit directions orthogonal to `w_c`, from SHA-256 counter-mode
  Gaussians as in 025, with the tag `101|iso|{frame_id}|{j}`.
- **Head-specific arms.** Each uses the NEC-READ or SUF-READ edit. It replaces `L03.H04`'s output at `p_t` with the
  captured output plus the exact difference `head(edited) − head(unedited)`, computed from the measured inputs with the
  edit applied at `p_c` only, through the exact LayerNorm rather than P1's fixed `σ_c^ref`. The other readers of
  `x(p_c)` see the unedited residual.
  - HEAD-NEC and HEAD-SUF recompute the head's attention row and values. That edit reaches the head's key at `p_c` too.
  - HEAD-NEC-V and HEAD-SUF-V keep the measured attention row and change only the value at `p_c`. That matches P1's
    fixed attention, while its LayerNorm is exact, as for every head arm, and the assessment's alternative of *the
    head's value input only* (research/synthesis/paper-readiness-2026-09-28.md@8e571ba:339).
  - `L05.H07` attends to the cue position at about 0.999 and blocked 0.128 of the patch in 005's runs (same report,
    lines 40 and 50). The residual edit reaches this second transporter; the head arms do not.

**Per-frame fractions,** each divided by the frame's `Δc_full`. `Δc(arm)` is the arm's contrast minus the REF contrast,
averaged over the nouns.
- `F_rem = Δc(NEC-READ)/Δc_full`, the share of the shift left after the read part is removed.
- `F_rem_off = Δc(NEC-OFF)/Δc_full`, the share left after an equal-length off-read slice is removed.
- `F_ins = Δc(SUF-READ)/Δc_full`, the share produced by installing the read part alone.
- `F_ins_off = Δc(SUF-OFF)/Δc_full`, the share produced by installing an equal-length off-read slice.
- `A_amp = Δc(NEC-AMP) − Δc_full`, negative if doubling the read change adds plural shift.
- `A_neg = Δc(SUF-NEG)`, positive if reversing the read change adds singular preference.

## 5. The program's predictions, committed before any target prompt

**The program** is 023's completed program, with its coordinated mask and its frozen code blobs: the encoding change,
block 0's exact terms, layers 1–2 with block 2's MLP in full, and 020's frozen readout through blocks 3–5, including
block 5's frozen-row path from the cue position
(docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md:102–116).

**Stage 1** of the confirmation runs only the 36 reference prompts and records each frame's reference state. From these
states and the weights, the program predicts the swap to `two` (and, for the stratum, to `three` and `many`):
`Δx̂3(p_c)` and `Δx̂3(p_t)`. It then defines its own quantities: `Δr̂ = ⟨Δx̂3(p_c), w_c⟩`, `ŝ = |Δr̂|/‖w_c‖`, and
`û_off` from the centred off-read part of `Δx̂3(p_c)`. Every arm's prediction feeds the readout an explicit change at
both positions, relative to the reference state:

| arm | `Δx3(p_c)` fed to the readout | `Δx3(p_t)` fed to the readout |
|---|---|---|
| PL | `Δx̂3(p_c)` | `Δx̂3(p_t)` |
| NEC-READ | `Δx̂3(p_c) − (Δr̂/‖w_c‖²)·w_c` | `Δx̂3(p_t)` |
| SUF-READ | `(Δr̂/‖w_c‖²)·w_c` | 0 |
| NEC-OFF | `Δx̂3(p_c) − ŝ·û_off` | `Δx̂3(p_t)` |
| SUF-OFF | `ŝ·û_off` | 0 |
| NEC-AMP | `Δx̂3(p_c) + (Δr̂/‖w_c‖²)·w_c` | `Δx̂3(p_t)` |
| SUF-NEG | `−(Δr̂/‖w_c‖²)·w_c` | 0 |
| NEC-HALF, SUF-HALF | half the read edit, as above | as for NEC-READ, SUF-READ |
| PC-SWAP-PL, PC-SWAP-REF | 0 and `Δx̂3(p_c)` | `Δx̂3(p_t)` and 0 |

The table holds, per frame and cue, the predicted `Δĉ` of every arm and the predicted fractions `F̂_rem`, `F̂_ins`,
`F̂_rem_off`, `F̂_ins_off`, `N̂ = F̂_rem_off − F̂_rem` and `Ŝ = F̂_ins − F̂_ins_off`. A frame with `Δĉ_full` above −1
nat is recorded, and counts as a disagreement in P. The table is written, digested and **committed before stage 2 runs
any target prompt**, and stage 2 refuses to start unless the committed digest matches the stage-1 record.

This is the assessment's first option (research/synthesis/paper-readiness-2026-09-28.md@8e571ba:345–347). It means the
assessment's gap 4, frame-conditional tables held by digest but not committed before scoring, does not apply to 101's
own predictions. It does not change how 013–023's tables were held.

**What P can show.** With block 0 completed, the program's `Δx̂3` is nearly exact
(experiments/022-upstream-error-localization/README.md@8e571ba:225–229). 020's blocks 3–5 are exact except for block 4's
frozen attention and block 5's frozen rows. So P mainly tests whether those frozen parts hold under artificial edits,
and an agreement is close to a plain recomputation of the layers. In coordinated frames, 023's readout reached R² 0.9574
and 0.9518 against ceilings of 0.9579 and 0.9531 (experiments/023-block0-completion/README.md@8e571ba:143–145). Each
arm's ceiling, the readout fed the measured edited `Δx3`, is reported.

## 6. Calibration on exposed frames

Before the lock, a calibration phase runs every arm on the 36 exposed coordinated frames, with the same cues and nouns.
These data are spent, and the phase is exploratory, as 021–024's calibrations were.
- **Reference states.** The program's predictions for these frames use their locked reference states. The plan restores
  `outputs/experiment-020/results.json` from the byte-exact archive in
  `experiments/023-block0-completion/evidence/archive/` and checks its pinned hash.
- **What it sets.** `τ_rem` and `τ_ins`, for the cue `two`: the nearest-rank 90th percentiles, the 33rd of 36 sorted
  values, of `|F̂_rem − F_rem|` and `|F̂_ins − F_ins|`. If `τ_rem` or `τ_ins` exceeds **0.20**, P is uninformative: it
  is not tested, and the label says so (§7). Some exposed frames entered 012's bases and `d̂_T`, so `τ` may be biased
  tight; P's count is set for that (§7).
- **What it reports.** The real-swap `Δr`, `‖Δx‖`, `‖off_c‖`, `s`, `s/‖off_c‖` and `Δc_full`. The exposed `N`, `S`,
  `A_amp`, `A_neg` and P distributions. The power check: the share of exposed frames with each sign as predicted, with
  the sign-test power at that share.
- **Revision triggers,** the only ones allowed:
  - a gate failure;
  - `‖off_c‖ < s` in more than 6 of 36 exposed frames;
  - `Δc_full` above −1 nat in more than 6 exposed frames;
  - a non-finite fraction.

  A revision driven by the size of the exposed N, S, D or P results is forbidden. Any revision is recorded, with its
  trigger.

## 7. Statistics and decision rules

The unit is the frame, n = 36, for the `two` cue. A frame whose `Δc_full` is above −1 nat is reported. It counts as a
failure in every count and enters every median as the worst value (−∞ for a median of positive quantities). Ties count
as failures.

**The families:**
- **R, removal.** `N_f = F_rem_off − F_rem`: how much more of the shift the read removal takes away than an equal-length
  off-read removal. R's statistic is the count of `N_f > 0`. R requires that count to reach its Holm threshold and the
  median `N_f` to be at least **0.05**.
- **I, installation.** `S_f = F_ins − F_ins_off`: how much more of the shift installing the read part produces than an
  equal-length off-read installation. I's statistic is the count of `S_f > 0`. I requires that count to reach its Holm
  threshold and the median `S_f` to be at least **0.05**.
- **D, direction.** D passes if `A_amp < 0` in at least **25 of 36** frames and `A_neg > 0` in at least **25 of 36**.
- **P, the program's prediction.** A frame agrees if `|F̂_rem − F_rem| ≤ max(τ_rem, 0.02)` and `|F̂_ins − F_ins| ≤
  max(τ_ins, 0.02)`; the floor of 0.02 keeps P from failing on immaterial differences. P passes if at least **27 of 36**
  frames agree. With the nearest-rank `τ`, the exposed joint agreement is at least 30 of 36 (83 % or more), so 27 of 36
  asks for no material loss of accuracy on new frames. `N̂` and `Ŝ` are compared with `N` and `S` descriptively.

**The order.**
- R and I are tested together by Holm at a family level of 0.025, applied to the two count p-values alone. Each count's
  exact binomial tail, under a symmetric sign, is its p-value. The smaller p-value must be at most 0.0125 and the larger
  at most 0.025. For n = 36 that means **26 of 36** for the better family and **25 of 36** for the other. The tails are
  0.0057 and 0.0144.
- A family passes if Holm rejects its count's null and its median is at least 0.05. A failed floor does not change the
  other family's threshold.
- D is tested only if both R and I pass, and P only if D passes.
- A family that is not reached is reported descriptively and cannot change the label.

**Uncertainty.** Every median has a frame-bootstrap 95 % interval: 10,000 resamples with a fixed seed, 20261001,
reported with no decision force.

**Power,** before the median floors. If 80 % of frames show each predicted sign, R and I both pass under Holm with
probability about 0.915, D with about 0.917, and P with 0.87–1.00. The top label's probability is then about 0.73–0.84.
At 75 % it falls to about 0.41–0.47.

**The behavioural precondition.** At least **30 of 36** frames must have `Δc_full` of −1 nat or less. Otherwise the
label is `BEHAVIOR_PRECONDITION_NOT_MET`.

| outcome | label |
|---|---|
| any gate in §8 fails | `NOT_INTERPRETABLE` |
| the precondition fails | `BEHAVIOR_PRECONDITION_NOT_MET` |
| neither R nor I passes | `READ_COMPONENT_EFFECTS_NOT_ESTABLISHED` |
| R passes, I does not | `READ_REMOVAL_EFFECT_INSTALLATION_NOT_ESTABLISHED` |
| I passes, R does not | `READ_INSTALLATION_EFFECT_REMOVAL_NOT_ESTABLISHED` |
| R and I pass, D fails | `READ_COMPONENT_EFFECTS_DIRECTION_NOT_ESTABLISHED` |
| R, I and D pass, P is uninformative | `READ_COMPONENT_EFFECTS_PROGRAM_NOT_EVALUABLE` |
| R, I and D pass, P fails | `READ_COMPONENT_EFFECTS_PROGRAM_NOT_CONFIRMED` |
| R, I, D and P pass | `READ_COMPONENT_EFFECTS_AS_PREDICTED` |

**The scope of every label:** the read component of the residual entering layer 3 at the cue position, as every reader
of that residual sees it, in coordinated frames, for the cue pair `one`/`two` from which the read was estimated. No
label says that the component is unused, or that it has no causal role.

**Secondary, with preregistered statistics and no label force:**
- the stratum `three` and `many`: R, I and D's counts and medians, per cue;
- the isotropic controls: the counts of `F_rem < F_rem_iso` and `F_ins > F_ins_iso`, against each frame's median over
  the 8 directions;
- the head arms: the counts and medians of their own fractions; the removal share `(Δc_full − Δc(HEAD-NEC))/(Δc_full −
  Δc(NEC-READ))`, reported only when `1 − F_rem ≥ 0.05`; and the installation share `Δc(HEAD-SUF)/Δc(SUF-READ)`,
  reported only when `F_ins ≥ 0.05`, each for both variants;
- the midpoint dose, PC-SWAP, the additivity `F_rem + F_ins`, and every family not reached.

**The manipulation check, reported for every arm:** `L03.H04`'s realized output change `⟨ΔT, d̂_T⟩` at `p_t`, against
PL.

## 8. Gates

- **G1, geometry, before any prompt.** `w_c` recomputed from the weights equals the centred `read_weight` of 011's lock
  within 1e-9. Every `v_j` is unit, mean-zero and orthogonal to `w_c` within 1e-12 in float64.
- **G2, the edits.** Every edit lands bit for bit, with exactly zero change outside the slice (the existing checks). For
  the residual-edit arms, from the float32 vector, in float64:
  - the realized `r(x′)` equals its target within 1e-4;
  - the centred length of `x′ − x` equals its intended length within 1e-4, relative;
  - the edit is parallel or orthogonal to `w_c`, as specified, within 1e-4;
  - `û_off` is mean-zero, unit and orthogonal to `w_c` within 1e-6.
- **G3, the hook path.** In every frame, the NOOP arm reproduces PL bit for bit, as in 025. In every residual-edit arm,
  the captured `RESID_PRE.L4` at `p_c` equals `x′` plus block 3's attention and MLP outputs at `p_c`, within 1e-4 in the
  max-absolute norm, the tolerance of 020's analogous identity.
- **G4, the program's identities, for every residual-edit arm.** The head arms fail these by construction, because the
  exact chain recomputes `L03.H04` from the unedited input; G5 and their landing checks cover them.
  - 023's I3 and I4;
  - 020's readout and Level-1 identities;
  - the ceiling: the readout fed the measured edited `Δx3`, reported per arm.
- **G5, the head arms.** The head output recomputed from the unedited input reproduces the captured output: the head's
  attention row within 1e-4 and its output along `d̂_T` within 1e-3, both max-absolute, 017's tolerances for the same
  recomputation (measured there at 1.2e-5 and 3.3e-5).
- **G6, determinism.**
  - The stage-2 REF runs reproduce the stage-1 reference contrasts bit for bit.
  - Stage 2 refuses to start unless the committed table's digest matches the stage-1 record.
  - The table is reproduced bit for bit from its committed inputs at the report.
- **G7, isolation and accounting.**
  - Every run has an arm-tagged key `frame_id|word|token_id|arm`, executed exactly once. The manifest lists every arm
    tag, the NOOP included.
  - No untagged key `frame_id|word|token_id` was executed before stage 1.
  - The check reads the committed confirmation files and the run ledgers it names in the plan.

## 9. Phases

Each phase runs once, with a read-only review between phases:
1. **Design** (this document): independent review, then the reviewer's decision.
2. **Plan and implementation:** a module and a runner, with tier-A and tier-B tests on fakes, and an implementation
   review. `CURRENT_EXPERIMENT` becomes 101, and `outputs/experiment-101/` is ignored.
3. **Freeze:** the 36 frames, chosen by the tokenizer only, written to `confirmation-v1.json` and committed alone.
4. **Calibrate** (exposed, exploratory): `τ_rem` and `τ_ins`. Calibration review.
5. **Lock:** weight-only. It binds:
   - the frames, `w_c`, the `v_j`;
   - the rule for `û_off`, which is computed per pair at run time;
   - the program and its blobs, `τ_rem` and `τ_ins`, and §7.

   The user commits the lock.
6. **Confirm, stage 1:** the 36 reference prompts and the program's table, committed.
7. **Confirm, stage 2:** every arm in every frame, once.
8. **Report and closure,** with an independent confirmation review.

**Compute** is small. Each frame needs 33 runs for `two` (13 main arms with the NOOP, 16 isotropic, 4 head arms) and 11
for each stratum cue, 55 in all. That is about 2,000 for the confirmation, stage 1 included, and as many for the
calibration, at about 0.07 s each on this machine.

## 10. What a positive result would support

A claim review would weigh a positive label as follows:
- **What it would support:** a claim scoped to the residual read component at `RESID_PRE.L3`, `p_c`, in coordinated
  frames, for the pair `one`/`two`.
- **Attribution to `L03.H04`.** Crediting the effect to the head's transport, rather than to every reader of the
  residual, needs the head arms. They are secondary here.
- **Generality.** Generality beyond the pair that defines the axis needs the stratum `three` and `many`.

## 11. The frames

**Eligibility, checked with the tokenizer only, in candidate order.**
- The text is new: it equals no frame text in any earlier experiment.
- ` one`, ` two`, ` three` and ` many` are single tokens at `p_c`, with `p_c` from 4 to 7.
- The adjective is one token at `p_t = p_c + 1`, and no two 101 frames share it. It is not:
  - a cue word of any earlier experiment;
  - a word on 025's eligible-cue list;
  - a form of a target noun;
  - a number word.

Many adjectives also appear at `p_t` in earlier frames; novelty is at the level of the text, as in earlier new-frame
sets.

**Candidates, in order.** `{cue}` is the cue slot. Each adjective is a plain attributive adjective, and each verb takes
a plural object naturally.

| # | text | # | text |
|---|---|---|---|
| 1 | Aria and Ben carried {cue} large | 31 | Ann and Bill tied {cue} loose |
| 2 | Cora and Dan folded {cue} soft | 32 | Chris and Di pulled {cue} broad |
| 3 | Eli and Faye stacked {cue} gray | 33 | Ed and Fran pushed {cue} slim |
| 4 | Gus and Hope washed {cue} sticky | 34 | Greg and Holly sent {cue} sleek |
| 5 | Ian and June sorted {cue} tiny | 35 | Jim and Kara brought {cue} bold |
| 6 | Kit and Lou counted {cue} clean | 36 | Liz and Mark stored {cue} sealed |
| 7 | Max and Nell wrapped {cue} fine | 37 | Nate and Olive fed {cue} sick |
| 8 | Oscar and Paula dried {cue} damp | 38 | Phil and Rosa chose {cue} bland |
| 9 | Quinn and Rex cleaned {cue} messy | 39 | Ron and Sue grew {cue} lush |
| 10 | Sara and Tom fixed {cue} bent | 40 | Tim and Vicky calmed {cue} angry |
| 11 | Uma and Vera moved {cue} full | 41 | Adam and Betty drew {cue} quick |
| 12 | Walt and Yara lifted {cue} wet | 42 | Clare and Doug wrote {cue} funny |
| 13 | Zack and Abby filled {cue} deep | 43 | Dean and Ellen sang {cue} slow |
| 14 | Beth and Carl peeled {cue} bad | 44 | Fred and Gwen told {cue} odd |
| 15 | Dora and Erik sliced {cue} thin | 45 | Hugh and Iris heard {cue} strange |
| 16 | Gina and Hank cooked {cue} sour | 46 | Jake and Karen made {cue} simple |
| 17 | Ivy and Jack grilled {cue} lean | 47 | Lisa and Matt bought {cue} decent |
| 18 | Kate and Luke served {cue} good | 48 | Neil and Olga found {cue} real |
| 19 | Mia and Ned planted {cue} pale | 49 | Paul and Rita adopted {cue} lost |
| 20 | Olga and Pete picked {cue} neat | 50 | Ray and Sally picked {cue} early |
| 21 | Rita and Seth hugged {cue} tired | 51 | Sam and Tina ate {cue} stale |
| 22 | Tess and Uri labeled {cue} dry | 52 | Ted and Una hired {cue} smart |
| 23 | Val and Wes measured {cue} cool | 53 | Vince and Wendy met {cue} nice |
| 24 | Ada and Bo watched {cue} happy | 54 | Alan and Bea fed {cue} cute |
| 25 | Cleo and Dev mended {cue} long | 55 | Carl and Dina drew {cue} pretty |
| 26 | Gil and Jo sewed {cue} cozy | 56 | Eric and Flora fixed {cue} torn |
| 27 | Kim and Lia carved {cue} dull | 57 | Gary and Helen trained {cue} keen |
| 28 | Mo and Nia built {cue} tidy | 58 | Ian and Jill chased {cue} fast |
| 29 | Pia and Ray painted {cue} open | 59 | Ken and Laura cleaned {cue} worn |
| 30 | Sol and Ty repaired {cue} dead | 60 | Leo and Mona scrubbed {cue} stained |

If fewer than 36 candidates are eligible, the freeze writes nothing and the design is revised.

## 12. Decisions for the reviewer

- **D1. Cues.** As proposed: the pair `one`/`two`, which defines the variable, in fresh frames, plus the secondary
  stratum `three` and `many`. The alternative is to add the ten eligible weak words as a further stratum.
- **D2. The primary control.** As proposed: the equal-length slice of the frame's own centred off-read change.
  - If the effect per unit length simply followed `Δx`, removing an equal-length slice of the off-read change, which is
    most of `‖Δx‖`, would take away more than the read part. So the slice is the demanding control, and the isotropic
    directions are the weak one.
  - The assessment's *off-read remainder* is covered by the complementarity of NEC-READ and SUF-READ.
- **D3. Necessity's removal value.** As proposed: the reference prompt's read coordinate, a counterfactual swap. The
  half dose, which moves only the read coordinate to the pair midpoint, is descriptive. 005's necessity-oriented source,
  the pair midpoint, replaced the whole state
  (docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md:343–364).
- **D4. The head arms.** As proposed: secondary, in two variants, with preregistered statistics. The alternative makes
  them a fifth family.
- **D5. The numbers:**
  - 36 frames;
  - Holm over R and I at 0.025 (26 and 25 of 36);
  - median floors of 0.05;
  - D at 25 of 36 for each sign;
  - `τ` at the nearest-rank 90th percentile, capped at 0.20, with a floor of 0.02 in P's agreement;
  - P at 27 of 36;
  - the precondition at 30 of 36.
- **D6. The order:** R and I by Holm, then D, then P. The alternatives are a fixed sequence, R then I, which revision 1
  proposed, or D before R and I.

## 13. Risks, and what changed from revision 1

**Risks.**
- **Every reader of `x(p_c)`.** The residual edit reaches MLP3, the other layer-3 heads, `L05.H07` and block 5's path
  from the cue position. A residual-level result is evidence about the read direction at that site. The head arms
  separate the head's own transport, descriptively.
- **Off-distribution states.** The edited residuals were never produced by the model. The controls are edits of the same
  length at the same site, so they are equally artificial. NEC-AMP and SUF-NEG go further from the natural range than
  the swaps do.
- **Self-repair.** Later components may compensate for a removal. That would weaken R and widen the gap from the
  program, which models no compensation. Holm keeps R from deciding the label alone.
- **The position `p_t`.** In a real swap, block 0 also changes `x(p_t)`, which 101 does not edit. PC-SWAP measures how
  much of the shift runs through `p_c` at all.
- **Saturation.** If the plural prompt is already near its limit, doubling the read change may add little, and D's
  removal count could fail for that reason. The calibration's power check shows whether that is likely, but it cannot
  change the rule.
- **History.** None of the four earlier routes established its effect. This design is powered by a strong cue pair and
  36 frames.

**What revision 2 changed,** following the review:
- **The program.** The program's side of every arm is specified, with both positions and its own quantities. P's scope
  is stated. `N̂` and `Ŝ` are in the table.
- **New arms.** NEC-AMP and SUF-NEG form family D. The midpoint is a descriptive dose, and PC-SWAP is a bound and a
  positive control.
- **The control.** `û_off` is centred, the lengths are matched in centred space, and LayerNorm footprints are recorded.
- **The swap identity** is stated at `p_c` only, with what differs elsewhere.
- **The cues.** The cue pair's role in defining the variable is stated, and the stratum `three` and `many` added.
- **Labels and order.** The labels are reworded and scoped. A behavioural precondition is added, and Holm replaces the
  fixed sequence for R and I.
- **Statistics.** Weak frames enter the medians as the worst value. The percentile method and a cap for `τ` are fixed.
  P's count is explained, the revision triggers are enumerated, and bootstrap intervals are added.
- **Gates.** G2–G7 are extended: per-arm geometry, the hook path, identities, the head arms, digests, and arm-tagged
  accounting.
- **Wording.** The gap-4 sentence is corrected, and so is the sentence on the 011 read weight's checks. The scale
  figures are labelled as E-patch values.
- **The frames.** The nine candidates that failed the tokenizer checks, and the semantically odd and noun-like ones, are
  replaced.
- **Definitions.** The lock binds the rule for `û_off`, not `û_off` itself. The program and the run criterion are named.

**What revision 3 changed,** following the re-check:
- **Holm and the floors.** Holm is applied to the two count p-values alone, and a failed floor does not change the other
  family's threshold.
- **The head share.** The removal share is now `(Δc_full − Δc(HEAD-NEC))/(Δc_full − Δc(NEC-READ))`. The head arms use
  the NEC-READ and SUF-READ edits through the exact LayerNorm.
- **Gates.** G2 and G4 are scoped to the residual-edit arms. G3 uses a NOOP arm and a max-absolute 1e-4, and G5 uses
  017's tolerances. The manifest lists every arm tag.
- **Labels.** D's failure label is `READ_COMPONENT_EFFECTS_DIRECTION_NOT_ESTABLISHED`, and an uninformative P has its
  own label.
- **Wording.** `τ`'s cap and cue, and the floor of 0.02 in P, are stated. The power, compute and agreement figures are
  corrected. PC-SWAP is the share routed through `p_c`, not a ceiling. The half dose's wording, the overshoot rule and
  the reason for running now are fixed.
- **The frames.** The eight candidates whose adjective is more than one token, the seven flagged picks and five
  noun-like or predicative backups are replaced.

## 14. The reviewer's decision (2026-10-02)

The reviewer authorized Experiment 101 on 2026-10-02 and decided §12 as proposed, with these rulings. They are part of
the frozen design.

- **D1, cues.** Approved. `one`/`two` is the primary pair. `three` and `many` stay explicitly secondary and cannot
  rescue a failed primary result. The primary claim concerns the `one`/`two` contrast in coordinated frames, not number
  in general.
- **D2, the primary control.** Approved, with random directions secondary. The non-read part and the norm matching are
  frozen by this design, before calibration:
  - the non-read part is the centred off-read part of `Δx` at `p_c` (§2);
  - the control edits have the centred length `s = |Δr|/‖w_c‖` along the centred unit `û_off`.

  Neither calibration nor the lock may change them.
- **D3, the removal value.** Approved. The plural prompt receives the reference's read coordinate. The midpoint is
  strictly descriptive.
- **D4, the head-only arms.** Approved as secondary. Editing the residual at the cue position reaches every component
  that reads it, so the primary result concerns the read direction at that site. A statement specific to `L03.H04`
  requires the head-only evidence and remains a separate, secondary conclusion.
- **D5, the numbers.** Approved and frozen exactly:
  - Holm thresholds of 26 and 25 of 36 for removal and installation;
  - medians of at least 0.05;
  - 25 of 36 for each direction count;
  - 27 of 36 for the program's prediction;
  - at least 30 of 36 frames for the behavioural precondition.

  None is loosened after any outcome is seen.
- **D6, the order.** Approved, with one hard constraint: every program prediction is generated and cryptographically
  frozen before any target intervention result is observed. The table's digest is committed after stage 1 and checked
  before stage 2 (§5, G6). Formal scoring then follows R and I, then D, then P.
- **The frame-freeze guard.** The 36 frames are the first 36 eligible candidates of §11, in the listed order. They are
  chosen by the tokenizer alone, with no model output, and frozen before calibration. No frame is replaced after the
  freeze. A frame may be excluded only under a preregistered validity rule. There is none beyond §7's weak-shift rule,
  which counts such a frame as a failure rather than excluding it.

**The frozen claim hierarchy.**
- **The primary question.** Does manipulating the decoded read direction at the cue-position residual causally alter the
  `one`/`two` singular–plural choice in new coordinated frames? It is answered by removal (R), installation (I) and the
  directional dose and reversal (D), each against the matched non-read control, with the labels of §7.
- **The secondary question.** Is `L03.H04` specifically the component that mediates the effect? The head-only arms
  answer it descriptively, and it cannot change the primary label.
- **Generalization.** Does the effect extend to `three` and `many`? This is secondary only, and it cannot rescue a
  failed primary result.
- **If the primary passes but the head-only evidence does not support the head:** the decoded residual direction is
  causal at this site, within the scope of §7, but its effect cannot be assigned uniquely to `L03.H04`.

No model is run until the implementation and its review are complete.
