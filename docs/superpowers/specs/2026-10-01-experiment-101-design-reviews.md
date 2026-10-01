# Experiment 101 design: the independent reviews

These are the three read-only reviews of the [design](2026-10-01-experiment-101-read-component-intervention-design.md)
on 2026-10-01, all by one independent agent. The harness blocked writing to the scratchpad, so each came back as a
message. Each is condensed from that message: every finding, its evidence and its fix are kept, in shorter wording. Line
numbers refer to the revision under review. What was done about each finding is in the design's §13.

| review | of | verdict |
|---|---|---|
| 1 | revision 1 | READY WITH CHANGES: no blocker, eleven SHOULDs, twelve notes |
| 2 | revision 2 | READY WITH CHANGES: no blocker, four SHOULDs, eight notes |
| 3 | revision 3's changes | READY FOR DECISION: four cosmetic notes |

---

## Review 1, of revision 1

**Verdict: READY WITH CHANGES.** The review found no blocker, and the variable is the right one. The edit algebra, the
arithmetic, the citations, the code paths and the frame freeze all check out. Before the plan is written, the design
needs a fuller specification of the program side and of the statistics, one more control, safer label wording and more
gates.

**What was done.** The review read the design in full and verified every record it cites. It recomputed ‖w_c‖ from 011's
lock, traced the intervention, program and readout code, and checked the binomial arithmetic. It loaded only the pinned
tokenizer, offline at e93a9faa…, to check the 50 candidates. Nothing was edited, no model was loaded, and nothing under
`outputs/` was read.

**Fidelity summary.** The design tests the assessment's question: remove, install and swap the read component at
`RESID_PRE.L3` `p_c` in coordinated frames, against an equal-norm control, with the program's predictions committed
before any target prompt. It departs in five places:
- **Cues:** exposed one/two instead of fresh cues. This is flagged as D1, but understated (S5).
- **Controls:** the assessment's two controls are merged into one slice. This is flagged as D2 (N1).
- **Necessity value:** the counterfactual rather than 005's midpoint. This is flagged as D3.
- **P's scope:** P scores only F_rem and F_ins, not "Δc for each arm". This is not flagged (S1).
- **Head arms:** they change the head's pattern and value, not "the value input only". This is not flagged (N2).

### BLOCKER
None.

### SHOULD

**S1. The program-side arms are underspecified** (§5 l.127–131). At stage 1 only reference states exist, so the program
cannot use the measured Δr or û_off. The design must define the program's own quantities: Δr̂ = ⟨Δx̂3(p_c), w_c⟩; û_off
from the program's off-read change; the `p_t` entry, Δx̂3(p_t) for the `two` arms and an explicit zero for the `one`
arms; and a weak-shift rule for Δĉ_full. The assessment asks for "Δc for each arm" (paper-readiness-2026-09-28.md:344),
but P scores only F̂_rem and F̂_ins, so N_f and S_f are never compared with N̂_f and Ŝ_f. Evidence:
`readout_decompilation.py:429–454` (`blocks_3_to_5` iterates over the keys of `dx3` and returns `dh4[p_t]`, so a SUF arm
passed only `{p_c: …}` raises an error); `upstream_localization.py:837–852`. Fix: write the five program-side edit
formulas into §5; require `p_t` in every `dx3`; make Δĉ_full > −1 count as disagreement; put N̂_f and Ŝ_f in the
committed table, scored in P or as descriptive predictions.

**S2. No arm separates read-specific causation from generic disruption** (§4 l.90–99; §12 l.265–266). An edit that only
degrades the cue's number information would move c toward neutral, which mimics both N and S. The off-read slice
controls for this only if it is as disruptive per unit norm as the read edit, which is not guaranteed because w_c is
γ₃-weighted; the isotropic directions are a weaker control still. Fix: add NEC-AMP (`x_pl + read part`) and SUF-NEG
(`x_ref − read part`), whose program predictions move Δc further plural-ward and singular-ward while disruption predicts
the opposite, with pre-specified sign counts, at least as secondary statistics and preferably as a conjunctive
condition. For D3, add a half dose (005's midpoint, 005 design:347–355) as a descriptive dose point with its program
prediction.

**S3. The primary control is not mean-zero, and norms are matched only in raw residual space** (§4 l.101–103; §2
l.57–58). û_off carries the mean(Δx)·1 component, which every reader removes behind its LayerNorm (009 design:38–39), so
the control's effective norm is below the read edit's. The edits' LayerNorm footprints differ, and σ can move by a few
percent (σ_ref at `p_c` is 0.46–0.53, 014 lock `locked_states.*.sigma_ref`; the centred residual is about 11 units long
and the edits about 2.1–2.5). The edit sets r exactly, but not the head's normalized read r/σ′, and not A_c under the
plural query at `p_t`. Fix: centre û_off; match centred norms; record σ(x′), ‖γ₃⊙e‖/σ and ‖γ_ln2⊙e‖/σ for each arm.

**S4. One equivalence statement is wrong at the level of the run, and the `p_c` route is not measured** (§4 l.106–108;
§12 l.269–270). "NEC-READ is also the reference prompt with only the off-read part installed" holds at `p_c` only:
NEC-READ runs the `two` prompt, so x3(p_t), L03.H04's query and block 0's change at `p_t` are the plural's, and the
reverse for SUF-READ. The share of the shift that runs through `p_c` bounds F_rem and F_ins but is never measured; 005
measured it in exploratory work (discovery-report-2026-09-18.md:34, 50: 0.956 recovered; L03.H04 blocks 0.865; L05.H07
0.128). Fix: correct the sentence; add PC-SWAP arms as descriptive context and as the hook path's positive control.

**S5. D1 understates the departure on cues** (§3 l.77–82; §11 l.245–247). The reference pair is not only exposed; it
defines the variable: d̂_T is the mean plural-minus-singular head output over the exposed frames' clean pairs, one/two
and each/several (`cue_suppression.py:261–275`; `plural_mechanism.py:1181–1195`; 009 design:84–85), and w_c is built
from d̂_T. Fix: state this in §3, D1 and the scope of any label; as the D1 alternative, prefer a secondary stratum with
a strong exposed plural cue that did not enter d̂_T, such as `three`, `many` or `few` (E-patch shifts in coordinated
frames −4.2 to −5.4 nats, 006 extract).

**S6. The labels overclaim** (§7 l.167–171). N and S compare an edit with an equal-norm control under a 0.05 median
floor, at a residual site that every block-3 component, MLP3 and block 5's path read; the methodology speaks of
necessity-oriented and sufficiency-oriented interventions (`docs/RESEARCH_METHODOLOGY.md:83–84`). Fix: use labels such
as READ_REMOVAL_EFFECT_NOT_ESTABLISHED, …_INSTALLATION_NOT_ESTABLISHED, …_PROGRAM_NOT_CONFIRMED and
READ_COMPONENT_EFFECTS_AS_PREDICTED, each with a fixed scope sentence; keep §1's "never no causal role" wording.

**S7. The statistics leave forks open and report no uncertainty** (§6 l.143–146; §7 l.150–160). (a) Say how weak frames
enter medians, for example as −∞. (b) Fix the percentile method for n = 36, for example nearest rank, the 33rd of 36.
(c) Pre-set a maximum τ, beyond which P is uninformative. (d) Two 90 % marginals give a joint exposed agreement of 80–92
%, so 27/36 is a "no material degradation" criterion; τ may be biased tight because some exposed frames entered 012's
bases and d̂_T. (e) Enumerate the revision triggers, forbid revisions driven by the exposed N, S or P magnitudes, and
define the power check. (f) Level 3 requires uncertainty (methodology:86): pre-specify frame-bootstrap confidence
intervals for the median N_f and S_f.

**S8. The gates are incomplete** (§8 l.178–185). (i) For each arm, check the realized r(x′), the centred ‖x′ − x‖, and
orthogonality or parallelism to w_c. (ii) Add a patch-path check (a REPLACE of x_pl with itself reproduces PL bit for
bit, as in 025 design:258) and a pass-through identity (the captured `RESID_PRE.L4` at `p_c` equals x′ + Δattn3 +
Δmlp3). (iii) Carry over 023's I3 and I4 (`block0_completion.py:126–136`), 020's readout and Level-1 identities
(`readout_decompilation.py:691–696`) and the per-arm ceiling. (iv) For the HEAD arms, check that the head output
recomputed from the unedited input reproduces the captured output. (v) Stage 2 must refuse to start unless the committed
stage-1 table's digest matches the stage-1 record. (vi) Use arm-tagged keys (`frame_id|word|token_id|arm`) with
exact-once accounting. (vii) Add a global behavioural precondition, for example at least 30 of 36 frames with Δc_full ≤
−1, or else a distinct label. (viii) Add a manipulation check: the realized ⟨ΔT, d̂_T⟩ of L03.H04 in every arm, against
PL.

**S9. The gap-4 claim is overstated** (§5 l.133–135). Committing 101's table means gap 4 would not apply to 101; it does
not change how 013–023's tables were held (C003:132–137). The gap is also the assessment's gap 4
(paper-readiness:244–245), not "C003's gap 4". Fix: reword.

**S10. The design does not say what P can show** (§5; §12 l.271–272). With block 0 completed, Δx̂3 is nearly exact (022
README:225–229), and 020's blocks 3–5 are exact except for frozen block-4 attention and frozen block-5 rows
(`readout_decompilation.py:423–427`). So P mainly tests that freezing under artificial edits. The risk line calling the
chain "the weakest part of the record" predates 023, whose coordinated R²₁ was 0.9574 and 0.9518 against ceilings of
0.9579 and 0.9531 (023 README:143, 145). Fix: state this in §5 and §12, and report the per-arm ceilings.

**S11. The scale figures come from E-patch runs, not real swaps** (§2 l.53–55; §3 l.74–75). 012's report:40–41 gives the
read of the E-patched Δr, 1.846 and 1.879 in coordinated frames; the range 1.6–1.9 spans all templates. The 006 file
holds E-patch residual responses, −4.10 to −5.20 for `two`. Fix: label these as E-patch values, and have the calibration
report the real-swap Δr, ‖Δx‖, ‖off‖ and Δc_full.

### NOTE

- **N1. D2's rationale should be scientific.** If the effect per unit norm simply follows Δx, an equal-norm slice of the
  off-read change, which is most of ‖Δx‖, removes more than the read part does, so the off-read slice is the demanding
  control. Citing 025's B "failure" is weak, since B was moot once A failed. The assessment's "off-read remainder" is
  covered by complementarity, not by the slice.
- **N2. D4, the head arms.** Pre-specify their statistics, including the head share; add a value-only variant that keeps
  the unedited attention row (paper-readiness:339); compute each replacement as the captured output plus
  [program(edited) − program(unedited)]. L05.H07 attends to the cue at 0.999 and blocks 0.128 of the patch (005
  report:40, 50).
- **N3. D6, the order.** N is the family most exposed to self-repair; testing S first, or a Holm split over {N, S},
  would avoid hinging the label on it. State the power for the top label.
- **N4. The frames, by tokenizer.** 41 of 50 eligible. Five fail as earlier cue words (#9 wooden, #11 yellow, #14 wild,
  #16 young, #22 wide) and four as two-token adjectives (#26 snowy, #28 windy, #29 chilly, #30 misty). Several selected
  frames are semantically odd (#13 noisy, #15 tame, #17 steep, #18 bare, #23 mild, #24 lean, #27 sunny), and some
  adjectives can read as nouns (light, solid, flat, round, plain, short).
- **N5.** Define the case ‖off‖ < |Δr|/‖w_c‖, and report Δr, ‖off‖ and the slice fraction for each frame.
- **N6.** The 023 loader expects `outputs/experiment-020/results.json`, which is hash-pinned; the plan must restore it
  from the archive or capture fresh exposed reference states and check them. G4 must name the ledgers it reads.
- **N7.** The read weight's check against the weights is visible in 012–018, not in 020–025.
- **N8.** Say how 101 meets the assessment's run criterion (paper-readiness:359–361) and the "clear, new, falsifiable
  question" criterion.
- **N9.** The lock can bind the v_j and the rule for û_off, but not û_off itself.
- **N10.** Name the program unambiguously: 023's completed program, coordinated mask 30, with its code blobs.
- **N11.** Optionally, calibrate on 48 spent coordinated frames, adding the 12 from 022 and 023.
- **N12.** For C003's review, a positive result would support a claim scoped to the residual read component for the
  one/two pair; attributing it to L03.H04's transport needs the HEAD arms, and generality needs S5's stratum.

### Checked and correct
The citations; the covector (‖w_c‖ 0.76592); the edit algebra and signs; the arithmetic (P(X ≥ 25 | 36, ½) = 0.01441;
sign-test power 0.958 at p = 0.8); the feasibility of REPLACE at `RESID_PRE.L3` and at L03.H04, of `head_output_change`,
`compose_dx3` and `contrast_of`; the token ids 581 and 767; the p_c range; and that the strong plural cues are
exhausted.

---

## Review 2, of revision 2

**Verdict: READY WITH CHANGES.** There is no blocker. Revision 2 resolves every revision-1 SHOULD and every note it
adopted. Four small textual fixes remain: how Holm interacts with the median floors; the formula for the removal head
share; the scope and tolerances of the gates; and the wording of one label.

**SHOULD-1. The interaction of Holm and the median floors is undefined.** Fix: Holm is applied to the two count p-values
alone; a family passes if Holm rejects its count null and its median is at least 0.05; a floor failure does not change
the other family's threshold.

**SHOULD-2. The removal head share has the wrong formula.** Δc(HEAD-NEC)/Δc(NEC-READ) is a ratio of the shifts that
remain. Fix: (Δc_full − Δc(HEAD-NEC))/(Δc_full − Δc(NEC-READ)), reported only when 1 − F_rem ≥ 0.05.

**SHOULD-3. Gate scope and tolerances would void the run if implemented as written.** (a) G4 "for every arm" fails by
construction for the HEAD arms (`readout_decompilation.py:456–491`); scope G4 and G2's geometric checks to the
residual-edit arms. (b) G5's 1e-5 is below the record's values for the same recomputation (017 measured 1.2e-5 and
3.3e-5, `experiments/017-transport-head-pattern/evidence/final-report-2026-09-20.md:18`, against tolerances of 1e-4 and
1e-3, `head_pattern.py:75–76`); adopt 017's tolerances and name the norm. (c) G3 should use max-abs; 020's analogous
identity uses 1e-4 (`readout_decompilation.py:71`).

**SHOULD-4. One label overclaims.** `READ_COMPONENT_EFFECTS_NOT_DIRECTION_SPECIFIC` asserts non-specificity, yet D can
fail through saturation. Fix: `READ_COMPONENT_EFFECTS_DIRECTION_NOT_ESTABLISHED`.

**Notes.**
- **N-a.** Write "if τ_rem or τ_ins exceeds 0.20"; say τ is computed for `two`; consider a floor max(τ, 0.02); consider
  a distinct label for "P uninformative".
- **N-b.** Under Holm, R and I both pass with probability 0.915 at an 80 % sign rate; D 0.917; P 0.87–1.00; the top
  label 0.73–0.84; at 75 % it falls to 0.41–0.47; the joint exposed agreement is 30–33 of 36.
- **N-c.** Counting gives 33 runs per frame for `two` (12 main arms, 16 isotropic, 4 head arms and the no-op), 11 per
  stratum cue, 55 per frame, about 2,016 for the confirmation. G7's manifest must list every arm tag, including the
  no-op.
- **N-d.** PC-SWAP is not a strict ceiling, since the off-read part can oppose the read (010; C003:167); call it the
  share routed through `p_c`.
- **N-e.** The HALF arms move only the read coordinate to the pair midpoint; 005's midpoint replaced the whole state.
- **N-f.** Define the confirmation rule when ‖off_c‖ < s.
- **N-g.** The head arms use the NEC-READ and SUF-READ edits, and the V variant uses the exact LayerNorm, not P1's fixed
  σ_ref.
- **N-h.** The assessment's criterion is "the paper's main claim, or the intended venue"; state the series' need as the
  reason.

**Frames.** 52 of 60 eligible; eight fail as multi-token adjectives (#6 rusty, #14 plump, #29 lumpy, #30 shabby, #40
frail, #57 glossy, #58 fluffy, #60 grimy). Among the picks, #12 stout reads as a noun, #18 "served one hot" and #21
"boxed one tight" read predicatively, and #24, #36 and #37 are odd.

**Checked and correct.** The §5 table's nine rows; family D's signs and logic; Holm's thresholds (26 and 25 of 36; tails
0.00567 and 0.01441) and familywise control; τ as the 33rd of 36; weak frames as −∞; the enumerated revision triggers;
the precondition and the scope sentence; the figures (` three` is token 1264 and ` many` 1142; mask 30 is emb + Bv +
Bp + T).

---

## Review 3, of revision 3's changes

**Verdict: READY FOR DECISION.** Revision 3 applies all four SHOULDs and all eight notes correctly. Four cosmetic items
remain; none affects a decision rule or a label.

- **The changes:** SHOULD-1 to SHOULD-4 and N-a to N-h all read correctly, including the new
  `READ_COMPONENT_EFFECTS_PROGRAM_NOT_EVALUABLE` label. A τ above 0.20 is known before the lock and is not a revision
  trigger, so there is no fork. The power, compute and agreement figures match an independent recomputation.
- **Frames:** all 60 of 60 eligible under the full rules. Every text is new; ` one`, ` two`, ` three` and ` many` (581,
  767, 1264, 1142) are single tokens at p_c 4–6; every adjective is a single token at `p_t`, distinct, and none is an
  earlier cue, on 025's lists, a target-noun form or a number word. The freeze would pick #1–36. A few verb–adjective
  pairs are mildly odd (#14, #23, #27, #33, #35), as some of the record's earlier frames were; none needs a change.
- **Cosmetic items:**
  1. "That matches P1's definition" contradicts the exact LayerNorm; write that it matches P1's fixed attention.
  2. Guard the installation share as the removal share is guarded (F_ins ≥ 0.05).
  3. With the 0.02 floor, the exposed agreement can exceed 33; write "at least 30 of 36".
  4. "the six flagged picks" should be "the seven flagged picks", and five backups were also replaced.
