# Neural Decompiler: scientific state memo (Experiments 005–025, and 101)

> **Provenance.** An AI-authored memo (a Claude survey agent, 2026-10-14), fact-checked by a second, independent AI agent (`checks/research-state-2026-10-14-verification.md`: 202 claims checked; its 38 required corrections applied; its optional clarifications not applied). Not a human validation. Tags: **[AI observation]** = the author's own analysis; **[proposal]** = a suggestion, not a decision. Line citations refer to the branch `exp101-phases` at `3621eec`.

**Date:** 2026-10-14. **Repository:** branch `exp101-phases`, HEAD `3621eec`, clean. Outside
`experiments/101-read-component-intervention/`, every cited file is identical at `40829d2`. The freeze's
`confirmation-v1.json` (installed `a857f9a`) and the calibration's `calibration-v1.json` (installed `d502777`) are
committed (see §4.1).

**Method:** a read-only reading of the claims, the synthesis, the paper, the literature review, the experiment READMEs
and reports, 101's design, reviews and plan, and `src/`. No repository file was changed, and no model, runner or test
was run.

**Tags:** **[AI observation]** marks my own analysis, not an established finding. **[proposal]** marks a suggestion, not
a decision.

**Citation keys.** Paths are relative to the repository root, with line numbers at HEAD.
- **Claims:** C002 = `research/claims/C002-count-cued-noun-number-circuit.md`; C003 =
  `research/claims/C003-frame-conditioned-decoded-program.md`; C002REV = `research/claims/reviews/C002-review-2026-09-27.md`.
- **Paper and synthesis:**
  - PAPER = `research/paper/draft-v3-2026-09-30.md`, the canonical text of the TMLR build;
  - SYN = `research/synthesis/synthesis-005-025.md`;
  - READY = `research/synthesis/paper-readiness-2026-09-28.md`;
  - MAP = `research/synthesis/source-map-005-025.md`;
  - LIT = `research/paper/literature-review/review-2026-09-28.md`;
  - METH = `docs/RESEARCH_METHODOLOGY.md`.
- **Experiments:**
  - Ennn = `experiments/nnn-*/README.md`;
  - FRnnn = its `evidence/final-report-*.md`; FR020 is `evidence/exploration-report-2026-09-22.md`;
  - DR005 = `experiments/005-*/evidence/discovery-report-2026-09-18.md`;
  - DSnnn = its design under `docs/superpowers/specs/`.
- **Experiment 101:** D101 = `docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md`;
  R101 = `docs/superpowers/specs/2026-10-01-experiment-101-design-reviews.md`; P101 =
  `docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md`.

E025's line numbers are 6 higher than the claims' "(025:L)" citations, which refer to commit `d7c5e62`. The other
READMEs are unchanged since that commit.

## 0. Key points

1. **Nothing is established as causal**, which is not a finding that nothing is causal (E025:255; D101:42–43). Both
   claims are `LOCALIZED` (C002:5; C003:5). "The programs are prospectively predictive;
   their causal gates are unestablished." (C003:392)
2. **The localization (C002) rests on interventions on exploratory and calibration data.** Its one confirmation failed a
   behavioral precondition (C002:51).
3. **[AI observation] The prediction (C003) is strong at the stage level, with recorded failures (015 Y2, 018 Y2,
   021 Y2, 023 Y1/cue_final; C003:174, 178, 183, 186).** Examples: 011's Spearman 0.848; 017's ΔT R² 0.999 and
   1.000; 023's g ≥ 0.9966 in three conditions. It is conditional on measured reference states, and much of its
   accuracy may come from exact recomputation, a rival the record has not distinguished (C003:331–337).
4. **[AI observation] No causal route that ran failed a primary criterion that C002 maps to necessity, sufficiency or
   control** (005's non-confirmatory P8 passed; only its secondary compensation band was missed, C002:51). They stopped
   on a behavioral precondition (005), a pair-level fidelity correlation (007), and a sign count over two strata that,
   descriptively, responded oppositely (025; E025:258–261). 006's route never ran: a gate on the separate program blocked its lock.
5. **[AI observation] Experiment 101 is well specified; R and I pass together with probability about 0.915 if 80 % of
   frames show each predicted sign (D101:274–276).** It tests one decoded variable, at one site, with the cue pair
   that defined it. A positive label would support reviewing a claim scoped to that stage, not C002 or C003 as wholes.
6. **[AI observation] 101 has risks the reviews did not raise:** no rowspace or dormant-direction diagnostics, a one-dimensional
   manipulation check, no site or layer negative control, a primary label that adds little information at the margin,
   and an unstated inferential population.

## 1. Established results, by kind of evidence

**The ladder.** A Level-2 localization must not be presented as proof of causality. Level 3 requires necessity-oriented,
sufficiency-oriented and control interventions that support a prediction documented beforehand (METH:80–89).

### 1.1 Localization: C002 (`LOCALIZED`; exploratory and calibration data)

- **The claim.** The cue's number reaches the noun through three stages (C002:15–24):
  - E: `L00.MLP` at the cue position;
  - T: `L03.H04`, transporting to the prediction position;
  - R: the layer-4 and layer-5 MLPs at the prediction position.
- **Development data (exploratory):**
  - `L03.H04` alone recovers 0.816, and freezing it blocks 0.865;
  - `m_T` 0.90 and `m_R` 0.99;
  - `L00.MLP` carries 0.535 of the transport-input axis;
  - the three-component set (k = 3) recovers 0.982, and isolation retains 0.783 (C002:49).
- **Calibration** (19 held-out nouns, 114 cases): P1 0.977, P3 0.782, P4 0.882, P5 0.816 / 0.865, P8 0.823, P9 0.987 /
  0.900 / 0.074 (C002:30, 50).
- **The supported conclusion:** recovery "about 98%", isolation "about 78%", the head "about 82%" (C002:151).
- **Second routes are small** on C002's pairs: `L05.H07` alone 0.122 (blocking 0.128), `L00.H05` 0.028 (C002:136).
- **Kind of evidence.** This is interventional localization by patching and pair-centred neutralization. It is not
  causal evidence on the ladder: the one confirmation failed its behavioral precondition, and 007's families passed on
  fresh data at criterion level only, with its composite route unmet (C002:96, 101–106).
- **Recorded failures that travel with it:**
  - S1: the templates' readout axes reach a minimum cosine of 0.292, against 0.70 (C002:129);
  - the cue lexicon's Spearman is 0.699, against 0.70 (C002:51);
  - for new cue classes, `L00.MLP` alone is not the whole cue stage at the pair level (C002:159).

### 1.2 Prospective prediction: C003 (`LOCALIZED`; predictive, not causal)

C003 claims that frozen programs, built from the weights, each frame's measured reference state and quantities locked
from exposed data, predicted the effects of a cue change:
- under the E-patch, stage by stage;
- under real cue swaps, the contrast change `Δc` (C003:37–45).

"It does not claim that the programs' variables are the model's causal variables." (C003:75)

| stage (experiment) | headline result | status | source |
|---|---|---|---|
| the head's read-out (009, 011) | Spearman and MAE 0.957 / 0.066 on 138 pairs; 0.974 / 0.076 on 142 | prospective, given the measured input | C003:168, 170; E011:87–89 |
| the encoding read predicts transport (011) | 0.848 / 0.077, 24 new cues in 6 new frames; `an` predicted 0.18, measured 0.30, the lowest | prospective; locked prediction | C003:170 |
| the layers-1–2 net correction (012) | 0.864 / 0.061, R² 0.795 | prospective, on exposed bases | C003:171 |
| the heads' value paths (013) | Y1 0.877 (R² 0.668); Y2 0.919 (R² 0.747) | Y2 conditional | C003:171 |
| neuron 1987 (014) | R² 0.970 / 0.957; the axis-only rival (R² −1.41) rejected | Y2 conditional | C003:172 |
| the attention-pattern change (015, 016) | 015's Y2 **failed** (R² 0.490 vs 0.50); 016 reaches R² 0.994 / 0.996 | mixed | C003:173–174 |
| the head's own pattern (017) | rows R² 0.997 / 0.998; ΔT 0.999 / 1.000; the frozen-pattern rival **not rejected** (0.958) | Y3 unestablished | C003:175–176; E017:126–128 |
| channel D (018, 019) | the locked 256-neuron subset closes κ 0.863 / 0.724; one new frame **failed** the guard (0.745 vs 0.847); a single neuron (0.219) rejected; routing Δκ₆₄ 0.090 / 0.144, in aggregate | mixed | C003:177–179; PAPER:395–396 |
| `Δc` on 020's wiring (021) | Y1 token-mean R² 0.016018, against a floor clamped to 0.0; Y2 coordinated −0.1538 **failed** | small margin; unestablished; erratum | C003:182–183, 189 |
| where the upstream error lies (022) | block-0 attention carries 0.948 / 0.938 of the cue-final gap; the layer-1–2 reductions at most 0.007 | prospective, per condition | C003:184 |
| the completed program (023) | g 0.9987, 0.9974 and 0.9966 pass; Y1/cue_final 0.99552 against F 0.99690, `ENVELOPE_ONLY_FAILURE` | no aggregate label | C003:185–186 |
| the readout error (024) | ρ 0.611 against 0.314; the E–N guard K = 1 of 12,870 | "predictive, not causal" | C003:187; E024:373 |

**Compact programs failed** (C003:166, 169; E006:72; E007:87):
- 005: `NO_COMPACT_MECHANISM`;
- 006: `QUALITY_GATE_FAILED`;
- 007: Y1 0.785, against 0.80;
- 009: the rank-1 transport rule, a rule for one stage (C003:338), 0.765 against 0.80.

Compact rules hold for parts only (C003:301–303).

### 1.3 Causal evidence: none

- **No necessity, sufficiency or control gate is checked.** Neither claim checks one (C002:98–106; C003:254–274); only
  the preregistration and the two reporting gates are checked (C003:273–274). "No
  preregistered, outcome-bearing necessity test in the model targeted a decoded variable." (C003:259–260)
- **The E-patch predictions of 009–019 do not count.** They forecast the effects of an intervention, but were ruled "not
  sufficiency evidence" (C002:104; C002REV:383).
- **Whether the ladder should credit prospective prediction** was left as "a separate methodology question"
  (C002REV:419).

### 1.4 Novelty

The literature review rates the protocol and the application "partly new, narrowed". The record does not fill the audit's
multi-component gap (LIT:23–55; READY:160–163).

## 2. The four causal routes

"Four preregistered routes to causal status were designed. … Three ran, and none established causal status"
(PAPER:516–518; SYN:83–87).

### Route 1: 005's confirmation (protocol v2, one run at `30e8b84`)

**What it intervened on.** C002's components (DS005:813–823):
- exact counterfactual replacement: P1, P4 (E alone), and P5 (T alone, and its blocked fraction);
- pair-centred neutralization: P3 (everything outside the circuit) and P8 (T or E);
- cross-cue and cross-frame resamples: P6 and P7;
- chain mediation: P9.

**Thresholds** (DS005:813–822):
- the B1 precondition: at least 103 of 120 conditions correct and at least 96 of 120 flips;
- P1: recovery of at least 0.70;
- P3: F of at least 0.50, with signs kept in at least 84 of 120 pairs;
- P8: a loss of at least 0.30.

**Observed.**
- B1 reached 98/120 and 75/120. The outcome was `BEHAVIOR_NOT_REPLICATED`, so no other family is confirmatory; 44 of the
  45 failures were plural conditions (E005:125–129).
- Recorded without outcome force: P1 0.985; P3 0.795, with retention 71/120; P8 loss 0.826; P9 0.987 / 0.900 / 0.074
  (E005:130–135).
- The compensation ratio, −6.038, missed its band [−5.842, −5.642] (FR005:120, 147).

**What it teaches.**
- The failure was behavioral and specific to the split. The reserve nouns flipped cleanly in 75/120 pairs, against
  98/120 on the holdout (C002:164), so the absolute counts in B1 and P3's 84/120 were out of reach.
- It is "inconclusive, not falsified" (DS005:867–868).
- [proposal] Gates should be set relative to each split's clean behavior, as 006's design already did (DS006:268–275).

**[AI observation] The self-repair measure was ill-posed for E and T.**
- The ratio is Σ_{j≠k}(DE′_j − DE_j)/DE_k, with "near 0 … no repair; near 1 … full backup" (DS005:911–915).
- The reported value is the mean, over the six manifest frames, of per-frame ratios that neutralize T in the two
  coordinated frames and E in the four cue-final frames (DS005:208–217, 822;
  `src/neural_decompiler/plural_mechanism.py:2512–2539`). It is not specific to T.
- Values of −5.742 (calibration) and −6.038 (confirmation; FR005:120, 147) mean that, on average, the other components'
  direct effects fell by about six times the neutralized component's own direct effect. That is consistent with
  downstream mediation, since E and T both act mostly through later components (`m_R` 0.987, FR005:148), not with
  compensation.
- A ratio normalized by a direct effect is ill-conditioned for a component that acts indirectly. The record reports no
  T-specific ratio and no co-neutralization result, so it contains no valid self-repair bound for `L03.H04`.

### Route 2: 006's circuit families (never ran)

- **The design.** The families were to supply "the confirmatory sufficiency (P1, P4, P5), necessity (P8), control (P7),
  and chain (P9) evidence that C002's Level 3 gates currently lack" (DS006:360–362). They were to run on fresh nouns and
  frames, with P3 redefined as fidelity: `corr(d_iso, d_clean) ≥ 0.90` (DS006:276–283).
- **What happened.** The program's quality gate failed before the lock: normalized LOCO RMSE 0.623, against 0.50. No lock
  and no confirmation followed (E006:67–72).
- **What it teaches [AI observation].** The causal route was tied to a pre-lock quality gate on the separate
  compressed program (DS006:252–264). 006's own exploration
  had already forecast P3's failure: a correlation of 0.756, "(the 0.90 confirmation floor would not be met)"
  (E006:58–59).

### Route 3: 007's circuit families (one run at `ede0f53`, reported without outcome force)

- **The rule.** C002 is reviewed for promotion if every family passes on both fresh sets; otherwise it is unchanged
  (DS007:248–252).
- **Observed** (FR007:99–115; E007:98–100):
  - P1 0.981 / 0.983; P4 0.887 / 0.913;
  - P5 0.825 / 0.865 and 0.793 / 0.845;
  - P7 −0.010 / 0.984 and −0.000 / 0.983;
  - P8 0.822 / 0.873; P9 passed;
  - **P3 failed** on its correlation, 0.696 / 0.840 against 0.90, with F 0.781 / 0.773 and 120/120 signs.
- **What it teaches.**
  - The only failing criterion is pair-level fidelity, "a stronger property than this claim states" (C002:55). It is
    not one of the families that C002 maps to its gates: necessity P8, sufficiency P1, P4 and P5, controls P2, P6 and P7
    (C002:101–106).
  - The 0.90 floor was carried over unchanged (DS007:248–249), although 006 had forecast its failure (E006:58–59).
  - [AI observation] Seven families on two sets, all required (DS007:248–252), made the route's chance of success a
    product of many probabilities.
  - [AI observation] This is an over-demanding composite gate. The record calls it a route not met, not a falsification
    (C002:54–55).

### Route 4: 025's rotation of the cue embedding (one run at `6f72d7d`)

- **What it intervened on.** The cue's input embedding at `("EMBED", p_c)`, rotated without changing its norm along
  024's nounness direction. The odd score component is ±0.32 (primary) and ±0.16 (E025:71–75). The dose 0.32 is 024's
  E-minus-N class difference in mean score (DS025:87).
- **The outcomes** (E025:84–87):
  - ℓ, the log pooled error of the frozen block-4/5 readout approximation;
  - `D_attn`, the distance of the L4/L5 rows from 020's locked rows.
- **Controls:** 7 random, nounness-neutral tangent directions at the same angle; a plurality control, secondary
  (E025:76–79).
- **Statistics.** A (+θ against −θ), B (A beyond the random controls' magnitude) and G (the routing summary) each need
  at least 27 of 40 positive, a tail of 0.01924. A failure of A fixes the label (E025:88–92, 256).
- **Observed.** A 26/40, B 23/40, G 37/40. The label is `CAUSAL_EFFECT_NOT_ESTABLISHED`, "not a finding of no causal
  effect" (E025:15–21, 250–255).
- **Descriptive, no outcome force** (E025:260–277; FR025:88–90):
  - adjectives against nouns: A 20/20 and 6/20; B 20/20 and 3/20; G 20/20 and 17/20;
  - the causal fraction 0.341; Spearman(A, s₀) = −0.83;
  - the nounness direction's cosine with plurality, +0.235;
  - +θ raised the error above baseline in 23 of 40 cues, and −θ lowered it in 30;
  - the even components were 0.0144 for the noun rotation and 0.0141 for the random ones.
- **What it teaches:**
  1. **The outcome variable [AI observation].** What was scored is "the error of a frozen approximation", not rerouting
     (E025:275–276): a model of the model.
  2. **Heterogeneity and saturation [AI observation].** 024's association may be largely a contrast between classes (no
     within-class ρ is recorded for the fresh cues; in calibration the within-stratum ρ were 0.26–0.43, FR024:13): mean
     MSE N 0.0714 against E 0.1250 (E024:310), and D_EN 0.568 (E024:277–279). An association between units did not
     become a dose-response within each unit for cues already near the noun centroid (Spearman(A, s₀) −0.83).
  3. **No calibration phase** (E025:120), so the per-unit sign rate was never measured; the design's power table assumed
     it (DS025:379–401).
  4. **A knife edge:** A missed its threshold by one cue.

  [AI observation] The odd/even decomposition looks reusable: the even parts were nearly equal for the noun and random
  rotations, 0.0144 and 0.0141 [descriptive] (FR025:90), as expected if the even part carries generic disruption.

### Across the routes [AI observation]

- **Routes 1–3 never failed a primary criterion that C002 maps to necessity, sufficiency or control.** They stopped on a
  behavioral precondition, a fidelity correlation and a program gate. Only 025 failed at a margin where sample size
  could matter (A 26 vs 27).
- **The one decoded-variable route** intervened two stages upstream of its outcome, and scored a program's error.
- **What follows.** A confirmation of P1, P4, P5, P7, P8 and P9 under a new protocol, on fresh data, with P3 descriptive
  and a precondition normalized by clean behavior, would have a high prior of passing. The choice follows knowledge of
  which criterion failed, so it needs the explicit rationale that METH requires (METH:89).

## 3. Limitations and threats to validity

1. **The axis comes from the pair being edited.** d̂_T is the mean plural-minus-singular head output over the exposed
   one/two and each/several pairs (D101:64–66; R101:75–80), and 101 edits one/two.
2. **Exposure and reuse.** Every C002 split shares twelve prompts (C002:163). The same 79 exposed nouns are scored in
   every experiment since 020 (D101:101–102). Confirmation sets were inherited unexecuted, 006→007 (E006:89–92) and 020→021 (E020:184–186; E021:28).
3. **Conditional frames.** From 013 on, new-frame results are conditional on measured reference states (C003:125–126, 412).
   Their tables were held by digest, "not committed before scoring" (C003:132–137; MAP:678). "Nothing here predicts a
   frame's attention state from its text alone." (C003:90–91) The programs do not run without the network
   (PAPER:31–35, 365).
4. **Recomputation** is an open rival (C003:331–337).
   - With exact inputs, the reduced layers 1–2 reach R² 0.9604 against the exact chain's 0.9607 in cue-final frames,
     and 0.9648 against 0.9652 in coordinated ones (FR022:66–67).
   - At block 0, the exact rule is what completes the program [descriptive]: in Y1 cue-final frames, cheaper rules
     recovered g 0.5045 (value-only) to 0.9145 (six heads) against 0.9955, but 0.9246 to 0.9865 in Y1 coordinated
     frames (FR023:68–69; E023:180–181).
5. **Frozen attention.** Block-4 attention is held at the reference, and block 5 keeps its reference rows (C003:64–72).
   Block-4 attention carries 0.58–0.73 of the readout error by class (E024:324). The ceilings are 0.9760 on exposed data
   (E020:132) and 0.9683 / 0.9652 on fresh data (E021:140–141).
6. **Numerical floors.**
   - Instrumented and plain forwards differ by up to 4.5e-3 nats (C002:52).
   - Float32 attention-score offsets (‖scores‖∞ medians 2.0e4–1.2e5) forced the Level-1 tolerance from 1e-3 to 7e-3
     (E020:74–80).
   - 101's Level-1 contrast tolerance is 2e-2. Before calibration the recorded maximum was 0.0119, a margin of
     1.7–1.85× (P101:719–729). 101's calibration ran the layer-3 edits on exposed frames: maximum 1.065e-2 (1.9×;
     `two`'s arms ≤ 5.01e-3), and every G4 tolerance held (E101:53–54;
     `experiments/101-read-component-intervention/evidence/calibrate-2026-10-14/README.md:67–78`).
7. **Floors and knife edges.**
   - Several floors were calibrated against the program's own exposed performance, and some were clamped to zero
     skill (C003:415–416). Exposed-like sets pass all three of 021's families only 16.6% of the time (E021:146–147).
   - Nine outcomes, in eight experiments, sit at or near their thresholds:
     - 005 X3, 0.699 vs 0.70 (E005:140);
     - 007 Y1, 0.785 vs 0.80 (E007:87);
     - 009 Y2, 0.765 vs 0.80 (E009:70);
     - 015 Y2, R² 0.490 vs 0.50 (E015:110);
     - 017 Y3, 0.958 vs 0.95, with a gap of 0.042 vs 0.05 (E017:127–128);
     - 021 Y1, 0.016018 vs 0 (E021:116);
     - 023, a pass by +7.979e-5 (E023:157) and a miss at 0.99552 vs 0.99690 (E023:153–155);
     - 025 A, 26 vs 27 (E025:17).
   - Single-run designs forbid reruns, so none of these is replicated. Binary labels at knife edges carry little
     information, and interval estimates would carry more [AI observation].
8. **Intervention validity.** The E-patch builds a state that "need not occur naturally" (PAPER:575). 025's rotation is a
   subspace intervention (PAPER:573–574). Necessity rests on pair-centred neutralization (PAPER:567–572), and 005's
   secondary compensation band was missed (PAPER:571–572); [AI observation] the record holds no valid self-repair
   bound for `L03.H04` (§2, Route 1).
9. **Scope.** One checkpoint, three templates, single-token regular nouns, and 23–40 fresh cues and 6–18 new frames per
   experiment (C003:408–411). Seeds differ by near-random rotations (PAPER:544–547). Under real swaps only `Δc` was
   scored (C003:397–398), and no standard baseline was run (PAPER:555–559; READY:266–267).
10. **The process.**
    - [AI observation] The sequence was adaptive: each design followed the previous result. 005's eligibility rule was
      amended after Tier A had been inspected (MAP:265).
    - Reviews were by separate instances of the same AI agent, and ChatGPT proposed the reviewer's decisions
      (PAPER:37–56). Committed review records start at 023 (PAPER:847).
    - MAP lists inconsistencies in the records; for example, 010's README says "320–700 of 2048" where its report shows
      726 (MAP:653–698).
11. **No absence claims.** No route can show that an effect is absent; only an equivalence bound could bound one
    (READY:321–324).

## 4. Experiment 101

### 4.1 Status

- **The design** is frozen at revision 3, decided on 2026-10-02 (D101:3–8, 503–549).
- **The implementation** passed its review and re-check (`4795f19`, `40829d2`).
- **The README** says CALIBRATED (E101:3–9).
- **The freeze** ran once on 2026-10-14, 08:14:27–08:14:42 UTC, at `40829d2` (tokenizer only), and wrote
  `confirmation-v1.json`. The AI freeze review found PASS WITH NOTES, and the file was installed in `a857f9a`
  (E101:45–49; `experiments/101-read-component-intervention/evidence/freeze-2026-10-14/README.md:3–9, 57`).
- **The calibration** (exposed, exploratory; spent data) ran once at `8f65062`: no revision trigger, every gate held,
  τ_rem 0.0252 and τ_ins 0.0351. The AI review found PASS WITH NOTES, and `calibration-v1.json` was installed in
  `d502777` (E101:5–8, 50–57).
- **Not run at HEAD `3621eec`:** lock, stage 1 and confirmation; no 101 prompt has been executed (E101:8–9).

### 4.2 What it tests

- **The variable.** The read covector `w_c = γ₃⊙m − mean(γ₃⊙m)·1` of `L03.H04`'s P1 read, with ‖w_c‖ ≈ 0.766
  (D101:48–59).
- **The site.** RESID_PRE.L3 at `p_c`, in 36 new coordinated frames, with the pair one/two (D101:89–102).
- **The arms** (D101:121–135). NEC-READ and SUF-READ remove and install the read part. The primary controls, NEC-OFF and
  SUF-OFF, move an equal centred length along the frame's own off-read change. NEC-AMP and SUF-NEG check the direction.
  Secondary: 8 isotropic directions, PC-SWAP, the half doses and the head-only arms.
- **The decision order** (D101:248–279): R and I by Holm (26 and 25 of 36, medians ≥ 0.05); then D (25 of 36 for each
  sign); then P (27 of 36 frames within max(τ, 0.02), τ capped at 0.20). The precondition is at least 30 of 36 frames
  with `Δc_full` ≤ −1 nat.
- **Power.** At an 80% per-frame sign rate, R and I pass together with probability about 0.915 (D101:274–276). I
  recomputed the binomial arithmetic: tails 0.01441 (25 of 36) and 0.00567 (26 of 36); joint pass 0.9148 at p = 0.8 and
  0.9998 at p = 0.9.

### 4.3 The labels and what each would change

The design states:
- an established effect gives "causal evidence for this stage"; a claim scoped to that stage "could then be reviewed",
  as a separate decision (D101:39–41);
- a negative result is never "no causal role", and there is no equivalence bound (D101:42–44);
- a positive result supports a claim scoped to the read component at that site, in those frames, for that pair;
  attribution to `L03.H04` needs the head arms, and generality needs `three` and `many` (D101:362–369, 538–547).

Rows marked [AI observation] are not stated in the design.

| label (D101:281–291) | consequence |
|---|---|
| `NOT_INTERPRETABLE` | No result: the incident procedure applies, and no frame is dropped (E101:67–68). A retry needs a new protocol version (E005:142). |
| `BEHAVIOR_PRECONDITION_NOT_MET` | Inconclusive, as 005 was; the boundary is unchanged [AI observation]. |
| `READ_COMPONENT_EFFECTS_NOT_ESTABLISHED` | "The boundary stays where it is"; never "no causal role" (D101:42–43). |
| `…REMOVAL_EFFECT_INSTALLATION_NOT_ESTABLISHED` or `…INSTALLATION_EFFECT_REMOVAL_NOT_ESTABLISHED` | One-sided evidence. Level 3 needs both arms and their controls (METH:83–85), so there is no promotion. A failed R with a passing I is the self-repair pattern the design anticipates (D101:458–459) [AI observation]. |
| `…DIRECTION_NOT_ESTABLISHED` | Disruption is not excluded, or saturation interfered (D101:462–464; R101:178–179). The control evidence is weaker [AI observation]. |
| `…PROGRAM_NOT_EVALUABLE` | The effects are established. Nothing is learned about whether the program predicts a new intervention. |
| `…PROGRAM_NOT_CONFIRMED` | The effects are established, but the frozen readout fails to predict an artificial edit. That is evidence against C003 on Level 6's criterion of predicting new interventions, which was already only partly met (C003:295–297) [AI observation]. |
| `READ_COMPONENT_EFFECTS_AS_PREDICTED` | Causal evidence scoped to the stage, plus one prospective prediction of a new intervention, which bears on Level 6. If the head arms do not support the head, the effect "cannot be assigned uniquely to `L03.H04`" (D101:546–547). |

### 4.4 What 101 cannot establish

- absence of an effect, or equivalence (D101:44);
- anything beyond one/two in coordinated frames: `three` and `many` "cannot rescue" a failed primary (D101:508–510);
- that `L03.H04` specifically mediates the effect, which is secondary (D101:518–521);
- anything about cue-final frames;
- the causal status of the other decoded variables: the opposing component, channel D, block 0's terms and the readout's
  routing;
- validity on the natural distribution, because the edited states are artificial (D101:455–457);
- C002's component-level Level-3 gates, because 101 edits a direction in the residual stream, not C002's components [my
  reading];
- anything about other models.

### 4.5 Risks the reviews did not raise [AI observations]

1. **No diagnostics for the subspace illusion.**
   - The project's own literature review advised "nullspace and rowspace checks for subspace interventions (B8)" and
     null directions matched to the real edit (LIT:335–338).
   - B8's illusion pairs a varying, "causally disconnected" direction with a connected but "normally dormant" one
     (LIT:692–696).
   - 101 already meets two of B8's other recommendations (LIT:695–696): its direction is derived from a circuit, and
     its site is a bottleneck, since the cue's information is consumed at layer 3 (E005:92–93).
   - For the head itself, `w_c` is the exact read, so there is no illusion at the head. The residual edit also reaches
     MLP3, the other heads and `L05.H07`, which attends to `p_c` at 0.999 (DR005:40). Those readers may respond to
     `w_c`-aligned states that natural variation never produces.
   - D101, R101 and P101 never mention nullspace, rowspace or dormant-direction checks; I searched all three, and the
     committed freeze and calibration reviews. 101 does use null directions matched in length to the real edit: the
     off-read slice and 8 isotropic directions (D101:127–134, 156–157).
   - A cheap addition: report the fraction of `w_c` that lies in the span of the natural cue-induced `Δx` at `p_c`
     across the exposed cues, and use the head-arm shares to split each arm's `Δc` into the part through the head and
     the rest.
2. **The manipulation check is one-dimensional.**
   - Only ⟨ΔT, d̂_T⟩ is reported (D101:306–307; `src/neural_decompiler/read_intervention.py:1448–1456`).
   - The record shows that `Δc` depends on more than that projection:
     - the readout comparator using `L03.H04`'s ΔT alone reached R² −0.2016 on exposed data (E020:135), and −0.5382
       and −0.2027 on fresh data (E021:143–144);
     - the templates' readout axes reach a minimum cosine of only 0.292 (C002:129).
   - The effect is likely small for the defining pair, but it is unmeasured. Report the full change of the head's output
     for each arm: its norm and its cosine with d̂_T.
3. **No site or layer negative control.**
   - Every control acts at the same site.
   - In 005, patching `x(p_c)` recovered 0.956 at the layer-3 input but only 0.125 at the layer-4 input (DR005:34–35;
     E005:92–93).
   - The same `w_c` edit at RESID_PRE.L4 `p_c`, or at a name token's position, would separate the read at the site
     where it is consumed from any `w_c`-aligned perturbation of `p_c`.
4. **The primary label adds little information at the margin.**
   - Given its measured input, `L03.H04`'s output is a near-exact linear read-out: 009's patched-attention level reached
     0.998 / 0.006 (E009:62–64). Moving ⟨x, w_c⟩ therefore moves the head's d̂_T output almost by construction.
   - 005's `m_R` of 0.99 (E005:90–93) makes a downstream response likely.
   - 101's exposed calibration found N > 0 and S > 0 in 36 of 36 frames [exploratory]
     (`experiments/101-read-component-intervention/evidence/calibrate-2026-10-14/README.md:90–93`). If the fresh frames
     behave alike, the pass probability is near 1.
   - The label then mainly turns exploratory evidence into confirmatory status. The questions that remain open, whether
     the head is specific and whether `three` and `many` follow, carry no label force (D101:297–304).
5. **The inferential population is not stated.**
   - The frames are the first 36 of one hand-written list of one construction (D101:93–94, 385–419), and they share cues,
     nouns and `w_c`.
   - 025 said its sign tests were "not unconditional, design-based randomization tests" (FR025:17); 101 says nothing
     comparable.
   - The median floor of 0.05 is small beside the expected size: the head carries about 0.82 of the shift (C002:151). A
     pass certifies consistency, not magnitude, and the bootstrap intervals are descriptive (D101:271–272).

**What the reviews did address:**
- the edit reaching every reader of the residual, off-distribution states, self-repair, the route through `p_t`, and
  saturation (D101:451–466);
- the G4 tolerance margin (P101:719–739).

## 5. Open questions, ranked by expected information gain [proposals, not decisions]

1. **Do the bespoke stage programs beat generic, automatable approximations?**
   - *Hypothesis:*
     - Fed the measured `Δx3`, a first-order Taylor expansion of c around each frame's reference state at RESID_PRE.L3
       comes within 0.02 of 023's frozen-readout ceiling R² (0.9531–0.9613; FR023:30–33).
     - End to end, attribution patching from the cue position reaches R² ≥ 0.90.
   - *Falsified* if either falls short in two or more of the four conditions.
   - *Cheapest test:* exploratory, with Jacobian-vector products on 023's 3,060 spent prompts (E023:159), then one
     preregistered head-to-head.
   - *Why first:* it fills the paper's admitted gap (PAPER:555–559), and decides whether automation should build on
     attribution or on program synthesis.
2. **Does prospective accuracy imply interventional validity beyond the transport read?**
   - *Hypothesis:* installing or removing the contribution of channel D's locked 256-neuron subset (κ 0.863; C003:177)
     moves the transport read by at least 0.70 of channel D's effect in at least 26 of 36 frames, beyond 018's random
     and bottom-256 subsets.
   - *Cheapest test:* 018's `MaskedChainModel` (`src/neural_decompiler/block_concentration.py:364`) with 101's run
     harness, on exposed frames first. READY names this its secondary candidate (READY:351–353).
   - *Why:* 101 tests the most favoured variable. A distributed variable tests whether prediction tracks causation.
3. **Does the E→T→R motif, with its weight-only read, hold in other models?**
   - *Hypothesis:* in Pythia-160M-deduped, one head carries at least 0.6 of the coordinated one/two shift (005: 0.816,
     DR005:51), and its weight-only read predicts the transport of at least 20 new cues with Spearman ≥ 0.8 (011:
     0.848).
   - *Falsified* if no head exceeds 0.4, or the Spearman is below 0.7. Repeat on a second Pythia-70M seed.
   - *Cheapest test:* `plural_mechanism.a2_position_map` and `a6_chain` (lines 1384 and 1590) and
     `encoding_read.EncodingRead` (line 93), exploratory first. The code supports GPT-NeoX models only
     (`src/neural_decompiler/components.py:401`).
4. **Is there a routing state that can be manipulated?** This bridges to the second series.
   - *Hypothesis:* for fresh cues whose nounness lies below 024's calibration maximum, +0.135 (FR024:15), rotations of
     +0.32 against −0.32 raise `D_attn` (G) and A in at least 27 of 40 cues, beyond the random controls.
   - *Cheapest test:* `cue_rotation.py` (lines 441 and 481), with G as a co-primary statistic and an exposed calibration
     phase; about 2 hours of CPU (025's confirm took 1 h 49 min:
     `experiments/025-nounness-direction-intervention/evidence/confirm-2026-09-26/README.md:8`). It needs 40 new fresh
     cues, and the curated pool is nearly used up (DS025:412; D101:108–110).
   - It must run under a new protocol. The population follows 025's descriptive split, and that must be disclosed.
5. **Can the programs run without the network?**
   - *Hypothesis:* the reference-state inputs can be predicted from the frame's text, with a loss of at most 0.05 of R²
     on new frames.
   - The existing comparators suggest a loss of about 0.1: template bases reach 0.5721 against 0.6891 (E020:134;
     FR020:23), and 0.5115 / 0.4838 against 0.6167 / 0.5797 (E021:132, 137, 143).
   - *Test:* leave frames out, on states already captured; then preregister.
6. **Automated rediscovery.**
   - *Hypothesis:* run blind, attribution patching followed by a weight-derived OV read of the top head puts C002's
     components in its top five (compare DR005:25–27) and reproduces 011's transport ranking on the 24 spent cues with
     Spearman ≥ 0.8.
   - Spent data only; hours of CPU.
7. **The compression frontier.**
   - *Hypothesis:* each exact step of 023's program has a rule with at most k parameters that loses at most 0.02 of g.
   - At block 0, every tested cheaper rule misses the 0.02 bound in Y1 cue-final frames (the best, six heads, g 0.9145
     against 0.9955), but six heads meets it in Y1 coordinated frames (0.9865 against 0.9987) [descriptive]
     (FR023:68–69). This bounds those rules, not every rule with at most k parameters.
   - *Test:* description length against accuracy, stage by stage, on the stored cells. C003 names a related test: each
     stage's reduced rule against the exact computation, on fresh data (C003:337).

## 6. Reusable tooling in `src/neural_decompiler`, and the gaps

The package has 34 modules and 37,796 lines. Most modules hold one experiment's code, with its constants frozen.

| capability | entry points | notes |
|---|---|---|
| pinned loading, capture, exact interventions | `models.py`; `capture.py:145` `run_capture`; `interventions.py:216` `run_interventions`, with ZERO and REPLACE operations and DIRECT, REFERENCE, MEAN or RESAMPLE sources | general within TransformerBridge |
| decompositions and provenance | `components.py:354`, the per-head attention identity, and `components.py:401`, the GPT-NeoX parallel-block identity; `provenance.py:159, 193` | the block identity is specific to Pythia |
| behavior screening | `candidate_screening.py:358` Wilson bound, `:553` aligned patch shift, `:653` compactness gate against random sets | reusable to screen the second series' behavior pairs |
| localization | `plural_mechanism.py:1206` direct effects through the final LayerNorm, `:1384` position map, `:1590` mediation, `:1711` neutralization, `:1780` isolation | the closest thing to circuit discovery; exhaustive and tied to the task |
| weight-derived reads | `head_transport.py:147` `ov_levels`; `read_assembly.py:220`; `encoding_read.py:93` | general OV-read machinery |
| stage programs | `layer_correction.py:170`; `attention_patterns.py:153` `LayerProgram` (LayerNorm → QKV → partial rotary → softmax); `frame_channels.py:170`; `head_pattern.py:331`; `neuron_feature.py:176`; `block_concentration.py:364, 727` (κ ladders); `block_routing.py:453, 734` | building blocks for a program representation |
| end-to-end composition and attribution | `readout_decompilation.py:379` `ReadoutProgram`, `:1269`; `upstream_localization.py:673, 697, 709, 837, 850, 1361` (exact five-factor Shapley attribution) | reusable composition and attribution |
| statistics and protocol | `readout_calibration.py:337, 419`; `upstream_localization.py:395` envelopes; `readout_routing.py:481, 570` permutation nulls; `cue_rotation.py:143, 460` exact binomial tails and SHA-256 controls; `read_intervention.py:278, 637, 650` Holm thresholds and covector edits | strong integrity tooling |

**The gaps:**
- **Representation:** no shared program representation or evaluator.
- **Discovery:** no automated search (AtP, EAP or ACDC) and no hypothesis generation.
- **Causal validation:** no interchange-intervention or DAS harness, no rowspace diagnostics, no co-ablation tooling for
  self-repair.
- **Portability:** locked to GPT-NeoX, on CPU, in float32.
- **Independence from the network:** no predictor of a frame's reference state.
- **Baselines:** no harness.
- **Duplicated utilities:** `tensor_digest` (`provenance.py:159`, `readout_calibration.py:146`), `spearman`
  (`plural_mechanism.py:2627`, `readout_routing.py:481`), and three implementations of `draw_indices`.
- **Runtime:** 025's confirmation took 1 h 49 min for 90,720 patched forward passes
  (`experiments/025-nounness-direction-intervention/evidence/confirm-2026-09-26/README.md:8`); the design estimated
  2.5–3.5 hours (E025:160).
