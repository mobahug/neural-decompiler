# VERIFY: fact-check of `research-state.md`

**Checker:** an independent, adversarial, read-only AI fact-checker (Claude), 2026-10-14. This is an AI check, not a
human validation.

**Memo checked:** `scratchpad/exp101/survey/research-state.md`, 480 lines. "line" below means a line of the memo.

**Method.**
- Every numeric claim and quoted phrase in the header, §0 (which restates §1–§3), §1, §2 and §3 was opened at the cited
  file and line ± 5.
- §4.5 and §5 were checked for their factual premises, the tags across the memo, and every causal, success or result
  statement.
- §4.1–§4.4 and §6 were spot-checked.
- Only read-only commands were run: `git --no-optional-locks`, `sed`, `grep`, `cat`, and one pure-arithmetic
  `python3 -I` binomial check. No repository file was touched, and no model, runner or test was run.

**Repository state.**
- HEAD is now `3621eec` on `exp101-phases`, not `40829d2` as the memo says.
- `git diff --stat 40829d2 HEAD` touches only `experiments/101-read-component-intervention/`: the freeze was installed in
  `a857f9a` with evidence in `8f65062`, and the calibration was installed in `d502777` with evidence in `3621eec`.
- So every citation outside E101 has the same line numbers at both commits. E101's lines and §4.1's status have changed.

**The E025 offset.**
- Commit `e34f16d` inserted 4 lines after line 28 of the `d7c5e62` README, and 2 more near line 230.
- So lines 29–228 moved +4, and lines from 231 on moved +6.
- The claims cite only `025:244`, now E025:250 (+6), so the memo's note holds for them. PAPER's `025:67–69` and
  `025:116` moved +4.
- Every E025 line number the memo itself gives is correct at HEAD.

**Classes:**
- CORRECT;
- WRONG, with the correct value and location;
- MISATTRIBUTED: the right value under the wrong citation;
- UNVERIFIABLE.

Opinions without a tag are listed in §3, not counted as claims.

---

## 1. Claim table

### 1.1 The header and §0 (key points)

| ID | line | claim | cited | class | evidence and notes |
|---|---|---|---|---|---|
| H1 | 3–5 | HEAD `40829d2`; the tree is clean except for an untracked `confirmation-v1.json` | — | **WRONG** (stale) | HEAD is `3621eec`, and the tree is clean. The file was installed in `a857f9a`; `calibration-v1.json` was installed in `d502777` |
| H2 | 33–34 | E025's lines are 6 higher than the claims' `(025:L)` citations; the other READMEs are unchanged since `d7c5e62` | — | CORRECT | The claims cite only 025:244, now E025:250. The note is incomplete: lines 29–228 moved +4. `git diff --name-only d7c5e62 HEAD -- experiments/` lists only 025 and 101 files |
| K1 | 38 | both claims are `LOCALIZED` | C002:5; C003:5 | CORRECT | |
| K2 | 38–39 | the quote "The programs are prospectively predictive; their causal gates are unestablished." | C003:392 | CORRECT | verbatim |
| K3 | 38 | "**Nothing is causal.**" | — | **WRONG** (overclaim) | The record establishes no causal status, and it forbids absence readings: "It is not a finding of no causal effect" (E025:255); D101:42–43; READY:321–324. The memo's own §3.11 says the same |
| K4 | 40 | the localization rests on interventions on exploratory and calibration data | — | CORRECT | C003:165 says exactly this; the memo does not cite it |
| K5 | 40–41 | its one confirmation failed a behavioral precondition | C002:51 | CORRECT | |
| K6 | 42 | 011's Spearman is 0.848 | — | CORRECT | C003:170; E011:76 |
| K7 | 42–43 | 017's ΔT R² is 0.999 and 1.000 | — | CORRECT | E017:118, 122; PAPER:394 |
| K8 | 43 | 023's g ≥ 0.9966 in three conditions | — | CORRECT | C003:185: 0.9987, 0.9974, 0.9966 |
| K9 | 43–44 | it is conditional on measured reference states, and partly an exact recomputation | — | CORRECT (phrasing) | Block 3's heads and block 0 are recomputed exactly by construction (C003:332–334). That this explains the accuracy is an open rival (C003:331–337; PAPER:30–31 says "may"). 011's Y1 is from the weights alone (C003:170) |
| K10 | 45–47 | no route that ran failed on a criterion C002 maps to necessity, sufficiency or control | — | CORRECT | 005 stopped on B1. Its P8 passed, and only P8's *secondary* compensation band was missed (FR005:147; C002:51). 007 failed P3 only (FR007:101–102). 025's A is not a C002 criterion (C002:66). No tag here (see §3) |
| K11 | 46–47 | 025 stopped on a sign count over two strata that responded oppositely | — | CORRECT | A was 20/20 and 6/20 (E025:260). The split is descriptive, and "no stratum-level claim is made" (E025:258–261); the memo does not say so |
| K12 | 47 | 006's route never ran: a gate on the program blocked its lock | — | CORRECT | E006:67–72 |
| K13 | 48–49 | 101 tests one decoded variable at one site, with the cue pair that defined it | — | CORRECT | D101:64–66, 89–95 |
| K14 | 49 | a positive label would support reviewing a claim scoped to that stage | — | CORRECT | D101:39–41, 362–369 |
| K15 | 50–52 | the reviews did not raise the listed risks | — | CORRECT | Searched D101, R101, P101 and the committed freeze and calibration reviews. None states rowspace, nullspace or dormant-direction checks, a site control, or a population. No tag here |

### 1.2 §1

| ID | line | claim | cited | class | evidence and notes |
|---|---|---|---|---|---|
| L1 | 56–57 | Level 3 requires necessity-, sufficiency- and control interventions that support a prediction documented beforehand | METH:80–89 | CORRECT | METH:78, 82–87. The memo omits "where applicable" (METH:83–84) and the rule that another causal test may substitute, with an explicit rationale (METH:89) |
| L2 | 61–64 | three stages, E, T and R | C002:15–24 | CORRECT | It omits T's condition, "when the cue is not the final token" (C002:18–19) |
| L3 | 66 | `L03.H04` alone recovers 0.816, and freezing it blocks 0.865 | C002:49 | CORRECT | also DR005:50–51 |
| L4 | 67 | `m_T` 0.90, `m_R` 0.99 | C002:49 | CORRECT | |
| L5 | 68 | `L00.MLP` carries 0.535 of the transport-input axis | C002:49 | CORRECT | |
| L6 | 69 | k = 3 recovers 0.982; isolation retains 0.783 | C002:49 | CORRECT | |
| L7 | 70 | "19 held-out nouns" | C002:50 | **MISATTRIBUTED** | C002:50 says "114 single-token holdout cases". The 19 is at C002:30 and 94; MAP:665 records the naming inconsistency |
| L8 | 70 | 114 cases | C002:50 | CORRECT | |
| L9 | 70–71 | P1 0.977, P3 0.782, P4 0.882, P5 0.816 / 0.865, P8 0.823, P9 0.987 / 0.900 / 0.074 | C002:50 | CORRECT | also FR005:113–121 |
| L10 | 72 | "about 98%", "about 78%", "about 82%" | C002:151 | CORRECT | The 82 % is "of the coordinated-template shift" |
| L11 | 73 | `L05.H07` 0.122 (blocking 0.128); `L00.H05` 0.028 | C002:136 | CORRECT | |
| L12 | 74–75 | it is not causal evidence "because the data were exploratory or calibration data" | C002:96, 101–106 | **WRONG** (reason) | The cited gates give other reasons. E003's precondition failed (C002:101, 103, 105). 007's fresh-data P1, P4, P5, P7 and P8 passed at criterion level, but its composite route was not met (C002:102, 104, 106). And the E-patch predictions are not sufficiency evidence (C002:104). 007's data were neither exploratory nor calibration |
| L13 | 77 | S1's minimum cosine is 0.292 against 0.70 | C002:129 | CORRECT | |
| L14 | 78 | the lexicon's Spearman is 0.699 against 0.70 | C002:51 | CORRECT | also E005:140 |
| L15 | 79 | `L00.MLP` alone is not the whole cue stage | C002:159 | CORRECT | |
| L16 | 83–86 | C003's claim, under the E-patch and under real swaps | C003:37–45 | CORRECT | |
| L17 | 88 | the quote "It does not claim that the programs' variables are the model's causal variables." | C003:75 | CORRECT | verbatim |
| T1 | 92 | 0.957 / 0.066 on 138 pairs | C003:168 | CORRECT | |
| T2 | 92 | 0.974 / 0.076 "on 142" | C003:170 | **MISATTRIBUTED** | "142" is not in C003:168–170. It is at E011:87 and PAPER:388 |
| T3 | 93 | 0.848 / 0.077 on 24 cues in 6 frames; `an` predicted 0.18, measured 0.30, the lowest | C003:170 | CORRECT | also E011:75–79 |
| T4 | 94 | 012: 0.864 / 0.061, R² 0.795, on exposed bases | C003:171 | CORRECT | "Exposed bases": C002REV:410–412 |
| T5 | 95 | 013: Y1 0.877 (R² 0.668), Y2 0.919 (R² 0.747) | C003:171 | CORRECT | |
| T6 | 96 | 014: R² 0.970 / 0.957; the axis-only rival at R² −1.41, rejected | C003:172 | CORRECT | |
| T7 | 97 | 015's Y2 R² 0.490 against 0.50; 016 at 0.994 / 0.996 | C003:173–174 | CORRECT | |
| T8 | 98 | 017: rows R² 0.997 / 0.998; ΔT 0.999 / 1.000 | C003:175 | CORRECT | |
| T9 | 98 | 017's frozen-pattern rival "not rejected (0.958)" | C003:175–176 | **MISATTRIBUTED** | C003:176 gives the label only. 0.958 is at E017:127 |
| T10 | 99 | 018: κ 0.863 / 0.724 | C003:177 | CORRECT | The two values are the decoded read and the pattern term, not Y1 / Y2 (C003:177; PAPER:395). The slash notation misleads |
| T11 | 99 | one new frame failed the guard, 0.745 against 0.847 | C003:178 | CORRECT | |
| T12 | 99 | a single neuron (0.219) rejected | C003:177 | CORRECT | |
| T13 | 99 | routing Δκ₆₄ 0.090 / 0.144, in aggregate | PAPER:395–396 | CORRECT | at PAPER:396. C003:179 gives ρ, not Δκ |
| T14 | 100 | 021: Y1 0.016018 against a floor clamped to 0.0; Y2 −0.1538 failed; erratum | C003:182–183, 189 | CORRECT | |
| T15 | 101 | 022: 0.948 / 0.938; reductions at most 0.007 | C003:184 | CORRECT | The status omits "Y2 conditional; the coordinated share descriptive" |
| T16 | 102 | 023: g 0.9987, 0.9974, 0.9966; 0.99552 against F 0.99690, `ENVELOPE_ONLY_FAILURE` | C003:185–186 | CORRECT | also E023:142–145 |
| T17 | 103 | 024: ρ 0.611 against 0.314; K = 1 of 12,870; "predictive, not causal" | C003:187; E024:373 | CORRECT | also E024:277–278 |
| C1 | 106 | 005: `NO_COMPACT_MECHANISM` | C003:166 | CORRECT | |
| C2 | 107 | 006: `QUALITY_GATE_FAILED` | C003:166 | **MISATTRIBUTED** | C003:166 says only that 006 "falsified the leading ≤ 4 unsupervised PCA directions". The label is at E006:72 and 85 |
| C3 | 108 | 007: Y1 0.785 against 0.80 | C003:166 | **MISATTRIBUTED** | C003:166 gives only `PROGRAM_NOT_SUPPORTED`. 0.785 is at E007:87 |
| C4 | 109 | 009: the rank-1 rule, 0.765 against 0.80 | C003:169 | CORRECT | It is a rule for the transport stage, not for the whole mechanism (C003:338) |
| C5 | 111 | compact rules hold for parts only | C003:301–303 | CORRECT | |
| N1 | 115 | "**No gate is checked.**" | — | **WRONG** | C002:100, 107 and 108, and C003:256, 268 and 270, are checked Level-3 gates. C003:273–274: only the necessity, sufficiency and control gates are unchecked |
| N2 | 115 | neither claim checks a necessity, sufficiency or control gate | C002:98–106; C003:254–274 | CORRECT | |
| N3 | 115–116 | the quote "No preregistered, outcome-bearing necessity test in the model targeted a decoded variable." | C003:259–260 | CORRECT | verbatim |
| N4 | 117–118 | the E-patch predictions of 009–019 are "not sufficiency evidence" | C002:104; C002REV:383 | CORRECT | |
| N5 | 119–120 | "a separate methodology question" | C002REV:419 | CORRECT | |
| V1 | 124 | the literature review rates the protocol and the application "partly new, narrowed" | LIT:23–55 | CORRECT | LIT:23, 30 |
| V2 | 125 | the record does not fill the multi-component gap | LIT:53–55; READY:160–163 | CORRECT | |

### 1.3 §2

| ID | line | claim | cited | class | evidence and notes |
|---|---|---|---|---|---|
| R0 | 129–130 | the quote "Four preregistered routes … Three ran, and none established causal status" | PAPER:516–518; SYN:83–87 | CORRECT | |
| R1a | 132 | one run at `30e8b84` | — | CORRECT | E005:121–122 |
| R1b | 134–138 | the families and their interventions | DS005:813–823 | CORRECT | |
| R1c | 141 | B1: at least 103/120 correct and at least 96/120 flips | DS005:813 | CORRECT | |
| R1d | 142 | P1: recovery ≥ 0.70 | DS005:815 | CORRECT | |
| R1e | 143 | P3: F ≥ 0.50, with signs in at least 84/120 | DS005:817 | CORRECT | |
| R1f | 144 | P8: loss ≥ 0.30 | DS005:822 | CORRECT | |
| R1g | 147–148 | B1 98/120 and 75/120; `BEHAVIOR_NOT_REPLICATED`; 44 of the 45 failures plural | E005:125–129 | CORRECT | |
| R1h | 149–150 | P1 0.985; P3 0.795 with 71/120; P8 0.826; P9 0.987 / 0.900 / 0.074 | E005:130–135 | CORRECT | |
| R1i | 151 | compensation −6.038, outside the band [−5.842, −5.642] | FR005:120, 147 | CORRECT | |
| R1j | 154–155 | reserve 75/120 against holdout 98/120 | C002:164 | CORRECT | 98/120 is the screen's holdout split (DS005:107). 005's 19-noun calibration had 95/114 flips (FR005:111) |
| R1k | 156 | "inconclusive, not falsified" | DS005:867–868 | CORRECT | |
| R1m | 160 | the ratio's formula; "near 0 … no repair; near 1 … full backup" | DS005:911–915 | CORRECT | |
| R1n | 159–165 | [AI observation] −5.742 and −6.038 are T's ratio ("ill-posed for T") | FR005:120, 147 | **WRONG** (premise) | The values are right, but each is a mean over the six manifest frames of per-frame ratios. T is neutralized in the 2 coordinated frames and E (`L00.MLP`) in the 4 cue-final frames (DS005:208–217, 822; `src/neural_decompiler/plural_mechanism.py:2512–2539`). The ratio is not specific to T |
| R1o | 162–163 | `m_R` 0.987 | — | CORRECT | FR005:121, 148; DR005:49 (not cited in the memo) |
| R2a | 169–170 | the quote "the confirmatory sufficiency (P1, P4, P5), necessity (P8), control (P7), and chain (P9) evidence…" | DS006:360–362 | CORRECT | verbatim |
| R2b | 170–171 | P3 as fidelity, `corr ≥ 0.90`, on fresh nouns and frames | DS006:276–283 | CORRECT | also DS006:344–347 |
| R2c | 172–173 | normalized LOCO RMSE 0.623 against 0.50; no lock and no confirmation | E006:67–72 | CORRECT | LOCO per DS006:260 |
| R2e | 175–176 | 0.756, and "(the 0.90 confirmation floor would not be met)" | E006:58–59 | CORRECT | verbatim |
| R3a | 178 | one run at `ede0f53` | — | CORRECT | C002:54 |
| R3b | 180–181 | the promotion rule | DS007:248–252 | CORRECT | |
| R3c | 183–186 | P1, P4, P5, P7, P8 and P9 values | FR007:99–115 | CORRECT | FR007:109, 115 |
| R3d | 187 | P3 failed: 0.696 / 0.840 against 0.90; F 0.781 / 0.773; 120/120 signs | FR007; E007:98–100 | CORRECT | |
| R3e | 189 | "a stronger property than this claim states" | C002:55 | CORRECT | verbatim |
| R3f | 190–191 | C002's gate mapping: P8; P1, P4, P5; P2, P6, P7 | C002:101–106 | CORRECT | |
| R3g | 192 | the 0.90 floor was carried over unchanged | — | CORRECT | DS007:248–249 (not cited in the memo) |
| R3h | 193 | seven families on two sets, all required | — | CORRECT | DS007:248–252 |
| R4a | 196 | one run at `6f72d7d` | — | CORRECT | C002:66 |
| R4b | 198–199 | a norm-preserving rotation at `("EMBED", p_c)`; odd part ±0.32 and ±0.16 | E025:71–75 | CORRECT | |
| R4c | 199–200 | the dose 0.32 is 024's E−N difference in mean score | DS025:87 | CORRECT | 0.3201 |
| R4d | 201–203 | the outcomes ℓ and `D_attn` | E025:84–87 | CORRECT | |
| R4e | 204–205 | 7 random, nounness-neutral directions; the plurality control secondary | E025:76–79 | CORRECT | |
| R4f | 206–207 | at least 27 of 40, tail 0.01924; a failure of A fixes the label | E025:88–92, 256 | CORRECT | recomputed tail 0.0192387 |
| R4g | 208–209 | A 26/40, B 23/40, G 37/40; "not a finding of no causal effect" | E025:15–21, 250–255 | CORRECT | |
| R4h | 211 | strata A 20/20 and 6/20; B 20/20 and 3/20; G 20/20 and 17/20 | E025:260 | CORRECT | |
| R4i | 212 | causal fraction 0.341 | E025:263 | CORRECT | FR025:92 |
| R4j | 212 | Spearman(A, s₀) = −0.83 | E025:265 | CORRECT | |
| R4k | 213 | cosine with plurality +0.235 | E025:277 | CORRECT | |
| R4l | 214 | above baseline at +θ in 23 of 40; below at −θ in 30 | FR025:88–90 | CORRECT | FR025:88 |
| R4m | 215 | even components 0.0144 and 0.0141 | FR025:90 | CORRECT | |
| R4n | 217–218 | "the error of a frozen approximation" | E025:275–276 | CORRECT | verbatim |
| R4o | 220 | mean MSE N 0.0714 against E 0.1250 | E024:310 | CORRECT | |
| R4p | 220 | D_EN 0.568 | E024:277–279 | CORRECT | E024:278 |
| R4q | 219 | [AI observation] "024's association was largely a contrast between classes" | — | **UNVERIFIABLE** | No within-class ρ is recorded for the fresh cues. On calibration data, the within-stratum ρ were 0.43, 0.26 and 0.31 (FR024:13), so a within-class association existed there |
| R4r | 222 | no calibration phase, so the per-unit sign rate was "never estimated" | E025:120 | CORRECT | Better "never measured": the design's power table assumed it from 024's slopes (DS025:379–401), and ran no pilot (DS025:13–14) |
| R4s | 223 | A missed by one cue | — | CORRECT | 26 against 27 |
| X1 | 230–232 | routes 1–3 never failed on a criterion C002 maps to N, S or C | — | CORRECT | as K10. The listed order (precondition, fidelity, program gate) is not routes 1, 2, 3 |
| X2 | 233 | 025 intervened "two stages upstream" of its outcome | — | CORRECT (ambiguous) | In C003:46–73, stages 2 and 3 lie between the embedding and the readout |

### 1.4 §3

| ID | line | claim | cited | class | evidence and notes |
|---|---|---|---|---|---|
| S1 | 240–241 | d̂_T comes from the exposed one/two and each/several pairs | D101:64–66; R101:75–80 | CORRECT | |
| S2 | 242 | every C002 split shares twelve prompts | C002:163 | CORRECT | |
| S3 | 242–243 | the same 79 exposed nouns since 020 | D101:101–102 | CORRECT | |
| S4 | 243 | sets inherited 006→007 and 020→021 | E006:89–92 | **MISATTRIBUTED** (part) | 006→007 is at E006:89–92. 020→021 is at E020:184–186 and E021:28. Both sets were unexecuted when inherited |
| S5 | 244 | "From 013 on", new-frame results are conditional | C003:125–126 | **MISATTRIBUTED** | C003:125–126 defines "conditional". "From 013 on" is at C003:412, also PAPER:214 |
| S6 | 245 | "not committed before scoring" | C003:132–137; MAP:678 | CORRECT | verbatim at MAP:678; C003:135 |
| S7 | 245–246 | "Nothing here predicts a frame's attention state from its text alone." | C003:90–91 | CORRECT | C003 quoting DS016:262–263 |
| S8 | 246–247 | the programs do not run without the network | PAPER:31–35, 365 | CORRECT | |
| S9 | 248 | recomputation is an open rival | C003:331–337 | CORRECT | |
| S10 | 249–250 | 0.9604 against 0.9607; 0.9648 against 0.9652 | FR022:66–67 | CORRECT | These are Y1 values, and descriptive records |
| S11 | 251–252 | g 0.5045 (value-only) to 0.9145 (six heads) in Y1 cue-final | FR023:69 | CORRECT | The values are right; "Block 0 must be exact" overclaims (§4, O7) |
| S12 | 253 | block-4 attention at the reference; block 5 on reference rows | C003:64–72 | CORRECT | |
| S13 | 254 | block-4 attention carries 0.58–0.73 by class | E024:324 | CORRECT | descriptive |
| S14 | 254–255 | ceilings 0.9760, and 0.9683 / 0.9652 | E020:132; E021:140–141 | CORRECT | |
| S15 | 257 | instrumented and plain forwards differ by up to 4.5e-3 nats | C002:52 | CORRECT | |
| S16 | 258–259 | ‖scores‖∞ medians 2.0e4–1.2e5; tolerance from 1e-3 to 7e-3 | E020:74–80 | CORRECT | Not "forced": the user authorized the amendment (E020:73–74) |
| S17 | 260–261 | 101's Level-1: 2e-2 against 0.0119, 1.7–1.85×, "untested under layer-3 edits" | P101:719–729 | **WRONG** (stale) | True per P101:725–729. At HEAD, the calibration ran every arm, layer-3 edits included, on exposed frames: maximum 1.065e-2 (1.9×), `two`'s arms ≤ 5.01e-3, and every G4 tolerance held (E101:6–7, 53–54; `experiments/101-read-component-intervention/evidence/calibrate-2026-10-14/README.md:67–78`) |
| S18 | 263–264 | floors calibrated on "each program's own exposed performance" | C003:415–416 | **WRONG** | C003:415 says "**Several** floors", as does PAPER:552. PAPER:841–844 shows other regimes: 009–012 tolerances from Tier-A residuals, 013–019 constants |
| S19 | 263–264 | some floors clamped at zero skill | C003:415–416 | CORRECT | |
| S20 | 264 | exposed-like sets pass all three of 021's families 16.6 % of the time | E021:146–147 | CORRECT | |
| S21 | 265 | "Eight outcomes sit at their thresholds" | — | **WRONG** (count) | The list holds nine outcomes in eight experiments: 023 contributes a pass and a miss |
| S22 | 266 | 005 X3: 0.699 against 0.70 | E005:140 | CORRECT | |
| S23 | 267 | 007 Y1: 0.785 against 0.80 | E007:87 | CORRECT | |
| S24 | 268 | 009 Y2: 0.765 against 0.80 | E009:70 | CORRECT | |
| S25 | 269 | 015 Y2: R² 0.490 against 0.50 | E015:110 | CORRECT | |
| S26 | 270 | 017 Y3: 0.958 against 0.95; gap 0.042 against 0.05 | E017:127–128 | CORRECT | |
| S27 | 271 | 021 Y1: 0.016018 against 0 | E021:116 | CORRECT | |
| S28 | 272 | 023: a pass by +7.979e-5 | E023:157 | CORRECT | |
| S29 | 272 | 023: a miss at 0.99552 against 0.99690 | E023:153–155 | CORRECT | |
| S30 | 273 | 025 A: 26 against 27 | — | CORRECT | E025:17 (not cited in the memo) |
| S31 | 276 | "need not occur naturally" | PAPER:575 | CORRECT | spans PAPER:575–576 |
| S32 | 276–277 | 025's rotation is a subspace intervention | PAPER:573–574 | CORRECT | |
| S33 | 277 | necessity rests on pair-centred neutralization | PAPER:567–572 | CORRECT | |
| S34 | 277–278 | "there is no valid self-repair bound" | §2, Route 1 | **UNVERIFIABLE** | This is an AI inference stated as fact. The record says only that the secondary compensation band was missed (PAPER:571–572; FR005:147), and its reports give no co-neutralization result. The inference's premise is wrong (R1n) |
| S35 | 279–280 | one checkpoint, three templates, single-token regular nouns, 23–40 cues, 6–18 frames | C003:408–411 | CORRECT | The 6–18 frames are "in 009–023" |
| S36 | 280 | seeds differ by near-random rotations | PAPER:544–547 | CORRECT | a cited literature result (McCann, 2026), not a project finding |
| S37 | 280–281 | under real swaps only Δc was scored | C003:397–398 | CORRECT | |
| S38 | 281 | no standard baseline was run | PAPER:555–559; READY:266–267 | CORRECT | |
| S39 | 283 | "The sequence was adaptive: each design followed the previous result" | MAP:264 | **UNVERIFIABLE** | MAP:264–265 records only 005's protocol and its amended rule. No record line states this; it is the memo's own characterization |
| S40 | 283–284 | 005's eligibility rule was amended after Tier A had been inspected | MAP:264 | CORRECT | at MAP:265, quoting DS005:120–122 |
| S41 | 285–286 | separate instances of the same agent reviewed; ChatGPT proposed the decisions | PAPER:37–56 | CORRECT | PAPER:40–42, 51–53 |
| S42 | 286 | committed review records start at 023 | PAPER:847 | CORRECT | |
| S43 | 287–288 | 010's README says "320–700 of 2048"; its report shows 726 | MAP:653–698 | CORRECT | MAP:672 |
| S44 | 289–290 | no absence claims; only an equivalence bound | READY:321–324 | CORRECT | |

### 1.5 The premises of §4.5

| ID | line | premise | cited | class | evidence and notes |
|---|---|---|---|---|---|
| P1a | 359–360 | the literature review advised nullspace and rowspace checks and matched null directions | LIT:335–338 | CORRECT | verbatim |
| P1b | 361–362 | B8: "causally disconnected", "normally dormant" | LIT:692–696 | CORRECT | |
| P1c | 363–364 | B8's other recommendations: a circuit-derived direction and a bottleneck site, since the cue is consumed at layer 3 | LIT:695–696; E005:92–93 | CORRECT | |
| P1d | 365–366 | `L05.H07` attends to `p_c` at 0.999 | DR005:40 | CORRECT | D101:166–167 and 452 already state that the edit reaches it |
| P1e | 368 | "D101, R101 and P101 never mention these checks" | grep | CORRECT, needs rewording | True for nullspace, rowspace and dormant-direction checks; the committed freeze and calibration reviews do not mention them either. It must not read as covering LIT's "null directions matched to the real edit": 101 has those, the equal-length off-read slices and 8 isotropic directions (D101:127–134, 156–157) |
| P1f | 369–371 | head-arm shares as a cheap "addition" | — | CORRECT, with a note | The head arms' removal and installation shares for the read edits are already preregistered secondary statistics (D101:301–303). The calibration reports about 88–89 % through the head (calibrate README:94–95). Only extending the shares to the control arms is new |
| P2a | 373 | only ⟨ΔT, d̂_T⟩ is reported | D101:306–307; `read_intervention.py:1448–1456` | CORRECT | docstring at 1444–1446 |
| P2b | 375–376 | the ΔT-only comparator: −0.2016; −0.5382 / −0.2027 | E020:135; E021:143–144 | CORRECT | It shows that Δc depends on paths beyond the head's output, not on the off-d̂_T part of that output |
| P2c | 377 | minimum cosine 0.292 | C002:129 | CORRECT | These are readout-input axes across templates, so only loosely relevant |
| P3a | 381 | every control acts at the same site | — | CORRECT | D101:121–135, 455–456 |
| P3b | 382–383 | 0.956 at the layer-3 input, 0.125 at the layer-4 input | DR005:34–35; E005:92–93 | CORRECT | |
| P4a | 387–388 | 009's patched-attention level 0.998 / 0.006 | E009:62–64 | CORRECT | at E009:65; descriptive (PAPER:358) |
| P4b | 389 | `m_R` 0.99 | E005:90–93 | CORRECT | |
| P4c | 390 | 007's 120/120 | — | CORRECT, with a note | The value is right (FR007:109, 115), but it counts P3's isolation signs, not removal or installation signs. 101's own exposed calibration now gives N > 0 and S > 0 in 36/36 frames (calibrate README:90–93) |
| P4d | 391–392 | the head question and `three`/`many` carry no label force | D101:297–304 | CORRECT | |
| P5a | 394–395 | the first 36 of one hand-written list of one construction | D101:93–94, 385–419 | CORRECT | |
| P5b | 396–397 | 025's "not unconditional, design-based randomization tests"; 101 has nothing comparable | FR025:17 | CORRECT | grep of D101 and R101 |
| P5c | 398 | the head carries about 0.82 | C002:151 | CORRECT | |
| P5d | 399 | the bootstrap intervals are descriptive | D101:271–272 | CORRECT | |
| P5e | 401–404 | what the reviews did address | D101:451–466; P101:719–739 | CORRECT | These are the design's and the plan's sections, written after the reviews |

### 1.6 The premises of §5

| ID | line | premise | cited | class | evidence and notes |
|---|---|---|---|---|---|
| Q1a | 411 | 023's ceiling R² 0.9531–0.9613 | FR023:30–33 | CORRECT | |
| Q1b | 414 | 3,060 spent prompts | E023:159 | CORRECT | |
| Q1c | 416 | the paper's admitted gap | PAPER:555–559 | CORRECT | |
| Q2a | 419 | channel D's subset at κ 0.863 | C003:177 | CORRECT | |
| Q2b | 421 | 018's random and bottom-256 subsets | — | CORRECT | E018:15–16 |
| Q2c | 422 | `MaskedChainModel` | `block_concentration.py:364` | CORRECT | |
| Q2d | 423 | READY's secondary candidate | READY:351–353 | CORRECT | |
| Q3a | 426–427 | 005: 0.816 | DR005:51 | CORRECT | |
| Q3b | 427–428 | 011: 0.848 | — | CORRECT | C003:170 |
| Q3c | 430–431 | `a2_position_map` 1384, `a6_chain` 1590, `EncodingRead` 93 | src | CORRECT | |
| Q3d | 431–432 | GPT-NeoX only | `components.py:401` | CORRECT | `components.py:412–417` raises for anything but parallel GPT-NeoX. "The code" generalizes from the block decomposition |
| Q4a | 434 | 024's calibration maximum +0.135 | FR024:15 | CORRECT | |
| Q4b | 436 | `cue_rotation.py` 441 and 481 | src | CORRECT | `rotate` and `control_directions` |
| Q4c | 437 | about 3 hours of CPU | E025:160 | CORRECT as the cited estimate | The estimate was 2.5–3.5 h. The production confirm took 1 h 49 min (`experiments/025-nounness-direction-intervention/evidence/confirm-2026-09-26/README.md:8`) |
| Q5a | 442–443 | template bases 0.5721 against 0.6891 | E020:134; FR020:23 | CORRECT | |
| Q5b | 443 | 0.5115 / 0.4838 against 0.6167 / 0.5797 | E021:132, 137, 143 | CORRECT | |
| Q5c | 442 | a loss of about 0.1 | — | CORRECT | 0.117, 0.105 and 0.096. The comparator replaces only the readout MLPs' operating point with the template base (E020:134), not every reference-state input |
| Q6a | 447 | compare with C002's components in the top five | DR005:25–27 | CORRECT | C002's four components rank first to fourth at `p_t` over all templates |
| Q7a | 452 | "Already partly falsified at block 0", where value-only reaches g 0.5045 | FR023:69 | **WRONG** (inference) | 0.5045 is right. But a descriptive comparator for one rule (FR023:44) cannot falsify the claim that a rule with at most k parameters exists. The best tested rule reached 0.9145 in Y1 cue-final frames and 0.9865 in Y1 coordinated ones (FR023:68–69), where it meets the 0.02 bound |
| Q7b | 453 | "C003 names this test" | C003:337 | **WRONG** | C003:337 names "each stage's reduced rule scored against the exact computation it replaces, on fresh data". The memo's test, description length against accuracy on stored cells, is related but different. C003:339–340 names "a preregistered compact program for a whole stage" |

**An unaddressed premise (Q4d, §5.4).** The proposal needs 40 fresh cues of low nounness, but the record says the
curated fresh pool is nearly used up:
- "025 consumes 20 of the 23 remaining curated fresh adjectives" (DS025:412);
- "Fresh plural cues are exhausted … The only cues with a current eligibility record are eight adjectives and nouns and
  two measure nouns" (D101:108–110).

New cues would have to be curated first.

### 1.7 Spot checks outside the required scope

| ID | line | claim | class | evidence |
|---|---|---|---|---|
| O-1 | 298 | the README says "No phase has run" (E101:3–4) | **WRONG** (stale) | E101:3–9 now reads "CALIBRATED — the freeze and the calibration ran once; the lock is next". The phrase no longer occurs in E101 |
| O-2 | 299–303 | the freeze's review verdict and install commit are placeholders; git shows the file untracked | **WRONG** (stale) | The freeze review found PASS WITH NOTES, and the file was installed in `a857f9a` (E101:45–49). The committed `evidence/freeze-2026-10-14/README.md:3–9, 57` gives the same run, 08:14:27–08:14:42 UTC |
| O-3 | 304 | "Nothing else has run: no calibration…" | **WRONG** (stale) | The calibration ran once, at `8f65062`, and was installed in `d502777` (E101:50–57) |
| O-4 | 334 | the incident procedure (E101:55–56) | **MISATTRIBUTED** at HEAD | now E101:67–68 |
| O-5 | 308–319 | ‖w_c‖ 0.766; 36 frames; Holm 26/25; medians 0.05; D 25/36; P 27/36; τ ≤ 0.20; precondition 30/36; power 0.915 | CORRECT | D101:59, 93, 248–279. The recomputed tails are 0.014408 and 0.005665, and the joint pass 0.91478 at p = 0.8 and 0.99975 at p = 0.9: the memo is right |
| O-6 | 321–354 | §4.3–§4.4 quotes and citations | CORRECT | D101:39–44, 281–291, 362–369, 455–464, 508–510, 518–521, 538–547; R101:178–179; METH:83–85; C003:295–297; E005:142 |
| O-7 | 480 | "025's confirmation took about 2.5–3.5 hours" (E025:160) | **WRONG** | That was the pre-run estimate. The run took 19:23:57.846–21:12:43.518 UTC, 1 h 49 min (`experiments/025-nounness-direction-intervention/evidence/confirm-2026-09-26/README.md:8`) |

---

## 2. Summary of counts

**The header, §0, §1, §2 and §3 (155 checked claims):**
- CORRECT: **137**;
- WRONG: **8** (H1, K3, L12, N1, R1n, S17, S18, S21);
- MISATTRIBUTED: **7** (L7, T2, T9, C2, C3, S4, S5);
- UNVERIFIABLE: **3** (R4q, S34, S39).

| section | claims | CORRECT | WRONG | MISATTRIBUTED | UNVERIFIABLE |
|---|---|---|---|---|---|
| header | 2 | 1 | 1 | 0 | 0 |
| §0 | 15 | 14 | 1 | 0 | 0 |
| §1 | 46 | 39 | 2 | 5 | 0 |
| §2 | 48 | 46 | 1 | 0 | 1 |
| §3 | 44 | 37 | 3 | 2 | 2 |
| **total** | **155** | **137** | **8** | **7** | **3** |

**The premises of §4.5 and §5 (40):**
- 38 CORRECT, 2 WRONG (Q7a, Q7b);
- three premises are correct but need rewording (P1e, P1f, P4c);
- one premise is unaddressed (Q4d).

**The spot checks outside scope (7):** 4 WRONG (O-1, O-2, O-3, O-7), 1 MISATTRIBUTED (O-4), 2 CORRECT.

**Every quoted phrase** in §1–§3 matches its source verbatim, or with a marked ellipsis. That includes the quotes in
K2, L10, L17, N3, N4, N5, V1, R0, R1k, R1m, R2a, R2e, R3e, R4g, R4n, S6, S7, S31 and S43.

---

## 3. Tags: analysis or opinion presented as fact

The tag legend (line 11) defines only **[AI observation]** and **[proposal]**. The following carry neither:

| line | text | issue |
|---|---|---|
| 42 | "The prediction (C003) is strong at the stage level." | an evaluation; it omits the stage failures (015 Y2, 018 Y2) |
| 45 | "No causal route that ran failed on a criterion that C002 maps to…" | the memo's synthesis; tagged in §2 (line 228) but not here |
| 48 | "Experiment 101 is well specified and well powered." | an opinion; the power depends on an assumed 80 % sign rate (D101:274–276) |
| 50 | "101 has risks the reviews did not raise" | the memo's analysis; tagged in §4.5 but not here |
| 157 | "Gates should be set relative to each split's clean behavior." | a recommendation; 006 already did this (DS006:268–275) |
| 174 | "tied to an unrelated gate" | "unrelated" is a judgement |
| 193–194 | "a product of many probabilities"; "an over-demanding composite gate, not a falsification" | analysis and opinion |
| 217–218 | "The wrong outcome variable … a model of the model" | opinion |
| 225–226 | "The odd/even decomposition worked … It is worth reusing." | an opinion and a proposal, from descriptive records |
| 251 | "Block 0 must be exact" | an inference beyond the descriptive record |
| 277–278 | "there is no valid self-repair bound" | an AI observation stated as fact |
| 283 | "The sequence was adaptive: each design followed the previous result" | a characterization with no record line |
| 330–341 | "[my reading]" | a third tag, not defined in the legend; use [AI observation] |
| 341 | "which bears on Level 6" in the `…AS_PREDICTED` row | an unmarked reading |
| §6 | "strong integrity tooling", "the closest thing to circuit discovery" | opinions (outside the required scope) |

The memo tags correctly: §2's "Across the routes", Route 1's self-repair block (though its premise is wrong, R1n),
Route 4's item 2, §3.7's last sentence, all of §4.5, and §5 as a whole.

---

## 4. Overclaims, and the separation of prediction, localization and causation

**The separation is maintained in structure:**
- §1.1 is localization ("not causal evidence on the ladder");
- §1.2 is prediction ("predictive, not causal"; C003:75);
- §1.3 is causal evidence, not established.

No decoded variable is called causal. No 101 result is claimed, and no success is claimed for 005, 006, 007 or 025.

**Exceptions to fix:**

| # | line | text | problem |
|---|---|---|---|
| O1 | 38 | "Nothing is causal." | An absence claim. The record says causal status is *not established*, and forbids reading it as no effect (E025:255; D101:42–43; READY:321–324; the memo's own §3.11) |
| O2 | 115 | "No gate is checked." | False. The preregistration and reporting gates are checked (C003:273–274) |
| O3 | 42–44 | "strong at the stage level … partly an exact recomputation" | A success framing without the failures. The accuracy share of recomputation is an open rival (C003:331–337; PAPER:30–31 "may") |
| O4 | 48 | "well powered" | Conditional on an assumed per-frame sign rate (D101:274–276) |
| O5 | 46–47 | "two strata that responded oppositely" | It rests on a descriptive split, for which "no stratum-level claim is made" (E025:258–261), and is not marked descriptive |
| O6 | 225 | "The odd/even decomposition worked" | A success claim from descriptive records (E025:258–268; FR025:90) |
| O7 | 251 | "Block 0 must be exact" | A necessity claim from descriptive comparators. Cheaper rules reach 0.9246–0.9865 in Y1 coordinated frames (FR023:44, 68) |
| O8 | 277–278; 159–165 | "no valid self-repair bound"; the ratio "ill-posed for T" | An inference stated as fact, on a misread premise: the ratio pools E and T (R1n) |
| O9 | 452 | "Already partly falsified" | A falsification claim from a descriptive comparator of one rule (Q7a) |
| O10 | 54 | "## 1. Established results" | Loose: §1.2 includes unestablished and failed families. "Recorded results" would be exact (optional) |

---

## 5. Required corrections: exact old text → new text

Line numbers are the memo's. Each old text is verbatim.

**C1 (lines 3–5, header; stale).** Re-check if the lock is committed before the memo is added.
```
**Date:** 2026-10-14. **Repository:** branch `exp101-phases`, HEAD `40829d2`. The tree is clean except for the untracked
`experiments/101-read-component-intervention/confirmation-v1.json`, which the 2026-10-14 freeze wrote, not this
analysis (see §4.1).
```
→
```
**Date:** 2026-10-14. **Repository:** branch `exp101-phases`, HEAD `3621eec`, clean. Outside
`experiments/101-read-component-intervention/`, every cited file is identical at `40829d2`. The freeze's
`confirmation-v1.json` (installed `a857f9a`) and the calibration's `calibration-v1.json` (installed `d502777`) are
committed (see §4.1).
```

**C2 (line 38; O1).**
```
1. **Nothing is causal.** Both claims are `LOCALIZED` (C002:5; C003:5).
```
→
```
1. **Nothing is established as causal**, which is not a finding that nothing is causal (E025:255; D101:42–43). Both
   claims are `LOCALIZED` (C002:5; C003:5).
```

**C3 (lines 42–44; O3, tag).**
```
3. **The prediction (C003) is strong at the stage level.** Examples: 011's Spearman 0.848; 017's ΔT R² 0.999 and
   1.000; 023's g ≥ 0.9966 in three conditions. It is conditional on measured reference states, and partly an exact
   recomputation.
```
→
```
3. **[AI observation] The prediction (C003) is strong at the stage level, with recorded failures (015 Y2, 018 Y2,
   021 Y2, 023 Y1/cue_final; C003:174, 178, 183, 186).** Examples: 011's Spearman 0.848; 017's ΔT R² 0.999 and
   1.000; 023's g ≥ 0.9966 in three conditions. It is conditional on measured reference states, and much of its
   accuracy may come from exact recomputation, a rival the record has not distinguished (C003:331–337).
```

**C4 (lines 45–47; tag, O5).**
```
4. **No causal route that ran failed on a criterion that C002 maps to necessity, sufficiency or control.** They stopped
   on a behavioral precondition (005), a pair-level fidelity correlation (007), and a sign count over two strata that
   responded oppositely (025).
```
→
```
4. **[AI observation] No causal route that ran failed a primary criterion that C002 maps to necessity, sufficiency or
   control** (005's non-confirmatory P8 passed; only its secondary compensation band was missed, C002:51). They stopped
   on a behavioral precondition (005), a pair-level fidelity correlation (007), and a sign count over two strata that,
   descriptively, responded oppositely (025; E025:258–261).
```

**C5 (line 48; O4, tag).**
```
5. **Experiment 101 is well specified and well powered.**
```
→
```
5. **[AI observation] Experiment 101 is well specified; R and I pass together with probability about 0.915 if 80 % of
   frames show each predicted sign (D101:274–276).**
```

**C6 (line 50; tag).** `6. **101 has risks the reviews did not raise:**` → `6. **[AI observation] 101 has risks the reviews did not raise:**`

**C7 (line 71; L7).** `  0.900 / 0.074 (C002:50).` → `  0.900 / 0.074 (C002:30, 50).`

**C8 (lines 74–75; L12).**
```
- **Kind of evidence.** This is interventional localization by patching and pair-centred neutralization. It is not
  causal evidence on the ladder, because the data were exploratory or calibration data (C002:96, 101–106).
```
→
```
- **Kind of evidence.** This is interventional localization by patching and pair-centred neutralization. It is not
  causal evidence on the ladder: the one confirmation failed its behavioral precondition, and 007's families passed on
  fresh data at criterion level only, with its composite route unmet (C002:96, 101–106).
```

**C9 (line 92; T2).** `| prospective, given the measured input | C003:168, 170 |` → `| prospective, given the measured input | C003:168, 170; E011:87–89 |`

**C10 (line 98; T9).** `| Y3 unestablished | C003:175–176 |` → `| Y3 unestablished | C003:175–176; E017:126–128 |`

**C11 (lines 105 and 109; C2, C3, C4).**
- `**Compact programs for the whole mechanism failed** (C003:166, 169):` → `**Compact programs failed** (C003:166, 169; E006:72; E007:87):`
- `- 009: the rank-1 rule, 0.765 against 0.80.` → `- 009: the rank-1 transport rule, a rule for one stage (C003:338), 0.765 against 0.80.`

**C12 (line 115; N1, O2).**
```
- **No gate is checked.** Neither claim checks a necessity, sufficiency or control gate (C002:98–106; C003:254–274). "No
```
→
```
- **No necessity, sufficiency or control gate is checked.** Neither claim checks one (C002:98–106; C003:254–274); only
  the preregistration and the two reporting gates are checked (C003:273–274). "No
```

**C13 (line 157; tag).** `- Gates should be set relative to each split's clean behavior.` → `- [proposal] Gates should be set relative to each split's clean behavior, as 006's design already did (DS006:268–275).`

**C14 (lines 159–165; R1n, O8).**
```
**[AI observation] The self-repair measure was ill-posed for T.**
```
→
```
**[AI observation] The self-repair measure was ill-posed for E and T.**
```
and
```
- Values of −5.742 and −6.038 mean the other components' direct effects fell by about six times the neutralized
  component's own direct effect. That is downstream mediation, consistent with T acting through the readout MLPs (`m_R`
  0.987), not compensation.
- A ratio normalized by a direct effect is ill-conditioned for a component that acts indirectly. The record therefore
  contains no valid self-repair bound for `L03.H04`.
```
→
```
- The reported value is the mean, over the six manifest frames, of per-frame ratios that neutralize T in the two
  coordinated frames and E in the four cue-final frames (DS005:208–217, 822;
  `src/neural_decompiler/plural_mechanism.py:2512–2539`). It is not specific to T.
- Values of −5.742 (calibration) and −6.038 (confirmation; FR005:120, 147) mean that, on average, the other components'
  direct effects fell by about six times the neutralized component's own direct effect. That is consistent with
  downstream mediation, since E and T both act mostly through later components (`m_R` 0.987, FR005:148), not with
  compensation.
- A ratio normalized by a direct effect is ill-conditioned for a component that acts indirectly. The record reports no
  T-specific ratio and no co-neutralization result, so it contains no valid self-repair bound for `L03.H04`.
```

**C15 (line 174; tag).**
```
- **What it teaches.** The causal route was tied to an unrelated gate on the compressed program.
```
→
```
- **What it teaches [AI observation].** The causal route was tied to a pre-lock quality gate on the separate
  compressed program (DS006:252–264).
```

**C16 (lines 192–194; tags).**
```
  - The 0.90 floor was carried over unchanged, although 006 had forecast its failure.
  - Seven families on two sets, all required, made the route's chance of success a product of many probabilities.
  - This is an over-demanding composite gate, not a falsification.
```
→
```
  - The 0.90 floor was carried over unchanged (DS007:248–249), although 006 had forecast its failure (E006:58–59).
  - [AI observation] Seven families on two sets, all required (DS007:248–252), made the route's chance of success a
    product of many probabilities.
  - [AI observation] This is an over-demanding composite gate. The record calls it a route not met, not a falsification
    (C002:54–55).
```

**C17 (line 217; tag).** `  1. **The wrong outcome variable.** What was scored is` → `  1. **The outcome variable [AI observation].** What was scored is`

**C18 (line 219; R4q).**
```
  2. **Heterogeneity and saturation [AI observation].** 024's association was largely a contrast between classes: mean
```
→
```
  2. **Heterogeneity and saturation [AI observation].** 024's association may be largely a contrast between classes (no
     within-class ρ is recorded for the fresh cues; in calibration the within-stratum ρ were 0.26–0.43, FR024:13): mean
```

**C19 (line 222).**
```
  3. **No calibration phase** (E025:120), so the per-unit sign rate was never estimated.
```
→
```
  3. **No calibration phase** (E025:120), so the per-unit sign rate was never measured; the design's power table assumed
     it (DS025:379–401).
```

**C20 (lines 225–226; O6, tag).**
```
  The odd/even decomposition worked: the equal even parts for the noun and random rotations isolate generic disruption.
  It is worth reusing.
```
→
```
  [AI observation] The odd/even decomposition looks reusable: the even parts were nearly equal for the noun and random
  rotations, 0.0144 and 0.0141 [descriptive] (FR025:90), as expected if the even part carries generic disruption.
```

**C21 (line 230).**
```
- **Routes 1–3 never failed on a criterion that C002 maps to necessity, sufficiency or control.** They stopped on a
```
→
```
- **Routes 1–3 never failed a primary criterion that C002 maps to necessity, sufficiency or control.** They stopped on a
```

**C22 (line 243; S4).**
```
Confirmation sets were inherited, 006→007 and 020→021 (E006:89–92).
```
→
```
Confirmation sets were inherited unexecuted, 006→007 (E006:89–92) and 020→021 (E020:184–186; E021:28).
```

**C23 (line 244; S5).** `(C003:125–126).` → `(C003:125–126, 412).`

**C24 (lines 251–252; O7).**
```
   - Block 0 must be exact: in Y1 cue-final frames, cheaper rules recover g 0.5045 (value-only) to 0.9145 (six heads)
     (FR023:69).
```
→
```
   - At block 0, the exact rule is what completes the program [descriptive]: in Y1 cue-final frames, cheaper rules
     recovered g 0.5045 (value-only) to 0.9145 (six heads) against 0.9955, but 0.9246 to 0.9865 in Y1 coordinated
     frames (FR023:68–69; E023:180–181).
```

**C25 (lines 260–261; S17, stale).**
```
   - 101's Level-1 contrast tolerance is 2e-2 against a recorded maximum of 0.0119: a margin of 1.7–1.85×, untested
     under layer-3 edits (P101:719–729).
```
→
```
   - 101's Level-1 contrast tolerance is 2e-2. Before calibration the recorded maximum was 0.0119, a margin of
     1.7–1.85× (P101:719–729). 101's calibration ran the layer-3 edits on exposed frames: maximum 1.065e-2 (1.9×;
     `two`'s arms ≤ 5.01e-3), and every G4 tolerance held (E101:53–54;
     `experiments/101-read-component-intervention/evidence/calibrate-2026-10-14/README.md:67–78`).
```

**C26 (lines 263–264; S18).**
```
   - Floors were calibrated on each program's own exposed performance, and some were clamped at zero skill
     (C003:415–416).
```
→
```
   - Several floors were calibrated against the program's own exposed performance, and some were clamped to zero
     skill (C003:415–416).
```

**C27 (line 265; S21).** `   - Eight outcomes sit at their thresholds:` → `   - Nine outcomes, in eight experiments, sit at or near their thresholds:`

**C28 (line 273).** `     - 025 A, 26 vs 27.` → `     - 025 A, 26 vs 27 (E025:17).`

**C29 (lines 277–278; S34, O8).**
```
   subspace intervention (PAPER:573–574). Necessity rests on pair-centred neutralization (PAPER:567–572), and there is
   no valid self-repair bound (§2, Route 1).
```
→
```
   subspace intervention (PAPER:573–574). Necessity rests on pair-centred neutralization (PAPER:567–572), and 005's
   secondary compensation band was missed (PAPER:571–572); [AI observation] the record holds no valid self-repair
   bound for `L03.H04` (§2, Route 1).
```

**C30 (lines 283–284; S39).**
```
    - The sequence was adaptive: each design followed the previous result, and 005's eligibility rule was amended after
      Tier A had been inspected (MAP:264).
```
→
```
    - [AI observation] The sequence was adaptive: each design followed the previous result. 005's eligibility rule was
      amended after Tier A had been inspected (MAP:265).
```

**C31 (lines 298–304; O-1 to O-3, stale).**
```
- **The README** still says "No phase has run" (E101:3–4).
- **The freeze has run.** A session record outside the repository says that the production `freeze` ran once on
  2026-10-14, 08:14:27–08:14:42 UTC, at `40829d2`, and wrote `confirmation-v1.json`
  (`/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/freeze/README.md:1–10`).
  - Its review verdict and install commit are still placeholders.
  - git shows the file untracked.
- **Nothing else has run:** no calibration, lock, stage 1 or confirmation.
```
→
```
- **The README** says CALIBRATED (E101:3–9).
- **The freeze** ran once on 2026-10-14, 08:14:27–08:14:42 UTC, at `40829d2` (tokenizer only), and wrote
  `confirmation-v1.json`. The AI freeze review found PASS WITH NOTES, and the file was installed in `a857f9a`
  (E101:45–49; `experiments/101-read-component-intervention/evidence/freeze-2026-10-14/README.md:3–9, 57`).
- **The calibration** (exposed, exploratory; spent data) ran once at `8f65062`: no revision trigger, every gate held,
  τ_rem 0.0252 and τ_ins 0.0351. The AI review found PASS WITH NOTES, and `calibration-v1.json` was installed in
  `d502777` (E101:5–8, 50–57).
- **Not run at HEAD `3621eec`:** lock, stage 1 and confirmation; no 101 prompt has been executed (E101:8–9).
```

**C32 (line 330 and rows 335, 337, 338, 340; tag legend).** Replace `Rows marked [my reading] are not stated in the
design.` with `Rows marked [AI observation] are not stated in the design.`, and each `[my reading]` with
`[AI observation]`.

**C33 (line 334; O-4).** `(E101:55–56)` → `(E101:67–68)`

**C34 (line 368; P1e).**
```
   - D101, R101 and P101 never mention these checks; I searched all three.
```
→
```
   - D101, R101 and P101 never mention nullspace, rowspace or dormant-direction checks; I searched all three, and the
     committed freeze and calibration reviews. 101 does use null directions matched in length to the real edit: the
     off-read slice and 8 isotropic directions (D101:127–134, 156–157).
```

**C35 (line 390; P4c).**
```
   - If the frames' signs are as consistent as 007's 120/120, the pass probability is near 1.
```
→
```
   - 101's exposed calibration found N > 0 and S > 0 in 36 of 36 frames [exploratory]
     (`experiments/101-read-component-intervention/evidence/calibrate-2026-10-14/README.md:90–93`). If the fresh frames
     behave alike, the pass probability is near 1.
```

**C36 (line 437; Q4c, Q4d).**
```
     phase; about 3 hours of CPU (E025:160).
```
→
```
     phase; about 2 hours of CPU (025's confirm took 1 h 49 min:
     `experiments/025-nounness-direction-intervention/evidence/confirm-2026-09-26/README.md:8`). It needs 40 new fresh
     cues, and the curated pool is nearly used up (DS025:412; D101:108–110).
```

**C37 (lines 452–453; Q7a, Q7b, O9).**
```
   - Already partly falsified at block 0, where the value-only rule recovers g 0.5045 (FR023:69).
   - *Test:* description length against accuracy, stage by stage, on the stored cells. C003 names this test (C003:337).
```
→
```
   - At block 0, every tested cheaper rule misses the 0.02 bound in Y1 cue-final frames (the best, six heads, g 0.9145
     against 0.9955), but six heads meets it in Y1 coordinated frames (0.9865 against 0.9987) [descriptive]
     (FR023:68–69). This bounds those rules, not every rule with at most k parameters.
   - *Test:* description length against accuracy, stage by stage, on the stored cells. C003 names a related test: each
     stage's reduced rule against the exact computation, on fresh data (C003:337).
```

**C38 (line 480; O-7).**
```
- **Runtime:** 025's confirmation took about 2.5–3.5 hours for 90,720 forward passes (E025:160).
```
→
```
- **Runtime:** 025's confirmation took 1 h 49 min for 90,720 patched forward passes
  (`experiments/025-nounness-direction-intervention/evidence/confirm-2026-09-26/README.md:8`); the design estimated
  2.5–3.5 hours (E025:160).
```

---

## 6. Recommended clarifications (optional)

- **Line 33.** After "which refer to commit `d7c5e62`", add: "Lines 29–228 of that commit are 4 higher, for example
  PAPER's 025:67–69 and 025:116, which are E025:71–73 and 120."
- **Lines 56–57.** Add "where applicable; another valid causal test may substitute, with an explicit rationale
  (METH:83–84, 89)".
- **Line 64.** Add "when the cue is not the final token (C002:18–19)".
- **Line 99.** "closes κ 0.863 / 0.724" → "carries κ 0.863 of channel D's contribution to the decoded read and 0.724 of
  its contribution to the pattern term".
- **Line 101.** For the status, write "prospective; Y2 conditional; the coordinated share descriptive" (C003:184).
- **Line 113.** "### 1.3 Causal evidence: none" → "### 1.3 Causal evidence: not established".
- **Line 155.** Add "(98/120 is the screen's holdout split, DS005:107; 005's 19-noun calibration had 95/114 flips,
  FR005:111)".
- **Line 231.** Order the routes 1–3: "a behavioral precondition, a program gate and a fidelity correlation".
- **Line 233.** "two stages upstream" → "at the cue embedding, upstream of block 0, layers 1–2 and block 3".
- **Lines 249–250.** Add "(Y1; descriptive)".
- **Line 258.** "forced the Level-1 tolerance" → "led to an authorized amendment raising the Level-1 tolerance"
  (E020:73–80).
- **Line 280.** Add "in 009–023" after "6–18 new frames".
- **Line 365.** "no illusion at the head" holds approximately. The edit also moves the LayerNorm scale and the head's
  key at `p_c` (D101:81–85, 162).
- **Line 370.** Note that the head-arm shares already exist for the read edits (D101:301–303), and that the calibration
  reports about 88–89 % through the head (calibrate README:94–95).
- **Lines 375–377.** Say that the ΔT-only comparator shows a dependence on paths beyond the head, not on the head's
  off-axis output.
- **Line 432.** "The code supports GPT-NeoX models only" → "The block decomposition supports only parallel GPT-NeoX
  models (`components.py:412–417`)".
- **Lines 442–443.** Say that the template-base comparator replaces only the readout MLPs' operating point (E020:134).
- **Line 54.** "Established results" → "Recorded results".
