# Experiment 024 — report

- Run `b04ce3526306a981`; phase commits {'calibrate': 'bf0049c85342d01caf9d6a830389ce1e72c05a71', 'confirm': 'af160cee0389bcdaf12bcbd92a8616f142777a52', 'lock': 'be74d23d086bd12d894db1f73e6a1b4160aeebd5'}; phases: calibrate complete, confirm complete, lock complete, report not_started
- Design revision 2 (`9d03dee`), plan revision 1 (`608088c`); configuration `production`


## Calibration (exposed only; 139 cues; B = 10000, P = 100000)

- F_ρ (element [249]) = 0.24411074612857814; undefined draws 0 (stop at 250); median of the defined draws 0.5285171092272343; tails element_lower 0.2441, element_upper 0.7376, max 0.8487, median 0.5285, min -0.0481
- null₉₇.₅ (element [97499] of 100000 permutations of 40 ranks) = 0.3136960600375234
- Effective threshold max(F_ρ, null₉₇.₅) = 0.3136960600375234, bound by null_975
- Draw rates (descriptive): ENVELOPE_ONLY_FAILURE 0.0000, GUARD_FAILURE 0.0593, NOT_INTERPRETABLE 0.0000, PASS 0.9407
- Calibration ρ over the population (descriptive) 0.5297; within strata adjective 0.4263, determiner-like 0.2561, quantity 0.3090
- Pronoun out-of-fit check (descriptive): ρ 0.4232, the line's median |log error| 0.2038, mean signed 0.1416
- The line (exposed-data-fitted, prospectively frozen; secondary): log MSE = -2.481260357420486 + 1.2992017464639374 · nounness; residual sd 0.2570037337349923; the largest calibration score 0.13502027836111233

## The outcome

**`NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`** — the frozen weight-derived score prospectively predicted the frozen readout's error on new lexical representations, including substantial extrapolation beyond the exposed score range, and the ordinary singular nouns had larger error than the non-noun controls beyond chance — which neither of the two preregistered simple alternatives (plural morphology alone, measure class alone) predicts.

- The guard eliminates the two preregistered simple alternatives, plurality-only and measure-class-only; it does not establish nounness as a unique causal factor or eliminate every correlated lexical property.
- Every outcome is predictive and associational; none establishes that nounness causally changes attention.
- 5 of the 8 E cues lie above the calibration population's maximum nounness (+0.135020); the fresh test is a prospective extrapolation test there. Calibration supplied no evidence for that range, and the post-hoc behaviour of 023's five spent cues in it is design motivation only.

## The primary test

- ρ = 0.6108818011257036 against F_ρ 0.24411074612857814 and null₉₇.₅ 0.3136960600375234: **PASS** — ρ ≥ max(F_ρ, null₉₇.₅): higher operational nounness prospectively predicted larger error of the frozen block-4/5 attention readout, at least as strongly as the exposed-like relationship and beyond chance

## The E–N disambiguation guard (exact one-sided permutation test)

- K = 1 of 12870 assignments at or above the observed contrast (bound 321); exact p = 1/12870 = 0.000078: **PASS** — the ordinary singular nouns had larger error than the non-noun controls beyond chance (an exact one-sided permutation test, K ≤ the bound)
- D_EN = 0.5680373703974243 (ratio of geometric means 1.7648); the derived threshold t = 0.32671524926158657 (descriptive)
- E: mean MSE 0.124994, mean log MSE -2.0887; N: mean MSE 0.071356, mean log MSE -2.6567

- Fresh cues by class: N: honest, polite, rude, sleepy, wise, lucky, merry, nervous; B: gallons, ounces, acres, herds, crowds, bundles, clusters, litres; D: gallon, ounce, acre, herd, crowd, bundle, cluster, litre; C: apples, horses, doctors, kings, rabbits, poets, dragons, lions; E: apple, horse, doctor, king, rabbit, poet, dragon, lion

- Not shown by any outcome:
  - that nounness causes the attention change
  - that the score measures linguistic nounhood
  - which part of the score matters: similarity to the target nouns or dissimilarity to the exposed cues (both separate E from N)
  - that no other property that differs between ordinary nouns and these adjectives explains E > N
  - that the exposed OLS line has been validated over the extrapolated range (the line is secondary)
  - generality beyond the 108 exposed frames, the 79 nouns and this checkpoint

| class | word | nounness | MSE | log MSE | predicted log MSE |
|---|---|---|---|---|---|
| N | honest | -0.2294 | 0.059384 | -2.8237 | -2.7792 |
| N | polite | -0.1903 | 0.073532 | -2.6100 | -2.7285 |
| N | rude | -0.1244 | 0.070628 | -2.6503 | -2.6429 |
| N | sleepy | -0.1371 | 0.057631 | -2.8537 | -2.6594 |
| N | wise | -0.1944 | 0.054583 | -2.9080 | -2.7338 |
| N | lucky | -0.1330 | 0.086308 | -2.4498 | -2.6541 |
| N | merry | -0.1151 | 0.074708 | -2.5942 | -2.6308 |
| N | nervous | -0.0451 | 0.094070 | -2.3637 | -2.5398 |
| B | gallons | 0.1651 | 0.101779 | -2.2849 | -2.2667 |
| B | ounces | 0.0760 | 0.098775 | -2.3149 | -2.3826 |
| B | acres | 0.1631 | 0.120283 | -2.1179 | -2.2693 |
| B | herds | 0.3966 | 0.143955 | -1.9383 | -1.9660 |
| B | crowds | 0.2372 | 0.140660 | -1.9614 | -2.1731 |
| B | bundles | 0.2928 | 0.101598 | -2.2867 | -2.1009 |
| B | clusters | 0.2618 | 0.118432 | -2.1334 | -2.1412 |
| B | litres | 0.0067 | 0.125779 | -2.0732 | -2.4726 |
| D | gallon | 0.0498 | 0.183428 | -1.6959 | -2.4165 |
| D | ounce | -0.0656 | 0.077189 | -2.5615 | -2.5664 |
| D | acre | -0.0318 | 0.150771 | -1.8920 | -2.5226 |
| D | herd | 0.3029 | 0.116670 | -2.1484 | -2.0877 |
| D | crowd | 0.1979 | 0.106157 | -2.2428 | -2.2242 |
| D | bundle | 0.1474 | 0.091167 | -2.3951 | -2.2898 |
| D | cluster | 0.1635 | 0.082468 | -2.4954 | -2.2689 |
| D | litre | 0.0452 | 0.093842 | -2.3661 | -2.4226 |
| C | apples | 0.2230 | 0.152490 | -1.8807 | -2.1916 |
| C | horses | 0.3459 | 0.117900 | -2.1379 | -2.0318 |
| C | doctors | 0.3187 | 0.107691 | -2.2285 | -2.0673 |
| C | kings | 0.2062 | 0.115493 | -2.1585 | -2.2134 |
| C | rabbits | 0.3240 | 0.170652 | -1.7681 | -2.0604 |
| C | poets | 0.3147 | 0.117044 | -2.1452 | -2.0724 |
| C | dragons | 0.3182 | 0.156312 | -1.8559 | -2.0678 |
| C | lions | 0.3097 | 0.134637 | -2.0052 | -2.0789 |
| E | apple | 0.1675 | 0.118032 | -2.1368 | -2.2637 |
| E | horse | 0.2983 | 0.142796 | -1.9463 | -2.0937 |
| E | doctor | 0.2177 | 0.121935 | -2.1043 | -2.1984 |
| E | king | 0.1175 | 0.113273 | -2.1780 | -2.3286 |
| E | rabbit | 0.1286 | 0.157289 | -1.8497 | -2.3141 |
| E | poet | 0.1660 | 0.101705 | -2.2857 | -2.2656 |
| E | dragon | 0.1882 | 0.111147 | -2.1969 | -2.2367 |
| E | lion | 0.1085 | 0.133772 | -2.0116 | -2.3404 |

## Descriptive records (no outcome force)

- Stage-2 identities (maxima): I1 1.5317713646822995e-05 (tolerance 1e-04), I3 2.1161813141654138e-05 (tolerance 1e-04), I4 6.424818726813442e-05 (tolerance 1e-03)
- C recomputed from the saved Δx3: bit for bit True over 4320 pairs
- Prompt accounting: {'equal': True, 'executed': 4320, 'ledger': 4320, 'manifest': 4320}
- Spearman by group: coordinated 0.6976, cue_final 0.5859; normalized-MSE Spearman 0.5919
- The line against the observed log MSE: median |log error| 0.1227, mean signed 0.0928; beyond the calibration range (23 cues) median |log error| 0.1052, within it 0.1742
- Class means of log MSE: B -2.1389, C -2.0225, D -2.2247, E -2.0887, N -2.6567
- measure effect among nouns (mean(m_B, m_D) − mean(m_C, m_E)): -0.1262 [-0.2523, 0.0042]
- measure × plurality interaction (descriptive) ((m_B − m_D) − (m_C − m_E)): 0.0196 [-0.2441, 0.2715]
- noun effect (mean(m_B, m_C, m_D, m_E) − m_N): 0.5380 [0.3945, 0.6774]
- plurality effect among nouns (mean(m_B, m_C) − mean(m_D, m_E)): 0.0760 [-0.0583, 0.2035]
- descriptive only; no explanation is declared a winner; the outcome is unchanged
- Ladder B: block-5 attention exact removes 0.3983 of Σ(Δc − C)²; block-4 attention then 0.6017
- Ladder C: block-5 attention exact removes 0.4245 of Σ(Δc − C)²; block-4 attention then 0.5755
- Ladder D: block-5 attention exact removes 0.3257 of Σ(Δc − C)²; block-4 attention then 0.6743
- Ladder E: block-5 attention exact removes 0.2704 of Σ(Δc − C)²; block-4 attention then 0.7295
- Ladder N: block-5 attention exact removes 0.4098 of Σ(Δc − C)²; block-4 attention then 0.5902
- Level-1 identity max |L1 − Δc| 0.009929 against 0.02 (descriptive)
