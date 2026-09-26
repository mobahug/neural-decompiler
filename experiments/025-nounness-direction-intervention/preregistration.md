# Experiment 025 — preregistration

- Lock `8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb`; run `914fa3afff5c73ec` at `d3ecbd6feb1bd2aefe08dc0f8d799815be47c519`; design revision 1 (`c0885e5`, corrected `26c9925`), plan revision 1 (`7d90d28`)
- The analysis module `src/neural_decompiler/cue_rotation.py` at blob `622aa832ae29047c3060d989928236320374e227`; configuration `production`
- The freeze `6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4`; manifest `0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61` (90720 condition-tagged runs: 40 cues × 108 frames × 21 conditions)

## The intervention

- `("EMBED", p_c)`, REPLACE; the direction `d = μ̂_noun − μ̂_cue` (digest `5f914283c39d2f37911ba3bb1391a98bbe3625737f8df80c42eb336868a1bf2a`, |d| 1.3576567483118904); cos(d̂, p̂) = 0.23521413845112732 (descriptive)
- The directional (odd) score component is exactly ±0.32 (primary) and ±0.16 (secondary); the complete change is s₀(cos θ − 1) ± odd
- 7 nounness-neutral random tangent controls per cue (tag `025|control|{token_id}|{j}`) at the same angle; a plurality control (secondary)
- The conditions: base, noun+0.32, noun-0.32, noun+0.16, noun-0.16, rand1+0.32, rand1-0.32, rand2+0.32, rand2-0.32, rand3+0.32, rand3-0.32, rand4+0.32, rand4-0.32, rand5+0.32, rand5-0.32, rand6+0.32, rand6-0.32, rand7+0.32, rand7-0.32, plur+0.32, plur-0.32

## The outcome-bearing statistics

- **A**: A_i = ½[ℓ_i(+θ_i) − ℓ_i(−θ_i)] at the primary dose
- **B**: B_i = A_i − (1/k)·Σ_j |A_ij|, A_ij the random controls' A
- **G**: G_i = ½[D_attn,i(+θ_i) − D_attn,i(−θ_i)] at the primary dose
- **ell**: ℓ = log(Σ SSE_C / Σ n) over the cue's 108 exposed frames × 79 scorable nouns (rr.fresh_pair_cells, rr.cue_mse)
- **D_attn**: total-variation distance of the captured L4/L5 rows at p_t from the locked 020 rows, equal mean over the 16 heads, then over the 108 frames
- **count_rule**: PASS iff at least 27 of 40 values are strictly positive; ties and zeros count against
- The criterion: pre-registered sign-count criteria with an exact Binomial(40, 0.5) reference tail under H0: P(positive) <= 0.5 for independent cue units; not unconditional, design-based randomization tests (the 40 lexical items are selected mechanically, not sampled at random); the causal reading comes from the within-cue intervention (+θ and −θ are both run for every cue), and the sign count measures consistency across the frozen test population; ties and zeros count against PASS; P(X ≥ 27) = 5288280983/274877906944 = 0.01923865414210013
- **A**: A_i > 0: the +nounness intervention produced greater frozen-readout approximation error than the matched −nounness intervention; A alone does not establish that +θ rises above the unperturbed baseline or that −θ falls below it (the baseline and the even component are descriptive only)
- **B**: B_i = A_i − mean_j |A_ij|: the nounness-direction effect in the predicted sign exceeds the typical size of an equal-angle, nounness-neutral directional effect; absolute values because the signed random mean has expectation 0 by the u → −u symmetry; t̂ is not assumed exchangeable with the controls
- **G**: G_i = ½[D_attn(+θ) − D_attn(−θ)], D_attn the total-variation distance of the captured L4/L5 rows at p_t from the locked 020 reference rows, equally averaged over the frozen 16 heads, then over the 108 frames; equal weighting is frozen before outcomes and avoids outcome-based head selection; irrelevant or oppositely responding heads can dilute or oppose the aggregate signal, so failure of G is conservative with respect to this particular routing summary

## The outcome

| condition | label |
|---|---|
| any failed validity gate or incident (no result is outcome-bearing) | `NOT_INTERPRETABLE` |
| A fails | `CAUSAL_EFFECT_NOT_ESTABLISHED` |
| A passes, B fails | `DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC` |
| A and B pass, G fails | `READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED` |
| A, B and G pass | `NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR` |

Not claimed by any label:
- semantic nounhood as the model's variable
- a general routing controller
- algorithm selection in general
- generalization beyond this checkpoint and this mechanism
- that the effect excludes a plurality-direction contribution (the nounness direction is partly aligned with plurality, cos +0.235)

- Strata: the positive counts of A, B and G are reported for the 20 adjectives and the 20 ordinary singular nouns separately; they cannot change the label; if a global PASS is strongly concentrated in one stratum the final claim must say so (reporting trigger: a stratum below 14 of 20)
- Secondary: the magnitudes, the even components and baselines, the half dose, the plurality control, nMSE, the ladder, G against the random directions, the per-template counts and the causal fraction are computed after the result is written and can never rescue, alter or redefine it

## The validity gates

- I7′: the geometry recomputed from the weights before any prompt, bit for bit, and every geometry check within its tolerance (angle32 1e-06, angle64 1e-12, d_dot_controls 1e-12, d_dot_plurality 1e-12, length32 1e-06, length64 1e-12, neutral32 1e-06, neutral64 1e-12, odd32 1e-06, odd64 1e-12, orth_E_controls 1e-12, orth_E_plurality 1e-12, orth_E_t 1e-12, orth_t_controls 1e-12, orth_t_plurality 1e-12, unit_controls 1e-12, unit_plurality 1e-12, unit_t 1e-12)
- The patch path on 4 spent keys before the ledger: cardinal-009-1|an|271, quantifier-009-1|least|1878, coordinated-adjective-009-1|black|2806, quantifier-new-2|he|344; the confirm-time patch-path check on four already-executed keys (a plain capture against a patched θ = 0 run, and the embedding hook against the block-0 residual input, bit for bit) only validates the intervention plumbing; a pass does not strengthen the scientific result
- I1, I3, I4 on every run (1e-04, 1e-04 relative, 1e-03); C recomputed bit for bit; the Level-1 identity ≤ 0.02 on every outcome-bearing primary-dose run

## The 40 cues and their geometry

| stratum | word | token id | s₀ | τ | θ primary (°) | θ half (°) | even (primary) |
|---|---|---|---|---|---|---|---|
| adjective | anxious | 20138 | -0.174899 | 1.346344 | 13.7497 | 6.8252 | +0.005012 |
| adjective | cheerful | 39567 | -0.191226 | 1.344122 | 13.7729 | 6.8365 | +0.005498 |
| adjective | curious | 14338 | -0.198345 | 1.343090 | 13.7837 | 6.8418 | +0.005712 |
| adjective | jealous | 23327 | -0.132124 | 1.351212 | 13.6992 | 6.8005 | +0.003759 |
| adjective | lonely | 25106 | -0.142298 | 1.350179 | 13.7099 | 6.8057 | +0.004054 |
| adjective | nasty | 26321 | -0.222269 | 1.339339 | 13.8230 | 6.8611 | +0.006437 |
| adjective | careful | 10182 | -0.234876 | 1.337186 | 13.8457 | 6.8722 | +0.006825 |
| adjective | careless | 48292 | -0.236565 | 1.336888 | 13.8489 | 6.8737 | +0.006877 |
| adjective | famous | 8530 | -0.125097 | 1.351881 | 13.6923 | 6.7971 | +0.003555 |
| adjective | friendly | 11453 | -0.101981 | 1.353821 | 13.6723 | 6.7873 | +0.002890 |
| adjective | gorgeous | 23535 | -0.220686 | 1.339600 | 13.8203 | 6.8597 | +0.006389 |
| adjective | hungry | 18254 | -0.067569 | 1.355974 | 13.6501 | 6.7765 | +0.001908 |
| adjective | weary | 35725 | -0.131817 | 1.351242 | 13.6989 | 6.8003 | +0.003750 |
| adjective | wicked | 26395 | -0.140913 | 1.350324 | 13.7084 | 6.8050 | +0.004014 |
| adjective | ugly | 19513 | -0.260125 | 1.332504 | 13.8954 | 6.8964 | +0.007612 |
| adjective | vivid | 24863 | -0.289037 | 1.326533 | 13.9592 | 6.9276 | +0.008536 |
| adjective | vague | 21248 | -0.305889 | 1.322748 | 13.9999 | 6.9475 | +0.009086 |
| adjective | rapid | 5233 | -0.283384 | 1.327752 | 13.9461 | 6.9212 | +0.008353 |
| adjective | rigid | 16572 | -0.194772 | 1.343613 | 13.7782 | 6.8391 | +0.005605 |
| adjective | clever | 19080 | -0.257528 | 1.333008 | 13.8900 | 6.8938 | +0.007531 |
| noun | soldier | 15796 | +0.231534 | 1.337768 | 13.8396 | 6.8691 | -0.006722 |
| noun | sailor | 45758 | +0.248606 | 1.334701 | 13.8720 | 6.8850 | -0.007251 |
| noun | priest | 14959 | +0.175074 | 1.346321 | 13.7499 | 6.8253 | -0.005017 |
| noun | knight | 30605 | +0.162775 | 1.347863 | 13.7339 | 6.8175 | -0.004654 |
| noun | onion | 21635 | +0.138606 | 1.350563 | 13.7059 | 6.8038 | -0.003947 |
| noun | carrot | 47215 | +0.166081 | 1.347460 | 13.7381 | 6.8195 | -0.004751 |
| noun | pirate | 42404 | +0.069023 | 1.355901 | 13.6509 | 6.7768 | -0.001950 |
| noun | tourist | 22777 | +0.070224 | 1.355839 | 13.6515 | 6.7772 | -0.001984 |
| noun | author | 2488 | +0.128155 | 1.351595 | 13.6952 | 6.7985 | -0.003644 |
| noun | bishop | 29417 | +0.211001 | 1.341160 | 13.8039 | 6.8517 | -0.006094 |
| noun | dancer | 42411 | +0.285207 | 1.327362 | 13.9503 | 6.9233 | -0.008412 |
| noun | duck | 27985 | +0.144939 | 1.349898 | 13.7128 | 6.8071 | -0.004131 |
| noun | goat | 23244 | +0.171380 | 1.346796 | 13.7450 | 6.8229 | -0.004908 |
| noun | guitar | 12609 | +0.165757 | 1.347500 | 13.7377 | 6.8193 | -0.004742 |
| noun | hunter | 32290 | +0.145198 | 1.349870 | 13.7131 | 6.8073 | -0.004139 |
| noun | lawyer | 11115 | +0.181421 | 1.345481 | 13.7587 | 6.8296 | -0.005206 |
| noun | monk | 35295 | +0.117867 | 1.352531 | 13.6856 | 6.7938 | -0.003346 |
| noun | nurse | 15339 | +0.242124 | 1.335892 | 13.8594 | 6.8788 | -0.007049 |
| noun | painter | 27343 | +0.239714 | 1.336327 | 13.8548 | 6.8766 | -0.006974 |
| noun | prince | 24012 | +0.169459 | 1.347040 | 13.7425 | 6.8216 | -0.004851 |

- Geometry check maxima: angle32 3.685e-09, angle64 2.220e-16, d_dot_controls 5.551e-17, d_dot_plurality 6.245e-17, length32 6.026e-09, length64 4.441e-16, neutral32 6.089e-09, neutral64 1.665e-16, odd32 2.110e-09, odd64 2.776e-16, orth_E_controls 6.072e-17, orth_E_plurality 3.816e-17, orth_E_t 6.505e-17, orth_t_controls 4.857e-17, orth_t_plurality 4.163e-17, unit_controls 5.551e-16, unit_plurality 4.441e-16, unit_t 6.661e-16
