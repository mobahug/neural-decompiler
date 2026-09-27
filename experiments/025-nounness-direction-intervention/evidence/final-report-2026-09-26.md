# Experiment 025 — report

- Run `914fa3afff5c73ec`; phases: lock complete, confirm complete, report not_started
- Design revision 1 (`c0885e5`), plan revision 1 (`7d90d28`); configuration `production`

## The outcome

**`CAUSAL_EFFECT_NOT_ESTABLISHED`** — A failed: the +nounness intervention did not consistently produce greater frozen-readout approximation error than the −nounness intervention.

Not claimed by any label:
- semantic nounhood as the model's variable
- a general routing controller
- algorithm selection in general
- generalization beyond this checkpoint and this mechanism
- that the effect excludes a plurality-direction contribution (the nounness direction is partly aligned with plurality, cos +0.235)

- The criterion: pre-registered sign-count criteria with an exact Binomial(40, 0.5) reference tail under H0: P(positive) <= 0.5 for independent cue units; not unconditional, design-based randomization tests (the 40 lexical items are selected mechanically, not sampled at random); the causal reading comes from the within-cue intervention (+θ and −θ are both run for every cue), and the sign count measures consistency across the frozen test population; ties and zeros count against PASS; reference tail 5288280983/274877906944 = 0.01923865414210013

| statistic | positive | of | threshold | result | adjectives | nouns |
|---|---|---|---|---|---|---|
| A | 26 | 40 | 27 | **FAIL** | 20 | 6 |
| B | 23 | 40 | 27 | **FAIL** | 20 | 3 |
| G | 37 | 40 | 27 | **PASS** | 20 | 17 |

## What the statistics mean

- **A**: A_i > 0: the +nounness intervention produced greater frozen-readout approximation error than the matched −nounness intervention; A alone does not establish that +θ rises above the unperturbed baseline or that −θ falls below it (the baseline and the even component are descriptive only)
- **B**: B_i = A_i − mean_j |A_ij|: the nounness-direction effect in the predicted sign exceeds the typical size of an equal-angle, nounness-neutral directional effect; absolute values because the signed random mean has expectation 0 by the u → −u symmetry; t̂ is not assumed exchangeable with the controls
- **G**: G_i = ½[D_attn(+θ) − D_attn(−θ)], D_attn the total-variation distance of the captured L4/L5 rows at p_t from the locked 020 reference rows, equally averaged over the frozen 16 heads, then over the 108 frames; equal weighting is frozen before outcomes and avoids outcome-based head selection; irrelevant or oppositely responding heads can dilute or oppose the aggregate signal, so failure of G is conservative with respect to this particular routing summary
- **Strata**: the positive counts of A, B and G are reported for the 20 adjectives and the 20 ordinary singular nouns separately; they cannot change the label; if a global PASS is strongly concentrated in one stratum the final claim must say so (reporting trigger: a stratum below 14 of 20)
- No stratum is below its reporting trigger on a globally passing statistic.
- **Secondary records**: the magnitudes, the even components and baselines, the half dose, the plurality control, nMSE, the ladder, G against the random directions, the per-template counts and the causal fraction are computed after the result is written and can never rescue, alter or redefine it
- **The patch-path check**: the confirm-time patch-path check on four already-executed keys (a plain capture against a patched θ = 0 run, and the embedding hook against the block-0 residual input, bit for bit) only validates the intervention plumbing; a pass does not strengthen the scientific result

| stratum | word | A | B | G |
|---|---|---|---|---|
| adjective | anxious | +0.257429 | +0.219335 | +0.018879 |
| adjective | cheerful | +0.271194 | +0.188700 | +0.010888 |
| adjective | curious | +0.264970 | +0.208122 | +0.024291 |
| adjective | jealous | +0.148249 | +0.103469 | +0.024276 |
| adjective | lonely | +0.319755 | +0.267042 | +0.018539 |
| adjective | nasty | +0.339904 | +0.290835 | +0.019259 |
| adjective | careful | +0.443823 | +0.324167 | +0.035595 |
| adjective | careless | +0.135820 | +0.087842 | +0.000647 |
| adjective | famous | +0.234023 | +0.204123 | +0.020676 |
| adjective | friendly | +0.284867 | +0.211988 | +0.026957 |
| adjective | gorgeous | +0.297097 | +0.257425 | +0.032306 |
| adjective | hungry | +0.308622 | +0.277208 | +0.018218 |
| adjective | weary | +0.276421 | +0.245698 | +0.031534 |
| adjective | wicked | +0.241402 | +0.175168 | +0.012393 |
| adjective | ugly | +0.325902 | +0.275144 | +0.017490 |
| adjective | vivid | +0.196697 | +0.128286 | +0.009568 |
| adjective | vague | +0.464140 | +0.379965 | +0.019907 |
| adjective | rapid | +0.195304 | +0.116666 | +0.007854 |
| adjective | rigid | +0.239528 | +0.217916 | +0.017053 |
| adjective | clever | +0.160296 | +0.132625 | +0.009171 |
| noun | soldier | -0.011332 | -0.079799 | +0.006837 |
| noun | sailor | -0.082372 | -0.113800 | +0.009748 |
| noun | priest | +0.008827 | -0.011955 | +0.012587 |
| noun | knight | +0.031537 | -0.011594 | +0.021130 |
| noun | onion | -0.116610 | -0.191370 | +0.030211 |
| noun | carrot | -0.058519 | -0.117947 | +0.004309 |
| noun | pirate | +0.111021 | +0.078829 | +0.024477 |
| noun | tourist | +0.148004 | +0.082017 | +0.035307 |
| noun | author | -0.000946 | -0.054241 | +0.012053 |
| noun | bishop | -0.053556 | -0.080571 | +0.007213 |
| noun | dancer | -0.126271 | -0.161514 | -0.000217 |
| noun | duck | +0.084376 | +0.013360 | +0.034236 |
| noun | goat | -0.199491 | -0.260742 | +0.014176 |
| noun | guitar | -0.115695 | -0.167298 | +0.023842 |
| noun | hunter | +0.015874 | -0.023379 | +0.008851 |
| noun | lawyer | -0.131968 | -0.182086 | -0.002462 |
| noun | monk | -0.045195 | -0.081872 | +0.006189 |
| noun | nurse | -0.159224 | -0.232609 | -0.004152 |
| noun | painter | -0.071416 | -0.117711 | +0.003053 |
| noun | prince | -0.049329 | -0.081657 | +0.005236 |

## Validity (all passed for a result to exist)

- I1 max 2.1541116871581778e-05 (at knight|cardinal-017-3|noun+0.16), I3 max 4.35303313102457e-05 (at painter|quantifier-016-4|rand5-0.32), I4 max 9.004566446524365e-05 (at lonely|cardinal-new-2|rand6-0.32), level1_outcome_bearing max 0.011862701913658569 (at knight|cardinal-012-1|rand4+0.32)
- C recomputed from the saved Δx3: bit for bit True over 90720 runs
- The prompt accounting: {'equal': True, 'executed': 90720, 'ledger': 90720, 'manifest': 90720}
- The patch-path check: passed True

## Descriptive records (no outcome force)

- Magnitudes: {'A': {'mean': 0.11457901336066345, 'median': 0.14191208776278663}, 'B': {'mean': 0.06289458391769001, 'median': 0.08492943627022942}, 'G': {'mean': 0.01570307697935662, 'median': 0.015614380833868485}}
- Counts: {'A_half_positive': 25, 'A_nmse_positive': 24, 'A_plurality_positive': 16, 'G_minus_random_positive': 35, 'minus_below_base': 30, 'plus_above_base': 23}
- A positive by template: {'cardinal': 26, 'coordinated-adjective': 31, 'quantifier': 25}
- Even components: noun 0.014446877229105298, random 0.014085822525458298
- Strata: {'A': {'adjective': {'below_trigger': False, 'of': 20, 'positive': 20}, 'noun': {'below_trigger': False, 'of': 20, 'positive': 6}}, 'B': {'adjective': {'below_trigger': False, 'of': 20, 'positive': 20}, 'noun': {'below_trigger': False, 'of': 20, 'positive': 3}}, 'G': {'adjective': {'below_trigger': False, 'of': 20, 'positive': 20}, 'noun': {'below_trigger': False, 'of': 20, 'positive': 17}}}
- Causal fraction: {'fraction': 0.34134442588745245, 'median_A': 0.14191208776278663, 'odd': 0.32, 'slope_024_exposed': 1.2992017464639374}
- Ladder base: block-5 attention 0.32268654564552046, block-4 attention 0.6772975524875838
- Ladder noun+0.32: block-5 attention 0.30679519427992763, block-4 attention 0.6931903591159422
- Ladder noun-0.32: block-5 attention 0.3292839297598919, block-4 attention 0.670699104031455
