# Experiment 024 — preregistration

- Lock run `b04ce3526306a981` at commit `be74d23d086bd12d894db1f73e6a1b4160aeebd5`; design revision 2 (`9d03dee`), plan revision 1 (`608088c`); configuration `production`
- Calibration record content sha256 `09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8` (file `81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d`)
- Confirmation file content sha256 `87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87`: 4320 S2-TARGET prompts (manifest sha256 `fcc437fca9730b8599333eeac95807786570f0d03bd01e49f52daec76ed4e32d`)
- Calibration source: 023's exposed cells, data `d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4`, index `628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe` (content `d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16`)
- Model parameters sha256 `fd953f1c745299bedfd1f242fc15c7fe31211f79d8bc0d3ca46372962e7fc0e5`; embedding sha256 `9cd6f39b3adfdb74cffe046af459634982684bff9aef2a981760326f1748eaaf`; 020's locked states sha256 `26c21d6386ebfaf45cd33f1e9ceaae08a225d5c4ae3a5a9476e28bf1683b0f83`

## The primary test

Statistic: ρ = Spearman(nounness, MSE) across the fresh cues; MSE = Σ SSE_C / Σ n over the cue's exposed pairs, SSE_C = Σ_nouns (Δc − C)²; ranks ascending with ties at the average rank, then the Pearson correlation of the two rank vectors in float64.

- F_ρ = 0.24411074612857814; null₉₇.₅ = 0.3136960600375234; a PASS needs ρ ≥ max(F_ρ, null₉₇.₅) = 0.3136960600375234 (binding: null_975)
- Results, in precedence order:
  - `NOT_INTERPRETABLE`: ρ cannot be defined (a constant rank vector); no result
  - `GUARD_FAILURE`: ρ < null₉₇.₅: no evidence that the score predicts the readout error on fresh cues beyond chance
  - `ENVELOPE_ONLY_FAILURE`: null₉₇.₅ ≤ ρ < F_ρ: the score carries predictive information beyond chance, but less than the exposed relationship would lead one to expect
  - `PASS`: ρ ≥ max(F_ρ, null₉₇.₅): higher operational nounness prospectively predicted larger error of the frozen block-4/5 attention readout, at least as strongly as the exposed-like relationship and beyond chance

## The E–N disambiguation guard (an exact one-sided permutation test)

- D_EN = mean(log MSE_E) − mean(log MSE_N); the natural log of the per-cue MSE; E first, then N, each class in its frozen order
- E: apple, horse, doctor, king, rabbit, poet, dragon, lion; N: honest, polite, rude, sleepy, wise, lucky, merry, nervous
- Every one of the C(16, 8) = 12870 assignments is enumerated (itertools.combinations(range(n_E + n_N), n_E); the observed assignment is the first); exact: every binary64 log MSE as its exact dyadic rational (as_integer_ratio), subset sums in Python integers
- K = #{assignments S : D(S) ≥ D_EN}, the observed one included (ties count against the guard); PASS iff K ≤ max_upper; max_upper = 321, so the exact size is at most 321/12870; K / assignments (exact)
- NOT_INTERPRETABLE iff an MSE is not finite and positive
  - `NOT_INTERPRETABLE`: an E or N MSE is not finite and positive; the guard cannot be computed
  - `FAIL`: the ordinary singular nouns did not have larger error than the non-noun controls beyond chance (K > the bound); this is not evidence against nounness
  - `PASS`: the ordinary singular nouns had larger error than the non-noun controls beyond chance (an exact one-sided permutation test, K ≤ the bound)

## The outcome (frozen hierarchy)

| primary result | E–N guard | outcome |
|---|---|---|
| an incident | — | `no result is written` |
| NOT_INTERPRETABLE | any | `NOT_INTERPRETABLE` |
| GUARD_FAILURE or ENVELOPE_ONLY_FAILURE | any (descriptive only) | `NOUNNESS_PREDICTION_NOT_ESTABLISHED` |
| PASS | FAIL or NOT_INTERPRETABLE | `ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED` |
| PASS | PASS | `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS` |

- `NOT_INTERPRETABLE`: no result
- `NOUNNESS_PREDICTION_NOT_ESTABLISHED`: the primary requirement failed; its four-way result says how, and the E–N guard is descriptive only
- `ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED`: the score predicted the error, but the result does not separate nounness from plural morphology or measure semantics; the secondary contrasts describe the shape of the effect, with no winner
- `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`: the frozen weight-derived score prospectively predicted the frozen readout's error on new lexical representations, including substantial extrapolation beyond the exposed score range, and the ordinary singular nouns had larger error than the non-noun controls beyond chance — which neither of the two preregistered simple alternatives (plural morphology alone, measure class alone) predicts

- Simple: the guard eliminates the two preregistered simple alternatives, plurality-only and measure-class-only; it does not establish nounness as a unique causal factor or eliminate every correlated lexical property.
- Every outcome is predictive and associational; none establishes that nounness causally changes attention.
- Secondary: the secondary analyses (the per-group Spearman, normalized MSE, Var(Δc), R²_C, bias, slope, the line, the B/C/D/E factorial contrasts and the block-4/5 ladder) are computed after the outcome is written and can never rescue, alter or redefine it.
- Incidents: an incident carries no result; an identity incident never coexists with an outcome.
- Not shown by any outcome:
  - that nounness causes the attention change
  - that the score measures linguistic nounhood
  - which part of the score matters: similarity to the target nouns or dissimilarity to the exposed cues (both separate E from N)
  - that no other property that differs between ordinary nouns and these adjectives explains E > N
  - that the exposed OLS line has been validated over the extrapolated range (the line is secondary)
  - generality beyond the 108 exposed frames, the 79 nouns and this checkpoint

## Extrapolation

5 of the 8 E cues lie above the calibration population's maximum nounness (+0.135020); the fresh test is a prospective extrapolation test there. Calibration supplied no evidence for that range, and the post-hoc behaviour of 023's five spent cues in it is design motivation only.

## The fresh cues and their frozen scores

Line (exposed-data-fitted, prospectively frozen; secondary): log MSE = -2.481260357420486 + 1.2992017464639374 · nounness (residual sd 0.2570037337349923, n = 139).

| class | word | token id | nounness | predicted log MSE | above the calibration maximum |
|---|---|---|---|---|---|
| N | honest | 8274 | -0.22935071244762803 | -2.7792332035851928 | no |
| N | polite | 30405 | -0.1903187108201974 | -2.7285227589028516 | no |
| N | rude | 30446 | -0.12439741386297484 | -2.64287769476686 | no |
| N | sleepy | 48849 | -0.13710941690038347 | -2.659393151314116 | no |
| N | wise | 15822 | -0.19440870921151793 | -2.7338364919558895 | no |
| N | lucky | 13476 | -0.13300627493744047 | -2.6540623421098712 | no |
| N | merry | 49570 | -0.11509060380825145 | -2.630786270889755 | no |
| N | nervous | 11219 | -0.04507626505814516 | -2.5398235197080994 | no |
| B | gallons | 42616 | 0.1651398612902963 | -2.2667103612213206 | yes |
| B | ounces | 28409 | 0.0759690465528922 | -2.382561239461768 | no |
| B | acres | 20046 | 0.1631186714045987 | -2.269336294650754 | yes |
| B | herds | 47862 | 0.3965740010055436 | -1.9660307227118925 | yes |
| B | crowds | 24597 | 0.23715703273226257 | -2.173145526308525 | yes |
| B | bundles | 25663 | 0.2927946066382398 | -2.100861093120863 | yes |
| B | clusters | 9959 | 0.2617790108645215 | -2.1411566093176977 | yes |
| B | litres | 47026 | 0.006685069640962965 | -2.472575103267714 | no |
| D | gallon | 46740 | 0.04983962896221633 | -2.41650862442966 | no |
| D | ounce | 38831 | -0.06555391097164939 | -2.5664281130423943 | no |
| D | acre | 36982 | -0.031825820807209665 | -2.522608519395861 | no |
| D | herd | 33361 | 0.30293746421868784 | -2.0876834748382103 | yes |
| D | crowd | 9539 | 0.19785964446846188 | -2.2242007617723267 | yes |
| D | bundle | 13204 | 0.14735202818805862 | -2.289820345053557 | yes |
| D | cluster | 7368 | 0.16347458133969206 | -2.268873895841497 | yes |
| D | litre | 43803 | 0.04518919075932214 | -2.4225504818646826 | no |
| C | apples | 28580 | 0.22296560088986245 | -2.1915830593429955 | yes |
| C | horses | 12074 | 0.34591626318430735 | -2.031845344161155 | yes |
| C | doctors | 11576 | 0.31865040433065667 | -2.067269195602657 | yes |
| C | kings | 25346 | 0.20618955939604797 | -2.213378521750511 | yes |
| C | rabbits | 29948 | 0.32395329800181705 | -2.060379666883773 | yes |
| C | poets | 32976 | 0.3146659432933707 | -2.0724458143410165 | yes |
| C | dragons | 41705 | 0.31820898038249223 | -2.067842694367043 | yes |
| C | lions | 44536 | 0.30971817587347183 | -2.0788739624140464 | yes |
| E | apple | 19126 | 0.16746245512379346 | -2.2636928432565147 | yes |
| E | horse | 8815 | 0.29829447778124146 | -2.0937156509265487 | yes |
| E | doctor | 7345 | 0.21768061353376064 | -2.1984493241460825 | yes |
| E | king | 6963 | 0.11749007419039456 | -2.328617047840148 | no |
| E | rabbit | 17876 | 0.1286387706515007 | -2.3141326419270825 | no |
| E | poet | 17502 | 0.16599880501353617 | -2.2655944200359732 | yes |
| E | dragon | 22159 | 0.18822636757466396 | -2.236716331936919 | yes |
| E | lion | 27405 | 0.10845152134948685 | -2.3403599514765614 | no |
