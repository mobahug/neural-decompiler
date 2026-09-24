# Experiment 022 — report

- Run `be529d6f87a9f24d` at commit `0de0fb8b066876766289fd51c290a7772a452c6b`; phases: calibrate complete, confirm complete, lock complete, replicate-021 complete, report complete
- Design revision 4 (`219cdc5`), plan revision 3 (`e6d8299`)


## The eight conditions (the result; no aggregate label)

| condition | statistic | value | envelope | meaning guard | G | CDF percentile (defined / undefined draws) | result |
|---|---|---|---|---|---|---|---|
| Y1/C1 block-0 attention dominates the cue-final gap | share(Bv) + share(Bp), cue-final | 0.9479 | ≥ v₍250₎ = 0.901075 | s1 ≥ 0.5 | 0.1602 | — | **PASS** |
| Y1/C2 the layer-1–2 reductions contribute little to the cue-final gap | share(R), cue-final | -0.0000 | ≤ v₍9751₎ = 0.005595 | s2 ≤ 0.1 | 0.1602 | — | **PASS** |
| Y1/C3 the layer-1–2 reductions contribute little to the coordinated gap | share(R), coordinated | 0.0060 | ≤ v₍9751₎ = 0.006904 | s3 ≤ 0.1 | 0.3375 | — | **PASS** |
| Y1/C4 block-0 cue→target attention contributes positively to the coordinated gap | share(T), coordinated | 0.4081 | ≥ v₍250₎ = 0.327059 | s4 > 0.0 | 0.3375 | — | **PASS** |
| Y2/C1 block-0 attention dominates the cue-final gap | share(Bv) + share(Bp), cue-final | 0.9381 | ≥ v₍250₎ = 0.888785 | s1 ≥ 0.5 | 0.1599 | — | **PASS** |
| Y2/C2 the layer-1–2 reductions contribute little to the cue-final gap | share(R), cue-final | 0.0039 | ≤ v₍9751₎ = 0.008226 | s2 ≤ 0.1 | 0.1599 | — | **PASS** |
| Y2/C3 the layer-1–2 reductions contribute little to the coordinated gap | share(R), coordinated | 0.0069 | ≤ v₍9751₎ = 0.014172 | s3 ≤ 0.1 | 0.4101 | — | **PASS** |
| Y2/C4 block-0 cue→target attention contributes positively to the coordinated gap | share(T), coordinated | 0.3834 | ≥ v₍250₎ = 0.207165 | s4 > 0.0 | 0.4101 | — | **PASS** |

Readings (frozen, design revision 4):

- Y1/C1: **PASS** — guard and envelope hold: the attribution lies within the exposed-like envelope and keeps the claim's meaning.
- Y1/C2: **PASS** — guard and envelope hold: the attribution lies within the exposed-like envelope and keeps the claim's meaning.
- Y1/C3: **PASS** — guard and envelope hold: the attribution lies within the exposed-like envelope and keeps the claim's meaning.
- Y1/C4: **PASS** — guard and envelope hold: the attribution lies within the exposed-like envelope and keeps the claim's meaning.
- Y2/C1: **PASS** — guard and envelope hold: the attribution lies within the exposed-like envelope and keeps the claim's meaning.
- Y2/C2: **PASS** — guard and envelope hold: the attribution lies within the exposed-like envelope and keeps the claim's meaning.
- Y2/C3: **PASS** — guard and envelope hold: the attribution lies within the exposed-like envelope and keeps the claim's meaning.
- Y2/C4: **PASS** — guard and envelope hold: the attribution lies within the exposed-like envelope and keeps the claim's meaning.

- The CDF percentile is descriptive only; the frozen envelope decides. For C2 and C3 (upper bounds) a high percentile lies toward the unfavorable upper tail.
- Layers 1–2: R off is Experiment 017's reduced chain with its layer-1/2 reference rows through the cue position p_c (as validated in 017–019); Experiment 020's rows through p_t (the defect of the 020/021 errata) enter only the descriptive historical comparator.
- C4: a PASS of C4 establishes a positive contribution of block-0 cue→target attention only, never a large or substantial one.
- None: the eight condition results are the result, each reported and read on its own; Y1 and Y2 are never pooled; no all-pass requirement.
- What a pass does not show: a pass does not show that any corrected program would predict well (022 builds none; the full composition is the identity endpoint only), nor which of 016/017's reductions carries σ_R.
- Incidents carry no result.

## Shapley values and shares

| population / group | G | φ R | φ emb | φ Bv | φ Bp | φ T | σ R | σ emb | σ Bv | σ Bp | σ T | efficiency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Y1/cue_final | 0.1602 | -0.0000 | 0.0084 | 0.1012 | 0.0507 | 0.0000 | -0.0000 | 0.0521 | 0.6317 | 0.3162 | 0.0000 | 0.0e+00 |
| Y1/coordinated | 0.3375 | 0.0020 | 0.0091 | 0.1844 | 0.0043 | 0.1378 | 0.0060 | 0.0269 | 0.5462 | 0.0128 | 0.4081 | 5.6e-17 |
| Y2/cue_final | 0.1599 | 0.0006 | 0.0093 | 0.0950 | 0.0550 | 0.0000 | 0.0039 | 0.0581 | 0.5941 | 0.3440 | 0.0000 | 0.0e+00 |
| Y2/coordinated | 0.4101 | 0.0028 | 0.0142 | 0.2211 | 0.0147 | 0.1572 | 0.0069 | 0.0346 | 0.5392 | 0.0358 | 0.3834 | 5.6e-17 |

## Descriptive records (no outcome force)

- Ladder Y1/cue_final: Level 0 0.8005, R 0.8004, R+emb 0.8065, R+emb+Bv+Bp 0.9607, all 0.9607; R only 0.8004; inputs only (reduced layers 1–2) 0.9604
- Ladder Y1/coordinated: Level 0 0.6277, R 0.6315, R+emb 0.6448, R+emb+Bv+Bp 0.8587, all 0.9652; R only 0.6315; inputs only (reduced layers 1–2) 0.9648
- Ladder Y2/cue_final: Level 0 0.7927, R 0.7934, R+emb 0.8002, R+emb+Bv+Bp 0.9526, all 0.9526; R only 0.7934; inputs only (reduced layers 1–2) 0.9518
- Ladder Y2/coordinated: Level 0 0.5440, R 0.5486, R+emb 0.5724, R+emb+Bv+Bp 0.8327, all 0.9541; R only 0.5486; inputs only (reduced layers 1–2) 0.9534
- Y1/coordinated/class/adjective: G 0.4536; shares R 0.007, emb -0.006, Bv 0.562, Bp -0.039, T 0.475
- Y1/coordinated/class/determiner-like: G 0.4962; shares R 0.006, emb 0.089, Bv 0.574, Bp 0.016, T 0.314
- Y1/coordinated/class/possessive-or-pronoun: G 0.1603; shares R 0.004, emb -0.047, Bv 0.504, Bp 0.071, T 0.467
- Y1/coordinated/class/quantity: G 0.3768; shares R 0.006, emb 0.039, Bv 0.522, Bp 0.034, T 0.399
- Y1/cue_final/class/adjective: G 0.1746; shares R -0.003, emb 0.051, Bv 0.615, Bp 0.337, T 0.000
- Y1/cue_final/class/determiner-like: G 0.1548; shares R 0.001, emb 0.057, Bv 0.661, Bp 0.281, T 0.000
- Y1/cue_final/class/possessive-or-pronoun: G 0.1681; shares R -0.000, emb 0.049, Bv 0.623, Bp 0.328, T 0.000
- Y1/cue_final/class/quantity: G 0.1517; shares R 0.003, emb 0.053, Bv 0.637, Bp 0.307, T 0.000
- Y1/template/cardinal: G 0.1662; shares R -0.001, emb 0.063, Bv 0.680, Bp 0.258, T 0.000
- Y1/template/coordinated-adjective: G 0.3375; shares R 0.006, emb 0.027, Bv 0.546, Bp 0.013, T 0.408
- Y1/template/quantifier: G 0.1837; shares R 0.001, emb 0.041, Bv 0.580, Bp 0.378, T 0.000
- Y2/coordinated/class/adjective: G 0.5634; shares R 0.005, emb 0.010, Bv 0.556, Bp -0.006, T 0.436
- Y2/coordinated/class/determiner-like: G 0.5715; shares R 0.008, emb 0.091, Bv 0.540, Bp 0.037, T 0.324
- Y2/coordinated/class/possessive-or-pronoun: G 0.1597; shares R 0.008, emb -0.031, Bv 0.452, Bp 0.099, T 0.471
- Y2/coordinated/class/quantity: G 0.5370; shares R 0.008, emb 0.028, Bv 0.555, Bp 0.051, T 0.358
- Y2/cue_final/class/adjective: G 0.1646; shares R 0.001, emb 0.069, Bv 0.604, Bp 0.326, T 0.000
- Y2/cue_final/class/determiner-like: G 0.1590; shares R 0.009, emb 0.074, Bv 0.591, Bp 0.326, T 0.000
- Y2/cue_final/class/possessive-or-pronoun: G 0.1891; shares R 0.003, emb 0.037, Bv 0.597, Bp 0.362, T 0.000
- Y2/cue_final/class/quantity: G 0.1295; shares R 0.002, emb 0.068, Bv 0.579, Bp 0.351, T 0.000
- Y2/template/cardinal: G 0.1342; shares R 0.005, emb 0.072, Bv 0.620, Bp 0.303, T 0.000
- Y2/template/coordinated-adjective: G 0.4101; shares R 0.007, emb 0.035, Bv 0.539, Bp 0.036, T 0.383
- Y2/template/quantifier: G 0.1962; shares R 0.003, emb 0.047, Bv 0.573, Bp 0.377, T 0.000
- Δx3 relative error Y1/coordinated: Level 0@p_c median 5.38e-01 max 8.80e-01, Level 0@p_t median 6.60e-01 max 9.34e-01, R@p_c median 5.38e-01 max 8.96e-01, R@p_t median 6.61e-01 max 9.32e-01, R+emb@p_c median 5.24e-01 max 8.68e-01, R+emb@p_t median 6.55e-01 max 9.15e-01, R+emb+Bv+Bp@p_c median 3.33e-06 max 9.75e-06, R+emb+Bv+Bp@p_t median 4.14e-01 max 7.08e-01, all@p_c median 3.33e-06 max 9.75e-06, all@p_t median 4.07e-06 max 1.38e-05
- Δx3 relative error Y1/cue_final: Level 0@p_c median 5.24e-01 max 9.86e-01, R@p_c median 5.26e-01 max 9.86e-01, R+emb@p_c median 5.10e-01 max 9.86e-01, R+emb+Bv+Bp@p_c median 3.30e-06 max 4.23e-05, all@p_c median 3.30e-06 max 4.23e-05
- Δx3 relative error Y2/coordinated: Level 0@p_c median 5.12e-01 max 7.22e-01, Level 0@p_t median 5.81e-01 max 8.97e-01, R@p_c median 5.13e-01 max 7.24e-01, R@p_t median 5.83e-01 max 8.97e-01, R+emb@p_c median 4.93e-01 max 7.10e-01, R+emb@p_t median 5.73e-01 max 8.78e-01, R+emb+Bv+Bp@p_c median 3.21e-06 max 9.34e-06, R+emb+Bv+Bp@p_t median 4.12e-01 max 6.69e-01, all@p_c median 3.21e-06 max 9.34e-06, all@p_t median 4.08e-06 max 9.53e-06
- Δx3 relative error Y2/cue_final: Level 0@p_c median 5.03e-01 max 9.57e-01, R@p_c median 5.04e-01 max 9.58e-01, R+emb@p_c median 4.87e-01 max 9.57e-01, R+emb+Bv+Bp@p_c median 3.14e-06 max 2.20e-05, all@p_c median 3.14e-06 max 2.20e-05
- Level 0 Y1/coordinated: flattened R² 0.6277 with 017's wiring (the empty coalition); historical comparator (the Level 0 that 020/021 ran, rows through p_t) -0.2257
- Level 0 Y1/cue_final: flattened R² 0.8005 with 017's wiring (the empty coalition); historical comparator (the Level 0 that 020/021 ran, rows through p_t) 0.8005
- Level 0 Y2/coordinated: flattened R² 0.5440 with 017's wiring (the empty coalition); historical comparator (the Level 0 that 020/021 ran, rows through p_t) -0.2839
- Level 0 Y2/cue_final: flattened R² 0.7927 with 017's wiring (the empty coalition); historical comparator (the Level 0 that 020/021 ran, rows through p_t) 0.7927
- Block-0 profile Y1/coordinated: self-weight [0.2916897100075006, 0.0996048754811383, 0.6461148331275236, 0.9156683804837089, 0.17742457337254672, 0.7655042756904789, 0.3455641294552416, 0.40737729081247454]; value-term norm [0.998471509662355, 0.2657668899507367, 1.4964061707397678, 1.6970174615709814, 0.7602267490997854, 2.827202541284095, 2.162164336672433, 1.3533537814473324]; p_t→p_c [0.04509995678412827, 0.33714277049612973, 0.002050463040022173, 5.683361835608286e-05, 0.12262770356735161, 0.6042227222347824, 0.2851258683191067, 0.10293703017124707]
- Block-0 profile Y1/cue_final: self-weight [0.1922937257511134, 0.0962315866627124, 0.5993330303603975, 0.9992378435806276, 0.16892309361114205, 0.4764086628669938, 0.38348854149464023, 0.5758045031967831]; value-term norm [0.7685553526494768, 0.2647326356782126, 1.6803049378232104, 2.0287983008481967, 0.7032181093248859, 2.1226527379608067, 2.3828169815480886, 2.160398699607599]; p_t→p_c None
- Block-0 profile Y2/coordinated: self-weight [0.2954027688177666, 0.10658866628655164, 0.6289054554067013, 0.9155764771704038, 0.1526132917489723, 0.7119293458078284, 0.2871544177172031, 0.4157068375515301]; value-term norm [1.0431048219949386, 0.2892022773271166, 1.4843138412222951, 1.6970268747409218, 0.7202432029076513, 2.9588549943684863, 1.8701379017566062, 1.4774716400687067]; p_t→p_c [0.07223613425829875, 0.3604661038374972, 0.0017256860325872753, 0.00013306198227241734, 0.15627549468470983, 0.5088360089640686, 0.24094132960845713, 0.05598413566860708]
- Block-0 profile Y2/cue_final: self-weight [0.18166096175125435, 0.08877746092477586, 0.5993097615558355, 0.999237668332474, 0.17227255633706695, 0.4099190785595469, 0.36395972805164617, 0.5092109952355285]; value-term norm [0.7413933363638544, 0.24320530947164884, 1.6821871441406022, 2.0289534669159592, 0.7103211344097377, 1.8128909994511668, 2.2969822404132207, 1.9295252746139178]; p_t→p_c None
- Validity (descriptive, selects nothing) cardinal-022-1: True
- Validity (descriptive, selects nothing) cardinal-022-2: True
- Validity (descriptive, selects nothing) cardinal-022-3: True
- Validity (descriptive, selects nothing) cardinal-022-4: True
- Validity (descriptive, selects nothing) cardinal-022-5: True
- Validity (descriptive, selects nothing) cardinal-022-6: True
- Validity (descriptive, selects nothing) coordinated-adjective-022-1: True
- Validity (descriptive, selects nothing) coordinated-adjective-022-2: True
- Validity (descriptive, selects nothing) coordinated-adjective-022-3: True
- Validity (descriptive, selects nothing) coordinated-adjective-022-4: True
- Validity (descriptive, selects nothing) coordinated-adjective-022-5: True
- Validity (descriptive, selects nothing) coordinated-adjective-022-6: True
- Validity (descriptive, selects nothing) quantifier-022-1: True
- Validity (descriptive, selects nothing) quantifier-022-2: True
- Validity (descriptive, selects nothing) quantifier-022-3: True
- Validity (descriptive, selects nothing) quantifier-022-4: True
- Validity (descriptive, selects nothing) quantifier-022-5: True
- Validity (descriptive, selects nothing) quantifier-022-6: True
- Stage-2 gates (maxima): I1 1.6446723193563884e-05, I2 4.440892098500626e-16, I3 4.22832397152792e-05, I4 7.693138650566311e-05

## EXPLORATORY replication on Experiment 021's spent set (after the report; no result names; affects nothing)

- Checks: I4 {'max': 4.6748638698357325e-05, 'at': 'Y1|average|cardinal-009-1', 'tolerance': 0.001, 'passed': True}, I5 {'max': 0.0, 'at': '', 'tolerance': 0.0, 'passed': True, 'compared': "the historical Level 0 (020's rows through p_t) against 021's stored Level 0"}, I6 max 1.1e-16; I1–I3 unavailable
- Y1: pairs {'cue_final': 1728, 'coordinated': 864}; s1–s4 {'C1': 0.9333848718448023, 'C2': 0.0031126199570528728, 'C3': 0.006426190352500834, 'C4': 0.37899132623374077}
- Y2: pairs {'cue_final': 288, 'coordinated': 144}; s1–s4 {'C1': 0.9339515111512411, 'C2': 0.004117118602702491, 'C3': 0.012429315702625749, 'C4': 0.2984019431728172}
