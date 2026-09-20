# Experiment 017 — preregistered predictions (the transport head's own pattern change and its end-to-end ΔT from the decoded upstream state)

- Lock run `6b3e41362af5d557` at commit `ebc2206eca3dff3e52fd9abc874573c3fb9dd7d3`; confirmation set sha256 `07fc6ea974e22300d2e4ea87a517204f9474351cddde010e69acc9ce9e969c8e`; Experiment 016 lock sha256 `c21a69fa9347cc3ebcb4d113a719d4a34fb52668e513600dcfbe627c50884c77`
- Floors: Y1/Y2 head-row entry R² ≥ 0.95; Π token means Spearman ≥ 0.9 and R² ≥ 0.9; ΔT pairs R² ≥ 0.95 and token means R² ≥ 0.95; Y2 frame guard: every valid fresh frame's row entry R² ≥ 0.9; Y3 (cue-final pairs of both sets): the frozen pattern rejected iff its ΔT R² < 0.95 and ≥ 0.05 below Level 0's; the coordinated split descriptive
- Y1 table: 1584 rows (fresh tokens × exposed frames), each with the predicted row change of L03.H04 at p_t, F̂, Π̂, ΔT̂, the frozen-pattern ΔT̂ and the exact-head, −D and template-head rungs; Y2 rows are computed at confirm stage 1 and digested before any fresh cue prompt
- Program: layer 1: d_head 64, rotary_dim 16, base 10000, layer 2: d_head 64, rotary_dim 16, base 10000, layer 3: d_head 64, rotary_dim 16, base 10000; ΔT, F̂, Π̂ in residual units along d̂_T (σ_T 1.014)

| token | class | frames | **predicted ΔT̂** | F̂ (frozen) | **Π̂** | self ΔÂ(p_t,p_c) | exact head ΔT̂ | −D ΔT̂ | template head ΔT̂ | σ̂'/σ (L3) | decoded c_L (D) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trio | ordinal-or-numeral | 66 | **1.4902** | 1.5990 | **-0.1088** | -0.140 | 1.4916 | 1.5170 | 1.5932 | 1.026 | 0.220 |
| costly | adjective | 66 | **1.4249** | 1.4427 | **-0.0179** | -0.072 | 1.4255 | 1.4718 | 1.5505 | 1.012 | 0.031 |
| usual | determiner-like | 66 | **1.3027** | 1.3761 | **-0.0734** | -0.205 | 1.3046 | 1.3313 | 1.3800 | 1.031 | 0.252 |
| them | possessive-or-pronoun | 66 | **1.2215** | 1.2471 | **-0.0257** | 0.099 | 1.2210 | 1.2774 | 1.3834 | 1.014 | 0.092 |
| finite | quantity | 66 | **1.2072** | 1.2075 | **-0.0003** | -0.171 | 1.2076 | 1.2560 | 1.2871 | 1.050 | 0.166 |
| blunt | adjective | 66 | **1.1849** | 1.1490 | **0.0359** | -0.277 | 1.1860 | 1.2037 | 1.2411 | 1.035 | 0.193 |
| us | possessive-or-pronoun | 66 | **1.1845** | 1.2116 | **-0.0271** | 0.004 | 1.1854 | 1.2327 | 1.3182 | 0.952 | 0.149 |
| partial | quantity | 66 | **1.1776** | 1.1473 | **0.0303** | -0.210 | 1.1782 | 1.2231 | 1.2522 | 1.073 | 0.086 |
| paper | adjective | 66 | **1.1689** | 1.1498 | **0.0191** | -0.236 | 1.1698 | 1.2036 | 1.2402 | 0.990 | 0.137 |
| leading | determiner-like | 66 | **1.1569** | 1.1053 | **0.0516** | -0.218 | 1.1583 | 1.2020 | 1.2414 | 1.003 | 0.115 |
| violet | adjective | 66 | **1.1377** | 1.1361 | **0.0016** | -0.175 | 1.1391 | 1.1698 | 1.2182 | 1.115 | 0.064 |
| nearby | determiner-like | 66 | **1.1289** | 1.1356 | **-0.0067** | -0.086 | 1.1309 | 1.1831 | 1.2250 | 1.037 | 0.181 |
| further | determiner-like | 66 | **1.1141** | 1.1052 | **0.0089** | -0.143 | 1.1165 | 1.1737 | 1.1942 | 1.007 | 0.186 |
| metal | adjective | 66 | **1.1105** | 1.0523 | **0.0583** | -0.234 | 1.1113 | 1.1487 | 1.1769 | 1.047 | 0.035 |
| score | ordinal-or-numeral | 66 | **1.1035** | 1.0375 | **0.0660** | -0.160 | 1.1051 | 1.1564 | 1.1986 | 1.035 | 0.123 |
| leather | adjective | 66 | **1.0823** | 0.9927 | **0.0896** | -0.244 | 1.0833 | 1.1200 | 1.1447 | 1.092 | 0.165 |
| complete | quantity | 66 | **1.0433** | 0.9777 | **0.0656** | -0.141 | 1.0440 | 1.0876 | 1.1313 | 1.002 | 0.163 |
| adjacent | determiner-like | 66 | **1.0161** | 1.0126 | **0.0035** | -0.104 | 1.0179 | 1.0772 | 1.1071 | 1.002 | 0.054 |
| none | possessive-or-pronoun | 66 | **0.9161** | 0.9156 | **0.0006** | 0.012 | 0.9178 | 0.9493 | 1.0246 | 1.011 | 0.225 |
| twice | ordinal-or-numeral | 66 | **0.8956** | 0.8295 | **0.0661** | -0.176 | 0.8957 | 0.9123 | 0.9544 | 1.075 | -0.005 |
| whole | quantity | 66 | **0.8776** | 0.7743 | **0.1033** | -0.129 | 0.8786 | 0.9087 | 0.9676 | 1.021 | -0.060 |
| you | possessive-or-pronoun | 66 | **0.8203** | 0.9051 | **-0.0849** | 0.150 | 0.8199 | 0.8985 | 0.9532 | 0.991 | -0.015 |
| entire | quantity | 66 | **0.7876** | 0.7468 | **0.0408** | -0.046 | 0.7889 | 0.8336 | 0.8845 | 1.013 | -0.071 |
| once | ordinal-or-numeral | 66 | **0.7406** | 0.7305 | **0.0101** | -0.020 | 0.7392 | 0.7828 | 0.8277 | 1.001 | -0.008 |
