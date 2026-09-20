# Experiment 017 Report

- Run ID: `6b3e41362af5d557`
- Confirmation sha256: `07fc6ea974e22300d2e4ea87a517204f9474351cddde010e69acc9ce9e969c8e`
- Experiment 016 lock sha256: `c21a69fa9347cc3ebcb4d113a719d4a34fb52668e513600dcfbe627c50884c77`
- Protocol/code commit at explore: `ebc2206eca3dff3e52fd9abc874573c3fb9dd7d3`

## Phases

- `confirm`: `complete`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

## Tier A — exposed pool (calibration record only; the floors are frozen constants)

- Replication of Experiment 016: 7764 pairs, max deviation 0.00e+00
- Identities (checks only): I1_patched_rows 2.2e-05, I1_reference_rows 9.8e-06, I2_chain 3.4e-06, I3_head_split 9.9e-07, I4_x3 8.5e-06, I5_head_row 1.2e-05, I5_reference_head_row 8.6e-06, I6_dT 3.3e-05, I7_split 8.6e-07, head_level1_recovery 6.4e-12, level0_015_recovery 1.0e-14, level1_recovery 1.4e-14, neuron_sum 2.3e-08, p1_cross_check 6.6e-16, rho_identity 1.1e-07; read weight vs Experiment 011 lock 0.0e+00
- Program: layer 1: d_head 64, rotary_dim 16, base 10000, layer 2: d_head 64, rotary_dim 16, base 10000, layer 3: d_head 64, rotary_dim 16, base 10000; layer-3 bases over cardinal: 22, coordinated-adjective: 22, quantifier: 22 frames
- Head row (7764 pairs): Level 0 entry R² 0.996 (TV ratio 0.052); per-frame minimum 0.974; self weight Spearman 0.997, R² 0.994, MAE 0.0073, bias 0.0017 (n 7764)
- Π: token means (207) Spearman 0.999, R² 0.999, MAE 0.0022, bias 0.0008 (n 207); pairs Spearman 0.997, R² 0.996, MAE 0.0056, bias 0.0007 (n 7764); Π spread sd 0.1050
- ΔT: token means Spearman 1.000, R² 1.000, MAE 0.0034, bias -0.0021 (n 207); pairs Spearman 1.000, R² 1.000, MAE 0.0079, bias -0.0016 (n 7764); F̂ vs F pairs Spearman 1.000, R² 1.000, MAE 0.0066, bias -0.0023 (n 7764); ΔT spread sd 0.4194
- Frozen pattern: pairs Spearman 0.976, R² 0.935, MAE 0.0970, bias 0.0092 (n 7764); token means Spearman 0.986, R² 0.936, MAE 0.0758, bias 0.0239 (n 207)
- Split — cue-final: 5176 pairs, 207 tokens: Level 0 ΔT R² 1.000, frozen pattern 0.918, gap 0.081; Π variance share 0.081; row entry R² 0.996; Π R² 0.996
- Split — coordinated: 2588 pairs, 206 tokens: Level 0 ΔT R² 1.000, frozen pattern 0.972, gap 0.027; Π variance share 0.028; row entry R² 0.996; Π R² 0.992
- Template cardinal: 2588 pairs, 206 tokens: Level 0 ΔT R² 0.998, frozen pattern 0.866, gap 0.132; Π variance share 0.131; row entry R² 0.993; Π R² 0.993
- Template coordinated-adjective: 2588 pairs, 206 tokens: Level 0 ΔT R² 1.000, frozen pattern 0.972, gap 0.027; Π variance share 0.028; row entry R² 0.996; Π R² 0.992
- Template quantifier: 2588 pairs, 206 tokens: Level 0 ΔT R² 1.000, frozen pattern 0.832, gap 0.167; Π variance share 0.167; row entry R² 0.998; Π R² 0.998
  - ladder (rows / ΔT / Π R²): `level0` 0.996 / 1.000 / 0.996; `exact_head` 0.999 / 1.000 / 0.999; `no_D` 0.944 / 0.962 / 0.828; `template_head` 0.782 / 0.932 / 0.543; `level1` (identity) 1.000 / 1.000 / 1.000; `frozen` — / 0.935 / ≡ 0 (`no_D` moves block 2's operating point at p_c only)
  - ablation costs rows: `no_D` 0.052, `template_head` 0.215 (template head ≥ −D holds: True); Π: `no_D` 0.168, `template_head` 0.453 (holds: True); frozen gap 0.064; Π variance share 0.065
  - x₃ remainder (relative) mean 0.0206 max 0.1508; scale remainder |σ(x₃') − σ̂'| mean 0.0009; σ̂'/σ mean 1.024
  - decoded c_L: `c_L_level0F` pairs R² 0.840 / token means R² 0.936; `c_L_level0D` pairs R² 1.000 / token means R² 1.000; `c_L_level1` pairs R² 1.000 / token means R² 1.000; Level 1's residual max 0.000003

## Lock

- Candidate lock sha256 `b4fc9014ade7d21d2cd2e5be391ce6234e46fb46887880c0c4d03517d411ed72`; predictions sha256 `14fed442a8bc943f365a7b3768687ae5cf04b4792e5f1606da186c964293f9e6`

## Confirmation — stage 1 (fresh frames' reference states; prediction table digested before any fresh cue prompt)

- Table rows 288; digest `7b62fa03c633547f954c9cd27b562b40c88151ff8c6c6a6080c2be151668302f`; commit `4b6f766a6b68807591ef6276c5a997faa18ee586`
  - cardinal-017-1: valid (plural head change 0.992, cue effect 79/72; p_c 4, p_t 4)
  - cardinal-017-2: valid (plural head change 1.130, cue effect 79/72; p_c 3, p_t 3)
  - cardinal-017-3: valid (plural head change 1.161, cue effect 79/72; p_c 3, p_t 3)
  - cardinal-017-4: valid (plural head change 1.016, cue effect 79/72; p_c 4, p_t 4)
  - coordinated-adjective-017-1: valid (plural head change 2.172, cue effect 79/72; p_c 5, p_t 6)
  - coordinated-adjective-017-2: valid (plural head change 2.409, cue effect 79/72; p_c 4, p_t 5)
  - coordinated-adjective-017-3: valid (plural head change 2.633, cue effect 79/72; p_c 5, p_t 6)
  - coordinated-adjective-017-4: valid (plural head change 2.055, cue effect 79/72; p_c 5, p_t 6)
  - quantifier-017-1: valid (plural head change 1.913, cue effect 79/72; p_c 3, p_t 3)
  - quantifier-017-2: valid (plural head change 2.165, cue effect 79/72; p_c 3, p_t 3)
  - quantifier-017-3: valid (plural head change 2.064, cue effect 79/72; p_c 3, p_t 3)
  - quantifier-017-4: valid (plural head change 2.212, cue effect 79/72; p_c 3, p_t 3)

## Confirmation — stage 2 — `HEAD_PATTERN_PREDICTED_TOKENS | HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL | FROZEN_PATTERN_NOT_REJECTED`

- Y1 (strict prospective: fresh cues × exposed frames): scored tokens 24 (precondition ok); head row entry R² 0.997; Π token means Spearman 0.998, R² 1.000, MAE 0.0008, bias -0.0002 (n 24); ΔT pairs Spearman 1.000, R² 0.999, MAE 0.0070, bias -0.0024 (n 1584); ΔT token means Spearman 1.000, R² 1.000, MAE 0.0024, bias -0.0024 (n 24) → pass 
  - frozen pattern on this set: pairs Spearman 0.982, R² 0.958, MAE 0.0694, bias -0.0152 (n 1584); token means Spearman 0.986, R² 0.914, MAE 0.0421, bias -0.0152 (n 24); gap to Level 0 0.042; F̂ vs F token means Spearman 0.998, R² 1.000, MAE 0.0022, bias -0.0022 (n 24)
  - split — cue-final: 1056 pairs, 24 tokens: Level 0 ΔT R² 1.000, frozen pattern 0.960, gap 0.040; Π variance share 0.041; row entry R² 0.997; Π R² 0.996; coordinated: 528 pairs, 24 tokens: Level 0 ΔT R² 0.999, frozen pattern 0.944, gap 0.054; Π variance share 0.052; row entry R² 0.996; Π R² 0.991
  - template cardinal: 528 pairs, 24 tokens: Level 0 ΔT R² 0.998, frozen pattern 0.911, gap 0.087; Π variance share 0.092; row entry R² 0.993; Π R² 0.991
  - template coordinated-adjective: 528 pairs, 24 tokens: Level 0 ΔT R² 0.999, frozen pattern 0.944, gap 0.054; Π variance share 0.052; row entry R² 0.996; Π R² 0.991
  - template quantifier: 528 pairs, 24 tokens: Level 0 ΔT R² 1.000, frozen pattern 0.904, gap 0.096; Π variance share 0.094; row entry R² 0.999; Π R² 0.998
  - ladder (rows / ΔT / Π R²): `level0` 0.997 / 0.999 / 0.995; `exact_head` 0.999 / 1.000 / 0.998; `no_D` 0.951 / 0.951 / 0.662; `template_head` 0.817 / 0.935 / 0.354; `level1` (identity) 1.000 / 1.000 / 1.000; `frozen` — / 0.958 / ≡ 0 (`no_D` moves block 2's operating point at p_c only)
  - ablation costs rows: `no_D` 0.046, `template_head` 0.180 (template head ≥ −D holds: True); Π: `no_D` 0.333, `template_head` 0.641 (holds: True); frozen gap 0.042; Π variance share 0.042
  - x₃ remainder (relative) mean 0.0209 max 0.1408; scale remainder |σ(x₃') − σ̂'| mean 0.0010; σ̂'/σ mean 1.026
  - decoded c_L: `c_L_level0F` pairs R² 0.792 / token means R² 0.839; `c_L_level0D` pairs R² 1.000 / token means R² 1.000; `c_L_level1` pairs R² 1.000 / token means R² 1.000; Level 1's residual max 0.000002
  - per-frame row-entry minimum over the exposed frames 0.970
- Y2 (frame-conditional prospective, aggregate with the frame-collapse guard: fresh cues × new frames): scored tokens 24 (precondition ok); head row entry R² 0.998; Π token means Spearman 1.000, R² 1.000, MAE 0.0010, bias 0.0002 (n 24); ΔT pairs Spearman 1.000, R² 1.000, MAE 0.0069, bias -0.0001 (n 288); ΔT token means Spearman 1.000, R² 1.000, MAE 0.0015, bias -0.0001 (n 24); frame guard (≥ 0.9): passed → pass 
  - frozen pattern on this set: pairs Spearman 0.979, R² 0.950, MAE 0.0761, bias -0.0147 (n 288); token means Spearman 0.989, R² 0.874, MAE 0.0485, bias -0.0147 (n 24); gap to Level 0 0.050; F̂ vs F token means Spearman 0.999, R² 1.000, MAE 0.0013, bias -0.0004 (n 24)
  - split — cue-final: 192 pairs, 24 tokens: Level 0 ΔT R² 1.000, frozen pattern 0.950, gap 0.050; Π variance share 0.051; row entry R² 0.999; Π R² 0.998; coordinated: 96 pairs, 24 tokens: Level 0 ΔT R² 0.999, frozen pattern 0.945, gap 0.054; Π variance share 0.042; row entry R² 0.996; Π R² 0.994
  - template cardinal: 96 pairs, 24 tokens: Level 0 ΔT R² 0.995, frozen pattern 0.590, gap 0.405; Π variance share 0.409; row entry R² 0.995; Π R² 0.994
  - template coordinated-adjective: 96 pairs, 24 tokens: Level 0 ΔT R² 0.999, frozen pattern 0.945, gap 0.054; Π variance share 0.042; row entry R² 0.996; Π R² 0.994
  - template quantifier: 96 pairs, 24 tokens: Level 0 ΔT R² 0.999, frozen pattern 0.415, gap 0.584; Π variance share 0.590; row entry R² 1.000; Π R² 0.999
  - ladder (rows / ΔT / Π R²): `level0` 0.998 / 1.000 / 0.997; `exact_head` 1.000 / 1.000 / 1.000; `no_D` 0.959 / 0.975 / 0.722; `template_head` 0.894 / 0.961 / 0.350; `level1` (identity) 1.000 / 1.000 / 1.000; `frozen` — / 0.950 / ≡ 0 (`no_D` moves block 2's operating point at p_c only)
  - ablation costs rows: `no_D` 0.039, `template_head` 0.104 (template head ≥ −D holds: True); Π: `no_D` 0.275, `template_head` 0.647 (holds: True); frozen gap 0.050; Π variance share 0.049
  - x₃ remainder (relative) mean 0.0197 max 0.0882; scale remainder |σ(x₃') − σ̂'| mean 0.0008; σ̂'/σ mean 1.020
  - decoded c_L: `c_L_level0F` pairs R² 0.883 / token means R² 0.950; `c_L_level0D` pairs R² 1.000 / token means R² 1.000; `c_L_level1` pairs R² 1.000 / token means R² 1.000; Level 1's residual max 0.000002
  - frame cardinal-017-1 (cardinal): 24 pairs; row entry R² 0.986 (TV 0.115); ΔT̂ vs ΔT Spearman 0.995, R² 0.993, MAE 0.0098, bias 0.0088 (n 24); Π̂ vs Π Spearman 0.990, R² 0.992, MAE 0.0084, bias 0.0074 (n 24) (Π sd 0.114); frozen ΔT R² 0.320; template-head rows 0.769
  - frame cardinal-017-2 (cardinal): 24 pairs; row entry R² 0.998 (TV 0.033); ΔT̂ vs ΔT Spearman 0.993, R² 0.997, MAE 0.0043, bias 0.0013 (n 24); Π̂ vs Π Spearman 0.995, R² 0.995, MAE 0.0028, bias 0.0002 (n 24) (Π sd 0.053); frozen ΔT R² 0.825; template-head rows 0.903
  - frame cardinal-017-3 (cardinal): 24 pairs; row entry R² 0.994 (TV 0.074); ΔT̂ vs ΔT Spearman 0.992, R² 0.990, MAE 0.0084, bias 0.0078 (n 24); Π̂ vs Π Spearman 0.997, R² 0.989, MAE 0.0039, bias 0.0020 (n 24) (Π sd 0.061); frozen ΔT R² 0.714; template-head rows 0.953
  - frame cardinal-017-4 (cardinal): 24 pairs; row entry R² 0.999 (TV 0.033); ΔT̂ vs ΔT Spearman 0.995, R² 0.997, MAE 0.0058, bias -0.0054 (n 24); Π̂ vs Π Spearman 0.999, R² 0.999, MAE 0.0024, bias 0.0002 (n 24) (Π sd 0.093); frozen ΔT R² 0.450; template-head rows 0.826
  - frame coordinated-adjective-017-1 (coordinated-adjective): 24 pairs; row entry R² 0.999 (TV 0.032); ΔT̂ vs ΔT Spearman 0.990, R² 0.998, MAE 0.0100, bias -0.0021 (n 24); Π̂ vs Π Spearman 0.997, R² 0.996, MAE 0.0033, bias -0.0023 (n 24) (Π sd 0.078); frozen ΔT R² 0.878; template-head rows 0.935
  - frame coordinated-adjective-017-2 (coordinated-adjective): 24 pairs; row entry R² 0.996 (TV 0.052); ΔT̂ vs ΔT Spearman 0.990, R² 0.995, MAE 0.0124, bias -0.0114 (n 24); Π̂ vs Π Spearman 0.995, R² 0.994, MAE 0.0037, bias -0.0022 (n 24) (Π sd 0.074); frozen ΔT R² 0.785; template-head rows 0.808
  - frame coordinated-adjective-017-3 (coordinated-adjective): 24 pairs; row entry R² 0.999 (TV 0.033); ΔT̂ vs ΔT Spearman 0.997, R² 0.998, MAE 0.0087, bias -0.0048 (n 24); Π̂ vs Π Spearman 0.998, R² 0.996, MAE 0.0020, bias 0.0015 (n 24) (Π sd 0.037); frozen ΔT R² 0.976; template-head rows 0.935
  - frame coordinated-adjective-017-4 (coordinated-adjective): 24 pairs; row entry R² 0.986 (TV 0.129); ΔT̂ vs ΔT Spearman 0.998, R² 0.999, MAE 0.0061, bias -0.0016 (n 24); Π̂ vs Π Spearman 0.964, R² 0.948, MAE 0.0051, bias -0.0038 (n 24) (Π sd 0.028); frozen ΔT R² 0.982; template-head rows 0.893
  - frame quantifier-017-1 (quantifier): 24 pairs; row entry R² 1.000 (TV 0.017); ΔT̂ vs ΔT Spearman 0.998, R² 0.999, MAE 0.0039, bias -0.0014 (n 24); Π̂ vs Π Spearman 0.996, R² 0.999, MAE 0.0019, bias 0.0004 (n 24) (Π sd 0.103); frozen ΔT R² 0.730; template-head rows 0.900
  - frame quantifier-017-2 (quantifier): 24 pairs; row entry R² 0.999 (TV 0.019); ΔT̂ vs ΔT Spearman 1.000, R² 1.000, MAE 0.0026, bias -0.0005 (n 24); Π̂ vs Π Spearman 0.997, R² 0.998, MAE 0.0049, bias 0.0029 (n 24) (Π sd 0.158); frozen ΔT R² -0.068; template-head rows 0.961
  - frame quantifier-017-3 (quantifier): 24 pairs; row entry R² 1.000 (TV 0.018); ΔT̂ vs ΔT Spearman 0.987, R² 0.998, MAE 0.0088, bias 0.0081 (n 24); Π̂ vs Π Spearman 0.997, R² 0.996, MAE 0.0045, bias -0.0037 (n 24) (Π sd 0.089); frozen ΔT R² 0.815; template-head rows 0.588
  - frame quantifier-017-4 (quantifier): 24 pairs; row entry R² 1.000 (TV 0.013); ΔT̂ vs ΔT Spearman 0.994, R² 1.000, MAE 0.0024, bias -0.0005 (n 24); Π̂ vs Π Spearman 0.999, R² 1.000, MAE 0.0021, bias 0.0002 (n 24) (Π sd 0.167); frozen ΔT R² -0.463; template-head rows 0.983
- Y3 (frozen pattern on the cue-final pairs of both sets, 1248 pairs, 24 tokens): ΔT R² 0.958 (rejected iff < 0.95) against Level 0 1.000, gap 0.042 (rejected iff ≥ 0.05) → NOT REJECTED; Π variance share 0.042
- Coordinated split (descriptive; predeclared: not rejected there): 624 pairs, 24 tokens: Level 0 ΔT R² 0.999, frozen pattern 0.945, gap 0.053; Π variance share 0.049; row entry R² 0.996; Π R² 0.992; would the criterion reject: True
- Template cardinal over both sets (descriptive): 624 pairs, 24 tokens: Level 0 ΔT R² 0.998, frozen pattern 0.907, gap 0.091; Π variance share 0.095; row entry R² 0.993; Π R² 0.991
- Template coordinated-adjective over both sets (descriptive): 624 pairs, 24 tokens: Level 0 ΔT R² 0.999, frozen pattern 0.945, gap 0.053; Π variance share 0.049; row entry R² 0.996; Π R² 0.992
- Template quantifier over both sets (descriptive): 624 pairs, 24 tokens: Level 0 ΔT R² 1.000, frozen pattern 0.883, gap 0.117; Π variance share 0.115; row entry R² 0.999; Π R² 0.998
- Ladder over both sets:
  - ladder (rows / ΔT / Π R²): `level0` 0.997 / 0.999 / 0.996; `exact_head` 0.999 / 1.000 / 0.999; `no_D` 0.952 / 0.955 / 0.674; `template_head` 0.830 / 0.939 / 0.354; `level1` (identity) 1.000 / 1.000 / 1.000; `frozen` — / 0.956 / ≡ 0 (`no_D` moves block 2's operating point at p_c only)
  - ablation costs rows: `no_D` 0.045, `template_head` 0.166 (template head ≥ −D holds: True); Π: `no_D` 0.322, `template_head` 0.642 (holds: True); frozen gap 0.043; Π variance share 0.043
  - x₃ remainder (relative) mean 0.0207 max 0.1408; scale remainder |σ(x₃') − σ̂'| mean 0.0010; σ̂'/σ mean 1.025
  - decoded c_L: `c_L_level0F` pairs R² 0.808; `c_L_level0D` pairs R² 1.000; `c_L_level1` pairs R² 1.000; Level 1's residual max 0.000002

| token | class | Y1 frames | Y1 ΔT̂ | Y1 ΔT | Y1 Π̂ | Y1 Π | Y1 frozen ΔT̂ | Y2 frames | Y2 ΔT̂ | Y2 ΔT | Y2 Π̂ | Y2 Π | self ΔÂ/ΔA (Y1) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trio | ordinal-or-numeral | 66 | 1.4902 | 1.4946 | -0.1088 | -0.1118 | 1.5990 | 12 | 1.4267 | 1.4337 | -0.1019 | -0.1033 | -0.140 / -0.143 |
| costly | adjective | 66 | 1.4249 | 1.4270 | -0.0179 | -0.0190 | 1.4427 | 12 | 1.3532 | 1.3565 | -0.0439 | -0.0431 | -0.072 / -0.075 |
| usual | determiner-like | 66 | 1.3027 | 1.3057 | -0.0734 | -0.0738 | 1.3761 | 12 | 1.2185 | 1.2197 | -0.1001 | -0.0996 | -0.205 / -0.206 |
| them | possessive-or-pronoun | 66 | 1.2215 | 1.2243 | -0.0257 | -0.0251 | 1.2471 | 12 | 1.2118 | 1.2093 | -0.0254 | -0.0276 | 0.099 / 0.096 |
| finite | quantity | 66 | 1.2072 | 1.2095 | -0.0003 | -0.0003 | 1.2075 | 12 | 1.1749 | 1.1751 | -0.0167 | -0.0182 | -0.171 / -0.173 |
| blunt | adjective | 66 | 1.1849 | 1.1867 | 0.0359 | 0.0356 | 1.1490 | 12 | 1.1533 | 1.1527 | 0.0343 | 0.0337 | -0.277 / -0.278 |
| us | possessive-or-pronoun | 66 | 1.1845 | 1.1857 | -0.0271 | -0.0264 | 1.2116 | 12 | 1.1428 | 1.1447 | -0.0342 | -0.0328 | 0.004 / 0.003 |
| partial | quantity | 66 | 1.1776 | 1.1803 | 0.0303 | 0.0300 | 1.1473 | 12 | 1.1416 | 1.1411 | 0.0226 | 0.0214 | -0.210 / -0.214 |
| paper | adjective | 66 | 1.1689 | 1.1696 | 0.0191 | 0.0187 | 1.1498 | 12 | 1.1371 | 1.1392 | 0.0317 | 0.0318 | -0.236 / -0.235 |
| leading | determiner-like | 66 | 1.1569 | 1.1591 | 0.0516 | 0.0515 | 1.1053 | 12 | 1.1303 | 1.1305 | 0.0389 | 0.0386 | -0.218 / -0.217 |
| violet | adjective | 66 | 1.1377 | 1.1415 | 0.0016 | 0.0009 | 1.1361 | 12 | 1.1035 | 1.1036 | 0.0127 | 0.0116 | -0.175 / -0.179 |
| nearby | determiner-like | 66 | 1.1289 | 1.1330 | -0.0067 | -0.0060 | 1.1356 | 12 | 1.1183 | 1.1177 | 0.0050 | 0.0044 | -0.086 / -0.088 |
| further | determiner-like | 66 | 1.1141 | 1.1178 | 0.0089 | 0.0095 | 1.1052 | 12 | 1.0931 | 1.0953 | 0.0026 | 0.0036 | -0.143 / -0.143 |
| metal | adjective | 66 | 1.1105 | 1.1126 | 0.0583 | 0.0581 | 1.0523 | 12 | 1.1103 | 1.1103 | 0.0714 | 0.0715 | -0.234 / -0.235 |
| score | ordinal-or-numeral | 66 | 1.1035 | 1.1058 | 0.0660 | 0.0675 | 1.0375 | 12 | 1.0807 | 1.0816 | 0.0741 | 0.0750 | -0.160 / -0.160 |
| leather | adjective | 66 | 1.0823 | 1.0850 | 0.0896 | 0.0909 | 0.9927 | 12 | 1.0658 | 1.0660 | 0.0940 | 0.0954 | -0.244 / -0.246 |
| complete | quantity | 66 | 1.0433 | 1.0444 | 0.0656 | 0.0654 | 0.9777 | 12 | 1.0458 | 1.0457 | 0.0747 | 0.0752 | -0.141 / -0.141 |
| adjacent | determiner-like | 66 | 1.0161 | 1.0191 | 0.0035 | 0.0042 | 1.0126 | 12 | 1.0006 | 1.0007 | 0.0346 | 0.0345 | -0.104 / -0.106 |
| none | possessive-or-pronoun | 66 | 0.9161 | 0.9191 | 0.0006 | 0.0017 | 0.9156 | 12 | 0.9060 | 0.9030 | 0.0000 | -0.0013 | 0.012 / 0.012 |
| twice | ordinal-or-numeral | 66 | 0.8956 | 0.8970 | 0.0661 | 0.0663 | 0.8295 | 12 | 0.8828 | 0.8821 | 0.0621 | 0.0628 | -0.176 / -0.177 |
| whole | quantity | 66 | 0.8776 | 0.8784 | 0.1033 | 0.1036 | 0.7743 | 12 | 0.8697 | 0.8696 | 0.1300 | 0.1310 | -0.129 / -0.129 |
| you | possessive-or-pronoun | 66 | 0.8203 | 0.8241 | -0.0849 | -0.0823 | 0.9051 | 12 | 0.8326 | 0.8316 | -0.0905 | -0.0924 | 0.150 / 0.147 |
| entire | quantity | 66 | 0.7876 | 0.7897 | 0.0408 | 0.0420 | 0.7468 | 12 | 0.7830 | 0.7822 | 0.0539 | 0.0550 | -0.046 / -0.046 |
| once | ordinal-or-numeral | 66 | 0.7406 | 0.7403 | 0.0101 | 0.0103 | 0.7305 | 12 | 0.7158 | 0.7097 | 0.0205 | 0.0175 | -0.020 / -0.020 |

## Execution ledger

- Executed prompt keys: 2106
- Executed noun keys: 80
