# Experiment 016 Report

- Run ID: `c1cc27bf24171b52`
- Confirmation sha256: `e5a3a4eb392e459117ebd7dfd132fffb66f02bbd0cd75b479f5ba1afd83e3855`
- Experiment 015 lock sha256: `6987704901dccf8c3638c7f770f64d77044320af5a1726e5c517130bc7423ebb`
- Protocol/code commit at explore: `6343f236ce907efe5e7711e87a425aa669a0ce83`

## Phases

- `confirm`: `complete`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

## Tier A — exposed pool (calibration record only; the floors are frozen constants)

- Replication of Experiment 015: 6180 pairs, max deviation 0.00e+00
- Identities (checks only): I1_patched_rows 1.7e-05, I1_reference_rows 9.8e-06, I2_chain 3.4e-06, I3_head_split 9.9e-07, level0_015_recovery 1.0e-14, level1_recovery 1.4e-14, neuron_sum 2.3e-08, p1_cross_check 6.6e-16, rho_identity 1.1e-07; read weight vs Experiment 011 lock 0.0e+00
- Program: layer 1: d_head 64, rotary_dim 16, base 10000, layer 2: d_head 64, rotary_dim 16, base 10000
- Token means (183): ĉ_ΔA vs c_ΔA — Spearman 0.998, R² 0.994, MAE 0.0013, bias -0.0012 (n 183); c_ΔA spread sd 0.0208; per layer: Spearman 0.999, R² 0.998, MAE 0.0006, bias -0.0005 (n 183) | Spearman 0.999, R² 0.996, MAE 0.0008, bias -0.0007 (n 183)
- Level 0-F entries (6180 pairs): layer 1 entry R² 0.996 (TV ratio 0.053), layer 2 entry R² 0.988 (TV ratio 0.079), both 0.993; per-frame layer-2 minimum 0.937; self weights pooled Spearman 0.995, R² 0.993, MAE 0.0082, bias 0.0023 (n 98880); pairs ĉ_ΔA vs c_ΔA Spearman 0.995, R² 0.991, MAE 0.0024, bias -0.0011 (n 6180)
- Scale-only: token means Spearman 0.976, R² 0.920, MAE 0.0045, bias -0.0036 (n 183); entries layer 1 entry R² 0.916 (TV ratio 0.245), layer 2 entry R² 0.708 (TV ratio 0.459), both 0.847; margin of Level 0-F over it at layer 2 0.280
- Experiment 015 Level 0 on the same pairs: token means Spearman 0.968, R² 0.804, MAE 0.0076, bias -0.0072 (n 183); layer-2 entry R² 0.710
- Diagonal-proportional (Level 0-F self weight): entries layer 1 entry R² 0.754 (TV ratio 0.452), layer 2 entry R² 0.788 (TV ratio 0.467), both 0.765; margin layer 1 0.242, layer 2 0.200 → interaction beyond the self logit: True
  - ablation ladder (layer-2 entry R²): `ablate_operands` 0.752 (cost 0.236); `ablate_scale` 0.960 (cost 0.027); `ablate_operating_point` 0.889 (cost 0.099); `level0_015` 0.710; `level1` 1.000; predeclared ordering operands ≥ operating point ≥ scale holds: True
  - each channel alone over Level 0 (layer-2 entry R²): `only_operands` 0.859; `only_scale` 0.708; `only_operating_point` 0.762; oracle diagonal-proportional 0.797; ‖Δ̂x‖/‖x−μ‖ 0.96 / 0.82
  - renormalization remainder (Level 1 − Level 0-F, layer-2 entry R²) 0.012; scale remainder |σ(x') − σ̂'| mean 0.0000 / 0.0005; σ̂'/σ mean 1.108 / 1.079
  - decoded c_L: `c_012` pairs R² 0.741 / token means R² 0.815; `c_013_own` pairs R² 0.949 / token means R² 0.968; `c_015_level0` pairs R² 0.807 / token means R² 0.932; `c_L_level0F` pairs R² 0.852 / token means R² 0.940; `c_L_level1` pairs R² 1.000 / token means R² 1.000; Level 1's residual max 0.000003

## Lock

- Candidate lock sha256 `c21a69fa9347cc3ebcb4d113a719d4a34fb52668e513600dcfbe627c50884c77`; predictions sha256 `85789c0f083f8d042f971447d06da5277bd7992b64477e841d4be31728196986`

## Confirmation — stage 1 (fresh frames' reference states; prediction table digested before any fresh cue prompt)

- Table rows 288; digest `cd1d1de7dc79162bb3cac78e556abe2fdb43d82813c9e5f1399e3564e2a2c368`; commit `dc4e2a77dd62be9e619671e30fb7408ebb88b734`
  - cardinal-016-1: valid (plural head change 1.477, cue effect 79/72; p_c 3)
  - cardinal-016-2: valid (plural head change 0.780, cue effect 79/72; p_c 4)
  - cardinal-016-3: valid (plural head change 1.731, cue effect 79/72; p_c 3)
  - cardinal-016-4: valid (plural head change 1.693, cue effect 79/72; p_c 3)
  - coordinated-adjective-016-1: valid (plural head change 1.905, cue effect 79/72; p_c 6)
  - coordinated-adjective-016-2: valid (plural head change 2.376, cue effect 79/72; p_c 5)
  - coordinated-adjective-016-3: valid (plural head change 2.064, cue effect 79/72; p_c 6)
  - coordinated-adjective-016-4: valid (plural head change 2.361, cue effect 79/72; p_c 5)
  - quantifier-016-1: valid (plural head change 2.882, cue effect 79/72; p_c 3)
  - quantifier-016-2: valid (plural head change 2.492, cue effect 79/72; p_c 3)
  - quantifier-016-3: valid (plural head change 2.056, cue effect 79/72; p_c 3)
  - quantifier-016-4: valid (plural head change 2.150, cue effect 79/72; p_c 3)

## Confirmation — stage 2 — `FRAME_CHANNELS_PREDICTED_TOKENS | FRAME_CHANNELS_PREDICTED_FRAMES_CONDITIONAL | SCALE_ONLY_REJECTED`

- Y1 (strict prospective: fresh cues × exposed frames): scored tokens 24 (precondition ok); token means ĉ_ΔA vs c_ΔA — Spearman 0.998, R² 0.994, MAE 0.0015, bias -0.0015 (n 24); entry R² layer 1 0.994, layer 2 0.988 → pass 
  - Level 0-F entries: layer 1 entry R² 0.994 (TV ratio 0.066), layer 2 entry R² 0.988 (TV ratio 0.081), both 0.992; self weights pooled Spearman 0.994, R² 0.992, MAE 0.0097, bias 0.0016 (n 20736); pairs ĉ_ΔA vs c_ΔA Spearman 0.994, R² 0.988, MAE 0.0029, bias -0.0015 (n 1296); per layer token means Spearman 0.995, R² 0.999, MAE 0.0006, bias -0.0006 (n 24) | Spearman 0.997, R² 0.993, MAE 0.0009, bias -0.0009 (n 24); c_ΔA spread sd 0.0210
  - scale-only on this set: token means Spearman 0.976, R² 0.944, MAE 0.0042, bias -0.0040 (n 24); entries layer 1 entry R² 0.899 (TV ratio 0.273), layer 2 entry R² 0.678 (TV ratio 0.476), both 0.823; margin at layer 2 0.310
  - Experiment 015 Level 0 on this set: token means Spearman 0.957, R² 0.816, MAE 0.0079, bias -0.0079 (n 24); layer-2 entry R² 0.679
  - diagonal-proportional (Level 0-F self weight): token means Spearman 0.789, R² 0.552, MAE 0.0113, bias 0.0111 (n 24); entries layer 1 entry R² 0.725 (TV ratio 0.486), layer 2 entry R² 0.780 (TV ratio 0.474), both 0.744; margin layer 1 0.269, layer 2 0.208
  - ablation ladder (layer-2 entry R²): `ablate_operands` 0.707 (cost 0.281); `ablate_scale` 0.946 (cost 0.042); `ablate_operating_point` 0.886 (cost 0.103); `level0_015` 0.679; `level1` 1.000; predeclared ordering operands ≥ operating point ≥ scale holds: True
  - each channel alone over Level 0 (layer-2 entry R²): `only_operands` 0.841; `only_scale` 0.678; `only_operating_point` 0.721; oracle diagonal-proportional 0.789; ‖Δ̂x‖/‖x−μ‖ 1.02 / 0.86
  - renormalization remainder (Level 1 − Level 0-F, layer-2 entry R²) 0.012; scale remainder |σ(x') − σ̂'| mean 0.0000 / 0.0007; σ̂'/σ mean 1.146 / 1.089
  - decoded c_L: `c_012` pairs R² 0.567 / token means R² 0.590; `c_013_own` pairs R² 0.944 / token means R² 0.955; `c_015_level0` pairs R² 0.705 / token means R² 0.893; `c_L_level0F` pairs R² 0.790 / token means R² 0.895; `c_L_level1` pairs R² 1.000 / token means R² 1.000; Level 1's residual max 0.000002
  - per-frame layer-2 minimum over the exposed frames 0.951
- Y2 (frame-conditional prospective, aggregate with the frame-collapse guard: fresh cues × new frames): scored tokens 24 (precondition ok); token means ĉ_ΔA vs c_ΔA — Spearman 1.000, R² 0.996, MAE 0.0012, bias -0.0012 (n 24); entry R² layer 1 0.995, layer 2 0.986; frame guard (≥ 0.8): passed → pass 
  - Level 0-F entries: layer 1 entry R² 0.995 (TV ratio 0.060), layer 2 entry R² 0.986 (TV ratio 0.087), both 0.992; self weights pooled Spearman 0.995, R² 0.992, MAE 0.0091, bias 0.0006 (n 4608); pairs ĉ_ΔA vs c_ΔA Spearman 0.997, R² 0.993, MAE 0.0026, bias -0.0012 (n 288); per layer token means Spearman 0.998, R² 0.999, MAE 0.0006, bias -0.0005 (n 24) | Spearman 0.997, R² 0.996, MAE 0.0007, bias -0.0006 (n 24); c_ΔA spread sd 0.0218
  - scale-only on this set: token means Spearman 0.895, R² 0.917, MAE 0.0049, bias -0.0012 (n 24); entries layer 1 entry R² 0.892 (TV ratio 0.277), layer 2 entry R² 0.642 (TV ratio 0.485), both 0.808; margin at layer 2 0.344
  - Experiment 015 Level 0 on this set: token means Spearman 0.896, R² 0.864, MAE 0.0065, bias -0.0039 (n 24); layer-2 entry R² 0.668
  - diagonal-proportional (Level 0-F self weight): token means Spearman 0.857, R² 0.418, MAE 0.0138, bias 0.0138 (n 24); entries layer 1 entry R² 0.720 (TV ratio 0.485), layer 2 entry R² 0.745 (TV ratio 0.501), both 0.728; margin layer 1 0.275, layer 2 0.241
  - ablation ladder (layer-2 entry R²): `ablate_operands` 0.721 (cost 0.265); `ablate_scale` 0.945 (cost 0.041); `ablate_operating_point` 0.900 (cost 0.085); `level0_015` 0.668; `level1` 1.000; predeclared ordering operands ≥ operating point ≥ scale holds: True
  - each channel alone over Level 0 (layer-2 entry R²): `only_operands` 0.842; `only_scale` 0.642; `only_operating_point` 0.737; oracle diagonal-proportional 0.758; ‖Δ̂x‖/‖x−μ‖ 1.03 / 0.87
  - renormalization remainder (Level 1 − Level 0-F, layer-2 entry R²) 0.014; scale remainder |σ(x') − σ̂'| mean 0.0000 / 0.0007; σ̂'/σ mean 1.149 / 1.089
  - decoded c_L: `c_012` pairs R² 0.393 / token means R² 0.513; `c_013_own` pairs R² 0.937 / token means R² 0.942; `c_015_level0` pairs R² 0.648 / token means R² 0.753; `c_L_level0F` pairs R² 0.761 / token means R² 0.794; `c_L_level1` pairs R² 1.000 / token means R² 1.000; Level 1's residual max 0.000002
  - frame cardinal-016-1: 24 pairs; Level 0-F entry R² layer 1 0.997 (TV 0.053), layer 2 0.986 (TV 0.094); ĉ_ΔA vs c_ΔA Spearman 0.997, R² 0.994, MAE 0.0014, bias -0.0001 (n 24); scale-only layer 2 0.684; 015 Level 0 layer 2 0.764
  - frame cardinal-016-2: 24 pairs; Level 0-F entry R² layer 1 0.987 (TV 0.120), layer 2 0.949 (TV 0.166); ĉ_ΔA vs c_ΔA Spearman 0.983, R² 0.947, MAE 0.0055, bias -0.0055 (n 24); scale-only layer 2 0.560; 015 Level 0 layer 2 0.512
  - frame cardinal-016-3: 24 pairs; Level 0-F entry R² layer 1 0.997 (TV 0.047), layer 2 0.993 (TV 0.070); ĉ_ΔA vs c_ΔA Spearman 0.997, R² 0.995, MAE 0.0016, bias 0.0015 (n 24); scale-only layer 2 0.789; 015 Level 0 layer 2 0.814
  - frame cardinal-016-4: 24 pairs; Level 0-F entry R² layer 1 0.994 (TV 0.079), layer 2 0.933 (TV 0.197); ĉ_ΔA vs c_ΔA Spearman 0.990, R² 0.985, MAE 0.0024, bias 0.0002 (n 24); scale-only layer 2 0.413; 015 Level 0 layer 2 0.507
  - frame coordinated-adjective-016-1: 24 pairs; Level 0-F entry R² layer 1 0.990 (TV 0.091), layer 2 0.984 (TV 0.109); ĉ_ΔA vs c_ΔA Spearman 0.980, R² 0.980, MAE 0.0047, bias -0.0047 (n 24); scale-only layer 2 0.512; 015 Level 0 layer 2 0.549
  - frame coordinated-adjective-016-2: 24 pairs; Level 0-F entry R² layer 1 0.992 (TV 0.074), layer 2 0.994 (TV 0.066); ĉ_ΔA vs c_ΔA Spearman 0.990, R² 0.960, MAE 0.0040, bias -0.0040 (n 24); scale-only layer 2 0.770; 015 Level 0 layer 2 0.783
  - frame coordinated-adjective-016-3: 24 pairs; Level 0-F entry R² layer 1 0.986 (TV 0.105), layer 2 0.980 (TV 0.136); ĉ_ΔA vs c_ΔA Spearman 0.988, R² 0.991, MAE 0.0027, bias 0.0007 (n 24); scale-only layer 2 0.725; 015 Level 0 layer 2 0.730
  - frame coordinated-adjective-016-4: 24 pairs; Level 0-F entry R² layer 1 0.991 (TV 0.079), layer 2 0.996 (TV 0.055); ĉ_ΔA vs c_ΔA Spearman 0.997, R² 0.996, MAE 0.0014, bias 0.0010 (n 24); scale-only layer 2 0.716; 015 Level 0 layer 2 0.747
  - frame quantifier-016-1: 24 pairs; Level 0-F entry R² layer 1 0.998 (TV 0.032), layer 2 0.996 (TV 0.045); ĉ_ΔA vs c_ΔA Spearman 0.995, R² 0.990, MAE 0.0021, bias -0.0011 (n 24); scale-only layer 2 0.507; 015 Level 0 layer 2 0.614
  - frame quantifier-016-2: 24 pairs; Level 0-F entry R² layer 1 0.999 (TV 0.032), layer 2 0.999 (TV 0.025); ĉ_ΔA vs c_ΔA Spearman 0.997, R² 0.998, MAE 0.0013, bias -0.0010 (n 24); scale-only layer 2 0.596; 015 Level 0 layer 2 0.641
  - frame quantifier-016-3: 24 pairs; Level 0-F entry R² layer 1 0.999 (TV 0.025), layer 2 0.998 (TV 0.036); ĉ_ΔA vs c_ΔA Spearman 0.993, R² 0.994, MAE 0.0027, bias -0.0022 (n 24); scale-only layer 2 0.589; 015 Level 0 layer 2 0.701
  - frame quantifier-016-4: 24 pairs; Level 0-F entry R² layer 1 1.000 (TV 0.015), layer 2 0.993 (TV 0.067); ĉ_ΔA vs c_ΔA Spearman 0.989, R² 0.997, MAE 0.0015, bias 0.0010 (n 24); scale-only layer 2 0.839; 015 Level 0 layer 2 0.568
- Y3 (scale-only alternative over both sets, 1584 pairs): layer-2 entry R² 0.672 (rejected iff < 0.85) against Level 0-F 0.988, margin 0.316 (rejected iff ≥ 0.15) → REJECTED; token means Spearman 0.930, R² 0.932, MAE 0.0045, bias -0.0026 (n 48)
- Comparator rule over both sets (1584 pairs): Level 0-F layer 1 entry R² 0.994 (TV ratio 0.065), layer 2 entry R² 0.988 (TV ratio 0.082), both 0.992; diagonal-proportional layer 1 entry R² 0.724 (TV ratio 0.486), layer 2 entry R² 0.774 (TV ratio 0.479), both 0.741; margins layer 1 0.270, layer 2 0.214 (required ≥ 0.1) → Level 0-F decodes the query/key interaction beyond the self-logit effect
- Ladder over both sets:
  - ablation ladder (layer-2 entry R²): `ablate_operands` 0.709 (cost 0.278); `ablate_scale` 0.946 (cost 0.042); `ablate_operating_point` 0.888 (cost 0.100); `level0_015` 0.677; `level1` 1.000; predeclared ordering operands ≥ operating point ≥ scale holds: True
  - each channel alone over Level 0 (layer-2 entry R²): `only_operands` 0.841; `only_scale` 0.672; `only_operating_point` 0.724; oracle diagonal-proportional 0.783; ‖Δ̂x‖/‖x−μ‖ 1.02 / 0.86
  - renormalization remainder (Level 1 − Level 0-F, layer-2 entry R²) 0.012; scale remainder |σ(x') − σ̂'| mean 0.0000 / 0.0007; σ̂'/σ mean 1.147 / 1.089
  - decoded c_L: `c_012` pairs R² 0.536; `c_013_own` pairs R² 0.943; `c_015_level0` pairs R² 0.694; `c_L_level0F` pairs R² 0.785; `c_L_level1` pairs R² 1.000; Level 1's residual max 0.000002

| token | class | Y1 frames | Y1 ĉ_ΔA | Y1 c_ΔA | Y2 frames | Y2 ĉ_ΔA | Y2 c_ΔA | scale-only ĉ (Y1) | 015 L0 ĉ (Y1) | σ̂'/σ L2 (Y1) |
|---|---|---|---|---|---|---|---|---|---|---|
| yourselves | possessive-or-pronoun | 54 | 0.0318 | 0.0327 | 12 | 0.0104 | 0.0114 | 0.0279 | 0.0293 | 1.025 |
| twin | ordinal-or-numeral | 54 | 0.0109 | 0.0138 | 12 | -0.0013 | 0.0014 | 0.0102 | 0.0075 | 1.101 |
| orange | adjective | 54 | 0.0102 | 0.0121 | 12 | 0.0065 | 0.0086 | 0.0044 | 0.0000 | 1.099 |
| respective | determiner-like | 54 | 0.0080 | 0.0093 | 12 | 0.0036 | 0.0052 | 0.0071 | 0.0056 | 1.071 |
| heavy | adjective | 54 | 0.0058 | 0.0071 | 12 | 0.0059 | 0.0079 | 0.0033 | -0.0015 | 1.081 |
| dual | ordinal-or-numeral | 54 | 0.0051 | 0.0078 | 12 | -0.0050 | -0.0019 | 0.0036 | 0.0022 | 1.169 |
| myself | possessive-or-pronoun | 54 | 0.0001 | 0.0009 | 12 | -0.0191 | -0.0187 | 0.0015 | -0.0016 | 1.047 |
| brown | adjective | 54 | -0.0094 | -0.0071 | 12 | -0.0080 | -0.0059 | -0.0128 | -0.0189 | 1.102 |
| whoever | possessive-or-pronoun | 54 | -0.0099 | -0.0087 | 12 | -0.0286 | -0.0279 | -0.0121 | -0.0174 | 1.010 |
| upper | determiner-like | 54 | -0.0106 | -0.0097 | 12 | -0.0150 | -0.0145 | -0.0085 | -0.0099 | 1.060 |
| unlimited | quantity | 54 | -0.0126 | -0.0102 | 12 | -0.0116 | -0.0107 | -0.0216 | -0.0248 | 1.146 |
| extensive | quantity | 54 | -0.0145 | -0.0134 | 12 | -0.0168 | -0.0159 | -0.0167 | -0.0209 | 1.154 |
| specific | determiner-like | 54 | -0.0154 | -0.0142 | 12 | -0.0232 | -0.0223 | -0.0146 | -0.0168 | 1.069 |
| total | quantity | 54 | -0.0169 | -0.0163 | 12 | -0.0131 | -0.0129 | -0.0242 | -0.0276 | 1.035 |
| individual | determiner-like | 54 | -0.0180 | -0.0171 | 12 | -0.0221 | -0.0213 | -0.0185 | -0.0201 | 1.066 |
| grey | adjective | 54 | -0.0204 | -0.0190 | 12 | -0.0241 | -0.0226 | -0.0245 | -0.0307 | 1.145 |
| bright | adjective | 54 | -0.0220 | -0.0202 | 12 | -0.0278 | -0.0266 | -0.0223 | -0.0283 | 1.106 |
| oneself | possessive-or-pronoun | 54 | -0.0230 | -0.0224 | 12 | -0.0316 | -0.0316 | -0.0232 | -0.0266 | 1.031 |
| countable | quantity | 54 | -0.0243 | -0.0213 | 12 | -0.0179 | -0.0162 | -0.0250 | -0.0306 | 1.156 |
| insufficient | quantity | 54 | -0.0265 | -0.0251 | 12 | -0.0313 | -0.0299 | -0.0283 | -0.0302 | 1.075 |
| twentieth | ordinal-or-numeral | 54 | -0.0316 | -0.0311 | 12 | -0.0432 | -0.0428 | -0.0354 | -0.0411 | 1.110 |
| initial | determiner-like | 54 | -0.0325 | -0.0313 | 12 | -0.0444 | -0.0432 | -0.0338 | -0.0377 | 1.062 |
| pink | adjective | 54 | -0.0353 | -0.0332 | 12 | -0.0382 | -0.0367 | -0.0417 | -0.0480 | 1.181 |
| quarter | ordinal-or-numeral | 54 | -0.0774 | -0.0770 | 12 | -0.0928 | -0.0931 | -0.0844 | -0.0947 | 1.029 |

Measured / predicted self-attention weight change ΔA_h(p_c, p_c), token means over the Y1 frames:

| token | L01.H00 | L01.H01 | L01.H02 | L01.H03 | L01.H04 | L01.H05 | L01.H06 | L01.H07 | L02.H00 | L02.H01 | L02.H02 | L02.H03 | L02.H04 | L02.H05 | L02.H06 | L02.H07 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yourselves | -0.00 / -0.00 | -0.12 / -0.12 | -0.21 / -0.21 | 0.09 / 0.09 | 0.04 / 0.04 | 0.01 / 0.00 | 0.07 / 0.07 | -0.16 / -0.16 | 0.11 / 0.11 | -0.01 / -0.01 | 0.03 / 0.03 | 0.42 / 0.42 | 0.05 / 0.05 | 0.16 / 0.16 | -0.01 / -0.00 | 0.02 / 0.02 |
| twin | -0.10 / -0.10 | 0.27 / 0.26 | -0.26 / -0.26 | 0.01 / 0.00 | -0.01 / -0.01 | -0.04 / -0.04 | 0.04 / 0.05 | -0.04 / -0.04 | 0.08 / 0.08 | 0.01 / 0.01 | -0.02 / -0.02 | 0.04 / 0.04 | 0.04 / 0.04 | -0.32 / -0.31 | -0.08 / -0.06 | -0.05 / -0.05 |
| orange | -0.08 / -0.08 | 0.33 / 0.33 | -0.09 / -0.09 | 0.01 / 0.01 | 0.13 / 0.14 | -0.13 / -0.13 | 0.26 / 0.26 | 0.11 / 0.11 | 0.07 / 0.06 | 0.00 / 0.00 | -0.08 / -0.08 | 0.16 / 0.16 | 0.03 / 0.03 | -0.12 / -0.11 | -0.28 / -0.27 | -0.03 / -0.03 |
| respective | 0.18 / 0.18 | -0.01 / -0.01 | 0.02 / 0.01 | 0.08 / 0.08 | 0.10 / 0.11 | -0.04 / -0.04 | 0.08 / 0.08 | -0.28 / -0.28 | 0.09 / 0.09 | 0.04 / 0.04 | -0.03 / -0.03 | 0.11 / 0.11 | -0.01 / -0.01 | 0.05 / 0.05 | -0.16 / -0.14 | -0.01 / -0.01 |
| heavy | -0.15 / -0.15 | 0.07 / 0.07 | -0.26 / -0.26 | -0.00 / -0.00 | 0.09 / 0.09 | -0.16 / -0.16 | -0.10 / -0.10 | -0.11 / -0.10 | 0.06 / 0.06 | 0.01 / 0.01 | -0.17 / -0.16 | -0.00 / -0.00 | -0.05 / -0.05 | -0.22 / -0.21 | -0.18 / -0.16 | -0.03 / -0.03 |
| dual | -0.05 / -0.06 | 0.08 / 0.07 | -0.03 / -0.03 | 0.02 / 0.02 | 0.14 / 0.15 | -0.14 / -0.15 | -0.08 / -0.08 | 0.12 / 0.12 | 0.15 / 0.15 | 0.06 / 0.06 | 0.03 / 0.04 | -0.05 / -0.04 | 0.03 / 0.03 | -0.13 / -0.12 | -0.21 / -0.18 | -0.03 / -0.03 |
| myself | -0.09 / -0.09 | -0.11 / -0.11 | -0.27 / -0.27 | 0.07 / 0.07 | -0.04 / -0.04 | -0.04 / -0.04 | 0.09 / 0.09 | -0.20 / -0.20 | 0.10 / 0.10 | -0.02 / -0.02 | 0.00 / 0.00 | 0.29 / 0.29 | 0.06 / 0.06 | 0.11 / 0.11 | 0.09 / 0.10 | 0.02 / 0.02 |
| brown | -0.02 / -0.02 | 0.26 / 0.26 | -0.22 / -0.22 | -0.03 / -0.04 | -0.03 / -0.02 | -0.10 / -0.10 | 0.03 / 0.03 | 0.20 / 0.20 | 0.02 / 0.02 | 0.02 / 0.02 | -0.13 / -0.12 | 0.03 / 0.03 | 0.14 / 0.14 | -0.19 / -0.18 | -0.05 / -0.03 | -0.03 / -0.03 |
| whoever | -0.01 / -0.01 | 0.19 / 0.19 | -0.27 / -0.27 | -0.02 / -0.02 | 0.17 / 0.17 | -0.04 / -0.04 | -0.12 / -0.12 | 0.10 / 0.11 | 0.17 / 0.17 | 0.03 / 0.03 | -0.04 / -0.04 | 0.45 / 0.44 | 0.19 / 0.19 | 0.03 / 0.02 | -0.36 / -0.36 | 0.02 / 0.02 |
| upper | -0.01 / -0.01 | 0.24 / 0.24 | 0.15 / 0.15 | -0.01 / -0.01 | 0.01 / 0.01 | -0.05 / -0.05 | -0.04 / -0.04 | 0.22 / 0.22 | 0.07 / 0.06 | -0.03 / -0.03 | -0.11 / -0.11 | 0.03 / 0.03 | 0.13 / 0.13 | -0.02 / -0.02 | 0.23 / 0.24 | -0.04 / -0.04 |
| unlimited | -0.09 / -0.09 | 0.31 / 0.30 | 0.03 / 0.03 | 0.02 / 0.02 | 0.24 / 0.25 | 0.08 / 0.08 | 0.02 / 0.03 | 0.18 / 0.18 | 0.09 / 0.09 | 0.05 / 0.04 | -0.06 / -0.06 | 0.01 / 0.01 | 0.10 / 0.10 | 0.01 / 0.03 | -0.12 / -0.09 | -0.00 / -0.00 |
| extensive | -0.13 / -0.13 | 0.13 / 0.13 | -0.20 / -0.20 | 0.04 / 0.04 | 0.05 / 0.05 | -0.09 / -0.10 | 0.05 / 0.05 | -0.17 / -0.16 | 0.07 / 0.06 | 0.01 / 0.01 | -0.11 / -0.11 | -0.08 / -0.08 | -0.01 / -0.01 | -0.05 / -0.03 | 0.13 / 0.16 | -0.02 / -0.02 |
| specific | -0.13 / -0.13 | -0.05 / -0.05 | -0.23 / -0.23 | 0.02 / 0.02 | 0.03 / 0.03 | -0.09 / -0.09 | 0.21 / 0.21 | -0.09 / -0.09 | 0.04 / 0.04 | 0.02 / 0.01 | -0.05 / -0.04 | -0.05 / -0.06 | 0.01 / 0.01 | 0.03 / 0.03 | -0.04 / -0.02 | -0.02 / -0.02 |
| total | 0.06 / 0.06 | 0.32 / 0.32 | 0.14 / 0.14 | 0.04 / 0.04 | 0.09 / 0.09 | -0.21 / -0.21 | 0.01 / 0.01 | 0.06 / 0.06 | 0.18 / 0.18 | -0.04 / -0.04 | 0.00 / 0.00 | 0.02 / 0.02 | 0.12 / 0.12 | 0.10 / 0.10 | 0.06 / 0.07 | -0.03 / -0.03 |
| individual | 0.03 / 0.03 | -0.02 / -0.02 | -0.22 / -0.22 | 0.04 / 0.04 | 0.00 / 0.00 | -0.12 / -0.12 | -0.04 / -0.04 | -0.10 / -0.10 | 0.04 / 0.04 | -0.01 / -0.01 | 0.04 / 0.04 | 0.03 / 0.03 | 0.02 / 0.02 | -0.04 / -0.03 | 0.06 / 0.08 | -0.04 / -0.04 |
| grey | -0.06 / -0.06 | 0.28 / 0.28 | -0.28 / -0.27 | 0.00 / 0.00 | 0.04 / 0.05 | -0.14 / -0.14 | 0.07 / 0.07 | 0.09 / 0.09 | 0.09 / 0.09 | -0.00 / -0.00 | -0.09 / -0.09 | 0.09 / 0.09 | 0.08 / 0.08 | -0.12 / -0.10 | 0.11 / 0.12 | -0.04 / -0.04 |
| bright | 0.10 / 0.10 | 0.04 / 0.04 | -0.29 / -0.29 | -0.03 / -0.04 | 0.05 / 0.06 | -0.18 / -0.18 | 0.12 / 0.12 | -0.05 / -0.05 | 0.03 / 0.03 | -0.01 / -0.01 | -0.15 / -0.15 | -0.02 / -0.02 | 0.09 / 0.09 | -0.11 / -0.10 | -0.08 / -0.05 | -0.02 / -0.02 |
| oneself | -0.07 / -0.07 | -0.03 / -0.03 | -0.15 / -0.15 | 0.10 / 0.10 | 0.01 / 0.01 | -0.01 / -0.01 | 0.23 / 0.23 | 0.02 / 0.02 | 0.11 / 0.11 | -0.01 / -0.01 | -0.05 / -0.05 | 0.38 / 0.38 | 0.02 / 0.02 | 0.15 / 0.16 | -0.26 / -0.25 | 0.00 / 0.00 |
| countable | 0.07 / 0.06 | 0.43 / 0.42 | -0.11 / -0.11 | 0.10 / 0.09 | 0.10 / 0.11 | -0.06 / -0.06 | 0.49 / 0.47 | 0.21 / 0.21 | -0.03 / -0.03 | 0.07 / 0.07 | -0.16 / -0.16 | 0.18 / 0.17 | 0.19 / 0.19 | 0.09 / 0.10 | -0.22 / -0.20 | -0.02 / -0.02 |
| insufficient | -0.13 / -0.13 | 0.21 / 0.21 | -0.04 / -0.03 | 0.04 / 0.04 | 0.16 / 0.16 | -0.03 / -0.03 | -0.09 / -0.09 | -0.06 / -0.05 | 0.07 / 0.07 | 0.01 / 0.01 | -0.13 / -0.13 | -0.06 / -0.06 | 0.02 / 0.02 | 0.08 / 0.09 | -0.09 / -0.07 | -0.02 / -0.02 |
| twentieth | 0.07 / 0.07 | 0.13 / 0.12 | -0.24 / -0.24 | 0.05 / 0.05 | 0.00 / 0.01 | -0.07 / -0.07 | 0.04 / 0.05 | 0.07 / 0.07 | -0.01 / -0.01 | 0.01 / 0.01 | -0.13 / -0.12 | 0.09 / 0.09 | 0.06 / 0.07 | -0.17 / -0.15 | -0.22 / -0.19 | -0.01 / -0.01 |
| initial | -0.17 / -0.17 | 0.04 / 0.04 | -0.07 / -0.07 | 0.01 / 0.01 | 0.08 / 0.09 | -0.06 / -0.06 | 0.06 / 0.06 | 0.10 / 0.10 | 0.11 / 0.10 | 0.02 / 0.02 | -0.08 / -0.08 | -0.07 / -0.07 | 0.01 / 0.01 | -0.03 / -0.02 | 0.06 / 0.08 | -0.03 / -0.03 |
| pink | -0.04 / -0.04 | 0.29 / 0.29 | -0.20 / -0.20 | 0.02 / 0.01 | 0.14 / 0.15 | -0.09 / -0.09 | 0.18 / 0.18 | 0.12 / 0.12 | 0.08 / 0.08 | 0.02 / 0.02 | -0.05 / -0.05 | 0.09 / 0.08 | 0.06 / 0.06 | -0.13 / -0.11 | 0.09 / 0.11 | -0.02 / -0.02 |
| quarter | -0.06 / -0.06 | 0.59 / 0.59 | -0.18 / -0.18 | 0.01 / 0.01 | 0.14 / 0.14 | -0.08 / -0.08 | -0.15 / -0.15 | 0.05 / 0.05 | -0.01 / -0.01 | 0.03 / 0.03 | 0.00 / 0.00 | 0.15 / 0.15 | 0.06 / 0.06 | 0.03 / 0.03 | -0.17 / -0.16 | -0.04 / -0.04 |

Measured / predicted self-attention weight change ΔA_h(p_c, p_c), token means over the Y2 (the new frames) frames:

| token | L01.H00 | L01.H01 | L01.H02 | L01.H03 | L01.H04 | L01.H05 | L01.H06 | L01.H07 | L02.H00 | L02.H01 | L02.H02 | L02.H03 | L02.H04 | L02.H05 | L02.H06 | L02.H07 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yourselves | -0.01 / -0.01 | -0.07 / -0.07 | -0.23 / -0.23 | 0.09 / 0.09 | 0.04 / 0.05 | -0.02 / -0.02 | 0.10 / 0.11 | -0.10 / -0.10 | 0.08 / 0.08 | -0.01 / -0.01 | 0.01 / 0.01 | 0.40 / 0.40 | 0.07 / 0.07 | 0.19 / 0.19 | 0.11 / 0.13 | 0.01 / 0.01 |
| orange | -0.04 / -0.03 | 0.36 / 0.36 | -0.12 / -0.12 | 0.00 / 0.00 | 0.13 / 0.13 | -0.14 / -0.14 | 0.27 / 0.26 | 0.17 / 0.16 | 0.04 / 0.04 | -0.01 / -0.01 | -0.09 / -0.10 | 0.19 / 0.18 | 0.03 / 0.03 | -0.10 / -0.10 | -0.24 / -0.23 | -0.02 / -0.02 |
| heavy | -0.14 / -0.14 | 0.10 / 0.09 | -0.29 / -0.29 | -0.01 / -0.01 | 0.11 / 0.11 | -0.17 / -0.17 | -0.12 / -0.12 | -0.03 / -0.03 | 0.06 / 0.06 | 0.00 / 0.00 | -0.17 / -0.17 | 0.02 / 0.02 | -0.04 / -0.04 | -0.20 / -0.20 | -0.17 / -0.15 | -0.03 / -0.03 |
| respective | 0.21 / 0.21 | 0.02 / 0.02 | 0.03 / 0.03 | 0.07 / 0.07 | 0.10 / 0.10 | -0.04 / -0.04 | 0.05 / 0.05 | -0.23 / -0.23 | 0.08 / 0.08 | 0.04 / 0.03 | -0.01 / -0.01 | 0.09 / 0.09 | -0.02 / -0.02 | 0.07 / 0.07 | -0.04 / -0.00 | -0.02 / -0.02 |
| twin | -0.08 / -0.08 | 0.26 / 0.26 | -0.29 / -0.29 | -0.00 / -0.00 | -0.03 / -0.03 | -0.05 / -0.05 | 0.05 / 0.05 | -0.00 / -0.00 | 0.07 / 0.06 | 0.01 / 0.00 | -0.05 / -0.05 | 0.05 / 0.04 | 0.03 / 0.03 | -0.30 / -0.29 | 0.00 / 0.04 | -0.04 / -0.04 |
| dual | -0.03 / -0.03 | 0.08 / 0.07 | -0.07 / -0.07 | 0.02 / 0.02 | 0.14 / 0.14 | -0.15 / -0.15 | -0.10 / -0.10 | 0.12 / 0.11 | 0.15 / 0.15 | 0.06 / 0.05 | 0.04 / 0.05 | -0.04 / -0.04 | 0.03 / 0.03 | -0.12 / -0.11 | -0.13 / -0.08 | -0.03 / -0.03 |
| brown | 0.02 / 0.01 | 0.28 / 0.27 | -0.26 / -0.26 | -0.04 / -0.05 | -0.04 / -0.04 | -0.12 / -0.12 | 0.04 / 0.04 | 0.29 / 0.28 | 0.01 / 0.01 | 0.02 / 0.01 | -0.14 / -0.14 | 0.09 / 0.08 | 0.13 / 0.13 | -0.15 / -0.15 | -0.02 / -0.00 | -0.02 / -0.02 |
| unlimited | -0.05 / -0.05 | 0.34 / 0.34 | 0.02 / 0.03 | -0.00 / -0.01 | 0.23 / 0.23 | 0.08 / 0.08 | 0.02 / 0.03 | 0.24 / 0.24 | 0.08 / 0.08 | 0.05 / 0.05 | -0.07 / -0.07 | 0.03 / 0.02 | 0.08 / 0.08 | 0.05 / 0.05 | -0.07 / -0.03 | -0.00 / -0.00 |
| total | 0.09 / 0.09 | 0.37 / 0.36 | 0.09 / 0.09 | 0.04 / 0.04 | 0.10 / 0.10 | -0.21 / -0.21 | -0.01 / -0.01 | 0.14 / 0.14 | 0.15 / 0.14 | -0.04 / -0.04 | 0.02 / 0.02 | 0.06 / 0.06 | 0.15 / 0.15 | 0.14 / 0.13 | 0.09 / 0.10 | -0.03 / -0.03 |
| upper | -0.01 / -0.01 | 0.28 / 0.28 | 0.13 / 0.13 | -0.03 / -0.03 | 0.01 / 0.01 | -0.05 / -0.05 | -0.06 / -0.07 | 0.28 / 0.27 | 0.05 / 0.05 | -0.04 / -0.04 | -0.12 / -0.12 | 0.05 / 0.05 | 0.11 / 0.11 | -0.02 / -0.02 | 0.31 / 0.32 | -0.04 / -0.04 |
| extensive | -0.11 / -0.11 | 0.14 / 0.13 | -0.22 / -0.23 | 0.02 / 0.02 | 0.03 / 0.04 | -0.10 / -0.09 | 0.04 / 0.03 | -0.11 / -0.10 | 0.06 / 0.06 | 0.01 / 0.00 | -0.12 / -0.11 | -0.08 / -0.09 | -0.01 / -0.01 | -0.02 / -0.02 | 0.18 / 0.21 | -0.02 / -0.02 |
| countable | 0.09 / 0.07 | 0.44 / 0.43 | -0.14 / -0.13 | 0.07 / 0.07 | 0.08 / 0.09 | -0.07 / -0.06 | 0.51 / 0.49 | 0.25 / 0.25 | -0.05 / -0.05 | 0.06 / 0.06 | -0.16 / -0.17 | 0.19 / 0.17 | 0.23 / 0.23 | 0.12 / 0.12 | -0.18 / -0.15 | -0.02 / -0.02 |
| myself | -0.06 / -0.06 | -0.07 / -0.07 | -0.30 / -0.30 | 0.08 / 0.08 | -0.04 / -0.04 | -0.04 / -0.04 | 0.18 / 0.18 | -0.14 / -0.14 | 0.06 / 0.06 | -0.03 / -0.03 | -0.01 / -0.01 | 0.28 / 0.28 | 0.06 / 0.06 | 0.14 / 0.14 | 0.26 / 0.27 | 0.01 / 0.01 |
| individual | 0.03 / 0.03 | -0.00 / -0.00 | -0.24 / -0.24 | 0.03 / 0.03 | -0.01 / -0.01 | -0.12 / -0.12 | -0.06 / -0.06 | -0.03 / -0.03 | 0.01 / 0.01 | -0.01 / -0.01 | 0.05 / 0.04 | 0.02 / 0.02 | 0.01 / 0.01 | -0.01 / -0.02 | 0.15 / 0.17 | -0.03 / -0.03 |
| specific | -0.10 / -0.10 | -0.01 / -0.01 | -0.26 / -0.26 | 0.02 / 0.02 | 0.03 / 0.03 | -0.08 / -0.07 | 0.19 / 0.19 | -0.01 / -0.01 | 0.02 / 0.02 | 0.01 / 0.01 | -0.02 / -0.02 | -0.08 / -0.08 | 0.01 / 0.01 | 0.04 / 0.04 | 0.05 / 0.07 | -0.02 / -0.02 |
| grey | -0.04 / -0.04 | 0.29 / 0.28 | -0.30 / -0.30 | -0.01 / -0.01 | 0.04 / 0.04 | -0.16 / -0.16 | 0.06 / 0.06 | 0.16 / 0.16 | 0.08 / 0.08 | -0.00 / -0.01 | -0.12 / -0.12 | 0.13 / 0.12 | 0.10 / 0.09 | -0.08 / -0.07 | 0.14 / 0.16 | -0.03 / -0.03 |
| bright | 0.12 / 0.12 | 0.05 / 0.05 | -0.31 / -0.30 | -0.05 / -0.05 | 0.06 / 0.07 | -0.18 / -0.18 | 0.12 / 0.12 | 0.04 / 0.04 | 0.03 / 0.04 | -0.01 / -0.02 | -0.16 / -0.16 | 0.02 / 0.02 | 0.08 / 0.08 | -0.11 / -0.10 | -0.04 / -0.01 | -0.02 / -0.02 |
| whoever | 0.00 / 0.00 | 0.22 / 0.22 | -0.31 / -0.31 | -0.03 / -0.03 | 0.14 / 0.13 | -0.04 / -0.04 | -0.11 / -0.11 | 0.14 / 0.14 | 0.13 / 0.13 | 0.02 / 0.02 | -0.04 / -0.04 | 0.41 / 0.41 | 0.25 / 0.25 | 0.08 / 0.08 | -0.28 / -0.28 | 0.01 / 0.01 |
| insufficient | -0.11 / -0.11 | 0.23 / 0.23 | -0.08 / -0.08 | 0.01 / 0.01 | 0.14 / 0.14 | -0.05 / -0.05 | -0.07 / -0.07 | -0.02 / -0.02 | 0.04 / 0.04 | 0.02 / 0.01 | -0.15 / -0.15 | -0.04 / -0.04 | 0.01 / 0.01 | 0.12 / 0.12 | -0.00 / 0.04 | -0.02 / -0.02 |
| oneself | -0.05 / -0.06 | 0.01 / 0.01 | -0.19 / -0.19 | 0.10 / 0.09 | 0.01 / 0.01 | -0.01 / -0.01 | 0.32 / 0.33 | 0.06 / 0.06 | 0.06 / 0.06 | -0.02 / -0.02 | -0.06 / -0.06 | 0.36 / 0.35 | 0.01 / 0.01 | 0.18 / 0.18 | -0.16 / -0.15 | -0.00 / -0.00 |
| pink | -0.00 / -0.01 | 0.28 / 0.28 | -0.24 / -0.25 | 0.00 / -0.00 | 0.12 / 0.13 | -0.11 / -0.11 | 0.15 / 0.15 | 0.19 / 0.19 | 0.07 / 0.07 | 0.02 / 0.02 | -0.07 / -0.08 | 0.12 / 0.12 | 0.07 / 0.07 | -0.10 / -0.09 | 0.13 / 0.16 | -0.02 / -0.02 |
| twentieth | 0.11 / 0.11 | 0.14 / 0.13 | -0.29 / -0.29 | 0.04 / 0.04 | -0.00 / -0.00 | -0.08 / -0.08 | 0.06 / 0.07 | 0.07 / 0.06 | -0.02 / -0.02 | 0.01 / 0.00 | -0.14 / -0.15 | 0.07 / 0.06 | 0.06 / 0.05 | -0.14 / -0.14 | -0.16 / -0.13 | -0.02 / -0.02 |
| initial | -0.17 / -0.17 | 0.07 / 0.07 | -0.13 / -0.13 | 0.01 / 0.01 | 0.06 / 0.07 | -0.04 / -0.04 | 0.02 / 0.02 | 0.20 / 0.20 | 0.09 / 0.09 | 0.01 / 0.01 | -0.07 / -0.07 | -0.06 / -0.07 | 0.03 / 0.03 | -0.02 / -0.02 | 0.14 / 0.16 | -0.03 / -0.03 |
| quarter | -0.03 / -0.03 | 0.62 / 0.62 | -0.23 / -0.23 | 0.00 / 0.00 | 0.13 / 0.13 | -0.08 / -0.08 | -0.13 / -0.13 | 0.11 / 0.11 | -0.02 / -0.02 | 0.03 / 0.03 | 0.00 / 0.00 | 0.21 / 0.21 | 0.05 / 0.05 | 0.05 / 0.05 | -0.12 / -0.11 | -0.04 / -0.04 |

## Execution ledger

- Executed prompt keys: 1782
- Executed noun keys: 80
