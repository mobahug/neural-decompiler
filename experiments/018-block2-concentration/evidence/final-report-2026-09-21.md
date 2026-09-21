# Experiment 018 Report

- Run ID: `86ccb888f52600ef`
- Confirmation sha256: `e88c0625d105297ef45da0605a1c4453fa0d974791f0cef00af5c776907557f2`
- Experiment 017 lock sha256: `b4fc9014ade7d21d2cd2e5be391ce6234e46fb46887880c0c4d03517d411ed72`
- Protocol/code commit at explore: `9b629bf905a59572d6fe9b0ca130460d7de469ef`

## Phases

- `confirm`: `complete`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

## Tier A — exposed pool (calibration record only; the floors are frozen constants)

- Replication of Experiment 017: 9636 pairs, max deviation 0.00e+00
- Identities (checks only): I1_patched_rows 2.2e-05, I1_reference_rows 9.8e-06, I2_chain 3.4e-06, I3_head_split 9.9e-07, I4_x3 8.5e-06, I5_head_row 1.2e-05, I5_reference_head_row 8.7e-06, I6_dT 3.3e-05, I7_split 8.6e-07, I8_reference_rung 1.5e-14, head_level1_recovery 6.4e-12, level0_015_recovery 1.0e-14, level1_recovery 1.4e-14, neuron_sum 2.3e-08, p1_cross_check 6.6e-16, rho_identity 1.1e-07; read weight vs Experiment 011 lock 0.0e+00
- Ranking pool: 9636 pairs (12848 records); top neurons [1987, 1102, 1726, 129, 1311, 1310, 1068, 1924, 1671, 815]; S_1 [1987]; overlaps {'R1&B256': 42, 'R1&R2': 36, 'R1&R3': 36, 'R2&B256': 31, 'R2&R3': 34, 'R3&B256': 27, 'S256&B256': 0, 'S256&R1': 34, 'S256&R2': 28, 'S256&R3': 37}; per-frame top-256 overlap with S_256 min/median/max 121/138/163
- Block-2 base at p_t over 26 coordinated frames; program: layer 1: d_head 64, rotary_dim 16, base 10000, layer 2: d_head 64, rotary_dim 16, base 10000, layer 3: d_head 64, rotary_dim 16, base 10000
  - pooled (9636 pairs) — rung: c_L R² (κ) / F R² (κ) / Π R² (κ) / ΔT R² (κ) / rows R² (κ):
    - `S0`: 0.836 (0.00) / 0.947 (0.00) / 0.812 (0.00) / 0.959 (—) / 0.943 (0.00)
    - `S1`: 0.903 (0.41) / 0.967 (0.38) / 0.873 (0.33) / 0.971 (—) / 0.945 (0.04)
    - `S4`: 0.943 (0.65) / 0.979 (0.61) / 0.903 (0.49) / 0.980 (—) / 0.945 (0.04)
    - `S16`: 0.953 (0.72) / 0.983 (0.68) / 0.911 (0.54) / 0.983 (—) / 0.948 (0.09)
    - `S64`: 0.960 (0.76) / 0.985 (0.73) / 0.917 (0.57) / 0.985 (—) / 0.949 (0.12)
    - `S256`: 0.980 (0.88) / 0.993 (0.88) / 0.935 (0.67) / 0.992 (—) / 0.955 (0.22)
    - `S1024`: 0.997 (0.99) / 0.999 (0.99) / 0.975 (0.88) / 0.998 (—) / 0.976 (0.62)
    - `S2048`: 1.000 (1.00) / 1.000 (1.00) / 0.996 (1.00) / 1.000 (—) / 0.996 (1.00)
    - `R1`: 0.906 (0.43) / 0.968 (0.41) / 0.890 (0.42) / 0.972 (—) / 0.951 (0.16)
    - `R2`: 0.870 (0.20) / 0.959 (0.22) / 0.848 (0.20) / 0.968 (—) / 0.952 (0.18)
    - `R3`: 0.911 (0.46) / 0.969 (0.43) / 0.876 (0.35) / 0.973 (—) / 0.947 (0.08)
    - `B256`: 0.836 (-0.00) / 0.946 (-0.00) / 0.811 (-0.00) / 0.959 (—) / 0.946 (0.05)
    - `oracle`: 0.990 (0.94) / 0.997 (0.94) / 0.961 (0.81) / 0.995 (—) / 0.970 (0.50)
    - gaps: F 0.053, Pi 0.184, c_L 0.163, dT 0.041, rows 0.054; frozen pattern ΔT R² 0.938; Experiment 016's decoded c_L R² 0.836
  - predeclared orderings: row diffuse κ_row(S_256) 0.22 < 0.5 holds True; random margin κ_c_L 0.88 vs best random 0.46 / bottom -0.00 holds True; κ_c_L ≥ κ_Π holds True
- S_256 firing per template (descriptive; effect above 0.001 in read units): cardinal: 71 of 256 fire in at least half of 3212 records, top [(1987, 0.0221), (1726, 0.0117), (1102, 0.0076), (1068, 0.0056), (129, 0.0052)]; coordinated-adjective: 15 of 256 fire in at least half of 6424 records, top [(1987, 0.0124), (1102, 0.006), (1726, 0.0051), (129, 0.0035), (1311, 0.0033)]; quantifier: 93 of 256 fire in at least half of 3212 records, top [(1987, 0.0351), (1102, 0.0166), (1726, 0.0097), (129, 0.0078), (1311, 0.0057)]
- Split — cue-final: 6424 pairs, 231 tokens, 52 frames: c_L R² S_0 0.811 / S_256 0.981 / S_2048 1.000, gap 0.189, κ_c_L 0.90 (S_1 0.45); Π κ 0.69 (gap 0.190); frozen ΔT R² 0.925 vs reference 1.000
- Split — coordinated: 3212 pairs, 230 tokens, 26 frames: c_L R² S_0 0.841 / S_256 0.966 / S_2048 0.999, gap 0.159, κ_c_L 0.79 (S_1 0.20); Π κ 0.54 (gap 0.164); frozen ΔT R² 0.971 vs reference 1.000
- Template cardinal: 3212 pairs, 230 tokens, 26 frames: c_L R² S_0 0.875 / S_256 0.978 / S_2048 0.999, gap 0.124, κ_c_L 0.83 (S_1 0.15); Π κ 0.47 (gap 0.086); frozen ΔT R² 0.872 vs reference 0.998
- Template coordinated-adjective: 3212 pairs, 230 tokens, 26 frames: c_L R² S_0 0.841 / S_256 0.966 / S_2048 0.999, gap 0.159, κ_c_L 0.79 (S_1 0.20); Π κ 0.54 (gap 0.164); frozen ΔT R² 0.971 vs reference 1.000
- Template quantifier: 3212 pairs, 230 tokens, 26 frames: c_L R² S_0 0.496 / S_256 0.958 / S_2048 1.000, gap 0.503, κ_c_L 0.92 (S_1 0.52); Π κ 0.73 (gap 0.241); frozen ΔT R² 0.839 vs reference 1.000

## Lock

- Candidate lock sha256 `fdfae9106898f51528fc6c1a3e235bd9be7f37f3ccb00628ccf7522e308fae46`; predictions sha256 `aac1b7c92a12a2af57cfa273e6bc8f8abb7a0fe26d2f5af47356bff1890d329d`

## Confirmation — stage 1 (fresh frames' reference states; prediction table digested before any fresh cue prompt)

- Table rows 288; digest `a8d55186e3c3aa99f61ad993be1e964a031b74fbee39bb271d344ef98755ff50`; commit `6fa4c489452b88f7409c7636ac1b120b1c208ae7`
  - cardinal-018-1: valid (plural head change 1.662, cue effect 79/72; p_c 3, p_t 3); own top-256 overlap with S_256 129
  - cardinal-018-2: valid (plural head change 1.667, cue effect 79/72; p_c 3, p_t 3); own top-256 overlap with S_256 140
  - cardinal-018-3: valid (plural head change 0.968, cue effect 77/72; p_c 3, p_t 3); own top-256 overlap with S_256 142
  - cardinal-018-4: valid (plural head change 1.155, cue effect 79/72; p_c 4, p_t 4); own top-256 overlap with S_256 132
  - coordinated-adjective-018-1: valid (plural head change 2.494, cue effect 79/72; p_c 5, p_t 6); own top-256 overlap with S_256 145
  - coordinated-adjective-018-2: valid (plural head change 2.121, cue effect 79/72; p_c 6, p_t 7); own top-256 overlap with S_256 144
  - coordinated-adjective-018-3: valid (plural head change 1.968, cue effect 79/72; p_c 7, p_t 8); own top-256 overlap with S_256 144
  - coordinated-adjective-018-4: valid (plural head change 2.844, cue effect 79/72; p_c 6, p_t 7); own top-256 overlap with S_256 148
  - quantifier-018-1: valid (plural head change 2.122, cue effect 79/72; p_c 5, p_t 5); own top-256 overlap with S_256 124
  - quantifier-018-2: valid (plural head change 2.313, cue effect 79/72; p_c 3, p_t 3); own top-256 overlap with S_256 139
  - quantifier-018-3: valid (plural head change 2.598, cue effect 79/72; p_c 3, p_t 3); own top-256 overlap with S_256 118
  - quantifier-018-4: valid (plural head change 1.902, cue effect 79/72; p_c 4, p_t 4); own top-256 overlap with S_256 131

## Confirmation — stage 2 — `CHANNEL_D_CONCENTRATED_TOKENS | CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL | SINGLE_NEURON_REJECTED | PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES`

- Y1 (strict prospective: fresh cues × exposed frames): scored tokens 24 (precondition ok); κ_c_L(S_256) 0.863 (≥ 0.7), κ_Π(S_256) 0.724 (≥ 0.5) → pass 
  - reference rung: c_L R² 0.999, rows 0.995, ΔT 0.999; gaps F 0.077, Pi 0.383, c_L 0.202, dT 0.051, rows 0.053; single neuron κ_c_L 0.24, gap to S_256 0.63
  - pooled pairs (1872 pairs) — rung: c_L R² (κ) / F R² (κ) / Π R² (κ) / ΔT R² (κ) / rows R² (κ):
    - `S0`: 0.798 (0.00) / 0.923 (0.00) / 0.610 (0.00) / 0.949 (0.00) / 0.941 (0.00)
    - `S1`: 0.846 (0.24) / 0.940 (0.22) / 0.701 (0.24) / 0.958 (0.18) / 0.943 (0.03)
    - `S4`: 0.918 (0.60) / 0.966 (0.56) / 0.816 (0.54) / 0.975 (0.51) / 0.943 (0.03)
    - `S16`: 0.931 (0.66) / 0.972 (0.64) / 0.830 (0.57) / 0.979 (0.60) / 0.945 (0.07)
    - `S64`: 0.942 (0.72) / 0.976 (0.69) / 0.846 (0.62) / 0.982 (0.65) / 0.947 (0.10)
    - `S256`: 0.972 (0.86) / 0.988 (0.86) / 0.888 (0.72) / 0.991 (0.83) / 0.953 (0.22)
    - `S1024`: 0.996 (0.98) / 0.998 (0.99) / 0.963 (0.92) / 0.998 (0.97) / 0.973 (0.59)
    - `S2048`: 0.999 (1.00) / 0.999 (1.00) / 0.993 (1.00) / 0.999 (1.00) / 0.995 (1.00)
    - `R1`: 0.845 (0.23) / 0.941 (0.24) / 0.733 (0.32) / 0.957 (0.16) / 0.950 (0.16)
    - `R2`: 0.850 (0.26) / 0.944 (0.27) / 0.698 (0.23) / 0.962 (0.26) / 0.948 (0.13)
    - `R3`: 0.862 (0.32) / 0.946 (0.30) / 0.712 (0.27) / 0.963 (0.27) / 0.946 (0.10)
    - `B256`: 0.798 (-0.00) / 0.922 (-0.01) / 0.613 (0.01) / 0.949 (0.00) / 0.946 (0.08)
    - `oracle`: 0.986 (0.93) / 0.995 (0.94) / 0.936 (0.85) / 0.995 (0.92) / 0.967 (0.49)
    - gaps: F 0.077, Pi 0.383, c_L 0.202, dT 0.051, rows 0.053; frozen pattern ΔT R² 0.968; Experiment 016's decoded c_L R² 0.798
  - predeclared orderings: row diffuse κ_row(S_256) 0.22 < 0.5 holds True; random margin κ_c_L 0.86 vs best random 0.32 / bottom -0.00 holds True; κ_c_L ≥ κ_Π holds True
  - token means: κ_c_L(S_256) 0.89, κ_Π 0.78, κ_F 0.91, κ_ΔT 0.89; reference c_L R² 1.000
  - split — cue-final: 1248 pairs, 24 tokens, 52 frames: c_L R² S_0 0.771 / S_256 0.973 / S_2048 0.999, gap 0.228, κ_c_L 0.89 (S_1 0.29); Π κ 0.75 (gap 0.418); frozen ΔT R² 0.970 vs reference 1.000; coordinated: 624 pairs, 24 tokens, 26 frames: c_L R² S_0 0.704 / S_256 0.933 / S_2048 0.999, gap 0.295, κ_c_L 0.78 (S_1 0.06); Π κ 0.61 (gap 0.288); frozen ΔT R² 0.953 vs reference 0.998
  - template cardinal: 624 pairs, 24 tokens, 26 frames: c_L R² S_0 0.802 / S_256 0.966 / S_2048 0.998, gap 0.196, κ_c_L 0.84 (S_1 0.04); Π κ 0.73 (gap 0.216); frozen ΔT R² 0.935 vs reference 0.998
  - template coordinated-adjective: 624 pairs, 24 tokens, 26 frames: c_L R² S_0 0.704 / S_256 0.933 / S_2048 0.999, gap 0.295, κ_c_L 0.78 (S_1 0.06); Π κ 0.61 (gap 0.288); frozen ΔT R² 0.953 vs reference 0.998
  - template quantifier: 624 pairs, 24 tokens, 26 frames: c_L R² S_0 0.401 / S_256 0.938 / S_2048 1.000, gap 0.598, κ_c_L 0.90 (S_1 0.35); Π κ 0.75 (gap 0.502); frozen ΔT R² 0.922 vs reference 1.000
- Y2 (frame-conditional prospective with the split and no-harm guards: fresh cues × new frames): scored tokens 24 (precondition ok); κ_c_L(S_256) 0.843 (≥ 0.7), κ_Π(S_256) 0.798 (≥ 0.5); split guard coordinated κ_c_L 0.86 (≥ 0.4, 4 frames) ok, cue_final κ_c_L 0.84 (≥ 0.6, 8 frames) ok; no-harm guard FAILED {'quantifier-018-4': -0.10152989354533914} → FAIL ['no_harm_guard']
  - reference rung: c_L R² 0.999, rows 0.996, ΔT 0.999; gaps F 0.140, Pi 0.752, c_L 0.249, dT 0.084, rows 0.060; single neuron κ_c_L 0.12, gap to S_256 0.73
  - pooled pairs (288 pairs) — rung: c_L R² (κ) / F R² (κ) / Π R² (κ) / ΔT R² (κ) / rows R² (κ):
    - `S0`: 0.750 (0.00) / 0.859 (0.00) / 0.239 (0.00) / 0.915 (0.00) / 0.935 (0.00)
    - `S1`: 0.779 (0.12) / 0.869 (0.08) / 0.310 (0.09) / 0.921 (0.08) / 0.935 (-0.00)
    - `S4`: 0.875 (0.50) / 0.923 (0.46) / 0.564 (0.43) / 0.953 (0.45) / 0.942 (0.12)
    - `S16`: 0.898 (0.59) / 0.937 (0.56) / 0.673 (0.58) / 0.958 (0.51) / 0.944 (0.15)
    - `S64`: 0.919 (0.68) / 0.952 (0.66) / 0.751 (0.68) / 0.965 (0.60) / 0.945 (0.16)
    - `S256`: 0.960 (0.84) / 0.980 (0.87) / 0.840 (0.80) / 0.985 (0.84) / 0.951 (0.26)
    - `S1024`: 0.996 (0.99) / 0.996 (0.98) / 0.950 (0.95) / 0.996 (0.97) / 0.976 (0.68)
    - `S2048`: 0.999 (1.00) / 0.998 (1.00) / 0.991 (1.00) / 0.999 (1.00) / 0.996 (1.00)
    - `R1`: 0.754 (0.01) / 0.864 (0.04) / 0.204 (-0.05) / 0.920 (0.06) / 0.926 (-0.16)
    - `R2`: 0.792 (0.17) / 0.889 (0.22) / 0.386 (0.20) / 0.933 (0.22) / 0.943 (0.13)
    - `R3`: 0.775 (0.10) / 0.875 (0.12) / 0.375 (0.18) / 0.925 (0.12) / 0.940 (0.08)
    - `B256`: 0.749 (-0.00) / 0.859 (0.00) / 0.303 (0.08) / 0.913 (-0.02) / 0.941 (0.10)
    - `oracle`: 0.985 (0.94) / 0.989 (0.93) / 0.903 (0.88) / 0.991 (0.91) / 0.961 (0.43)
    - gaps: F 0.140, Pi 0.752, c_L 0.249, dT 0.084, rows 0.060; frozen pattern ΔT R² 0.972; Experiment 016's decoded c_L R² 0.750
  - predeclared orderings: row diffuse κ_row(S_256) 0.26 < 0.5 holds True; random margin κ_c_L 0.84 vs best random 0.17 / bottom -0.00 holds True; κ_c_L ≥ κ_Π holds True
  - token means: κ_c_L(S_256) 0.91, κ_Π 0.82, κ_F 0.91, κ_ΔT 0.91; reference c_L R² 0.999
  - split — cue-final: 192 pairs, 24 tokens, 8 frames: c_L R² S_0 0.613 / S_256 0.936 / S_2048 0.999, gap 0.386, κ_c_L 0.84 (S_1 0.16); Π κ 0.81 (gap 1.088); frozen ΔT R² 0.975 vs reference 0.999; coordinated: 96 pairs, 24 tokens, 4 frames: c_L R² S_0 0.648 / S_256 0.951 / S_2048 0.999, gap 0.351, κ_c_L 0.86 (S_1 -0.07); Π κ 0.72 (gap 0.342); frozen ΔT R² 0.964 vs reference 0.997
  - template cardinal: 96 pairs, 24 tokens, 4 frames: c_L R² S_0 0.730 / S_256 0.930 / S_2048 0.996, gap 0.266, κ_c_L 0.75 (S_1 0.29); Π κ 0.52 (gap 0.187); frozen ΔT R² 0.956 vs reference 0.996
  - template coordinated-adjective: 96 pairs, 24 tokens, 4 frames: c_L R² S_0 0.648 / S_256 0.951 / S_2048 0.999, gap 0.351, κ_c_L 0.86 (S_1 -0.07); Π κ 0.72 (gap 0.342); frozen ΔT R² 0.964 vs reference 0.997
  - template quantifier: 96 pairs, 24 tokens, 4 frames: c_L R² S_0 0.210 / S_256 0.887 / S_2048 1.000, gap 0.789, κ_c_L 0.86 (S_1 0.13); Π κ 0.83 (gap 1.670); frozen ΔT R² 0.886 vs reference 0.999
  - frame cardinal-018-1 (cardinal): 24 pairs; c_L R² S_0 0.580 / S_1 0.864 / S_256 0.960 / S_2048 0.995 / oracle 0.979 (κ_c_L 0.91, no-harm 0.380); rows S_256 0.862 / S_2048 0.894; Π S_256 0.884 / S_2048 0.929; ΔT reference 0.990, frozen 0.951
  - frame cardinal-018-2 (cardinal): 24 pairs; c_L R² S_0 0.743 / S_1 0.616 / S_256 0.958 / S_2048 0.986 / oracle 0.966 (κ_c_L 0.89, no-harm 0.215); rows S_256 0.741 / S_2048 0.995; Π S_256 0.719 / S_2048 0.979; ΔT reference 0.987, frozen 0.927
  - frame cardinal-018-3 (cardinal): 24 pairs; c_L R² S_0 0.940 / S_1 0.885 / S_256 0.982 / S_2048 0.998 / oracle 0.981 (κ_c_L 0.72, no-harm 0.042); rows S_256 0.920 / S_2048 0.988; Π S_256 0.934 / S_2048 0.990; ΔT reference 0.997, frozen 0.913
  - frame cardinal-018-4 (cardinal): 24 pairs; c_L R² S_0 0.542 / S_1 0.724 / S_256 0.808 / S_2048 1.000 / oracle 0.961 (κ_c_L 0.58, no-harm 0.267); rows S_256 0.976 / S_2048 0.994; Π S_256 0.928 / S_2048 0.994; ΔT reference 0.996, frozen 0.771
  - frame coordinated-adjective-018-1 (coordinated-adjective): 24 pairs; c_L R² S_0 0.722 / S_1 0.606 / S_256 0.961 / S_2048 0.997 / oracle 0.981 (κ_c_L 0.87, no-harm 0.239); rows S_256 0.963 / S_2048 1.000; Π S_256 0.932 / S_2048 0.994; ΔT reference 0.995, frozen 0.844
  - frame coordinated-adjective-018-2 (coordinated-adjective): 24 pairs; c_L R² S_0 -0.633 / S_1 -1.044 / S_256 0.776 / S_2048 0.997 / oracle 0.926 (κ_c_L 0.86, no-harm 1.409); rows S_256 0.937 / S_2048 0.998; Π S_256 0.741 / S_2048 0.972; ΔT reference 0.977, frozen 0.922
  - frame coordinated-adjective-018-3 (coordinated-adjective): 24 pairs; c_L R² S_0 0.667 / S_1 0.704 / S_256 0.936 / S_2048 0.999 / oracle 0.940 (κ_c_L 0.81, no-harm 0.269); rows S_256 0.977 / S_2048 0.996; Π S_256 0.924 / S_2048 0.990; ΔT reference 0.999, frozen 0.949
  - frame coordinated-adjective-018-4 (coordinated-adjective): 24 pairs; c_L R² S_0 0.627 / S_1 0.860 / S_256 0.961 / S_2048 0.999 / oracle 0.918 (κ_c_L 0.90, no-harm 0.333); rows S_256 0.974 / S_2048 0.998; Π S_256 0.905 / S_2048 0.995; ΔT reference 0.998, frozen 0.965
  - frame quantifier-018-1 (quantifier): 24 pairs; c_L R² S_0 0.730 / S_1 0.831 / S_256 0.966 / S_2048 0.999 / oracle 0.985 (κ_c_L 0.88, no-harm 0.236); rows S_256 0.963 / S_2048 1.000; Π S_256 0.899 / S_2048 0.996; ΔT reference 0.999, frozen 0.871
  - frame quantifier-018-2 (quantifier): 24 pairs; c_L R² S_0 0.662 / S_1 0.817 / S_256 0.985 / S_2048 1.000 / oracle 0.982 (κ_c_L 0.96, no-harm 0.323); rows S_256 0.974 / S_2048 0.999; Π S_256 0.946 / S_2048 0.998; ΔT reference 1.000, frozen 0.882
  - frame quantifier-018-3 (quantifier): 24 pairs; c_L R² S_0 -3.401 / S_1 -3.084 / S_256 0.616 / S_2048 1.000 / oracle 0.933 (κ_c_L 0.91, no-harm 4.017); rows S_256 0.956 / S_2048 1.000; Π S_256 0.092 / S_2048 0.995; ΔT reference 0.999, frozen 0.813
  - frame quantifier-018-4 (quantifier): 24 pairs; c_L R² S_0 0.847 / S_1 0.854 / S_256 0.745 / S_2048 0.999 / oracle 0.924 (κ_c_L -0.67, no-harm -0.102); rows S_256 0.949 / S_2048 0.999; Π S_256 0.756 / S_2048 0.997; ΔT reference 1.000, frozen 0.858
- Y3 (single neuron on the pooled pairs of both sets, 2160 pairs = 1872 + 288): κ_c_L(S_1) 0.219 (rejected iff < 0.5) against S_256 0.860, gap 0.641 (rejected iff ≥ 0.25); pooled c_L gap 0.208; κ_Π(S_1) 0.21 → REJECTED; per set: Y1 0.24 / gap 0.63, Y2 0.12 / gap 0.73
- Y4 (secondary, reference rung only; 8 of the valid fresh cue-final frames evaluable, ≥ 6 required): frozen ΔT R² below 0.9 in every evaluable frame → FAIL ['cardinal-018-1', 'cardinal-018-2', 'cardinal-018-3']
  - cardinal-018-1: reference ΔT R² 0.990, frozen 0.951 (evaluable)
  - cardinal-018-2: reference ΔT R² 0.987, frozen 0.927 (evaluable)
  - cardinal-018-3: reference ΔT R² 0.997, frozen 0.913 (evaluable)
  - cardinal-018-4: reference ΔT R² 0.996, frozen 0.771 (evaluable, below)
  - quantifier-018-1: reference ΔT R² 0.999, frozen 0.871 (evaluable, below)
  - quantifier-018-2: reference ΔT R² 1.000, frozen 0.882 (evaluable, below)
  - quantifier-018-3: reference ΔT R² 0.999, frozen 0.813 (evaluable, below)
  - quantifier-018-4: reference ΔT R² 1.000, frozen 0.858 (evaluable, below)
  - coordinated-adjective-018-1 (coordinated, descriptive): reference ΔT R² 0.995, frozen 0.844
  - coordinated-adjective-018-2 (coordinated, descriptive): reference ΔT R² 0.977, frozen 0.922
  - coordinated-adjective-018-3 (coordinated, descriptive): reference ΔT R² 0.999, frozen 0.949
  - coordinated-adjective-018-4 (coordinated, descriptive): reference ΔT R² 0.998, frozen 0.965
  - both sets pooled (2160 pairs) — rung: c_L R² (κ) / F R² (κ) / Π R² (κ) / ΔT R² (κ) / rows R² (κ):
    - `S0`: 0.791 (0.00) / 0.916 (0.00) / 0.575 (0.00) / 0.945 (0.00) / 0.940 (0.00)
    - `S1`: 0.837 (0.22) / 0.932 (0.20) / 0.663 (0.21) / 0.954 (0.16) / 0.942 (0.03)
    - `S4`: 0.912 (0.58) / 0.961 (0.54) / 0.791 (0.52) / 0.972 (0.50) / 0.943 (0.04)
    - `S16`: 0.926 (0.65) / 0.968 (0.63) / 0.815 (0.57) / 0.977 (0.58) / 0.945 (0.08)
    - `S64`: 0.939 (0.71) / 0.973 (0.69) / 0.837 (0.63) / 0.980 (0.64) / 0.946 (0.11)
    - `S256`: 0.970 (0.86) / 0.988 (0.86) / 0.883 (0.74) / 0.990 (0.83) / 0.953 (0.23)
    - `S1024`: 0.996 (0.99) / 0.998 (0.98) / 0.962 (0.93) / 0.998 (0.97) / 0.973 (0.60)
    - `S2048`: 0.999 (1.00) / 0.999 (1.00) / 0.993 (1.00) / 0.999 (1.00) / 0.995 (1.00)
    - `R1`: 0.833 (0.20) / 0.933 (0.20) / 0.682 (0.26) / 0.953 (0.14) / 0.947 (0.12)
    - `R2`: 0.842 (0.24) / 0.938 (0.26) / 0.668 (0.22) / 0.959 (0.25) / 0.947 (0.13)
    - `R3`: 0.850 (0.28) / 0.938 (0.27) / 0.679 (0.25) / 0.959 (0.25) / 0.946 (0.10)
    - `B256`: 0.791 (-0.00) / 0.915 (-0.01) / 0.583 (0.02) / 0.945 (-0.00) / 0.945 (0.09)
    - gaps: F 0.083, Pi 0.418, c_L 0.208, dT 0.054, rows 0.054; frozen pattern ΔT R² 0.968; Experiment 016's decoded c_L R² 0.791
- S_256 firing on Y1 (descriptive): cardinal: 93 of 256 fire in at least half of 624 records, top [(1987, 0.0279), (1726, 0.0165), (1102, 0.0087), (129, 0.0062), (1068, 0.0056)]; coordinated-adjective: 23 of 256 fire in at least half of 1248 records, top [(1987, 0.0115), (1726, 0.0066), (1102, 0.0059), (1311, 0.004), (1463, 0.0038)]; quantifier: 102 of 256 fire in at least half of 624 records, top [(1987, 0.0235), (1102, 0.0209), (1726, 0.0131), (129, 0.0099), (815, 0.0068)]
- S_256 firing on Y2 (descriptive): cardinal: 119 of 256 fire in at least half of 96 records, top [(1987, 0.0358), (1726, 0.0112), (1311, 0.0086), (1924, 0.0079), (1310, 0.0067)]; coordinated-adjective: 38 of 256 fire in at least half of 192 records, top [(1987, 0.0109), (1726, 0.0097), (1463, 0.0044), (1671, 0.0039), (129, 0.0034)]; quantifier: 105 of 256 fire in at least half of 96 records, top [(1102, 0.0246), (1726, 0.0141), (129, 0.0124), (1987, 0.0115), (791, 0.0086)]

| token | class | Y1 frames | Y1 ĉ_L S_256 | Y1 ĉ_L S_2048 | Y1 c_L | Y1 ΔT̂ S_256 | Y1 ΔT | Y2 frames | Y2 ĉ_L S_256 | Y2 ĉ_L S_2048 | Y2 c_L | Y2 ΔT̂ S_256 | Y2 ΔT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fancy | adjective | 78 | 0.2153 | 0.1984 | 0.1992 | 1.4157 | 1.4028 | 12 | 0.2044 | 0.1944 | 0.1954 | 1.4608 | 1.4450 |
| duo | ordinal-or-numeral | 78 | 0.2160 | 0.2039 | 0.2064 | 1.3973 | 1.3942 | 12 | 0.2228 | 0.2042 | 0.2070 | 1.4688 | 1.4495 |
| massive | quantity | 78 | 0.2130 | 0.1986 | 0.1997 | 1.3893 | 1.3763 | 12 | 0.2152 | 0.1888 | 0.1911 | 1.4436 | 1.4206 |
| distinct | determiner-like | 78 | 0.2038 | 0.1918 | 0.1932 | 1.3738 | 1.3632 | 12 | 0.1733 | 0.1656 | 0.1685 | 1.3463 | 1.3361 |
| modest | quantity | 78 | 0.2289 | 0.2169 | 0.2178 | 1.3507 | 1.3452 | 12 | 0.2199 | 0.2046 | 0.2063 | 1.3960 | 1.3850 |
| tens | ordinal-or-numeral | 78 | 0.2675 | 0.2569 | 0.2586 | 1.3444 | 1.3460 | 12 | 0.2516 | 0.2381 | 0.2407 | 1.3634 | 1.3585 |
| spicy | adjective | 78 | 0.2090 | 0.1970 | 0.1990 | 1.3051 | 1.2978 | 12 | 0.2165 | 0.2036 | 0.2068 | 1.3486 | 1.3459 |
| immense | quantity | 78 | 0.2010 | 0.1883 | 0.1898 | 1.3060 | 1.2957 | 12 | 0.2061 | 0.1896 | 0.1917 | 1.3474 | 1.3347 |
| sweet | adjective | 78 | 0.1918 | 0.1799 | 0.1815 | 1.2943 | 1.2916 | 12 | 0.2116 | 0.1998 | 0.2019 | 1.3648 | 1.3573 |
| bitter | adjective | 78 | 0.1902 | 0.1780 | 0.1792 | 1.2545 | 1.2452 | 12 | 0.1985 | 0.1809 | 0.1832 | 1.2977 | 1.2924 |
| raw | adjective | 78 | 0.1542 | 0.1375 | 0.1383 | 1.2321 | 1.2227 | 12 | 0.1524 | 0.1343 | 0.1366 | 1.2527 | 1.2347 |
| solo | ordinal-or-numeral | 78 | 0.1870 | 0.1711 | 0.1734 | 1.2385 | 1.2233 | 12 | 0.1731 | 0.1505 | 0.1529 | 1.2528 | 1.2283 |
| secondary | determiner-like | 78 | 0.1890 | 0.1786 | 0.1793 | 1.2150 | 1.2042 | 12 | 0.1945 | 0.1723 | 0.1733 | 1.2466 | 1.2252 |
| triplet | ordinal-or-numeral | 78 | 0.0983 | 0.0883 | 0.0923 | 1.2104 | 1.2057 | 12 | 0.0955 | 0.0778 | 0.0838 | 1.2419 | 1.2355 |
| opposite | determiner-like | 78 | 0.1225 | 0.1120 | 0.1133 | 1.1974 | 1.1850 | 12 | 0.1122 | 0.1049 | 0.1073 | 1.2049 | 1.1939 |
| principal | determiner-like | 78 | 0.1327 | 0.1193 | 0.1205 | 1.1420 | 1.1309 | 12 | 0.1122 | 0.0927 | 0.0957 | 1.1176 | 1.1068 |
| ripe | adjective | 78 | 0.3076 | 0.2931 | 0.2958 | 1.1165 | 1.1125 | 12 | 0.3103 | 0.2944 | 0.2975 | 1.1312 | 1.1212 |
| maximal | quantity | 78 | 0.1379 | 0.1266 | 0.1296 | 1.1122 | 1.1032 | 12 | 0.1161 | 0.0989 | 0.1038 | 1.0583 | 1.0469 |
| ye | possessive-or-pronoun | 78 | 0.0479 | 0.0340 | 0.0340 | 1.0885 | 1.0690 | 12 | 0.0296 | 0.0201 | 0.0204 | 1.0848 | 1.0584 |
| greater | quantity | 78 | 0.1772 | 0.1662 | 0.1670 | 1.0777 | 1.0593 | 12 | 0.1510 | 0.1380 | 0.1398 | 1.0446 | 1.0257 |
| primary | determiner-like | 78 | 0.1031 | 0.0889 | 0.0892 | 1.0716 | 1.0512 | 12 | 0.1117 | 0.0998 | 0.1008 | 1.0764 | 1.0602 |
| me | possessive-or-pronoun | 78 | 0.0141 | 0.0005 | 0.0005 | 0.8611 | 0.8291 | 12 | 0.0109 | -0.0037 | -0.0030 | 0.8578 | 0.8312 |
| him | possessive-or-pronoun | 78 | 0.0744 | 0.0610 | 0.0627 | 0.8002 | 0.7686 | 12 | 0.0684 | 0.0539 | 0.0556 | 0.8116 | 0.7702 |
| it | possessive-or-pronoun | 78 | -0.1982 | -0.2070 | -0.2057 | 0.6120 | 0.5789 | 12 | -0.2056 | -0.2130 | -0.2100 | 0.6069 | 0.5761 |

## Execution ledger

- Executed prompt keys: 2430
- Executed noun keys: 80
