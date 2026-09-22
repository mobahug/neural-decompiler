# Experiment 019 Report

- Run ID: `aa8f8607ca8c4cb5`
- Confirmation sha256: `87540d9b5dacd71cb2e87443e736822bcc6290325516aa8a65095794826db5a5`
- Experiment 018 lock sha256: `fdfae9106898f51528fc6c1a3e235bd9be7f37f3ccb00628ccf7522e308fae46`
- Protocol/code commit at explore: `2dbfdd126432ed547384ed0bcda010c88f475185`

## Phases

- `confirm`: `complete`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

> O* is a cross-validated empirical headroom witness (a leave-one-cue-out greedy fit of the scored objective), not a global oracle, ceiling, certificate or upper bound: a high H* establishes transferable selectable headroom under the frozen witness; a low H* means the witness did not establish enough headroom for the fixed-population-versus-unpredicted distinction and is never a proof that no frame-specific headroom exists.

## Tier A — exposed pool (calibration record only; the floors are frozen constants)

- Replication of Experiment 018: 11796 pairs, max deviation 0.00e+00; I9 (the E rule with Experiment 018's inputs reproduces its per-frame lists): {'n_frames': 90, 'passed': True}
- Identities (checks only): I10_read_identity 5.6e-16, I1_patched_rows 2.2e-05, I1_reference_rows 1.1e-05, I2_chain 3.4e-06, I3_head_split 9.9e-07, I4_x3 8.5e-06, I5_head_row 1.2e-05, I5_reference_head_row 8.7e-06, I6_dT 3.3e-05, I7_split 8.6e-07, I8_reference_rung 1.5e-14, I9_frame_lists 0.0e+00, head_level1_recovery 6.4e-12, level0_015_recovery 1.0e-14, level1_recovery 1.4e-14, neuron_sum 2.3e-08, p1_cross_check 6.6e-16, rho_identity 1.1e-07
- Licensed pool: 11796 pairs; selectors digest `62108946c0061d7ecc02b878d798de10d1709e01c84c6d66397c0967d32136a1`; S'_1 [1987]; S'_64 at p_c (first 10) [111, 129, 173, 228, 287, 383, 434, 511, 561, 626]; inherited S_64 (first 10) [129, 228, 287, 383, 434, 511, 561, 576, 626, 637]
- Exposed (cue-in-sample; 11796 pairs): reference rung c_L R² 1.000, rows 0.996; gaps F 0.057, Pi 0.202, c_L 0.170, dT 0.043, rows 0.054
  - k=16: κ_c_L Sp16 0.702, T16 0.673, E16 0.749, G16 0.710, L16 0.702, O16 0.749, Os16 0.903, R1_16 0.008, R2_16 0.008, R3_16 0.004; gains over S'_16: T16 -0.029, E16 0.047, G16 0.008, L16 0.000, O16 0.046, Os16 0.200, R1_16 -0.694, R2_16 -0.695, R3_16 -0.698; ρ —; E wins 58/90; κ_Π E 0.588 / S' 0.540; κ_row E 0.106
  - k=64: κ_c_L Sp64 0.754, T64 0.777, E64 0.838, G64 0.822, L64 0.748, O64 0.838, Os64 0.946, R1_64 0.384, R2_64 -0.007, R3_64 -0.015; gains over S'_64: T64 0.022, E64 0.084, G64 0.068, L64 -0.006, O64 0.083, Os64 0.191, R1_64 -0.370, R2_64 -0.762, R3_64 -0.770; ρ 0.814; E wins 66/90; κ_Π E 0.671 / S' 0.579; κ_row E 0.281
  - k=256: κ_c_L Sp256 0.875, T256 0.901, E256 0.948, G256 0.939, L256 0.878, O256 0.948, Os256 0.963, R1_256 0.227, R2_256 0.408, R3_256 0.083; gains over S'_256: T256 0.026, E256 0.073, G256 0.064, L256 0.003, O256 0.073, Os256 0.088, R1_256 -0.647, R2_256 -0.467, R3_256 -0.792; ρ 0.874; E wins 78/90; κ_Π E 0.824 / S' 0.681; κ_row E 0.508
  - membership at k=64 (mean overlap with the ranking oracle's top-64 at p_c): E 0.925, G 0.738, T 0.487, S' 0.420; witness headroom H*_64 0.191 (ranking proxy 0.083)
  - split cue_final: 7864 pairs, gap 0.197; gains at k=64 E 0.070, G 0.063, T 0.034, O* 0.155
  - split coordinated: 3932 pairs, gap 0.176; gains at k=64 E 0.146, G 0.091, T -0.033, O* 0.353
  - template cardinal: 3932 pairs, gap 0.134; gains at k=64 E 0.084, G 0.011, T -0.001
  - template coordinated-adjective: 3932 pairs, gap 0.176; gains at k=64 E 0.146, G 0.091, T -0.033
  - template quantifier: 3932 pairs, gap 0.522; gains at k=64 E 0.066, G 0.075, T 0.042

## Lock

- Candidate lock sha256 `6c0cb1849955e60bd84be34f89239221e42ce1ddb8cdb5cb00a14c18e6e48165`; predictions sha256 `1781102d748f5ab44ec0cf180917f2daca68da6589adeb3c9f7d18c2ebe15b9f`

## Confirmation — stage 1 (fresh frames' reference states; the frames' E and G lists and the prediction table digested before any fresh cue prompt)

- Table rows 432; digest `19149bd32e71875c1f934a7b85103a36b7897e6fd3f51946d3034ac831c592cd`; commit `9033041eb119194d8413d053ca11a4d45285b1f3`
  - cardinal-019-1: valid (plural head change 1.463, cue effect 79/72; p_c 3, p_t 3); E_64 ∩ S'_64 28, E_64 ∩ T_64 33, E_64 ∩ G_64 47, G_64 ∩ S'_64 25
  - cardinal-019-2: valid (plural head change 1.437, cue effect 79/72; p_c 3, p_t 3); E_64 ∩ S'_64 26, E_64 ∩ T_64 28, E_64 ∩ G_64 46, G_64 ∩ S'_64 21
  - cardinal-019-3: valid (plural head change 1.480, cue effect 79/72; p_c 3, p_t 3); E_64 ∩ S'_64 28, E_64 ∩ T_64 26, E_64 ∩ G_64 47, G_64 ∩ S'_64 27
  - cardinal-019-4: valid (plural head change 1.600, cue effect 79/72; p_c 3, p_t 3); E_64 ∩ S'_64 24, E_64 ∩ T_64 27, E_64 ∩ G_64 51, G_64 ∩ S'_64 26
  - cardinal-019-5: valid (plural head change 1.563, cue effect 79/72; p_c 3, p_t 3); E_64 ∩ S'_64 30, E_64 ∩ T_64 31, E_64 ∩ G_64 49, G_64 ∩ S'_64 26
  - cardinal-019-6: valid (plural head change 1.113, cue effect 79/72; p_c 4, p_t 4); E_64 ∩ S'_64 25, E_64 ∩ T_64 32, E_64 ∩ G_64 49, G_64 ∩ S'_64 24
  - coordinated-adjective-019-1: valid (plural head change 1.911, cue effect 79/72; p_c 7, p_t 8); E_64 ∩ S'_64 25, E_64 ∩ T_64 28, E_64 ∩ G_64 42, G_64 ∩ S'_64 21
  - coordinated-adjective-019-2: valid (plural head change 2.537, cue effect 79/72; p_c 7, p_t 8); E_64 ∩ S'_64 22, E_64 ∩ T_64 24, E_64 ∩ G_64 47, G_64 ∩ S'_64 23
  - coordinated-adjective-019-3: valid (plural head change 2.827, cue effect 79/72; p_c 5, p_t 6); E_64 ∩ S'_64 27, E_64 ∩ T_64 31, E_64 ∩ G_64 47, G_64 ∩ S'_64 23
  - coordinated-adjective-019-4: valid (plural head change 2.173, cue effect 79/72; p_c 5, p_t 6); E_64 ∩ S'_64 26, E_64 ∩ T_64 26, E_64 ∩ G_64 49, G_64 ∩ S'_64 23
  - coordinated-adjective-019-5: valid (plural head change 2.160, cue effect 79/72; p_c 5, p_t 6); E_64 ∩ S'_64 29, E_64 ∩ T_64 30, E_64 ∩ G_64 49, G_64 ∩ S'_64 32
  - coordinated-adjective-019-6: valid (plural head change 2.828, cue effect 79/72; p_c 5, p_t 6); E_64 ∩ S'_64 27, E_64 ∩ T_64 30, E_64 ∩ G_64 46, G_64 ∩ S'_64 22
  - quantifier-019-1: valid (plural head change 2.025, cue effect 79/72; p_c 4, p_t 4); E_64 ∩ S'_64 24, E_64 ∩ T_64 29, E_64 ∩ G_64 48, G_64 ∩ S'_64 25
  - quantifier-019-2: valid (plural head change 2.878, cue effect 79/72; p_c 3, p_t 3); E_64 ∩ S'_64 21, E_64 ∩ T_64 29, E_64 ∩ G_64 51, G_64 ∩ S'_64 24
  - quantifier-019-3: valid (plural head change 1.475, cue effect 79/72; p_c 4, p_t 4); E_64 ∩ S'_64 23, E_64 ∩ T_64 29, E_64 ∩ G_64 57, G_64 ∩ S'_64 22
  - quantifier-019-4: valid (plural head change 2.848, cue effect 79/72; p_c 3, p_t 3); E_64 ∩ S'_64 22, E_64 ∩ T_64 29, E_64 ∩ G_64 40, G_64 ∩ S'_64 20
  - quantifier-019-5: valid (plural head change 2.461, cue effect 79/72; p_c 3, p_t 3); E_64 ∩ S'_64 17, E_64 ∩ T_64 29, E_64 ∩ G_64 43, G_64 ∩ S'_64 17
  - quantifier-019-6: valid (plural head change 2.331, cue effect 79/72; p_c 4, p_t 4); E_64 ∩ S'_64 25, E_64 ∩ T_64 28, E_64 ∩ G_64 47, G_64 ∩ S'_64 23

## Confirmation — stage 2 — `ROUTING_PREDICTED_TOKENS | ROUTING_PREDICTED_FRAMES_CONDITIONAL | OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT | TEMPLATE_FAMILY_INSUFFICIENT | MEMBERSHIP_PREDICTED`

- Y1: scored tokens 24 (2160 pairs, 90 frames); precondition ok → **ROUTING_PREDICTED_TOKENS**
  - Δκ_64(E) 0.090 (floor 0.05); E_64 wins 64/90 (ties 0); split gains cue-final 0.070, coordinated 0.159 (floor 0.02); witness headroom H*_64 0.210 (ranking proxy H_64 0.100; E's share of H* 0.43); conditions {'frame_count': True, 'gain': True, 'split_coordinated': True, 'split_cue_final': True}
  - reference rung c_L R² 0.999, rows 0.995, ΔT 0.999; gaps pooled 0.252, cue-final 0.281, coordinated 0.315
  - k=16: κ_c_L Sp16 0.675, T16 0.633, E16 0.731, G16 0.692, L16 0.675, O16 0.738, Os16 0.906, R1_16 0.002, R2_16 0.006, R3_16 0.010; gains over S'_16: T16 -0.042, E16 0.056, G16 0.017, L16 0.000, O16 0.063, Os16 0.231, R1_16 -0.673, R2_16 -0.669, R3_16 -0.666; ρ 0.302; E wins 53/90; κ_Π E 0.571 / S' 0.513; κ_row E 0.113
  - k=64: κ_c_L Sp64 0.722, T64 0.747, E64 0.813, G64 0.800, L64 0.709, O64 0.823, Os64 0.932, R1_64 0.294, R2_64 -0.039, R3_64 -0.013; gains over S'_64: T64 0.024, E64 0.090, G64 0.078, L64 -0.013, O64 0.100, Os64 0.210, R1_64 -0.428, R2_64 -0.761, R3_64 -0.736; ρ 0.862; E wins 64/90; κ_Π E 0.659 / S' 0.558; κ_row E 0.306
  - k=256: κ_c_L Sp256 0.855, T256 0.881, E256 0.943, G256 0.935, L256 0.849, O256 0.941, Os256 0.933, R1_256 0.269, R2_256 0.315, R3_256 0.095; gains over S'_256: T256 0.026, E256 0.088, G256 0.080, L256 -0.006, O256 0.086, Os256 0.078, R1_256 -0.586, R2_256 -0.539, R3_256 -0.760; ρ 0.908; E wins 84/90; κ_Π E 0.832 / S' 0.690; κ_row E 0.511
  - jackknife of Δκ_64(E): min 0.084, max 0.104; frames where E_64 loses more than 0.05 R² to S'_64: 7 (G: 10)
  - template cardinal: 720 pairs, gap 0.202; κ S' 0.641, T 0.634, E 0.725, G 0.630, O* 0.843
  - template coordinated-adjective: 720 pairs, gap 0.315; κ S' 0.546, T 0.517, E 0.705, G 0.662, O* 0.906
  - template quantifier: 720 pairs, gap 0.865; κ S' 0.799, T 0.848, E 0.867, G 0.882, O* 0.959
  - per frame at k=64 (c_L R²: S_0 | S' | T | E | G | O | O* | S_2048; gap):
    - cardinal-009-1 (cardinal, 24 pairs): 0.919 | 0.919 | 0.934 | 0.904 | 0.866 | 0.930 | 0.973 | 0.999; gap 0.080
    - cardinal-009-2 (cardinal, 24 pairs): 0.629 | 0.822 | 0.832 | 0.942 | 0.940 | 0.911 | 0.955 | 0.999; gap 0.370
    - cardinal-011-1 (cardinal, 24 pairs): 0.092 | 0.937 | 0.907 | 0.967 | 0.972 | 0.972 | 0.973 | 0.998; gap 0.907
    - cardinal-011-2 (cardinal, 24 pairs): 0.946 | 0.967 | 0.920 | 0.960 | 0.937 | 0.945 | 0.975 | 1.000; gap 0.053
    - cardinal-012-1 (cardinal, 24 pairs): 0.681 | 0.876 | 0.911 | 0.931 | 0.894 | 0.941 | 0.900 | 0.993; gap 0.312
    - cardinal-012-2 (cardinal, 24 pairs): -0.038 | 0.699 | 0.828 | 0.686 | 0.615 | 0.733 | 0.960 | 0.999; gap 1.037
    - cardinal-013-1 (cardinal, 24 pairs): 0.742 | 0.616 | 0.824 | 0.945 | 0.898 | 0.887 | 0.965 | 0.995; gap 0.253
    - cardinal-013-2 (cardinal, 24 pairs): 0.869 | 0.819 | 0.539 | 0.720 | 0.402 | 0.674 | 0.958 | 0.996; gap 0.127
    - cardinal-014-1 (cardinal, 24 pairs): 0.736 | 0.749 | 0.801 | 0.850 | 0.873 | 0.917 | 0.831 | 0.981; gap 0.244
    - cardinal-014-2 (cardinal, 24 pairs): 0.612 | 0.896 | 0.898 | 0.844 | 0.798 | 0.905 | 0.858 | 0.998; gap 0.386
    - cardinal-015-1 (cardinal, 24 pairs): 0.549 | 0.920 | 0.817 | 0.924 | 0.941 | 0.925 | 0.896 | 0.999; gap 0.450
    - cardinal-015-2 (cardinal, 24 pairs): 0.762 | 0.923 | 0.939 | 0.947 | 0.942 | 0.958 | 0.918 | 0.998; gap 0.236
    - cardinal-016-1 (cardinal, 24 pairs): 0.656 | 0.959 | 0.950 | 0.976 | 0.976 | 0.976 | 0.984 | 0.999; gap 0.343
    - cardinal-016-2 (cardinal, 24 pairs): 0.776 | 0.921 | 0.874 | 0.832 | 0.621 | 0.839 | 0.857 | 0.998; gap 0.222
    - cardinal-016-3 (cardinal, 24 pairs): 0.904 | 0.957 | 0.969 | 0.959 | 0.971 | 0.972 | 0.969 | 1.000; gap 0.096
    - cardinal-016-4 (cardinal, 24 pairs): 0.679 | 0.907 | 0.919 | 0.923 | 0.839 | 0.918 | 0.981 | 0.999; gap 0.320
    - cardinal-017-1 (cardinal, 24 pairs): 0.951 | 0.950 | 0.927 | 0.953 | 0.927 | 0.966 | 0.968 | 0.999; gap 0.049
    - cardinal-017-2 (cardinal, 24 pairs): 0.938 | 0.958 | 0.961 | 0.964 | 0.969 | 0.954 | 0.920 | 0.996; gap 0.058
    - cardinal-017-3 (cardinal, 24 pairs): 0.852 | 0.917 | 0.896 | 0.911 | 0.957 | 0.933 | 0.991 | 0.999; gap 0.147
    - cardinal-017-4 (cardinal, 24 pairs): 0.900 | 0.937 | 0.919 | 0.932 | 0.807 | 0.945 | 0.943 | 0.996; gap 0.096
    - cardinal-018-1 (cardinal, 24 pairs): 0.630 | 0.943 | 0.961 | 0.915 | 0.959 | 0.912 | 0.981 | 0.995; gap 0.365
    - cardinal-018-2 (cardinal, 24 pairs): 0.870 | 0.895 | 0.913 | 0.942 | 0.947 | 0.949 | 0.898 | 0.988; gap 0.118
    - cardinal-018-3 (cardinal, 24 pairs): 0.959 | 0.947 | 0.955 | 0.936 | 0.967 | 0.969 | 0.957 | 0.998; gap 0.040
    - cardinal-018-4 (cardinal, 24 pairs): 0.142 | 0.456 | 0.275 | 0.637 | 0.499 | 0.706 | 0.963 | 0.999; gap 0.858
    - cardinal-1 (cardinal, 24 pairs): 0.766 | 0.828 | 0.782 | 0.878 | 0.882 | 0.904 | 0.890 | 0.982; gap 0.216
    - cardinal-2 (cardinal, 24 pairs): -0.224 | 0.819 | 0.787 | 0.870 | 0.796 | 0.809 | 0.970 | 0.999; gap 1.223
    - cardinal-fresh-1 (cardinal, 24 pairs): -0.721 | 0.883 | 0.945 | 0.961 | 0.898 | 0.938 | 0.963 | 0.998; gap 1.719
    - cardinal-fresh-2 (cardinal, 24 pairs): 0.877 | 0.929 | 0.939 | 0.939 | 0.946 | 0.928 | 0.976 | 0.997; gap 0.120
    - cardinal-new-1 (cardinal, 24 pairs): 0.800 | 0.944 | 0.906 | 0.951 | 0.934 | 0.937 | 0.981 | 0.998; gap 0.198
    - cardinal-new-2 (cardinal, 24 pairs): 0.747 | 0.910 | 0.907 | 0.891 | 0.920 | 0.911 | 0.940 | 0.996; gap 0.249
    - coordinated-adjective-009-1 (coordinated-adjective, 24 pairs): 0.796 | 0.924 | 0.941 | 0.868 | 0.853 | 0.888 | 0.951 | 0.999; gap 0.203
    - coordinated-adjective-009-2 (coordinated-adjective, 24 pairs): 0.589 | 0.870 | 0.877 | 0.849 | 0.855 | 0.862 | 0.969 | 1.000; gap 0.411
    - coordinated-adjective-011-1 (coordinated-adjective, 24 pairs): -0.099 | 0.891 | 0.866 | 0.925 | 0.911 | 0.948 | 0.965 | 0.999; gap 1.098
    - coordinated-adjective-011-2 (coordinated-adjective, 24 pairs): 0.900 | 0.772 | 0.883 | 0.955 | 0.945 | 0.927 | 0.969 | 0.999; gap 0.099
    - coordinated-adjective-012-1 (coordinated-adjective, 24 pairs): 0.674 | 0.932 | 0.936 | 0.958 | 0.949 | 0.956 | 0.966 | 0.999; gap 0.326
    - coordinated-adjective-012-2 (coordinated-adjective, 24 pairs): 0.951 | 0.897 | 0.947 | 0.947 | 0.967 | 0.951 | 0.965 | 0.999; gap 0.048
    - coordinated-adjective-013-1 (coordinated-adjective, 24 pairs): 0.687 | 0.562 | 0.460 | 0.804 | 0.610 | 0.799 | 0.924 | 0.999; gap 0.311
    - coordinated-adjective-013-2 (coordinated-adjective, 24 pairs): 0.892 | 0.923 | 0.915 | 0.951 | 0.937 | 0.930 | 0.933 | 0.998; gap 0.106
    - coordinated-adjective-014-1 (coordinated-adjective, 24 pairs): 0.512 | 0.811 | 0.728 | 0.893 | 0.848 | 0.918 | 0.921 | 0.997; gap 0.485
    - coordinated-adjective-014-2 (coordinated-adjective, 24 pairs): 0.655 | 0.717 | 0.472 | 0.793 | 0.695 | 0.735 | 0.909 | 0.996; gap 0.341
    - coordinated-adjective-015-1 (coordinated-adjective, 24 pairs): 0.631 | 0.808 | 0.742 | 0.935 | 0.842 | 0.922 | 0.965 | 0.999; gap 0.368
    - coordinated-adjective-015-2 (coordinated-adjective, 24 pairs): 0.237 | 0.878 | 0.776 | 0.875 | 0.780 | 0.896 | 0.979 | 1.000; gap 0.762
    - coordinated-adjective-016-1 (coordinated-adjective, 24 pairs): -3.655 | -0.075 | -0.080 | 0.210 | 0.240 | 0.676 | 0.920 | 0.997; gap 4.652
    - coordinated-adjective-016-2 (coordinated-adjective, 24 pairs): 0.905 | 0.869 | 0.801 | 0.895 | 0.927 | 0.934 | 0.972 | 1.000; gap 0.094
    - coordinated-adjective-016-3 (coordinated-adjective, 24 pairs): 0.885 | 0.902 | 0.926 | 0.932 | 0.917 | 0.935 | 0.968 | 0.998; gap 0.113
    - coordinated-adjective-016-4 (coordinated-adjective, 24 pairs): 0.331 | 0.917 | 0.963 | 0.938 | 0.960 | 0.948 | 0.956 | 1.000; gap 0.668
    - coordinated-adjective-017-1 (coordinated-adjective, 24 pairs): 0.234 | 0.517 | 0.502 | 0.681 | 0.837 | 0.548 | 0.972 | 0.999; gap 0.766
    - coordinated-adjective-017-2 (coordinated-adjective, 24 pairs): 0.710 | 0.389 | 0.476 | 0.578 | 0.358 | 0.697 | 0.959 | 0.999; gap 0.289
    - coordinated-adjective-017-3 (coordinated-adjective, 24 pairs): 0.940 | 0.900 | 0.947 | 0.955 | 0.964 | 0.947 | 0.959 | 0.999; gap 0.059
    - coordinated-adjective-017-4 (coordinated-adjective, 24 pairs): -0.336 | 0.859 | 0.848 | 0.873 | 0.899 | 0.866 | 0.886 | 0.999; gap 1.335
    - coordinated-adjective-018-1 (coordinated-adjective, 24 pairs): 0.406 | 0.831 | 0.917 | 0.960 | 0.920 | 0.960 | 0.838 | 0.998; gap 0.591
    - coordinated-adjective-018-2 (coordinated-adjective, 24 pairs): -0.384 | 0.558 | 0.529 | 0.693 | 0.792 | 0.817 | 0.946 | 0.998; gap 1.382
    - coordinated-adjective-018-3 (coordinated-adjective, 24 pairs): 0.451 | 0.363 | 0.314 | 0.834 | 0.684 | 0.837 | 0.957 | 0.999; gap 0.548
    - coordinated-adjective-018-4 (coordinated-adjective, 24 pairs): 0.503 | 0.952 | 0.864 | 0.908 | 0.960 | 0.933 | 0.980 | 0.999; gap 0.496
    - coordinated-adjective-1 (coordinated-adjective, 24 pairs): 0.597 | 0.679 | 0.700 | 0.834 | 0.702 | 0.742 | 0.983 | 1.000; gap 0.403
    - coordinated-adjective-2 (coordinated-adjective, 24 pairs): 0.876 | 0.868 | 0.850 | 0.830 | 0.851 | 0.955 | 0.945 | 0.999; gap 0.123
    - coordinated-adjective-fresh-1 (coordinated-adjective, 24 pairs): 0.713 | 0.900 | 0.789 | 0.910 | 0.944 | 0.946 | 0.960 | 0.998; gap 0.285
    - coordinated-adjective-fresh-2 (coordinated-adjective, 24 pairs): 0.790 | 0.928 | 0.910 | 0.917 | 0.943 | 0.914 | 0.977 | 0.998; gap 0.208
    - coordinated-adjective-new-1 (coordinated-adjective, 24 pairs): 0.365 | 0.881 | 0.912 | 0.907 | 0.881 | 0.818 | 0.971 | 0.999; gap 0.634
    - coordinated-adjective-new-2 (coordinated-adjective, 24 pairs): 0.769 | 0.601 | 0.719 | 0.798 | 0.777 | 0.866 | 0.955 | 1.000; gap 0.230
    - quantifier-009-1 (quantifier, 24 pairs): -0.522 | 0.562 | 0.733 | 0.896 | 0.856 | 0.905 | 0.938 | 0.999; gap 1.521
    - quantifier-009-2 (quantifier, 24 pairs): 0.776 | 0.814 | 0.836 | 0.866 | 0.896 | 0.891 | 0.953 | 1.000; gap 0.223
    - quantifier-011-1 (quantifier, 24 pairs): 0.801 | 0.955 | 0.963 | 0.958 | 0.955 | 0.962 | 0.947 | 1.000; gap 0.199
    - quantifier-011-2 (quantifier, 24 pairs): 0.620 | 0.956 | 0.925 | 0.946 | 0.971 | 0.946 | 0.967 | 1.000; gap 0.380
    - quantifier-012-1 (quantifier, 24 pairs): 0.421 | 0.753 | 0.819 | 0.969 | 0.951 | 0.954 | 0.943 | 1.000; gap 0.579
    - quantifier-012-2 (quantifier, 24 pairs): -1.078 | 0.919 | 0.909 | 0.804 | 0.927 | 0.753 | 0.924 | 0.999; gap 2.077
    - quantifier-013-1 (quantifier, 24 pairs): 0.712 | 0.954 | 0.945 | 0.968 | 0.978 | 0.970 | 0.980 | 0.999; gap 0.286
    - quantifier-013-2 (quantifier, 24 pairs): 0.757 | 0.906 | 0.921 | 0.877 | 0.873 | 0.761 | 0.938 | 0.999; gap 0.243
    - quantifier-014-1 (quantifier, 24 pairs): -1.082 | 0.794 | 0.665 | 0.916 | 0.905 | 0.923 | 0.912 | 0.999; gap 2.081
    - quantifier-014-2 (quantifier, 24 pairs): -5.938 | 0.233 | 0.661 | 0.750 | 0.797 | 0.625 | 0.956 | 0.998; gap 6.937
    - quantifier-015-1 (quantifier, 24 pairs): 0.837 | 0.914 | 0.919 | 0.943 | 0.952 | 0.933 | 0.984 | 1.000; gap 0.163
    - quantifier-015-2 (quantifier, 24 pairs): -2.152 | 0.031 | 0.203 | -0.162 | 0.021 | -0.168 | 0.940 | 0.996; gap 3.147
    - quantifier-016-1 (quantifier, 24 pairs): 0.891 | 0.960 | 0.958 | 0.950 | 0.955 | 0.961 | 0.965 | 0.998; gap 0.107
    - quantifier-016-2 (quantifier, 24 pairs): -2.885 | 0.308 | 0.530 | 0.773 | 0.793 | 0.807 | 0.949 | 0.999; gap 3.884
    - quantifier-016-3 (quantifier, 24 pairs): -0.242 | 0.545 | 0.711 | 0.630 | 0.555 | 0.690 | 0.955 | 0.999; gap 1.241
    - quantifier-016-4 (quantifier, 24 pairs): -0.172 | 0.856 | 0.837 | 0.937 | 0.920 | 0.937 | 0.982 | 1.000; gap 1.172
    - quantifier-017-1 (quantifier, 24 pairs): 0.704 | 0.909 | 0.926 | 0.957 | 0.955 | 0.955 | 0.961 | 0.999; gap 0.296
    - quantifier-017-2 (quantifier, 24 pairs): 0.921 | 0.942 | 0.955 | 0.971 | 0.974 | 0.982 | 0.964 | 1.000; gap 0.079
    - quantifier-017-3 (quantifier, 24 pairs): 0.888 | 0.858 | 0.914 | 0.933 | 0.943 | 0.951 | 0.932 | 0.999; gap 0.111
    - quantifier-017-4 (quantifier, 24 pairs): -1.343 | 0.520 | 0.695 | 0.658 | 0.838 | 0.711 | 0.954 | 1.000; gap 2.343
    - quantifier-018-1 (quantifier, 24 pairs): 0.499 | 0.965 | 0.968 | 0.950 | 0.962 | 0.953 | 0.978 | 1.000; gap 0.500
    - quantifier-018-2 (quantifier, 24 pairs): 0.239 | 0.948 | 0.940 | 0.923 | 0.919 | 0.968 | 0.982 | 1.000; gap 0.761
    - quantifier-018-3 (quantifier, 24 pairs): -5.453 | -0.431 | 0.295 | 0.652 | 0.836 | 0.477 | 0.876 | 1.000; gap 6.453
    - quantifier-018-4 (quantifier, 24 pairs): 0.778 | 0.873 | 0.893 | 0.911 | 0.813 | 0.927 | 0.925 | 1.000; gap 0.221
    - quantifier-1 (quantifier, 24 pairs): 0.588 | 0.910 | 0.876 | 0.908 | 0.924 | 0.923 | 0.942 | 1.000; gap 0.412
    - quantifier-2 (quantifier, 24 pairs): -0.296 | 0.791 | 0.693 | 0.932 | 0.944 | 0.921 | 0.975 | 0.999; gap 1.295
    - quantifier-fresh-1 (quantifier, 24 pairs): 0.530 | 0.871 | 0.887 | 0.905 | 0.917 | 0.920 | 0.934 | 0.999; gap 0.469
    - quantifier-fresh-2 (quantifier, 24 pairs): -1.226 | 0.686 | 0.878 | 0.877 | 0.810 | 0.906 | 0.947 | 0.999; gap 2.225
    - quantifier-new-1 (quantifier, 24 pairs): -0.389 | 0.887 | 0.849 | 0.933 | 0.922 | 0.902 | 0.907 | 0.999; gap 1.388
    - quantifier-new-2 (quantifier, 24 pairs): 0.771 | 0.926 | 0.933 | 0.805 | 0.908 | 0.898 | 0.938 | 1.000; gap 0.229
  - membership at k=64 over 90 frames: E 0.867, G 0.723, T 0.472, S' 0.410
- Y2: scored tokens 24 (432 pairs, 18 frames); precondition ok → **ROUTING_PREDICTED_FRAMES_CONDITIONAL**
  - Δκ_64(E) 0.144 (floor 0.05); E_64 wins 14/18 (ties 0); split gains cue-final 0.147, coordinated 0.133 (floor 0.02); witness headroom H*_64 0.241 (ranking proxy H_64 0.143; E's share of H* 0.60); conditions {'frame_count': True, 'gain': True, 'split_coordinated': True, 'split_cue_final': True}
  - reference rung c_L R² 1.000, rows 0.995, ΔT 0.999; gaps pooled 0.198, cue-final 0.232, coordinated 0.255
  - k=16: κ_c_L Sp16 0.679, T16 0.679, E16 0.741, G16 0.664, L16 0.679, O16 0.726, Os16 0.870, R1_16 0.025, R2_16 0.010, R3_16 0.005; gains over S'_16: T16 0.000, E16 0.062, G16 -0.015, L16 0.000, O16 0.047, Os16 0.191, R1_16 -0.653, R2_16 -0.669, R3_16 -0.673; ρ -0.240; E wins 13/18; κ_Π E 0.535 / S' 0.458; κ_row E 0.186
  - k=64: κ_c_L Sp64 0.649, T64 0.713, E64 0.793, G64 0.813, L64 0.650, O64 0.793, Os64 0.890, R1_64 0.272, R2_64 -0.107, R3_64 0.020; gains over S'_64: T64 0.064, E64 0.144, G64 0.164, L64 0.001, O64 0.143, Os64 0.241, R1_64 -0.377, R2_64 -0.757, R3_64 -0.629; ρ 1.136; E wins 14/18; κ_Π E 0.595 / S' 0.411; κ_row E 0.341
  - k=256: κ_c_L Sp256 0.836, T256 0.846, E256 0.918, G256 0.917, L256 0.810, O256 0.928, Os256 0.889, R1_256 0.413, R2_256 0.177, R3_256 0.170; gains over S'_256: T256 0.011, E256 0.082, G256 0.081, L256 -0.025, O256 0.092, Os256 0.053, R1_256 -0.423, R2_256 -0.658, R3_256 -0.665; ρ 0.986; E wins 15/18; κ_Π E 0.800 / S' 0.591; κ_row E 0.548
  - jackknife of Δκ_64(E): min 0.113, max 0.169; frames where E_64 loses more than 0.05 R² to S'_64: 2 (G: 0)
  - template cardinal: 144 pairs, gap 0.256; κ S' 0.653, T 0.731, E 0.818, G 0.878, O* 0.908
  - template coordinated-adjective: 144 pairs, gap 0.255; κ S' 0.569, T 0.493, E 0.701, G 0.770, O* 0.819
  - template quantifier: 144 pairs, gap 0.847; κ S' 0.677, T 0.784, E 0.815, G 0.800, O* 0.907
  - per frame at k=64 (c_L R²: S_0 | S' | T | E | G | O | O* | S_2048; gap):
    - cardinal-019-1 (cardinal, 24 pairs): 0.144 | 0.954 | 0.930 | 0.942 | 0.977 | 0.961 | 0.981 | 0.999; gap 0.855
    - cardinal-019-2 (cardinal, 24 pairs): 0.883 | 0.924 | 0.938 | 0.957 | 0.971 | 0.967 | 0.936 | 0.999; gap 0.115
    - cardinal-019-3 (cardinal, 24 pairs): 0.854 | 0.897 | 0.926 | 0.936 | 0.960 | 0.958 | 0.974 | 1.000; gap 0.146
    - cardinal-019-4 (cardinal, 24 pairs): 0.111 | 0.868 | 0.890 | 0.900 | 0.922 | 0.939 | 0.975 | 0.999; gap 0.888
    - cardinal-019-5 (cardinal, 24 pairs): 0.928 | 0.647 | 0.789 | 0.929 | 0.937 | 0.937 | 0.944 | 0.997; gap 0.069
    - cardinal-019-6 (cardinal, 24 pairs): 0.697 | 0.888 | 0.884 | 0.890 | 0.929 | 0.890 | 0.957 | 0.995; gap 0.297
    - coordinated-adjective-019-1 (coordinated-adjective, 24 pairs): 0.471 | 0.921 | 0.972 | 0.974 | 0.895 | 0.953 | 0.965 | 0.999; gap 0.528
    - coordinated-adjective-019-2 (coordinated-adjective, 24 pairs): 0.898 | 0.933 | 0.961 | 0.971 | 0.977 | 0.961 | 0.961 | 0.999; gap 0.101
    - coordinated-adjective-019-3 (coordinated-adjective, 24 pairs): 0.707 | 0.679 | 0.689 | 0.962 | 0.944 | 0.952 | 0.959 | 0.998; gap 0.291
    - coordinated-adjective-019-4 (coordinated-adjective, 24 pairs): 0.739 | 0.909 | 0.828 | 0.832 | 0.883 | 0.883 | 0.884 | 1.000; gap 0.260
    - coordinated-adjective-019-5 (coordinated-adjective, 24 pairs): -0.051 | 0.702 | 0.474 | 0.895 | 0.905 | 0.933 | 0.889 | 0.999; gap 1.050
    - coordinated-adjective-019-6 (coordinated-adjective, 24 pairs): 0.953 | 0.950 | 0.945 | 0.793 | 0.922 | 0.788 | 0.961 | 0.999; gap 0.046
    - quantifier-019-1 (quantifier, 24 pairs): -3.265 | 0.351 | 0.659 | 0.722 | 0.707 | 0.553 | 0.903 | 0.999; gap 4.264
    - quantifier-019-2 (quantifier, 24 pairs): 0.552 | 0.739 | 0.787 | 0.793 | 0.769 | 0.783 | 0.948 | 0.999; gap 0.447
    - quantifier-019-3 (quantifier, 24 pairs): 0.616 | 0.904 | 0.897 | 0.898 | 0.872 | 0.910 | 0.949 | 1.000; gap 0.383
    - quantifier-019-4 (quantifier, 24 pairs): 0.204 | 0.511 | 0.731 | 0.878 | 0.768 | 0.905 | 0.918 | 0.999; gap 0.795
    - quantifier-019-5 (quantifier, 24 pairs): 0.479 | 0.740 | 0.893 | 0.809 | 0.948 | 0.838 | 0.921 | 0.998; gap 0.519
    - quantifier-019-6 (quantifier, 24 pairs): 0.800 | 0.858 | 0.822 | 0.862 | 0.846 | 0.798 | 0.830 | 0.999; gap 0.200
  - membership at k=64 over 18 frames: E 0.867, G 0.707, T 0.435, S' 0.375
- Y3 (the operating-point-plus-drive rule against the full evaluation; ρ_64 per set and pooled): **OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT**; ρ {'Y1': '0.862', 'Y2': '1.136', 'pooled': '0.913'}; denominators Δκ_64(E) {'Y1': '0.090', 'Y2': '0.144', 'pooled': '0.097'}
- Y4 (the template-family account; A = κ(E_64) − κ(T_64), B = Δκ_64(T), on Y2 and pooled): **TEMPLATE_FAMILY_INSUFFICIENT** — A ≥ margin pooled and on the new frames; values {'Y1': {'A': '0.066', 'B': '0.024'}, 'Y2': {'A': '0.080', 'B': '0.064'}, 'pooled': {'A': '0.068', 'B': '0.029'}}
- Y5 (membership per set: mean |E_64 ∩ O_64| / 64 ≥ 0.6 and ≥ S' + 0.2): **MEMBERSHIP_PREDICTED**; Y1: E 0.867, S' 0.410, G 0.723, T 0.472; Y2: E 0.867, S' 0.375, G 0.707, T 0.435
- Pooled (both sets, 2592 pairs): κ_c_L at k=64 S' 0.713, T 0.742, E 0.810, G 0.802, O* 0.927; ρ 0.913; headroom H*_64 0.213
- Descriptive expectations on Y1: (i) other sizes {'16': ('0.056', True), '256': ('0.088', True)}; (ii) gain over the inherited S_64 0.103 vs over S' 0.090 holds True; (iii) random at k=64 best 0.294 below 0.5 True, margin holds True; (iv) κ_c_L ≥ κ_Π True, row diffuse True; (v) G's gain per template {'cardinal': '-0.012', 'coordinated-adjective': '0.115', 'quantifier': '0.083'}; (vi) S'_1 0.281, E_1 0.373; (x) witness κ per k {'16': '0.906', '256': '0.933', '64': '0.932'}, E's share of H*_64 0.43
- Descriptive expectations on Y2: (i) other sizes {'16': ('0.062', True), '256': ('0.082', True)}; (ii) gain over the inherited S_64 0.143 vs over S' 0.144 holds False; (iii) random at k=64 best 0.272 below 0.5 True, margin holds True; (iv) κ_c_L ≥ κ_Π True, row diffuse True; (v) G's gain per template {'cardinal': '0.226', 'coordinated-adjective': '0.201', 'quantifier': '0.123'}; (vi) S'_1 0.231, E_1 0.256; (x) witness κ per k {'16': '0.870', '256': '0.889', '64': '0.890'}, E's share of H*_64 0.60
- Descriptive expectations on pooled: (i) other sizes {'16': ('0.057', True), '256': ('0.087', True)}; (ii) gain over the inherited S_64 0.108 vs over S' 0.097 holds True; (iii) random at k=64 best 0.292 below 0.5 True, margin holds True; (iv) κ_c_L ≥ κ_Π True, row diffuse True; (v) G's gain per template {'cardinal': '0.038', 'coordinated-adjective': '0.125', 'quantifier': '0.087'}; (vi) S'_1 0.275, E_1 0.359; (x) witness κ per k {'16': '0.902', '256': '0.927', '64': '0.927'}, E's share of H*_64 0.45

| token | class | Y1 frames | Y1 ĉ_L E_64 | Y1 ĉ_L S'_64 | Y1 ĉ_L S_2048 | Y1 c_L | Y2 frames | Y2 ĉ_L E_64 | Y2 ĉ_L S'_64 | Y2 ĉ_L S_2048 | Y2 c_L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| couple | ordinal-or-numeral | 90 | 0.0733 | 0.0790 | 0.0665 | 0.0665 | 18 | 0.0661 | 0.0799 | 0.0627 | 0.0628 |
| different | determiner-like | 90 | 0.2648 | 0.2726 | 0.2571 | 0.2584 | 18 | 0.2701 | 0.2810 | 0.2746 | 0.2757 |
| generous | quantity | 90 | 0.1531 | 0.1622 | 0.1379 | 0.1393 | 18 | 0.1627 | 0.1748 | 0.1490 | 0.1492 |
| pair | ordinal-or-numeral | 90 | 0.1504 | 0.1564 | 0.1457 | 0.1465 | 18 | 0.1611 | 0.1742 | 0.1610 | 0.1612 |
| stolen | adjective | 90 | 0.1852 | 0.1935 | 0.1662 | 0.1681 | 18 | 0.2108 | 0.2199 | 0.1688 | 0.1704 |
| separate | determiner-like | 90 | 0.0525 | 0.0605 | 0.0431 | 0.0438 | 18 | 0.0673 | 0.0804 | 0.0577 | 0.0582 |
| gentle | adjective | 90 | 0.2178 | 0.2286 | 0.2057 | 0.2084 | 18 | 0.2294 | 0.2408 | 0.2121 | 0.2136 |
| giant | adjective | 90 | 0.1567 | 0.1675 | 0.1520 | 0.1535 | 18 | 0.1696 | 0.1820 | 0.1556 | 0.1564 |
| remaining | determiner-like | 90 | 0.1280 | 0.1435 | 0.1123 | 0.1133 | 18 | 0.1324 | 0.1525 | 0.1073 | 0.1078 |
| faded | adjective | 90 | 0.2105 | 0.2190 | 0.1938 | 0.1940 | 18 | 0.2215 | 0.2333 | 0.1981 | 0.1969 |
| similar | determiner-like | 90 | 0.2842 | 0.2955 | 0.2771 | 0.2783 | 18 | 0.2970 | 0.3094 | 0.2995 | 0.2998 |
| negligible | quantity | 90 | 0.1720 | 0.1836 | 0.1583 | 0.1600 | 18 | 0.1839 | 0.2029 | 0.1613 | 0.1618 |
| rotten | adjective | 90 | 0.1677 | 0.1750 | 0.1585 | 0.1603 | 18 | 0.1895 | 0.1977 | 0.1691 | 0.1693 |
| triple | ordinal-or-numeral | 90 | 0.1620 | 0.1683 | 0.1620 | 0.1646 | 18 | 0.1548 | 0.1684 | 0.1508 | 0.1521 |
| prior | determiner-like | 90 | 0.0234 | 0.0318 | 0.0107 | 0.0104 | 18 | 0.0202 | 0.0346 | 0.0079 | 0.0069 |
| silver | adjective | 90 | 0.1625 | 0.1717 | 0.1515 | 0.1525 | 18 | 0.1696 | 0.1804 | 0.1512 | 0.1515 |
| they | possessive-or-pronoun | 90 | 0.0478 | 0.0603 | 0.0384 | 0.0397 | 18 | 0.0837 | 0.0951 | 0.0708 | 0.0716 |
| we | possessive-or-pronoun | 90 | 0.0713 | 0.0823 | 0.0580 | 0.0588 | 18 | 0.0888 | 0.0996 | 0.0675 | 0.0683 |
| double | ordinal-or-numeral | 90 | 0.1857 | 0.1925 | 0.1796 | 0.1802 | 18 | 0.1878 | 0.2000 | 0.1730 | 0.1725 |
| maximum | quantity | 90 | 0.1286 | 0.1363 | 0.1198 | 0.1194 | 18 | 0.1261 | 0.1411 | 0.1119 | 0.1105 |
| minimum | quantity | 90 | 0.1371 | 0.1438 | 0.1254 | 0.1265 | 18 | 0.1430 | 0.1537 | 0.1207 | 0.1205 |
| overall | quantity | 90 | 0.0376 | 0.0462 | 0.0212 | 0.0218 | 18 | 0.0403 | 0.0538 | 0.0153 | 0.0155 |
| he | possessive-or-pronoun | 90 | -0.0034 | 0.0086 | -0.0142 | -0.0129 | 18 | 0.0156 | 0.0260 | -0.0005 | 0.0006 |
| she | possessive-or-pronoun | 90 | -0.0348 | -0.0231 | -0.0429 | -0.0411 | 18 | -0.0059 | 0.0023 | -0.0207 | -0.0194 |

## Execution ledger

- Executed prompt keys: 2916
- Executed noun keys: 80
