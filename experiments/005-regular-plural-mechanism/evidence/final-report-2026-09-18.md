# Experiment 005 Report

- Run ID: `050405f9f4f7e629`
- Manifest sha256: `1c50c2e8bbf95f9aa4331ac57899f77c2a6be6b17d4cb2a5ac4ecc332c5e703c`
- Extension sha256: `1c6852547f1c2d8ecfe8a736599ba3670fac51e43c93f04057091c56ba179da5`
- Protocol/code commit at discover: `83d1ae435e05451666b1ca955d1939aa740789a1`
- Model: `EleutherAI/pythia-70m-deduped` @ `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`

## Phases

- `calibrate`: `complete`
- `confirm`: `complete`
- `continue`: `complete`
- `discover`: `complete`
- `lock`: `complete`
- `report`: `not_started`
- `revise`: `not_started`

- A0 contract test: passed
- A1 baseline: 240 cases compared to the screen (max gap 0.00e+00); development mean d_full 5.231; prompt-level gap 4.52e-03
- Hypothesis tree: `H1` from q1=0.816, q2=0.865, q3=0.987, q4=0.606, q5=0.953 (top head `L03.H04`)
- Encoding branch: `L00.MLP` with E = ['L00.MLP'] (E share 0.535, embedding share 0.009)

### A2 rankings

- p_t (all templates): L00.MLP, L03.H04, L05.MLP, L04.MLP, L02.MLP, L05.H07, L00.H05, L01.MLP
- p_t (coordinated): L03.H04, L04.MLP, L05.MLP, L05.H07, L01.MLP, L00.H05, L01.H05, L01.H03
- p_c (coordinated): L00.MLP, L02.MLP, L04.MLP, L00.H07, L03.H04, L00.H06, L01.H00, L01.H05

### A3 residual profile at p_c (coordinated)

- L0: recovery 1.000
- L1: recovery 0.965
- L2: recovery 0.938
- L3: recovery 0.956
- L4: recovery 0.125
- L5: recovery 0.125

### A4 attention to the cue from p_t

- L05.H07: 0.999
- L02.H07: 0.893
- L00.H05: 0.829
- L03.H04: 0.747
- L05.H02: 0.534
- L01.H05: 0.519

### A6 chain and path

- E alone: recovery 0.885, m_T 0.900, m_R 0.987
- residual patch at p_c: unfrozen 0.956; blocked fractions L00.H05=0.000, L03.H04=0.865, L05.H07=0.128, top3=0.986
- T alone: L00.H05=0.028, L03.H04=0.816, L05.H07=0.122; cumulative top1=0.816, top2=0.935, top3=0.953
- R alone: 0.606

## Mechanism version M1 — rejected

- Outcome: NO_COMPACT_MECHANISM 

## Mechanism version M2 — superseded (protocol v2 continuation, PROGRAM_CAPPED)

```text
VARIABLES
  n_c(x)  = ( E_program(cue token) − μ_E ) · d̂_E / σ_E          σ_E = 2.4910; E_program = L00.MLP
  n_t(x)  = k_T · n_c(x) when p_c ≠ p_t (k_T = 0.9624); n_t(x) = n_c(x) otherwise
  c(x)    = log P(singular | x) − log P(plural | x)

STAGES
  E  [L00.MLP] at p_c : cue token → n_c   (encoding; branch L00.MLP; share of the transport-input axis 0.535)
  T  [L03.H04] at p_t : n_t := k_T · n_c  (transport; hypothesis H1)
  R  [L05.MLP] at p_t : δ(x) = n_t(x) · v_T with g_R = |v_T|: cardinal 4.816, quantifier 7.177, coordinated-adjective 3.144
  D  direct paths: reported under residual accounting

READOUT
  ĉ(x)      = −u_N · LN( ρ + δ(x) )   exact final LayerNorm, ρ = frozen frame/template context
  d̂_full(N) = ĉ(x_A) − ĉ(x_B)         predicted positive for every pair

PROGRAM  CAPPED — a contextual encoding component is required; the decompilation axis cannot pass
```

- Development recovery 0.971; isolation faithfulness 0.513; program floors FAILED; program capped: True

## Mechanism version M3 — candidate (protocol v2 continuation, PROGRAM_CAPPED)

```text
VARIABLES
  n_c(x)  = ( E_program(cue token) − μ_E ) · d̂_E / σ_E          σ_E = 2.4910; E_program = L00.MLP
  n_t(x)  = k_T · n_c(x) when p_c ≠ p_t (k_T = 0.9624); n_t(x) = n_c(x) otherwise
  c(x)    = log P(singular | x) − log P(plural | x)

STAGES
  E  [L00.MLP] at p_c : cue token → n_c   (encoding; branch L00.MLP; share of the transport-input axis 0.535)
  T  [L03.H04] at p_t : n_t := k_T · n_c  (transport; hypothesis H1)
  R  [L05.MLP, L04.MLP] at p_t : δ(x) = n_t(x) · v_T with g_R = |v_T|: cardinal 5.158, quantifier 7.900, coordinated-adjective 3.800
  D  direct paths: reported under residual accounting

READOUT
  ĉ(x)      = −u_N · LN( ρ + δ(x) )   exact final LayerNorm, ρ = frozen frame/template context
  d̂_full(N) = ĉ(x_A) − ĉ(x_B)         predicted positive for every pair

PROGRAM  CAPPED — protocol v1 program floor failed (recorded); continuation adopted under design revision 5; recorded quantities: cardinal gap 0.435, coordinated-adjective gap 0.859, quantifier gap 1.437; the decompilation axis cannot pass
```

- Development recovery 0.982; isolation faithfulness 0.783; program floors FAILED; program capped: True

## Calibration pass 1 — M3 on 114 holdout cases

- Floors: passed; program residual RMSE_B 0.6088 (τ = 1.826)

| Family | Value | Floor | Band | Hit |
|---|---|---|---|---|
| B1 | flips=95; n=114; primary_correct=103 | pass |  | — |
| B2 | positive_pairs=114 | pass | cardinal=[4.397, 5.397]; coordinated-adjective=[4.833, 5.833]; quantifier=[4.728, 5.728] | — |
| P1 | recovery=0.977 | pass | overall=[0.877, 1.077]; rule:consonant-y=[0.874, 1.074]; rule:sibilant-es=[0.865, 1.065] | — |
| P2 | recovery=0.977; reference=0.050 | pass | median=[-0.092, 0.108] | — |
| P3 | faithfulness=0.782 | pass | overall=[0.682, 0.882]; retention_rate=[0.698, 0.898]; template:cardinal=[0.861, 1.061] | — |
| P4 | value=0.882 | pass | value=[0.782, 0.982] | — |
| P5 | a_declared=0.816; a_single=0.816; b_blocked_fraction=0.865 | pass | a_declared=[0.716, 0.916]; a_single=[0.716, 0.916]; b_blocked_fraction=[0.765, 0.965] | — |
| P6 | opposite=0.854; same=0.070 | pass | opposite=[0.754, 0.954]; same=[-0.030, 0.170] | — |
| P7 | opposite=0.976; same=-0.006 | pass | opposite=[0.876, 1.076]; same=[-0.106, 0.094] | — |
| P8 | compensation_ratio=-5.742; loss=0.823 | pass | compensation_ratio=[-5.842, -5.642]; loss=[0.723, 0.923] | — |
| P9 | m_R=0.987; m_R_given_T_frozen=0.074; m_T=0.900 | pass | m_R=[0.887, 1.087]; m_R_given_T_frozen=[-0.026, 0.174]; m_T=[0.800, 1.000] | — |
| S1 (secondary) | minimum=0.292 | FAIL |  | — |
| S3 (secondary) | n_t_matches=19; n_wrong=19 | pass |  | — |

## Lock

- Candidate lock sha256 `ec2a8879bab35fd6e4d45484321595aed565fb40f19bf608daf2574ea50ccc97` for M3

## Confirmation — outcome `BEHAVIOR_NOT_REPLICATED`

- Axes: circuit `CIRCUIT_FAIL`, decompilation `PROGRAM_FAIL`, new frames `NOT_GENERALIZED`
- Circuit failures: ['P3']; program failures: ['X2', 'X3', 'X4']; missed bands: ['B2:cardinal', 'B2:coordinated-adjective', 'B2:quantifier', 'P3:retention_rate']; contested: none

### Reserve nouns (readout generalization)

| Family | Value | Floor | Band | Hit |
|---|---|---|---|---|
| B1 | flips=75; n=120; primary_correct=98 | FAIL |  | — |
| B2 | positive_pairs=120 | pass |  | cardinal=no; coordinated-adjective=no; quantifier=no |
| P1 | recovery=0.985 | pass |  | overall=yes; rule:consonant-y=yes; rule:sibilant-es=yes |
| P2 | recovery=0.985; reference=0.050 | pass |  | median=yes |
| P3 | faithfulness=0.795 | FAIL |  | overall=yes; retention_rate=no; template:cardinal=yes |
| P4 | value=0.873 | pass |  | value=yes |
| P5 | a_declared=0.817; a_single=0.817; b_blocked_fraction=0.869 | pass |  | a_declared=yes; a_single=yes; b_blocked_fraction=yes |
| P6 | opposite=0.840; same=0.074 | pass |  | opposite=yes; same=yes |
| P7 | opposite=0.984; same=-0.007 | pass |  | opposite=yes; same=yes |
| P8 | compensation_ratio=-6.038; loss=0.826 | pass |  | compensation_ratio=no; loss=yes |
| P9 | m_R=0.987; m_R_given_T_frozen=0.074; m_T=0.900 | pass |  | m_R=yes; m_R_given_T_frozen=yes; m_T=yes |
| S1 (secondary) | minimum=0.292 | FAIL |  | — |
| S3 (secondary) | n_t_matches=45; n_wrong=45 | pass |  | — |

### Extension prompts (prompt-side generalization)

- X1 frame invariance of behavior: pass (positive pairs 120/108 required; variables {'n_c': 12, 'n_t': 12, 'prompts': 12})
- X2 frame invariance of the circuit: FAIL (P1=pass, P3=FAIL, P4=pass, P5=pass, P8=pass, P9=pass)
- X3 cue lexicon: FAIL (Spearman 0.699, 5 confident words, signs False, ambiguous False)
- X4 E-patch prediction: FAIL (Spearman 0.692, MAE by template cardinal=1.200, coordinated-adjective=1.573, quantifier=1.616)
- X band hits: X3: 5/12 words, X4: 5/12 words

### Cue words

| Word | n_c | mean shift | E-patch measured | E-patch predicted |
|---|---|---|---|---|
| a | 0.055 | -0.209 | -0.119 | -1.980 |
| all | -0.086 | -4.240 | -4.099 | -1.719 |
| both | 0.295 | -4.121 | -4.005 | -2.423 |
| every | -0.607 | 0.033 | 0.383 | -0.743 |
| few | 0.477 | -4.816 | -4.763 | -2.756 |
| five | 0.802 | -4.588 | -4.464 | -3.349 |
| four | 0.941 | -4.467 | -4.453 | -3.600 |
| many | 0.353 | -4.630 | -4.596 | -2.530 |
| some | 0.070 | -3.820 | -3.568 | -2.008 |
| ten | 0.535 | -4.551 | -4.519 | -2.862 |
| the | 0.118 | -2.572 | -2.386 | -2.096 |
| three | 1.002 | -4.509 | -4.456 | -3.709 |

### Residual accounting (Q1)

- Unexplained sufficiency residual 1 − R: 0.015
- Unexplained isolation residual 1 − F: 0.205
- Program residual on reserve conditions: RMSE 0.5444, MAE 0.4254
- Program residual on new frames: RMSE 0.8547, MAE 0.6500

## Execution ledger

- Executed prompt keys: 96
- Executed noun keys: 60
- Invalidated runs: 0
