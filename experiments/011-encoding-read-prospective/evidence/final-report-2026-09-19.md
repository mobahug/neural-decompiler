# Experiment 011 Report

- Run ID: `806e4fc046fc0a18`
- Confirmation sha256: `174cf22214503836d5da3fc464a5f36fbb7cd9794ab74b00de0c793f12979358`
- Protocol/code commit at explore: `69c7a142a30bc3362077186fbd6607182d549140`

## Phases

- `confirm`: `complete`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

## Tier A — exposed pool (calibration only)

- Replication against Experiment 010: 1512 pairs, max deviation 0.00e+00
- Template denominators r(E(pl) − E(ref)): {'cardinal': 1.629910036913704, 'coordinated-adjective': 1.629910036913704, 'quantifier': 1.4960738062126115}; σ_r 0.793; defined ['cardinal', 'quantifier', 'coordinated-adjective']
- τ_g 0.248 (from 63 token-mean residuals); τ_M 0.313 (from 1488 pairs)
- Exposed (descriptive): ḡ_E vs q̄_T Spearman 0.925, MAE 0.064; P1 pair MAE 0.073

## Lock

- Candidate lock sha256 `769bfeacd7c49fc18bed2ff3c5cf8d5ea2be4231e9f8e9c493819b5d7ba69d0b`; predictions sha256 `6ce4cd0c1bf8c0bc7b312972513499e97c3bf04cfa455a98dc9448421042d033`

## Confirmation — `ENCODING_READ_PREDICTS_TRANSPORT | HEAD_P1_REPLICATED`

- Valid frames 6/6 (min 4); scored tokens 24 (min 16)
  - cardinal-011-1: valid (plural head change 1.808, cue effect 79/72)
  - cardinal-011-2: valid (plural head change 1.084, cue effect 79/72)
  - coordinated-adjective-011-1: valid (plural head change 1.844, cue effect 79/72)
  - coordinated-adjective-011-2: valid (plural head change 2.310, cue effect 79/72)
  - quantifier-011-1: valid (plural head change 2.204, cue effect 79/72)
  - quantifier-011-2: valid (plural head change 1.878, cue effect 79/72)
- Y1 (ḡ_E vs q̄_T over 24 scored tokens): Spearman 0.848, MAE 0.077 (τ_g 0.248) → pass 
- Y2 (P1 vs q_T over 142 pairs): Spearman 0.974, MAE 0.076 (τ_M 0.313) → pass
- Baseline (Experiment 009 rank-1 rule, reported only): Spearman 0.816, MAE 0.091
- Descriptive: tokens with locked ḡ_E ≤ 0.35 ['an', 'mine'] with f_⊥ < 0 in frames {'an': 4, 'mine': 4}; mean net layer change 0.177, mean gross 0.560 (Experiment 010: median gross layer change 0.44 over the 63 exposed tokens; no relay claim is made without this baseline)

| token | class | scored | valid frames | predicted ḡ_E | measured q̄_T | g_∥ | g_⊥ | f_E | f_⊥ | net L1–2 | P1 | 009 rule | contrast | behavior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| an | determiner-like | True | 4 | 0.178 | 0.303 | 0.538 | -0.360 | 0.145 | -0.299 | 0.113 | 0.257 | 0.570 | -3.015 | -2.895 |
| mine | possessive-or-pronoun | True | 6 | 0.346 | 0.447 | 0.324 | 0.021 | 0.299 | 0.015 | 0.147 | 0.445 | 0.475 | -3.518 | -3.844 |
| much | determiner-like | True | 6 | 0.415 | 0.588 | 0.456 | -0.041 | 0.359 | -0.039 | 0.218 | 0.578 | 0.577 | -3.236 | -3.021 |
| such | determiner-like | True | 6 | 0.433 | 0.596 | 0.409 | 0.024 | 0.375 | 0.018 | 0.234 | 0.609 | 0.483 | -3.961 | -4.736 |
| yours | possessive-or-pronoun | True | 6 | 0.444 | 0.334 | 0.313 | 0.131 | 0.385 | 0.111 | 0.026 | 0.410 | 0.466 | -3.253 | -3.442 |
| sufficient | quantity | True | 6 | 0.445 | 0.507 | 0.558 | -0.114 | 0.385 | -0.102 | 0.109 | 0.494 | 0.748 | -3.157 | -3.522 |
| its | possessive-or-pronoun | True | 6 | 0.484 | 0.402 | 0.520 | -0.035 | 0.420 | -0.034 | 0.018 | 0.438 | 0.591 | -2.581 | -2.774 |
| ample | quantity | True | 6 | 0.512 | 0.597 | 0.632 | -0.119 | 0.444 | -0.107 | 0.153 | 0.598 | 0.842 | -3.381 | -3.587 |
| abundant | quantity | True | 6 | 0.515 | 0.713 | 0.510 | 0.006 | 0.447 | 0.001 | 0.282 | 0.728 | 0.766 | -3.668 | -3.973 |
| extra | quantity | True | 6 | 0.516 | 0.580 | 0.506 | 0.010 | 0.447 | 0.005 | 0.129 | 0.576 | 0.658 | -3.621 | -4.045 |
| whose | possessive-or-pronoun | True | 6 | 0.533 | 0.509 | 0.495 | 0.038 | 0.462 | 0.030 | 0.097 | 0.559 | 0.585 | -3.092 | -3.638 |
| green | adjective | True | 6 | 0.550 | 0.561 | 0.479 | 0.071 | 0.477 | 0.058 | 0.071 | 0.547 | 0.701 | -3.355 | -3.907 |
| less | determiner-like | True | 6 | 0.570 | 0.511 | 0.497 | 0.072 | 0.494 | 0.060 | -0.022 | 0.472 | 0.619 | -3.513 | -4.032 |
| warm | adjective | True | 6 | 0.609 | 0.744 | 0.534 | 0.075 | 0.529 | 0.062 | 0.260 | 0.789 | 0.728 | -4.282 | -4.626 |
| little | determiner-like | True | 6 | 0.621 | 0.591 | 0.478 | 0.144 | 0.539 | 0.122 | 0.015 | 0.555 | 0.614 | -3.804 | -4.242 |
| additional | quantity | True | 6 | 0.648 | 0.620 | 0.615 | 0.033 | 0.562 | 0.026 | 0.066 | 0.628 | 0.741 | -3.568 | -3.902 |
| dark | adjective | True | 6 | 0.694 | 0.736 | 0.503 | 0.190 | 0.602 | 0.163 | 0.185 | 0.788 | 0.720 | -4.050 | -4.490 |
| huge | adjective | True | 6 | 0.696 | 0.707 | 0.604 | 0.093 | 0.605 | 0.077 | 0.125 | 0.730 | 0.836 | -4.061 | -4.273 |
| cheap | adjective | True | 6 | 0.822 | 0.766 | 0.562 | 0.260 | 0.714 | 0.224 | 0.104 | 0.818 | 0.778 | -3.590 | -3.768 |
| thirty | numeral | True | 6 | 0.835 | 0.903 | 0.687 | 0.148 | 0.725 | 0.126 | 0.369 | 1.094 | 0.821 | -4.701 | -4.870 |
| nineteen | numeral | True | 6 | 0.848 | 0.949 | 0.801 | 0.047 | 0.737 | 0.038 | 0.395 | 1.132 | 0.945 | -4.812 | -4.865 |
| forty | numeral | True | 6 | 0.871 | 0.890 | 0.742 | 0.129 | 0.757 | 0.109 | 0.414 | 1.171 | 0.874 | -4.810 | -4.899 |
| twenty | numeral | True | 6 | 0.880 | 0.907 | 0.729 | 0.151 | 0.765 | 0.128 | 0.439 | 1.204 | 0.828 | -4.796 | -4.885 |
| eighteen | numeral | True | 6 | 0.925 | 0.988 | 0.903 | 0.023 | 0.805 | 0.016 | 0.312 | 1.116 | 0.975 | -4.654 | -4.821 |

## Execution ledger

- Executed prompt keys: 232
- Executed noun keys: 80
