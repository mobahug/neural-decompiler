# Experiment 013 Report

- Run ID: `db79dab2e9f63182`
- Confirmation sha256: `25f891e0a52da6cf60cd428f191010bf5cdc02d279b495b699f8cecc3e6c8f06`
- Experiment 012 lock sha256: `830abc3b4a2d86a8c904cc467d7b08223f0b197623945f1edb8f5edd55c2a6eb`
- Protocol/code commit at explore: `cc14911f7efe27f7e60c26b132555273a4bf64bb`

## Phases

- `confirm`: `not_started`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

## Tier A — exposed pool (calibration record only; tolerances are frozen constants)

- Replication of Experiment 012: 2724 pairs, max deviation 0.00e+00; ĉ_012 reproduced to 0.00e+00
- Identities (max relative errors): neuron_sum 2.0e-08, p1_cross_check 6.6e-16, rho_identity 1.1e-07; per-head OV identity max 8.7e-07; read weight vs Experiment 011 lock 0.0e+00
- Frozen τ_r 0.07 (design RMSE 0.0233, recomputed 0.0233); τ_A 0.106 (design RMSE 0.0355, recomputed 0.0355); c_H degenerate threshold 0.017
- Token means (111 tokens): residual r̂ vs r — Spearman 0.946, MAE 0.018, RMSE 0.0233, R² 0.870, bias 0.004 (n 111); r spread sd 0.0647; base-point only R² 0.393
- Token means: full ĉ_L vs c_L — Spearman 0.987, MAE 0.018, RMSE 0.0233, R² 0.976, bias 0.004 (n 111); the 012 model vs c_L — Spearman 0.924, MAE 0.055, RMSE 0.0703, R² 0.781, bias 0.027 (n 111); own base vs c_L — Spearman 0.949, MAE 0.042, RMSE 0.0504, R² 0.887, bias -0.013 (n 111)
- Pairs (2724): residual — Spearman 0.885, MAE 0.036, RMSE 0.0466, R² 0.801, bias 0.002 (n 2724); ĉ_H vs c_H — Spearman 0.856, MAE 0.028, RMSE 0.0355, R² 0.729, bias 0.006 (n 2724) (c_H spread sd 0.0681); ĉ_M vs c_M — Spearman 0.986, MAE 0.029, RMSE 0.0391, R² 0.972, bias -0.004 (n 2724); attention-input — Spearman 0.848, MAE 0.029, RMSE 0.0391, R² 0.751, bias -0.004 (n 2724)
- Per template (pairs, residual): {cardinal: 0.896/0.032, coordinated-adjective: 0.866/0.031, quantifier: 0.891/0.045}
- Ladder means: base-point -0.041, frozen-attention 0.010, remainder -0.002 (|.| 0.036; direct pattern change |.| 0.028, through block 2's MLP |.| 0.029, layer-2 heads' arrival |.| 0.005; token-mean |.| 0.018)
- Per head (pairs Spearman): {L01.H00 0.68, L01.H01 0.87, L01.H02 0.76, L01.H03 0.91, L01.H04 0.95, L01.H05 0.94, L01.H06 0.71, L01.H07 0.85, L02.H00 0.81, L02.H01 0.92, L02.H02 0.89, L02.H03 0.76, L02.H04 0.67, L02.H05 0.92, L02.H06 0.79, L02.H07 0.86}
- Compensation cases (frozen rule): 742 pairs, measured signs as predicted in 0.97
- thy (measured c_M / c_H vs predicted): {cardinal-012-1: (-0.157, -0.066) vs (-0.149, -0.049), cardinal-012-2: (-0.044, -0.048) vs (-0.066, -0.020), coordinated-adjective-012-1: (-0.111, -0.048) vs (-0.161, -0.021), coordinated-adjective-012-2: (-0.103, -0.017) vs (-0.131, -0.010), quantifier-012-1: (0.339, -0.282) vs (0.318, -0.201), quantifier-012-2: (0.184, -0.118) vs (0.157, -0.113)}

## Lock

- Candidate lock sha256 `ae6ec5937f0b11c82cbc26ba0d9a04df2fb644b85acf5625e3682e45da578519`; predictions sha256 `d7928546bc2e10200672090f3d6d570afedf83eed938a5776dccc7a0eadecd55`

## Execution ledger

- Executed prompt keys: 108
- Executed noun keys: 80
