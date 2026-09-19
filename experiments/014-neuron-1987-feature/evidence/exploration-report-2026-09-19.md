# Experiment 014 Report

- Run ID: `00eef3b8bb95c8b8`
- Confirmation sha256: `a28edf831413489b2c8ffd4710bba4d742d4e542348120b0cab90c4f3ab3796d`
- Experiment 013 lock sha256: `ae6ec5937f0b11c82cbc26ba0d9a04df2fb644b85acf5625e3682e45da578519`
- Protocol/code commit at explore: `bcb0bfe3e94ee24a2740ee22b4e9228789a7436d`

## Phases

- `confirm`: `not_started`
- `explore`: `complete`
- `lock`: `complete`
- `report`: `not_started`

## Tier A — exposed pool (calibration record only; the floors are frozen constants)

- Replication of Experiment 013: 3732 pairs, max deviation 0.00e+00; identities neuron_sum 2.0e-08, p1_cross_check 6.6e-16, rho_identity 1.1e-07; read weight vs Experiment 011 lock 0.0e+00
- Neuron 2/1987: cos(γ₂ ⊙ W_in, d̂_E) 0.241; r(W_out) 0.333; r(W_out)/denominator per template {cardinal: 0.204, coordinated-adjective: 0.204, quantifier: 0.222}; b_in -0.659
- Operating points: pre_ref -1.60 to 0.49; reference activation max 0.337 (below the firing threshold 0.5)
- Token means (135): Δâ vs Δa — Spearman 0.985, R² 0.989, MAE 0.049, bias 0.020 (n 135); Δa spread sd 0.825; axis-only Δâ vs Δa — Spearman 0.562, R² 0.166, MAE 0.686, bias 0.454 (n 135)
- Pairs (3732): Δâ vs Δa — Spearman 0.976, R² 0.963, MAE 0.094, bias 0.015 (n 3732); Δp̂re vs Δpre — Spearman 0.980, R² 0.968, MAE 0.175, bias 0.025 (n 3732); Jacobian form vs Δpre — Spearman 0.963, R² 0.913, MAE 0.309, bias 0.203 (n 3732) (vs the exact form R² 0.952)
- Classification (pairs): predictor balanced accuracy 0.989 (sensitivity 0.991, specificity 0.987; raw 0.988, majority baseline 0.689; measured firing 1160/3732); axis-only balanced accuracy 0.611 (sensitivity 0.938, specificity 0.284; raw 0.487, majority baseline 0.689; measured firing 1160/3732)
- Jacobian form (descriptive): Δâ_J vs Δa token means — Spearman 0.952, R² 0.913, MAE 0.161, bias 0.154 (n 135); pairs — Spearman 0.947, R² 0.868, MAE 0.191, bias 0.151 (n 3732); balanced accuracy 0.979 (sensitivity 0.993, specificity 0.965; raw 0.974, majority baseline 0.689; measured firing 1160/3732)
- Amplitude among firing pairs: error 0.219 on a mean Δa of 1.66; MAE all pairs 0.094
- Accounting (mean |Jacobian contribution|): E 1.08, MLP₁ 0.33, heads₁ 0.41 (negative in 0.94 of pairs); axis 1.35, off-axis 1.14; remainders: pattern change |.| 0.175, LayerNorm linearization |.| 0.229
- Neuron's share of the block-2 read change among firing pairs 1.63; mean term firing 0.351, not firing -0.016
- Measured firing set (≥ half of frames), by class: extension: few, five, four, many, ten, three; inherited: hundred, multiple, numerous, twelve; numeral: billion, eight, eighteen, eighty, fifty, forty, million, nine, nineteen, ninety, seven, seventy, six, sixty, thirty, thousand, trillion, twenty; original-cue: cardinal:pl, quantifier:pl; plural-numeral: eleven, fifteen, fourteen, seventeen, sixteen, thirteen; quantity: countless, dozen, fewer, myriad, various
- Predicted firing set: extension: few, five, four, many, ten, three; inherited: hundred, multiple, numerous, twelve; numeral: billion, eight, eighteen, eighty, fifty, forty, million, nine, nineteen, ninety, seven, seventy, six, sixty, thirty, thousand, trillion, twenty; original-cue: cardinal:pl, quantifier:pl; plural-numeral: eleven, fifteen, fourteen, seventeen, sixteen, thirteen; quantity: countless, dozen, fewer, myriad, various

## Lock

- Candidate lock sha256 `34b97174ebd68617f7a13a7d876f405761a8faaf42c6675ac8fa42de8c805d01`; predictions sha256 `a0b7448a4c42faee538b180794f95abf6fbf5e3049ab8358ba83875ce05b091e`

## Execution ledger

- Executed prompt keys: 126
- Executed noun keys: 80
