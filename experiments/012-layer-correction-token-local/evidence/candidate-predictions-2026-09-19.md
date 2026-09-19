# Experiment 012 — preregistered predictions (token-local MLP model of the layers-1–2 correction)

- Lock run `cec6c3908508f723` at commit `ddbbbd3a53977aaa5eb3cc6fdd1056c05783112d`; confirmation set sha256 `32892d23d8f853f6af71218a0562a6ffc7c375ae0fae81fc9566a79866964e07`; Experiment 011 lock sha256 `769bfeacd7c49fc18bed2ff3c5cf8d5ea2be4231e9f8e9c493819b5d7ba69d0b`
- τ_c 0.207 (Y1 MAE ceiling, token means, leave-one-frame-out); τ_P 0.130 (Y2); floors Spearman ≥ 0.8 and R² ≥ 0.5 (Y1), Spearman ≥ 0.9 (Y2)
- Defined templates: ['cardinal', 'quantifier', 'coordinated-adjective']; weight-only denominators {'cardinal': 1.629910036913704, 'coordinated-adjective': 1.629910036913704, 'quantifier': 1.4960738062126115}; modelled plural totals D̂_T {cardinal: 1.7450, quantifier: 2.2127, coordinated-adjective: 1.8512}

| token | class | template | predicted ĉ | MLP₁ part | MLP₂ part | ĉ(ΔE_∥) | ĉ(ΔE_⊥) | interaction | g_E | q̂' = g_E + ĉ | q̂ (head's units) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| half | determiner-like | cardinal | 0.030 | -0.027 | 0.057 | 0.051 | 0.079 | -0.100 | 0.430 | 0.460 | 0.430 |
| half | determiner-like | quantifier | 0.375 | 0.022 | 0.353 | -0.027 | 0.398 | 0.004 | 0.550 | 0.925 | 0.625 |
| half | determiner-like | coordinated-adjective | 0.010 | -0.065 | 0.075 | 0.040 | 0.102 | -0.132 | 0.430 | 0.441 | 0.388 |
| **half** | | **mean over 6 licensed frames** | **0.138** | -0.023 | 0.162 | 0.021 | 0.193 | -0.076 | 0.470 | 0.609 | **0.481** |
| whichever | determiner-like | cardinal | 0.051 | -0.024 | 0.075 | 0.033 | 0.157 | -0.139 | 0.339 | 0.390 | 0.364 |
| whichever | determiner-like | quantifier | 0.415 | 0.230 | 0.185 | -0.043 | 0.462 | -0.003 | 0.451 | 0.866 | 0.586 |
| whichever | determiner-like | coordinated-adjective | 0.116 | 0.019 | 0.097 | 0.018 | 0.226 | -0.129 | 0.339 | 0.455 | 0.401 |
| **whichever** | | **mean over 6 licensed frames** | **0.194** | 0.075 | 0.119 | 0.002 | 0.282 | -0.090 | 0.377 | 0.570 | **0.450** |
| whatever | determiner-like | cardinal | -0.101 | -0.057 | -0.043 | 0.025 | 0.024 | -0.149 | 0.418 | 0.317 | 0.296 |
| whatever | determiner-like | quantifier | 0.215 | 0.157 | 0.058 | -0.047 | 0.315 | -0.054 | 0.537 | 0.752 | 0.508 |
| whatever | determiner-like | coordinated-adjective | 0.032 | -0.079 | 0.111 | 0.009 | 0.158 | -0.134 | 0.418 | 0.451 | 0.397 |
| **whatever** | | **mean over 6 licensed frames** | **0.049** | 0.007 | 0.042 | -0.004 | 0.166 | -0.112 | 0.458 | 0.507 | **0.401** |
| which | determiner-like | cardinal | -0.163 | -0.158 | -0.005 | 0.040 | 0.036 | -0.239 | 0.424 | 0.261 | 0.244 |
| which | determiner-like | quantifier | 0.168 | 0.117 | 0.051 | -0.038 | 0.341 | -0.134 | 0.544 | 0.712 | 0.481 |
| which | determiner-like | coordinated-adjective | -0.169 | -0.177 | 0.008 | 0.026 | 0.052 | -0.247 | 0.424 | 0.256 | 0.225 |
| **which** | | **mean over 6 licensed frames** | **-0.055** | -0.073 | 0.018 | 0.009 | 0.143 | -0.207 | 0.464 | 0.409 | **0.317** |
| what | determiner-like | cardinal | -0.058 | -0.135 | 0.077 | 0.041 | 0.141 | -0.240 | 0.521 | 0.464 | 0.433 |
| what | determiner-like | quantifier | 0.166 | 0.014 | 0.152 | -0.038 | 0.378 | -0.174 | 0.650 | 0.816 | 0.551 |
| what | determiner-like | coordinated-adjective | -0.029 | -0.121 | 0.092 | 0.027 | 0.166 | -0.222 | 0.521 | 0.493 | 0.434 |
| **what** | | **mean over 6 licensed frames** | **0.027** | -0.081 | 0.107 | 0.010 | 0.228 | -0.212 | 0.564 | 0.591 | **0.473** |
| fifty | numeral | cardinal | 0.390 | 0.042 | 0.348 | 0.099 | 0.253 | 0.038 | 0.747 | 1.137 | 1.062 |
| fifty | numeral | quantifier | 0.797 | 0.173 | 0.624 | 0.051 | 0.597 | 0.149 | 0.896 | 1.693 | 1.145 |
| fifty | numeral | coordinated-adjective | 0.422 | 0.078 | 0.344 | 0.105 | 0.262 | 0.055 | 0.747 | 1.169 | 1.029 |
| **fifty** | | **mean over 6 licensed frames** | **0.536** | 0.098 | 0.439 | 0.085 | 0.371 | 0.081 | 0.797 | 1.333 | **1.079** |
| sixty | numeral | cardinal | 0.266 | 0.002 | 0.264 | 0.092 | 0.124 | 0.050 | 0.777 | 1.043 | 0.975 |
| sixty | numeral | quantifier | 0.698 | 0.134 | 0.564 | 0.039 | 0.470 | 0.190 | 0.928 | 1.626 | 1.100 |
| sixty | numeral | coordinated-adjective | 0.321 | 0.029 | 0.292 | 0.096 | 0.147 | 0.078 | 0.777 | 1.098 | 0.967 |
| **sixty** | | **mean over 6 licensed frames** | **0.429** | 0.055 | 0.373 | 0.076 | 0.247 | 0.106 | 0.828 | 1.256 | **1.014** |
| seventy | numeral | cardinal | 0.393 | 0.096 | 0.296 | 0.097 | 0.233 | 0.063 | 0.773 | 1.165 | 1.089 |
| seventy | numeral | quantifier | 0.832 | 0.241 | 0.591 | 0.048 | 0.593 | 0.191 | 0.923 | 1.755 | 1.187 |
| seventy | numeral | coordinated-adjective | 0.450 | 0.122 | 0.328 | 0.103 | 0.280 | 0.068 | 0.773 | 1.223 | 1.077 |
| **seventy** | | **mean over 6 licensed frames** | **0.558** | 0.153 | 0.405 | 0.082 | 0.368 | 0.107 | 0.823 | 1.381 | **1.117** |
| eighty | numeral | cardinal | 0.283 | 0.062 | 0.220 | 0.088 | 0.179 | 0.015 | 0.780 | 1.062 | 0.992 |
| eighty | numeral | quantifier | 0.718 | 0.201 | 0.517 | 0.031 | 0.540 | 0.146 | 0.931 | 1.649 | 1.115 |
| eighty | numeral | coordinated-adjective | 0.336 | 0.082 | 0.255 | 0.091 | 0.208 | 0.037 | 0.780 | 1.116 | 0.982 |
| **eighty** | | **mean over 6 licensed frames** | **0.445** | 0.115 | 0.331 | 0.070 | 0.309 | 0.066 | 0.830 | 1.276 | **1.030** |
| ninety | numeral | cardinal | 0.238 | 0.078 | 0.160 | 0.082 | 0.101 | 0.055 | 0.806 | 1.044 | 0.975 |
| ninety | numeral | quantifier | 0.644 | 0.216 | 0.428 | 0.021 | 0.433 | 0.190 | 0.960 | 1.603 | 1.084 |
| ninety | numeral | coordinated-adjective | 0.254 | 0.083 | 0.171 | 0.083 | 0.119 | 0.053 | 0.806 | 1.060 | 0.934 |
| **ninety** | | **mean over 6 licensed frames** | **0.379** | 0.126 | 0.253 | 0.062 | 0.217 | 0.099 | 0.857 | 1.236 | **0.998** |
| scarce | quantity | cardinal | -0.083 | -0.163 | 0.080 | 0.057 | 0.076 | -0.216 | 0.721 | 0.638 | 0.596 |
| scarce | quantity | quantifier | 0.362 | 0.044 | 0.318 | -0.020 | 0.461 | -0.079 | 0.867 | 1.229 | 0.831 |
| scarce | quantity | coordinated-adjective | 0.018 | -0.137 | 0.155 | 0.047 | 0.177 | -0.206 | 0.721 | 0.739 | 0.651 |
| **scarce** | | **mean over 6 licensed frames** | **0.099** | -0.085 | 0.185 | 0.028 | 0.238 | -0.167 | 0.770 | 0.869 | **0.693** |
| least | quantity | cardinal | -0.079 | -0.027 | -0.052 | 0.040 | 0.043 | -0.162 | 0.380 | 0.301 | 0.281 |
| least | quantity | quantifier | 0.307 | 0.163 | 0.144 | -0.038 | 0.458 | -0.113 | 0.495 | 0.802 | 0.543 |
| least | quantity | coordinated-adjective | -0.080 | -0.061 | -0.019 | 0.026 | 0.013 | -0.120 | 0.380 | 0.300 | 0.264 |
| **least** | | **mean over 6 licensed frames** | **0.049** | 0.025 | 0.024 | 0.009 | 0.171 | -0.132 | 0.418 | 0.468 | **0.362** |
| myriad | quantity | cardinal | 0.378 | -0.055 | 0.433 | 0.071 | 0.405 | -0.098 | 0.754 | 1.132 | 1.058 |
| myriad | quantity | quantifier | 0.890 | 0.130 | 0.760 | 0.001 | 0.869 | 0.021 | 0.903 | 1.794 | 1.213 |
| myriad | quantity | coordinated-adjective | 0.489 | -0.014 | 0.503 | 0.066 | 0.465 | -0.042 | 0.754 | 1.244 | 1.095 |
| **myriad** | | **mean over 6 licensed frames** | **0.586** | 0.020 | 0.566 | 0.046 | 0.579 | -0.040 | 0.804 | 1.390 | **1.122** |
| manifold | quantity | cardinal | 0.177 | -0.042 | 0.219 | 0.056 | 0.228 | -0.107 | 0.563 | 0.740 | 0.691 |
| manifold | quantity | quantifier | 0.608 | 0.077 | 0.531 | -0.021 | 0.622 | 0.007 | 0.695 | 1.303 | 0.881 |
| manifold | quantity | coordinated-adjective | 0.352 | -0.005 | 0.357 | 0.046 | 0.385 | -0.079 | 0.563 | 0.915 | 0.806 |
| **manifold** | | **mean over 6 licensed frames** | **0.379** | 0.010 | 0.369 | 0.027 | 0.412 | -0.060 | 0.607 | 0.986 | **0.793** |
| sparse | quantity | cardinal | 0.125 | -0.084 | 0.209 | 0.066 | 0.174 | -0.115 | 0.613 | 0.738 | 0.690 |
| sparse | quantity | quantifier | 0.393 | 0.033 | 0.360 | -0.006 | 0.495 | -0.096 | 0.750 | 1.143 | 0.773 |
| sparse | quantity | coordinated-adjective | 0.210 | -0.059 | 0.268 | 0.060 | 0.247 | -0.098 | 0.613 | 0.823 | 0.724 |
| **sparse** | | **mean over 6 licensed frames** | **0.243** | -0.037 | 0.279 | 0.040 | 0.305 | -0.103 | 0.659 | 0.901 | **0.729** |
| hers | possessive-or-pronoun | cardinal | 0.110 | -0.022 | 0.132 | 0.029 | 0.278 | -0.197 | 0.174 | 0.284 | 0.265 |
| hers | possessive-or-pronoun | quantifier | 0.425 | 0.116 | 0.308 | -0.045 | 0.586 | -0.116 | 0.271 | 0.696 | 0.470 |
| hers | possessive-or-pronoun | coordinated-adjective | 0.160 | 0.035 | 0.125 | 0.014 | 0.331 | -0.185 | 0.174 | 0.334 | 0.294 |
| **hers** | | **mean over 6 licensed frames** | **0.232** | 0.043 | 0.188 | -0.001 | 0.398 | -0.166 | 0.206 | 0.438 | **0.343** |
| theirs | possessive-or-pronoun | cardinal | -0.027 | -0.107 | 0.080 | 0.041 | 0.059 | -0.128 | 0.380 | 0.353 | 0.329 |
| theirs | possessive-or-pronoun | quantifier | 0.302 | 0.067 | 0.235 | -0.037 | 0.346 | -0.006 | 0.496 | 0.798 | 0.539 |
| theirs | possessive-or-pronoun | coordinated-adjective | 0.023 | -0.044 | 0.067 | 0.028 | 0.103 | -0.108 | 0.380 | 0.403 | 0.355 |
| **theirs** | | **mean over 6 licensed frames** | **0.099** | -0.028 | 0.127 | 0.011 | 0.169 | -0.081 | 0.419 | 0.518 | **0.408** |
| ours | possessive-or-pronoun | cardinal | 0.071 | -0.074 | 0.145 | 0.029 | 0.199 | -0.158 | 0.277 | 0.347 | 0.324 |
| ours | possessive-or-pronoun | quantifier | 0.396 | 0.066 | 0.330 | -0.045 | 0.533 | -0.092 | 0.383 | 0.779 | 0.527 |
| ours | possessive-or-pronoun | coordinated-adjective | 0.071 | -0.027 | 0.098 | 0.014 | 0.191 | -0.134 | 0.277 | 0.348 | 0.306 |
| **ours** | | **mean over 6 licensed frames** | **0.179** | -0.012 | 0.191 | -0.001 | 0.308 | -0.128 | 0.312 | 0.491 | **0.386** |
| thy | possessive-or-pronoun | cardinal | -0.240 | -0.130 | -0.110 | 0.053 | -0.099 | -0.194 | 0.536 | 0.296 | 0.277 |
| thy | possessive-or-pronoun | quantifier | 0.114 | 0.072 | 0.043 | -0.025 | 0.223 | -0.085 | 0.665 | 0.780 | 0.527 |
| thy | possessive-or-pronoun | coordinated-adjective | -0.106 | -0.092 | -0.014 | 0.043 | -0.017 | -0.131 | 0.536 | 0.430 | 0.379 |
| **thy** | | **mean over 6 licensed frames** | **-0.077** | -0.050 | -0.027 | 0.024 | 0.036 | -0.137 | 0.579 | 0.502 | **0.394** |
| black | adjective | cardinal | -0.000 | -0.099 | 0.099 | 0.072 | 0.157 | -0.229 | 0.566 | 0.566 | 0.528 |
| black | adjective | quantifier | 0.357 | 0.010 | 0.347 | 0.003 | 0.474 | -0.120 | 0.698 | 1.055 | 0.713 |
| black | adjective | coordinated-adjective | 0.031 | -0.082 | 0.113 | 0.068 | 0.173 | -0.211 | 0.566 | 0.597 | 0.525 |
| **black** | | **mean over 6 licensed frames** | **0.129** | -0.057 | 0.186 | 0.048 | 0.268 | -0.187 | 0.610 | 0.739 | **0.589** |
| white | adjective | cardinal | 0.027 | -0.061 | 0.088 | 0.058 | 0.172 | -0.203 | 0.467 | 0.493 | 0.461 |
| white | adjective | quantifier | 0.327 | 0.066 | 0.261 | -0.018 | 0.463 | -0.118 | 0.590 | 0.917 | 0.620 |
| white | adjective | coordinated-adjective | 0.067 | -0.026 | 0.092 | 0.049 | 0.190 | -0.172 | 0.467 | 0.533 | 0.469 |
| **white** | | **mean over 6 licensed frames** | **0.140** | -0.007 | 0.147 | 0.030 | 0.275 | -0.164 | 0.508 | 0.648 | **0.517** |
| young | adjective | cardinal | -0.069 | -0.140 | 0.071 | 0.049 | 0.104 | -0.222 | 0.595 | 0.526 | 0.491 |
| young | adjective | quantifier | 0.387 | 0.143 | 0.244 | -0.029 | 0.545 | -0.128 | 0.730 | 1.117 | 0.755 |
| young | adjective | coordinated-adjective | 0.008 | -0.113 | 0.120 | 0.037 | 0.159 | -0.189 | 0.595 | 0.603 | 0.531 |
| **young** | | **mean over 6 licensed frames** | **0.108** | -0.037 | 0.145 | 0.019 | 0.269 | -0.180 | 0.640 | 0.749 | **0.592** |
| empty | adjective | cardinal | -0.047 | -0.026 | -0.022 | 0.049 | 0.148 | -0.245 | 0.509 | 0.461 | 0.431 |
| empty | adjective | quantifier | 0.232 | 0.070 | 0.162 | -0.029 | 0.427 | -0.166 | 0.636 | 0.868 | 0.587 |
| empty | adjective | coordinated-adjective | 0.051 | 0.051 | 0.000 | 0.038 | 0.217 | -0.203 | 0.509 | 0.560 | 0.493 |
| **empty** | | **mean over 6 licensed frames** | **0.079** | 0.032 | 0.047 | 0.019 | 0.264 | -0.205 | 0.551 | 0.630 | **0.504** |
| wooden | adjective | cardinal | 0.087 | -0.038 | 0.124 | 0.077 | 0.296 | -0.286 | 0.507 | 0.593 | 0.554 |
| wooden | adjective | quantifier | 0.384 | 0.050 | 0.334 | 0.012 | 0.618 | -0.246 | 0.633 | 1.017 | 0.688 |
| wooden | adjective | coordinated-adjective | 0.165 | 0.039 | 0.126 | 0.075 | 0.356 | -0.266 | 0.507 | 0.672 | 0.592 |
| **wooden** | | **mean over 6 licensed frames** | **0.212** | 0.017 | 0.195 | 0.055 | 0.423 | -0.266 | 0.549 | 0.761 | **0.611** |
