# Experiment 019 — preregistered predictions (frame-conditioned routing of channel D; every prospective rung of the per-position masked chain)

- Lock run `aa8f8607ca8c4cb5` at commit `2dbfdd126432ed547384ed0bcda010c88f475185`; confirmation set sha256 `87540d9b5dacd71cb2e87443e736822bcc6290325516aa8a65095794826db5a5`; Experiment 018 lock sha256 `fdfae9106898f51528fc6c1a3e235bd9be7f37f3ccb00628ccf7522e308fae46`
- Decision size k = 64; Y1/Y2 Δκ_64(E) ≥ 0.05 pooled, E_64 above S'_64 in strictly more than half of the valid frames, split gains ≥ 0.02; a negative result is NOT_EVALUABLE iff the witness headroom H*_64 < 0.05 and E's gain < 0.05; Y3 ρ_64 ≥ 0.6 pooled and ≥ 0.5 per set (denominators ≥ 0.05); Y4 margins 0.05 pooled and on Y2; Y5 per set mean overlap ≥ 0.6 and ≥ S' + 0.2; precondition: reference rung c_L ≥ 0.98, rows ≥ 0.95, ΔT ≥ 0.95, c_L gaps ≥ 0.05 pooled and per family split
- Locked lists: S'_1 [1987]; S'_64 at p_c (first 16) [111, 129, 173, 228, 287, 383, 434, 511, 561, 626, 637, 658, 702, 741, 745, 761]…; inherited S_64 (first 16) [129, 228, 287, 383, 434, 511, 561, 576, 626, 637, 658, 702, 741, 745, 761, 764]…; E_64 and G_64 per exposed frame and position in the lock
- O* is a cross-validated empirical headroom witness (a leave-one-cue-out greedy fit of the scored objective), not a global oracle, ceiling, certificate or upper bound: a high H* establishes transferable selectable headroom under the frozen witness; a low H* means the witness did not establish enough headroom for the fixed-population-versus-unpredicted distinction and is never a proof that no frame-specific headroom exists.
- Y1 table: 2160 rows (fresh tokens × exposed frames), each with every prospective rung's ĉ_L, F̂, Π̂, ΔT̂ and head row; Y2 rows are computed at confirm stage 1 and digested before any fresh cue prompt

| token | class | frames | **ĉ_L E_64** | ĉ_L S'_64 | ĉ_L G_64 | ĉ_L T_64 | ĉ_L S_0 | ĉ_L S_2048 | **ΔT̂ E_64** | ΔT̂ S'_64 | ΔT̂ S_2048 | frozen ΔT̂ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| couple | ordinal-or-numeral | 90 | **0.0733** | 0.0790 | 0.0685 | 0.0776 | 0.1100 | 0.0665 | **1.5836** | 1.5907 | 1.5814 | 1.6330 |
| different | determiner-like | 90 | **0.2648** | 0.2726 | 0.2640 | 0.2698 | 0.3054 | 0.2571 | **1.5890** | 1.5969 | 1.5805 | 1.6270 |
| generous | quantity | 90 | **0.1531** | 0.1622 | 0.1531 | 0.1596 | 0.1752 | 0.1379 | **1.4314** | 1.4475 | 1.4065 | 1.3898 |
| pair | ordinal-or-numeral | 90 | **0.1504** | 0.1564 | 0.1478 | 0.1551 | 0.1852 | 0.1457 | **1.3874** | 1.3930 | 1.3886 | 1.4647 |
| stolen | adjective | 90 | **0.1852** | 0.1935 | 0.1824 | 0.1912 | 0.2112 | 0.1662 | **1.3785** | 1.3887 | 1.3614 | 1.4590 |
| separate | determiner-like | 90 | **0.0525** | 0.0605 | 0.0503 | 0.0580 | 0.0818 | 0.0431 | **1.3552** | 1.3669 | 1.3436 | 1.3410 |
| gentle | adjective | 90 | **0.2178** | 0.2286 | 0.2185 | 0.2259 | 0.2409 | 0.2057 | **1.3429** | 1.3580 | 1.3286 | 1.3367 |
| giant | adjective | 90 | **0.1567** | 0.1675 | 0.1565 | 0.1637 | 0.1824 | 0.1520 | **1.3167** | 1.3337 | 1.3136 | 1.3308 |
| remaining | determiner-like | 90 | **0.1280** | 0.1435 | 0.1271 | 0.1395 | 0.1604 | 0.1123 | **1.3013** | 1.3229 | 1.2827 | 1.2936 |
| faded | adjective | 90 | **0.2105** | 0.2190 | 0.2103 | 0.2156 | 0.2322 | 0.1938 | **1.2801** | 1.2876 | 1.2680 | 1.3114 |
| similar | determiner-like | 90 | **0.2842** | 0.2955 | 0.2854 | 0.2926 | 0.3077 | 0.2771 | **1.2603** | 1.2766 | 1.2529 | 1.2462 |
| negligible | quantity | 90 | **0.1720** | 0.1836 | 0.1693 | 0.1806 | 0.1961 | 0.1583 | **1.2538** | 1.2719 | 1.2371 | 1.2347 |
| rotten | adjective | 90 | **0.1677** | 0.1750 | 0.1706 | 0.1728 | 0.1853 | 0.1585 | **1.1699** | 1.1779 | 1.1650 | 1.1292 |
| triple | ordinal-or-numeral | 90 | **0.1620** | 0.1683 | 0.1602 | 0.1669 | 0.2036 | 0.1620 | **1.1456** | 1.1523 | 1.1460 | 1.0696 |
| prior | determiner-like | 90 | **0.0234** | 0.0318 | 0.0194 | 0.0310 | 0.0575 | 0.0107 | **1.1576** | 1.1716 | 1.1365 | 1.1366 |
| silver | adjective | 90 | **0.1625** | 0.1717 | 0.1629 | 0.1705 | 0.1942 | 0.1515 | **1.1022** | 1.1147 | 1.0912 | 1.0770 |
| they | possessive-or-pronoun | 90 | **0.0478** | 0.0603 | 0.0435 | 0.0573 | 0.0718 | 0.0384 | **1.1031** | 1.1275 | 1.0819 | 1.1108 |
| we | possessive-or-pronoun | 90 | **0.0713** | 0.0823 | 0.0697 | 0.0783 | 0.0946 | 0.0580 | **1.0615** | 1.0820 | 1.0337 | 1.0225 |
| double | ordinal-or-numeral | 90 | **0.1857** | 0.1925 | 0.1833 | 0.1897 | 0.2117 | 0.1796 | **1.0235** | 1.0328 | 1.0144 | 0.9101 |
| maximum | quantity | 90 | **0.1286** | 0.1363 | 0.1261 | 0.1336 | 0.1442 | 0.1198 | **0.9535** | 0.9663 | 0.9377 | 0.8675 |
| minimum | quantity | 90 | **0.1371** | 0.1438 | 0.1352 | 0.1417 | 0.1554 | 0.1254 | **0.9113** | 0.9235 | 0.8952 | 0.8030 |
| overall | quantity | 90 | **0.0376** | 0.0462 | 0.0356 | 0.0449 | 0.0546 | 0.0212 | **0.9151** | 0.9306 | 0.8918 | 0.6142 |
| he | possessive-or-pronoun | 90 | **-0.0034** | 0.0086 | -0.0066 | 0.0038 | 0.0226 | -0.0142 | **0.7602** | 0.7852 | 0.7276 | 0.7343 |
| she | possessive-or-pronoun | 90 | **-0.0348** | -0.0231 | -0.0366 | -0.0277 | -0.0101 | -0.0429 | **0.6452** | 0.6715 | 0.6083 | 0.5967 |
