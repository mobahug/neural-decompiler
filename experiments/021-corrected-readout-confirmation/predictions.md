# Experiment 021 — preregistered predictions and floors

- Lock run `9725893f39907cbd` at commit `ad226714ccaac7a50d81b818b654b681cad172a7`; confirmation set sha256 `e098e2b44a1702d2b35c20db2ce111358897ea867996024c3e51a79f379c309d`; program blob `caa73b40192f4c910dc63371bd19db75a3258339`
- Calibration record content sha256 `f939a84df3ff3dcf55f95877c951059c5157857d72d6be8ef036e10f7923044d`: 1 Y1 row, 64 Y2 rows, 84 Y3 rows
- Y1 floors: {'cue_mae_k80': 1.0258303158503683, 'pair_mean_r2': 0.0, 'pooled_mae': 0.797678095698689, 'token_mean_r2': 0.0}
- Y2 floors at 6/6/6 (the row is selected at stage 1 from the valid-frame composition): {'coordinated_r2': 0.0, 'cue_final_r2': 0.7422346561720841, 'frame_mean_r2': 0.0, 'frame_r2_k75': 0.0}
- Y3 floors at 8/8/8 (the row is selected from the measured scorability of the fresh nouns): {'noun_bias_k90': 0.7673115211477541, 'noun_median_r2': 0.25533781815393086, 'noun_r2_k90': 0.03562825423475868, 'noun_slope_dev_k90': 0.22402394345475662}
- Pass predicates: R²-type finite and > 0 and ≥ floor; error-type finite and ≤ floor
- Y1 table: 2592 rows (fresh cues × exposed frames), each with the predicted Δĉ of the 79 scorable exposed nouns and the 24 fresh nouns

| token | frame | template | mean Δĉ | mean Δĉ (no L05 heads) | mean Δĉ (template-base MLPs) | ΔT̂ |
|---|---|---|---|---|---|---|
| recent | cardinal-1 | cardinal | -3.5303 | -3.5340 | -3.2514 | 0.7747 |
| current | cardinal-1 | cardinal | -2.8731 | -2.8965 | -2.8091 | 0.6905 |
| original | cardinal-1 | cardinal | -2.9622 | -2.9768 | -3.0105 | 0.6826 |
| typical | cardinal-1 | cardinal | -3.3007 | -3.3075 | -3.0696 | 0.7069 |
| ordinary | cardinal-1 | cardinal | -3.6355 | -3.6264 | -3.1784 | 0.7556 |
| identical | cardinal-1 | cardinal | -3.2061 | -3.2124 | -3.2028 | 0.6170 |
| average | cardinal-1 | cardinal | -2.6851 | -2.6716 | -2.7115 | 0.5639 |
| excessive | cardinal-1 | cardinal | -3.2231 | -3.2128 | -2.8715 | 0.8094 |
| exhaustive | cardinal-1 | cardinal | -3.1338 | -3.1211 | -2.9143 | 0.8320 |
| comprehensive | cardinal-1 | cardinal | -2.5921 | -2.6007 | -2.5123 | 0.7642 |
| thorough | cardinal-1 | cardinal | -2.9887 | -2.9707 | -2.8304 | 0.7559 |
| sweeping | cardinal-1 | cardinal | -3.6074 | -3.6154 | -3.6359 | 0.7671 |
| anything | cardinal-1 | cardinal | -3.1677 | -3.1750 | -3.4179 | 0.1314 |
| something | cardinal-1 | cardinal | -2.5849 | -2.5924 | -2.9251 | 0.2270 |
| everything | cardinal-1 | cardinal | -3.7199 | -3.7234 | -3.2935 | 0.3409 |
| nothing | cardinal-1 | cardinal | -3.3726 | -3.3738 | -3.3248 | 0.4519 |
| who | cardinal-1 | cardinal | -3.7658 | -3.7550 | -3.0862 | 0.5052 |
| thee | cardinal-1 | cardinal | -2.4676 | -2.5023 | -2.6157 | 0.4602 |
| purple | cardinal-1 | cardinal | -3.4732 | -3.5017 | -3.2591 | 0.7125 |
| yellow | cardinal-1 | cardinal | -3.2629 | -3.2730 | -3.1390 | 0.6026 |
| hidden | cardinal-1 | cardinal | -4.1253 | -4.1199 | -4.0544 | 0.8211 |
| hard | cardinal-1 | cardinal | -4.5897 | -4.5705 | -4.2022 | 0.7963 |
| dirty | cardinal-1 | cardinal | -4.2956 | -4.2759 | -4.1080 | 0.8025 |
| rare | cardinal-1 | cardinal | -4.3965 | -4.4086 | -4.0002 | 0.7931 |
| recent | cardinal-2 | cardinal | -3.9671 | -3.9495 | -3.1250 | 0.7439 |
| current | cardinal-2 | cardinal | -3.4161 | -3.4085 | -3.1372 | 0.5694 |
| original | cardinal-2 | cardinal | -2.9335 | -2.9281 | -2.5762 | 0.6004 |
| typical | cardinal-2 | cardinal | -3.1417 | -3.1298 | -2.6169 | 0.6069 |
| ordinary | cardinal-2 | cardinal | -3.6288 | -3.6023 | -3.2997 | 0.6375 |
| identical | cardinal-2 | cardinal | -3.0070 | -2.9989 | -2.6470 | 0.5495 |
| average | cardinal-2 | cardinal | -3.0616 | -3.0341 | -2.4452 | 0.4637 |
| excessive | cardinal-2 | cardinal | -3.5573 | -3.5348 | -2.8873 | 0.7950 |
| exhaustive | cardinal-2 | cardinal | -3.7203 | -3.6894 | -2.8982 | 0.7917 |
| comprehensive | cardinal-2 | cardinal | -3.0837 | -3.0719 | -2.3463 | 0.6827 |
| thorough | cardinal-2 | cardinal | -3.2918 | -3.2630 | -2.5815 | 0.7199 |
| sweeping | cardinal-2 | cardinal | -4.3753 | -4.3570 | -3.8734 | 0.8214 |
| anything | cardinal-2 | cardinal | -2.8616 | -2.8481 | -3.1637 | 0.1789 |
| something | cardinal-2 | cardinal | -2.4120 | -2.4052 | -2.7823 | 0.2392 |
| everything | cardinal-2 | cardinal | -3.6740 | -3.6584 | -3.3904 | 0.2245 |
| nothing | cardinal-2 | cardinal | -3.4066 | -3.3951 | -3.3657 | 0.4111 |
| … | … | … | … | … | … | … |  <!-- 2552 further rows in the lock -->

Every row is a prediction of the model's own contrast change from the frame's reference state, the committed Experiment 011/012/017 locks and the cue's token id alone.
