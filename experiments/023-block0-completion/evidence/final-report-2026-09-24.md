# Experiment 023 — report

- Run `9428b2fde588ac75`; phase commits {'calibrate': '4e5deebd021bc284143b01b76b45efd28e1a1344', 'confirm': 'c7efec7f32709dc20ecb83cae16bb73927a14366', 'extract': '2111271892a8237f69c3ef18d6eec774e798ac5c', 'lock': '13b8d392f9e88844e2673c530a37228337cc64b7'}; phases: calibrate complete, confirm complete, extract complete, lock complete, report not_started
- Design revision 2 (`5b38aba`), plan revision 1 (`3a795fb`)
- Scope: Experiment 023 prospectively tests completion across new determiner-like, quantity and adjective cues and new sentence frames only; it makes no claim about all grammatical-number cue classes and no new prospective claim for possessive or pronoun cues.


## Extraction (exposed cells from Experiment 022's verified table)

- E1 digests and orders verified; E2 and E3 bit for bit; E4 max |ΔP1| 0.000; E5 max 4.0e-15 over 192; E6 pairs 1.4e-14, draws 4.0e-15
- Artifact data sha256 `d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4`, index `628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe`

## Calibration (exposed only, B = 10000)

- Kernel/loop check 8.6e-14 over 192; E6 max 7.5e-15

| condition | envelope F | min of defined draws | median | max | undefined | guard-bound | PASS | ENVELOPE_ONLY | GUARD | NOT_INTERPRETABLE |
|---|---|---|---|---|---|---|---|---|---|---|
| Y1/cue_final | ≥ v₍250₎ = 0.996897 | 0.995470 | 0.998224 | 1.000004 | 0 | no | 0.9751 | 0.0249 | 0.0000 | 0.0000 |
| Y1/coordinated | ≥ v₍250₎ = 0.998634 | 0.997933 | 0.999205 | 0.999779 | 0 | no | 0.9751 | 0.0249 | 0.0000 | 0.0000 |
| Y2/cue_final | ≥ v₍250₎ = 0.995053 | 0.989844 | 0.998071 | 1.006282 | 0 | no | 0.9751 | 0.0249 | 0.0000 | 0.0000 |
| Y2/coordinated | ≥ v₍250₎ = 0.995186 | 0.986847 | 0.998423 | 1.001055 | 0 | no | 0.9751 | 0.0249 | 0.0000 | 0.0000 |

- Joint rate, all four passing (descriptive only): 0.9178

## The four conditions (the result; no aggregate label)

| condition | g | envelope F | meaning guard | result | gap (SSE0 − SSEC)/SST | R²₀ | R²₁ | R²_C | ceiling-limited | CDF percentile (descriptive) | pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Y1/cue_final | 0.995520 | 0.996897 | g ≥ 0.9 | **ENVELOPE_ONLY_FAILURE** | 0.1411 | 0.8182 | 0.9587 | 0.9593 | no | 0.0001 (10000 / 0) | 1728 |
| Y1/coordinated | 0.998714 | 0.998634 | g ≥ 0.9 | **PASS** | 0.4159 | 0.5420 | 0.9574 | 0.9579 | no | 0.0431 (10000 / 0) | 864 |
| Y2/cue_final | 0.997419 | 0.995053 | g ≥ 0.9 | **PASS** | 0.1468 | 0.8146 | 0.9610 | 0.9613 | no | 0.3180 (10000 / 0) | 288 |
| Y2/coordinated | 0.996563 | 0.995186 | g ≥ 0.9 | **PASS** | 0.3812 | 0.5719 | 0.9518 | 0.9531 | no | 0.0927 (10000 / 0) | 144 |

Readings (frozen, design revision 2):

- Y1/cue_final: **ENVELOPE_ONLY_FAILURE** — g ≥ 0.90, so essentially all of the explainable gap is recovered, but g lies below the calibrated exposed-like envelope: a quantitative shift, never a refutation.
- Y1/coordinated: **PASS** — the completed program recovers essentially all of the explainable upstream gap within the exposed-like envelope: the weight-derived upstream program reaches the ceiling the decoded downstream readout allows.
- Y2/cue_final: **PASS** — the completed program recovers essentially all of the explainable upstream gap within the exposed-like envelope: the weight-derived upstream program reaches the ceiling the decoded downstream readout allows.
- Y2/coordinated: **PASS** — the completed program recovers essentially all of the explainable upstream gap within the exposed-like envelope: the weight-derived upstream program reaches the ceiling the decoded downstream readout allows.

- The ceiling: C is the frozen downstream ceiling comparator, not a mathematical upper bound on a finite sample's R²; P1 slightly outperforming it (g > 1) is permitted and never an incident, and says nothing beyond 'essentially all'.
- Aggregate: none: the four condition results are the result, each read on its own; Y1 and Y2 are never pooled; no all-pass requirement.
- What a pass does not show: a pass does not show that the downstream readout is complete, anything about possessive or pronoun cues, new nouns or behavior beyond Δc at p_t, or that a cheaper block-0 rule would suffice (the comparators are descriptive).
- Kernel against the direct recomputation: 1.6e-15 over 8

## Descriptive records (no outcome force)

- Stage-2 identities (maxima): I1 1.324621902565326e-05 (tolerance 1e-04), I3 2.028769855580874e-05 (tolerance 1e-04), I4 4.332195089062907e-05 (tolerance 1e-03)
- Y1/coordinated/stratum/adjective: g 0.9988; gap 0.3781; R²₁ 0.9621; R²_C 0.9625
- Y1/coordinated/stratum/determiner-like: g 0.9992; gap 0.7016; R²₁ 0.9589; R²_C 0.9594
- Y1/coordinated/stratum/quantity: g 0.9971; gap 0.2143; R²₁ 0.9505; R²_C 0.9511
- Y1/coordinated/template/coordinated-adjective: g 0.9987; gap 0.4159; R²₁ 0.9574; R²_C 0.9579
- Y1/cue_final/stratum/adjective: g 0.9926; gap 0.1294; R²₁ 0.9635; R²_C 0.9644
- Y1/cue_final/stratum/determiner-like: g 0.9959; gap 0.1187; R²₁ 0.9672; R²_C 0.9676
- Y1/cue_final/stratum/quantity: g 0.9974; gap 0.1828; R²₁ 0.9432; R²_C 0.9437
- Y1/cue_final/template/cardinal: g 0.9955; gap 0.1536; R²₁ 0.9631; R²_C 0.9638
- Y1/cue_final/template/quantifier: g 0.9955; gap 0.1531; R²₁ 0.9457; R²_C 0.9463
- Y2/coordinated/stratum/adjective: g 0.9971; gap 0.4474; R²₁ 0.9520; R²_C 0.9533
- Y2/coordinated/stratum/determiner-like: g 0.9991; gap 0.5658; R²₁ 0.9706; R²_C 0.9711
- Y2/coordinated/stratum/quantity: g 0.9858; gap 0.1496; R²₁ 0.9327; R²_C 0.9348
- Y2/coordinated/template/coordinated-adjective: g 0.9966; gap 0.3812; R²₁ 0.9518; R²_C 0.9531
- Y2/cue_final/stratum/adjective: g 0.9933; gap 0.1432; R²₁ 0.9654; R²_C 0.9663
- Y2/cue_final/stratum/determiner-like: g 0.9973; gap 0.1407; R²₁ 0.9702; R²_C 0.9706
- Y2/cue_final/stratum/quantity: g 1.0014; gap 0.1694; R²₁ 0.9436; R²_C 0.9434
- Y2/cue_final/template/cardinal: g 0.9985; gap 0.1497; R²₁ 0.9603; R²_C 0.9605
- Y2/cue_final/template/quantifier: g 0.9962; gap 0.1618; R²₁ 0.9569; R²_C 0.9575
- Cheaper block-0 rules, Y1/coordinated (descriptive g): value_only 0.9246, relative_self_logit 0.9671, oracle_self_weight 0.9681, linear_response 0.9728, heads_4 0.9391, heads_6 0.9865
- Cheaper block-0 rules, Y1/cue_final (descriptive g): value_only 0.5045, relative_self_logit 0.7426, oracle_self_weight 0.7533, linear_response 0.7596, heads_4 0.6862, heads_6 0.9145
- Cheaper block-0 rules, Y2/coordinated (descriptive g): value_only 0.8868, relative_self_logit 0.9393, oracle_self_weight 0.9320, linear_response 0.9624, heads_4 0.9292, heads_6 0.9752
- Cheaper block-0 rules, Y2/cue_final (descriptive g): value_only 0.4878, relative_self_logit 0.7571, oracle_self_weight 0.7464, linear_response 0.7714, heads_4 0.6797, heads_6 0.9352
- P1 Δx3 relative error Y1/coordinated: p_c median 2.69e-02 max 1.09e-01, p_t median 1.12e-02 max 4.07e-02
- P1 Δx3 relative error Y1/cue_final: p_c median 2.10e-02 max 1.05e-01
- P1 Δx3 relative error Y2/coordinated: p_c median 3.26e-02 max 8.03e-02, p_t median 1.24e-02 max 4.25e-02
- P1 Δx3 relative error Y2/cue_final: p_c median 1.91e-02 max 7.32e-02
- Block-0 profile Y1/coordinated: self-weight [0.2916897100075006, 0.0996048754811383, 0.6461148331275236, 0.9156683804837089, 0.17742457337254672, 0.7655042756904789, 0.3455641294552416, 0.40737729081247454]; p_t→p_c [0.04509995678412827, 0.33714277049612973, 0.002050463040022173, 5.683361835608286e-05, 0.12262770356735161, 0.6042227222347824, 0.2851258683191067, 0.10293703017124707]
- Block-0 profile Y1/cue_final: self-weight [0.1922937257511134, 0.0962315866627124, 0.5993330303603975, 0.9992378435806276, 0.16892309361114205, 0.4764086628669938, 0.38348854149464023, 0.5758045031967831]; p_t→p_c None
- Block-0 profile Y2/coordinated: self-weight [0.2842439364852824, 0.10558531585157778, 0.6476980184963254, 0.9156967064534631, 0.1977889607761801, 0.4867123227024655, 0.3041366246136388, 0.4373959401499004]; p_t→p_c [0.03829982280372706, 0.3511236709864902, 0.0015066322130348818, 2.1013162222749377e-05, 0.12964502589388027, 0.4483849143956251, 0.2712576024839005, 0.14658878798983466]
- Block-0 profile Y2/cue_final: self-weight [0.2115311999231467, 0.09363585839683194, 0.5992866692576003, 0.9992273966123798, 0.16677249305464134, 0.5765031461044133, 0.37990732390581733, 0.5564918335518145]; p_t→p_c None
- Validity (descriptive, selects nothing) cardinal-023-1: True
- Validity (descriptive, selects nothing) cardinal-023-2: True
- Validity (descriptive, selects nothing) cardinal-023-3: True
- Validity (descriptive, selects nothing) cardinal-023-4: True
- Validity (descriptive, selects nothing) cardinal-023-5: True
- Validity (descriptive, selects nothing) cardinal-023-6: True
- Validity (descriptive, selects nothing) coordinated-adjective-023-1: True
- Validity (descriptive, selects nothing) coordinated-adjective-023-2: True
- Validity (descriptive, selects nothing) coordinated-adjective-023-3: True
- Validity (descriptive, selects nothing) coordinated-adjective-023-4: True
- Validity (descriptive, selects nothing) coordinated-adjective-023-5: True
- Validity (descriptive, selects nothing) coordinated-adjective-023-6: True
- Validity (descriptive, selects nothing) quantifier-023-1: True
- Validity (descriptive, selects nothing) quantifier-023-2: True
- Validity (descriptive, selects nothing) quantifier-023-3: True
- Validity (descriptive, selects nothing) quantifier-023-4: True
- Validity (descriptive, selects nothing) quantifier-023-5: True
- Validity (descriptive, selects nothing) quantifier-023-6: True
