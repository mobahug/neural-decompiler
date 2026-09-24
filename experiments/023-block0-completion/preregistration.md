# Experiment 023 — preregistered conditions

- Lock run `9428b2fde588ac75` at commit `13b8d392f9e88844e2673c530a37228337cc64b7`; design revision 2 (`5b38aba`), plan revision 1 (`3a795fb`)
- Calibration record content sha256 `a391500c28868612df5c818e15a82174575b627e2ddea1fc88fbaa44644388c3` (file `94db6df26156d7a134e51f8afbf10707ca18030525f62dd08f02fb298e9ba03a`), B = 10000
- Exposed cells: data `d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4`, index `628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe`
- Confirmation file content sha256 `4e64d4c2c85ae8f9c710171372a4bf28f0964a8e8864a0326182edba44aa14ed`: {'S1-REF': 18, 'S1-VALIDITY': 18, 'Y1': 2592, 'Y2': 432}
- Y1 table `experiments/023-block0-completion/locked-y1-table.f64`: sha256 `37603f058500e056214a8a6ae413ef2b27eb94c5e1d618caaa8c9a61121a2f96`, index sha256 `3bf85e186db7dfbf8c80f45557275058317e259cb5995dcd2f73e327dfd9644d`, blocks [('cue_final', [1728, 2, 79]), ('coordinated', [864, 2, 79])]
- Y2 table: written at stage 1 as `outputs/experiment-023/y2-table.f64` in the same format, blocks [('cue_final', [288, 2, 79]), ('coordinated', [144, 2, 79])]; frozen at the stage-1 barrier

Statistic: g = (SSE0 − SSE1) / (SSE0 − SSEC), pooled from per-pair sufficient statistics; unclipped. Meaning guard: g ≥ 0.9. Effective requirement of a PASS: g ≥ max(F, 0.9).

| condition | envelope F (lower, exact order statistic) | meaning guard | guard-bound |
|---|---|---|---|
| Y1/cue_final | ≥ v₍250₎ = 0.996897 | g ≥ 0.9 | no |
| Y1/coordinated | ≥ v₍250₎ = 0.998634 | g ≥ 0.9 | no |
| Y2/cue_final | ≥ v₍250₎ = 0.995053 | g ≥ 0.9 | no |
| Y2/coordinated | ≥ v₍250₎ = 0.995186 | g ≥ 0.9 | no |

Each condition has exactly one result, decided in the order NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS:

- `NOT_INTERPRETABLE`: too little Level-0 → ceiling gap remains (SST ≤ 0 or SSE0 − SSEC < 0.02·SST on the full aggregate); neither a pass nor a failure
- `GUARD_FAILURE`: g < 0.90: the completed program does not recover essentially all of the explainable upstream gap on that population and group
- `ENVELOPE_ONLY_FAILURE`: g ≥ 0.90, so essentially all of the explainable gap is recovered, but g lies below the calibrated exposed-like envelope: a quantitative shift, never a refutation
- `PASS`: the completed program recovers essentially all of the explainable upstream gap within the exposed-like envelope: the weight-derived upstream program reaches the ceiling the decoded downstream readout allows

- The ceiling: C is the frozen downstream ceiling comparator, not a mathematical upper bound on a finite sample's R²; P1 slightly outperforming it (g > 1) is permitted and never an incident, and says nothing beyond 'essentially all'.
- Aggregate: none: the four condition results are the result, each read on its own; Y1 and Y2 are never pooled; no all-pass requirement.
- Gap rule: evaluated on the condition's full aggregate; it removes, re-weights or selects no pair, cue, frame or noun.
- Scope: Experiment 023 prospectively tests completion across new determiner-like, quantity and adjective cues and new sentence frames only; it makes no claim about all grammatical-number cue classes and no new prospective claim for possessive or pronoun cues.
- What a pass does not show: a pass does not show that the downstream readout is complete, anything about possessive or pronoun cues, new nouns or behavior beyond Δc at p_t, or that a cheaper block-0 rule would suffice (the comparators are descriptive).
