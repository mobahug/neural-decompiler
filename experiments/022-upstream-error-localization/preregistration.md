# Experiment 022 — preregistered conditions

- Lock run `be529d6f87a9f24d` at commit `f947305ebfde9739bc79f2559e0233c9292ec47a`; design revision 4 (`219cdc5`), plan revision 3 (`e6d8299`)
- Calibration record content sha256 `46985fd59d9d22dc08c9344a2fa96eacaf832dfe954e84526fc76c5ac0d12f6d` (file `db745653cfa8f411514e80839bf986bd6cbbe3e816e3bb3095cf43c2d9d11a87`), B = 10000
- Confirmation file content sha256 `af848ae4bd48551cffec67f169a2a6d7b47f0eaaa5d2451010cc4c8bfd8fcda9`: {'S1-REF': 18, 'S1-VALIDITY': 18, 'Y1': 2592, 'Y2': 432}
- Y1 table companion `experiments/022-upstream-error-localization/locked-y1-table.f64`: sha256 `cb9fbf74cd1eacd72b02257b27941cfed4a8cfbcd703583a19f18d65fed7caaf`, index sha256 `a6f413b24c1dfed546df7c88187284b8db45ab3a62c59a3d11265a0cd16247f3`, blocks [('cue_final', [1728, 16, 79]), ('coordinated', [864, 32, 79])]
- Y2 table: written at stage 1 as `outputs/experiment-022/y2-table.f64` in the same format, blocks [('cue_final', [288, 16, 79]), ('coordinated', [144, 32, 79])]; frozen at the stage-1 barrier

| condition | statistic | envelope (exact order statistic) | meaning guard | guard-bound |
|---|---|---|---|---|
| Y1/C1 block-0 attention dominates the cue-final gap | share(Bv) + share(Bp), cue-final | ≥ v₍250₎ = 0.901075 | s1 ≥ 0.5 | no |
| Y1/C2 the layer-1–2 reductions contribute little to the cue-final gap | share(R), cue-final | ≤ v₍9751₎ = 0.005595 | s2 ≤ 0.1 | no |
| Y1/C3 the layer-1–2 reductions contribute little to the coordinated gap | share(R), coordinated | ≤ v₍9751₎ = 0.006904 | s3 ≤ 0.1 | no |
| Y1/C4 block-0 cue→target attention contributes positively to the coordinated gap | share(T), coordinated | ≥ v₍250₎ = 0.327059 | s4 > 0.0 | no |
| Y2/C1 block-0 attention dominates the cue-final gap | share(Bv) + share(Bp), cue-final | ≥ v₍250₎ = 0.888785 | s1 ≥ 0.5 | no |
| Y2/C2 the layer-1–2 reductions contribute little to the cue-final gap | share(R), cue-final | ≤ v₍9751₎ = 0.008226 | s2 ≤ 0.1 | no |
| Y2/C3 the layer-1–2 reductions contribute little to the coordinated gap | share(R), coordinated | ≤ v₍9751₎ = 0.014172 | s3 ≤ 0.1 | no |
| Y2/C4 block-0 cue→target attention contributes positively to the coordinated gap | share(T), coordinated | ≥ v₍250₎ = 0.207165 | s4 > 0.0 | no |

Each condition has exactly one result, decided in the order NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS:

- `NOT_INTERPRETABLE`: the gap rule fails (G < 0.02 or SST = 0 on the full aggregate): localization is not interpretable; neither a pass nor a failure
- `GUARD_FAILURE`: the meaning guard fails: the preregistered qualitative localization claim fails for that population, whatever the envelope
- `ENVELOPE_ONLY_FAILURE`: the guard holds and the envelope fails: a quantitative shift relative to the exposed-like calibration; the qualitative claim survives
- `PASS`: guard and envelope hold: the attribution lies within the exposed-like envelope and keeps the claim's meaning

An envelope-only failure keeps the qualitative statement; a guard failure refutes it for that population:

- C1: kept — block-0 attention dominates the cue-final gap; refuted — block-0 attention does not dominate the cue-final gap
- C2: kept — the reductions contribute little to the cue-final gap; refuted — the reductions do not contribute little to the cue-final gap
- C3: kept — the reductions contribute little to the coordinated gap; refuted — the reductions do not contribute little to the coordinated gap
- C4: kept — block-0 cue→target attention contributes positively; refuted — the block-0 cue→target attention does not contribute positively

- Layers 1–2: R off is Experiment 017's reduced chain with its layer-1/2 reference rows through the cue position p_c (as validated in 017–019); Experiment 020's rows through p_t (the defect of the 020/021 errata) enter only the descriptive historical comparator.
- CDF percentile: descriptive only; the frozen envelope decides; for the upper bounds C2 and C3 a high percentile lies toward the unfavorable upper tail.
- C4: a PASS of C4 establishes a positive contribution of block-0 cue→target attention only, never a large or substantial one.
- Aggregate: none: the eight condition results are the result, each reported and read on its own; Y1 and Y2 are never pooled; no all-pass requirement.
- Gap rule: evaluated on the condition's full group aggregate; it removes, re-weights or selects no pair, cue, frame or noun.
- What a pass does not show: a pass does not show that any corrected program would predict well (022 builds none; the full composition is the identity endpoint only), nor which of 016/017's reductions carries σ_R.
