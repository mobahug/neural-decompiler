# Experiment 025: Does Moving the Cue Embedding Along the Frozen Nounness Direction Causally Change the Frozen Readout Error and the Attention Routing?

**Status: IMPLEMENTED; no scientific phase has run.** The next step is the production `freeze`, only when separately
authorized.

Implements the design
[`docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md`](../../docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md)
(revision 1, `c0885e5`, corrected in `26c9925`) through the plan
[`docs/superpowers/plans/2026-09-26-experiment-025-nounness-direction-intervention-plan.md`](../../docs/superpowers/plans/2026-09-26-experiment-025-nounness-direction-intervention-plan.md)
(revision 1, `7d90d28`). It includes the confirm-time patch-path check, which the reviewer signed off.

This is the final experiment of the sequence. After it, on any clean result, the project stops and is written up.

The question: Experiment 024 showed that a frozen, weight-derived nounness score *predicts* the frozen block-4/5
readout error. Does *moving* a fresh cue's input embedding along that score's own direction *cause* the error, and the
directly captured L4/L5 routing, to change in the predicted direction?

- **The intervention:** a norm-preserving rotation of the cue's input embedding at `("EMBED", p_c)`.
  - `d = μ̂_noun − μ̂_cue` is 024's frozen direction, from 024's calibration record.
  - `R(±θ) = |E|·(cos θ·Ê ± sin θ·t̂)`, where `t̂` is the unit tangent part of `d` at `Ê` and `θ = asin(odd/τ)`.
  - The directional (odd) score component is exactly ±0.32 (primary) and ±0.16 (secondary). The complete change is
    `s₀(cos θ − 1) ± odd`.
- **The controls:** 7 deterministic random tangent directions per cue, at the same angle.
  - They are orthogonal to `Ê` and `t̂`, so they are nounness-neutral.
  - They come from a SHA-256 counter-mode stream (tag `025|control|{token_id}|{j}`) through Box–Muller in float64.
  - A plurality control is secondary.
- **The 21 conditions per pair:** `base`, `noun±0.32`, `noun±0.16`, `rand1…7±0.32` and `plur±0.32`.
  - The 16 outcome-bearing ones are `noun±0.32` and `rand1…7±0.32`.
  - 40 cues × 108 exposed frames × 21 conditions = 90,720 runs, each under a condition-tagged key
    `frame_id|word|token_id|condition`.
- **The measurements**, per cue and condition:
  - `ℓ = log(Σ SSE_C / Σ n)` over the 108 frames × 79 nouns;
  - `D_attn`, the total-variation distance of the captured L4/L5 rows at `p_t` from 020's locked rows, averaged
    equally over the 16 heads, then over the frames.
- **The outcome-bearing statistics**, each a sign count that passes at ≥ 27 of 40 strictly positive (ties and zeros
  count against; the exact Binomial(40, ½) reference tail is 0.01924):
  - `A_i = ½[ℓ(+θ) − ℓ(−θ)]`: causal direction, + against −;
  - `B_i = A_i − mean_j |A_ij|`: specificity against the absolute random-control effects;
  - `G_i = ½[D_attn(+θ) − D_attn(−θ)]`: the routing summary moves in the predicted direction.
- **The outcome**, in the frozen hierarchy:
  - `NOT_INTERPRETABLE`;
  - `CAUSAL_EFFECT_NOT_ESTABLISHED`;
  - `DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC`;
  - `READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED`;
  - `NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR`.

  The stratum counts (20 adjectives, 20 ordinary singular nouns), the magnitudes, the baselines, the half dose, the
  plurality control and the block-4/5 ladder are descriptive. They run after the result is written, and can never
  alter it.
- **The population:** 40 fresh cues, picked mechanically.
  - The adjectives are the first 20 of 024's adjective reserves.
  - The nouns are 024's 8 eligible ordinary reserves, then the first 12 eligible words of a frozen alphabetical
    39-word list.
  - The tightened prior-noun rule also excludes the 24 nouns of 020's confirmation file, which 021 read; that
    excludes `statue`.

## Code

- **Module:** `src/neural_decompiler/cue_rotation.py` (`cr`).
  - It calls 024's `readout_routing.py`, 023's `block0_completion.py` and 022's `upstream_localization.py`, pinned by
    git blob with 022's ten.
  - It binds 024's reviewed calibration record, lock and freeze, and 020's confirmation file, by file and content
    digest.
  - The production configuration is literal and immutable: 20 + 20 cues, 7 controls, the doses 0.32 and 0.16, and
    the threshold 27, derived from the Binomial tail. A test world can only pass its own configuration explicitly.
- **Runner:** `experiments/025-nounness-direction-intervention/run.py`.
  - Phases: `validate`, `freeze`, `lock`, `confirm` and `report`. There is no calibration phase.
  - It takes no option and always runs the production configuration.
- **Tests:**
  - `tests/test_cue_rotation.py` holds tier A (pure) and the tier-C contracts (the pinned model, opt-in, spent
    prompts only).
  - `tests/test_experiment_025_runner.py` holds tier B (the six-layer fake, on 024's locked fake world).

## Phases (each only when separately authorized)

1. **`freeze`** (tokenizer only) writes `confirmation-v1.json`, which is committed by hand and reviewed.
   - A shortfall, or picks other than the design's expected 40, writes nothing and creates no state.
2. **`lock`** (weights only; every forward refused) writes the candidate `preregistration-lock.json` and
   `preregistration.md`. They are installed byte-identically, committed and reviewed.
   - It computes the direction, each cue's geometry, the controls, the plurality tangent and all 840 patched vectors.
   - It enforces every geometry gate: float64 within 1e-12; the float32-patched vectors within 1e-6; the θ = 0
     vector equals the model's row bit for bit.
3. **`confirm`** (once; never resumed):
   - `validate_lock`, including the direct module-blob check, then the model digests;
   - I7′ before any prompt: the lock's geometry recomputed bit for bit, and every geometry gate;
   - the patch-path check on four already-executed keys. It compares a plain capture with a patched θ = 0 run, and
     the embedding hook with the block-0 residual input, bit for bit. It records equality and digests only, outside
     the ledger, the manifest and the accounting;
   - the ledger of all 90,720 keys, then every run once, with the measurements saved first. Stage 2 executes frame by
     frame, as 022 and 024 did; the tensors are stored in `ul.table_units` order, and the order of independent
     forwards changes no value;
   - the accounting, and `C` recomputed from the saved `Δx3` bit for bit;
   - I1, I3 and I4 on every run, and the Level-1 identity on the 69,120 outcome-bearing runs;
   - the result, with every cue's per-condition responses, and the completed phase in one atomic write;
   - then the descriptive records.
4. **`report`**.

An I7′ or patch-path failure is an incident before the ledger. Any failure after the ledger is an incident that
carries no result. The measurements are kept (if stage 2 itself fails, the runs measured so far, saved after the
incident is on disk), and the outcome is `NOT_INTERPRETABLE`.

- **What the patch-path check can and cannot show.** It checks the θ = 0 plumbing only: a patched run with the
  model's own row reproduces the plain run, and the embedding hook equals the block-0 residual input. It cannot show
  that a rotated vector propagates. That is shown on every run by the vector I1 gate: `d_emb + ΔE + ΔA0` against the
  measured `Δx1`, with `d_emb` the rotated vector's change. A patch that did not land would miss I1 by orders of
  magnitude. A pass of the check does not strengthen the scientific result.
- **Runtime.** The estimate is about 2.5–3.5 hours on this machine: 90,720 patched forwards, the identity gates, and
  the Level-1 identity on every run. The ladder adds about 26,000 more Level-1 evaluations after the result.
- **The production confirm** runs through an external guarded launcher, as in 024. It must live under `evidence/`, or
  it counts as a scientific-path change. The launcher:
  - logs each patched run's condition, for example by mapping the replacement's digest to the lock's vectors, since
    `run_patched` sees untagged prompt keys;
  - keeps the four spent-key patch-path runs apart from the fresh runs.

The local outputs stay in `outputs/experiment-025/`.
