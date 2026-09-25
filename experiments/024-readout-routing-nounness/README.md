# Experiment 024: Does an Operational Nounness Score Predict When the Frozen Downstream Routing Stops Holding?

**Status: IMPLEMENTED; no scientific phase has run.** The next step is the production `freeze`, only when separately
authorized.

Implements the design
[`docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md`](../../docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md)
(revision 2, `9d03dee`) through the plan
[`docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md`](../../docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md)
(revision 1, `608088c`) and the reviewer's six implementation requirements.

The question: does a frozen, weight-derived operational nounness score prospectively predict, on 40 never-executed cue
words, how much the frozen block-4/5 attention readout errs in the 108 exposed frames?

- **The score:** `nounness(w) = cos(E_w, μ_noun) − cos(E_w, μ_cue)`.
  - `E_w` is the input embedding row, in float64.
  - `μ_noun` is the mean of the 158 singular and plural rows of the 79 scorable exposed nouns.
  - `μ_cue` is the mean of the 139 exposed calibration cues' rows, leave-one-out for a calibration cue.
- **The response:** each cue's `MSE = Σ SSE_C / Σ n` over its 108 exposed pairs.
  - `SSE_C = Σ_nouns (Δc − C)²`.
  - `C` is Experiment 020's frozen Level-0 readout fed the measured `Δx3`.
- **The primary test:** the Spearman `ρ` across the 40 cues. A PASS needs `ρ ≥ max(F_ρ, null₉₇.₅)`.
  - `F_ρ` is element `[249]` of 10,000 SHA-indexed draws of 40 calibration cues.
  - `null₉₇.₅` is element `[97499]` of 100,000 SHA-indexed permutations.
  - The four-way results, in precedence order: `NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS`.
- **The E–N disambiguation guard:** an exact one-sided permutation test of
  `D_EN = mean(log MSE_E) − mean(log MSE_N)`, comparing the 8 ordinary singular nouns with the 8 non-noun controls.
  - All `C(16, 8) = 12,870` assignments are enumerated, in exact integer arithmetic on the observed binary64 values.
  - It passes iff `K ≤ 321`, where `K` counts the assignments at or above the observed contrast, ties included.
  - An all-tied population fails.
- **The outcome**, in the frozen hierarchy:
  - `NOUNNESS_PREDICTION_NOT_ESTABLISHED`;
  - `ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED`;
  - `NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`.

  Every outcome is predictive and associational. The secondary B/C/D/E contrasts and the block-4/5 ladder run after
  the result is written, and can never alter it.

## Code

- **Module:** `src/neural_decompiler/readout_routing.py` (`rr`).
  - It calls 022's `upstream_localization.py` and 023's `block0_completion.py`, pinned by git blob with 022's ten.
  - It binds 023's reviewed exposed cells, confirmation and lock by file and content digest.
  - The production configuration is literal and immutable: 8 per class, a guard of 8 + 8 with 12,870 assignments and
    a largest passing `K` of 321, `B` = 10,000 and `P` = 100,000. A test world can only pass its own configuration
    explicitly.
- **Runner:** `experiments/024-readout-routing-nounness/run.py`.
  - Phases: `validate`, `freeze`, `calibrate`, `lock`, `confirm` and `report`.
  - It takes no option and always runs the production configuration.

## Phases (each only when separately authorized)

1. **`freeze`** (tokenizer only) writes `confirmation-v1.json`, which is committed by hand and reviewed.
   - It picks the first 8 eligible entries of each ordered list.
   - A shortfall, or picks other than the design's expected 40, writes nothing and creates no state.
2. **`calibrate`** (once; 023's committed exposed cells and the weights; no forward pass) writes the candidate
   `calibration-v1.json`. It is installed byte-identically, committed and reviewed.
3. **`lock`** (weights only) writes the candidate `preregistration-lock.json` and `preregistration.md`. They are
   installed byte-identically, committed and reviewed.
4. **`confirm`** (once; never resumed):
   - `validate_lock`, then the model digests;
   - I7 before any prompt: the lock's scores, centroids and predictions, bit for bit;
   - the 4,320 prompts, once each, with the measurements saved first;
   - the accounting, and `C` recomputed from the saved `Δx3` bit for bit;
   - I1, I3 and I4;
   - the result and the completed phase in one atomic write;
   - then the descriptive records.
5. **`report`**.

The local outputs stay in `outputs/experiment-024/`.
