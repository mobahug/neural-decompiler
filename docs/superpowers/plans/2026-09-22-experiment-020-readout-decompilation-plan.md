# Experiment 020 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 020 design (revision 2, commit `cb4e7bb`, with the Level 1 error formula `E₁` added before planning): carry the already-decoded cue-to-transport chain through the post-`L03` readout path to the model's own singular-versus-plural logit contrast, and test it prospectively on unseen cues (Y1), unseen frames conditional on their stage-1 reference state (Y2) and unseen nouns (Y3).

**Architecture:** One module `readout_decompilation.py` that *consumes* the committed Experiment 011/012/017 objects through `head_pattern.HeadChainModel.upstream` (variant `LEVEL0`) and adds only what is downstream of it: blocks 3–5 at the changed positions, `LN_final`, and the weight-only noun read. Nothing inherited is refitted, re-derived or re-tuned; `layer_correction.LayerWeights` is constructed for layers `(3, 4, 5)` (it already accepts a `layers` argument), and `attention_patterns.LayerProgram` for layers 3–5. The pool builder extends `block_routing.build_pool_019` to `build_pool_020` (279 exposed tokens, 108 exposed frames, 80 nouns of which 79 scorable). The runner follows Experiments 017–019 exactly, including the two-stage barrier.

**Spec:** `docs/superpowers/specs/2026-09-22-experiment-020-readout-decompilation-design.md` (revision 2, `cb4e7bb`, `E₁` amendment). **The spec wins over this plan.**

## Global constraints

- Pinned model, runtime and seeds as Experiments 011–019 (`20260916` / `20260924`); float64 for every chain computation, float32 captures.
- **No fresh prompt of any kind before `confirm`.** `explore` executes only the 108 exposed frames' reference prompts and the exposed cue prompts of the exposed pool; `lock` runs **no forward pass** (`_weights_only`, the model object deleted after its parameters are copied, no capture API reachable); `confirm` runs the S1 keys, then — only after the barrier — the S2-TARGET keys.
- **No refitting.** The Experiment 011 lock (axes `T`, `R0`, read weight, `σ_T`), the Experiment 012 lock (template bases) and the Experiment 017 lock (layer-3 bases) are read verbatim, their digests recorded, and their objects passed unchanged into `hp.model_from_locks`. A test asserts that no Experiment 020 code path writes to them or re-derives them.
- **Noun freshness is defined by inspection, not by prompting.** No Experiment 020 phase before `confirm` may compute any quantity of a fresh noun; `explore` records nothing keyed by a fresh noun id, and a test asserts that the 24 fresh ids appear nowhere in the exploration record.
- **Label-bearing populations** (asserted in code, not only in the report): Y1 = 24 fresh cues × 108 exposed frames × the **79 scorable exposed** nouns; Y2 = 24 fresh cues × valid fresh frames × the **79 scorable exposed** nouns; Y3 = the **24 fresh** nouns over the Y1 cue/frame population; fresh nouns on Y2 = descriptive joint diagnostic, no threshold, no label.
- **Exposed nouns:** 80 entries, 79 scorable; `peach` is non-scorable by the frozen single-token rule (ids 759,607 / 759,3844). Both counts are asserted.
- **Fresh nouns:** the 24 of the spec, `pillar`/`pillars` (43035/40652) in place of the withdrawn `tunnel`.
- **Comparators keep their frozen standing** and are computed on both fresh sets: `ΔT`-only (exposed-disfavoured baseline), template-base MLPs (exposed-disfavoured operating-point comparator), no-L05-head (nested simplification comparator), rank-1 noun factorization (descriptive; its `d_noun` is fitted on exposed data at `explore`, written into the lock, and never refitted).
- **Incident semantics.** Scientific invalidity (invalid fresh frame, unscored cue, non-scorable noun) feeds the `PRECONDITION_FAILED_*` counts and never stops a phase. Incidents are implementation/protocol failures only, and every incident-triggering check uses its frozen tolerance:

  | check | constant | tolerance |
  |---|---|---|
  | readout identity | `READOUT_IDENTITY_TOLERANCE` | 2e-2 nats |
  | logit reconstruction | `LOGIT_IDENTITY_TOLERANCE` | 2e-2 |
  | additive residual identity | `ADDITIVE_IDENTITY_TOLERANCE` | 1e-4 |
  | Level 1 chain, `E₁ = ‖Δh6^exact − Δh6^meas‖_∞ / max(‖Δh6^meas‖_∞, 1e-12)` | `LEVEL1_TOLERANCE` | 7e-3 (revision 3; 1e-3 until the 2026-09-22 explore incident — see the design's amendment note) |
  | inherited 017 reproduction | `INHERITED_017_TOLERANCE` | 1e-6 |
  | locked Y1 prediction reproduction | `PREDICTION_REPRODUCTION_TOLERANCE` | 0.0 (exact) |
  | stage-1 table reproduction | digest equality + exact rows | 0.0 (exact) |
  | provenance invariant | `PROVENANCE_TOLERANCE` | 1e-6 |

  `E₁` is computed pairwise with the frozen norm-based formula; the maximum, the median and the argmax pair are recorded. No componentwise ratio is used anywhere.

## Data flow and leakage boundaries (what each phase may read and write; the test that enforces it)

| phase | may read | may execute (model) | writes | must not | enforced by |
|---|---|---|---|---|---|
| `validate` | manifest, extension, the 006–019 confirmation files and the 011/012/017 locks (digests), the frozen 020 confirmation file | nothing | nothing | load a model | test: `validate` fails with a `model_loader` that raises |
| `freeze-confirmation` | the above and the tokenizer | nothing | `confirmation-v1.json`: 24 cues, 18 frames, **24 fresh nouns with both ids**, the validity rule, and the complete prompt-key manifest in three classes (S1-REF 18, S1-VALIDITY 18, S2-TARGET 2592 + 432) | any model output; overwrite an existing file | test: refusal to overwrite; tokenizer-only construction; manifest key counts |
| `explore` (once) | the frozen inputs, the 011/012/017 locks, the weights | the 108 exposed frames' reference prompts and the exposed cue prompts of the exposed pool | `exploration`: reference states and captures, the identities with their measured values, the Level 0 and ceiling predictions and measured `Δc` over the exposed pool (79 scorable nouns), the comparators, the descriptive tables (contribution shares, the S1 cosine ladder, the reserve-margin table), the fitted `d_noun` of the descriptive rank-1 comparator | execute any confirmation key; compute any fresh-noun quantity; write any object of the inherited locks | tests: ledger ∩ manifest = ∅; fresh-noun ids absent from the record; inherited-lock objects unchanged (digest compare) |
| `lock` | `exploration`, the 011/012/017 locks, the confirmation file, the weights | **nothing** | `candidate-lock.json` (the 2592 × 79 Y1 prediction rows with every comparator column, the frozen floors and tolerances, the reference states of the 108 exposed frames, `d_noun`, the confirmation digest and the prompt-key manifest) and `candidate-predictions.md` | run any forward pass; read a fresh-frame or fresh-cue quantity | test: every capture/intervention entry point raises while `lock` runs |
| `confirm` stage 1 | the installed lock (byte-identical to the candidates), the confirmation file, the weights | per fresh frame: its **S1-REF** and **S1-VALIDITY** prompts only | `confirmation.stage1`: validity per frame with its measured values, the frames' reference captures and reference contrasts of all 103 scorable nouns, the frame-conditional prediction table (Y2 + the descriptive fresh-noun columns), serialized and **digested** | execute any S2-TARGET key; use a fresh-cue measurement | tests: the ledger at the barrier contains no S2-TARGET key; a spy on `measure_pair` sees only S1 token ids |
| barrier | the results state on disk | nothing | — | proceed on a missing or mismatched digest | `assert_stage_one_digest` re-reads from disk and verifies before any fresh cue prompt (test) |
| `confirm` stage 2 | the lock, stage 1's record, the weights | the S2-TARGET keys: 2592 Y1 prompts, then the Y2 prompts of the **valid** fresh frames | measured contrasts of all 103 scorable nouns per pair; the recomputed predictions compared with the locked/digested rows at their tolerances; the scoring on the frozen populations; the comparators | score a fresh noun into Y1 or Y2; execute a Y2 key of an invalid frame | tests: population assertions; invalid frames' keys never in the ledger |
| `report` | the results state | nothing | `outputs/experiment-020/report.md` | — | — |

Two structural rules make the table hold without relying on discipline: (1) the Level 0 program is one function whose arguments are the frame's reference capture, the inherited locks' objects, the weights and a **token id** — it cannot reach a cue prompt, which the provenance test proves by recomputing predictions in a process that never executes one; (2) the noun population of every scored statistic is an explicit index list built once from the confirmation file, so a fresh noun cannot leak into Y1/Y2 by iteration order.

## Objects and naming

- `ReadoutProgram`: the Level 0 downstream program. `blocks_3_to_5(state, dx3, *, l5_heads=True, mlps_at_base=None)` returns `Δ̂h6` at `p_t` and the per-block parts; `level1(state, dx3_measured)` the exact chain; `contrast(state, dh6, nouns)` the noun read `⟨LN_final(h6_ref + Δ̂h6) − LN_final(h6_ref), Δw(n)⟩`.
- `FrameState020`: the reference capture — `x1…x5` at every position ≤ `p_t`, `h6`, the layer-4 and layer-5 reference rows at `p_t`, the reference contrasts of the scorable nouns, `p_c`, `p_t`, plus the Experiment 017 `FrameState017` it embeds.
- `NounSet`: `Δw(n)` and the id pair per noun, with the `scorable` flag and the population masks `exposed_scorable` (79) and `fresh` (24).
- Prediction columns per pair: `dc_hat` (Level 0, per noun), `dc_hat_nol5`, `dc_hat_base` (template-base MLPs), `dc_hat_dT` (the `ΔT`-only baseline with its exposed-fitted noun vector, carried in the lock), `dc_hat_rank1` (descriptive), and the pair-level scalars (noun-mean, per-noun slope inputs).
- Scoring output: `Y1` / `Y2` / `Y3` each with `population` (the explicit counts), `precondition`, the statistics of their table, `label`; `joint_fresh_nouns` descriptive; `comparators`; `identities` with the measured value of every frozen check; `outcome.label = "Y1 | Y2 | Y3"`.

## File map

- `src/neural_decompiler/readout_decompilation.py` (new; imports `block_routing`, `block_concentration`, `head_pattern`, `attention_patterns`, `layer_correction`, `read_assembly`, `plural_mechanism`).
- `src/neural_decompiler/layer_correction.py`: no change (its `LayerWeights.from_model(model, layers=(3, 4, 5))` already covers blocks 3–5 — a test asserts the layer-3/4/5 MLP weights match the checkpoint).
- `experiments/020-readout-decompilation/run.py`, `confirmation-v1.json` (frozen, tokenizer only), `README.md`.
- `tests/test_readout_decompilation.py` (unit, tier A/B), `tests/test_experiment_020_runner.py` (runner, tier B as `current`).
- `tests/conftest.py`: `CURRENT_EXPERIMENT = "020"`.
- `.gitignore`: `outputs/experiment-020/*`.

## Testing (the tiered architecture of `296d7e6`)

- During implementation: `uv run pytest --tier B` (every unit test plus the 020 runner test; historical runners excluded).
- Before `explore`, before `lock` and before `confirm`: `HF_HUB_OFFLINE=1 uv run pytest --tier C` on a clean tree at the gated commit.
- **Tier D only if** this work changes shared infrastructure that historical runners depend on. The plan's only shared-file change is `tests/conftest.py`'s `CURRENT_EXPERIMENT` (which moves `test_experiment_019_runner.py` from `current` to `historical` — a classification change, not a behavioural one) and, if it proves necessary, a backward-compatible addition to `layer_correction.py`. If `src/neural_decompiler/*.py` outside `readout_decompilation.py` is touched at all, run `uv run pytest --tier D` before the `explore` gate and record the result.
- The runner test uses the tiered fixtures of Phase 1 (`fake_world` module-scoped, `explored_world` for post-lock states) so that it stays within the tier-B budget.

---

### Task 1: Module, program, identities, pool and confirmation builder

- [ ] Commit this plan; `.gitignore`; `CURRENT_EXPERIMENT = "020"` in `tests/conftest.py` (tier-membership test updated).
- [ ] `readout_decompilation.py`: frozen constants (floors, tolerances, populations, the fresh-noun and fresh-cue/frame lists' digests), `build_pool_020`, `NounSet` with the scorable rule, `FrameState020` and its capture, `ReadoutProgram` (blocks 3–5 with the retained L05 frozen-row second transport at `p_c` and `p_t`, MLPs at the frame's reference operating points, layer-4 attention frozen), `level1`, the noun read, the four comparators, and every identity with its frozen tolerance and `E₁`'s exact formula.
- [ ] `tests/test_readout_decompilation.py`: the identities on the small fake; the Level 0 program's provenance (a poisoned cue capture changes no prediction; a prediction computed without any cue prompt is bit-identical); `E₁`'s formula on planted vectors including a near-zero `Δh6^meas` (the floor `1e-12` applies, no explosion); the population masks (no fresh noun in Y1/Y2, no exposed noun in Y3); the scorable rule on `peach`; the scoring floors and every outcome branch on synthetic tables; the comparators' standing fields; the confirmation policy (24 cues, 18 frames, 24 nouns, the three manifest classes with their key counts).
- [ ] Commit; freeze `confirmation-v1.json` (tokenizer only, no model); commit.

### Task 2: Exploration, lock, two-stage confirmation, scoring, runner

- [ ] `run_exploration`: the 108 reference captures; the exposed pool's measured contrasts and Level 0 predictions; the identities with their measured values; the comparators; the descriptive tables; the fitted `d_noun`; the fresh-noun absence assertion.
- [ ] `lock_predictions` and `build_candidate_lock` (no forward pass), `render_predictions`.
- [ ] `stage_one`, `assert_stage_one_digest`, `assert_no_target_pair_executed`, `stage_two`, `score_confirmation` with the frozen populations and floors, `render_report`.
- [ ] `experiments/020-readout-decompilation/run.py` with the six phases and the 017–019 refusal rules (one-shot phases, incident gating, changed-scientific-path refusal, runtime equality, A0 contract test).
- [ ] `tests/test_experiment_020_runner.py`: the full state machine on the fake with the tiered fixtures; the no-forward-pass lock; the stage barrier with a spy that records every executed token id in stage 1; the tamper refusals; the incident semantics (an identity beyond tolerance is an incident; an invalid fresh frame is **not**); one-shot semantics.
- [ ] `uv run pytest --tier B`; commit.

### Task 3: Verification, Tier A, stop

- [ ] Independent implementation review against the spec — the data-flow table above line by line, then the floors, the populations and the tolerances; fix; commit.
- [ ] `HF_HUB_OFFLINE=1 uv run pytest --tier C` on the clean gated commit (plus `--tier D` if shared source changed); record the result.
- [ ] `explore` once; `lock`; the read-only post-lock check (byte identity, only the exempt files changed since the explore commit, the predictions and the reference states reproduced with zero difference, the runtime equal to the explore record, no fresh prompt in the ledger); present the candidate lock and predictions. **Stop**: the user's lock commit and the reviewer's sign-off precede `confirm`.

### Task 4: Confirmation and closure

- [ ] `confirm` once (two stages); `report`; evidence copies; experiment README; root README registry; claim C002 revisited against the outcome; memory.

## Stopping conditions

- After Task 1's freeze and after Task 2's tests, report and continue.
- **Hard stop before `confirm`** (Task 3's end): the lock must be installed and committed by the user and signed off by the reviewer.
- Any incident stops the phase and is reported with its commit; no phase is retried in this protocol version.
