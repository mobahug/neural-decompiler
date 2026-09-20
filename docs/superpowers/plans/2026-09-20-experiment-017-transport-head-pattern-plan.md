# Experiment 017 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 017 design (revision 2, commit `4a2e409`): predict `L03.H04`'s own attention-row change at the transport position and its end-to-end output change `ΔT` from the decoded upstream state — Experiment 016's Level 0-F at the cue position, block 2's MLP at the frame's operating point (channel D), one propagation step to `p_t` in coordinated frames, and the head's own query/key/value changes reduced as Experiment 016 reduced layers 1–2 at locked layer-3 template bases — scored on unseen cues (Y1) and twelve unseen frames (Y2, per-frame row guard), the frozen-pattern account committed for rejection on the cue-final pairs (Y3), the exact-head rung, the two ablations and the identities I4–I7 descriptive/checked, and the leak-proof boundary tested with a poisoned capture at both positions.

**Architecture:** One module `head_pattern.py` reusing `frame_channels.py` (`FrameChannelModel.predict_channels` for the layers-1–2 change at `p_c`, its `Channels` switches for the Level 1 recovery, `analyse_pair_016` for the inherited quantities), `attention_patterns.py` (`LayerProgram`, `ReferenceRow`, row statistics, `_max_numeric_difference`), `attention_paths.py` (`FrameState013`, `table_digest`, `assert_stage_one_digest`), `layer_correction.py` (`LayerWeights.delta_out`), `read_assembly.py` (`capture_reference`, `measure_token_010` with the layer-3 extra sites, `head_change`); a `FrameState017` (the 013 state plus the layers-1–3 residuals at every position `≤ p_t` and the head's reference row); a `HeadChainModel` with a `Variant` of switches (`upstream` channels, `block2_own`, `head_channels`, `exact_head`) so that Level 0, the exact-head rung, the two ablations and Level 1 are one code path, with an independent `exact_chain` for the identities and the Level 1 recovery; layer-3 template bases computed and locked at `explore`; a committed extract of Experiment 016's per-pair `c_ΔA` (measured and predicted), sixteen self-weight changes, `c_L`, `c_M`, `c_H` and decoded `c_L` (7764 pairs) with the digests of its twelve fresh frames' stage-1 states; the confirmation builder with the frozen lists and twelve frames; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm`, `report`, `confirm` in two stages.

**Spec:** `docs/superpowers/specs/2026-09-20-experiment-017-transport-head-pattern-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; seeds `20260916` / `20260924`. Nothing outside the 207 exposed tokens, 66 exposed frames and 80 nouns is executed before `confirm`; the lock phase runs no forward pass; no fresh cue prompt before its frame's stage-1 table is digested.
- **Boundary:** Level 0 receives `ΔE`, the weights, the locked axes, read and bases and the frame's reference run only (residuals at layers 1–3 for every position `≤ p_t`); every `σ'` is of a predicted residual; a test poisons the patched capture at `p_c` and `p_t` (residuals, rows, head output) and checks that every table column is unchanged.
- **Objects:** `ΔA_H(p_t, k)`, `Π = ⟨Σ_k ΔA_H o'(k), d̂_T⟩`, `F = ⟨Σ_k A_H [o'(k) − o(k)], d̂_T⟩`, `ΔT = F + Π` (I7); predicted counterparts on predicted values only; raw units along `d̂_T`.
- **Floors:** row entry `R² ≥ 0.95`; `Π` token means Spearman `≥ 0.90`, `R² ≥ 0.90`; `ΔT` pairs `R² ≥ 0.95` and token means `R² ≥ 0.95`; Y2 guard: every valid fresh frame's row entry `R² ≥ 0.90`; Y3 on the pooled cue-final pairs of both sets: frozen `ΔT` `R² < 0.95` and `≥ 0.05` below Level 0's; coordinated split descriptive. Preconditions 3 / 16 / 8. Identities: I4 `1e-4` relative, I5 `1e-4`, I6 `1e-3`, I7 `1e-4`; Level 1 recovery `1e-9`.
- Every constant is a named constant tested against the spec.

## File map

- `src/neural_decompiler/head_pattern.py`
- `experiments/017-transport-head-pattern/run.py`, `confirmation-v1.json` (frozen), `inherited/experiment-016-pair-extract.json`, `preregistration-lock.json` and `predictions.md` (installed by hand), `README.md`, `evidence/`
- `tests/test_head_pattern.py`, `tests/test_experiment_017_runner.py`
- `.gitignore`: `outputs/experiment-017/*`

---

### Task 1: Plan, extract, model, confirmation builder

- [ ] Commit this plan; `.gitignore`; the Experiment 016 extract (from the closed 016 results state; no model run); `head_pattern` (`Variant`, `FrameState017`, capture and measurement with the layer-3 sites, `exact_chain`, `HeadChainModel`, `analyse_pair_017`, pool 017, builder/validator). Tests: I4–I7 and the Level 1 recovery on the fake incl. a coordinated frame; the boundary with a poisoned capture at both positions; row/table shapes; builder policy; extract.
- [ ] Commit; freeze `confirmation-v1.json`; commit.

### Task 2: Exploration, lock, two-stage confirmation, outcome, runner

- [ ] `run_exploration` (states first, layer-3 bases, then the 7764 pairs), `lock_predictions`, lock build/validate/reproduce, `stage_one`/`stage_two`, `score_confirmation` (floors, guard, Y3 cue-final, split, ladder, per frame), `outcome`, `render_report`, `run.py`. Tests: synthetic tables for every floor branch, the guard and the cue-final Y3; the stage barrier; the state machine on the fake; the lock phase performs no capture; incidents.
- [ ] Commit.

### Task 3: Verification, Tier A, stop

- [ ] Full suite; independent review against the spec; fix; commit; push.
- [ ] `explore` once; `lock`; present the candidate lock and predictions. **Stop**: the user's lock commit and the reviewer's sign-off precede `confirm`.

### Task 4: Confirmation and closure

- [ ] `confirm` once (two stages); `report`; evidence; README; root README; memory.
