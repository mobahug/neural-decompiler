# Experiment 018 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Experiment 018 design (revision 2, commit `ab9e53a`): Experiment 017's chain with channel D replaced by its masked form `D_S` — the neurons of a locked subset `S` at the frame's own operating point, every other neuron at the template base (the Experiment 012 base at `p_c`; a block-2 base at `p_t` for the coordinated template locked at `explore`) — with the subset chosen once by the frozen read-unit ranking rule on the 9636 licensed exposed pairs from Level 0-F's *predicted* arriving change (no measured quantity), the ladder `S_0 … S_2048`, three seeded random 256-subsets and `bottom_256` as controls, the closure fraction `κ` (unclipped; undefined below a `0.05` gap) as the statistic, scored on unseen cues (Y1), twelve unseen frames (Y2: split guard and no-harm guard), the single-neuron account committed for rejection (Y3, pooled, pair-weighted, per-set descriptives), and the within-frame pattern-term question from the reference rung as the secondary Y4; the reference rung `S_2048` reproduces Experiment 017's Level 0 (I8) and the whole 017 record is replicated from a committed extract.

**Architecture:** One module `block_concentration.py` reusing `head_pattern.py` (`FrameState017`, `capture_frame_017`, `measure_pair`, `analyse_pair_017` for the identities I4–I7, the 017 Level 0 and its rungs, `HeadChainModel.head`, `head_terms`, `layer3_bases`, the confirmation-builder helpers, the results-state and lock machinery patterns), `frame_channels.py` (`FrameChannelModel.predict_channels` for Level 0-F at `p_c`), `layer_correction.py` (`LayerWeights.hidden` for the own/template hidden deltas), `attention_patterns.py`, `attention_paths.py`, `read_assembly.py`, `neuron_feature.py`, `plural_mechanism.py`; a `MaskedChainModel` (the 017 `HeadChainModel`, the locked `p_t` base and a named mask table) whose `upstream_parts` computes each pair's per-position arriving change and own/template hidden deltas once and whose rungs differ only by the mask (S2048 ≡ 017 Level 0 to `1e-9`; S0 at `p_c` ≡ 017's `−D`); the ranking (`score_j`, `S_n`, the seeded controls, `bottom_256`, the per-frame rankings and overlaps, the frame-specific oracle) as pure functions of the locked states, the weights and the exposed `ΔE`; a committed extract of Experiment 017's per-pair `F`, `Π`, `ΔT`, `c_L`, row and Level 0 predictions (9636 pairs) with the digests of its twelve fresh frames' stage-1 states; the confirmation builder with the frozen lists and twelve frames; a runner with phases `validate`, `freeze-confirmation`, `explore`, `lock`, `confirm` (two stages), `report`.

**Spec:** `docs/superpowers/specs/2026-09-20-experiment-018-block2-concentration-design.md` (revision 2). The spec wins over this plan.

## Global constraints

- Pinned model and runtime; seeds `20260916` / `20260924`. Nothing outside the 231 exposed tokens, 78 exposed frames and 80 nouns is executed before `confirm`; the lock phase runs no forward pass; no fresh cue prompt before its frame's stage-1 table is digested.
- **Boundary:** every rung receives `ΔE`, the weights, the locked axes, read and bases, the locked subsets and the frame's reference run only; the ranking reads no measured quantity and no confirmation token or frame (tests poison both); the poisoned-capture test at `p_c` and `p_t` covers every column of every rung.
- **Objects:** per rung `ĉ_L = r(Δ̂x₃(p_c) − ΔE) / D_T`, `F̂`, `Π̂`, `ΔT̂ = F̂ + Π̂`, the row; `ΔT̂_frozen = F̂(S_2048)`; `κ_X(S) = (R²_X(S) − R²_X(S_0)) / (R²_X(S_2048) − R²_X(S_0))`, unclipped, `None` when the denominator is below `0.05`.
- **Floors:** Y1/Y2 `κ_{c_L}(S_256) ≥ 0.70`, `κ_Π(S_256) ≥ 0.50` on pooled pairs; Y2 split guard cue-final `≥ 0.60`, coordinated `≥ 0.40`; no-harm `R²_{c_L}(S_256) ≥ R²_{c_L}(S_0) − 0.05` per valid fresh frame; precondition: reference rung `c_L ≥ 0.98`, rows `≥ 0.95`, `ΔT ≥ 0.95`, gaps (`c_L`, `Π`, and for Y2 each guard split's `c_L`) `≥ 0.05`, 16 scored tokens, Y2 ≥ 6 valid cue-final and ≥ 3 valid coordinated frames, tokens scored with ≥ 3 valid frames; Y3 pooled both sets `κ_{c_L}(S_1) < 0.50` and `κ_{c_L}(S_256) − κ_{c_L}(S_1) ≥ 0.25`; Y4 per valid fresh cue-final frame with reference `ΔT R² ≥ 0.95` (≥ 6 of 8 evaluable): frozen `ΔT R² < 0.90` in every evaluable frame. Identities I4–I7 as Experiment 017; I8 `1e-9`; extract replication `1e-6`.
- Every constant is a named constant tested against the spec.

## File map

- `src/neural_decompiler/block_concentration.py`
- `experiments/018-block2-concentration/run.py`, `confirmation-v1.json` (frozen), `inherited/experiment-017-pair-extract.json`, `preregistration-lock.json` and `predictions.md` (installed by hand), `README.md`, `evidence/`
- `tests/test_block_concentration.py`, `tests/test_experiment_018_runner.py`
- `.gitignore`: `outputs/experiment-018/*`

---

### Task 1: Plan, extract, model, ranking, confirmation builder

- [ ] Commit this plan; `.gitignore`; the Experiment 017 extract (from the closed 017 results state; no model run); `block_concentration` (constants, pool 018, `MaskedChainModel`, the ranking and controls, the `p_t` base, `analyse_pair_018` with I8 and the rung table, builder/validator/freeze/load). Tests: S2048 ≡ 017 Level 0 and S0 ≡ `−D` at `p_c` on the fake; the ranking's invariance to poisoned measurements and a poisoned confirmation set; the controls' contract (256 unique indices, exact seed reproduction, overlaps recorded, no redraw); the boundary with a poisoned capture at both positions for every rung; builder policy; extract.
- [ ] Commit; freeze `confirmation-v1.json`; commit.

### Task 2: Exploration, lock, two-stage confirmation, scoring, runner

- [ ] `run_exploration` (states, `p_t` base, ranking and subsets first, then the 9636 pairs re-measured and replicated), `lock_predictions`, lock build/validate/reproduce, `stage_one` (with the per-frame rankings, overlaps and the oracle), `stage_two`, `score_confirmation` (`κ`, preconditions incl. split gaps, Y1, Y2 with both guards, Y3 pooled with per-set descriptives, Y4 from the reference rung, the predeclared orderings, ladder, per frame), `outcome`, `render_report`, `run.py`. Tests: synthetic tables for every floor, guard and precondition branch (a non-evaluable gap, a split below the gap, a frame below the reference-rung condition, a frame at exactly `0.90`, κ outside `[0, 1]` unclipped); the stage barrier; the state machine on the fake; the lock phase performs no capture; incidents.
- [ ] Commit.

### Task 3: Verification, Tier A, stop

- [ ] Full suite; independent review against the spec; fix; commit.
- [ ] `explore` once; `lock`; present the candidate lock and predictions. **Stop**: the user's lock commit and the reviewer's sign-off precede `confirm`.

### Task 4: Confirmation and closure

- [ ] `confirm` once (two stages); `report`; evidence; README; root README; memory.
