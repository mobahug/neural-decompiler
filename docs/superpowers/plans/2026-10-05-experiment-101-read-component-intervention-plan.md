# Experiment 101 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan revision 3** (2026-10-05). It implements the frozen design, revision 3 as the reviewer decided it on 2026-10-02
(`docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md`, commits `a87e284` and
`516df20`). Revision 2 applied the independent plan review of 2026-10-05, which found revision 1 NOT READY with two
BLOCKERs, nine SHOULDs and fourteen NOTEs. Revision 3 applies the same reviewer's re-check of revision 2, READY WITH
CHANGES with no BLOCKER, four SHOULDs and five NOTEs. § Changes maps every finding to its change. One point needs the
reviewer's decision before implementation: the form of G4 (§ Decision needed). This is a plan only: there is no code, no
runner, no freeze, no calibration, no lock and no prompt. No model runs until the implementation and its independent
review are complete (design §14).

**Goal.** Implement the decided design:
- **The intervention.** Exact edits of the residual entering layer 3 (`RESID_PRE.L3`) at the cue position `p_c`, in 36
  new coordinated frames. Each edit moves only the read coordinate `r(x) = ⟨x, w_c⟩` of `L03.H04` (removal,
  installation, a doubled change, a reversed change, a half dose), or moves an equal centred length along the frame's
  own non-read change or along isotropic directions (the controls). Two further kinds of edit:
  - the whole-position swaps (PC-SWAP), and the NOOP;
  - the head-only arms, which replace `L03.H04`'s output at `p_t` instead.
- **The predictions.** The completed program (023's chain and 020's readout) predicts every main arm's `Δc`. The
  prediction table is computed from the stage-1 reference states and the weights alone. It is committed by the user, and
  its digest is checked, before any target prompt runs.
- **The outcome.** Families R and I (by Holm), then D, then P. The labels and the scope sentence are those of design §7,
  under the precondition and the gates of design §8.

**Architecture.** One new module, `src/neural_decompiler/read_intervention.py` (imported as `ri`), and one runner,
`experiments/101-read-component-intervention/run.py`. Like 025, the module *calls* frozen modules and never edits them.
Every called module is pinned by git blob, and the runner refuses any drift.
- **The frozen chain and readout:**
  - `upstream_localization.py` (`ul`): `load_frozen_inputs`, `ModelPrograms`, `reference_rows_017`, `pair_context`,
    `compose_dx3`, `contrast_of`, `i3_error`, `write_table`, `read_table`, `verify_table_index`, `extract_exclusion`;
  - `block0_completion.py` (`b0c`): `P1_MASK`, `FULL_MASK`, `exposed_units`, `exclusion`, `verify_022_inputs`, and the
    patterns of `stage_one`, `stage_one_digest`, `verified_y2_blocks` and `barrier`;
  - `readout_decompilation.py` (`rd`): `capture_frame_020`, `locked_state`, `state_from_locked`, `state_digest`,
    `NounSet`, `ReadoutProgram` (`blocks_3_to_5`, `level1_detail`, `contrast`, `ln_final`), `head_output_change`,
    `logit_identity`, `level1_error`, `READOUT_IDENTITY_TOLERANCE`, `LOGIT_IDENTITY_TOLERANCE`, `RUNTIME_SEED`,
    `load_results_state`;
  - `readout_calibration.py` (`rc`): `verify_020_closure`, `json_safe`, `content_digest`, `file_sha256`,
    `tensor_digest`, `program_blob_sha1`, `LOCK_DIGESTS`, `LOCKED_STATE_TOLERANCE`, and the patterns of `_assert_keys`
    and `_difference`;
  - `read_assembly.py` (`ra`), `cue_suppression.py` (`cs`, whose `Pool008` is 020's pool) and `attention_paths.py`
    (`ap`), which lie on the stage-1 capture path;
  - `attention_patterns.py` (`atp`): `ReferenceRow` and `LayerProgram`, for the head recomputation.
- **Measurement:** `plural_mechanism.py` (`pm`): `run_patched`, `capture_prompt`, `Prompt`, `Frame`, `_build_new_frame`,
  `ReplacementSource`, `Weights`, `exact_layer_norm`, `contrasts`, `canonical_json`, `sha256_text`.
- **The head:** `head_transport.py` (`ht`): `HeadWeights`, `HEAD_KEY`, and the P2 value-only form in `ov_levels`.
  `head_pattern.py` (`hp`): `_row_and_values`, `HEAD_INDEX`.
- **025's runner machinery,** reused as is: `cue_rotation.py` (`cr`): `sha_gaussians`, `project_off`, `unit`,
  `tagged_key`, `untagged`, `assert_ledger_isolated`, `prior_nouns`, and the patterns of its runner (`_record_incident`,
  `_record_phase_incident`, `_preserve_partial`, `_descriptives`). `readout_routing.py` (`rr`): `write_state_atomic`,
  `save_durably`, `fsync_directory`, `parameters_digest`, `embedding_digest`, `exclusion`, `target_noun_form_ids`.
- **Newly pinned:** `interventions.py`, `components.py` and `capture.py`, which 025 left to the `src/` path rule, and
  `read_assembly.py`, `cue_suppression.py` and `attention_paths.py`.

## Global constraints

- **No frozen module is edited.** New code lives only in `ri`, the 101 runner, the 101 tests and the files listed in the
  file map. `ri.assert_frozen_blobs()` runs before anything is loaded.
- **Numerics.**
  - Geometry and edits are computed in float64 from float32 captures cast to float64.
  - Each replacement is the float32 cast of its float64 vector, with exactly the captured slice's shape (`[1, 1, 512]`
    at `RESID_PRE.L3`, `[1, 1, 1, 512]` at `L03.H04`).
  - Every outcome contrast uses `rd.NounSet.contrasts` (float64) over the 79 scorable nouns, for runs and references
    alike. The only use of `pm.contrasts` (float32) is G6's comparison with the stage-1 `c_ref`, which
    `rd.capture_frame_020` builds from it.
  - Canonical digests use `rc.content_digest`, `rc.tensor_digest` and `pm.canonical_json`.
  - A non-finite number is never written as such. It is written as `null`, with a sibling key `<name>_nonfinite` holding
    `"-inf"`, `"+inf"` or `"nan"` (`ri.encode_nonfinite`, with `rc.json_safe` as the final net).
- **Phase discipline.** The phases are `validate`, `freeze`, `calibrate`, `lock`, `stage1`, `confirm` and `report`.
  - `freeze`, `lock`, `stage1` and `confirm` each run once. `report` may re-render.
  - `calibrate` may rerun only after a recorded calibrate incident, at a new commit that changes no scientific path, as
    in 024 (`rr.assert_phase_allowed`).
  - A ledger is written before any fresh prompt. State writes are atomic (`rr.write_state_atomic`). An incident never
    coexists with a result.
- **Three failure classes.**
  - A **refusal** (`PhaseError`, exit 1) records nothing and may be retried once its cause is fixed: a missing,
    untracked or uncommitted file, a dirty tree, a lock that fails `validate_lock`, a runtime, version, blob or
    input-pin mismatch, an isolation failure before any write.
  - A **stop for review** (exit 3) writes no phase file: a freeze shortfall or deviation, a calibration revision
    trigger.
  - An **incident** (exit 2) is recorded, with its commit, and the phase is refused from then on; a further attempt
    needs a new protocol version. Incidents are reserved for the cases listed under § Incidents, stops and refusals.
- **No target result before the prediction freeze.** No code path, test or script before `confirm` may run a 101 frame
  prompt other than the 36 stage-1 reference prompts. None may compute an arm outcome on a 101 frame. The tier-C guards
  and the guarded launchers enforce this (§ The prediction freeze).
- **Environment.** `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`, the repository's `.venv`, and 4 torch threads. Never
  `uv run` or `uv sync`. The runtime and versions must equal 020's explore record, as in 025.
- **An operator prerequisite.** `ul.load_frozen_inputs` always verifies 020's closure, so every phase needs
  `outputs/experiment-020/results.json`. Before any phase, the operator restores it from the committed archive
  `experiments/023-block0-completion/evidence/archive/experiment-020-results.json.gz` (sha256 `69d726ac…`), with the
  command that folder's README gives. The runner's existing check refuses any file whose sha256 is not `da63b8c2…`, so
  no new code writes into another experiment's outputs.

## Constants (a tier-A test pins every one)

- **Identity:** `EXPERIMENT = "101"`; `EXPERIMENT_DIR = "experiments/101-read-component-intervention"`; `DESIGN = {path,
  revision 3, commit 516df20}`; `PLAN = {path, revision, commit}`.
- **`FROZEN_BLOBS`**, the blobs at `516df20`, checked against `git ls-tree` by `ri.assert_frozen_blobs()`:

  | module | blob |
  |---|---|
  | `upstream_localization` | `465856962aa380747d1a4f1338d1d2762d03c9f9` |
  | `block0_completion` | `16d310fc7fab5599ec83b8bd8162fb9613f8dad2` |
  | `readout_decompilation` | `caa73b40192f4c910dc63371bd19db75a3258339` |
  | `readout_calibration` | `9107da975128d9b0383f3b346695675104e4cde9` |
  | `plural_mechanism` | `d39da8a8d9931005d411258bcddbb7f9beed35e4` |
  | `readout_routing` | `e6cb37767d1d06c6ff40804a88eab569723afdb5` |
  | `cue_rotation` | `622aa832ae29047c3060d989928236320374e227` |
  | `models` | `b1c6f03379af7e0918d0d1a6460a264651603fb2` |
  | `layer_correction` | `04df0cc9a20093cc48ee5ef62da7f206bf1a3186` |
  | `head_pattern` | `386682fe0a47d093dfbee2507afc1df61811ab15` |
  | `frame_channels` | `c95d6fb4c98b8069c26ec46e9d8f85eb927bb9e8` |
  | `attention_patterns` | `3f5acd65397bd11723ba6fc2ec4253f1d5c81943` |
  | `encoding_read` | `cab99c942de970332e726a5626b584242d4cf600` |
  | `head_transport` | `936093a4e82c83e299a65a7c85529a7226f80c95` |
  | `read_assembly` | `4820191e0e067b382b8f8807279405a7589f7059` |
  | `cue_suppression` | `21253ee61c04796966dec4733909476a09404c0c` |
  | `attention_paths` | `42f8d1588f1f960f60498854a9ccb911bc4e427a` |
  | `interventions` | `3a1e4186aeb3c8ae9625af336e8da7d2818505c7` |
  | `components` | `28af3f3d8ba5512b6850f485f85ff88983bed24e` |
  | `capture` | `4a4a78880b1c2ee410486d5bcfc1993ba2e9d316` |

  Every other module under `src/` is covered by the scientific-path rule after the lock.
- **Pinned inputs** (file and content digests; refused on drift):
  - 011's lock (`rc.LOCK_DIGESTS["lock_011"]`), for `read_weight` and `axes_vectors.T`;
  - 020's results (file sha256 `da63b8c2…`, state `2e5485dc…`);
  - the committed confirmation files of 020, 022, 023, 024 and 025, for the forbidden set and the freeze's exclusions;
  - the 023 archive (gz sha256 `69d726ac…`).
- **Cues and prompt labels.** `REF = ("one", 581)` and `PLURAL = ("two", 767)`; the secondary stratum is `("three",
  1264)` and `("many", 1142)`. In 101's frames the prompt labels are `ref` (every reference-prompt run, as the lineage
  code labels reference prompts: `rd.Confirmation020.reference_prompt`, `ul.Confirmation022.reference_prompt`), `two`,
  `three` and `many`. The calibration's prompts carry 020's own labels (§ Phases).
- **Frames:**
  - `TEMPLATE = "coordinated-adjective"`;
  - `FRAME_CANDIDATES`, the 60 texts of design §11 in order;
  - `N_FRAMES = 36`, and `P_C_RANGE = 4..7`;
  - `EXPECTED_PICKS`, candidates 1–36, the tokenizer outcome both reviews reproduced. A different outcome is a
    `FreezeDeviation`;
  - `NUMBER_WORDS`, a fixed list of English number words (zero to twenty, the tens, hundred, thousand, million, billion,
    dozen, half, single, double, both, several, few, many, couple, pair).
- **Arms and order.** `CUE_ORDER = ("two", "three", "many")`. Within a frame, the runs follow `CUE_ORDER`, and each
  cue's arms follow its fixed order:
  - for `two`: `REF`, `PL`, `NOOP`, `NEC-READ`, `SUF-READ`, `NEC-OFF`, `SUF-OFF`, `NEC-AMP`, `SUF-NEG`, `NEC-HALF`,
    `SUF-HALF`, `PC-SWAP-PL`, `PC-SWAP-REF`, `NEC-ISO-1` … `NEC-ISO-8`, `SUF-ISO-1` … `SUF-ISO-8`, `HEAD-NEC`,
    `HEAD-SUF`, `HEAD-NEC-V`, `HEAD-SUF-V` (33 runs);
  - for `three`, then `many`: `PL`, `NEC-READ`, `SUF-READ`, `NEC-OFF`, `SUF-OFF`, `NEC-AMP`, `SUF-NEG`, `NEC-HALF`,
    `SUF-HALF`, `PC-SWAP-PL`, `PC-SWAP-REF` (11 runs each).

  That is 55 runs per frame. `K_ISO = 8`, with `ISO_TAG = "101|iso|{frame_id}|{j}"`, always formatted with keywords.
- **Run tags.** `S1` is the stage-1 reference capture and `REF` the stage-2 plain reference run. Every other tag is
  `ARM@cue`, such as `PL@two`, `SUF-READ@three` or `HEAD-SUF-V@two`. The cue qualifier is required: the stratum's
  `SUF-*` and `PC-SWAP-REF` arms run the same reference prompt as `two`'s.
- **The statistics:**
  - `N = 36`; `WEAK_SHIFT = −1.0` nats; `PRECONDITION = 30`;
  - `HOLM_ALPHA = 0.025`; the thresholds 26 and 25, derived in a tier-A test from the exact rationals `P(Bin(36, ½) ≥
    k)`, with tails 0.005665… and 0.014408…;
  - `MEDIAN_FLOOR = 0.05`; `D_THRESHOLD = 25`; `P_THRESHOLD = 27`;
  - `TAU_RANK = 33` (nearest rank, 90th percentile of 36), `TAU_CAP = 0.20`, `TAU_FLOOR = 0.02`;
  - `HEAD_SHARE_GUARD = 0.05`;
  - `BOOTSTRAP = 10_000`, seed `20261001`, interval ranks 250 and 9,750;
  - `WORST`, the value a weak frame takes in a median: `−∞` for `N`, `S`, `A_neg`, the head shares and the isotropic
    margins; `+∞` for `A_amp`;
  - `DIRECTION_FREE`, the descriptive quantities with no success direction (the midpoint fractions, the PC-SWAP
    fractions, the additivity `F_rem + F_ins`, P's errors): weak frames are excluded from their summaries, with their
    count.
- **Tolerances** (design §8; G4's form awaits the reviewer's decision):

  | gate | check | tolerance |
  |---|---|---|
  | G1 | `w_c` from the weights against the centred 011 `read_weight` | 1e-9 max-abs |
  | G1 | `v_j` unit, mean-zero, orthogonal to `w_c`; the basis (`𝟙/√512`, `ŵ`) orthonormal (float64) | 1e-12 |
  | G1 | the `w_c` and `v_j` digests recomputed at `stage1` and `confirm` against the lock | bit for bit |
  | G2 | landing: the captured slice equals the replacement, zero change outside it (`pm.run_patched`) | exact |
  | G2 | realized `r(x′)` against its target | 1e-4 |
  | G2 | centred length of `e = x′ − x` against its intended length | 1e-4 relative |
  | G2 | read arms parallel: `‖e_c − ⟨e_c, ŵ⟩ŵ‖/‖e_c‖`; off and isotropic arms orthogonal: `|⟨e_c, ŵ⟩|/‖e_c‖` | 1e-4 |
  | G2 | the float32 `û_off`: mean, norm − 1, `⟨·, ŵ⟩` | 1e-6 |
  | G3 | NOOP against `PL@two`, logits and each of NOOP's capture sites | bit for bit |
  | G3 | `RESID_PRE.L4(p_c) = x′ + Σ_h L03.H0h(p_c) + b_O[3] + L03.MLP(p_c)` | 1e-4 max-abs |
  | G4 | `RESID_PRE.L3(p_t)` of an edited run against the same prompt's plain run | bit for bit |
  | G4 | 023's I3 on each plain pair: the FULL composition against the measured `Δx3` | 1e-4 relative |
  | G4 | 023's I4 on each G4 run: the composition proxy's `Δĉ` against the run's ceiling | 1e-3 nats max-abs |
  | G4 | 020's Level-1 identity, in 025's contrast form, on each G4 run | 2e-2 nats max-abs |
  | G4 | 020's readout identity on each G4 run | 2e-2 nats max-abs |
  | G4 | 020's logit identity (computed on every run, gated on the G4 runs) | 2e-2 |
  | G4 | 020's normalized `E₁` | recorded, not gated |
  | G5 | the head row recomputed from the unedited inputs against the captured row | 1e-4 max-abs |
  | G5 | the head output along `d̂_T`, recomputed against captured | 1e-3 max-abs |
  | G6 | the stage-2 REF run's `pm.contrasts` against the stage-1 `c_ref` | bit for bit |
  | G6 | the stage-2 REF run's `x3(p_c)`, `x3(p_t)` and `h6` against the stage-1 state | bit for bit |
  | G6 | the table recomputed from the committed states and the weights | bit for bit |
  | G7 | isolation; exact-once accounting | exact |

- **The labels and semantics:** the nine outcome labels of design §7 and §14, the scope sentence, the claim hierarchy,
  the non-claims, and the rule that the stratum cannot rescue a failed primary result.

## The canonical computations (one implementation each)

1. **The read covector.** `rw = read_weight` of 011's lock, in float64. `w_c = rw − mean(rw)·𝟙`, `ŵ = w_c/‖w_c‖` and
   `d̂_T = axes_vectors.T`.
   - G1 recomputes `HeadWeights.ln_w ⊙ (W_V @ (W_O @ d̂_T))` from the model, centres it, and compares it with `w_c`.
   - The digests of `w_c` and of every `v_j` are written to the lock and recomputed, bit for bit, at `stage1` and
     `confirm`.
2. **The pair quantities,** per frame and cue `w`. They come from the captured float32 `x_ref` and `x_w`, `RESID_PRE.L3`
   at `p_c` in the plain `REF` and `PL@w` runs, cast to float64:
   - `Δx = x_w − x_ref`, `Δr = ⟨Δx, w_c⟩`;
   - `read = (Δr/‖w_c‖²)·w_c`, `off = Δx − read`, `off_c = off − mean(off)·𝟙`;
   - `û_off = off_c/‖off_c‖` and `s = |Δr|/‖w_c‖`;
   - frames with `‖off_c‖ < s` are flagged, and `s/‖off_c‖` is recorded.
3. **The edits,** each cast to float32. `e = x′ − x`, where `x` is the same prompt's plain residual at `p_c`:

   | arm | replacement at `RESID_PRE.L3`, `p_c` | G2 checks |
   |---|---|---|
   | `NOOP` | `x_w` as captured, the same tensor | landing only (G3 compares the run) |
   | `NEC-READ`, `SUF-READ` | `x_w − read`, `x_ref + read` | `r`, length, parallel |
   | `NEC-OFF`, `SUF-OFF` | `x_w − s·û_off`, `x_ref + s·û_off` | `r` unchanged, length `s`, orthogonal |
   | `NEC-AMP`, `SUF-NEG` | `x_w + read`, `x_ref − read` | `r`, length, parallel |
   | `NEC-HALF`, `SUF-HALF` | `x_w − read/2`, `x_ref + read/2` | `r`, length `s/2`, parallel |
   | `PC-SWAP-PL`, `PC-SWAP-REF` | `x_ref` in the `w` prompt, `x_w` in the REF prompt | `r` and centred length only |
   | `NEC-ISO-j`, `SUF-ISO-j` | `x_two − s·v_j`, `x_ref + s·v_j` | `r` unchanged, length `s`, orthogonal |

   The targets of `r(x′)`: `r(x_ref)` for `NEC-READ`; `r(x_w)` for `SUF-READ`; `r(x_w) + Δr` for `NEC-AMP`; `r(x_ref) −
   Δr` for `SUF-NEG`; `r(x_ref) + Δr/2` for both halves; the unedited `r(x)` for the off and isotropic arms; the other
   prompt's `r` for PC-SWAP.
4. **The isotropic directions.** `v_j = cr.unit(cr.project_off(cr.sha_gaussians(ISO_TAG.format(frame_id=frame_id, j=j),
   512), (𝟙/√512, ŵ)))`, for `j = 1..8`, in float64. A tier-A test asserts that the basis is orthonormal, and a
   regression digest pins one `v_j`.
5. **The head-only arms.** Each replaces `L03.H04` at `p_t` with the plain run's captured output plus an exact float64
   difference `δ`. HEAD-NEC and HEAD-NEC-V run the `two` prompt with the NEC-READ edit `e = −read`; HEAD-SUF and
   HEAD-SUF-V run the REF prompt with the SUF-READ edit `e = +read`.
   - **The inputs.** `ri.HeadInput(x3_all, p_t)` is a frozen dataclass with exactly the two fields
     `rd.head_output_change` reads (`readout_decompilation.py:1269–1280`). `x3_all` is the plain run's captured
     `RESID_PRE.L3` at positions `0..p_t`, in float64.
   - **The full variants.** `δ = rd.head_output_change(progs.readout, HeadInput, {p_c: e})`. It recomputes the row and
     the values through the exact LayerNorm; the query at `p_t` is unchanged.
   - **The value-only variants.** `δ = A_meas[p_c] · (LN(x + e) − LN(x)) @ W_V @ W_O`, 009's P2 form (`ht.ov_levels`).
     `A_meas` is head 4's captured attention row at `p_t` in the plain run, and LN is the exact layer-3 attention
     LayerNorm (`ht.HeadWeights.normalize`).
   - **G5.** From the same unedited inputs, `atp.ReferenceRow(layer 3, x3_all)` gives head 4's row and output. Both are
     compared with the **plain** run's captures: `PL@two` for HEAD-NEC and HEAD-NEC-V, `REF` for HEAD-SUF and
     HEAD-SUF-V. The row is compared with the captured `ATTN_PATTERN.L3` at `p_t`, and the output along `d̂_T` with the
     captured `L03.H04` at `p_t` along `d̂_T`. A head arm's own `L03.H04@p_t` capture is the value after its replacement
     (captures observe post-intervention values, `interventions.py:222–227`), so it is never G5's target; that it equals
     the replacement is the landing check.
6. **The program's predictions,** per frame and cue `w`:
   - The state is `rd.state_from_locked(entry, frame)`, rebuilt from the recorded locked state (stage 1's in `confirm`,
     020's in `calibrate`). Then `rows16 = ul.reference_rows_017(progs.programs, state)`, `ctx_w =
     ul.pair_context(progs, frame, state, 581, id_w, w, rows16)` and `dx̂ = ul.compose_dx3(progs, ctx_w,
     b0c.P1_MASK["coordinated"])`.
   - From `dx̂[p_c]`, compute the program's own `Δr̂`, read part, `û_off` and `ŝ`, by computation 2.
   - Build each main arm's `dx3`, with **both** keys `p_c` and `p_t` and explicit float64 zeros, as in the table of
     design §5.
   - `Δĉ(arm) = ul.contrast_of(progs, state, dx3)`, kept per noun (79) and as the mean.
   - Record `Δr̂`, `ŝ`, the predicted weak flag (`mean Δĉ_full > −1`), and the fractions `F̂_rem`, `F̂_ins`,
     `F̂_rem_off`, `F̂_ins_off`, `N̂ = F̂_rem_off − F̂_rem` and `Ŝ = F̂_ins − F̂_ins_off`, under the phase switch of
     computation 7.
7. **The measured quantities.**
   - **Per run, before its logits are reduced to a digest:** `NounSet.contrasts` over the 79 scorable nouns; the logit
     identity `rd.logit_identity(progs.readout, progs.weights, h6, logits)`; for the REF run, also `pm.contrasts`
     arranged as `rd.capture_frame_020` arranges `c_ref` (G6); the logits' `rc.tensor_digest`.
   - **Per arm:** `Δc(arm) = contrasts(arm) − contrasts(REF)`, per noun and as the mean, both from stage 2.
   - **Per frame and cue:** `F_rem`, `F_rem_off`, `F_ins`, `F_ins_off`, `A_amp`, `A_neg`, `N_f`, `S_f` as in design §4
     and §7, and the secondary fractions.
   - **One phase switch, `fractions_for_weak`,** governs the fractions of a weak frame, measured or predicted. At
     `confirm` it is off: a weak frame's fractions are not computed and are recorded as `null` with the reason `weak`
     (its outcomes are fixed by §7). At `calibrate` it is on: every frame's fractions are computed and flagged, since
     `τ` needs all 36 values (design §6), and a non-finite fraction is a revision trigger. The prediction table is
     always computed with the switch off, the `confirm` setting: at `stage1`, at the barrier and at `report` alike. The
     setting is recorded in the table's metadata, which the lock binds, so a predicted-weak frame cannot break the
     bitwise recomputation.
   - **The LayerNorm records, per residual-edit arm (design §2):** `σ(x′) = HeadWeights.scale(x′)`, `‖γ₃ ⊙ e‖/σ(x′)` and
     `‖γ_ln2 ⊙ e‖/σ(x′)`, with `γ_ln2 = progs.lw.ln_w[3]`. Pythia's parallel block feeds both LayerNorms the same `x′`.
   - **The manipulation check, per arm (design §7):** `ΔT(arm) = ⟨T(arm) − T(REF), d̂_T⟩`, with `T` the captured
     `L03.H04` at `p_t`, and its ratio to `ΔT(PL@w)` for the arm's cue.
8. **The statistics.**
   - **Weak frames.** A frame is weak for cue `w` when that cue's `Δc_full > −1` (strict). A weak frame counts as a
     failure in every count, enters every median as `WORST[quantity]`, and is excluded, with its count, from the
     `DIRECTION_FREE` summaries.
   - **The strict inequalities.** A frame counts for R when `N_f > 0`, for I when `S_f > 0`, for D when `A_amp < 0` and
     when `A_neg > 0`; ties fail. The precondition counts `Δc_full ≤ −1`. P agrees when both errors are `≤
     max(τ, 0.02)`. A median passes its floor when `≥ 0.05`, and a count its threshold when `≥` it. P is not evaluable
     when `τ_rem > 0.20` or `τ_ins > 0.20`.
   - **The median** of `n` values is the mean of the `n/2`-th and `(n/2 + 1)`-th sorted values for even `n` (18th and
     19th for 36; 4th and 5th for the 8 isotropic directions), and the middle value for odd `n`; `torch.median`, which
     returns the lower one, is not used. The mean of an infinite and a finite value is the infinite one.
   - **R and I.** The p-value of each count is its exact binomial tail, as a `fractions.Fraction`. Holm runs on the two
     count p-values alone, against exactly 0.0125 and 0.025. A family passes when Holm rejects its count and its median
     is at least 0.05; a failed floor does not change the other family's threshold.
   - **D, P, the precondition and the label table,** as in design §7. P counts a frame as agreeing only when neither
     `Δc_full` nor `Δĉ_full` is weak.
   - **The secondary statistics:** the stratum per cue (R, I and D's counts and medians, weak status from that cue's own
     `Δc_full`); the isotropic counts `F_rem < F_rem_iso` and `F_ins > F_ins_iso` against each frame's median over the 8
     directions; the head arms' fractions, and the removal and installation shares under their `HEAD_SHARE_GUARD`, with
     the frames the guard excludes counted; the midpoint, PC-SWAP, the additivity; every family not reached.
   - **Bootstrap intervals** for every median, computed in the secondary step. One index matrix of shape `[10,000 × 36]`
     is drawn once, with `torch.Generator().manual_seed(20261001)` and `torch.randint(0, 36, (10_000, 36))`, and shared
     by every median. Each resample's median uses the rule above. The interval is the 250th and the 9,750th of the
     10,000 sorted resampled medians (nearest rank 2.5 % and 97.5 %).

## The frame freeze

- **What it runs.** The freeze loads the tokenizer only, at the pinned revision, and runs no forward pass.
- **The candidates.** It builds each candidate with `pm._build_new_frame(tokenizer, TEMPLATE, text, {"sg": 581, "pl":
  767}, frame_id)`, in candidate order, with frame ids `coordinated-adjective-101-01` … `coordinated-adjective-101-60`.
  It repeats the build with `three` (1264) and `many` (1142) to check that they share the prefix and suffix.
- **Eligibility** (design §11), against named, digested sources, each recorded with its path and sha256 as 022–025 did:
  - **The text is new.** The candidate's `text_template` is in none of the frame texts of `b0c.exclusion` (020's pool,
    the 006–020 confirmations, the 005 extension and 022's frames; 144 texts, which reproduce 023's committed
    `exclusion.frame_texts`) and of 023's 18 frame templates. 024 and 025 added no frames.
  - **The structure.** ` one`, ` two`, ` three` and ` many` are single tokens at `p_c` 4–7, and the adjective is one
    token at `p_t = p_c + 1`.
  - **The adjective is not an earlier cue.** Its token id is not in 024's committed exclusion
    (`exclusion.cue_token_ids`, the 351 cue ids of 005–023), not one of 024's cues (391 ids with them, equal to 025's
    committed `blocked.earlier_cues`) or of 025's cues (431 ids in all), and not a word of 025's ordered candidate lists
    (`candidates`: the adjective reserve, the new list and both forms of the ordinary reserve).
  - **The adjective is not a target-noun form.** Its id is not in `rr.target_noun_form_ids(pool)` (161 ids, equal to
    025's `blocked.target_forms`) or among the forms of 020's confirmation nouns (`cr.prior_nouns`, 48 ids).
  - **Not applied:** the `frame_tokens` set of `cr.blocked_sets` ("a token of an exposed frame"), which 025 applied to
    its cues. Twelve of the 36 picks (soft, tiny, clean, damp, wet, thin, sour, pale, neat, dry, cool, loose) are tokens
    of exposed frames, and design §11 allows that: novelty is at the level of the text. `cr.blocked_sets` is therefore
    not reused as a whole; applying it would select other frames, a `FreezeDeviation`.
  - **The adjective is not a number word** (`NUMBER_WORDS`), and no two 101 frames share it.
- **The selection** is the first 36 eligible candidates, with no other choice. Fewer than 36 is a `FreezeShortfall`, and
  a different set from `EXPECTED_PICKS` is a `FreezeDeviation`. Either writes nothing and exits 3. No frame is replaced
  after the freeze.
- **`confirmation-v1.json`** holds:
  - the 36 frames (ids, token ids, `p_c`, `p_t`, texts), with their order;
  - the eligibility outcome of all 60 candidates, with each failed rule;
  - the cue ids, labels, `CUE_ORDER` and the arm order;
  - the exclusion sources and their digests;
  - the manifests `S1-REF` (the 36 keys `frame|ref|581|S1`) and `S2-TARGET` (the 1,980 tagged keys), with their digests
    and counts;
  - `content_sha256`.

  The runner writes it once, refuses if the file exists, and re-reads it. The user commits it alone.

## Phases and control flow

| phase | requires | runs | writes | then |
|---|---|---|---|---|
| `validate` | — | nothing | nothing | exit 0 or 1 |
| `freeze` | no confirmation file | the tokenizer; no forward | `confirmation-v1.json` | the user commits it; freeze review |
| `calibrate` | the confirmation file committed and verified | 36 re-captures and 1,980 arm runs on the 36 exposed coordinated frames | `outputs/experiment-101/calibration-measurements.pt`, candidate `calibration-v1.json` | calibration review; the user installs and commits it |
| `lock` | the calibration file committed | no forward; the weights for G1 and the digests | candidate `preregistration-lock.json` and `preregistration.md` | lock review; the user installs and commits both |
| `stage1` | the lock committed | the 36 S1 reference captures only | the stage-1 files at the lock's paths; the stage-1 record | the user commits the stage-1 files |
| `confirm` | the stage-1 files committed | the barrier, then every S2 run once | `outputs/…/stage2-measurements.pt`, the result | confirmation review |
| `report` | a completed confirm, or a recorded incident | no forward; the weights for G6 | `report.md` | closure |

- **`calibrate`** (exposed, exploratory, design §6):
  - **Preconditions** (refusals): `confirmation-v1.json` is committed and verifies, and its sha256 is bound into the
    calibration record (design §14; `experiments/024-readout-routing-nounness/run.py:365–368`).
  - **Keys.** The prompts carry 020's own labels: the reference from `pool.reference_prompt(frame)` and each cue prompt
    with `pool.token_prompt(frame, name)` (`cs.Pool008`, whose `tokens` are `(name, id)` pairs). Before the first
    forward, every untagged calibration key must be in 020's exposed ledger and in no confirmation manifest (020's,
    022's, 023's, 024's or 025's), on the pattern of `rc._assert_keys`; a failure is a refusal. The run keys use the
    same tags as the confirmation (`S1`, `REF`, `ARM@cue`), 2,016 in all, and are recorded in the state as
    `calibration_executed`, with exact-once accounting.
  - **The re-captures.** Each of the 36 coordinated reference prompts is re-captured with `rd.capture_frame_020`. Its
    `rd.locked_state` is compared with `exploration["locked_states"]` at `rc.LOCKED_STATE_TOLERANCE` (1e-9), on the
    pattern of `rc._difference`. This replaces `rc.rematerialize`, which would rerun all of 020's exploration.
  - **The runs.** All 55 arms per frame, the stratum and the head arms included, run through the same code as `confirm`.
    G1–G6 are computed exactly as `confirm` will compute them (G6 against the re-captures), and every G4 quantity's
    maximum is recorded per arm.
  - **The program** uses the locked states, by computation 6.
  - **What it sets.** `τ_rem` and `τ_ins`, for `two`: the 33rd of the 36 sorted exposed values of `|F̂_rem − F_rem|` and
    `|F̂_ins − F_ins|`. Design §6 sets `τ` from all 36 exposed frames, so calibration runs with `fractions_for_weak` on:
    the fractions of a measured-weak or predicted-weak exposed frame are computed and flagged, not withheld, and all 36
    values enter `τ`. A non-finite fraction is a revision trigger. It records whether either `τ` exceeds 0.20, which
    makes P not evaluable; that is not a trigger.
  - **What it reports.** The real-swap `Δr`, `‖Δx‖`, `‖off_c‖`, `s`, `s/‖off_c‖` and `Δc_full`; the exposed N, S,
    `A_amp`, `A_neg` and P distributions; the power check (the share of exposed frames with each sign as predicted, and
    the sign-test power at that share).
  - **The revision triggers** of design §6, the only ones: a gate failure, `‖off_c‖ < s` in more than 6 frames,
    `Δc_full > −1` in more than 6 frames, a non-finite fraction. Any one gives `stopped_for_review` and exit 3, and
    writes no calibration file. The size of an exposed result is never a trigger.
- **`lock`.** It binds:
  - the frames (the confirmation file's sha256), `w_c` and every `v_j` (digests), the rule for `û_off`;
  - the program's blobs and dependencies, the calibration file's sha256, `τ_rem` and `τ_ins`;
  - `CUE_ORDER` and the arm order;
  - the stage-1 files: their paths, the states table's layout and metadata, and the table's schema with its
    `fractions_for_weak` setting (as 023 bound `y2_table`);
  - the statistics (with the median rule, `WORST`, `DIRECTION_FREE`, the strict inequalities and the bootstrap rule),
    the labels, the scope and the tolerances.

  It renders `preregistration.md` from the re-read lock, and refuses any forward pass.
- **`stage1`.** In order:
  1. The refusals: the lock validates, the tree is clean, isolation holds, the runtime and pins match, the stage-1 files
     do not exist yet.
  2. G1, and the `w_c` and `v_j` digests against the lock; a failure is a phase incident.
  3. Before the ledger, it creates the `stage1/` directory and confirms that each of the three stage-1 paths may be
     written; under the launcher this asks the launcher's write predicate (§ One-shot protections). A failure is a
     refusal.
  4. It writes the 36 `S1-REF` keys to the ledger, before the first reference prompt.
  5. For each frame in frozen order, `ri.capture_reference` runs `rd.capture_frame_020`. It asserts that the capture's
     `(p_c, p_t)` are the frozen frame's (as `b0c.stage_one` does), and stores `rd.locked_state` with its
     `rd.state_digest`.
  6. It seals the model, so that no further forward pass is possible.
  7. It computes the full prediction table from the rebuilt states (`rd.state_from_locked` of the recorded locked
     states) and the weights.
  8. It writes the states with `ul.write_table` (data and index) and the table as canonical JSON, at the lock's paths in
     the experiment directory. Step 1 refused if any of the three existed; if one exists now, after the ledger and the
     captures, that is an incident.
  9. It writes the stage-1 record: the frames, the state digests, the files' sha256s, the table's `content_sha256`, the
     lock's `content_sha256`, the commit, the noun keys, and the record's own digest (the `b0c.stage_one_digest`
     pattern). The phase ends `complete`, with the flag `awaiting_commit`.
- **`confirm`.** See the next section for the barrier. After it:
  1. It writes all 1,980 `S2-TARGET` keys to the ledger before the first S2 prompt.
  2. It runs the frames in frozen order, each frame's runs in `CUE_ORDER` and arm order, each through `ri.run_arm`.
  3. It saves the measurements durably and writes their digests to the state.
  4. It checks the accounting: `Counter(executed) == expected`, and set equality with the ledger.
  5. It reloads the measurements, checks their digests, and computes G2–G6 from them; the gate values are written before
     any is enforced.
  6. On a gate failure, it writes the `NOT_INTERPRETABLE` result (§ Gates). Otherwise it recomputes the table once more
     (G6), re-checks the committed inputs, computes the primary statistics and the label, and writes the result and the
     completed phase in one atomic write.
  7. Unless the result is `NOT_INTERPRETABLE`, an isolated step on 025's `_descriptives` pattern then computes the
     secondary statistics, the bootstrap intervals and the per-arm records. A failure there is recorded and leaves the
     result standing. Under `NOT_INTERPRETABLE` the step does not run: the saved measurements stay on disk under their
     digests for the confirmation review, and nothing derived from them is written beyond the gate values.
- **`report`** renders `report.md` from the state, after recomputing the table from the committed states and the weights
  (G6). A mismatch is recorded in the state's report block, and the report then renders the outcome as
  `NOT_INTERPRETABLE`, quoting the stored label as void; the confirm result is never rewritten. `report` may run again,
  and each run records its sha256 and time.

## The prediction freeze: how no target result can be seen first

1. **Stage 1 runs only reference prompts.** The `S1-REF` manifest holds the 36 reference prompts. The stage-1 launcher
   allows exactly those keys, each once, and only through `ri.capture_reference`.
2. **The table needs no target prompt.** It comes from the rebuilt stage-1 states and the weights, after the model is
   sealed.
3. **The cryptographic freeze.** Stage 1 records the files' sha256s, the table's `content_sha256`, the per-frame state
   digests and the record's own digest. The user commits the files.
4. **The barrier before any S2 key**, on the state re-read from disk:
   - **Refusals** (exit 1, nothing recorded, retryable): the stage-1 files are missing, untracked or uncommitted; the
     tree is dirty; the lock or the preregistration fails `validate_lock`; the runtime, versions, blobs or input pins
     differ; the forbidden-set isolation fails.
   - **Phase incidents** (exit 2, no ledger entry, `confirm` refused from then on): G1, or the `w_c` and `v_j` digests
     against the lock; the stage-1 record does not reproduce its digest, or names another lock; a committed file's
     sha256 differs from the record; the states index fails `ul.verify_table_index` against the lock's layout and
     metadata, or a state's digest differs; the recomputed table differs from the committed one in any bit; an S2 key is
     already in the ledger (the items of `b0c.barrier`).
5. **The ledger before any target.** Only after the barrier does the S2 ledger reach disk. The confirm launcher allows
   an S2 run only through `ri.run_arm`, only when the ledger is on disk and the barrier passed in this process, and only
   for a tagged key that has not run.
6. **Scoring comes last.** R, I, D and P are computed only after every run, the accounting and the gates.
7. **No earlier code computes 101 outcomes.** Tier-C tests run only on spent exposed prompts. Their guards identify a
   101 prompt by its token sequence (any of the 60 candidates with any of the four cues), not by its frame id, and
   refuse it. They also refuse every outcome function (`ri.fractions`, `ri.statistics`, and `ul.contrast_of` on a 101
   frame), as 025's `_refuse_outcomes` did.

## Ledger, isolation and accounting

- **Keys.** A prompt's key is `frame_id|label|token_id`; a run's key is `prompt.key|tag` (`cr.tagged_key`).
  - Stage 1: the 36 keys `frame|ref|581|S1`.
  - Stage 2, per frame: `frame|ref|581|REF`, 32 keys `…|ARM@two`, and 11 each `…|ARM@three` and `…|ARM@many`, on the
    prompts `ref`, `two`, `three` and `many`. That is 55 distinct keys per frame and 1,980 in all.
  - A tier-A test asserts that `S2-TARGET` has 1,980 distinct keys, each with count 1, that `S1-REF` has 36, and that
    the two sets are disjoint.
- **The forbidden set** is 025's 43,632 keys, plus the untagged forms of 025's 4,320 manifest prompts. No 101 key's
  untagged form (`cr.untagged`) may be in it. This is checked at validate, lock, stage 1 and confirm, with
  `cr.assert_ledger_isolated`.
- **Calibration** keys are checked against 020's exposed ledger and every confirmation manifest (§ Phases), and kept in
  `calibration_executed`, outside the confirmation ledger.
- **Exact-once accounting** applies separately to calibration (2,016), stage 1 (36) and stage 2 (1,980).

## Gates, as implemented

- **G1:** computation 1 and the digests, before any prompt, in `calibrate`, `lock`, `stage1` and `confirm`.
- **G2:** for each residual-edit run, the landing checks inside `pm.run_patched`, then the realized geometry by arm type
  (computation 3).
- **G3:** NOOP against `PL@two` bit for bit, in its logits and at every one of NOOP's capture sites (a subset of the
  plain run's, which adds `RESID_PRE.L3` at `0..p_t`); the pass-through identity in every residual-edit run, from
  `RESID_PRE.L4@p_c`, the eight `L03.H0h@p_c`, `L03.MLP@p_c` and `pm.Weights.attn_b_O[3]`.
- **G4,** on the G4 runs: the residual-edit runs, namely `NOOP`, `NEC-READ`, `SUF-READ`, `NEC-AMP`, `SUF-NEG`,
  `NEC-HALF`, `SUF-HALF`, `NEC-OFF`, `SUF-OFF`, `PC-SWAP-PL` and `PC-SWAP-REF` of every cue and `NEC-ISO-j` and
  `SUF-ISO-j`; and the plain `PL@w` runs, which go beyond the design's "every residual-edit arm" and are named in the
  decision. The head arms are excluded (design §8), and `REF`, whose change is zero. The form below awaits the
  reviewer's decision (§ Decision needed).
  - `RESID_PRE.L3@p_t` equals the same prompt's plain run, bit for bit.
  - **023's I3, on each plain pair.** For each frame and cue `w`: `ul.i3_error(full_w, Δx3_w) ≤ 1e-4`, with `full_w =
    ul.compose_dx3(progs, ctx_w, b0c.FULL_MASK["coordinated"])` on the stage-1 state (calibration: the locked state) and
    `Δx3_w` the measured plain change at `p_c` and `p_t`.
  - **023's I4, on each G4 run.** `max_n |ul.contrast_of(progs, state, proxy) − ul.contrast_of(progs, state, Δx3_run)| ≤
    1e-3` nats. `Δx3_run` is the run's measured change against the reference state, `{p_c: x′ − x_ref(p_c), p_t:
    x_run(p_t) − x_ref(p_t)}`, and the second term is the run's ceiling. The composition proxy is `{p_c: full_w(p_c) +
    (x′ − x_w(p_c)), p_t: full_w(p_t)}` for runs on the `w` prompt, and `{p_c: x′ − x_ref(p_c), p_t: 0}` for runs on the
    reference prompt. `x′` is the run's captured `RESID_PRE.L3@p_c`; for a plain `PL@w` run it is the run's own
    residual, so the proxy is `full_w` and I4 is 023's I4 exactly. On a reference-prompt run the proxy equals `Δx3_run`
    (the bitwise `x3(p_t)` check makes its `p_t` entry `+0.0`), so I4 holds exactly and is vacuous there: such a run has
    no upstream composition to test, and the Level-1, readout and logit identities carry the identity content.
  - **020's Level-1 identity, in 025's contrast form** (`cr.level1_gates`): `max_n |progs.readout.contrast(state,
    progs.readout.level1_detail(state, Δx3_run, l4_heads=True)["dh6"], progs.nouns)[scorable] − Δc_run| ≤ 2e-2` nats.
  - **020's readout identity:** `max_n |progs.nouns.read(progs.readout.ln_final(h6_run) −
    progs.readout.ln_final(h6_ref))[scorable] − Δc_run| ≤ 2e-2` nats (`rd.READOUT_IDENTITY_TOLERANCE`).
  - **020's logit identity:** computed on every run before its logits are reduced, gated on the G4 runs at 2e-2
    (`rd.LOGIT_IDENTITY_TOLERANCE`), recorded on the others.
  - **020's `E₁`** (`rd.level1_error` of the exact chain's `Δh6` against the measured one) is recorded per run, not
    gated.
  - **The ceiling** of every G4 run is recorded.
- **G5:** for each head arm, computation 5's row and output checks.
- **G6:**
  - the stage-2 REF run's `pm.contrasts` against the stage-1 `c_ref`, and its `x3(p_c)`, `x3(p_t)` and `h6` against the
    stage-1 state, bit for bit;
  - the table recomputed from the committed states and the weights, bit for bit: at the barrier, before the result
    write, and at the report.
- **G7:** isolation and exact-once accounting.

**What a gate failure does.**
- **After the measurements are saved,** a failure of G2's geometry, G3, G4, G5 or G6 gives the result
  `NOT_INTERPRETABLE`. The result holds the label, the failing gates, their values and locations, and nothing else. No
  fraction, count, median or family statistic is computed or written, then or later.
- **Inside a run,** a landing failure (`pm.run_patched`) or a non-finite contrast (`NounSet.contrasts`) raises, and is
  an incident.
- **Before the ledger,** a failure is a refusal or a phase incident, as classified in § The prediction freeze.

## Incidents, stops and refusals

- **Incidents** (exit 2), each recorded as in 025 (`_record_incident`, `_record_phase_incident`, `_preserve_partial`):
  - before a ledger: G1, the digest checks against the lock, and the barrier's incident items;
  - after a ledger: any exception inside the phase, a landing failure, a non-finite contrast, a failed accounting, a
    failed save or digest check, a failed result write, an interruption.
- **What an incident spends.** A stage-1 incident spends the 36 frames, since their reference prompts are in the ledger.
  A stage-2 incident spends the confirmation. Either way the report renders `NOT_INTERPRETABLE` ("an incident carries no
  result"), and a further attempt needs a new protocol version with new frames.
- **Stops for review** (exit 3): a freeze shortfall or deviation; a calibration revision trigger.
- **Refusals** (exit 1): as listed under § Global constraints and § The prediction freeze.
- **Labels, not aborts:** the precondition (`BEHAVIOR_PRECONDITION_NOT_MET`, with R, I, D and P reported descriptively);
  a gate failure after the measurements (`NOT_INTERPRETABLE`); an uninformative P
  (`READ_COMPONENT_EFFECTS_PROGRAM_NOT_EVALUABLE`).

## One-shot protections and guarded launchers

- **Phase rules.** `ri.assert_phase_allowed` extends 025's rules:
  - `calibrate` follows 024's rerun rule (`rr.assert_phase_allowed`): a rerun only after a recorded incident, at a new
    commit that changes no scientific path;
  - `stage1` needs a complete lock and runs once;
  - `confirm` needs `stage1` complete, no stage-1 incident and no confirm incident;
  - `report` needs a completed confirm or a recorded incident, and may re-render.
- **Non-production refusal.** The runner refuses any non-production configuration against the real repository.
- **The sanctioned entry points.** `ri.run_arm(model, prompt, tag, *, replacements, sources, capture_sites)` is the only
  `ri` function that calls `pm.run_patched`. It checks that the tag fits the prompt (an arm on the `w` prompt carries
  `@w`; `SUF-*`, `PC-SWAP-REF`, `HEAD-SUF*` and `REF` run the reference prompt). `ri.capture_reference(model, progs,
  prompt, tag)` is the only one that calls `rd.capture_frame_020`.
- **Launchers.** Each phase that touches the model has a guarded launcher, under that phase's `evidence/` directory, on
  025's v3 pattern:
  - a production authorization, an environment variable equal to the lock's `content_sha256` (and, for `confirm`, the
    stage-1 table's);
  - an exclusive sentinel and preflight checks N1–N4. The calibrate launcher names its sentinel by the commit, so the
    rerun the phase rules permit stays possible, and a second run at the same commit does not;
  - forward guarding: the two entry points are wrapped by identity, and every other forward path is refused. A wrapped
    call is allowed only for a `prompt.key|tag` in the phase's manifest that has not run;
  - model sealing, one model load, a write allow-list, and a log of every run key;
  - for `stage1`, the allow-list names the three stage-1 files exactly and allows `mkdir` of the `stage1/` directory.
    025's hook refuses any other repository write and any repository `mkdir` but the outputs directory
    (`launch_confirm_v3.py:138–152`), and `ul.write_table` calls `mkdir(parents=True, exist_ok=True)`, which raises the
    audit event even when the directory exists. The launcher's preflight checks the allow-list before the S1 ledger, and
    its rehearsal mode exercises the stage-1 write path under the hook.
- **What each launcher allows:**
  - `calibrate`: only the 2,016 calibration keys;
  - `lock` and `report`: the weights, and no forward pass at all;
  - `stage1`: only the 36 S1 keys, then seals the model;
  - `confirm`: only S2 keys, after the barrier and the ledger, each once.

## Testing

- **Tier A** (`tests/test_read_intervention.py`, unit):
  - the pins, the blob table and the constants;
  - the derived thresholds and tails;
  - the geometry and every edit formula on synthetic vectors, per arm type (exact read coordinates, centred lengths, the
    parallel and orthogonal metrics, `û_off`, the overshoot flags);
  - the isotropic generator: the keyword formatting, the orthonormal basis, a regression digest;
  - the freeze on a toy tokenizer: first-36 selection, no replacement, shortfall and deviation write nothing, the
    `three`/`many` check, each exclusion source, `frame_tokens` not applied (`soft` and `dry` stay eligible), the
    manifests;
  - the keys: 1,980 distinct S2 keys each with count 1, 36 S1 keys, disjoint; 2,016 distinct calibration keys;
  - the statistics: weak frames, `WORST` per quantity, the even-`n` median, the strict inequalities, Holm cases
    including a failed floor, D, P with the cap and the floor, the precondition, every label row, the head-share guards,
    the shared bootstrap matrix and its interval ranks; the `fractions_for_weak` switch, with a predicted-weak and a
    measured-weak exposed frame under `calibrate` and under `confirm`; the table recomputed bit for bit, with the switch
    off, for a frame predicted weak (for `two` and for a stratum cue);
  - non-finite values: many weak frames give infinite medians and bounds, and the result still writes, with `null` and
    the `_nonfinite` flags;
  - phase rules, the rendered preregistration, the report.
- **Tier B** (`tests/test_experiment_101_runner.py`, runner):
  - the chained fake worlds (020 → 022 → 023 → 024 → 025 → 101), on `TinyPlural`;
  - freeze → calibrate (the key checks, the re-capture check, a revision trigger stopping with no file) → lock (no
    forward) → stage1 (only S1 keys; a sealed model; the stage-1 files and record) → confirm;
  - `confirm` refused without the committed stage-1 files (a refusal, nothing recorded), and incidents on a digest
    mismatch, a recompute mismatch and an S2 key in the ledger, each before any S2 key;
  - accounting; each of G2–G6 failing gives `NOT_INTERPRETABLE` with no statistic; a landing failure and a non-finite
    contrast give incidents; a failed result write; a failing secondary step leaves the result standing; the report, its
    re-render and a report-time G6 mismatch;
  - G5 against the plain runs' captures, and a head arm's own capture equal to its replacement;
  - the stage-1 write path under a fake write hook that mirrors the launcher's allow-list;
  - a refused production run with a test configuration.
- **Tier C** (pythia smoke, spent exposed prompts only):
  - an in-memory freeze with the real tokenizer and the real exclusion sources, asserting `EXPECTED_PICKS` and writing
    nothing (as `tests/test_cue_rotation.py:602–617`);
  - a REPLACE at `RESID_PRE.L3@p_c` lands bit for bit; NOOP equals the plain run; `x3(p_t)` unchanged; the pass-through
    identity;
  - a real read edit on spent exposed prompts that records the maxima of G3 and every G4 quantity, and no outcome;
  - the head REPLACE at `L03.H04@p_t`: a no-op replacement reproduces the plain run bit for bit, and `RESID_PRE.L4@p_t`
    moves by the replaced difference (float32 rounding only); the head-arm recomputation (G5);
  - REF determinism across capture sets (G6's comparisons);
  - the guards, refusing a 101 prompt identified by its token sequence and every outcome function.
- **Integration.** `CURRENT_EXPERIMENT = "101"` in `tests/conftest.py`, which `tests/test_tiers.py` requires to have
  `tests/test_experiment_101_runner.py`. Tiers A, B and C pass before the implementation review.

## Runtime and storage (estimates on this machine)

- **Calibration:** 36 re-captures and 1,980 arm runs, at about 0.07–0.1 s each with captures, so about 3–4 minutes, plus
  the gates and the program's predictions.
- **Confirmation:** 36 stage-1 captures, then 1,980 stage-2 runs.
- **Captures per run:** `RESID_PRE.L3@p_c` and `@p_t`, `RESID_PRE.L4@p_c`, `L03.H0h@p_c` for `h = 0..7`, `L03.MLP@p_c`,
  `L03.H04@p_t`, `ATTN_PATTERN.L3@p_t`, `RESID_POST.L5@p_t` and the logits. The plain `REF` and `PL@w` runs also capture
  `RESID_PRE.L3` at every position `0..p_t`.
- **Storage:** per run, the 79 contrasts, the captures, the identity values and the logits' digest. That is about 65 MB
  under `outputs/experiment-101/`. The committed stage-1 states are about 6 MB of float64, and the table about 2 MB of
  canonical JSON.

## File map

- **New code:** `src/neural_decompiler/read_intervention.py`; `experiments/101-read-component-intervention/run.py` and
  `README.md`; `tests/test_read_intervention.py`; `tests/test_experiment_101_runner.py`.
- **Changed:**
  - `tests/conftest.py` (`CURRENT_EXPERIMENT`);
  - `.gitignore` (`outputs/experiment-101/*`);
  - `README.md` at the root, whose experiments paragraph says that 025 ends the first sequence and 101 opens the second.
- **Written by the phases, not by the implementation:**
  - `confirmation-v1.json`, `calibration-v1.json`, `preregistration-lock.json`, `preregistration.md`;
  - the stage-1 files `stage1/states.f64`, `stage1/states.json` and `stage1/table.json`;
  - the `evidence/` directories with their launchers, the final report, the confirmation record and the closure.
- **The scientific-path rule.** The prefixes are `src/` and `experiments/101-…/`, with 025's chain. The installed phase
  files, the three stage-1 files (each listed exactly in `NON_SCIENTIFIC_PATHS`, not as a `stage1/` prefix) and the
  experiment README are non-scientific paths, and `evidence/` is excluded.

## Tasks (small reviewable commits; the checkboxes are for execution)

- [ ] **1. The plan review.** Revision 1: NOT READY (2 BLOCKERs, 9 SHOULDs, 14 NOTEs). Revision 2: READY WITH CHANGES (4
  SHOULDs, 5 NOTEs). Revision 3, with the final re-check's four NOTEs applied: READY TO IMPLEMENT once the reviewer
  decides G4's form. Commit the plan, then record the decision.
- [ ] **2. Skeleton.** `ri` constants, `Configuration` (with `PRODUCTION` and test configurations), `FROZEN_BLOBS`, the
  input pins, `assert_frozen_blobs`, `encode_nonfinite`. Tier-A pin tests.
- [ ] **3. Geometry and edits.** Computations 1–4, and the G1 and G2 checks. Tier-A tests.
- [ ] **4. The freeze.** Candidates, the exclusion sources, eligibility, selection, the stratum check, the keys and
  manifests, the payload and its verification. Tier-A tests on a toy tokenizer.
- [ ] **5. The program's predictions.** Computation 6, the table payload, the stage-1 files' writer and reader, digests.
  Tier-A and tier-B tests on the fake world, including the `p_t`-key rule.
- [ ] **6. Measurement.** `ri.run_arm` and `ri.capture_reference`, the capture set, the plain captures, arm execution,
  the head arms (computation 5), the per-run and per-arm records, and gates G2–G6. Tier-B tests.
- [ ] **7. Statistics and labels.** Computations 7–8, the label table, the secondary step, the report renderer. Tier-A
  tests.
- [ ] **8. The runner.** All seven phases, with the failure classes, the barrier, ledgers, isolation, accounting,
  incidents and atomic state. Tier-B runner tests.
- [ ] **9. Tier C and the guards.** The smoke tests on spent exposed prompts, the real-tokenizer freeze, and the outcome
  guards.
- [ ] **10. Integration.** `conftest`, `.gitignore`, the experiment README skeleton, the root README. Tiers A, B and C
  pass.
- [ ] **11. The implementation review.** An independent read-only review of the code against the design and this plan:
  - the thresholds, the arm order, the keys and the accounting;
  - the frame freeze and no replacement;
  - the norm matching;
  - the prediction freeze and the barrier;
  - phase isolation and the failure classes.

  Fix and re-review until it passes.

After task 11, and only then, the phases begin: restore 020's results file, then `freeze`, `calibrate`, `lock`,
`stage1`, `confirm` and `report`, each with its review.

## Stopping conditions

- A conflict with the frozen design or its decision.
- Any need to edit a frozen module.
- Any test or script that would run a 101 frame prompt outside `stage1` and `confirm`.
- Any tier-C failure.
- An implementation-review BLOCKER that cannot be fixed within the design.

## Decision needed: the form of G4 (review B1)

Design §8 names, for every residual-edit arm, "023's I3 and I4" and "020's readout and Level-1 identities", without
restating their forms. Revision 1 gated the exact chain fed the measured `Δx3` against the measured `Δc` at 1e-3 nats.
That cannot be met, because it compares the float32 model with a float64 recomputation. The direct evidence is 025's
maximum of 0.0119 for this computation in contrast form
(`experiments/025-nounness-direction-intervention/evidence/final-report-2026-09-26.md:80`). 020's explore also failed a
1e-3 tolerance on the normalized form of the same identity, `E₁` (4.593e-3,
`experiments/020-readout-decompilation/evidence/exploration-report-2026-09-22.md:18`), and its diagnosis placed the
floor in the float32 forward's own attention scores (`src/neural_decompiler/readout_decompilation.py:72–79`).
Calibration would have stopped on trigger 1, and no run could have reached confirm.

The plan's choice, for the reviewer to confirm or change:
1. **023's I3 as 023 defined it,** on each plain pair rather than per arm: the FULL composition against the measured
   `Δx3`, relative 1e-4. Per arm it would fail by construction for PC-SWAP-PL, whose measured `Δx3(p_c)` is exactly
   zero.
2. **023's I4 on each G4 run:** the composition proxy (the FULL composition plus the realized edit) against the run's
   ceiling, both through the frozen readout, 1e-3 nats. On a plain `PL@w` run, where `x′` is the run's own residual,
   this is 023's I4 exactly. On a reference-prompt run the proxy is the realized edit itself, so I4 holds exactly and is
   vacuous there; the identities below carry the identity content on those runs.
3. **020's Level-1 identity in 025's contrast form,** at 2e-2 nats, the tolerance 025 used for the same computation.
4. **020's readout and logit identities** at their own tolerance, 2e-2.
5. **020's normalized `E₁`, recorded and not gated.** Its denominator, `‖Δh6‖∞`, is near zero for arms with small
   effects, and its 7e-3 tolerance was set from real swaps with `‖Δh6‖∞ ≥ 0.79`.
6. **The runs covered:** the residual-edit arms by name (`NOOP`; `NEC-READ`, `SUF-READ`, `NEC-AMP`, `SUF-NEG`,
   `NEC-HALF`, `SUF-HALF`; `NEC-OFF`, `SUF-OFF`; `PC-SWAP-PL`, `PC-SWAP-REF`; `NEC-ISO-j`, `SUF-ISO-j`), and also the
   plain `PL@w` runs, which the design's "every residual-edit arm" does not name. The head arms are excluded, as design
   §8 says.

**Feasibility against the record,** from the re-check:

| identity | tolerance | recorded maximum | margin |
|---|---|---|---|
| 023's I3 | 1e-4 relative | 2.03e-5 (023), 4.35e-5 (025) | 2.3–5× |
| 023's I4 | 1e-3 nats | 4.33e-5 (023), 9.0e-5 (025) | 11–23× |
| 020's readout identity | 2e-2 | 5.2e-3 (020) | about 4× |
| 020's logit identity | 2e-2 | 3.3e-3 (020) | about 6× |
| Level-1, contrast form | 2e-2 | 0.0119 (025, outcome-bearing), 0.0108 (025, secondary, in a coordinated frame) | 1.7–1.85× |

The sources are `experiments/023-block0-completion/evidence/final-report-2026-09-24.md:49`, 025's final report (line 80)
and confirmation record (`gates`), and 020's exploration report (line 26). The Level-1 margin is the tightest, and it
has not been tested under layer-3 residual edits; NEC-AMP doubles the read change. Calibration computes every G4
quantity on the exposed runs exactly as `confirm` will, and records each maximum per arm. A failure there is revision
trigger 1, before the lock.

## Changes

### Revision 2 (the review of revision 1)

- **B1, G4's form:** as above, pending the reviewer's decision. `ul.i3_error` and `b0c.FULL_MASK` are now used.
- **B2, key collisions:** every tag except `S1` and `REF` carries its cue (`ARM@cue`); a tier-A test asserts 1,980
  distinct S2 keys.
- **S1, G6 without stage-1 logits:** G6 compares the stage-2 REF run's `pm.contrasts` with `c_ref`, and its `x3(p_c)`,
  `x3(p_t)` and `h6` with the stage-1 state.
- **S2, calibration:** 36 re-captures checked at `rc.LOCKED_STATE_TOLERANCE` instead of `rc.rematerialize`; 020's
  labels; the key check against 020's ledger and every manifest; exact-once accounting; the committed confirmation file
  required and bound; G1–G6 computed as in `confirm`.
- **S3, failure classes:** precondition failures are refusals; incidents are reserved for G1, the digests and the
  barrier's integrity items; 023's barrier items added.
- **S4, non-finite values:** `null` with a `_nonfinite` flag; the primary result is written atomically, and the
  secondary statistics follow in an isolated step.
- **S5, worst values:** `WORST` per quantity; `DIRECTION_FREE` summaries exclude weak frames; stratum weak status from
  the cue's own `Δc_full`.
- **S6, records and captures:** the LayerNorm records and the manipulation check; the fixed capture set; `ri.HeadInput`
  for `rd.head_output_change`.
- **S7, the launchers:** `ri.run_arm` and `ri.capture_reference` carry the tag and are wrapped by identity.
- **S8, gate failures:** landing failures and non-finite contrasts are incidents; under `NOT_INTERPRETABLE` no statistic
  is computed; an incident after a ledger reports `NOT_INTERPRETABLE`; G6 is recomputed before the result write; a
  report-time mismatch is defined.
- **S9, tier C:** the real-tokenizer freeze, the identity maxima of a real read edit, the head REPLACE checks,
  token-sequence guards; a tier-C failure is a stopping condition.
- **N1:** `read_assembly`, `cue_suppression` and `attention_paths` pinned, and every blob written in full. **N2:**
  `ISO_TAG` with keywords; the basis asserted orthonormal. **N3:** reference prompts labelled `ref`. **N4:** the
  even-`n` median, one shared bootstrap matrix, the interval ranks. **N5:** the strict inequalities stated; no fractions
  for weak frames at `confirm` (calibration's `τ` follows design §6 and uses all 36). **N6:** G2 per arm type, with its
  metrics. **N7:** the exclusion sources named. **N8:** the lock binds the stage-1 files; stage 1 asserts the
  positions. **N9:** a stage-1 incident spends the frames. **N10:** `report` may re-render; stage 1 ends `complete` with
  `awaiting_commit`. **N11:** `CUE_ORDER` frozen. **N12:** the logit identity is computed before the logits are
  reduced. **N13:** stage 1 and confirm recompute the `w_c` and `v_j` digests. **N14:** the calibrate sentinel is named
  by the commit.
- **Also changed:** stage 1 writes its files directly at the lock's paths in the experiment directory, so the user
  commits exactly what stage 1 wrote, with no copy step.

### Revision 3 (the re-check of revision 2)

- **SA1, `τ`'s inputs:** one phase switch, `fractions_for_weak`, off at `confirm` (N5) and on at `calibrate`, so a
  predicted-weak or measured-weak exposed frame still contributes its flagged fractions to `τ`; a tier-A test covers
  both.
- **SA2, G5's target:** the plain run's captures (`PL@two` for HEAD-NEC and HEAD-NEC-V, `REF` for HEAD-SUF and
  HEAD-SUF-V), since a head arm's own capture is the value after its replacement.
- **SA3, the freeze's exclusions:** `frame_tokens` is not applied (twelve picks are tokens of exposed frames, which
  design §11 allows), and `cr.blocked_sets` is not reused as a whole; the source counts are stated; a tier-A test keeps
  `soft` and `dry` eligible.
- **SA4, stage-1 writes under the launcher's hook:** the stage-1 allow-list names the three files and the `stage1/`
  directory; stage 1 checks writability before its ledger; the three files are listed exactly as non-scientific paths;
  the rehearsal exercises the write path.
- **The notes:** the module aliases corrected (`ap` is `attention_paths`, `atp` is `attention_patterns`); calibration
  prompts built with `pool.token_prompt`; the decision text states I4's vacuity on reference-prompt runs, `x′` for plain
  runs, the gated `PL@w` runs, the arms by name and the corrected `E₁` citation; under `NOT_INTERPRETABLE` the secondary
  step does not run; NOOP's G3 comparison is over its own capture sites.
- **The final re-check's notes** (READY TO IMPLEMENT once the reviewer decides G4's form): the cue-id totals corrected
  (351, 391 with 024's cues, 431 with 025's); a stage-1 file found after the ledger is an incident; the table is always
  computed with `fractions_for_weak` off, recorded in its metadata and tested with a predicted-weak frame; the G3 row of
  the tolerance table aligned with the Gates section.
