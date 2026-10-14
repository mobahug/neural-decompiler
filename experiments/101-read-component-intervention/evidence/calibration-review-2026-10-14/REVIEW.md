# Experiment 101: review of the `calibrate` phase

**AI review (independent, read-only), not human validation.** The reviewer is an AI agent (Claude). It worked
read-only on 2026-10-14, from about 08:52 to 09:30 UTC, against `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler`
at HEAD `8f65062627c77815adf9289c3fe60191d9fc3c17` (branch `exp101-phases`, clean tree). It wrote only inside
`.../scratchpad/exp101/calibration-review/`. It created, changed and deleted nothing in the repository, including
`outputs/`; the sha256 of the three files in `outputs/experiment-101/` are the same before and after the review
(`out/repo_snapshot_start.txt`, `out/repo_snapshot_end.txt`).

- **Model use.** It ran no forward pass of the model, no capture and no intervention. In particular it ran no prompt of
  the 36 frozen 101 frames, none of the 60 candidate texts with any cue, and no exposed prompt either; it did no
  determinism spot check. Cross-process determinism comes from the saved digests instead (check 13).
- **Weights.** Weights were read for weight-only recomputation, in two ways:
  - scripts `b`, `c` and `f` read tensors straight from the pinned `model.safetensors`. They hold no model object and
    have no token-input path; every recomputation starts from saved residuals;
  - script `d` loaded the model object once, through `models.load_model`, for the frozen functions. Before any
    `neural_decompiler` import it installed a guard. `torch.nn.Module.__call__` refused every call, after checking any
    integer input against the 240 forbidden token sequences (the 60 design texts × `one`/`two`/`three`/`many`, plus
    prefix relations). Every capture and intervention entry point (`pm.capture_prompt`, `pm.run_patched`,
    `run_capture`, `run_interventions`, `rd.capture_frame_020`, `ri.run_arm`, `ri.capture_reference`, `ri.stage_runs`,
    `ri.measure_frame`, and their aliases) refused every call after checking the prompt's token ids.
  - **The guard was installed and never fired:** 0 module calls refused, 0 entry points refused, 0 hits on a 101
    sequence. Script `g` refused every forward and every model load.
- **What it did not run.** No runner phase ran, `validate` included.
- **Its own write guard.** Every script carried an audit hook. It refused, and would have recorded, any write, rename,
  delete or mkdir in the repository, and any read under `outputs/` other than `outputs/experiment-020/results.json` and
  `outputs/experiment-101/`. It recorded nothing. `.git/index` is unchanged.
  - My two snapshot `git status` calls ran without `--no-optional-locks`. Git created and removed its transient lock,
    so the mtime of the `.git` directory changed at 09:22:50 UTC. No file content, index, ref or object changed.

## Verdict: **PASS WITH NOTES**

`calibrate` was executed and recorded as the frozen protocol requires, and every number in its state and its candidate
file is correct.
- **My own code reproduces everything.** It imports no `neural_decompiler` code and was written and run first. It
  reproduces:
  - the 2,016 keys and their planned digest, and the run log's order;
  - every measurement digest and the state digest;
  - the landing of every edit and every bitwise gate;
  - every tolerance gate's maximum and its location;
  - the exact-chain Level-1 values, within 7.4e-12;
  - the head deltas, within 1.9e-15;
  - every measured quantity, τ and the four triggers.
- **The frozen functions agree, run afterwards in a fresh process.** They regenerate:
  - the prediction table's digest;
  - the full gate record, bit for bit;
  - the candidate calibration file's `content_sha256` `1eb1fe68…`, bit for bit.
- **The bindings hold.** `ri.load_calibration_101` accepts the candidate. No revision trigger fired and no tolerance
  changed.

There is no blocker. One should-fix concerns preserving the calibrate evidence, which today lives only in a temporary
scratchpad, and correcting its draft README. The notes concern the Level-1 margin at confirm, which is an estimate with
no decision force, the strength of some gates at calibration, and descriptive observations.

## Key numbers

| item | value |
|---|---|
| τ_rem / τ_ins | **0.025153865322968608** / **0.03511072680382421** (33rd of 36; both ≤ 0.20, so P is evaluable); exposed joint agreement 31/36 |
| triggers | gate failures **0**; `‖off_c‖ < s` in **0**/36 (max `s/‖off_c‖` 0.474, `coordinated-adjective-009-1`); `Δc_full > −1` in **0**/36 (least negative −2.604, `018-3`); non-finite fractions **0** (predicted and measured, all cues). Stratum: 0 overshoot and 0 weak for `three` and `many` |
| G4 maxima (margin = tolerance / maximum) | Level-1 **0.010654** (NEC-HALF@many, `018-3`; **1.88×**); readout 0.004865 (PL@two, `013-1`; 4.11×); logit 0.003032 (SUF-ISO-6@two, `011-2`; 6.60×); I3 1.178e-5 (`016-4`, two; 8.49×); I4 3.439e-5 (PC-SWAP-PL@many, `016-4`; 29.1×); `x3(p_t)` 0/1,692 bitwise failures |
| other gates | G1 `read_weight` max-abs 0.0, basis ≤ 6.7e-16; G2 r 2.87e-8, length 3.34e-8, direction 5.37e-7, `û_off` 5.04e-9; G3 NOOP 0/36 failures, pass-through 6.62e-7; G5 row 5.68e-6, output 1.12e-5, landing 0/144 failures; G6 0/36 + 0/36 failures; environment drift **0.0** (1e-9) |
| exposed shares (`two`) | N > 0 **36/36**, S > 0 **36/36**, A_amp < 0 **30/36**, A_neg > 0 **36/36**, P agree **31/36**. Sign-test power: R and I 1.000 at 26 and at 25; D 0.989 (A_amp) and 1.000 (A_neg) at 25; P 0.978 at 27 |
| Level-1 at confirm (no decision force) | P(max over a same-size confirm > 2e-2): about **0.3 %** (GEV on frame maxima); 0.02–1.1 % (POT on runs); frame-bootstrap upper range 6.6 % (90th) to 10.8 % (95th) |
| digests | state `729c9cb5…` (recomputes; file sha256 `935a0df2…`); candidate file `6b1db338…`, content `1eb1fe68…` (recomputes); measurements file `e8f42c1d…`, 2,556 tensors; table `ce055e09…`; keys `5d4bcb51…`; launcher `0ad9d0e1…` |

## Method and order of operations

The file times in `out/` record the order. Local time is UTC+3.

1. **Reading (11:52–11:57 local).** The brief; the launcher, its records and its logs; design §2, §4–§8 and §14; plan
   § Constants, § Canonical computations, § Phases, § Gates and § G4's decision; `ri` and the runner.
   - Start snapshot: `out/repo_snapshot_start.txt`.
2. **My own code, with no `neural_decompiler` import and no model object.**
   - **a, 12:01.** The keys, the ledger, the run log and the isolation. This uses the tokenizer only.
   - **b, 12:05.** The state, the measurements, G2–G6 from the tensors, the measured quantities and the triggers.
   - **c, 12:09.** My own float64 GPT-NeoX layers 3–5 from the safetensors, which give Level-1, the frozen-readout
     ceilings, G5 and the head deltas.
3. **The frozen cross-check (d, 12:13), only then.** Run under the guard: `Runner._base()`, `ri.geometry_block`,
   `ri.prediction_table`, `ri.frame_values`, `ri.run_gates`, the frozen per-run Level-1, I3 and I4,
   `ri.calibration_results`, `ri.calibration_payload` and `ri.load_calibration_101`.
4. **Analysis (e, 12:17; f, 12:19; g, 12:20).**
   - e: τ from my measured fractions and the frozen predictions; the G4 maxima per arm type; the Level-1 risk; the head
     and isotropic descriptives; the smoke reference against production.
   - f: the decomposition of the Level-1 error.
   - g: the state's inputs and the phase rules, with every forward refused.
   - End snapshot: `out/repo_snapshot_end.txt`.

## Checks

### 1. The launcher's evidence: PASS

- **One run, one load, one record.** `production_run.json` records:
  - `calibrate_invocations` 1, `load_model` 1, `exit_status` 0, `error` null (`production.exit`: `exit 0`);
  - `refused`, `forbidden_reads` and `forbidden_writes` all empty;
  - `post.keys_done` 2016 and `all_keys_ran_once` true;
  - `tree_after` "".

  The three outputs are git-ignored (`.gitignore:34`). The launcher's sha256 is `0ad9d0e1…` in all three mode records,
  in `launcher_sha256_at_production.txt`, in the sentinel's content and in the file now.
- **Keys.** My reconstruction never touches `ri` or the pool code:
  - the 36 exposed coordinated frames come from 020's committed `exposed_frame_prompts` token ids and 020's locked
    `p_c`/`p_t`;
  - each frame is re-tokenized from its text with `one`, `two`, `three` and `many`;
  - the 55 tags per frame are in the plan's order, with 020's labels: `ref|581`, `cardinal:pl|767`, `three|1264`,
    `many|1142`.

  It gives 2,016 distinct keys whose planned digest is `5d4bcb51…`. That equals:
  - the state's `calibration.planned` (count and sha256);
  - `calibration_executed`, as a set and sorted;
  - the candidate's `keys`;
  - the frozen `Runner._calibration_schedules`.
- **The run log.** `production_runs.jsonl` has 2,016 lines with `n` = 1…2016, and each key appears once. Its order is
  the expected one: the 36 S1 re-captures, then each frame's 55 runs in plan order, frames in pool order.
  - Each logged replacement digest equals the float32 digest of the saved `replacement/…` or `head_replacement/…`
    tensor, or is empty for the plain runs: 0 mismatches.
  - `sanctioned_forward_calls` 685,440 = 2,016 × 340, the same count per run as the smoke's 19,040 = 56 × 340. That is
    consistent with exactly one forward per sanctioned run.
- **Writes.** Only the allowed outputs were written:
  - `results.json`, five times, each through a `.results-*.json` temporary file and one rename;
  - `calibration-measurements.pt`, once;
  - `candidate-calibration-v1.json`, once;
  - `mkdir` of `outputs/experiment-101/`, `exist_ok`.

  No `calibration-partial.pt` was written. The reads under `outputs/` were 020's results and 101's own files.
- **The guards were installed before the runner was imported.**
  - In `launch_calibrate.py` the audit hook comes first (`:153`). It is followed by `torch.nn.Module.__call__`
    (`:296`), the low-level and sanctioned wrappers (`:297–380`), `models.load_model` (`:454`) and the first alias sweep
    (`:455`).
  - All of these run at import, before `preflight()` (`:513`) loads the runner (`:232–235`). The runner's namespace is
    swept again at `:514`, before `calibrate()` (`:525`).
  - The runner imports `load_model` by name, so it binds the counted loader. The launcher asserts this (`:236`).
  - 11 aliases were wrapped, as recorded. None is in the runner's namespace, which uses module attributes.
- **The ledger was on disk before the first prompt.**
  - `_admit` → `_ledger_on_disk` (`:314–322`) refuses the first sanctioned call unless `results.json` already holds:
    - `calibration_executed` ⊇ the 2,016 keys;
    - `planned.count` 2016;
    - `planned.sha256` = sha256 of the sorted keys;
    - an empty `executed_prompt_keys`.
  - Nothing was refused. The runner writes the ledger at `run.py:500` before the model loads.
  - The first repository write in `output_writes` is the state (`.results-picbk0fo.json`), before the measurements.
- **No 101 prompt.**
  - I parsed the 60 candidate texts from design §11 as committed at HEAD and tokenized them. The 36 frozen frames of
    `confirmation-v1.json` match my tokenization (144 of 144), and `ri.FRAME_CANDIDATES` gives the same 240 sequences.
  - None of the 144 calibration prompt sequences equals a 101 sequence, is a prefix of one, or has one as a prefix.
  - All 144 untagged keys are in 020's ledger (30,132 keys). None is in the manifests of 020 (3,060), 022 (3,060),
    023 (3,060) or 024 (4,320), or among the untagged keys of 025 (4,320), or among 101's own keys.

### 2. The rehearsal and the smoke: PASS (claims hold; see N4)

- **The rehearsal (08:43:31–08:43:42 UTC).** It made no forward call and no write, and loaded no model. Its 20
  preflight checks are all true. The README says "All 21", but 21 is the production's count, which adds the token check
  (SF1).
- **The smoke (08:44:02–08:44:28 UTC).** It ran one exposed frame's 56 keys (`coordinated-adjective-1`) under the
  production guards, and every claim matches its record:
  - `keys_allowed` 56 and `keys_done` 56;
  - tensors, logits and the state digest equal to the unguarded process;
  - no gate failure for the frame, and the table computed;
  - 430 module instances and 20 class methods sealed, with their signatures kept;
  - seven unsanctioned routes refused, each with 0 forward calls;
  - no output written and an empty `tree_after`.

  The code matches: `smoke()` at `:458–499`, with `ok` requiring every condition.
- **Does running the frame outside the phase compromise anything?** No.
  - These are spent exposed prompts in 020's ledger, outside every 101 ledger, like the tier-C runs.
  - Nothing was written to the repository.
  - The commit and the launcher (`0ad9d0e1…`) are the same between smoke and production, so nothing was tuned.
  - The phase's exact-once accounting covers the phase's own executions, and production recomputed everything.
  - Check 13 shows that production's 71 tensors and the logits digest for this frame equal both earlier processes,
    bit for bit.

### 3. The state: PASS

- **Digest and format.** `state_sha256` `729c9cb5…` recomputes over the canonical JSON, and the file is exactly the
  canonical JSON plus a newline.
- **Phases.** `calibrate` is complete at `8f65062…`, from 08:45:07 to 08:51:04, with the 020 runtime: cpu, float32,
  4 threads, seed 20260916. `lock`, `stage1`, `confirm` and `report` are not started. `lock`, `stage1`,
  `confirmation` and `report` are null.
- **Ledger.** `executed_prompt_keys` is `[]`.
- **The `calibration` block.**
  - `planned`, `accounting` (2016/2016, equal), `measurements` (2,556 digests, `logits_sha256`, `n_executed` 2016);
  - `gates` (failures `[]`, `g1` `[]`), `ceilings` (36 frames × 50 G4 runs), `tau`, `candidate_path`,
    `content_sha256`, `file_sha256`;
  - no `stopped`, `incidents` or `partial`.
- **Inputs, versions and bindings.**
  - The 42 `inputs` digests equal the runner's `base.digests` now.
  - `versions` equal `collect_versions()`.
  - `configuration` equals `ri.PRODUCTION.to_json()`, and `module_blobs` equals `ri.FROZEN_BLOBS`, which
    `assert_frozen_blobs` confirms.
  - `confirmation_101` equals the committed file: file `6da26cd7…`, content `ef356506…`, manifest `7c7c9494…`.
- **Phase rules.** `ri.assert_phase_allowed` refuses a calibrate rerun and allows `lock`.
- **The extract.** The operator's `state_extract.json` agrees with the state: results file sha256, state sha256,
  inputs, versions, phases, run id and the calibration fields.

### 4. The measurements: PASS

- **Digests and contents.** All 2,556 tensors are exactly the expected set: 71 per frame, enumerated independently.
  My tensor digests equal the state's and the candidate's (0 mismatches). Every float tensor is finite, and every frame
  has 55 rows.
- **The environment check.** Each frame's REF row equals 020's locked reference state exactly, at max-abs 0.0 and bit
  for bit, in five places:
  - `x3(p_c)`, `x3(p_t)` and `h6`;
  - `x3_all/REF` at every position;
  - `c_ref_pm` against 020's `c_ref` (80 values).

  That agrees with the recorded max drift of 0.0 against tolerance 1e-9.
- **G6.** The bitwise comparisons hold (`G6_c_ref`, `G6_state`: 0/36 failures each). The positions before `p_c` of
  `x3_all/REF` and `x3_all/PL@w` are identical (0 failures).

### 5. The geometry and the edits: PASS

- **`w_c`.**
  - From 011's lock (`read_weight`, centred) `‖w_c‖` is 0.76592.
  - From the weights, `γ₃ ⊙ (W_V W_O d̂_T)` of head 3.4, which I sliced myself from `query_key_value` and `dense`,
    equals the lock's `read_weight` at max-abs **0.0**.
  - G1 as recorded: 0.0, basis 1.1e-16, `v_j` ≤ 6.7e-16.
- **The pair quantities.** Per frame and cue (Δr, ‖Δx‖, ‖off_c‖, s, s/‖off_c‖, overshoot) they equal the candidate's
  `reports` exactly (max diff 0.0). `min_s`: two 2.103 (`017-4`), three 1.874 (`018-2`), many 1.803 (`018-2`).
- **Every edit, recomputed.** I recomputed all 1,692 residual edits from the plain captures, in float64 and then cast
  to float32. That includes the 16 isotropic directions per frame, from my own SHA-256 counter-mode Box–Muller
  implementation of the plan's rule. All 1,692 equal the saved replacements bit for bit (0 ulp), and each captured
  `x3(p_c)` equals its replacement (landing, 0 failures).
- **G2's realized geometry.** My maxima and locations equal the recorded ones exactly:
  - r 2.87e-8 (NEC-AMP@two, `013-1`);
  - centred length 3.34e-8 (NEC-OFF@three, `018-3`);
  - direction 5.37e-7 (SUF-HALF@three, `018-3`);
  - `û_off` 5.04e-9 (`009-2`, many).

  The per-arm maxima differ by 0.0.

### 6. The gates' values and G4 viability: PASS

- **Recomputed from the tensors, my code.**
  - G3 NOOP equals PL@two bit for bit at every capture field, in its contrasts, its logit identity and its logits
    digest (0/36 failures).
  - G3 pass-through: 6.62e-7, recomputed with `b_O[3]` from the weights.
  - G4 `x3(p_t)` bitwise: 0/1,692 failures.
  - G5 landing: 0/144 failures.
  - G5 row and output, from my own layer-3 attention on the plain runs' `x3_all`: 5.68e-6 and 1.12e-5, at the recorded
    location, `019-4` HEAD-SUF@two.
  - Every head arm's δ reproduces within 1.9e-15, and its float32 replacement bit for bit (144/144).
  - G4 readout: 0.004865, at the same location; per-arm maxima within 2.7e-15.
- **The logit identity.** The logits are not saved. The per-run values saved in `logit_identity` are consistent with
  the record:
  - their maximum over the G4 runs, 0.0030324 (SUF-ISO-6@two, `011-2`), is the recorded gate value;
  - the per-arm maxima are identical;
  - the maximum over the other runs, 0.0029138 (HEAD-NEC@two, `017-1`), is the recorded `logit_other`.
- **The program-based quantities.**
  - My float64 exact chain reproduces the frozen Level-1 on all 1,800 G4 runs within 7.4e-12, and recorded `E₁`
    exactly (0.016048, not gated).
  - `ri.run_gates` on the saved tensors gives the recorded gate record exactly. G6 was run here against 020's locked
    states, which equal the re-captures bit for bit.
- **G4 maxima per arm type.** Arm type means: the plain PL@w runs; NOOP; the read arms (NEC-READ, NEC-AMP, NEC-HALF,
  and SUF-READ, SUF-NEG, SUF-HALF); the off arms; the isotropic arms; PC-SWAP. All cues are included. The margin is the
  tolerance divided by the maximum.

| arm type | Level-1 (2e-2) | I4 (1e-3) | readout (2e-2) | logit (2e-2) |
|---|---|---|---|---|
| read, NEC | **0.010654** NEC-HALF@many `018-3` (**1.88×**) | 3.26e-5 (31×) | 0.004260 (4.7×) | 0.002839 (7.0×) |
| read, SUF | 0.007277 SUF-NEG@many `011-1` (2.75×) | 0 (vacuous) | 0.004419 (4.5×) | 0.002790 (7.2×) |
| PL@w (plain) | 0.005414 PL@three `019-5` (3.69×) | 2.02e-5 (49×) | **0.004865** PL@two `013-1` (4.11×) | 0.002884 (6.9×) |
| PC-SWAP | 0.005253 PC-SWAP-REF@many `018-3` (3.81×) | **3.44e-5** PC-SWAP-PL@many `016-4` (29×) | 0.004697 (4.3×) | 0.003013 (6.6×) |
| off | 0.005008 NEC-OFF@two `015-2` (3.99×) | 2.06e-5 (49×) | 0.004190 (4.8×) | 0.002780 (7.2×) |
| iso | 0.004905 SUF-ISO-7@two `016-3` (4.08×) | 1.38e-5 (72×) | 0.004558 (4.4×) | **0.003032** SUF-ISO-6@two `011-2` (6.6×) |
| NOOP | 0.004821 (4.15×) | 1.34e-5 (75×) | 0.004865 (4.1×) | 0.002884 (6.9×) |

I3 runs on each plain pair: maximum **1.1776e-5** (`016-4`, two), 8.49×; three 1.002e-5; many 1.033e-5.

**Did the frozen tolerances hold?** Every G4 tolerance held, with margins of at least 1.88×. Every other gate's margin
is at least 17×.
- **The Level-1 margin is the only thin one.** It is 1.88× overall: NEC-HALF@many in `018-3`. The next frame maxima
  are 0.00728 (2.75×) and 0.00541 (3.69×).
- **Level-1 by cue.** two 0.00501, three 0.00541, many 0.01065.
- **At confirm.** Its exceedance risk is estimated in item 11. It is small but not negligible, because a G4 failure
  consumes the confirmation (N1).
- **No other margin is a confirm risk.** Readout 4.1×, logit 6.6×, I3 8.5×, I4 29×, G2 ≥ 186×, G3 151×, G5 ≥ 17×.
  Their exposed distributions are narrow, and none depends on the edit size.

### 7. The measured quantities and τ: PASS

- **Measured quantities.** My `Δc(arm)` is the mean over the 79 nouns of `contrasts(arm) − contrasts(REF)`. From it come
  `Δc_full`, `F_rem`, `F_rem_off`, `F_ins`, `F_ins_off`, `N_f`, `S_f`, `A_amp` and `A_neg`, the half, PC-SWAP and
  additivity fractions, and the isotropic and head quantities, for every frame and cue. They equal `ri.frame_values`
  (with `fractions_for_weak=True`) at max-abs **0.0**, and the candidate's `distributions` at 0.0.
- **Nouns.** The 79 nouns are 020's exposed order minus `peach`, with ids from the committed screening manifest and
  006's confirmation file. They equal the runner's `_check_nouns`.
- **The program's predictions.**
  - `ri.prediction_table` on 020's locked states with `fractions_for_weak=True` gives `content_sha256` **`ce055e09…`**,
    which equals the state's `table_content_sha256`, in a fresh process (25.7 s). This is the check that `lock` will
    repeat (`run.py:632`).
  - No frame is predicted weak, for any cue.
- **τ.** With my measured fractions and the frozen predictions:
  - `τ_rem` = 33rd of the 36 sorted `|F̂_rem − F_rem|` = **0.025153865322968608**;
  - `τ_ins` = **0.03511072680382421**;

  both equal to the recorded values, and the 36 errors equal the candidate's `P_error_*` lists. Both are ≤ 0.20, so P
  is evaluable.
  - The four largest errors are 0.0252, 0.0262, 0.0283 and 0.0405 (rem), and 0.0351, 0.0469, 0.0495 and 0.0575 (ins).
  - With the floors `max(τ, 0.02)`, 31/36 frames agree. The disagreeing frames are `fresh-1`, `013-1`, `014-1`,
    `017-3` and `019-4`.
- **The program against the measured values, descriptively.** `N̂ − N` lies in [−0.021, 0.044] and `Ŝ − S` in
  [−0.049, 0.059]. `Δĉ_full − Δc_full` lies in [−0.171, 0.252] nats. The program predicts the sign of `A_amp` in 35/36
  frames.

### 8. The four revision triggers: PASS (none fired)

| trigger | value | locations |
|---|---|---|
| a gate failure (G1–G6, the environment) | `[]` | — |
| `‖off_c‖ < s` in > 6 of 36 | **0** for two; also 0 for three and many | the largest `s/‖off_c‖`: two 0.474 (`009-1`), three 0.411 (`009-1`), many 0.367 (`018-4`) |
| `Δc_full > −1` in > 6 of 36 | **0**; also 0 for three and many | the least negative `Δc_full`: two −2.604, three −2.672, many −2.834, all in `018-3` |
| a non-finite fraction | **0**, predicted and measured, all cues | — |

- **No trigger depends on the size of an exposed result.** In `ri.calibration_results`
  (`read_intervention.py:1768–1829`, triggers at `:1799–1806`) the triggers read only:
  - the gate book's failures, G1 and the environment check;
  - the overshoot and weak flags of the primary cue;
  - the non-finite flags.

  N, S, D and P enter only `shares`, `power` and `distributions`, which have no outcome force.
- **No tolerance was changed.**
  - `ri.TOLERANCES` (`:388`) equals the plan's tolerance table entry for entry, and the candidate's `tolerances`
    equal it.
  - The module blob `d7db6197…` is the HEAD blob, the imported file and `own_blob()`.
  - `git diff 40829d2 8f65062 -- src tests experiments/101-*/run.py` is empty.
- **G4 as an integrity gate.** All 36 frames entered τ, and no frame was dropped, replaced or ignored.

### 9. The candidate calibration file: PASS

- **Digests.** The `content_sha256` `1eb1fe68…` recomputes, and the file is canonical JSON plus a newline. Its file
  sha256 `6b1db338…` equals the state's.
- **What it binds.**
  - the confirmation file: file, content and manifest digests, and counts;
  - the module blob `d7db6197…`, which is `ri.own_blob()`;
  - `tolerances` equal to `ri.TOLERANCES`;
  - τ, as above;
  - the triggers, none fired;
  - the keys: count 2016, sha256 `5d4bcb51…`;
  - `measurements_sha256`, equal to the state's 2,556 digests;
  - the table digest, `g1`, the environment check (drift 0.0, tolerance 1e-9, passed), and the commit `8f65062…`.
- **Reproduced and accepted.** `ri.calibration_payload` rebuilt from the saved measurements reproduces the file's
  `content_sha256` and its canonical text bit for bit. The environment block is taken from the record, since the
  re-captures are not saved. `ri.load_calibration_101` accepts the candidate.

### 10. Descriptive report (exposed; no outcome force)

**Real swaps, for `two`** (ranges, with the median in brackets):

| quantity | range |
|---|---|
| Δr | 1.610–2.175 (1.981) |
| ‖Δx‖ | 6.27–7.46 (6.74) |
| ‖off_c‖ | 5.81–6.96 (6.21) |
| s | 2.10–2.84 (2.59) |
| s/‖off_c‖ | 0.320–0.474 (0.407) |
| `Δc_full` | −6.04 to −2.60 (−4.94) |

- **The stratum.** Δr for `three` is 1.44–2.08 and for `many` 1.38–2.03. `Δc_full` for `three` is −6.13 to −2.67 and
  for `many` −6.31 to −2.83.
- **LayerNorm.** The medians of σ(x′) are 0.55–0.60 per arm; the plain residuals' σ is 0.562 for ref and 0.574 for
  two. `‖γ₃⊙e‖/σ` reaches 8.9 (NEC-READ@two), 9.5 (NEC-OFF@two) and 24.9 (PC-SWAP-PL@two).

**Fractions and family statistics (`two`):**

| quantity | median | range |
|---|---|---|
| `F_rem` | −0.035 | −0.451 to 0.158 |
| `F_rem_off` | 0.995 | 0.976–1.024 |
| `F_ins` | 0.980 | 0.886–1.076 |
| `F_ins_off` | −0.041 | −0.172 to 0.030 |
| N | 1.031 | 0.835–1.437 |
| S | 1.033 | 0.878–1.140 |
| A_amp | −0.145 | −0.448 to +0.128 |
| A_neg | 1.167 | 0.379–3.411 |

- **Half dose.** `F_half_rem` 0.734 and `F_half_ins` 0.727.
- **PC-SWAP.** `F_swap_pl` 0.035 (−0.084 to 0.185) and `F_swap_ref` 0.995 (0.907–1.054).
- **Additivity.** `F_rem + F_ins` 0.942 (0.518–1.141).
- **The stratum, by count.** `three`: N > 0 36, S > 0 36, A_amp < 0 29, A_neg > 0 36. `many`: 36, 36, 26 and 36,
  with a weaker removal (`F_rem` median 0.206, N median 0.775).

**Observations** (flagged; none may revise the calibration):
- **Saturation of the doubled read change.** `A_amp ≥ 0` in 6/36 frames: `fresh-1` +0.128, `015-2` +0.053, `019-5`
  +0.049, `1` +0.043, `016-1` +0.034, `009-1` +0.017. D's A_amp count is 30 against a threshold of 25: power 0.989 at
  that share, but the thinnest of the primary sign margins. The program predicts 31 negative and gets the sign right
  in 35/36 frames. `many` is weakest (26/36), but the stratum has no label force.
- **R and I are far from their thresholds on exposed data.** Removing the read part takes away about all of the shift,
  and installing it gives about all of it, against about none for the matched off-read slice. `F_rem` is negative in
  most frames: the removal slightly overshoots toward singular.
- **The frozen readout's ceiling errors.** These are the readout fed the measured `Δx3`, against the measured `Δc`.
  - Per noun, max-abs: up to **1.30 nats** (PC-SWAP-REF@many, `019-4`); 0.73 for NEC-READ@two and SUF-READ@two.
  - At the level of the mean over nouns: medians 0.02–0.05 nats, maxima 0.16 (NEC-READ@two), 0.086 (SUF-READ@two) and
    0.50 (PC-SWAP-REF@many).

  P works on fractions of means, and its exposed errors are small, at the τ above. Some exposed frames entered 012's
  bases and `d̂_T`, so τ may be tight (design §6).
- **`018-3` is the exposed frame with the smallest shift and the largest Level-1 errors.** It was also 020's
  `level1_worst` frame.

### 11. The Level-1 margin at confirm (a risk estimate; no decision force)

**The distribution.** The data are the frozen Level-1 error of all 1,800 G4 runs: the 1,692 residual-edit runs plus
the 108 plain PL@w runs. Mine equal them within 7.4e-12 (`out/d_per_run_frozen.json`).

| quantile | all 1,800 G4 runs | the 1,692 residual-edit runs |
|---|---|---|
| median | 0.003024 | 0.003023 |
| 90th | 0.003783 | 0.003775 |
| 95th | 0.004055 | 0.004045 |
| 99th | 0.004740 | 0.004610 |
| 99.9th | 0.005788 | 0.005879 |
| maximum | 0.010654 | 0.010654 |

- **Counts above.** 7 runs exceed 0.005, 2 exceed 0.006, 1 exceeds 0.0075 and 1 exceeds 0.01.
- **Frame maxima.** `018-3` 0.010654; `011-1` 0.007277; `019-5` 0.005414; the other 33 lie between 0.00346 and
  0.00503.

**The top 20** (value, arm@cue, frame, `p_c`, the noun at the maximum):

| # | value | arm@cue | frame | `p_c` | noun |
|---|---|---|---|---|---|
| 1 | 0.010654 | NEC-HALF@many | `018-3` | 7 | army |
| 2 | 0.007277 | SUF-NEG@many | `011-1` | 5 | cross |
| 3 | 0.005414 | PL@three | `019-5` | 5 | bus |
| 4 | 0.005253 | PC-SWAP-REF@many | `018-3` | 7 | army |
| 5 | 0.005176 | SUF-NEG@three | `018-3` | 7 | army |
| 6 | 0.005030 | PC-SWAP-PL@many | `017-4` | 5 | train |
| 7 | 0.005008 | NEC-OFF@two | `015-2` | 4 | island |
| 8 | 0.004905 | SUF-ISO-7@two | `016-3` | 6 | tunnel |
| 9 | 0.004905 | PC-SWAP-REF@many | `011-1` | 5 | wish |
| 10 | 0.004883 | NEC-AMP@many | `018-3` | 7 | army |
| 11 | 0.004877 | PL@many | `011-1` | 5 | wish |
| 12 | 0.004860 | SUF-NEG@many | `015-2` | 4 | island |
| 13 | 0.004833 | NEC-OFF@many | `018-3` | 7 | dish |
| 14 | 0.004821 | PL@two | `019-5` | 5 | bus |
| 15 | 0.004821 | NOOP@two | `019-5` | 5 | bus |
| 16 | 0.004795 | PL@three | `013-1` | 7 | bus |
| 17 | 0.004784 | NEC-ISO-4@two | `018-2` | 6 | cat |
| 18 | 0.004740 | PL@two | `1` | 5 | brush |
| 19 | 0.004740 | NOOP@two | `1` | 5 | brush |
| 20 | 0.004711 | NEC-ISO-7@two | `017-4` | 5 | train |

**What drives the outliers.** I split each run's error vector exactly into two parts. The Δh6 part is the exact chain's
`h6` against the measured `h6`, through `LN_final` and `Δw`. The readout part is the measured `h6` → contrast path,
that is, the readout-identity residual.
- **The bulk floor is the float32 readout path.** The readout part has a median of 0.0028, the Δh6 part 0.0009, and the
  run-level Spearman correlation of Level-1 with the readout-identity error is 0.68.
- **The two outliers come from Δh6.** That part is 0.0086 of the 0.0107, and 0.0063 of the 0.0073. It is a float32
  error of the measured blocks 4–5 output at `p_t`: `‖Δh6^exact − Δh6^meas‖∞` is 0.0063, the largest of all runs. The
  readout part adds the same sign.
- **Block 3 is accurate.** Its output at `p_c` matches the captured `RESID_PRE.L4@p_c` to 2.8e-5 in the top run.
- **This is 020's floor.** The reference attention scores carry common offsets of about 1.2–3.3e4, 6–10e4 and 6–15e4
  at layers 3, 4 and 5, which is 020's float32 attention-score floor.
- **Not the edit size.** The Spearman correlation with `‖Δx3(p_c)‖` is −0.02, with `|Δc|` −0.01, and with `‖Δh6‖`
  0.005. The plain PL and NOOP runs show the same floor (maxima 0.0054 and 0.0048). NEC-AMP, the largest edit, peaks
  at 0.0049.
- **Not the frame's structure.** At frame level the correlation with `p_c` is −0.08, with the layer-5 score scale
  0.04, and with the layer-3 and layer-4 score scales 0.22 and 0.20.
- **The nouns.** The top run's noun `army` has an ordinary `‖Δw‖` (rank 43/79). The second run's noun `cross` has the
  largest (0.706).
- In short: rounding in the float32 forward, with a frame-specific susceptibility, not a new floor from the residual
  edit.

**The exceedance estimate.** The target is the probability that the maximum over a confirm of the same size exceeds
2e-2. A confirm of the same size means 36 new frames and the same 50 G4 runs per frame. Frames are the natural blocks.
The estimates come from my own numpy MLE fits (Nelder–Mead; there is no scipy in the venv), in
`scripts/e_tau_risk_descriptives.py`.

| method | estimate of P(confirm max > 0.02) |
|---|---|
| GEV fitted to the 36 frame maxima (μ 0.00413, σ 0.00046, ξ **+0.23**) | **0.27 %** (per frame 7.6e-5) |
| peaks over threshold on the 1,800 runs, GPD, threshold at the 90th percentile (ξ 0.15) | 0.02 % |
| the same at the 95th percentile (ξ 0.33) | 1.1 % |
| frame bootstrap of the GEV estimate (2,000 resamples; seed 20261014) | median 0.24 %, mean 2.2 %, 90th 6.6 %, **95th 10.8 %** |
| GEV without the single extreme frame `018-3` | 4e-7 |
| lognormal on the frame maxima | about 1e-12 (light-tailed; it does not fit the two outliers, so it is not credible) |

How to read the methods:
- The peaks-over-threshold estimates treat the runs as independent. Clustering of exceedances within a frame only
  lowers the probability of at least one exceedance, so they are conservative in that respect.
- The bootstrap range reflects how much the tail fit leans on `018-3`. In resamples that contain it two or three times,
  the fitted tail is heavier.

**Context.**
- The tolerance is 1.88× the observed maximum, and the observed maximum is 1.46× the second-largest frame maximum.
- 025 recorded 0.0119 (outcome-bearing) and 0.0108 (secondary, in a coordinated frame) for the same identity in
  contrast form; this calibration's maximum is 0.0107. The observed maxima so far lie at 0.0107–0.0119.
- The confirm frames have `p_c` 4–6 (10, 22, 4 frames) against 4–7 for the exposed ones. I credit no reduction from
  that, because the frame maxima show no dependence on `p_c`.

**My estimate.** The probability is of order **0.3 %**, plausibly between 0.02 % and 1 %. Under a heavy-tailed
extrapolation from 36 frames it cannot be excluded up to about 10 %.

The calibration found no new numerical floor introduced by the intervention, which is the condition under which the
reviewer's decision anticipated an amendment. Whether a few-percent tail risk of consuming the confirmation justifies
an explicit reviewed amendment before the lock is the human's decision. Rule 1 forbids this review from changing any
tolerance.

### 12. The head arms and the isotropic controls (exposed; no label force)

- **The head arms** (`two`; median, with the range in brackets).
  - **Removal.** `F_head_rem` = `Δc(HEAD-NEC)/Δc_full` is 0.069 (−0.333 to 0.259); the value-only variant is 0.075.
    Replacing only `L03.H04`'s output at `p_t` with its read-removed value removes about 93 % of the shift.
  - **Installation.** `Δc(HEAD-SUF)/Δc(SUF-READ)` is 0.892 (0.833–0.977); the value-only variant is 0.890
    (0.831–0.970). Against `Δc_full`, `F_head_ins` is 0.877 (0.788–0.990).
  - **The shares under the 0.05 guard.** Neither guard excludes any frame: `1 − F_rem ≥ 0.05` and `F_ins ≥ 0.05` hold
    everywhere.
    - The removal share `(Δc_full − Δc(HEAD-NEC))/(Δc_full − Δc(NEC-READ))` is 0.884 (0.842–0.990), and for the
      value-only variant 0.879 (0.812–0.996).
    - The installation share is 0.892 (0.833–0.977), and for the value-only variant 0.890 (0.831–0.970).
  - **Counts.** `F_head_rem < 1` in 36/36 and `F_head_ins > 0` in 36/36, for both variants.
  - About 88–89 % of the read edit's effect passes through the head's output at `p_t`. The value-only and full
    variants are nearly equal, so the attention change at `p_c` contributes little.
- **The isotropic controls.**
  - `F_rem` is below the frame's median over the 8 directions in 36/36 frames (margin median 1.033, minimum 0.838).
  - `F_ins` is above it in 36/36 (margin median 0.969, minimum 0.884).
  - Over all frames and directions, `F_rem_iso` lies in [0.924, 1.033] and `F_ins_iso` in [−0.214, 0.149].
- My values equal `ri.frame_values` at 0.0.

### 13. The unguarded smoke reference: PASS (it can affect neither phase)

- **What it is.** `smoke_reference.py` (sha256 `b2036a6b…`) ran unguarded at 08:15 UTC. HEAD was then `40829d2`; the
  confirmation file was untracked and the install commit `a857f9a` came later.
  - It ran frame `coordinated-adjective-1`'s S1 re-capture and its 55 arms through the same `ri` functions.
  - It wrote only `smoke_reference.pt`, of digests, beside itself, and computed no gate, table, fraction or statistic.
- **Why it ran the same code.** `git diff 40829d2 8f65062 -- src tests experiments/101-*/run.py` is empty.
- **Cross-process determinism.** Its 71 tensor digests and its logits digest equal the **production** measurements of
  that frame, 71/71 and equal. With the smoke this gives bitwise determinism across three processes: 08:15, 08:44 and
  08:45 UTC.
- **Its keys.** The 56 keys are calibration keys: spent exposed prompts in 020's ledger.
- **Could it affect anything?** No.
  - It ran no 101 prompt, and its prompts' token sequences are disjoint from all 240 sequences of the 101 frames and
    candidates.
  - Nothing it produced is read by `calibrate`, `lock`, `stage1` or `confirm`.
  - Only the launcher's smoke mode reads it, as an engineering comparison.

## Findings (most severe first)

**Blockers: none.**

**SF1 (should-fix, before lock): preserve the calibrate evidence in the repository, and correct the draft README.**
- **The problem.** Everything that documents the run outside the git-ignored state lives only in the session
  scratchpad `/private/tmp/…/scratchpad/exp101/calibrate/`:
  - the launcher (`0ad9d0e1…`);
  - `production_run.json`;
  - `production_runs.jsonl`, the only per-key execution log;
  - the rehearsal and smoke records;
  - the smoke reference;
  - `state_extract.json`.
- **The scenario.** The scratchpad is session-local and can be purged. A later review, or the confirmation review,
  could then no longer verify the calibration's execution. That is the situation the freeze review's SF1 named.
- **What was promised.** The launcher's docstring (`launch_calibrate.py:2–3`) promises a byte-identical copy into
  `experiments/101-read-component-intervention/evidence/`.
- **The README draft** (`calibrate/README.md`):
  - it has unfilled placeholders (`{OUTCOME}`, `{PRODUCTION}`, `{RESULTS}`, `{STATE_EXTRACT}`);
  - it lists `post_install_checks.out` and `SHA256SUMS`, which do not exist yet;
  - it says the rehearsal's "All 21 checks true", while the rehearsal recorded 20. The 21st, the production token
    check, exists only in production.
- **The fix.**
  - Copy the directory with `SHA256SUMS` under `evidence/` and fill in the README.
  - Keep every added file under `evidence/` or on a listed non-scientific path. Otherwise `lock` refuses: it requires
    no scientific path change since `8f65062` (`run.py:603–605`, `ri.scientific_changes`).
  - Commit `calibration-v1.json` alone, as `post_install_checks.py:67` expects.

**Notes.**
- **N1, the Level-1 margin at confirm** (item 11). It is 1.88× on the exposed data. The estimated exceedance
  probability at confirm is about 0.3 %, with an upper sensitivity of about 10 %. A failure would make confirm
  `NOT_INTERPRETABLE` and consume the frames under rule 2. This is for the human's lock decision and has no force here.
- **N2, G6 at calibration is a same-process check** (`run.py:546`, `ri.frame_gates`; G6 at `:1425–1427`). It compares
  the REF run with a re-capture from the same process, so by itself it cannot detect cross-process drift.
  - At calibration that role was filled by the environment check (drift exactly 0.0 against 020's three-week-old
    states), by the smoke, and by my smoke-reference comparison (13).
  - The cross-process G6 proper runs at confirm (`run.py:839`, against the stage-1 states).
- **N3, I4 is vacuous on reference-prompt runs**, as the plan states. It is exactly 0 on all 828 of them
  (`read_intervention.py:1395`). I4 thus tests the 972 cue-prompt G4 runs only; Level-1, readout and logit carry the
  identity content on the rest.
- **N4, the calibration keys of `coordinated-adjective-1` ran three times in total.** They ran once in the unguarded
  reference, once in the smoke and once in the phase. All are spent and exposed, outside every 101 ledger, and the
  results are bit-identical, so this compromises nothing (2, 13). The evidence README should say so in these terms.
- **N5, the run log records admissions, not completions.** `launch_calibrate.py:345–347` writes the line before the
  forward. With exit 0, equal accounting and `all_keys_ran_once` true, every logged key completed, but a failing run
  would still appear in the log. This matters for reading future logs.
- **N6, a git index refresh during the preflight.** `provenance.collect_git_state()` (`provenance.py:193–208`) runs
  `git status` without `--no-optional-locks`. The rehearsal's preflight therefore refreshed `.git/index`, whose mtime
  is 08:43:31 UTC, the rehearsal's start. This is harmless: no content changed. The launcher's own git calls use the
  flag.
- **N7, descriptive.** These are the observations under 10: the saturation of A_amp in 6/36 frames; the frozen
  readout's per-noun ceiling errors of up to 1.30 nats; τ possibly being tight. None may revise the calibration.

## Files (this directory)

| file | what |
|---|---|
| `PROMPT.md` | the brief |
| `REVIEW.md` | this report |
| `scripts/common.py` | the read-only write guard, my digest and canonical-JSON functions, my arm and tag list |
| `scripts/a_keys_ledger.py` → `out/a_keys_ledger.json`, `out/forbidden_101_sequences.json`, `out/exposed_frames.json` | check 1, independent: keys, ledger, run log, isolation, token sequences |
| `scripts/b_state_measurements.py` → `out/b_state_measurements.json`, `out/b_values.json` | checks 3–8 and 10, independent: state, measurements, landing, G2–G6, quantities, triggers |
| `scripts/c_exact_chain.py` → `out/c_exact_chain.json`, `out/c_per_run.json` | my float64 layers 3–5: Level-1, `E₁`, ceilings, G5, head deltas |
| `scripts/d_frozen_crosscheck.py` → `out/d_frozen_crosscheck.json`, `out/d_per_run_frozen.json` | the frozen cross-check under the guard: table, gates, payload, `load_calibration_101` |
| `scripts/e_tau_risk_descriptives.py` → `out/e_tau_risk_descriptives.json` | τ, P and power; G4 per arm type; the Level-1 risk; the head and isotropic quantities; the smoke reference against production |
| `scripts/f_level1_decomposition.py` → `out/f_level1_decomposition.json` | the Level-1 error split into its Δh6 and readout parts |
| `scripts/g_state_inputs.py` → `out/g_state_inputs.json` | the state's inputs, versions and phase rules |
| `out/repo_snapshot_start.txt`, `out/repo_snapshot_end.txt` | the repository before and after: HEAD, status, the outputs' sha256 and mtimes |

Run with `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python -B` and `HF_HUB_OFFLINE=1
PYTHONDONTWRITEBYTECODE=1`, from `scripts/`.
