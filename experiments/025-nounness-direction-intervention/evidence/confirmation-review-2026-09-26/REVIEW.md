> **Archived verdict: PASS WITH NOTES — no blockers.** Accepted by the reviewer on 2026-09-27, who authorized `report` exactly once and the closure.
> Everything below the rule is the independent reviewer's report, unchanged. Before this header was added, its sha256 was `59f59eb3ae3ef78516f730529b6e4a01d0d1f9ddbdbb1c9da97414ee45ae25b4`.

---

# Experiment 025: independent review of the production confirmation

- **Reviewer:** independent and read-only. **Date:** 2026-09-27.
- **Repository:** `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler` at
  `6f72d7d8afcaa079ee38256b7c1250352b1b1bdd`. The tree is clean and `origin/main` is equal.
- **Object:** the one production `confirm` of Experiment 025 (guarded launcher v3, 2026-09-26 19:23:57–21:12:44 UTC),
  its results state, its durable measurements, and the launcher's records.

## Verdict

**PASS WITH NOTES. No blockers.**

I reproduced the recorded outcome independently and exactly. My own code recomputed it from the saved measurements,
020's locked reference states and the freeze. The module's statistic functions were refused while that code ran.

| field | recorded | reproduced | max disagreement |
|---|---|---|---|
| label | `CAUSAL_EFFECT_NOT_ESTABLISHED` | `CAUSAL_EFFECT_NOT_ESTABLISHED` | — |
| A | 26/40 (FAIL) | 26/40 | 0.0 on every `A_i` |
| B | 23/40 (FAIL) | 23/40 | 0.0 on every `B_i` and `A_ij` |
| G | 37/40 (PASS) | 37/40 | 2.8e-17 on `G_i` |
| threshold | 27 | 27, the smallest k with P(X ≥ k) ≤ 0.025 under Bin(40, ½) | — |
| reference tail | 0.01923865414210013 | 5288280983/274877906944 = 0.01923865414210013; P(X ≥ 26) = 0.0403 | exact |

- **The scientific result can be accepted as final:** `CAUSAL_EFFECT_NOT_ESTABLISHED`.
- **The evidence may be archived.** Notes N1–N6 say what to record with it.
- **`report` may run exactly once,** from this state (`1d5d37f0…`). Its rendering was checked in memory; see item 14.
- **No confirm rerun is warranted.** Every gate passed, there is no incident, the accounting is exact, and the result
  is reproduced. A rerun is also refused by the protocol: `assert_phase_allowed("confirm")` requires `not_started`,
  and the sentinel exists. Under the stopping rule (item 15), nothing follows 025.

## Blockers

None.

## Notes (non-blocking)

- **N1. The exit-127 shell event was not a launch.** zsh treated the whole command string as one command name, so
  nothing ran: not `caffeinate`, not `env`, and not the Python launcher. It lasted 2.9 ms. It created no sentinel,
  loaded no model and changed no state, output or ledger. It was not a `runner.confirm()` invocation or a scientific
  attempt (item 1). Its three files were renamed to `shell_error_production.*` at 19:23:43 UTC, 11 s before the real
  launch, and the rename kept their inodes and times. Archive them with a one-line explanation.
- **N2. The operator's `verify_confirm.py` is not independent.** It scores through `cr.per_cue` and `cr.statistics`,
  and it recomputes `C` only on every 97th row. Its numbers are nevertheless right: every one of its values is
  reproduced here by independent code with full coverage.
- **N3. The launcher's `outside_repo_writes: 8` is library noise.** It is identical in the smoke, rehearsal and
  production records: `/dev/null` opens and system temp-file probes during imports and model loading. My own guarded
  processes show the same pattern. None of these writes is in the repository.
- **N4. The frozen attention program is not exact at the reference state.** Recomputing each frame's L4/L5 reference
  attention row from its own locked residuals gives a median difference of 6e-4 (L4) and 1.3e-3 (L5) from the
  captured rows, and at most 6.7e-3. The worst frame is `cardinal-012-1`, which is also where the Level-1 maximum lies
  (`knight|cardinal-012-1|rand4+0.32`). This is the known non-exactness that the Level-1 gate bounds (at most 0.0119
  against 2e-2). It does not enter `D_attn`, which compares captured rows only.
- **N5. The report and the write-up.**
  - `report` will print "No stratum is below its reporting trigger on a globally passing statistic." That is correct
    under the frozen rule: the trigger applies only to passing statistics, and for G the nouns have 17 ≥ 14.
  - The write-up should still state the stratum heterogeneity of A and B descriptively: adjectives 20/20 and 20/20,
    nouns 6/20 and 3/20.
  - It must keep the allowed reading of item 14 and must never say "no causal effect".
  - G's recorded `PASS` (37/40) may be reported as "G passed strongly", but only descriptively. It is not an
    alternative pass of the experiment and establishes no routing claim: after A's failure, the frozen hierarchy gives
    it no force.
- **N6. My guard audits Python-level file access only.** Its audit hook sees Python `open` calls, not C++-level
  `torch.save`/`torch.load`. My scripts `torch.save` only into this directory and memory-map the artifact read-only
  (private mapping). At the end, the repository and `outputs/experiment-025/` were checked unchanged: the file hashes,
  mtimes and file set, and a clean `git status`.

## How the review was done (hard-rule compliance)

- **Location.** Every script, log and output is in this directory (see Files). Nothing was written in the repository.
- **Never run:** the runner's `confirm`, `report`, `lock` or `freeze`, and the launcher in any mode.
- **Environment:** `.venv/bin/python` with `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`; git only as
  `git --no-optional-locks` (the guard refuses any other subprocess). `uv` was never used.
- **`guard.py`, imported first by every script:**
  - an audit hook refuses every write, delete, rename, link, mkdir, truncate or chmod under the repository (`.venv`
    included), and every read of `outputs/experiment-023/`, `outputs/experiment-024/` or `calibration-table.pt`;
  - `seal()` refuses `torch.nn.Module.__call__` and every capture and intervention entry point, and every alias of
    them in the `neural_decompiler` modules;
  - `seal_model()` refuses every module class's `forward` and the bridge's `run_with_hooks`, `run_with_cache` and
    `generate`.
- **The model** was loaded for its weights only, in the benchmark (b0), geometry (r3), gate (r4) and reference-row
  (r9, r9b) scripts.
  **No forward of any kind ran, and no refusal fired in any script.**
- **Independence.**
  - While my own responses and outcome were computed (r2), the guard refused these functions: `cr.per_cue`,
    `statistics`, `count_positive`, `routing_distance`, `outcome_label`, `descriptives`, `ladder`, `run_gates`,
    `identity_gates`, `level1_gates` and `recompute_c`; `rr.fresh_pair_cells`, `cue_mse` and `cue_mse_torch`. They
    were restored only afterwards, for the cross-check.
  - The geometry script (r3) refused every `cr` geometry function.
  - The gate workers (r4) refused `cr.run_gates`, `identity_gates`, `level1_gates`, `recompute_c`, `vector_factors`
    and `lexicon_of`. They ran my own loops, row indexing, reconstructed vectors and patched-vector lexicon through
    the allowed pinned routines only: `ul.compose_dx3`, `i3_error`, `contrast_of`, `block0_terms`,
    `reference_embeddings`, `reference_rows_017`, `rd.ReadoutProgram.level1_detail` (with its `contrast`), and
    `pm.lexicon_vector`.
- **Memory.** The artifact was memory-mapped read-only once per process (`torch.load(..., mmap=True)`).

## Items

### 1. Provenance

**The production record.**
- **The launcher.** `launch_confirm_v3.py` has sha256 `58b760bea9d8…d1e307` today. That equals:
  - the `launcher_sha256_before` in `production_command.txt`;
  - `environment.launcher_sha256` in `production_run.json`;
  - the sentinel's content.

  v1 is `aafd8954…` and v2 is `480fbf55…`. v1 was used only for the first rehearsal and v2 only for the crashed smoke
  test. Neither ran in production mode.
- **The commit** is `6f72d7d8…`, as the environment head and `head_after`.
- **The times.** The record gives `started_at` 19:23:57.846494Z and `ended_at` 21:12:43.518374Z; `production.exit`
  gives `exit 0` at 21:12:44Z. `exit_status` is 0 and `error` is null.
- **The events:**
  - `load_model` 1;
  - `refused` [], `forbidden_reads` [], `forbidden_writes` [];
  - `confirm_invocations` 1, `fresh_runs` 90,720;
  - `spent_runs` 8: a plain run and a patched θ = 0 run for each of the 4 patch-path keys;
  - `ledger_checks` 8 and `output_renames` 8;
  - 30,847,520 sanctioned module calls, exactly 340 per forward over all 90,728 forwards;
  - `tree_after` empty, HEAD unchanged.
- **The log** has 108 frame lines, then "90720 condition-tagged runs measured", then the one line `confirm complete:
  CAUSAL_EFFECT_NOT_ESTABLISHED (A 26/40, B 23/40, G 37/40; threshold 27); results sha256 aa3db023…`. It has no
  traceback and no `INCIDENT`.
- **The preflight:** all 22 checks true.
- **No earlier production-mode launch by any version.** v1, v2 and v3 all open the same `production_runs.jsonl` in
  append mode at import and create the same `confirm.sentinel`. So the birth times below (19:23:57.843 and .8466)
  exclude one by any of them.

**The exit-127 event (`shell_error_production.*`).**
- **Its files.** Its command file is byte-identical to `production_command.txt` except `launched_at`: 19:23:09Z
  against 19:23:54Z. Its log is a single zsh line, `(eval):1: no such file or directory: caffeinate -i -s env
  CONFIRM025_MODE=production … launch_confirm_v3.py`. zsh looked for a command named after the entire string, which
  is why the exit status is 127. So `caffeinate`, `env` and Python never executed.
- **Its times.** The three files were born between 19:23:09.558234 and 19:23:09.558997. The exit file was last
  written at 19:23:09.561141, so the event lasted 2.9 ms. Importing torch alone takes seconds. Their ctimes are
  19:23:43.62: they were renamed then, with inodes 34712739–41 kept.
- **Where the launcher's first side effects would show.** In production mode, the launcher's first side effects
  happen at import:
  - it opens `production_runs.jsonl` in append mode. That file was born at 19:23:57.843134, between the record's
    `recorded_at` (19:23:57.841804) and its `started_at` (19:23:57.846494);
  - it creates the sentinel with `O_CREAT|O_EXCL`. The sentinel was born at 19:23:57.846591, 97 µs after
    `started_at`. Its inode 34712850 is above `production_command.txt`'s 34712837, which was created at 19:23:54.

  Had the launcher started at 19:23:09, both files would carry that birth time. A sentinel would also have existed,
  so the `O_EXCL` creation at 19:23:57 would have failed. The record instead shows `error` null.
- **The state and outputs.**
  - At 19:23:57+, the production preflight asserted:
    - `results_state_is_the_lock_state` (`8eeb8184…`, lock complete, confirm `{"status": "not_started"}`,
      `confirmation` null);
    - `ledger_empty` and `confirm_eligible_exactly_once`;
    - `only_the_lock_outputs_exist`.
  - Independently, resetting the four fields confirm writes (the confirm phase, the two ledgers, `confirmation`) in
    today's final state reproduces `8eeb8184…` exactly. Confirm changed nothing else in the state.
- **Conclusion.** The Python launcher never started, and no sentinel was created by the event. No model loaded, and no
  state, output or ledger changed. It was neither a `runner.confirm()` invocation nor a scientific attempt.

### 2. The pre-fresh barrier

Each step below held before the first fresh run.

1. **The cwd.** At module level, the launcher does `os.chdir(ROOT)` and asserts the cwd. The preflight's
   `N1_cwd_is_repo_root` is true, and the recorded `environment.cwd` is the root.
2. **The git, environment and checkpoint bindings.**
   - The preflight's N1 checks passed: HEAD = `origin/main` = remote = `6f72d7d`, a clean tree, and the runner's own
     `collect_git_state()` from the root.
   - N2 passed with no difference from the lock run: Python, platform, torch, transformers, transformer-lens, hub,
     numpy, 4 threads, model, tokenizer revision, and the checkpoint files.
   - Inside `runner.confirm()`, before any prompt, it checks `_check_runtime`, the parameter digest, and (after
     `from_model`) the embedding digest.
   - Today, the checkpoint files equal the production record (r7), and the parameter and embedding digests equal the
     lock's (r3).
3. **The lock and the preregistration.**
   - In the preflight: the installed lock (`1881a790…`, content `8e5300de…`), the preregistration (`e4b12656…`,
     rendered-equal) and the production token.
   - The module binding `622aa832…`: the lock = git HEAD = the imported file = `own_blob()`. The `rr` binding and all
     13 pinned blobs also verified.
   - `full_validate_lock` passed in the preflight, and again inside `confirm` (`run.py` l. 422).
   - Today: `cr.validate_lock` passes on today's tree, all pinned blobs verify, and the 58 paths changed since
     `d3ecbd6` are all evidence, README or installed-artifact paths (`scientific_changes` = []).
4. **I7′.** The state records `phases.confirm.I7` = `{"bitwise_equal": true, "differing": []}`. My own reconstruction
   (r3; items 5 and 11) equals the lock bit for bit.
5. **The four patch-path checks.** `patch_path_check.passed` is true. For each of the 4 keys, both
   `plain_equals_patched_theta0` and `embed_equals_resid_pre0` are true. The launcher refused a spent run once a fresh
   run had happened; nothing was refused, so all 8 spent runs came first.
6. **Only then the ledger.**
   - The code order is I7′ (l. 436) → the patch path (l. 443) → the running phase with I7 and the patch-path record,
     the ledger, the nouns, and one write (l. 456–461).
   - The launcher's fresh-run guard required `_ledger_on_disk()`: 90,720 keys in `results.json` and no spent key.
   - `phases.confirm.started_at` is 19:24:19, the ledger write.

**The patch-path keys stay outside the ledger and the accounting.**
- None of the 4 keys appears, untagged or as a key prefix, in the ledger, the run log or the freeze manifest.
- All 90,720 ledger keys have 4 fields.
- The launcher checked every one of the 8 state writes for spent or untagged keys; none was refused.
- The accounting counts 90,720.
- Today, the keys are in 020's ledger, inside the 43,632-key spent set, and disjoint from 025's 4,320 untagged and
  90,720 tagged keys (r7).

### 3. Accounting

**The keys.**
- My own 90,720 keys come from the freeze: 40 cues × 108 frames × 21 conditions, all unique.
- **The freeze manifest:** 90,720 entries, sorted, set-equal.
- **The run log:** 90,720 lines, 90,720 unique; 0 duplicate, 0 missing, 0 extra; `n` contiguous from 1 to 90,720.
- **The ledger:** 90,720 unique; 0 duplicate, 0 missing, 0 extra.
- **Per condition:** exactly 4,320 for each of the 21 conditions, in both the log and the ledger.
- **The state:** `accounting = {manifest 90720, executed 90720, ledger 90720, equal true}`, and `stage2.n_executed`
  is 90,720.
- **The artifact:** 2,880 cue-final and 1,440 coordinated rows × 21 conditions = 90,720 slots, all filled and finite.
  Every row's saved `positions` = (p_c, p_t) and `keys` = p_t + 1 match my own (token id, `frame_id`) row mapping.

**The pairing proof.**
- I rebuilt the runner's loop order myself: frames sorted by `frame_id`, cues by token id, conditions in frozen order.
  **The log's n-th key equals the n-th loop key for all n = 1…90,720: 0 mismatches.** The first is
  `cardinal-009-1|author|2488|base`, the last `quantifier-new-2|careless|48292|plur-0.32`.
- The production log's frame order equals the sorted frame ids, and every 21-line block is in frozen condition order.
- **Where the log's condition comes from.** The condition in each log key is the one whose float32 vector has the
  replacement's digest. My reconstruction of all 840 vectors equals the lock bit for bit. Their 840 digests are
  distinct, and no spent row's digest is among them, so the digest → (cue, condition) map is injective.
- **The chain from vector to slot.**
  - At iteration n, the runner passes `vectors32[token][condition_n]` and stores the result in slot
    (group, `condition_n`, row(token, frame)).
  - The guard logs the condition read from the vector itself.
  - The two agree for every n.

  **Hence every stored slot holds the run of its labelled condition.**

**The I1 argument.** I1 ties each saved `x1` to its labelled condition's vector, independently of the log.
- **What I1 checks.** On every run it compares the saved `Δx1` with `d_emb + ΔE + ΔA0`, a function of the patched
  vector `v` alone.
- **With the correct labels:** at most 2.15e-5 over all 90,720 runs (item 5).
- **With a deliberately wrong vector on the same saved `x1`** (r4, 4,320 runs per case, every cue × frame):

  | slot ← vector | min I1 | per stratum (adjective / noun) |
  |---|---|---|
  | noun+0.32 ← noun−0.32 | 0.634 | 0.634 / 1.330 |
  | noun−0.32 ← noun+0.32 | 0.634 | 0.634 / 1.330 |
  | noun+0.32 ← base | 0.372 | 0.372 / 0.484 |
  | noun+0.32 ← plur+0.32 | 0.379 | 0.379 / 0.666 |
  | rand1+0.32 ← rand1−0.32 | 0.290 | 0.315 / 0.290 |
  | rand1+0.32 ← rand2+0.32 | 0.197 | 0.215 / 0.197 |
  | rand4+0.32 ← rand4−0.32 | 0.314 | 0.325 / 0.314 |
  | plur+0.32 ← plur−0.32 | 0.375 | 0.375 / 0.399 |
  | plur+0.32 ← noun+0.32 | 0.379 | 0.379 / 0.666 |

  **The smallest wrong-vector I1 is 0.197.** That is 9,162 times the largest correct I1, and 1,973 times the 1e-4
  tolerance.
- **Conclusion.** A single swapped or mislabelled slot anywhere would have failed the recorded I1 gate by orders of
  magnitude. The gate passed on every run, and I recomputed it on every run.

### 4. The durable artifact

- **The file:** `stage2-measurements.pt`, 898,673,151 bytes, sha256 `87284bdd…a7788` today.
- **The schema:** 256 tensors, set-equal to the expected names: per group, `positions` and `keys`, then 21 conditions
  × {`x1`, `x3`, `dc`, `C`, `rows4`, `rows5`}.
- **The dimensions** (rows N = 2,880 cue-final and 1,440 coordinated):
  - `x1` and `x3`: [N, 2, 512], float32;
  - `dc` and `C`: [N, 79], float64;
  - `rows4` and `rows5`: [N, 8, 6] cue-final and [N, 8, 9] coordinated, float32.
- **Finiteness:** everything is finite. Attention entries beyond p_t are exactly 0. In cue-final frames, slot 1 equals
  slot 0 bit for bit.
- **The digests.** All 256 `rc.tensor_digest` values equal `state.confirmation.stage2.tensors_sha256`. All 256 also
  equal my own implementation of the digest definition.
- **The scores came from the saved, re-read artifact** (`run.py` l. 466–491):
  1. `save_durably` (with fsync), then the digests;
  2. the stage-2 record is written, then `tensors.clear()`;
  3. the accounting, then `saved = torch.load(stage2_path)`, and its digests are re-checked;
  4. `assert_measurements`, `recompute_c`, `run_gates`, `per_cue` and `statistics`, then `_descriptives` (the records
     and the ladder), all on `saved`.

  Empirically too: my recomputation from today's file reproduces every recorded response exactly (item 6).

### 5. The validity gates, recomputed

**Patch fidelity held on every run.**
- `pm.run_patched` calls `_check_execution` on every run. It raises an `IncidentError` on any problem: an inexact
  replacement, a change outside the patched site, or a wrong hook, position, source, shape or dtype.
- In stage 2, such an error would have become a confirm incident with no result.
- There is none, so fidelity held on all 90,720 runs.

**Coverage: all of it.** Two weights-only workers (r4) recomputed every gate on every run: 108 frames × 40 cues × 21
conditions = 90,720 runs, each exactly once, 4,320 per condition. That is more than the required minimum. Each run
used:
- my own row indexing;
- my own reconstructed float32 vector (bit-identical to the lock, r3);
- my own `MLP₀(LN₂(v))`, bit-identical to `pm.lexicon_vector` on the base vectors.

My vector I1 factors equal `cr.vector_factors` exactly (checked only afterwards). The two workers took 51 minutes
each, at 4 threads, the production thread count.

| gate | tolerance | recorded (max, at) | recomputed (max, at) | max disagreement |
|---|---|---|---|---|
| C from the saved Δx3 | bit for bit | bit for bit over 90,720 | **90,720 / 90,720 bit-equal**, including all 69,120 outcome-bearing runs | 0.0 |
| I1 (vector) | 1e-4 | 2.1541116871581778e-05 at `knight\|cardinal-017-3\|noun+0.16` | the same value, the same location | 0.0 |
| I3 | 1e-4 relative | 4.35303313102457e-05 at `painter\|quantifier-016-4\|rand5-0.32` | the same value, the same location | 0.0 |
| I4 | 1e-3 | 9.004566446524365e-05 at `lonely\|cardinal-new-2\|rand6-0.32` | the same value, the same location | 0.0 |
| Level-1, exactly the 16 outcome-bearing conditions (69,120 runs) | 2e-2 | 0.011862701913658569 at `knight\|cardinal-012-1\|rand4+0.32` | the same value, the same location | 0.0 |
| Level-1, the 5 secondary conditions (recorded, not gated) | — | 0.010784592608527932 at `rapid\|coordinated-adjective-018-3\|plur-0.32` | the same value, the same location | 0.0 |

**Level-1 per condition** (the 16 outcome-bearing conditions are gated). Every value is within 0.02. All 21
per-condition maxima and locations equal the operator's `verify_confirm.json` (max |Δ| 0.0).

| condition | max \|L1 − Δc\| | at | I1 max | I3 max | I4 max |
|---|---|---|---|---|---|
| noun+0.32 | 0.009483 | prince\|cardinal-012-1 | 1.50e-5 | 1.76e-5 | 3.92e-5 |
| noun−0.32 | 0.009742 | knight\|cardinal-012-1 | 1.43e-5 | 3.10e-5 | 5.63e-5 |
| rand1+0.32 | 0.010428 | bishop\|cardinal-013-1 | 1.70e-5 | 2.13e-5 | 4.56e-5 |
| rand1−0.32 | 0.010619 | knight\|cardinal-013-1 | 1.58e-5 | 2.98e-5 | 6.57e-5 |
| rand2+0.32 | 0.010074 | tourist\|cardinal-013-1 | 1.31e-5 | 2.73e-5 | 6.77e-5 |
| rand2−0.32 | 0.010407 | onion\|cardinal-012-1 | 1.57e-5 | 3.05e-5 | 7.60e-5 |
| rand3+0.32 | 0.011139 | knight\|cardinal-012-1 | 1.52e-5 | 3.38e-5 | 7.47e-5 |
| rand3−0.32 | 0.009701 | knight\|cardinal-012-1 | 1.84e-5 | 2.48e-5 | 5.55e-5 |
| rand4+0.32 | **0.011863** | knight\|cardinal-012-1 | 1.56e-5 | 3.42e-5 | 6.42e-5 |
| rand4−0.32 | 0.010036 | knight\|cardinal-012-1 | 2.07e-5 | 3.44e-5 | 7.89e-5 |
| rand5+0.32 | 0.010934 | weary\|cardinal-018-1 | 1.40e-5 | 3.26e-5 | 7.64e-5 |
| rand5−0.32 | 0.010211 | tourist\|cardinal-012-1 | 1.60e-5 | **4.35e-5** | 8.69e-5 |
| rand6+0.32 | 0.010898 | prince\|cardinal-012-1 | 1.48e-5 | 2.38e-5 | 3.76e-5 |
| rand6−0.32 | 0.010424 | tourist\|cardinal-012-1 | 1.58e-5 | 3.21e-5 | **9.00e-5** |
| rand7+0.32 | 0.010610 | knight\|cardinal-012-1 | 1.52e-5 | 2.65e-5 | 6.09e-5 |
| rand7−0.32 | 0.010142 | knight\|cardinal-012-1 | 1.52e-5 | 2.68e-5 | 4.43e-5 |
| *base* | 0.010141 | onion\|cardinal-012-1 | 2.11e-5 | 3.51e-5 | 8.19e-5 |
| *noun+0.16* | 0.010175 | tourist\|cardinal-012-1 | **2.15e-5** | 1.91e-5 | 3.58e-5 |
| *noun−0.16* | 0.010527 | tourist\|cardinal-012-1 | 1.66e-5 | 3.64e-5 | 6.76e-5 |
| *plur+0.32* | 0.010135 | careful\|coordinated-adjective-018-3 | 1.84e-5 | 1.32e-5 | 3.02e-5 |
| *plur−0.32* | 0.010785 | rapid\|coordinated-adjective-018-3 | 1.48e-5 | 2.28e-5 | 6.00e-5 |

Across the 16 outcome-bearing conditions, the maxima are I1 2.07e-5, I3 4.35e-5 and I4 9.00e-5; the overall I1
maximum is in the secondary `noun+0.16`. No gate is near its tolerance: Level-1 uses 59% of its bound, I1 22%, I3
44% and I4 9%.

**Also verified:**
- I7′: my geometry equals the lock bit for bit (item 11). The recorded I7 is `bitwise_equal: true`.
- The 020–024 inputs verify, today and at the recorded re-check (item 13).
- Every measurement is finite (item 4).

### 6. `ℓ` and `D_attn`, rebuilt independently

**What I computed.** For every cue and condition (40 × 21), over the 108 frames × 79 nouns:
- MSE = Σ SSE_C / Σ n, with SSE_C = Σ_nouns (Δc − C)² per pair and Σ n = 8,532;
- ℓ = ln MSE;
- nMSE = Σ SSE_C / Σ Q, with Q = Σ Δc²;
- the per-template ℓ;
- `D_attn`: for each run, the total-variation distance ½ Σ_{k ≤ p_t} |a − a_ref| of each of the 16 L4/L5 heads,
  against 020's locked rows for that frame; the equal mean over the 16 heads; then the mean over the 108 frames.

**Disagreement with the recorded responses:**
- MSE, ℓ, nMSE and the per-template ℓ are exactly equal (0.0);
- `D_attn` agrees to at most 1.1e-16 (relative 3.7e-16);
- the frame count per cell is 108 in every cell.

**The module, run only afterwards.**
- `cr.per_cue`: 0.0 on MSE, ℓ and nMSE, and 1.1e-16 on `D_attn`.
- `cr.statistics` equals the recorded results exactly.
- On sampled runs, `rr.fresh_pair_cells` agrees to 0.0 and `cr.routing_distance` to 5.6e-17.

**Row alignment.**
- For all 108 frames, the locked `rows4`/`rows5` have shape [8, p_t + 1], and their `p_c` and `p_t` equal the frame's.
- The rows sum to 1 within 1.7e-7 (reference) and 2.1e-7 (captured). The saved rows are zero beyond p_t.
- Each saved row's frame, heads and keys match the stored `positions` and `keys` under my own mapping.
- **Frame binding (r9, weights only).** For every frame, the L4/L5 attention recomputed from that frame's own locked
  residuals matches its locked rows within 6.7e-3. Every other same-length frame's rows differ by at least 0.0755.
  The lock review had also verified bit for bit, by capture, the rows of 6 spent reference prompts.
- A naive test was not diagnostic, so I did not use it: whether the fresh base runs are nearest to their own frame's
  rows. A fresh cue moves attention more than a frame change does.

### 7. A

- For every cue, `A_i = ½[ℓ(noun+0.32) − ℓ(noun−0.32)]`. **26 of 40 are strictly positive.**
- All 40 equal the recorded values: maximum |Δ| 0.0, and every sign identical.
- There are no zeros or ties. The smallest |A| is 9.5e-4 (`author`, −0.000946); it is negative and counts against.
- The threshold is 27, since P(X ≥ 27) = 0.01924 ≤ 0.025 < P(X ≥ 26) = 0.0403. **A FAILS.**
- Per-cue values: see the table below.

### 8. B

- `A_ij` for the 7 controls: maximum |Δ| 0.0 against the recorded `A_random`.
- `B_i = A_i − (1/7)Σ_j|A_ij|` has maximum |Δ| 0.0. **23 of 40 are positive.**
- The smallest |B| is 0.0116; there are no zeros. **B FAILS.**

### 9. G

- `G_i = ½[D_attn(noun+0.32) − D_attn(noun−0.32)]` has maximum |Δ| 2.8e-17. **37 of 40 are positive.**
- The negatives are `dancer` −0.000217, `lawyer` −0.002462 and `nurse` −0.004152. The smallest |G| is 2.2e-4, far
  above the disagreement.
- **G PASSES its count.** It cannot change the label, because A failed (item 10).

### 10. The outcome precedence

**`cr.outcome_label`.** I evaluated all 8 combinations. Every combination with A failing returns
`CAUSAL_EFFECT_NOT_ESTABLISHED`, whatever B and G are.

**`cr.statistics`** computes the label only as `outcome_label(A PASS, B PASS, G PASS)`. B and G enter only after A
passes. `NOT_INTERPRETABLE` arises only through an incident, and here:
- there is none;
- the gates were written and then enforced before scoring;
- the recorded gate values pass their tolerances.

The only legal label is `CAUSAL_EFFECT_NOT_ESTABLISHED`, which is the one recorded. My own label code gives it too.

### 11. The adjective/noun split as a correctness check

The split reproduces: **adjectives A 20/20, B 20/20, G 20/20; nouns A 6/20, B 3/20, G 17/20.**

**The cues and strata.**
- `results.per_cue` lists (word, token id, stratum) in freeze order, identical to the freeze. The lock's per-cue
  geometry carries the same word and stratum.
- Re-tokenizing with the pinned tokenizer: every cue id is exactly `' ' + word` as a single token, and every noun's
  plural id is its regular plural's single token. 0 mismatches (r10).

**The ± mapping, on the actual float32 patches.**
- s(noun+0.32) − s(noun−0.32) lies in [0.6399999969, 0.6400000031] for all 40 cues (adjectives 0.63999999693–0.64000000210,
  nouns 0.63999999874–0.64000000305). The odd-component deviation is at most 1.5e-9.
- For every cue, s(noun+0.32) > s(base) > s(noun−0.32): the + vector raises the frozen score in both strata.

**The t̂ sign.** d·t̂ = τ > 0 for every cue (|d·t̂ − τ| ≤ 8.9e-16), so t̂ points along +d in both strata; there is no
reversal. My t̂, controls, plurality tangents and 21 vectors per cue equal the lock's digests bit for bit (r3).

**The lookup and pairing.**
- The 840 vector digests are distinct.
- The log's digest-derived condition equals the slot's label for every run (item 3).
- I1 binds each saved `x1` to its labelled vector (item 3). Swapping noun+ and noun− would give I1 ≥ 0.634 for
  adjectives and ≥ 1.330 for nouns, against at most 2.2e-5 observed. Neither stratum has a sign reversal or a swapped
  slot.
- The condition names parse as intended (`cr.PRODUCTION.spec`): `noun+0.32` is sign +1 with odd 0.32, `noun-0.32`
  is sign −1, and `rand{j}±0.32` is the control j with its sign.
- A is built from exactly `noun±0.32`. No cue's A equals its half-dose, plurality or any random-control analogue.

**Is the heterogeneity real data or an artifact?** It is real data, and consistent in every cut. The figures are
descriptive only: no test is made and no stratum becomes a primary criterion.

- **Adjectives:**
  - `A > 0` in each template (20, 20 and 20 of 20);
  - the half dose gives 20/20 and nMSE gives 20/20;
  - + is above the baseline for 20/20 and − is below it for 20/20;
  - within each adjective, SSE(+) > SSE(−) in 72%–90% of its 108 frames (median 82%);
  - `A` ranges from 0.136 to 0.464 (median 0.268).
- **Nouns:**
  - `A` ranges from −0.199 to +0.148 (median −0.051);
  - frame-level SSE(+) > SSE(−) has a median of 45% (range 19%–78%);
  - the half dose gives 5/20, nMSE 4/20, and + above the baseline 3/20;
  - the noun baseline ℓ is higher: median −2.22, against −2.76 for adjectives.
- **The strata coincide with the sign of the starting score.** The adjectives' s₀ runs from −0.306 to −0.068, the
  nouns' from +0.069 to +0.285. Across all 40 cues, Spearman(A, s₀) = −0.83; within the nouns it is −0.63. The effect
  weakens as the starting nounness rises: a continuous geometric property, not a label.
- **The orthogonalized plurality control does not track the split:** its A is positive for 8/20 in each stratum.

### 12. Descriptive quantities

These were checked only after the official result was reproduced. All reproduce, with a maximum per-cue difference
of 5.6e-17.
- **Medians:** A 0.14191, B 0.08493, G 0.01561. The means are A 0.11458, B 0.06289 and G 0.01570.
- **Counts:**
  - half-dose A 25/40;
  - plurality A 16/40;
  - nMSE A 24/40;
  - G above random 35/40;
  - + above the baseline 23/40;
  - − below the baseline 30/40.
- **Template A counts:** cardinal 26, coordinated-adjective 31, quantifier 25.
- **Even components:** nounness 0.014447 and random 0.014086.
- **The causal fraction:** 0.341344 = 0.141912 / (1.2992017 × 0.32).
- **The stratum trigger** is 14. No `below_trigger` flag can fire for A or B, because they failed.
- **The ladder.** My own pooled sums over all 4,320 pairs, from the r4 workers' `level1_detail` with and without the
  L4 heads, equal the recorded ladder exactly (max |Δ| 0.0):

  | condition | SSE C | SSE C5 | SSE L1 | block-5 attention share | block-4 attention share |
  |---|---|---|---|---|---|
  | base | 30944.296 | 20958.988 | 0.492 | 0.322687 | 0.677298 |
  | noun+0.32 | 33801.708 | 23431.506 | 0.488 | 0.306795 | 0.693190 |
  | noun−0.32 | 28969.712 | 19430.451 | 0.492 | 0.329284 | 0.670699 |

None of these can alter the result. They are computed only after the atomic result write, and they are written only
under `confirmation.descriptives` (item 13). `SEMANTICS["secondary"]` says they "can never rescue, alter or redefine"
the result.

### 13. The write order

**The code order** (`run.py`), with the state writes numbered:
1. The ledger write (l. 461).
2. Stage 2, the durable save and the stage-2 record (l. 469).
3. The accounting, written (l. 475) and then enforced.
4. The re-read and its digests; `assert_measurements`; `C`, written (l. 484) and then enforced.
5. The gates, written (l. 489) and then enforced.
6. Scoring (l. 491–492), then the 020–024 binding re-check `_recheck()` (l. 493–495).
7. **One atomic write** of the results, `lock_sha256`, `completed_at` and the completed phase (l. 504–508), through
   `write_state_atomic`: a temporary file, fsync, `os.replace`, and a directory fsync.
8. The log line, then the descriptives (l. 517–543), which write only `confirmation.descriptives`. A failure there is
   recorded and the result stands.

That is 8 state writes in all. The launcher saw 8 renames and made 8 ledger checks.

**No incident.** `confirmation.incident` is absent and no phase has incidents.

**The digests.**
- Removing `confirmation.descriptives` from today's final state reproduces the result-write digest
  `aa3db023c4a4…aad11907c4` exactly: the digest logged at the write.
- The remainder is identical to the final state. So the result-write state already held the results and the
  completed phase, and the two descriptive writes added only the descriptives.
- The final state digest `1d5d37f035bc…a98b33c` verifies. The file sha256 is `c827bb69…`, equal to the launcher
  record.
- The results completed at 21:09:04. The final state file was born at 21:12:42.96, at the last descriptive write.

### 14. The interpretation

The allowed reading of `CAUSAL_EFFECT_NOT_ESTABLISHED`, from `cr.SEMANTICS["outcomes"]`, the criterion text and the
preregistration:

> The preregistered test did not establish that moving fresh cue embeddings in the +nounness direction consistently
> causes greater frozen block-4/5 readout-approximation error than the matched −nounness intervention across the
> frozen 40-cue population.

- **It must not be rewritten as "no causal effect".** The criterion is a sign-count consistency criterion over a
  mechanically selected population, not a design-based randomization test. 26/40 is one short of 27, and
  P(X ≥ 26) = 0.040 > 0.025.
- **What may be reported, only as descriptive findings and never as alternative PASSes:**
  - G passed strongly (37/40), but after A's failure it establishes no label and no routing claim;
  - the effect is highly heterogeneous by stratum;
  - all 20 adjectives have positive A, B and G;
  - noun cues mostly have positive G (17/20) but not positive A (6/20) or B (3/20).
- **Not claimed by any label:**
  - semantic nounhood as the model's variable;
  - a general routing controller;
  - algorithm selection in general;
  - generalization beyond this checkpoint and this mechanism;
  - that the effect excludes a plurality-direction contribution (cos +0.235).
- **What `report` will print.** The in-memory rendering (`cr.render_report(state)`, a pure function, nothing written)
  prints exactly the allowed reading as its label line. Its A/B/G table (26/23/37, with the stratum columns 20/6,
  20/3, 20/17) is correct, and its text never says "no causal effect". The lock's semantics equal `cr.SEMANTICS`.

### 15. The stopping rule

- **The texts:**
  - the design: "After 025, on any clean result, the experiments stop, and the project is written up";
  - the plan: "After 025, on any clean result, stop";
  - the README: "This is the final experiment of the sequence".
- **This is a clean result:** there is no incident, every gate passed, and the label is the one the preregistered rule
  gives. **The sequence ends.**
- **024 is unaffected.** It remains a prospective predictor, and its predictive result stands. 025 tested a causal
  claim, and not establishing it does not retract 024's association.
- **No Experiment 026 exists:** no directory, design, plan, tracked file or commit. None should follow to rescue this
  result, for example with a noun-only rerun, a new dose, or post hoc heads or strata. The design also notes that 025
  consumed 20 of the last 23 curated fresh adjectives.

## Per-cue values (own code; recorded values equal)

| # | stratum | word | token id | s₀ | A (mine) | mean\|A_ij\| | B (mine) | G (mine) | recorded A/B/G equal (Δ max) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | adjective | anxious | 20138 | -0.1749 | +0.257429 | 0.038093 | +0.219335 | +0.018879 | yes (0.0e+00) |
| 2 | adjective | cheerful | 39567 | -0.1912 | +0.271194 | 0.082494 | +0.188700 | +0.010888 | yes (2.8e-17) |
| 3 | adjective | curious | 14338 | -0.1983 | +0.264970 | 0.056849 | +0.208122 | +0.024291 | yes (0.0e+00) |
| 4 | adjective | jealous | 23327 | -0.1321 | +0.148249 | 0.044780 | +0.103469 | +0.024276 | yes (0.0e+00) |
| 5 | adjective | lonely | 25106 | -0.1423 | +0.319755 | 0.052713 | +0.267042 | +0.018539 | yes (2.8e-17) |
| 6 | adjective | nasty | 26321 | -0.2223 | +0.339904 | 0.049069 | +0.290835 | +0.019259 | yes (0.0e+00) |
| 7 | adjective | careful | 10182 | -0.2349 | +0.443823 | 0.119656 | +0.324167 | +0.035595 | yes (0.0e+00) |
| 8 | adjective | careless | 48292 | -0.2366 | +0.135820 | 0.047978 | +0.087842 | +0.000647 | yes (0.0e+00) |
| 9 | adjective | famous | 8530 | -0.1251 | +0.234023 | 0.029900 | +0.204123 | +0.020676 | yes (0.0e+00) |
| 10 | adjective | friendly | 11453 | -0.1020 | +0.284867 | 0.072879 | +0.211988 | +0.026957 | yes (0.0e+00) |
| 11 | adjective | gorgeous | 23535 | -0.2207 | +0.297097 | 0.039672 | +0.257425 | +0.032306 | yes (0.0e+00) |
| 12 | adjective | hungry | 18254 | -0.0676 | +0.308622 | 0.031415 | +0.277208 | +0.018218 | yes (0.0e+00) |
| 13 | adjective | weary | 35725 | -0.1318 | +0.276421 | 0.030723 | +0.245698 | +0.031534 | yes (0.0e+00) |
| 14 | adjective | wicked | 26395 | -0.1409 | +0.241402 | 0.066234 | +0.175168 | +0.012393 | yes (0.0e+00) |
| 15 | adjective | ugly | 19513 | -0.2601 | +0.325902 | 0.050758 | +0.275144 | +0.017490 | yes (0.0e+00) |
| 16 | adjective | vivid | 24863 | -0.2890 | +0.196697 | 0.068412 | +0.128286 | +0.009568 | yes (0.0e+00) |
| 17 | adjective | vague | 21248 | -0.3059 | +0.464140 | 0.084175 | +0.379965 | +0.019907 | yes (0.0e+00) |
| 18 | adjective | rapid | 5233 | -0.2834 | +0.195304 | 0.078638 | +0.116666 | +0.007854 | yes (0.0e+00) |
| 19 | adjective | rigid | 16572 | -0.1948 | +0.239528 | 0.021611 | +0.217916 | +0.017053 | yes (0.0e+00) |
| 20 | adjective | clever | 19080 | -0.2575 | +0.160296 | 0.027671 | +0.132625 | +0.009171 | yes (0.0e+00) |
| 21 | noun | soldier | 15796 | +0.2315 | -0.011332 | 0.068468 | -0.079799 | +0.006837 | yes (0.0e+00) |
| 22 | noun | sailor | 45758 | +0.2486 | -0.082372 | 0.031427 | -0.113800 | +0.009748 | yes (2.8e-17) |
| 23 | noun | priest | 14959 | +0.1751 | +0.008827 | 0.020782 | -0.011955 | +0.012587 | yes (0.0e+00) |
| 24 | noun | knight | 30605 | +0.1628 | +0.031537 | 0.043131 | -0.011594 | +0.021130 | yes (0.0e+00) |
| 25 | noun | onion | 21635 | +0.1386 | -0.116610 | 0.074760 | -0.191370 | +0.030211 | yes (2.8e-17) |
| 26 | noun | carrot | 47215 | +0.1661 | -0.058519 | 0.059429 | -0.117947 | +0.004309 | yes (0.0e+00) |
| 27 | noun | pirate | 42404 | +0.0690 | +0.111021 | 0.032192 | +0.078829 | +0.024477 | yes (0.0e+00) |
| 28 | noun | tourist | 22777 | +0.0702 | +0.148004 | 0.065987 | +0.082017 | +0.035307 | yes (0.0e+00) |
| 29 | noun | author | 2488 | +0.1282 | -0.000946 | 0.053296 | -0.054241 | +0.012053 | yes (0.0e+00) |
| 30 | noun | bishop | 29417 | +0.2110 | -0.053556 | 0.027015 | -0.080571 | +0.007213 | yes (0.0e+00) |
| 31 | noun | dancer | 42411 | +0.2852 | -0.126271 | 0.035243 | -0.161514 | -0.000217 | yes (0.0e+00) |
| 32 | noun | duck | 27985 | +0.1449 | +0.084376 | 0.071015 | +0.013360 | +0.034236 | yes (0.0e+00) |
| 33 | noun | goat | 23244 | +0.1714 | -0.199491 | 0.061250 | -0.260742 | +0.014176 | yes (0.0e+00) |
| 34 | noun | guitar | 12609 | +0.1658 | -0.115695 | 0.051602 | -0.167298 | +0.023842 | yes (0.0e+00) |
| 35 | noun | hunter | 32290 | +0.1452 | +0.015874 | 0.039254 | -0.023379 | +0.008851 | yes (0.0e+00) |
| 36 | noun | lawyer | 11115 | +0.1814 | -0.131968 | 0.050118 | -0.182086 | -0.002462 | yes (2.8e-17) |
| 37 | noun | monk | 35295 | +0.1179 | -0.045195 | 0.036677 | -0.081872 | +0.006189 | yes (0.0e+00) |
| 38 | noun | nurse | 15339 | +0.2421 | -0.159224 | 0.073386 | -0.232609 | -0.004152 | yes (0.0e+00) |
| 39 | noun | painter | 27343 | +0.2397 | -0.071416 | 0.046295 | -0.117711 | +0.003053 | yes (0.0e+00) |
| 40 | noun | prince | 24012 | +0.1695 | -0.049329 | 0.032328 | -0.081657 | +0.005236 | yes (0.0e+00) |

## Files (this directory)

| file | what |
|---|---|
| `PROMPT.md` | the brief |
| `guard.py` | the shared read-only guard (audit hook, forward and entry-point refusals) |
| `b0_bench.py` | the cost benchmark (no forward) |
| `r1_records.py`, `.json`, `.log` | items 1, 2, 3 and 13: provenance, the exit-127 timeline, the keys, the log ↔ loop-order pairing, the digests |
| `r2_scores.py`, `.json`, `.log`, `r2_values.pt` | items 4 and 6–10: the artifact, alignment, my own responses and outcome, then the cross-checks |
| `r3_geometry.py`, `.json`, `.log`, `vectors32_own.pt` | items 5 and 11: d, t̂, θ, the controls and all 840 vectors rebuilt; the ± mapping; the tag map |
| `r4_gates.py`, `r4_worker{0,1}.{pt,json,log}` | item 5: C, Level-1, I1, I3 and I4 on every run; the ladder; the wrong-vector I1 control |
| `r5_descriptives.py`, `.json`, `.log` | items 11 and 12: the descriptives and the heterogeneity |
| `r6_aggregate.py`, `.json`, `.log` | item 5 and the ladder, aggregated |
| `r7_today.py`, `.json`, `.log` | items 2, 14 and 15 re-checked today: the inputs, the freeze, `validate_lock`, the spent set, the checkpoint, the semantics, the stopping rule |
| `r8_extra.py`, `.json`, `.log` | the smoke keys, a naive frame test (not diagnostic), the in-memory report rendering |
| `r9_refrows.py`, `r9b_refrows_detail.py`, `.json` | the frame binding of the reference rows (weights only) |
| `r10_tokens.py`, `.json` | the re-tokenized cue and plural ids |
| `table_percue.md` | the per-cue table |
| `final_check.sh`, `final_check.log` | the end-of-review check (2026-09-27 09:05 UTC): `git status` clean at `6f72d7d`, the 4 output files byte- and time-identical to the production run, no `report.md`, the run records untouched |
