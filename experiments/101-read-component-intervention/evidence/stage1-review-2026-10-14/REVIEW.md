# Experiment 101 — review of production stage 1 (2026-10-14)

**AI review (independent, read-only), not human validation.** A separate Claude agent (Opus 5.5) wrote this review from
the brief `PROMPT.md` in this directory. Every claim below is backed by a script in `scripts/` and its output in
`out/` (digests in `SHA256SUMS`).

How the review stayed inside the brief's rules:
- **No repository writes.** It wrote nothing in the repository. A snapshot taken before and after is identical apart
  from its timestamp line (`out/repo_snapshot_start.txt`, `…_end.txt`, `…_final.txt`). The snapshot covers `git
  status`, the index, and the sha256, size and mtime of every file under `outputs/experiment-101/`,
  `experiments/101-read-component-intervention/` and `src/`.
- **No prompt and no forward.** It ran no 101 prompt and no forward pass of any kind. It never ran the runner's
  `stage1`, `confirm` or `report`; it ran `validate` once, under the guard.
- **The guard.** Every model-loading script installed the token-sequence guard: 480 sequences, each of the 60
  candidate texts with each of the four cues, plus the 36 frozen frames. The same scripts also refused every
  `torch.nn.Module.__call__`. Neither ever fired (§ How).

## Verdict: PASS WITH NOTES

The user can commit the three stage-1 files as they are.
- **Blockers: none. Should-fix: none.**
- **The run.** Stage 1 ran once, as specified.
- **The files.** They are exactly what the stage-1 record names.
- **The barrier passes on these bytes.** I ran the runner's own `_barrier` read-only, and it passes.
- **The table reproduces.** It recomputes bit for bit and byte for byte in my process, for every thread count tried.
- **It matches my own reading of the design.** An independent recomputation from my reading of computation 6 and design
  §4–§5, with the readout rebuilt from the raw checkpoint, agrees to ≤ 1.6e-13.

The notes are operational: what to know before committing and before `confirm`.

| object | digest |
|---|---|
| `stage1/states.f64` | sha256 `1c7e51c049e1c61cf669f1783736054956073dc5af6947354b1d13c058ac5b66` (5,240,064 bytes) |
| `stage1/states.json` | sha256 `2b2c3a61b4a2ce05eba0dcf875c19bfdb1c2627da1ccb1e8d49710af95a873ca` (34,492 bytes) |
| `stage1/table.json` | sha256 `71d339cd1d7151ad57949384388679593264ac07dec8c9c944c121d9918795fa` (1,946,190 bytes); content `9191800440af37909d2068459e360d7772191fc66bf2ad5fb9cf2c133246ac86` |
| stage-1 record (`state["stage1"]`) | digest `d2f1ed03920a7dd66924918b21ea36e754b4981b3517a8621b13edc3286cfc3f` (reproduces) |
| `outputs/experiment-101/results.json` | file `6372d113d4ed71af0e59916b509c80c7ebdff4abe59848e957e0cc86bd9f3e2a`; state `aecd76fc3984e16041ec4f6b0d6b656f07707e28ea4a3665f7bfd8c0fc2a0513` |
| installed lock | content `8da97e35556d0baa4739899cb841e86cbd59cced21611c732fdd9d8c9edd9061` (file `8dc6e6d3…`), committed at `961e814` |
| launcher v2 | `13b1992f7f0feb652207ca835df044244f20faea750e92f42d8d51bced7c3e1c` (= the committed `evidence/lock-2026-10-14/prepared/` copy, git blob equal) |

## Findings (most severe first)

### Blockers: none.

### Should-fix: none.

### Notes (what to know first)

**N1. Commit exactly these bytes, alone, and touch nothing else in `outputs/experiment-101/`.**
- **What the barrier rejects.** It rejects any difference as a phase incident, and `confirm` is then refused for good.
  The checks are a file's sha256, the index, a state digest, the table digest or a single bit of the recomputed
  table (`run.py:737–782`; a barrier `IncidentError` is recorded at `run.py:804–809`).
- **What that means in practice.** An editor, a JSON formatter or a re-save that changes a byte consumes the
  confirmation once the changed file is committed and `confirm` reaches the barrier. Uncommitted, the change only
  leaves a dirty tree, which is a retryable refusal.
- **Nothing here alters the bytes on commit.** There is no `.gitattributes` (repository, `info/` or global), no
  `core.autocrlf`, `eol` or `filter`, no active hook in `.git/hooks`, no `.pre-commit-config.yaml`, and the paths are
  not ignored.
- **The only copy of the stage-1 record** (state digests, file digests, record digest) is in the gitignored,
  uncommitted `outputs/experiment-101/results.json`. If it is edited, the state digest fails. If it is deleted, there
  is nothing to compare with. Do not open it in an editor that saves.
- **Do not rewrite history** between `3621eec` (the lock's `protocol_code_commit`) and confirm: no rebase and no
  squash. `validate_lock` needs `3621eec` to be an ancestor of `HEAD`.
- **A safe sequence** (the reviewer ran none of it):
  ```
  git --no-optional-locks status --porcelain --untracked-files=all   # exactly the three ?? lines
  shasum -a 256 experiments/101-read-component-intervention/stage1/*  # the three digests above
  git add -- experiments/101-read-component-intervention/stage1/states.f64 \
             experiments/101-read-component-intervention/stage1/states.json \
             experiments/101-read-component-intervention/stage1/table.json
  git commit                                                          # these three files alone
  git --no-optional-locks status --porcelain --untracked-files=all   # empty
  for f in states.f64 states.json table.json; do git cat-file blob HEAD:experiments/101-read-component-intervention/stage1/$f | shasum -a 256; done
  ```

**N2. Keep the machine's environment fixed until `confirm`, including the operating system.**
- **What must stay fixed.** The barrier recomputes the table bit for bit (`run.py:758–762`), and stage 2's G6 needs the
  REF runs to reproduce the stage-1 states bit for bit. Torch's CPU kernels here come from Apple Accelerate, a system
  library (`torch.__config__`: `BLAS_INFO=accelerate`, `LAPACK_INFO=accelerate`).
- **The gap in the runner's check.** `_check_runtime` (`run.py:429–437`, `runtime_record` `:97–100`) refuses
  package-version, device, dtype and thread changes before anything is spent, but not an OS update. A macOS update
  could therefore change bits and turn into a barrier incident, or a G6 integrity failure after the S2 ledger.
- **Evidence for the current environment** (macOS 15.7.3, build 24G419): the table recompute is bitwise identical
  across processes (stage 1's and mine) and across 1, 4 and 8 threads (`out/d_table.json`). Calibration's re-captures
  matched 020's three-week-old states with drift exactly 0.0.
- **Recommendation.** The confirm launcher's preflight should refuse on any platform difference, as the stage-1
  launcher did against the lock run's record (`launch_stage1.py:423–428`). The stage-1 record names
  `macOS-15.7.3-arm64-arm-64bit`.

**N3. A latent phase-rule gap for `confirm` (the launcher re-check's N1, still open; it does not affect today's
state).**
- **The gap.** `ri.assert_phase_allowed("confirm")` reads `state["stage1"].get("incident")`
  (`read_intervention.py:2077–2079`), a key nothing writes. Stage-1 incidents are written to
  `state["phases"]["stage1"]["incidents"]` (`run.py:394–402`), which the rule never reads.
- **Shown on a copy.** On an in-memory copy with a synthetic stage-1 incident added, `confirm` is still allowed
  (`out/b_state.json`, `N1_confirm_rule_ignores_stage1_phase_incidents: true`).
- **Today.** No incident exists anywhere (§2), so `confirm` is correctly allowed now.
- **Recommendation.** The confirm launcher's preflight should refuse unless `phases.stage1` is `complete` with no
  `incidents`.

**N4. The barrier's reference point is uncommitted; anchor it outside the state.**
- **What the barrier compares against.** The stage-1 record in the gitignored state. The record and the files are
  consistent today.
- **Anchors to add.** After the commit, git holds the files. Commit the launcher's records into
  `evidence/stage1-2026-10-14/` byte-identically, as its docstring says; `production_run_*.json` names the three file
  sha256s and the table content `9191800440af…`.
- **The confirm launcher's authorization** should name the table's content sha256, as the plan requires ("and, for
  `confirm`, the stage-1 table's"). It could also pin the three file digests.

**N5. Stage 1 never read its own files back; this review did, and the barrier passes.**
- **What stage 1 recorded.** It computed the state digests from the in-memory states (`run.py:715`). The launcher's
  post-checks compared file sha256s and the record digest only.
- **What this review ran.** The first in-protocol read-back would have been the barrier. I ran the runner's own
  `_barrier`, `_committed_stage1` and `_recompute_table` read-only on these exact bytes, and they pass (§4, §6).
- **Optional.** The confirm launcher's rehearsal could run the same read-only checks, so that an operational slip shows
  before the one-shot run. How such a finding is classified stays the user's decision under the plan.

**N6. The path rule for the next commits, and a stale README.**
- **Which paths are safe.** `validate_lock` refuses `confirm` if any scientific path changed since `3621eec`
  (`read_intervention.py:1972–1981`, `:2016–2020`).
  - Non-scientific, shown with `ri.scientific_changes` (`out/d_table.json`): the three stage-1 paths, anything under
    `experiments/101-read-component-intervention/evidence/`, the experiment's `README.md`, and `outputs/`.
  - Scientific, so a later refusal (retryable, but avoidable): a new file elsewhere in the experiment directory, such as
    `launch_confirm.py` at its root or a note inside `stage1/`, and any file under `src/`.
- **Simulated post-commit check.** With the three paths added to today's changed paths and a clean, tracked tree,
  `ri.validate_lock` passes.
- **Stale README.** `experiments/101-read-component-intervention/README.md:3` and `:10` still say "LOCK CANDIDATE …
  No stage 1 or confirmation has run".
- **A look-alike, harmless.** The lock review's committed probe files under
  `evidence/lock-review-2026-10-14/out/audit_event_probe/experiment/stage1/` carry the stage-1 file names. They are not
  at the bound paths and are non-scientific.

**N7. Scratchpad artifacts a later audit should expect: not results, and never to be copied into the repository
unmarked.**
- **Fabricated labels.** A concurrent confirm-launcher review runs a `StubRunner` in
  `…/exp101/confirm-launcher-review/main_block/*/repo/`. Six stub `results.json` there carry the fabricated label
  `READ_COMPONENT_EFFECTS_AS_PREDICTED`.
- **Zero tensors under real key names.** Three stub `stage2-partial.pt` hold `torch.zeros(4)` under real 101 key names
  (`…-101-0{1,2,3}|ref|581|REF`).
- **They are stubs.** I verified both against the stub runner's source (`e_main_block_scenario.py`: "labels only;
  nothing runs") and by loading the tensors (`out/g_leakage.json`).
- **FIFOs.** The launcher re-check left FIFOs (`stage1-launcher-recheck/mirror_{production,rehearsal}/repo/fifo`) that
  block a naive file sweep. They blocked my first sweep, which I stopped and rewrote to skip non-regular files.

**N8. Record gaps the review closed by other means (cosmetic).**
- **State writes are not counted.** The launcher declares `output_writes` and `output_renames` but never fills them
  (`launch_stage1.py:63`, `_hook` `:155–170`).
- **How I counted them instead.** `outputs_reads` lists every allowed open under `outputs/`, writes included. It shows
  exactly two state temporaries, `.results-09rbmedr.json` and `.results-umf8udkn.json`: the ledger write and the
  completion write, so no incident write.
- **An independent sweep.** An mtime/ctime sweep of the whole repository (`.git`, `.venv` and ignored paths included)
  over 13:08:30–13:09:40Z lists only:
  - `experiments/101-read-component-intervention/` (13:09:08.77Z, `stage1/` created);
  - `stage1/` and its three files (13:09:26.18–.22Z);
  - `outputs/experiment-101/` and `results.json` (13:09:26.45Z).

  `.git/index` is unchanged since 12:06:03Z, the lock commit.

## The brief's items, one by one

### 1. The launcher's evidence: confirmed (`out/a_launcher.json`, 45 checks, all pass)

- **The launcher.**
  - Its sha256 `13b1992f…` equals `launcher_sha256_at_production.txt` (written 13:08:38Z), the sentinel's content
    (13:08:56Z) and the committed `prepared/` copy (same git blob at HEAD).
  - The file's mtime (10:12Z) predates the production check.
- **One invocation, exit 0.**
  - There is exactly one `production_run_*.json` (and one rehearsal record), with `stage1_invocations: 1`.
  - The record has `exit_status: 0` and `error: null`; `production.exit` says `exit 0`.
  - The head is `961e81408e5c4da4fffdf9dfab0343e449a4e1f2` on `exp101-phases`, with `HF_HUB_OFFLINE=1` and
    `PYTHONDONTWRITEBYTECODE=1`.
  - The preflight passed 17/17 checks: the rehearsal's 16 plus the production token. The hook self-test passed.
- **36 sanctioned runs.** The run log has 36 lines, `n` = 1…36, and the keys are the 36 `frame|ref|581|S1` keys of
  the frozen manifest, in frozen order.
  - The wrapper admits only the `S1` tag, only a key in the set, each once, and only after the ledger check
    (`launch_stage1.py:299–314`).
  - `ri.run_arm` refuses outright and was never called (`refused: []`).
- **No other forward.**
  - `torch.nn.Module.__call__` refuses outside a sanctioned capture, and nothing was refused.
  - 12,240 module calls = 36 × 340. The per-run 340 is constant across 2,108 sanctioned runs in three processes and
    two entry points: calibrate's production 685,440 / 2,016, its smoke 19,040 / 56, and stage 1.
  - By code reading, each capture is exactly one forward: `ri.capture_reference` → `rd.capture_frame_020` →
    `ra.capture_reference` → `pm.capture_prompt` → `capture.run_capture` → one `model.run_with_hooks` → one
    `self.forward(...)` (`transformer_lens/model_bridge/bridge.py:3297`).
  - I did not measure 340 with a forward of my own; the brief allows weights-only work.
- **One model load.** `load_model: 1` (the launcher would refuse a second; none was attempted, `refused: []`), and one
  "Loading weights" bar in the log.
- **The ledger on disk before the first prompt.**
  - The first capture was admitted only once `results.json` held exactly the 36 keys with `stage1` `running`
    (`_ledger_on_disk`, `launch_stage1.py:289–296`, `:303`), and nothing was refused.
  - Timeline (UTC):

    | time | event |
    |---|---|
    | 13:08:48.32 | the record starts |
    | 13:08:56 | the sentinel |
    | 13:09:08 | `stage1` running: the ledger |
    | 13:09:11.98 | the run log's last write |
    | 13:09:26.18–.22 | the three files |
    | 13:09:26 | `stage1` complete |
    | 13:09:26.68 | the record ends |
    | 13:09:27 | the exit file |
- **Repository writes.**
  - `forbidden_writes: []` under a hook that refuses every other repository write.
  - `stage1_writes` names exactly the three files.
  - `stage1_mkdirs: 2` counts two audit events of `mkdir(parents=True, exist_ok=True)` (`run.py:685` and
    `upstream_localization.py:464`) but only one creation.
  - The state was written twice, and nothing else changed (N8's sweep).
- **Post-run checks.** All five are true: all 36 ran once, the ledger is exactly the 36 S1 keys, the tree is exactly
  the three untracked files, the record digest reproduces, and the files equal the record. `tree_after` and `git
  status` now both show exactly the three `??` lines.

### 2. The state: confirmed (`out/b_state.json`, 39 checks, all pass)

- **The state verifies.** `rd.load_results_state` verifies the state digest (`aecd76fc…`). The experiment, the
  configuration (`ri.PRODUCTION`), the inputs (equal to the base digests) and the versions (equal to now) all match.
- **Phases.** `calibrate`, `lock` and `stage1` are `complete`. `confirm` and `report` are exactly `{"status":
  "not_started"}`.
  - `phases.stage1` has `awaiting_commit: true`, commit `961e814…`, runtime equal to 020's explore (4 threads,
    `deterministic_algorithms` false), `geometry` `{"bitwise_equal": true, "differing": []}`, started 13:09:08Z and
    completed 13:09:26Z.
- **No incident.** None anywhere: no `phases.*.incidents`, no `calibration.incidents`, no `stage1.incident`.
  `confirmation` and `report` are null.
- **The ledger.** `executed_prompt_keys` is exactly the 36 S1 keys, sorted, with no S2 key and no target-prompt key
  (`two`, `three`, `many`). `calibration_executed` holds 2,016 exposed keys and no 101 frame.
- **The stage-1 record.** Its keys are `frames`, `state_digests`, `files`, `table_content_sha256`, `lock_sha256`,
  `commit`, `noun_keys`, `executed` and `digest`.
  - The digest `d2f1ed03…` reproduces with `b0c.stage_one_digest`.
  - The frames are the frozen 36, with `p_t = p_c + 1` and `p_c` 4–6: 10, 22 and 4 frames.
  - There are 36 state digests.
  - The files, paths and sha256s equal the brief's and the lock's `stage1.paths`.
  - `table_content_sha256` is `9191800440af…`.
  - `lock_sha256` equals the installed lock's content digest `8da97e35…`, which itself verifies.
  - The commit is `961e814…`, and the noun keys equal the lock's (79). `executed` is 36.
- **The phase rules on this state.** `confirm` is allowed; `stage1`, `lock` and `calibrate` are refused. The state's
  `lock` block equals the installed lock and preregistration.

### 3. The files: confirmed (`out/c_files.json`, 28 checks; `out/h_cref.json`)

- **Digests.** Each file's sha256 equals the record's (and the brief's), at the record's path.
- **`states.json`.**
  - It is canonical JSON plus a newline and passes `ul.verify_table_index` against the lock's `stage1.states.layout`
    (324 blocks) and `meta`, with no unbound key.
  - Its `file_sha256` names `states.f64`. Its offsets equal my own contiguous layout from the lock (total 5,240,064
    bytes).
- **`states.f64`.**
  - `ul.read_table` reads it back, equal to my independent numpy parse, and every value is finite.
  - `ri.states_from_blocks` then gives 36 states whose `rd.state_digest` all equal the record's. So do the entries I
    rebuilt without `ri`.
- **Positions.**
  - `states_from_blocks` takes `p_c`/`p_t` from the frozen frames, so that check alone would be circular.
  - **The file carries the positions in two places**, both verified against the lock, which was derived from the frozen
    frames:
    - the index's `meta.frames` list (`frame_id`, `p_c`, `p_t`), checked by `verify_table_index`;
    - every block's shape: `p_t + 1` rows of 512, and attention rows `[8, p_t+1]` that sum to 1 within 1.5e-7.
  - **The record's digests** were computed at capture time from `rd.locked_state(captured)`, which embeds the capture's
    `(p_c, p_t)`. They reproduce with the frozen values.
  - **At capture time** the runner asserted that the capture's positions equal the frozen frame's (`run.py:697–698`).
    `rd.capture_frame_020` also refuses a run that does not cover exactly positions `0..p_t`.
- **Field alignment.** Each frame's stored `c_ref` equals the noun read of `LN_final(h6)` from the same stored state
  (raw checkpoint weights).
  - The agreement is within 1.4e-3 to 2.9e-3: float32 logit precision, against calibration's recorded logit identity
    maximum of 3.03e-3 and the frozen tolerance of 2e-2. A misalignment would give errors of order 1.
  - The reference contrasts are positive in all 36 frames: the mean over the 79 nouns runs from +2.909 to +5.334 nats,
    median +4.256. That is consistent with the `one` prompts.

### 4. The table, recomputed: bit for bit (`out/d_table.json`, 20 checks; `out/e_independent.json`, 9 checks)

**The recomputation.** In my own process, with every forward refused:
1. The model was loaded for its weights; its parameter and embedding digests equal the lock's.
2. `ul.ModelPrograms.from_model`, then G1 and the geometry digests, bit for bit against the lock.
3. The stage-1 binding, re-derived, equals the lock's.
4. The states were rebuilt from the files (`ul.read_table` → `ri.states_from_blocks` → `rd.state_from_locked`), and
   their digests equal the record's.
5. `ri.prediction_table(..., fractions_for_weak=False)` gave `content_sha256` `9191800440af…`.
   - Its canonical JSON equals `table.json` (0 differing leaves).
   - `canonical_json(table) + "\n"` equals the file **byte for byte**.
   - A second run gave the same result, and so did 1, 4 and 8 torch threads.

**The barrier.** The runner's own `_barrier` passes on the current files, called exactly as `confirm` calls it,
read-only, after `_geometry_against_lock`.

**The metadata.** All 11 of the lock's `stage1.table` keys equal the table's: `experiment`, `schema_version`, `kind`,
`construction`, `mask` 30, `fractions_for_weak` false, `weak_shift` −1.0, the 79 `noun_keys` in order, the 11 `arms`,
`cue_order` (two, three, many) and `reference_id` 581. The table has no other top-level key besides `frames` and
`content_sha256`.

**Independently of `ri`.** For all 36 frames × 3 cues, I recomputed from my own reading of computation 6 and design
§4–§5:
- Δx̂3 comes from the frozen program itself (`ul.reference_rows_017`, `ul.pair_context`, `ul.compose_dx3` at mask 30).
- Mine:
  - `w_c` from 011's `read_weight`; ‖w_c‖ is 0.7659205190757624, which equals the lock's;
  - Δr̂, the read part, `off_c`, `û_off` and ŝ;
  - all 11 arms' inputs at both positions per §5;
  - the fractions per §4, with N̂, Ŝ, A_amp, A_neg and the weak flag.
- Δĉ was computed twice:
  - with `ul.contrast_of` on my inputs;
  - with **my own Level-0 readout from `model.safetensors`**, using no TransformerBridge and no `rd`, `ul`, `atp` or
    `lc` code. Block 3 is exact (rotary, causal softmax) with its MLP; block 4 is its MLP only, with frozen attention;
    block 5 is the frozen reference rows times the value changes, plus its MLP; then `LN_final` and each noun's
    unembedding difference.

| worst difference against the table | value |
|---|---|
| pair quantities (Δr̂, ŝ, ‖off_c‖, ŝ/‖off_c‖, ‖Δx̂‖), relative | 5.2e-16 |
| Δĉ per noun, `ul.contrast_of` on my inputs (1,188 arm × cue × frame; 996 bitwise equal) | 1.6e-13 |
| Δĉ per noun, my raw-checkpoint readout | 1.6e-13 (against `ul.contrast_of` on the same inputs: 9.1e-15) |
| arm means | 6.0e-14 |
| fractions (all 13 keys) | 1.2e-14 |
| weak and overshoot flags | 0 mismatches |

| frame (two) | Δr̂ table = mine | ŝ table = mine | Δĉ_full table | Δĉ_full, my raw readout |
|---|---|---|---|---|
| 01 | 2.038164294182915 | 2.661065010560588 | −4.453686089527146 | −4.453686089527147 |
| 09 | 1.909537091434535 | 2.493126954920566 | −4.534157559824145 | −4.534157559824146 |
| 18 | 1.662447160987447 | 2.170521770318315 | −5.497811862407459 | −5.497811862407459 |
| 27 | 1.996730213070631 | 2.606967907688501 | −4.753751165263767 | −4.753751165263768 |
| 36 | 1.647731482532830 | 2.151308708273217 | −4.633142326100747 | −4.633142326100745 |

Over all 36 frames, my raw readout's Δĉ_full differs from the table by at most 1.8e-15. **Not re-implemented:** the
upstream program Δx̂3 (023's block-0 terms and the 017 chain), which is taken from the pinned `ul.compose_dx3`.

### 5. The predictions, described (`out/f_predictions.json`)

No target result exists to compare with. For `two`, n = 36; the medians follow the design's even-n rule.

| quantity | min | p10 | median | p90 | max | count |
|---|---|---|---|---|---|---|
| Δĉ_full | −6.643 | −6.544 | −4.999 | −4.272 | −3.986 | ≤ −1: 36/36 |
| F̂_rem | −0.171 | −0.128 | −0.010 | +0.070 | +0.093 | |
| F̂_rem_off | +0.981 | +0.987 | +1.001 | +1.018 | +1.025 | |
| F̂_ins | +0.801 | +0.945 | +0.993 | +1.040 | +1.052 | |
| F̂_ins_off | −0.095 | −0.070 | −0.031 | −0.007 | +0.007 | |
| N̂ | +0.908 | +0.926 | +1.016 | +1.130 | +1.181 | > 0: 36/36 |
| Ŝ | +0.807 | +0.963 | +1.034 | +1.087 | +1.139 | > 0: 36/36 |
| A_amp | −0.579 | −0.477 | −0.103 | +0.051 | +0.110 | < 0: 30/36 |
| A_neg | +0.153 | +0.402 | +0.944 | +1.705 | +2.480 | > 0: 36/36 |

- **Pair quantities.** Δr̂ runs from 1.648 to 2.323 (median 1.953), ŝ from 2.151 to 3.032, and ŝ/‖off_c‖ from 0.306
  to 0.484, so no frame overshoots.
- **Predicted-weak frames.** None, for `two`, `three` or `many`. Every frame's fractions are present.
- **Non-finite values.** None.
- **The stratum.** Same pattern. `three`: N̂ > 0 and Ŝ > 0 in 36/36, A_amp < 0 in 27/36, Δĉ_full median −5.011.
  `many`: 36/36, 36/36 and 24/36, Δĉ_full median −5.136, with F̂_rem median 0.214, the largest of the three cues.
- **What the program predicts.** The read part carries essentially the whole shift (F̂_rem ≈ 0, F̂_ins ≈ 1), with A_amp
  negative in 30 of 36 frames for `two`.
- **Context, not a comparison.** The exposed calibration frames' measured values were similar: Δc_full(two) median
  −4.94, Δr median 1.98, measured N and S medians 1.03 and 1.03, A_amp −0.145, A_neg 1.167.

### 6. Leakage and the barrier (`out/g_leakage.json`, 8 checks; `out/d_table.json`)

- **No S2 prompt ran.**
  - The ledger, the run log and the launcher's admission rule hold only S1 keys.
  - No module call happened outside the 36 sanctioned captures.
  - The ledger shares no key with the 1,980 S2 keys or the 108 target prompts.
- **No target result exists anywhere.**
  - `outputs/experiment-101/` holds only `calibration-measurements.pt` (2,556 tensors, no 101 frame),
    `candidate-calibration-v1.json`, `candidate-lock.json`, `candidate-preregistration.md` and `results.json`
    (`confirmation` null). There is no `stage2-*` and no `report.md`.
  - A byte scan of every regular file in the repository (`.git`, `.venv` and `.worktrees` excluded) and in the exp101
    scratchpad found 101 S2 or target-prompt key strings only in:
    - the S2 manifest (`confirmation-v1.json`);
    - the freeze review's key derivations;
    - tier-A key-format assertions (`tests/test_read_intervention.py:136–139`);
    - a concurrent review's git clone;
    - that review's stubs (N7).
  - The two files over 200 MB (022's calibration table, 025's stage-2 measurements, both older than 101's frames) were
    chunk-scanned and contain none.
  - Git history has no `stage2` file and no `outputs/experiment-101/` file.
- **The barrier after the commit.** It will pass if the committed bytes are these. I ran it read-only on them, with
  the runner's own methods:
  - the record digest and the lock binding;
  - each file's sha256 and path;
  - the index against the lock;
  - `read_table`'s own digest check;
  - the 36 state digests;
  - the table's digest and construction;
  - the bit-for-bit recompute;
  - no S2 key in the ledger.

  The steps of `confirm` before it pass or are expected to pass after the commit:
  - `validate_lock`: simulated clean and tracked tree, three non-scientific paths added;
  - the runtime check and `_geometry_against_lock`;
  - the stage-1 binding and the model digests;
  - the phase rule.

  Before the commit, `_validated_lock` correctly refuses: the tree holds untracked files.
- **What would make it fail.**
  - Any byte change to the three files, or an edit of `results.json` (N1).
  - A numerics change such as an OS or Accelerate update (N2).
  - Library or source changes are refused earlier, retryably, by the version, blob and scientific-path checks.
- **What would make it pass wrongly.** The only way is to replace the files and the state's stage-1 record together,
  consistently, which is deliberate tampering. The external anchors are git, the evidence records and the table digest
  in the confirm token (N4).
  - The barrier checks integrity since stage 1, not capture correctness. Capture correctness is G6's job at stage 2,
    and N2 applies there too.

### 7. Anything that should stop the commit

- **Nothing.** No blocker.
- **Before committing:** N1.
- **Before `confirm`:**
  - N2: freeze the OS and the environment;
  - N3: the confirm launcher must check `phases.stage1.incidents`;
  - N4: anchor the table digest;
  - N6: path hygiene, and update the README in the evidence commit.

## How (the rules, kept)

- **The guard** (`scripts/guard.py`, imported first by every script).
  - An audit hook refuses every write-type event that resolves inside the repository, in any spelling: case,
    `/System/Volumes/Data`, relative names, `dir_fd`, `O_TRUNC`. It also refuses every child process except read-only
    `git`.
  - `GIT_OPTIONAL_LOCKS=0` is set, `TMPDIR` points into this directory, and `-B` is required.
  - A decision-level self-test runs first in every script: 15 refusals and 5 passes on synthetic tuples, with no real
    I/O against the repository.
  - The two model-loading scripts (`d_table`, `e_independent`) also:
    - installed the token-sequence guard (480 sequences);
    - refused every `torch.nn.Module.__call__`;
    - replaced 12 forward entry points by identity;
    - sealed all 430 module instances and 20 class methods.
  - `h_cref` reads raw checkpoint tensors from `model.safetensors` only. No model object exists there, so no forward is
    possible, and it installs only the audit hook.
- **What fired.** Across all ten records: 0 repository-write refusals, 0 child refusals, 0 module-call refusals, 0
  entry-point refusals and **0 token-guard hits**. The only writes outside the repository were `/dev/null` and this
  directory.
- **Model use.** Weights only: `models.load_model` (no forward), `model.safetensors` read directly, and the frozen
  programs.
- **Runner use.** The runner's `validate` ran once under the guard (exit 0). `stage1`, `confirm` and `report` were never
  run.
- **Limits.**
  - I did not measure the per-forward module-call count (no forward).
  - I did not re-derive the stored states from the model (no prompt). The c_ref/h6 identity and calibration's
    validated capture path stand in, and stage 2's G6 is the protocol's check.
  - I did not re-implement the upstream Δx̂3.
  - A concurrent confirm-launcher review was active in the scratchpad; it did not touch the repository (the snapshots
    are identical).

## Files (this directory; full digests in `SHA256SUMS`, sha256 `454497c47455d9a5b98f7a922dbef5936a0b1ba7632022e9c00c19cc914b4089`)

| file | what |
|---|---|
| `PROMPT.md` | the brief |
| `scripts/guard.py` | the guard: audit hook, self-test, token-sequence guard, forward refusals, sealing |
| `scripts/snapshot.sh` → `out/repo_snapshot_{start,end,final}.txt` | the repository before, after the checks and at hand-back (identical but for the timestamp) |
| `scripts/a_launcher.py` → `out/a_launcher.{json,log}` | item 1 |
| `scripts/b_state.py` → `out/b_state.{json,log}` | item 2, and N3's demonstration |
| `scripts/c_files.py` → `out/c_files.{json,log}` | item 3 |
| `scripts/h_cref.py` → `out/h_cref.{json,log}` | item 3: c_ref against LN_final(h6), raw weights |
| `scripts/d_table.py` → `out/d_table.{json,log}` | item 4: the bit-for-bit recompute, the barrier, the confirm pre-checks, thread sensitivity |
| `scripts/e_independent.py` → `out/e_independent.{json,log}` | item 4: the independent recomputation and raw-checkpoint readout |
| `scripts/f_predictions.py` → `out/f_predictions.{json,log}` | item 5 |
| `scripts/g_leakage.py` → `out/g_leakage.{json,log}` | item 6 |
| `scripts/i_validate.py` → `out/i_validate.{json,log}` | the runner's `validate`, guarded |
| `out/*_guard.json` | each script's guard record |
| `out/stage1_record_extract.json` | a read-only extract of the reviewed state (phases, ledger, stage-1 record) |
