# Experiment 101 — confirm launcher version 2: diff check against the review (2026-10-14)

**AI review (independent, read-only), not human validation.** The same Claude agent that wrote `REVIEW.md` did this check,
under the same rules: read-only in the repository; the launcher never run against the repository in any mode; no runner
`confirm`, `stage1` or `report`; no 101 prompt. Isolated tests ran in scratch only: an `ast` extraction, the main block
with stub runners, and the scratch clone. Every forward route was refused in the clone processes.

## Verdict: READY FOR PRODUCTION

- **The versions.**
  - Version 2 is `launch_confirm.py`, sha256 `56a687f372605e5d6e16e2cc62bc9fd1197f21fdd8dccf0dda6d6ac75723309f`.
  - Version 1 as reviewed is `launch_confirm_v1_as_reviewed.py`, sha256 `4f132e8e…b209`, byte-identical to my
    `reviewed/` copy.
- **S1, N1, N3, N7, N8 and N9 are resolved** as the review intended. The details are below.
- **No new defect found.** Specifically:
  - `barrier_dry_run` cannot write state or run a forward (by construction, and demonstrated in the clone, including
    two negative controls);
  - the signal handling is right in every phase (18 executed scenarios);
  - `post_ok` cannot fail a correct run.
- **The remaining notes need no code change.** R1 is an optional one-line hardening (refuse a pipe as well as a
  terminal). R2 to R4 carry over or are cosmetic.
- **The production conditions (operational):**
  1. The user commits the three stage-1 files.
  2. Run rehearsal → smoke → production with this sha256 (`56a687f3…`), each checked immediately before.
  3. Use the content tokens `CONFIRM101_LOCK=8da97e35…9061` and `CONFIRM101_TABLE=91918004…ac86`.
  4. Start it detached with `nohup`, output redirected to a regular file (`> production.log 2>&1`), with
     `GIT_OPTIONAL_LOCKS=0`.
  5. Do not interrupt it, and do not change the working tree until it ends.

## What changed (`out/v2_diff.txt`; `ast` comparison `out/v2_a2_ast_diff.log`)

- **Identical to version 1 (`ast`):**
  - every hook function: `_path`, `_inside`, `_in_repo`, `_temp_state`, `write_allowed`, `decide`, `_hook`,
    `hook_self_test`;
  - every guard: `_deny`, `_guarded_call`, `_low_level`, `_ledger_on_disk`, `_vector_digest`, `_sanctioned_run_arm`,
    `_refuse_capture_reference`, `_sweep_aliases`, `_seal_model`, `_counted_load`;
  - `preflight`, `_signal_to_interrupt`, `_git` and `_sha256_file`.

  So every hook and guard result of `REVIEW.md` carries over. The self-test re-run from version 2's own source passes
  (9/9 allowed, 12/12 refused, rehearsal refuses the measurements; `out/v2_self_test.log`). The operator's
  `hook_self_test.out.json` is byte-identical.
- **Changed:**
  - `smoke` (N9); `barrier_dry_run` added (N7);
  - at module level: `SMOKE_REFERENCE_SHA256`, `signal.signal(SIGHUP, SIG_IGN)` at `:83`, the main block (the
    rehearsal's dry run; production's terminal check, sentinel mode 0644, and SIGTERM converted around `confirm()`
    only), `os.fsync` of the record, and `post_ok`;
  - the docstring.

## Item by item

| review item | version 2 | evidence | status |
|---|---|---|---|
| **S1** SIGHUP cancelled `nohup` | `:83` `signal.signal(SIGHUP, SIG_IGN)` before the hook, for the whole run; production refuses when `os.isatty(1) or os.isatty(2)` (`:590–591`), before the sentinel | `sighup_under_nohup` and `sighup_without_nohup` (SIGHUP sent after the ledger): ignored, complete, exit 0; `tty_stdout`, `tty_stderr_only` (under `script`): refused, no sentinel; clone production under a pty: refused, no sentinel | **resolved** |
| **N1** a signal after `confirm()` could lose the record | SIGTERM → `KeyboardInterrupt` only inside `try: confirm() finally: SIG_IGN` (`:597–601`) | `sigterm_during_tree_after` and `sigterm_in_post_block`: ignored (`signals: []`), record written, exit 0 | **resolved** |
| **N3** the exit code ignored the post checks | `post_ok` (`:628–631`) requires phase `complete`, `all_1980_ran_once`, and no refused or forbidden event, in production | `keys_not_all_run`, `refusal_swallowed`, `forbidden_write_recorded`: exit 1; `correct`: exit 0 | **resolved** |
| **N7** no barrier dry run | `barrier_dry_run` (`:508–527`) in rehearsal: counted and sealed load, `_model_dependencies`, `_check_nouns`, `_check_stage1_binding`, `_geometry_against_lock`, `_barrier` | clone: ok, barrier passed (stage-1 digest `d2f1ed03…`, table `91918004…` recomputed bit for bit), 0 forwards, state unchanged | **resolved** |
| **N8** cosmetics | sentinel `0o644` (`:592`); record `os.fsync` (`:622`); docstring N2 corrected | sentinel mode `0o644` in every scenario | **resolved** (`record["logs"]` on an exception still skipped: R3) |
| **N9** the smoke reference unchecked | the reference's sha256 against `ba4698a5…` (= the committed `SHA256SUMS` line 16) and its 55 run keys against the frame's (`:543–547`) | evaluated without running anything: both checks pass on the real reference and schedule (the order is equal too; `out/v2_i2_smoke_checks.log`) | **resolved** |

## The three questions

### 1. `barrier_dry_run` cannot write state or run a forward

**By construction:**
- **No call writes.** `_installed_lock` reads. `models.load_model` reads the HF cache; its only writes are the
  `TMPDIR` mkdir. `ModelPrograms.from_model`, `_model_dependencies`, `_check_nouns`, `read_geometry` and
  `_check_stage1_binding` are weights only. `_geometry_against_lock` and `_barrier` read the state and the stage-1
  files; their computations run under the runner's `pytest_free_guard`.
- **A failure records nothing.** It raises `IncidentError` or `PhaseError`. The phase-incident write
  (`_record_phase_incident`) lives in `confirm()` itself (`run.py:807–809`), which the dry run never calls.
- **Every repository write would be refused anyway.** The dry run runs only in rehearsal, whose rules refuse them all.
- **No forward can be admitted.** In rehearsal `allowed_keys` is empty, so `ri.run_arm` refuses every key and
  `ri.capture_reference` refuses outright. `inside` stays empty, so `Module.__call__`, every sealed instance and class
  `forward`, and the five low-level entry points all refuse.

**Demonstrated in the scratch clone** (stage-1 files committed; version 2 copy with exactly two substitutions,
`v2/clone_launcher/PATCH.txt`):
- **The setup.** The model load was allowed. My wrapper refused every forward route before the launcher captured them,
  which is stronger than a token-sequence guard.
- **The rehearsal** (`out/v2_g2_rehearsal_dry_run*`):
  - exit 0 in 32 s, with 21/21 preflight checks;
  - the dry run ok: dependencies, binding and geometry bit for bit; the barrier passed, with the committed table
    recomputed bit for bit (`91918004…`);
  - `load_model` 1, `sanctioned_forward_calls` 0, 0 forward attempts seen by my wrapper, `refused` [], `forbidden` [];
  - the clone's `results.json` sha256 unchanged (`6372d113…`), no write event in the clone, `tree_after` empty.
- **Two negative controls** (`out/v2_g2_negative_{geometry,table}*`): a falsified geometry comparison and a falsified
  table recomputation each make the dry run raise the runner's `IncidentError`. Each time: rehearsal exit 1, state
  unchanged, no write event in the clone, 0 forward attempts. So a barrier problem now surfaces as a harmless
  rehearsal refusal instead of a production phase incident.

### 2. The signal handling, phase by phase

| phase | SIGHUP | SIGTERM | effect of a SIGTERM | demonstrated |
|---|---|---|---|---|
| imports before `:83` (milliseconds) | inherited | default | dies; nothing written | — |
| hook, imports, preflight; rehearsal's dry run; smoke | ignored | default | dies; nothing written, no sentinel; rehearsal and smoke write nothing | — |
| production: terminal check, sentinel, `allowed_keys` (microseconds) | ignored | default | dies; the sentinel may remain; no ledger | — |
| `confirm()` before the ledger (load, G1, barrier) | ignored | `KeyboardInterrupt` | propagates: no incident, state `not_started`, record written, exit 1; move the sentinel aside to retry | `sigterm_before_ledger` |
| `confirm()` after the ledger (runs, save, gates, G6, `_recheck`) | ignored | `KeyboardInterrupt` | the runner records the incident and saves the partial file; record written; exit 1 | `sigterm_after_ledger` (incident "signal 15", partial saved) |
| `confirm()` after the result (`_secondary`) | ignored | `KeyboardInterrupt` | a secondary failure is recorded and the exception re-raised; the result stands; exit 1 | `sigterm_during_secondary` (state complete, failure recorded) |
| after `confirm()` returns or raises (post block, `git status`, record write, exit) | ignored | ignored | none | `sigterm_in_post_block`, `sigterm_during_tree_after`: exit 0, record written |

- **SIGHUP is harmless everywhere,** with or without `nohup` (`sighup_*`: exit 0, complete).
- **Inherited dispositions.** `git` children inherit the ignored dispositions, which is harmless.
- **A signal already pending when `SIG_IGN` is installed** is dropped: CPython skips ignored handlers.
- **Residual race (negligible).** A second SIGTERM arriving while the `finally` is still installing `SIG_IGN`, with
  another during the record write, could still lose the record.
- **SIGINT keeps Python's default.** A deliberate `kill -INT` during the post section would lose the record (R3).

### 3. `post_ok` cannot fail a correct run

A correct production run has:
- `phases.confirm == "complete"` (the result write; `_secondary` keeps it);
- every one of the 1,980 keys admitted once, so `done_keys == allowed_keys`;
- no refused event and no forbidden read or write.

On the last point:
- The preflight and the post-run code call no guard.
- Every recorded run so far had `refused` and `forbidden_*` empty: the calibrate and stage-1 production runs, both of
  my clone rehearsals, and my live load, forward and post-run tests.

**Executed:** `correct`, `sighup_*`, `sigterm_in_post_block`, `sigterm_during_tree_after` and `git_fails_after` all
exit 0 with `post_ok` true.

**What `post_ok` changes.** It can only turn exit 0 into exit 1. It never touches the state, so a label on disk stands
whatever the exit code. If some library ever swallowed a refused event, the exit code would flag it for review.

## Remaining notes (no change required)

- **R1. A pipe or socket on stdout or stderr is not refused.**
  - **The gap.** The docstring (`:48–49`) says "never through a pipe or `tee`", but `:590` refuses only terminals.
  - **The risk.** A pipe whose reader goes away during the run would make the runner's per-frame `print` raise
    `BrokenPipeError` after the ledger, which is an incident. One example is a tool capturing output that ends before
    the run: `nohup` redirects only when stdout is a terminal.
  - **Optional one-line hardening:** refuse `stat.S_ISFIFO` or `stat.S_ISSOCK` of `os.fstat(1)` and `os.fstat(2)`.
  - **Operationally,** `> production.log 2>&1` (a regular file), as calibrate and stage 1 did, is enough.
- **R2. Carried over from `REVIEW.md`, unchanged** (the runner's design or operator error):
  - N2: the microsecond window between `run.py:814` and `:820`; a second signal during `_record_incident`'s `_recheck`
    skips the partial save; an interrupted `_secondary` exits 1 although the result is complete;
  - N5: re-check N6's residual permissive cases, unreachable from confirm's code;
  - N6: concurrent launches from different directories.
- **R3. Cosmetic.**
  - `record["logs"]` is still set only on the non-exception path (`:609`).
  - The smoke's two new checks run after its 55 runs; before the model load would fail faster.
  - `barrier_dry_run` does not call `seed_runtime` as `confirm()` does. This has no effect: the isotropic directions
    are SHA-indexed Gaussians, and neither the geometry nor the table uses the global RNG; the clone dry run reproduced
    every digest. Calling it would make the mirror literal.
  - The terminal check runs after the ~10–25 s preflight; it is still before the sentinel.
  - SIGINT in the post section can still lose the record, but only by a deliberate `kill -INT`.
- **R4. The repository moved during this check, not by me.**
  - The user committed `f4e3a06` at 17:02: the stage-1 evidence, the launcher re-check, the stage-1 review and the
    README.
  - Since the lock commit `3621eec`, 151 paths changed, and `ri.scientific_changes` of them is `[]`. So `_validated_lock`
    stays satisfied.
  - The three stage-1 data files are still untracked. `outputs/`, `src/`, `run.py` and the stage-1 files are unchanged
    (`out/repo_snapshot_end_v2.txt` against the start snapshot: only HEAD, the README, `.git/index`, `COMMIT_EDITMSG`
    and the objects directory differ, all from that commit).
  - My clone predates `f4e3a06`; the difference is non-scientific.

## Executed scenarios (version 2's module-level SIGHUP statement and main block, unchanged; stub runner; `out/v2_e2_main_block.json`)

| scenario | exit | sentinel | state on disk | record |
|---|---|---|---|---|
| correct | 0 | 0644 | complete | `post` complete, every key once |
| incident after the ledger (returns 2) | 2 | 0644 | running + incident | `post` |
| SIGTERM after the ledger | 1 | 0644 | running + incident "signal 15", partial saved | error `KeyboardInterrupt` |
| SIGHUP under `nohup` / without `nohup` | 0 / 0 | 0644 | complete | `post`, no signal recorded |
| SIGTERM before the ledger | 1 | 0644 (remains) | not_started, no incident | error |
| SIGTERM during the secondary step | 1 | 0644 | complete + secondary failure | error |
| SIGTERM in the post block / during `tree_after` | 0 / 0 | 0644 | complete | written; SIGTERM ignored |
| stdout a terminal / stderr only a terminal | 1 / 1 | none | not_started | the terminal refusal |
| refusal before the ledger; preflight refuses; sentinel exists | 1 / 1 / 1 | remains / none / kept | not_started | error |
| git fails for `tree_after` | 0 | 0644 | complete | `tree_after` "unavailable" |
| a key not run / a swallowed refusal / a swallowed forbidden write | 1 / 1 / 1 | 0644 | complete | `post_ok` false |

## Files (this directory)

| file | sha256 | what |
|---|---|---|
| `reviewed_v2/launch_confirm_v2.py`, `hook_self_test.py`, `hook_self_test.out.json` | `56a687f3…`, `e4fe4332…`, `0520b32f…` | the copies checked |
| `out/v2_diff.txt`; `v2/a2_ast_diff_v1_v2.py` → `out/v2_a2_ast_diff.{json,log}` | `1f1f191e…`; `82e647e8…` → `cf7c3f04…` | the diff and the `ast` comparison |
| `out/v2_self_test.log` | `7e9516d4…` | the self-test from version 2's source |
| `v2/e2_main_block_driver.py`, `v2/e2_main_block_scenario.py` → `out/v2_e2_main_block.{json,log}` | `06d91898…`, `02301b10…` → `a6820b5b…` | the 18 scenarios |
| `v2/g2_make_clone_launcher.py`, `v2/g2_guarded_runpy.py` → `v2/clone_launcher/` (`PATCH.txt` `c5a44a4e…`, records), `out/v2_g2_*` | `54a58029…`, `368e57d0…` | the clone: rehearsal with the dry run, the two negative controls, production under a pty and to a file |
| `v2/i2_smoke_checks.py` → `out/v2_i2_smoke_checks.log` | `6251d913…` → `ba9bccc8…` | the smoke's new checks, evaluated without running |
| `out/v2_changed_since_lock_commit.txt`, `out/repo_snapshot_end_v2.txt` | — | R4 |
