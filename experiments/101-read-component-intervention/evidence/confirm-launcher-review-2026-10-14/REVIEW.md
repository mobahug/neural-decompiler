# Experiment 101 — review of the confirm launcher (2026-10-14)

**AI review (independent, read-only), not human validation.** A separate Claude agent (Opus 5.5) did this review from
the brief `PROMPT.md` in this directory. It modified nothing in the repository: a snapshot taken before and after is
identical apart from its timestamp line (`out/repo_snapshot_start.txt`, `out/repo_snapshot_end.txt`; `.git/index`
included). It never ran the launcher against the repository in any mode, never ran the runner's `confirm`, `stage1` or
`report`, and ran no 101 prompt. The only model forwards were three runs of one neutral sentence ("The quick brown fox
jumps over the lazy dog.", absent from every 101 file) behind a token-sequence guard that refused every other sequence.

## Verdict: READY WITH CHANGES

`launch_confirm.py`, sha256 `4f132e8e82d46e19dfad039cfcdf3b7bbdfa8d6199976f8b791dfbe3fba0b209` (the operator's copy is
still that file).

- **Blockers: none. Should-fix: one (S1), a one-line change.**
- **S1.** The launcher's SIGHUP handler cancels the protection of the `nohup` launch that its own docstring
  recommends. A terminal hangup during the run then becomes an incident that spends the confirmation, where under
  `nohup` alone it would have done nothing. Ignore SIGHUP instead (`SIG_IGN`), keep SIGTERM's conversion, require
  stdout and stderr not to be terminals, then rehearse again.
- **The guards are the calibrate launcher's, unchanged.** That machinery ran 2,016 sanctioned runs in production.
  - I executed it extracted from this file. The ledger check refused every pre-ledger state.
  - It admitted the three sanctioned run kinds: a plain run, a residual REPLACE and a head REPLACE.
  - It refused all nine unsanctioned routes, with 0 forwards.
  - Confirm's post-run computations ran outside any sanctioned call with every guard active: 0 forwards, 0 refusals.
- **The hook refuses no event of a correct confirm.** Every write and read path confirm takes was judged:
  - live, on the real write functions in a mirror;
  - live, during the counted, sealed model load and the neutral forwards;
  - on the real repository for the reads no rehearsal reaches: `_base`, the barrier's and G6's re-read of the stage-1
    files, and `_recheck`, which confirm calls on its success path before the result.

  0 refusals in all three; under the rehearsal rules, every repository write is refused.
- **The preflight passes in the post-commit state.** In a scratch clone with the three stage-1 files committed, a copy
  of the launcher differing only in ROOT and with `confirm()` patched out passed:
  - all 21 rehearsal checks;
  - the production preflight with the correct tokens (sentinel created, then the patched-out call);
  - and it refused wrong, swapped and half-wrong tokens before the sentinel.
- **The production path behaves as specified**, except for the signal points below. I executed the main block
  unchanged in 12 scenarios plus a `nohup` control.

**Before production (conditions):**
1. Apply S1, and N1 if convenient.
2. Re-verify the launcher's sha256.
3. After the user's commit, run rehearsal → smoke → production.
4. Start production detached, with stdout and stderr redirected to a file.

The tokens are the **content** sha256s:
- `CONFIRM101_LOCK=8da97e35556d0baa4739899cb841e86cbd59cced21611c732fdd9d8c9edd9061`;
- `CONFIRM101_TABLE=9191800440af37909d2068459e360d7772191fc66bf2ad5fb9cf2c133246ac86`.

## What was reviewed

| file | sha256 | role |
|---|---|---|
| `…/exp101/confirm/launch_confirm.py` (copied to `reviewed/`) | `4f132e8e…b209` | under review |
| `…/exp101/confirm/hook_self_test.py`, `.out.json` | `e4fe4332…`, `0520b32f…` | the operator's isolated self-test |
| `…/exp101/calibrate/launch_calibrate.py` (= committed `evidence/calibrate-2026-10-14/`) | `0ad9d0e1…` | guard precedent (ran in production) |
| `…/exp101/stage1/launch_stage1.py` (= committed `evidence/lock-2026-10-14/prepared/`) | `13b1992f…` | hook precedent (v2, found READY) |
| `experiments/101-read-component-intervention/run.py` | `9ce7e996…` | the runner (`confirm`, `_barrier`, `_committed_stage1`, `_record_incident`, `_secondary`) |
| `src/neural_decompiler/read_intervention.py` | `a93be725…` (blob `d7db6197…`, the lock's) | `run_arm`, `stage_runs`, `measure_frame`, `run_gates`, `primary_statistics`, `assert_phase_allowed` |
| `experiments/025-…/evidence/confirm-2026-09-26/launch_confirm_v3.py` | `58b760be…` | precedent |
| plan `docs/superpowers/plans/2026-10-05-…-plan.md` | `fbfa3345…` | §§ Phases (`confirm`), The prediction freeze, Gates, Incidents, One-shot protections |

The repository was at HEAD `961e81408e5c4da4fffdf9dfab0343e449a4e1f2` (`exp101-phases`), with the three stage-1 files
untracked. The state on disk: `phases.stage1` `complete` with `awaiting_commit` and no incident, a ledger of exactly the
36 S1 keys, and `phases.confirm` `not_started`.

## How

All scripts and outputs are in this directory. Each script installs my guard first, which refuses any write outside this
directory; processes touching the repository ran with `GIT_OPTIONAL_LOCKS=0`. Environment: the repository's
`.venv/bin/python -B`, `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`, and `TMPDIR` in scratch.

| step | script → output | result |
|---|---|---|
| a | `a_ast_diff.py` → `out/a_ast_diff.json` | `_guarded_call`, `_low_level`, `_sweep_aliases`, `_seal_model`, `_counted_load` and `_vector_digest` are identical to calibrate's; `_path`, `_inside`, `_in_repo` and `_temp_state` are identical to stage-1 v2's; `decide` differs from v2 only by `O_TRUNC`, the writable set and the `mkdir` allowance (diff checked by hand) |
| b | `b_decide_battery.py` → `out/b_decide_battery.json` | `decide` on a mirror: 27/27 allowed, 40/40 refused, and 47/47 repository writes refused under the rehearsal rules. The operator's self-test re-run with the launcher's own constants passes (real root and mirror root). 4 probes are re-check-N6 residuals (N5) |
| c | `c_real_ops.py` → `out/c_real_ops_{production,rehearsal}.json` | the real `_hook` in a mirror, working directory = mirror root, on the real write path: `rr.write_state_atomic` ×2, its cleanup on an injected `os.replace` failure, `rr.save_durably` (measurements and partial), `torch.load`, `rd.load_results_state`, `rc.file_sha256`, `rr.fsync_directory`. Production rules: all succeed and the 9 controls are refused. Rehearsal rules: every repository write is refused |
| d | `d_live_guards_load_forward.py` → `out/d_live_guards_load_forward.json` | the launcher's guard machinery (`ast`-extracted, production mode, `OUTPUTS` = mirror), the counted and sealed model load, and three sanctioned neutral runs; detail under check 1. 51 write-type or `outputs/` events judged live: 0 refusals |
| e | `e_main_block_driver.py` + `e_main_block_scenario.py` → `out/e_main_block.json` | the launcher's main block (`:535–587`) unchanged, with a stub preflight and a `Runner` of the real `run.py` at a scratch git repository: its real `_write`, `_record_incident` and `_preserve_partial`; a stub `confirm` that follows `run.py:810–887` (ledger, `try`, `except BaseException` → incident, partial, re-raise) without running anything; `_recheck` stubbed. 12 scenarios plus a `nohup` control (check 4) |
| f | `f_reads_real_repo.py` → `out/f_reads_real_repo.json` | the real `_hook` on the real repository (rehearsal rules, so every write is refused; reads are judged as in production). `_base` 7.3 s, `_committed_stage1` 0.3 s, `_recheck` 13.3 s → `{"ok": true}`; 0 refusals; under `outputs/` only 020's and 101's `results.json` |
| g | `g_make_clone_launcher.py`, `g_guarded_runpy.py` → `clone_launcher/*`, `out/g_clone_*.log` | a `--no-hardlinks` clone with the stage-1 files committed (blobs equal the record's sha256s) and its remote removed afterwards. Launcher copy with exactly two substitutions (`clone_launcher/PATCH.txt`); my wrapper also refused every model load and forward. Rehearsal exit 0, 21/21 checks, `tree_after` empty. Production: wrong, half-wrong and swapped tokens refused with no sentinel; correct tokens → sentinel, then the patched-out call |
| h | `h_import_probe.py` → `out/h_import_probe.log` | `import torch` raises the tempfile probe and `import transformers` raises filelock's probe, both in `TMPDIR` and both under the launcher's hook in every mode. The `torchinductor_<user>` mkdir comes at the model load |

## Findings (most severe first)

### Blockers: none.

### Should-fix

**S1. The SIGHUP handler cancels the protection of `nohup`. A hangup then becomes an incident that spends the
confirmation.**
- **Where.**
  - `launch_confirm.py:555`: `signal.signal(signal.SIGHUP, _signal_to_interrupt)`.
  - The docstring's own advice at `:41–42`: "Launch it detached (`nohup`)".
- **Mechanism.**
  1. `nohup` starts the process with SIGHUP set to `SIG_IGN`, so a hangup is harmless.
  2. Just before the sentinel, the launcher replaces `SIG_IGN` with a handler that raises `KeyboardInterrupt`.
  3. A SIGHUP during the ~10-minute run then interrupts `ri.stage_runs` (or the save, the gates or `_recheck`) after
     the ledger.
  4. The runner's `except BaseException` (`run.py:862–867`) records an incident. The one-shot confirmation is consumed
     and the report renders `NOT_INTERPRETABLE`.
- **Scenario.** The run is started from an interactive terminal, as the docstring suggests:
  `nohup env CONFIRM101_MODE=production … python -B launch_confirm.py > production.log 2>&1 &`. The terminal tab or
  window, or the session holding it, is then closed before the run ends, and zsh forwards SIGHUP to its jobs when it
  exits.
- **Evidence** (`out/e_main_block.json`).
  - **`sighup_under_nohup`**, started through `/usr/bin/nohup`:
    - the SIGHUP disposition at start is 1 (`SIG_IGN`);
    - inside `confirm` it is the launcher's handler;
    - a SIGHUP after the ledger leaves the state `running`, with incident "signal 1" and the partial file saved, and
      the launcher exits 1.
  - **`nohup_control`**: the same SIGHUP against the inherited `SIG_IGN`. The process prints "still running after
    SIGHUP" and exits 0.
- **Why ignoring is strictly better.**
  - Under `nohup`, `SIG_IGN` lets the run finish (`nohup` also sends a terminal's stdout and stderr to `nohup.out`).
  - Without `nohup`, the default action would kill the run silently after the ledger. With `SIG_IGN` and the output in
    a file, it finishes.
  - The handler gives an incident in both cases.
- **Fix.**
  - `signal.signal(signal.SIGHUP, signal.SIG_IGN)` in place of `:555`.
  - Keep `:554`: a SIGTERM is a deliberate stop, and converting it records the incident (`sigterm_after_ledger`).
  - In production, refuse unless stdout and stderr are not terminals:
    `need("output_not_a_terminal", not sys.stdout.isatty() and not sys.stderr.isatty())`.
- **Why the output check matters.** After a hangup, a write to the dead terminal raises `OSError` in the runner's
  per-frame `print` (`run.py` via `stage_runs(log=self.log)`). That is an incident too.
- **Then.** Re-run the self-test and the rehearsal, and record the new sha256.

### Notes

**N1. A signal after `confirm()` returns can lose the launcher's record. The science is unaffected.**
- **Where.** The handlers stay installed after `:561`. The code after the `try` (`:572–582`: `record.update`, `git
  status` for `tree_after`, the O_EXCL record write) is outside every `except BaseException`.
- **Evidence.** `sigterm_during_tree_after` gives exit −2 (Python dies by SIGINT after the uncaught
  `KeyboardInterrupt`) and 0 records, while the state on disk is `complete`.
- **Window.** About 10–20 ms (`git status` takes 0.01 s here).
- **Fix, with S1.** Set SIGTERM (and SIGHUP) to `SIG_IGN` as the first statement after `confirm()` returns or raises,
  for example in a `finally` around `:561–568`.

**N2. Residual interruption windows after the ledger (the runner's design, inherited by the SIGTERM conversion as by
SIGINT).**
- **Between `run.py:814` and `:820`**, or inside the ledger's own `write_state_atomic` after its `os.replace` (during
  `fsync_directory`), no incident is recorded.
  - The state is then `running`, with the 1,980 keys.
  - `confirm` is refused for good, and so is `report`, which needs a complete confirm or a recorded incident
    (`read_intervention.py:2082–2086`).
  - This window is microseconds long.
- **A second signal during `_record_incident`'s `_recheck`** (13.3 s measured) leaves the incident on disk from its
  first write but skips `_preserve_partial`, so the partial tensors are not saved.
- **An interruption during `_secondary`, after the result:**
  - it is recorded as a secondary failure and re-raised;
  - the launcher exits 1 with no `post` block, although `phases.confirm` is `complete` with its label.
  - Read the state before concluding anything from an exit code of 1.

**N3. The exit code ignores the launcher's own post checks (`:587`).**
- **What.** `all_1980_ran_once`, `events["refused"]` and `forbidden_writes`/`forbidden_reads` are recorded but do not
  gate exit 0. Stage 1's `post_ok` did gate it.
- **Evidence.** `keys_not_all_run` (`all_1980_ran_once: false`) and `refusal_swallowed` both exit 0.
- **Why it matters little.** The runner's accounting (`run.py:827–834`) catches any mismatch in the runs. A refusal
  swallowed in `_secondary` would only show as `secondary_failures`.
- **Optional.** An additive `post_ok` (status 0, phase `complete`, every key once, no refusal, no forbidden event)
  gating exit 0.

**N4. Operational conditions for production.**
- **Detached, with output to a file.** Never pipe it through `tee` or any reader that may die: a `BrokenPipeError`
  from the per-frame log after the ledger is an incident.
- **Duration.** Expect about 10 minutes:
  - calibrate's 2,016 runs took 6.5 minutes;
  - `_base` runs twice, at about 7 s each;
  - then the barrier, the gates, `_recheck` (13 s) and the bootstrap.
- **No working-tree change in the repository during the run.**
  - `checkout`, `reset`, `stash` or `pull` could change the stage-1 files or 020–025's committed inputs.
  - G6-before-result (`run.py:847`) and `_recheck` (`run.py:859`) re-read them after the ledger, so either would be
    an incident.
  - Committing other files (index and HEAD only) is harmless.
- **`GIT_OPTIONAL_LOCKS=0`** in the environment, or `os.environ.setdefault(...)` in the launcher. The runner's
  `collect_git_state` runs a plain `git status`, which may rewrite `.git/index` (lock review N10).
- **The launcher's sha256** is checked immediately before the run.
- **The tokens are content sha256s**, not file sha256s:
  - the lock: `8da97e35…` (its file sha256 is `8dc6e6d3…`);
  - the table: `91918004…` (its file sha256 is `71d339cd…`).
- **Durable paths.** `HERE` holds the sentinel, `production_runs.jsonl` and the record, and the log should live there
  too. Keep them where they outlive the session: the scratchpad is session-scoped. If `HERE` vanished mid-run:
  - the final O_EXCL record write would fail, so there would be no record;
  - the open run log would be unlinked;
  - the results state in the repository would be unaffected.
- **Order.** Run rehearsal, then smoke, then production, all after the user's commit. My clone rehearsal predicts
  21/21 checks.

**N5. Re-check N6 is only partly addressed.**
- **Fixed.** `O_TRUNC` now counts as a write (`:130`); verified in the battery and in the real hook (an `O_RDONLY|O_TRUNC`
  open of a stage-1 file is refused).
- **Still open, none reachable from confirm's code.** Every confirm write is an absolute path through Python's `open`,
  `os.replace` or `os.mkdir`.
  - A relative `open` that names an allowed file is judged from the working directory: the `open` event carries no
    `dir_fd`.
  - Case and firmlink spellings of the repository pass the string-prefix test (probe: allowed).
  - Unhandled events are not judged: `os.utime`, `chflags`, xattrs.
  - `torch.save` to a path writes through C++ I/O, which raises no `open` event. Demonstrated: it wrote
    `probe-torch-save.pt` in my mirror under the production hook.
  - Child processes are not judged.

**N6. Concurrency (re-check N9, carried over).**
- **The gap.** The sentinel is per launcher directory. Two production launchers started from different directories
  would both pass the preflight and the runner's phase rule, whose window runs from `_state_for` through the model
  load, G1 and the barrier. Each would then run every S2 key.
- **Likelihood.** Operator error only.
- **The fix.** An exclusive lock at a fixed path outside the repository and `HERE`. It would also make the plan's "the
  barrier passed in this process" (§ The prediction freeze 5) literal: the launcher checks the barrier on disk, which
  equals "in this process" only absent concurrency.

**N7. The rehearsal does not dry-run the weights-only part of the barrier, as 025's did with I7′.**
- **What is not exercised.** `_model_dependencies`, `_check_stage1_binding`, `_geometry_against_lock`, and `_barrier`
  with its table recomputation.
- **What a failure there costs in production.** It is a phase incident that refuses confirm for good, with no S2
  prompt run (plan § The prediction freeze 4).
- **Risk: low.**
  - Stage 1 passed G1 and the digests in this environment today.
  - The lock showed the exposed table reproduces bit for bit across processes.
  - If the stage-1 review recomputed the table from the committed states and the weights, the remaining part is
    covered too.
- **Optional hardening.** A rehearsal step that loads the counted model and calls those four methods. They write
  nothing and run no forward, so such a surprise would become a refusal before the sentinel.

**N8. Record and docstring cosmetics.**
- `record["logs"]` is set only on the non-exception path (`:569`). After an interruption, the runner's
  "INCIDENT (confirm)" line exists only in the console log (`has_logs: false` in the signal scenarios).
- The sentinel is created with the default mode, 0755 (`:556`).
- The record is not fsynced.
- `events["outputs_reads"]` also lists the allowed writes under `outputs/`.
- Docstring `:13` says N2 also compares "the stage-1 run's [record], once committed". Only the lock run's is compared;
  the two are equal transitively.

**N9. Smoke specifics.** The code is correct (check 5). Three things to know when reading its result:
- **The reference's history differs.** The unguarded reference ran an S1 re-capture before the 55 runs; the confirm
  smoke runs none. Bit equality is still expected: the calibration's production digests for that frame equal the
  reference's although 36 re-captures preceded them. A mismatch would be a false alarm about history, not a guard
  defect.
- **The reference is not hash-checked.** It is read from the scratch copy (`../calibrate/smoke_reference.pt`, sha256
  `ba4698a5…`, identical to the committed `evidence/calibrate-2026-10-14/smoke_reference.pt`).
- **The keys are not re-asserted.** The 55 keys are calibration keys whose spent-exposed status and disjointness from
  101 the calibrate preflight verified. The confirm smoke does not re-assert either.

## The brief's checks, one by one

### 1. No target prompt can run outside the protocol

- **The runner's order** is barrier → ledger → runs.
  - `run.py:804–809`: G1, the digests and `_barrier`.
  - `:810–814`: `phases.confirm` running, the 1,980 S2 keys added to the ledger, `confirmation.barrier.passed`, one
    atomic write.
  - `:821`: `ri.stage_runs`.
- **The launcher's ledger check** (`_ledger_on_disk`, `:299–307`) needs all three on disk:
  - every allowed key in the ledger;
  - `confirmation.barrier.passed is True`;
  - `phases.confirm.status == "running"`.

  Only `run.py:810–814` writes them, after `_barrier` returned; a barrier incident (`_record_phase_incident`) writes
  none of them. Executed (step d): the state as confirm finds it, the barrier without the keys, and the keys without
  the barrier are each refused with 0 forwards; the full ledger is admitted.
- **The sanctioned route** is `ri.run_arm` wrapped by identity (`:314–335`). It admits a call only if:
  - the key `prompt.key|tag` is one of the preflight's 1,980 S2 keys;
  - the key has not run (it is marked done at admission, so a failed run is not retried);
  - the ledger check passes.

  `measure_frame` calls `run_arm` through `ri`'s module global, which the sweep replaced. Each admission is logged with
  its replacement digests (verified: `out/d_production_runs_mirror.jsonl`).
- **Everything else is refused** (step d, each with 0 forwards):
  - `ri.capture_reference`, outright (`:338–339`);
  - `Module.__call__`;
  - the instance `forward` (430 instances sealed) and the class `forward`, `run_with_hooks`, `run_with_cache` and
    `generate` (20 class methods; signatures kept);
  - `pm.capture_prompt`, `pm.run_patched` and `rd.capture_frame_020` outside a run;
  - a second run, a key outside the set, and a second model load.
- **The production sweep wraps 11 aliases** (calibrate's and stage 1's records).
- **Confirm never captures a reference.** The barrier and G6 use the committed states.
- **Inside a sanctioned run there is exactly one forward.** Each run kind made 340 module calls (plain, residual
  REPLACE, head REPLACE), as calibrate's 685,440 / 2,016 did. The NOOP-like residual and head replacements reproduce
  the plain logits bit for bit.

### 2. The audit hook

**Events of a correct confirm (Python 3.12.13) and their decisions under the production rules.** Recorded live in steps
c, d, f and h. All are allowed; under the rehearsal rules every repository write is refused.

| step | events, as raised | decision |
|---|---|---|
| `rr.write_state_atomic` (ledger, gates, result, secondary, incident) | `os.mkdir(OUTPUTS, 511, -1)`; `tempfile.mkstemp(…/.results-XXXXXXXX.json)`; `open(tmp, None, 16780034)` (`O_RDWR\|O_CREAT\|O_EXCL\|O_NOFOLLOW\|O_CLOEXEC`); `open(<fd>, 'w', 16778753)`; `os.rename(tmp, results.json, -1, -1)` (from `os.replace`); `open(OUTPUTS, None, 16777216)` (directory fsync). On failure: `os.remove(tmp, -1)` | allow (mkdir only of `OUTPUTS`, production only; `_temp_state`; the descriptor is not a path; the one rename onto `results.json`; a read inside `OUTPUTS`) |
| `rr.save_durably` (`stage2-measurements.pt`, `stage2-partial.pt`) | `os.mkdir(OUTPUTS)`; `open(path, 'w', 16778753)`; the directory fsync | allow (`WRITABLE`) |
| `torch.save(value, handle)` | none of its own (it writes through the Python file object) | — |
| `torch.load` (torch 2.14, `weights_only=True`) | `open(path, 'r', 16777216)`; no `pickle.find_class` | allow (read inside `OUTPUTS`) |
| imports, under the hook in every mode | `/dev/null` `r+`/`w` (dill); `import torch`: tempfile's probe `open(TMPDIR/x, None, 16780034)` + `os.remove`; `import transformers`: filelock's probe `mkdtemp`, `os.mkdir(…, 448)`, `os.utime`, `open(probe-source)`, `os.link`, `shutil.rmtree`, `os.remove('probe-source', 3)` and `os.remove('probe-link', 3)` (`dir_fd` resolved by `F_GETPATH` while open), `os.rmdir` | allow (outside the repository) |
| `seed_runtime`, model load | `os.mkdir(TMPDIR/torchinductor_<user>, 511, -1)`; 1,412 reads (HF cache, `.venv`) | allow |
| `ModelPrograms`, `_base`, `_recheck` | reads; under `outputs/` only `experiment-020/results.json` and `experiment-101/results.json` | allow (`READABLE_OUTPUTS`) |
| `_barrier` / G6-before-result (`_committed_stage1`) | reads of the three stage-1 files (`read_bytes`, `read_text`, `rc.file_sha256`) | allow |
| forwards (plain, residual REPLACE, head REPLACE) | none; the first admission reads `results.json` once | allow |
| the secondary step (`ri.run_records`, `ri.secondary_statistics`, `encode_nonfinite`) | none, besides the state writes above | allow |

**Refusals.** No event of a correct confirm is refused. The decision battery, the controls and the probes give:
- **Refused** (each as expected):
  - writes to the stage-1 files, the lock, the runner, a source file, evidence, `report.md`,
    `calibration-measurements.pt` or another output;
  - a renamed measurement;
  - removing the state or the measurements;
  - `mkdir` under `outputs/` or of `stage1/`;
  - `chmod` or `truncate`;
  - relative names at the root;
  - an unresolvable or closed `dir_fd`;
  - reads of another experiment's outputs or of `outputs/` itself.
- **Not refused:** only the N5 residuals.

### 3. The preflight

All of these are refusal-class: `:423–496` raises before the sentinel (`:556`).

| item | where | status |
|---|---|---|
| `-B`; `HF_HUB_OFFLINE=1` | `:431–432` | present |
| N1: root, clean tree (untracked files included), the runner's git state | `:433–436` | present |
| N2: environment identity with the lock run (python, executable, platform, packages, threads, model, checkpoint files) | `:437–442` | present (N8: the stage-1 run's is not compared) |
| the lock: content digest, rendering, committed | `:443–446` | present |
| module binding: `ri`'s blob = the lock's = HEAD's = the imported file's; all 20 pinned blobs = the lock's | `:447–450` | present |
| stage-1 phase `complete`, `awaiting_commit`, **no incidents** (re-check N1) | `:454–455` | present |
| the stage-1 record's digest reproduces and names the lock | `:456` | present |
| the three files tracked, sha256 = the record's, in HEAD, and exactly `STAGE1_FILES` | `:457–462` | present (blob = bytes: no `.gitattributes`, no `autocrlf`; verified in the clone with `git cat-file`) |
| confirm `not_started`, no confirmation block | `:463` | present |
| the two production tokens (lock content, table content) | `:464–466` | present (tested: wrong, half-wrong and swapped are refused) |
| production configuration; `_state_for("confirm")`; `_validated_lock` with the stage-1 files as extra tracked paths | `:481–487` | present |
| the 020 runtime and versions | `:488` | present |
| ledger = exactly the 36 S1 keys; 1,980 distinct S2 keys, none in the ledger, none forbidden | `:489–493` | present |
| the hook's self-test | `:494–495` | present |

In the clone with the files committed, all 21 checks are true (`clone_launcher/rehearsal_run_*.json`).

### 4. Production behaviour

**Order.** The sentinel is created after the preflight (`:556`), and `confirm()` is called once (`:561`). Executed
scenarios (`out/e_main_block.json`):

| scenario | exit | sentinel | record | state on disk |
|---|---|---|---|---|
| correct | 0 | yes | post complete, all keys once | complete |
| incident after the ledger (returns 2) | 2 | yes | post | running + incident, partial saved |
| SIGTERM after the ledger | 1 | yes | error `KeyboardInterrupt: signal 15`, no logs | running + incident "signal 15", partial saved |
| **SIGHUP under nohup** | 1 | yes | error `… signal 1` | running + incident "signal 1": **S1** |
| **SIGTERM during `tree_after`** | −2 | yes | **none** | complete: **N1** |
| refusal before the ledger (`PhaseError`) | 1 | yes (move it aside to retry) | error | not_started |
| preflight refuses | 1 | no | error | not_started |
| sentinel exists | 1 | kept | `FileExistsError`; confirm not called | not_started |
| earlier records present | 0 | yes | a new O_EXCL file; the old one kept | complete |
| git fails for `tree_after` | 0 | yes | `"unavailable: …"` | complete |
| keys not all run / refusal swallowed | 0 / 0 | yes | `all_1980_ran_once: false` / a refusal in `events` | complete: **N3** |

- **Signals.** After the ledger, SIGTERM (and SIGHUP) do reach the runner's `except BaseException` as an interruption:
  the incident is recorded, the partial tensors are saved, the exception is re-raised, and the record is written. The
  exceptions are S1 (SIGHUP under `nohup`) and the windows in N1 and N2.
- **The `post` block** reads only the state on disk, after `confirm()` returns.
- **The record** is a new O_EXCL file with a microsecond stamp and mode 0644 (`:579–582`).
- **The exit code** is 0 only for status 0 with no error; otherwise the runner's status, or 1 (N3).
- **`_secondary` runs no forward.** On synthetic data under every guard: 0 forwards and 0 refusals. A failure there is
  recorded and the result stands. Its state writes are ordinary atomic writes, allowed. The final `state_sha256` is the
  one after `_secondary`, and the `post` block records it.

### 5. The smoke mode

- **The runs** (`:499–532`): `coordinated-adjective-1`'s 55 keys from 020's exposed pool, through
  `runner._calibration_schedules(...)[0]` and the same `ri.stage_runs`. They are calibration keys, spent and exposed.
- **The comparison:** the 71 tensor digests and the 55 logits digests against `../calibrate/smoke_reference.pt`.
- **The probes**:
  - `torch.tensor([[1, 2, 3]])` through `model(...)` and `model.forward`;
  - `pm.capture_prompt` and `ri.capture_reference` on the same exposed prompt;
  - `ri.run_arm` on the second exposed frame's reference prompt (outside the set);
  - a second run of the first key.

  Each must be refused with 0 forwards.
- **What it never does.** It never builds or runs a 101 prompt as a probe. The preflight, in every mode, constructs
  101 `Prompt` objects only to derive the 1,980 key strings.
- **What it writes:** nothing in the repository (rehearsal rules).
- **Not executed here.** The brief allows only a neutral sentence if the model is loaded. Step d exercised the same
  wrappers on the neutral sentence. N9 notes the caveats.

### 6. Anything else

- **A target prompt twice:** prevented by the runner's one-shot phase rule, the O_EXCL sentinel and `done_keys`. The
  only gap is concurrent launches from different directories (N6).
- **A file other than the allowed ones:** no confirm code path writes one. The residuals are adversarial only (N5).
- **The record lost:** N1 (a signal after `confirm()` returns), N4 (the session-scoped `HERE`), or SIGKILL.
- **A correct confirm failing:**
  - not through the hook or the guards (checks 1 and 2);
  - through a hangup (S1);
  - through the output channel or a working-tree change (N4);
  - through a barrier-time environment surprise, which costs a phase incident (N7).

## Files (this directory)

| file | sha256 | what |
|---|---|---|
| `PROMPT.md` | — | the brief |
| `REVIEW.md` | — | this report |
| `reviewed/launch_confirm.py`, `hook_self_test.py`, `hook_self_test.out.json` | `4f132e8e…`, `e4fe4332…`, `0520b32f…` | the reviewed copies |
| `scripts/common.py` | `432623ba…` | my guard, the `ast` extractor, the live judge |
| `scripts/a_ast_diff.py` → `out/a_ast_diff.json` | `f3a337c7…` → `102ab25d…` | step a |
| `scripts/b_decide_battery.py` → `out/b_decide_battery.{json,log}` | `529dafd6…` → `ff73b4e0…` | step b |
| `scripts/c_real_ops.py` → `out/c_real_ops_{production,rehearsal}.{json,log}` | `75c28bd0…` → `50d2539c…`, `194bae1c…` | step c |
| `scripts/d_live_guards_load_forward.py` → `out/d_live_guards_load_forward.{json,log}`, `out/d_production_runs_mirror.jsonl` | `e555bbb2…` → `60573aab…` | step d |
| `scripts/e_main_block_driver.py`, `scripts/e_main_block_scenario.py` → `out/e_main_block.{json,log}` | `ef3c94ad…`, `99b0e264…` → `6183dc2f…` | step e |
| `scripts/f_reads_real_repo.py` → `out/f_reads_real_repo.{json,log}` | `a3ac981d…` → `18fae52e…` | step f |
| `scripts/g_make_clone_launcher.py`, `scripts/g_guarded_runpy.py` → `clone_launcher/` (`PATCH.txt`, records), `out/g_clone_*.log` | `753f9048…`, `8a20561c…` | step g |
| `scripts/h_import_probe.py` → `out/h_import_probe.log`; `out/h_import_probe_launcher_order.log` | `a3256986…` → `c6355628…` | step h (the second log: the launcher's own import order; no write-type event after `transformers`, so the inductor `mkdir` comes at the load) |
| `out/smoke_reference_vs_calibration.txt` | — | the calibration's digests for `coordinated-adjective-1` equal the smoke reference's |
| `out/repo_snapshot_start.txt`, `out/repo_snapshot_end.txt` | — | the repository before and after: identical apart from the timestamp line |

**Scratch trees.**
- `mirror_*`: the mirror trees.
- `main_block/<scenario>/`: the scratch git repositories and launcher directories.
- `clone/`: the scratch clone (about 0.5 GB; its remote removed; deletable).
- `tmpdir*`, `work_d`: the `TMPDIR`s and the working directory.

**The 9 aliases of the clone rehearsal versus 11 in production** are an artefact of my wrapper, which pre-replaced
`capture.run_capture`. `candidate_screening`'s two aliases then no longer matched the launcher's originals. Production
records 11.
