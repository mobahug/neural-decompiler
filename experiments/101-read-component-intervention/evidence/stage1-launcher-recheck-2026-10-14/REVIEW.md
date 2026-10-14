# Experiment 101 — re-check of the stage-1 guarded launcher, version 2 (2026-10-14)

**AI review (independent, read-only), not human validation.** A separate Claude agent (Opus 5.5) did this re-check from
the brief `PROMPT.md` in this directory. It modified nothing in the repository: a snapshot taken before and after is
identical (`out/repo_snapshot_start.txt`, `out/repo_snapshot_end.txt`). It did not run the launcher in production mode
or the runner's `stage1`, `confirm` or `report`, and it ran no 101 frame prompt.

## Verdict: READY FOR PRODUCTION

Version 2 can run the one production stage 1 **unchanged**. That is `launch_stage1.py`, sha256
`13b1992f7f0feb652207ca835df044244f20faea750e92f42d8d51bced7c3e1c`: the committed copy in
`evidence/lock-2026-10-14/prepared/` and the operator's working copy, which is byte-identical.

- **Blockers: none. Should-fix: none.**
- **The lock review's findings are fixed.** S1, S2, S3 and N8 are fixed. Nothing version 1 got right regressed: the forward guards, the sealing, the counted load, `write_check` and the preflight are textually unchanged from version 1.
- **The hook refuses no event of a correct stage 1.** I installed v2's real audit hook in a scratch mirror tree and drove the real stage-1 write functions through it. I also judged, at the moment each event was raised, every file event of:
  - the model load;
  - two forwards with stage 1's site set (a neutral, non-101 sentence);
  - the program construction;
  - G1, the prediction table and the states serialization (on the 36 exposed, spent frames, as the lock phase did).

  The decision function refused none of them, so the hook cannot turn a correct run into a stage-1 incident after the ledger.
- **The preflight passes in the real state and refuses before the sentinel.** My rehearsal from a copy passes all 16 checks and reproduces the operator's record except for timestamps and the launcher path.
- **The production path behaves as specified.** I executed the launcher's own main block against a stub runner in a scratch git repository, in 12 scenarios. The post-run checks, the O_EXCL record, the sentinel and the exit code all behaved as specified.

**Conditions (operational only; no code change):**
- Launch it detached, so no tool timeout or closed session can kill it after the ledger (N2).
- Keep everything else from writing in the repository during the run (N3).
- Verify the launcher's sha256 and set `HF_HUB_OFFLINE=1` (N4).
- Carry **N1** into the confirm launcher.

## What was reviewed

| file | sha256 | role |
|---|---|---|
| `evidence/lock-2026-10-14/prepared/launch_stage1.py` (= the operator's `…/exp101/stage1/launch_stage1.py`) | `13b1992f…c1c` | under review |
| `evidence/lock-2026-10-14/prepared/launch_stage1_v1_as_reviewed.py` | `bf40e043…5f1` | version 1, for the diff |
| `evidence/lock-2026-10-14/prepared/hook_self_test.py`, `.out.json` | `7f330d51…`, `e4235392…` | the operator's isolated self-test |
| the operator's `rehearsal.log`, `rehearsal_run_20261014T120823753435Z.json` | `12a469d6…`, `0a622660…` | the operator's rehearsal |
| `experiments/101-read-component-intervention/run.py` | `9ce7e996…` | the runner (frozen) |
| `src/neural_decompiler/read_intervention.py` | `a93be725…` (blob `d7db6197…`, the lock's) | `capture_reference`, `run_arm`, the phase rules |
| `src/neural_decompiler/upstream_localization.py` | `0b0cdbbd…` | `write_table` |
| `preregistration-lock.json`, `preregistration.md` | `8dc6e6d3…`, `3761b813…` (content `8da97e35…`) | at HEAD `961e81408e5c4da4fffdf9dfab0343e449a4e1f2` |

The specification is plan §§ Global constraints, Phases (`stage1`), The prediction freeze, Incidents, stops and refusals,
and One-shot protections and guarded launchers. I also read the lock review's S1–S3, N8 and addendum, and
`f_audit_events.json`.

## How

1. **Reading.**
   - I read v2 in full, line by line.
   - An `ast`-level diff against v1 shows that only `_path`, `_in_repo`, `_temp_state`, `write_allowed` and `_hook`, the new `decide` and `hook_self_test`, and the main block changed.
   - `_guarded_call`, `_low_level`, `_seal_model`, `_sweep_aliases` and `_counted_load` are also identical to the calibrate launcher's, which ran in production (2,016 sanctioned runs, 36 of them through `ri.capture_reference`).
   - I read the runner's `stage1` path (`run.py:660–733`) and every helper it reaches.
2. **Decision battery** (`scripts/b_decide_battery.py`). `decide()` was extracted with `ast` and run on 70 event tuples against a scratch mirror tree, judged as from the repository root.
   - **Allowed (28):** realistic stage-1 events and writes outside the repository.
   - **Refused (42):** adversarial and relative-path cases.
   - **Probes:** 13 known gaps, reported rather than asserted.
   - **Result:** 70/70 as expected; 0 of the 41 repository-write cases is allowed under the rehearsal rules.
3. **The real hook on real operations** (`scripts/c_real_ops.py`). `_hook` was installed as an audit hook (ROOT = the scratch mirror, working directory = mirror root). It drove `rr.write_state_atomic` (including its cleanup path), the runner's `stage1/` `mkdir(parents=True, exist_ok=True)`, the launcher's `write_check`, `ul.write_table`, `Path.write_text`, `rc.file_sha256` and `rd.load_results_state`, plus 13 refused controls and 7 bypass probes.
   - **Production rules:** the whole write path succeeds and every control is refused.
   - **Rehearsal rules:** every repository write is refused.
4. **Model load and forwards** (`scripts/d2_load_forward_live.py`). I recorded every audit event of the steps no rehearsal reaches:
   - `seed_runtime`;
   - `models.load_model`;
   - `ul.load_frozen_inputs`, `ul.ModelPrograms.from_model` and `rr.parameters_digest`;
   - two forwards through `pm.capture_prompt` with stage 1's plain site set, on "The quick brown fox jumps over the lazy dog." (absent from `read_intervention.py`, `confirmation-v1.json`, `calibration-v1.json` and the design).

   Each write-type event was judged **at event time** by v2's `decide` (production rules, ROOT = the repository, working directory taken as ROOT), with `TMPDIR` in scratch: **14 write-type events (15 with the script's own output file), 0 refusals.**
   - **Model load.** It runs `filelock`'s import-time probe: `tempfile.TemporaryDirectory`, `probe-source`, `os.link`, then `shutil.rmtree`, which removes entries by name with `dir_fd = 3`. v2 resolves that descriptor correctly while it is open.
   - **Forwards.** They raise no file event at all.
   - **An earlier version of this script** (`d_load_forward_events.py`) judged after the fact and so mis-resolved `dir_fd` 3. It is kept for the record; the live judge supersedes it.
5. **Pure computations** (`scripts/f_program_events.py`). `ri.read_geometry`, `ri.geometry_block` + `ri.enforce_g1`, `ri.prediction_table`, `ri.states_blocks`, `ul.table_bytes` and `rd.state_digest` ran on the 36 exposed, spent frames, as the lock phase did, under the same live judge.
   - **Events:** no file event beyond the load's; 0 refusals.
   - **Bonus cross-check:** the exposed table reproduces the calibration's digest bit for bit (`ce055e09c73613b7…`), and the states layout has 324 blocks (5,755,264 bytes), the lock's layout. So the environment still reproduces (the lock review's N9).
6. **The main block** (`scripts/e_main_block.py`). The launcher's code from `record: dict = …` (`:475`) to the end was executed unchanged, with a stub `preflight()` and a stub runner, in a scratch git repository whose ignore rules for these paths are the repository's. It used the real `rr.write_state_atomic`, `rd.load_results_state` and `b0c.stage_one_digest`, across 12 scenarios (table under check 4).
7. **The rehearsal from my copy** (`rehearsal/`, with `GIT_OPTIONAL_LOCKS=0`): exit 0, 16/16 preflight checks, self-test ok.
8. **Platform facts** (`scripts/g_platform_probes.py`): `O_RDONLY|O_TRUNC` truncates on this macOS, and the repository is reachable under spellings the hook's string test does not match (N6).

## Findings (most severe first)

### Blockers: none.

### Should-fix: none.

### Notes

**N1. A stage-1 incident recorded after the status turned "complete" is invisible to `confirm`'s phase rule, and the
launcher's post block then reads all green. Carry this into the confirm launcher.**
- **Where.**
  - `run.py:720–722`: the runner sets `state["stage1"] = record` and `phases.stage1.status = "complete"` (with `awaiting_commit`) in memory before the final `self._write(state)`.
  - `run.py:723–724`: the `except` records the incident on that same object (`_record_phase_incident`, `run.py:398`, which writes `phases.stage1.incidents` and keeps the status).
  - `read_intervention.py:2077–2079`: `assert_phase_allowed("confirm")` checks `state["stage1"].get("incident")`, a key nothing in 101 writes, and never reads `phases.stage1.incidents`.
  - Launcher `:499–509` and `:524–526`: the post block and `post_ok` never read `state["phases"]["stage1"]`.
- **Scenario.** The final state write is interrupted (SIGINT during its `fsync`) or fails transiently, and the incident write succeeds.
  - **On disk:** status `complete`, `awaiting_commit: true`, an `incidents` entry (`after_ledger: true`) and an intact stage-1 record.
  - **The launcher** exits 2 (or 1 on an interruption), which is right. Its `post` block reports all five checks `true` (my scenario h).
  - **Later:** `confirm`'s phase rule and its barrier would both pass.
- **Likelihood.** Low: a millisecond window. A disk-full error fails both writes and leaves the status `running`, which confirm refuses.
- **Recommendation.**
  - The confirm launcher's preflight must refuse unless `phases.stage1` is `complete` with `awaiting_commit` and **no `incidents`**.
  - Optional here: add `"phase": state["phases"]["stage1"]` to `record["post"]`, and `status == "complete" and not incidents` to `post_ok`. Both are additive and run after `stage1()` returns, so they cannot affect the run.

**N2. Launch it so that nothing can kill it after the ledger.**
- **The gap.**
  - The launcher writes its record only at the end (`:513–521`).
  - The runner records an interruption only if it arrives as a Python exception (SIGINT, `run.py:723–727`).
  - SIGTERM, SIGHUP or SIGKILL end the process silently: a tool's command timeout, a closed terminal or a stopped session. The 36 keys are then in the ledger with status `running`, so the frames are spent and stage 1 is refused for good, with no incident entry and no launcher record.
- **Duration.** About a minute: the lock run, with the same load, G1 and a 36-frame table, took 39 s; the rehearsal takes about 15–25 s with imports and checkpoint hashing; and 36 captures are a few seconds at calibrate's rate (380 s for 2,016 runs). That is close to a 2-minute default command timeout.
- **Recommendation.**
  - Run detached (`nohup … &` or the tool's background mode), with stdout/stderr to a log in the launcher's directory and the exit code to a file. Do not interrupt it.
  - Optional: a SIGTERM handler that raises `KeyboardInterrupt`, so that a kill is at least recorded as an incident.

**N3. Nothing else may write in the repository during the run.**
- **The check.** The post-run tree check (`:501–505`) requires `git status --porcelain --untracked-files=all` to list exactly the three untracked stage-1 files. That is right (lock review N8), and I verified the three paths are not gitignored.
- **The risk.** Any other non-ignored file appearing during the run fails the check after a complete stage 1: exit 1, no "Stage 1 complete" line, and a judgment call left to the user. Examples:
  - `.claude/` exists at the repository root and is **not ignored** (`git check-ignore .claude/settings.local.json` matches nothing);
  - an editor's swap file.

  `.DS_Store` is ignored.
- **If it happens.** Nothing is spent. The record's other checks and `tree_after` let the user decide.

**N4. Environment and identity at production (procedure).**
- **`HF_HUB_OFFLINE=1`.** The launcher records it (`:247`) but does not require it, unlike `-B` (`:479–480`). The plan's environment requires it, and the calibrate and lock production runs had it. The lock's parameter and embedding digests (`run.py:676`) protect the science either way.
- **Launcher sha256.** Check `shasum -a 256 launch_stage1.py` = `13b1992f…` immediately before the run, as `launcher_sha256_at_production.txt` did for calibrate and lock. The sentinel then records it.
- **`STAGE1101_LOCK`.** It must be the lock's **content** sha256 `8da97e35556d0baa4739899cb841e86cbd59cced21611c732fdd9d8c9edd9061`, not the file sha256 `8dc6e6d3…`. A mistake is refused before the sentinel.
- **Optional: `GIT_OPTIONAL_LOCKS=0`.** It makes the runner's plain `git status` (lock review N10) take no optional lock. My rehearsal ran with it and passed.

**N5. The residual of the lock review's S3(a): a refusal inside `stage1()` before its ledger leaves the sentinel.**
- **Where it comes from.** The sentinel is now created immediately before `stage1()` (`:493–495`), as the lock review asked.
- **What can still refuse.** After that point and before the ledger, the runner can still refuse on:
  - a model-load failure (`run.py:673`);
  - the model's dependencies (`:676`);
  - the nouns (`:677`);
  - the stage-1 binding (`:679`);
  - `write_check` (`:686`).

  None is expected: the preflight checks the checkpoint files' sha256s, the versions and the lock. The runner's paths, the lock's bound paths and `STAGE1_FILES` are identical (I checked; the preflight does not assert it, N9).
- **What happens.** My scenario c shows the launcher records the error and exits 1. A retry needs the sentinel moved aside by hand.
- **The recovery.**
  1. Confirm `phases.stage1 == {"status": "not_started"}`, an empty ledger and no stage-1 file. An empty `stage1/` directory may remain from `run.py:685`.
  2. Fix the cause.
  3. Rename the sentinel, keeping it as evidence.
  4. Rehearse, then run.
- **Not retryable.** A G1 or digest mismatch (`run.py:680–684`) is a phase incident by design (plan, `stage1` step 2).

**N6. Repository writes the hook allows beyond the allow-list, or cannot see.** None is reachable from stage 1's code. Each
needs adversarial code, or a library behaving unlike the recorded runs.
- **Allowed by design.**
  - Any `.results-*.json` in `outputs/experiment-101/`: write, remove, and rename onto `results.json`.
  - Direct writes of `results.json`.
  - Anything under `.venv/` (`:99`).
- **Not judged** (demonstrated on the mirror, `out/c_real_ops_production.json`, probes):
  - `torch.save` and safetensors to a path: file I/O in C++/Rust raises no `open` event;
  - SQLite databases;
  - `os.mkfifo`, which raises no audit event at all;
  - `os.utime`, `os.chflags`, `os.setxattr`, `os.chown` (unhandled events);
  - child processes: `subprocess.Popen` is not judged. Stage 1 runs only `git`, but the runner's plain `git status` may refresh `.git/index` (lock review N10).
- **Misjudged.**
  - **`open` with a relative name and a `dir_fd`.** Python's `open` audit event carries no `dir_fd`, so the name is judged from ROOT. Demonstrated: `os.open("outputs/experiment-101/results.json", O_WRONLY|O_CREAT, dir_fd=<src/>)` was allowed and wrote `src/outputs/experiment-101/results.json` in the mirror. If desired, the fix is to refuse every writing `open` with a relative path; stage 1 writes only absolute paths.
  - **A symlink's own entry.** `os.remove`, `os.rename` and `os.link` are judged by the link's target, because `_path` resolves the final component (demonstrated). No tracked symlink exists; the only ones outside `.venv/` are the gitignored `.worktrees/*/.venv/bin/python*`.
  - **Aliases of the repository.** `/System/Volumes/Data/Users/…/neural-decompiler` (a firmlink) and case variants (this APFS volume is case-insensitive) reach the same inode, but fail the string-prefix test (`:94–99`). An inode-based check would close this.
  - **`O_RDONLY|O_TRUNC`.** It is classified as a read (`:121`), but truncates on this macOS.
- **Backstop.** The post-run tree check catches all of these for non-ignored paths. It cannot see ignored paths (`outputs/experiment-*/`, `.venv/`, `.worktrees/`, `__pycache__/`, `*.py[cod]`) or metadata.

**N7. Completeness of the record (cosmetic).**
- **Logs on an exception.** When an exception escapes `stage1()` (an interruption), `record["logs"]` (`:510`) is skipped, so the runner's "INCIDENT (stage1)" line exists only on the console (scenario d). On the return-2 path (scenario b) it is in the record.
- **`output_writes` and `output_renames`** (`:63`) are never filled in v2 (version 1 filled them), so the record does not count the state writes.
- **`post_ok`** (`:524–526`) does not require `events["refused"]`, `["forbidden_writes"]` or `["forbidden_reads"]` to be empty, so a refused event that some code swallowed would not fail the verdict.
- **`tree_after`** is computed outside the `try` (`:513–514`), so a failing `git` there would lose the record.
- **The sentinel and the record** are created with `os.open`'s default mode (0755; `:493`, `:519`), with one `os.write` and no `fsync`.
- **Docstring.** "one rename each" (`:27–28`) is not enforced by the hook. "Version 2 has not been re-reviewed" (`:9`) is now stale.

**N8. The self-test is decision-level** (`:173–206`, run in every mode before the sentinel, `:485–487`).
- **What it does.** It calls `decide()` on hand-transcribed tuples, not the audit hook on real operations.
- **How faithful the tuples are.**
  - They match Python 3.12.13's real events: the arguments are `str`, as `os` and `io` pass them after `__fspath__`.
  - The `mkstemp` tuple lacks `O_NOFOLLOW` (real flags 16780034), which does not change the decision.
  - The `fdopen` open, the directory-`fsync` open and the state reads are omitted.
- **Conclusion.** My real-hook mirror test covers all of these, so the lock review's S1 is fixed in substance and as that review recommended.

**N9. Minor robustness.**
- **Stage-1 paths.** The preflight does not assert that `runner.stage1_paths()` equals `STAGE1_FILES`. They are equal, and equal to the lock's bound `stage1.paths`, so a mismatch is not possible now.
- **Concurrent launches.** Two production launchers started concurrently from **different** directories would both pass the preflight. Each has its own sentinel, and the runner's phase rule has a window between `_state_for` (`run.py:664`) and the ledger write (`:690`). That needs an operator error. A fixed global lock path would close it.

## The brief's checks, one by one

### 1. S1–S3 and N8 are fixed; no regression

- **S1, the write path under the hook in rehearsal: fixed.**
  - `decide()` is the hook's pure decision. `_hook` (`:155–170`) adds only bookkeeping and the `PermissionError`.
  - `hook_self_test()` decides the recorded stage-1 tuples under production rules, with 10 refused controls and the rehearsal-refuses check. It gates the run before the sentinel.
  - The docstring (`:18–20`) now describes this accurately.
  - My mirror test shows the real hook allowing the real operations (N8).
- **S2, fail closed and `dir_fd`: fixed.**
  - `_path` (`:71–91`) treats only an integer `dir_fd ≥ 0` (not a bool) as a descriptor, resolved with `F_GETPATH`. `-1`, `None` and other negatives use the working directory, which is the right semantics: Python reports `AT_FDCWD` as `-1`.
  - An unresolvable name becomes `UNRESOLVED`, which `_in_repo` counts as inside, so every write check refuses it. Missing arguments also become `UNRESOLVED` (`:133`, `:143–145`).
  - Battery: relative `mkdir`, `remove`, `rmdir`, `chmod`, `rename`, `link`, `symlink`, `rmtree` and `truncate` are all refused. `dir_fd` ≥ 0 resolves both inside and outside the repository, and a closed descriptor number is refused. A huge `dir_fd` raises `OverflowError` inside the hook, which also fails the operation.
- **S3, sentinel and records: fixed.**
  - The sentinel is created after the preflight, the second sweep and the self-test, immediately before `stage1()` (`:493–498`).
  - Each record is a new O_EXCL file with a microsecond stamp (`:518–521`). Scenario l keeps two earlier records, and scenario i does not overwrite an existing sentinel.
  - For the residual, see N5.
- **N8 of the lock review: fixed.**
  - Rehearsal allows no repository write: `write_allowed` returns False unless in production (`:106–110`); 41/41 cases refused.
  - `-B` is required (`:479–480`).
  - The ledger is asserted (`:504`), and so is the tree (`:501–505`).
  - All of these are in `post_ok` and the exit code (`:524–529`).
- **No regression.** The sanctioned route is unchanged (`:289–314`):
  - `ri.capture_reference` runs only under tag `S1`;
  - only for a key among the preflight's 36 `frame|ref|581|S1` keys, each once;
  - only after `_ledger_on_disk()` sees exactly those keys with `stage1` `running`, which matches `run.py:688–690`.

  The rest is also unchanged:
  - `ri.run_arm` refuses (`:317–318`);
  - the five low-level entry points (by identity and aliases) and `Module.__call__` refuse outside a sanctioned call (`:265–286`, `:321–336`);
  - every module instance and model-class method is sealed (`:339–382`);
  - a second load is refused (`:389–395`).

  A target prompt's key carries `two`, `three` or `many` and never matches, and `ri.capture_reference` refuses any prompt but `ref` (`read_intervention.py:1138–1142`).

### 2. The hook

- **Fail closed.** Every handled write-type event refuses an unresolvable path. The one exception is `open`, whose audit event carries no `dir_fd`: a relative name is judged from ROOT, which fails closed except when the name happens to name an allowed file from ROOT (N6).
- **Events a correct run raises, and their decisions.** All are allowed in production rules; all repository writes are refused in rehearsal rules.

  | step | events (Python 3.12.13, as raised) | evidence |
  |---|---|---|
  | imports, preflight | reads; `open('/dev/null','r+')` by dill; tempfile's probe in `$TMPDIR` | rehearsal (real hook); d2/f live judge |
  | `seed_runtime` | `os.mkdir($TMPDIR/torchinductor_<user>)` | d2, f |
  | model load | filelock's probe in a `$TMPDIR` temporary directory (`mkdtemp`, `open`, `utime`, `link`, `rmtree` → `remove(name, dir_fd=3)` ×2, `rmdir`) | d2, f |
  | programs, G1, table, states blocks | none | f |
  | 36 captures | none | d2 (two forwards with stage 1's plain site set; `capture_frame_020` adds sites to the same single forward) |
  | `stage1/` mkdir (runner `:685`; again in `write_table`) | `os.mkdir(stage1, 511, -1)` ×2 | c |
  | the ledger and the final state | `os.mkdir(outputs/experiment-101, 511, -1)`, `tempfile.mkstemp`, `open(tmp, None, 16780034)`, `open(fd, 'w')`, `os.rename(tmp, results.json, -1, -1)`, `open(outputs dir, None, O_RDONLY)` | c |
  | `write_table`, `table.json` | `open(states.f64|states.json|table.json, 'w', 16778753)` | c |
  | `file_sha256`, `load_results_state` | `open(…, 'r', 16777216)` | c |
  | an incident's recording | the same state writes; `_recheck` reads only what `_base` reads (020's results under `outputs/`) | reading; rehearsal |

  **No event of a correct run is refused.**
- **Repository writes allowed besides the three files and the results state.** Only those listed in N6.

### 3. The preflight

- **It refuses before the sentinel, with nothing recorded in the repository,** on each of these:
  - **Identity and environment:** a mistyped token (`:433–434`); an environment drift against the lock run's record (`:423–428`: Python, executable, platform, packages, threads, model, checkpoint files' sha256s); a missing `lock_run.json` (raises inside the `try`).
  - **Repository state:** a dirty tree or a wrong git state (`:420–422`); an uncommitted lock or preregistration (`:431–432`).
  - **The lock and the code:** a mismatched module blob or any pinned blob (`:435–438`); the full `_validated_lock` (`:454–459`).
  - **The phase's own state:** stage 1 already started, or a non-empty ledger (`:460`); an existing stage-1 file (`:461`); a runtime or version drift against 020 (`:462`).
  - **The keys and the predicate:** a wrong or forbidden S1 key set (`:463–465`); the write predicate (`:466–471`).
  - **The run itself:** a missing `-B` (`:479–480`), and the self-test (`:485–487`).
- **Evidence.** Scenarios j and k (stub) show no sentinel and exit 1.
- **In the real state.** It passes: my rehearsal shows exit 0 and 16/16 checks, identical to the operator's except for timestamps and the launcher path.

### 4. The production path

- **Sentinel and invocation.** The sentinel comes after the preflight, and `stage1()` is called once (`:493–498`).
- **`write_check` is the launcher's predicate** (`:452`, `:402–408`). It accepts the runner's three paths in production and refuses in rehearsal, after files exist, or for any other path (mirror test).
- **Post-run checks** (`:499–509`): the 36 keys once, the ledger exactly the 36 S1 keys, the tree exactly the three untracked stage-1 files, the record's digest (`b0c.stage_one_digest` excludes `digest`), and the files' sha256s against the record.
- **The record** is written with O_EXCL (`:519`).
- **The exit code** is 0 only in production when `status == 0`, no error and all five checks pass (`:524–529`).
- **Executed scenarios** (`out/e_main_block.json`):

  | scenario | exit | sentinel | notes |
  |---|---|---|---|
  | a correct run | 0 | yes | all five checks true; "Stage 1 complete" printed |
  | b incident after the ledger (stage1 returns 2) | 2 | yes | record written with the logs; tree, digest and files checks false |
  | c refusal inside stage1 before the ledger | 1 | yes | error recorded (N5) |
  | d interruption after the ledger | 1 | yes | error recorded; no post and no logs in the record (N7) |
  | e a stray file in the tree | 1 | yes | tree check false |
  | f record digest tampered | 1 | yes | digest check false |
  | g a file changed after the record | 1 | yes | files check false |
  | h complete, then incident | 2 | yes | **all five checks true** (N1) |
  | i sentinel already exists | 1 | kept, not overwritten | `stage1` not called |
  | j preflight refuses | 1 | no | — |
  | k self-test fails | 1 | no | — |
  | l earlier records present | 0 | yes | earlier records kept |

- **If `stage1()` raises after the ledger.** For an `Exception`, the runner records a phase incident with `after_ledger: true` and **returns 2** rather than raising. The launcher then records `exit_status: 2`, the runner's "INCIDENT (stage1): …" log line and failing post checks, and exits 2. For an interruption the runner records and re-raises, and the launcher records the error and exits 1. Either way, the exit code is non-zero. The gaps are N1 and N7.

### 5. Anything else

- **A target (S2) prompt.** None can run: no route reaches a forward except the sanctioned `S1` capture (check 1).
- **A file other than the three.** None can be written by stage 1's code: the hook plus the tree check, with the adversarial-only gaps of N6.
- **The phase twice.** It cannot run twice: the O_EXCL sentinel, the runner's phase rule (`read_intervention.py:2073–2076`), and the preflight's empty-ledger and no-file checks. The only window is concurrent launches from different directories (N9).
- **A correct run failing.**
  - Not through the hook or the guards.
  - The guard machinery is the calibrate launcher's, proven on 36 `ri.capture_reference` calls.
  - `seed_runtime` uses `deterministic_algorithms=False`, as calibrate did.
  - The remaining risks are operational: N2 (a kill after the ledger) and N3 (a stray file failing the tree check).

## Files (this directory)

| file | sha256 | what |
|---|---|---|
| `PROMPT.md` | — | the brief |
| `REVIEW.md` | — | this report |
| `scripts/common.py` | `20f0d5b9…` | my guard (refuses any write outside this directory; edited once, after `b` and `c` ran, to allow `/dev/null`, which dill opens `r+` at import) and the `ast` extractor |
| `scripts/b_decide_battery.py` → `out/b_decide_battery.json`, `.log` | `21521244…` → `0b891f21…`, `1efc50f0…` | check 2: 70 cases, probes, edges, the self-test on the mirror |
| `scripts/c_real_ops.py` → `out/c_real_ops_{production,rehearsal}.json`, `.log` | `4cf8ea7d…` → `6506d3c5…`, `9e713d84…` | the real hook on the real write functions (mirror) |
| `scripts/d_load_forward_events.py` → `out/d_load_forward_events.json`, `.log` | `6f5de530…` → `898f0860…` | the first recording (after-the-fact judging; superseded) |
| `scripts/d2_load_forward_live.py` → `out/d2_load_forward_live.json`, `.log` | `ed6e001d…` → `58668def…`, `a8747f69…` | load and forwards, judged live: 0 refusals |
| `scripts/f_program_events.py` → `out/f_program_events.json`, `.log` | `889e049a…` → `7bf4f34e…`, `29896d82…` | G1, table, states blocks, judged live: 0 refusals; table `ce055e09…` reproduced |
| `scripts/e_main_block.py` → `out/e_main_block.json`, `.log` | `d2deadf8…` → `575a979d…`, `7eab1929…` | the main block in 12 scenarios |
| `scripts/g_platform_probes.py` → `out/g_platform_probes.log` | `90ed476f…` → `2e454be3…` | `O_TRUNC` and path aliases |
| `rehearsal/launch_stage1.py`, `rehearsal.log`, `rehearsal.exit`, `rehearsal_run_20261014T124825080100Z.json` | `13b1992f…`, `fa9860ca…`, `28d3b9e8…`, `b8790628…` | my rehearsal from a copy |
| `out/repo_snapshot_start.txt`, `out/repo_snapshot_end.txt` | `f4fdc79e…`, `487c4255…` | the repository before and after: identical apart from the timestamp line |

Scratch trees: `mirror*/`, `main_block/`, `tmpdir*/` and `work_*/`. These are the mirror repositories, the stub-run git
repositories, the scratch `TMPDIR` directories and the working directories, all inside this directory.
