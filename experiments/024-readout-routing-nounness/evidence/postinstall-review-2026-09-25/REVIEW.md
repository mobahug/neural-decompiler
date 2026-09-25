# Independent post-install verification — 2026-09-25

**Verdict: PASS WITH NOTES, no blockers.** A separate agent verified the installed lock read-only, at `af160ce`. That
commit came after the lock artifact commit `3affe55` and the lock evidence commit.
- **Result:** the installed lock is verified, and confirm was eligible to be authorized. All 10 items passed.
- **Method:** it wrote its own code for every check, and called the canonical module only where canonical behaviour or
  byte identity was the thing being checked.
- **Guard:** every script ran under `guard.py`, an audit hook refusing any repository write. It allowed only read-only
  git subprocesses, and self-tested 18 cases before each run. It refused 0 writes.
- **Afterwards:** the tree stayed clean, HEAD, origin and the remote stayed `af160ce`, and the five output files kept
  their hashes and modification times.

| item | result |
|---|---|
| 1. Git state (33/33, `s01`) | HEAD, origin and `ls-remote` are `af160ce`; the tree is clean. `3affe55` (parent `be74d23`) adds exactly `preregistration-lock.json` and `preregistration.md`. `af160ce` (parent `3affe55`) touches 30 non-scientific paths. `rr.scientific_changes` over `be74d23..HEAD` (32 paths) is empty. |
| 2. Installed bytes (34/34, `s02`) | The working tree, `git show HEAD:`, `git show 3affe55:` and the candidates are identical. The lock is `5c2a9b90…` (32,729 bytes), with its own content digest `a39668bf…`; the preregistration is `007c9e6c…` (9,558 bytes) and equals `render_preregistration(lock)` byte for byte. |
| 3. Stock `validate` (12/12, `s03`) | Returned 0 with both loaders refusing and never called, and reported "lock and preregistration verified". |
| 4. Full `validate_lock` (27/27, `s04`, `s04a`) | A static check found no write reachable from the helpers. `run.py` 510–526 was replicated statement by statement, and `validate_lock` accepted the lock. Tracked-file enforcement is live, and 10 further refusal branches fired. The runtime check passed. |
| 5. Module blob (12/12, `s05`) | `e6cb37767d1d06c6ff40804a88eab569723afdb5` by 4 routes: its own sha1, `rr.own_blob()`, `git rev-parse HEAD:…` and the lock's `module.blob`. `rr` was imported from the repository file. All 12 pinned blobs match. |
| 6. I7 rehearsal (30/30 + 6/6, `s06`, `s12`) | Confirm's pre-prompt block was replicated with a sealed load: 1 load, 0 module calls during or after it, 0 forward-stub hits, 0 capture hits. `bitwise_equal` True; scores `a703ac16…`. A supplement sealed before the load was also bitwise equal. Its own score routes agree within 2.22e-16, with identical ordering. |
| 7. Manifest and ledger (17/17, `s07`) | 4,320 unique keys, `fcc437fc…`. The spent keys are 30,132 + 3,060 + 3,060 + 3,060 = 39,312, with 0 overlap. Both ledgers are empty. |
| 8. Phase eligibility (22/22, `s08`) | The state digest `58891c24…` recomputes; `assert_phase_allowed("confirm")` passes; rerunning calibrate or lock is refused. |
| 9. No fresh prompt or outcome (20/20, `s09`) | The outputs held exactly the five pre-confirm files, and the state's confirmation and report were null. Among 524 tracked files, only the freeze carries fresh keys. |
| 10. README and evidence (82/82 + rules 14/14, `s10`, `s11`) | Every `SHA256SUMS` entry matches, and the archived files are byte-identical to their originals. Every hash the READMEs state is verified. |

## Notes (non-blocking)
1. **The confirm launcher must implement the direct blob assertion** (`rr.own_blob() == installed_lock["module"]["blob"]`
   before the load, with the expected and observed values recorded). It did; see `../confirm-2026-09-25/`.
2. **Confirm must run in the same environment**: 4 torch threads, the same package versions (no `uv sync` or `uv run`)
   and the same cached checkpoint. Any drift is refused before a prompt.
3. **`collect_git_state`'s `git status` may refresh the index's stat data.** That is not a content change;
   `GIT_OPTIONAL_LOCKS=0` avoids it.
4. **Two files had no scratch original:** `REVIEW.md`, which is the reviewer's report with decisions appended, and
   `SHA256SUMS`, which excludes `README.md` by design.
5. **Transparency about its own process.** One early snippet ran before its guard existed, and it read only. Four check
   bugs of its own were fixed and re-run; none was a finding.

The files here (`guard.py`, then `s01`–`s12` and `s04a`, each a `.py` with its `.out`; 27 files) are the reviewer's, as
written and run. The instruction counted 28; 27 is the actual number. `SHA256SUMS` lists them.
