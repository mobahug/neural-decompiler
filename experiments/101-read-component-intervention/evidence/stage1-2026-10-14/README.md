# Experiment 101 — stage 1 (2026-10-14)

`stage1` ran **exactly once**, at commit `961e81408e5c4da4fffdf9dfab0343e449a4e1f2` on `exp101-phases` (the user's lock
commit), under the guarded stage-1 launcher version 2, on the user's authorization of 2026-10-14 ("run it after the
re-check"). It ran only the 36 reference prompts `frame|ref|581|S1` of the frozen frames, sealed the model, computed the
program's prediction table from the reference states and the weights, and wrote the three stage-1 files, which **the
user commits** before confirm (the prediction freeze; design §5 and §14 D6):

| file | sha256 |
|---|---|
| `stage1/states.f64` | `1c7e51c049e1c61cf669f1783736054956073dc5af6947354b1d13c058ac5b66` |
| `stage1/states.json` | `2b2c3a61b4a2ce05eba0dcf875c19bfdb1c2627da1ccb1e8d49710af95a873ca` |
| `stage1/table.json` | `71d339cd1d7151ad57949384388679593264ac07dec8c9c944c121d9918795fa` (table content sha256 `9191800440af37909d2068459e360d7772191fc66bf2ad5fb9cf2c133246ac86`) |

The independent stage-1 review (`../stage1-review-2026-10-14/`, an AI review, not a human validation): **PASS WITH NOTES — no blocker, no should-fix; the three files can be committed as they are**.

## The launcher (`launch_stage1.py`, version 2, sha256 `13b1992f7f0feb652207ca835df044244f20faea750e92f42d8d51bced7c3e1c`)

Version 1 was read by the lock review (S1–S3, N8); version 2 applies them, and an independent re-check
(`../stage1-launcher-recheck-2026-10-14/`) found it **READY FOR PRODUCTION**, with no blocker and no should-fix. It ran
unchanged (the sha256 was checked immediately before the run: `launcher_sha256_at_production.txt`).

- **The rehearsal** (`rehearsal_run_20261014T120823753435Z.json`, `rehearsal.log`; 12:08:14–12:08:23 UTC): no model, no
  forward, no write; 16/16 preflight checks true, among them the full `_validated_lock` against the committed lock, the
  environment equal to the lock run's, the module binding, stage 1 not started with an empty ledger, no stage-1 file
  yet, the runtime, the 36 S1 keys; the hook's self-test under the production rules (9 allowed, 10 refused, as
  expected).
- **The production run** (`production_run_20261014T130926706170Z.json`, `production.log`, `production.exit`,
  `production_runs.jsonl`, `production_started_utc.txt`, `stage1.sentinel`): launched detached with `nohup` (the
  re-check's N2), with `STAGE1101_LOCK` = the lock's content sha256, `HF_HUB_OFFLINE=1`, `-B`, `GIT_OPTIONAL_LOCKS=0`;
  13:08:48–13:09:26 UTC, **exit 0**:
  - **stage-1 invocations 1, model loads 1, refusals 0**, forbidden reads or writes 0;
  - **sanctioned runs 36**, each one S1 key once, only through `ri.capture_reference`, after the S1 ledger was on disk;
    12,240 sanctioned module calls (36 × 340, one forward per run, as in calibrate);
  - repository writes: the three stage-1 files and the `stage1/` mkdir, and the results state;
  - the post-run checks, all true: the 36 keys once; the ledger exactly the 36 S1 keys; the tree exactly the three
    untracked stage-1 files; the stage-1 record's digest reproduces; the files' sha256s equal the record's.
- **The operator's own check after the run** (the re-check's N1): `phases.stage1` is `complete` with `awaiting_commit`
  and **no incident entry**; confirm `not_started`; the results state `aecd76fc…`.

## Files

| file | what |
|---|---|
| `launch_stage1.py`, `launcher_sha256_at_production.txt` | the launcher (version 2) and its sha256 at production |
| `rehearsal_run_20261014T120823753435Z.json`, `rehearsal.log` | the rehearsal |
| `production_run_20261014T130926706170Z.json`, `production.log`, `production.exit`, `production_started_utc.txt` | the production run |
| `production_runs.jsonl` | one line per sanctioned run (36), at admission |
| `stage1.sentinel` | the production sentinel |
| `SHA256SUMS` | this directory's digests |
