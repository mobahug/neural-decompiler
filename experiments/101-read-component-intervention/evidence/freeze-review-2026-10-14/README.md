# Experiment 101 — the independent freeze review (2026-10-14)

**An AI review (independent, read-only), not a human validation.** A separate Claude agent verified the production
freeze from the brief `PROMPT.md`, with no access to the operator's reasoning beyond the brief and the files.

**Verdict: PASS WITH NOTES — no blocker** (`REVIEW.md`, sha256
`f560b51a8b1ed2e6779ecdbec6f73cf68c0854282fce6ea464c455b8ae0ad17f`).
- Its own derivation, which imports no `neural_decompiler` code, reproduces the 60 eligibility outcomes, the 36 frames,
  every exclusion count and digest, both manifests and the content digest; `ri.freeze_payload`, run afterwards under
  guards, regenerates the file byte for byte. Two tokenizer implementations agree, and 3,132 committed sequences of
  020, 022 and 023 re-tokenize with no mismatch.
- The spent set is 47,952 keys, equal to the runner's `base.forbidden`, with 0 collisions.
- It loaded no model, opened no weight file, ran no forward pass and read nothing under `outputs/` but 020's results.

## The findings and what was done

| finding | disposition |
|---|---|
| SF1 — preserve the freeze evidence in the repository | done: `../freeze-2026-10-14/` holds the launcher, both run records and logs, the checks and the test-gate logs, verified against the review's sha256s. The first dry run's launcher was not preserved; a reconstruction is recorded as such. The draft README's two errors it named ("7 commits ahead"; unfilled placeholders) are corrected. |
| N1 — the hook's argument mapping (modes and lengths read as descriptors; `os.open`'s `dir_fd`) | applied to the calibrate and lock launchers: `dir_fd` per event (`os.mkdir` and `os.chmod` take it third; `os.remove`, `os.rmdir`, `shutil.rmtree` second; `os.truncate` has none; rename and link resolve source and destination separately). The `open` audit event carries no `dir_fd`, so a relative name opened through one is resolved against the working directory (the repository root); the later launchers record every refusal, and their smoke and rehearsal modes run before production. |
| N2 — pure helpers not refused | the later phases compute with them on their own data, so a launcher refuses them only where its phase does not need them (protocol note 3 in the experiment README). |
| N3 — the launcher's checks lean partly on the module | closed by this review's independent derivation. |
| N4 — provenance anchoring | the local commits (`1e48c1c` … `40829d2`, `a857f9a` and later) are not pushed; the user pushes. No commit on `exp101-phases` is rewritten. |
| N5, N6, N10, N11 | recorded; no action. |
| N7 — the unguarded smoke reference on the spent exposed frame `coordinated-adjective-1` | recorded in the calibrate evidence and its review brief. |
| N8 — gate logs without HEAD and time | later gate logs record both. |
| N9 — stale experiment README | updated in the same commit as this directory. |
| N12 — act on the hand-back, not a draft | **process lapse, recorded:** the install commit `a857f9a` (11:40:09 local) was made after `REVIEW.md` stated its verdict but before the reviewer had handed back. The verdict did not change in the final report. Later phases wait for the hand-back. |

## Files

| file | what |
|---|---|
| `PROMPT.md` | the brief |
| `REVIEW.md` | the report |
| `independent_derivation.py/.json` | the derivation without `neural_decompiler` |
| `compare_with_file.py/.json` | the field-by-field comparison |
| `spent_set.py/.json` | the spent set and the collisions |
| `cross_check_ri.py/.json` | the module and runner cross-check, under guards |
| `cross_check_pool.py`, `tokenizer_integrity.py`, `ls_tree_src.txt` | the pool-derived sets, the tokenizer check, the module tree |
| `SHA256SUMS` | this directory's digests |
