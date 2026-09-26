# Experiment 024 — evidence index

This folder holds the run record and the independent review of every phase, and the closure data.
- The scripts ran from the session's scratch directory, and the paths inside them refer to it. They are kept here
  exactly as written and run, not adapted to this location.
- Every script is read-only with respect to the repository, except each phase's own launcher, which wrote only that
  phase's outputs.
- Every phase ran exactly once. Each record was reviewed by a separate agent with its own code.

| folder or file | what | outcome |
|---|---|---|
| [`freeze-2026-09-25/`](freeze-2026-09-25/README.md) | the guarded launcher around the stock `freeze` (tokenizer only), its run record, the read-only verification, the post-commit checks | freeze ran once at `44c3c2b`, exit 0; the design's 40 cues; `confirmation-v1.json` installed byte-identically in `566eb6f` (file `68510e1b…`, content `87f8aff1…`); manifest 4,320 keys `fcc437fc…` |
| [`freeze-review-2026-09-25/`](freeze-review-2026-09-25/REVIEW.md) | the independent freeze review | PASS WITH NOTES, no blockers |
| [`calibrate-2026-09-25/`](calibrate-2026-09-25/README.md) | the pre-calibrate check, the guarded launcher (1 weights load, 0 module calls, 0 captures), its run record, the verification, the post-install checks | calibrate ran once at `bf0049c`, exit 0; `calibration-v1.json` installed byte-identically in `1084efc` (file `81fb499e…`, content `09bf093e…`); F_ρ `0.24411074612857814`, null₉₇.₅ `0.3136960600375234`, the null binds |
| [`calibration-review-2026-09-25/`](calibration-review-2026-09-25/REVIEW.md) | the independent calibration review | PASS WITH NOTES, no blockers |
| [`lock-2026-09-25/`](lock-2026-09-25/README.md) | the pre-lock check, the guarded launcher and its run record (`lock_run.json` `b7a51118…`), the verification with an I7 rehearsal, the pre-install check, the install transcript, the post-commit checks, `SHA256SUMS` | lock ran once at `be74d23`, exit 0; installed byte-identically in `3affe55` (`preregistration-lock.json` `5c2a9b90…` / content `a39668bf…`; `preregistration.md` `007c9e6c…`); analysis module blob `e6cb3776…`; 40-score digest `a703ac16…` |
| [`lock-review-2026-09-25/`](lock-review-2026-09-25/REVIEW.md) | the independent lock review | PASS WITH NOTES, no blockers; the direct module-blob check was made mandatory for the confirm launcher |
| [`postinstall-review-2026-09-25/`](postinstall-review-2026-09-25/REVIEW.md) | the independent post-install verification at `af160ce`: validate, the full `validate_lock`, the module blob, the I7 rehearsal, the manifest, eligibility; `SHA256SUMS` | PASS WITH NOTES, no blockers; confirm eligible |
| [`confirm-2026-09-25/`](confirm-2026-09-25/README.md) | the pre-confirm check; the guarded launcher (sentinel, the direct module-blob assertion before the load, environment, prompt log); the dry run; the official run's log, record and prompt log; the post-confirm verification; `SHA256SUMS` | confirm ran **once** at `af160ce`, exit 0, no incident; 4,320 of 4,320 keys once each; stage-2 `b21babe1…` (76,314,087 bytes) |
| [`confirmation-review-2026-09-25/`](confirmation-review-2026-09-25/REVIEW.md) | the independent confirmation review: the result re-derived from the saved measurements with its own code; `SHA256SUMS` | PASS WITH NOTES, no blockers; ρ = 1628/2665, K = 1 reproduced |
| [`report-2026-09-25/`](report-2026-09-25/README.md) | the check-only run, the guarded report launcher (no model, no prompt), the official run's log and record, the post-report verification; `SHA256SUMS` | report ran **once** at `b80409b`, exit 0, no incident; `8234ed9b…`; final state `20ac6095…` |
| [`final-report-2026-09-25.md`](final-report-2026-09-25.md) | `outputs/experiment-024/report.md`, byte-identical, not re-rendered or edited | `8234ed9b41533c5aa90172cfb1bc0b70c656848eaeff234e9dcfc9a5d09a8193` (8,310 bytes) |
| [`confirmation-record-2026-09-25.json`](confirmation-record-2026-09-25.json), [`closure-2026-09-25/`](closure-2026-09-25/) | the confirmation-record extract of the gitignored final state (022/023 convention: the ledger replaced by digests; the prompt accounting and the official result summary copied from the state), its builder, the pre-closure check; `SHA256SUMS` | content `fbe0c68995ce926c5360c2098f8556fb2e70df32919b727ae4bc50cba80ebed3`, file `131222a7d425a676be347b1cf9aad4713d3b04a405bcf99588ebf2b5059268b0` |

## The provenance chain (immutable hashes)

| step | artifact | hash |
|---|---|---|
| design, plan | revision 2 `9d03dee`; plan revision 1 `608088c` | — |
| freeze | `confirmation-v1.json` | file `68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60`, content `87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87` |
| manifest | 40 cues × 108 frames = 4,320 keys | `fcc437fca9730b8599333eeac95807786570f0d03bd01e49f52daec76ed4e32d` |
| calibration | `calibration-v1.json` | file `81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d`, content `09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8` |
| lock | `preregistration-lock.json` | file `5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d`, content `a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739` |
| preregistration | `preregistration.md` | `007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54` |
| analysis module | `src/neural_decompiler/readout_routing.py` | git blob `e6cb37767d1d06c6ff40804a88eab569723afdb5` |
| model | pythia-70m-deduped `e93a9faa…` | checkpoint `3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd`; parameters `fd953f1c…`; embedding `9cd6f39b…` |
| the 40 frozen scores | the lock's `fresh.scores_sha256` | `a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995` |
| fresh measurements (gitignored) | `outputs/experiment-024/stage2-measurements.pt` | `b21babe189d1cccb9ae359cfabf74baaa3a8337cac04612bc0d4aaa232b4dc93` (76,314,087 bytes) |
| state at the result write | — | `37f2f88b7f0fdde73ebbedc319fc5d9edcab59c35133a8cfa8b24c4e3201f9fd` |
| confirm-final state | — | `076ab9f984f3ba6244b4df767d8c8fd7ebd40a634521a46a3b881d7da1604e92` (file `e636210c…`) |
| report | `final-report-2026-09-25.md` | `8234ed9b41533c5aa90172cfb1bc0b70c656848eaeff234e9dcfc9a5d09a8193` |
| final state (gitignored) | `outputs/experiment-024/results.json` | `20ac6095a7e8534ff933a6405ae74b5f707aafd2ec9ac42e0ef5ffb1c5267376` (file `edbb6dc9…`) |
| confirmation-record extract | `confirmation-record-2026-09-25.json` | content `fbe0c689…`, file `131222a7…` |

Each state digest is `sha256(canonical_json(state without state_sha256))`, as `rr.write_state_atomic` writes it.
Reverting only the report entries of the final state gives the confirm-final state. Removing only the descriptive
records from that gives the state at the result write.
