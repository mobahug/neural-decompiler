# Experiment 101 — evidence

**Status (2026-10-14):** the freeze and the calibration ran once and are installed; the lock ran once and its candidates await the user's commit ([DECISION.md](lock-2026-10-14/DECISION.md)).

| directory | what |
|---|---|
| [`freeze-2026-10-14/`](freeze-2026-10-14/README.md) | the test gate at `40829d2` (A 578, C 645 passed; 0 failed), the guarded production freeze (one dry run kept, then the one production run), the pre-install and post-commit checks (13/13) |
| [`freeze-review-2026-10-14/`](freeze-review-2026-10-14/README.md) | the independent AI freeze review: **PASS WITH NOTES — no blocker**, and the disposition of each finding |
| [`calibrate-2026-10-14/`](calibrate-2026-10-14/README.md) | the guarded calibration (rehearsal, smoke, the one production run at `8f65062`: 2,016 spent keys, exit 0, no trigger), the post-install checks (15/15), the state extract |
| [`calibration-review-2026-10-14/`](calibration-review-2026-10-14/README.md) | the independent AI calibration review: **PASS WITH NOTES — no blocker**, the Level-1 risk estimate, and the disposition of each finding |
| [`lock-2026-10-14/`](lock-2026-10-14/README.md) | the guarded lock (one dry run, the one production run at `3621eec`: exit 0, 0 module calls), **`DECISION.md`** for the user, and the prepared stage-1 launcher (not run) |
| [`lock-review-2026-10-14/`](lock-review-2026-10-14/README.md) | the independent AI lock review: **PASS WITH NOTES — no blocker to committing the lock**, and the disposition of each finding |

The freeze artifact `../confirmation-v1.json` was installed byte-identically in
`a857f9a2326ff61cae8485deef2ab9337f5d6e02`:
- file sha256 `6da26cd7d137510bef384af3a581e857244376c0b5825f071c069ebaabc10922`;
- content sha256 `ef35650678e98a638838af10b75f340cec331f542589d6f089100eee8822624a`;
- manifest sha256 `7c7c94942dc883505b97403042e095d8d616168f3a0a1d8cf495edef25f8b8bd` (S1-REF 36, S2-TARGET 1,980).

The calibration `../calibration-v1.json` was installed byte-identically in `d502777c869eb7345dc05af0f5bc8445877e9b27`:
- file sha256 `6b1db3388b31b1d8840dc32fbaf70976a005331b366ba02e2d7a794f0cc8ad90`;
- content sha256 `1eb1fe68ac2d49228ff8eff6c72391cc0695eaeeb3ae82ed122f9aa738f42e1e`;
- τ_rem `0.025153865322968608`, τ_ins `0.03511072680382421` (P evaluable); the exposed prediction table
  `ce055e09…`.

Every review in this directory is an AI review, labelled as such, not a human validation.
