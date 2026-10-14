# Experiment 101 — evidence

**Status (2026-10-14):** the freeze ran once and is installed; no other phase has run.

| directory | what |
|---|---|
| [`freeze-2026-10-14/`](freeze-2026-10-14/README.md) | the test gate at `40829d2` (A 578, C 645 passed; 0 failed), the guarded production freeze (one dry run kept, then the one production run), the pre-install and post-commit checks (13/13) |
| [`freeze-review-2026-10-14/`](freeze-review-2026-10-14/README.md) | the independent AI freeze review: **PASS WITH NOTES — no blocker**, and the disposition of each finding |

The freeze artifact `../confirmation-v1.json` was installed byte-identically in
`a857f9a2326ff61cae8485deef2ab9337f5d6e02`:
- file sha256 `6da26cd7d137510bef384af3a581e857244376c0b5825f071c069ebaabc10922`;
- content sha256 `ef35650678e98a638838af10b75f340cec331f542589d6f089100eee8822624a`;
- manifest sha256 `7c7c94942dc883505b97403042e095d8d616168f3a0a1d8cf495edef25f8b8bd` (S1-REF 36, S2-TARGET 1,980).

Every review in this directory is an AI review, labelled as such, not a human validation.
