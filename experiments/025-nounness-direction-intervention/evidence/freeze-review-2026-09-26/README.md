# Experiment 025 — the independent freeze verification (2026-09-26)

**PASS WITH NOTES — no blockers.** The reviewer accepted it, with the decision on N3 recorded in
`../freeze-2026-09-26/README.md` (section 6).

The verifier worked independently and read-only: tokenizer and committed files only, with no model, no forward, no
score, no geometry and no runner phase.
- It re-derived, with its own code, before touching the module:
  - the four exclusion sets;
  - the 40 cues and every rejection reason;
  - the 90,720-key manifest and its digest;
  - the configuration;
  - the 43,632-key spent set.
- Everything matched the file. Afterwards, the module's `cr.freeze_payload` reproduced the file byte for byte.
- A replay of the launcher's setup code confirmed that its guards refuse a model load, a forward, a capture, score and
  geometry calls, and forbidden reads and writes.

| file | what |
|---|---|
| `REVIEW.md` | the verifier's report. An archival header was added; the body below the rule is unchanged (sha256 `8e14258f…cdf71`) |
| `PROMPT.md` | the verification brief. Its tokenizer id `EleutherAI/pythia-70m` is a wording mistake for the pinned `EleutherAI/pythia-70m-deduped` (no effect) |
| `derive_independent.py`, `.out.json` | the independent re-derivation |
| `crosscheck_module.py`, `.out.json` | the module cross-check, run afterwards |
| `guard_test.py`, `.out.json` | the replay of the launcher's guards |
| `repo_state.log` | the repository state during the review |
| `SHA256SUMS` | this directory's digests |
