# Experiment 025 — the independent confirmation review (2026-09-27)

**PASS WITH NOTES — no blockers.** The reviewer's statements:
- the scientific result can be accepted as final: **`CAUSAL_EFFECT_NOT_ESTABLISHED`** (A 26/40 FAIL, B 23/40 FAIL,
  G 37/40 PASS, threshold 27, reference tail 0.01923865414210013);
- the evidence may be archived;
- `report` may run exactly once, from state `1d5d37f0…`;
- no confirm rerun is warranted.

The review was done by a separate agent, read-only.
- Nothing in the repository or `outputs/` changed.
- It did not run confirm, report, lock, freeze or the launcher.
- It ran no model forward: `Module.__call__`, the class forwards and every capture/intervention entry point were
  refused, and none fired.

It computed its own responses and outcome first, with the module's scoring functions refused. The gates were
recomputed through its own loops, row indexing, reconstructed vectors and lexicon.

## What it established

- **Provenance.**
  - The launcher's sha256 today is `58b760be…`, matching the command file, the run record and the sentinel.
  - HEAD `6f72d7d`, exit 0; 1 model load; 1 confirm invocation.
  - **The exit-127 event** never started Python. It lasted 2.9 ms, the sentinel and the run log were born inside the
    real launcher's window, and resetting only the four fields confirm writes reproduces the lock state `8eeb8184…`.
    So it was not a `runner.confirm()` invocation or a scientific attempt.
- **Barrier.** The cwd, git, environment and checkpoint checks, the lock, I7′ and the four patch-path checks all came
  before the ledger. The spent keys are absent from the ledger, the run log and the manifest.
- **Accounting.**
  - The keys it rebuilt equal the manifest, the run log and the ledger: 90,720 each, with 0 missing, duplicate or
    extra, and 4,320 per condition.
  - **The log's n-th key equals the runner's loop-order n-th key for all n.**
  - I1 with a wrong vector is ≥ 0.197, and with + and − swapped ≥ 0.634, against 2.15e-5 when correct.
- **Artifact.** All 256 tensor digests match the state; the scores come from the saved and re-read copy.
- **Gates, recomputed on all 90,720 runs.** I1, I3, I4, C (bit-equal on 90,720), and Level-1 over the 16
  outcome-bearing conditions (0.011862701913658569): every maximum and its location match exactly.
- **ℓ, D_attn, A, B, G.** They match the state exactly (G to within 2.8e-17). The rows are aligned by frame, head and
  key, and the label precedence is correct.
- **The adjective/noun split is real data, not an artifact.**
  - The cue order, strata and token ids are correct.
  - s(+) − s(−) = 0.64 for every cue, and d·t̂ = τ > 0, so there is no sign reversal.
  - The condition names parse correctly.
  - Descriptively, the strata separate by the sign of the starting score s₀, and Spearman(A, s₀) = −0.83. This is not
    a new test.
- **The write order.** Removing only the descriptive records from the final state reproduces the result-write digest
  `aa3db023…` exactly.
- **The interpretation.** Only the preregistered reading may be used. The stopping rule holds: the sequence ends, and
  no Experiment 026 follows.

## Notes (non-blocking)

- **N1:** archive the exit-127 event with an explanation. That is done in `../confirm-2026-09-26/README.md`.
- **N2:** the operator's `verify_confirm.py` is not independent, although its values are correct.
- **N3:** the launcher's 8 writes outside the repository are library noise.
- **N4:** the frozen attention program differs from the captured reference rows by up to 6.7e-3, at `cardinal-012-1`.
  That is the known non-exactness the Level-1 gate bounds, and it does not enter D_attn.
- **N5:** the stratum heterogeneity and G's 37/40 are descriptive only.
- **N6:** the reviewer's guard audited Python-level opens only; its final byte-and-time check covers the rest.

## Files

- `REVIEW.md`: the report. An archival header was added; the body below the rule is unchanged, sha256
  `59f59eb3…25b4`.
- `PROMPT.md`: the brief.
- `guard.py` and `final_check.{sh,log}`.
- `r1_records.*`: provenance and records.
- `r2_scores.*` and `r2_values.pt`: the independent ℓ, D_attn, A, B and G.
- `r3_geometry.*` and `vectors32_own.pt`: the reconstructed vectors, and the pairing checks.
- `r4_gates.py`, and `r4_worker{0,1}.{json,log,pt}`: the gates, all 90,720 runs, two workers.
- `r5_descriptives.*`, `r6_aggregate.*`, `r7_today.*`, `r8_extra.*`: descriptives, aggregation, today's state and
  extra checks.
- `r9_refrows.*`, `r9b_refrows_detail.*`: the reference-row alignment.
- `r10_tokens.*`: the token ids.
- `b0_bench.py`, `table_percue.md`.
- `SHA256SUMS`: this directory's digests.
