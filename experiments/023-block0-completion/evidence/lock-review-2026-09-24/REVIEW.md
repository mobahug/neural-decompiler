# Independent lock review — 2026-09-24

**Verdict: PASS WITH NOTES, no blockers.** A separate agent reviewed the lock read-only, at `13b8d39`, before
installation. It rebuilt all four candidate files byte for byte with its own code and found they could be installed
without regeneration; that was done in `f2294ab`.

The files, as reviewed and as installed:
- `preregistration-lock.json`: `4bd5a5b1…`, content `97ca520f…`;
- `preregistration.md`: `dc198785…`;
- `locked-y1-table.f64`: `37603f05…`;
- `locked-y1-table.json`: `3bf85e18…`.

Guards in every review script, before any work:
- 022's table was blocked and proved blocked (`open`, `torch.load` and `os.open` all refused);
- the model was loaded only for its weights, with `torch.nn.Module.__call__` and the capture entry points refused.

| item | result |
|---|---|
| 1. Provenance | HEAD, origin and the lock commit are `13b8d39`; the tree is clean. `51b5c7d` holds only `calibration-v1.json`, identical to the candidate. The state digest `174390d1…` recomputes. Extract, calibrate and lock are complete once each, with no incidents. Nothing scientific changed since calibrate. |
| 2. Lock rebuilt | See the details below this table. |
| 3. Y1 table | The whole table was regenerated in the reviewer's own loop and serialized with its own code: 409,536 values, 0 differing, max 0.0, sha `37603f05…`. Its own index is byte-identical (`3bf85e18…`). `p0_dx3_sha256` and `factors_sha256` reproduce. |
| 4. No prompt or model execution | Loading the weights is expected at lock; executing the model is not. The lock path calls no forward, capture or prompt. The weights object keeps no model or callable (293 objects scanned). The run finished under the call refusal. |
| 5. 022 isolation | The whole review, the Y1 regeneration included, ran with the table blocked. The lock run's `refused_during_run` is empty. |
| 6. Calibration binding | The conditions are the record's envelopes plus the 0.90 guard; none is guard-bound. The four gates are marginal: no aggregate label, no all-pass rule. Effective PASS thresholds: 0.996897 / 0.998634 / 0.995053 / 0.995186. |
| 7. Invariants | I5: 0.0 on all 2,592 pairs. V + P = ΔA0(p_c) holds by definition in 022's code; the reviewer's own block-0 code agrees with it within 2.2e-14. The closed-form T, the reviewer's own implementation, is within 5.0e-15 of 022's exact ΔA0(p_t) on 864 pairs; the lock recorded 4.9e-15 at the same pair. |
| 8. Stage isolation | No Y2 table, no stage-2 file and no report exist. The state has no confirmation, the ledgers are empty and confirm is `not_started`. |

Item 2 in detail:
- **Floors.** They were rebuilt from the draws, whose 32 array digests match the record: element [249] with undefined
  at −∞, exactly. A full independent recalibration from the committed cells agreed within 1.4e-15 in g and 6.7e-16
  in F.
- **Other fields.** The masks (0; 14 / 30), the 79 nouns and the exposed-state digest `26c21d63…` match, and the 108
  per-frame digests match 020's committed extract. So do all bindings, the Y2 specification, the 26 input digests
  and the 11 blobs.
- **The whole file.** The reviewer's own lock serializes to exactly `4bd5a5b1…`, and its preregistration is identical
  (`dc198785…`).

**Notes (non-blocking).**
1. The lock's algebra maximum (4.88e-15) is a rounding-level value from b0c's own closed-form code. It was the only
   field the reviewer copied instead of reproducing; its own value is 5.0e-15 at the same pair.
2. The lock's V + P check confirms consistency only, since it is definitional in 022. The reviewer's independent
   check is the substantive one.
3. `results.json`'s top-level `protocol_code_commit` still reads `4e5deeb`, because `Runner.lock` does not update it.
   The lock, `phases.lock.commit` and the preregistration all carry `13b8d39`, and confirm reads the lock's value.
   This is cosmetic.
4. `preregistration.md` prints F to 6 decimals. The lock's exact bounds decide.
5. On the reviewer's side, eight short inspection commands ran without the file guard. None opens or names 022's
   table, and every computation ran guarded.

**The freeze's no-weights evidence, restated.** This is recorded here by the reviewer's instruction instead of
editing the design text the lock binds.

> Python-level file-open logging cannot establish whether safetensors weight files were read, because native
> mmap/open activity may bypass that audit hook. Therefore the freeze no-weights conclusion relies on the freeze code
> path and the independent tokenizer-only byte-identical reconstruction, not on the file-open log.

The lock run shows why. It loaded the weights, yet its file-open log (`lock-2026-09-24/lock_files.json`) and the
reviewer's own log list only tokenizer and config blobs, never the weight blob `3da38833…`. The freeze's conclusion
itself is unchanged.
