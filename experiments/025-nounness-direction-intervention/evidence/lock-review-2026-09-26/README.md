# Experiment 025 — the independent lock review (2026-09-26)

**PASS WITH NOTES — no blockers.** The reviewer's statement: the exact candidate lock and preregistration can be
installed byte for byte, with no regeneration. They were installed that way in `5ee7a74`. The reviewer accepted the
review and turned notes N1–N4 into requirements for the confirm launcher (`../lock-2026-09-26/README.md`).

The review was a separate agent, working read-only under a guard (`guard.py`):
- no repository write, and no freeze, lock, confirm or report;
- no fresh or rotated prompt, and no ℓ, MSE, A, B, G, `D_attn` or Level-1 value;
- weights-only work, plus 6 guarded plain captures of spent reference prompts.

## The independent reconstruction

- **The geometry was reproduced independently.** The reviewer wrote their own code for `d`, the 40 cue geometries,
  the 840 vectors, the SHA-256 counter-mode Box–Muller controls and `p̂`. Every `cr` geometry function was refused
  until their values were saved (`r2a_stream.py`, `r2a_own_geometry.py`, `r2a_own.*`). The cross-check against `cr`
  and the lock (`r2b_crosscheck.*`) agrees **bit for bit**: the maximum disagreement is 0.0 for every scalar, vector,
  control, uniform and Gaussian. The one nonzero, ≤ 1.7e-16, is in the angle checks, where a different formula was
  used on purpose. A 40-digit mpmath reference bounds the float64 error at ≤ 6.7e-17 for `d` and ≤ 4.9e-16 for τ.
- **The random controls are nounness-neutral on the actual float32 patches.** Over all 280 controls:
  - |s(+θ) − s(−θ)| ≤ 6.088945903037768e-09 (bound 1e-6);
  - the angle is within 3.7e-9 rad of θ, and within 4.2e-9 of the nounness rotation's angle;
  - the norm is within 6.0e-9;
  - |d·u|, |Ê·u| and |t̂·u| are ≤ 6.1e-17;
  - the effective float32 displacement has |d·u_eff| ≤ 1.6e-8 and cos(u_eff, u) ≥ 0.99999999.

  B's premise holds.
- **Determinism (`r3_determinism.*`).** Two extra processes gave identical controls:
  - every RNG was seeded and advanced differently, with different `PYTHONHASHSEED` values;
  - the geometry block is bitwise equal to the lock under 1, 4 and 8 threads, and under a shuffled cue order;
  - the code contains no RNG.
- **The `D_attn` rows were checked against spent reference captures (`r4_dattn_binding.*`).**
  - The exposed-states digest `26c21d63…` recomputes and covers `rows4`/`rows5` ([8, p_t+1]; rows sum to 1).
  - Six guarded spent reference captures (2 per template, keys `frame|ref|id` in 020's ledger) are bitwise equal to
    the locked rows, through confirm's own capture and slicing path. Only equality flags were recorded.
- **Confirm ordering and Level-1 coverage (`r5_logic.*`, and code reading).**
  - Synthetic tests reproduce the A/B/G formulas, the strict-positive count, and the exact tail
    5288280983/274877906944 = 0.01923865414210013.
  - The label hierarchy is correct on all 16 combinations.
  - Level-1 is gated on exactly the 16 outcome-bearing conditions.
  - Confirm's pre-prompt ordering and write ordering show no deviation with any consequence.
- **Bindings and isolation (`r1_integrity.*`, `r6_launcher_guards.*`).**
  - The reviewer's own code recomputes all 13 module blobs, `cr`'s own blob `622aa832…`, 024's three files, 020's
    prior-noun file, the parameter and embedding digests, and the revision.
  - `validate_lock` refuses every drift case tested.
  - The manifest is 90,720 keys, `0e1f068b…`, with 0 collisions with the 43,632 spent keys or the 4 patch-path
    keys.
  - The launcher records show zero refusals and one model load.

## Notes (non-blocking; see `REVIEW.md`)

- **N1:** confirm's git check runs in the process's working directory, so the launcher must `chdir` to the repository
  root.
- **N2:** confirm must run on the same machine and stack as the lock.
- **N3:** the lock launcher did not seal submodule classes' `forward`. Nothing used that route.
- **N4:** the lock launcher's write allow-list was the whole output directory.
- **N5:** the extra integrity reads, the same kind accepted at the freeze.
- **N6:** the model identity is checked by digests, and by the revision pinned in the loader.
- **N7, N8 (informational):**
  - the combined vector digest is of the [40, 21, 512] stack (a [840, 512] stack gives `a8d1baa3…`);
  - stage 2 is frame-major;
  - `+inf` counts against;
  - the preregistration has two "A" lines (the formula and its meaning).

**The reviewer's disclosure:** a final `find -newer` over the repository listed file metadata (times) under
`outputs/experiment-023/` and `outputs/experiment-024/`, and 022's `calibration-table.pt`. No file there was opened or
read.

## Files

| file | what |
|---|---|
| `REVIEW.md` | the report. An archival header was added; the body below the rule is unchanged (sha256 `991102b1…e1`) |
| `PROMPT.md` | the review brief (the reviewer's 20 items) |
| `guard.py` | the reviewer's guard |
| `r1_integrity.*` | repository, phase, candidate and binding integrity |
| `r2a_stream.py`, `r2a_own_geometry.py`, `r2a_own.json`, `r2a_own.log`, `r2a_own.pt` (9.6 MB of reconstruction tensors) | the independent reconstruction |
| `r2b_crosscheck.*` | the cross-check against `cr` and the lock |
| `r3_determinism.py`, `r3_determinism_*.{log,out.json}` | the determinism processes (seeds 271828, 31337) |
| `r4_dattn_binding.*` | the `D_attn` reference-row binding |
| `r5_logic.*` | the statistics, the outcome hierarchy, Level-1 coverage |
| `r6_launcher_guards.*` | the lock launcher's guards |
| `first_pass/` | the reviewer's first-pass outputs, superseded by the final ones above; kept as provenance |
| `SHA256SUMS` | this directory's digests |
