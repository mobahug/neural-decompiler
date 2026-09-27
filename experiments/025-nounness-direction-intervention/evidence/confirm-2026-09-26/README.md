# Experiment 025 — the production confirmation (2026-09-26)

**The official run.** The production `confirm` ran **exactly once**, at commit
`6f72d7d8afcaa079ee38256b7c1250352b1b1bdd`.
- **The launcher:** `launch_confirm_v3.py`, sha256 `58b760bea9d8a7bf66bef4352deb5c7153f7aea46c264de02f49c9ab96d1e307`.
- **The command:** `production_command.txt`. It set `CONFIRM025_MODE=production` and the installed lock's content hash
  `8e5300de…`, and ran under `caffeinate -i -s`.
- **The time:** 19:23:57.846 to 21:12:43.518 UTC (`production_run.json`), with **exit status 0** and no error.
- **The independent confirmation review** is in `../confirmation-review-2026-09-26/`: **PASS WITH NOTES — no
  blockers**.

**The official result (independently reviewed): `CAUSAL_EFFECT_NOT_ESTABLISHED`**

| statistic | positive | result |
|---|---|---|
| A | 26/40 | FAIL |
| B | 23/40 | FAIL |
| G | 37/40 | PASS, but under the frozen hierarchy it cannot change the label |

The threshold is 27 of 40 strictly positive, and the reference tail is 0.01923865414210013. The preregistered reading:
the Experiment 025 test did not establish that moving fresh cue embeddings in the +nounness direction consistently
causes greater frozen block-4/5 readout-approximation error than the matched −nounness intervention, across the
frozen 40-cue population.

## The run, from the launcher's record (`production_run.json`, `production.log`)

- **Preflight: 22 checks true before the model load.**
  - The working directory is the repository root (asserted).
  - HEAD, origin and the remote were `6f72d7d`, and the tree was clean; the runner's own git state was taken at the
    root.
  - The environment and checkpoint are identical to the lock run.
  - The installed lock and preregistration hashes, and the direct module binding (`cr` blob `622aa832…`).
  - The full `cr.validate_lock`.
  - The results state was the lock state; the ledger was empty; confirm was eligible once; the outputs were the lock
    outputs only.
  - The token matched the installed lock.
- **Guards.**
  - 1 model load; **0 refusals**; 0 forbidden reads or writes.
  - **8 spent patch-path runs**: a plain run and a θ = 0 run for each of the four locked keys.
  - **90,720 fresh runs**, each logged with its tagged key in `production_runs.jsonl`. The condition in each key
    comes from the replacement vector's digest, mapped to the lock's 840 vectors.
  - 30,847,520 module calls, all inside sanctioned runs.
  - 8 atomic state writes (a temporary file, then a rename to `results.json`), and the one measurements file.
  - The 8 writes outside the repository are library noise (`/dev/null` and temporary probes), as in the smoke and
    rehearsal runs.
- **The runner's confirm, in order** (the confirm phase record in the results state):
  - the lock was validated; I7′ was bit for bit;
  - the patch-path check passed on all four keys (plain equal to θ = 0 bit for bit; the embedding hook equal to
    block 0's input);
  - the 90,720-key ledger was written before the first fresh run;
  - stage 2 ran; the durable save followed, then the accounting (manifest = executed = ledger = 90,720), the re-read
    with its digests, and C recomputed bit for bit on 90,720 runs;
  - the gates, all passing:

    | gate | max | limit | where |
    |---|---|---|---|
    | I1 | 2.15e-5 | 1e-4 | |
    | I3 | 4.35e-5 | 1e-4 relative | |
    | I4 | 9.00e-5 | 1e-3 | |
    | Level-1, the 16 outcome-bearing conditions | 0.011863 | 0.02 | `knight\|cardinal-012-1\|rand4+0.32` |

  - ℓ and D_attn; A, B, G; the binding re-check;
  - **one atomic result write at 21:09:04 UTC**, digest `aa3db023c4a4afbe730a0e55df00d6d378184bcf8fdf77c967f4d0aad11907c4`;
  - then the descriptive records. The final state is `1d5d37f035bc3153fec6efac527b088f64cd42fcd74e59680f4388daea98b33c`.
- **The measurements** are `outputs/experiment-025/stage2-measurements.pt`: 898,673,151 bytes, sha256
  `87284bddfba16a1b7709d7d33f98aa1a0898b1d85767956a524d2950d80a7788`, 256 tensors. They stay local and git-ignored,
  as in 024; their digests are recorded here and in the state.

## The exit-127 event (provenance only, not a scientific attempt)

At 19:23:09 UTC, the first launch command failed in the shell before Python started.
- **What happened:** the command had been stored in a shell variable. zsh does not word-split an unquoted variable, so
  it looked for a program named after the whole string. It exited with 127 (`no such file or directory`) after about
  3 ms.
- **What did not happen:** no launcher code ran. No sentinel was created, no model loaded, and no state, output or
  ledger changed.
- **So it was not a `runner.confirm()` invocation and not a scientific attempt.**
- **The launch of record followed:** the command was written out literally, at 19:23:54 UTC. The launcher's
  exclusive sentinel (`confirm.sentinel`) and run log were both created inside that launcher's own window. The
  independent review established all of this from the file contents and timestamps.
- **Files:** `shell_error_production.log`, `shell_error_production.exit` and `shell_error_production_command.txt`.

## Before the confirm: the engineering provenance (spent prompts only; no scientific quantity)

- **`launch_confirm.py`, v1, `aafd8954…`.** Used for the first post-install rehearsal, which was read-only with no
  forward: `rehearsal_v1_run.json`, `rehearsal_v1.log`.
- **`launch_confirm_v2.py`, v2, `480fbf55…`.** It added the smoke mode and the signature recording.
  - Its smoke attempt crashed while writing its record, after 6 guarded forwards on the three spent reference prompts
    (`smoke_v2_crashed.log`). The crash came from an `except … as error` at module level, which deletes that name.
  - Its results were not recorded, and no scientific state was affected.
- **`launch_confirm_v3.py`, v3, `58b760be…`.** v2 with a one-line rename. Its complete smoke test passed (`smoke.log`,
  `smoke_run.json`):
  - three spent reference prompts (`cardinal-009-1|ref|581`, `quantifier-009-1|ref|1016`,
    `coordinated-adjective-009-1|ref|581`), each in 020's ledger and in the committed 020 archive, and none in the
    manifest;
  - through the production guards, every logit and captured slice was bit for bit equal to the unguarded reference
    process (`smoke_reference.py`, `.json`, `.log`, `.pt`);
  - the 430 instance and 20 class `forward` wrappers preserve `inspect.signature`;
  - 12 unsanctioned routes were refused before any forward.

  v3 then passed a fresh read-only rehearsal (`rehearsal_run.json`, `rehearsal_v3.log`). It is the production
  launcher.
- **`verify_confirm.py`, `.json`, `.log`.** The operator's read-only post-confirm check, with no forward. It
  reproduced the result, the counts, the per-condition Level-1 maxima and the accounting. It is not independent: it
  scores through `cr.per_cue` and `cr.statistics`. The independent recomputation is in the confirmation review.

`SHA256SUMS` lists this directory's digests.
