# Experiment 101 — the calibration (2026-10-14; exposed, exploratory)

`calibrate` ran **exactly once**, at commit `8f65062627c77815adf9289c3fe60191d9fc3c17` on `exp101-phases`, under a
guarded launcher, on spent data only: the 36 exposed coordinated frames of Experiment 020's pool, their 36 reference
re-captures and 1,980 arm runs (2,016 keys, each once). No 101 frame prompt ran.

**Outcome: complete, exit 0; no revision trigger fired.** `τ_rem` = 0.025153865322968608 and `τ_ins` =
0.03511072680382421 (the 33rd of 36 sorted exposed errors; both ≤ 0.20, so P is evaluable). Every gate held, G4 included
(§ 3). The candidate `calibration-v1.json` (file sha256
`6b1db3388b31b1d8840dc32fbaf70976a005331b366ba02e2d7a794f0cc8ad90`, content sha256
`1eb1fe68ac2d49228ff8eff6c72391cc0695eaeeb3ae82ed122f9aa738f42e1e`) was independently reviewed
(`../calibration-review-2026-10-14/`, an AI review, not a human validation: **PASS WITH NOTES, no blocker**) and then
installed byte-identically in `d502777c869eb7345dc05af0f5bc8445877e9b27`, alone.

## 1. The test gate

No code changed after the freeze's test gate: `git diff 40829d2 8f65062 -- src tests experiments/101-*/run.py` is
empty, so the gate at `40829d2` (tier A 578, tier C 645 passed, 0 failed; `../freeze-2026-10-14/test-gate/`) is the gate
of this phase.

## 2. The launcher (`launch_calibrate.py`, sha256 `0ad9d0e1c49e136c8d765628f31845de5cdae4ad322d03f54acb5785caa6bed8`)

It follows 025's confirm-launcher v3 pattern, with the freeze review's N1 applied to its audit hook. It ran from a
scratchpad directory outside the repository; its records were copied here byte-identically afterwards. The modes, run in
this order:
- **`rehearsal`** (`rehearsal_run.json`, `rehearsal.log`; 08:43:31–08:43:42 UTC): the preflight only, no forward and no
  write. All 20 checks true (the 21st, the production token, exists only in production): the repository root, a
  clean tree, the runner's git state, HEAD descending from `origin/main` (8 commits ahead, not pushed), the freeze record, the environment against the freeze run's, the 20
  pinned blobs, `ri`'s and the runner's blobs at HEAD, the production configuration, the committed confirmation file
  (both hashes), no results state and an empty outputs directory, calibrate eligible, the runtime and versions equal to
  020's explore, and the 2,016 calibration keys (distinct, spent exposed keys only, disjoint from 101's, no 101 token
  sequence). 11 entry-point aliases were wrapped.
- **`smoke`** (`smoke_run.json`, `smoke.log`; 08:44:02–08:44:28 UTC), engineering only: the production guards with one
  exposed frame's 56 keys (`coordinated-adjective-1`): its S1 re-capture and its 55 runs through the sanctioned
  wrappers, then `ri.run_gates` and `ri.prediction_table` for that frame. Every tensor digest, every logits digest and
  the re-captured state's digest equal the unguarded process's (`smoke_reference.py`, run at 08:15 UTC; the freeze
  review's N7) bit for bit; 430 module instances and 20 class methods were sealed with their signatures kept; and seven
  unsanctioned routes (a model call, `model.forward`, `pm.capture_prompt`, `rd.capture_frame_020` directly, a key
  outside the manifest, a second run of a key, a second re-capture) were each refused with 0 forward calls. The smoke
  wrote nothing in the repository. Its 56 runs, like the unguarded reference's, are outside every 101 ledger: spent
  exposed prompts, as tier C's runs on the same frames.
- **`production`** (`production_run.json`, `production.log`, `production_runs.jsonl`, `production.exit`; started
  08:44:40 UTC): token `CALIBRATE101_CONFIRMATION` = the confirmation's content sha256; an exclusive sentinel named by
  the commit (`calibrate-8f65062….sentinel`). Ended 08:51:04 UTC, **exit 0**:
  - **calibrate invocations 1, model loads 1, refusals 0**, forbidden reads 0, forbidden writes 0;
  - **sanctioned runs 2,016, each key once** (`post.all_keys_ran_once` true), and 685,440 sanctioned module calls (340 per
    run, as in the smoke: one forward per run);
  - the calibration ledger (`calibration_executed` and the planned digest `5d4bcb51…`) was on disk before the first
    prompt, as the launcher requires before admitting any run;
  - repository writes: only `outputs/experiment-101/` (the results state five times, each through a temporary file and
    one rename; `calibration-measurements.pt` once; `candidate-calibration-v1.json` once); the tree was clean after;
  - the run log records each admission before its forward (the calibration review's N5): with exit 0 and equal
    accounting every logged key completed, but a failing run would also appear in it.

  The keys of `coordinated-adjective-1` thus ran three times in all (the unguarded smoke reference, the smoke and the
  phase; the calibration review's N4): spent exposed prompts, outside every 101 ledger, with bit-identical results
  across the three processes.

## 3. The results

All values are exposed and exploratory (spent data; some of these frames entered `d̂_T` and 012's bases). The design
forbids any revision on their size, and none was made.

**The triggers** (design §6): gate failures 0; `‖off_c‖ < s` in 0 of 36 frames (largest `s/‖off_c‖` 0.474); `Δc_full >
−1` in 0 of 36 (least negative −2.604); non-finite fractions 0.

**G4, the question this phase answers** (the reviewer's rule 1: test, never widen). Every frozen tolerance held on all
1,800 G4 runs (the 1,692 residual-edit runs and the 108 plain cue runs):

| G4 quantity | tolerance | exposed maximum | margin |
|---|---|---|---|
| `x3(p_t)` unchanged | bit for bit | 1,692 / 1,692 equal | — |
| 023's I3 (each plain pair) | 1e-4 relative | 1.18e-5 | 8.5× |
| 023's I4 | 1e-3 nats | 3.44e-5 | 29× |
| 020's Level-1, 025's contrast form | 2e-2 nats | 1.065e-2 (`NEC-HALF@many`, frame `018-3`); `two`'s arms ≤ 5.01e-3 | 1.9× |
| 020's readout identity | 2e-2 | 4.87e-3 | 4.1× |
| 020's logit identity | 2e-2 | 3.03e-3 | 6.6× |
| `E₁` (recorded, not gated) | — | 1.60e-2 | — |

The other gates: G1 exact (`read_weight` max-abs 0.0; the basis to 1e-15); G2 margins ≥ 186×; G3's NOOP bit for bit in
36/36 and the pass-through identity 6.6e-7; G5 row 5.7e-6 and output 1.1e-5, landings 144/144; G6 36/36; the
re-captured reference states equal 020's locked states exactly (drift 0.0).

The calibration review attributes the Level-1 bulk to the float32 readout path and the two outliers to float32 error of
the measured blocks 4–5 output (020's attention-score floor), not to the edit size (Spearman with `‖Δx3(p_c)‖` −0.02). Its
estimate of the probability that one of the ~1,800 confirm runs exceeds 2e-2 is about 0.3 % (0.02–1 % across methods; a
heavy-tailed frame bootstrap reaches about 10 % at its 95th percentile). A failure would make the confirmation
`NOT_INTERPRETABLE` (rule 2). This bears on the lock decision; it changes no tolerance.

**The measured quantities for `two`** (median, range): `Δc_full` −4.94 (−6.04 to −2.60); `F_rem` −0.035 (−0.451 to
0.158); `F_rem_off` 0.995 (0.976 to 1.024); `F_ins` 0.980 (0.886 to 1.076); `F_ins_off` −0.041 (−0.172 to 0.030); N
1.031; S 1.033; `A_amp` −0.145 (−0.448 to +0.128); `A_neg` 1.167. The shares: N > 0 and S > 0 in 36/36, `A_amp` < 0 in
30/36, `A_neg` > 0 in 36/36, P's joint agreement 31/36; sign-test power at those shares 1.0 (R, I), 0.989 and 1.0 (D),
0.978 (P). Descriptively, about 88–89 % of the read edit's effect passes through `L03.H04`'s output at `p_t` (the
head-only arms), and the eight isotropic controls do nothing (36/36 frames on each side).

## Files

| file | what |
|---|---|
| `launch_calibrate.py` | the guarded launcher (modes `rehearsal`, `smoke`, `production`) |
| `rehearsal_run.json`, `rehearsal.log` | the rehearsal |
| `smoke_reference.py`, `smoke_reference.log`, `smoke_reference.pt` | the unguarded reference for the smoke (digests only) |
| `smoke_run.json`, `smoke.log` | the smoke |
| `production_run.json`, `production.log`, `production.exit`, `production_runs.jsonl` | the production run's record, log, exit status and run log (one line per sanctioned run: key and replacement digest) |
| `calibrate-8f65062….sentinel` | the production sentinel |
| `post_install_checks.py/.out` | the checks after the calibration file's install commit (15/15 true) |
| `state_extract.json` | an extract of the git-ignored results state after calibrate (`state_sha256` `729c9cb5…`; the ledgers by count and digest) |
| `SHA256SUMS` | this directory's digests |
