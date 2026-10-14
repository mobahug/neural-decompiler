# Experiment 101 — the confirmation (2026-10-14)

`confirm` ran **exactly once**, at commit `59d8ec49e9a34cb40f23b7c3928e8b36522972fe` on `exp101-phases` (the user's
stage-1 commit, pushed), on the user's authorization ("authorize confirm"), under the guarded confirm launcher. It
passed the barrier, wrote the ledger of the 1,980 S2 keys, ran each once, saved the measurements, checked the
accounting, computed G2–G6, scored R, I, D and P, and wrote the result in one atomic write; then the secondary
records.

**The preregistered outcome: `READ_COMPONENT_EFFECTS_AS_PREDICTED`** — independently recomputed by the confirmation
review (`../confirmation-review-2026-10-14/`, an AI review, not a human validation): **PASS WITH NOTES — no blocker; the label recomputed independently, bit for bit** (REVIEW.md sha256 `b13dff8fecb23ee1c5f625bc2d273b975fe39e09045beba15c9aff1a6ce69f65`).

| family | statistic | preregistered threshold | result |
|---|---|---|---|
| precondition | 36 of 36 frames with `Δc_full` ≤ −1 | ≥ 30 | met |
| R (removal) | `N_f` > 0 in **36/36**; median 1.006 (bootstrap 95 % 0.988–1.051); exact tail 2⁻³⁶ | Holm 26/25 of 36; median ≥ 0.05 | PASS |
| I (installation) | `S_f` > 0 in **36/36**; median 1.035 (1.017–1.059); exact tail 2⁻³⁶ | Holm 26/25 of 36; median ≥ 0.05 | PASS |
| D (direction) | `A_amp` < 0 in **28/36** (median −0.136); `A_neg` > 0 in **36/36** (median 0.922) | 25 of 36 each | PASS |
| P (the program) | **33/36** frames within max(τ, 0.02) for both fractions (τ_rem 0.0252, τ_ins 0.0351) | 27 of 36 | PASS |

**The scope of the label** (design §7, frozen): the read component of the residual entering layer 3 at the cue
position, as every reader of that residual sees it, in coordinated frames, for the cue pair one/two from which the read
was estimated. **Not claimed:** that the component is the only route or that `L03.H04` alone mediates the effect (the
residual edit reaches every reader of `x(p_c)`), or anything beyond the one/two pair, this site, this checkpoint and
these frames.

## The gates (all held; no frame dropped, replaced or ignored)

| gate | value | tolerance | margin |
|---|---|---|---|
| G2 (realized edit geometry) | max 4.5e-7 (direction), 3.3e-8 (read), 3.5e-8 (length); û_off 5.5e-9 | 1e-4; 1e-6 | ≥ 181× |
| G3 | NOOP bit for bit 36/36; pass-through 7.3e-7 | bitwise; 1e-4 | 137× |
| G4 | `x3(p_t)` unchanged 1,692/1,692; I3 1.47e-5; I4 4.19e-5; **Level-1 6.14e-3**; readout 4.70e-3; logit 3.14e-3 | bitwise; 1e-4; 1e-3; 2e-2; 2e-2; 2e-2 | 6.8×, 24×, **3.3×**, 4.3×, 6.4× |
| G5 (head arms) | row 7.1e-6; output 1.3e-5; landings 144/144 | 1e-4; 1e-3; bitwise | 14×, 77× |
| G6 | the stage-2 REF runs reproduce the committed stage-1 states and `c_ref` bit for bit (36/36); the table recomputed from the committed states before the result write, bit for bit | bitwise | — |
| G7 | 1,980 S2 keys, each once; the ledger equals the manifest; accounting equal | exact | — |

Recorded, not gated: `E₁` max 0.0217; the frozen readout's ceiling error max 1.20 nats on one noun
(`NEC-READ@many`, frame 20).

## Secondary records (no label force; computed after the result was written)

- **The head-only arms** (`L03.H04`'s output at `p_t` replaced by the read edit's exact effect): removal share median
  0.882 (95 % 0.874–0.891), installation share 0.894 (0.886–0.906); the value-only variants 0.877 and 0.892. So about
  88–89 % of the read edit's effect passes through `L03.H04`'s output; the rest reaches the logits through other readers
  of `x(p_c)`.
- **The isotropic controls:** the read edits beat the median of the 8 isotropic directions in 36/36 frames for both
  removal and installation.
- **The stratum** (`three`, `many`; cannot rescue or change the primary result): R 36/36 and I 36/36 for both; D's
  `A_amp` < 0 in 28/36 (`three`) and 23/36 (`many`), `A_neg` > 0 36/36 for both.
- **The manipulation check** (`ΔT` along `d̂_T` relative to `PL`): NEC-READ −0.008, SUF-READ 0.979, NEC-AMP 1.943,
  SUF-NEG −0.891, NEC-OFF 1.026, SUF-OFF 0.003; the half doses 0.499 and 0.484.
- **The doses and swaps:** half dose `F_half_rem` 0.703, `F_half_ins` 0.717 (non-linear); `F_swap_ref` 1.008 and
  `F_swap_pl` 0.044 (the whole `x(p_c)` swap carries the full shift); additivity `F_rem + F_ins` 0.989.
- **The program's errors:** `|F̂_rem − F_rem|` median 0.0060, `|F̂_ins − F_ins|` 0.0113.

## The launcher (`launch_confirm.py`, sha256 `56a687f372605e5d6e16e2cc62bc9fd1197f21fdd8dccf0dda6d6ac75723309f`)

Version 1 (`launch_confirm_v1_as_reviewed.py`, `4f132e8e…`) was reviewed READY WITH CHANGES; version 2 applies the
review's S1 (SIGHUP ignored; production refuses a terminal) and N1/N3/N7/N8/N9, and its diff-check found it READY FOR
PRODUCTION (`../confirm-launcher-review-2026-10-14/`). Its sha256 was checked before each mode.
- **Rehearsal** (`rehearsal_run_20261014T173655169844Z.json`; 17:36:27–17:36:55 UTC): 21/21 preflight checks (the
  stage-1 files committed and equal to the record; stage 1 complete with no incident; the full `_validated_lock` with the
  stage-1 files; the environment equal to the lock run's; the S2 keys; the hook's self-test) and the weights-only
  barrier dry run: the geometry bit for bit against the lock, the runner's own `_barrier` passed (table `91918004…`
  recomputed bit for bit); 0 forwards; nothing written.
- **Smoke** (`smoke_run_20261014T173725365643Z.json`): one spent exposed frame's 55 runs through the production guards,
  bit-identical to the calibrate launcher's unguarded reference (71 tensors, logits); six unsanctioned routes refused
  with 0 forwards.
- **Production** (`production_run_20261014T174234317359Z.json`, `production.log`, `production.exit`,
  `production_runs.jsonl`, `production_started_utc.txt`, `confirm.sentinel`): launched detached with `nohup`, output to
  a file, tokens = the lock's and the table's content sha256s, `HF_HUB_OFFLINE=1`, `-B`, `GIT_OPTIONAL_LOCKS=0`;
  17:37:40–17:42:34 UTC, **exit 0**; confirm invocations 1, model loads 1, **1,980 sanctioned runs, each S2 key once**
  (673,200 module calls = 1,980 × 340), refusals 0, forbidden reads or writes 0, no signal; the tree clean after;
  the state `1e3eaeb3…`.
- **The report check** (`../report-2026-10-14/check_run_*.json`): G6 at the report, bit for bit, read-only.

## Files

| file | what |
|---|---|
| `launch_confirm.py`, `launch_confirm_v1_as_reviewed.py`, `launcher_sha256_at_production.txt` | the launcher (version 2, which ran), version 1 as reviewed, the sha256 at production |
| `hook_self_test.py`, `hook_self_test.out.json` | the isolated test of the hook's decisions |
| `rehearsal_run_*.json`, `rehearsal.log` | the rehearsal with the barrier dry run |
| `smoke_run_*.json`, `smoke.log` | the smoke |
| `production_run_*.json`, `production.log`, `production.exit`, `production_started_utc.txt`, `confirm.sentinel` | the production run |
| `production_runs.jsonl` | one line per sanctioned run (1,980), at admission, with the replacement digests |
| `outputs_after_confirm.sha256` | the results state and the measurements after confirm |
| `SHA256SUMS` | this directory's digests |
