# Experiment 101: review of the confirmation

**AI review (independent, read-only), not human validation.** This review was written by a Claude agent (Opus 5.5) on
2026-10-14 under the brief in `PROMPT.md`. The repository was at HEAD `59d8ec49e9a34cb40f23b7c3928e8b36522972fe` on
`exp101-phases`, clean, and equal to `origin/exp101-phases`. Nothing in the repository was created, changed, staged or
committed, `outputs/` included. The snapshots before and after are byte-identical (`out/repo_snapshot_{start,end}.txt`).
No 101 prompt ran: neither S1 nor S2, and none of the 60 candidate texts with any cue. No runner `confirm`, `stage1` or
`report` ran. Every script that read the weights installed this review's token-sequence guard. The guard refuses every
module call and every forward entry point, and it never fired: 0 refusals and 0 token hits in the
`out/*_guard.json` records.

## Verdict on execution and computation: **PASS WITH NOTES**

- **The execution was clean.** `confirm` ran exactly once, at `59d8ec4`, under the reviewed launcher (sha256 `56a687f3…`,
  READY FOR PRODUCTION). The barrier and the ledger came before any S2 forward. All 1,980 S2 keys ran once each, in
  frozen order, through `ri.run_arm`. The forward count is 673,200, exactly 1,980 × 340 module calls, so nothing else
  ran. No S1 key ran again, no incident occurred, and no partial file exists. The only writes were 9 atomic state
  writes and `stage2-measurements.pt`.
- **Every gate G2–G6 holds.** This review recomputed each one from the saved measurements, independently where that was
  feasible: every one of the 1,692 residual-edit replacements was rebuilt bit for bit from this review's own formulas,
  and G3, G5 and G6 were recomputed with this review's own code and the checkpoint's weights. The frozen functions
  reproduce the state's gate book and ceilings bit for bit. The tightest margin is Level-1's, 3.26×.
- **The primary statistics and the label match bit for bit.** This review recomputed every per-frame value, every count,
  every exact binomial tail, Holm, every median, D, P, the precondition and the label with its own code, and all of them
  equal the state bit for bit.
- **The prediction freeze held.** The table used for P is the committed `stage1/table.json` (content `9191800440af…`). Its
  digests were committed in evidence at `f4e3a06` (14:02Z) and the file itself at `59d8ec4` (17:35:06Z, pushed). Both
  came before the S2 ledger write (17:38:22Z), and the barrier recomputed the table bit for bit. No scientific path
  changed after the lock. Every tolerance and threshold used is the lock's.
- **The secondary records match bit for bit.** They were recomputed both independently and through the frozen
  functions. They were written only after the result: the digest the runner logged for the result write (`04221aa2…`)
  is reproduced exactly by the final state minus its `secondary` block.
- **Findings.** There is no blocker. Two findings are should-fix, both about what the report and closure must carry: the
  design §2 per-frame overshoot ratio, and the ceilings and E₁ with what they mean. The other findings are notes on
  wording and interpretation. **No finding changes the label.**

**Recomputed label: `READ_COMPONENT_EFFECTS_AS_PREDICTED`**

| family | recomputed | criterion (lock) | result |
|---|---|---|---|
| precondition | 36 of 36 frames with `Δc_full ≤ −1` (range −6.581 to −3.947; no weak frame) | ≥ 30 | met |
| R | 36 of 36 with `N_f > 0`; p = 1/2³⁶ = 1/68,719,476,736 ≈ 1.46e-11; median `N` = 1.0063799347533808 | Holm at 1/40 (26 or 25), median ≥ 0.05 | PASS |
| I | 36 of 36 with `S_f > 0`; p = 1/2³⁶; median `S` = 1.0350265212692265 | Holm at 1/40, median ≥ 0.05 | PASS |
| Holm | both p ≤ 1/80: the tie is ordered I then R, I is tested against 1/80 and R against 1/40, and both are rejected | thresholds 26 and 25 (tails 0.005665…, 0.014408…; 24 gives 0.0326) | — |
| D | `A_amp < 0` in 28 of 36 (median −0.1360); `A_neg > 0` in 36 of 36 (median +0.9223) | 25 of 36 each | PASS (margin 3 frames on `A_amp`) |
| P | 33 of 36 agree (frames 03, 08 and 09 disagree) | `|F̂ − F| ≤ max(τ, 0.02)`, τ_rem 0.025153865322968608, τ_ins 0.03511072680382421; ≥ 27 | PASS (margin 6) |

## How this review worked

The scripts are in `scripts/` and their outputs in `out/`. All of them ran with `.venv/bin/python -B`,
`HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1` and `GIT_OPTIONAL_LOCKS=0`.

| script | what it does | model |
|---|---|---|
| `rguard.py` | Imported first everywhere. An audit hook refuses any write in the repository, any read of other experiments' outputs except 020's `results.json`, and any child process except read-only git. The forward guards refuse every module call and every forward entry point (12 replaced by identity), after scanning the arguments against 624 forbidden token sequences (60 candidates × 4 cues, with and without a leading space, plus the 36 frozen frames' prefix + cue [+ suffix]). It also seals the model. | — |
| `a_integrity.py` | Digests, the result-write digest, keys, the run log against the measurements, landing, the launcher records | none |
| `b_primary.py` | The primary statistics and the label, own code (no `neural_decompiler` import) | none |
| `c_gates_independent.py` | G1–G6 with own code; weights read directly from the checkpoint's safetensors file (sha256 `3da38833…`, equal to the production record); the overshoot ratios; the LayerNorm and manipulation records | none (no model object) |
| `d_frozen_crosscheck.py` | The frozen functions as a cross-check: the table (G6), `ri.run_gates`, `ri.primary_statistics`, `ri.run_records` and `ri.secondary_statistics` against the state, bit for bit | loaded and sealed; 0 module calls |
| `e_secondary.py` | The secondary statistics and the bootstrap, own code, against the state; interpretive numbers | none |
| `f_e1_by_arm.py`, `g_ceiling_by_arm.py` | Descriptive breakdowns of E₁ and of the ceiling error per arm (recorded, not gated) | loaded and sealed; 0 module calls |

## 1. Execution integrity (item 1)

| check | evidence | result |
|---|---|---|
| one confirm invocation | launcher `confirm_invocations` 1, `load_model` 1; sentinel = launcher sha256 `56a687f3…` = the reviewed v2 (`confirm-launcher-review/REVIEW-v2-diffcheck.md`: READY FOR PRODUCTION); the phase rule refuses a second confirm | ✓ |
| the order: rehearsal, then smoke, then production | rehearsal 17:36:27–17:36:55Z (0 forwards; its barrier dry run passed on the same table digest); smoke 17:37:09–17:37:25Z (spent exposed frame `coordinated-adjective-1` only; bit-equal to the unguarded reference; 6 refusal probes refused); production 17:37:39–17:42:33Z, exit 0, no signal | ✓ |
| the barrier before the ledger | `run.py:804–814`: `_geometry_against_lock` and `_barrier` run before the one write that puts all 1,980 S2 keys in the ledger together with `barrier.passed` and `phases.confirm = running`. The barrier's record digest `d2f1ed03…` equals the stage-1 record committed in evidence at `f4e3a06` | ✓ |
| the ledger before the first S2 forward | the launcher's `_ledger_on_disk` reads `results.json` from disk at the first sanctioned `run_arm` and requires the S2 keys, `barrier.passed` and `running`; 0 refusals; the first temporary state file opened (`.results-w9en66lw.json`) precedes the measurement file in the launcher's open log | ✓ |
| each S2 key exactly once, through `ri.run_arm` | `production_runs.jsonl`: 1,980 rows numbered 1–1,980, in exactly the frozen order (frames 01–36 × the lock's 55 tags), each key once, none `|S1`; `done_keys == allowed_keys`; the state's ledger is the 36 S1 keys plus the 1,980 S2 keys (2,016, distinct), equal to the committed manifest | ✓ |
| the run log against the measurement rows | every logged replacement digest (the sha256 of the float32 vector at admission) equals the saved replacement: 1,692 residual edits at `RESID_PRE.L3@p_c` and 144 head edits at `L03.H04@p_t`; the 144 plain runs logged none; every saved replacement equals the captured value of its own row bit for bit (`x3_pc` or `T_pt`) | ✓ |
| no other forward | 673,200 sanctioned module calls = 1,980 × 340, the per-forward count measured in calibrate and stage 1; any module call outside `ri.run_arm` is refused, and none was recorded; `ri.capture_reference` refused outright | ✓ |
| the accounting | `Counter(executed) == manifest`, ledger set equal: 1,980/1,980/1,980 | ✓ |
| the measurement digests | file sha256 `4e9a3a63…` (= `outputs_after_confirm.sha256`); all 2,556 tensors' digests recomputed with this review's own implementation equal the state's; names exactly the expected set; every value finite | ✓ |
| the state digests | `state_sha256` `1e3eaeb3…` reproduces; file sha256 `adc83b1a…`; the stage-1 block equals the extract committed at `f4e3a06` | ✓ |
| the writes | the launcher's hook allowed only `results.json` (9 distinct `.results-*.json` temporaries, none left) and `stage2-measurements.pt`; the 9 writes match the code path with no incident (ledger, stage-2 digests, accounting, gates, G6, the result, 3 secondary); no `stage2-partial.pt`; `tree_after` empty; only the expected files in `outputs/experiment-101/` | ✓ |
| incidents | none: no `confirmation.incident`, no phase incident; the secondary step recorded no failure | ✓ |

## 2. The gates, recomputed (item 2)

All maxima below are over all 36 frames. "Own" means this review's code, with the checkpoint's weights read from
safetensors. "Frozen" means `ri.run_gates` re-run on the saved measurements with the committed states; it reproduced
the state's gate book and ceilings bit for bit.

| gate | maximum | where | tolerance | margin | how recomputed |
|---|---|---|---|---|---|
| G1 read weight (γ₃ ⊙ W_V W_O d̂_T, centred, against `w_c`) | 0.0 | — | 1e-9 | — | own, from safetensors; `w_c`, `ŵ`, `d̂_T` and all 288 `v_j` digests equal the lock |
| G1 basis (`v_j` unit, mean-zero, ⊥ `w_c`) | 6.66e-16 | — | 1e-12 | 1,500× | own |
| G2 replacements | 1,692 of 1,692 rebuilt bit for bit | — | exact | — | own formulas (float64 edit, float32 cast); NOOP and PC-SWAP reuse the captured float32 vectors |
| G2 `r(x′)` against its target | 3.2507e-8 | 101-21 `SUF-ISO-3@two` | 1e-4 | 3,076× | own = state, bit for bit |
| G2 centred length (relative) | 3.5229e-8 | 101-14 `SUF-HALF@three` | 1e-4 | 2,839× | own = state |
| G2 parallel / orthogonal | 4.4754e-7 | 101-28 `NEC-HALF@three` | 1e-4 | 223× | own = state |
| G2 float32 `û_off` | 5.512e-9 | 101-21 `three` | 1e-6 | 181× | own = state |
| G3 NOOP against `PL@two` | 36 of 36 bit for bit (8 capture fields, contrasts, logit identity, logits digest) | — | exact | — | own |
| G3 pass-through | 7.2736e-7 | 101-34 `NEC-ISO-5@two` | 1e-4 | 137× | own (b_O[3] from the checkpoint) = state |
| G4 `x3(p_t)` | 1,692 of 1,692 bit for bit (and all 1,836 non-plain runs, head arms included) | — | exact | — | own |
| G4 023's I3 (per plain pair, relative) | 1.4675e-5 | 101-18 `two` | 1e-4 | 6.8× | frozen |
| G4 023's I4 | 4.1905e-5 | 101-21 `NEC-HALF@three` | 1e-3 | 23.9× | frozen |
| G4 020's Level-1 (contrast form) | 6.1370e-3 | 101-09 `NEC-HALF@two` | 2e-2 | **3.26×** (tightest; calibration 1.88×) | frozen; per arm ≤ 6.14e-3 (`out/f_e1_by_arm.json`) |
| G4 020's readout identity | 4.6974e-3 | 101-04 `SUF-NEG@two` | 2e-2 | 4.26× | own (final LN and W_U from the checkpoint) = state |
| G4 020's logit identity | 3.1447e-3 | 101-02 `NEC-ISO-4@two` | 2e-2 | 6.36× | the per-run values were computed at run time from the logits, which are kept only as digests; their maximum was re-read from the saved tensor. Own contrast-level analogue on all 1,980 runs, from each row's saved `h6` through the checkpoint's final LN and W_U: max 3.34e-3 |
| G4 E₁ (recorded, not gated) | 0.021730 | 101-28 `SUF-ISO-5@two` (‖Δh6‖∞ 0.077) | — | — | per arm, ≤ 0.0066 for every arm except SUF-OFF (≤ 0.0151; ‖Δh6‖∞ 0.09–0.15) and SUF-ISO (0.0217), whose Δh6 is near zero |
| G4 ceiling error (recorded, not gated) | 1.1957 nats per noun | 101-20 `NEC-READ@many` | — | — | per arm below (§5) |
| G5 row | 7.058e-6 | 101-13 `HEAD-NEC@two` | 1e-4 | 14.2× | own float64 rotary attention: 7.057777231644913e-06 (state 7.05777723297718e-06) |
| G5 output along d̂_T | 1.2985e-5 | 101-13 `HEAD-NEC@two` | 1e-3 | 77× | own |
| G5 landing | 144 of 144 bit for bit | — | exact | — | own; each head replacement = float32(captured + saved δ); δ recomputed to 7.7e-15 relative (full) and exactly (value-only) |
| G6 REF against stage 1 | `c_ref` 36 of 36 and `x3(p_c)`, `x3(p_t)`, `h6` 36 of 36, bit for bit; also `x3` at every position | — | exact | — | own reader of `states.f64` (sha256 `1c7e51c0…`) |
| G6 table | bit for bit, recomputed from the committed states and the weights (here, as at the barrier and before the result write) | — | exact | — | frozen (`ri.prediction_table`) |

**Order and stopping.** In `run.py:839–853` the gate values are written before they are enforced. Any failure raises an
incident before `frame_values` or `primary_statistics` runs (`run.py:854–858`). G6 is recomputed before the result
write. No gate failed, so the stopping path was not exercised in production; the tier-B tests cover it, and they were not
re-run here.

## 3. The primary statistics, recomputed (item 3)

For each frame and arm, `Δc(arm)` is the mean over the 79 scorable nouns of the arm's contrasts minus the stage-2 REF
contrasts. Its maximum difference from an `fsum` implementation is 1.8e-15. The fractions, `N_f`, `S_f`, `A_amp` and
`A_neg` were recomputed for every frame and equal the state's per-frame records bit for bit. So do P's `F̂_rem` and
`F̂_ins`, recomputed from the committed table's arm means; the table is internally consistent.

| quantity (`two`, 36 frames) | min | median | max |
|---|---|---|---|
| `Δc_full` | −6.581 | −4.945 | −3.947 |
| `F_rem` (the read part removed) | −0.170 | −0.009 | +0.083 |
| `F_rem_off` (an equal-length off-read slice removed) | 0.980 | 1.000 | 1.024 |
| `F_ins` (the read part installed) | 0.814 | 1.004 | 1.069 |
| `F_ins_off` | −0.090 | −0.033 | +0.007 |
| `N_f` | 0.917 (frame 06) | 1.006 | 1.166 |
| `S_f` | 0.824 (frame 11) | 1.035 | 1.123 |
| `A_amp` | −0.598 | −0.136 | +0.081 |
| `A_neg` | +0.197 | +0.922 | +2.490 |

- **D's eight `A_amp ≥ 0` frames** are 06 (+0.081), 09 (+0.012), 15 (+0.043), 22 (+0.023), 23 (+0.008), 32 (+0.065),
  34 (+0.024) and 36 (+0.061). The margin is 3 frames. Only one counted frame lies within 0.01 of zero (16, −0.009).
- **P's three disagreements:**
  - 03: rem error 0.0290 > 0.02515;
  - 08: ins error 0.0461 > 0.03511;
  - 09: rem error 0.0395.

  No frame was weak, measured or predicted, for any cue. The `WORST` and weak-frame paths therefore did not apply in
  production; every median is the plain even-n median, the mean of the 18th and 19th values.
- **The label path**, preregistered table, in order: the precondition is met; R and I pass; D passes; P is evaluable
  (both τ ≤ 0.20) and passes. The label is `READ_COMPONENT_EFFECTS_AS_PREDICTED`. The full per-frame table is in
  Appendix A.

## 4. The prediction freeze (item 4)

| time (UTC, 2026-10-14) | event |
|---|---|
| 13:08:48–13:09:26 | stage 1 (36 S1 keys only); the three files written. Their mtimes are still 13:09:26 |
| 14:02:01 | `f4e3a06` commits the stage-1 evidence, which records the files' sha256s and the table content `9191800440af…` |
| 17:35:06 | `59d8ec4` commits the three files, byte-identical to the record; it is pushed (`origin/exp101-phases` = `59d8ec4` in the launcher's record at 17:37:39) |
| 17:36:27–17:36:55 | the rehearsal's barrier dry run passes on these bytes |
| 17:38:22 | the confirm phase starts. The barrier re-reads the committed files, checks every digest, the layout and the metadata, and recomputes the table bit for bit. Then comes the S2 ledger write |
| after 17:38:22 | the first S2 forward |

- **Nothing was tuned after target results existed.** `src/` and `run.py` are unchanged since `40829d2` (2026-10-05).
  `read_intervention.py` is at blob `d7db6197…`, which is the lock's blob, HEAD's, and the imported file's.
  `git diff 3621eec 59d8ec4` touches only non-scientific paths: the lock files, the stage-1 files, READMEs, evidence and
  `research/synthesis/`.
- **No tolerance or threshold differs from the lock.** `validate_lock` passed at confirm, and it requires the module's
  constants, tolerances, statistics and wording to equal the lock's. The thresholds in the state are 26 and 25, 25, 27
  and 30. The state's τ is the lock's, which is the calibration file's.

## 5. The secondary records (item 5): no label force

Two recomputations reproduce every secondary statistic bit for bit, the bootstrap intervals included:
- this review's own code (one `[10,000 × 36]` index from `torch.Generator().manual_seed(20261001)`, the even-n median,
  ranks 250 and 9,750);
- the frozen functions (`run_records`, `secondary_statistics`).

The secondary block is absent from the state at the result write (§1, the `04221aa2…` reconstruction). The label is
computed by `primary_statistics` from the primary values alone, so nothing secondary can change it.

| record | value (median [95 % bootstrap interval]) |
|---|---|
| primary medians | N 1.0064 [0.9879, 1.0512]; S 1.0350 [1.0172, 1.0592]; A_amp −0.1360 [−0.2754, −0.0558]; A_neg 0.9223 [0.7266, 1.1252] |
| stratum `three` | R 36/36, I 36/36, `A_amp<0` 28/36, `A_neg>0` 36/36; N 0.9547 [0.9201, 1.0028]; S 1.0154 [0.9922, 1.0321]; F_rem 0.052, F_ins 0.994; PC-SWAP-PL 0.079, PC-SWAP-REF 0.999 |
| stratum `many` | R 36/36, I 36/36, **`A_amp<0` 23/36** (below the primary D threshold of 25), `A_neg>0` 36/36; N 0.7658 [0.7019, 0.7902]; S 0.9277 [0.8964, 0.9505]; A_amp −0.0608 [−0.1137, +0.0073]; F_rem 0.214, F_ins 0.936; PC-SWAP-PL 0.152 [0.090, 0.189]; additivity 1.134 |
| isotropic controls | `F_rem < median_j F_rem_iso` 36/36, margin 1.0128 [0.9796, 1.0427]; `F_ins > median_j F_ins_iso` 36/36, margin 0.9826 [0.9648, 1.0151] |
| head arms (full / value-only) | F_head_rem 0.1113 / 0.1089; F_head_ins 0.8984 / 0.8990. Removal share 0.8816 [0.8740, 0.8914] / 0.8774 [0.8613, 0.8870]; installation share 0.8943 [0.8862, 0.9063] / 0.8924 [0.8861, 0.9055]. No frame was excluded by a guard |
| midpoint | F_half_rem 0.7028 [0.6622, 0.7407]; F_half_ins 0.7166 [0.6651, 0.7619] |
| PC-SWAP | F_swap_pl 0.0441 [0.0097, 0.0611]; F_swap_ref 1.0076 [0.9895, 1.0212] |
| additivity `F_rem + F_ins` | 0.9885 [0.9525, 1.0170] |
| the program | P errors: rem 0.0060, ins 0.0113; `N̂ − N` 0.0004, `Ŝ − S` −0.0073. The table's own `Δr̂` lies within 0.0084 of the measured `Δr`, and its `ŝ/‖ô_off‖` within 0.0036 |
| manipulation check, ΔT along d̂_T at p_t relative to PL | NEC-READ −0.008; SUF-READ 0.979; NEC-OFF 1.026; SUF-OFF 0.003; **NEC-AMP 1.943**; SUF-NEG −0.891; NEC-HALF 0.499; SUF-HALF 0.484; PC-SWAP-PL −0.029; PC-SWAP-REF 0.951; NEC-ISO 0.99–1.00; SUF-ISO 0.001–0.016; HEAD-NEC −0.008, HEAD-SUF 0.979, the V variants −0.005 and 0.966 |
| LayerNorm records (`two`, medians) | read edits ‖γ₃⊙e‖/σ 7.44, ‖γ_ln2⊙e‖/σ 5.90; off-read edits 7.95, 6.08; isotropic 7.32, 5.74; σ(x′) 0.50–0.67 throughout. Every record reproduced exactly |
| ceilings (`two` arms) | the frozen readout fed the measured Δx3, compared with the measured Δc. The median error of the mean is at most 0.026 nats in magnitude for every arm. The maximum per-noun error reaches 0.71 (PL), 1.06 (SUF-READ, frame 21) and 0.96 (NEC-AMP), and the maximum error of the mean reaches 0.20 (PL), 0.49 (NEC-AMP) and 0.32 (SUF-NEG) (`out/g_ceiling_by_arm.json`) |

## 6. Interpretation (item 6)

### 6.1 What the label means, within the preregistered scope

**The site, the frames and the pair.** The scope is the residual entering layer 3 (`RESID_PRE.L3`) at the cue
position, in 36 new frames of one coordinated template (`NAME and NAME VERBED {cue} ADJ`), for the cue pair `one`/`two`,
on Pythia-70M-deduped at `e93a9faa`.

**The read coordinate.** The scalar `r(x) = ⟨x, w_c⟩` is the coordinate along `L03.H04`'s P1 read covector, which is
fixed by 011's lock.

**What R, I and D show.**
- Setting the plural prompt's `r` to its `one` value removes essentially the whole plural shift of `Δc`: `F_rem` has
  median −0.009.
- Installing the plural `r` in the `one` prompt produces essentially all of it: `F_ins` has median 1.004.
- Each of these beats an equal-length edit along the cue's own centred off-read change in every frame. That control
  leaves the shift intact (1.000) and installs nothing (−0.033).
- Doubling the read change adds plural shift in 28 of 36 frames, and reversing it adds singular preference in 36 of
  36. That is the opposite of disruption.

**What P shows.** The program's table was committed before any target prompt, and its `F̂_rem` and `F̂_ins` agree within
the calibrated tolerances in 33 of 36 frames.

**The result is non-trivial in size.** The read part is only 14 % (8.6–18.9 %) of the cue change's centred squared
length at `p_c`; its cosine with `ŵ` is 0.29–0.44. The off-read remainder, about 86 %, has close to zero net effect
through all readers combined. NEC-READ, which sets `x(p_c) = x_ref + off-read part`, gives −0.009; PC-SWAP-PL, which
sets `x(p_c) = x_ref`, gives 0.044.
- **In the plural prompt**, adding the remainder changes the fraction by −0.037 per frame (median; range −0.254 to
  +0.043). That is slightly toward singular, in 29 of 36 frames. It is a small instance of the opposition between the
  off-read part and the read part that design §4 (lines 151–152) cites from 010 for `this`, `a`, `another` and `every`.
- **In the reference prompt**, adding it changes the fraction by +0.008 (PC-SWAP-REF minus SUF-READ; range −0.060 to
  +0.071).

**What it supports.** A claim scoped to the read component at that site can now be reviewed for higher maturity.
That review is a separate decision (design §1, lines 39–41). The label alone does not change C002 or C003.

### 6.2 The non-claims and the claim hierarchy

- **The primary question** (R, I, D against the matched control) is answered positively within the scope.
- **The non-claims, as preregistered:**
  - no label says that the component is unused or has no causal role;
  - **no label assigns the effect to `L03.H04` alone**, because the residual edit reaches every reader of `x(p_c)`;
  - no label generalizes beyond the pair, this site, this checkpoint and these frames.
- **The secondary question**, whether `L03.H04` specifically mediates the effect, is answered descriptively only.
  - The head-only arms install about 89 % of the read component's effect through `L03.H04`'s output at `p_t` alone
    (installation share 0.894; value-only 0.892), and remove about 88 % (removal share 0.882; value-only 0.877). The
    intervals are tight.
  - The value-only variant, which keeps the measured attention row, is about as large as the full one. The head's key or
    attention change at `p_c` therefore contributes almost nothing; the effect runs through the value at `p_c` (P2's
    form).
  - About 11–12 % reaches `Δc` through other readers of `x(p_c)`: the other layer-3 heads, MLP3 and later layers at
    `p_c`, `L05.H07`, and block 5's path. The design does not separate them.
  - The fair summary is that `L03.H04` is the **predominant but not the unique** mediator. The hierarchy's sentence
    applies in its literal form: the effect "cannot be assigned uniquely to `L03.H04`". The preregistered head counts
    (36 of 36 each) carry almost no information (N1).
- **Generalization** is secondary and cannot rescue or alter the label.
  - `three`, which did not enter d̂_T, behaves like `two` on every statistic, D included (28 of 36).
  - `many` shows R and I in all 36 frames, but more weakly: `F_rem` 0.21, N 0.77, and 15 % of its shift does not pass
    through `p_c` at all. D's `A_amp < 0` holds in only 23 of 36 frames.
  - Under the primary rules, `many` would therefore read `…DIRECTION_NOT_ESTABLISHED`. This is descriptive only.

### 6.3 Simpler or competing explanations

| explanation | status | why |
|---|---|---|
| generic disruption of the cue's information | **excluded** | D: more plural under doubling (28 of 36) and more singular under reversal (36 of 36). The equal-length off-read and isotropic edits change nothing (margins about 1.0, 36 of 36) |
| the read edit is "louder" after LayerNorm | **excluded** | equal centred lengths by construction, and post-LN sizes comparable: ‖γ₃⊙e‖/σ is 7.44 for read edits against 7.95 for off-read and 7.32 for isotropic ones, so the control is, if anything, larger |
| a subspace-patching or "dormant direction" illusion | **largely excluded** by the complementary arms | the read part alone reproduces the whole-position swap (F_ins 1.004 against F_swap_ref 1.008), and the full natural remainder alone does nothing (F_rem −0.009 against F_swap_pl 0.044). The natural causal variation at `p_c` is captured by the projection on `w_c` |
| off-distribution edited states | **partly addressed, not excluded** | the swaps move `r` only to the other prompt's natural value, but the composite states are novel. The controls are equally artificial. The behaviour is graded (half dose 0.70, additivity 0.99). NEC-AMP and SUF-NEG go furthest. Nothing tests whether the composites lie on the data manifold |
| other readers of `x(p_c)`: MLP3, the other layer-3 heads, `L05.H07`, block 5 | **within the primary scope by design**; quantified by the head arms | about 11–12 % runs outside `L03.H04`'s output at `p_t`; it is not decomposed further |
| saturation | **explains D's weak `A_amp`**; does not threaten R or I | the head's transported signal nearly doubles under NEC-AMP (ΔT ×1.94) while `Δc` moves only −0.136. The half dose gives 70 %, not 50 %. R and I are not affected, because NEC-READ tests the off-read remainder from the unsaturated reference side |
| `w_c` defined in sample from `one`/`two` (d̂_T from `one`/`two` and `each`/`several` on exposed frames) | **not excluded for the primary claim**; out of frame, in pair | the stratum shows transfer to `three` and partial transfer to `many` |
| R and I follow partly from the definition of `w_c` | **true for the head-mediated route** | `w_c` is the covector through which the head writes onto d̂_T, so ΔT tracks `r` closely (NEC-READ −0.008, SUF-READ 0.979, NEC-OFF 1.026, SUF-OFF 0.003). The empirical content is threefold: (i) the head's output at `p_t` is the bottleneck for about 88–89 % of `Δc`; (ii) the off-read remainder has close to zero net effect through all readers combined; (iii) the effect is complete, not partial |
| P is a near recomputation | **by design** (§5, lines 212–215) | P checks that the frozen readout parts hold under artificial edits, a check of the program's fidelity rather than an independent mechanistic test. Per noun, the frozen readout's ceiling error reaches about 1 nat even with the exact measured `Δx3` (§5); agreement holds in mean fractions, where such errors largely cancel |
| site, layer, template or checkpoint specificity | **not tested** | there is no site or layer negative control, no cue-final frames, one checkpoint, and a one-dimensional manipulation check |
| the inferential population | **unstated** | the binomial tails treat the 36 frames of one template as exchangeable units. p ≈ 1.5e-11 is a statement about these frames under a sign-symmetric null, not a population rate |

### 6.4 The design §2 overshoot report

Confirm does not record this report (the lock review's N2). It is computed here from the saved `x3(p_c)`, by computation
2, for every frame and cue (`out/c_overshoot_per_frame.json`).

| cue | frames flagged (`‖off_c‖ < s`) | s/‖off_c‖ min | median | max | Δr range |
|---|---|---|---|---|---|
| `two` | **0 of 36** | 0.306 (frame 36) | 0.408 | 0.483 (frames 05, 31) | 1.648–2.320 |
| `three` | 0 of 36 | 0.238 | — | 0.382 | 1.469–2.063 |
| `many` | 0 of 36 | 0.252 | — | 0.349 | 1.357–2.052 |

- **For `two`:** `s` has median 2.55 and ‖off_c‖ median 6.24.
- **The controls never overshoot.** NEC-OFF and SUF-OFF move 31–48 % of the centred off-read change for `two`.
- **The program agrees.** It predicted no overshoot, and its predicted ratios lie within 0.0036 of the measured ones.

The per-frame values are in Appendix A.

## 7. Findings, most severe first

**Blocker:** none.

### Should-fix (reporting; none changes the label)

**SF1. The design's per-frame overshoot report is not recorded or rendered.**

Design §2 (`…-design.md:73–75`) requires it: "s/‖off_c‖ is reported for every frame". So does plan computation 2
(`…-plan.md:211`): "frames with ‖off_c‖ < s are flagged, and s/‖off_c‖ is recorded".

Confirm computes the measured pairs but stores none of their records:
- `read_intervention.py:1234` and `1356` compute the pairs;
- `Pair.record()` exists at `read_intervention.py:618–620`;
- `run.py:854–858` and `render_report` (`read_intervention.py:2095–2156`) carry no pair record.

**The scenario.** A reader of `report.md` cannot tell whether the off-read control ever overshot. An overshooting frame
would remove more than its whole off-read change in NEC-OFF and inflate `N_f`.

**The fix.** The closure should add §6.4's table, or Appendix A's per-frame column: none overshoots, and the ratios run
0.306–0.483 for `two`. It should cite this review as the source, since the state has no such record.

**SF2. The ceilings and E₁ are recorded but not reported, and their meaning for P is not stated.**

The requirements:
- design §5 (`…-design.md:216–217`): "Each arm's ceiling … is reported";
- design §8 (line 326): "reported per arm";
- plan line 502: E₁ "recorded per run".

What is implemented:
- the state keeps only the per-run mean ceilings (`confirmation.ceilings`) and a single overall maximum of each of
  `ceiling_error` and E₁ (`read_intervention.py:1281`, `1316–1317`, `1404–1408`);
- E₁ is therefore not per run, and `render_report` prints neither.

**The scenario.** A reader sees "P PASS" and E₁ 0.0217 above 020's 7e-3 tolerance with no context. A reader might cite
"the program predicts each noun's contrast", yet the frozen readout's per-noun ceiling error reaches 1.06 nats on
SUF-READ@two and 1.20 on NEC-READ@many.

**The fix.** The closure should report three things:
- the per-arm ceilings and ceiling errors, from §5 or `out/g_ceiling_by_arm.json`;
- E₁ per arm, from `out/f_e1_by_arm.json`: at most 0.0066 for every arm except SUF-OFF (up to 0.0151) and SUF-ISO
  (0.0217). Those two exceed 7e-3 only because their ‖Δh6‖∞ is 0.08–0.15, which is the reason the reviewer's G4 decision
  gave for not gating E₁;
- the sentence of design §5 that P is close to a recomputation, so that "AS_PREDICTED" is read as program fidelity under
  edits, at the level of mean fractions.

### Notes

**N1. The preregistered head counts are near-vacuous criteria.** The criteria are `F_head_rem < 1` and
`F_head_ins > 0` (`read_intervention.py:269–270`, `1707–1710`), and `report.md` will show 36 of 36 for each.
- **The scenario.** A reader takes "36 of 36" as strong head-specific evidence, when any nonzero contribution of the head
  satisfies the criterion.
- **The fix.** State the head-specific result through the shares: about 0.88 for removal and 0.89 for installation,
  with tight intervals, in both variants. Say "predominantly, not uniquely, `L03.H04`".

**N2. Say plainly in the closure that `many` would fail D under the primary rules.** It has `A_amp < 0` in 23 of 36
frames, and its interval for A_amp includes 0. Its R and I are weaker (N 0.77; F_rem 0.21), and 15 % of its shift
bypasses `p_c`.
- **Where it is hidden.** `render_report` prints the stratum only as a raw dict (`read_intervention.py:2149–2152`), so
  this is easy to miss.
- **What it supports.** Generalization: full to `three`, partial to `many`. It has no label force, and the stratum cannot
  rescue or alter the label.

**N3. D is the narrowest family.** `A_amp` passes by 3 frames (28 against 25), and the eight non-negative values are
small (≤ +0.081 nats). The manipulation check (ΔT ×1.94 under NEC-AMP) shows the cause is saturation of `c`, not a
missing directional effect. A_neg (36 of 36, median +0.92) carries the direction. The closure should say so, citing
design §13's saturation risk (lines 462–464).

**N4. The log's "results sha256" is not a file digest.** The run's log line labels as "results sha256" `04221aa2…`
(`run.py:881–882`). That is the canonical state content digest at the result write, which is what
`rr.write_state_atomic` returns, not a file sha256.
- **The scenario.** In evidence, a reader compares it with the file sha256 `adc83b1a…` or the final `state_sha256`
  `1e3eaeb3…`, finds three different digests, and suspects that the state changed after the result.
- **The fix.** The evidence README should name all three digests. It should state that `04221aa2…` is reproduced exactly
  by the final state minus `confirmation.secondary`, which proves that the secondary step changed nothing else.

**N5. `phases.stage1.awaiting_commit` is still true in the final state.** The flag is set at `run.py:721` and nothing
clears it.
- **The scenario.** A reader of the state infers that confirm ran on uncommitted stage-1 files.
- **The facts.** The files were committed in `59d8ec4` and pushed before confirm, and the launcher's preflight checked
  that. The closure should state it. Cosmetic.

**N6. `render_report` prints the secondary statistics as raw Python dicts** (`read_intervention.py:2149–2152`). The
closure should tabulate them: §5 here can be reused.

**N7. Stated limitations that the label's scope does not make explicit should appear in the closure:**
- no site or layer negative control;
- a one-dimensional manipulation check;
- one template and one checkpoint;
- the frames treated as exchangeable units, so p-values are not population rates;
- the in-pair definition of `w_c`;
- composite edited states whose naturalness is untested.

**N8. Preserve the confirm records in evidence.** The confirm launcher and its records are still only in the
scratchpad (`exp101/confirm/`). They should go into the repository's evidence with SHA256SUMS, as for the earlier
phases. That includes `production_runs.jsonl`, the run records and `outputs_after_confirm.sha256`, and the launcher
`56a687f3…` with v1 as reviewed.

**N9. The smoke ran spent exposed keys once more.** It ran the 55 keys of the spent exposed frame
`coordinated-adjective-1` (020's pool) again, outside every 101 ledger, bit-equal to the unguarded reference. This is
engineering only, with no bearing on 101. Record it with the calibration review's N4 count.

**N10. `README.md` is stale.** It still says "STAGE 1 COMPLETE" (`README.md:3`), and its section "Report and closure"
says "Not yet" (lines 127–129). Both should be updated at closure. `results.json` has mode 0600 from `mkstemp`, which is
cosmetic.

## Appendix A: per frame (cue `two`, with the overshoot ratios of all three cues)

| frame | Δc_full | F_rem | F_rem_off | F_ins | F_ins_off | N | S | A_amp | A_neg | s/‖off_c‖ (two) | s/‖off_c‖ (three, many) | P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | -4.463 | +0.072 | 1.011 | 0.980 | -0.017 | 0.939 | 0.997 | -0.177 | +1.017 | 0.423 | 0.360, 0.317 | yes |
| 02 | -5.078 | +0.064 | 0.997 | 0.972 | -0.020 | 0.933 | 0.992 | -0.088 | +0.826 | 0.441 | 0.356, 0.335 | yes |
| 03 | -4.226 | +0.063 | 1.003 | 1.051 | -0.034 | 0.940 | 1.085 | -0.086 | +1.652 | 0.435 | 0.357, 0.330 | no |
| 04 | -4.642 | -0.170 | 0.996 | 1.033 | -0.089 | 1.166 | 1.121 | -0.598 | +1.646 | 0.336 | 0.282, 0.283 | yes |
| 05 | -4.968 | +0.003 | 1.000 | 0.959 | -0.018 | 0.997 | 0.976 | -0.096 | +0.919 | 0.483 | 0.372, 0.315 | yes |
| 06 | -6.581 | +0.062 | 0.980 | 0.938 | -0.018 | 0.917 | 0.955 | +0.081 | +0.792 | 0.470 | 0.360, 0.331 | yes |
| 07 | -4.496 | -0.113 | 1.007 | 1.036 | -0.081 | 1.119 | 1.116 | -0.074 | +1.513 | 0.349 | 0.256, 0.301 | yes |
| 08 | -4.442 | +0.003 | 0.997 | 0.918 | -0.048 | 0.994 | 0.965 | -0.330 | +1.734 | 0.406 | 0.334, 0.303 | no |
| 09 | -4.353 | -0.151 | 0.993 | 1.013 | -0.083 | 1.144 | 1.096 | +0.012 | +0.625 | 0.363 | 0.263, 0.274 | no |
| 10 | -5.734 | -0.011 | 0.989 | 0.925 | +0.006 | 1.001 | 0.919 | -0.487 | +0.197 | 0.446 | 0.350, 0.324 | yes |
| 11 | -5.225 | +0.047 | 0.991 | 0.814 | -0.010 | 0.944 | 0.824 | -0.275 | +0.306 | 0.409 | 0.327, 0.297 | yes |
| 12 | -5.405 | -0.020 | 1.011 | 1.007 | -0.026 | 1.031 | 1.033 | -0.454 | +0.731 | 0.403 | 0.333, 0.333 | yes |
| 13 | -3.947 | -0.055 | 0.985 | 0.983 | -0.038 | 1.040 | 1.022 | -0.041 | +1.862 | 0.405 | 0.363, 0.315 | yes |
| 14 | -5.884 | -0.085 | 1.010 | 1.050 | -0.048 | 1.095 | 1.099 | -0.568 | +0.662 | 0.379 | 0.303, 0.291 | yes |
| 15 | -4.759 | +0.003 | 0.999 | 0.991 | -0.021 | 0.996 | 1.012 | +0.043 | +1.468 | 0.406 | 0.356, 0.329 | yes |
| 16 | -4.709 | -0.086 | 0.994 | 0.912 | -0.082 | 1.080 | 0.993 | -0.009 | +2.490 | 0.358 | 0.297, 0.292 | yes |
| 17 | -4.157 | -0.093 | 1.015 | 0.967 | -0.066 | 1.108 | 1.033 | -0.512 | +0.629 | 0.412 | 0.358, 0.324 | yes |
| 18 | -5.574 | -0.133 | 0.994 | 0.975 | -0.055 | 1.127 | 1.029 | -0.056 | +1.720 | 0.341 | 0.238, 0.252 | yes |
| 19 | -4.686 | +0.061 | 1.002 | 1.016 | -0.035 | 0.941 | 1.051 | -0.166 | +1.172 | 0.452 | 0.382, 0.336 | yes |
| 20 | -6.349 | +0.053 | 1.009 | 0.982 | -0.005 | 0.956 | 0.988 | -0.302 | +0.854 | 0.438 | 0.349, 0.318 | yes |
| 21 | -5.872 | +0.035 | 1.013 | 1.050 | +0.007 | 0.978 | 1.043 | -0.441 | +0.390 | 0.411 | 0.319, 0.349 | yes |
| 22 | -5.919 | +0.042 | 1.000 | 0.975 | -0.017 | 0.957 | 0.992 | +0.023 | +0.512 | 0.410 | 0.324, 0.292 | yes |
| 23 | -6.494 | -0.063 | 1.010 | 1.016 | -0.041 | 1.073 | 1.057 | +0.008 | +0.968 | 0.360 | 0.277, 0.323 | yes |
| 24 | -5.852 | +0.012 | 1.000 | 1.006 | -0.011 | 0.988 | 1.017 | -0.152 | +1.081 | 0.391 | 0.282, 0.305 | yes |
| 25 | -4.261 | +0.075 | 1.021 | 1.021 | -0.016 | 0.946 | 1.037 | -0.286 | +0.926 | 0.406 | 0.331, 0.322 | yes |
| 26 | -5.861 | +0.082 | 1.002 | 1.001 | -0.009 | 0.919 | 1.010 | -0.032 | +0.996 | 0.419 | 0.353, 0.323 | yes |
| 27 | -4.724 | +0.008 | 1.007 | 1.028 | -0.031 | 0.999 | 1.059 | -0.120 | +1.187 | 0.400 | 0.285, 0.300 | yes |
| 28 | -4.921 | -0.007 | 1.024 | 1.051 | -0.011 | 1.031 | 1.063 | -0.345 | +0.661 | 0.383 | 0.316, 0.317 | yes |
| 29 | -4.731 | -0.114 | 1.015 | 1.069 | -0.054 | 1.129 | 1.123 | -0.207 | +0.608 | 0.393 | 0.338, 0.304 | yes |
| 30 | -5.307 | -0.017 | 1.020 | 1.014 | -0.032 | 1.037 | 1.045 | -0.510 | +1.079 | 0.390 | 0.281, 0.295 | yes |
| 31 | -6.522 | -0.018 | 0.994 | 1.001 | -0.029 | 1.012 | 1.029 | -0.033 | +0.424 | 0.483 | 0.357, 0.344 | yes |
| 32 | -5.047 | -0.104 | 0.992 | 1.048 | -0.039 | 1.095 | 1.087 | +0.065 | +0.366 | 0.420 | 0.350, 0.325 | yes |
| 33 | -5.525 | -0.051 | 1.014 | 1.041 | -0.046 | 1.065 | 1.087 | -0.314 | +0.858 | 0.414 | 0.301, 0.317 | yes |
| 34 | -4.626 | +0.030 | 0.983 | 1.000 | -0.061 | 0.952 | 1.061 | +0.024 | +0.653 | 0.451 | 0.333, 0.290 | yes |
| 35 | -4.909 | -0.020 | 0.999 | 1.026 | -0.069 | 1.019 | 1.095 | -0.153 | +1.228 | 0.437 | 0.368, 0.338 | yes |
| 36 | -4.535 | -0.130 | 0.990 | 0.987 | -0.090 | 1.121 | 1.078 | +0.061 | +2.113 | 0.306 | 0.240, 0.270 | yes |

## Appendix B: files in this directory

- `PROMPT.md`: the brief. `REVIEW.md`: this report.
- `scripts/`:
  - `rguard.py`, `snapshot.sh`;
  - `a_integrity.py`, `b_primary.py`, `c_gates_independent.py`, `d_frozen_crosscheck.py`, `e_secondary.py`,
    `f_e1_by_arm.py`, `g_ceiling_by_arm.py`.
- `out/`:
  - each script's `.json` and `.log`, and its `_guard.json`;
  - `b_primary_values.json` (all per-frame values, all cues), `c_overshoot_per_frame.json`, `per_frame_table.md`;
  - `repo_snapshot_start.txt` and `repo_snapshot_end.txt`, identical;
  - `inline_pt_listing_guard.json`, the guard record of a one-line listing of the measurement file's tensor names.
- `SHA256SUMS`: every file above.

Every check in `out/*.json` passes. One check in an early run of `c_gates_independent.py` compared the contrast-level
logit consistency with an arbitrary 1e-3 and reported the known float32 floor of 3.34e-3 as a failure. The check now
uses the frozen tolerance of 2e-2, and the script was re-run; its comment says so.
