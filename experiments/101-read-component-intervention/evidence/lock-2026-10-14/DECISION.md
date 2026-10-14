# Experiment 101 — the lock decision (2026-10-14)

**For the user.** The design reserves this step for you: "The user commits the lock" (design §9). Prepared by the
operator (Claude, Opus 5.5). Every review cited here is an AI review by a separate agent, not a human validation.

## 1. Where the experiment stands

| phase | status | evidence |
|---|---|---|
| freeze (tokenizer only) | ran once at `40829d2`; installed in `a857f9a`; AI review PASS WITH NOTES, no blocker | `../freeze-2026-10-14/`, `../freeze-review-2026-10-14/` |
| calibrate (exposed, exploratory; spent data) | ran once at `8f65062`; no revision trigger; installed in `d502777`; AI review PASS WITH NOTES, no blocker | `../calibrate-2026-10-14/`, `../calibration-review-2026-10-14/` |
| lock (weights only) | ran once at `3621eec`; candidates written, **not installed**; AI review PASS WITH NOTES, no blocker to committing the lock | this directory, `../lock-review-2026-10-14/` |
| stage 1, confirm, report | not run | — |

No 101 frame prompt has run; the confirmation ledger is empty. All commits are on the local branch `exp101-phases`;
nothing has been pushed.

## 2. The G4 question, resolved with evidence

- **The form** was decided by the reviewer on 2026-10-05 (plan § G4's form and the reviewer's decision, recorded in
  `1e48c1c`). That commit is local: the GitHub copy of the plan (`96f0cc8`) still shows the question as pending. The
  two frozen rules: calibration may test the tolerances but never widen them; G4 is an integrity gate, not an outcome
  filter.
- **The calibration tested the form** on the 36 exposed coordinated frames: 1,800 G4 runs. Every frozen tolerance held,
  so no trigger fired and no amendment arose:

  | G4 quantity | tolerance | exposed maximum | margin |
  |---|---|---|---|
  | `x3(p_t)` unchanged | bit for bit | 1,692 / 1,692 | — |
  | 023's I3 | 1e-4 relative | 1.18e-5 | 8.5× |
  | 023's I4 | 1e-3 nats | 3.44e-5 | 29× |
  | 020's Level-1 (contrast form) | 2e-2 nats | 1.065e-2 (one run, `NEC-HALF@many`, frame `018-3`); `two`'s arms ≤ 5.01e-3 | 1.88× |
  | 020's readout identity | 2e-2 | 4.87e-3 | 4.1× |
  | 020's logit identity | 2e-2 | 3.03e-3 | 6.6× |

- **The residual risk.** The calibration review traced the two Level-1 outliers to float32 rounding in the measured
  blocks 4–5 (020's attention-score floor), not to the edit (Spearman with the edit size −0.02): the intervention adds no
  new floor, which is the condition the reviewer's decision named for an amendment. Its estimate of the chance that one
  of confirm's ~1,800 G4 runs exceeds 2e-2: **about 0.3 %** (0.02–1 % across methods; about 10 % at the 95th percentile
  of a heavy-tailed frame bootstrap that leans on the single frame `018-3`). If it happened, the confirmation would be
  `NOT_INTERPRETABLE` and the 36 frames spent (rule 2); it could never produce a false positive.
- **Recommendation: no amendment; keep G4 as frozen.**
  - The central risk is small, and the record agrees: three campaigns (020, 025, this calibration) topped out at
    0.0107–0.0119 for this identity.
  - A wider tolerance would weaken the check that catches wiring errors, which is what G4 is for.
  - An amendment is not available inside this protocol version anyway: `ri.TOLERANCES` is bound by the installed
    calibration (module blob and tolerances), and calibrate runs once. Changing it means a new protocol version and a
    new calibration, for a ~0.3 % risk.

## 3. What the exposed calibration shows (exploratory; spent data; no force)

On the 36 exposed frames, for `one`→`two` (medians, with ranges):
- **Removal.** Setting the plural prompt's read coordinate to the reference's leaves `F_rem` −0.035 (−0.45 to 0.16) of
  the shift; removing an equal centred length along the frame's own off-read change leaves `F_rem_off` 0.995 (0.976 to
  1.024).
- **Installation.** Installing the read part alone into the reference prompt gives `F_ins` 0.980 (0.886 to 1.076); the
  equal-length off-read installation gives `F_ins_off` −0.041.
- **Direction.** Doubling the read change adds plural shift in 30/36 frames (`A_amp` median −0.145: the plural prompt
  is near saturation); reversing it adds singular preference in 36/36 (`A_neg` median 1.17 nats).
- **The program.** Its predicted fractions agree with the measured ones within τ in 31/36 frames
  (τ_rem 0.025, τ_ins 0.035; tight).
- **Secondary, descriptive.** About 88–89 % of the read edit's effect passes through `L03.H04`'s output at `p_t` (the
  head-only arms; the value-only variant is nearly identical). The eight isotropic controls change little: `F_rem_iso`
  0.92 to 1.03 and `F_ins_iso` −0.21 to 0.15, against `F_rem` and `F_ins` near 0 and 1. The stratum
  behaves alike for `three`; `many` has a weaker removal (`F_rem` median 0.21) and `A_amp` < 0 in 26/36.

**Caveats.** These frames are in-sample for the variable: `d̂_T`, hence `w_c`, was estimated on them, and some entered
012's bases. The design forbids any revision on these sizes, and none was made. The 36 frozen frames are new.

## 4. What the confirmation can and cannot add

- **It can add the project's first preregistered, outcome-bearing causal evidence on a decoded variable** (removal,
  installation and their direction, against the matched off-read control), scoped to the read component at
  `RESID_PRE.L3`, `p_c`, coordinated frames, `one`/`two`. Exposed power at these shares: R and I about 1.0, D 0.99,
  P 0.98.
- **Where the information is.** If the new frames behave like the exposed ones (which are in-sample for `d̂_T`), the
  primary families are very likely to pass; the more
  informative parts are P (the frozen program predicting two artificial edits within τ ≈ 0.03 on new frames), D (near
  saturation), the head-only arms (how much is `L03.H04`'s), and the stratum. An independent AI survey of the record
  (session memo, not in the repository) also flags limits no label can cross: no site or layer negative control (the
  same `w_c` edit at another layer or position), a one-dimensional manipulation check, no rowspace or
  dormant-direction diagnostics for the readers other than the head, and an inference population (one hand-written
  frame list) that is not stated. These are candidates for analyses after 101 closes, not changes to 101.
- **It cannot show** that the component has no causal role (no equivalence bound), anything beyond `one`/`two` in
  coordinated frames at this site and checkpoint, or that `L03.H04` alone mediates the effect.

## 5. The decisions requested

1. **The lock.** Read `outputs/experiment-101/candidate-preregistration.md` (8,703 bytes). If you accept it, install
   both candidates byte-identically and commit them (the commands below). The lock phase runs once: a different lock
   needs a new protocol version.
2. **Push.** `exp101-phases` holds `1e48c1c`…`40829d2` (main's six unpushed commits) and the phase commits. Pushing
   anchors the preregistration chain before any target data exist (the freeze review's N4).
3. **Stage 1, then confirm.** When you authorize it, `stage1` runs only the 36 reference prompts and writes
   `stage1/states.f64`, `stage1/states.json` and `stage1/table.json` (the program's prediction table); **you commit
   those three files** (the prediction freeze); then `confirm` runs once, and `report`.
   - A guarded stage-1 launcher is prepared (`prepared/launch_stage1.py`). The lock review read version 1 and found
     three should-fix points (S1–S3); version 2 applies them and N8, and passed an isolated test of its hook decisions
     (`prepared/hook_self_test.out.json`), but it has **not been re-reviewed or run**. A short re-check before stage 1
     is recommended.
   - **Keep the environment fixed until confirm** (the lock review's N9): stage 1 and confirm recompute the geometry
     and compare it with the lock bit for bit, and a mismatch is a phase incident that spends the protocol version.
     Avoid OS, Python or package updates on this machine before then.
   - The lock evidence (`lock_run.json`) must stay committed: the stage-1 launcher reads it (the lock review's N5).

### The commands (lock install; run from the repository root, on `exp101-phases`)

```bash
cp -p outputs/experiment-101/candidate-lock.json experiments/101-read-component-intervention/preregistration-lock.json
```

```bash
cp -p outputs/experiment-101/candidate-preregistration.md experiments/101-read-component-intervention/preregistration.md
```

```bash
shasum -a 256 experiments/101-read-component-intervention/preregistration-lock.json experiments/101-read-component-intervention/preregistration.md
```

The two digests must be `8dc6e6d38c876a2dd9bf905fbeaf36b774b487967597795beb032bb4a0751087` and
`3761b8130203d5d15ac0e30055aa135626a6d97f0ab807df6cf4d31b6af8e980`.

```bash
git add experiments/101-read-component-intervention/preregistration-lock.json experiments/101-read-component-intervention/preregistration.md && git commit -m "data: experiment 101 lock — preregistration-lock.json and preregistration.md, installed byte-identically"
```

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/101-read-component-intervention/run.py validate
```

`validate` should end with "lock and preregistration verified".
