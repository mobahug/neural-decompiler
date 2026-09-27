# Experiment 025: Does Moving the Cue Embedding Along the Frozen Nounness Direction Causally Change the Frozen Readout Error and the Attention Routing?

**Status (2026-09-27): CLOSED.** Every phase ran exactly once:

| phase | status |
|---|---|
| freeze | complete, reviewed |
| lock | complete, reviewed |
| confirm | complete, reviewed |
| report | complete, verified |
| closure | complete |

**The official result: `CAUSAL_EFFECT_NOT_ESTABLISHED`.**

| statistic | positive | threshold | result |
|---|---|---|---|
| A | 26/40 | 27 | FAIL |
| B | 23/40 | 27 | FAIL |
| G | 37/40 | 27 | PASS as a statistic; descriptive only, because under the frozen hierarchy it cannot change the label |

The reference tail is 0.01923865414210013.

**The preregistered reading:** the preregistered Experiment 025 test did not establish that moving fresh cue embeddings
in the +nounness direction consistently causes greater frozen block-4/5 readout-approximation error than the matched
−nounness intervention across the frozen 40-cue population.

See [Report and closure](#report-and-closure-2026-09-27) and the [evidence index](evidence/README.md).

**Experiment 025 is the endpoint of this sequence.** No Experiment 026 follows.

The phase history:
- **The implementation** is complete and independently reviewed: PASS WITH NOTES, no blockers. The fixes run through
  `c765148fdb612ae249e21ef3c830e4ceb1d12dd0`. The final test gate on that commit: tier A 542 passed, tier B 575
  passed, tier C 6 passed, 0 failed.
- **The production freeze** ran exactly once, at `c765148`, and was independently verified: PASS WITH NOTES, no
  blockers. It is installed byte-identically as `confirmation-v1.json` (`57f1f98`):
  - file sha256 `54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a`;
  - content sha256 `6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4`;
  - 90,720 condition-tagged keys, manifest sha256 `0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61`.
- **The production lock** ran exactly once, at `d3ecbd6feb1bd2aefe08dc0f8d799815be47c519`, weights only (0 prompts,
  forwards, captures or interventions). It was independently reviewed: PASS WITH NOTES, no blockers. The geometry,
  the 280 random controls (float32 neutrality max 6.09e-9 against 1e-6) and the `D_attn` reference rows were
  reconstructed independently. It is installed byte-identically (`5ee7a74`):
  - `preregistration-lock.json`: file sha256 `1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be`,
    content sha256 `8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb`;
  - `preregistration.md`: sha256 `e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919`.
- **The production confirm** ran exactly once, on 2026-09-26, at `6f72d7d`, with exit status 0 and no incident; see
  `evidence/confirm-2026-09-26/`. The independent confirmation review: PASS WITH NOTES, no blockers; see
  `evidence/confirmation-review-2026-09-26/`.
- **The production report** ran exactly once, on 2026-09-27, at `c8d9fad`, with exit status 0; it loaded no model and
  ran no prompt. Its output is preserved byte-identically as
  [`evidence/final-report-2026-09-26.md`](evidence/final-report-2026-09-26.md) (`ad9f85e6…`).
- **The evidence** is under [`evidence/`](evidence/README.md).

Implements the design
[`docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md`](../../docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md)
(revision 1, `c0885e5`, corrected in `26c9925`) through the plan
[`docs/superpowers/plans/2026-09-26-experiment-025-nounness-direction-intervention-plan.md`](../../docs/superpowers/plans/2026-09-26-experiment-025-nounness-direction-intervention-plan.md)
(revision 1, `7d90d28`). It includes the confirm-time patch-path check, which the reviewer signed off.

This is the final experiment of the sequence. After it, on any clean result, the project stops and is written up.

The question: Experiment 024 showed that a frozen, weight-derived nounness score *predicts* the frozen block-4/5
readout error. Does *moving* a fresh cue's input embedding along that score's own direction *cause* the error, and the
directly captured L4/L5 routing, to change in the predicted direction?

- **The intervention:** a norm-preserving rotation of the cue's input embedding at `("EMBED", p_c)`.
  - `d = μ̂_noun − μ̂_cue` is 024's frozen direction, from 024's calibration record.
  - `R(±θ) = |E|·(cos θ·Ê ± sin θ·t̂)`, where `t̂` is the unit tangent part of `d` at `Ê` and `θ = asin(odd/τ)`.
  - The directional (odd) score component is exactly ±0.32 (primary) and ±0.16 (secondary). The complete change is
    `s₀(cos θ − 1) ± odd`.
- **The controls:** 7 deterministic random tangent directions per cue, at the same angle.
  - They are orthogonal to `Ê` and `t̂`, so they are nounness-neutral.
  - They come from a SHA-256 counter-mode stream (tag `025|control|{token_id}|{j}`) through Box–Muller in float64.
  - A plurality control is secondary.
- **The 21 conditions per pair:** `base`, `noun±0.32`, `noun±0.16`, `rand1…7±0.32` and `plur±0.32`.
  - The 16 outcome-bearing ones are `noun±0.32` and `rand1…7±0.32`.
  - 40 cues × 108 exposed frames × 21 conditions = 90,720 runs, each under a condition-tagged key
    `frame_id|word|token_id|condition`.
- **The measurements**, per cue and condition:
  - `ℓ = log(Σ SSE_C / Σ n)` over the 108 frames × 79 nouns;
  - `D_attn`, the total-variation distance of the captured L4/L5 rows at `p_t` from 020's locked rows, averaged
    equally over the 16 heads, then over the frames.
- **The outcome-bearing statistics**, each a sign count that passes at ≥ 27 of 40 strictly positive (ties and zeros
  count against; the exact Binomial(40, ½) reference tail is 0.01924):
  - `A_i = ½[ℓ(+θ) − ℓ(−θ)]`: causal direction, + against −;
  - `B_i = A_i − mean_j |A_ij|`: specificity against the absolute random-control effects;
  - `G_i = ½[D_attn(+θ) − D_attn(−θ)]`: the routing summary moves in the predicted direction.
- **The outcome**, in the frozen hierarchy:
  - `NOT_INTERPRETABLE`;
  - `CAUSAL_EFFECT_NOT_ESTABLISHED`;
  - `DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC`;
  - `READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED`;
  - `NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR`.

  The stratum counts (20 adjectives, 20 ordinary singular nouns), the magnitudes, the baselines, the half dose, the
  plurality control and the block-4/5 ladder are descriptive. They run after the result is written, and can never
  alter it.
- **The population:** 40 fresh cues, picked mechanically.
  - The adjectives are the first 20 of 024's adjective reserves.
  - The nouns are 024's 8 eligible ordinary reserves, then the first 12 eligible words of a frozen alphabetical
    39-word list.
  - The tightened prior-noun rule also excludes the 24 nouns of 020's confirmation file, which 021 read; that
    excludes `statue`.

## Code

- **Module:** `src/neural_decompiler/cue_rotation.py` (`cr`).
  - It calls 024's `readout_routing.py`, 023's `block0_completion.py` and 022's `upstream_localization.py`, pinned by
    git blob with 022's ten.
  - It binds 024's reviewed calibration record, lock and freeze, and 020's confirmation file, by file and content
    digest.
  - The production configuration is literal and immutable: 20 + 20 cues, 7 controls, the doses 0.32 and 0.16, and
    the threshold 27, derived from the Binomial tail. A test world can only pass its own configuration explicitly.
- **Runner:** `experiments/025-nounness-direction-intervention/run.py`.
  - Phases: `validate`, `freeze`, `lock`, `confirm` and `report`. There is no calibration phase.
  - It takes no option and always runs the production configuration.
- **Tests:**
  - `tests/test_cue_rotation.py` holds tier A (pure) and the tier-C contracts (the pinned model, opt-in, spent
    prompts only).
  - `tests/test_experiment_025_runner.py` holds tier B (the six-layer fake, on 024's locked fake world).

## Phases (each only when separately authorized)

1. **`freeze`** (tokenizer only) writes `confirmation-v1.json`, which is committed by hand and reviewed.
   - A shortfall, or picks other than the design's expected 40, writes nothing and creates no state.
2. **`lock`** (weights only; every forward refused) writes the candidate `preregistration-lock.json` and
   `preregistration.md`. They are installed byte-identically, committed and reviewed.
   - It computes the direction, each cue's geometry, the controls, the plurality tangent and all 840 patched vectors.
   - It enforces every geometry gate: float64 within 1e-12; the float32-patched vectors within 1e-6; the θ = 0
     vector equals the model's row bit for bit.
3. **`confirm`** (once; never resumed):
   - `validate_lock`, including the direct module-blob check, then the model digests;
   - I7′ before any prompt: the lock's geometry recomputed bit for bit, and every geometry gate;
   - the patch-path check on four already-executed keys. It compares a plain capture with a patched θ = 0 run, and
     the embedding hook with the block-0 residual input, bit for bit. It records equality and digests only, outside
     the ledger, the manifest and the accounting;
   - the ledger of all 90,720 keys, then every run once, with the measurements saved first. Stage 2 executes frame by
     frame, as 022 and 024 did; the tensors are stored in `ul.table_units` order, and the order of independent
     forwards changes no value;
   - the accounting, and `C` recomputed from the saved `Δx3` bit for bit;
   - I1, I3 and I4 on every run, and the Level-1 identity on the 69,120 outcome-bearing runs;
   - the result, with every cue's per-condition responses, and the completed phase in one atomic write;
   - then the descriptive records.
4. **`report`**.

An I7′ or patch-path failure is an incident before the ledger. Any failure after the ledger is an incident that
carries no result. The measurements are kept (if stage 2 itself fails, the runs measured so far, saved after the
incident is on disk), and the outcome is `NOT_INTERPRETABLE`.

- **What the patch-path check can and cannot show.** It checks the θ = 0 plumbing only: a patched run with the
  model's own row reproduces the plain run, and the embedding hook equals the block-0 residual input. It cannot show
  that a rotated vector propagates. That is shown on every run by the vector I1 gate: `d_emb + ΔE + ΔA0` against the
  measured `Δx1`, with `d_emb` the rotated vector's change. A patch that did not land would miss I1 by orders of
  magnitude. A pass of the check does not strengthen the scientific result.
- **Runtime.** The estimate is about 2.5–3.5 hours on this machine: 90,720 patched forwards, the identity gates, and
  the Level-1 identity on every run. The ladder adds about 26,000 more Level-1 evaluations after the result.
- **The production confirm** runs through an external guarded launcher, as in 024. It must live under `evidence/`, or
  it counts as a scientific-path change. The launcher:
  - logs each patched run's condition, for example by mapping the replacement's digest to the lock's vectors, since
    `run_patched` sees untagged prompt keys;
  - keeps the four spent-key patch-path runs apart from the fresh runs.

The local outputs stay in `outputs/experiment-025/`.

## Protocol notes

- **The freeze's integrity reads (the reviewer's decision on the freeze review's N3).** The freeze implementation
  performs shared dependency/integrity reads beyond the minimal phase-table inputs. Independent call-flow verification
  established that these data do not feed population selection or manifest construction. Freeze selection remains
  tokenizer/list/exclusion based.
  - The reads are 020's local results, 023's committed cells input, and 024's calibration record and lock.
  - They were accepted as integrity and precondition reads only. The code was not changed, and the reviewed freeze
    remains authoritative.
- **The tokenizer.** The pinned checkpoint and tokenizer family is `EleutherAI/pythia-70m-deduped` (revision
  `e93a9faa…`). The freeze-verification brief said `pythia-70m`, a wording mistake that did not affect the production
  freeze.
- **Requirements for the lock and confirm launchers** (from the freeze review; recorded, with no code change for the
  completed freeze). Beyond 024's guards, the launcher must:
  - record the full HEAD commit, the environment, package and runtime information, and its own file sha256;
  - block and log direct `.forward()` routes, and unpatched aliases or copies of the capture and intervention
    functions;
  - block repository deletes and renames;
  - keep the 4 spent patch-path runs apart from the 90,720 fresh condition-tagged runs;
  - log the condition of every patched fresh run.

  For confirm, these protections must be in place before any fresh ledger entry and before any prompt.
- **Requirements for the confirm launcher, from the lock review** (`evidence/lock-2026-09-26/README.md` has the full
  text):
  - **N1:** `chdir` to the resolved repository root, assert `cwd == repo_root`, and run every git and provenance check
    there. `collect_git_state()` uses the working directory. A mismatch stops before the patch-path check, the
    ledger or any prompt.
  - **N2:** use the lock's machine and stack: macOS arm64, Python 3.12.13, torch 2.14.0, the same packages, 4
    threads, and the same checkpoint files. No `uv sync`, upgrade or environment recreation; I7′ must reproduce the
    geometry bit for bit.
  - **N3:** allow forwards only inside sanctioned calls, and block and log everything else: instance and class
    `.forward`, `__call__`, hook, cache and generate routes, and every capture or intervention alias.
  - **N4:** an explicit allow-list of the expected output paths and state writes, even inside
    `outputs/experiment-025/`. Deletes and renames only for the atomic temporary-file → target operations.

## Report and closure (2026-09-27)

**The report run.** The production `report` was invoked exactly once, at `c8d9fad`, from the confirm-final state
`1d5d37f0…`. It ran from 11:13:34.742 to 11:13:42.044 UTC on 2026-09-27, and exited with status 0 and no incident.
- A guarded launcher refused every model load, tokenizer load, module call, and every prompt, capture, intervention,
  measurement, gate and ledger entry point. Each was attempted 0 times.
- It rendered only the completed confirmation state.
- Its only writes were `report.md` and one atomic state write, which adds the report entries. Reverting them
  reproduces the confirm-final state `1d5d37f0…` byte for byte.
- The post-report verification passed all 15 checks. They include the frozen renderer reproducing `report.md` byte
  for byte.
- Evidence: [`evidence/report-2026-09-26/`](evidence/report-2026-09-26/README.md).

**The closure data**, byte copies and an extract of the gitignored outputs; nothing was re-rendered or recomputed:
- [`evidence/final-report-2026-09-26.md`](evidence/final-report-2026-09-26.md): `outputs/experiment-025/report.md`
  byte for byte. 7,868 bytes, sha256 `ad9f85e62e74bf10b22060dc6f49ab75ddf3e79b7c81814d29ac6aa19fb75407`.
- [`evidence/confirmation-record-2026-09-26.json`](evidence/confirmation-record-2026-09-26.json): the
  confirmation-record extract of the final state, content `30da650f3d8ce15fdfef86e51a5e2adb8ebb5bb4ec45f1b5aedaa738796052ca`.
  - It follows the 022–024 convention: the prompt ledger is replaced by its counts and digests.
  - It adds:
    - the run accounting: manifest = run log = ledger = 90,720 keys;
    - the isolation against the 43,632 earlier spent keys, of which 0 were executed;
    - the exit-127 event;
    - a summary of the official result, copied from the state.
  - Its builder and the pre-closure check (24/24) are in
    [`evidence/closure-2026-09-26/`](evidence/closure-2026-09-26/README.md).
- The fresh measurements `outputs/experiment-025/stage2-measurements.pt` (898,673,151 bytes, sha256
  `87284bddfba16a1b7709d7d33f98aa1a0898b1d85767956a524d2950d80a7788`) and the final state (`534dad39…`, file
  `fef818dc…`) stay local and unchanged. The repository tracks no binary measurement artifact; their hashes are
  recorded in the evidence and the extract.
- The provenance chain: [`evidence/README.md`](evidence/README.md).

**The exit-127 shell event (provenance only).** At 19:23:09 UTC on 2026-09-26, the first confirm launch command failed
in the shell, before Python started.
- The command had been stored in a shell variable. zsh does not word-split an unquoted variable, so it looked for a
  program named after the whole string and exited with 127.
- No launcher code ran and `runner.confirm()` was not invoked. No state, output or ledger changed.
- **It was not a scientific attempt.**
- The launch of record followed, with the command written out literally; the launcher started at 19:23:57.846 UTC.
- The event is archived in `evidence/confirm-2026-09-26/` (`shell_error_production.*`). The independent confirmation
  review verified it from the file contents and timestamps.

**The result, in its scope.**
- The official result is **`CAUSAL_EFFECT_NOT_ESTABLISHED`**: A = 26/40, B = 23/40, G = 37/40, threshold = 27,
  reference tail = 0.01923865414210013.
- The preregistered reading: the preregistered Experiment 025 test did not establish that moving fresh cue embeddings
  in the +nounness direction consistently causes greater frozen block-4/5 readout-approximation error than the matched
  −nounness intervention across the frozen 40-cue population.
- The label records that this test did not establish the effect. It is not a finding of no causal effect.
- A failed, so the frozen hierarchy assigns this label whatever B and G are. G's 37/40 cannot change it.

**Descriptive records (no outcome force).** They were computed after the result was written, or in the independent
review. None is a preregistered test, no claim rests on them, and they can never rescue, alter or redefine the result:
- the adjective/noun stratum split: A 20/20 and 6/20, B 20/20 and 3/20, G 20/20 and 17/20. The stratum counts cannot
  change the label, and no stratum-level claim is made;
- G = 37/40;
- the causal fraction, 0.341;
- the block-4/5 ladder;
- the related correlations, such as Spearman(A, s₀) = −0.83 across the 40 cues, reported descriptively in the
  independent confirmation review;
- the magnitudes, the even components and baselines, the half dose, the plurality control, nMSE, G against the random
  directions and the per-template counts, in the [final report](evidence/final-report-2026-09-26.md).

**Limitations:**
- **The scale is small:** one checkpoint (Pythia-70M deduped, `e93a9faa…`), 40 fresh cues picked mechanically (not
  sampled at random), 108 exposed frames and 79 scored nouns.
- **The intervention is narrow:** one norm-preserving rotation of the cue's input embedding, along one frozen
  weight-derived direction, at two doses.
- **What was measured** is the error of a frozen approximation, and one frozen L4/L5 routing summary. Neither is
  directly observed rerouting.
- **The nounness direction is partly aligned with plurality** (cos +0.235).
- **Not claimed by any label:** semantic nounhood as the model's variable; a general routing controller; algorithm
  selection in general; generalization beyond this checkpoint and this mechanism.

**Experiment 024's result is unchanged.** Its frozen nounness score prospectively predicted the frozen readout error,
which is a predictive result, not a causal one. Experiment 025's preregistered test did not establish the causal
reading it tested.

**The endpoint.** Experiment 025 is the endpoint of this sequence, as its design fixed in advance. Confirm is not
rerun, and no Experiment 026 follows.
