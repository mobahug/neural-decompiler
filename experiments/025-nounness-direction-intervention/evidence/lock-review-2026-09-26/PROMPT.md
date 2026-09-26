You are the independent, read-only reviewer of Experiment 025's production **lock** in the research repository
/Users/gaborhorvath-ulenius/myprojects/neural-decompiler (a mechanistic-interpretability project on
EleutherAI/pythia-70m-deduped @ e93a9faa9c77e5d09219f6c868bfc7a1bd65593c).

The lock ran exactly once, at commit `d3ecbd6feb1bd2aefe08dc0f8d799815be47c519`, under a guarded launcher. It wrote
three candidate files, none of them installed or committed:

| file | expected |
|---|---|
| `outputs/experiment-025/candidate-lock.json` | file sha256 `1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be`; content sha256 `8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb` |
| `outputs/experiment-025/candidate-preregistration.md` | sha256 `e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919` |
| `outputs/experiment-025/results.json` | `state_sha256` `8eeb8184faf556211e68579d4682c85f545c877051641afe07dd5eac2b701dad` |

The lock's evidence is in
/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/lock025/
- `launch_lock.py`, sha256 `f66c12f2…`, the version that ran in production;
- `lock_run.json` and `lock.log`, the production run;
- `dryrun1_run.json` and `dryrun1.log`, the first dry run, with a guard false positive, run by an earlier launcher
  version `cdc466f3…`;
- `dryrun_run.json` and `dryrun.log`, the second dry run, clean;
- `prelock.py`/`.out` and `verify_lock.py`/`.out`, the operator's own checks.

Treat the operator's checks as claims to test, not as evidence.

The specifications:
- the design `docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md` (revision
  1, c0885e5, corrected 26c9925);
- the plan `docs/superpowers/plans/2026-09-26-experiment-025-nounness-direction-intervention-plan.md` (7d90d28);
- the module `src/neural_decompiler/cue_rotation.py` (`cr`) and the runner
  `experiments/025-nounness-direction-intervention/run.py`;
- the pinned 020–024 modules they call: `readout_routing.py` (rr), `upstream_localization.py` (ul),
  `block0_completion.py` (b0c), `readout_decompilation.py` (rd), `plural_mechanism.py` (pm), and the others;
- the installed freeze `experiments/025-nounness-direction-intervention/confirmation-v1.json` and its evidence under
  `experiments/025-nounness-direction-intervention/evidence/`.

## Hard rules (a violation invalidates the review)

- **Read-only.** Never create, edit, move, delete, stage or commit anything in the repository. That includes the
  candidates in `outputs/experiment-025/`. Never install the candidate lock or the preregistration.
- Write only under
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/lock-review025/
  (scripts, logs and `REVIEW.md`).
- **Never run** the runner's `freeze`, `lock`, `confirm` or `report`. `validate` is allowed.
- **No fresh or rotated prompt, ever:**
  - never run a forward on any Experiment 025 manifest key or any 025 cue;
  - never apply a rotated vector to the model;
  - never compute ℓ, an MSE, A, B, G, `D_attn` or a Level-1 value for any run.
- **Weights-only work is allowed:** loading the model to read its weights, embedding rows, parameter digests and the
  geometry.
- **Spent-prompt captures are optional, and allowed only for item 10's alignment check:**
  - plain captures (`pm.capture_prompt`) of at most 6 reference prompts, two per template family;
  - the key must be `frame_id|ref|<reference id>` and must be in 020's ledger (`ul.load_frozen_inputs(ROOT).closure["ledger"]`);
  - record equality or closeness of the captured `ATTN_PATTERN.L4/L5@p_t` rows to the locked 020 rows, never a
    distance statistic;
  - guard the entry points so that any other key is refused before it runs.
- **Environment:** `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python` with
  `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`. Never `uv`. Use `git --no-optional-locks` for status.
- **Never read** anything under `outputs/experiment-023/`, `outputs/experiment-024/`, or 022's `calibration-table.pt`.

## Independence

- Write your own code first for every reconstruction: `d`, the cue geometry, the rotations, the SHA-256
  counter-mode controls, the plurality control, the manifest, and the outcome logic.
- Do not call `cr.geometry_block`, `cr.cue_vectors`, `cr.cue_geometry`, `cr.direction`, `cr.control_directions`,
  `cr.sha_uniforms`, `cr.sha_gaussians`, `cr.plurality_direction` or `cr.theta_for` until your own values exist. Then
  cross-check against them and against the lock.
- You may use the model loader, `pm.Weights.from_model`, `rc.tensor_digest` and `pm.canonical_json`, the latter two
  to compare digests, and the committed JSON records.
- Report the maximum numerical disagreement for each quantity.

## Review items (the reviewer's 20; answer each with evidence)

1. **Repository and phase integrity.**
   - HEAD, `origin/main` and the remote are `d3ecbd6…`; the tree is clean; the installed freeze is unchanged.
   - The lock completed exactly once; confirm and report have not started; the prompt ledger is empty.
   - No fresh 025 measurement artifact exists (only the three candidate files are in `outputs/experiment-025/`), and
     no incident exists.
   - A second lock is refused: check `cr.assert_phase_allowed("lock", state)` without running lock.
   - If the format permits it, check that removing the lock-phase fields reconstructs a fresh
     `cr.new_results_state`, apart from `run_id` and `created_at`.
2. **Candidate integrity.**
   - The file hashes and the content digest (`rc.content_digest`); canonical JSON (re-serializing equals the bytes).
   - Every value is finite.
   - The preregistration equals `cr.render_preregistration(json.loads(lock))` byte for byte.
   - Check every number rendered in the preregistration against the lock.
3. **Lock-run isolation.** From the launcher code and its records, establish that the production lock:
   - loaded the weights once;
   - ran zero prompts, module calls, `.forward()` routes, captures, interventions and confirm-only computations;
   - wrote only the expected three output files and the state's temporary-file-then-rename.

   Check that the guards were in place before the runner import and the model load, and that they would have caught
   a violation. Explain why the first dry run's false positive cannot have affected the production lock (a separate
   process and launcher version; nothing written).
4. **Dependency bindings.**
   - Independently hash every scientific dependency the lock binds:
     - the freeze (file and content digests) and the 90,720-key manifest digest;
     - the pinned module blobs (compute git blob ids yourself);
     - 024's calibration record, lock and freeze;
     - 020's confirmation file (the prior-noun source);
     - the model parameter digest and the embedding digest;
     - the model and tokenizer revision;
     - 020's exposed-states digest (`dependencies.readout_020.exposed_states_sha256`).
   - Verify that the lock binds `cr`'s own module blob, and compare it directly with the current file and the
     imported module.
   - Verify, by reading `cr.validate_lock` and the runner's confirm, that scientific-path drift makes a future
     confirm refuse before any fresh execution: a changed module, a changed input, a dirty tree, a lock commit that is
     not an ancestor, or an untracked lock. You may call `cr.validate_lock` or `cr.scientific_changes` with synthetic
     inputs.
5. **Independently reconstruct `d`.**
   - Build `μ̂_noun` and `μ̂_cue` from 024's calibration record id lists (`score.noun_row_ids`,
     `score.calibration_cue_ids`), as float64 means of the model's float32 embedding rows, then normalize.
   - Verify the centroid digests and `d`'s digest `5f914283c39d2f37911ba3bb1391a98bbe3625737f8df80c42eb336868a1bf2a`.
     Expect |d| ≈ 1.35766.
   - Report the maximum disagreement with `cr.direction`.
6. **Independently reconstruct all 40 cue geometries.**
   - For each cue: E, |E|, Ê, s₀ = d·Ê, t = d − s₀Ê, τ = |t|, t̂, θ = asin(0.32/τ) and θ_half = asin(0.16/τ).
   - Expect τ ≈ 1.3227–1.3560, s₀ ≈ −0.306…+0.285, θ ≈ 13.65–14.00°, θ_half ≈ 6.78–6.95°.
   - Check finiteness, valid arcsine arguments and the exact frozen cue order.
   - Compare with every stored lock value (s0, tau, theta_primary, theta_half, even_primary, even_half, norm, and the
     t̂ digest).
7. **Independently reconstruct the vectors.**
   - Build the nounness ±primary and ±half, the base, and the plurality ± vectors as float64 rotations
     `|E|(cos θ Ê ± sin θ û)`, and their float32 casts.
   - Verify norm preservation, the odd component ±0.32 and ±0.16, the complete change `s₀(cos θ − 1) ± odd`, and the
     angle to E, in float64 and on the float32 casts.
   - Rebuild the per-cue `vectors64_sha256` and `vectors32_sha256` (the 21 conditions stacked in the frozen order),
     and the combined 840-vector float32 digest `30a35c0050421b0b07579ee5827e23386dde95ec777aba8ac36a940551df7ca9`
     (all 40 cues' 21 float32 vectors stacked in frozen cue order, via `rc.tensor_digest`). Do this only after your
     own reconstruction.
8. **Independently reconstruct the 280 random controls. This is critical.**
   - Reimplement the frozen stream:
     - tag `025|control|{token_id}|{j}`, j = 1..7;
     - SHA-256 of `f"{tag}|{counter}"`, each digest giving four 64-bit big-endian words, each shifted right by 11,
       then `(v + 1)/2**53`;
     - Box–Muller pairs in float64 over 512 dimensions;
     - project off Ê and t̂, repeated twice, then normalize.
   - Rebuild each control's ±θ float32 patch.
   - Verify the per-cue `controls_sha256` and the combined control digest
     `e12581d4be183798b42e2f673670eda711ddaf87359c28df5350c1c806a37315` (stacked [40, 7, 512] float64 in frozen cue
     order).
   - On the actual float32 patched vectors, independently establish:
     - the same angle as the nounness rotation;
     - norm preservation;
     - |s(+θ) − s(−θ)| ≤ 1e-6, with s(v) = d·v̂ (expected maximum about 6.1e-9);
     - |d·u|, |Ê·u| and |t̂·u| near 0.
   - A systematic failure here is a blocker: B's interpretation rests on it.
9. **Control determinism.**
   - No Python, NumPy or torch RNG, and no library RNG, may influence generation. Read the code, and prove it with a
     second process that seeds every RNG differently and still gets identical controls.
   - The SHA stream and its counter convention are frozen.
   - Cue order and control index determine every vector.
10. **`D_attn` binding.**
    - Inspect how 020's locked states hold `rows4`/`rows5` (rd.FrameState020, `rd.state_from_locked`, and 020's
      capture of `ATTN_PATTERN.L4/L5@p_t`): the frame and reference association, the head order, and the key-axis
      alignment `[:, : p_t + 1]`, with causal zeros beyond `p_t` and rows summing to 1.
    - Verify that the exposed-states digest `26c21d63…` covers these rows (`rd.state_digest` of each entry).
    - Check that confirm's capture (`cr.measure_rotated` → `run.vector(("ATTN_PATTERN.L4", p_t))`) has the same head
      and key axes as the locked rows (`cr.routing_distance`).
    - Check that the definition is: TV per head, the 8 L4 heads plus 8 L5 heads averaged equally, then the mean over
      108 frames, with no head selection possible.
    - Optionally, the spent reference-prompt capture described in the hard rules.
11. **A/B/G semantics from the lock.**
    - `A_i = ½[ℓ(+) − ℓ(−)]`, `A_ij` likewise for control j, `B_i = A_i − mean_j |A_ij|`, and
      `G_i = ½[D_attn(+) − D_attn(−)]`.
    - PASS iff at least 27 of 40 are strictly positive; zeros, ties and NaN count against.
    - P(Bin(40, ½) ≥ 27) = 0.01923865414210013, computed exactly yourself.
    - Read `cr.statistics`, `cr.count_positive` and `cr.per_cue` to confirm the code computes exactly that.
    - Confirm that no secondary analysis can alter A, B, G or the label.
12. **The outcome hierarchy, exhaustively.** Enumerate every combination of gate validity (incident or not) × A × B
    × G. Confirm the only labels are the five, that an earlier failure is never overridden, and that
    `NOT_INTERPRETABLE` arises from any incident or gate failure (the runner's control flow).
13. **The interpretation of A.** Verify that the lock, the preregistration and the report rendering all keep this:
    A > 0 means only that +nounness has greater readout-approximation error than −nounness; the baseline and even
    effects are descriptive.
14. **Level-1 coverage.**
    - Verify that `cr.level1_gates` and `cr.enforce_gates` gate `level1_outcome_bearing` ≤ 2e-2 over exactly the 16
      conditions (`noun±0.32` and `rand1..7±0.32`), and record the half-dose and plurality conditions without
      enforcing them.
    - Check the indexing (conditions, groups, rows).
    - A single violation → an incident → `NOT_INTERPRETABLE`.
15. **The remaining identities.**
    - Vector I1 (`cr.vector_factors`), I3 and I4, their tolerances (1e-4, 1e-4 relative, 1e-3), and their
      precedence.
    - `C` is recomputed from the durably saved and re-read measurements (`torch.load` of the stage-2 file, digests
      verified), not from an in-memory copy.
    - Patch fidelity (`pm.run_patched`'s integrity check) and the θ = 0 identity.
16. **Patch-path isolation.** The four keys `cr.PATCH_PATH_SPENT_KEYS`:
    - are all in 020's ledger and outside the manifest;
    - cannot enter the fresh ledger or the accounting;
    - run only after I7′ and before the fresh ledger;
    - record only equality flags and digests (`cr.patch_path_check`), with no ℓ, A, B, G, D_attn or Level-1;
    - on failure, raise an incident before any fresh key.
17. **The manifest.** Rebuild the 40 × 108 × 21 = 90,720 tagged keys yourself from the freeze.
    - They are unique, in deterministic sorted order, with digest `0e1f068b…`.
    - There are exactly 16 outcome-bearing conditions.
    - There are 0 collisions with the 43,632 spent keys (recompute the spent set from 020's ledger, 020's
      confirmation prompts, and the 022, 023 and 024 manifests) and 0 with the patch-path keys.
    - The ledger is empty.
18. **Pre-prompt ordering in confirm.** From the actual code, before any fresh ledger entry or prompt:
    1. lock and preregistration validation;
    2. scientific-path and hash validation;
    3. model, environment and checkpoint validation;
    4. I7′ reconstruction;
    5. the geometry gates;
    6. the patch-path check;
    7. only then the ledger write and the 90,720 runs.

    Any failure in steps 1–6 prevents all fresh execution.
19. **Measurement and write ordering in confirm.** From the actual code:
    1. every tagged run;
    2. the durable save;
    3. the accounting;
    4. the re-read and digests;
    5. `C` recomputed;
    6. I1, I3 and I4;
    7. Level-1 over the 16 conditions;
    8. `D_attn`;
    9. per-cue ℓ, MSE and nMSE;
    10. A, B and G;
    11. the outcome;
    12. the binding re-check;
    13. one atomic result write;
    14. only then the descriptives.

    Also verify:
    - an incident never coexists with a result;
    - a partial-measurement failure writes the incident before the partial save;
    - a descriptive failure cannot change a valid result;
    - a second confirm is refused.

    Report any deviation from this order, with its consequence.
20. **No outcome leakage.** Establish that no fresh 025 prompt ran and no rotated fresh embedding was measured. No
    fresh Δx3, Δc, `D_attn`, ℓ, MSE, A, B or G exists anywhere: search `outputs/`, the scratchpad records and the
    state.

## The report (`REVIEW.md`) and your final message

- **Verdict:** PASS, PASS WITH NOTES, or NOT READY.
- **Blockers, separately from notes,** each with file:line, a concrete scenario and a suggested fix.
- **For every item:** the result, the evidence (numbers, digests, short outputs), and the maximum disagreements.
- **If PASS or PASS WITH NOTES:** state explicitly whether the exact existing `candidate-lock.json` and
  `candidate-preregistration.md` can be installed byte for byte, with no regeneration.
- **The final message:** return the verdict, the blockers and notes, and the key numbers.
