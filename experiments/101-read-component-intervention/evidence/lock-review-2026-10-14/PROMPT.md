You are the independent, adversarial, read-only reviewer of Experiment 101's production **lock** in the research
repository /Users/gaborhorvath-ulenius/myprojects/neural-decompiler (a mechanistic-interpretability project on
EleutherAI/pythia-70m-deduped @ e93a9faa9c77e5d09219f6c868bfc7a1bd65593c). You are an AI reviewer; your report is an
AI review, not a human validation — say so in it.

The lock ran exactly once, at commit `3621eec4b6a901119a4b96d4fdc07fa3b19652ab` on the branch `exp101-phases`, under a guarded launcher (weights only;
every forward refused). It wrote candidate files that are NOT installed or committed; per design §9 "The user commits
the lock", so the human decides whether to install and commit them after this review:

| file | expected |
|---|---|
| `outputs/experiment-101/candidate-lock.json` | file sha256 `8dc6e6d38c876a2dd9bf905fbeaf36b774b487967597795beb032bb4a0751087`; content sha256 `8da97e35556d0baa4739899cb841e86cbd59cced21611c732fdd9d8c9edd9061` |
| `outputs/experiment-101/candidate-preregistration.md` | sha256 `3761b8130203d5d15ac0e30055aa135626a6d97f0ab807df6cf4d31b6af8e980` |
| `outputs/experiment-101/results.json` | `state_sha256` `a4657f80eed3e23587ea44597b77ce064c432eea5623799c838acdb668f40e12` |

The lock's evidence is in
/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/lock/
(`launch_lock.py`, sha256 `90e8159f…` at production; `dryrun_run.json`, `dryrun.log` (one dry run at 09:30 UTC, which did not call `lock` and left the state unchanged, `state_before_dryrun.txt`); `lock_run.json`, `lock.log`, `lock.exit`). Treat the operator's records as
claims to test, not as evidence.

The specifications:
- the design `docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md` (revision 3,
  decided `516df20`), especially §5–§9 and §14;
- the plan `docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md` (revision 3, G4
  decided `1e48c1c`), especially § Constants, § The canonical computations, § Phases (`lock`, `stage1`), § The
  prediction freeze, § Gates, as implemented, and § G4's form and the reviewer's decision;
- the module `src/neural_decompiler/read_intervention.py` (`ri`) and the runner
  `experiments/101-read-component-intervention/run.py`, and the pinned modules they call;
- the committed `confirmation-v1.json` and `calibration-v1.json` of Experiment 101 and their evidence and reviews under
  `experiments/101-read-component-intervention/evidence/`.

## Hard rules (a violation invalidates the review)

- **Read-only.** Never create, edit, move, delete, stage or commit anything in the repository, including the candidates
  in `outputs/experiment-101/`. Never install the candidate lock or preregistration.
- Write only under
  /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/lock-review/
  (scripts, logs and `REVIEW.md`).
- **Never run** the runner's `freeze`, `calibrate`, `lock`, `stage1`, `confirm` or `report`. `validate` is allowed.
- **No 101 frame prompt, ever:** never run a forward on any of the 60 candidate texts with any cue. Install a guard that
  refuses those token sequences in every script that loads the model, and record that it never fired.
- **Weights-only work is allowed:** loading the model to read weights and digests, recomputing `w_c`, the `v_j`, the
  geometry block and digests, and the exposed prediction table on 020's locked states (spent data). Forward passes:
  none needed; if you run any, only on spent exposed prompts of 020's pool, at most 3.
- **Environment:** `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python -B` with
  `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`. Never `uv`. Use `git --no-optional-locks`.
- Never read `outputs/` of other experiments except `outputs/experiment-020/results.json`.

## Independence

Write your own code first for every reconstruction (the read covector from 011's lock and from the weights, the SHA-256
counter-mode isotropic directions with the tag `101|iso|{frame_id}|{j}`, their orthonormality, the stage-1 layout, the
digests), then cross-check with `ri.read_geometry`, `ri.iso_direction`, `ri.geometry_block`, `ri.stage1_binding`,
`ri.build_lock` and `ri.render_preregistration`. Report the maximum numerical disagreement for each quantity.

## Review items

1. **Repository and phase integrity:** HEAD `3621eec4b6a901119a4b96d4fdc07fa3b19652ab`, clean tree; the installed freeze and calibration unchanged and
   committed; the lock ran exactly once (the state's `phases.lock`), stage 1 and confirm not started, the confirmation
   ledger (`executed_prompt_keys`) empty; no incident; a second lock refused (`ri.assert_phase_allowed("lock", state)`
   raises; do not run lock). Only the expected files exist in `outputs/experiment-101/`.
2. **The launcher's evidence:** one lock invocation, exit 0; zero module calls, zero forwards, zero captures, zero
   measurement-function calls; the model load counted and sealed; the audit hook's allowed writes only under
   `outputs/experiment-101/`; the dry run's exposed-table digest equals the calibration's `table_content_sha256`. Check
   that the guards were installed before the runner was imported and would have caught a forward or a capture.
3. **The lock's content, field by field:** `content_sha256` recomputes; `run_id`, `protocol_code_commit`; design and
   plan records; the configuration (`ri.PRODUCTION`); the 20 module blobs and the module's own blob (equal to the git
   blob at HEAD); the inputs' digests (equal to the runner's `base.digests`); the 025 inheritance; the confirmation
   binding (file, content, manifest digests and counts); the calibration binding (file and content digests, τ_rem,
   τ_ins, evaluable); the dependencies (the 020 readout, the exposed states digest, the model's parameter and embedding
   digests, recomputed by you); the noun keys (79, in `NounSet` order); the run order and cue order; the statistics
   text, constants (Holm 1/40, thresholds, WORST, bootstrap seed and ranks), tolerances (equal to the frozen
   `ri.TOLERANCES` — rule 1: no tolerance widened), the outcome table and the semantics (scope, non-claims, hierarchy)
   against the design's §7 and §14 wording.
4. **The geometry, recomputed by you:** `w_c` (from 011's `read_weight`, centred; and from the weights), its norm
   (about 0.766) and digest; G1's values; every `v_j` (36 frames × 8) unit, mean-zero and orthogonal to `w_c` to 1e-12,
   and their digests; the û_off rule text.
5. **The stage-1 binding:** the paths, the states layout and metadata (recompute from the 36 frozen frames, `d_model`
   512, 8 heads, the noun count), the table's metadata (`fractions_for_weak` false, the mask, the arms, the cue order,
   the noun keys), and the freeze text. Check that the barrier in `run.py` (`_barrier`, `_committed_stage1`) would
   refuse or record an incident on each tamper scenario of plan § The prediction freeze, by reading the code (no run).
6. **The preregistration:** `candidate-preregistration.md` equals `ri.render_preregistration` of the candidate lock
   byte for byte, and its text says what the design says (labels, thresholds, order, scope, non-claims). Flag any
   wording that over- or under-states the design.
7. **Determinism:** recompute the exposed prediction table (`fractions_for_weak=True`) in your own process and compare
   its digest with the calibration's `table_content_sha256`; recompute the geometry block twice and compare bit for
   bit.
8. **The calibration's numbers as bound:** τ_rem and τ_ins in the lock equal the installed calibration file's, and P is
   evaluable; nothing in the lock depends on the size of an exposed N, S, D or P result.
9. **Leakage and the prediction freeze:** nothing in the lock path ran or could run a 101 frame prompt; the stage-1
   launcher requirements (plan § One-shot protections: only the 36 S1 keys through `ri.capture_reference`, the sealed
   model, the write allow-list naming the three stage-1 files and the `stage1/` mkdir) are stated in the evidence for
   the next phase.
10. **Anything that should stop the human from committing this lock** (a blocker), or that they should know before
    committing it (should-fix / note), including residual risks for confirm recorded by the calibration review.

11. **The prepared stage-1 launcher (read the code; do not run it).** A draft guarded launcher for `stage1` is at
    `.../scratchpad/exp101/stage1/launch_stage1.py`. It has never run and cannot run before the lock is committed. Check
    it against plan § One-shot protections and § The prediction freeze: the allow-list (exactly the three stage-1 files
    and the `stage1/` mkdir, the results state through its temporary files), the runner's `write_check` predicate, the
    sanctioned `ri.capture_reference` for exactly the 36 S1 keys after the S1 ledger is on disk, `ri.run_arm` refused,
    the model sealing, the audit hook's argument mapping (the freeze review's N1), the preflight (including
    `_validated_lock`), and the post-run checks. Report anything that would let a target prompt run, a file other than
    the three be written, or the phase run twice; and anything that would make a correct run fail.

## Report (REVIEW.md) and final message

- Header: "AI review (independent, read-only), not human validation".
- Verdict: PASS, PASS WITH NOTES, or FAIL.
- Every item with evidence; findings most severe first (blocker / should-fix / note) with file:line and a concrete
  scenario; the maximum numerical disagreement per recomputed quantity.
- Final message: the verdict, the key digests, and the findings. Do not hand back until REVIEW.md is final.
