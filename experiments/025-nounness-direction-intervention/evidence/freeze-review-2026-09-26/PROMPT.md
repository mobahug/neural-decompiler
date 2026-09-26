You are the independent, read-only verifier of Experiment 025's production `freeze` in the research repository
/Users/gaborhorvath-ulenius/myprojects/neural-decompiler (a mechanistic-interpretability project on Pythia-70M).

The freeze ran exactly once, under a guarded launcher, at implementation commit c765148fdb612ae249e21ef3c830e4ceb1d12dd0.
It wrote the one artifact `experiments/025-nounness-direction-intervention/confirmation-v1.json`, which is not
committed. The launcher and its run record are in
/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/freeze025/
(`launch_freeze.py`, `freeze_run.json`, `launch.log`). Before the production invocation, the same launcher ran once
in dry-run mode (`FREEZE025_DRY_RUN=1`, record `dryrun_run.json`). That run executed every guard and pre-write check
in memory, did not call `freeze` and wrote nothing. Check that this is what the code and its record show.

Specifications:
- the design `docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md` (revision 1,
  c0885e5, corrected 26c9925), especially "Population", "Conditions, manifest and budget" and "Phases and leakage
  boundaries";
- the plan `docs/superpowers/plans/2026-09-26-experiment-025-nounness-direction-intervention-plan.md` (7d90d28);
- the module `src/neural_decompiler/cue_rotation.py` and the runner `experiments/025-nounness-direction-intervention/run.py`.

## Hard rules

- **Read-only.** Do not create, edit, move or delete anything in the repository, and do not stage or commit.
  - Write your report and any scratch scripts only under
    /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/freeze-review025/
  - The report is `REVIEW.md`.
- **Tokenizer and committed files only.**
  - Never load the model. Never run a forward pass, a capture or an intervention.
  - Never compute a nounness score, a centroid, a direction, a rotation, ℓ, A, B, G, D_attn or a Level-1 value.
  - Never run the runner's `freeze`, `lock`, `confirm` or `report`.
  - The runner's `validate` would refuse the untracked file, so do not rely on it.
- **Environment.** Use `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python` with
  `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`. Never `uv`.
  - Tokenizer: `AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision=<the pinned revision in neural_decompiler.models.PYTHIA_70M>, local_files_only=True)`.
- **Never read** anything under `outputs/experiment-023/`, `outputs/experiment-024/`, or 022's `calibration-table.pt`.

## Verify independently

Re-derive everything with your own code from the design's rules, the committed files and the tokenizer. Do not call
`cr.select_cues`, `cr.freeze_payload`, `cr.blocked_sets` or `cr.confirmation_from_payload` to produce the values you
compare against. You may use them afterwards as a cross-check.

1. **The file's integrity:**
   - its sha256;
   - `content_sha256` equals `rc.content_digest` of the payload;
   - the `manifest_sha256` recomputes;
   - `experiment` is "025" and `schema_version` is 1;
   - the `configuration` is `cr.PRODUCTION.to_json()`: threshold 27, reference tail 0.01923865414210013, the 21
     conditions in frozen order, 16 of them outcome-bearing;
   - `design` and `plan` are the recorded commits.
2. **The four exclusion sets**, recomputed from their sources, equal the file's `blocked` sets:
   - the earlier cues: 024's committed freeze `exclusion.cue_token_ids` plus 024's 40 cue ids;
   - the pool target-noun forms: every sg and pl id of the 80 pool nouns;
   - the prior-noun forms: every sg and pl id of the 24 nouns in 020's committed `confirmation-v1.json`
     (`rd.CONFIRMATION_RELATIVE_PATH`), including statue and barrel;
   - the exposed-frame tokens: every token id of the 108 exposed frames.

   The pool and frames come from `ul.load_frozen_inputs(ROOT)`. Report each set's size.
3. **The mechanical selection:**
   - Walk 024's adjective reserves (23, in order), then 024's ordinary reserves (9 pairs), then the frozen 39-word new
     list (regular plurals).
   - Apply the rules: `" "+word` is a single token; the id is in no exclusion set; the id is not already picked; a noun
     needs both forms eligible and distinct.
   - Take the first 20 adjectives and the first 20 nouns.
   - They must equal the file's cues (word, token id, stratum, source, rank, plural and plural id), the approved 40,
     in order.
   - Check every rejection entry's reason against your own derivation.
   - Report the candidates, rejected, eligible, picked and unpicked-eligible counts per list.
   - statue must be rejected by the prior-noun rule.
4. **The manifest:**
   - Reconstruct the 40 × 108 × 21 = 90,720 condition-tagged keys `frame_id|word|token_id|condition` and compare
     them with the file's sorted list.
   - Every key must be unique and tagged with a frozen condition.
   - The untagged prompt keys must be the 4,320 cue × frame prompts.
   - No untagged or tagged key may be in the spent set: 020's ledger, 020's confirmation prompts, and the 022, 023 and
     024 manifests. The runner's `Runner()._base().forbidden` gives it; recompute it yourself too if you can. Report
     its size (the plan says 43,632).
   - No key may be one of the four patch-path keys `cr.PATCH_PATH_SPENT_KEYS`, and those four must be in the spent
     set.
5. **The launcher's evidence (`freeze_run.json`, `launch.log`):**
   - one freeze invocation and exit status 0;
   - 0 model loads, 0 module calls, 0 capture calls;
   - no score or geometry call;
   - no forbidden read or write;
   - exactly one confirmation write;
   - the pre-write checks all true;
   - the written bytes equal to the pre-verified payload.

   Check that the launcher's guards were actually installed before the runner was imported, and that they would have
   caught a forward, a model load or a score.
6. **The repository state:**
   - HEAD, `origin/main` and the remote are equal (c765148…);
   - `git status` shows only the untracked confirmation file;
   - `outputs/experiment-025/` does not exist, so no results state and no ledger exist;
   - no other file under `experiments/025-nounness-direction-intervention/` is new;
   - no lock or preregistration exists.
7. **Leakage:** confirm, as far as the evidence allows, that nothing in the freeze path can compute or read a score,
   a geometry quantity, a model output or a measurement. Trace the code path of `Runner.freeze` →
   `cr.freeze_payload`.

## Report (REVIEW.md) and final message

- Verdict: PASS, PASS WITH NOTES, or FAIL.
- Every check with its result and the evidence: numbers, digests, and short command outputs.
- Findings, most severe first (blocker / should-fix / note), with file:line and a concrete scenario.
- Return the verdict, the key numbers (the 40 cues with token ids, the per-list counts, the exclusion sizes, the
  manifest count and digest, the file and content digests, the spent-set size and the collision result) and the
  findings as your final message.
