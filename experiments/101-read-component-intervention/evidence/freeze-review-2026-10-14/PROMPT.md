You are the independent, read-only verifier of Experiment 101's production `freeze` in the research repository
/Users/gaborhorvath-ulenius/myprojects/neural-decompiler (a mechanistic-interpretability project on Pythia-70M-deduped).
You are an AI reviewer; say so in your report. Your report is an AI review, not a human validation.

The freeze ran exactly once, under a guarded launcher, at commit `40829d21a795a501d9a61c953587b8afcf5f0b86` on branch `exp101-phases`. It wrote the one
artifact `experiments/101-read-component-intervention/confirmation-v1.json`, which is not committed yet. The launcher
and its records are in
/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/freeze/
(`launch_freeze.py`, `freeze_run.json`, `launch.log`). Before the production invocation, the same launcher ran in
dry-run mode (`FREEZE101_DRY_RUN=1`, record `dryrun_run.json`, log `dryrun.log`), which executed every guard and
pre-write check in memory, did not call `freeze` and wrote nothing. (An earlier dry run of an earlier version of the
launcher recorded two false-positive refusals of `os.remove` on relative names given with a `dir_fd` — filelock's
import-time probe inside a temporary directory; the launcher was fixed to resolve `dir_fd` before the second dry run.
Check that the final launcher and records are consistent with this.)

Specifications:
- the design `docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md` (revision 3,
  `a87e284`, decided `516df20`), especially §3, §11 and §14 (the frame-freeze guard);
- the plan `docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md` (revision 3,
  `96f0cc8`; G4 decision `1e48c1c`), especially § Constants, § The frame freeze and § Ledger, isolation and accounting;
- the module `src/neural_decompiler/read_intervention.py` (`ri`) and the runner
  `experiments/101-read-component-intervention/run.py`.

## Hard rules

- **Read-only.** Do not create, edit, move or delete anything in the repository, and do not stage or commit.
  - Write your report and any scratch scripts only under
    /private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/freeze-review/
  - The report is `REVIEW.md` in that directory.
- **Tokenizer and committed files only.**
  - Never load the model or open a weight file (`*.safetensors`, `pytorch_model.bin`, `*.ckpt`). Never run a forward
    pass, a capture or an intervention.
  - Never compute the read covector, any geometry, any prediction, any measurement or any statistic.
  - Never run the runner's `freeze`, `calibrate`, `lock`, `stage1`, `confirm` or `report`. (`validate` refuses an
    untracked confirmation file; do not rely on it.)
- **Environment.** Use `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python` with
  `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`. Never `uv`.
  - Tokenizer: `AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision="e93a9faa9c77e5d09219f6c868bfc7a1bd65593c", local_files_only=True)`.
    Cross-check it with a second implementation (the raw `tokenizers` library on the snapshot's `tokenizer.json`).
- **Never read** anything under `outputs/` except `outputs/experiment-020/results.json` if you need 020's ledger (you
  may instead use the runner's `Runner()._base().forbidden`, which reads it).

## Verify independently

Re-derive everything with your own code from the design's rules, the committed files and the tokenizer, before you
use `ri.freeze_payload`, `ri.candidate_status` or `ri.exclusions_from_committed`; use them only afterwards as a
cross-check, and record the order.

1. **The file's integrity:** its sha256 and byte count; `content_sha256` equals `rc.content_digest` of the payload;
   `manifest_sha256` recomputes; `experiment` "101", `schema_version` 1; the `configuration` equals
   `ri.PRODUCTION.to_json()` (Holm 26/25, D 25, P 27, precondition 30, τ rank 33, 55 runs per frame, the cues one 581 /
   two 767 / three 1264 / many 1142, k_iso 8); `design` and `plan` are the recorded commits.
2. **The exclusion sources** (design §11; plan § The frame freeze), recomputed from the committed files and compared
   with the file's `exclusions` record (counts and digests): the frame texts (b0c's exclusion as 023's committed
   `exclusion.frame_texts`, plus 023's own frame templates); the earlier cue ids (024's committed exclusion
   `cue_token_ids` plus 024's cues plus 025's cues: 431); 025's candidate-list words (the adjective reserve, the new
   list and both forms of the ordinary reserve) and their single-token ids; the target-noun forms (161) and 020's
   confirmation-noun forms (48); `NUMBER_WORDS`. Report each set's size.
3. **The mechanical selection:** walk the 60 candidates of design §11 in order. For each, with the raw tokenizer: put
   each cue word (` one`, ` two`, ` three`, ` many`) in the slot; the cue is a single token occurring once, at `p_c`
   from 4 to 7; all four prompts share the prefix and the suffix; the adjective is one final token at `p_t = p_c + 1`;
   the text is not an earlier frame text; the adjective is no earlier cue, no 025 candidate word (by word or
   single-token id), no target-noun form, no number word, and not already taken. The first 36 eligible must be
   candidates 1–36 (`EXPECTED_PICKS`), with no replacement. Compare every candidate's eligibility and reasons with the
   file's `candidates` entries, and every frozen frame's `prefix_ids`, `suffix_ids`, `cue_ids`, `p_c`, `p_t`, ids and
   texts with your derivation. Check that `frame_tokens` (025's exposed-frame-token set) is deliberately NOT applied
   (plan SA3) and list the picks it would have blocked.
4. **The manifests:** reconstruct `S1-REF` (36 keys `frame_id|ref|581|S1`) and `S2-TARGET` (per frame: `REF` on the
   reference prompt; 32 `ARM@two` runs; 11 `ARM@three` and 11 `ARM@many`, in `CUE_ORDER` and the plan's arm order,
   with the reference-prompt arms carrying the `ref|581` prompt key), 1,980 distinct keys. Compare with the file's
   sorted lists, the counts and the digests. The untagged prompt keys must be 144 (36 × ref/two/three/many). No tagged
   or untagged key may be in the spent set (025's forbidden set: 020's ledger, 020's confirmation prompts, the 022, 023
   and 024 manifests, and the untagged forms of 025's manifest); report its size and the collision count.
5. **The launcher's evidence:** one freeze invocation and exit 0; 0 model loads, 0 module calls, 0 capture calls, no
   refused-function call, no forbidden write or filesystem operation; exactly one confirmation write; every pre-write
   check true; the written bytes equal to the pre-verified payload; the only `outputs/` read is 020's results state.
   Check that the guards were installed before the runner was imported and that they would have caught a forward, a
   model load or a geometry/prediction/outcome call.
6. **The repository state:** HEAD is `40829d21a795a501d9a61c953587b8afcf5f0b86` on `exp101-phases`; `git status --porcelain --untracked-files=all` shows
   only the untracked confirmation file; `outputs/experiment-101/` does not exist (no results state, no ledger); no
   lock, calibration or stage-1 file exists. Record how far HEAD is ahead of `origin/main` (the local commits are not
   pushed; this is a provenance note, not a blocker by itself — say whether you agree).
7. **Leakage:** trace `Runner.freeze` → `ri.freeze_payload` → `ri.candidate_status` and confirm, as far as the evidence
   allows, that nothing in the freeze path can compute or read a model output, a geometry quantity or a measurement.

## Report (REVIEW.md) and final message

- Header: "AI review (independent, read-only), not human validation".
- Verdict: PASS, PASS WITH NOTES, or FAIL.
- Every check with its result and evidence: numbers, digests, short command outputs.
- Findings, most severe first (blocker / should-fix / note), with file:line and a concrete scenario.
- Your final message: the verdict, the key numbers (the 36 frames with adjectives and `p_c`, exclusion sizes, manifest
  counts and digest, the file and content digests, the spent-set size and collision result), and the findings.
