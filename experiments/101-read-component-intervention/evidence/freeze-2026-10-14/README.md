# Experiment 101 — the production freeze (2026-10-14)

The production `freeze` ran **exactly once**, at commit `40829d21a795a501d9a61c953587b8afcf5f0b86` on the branch
`exp101-phases`, under a guarded launcher. It wrote one file, `confirmation-v1.json`.
- The file was independently verified before installation (`../freeze-review-2026-10-14/`, an AI review, not a human
  validation): **PASS WITH NOTES, no blocker.** Its one should-fix, SF1 (preserve this evidence in the repository), is
  answered by the evidence commit that adds this directory.
- It was installed byte-identically in commit `a857f9a2326ff61cae8485deef2ab9337f5d6e02`, which touches that one path.
  It was never regenerated, parsed and rewritten, or re-serialized.
- No model was loaded and no prompt ran. No calibration, lock, stage 1, confirm or report has run.

**Provenance note.** HEAD was 6 commits ahead of `origin/main` (`96f0cc8`): the plan's G4 decision (`1e48c1c`, docs)
and the five reviewed implementation commits (`af06afb` … `40829d2`) are local and not pushed, and the freeze ran on the
branch `exp101-phases` cut from them. 025's
launchers asserted `HEAD == origin/main`; this launcher records the relation instead. The user pushes.

## 1. The test gate (before the freeze)

At `40829d2`, with a clean tree. The logs (tails) are in `test-gate/`.

| tier | result | log |
|---|---|---|
| A (`pytest`) | **578 passed**, 306 deselected, 0 failed (67 s) | `test-gate/tierA.log` |
| C (`HF_HUB_OFFLINE=1 pytest --tier C`: tier B plus the pinned-model contract) | **645 passed**, 239 deselected, 0 failed (12 min 2 s) | `test-gate/tierC.log` |

The tier-C log's tail records the 101 smoke's G-maxima on two spent exposed coordinated frames (engineering only),
for example G4 Level-1 4.74e-3 against 2e-2 and G4 I4 2.56e-5 against 1e-3; every bitwise gate held.

## 2. The freeze run

- **The launcher (`launch_freeze.py`, sha256 `ea4f55b9139842d0ad280a68b15fecaa8442b2bf58d92234e5b43dbdd1ed4203`)**
  follows 025's freeze-launcher pattern. It ran from a scratchpad directory outside the repository and wrote its
  records there; they were copied here byte-identically afterwards. Before the runner is imported, it refuses:
  - the model loader and the transformer-lens and transformers model constructors, and every
    `torch.nn.Module.__call__`;
  - the capture and intervention entry points (`plural_mechanism`'s four, `capture.run_capture`,
    `interventions.run_interventions`, `readout_decompilation.capture_frame_020`);
  - every geometry, prediction, measurement and outcome function of `read_intervention` (23 names) and the frozen
    program's `compose_dx3`, `contrast_of`, `reference_rows_017` and `pair_context`;
  - through an audit hook, every repository write, delete, rename and `mkdir` except the confirmation file (and the
    `mkdir(exist_ok=True)` of its existing directory). Every open under `outputs/` is recorded.

  It re-derives the 60 candidates from the raw tokenizer, independently of `pm._build_new_frame`, checks every rule
  against the committed exclusion sources, verifies the manifests and their isolation in memory, calls
  `runner.freeze()` once, and checks that the written bytes are the pre-verified payload.
- **Dry runs (`FREEZE101_DRY_RUN=1`).** The first dry run (08:05:03–08:05:13 UTC) passed every check but recorded two
  refused `os.remove` events on the relative names `probe-source` and `probe-link`: filelock's import-time probe deletes
  them inside a temporary directory through a `dir_fd`, and the first version of the hook resolved relative names
  against the working directory. They were false positives (nothing in the repository was touched). The hook was fixed
  to resolve a `dir_fd` through `F_GETPATH`; the record of that first dry run was overwritten by the second. The second
  dry run (`dryrun_run.json`, `dryrun.log`; 08:06:08–08:06:16 UTC) passed every check with no refusal, did not call
  `freeze` and wrote nothing.
  - **The first version is not preserved** (the freeze review's SF1 found the gap). `launch_freeze_v1_reconstructed.py`
    is a reconstruction: the final launcher with the one edit reversed (the `dir_fd` block of the hook), sha256
    `41806514b13360aaa463a6ed3e5abf7d40afa137bc0ab365f33092920884cf8a`. No hash of the original was recorded, so the
    reconstruction is not attested; it differs from the final launcher only in that block.
- **The production run (`freeze_run.json`, `launch.log` = stdout and stderr),** 08:14:27–08:14:42 UTC:
  - **exit status 0**; **freeze invocations: 1**;
  - **model loads 0, model constructors 0, module calls 0, capture calls 0, refused-function calls 0**;
  - forbidden writes 0, forbidden filesystem operations 0; the only `outputs/` read: 020's results state (the closure
    check every phase runs);
  - **repository writes: 1**, the confirmation file (one allowed `mkdir(exist_ok=True)` of its existing directory);
  - all 19 pre-write checks true; the written bytes equal to the pre-verified payload.
- **Pre-install checks (`pre_install_checks.py/.out`):** HEAD the freeze's; the tree held exactly the untracked freeze
  file; no `outputs/experiment-101/`; no lock, calibration or stage-1 file; both hashes, the manifest counts and digest
  exact; no `.gitattributes`, no active hook, no `autocrlf`; one trailing newline.
- **Post-commit checks (`post_commit_checks.py/.out`)**, with the model, every forward and every geometry, prediction
  and outcome function refused: all 13 true — the install commit's parent is the freeze head and it touches exactly the file; the committed bytes equal the working file and both hashes and the manifest digest are exact; the runner's `validate` exits 0; 2,016 distinct keys, 144 untagged prompts and zero collisions with the 47,952 forbidden keys; the tree is clean; nothing was refused.

## 3. The artifact

- `experiments/101-read-component-intervention/confirmation-v1.json`, 135,675 bytes
- **file sha256 `6da26cd7d137510bef384af3a581e857244376c0b5825f071c069ebaabc10922`**
- **content sha256 `ef35650678e98a638838af10b75f340cec331f542589d6f089100eee8822624a`**
- the manifests: `S1-REF` 36 keys, `S2-TARGET` 1,980 keys (55 per frame), 144 untagged prompts; no collision with the
  47,952 forbidden keys (025's 43,632 spent keys plus the 4,320 untagged forms of 025's manifest).

## 4. The frames

All 60 candidates of design §11 are eligible under the tokenizer rules; the freeze takes candidates 1–36, in order,
as `EXPECTED_PICKS` and the frame-freeze guard require. `p_c` is 4 in 10 frames, 5 in 22 and 6 in 4; `p_t = p_c + 1`.
The 36 adjectives: large, soft, gray, sticky, tiny, clean, fine, damp, messy, bent, full, wet, deep, bad, thin, sour,
lean, good, pale, neat, tired, dry, cool, happy, long, cozy, dull, tidy, open, dead, loose, broad, slim, sleek, bold,
sealed. Twelve of them (soft, tiny, clean, damp, wet, thin, sour, pale, neat, dry, cool, loose) are tokens of exposed
frames; design §11 allows that and plan SA3 does not apply 025's `frame_tokens` set (novelty is at the level of the
text).

The exclusion sources (counts): earlier frame texts 162 (b0c's 144 plus 023's 18 templates); earlier cue ids 431;
025's candidate words 80 (76 single-token ids); target-noun forms 161; 020's confirmation-noun forms 48; the 42 number
words.

## Files

| file | what |
|---|---|
| `test-gate/tierA.log`, `test-gate/tierC.log` | the test gate at `40829d2` (tails) |
| `launch_freeze.py` | the guarded launcher (dry-run mode by `FREEZE101_DRY_RUN=1`) |
| `launch_freeze_v1_reconstructed.py` | the first dry run's launcher, reconstructed (not attested; see § 2) |
| `dryrun_run.json`, `dryrun.log` | the second dry run's record and output (no freeze call, no write) |
| `freeze_run.json`, `launch.log` | the production run's record and its stdout/stderr |
| `pre_install_checks.py/.out` | the checks before installing the file |
| `post_commit_checks.py/.out` | the checks after the install commit |
| `SHA256SUMS` | this directory's digests |
