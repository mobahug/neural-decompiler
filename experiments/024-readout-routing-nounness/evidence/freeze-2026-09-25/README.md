# Experiment 024 — the production freeze (2026-09-25)

The production `freeze` ran **exactly once**, at implementation commit `44c3c2b`, from 09:34:55.674 to 09:35:05.812 UTC,
with exit status 0. It was tokenizer and text only. The file it wrote was reviewed independently (see
[`../freeze-review-2026-09-25/REVIEW.md`](../freeze-review-2026-09-25/REVIEW.md)) and installed byte for byte in
`566eb6f`.

**The file.** `confirmation-v1.json` is 174,160 bytes.
- file sha256 `68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60`;
- content sha256 `87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87`;
- git blob `0ebacabed6605401c105569d8f985dabd4bb8d7d`.

## What ran, and under which guards (`launch_freeze.py`, `freeze_run.json`)

The launcher installed its guards before the runner module was imported:
- `neural_decompiler.models.load_model` refused, so the model was never loaded and no weight, logit, hidden state or
  attention existed;
- `torch.nn.Module.__call__` refused;
- every capture and intervention entry point of `plural_mechanism` refused;
- an audit hook refused opening 022's local calibration table or anything under `outputs/experiment-023/`, and refused
  any repository write except the one confirmation file.

It then ran the stock `Runner().freeze()` once, with the production configuration. The run record shows:
- 1 invocation, exit status 0, no error;
- 0 model loads, 0 module calls and 0 capture calls;
- no forbidden read or write, and exactly 1 write, the confirmation file.

The freeze creates no results state, and none exists.

**A limit of this evidence.** The launcher's hook did not log ordinary reads, so on its own it cannot exclude a direct
read of the weight file. The independent review closed this: it re-ran the same path with every open logged, and no
model-weight file was opened.

## The frozen population (`verify_freeze.py`, `.md`, `.json` — the implementer's post-freeze check, read-only)

| class | cues (token id) |
|---|---|
| N | honest 8274, polite 30405, rude 30446, sleepy 48849, wise 15822, lucky 13476, merry 49570, nervous 11219 |
| B | gallons 42616, ounces 28409, acres 20046, herds 47862, crowds 24597, bundles 25663, clusters 9959, litres 47026 |
| D | gallon 46740, ounce 38831, acre 36982, herd 33361, crowd 9539, bundle 13204, cluster 7368, litre 43803 |
| C | apples 28580, horses 12074, doctors 11576, kings 25346, rabbits 29948, poets 32976, dragons 41705, lions 44536 |
| E | apple 19126, horse 8815, doctor 7345, king 6963, rabbit 17876, poet 17502, dragon 22159, lion 27405 |

- **Selection:** these are exactly design revision 2's expected picks, taken mechanically as the first eligible entries
  of each frozen list. No reserve was needed. The class comes only from the curated lists.
- **Lemma pairs:** B/D and C/E are lemma pairs position by position, and each plural is its lemma's listed plural.
- **Tokens:** each cue is one token with its leading space and decodes back, and all 40 ids are distinct.
- **Exclusions:**
  - 351 earlier cue ids: 023's 327, plus 023's own cues and its frames' cue ids;
  - 161 target-noun form ids;
  - 316 exposed-frame token ids.

  None overlaps the 40 ids.
- **Rejections:** 30 in total (N 6, measure 13, ordinary 11), each with its reason. The frame-token rule removed teacher,
  farmer, crate and basket.
- **Reserves:** N 23, measure 3, ordinary 9. They are recorded and were never used.
- **Manifest:** 4,320 S2-TARGET keys (40 cues × 108 exposed frames).
  - They are unique, in canonical sorted order, and every key carries a fresh cue.
  - Manifest sha256 `fcc437fca9730b8599333eeac95807786570f0d03bd01e49f52daec76ed4e32d`.
  - There is 0 overlap with the 39,312 spent keys: 020's ledger, 020's confirmation set, 022's manifest and 023's
    manifest.
- **Configuration bound in the file:** the production configuration, including 8 per class, B = 10,000, P = 100,000,
  and the E–N guard (8 + 8, 12,870 assignments, pass iff K ≤ 321, ties counted against it).

**Expected values for calibration and lock (not freeze contents).** Part 2 of the check was computed after the file
existed, from the weights only: 0 module calls during the load, and every forward refused afterwards. The freeze
contains no nounness score by design. These are the values calibration and lock must reproduce with the canonical
module:
- parameter digest `fd953f1c…`, embedding digest `9cd6f39b…`;
- μ_noun: the 158-id list `08fb9b4f…`, centroid `87930f01…`;
- μ_cue: the 139-id list `106cfecc…`, centroid `f8cca954…`;
- the 40-score digest `a703ac16…`;
- the calibration maximum +0.135020;
- E above it: 5 of 8 (apple, horse, doctor, poet, dragon);
- every E scores above every N.

## After installation (`post_commit_checks.py`, `.out`, run at `566eb6f`)

Read-only, with model loads, forwards and repository writes refused:
- The tracked blob is `0ebacabe…`, and the bytes at HEAD equal the reviewed bytes, with both hashes exact.
- The stock `validate` accepts the tracked freeze (exit 0).
- The freeze rebuilt mechanically in memory with the real tokenizer is byte-identical to the committed file, with the
  same 40 cues.
- The manifest has 4,320 unique keys, with 0 overlap with the 39,312 spent keys.
- `outputs/experiment-024/` does not exist.

The scripts here are archived copies; they wrote their records to the session's scratch directory. Re-running one in
place would be refused by its own guard, which permits no repository write.
