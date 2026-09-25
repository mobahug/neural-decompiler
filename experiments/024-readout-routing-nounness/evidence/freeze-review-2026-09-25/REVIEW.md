# Independent freeze review — 2026-09-25

**Verdict: PASS WITH NOTES, no blockers.** A separate agent reviewed the freeze file read-only, at `44c3c2b`.
- It used its own code: it did not reuse the implementer's verification logic, and it called no `Runner` phase.
- It found the existing file could be committed byte for byte without regeneration. That was done in `566eb6f`.

**The reviewed file.** `experiments/024-readout-routing-nounness/confirmation-v1.json`, 174,160 bytes:
- file sha256 `68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60`;
- content sha256 `87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87`;
- git blob `0ebacabed6605401c105569d8f985dabd4bb8d7d`, the same with and without git's filters.

The scripts, logs and records are in this directory. All ran with `rguard.py`, which refuses every repository write.

| item | result |
|---|---|
| 1. State (`review_freeze`) | **Commits.** HEAD, local `origin/main` and `git ls-remote` are `44c3c2b`, and the reflog shows no later commit. **Changes.** The only change is the untracked file; there is no `outputs/experiment-024/`, calibration record, lock, preregistration or ledger. **Timing.** Every source file predates the commit. The file was created inside the run window and never modified. **Incidents.** No incident or stop. |
| 2. Reconstruction (`review_freeze`) | **Lists.** The design table matches the module's lists (37 N words, 24 measure and 28 ordinary lemmas); the plan's written-out plurals match, and each is the regular form. **Marks.** The design's ✗ marks are exactly the ineligible entries, and its bold entries are the picks. **Selection.** An own selection with the pinned tokenizer gives exactly the 40 words and ids. The picks are mechanical; the last picks sit at ranks 9 (N), 15 (measure) and 11 (ordinary). No reserve was needed, and the rule against reusing a picked id never decided anything. |
| 3. Eligibility and exclusions (`review_freeze`) | **Chain.** 022's 303 excluded ids, plus 022's cues and frame cue ids, give exactly 023's 327. Both reproduce from the frozen inputs. With 023's 24 cues added: 351. **Other sets.** 161 target-noun form ids (80 nouns; `peach` is multi-token). 316 exposed-frame token ids across the 108 frames (36/36/36). **Selected cues.** None of the 40 ids is in any set. Each is one token with its leading space and decodes back, and no fresh word appears as text in any frame. **Rejections.** All 30 match exactly, in order and reason: N 6, measure 13, ordinary 11. teacher (9732) is a token of quantifier-016-2, farmer (24718) of cardinal-015-1, crate (47143) of cardinal-fresh-1, basket (11281) of cardinal-new-1. |
| 4. Manifest (`review_freeze`, `scan_outputs`) | **Keys.** 4,320 unique keys in the stored canonical order, in `pm.Prompt.key` format, digest `fcc437fca9730b85…`. **Spent keys.** 39,312 in total: 020's ledger 30,132, 020's confirmation set 3,060, 022's manifest 3,060 and 023's manifest 3,060. These sets are pairwise disjoint, none overlaps the new keys, and none uses a fresh id or word. **Scans.** 33,421 keys in 324 tracked files and 95,614 in the local outputs of 001–021 contain none of the 40. 022's and 023's local outputs were not opened. |
| 5. No model or outcome dependence (`static_freeze_path`, `review_freeze`) | **Code scan.** `Runner.freeze`, `_base`, `_inputs`, the tokenizer loader and 19 module functions on the freeze path hold no reference to a score, embedding, model, capture, MSE or state. **Re-run.** The same path was re-run with every open logged. Model loading, module calls, capture, `torch.load` and the safetensors loaders were blocked; all were called 0 times. **Files opened.** No model-weight file; the selection and `freeze_payload` steps opened no file at all. The tokenizer files match their pinned blobs (`tokenizer.json` `f74dfbfa…`). **Run record.** 0 loads, 0 module calls, 0 captures, no forbidden read, exactly one write. |
| 6. Contents (`review_freeze`) | **Structure.** Class order N, B, D, C, E with 8 each; all 108 exposed frames in pool order. **Sizes.** B = 10,000 and P = 100,000. **Guard.** 8 vs 8, 12,870 assignments, maximum 321, with C(16,8) and ⌊0.025·12,870⌋ recomputed. The rule reads "PASS iff K ≤ max_upper", with ties against the guard. **Pinned.** The candidate lists, rules, design, plan and model are pinned, and the top-level fields match the planned schema. **No numbers.** There is no floating-point value in the file: nounness is absent by design. |
| 7. Determinism (`review_freeze`) | `rr.freeze_payload` run in memory gives bytes identical to the file, with both hashes exact. `rr.load_confirmation_024` verifies the file with the exclusion recomputed now and `rr.PRODUCTION`. |
| 8. Expected nounness values (`review_scores`; weights only, not freeze contents) | **Model use.** 0 module calls during the load; every forward blocked afterwards. **Digests.** Parameters (113 tensors) `fd953f1c745299be…`; embedding `9cd6f39b3adfdb74…`, whose blob matches its sha256 name. **Centroids.** 158 noun-row ids `08fb9b4faf7e8a80…`, noun centroid `87930f0150e72639…`; 139 calibration-cue ids `106cfeccd7b9a103…`, cue centroid `f8cca95464092089…`. **Scores.** 40-score digest `a703ac16c01f0960…`. Calibration maximum +0.135020278 (the adjective cue "paper"). E above it: 5 of 8 (apple, horse, doctor, poet, dragon). Minimum E +0.108452 against maximum N −0.045076. **Background item 8 reproduces.** Class gap +0.1535 and mean difference +0.3201. Cosine to μ_noun +0.220 for E against +0.046 for N; to μ_cue +0.046 against +0.192. Plural shift +0.099 (B/D) and +0.121 (C/E). |

**Commit statement (the reviewer's).** Commit the existing file as it is.
- There is no `.gitattributes`, `autocrlf` or `eol` setting.
- The file is not ignored, contains no carriage returns and contains no absolute paths.
- The 024 tests use temporary roots. The tests that touch the real root pass whether or not the file exists.

## Notes (non-blocking; the frozen data is unchanged)
1. **The launcher's evidence has limits.** Its hook blocked model loading, module calls, capture, reads of 022's table
   and 023's local outputs, and writes. It did not log ordinary reads, so on its own it cannot exclude a direct read of
   the weight file. It also did not watch deletes, renames or links, and it allowed writes into `.venv`. Items 5 and 7
   close this gap.
2. **The freeze reads more than the design's phase table lists, for integrity checks only.** The shared setup reads
   020's hash-pinned local results state, 022's calibration record, 023's lock, and the bytes of `exposed-cells.f64`
   (hashed only, never parsed). None of these values reaches the selection. This is a documentation point.
3. **Provenance lives outside the file.** The freeze file does not record its code commit (`44c3c2b`) or a tokenizer
   digest, so that provenance is recorded here, in the run record and in the installation commit's message.
4. **The 024 README was stale.** It still said no scientific phase had run; it is updated with this evidence.
5. **For calibration and lock (not the freeze):**
   - An independent numpy score computation agrees with `rr` to within 2.2e-16, but its digest differs (`01cb09f5…`
     against `a703ac16…`). The lock and I7 therefore rely on `rr`'s canonical functions in the same environment, as the
     plan specifies.
   - 23 of the 40 fresh cues score above the calibration maximum: 6 B, 4 D, all 8 C, 5 E and 0 N. The design's
     extrapolation sentence names only E.

## The protocol decisions preserved with this review (reviewer, 2026-09-25)
1. **No scores in the freeze.** The freeze intentionally contains no nounness scores; it is tokenizer and text only.
2. **Nounness is bound later.** Calibration binds the embedding, the reference populations and the centroids. The lock
   binds the 40 scores, the extrapolation count and the outcome semantics. Confirm re-derives them before any fresh
   prompt.
3. **One canonical score implementation.** The independent implementation agreed to about 2.2e-16 but not byte for
   byte. The protocol relies on the canonical module for the score digest; no external implementation is required to
   reproduce the float digest bit for bit.
4. **The preregistered extrapolation statement is unchanged:** 5 of the 8 E cues lie above the calibration maximum.
5. **A descriptive pre-outcome note.** Before any outcome, 23 of the 40 cues lie above the calibration maximum (6 B,
   4 D, 8 C, 5 E, 0 N). This has no outcome force and does not alter the design.
6. **Extra reads.** The additional reads the review identified were integrity checks only and did not feed the
   selection.
