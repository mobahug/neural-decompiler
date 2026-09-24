# Independent freeze review — 2026-09-24

**Verdict: PASS WITH NOTES, no blockers.** A separate agent reviewed the freeze file read-only, at `89536b0`, with a
clean-room reconstruction; `r2_independent` never imports `neural_decompiler`. It found that the existing file could be
committed byte for byte without regeneration; that was done in `a9287ec`.

Reviewed file `experiments/023-block0-completion/confirmation-v1.json`: 128,228 bytes, file sha256
`5fadfa503f4fe35308cb4473220a1d9cd6f7a46e3852f638594b8081909825f4`, content sha256
`4e64d4c2c85ae8f9c710171372a4bf28f0964a8e8864a0326182edba44aa14ed`.

| item | result |
|---|---|
| 1. State (`r1_state`) | HEAD, origin and `ls-remote` are `89536b0`; the only change is the untracked file. The results-state digest recomputes. Extract ran once; the later phases are `not_started`; the ledgers are empty. The installed exposed cells equal the committed bytes. The stock `validate` refused only because the file was untracked (`validate.out`). |
| 2. Reconstruction (`r2_independent`) | The ordered lists were parsed from the design text and equal the code's. The checker tested every candidate itself; all are eligible, so the picks are mechanically ranks 0–7 and 0–5, the 24 cues and ids and 6 frames per template as frozen. Every frame field matches, and `rejected` is empty. |
| 3. Eligibility (`r2`, `r3_crosscheck`) | Each `" "+word` is one token; without the space, 7 words have another id and 17 split. For frames, `p_c` is the prefix length; cue-final `p_t = p_c` (3–4), coordinated `p_t = p_c + 1` (5–7), all within 022's ranges. The reference ids match the pool, `pm._build_new_frame` agrees on all 39 candidates, and all 3,024 target texts tokenize as prefix + cue + suffix. |
| 4. Freshness (`r2`, `r4_scan`) | The exclusion rebuilt from the raw files equals 022's own (303 ids, 126 texts); with 022's units it is 327 ids, 144 texts and 15 sources, equal to the file. Overlap is 0 with 020's pool, 020's confirmation set (021's spent set) and 022's units. No ledger from 005 to 022, and no committed `token_id`, uses the 24 ids. The scope statement is correct. |
| 5. Manifest (`r2`) | S1-REF 18, S1-VALIDITY 18, Y1 2,592, Y2 432: 3,060 unique keys, sorted, equal to the file. There are 0 collisions with the 023 ledger and with the 36,252 forbidden keys. Every target has a new cue; every stage-1 key has a new frame. |
| 6. No model dependence (`r3_crosscheck`) | Only the tokenizer, the design's lists and the committed cue and frame sets reach a selection decision; nothing on `freeze`'s code path loads the model. The run's `load_model` and `nn.Module.__call__` refusals never fired. The Hugging Face files opened were `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` and `config.json`; the weights blob `3da38833…` is absent. |
| 7. Determinism (`r2`) | The file rebuilt from the checker's own reconstruction and canonical JSON is byte-identical. Only constant text (rule strings, `kind`, scope, the design binding) was taken from the frozen code, and it was parsed, not imported. The model binding is `pythia-70m-deduped` at `e93a9faa…`. |

**What the weights evidence establishes, stated precisely:**
- the freeze code path contains no model-loading call;
- the independent reconstruction is tokenizer-only and byte-identical;
- the observed Python-level file accesses contain tokenizer and config files and not the known weight blob.

The launcher's original name-based weight guard was ineffective against Hugging Face's hash-named blobs, so it is not
relied upon as evidence. The audit hook also sees only Python-level opens.

**Notes (non-blocking; the frozen data is unchanged).**
1. `" shiny"` (30006) is the adjective at `p_t` of exposed frame `coordinated-adjective-009-1` ("Ida and Tomas counted
   {cue} shiny"). So the Y1 target shiny × that frame has the same token at `p_c` and `p_t`: 1 of 864 coordinated Y1
   pairs, 1 of 3,024 targets.
   - The frozen exclusion covers earlier cue slots only, so it is kept.
   - The rule text recorded in the file ("whose id no earlier experiment … used") claims more than that.
   - Confirm reports this pair explicitly. A descriptive result without it may be shown as a secondary check; the
     frozen population remains the primary analysis.
2. `" batches"` (39657) is the plural of Experiment 020's fresh noun "batch". It was never a cue and is not one of
   023's 79 nouns; no action is needed.
3. Freeze reads more than the design's phase table lists. Through the frozen-input loader it also reads prior numerical
   artifacts: 020's results state (about 1.57 M floats), closure and exploration record; the 011/012/017 locks; and
   022's calibration record.
   - These feed only integrity, structure and isolation checks, which can only abort the freeze.
   - No numerical value from them takes part in candidate eligibility, ranking or selection, as the byte-identical
     rebuild without them shows.
