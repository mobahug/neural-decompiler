# Independent extraction review — 2026-09-24

**Verdict: PASS WITH NOTES, no blockers.** A separate agent reviewed the candidate read-only, at `2111271`, with a
checker it wrote itself. It did not use 023's extraction or verification functions for items 1–7. It used 022's frozen
functions only for the program definition (item 7), and 023's reader and kernel only in item 8, where it compared them
with its own code. It found that the exact candidate pair could be installed byte for byte and committed without
regeneration; that was done in `83d9c58`.

Reviewed candidate:
- `exposed-cells.f64`: sha256 `d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4`, 1,209,600 bytes;
- `exposed-cells.json`: file sha256 `628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe`, content
  `d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16`.

| item | result |
|---|---|
| 1. State (`a_state_artifact`) | The state digest recomputes with the checker's own canonical JSON. Extract ran once with no incident or stop; the later phases are `not_started`; the ledgers are empty. |
| 2. Integrity (`a_state_artifact`) | The three hashes match. The data is 151,200 finite little-endian float64 values, with `n = 79` in every row, and the index binds the data consistently. |
| 3. Order (`b_table_checks`) | The cues and frames equal the table's own lists and 022's ordering rule. The tokenizer confirms all 175 cue ids; the masks are 14 and 30; the 79 nouns are in the pool's order. The 36 pronoun cues (3,888 rows) are kept. The reimplemented draw formula equals 023's for all 10,000 draws, and no draw selects a pronoun cue. |
| 4. Provenance (`b_table_checks`) | The table is 424,131,002 bytes, `04659d5e…`. The recomputed 13 tensor digests equal 022's record, and a one-ulp change is detected. The record and confirmation digests recompute. The module blob `16d310fc…` and all 11 pins equal HEAD. |
| 5. Reconstruction (`b_table_checks`) | Over 18,900 rows: 0 differing rows against 022's stored values and against the checker's own reductions, for every column. Against `math.fsum`, the largest difference is 4.2e-16 relative. |
| 6. SST (`b_table_checks`) | 409 selections with repeated pairs: one-pass against the merge, 2.3e-15; one-pass against direct per-noun values, 2.2e-15; merge against direct, 2.2e-16. E6's tolerance is 1e-10. |
| 7. E4 recheck (`c_e4_recheck`) | P0 and P1 from the weights, under the `pm` guard and a `torch.nn.Module.__call__` refusal (neither fired), are bit-identical over 1,493,100 values each. SSE0 and SSE1 match the artifact. |
| 8. Independence (`d_calib_independence`) | In a separate process an audit hook blocked every route to 022's table (8 probes refused). The reader and kernel on 16 draws × 4 conditions agree with the checker's own code within 2.2e-15; the result codes used an illustrative bound, not an envelope. The guard fired 0 times after the probes. |

Unchanged afterwards (`e_final_unchanged`): the outputs, 022's table and the git status. The artifact also reproduces
the design spike's R² and gap-closure table (`f_spike_consistency`).

**Notes (non-blocking).**
1. The README said "not run"; it was updated at installation.
2. E4 is exactly 0.0, better than the spike's 7e-12.
3. The empty `"at"` in E4's record is cosmetic: 022's max-tracker records a location only when the maximum rises
   above 0.0.
4. Every phase still reads Experiment 020's git-ignored `outputs/experiment-020/results.json` through
   `load_frozen_inputs`. This is inherited from 020–022 and outside 023's table-independence claim.
5. Install byte for byte and change nothing under `src/` before `calibrate`; both were observed.
