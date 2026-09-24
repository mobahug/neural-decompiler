# Archive: Experiment 020's results state (a hash-pinned input of Experiment 023)

Experiment 020's local results state, `outputs/experiment-020/results.json`, is gitignored. Every Experiment 023
phase reads it through the frozen-input loader (`ul.load_frozen_inputs` → `rc.verify_020_closure`), because it is the
only copy of:
- 020's locked exposed reference states, which extract's E4, lock's Y1 table and confirm's I7 are built from;
- its ledger, which the isolation checks use;
- its template bases.

The file is **hash-pinned** in three places:
- the literal constants `RESULTS_020_FILE_SHA256` and `RESULTS_020_STATE_SHA256` in the pinned
  `src/neural_decompiler/readout_calibration.py` (blob `9107da97…`);
- Experiment 020's committed `experiments/020-readout-decompilation/closure.json` (`explore.results_file_sha256`,
  `explore.results_state_sha256`), whose locked-state digests the committed exploration extract also binds;
- Experiment 023's results state and calibration record (inputs `results_020_file`, `results_020_state`).

This archive **preserves** the exact bytes, so that an independent rerun does not depend on a working-tree copy.

| file | bytes | sha256 |
|---|---|---|
| `experiment-020-results.json.gz` (`gzip -9 -n`, deterministic) | 14,252,155 | `69d726ac54b86ea98cc49a3e995de631875fb5417771b2212bd99cbe6fef6f86` |
| its decompressed content = `outputs/experiment-020/results.json` | 32,376,872 | `da63b8c29f9553a9da62bea7a442e11cef999ccddc71a7a332ac7c938abc9e00` |

Its results-state digest (`state_sha256`) is `2e5485dccb39c018eb02ee8d0a3086004399690a4dfd9c114008f057b23bf8f5`.

**Recovery.**

```bash
mkdir -p outputs/experiment-020
gunzip -c experiments/023-block0-completion/evidence/archive/experiment-020-results.json.gz > outputs/experiment-020/results.json
shasum -a 256 outputs/experiment-020/results.json   # must print da63b8c2…
```

The loader re-verifies both digests on every read and refuses any other bytes.
