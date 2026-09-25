# Experiment 024 — the production calibration (2026-09-25)

The production `calibrate` ran **exactly once**, at commit `bf0049c`, from 13:08:12.410 to 13:08:45.764 UTC, with exit
status 0.
- **Inputs:** 023's committed exposed cells and the weights alone. There was no forward pass and no prompt, fresh or
  spent.
- **Review:** the candidate record was reviewed independently (see
  [`../calibration-review-2026-09-25/REVIEW.md`](../calibration-review-2026-09-25/REVIEW.md)).
- **Installation:** installed byte for byte as `calibration-v1.json` in `1084efc`.

**The record.** `calibration-v1.json`:
- file sha256 `81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d`;
- content sha256 `09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8`.

The local results state after the run has digest `5ed26580be4c28310d072bf2c451dbeba01e4b1f8710b65072d25c9281d31252`.
Calibrate is complete; lock, confirm and report have not started; the prompt ledger is empty.

## Before the run (`precalibrate.py`, `.out`)
Read-only; every check passed:
- **Repository and freeze:**
  - HEAD and origin were `bf0049c` and the tree was clean; the stock `validate` passed.
  - The freeze hashes were exact, and the freeze rebuilt mechanically byte for byte.
  - The manifest had 4,320 keys (`fcc437fc…`), with 0 overlap with the 39,312 spent keys.
  - The module pins and 022's and 023's committed files verified.
- **Weight-derived values:** every expected value reproduced exactly with the canonical module, before any calibration
  outcome was computed.
  - Digests: parameters `fd953f1c…`, embedding `9cd6f39b…`, the 158 noun ids `08fb9b4f…`, the 139 cue ids
    `106cfecc…`, μ_noun `87930f01…`, μ_cue `f8cca954…`.
  - The 40-score digest `a703ac16…` and the calibration maximum `0.13502027836111233`.
  - 5 of the 8 E cues lie above that maximum, and every E scores above every N.

## The run (`launch_calibrate.py`, `calibrate_run.json`, `calibrate.out`)
- **Guards:**
  - every capture and intervention entry point refused;
  - torch module calls counted while the weights loaded and the programs were built, and refused from the parameter
    digest onwards;
  - an audit hook refusing 022's local table, 023's local outputs and any repository write outside
    `outputs/experiment-024/`.
- **The record of the run:**
  - 1 weights load, 0 module calls while loading, 0 refused, 0 capture calls;
  - no forbidden read or write;
  - writes only to `outputs/experiment-024/`: the state (atomic writes through temporary files), the arrays and the
    candidate record.

## The result (full precision)

| quantity | value |
|---|---|
| the exposed line, log MSE on the leave-one-out score (139 cues) | slope `1.2992017464639374`, intercept `-2.481260357420486`, residual sd `0.2570037337349923` |
| the exposed Spearman (139 cues) | `0.5296617364493499` |
| F_ρ, element [249] of 10,000 SHA-indexed draws | `0.24411074612857814`; 0 undefined draws; median `0.5285171092272343` |
| null₉₇.₅, element [97499] of 100,000 SHA-indexed permutations | `0.3136960600375234` |
| **the effective primary threshold** | **ρ ≥ `0.3136960600375234`, bound by the null** |

- **Digests:** draw indices `955930ce…`, draw values `7c03376c…`, draw flags `76b5b970…`, null values `92e63029…`,
  permutations `a68f7291…`.
- **Cross-checks:** the MSE route differs by at most 2.2e-16 (relative); the Spearman route by 0.0 over 17 checks.
- **The E–N guard** is bound (8 + 8, 12,870 assignments, pass iff K ≤ 321, ties against the guard) and was not
  evaluated.
- **The 40 fresh scores** are not in the record, by design; the lock binds them.

## After the run (`verify_calibrate.py`, `.out`)
- The candidate verifies with the frozen checker.
- The two order statistics were recomputed from the saved arrays.
- The draws, the permutations and the line were recomputed through the canonical module. This shows the code is
  deterministic, not that it matches the specification; the independent review checked the specification with its
  own implementations.
- The expected lock values were computed in memory against this record: score digest `a703ac16…`, and 5 of 8 E cues
  above the maximum.

## After installation (`post_install_checks.py`, `.out`, at `1084efc`; the lock was not run)
- The tracked, working and candidate bytes are identical, and both hashes are exact.
- The stock `validate` passes, and so does the record checker.
- The freeze binding matches in both the record and the state.
- The lock's own preconditions hold:
  - its phase rule allows the lock;
  - the only change since calibrate is non-scientific;
  - `_installed_record` reads the installed record;
  - the confirmation binding verifies.
- Every calibrated value and digest is preserved exactly.

The scripts here are archived copies; they wrote their records to the session's scratch directory.
