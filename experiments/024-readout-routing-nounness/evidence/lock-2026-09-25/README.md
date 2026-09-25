# Experiment 024 — the production lock (2026-09-25)

The production `lock` ran **exactly once**, at commit `be74d23`, from 17:13:39.160 to 17:14:05.116 UTC, with exit
status 0.
- **Inputs:** the committed freeze, the committed calibration record and the weights alone. There was no forward pass
  and no prompt, fresh or spent.
- **Review:** the candidates were reviewed independently (PASS WITH NOTES, no blockers; see
  [`../lock-review-2026-09-25/REVIEW.md`](../lock-review-2026-09-25/REVIEW.md)).
- **Installation:** installed byte for byte in `3affe55`, a commit that touches exactly these two paths:
  - `preregistration-lock.json`: file sha256 `5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d`,
    content sha256 `a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739`;
  - `preregistration.md`: sha256 `007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54`.

The local results state after the run has digest `58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d`
(file sha256 `14459cdd…`). Calibrate and lock are complete; confirm and report have not started; both ledgers are empty.
The installation did not touch it.

## Before the run (`prelock.py`, `.out`)
Read-only; every check passed:
- HEAD and origin were `be74d23` and the tree was clean; the stock `validate` passed.
- The freeze and calibration bytes and the state digest `5ed26580…` were exact.
- The module pins and 022's and 023's committed files verified.
- No scientific path had changed since calibrate.
- The lock's weight-derived values reproduced first, with the canonical module and every forward refused:
  - the parameter digest and the score bindings equal the record's;
  - the 40-score digest is `a703ac16…` and the calibration maximum is `0.13502027836111233`;
  - 5 of the 8 E cues lie above it, and every E scores above every N.

## The run (`launch_lock.py`, `lock_run.json`, `lock.out`)
- **Guards:**
  - every capture and intervention entry point refused;
  - torch module calls counted while the weights loaded and the programs were built, then refused from the parameter
    digest onwards;
  - `load_model` counted;
  - an audit hook refusing 022's local table, 023's local outputs and any repository write outside
    `outputs/experiment-024/`.
- **The record of the run:**
  - 1 weights load, 0 module calls while loading, 0 refused, 0 capture calls;
  - no forbidden read or write;
  - writes only to `outputs/experiment-024/`: `candidate-lock.json`, `candidate-preregistration.md` and the state
    (an atomic write through a temporary file).
- **`lock_run.json` sha256: `b7a51118ddb998fa005d7dc32cba7bce2a18278a2ec79f94ea19db973be5f5e2`.** The archived copy is
  byte-identical to the file the launcher wrote. `SHA256SUMS` lists every archived file.
- **What the record does not prove** (review note 3):
  - It is the launcher's self-report.
  - It counted `Module.__call__` only, not a direct `.forward()`, and it sealed at the first parameter digest.
  - `invocations: 1` is written by the script, not measured.

  The independent review corroborated the zero-prompt claim without it:
  - undoing only the lock's entries in the state reproduces the pre-lock state `5ed26580…` (file `4b319d62…`) byte for
    byte;
  - a sealed reload made 0 module calls;
  - `Runner.lock` has no code path to a capture or to `stage_two_022`.

## What the lock binds (full precision)

| quantity | value |
|---|---|
| the analysis module `src/neural_decompiler/readout_routing.py` | git blob `e6cb37767d1d06c6ff40804a88eab569723afdb5` (with the 12 pinned module blobs) |
| the 40 frozen scores, in the freeze's order | digest `a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995`, each with its line prediction and above-maximum flag |
| the calibration maximum | `0.13502027836111233`; exactly 5 of the 8 E cues above it (apple, horse, doctor, poet, dragon); every E above every N (`+0.108452` > `−0.045076`) |
| **the primary requirement** | **ρ ≥ `0.3136960600375234`** (F_ρ `0.24411074612857814`, bound by the null) |
| the exact E–N guard | 8 + 8, 12,870 assignments; PASS iff K ≤ 321, so K ≥ 322 fails; ties count against the guard |
| the outcome | the frozen hierarchy and semantics, the secondary text included |
| the manifest | 4,320 keys, `fcc437fca9730b8599333eeac95807786570f0d03bd01e49f52daec76ed4e32d` |

Descriptively, 23 of the 40 cues lie above the calibration maximum (6 B, 4 D, 8 C, 5 E, 0 N). This has no outcome
force.

## After the run (`verify_lock.py`, `.out`)
Read-only:
- The candidates verify against the committed freeze and calibration record, the frozen semantics and the
  configuration.
- The 40 bound scores match in order, values and digest, and the manifest has 4,320 keys.
- A rehearsal of confirm's I7 regenerated the lock's weight-derived quantities from the weights **bit for bit**, with
  every forward refused and 0 module calls during the load.

## Installation (`preinstall.py`, `.out`; `install.out`; `post_commit_checks.py`, `.out`)
- **Before the copy** (every check passed):
  - HEAD, origin and `ls-remote` were `be74d23`, and the tree was clean;
  - the candidate hashes, the state digest and file, and the freeze and calibration bytes were exact, each committed
    exactly once;
  - lock was complete once and a second lock is refused; confirm and report had not started; the ledgers were empty;
  - the outputs held exactly the five pre-confirm files.
- **The copy:** a plain `cp` of each candidate. `cmp` and `shasum` show the installed files are identical to the
  reviewed candidates. There are no git attributes and no autocrlf setting.
- **After the commit `3affe55`** (every check passed):
  - The commit touches exactly the two lock paths, and its parent is `be74d23`.
  - `git show HEAD:` reproduces the reviewed bytes, and the installed hashes are exact.
  - `render_preregistration(installed lock)` equals the committed `preregistration.md` byte for byte.
  - The stock `validate` passes and verifies the installed lock and preregistration.
  - Confirm's own pre-model path was replicated statement by statement, without calling `Runner.confirm`:
    - `_state_for('confirm')`, including the phase rule;
    - the ledger isolation and the installed record;
    - the **full `validate_lock`**, with tracked-file enforcement shown live (untracked lock files are refused);
    - the runtime check.
  - The only paths changed since `be74d23` are the two installed lock files.
  - **The direct module-blob assertion holds.** An own sha1 of the file, `rr.own_blob()`, git's HEAD blob and the
    lock's `module.blob` all equal `e6cb3776…`.
  - Every bound value above is preserved exactly. The outcome function matches the reviewed hierarchy on all 12 pairs.
    The threshold and the E–N decision rule were checked on bound values and integers only; no data was used.
  - The state is unchanged; there is no stage-2 file and no report.

The scripts here are archived copies; they wrote their records to the session's scratch directory. The post-install
verification after the evidence commit is reported separately.
