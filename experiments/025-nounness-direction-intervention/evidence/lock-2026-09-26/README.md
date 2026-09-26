# Experiment 025 — the production lock (2026-09-26)

**The official lock run** is `lock_run.json` / `lock.log`:
- It ran exactly once, at commit `d3ecbd6feb1bd2aefe08dc0f8d799815be47c519`, 16:51:22–16:51:50 UTC, exit status 0.
- It used the guarded launcher `launch_lock.py`, sha256
  `f66c12f2357d4fec982a6e4ce7b12de71793291dc46121f1d32e4fae84e85856`.
- It was weights only: **1 model load; 0 module calls, `.forward()` calls, captures, interventions, confirm-only
  computations, or refused or forbidden events.**
- It wrote only `outputs/experiment-025/` (`candidate-lock.json`, `candidate-preregistration.md`, and `results.json`
  via one temporary file and rename).

The candidates were independently reviewed (`../lock-review-2026-09-26/`, **PASS WITH NOTES — no blockers**). They
were installed by plain byte copy in `5ee7a7474c5531bf7a7c19ceff4c8ca857d16bf5`:

| file | sha256 |
|---|---|
| `preregistration-lock.json` | file `1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be`; content `8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb` |
| `preregistration.md` | `e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919` |
| `outputs/experiment-025/results.json` (local, not committed) | `state_sha256` `8eeb8184faf556211e68579d4682c85f545c877051641afe07dd5eac2b701dad`; lock complete, confirm and report not started, ledger empty |

Confirm and report have not run. No fresh Experiment 025 prompt has run.

## The run, in order

1. **`prelock.py`/`.out`: 18 read-only checks before the lock, all true.**
   - `HEAD == origin == remote == d3ecbd6`, and the tree was clean.
   - Stock `validate` passed; the freeze and the manifest were exact.
   - There was no results state, lock or `outputs/experiment-025/`.
   - The pinned blobs and the 020–024 re-check matched, as did the runtime and versions against 020's explore.
   - The model binding was the freeze's and 024's lock's, and the local snapshot was present.
2. **The guarded launcher (`launch_lock.py`).**
   - **The environment record, taken before the model load:**
     - the full HEAD, the tree state, the repository path and the launcher's sha256;
     - Python 3.12.13 (`.venv`), macOS-15.7.3-arm64;
     - torch 2.14.0, transformers 5.17.0, transformer-lens 3.9.0, huggingface-hub 1.31.0, numpy 2.5.3;
     - 4 torch threads; cpu/float32;
     - model and tokenizer revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`;
     - `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`.

     The checkpoint snapshot's files:

     | file | sha256 | bytes |
     |---|---|---|
     | `config.json` | `002050231a9b1ec3ac77aa6b9b3bbdc4d923f4068a7dd33b8da72a9bd6ad9a43` | 567 |
     | `model.safetensors` | `3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd` | 166,029,852 |
     | `special_tokens_map.json` | `6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad` | 99 |
     | `tokenizer.json` | `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624` | 2,113,710 |
     | `tokenizer_config.json` | `70e38394e494931c6f773ba41e19460dd4436526b852207367f04341b4066d3f` | 396 |
   - **The guards:**
     - `torch.nn.Module.__call__` refused for the whole run;
     - every module instance's `forward` sealed right after the load (430 modules), and the model class's forward,
       run_with_hooks, run_with_cache and generate refused;
     - every capture and intervention entry point refused, including all 8 aliases found by identity;
     - every confirm-only computation refused;
     - an audit hook refusing forbidden reads, and any repository write, delete, rename, link or mkdir outside
       `outputs/experiment-025/`.
3. **Dry run 1 (`dryrun1_run.json`, `dryrun1.log`) — provenance only, non-scientific, resolved before production.**
   - It was launcher version `cdc466f3…` (that file was not kept; the fix below is the only difference). It loaded
     the weights and built the programs; it did not call `lock` and computed no 025 geometry.
   - Its audit hook refused two deletions by `filelock`'s import-time probe. The probe unlinks `probe-source` and
     `probe-link` inside a temporary directory by `dir_fd`, and the hook wrongly resolved the names against the
     repository root.
   - That was a false positive in the guard: nothing in the repository was touched, and the only effect was that
     library's probe result in that one process.
   - The hook was fixed to resolve `dir_fd`-relative names (macOS `F_GETPATH`).
4. **Dry run 2 (`dryrun_run.json`, `dryrun.log`)** was clean, on the fixed launcher `f66c12f2…`: zero refused or
   forbidden events, 1 load, 430 modules sealed.
5. **The official lock (`lock_run.json`, `lock.log`),** as above.
6. **`verify_lock.py`/`.out`/`.err`: the operator's read-only post-lock verification, weights only, 33/33 checks.**
   - The geometry recomputes bit for bit in a separate process.
   - The 280 random controls on the actual float32 patches:
     - neutrality max `6.088945903037768e-09` against 1e-6 (float64: 1.7e-16);
     - |d·u| ≤ 5.6e-17;
     - angle to θ ≤ 3.7e-9.
   - The nounness odd component: float32 max 1.5e-9 (primary) and 2.1e-9 (half).
   - The combined digests: controls `e12581d4…`, 840 float32 vectors `30a35c00…`, `d` `5f914283…`.
   - The model digests equal 024's lock.
7. **`preinstall.py`/`.out`:** 15 checks before the install, all true.
8. **`post_commit_checks.py`/`.out`:** 20 checks after the install commit, all true. They include:
   - byte identity with the candidates;
   - rendering of the installed lock;
   - stock `validate` ("lock and preregistration verified");
   - the **full `cr.validate_lock`** against the installed artifacts, with `git_state` taken from the asserted
     repository root;
   - the bindings (`d`, the neutrality maximum, the manifest, 27/40, the tail, Level-1 on the 16 conditions, the
     outcome table, and the `D_attn` reference-state digest `26c21d63…`).

## Requirements for the confirm launcher (the lock review's notes; recorded, with no scientific code changed)

- **N1 — the working directory.** `collect_git_state()` runs git in the process's working directory when no path is
  given. Before any model load or fresh execution, the confirm launcher must:
  1. resolve the repository root;
  2. `chdir` to that exact root, and assert `cwd == repo_root`;
  3. run every git and provenance check there;
  4. assert HEAD, `origin/main` and the remote from that root.

  A mismatch stops before the patch-path check, the ledger or any fresh prompt.
- **N2 — the environment identity.** Confirm must run on the same machine and stack as the lock:
  - macOS arm64, Python 3.12.13, torch 2.14.0, and the same remaining pinned package versions (above);
  - 4 torch threads;
  - the same local checkpoint revision and file digests (above).

  No `uv sync`, package upgrade or environment recreation before confirm: I7′ must reproduce the lock's geometry bit
  for bit.
- **N3 — forward routes.** The confirm launcher allows a forward only inside a sanctioned call: a fresh
  condition-tagged run, or one of the four spent patch-path runs. It blocks and logs everything else:
  - every module instance's `.forward`, the relevant class-level `.forward` methods, and `__call__`;
  - hook, cache and generate routes;
  - patched and unpatched capture and intervention aliases.
- **N4 — the write scope.** A stricter allow-list than the lock's:
  - the expected Experiment 025 output paths and state writes are enumerated explicitly;
  - unexpected writes are blocked even inside `outputs/experiment-025/`;
  - deletes and renames are blocked, except the known atomic temporary-file → target operations.
- **The operational safeguards (the reviewer's):**
  - record its own sha256 and the full environment;
  - log every fresh patched execution with its condition tag;
  - identify the four spent patch-path runs separately, and forbid them from entering the fresh ledger or
    accounting;
  - I7′ before the patch-path check, and the patch-path check before the fresh ledger;
  - the incident before any partial save;
  - never retry confirm automatically.

The completed lock remains valid; no scientific code is changed to alter it retroactively.

## Files

| file | what |
|---|---|
| `prelock.py/.out` | the checks before the lock |
| `launch_lock.py` | the guarded launcher (`f66c12f2…`; dry-run mode by `LOCK025_DRY_RUN=1`) |
| `dryrun1_run.json`, `dryrun1.log` | dry run 1: the guard false positive, provenance only |
| `dryrun_run.json`, `dryrun.log` | dry run 2: clean |
| `lock_run.json`, `lock.log` | **the official lock run**: the environment record, the events, the outputs |
| `verify_lock.py/.out/.err` | the post-lock verification (33/33) |
| `preinstall.py/.out`, `post_commit_checks.py/.out` | the install checks (15/15, 20/20) |
| `SHA256SUMS` | this directory's digests |
