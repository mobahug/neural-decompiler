> **Archived verdict: PASS WITH NOTES — no blockers.** The reviewer accepted it on 2026-09-26 and authorized installing the exact candidates. They were installed byte for byte in `5ee7a7474c5531bf7a7c19ceff4c8ca857d16bf5`.
> Everything below the rule is the independent reviewer's report, unchanged. Before this header was added, its sha256 was `991102b145e54cd70ecac050ba3c56e3211265b7fe3f612f18166b40a55836e1`. Its notes N1–N4 became requirements for the confirm launcher, recorded in `../lock-2026-09-26/README.md` and in the experiment README.

---

# Experiment 025 lock: independent verification

**Verdict: PASS WITH NOTES. No blockers.**

The exact existing `outputs/experiment-025/candidate-lock.json` (file sha256 `1881a790…`, content `8e5300de…`) and
`outputs/experiment-025/candidate-preregistration.md` (sha256 `e4b12656…`) **can be installed byte for byte, with no
regeneration**, as `experiments/025-nounness-direction-intervention/preregistration-lock.json` and `preregistration.md`.
See the install statement at the end.

- Reviewed: the lock that ran once at `d3ecbd6feb1bd2aefe08dc0f8d799815be47c519`, its three candidate files, its launcher
  and records in `scratchpad/lock025/`, and the design (`c0885e5` and `26c9925`), plan (`7d90d28`), `cue_rotation.py`,
  runner and pinned modules they bind.
- Every quantity the lock binds was rebuilt by my own code **before** any `cr` geometry function could be called. The
  functions were refused in that process by a guard. Every per-cue scalar, vector and digest agrees **bit for bit** with
  `cr` and with the lock: maximum disagreement 0.0.
- The critical item 8 holds on the actual float32 patched vectors. For the 280 random controls, `|s(+θ) − s(−θ)|` is at
  most `6.088945903037768e-09`, against the 1e-6 bound.

## How this review stayed inside the rules

- **Nothing in the repository was written, staged or committed.**
  - Every review process ran under an audit hook (`guard.py`). The hook refuses any repository write, delete, rename,
    link, mkdir or truncate, the `.venv` included. It also refuses any open or listing of `outputs/experiment-023/`,
    `outputs/experiment-024/` or `calibration-table.pt`.
  - Every script's guard log shows zero forbidden reads, zero refused writes and zero refused filesystem operations.
  - Afterwards `git status --porcelain --untracked-files=all` is empty, HEAD is still `d3ecbd6`, and the three candidate
    files keep their hashes.
- **No freeze, lock, confirm or report ran.** Only `Runner.validate()` and its read-only helpers ran (`_base`,
  `_confirmation`, `_noun_keys`); the runner's model and tokenizer loaders were replaced by refusals.
- **No fresh or rotated prompt ran.** No forward ran on any 025 key, and no rotated vector was ever given to the model.
  No ℓ, MSE, A, B, G, `D_attn` or Level-1 value was computed for any run. Every such number in `r5_logic` is made up by
  the script to probe the code.
- **All but one model use was weights-only.**
  - `torch.nn.Module.__call__`, every capture and intervention entry point (and their aliases, found by identity) and
    every confirm-only computation refused. Every module of the loaded model was sealed.
  - The one exception is item 10's allowance. `r4_dattn_binding.py` made exactly **6 plain captures of spent 020
    reference prompts, 2 per template family**, through a single door:
    - each key had the form `frame_id|ref|<reference id>` and was checked to be in 020's ledger;
    - each capture took `ATTN_PATTERN.L4/L5@p_t` only;
    - only equality flags were recorded.
  - A non-reference key (a spent key, not an 025 one) was refused by the door before it ran.
  - `r4` was run once and never re-run.
- **Environment:** `.venv/bin/python` with `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`, and `git --no-optional-locks`
  for status. `uv` was never used.
- **Disclosure (procedural).**
  - The final confirmation that no repository file changed during the review was a
    `find . -path ./.venv -prune -o -newer PROMPT.md -type f -print`. It did not prune `outputs/experiment-023/`,
    `outputs/experiment-024/` or 022's `calibration-table.pt`.
  - `find` therefore listed those directories and compared file modification times (metadata only). It opened no file
    and printed nothing, because no file was newer than the brief.
  - No content under those paths was read at any point. Every content search excluded them.

## Blockers

None.

## Notes (non-blocking; most severe first)

**N1. The clean-tree check at confirm depends on the process working directory.**
- **Where:** `Runner.git_state` defaults to `collect_git_state()` (`run.py:149`, used at `run.py:265-267` and
  `run.py:420`). That function runs `git rev-parse HEAD` and `git status --porcelain` in the **current working
  directory** (`provenance.py:193-214`), not in `ROOT`.
- **Scenario:** a confirm launcher started with its working directory inside a different Git work tree would check
  that tree's cleanliness.
  - Outside any work tree, `_state_for` still refuses, because the commit is not a 40-character SHA
    (`run.py:273-274`).
  - The lock evidently ran from the repository: the state records `d3ecbd6` and `git_dirty: false`.
- **Side effect:** that `git status` runs without `--no-optional-locks`, so it may refresh `.git/index`. This write is
  invisible to a Python audit hook. It is benign.
- **Suggestion:** have the confirm launcher `chdir` to `ROOT`, record `os.getcwd()`, and assert `git_state()["commit"]`
  equals `git -C ROOT rev-parse HEAD` before calling `confirm`. No code change is needed.

**N2. I7′ is bit for bit, so the software and hardware stack must not change before confirm.**
- **Where:** `run.py:436-440` and `cue_rotation.py:986-989`.
- **Which parts are exposed:**
  - the controls use Python's `math.log/cos/sin/sqrt` for Box–Muller (`cue_rotation.py:460-469`);
  - θ uses `math.asin`;
  - the rest uses torch float64 kernels.
- **What I verified:** they reproduce bit for bit in fresh processes under different seeds for every random-number
  generator, under different `PYTHONHASHSEED` values and under 1, 4 and 8 torch threads (`r3_*`).
- **Scenario:** an OS/libm or Python rebuild between lock and confirm could still flip a last bit. The result would be a
  pre-prompt I7′ incident: fail-safe, but it spends the protocol version.
  - Torch and the other pinned packages are already refused on version drift (`_check_runtime`).
- **Suggestion:** run confirm on the same machine (macOS 15.7.3 arm64, Python 3.12.13, torch 2.14.0) without updates in
  between, and record `platform.platform()` in the confirm launcher.

**N3. The lock launcher's forward seal covers the model's class and every module instance, but not the submodules'
classes.**
- **Where:** `launch_lock.py:226-231`.
- **Scenario:** a call `type(sub).forward(sub, x)` on a submodule would bypass both the instance seal and
  `Module.__call__`. Nothing in the lock path does this.
- **Evidence:**
  - the code path contains no forward;
  - all refusal counters were zero;
  - `r6` confirms that every other route (`module(x)`, `module.forward(x)`, aliases of capture and intervention
    functions) is refused.
- **Suggestion (future weights-only phases only):** also seal `forward` on each submodule class.

**N4. The launcher's audit hook allows any write inside `outputs/experiment-025/`.**
- **Where:** `launch_lock.py:91-94`.
- **What rests on it:** the claim of "only the three expected files" rests on the recorded `output_writes` and
  `output_renames` and the final directory listing, not on a refusal.
- **Evidence:** both show exactly `candidate-lock.json`, `candidate-preregistration.md` and `.results-v87zujap.json`
  renamed to `results.json`. The directory holds exactly these three files, with no hidden leftovers.
- **Suggestion:** none needed. For confirm, the expected files are `stage2-measurements.pt`, `results.json` (and
  possibly `stage2-partial.pt`).

**N5. The lock phase reads more than the design's phase table lists.**
- **What it reads:** `_base()` loads:
  - 020's local results (the ledger and locked states);
  - 022's and 023's committed files;
  - 020's confirmation file.

  It also builds the weight-derived programs and hashes 020's locked states for `dependencies.readout_020`
  (`run.py:363-387`).
- **Why it does not matter:** these are integrity and binding reads of spent or committed data, the same kind the
  reviewer accepted at the freeze (freeze review N3). They carry no 025 outcome information, and the state digest they
  produce is what item 10 requires.

**N6. `_model_dependencies` does not compare `model_id` or `revision`.**
- **Where:** `run.py:359-361`, called at `run.py:430` and `434`. The recorded dict is compared with itself for those
  keys, so only the parameters and embedding digests are compared.
- **Why this is not a gap:**
  - the revision is enforced by `load_model`'s resolved-revision check against `models.py`, whose blob is pinned
    (`b1c6f033…`);
  - the weights are bound by the parameters digest `fd953f1c…`.

**N7. The brief's combined 840-vector digest depends on the stacking shape.**
- `30a35c00…` is the `rc.tensor_digest` of the `[40, 21, 512]` stack (cues in frozen order, conditions in frozen order).
- `rc.tensor_digest` includes the shape, so a `[840, 512]` stack gives `a8d1baa3…`.
- The control digest `e12581d4…` is the `[40, 7, 512]` stack, as the brief states.

**N8. Informational.**
- **Execution order.** `stage_two` executes frame-major (`cue_rotation.py:1105-1123`), while the plan says "for each
  pair in `ul.table_units` order" (plan line 219). The README (`25377d0`) documents this. The forwards are independent
  and the tensors are stored in `table_units` order, so nothing changes.
- **Non-finite values.** `count_positive` counts `+inf` against (`cue_rotation.py:1290-1292`). This is unreachable,
  because `per_cue` raises on a non-finite MSE.
- **Cosmetic.** The preregistration prints an **A** line twice: the formula, then the semantics
  (`cue_rotation.py:917-920`).

## Key numbers

| quantity | value |
|---|---|
| HEAD = origin/main = remote `refs/heads/main` | `d3ecbd6feb1bd2aefe08dc0f8d799815be47c519`; tree clean |
| candidate lock file / content | `1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be` / `8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb` |
| candidate preregistration | `e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919` (byte-equal to `cr.render_preregistration(lock)`) |
| results state (`state_sha256`) | `8eeb8184faf556211e68579d4682c85f545c877051641afe07dd5eac2b701dad` (file `05f17a42…`); lock complete, confirm/report `not_started`, ledger empty |
| freeze file / content / manifest | `54c8947c…` / `6eca7024…` / `0e1f068b…` (90,720 keys) |
| `cr` blob (lock, file, `git ls-tree`, `cr.own_blob()`) | `622aa832ae29047c3060d989928236320374e227` |
| model parameters / embedding | `fd953f1c…` / `9cd6f39b…` (my own code; the embedding also from raw safetensors fp16→fp32) |
| 020 exposed states | `26c21d6386ebfaf45cd33f1e9ceaae08a225d5c4ae3a5a9476e28bf1683b0f83` |
| `d` digest, \|d\| | `5f914283c39d2f37911ba3bb1391a98bbe3625737f8df80c42eb336868a1bf2a`, `1.3576567483118904` (cos(μ̂_noun, μ̂_cue) 0.07838) |
| `p̂` digest, cos(d̂, p̂) | `2dd4a1f4…`, `0.23521413845112732` |
| combined float32 vectors `[40,21,512]` / controls `[40,7,512]` | `30a35c0050421b0b07579ee5827e23386dde95ec777aba8ac36a940551df7ca9` / `e12581d4be183798b42e2f673670eda711ddaf87359c28df5350c1c806a37315` |
| τ, s₀, θ, θ_half, even (primary) | 1.32275–1.35597; −0.30589…+0.28521; 13.6501–13.9999°; 6.7765–6.9475°; −0.00841…+0.00909 |
| float32 neutrality (controls and plurality) | max `6.088945903037768e-09` (bound 1e-6) |
| float32 odd component, length, angle | 2.11e-9, 6.03e-9, 3.69e-9 rad (bounds 1e-6) |
| float64 checks (unit, orthogonality, d·u, neutral, odd, length, angle) | ≤ 6.7e-16 (bounds 1e-12) |
| Binomial tail | P(X ≥ 27) = 5288280983/274877906944 = 0.01923865414210013; P(X ≥ 26) = 0.0403452… |
| spent set / collisions | 43,632 keys (020 ledger 30,132 ∪ 020 confirmation 3,060 ∪ 022 3,060 ∪ 023 3,060 ∪ 024 4,320); 0 collisions |
| maximum disagreement, mine vs `cr` vs lock | 0.0 for every scalar, vector, control, uniform and Gaussian; angle checks ≤ 1.7e-16 (a different angle formula, by design) |

## Item by item

### 1. Repository and phase integrity: PASS

Evidence is in `r1_integrity.out.json`, item 1.

- **Git state.**
  - HEAD, `origin/main` and `git ls-remote origin refs/heads/main` are all `d3ecbd6…`.
  - `status --porcelain --untracked-files=all` is empty.
  - The reflog shows no commit after 16:38 UTC; the lock ran from 16:51:22 to 16:51:50 UTC.
- **The installed freeze is unchanged.**
  - The file sha256 is `54c8947c…`.
  - The blob is `6e0488cb…` in my own computation, at HEAD, and at the install commit `57f1f98`.
  - `git diff 57f1f98 HEAD` on it is empty.
- **The lock completed exactly once.**
  - `lock_run.json` records `lock_invocations: 1`, launcher `f66c12f2…`, exit 0 and no error. Both dry runs record 0
    lock invocations.
  - The state's `run_id` `914fa3afff5c73ec` equals the lock's.
  - The state was created at 16:51:30 and written at 16:51:50, inside the production window.
  - The pre-lock check at 16:47 found no `outputs/experiment-025/`.
- **Confirm and report have not started.**
  - `phases`: lock complete (commit `d3ecbd6`, runtime cpu/float32/4 threads); confirm and report `not_started`.
  - No `incidents` key in any phase.
  - `confirmation: null`, `report: null`.
  - `executed_prompt_keys` and `executed_noun_keys` are empty.
- **No fresh artifact and no incident.**
  - `outputs/experiment-025/` holds exactly the three candidate files, hidden files included.
  - The lock and preregistration were never committed (`git log --all` on the paths is empty) and are not installed.
- **A second lock is refused.** `cr.assert_phase_allowed("lock", state)` raises "lock already written; a new candidate
  lock requires a new protocol version".
- **The state reconstructs.**
  - I reset `confirmation_025`, `lock` and `phases.lock` to their initial values.
  - The result equals `cr.new_results_state(base digests, d3ecbd6, current versions, PRODUCTION)` in every field except
    `run_id` and `created_at`: 0 differing fields.
  - The state's `state_sha256` recomputes, and the file is canonical JSON.

### 2. Candidate integrity: PASS

Evidence is in `r1_integrity.out.json` (item 2) and `r2b_crosscheck.out.json`.

- **The lock file.**
  - File sha256 `1881a790…`.
  - Content digest `8e5300de…` from my own canonical JSON and from `rc.content_digest`.
  - The bytes equal `canonical_json(lock) + "\n"`; my serializer equals `pm.canonical_json`.
  - All 1,226 numbers are finite: a strict parse rejects NaN and Infinity, and a walk finds none.
- **The preregistration.**
  - Sha256 `e4b12656…`.
  - Byte-equal to `cr.render_preregistration(json.loads(lock))`.
  - Equal to the state's `preregistration_sha256`.
- **Every rendered number.**
  - **The header:**
    - the lock digest, run id, commit, design and plan revisions, and the module blob `622aa832…`;
    - the freeze `6eca7024…` and manifest `0e1f068b…`;
    - 90720 = 40 × 108 × 21;
    - |d| `1.3576567483118904` and cos `0.23521413845112732`, equal to the lock's `repr`;
    - ±0.32 and ±0.16; 7 controls and the tag; the 21 conditions;
    - the tail `5288280983/274877906944 = 0.01923865414210013`, which I recomputed exactly; the threshold 27, which I
      derived myself;
    - the 18 geometry tolerances; the patch-path keys;
    - I1/I3/I4 at 1e-4, 1e-4 relative and 1e-3, and Level-1 at 0.02.
  - **The 40-row cue table**, re-rendered from **my own** geometry values in the same format, is identical line for
    line.
  - **The check-maxima line** equals the lock's values.

### 3. Lock-run isolation: PASS

- **The guards came first.** In `launch_lock.py` (sha256 `f66c12f2…`, equal to the recorded one), the order is:
  1. the audit hook (line 127), installed before torch or the project is imported;
  2. `torch.nn.Module.__call__` refusing for the whole run (line 186);
  3. the alias sweep by identity (line 205);
  4. the confirm-only refusals (lines 206-214);
  5. the counted, sealing `load_model` (line 235);
  6. the runner import (lines 237-240), with `assert runner_module.load_model is _counted_load` and a second sweep;
  7. `runner.lock()` (line 267).
- **The records.** `lock_run.json` records:
  - `load_model: 1` and `modules_sealed: 430`;
  - `module_calls_refused: 0`;
  - empty `forward_refused`, `model_method_refused`, `capture_or_intervention_refused`, `confirm_only_refused`,
    `forbidden_reads`, `forbidden_writes` and `forbidden_deletes_renames_links_mkdirs`;
  - `output_writes` of exactly the two candidates plus `.results-v87zujap.json`, and one rename of it to
    `results.json`;
  - two idempotent `mkdir experiment-025` events;
  - outside-repository writes only to `/dev/null`, a temp file and filelock's import probe. My own processes show the
    same kind of entries at import.
- **Code reading of `Runner.lock()` (`run.py:363-408`).** It runs:
  - the pinned-blob check;
  - the committed inputs;
  - the runtime check;
  - one model load;
  - `ModelPrograms.from_model` (weights only);
  - `geometry_block` twice, with a bitwise self-comparison, inside `pytest_free_guard`;
  - `nearest_tokens`;
  - `scientific_dependencies`;
  - `_recheck`;
  - the writes.

  It contains no prompt, capture, intervention or confirm-only computation.
- **The guards would have caught a violation (`r6_launcher_guards.out.json`).** The launcher's own functions were
  extracted verbatim and exercised offline.
  - **Refused:** reads of 022's table and of `outputs/experiment-023|024`; a write of `src/` or
    `experiments/025/...`; a removal, a rename out of `outputs/`, a mkdir and a symlink in the repository; a module
    call; direct instance `.forward()`; and an alias of `capture_prompt` in a probe module, which was swept by identity.
  - **Allowed:** writes and renames inside `outputs/experiment-025/`, and writes outside the repository.
  - See N3 and N4 for the limits.
- **The first dry run's false positive cannot have affected the production lock.**
  - It was a separate process (16:49:23–16:49:37) with a different launcher version (`cdc466f3…`) in dry-run mode.
    It made no `runner.lock()` call (`lock_invocations: 0`), wrote nothing (`outputs: {}`, no output events), and
    created no state.
  - Its two "forbidden" events were `os.remove` of filelock's import-time probe files. They were addressed by
    `dir_fd` inside a temp directory but resolved against the working directory, so the old hook refused them.
  - `probe-source` and `probe-link` do not exist in the repository.
  - The fixed launcher resolves `dir_fd` with `F_GETPATH`. `r6` shows that exact case is now allowed.
  - The second dry run (the production launcher `f66c12f2`) was clean. The production lock ran afterwards with the
    tree clean (`tree_porcelain: ""`) and the same launcher sha256.

### 4. Dependency bindings: PASS

All hashes below were computed by my own code (`r1`, `r2a`, `r4`).

- **The freeze.** File `54c8947c…`, content `6eca7024…`, manifest `0e1f068b…` (90,720 keys). These equal
  `lock.confirmation_025` and the runner's `confirmation_sha`.
- **The 13 pinned module blobs.** My own `sha1("blob n\0"+bytes)`, `git rev-parse HEAD:<path>`, `cr.FROZEN_BLOBS`,
  `lock.module_blobs`, `lock.dependencies.module_blobs` and the state's `module_blobs` are all equal. Every imported
  module is the repository file.
- **`cr`'s own blob.**
  - `622aa832…` equals `lock.module.blob`, `cr.own_blob()`, `git ls-tree HEAD`, and my hash of the imported file,
    which is the repository file.
  - It has not changed since the lock commit.
  - `run.py` is blob `ed027a96…` at HEAD. It is not bound directly; it is a scientific path.
- **024's three files.** Calibration `81fb499e…/09bf093e…`, lock `5c2a9b90…/a39668bf…`, freeze `68510e1b…/87f8aff1…`.
  Each file and content digest equals `lock.inherited_024` and `lock.inputs`.
- **020's confirmation file (the prior-noun source).** `de4ebb7b…`, with 24 nouns and 48 ids. Its id digest
  `4a1e2268…` equals the freeze's `blocked.prior_nouns`.
- **The input digests.** `lock.inputs` (40 digests) equals the runner's base digests, and
  `dependencies.readout_020.frozen_input_digests` (21) equals `inputs.digests`.
- **The model.**
  - Parameters `fd953f1c…`, hashed over the 113 named parameters in name order.
  - Embedding `9cd6f39b…`. The raw `model.safetensors` (fp16, sha256 `3da38833…`) cast to fp32 equals `W_E` bit for
    bit.
  - Resolved revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`. It equals `lock.dependencies.model.revision`, the
    only HF snapshot and `refs/main`. The tokenizer files are in the same snapshot.
- **020's exposed states.** `26c21d63…`: see item 10.
- **Drift refuses before any fresh execution** (`r5_logic.out.json` → `validate_lock`). The real candidate was passed
  as the installed lock, with the real state, digests, freeze, noun keys and preregistration.
  - **Accepted:** a clean, tracked tree whose changed paths are only the lock, the preregistration, the README and
    `evidence/`.
  - **Refused, with the expected message each time:**
    - a dirty tree;
    - an untracked lock;
    - a lock commit that is not an ancestor (`None`);
    - a changed `src/` file;
    - a changed `run.py`;
    - a changed 024 input path;
    - a changed input digest;
    - a running `cr` blob that differs;
    - a preregistration with one character changed;
    - a re-signed, edited lock.
  - **Ordering.** In the runner, `validate_lock` runs before the model loads (`run.py:422-424`). Changed modules are
    also caught earlier by `assert_frozen_blobs` in `_base()`, and changed inputs by the hash checks in `_base()`.
  - **Classification.** `cr.scientific_changes` treats `src/`, `experiments/005…025/` and the screening manifest as
    scientific. It exempts 025's freeze, lock, preregistration and README, the `evidence/` directories, and a few
    hash-bound 022 files.

### 5. `d`: PASS (maximum disagreement 0.0)

Evidence is in `r2a_own.json` and `r2b_crosscheck.out.json`.

- **Construction.** `μ_noun` and `μ_cue` are float64 means of the model's float32 rows over the calibration record's
  `noun_row_ids` (158) and `calibration_cue_ids` (139). Both id lists' digests match the record.
- **Digests.**
  - `μ_noun` `87930f01…` and `μ_cue` `f8cca954…` equal the record and the lock.
  - `d` is `5f914283…`, with |d| = `1.3576567483118904`.
- **Against `cr.direction`:** bitwise equal.
- **High precision.** A 40-digit mpmath reference bounds the float64 errors: `μ` ≤ 3.2e-18 and `d` ≤ 6.7e-17.

### 6. The 40 cue geometries: PASS (maximum disagreement 0.0)

- **Checked per cue against every stored lock value:** E, |E|, Ê, s₀, t, τ, t̂, θ = asin(0.32/τ), θ_half = asin(0.16/τ),
  and the even terms.
  - `norm`, `s0`, `tau`, `theta_primary`, `theta_half`, `even_primary` and `even_half` are exactly equal for 40 of 40.
  - The `t_hat_sha256` values are equal for 40 of 40.
- **Ranges:**
  - |E| 0.70147–0.73004;
  - s₀ −0.305889…+0.285207;
  - τ 1.322748–1.355974;
  - θ 13.6501–13.9999°;
  - θ_half 6.7765–6.9475°.
- **Arcsine arguments** are at most 0.242, so all are valid. Every value is finite.
- **The order** is the freeze's frozen order: 20 adjectives, then 20 nouns.
- **High-precision errors:** s₀ ≤ 1.2e-16, τ ≤ 4.9e-16, θ ≤ 9.5e-17, t̂ ≤ 8.7e-17.

### 7. The vectors: PASS (maximum disagreement 0.0)

- **Construction.** The 21 conditions are built as `|E|(cos θ Ê ± sin θ û)` in float64, then cast to float32.
- **Digests.** `vectors64_sha256` and `vectors32_sha256` match 40 of 40. Both are bitwise equal to `cr.cue_vectors`.
  The combined float32 `[40,21,512]` digest is `30a35c00…` (N7).
- **Checks:**

| check | float64 | float32 |
|---|---|---|
| norm preservation | ≤ 4.4e-16 | ≤ 6.0e-9 |
| odd component (±0.32 and ±0.16) | ≤ 2.8e-16 | ≤ 2.1e-9 |
| complete change `s₀(cos θ − 1) ± odd` | ≤ 2.8e-16 | ≤ 3.5e-9 |
| angle to E (my `atan2(|b⊥|, a·b)`) | ≤ 1.9e-16 | ≤ 3.7e-9 rad |

- **The base condition.** `R(0)` cast to float32 equals the model row bit for bit, 40 of 40.
- **High precision.** The float64 `noun+0.32` vectors are within 3.1e-17 of the mpmath reference.

### 8. The 280 random controls: PASS (critical; maximum disagreement 0.0)

- **My implementation of the frozen stream:**
  - the tag `025|control|{token_id}|{j}`;
  - SHA-256 of `"{tag}|{counter}"`, counter from 0;
  - four big-endian 64-bit words, each `>> 11`, then `(v+1)/2**53`;
  - Box–Muller pairs (radius from the first uniform, phase from the second, cos then sin);
  - projection off Ê then t̂, twice;
  - normalization.
- **Agreement with `cr`.** All 280 uniform streams and 280 Gaussian vectors are bitwise equal to `cr.sha_uniforms` and
  `cr.sha_gaussians`. All 280 controls are bitwise equal to `cr.control_directions`.
- **Digests.** The per-cue `controls_sha256` values match 40 of 40. The combined `[40,7,512]` digest is `e12581d4…`.
- **In float64:**
  - `||u|−1|` ≤ 5.6e-16;
  - `|Ê·u|` ≤ 6.1e-17, `|t̂·u|` ≤ 4.9e-17, `|d·u|` ≤ 5.6e-17.
- **On the actual float32 patched vectors:**
  - **Neutrality:** `|s(+θ) − s(−θ)|` ≤ **6.088945903037768e-09** (the plurality pair is included; the maximum is the
    same for the controls alone).
  - **Angle:** at most 3.7e-9 rad from θ_i. Each control's angle is within 4.2e-9 rad of the nounness rotation's angle
    for the same cue.
  - **Norm:** ≤ 6.0e-9.
  - **Even term only:** each control's score equals `s₀ cos θ` within 4.3e-9.
- **The patch actually applied.** For the float32 pair's effective displacement
  `u_eff = (v₊ − v₋)/(2|E| sin θ)`:
  - `|d·u_eff|` ≤ 1.6e-8, `|Ê·u_eff|` ≤ 1.8e-8, `|t̂·u_eff|` ≤ 9.4e-9;
  - `cos(u_eff, u)` ≥ 0.99999999.

  So the runs move along the intended neutral directions.
- **Sanity of the draws:**
  - the largest |cos| between a cue's 7 controls is 0.150;
  - |cos(u, p̂′)| ≤ 0.0998;
  - the Gaussian means lie in [−0.071, 0.108] and the variances in [0.918, 1.102].
- **Conclusion.** No systematic failure: B's premise, equal-angle and nounness-neutral comparators, holds on the vectors
  confirm will patch.

### 9. Control determinism: PASS

- **Code.** `cue_rotation.py` has no `random`, NumPy or torch random-number-generator call and no seed.
  - The generation path is `hashlib` → `math` → torch arithmetic (lines 446-485). "Random" appears only in strings and
    variable names.
- **A second process, twice** (`r3_determinism_31337.out.json` and `r3_determinism_271828.out.json`, with
  `PYTHONHASHSEED=12345` and `random`).
  - Python's, NumPy's and torch's generators are seeded differently and advanced before each cue and each block.
  - My stream and `cr.control_directions` both give `e12581d4…`, and every per-cue `controls_sha256` equals the lock.
  - `cr.geometry_block` is bitwise equal to the lock's geometry, and the float32 digest is `30a35c00…`, under torch 1, 4
    and 8 threads.
  - With the cue order shuffled, every per-cue record still equals the lock's. Each vector depends only on
    `(token_id, j)`; the cue order sets only the stacking.

### 10. The `D_attn` binding: PASS

Evidence is in `r4_dattn_binding.out.json`.

- **020's locked rows.**
  - All 108 exposed frames are present.
  - `p_c` and `p_t` match the frames, and each reference prompt ends at `p_t`.
  - `rows4` and `rows5` are `[8, p_t+1]` (key lengths 4–9): non-negative, with each row summing to 1 within 1.7e-7.
  - Because the prompt ends at `p_t`, `[:, :p_t+1]` is the whole causal row; no key lies beyond `p_t`.
- **The exposed-states digest.**
  - `26c21d63…`, recomputed by my own canonical JSON, equals the lock and `ul.exposed_states_digest`.
  - The per-frame digests equal the committed 020 extract's `locked_state_digests`.
  - `rows4` and `rows5` are hashed fields: changing one `rows5` entry changes the frame digest.
  - 020's results file hash `da63b8c2…` is pinned.
- **The states confirm uses.** `ul.y1_states` / `rd.state_from_locked` hold exactly these rows as float64, each bound
  to its frame.
- **The capture axes.**
  - Confirm's capture (`cr.measure_rotated` → `PromptRun.vector(("ATTN_PATTERN.L4", p_t))`) selects query `p_t` on the
    `hook_pattern` axes `(batch, head, query, key)`, giving `[8 heads, keys]` in model head order.
  - This is the same `_requests_for`/`select_tensor` path through which 020's `capture_frame_020` obtained its rows.
  - `stage_two` stores `[:, :p_t+1]` into zero-padded tensors, and `routing_distance` slices `[:, :reference.shape[1]]`.
- **The spent captures.** 6 plain captures of 020 reference prompts, each in 020's ledger:
  `cardinal-009-1|ref|581`, `cardinal-009-2|ref|581`, `coordinated-adjective-009-1|ref|581`,
  `coordinated-adjective-009-2|ref|581`, `quantifier-009-1|ref|1016` and `quantifier-009-2|ref|1016`.
  - Taken through the same `vector → float32 → double()[:, :keys]` path, both L4 and L5 rows are **bitwise equal** to
    the locked rows in all 6, with shapes `[8, p_t+1]`.
  - This establishes the same frame and reference association, head order and key axis. No distance was computed.
- **The definition (`cue_rotation.py:1024-1034, 1257-1287`).**
  - Per head, the total-variation distance `½·fsum|a−b|`, over the 8 L4 heads then the 8 L5 heads.
  - `fsum/16`, then `fsum/108` over the cue's frames.
  - No head index, weight or selection parameter exists.
- **A synthetic check** (made-up rows, `r5`): the code equals my own mean of 16 per-head distances, ignores padding, and
  refuses a mismatched key axis.

### 11. A/B/G semantics: PASS

- **The formulas** (`cue_rotation.py:1305-1330`):
  - `A = ½[ℓ(noun+0.32) − ℓ(noun−0.32)]`;
  - `A_ij` likewise for `rand{j}`;
  - `B = A − fsum(|A_ij|)/7`;
  - `G = ½[D_attn(+) − D_attn(−)]`.
- **The pass rule.** PASS iff `count_positive` ≥ 27, where `count_positive` counts only finite values `> 0`. Ties,
  zeros, −0.0, NaN and None count against.
- **The tail.** Exact `5288280983/274877906944 = 0.01923865414210013`, recomputed by me. 26 would give 0.04035.
- **`per_cue`.** ℓ = `log(fsum SSE_C / fsum n)` over the cue's 108 frames, via `rr.fresh_pair_cells` and `rr.cue_mse`.
  `D_attn` is the frame mean. An MSE that is not finite and positive raises an incident.
- **Synthetic tests** (`r5`). Four synthetic scenarios, including exactly 27 against 26 and exact ties, give A, B and G
  equal to my formulas and the expected counts and labels.
- **No secondary analysis can alter the result.** The descriptives and the ladder run only after the result write
  (`run.py:519`), and they read `results` without mutating it.

### 12. The outcome hierarchy: PASS

| gate/incident | A | B | G | label |
|---|---|---|---|---|
| any incident or failed gate (8 combinations) | any | any | any | `NOT_INTERPRETABLE` |
| none | fail | any | any | `CAUSAL_EFFECT_NOT_ESTABLISHED` (4 combinations) |
| none | pass | fail | any | `DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC` (2) |
| none | pass | pass | fail | `READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED` (1) |
| none | pass | pass | pass | `NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR` (1) |

- **`outcome_label`** was verified on all 8 A×B×G combinations. An earlier failure is never overridden.
- **Every incident path gives `NOT_INTERPRETABLE`.** Each gate or identity failure raises `IncidentError`, which is
  recorded without a result (`run.py:496-503`), and `render_report` shows `NOT_INTERPRETABLE`. This was tested on
  synthetic states: a confirmation incident and a pre-ledger incident. `outcome_label` never returns
  `NOT_INTERPRETABLE`; only the incident path produces it.
- **Only the five labels exist.**

### 13. The interpretation of A: PASS

The same wording appears in all three places:
- the lock's `semantics.A`: "+nounness … greater … than the matched −nounness …; A alone does not establish that +θ
  rises above the unperturbed baseline or that −θ falls below it (the baseline and the even component are descriptive
  only)";
- the preregistration, which renders it;
- `render_report`, which prints `SEMANTICS["A"]`. `validate_lock` forces the lock's semantics to equal `SEMANTICS`.

The outcome readings say "(+ against −)". The baseline comparisons (`plus_above_base`, `minus_below_base`, the even
terms) exist only in the descriptives, labelled "no outcome force".

### 14. Level-1 coverage: PASS

- **The gated domain.** `level1_gates` routes `condition in set(config.outcome_bearing)`: exactly the 16 conditions
  `noun±0.32` and `rand1..7±0.32` go to `level1_outcome_bearing`. `base`, `noun±0.16` and `plur±0.32` go to
  `level1_secondary`.
- **Enforcement.** `enforce_gates` enforces only the former, at 2e-2, with `≤`. A `None`/NaN value fails; NaN is never
  displaced (`ul._worse`).
- **Synthetic routing test** (both groups, all 21 conditions, per-row tensors): the maxima land in the right bucket at
  the right condition (`rand7-0.32` and `plur-0.32`).
- **Boundaries:** 0.02 passes and 0.0200001 fails. A secondary value of 99 is not enforced.
- **Consequence.** A violation raises `IncidentError`, which becomes an incident with no result, and hence
  `NOT_INTERPRETABLE`.

### 15. The remaining identities: PASS

- **Vector I1.** `vector_factors` builds `d_emb = v − W_E[ref]` and `delta_e = MLP₀(LN₂(v)) − lexicon(chain reference)`:
  the same reference as `ul.pair_factors`/`encoding_delta`, with the same block-0 terms. I1 (1e-4), I3 (relative 1e-4)
  and I4 (1e-3) use `rr.target_gates`' formulas.
- **Precedence.** They are enforced in the order I1, I3, I4, Level-1, each with `≤`. This was tested at the tolerance
  boundaries.
- **`C` recomputation.** `C` is recomputed from `saved = torch.load(stage2_path)`, after the in-memory copy is cleared
  and the digests are verified (`run.py:470-486`), bit for bit.
- **Patch fidelity.** `pm.run_patched` → `_check_execution` requires, on every run:
  - REPLACE, the declared DIRECT source, and the right hook, shape, dtype and position;
  - `outside_max_abs_change == 0`;
  - `torch.equal(after, replacement)`.
- **The θ = 0 identity** is the patch-path check (item 16), and the base vector equals the model row bit for bit.

### 16. Patch-path isolation: PASS

- **The keys.** The 4 keys:
  - are in 020's ledger and in the spent set;
  - their frames are exposed frames;
  - they are outside the manifest, with 0 collisions in tagged or untagged form;
  - `lock.patch_path_spent_keys` equals `cr.PATCH_PATH_SPENT_KEYS`.
- **Order.** The check runs after I7′ and before the ledger write (`run.py:441-461`).
- **Isolation.** It records only equality flags and digests in `phases.confirm.patch_path_check`. It never touches
  `executed` or `executed_prompt_keys`, which is created afterwards, or the accounting.
- **Failure and refusal.**
  - Any failure, or an inability to run, becomes a phase incident before the ledger.
  - A key that is not spent is refused before any forward. This was tested synthetically.
  - A condition-tagged key whose prompt is spent is refused by `cr.assert_ledger_isolated`.

### 17. The manifest: PASS

- **Rebuilt from the freeze's cues × exposed frames × 21 conditions:**
  - 90,720 keys, unique, in sorted order;
  - equal to the freeze's list in order;
  - digest `0e1f068b…`;
  - 16 outcome-bearing conditions.
- **The spent set, recomputed from its sources:**
  - 020's ledger, read from `outputs/experiment-020/results.json`: 30,132;
  - 020's confirmation manifest: 3,060;
  - 022's manifest: 3,060;
  - 023's manifest: 3,060;
  - 024's manifest: 4,320;
  - the union: 43,632, equal to the runner's `forbidden`.
- **Collisions.** 0 for untagged keys, tagged keys and untagged forms, and 0 with the patch-path keys.
- **The ledger** is empty.

### 18. Pre-prompt ordering in confirm: PASS

`run.py:410-461`:
1. `_base()`: the pinned blobs first, then the committed inputs by hash. Then the freeze, the phase rule and ledger
   isolation.
2. `validate_lock`: the lock's content and candidate identity; the inputs, module blobs (`cr` and `rr` directly),
   configuration, freeze binding, dependencies and wording; the preregistration's rendering and sha; tracked; clean;
   ancestor; no scientific change.
3. The runtime and versions against 020's; the seed; the model load, with its revision check; the parameters digest; the
   programs; the nouns; the embedding digest.
4. I7′: `geometry_block` recomputed, with every geometry gate enforced inside it, then compared with the lock bit for
   bit.
5. The patch-path check.
6. Only then the ledger write (`run.py:456-461`), and then stage 2.

A `PhaseError` in steps 1–3 propagates with no state change and no execution. An `IncidentError` in step 4 or 5 is
recorded as a phase incident, which refuses any later confirm, and returns before the ledger.

### 19. Measurement and write ordering in confirm: PASS

`run.py:462-519`:
1. Every tagged run (`stage_two`).
2. The durable save (`rr.save_durably`: fsync plus a directory fsync). The digests and the stage-2 record are written.
3. The accounting, written, then enforced.
4. The re-read (`torch.load`), with the digests checked and `assert_measurements`.
5. `C` recomputed, written, then enforced.
6. I1, I3 and I4, then
7. Level-1 over the 16 conditions: both computed in `run_gates`, written, then enforced in the order I1, I3, I4,
   Level-1.
8. `D_attn`, then
9. ℓ, MSE and nMSE: computed together in `per_cue`.
10. A, B and G, then
11. the outcome (`statistics`).
12. The 020–024 re-check (`_recheck`).
13. One atomic write of the result and the completed phase (`write_state_atomic`: temp file, fsync, `os.replace`).
14. Only then the descriptives.

- **An incident never coexists with a result.**
  - Every incident path runs before `results` is set.
  - A failed result write pops `results`, `lock_sha256` and `completed_at` and restores the running phase before
    recording the incident.
  - Nothing after the result write records an incident.
- **A partial-measurement failure** records the incident (`_record_incident`) before `_preserve_partial` saves the runs
  measured so far. The partial save is skipped once the full save exists.
- **A descriptive failure** is recorded under `descriptives.failures`, and the phase stays complete. The descriptives
  never mutate `results`. An interruption is re-raised after recording, and the result stays on disk.
- **A second confirm is refused.** Once the ledger is written the status is `running`, and a pre-ledger incident also
  refuses.
- **Deviations from the brief's order:** none with consequence.
  - Steps 6–7 and 8–9 are computed within one call each, and enforcement follows the listed precedence.
  - The execution order (frame-major) is covered in N8.

### 20. No outcome leakage: PASS

- **`outputs/experiment-025/`** holds only the three candidate files.
- **No 025 measurement file exists anywhere.** No `stage2-measurements.pt`, `stage2-partial.pt` or 025 report exists in
  `outputs/` (023 and 024 excluded) or in the scratchpad. The only `stage2*` files are 021's and 022's.
- **Condition tags occur only in expected places:**
  - in `outputs/`, only in the three candidate files;
  - in the scratchpad, outside this directory, only in code and docs (`lock025/verify_lock.py`,
    `design025/plan-draft.md`) and in a fake-world tier-B assertion log (`tierB-full.log`).
- **`probe025/test_probe.py`** is a fake-world fixture probe.
- **The state** has `confirmation: null`, and the launcher records show zero forwards.
- **The operator's `verify_lock.py`** ran under module-call, forward, capture and confirm-only refusals with one
  weights-only load (`refused: []`). It computed no scientific quantity.
- **This review** ran no fresh prompt and no rotated forward.

Hence no fresh Δx3, Δc, `D_attn`, ℓ, MSE, A, B or G exists.

## The operator's claims

Every check in `prelock.out` and `verify_lock.out` (33 checks, `all: true`) that concerns a quantity was reproduced
independently above, with the same values. The claimed digests match my own computations: `30a35c00…`, `e12581d4…`,
`26c21d63…`, `fd953f1c…`, `9cd6f39b…`, `5f914283…` and `2dd4a1f4…`.

## Install statement

The exact existing `outputs/experiment-025/candidate-lock.json` and `outputs/experiment-025/candidate-preregistration.md`
**can be installed byte for byte, with no regeneration**. Install them as
`experiments/025-nounness-direction-intervention/preregistration-lock.json` and
`experiments/025-nounness-direction-intervention/preregistration.md`.

- **What makes this safe.**
  - The lock is canonical JSON ending in a newline. The preregistration is byte-equal to its rendering.
  - The repository has no `.gitattributes` and no `autocrlf` or `eol` setting, and neither path is ignored.
  - `results.json` binds exactly this content digest (`8e5300de…`) and preregistration sha (`e4b12656…`).
  - `cr.validate_lock` accepts exactly these bytes as the installed lock, given a clean, tracked tree whose only changes
    since `d3ecbd6` are the two files, the README and `evidence/`.
- **Do:**
  - copy the files (do not re-render them);
  - verify the two sha256 values after the copy;
  - leave `outputs/experiment-025/results.json` untouched;
  - commit only non-scientific paths;
  - run confirm with its working directory at the repository root, on the same machine and software stack (N1, N2).

## Files (in this directory)

- `guard.py`: the audit hook and the refusals; the single capture door for item 10.
- `r1_integrity.py` and `.out.json`: items 1, 2, 4 (file level) and 17.
- `r2a_own_geometry.py`, `r2a_stream.py`, `r2a_own.json` and `.pt`: my own reconstruction, with `cr` geometry refused.
- `r2b_crosscheck.py` and `.out.json`: the cross-check against `cr` and the lock; I7′ recomputed.
- `r3_determinism.py` and `r3_determinism_{31337,271828}.out.json`: item 9.
- `r4_dattn_binding.py` and `.out.json`: item 10, including the 6 guarded spent captures. It ran once, and must never
  be re-run, to keep the capture budget. It ran with `guard.py` as it was before the `confirm_only` option was added;
  `r4` uses that option's default, which is the earlier behaviour.
- `r5_logic.py` and `.out.json`: synthetic logic tests (items 4, 11, 12, 14–16, 19).
- `r6_launcher_guards.py` and `.out.json`: item 3, the launcher's guards exercised offline.
- `first_pass/`: the first-pass outputs. The final pass re-ran every script except `r4` with the final `guard.py`. The
  results are identical, apart from temp-file names and `r5`'s unseeded synthetic `D_attn` example, whose conclusions
  hold on both draws.
