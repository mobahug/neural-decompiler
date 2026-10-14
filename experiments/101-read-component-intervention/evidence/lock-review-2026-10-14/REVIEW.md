# Experiment 101: review of the `lock` phase

**AI review (independent, read-only), not human validation.** The reviewer is an AI agent (Claude). It worked read-only
on 2026-10-14, from about 09:33 to 10:15 UTC, against `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler` at HEAD
`3621eec4b6a901119a4b96d4fdc07fa3b19652ab` (branch `exp101-phases`, clean tree). It wrote only inside
`.../scratchpad/exp101/lock-review/`. It created, changed and deleted nothing in the repository, including `outputs/`
and the two candidates. Every git call of mine used `--no-optional-locks`. The end snapshot
(`out/repo_snapshot_end.txt`, 10:11 UTC) shows:
- the same HEAD and an empty `git status --porcelain --untracked-files=all`;
- the same sha256 for the five files in `outputs/experiment-101/`;
- `.git/index` unchanged since the `3621eec` commit (12:29:58 local), with no ref change and no new object;
- no repository file outside `.git` newer than the lock's end.

The `.git` directory entry's own mtime moved once during the review, at 10:09:10 UTC. No command of mine called git at
that time. No file inside `.git` changed. That matches a transient `index.lock` from another process's plain
`git status` (N10).

- **No forward pass at all.** Nothing needed one. No prompt ran: no 101 frame or candidate text with any cue, and no
  exposed prompt either.
- **Weights only.** Weights were read in two ways:
  - from the pinned `model.safetensors` directly (scripts `b`, `c`, `d`);
  - from one model object per script, loaded through `models.load_model` (`c` and `d`), for the frozen functions and
    the digests.
- **The guard** (`scripts/common.py`) was installed before any `neural_decompiler` import in every script:
  - an audit hook refused, and would have recorded, any repository write, rename, delete, link, chmod or mkdir, and
    any read under `outputs/` other than 020's `results.json` and `outputs/experiment-101/`;
  - `torch.nn.Module.__call__` refused every call, after checking integer inputs against the 240 forbidden token
    sequences (the 60 candidate texts of design §11, parsed from the committed markdown, × `one`/`two`/`three`/`many`,
    tokenized with the pinned `tokenizer.json`, with prefix relations);
  - 17 capture, intervention and measurement entry points, and every alias of one by identity (21 aliases wrapped),
    refused after checking the prompt's token ids;
  - the loaded model was sealed at once (430 instance `forward`s, plus the root's `run_with_hooks`, `run_with_cache`
    and `generate`).
- **The guard never fired:** 0 module calls, 0 entry-point calls, 0 hits on a 101 sequence, 0 refused writes and 0
  refused reads.
- **What ran of the runner.** The only phase run was `validate`, through `Runner(tracked=…, git_state=…,
  changed_paths=…)` with my `git --no-optional-locks` functions, and it exited 0. The read-only methods `_base`,
  `_confirmation`, `_calibration`, `_state_for("lock")` (which refused), `_check_nouns` and `_model_dependencies` were
  called. `freeze`, `calibrate`, `lock`, `stage1`, `confirm` and `report` were not run.
- **The stage-1 launcher** was read, not run or imported.

## Verdict: **PASS WITH NOTES**

**The candidate lock is correct and complete, and it is reproducible.**
- My own code, written and run first with no `neural_decompiler` import, rederives every field from the committed
  files, the raw checkpoint and the documents.
- In a fresh process of its own, the frozen code:
  - rebuilds `candidate-lock.json` byte for byte with `ri.build_lock`;
  - renders `candidate-preregistration.md` byte for byte;
  - reproduces the geometry block bit for bit, twice;
  - reproduces the exposed prediction table's digest `ce055e09…`.
- **`ri.validate_lock` accepts the candidate as it would be installed,** with simulated git facts. It refuses all nine
  negative controls: a scientific-path change, a `run.py` change, a dirty tree, untracked files, a lock commit that is
  not an ancestor, an edited preregistration, a re-digested widened tolerance, a widened tolerance without a matching
  state, and a changed noun order.
- **The lock phase ran exactly once and changed only what it should.** Undoing just the state's `lock` block and
  `phases.lock` reproduces the calibrated state bit for bit.

**There is no blocker to committing the lock.**
- The three should-fix findings concern version 1 of the prepared stage-1 launcher. It has never run and it is not part
  of the lock. A version 2, written during this review, appears to address them; my limited re-check is in the
  addendum. A full re-check is still due before `stage1`.
- The notes concern:
  - the residual Level-1 risk at confirm;
  - a descriptive quantity that design §2 asks for but confirm does not record;
  - preregistration wording;
  - provenance and evidence preservation.

## Key digests

| object | digest | status |
|---|---|---|
| `candidate-lock.json` | file `8dc6e6d38c876a2dd9bf905fbeaf36b774b487967597795beb032bb4a0751087`, content `8da97e35556d0baa4739899cb841e86cbd59cced21611c732fdd9d8c9edd9061` | recomputed; canonical JSON + newline; rebuilt byte for byte by `ri.build_lock` |
| `candidate-preregistration.md` | `3761b8130203d5d15ac0e30055aa135626a6d97f0ab807df6cf4d31b6af8e980` | = `ri.render_preregistration(lock)` byte for byte; = the state's `preregistration_sha256` |
| `results.json` (git-ignored) | file `cc98ae1e…14ca`; `state_sha256` `a4657f80eed3e23587ea44597b77ce064c432eea5623799c838acdb668f40e12` | recomputed; the pre-lock state rebuilt as `729c9cb5…` (file `935a0df2…`), equal to the committed `state_extract.json` and to the operator's `state_before_dryrun.txt` |
| freeze `confirmation-v1.json` | file `6da26cd7…0922`, content `ef356506…624a`, manifest `7c7c9494…b8bd` | committed alone in `a857f9a`; unchanged; manifest rebuilt by me |
| calibration `calibration-v1.json` | file `6b1db338…ad90`, content `1eb1fe68…2e1e` | committed alone in `d502777`; byte-equal to the candidate; unchanged |
| τ_rem, τ_ins | `0.025153865322968608`, `0.03511072680382421` | the 33rd of the file's own 36 sorted `P_error_*` values; both ≤ 0.20, so P is evaluable |
| `w_c`, `ŵ`, `d̂_T` | `253a4733…8bd8` (‖w_c‖ 0.7659205190757624), `f4e9701b…f905`, `ec1c0dfb…e81d` | reproduced by my code |
| the 288 `v_j` | the lock's `geometry.iso_sha256` | 288/288 reproduced by my code |
| module `read_intervention.py` | blob `d7db6197140fe871da591b5c578cc675f05f7d69` | = `git ls-tree HEAD` = the calibration's `module_blob` |
| the 20 pinned blobs | the plan's table | = `git ls-tree HEAD` = `git ls-tree 516df20` = the working files |
| model | parameters `fd953f1c…0e5`, embedding `9cd6f39b…aaf`, `model.safetensors` `3da38833…33fd` | recomputed (my hashing code) |
| 020 exposed states | `26c21d63…0f83` (108 locked states) | recomputed from `outputs/experiment-020/results.json` (`da63b8c2…`) |
| exposed prediction table | `ce055e09c73613b74caae31222192f95528e6f58e101469917138786b2d9758f` | reproduced in my process = the calibration's `table_content_sha256` |
| `launch_lock.py` | `90e8159f97fc38234835c58e6443474677dfe0236ca48d09e6e67fe124e5e485` | = `launcher_sha256_at_production.txt` = the sentinel's content = both run records |
| `launch_stage1.py` (as reviewed) | `bf40e04368b715b52cac9e83e0ddeaf256eaf50a1b507c33846e95b28311c5f1` | read only |

## Method and order

Times are UTC. Each script's output is in `out/`, and each was run with
`/Users/…/neural-decompiler/.venv/bin/python -B` and `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`, from `scripts/`.

1. **Reading (09:33–09:45).** The brief; the design (all of it, §5–§9 and §14 closely); the plan (all of it); `ri` and
   `run.py` in full; the pinned helpers they call; the lock launcher and its records; the stage-1 launcher; the freeze
   and calibration reviews.
2. **My own code first, with no `neural_decompiler` import.**
   - `a_repo.py` (09:49): git, the installed artifacts, the outputs, the state, the evidence digests.
   - `b_independent.py` (09:51): every lock field, from the committed files, the raw checkpoint and the documents. It
     uses `mine.py`, my reconstructions of `w_c`, `v_j`, the manifests, the layout, the noun order and the constants.
3. **The frozen cross-check, only then.** `c_frozen.py` (09:54) ran under the guard. `d_params.py` (09:55) checked the
   bridge's parameters against the raw checkpoint.
4. **The deltas and the launcher evidence.**
   - `e_state_delta.py` (09:56): the pre-lock state.
   - `f_audit_events.py` (09:59): the real Python 3.12.13 audit-event signatures of stage 1's file operations,
     performed by me on scratch paths. The launcher was not run.
   - `g_iso_checks.py` (10:04): G1's checks on the exact `v_j`.
   - A probe of the relative-path audit events, on scratch paths (10:10).
   - The end snapshot (10:11).

## Items

### 1. Repository and phase integrity: PASS

- **HEAD and tree.** HEAD is `3621eec…` on `exp101-phases`, and `git status --porcelain --untracked-files=all` is
  empty. The branch is 10 commits ahead of `origin/main` (`96f0cc8`); see N4.
- **The freeze and the calibration.**
  - Each was installed alone: `git diff-tree` of `a857f9a` shows exactly `A …/confirmation-v1.json`, and of `d502777`
    exactly `A …/calibration-v1.json`.
  - Each working file equals its HEAD blob and is canonical JSON plus a newline, and each recorded `content_sha256`
    recomputes.
  - The calibration equals `outputs/…/candidate-calibration-v1.json` byte for byte.
- **What changed since the calibrate commit `8f65062`:**
  - `README.md`, `calibration-v1.json` and files under `evidence/`, all non-scientific paths;
  - nothing in `src`, `tests` or `run.py` since `40829d2`;
  - the design and the plan are unchanged since `516df20` and `1e48c1c`.
- **The phases.**
  - `calibrate` is complete at `8f65062`.
  - `lock` is complete at `3621eec`, written at 09:32:09.
  - `stage1`, `confirm` and `report` are `not_started`.
  - The `stage1`, `confirmation` and `report` blocks are null, and `executed_prompt_keys` is `[]`. The 2,016
    calibration keys live in `calibration_executed`.
  - The state holds no incident, stop or partial key anywhere.
- **The lock ran exactly once.**
  - The state holds one completed lock and no lock incident.
  - The launcher's preflight recorded `state_lock_not_started: true` at 09:31:30, so no lock had completed before.
  - The O_EXCL sentinel (mtime 12:31:30 local, content `90e8159f…`) permits one production invocation of this
    launcher.
  - `lock_run.json` records `lock_invocations` 1.
  - Undoing only `state.lock` and `state.phases.lock` gives state `729c9cb5…` and file `935a0df2…`. Those are the
    calibrated state of the committed `state_extract.json`, the calibration review's value and the operator's
    pre-dry-run snapshot. So the lock wrote nothing else into the state: no ledger key, no calibration change.
- **A second lock is refused.**
  - `ri.assert_phase_allowed("lock", state)` raises "lock already ran or recorded an incident; a new candidate lock
    requires a new protocol version". The runner's own `_state_for("lock", …)` raises the same.
  - Of the other phases, `stage1` is allowed, `calibrate`, `confirm` and `report` are refused.
- **`outputs/experiment-101/`** holds exactly five files: `calibration-measurements.pt` (`e8f42c1d…`) and
  `candidate-calibration-v1.json` (`6b1db338…`) from calibrate, the two candidates, and `results.json`. There is no
  partial, temporary, stage-2 or report file.

### 2. The launcher's evidence: PASS (see S2 and N5)

- **One invocation, exit 0.**
  - `lock.exit` reads `exit 0`.
  - `lock_run.json` records `exit_status` 0, `error` null, `lock_invocations` 1 and `load_model` 1.
  - The events show `modules_sealed` 430, `module_calls_refused` 0, and empty `forward_refused`,
    `model_method_refused`, `capture_or_intervention_refused`, `measurement_refused`, `forbidden_reads`,
    `forbidden_writes` and `forbidden_deletes_renames_links_mkdirs`.
  - Every route refuses, so zero refusals means zero calls.
- **The writes** were `candidate-lock.json`, `candidate-preregistration.md` and one `.results-*.json` temporary file
  renamed to `results.json`, plus two `exist_ok` mkdirs of `outputs/experiment-101`.
  - The reads under `outputs/` were 020's state, 101's state, the candidate re-read for rendering, and the
    directory's fsync open. After the run, the launcher hashed 101's outputs.
  - Outside `.git`, the only repository files newer than 12:29 local are these outputs and the evidence committed in
    `3621eec` at 12:29:58. The tree is clean, so nothing else changed.
- **The guards were installed before the runner was imported.**
  - In `launch_lock.py` the audit hook comes first (`:123`).
  - Then `torch.nn.Module.__call__` is refused for the whole run (`:182`).
  - The 5 capture and intervention entry points and their aliases follow (`:183–201`, 9 aliases recorded), with 13
    `ri` measurement functions, `ul.i3_error`, `rd.NounSet.contrasts` and `rd.ReadoutProgram.level1_detail`
    (`:202–207`).
  - Then the counting and sealing loader (`:209–225`).
  - Only after these is the runner executed (`:227–230`), then swept again (`:232`). The runner's `load_model` is
    asserted to be the counting loader (`:231`).
  - The `neural_decompiler` modules were imported before the guards (`:131–137`), which runs no model. The guards act
    on module attributes, which the runner and `ri` look up at call time.
- **The guards would have caught a forward or a capture.**
  - `model(...)` goes through the refused `Module.__call__`; the bridge defines no `__call__`.
  - A direct `forward` of any of the 430 modules hits the refusing instance attribute.
  - `run_with_hooks`, `run_with_cache` and `generate` are defined in `TransformerBridge`'s own class dict
    (`model_bridge/bridge.py:2809–3750`) and were refused at class level.
  - Every capture and intervention entry point refuses.
- **The lock's own path never touches a 101 token.** By reading `run.py:596–656`:
  - the prediction table is computed only for the exposed frames (`:628–631`);
  - the 101 frames enter only by id, in the `v_j` tags, and by `p_c`/`p_t`, in the stage-1 layout.
- **The dry run.**
  - `dryrun_run.json` runs from 09:30:29 to 09:30:55 with `lock_invocations` 0; its code path never calls
    `runner.lock()` (`:261–273`).
  - The state's sha256 was `935a0df2…` before and after it.
  - Its exposed-table digest, `ce055e09…`, equals the calibration's `table_content_sha256`. I reproduced it in my own
    process (item 7).
  - Each run overwrites `dryrun_run.json`, so the record cannot prove that there was only one dry run. A dry run
    cannot call `lock` or write to the repository, so the count has no bearing on validity.
- **The launcher's identity.** `launch_lock.py` hashes to `90e8159f…`, and so do `launcher_sha256_at_production.txt`,
  the sentinel's content and both run records. The file's mtime is 11:41:26 local, before both runs.

### 3. The lock's content, field by field: PASS

My code first (`out/b_independent.json`, 81 of 82 checks pass; the one open trace is N6), then the frozen code
(`out/c_frozen.json`, 43 of 44; the one failure is an artifact of my first test, which compared whole tensors only, and
is resolved by `d_params`).

- **The digest.** `content_sha256` recomputes, and the file is canonical JSON plus a newline. The top-level keys are
  exactly the 26 that `ri.build_lock` writes.
- **The identity fields.**
  - `run_id` `30ca8697c8702496` is the state's.
  - `protocol_code_commit` is `3621eec…`, both HEAD and the lock phase's commit.
  - `design` is `{a87e284, decision 516df20, revision 3}` and `plan` is `{96f0cc8, decision 1e48c1c, revision 3}`.
    Each commit touches its file.
- **The configuration** equals `ri.PRODUCTION.to_json()`. It also equals my independent rendering of the plan and the
  design:
  - the 60 texts parsed from design §11 and picks 1–36;
  - reference `ref`/581 and cues two/767, three/1264, many/1142;
  - `k_iso` 8 and 79 scored nouns;
  - Holm thresholds 26 and 25, recomputed from exact rationals with tails 0.005665492091793567 and
    0.014408359827939421;
  - D 25, P 27, the precondition 30 and τ's rank 33 (⌈0.9·36⌉);
  - overshoot and weak limits of 6;
  - 55 runs per frame, with the plan's tag order.
- **The module blobs.**
  - The 20 blobs equal `git ls-tree HEAD`, `git ls-tree 516df20`, the git blob of each working file, the plan's blob
    table and `ri.FROZEN_BLOBS`.
  - `dependencies.module_blobs` equals them.
  - The module's own blob, `d7db6197…`, equals `git ls-tree HEAD` and `ri.own_blob()`.
- **The inputs.**
  - All 42 digests equal the runner's `base.digests` (the state's `inputs` equal them too).
  - I traced every `*_file`/`*_content` input, and the 011, 012 and 017 locks, the 020 closure and extract, the
    manifest and the extension, to a committed file whose bytes or verified content digest match.
  - The exception is `table_022_file` (N6).
- **The 025 inheritance.** The file sha256 `54c8947c…` and content sha256 `6eca7024…` recompute.
- **The confirmation binding.**
  - The file, content and manifest digests recompute.
  - I rebuilt the manifest from the frozen frames and the plan's tags. It equals the file's: 36 S1-REF keys and 1,980
    distinct S2-TARGET keys, disjoint, with 144 untagged prompts.
  - I re-tokenized the 36 frozen frames from the design text with `one`. Each gives `prefix_ids + [581] +
    suffix_ids`, with `p_t = p_c + 1` and a one-token suffix.
- **The calibration binding.** Path, file, content, τ_rem, τ_ins and `evaluable: true` all equal the installed file
  (item 8).
- **The dependencies.**
  - The 020 readout blob is `caa73b40…`.
  - `exposed_states_sha256` `26c21d63…` recomputes from 020's 108 locked states.
  - The 21 `frozen_input_digests` are a subset of `inputs`.
  - `measurement` and `table` are the module's texts.
  - The model: the embedding digest `9cd6f39b…` recomputes from the raw checkpoint's `gpt_neox.embed_in.weight`. The
    parameters digest `fd953f1c…` recomputes with my hashing code over the bridge's 113 named parameters, and equals
    `rr.parameters_digest` and the lock.
  - Of those 113 parameters, 76 are whole raw tensors and 36 are the per-head q/k/v slices of the raw fused
    `query_key_value`. The last is the bridge's added `lm_head` bias, which is all zero; the checkpoint has no
    `embed_out.bias`. So the bound digest is the pinned checkpoint's values.
- **The noun keys.** My rebuild gives the same 79 in the same order: 020's pool order from the screening manifest (20
  per split, by first appearance), then 006's 20 nouns, with only `peach` non-scorable. It also equals
  `runner._noun_keys`, `progs.nouns` (80 nouns, 79 scorable) and 020's scorable set.
- **The run order and cue order** equal the plan's.
- **The statistics, constants and tolerances.**
  - The statistics texts equal `ri.STATISTICS`.
  - The constants equal `ri.statistics_constants()`:
    - Holm `1/40`, median floor 0.05, τ cap 0.20 and floor 0.02, head-share guard 0.05;
    - bootstrap 10,000 with seed 20261001 and ranks 250 and 9,750;
    - `WORST` −∞ for N, S, A_neg, `F_head_ins`, the head shares and the isotropic margins, and +∞ for A_amp and
      `F_head_rem`;
    - `DIRECTION_FREE` as the plan lists it.
  - **Rule 1 holds.** The tolerances equal `ri.TOLERANCES` and the plan's tolerance table entry for entry, and also
    the calibration file's `tolerances`. No tolerance was widened.
- **The outcome and the semantics.**
  - The nine labels are in design §7's order. The first condition reads "any gate fails, or an incident is recorded",
    the plan's extension of §7's "any gate in §8 fails". The other eight conditions are §7's verbatim.
  - The scope is §7's sentence verbatim.
  - The non-claims and the hierarchy follow §7 and §14, with the wording notes in N3.

### 4. The geometry, recomputed by me: PASS

- **`w_c` from 011's lock.** I computed `read_weight − mean·𝟙` two ways, and both give ‖w_c‖ =
  **0.7659205190757624**, the lock's value:
  - with torch float64 arithmetic: digest `253a4733…`, identical to the lock;
  - with a correctly rounded `math.fsum` mean: max-abs 6.9e-18 from the lock's vector.
- **`w_c` from the weights.**
  - I sliced head 4 of layer 3 from the raw checkpoint myself: the V rows `h·192+128…h·192+191` of
    `query_key_value`, the head's 64 input columns of `dense`, and `input_layernorm.weight`.
  - `γ₃ ⊙ (W_V (W_O d̂_T))` in float64 equals 011's `read_weight` at max-abs **0.0**, and centred equals `w_c` at
    **0.0**. The frozen `ra.read_weight` equals mine at 0.0.
  - G1's recorded value, 0.0, is reproduced: tolerance 1e-9.
- **The `v_j`, 36 frames × 8.**
  - My own SHA-256 counter-mode Box–Muller generator, with the tag `101|iso|{frame_id}|{j}`, two Gram–Schmidt passes
    off `(𝟙/√512, ŵ)` and normalization, reproduces all 288 lock digests bit for bit with torch float64 arithmetic.
    With numpy arithmetic the maximum difference is 8.3e-17.
  - On the exact vectors, with `math.fsum`:

    | check | worst |
    |---|---|
    | \|‖v‖ − 1\| | 5.6e-16 |
    | \|mean\| | 2.2e-18 |
    | \|⟨v, w_c⟩\| | 2.4e-17 |
    | \|⟨v, ŵ⟩\| | 3.2e-17 |

    All are ≤ 1e-12. The lock's own G1 values are 6.7e-16, 2.6e-18 and 5.9e-17.
  - The basis: `|‖𝟙/√d‖ − 1|` is 1.1e-16 and `|⟨𝟙/√d, ŵ⟩|` is 1.4e-17. The lock records 1.1e-16 and 1.0e-17.
  - The `v_j` within a frame are not mutually orthogonal: the largest |cosine| is 0.159. The design does not ask for
    that.
- **The û_off rule.** The bound text is design §2's construction, word for word in content: `Δx`, `Δr`,
  `read = (Δr/⟨w_c,w_c⟩)·w_c`, the centred off-read part, `û_off` and `s = |Δr|/‖w_c‖`, citing §2 and §14 D2. It
  omits §2's overshoot clause (N2).

### 5. The stage-1 binding: PASS

- **The paths** are the three files in `experiments/101-…/stage1/`, which are listed exactly in
  `NON_SCIENTIFIC_PATHS`. A simulated `validate_lock` with them committed accepts.
- **The layout.** My layout from the 36 frozen frames equals the lock's 324 blocks and `ri.stage1_binding`, both with
  the lock's `d_model` and with stage 1's `W_E.shape[1]` form. It has 9 fields per frame: `x1_all`…`x5_all`
  `[p_t+1, 512]`, `h6 [512]`, `rows4` and `rows5` `[8, p_t+1]`, and `c_ref [80]`. `d_model` 512 comes from the
  checkpoint and 8 heads from `config.json`.
  - **`c_ref` has 80 entries, not 79:** `rd.capture_frame_020` arranges `c_ref` over all of `nouns.nouns`, with
    `peach` as 0.0. The same formula fits all 108 of 020's locked states (built by the same `rd.locked_state`).
  - **A synthetic test** (random float64 values, no model output, in this review's scratch directory):
    `ul.write_table` → `ul.verify_table_index` (in memory and re-read) → `ul.read_table` → `ri.states_from_blocks`
    accepts the lock's layout and metadata. The state digests round-trip exactly. So the bound layout and metadata will
    not make the stage-1 writer fail after the ledger.
- **The table's metadata** is the bound dict: `fractions_for_weak: false`, mask 30
  (`b0c.P1_MASK["coordinated"]`, 023's completed program), the 11 predicted arms, cue order two/three/many, the 79
  noun keys, reference 581, `weak_shift` −1.0 and schema 1.
  - I computed a table with the frozen code, the confirm setting and the exposed spent states, and passed it through a
    JSON round trip. It still passes the barrier's meta check, and its canonical JSON and content digest are unchanged.
    No exposed frame is predicted weak.
- **The freeze text** states the write-once, the digests, the user's commit and the barrier.
- **The barrier, by reading the code** (`run.py:737–809`), against each tamper scenario of plan § The prediction
  freeze. Every scenario is covered.

  | scenario | where | class |
  |---|---|---|
  | stage-1 files missing or untracked | `run.py:790–791` | refusal, before any write or load |
  | staged but uncommitted, or a dirty tree | `_state_for` (`:352`), `validate_lock` (`read_intervention.py:2014`) | refusal |
  | lock or preregistration invalid | `:792`, `validate_lock` | refusal |
  | runtime, versions, blobs or input pins | `_inputs`, `_state_for`, `_check_runtime`, `_model_dependencies`, `_check_stage1_binding` | refusal |
  | forbidden-set isolation | `_confirmation`, `:793` | refusal |
  | G1, or the `w_c`/`v_j` digests | `_geometry_against_lock` (`:457–466`) | incident, recorded at `:807–809` |
  | the stage-1 record's own digest, or another lock | `_barrier` (`:772–775`) | incident |
  | a committed file's sha256 or path | `_committed_stage1` (`:742–744`) | incident |
  | the states index against the layout or metadata, or a state digest | `:746–749` (`read_table` also checks the bytes) | incident |
  | the table's digest, or its bound construction | `:751–755` | incident |
  | the table recomputed, any bit | `:777–778` | incident |
  | an S2 key already in the ledger | `:779–781` | incident |

  - The S2 ledger reaches disk only after the barrier (`:812–814`). The table is recomputed again before the result
    write (`:847–853`).
  - A non-`IncidentError` raised inside the barrier would crash without a record. That needs a record that is
    digest-consistent but malformed, or a state file corrupted between `_state_for` and the barrier. Neither is a
    realistic path.

### 6. The preregistration: PASS (wording notes in N3)

- **The file** equals `ri.render_preregistration(candidate)` byte for byte, and also the rendering of my rebuilt lock.
  Its sha256 equals the state's `preregistration_sha256`.
- **The labels, thresholds, order, scope and hierarchy** match the design: Holm 26/25 at 1/40 on the two count
  p-values alone, median floors 0.05, D 25/25, P 27 with `max(τ, 0.02)`, the precondition 30, the R → I → D → P
  order, the weak-frame rule, the scope sentence and the four hierarchy entries.
- **Over- and under-statements** are listed in N3. The most material is the third non-claim, "no label generalizes
  beyond … these frames", which is narrower than the scope's "in coordinated frames" and §14's "in new coordinated
  frames".

### 7. Determinism: PASS

- **The exposed prediction table** (`fractions_for_weak=True`, 020's locked states) recomputed in my process gives
  `ce055e09…`, equal to the calibration's `table_content_sha256`. The processes so far: calibration, the calibration
  review, the dry run, the production lock and this review. My run took 13.4 s, on 4 torch threads (the 020 runtime).
- **The geometry block**, computed twice in my process, is identical in canonical JSON. It also equals the lock's
  bound block bit for bit through `ri.verify_geometry_against_lock`.

### 8. The calibration's numbers as bound: PASS

- **τ.** `calibration_101.tau_rem` and `tau_ins` equal the installed file's `tau.rem` and `tau.ins`. Each is the 33rd
  of the 36 sorted values in that file's `distributions.P_error_rem` and `P_error_ins`, all 36 present. Both are below
  0.20, `evaluable` is true, no trigger fired, and the calibration's module blob is HEAD's.
- **Nothing in the lock depends on the size of an exposed N, S, D or P result.**
  - From the calibration the lock binds only the path, the file and content digests, τ and `evaluable`. No `shares`,
    `power`, `distributions`, `reports`, `min_s` or measurement digest appears in it.
  - τ depends on P's exposed errors by design (§6).
  - Every threshold, tolerance and constant equals the design, the plan and the module, not a calibration value.

### 9. Leakage and the prediction freeze: PASS (see S1)

- **Nothing in the lock path ran, or could run, a 101 frame prompt** (item 2). The lock computed nothing on a 101
  token sequence, even through the program. The ledger is empty.
- **The stage-1 launcher requirements** of plan § One-shot protections are stated in the draft launcher's docstring
  (`launch_stage1.py:20–29`): only the 36 S1 keys through `ri.capture_reference`, the sealed model, and the write
  allow-list of the three files plus the `stage1/` mkdir.
  - Its rehearsal does not do what the docstring and the plan say (S1).
  - The lock evidence README does not itself restate the requirements. Add a line pointing to the plan section and the
    launcher.

### 10. What the human should know before committing

There is no blocker. The findings are below. The residual risk for confirm that the calibration review recorded (its
N1) stands unchanged (N1 here).

### 11. The prepared stage-1 launcher (version 1): three should-fix findings and notes

I read `launch_stage1.py` version 1 (sha256 `bf40e043…`, now preserved as `launch_stage1_v1_as_reviewed.py`) against
plan § One-shot protections and § The prediction freeze. A version 2 appeared during the review; see the addendum.

**What it gets right:**
- **The forward route.** It allows a forward only inside the sanctioned `ri.capture_reference` (`:241–256`): tag `S1`,
  a key among the 36 `frame|ref|581|S1` keys, each once (`done_keys`), and only after `_ledger_on_disk()` has seen
  exactly those 36 keys in the on-disk ledger with `stage1` running (`:231–238`). That matches `run.py:688–690`.
- **Everything else refuses.**
  - `ri.run_arm` refuses outright (`:259–265`).
  - The low-level entry points refuse outside a sanctioned call (`:215–228`), and so does `Module.__call__`
    (`:207–214`).
  - Every module instance and class method of the loaded model is gated (`:281–324`).
  - A second model load is refused (`:331–337`).
- **No target prompt can run.** A target prompt's key carries `two`/`three`/`many` and never matches an allowed key,
  and `ri.capture_reference` itself refuses any prompt other than `ref` under `S1`.
- **The write allow-list** (`:63`, `:87–94`, `:122–146`) is exactly `results.json` with its `.results-*.json`
  temporary files and one rename each, the three stage-1 files, and the mkdir of `outputs/experiment-101` and
  `stage1/`. The runner's `write_check` is the launcher's predicate (`:344–350`, passed at `:394`).
- **The hook against the real events.** I recorded the events Python 3.12.13 raises for the exact stage-1 operations
  (`out/f_audit_events.json`) and read the hook against them:
  - `os.mkdir (path, 511, -1)`;
  - `open (path, "w", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC)` for `write_bytes` and `write_text`;
  - `open (path, "r", O_CLOEXEC)` for the hashing reads;
  - `tempfile.mkstemp`, then `open (tmp, None, O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC)`, `open (fd, "w", …)`,
    `os.rename (tmp, results.json, -1, -1)` and `open (dir, None, O_CLOEXEC)`;
  - `os.remove (tmp, -1)`.

  Every event of a correct run is allowed, and every other repository write through `open` is refused.
- **It cannot run twice.** The O_EXCL sentinel, the runner's phase rules (`stage1` refuses once it is running,
  complete or has an incident) and the existence check on the stage-1 files each prevent a second run.
- **The preflight** includes the full `_validated_lock`, `_state_for("stage1")`, the 020 runtime, the empty ledger,
  the absence of the stage-1 files, the 36 S1 keys, the module binding and the environment against the lock run.
- **The post-run checks** confirm that the files exist, that their sha256s equal the record's, that the record's digest
  reproduces, and that all 36 keys ran.

The findings are S1–S3 and N8 below.

## Maximum numerical disagreement per recomputed quantity

| quantity | mine vs the lock or `ri` |
|---|---|
| `w_c` from 011's lock | torch arithmetic **0.0** (digest equal); numpy `fsum` mean 6.9e-18 |
| ‖w_c‖, ⟨w_c, w_c⟩ | **0.0** |
| `ŵ`, `d̂_T` | **0.0** (digests equal) |
| read weight from the raw safetensors vs 011's `read_weight`; centred vs `w_c`; vs frozen `ra.read_weight` | **0.0**; **0.0**; **0.0** |
| G1 `read_weight_max_abs` | 0.0 = the lock |
| the 288 `v_j` vs `ri.iso_direction` | torch arithmetic: 288/288 digests equal (**0.0**); numpy arithmetic 8.3e-17 |
| `v_j` unit, mean, ⟨·, w_c⟩, ⟨·, ŵ⟩ (mine, `fsum`) | 5.6e-16, 2.2e-18, 2.4e-17, 3.2e-17 (all ≤ 1e-12) |
| basis norms, ⟨𝟙/√d, ŵ⟩ | 1.1e-16 (lock 1.1e-16), 1.4e-17 (lock 1.0e-17) |
| stage-1 layout (324 blocks), metadata | exact |
| lock content; confirmation content and manifest; calibration content; 020 exposed states; embedding; parameters; state digests | exact |
| exposed prediction table digest | exact (`ce055e09…`) |
| τ_rem, τ_ins | exact |
| Holm thresholds and tails | exact (26, 25; 0.005665492091793567, 0.014408359827939421) |
| `ri.build_lock`, `ri.render_preregistration` | byte for byte |

## Findings (most severe first)

**Blockers: none.**

**Should-fix: before `stage1`, in the prepared launcher (version 1, `bf40e043…`).** None of them affects the lock or
its commit. Version 2 (`13b1992f…`) appears to address all three; see the addendum.

- **S1. The rehearsal does not exercise the stage-1 write path under the hook.**
  - **The requirement.** Plan § One-shot protections asks for it ("its rehearsal mode exercises the stage-1 write path
    under the hook"). The launcher's docstring (`launch_stage1.py:16–18`) claims "the hook is exercised with synthetic
    events for each (no file is created)".
  - **The code.** No such code exists. In rehearsal, `WRITABLE` excludes the three files (`:63`). The preflight only
    asks the predicate `write_allowed` (`:408–413`), and the rehearsal branch ends at `:429–430`.
  - **The scenario.** The hook first meets the real events (the `stage1/` mkdir, three opens for writing, the state's
    temporary file, rename and fsync) in production. By then the 36 S1 keys are on disk and the 36 captures have run.
    Any mishandled event makes `ul.write_table` raise `PermissionError`, which is recorded as a stage-1 incident
    `after_ledger` (`run.py:723–727`). The 36 frames are then spent and a new protocol version is needed (plan
    § Incidents).
  - **My reading.** Against the recorded event signatures the current hook would allow a correct run. The rehearsal
    should prove it, as planned.
  - **The fix.** In rehearsal, feed the hook the recorded event tuples for the three paths, the `stage1/` mkdir and
    the state's temporary, rename and fsync events with production's predicate, and assert allowed versus refused.
    Then fix the docstring.

- **S2. The hook fails open for relative paths in non-`open` events.**
  - **Where.** `launch_stage1.py:67–76` with `:122–128` and `:136–142`. The same `_path` is in
    `launch_lock.py:53–62/96–120` and `launch_calibrate.py:80–89`.
  - **The defect.** Python 3.12 passes `dir_fd = -1` when no descriptor is given. I recorded this, on scratch paths,
    for `os.mkdir (p, 511, -1)`, `os.chmod (p, 493, -1)`, `os.remove (p, -1)`, `os.rmdir (p, -1)` and
    `os.rename (a, b, -1, -1)`. `_path` treats −1 as a descriptor, and `fcntl.fcntl(-1, F_GETPATH)` raises
    `ValueError`, which I tested ("file descriptor cannot be a negative integer"). `_path` then returns `None`, and the
    hook returns without refusing or recording.
  - **The scenario.** With the working directory at the repository root (`:57`), a relative `os.mkdir("x")`,
    `os.remove("x")`, `os.rmdir("x")`, `os.rename("a", "b")` or `os.chmod("x", …)` passes silently. So does any of
    them on a relative path under `experiments/` or `src/`. Writes through `open` are not affected, because that branch
    passes no descriptor and resolves relative paths against the working directory.
  - **A regression.** The freeze launcher used only arguments ≥ 0 as descriptors and resolved relative paths against
    the working directory. The per-event mapping that applied the freeze review's N1 regressed this.
  - **Effect on the lock: none.** Every runner path is absolute, the recorded events match the code, and a scan finds
    no repository change other than the three outputs.
  - **The fix.** Treat `-1`/`None` as no descriptor, resolve against the working directory, and refuse when a path
    cannot be resolved (fail closed). While there, resolve `os.open(..., dir_fd=…)` in the `open` branch; that is the
    freeze review's N1, third bullet, still open.

- **S3. The sentinel comes before the preflight, and the run record is always overwritten.**
  - **Where.** `launch_stage1.py:419–425` and `:449`.
  - **(a) A blocked retry.** A refusal-class problem leaves `stage1.sentinel` behind, so the retry that plan § Global
    constraints allows for refusals needs the sentinel deleted or the launcher edited. The problems include:
    - a mistyped `STAGE1101_LOCK`;
    - an OS or package update caught by the launcher's environment check (its preflight item N2, `:365–370`);
    - `evidence/lock-2026-10-14/lock_run.json` not yet committed (`:55`, `:365`);
    - a dirty tree;
    - a refusal inside `stage1()` before its ledger.
  - **(b) A lost record.** Any later invocation, even one refused at the sentinel, overwrites `production_run.json`,
    the only launcher record of the one stage 1, with its refusal.
  - **The fix.** Create the sentinel after the preflight, immediately before `stage1()`, as the lock launcher did
    (`launch_lock.py:274–281`). Write records with O_EXCL or unique names.

**Notes.**

- **N1. The Level-1 risk at confirm (the calibration review's N1, unchanged).**
  - The lock binds `G4_level1` = 2e-2 against an exposed maximum of 0.010654 (1.88×, a stratum arm; `two`'s arms at
    most 0.00501).
  - The calibration review estimated the chance of a confirm exceedance at about 0.3 % (0.02–1 %, and about 10 % at
    the 95th percentile of a heavy-tailed bootstrap) (`calibration-review…/REVIEW.md:400–497`). An exceedance makes
    confirm `NOT_INTERPRETABLE` and spends the frames (rule 2). It can never produce a false positive.
  - **No amendment is possible in this protocol version.** `load_calibration_101` binds the module blob and the
    tolerances (`read_intervention.py:1855–1856`), and calibrate does not rerun. So an amendment needs a new protocol
    version whether or not the lock is committed.
  - Accepting this risk is the human's decision.
- **N2. Design §2's overshoot report is not recorded at confirm.**
  - **The requirement.** §2 (`design.md:73–75`): "the frame is flagged; `s/‖off_c‖` is reported for every frame".
  - **The gap.** The bound `u_off_rule` omits the clause. At confirm no code records the measured pair quantities
    (`Δr`, `‖Δx‖`, `‖off_c‖`, `s`, `s/‖off_c‖`, the overshoot flag):
    - `ri.frame_values` (`:1475–1508`), `run_records` and `secondary_statistics` carry none of them;
    - only calibrate does (`run.py:549–555`);
    - the table holds the program's `Δx̂3` pair, not the measured one.
  - **The impact.** The flag has no label force, and the values are recomputable from the saved `x3_pc` rows of REF
    and `PL@w`. The confirmation review or the report should produce them.
- **N3. Preregistration wording.** It is bound by the module blob, so a change needs a new protocol version.
  - **(a)** The non-claim "no label generalizes beyond the one/two pair, this site, this checkpoint and these frames"
    (`read_intervention.py:429`) is narrower than the scope sentence on the same page ("in coordinated frames") and
    than §14's primary question ("in new coordinated frames"). It is the conservative direction. A later claim review
    should state which population it reads.
  - **(b)** §5's rule that a frame whose predicted `Δĉ_full` lies above −1 counts as a P disagreement is implemented
    (`:1592`) but not stated in the rendering.
  - **(c)** "The gates" lists only the 15 tolerances, not the bit-for-bit G3 NOOP, G4 `x3(p_t)`, G5 landing and G6
    checks, nor G7 (`:1964–1965`).
  - **(d)** The head counts `F_head_rem < 1` and `F_head_ins > 0` (`:269–270`, `:1880–1882`) are minimal-effect
    criteria that §7 left open. Exposed values are 36/36. They do not by themselves show that the head carries most
    of the effect; the shares do that.
  - **(e)** "P evaluable True" uses Python's bool formatting.
- **N4. Provenance.**
  - `3621eec` and its nine predecessors since `96f0cc8` exist only locally.
  - `stage1` and `confirm` require the lock commit to be an ancestor of HEAD (`read_intervention.py:2016–2017`,
    `run.py:81–88`). A rebase, squash or amend of `exp101-phases` would make every later phase refuse.
  - Push the branch as it is; do not rewrite it (the freeze review's N4).
- **N5. Evidence preservation.**
  - The lock's records live only in the scratchpad.
  - `launch_lock.py:293` rewrites `lock_run.json` on every invocation, including one whose preflight refuses because
    the lock has already run. So an accidental re-run destroys the production record.
  - Copy the directory byte for byte, now, with a `SHA256SUMS` file. The digests as reviewed:

    | file | sha256 |
    |---|---|
    | `launch_lock.py` | `90e8159f97fc38234835c58e6443474677dfe0236ca48d09e6e67fe124e5e485` |
    | `lock_run.json` | `4d98ec9fd26d135ee59a3f3e91a294bf690826b64fd9af498ac905d6050cfe5f` |
    | `lock.log` | `c306135acc773541d0d4bb159789083a0284892f5d7abcd3d441b8beb502fa47` |
    | `lock.exit` | `28d3b9e880a77975493dc7e359144c0295a4f694cfe0af4f928c22307bc5c320` |
    | `lock.sentinel` | `6c2c645453887bed8b54ff822eae002d11a67e089e129858af02d33596faec1e` |
    | `dryrun_run.json` | `7768ef30ea744bce6e34af683a303a6ff58e634a9b7707743d72b852b5a7c8b2` |
    | `dryrun.log` | `628196714179fe33ef48de670cab5929ddca10a1dace460d55628847db0a4142` |
    | `state_before_dryrun.txt` | `ce0d7abec828afe1c2151d2628546e45dca457d86e5fcefcd8b8fe4925a2c660` |
    | `launcher_sha256_at_production.txt` | `0837bafd92d5c0de76fc8076fe1e63eeea78d5e4eed8fc920b36abb5928644ed` |

  - The draft stage-1 launcher reads `experiments/101-…/evidence/lock-2026-10-14/lock_run.json` from the repository
    (`launch_stage1.py:55`, `:365`). That copy must be committed before stage 1. `DECISION.md`'s commands install only
    the two lock files.
- **N6. `inputs.table_022_file` cannot be verified here.** The value `04659d5e…` is a constant carried from 023's code
  (`block0_completion.py:96`, `:102`, `:211`): the sha256 of the git-ignored `outputs/experiment-022/calibration-table.pt`,
  which no 101 phase reads or verifies and which my read restriction forbids. It binds nothing 101 computes.
- **N7. The lock ran before the human's decision on N1.** The calibration review had put that decision "before the
  lock". Running first was harmless:
  - the lock ran no prompt and only wrote candidates;
  - an amendment needs a new protocol version either way (N1).

  The operator's README says so (`lock/README.md:20–25`). The human's decision is the commit.
- **N8. Minor points in the stage-1 launcher.**
  - In rehearsal the hook still allows a direct write of `results.json` (`:63`).
  - `tree_after` is recorded but not asserted to list exactly the three stage-1 files (`:446`).
  - `sys.flags.dont_write_bytecode` is not required. Without it, the refused `__pycache__` writes would appear as
    forbidden writes, which is harmless but alarming.
  - The post-run checks do not assert that the ledger is exactly the 36 S1 keys. The runner guarantees it.
  - The launcher's environment check (its preflight N2) compares `platform.platform()`, so it refuses after an OS
    update. That is right; see S3(a) for the retry.
- **N9. The geometry is compared bit for bit, and a mismatch is an incident.** At stage 1 and confirm the whole block,
  `g1`'s float residuals included, is compared bit for bit. A mismatch is a phase incident, not a refusal
  (`run.py:680–684`, plan step 2). So an environment change that the runtime and version checks miss, such as an OS or
  BLAS update, would consume the protocol version before any 101 prompt runs.
  - The stage-1 launcher's environment check (its preflight N2: platform, packages, checkpoint) mitigates it.
  - I reproduced the block bit for bit in a separate process, 30 minutes after the lock.
  - Run stage 1 on this machine with no system update in between.
- **N10. The runner's `git status` runs without `--no-optional-locks`.** Its `collect_git_state` uses plain
  `git status` (`provenance.py:205–211`).
  - `.git`'s mtime moved at 12:32:10 local, just after the lock, and again at 13:09:10 local (10:09:10 UTC), during
    this review, from a process other than mine. Both are consistent with a transient `index.lock`.
  - `.git/index` is unchanged since the `3621eec` commit (12:29:58 local), and no ref or object changed.
  - It is harmless; the calibration review's N6 says the same.
- **N11. `DECISION.md`, which is not part of the lock.** These points concern the version I first read
  (`7ee2f6f7…`). The revised version (`c4e9482a…`, 13:12 local) fixes all three.
  - The `validate` command (`:119`) lacked `-B` and the project's environment flags. The `.pyc` files it would write
    are git-ignored.
  - "The eight isotropic controls do nothing" (`:63`) slightly overstated: exposed `F_ins_iso` ranged −0.214 to
    0.149.
  - "very likely to pass" (`:75`) leaned on exposed frames that are in-sample for `d̂_T`, a caveat its §3 already
    stated.

  Both versions still install only the two lock files (N5).

## Addendum: stage-1 launcher version 2 and the operator's use of this draft (13:10–13:12 local)

While this report was being finalized, the operator worked from its draft.
- **What changed.**
  - It preserved version 1 byte for byte, as `stage1/launch_stage1_v1_as_reviewed.py` and
    `lock/prepared/launch_stage1_v1_as_reviewed.py` (both `bf40e043…`, verified).
  - It wrote version 2 to `stage1/launch_stage1.py` and `lock/prepared/launch_stage1.py` (both
    `13b1992f7f0feb652207ca835df044244f20faea750e92f42d8d51bced7c3e1c`).
  - It added `lock/prepared/hook_self_test.py`, which extracts the decision functions with `ast` and runs them in a
    bare namespace, and its output `hook_self_test.out.json` (`ok: true`).
  - It revised `lock/DECISION.md` (now `c4e9482a…`) to cite S1–S3, N5, N8 and N9.
- **This is N12 of the freeze review again.** The draft was cited before hand-back. The verdict and every finding are
  unchanged since that draft. Afterwards I corrected counts and wording, added evidence (the relative-path probe and
  the `.git` mtime note), and replaced `shutil.rmtree` by `os.rmdir` in S2's scenario list (`shutil.rmtree` passes
  `dir_fd` None, so it was never fail-open).
- **A limited re-check of version 2.** I read the full diff from version 1, not the whole file, and ran nothing.
  - **S1.** `decide()` is a pure decision function shared by the hook and the self-test. `hook_self_test()` runs in
    every mode before any stage-1 action. It checks 9 recorded stage-1 event tuples, which must be allowed, against 10
    refused controls, and the rehearsal refusing the stage-1 writes.
  - **S2.** `_path` treats only `dir_fd` ≥ 0 as a descriptor and resolves other relative names against the working
    directory. An unresolvable name becomes `UNRESOLVED`, which `_in_repo` counts as inside the repository, so every
    write check refuses it (fail closed). The `open` branch still ignores `os.open(..., dir_fd=…)`, which no stage-1
    code uses.
  - **S3.** The sentinel is now created after the preflight and the self-test, immediately before `stage1()`. Each
    record is a new O_EXCL file with a timestamped name.
  - **N8.** The rehearsal predicate allows nothing. `-B` is required. The post-run checks assert that the ledger is the
    36 S1 keys and that the tree shows exactly the three untracked stage-1 files, and the exit status requires them.

  I found no new defect in the diff. This is not a full review of version 2. The re-check before stage 1 that
  `DECISION.md` recommends still applies.

## Files (this directory)

| file | what |
|---|---|
| `PROMPT.md` | the brief |
| `REVIEW.md` | this report |
| `scripts/common.py` | the guard (audit hook, forbidden sequences, refusing `Module.__call__`, entry-point wrapping, sealing) and my digest and canonical-JSON functions |
| `scripts/mine.py` | my reconstructions: `w_c` (from 011's lock and the raw weights), the SHA-256 counter-mode `v_j`, the manifests, the layout, the noun order, the plan's constants |
| `scripts/a_repo.py` → `out/a_repo.json`, `out/a_repo.log` | item 1 and the evidence digests (independent) |
| `scripts/b_independent.py` → `out/b_independent.json`, `.log` | items 3–6 and 8, my code (82 checks) |
| `scripts/c_frozen.py` → `out/c_frozen.json`, `.log` | the frozen cross-check under the guard (`validate`, `base.digests`, phase rules, the geometry ×2, the exposed table, `stage1_binding`, `build_lock`, `render_preregistration`, `validate_lock` simulations, the synthetic stage-1 writer test in `out/stage1_writer_test/`) |
| `scripts/d_params.py` → `out/d_params.json`, `.log` | the bridge's parameters against the raw checkpoint |
| `scripts/e_state_delta.py` → `out/e_state_delta.json`, `.log` | the pre-lock state rebuilt |
| `scripts/f_audit_events.py` → `out/f_audit_events.json`, `.log` | Python 3.12.13's audit events for stage 1's file operations, on scratch paths (`out/audit_event_probe/`) |
| `scripts/g_iso_checks.py` → `out/g_iso_checks.json`, `.log` | G1's checks on the exact `v_j` |
| `scripts/h_relative_events.py` → `out/h_relative_events.log` | S2: the audit arguments of relative-path calls (`dir_fd` −1) and `F_GETPATH(-1)`, on scratch paths |
| `out/repo_snapshot_end.txt` | the repository at the end (the start values are in `out/a_repo.json`) |
| `SHA256SUMS` | this directory's digests |
