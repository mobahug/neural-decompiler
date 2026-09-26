# Experiment 025 — the production freeze (2026-09-26)

The production `freeze` ran **exactly once**, at implementation commit `c765148fdb612ae249e21ef3c830e4ceb1d12dd0`, under
a guarded launcher. It wrote one file, `confirmation-v1.json`.
- The file was independently verified before installation (`../freeze-review-2026-09-26/`): **PASS WITH NOTES — no
  blockers**.
- It was installed byte-identically in commit `57f1f989b1095708f4805c57be251b2392febc51`. It was never regenerated,
  parsed and rewritten, or re-serialized.
- No scientific or rotated prompt ran. No lock, confirm or report has run.

## 1. The final implementation test gate (before the freeze)

The implementation was complete at `c765148fdb612ae249e21ef3c830e4ceb1d12dd0`, after the independent implementation
review (PASS WITH NOTES, no blockers) and its fixes. On that exact commit, `HEAD == origin/main == remote` and the tree
was clean before and after each run. The logs are in `test-gate/`.

| tier | result | log |
|---|---|---|
| A (`pytest`) | **542 passed, 0 failed** | `test-gate/tierA.log` |
| B (`pytest --tier B`, one full run from scratch, 15:45:04–15:53:53 UTC) | **575 passed, 0 failed**; 20 model tests skipped (the opt-in tier-C tests); 210 historical tests deselected | `test-gate/tierB.log` |
| C (`NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 … -m pythia_smoke tests/test_cue_rotation.py`, the four spent keys only) | **6 passed, 0 failed** | `test-gate/tierC.log` |

**Correction.** An earlier status message said "tier B 595 passed". That was wrong: 595 was the number of tests
*selected*, not passed. The run it described had 574 passed, 1 failed and 20 skipped. The one failure was an
order-sensitive test assertion, fixed by `c765148`. The gate above is one complete run on the final commit; no counts
from earlier runs are combined.

## 2. The freeze run

- **The launcher (`launch_freeze.py`)** follows 024's guarded-launcher pattern. Before the runner is imported, it
  refuses:
  - the model loader, and every `torch.nn.Module.__call__`;
  - the four capture and intervention entry points of `plural_mechanism`;
  - every score and geometry function: 024's nounness score, centroids and bindings, and 025's direction, cue
    geometry, controls, plurality direction, nearest tokens, patched measurement and statistics;
  - through an audit hook, reads of 023's or 024's local outputs or 022's calibration table, and every repository
    write except the confirmation file.

  It then verifies everything in memory before writing, calls `runner.freeze()` once, and checks that the written
  bytes are the pre-verified payload.
- **The dry run (`dryrun_run.json`),** 15:56:14–15:56:25 UTC:
  - It ran the same launcher with `FREEZE025_DRY_RUN=1`: every guard and every pre-write check, in memory.
  - It did not call `freeze`, and it wrote nothing: no confirmation file existed afterwards.
  - Its standard output was not saved; its JSON record holds the checks and facts.
- **The production run (`freeze_run.json`, `launch.log` = stdout and stderr),** 15:56:49–15:57:07 UTC:
  - **exit status 0**;
  - **freeze invocations: 1**;
  - **model loads 0, forwards (module calls) 0, captures and interventions 0, score calls 0, geometry calls 0**;
  - forbidden reads 0, forbidden writes 0;
  - **repository writes: 1**, the confirmation file;
  - all 18 pre-write checks true;
  - written bytes equal to the pre-verified payload.
- **Pre-install checks (`pre_install_checks.py/.out`):**
  - `HEAD == origin/main == remote == c765148…`;
  - the tree held exactly the untracked freeze file;
  - no `outputs/experiment-025/` and no lock;
  - both hashes, the manifest count and its digest exact;
  - no `.gitattributes`, no active hook, and no `autocrlf`.
- **Post-commit checks (`post_commit_checks.py/.out`)** ran with the model, forwards and score/geometry functions all
  refused. They show:
  - the install commit touches exactly one path;
  - `git show HEAD:<path>` is byte-identical to the reviewed file, and both hashes are exact;
  - the 40 cues, and the whole file, reconstruct mechanically byte for byte;
  - the manifest is 90,720 keys with the same digest;
  - there are zero collisions with the 43,632 spent keys and with the 4 patch-path keys;
  - the runner's `validate` passes.

## 3. The artifact

- `experiments/025-nounness-direction-intervention/confirmation-v1.json`, 4,123,258 bytes
- **file sha256 `54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a`**
- **content sha256 `6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4`**
- **manifest: 90,720 keys, sha256 `0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61`**

## 4. The population (as reviewed)

The selection used only:
- the frozen textual lists (024's adjective and ordinary reserves, and the frozen alphabetical 39-word new list);
- the pinned tokenizer (`EleutherAI/pythia-70m-deduped` @ `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`);
- the exclusion sets;
- mechanical eligibility in list order.

It used no nounness score and no intervention geometry: every score and geometry function was refused and none was
called.

**The exclusion sets:**

| set | ids |
|---|---|
| earlier cues (024's committed exclusion, 351, plus 024's 40 cues) | 391 |
| pool target-noun forms (80 pool nouns) | 161 |
| prior confirmation-noun forms (the 24 nouns of 020's confirmation file, which 021 read) | 48 |
| exposed-frame tokens (108 frames) | 316 |

- `statue` (23957/38490) is excluded by the tightened prior-noun rule, and only by that rule.
- `barrel` (15474/33545) belongs to the same prior-noun exclusion; it is on none of 025's lists.

| list | candidates | rejected | eligible | picked | eligible, not picked |
|---|---|---|---|---|---|
| adjective reserves | 23 | 0 | 23 | 20 | fuzzy, latest, earliest |
| ordinary reserves | 9 | 1 (statue: prior-noun rule) | 8 | 8 | none |
| new list | 39 | 22 (19 multi-token; mirror, planet, rocket are pool nouns) | 17 | 12 | robot, shark, singer, snake, whale |

The new-list picks are its first 12 eligible entries, in the list's frozen textual order.

**The 40 cues** (singular id; plural id for nouns):
- **Adjectives (20):** anxious 20138, cheerful 39567, curious 14338, jealous 23327, lonely 25106, nasty 26321, careful
  10182, careless 48292, famous 8530, friendly 11453, gorgeous 23535, hungry 18254, weary 35725, wicked 26395, ugly
  19513, vivid 24863, vague 21248, rapid 5233, rigid 16572, clever 19080.
- **Ordinary reserves (8):** soldier 15796/9647, sailor 45758/35089, priest 14959/24061, knight 30605/44773, onion
  21635/27408, carrot 47215/39033, pirate 42404/44545, tourist 22777/24359.
- **New list (12):** author 2488/4477, bishop 29417/39180, dancer 42411/39576, duck 27985/46495, goat 23244/36507,
  guitar 12609/47087, hunter 32290/35041, lawyer 11115/16099, monk 35295/34955, nurse 15339/16675, painter
  27343/41472, prince 24012/47776.

The 60 ids (40 cue ids, 20 plural ids) are distinct, and none is in any exclusion set.

## 5. The manifest's semantics

`40 cues × 108 frames × 21 conditions = 90,720` unique condition-tagged scientific runs, each keyed
`frame_id|word|token_id|condition`. They cover the 4,320 cue × frame prompts.

The 21 frozen conditions:

| conditions | count |
|---|---|
| baseline (`base`) | 1 |
| nounness, primary dose ± (`noun±0.32`) | 2 |
| nounness, half dose ± (`noun±0.16`) | 2 |
| 7 random tangent controls × ± (`rand1…7±0.32`) | 14 |
| plurality control ± (`plur±0.32`) | 2 |

- **A and B use exactly 16 primary-dose conditions:** `noun±0.32` and the seven random directions × ±. G uses
  `D_attn` at `noun±0.32`.
- **The spent set is 43,632 keys:**
  - 020's ledger;
  - 020's confirmation set;
  - the 022, 023 and 024 manifests.

  No tagged or untagged manifest key collides with it.
- **The four patch-path engineering keys** are all in 020's ledger, and none is a manifest key. They stay outside the
  scientific manifest, the scientific ledger and the scientific accounting.

## 6. Protocol note (the reviewer's decision on the verifier's N3)

> The freeze implementation performs shared dependency/integrity reads beyond the minimal phase-table inputs.
> Independent call-flow verification established that these data do not feed population selection or manifest
> construction. Freeze selection remains tokenizer/list/exclusion based.

The reads in question are 020's local results, 023's committed cells input, and 024's calibration record and lock. The
reviewer accepted them as integrity and precondition reads only. The implementation was not changed. The reviewed
freeze remains authoritative.

**The tokenizer name.** The verification brief named the tokenizer `EleutherAI/pythia-70m`, which was a wording
mistake. The pinned checkpoint and tokenizer family is `EleutherAI/pythia-70m-deduped`. The production freeze and the
verifier both used it; the mistake had no effect.

## 7. Requirements for the lock and confirm launchers (from the freeze review)

These are recorded now; no code was changed for the completed freeze. The lock launcher, and especially the confirm
launcher, must also:
- record the full HEAD commit, the environment, package and runtime information, and the launcher's own file sha256;
- block and log direct `.forward()` routes, not only `torch.nn.Module.__call__`;
- block and log unpatched aliases or copies of the capture and intervention functions (`capture.run_capture`,
  `interventions.run_interventions`), not only `plural_mechanism`'s;
- block repository deletes and renames, not only opens;
- keep the 4 spent patch-path runs apart from the 90,720 fresh condition-tagged runs;
- log the condition of every patched fresh run: `run_patched` sees untagged keys, so map the replacement's digest to
  the lock's vectors;
- live under `evidence/`, so that it is not a scientific-path change.

For confirm, these protections must be in place before any fresh ledger entry and before any prompt.

## Files

| file | what |
|---|---|
| `test-gate/tierA.log`, `tierB.log`, `tierC.log` | the final implementation test gate at `c765148` |
| `launch_freeze.py` | the guarded launcher (dry-run mode by `FREEZE025_DRY_RUN=1`) |
| `dryrun_run.json` | the dry run's record (no freeze call, no write) |
| `freeze_run.json`, `launch.log` | the production run's record and its stdout/stderr |
| `pre_install_checks.py/.out` | the checks before installing the file |
| `post_commit_checks.py/.out` | the checks after the install commit |
| `SHA256SUMS` | this directory's digests |
