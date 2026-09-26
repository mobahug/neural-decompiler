> **Archived verdict: PASS WITH NOTES — no blockers.** The reviewer accepted it on 2026-09-26, together with the decision on N3: the extra reads are accepted as integrity/precondition reads, with no code change. That decision is recorded in `../freeze-2026-09-26/README.md`.
> Everything below the rule is the independent verifier's report, unchanged. Before this header was added, its sha256 was `8e14258f8f4fce17c63190aea88c92ac22d8eb1824685d0cc0b0fd5d659cdf71`.
> One point in it is resolved: the brief (`PROMPT.md`) named the tokenizer `EleutherAI/pythia-70m`, which was a wording mistake. The pinned checkpoint and tokenizer are `EleutherAI/pythia-70m-deduped`; the verifier used them, and so did the production freeze.

---

# Experiment 025 freeze: independent verification

**Verdict: PASS WITH NOTES.** There are no blockers and nothing to fix in the artifact.

`experiments/025-nounness-direction-intervention/confirmation-v1.json` (untracked) is the approved freeze:
- file sha256 `54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a`;
- content sha256 `6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4`.

I re-derived all of the following from the design's rules, the committed files and the pinned tokenizer, with my own
code, before touching the module:
- the four exclusion sets;
- the 40 cues and every rejection reason;
- the 90,720-key manifest and its digest;
- the configuration;
- the 43,632-key spent set.

Every one agrees with the file. Afterwards, `cr.freeze_payload` run in memory reproduces the file **byte for byte**.
The launcher's records are consistent with its code and with the repository. The notes (N1–N8) concern the evidence
record, the scope of the freeze's reads, the launcher's guard coverage and the commit housekeeping. None of them can
change the artifact.

Reviewed on 2026-09-26 at HEAD `c765148fdb612ae249e21ef3c830e4ceb1d12dd0`.

## How this review stayed inside the rules

- **Nothing in the repository was written, staged or committed.**
  - After all my runs, `git status` shows only the untracked artifact, and `.git/index` still has its 18:26:28 mtime.
  - No file in the repository (outside `.git` and `.venv`) is newer than the artifact.
- **Environment.** Every Python run used `.venv/bin/python` with `HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`. `uv` was
  never used.
- **The model and the weights.**
  - The model was never loaded. No weight file was opened: my audit hook refuses `*.safetensors`,
    `pytorch_model.bin` and `*.ckpt`, and it recorded no refusal.
  - No forward pass, capture or intervention ran. The only `nn.Module` call was the refused `torch.nn.Identity` in the
    guard test.
  - No score, centroid, direction, rotation, ℓ, A, B, G, D_attn or Level-1 value was computed.
- **Forbidden paths and phases.**
  - Nothing under `outputs/experiment-023/`, `outputs/experiment-024/` or any `calibration-table.pt` was opened. The
    hook refuses them and recorded no attempt; the launcher's hook was tested with synthetic arguments only.
  - `freeze`, `lock`, `confirm`, `report` and `validate` were never run.
- **Tokenizer.** `AutoTokenizer.from_pretrained(PYTHIA_70M.model_id, revision=PYTHIA_70M.revision,
  local_files_only=True)` is `EleutherAI/pythia-70m-deduped` @ `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` (see N6).
  - A second implementation cross-checks it: the raw `tokenizers` library on the snapshot's `tokenizer.json`, sha256
    `c24618a1b3e6…`.
  - The two agree on all 119 encodings.
- **Order of work.** The independent derivation (`derive_independent.py`) never imports `cue_rotation`; its output
  records `cue_rotation_imported: false`. The module and runner cross-check (`crosscheck_module.py`) ran afterwards.

## Key numbers

| quantity | value |
|---|---|
| artifact file sha256 | `54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a` (4,123,258 bytes; canonical JSON + `\n`) |
| `content_sha256` | `6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4` (recomputes) |
| `manifest_sha256` | `0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61` (recomputes; equals my own manifest's) |
| manifest | 90,720 keys = 40 × 108 × 21, unique, sorted; 4,320 untagged prompts; 4,320 per condition |
| exclusion sets | earlier cues 391, pool target-noun forms 161, prior-noun forms 48, exposed-frame tokens 316 |
| spent set | 43,632 = 30,132 + 3,060 + 3,060 + 3,060 + 4,320, pairwise disjoint; sha256 of the sorted list `4be44f20e581…4d4e62` (mine = the runner's `forbidden`) |
| collisions | 0 untagged, 0 tagged, 0 (frame, cue id) pairs, 0 of the 60 cue/plural ids ever prompted in any spent key; the 4 patch-path keys are all in 020's ledger and none is in the manifest |
| configuration | production; threshold 27; tail 5288280983/274877906944 = 0.01923865414210013 (26 gives 0.0403452…); 21 conditions; 16 outcome-bearing |

### The 40 cues (the file's order; each derived independently)

| # | stratum | word | id | plural | plural id | source, rank |
|---|---|---|---|---|---|---|
| 1 | adjective | anxious | 20138 | | | adjective-reserve 0 |
| 2 | adjective | cheerful | 39567 | | | adjective-reserve 1 |
| 3 | adjective | curious | 14338 | | | adjective-reserve 2 |
| 4 | adjective | jealous | 23327 | | | adjective-reserve 3 |
| 5 | adjective | lonely | 25106 | | | adjective-reserve 4 |
| 6 | adjective | nasty | 26321 | | | adjective-reserve 5 |
| 7 | adjective | careful | 10182 | | | adjective-reserve 6 |
| 8 | adjective | careless | 48292 | | | adjective-reserve 7 |
| 9 | adjective | famous | 8530 | | | adjective-reserve 8 |
| 10 | adjective | friendly | 11453 | | | adjective-reserve 9 |
| 11 | adjective | gorgeous | 23535 | | | adjective-reserve 10 |
| 12 | adjective | hungry | 18254 | | | adjective-reserve 11 |
| 13 | adjective | weary | 35725 | | | adjective-reserve 12 |
| 14 | adjective | wicked | 26395 | | | adjective-reserve 13 |
| 15 | adjective | ugly | 19513 | | | adjective-reserve 14 |
| 16 | adjective | vivid | 24863 | | | adjective-reserve 15 |
| 17 | adjective | vague | 21248 | | | adjective-reserve 16 |
| 18 | adjective | rapid | 5233 | | | adjective-reserve 17 |
| 19 | adjective | rigid | 16572 | | | adjective-reserve 18 |
| 20 | adjective | clever | 19080 | | | adjective-reserve 19 |
| 21 | noun | soldier | 15796 | soldiers | 9647 | ordinary-reserve 0 |
| 22 | noun | sailor | 45758 | sailors | 35089 | ordinary-reserve 1 |
| 23 | noun | priest | 14959 | priests | 24061 | ordinary-reserve 2 |
| 24 | noun | knight | 30605 | knights | 44773 | ordinary-reserve 3 |
| 25 | noun | onion | 21635 | onions | 27408 | ordinary-reserve 4 |
| 26 | noun | carrot | 47215 | carrots | 39033 | ordinary-reserve 5 |
| 27 | noun | pirate | 42404 | pirates | 44545 | ordinary-reserve 6 |
| 28 | noun | tourist | 22777 | tourists | 24359 | ordinary-reserve 7 |
| 29 | noun | author | 2488 | authors | 4477 | new-list 0 |
| 30 | noun | bishop | 29417 | bishops | 39180 | new-list 2 |
| 31 | noun | dancer | 42411 | dancers | 39576 | new-list 6 |
| 32 | noun | duck | 27985 | ducks | 46495 | new-list 9 |
| 33 | noun | goat | 23244 | goats | 36507 | new-list 12 |
| 34 | noun | guitar | 12609 | guitars | 47087 | new-list 13 |
| 35 | noun | hunter | 32290 | hunters | 35041 | new-list 15 |
| 36 | noun | lawyer | 11115 | lawyers | 16099 | new-list 17 |
| 37 | noun | monk | 35295 | monks | 34955 | new-list 19 |
| 38 | noun | nurse | 15339 | nurses | 16675 | new-list 20 |
| 39 | noun | painter | 27343 | painters | 41472 | new-list 22 |
| 40 | noun | prince | 24012 | princes | 47776 | new-list 27 |

The 60 forms have 60 distinct ids, and none of them is in any exclusion set.

### Per-list counts (independent; equal to the launcher's `facts`)

| list | candidates | rejected | eligible | picked | unpicked eligible |
|---|---|---|---|---|---|
| 024 adjective reserves | 23 | 0 | 23 | 20 | 3 (fuzzy, latest, earliest) |
| 024 ordinary reserves (pairs) | 9 | 1 (statue/statues) | 8 | 8 | 0 |
| frozen new list | 39 | 22 (19 multi-token; mirror, planet, rocket as pool target-noun forms) | 17 | 12 | 5 (robot, shark, singer, snake, whale) |

## Checks

### 1. File integrity: PASS

- **Digests.**
  - The sha256 of the file is `54c8947c…b64a`. It equals the launcher's `expected_file_sha256` and `post_write.file_sha256`.
  - The bytes are exactly `canonical_json(payload) + "\n"`, so there are no duplicate keys and no formatting freedom.
    The file has no CR, and ends in a single LF.
  - `content_sha256` equals sha256(canonical JSON of the payload without `content_sha256`), with my implementation.
    `cr.confirmation_from_payload(verify=True)`, which checks `rc.content_digest`, also passes.
  - `manifest_sha256` recomputes, and equals the digest of my own manifest.
- **Identity.**
  - `experiment` is "025" and `schema_version` is 1.
  - `model` is `{EleutherAI/pythia-70m-deduped, e93a9faa…}`, which is `models.PYTHIA_70M`.
- **Configuration.** It equals the configuration I built from the design and plan text, field by field, with an empty
  diff, and it also equals `cr.PRODUCTION.to_json()`:
  - `count_threshold` 27, derived as the smallest k with P(Bin(40, ½) ≥ k) ≤ 1/40;
  - `reference_tail` exact `5288280983/274877906944`, value `0.01923865414210013`;
  - the conditions `base, noun±0.32, noun±0.16, rand1…7±0.32, plur±0.32`, in the plan's frozen order;
  - 16 outcome-bearing conditions (`noun±0.32`, `rand1…7±0.32`);
  - `n_frames` 108, `n_scored_nouns` 79, `k_controls` 7, doses 0.32 and 0.16.
- **Design and plan.**
  - `design` is `{c0885e5, correction 26c9925, revision 1}`, and `plan` is `{7d90d28, revision 1}`.
  - Those commits exist and are exactly the commits that created (c0885e5, 7d90d28) and corrected (26c9925) the two
    files.
  - `git diff 26c9925 HEAD` on the design and `git diff 7d90d28 HEAD` on the plan are both empty.
  - The correction changed only the plurality-alignment figures, not the population, the rules or the conditions.
- **Other fields.**
  - `sources.freeze_024` names 024's committed freeze. Its file sha256 is `68510e1b…` (the pin) and its content
    `87f8aff1…` recomputes.
  - `sources.prior_nouns_020` names 020's committed file, with sha256 `de4ebb7b…` (the pin), and its 24 noun keys.
  - `candidates` equals 024's `reserves.N`, 024's `reserves.ordinary` and the design's 39-word list, which is in
    alphabetical order.
  - `exposed_frame_ids` equals the pool's order, 024's committed list and 020's committed list.
  - `reference_cue_ids` is {cardinal 581, coordinated-adjective 581, quantifier 1016}, which the pool, 020 and 024 all
    give.
  - `counts` is {strata 20/20, frames 108, conditions 21, runs 90,720}.
  - `picks`, `expected_picks` and `picks_match_expected: true` match the design's table.
  - `rules` equals `cr.FREEZE_RULES`, and its wording states the design's rules.

### 2. The four exclusion sets: PASS

Each set was recomputed from its source and equals the file's `blocked` set exactly. Each set's `sha256` field also
recomputes.

| set | source | size | cross-check |
|---|---|---|---|
| earlier cues | 024's committed freeze `exclusion.cue_token_ids` (351) ∪ 024's 40 cue ids | 391 | the two parts are disjoint; every cue id in every spent key, 020's 279 exposed cue ids, 020's, 022's and 023's fresh cues, and the frames' own cue ids all lie in it |
| pool target-noun forms | every `sg_ids` and `pl_ids` id of the 80 pool nouns (`ul.load_frozen_inputs`) | 161 | 79 single-token nouns × 2, plus peach's sub-token ids 607, 759, 3844. It equals 024's committed `target_noun_form_ids`, and the pool's noun keys equal 020's committed `exposed_noun_keys`, in order |
| prior-noun forms | every sg and pl id of the 24 nouns of 020's committed `confirmation-v1.json` | 48 | statue 23957/38490 and barrel 15474/33545 are included, and neither is in any other set |
| exposed-frame tokens | every prefix and suffix id of the 108 exposed frames | 316 | equal to 024's committed `frame_token_ids`; the frames' token sequences match all 2,592 of 020's committed `exposed_frame_prompts` |

### 3. Mechanical selection: PASS

**The walk.** My walk covers:
- 024's 23 adjective reserves;
- then its 9 ordinary pairs, whose plurals are given;
- then the design's 39-word list, each with its standard regular plural (+s for all 39).

**The rules applied:**
- `" " + word` is one token under both tokenizer routes;
- the id is in no exclusion set;
- the id is not already picked;
- a noun needs both forms eligible and distinct.

**The picks.** The first 20 adjectives and the first 20 nouns equal the file's `cues` exactly (a whole-object equality
of every field), and they equal the design's 40, in order.

**The rejections.** There are 23 rejections, in the same order. Every reason was parsed and checked against my own
derivation:
- the token counts;
- the ids;
- the named exclusion set, which is the first match in the order earlier → target → prior → frame;
- "eligible" for the other form.

Every blocked form matched exactly one set, so the order in which reasons are named has no effect here.
- **statue/statues** (ordinary rank 8) is rejected only by the prior-noun rule: "a form of a noun of 020's confirmation
  list (read by 021)".
- **mirror, planet and rocket** are rejected as pool target-noun forms.
- The other 19 new-list rejections are multi-token: baker, camel, carpet, clerk, dolphin, donkey, eagle, frog, hammer,
  ladder, owl, parrot, pencil, pillow, queen, tiger, tractor, violin, wizard.

**Literal constants.** The module's literal lists (`ADJECTIVE_RESERVES`, `ORDINARY_RESERVES`, `NEW_NOUN_LIST`,
`EXPECTED_PICKS`) equal their committed sources and the design text, and `cr.select_cues` and `cr.blocked_sets` agree
with me.

### 4. Manifest and isolation: PASS

**The manifest.**
- My 40 × 108 × 21 keys `frame_id|word|token_id|condition`, sorted, equal the file's list element for element.
- The 90,720 keys are unique. Each has 4 fields, and its tag is one of the 21 frozen conditions.
- The untagged keys are exactly the 4,320 cue × frame prompts.

**The spent set, recomputed from its sources.** It has 43,632 keys, as the plan says, in five parts, all keys with 3
fields:

| part | keys | how it was checked |
|---|---|---|
| 020's ledger | 30,132 | `outputs/experiment-020/results.json`. Its file sha256 `da63b8c2…` equals the committed closure's, and the digest of the sorted key list equals the committed extract's `executed_prompt_keys_sha256`, `8258198b…` |
| 020's confirmation set | 3,060 | S1-REF, S1-VALIDITY and S2-TARGET; S2-TARGET equals the file's two prompt lists |
| 022's manifest | 3,060 | |
| 023's manifest | 3,060 | |
| 024's manifest | 4,320 | |

- The parts are pairwise disjoint.
- It equals `Runner()._base().forbidden` exactly: the two sha256 digests are equal.

**No collision.**
- No untagged key and no tagged key is in the spent set.
- A stronger check also finds nothing: no (frame_id, cue id) pair of 025 is spent, and none of the 40 cue ids or 20
  plural ids ever appears as the cue of a spent key.
- `cr.assert_ledger_isolated` passes.

**The patch-path keys.** The four `cr.PATCH_PATH_SPENT_KEYS` equal the plan's list, and each is in 020's ledger. None
is a manifest key, tagged or untagged.

### 5. The launcher's evidence: PASS (see N1, N2 and N4)

**`freeze_run.json` and `launch.log` against the requirements:**
- `freeze_invocations` 1 and `exit_status` 0, with `error` null;
- `module_calls` 0, `load_model` 0, `capture_calls` 0;
- `score_or_geometry_calls` empty;
- `forbidden_reads` and `forbidden_writes` empty;
- `confirmation_writes` 1: `Path.write_text` raises exactly one `open` audit event with mode `w`, which I measured;
- all 18 `pre_write_checks` true;
- `post_write.bytes_equal_pre_verified_payload` true, with file sha256 `54c8947c…`, the current file.

The runner's own log line in `launch.log` is printed only after its re-read verification succeeds. Every `facts` field
equals my independent values: the cues, the rejections, the exclusion sizes, the per-list counts, the forbidden size,
the manifest, content and file digests.

The record's `exit_status` is `runner.freeze()`'s return value. The process's own exit code is not logged, but the code
fixes it: with `status == 0`, no error and equal bytes, the launcher exits 0 (`launch_freeze.py:227-228`).

**The dry run (`dryrun_run.json`).** The record shows exactly what the code says:
- `dry_run` is true, with `freeze_invocations` 0 and `confirmation_writes` 0;
- the other counters are 0 and the lists empty;
- `post_write` is `{dry_run: true, confirmation_exists: false}`;
- `logs` is empty, so there is no freeze log line;
- its 18 checks and all its `facts` are identical to the production run's.

In the code (`launch_freeze.py:209-211`), the `DRY_RUN` branch only sets `status = 0`, and the single `runner.freeze()`
call site (line 214, confirmed by `ast`) is in the `else` branch.

**Timeline and file state:**
- the launcher's mtime is 15:56:13Z;
- the dry run ran 15:56:14–15:56:25Z;
- `launch.log` was created at 15:56:48Z;
- the production run ran 15:56:49–15:57:07.47Z;
- the artifact's birth time and mtime are both 15:57:07Z;
- the launcher (sha256 `69733620…3b94`) has not been modified since before either run.

**The guards were installed before the runner was imported.** `ast` gives the order:
- `sys.addaudithook(_hook)` at line 64;
- `import torch` at 66;
- the project imports at 68–72;
- the patches at 98–108;
- the runner's spec and exec at 110–113;
- `assert runner_module.load_model is _refuse_load` at 114. The run went past it, so it held.

**The guards would have caught a forward, a model load or a score.** `guard_test.py` executed only the launcher's prefix
(lines 1–109) and then tried each route:
- **A model load.** `models.load_model` and the runner's default `model_loader` both raise and are counted.
- **A forward.** A `torch.nn.Identity` forward raises (`module_calls`). TransformerBridge does not override `__call__`.
- **A capture.** `pm.run_patched` and `pm.capture_prompt` raise.
- **A score or geometry quantity.**
  - `rr.nounness`, `rr.centroid`, `rr.full_scores`, `cr.score` and `cr.geometry_block` raise.
  - So do the internal calls: the original, unpatched `cr.direction` reaches the patched `cr.centroids`, and the
    original `cr.centroids` reaches the patched `rr.centroid`, because module globals are looked up at call time.
  - Every name in the launcher's lists exists, so no `hasattr` skipped anything.
  - No module imports a score or geometry name by value.
  - Only `run.py` imports `load_model` by value, and it binds it after the patch.
  - The capture guard patches `pm`'s attributes. The originals `capture.run_capture` and
    `interventions.run_interventions` stay unpatched; `candidate_screening` also holds `run_capture` by value. Any
    capture needs a model, though, and both the load and the forward are refused.
- **The hook.** It refuses opens of `outputs/experiment-023/…`, `outputs/experiment-024/…` and `calibration-table.pt`,
  and writes under the repository by mode or by `os.open` flags. It allows and counts the confirmation write, and
  allows reads.

### 6. Repository state: PASS

- **Refs.**
  - HEAD = `origin/main` = the remote's `HEAD` and `refs/heads/main` (`git ls-remote`) = `c765148fdb612ae249e21ef3c830e4ceb1d12dd0`.
  - The reflog shows no HEAD movement after the 18:26:28+0300 commit. There is no stash, and the only other worktree
    is an unrelated Sep 16 branch.
- **Working tree.**
  - `git status --porcelain -uall` shows only `?? experiments/025-nounness-direction-intervention/confirmation-v1.json`.
  - In the 025 directory, only `README.md` and `run.py` are tracked. There is no ignored file and no `evidence/`.
  - `preregistration-lock.json` and `preregistration.md` are absent.
- **Outputs.** `outputs/experiment-025/` does not exist, so there is no results state, ledger or candidate lock.
- **The code at freeze time was HEAD.** No file in the repository (outside `.git` and `.venv`) is newer than the
  launcher's last modification, except the artifact and its directory. So no tracked file was touched around the
  freeze, and no bytecode was written.
- **Committing byte for byte.** There is no `.gitattributes`, no `core.autocrlf` and no active hook, so the artifact
  can be committed byte-identically.

### 7. Leakage: PASS (see N3)

**The code path of `Runner.freeze`** (`run.py:240-261`):
1. `_base()` (`run.py:180-193`):
   - `cr.assert_frozen_blobs()`;
   - `ul.load_frozen_inputs`;
   - `b0c.verify_022_inputs`, `rr.verify_023_inputs`, `cr.verify_024_inputs`, `cr.load_024` and `cr.prior_nouns`;
   - the 022 and 023 confirmation JSON;
   - the `forbidden` union.
2. It refuses to run if the file exists.
3. It loads the tokenizer.
4. `pytest_free_guard`, which refuses pm's capture entry points, wraps:
5. `cr.freeze_payload` (`cue_rotation.py:772-796`):
   - `blocked_sets`, which uses `rr.target_noun_form_ids` and `rr.frame_token_ids`;
   - `select_cues`, `_status` and `pm._encode`, on literal lists;
   - `confirmation_from_payload(verify=False)`;
   - `Confirmation025.counts` and `manifest` (`pm.Prompt.key` × conditions);
   - `canonical_json`, `sha256_text` and `rc.content_digest`.
6. `confirmation_from_payload(verify=True)`.
7. It refuses if any untagged key is spent.
8. It writes the file.
9. `load_confirmation_025` re-reads and verifies it.

**Dynamic confirmation.** I replayed `_base()` and `cr.freeze_payload` under a profiler, with a broader set of refusals
than the launcher: every geometry and gate entry point of `cr`, plus `rr.cosine`, `rr.embedding_digest` and
`rr.parameters_digest`.
- **Nothing was refused**, and no guard was hit.
- **`freeze_payload`** called only:
  - the functions listed in step 5, plus the config properties and `binomial_tail`;
  - `pm._encode`, 119 times: 23 + 2 × 48;
  - `pm.Prompt`, 4,320 times;
  - `.key`, 90,720 times.
- **Files.** `freeze_payload` opened no file.
- **`_base()`** called only loaders, validators and digest functions: 159 distinct functions, listed in
  `crosscheck_module.out.json`. None of them touches weights, a model, a capture, a score or a geometry quantity.

**No 025 score exists in anything the freeze reads.** 024's committed calibration record and lock (which `_base()`
loads) contain none of the 40 cue ids or plural ids. The only candidate words that appear in them are
mirror/planet/rocket, as pool target-noun keys.

## Findings (most severe first)

There are no blockers and nothing to fix in the artifact. All eight findings are notes.

**N1. The run record omits provenance.** `launch_freeze.py:221-223` builds the record from timestamps, counters, checks
and facts only. It does not record:
- the git HEAD and the dirty flag;
- the interpreter path;
- `HF_HUB_OFFLINE` and `PYTHONDONTWRITEBYTECODE`;
- the library versions;
- the launcher's own sha256.

*Scenario:* a run from a dirty tree, or from another interpreter, would leave an identical-looking record.
- For this freeze the gap is closed by other evidence:
  - the reflog (HEAD at `c765148` since 18:26:28+0300);
  - the file mtimes (nothing touched after the launcher's last edit except the artifact);
  - the clean tree;
  - the determinism of the output (reproduced byte for byte).
- *Recommendation:* the lock and confirm launchers should record these fields.

**N2. "Exactly once" can be shown only indirectly.** Each run overwrites its record (`launch_freeze.py:223`), so no
append-only log proves that there was one dry run and one production run. The following mitigate it:
- `Runner.freeze` refuses an existing file (`run.py:243-244`);
- the artifact's birth time equals its mtime, 15:57:07Z, inside the production window;
- the launcher has a single `freeze()` call site, in the non-dry-run branch;
- the freeze has no degrees of freedom: any repetition yields the same bytes. `cr.freeze_payload` reproduces the file
  with the runner's tokenizer loader and with a `local_files_only` one, and my independent derivation agrees.

**N3. The freeze reads more than the design's phase table lists.** The table
(`docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md:355`) says: "the
tokenizer, the committed lists, noun sets and frames". `Runner._base()` (`run.py:180-193`) verifies the whole
inherited chain. It opens 29 files, including:
- the gitignored `outputs/experiment-020/results.json`, whose ledger and locked reference states are hashed by
  `rd.state_digest` inside `rc.verify_020_closure` (`readout_calibration.py:183-226`);
- 023's `exposed-cells.f64`;
- 022's calibration record;
- 024's calibration record and lock, which holds 024's scores.

*Scenario considered:* could any of these values steer the picks? No:
- `freeze_payload` receives only the pool's nouns, frames and reference ids, 024's exclusion ids, cue ids and content
  digest, 020's prior-noun ids, the tokenizer and literal lists;
- the call trace and the byte-identical reproduction confirm this;
- none of those files holds a 025 cue id.

The read of 020's ledger is required by the design's isolation rule, and 021–024's freezes follow the same pattern. No
action is needed beyond saying so in the freeze evidence README.

**N4. The launcher's guards have limits.**
- The audit hook (`launch_freeze.py:42-61`) sees only `open` events, not `os.remove`, `os.rename`, `os.replace` or
  `os.mkdir`.
- The `nn.Module.__call__` patch (`:98`) does not catch a direct `.forward()` call on a module with no submodules.
  Any real model forward goes through submodules, so it would be caught.
- The score guard (`:103-108`) covers named attributes only. For example, `rr.cosine`, `cr.unit`, `cr.rotate` and
  `cr.routing_distance` are unpatched.
- The capture guard (`:100-102`) patches only `pm`'s attributes, not the originals `capture.run_capture` and
  `interventions.run_interventions`.
- Weight files are not refused.

*Scenario:* a code path that loaded weights by another route and computed a dot product directly would not be counted.
- For this freeze, no such path exists (see the trace in check 7). My replay, with broader refusals, hit nothing and
  opened no weight file, and the repository-state check shows no other file changed.
- *Recommendation:* for the lock and confirm launchers, also refuse `os.remove`, `os.rename` and `os.replace` under the
  repository. For a freeze-type launcher, also refuse weight-file opens.

**N5. The runner's tokenizer loader does not pass `local_files_only=True`.** `_load_tokenizer` (`run.py:91-94`) relies
on `HF_HUB_OFFLINE=1`, which the record does not show (N1).
- The revision is an immutable commit, so the content cannot differ.
- Offline, the runner's loader and a `local_files_only` loader give the byte-identical payload.
- The raw `tokenizers` route agrees on all 119 encodings.

**N6. The brief names the wrong model id.** It says the tokenizer is "EleutherAI/pythia-70m". The pinned spec
(`src/neural_decompiler/models.py:48-51`) and the artifact's `model` field are `EleutherAI/pythia-70m-deduped` @
`e93a9faa…`, which is also the only Pythia-70M snapshot in the local cache. I used the pinned spec.

**N7. Housekeeping for the commit.**
- `experiments/025-nounness-direction-intervention/README.md:3-4` still says "no scientific phase has run; the next
  step is the production `freeze`". Update it in the freeze commit; the README is a non-scientific path
  (`cue_rotation.py:940`).
- Commit the artifact byte-identically (sha256 `54c8947c…`).
- As 024 did (`evidence/freeze-2026-09-25/`, `evidence/freeze-review-2026-09-25/`), commit the following under
  `experiments/025-nounness-direction-intervention/evidence/`, which is a non-scientific prefix:
  - the launcher, `launch.log`, `freeze_run.json` and `dryrun_run.json`;
  - this review and its scripts and outputs.

**N8. The frame-token rule's wording is broader than its implementation.** `rr.frame_token_ids`
(`readout_routing.py:817-818`) takes the frames' prefix and suffix ids only. The rule text says "occurs nowhere in the
108 exposed frames". It makes no difference here:
- every frame's own `cue_ids` (sg/pl) is in the earlier-cue set;
- no picked id is in any frame's cue ids, prefix or suffix.

## Not verified (outside the mechanical scope)

- **The design-time judgment criteria**, which the design review approved:
  - the semantic criteria (no adjective or modal use, not a container or measure);
  - the provenance of the 39-word list.
- **The completeness of 024's committed exclusion over 005–023.** The design takes it as the definition. It is
  spot-checked above: every spent key's cue id, and 020's, 022's and 023's cues, lie in it.
- **The tier-A/B/C tests**, which were not audited. `gate025/` shows them green at `c765148` shortly before the freeze.

## Files (in this directory)

- `derive_independent.py` and `derive_independent.out.json`: the independent derivation. It does not import `cr`, and
  its audit hook refuses writes, forbidden paths and weights.
- `crosscheck_module.py` and `crosscheck_module.out.json`: the module and runner cross-check, with the call trace and
  the files each step opened.
- `guard_test.py` and `guard_test.out.json`: the launcher's statement order and its guard behaviour.
- `repo_state.log`: the git, remote, status, reflog, timestamps and digests.
