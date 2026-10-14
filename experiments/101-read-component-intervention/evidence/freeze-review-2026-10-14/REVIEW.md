# Experiment 101: review of the production `freeze`

**AI review (independent, read-only), not human validation.** The reviewer is an AI agent (Claude). It worked
read-only on 2026-10-14 against `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler`, using only the tokenizer
and committed files. It loaded no model and opened no weight file. It ran no forward pass, capture or intervention.
It computed no read covector, geometry, prediction, measurement or statistic, and it ran none of the runner's phases.
Under `outputs/` it read only `outputs/experiment-020/results.json`, for 020's ledger. It wrote only inside
`.../scratchpad/exp101/freeze-review/`.

## Verdict: **PASS WITH NOTES**

The artifact `experiments/101-read-component-intervention/confirmation-v1.json` is correct in every field checked.
- An independent re-derivation, which imports no `neural_decompiler` code, reproduces the 60 eligibility outcomes, the
  36 frames, every exclusion count and digest, both manifests and their digest, and the content digest.
- A cross-check run afterwards through `ri.freeze_payload` regenerates the file byte for byte.
- The launcher's records match the design, the plan and the repository state.
- There is no blocker. One should-fix concerns preserving the freeze evidence, and the notes concern provenance and the
  hardening of later launchers.

## Key numbers

| item | value |
|---|---|
| file sha256 / bytes | `6da26cd7d137510bef384af3a581e857244376c0b5825f071c069ebaabc10922` / 135,675 |
| `content_sha256` | `ef35650678e98a638838af10b75f340cec331f542589d6f089100eee8822624a` (recomputes) |
| `manifest_sha256` | `7c7c94942dc883505b97403042e095d8d616168f3a0a1d8cf495edef25f8b8bd` (recomputes; equals my independent manifest) |
| per-list digests (mine, not recorded in the file) | S1-REF `ffcfcbd7…2aab`, S2-TARGET `d2da8653…91fe8`, untagged prompts `6c09174c…a885` |
| picks | candidates 1–36 = `EXPECTED_PICKS`; all 60 candidates are eligible; no replacement |
| `p_c` of the picks | 4: 10 frames, 5: 22, 6: 4, 7: 0 |
| manifests | S1-REF 36; S2-TARGET 1,980 distinct (55 per frame); disjoint; 2,016 keys in all; 144 untagged prompts |
| exclusion sets | frame texts 162 (144 + 18); earlier cue ids 431 (351 → 391 → 431); 025 words 80, with 76 single-token ids; target-noun forms 161; 020 noun forms 48; number words 42; `frame_tokens` 316, not applied |
| spent set | 47,952 keys = 025's 43,632 + 4,320 untagged 025 prompts; 0 collisions (tagged 0 of 2,016, untagged 0 of 144) |
| repository during the review | HEAD `40829d21…` on `exp101-phases`; only `?? …/confirmation-v1.json`; no `outputs/experiment-101/`; no lock, calibration or stage-1 file; 6 ahead and 0 behind `origin/main` (`96f0cc8`, as of the last fetch on 2026-10-05) |
| after the review | install commit `a857f9a` (11:40:09): parent `40829d2`, adds only this file, and the committed bytes are `6da26cd7…`; see the addendum |

The 36 frozen frames are listed with each pick's `p_c` and adjective token id. Every `p_t` is `p_c + 1`; prefix and
suffix ids are in the file and were re-derived.

| # | text | p_c | adj (id) | # | text | p_c | adj (id) |
|---|---|---|---|---|---|---|---|
| 1 | Aria and Ben carried {cue} large | 5 | large (1781) | 19 | Mia and Ned planted {cue} pale | 5 | pale (14530) |
| 2 | Cora and Dan folded {cue} soft | 5 | soft (2602) | 20 | Olga and Pete picked {cue} neat | 5 | neat (18176) |
| 3 | Eli and Faye stacked {cue} gray | 6 | gray (11978) | 21 | Rita and Seth hugged {cue} tired | 5 | tired (11870) |
| 4 | Gus and Hope washed {cue} sticky | 5 | sticky (31714) | 22 | Tess and Uri labeled {cue} dry | 5 | dry (6079) |
| 5 | Ian and June sorted {cue} tiny | 5 | tiny (10058) | 23 | Val and Wes measured {cue} cool | 4 | cool (4484) |
| 6 | Kit and Lou counted {cue} clean | 4 | clean (4076) | 24 | Ada and Bo watched {cue} happy | 4 | happy (5211) |
| 7 | Max and Nell wrapped {cue} fine | 4 | fine (4030) | 25 | Cleo and Dev mended {cue} long | 6 | long (1048) |
| 8 | Oscar and Paula dried {cue} damp | 5 | damp (16109) | 26 | Gil and Jo sewed {cue} cozy | 5 | cozy (47400) |
| 9 | Quinn and Rex cleaned {cue} messy | 5 | messy (36396) | 27 | Kim and Lia carved {cue} dull | 5 | dull (22544) |
| 10 | Sara and Tom fixed {cue} bent | 5 | bent (16535) | 28 | Mo and Nia built {cue} tidy | 5 | tidy (49031) |
| 11 | Uma and Vera moved {cue} full | 5 | full (2120) | 29 | Pia and Ray painted {cue} open | 5 | open (1527) |
| 12 | Walt and Yara lifted {cue} wet | 6 | wet (9685) | 30 | Sol and Ty repaired {cue} dead | 4 | dead (3846) |
| 13 | Zack and Abby filled {cue} deep | 6 | deep (3676) | 31 | Ann and Bill tied {cue} loose | 4 | loose (13155) |
| 14 | Beth and Carl peeled {cue} bad | 5 | bad (3076) | 32 | Chris and Di pulled {cue} broad | 4 | broad (3862) |
| 15 | Dora and Erik sliced {cue} thin | 5 | thin (6906) | 33 | Ed and Fran pushed {cue} slim | 4 | slim (27973) |
| 16 | Gina and Hank cooked {cue} sour | 5 | sour (18988) | 34 | Greg and Holly sent {cue} sleek | 4 | sleek (47115) |
| 17 | Ivy and Jack grilled {cue} lean | 5 | lean (9644) | 35 | Jim and Kara brought {cue} bold | 5 | bold (13433) |
| 18 | Kate and Luke served {cue} good | 4 | good (1175) | 36 | Liz and Mark stored {cue} sealed | 5 | sealed (18495) |

Frame ids are `coordinated-adjective-101-01` … `-36`. Every frame has `cue_ids {sg: 581, pl: 767}` and template
`coordinated-adjective`.

## Method and order of operations

The work ran in this order; the file times in the review directory record it.
1. **Reading.** I read the brief, the launcher, its records and logs, design revision 3 and plan revision 3. I looked
   at the file's structure and at the committed 020/022/023/024/025 confirmation files. For the digest format I read
   `pm.canonical_json`, `pm.sha256_text` and `rc.content_digest`. Before deriving anything I had also seen `ri`'s
   constant block (`FRAME_CANDIDATES`, `NUMBER_WORDS`) and the runner's `_base()` forbidden-set composition. I had not
   read `ri.freeze_payload`, `ri.candidate_status` or `ri.exclusions_from_committed`.
2. **The independent derivation (11:20–11:23).** None of these scripts imports `neural_decompiler`. That holds by
   construction, and `independent_derivation.py` also prints the loaded `neural_decompiler` modules, which are none.
   - `independent_derivation.py` parses the 60 candidates from design §11 as committed at `516df20`; the design has
     not changed since. It uses both tokenizers, the committed files read through `git show HEAD:` (each equal to the
     working copy) and the plan's number-word list.
   - `compare_with_file.py` compares that derivation with the file, field by field.
   - `spent_set.py` rebuilds the spent set from the committed manifests and 020's ledger.
3. **The cross-check, only then (11:24 onward).** I read `ri`'s freeze code (`read_intervention.py:700–960`) and ran
   `cross_check_ri.py`: `Runner()._base()`, which calls `ri.exclusions_from_committed`; then `ri.freeze_payload`,
   `ri.confirmation_from_payload(verify=True)` and `ri.load_confirmation_101`. Three guards were active: an audit hook
   refusing repository writes and weight-file opens, a refusing `torch.nn.Module.__call__`, and a refusing
   `models.load_model` and HF `from_pretrained`.
4. **Supplements.** `tokenizer_integrity.py` checks tokenizer integrity, and `cross_check_pool.py` recomputes the
   161 target-noun forms and the 316 frame tokens from 020's pool.

**Tokenizers.** I used `AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision="e93a9faa…",
local_files_only=True)` (transformers 5.17.0) and the raw `tokenizers` 0.23.2 library on the snapshot's
`tokenizer.json` (sha256 `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624`). They gave identical ids
on every encoding: 0 disagreements. The raw tokenizer also reproduces 3,132 committed token sequences from 020, 022
and 023 (frames under both cues, plus 020's 3,024 target prompts) with 0 mismatches. The tokenizer is therefore the one
the earlier freezes used.

## Checks

### 1. The file's integrity: PASS
- **Bytes and digests.**
  - sha256 `6da26cd7…0922` and 135,675 bytes, equal to the launcher's `expected_file_sha256`/`expected_bytes` and its
    `post_write`.
  - The bytes are exactly `canonical_json(payload) + "\n"`: one trailing newline, and duplicate keys are impossible.
  - `content_sha256` equals the sha256 of the canonical payload without that key, which is `rc.content_digest`'s
    rule; `ri.confirmation_from_payload(verify=True)` also passes. `manifest_sha256` recomputes.
- **Identity.** `experiment` "101", `schema_version` 1. The model is `EleutherAI/pythia-70m-deduped` at `e93a9faa…`.
- **Configuration.** It equals `ri.PRODUCTION.to_json()`:
  - Holm `[26, 25]`, `d_threshold` 25, `p_threshold` 27, `precondition` 30, `tau_rank` 33;
  - `runs_per_frame` 55, `k_iso` 8, `n_scored_nouns` 79, `exposed_frames` 36, `max_overshoot_frames` 6 and
    `max_weak_frames` 6;
  - reference `["ref", 581]` and cues `two 767 / three 1264 / many 1142`;
  - `frame_candidates` equal to the design §11 text, and `tags` equal to my derivation of the plan's arm order.

  I recomputed the Holm thresholds as exact rationals without `ri`: P(Bin(36,½) ≥ 26) = 0.005665, ≥ 25 = 0.014408 and
  ≥ 24 = 0.032623, giving 26 and 25. τ's rank is ⌈0.9·36⌉ = 33.
- **Design and plan.** `design` = {revision 3, `a87e284`, decision `516df20`} and `plan` = {revision 3, `96f0cc8`,
  decision `1e48c1c`}. Both files are unchanged since those commits: the last commits touching them are `516df20` and
  `1e48c1c`.
- **Pinned blobs.** All 20 pinned module blobs are identical at HEAD, at `516df20`, in the working tree, in
  `ri.FROZEN_BLOBS` and in the plan's table. The only `src/` change since `516df20` is the new `read_intervention.py`.

### 2. The exclusion sources: PASS

Every set was recomputed from the committed files and compared with the file's `exclusions` record by count and by
sha256 over the sorted canonical list.

| set | size | digest (file = mine) | derivation |
|---|---|---|---|
| frame texts | 162 | `5c8a47a4…19f3d` | 023's `exclusion.frame_texts` (144, whose own `frame_texts_sha256` recomputes) ∪ 023's 18 frame templates; no overlap |
| earlier cue ids | 431 | `adf6ad55…38eb0ce` | 024's `exclusion.cue_token_ids` (351) ∪ 024's 40 cues (→ 391) ∪ 025's 40 cues |
| 025 candidate words | 80 | `e8caa859…3fadf4` | adjective reserve 23 + new list 39 + ordinary reserve 9 × 2 |
| their single-token ids | 76 | `0bff17e0…abb2f7` | the four two-token words are baker, dolphin, owl and parrot |
| target-noun forms | 161 | `e0f8ea0a…aa0d8c7` | 024's `target_noun_form_ids`, equal to 025's `blocked.target_forms` and to `rr.target_noun_form_ids(pool)` (80 nouns) |
| 020 confirmation-noun forms | 48 | `4a1e2268…e00295` | 24 nouns' `sg_ids` ∪ `pl_ids` recomputed from 020's committed file (sha `de4ebb7b…`); equals 025's `blocked.prior_nouns` |
| `NUMBER_WORDS` | 42 | (list equal) | the plan's enumeration |

- **Cross-checks of the cue chain.**
  - 024's exclusion equals 023's exclusion (327) plus 023's cues (24).
  - 023's exclusion contains 022's exclusion and 022's cues.
  - 022's exclusion contains 020's 24 confirmation cues.
  - 025's `blocked.earlier_cues` (391) equals 024's exclusion plus 024's cues.
- **Source digests.** The recorded `content_sha256` of 023, 024 and 025 and the file sha256 of 020 all recompute.
- **Broader novelty, beyond the rule.**
  - None of the 60 texts matches, even case-insensitively, any of 183 texts: the rule's 162 plus 022's exclusion and
    frames, 020's frames, and the unpicked 022 and 023 candidate lists.
  - `git grep` of each candidate's full prefix and of its name pair (for example "Aria and Ben") over every tracked
    file finds no hit outside 101's own design, plan, module and tests.
  - The search itself was validated: "Hugo and Nina" hits 022 and 023.
  - A wider reading of "025's eligible-cue list" also hits no adjective of the 60. It adds the new list's regular
    plurals and the 025 cues' plural ids (119 words and 96 ids).

### 3. The mechanical selection: PASS

**The walk.** For each of the 60 candidates in design order, I placed ` one`, ` two`, ` three` and ` many` (581, 767,
1264, 1142, each verified as a single token) in the slot. Each cue occurs exactly once, at the same `p_c` in all four
prompts. The four prompts share the prefix and suffix, and the whole-text tokenization equals the piecewise one. The
suffix is exactly one token, equal to `" " + ADJ`, so `p_t = p_c + 1`. Then the rules:
- the text is not an earlier frame text;
- the adjective is not an earlier cue;
- it is not a 025 candidate word, by word or by single-token id;
- it is not a target-noun form, from the pool or from 020's nouns;
- it is not a number word;
- it is not already taken.

**Result.** All 60 candidates are eligible, with `p_c` 4–7 (only #60 has 7). The first 36 eligible are candidates
1–36, exactly `EXPECTED_PICKS`, with no replacement.

**Against the file.** All 60 `candidates` entries match my derivation in `text`, `frame_id`, `adjective`,
`adjective_id`, `p_c`, `p_t`, `eligible` and `reasons` (all empty): 0 differences. All 36 `frames` match in `number`,
`frame_id`, `text_template`, `prefix_ids`, `suffix_ids`, `p_c`, `p_t`, `cue_ids` and `template_id`. Each frame's
`prefix + cue + suffix` equals my tokenization for all four cues: 0 differences.

**`frame_tokens` is deliberately not applied** (plan SA3; the file's `rules.not_applied`). Applied, 025's 316-token
set would block 12 picks: #2 soft, #5 tiny, #6 clean, #8 damp, #12 wet, #15 thin, #16 sour, #19 pale, #20 neat,
#22 dry, #23 cool and #31 loose. It blocks no candidate from 37 to 60. Applying it would have shifted the selection,
a `FreezeDeviation`. Design §11 allows such adjectives, since novelty is at the level of the text.

### 4. The manifests: PASS
- **The keys.** From the plan's `CUE_ORDER` and arm orders, I built S1-REF: 36 keys `frame_id|ref|581|S1`. S2-TARGET
  holds, per frame:
  - `REF` on `ref|581`;
  - 32 `ARM@two`, with the reference-prompt arms on `ref|581` (`SUF-*`, `SUF-ISO-j`, `PC-SWAP-REF`, `HEAD-SUF`,
    `HEAD-SUF-V`) and the rest on `two|767`;
  - 11 `ARM@three` and 11 `ARM@many`, with `SUF-*` and `PC-SWAP-REF` on `ref|581`.
- **The counts.** 1,980 distinct S2 keys, 55 in each of 36 frames; S1 and S2 are disjoint.
- **Against the file.** My sorted lists are identical to the file's `manifest`, and my digest equals
  `manifest_sha256`. `counts` = {frames 36, runs_per_frame 55, S1-REF 36, S2-TARGET 1980, untagged_prompts 144}.
  `run_order` and `configuration.tags` equal my 55-tag order. No key places an arm on the wrong prompt.
- **Untagged prompt keys:** 144, which is 36 × {ref|581, two|767, three|1264, many|1142}.
- **The spent set,** rebuilt independently as 025's forbidden set plus the untagged forms of 025's manifest:

  | component | keys |
  |---|---|
  | 020's ledger, from `outputs/experiment-020/results.json` (file sha `da63b8c2…`; the state digest `2e5485dc…` verifies) | 30,132 |
  | 020's confirmation prompts | 3,060 |
  | 022's manifest | 3,060 |
  | 023's manifest | 3,060 |
  | 024's manifest | 4,320 |
  | untagged forms of 025's 90,720 tagged keys | 4,320 |
  | **total (pairwise disjoint)** | **47,952** |

  The first five components make up 025's own set of 43,632. The total equals the runner's `base.forbidden`: the
  sorted-canonical sha256 is identical, `fc0cbc69…015e3`.
- **Collisions: 0.** No tagged key (of 2,016) and no untagged key (of 144) is in the spent set. No spent key, and none
  of 025's tagged keys, has a `-101-` frame id.

### 5. The launcher's evidence: PASS (with notes N1–N3)

**What `freeze_run.json` and `launch.log` record:**
- `freeze_invocations` 1, `exit_status` 0, `error` null;
- `module_calls` 0, `load_model` 0, `model_constructors` [], `capture_calls` [], `refused_functions` [];
- `forbidden_writes` [] and `forbidden_fs_ops` [];
- `confirmation_writes` 1 and `allowed_mkdir` 1 (the `mkdir(exist_ok=True)` of the existing experiment directory);
- `outputs_reads` contains only `outputs/experiment-020/results.json`;
- all 19 pre-write checks are true;
- `post_write.bytes_equal_pre_verified_payload` is true, with file sha `6da26cd7…`;
- the environment shows HEAD `40829d2`, branch `exp101-phases`, a clean status before the run, `HF_HUB_OFFLINE=1`,
  `PYTHONDONTWRITEBYTECODE=1`, torch 2.14.0, transformers 5.17.0 and Python 3.12.13;
- the run lasted 08:14:27–08:14:42 UTC, and the file's mtime, 11:14:42 EEST, matches.

**The dry run.** `dryrun_run.json` shows `freeze_invocations` 0, `confirmation_writes` 0, `allowed_mkdir` 0,
`confirmation_exists: false` after the run, the same 19 checks all true, and identical `facts`.

**Launcher identity.** Both records give `launcher_sha256` `ea4f55b9…4203`, which is the sha256 of the
`launch_freeze.py` on disk. Its mtime (11:05:56) precedes the second dry run (11:06:08).

**The dir_fd account is consistent.** The final launcher resolves `dir_fd` through `F_GETPATH`
(`launch_freeze.py:75–95`). filelock 3.32.6 has an import-time probe (`filelock/_strict.py:60–77`): it creates a
`TemporaryDirectory`, touches `probe-source` and links `probe-link`, and the fd-based `rmtree` then removes exactly two
relative names with a `dir_fd`. Resolved against the working directory, the repository root, they would look like
repository deletions, which accounts for the two reported false positives. Neither the second dry run nor the
production record shows a filesystem refusal. The first dry run's own record no longer exists (see SF1).

**The order of installation.**
1. The audit hook is installed at line 104, before `torch` or `neural_decompiler` is imported (106–116).
2. Lines 137–164 refuse, in turn:
   - `torch.nn.Module.__call__` (137);
   - `models.load_model` (138);
   - `HookedTransformer.from_pretrained*` (139–146);
   - the `from_pretrained` of `transformers.AutoModel`, `AutoModelForCausalLM`, `GPTNeoXForCausalLM` and
     `GPTNeoXModel` (147–152);
   - `pm.capture_prompt`, `pm.run_patched`, `pm.run_capture`, `pm.run_interventions`, `capture.run_capture`,
     `interventions.run_interventions` and `rd.capture_frame_020` (153–157);
   - 23 `ri` geometry, prediction, measurement and outcome functions (158–162);
   - `ul.compose_dx3`, `contrast_of`, `reference_rows_017` and `pair_context` (163–164).
3. Only then is the runner imported (166–169). Line 170 asserts that the runner's `load_model` binding is the refusal.

**Coverage.**
- **A forward.** It would be caught, because every forward of a TransformerBridge or HF model passes through
  `nn.Module.__call__`.
- **A model load.** It would be caught on both paths: through `models.load_model`, and through a raw
  `TransformerBridge.boot_transformers`. The latter resolves GPT-NeoX to `AutoModelForCausalLM`
  (`transformer_lens/model_bridge/sources/transformers.py:438–488`) and calls `model_class.from_pretrained` (line 826),
  which the launcher replaced on that same class object.
- **Geometry, prediction and outcome entry points.** The refusals are setattr's on the module, so they also catch calls
  made from inside `ri`.

### 6. The repository state: PASS

This is the state during the review. The install commit that followed is covered in the addendum.
- HEAD is `40829d21a795a501d9a61c953587b8afcf5f0b86`, on `exp101-phases`.
- `git status --porcelain --untracked-files=all` shows exactly
  `?? experiments/101-read-component-intervention/confirmation-v1.json`.
- `outputs/experiment-101/` does not exist. The experiment directory holds only `README.md`, `run.py` and the
  confirmation file: no `preregistration-lock.json`, `preregistration.md`, `calibration-v1.json` or `stage1/`.
- **No file changed after the freeze.** `find -newermt "2026-10-14 10:55"` over the whole repository, including
  `outputs/` and `.venv/` but excluding `.git`, returns only the experiment directory and the confirmation file. So no
  other repository file was written during or after the freeze, which `git status` alone could not show for ignored
  paths. This holds after all of this review's runs too.
- **Commit hygiene.** There is no `.gitattributes`, no active hook, no `core.hooksPath`, and `core.autocrlf` is unset.
  The file is not ignored.
- **Provenance.** HEAD is 6 commits ahead of `origin/main` and 0 behind (`1e48c1c`, `af06afb`, `439ac3c`, `aec5311`,
  `4795f19`, `40829d2`). The remote-tracking ref dates from the last fetch, on 2026-10-05 15:16, and I did not fetch.
  `exp101-phases` was created from HEAD at 11:00:06 today, and local `main` is at the same commit. **I agree this is
  not a blocker by itself.** The freeze is a deterministic, tokenizer-only function of committed inputs, and I
  regenerated it byte for byte at this HEAD. But see N4.

### 7. Leakage: PASS

**The freeze path.** `Runner.freeze` (`run.py:319–340`) runs as follows.
1. `_base()` (`run.py:205–227`) performs:
   - `ri.assert_frozen_blobs`;
   - `ul.load_frozen_inputs`: the committed confirmations of 006–020, the locks of 011, 012 and 017, 020's closure and
     extract, and `outputs/experiment-020/results.json`;
   - the verification of 022, 023, 024 and 025's inputs;
   - `cr.prior_nouns`, which reads 020's confirmation nouns;
   - the forbidden set, made of keys only;
   - `ri.exclusions_from_committed` (`read_intervention.py:746–771`), which uses texts, ids and words from the
     committed confirmation files only.
2. The tokenizer is loaded (`run.py:91–94`, offline).
3. `ri.freeze_payload` (`read_intervention.py:875–910`) runs inside `pytest_free_guard`, which refuses
   `pm.capture_prompt`, `run_patched`, `run_capture` and `run_interventions`. It calls:
   - `ri.candidate_status` (778–817);
   - `pm._build_new_frame` and `pm._decode`/`_encode` (`plural_mechanism.py:287–322`), which only encode and decode;
   - `pm.Frame.__post_init__` (125–138), which enforces exactly one adjective token for a coordinated frame;
   - `config.to_json()`, the manifest construction and `rc.content_digest`.
4. Finally come `confirmation_from_payload`, the overlap check, one write and a verified re-read
   (`load_confirmation_101`).

**The payload's only inputs** are the tokenizer, the exclusion sets and the configuration constants. No model, weight,
geometry or measurement reaches it.

**The evidence.**
- The launcher's counters show 0 for every model, module, capture and refused-function event.
- My re-derivation imports no `neural_decompiler` code and reads no lock, no `outputs/` file and no calibration. It
  still reproduces every field, and `ri.freeze_payload` regenerates the identical bytes.
- The candidate list itself was revised only on tokenizer and semantic grounds. Design review 1 states "no model was
  loaded" (`2026-10-01-experiment-101-design-reviews.md:25`), and the replacements listed in §13 and the reviews were
  for multi-token, earlier-cue, odd or noun-like candidates.

## Findings (most severe first)

**Blockers: none.**

**SF1 (should-fix, before calibrate): preserve the freeze evidence in the repository.**
- **The risk.** The launcher, both run records, both logs, the pre-install and post-commit check scripts and the
  test-gate logs exist only under `/private/tmp/.../scratchpad/exp101/freeze/`. That location is volatile.
  `launch_freeze.py:3–4` promises a byte-identical copy into
  `experiments/101-read-component-intervention/evidence/freeze-2026-10-14/`. If the temporary directory is cleared
  before that copy, for example by a reboot, the only proof of what was run is lost: "one guarded invocation, zero model
  loads, bytes equal the pre-verified payload". The file itself stays reproducible.
- **A gap already exists.** The first dry run, with the earlier launcher and the two false-positive refusals, left no
  record: the second dry run overwrote `dryrun_run.json` and `dryrun.log`. The account in the brief is therefore not
  backed by evidence on disk.
- **The fix.** Copy the files, verified against these sha256s:

  | file | sha256 |
  |---|---|
  | `launch_freeze.py` | `ea4f55b9139842d0ad280a68b15fecaa8442b2bf58d92234e5b43dbdd1ed4203` |
  | `freeze_run.json` | `9700c202b5a6819d9fc22ea1ed6353e8129ccd81c5d3e1b06468d550c8d25ee2` |
  | `launch.log` | `e73f5eeb7bc777e5052b373b5a5e1141c9e0f11db977d07b64026129d88d5fa0` |
  | `dryrun_run.json` | `b0ad7d4a1bd17e9b4f2ed561ade0a5b8fff34cd613b01b6c2aefa58b42383e4f` |
  | `dryrun.log` | `dde364ed79e23b011b6f99c268d2499b311f8eba89c88cb857765589ef9bb467` |
  | `pre_install_checks.py` | `c45160dc953ef6d9830ed9d3dacf5de97dc854c7141d897d74035b637a776670` |
  | `pre_install_checks.out` | `bc79571fd49f55410ac09f744907262f3413d33430664457309db0745cb7ebe3` |
  | `post_commit_checks.py` | `b299fdde9236840d69fe2f2aac7bedfc91b4b89d5b3501655e0c47047b547b1e` |
  | `test-gate/tierA.log` | `ef04adce…adc4` |
  | `test-gate/tierC.log` | `9b48da98…7c00` |

  Then add a short note recording the first dry run, the defect, the fix and the earlier launcher's hash if it is
  known. Commit the evidence separately from the confirmation file, which must be committed alone.
- **Status at the end of this review.** The operator's draft `scratchpad/exp101/freeze/README.md` now documents the
  first dry run (08:05:03–08:05:13 UTC) and plans the copy with a `SHA256SUMS` file. The first dry run's record itself
  cannot be recovered. SF1 is closed once the copy is committed and its `SHA256SUMS` match the table above. Fix the
  README draft before committing it:
  - **The provenance count.** It says HEAD "was 7 commits ahead of `origin/main`" and calls `1e48c1c` … `40829d2` "the
    six reviewed implementation commits". At the freeze, HEAD was 6 ahead. Of those six, `1e48c1c` is the plan's G4
    decision, a docs commit, and five are implementation commits. Only the install commit makes the count 7.
  - **The placeholders.** `{VERDICT}`, `{INSTALL}` and `{POSTCOMMIT}` must be filled.

**Notes.**
- **N1, launcher hook robustness** (`launch_freeze.py:73–101`). It had no effect on this run but matters if the hook is
  reused for the calibrate, stage-1 or confirm launchers.
  - **Ints read as descriptors.** Line 77 treats every non-negative int argument as a descriptor. Python 3.12's
    audit arguments are `os.mkdir (path, mode, dir_fd)`, `os.chmod (path, mode, dir_fd)` and
    `os.truncate (path, length)`, which I confirmed in a scratch directory. So a mode such as 511 or 493, or a length
    such as 0 (stdin), is taken as a descriptor and passed to `F_GETPATH`. A *relative* path would then resolve against
    an unrelated file, or against `<unresolved dir_fd>`.
  - **`os.rename` and `os.link`.** These use the last descriptor (`dst_dir_fd`) for both source and destination.
  - **The `open` handler.** Lines 52–72 resolve paths opened through `os.open(..., dir_fd=fd)` against the working
    directory.
  - **Why it did not matter here.** The freeze's only repository filesystem event was an absolute-path `mkdir`, and the
    post-hoc `find` shows no other repository write.
  - **For the later launchers.** Map arguments per event (`dir_fd` is the named last slot; `-1` means none). The stage-1
    launcher writes through `ul.write_table`, which calls `mkdir(parents=True, exist_ok=True)`.
- **N2, the refusal list** (`launch_freeze.py:158–160`). `REFUSED_RI` omits pure helpers that could compute outcome-like
  quantities from inputs, among them `ri.main_fractions` (refused by tier C's guard), `ri.predicted_change`, `ri.ratio`
  and `ri.g2_values`. They cannot be reached in the freeze path, and without measurements they compute nothing. Add
  them for the phases that do hold measurements.
- **N3, the launcher's pre-write checks lean partly on the module.** The checks are at `launch_freeze.py:215–316`.
  - They use `ri.FRAME_CANDIDATES`, `ri.NUMBER_WORDS`, `ri.PRODUCTION.tags()`, the runner's `base.forbidden` and its
    tokenizer loader.
  - `exclusion_record_matches_committed_sources` compares counts only, not digests, and omits `candidate_words_025`.
  - The prior nouns are read from 025's record rather than from 020's file.
  - `exposed_frame_token_picks` is a hard-coded list.

  This review closes each gap: the design-text candidates, digests for every set, the prior nouns recomputed from
  020's file and `frame_tokens` derived from 025's committed set all agree.
- **N4, provenance anchoring.** Agreed: not a blocker. The file cites `plan.decision = 1e48c1c`, and the freeze head is
  `40829d2`. Neither is on `origin` as of the last fetch, and this local history was rewritten once already: on
  2026-10-05 17:36:55, `2fe8001`/`feb44b6`/`1cb4c21` became `af06afb`/`439ac3c`/`aec5311`, with identical trees. The
  confirmation file does not record its code commit; `post_commit_checks.py` ties it to the install commit's parent.
  Push or otherwise anchor these commits and the install commit before calibrate, which binds the confirmation file's
  sha256.
- **N5, earlier experiments' model-derived files.** The freeze path's `_base()` reads committed, spent, model-derived
  files of earlier experiments for integrity verification:
  - 011's lock, with `read_weight` and the axes, and the locks of 012 and 017;
  - 020's results state, with the locked reference states, and the exploration extract;
  - 022 and 024's calibrations;
  - 023's `exposed-cells.f64` and lock.

  The file list was observed by my audit hook. None of these files is an input to `freeze_payload`, none concerns a
  101 frame, and the selection reproduces without any of them. The only read under `outputs/` is 020's results, as
  required. The launcher's docstring cites 025's freeze review (its note N3) as accepting integrity reads of 020's
  closure. The other reads are the same kind of input verification, shared by every phase.
- **N6, tokenizer digest.** The confirmation file records the model revision but not the `tokenizer.json` digest
  (`c24618a1…5624`). Integrity was verified here through 3,132 earlier committed token sequences.
- **N7, post-freeze activity.** It does not affect the freeze. At 11:15:22, after the freeze (11:14:42) and before this
  review and the install commit, `scratchpad/exp101/calibrate/smoke_reference.py` ran *unguarded*: it loaded the model
  and executed the S1 re-capture and the 55 arms of the spent exposed frame `coordinated-adjective-1` from 020's pool,
  writing only digests to the scratchpad. No 101 frame and no repository file was touched; the find above shows none.
  The calibrate review should still record it. These 56 runs sit outside any 101 ledger, like the tier-C runs on the
  same spent frames.
- **N8, test-gate logs.** `test-gate/tierA.log` (578 passed) and `tierC.log` (645 passed) carry no HEAD and no time of
  their own. Their mtimes (11:14:22) only show when they were placed. Future gate logs should print the HEAD and a
  timestamp.
- **N9, stale README.** `experiments/101-read-component-intervention/README.md:3` still says "No frame is frozen".
  Update it in a later commit, separate from the install commit.
- **N10, cosmetic.**
  - `frames[].origin` is "extension", the label that `pm._build_new_frame` gives every new frame, as for 022 and 023.
  - Design review 3 says the cues sit at "p_c 4–6", but candidate #60 has `p_c` 7. That is still within 4–7, and it is
    not a pick.
- **N11, this review's own process.** My first `neural_decompiler` import (`cross_check_ri.py`) ran under `python -I`,
  which ignores `PYTHONDONTWRITEBYTECODE`. Python tried three bytecode writes in the repository: two temporary `.pyc`
  files in `src/neural_decompiler/__pycache__/` and `mkdir experiments/101-…/__pycache__`. My audit hook refused all
  three, and the later runs used `-B`. The `find` check confirms that no repository file changed and that `.git/index`
  is untouched (mtime 2026-10-05 19:07:22). Operators combining `-I` with that variable should add `-B`.

## Addendum: the install commit appeared during the review

Check 6 describes the state while this review ran, from about 11:15 to 11:36 EEST: HEAD `40829d2`, with the file
untracked. At 11:40:09, while I was finalizing this report, the operator committed the file.
- **The commit.** `a857f9a2326ff61cae8485deef2ab9337f5d6e02`, "data: experiment 101 freeze — confirmation-v1.json,
  installed byte-identically", on `exp101-phases`.
- **Verified read-only.**
  - Its parent is `40829d21…`.
  - `git diff-tree` shows exactly `A experiments/101-read-component-intervention/confirmation-v1.json`.
  - `git show HEAD:<path> | shasum -a 256` gives `6da26cd7…0922`: the reviewed bytes.
  - `git status --porcelain --untracked-files=all` is now empty, and no other repository file changed (the `find`
    check again).
  - The operator's `post_commit_checks.out` (11:40:32) shows all 13 checks true, with `validate` exiting 0, 47,952
    forbidden keys and nothing refused.
- **N12, process.** The commit message cites "Independent AI freeze review: PASS WITH NOTES, no blocker". That was
  taken from this file before the review was handed back. The verdict and the findings did not change after the first
  write; only digest abbreviations and wording were corrected. The citation is accurate, but a later phase should wait
  for the handback.
- **Provenance after the commit.** HEAD is now 7 commits ahead of `origin/main` as last fetched; N4 applies to `a857f9a`
  too.

## Review artifacts (in this directory)
- `independent_derivation.py` / `.json`: the candidates from the design, both tokenizers, the exclusion sets, the walk
  and the manifests, with no `neural_decompiler` import.
- `compare_with_file.py` / `.json`: the field-by-field comparison with the file.
- `spent_set.py` / `.json`: the spent set and the collisions.
- `cross_check_ri.py` / `.json`: the `ri` and runner cross-check under guards, run after the above.
- `tokenizer_integrity.py`: 3,132 earlier committed sequences.
- `cross_check_pool.py`: the 161 target forms and the 316 frame tokens from 020's pool.
- `ls_tree_src.txt`: `git ls-tree HEAD src/neural_decompiler/`.
