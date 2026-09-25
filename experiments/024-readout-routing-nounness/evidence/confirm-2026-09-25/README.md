# Experiment 024 — the production confirmation (2026-09-25)

**The official run.** The production `confirm` was invoked **exactly once**, at commit
`af160cee0389bcdaf12bcbd92a8616f142777a52`. The guarded launcher `launch_confirm.py --confirm-once` (pid 83432) called
it at 19:31:56.643 UTC; it returned at 19:38:45.531 UTC with **exit status 0** and **no incident**.
- It was never retried, and no other invocation of `confirm` exists.
- The dry run (`--dry-run`, pid 83202) invoked neither `confirm` nor the model.
- Review: independent (PASS WITH NOTES, no blockers; see
  [`../confirmation-review-2026-09-25/REVIEW.md`](../confirmation-review-2026-09-25/REVIEW.md)).

| | |
|---|---|
| **outcome (frozen)** | **`NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`** |
| primary ρ (Spearman, 40 fresh cues) | **`0.6108818011257036`** (= 1628/2665) ≥ the locked `0.3136960600375234`: **PASS** |
| E–N guard | D_EN **`0.5680373703974243`** (= 20465703117234853/2⁵⁵); exact **K = 1** of 12,870 (PASS iff K ≤ 321): **PASS** |
| prompts | **4,320 of 4,320** frozen keys, each executed exactly once; 0 collisions with the 39,312 spent keys |
| stage-2 artifact | `outputs/experiment-024/stage2-measurements.pt`, 76,314,087 bytes, sha256 **`b21babe189d1cccb9ae359cfabf74baaa3a8337cac04612bc0d4aaa232b4dc93`** |
| incidents | **none** |

## The sequence
1. **`preconfirm.py`, `.out`** — the read-only pre-confirm check, with no model load. Every item passed:
   - HEAD, origin and the remote were `af160ce`; the tree was clean, with no stash and no untracked file.
   - The stock `validate` passed. Confirm's own pre-model path passed, replicated without calling it: the full
     `validate_lock` and the runtime check.
   - The phase rule permitted confirm.
   - Both ledgers were empty, there was no stage-2 file, and no outcome existed.
   - The manifest isolation held, and the lock, preregistration and state hashes were preserved.
   - The direct module-blob check agreed, and the environment matched.
2. **`launch_confirm.py`** — the guarded launcher, external to the repository code. Before anything else it creates a
   sentinel exclusively, so the file refuses a second launch in either mode.
   - Its audit hook only records: it never raises, so it could not interrupt the run.
   - **The direct module-blob assertion runs before the model loads** (the reviewer's decision on lock-review note 1).
     It then checks the environment and the preserved hashes. A prompt log wraps `plural_mechanism.capture_prompt`,
     each key written before its call, then delegated unchanged.
3. **The dry run** (`DRY_RUN_LAUNCHED`, `dry_run.out`, `dry_run_record.json`, `dry_run_prompts.jsonl`):
   - it ran every pre-confirm step of the launcher, then stopped;
   - it loaded no model and invoked no confirm; every counter is 0;
   - its prompt log holds one stub probe, `dry-run|probe|0`, which never reached the real entry point.
4. **The official run** (`CONFIRM_LAUNCHED`, `confirm.out`, `confirm.exit`, `confirm_record.json`,
   `confirm_prompts.jsonl`):
   - **The module-blob assertion** agreed at 19:31:56.383, before confirm was called. The lock's expected blob, the
     observed `rr.own_blob()`, `git hash-object` of the imported file, git's HEAD blob and an own sha1 all equal
     `e6cb37767d1d06c6ff40804a88eab569723afdb5`. `rr` was imported from
     `/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/src/neural_decompiler/readout_routing.py`.
   - **The environment:** Python 3.12.13; torch 2.14.0, transformers 5.17.0, transformer-lens 3.9.0,
     huggingface-hub 1.31.0, numpy 2.5.3, safetensors 0.8.0, tokenizers 0.23.2; **4 torch threads**; `HF_HUB_OFFLINE=1`;
     the repository's `.venv`. The model is `EleutherAI/pythia-70m-deduped` at `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`;
     the cached `model.safetensors` has sha256 `3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd`, its
     content id.
   - **The counts:** confirm invocations 1; `load_model` 1; `capture_prompt` 4,320; `run_capture` 4,320
     (`capture_prompt`'s own inner call); `run_patched` 0; `run_interventions` 0; prompt-log errors 0.
   - **The monitoring record:** no repository write outside `outputs/experiment-024/`; no read of 022's table or 023's
     outputs; 0 hook errors.
   - **The writes, in first-write order:** the state (the ledger), then `stage2-measurements.pt`, then 8 more atomic
     state writes.
   - **The timeline (UTC):**

     | event | time |
     |---|---|
     | phase started | 19:32:10 |
     | first capture | 19:32:10.705 |
     | last capture | 19:35:01.147 |
     | stage-2 file written | 19:35:01.288 |
     | result and completed phase, in one atomic write | 19:37:04 |
     | the descriptive records, appended | until 19:38:44.95 |
5. **`verify_confirm.py`, `.out`, `.json`** — the implementer's read-only post-confirm check; every check passed. It
   used the canonical functions, so it shows determinism. The independent review re-derived everything with its own
   code.

## Record (the gitignored outputs and the key digests)

| item | value |
|---|---|
| `stage2-measurements.pt` | 76,314,087 bytes; sha256 `b21babe189d1cccb9ae359cfabf74baaa3a8337cac04612bc0d4aaa232b4dc93`; 20 tensors; 4,320 pairs (cue_final 2,880, coordinated 1,440; Y2 empty by design) |
| `results.json` after confirm | file sha256 `e636210c84a4f3f329e9a5fde5baa493d730b6bf77cb419273056e6c89085a7a`; state `076ab9f984f3ba6244b4df767d8c8fd7ebd40a634521a46a3b881d7da1604e92` |
| the state at the one result write | `37f2f88b7f0fdde73ebbedc319fc5d9edcab59c35133a8cfa8b24c4e3201f9fd`; removing only the descriptive records from the final state reproduces it |
| the pre-confirm state | `58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d`, file `14459cdde6ec2d832c3965d5b0ebaadaf06cef6aa0b0460c3a31b29dd442b262`; reverting only confirm's four fields reproduces both byte for byte |
| the manifest | `fcc437fca9730b8599333eeac95807786570f0d03bd01e49f52daec76ed4e32d` (canonical `{"S2-TARGET": keys}`) |
| the ledger, sorted | `5f386b5353e5ad1fa021b606c63ef507fbe22fa34852c4ed24f91cde13efe627`, equal to the sorted manifest |
| the prompt log | sequence `7654c0e532bee29271e2a95502c0558726cc101d029fff9418c7fa1bafbc0974`; file `d9741815407e3207ba508a5266e6a4434066fea0bdca3c9ca5729a524f8f0ba9` |
| the launcher's record | `confirm_record.json` `3ad7e3b5e1c467370979b8b4ae1a554b0fe7b34607cef464ce5466e4175b7cd8` |
| the model | parameters `fd953f1c745299bedfd1f242fc15c7fe31211f79d8bc0d3ca46372962e7fc0e5`; embedding `9cd6f39b3adfdb74cffe046af459634982684bff9aef2a981760326f1748eaaf`; checkpoint `3da38833…` |
| I7 | bit for bit, nothing differing; scores `a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995` |
| identities | C recomputed from the saved Δx3: bit for bit on 4,320 pairs (max 0.0); I1 `1.5317713646822995e-05` (tolerance 1e-4); I3 `2.1161813141654138e-05` (1e-4); I4 `6.424818726813442e-05` (1e-3); Spearman cross-check difference 0.0 (1e-12); E–N float check 1.1e-16 (1e-12) |

The stage-2 tensor digests (`readout_calibration.tensor_digest`), equal to the state's and re-derived from disk:

| tensor | sha256 |
|---|---|
| `Y1/coordinated/ceiling` | `3ef63e0076530f12b302d094a9db6ec9c1d3d413aaa6ced5668f0e066770a445` |
| `Y1/coordinated/dc` | `190f48814bd2eac7ca3d73facd190e6f6fda1151b80d725db671f5e0f78c43e0` |
| `Y1/coordinated/dx1` | `164bd6ed6acdca36e8030da13f3b20f94eb0c04644d41ed4b62b93defd8283f2` |
| `Y1/coordinated/dx3` | `422439494916fb25feed4a30b081f70046aab3f5a2c71009c2fe1224db586d46` |
| `Y1/coordinated/positions` | `606062edeb075b2df038958c2c5b7f3ee25928ba63d255f5f72d2090cbc74da1` |
| `Y1/cue_final/ceiling` | `35b563decaba7e0e17fa6c9dc54ae4c4bd366e921d9202c3c3a4950d5bccc590` |
| `Y1/cue_final/dc` | `4e15992dfe7eed7ef5745c659171b76cb0d4bc089da17a614d298b55e3b1719b` |
| `Y1/cue_final/dx1` | `5b996e844643e2133701563533ab1f9978f8f313fe57c94a32bebed68dbf14b3` |
| `Y1/cue_final/dx3` | `10067fbff504bd88beb4d0b7d7067c0a78b7602be71f60752ff1e062f1f9d099` |
| `Y1/cue_final/positions` | `9973307c080d835dfdf07114d929f4f0a196f6e40a9ee146888e9b7909045397` |
| `Y2/*/ceiling` and `Y2/*/dc` (0 rows) | `86a734c18c5c99e30d195c8c0ecad8b3a9975c7c0b25e7cfcd306671071d512e` |
| `Y2/*/dx1` and `Y2/*/dx3` (0 rows) | `a5a513bfaf0cf46a64ffa0fae67e10d111d1b142e8a0e44068d51609f22b8023` |
| `Y2/*/positions` (0 rows) | `912167b677a3a5e989045e8d3b7a037ca823c08287ae327c75a34e5b25dfb6eb` |

**Preservation.** `stage2-measurements.pt` and `results.json` are gitignored and are preserved unchanged in
`outputs/experiment-024/`.
- The repository tracks no binary measurement artifact, and it has no convention for backing up large artifacts
  externally, so neither is committed. Their hashes are recorded here.
- The report phase adds only its own entries to `results.json`; reverting them must reproduce the state above.

`SHA256SUMS` lists every archived file in this directory. These are the files as written and run in the session's
scratch directory; none was edited.
