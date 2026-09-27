# Experiment 025 — evidence

**Experiment 025 is CLOSED.** Every phase ran exactly once. The official result, independently reviewed:
**`CAUSAL_EFFECT_NOT_ESTABLISHED`**, with A = 26/40, B = 23/40, G = 37/40, threshold = 27, reference tail =
0.01923865414210013.

| directory or file | what |
|---|---|
| `freeze-2026-09-26/` | the final implementation test gate (A 542, B 575, C 6 passed; 0 failed) at `c765148`, the guarded production freeze (a dry run, then the one production run), and the pre-install and post-commit checks |
| `freeze-review-2026-09-26/` | the independent, read-only freeze verification: **PASS WITH NOTES — no blockers** |
| `lock-2026-09-26/` | the guarded production lock (two dry runs, then the one official run at `d3ecbd6`): the environment and checkpoint record, the post-lock verification (33/33), the install checks, and the confirm-launcher requirements N1–N4 |
| `lock-review-2026-09-26/` | the independent, read-only lock review, with its own reconstruction of the geometry, the 280 controls and the `D_attn` rows: **PASS WITH NOTES — no blockers** |
| `confirm-2026-09-26/` | the one production confirmation at `6f72d7d` (launcher v3 `58b760be…`, exit 0): the run record, the 90,720-line run log, the exit-127 shell event (it never started Python; not a scientific attempt), and the pre-confirm launcher versions, smoke test and rehearsals |
| `confirmation-review-2026-09-26/` | the independent, read-only confirmation review, with full recomputation of the gates, ℓ, D_attn, A, B and G: **PASS WITH NOTES — no blockers** |
| [`report-2026-09-26/`](report-2026-09-26/README.md) | the check-only run, the guarded report launcher (no model, no prompt), the official run's log and record, and the post-report verification (15/15); `SHA256SUMS` — report ran **once** at `c8d9fad`, exit 0, no incident; `ad9f85e6…`; final state `534dad39…` |
| [`final-report-2026-09-26.md`](final-report-2026-09-26.md) | `outputs/experiment-025/report.md`, byte-identical, not re-rendered or edited: `ad9f85e62e74bf10b22060dc6f49ab75ddf3e79b7c81814d29ac6aa19fb75407` (7,868 bytes) |
| [`confirmation-record-2026-09-26.json`](confirmation-record-2026-09-26.json), [`closure-2026-09-26/`](closure-2026-09-26/README.md) | the confirmation-record extract of the gitignored final state (the 022–024 convention: the ledger replaced by digests; the run accounting, the isolation, the exit-127 event and the official-result summary, copied from the state), its builder, and the pre-closure check (24/24); `SHA256SUMS` — content `30da650f3d8ce15fdfef86e51a5e2adb8ebb5bb4ec45f1b5aedaa738796052ca`, file `c7fca5c06de7c3722fd223a9a1805967c697626e86b85d85f05b7ff6e3a592ba` |
| [`erratum-2026-09-27-tracked-binary-files.md`](erratum-2026-09-27-tracked-binary-files.md) | a post-closure documentation correction: the README's clause "The repository tracks no binary measurement artifact" was wrong; six binary tensor files are tracked in the evidence. No result, digest or record changes |

The freeze artifact `../confirmation-v1.json` was installed byte-identically in `57f1f989b1095708f4805c57be251b2392febc51`:
- file sha256 `54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a`;
- content sha256 `6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4`;
- 90,720 condition-tagged keys, manifest sha256 `0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61`.

The lock was installed byte-identically in `5ee7a7474c5531bf7a7c19ceff4c8ca857d16bf5`:
- `preregistration-lock.json`: file sha256 `1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be`, content
  sha256 `8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb`;
- `preregistration.md`: sha256 `e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919`.

## The provenance chain (immutable hashes)

| step | artifact | hash |
|---|---|---|
| design, plan | revision 1 `c0885e5` (corrected in `26c9925`); plan revision 1 `7d90d28` | — |
| freeze | `confirmation-v1.json` | file `54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a`, content `6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4` |
| manifest | 40 cues × 108 frames × 21 conditions = 90,720 condition-tagged keys | `0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61` |
| lock | `preregistration-lock.json` | file `1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be`, content `8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb` |
| preregistration | `preregistration.md` | `e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919` |
| analysis module | `src/neural_decompiler/cue_rotation.py` | git blob `622aa832ae29047c3060d989928236320374e227` |
| confirm launcher | `confirm-2026-09-26/launch_confirm_v3.py` | `58b760bea9d8a7bf66bef4352deb5c7153f7aea46c264de02f49c9ab96d1e307` |
| fresh measurements (gitignored) | `outputs/experiment-025/stage2-measurements.pt` | `87284bddfba16a1b7709d7d33f98aa1a0898b1d85767956a524d2950d80a7788` (898,673,151 bytes, 256 tensors) |
| state at the result write | — | `aa3db023c4a4afbe730a0e55df00d6d378184bcf8fdf77c967f4d0aad11907c4` |
| confirm-final state | — | `1d5d37f035bc3153fec6efac527b088f64cd42fcd74e59680f4388daea98b33c` (file `c827bb69…`) |
| report | `final-report-2026-09-26.md` | `ad9f85e62e74bf10b22060dc6f49ab75ddf3e79b7c81814d29ac6aa19fb75407` |
| final state (gitignored) | `outputs/experiment-025/results.json` | `534dad3907d94fa78c3a88a9e74bd88051b6ada57a09daff8007d2a3495bcdb8` (file `fef818dc…`) |
| confirmation-record extract | `confirmation-record-2026-09-26.json` | content `30da650f…`, file `c7fca5c0…` |

Each state digest is `sha256(canonical_json(state without state_sha256))`. Reverting only the report entries of the
final state gives the confirm-final state. Removing only the descriptive records from that gives the state at the
result write.
