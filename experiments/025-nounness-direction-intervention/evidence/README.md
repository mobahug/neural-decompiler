# Experiment 025 — evidence

| directory | what |
|---|---|
| `freeze-2026-09-26/` | the final implementation test gate (A 542, B 575, C 6 passed; 0 failed) at `c765148`, the guarded production freeze (a dry run, then the one production run), and the pre-install and post-commit checks |
| `freeze-review-2026-09-26/` | the independent, read-only freeze verification: **PASS WITH NOTES — no blockers** |

The freeze artifact `../confirmation-v1.json` was installed byte-identically in `57f1f989b1095708f4805c57be251b2392febc51`:
- file sha256 `54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a`;
- content sha256 `6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4`;
- 90,720 condition-tagged keys, manifest sha256 `0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61`.

Lock and confirm have not run. No fresh scientific prompt of Experiment 025 has run.
