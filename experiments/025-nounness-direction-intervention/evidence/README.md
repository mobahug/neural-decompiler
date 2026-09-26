# Experiment 025 — evidence

| directory | what |
|---|---|
| `freeze-2026-09-26/` | the final implementation test gate (A 542, B 575, C 6 passed; 0 failed) at `c765148`, the guarded production freeze (a dry run, then the one production run), and the pre-install and post-commit checks |
| `freeze-review-2026-09-26/` | the independent, read-only freeze verification: **PASS WITH NOTES — no blockers** |
| `lock-2026-09-26/` | the guarded production lock (two dry runs, then the one official run at `d3ecbd6`): the environment and checkpoint record, the post-lock verification (33/33), the install checks, and the confirm-launcher requirements N1–N4 |
| `lock-review-2026-09-26/` | the independent, read-only lock review, with its own reconstruction of the geometry, the 280 controls and the `D_attn` rows: **PASS WITH NOTES — no blockers** |

The freeze artifact `../confirmation-v1.json` was installed byte-identically in `57f1f989b1095708f4805c57be251b2392febc51`:
- file sha256 `54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a`;
- content sha256 `6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4`;
- 90,720 condition-tagged keys, manifest sha256 `0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61`.

The lock was installed byte-identically in `5ee7a7474c5531bf7a7c19ceff4c8ca857d16bf5`:
- `preregistration-lock.json`: file sha256 `1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be`, content
  sha256 `8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb`;
- `preregistration.md`: sha256 `e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919`.

Confirm and report have not run. No fresh scientific prompt of Experiment 025 has run, and no fresh measurement exists.
