# Experiment 025 — post-closure documentation correction (2026-09-27): tracked binary files

This correction was added after the closure (`d7c5e62`). It is documentation only. It changes no result, label,
digest, report, record or artifact of Experiment 025. It adds this file, adds one line to the evidence index and edits
two passages of the README. It was found while building the synthesis source map (`research/synthesis/`).

## The wrong sentence

The README's "Report and closure" section, as committed in `d7c5e62` (lines 227–230), says:

> The fresh measurements `outputs/experiment-025/stage2-measurements.pt` (898,673,151 bytes, sha256
> `87284bddfba16a1b7709d7d33f98aa1a0898b1d85767956a524d2950d80a7788`) and the final state (`534dad39…`, file
> `fef818dc…`) stay local and unchanged. The repository tracks no binary measurement artifact; their hashes are
> recorded in the evidence and the extract.

The clause "The repository tracks no binary measurement artifact" is wrong. It was carried over from Experiment 024's
README (line 356).
- For 024 the sentence holds for 024's own directory, where no binary file is tracked.
- The repository as a whole already tracked binary tables of Experiments 022 and 023. Among them is 023's
  `exposed-cells.f64`, which is derived from 022's exposed measurements.

## What is true

- **What stands.** The stage-2 measurement file and the final results state are local, git-ignored and unchanged, and
  their digests are recorded in the evidence and in the confirmation-record extract.
- **Six binary tensor files are tracked** in 025's evidence directories:

  | file | bytes | sha256 | committed in | contents |
  |---|---|---|---|---|
  | `confirm-2026-09-26/smoke_reference.pt` | 1,282,873 | `e939adcf0f69e4790e2e47fe520d2c7db55a5b97038517280dfedf3260c68472` | `c8d9fad` | the unguarded reference process's logits and captured slices for the guard smoke test's three spent reference prompts; no fresh prompt |
  | `confirmation-review-2026-09-26/r2_values.pt` | 2,006,963 | `41d03989dde70beaa0b94cb37282b47c666f8d37754bedd0704e64c0e1f527d7` | `c8d9fad` | the confirmation reviewer's recomputed per-cue values (MSE, ℓ, nMSE, `D_attn`) and per-pair sums, **derived from the stage-2 measurements** |
  | `confirmation-review-2026-09-26/r4_worker0.pt` | 5,382,143 | `3aa8796992c0a41caa3485ac83166a08ec011a6c41e425e7934c2b32a87e94fd` | `c8d9fad` | the reviewer's per-run gate records for the first half of the 90,720 runs, **derived from the stage-2 measurements** |
  | `confirmation-review-2026-09-26/r4_worker1.pt` | 5,382,143 | `42559adb4fcb5f4a9b0c32d263fe387e937bb356d57fba6e91e12bb6644fbb30` | `c8d9fad` | the same, for the second half |
  | `confirmation-review-2026-09-26/vectors32_own.pt` | 1,732,415 | `5c3bc53892c9c56525e67267e8999020b623b4864892a88715b4f47c2dc8cd0c` | `c8d9fad` | the reviewer's own reconstruction of the float32 patch vectors, from the weights |
  | `lock-review-2026-09-26/r2a_own.pt` | 9,578,275 | `85e515fc9a1d98372f955060edd878825d6c0dad1322d2366818c58eec3d1d96` | `6f72d7d` | the lock reviewer's own reconstruction of the direction, the geometry and the controls, from the weights |

- **So the sentence was wrong in substance, not only in wording.** Three of the files hold values derived from the
  fresh measurements. The measurement file itself is not tracked.

## What changes

- The README keeps the wrong clause, struck through, followed by a dated correction and a link to this file.
- A pointer at the top of the README names the correction.
- The evidence index lists this file.

Nothing else changes. The official result, every digest, the report, the confirmation-record extract and every other
evidence file are unchanged.

**This file's own wording was made exact on 2026-09-27, in the commit after `e34f16d`.** The first version said no
evidence file changed, and called 024's sentence simply true. The independent review of the synthesis source map
pointed out both statements.
