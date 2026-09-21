# Experiment 019: Frame-Conditioned Routing of Channel D — Is the Frame-Specific Membership of Block 2's Recovery Subset Predicted from the Frame's Reference State?

Implements the approved design
[`docs/superpowers/specs/2026-09-21-experiment-019-block2-routing-design.md`](../../docs/superpowers/specs/2026-09-21-experiment-019-block2-routing-design.md)
(revision 3, commit `b130b50`; wording `efe0623`) through the plan
[`docs/superpowers/plans/2026-09-21-experiment-019-block2-routing-plan.md`](../../docs/superpowers/plans/2026-09-21-experiment-019-block2-routing-plan.md).
Prospective, zero-parameter, mechanism (routing). Experiment 018 showed that channel D — block 2's MLP at the frame's
own operating point — is carried for unseen cues by a locked subset of 256 neurons and, descriptively, that *which*
neurons carry it depends on the frame. This experiment asks whether that redistribution is predictable: Experiment
018's masked chain now takes one mask per changed position, and a nested ladder of selectors scored with one read-unit
rule is evaluated at `k ∈ {16, 64, 256}` (decision size 64) — the matched global `S'_k` (the decision comparator), the
template ranking `T_k(τ, p)`, the full prospective evaluation `E_k(f, p)` (the template's 254 exposed cues' *predicted*
effects at the frame's own reference state), the operating-point-plus-drive rule `G_k(f, p)` (the frame's reference
pre-activations plus a frozen per-template three-quantile drive profile through the exact GELU), with the inherited
Experiment 018 `S_k`, seeded random `k`-subsets, `S'_1` and the frame's top neuron `E_1(f)` as controls. Every
selector is a frozen rule applied to locked reference states, the weights and the exposed cues' `ΔE`; every subset is an
explicit index list committed before the fresh prompts it is scored on. Two post-confirmation objects enter no
selector: the measured-effect ranking `O_k(f, p)` (descriptive; the target of the membership question) and the
leave-one-cue-out greedy `O*_k(f, −w)` — a *cross-validated empirical headroom witness*, never a global oracle, ceiling
or upper bound — whose headroom `H*_64` classifies only a negative routing result (a low `H*` means the witness did
not establish enough selectable headroom for the fixed-population-versus-unpredicted distinction; it is never a proof
that no frame-specific headroom exists).

## Inputs

- Exposed pool (calibration record only; the floors are frozen constants): the 255 exposed cue tokens (Experiment
  018's 231 and its 24 confirmed) and the 90 exposed frames (its 78 and its 12), with the 80 nouns; the licensed pairs
  are Experiment 018's 9636 exposed and 2160 confirmed pairs (11 796).
- [`inherited/experiment-018-pair-extract.json`](inherited/experiment-018-pair-extract.json) — Experiment 018's
  per-pair measured `F`, `Π`, `ΔT`, `c_L`, head row and its `S0` / `S256` / `S2048` predictions over the 11 796 pairs, the
  digests of its twelve fresh frames' stage-1 reference states and its per-frame top-256 lists (78 at explore, 12 at
  stage 1). Built deterministically by the committed
  [`build_inherited_extract.py`](build_inherited_extract.py) from the closed 018 results state
  (`outputs/experiment-018/results.json`, run `86ccb888f52600ef`, state sha256 `2c9195d5…`) and the 018 lock; its
  `source` names the results-state file sha256 and `state_sha256`, the lock file sha256 and content sha256, the run id,
  the explore and confirm commits and the extraction schema version; `build_inherited_extract.py --check` rebuilds it
  and compares (content sha256 `625394936cadb4bd9ac348d07f22af96b60b4ab0c7399d24df7d5e493bf1ae04`, byte-identical on
  2026-09-21). The Experiment 011 lock's axes and read weight, the Experiment 012 lock's bases, the Experiment 018
  lock's layer-3 bases, `p_t` base, 78 reference states and subsets are used verbatim.
- `confirmation-v1.json` — frozen by tokenizer rules before any Experiment 019 model output: 24 fresh cue tokens by
  lexical class (the first eligible entries of Experiment 017's lists extended), **with no expectation attached**;
  eighteen fresh frames, six per template; the complete prompt-key manifest (every fresh cue in the 90 exposed frames
  and the 18 fresh frames, the fresh frames' reference and cue-pair prompts). Executed only by `confirm`.

## Commands

```bash
uv run python experiments/019-block2-routing/run.py validate
uv run python experiments/019-block2-routing/run.py freeze-confirmation
uv run python experiments/019-block2-routing/run.py explore
uv run python experiments/019-block2-routing/run.py lock
uv run python experiments/019-block2-routing/run.py confirm
uv run python experiments/019-block2-routing/run.py report
```

- `explore` (once, exposed pool): the 90 frames' reference states (the 78 checked against the Experiment 018 lock and
  the 12 against its stage-1 digests; locked); every prospective selector — `S'`, `T`, the drive-quantile tables, `E`
  and `G` for every exposed frame and position, `S'_1`, `E_1`, the random controls, the inherited `S_k` — computed and
  digested from the locked states, the weights and the exposed cues' `ΔE` *before any pair is measured*; I9 (the `E`
  rule with Experiment 018's inputs reproduces its recorded per-frame lists); the 11 796 licensed pairs re-measured with
  the layer-3 residuals, the head's row and the block-2 input at both changed positions captured; replication of the
  018 extract; the identities I1–I10; every prospective rung predicted; the exposed oracles (cue-in-sample ranking; the
  leave-one-cue-out witness) and the exposed statistics.
- `lock` (weights, the locked axes, read, bases, states and lists — no forward pass): the prediction table for every
  fresh token × exposed frame and every prospective rung; the frozen floors; `candidate-lock.json` and
  `candidate-predictions.md`. Installing them as `preregistration-lock.json` and `predictions.md` and committing them
  is the preregistration act.
- `confirm` (once): validates the artifacts (digests, floors, the lists as the frozen rule applied to the locked
  scores), recomputes the selectors from the locked states and every locked prediction (refusing before any fresh
  prompt on a mismatch), then **stage 1** (the eighteen fresh frames' reference and cue-pair prompts only; validity;
  each frame's own `E` and `G` lists from its locked state; the frame-conditional prediction table; serialized and
  digested — zero of the 432 fresh-cue × fresh-frame target pairs and none of the 2160 exposed-frame pairs run before
  the barrier) and, only after re-reading that digest from disk, **stage 2** (every fresh cue in both sets; the
  measured objects and per-neuron effects; then, and only then, the ranking oracle per frame and the witness per pair;
  scoring: Y1, Y2 with the frame-count and split guards and the headroom classification, Y3 with the denominator
  condition and per-set floors, Y4 on Y2 and pooled, Y5 per set).
- `report`: renders `outputs/experiment-019/report.md`.

## Status — implementation; no scientific phase has run

Nothing outside the exposed pool is executed before `confirm`; `explore` runs once, `lock` writes the candidate
artifacts, and `confirm` runs once after the user's lock commit and the reviewer's sign-off.
