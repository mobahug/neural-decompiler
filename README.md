# Neural Decompiler

Neural Decompiler is a mechanistic-interpretability research project investigating whether learned transformer computations can be turned into increasingly complete, causal, and human-understandable explanations.

The project advances through small, reproducible experiments. It separates what was done from what the resulting evidence justifies us in believing, preserves negative and contradictory results, and limits every conclusion to the models, tasks, prompts, and interventions actually tested.

## Research framework

- [Research methodology](docs/RESEARCH_METHODOLOGY.md) — evidence levels, claim governance, falsification rules, and the claim template.
- [Claim registry](research/claims/README.md) — an index of the project's scientific claims. Each claim file is the authoritative record of its evidence and current status.
- [Anomaly registry](research/anomalies/README.md) — a lightweight inbox for unexpected measurements that merit verification, replication, or explanation before they become claims.
- [Causal instrumentation](docs/INSTRUMENTATION.md) — the Pythia-first capture, intervention, decomposition, and provenance contracts used by future experiments.

## Experiments

- [Experiment 001: Capital Recall Logit Lens](experiments/001-capital-recall/README.md) — an observational look at how target-token support changes through Pythia-70M's residual stream.
- [Experiment 002: Held-out Capital-Target Selectivity](experiments/002-capital-selectivity/README.md) — tests whether country context favors the correct capital over balanced incorrect capital-token controls.
- [Experiment 003: Pythia Scale-and-Behavior Comparison](experiments/003-pythia-scale-comparison/README.md) — compares fresh Pythia-70M and Pythia-160M behavior, selectivity, and normalized-depth trajectories under one frozen pipeline.
- [Experiment 004: Prompt Elicitation](experiments/004-prompt-elicitation/README.md) — tests a preregistered prompt intervention selected on development data and evaluated on held-out cases.

Experiments 001–004 are preserved historical observational and infrastructure work. The behavior candidate screen below has since been executed once: it proposes `regular-plural` (count-cued noun number selection in Pythia-70M) as the single target and eliminates `ordinal-suffix`. Experiment 005 ([design, revision 5](docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md); [plan](docs/superpowers/plans/2026-09-17-experiment-005-regular-plural-mechanism-plan.md); [experiment](experiments/005-regular-plural-mechanism/README.md)) has been executed to completion: protocol v1's Tier A ended in `NO_COMPACT_MECHANISM` (the compact program missed a template mean), and the protocol v2 continuation's single preregistered confirmation ended in `BEHAVIOR_NOT_REPLICATED` (the reserve nouns failed the behavioral precondition), leaving claim [C002](research/claims/C002-count-cued-noun-number-circuit.md) at `LOCALIZED`. Experiment 006 ([design, revision 4](docs/superpowers/specs/2026-09-18-experiment-006-low-rank-cue-decompilation-design.md); [experiment](experiments/006-low-rank-cue-decompilation/README.md)) ran its Tier A once and stopped at the pre-lock quality gate: a rank ≤ 4 PCA projection of the layer-0 MLP cue encoding cannot represent the cue-to-readout map (normalized leave-one-cue-out RMSE 0.62), so no lock was written and the fresh confirmation set remains unexecuted; Experiment 006 is closed at `QUALITY_GATE_FAILED`. Experiment 007 ([design, revision 3](docs/superpowers/specs/2026-09-18-experiment-007-supervised-cue-subspace-design.md); [plan](docs/superpowers/plans/2026-09-18-experiment-007-supervised-cue-subspace-plan.md); [experiment](experiments/007-supervised-cue-subspace/README.md)) tested a response-supervised shared low-rank cue subspace on the same untouched holdout: Tier A selected rank 1 and passed the pre-lock quality gate (leave-one-cue-out Spearman 0.77, normalized RMSE 0.32; about half the PCA family's error at every rank), and the single preregistered confirmation ended in `PROGRAM_NOT_SUPPORTED`: the rank-1 direction predicts unseen tokens about as well as a full-dimensional ridge map (Y1 MAE 1.07 vs 1.05) but misses the Y1 rank-correlation floor (0.785 vs 0.80) and the confident-sign rule on the singular-selecting determiners `this` and `another`, whose anomalously weak E-patch response neither the selected program nor the tested ridge baseline explains; C002 stays at `LOCALIZED` (P3's correlation floor failed on both fresh sets). Experiment 007 is closed. Experiment 008 ([design, revision 4](docs/superpowers/specs/2026-09-18-experiment-008-cue-suppression-localization-design.md); [experiment](experiments/008-cue-suppression-localization/README.md)), a discovery-only causal trace over the fully exposed pool (40 cue tokens × 18 frames), ran once: its frozen summary is `MIXED`, `every` is an axis artifact of the 007 direction, context gating and late cancellation are rejected, and the consistent finding is at transport — `this`, `a`, `another` still carry about half of the plural cue's encoding-axis signal at the input of `L03.H04`, which attends to the cue normally but outputs 3–19% of the number signal (versus ≈ 100% for numerals and 150–180% for `these`/`those`); the axis component of their encoding alone drives 50–64% of the plural effect while the whole encoding drives none. This transport-selectivity hypothesis is handed to a prospective Experiment 009 ([design, revision 2](docs/superpowers/specs/2026-09-18-experiment-009-head-transport-rule-design.md); [experiment](experiments/009-head-transport-rule/README.md)), whose Tier A ran once: a fixed-normalization linear OV read-out of the residual arriving at `L03.H04` explains the head's number-axis output change over the exposed pool (R² 0.91), so the mechanism level P1 is locked; a weight-only rank-1 transport rule passed its leave-one-cue-out gate while the forty-token contrast rule did not; and the single preregistered confirmation on 23 frozen new cues in 6 new frames ended `HEAD_MECHANISM_CONFIRMED_P1 | TRANSPORT_RULE_FAILED | NOT_LOCKED`: the fixed-normalization linear OV read-out predicted the head's measured transport of every new cue (Spearman 0.96, MAE 0.07 over 138 pairs), while the weight-only rank-1 rule from the encoding missed its rank floor (0.77 vs 0.80) and the frozen category checks (`either`/`neither` are transported mid-way, unlike `this`). The head-level mechanism is confirmed. Experiment 010 ([design, revision 3](docs/superpowers/specs/2026-09-18-experiment-010-read-direction-assembly-design.md); [experiment](experiments/010-read-direction-assembly/README.md)) then attributed the head-readable number signal exactly — no fitting — to the encoding and to every head and MLP of layers 1–2 over 63 exposed cues × 24 frames: the frozen summary is `MIXED` (its relay bound on gross layer change, 0.25, is below every token's baseline of 0.29–0.71), but the ledger shows the opposition inside the encoding itself — for `this`, `a`, `another`, `every` the number-axis part of `E` reads +0.17 to +0.46 of the plural cue's signal through the head's direction and the rest of `E` reads −0.17 to −0.35 in every frame, while layers 1–2 net at most ±0.07 — and the zero-parameter encoding read ranks the 63 tokens' transport with Spearman 0.93. Experiment 011 ([design, revision 2](docs/superpowers/specs/2026-09-19-experiment-011-encoding-read-prospective-design.md); [experiment](experiments/011-encoding-read-prospective/README.md)) now tests that read prospectively: 24 new cue words and 6 new frames are frozen, the zero-parameter encoding read `g_E` of every new cue is committed from the weights alone (no fresh forward pass), and the single confirmation will score it against the head's measured transport (floors Spearman ≥ 0.80, MAE ≤ 0.248) together with the replication of the linear head read-out; the candidate lock awaits installation and the reviewer's sign-off.

## Behavior candidate screening

The candidate-selection study is frozen in [the screening design](docs/superpowers/specs/2026-09-16-behavior-candidate-screening-design.md) and its [implementation plan](docs/superpowers/plans/2026-09-16-behavior-candidate-screening.md). It screens exactly two retained candidates, `regular-plural` and `ordinal-suffix`, against fixed behavioral gates, trivial baselines, an exploratory causal compactness probe, and a finalist prior-art audit. It can return zero or one proposed target. It is candidate selection, not Experiment 005, and it creates no Experiment 005 directory, claim, or preregistration.

The committed manifest `screening/behavior-candidates/manifest-v1.json` is the sole authority for every case, split, tokenization decision, control, seed, and threshold. The runner exposes four phases and no flag that selects a candidate, model scale, threshold, or template:

```bash
uv run python screening/behavior-candidates/run.py validate
uv run python screening/behavior-candidates/run.py behavioral
uv run python screening/behavior-candidates/run.py compactness
uv run python screening/behavior-candidates/run.py report
```

- `validate` checks the manifest digest, counts, split disjointness, and single-token development cases without loading a model.
- `behavioral` runs both candidates on pinned Pythia-70M and repeats the byte-identical protocol on pinned Pythia-160M only when zero candidates pass every 70M gate.
- `compactness` probes only behavioral passers with development data; holdout and future-reserve cases are never executed during selection.
- `report` renders the candidate matrix; `report --audit-json JSON` records the finalist prior-art audits (write-once per candidate) and the final zero/one-target decision.

The screen was executed once on 2026-09-17; the outcome, evidence copy, and finalist audit are recorded in [`screening/behavior-candidates/README.md`](screening/behavior-candidates/README.md): `regular-plural` (count-cued noun number selection) is the single proposed target and `ordinal-suffix` was eliminated. Experiment 005 has run once under protocols v1 and v2; see its README for the recorded outcomes.

Scientific phases require a clean, committed tree and run once. `--resume` continues only an interrupted, provenance-identical run. A completed screen is rerun only through `--incident-note TRACKED_PATH`, which invalidates the whole affected model screen for every candidate after a committed software fix. Generated outputs live under `outputs/behavior-candidate-screening/` (`results.json`, `report.md`) and are not committed.

## Causal instrumentation

The Phase 1 instrumentation layer provides pinned Pythia loading, canonical TransformerBridge component resolution, selective activation capture, exact zero/replacement interventions, Pythia-specific decomposition checks, task-independent behavior declarations, and reproducible provenance records.

Its purpose is to support progressively stronger causal accounts of learned computation. A captured activation, a decomposition identity, or a behavioral change after ablation is not by itself a mechanistic explanation. Future experiments must preregister controls, competing explanations, predicted intervention outcomes, and success criteria.

The default test suite is offline. The separately marked live contract test checks the exact pinned Pythia-70M adapter on CPU when the model is available:

```bash
NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 uv run pytest tests/test_pythia_bridge_contract.py -m pythia_smoke -q
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to select Python 3.12 and install the locked dependencies:

```bash
uv sync
```

Model weights are downloaded only when an experiment is run. Hugging Face and TransformerLens keep them in their normal user cache outside this repository.
