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

Experiments 001–004 are preserved historical observational and infrastructure work. The behavior candidate screen below has since been executed once: it proposes `regular-plural` (count-cued noun number selection in Pythia-70M) as the single target and eliminates `ordinal-suffix`. Experiment 005 is designed ([approved design, revision 4](docs/superpowers/specs/2026-09-17-experiment-005-regular-plural-mechanism-design.md); [plan](docs/superpowers/plans/2026-09-17-experiment-005-regular-plural-mechanism-plan.md)) and implemented under [`experiments/005-regular-plural-mechanism/`](experiments/005-regular-plural-mechanism/README.md) with a frozen extension set; no scientific phase has run yet, no preregistration lock exists, and the future-reserve cases remain unexecuted.

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

The screen was executed once on 2026-09-17; the outcome, evidence copy, and finalist audit are recorded in [`screening/behavior-candidates/README.md`](screening/behavior-candidates/README.md): `regular-plural` (count-cued noun number selection) is the single proposed target and `ordinal-suffix` was eliminated. Experiment 005 (approved design revision 4) is implemented and awaits its Tier A run.

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
