# Neural Decompiler

Neural Decompiler is a mechanistic-interpretability research project investigating whether learned transformer computations can be turned into increasingly complete, causal, and human-understandable explanations.

The project advances through small, reproducible experiments. It separates what was done from what the resulting evidence justifies us in believing, preserves negative and contradictory results, and limits every conclusion to the models, tasks, prompts, and interventions actually tested.

## Research framework

- [Research methodology](docs/RESEARCH_METHODOLOGY.md) — evidence levels, claim governance, falsification rules, and the claim template.
- [Claim registry](research/claims/README.md) — an index of the project's scientific claims. Each claim file is the authoritative record of its evidence and current status.
- [Anomaly registry](research/anomalies/README.md) — a lightweight inbox for unexpected measurements that merit verification, replication, or explanation before they become claims.

## Experiments

- [Experiment 001: Capital Recall Logit Lens](experiments/001-capital-recall/README.md) — an observational look at how target-token support changes through Pythia-70M's residual stream.
- [Experiment 002: Held-out Capital-Target Selectivity](experiments/002-capital-selectivity/README.md) — tests whether country context favors the correct capital over balanced incorrect capital-token controls.
- [Experiment 003: Pythia Scale-and-Behavior Comparison](experiments/003-pythia-scale-comparison/README.md) — compares fresh Pythia-70M and Pythia-160M behavior, selectivity, and normalized-depth trajectories under one frozen pipeline.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to select Python 3.12 and install the locked dependencies:

```bash
uv sync
```

Model weights are downloaded only when an experiment is run. Hugging Face and TransformerLens keep them in their normal user cache outside this repository.
