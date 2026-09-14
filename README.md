# neural-decompiler
Reverse engineering learned computation in transformer language models through causal circuit discovery and interpretability.

## Experiments

- [Experiment 001: Capital Recall Logit Lens](experiments/001-capital-recall/README.md) — an observational look at how target-token support changes through Pythia-70M's residual stream.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to select Python 3.12 and install the locked dependencies:

```bash
uv sync
```

Model weights are downloaded only when an experiment is run. Hugging Face and TransformerLens keep them in their normal user cache outside this repository.
