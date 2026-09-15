# Experiment 003: Pythia Scale-and-Behavior Comparison

## Research question

Does scaling from Pythia-70M to Pythia-160M improve factual-capital completion and correct-target selectivity, and does the observational raw-logit trajectory associated with C001 persist?

This experiment is an observational cross-model comparison. It does not test whether adding layers causes a behavior, whether corresponding normalized depths perform equivalent computations, or whether any layer or component stores or retrieves a fact.

## Frozen protocol

This protocol and implementation are committed before any Pythia-160M output is inspected.

The two pinned models are processed sequentially on CPU in float32 through the same Experiment 003 code path:

| Role | Model | Revision | Layers |
|---|---|---|---:|
| Baseline | `EleutherAI/pythia-70m-deduped` | `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` | 6 |
| Comparison | `EleutherAI/pythia-160m-deduped` | `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e` | 12 |

The experiment reuses Experiment 002's 24 country–capital pairs, two exact templates, cyclic control offsets, and single-token eligibility rules. Every case retains three balanced incorrect capital controls. Results include each individual control, their mean logit, and correct-versus-control pairwise comparisons.

Both models use the current TransformerBridge API in raw coordinates with compatibility mode disabled. Normalized accumulated residual states are projected using the bridge's complete unembedding. Every final projection must satisfy raw-logit tolerance, target-rank parity, and top-1 parity against the actual model output. A failure stops the run.

## Historical 70M consistency gate

The fresh Experiment 003 Pythia-70M cases are checked against `outputs/experiment-002/results.json` using `atol=1e-5` and `rtol=1e-5`.

Scale interpretation stops on any difference in:

- tokenized prompt or token IDs;
- correct or control target construction;
- actual top-1 token ID;
- intended-target rank;
- final correct-minus-control margin;
- final-block selectivity change; or
- final-block raw target-logit change.

Run Experiment 002 first if its generated result is not present.

## Behavioral decision criterion

Pythia-160M qualifies as a substantially better factual-recall substrate for this project only if, on the canonical template:

1. the intended capital is the actual top-1 token for at least 12 of 24 prompts; and
2. its top-1 count exceeds Pythia-70M's count.

This is a predeclared project decision threshold, not a universal scientific threshold. The paraphrase condition is reported separately and cannot rescue canonical failure.

## Measurements

For every model × country × template case, the JSON records prompt tokenization, target and control tokens, actual top-1 output, final intended-target rank/logit/probability, raw target-logit trajectory, three-control selectivity trajectory, adjacent-stage extrema, final-block changes, normalized depth, and final-projection validation.

Cross-model summaries remain separate by template and report rank improvement, selectivity-margin change, top-1 gains/losses, variation, and deterministic paired percentile-bootstrap intervals. The bootstrap uses seed `3003`, 10,000 resamples, and the median paired effect. These intervals are descriptive: the 24 countries are a small tokenization-constrained convenience sample.

Normalized depth is defined as:

```text
embedding = 0
post-block i = (i + 1) / number_of_layers
final block = 1
```

Normalized depth is a plotting coordinate only. Equal normalized depth does not establish functional equivalence.

## Run

```bash
uv run python experiments/003-pythia-scale-comparison/run.py
```

The command downloads only the pinned Pythia-160M checkpoint if it is absent from the normal Hugging Face cache. It does not vendor weights, train either model, or download a larger model.

Generated outputs:

```text
outputs/experiment-003/results.json
outputs/experiment-003/report.html
outputs/experiment-003/behavior-comparison.svg
outputs/experiment-003/selectivity-comparison.svg
outputs/experiment-003/normalized-depth-trajectories.svg
```

## Interpretation boundaries

The report separates four questions:

- **BEHAVIOR:** whether the intended capital is the actual next-token prediction;
- **SELECTIVITY:** whether the correct capital's logit exceeds balanced incorrect capital logits;
- **TRAJECTORY:** how projected support changes through normalized residual depth;
- **GENERALIZATION:** whether C001's raw-logit pattern appears at another model scale.

Experiment 003 cannot establish that more layers caused better recall, that a specific layer stores knowledge, that the same circuit exists in both models, or that attention or MLP components caused any measured change. Those require interventions in a later approved experiment.

## Completion

- [x] Protocol frozen before Pythia-160M output inspection
- [x] Unit tests added without requiring model downloads
- [ ] Real two-model run completed
- [ ] Historical 70M consistency gate passed
- [ ] Final-projection validation passed for both models
- [ ] HTML and SVG artifacts inspected
- [ ] Claim and anomaly evidence reviewed

