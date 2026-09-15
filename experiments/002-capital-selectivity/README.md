# Experiment 002: Held-out Capital-Target Selectivity

## Research question

For the pinned Pythia-70M checkpoint, does country context selectively increase relative support for the correct single-token capital over balanced incorrect capital tokens? Separately, does C001's final-block raw-logit pattern replicate on held-out country–capital pairs?

Experiment 001 showed the same large final-block raw-logit increase for four intended targets, but the model predicted none of those targets as top-1. A vocabulary-wide diagnostic also suggested that raw-logit movement alone was not target-selective. Experiment 002 therefore measures a shift-invariant contrast instead of trying to make Experiment 001 look successful.

## Confirmatory status and frozen protocol

This protocol is confirmatory with respect to the predictions below. It must be committed before the held-out next-token outputs are inspected. The four Experiment 001 countries—France, Germany, Italy, and Finland—are excluded.

The experiment uses 24 convenience-sampled country–capital pairs selected by a tokenizer-only eligibility screen. Both the country and capital must each be one leading-space Pythia tokenizer token. Eligibility was determined without inspecting the model's next-token outputs.

The two templates are analyzed separately:

```text
The capital of {country} is
In {country}, the capital is
```

The frozen pairs, in protocol order, are:

| Country | Correct capital | Country | Correct capital |
|---|---|---|---|
| Spain | Madrid | Portugal | Lisbon |
| Austria | Vienna | Belgium | Brussels |
| Netherlands | Amsterdam | Denmark | Copenhagen |
| Norway | Oslo | Sweden | Stockholm |
| Poland | Warsaw | Hungary | Budapest |
| Greece | Athens | Ireland | Dublin |
| Canada | Ottawa | Japan | Tokyo |
| China | Beijing | Russia | Moscow |
| Egypt | Cairo | Thailand | Bangkok |
| Chile | Santiago | Iraq | Baghdad |
| Iran | Tehran | Philippines | Manila |
| Switzerland | Bern | Syria | Damascus |

For pair index `i`, the three incorrect capital controls are taken from indices `(i + 1) mod 24`, `(i + 7) mod 24`, and `(i + 13) mod 24`. This cyclic derangement prevents self-controls and makes every capital appear exactly three times as an incorrect control in each template. Every candidate capital is therefore evaluated as both a correct and incorrect token under the same template.

## Measurements

Each prompt is processed sequentially in `torch.inference_mode()` using:

```python
TransformerBridge.boot_transformers(
    "EleutherAI/pythia-70m-deduped",
    revision="e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
    device="cpu",
)
```

Compatibility mode remains disabled. Normalized accumulated residual states are projected through the bridge's complete unembedding operation, including bias. All four candidate trajectories for a prompt are calculated from the same projected logit tensor.

At each stage, the primary selectivity quantity is:

```text
correct target logit − mean(logit of the three balanced incorrect targets)
```

Subtracting candidates within the same projected vocabulary vector removes any uniform logit offset. The experiment also records candidate ranks, softmax values labeled as logit-lens projections, actual final top-1 output, pairwise correct-over-control counts, and the change in selectivity across the final transformer block.

The final projected logits must be numerically all-close to the model's actual logits at `atol=1e-5` and `rtol=1e-5`. Target rank and top-1 token must also agree. The run stops on any failure.

## Preregistered predictions and decision rules

The primary hypothesis is supported only if **both templates independently** satisfy all three conditions:

1. Median final correct-minus-control logit is greater than zero.
2. Median final-block selectivity change is greater than zero.
3. The lower bound of the two-sided 95% Wilson interval for the fraction of 24 cases with a positive final margin is greater than 0.5.

Strictly positive margins count as successes; zero counts as not positive. Results are reported even if the rule fails.

C001 replication is a separate secondary outcome. Its preregistered criterion is met if at least 18 of the 24 canonical-template held-out targets have their largest positive adjacent-stage raw-logit change across the final block. The paraphrase result is reported separately and cannot rescue failure of this canonical criterion.

## Investigation stop rule

If the primary rule is not satisfied, this experiment will not be redefined, filtered, or rerun with hand-selected prompts. Pythia-70M capital recall will not proceed directly to internal causal intervention. The next decision would be to choose a behavior that Pythia-70M demonstrably performs or separately preregister a model-suitability comparison before downloading a larger model.

## Run it

From the repository root:

```bash
uv sync
uv run python experiments/002-capital-selectivity/run.py
```

The run uses the normal Hugging Face and TransformerLens caches. It downloads no model other than the pinned Pythia-70M checkpoint, trains nothing, and writes no model weights into the repository.

Generated outputs are written to `outputs/experiment-002/`:

- `results.json` — complete protocol, tokenization, per-case trajectories, parity validation, and aggregate results;
- `report.html` — self-contained local report;
- `selectivity-by-stage.svg` — median selectivity trajectory with interquartile bands;
- `final-block-selectivity.svg` — per-country final-block selectivity changes.

## Completion

- [x] Research question and frozen protocol documented
- [x] Held-out cases and controls fixed before outcome inspection
- [x] Implementation covered by automated tests
- [ ] Real Pythia-70M execution completed
- [ ] Final-projection validity checks passed
- [ ] Generated artifacts inspected
- [ ] Claim evidence reviewed

## Interpretation limits

Experiment 002 compares outputs under controlled prompt–target relationships. It does not intervene on internal activations and cannot identify a causal internal mechanism. Even a positive result would not show that a layer stores or retrieves a fact, that an attention head or MLP is responsible, or that a circuit has been found.

The 24 pairs form a tokenization-constrained convenience sample, not a random sample of countries. Balanced target reuse controls token identity within the candidate set, but does not directly measure training-corpus frequency. The two templates do not establish robustness to arbitrary paraphrases, other languages, multi-token capitals, other facts, checkpoints, models, or model families.

## Related claims

- [C001: Largest target-logit increase occurs across the final block for four tested capital prompts](../../research/claims/C001-final-block-target-logit-increase.md) — Experiment 002 provides a preregistered held-out replication test for the canonical template.
