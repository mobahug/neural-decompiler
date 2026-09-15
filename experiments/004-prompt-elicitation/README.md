# Experiment 004: Prompt Elicitation and Held-Out Factual Recall

## Research question

Can one of five predefined prompt formats, selected using 12 development countries, elicit strict next-token capital recall from pinned Pythia-160M on 12 held-out countries?

Experiment 003 found that Pythia-160M improved intended-capital rank and canonical selectivity relative to Pythia-70M but still predicted `" the"` in every case. Experiment 004 distinguishes prompt-selection evidence from confirmation evidence and prevents searching prompt formats on the held-out outcomes.

## Frozen protocol

- Model: `EleutherAI/pythia-160m-deduped`
- Revision: `582159a2dfe3e712a8d47ae83dec95ae3bde8e7e`
- CPU float32, sequential prompts, `torch.inference_mode()`
- TransformerBridge raw coordinates; compatibility mode disabled
- Complete unembedding including bias
- Final logits all-close at `atol=1e-5`, `rtol=1e-5`, with exact rank and top-1 parity

Five frozen formats are evaluated on the development set: canonical completion, possessive completion, direct question, explicit answer prefix, and a fixed two-shot question-answer pattern. Selection is deterministic: top-1 count, median target rank, mean correct-minus-control margin, then fixed format ID.

Each case retains the three balanced capital-token controls from Experiments 002 and 003. All target capitals are single whitespace-prefixed tokens.

## Hard selection boundary

Development and held-out execution are separate:

```bash
uv run python experiments/004-prompt-elicitation/run.py development
```

This writes generated development results and a candidate lock. The candidate must be reviewed, installed as `selection-lock.json`, and committed. Only then may the following run:

```bash
uv run python experiments/004-prompt-elicitation/run.py heldout
```

The held-out command has no format override. It refuses to run unless the committed lock matches the frozen split, formats, selection rules, pinned model, development-artifact digest, deterministic winner, and unchanged protocol/code commit. A failed gate stops execution rather than regenerating the lock.

The held-out countries were not used for Experiment 004 prompt selection, but all 24 countries were measured under older formats in Experiment 003. The result is therefore conditionally confirmatory for transfer of the selected prompt format, not independent country-level replication.

## Decision rule

- At least 6/12 held-out top-1 targets: `strong_support`
- 1–5/12: `partial_insufficient`
- 0/12: `heldout_failure`

This is a predeclared project decision threshold. Rank and selectivity improvements cannot rescue failure of the top-1 criterion.

## Generated outputs

```text
outputs/experiment-004/development-results.json
outputs/experiment-004/results.json
outputs/experiment-004/report.html
outputs/experiment-004/development-prompt-comparison.svg
outputs/experiment-004/heldout-behavior.svg
outputs/experiment-004/heldout-selectivity.svg
```

Outputs are generated and gitignored. Model weights remain in the normal external cache.

## Interpretation boundary

This experiment measures prompt-sensitive behavior and residual-stream projections. It does not establish where a fact is stored or retrieved, identify a causal layer, attention head, MLP, or circuit, or demonstrate a general capacity for factual recall. C001 remains observational regardless of behavioral success.
