# Experiment 001: Capital Recall Logit-Lens Design

## Purpose

Experiment 001 is an observational study of how support for a correct next token changes through the residual stream of a small pretrained transformer. It asks:

> For a small pretrained transformer given a simple factual-recall prompt, how does support for the correct next token evolve through the residual stream across transformer layers?

The primary example is `"The capital of France is"` with target token `" Paris"`. Comparison examples use Germany/Berlin, Italy/Rome, and Finland/Helsinki.

This experiment does not identify where facts are stored or retrieved, attribute changes to attention or MLPs, or make causal claims. Component-level interventions belong to Experiment 002.

## Scope

The implementation contains one small reusable logit-lens primitive and one experiment-specific runner. It does not create a general experiment framework, notebook infrastructure, causal-intervention utilities, or scaffolding for later experiments.

Proposed repository layout:

```text
pyproject.toml
src/neural_decompiler/
    __init__.py
    logit_lens.py
experiments/001-capital-recall/
    run.py
    README.md
tests/
    test_logit_lens.py
    test_experiment_001.py
outputs/experiment-001/       # generated and gitignored
    results.json
    report.html
    target-logits.svg
    target-ranks.svg
```

## Model and Runtime

The default and only automatic model is `EleutherAI/pythia-70m-deduped`. It is loaded on CPU through the current TransformerLens 3 bridge API:

```python
from transformer_lens.model_bridge import TransformerBridge

model = TransformerBridge.boot_transformers(
    "EleutherAI/pythia-70m-deduped",
    device="cpu",
)
```

TransformerLens and Hugging Face use their normal external caches. Model weights are never stored in the repository. The runner never trains the model and never selects or downloads a larger model.

The four prompts run sequentially under `torch.inference_mode()` in float32 on CPU. Only the activations required by TransformerLens's cache/logit-lens workflow are retained for the current prompt, keeping memory use appropriate for an 8 GB Apple Silicon Mac.

The experiment is run, after one-time dependency installation, with:

```bash
python experiments/001-capital-recall/run.py
```

## Why Compatibility Mode Is Disabled

`TransformerBridge` preserves the raw Hugging Face weights by default. This is the desired coordinate system because the experiment must compare its final logit-lens projection directly with the bridge's actual model output.

Compatibility mode exists to apply the legacy `HookedTransformer` weight-processing conventions, including LayerNorm folding and weight centering. The experiment does not depend on those conventions. It will therefore not call `enable_compatibility_mode()` unless implementation-time validation proves that a required cache helper is unavailable or numerically incorrect without it. If that unexpected condition occurs, implementation stops and the reason is reported before compatibility mode is enabled.

## Analysis Flow

For each prompt and target:

1. Tokenize the prompt with the bridge tokenizer and record token strings and token IDs.
2. Convert the target, including its leading space, with the bridge's single-token helper. If it is not exactly one token, stop with a precise error.
3. Run one forward pass with `run_with_cache` and retain the returned actual logits.
4. Obtain the accumulated residual stream at the final prompt position for the embedding stage and after every transformer block.
5. Apply the model's final normalization independently to each stage using TransformerLens's `accumulated_resid(..., apply_ln=True)` behavior.
6. Project each normalized state through the bridge's full unembedding operation, including any unembedding bias.
7. For every stage, compute the target logit, 1-based target rank, and softmax value. The softmax value is named `logit_lens_probability` and is never described as an actual intermediate model prediction.
8. Compare adjacent stages to find the largest target-logit increase and decrease and the largest target-rank improvement and deterioration.

TransformerLens cache labels are converted to explicit public labels:

- `embedding`
- `after_layer_0`
- `after_layer_1`
- and so on through the final transformer block

## Final-Projection Validation

For every prompt, the final projected vocabulary-logit vector is compared with `actual_logits[0, -1, :]` from the same forward pass.

The result records:

- maximum absolute raw-logit difference;
- mean actual-minus-projected logit offset;
- maximum absolute difference after removing that constant offset;
- configured absolute and relative tolerances;
- whether all logits are numerically close;
- whether the target's 1-based rank agrees;
- whether the top-1 token ID agrees.

Target-rank and top-1 disagreement is a correctness failure and stops the run with a clear error. A raw-logit difference outside tolerance that preserves rank and top-1 is retained as a completed run, but it produces a prominent terminal warning and report note containing the measured error and offset. This makes normalization or centering differences visible rather than silently accepting them.

## Factual-Recall Status

The conservative per-prompt success criterion is whether the intended target token is the actual final top-1 next token. The JSON contains this Boolean directly.

If Paris is not the primary prompt's actual top-1 next token, the run still produces the observational artifacts, but the terminal trace and report state that Pythia-70M did not predict the intended answer. No replacement model is loaded. Considering a larger model requires a separate user decision after reporting why the current model was insufficient and estimating the replacement's memory requirements.

## Outputs

### Machine-readable result

`outputs/experiment-001/results.json` contains:

- schema version and run timestamp;
- exact requested model identifier and revision information available at runtime;
- device and dtype;
- Python, TransformerLens, PyTorch, Transformers, Hugging Face Hub, and Matplotlib versions;
- prompt and target text;
- prompt token strings and IDs;
- target token string and ID;
- actual final top-1 token, ID, logit, and probability;
- per-stage target logit, 1-based rank, and `logit_lens_probability`;
- adjacent-stage deltas and extrema used by the observations;
- final-projection validation metrics;
- target-is-final-top-1 status;
- deterministic observations derived from recorded measurements.

### Terminal trace

The runner prints tokenization, target token information, every stage's target logit/rank/logit-lens probability, actual final prediction, final-projection validation, and measured observations. It prints an explicit factual-recall warning for every prompt whose target is not top-1.

### Visual report

The runner creates:

- `target-logits.svg`, comparing all four targets across stages;
- `target-ranks.svg`, comparing all four 1-based ranks with rank 1 at the top;
- `report.html`, embedding both charts, measured tables, validation results, observations, and interpretation limits.

The report clearly distinguishes actual final output from intermediate logit-lens projections.

Generated outputs are gitignored and reproducible by rerunning the experiment.

## Documentation Language

`experiments/001-capital-recall/README.md` explains the residual stream, logits, unembedding, and the logit lens in accessible language. It states that the graphs describe projected token support and do not prove that a layer stores or retrieves a fact, that attention explains a change, or that any component causally causes the prediction.

Generated observations use only descriptive language, for example:

> The Paris target logit increased most strongly from `after_layer_2` to `after_layer_3`.

They never use causal verbs such as “stored,” “retrieved,” or “caused.”

## Dependencies

Runtime dependencies are intentionally small:

- Python 3.10 through 3.12;
- `transformer-lens ~= 3.9.0`;
- PyTorch compatible with the selected TransformerLens release and macOS arm64;
- Matplotlib for SVG generation.

PyTorch is declared directly because the project imports it even though TransformerLens also depends on it. No pandas, Plotly, Jupyter, web framework, template engine, or training dependency is added.

The development dependency is `pytest`.

## Testing

Implementation follows test-driven development. Unit tests use small deterministic tensors and lightweight fakes, so the default test suite neither downloads model weights nor requires network access.

Tests cover:

- single-token target validation;
- residual projection and full-unembedding behavior;
- 1-based rank and top-1 calculation;
- stage-label conversion;
- final-logit numerical validation and mismatch failures;
- deterministic delta/observation generation;
- JSON schema content;
- HTML/SVG artifact creation;
- language checks preventing prohibited causal claims in generated observations.

A separately marked integration smoke test may load the real Pythia model from the normal cache. It is not part of the fast default test suite. The real experiment run is the final end-to-end validation and produces the required artifacts.
