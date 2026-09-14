# Experiment 001: Capital Recall Logit Lens

## Research question

For a small pretrained transformer given a simple factual-recall prompt, how does support for the correct next token evolve through the residual stream across transformer layers?

The primary prompt is:

```text
The capital of France is
```

with the intended next token `" Paris"`. The experiment also measures Germany/Berlin, Italy/Rome, and Finland/Helsinki.

## Run it

From the repository root, install the locked dependencies once:

```bash
uv sync
```

Then run the complete experiment with one command:

```bash
uv run python experiments/001-capital-recall/run.py
```

The first run downloads only `EleutherAI/pythia-70m-deduped` through TransformerLens and Hugging Face. Later runs reuse the normal user cache. No model weights are written into this repository.

The model runs on CPU in inference mode. Prompts are processed sequentially to keep memory usage comfortable on an 8 GB M1 MacBook Air. Nothing is trained.

Generated artifacts are written to `outputs/experiment-001/`:

- `results.json` — complete machine-readable measurements and dependency versions;
- `report.html` — self-contained local report;
- `target-logits.svg` — comparison of target logits;
- `target-ranks.svg` — comparison of target ranks.

The terminal also prints prompt tokens, token IDs, the target token ID, every stage's measurements, the actual final prediction, numerical validation, and observations.

## Concepts

### Residual stream

A decoder-only transformer repeatedly updates a vector associated with each token position. This shared channel is called the residual stream. Each transformer block reads from it and adds an update. Experiment 001 examines the final prompt position before the first block and after each complete block.

### Logits

The model's final output is one score for every vocabulary token. These scores are logits. A larger logit gives a token more support relative to other tokens, but an individual logit's absolute value is not meaningful without the rest of the vocabulary distribution.

### Unembedding

The unembedding is the model's final mapping from a residual-stream vector to vocabulary logits. Pythia applies a final normalization before this mapping. Experiment 001 applies that same normalization independently to every accumulated residual state and then calls TransformerBridge's complete unembedding operation, including bias.

### Logit lens

A logit lens sends an intermediate residual-stream state through the model's final normalization and unembedding. This puts every stage into vocabulary space so the target token's logit and rank can be compared across layers.

The reported `logit_lens_probability` is softmax applied to one of these intermediate projected logit vectors. It is a convenient normalized view of the projection. It is not an actual prediction the model made at that intermediate stage.

Target rank is 1-based: rank 1 means no vocabulary token has a larger logit. The final stage is also compared directly against the actual next-token logits returned by the model. Target rank and top-1 token must agree. Raw numerical differences and any constant offset are recorded rather than hidden.

## How to read the graphs

The logit graph shows whether the vocabulary projection assigns the intended target a larger or smaller score after each block. The rank graph shows how many vocabulary alternatives score above the target, with rank 1 at the top. A star marks the actual final stage.

The report identifies the largest measured adjacent-stage increases, decreases, rank improvements, and rank deteriorations. These are descriptions of the recorded trajectory.

## What this does not prove

Experiment 001 is observational. Its graphs do not establish:

- that a particular layer stores a fact;
- that a particular layer retrieves a fact;
- that attention explains a measured change;
- that an attention head, MLP, neuron, or other component is responsible;
- that any component causally causes the final prediction.

Those questions require interventions such as activation patching or ablation and are outside this experiment.

## Small-model limitations

Pythia-70M may not make every intended capital token its top-1 next-token prediction. That outcome is recorded as a result, not treated as an implementation error. The runner still writes the trajectory and clearly names the model's actual prediction. It never switches to a larger model automatically.

By contrast, disagreement between the final logit-lens projection and the model's actual target rank or top-1 token is an implementation correctness failure. The runner stops instead of weakening or bypassing that validation.
