# Causal Instrumentation

This document defines the Phase 1 instrumentation boundary for future Neural
Decompiler experiments. The implementation is deliberately Pythia-first and
TransformerBridge-first. It measures and changes named activation sites; it
does not infer a mechanism by itself.

## Research purpose

The long-term research objective is to produce a progressively stronger causal
account of one simple, naturally learned computation in Pythia-70M, and only
then test whether the account generalizes to a larger or independently trained
model. A complete account should eventually:

1. localize the computation;
2. identify what information relevant components read and write;
3. characterize the transformations performed by attention, MLP, and residual
   components;
4. establish necessity and sufficiency under matched interventions;
5. distinguish competing mechanistic explanations;
6. make preregistered predictions about unseen inputs and interventions;
7. reconstruct or simplify the circuit while retaining the behavior; and
8. quantify residual unexplained behavior.

Phase 1 only supplies trustworthy measurement, intervention, and provenance
primitives. The target behavior and Experiment 005 remain deliberately
unselected.

## Pinned runtime

`models.py` pins `EleutherAI/pythia-70m-deduped` to an immutable Hugging Face
revision. CPU float32 is the correctness reference. MPS and CUDA are optional
execution backends whose device/dtype pair must validate before loading.

TransformerBridge compatibility mode must remain disabled. This preserves raw
Hugging Face numerics and prevents experiments from depending on legacy hook
aliases. Models run in evaluation mode, and every Phase 1 capture or
intervention forward runs under `torch.inference_mode()`. Gradient attribution
is not part of this execution path.

## Canonical component sites

All internal resolution uses canonical raw-Bridge hooks:

| Component | Canonical hook | Tensor axes |
| --- | --- | --- |
| token embedding | `embed.hook_out` | batch, position, d_model |
| positional embedding, when present | `pos_embed.hook_out` | batch, position, d_model |
| residual before block L | `blocks.L.hook_in` | batch, position, d_model |
| whole attention contribution | `blocks.L.attn.hook_out` | batch, position, d_model |
| per-head residual contribution | `blocks.L.attn.hook_result` | batch, position, head, d_model |
| attention pattern | `blocks.L.attn.hook_pattern` | batch, head, query position, key position |
| MLP post-activation neurons | `blocks.L.mlp.out.hook_in` | batch, position, neuron |
| whole MLP contribution | `blocks.L.mlp.hook_out` | batch, position, d_model |
| residual after block L | `blocks.L.hook_out` | batch, position, d_model |
| final normalized residual | `ln_final.hook_out` | batch, position, d_model |
| logits | `unembed.hook_out` | batch, position, vocabulary |

Pythia uses rotary position information, so requesting an additive positional
embedding fails explicitly. Per-head residual contributions require
`use_attn_result`; a plan must request it explicitly. The runner calls
`set_use_attn_result(True)`, records the effective state, and restores the
previous state in a `finally` path.

## Selective capture

A `CaptureRequest` names one `ComponentRef` and optional positions. Attention
patterns may also select key positions. Python-style negative positions are
normalized only after the sequence length is known; out-of-range and duplicate
normalized positions fail.

Callbacks slice before cloning, so a narrow request does not retain a full
activation cache. Captures are detached and stored on CPU by default, while
preserving dtype and the original hook shape. An empty capture plan is a direct
inference forward and must return identical logits.

Example:

```python
from neural_decompiler.capture import CapturePlan, CaptureRequest, run_capture
from neural_decompiler.components import ComponentKind, ComponentRef

request = CaptureRequest(
    ComponentRef(ComponentKind.RESID_PRE, layer=0),
    positions=(-1,),
)
result = run_capture(model, token_ids, CapturePlan((request,)))
last_position_residual = result.activations[request].tensor
```

## Interventions

Phase 1 supports two execution operations:

- `ZERO` replaces exactly the selected slice with zeros;
- `REPLACE` writes an exact-shape, exact-dtype tensor.

Replacement provenance distinguishes direct, reference, mean, and resampled
sources. Those labels describe how an experiment obtained a tensor; the common
runner does not choose reference cases, estimate means, or sample datasets.

Replacement tensors never broadcast and never receive implicit dtype
conversion. A device transfer is allowed only as an explicit runner action and
is recorded. Multiple mutations on one hook must be provably disjoint after
negative positions are normalized. Attention patterns, final normalization,
and logits are capture-only sites in Phase 1.

Each execution record contains the resolved hook, normalized positions,
operation, source, selected shape, dtype, device, transfer flag, before/after
slices, and the maximum change outside the requested slice.

## Pythia decomposition checks

The per-head `hook_result` tensors are residual-stream contributions before
the attention output bias. Attention reconstruction is therefore:

```text
head_sum = sum(per_head_results, over=head)
reconstructed_attention = head_sum + output_bias
```

The helper reports the individual head results, their sum, the output bias,
the reconstruction, the independently measured attention output, and maximum
absolute error. Omitting a present bias is an error.

Pythia-70M uses GPT-NeoX parallel residuals. Its supported block identity is:

```text
resid_post = resid_pre + attention_output + mlp_output
```

The helper verifies the declared architecture, parallel-residual flag, tensor
shapes, finiteness, and caller-provided absolute/relative tolerances. It fails
for unsupported architectures instead of applying a generic serial-block
equation.

A reconstruction identity shows that tensors add as declared. It is not proof
that the components implement the proposed behavioral mechanism.

## Behavior and provenance

`BehaviorSpec` describes cases, inputs, target positions/outputs, controls,
counterfactuals, a named metric, inclusion criteria, and success criteria. It
does not execute the metric and contains no country/capital assumptions.

`RunProvenance` retains the requested and resolved model revision, tokenizer
identity and revision, BOS/EOS and tokenization settings, exact token IDs and
strings, device/backend/dtype, seed, deterministic-algorithm state,
TransformerLens and dependency versions, compatibility mode,
`use_attn_result`, tolerances, Git state, capture/intervention definitions,
replacement metadata, tensor digests, and JSON-safe raw measurements.

Canonical JSON uses UTF-8, sorted keys, stable separators, finite values, and a
single trailing newline when written to disk. Large tensors are referenced by
shape, dtype, and digest rather than silently expanded into JSON.

## Verification

The normal suite is offline:

```bash
uv run pytest -q
```

The live adapter contract is opt-in:

```bash
NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 \
  uv run pytest tests/test_pythia_bridge_contract.py -m pythia_smoke -q
```

It loads the exact pinned Pythia-70M revision on CPU float32 and verifies direct
output parity, canonical hook firing, declared tensor shapes, explicit
per-head capture, MLP neuron capture, same-activation no-op replacement,
bias-aware attention reconstruction, and the parallel-residual block identity.
This test must pass before Experiment 005 is designed.

## Scientific boundary

These APIs make causal tests possible; they do not make an experiment causal
by declaration. A future experiment must freeze the behavior, cases, controls,
metrics, tolerances, intervention sources, competing explanations, predicted
outcomes, and success criteria before examining decisive results. Negative and
contradictory results remain part of the scientific record.
