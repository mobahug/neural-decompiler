# Causal Instrumentation Foundation Design

## Purpose

Phase 1 adds the smallest trustworthy instrumentation layer needed to study a
naturally learned computation causally in a pinned Pythia checkpoint. It does
not choose that computation, create Experiment 005, or claim that any captured
component is a mechanism.

The new code is TransformerLens-first. It supports the existing
`TransformerBridge` path and uses TransformerLens hook semantics directly
instead of introducing a generalized model-backend framework.

## Scientific boundary

The instrumentation can measure and intervene on named activation sites. The
existence of a decomposition, a large activation, or an intervention effect is
evidence that an experiment may interpret only under its preregistered design,
controls, and competing explanations.

In particular:

- an ablation is not automatically evidence about the model's normal
  computation;
- attention patterns are measurements, not explanations;
- individual neurons are coordinates, not assumed human concepts;
- a logit-lens projection is not an intermediate model prediction;
- mean, reference, and resample replacement sources must be justified by the
  experiment that constructs them;
- all intervention claims remain scoped to the tested model, inputs, sites,
  positions, metrics, and replacement distribution.

## Existing history and compatibility

Experiments 001--004, their reports, the research methodology, the claim
registry, the anomaly registry, selection locks, and negative results remain
historical records. Phase 1 does not rewrite their protocols, result schemas,
or scientific conclusions.

The following existing modules remain public and unchanged unless a narrow bug
is found by the new tests:

- `logit_lens.py`
- `selectivity.py`
- `scale_comparison.py`
- `prompt_elicitation.py`

New experiments will use the new instrumentation modules instead of importing
functions from older experiment runner files. Migrating Experiments 001--004
onto the new APIs is explicitly out of scope.

The implementation starts from merged Experiment 004 commit
`ea65363521d46d50614b7dfb17cb9f52feb5e712`. The root README will be corrected
to list Experiment 004 and describe the new instrumentation boundary.

## Module map

```text
src/neural_decompiler/
    models.py          # pinned loading and explicit runtime selection
    components.py      # component vocabulary, hook resolution, decomposition
    capture.py         # selective activation capture
    interventions.py   # auditable zero and replacement interventions
    behavior.py        # task-independent behavior declarations
    provenance.py      # stable run metadata and JSON serialization

    logit_lens.py      # preserved observational analysis
    selectivity.py     # preserved observational analysis
    scale_comparison.py
    prompt_elicitation.py
```

There is no new `metrics.py` in Phase 1. The existing numerical modules already
provide metrics, and future behavioral work should add a metric only when an
actual experiment requires it.

## Model and runtime control

`models.py` defines an immutable `ModelSpec` containing:

- exact model identifier;
- exact revision;
- explicit device string;
- explicit dtype name.

The loader:

1. validates device and dtype before loading;
2. calls `TransformerBridge.boot_transformers` with the identifier, revision,
   device, and dtype explicitly;
3. requires compatibility mode to remain disabled;
4. verifies the resolved checkpoint revision against the requested revision;
5. puts the model in evaluation mode;
6. returns the model without enabling optional hook behavior implicitly.

The loader accepts an injected loader in tests so the offline suite never
downloads a model. Supported Phase 1 devices are CPU, MPS, `cuda`, and an
explicit `cuda:N` index when PyTorch reports them available. Unsupported or
unavailable devices fail before model loading.

A small seed helper sets Python and PyTorch seeds. Deterministic-algorithm mode
is an explicit runtime choice recorded in provenance rather than silently
enabled, because some device operations may not support it.

## Component vocabulary

`components.py` defines an immutable `ComponentRef` and a closed
`ComponentKind` enumeration for Phase 1:

- token embedding;
- positional embedding, when present;
- residual stream before a block;
- whole attention-block output;
- individual attention-head residual contribution;
- attention pattern;
- MLP post-activation neurons;
- individual MLP neuron;
- whole MLP-block output;
- residual stream after a block;
- final normalized residual / unembedding input;
- output logits.

Layer, head, and neuron indices are explicit fields with kind-specific
validation. A component that does not exist for the loaded model, such as a
learned positional embedding in a rotary-only model, fails rather than
returning an empty tensor.

Resolution maps these references to documented TransformerLens hook names and
records tensor-axis semantics. Individual head contributions use
`blocks.L.attn.hook_result`, whose tensor is the per-head contribution in
residual-stream coordinates. This hook requires TransformerLens
`use_attn_result`; requests that need it must opt into an explicit
instrumentation setting. The capture or intervention runner refuses a head
request when that setting is disabled.

Attention-pattern requests are capture-only in Phase 1. Their position filter
means query positions and may separately name key positions. Logits and final
normalization are also capture-only. Phase 1 interventions operate only on
well-defined additive or activation sites.

Pure decomposition helpers provide:

- summing per-head residual contributions into the whole attention output;
- reconstructing a block output from residual input plus additive attention
  and MLP outputs;
- numerical validation with caller-supplied absolute and relative tolerances.

These helpers report reconstruction error and never label the decomposition a
mechanism.

## Selective capture

`capture.py` defines:

- `CaptureRequest`, containing a `ComponentRef` and explicit position
  selection;
- `CapturePlan`, containing one or more unique requests and explicit optional
  hook settings;
- `CapturedActivation`, containing the request, original tensor shape, and a
  detached clone of the selected tensor;
- `CaptureResult`, containing model output logits and captured activations.

Capture uses only the resolved requested hooks. Hook callbacks slice requested
positions, heads, neurons, query positions, and key positions before cloning,
so selecting one site does not retain a full-model activation cache.

Position indices follow Python negative-index rules after sequence length is
known. Out-of-range indices fail. Request order is retained for serialization,
but duplicate requests fail to avoid ambiguous results.

Captured tensors preserve dtype. The caller selects whether stored captures
remain on the execution device or move to CPU; CPU storage is the default for
research artifacts. The move is explicit in the plan and recorded in the
result metadata.

The returned logits are the same output produced during the instrumented
forward pass. An empty plan must be numerically identical to a direct forward
pass.

## Interventions

`interventions.py` defines immutable intervention declarations and a runner.
Two execution operations exist in Phase 1:

- `ZERO`: clone the activation and replace only the selected slice with zeros;
- `REPLACE`: clone the activation and replace only the selected slice with an
  exact-shape tensor.

Replacement provenance distinguishes:

- direct provided activation;
- reference-case activation;
- mean activation;
- resampled activation.

These labels document where a replacement came from. Phase 1 does not estimate
means, choose reference distributions, or resample datasets automatically.
Experiments construct those tensors under their own frozen protocols and pass
them to the common replacement executor.

Replacement tensors must match the selected slice exactly in shape and dtype.
No broadcasting or dtype coercion is allowed. A replacement may be transferred
to the activation's device explicitly by the runner, and that transfer is
recorded; its dtype is not silently changed.

Multiple interventions may share a hook only when their selected slices are
provably disjoint. Duplicate or overlapping selections fail before execution.
Unsupported sites and missing required hook configuration also fail before
execution.

The runner returns logits plus an intervention execution record containing the
resolved hook, selected axes, operation, replacement source, shape, dtype, and
device. It does not interpret the resulting behavioral change.

## Behavior declarations

`behavior.py` defines compact, JSON-safe declarations rather than a full task
framework:

- `BehaviorCase`: stable case ID, input text or explicit input token IDs,
  target positions, expected outputs when applicable, control or
  counterfactual references, and JSON-safe metadata;
- `MetricSpec`: stable metric name plus JSON-safe parameters;
- `BehaviorSpec`: stable behavior ID and version, cases, metric, inclusion
  criteria, and success criteria.

Case IDs must be unique. Each case must provide exactly one input form. Empty
or non-serializable criteria and parameters fail validation. The declaration
does not execute a metric; the next experiment owns metric implementation and
must preregister how its declared metric is calculated.

This separates reusable behavior descriptions from country--capital-specific
code without prematurely designing a universal task engine.

## Provenance and artifacts

`provenance.py` provides stable dataclasses and canonical JSON serialization
for:

- requested and resolved model identity;
- device, dtype, seed, and deterministic-algorithm setting;
- Python, PyTorch, TransformerLens, Transformers, and Hugging Face Hub
  versions;
- Git commit and dirty state when obtainable;
- exact input text, token strings, token IDs, and target positions;
- capture and intervention definitions;
- replacement source metadata and tensor digests;
- JSON-safe raw measurements needed to reproduce derived claims.

JSON writers use UTF-8, sorted keys, stable separators for hashing, and a
trailing newline for human-readable files. Non-finite floats and live tensors
are rejected from JSON records. Large activation tensors are not silently
expanded into JSON; a future experiment may store them as a separately named
artifact with path, shape, dtype, and SHA-256 digest.

Phase 1 creates no database and no global experiment registry.

## Error behavior

The instrumentation fails before execution when possible. Errors name the
invalid component, field, or shape and include the allowed range. It rejects:

- missing or mutable model revisions;
- unavailable devices or unknown dtypes;
- invalid layer, head, neuron, or position indices;
- unsupported component/hook combinations;
- duplicate capture requests;
- overlapping interventions;
- absent gated TransformerLens hooks;
- replacement shape or dtype mismatch;
- non-finite decomposition inputs or measurements;
- non-JSON-safe behavior or provenance fields.

No error path weakens validation, changes an intervention target, chooses a
different model, or falls back to another component silently.

## Testing strategy

All implementation follows red-green-refactor test-driven development. The
default suite remains offline and does not download model weights.

Pure deterministic tensor tests cover:

- component validation and hook resolution;
- head-sum and residual reconstruction identities;
- shape, finite-value, and tolerance failures;
- behavior validation;
- canonical provenance serialization and digests.

A tiny deterministic hookable transformer fixture covers end-to-end semantics:

- a captured tensor is from the requested component, layer, and position;
- selective capture does not retain unrequested sites;
- direct forward and empty-plan logits match;
- a same-activation replacement is a no-op within strict tolerance;
- zeroing or replacing a known site changes only the requested activation
  slice before downstream propagation;
- position-specific, head-specific, and neuron-specific selections isolate the
  requested slice;
- output logits remain unchanged when no intervention is active;
- invalid or overlapping declarations fail before forward execution.

Model loading tests use an injected fake loader. A separately marked cached
Pythia-70M smoke test may be added later, but Phase 1 completion does not depend
on model downloads or CUDA.

## Documentation

The root README will:

- list Experiment 004;
- state that Experiments 001--004 are historical observational/infrastructure
  work;
- link to a concise instrumentation guide;
- state that no target for Experiment 005 has been selected.

`docs/INSTRUMENTATION.md` will document component meanings, capture shape
semantics, intervention replacement rules, configuration requirements, and a
small non-domain-specific example. It will repeat that decomposition and
intervention effects require experimental interpretation.

## Deliberate exclusions

Phase 1 does not implement:

- Experiment 005 or behavior selection;
- sparse autoencoders;
- automated circuit discovery or search;
- automated mean/reference dataset construction;
- causal scrubbing or path patching;
- training-checkpoint sweeps;
- generalized model-family adapters;
- dashboards, databases, notebooks, or interactive visualization;
- migration or regeneration of Experiments 001--004;
- claims, anomaly records, or novelty statements.

## Prerequisites before Experiment 005

After Phase 1 passes review, a separate research task must:

1. screen candidate behaviors that Pythia-70M performs reliably;
2. reject candidates unlikely to admit a compact complete account;
3. review prior literature for already-mapped mechanisms;
4. freeze inputs, inclusion rules, behavior metrics, and success thresholds;
5. state competing mechanisms and discriminating interventions;
6. reserve held-out inputs and intervention predictions before inspection;
7. define necessity, sufficiency, specificity, control, reconstruction, and
   residual-unexplained-behavior criteria;
8. only then preregister Experiment 005.

