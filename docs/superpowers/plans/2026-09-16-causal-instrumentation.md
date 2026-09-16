# Causal Instrumentation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, auditable TransformerBridge-first capture and intervention layer for pinned Pythia-70M without changing Experiments 001--004 or selecting Experiment 005.

**Architecture:** Immutable declarations (`ModelSpec`, `ComponentRef`, `CaptureRequest`, `Intervention`, `BehaviorSpec`, and provenance records) are validated before a single inference-only forward. Canonical TransformerBridge hook names are resolved centrally; capture and intervention runners share exact slice semantics, while Pythia-specific decomposition helpers verify attention bias and GPT-NeoX parallel residual identities explicitly.

**Tech Stack:** Python 3.10--3.12, PyTorch 2.x, TransformerLens 3.9.x `TransformerBridge`, pytest 8.x, standard-library dataclasses/enum/json/hashlib.

**Spec:** `docs/superpowers/specs/2026-09-16-causal-instrumentation-design.md`

## Global Constraints

- Keep TransformerBridge compatibility mode disabled; use canonical Bridge hooks only.
- The pinned Phase 1 checkpoint is `EleutherAI/pythia-70m-deduped` revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`.
- CPU float32 is the correctness/reference path; CUDA is never required by the offline suite.
- Every capture/intervention forward runs in evaluation mode under `torch.inference_mode()`.
- Use `set_use_attn_result(True)` for per-head results, restore its prior value, and never mutate Bridge hook flags directly.
- Replacement is exact-shape and exact-dtype; no broadcasting or implicit dtype coercion.
- Preserve Experiments 001--004 and their result/history formats unchanged.
- Do not add SAEs, path patching, Q/K/V decomposition, automated circuit search, dashboards, databases, or Experiment 005.
- Follow red-green-refactor for every implementation task and keep the default pytest run fully offline.

## File map

- `src/neural_decompiler/models.py`: pinned model declarations, device/dtype-pair validation, deterministic runtime setup, Bridge loading.
- `src/neural_decompiler/components.py`: component vocabulary, canonical hook resolution, normalized selectors, Pythia decomposition identities.
- `src/neural_decompiler/capture.py`: selective capture declarations and inference runner.
- `src/neural_decompiler/interventions.py`: exact zero/replacement declarations, overlap checks, inference runner, execution records.
- `src/neural_decompiler/behavior.py`: JSON-safe behavior/case/metric declarations only.
- `src/neural_decompiler/provenance.py`: stable provenance records, canonical JSON, digests, Git/runtime collection.
- `tests/instrumentation_fakes.py`: deterministic hookable Bridge-shaped fixture shared only by tests.
- `tests/test_models.py`, `tests/test_components.py`, `tests/test_capture.py`, `tests/test_interventions.py`, `tests/test_behavior.py`, `tests/test_provenance.py`: offline semantic tests.
- `tests/test_pythia_bridge_contract.py`: explicitly selected live-model contract test.
- `docs/INSTRUMENTATION.md`, `README.md`, `src/neural_decompiler/__init__.py`, `pyproject.toml`: public documentation, exports, and smoke marker.

---

### Task 1: Pinned model and runtime control

**Files:**
- Create: `src/neural_decompiler/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces: `DTypeName`, `ModelSpec`, `RuntimeInfo`, `PYTHIA_70M`, `resolve_dtype(name)`, `validate_runtime(spec)`, `seed_runtime(seed, deterministic_algorithms)`, `resolved_revision(model)`, `load_model(spec, loader=None)`.
- `load_model` calls an injected loader with keyword arguments `revision`, `device`, and `dtype`; later tasks consume the returned Bridge-shaped object.

- [ ] **Step 1: Write failing model/runtime tests**

```python
def test_pinned_pythia_spec_is_immutable_and_exact():
    assert PYTHIA_70M.model_id == "EleutherAI/pythia-70m-deduped"
    assert PYTHIA_70M.revision == "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"

def test_load_model_passes_explicit_runtime_and_requires_raw_bridge():
    calls = []
    fake = make_fake_bridge(revision=PYTHIA_70M.revision, compatibility_mode=False)
    loaded = load_model(PYTHIA_70M, loader=lambda name, **kw: calls.append((name, kw)) or fake)
    assert calls == [(PYTHIA_70M.model_id, {"revision": PYTHIA_70M.revision,
        "device": "cpu", "dtype": torch.float32})]
    assert loaded is fake and fake.training is False

@pytest.mark.parametrize("device,dtype", [("cpu", "float16"), ("mps", "bfloat16")])
def test_invalid_device_dtype_pairs_fail_before_loader(device, dtype):
    with pytest.raises(ValueError, match="device/dtype"):
        validate_runtime(dataclasses.replace(PYTHIA_70M, device=device, dtype=dtype))
```

- [ ] **Step 2: Run tests to verify the module is missing**

Run: `.venv/bin/pytest tests/test_models.py -q`

Expected: collection fails with `ModuleNotFoundError: neural_decompiler.models`.

- [ ] **Step 3: Implement the minimal pinned loader**

```python
@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    device: str = "cpu"
    dtype: str = "float32"
    deterministic_algorithms: bool = False

PYTHIA_70M = ModelSpec(
    "EleutherAI/pythia-70m-deduped",
    "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
)

def load_model(spec: ModelSpec, loader=None):
    runtime = validate_runtime(spec)
    if loader is None:
        from transformer_lens.model_bridge import TransformerBridge
        loader = TransformerBridge.boot_transformers
    model = loader(spec.model_id, revision=spec.revision,
                   device=runtime.device, dtype=runtime.torch_dtype)
    if resolved_revision(model) != spec.revision:
        raise RuntimeError("Pinned model revision mismatch")
    if bool(getattr(model, "compatibility_mode", False)):
        raise RuntimeError("TransformerBridge compatibility mode must remain disabled")
    model.eval()
    return model
```

Validate `cpu/float32`, available `mps/float32`, and available CUDA with `float32`, `float16`, or `bfloat16`; reject CPU float16 and MPS bfloat16 in Phase 1. Perform device availability checks without allocating model weights. Return a frozen `RuntimeInfo(device, backend, dtype_name, torch_dtype)`.

- [ ] **Step 4: Run model tests and the existing suite**

Run: `.venv/bin/pytest tests/test_models.py -q && .venv/bin/pytest -q`

Expected: model tests pass and the historical suite remains at least 73 passing tests.

- [ ] **Step 5: Commit the runtime foundation**

```bash
git add src/neural_decompiler/models.py tests/test_models.py
git commit -m "feat: add pinned Bridge runtime control"
```

### Task 2: Component vocabulary, selectors, and decomposition

**Files:**
- Create: `src/neural_decompiler/components.py`
- Create: `tests/test_components.py`

**Interfaces:**
- Consumes: `ModelSpec` only for model-family checking.
- Produces: `ComponentKind`, `ComponentRef`, `ResolvedComponent`, `AxisSelection`, `resolve_component(ref, model)`, `normalize_positions(positions, sequence_length)`, `selection_index(component, selection, sequence_length)`, `selections_overlap(left, right, sequence_length)`, `AttentionDecomposition`, `decompose_attention(per_head, output_bias, measured, *, atol, rtol)`, `BlockDecomposition`, `decompose_pythia_parallel_block(resid_pre, attn_out, mlp_out, measured, *, architecture, parallel_attn_mlp, atol, rtol)`.
- Canonical hook names are returned exactly as specified in the design; aliases are rejected if supplied externally.

- [ ] **Step 1: Write failing validation and resolution tests**

```python
def test_component_resolution_uses_canonical_bridge_hooks():
    model = fake_geometry(n_layers=6, n_heads=8, d_mlp=2048, positional_embedding_type="rotary")
    assert resolve_component(ComponentRef(ComponentKind.RESID_PRE, layer=2), model).hook_name == "blocks.2.hook_in"
    assert resolve_component(ComponentRef(ComponentKind.ATTN_HEAD, layer=2, head=3), model).hook_name == "blocks.2.attn.hook_result"
    assert resolve_component(ComponentRef(ComponentKind.MLP_NEURON, layer=2, neuron=17), model).hook_name == "blocks.2.mlp.out.hook_in"

def test_negative_positions_normalize_before_overlap():
    assert normalize_positions((-1,), 5) == (4,)
    assert selections_overlap(position_only((-1,)), position_only((4,)), sequence_length=5)

def test_attention_decomposition_includes_output_bias():
    heads = torch.arange(24.0).reshape(1, 2, 3, 4)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
    measured = heads.sum(dim=2) + bias
    result = decompose_attention(heads, bias, measured, atol=0.0, rtol=0.0)
    assert torch.equal(result.head_sum, heads.sum(dim=2))
    assert torch.equal(result.reconstructed, measured)

def test_block_decomposition_rejects_non_parallel_architecture():
    tensor = torch.zeros(1, 2, 4)
    with pytest.raises(ValueError, match="parallel GPT-NeoX"):
        decompose_pythia_parallel_block(
            tensor, tensor, tensor, tensor,
            architecture="GPTNeoXForCausalLM",
            parallel_attn_mlp=False,
            atol=0.0,
            rtol=0.0,
        )
```

- [ ] **Step 2: Run component tests to verify failure**

Run: `.venv/bin/pytest tests/test_components.py -q`

Expected: collection fails because `neural_decompiler.components` does not exist.

- [ ] **Step 3: Implement closed component and selector semantics**

```python
class ComponentKind(str, Enum):
    TOKEN_EMBED = "token_embed"
    POSITION_EMBED = "position_embed"
    RESID_PRE = "resid_pre"
    ATTN_OUT = "attn_out"
    ATTN_HEAD = "attn_head"
    ATTN_PATTERN = "attn_pattern"
    MLP_POST = "mlp_post"
    MLP_NEURON = "mlp_neuron"
    MLP_OUT = "mlp_out"
    RESID_POST = "resid_post"
    FINAL_NORM = "final_norm"
    LOGITS = "logits"

@dataclass(frozen=True)
class ComponentRef:
    kind: ComponentKind
    layer: int | None = None
    head: int | None = None
    neuron: int | None = None
```

Encode each kind's required/forbidden fields and axis tuple. Resolve geometry from `model.cfg`; fail for missing hooks/components, invalid indices, or Pythia rotary positional embedding requests. `AxisSelection` contains `positions` plus optional `key_positions`; head/neuron selection comes from `ComponentRef`. Normalize every position only after the input sequence length is known.

- [ ] **Step 4: Implement bias-aware and architecture-declared decompositions**

```python
@dataclass(frozen=True)
class AttentionDecomposition:
    per_head: torch.Tensor
    head_sum: torch.Tensor
    output_bias: torch.Tensor
    reconstructed: torch.Tensor
    measured: torch.Tensor
    max_abs_error: float

def decompose_attention(per_head, output_bias, measured, *, atol, rtol):
    _require_finite(per_head, output_bias, measured)
    head_sum = per_head.sum(dim=-2)
    reconstructed = head_sum + output_bias
    _require_same_shape_and_close(reconstructed, measured, atol=atol, rtol=rtol)
    return AttentionDecomposition(per_head, head_sum, output_bias,
                                  reconstructed, measured,
                                  (reconstructed - measured).abs().max().item())
```

For blocks, require `architecture == "GPTNeoXForCausalLM"` (or the loaded config's exact equivalent) and `parallel_attn_mlp is True`; reconstruct `resid_pre + attn_out + mlp_out`, require equal shapes/finite values, and assert closeness to measured `resid_post`.

- [ ] **Step 5: Run focused and historical tests**

Run: `.venv/bin/pytest tests/test_components.py -q && .venv/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit component semantics**

```bash
git add src/neural_decompiler/components.py tests/test_components.py
git commit -m "feat: define canonical components and decompositions"
```

### Task 3: Selective inference-only capture

**Files:**
- Create: `tests/instrumentation_fakes.py`
- Create: `src/neural_decompiler/capture.py`
- Create: `tests/test_capture.py`

**Interfaces:**
- Consumes: `ComponentRef`, `AxisSelection`, `ResolvedComponent`, `resolve_component`, and position normalization from Task 2.
- Produces: `InstrumentationSettings(use_attn_result=False)`, `CaptureRequest`, `CapturePlan`, `CapturedActivation`, `CaptureResult`, `run_capture(model, inputs, plan, *, prepend_bos=False)`.
- The fake model exposes `cfg`, `hook_dict`, `set_use_attn_result`, `eval`, `__call__`, and `run_with_hooks` with canonical hook names.

- [ ] **Step 1: Create the deterministic hookable fixture and failing capture tests**

```python
def test_empty_capture_is_exact_direct_forward_and_uses_inference_mode(tiny_bridge, tokens):
    direct = tiny_bridge(tokens)
    result = run_capture(tiny_bridge, tokens, CapturePlan(()))
    assert torch.equal(result.logits, direct)
    assert tiny_bridge.last_grad_enabled is False

def test_capture_slices_before_storage(tiny_bridge, tokens):
    request = CaptureRequest(ComponentRef(ComponentKind.RESID_PRE, layer=1), positions=(2,))
    result = run_capture(tiny_bridge, tokens, CapturePlan((request,)))
    capture = result.activations[request]
    assert capture.original_shape == (1, 4, tiny_bridge.cfg.d_model)
    assert capture.tensor.shape == (1, 1, tiny_bridge.cfg.d_model)
    assert tiny_bridge.fired_hooks == {"blocks.1.hook_in"}

def test_head_capture_uses_setter_and_restores_setting(tiny_bridge, tokens):
    request = CaptureRequest(ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=1), positions=(0,))
    run_capture(tiny_bridge, tokens, CapturePlan((request,), InstrumentationSettings(True)))
    assert tiny_bridge.setter_calls == [True, False]
    assert tiny_bridge.cfg.use_attn_result is False
```

The fixture should produce deterministic `[batch, pos, d_model]`, `[batch, pos, head, d_model]`, `[batch, head, query, key]`, and `[batch, pos, d_mlp]` tensors, call only registered hooks, and expose whether gradients were enabled during forward.

- [ ] **Step 2: Run capture tests to verify failure**

Run: `.venv/bin/pytest tests/test_capture.py -q`

Expected: collection fails because `neural_decompiler.capture` does not exist.

- [ ] **Step 3: Implement plan validation and hook-scoped execution**

```python
@dataclass(frozen=True)
class CaptureRequest:
    component: ComponentRef
    positions: tuple[int, ...] | None = None
    key_positions: tuple[int, ...] | None = None

@dataclass(frozen=True)
class CapturePlan:
    requests: tuple[CaptureRequest, ...]
    settings: InstrumentationSettings = InstrumentationSettings()
    storage_device: str = "cpu"
```

Reject duplicate requests and incompatible key-position filters before forward. Resolve only requested hooks. In each callback, normalize positions against the live tensor's sequence axes, slice first, then `detach().clone()` and optionally move to CPU. Keep request order stable.

- [ ] **Step 4: Implement the inference runner and scoped setter**

Use `model.eval()`, `torch.inference_mode()`, and `model.run_with_hooks(inputs, fwd_hooks=hook_pairs, prepend_bos=prepend_bos)`, where `hook_pairs` is the exact list built from the resolved requests. When head results are requested, require `settings.use_attn_result`, save `bool(model.cfg.use_attn_result)`, call `model.set_use_attn_result(True)`, and restore with the setter in `finally`. Return the forward logits and effective settings; never call `run_with_cache`.

- [ ] **Step 5: Run capture, component, and full offline tests**

Run: `.venv/bin/pytest tests/test_capture.py tests/test_components.py -q && .venv/bin/pytest -q`

Expected: all tests pass, and capture retains no unrequested activation.

- [ ] **Step 6: Commit selective capture**

```bash
git add tests/instrumentation_fakes.py src/neural_decompiler/capture.py tests/test_capture.py
git commit -m "feat: add selective activation capture"
```

### Task 4: Exact interventions and isolation

**Files:**
- Create: `src/neural_decompiler/interventions.py`
- Create: `tests/test_interventions.py`

**Interfaces:**
- Consumes: component resolution/selector functions and `InstrumentationSettings` from Tasks 2--3.
- Produces: `InterventionOperation`, `ReplacementSource`, `Intervention`, `InterventionPlan`, `InterventionExecution`, `InterventionResult`, `run_interventions(model, inputs, plan, *, prepend_bos=False)`.

- [ ] **Step 1: Write failing no-op, isolation, and validation tests**

```python
def test_same_activation_replacement_is_noop(tiny_bridge, tokens):
    target = ComponentRef(ComponentKind.MLP_POST, layer=0)
    baseline = run_capture(tiny_bridge, tokens, CapturePlan((CaptureRequest(target, (1,)),)))
    replacement = baseline.activations[next(iter(baseline.activations))].tensor
    result = run_interventions(tiny_bridge, tokens, InterventionPlan((
        Intervention(target, (1,), InterventionOperation.REPLACE,
                     replacement, ReplacementSource.DIRECT),)))
    assert torch.equal(result.logits, baseline.logits)

def test_replacement_changes_only_selected_slice_at_site(tiny_bridge, tokens):
    target = ComponentRef(ComponentKind.MLP_NEURON, layer=0, neuron=2)
    replacement = torch.tensor([[[99.0]]], dtype=torch.float32)
    plan = InterventionPlan((Intervention(
        target, (1,), InterventionOperation.REPLACE,
        replacement, ReplacementSource.DIRECT,
    ),))
    result = run_interventions(tiny_bridge, tokens, plan)
    before, after = result.executions[0].before, result.executions[0].after
    mask = torch.ones_like(before, dtype=torch.bool); mask[:, 1, 2] = False
    assert torch.equal(before[mask], after[mask])
    assert after[:, 1, 2].item() == 99.0

def test_equivalent_negative_and_positive_positions_overlap(tiny_bridge, tokens):
    plan = InterventionPlan((zero_at(-1), zero_at(tokens.shape[1] - 1)))
    with pytest.raises(ValueError, match="overlap"):
        run_interventions(tiny_bridge, tokens, plan)
```

Also test exact shape/dtype failures, duplicate sites, capture-only components, head-setting refusal, and no-hook/no-intervention parity.

- [ ] **Step 2: Run intervention tests to verify failure**

Run: `.venv/bin/pytest tests/test_interventions.py -q`

Expected: collection fails because `neural_decompiler.interventions` does not exist.

- [ ] **Step 3: Implement immutable declarations and preflight validation**

```python
class InterventionOperation(str, Enum):
    ZERO = "zero"
    REPLACE = "replace"

class ReplacementSource(str, Enum):
    DIRECT = "direct"
    REFERENCE = "reference"
    MEAN = "mean"
    RESAMPLE = "resample"

@dataclass(frozen=True)
class Intervention:
    component: ComponentRef
    positions: tuple[int, ...]
    operation: InterventionOperation
    replacement: torch.Tensor | None = None
    source: ReplacementSource | None = None
```

Require no replacement/source for zero and both for replacement. Resolve the actual sequence length from tensor input before grouping by hook and checking disjointness. Reject logits, final norm, and attention patterns as intervention sites in Phase 1.

- [ ] **Step 4: Implement exact mutation callbacks and execution records**

Each callback clones the full activation, extracts the exact selected slice, validates replacement shape/dtype, explicitly transfers the replacement device if necessary, assigns only that slice, and records hook, normalized axes, before/after selected slices, shape, dtype, device, transfer flag, operation, and source. Use the same scoped head setting and inference semantics as capture.

- [ ] **Step 5: Run intervention and full offline suites**

Run: `.venv/bin/pytest tests/test_interventions.py tests/test_capture.py -q && .venv/bin/pytest -q`

Expected: all tests pass; the no-op and isolation assertions are exact on the deterministic fixture.

- [ ] **Step 6: Commit interventions**

```bash
git add src/neural_decompiler/interventions.py tests/test_interventions.py
git commit -m "feat: add auditable activation interventions"
```

### Task 5: Behavior declarations

**Files:**
- Create: `src/neural_decompiler/behavior.py`
- Create: `tests/test_behavior.py`

**Interfaces:**
- Produces: `MetricSpec`, `BehaviorCase`, `BehaviorSpec`, each with `to_dict()` returning stable JSON-safe Python values.

- [ ] **Step 1: Write failing behavior validation tests**

```python
def test_behavior_is_domain_independent_and_serializable():
    case = BehaviorCase("case-1", input_text="A B", target_positions=(-1,),
                        expected_outputs=(" C",), metadata={"split": "held-out"})
    spec = BehaviorSpec("copy-next", "1", (case,), MetricSpec("target_logit", {}),
                        inclusion_criteria=("single-token target",),
                        success_criteria=("positive held-out margin",))
    json.dumps(spec.to_dict(), allow_nan=False)

@pytest.mark.parametrize("case", [
    BehaviorCase("x", input_text="a", input_token_ids=(1,)),
    BehaviorCase("x"),
])
def test_case_requires_exactly_one_input_form(case):
    with pytest.raises(ValueError, match="exactly one"):
        case.validate()
```

Also reject duplicate case IDs, empty criteria, live tensors, sets, non-string mapping keys, and non-finite floats.

- [ ] **Step 2: Run behavior tests to verify failure**

Run: `.venv/bin/pytest tests/test_behavior.py -q`

Expected: missing-module collection failure.

- [ ] **Step 3: Implement compact JSON-safe dataclasses**

Use frozen dataclasses, defensive mapping copies, recursive JSON-safety validation, exact-one input validation, and unique IDs. Keep metric execution absent by design.

- [ ] **Step 4: Run behavior and full suites**

Run: `.venv/bin/pytest tests/test_behavior.py -q && .venv/bin/pytest -q`

Expected: all pass.

- [ ] **Step 5: Commit behavior declarations**

```bash
git add src/neural_decompiler/behavior.py tests/test_behavior.py
git commit -m "feat: add behavior specifications"
```

### Task 6: Provenance and canonical artifacts

**Files:**
- Create: `src/neural_decompiler/provenance.py`
- Create: `tests/test_provenance.py`

**Interfaces:**
- Consumes: JSON-safe dictionaries produced by earlier declarations.
- Produces: `TokenizerProvenance`, `RuntimeProvenance`, `InputProvenance`, `RunProvenance`, `tensor_digest(tensor)`, `canonical_json(value)`, `write_json(path, value)`, `collect_versions()`, `collect_git_state(path)`.

- [ ] **Step 1: Write failing serialization and completeness tests**

```python
def test_canonical_provenance_is_stable_and_complete(tmp_path):
    run = complete_run_provenance()
    first = canonical_json(run.to_dict())
    second = canonical_json(run.to_dict())
    assert first == second
    assert json.loads(first)["tokenizer"]["prepend_bos"] is False
    assert json.loads(first)["bridge"]["use_attn_result"] is True
    path = tmp_path / "run.json"
    write_json(path, run.to_dict())
    assert path.read_bytes().endswith(b"\n")

def test_tensor_digest_includes_shape_dtype_and_bytes():
    assert tensor_digest(torch.zeros(2, dtype=torch.float32)) != tensor_digest(torch.zeros(2, dtype=torch.float64))

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), torch.tensor(1), {1: "x"}])
def test_non_json_artifacts_fail_loudly(bad):
    with pytest.raises((TypeError, ValueError)):
        canonical_json({"bad": bad})
```

- [ ] **Step 2: Run provenance tests to verify failure**

Run: `.venv/bin/pytest tests/test_provenance.py -q`

Expected: missing-module collection failure.

- [ ] **Step 3: Implement stable records and serialization**

`TokenizerProvenance` contains tokenizer ID/revision, BOS/EOS token IDs and strings, prepend-BOS flag, padding/truncation settings. `InputProvenance` retains exact text, token IDs, token strings, and target positions. `RuntimeProvenance` contains device, backend, dtype, seed, deterministic algorithms, atol/rtol, dependency versions, compatibility mode, and optional hook settings. Serialize with `sort_keys=True`, `ensure_ascii=False`, `allow_nan=False`, and compact separators for hashes; append one newline only in `write_json`.

- [ ] **Step 4: Implement digests and best-effort Git/version collection**

Detach tensors to contiguous CPU memory before hashing a header containing dtype and shape plus raw bytes. `collect_git_state` runs read-only `git rev-parse HEAD` and `git status --porcelain`, returning nullable fields when outside Git. Use `importlib.metadata.version` for dependency versions.

- [ ] **Step 5: Run provenance and full suites**

Run: `.venv/bin/pytest tests/test_provenance.py -q && .venv/bin/pytest -q`

Expected: all pass and canonical output is byte-stable.

- [ ] **Step 6: Commit provenance**

```bash
git add src/neural_decompiler/provenance.py tests/test_provenance.py
git commit -m "feat: add reproducible instrumentation provenance"
```

### Task 7: Pinned Pythia TransformerBridge contract test

**Files:**
- Create: `tests/test_pythia_bridge_contract.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: all runtime, component, capture, intervention, and decomposition APIs.
- Produces: pytest marker `pythia_smoke`; no production API.

- [ ] **Step 1: Register the marker and write the opt-in live test**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "pythia_smoke: downloads or loads the pinned Pythia-70M TransformerBridge contract fixture",
]
```

```python
pytestmark = pytest.mark.pythia_smoke

@pytest.fixture(scope="module")
def pythia_bridge():
    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")
    return load_model(PYTHIA_70M)

def test_pythia_bridge_contract(pythia_bridge):
    tokens = pythia_bridge.to_tokens("The quick brown fox", prepend_bos=False)
    direct = pythia_bridge(tokens, prepend_bos=False)
    requests = (
        CaptureRequest(ComponentRef(ComponentKind.TOKEN_EMBED), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.RESID_PRE, layer=0), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.ATTN_OUT, layer=0), (0,)),
        *(CaptureRequest(ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=head), (0,))
          for head in range(pythia_bridge.cfg.n_heads)),
        CaptureRequest(ComponentRef(ComponentKind.ATTN_PATTERN, layer=0, head=0), (0,), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.MLP_POST, layer=0), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.MLP_NEURON, layer=0, neuron=0), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.MLP_OUT, layer=0), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.RESID_POST, layer=0), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.FINAL_NORM), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.LOGITS), (0,)),
    )
    result = run_capture(
        pythia_bridge, tokens,
        CapturePlan(requests, InstrumentationSettings(use_attn_result=True)),
        prepend_bos=False,
    )
    assert torch.allclose(result.logits, direct, atol=1e-5, rtol=1e-5)
    assert torch.equal(result.logits.argmax(dim=-1), direct.argmax(dim=-1))
    assert result.activations[requests[0]].tensor.shape == (1, 1, 512)
    assert result.activations[requests[3]].tensor.shape == (1, 1, 1, 512)
    assert result.activations[requests[12]].tensor.shape == (1, 1, 2048)
    assert result.activations[requests[13]].tensor.shape == (1, 1, 1)

    captured_heads = torch.cat(
        tuple(result.activations[request].tensor for request in requests[3:11]),
        dim=2,
    )
    decompose_attention(
        captured_heads,
        pythia_bridge.blocks[0].attn.b_O,
        result.activations[requests[2]].tensor,
        atol=1e-5,
        rtol=1e-5,
    )
    decompose_pythia_parallel_block(
        result.activations[requests[1]].tensor,
        result.activations[requests[2]].tensor,
        result.activations[requests[14]].tensor,
        result.activations[requests[15]].tensor,
        architecture=pythia_bridge.original_model.config.architectures[0],
        parallel_attn_mlp=pythia_bridge.cfg.parallel_attn_mlp,
        atol=1e-5,
        rtol=1e-5,
    )
    noop = run_interventions(
        pythia_bridge,
        tokens,
        InterventionPlan((Intervention(
            ComponentRef(ComponentKind.MLP_POST, layer=0),
            (0,),
            InterventionOperation.REPLACE,
            result.activations[requests[12]].tensor,
            ReplacementSource.DIRECT,
        ),)),
        prepend_bos=False,
    )
    assert torch.allclose(noop.logits, direct, atol=1e-5, rtol=1e-5)
```

Use direct same-process logits as baseline with CPU float32 `atol=1e-5`, `rtol=1e-5`; assert top-1 equality as an additional discrete check. Read `b_O` from `pythia_bridge.blocks[0].attn.b_O`. Assert `cfg.parallel_attn_mlp is True`, `cfg.n_layers == 6`, and `cfg.n_heads == 8`.

- [ ] **Step 2: Prove the default suite skips without loading**

Run: `.venv/bin/pytest -q`

Expected: all offline tests pass with one smoke skip and no network/model load.

- [ ] **Step 3: Run the explicit CPU smoke test when weights are available**

Run: `NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 .venv/bin/pytest tests/test_pythia_bridge_contract.py -m pythia_smoke -q`

Expected: contract passes. If the model is not cached and network use is unavailable, record the smoke test as not run; Phase 1 may be code-complete but must be reported as not scientifically ready, and Experiment 005 remains blocked.

- [ ] **Step 4: Commit the live adapter contract**

```bash
git add pyproject.toml tests/test_pythia_bridge_contract.py
git commit -m "test: add pinned Pythia Bridge contract"
```

### Task 8: Public documentation, exports, and final compatibility verification

**Files:**
- Create: `docs/INSTRUMENTATION.md`
- Modify: `README.md`
- Modify: `src/neural_decompiler/__init__.py`

**Interfaces:**
- Consumes: final public APIs from Tasks 1--7.
- Produces: importable documented surface and migration guidance; no new execution logic.

- [ ] **Step 1: Add an import-surface test before exporting**

Add to the most relevant tests:

```python
def test_public_instrumentation_surface_imports():
    from neural_decompiler import (BehaviorSpec, CapturePlan, CaptureRequest,
        ComponentRef, Intervention, ModelSpec, PYTHIA_70M, RunProvenance)
```

Run: `.venv/bin/pytest tests/test_models.py::test_public_instrumentation_surface_imports -q`

Expected: fail until exports are added.

- [ ] **Step 2: Export only stable Phase 1 declaration types**

Update `__init__.py` with explicit imports and `__all__`; keep lower-level helpers in their modules. Do not re-export experiment-specific historical functions.

- [ ] **Step 3: Write the instrumentation guide**

Document the canonical hook table, tensor axes, raw-Bridge/compatibility policy, scoped `use_attn_result`, inference-only semantics, exact replacement rules, bias-aware attention decomposition, Pythia parallel block equation, provenance requirements, CPU smoke command, and the distinction between an intervention effect and a mechanistic explanation. Include a small synthetic-token example, not a country/capital example.

- [ ] **Step 4: Correct and extend the root README**

List Experiment 004, label Experiments 001--004 as historical observational/infrastructure work, link `docs/INSTRUMENTATION.md`, state that Experiment 005 and its target behavior have not been selected, and retain existing setup commands.

- [ ] **Step 5: Run formatting-independent checks and the complete offline suite**

Run: `git diff --check && .venv/bin/pytest -q`

Expected: no whitespace errors; all tests pass and the live Pythia test is skipped unless explicitly enabled.

- [ ] **Step 6: Verify historical experiment compatibility directly**

Run: `.venv/bin/pytest tests/test_experiment_001.py tests/test_experiment_002.py tests/test_experiment_003.py tests/test_experiment_004.py -q`

Expected: every historical experiment test passes without changes to experiment runners or records.

- [ ] **Step 7: Review scope and commit documentation**

Run: `git diff --stat ea65363521d46d50614b7dfb17cb9f52feb5e712..HEAD && git status --short`

Confirm no Experiment 005 directory, behavior choice, SAE/path-patching code, database, or historical result rewrite exists.

```bash
git add README.md docs/INSTRUMENTATION.md src/neural_decompiler/__init__.py tests/test_models.py
git commit -m "docs: describe causal instrumentation foundation"
```

### Task 9: Completion verification and scientific-readiness report

**Files:**
- Modify only if verification exposes a defect covered by the approved spec.

**Interfaces:**
- Produces: verified branch state and an evidence-backed completion report.

- [ ] **Step 1: Run static compilation and the full offline suite from a clean process**

Run: `.venv/bin/python -m compileall -q src tests experiments && .venv/bin/pytest -q`

Expected: compilation succeeds; all offline tests pass; only the opt-in smoke test is skipped.

- [ ] **Step 2: Run the exact-model contract if available**

Run: `NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 .venv/bin/pytest tests/test_pythia_bridge_contract.py -m pythia_smoke -q`

Expected: pass on CPU float32. If unavailable, report that exact gap verbatim and do not call the instrumentation scientifically ready.

- [ ] **Step 3: Inspect the final diff and repository state**

Run: `git diff --check ea65363521d46d50614b7dfb17cb9f52feb5e712..HEAD && git status --short --branch && git log --oneline --decorate -10`

Expected: clean working tree on `codex/causal-instrumentation`, cohesive commits, no history rewrite.

- [ ] **Step 4: Report against every requested deliverable**

Summarize old architecture, preserved history, module map, newly enabled causal capabilities, exact tests/results, deliberate exclusions, Experiments 001--004 compatibility, prerequisites for Experiment 005, README inconsistency correction, and whether the live Pythia contract passed. Never imply that the instrumentation itself identifies a mechanism.
