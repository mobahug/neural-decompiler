# Experiment 001 Capital Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-command, observational logit-lens experiment that measures how four capital-answer target tokens change rank and logit across Pythia-70M's residual stream.

**Architecture:** A small `neural_decompiler.logit_lens` module owns tensor projection, ranking, stage labels, and final-logit parity validation. The Experiment 001 runner owns prompts, model loading, deterministic observations, serialization, terminal output, and HTML/SVG reporting. The real model is loaded sequentially on CPU through raw `TransformerBridge`; tests exercise deterministic tensors and fakes without network access.

**Tech Stack:** Python 3.12 managed by uv, PyTorch, TransformerLens 3.9, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-experiment-001-capital-recall-design.md`

## Global Constraints

- The only automatic model is `EleutherAI/pythia-70m-deduped`; never select or download a larger model.
- Load with `TransformerBridge.boot_transformers(..., device="cpu")` and do not enable compatibility mode.
- Use normalized accumulated residual states and the bridge's complete unembedding operation, including bias.
- The actual final next-token logits are the source of truth; target-rank or top-1 disagreement is fatal.
- Run prompts sequentially under `torch.inference_mode()` and never train the model.
- Model weights use normal Hugging Face/TransformerLens caches and never enter the repository.
- Generated files under `outputs/experiment-001/` are gitignored.
- Intermediate softmax values are always named and described as logit-lens probabilities, not actual predictions.
- Observations remain descriptive and make no storage, retrieval, attention, component, or causal claims.

---

### Task 1: Python project and core metric contracts

**Files:**
- Create: `.python-version`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/neural_decompiler/__init__.py`
- Create: `src/neural_decompiler/logit_lens.py`
- Create: `tests/test_logit_lens.py`

**Interfaces:**
- Produces: `StageMetric`, `FinalProjectionValidation`, `stage_labels`, `metrics_from_logits`, and `validate_final_projection`.
- Consumes: only PyTorch tensors; no TransformerLens object is needed for these pure contracts.

- [ ] **Step 1: Add the failing tests for labels, metrics, and parity**

Create literal tensor fixtures that independently establish the expected values:

```python
import pytest
import torch

from neural_decompiler.logit_lens import (
    FinalProjectionMismatch,
    metrics_from_logits,
    stage_labels,
    validate_final_projection,
)


def test_stage_labels_describe_embedding_and_completed_layers():
    assert stage_labels(["0_pre", "1_pre", "2_pre", "final_post"], 3) == [
        "embedding",
        "after_layer_0",
        "after_layer_1",
        "after_layer_2",
    ]


def test_metrics_from_logits_uses_one_based_rank_and_softmax():
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    metrics = metrics_from_logits(logits, target_token_id=2, labels=["embedding", "after_layer_0"])
    assert [metric.target_rank for metric in metrics] == [2, 2]
    assert [metric.target_logit for metric in metrics] == [1.0, 2.0]
    assert metrics[0].logit_lens_probability == pytest.approx(0.244728, abs=1e-6)


def test_final_projection_records_constant_offset_without_hiding_it():
    projected = torch.tensor([1.0, 3.0, 2.0])
    actual = torch.tensor([1.25, 3.25, 2.25])
    result = validate_final_projection(projected, actual, target_token_id=2, atol=1e-6, rtol=1e-6)
    assert result.logits_allclose is False
    assert result.target_rank_matches is True
    assert result.top1_token_id_matches is True
    assert result.mean_actual_minus_projected_offset == pytest.approx(0.25)
    assert result.max_abs_difference_after_offset == pytest.approx(0.0)


@pytest.mark.parametrize("projected", [torch.tensor([3.0, 2.0, 1.0]), torch.tensor([1.0, 3.0, 2.0])])
def test_final_projection_rejects_rank_or_top1_disagreement(projected):
    actual = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(FinalProjectionMismatch):
        validate_final_projection(projected, actual, target_token_id=2)
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `uv run --python 3.12 pytest tests/test_logit_lens.py -v`

Expected: FAIL during collection because `neural_decompiler.logit_lens` does not exist.

- [ ] **Step 3: Add packaging and minimal implementations**

Use `.python-version` value `3.12`. Configure a `src` package and declare `torch`, `transformer-lens~=3.9.0`, and `matplotlib`; put pytest in a `dev` dependency group. Ignore `.venv/`, caches, and `outputs/experiment-001/*` while retaining an optional `.gitkeep`.

Implement frozen dataclasses with JSON-safe scalar fields. `metrics_from_logits` computes rank as `1 + count(logit > target_logit)` and softmax at the target. `validate_final_projection` computes raw and constant-offset-adjusted errors, records tolerance results, and raises `FinalProjectionMismatch` whenever either target rank or `argmax` differs.

- [ ] **Step 4: Run the focused tests and verify green**

Run: `uv run pytest tests/test_logit_lens.py -v`

Expected: all Task 1 tests PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add .python-version .gitignore pyproject.toml src/neural_decompiler tests/test_logit_lens.py uv.lock
git commit -m "feat: add logit lens metric contracts"
```

### Task 2: Raw TransformerBridge projection path

**Files:**
- Modify: `src/neural_decompiler/logit_lens.py`
- Modify: `tests/test_logit_lens.py`

**Interfaces:**
- Consumes: a bridge exposing `cfg.n_layers`, `to_single_token`, and callable `unembed`; a cache exposing `accumulated_resid(return_labels=True, apply_ln=True)`.
- Produces: `analyze_cached_prompt(model, cache, actual_final_logits, target_token_id) -> PromptProjection`.

- [ ] **Step 1: Add a failing projection test using complete unembedding**

Use lightweight real fakes whose unembedding adds a nonzero bias:

```python
class FakeCache:
    def accumulated_resid(self, *, return_labels, apply_ln):
        assert return_labels is True
        assert apply_ln is True
        residuals = torch.tensor([
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ])
        return residuals, ["0_pre", "final_post"]


class FakeBridge:
    cfg = SimpleNamespace(n_layers=1)

    def unembed(self, residual):
        weight = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
        bias = torch.tensor([0.5, -0.5, 1.0])
        return residual @ weight + bias


def test_analyze_cached_prompt_normalizes_then_uses_complete_unembedding():
    actual = torch.tensor([0.5, 2.5, 2.0])
    projection = analyze_cached_prompt(FakeBridge(), FakeCache(), actual, target_token_id=1)
    assert projection.stages[0].target_logit == -0.5
    assert projection.stages[1].target_logit == 2.5
    assert projection.validation.logits_allclose is True
```

This test catches replacing `model.unembed` with direct `@ W_U`, omitting bias, omitting `apply_ln=True`, or selecting the wrong token position.

- [ ] **Step 2: Run the focused test and verify the missing-function failure**

Run: `uv run pytest tests/test_logit_lens.py::test_analyze_cached_prompt_normalizes_then_uses_complete_unembedding -v`

Expected: FAIL because `analyze_cached_prompt` is not defined.

- [ ] **Step 3: Implement the bridge/cache analysis boundary**

Add `PromptProjection` and `analyze_cached_prompt`. Call only:

```python
normalized_stack, cache_labels = cache.accumulated_resid(
    return_labels=True,
    apply_ln=True,
)
last_position = normalized_stack[:, 0, -1, :]
stage_logits = model.unembed(last_position)
```

Then label stages, calculate metrics, validate `stage_logits[-1]` against the actual logits, and return CPU-native dataclasses. Do not call direct-logit-attribution helpers, access `W_U`, or enable compatibility mode.

- [ ] **Step 4: Run all core tests and verify green**

Run: `uv run pytest tests/test_logit_lens.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/neural_decompiler/logit_lens.py tests/test_logit_lens.py
git commit -m "feat: project normalized bridge residuals"
```

### Task 3: Experiment orchestration and deterministic results

**Files:**
- Create: `experiments/001-capital-recall/run.py`
- Create: `tests/test_experiment_001.py`

**Interfaces:**
- Consumes: `analyze_cached_prompt` and raw TransformerBridge methods `to_tokens`, `to_str_tokens`, `to_single_token`, `to_string`, and `run_with_cache`.
- Produces: `analyze_prompt`, `derive_observations`, `build_results`, `write_results_json`, `print_trace`, and `main` in the experiment runner.

- [ ] **Step 1: Add failing tests for target validation, observations, and JSON**

Dynamically import the runner by file path, then exercise behavior through a fake bridge. Cover these literal contracts:

```python
def test_analyze_prompt_rejects_a_multi_token_target(fake_bridge):
    fake_bridge.single_token_error = ValueError("not one token")
    with pytest.raises(ValueError, match="Target ' Paris' must tokenize to exactly one token"):
        runner.analyze_prompt(fake_bridge, "The capital of France is", " Paris")


def test_observations_report_measured_extrema_without_causal_language():
    stages = [
        StageMetric("embedding", 0.0, 10, 0.01),
        StageMetric("after_layer_0", 2.0, 3, 0.10),
        StageMetric("after_layer_1", 1.5, 5, 0.05),
    ]
    observations = runner.derive_observations(" Paris", stages)
    assert observations == [
        "The Paris target logit increased most strongly from embedding to after_layer_0 (+2.000000).",
        "The Paris target logit decreased most strongly from after_layer_0 to after_layer_1 (-0.500000).",
        "The Paris target rank improved most strongly from embedding to after_layer_0 (10 to 3).",
        "The Paris target rank worsened most strongly from after_layer_0 to after_layer_1 (3 to 5).",
    ]
    assert not ({"stored", "retrieved", "caused"} & set(" ".join(observations).lower().replace(".", "").split()))


def test_write_results_json_preserves_required_machine_readable_fields(tmp_path, sample_result):
    path = tmp_path / "results.json"
    runner.write_results_json(sample_result, path)
    decoded = json.loads(path.read_text())
    assert decoded["schema_version"] == "1.0"
    assert decoded["model"]["name"] == "EleutherAI/pythia-70m-deduped"
    assert decoded["prompts"][0]["target_token"]["text"] == " Paris"
    assert decoded["prompts"][0]["stages"][0]["label"] == "embedding"
    assert "final_projection_validation" in decoded["prompts"][0]
```

- [ ] **Step 2: Run the runner tests and verify failure because the runner is absent**

Run: `uv run pytest tests/test_experiment_001.py -v`

Expected: FAIL because `experiments/001-capital-recall/run.py` does not exist.

- [ ] **Step 3: Implement the experiment flow without loading during import**

Define the exact four prompt/target pairs and the model name as constants. Import `TransformerBridge` only inside `load_model`. The loader must be exactly the raw bridge path:

```python
def load_model():
    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers(DEFAULT_MODEL, device="cpu")
```

`analyze_prompt` must call `to_single_token` and wrap its failure with the experiment-specific message, call `run_with_cache` once under inference mode, pass `actual_logits[0, -1, :]` to `analyze_cached_prompt`, and serialize tokens and actual top-1 output. `build_results` runs prompts in input order and records dependency versions through `importlib.metadata`. `write_results_json` uses UTF-8, two-space indentation, and a trailing newline. `print_trace` prints every stage and any raw-logit or factual-recall warning.

- [ ] **Step 4: Run Tasks 1–3 tests and verify green**

Run: `uv run pytest -v`

Expected: all tests PASS without downloading a model.

- [ ] **Step 5: Commit Task 3**

```bash
git add experiments/001-capital-recall/run.py tests/test_experiment_001.py
git commit -m "feat: add experiment 001 analysis runner"
```

### Task 4: SVG and HTML report generation

**Files:**
- Modify: `experiments/001-capital-recall/run.py`
- Modify: `tests/test_experiment_001.py`

**Interfaces:**
- Consumes: the JSON-safe dictionary returned by `build_results`.
- Produces: `write_plots(result, output_dir)` and `write_html_report(result, output_dir)`.

- [ ] **Step 1: Add a failing real-artifact test**

```python
def test_report_writers_create_readable_svg_and_html(tmp_path, sample_result):
    runner.write_plots(sample_result, tmp_path)
    runner.write_html_report(sample_result, tmp_path)

    logits_svg = (tmp_path / "target-logits.svg").read_text()
    ranks_svg = (tmp_path / "target-ranks.svg").read_text()
    html = (tmp_path / "report.html").read_text()

    assert "<svg" in logits_svg
    assert "<svg" in ranks_svg
    assert "The capital of France is" in html
    assert "Intermediate logit-lens projection" in html
    assert "does not establish where a fact is stored or retrieved" in html
    assert "Final-projection validation" in html
```

The test catches missing artifacts, plotting only one prompt, or omitting the interpretation and validation notices.

- [ ] **Step 2: Run the artifact test and verify missing-function failure**

Run: `uv run pytest tests/test_experiment_001.py::test_report_writers_create_readable_svg_and_html -v`

Expected: FAIL because the report writers do not exist.

- [ ] **Step 3: Implement Matplotlib SVGs and a self-contained HTML report**

Use the noninteractive `Agg` backend. Plot all prompt series against the same ordered labels. Use a standard y-axis for logits and an inverted, logarithmic y-axis for 1-based ranks so rank 1 is prominent without flattening large early ranks. Mark each final point and label it as the actual final stage.

Build HTML with `html.escape`, inline the generated SVG bodies, and include tokenization, stage tables, factual-recall status, parity metrics, deterministic observations, and the non-causal interpretation notice. Do not add Jinja, Plotly, JavaScript, or a web server.

- [ ] **Step 4: Run the full offline suite and verify green**

Run: `uv run pytest -v`

Expected: all tests PASS and no model is downloaded.

- [ ] **Step 5: Commit Task 4**

```bash
git add experiments/001-capital-recall/run.py tests/test_experiment_001.py
git commit -m "feat: render experiment 001 report"
```

### Task 5: User documentation and end-to-end Pythia validation

**Files:**
- Create: `experiments/001-capital-recall/README.md`
- Modify: `README.md`
- Generated and ignored: `outputs/experiment-001/results.json`
- Generated and ignored: `outputs/experiment-001/report.html`
- Generated and ignored: `outputs/experiment-001/target-logits.svg`
- Generated and ignored: `outputs/experiment-001/target-ranks.svg`

**Interfaces:**
- Consumes: the completed runner and locked uv environment.
- Produces: the documented one-command workflow and measured Experiment 001 artifacts.

- [ ] **Step 1: Write the documentation**

Explain setup with `uv sync` and execution with:

```bash
uv run python experiments/001-capital-recall/run.py
```

Define residual stream, logits, unembedding, logit lens, 1-based target rank, and logit-lens probability. State prominently that the experiment is observational and does not establish storage, retrieval, attention explanations, component responsibility, or causation. Explain that Pythia-70M missing a target is reported rather than replaced.

- [ ] **Step 2: Run the full offline test suite before the download-enabled test**

Run: `uv run pytest -v`

Expected: all tests PASS without network access.

- [ ] **Step 3: Run the real experiment through the normal model cache**

Run: `uv run python experiments/001-capital-recall/run.py`

Expected: the bridge downloads or reuses only `EleutherAI/pythia-70m-deduped`, processes four prompts sequentially, prints every stage, passes rank/top-1 parity for every prompt, and writes all four required output files.

If target-rank or top-1 parity fails, stop and diagnose the raw bridge normalization/unembedding path. Do not loosen validation tolerances, enable compatibility mode, switch projection helpers, or continue to claim completion.

- [ ] **Step 4: Inspect measured artifacts**

Run:

```bash
python3 -m json.tool outputs/experiment-001/results.json >/dev/null
rg -n "WARNING|did not predict|Final-projection validation|does not establish" outputs/experiment-001/report.html
ls -lh outputs/experiment-001
```

Expected: valid JSON; explicit validation and interpretation language; factual-recall limitations shown when measured; no model weights under the repository.

- [ ] **Step 5: Run final verification**

Run:

```bash
uv run pytest -v
git status --short
find . -type f -size +10M -not -path './.git/*' -print
```

Expected: all tests PASS; only intended source/docs changes are uncommitted; no repository file exceeds 10 MB.

- [ ] **Step 6: Commit Task 5**

```bash
git add README.md experiments/001-capital-recall/README.md
git commit -m "docs: explain experiment 001"
```
