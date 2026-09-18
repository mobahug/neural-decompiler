"""Phase isolation, extension freeze, and non-execution tests for the Experiment 005 runner."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from neural_decompiler import plural_mechanism as pm
from neural_decompiler.candidate_screening import load_manifest

from test_plural_mechanism import _toy_tokenizer

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / pm.MANIFEST_RELATIVE_PATH


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_005_runner", ROOT / "experiments/005-regular-plural-mechanism/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()
manifest = load_manifest(MANIFEST_PATH)


@pytest.fixture
def sandbox(tmp_path):
    """A root with the real manifest and an empty experiment directory."""
    (tmp_path / "screening/behavior-candidates").mkdir(parents=True)
    shutil.copy(MANIFEST_PATH, tmp_path / pm.MANIFEST_RELATIVE_PATH)
    (tmp_path / "experiments/005-regular-plural-mechanism").mkdir(parents=True)
    return tmp_path


def make_runner(sandbox, *, tracked=True, logs=None):
    logs = logs if logs is not None else []
    return runner_module.Runner(
        root=sandbox,
        experiment_dir=sandbox / "experiments/005-regular-plural-mechanism",
        results_path=sandbox / "outputs/experiment-005/results.json",
        report_path=sandbox / "outputs/experiment-005/report.md",
        model_loader=lambda spec: pytest.fail("no model may load in this phase"),
        tokenizer_loader=lambda spec: _toy_tokenizer(manifest),
        git_state=lambda: {"commit": "a" * 40, "dirty": False},
        versions=lambda: {"torch": "test"},
        tracked=lambda path: tracked,
        log=logs.append,
    ), logs


def test_validate_reports_missing_extension(sandbox):
    runner, logs = make_runner(sandbox)
    assert runner.validate() == 1
    assert any("not frozen" in line for line in logs)


def test_freeze_then_validate_and_refuse_overwrite(sandbox):
    runner, logs = make_runner(sandbox)
    assert runner.freeze_extension() == 0
    extension_path = sandbox / pm.EXTENSION_RELATIVE_PATH
    payload = json.loads(extension_path.read_text())
    assert payload["content_sha256"] == pm.extension_content_digest(payload)
    assert runner.validate() == 0
    assert runner.freeze_extension() == 1
    assert json.loads(extension_path.read_text()) == payload


def test_validate_requires_tracked_extension(sandbox):
    runner, _ = make_runner(sandbox, tracked=False)
    assert runner.freeze_extension() == 0
    assert runner.validate() == 1


def test_parser_exposes_only_implemented_phases():
    parser = runner_module.build_parser()
    assert parser.parse_args(["validate"]).phase == "validate"
    assert parser.parse_args(["freeze-extension"]).phase == "freeze-extension"
    with pytest.raises(SystemExit):
        parser.parse_args(["confirm"])


# ---------------------------------------------------------------------------
# discover phase on a fake model


from plural_fakes import TinyPlural  # noqa: E402


def _stub_a1(ctx, screening_results_path):
    return {"stubbed": True, "screening_results_compared": 0}


def make_discover_runner(sandbox, monkeypatch, *, contract_passed=True, logs=None):
    logs = logs if logs is not None else []
    monkeypatch.setattr(pm, "a1_baseline", _stub_a1)
    # A random fake has no real cue effect on the manifest prompts; lift the denominator floors for plumbing.
    monkeypatch.setattr(pm, "DENOMINATOR_FLOOR_OVERALL", 0.0)
    monkeypatch.setattr(pm, "DENOMINATOR_FLOOR_STRATUM", 0.0)
    runner = runner_module.Runner(
        root=sandbox,
        experiment_dir=sandbox / "experiments/005-regular-plural-mechanism",
        results_path=sandbox / "outputs/experiment-005/results.json",
        report_path=sandbox / "outputs/experiment-005/report.md",
        parameters_dir=sandbox / "outputs/experiment-005/parameters",
        program_path=ROOT / "experiments/005-regular-plural-mechanism/mechanism_program.py",
        model_loader=lambda spec: TinyPlural(seed=3, d_vocab=50304),
        tokenizer_loader=lambda spec: _toy_tokenizer(manifest),
        git_state=lambda: {"commit": "a" * 40, "dirty": False},
        versions=lambda: {"torch": "test"},
        tracked=lambda path: True,
        contract_runner=lambda: {"passed": contract_passed, "returncode": 0 if contract_passed else 1, "output_tail": "", "output_sha256": "x"},
        log=logs.append,
    )
    return runner, logs


def test_discover_refuses_without_frozen_extension(sandbox, monkeypatch):
    runner, _ = make_discover_runner(sandbox, monkeypatch)
    with pytest.raises(pm.PhaseError):
        runner.discover()


def test_discover_refuses_when_contract_test_fails(sandbox, monkeypatch):
    runner, logs = make_discover_runner(sandbox, monkeypatch, contract_passed=False)
    assert runner.freeze_extension() == 0
    assert runner.discover() == 1
    assert any("A0 contract test failed" in line for line in logs)
    assert not runner.results_path.exists()


def test_discover_runs_once_and_never_touches_reserve_or_extension(sandbox, monkeypatch):
    runner, logs = make_discover_runner(sandbox, monkeypatch)
    assert runner.freeze_extension() == 0
    assert runner.discover() == 0
    state = pm.load_results_state(runner.results_path)
    assert state["phases"]["discover"]["status"] == "complete"
    assert state["discovery"]["a0_contract_test"]["passed"] is True
    assert state["discovery"]["a1"] == {"stubbed": True, "screening_results_compared": 0}
    assert len(state["executed_prompt_keys"]) == 12
    reserve = {noun.key for noun in pm.nouns_for(manifest, pm.Split.FUTURE_RESERVE)}
    assert not reserve & set(state["executed_noun_keys"])
    assert len(state["executed_noun_keys"]) == 40
    assert state["mechanism_versions"][0]["version"] == "M1"
    with pytest.raises(pm.PhaseError):
        runner.discover()
