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


def test_parser_exposes_every_phase_and_no_override_flags():
    parser = runner_module.build_parser()
    for phase in ("validate", "freeze-extension", "discover", "calibrate", "revise", "lock", "confirm", "report"):
        assert parser.parse_args([phase]).phase == phase
    with pytest.raises(SystemExit):
        parser.parse_args(["confirm", "--force"])
    with pytest.raises(SystemExit):
        parser.parse_args(["behavioral"])


# ---------------------------------------------------------------------------
# discover phase on a fake model


from plural_fakes import TinyPlural  # noqa: E402


def _stub_a1(ctx, screening_results_path):
    return {"stubbed": True, "screening_results_compared": 0}


def make_discover_runner(sandbox, monkeypatch, *, contract_passed=True, logs=None):
    logs = logs if logs is not None else []
    monkeypatch.setattr(pm, "a1_baseline", _stub_a1)
    # A random fake has no real cue effect on the manifest prompts; lift the denominator floors for plumbing.
    monkeypatch.setattr(pm, "DENOMINATOR_FLOOR_OVERALL", -1e9)
    monkeypatch.setattr(pm, "DENOMINATOR_FLOOR_STRATUM", -1e9)
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


# ---------------------------------------------------------------------------
# calibrate → lock → confirm → report on the fake (floors forced to pass for the state machine)


def _force_pass(real):
    def wrapped(results, *, hypothesis, n_cases):
        floors = real(results, hypothesis=hypothesis, n_cases=n_cases)
        floors["primary_failures"] = []
        floors["precondition_passed"] = True
        floors["circuit_passed"] = True
        return floors
    return wrapped


def test_full_state_machine_on_the_fake(sandbox, monkeypatch):
    runner, logs = make_discover_runner(sandbox, monkeypatch)
    runner.changed_paths = lambda commit: []
    # A random fake cannot pass scientific floors; lift the Tier A floors so the state machine can be exercised.
    monkeypatch.setattr(pm, "TIER_A_RECOVERY_FLOOR_OVERALL", -10.0)
    monkeypatch.setattr(pm, "TIER_A_RECOVERY_FLOOR_STRATUM", -10.0)
    monkeypatch.setattr(pm, "TIER_A_ISOLATION_FLOOR", -10.0)
    monkeypatch.setattr(pm, "program_development_floors", lambda program, ctx: {"passed": True, "sign_failures": [], "templates": {}})
    assert runner.freeze_extension() == 0
    assert runner.discover() == 0
    state = pm.load_results_state(runner.results_path)
    assert state["mechanism_versions"][-1]["status"] == "candidate"
    with pytest.raises(pm.PhaseError):
        runner.lock()  # needs a calibration pass first
    with pytest.raises(pm.PhaseError):
        runner.confirm()
    monkeypatch.setattr(pm, "circuit_floors", _force_pass(pm.circuit_floors))
    assert runner.calibrate() == 0
    state = pm.load_results_state(runner.results_path)
    assert len(state["calibration"]["passes"]) == 1 and state["calibration"]["passes"][0]["floors_passed"]
    assert len([key for key in state["executed_noun_keys"] if key.startswith("future-reserve")]) == 0
    with pytest.raises(pm.PhaseError):
        runner.revise()  # floors passed: nothing to revise
    with pytest.raises(pm.PhaseError):
        runner.calibrate()  # a second pass needs a revise phase
    assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    lock = json.loads(candidate.read_text())
    assert lock["content_sha256"] == pm.sha256_text(pm.canonical_json({k: v for k, v in lock.items() if k != "content_sha256"}))
    assert set(lock["x_predictions"]["cue_words"]) == set(pm.EXTENSION_CUE_WORDS[:12])
    assert len(lock["x_predictions"]["new_frames"]) == 6
    with pytest.raises(pm.PhaseError):
        runner.confirm()  # lock not installed yet
    installed = sandbox / pm.LOCK_RELATIVE_PATH
    shutil.copy(candidate, installed)
    tampered = json.loads(installed.read_text())
    tampered["bands"]["P1"]["overall"]["low"] -= 1.0
    installed.write_text(json.dumps(tampered))
    with pytest.raises(pm.PhaseError):
        runner.confirm()  # digest mismatch
    shutil.copy(candidate, installed)
    runner.changed_paths = lambda commit: ["src/neural_decompiler/plural_mechanism.py"]
    with pytest.raises(pm.PhaseError):
        runner.confirm()  # scientific path changed since the lock commit
    runner.changed_paths = lambda commit: ["README.md"]
    assert runner.confirm() == 0
    state = pm.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "complete"
    assert state["confirmation"]["outcome"]["label"] in {"MECHANISM_CONFIRMED", "MECHANISM_SUPPORTED_MISCALIBRATED", "CIRCUIT_ONLY", "CIRCUIT_NOT_GENERALIZED", "MECHANISM_CONTESTED", "MECHANISM_NOT_SUPPORTED", "BEHAVIOR_NOT_REPLICATED"}
    assert len([key for key in state["executed_noun_keys"] if key.startswith("future-reserve")]) == 20
    assert len(state["executed_prompt_keys"]) == 12 + 12 + 72
    with pytest.raises(pm.PhaseError):
        runner.confirm()  # once only
    assert runner.report() == 0
    text = runner.report_path.read_text()
    assert "## Confirmation — outcome" in text and "### Cue words" in text


def test_continuation_after_program_floor_rejection(sandbox, monkeypatch):
    """Protocol v2: a discover that fails only the program floor is adopted with PROGRAM_CAPPED and runs to confirm."""
    runner, logs = make_discover_runner(sandbox, monkeypatch)
    runner.changed_paths = lambda commit: []
    monkeypatch.setattr(pm, "TIER_A_RECOVERY_FLOOR_OVERALL", -10.0)
    monkeypatch.setattr(pm, "TIER_A_RECOVERY_FLOOR_STRATUM", -10.0)
    monkeypatch.setattr(pm, "TIER_A_ISOLATION_FLOOR", -10.0)
    monkeypatch.setattr(pm, "CONTINUATION_ISOLATION_TEMPLATE_FLOOR", -10.0)
    monkeypatch.setattr(pm, "program_development_floors", lambda program, ctx: {"passed": False, "sign_failures": [], "templates": {"cardinal": {"gap": 2.0}}})
    assert runner.freeze_extension() == 0
    with pytest.raises(pm.PhaseError):
        runner.continue_v2()  # needs discover first
    assert runner.discover() == 0
    state = pm.load_results_state(runner.results_path)
    assert state["mechanism_versions"][-1]["status"] == "rejected"
    with pytest.raises(pm.PhaseError):
        runner.calibrate()  # no candidate version
    assert runner.continue_v2() == 0
    state = pm.load_results_state(runner.results_path)
    version = state["mechanism_versions"][-1]
    assert version["version"] == "M2" and version["continuation"] and version["program_capped"] and version["k"] >= 2
    assert version["adopted_attempt_k"] == version["k"] and version["continuation_rule"] == pm.CONTINUATION_RULE_ID
    assert "protocol v1 program floor failed" in version["statement"] and "contextual encoding" not in version["statement"]
    assert version["retention_diagnostic"]["note"].startswith("diagnostic only")
    assert state["phases"]["continue"]["status"] == "complete"
    assert "a5_final" in state["discovery"]
    with pytest.raises(pm.PhaseError):
        runner.continue_v2()  # already adopted under the current rule
    # A version adopted under an older rule is superseded, not deleted, when the current rule selects another k.
    older = pm.load_results_state(runner.results_path)
    older["mechanism_versions"][-1]["continuation_rule"] = "revision-5"
    older["mechanism_versions"][-1]["adopted_attempt_k"] = -1
    pm.write_results_state(runner.results_path, older)
    assert runner.continue_v2() == 0
    state = pm.load_results_state(runner.results_path)
    assert [entry["status"] for entry in state["mechanism_versions"]] == ["rejected", "superseded", "candidate"]
    assert state["mechanism_versions"][-1]["version"] == "M3" and state["phases"]["continue"]["history"]
    monkeypatch.setattr(pm, "circuit_floors", _force_pass(pm.circuit_floors))
    assert runner.calibrate() == 0
    with pytest.raises(pm.PhaseError):
        runner.revise()  # frozen in the continuation
    assert runner.lock() == 0
    lock = json.loads((runner.results_path.parent / "candidate-lock.json").read_text())
    assert lock["protocol_version"] == 2 and lock["continuation"]["continuation_of"] == "M2" and lock["mechanism"]["program_capped"] is True
    shutil.copy(runner.results_path.parent / "candidate-lock.json", sandbox / pm.LOCK_RELATIVE_PATH)
    assert runner.confirm() == 0
    state = pm.load_results_state(runner.results_path)
    verdict = state["confirmation"]["outcome"]
    assert verdict["axes"]["decompilation"] == "PROGRAM_FAIL"
    assert verdict["label"] in {"CIRCUIT_ONLY", "CIRCUIT_NOT_GENERALIZED", "MECHANISM_NOT_SUPPORTED", "BEHAVIOR_NOT_REPLICATED"}
    assert state["confirmation"]["decompilation_floors"]["program_capped"] is True
