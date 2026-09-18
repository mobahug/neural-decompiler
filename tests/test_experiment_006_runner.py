"""Phase isolation, confirmation freeze, and the full state machine for the Experiment 006 runner (fake model)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import plural_mechanism as pm
from neural_decompiler.candidate_screening import load_manifest
from plural_fakes import TinyPlural
from test_cue_decompilation import toy_tokenizer_006

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / pm.MANIFEST_RELATIVE_PATH
EXTENSION_PATH = ROOT / pm.EXTENSION_RELATIVE_PATH
FAKE_CIRCUIT = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L01.H00",), ("L01.MLP",))


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_006_runner", ROOT / "experiments/006-low-rank-cue-decompilation/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()
manifest = load_manifest(MANIFEST_PATH)


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "screening/behavior-candidates").mkdir(parents=True)
    shutil.copy(MANIFEST_PATH, tmp_path / pm.MANIFEST_RELATIVE_PATH)
    (tmp_path / "experiments/005-regular-plural-mechanism").mkdir(parents=True)
    shutil.copy(EXTENSION_PATH, tmp_path / pm.EXTENSION_RELATIVE_PATH)
    (tmp_path / "experiments/006-low-rank-cue-decompilation").mkdir(parents=True)
    return tmp_path


def make_runner(sandbox, monkeypatch, *, logs=None):
    logs = logs if logs is not None else []
    monkeypatch.setattr(pm, "DENOMINATOR_FLOOR_OVERALL", -1e9)
    monkeypatch.setattr(pm, "DENOMINATOR_FLOOR_STRATUM", -1e9)
    runner = runner_module.Runner(
        root=sandbox,
        results_path=sandbox / "outputs/experiment-006/results.json",
        report_path=sandbox / "outputs/experiment-006/report.md",
        parameters_dir=sandbox / "outputs/experiment-006/parameters",
        program_path=ROOT / "experiments/006-low-rank-cue-decompilation/low_rank_program.py",
        program_005_path=ROOT / "experiments/005-regular-plural-mechanism/mechanism_program.py",
        lock_005_path=sandbox / "missing-005-lock.json",
        circuit=FAKE_CIRCUIT,
        model_loader=lambda spec: TinyPlural(seed=3, d_vocab=60000),
        tokenizer_loader=lambda spec: toy_tokenizer_006(manifest),
        git_state=lambda: {"commit": "a" * 40, "dirty": False},
        versions=lambda: {"torch": "test"},
        tracked=lambda path: True,
        changed_paths=lambda commit: [],
        contract_runner=lambda: {"passed": True, "returncode": 0, "output_tail": "", "output_sha256": "x"},
        log=logs.append,
    )
    return runner, logs


def test_validate_and_freeze(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.validate() == 1
    assert runner.freeze_confirmation() == 0
    assert runner.validate() == 0
    assert runner.freeze_confirmation() == 1
    with pytest.raises(cd.PhaseError):
        runner.calibrate()  # no state yet


def test_parser_has_every_phase_and_no_overrides():
    parser = runner_module.build_parser()
    for phase in ("validate", "freeze-confirmation", "explore", "calibrate", "lock", "confirm", "report"):
        assert parser.parse_args([phase]).phase == phase
    with pytest.raises(SystemExit):
        parser.parse_args(["confirm", "--force"])


def test_full_state_machine_on_the_fake(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    monkeypatch.setattr(cd, "RANKS", (1, 2))
    assert runner.freeze_confirmation() == 0
    confirmation = cd.load_confirmation(runner.confirmation_path, manifest, pm.manifest_digest_from_path(runner.root / pm.MANIFEST_RELATIVE_PATH),
                                        pm.load_extension(runner.root / pm.EXTENSION_RELATIVE_PATH, manifest, pm.manifest_digest_from_path(runner.root / pm.MANIFEST_RELATIVE_PATH)))
    assert runner.explore() == 0
    state = cd.load_results_state(runner.results_path)
    assert state["phases"]["explore"]["status"] == "complete"
    assert state["exploration"]["selection"]["selected"] in (1, 2)
    assert set(state["exploration"]["loco"]) == {"1", "2"} and len(state["exploration"]["loco"]["1"]) == 16
    assert len(state["exploration"]["epatch"]) == 12 * 16
    assert not {noun.key for noun in confirmation.nouns} & set(state["executed_noun_keys"])
    assert not {prompt.key for prompt in confirmation.all_prompts} & set(state["executed_prompt_keys"])
    with pytest.raises(cd.PhaseError):
        runner.explore()  # once
    assert runner.calibrate() == 0
    if not state["exploration"]["quality_gate"]["passed"]:
        with pytest.raises(cd.PhaseError):
            runner.lock()  # the gate blocks the lock
        forced = cd.load_results_state(runner.results_path)
        forced["exploration"]["quality_gate"]["passed"] = True
        cd.write_results_state(runner.results_path, forced)
    assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    lock = json.loads(candidate.read_text())
    assert set(lock["parameters"]) == {"selected", "r1-pca", "e005-scalar"}
    assert len(lock["predictions"]["tokens"]) == 24 and len(lock["predictions"]["pairs"]) == 6
    with pytest.raises(cd.PhaseError):
        runner.confirm()  # lock not installed
    installed = sandbox / cd.LOCK_RELATIVE_PATH
    shutil.copy(candidate, installed)
    runner.changed_paths = lambda commit: ["src/neural_decompiler/cue_decompilation.py"]
    with pytest.raises(cd.PhaseError):
        runner.confirm()
    runner.changed_paths = lambda commit: ["README.md"]
    assert runner.confirm() == 0
    state = cd.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "complete"
    assert state["confirmation"]["outcome"]["label"] in {"DECOMPILED", "DECOMPILED_MISCALIBRATED", "CIRCUIT_ONLY", "CIRCUIT_NOT_GENERALIZED", "NOT_SUPPORTED", "CUE_EFFECT_NOT_REPLICATED"}
    assert len(state["confirmation"]["tokens"]["per_token"]) == 24
    assert {noun.key for noun in confirmation.nouns} <= set(state["executed_noun_keys"])
    with pytest.raises(cd.PhaseError):
        runner.confirm()
    assert runner.report() == 0
    text = runner.report_path.read_text()
    assert "## Confirmation — outcome" in text and "Rank selection" in text
