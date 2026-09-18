"""Phase isolation and the full state machine for the Experiment 007 runner (fake model, real frozen inputs)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import supervised_subspace as ss
from plural_fakes import TinyPlural

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / pm.MANIFEST_RELATIVE_PATH
EXTENSION_PATH = ROOT / pm.EXTENSION_RELATIVE_PATH
CONFIRMATION_PATH = ROOT / ss.CONFIRMATION_RELATIVE_PATH
FAKE_CIRCUIT = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L01.H00",), ("L01.MLP",))


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_007_runner", ROOT / "experiments/007-supervised-cue-subspace/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000)


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "screening/behavior-candidates").mkdir(parents=True)
    shutil.copy(MANIFEST_PATH, tmp_path / pm.MANIFEST_RELATIVE_PATH)
    (tmp_path / "experiments/005-regular-plural-mechanism").mkdir(parents=True)
    shutil.copy(EXTENSION_PATH, tmp_path / pm.EXTENSION_RELATIVE_PATH)
    (tmp_path / "experiments/006-low-rank-cue-decompilation").mkdir(parents=True)
    shutil.copy(CONFIRMATION_PATH, tmp_path / ss.CONFIRMATION_RELATIVE_PATH)
    (tmp_path / "experiments/007-supervised-cue-subspace/inherited").mkdir(parents=True)
    # The inherited extract for the fake: the fake's own exposed E-patch responses, recorded in the committed format.
    manifest, manifest_sha256, extension = pm.load_inputs(tmp_path)
    pool = cd.exposed_pool(manifest, extension)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    responses = cd.measure_epatch_responses(model, weights, cache, pool.frames, pool.tokens, pool.reference_ids, circuit=FAKE_CIRCUIT)
    recorded = {f"{token}|{frame_id}": {"template_id": r.template_id, "mean_shift": pm._mean(list(r.shifts.values())), "delta_norm": float(r.delta_residual.norm()), "circuit_share": r.circuit_share}
                for (token, frame_id), r in responses.items()}
    payload = ss.inherited_extract_payload(recorded, source={"path": "fake", "file_sha256": "0" * 64, "state_sha256": "1" * 64, "run_id": "fake", "protocol_code_commit": "b" * 40, "explore_completed_at": "t"},
                                           manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=ss.INHERITED_CONFIRMATION_SHA256, model={"model_id": "fake", "revision": "0"})
    (tmp_path / ss.INHERITED_EXTRACT_RELATIVE_PATH).write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    return tmp_path


def make_runner(sandbox, monkeypatch, *, logs=None):
    logs = logs if logs is not None else []
    monkeypatch.setattr(pm, "DENOMINATOR_FLOOR_OVERALL", -1e9)
    monkeypatch.setattr(pm, "DENOMINATOR_FLOOR_STRATUM", -1e9)
    runner = runner_module.Runner(
        root=sandbox,
        results_path=sandbox / "outputs/experiment-007/results.json",
        report_path=sandbox / "outputs/experiment-007/report.md",
        parameters_dir=sandbox / "outputs/experiment-007/parameters",
        program_path=ROOT / "experiments/007-supervised-cue-subspace/linear_cue_program.py",
        program_005_path=ROOT / "experiments/005-regular-plural-mechanism/mechanism_program.py",
        lock_005_path=sandbox / "missing-005-lock.json",
        results_006_path=sandbox / "missing-006-results.json",
        circuit=FAKE_CIRCUIT,
        model_loader=lambda spec: make_fake_model(),
        git_state=lambda: {"commit": "a" * 40, "dirty": False},
        versions=lambda: {"torch": "test"},
        tracked=lambda path: True,
        changed_paths=lambda commit: [],
        contract_runner=lambda: {"passed": True, "returncode": 0, "output_tail": "", "output_sha256": "x"},
        log=logs.append,
    )
    return runner, logs


def test_parser_has_exactly_the_six_phases_and_no_overrides():
    parser = runner_module.build_parser()
    for phase in ("validate", "explore", "calibrate", "lock", "confirm", "report"):
        assert parser.parse_args([phase]).phase == phase
    with pytest.raises(SystemExit):
        parser.parse_args(["freeze-confirmation"])
    with pytest.raises(SystemExit):
        parser.parse_args(["confirm", "--force"])


def test_validate_refuses_a_confirmation_that_is_not_the_frozen_set(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.validate() == 0
    path = sandbox / ss.CONFIRMATION_RELATIVE_PATH
    payload = json.loads(path.read_text())
    payload["construction"] = payload["construction"] + " (altered)"
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in payload.items() if k != "content_sha256"}))
    path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    assert runner.validate() == 1 and "not the frozen Experiment 006 digest" in logs[-1]
    with pytest.raises(ss.PhaseError):
        runner.explore()
    runner.tracked = lambda path: False
    assert runner.validate() == 1 and "tracked" in logs[-1]


def test_explore_stops_as_an_incident_when_the_inherited_responses_do_not_replicate(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    path = sandbox / ss.INHERITED_EXTRACT_RELATIVE_PATH
    payload = json.loads(path.read_text())
    key = next(iter(payload["responses"]))
    payload["responses"][key]["mean_shift"] += 1e-3
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in payload.items() if k != "content_sha256"}))
    path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(ss, "RANKS", (1, 2))
    assert runner.explore() == 2
    state = cd.load_results_state(runner.results_path)
    assert state["phases"]["explore"]["status"] == "running" and "deviates" in state["exploration"]["incident"]["message"]
    assert "selection" not in state["exploration"]
    with pytest.raises(cd.PhaseError):
        runner.calibrate()


def test_full_state_machine_on_the_fake(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    monkeypatch.setattr(ss, "RANKS", (1, 2))
    manifest, manifest_sha256, extension = pm.load_inputs(sandbox)
    confirmation = cd.load_confirmation(sandbox / ss.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    assert runner.validate() == 0
    assert runner.explore() == 0
    state = cd.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["status"] == "complete"
    assert exploration["inherited_check"]["passed"] and exploration["inherited_check"]["max_abs_deviation"] == 0.0
    assert exploration["selection"]["selected"] in (1, 2)
    assert set(exploration["loco"]["supervised"]) == {"1", "2"} and len(exploration["loco"]["supervised"]["1"]) == 16
    assert set(exploration["loco"]["pca-006"]) == {"1", "2"} and len(exploration["loco"]["ridge-full"]) == 16 and len(exploration["folds"]) == 16
    assert set(exploration["comparison"]["per_rank"]) == {"1", "2"} and exploration["comparison"]["ridge_full_error"] > 0
    assert set(exploration["parameters"]) == set(ss.PROGRAM_NAMES)
    assert len(exploration["epatch"]) == 12 * 16
    assert not {noun.key for noun in confirmation.nouns} & set(state["executed_noun_keys"])
    assert not {prompt.key for prompt in confirmation.all_prompts} & set(state["executed_prompt_keys"])
    with pytest.raises(cd.PhaseError):
        runner.explore()  # once
    if not exploration["quality_gate"]["passed"]:
        assert exploration["outcome"]["label"] == "QUALITY_GATE_FAILED"
        with pytest.raises(ss.PhaseError):
            runner.calibrate()  # the experiment ended at Tier A
        assert runner.report() == 0 and "QUALITY_GATE_FAILED" in runner.report_path.read_text()
        forced = cd.load_results_state(runner.results_path)
        forced["exploration"]["quality_gate"]["passed"] = True
        forced["exploration"]["outcome"] = {"label": None, "note": "forced for the fake"}
        cd.write_results_state(runner.results_path, forced)
    assert runner.calibrate() == 0
    with pytest.raises(cd.PhaseError):
        runner.confirm()  # lock phase not run
    assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    lock = json.loads(candidate.read_text())
    assert lock["experiment"] == "007" and set(lock["parameters"]) == set(ss.PROGRAM_NAMES)
    assert lock["inherited"]["confirmation_sha256"] == ss.INHERITED_CONFIRMATION_SHA256 and lock["estimator"]["gap_min"] == 1e-8
    assert len(lock["predictions"]["tokens"]) == 24 and len(lock["predictions"]["pairs"]) == 6
    assert set(next(iter(lock["predictions"]["tokens"].values()))["frames"].values().__iter__().__next__()["predicted"]) == set(ss.PROGRAM_NAMES)
    with pytest.raises(cd.PhaseError):
        runner.confirm()  # lock not installed
    installed = sandbox / ss.LOCK_RELATIVE_PATH
    shutil.copy(candidate, installed)
    runner.changed_paths = lambda commit: ["src/neural_decompiler/supervised_subspace.py"]
    with pytest.raises(cd.PhaseError):
        runner.confirm()
    runner.changed_paths = lambda commit: ["experiments/006-low-rank-cue-decompilation/confirmation-v1.json"]
    with pytest.raises(cd.PhaseError):
        runner.confirm()
    runner.changed_paths = lambda commit: ["README.md"]
    assert runner.confirm() == 0
    state = cd.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "complete"
    verdict = state["confirmation"]["outcome"]
    assert verdict["label"] in {"DECOMPILED", "DECOMPILED_MISCALIBRATED", "PROGRAM_NOT_SUPPORTED", "CUE_EFFECT_NOT_REPLICATED"}
    assert set(state["confirmation"]["baselines"]) == {"pca-006", "e005-scalar", "ridge-full"}
    assert state["confirmation"]["circuit"]["enters_outcome"] is False
    assert len(state["confirmation"]["tokens"]["per_token"]) == 24
    assert {noun.key for noun in confirmation.nouns} <= set(state["executed_noun_keys"])
    with pytest.raises(cd.PhaseError):
        runner.confirm()
    assert runner.report() == 0
    text = runner.report_path.read_text()
    assert "## Confirmation — outcome" in text and "Rank selection" in text and "Circuit families (reported; not in the outcome)" in text
