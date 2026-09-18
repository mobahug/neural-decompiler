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
from neural_decompiler.models import PYTHIA_70M
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
                                           manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=ss.INHERITED_CONFIRMATION_SHA256, model={"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision})
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
    assert state["phases"]["explore"]["status"] == "running" and "deviates" in state["exploration"]["incidents"][-1]["message"]
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


def test_quality_gate_failure_closes_tier_a(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    monkeypatch.setattr(ss, "RANKS", (1, 2))
    monkeypatch.setattr(cd, "QUALITY_SPEARMAN_FLOOR", 1.1)  # unreachable: forces the gate to fail on the fake
    assert runner.explore() == 0
    state = cd.load_results_state(runner.results_path)
    assert not state["exploration"]["quality_gate"]["passed"] and state["exploration"]["outcome"]["label"] == "QUALITY_GATE_FAILED"
    with pytest.raises(ss.PhaseError, match="QUALITY_GATE_FAILED"):
        runner.calibrate()
    with pytest.raises(cd.PhaseError):
        runner.lock()
    assert runner.report() == 0
    text = runner.report_path.read_text()
    assert "Outcome at explore: `QUALITY_GATE_FAILED`" in text


def test_singular_gap_incident_is_recorded_and_blocks_a_rerun_at_the_same_commit(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    monkeypatch.setattr(ss, "RANKS", (1, 2))
    monkeypatch.setattr(ss, "GAP_MIN", 2.0)  # every fold violates the gap rule
    assert runner.explore() == 2
    state = cd.load_results_state(runner.results_path)
    incident = state["exploration"]["incidents"][-1]
    assert "singular gap" in incident["message"] and incident["commit"] == "a" * 40 and state["phases"]["explore"]["status"] == "running"
    assert "selection" not in state["exploration"]
    with pytest.raises(ss.PhaseError, match="incident is recorded at this commit"):
        runner.explore()
    assert runner.report() == 0 and "## Incidents" in runner.report_path.read_text()
    # After a committed fix (a different commit) explore may run again; the state records the new commit.
    monkeypatch.setattr(ss, "GAP_MIN", 1e-8)
    runner.git_state = lambda: {"commit": "b" * 40, "dirty": False}
    assert runner.explore() == 0
    state = cd.load_results_state(runner.results_path)
    assert state["protocol_code_commit"] == "b" * 40 and state["phases"]["explore"]["attempt_commits"] == ["a" * 40, "b" * 40]
    assert state["phases"]["explore"]["status"] == "complete" and len(state["exploration"]["incidents"]) == 1


def test_validate_lock_refusals(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    monkeypatch.setattr(ss, "RANKS", (1, 2))
    assert runner.explore() == 0
    state = cd.load_results_state(runner.results_path)
    if not state["exploration"]["quality_gate"]["passed"]:
        state["exploration"]["quality_gate"]["passed"] = True
        state["exploration"]["outcome"] = {"label": None, "note": "forced for the fake"}
        cd.write_results_state(runner.results_path, state)
    assert runner.calibrate() == 0 and runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    installed = sandbox / ss.LOCK_RELATIVE_PATH
    manifest, manifest_sha256, extension = pm.load_inputs(sandbox)
    confirmation = cd.load_confirmation(sandbox / ss.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    inherited = ss.load_inherited_extract(sandbox / ss.INHERITED_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation.content_sha256)
    state = cd.load_results_state(runner.results_path)

    def validate(lock, **overrides):
        arguments = dict(state=state, manifest_sha256=manifest_sha256, extension=extension, confirmation=confirmation, inherited=inherited, parameters_dir=runner.parameters_dir,
                         program_path=runner.program_path, program_005_path=runner.program_005_path, git_state={"commit": "a" * 40, "dirty": False}, tracked=True, changed_paths=[])
        arguments.update(overrides)
        ss.validate_lock(lock, **arguments)

    lock = json.loads(candidate.read_text())
    validate(lock)
    validate(lock, changed_paths=[ss.LOCK_RELATIVE_PATH, "experiments/007-supervised-cue-subspace/README.md", "experiments/007-supervised-cue-subspace/evidence/x.md", "docs/a.md"])
    with pytest.raises(ss.PhaseError, match="scientific paths"):
        validate(lock, changed_paths=["experiments/005-regular-plural-mechanism/preregistration-lock.json"])
    with pytest.raises(ss.PhaseError, match="clean Git tree"):
        validate(lock, git_state={"commit": "a" * 40, "dirty": True})
    with pytest.raises(ss.PhaseError, match="tracked"):
        validate(lock, tracked=False)
    with pytest.raises(ss.PhaseError, match="ancestor"):
        validate(lock, changed_paths=None)

    def resigned(**changes):
        edited = {**lock, **changes}
        unsigned = {k: v for k, v in edited.items() if k != "content_sha256"}
        edited["content_sha256"] = pm.sha256_text(pm.canonical_json(unsigned))
        return edited

    with pytest.raises(ss.PhaseError, match="candidate lock written by the lock phase"):
        validate(resigned(tau=lock["tau"] * 10))
    with pytest.raises(ss.PhaseError, match="digest mismatch"):
        validate({**lock, "tau": lock["tau"] * 10})
    with pytest.raises(ss.PhaseError, match="parameters differ"):
        index_path = runner.parameters_dir / "pca-006" / "parameters.json"
        original = index_path.read_text()
        index_path.write_text(original + "\n")
        try:
            validate(lock)
        finally:
            index_path.write_text(original)
    with pytest.raises(ss.PhaseError, match="program source"):
        validate(lock, program_path=ROOT / "experiments/006-low-rank-cue-decompilation/low_rank_program.py")
    with pytest.raises(ss.PhaseError, match="inherited extract differs"):
        validate(lock, inherited={**inherited, "content_sha256": "0" * 64})
    with pytest.raises(ss.PhaseError, match="not match the results state"):
        validate(lock, state={**state, "run_id": "other"})


def test_confirm_refuses_before_execution_when_programs_do_not_reproduce_the_lock_and_records_a_late_incident(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    monkeypatch.setattr(ss, "RANKS", (1, 2))
    assert runner.explore() == 0
    state = cd.load_results_state(runner.results_path)
    if not state["exploration"]["quality_gate"]["passed"]:
        state["exploration"]["quality_gate"]["passed"] = True
        state["exploration"]["outcome"] = {"label": None, "note": "forced for the fake"}
        cd.write_results_state(runner.results_path, state)
    assert runner.calibrate() == 0 and runner.lock() == 0
    shutil.copy(runner.results_path.parent / "candidate-lock.json", sandbox / ss.LOCK_RELATIVE_PATH)
    runner.changed_paths = lambda commit: []
    # Pre-execution: a program that no longer reproduces the lock is a PhaseError and nothing fresh runs.
    original = ss.load_programs

    def altered(*args, **kwargs):
        programs = original(*args, **kwargs)
        selected = programs["selected"]
        programs["selected"] = type("Shifted", (), {"predict_epatch_shift": lambda self, *a: selected.predict_epatch_shift(*a) + 1e-3,
                                                    "predict_behavior_shift": lambda self, *a: selected.predict_behavior_shift(*a) + 1e-3,
                                                    "predict_pair": lambda self, *a: selected.predict_pair(*a)})()
        return programs

    monkeypatch.setattr(ss, "load_programs", altered)
    with pytest.raises(ss.PhaseError, match="nothing was executed"):
        runner.confirm()
    state = cd.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "not_started"
    manifest, manifest_sha256, extension = pm.load_inputs(sandbox)
    confirmation = cd.load_confirmation(sandbox / ss.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    assert not {noun.key for noun in confirmation.nouns} & set(state["executed_noun_keys"])
    monkeypatch.setattr(ss, "load_programs", original)
    # Post-execution: an incident after the fresh prompts ran is recorded with the invalidated measurements and blocks any re-run.
    monkeypatch.setattr(ss, "check_locked_predictions", lambda measurements, lock: (_ for _ in ()).throw(ss.ConfirmationIncident("late mismatch", {"tokens": measurements})))
    assert runner.confirm() == 2
    state = cd.load_results_state(runner.results_path)
    incident = state["confirmation"]["incident"]
    assert incident["phase"] == "confirm" and "late mismatch" in incident["message"] and len(incident["invalidated"]["tokens"]["per_token"]) == 24
    assert state["phases"]["confirm"]["status"] == "running" and "lock_predictions_reproduced_max_difference" in state["phases"]["confirm"]
    with pytest.raises(cd.PhaseError):
        runner.confirm()
    assert runner.report() == 0 and "## Incidents" in runner.report_path.read_text()
