"""Experiment 021's runner on the six-layer fake, inside a fake-closed Experiment 020 world: 020's own runner explores
the fake (five exposed cues per frame, so every frame measures its template's plural cue, ``a`` and ``the``), then a
closure record and an evidence extract are written in the committed format and 021's frozen digest constants are
pointed at them. On that world: validate without a model, calibrate re-executing exactly 020's ledger, the gate, the
screen, the precondition stop, incident gating, the leakage plants, the no-forward-pass lock, stage 1 with the Y2 row
digested before the barrier, stage 2 with the ceiling, the scoring and the report — and the tests that need the fake:
020's call order in the re-materialization, the environment check, and stage 2's equivalence with 020's."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from neural_decompiler import attention_patterns as atp
from neural_decompiler import cue_suppression as cs
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from test_experiment_020_runner import ExecutionSpy, _copy_repo_files, fake_world, make_fake_model  # noqa: F401 — fake_world is a fixture
from test_experiment_020_runner import runner_module as runner_020
from test_readout_calibration import write_closed_020
from test_readout_decompilation import toy_tokenizer_020

ROOT = Path(__file__).parents[1]
TOKEN_LIMIT = 5  # the smallest limit at which every frame measures its template's plural cue (767 / 2067) and a, the
COMMIT_A, COMMIT_B = "a" * 40, "b" * 40


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_021_runner", ROOT / "experiments/021-corrected-readout-confirmation/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def _contract():
    return {"passed": True, "returncode": 0, "output_tail": "", "output_sha256": "x"}


@pytest.fixture(scope="module")
def closed_world(fake_world, tmp_path_factory):
    """020's own runner explores the fake once; the closure trio is written in the committed format."""
    manifest, small, digests, lock_011, lock_012, lock_017 = fake_world
    root = tmp_path_factory.mktemp("closed020")
    _copy_repo_files(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
        patch.setattr(rd, "EXPLORE_TOKEN_LIMIT", TOKEN_LIMIT)
        explorer = runner_020.Runner(root=root, results_path=root / rc.RESULTS_020_RELATIVE_PATH, report_path=root / "outputs/experiment-020/report.md",
                                     diagnostic_path=root / "outputs/experiment-020/level1-diagnostic.json", model_loader=lambda spec: make_fake_model(),
                                     tokenizer_loader=lambda spec: toy_tokenizer_020(manifest, small), lock_011_loader=lambda path: dict(lock_011),
                                     lock_012_loader=lambda path: dict(lock_012), lock_017_loader=lambda path: dict(lock_017),
                                     git_state=lambda: {"commit": COMMIT_A, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True,
                                     changed_paths=lambda commit: [], contract_runner=_contract, log=lambda message: None)
        assert explorer.explore() == 0
    state = rd.load_results_state(root / rc.RESULTS_020_RELATIVE_PATH)
    confirmation = rd.load_confirmation(root / rd.CONFIRMATION_RELATIVE_PATH, small, digests)
    constants = write_closed_020(root, state=state, confirmation=confirmation)
    return root, constants, state


def _fake_pools(pool):
    """The real frame and noun pools; every cue stratum is {a, the}, which the fake measured in all 108 frames."""
    real = rc.build_pools(pool)
    cues = {cls: (("a", 247), ("the", 253)) for cls in rc.CUE_CLASSES}
    return rc.CalibrationPools(cues, cues, {"confirmation-011": cues["quantity"], "confirmation-019": (("the", 253),)}, real.frames_unscreened, real.frames_all_unscreened, real.nouns)


@pytest.fixture
def sandbox(closed_world, fake_world, tmp_path, monkeypatch):
    """A fresh copy of the closed world, and every patch the fake needs (frozen constants pointed at the fake)."""
    root, constants, state_020 = closed_world
    world = tmp_path / "world"
    shutil.copytree(root, world)
    for name, value in constants.items():
        monkeypatch.setattr(rc, name, value)
    for module, name, value in ((cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0), (rd, "EXPLORE_TOKEN_LIMIT", TOKEN_LIMIT), (rd, "MIN_VALID_FRESH_FRAMES", 3),
                                (rd, "MIN_VALID_COORDINATED_FRAMES", 1), (rc, "B", 40), (rc, "CROSS_CHECK_DRAWS", 2), (rc, "MIN_SCREENED_FRAMES_PER_TEMPLATE", 1)):
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(rc, "production_pools", _fake_pools)
    monkeypatch.setattr(rc, "LOCK_DIGESTS", {"lock_011": fake_world[3]["content_sha256"], "lock_012": fake_world[4]["content_sha256"], "lock_017": fake_world[5]["content_sha256"]})
    real_validity = rd.frame_validity
    monkeypatch.setattr(rd, "frame_validity", lambda *args, **kwargs: {**real_validity(*args, **kwargs), "valid": True})  # a random-weight fake's verdicts carry no meaning
    return world, fake_world, state_020


def make_runner(sandbox, **overrides):
    world, fake, _ = sandbox
    manifest, small, digests, lock_011, lock_012, lock_017 = fake
    logs: list[str] = []
    arguments = dict(root=world, results_path=world / "outputs/experiment-021/results.json", report_path=world / "outputs/experiment-021/report.md",
                     model_loader=lambda spec: make_fake_model(), lock_011_loader=lambda path: dict(lock_011), lock_012_loader=lambda path: dict(lock_012),
                     lock_017_loader=lambda path: dict(lock_017), git_state=lambda: {"commit": COMMIT_A, "dirty": False}, versions=lambda: {"torch": "test"},
                     tracked=lambda path: True, changed_paths=lambda commit: [], contract_runner=_contract, log=logs.append)
    arguments.update(overrides)
    return runner_module.Runner(**arguments), logs


def _reclose(world: Path, state: dict, monkeypatch, fake) -> None:
    """Rewrite the fake 020 state and its closure trio, and point the constants at them (a planted change)."""
    manifest, small, digests, *_ = fake
    confirmation = rd.load_confirmation(world / rd.CONFIRMATION_RELATIVE_PATH, small, digests)
    for name, value in write_closed_020(world, state=state, confirmation=confirmation).items():
        monkeypatch.setattr(rc, name, value)


def _files_020(world: Path) -> dict[str, str]:
    return {path: rc.file_sha256(world / path) for path in (rc.CLOSURE_020_RELATIVE_PATH, rc.EXTRACT_020_RELATIVE_PATH, rc.RESULTS_020_RELATIVE_PATH)}


# ---------------------------------------------------------------------------


def test_parser_has_exactly_the_five_phases_and_no_overrides():
    parser = runner_module.build_parser()
    for phase in ("validate", "calibrate", "lock", "confirm", "report"):
        assert parser.parse_args([phase]).phase == phase
    for forbidden in (["explore"], ["freeze-confirmation"], ["calibrate", "--draws", "10"], ["confirm", "--stage", "2"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden)
    assert runner_module.PHASES == rc.PHASES == ("validate", "calibrate", "lock", "confirm", "report")


def test_validate_loads_no_model_and_refuses_every_tampering(sandbox, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("validate loaded a model")

    runner, logs = make_runner(sandbox, model_loader=refuse)
    assert runner.validate() == 0, logs
    world = sandbox[0]
    closure = json.loads((world / rc.CLOSURE_020_RELATIVE_PATH).read_text())
    (world / rc.CLOSURE_020_RELATIVE_PATH).write_text(json.dumps({**closure, "status": "reopened"}))
    assert runner.validate() == 1 and "closure" in logs[-1]
    (world / rc.CLOSURE_020_RELATIVE_PATH).write_text(json.dumps(closure))
    monkeypatch.setattr(rc, "PROGRAM_BLOB_SHA1", "0" * 40)
    assert runner.validate() == 1 and "blob" in logs[-1]


def test_calibrate_reexecutes_exactly_020s_ledger_and_writes_the_record(sandbox, monkeypatch):
    world, fake, state_020 = sandbox
    before = _files_020(world)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(sandbox)
    assert runner.calibrate() == 0, logs[-3:]
    state = rd.load_results_state(runner.results_path)
    ledger_020 = set(state_020["executed_prompt_keys"])
    assert set(spy.counts) == ledger_020 == set(state["executed_prompt_keys"]) and all(count == 1 for count in spy.counts.values())
    confirmation = runner._inputs()[4]
    assert not ({prompt.key for prompt in confirmation.all_prompts} & set(spy.counts))
    assert _files_020(world) == before  # Experiment 020's closure, extract and state are untouched
    calibration = state["calibration"]
    assert state["phases"]["calibrate"]["status"] == "complete" and calibration["gate"]["passed"] and calibration["gate"]["max_difference"] <= rc.RECONSTRUCTION_TOLERANCE
    assert calibration["environment"]["max_state_drift"] <= rc.LOCKED_STATE_TOLERANCE and calibration["precondition"]["ok"]
    assert set(calibration["screen"]) == {frame.frame_id for frame in fake[1].frames}
    record = json.loads(runner.candidate_calibration_path.read_text())
    assert rc.file_sha256(runner.candidate_calibration_path) == calibration["record_sha256"] and record["content_sha256"] == rc.content_digest(record)
    for key, lock in (("lock_011", fake[3]), ("lock_012", fake[4]), ("lock_017", fake[5])):  # the exact inherited objects, bound
        assert state[f"{key}_sha256"] == record["inputs"][key] == lock["content_sha256"]
    assert set(record["inputs"]) == set(rc.DIGEST_KEYS)
    assert [len(record["rows"][o]) for o in ("Y1", "Y2", "Y3")] == [1, len(rc.y2_rows()), len(rc.y3_rows())]
    assert record["cross_check"]["n_checked"] == 2 * (1 + len(rc.y2_rows()) + len(rc.y3_rows())) and record["cross_check"]["max_difference"] <= rc.STATISTIC_AGREEMENT_TOLERANCE
    rd.assert_fresh_nouns_absent(record, confirmation)
    rd.assert_fresh_nouns_absent(state["calibration"], confirmation)
    table = torch.load(runner.table_path)
    assert {name: rc.tensor_digest(table[name]) for name in ("measured", "level0", "ceiling", "no_l5", "base", "dT")} == calibration["table_sha256"] == record["table_sha256"]
    arrays = torch.load(runner.draws_path)
    assert all(rc.tensor_digest(values) == record["draw_arrays_sha256"][o][k] for o, entries in arrays.items() for k, values in entries.items())
    with pytest.raises(rd.PhaseError, match="once"):
        runner.calibrate()
    assert runner.report() == 0 and "Calibration (exposed only)" in runner.report_path.read_text()


def test_calibrate_refuses_a_version_or_thread_mismatch_and_a_failed_contract(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, versions=lambda: {"torch": "other"})
    with pytest.raises(rd.PhaseError, match="versions"):
        runner.calibrate()
    real_runtime = runner_module.runtime_record
    monkeypatch.setattr(runner_module, "runtime_record", lambda spec: {**real_runtime(spec), "torch_num_threads": 999})
    runner, logs = make_runner(sandbox)
    with pytest.raises(rd.PhaseError, match="runtime"):
        runner.calibrate()
    monkeypatch.setattr(runner_module, "runtime_record", real_runtime)
    runner, logs = make_runner(sandbox, contract_runner=lambda: {"passed": False})
    assert runner.calibrate() == 1 and "A0 contract" in logs[-1]


def test_leakage_plants_are_refused_before_or_during_calibrate(sandbox, monkeypatch):
    world, fake, state_020 = sandbox
    confirmation = rd.load_confirmation(world / rd.CONFIRMATION_RELATIVE_PATH, fake[1], fake[2])
    planted = dict(state_020, executed_prompt_keys=sorted(set(state_020["executed_prompt_keys"]) | {confirmation.all_prompts[0].key}))
    _reclose(world, planted, monkeypatch, fake)
    runner, logs = make_runner(sandbox)
    with pytest.raises(rd.PhaseError, match="confirmation prompts"):
        runner.calibrate()
    # a key the re-materialization would run that 020 never ran: an incident before that key executes
    missing = sorted(state_020["executed_prompt_keys"])
    victim = next(key for key in missing if key.split("|")[1] == "the")
    _reclose(world, dict(state_020, executed_prompt_keys=[key for key in missing if key != victim]), monkeypatch, fake)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(sandbox)
    assert runner.calibrate() == 2
    state = rd.load_results_state(runner.results_path)
    assert "outside Experiment 020's exposed ledger" in state["calibration"]["incidents"][-1]["message"] and victim not in spy.counts
    assert set(state["executed_prompt_keys"]) == set(spy.counts) and "gate" not in state["calibration"]


def test_a_perturbed_record_stops_calibrate_at_the_gate_with_nothing_drawn(sandbox, monkeypatch):
    world, fake, state_020 = sandbox
    planted = json.loads(json.dumps(state_020))
    first = sorted(planted["exploration"]["statistics"]["per_cue"])[0]
    planted["exploration"]["statistics"]["per_cue"][first]["mae"] += 1e-8
    _reclose(world, planted, monkeypatch, fake)
    drawn = []
    monkeypatch.setattr(rc, "run_calibration", lambda *args, **kwargs: drawn.append(1))
    runner, logs = make_runner(sandbox)
    assert runner.calibrate() == 2
    state = rd.load_results_state(runner.results_path)
    assert "reproduction gate failed" in state["calibration"]["incidents"][-1]["message"] and not state["calibration"]["gate"]["passed"]
    assert state["calibration"]["gate"]["max_difference"] == pytest.approx(1e-8, rel=1e-3) and "per_cue" in state["calibration"]["gate"]["at"]
    assert not drawn and not runner.candidate_calibration_path.exists() and not runner.draws_path.exists()


def test_the_precondition_stops_for_review_and_nothing_follows(sandbox, monkeypatch):
    world, fake, _ = sandbox
    monkeypatch.setattr(rc, "MIN_SCREENED_FRAMES_PER_TEMPLATE", 6)
    real_screen = rc.validity_screen

    def five_quantifier_frames(*args, **kwargs):
        verdicts = real_screen(*args, **kwargs)
        pools = rc.build_pools(fake[1])
        keep = set(pools.frames_unscreened["quantifier"][:5])
        return {frame_id: {**verdict, "valid": (frame_id in keep) or not frame_id.startswith("quantifier")} for frame_id, verdict in verdicts.items()}

    monkeypatch.setattr(rc, "validity_screen", five_quantifier_frames)
    drawn = []
    monkeypatch.setattr(rc, "run_calibration", lambda *args, **kwargs: drawn.append(1))
    runner, logs = make_runner(sandbox)
    assert runner.calibrate() == 3
    state = rd.load_results_state(runner.results_path)
    assert state["phases"]["calibrate"]["status"] == "stopped_for_review" and state["calibration"]["precondition"]["counts"]["quantifier"] == 5
    assert not drawn and not runner.candidate_calibration_path.exists()
    for phase in (runner.calibrate, runner.lock, runner.confirm):
        with pytest.raises(rd.PhaseError):
            phase()
    assert runner.report() == 0


def test_an_incident_blocks_its_commit_and_a_committed_fix_resumes(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox)
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(rd, "capture_frame_020", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic capture failure")))
        assert runner.calibrate() == 2
    state = rd.load_results_state(runner.results_path)
    assert state["phases"]["calibrate"]["status"] == "running" and state["calibration"]["incidents"][-1]["commit"] == COMMIT_A
    with pytest.raises(rd.PhaseError, match="incident"):
        runner.calibrate()
    runner.git_state = lambda: {"commit": COMMIT_B, "dirty": False}
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(rd, "capture_frame_020", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic second failure")))
        assert runner.calibrate() == 2
    state = rd.load_results_state(runner.results_path)
    assert state["phases"]["calibrate"]["attempts"] == 2 and state["phases"]["calibrate"]["attempt_commits"] == [COMMIT_A, COMMIT_B]
    with pytest.raises(rd.PhaseError, match="calibrate"):
        runner.lock()


def test_an_interruption_is_recorded_and_a_running_phase_without_an_incident_never_resumes(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox)
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(rd, "capture_frame_020", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            runner.calibrate()
    state = rd.load_results_state(runner.results_path)
    assert state["calibration"]["incidents"][-1]["type"] == "KeyboardInterrupt" and state["calibration"]["incidents"][-1]["closure_020_recheck"] == {"ok": True}
    with pytest.raises(rd.PhaseError, match="incident"):
        runner.calibrate()  # the interruption's own commit is blocked
    # an attempt that ended with no incident at all (a kill): recorded at the next start, and its commit never reused
    state["phases"]["calibrate"]["attempt_commits"].append(COMMIT_B)
    state["phases"]["calibrate"]["commit"] = COMMIT_B
    rd.write_results_state(runner.results_path, {key: value for key, value in state.items() if key != "state_sha256"})
    runner.git_state = lambda: {"commit": COMMIT_B, "dirty": False}
    with pytest.raises(rd.PhaseError, match="incident is recorded at this commit"):
        runner.calibrate()
    reconciled = rd.load_results_state(runner.results_path)["calibration"]["incidents"][-1]
    assert reconciled["type"] == "UnrecordedTermination" and reconciled["commit"] == COMMIT_B
    runner.git_state = lambda: {"commit": "c" * 40, "dirty": False}
    with pytest.MonkeyPatch.context() as guard:  # a new commit may resume; a model that fails to load is an incident, not a stuck phase
        runner.model_loader = lambda spec: (_ for _ in ()).throw(pm.IncidentError("synthetic load failure"))
        assert runner.calibrate() == 2
    state = rd.load_results_state(runner.results_path)
    assert state["calibration"]["incidents"][-1]["message"] == "synthetic load failure" and state["phases"]["calibrate"]["attempt_commits"] == [COMMIT_A, COMMIT_B, "c" * 40]


def test_a_cross_check_failure_through_calibrate_is_persisted_with_its_location_and_nothing_follows(sandbox, monkeypatch):
    original = rc.direct_y1
    monkeypatch.setattr(rc, "direct_y1", lambda *args, **kwargs: {**original(*args, **kwargs), "token_mean_r2": None})  # undefined on one side: a non-finite difference
    runner, logs = make_runner(sandbox)
    assert runner.calibrate() == 2
    state = rd.load_results_state(runner.results_path)
    details = state["calibration"]["cross_check"]
    assert details["outcome"] == "Y1" and details["statistic"] == "token_mean_r2" and details["row"] == "all" and details["max_difference"] is None and details["n_exceeding"] >= 1
    assert state["calibration"]["incidents"][-1]["type"] == "CrossCheckError" and state["phases"]["calibrate"]["status"] == "running"
    assert not runner.candidate_calibration_path.exists() and not runner.draws_path.exists() and "record_sha256" not in state["calibration"]
    with pytest.raises(rd.PhaseError, match="incident"):
        runner.calibrate()


def test_a_structural_environment_mismatch_is_an_incident_with_a_writable_record(sandbox, monkeypatch):
    world, fake, state_020 = sandbox
    planted = json.loads(json.dumps(state_020))
    frame_id = sorted(planted["exploration"]["locked_states"])[0]
    del planted["exploration"]["locked_states"][frame_id]["rows5"]
    _reclose(world, planted, monkeypatch, fake)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(sandbox)
    assert runner.calibrate() == 2
    state = rd.load_results_state(runner.results_path)
    environment = state["calibration"]["environment"]
    assert environment["structural_state_mismatch"] is True and environment["max_state_drift"] is None and frame_id in environment["at"]
    assert state["calibration"]["incidents"][-1]["type"] == "EnvironmentIncident" and "gate" not in state["calibration"]
    references = {fake[1].reference_prompt(frame).key for frame in fake[1].frames}
    assert set(spy.counts) == references and all(count == 1 for count in spy.counts.values())  # the references only: no cue prompt ran


@pytest.fixture
def calibrated(sandbox):
    runner, logs = make_runner(sandbox)
    assert runner.calibrate() == 0, logs[-3:]
    return runner, logs


def _install_record(runner) -> None:
    target = runner.root / rc.CALIBRATION_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(runner.candidate_calibration_path, target)


def _lock(runner) -> dict:
    with pytest.MonkeyPatch.context() as guard:
        for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, name):
                guard.setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the lock phase ran a forward pass")))
        assert runner.lock() == 0
    return json.loads((runner.output_dir / "candidate-lock.json").read_text())


def _install_lock(runner) -> None:
    shutil.copy(runner.output_dir / "candidate-lock.json", runner.root / rc.LOCK_RELATIVE_PATH)
    shutil.copy(runner.output_dir / "candidate-predictions.md", runner.root / rc.PREDICTIONS_RELATIVE_PATH)


def test_lock_confirm_and_report_through_every_boundary(calibrated, sandbox, monkeypatch):
    runner, logs = calibrated
    world, fake, _ = sandbox
    with pytest.raises(rd.PhaseError, match="install the candidate calibration record"):
        runner.lock()
    _install_record(runner)
    runner.changed_paths = lambda commit: ["src/neural_decompiler/readout_calibration.py"]
    with pytest.raises(rd.PhaseError, match="scientific paths"):
        runner.lock()
    runner.changed_paths = lambda commit: [rc.CALIBRATION_RELATIVE_PATH, f"{rc.EXPERIMENT_DIR}/README.md"]
    real_runtime = runner_module.runtime_record
    monkeypatch.setattr(runner_module, "runtime_record", lambda spec: {**real_runtime(spec), "torch_num_threads": 999})
    with pytest.raises(rd.PhaseError, match="runtime"):
        runner.lock()
    monkeypatch.setattr(runner_module, "runtime_record", real_runtime)
    lock = _lock(runner)
    record = json.loads((runner.root / rc.CALIBRATION_RELATIVE_PATH).read_text())
    confirmation = runner._inputs()[4]
    assert lock["experiment"] == "021" and lock["floor_tables"] == rc.floor_tables(record) and lock["calibration_content_sha256"] == record["content_sha256"]
    assert len(lock["predictions"]["rows"]) == len(confirmation.tokens) * len(fake[1].frames) and lock["program_blob_sha1"] == rc.PROGRAM_BLOB_SHA1
    assert rd.load_results_state(runner.results_path)["lock"]["provenance_difference"] <= rd.PROVENANCE_TOLERANCE
    with pytest.raises(rd.PhaseError, match="lock"):
        runner.confirm()  # not installed
    _install_lock(runner)
    # the lock's floor tables must be the committed record's, float for float
    state = rd.load_results_state(runner.results_path)
    altered = json.loads(json.dumps(lock))
    first = sorted(altered["floor_tables"]["Y2"])[0]
    altered["floor_tables"]["Y2"][first]["frame_mean_r2"] += 1e-12
    altered["content_sha256"] = rc.content_digest(altered)
    with pytest.raises(rd.PhaseError, match="floor tables"):
        rc.validate_lock(altered, state={**state, "lock": {**state["lock"], "content_sha256": altered["content_sha256"]}}, digests=runner._inputs()[5], confirmation=confirmation,
                         record=record, record_file_sha256=rc.file_sha256(runner.root / rc.CALIBRATION_RELATIVE_PATH), predictions_text=(runner.root / rc.PREDICTIONS_RELATIVE_PATH).read_text(),
                         git_state={"dirty": False}, tracked=True, changed_paths=[])
    assert all(lock[f"{key}_sha256"] == runner._inputs()[5][key] for key in rc.DIGEST_KEYS)  # the lock binds every input, the inherited locks included
    rebound = json.loads(json.dumps(lock))
    rebound["lock_017_sha256"] = "1" * 64
    rebound["content_sha256"] = rc.content_digest(rebound)
    with pytest.raises(rd.PhaseError, match="different frozen inputs"):
        rc.validate_lock(rebound, state={**state, "lock": {**state["lock"], "content_sha256": rebound["content_sha256"]}}, digests=runner._inputs()[5], confirmation=confirmation,
                         record=record, record_file_sha256=rc.file_sha256(runner.root / rc.CALIBRATION_RELATIVE_PATH), predictions_text=(runner.root / rc.PREDICTIONS_RELATIVE_PATH).read_text(),
                         git_state={"dirty": False}, tracked=True, changed_paths=[])
    runner.changed_paths = lambda commit: [rc.LOCK_RELATIVE_PATH, rc.PREDICTIONS_RELATIVE_PATH]
    invalid_fresh = {"cardinal-020-1", "coordinated-adjective-020-1", "coordinated-adjective-020-2"}
    permissive = rd.frame_validity
    monkeypatch.setattr(rd, "frame_validity", lambda program, state, *args, **kwargs: {**permissive(program, state, *args, **kwargs), "valid": state.frame.frame_id not in invalid_fresh})
    spy = ExecutionSpy(monkeypatch)
    seen = {"stage": None, "stage1_keys": [], "at_stage_two": None}
    original_stage_one, original_stage_two = rd.stage_one, rc.stage_two_021

    def guarded_stage_one(*args, **kwargs):
        before = dict(spy.counts)
        try:
            return original_stage_one(*args, **kwargs)
        finally:
            seen["stage1_keys"] = [key for key, count in spy.counts.items() if count != before.get(key, 0)]

    def guarded_stage_two(*args, **kwargs):
        on_disk = rd.load_results_state(runner.results_path)
        stage1 = on_disk["confirmation"]["stage1"]
        seen["at_stage_two"] = {"selection": stage1["y2_selection"], "digest_ok": stage1["y2_selection_sha256"] == rc.selection_digest(stage1["y2_selection"], stage1["digest"]),
                                "ledger": set(on_disk["executed_prompt_keys"]), "executed": dict(spy.counts)}
        return original_stage_two(*args, **kwargs)

    monkeypatch.setattr(rd, "stage_one", guarded_stage_one)
    monkeypatch.setattr(rc, "stage_two_021", guarded_stage_two)
    assert runner.confirm() == 0, logs[-3:]
    classes = rd.manifest_classes(confirmation)
    assert set(seen["stage1_keys"]) == set(classes["S1-REF"]) | set(classes["S1-VALIDITY"])  # stage 1 ran its manifest keys and nothing else
    assert seen["at_stage_two"]["digest_ok"] and not (set(classes["S2-TARGET"]) & set(seen["at_stage_two"]["executed"]))  # the row was on disk before any target ran
    state = rd.load_results_state(runner.results_path)
    results = state["confirmation"]
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["lock_predictions_reproduced_max_difference"] == 0.0
    assert state["phases"]["confirm"]["stage1_rows_reproduced_max_difference"] == 0.0
    stage2 = results["stage2"]
    saved = torch.load(stage2["path"])
    assert rc.tensor_digest(saved["Y1"]["exposed"]["measured"]) == stage2["tables_sha256"]["Y1"]["exposed"]["measured"] and set(stage2["identities"]) >= {"readout", "level1"}
    selection = results["stage1"]["y2_selection"]
    assert selection["composition"] == [5, 6, 4] and selection["row"] == "5/6/4" and set(selection["valid_frames"]).isdisjoint(invalid_fresh)
    assert selection == seen["at_stage_two"]["selection"] and results["Y2"]["row"] == "5/6/4" and results["Y2"]["floors"] == lock["floor_tables"]["Y2"]["5/6/4"]
    assert results["Y2"]["values"] is not None and len(results["Y2"]["statistics_020"]["per_frame"]) == 15
    assert results["Y3"]["row"] is None or results["Y3"]["floors"] == lock["floor_tables"]["Y3"][results["Y3"]["row"]]
    labels = results["outcome"]["labels"]
    assert labels[0] in rd.OUTCOME_Y1 and labels[1] in rd.OUTCOME_Y2 and labels[2] in rd.OUTCOME_Y3
    assert results["agreement"]["max_difference"] <= rc.STATISTIC_AGREEMENT_TOLERANCE
    assert "Y1" in results["ceiling"] and set(results["ceiling"]) <= {"Y1", "Y2"} and results["ceiling"]["Y1"]["error_split"] is not None
    executed_targets = [key for key in spy.counts if key in set(classes["S2-TARGET"])]
    assert executed_targets and all(spy.counts[key] == 1 for key in executed_targets)
    invalid = [frame_id for frame_id, entry in results["stage1"]["frames"].items() if not entry["valid"]]
    assert set(invalid) == invalid_fresh
    for frame in confirmation.frames:
        keys = {pm.Prompt(frame, int(token["token_id"]), token["word"]).key for token in confirmation.tokens}
        assert (keys <= set(state["executed_prompt_keys"])) is (frame.frame_id not in invalid)
    with pytest.raises(rd.PhaseError):
        runner.confirm()  # once
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "Experiment 021 Report" in report and "stage 2" in report and "Interpretation limit" in report
    assert "exposed-like draw median" in report and "fresh percentile among the draws" in report and "percentile among the row's exposed-like draws" in report
    arrays = torch.load(runner.draws_path)
    first_key = sorted(arrays["Y2"])[0]
    arrays["Y2"][first_key] = arrays["Y2"][first_key] + 1e-9
    torch.save(arrays, runner.draws_path)
    with pytest.raises(rd.PhaseError, match="draw values"):
        runner.report()


def test_a_replaced_inherited_lock_cannot_reuse_the_state_or_the_calibration_record(calibrated, sandbox, monkeypatch):
    runner, logs = calibrated
    _install_record(runner)
    runner.changed_paths = lambda commit: [rc.CALIBRATION_RELATIVE_PATH]
    world, fake, _ = sandbox
    record = json.loads((runner.root / rc.CALIBRATION_RELATIVE_PATH).read_text())

    def resealed(lock: dict, **changes) -> dict:
        out = {**dict(lock), **changes}
        out["content_sha256"] = rc.content_digest(out)  # self-consistent, and not the frozen object
        return out

    lock_011 = resealed(fake[3], sigma_T=fake[3]["sigma_T"] + 1e-3)
    lock_012 = resealed(fake[4], sigma_T=fake[4]["sigma_T"] + 1e-3)
    lock_017 = resealed(fake[5], lock_016_sha256="0" * 64)
    chained_012 = resealed(fake[4], lock_011_sha256=lock_011["content_sha256"])  # 012 re-pointed at the replaced 011
    # a replaced 011 alone breaks 012's own binding to it: refused before anything else
    alone, _ = make_runner(sandbox, lock_011_loader=lambda path: dict(lock_011))
    with pytest.raises(rd.PhaseError, match="012 lock does not carry"):
        alone.lock()
    cases = (({"lock_011": lock_011, "lock_012": chained_012}, {"lock_011_loader": lambda path: dict(lock_011), "lock_012_loader": lambda path: dict(chained_012)}),
             ({"lock_012": lock_012}, {"lock_012_loader": lambda path: dict(lock_012)}),
             ({"lock_017": lock_017}, {"lock_017_loader": lambda path: dict(lock_017)}))
    for replaced, loaders in cases:
        tampered, _ = make_runner(sandbox, **loaders)
        with pytest.raises(rd.PhaseError, match="not the frozen ones"):
            tampered.lock()
        assert tampered.validate() == 1
        with pytest.MonkeyPatch.context() as guard:  # even with the frozen constants moved to the new locks, nothing recorded is reused
            guard.setattr(rc, "LOCK_DIGESTS", {**rc.LOCK_DIGESTS, **{key: value["content_sha256"] for key, value in replaced.items()}})
            with pytest.raises(rd.PhaseError, match="differ from the recorded run"):
                tampered.lock()
            digests = tampered._inputs()[5]
            assert all(digests[key] == value["content_sha256"] for key, value in replaced.items())
            with pytest.raises(rd.PhaseError, match="different inputs"):
                rc.assert_record_inputs(record, digests)
    assert not (runner.output_dir / "candidate-lock.json").exists()  # no candidate lock was written by any of them


def test_a_failure_after_stage_two_keeps_every_fresh_measurement_on_disk(calibrated, sandbox, monkeypatch):
    runner, logs = calibrated
    _install_record(runner)
    runner.changed_paths = lambda commit: [rc.CALIBRATION_RELATIVE_PATH]
    _lock(runner)
    _install_lock(runner)
    runner.changed_paths = lambda commit: [rc.LOCK_RELATIVE_PATH, rc.PREDICTIONS_RELATIVE_PATH, rc.CALIBRATION_RELATIVE_PATH]
    monkeypatch.setattr(rc, "score_021", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic scoring failure")))
    assert runner.confirm() == 2
    state = rd.load_results_state(runner.results_path)
    stage2 = state["confirmation"]["stage2"]
    saved = torch.load(stage2["path"])
    for block in ("Y1", "Y2"):
        for part in ("exposed", "fresh", "ceiling"):
            if saved[block][part] is not None:
                assert rc.tensor_digest(saved[block][part]["measured"]) == stage2["tables_sha256"][block][part]["measured"]
    assert saved["Y1"]["exposed"]["measured"].shape[0] == len(state["confirmation"]["tokens_meta"]) * 108
    assert state["confirmation"]["incident"]["message"] == "synthetic scoring failure" and state["confirmation"]["incident"]["closure_020_recheck"] == {"ok": True}


def test_a_failing_identity_after_stage_two_is_an_incident_with_the_measurements_already_on_disk(calibrated, sandbox, monkeypatch):
    runner, logs = calibrated
    _install_record(runner)
    runner.changed_paths = lambda commit: [rc.CALIBRATION_RELATIVE_PATH]
    _lock(runner)
    _install_lock(runner)
    runner.changed_paths = lambda commit: [rc.LOCK_RELATIVE_PATH, rc.PREDICTIONS_RELATIVE_PATH, rc.CALIBRATION_RELATIVE_PATH]
    original = rc.stage_two_021

    def strict_after_stage_one(*args, **kwargs):  # stage 1 enforced its identities already; tighten the logit identity for stage 2's only
        monkeypatch.setattr(rd, "LOGIT_IDENTITY_TOLERANCE", 0.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(rc, "stage_two_021", strict_after_stage_one)
    scored = []
    monkeypatch.setattr(rc, "score_021", lambda *args, **kwargs: scored.append(1))
    assert runner.confirm() == 2
    state = rd.load_results_state(runner.results_path)
    assert state["confirmation"]["incident"]["message"].startswith("logit failed") and not scored
    stage2 = state["confirmation"]["stage2"]
    saved = torch.load(stage2["path"])
    assert rc.tensor_digest(saved["Y1"]["fresh"]["measured"]) == stage2["tables_sha256"]["Y1"]["fresh"]["measured"] and stage2["identities"]["logit"] > 0.0


def test_the_barrier_refuses_a_y2_row_that_the_reread_verdicts_do_not_select(calibrated, sandbox, monkeypatch):
    runner, logs = calibrated
    _install_record(runner)
    runner.changed_paths = lambda commit: [rc.CALIBRATION_RELATIVE_PATH]
    _lock(runner)
    _install_lock(runner)
    runner.changed_paths = lambda commit: [rc.LOCK_RELATIVE_PATH, rc.PREDICTIONS_RELATIVE_PATH, rc.CALIBRATION_RELATIVE_PATH]
    calls = {"n": 0}
    original = rc.select_y2_row

    def drifting(frames, tables):
        calls["n"] += 1
        selection = original(frames, tables)
        if calls["n"] > 1:  # the barrier's recomputation disagrees with what stage 1 recorded
            selection = {**selection, "composition": [0, 0, 0]}
        return selection

    monkeypatch.setattr(rc, "select_y2_row", drifting)
    spy = ExecutionSpy(monkeypatch)
    with pytest.raises(rd.PhaseError, match="re-read validity verdicts"):
        runner.confirm()
    confirmation = runner._inputs()[4]
    assert not (set(rd.manifest_classes(confirmation)["S2-TARGET"]) & set(spy.counts))  # no fresh cue prompt ran
    state = rd.load_results_state(runner.results_path)
    assert "incident" in state["confirmation"]


# ---------------------------------------------------------------------------
# The tests that need the fake itself: 020's call order, the environment check, stage 2's equivalence.


def _small_pool(pool, frames_per_template=1):
    frames = tuple(frame for template in pm.TEMPLATE_ORDER for frame in [f for f in pool.frames if f.template_id == template][:frames_per_template])
    return cs.Pool008(frames, pool.frame_origin, pool.tokens, pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)


@pytest.mark.slow
def test_the_rematerialization_calls_020s_functions_in_020s_order_and_checks_the_environment_first(fake_world, monkeypatch):
    manifest, small, digests, lock_011, lock_012, lock_017 = fake_world
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    monkeypatch.setattr(rd, "EXPLORE_TOKEN_LIMIT", 3)
    pool = _small_pool(small)
    confirmation = rd.load_confirmation(ROOT / rd.CONFIRMATION_RELATIVE_PATH, small, digests)
    names = ("capture_frame_020", "measure_pair", "predict_pair", "pair_identities", "inherited_reproduction", "ceiling_prediction")
    calls: list[tuple] = []
    for name in names:
        original = getattr(rd, name)

        def recording(*args, _name=name, _original=original, **kwargs):
            if _name == "capture_frame_020":
                calls.append((_name, args[2].key))
            elif _name == "measure_pair":
                calls.append((_name, args[1].frame.frame_id, args[3]))
            else:
                calls.append((_name, next(arg for arg in args if isinstance(arg, rd.FrameState020)).frame.frame_id))
            return _original(*args, **kwargs)

        monkeypatch.setattr(rd, name, recording)
    state = rd.new_results_state(digests={key: digests[key] for key in rd.CONFIRMATION_DIGEST_KEYS} | {"confirmation_020": confirmation.content_sha256},
                                 protocol_code_commit=COMMIT_A, git_dirty=False, versions={"torch": "test"})
    exploration = rd.run_exploration(make_fake_model(), pool, lock_011=lock_011, lock_012=lock_012, lock_017=lock_017, confirmation=confirmation, state=state, results_path=None)
    order_020, calls[:] = list(calls), []
    ledger = frozenset(state["executed_prompt_keys"])
    manifest_keys = frozenset(prompt.key for prompt in confirmation.all_prompts)
    table, context = rc.rematerialize(make_fake_model(), pool, lock_011=lock_011, lock_012=lock_012, lock_017=lock_017, exploration_020=json.loads(pm.canonical_json(exploration)),
                                      ledger_020=ledger, manifest_keys=manifest_keys, confirmation=confirmation)
    assert calls == order_020 and {prompt.key for prompt in context["executed"]} == ledger
    assert rc.reproduction_gate(table, json.loads(pm.canonical_json(exploration)))["max_difference"] == 0.0
    # the environment check runs between the loops: a planted drift stops the phase before any cue prompt
    for plant in (lambda e: e["locked_states"][pool.frames[0].frame_id]["h6"].__setitem__(0, e["locked_states"][pool.frames[0].frame_id]["h6"][0] + 1e-8),
                  lambda e: e["template_bases"]["cardinal"]["4"].__setitem__(0, e["template_bases"]["cardinal"]["4"][0] + 1e-8)):
        planted = json.loads(pm.canonical_json(exploration))
        plant(planted)
        calls[:] = []
        with pytest.raises(rc.EnvironmentIncident) as caught:
            rc.rematerialize(make_fake_model(), pool, lock_011=lock_011, lock_012=lock_012, lock_017=lock_017, exploration_020=planted, ledger_020=ledger, manifest_keys=manifest_keys,
                             confirmation=confirmation)
        assert not any(call[0] == "measure_pair" for call in calls) and len(caught.value.executed) == len(pool.frames)


@pytest.mark.slow
def test_stage_two_021_is_020s_stage_two_plus_the_ceiling_from_the_same_measurement(fake_world, monkeypatch):
    manifest, small, digests, lock_011, lock_012, lock_017 = fake_world
    confirmation = rd.load_confirmation(ROOT / rd.CONFIRMATION_RELATIVE_PATH, small, digests)
    model = make_fake_model()
    pool = _small_pool(small)
    weights = pm.Weights.from_model(model)
    nouns = rd.NounSet.build(weights, pool.nouns, confirmation.nouns)
    axis_T = pm.SiteAxis("T", torch.zeros(int(model.cfg.d_model), dtype=torch.float64), torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64), float(lock_011["sigma_T"]))
    states = {frame.frame_id: rd.capture_frame_020(model, ht.HeadWeights.from_model(model), pool.reference_prompt(frame), nouns, axis_T) for frame in pool.frames}
    lw = lc.LayerWeights.from_model(model, layers=rd.PROGRAM_LAYERS)
    programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in rd.PROGRAM_LAYERS}
    program = rd.ReadoutProgram.from_model(model)
    chain = rd.chain_from_locks(lock_011, lock_012, lock_017, lw, programs, pool)
    rows16 = {frame_id: atp.reference_rows(programs, state.state_017.x1_all, state.state_017.x2_all) for frame_id, state in states.items()}
    bases = {template: {4: torch.zeros(int(model.cfg.d_model), dtype=torch.float64), 5: torch.zeros(int(model.cfg.d_model), dtype=torch.float64)} for template in pm.TEMPLATE_ORDER}
    tokens = confirmation.tokens[:2]
    rows = rd.prediction_rows(program, chain, weights, states, rows16, nouns, list(pool.frames), list(tokens), bases, axis_T.direction.double())
    lock = {"content_sha256": "0" * 64, "template_bases": {}, "locked_states": {frame_id: rd.locked_state(state) for frame_id, state in states.items()}, "predictions": {"rows": rows}}
    two_tokens = type("C", (), {k: getattr(confirmation, k) for k in ("reference_ids", "frames", "exposed_frame_ids", "nouns", "token_prompts", "exposed_frame_prompts", "content_sha256")})()
    two_tokens.tokens = tuple(tokens)
    stage1 = {"frames": {}, "states": {}, "rows": []}
    ours = rc.stage_two_021(model, pool, two_tokens, lock, lock_011, lock_012, lock_017, stage1)
    theirs = rd.stage_two(model, pool, two_tokens, lock, lock_011, lock_012, lock_017, stage1)
    assert ours["identities"] == theirs["identities"] and ours["noun_keys"] == theirs["noun_keys"] and ours["fresh_noun_keys"] == theirs["fresh_noun_keys"]
    for name in ("exposed", "fresh", "no_l5", "base"):
        a, b = ours["tables"]["Y1"][name], theirs["tables"]["Y1"][name]
        assert (a.cues, a.frames, a.templates, a.noun_keys) == (b.cues, b.frames, b.templates, b.noun_keys) and torch.equal(a.measured, b.measured) and torch.equal(a.predicted, b.predicted)
    assert torch.equal(ours["tables"]["Y1"]["dT"], theirs["tables"]["Y1"]["dT"]) and ours["tables"]["Y2"]["exposed"] is None and theirs["tables"]["Y2"]["exposed"] is None
    frame, token = pool.frames[0], tokens[0]
    state = rd.state_from_locked(lock["locked_states"][frame.frame_id], frame)
    measurement = rd.measure_pair(model, state, nouns, token["word"], int(token["token_id"]), c_ref=nouns.contrast_from_residual(program, state.h6))
    expected = rd.ceiling_prediction(program, state, nouns, measurement.dx3)[list(nouns.exposed_scorable)]
    assert torch.equal(ours["tables"]["Y1"]["ceiling"].predicted[0], expected) and torch.equal(ours["tables"]["Y1"]["ceiling"].measured, ours["tables"]["Y1"]["exposed"].measured)
