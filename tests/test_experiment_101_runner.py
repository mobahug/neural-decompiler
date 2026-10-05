"""Experiment 101's runner on the six-layer fake, inside the fake world of Experiment 025's runner test.

020 is explored and closed on the fake; 022–024 are frozen, calibrated and locked; 025 is frozen. So 025's
confirmation file exists in 025's committed format, and its digests are pointed at the fake world's
(``ri.INHERITED_025``). The world uses an explicit test configuration, never a patched production constant:
``config`` below, with the fake's reference cue, its two explored cues of the coordinated frames (the fake's 020 ledger
holds only those), two isotropic directions, and revision-trigger limits that the fake's weak shifts cannot reach. On
that world it checks every phase and its refusals; the calibration's key checks and its stop; the no-forward-pass lock;
stage 1's only-reference runs, its sealed model and its files; the barrier's incidents before any S2 key; the
accounting; the integrity incidents; the single result write; the secondary step; and the report.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_intervention as ri
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from test_experiment_020_runner import fake_world, make_fake_model  # noqa: F401 — fake_world is a fixture
from test_experiment_022_runner import COMMIT_A
from test_experiment_022_runner import calibrated, frozen, world  # noqa: F401 — fixtures
from test_experiment_023_runner import base023, calibrated023, extracted, frozen023, locked023  # noqa: F401 — fixtures
from test_experiment_024_runner import _install
from test_experiment_024_runner import base024, calibrated024, frozen024, locked024  # noqa: F401 — fixtures
from test_experiment_025_runner import FAKE as FAKE_025
from test_experiment_025_runner import _apply as apply_025
from test_experiment_025_runner import base025, frozen025, toy_tokenizer_025  # noqa: F401 — fixtures
from test_experiment_025_runner import make_runner as make_runner_025

ROOT = Path(__file__).parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_101_runner", ROOT / "experiments/101-read-component-intervention/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def toy_tokenizer_101(manifest, pool):
    """025's toy tokenizer plus every 101 candidate's words (fresh ids below the fake's vocabulary)."""
    base = toy_tokenizer_025(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    for text in ri.FRAME_CANDIDATES:
        for index, piece in enumerate(text.replace("{cue}", "").split()):
            word = piece if index == 0 else " " + piece
            if word not in vocabulary:
                vocabulary[word] = next_id
                next_id += 1
    assert next_id < 60000
    return type(base)(vocabulary)


@pytest.fixture(scope="module")
def base101(world, base023, base024, base025, frozen025, tmp_path_factory):
    """025 frozen on the fake; the 101 world constants: the digests of the fake's 025 confirmation file, and the test
    configuration built from the fake's own coordinated reference cue and its two explored cues."""
    root = tmp_path_factory.mktemp("base101")
    _clone_tree(frozen025, root)
    path = root / cr.CONFIRMATION_RELATIVE_PATH
    inherited = {"confirmation_file_sha256": rc.file_sha256(path), "confirmation_content_sha256": json.loads(path.read_text())["content_sha256"]}
    with pytest.MonkeyPatch.context() as patch:
        apply_025(patch, world, base023, base024, base025)
        inputs = make_runner_025(root, world, FAKE_025)[0]._inputs()
    pool = inputs.pool
    reference = int(pool.reference_ids[ri.TEMPLATE])
    explored = [(name, int(token_id)) for name, token_id in pool.tokens if int(token_id) != reference][:2]
    ledger = frozenset(inputs.closure["ledger"])
    for frame in (frame for frame in pool.frames if frame.template_id == ri.TEMPLATE):
        assert all(f"{frame.frame_id}|{name}|{token_id}" in ledger for name, token_id in explored)
    config = ri.Configuration(name="fake-world", n_frames=36, frame_candidates=ri.FRAME_CANDIDATES, expected_picks=ri.EXPECTED_PICKS, reference=("ref", reference),
                              cues=(("two", explored[0][1]), ("alt", explored[1][1])), k_iso=2, n_scored_nouns=79, exposed_frames=36, d_threshold=25,
                              p_threshold=27, precondition=30, tau_rank=33, max_overshoot_frames=36, max_weak_frames=36)
    return {"root": root, "inherited": inherited, "config": config}


def _clone_tree(source: Path, target: Path) -> None:
    """A copy-on-write clone of a stage directory where the filesystem supports it (APFS), else a plain copy: the stage
    roots are ~75 MB each, and a test that modifies a cloned file changes only its own copy."""
    target.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin" and subprocess.run(["cp", "-cR", f"{source}/.", str(target)], capture_output=True).returncode == 0:
        return
    shutil.copytree(source, target, dirs_exist_ok=True)


def _apply(patch, world, base, base24, base25, base101) -> None:
    apply_025(patch, world, base, base24, base25)
    for key, value in base101["inherited"].items():
        patch.setitem(ri.INHERITED_025, key, value)


def _unreadable_outputs(patch) -> None:
    """022's calibration table and 023's, 024's and 025's local outputs can never be opened by a 101 phase."""
    original_load, original_read_bytes, original_read_text = torch.load, Path.read_bytes, Path.read_text

    def forbidden(path) -> bool:
        text = str(path)
        return text.endswith("calibration-table.pt") or any(f"outputs/experiment-0{n}/" in text for n in (23, 24, 25))

    def load(path, *args, **kwargs):
        if forbidden(path):
            raise AssertionError(f"a forbidden file was opened by a 101 phase: {path}")
        return original_load(path, *args, **kwargs)

    def read_bytes(self):
        if forbidden(self):
            raise AssertionError(f"a forbidden file was read by a 101 phase: {self}")
        return original_read_bytes(self)

    def read_text(self, *args, **kwargs):
        if forbidden(self):
            raise AssertionError(f"a forbidden file was read by a 101 phase: {self}")
        return original_read_text(self, *args, **kwargs)

    patch.setattr(torch, "load", load)
    patch.setattr(Path, "read_bytes", read_bytes)
    patch.setattr(Path, "read_text", read_text)


def make_runner(root: Path, world, config, **overrides):
    manifest, small, digests, lock_011, lock_012, lock_017 = world["fake"]
    logs: list[str] = []
    arguments = dict(root=root, results_path=root / "outputs/experiment-101/results.json", report_path=root / "outputs/experiment-101/report.md",
                     model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_101(manifest, small),
                     lock_011_loader=lambda path: dict(lock_011), lock_012_loader=lambda path: dict(lock_012), lock_017_loader=lambda path: dict(lock_017),
                     git_state=lambda: {"commit": COMMIT_A, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True,
                     changed_paths=lambda commit: [], log=logs.append, config=config)
    arguments.update(overrides)
    return runner_module.Runner(**arguments), logs


def _stage(world, base, base24, base25, base101, tmp_path_factory, name: str, source: Path, step) -> Path:
    root = tmp_path_factory.mktemp(name)
    _clone_tree(source, root)
    with pytest.MonkeyPatch.context() as patch:
        _apply(patch, world, base, base24, base25, base101)
        _unreadable_outputs(patch)
        runner, logs = make_runner(root, world, base101["config"])
        step(runner, logs, root, patch)
    return root


class ForwardSpy:
    """Counts every forward by its prompt key: patched runs and plain captures."""

    def __init__(self, monkeypatch):
        self.patched: Counter[str] = Counter()
        self.plain: Counter[str] = Counter()
        original_patched, original_plain = pm.run_patched, pm.capture_prompt

        def patched(model, prompt, replacements, sources, *, capture_sites=()):
            self.patched[prompt.key] += 1
            return original_patched(model, prompt, replacements, sources, capture_sites=capture_sites)

        def plain(model, prompt, sites):
            self.plain[prompt.key] += 1
            return original_plain(model, prompt, sites)

        monkeypatch.setattr(pm, "run_patched", patched)
        monkeypatch.setattr(pm, "capture_prompt", plain)


@pytest.fixture(scope="module")
def frozen101(world, base023, base024, base025, base101, tmp_path_factory):
    def step(runner, logs, root, patch):
        assert runner.freeze() == 0, logs[-2:]

    return _stage(world, base023, base024, base025, base101, tmp_path_factory, "frozen101", base101["root"], step)


@pytest.fixture(scope="module")
def calibrated101(world, base023, base024, base025, base101, frozen101, tmp_path_factory):
    record: dict = {}

    def step(runner, logs, root, patch):
        spy = ForwardSpy(patch)
        assert runner.calibrate() == 0, logs[-4:]
        record["plain"], record["patched"] = spy.plain, spy.patched
        _install(runner.output("candidate-calibration-v1.json"), root / ri.CALIBRATION_RELATIVE_PATH)

    root = _stage(world, base023, base024, base025, base101, tmp_path_factory, "calibrated101", frozen101, step)
    return {"root": root, **record}


@pytest.fixture(scope="module")
def locked101(world, base023, base024, base025, base101, calibrated101, tmp_path_factory):
    def step(runner, logs, root, patch):
        assert runner.lock() == 0, logs[-4:]
        _install(runner.output("candidate-lock.json"), root / ri.LOCK_RELATIVE_PATH)
        _install(runner.output("candidate-preregistration.md"), root / ri.PREREGISTRATION_RELATIVE_PATH)

    return _stage(world, base023, base024, base025, base101, tmp_path_factory, "locked101", calibrated101["root"], step)


@pytest.fixture(scope="module")
def staged101(world, base023, base024, base025, base101, locked101, tmp_path_factory):
    record: dict = {}

    def step(runner, logs, root, patch):
        spy = ForwardSpy(patch)
        assert runner.stage1() == 0, logs[-4:]
        record["plain"], record["patched"] = spy.plain, spy.patched

    root = _stage(world, base023, base024, base025, base101, tmp_path_factory, "staged101", locked101, step)
    return {"root": root, **record}


@pytest.fixture(scope="module")
def confirmed101(world, base023, base024, base025, base101, staged101, tmp_path_factory):
    record: dict = {"writes": []}

    def step(runner, logs, root, patch):
        spy = ForwardSpy(patch)
        original_write = runner_module.Runner._write

        def recording(self, state):
            confirmation = state.get("confirmation") or {}
            record["writes"].append({"confirmation": sorted(confirmation), "status": state["phases"]["confirm"]["status"],
                                     "ledger": len(state["executed_prompt_keys"])})
            return original_write(self, state)

        patch.setattr(runner_module.Runner, "_write", recording)
        assert runner.confirm() == 0, logs[-4:]
        record["plain"], record["patched"], record["logs"] = spy.plain, spy.patched, list(logs)

    root = _stage(world, base023, base024, base025, base101, tmp_path_factory, "confirmed101", staged101["root"], step)
    return {"root": root, **record}


@pytest.fixture
def sandbox(world, base023, base024, base025, base101, tmp_path, monkeypatch):
    """A per-test copy of a stage directory, with every patch the fake needs and the forbidden outputs unreadable."""
    _apply(monkeypatch, world, base023, base024, base025, base101)
    _unreadable_outputs(monkeypatch)

    def copy(source: Path) -> Path:
        root = tmp_path / "root"
        _clone_tree(source, root)
        return root

    return copy


def _state(runner):
    return rd.load_results_state(runner.results_path)


def _refuse(*args, **kwargs):
    raise AssertionError("a model or tokenizer was loaded where none may be")


# ---------------------------------------------------------------------------
# The parser, validate, freeze.


def test_parser_has_exactly_the_seven_phases_no_option_and_production_only():
    parser = runner_module.build_parser()
    for phase in ri.PHASES:
        assert parser.parse_args([phase]).phase == phase
    for forbidden_args in (["explore"], ["confirm", "--config", "fake"], ["freeze", "--frames", "4"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden_args)
    assert runner_module.PHASES == ("validate", "freeze", "calibrate", "lock", "stage1", "confirm", "report")
    assert runner_module.Runner().config is ri.PRODUCTION
    with pytest.raises(ri.PhaseError, match="non-production configuration"):
        runner_module.Runner(config=dataclasses.replace(ri.PRODUCTION, name="other"))


def test_freeze_is_tokenizer_only_and_its_stops_write_nothing(world, base101, sandbox):
    root = sandbox(base101["root"])
    config = base101["config"]
    target = root / ri.CONFIRMATION_RELATIVE_PATH
    deviating = dataclasses.replace(config, expected_picks=tuple(range(2, 38)))
    runner, logs = make_runner(root, world, deviating, model_loader=_refuse)
    assert runner.freeze() == 3 and "not the expected" in logs[-1] and not target.exists() and not runner.results_path.exists()
    runner, logs = make_runner(root, world, config, model_loader=_refuse)
    assert runner.validate() == 0 and "not frozen yet" in logs[-1]
    assert runner.freeze() == 0, logs
    payload = json.loads(target.read_text())
    assert payload["picks"] == list(range(1, 37)) and payload["configuration"] == config.to_json()
    assert payload["counts"] == {"frames": 36, "runs_per_frame": config.runs_per_frame, "S1-REF": 36, "S2-TARGET": 36 * config.runs_per_frame,
                                 "untagged_prompts": 36 * 3}
    assert len(set(payload["manifest"]["S2-TARGET"])) == 36 * config.runs_per_frame and not runner.results_path.exists()
    with pytest.raises(ri.PhaseError, match="runs once"):
        runner.freeze()
    assert runner.validate() == 0 and "36 frames" in logs[-1]
    other, logs = make_runner(root, world, dataclasses.replace(config, k_iso=3), model_loader=_refuse)
    assert other.validate() == 1 and "different configuration" in logs[-1]


# ---------------------------------------------------------------------------
# Calibrate.


def test_calibration_runs_only_spent_exposed_keys_each_once_and_writes_its_candidate(world, base101, calibrated101, sandbox):
    root = sandbox(calibrated101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    state = _state(runner)
    assert state["phases"]["calibrate"]["status"] == "complete" and state["executed_prompt_keys"] == []
    planned = state["calibration"]["planned"]["count"]
    assert planned == 36 * (1 + base101["config"].runs_per_frame) and len(state["calibration_executed"]) == planned
    assert state["calibration"]["accounting"]["equal"] is True
    plain, patched = calibrated101["plain"], calibrated101["patched"]
    assert sum(plain.values()) + sum(patched.values()) == planned
    payload = json.loads((root / ri.CALIBRATION_RELATIVE_PATH).read_text())
    assert payload["content_sha256"] == rc.content_digest(payload) and payload["tau"]["rank"] == 33
    assert not any(entry["fired"] for entry in payload["triggers"].values()) and payload["gates"]["failures"] == []
    assert payload["environment"]["passed"] is True and payload["keys"]["count"] == planned
    for name in ("G4_I3", "G4_I4", "G4_level1", "G4_readout", "G4_logit"):
        assert name in payload["gates"]["per_arm"]


def test_a_revision_trigger_stops_the_calibration_with_no_file_and_it_is_never_retried(world, base101, sandbox):
    root = sandbox(base101["root"])
    strict = dataclasses.replace(base101["config"], max_weak_frames=-1)
    runner, logs = make_runner(root, world, strict)
    assert runner.freeze() == 0, logs
    assert runner.calibrate() == 3 and "weak" in logs[-1]
    state = _state(runner)
    assert state["phases"]["calibrate"]["status"] == "stopped_for_review" and state["calibration"]["stopped"]["triggers"]["weak"]["fired"]
    assert not runner.output("candidate-calibration-v1.json").exists()
    with pytest.raises(ri.PhaseError, match="runs once"):
        runner.calibrate()


def test_calibrate_refuses_unfrozen_or_reserved_keys(world, base101, sandbox, monkeypatch):
    root = sandbox(base101["root"])
    runner, logs = make_runner(root, world, base101["config"], model_loader=_refuse)
    with pytest.raises(ri.PhaseError, match="frozen and committed first"):
        runner.calibrate()
    assert runner.freeze() == 0
    original = runner_module.Runner._calibration_schedules

    def foreign(self, inputs, frames):
        schedules = original(self, inputs, frames)
        first = schedules[0][1]
        swapped = ri.RunSpec(first.frame, first.arm, first.label, pm.Prompt(first.frame, 99, "nowhere"))
        return [(schedules[0][0], swapped, *schedules[0][2:]), *schedules[1:]]

    monkeypatch.setattr(runner_module.Runner, "_calibration_schedules", foreign)
    with pytest.raises(ri.PhaseError, match="outside 020's ledger"):
        runner.calibrate()
    assert not runner.results_path.exists()  # a refusal records nothing


# ---------------------------------------------------------------------------
# Lock.


def test_lock_runs_no_forward_binds_everything_and_refuses_a_rerun(world, base101, locked101, sandbox):
    root = sandbox(locked101)
    runner, logs = make_runner(root, world, base101["config"])
    lock = json.loads((root / ri.LOCK_RELATIVE_PATH).read_text())
    assert lock["content_sha256"] == rc.content_digest(lock) and lock["configuration"] == base101["config"].to_json()
    assert lock["calibration_101"]["file_sha256"] == rc.file_sha256(root / ri.CALIBRATION_RELATIVE_PATH)
    assert set(lock["geometry"]["iso_sha256"]) == {ri.frame_id_for(k) for k in range(1, 37)} and lock["tolerances"]["G4_level1"] == 2e-2
    assert len(lock["stage1"]["states"]["layout"]) == 36 * len(ri.STATE_FIELDS) and lock["stage1"]["table"]["fractions_for_weak"] is False
    assert (root / ri.PREREGISTRATION_RELATIVE_PATH).read_text() == ri.render_preregistration(lock)
    with pytest.raises(ri.PhaseError, match="already ran"):
        runner.lock()
    assert runner.validate() == 0 and "lock and preregistration verified" in logs[-1]


def test_lock_refuses_a_calibration_file_that_is_not_the_candidate(world, base101, calibrated101, sandbox):
    root = sandbox(calibrated101["root"])
    path = root / ri.CALIBRATION_RELATIVE_PATH
    payload = json.loads(path.read_text())
    payload["tau"]["rem"] = 0.01
    payload["content_sha256"] = rc.content_digest(payload)
    path.write_text(pm.canonical_json(payload) + "\n")
    runner, _ = make_runner(root, world, base101["config"], model_loader=_refuse)
    with pytest.raises(ri.PhaseError, match="not the candidate"):
        runner.lock()


# ---------------------------------------------------------------------------
# Stage 1.


def test_stage1_runs_only_the_36_reference_prompts_and_writes_the_bound_files(world, base101, staged101, sandbox):
    root = sandbox(staged101["root"])
    runner, logs = make_runner(root, world, base101["config"])
    assert sum(staged101["patched"].values()) == 0 and sum(staged101["plain"].values()) == 36
    assert all(key.split("|")[1] == "ref" and count == 1 for key, count in staged101["plain"].items())
    state = _state(runner)
    assert state["phases"]["stage1"]["status"] == "complete" and state["phases"]["stage1"]["awaiting_commit"] is True
    assert sorted(state["executed_prompt_keys"]) == sorted(f"{ri.frame_id_for(k)}|ref|{base101['config'].reference[1]}|S1" for k in range(1, 37))
    record = state["stage1"]
    from neural_decompiler import block0_completion as b0c

    assert record["digest"] == b0c.stage_one_digest(record) and len(record["state_digests"]) == 36
    for name, entry in record["files"].items():
        assert rc.file_sha256(root / entry["path"]) == entry["sha256"]
    table = json.loads((root / ri.STAGE1_TABLE).read_text())
    assert table["content_sha256"] == record["table_content_sha256"] and table["fractions_for_weak"] is False
    assert set(table["frames"]) == {ri.frame_id_for(k) for k in range(1, 37)} and set(table["frames"][ri.frame_id_for(1)]["cues"]) == {"two", "alt"}
    for label in ("two", "alt"):  # the fake's shifts are weak: the barrier's bit-for-bit recompute covers predicted-weak frames, switch off
        assert any(entry["cues"][label]["weak"] and entry["cues"][label]["fractions"] is None for entry in table["frames"].values())
    with pytest.raises(ri.PhaseError):
        runner.stage1()


def test_stage1_refuses_existing_files_and_records_a_geometry_mismatch_as_an_incident(world, base101, locked101, sandbox, monkeypatch):
    root = sandbox(locked101)
    (root / ri.STAGE1_DIR).mkdir(parents=True, exist_ok=True)
    (root / ri.STAGE1_TABLE).write_text("{}")
    runner, _ = make_runner(root, world, base101["config"])
    with pytest.raises(ri.PhaseError, match="already exists"):
        runner.stage1()
    (root / ri.STAGE1_TABLE).unlink()
    original = ri.iso_direction
    monkeypatch.setattr(ri, "iso_direction", lambda geometry, frame_id, j: original(geometry, frame_id, j + 1))
    spy = ForwardSpy(monkeypatch)
    assert runner.stage1() == 2
    state = _state(runner)
    assert state["phases"]["stage1"]["incidents"] and state["executed_prompt_keys"] == [] and not spy.plain and not spy.patched


def test_a_stage1_failure_after_the_ledger_spends_the_frames(world, base101, locked101, sandbox, monkeypatch):
    root = sandbox(locked101)
    runner, _ = make_runner(root, world, base101["config"])

    def broken(*args, **kwargs):
        raise RuntimeError("the table could not be computed")

    monkeypatch.setattr(ri, "prediction_table", broken)
    assert runner.stage1() == 2
    state = _state(runner)
    incident = state["phases"]["stage1"]["incidents"][-1]
    assert incident["after_ledger"] is True and len(state["executed_prompt_keys"]) == 36
    with pytest.raises(ri.PhaseError):
        runner.stage1()
    with pytest.raises(ri.PhaseError):
        runner.confirm()
    assert runner.report() == 0 and "NOT_INTERPRETABLE" in runner.report_path.read_text()


# ---------------------------------------------------------------------------
# Confirm.


def test_confirm_runs_every_s2_key_once_after_the_ledger_and_writes_one_result(world, base101, confirmed101, sandbox):
    root = sandbox(confirmed101["root"])
    runner, logs = make_runner(root, world, base101["config"])
    config = base101["config"]
    state = _state(runner)
    confirmation = state["confirmation"]
    assert state["phases"]["confirm"]["status"] == "complete" and confirmation["accounting"]["equal"] is True
    assert confirmation["accounting"]["executed"] == 36 * config.runs_per_frame == confirmation["accounting"]["manifest"]
    runs = Counter(confirmed101["plain"]) + Counter(confirmed101["patched"])
    assert sum(runs.values()) == 36 * config.runs_per_frame
    assert confirmation["gates"]["failures"] == [] and confirmation["barrier"]["passed"] is True and confirmation["g6_before_result"]["bitwise_equal"] is True
    results = confirmation["results"]
    assert results["n"] == 36 and results["outcome"]["label"] in ri.OUTCOMES[1:] and len(results["frames"]) == 36
    assert set(confirmation["secondary"]) >= {"values", "records", "statistics"} and "failures" not in confirmation["secondary"]
    writes = confirmed101["writes"]
    ledger_full = 36 + 36 * config.runs_per_frame
    first_s2 = next(i for i, write in enumerate(writes) if write["ledger"] == ledger_full)
    assert writes[first_s2]["status"] == "running" and writes[first_s2]["confirmation"] == ["barrier"]
    with_results = [i for i, write in enumerate(writes) if "results" in write["confirmation"]]
    assert with_results and writes[with_results[0]]["status"] == "complete"
    with pytest.raises(ri.PhaseError):
        runner.confirm()
    assert runner.report() == 0
    text = runner.report_path.read_text()
    assert results["outcome"]["label"] in text and _state(runner)["report"]["g6"]["bitwise_equal"] is True
    assert runner.report() == 0 and _state(runner)["report"]["renders"] == 2


def test_confirm_refuses_uncommitted_stage1_files_and_records_nothing(world, base101, staged101, sandbox):
    root = sandbox(staged101["root"])
    stage1 = {str(root / path) for path in ri.STAGE1_PATHS}
    runner, _ = make_runner(root, world, base101["config"], tracked=lambda path: str(path) not in stage1, model_loader=_refuse)
    before = (root / "outputs/experiment-101/results.json").read_bytes()
    with pytest.raises(ri.PhaseError, match="committed"):
        runner.confirm()
    assert (root / "outputs/experiment-101/results.json").read_bytes() == before


@pytest.mark.parametrize("tamper", ["table_bytes", "table_values", "states", "ledger"])
def test_the_barrier_records_an_incident_before_any_s2_key(world, base101, staged101, sandbox, tamper):
    root = sandbox(staged101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    if tamper == "table_bytes":
        path = root / ri.STAGE1_TABLE
        path.write_text(path.read_text() + " ")
    elif tamper == "table_values":
        path = root / ri.STAGE1_TABLE
        table = json.loads(path.read_text())
        entry = table["frames"][ri.frame_id_for(1)]["cues"]["two"]["arms"]["PL"]
        entry["mean"] += 1e-9
        table["content_sha256"] = rc.content_digest(table)
        path.write_text(pm.canonical_json(table) + "\n")
        state = _state(runner)
        state["stage1"]["files"]["table"]["sha256"] = rc.file_sha256(path)
        state["stage1"]["table_content_sha256"] = table["content_sha256"]
        from neural_decompiler import block0_completion as b0c

        state["stage1"]["digest"] = b0c.stage_one_digest(state["stage1"])
        runner._write(state)
    elif tamper == "states":
        path = root / ri.STAGE1_STATES_DATA
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(bytes(data))
    else:
        state = _state(runner)
        state["executed_prompt_keys"].append(f"{ri.frame_id_for(1)}|ref|{base101['config'].reference[1]}|REF")
        runner._write(state)
    with pytest.MonkeyPatch.context() as patch:
        spy = ForwardSpy(patch)
        assert runner.confirm() == 2
    state = _state(runner)
    assert state["phases"]["confirm"]["incidents"] and not spy.plain and not spy.patched
    assert not any(key.rsplit("|", 1)[1] not in ("S1",) and key.endswith("@two") for key in state["executed_prompt_keys"])


def test_a_gate_failure_after_the_measurements_is_an_integrity_incident_with_no_statistic(world, base101, staged101, sandbox, monkeypatch):
    root = sandbox(staged101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    monkeypatch.setitem(ri.TOLERANCES, "G4_I3", 0.0)
    with pytest.raises(ri.PhaseError, match="tolerance"):
        runner.confirm()  # the lock binds the tolerances: a changed one is refused before any prompt
    monkeypatch.setitem(ri.TOLERANCES, "G4_I3", 1e-4)
    from neural_decompiler import upstream_localization as ul

    monkeypatch.setattr(ul, "i3_error", lambda predicted, measured: 1.0)  # the composition no longer reproduces the plain pairs
    assert runner.confirm() == 2
    state = _state(runner)
    confirmation = state["confirmation"]
    assert "integrity failure" in confirmation["incident"]["message"] and "G4_I3" in confirmation["gates"]["failures"]
    assert "results" not in confirmation and "secondary" not in confirmation and state["phases"]["confirm"]["status"] == "running"
    assert runner.report() == 0 and "NOT_INTERPRETABLE" in runner.report_path.read_text()


def test_a_landing_failure_inside_a_run_is_an_incident_and_the_partial_runs_are_kept(world, base101, staged101, sandbox, monkeypatch):
    root = sandbox(staged101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    original = ri.residual_replacement
    calls = {"n": 0}

    def drifting(*args, **kwargs):
        calls["n"] += 1
        replacement = original(*args, **kwargs)
        return replacement.double() if calls["n"] == 5 else replacement

    monkeypatch.setattr(ri, "residual_replacement", drifting)
    assert runner.confirm() == 2
    confirmation = _state(runner)["confirmation"]
    assert confirmation["incident"] and confirmation["partial"]["n_executed"] > 0 and "results" not in confirmation


def test_a_failed_result_write_is_an_incident_and_a_secondary_failure_leaves_the_result(world, base101, staged101, sandbox, monkeypatch):
    root = sandbox(staged101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    monkeypatch.setattr(ri, "secondary_statistics", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no secondary")))
    assert runner.confirm() == 0
    confirmation = _state(runner)["confirmation"]
    assert confirmation["results"]["outcome"]["label"] and confirmation["secondary"]["failures"]["statistics"]["message"] == "no secondary"


def test_a_report_time_g6_mismatch_voids_the_label_in_the_report(world, base101, confirmed101, sandbox):
    root = sandbox(confirmed101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    path = root / ri.STAGE1_TABLE
    table = json.loads(path.read_text())
    table["frames"][ri.frame_id_for(2)]["cues"]["two"]["arms"]["PL"]["mean"] += 1.0
    path.write_text(pm.canonical_json(table) + "\n")
    assert runner.report() == 0
    text = runner.report_path.read_text()
    assert "NOT_INTERPRETABLE" in text and "void" in text and _state(runner)["confirmation"]["results"]  # the result is never rewritten


@pytest.mark.parametrize("gate", ["G2_r", "G5_output", "G6_state"])
def test_each_gate_failure_after_the_measurements_is_an_integrity_incident(world, base101, staged101, sandbox, monkeypatch, gate):
    root = sandbox(staged101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    original_tol, original_bit = ri.GateBook.tol, ri.GateBook.bit
    monkeypatch.setattr(ri.GateBook, "tol", lambda self, name, value, where, arm: original_tol(self, name, 1.0 if name == gate else value, where, arm))
    monkeypatch.setattr(ri.GateBook, "bit", lambda self, name, ok, where: original_bit(self, name, False if name == gate else ok, where))
    assert runner.confirm() == 2
    confirmation = _state(runner)["confirmation"]
    assert gate in confirmation["gates"]["failures"] and "integrity failure" in confirmation["incident"]["message"] and "results" not in confirmation


def test_a_non_finite_contrast_is_an_incident(world, base101, staged101, sandbox, monkeypatch):
    root = sandbox(staged101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    original = rd.NounSet.contrasts
    calls = {"n": 0}

    def contrasts(self, logits):
        calls["n"] += 1
        if calls["n"] == 7:
            raise pm.IncidentError("non-finite contrast for cat")
        return original(self, logits)

    monkeypatch.setattr(rd.NounSet, "contrasts", contrasts)
    assert runner.confirm() == 2
    confirmation = _state(runner)["confirmation"]
    assert "non-finite contrast" in confirmation["incident"]["message"] and "results" not in confirmation


def test_a_failed_result_write_is_an_incident_and_never_a_result(world, base101, staged101, sandbox, monkeypatch):
    root = sandbox(staged101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    original = runner_module.Runner._write

    def failing(self, state):
        if "results" in (state.get("confirmation") or {}):
            raise OSError("the disk is full")
        return original(self, state)

    monkeypatch.setattr(runner_module.Runner, "_write", failing)
    assert runner.confirm() == 2
    state = _state(runner)
    assert "disk is full" in state["confirmation"]["incident"]["message"] and "results" not in state["confirmation"]
    assert state["phases"]["confirm"]["status"] == "running"


def test_stage1_refuses_a_path_the_write_hook_does_not_allow_before_its_ledger(world, base101, locked101, sandbox):
    root = sandbox(locked101)

    def hook(paths):
        if any(path.name == "table.json" for path in paths):
            raise ri.PhaseError("the launcher's allow-list does not name stage1/table.json")

    runner, _ = make_runner(root, world, base101["config"], write_check=hook)
    with pytest.MonkeyPatch.context() as patch:
        spy = ForwardSpy(patch)
        with pytest.raises(ri.PhaseError, match="allow-list"):
            runner.stage1()
    state = _state(runner)
    assert state["executed_prompt_keys"] == [] and state["phases"]["stage1"]["status"] == "not_started" and not spy.plain


def test_a_calibration_gate_failure_stops_for_review_with_the_values_and_no_file(world, base101, frozen101, sandbox, monkeypatch):
    root = sandbox(frozen101)
    runner, logs = make_runner(root, world, base101["config"])
    original_tol = ri.GateBook.tol
    monkeypatch.setattr(ri.GateBook, "tol", lambda self, name, value, where, arm: original_tol(self, name, 0.5 if name == "G4_level1" else value, where, arm))
    assert runner.calibrate() == 3 and "gate_failure" in logs[-1]
    state = _state(runner)
    assert state["phases"]["calibrate"]["status"] == "stopped_for_review" and not runner.output("candidate-calibration-v1.json").exists()
    gates = state["calibration"]["gates"]
    assert "G4_level1" in gates["failures"] and gates["tolerance"]["G4_level1"]["max"] == 0.5 and gates["tolerance"]["G4_level1"]["at"]
    assert state["calibration"]["stopped"]["triggers"]["gate_failure"]["gates"] == ["G4_level1"]


def test_a_failed_environment_check_stops_the_calibration(world, base101, frozen101, sandbox, monkeypatch):
    root = sandbox(frozen101)
    runner, logs = make_runner(root, world, base101["config"])
    monkeypatch.setattr(ri, "state_drift", lambda now, recorded: 1.0)
    assert runner.calibrate() == 3
    state = _state(runner)
    assert "environment" in state["calibration"]["stopped"]["triggers"]["gate_failure"]["gates"]


def test_calibrate_refuses_a_key_of_a_confirmation_manifest(world, base101, frozen101, sandbox, monkeypatch):
    root = sandbox(frozen101)
    runner, _ = make_runner(root, world, base101["config"], model_loader=_refuse)
    original = runner_module.Runner._base

    def with_reserved(self):
        base = original(self)
        frame = next(frame for frame in base.inputs.pool.frames if frame.template_id == ri.TEMPLATE)
        return dataclasses.replace(base, manifests=base.manifests | {base.inputs.pool.reference_prompt(frame).key})

    monkeypatch.setattr(runner_module.Runner, "_base", with_reserved)
    with pytest.raises(ri.PhaseError, match="in a confirmation manifest"):
        runner.calibrate()


def test_lock_refuses_code_changed_since_calibration(world, base101, calibrated101, sandbox):
    root = sandbox(calibrated101["root"])
    runner, logs = make_runner(root, world, base101["config"], changed_paths=lambda commit: ["src/neural_decompiler/read_intervention.py"])
    with pytest.raises(ri.PhaseError, match="changed since the calibration"):
        runner.lock()
    runner, logs = make_runner(root, world, base101["config"])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ri, "own_blob", lambda: "0" * 40)
        with pytest.raises(ri.PhaseError, match="another analysis module"):
            runner.lock()
    assert _state(runner)["phases"]["lock"]["status"] == "not_started"


def test_lock_runs_no_forward(world, base101, calibrated101, sandbox):
    root = sandbox(calibrated101["root"])
    runner, logs = make_runner(root, world, base101["config"])
    with pytest.MonkeyPatch.context() as patch:
        spy = ForwardSpy(patch)
        assert runner.lock() == 0, logs[-3:]
    assert not spy.plain and not spy.patched


def test_lock_records_an_incident_when_the_exposed_table_does_not_reproduce(world, base101, calibrated101, sandbox, monkeypatch):
    root = sandbox(calibrated101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    original = ri.prediction_table
    monkeypatch.setattr(ri, "prediction_table", lambda *args, **kwargs: {**original(*args, **kwargs), "content_sha256": "0" * 64})
    assert runner.lock() == 2
    assert _state(runner)["phases"]["lock"]["incidents"] and not runner.output("candidate-lock.json").exists()


def test_a_confirm_incident_names_the_confirm_commit(world, base101, staged101, sandbox, monkeypatch):
    root = sandbox(staged101["root"])
    confirm_commit = "c" * 40
    runner, _ = make_runner(root, world, base101["config"], git_state=lambda: {"commit": confirm_commit, "dirty": False})
    original = rd.NounSet.contrasts
    calls = {"n": 0}

    def contrasts(self, logits):
        calls["n"] += 1
        if calls["n"] == 3:
            raise pm.IncidentError("non-finite contrast for cat")
        return original(self, logits)

    monkeypatch.setattr(rd.NounSet, "contrasts", contrasts)
    assert runner.confirm() == 2
    state = _state(runner)
    assert state["confirmation"]["incident"]["commit"] == confirm_commit != state["phases"]["calibrate"]["commit"]
    assert confirm_commit in ri.render_report(state)


def test_an_accounting_failure_is_an_incident(world, base101, staged101, sandbox, monkeypatch):
    root = sandbox(staged101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    original = ri.stage_runs

    def short(model, progs, schedules, geometry, config, *, executed, out, logits, log=None):
        original(model, progs, schedules, geometry, config, executed=executed, out=out, logits=logits, log=log)
        executed.pop()

    monkeypatch.setattr(ri, "stage_runs", short)
    assert runner.confirm() == 2
    confirmation = _state(runner)["confirmation"]
    assert "accounting failed" in confirmation["incident"]["message"] and confirmation["accounting"]["equal"] is False and "results" not in confirmation


def test_the_barrier_refuses_a_stage1_record_of_another_lock(world, base101, staged101, sandbox):
    from neural_decompiler import block0_completion as b0c

    root = sandbox(staged101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    state = _state(runner)
    state["stage1"]["lock_sha256"] = "0" * 64
    state["stage1"]["digest"] = b0c.stage_one_digest(state["stage1"])
    runner._write(state)
    assert runner.confirm() == 2
    assert "different lock" in _state(runner)["phases"]["confirm"]["incidents"][-1]["message"]


def test_the_calibration_and_the_secondary_records_carry_the_review_items(world, base101, calibrated101, confirmed101, sandbox):
    payload = json.loads((calibrated101["root"] / ri.CALIBRATION_RELATIVE_PATH).read_text())
    assert set(payload["min_s"]) == {"two", "alt"} and payload["module_blob"] == ri.own_blob() and payload["tolerances"] == ri.TOLERANCES
    assert len(payload["table_content_sha256"]) == 64
    root = sandbox(confirmed101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    head = _state(runner)["confirmation"]["secondary"]["statistics"]["head"]
    assert head["F_head_rem"]["criterion"] == "F_head_rem < 1.0" and "count" in head["F_head_ins_v"]
    per_arm = _state(runner)["confirmation"]["gates"]["per_arm"]
    assert set(per_arm["G5_output"]) == {"HEAD-NEC@two", "HEAD-SUF@two", "HEAD-NEC-V@two", "HEAD-SUF-V@two"}


def test_g5_compares_the_recomputation_with_the_plain_runs_capture(world, base101, staged101, sandbox, monkeypatch):
    """With each head arm's change amplified a thousandfold, the head arm's own capture is far from the unedited
    recomputation; G5 still passes because it compares with the plain run's capture (and its landing check compares
    the head arm's own capture with its replacement)."""
    root = sandbox(staged101["root"])
    runner, _ = make_runner(root, world, base101["config"])
    original = ri.head_delta
    monkeypatch.setattr(ri, "head_delta", lambda *args, **kwargs: 1000.0 * original(*args, **kwargs))
    assert runner.confirm() == 0
    state = _state(runner)
    assert state["confirmation"]["gates"]["failures"] == []
    saved = torch.load(runner.stage2_path)
    index = {tag: row for row, tag in enumerate(base101["config"].tags())}
    T = saved[f"{ri.frame_id_for(1)}/T_pt"].double()
    gap = max(float((T[index[f"HEAD-{kind}@two"]] - T[index["PL@two" if kind.startswith("NEC") else "REF"]]).abs().max()) for kind in ("NEC", "SUF", "NEC-V", "SUF-V"))
    assert gap > ri.TOLERANCES["G5_output"]  # comparing with the head arm's own capture would have failed G5
