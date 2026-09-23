"""Experiment 022's runner on the six-layer fake, inside a fake-closed Experiment 020 world.

020's own runner explores the fake (eight exposed cues per frame, so ``a``, ``the``, ``three`` and ``four`` are measured in
every frame); a closure record and an evidence extract are written in the committed format; Experiment 021's exposed
table is produced by 021's own ``rematerialize`` on the fake and bound by a record in 021's committed format; and the
frozen digest constants are pointed at them. The quotas are one cue per class and one frame per template (4 cues,
3 frames), the four calibration strata hold one fake cue each, and ``B`` is 40. On that world: every phase, the
ledger and its leakage plants, the incidents, the undefined-draw stop, the no-forward-pass lock, I7, stage 1 with the
Y2 table, the barrier, the post-barrier mutation incident, stage 2, the scoring, the report, and the exploratory
replication on a fake Experiment 021 spent set.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_suppression as cs
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import upstream_localization as ul
from test_experiment_020_runner import _copy_repo_files, fake_world, make_fake_model  # noqa: F401 — fake_world is a fixture
from test_experiment_020_runner import runner_module as runner_020
from test_readout_calibration import write_closed_020
from test_readout_decompilation import toy_tokenizer_020

ROOT = Path(__file__).parents[1]
TOKEN_LIMIT = 8  # every frame then measures a, the, three and four besides its template's cues
FAKE_CUES = (("a", 247), ("the", 253), ("three", 1264), ("four", 1740))
COMMIT_A, COMMIT_B = "a" * 40, "b" * 40


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_022_runner", ROOT / "experiments/022-upstream-error-localization/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def _contract():
    return {"passed": True, "returncode": 0, "output_tail": "", "output_sha256": "x"}


def toy_tokenizer_022(manifest, pool):
    """020's toy tokenizer plus every 022 candidate cue and frame piece (fresh ids below the fake's vocabulary)."""
    base = toy_tokenizer_020(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in ul.CUE_CANDIDATES.values() for word in words]
    for texts in ul.FRAME_CANDIDATES.values():
        for text in texts:
            words += [piece if index == 0 else " " + piece for index, piece in enumerate(text.replace("{cue}", "").split())]
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    assert next_id < 60000
    return type(base)(vocabulary)


def _fake_pools(pool):
    """The real frame pools; one fake cue per class stratum, each measured by the fake's 020 ledger in all 108 frames."""
    real = rc.build_pools(pool)
    cues = {cls: (entry,) for cls, entry in zip(rc.CUE_CLASSES, FAKE_CUES)}
    return rc.CalibrationPools(cues, cues, {"confirmation-011": tuple(FAKE_CUES)}, real.frames_unscreened, real.frames_all_unscreened, real.nouns)


@pytest.fixture(scope="module")
def world(fake_world, tmp_path_factory):
    """020 explored and closed on the fake; 021's exposed table by 021's own function, bound by a committed-format record."""
    manifest, small, digests, lock_011, lock_012, lock_017 = fake_world
    root = tmp_path_factory.mktemp("world022")
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
    state_020 = rd.load_results_state(root / rc.RESULTS_020_RELATIVE_PATH)
    confirmation_020 = rd.load_confirmation(root / rd.CONFIRMATION_RELATIVE_PATH, small, digests)
    constants = write_closed_020(root, state=state_020, confirmation=confirmation_020)
    with pytest.MonkeyPatch.context() as patch:
        for name, value in constants.items():
            patch.setattr(rc, name, value)
        patch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
        patch.setattr(rd, "EXPLORE_TOKEN_LIMIT", TOKEN_LIMIT)
        closure = rc.verify_020_closure(root, confirmation_020)
        table, _ = rc.rematerialize(make_fake_model(), small, lock_011=lock_011, lock_012=lock_012, lock_017=lock_017, exploration_020=closure["exploration"],
                                    ledger_020=closure["ledger"], manifest_keys=frozenset(prompt.key for prompt in confirmation_020.all_prompts), confirmation=confirmation_020)
    (root / ul.EXPOSED_TABLE_021_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(table.tensors(), root / ul.EXPOSED_TABLE_021_RELATIVE_PATH)
    record_021 = {"experiment": "021", "kind": "a fake world's stand-in in the committed format", "table_sha256": table.digests()}
    record_021["content_sha256"] = rc.content_digest(record_021)
    (root / rc.CALIBRATION_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / rc.CALIBRATION_RELATIVE_PATH).write_text(pm.canonical_json(record_021) + "\n", encoding="utf-8")
    inherited = {"calibration_file_sha256": rc.file_sha256(root / rc.CALIBRATION_RELATIVE_PATH), "calibration_content_sha256": record_021["content_sha256"],
                 "exposed_measured_sha256": record_021["table_sha256"]["measured"]}
    lock_digests = {"lock_011": lock_011["content_sha256"], "lock_012": lock_012["content_sha256"], "lock_017": lock_017["content_sha256"]}
    return {"root": root, "constants": constants, "inherited": inherited, "lock_digests": lock_digests, "state_020": state_020, "fake": fake_world}


def _apply(patch, world) -> None:
    for name, value in world["constants"].items():
        patch.setattr(rc, name, value)
    patch.setattr(rc, "LOCK_DIGESTS", world["lock_digests"])
    patch.setattr(rc, "production_pools", _fake_pools)
    patch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    for key, value in world["inherited"].items():
        patch.setitem(ul.INHERITED_021, key, value)
    for name, value in (("B", 40), ("CROSS_CHECK_DRAWS", 2), ("CUE_QUOTA", 1), ("FRAME_QUOTA", 1)):
        patch.setattr(ul, name, value)


def make_runner(root: Path, world, **overrides):
    manifest, small, digests, lock_011, lock_012, lock_017 = world["fake"]
    logs: list[str] = []
    arguments = dict(root=root, results_path=root / "outputs/experiment-022/results.json", report_path=root / "outputs/experiment-022/report.md",
                     model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_022(manifest, small),
                     lock_011_loader=lambda path: dict(lock_011), lock_012_loader=lambda path: dict(lock_012), lock_017_loader=lambda path: dict(lock_017),
                     git_state=lambda: {"commit": COMMIT_A, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True,
                     changed_paths=lambda commit: [], contract_runner=_contract, log=logs.append)
    arguments.update(overrides)
    return runner_module.Runner(**arguments), logs


def _install(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)  # byte-identical, as the protocol installs a candidate


def _stage(world, tmp_path_factory, name: str, source: Path, step) -> Path:
    root = tmp_path_factory.mktemp(name)
    shutil.copytree(source, root, dirs_exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        _apply(patch, world)
        runner, logs = make_runner(root, world)
        step(runner, logs, root)
    return root


@pytest.fixture(scope="module")
def frozen(world, tmp_path_factory):
    def step(runner, logs, root):
        assert runner.freeze() == 0, logs

    return _stage(world, tmp_path_factory, "frozen022", world["root"], step)


@pytest.fixture(scope="module")
def calibrated(world, frozen, tmp_path_factory):
    def step(runner, logs, root):
        assert runner.calibrate() == 0, logs[-4:]
        _install(runner.candidate_calibration_path, root / ul.CALIBRATION_RELATIVE_PATH)

    return _stage(world, tmp_path_factory, "calibrated022", frozen, step)


@pytest.fixture(scope="module")
def locked(world, calibrated, tmp_path_factory):
    def step(runner, logs, root):
        assert runner.lock() == 0, logs[-4:]
        for name, relative in (("candidate-lock.json", ul.LOCK_RELATIVE_PATH), ("candidate-preregistration.md", ul.PREREGISTRATION_RELATIVE_PATH),
                               ("candidate-locked-y1-table.f64", ul.Y1_TABLE_RELATIVE_PATH), ("candidate-locked-y1-table.json", ul.Y1_TABLE_INDEX_RELATIVE_PATH)):
            _install(runner.output(name), root / relative)

    return _stage(world, tmp_path_factory, "locked022", calibrated, step)


@pytest.fixture(scope="module")
def confirmed(world, locked, tmp_path_factory):
    def step(runner, logs, root):
        assert runner.confirm() == 0, logs[-4:]

    return _stage(world, tmp_path_factory, "confirmed022", locked, step)


@pytest.fixture
def sandbox(world, tmp_path, monkeypatch):
    """A per-test copy of a stage directory, with every patch the fake needs."""
    _apply(monkeypatch, world)

    def copy(source: Path) -> Path:
        root = tmp_path / "root"
        shutil.copytree(source, root)
        return root

    return copy


class ExecutionSpy:
    """Counts actual forward executions by prompt key (multiplicity, not membership)."""

    def __init__(self, monkeypatch):
        self.counts: Counter[str] = Counter()
        original = pm.capture_prompt

        def spy(model, prompt, sites):
            self.counts[prompt.key] += 1
            return original(model, prompt, sites)

        monkeypatch.setattr(pm, "capture_prompt", spy)


def _state(runner):
    return rd.load_results_state(runner.results_path)


def _confirmation(runner):
    return runner._all()[1]


# ---------------------------------------------------------------------------


def test_parser_has_exactly_the_seven_phases_and_no_overrides():
    parser = runner_module.build_parser()
    for phase in ul.PHASES:
        assert parser.parse_args([phase]).phase == phase
    for forbidden in (["explore"], ["calibrate", "--draws", "10"], ["confirm", "--stage", "2"], ["lock", "--force"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden)
    assert runner_module.PHASES == ("validate", "freeze", "calibrate", "lock", "confirm", "report", "replicate-021")


def test_validate_loads_no_model_and_refuses_every_tampering(world, sandbox, monkeypatch):
    root = sandbox(world["root"])

    def refuse(*args, **kwargs):
        raise AssertionError("validate loaded a model")

    runner, logs = make_runner(root, world, model_loader=refuse, tokenizer_loader=refuse)
    assert runner.validate() == 0, logs
    assert "not frozen yet" in logs[-1]
    record_path = root / rc.CALIBRATION_RELATIVE_PATH
    original = record_path.read_text()
    record_path.write_text(original.replace("fake world", "fake  world"))
    assert runner.validate() == 1 and "021" in logs[-1]
    record_path.write_text(original)
    table = torch.load(root / ul.EXPOSED_TABLE_021_RELATIVE_PATH)
    table["measured"][0, 0] += 1.0
    torch.save(table, root / ul.EXPOSED_TABLE_021_RELATIVE_PATH)
    assert runner.validate() == 1 and "exposed" in logs[-1]
    monkeypatch.setitem(ul.FROZEN_BLOBS, "models.py", "0" * 40)
    assert runner.validate() == 1 and "frozen modules" in logs[-1]


def test_freeze_is_tokenizer_only_writes_once_and_a_shortfall_writes_nothing(world, sandbox, monkeypatch):
    root = sandbox(world["root"])

    def refuse(*args, **kwargs):
        raise AssertionError("freeze loaded a model")

    runner, logs = make_runner(root, world, model_loader=refuse)
    monkeypatch.setattr(ul, "CUE_QUOTA", 99)
    assert runner.freeze() == 3 and "shortfall" in logs[-1]
    assert not (root / ul.CONFIRMATION_RELATIVE_PATH).exists()
    monkeypatch.setattr(ul, "CUE_QUOTA", 1)
    assert runner.freeze() == 0, logs
    payload = json.loads((root / ul.CONFIRMATION_RELATIVE_PATH).read_text())
    assert payload["counts"] == {"classes": {cls: 1 for cls in ul.CUE_CLASSES}, "templates": {template: 1 for template in ul.TEMPLATES}}
    assert [entry["class"] for entry in payload["cues"]] == list(ul.CUE_CLASSES) and all(entry["word"] in ul.CUE_CANDIDATES[entry["class"]] for entry in payload["cues"])
    assert [entry["frame_id"] for entry in payload["frames"]] == ["cardinal-022-1", "quantifier-022-1", "coordinated-adjective-022-1"]
    with pytest.raises(ul.PhaseError, match="once"):
        runner.freeze()
    assert runner.validate() == 0 and "confirmation" in logs[-1]


def test_calibrate_executes_exactly_the_calibration_keys_and_writes_the_record(world, frozen, sandbox, monkeypatch):
    root = sandbox(frozen)
    before = {path: rc.file_sha256(root / path) for path in (rc.CLOSURE_020_RELATIVE_PATH, rc.EXTRACT_020_RELATIVE_PATH, rc.RESULTS_020_RELATIVE_PATH, rc.CALIBRATION_RELATIVE_PATH)}
    assert not (root / ul.RESULTS_021_RELATIVE_PATH).exists() and not (root / ul.STAGE2_021_RELATIVE_PATH).exists()  # 021's spent set is not needed
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world)
    assert runner.calibrate() == 0, logs[-4:]
    state = _state(runner)
    frames = world["fake"][1].frames
    expected = {pm.Prompt(frame, token_id, word).key for word, token_id in FAKE_CUES for frame in frames}
    assert set(spy.counts) == expected == set(state["executed_prompt_keys"]) and set(spy.counts.values()) == {1}
    assert expected <= set(world["state_020"]["executed_prompt_keys"])
    confirmation = _confirmation(runner)
    assert not (set(spy.counts) & confirmation.manifest_keys())
    assert {path: rc.file_sha256(root / path) for path in before} == before  # 020's closure and 021's record untouched
    calibration = state["calibration"]
    assert state["phases"]["calibrate"]["status"] == "complete" and calibration["r1"]["passed"] and calibration["r1"]["max_difference"] <= ul.TOLERANCES["R1"]
    assert all(calibration["gates"][name]["max"] <= ul.TOLERANCES[name] for name in ("I1", "I2", "I3", "I4", "I5"))
    record = json.loads(runner.candidate_calibration_path.read_text())
    assert rc.file_sha256(runner.candidate_calibration_path) == calibration["record_sha256"]
    ul.verify_calibration_record(record)
    assert record["constants"]["B"] == 40 and record["inputs"] == state["inputs"] and set(record["inputs"]) == set(ul.DIGEST_KEYS)
    assert record["cross_check"]["n_exceeding"] == 0 and record["cross_check"]["n_checked"] == 2 * 4 * 44
    assert record["pools"]["slots"] == {**{f"cue/{cls}": 1 for cls in ul.CUE_CLASSES}, **{f"frame/{t}": 1 for t in ul.TEMPLATES}}
    saved = torch.load(runner.table_path)
    assert {name: rc.tensor_digest(saved[name]) for name in ul.CalibrationTable.TENSORS} == {name: calibration["table_sha256"][name] for name in ul.CalibrationTable.TENSORS}
    arrays = torch.load(runner.draws_path)
    assert {key: rc.tensor_digest(value) for key, value in arrays.items()} == record["draw_arrays_sha256"]
    with pytest.raises(ul.PhaseError, match="once"):
        runner.calibrate()
    assert runner.report() == 0 and "Calibration (exposed only" in runner.report_path.read_text()


def test_calibrate_incidents_are_recorded_and_bind_their_commit(world, frozen, sandbox, monkeypatch):
    root = sandbox(frozen)
    monkeypatch.setitem(ul.TOLERANCES, "I1", 0.0)
    runner, logs = make_runner(root, world)
    assert runner.calibrate() == 2 and "I1" in logs[-1]
    state = _state(runner)
    assert state["phases"]["calibrate"]["status"] == "running" and state["calibration"]["incidents"][-1]["commit"] == COMMIT_A
    assert state["calibration"]["gates"]["I1"]["max"] > 0.0 and not runner.candidate_calibration_path.exists()  # the maxima reached disk first
    assert state["calibration"]["incidents"][-1]["recheck_020_021"]["ok"]
    with pytest.raises(ul.PhaseError, match="incident is recorded at this commit"):
        runner.calibrate()
    monkeypatch.setitem(ul.TOLERANCES, "I1", 1e-4)
    monkeypatch.setitem(ul.TOLERANCES, "R1", -1.0)  # R1 can never pass: an incident after the gates
    runner, logs = make_runner(root, world, git_state=lambda: {"commit": COMMIT_B, "dirty": False})
    assert runner.calibrate() == 2 and "R1" in logs[-1]
    assert _state(runner)["calibration"]["r1"]["passed"] is False


def test_a_planted_manifest_cue_is_refused_before_its_frame_runs(world, frozen, sandbox, monkeypatch):
    root = sandbox(frozen)
    runner, _ = make_runner(root, world)
    fresh = _confirmation(runner).tokens[0]

    def planted(pool):
        pools = _fake_pools(pool)
        cues = dict(pools.cues)
        cues["quantity"] = cues["quantity"] + ((fresh["word"], int(fresh["token_id"])),)
        return rc.CalibrationPools(cues, cues, pools.cohorts, pools.frames_unscreened, pools.frames_all_unscreened, pools.nouns)

    monkeypatch.setattr(rc, "production_pools", planted)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world)
    assert runner.calibrate() == 2 and "manifest keys" in logs[-1]
    assert not spy.counts and not _state(runner)["executed_prompt_keys"]  # refused before any of the first frame's prompts ran or was recorded


def test_calibrate_records_each_frames_keys_before_they_run(world, frozen, sandbox, monkeypatch):
    root = sandbox(frozen)
    runner, logs = make_runner(root, world)
    original = ul.measure_prompt
    seen: list[bool] = []

    def checking(model, progs, frame, state, token_id, word):
        seen.append(pm.Prompt(frame, token_id, word).key in set(rd.load_results_state(runner.results_path)["executed_prompt_keys"]))
        if len(seen) == 3:
            raise KeyboardInterrupt  # a hard stop mid-frame: the frame's keys are already on disk
        return original(model, progs, frame, state, token_id, word)

    monkeypatch.setattr(ul, "measure_prompt", checking)
    with pytest.raises(KeyboardInterrupt):
        runner.calibrate()
    state = _state(runner)
    first = sorted(world["fake"][1].frames, key=lambda frame: frame.frame_id)[0]
    assert seen == [True, True, True] and {pm.Prompt(first, token_id, word).key for word, token_id in FAKE_CUES} <= set(state["executed_prompt_keys"])
    assert state["calibration"]["incidents"][-1]["type"] == "KeyboardInterrupt"


def test_the_undefined_draw_stop_writes_no_floor_and_is_never_retried(world, frozen, sandbox, monkeypatch):
    root = sandbox(frozen)
    monkeypatch.setattr(ul, "GAP_MIN", 1e9)  # no draw is interpretable: every statistic undefined
    runner, logs = make_runner(root, world)
    assert runner.calibrate() == 3 and "stop" in logs[-1]
    state = _state(runner)
    assert state["phases"]["calibrate"]["status"] == "stopped_for_review" and not runner.candidate_calibration_path.exists()
    assert state["calibration"]["stop"]["offending"] == {f"{p}/{c}": 40 for p in ul.POPULATIONS for c in ul.CLAIMS}
    with pytest.raises(ul.PhaseError, match="stopped_for_review"):
        runner.calibrate()
    with pytest.raises(ul.PhaseError, match="lock requires"):
        runner.lock()
    assert runner.report() == 0 and "stopped for review" in runner.report_path.read_text()


def test_lock_runs_no_forward_pass_and_binds_the_y1_companion_and_the_y2_specification(world, calibrated, sandbox, monkeypatch):
    root = sandbox(calibrated)
    runner, logs = make_runner(root, world, changed_paths=lambda commit: ["src/neural_decompiler/upstream_localization.py"])
    with pytest.raises(ul.PhaseError, match="scientific paths changed"):
        runner.lock()
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world, changed_paths=lambda commit: [ul.CALIBRATION_RELATIVE_PATH, f"{ul.EXPERIMENT_DIR}/README.md", "docs/x.md"])
    assert runner.lock() == 0, logs[-3:]
    assert not spy.counts
    lock = json.loads(runner.output("candidate-lock.json").read_text())
    assert lock["content_sha256"] == rc.content_digest(lock) and set(lock["conditions"]) == {f"{p}/{c}" for p in ul.POPULATIONS for c in ul.CLAIMS}
    assert lock["y1_table"]["layout"] == [{"name": "cue_final", "shape": [4 * 72, 16, 79]}, {"name": "coordinated", "shape": [4 * 36, 32, 79]}]
    assert lock["y2_table"]["layout"] == [{"name": "cue_final", "shape": [4 * 2, 16, 79]}, {"name": "coordinated", "shape": [4 * 1, 32, 79]}]
    assert lock["y1_table"]["gates"]["I5"]["max"] == 0.0 and lock["y1_table"]["gates"]["I2"]["max"] <= 1e-12
    index = json.loads(runner.output("candidate-locked-y1-table.json").read_text())
    assert index["file_sha256"] == rc.file_sha256(runner.output("candidate-locked-y1-table.f64")) == lock["y1_table"]["file_sha256"]
    assert index["pairs"]["cue_final"][0][1] == min(int(t["token_id"]) for t in _confirmation(runner).tokens)  # token id first, then frame_id
    assert runner.output("candidate-preregistration.md").read_text() == ul.render_preregistration(lock)
    with pytest.raises(ul.PhaseError, match="lock already written"):
        runner.lock()


def test_lock_refuses_an_uninstalled_or_altered_calibration_record(world, frozen, calibrated, sandbox):
    root = sandbox(calibrated)
    path = root / ul.CALIBRATION_RELATIVE_PATH
    text = path.read_text()
    path.write_text(text[:-1] + " \n")
    runner, _ = make_runner(root, world)
    with pytest.raises(ul.PhaseError, match="not the candidate"):
        runner.lock()
    path.unlink()
    with pytest.raises(ul.PhaseError, match="install the candidate"):
        runner.lock()


def test_confirm_runs_stage1_the_barrier_and_stage2_once_each_and_scores_eight_conditions(world, locked, sandbox, monkeypatch):
    root = sandbox(locked)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world)
    assert runner.confirm() == 0, logs[-4:]
    state = _state(runner)
    confirmation = _confirmation(runner)
    assert Counter(spy.counts) == Counter({prompt.key: 1 for prompt in confirmation.all_prompts})  # each manifest key exactly once, nothing else
    calibration_keys = {pm.Prompt(frame, token_id, word).key for word, token_id in FAKE_CUES for frame in world["fake"][1].frames}
    assert set(state["executed_prompt_keys"]) == calibration_keys | confirmation.manifest_keys()
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["I7"]["bitwise_equal"]
    results = state["confirmation"]
    assert set(results["conditions"]) == {f"{p}/{c}" for p in ul.POPULATIONS for c in ul.CLAIMS} and results["aggregate_label"] is None
    assert all(entry["result"] in ul.RESULTS for entry in results["conditions"].values())
    stage1 = results["stage1"]
    y2 = stage1["y2_table"]
    assert rc.file_sha256(root / y2["data_path"]) == y2["file_sha256"] and rc.file_sha256(root / y2["index_path"]) == y2["index_sha256"]
    assert stage1["digest"] == ul.stage_one_digest(stage1) and all(frame["selects"].startswith("nothing") for frame in stage1["frames"].values())
    assert all(results["gates"][name]["max"] <= ul.TOLERANCES[name] for name in ("I1", "I2", "I3", "I4"))
    assert results["cross_check"]["n_exceeding"] == 0 and max(results["efficiency_I6"].values()) <= 1e-12
    tensors = torch.load(runner.stage2_path)
    assert {key: rc.tensor_digest(value) for key, value in tensors.items()} == results["stage2"]["tensors_sha256"]
    with pytest.raises(ul.PhaseError, match="confirm already started"):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "The eight conditions" in report and "CDF percentile" in report and "unfavorable upper tail" in report and "no aggregate label" in report
    assert "Layers 1–2:" in report and "What a pass does not show" in report and all(f"{key}: **" in report for key in results["conditions"])
    assert set(results["descriptives"]["historical_comparator"]) == {"Y1/cue_final", "Y1/coordinated", "Y2/cue_final", "Y2/coordinated"}
    assert _state(runner)["phases"]["report"]["status"] == "complete"


def test_i7_refuses_a_y1_table_that_does_not_reproduce_before_any_fresh_prompt(world, locked, sandbox, monkeypatch):
    root = sandbox(locked)
    original = ul.composition_tables

    def drifted(*args, **kwargs):
        out = original(*args, **kwargs)
        out["blocks"][0][1][0, 0, 0] += 1e-12  # a one-ulp-scale drift in the reconstruction
        return out

    monkeypatch.setattr(ul, "composition_tables", drifted)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world)
    assert runner.confirm() == 2 and "I7" in logs[-1]
    confirm = _state(runner)["phases"]["confirm"]
    assert not spy.counts and confirm["status"] == "not_started" and confirm["incidents"][-1]["commit"] == COMMIT_A and "I7" in confirm["incidents"][-1]["message"]
    monkeypatch.setattr(ul, "composition_tables", original)
    runner, logs = make_runner(root, world, git_state=lambda: {"commit": COMMIT_B, "dirty": False})
    with pytest.raises(ul.PhaseError, match="I7 incident"):  # never retried until it passes, at any commit
        runner.confirm()
    assert not spy.counts


def test_a_lock_identity_failure_is_an_incident_that_blocks_lock(world, calibrated, sandbox, monkeypatch):
    root = sandbox(calibrated)
    original = ul.composition_tables

    def inexact(*args, **kwargs):
        out = original(*args, **kwargs)
        out["gates"]["I5"] = {"max": 5e-324, "at": "planted"}  # the empty coalition no longer equals the committed Level 0 exactly
        return out

    monkeypatch.setattr(ul, "composition_tables", inexact)
    runner, logs = make_runner(root, world)
    assert runner.lock() == 2 and "I5" in logs[-1]
    lock_phase = _state(runner)["phases"]["lock"]
    assert lock_phase["status"] == "not_started" and lock_phase["incidents"][-1]["commit"] == COMMIT_A and not runner.output("candidate-lock.json").exists()
    monkeypatch.setattr(ul, "composition_tables", original)
    runner, logs = make_runner(root, world, git_state=lambda: {"commit": COMMIT_B, "dirty": False})
    with pytest.raises(ul.PhaseError, match="lock identity incident"):
        runner.lock()


def test_the_lock_companions_are_bound(world, locked, sandbox):
    root = sandbox(locked)
    runner, _ = make_runner(root, world)
    data = bytearray((root / ul.Y1_TABLE_RELATIVE_PATH).read_bytes())
    data[100] ^= 0x01
    (root / ul.Y1_TABLE_RELATIVE_PATH).write_bytes(bytes(data))
    with pytest.raises(ul.PhaseError, match="Y1 table companion"):
        runner.confirm()
    shutil.copyfile(locked / ul.Y1_TABLE_RELATIVE_PATH, root / ul.Y1_TABLE_RELATIVE_PATH)
    lock = json.loads((root / ul.LOCK_RELATIVE_PATH).read_text())
    lock["conditions"]["Y1/C1"]["envelope"]["bound"] = 0.0
    lock["content_sha256"] = rc.content_digest(lock)
    (root / ul.LOCK_RELATIVE_PATH).write_text(pm.canonical_json(lock) + "\n")
    with pytest.raises(ul.PhaseError, match="not the candidate"):
        runner.confirm()
    shutil.copyfile(locked / ul.LOCK_RELATIVE_PATH, root / ul.LOCK_RELATIVE_PATH)
    (root / ul.PREREGISTRATION_RELATIVE_PATH).write_text("edited\n")
    with pytest.raises(ul.PhaseError, match="preregistration"):
        runner.confirm()


def test_a_y2_table_changed_before_the_barrier_is_an_incident_and_the_artifact_is_preserved(world, locked, sandbox, monkeypatch):
    root = sandbox(locked)
    original = ul.stage_one_022

    def tampering(*args, **kwargs):
        record = original(*args, **kwargs)
        path = kwargs["root"] / record["y2_table"]["data_path"]
        data = bytearray(path.read_bytes())
        data[8] ^= 0x01
        path.write_bytes(bytes(data))
        return record

    monkeypatch.setattr(ul, "stage_one_022", tampering)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world)
    assert runner.confirm() == 2 and "do not match their index digest" in logs[-1]
    confirmation = _confirmation(runner)
    assert not (set(spy.counts) & {prompt.key for prompt in confirmation.target_prompts})  # no S2-TARGET prompt ran
    state = _state(runner)
    assert state["phases"]["confirm"]["status"] == "running" and state["confirmation"]["incident"]["type"] == "IncidentError"
    assert (root / ul.Y2_TABLE_OUTPUT).exists() and (root / ul.Y2_TABLE_INDEX_OUTPUT).exists()  # preserved, never restored
    with pytest.raises(ul.PhaseError, match="confirm already started"):
        runner.confirm()


def test_a_consistent_rewrite_of_the_stage1_record_is_caught_at_the_barrier(world, locked, sandbox, monkeypatch):
    root = sandbox(locked)
    original = rd.load_results_state

    def rewritten(path):
        state = original(path)
        stage1 = (state.get("confirmation") or {}).get("stage1")
        if stage1 is not None and "stage2" not in state["confirmation"]:
            frame_id = sorted(stage1["frames"])[0]
            stage1["frames"][frame_id]["validity"]["valid"] = not stage1["frames"][frame_id]["validity"]["valid"]
            stage1["digest"] = ul.stage_one_digest(stage1)  # self-consistent: only the in-memory digest can tell
        return state

    monkeypatch.setattr(rd, "load_results_state", rewritten)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world)
    assert runner.confirm() == 2 and "not the one stage 1 wrote" in logs[-1]
    assert not (set(spy.counts) & {prompt.key for prompt in _confirmation(runner).target_prompts})


def test_a_y2_table_changed_after_the_barrier_is_an_incident(world, locked, sandbox, monkeypatch):
    root = sandbox(locked)
    original = ul.stage_two_022

    def mutating(*args, **kwargs):
        measured = original(*args, **kwargs)
        path = root / ul.Y2_TABLE_OUTPUT
        data = bytearray(path.read_bytes())
        data[-1] ^= 0x01
        path.write_bytes(bytes(data))
        return measured

    monkeypatch.setattr(ul, "stage_two_022", mutating)
    runner, logs = make_runner(root, world)
    assert runner.confirm() == 2 and "index digest" in logs[-1]
    state = _state(runner)
    assert "stage2" in state["confirmation"] and runner.stage2_path.exists()  # the measurements reached disk before the check
    assert "conditions" not in state["confirmation"]


def test_not_interpretable_is_reachable_without_removing_any_unit(world, locked, sandbox, monkeypatch):
    root = sandbox(locked)
    original = ul.score_022

    def raised_gap(*args, **kwargs):
        monkeypatch.setattr(ul, "GAP_MIN", 1e9)
        return original(*args, **kwargs)

    monkeypatch.setattr(ul, "score_022", raised_gap)
    runner, logs = make_runner(root, world)
    assert runner.confirm() == 0, logs[-3:]
    conditions = _state(runner)["confirmation"]["conditions"]
    assert {entry["result"] for entry in conditions.values()} == {"NOT_INTERPRETABLE"} and all(entry["value"] is None for entry in conditions.values())


def _fake_021_spent_set(root: Path, world) -> dict[str, str]:
    """A small stand-in for Experiment 021's closed confirmation, in its stored formats: two of its cues, three exposed
    frames (Y1) and two of its fresh frames (Y2), measured on the fake with its own noun set."""
    manifest, small, digests, lock_011, lock_012, lock_017 = world["fake"]
    model = make_fake_model()
    inputs = ul.load_frozen_inputs(root, lock_011_loader=lambda path: dict(lock_011), lock_012_loader=lambda path: dict(lock_012), lock_017_loader=lambda path: dict(lock_017))
    progs = ul.ModelPrograms.from_model(model, inputs)
    confirmation = inputs.confirmation_020
    nouns = rd.NounSet.build(progs.weights, small.nouns, confirmation.nouns)
    import dataclasses

    progs_021 = dataclasses.replace(progs, nouns=nouns, scorable=list(nouns.exposed_scorable))
    keys = [nouns.nouns[index].lexical_key for index in nouns.exposed_scorable]
    first = lambda frames, template: next(frame for frame in frames if frame.template_id == template)  # noqa: E731
    y1_frames = [first(small.frames, template) for template in ul.TEMPLATES]
    y2_frames = [first(confirmation.frames, "cardinal"), first(confirmation.frames, ul.COORDINATED)]
    stage1_states = {frame.frame_id: rd.locked_state(rd.capture_frame_020(model, progs.head, confirmation.reference_prompt(frame), nouns, progs.axis_T))
                     for frame in y2_frames}
    stored, digests_021 = {}, {}
    for population, frames, locked_states in (("Y1", y1_frames, inputs.closure["exploration"]["locked_states"]), ("Y2", y2_frames, stage1_states)):
        cues, frame_ids, measured, predicted, ceiling = [], [], [], [], []
        for frame in frames:
            state = rd.state_from_locked(locked_states[frame.frame_id], frame)
            rows16 = ul.atp.reference_rows(progs.programs, state.state_017.x1_all, state.state_017.x2_all)
            for token in confirmation.tokens[:2]:
                c_ref = nouns.contrast_from_residual(progs.readout, state.h6)
                measurement = rd.measure_pair(model, state, nouns, token["word"], int(token["token_id"]), c_ref=c_ref)
                cues.append(token["word"]); frame_ids.append(frame.frame_id)
                measured.append(measurement.dc[list(nouns.exposed_scorable)])
                level0 = rd.predicted_dx3(progs.chain, progs.weights, state, rows16, int(token["token_id"]), frame.template_id)
                predicted.append(ul.contrast_of(progs_021, state, level0))
                ceiling.append(ul.contrast_of(progs_021, state, measurement.dx3))
        common = {"cues": cues, "frames": frame_ids, "templates": [f.template_id for f in frames for _ in range(2)], "noun_keys": keys}
        stored[population] = {"exposed": {**common, "measured": torch.stack(measured), "predicted": torch.stack(predicted)},
                              "ceiling": {**common, "measured": torch.stack(measured), "predicted": torch.stack(ceiling)}}
        digests_021[population] = {part: {key: rc.tensor_digest(stored[population][part][key]) for key in ("measured", "predicted")} for part in ("exposed", "ceiling")}
    (root / ul.STAGE2_021_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stored, root / ul.STAGE2_021_RELATIVE_PATH)
    state = {"experiment": "021", "phases": {"confirm": {"status": "complete"}}, "confirmation": {"stage1": {"states": stage1_states}, "stage2": {"tables_sha256": digests_021}}}
    state_sha = rd.write_results_state(root / ul.RESULTS_021_RELATIVE_PATH, state)
    extract = {"experiment": "021", "kind": "fake extract"}
    extract["content_sha256"] = rc.content_digest(extract)
    (root / ul.EXTRACT_021_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / ul.EXTRACT_021_RELATIVE_PATH).write_text(pm.canonical_json(extract) + "\n")
    return {"results_file_sha256": rc.file_sha256(root / ul.RESULTS_021_RELATIVE_PATH), "results_state_sha256": state_sha, "extract_content_sha256": extract["content_sha256"]}


def test_replicate_021_runs_only_after_the_report_and_is_exploratory(world, confirmed, sandbox, monkeypatch):
    root = sandbox(confirmed)
    for key, value in _fake_021_spent_set(root, world).items():
        monkeypatch.setitem(ul.INHERITED_021, key, value)
    runner, logs = make_runner(root, world)
    with pytest.raises(ul.PhaseError, match="after a completed confirm and its report"):
        runner.replicate_021()
    assert runner.report() == 0
    before = _state(runner)["confirmation"]
    spy = ExecutionSpy(monkeypatch)
    assert runner.replicate_021() == 0, logs[-3:]
    assert not spy.counts
    state = _state(runner)
    replication = state["replication"]
    assert replication["kind"].startswith("EXPLORATORY") and all(replication["checks"][name].startswith("unavailable") for name in ("I1", "I2", "I3"))
    assert replication["checks"]["I5"]["max"] == 0.0 and replication["checks"]["I4"]["passed"] and replication["checks"]["I6"]["passed"]
    assert "result" not in json.dumps(replication["populations"]) and state["confirmation"] == before  # it changes no 022 result
    assert all(replication["populations"][p]["n_pairs"] == ({"cue_final": 4, "coordinated": 2} if p == "Y1" else {"cue_final": 2, "coordinated": 2}) for p in ul.POPULATIONS)
    assert "EXPLORATORY" in runner.replication_path.with_suffix(".md").read_text()
    with pytest.raises(ul.PhaseError, match="already ran"):
        runner.replicate_021()
