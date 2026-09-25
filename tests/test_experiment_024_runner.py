"""Experiment 024's runner on the six-layer fake, inside the fake world of Experiment 023's runner test.

020 is explored and closed on the fake, 022 is frozen and calibrated, and 023 is extracted, frozen, calibrated and locked
on the fake by its own runner — so 023's exposed cells, confirmation and lock exist in 023's committed formats. The
024 constants that name 023's reviewed files (``INHERITED_023``) are pointed at that world's files. The world uses an
explicit test configuration (``FAKE``: 4 cues per class, so the exact E–N guard enumerates C(8, 4) = 70 assignments with
the bound 1; B = 40, P = 400) passed to the runner — never a patched production constant. On that world: every phase
and its refusals, the freeze's stops, the calibration's stops and incidents, the no-forward-pass calibration and lock,
I7, the accounting, the saved measurements, C recomputed bit for bit, the identity incidents, the single result write,
the descriptive records' failures, the report, and the leakage boundary (022's table and 023's outputs unreadable).
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

from neural_decompiler import block0_completion as b0c
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul
from test_experiment_020_runner import fake_world, make_fake_model  # noqa: F401 — fake_world is a fixture
from test_experiment_022_runner import COMMIT_A, COMMIT_B
from test_experiment_022_runner import calibrated, frozen, world  # noqa: F401 — fixtures
from test_experiment_023_runner import _apply as apply_023
from test_experiment_023_runner import base023, calibrated023, extracted, frozen023, locked023, toy_tokenizer_023  # noqa: F401 — fixtures

ROOT = Path(__file__).parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def toy_tokenizer_024(manifest, pool):
    """023's toy tokenizer plus every 024 candidate word (fresh ids below the fake's vocabulary)."""
    base = toy_tokenizer_023(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for word in rr.N_CANDIDATES] + [" " + word for pair in rr.MEASURE_LEMMAS + rr.ORDINARY_LEMMAS for word in pair]
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    assert next_id < 60000
    return type(base)(vocabulary)


@pytest.fixture(scope="module")
def base024(world, base023, locked023, tmp_path_factory):
    """023 locked on the fake; the 024 inherited-digest constants for that world's 023 files."""
    root = tmp_path_factory.mktemp("base024")
    shutil.copytree(locked023, root, dirs_exist_ok=True)
    inherited = {}
    for kind, relative in rr.INHERITED_023_PATHS.items():
        inherited[f"{kind}_file_sha256"] = rc.file_sha256(root / relative)
        if kind != "cells_data":
            inherited[f"{kind}_content_sha256"] = json.loads((root / relative).read_text())["content_sha256"]
    return {"root": root, "inherited": inherited}


def _apply(patch, world, base, base24) -> None:
    apply_023(patch, world, base)
    for key, value in base24["inherited"].items():
        patch.setitem(rr.INHERITED_023, key, value)


def make_runner(root: Path, world, config, **overrides):
    manifest, small, digests, lock_011, lock_012, lock_017 = world["fake"]
    logs: list[str] = []
    arguments = dict(root=root, results_path=root / "outputs/experiment-024/results.json", report_path=root / "outputs/experiment-024/report.md",
                     model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_024(manifest, small),
                     lock_011_loader=lambda path: dict(lock_011), lock_012_loader=lambda path: dict(lock_012), lock_017_loader=lambda path: dict(lock_017),
                     git_state=lambda: {"commit": COMMIT_A, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True,
                     changed_paths=lambda commit: [], log=logs.append, config=config)
    arguments.update(overrides)
    return runner_module.Runner(**arguments), logs


FAKE = rr.Configuration(
    name="fake-world", class_quota=4, draws=40, null_permutations=400, contrast_resamples=40, cross_check_draws=4,
    calibration_counts=(("determiner-like", 1), ("quantity", 1), ("adjective", 1)), n_pronoun=1, n_frames=108, n_nouns=79,
    expected_picks=(("N", ("eager", "fierce", "honest", "polite")), ("measure", ("gallon", "ounce", "acre", "pint")),
                    ("ordinary", ("apple", "horse", "doctor", "king"))),
    guard=rr.GuardSpec.derive(4, 4))


def _install(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)  # byte-identical, as the protocol installs a candidate


def _unreadable_outputs(patch) -> None:
    """022's calibration table and 023's local outputs can never be opened by a 024 phase."""
    original_load, original_read_bytes, original_read_text = torch.load, Path.read_bytes, Path.read_text

    def forbidden(path) -> bool:
        text = str(path)
        return text.endswith("calibration-table.pt") or "outputs/experiment-023/" in text

    def load(path, *args, **kwargs):
        if forbidden(path):
            raise AssertionError(f"a forbidden file was opened by a 024 phase: {path}")
        return original_load(path, *args, **kwargs)

    def read_bytes(self):
        if forbidden(self):
            raise AssertionError(f"a forbidden file was read by a 024 phase: {self}")
        return original_read_bytes(self)

    def read_text(self, *args, **kwargs):
        if forbidden(self):
            raise AssertionError(f"a forbidden file was read by a 024 phase: {self}")
        return original_read_text(self, *args, **kwargs)

    patch.setattr(torch, "load", load)
    patch.setattr(Path, "read_bytes", read_bytes)
    patch.setattr(Path, "read_text", read_text)


def _stage(world, base, base24, tmp_path_factory, name: str, source: Path, step) -> Path:
    root = tmp_path_factory.mktemp(name)
    shutil.copytree(source, root, dirs_exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        _apply(patch, world, base, base24)
        _unreadable_outputs(patch)
        runner, logs = make_runner(root, world, FAKE)
        step(runner, logs, root, patch)
    return root


@pytest.fixture(scope="module")
def frozen024(world, base023, base024, tmp_path_factory):
    def step(runner, logs, root, patch):
        assert runner.freeze() == 0, logs[-2:]

    return _stage(world, base023, base024, tmp_path_factory, "frozen024", base024["root"], step)


@pytest.fixture(scope="module")
def calibrated024(world, base023, base024, frozen024, tmp_path_factory):
    def step(runner, logs, root, patch):
        assert runner.calibrate() == 0, logs[-4:]
        _install(runner.candidate_calibration_path, root / rr.CALIBRATION_RELATIVE_PATH)

    return _stage(world, base023, base024, tmp_path_factory, "calibrated024", frozen024, step)


@pytest.fixture(scope="module")
def locked024(world, base023, base024, calibrated024, tmp_path_factory):
    def step(runner, logs, root, patch):
        assert runner.lock() == 0, logs[-4:]
        _install(runner.output("candidate-lock.json"), root / rr.LOCK_RELATIVE_PATH)
        _install(runner.output("candidate-preregistration.md"), root / rr.PREREGISTRATION_RELATIVE_PATH)

    return _stage(world, base023, base024, tmp_path_factory, "locked024", calibrated024, step)


@pytest.fixture(scope="module")
def confirmed024(world, base023, base024, locked024, tmp_path_factory):
    """One real confirmation on the fake, with every forward counted and every state write observed (what each write
    carried: the confirmation keys, the results' keys and the confirm phase's status)."""
    counts: Counter[str] = Counter()
    writes: list[dict] = []

    def step(runner, logs, root, patch):
        original = pm.capture_prompt
        original_write = runner_module.Runner._write

        def spy(model, prompt, sites):
            counts[prompt.key] += 1
            return original(model, prompt, sites)

        def recording(self, state):
            confirmation = state.get("confirmation") or {}
            writes.append({"confirmation": sorted(confirmation), "results": sorted(confirmation.get("results") or {}),
                           "status": state["phases"]["confirm"]["status"]})
            return original_write(self, state)

        patch.setattr(pm, "capture_prompt", spy)
        patch.setattr(runner_module.Runner, "_write", recording)
        assert runner.confirm() == 0, logs[-4:]

    root = _stage(world, base023, base024, tmp_path_factory, "confirmed024", locked024, step)
    return {"root": root, "counts": counts, "writes": writes}


@pytest.fixture
def sandbox(world, base023, base024, tmp_path, monkeypatch):
    """A per-test copy of a stage directory, with every patch the fake needs and the forbidden outputs unreadable."""
    _apply(monkeypatch, world, base023, base024)
    _unreadable_outputs(monkeypatch)

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
    return runner._confirmation(runner._base())[0]


def _refuse(*args, **kwargs):
    raise AssertionError("a model or tokenizer was loaded where none may be")


# ---------------------------------------------------------------------------


def test_parser_has_exactly_the_six_phases_no_option_and_production_only():
    parser = runner_module.build_parser()
    for phase in rr.PHASES:
        assert parser.parse_args([phase]).phase == phase
    for forbidden_args in (["extract"], ["calibrate", "--draws", "10"], ["confirm", "--config", "fake"], ["freeze", "--quota", "4"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden_args)
    assert runner_module.PHASES == ("validate", "freeze", "calibrate", "lock", "confirm", "report")
    assert runner_module.Runner().config is rr.PRODUCTION  # main() builds this runner; a test world passes FAKE explicitly
    assert FAKE.guard == rr.GuardSpec(4, 4, 70, 1) and rr.PRODUCTION.guard == rr.GuardSpec(8, 8, 12_870, 321)


def test_validate_loads_no_model_and_refuses_a_tampered_calibration_source_or_pin(world, base024, sandbox, monkeypatch):
    root = sandbox(base024["root"])
    runner, logs = make_runner(root, world, FAKE, model_loader=_refuse, tokenizer_loader=_refuse)
    assert runner.validate() == 0, logs
    assert "not frozen yet" in logs[-1] and "no calibration record installed" in logs[-1]
    data = root / b0c.CELLS_DATA_RELATIVE_PATH
    original = data.read_bytes()
    data.write_bytes(original[:8] + bytes([original[8] ^ 1]) + original[9:])
    assert runner.validate() == 1 and "cells data" in logs[-1]
    data.write_bytes(original)
    monkeypatch.setitem(rr.FROZEN_BLOBS, "block0_completion.py", "0" * 40)
    monkeypatch.setattr(ul, "load_frozen_inputs", lambda *args, **kwargs: pytest.fail("the frozen inputs were loaded before the pins were checked"))
    assert runner.validate() == 1 and "frozen modules" in logs[-1]


def test_freeze_is_tokenizer_only_and_its_stops_write_nothing(world, base024, sandbox):
    import dataclasses

    root = sandbox(base024["root"])
    target = root / rr.CONFIRMATION_RELATIVE_PATH
    deviating = dataclasses.replace(FAKE, expected_picks=(("N", ("honest", "polite", "rude", "sleepy")),) + FAKE.expected_picks[1:])
    runner, logs = make_runner(root, world, deviating, model_loader=_refuse)
    assert runner.freeze() == 3 and "differ from the design's expected picks" in logs[-1]
    assert not target.exists() and not runner.results_path.exists()  # a deviation consumes nothing: no file, no state
    words = tuple(f"w{k}" for k in range(40))
    too_many = dataclasses.replace(FAKE, class_quota=40, guard=rr.GuardSpec.derive(40, 40), expected_picks=(("N", words), ("measure", words), ("ordinary", words)))
    runner, logs = make_runner(root, world, too_many, model_loader=_refuse)
    assert runner.freeze() == 3 and "list N: 37 eligible of 40" in logs[-1] and not target.exists()
    runner, logs = make_runner(root, world, FAKE, model_loader=_refuse)
    assert runner.freeze() == 0, logs
    payload = json.loads(target.read_text())
    assert payload["picks"] == {"N": ["eager", "fierce", "honest", "polite"], "measure": ["gallon", "ounce", "acre", "pint"], "ordinary": ["apple", "horse", "doctor", "king"]}
    assert [entry["word"] for entry in payload["cues"] if entry["class"] == "B"] == ["gallons", "ounces", "acres", "pints"]
    assert payload["configuration"] == FAKE.to_json() and len(payload["manifest"]["S2-TARGET"]) == 20 * 108
    assert payload["exclusion"]["sources"][-1]["source"] == "confirmation-023"
    with pytest.raises(rr.PhaseError, match="runs once"):
        runner.freeze()
    assert runner.validate() == 0 and "confirmation" in logs[-1]


def test_calibrate_is_weights_only_reads_the_committed_cells_and_runs_once(world, base024, frozen024, sandbox, monkeypatch):
    root = sandbox(frozen024)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world, FAKE, tokenizer_loader=_refuse)
    assert runner.calibrate() == 0, logs[-4:]
    assert not spy.counts  # weights only, under the no-forward-pass guard
    state = _state(runner)
    record = json.loads(runner.candidate_calibration_path.read_text())
    rr.verify_calibration_record(record, FAKE)
    assert rc.file_sha256(runner.candidate_calibration_path) == state["calibration"]["record_sha256"] and state["executed_prompt_keys"] == []
    assert state["phases"]["calibrate"]["status"] == "complete" and state["configuration"] == FAKE.to_json() and state["inputs"]["023_cells_data_file"]
    assert record["exposed_cells"] == rr.calibration_source() and record["dependencies"]["calibration_source"] == rr.calibration_source()
    assert record["dependencies"]["module_blobs"] == rr.FROZEN_BLOBS and len(record["dependencies"]["model"]["parameters_sha256"]) == 64
    assert [entry["word"] for entry in record["calibration_cues"]] == ["a", "the", "four"] and [entry["word"] for entry in record["pronoun_cues"]] == ["three"]
    assert record["primary_floor"]["element"] == 0 and record["null"]["element"] == 389 and record["checks"]["spearman"]["passed"]
    assert record["confirmation_024"]["content_sha256"] == _confirmation(runner).content_sha256
    arrays = torch.load(runner.arrays_path)
    assert rr.arrays_digests(arrays) == record["arrays_sha256"]
    with pytest.raises(rr.PhaseError, match="once"):
        runner.calibrate()
    assert runner.report() == 0 and "Calibration (exposed only" in runner.report_path.read_text()


def test_calibrate_refuses_a_foreign_calibration_source_or_an_uncommitted_confirmation(world, base024, frozen024, sandbox):
    root = sandbox(frozen024)
    runner, _ = make_runner(root, world, FAKE, tracked=lambda path: path.name != "confirmation-v1.json" or "024" not in str(path))
    with pytest.raises(rr.PhaseError, match="frozen and committed first"):
        runner.calibrate()
    index = root / b0c.CELLS_INDEX_RELATIVE_PATH
    payload = json.loads(index.read_text())
    payload["cells_version"] = "resealed"
    payload["content_sha256"] = rc.content_digest(payload)
    index.write_text(pm.canonical_json(payload) + "\n")  # a valid-looking index that is not the reviewed one
    runner, _ = make_runner(root, world, FAKE)
    with pytest.raises(rr.PhaseError, match="cells index"):
        runner.calibrate()
    assert not runner.results_path.exists() and not runner.candidate_calibration_path.exists()


def test_the_calibration_stop_writes_no_record_and_is_never_retried(world, base024, frozen024, sandbox, monkeypatch):
    root = sandbox(frozen024)
    monkeypatch.setattr(rr, "order_statistic", lambda values, rank: 2.0)  # a reversed direction
    runner, logs = make_runner(root, world, FAKE)
    assert runner.calibrate() == 3 and "reversed direction" in logs[-1]
    state = _state(runner)
    assert state["phases"]["calibrate"]["status"] == "stopped_for_review" and not runner.candidate_calibration_path.exists()
    assert "reversed direction" in state["calibration"]["stop"]["reason"]
    with pytest.raises(rr.PhaseError, match="stopped_for_review"):
        runner.calibrate()
    with pytest.raises(rr.PhaseError, match="lock requires"):
        runner.lock()
    assert runner.report() == 0 and "Calibration stopped for review" in runner.report_path.read_text()


def test_a_calibration_cross_check_failure_is_an_incident_bound_to_its_commit(world, base023, base024, frozen024, sandbox, monkeypatch):
    root = sandbox(frozen024)
    monkeypatch.setattr(rr, "spearman_direct", lambda x, y: 2.0)
    runner, logs = make_runner(root, world, FAKE)
    assert runner.calibrate() == 2 and "spearman cross-check failed" in logs[-1]
    state = _state(runner)
    assert state["calibration"]["incidents"][-1]["commit"] == COMMIT_A and state["calibration"]["incidents"][-1]["recheck_020_022_023"]["ok"]
    assert not runner.candidate_calibration_path.exists()
    with pytest.raises(rr.PhaseError, match="incident is recorded at this commit"):
        runner.calibrate()
    monkeypatch.undo()
    _apply(monkeypatch, world, base023, base024)
    _unreadable_outputs(monkeypatch)
    runner, _ = make_runner(root, world, FAKE, git_state=lambda: {"commit": COMMIT_B, "dirty": False}, changed_paths=lambda commit: ["src/neural_decompiler/x.py"])
    with pytest.raises(rr.PhaseError, match="must change no scientific path"):
        runner.calibrate()
    runner, logs = make_runner(root, world, FAKE, git_state=lambda: {"commit": COMMIT_B, "dirty": False}, changed_paths=lambda commit: ["docs/incident-note.md"])
    assert runner.calibrate() == 0, logs[-3:]


def test_lock_is_weights_only_binds_the_scores_and_the_outcome_and_runs_once(world, base024, calibrated024, sandbox, monkeypatch):
    root = sandbox(calibrated024)
    runner, _ = make_runner(root, world, FAKE, changed_paths=lambda commit: ["experiments/023-block0-completion/exposed-cells.f64"])
    with pytest.raises(rr.PhaseError, match="scientific paths changed since calibrate"):
        runner.lock()
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world, FAKE, changed_paths=lambda commit: [rr.CALIBRATION_RELATIVE_PATH, f"{rr.EXPERIMENT_DIR}/README.md", "docs/x.md"])
    assert runner.lock() == 0, logs[-3:]
    assert not spy.counts
    lock = json.loads(runner.output("candidate-lock.json").read_text())
    record = json.loads((root / rr.CALIBRATION_RELATIVE_PATH).read_text())
    assert lock["content_sha256"] == rc.content_digest(lock) and lock["configuration"] == FAKE.to_json() and lock["guard"]["spec"]["max_upper"] == 1
    assert (lock["primary"]["F_rho"], lock["primary"]["null_975"]) == (record["primary_floor"]["F_rho"], record["null"]["null_975"]) and lock["line"] == record["line"]
    assert [cue["word"] for cue in lock["fresh"]["cues"] if cue["class"] == "E"] == ["apple", "horse", "doctor", "king"]
    assert [unit["word"] for unit in lock["guard"]["units"]["N"]] == ["eager", "fierce", "honest", "polite"] and lock["semantics"] == rr.SEMANTICS
    assert lock["dependencies"] == record["dependencies"] and lock["exposed_cells"] == rr.calibration_source()
    assert runner.output("candidate-preregistration.md").read_text() == rr.render_preregistration(lock)
    with pytest.raises(rr.PhaseError, match="lock already written"):
        runner.lock()


def test_lock_refuses_a_foreign_record_or_drifted_weights_before_writing(world, base024, calibrated024, sandbox, monkeypatch):
    root = sandbox(calibrated024)
    installed = root / rr.CALIBRATION_RELATIVE_PATH
    original = installed.read_bytes()
    runner, _ = make_runner(root, world, FAKE, model_loader=_refuse, tracked=lambda path: path != installed)
    with pytest.raises(rr.PhaseError, match="install the candidate calibration record"):
        runner.lock()
    record = json.loads(original)
    record["run_id"] = "another run"
    record["content_sha256"] = rc.content_digest(record)
    installed.write_text(pm.canonical_json(record) + "\n")
    runner, _ = make_runner(root, world, FAKE, model_loader=_refuse)
    with pytest.raises(rr.PhaseError, match="not the candidate this run wrote"):
        runner.lock()
    installed.write_bytes(original)
    monkeypatch.setattr(rr, "parameters_digest", lambda model: "0" * 64)  # the weights are not the calibrated ones
    runner, _ = make_runner(root, world, FAKE)
    with pytest.raises(rr.PhaseError, match="different scientific dependencies"):
        runner.lock()
    assert not runner.output("candidate-lock.json").exists() and _state(runner)["phases"]["lock"]["status"] == "not_started"


def test_a_failed_recheck_at_lock_is_an_incident_and_writes_no_lock(world, base024, calibrated024, sandbox, monkeypatch):
    root = sandbox(calibrated024)
    monkeypatch.setattr(runner_module.Runner, "_recheck", lambda self: {"ok": False, "message": "planted recheck failure"})
    runner, logs = make_runner(root, world, FAKE)
    assert runner.lock() == 2 and "planted recheck failure" in logs[-1]
    assert not runner.output("candidate-lock.json").exists() and _state(runner)["phases"]["lock"]["incidents"][-1]["commit"] == COMMIT_A
    with pytest.raises(rr.PhaseError, match="lock identity incident is recorded"):
        runner.lock()


def test_a_forbidden_key_in_the_ledger_is_refused(world, base024, calibrated024, sandbox):
    root = sandbox(calibrated024)
    runner, _ = make_runner(root, world, FAKE)
    state = _state(runner)
    confirmation_023 = json.loads((root / b0c.CONFIRMATION_RELATIVE_PATH).read_text())
    state["executed_prompt_keys"] = [confirmation_023["manifest"]["S2-TARGET"]["Y1"][0]]  # a spent 023 key
    rr.write_state_atomic(runner.results_path, state)
    with pytest.raises(rr.PhaseError, match="forbidden keys"):
        runner.lock()


# ---------------------------------------------------------------------------
# Confirm.


def test_confirm_runs_every_target_once_and_writes_the_result_in_one_write(world, base024, confirmed024, sandbox):
    root = sandbox(confirmed024["root"])
    runner, _ = make_runner(root, world, FAKE)
    confirmation = _confirmation(runner)
    assert Counter(confirmed024["counts"]) == Counter({prompt.key: 1 for prompt in confirmation.target_prompts})  # each manifest key once, nothing else
    state = _state(runner)
    assert set(state["executed_prompt_keys"]) == confirmation.manifest_keys() and len(confirmation.manifest_keys()) == 20 * 108
    confirm = state["phases"]["confirm"]
    assert confirm["status"] == "complete" and confirm["I7"]["bitwise_equal"] and confirm["confirm_commit"] == COMMIT_A
    results = state["confirmation"]["results"]
    assert state["confirmation"]["accounting"] == {"manifest": 2160, "executed": 2160, "ledger": 2160, "equal": True}
    assert state["confirmation"]["c_recompute"]["bitwise_equal"] and state["confirmation"]["c_recompute"]["pairs"] == 2160
    assert all(state["confirmation"]["gates"][name]["max"] <= rr.TOLERANCES[name] for name in ("I1", "I3", "I4"))
    assert results["outcome"]["label"] in rr.OUTCOMES and results["primary"]["result"] in rr.PRIMARY_RESULTS and results["guard"]["result"] in rr.GUARD_RESULTS
    assert results["outcome"]["label"] == rr.outcome(results["primary"]["result"], results["guard"]["result"])
    assert results["guard"]["assignments"] == 70 and results["guard"]["max_upper"] == 1 and len(results["per_cue"]) == 20
    # Requirement 6: the result appears in exactly one write, the one that also completes the phase, after the saved
    # measurements, the accounting, the C recomputation and the gates; the descriptive records come only after it.
    writes = confirmed024["writes"]
    first = next(index for index, write in enumerate(writes) if "results" in write["confirmation"])
    assert all(write["status"] != "complete" and "results" not in write["confirmation"] for write in writes[:first])
    assert writes[first]["status"] == "complete" and {"primary", "guard", "outcome", "per_cue", "checks"} <= set(writes[first]["results"])
    seen = [key for write in writes[:first] for key in write["confirmation"]]
    order = [seen.index(key) for key in ("stage2", "accounting", "c_recompute", "gates")]
    assert order == sorted(order) and "descriptives" not in writes[first]["confirmation"]
    assert all(write["status"] == "complete" and write["results"] == writes[first]["results"] for write in writes[first:])
    assert any("descriptives" in write["confirmation"] for write in writes[first + 1:])
    lock = json.loads((root / rr.LOCK_RELATIVE_PATH).read_text())
    saved = torch.load(runner.stage2_path)
    assert {key: rc.tensor_digest(value) for key, value in saved.items()} == state["confirmation"]["stage2"]["tensors_sha256"]
    cells = rr.fresh_cue_cells(rr.target_units(confirmation), saved, confirmation)
    assert json.loads(pm.canonical_json(rr.score(cells, confirmation, lock, FAKE))) == results  # the result reproduces from the saved measurements
    descriptives = state["confirmation"]["descriptives"]
    assert set(descriptives) == {"secondary", "contrasts", "ladder"} and descriptives["ladder"]["level1_identity"]["max"] is not None
    assert set(descriptives["contrasts"]["contrasts"]) == set(rr.CONTRASTS)
    with pytest.raises(rr.PhaseError, match="confirm already started"):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert f"**`{results['outcome']['label']}`**" in report and "exact one-sided permutation test" in report and "Not shown by any outcome" in report
    assert "prospective extrapolation test" in report and _state(runner)["phases"]["report"]["status"] == "complete"


def test_i7_refuses_a_drift_before_any_fresh_prompt(world, base024, locked024, sandbox, monkeypatch):
    root = sandbox(locked024)
    original = rr.fresh_quantities

    def drifted(*args, **kwargs):
        out = original(*args, **kwargs)
        out["cues"][0]["nounness"] += 1e-15  # a one-ulp-scale drift in a frozen score
        return out

    monkeypatch.setattr(rr, "fresh_quantities", drifted)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world, FAKE)
    assert runner.confirm() == 2 and "I7" in logs[-1]
    confirm = _state(runner)["phases"]["confirm"]
    assert not spy.counts and confirm["status"] == "not_started" and confirm["incidents"][-1]["commit"] == COMMIT_A
    monkeypatch.setattr(rr, "fresh_quantities", original)
    runner, _ = make_runner(root, world, FAKE, git_state=lambda: {"commit": COMMIT_B, "dirty": False})
    with pytest.raises(rr.PhaseError, match="I7 incident"):  # never retried, at any commit
        runner.confirm()
    assert not spy.counts and _state(runner)["executed_prompt_keys"] == []


def test_confirm_refuses_drifted_weights_before_any_prompt(world, base024, locked024, sandbox, monkeypatch):
    root = sandbox(locked024)
    monkeypatch.setattr(rr, "parameters_digest", lambda model: "0" * 64)
    spy = ExecutionSpy(monkeypatch)
    runner, _ = make_runner(root, world, FAKE)
    with pytest.raises(rr.PhaseError, match="different scientific dependencies"):
        runner.confirm()
    state = _state(runner)
    assert not spy.counts and state["phases"]["confirm"]["status"] == "not_started" and state["executed_prompt_keys"] == []


def test_confirm_refuses_tampered_untracked_or_changed_lock_files_before_the_model(world, base024, locked024, sandbox, monkeypatch):
    root = sandbox(locked024)
    spy = ExecutionSpy(monkeypatch)

    def resealed(text):
        lock = json.loads(text)
        lock["primary"]["null_975"] = -1.0
        lock["content_sha256"] = rc.content_digest(lock)
        return pm.canonical_json(lock) + "\n"

    cases = [(rr.PREREGISTRATION_RELATIVE_PATH, lambda text: text + "\nedited", "preregistration"),
             (rr.LOCK_RELATIVE_PATH, lambda text: text.replace('"schema_version":1', '"schema_version":2', 1), "not a verified Experiment 024 lock"),
             (rr.LOCK_RELATIVE_PATH, resealed, "not the candidate this run wrote")]
    for relative, edit, message in cases:
        path = root / relative
        original = path.read_text()
        path.write_text(edit(original))
        runner, _ = make_runner(root, world, FAKE, model_loader=_refuse)
        with pytest.raises(rr.PhaseError, match=message):
            runner.confirm()
        path.write_text(original)
    for overrides, message in (({"tracked": lambda path: path.name != "preregistration.md"}, "tracked and committed"),
                               ({"changed_paths": lambda commit: ["src/neural_decompiler/readout_routing.py"]}, "scientific paths changed since the lock"),
                               ({"changed_paths": lambda commit: None}, "not an ancestor"), ({"config": rr.PRODUCTION}, "configuration")):
        runner, _ = make_runner(root, world, overrides.pop("config", FAKE), model_loader=_refuse, **overrides)
        with pytest.raises(rr.PhaseError, match=message):
            runner.confirm()
    state = _state(runner)
    assert not spy.counts and state["phases"]["confirm"]["status"] == "not_started" and state["executed_prompt_keys"] == []


@pytest.mark.parametrize("plant", ["c_recompute", "gate", "accounting", "cross_check", "recheck"])
def test_a_post_measurement_incident_keeps_every_measurement_and_carries_no_result(world, base024, locked024, sandbox, monkeypatch, plant):
    """Requirement 6: the measurements are durably saved first; an identity or validity failure is an incident, and an
    incident never coexists with a result."""
    root = sandbox(locked024)
    if plant == "c_recompute":
        monkeypatch.setattr(rr, "recompute_c", lambda *args, **kwargs: {"max": 1e-9, "at": "planted", "pairs": 1, "bitwise_equal": False})
    elif plant == "gate":
        original_gates = rr.target_gates

        def failing(*args, **kwargs):
            out = original_gates(*args, **kwargs)
            out["I3"] = {"max": 1.0, "at": "planted"}
            return out

        monkeypatch.setattr(rr, "target_gates", failing)
    elif plant == "accounting":
        original_stage = ul.stage_two_022

        def dropping(*args, **kwargs):
            out = original_stage(*args, **kwargs)
            kwargs["executed"].pop()
            return out

        monkeypatch.setattr(ul, "stage_two_022", dropping)
    elif plant == "cross_check":
        monkeypatch.setattr(rr, "spearman_direct", lambda x, y: 2.0)
    else:
        monkeypatch.setattr(runner_module.Runner, "_recheck", lambda self: {"ok": False, "message": "planted recheck failure"})
    runner, logs = make_runner(root, world, FAKE)
    assert runner.confirm() == 2 and "INCIDENT" in logs[-1]
    state = _state(runner)
    confirmation_state = state["confirmation"]
    assert "incident" in confirmation_state and "results" not in confirmation_state and state["phases"]["confirm"]["status"] == "running"
    saved = torch.load(runner.stage2_path)  # every fresh measurement was saved before any gate
    assert {key: rc.tensor_digest(value) for key, value in saved.items()} == confirmation_state["stage2"]["tensors_sha256"]
    with pytest.raises(rr.PhaseError, match="confirm already started"):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "Confirmation incident" in report and "## The outcome" not in report


@pytest.mark.parametrize("error", [RuntimeError("a descriptive ladder failed"), KeyboardInterrupt()], ids=["failure", "interruption"])
def test_a_descriptive_failure_after_the_result_keeps_the_result_unchanged(world, base024, locked024, confirmed024, sandbox, monkeypatch, error):
    root = sandbox(locked024)

    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(rr, "ladder", broken)
    runner, logs = make_runner(root, world, FAKE)
    if isinstance(error, Exception):
        assert runner.confirm() == 0, logs[-3:]
    else:
        with pytest.raises(KeyboardInterrupt):
            runner.confirm()
    state = _state(runner)
    reference = rd.load_results_state(confirmed024["root"] / "outputs/experiment-024/results.json")["confirmation"]["results"]
    assert state["phases"]["confirm"]["status"] == "complete" and "incident" not in state["confirmation"]
    assert state["confirmation"]["results"] == reference  # the descriptive failure touched nothing of the result
    descriptives = state["confirmation"]["descriptives"]
    assert descriptives["failures"]["ladder"]["type"] == type(error).__name__ and "ladder" not in descriptives and "secondary" in descriptives
    assert runner.report() == 0 and "ladder was not computed" in runner.report_path.read_text()


@pytest.mark.parametrize("primary, guard, label", [("NOT_INTERPRETABLE", "PASS", "NOT_INTERPRETABLE"),
                                                   ("GUARD_FAILURE", "PASS", "NOUNNESS_PREDICTION_NOT_ESTABLISHED"),
                                                   ("PASS", "FAIL", "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED"),
                                                   ("PASS", "PASS", "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS")])
def test_every_outcome_is_written_with_its_reading_and_reported(world, base024, locked024, sandbox, monkeypatch, primary, guard, label):
    """The two decisions forced (their computations are tested in tier A); the hierarchy, the single write and the report
    are the runner's real ones."""
    root = sandbox(locked024)
    monkeypatch.setattr(rr, "classify_primary", lambda rho, floor, null: primary)
    monkeypatch.setattr(rr, "en_decision", lambda k, spec: guard)
    runner, logs = make_runner(root, world, FAKE)
    assert runner.confirm() == 0, logs[-3:]
    results = _state(runner)["confirmation"]["results"]
    assert (results["primary"]["result"], results["guard"]["result"], results["outcome"]["label"]) == (primary, guard, label)
    assert results["outcome"]["reading"] == rr.SEMANTICS["outcomes"][label] and results["outcome"]["simple"] == rr.SEMANTICS["simple"]
    assert runner.report() == 0 and f"**`{label}`**" in runner.report_path.read_text()


def test_the_state_is_written_atomically_in_the_state_format(tmp_path):
    path = tmp_path / "results.json"
    digest = rr.write_state_atomic(path, {"experiment": "024", "value": 1.5})
    assert rd.load_results_state(path)["state_sha256"] == digest and [p.name for p in tmp_path.iterdir()] == ["results.json"]
    with pytest.raises(Exception):
        rr.write_state_atomic(path, {"experiment": "024", "value": float("nan")})  # an unwritable state never replaces the old one
    assert rd.load_results_state(path)["value"] == 1.5 and [p.name for p in tmp_path.iterdir()] == ["results.json"]
