"""Experiment 023's runner on the six-layer fake, inside the fake world of Experiment 022's runner test.

020 is explored and closed on the fake, 021's exposed table is bound by a committed-format record, and Experiment 022
is frozen and calibrated on the fake by its own runner — so 022's calibration table exists in 022's own format and its
record is installed. The 023 constants are pointed at that world's 022 files. The quotas are one cue per stratum and
one frame per template (3 cues, 3 frames) and ``B`` is 40. Right after ``extract`` the fake 022 table is deleted from
every later stage, so ``freeze``, ``calibrate``, ``lock``, ``confirm`` and ``report`` are shown to run without it; a
guard also makes any attempt to load it fail. On that world: every phase, every refusal, the extraction's stop and
incident, the ledger isolation, the no-forward-pass extraction and lock, I7, stage 1 with the write-once Y2 table, the
barrier and its tampering incident, stage 2, the canonical scoring and the report.
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
from neural_decompiler import upstream_localization as ul
from test_experiment_020_runner import fake_world, make_fake_model  # noqa: F401 — fake_world is a fixture
from test_experiment_022_runner import COMMIT_A, COMMIT_B, toy_tokenizer_022
from test_experiment_022_runner import _apply as apply_022
from test_experiment_022_runner import calibrated, frozen, world  # noqa: F401 — fixtures

ROOT = Path(__file__).parents[1]
TABLE_022 = b0c.TABLE_022_RELATIVE_PATH


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_023_runner", ROOT / "experiments/023-block0-completion/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def toy_tokenizer_023(manifest, pool):
    """022's toy tokenizer plus every 023 candidate cue and frame piece (fresh ids below the fake's vocabulary)."""
    base = toy_tokenizer_022(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in b0c.CUE_CANDIDATES.values() for word in words]
    for texts in b0c.FRAME_CANDIDATES.values():
        for text in texts:
            words += [piece if index == 0 else " " + piece for index, piece in enumerate(text.replace("{cue}", "").split())]
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    assert next_id < 60000
    return type(base)(vocabulary)


@pytest.fixture(scope="module")
def base023(world, calibrated, tmp_path_factory):
    """022 frozen and calibrated on the fake; the 023 inherited-digest constants for that world's 022 files."""
    root = tmp_path_factory.mktemp("base023")
    shutil.copytree(calibrated, root, dirs_exist_ok=True)
    record = json.loads((root / ul.CALIBRATION_RELATIVE_PATH).read_text())
    confirmation_022 = json.loads((root / ul.CONFIRMATION_RELATIVE_PATH).read_text())
    inherited = {"calibration_file_sha256": rc.file_sha256(root / ul.CALIBRATION_RELATIVE_PATH), "calibration_content_sha256": record["content_sha256"],
                 "confirmation_file_sha256": rc.file_sha256(root / ul.CONFIRMATION_RELATIVE_PATH), "confirmation_content_sha256": confirmation_022["content_sha256"],
                 "table_file_sha256": rc.file_sha256(root / TABLE_022)}
    return {"root": root, "inherited": inherited}


def _apply(patch, world, base) -> None:
    apply_022(patch, world)
    for key, value in base["inherited"].items():
        patch.setitem(b0c.INHERITED_022, key, value)
    for name, value in (("B", 40), ("CROSS_CHECK_DRAWS", 2), ("CUE_QUOTA", 1), ("FRAME_QUOTA", 1)):
        patch.setattr(b0c, name, value)
    # The fake's measured-Δx3 ceiling is sometimes no better than its Level 0 in a Y2-like coordinated frame (gaps down
    # to −0.04), so with B = 40 (stop at one undefined draw) the fake would always stop. These tests exercise the phase
    # machinery; the gap rule itself is tested in tier A and by the undefined-draw stop below (GAP_MIN = 1e9).
    patch.setattr(b0c, "GAP_MIN", -1e9)


def _refuse_022_table(patch) -> None:
    """Any attempt to load Experiment 022's calibration table fails loudly."""
    original = torch.load

    def guarded(path, *args, **kwargs):
        if str(path).endswith("calibration-table.pt"):
            raise AssertionError(f"Experiment 022's calibration table was opened after extract: {path}")
        return original(path, *args, **kwargs)

    patch.setattr(torch, "load", guarded)


def make_runner(root: Path, world, **overrides):
    manifest, small, digests, lock_011, lock_012, lock_017 = world["fake"]
    logs: list[str] = []
    arguments = dict(root=root, results_path=root / "outputs/experiment-023/results.json", report_path=root / "outputs/experiment-023/report.md",
                     model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_023(manifest, small),
                     lock_011_loader=lambda path: dict(lock_011), lock_012_loader=lambda path: dict(lock_012), lock_017_loader=lambda path: dict(lock_017),
                     git_state=lambda: {"commit": COMMIT_A, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True,
                     changed_paths=lambda commit: [], log=logs.append)
    arguments.update(overrides)
    return runner_module.Runner(**arguments), logs


def _install(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)  # byte-identical, as the protocol installs a candidate


def _stage(world, base, tmp_path_factory, name: str, source: Path, step) -> Path:
    root = tmp_path_factory.mktemp(name)
    shutil.copytree(source, root, dirs_exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        _apply(patch, world, base)
        runner, logs = make_runner(root, world)
        step(runner, logs, root, patch)
    return root


@pytest.fixture(scope="module")
def extracted(world, base023, tmp_path_factory):
    def step(runner, logs, root, patch):
        assert runner.extract() == 0, logs[-4:]
        for candidate, target in zip(runner.candidate_cells_paths, (b0c.CELLS_DATA_RELATIVE_PATH, b0c.CELLS_INDEX_RELATIVE_PATH)):
            _install(candidate, root / target)

    return _stage(world, base023, tmp_path_factory, "extracted023", base023["root"], step)


@pytest.fixture(scope="module")
def frozen023(world, base023, extracted, tmp_path_factory):
    def step(runner, logs, root, patch):
        (root / TABLE_022).unlink()  # from here on the 022 table does not exist
        _refuse_022_table(patch)
        assert runner.freeze() == 0, logs[-2:]

    return _stage(world, base023, tmp_path_factory, "frozen023", extracted, step)


@pytest.fixture(scope="module")
def calibrated023(world, base023, frozen023, tmp_path_factory):
    def step(runner, logs, root, patch):
        _refuse_022_table(patch)
        assert runner.calibrate() == 0, logs[-4:]
        _install(runner.candidate_calibration_path, root / b0c.CALIBRATION_RELATIVE_PATH)

    return _stage(world, base023, tmp_path_factory, "calibrated023", frozen023, step)


@pytest.fixture(scope="module")
def locked023(world, base023, calibrated023, tmp_path_factory):
    def step(runner, logs, root, patch):
        _refuse_022_table(patch)
        assert runner.lock() == 0, logs[-4:]
        for name, relative in (("candidate-lock.json", b0c.LOCK_RELATIVE_PATH), ("candidate-preregistration.md", b0c.PREREGISTRATION_RELATIVE_PATH),
                               ("candidate-locked-y1-table.f64", b0c.Y1_TABLE_RELATIVE_PATH), ("candidate-locked-y1-table.json", b0c.Y1_TABLE_INDEX_RELATIVE_PATH)):
            _install(runner.output(name), root / relative)

    return _stage(world, base023, tmp_path_factory, "locked023", calibrated023, step)


@pytest.fixture
def sandbox(world, base023, tmp_path, monkeypatch):
    """A per-test copy of a stage directory, with every patch the fake needs."""
    _apply(monkeypatch, world, base023)

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
    inputs, confirmation_022, sha_022, _, forbidden = runner._base()
    return runner._confirmation(inputs, confirmation_022, sha_022, forbidden)[0]


# ---------------------------------------------------------------------------


def test_parser_has_exactly_the_seven_phases_and_no_overrides():
    parser = runner_module.build_parser()
    for phase in b0c.PHASES:
        assert parser.parse_args([phase]).phase == phase
    for forbidden_args in (["replicate-022"], ["calibrate", "--draws", "10"], ["extract", "--force"], ["confirm", "--stage", "2"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden_args)
    assert runner_module.PHASES == ("validate", "extract", "freeze", "calibrate", "lock", "confirm", "report")


def test_validate_loads_no_model_never_opens_the_022_table_and_refuses_tampering(world, base023, sandbox, monkeypatch):
    root = sandbox(base023["root"])

    def refuse(*args, **kwargs):
        raise AssertionError("validate loaded a model")

    _refuse_022_table(monkeypatch)
    runner, logs = make_runner(root, world, model_loader=refuse, tokenizer_loader=refuse)
    assert runner.validate() == 0, logs
    assert "not extracted yet" in logs[-1] and "not frozen yet" in logs[-1]
    record_path = root / ul.CALIBRATION_RELATIVE_PATH
    original = record_path.read_text()
    record_path.write_text(original + " ")
    assert runner.validate() == 1 and "022" in logs[-1]
    record_path.write_text(original)
    monkeypatch.setitem(b0c.FROZEN_BLOBS, "upstream_localization.py", "0" * 40)
    monkeypatch.setattr(ul, "load_frozen_inputs", lambda *args, **kwargs: pytest.fail("the frozen inputs were loaded before the pins were checked"))
    assert runner.validate() == 1 and "frozen modules" in logs[-1]


def test_changed_paths_list_both_sides_of_a_rename(tmp_path, monkeypatch):
    """A scientific file moved out of a scientific path is still a scientific change (git diff --no-renames)."""
    import subprocess

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src/program.py").write_text("x = 1\n" * 20)

    def git(*args):
        return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "one")
    first = git("rev-parse", "HEAD")
    (repo / "docs").mkdir()
    git("mv", "src/program.py", "docs/program.py")
    git("commit", "-q", "-m", "two")
    monkeypatch.setattr(runner_module, "ROOT", repo)
    changed = runner_module._git_changed_paths(first)
    assert changed == ["docs/program.py", "src/program.py"] and b0c.scientific_changes(changed) == ["src/program.py"]
    assert runner_module._git_changed_paths("0" * 40) is None  # not an ancestor


def test_extract_runs_no_prompt_verifies_e1_to_e6_and_writes_the_candidate_once(world, base023, sandbox, monkeypatch):
    root = sandbox(base023["root"])
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world)
    assert runner.extract() == 0, logs[-4:]
    assert not spy.counts  # weights only, under the no-forward-pass guard
    state = _state(runner)
    assert state["executed_prompt_keys"] == [] and state["phases"]["extract"]["status"] == "complete" and state["phases"]["extract"]["commit"] == COMMIT_A
    data, index = runner.candidate_cells_paths
    assert rc.file_sha256(data) == state["extract"]["data_sha256"] and rc.file_sha256(index) == state["extract"]["index_sha256"]
    loaded = json.loads(index.read_text())
    checks = loaded["extraction"]["checks"]
    assert set(checks) == {"E1", "E2", "E3", "E4", "E5", "E6"} and checks["E4"]["max"] == 0.0 and checks["E5"]["max_difference"] <= 1e-10
    assert checks["E3"]["columns"] == ["S", "Q", "SSEC"] and loaded["content_sha256"] == rc.content_digest(loaded)
    assert loaded["extraction"]["protocol_code_commit"] == COMMIT_A and loaded["extraction"]["module_blob"] == b0c.own_blob()
    assert loaded["source_022"]["table_file_sha256"] == base023["inherited"]["table_file_sha256"] and loaded["columns"] == list(b0c.CELL_COLUMNS)
    assert [entry[2] for entry in loaded["frames"]] == ["coordinated" if frame.template_id == ul.COORDINATED else "cue_final" for frame in b0c.exposed_units(runner._inputs()).frames]
    cells = ul.read_table(data, loaded)["cells"]
    table = torch.load(root / TABLE_022)
    units = b0c.exposed_units(runner._inputs())
    assert torch.equal(cells, b0c.cells_from_table(table, units)) and tuple(cells.shape) == (units.n_pairs, 8)
    with pytest.raises(b0c.PhaseError, match="extract runs once"):
        runner.extract()
    for candidate, target in zip(runner.candidate_cells_paths, (b0c.CELLS_DATA_RELATIVE_PATH, b0c.CELLS_INDEX_RELATIVE_PATH)):
        _install(candidate, root / target)
    assert runner.validate() == 0 and "exposed cells" in logs[-1]
    with pytest.raises(b0c.PhaseError, match="already installed"):
        runner.extract()


def test_extract_requires_the_bound_022_table_file_before_it_writes_anything(world, base023, sandbox):
    root = sandbox(base023["root"])
    runner, _ = make_runner(root, world)
    table = root / TABLE_022
    original = table.read_bytes()
    table.unlink()
    with pytest.raises(b0c.PhaseError, match="precondition"):
        runner.extract()
    table.write_bytes(original + b"\0")  # another file at the path
    with pytest.raises(b0c.PhaseError, match="precondition"):
        runner.extract()
    assert not runner.results_path.exists() and not any(path.exists() for path in runner.candidate_cells_paths)  # extract has not started


def test_a_table_whose_tensors_are_not_022s_stops_extract_for_review_and_writes_nothing(world, base023, sandbox, monkeypatch):
    root = sandbox(base023["root"])
    table = torch.load(root / TABLE_022)
    table["dc"][0, 0, 0] += 1e-9
    torch.save(table, root / TABLE_022)
    monkeypatch.setitem(b0c.INHERITED_022, "table_file_sha256", rc.file_sha256(root / TABLE_022))  # the file precondition holds; E1 checks the tensors
    runner, logs = make_runner(root, world)
    assert runner.extract() == 3 and "E1" in logs[-1] and "dc" in logs[-1]
    state = _state(runner)
    assert state["phases"]["extract"]["status"] == "stopped_for_review" and not any(path.exists() for path in runner.candidate_cells_paths)
    with pytest.raises(b0c.PhaseError, match="stopped_for_review"):
        runner.extract()


def test_a_failed_recheck_after_the_extraction_is_an_incident_and_writes_no_candidate(world, base023, sandbox, monkeypatch):
    root = sandbox(base023["root"])
    monkeypatch.setattr(runner_module.Runner, "_recheck", lambda self: {"ok": False, "message": "planted recheck failure"})
    runner, logs = make_runner(root, world)
    assert runner.extract() == 2 and "planted recheck failure" in logs[-1]
    state = _state(runner)
    assert state["phases"]["extract"]["incidents"][-1]["commit"] == COMMIT_A and not state["extract"].get("data_sha256")
    assert not any(path.exists() for path in runner.candidate_cells_paths)
    with pytest.raises(b0c.PhaseError, match="incident is recorded at this commit"):
        runner.extract()


def test_an_extraction_identity_failure_is_an_incident_bound_to_its_commit(world, base023, sandbox, monkeypatch):
    root = sandbox(base023["root"])
    monkeypatch.setitem(b0c.TOLERANCES, "E4", -1.0)  # E4 can never pass
    runner, logs = make_runner(root, world)
    assert runner.extract() == 2 and "E4" in logs[-1]
    state = _state(runner)
    assert state["phases"]["extract"]["incidents"][-1]["commit"] == COMMIT_A and not any(path.exists() for path in runner.candidate_cells_paths)
    assert state["phases"]["extract"]["incidents"][-1]["recheck_020_022"]["ok"]
    with pytest.raises(b0c.PhaseError, match="incident is recorded at this commit"):
        runner.extract()
    monkeypatch.setitem(b0c.TOLERANCES, "E4", 1e-9)
    runner, logs = make_runner(root, world, git_state=lambda: {"commit": COMMIT_B, "dirty": False})
    assert runner.extract() == 0, logs[-3:]  # a committed fix: a new commit may extract after a recorded incident


def test_freeze_is_tokenizer_only_excludes_022_and_a_shortfall_writes_nothing(world, base023, extracted, sandbox, monkeypatch):
    root = sandbox(extracted)

    def refuse(*args, **kwargs):
        raise AssertionError("freeze loaded a model")

    runner, logs = make_runner(root, world, model_loader=refuse)
    monkeypatch.setattr(b0c, "CUE_QUOTA", 99)
    assert runner.freeze() == 3 and "shortfall" in logs[-1]
    assert not (root / b0c.CONFIRMATION_RELATIVE_PATH).exists()
    monkeypatch.setattr(b0c, "CUE_QUOTA", 1)
    assert runner.freeze() == 0, logs
    payload = json.loads((root / b0c.CONFIRMATION_RELATIVE_PATH).read_text())
    assert payload["counts"] == {"classes": {stratum: 1 for stratum in b0c.STRATA}, "templates": {template: 1 for template in b0c.TEMPLATES}}
    assert [entry["word"] for entry in payload["cues"]] == ["general", "tons", "humble"]
    assert [entry["frame_id"] for entry in payload["frames"]] == ["cardinal-023-1", "quantifier-023-1", "coordinated-adjective-023-1"]
    assert payload["exclusion"]["sources"][-1]["source"] == "confirmation-022"
    assert json.loads((root / ul.CONFIRMATION_RELATIVE_PATH).read_text())["cues"][0]["token_id"] in payload["exclusion"]["cue_token_ids"]
    with pytest.raises(b0c.PhaseError, match="once"):
        runner.freeze()
    assert runner.validate() == 0 and "confirmation" in logs[-1]


def test_calibrate_runs_from_the_committed_cells_alone_without_the_022_table(world, base023, frozen023, sandbox, monkeypatch):
    root = sandbox(frozen023)
    assert not (root / TABLE_022).exists()
    _refuse_022_table(monkeypatch)

    def refuse(*args, **kwargs):
        raise AssertionError("calibrate loaded a model")

    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world, model_loader=refuse, tokenizer_loader=refuse)
    assert runner.calibrate() == 0, logs[-4:]
    assert not spy.counts
    state = _state(runner)
    record = json.loads(runner.candidate_calibration_path.read_text())
    b0c.verify_calibration_record(record)
    assert rc.file_sha256(runner.candidate_calibration_path) == state["calibration"]["record_sha256"] and state["executed_prompt_keys"] == []
    assert record["constants"]["B"] == 40 and record["kernel_check"]["passed"] and record["e6_max"] <= 1e-10 and set(record["conditions"]) == set(b0c.CONDITIONS)
    assert record["exposed_cells"]["data_sha256"] == state["extract"]["data_sha256"] and record["confirmation_023"]["content_sha256"] == _confirmation(runner).content_sha256
    assert record["pools"]["cues"] == {stratum: 1 for stratum in b0c.STRATA}
    arrays = torch.load(runner.draws_path)
    assert {key: rc.tensor_digest(value) for key, value in arrays.items()} == record["draw_arrays_sha256"]
    with pytest.raises(b0c.PhaseError, match="once"):
        runner.calibrate()
    assert runner.report() == 0 and "Calibration (exposed only" in runner.report_path.read_text()


def test_calibrate_refuses_an_uninstalled_altered_or_foreign_artifact(world, base023, frozen023, sandbox):
    root = sandbox(frozen023)
    runner, _ = make_runner(root, world, changed_paths=lambda commit: ["src/neural_decompiler/block0_completion.py"])
    with pytest.raises(b0c.PhaseError, match="scientific paths changed since extract"):
        runner.calibrate()
    runner, _ = make_runner(root, world, changed_paths=lambda commit: [b0c.CELLS_DATA_RELATIVE_PATH, b0c.CELLS_INDEX_RELATIVE_PATH, b0c.CONFIRMATION_RELATIVE_PATH])
    data = root / b0c.CELLS_DATA_RELATIVE_PATH
    raw = bytearray(data.read_bytes())
    raw[10] ^= 1
    data.write_bytes(bytes(raw))
    with pytest.raises(b0c.PhaseError, match="does not match its index"):
        runner.calibrate()
    (root / b0c.CELLS_INDEX_RELATIVE_PATH).unlink()
    with pytest.raises(b0c.PhaseError, match="install the candidate exposed cells"):
        runner.calibrate()


def test_the_undefined_draw_stop_writes_no_envelope_and_is_never_retried(world, base023, frozen023, sandbox, monkeypatch):
    root = sandbox(frozen023)
    monkeypatch.setattr(b0c, "GAP_MIN", 1e9)  # no draw is interpretable
    runner, logs = make_runner(root, world)
    assert runner.calibrate() == 3 and "stop" in logs[-1]
    state = _state(runner)
    assert state["phases"]["calibrate"]["status"] == "stopped_for_review" and not runner.candidate_calibration_path.exists()
    assert state["calibration"]["stop"]["offending"] == {condition: 40 for condition in b0c.CONDITIONS}
    assert runner.report() == 0
    assert "Calibration stopped for review" in runner.report_path.read_text() and "Y2/coordinated 40" in runner.report_path.read_text()
    with pytest.raises(b0c.PhaseError, match="stopped_for_review"):
        runner.calibrate()
    with pytest.raises(b0c.PhaseError, match="lock requires"):
        runner.lock()


def test_lock_runs_no_forward_pass_and_binds_the_y1_table_and_the_y2_specification(world, base023, calibrated023, sandbox, monkeypatch):
    root = sandbox(calibrated023)
    _refuse_022_table(monkeypatch)
    runner, _ = make_runner(root, world, changed_paths=lambda commit: ["src/neural_decompiler/upstream_localization.py"])
    with pytest.raises(b0c.PhaseError, match="scientific paths changed"):
        runner.lock()
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world, changed_paths=lambda commit: [b0c.CALIBRATION_RELATIVE_PATH, f"{b0c.EXPERIMENT_DIR}/README.md", "docs/x.md"])
    assert runner.lock() == 0, logs[-3:]
    assert not spy.counts
    lock = json.loads(runner.output("candidate-lock.json").read_text())
    n_nouns = len(lock["noun_keys"])
    assert lock["content_sha256"] == rc.content_digest(lock) and set(lock["conditions"]) == set(b0c.CONDITIONS)
    assert lock["y1_table"]["layout"] == [{"name": "cue_final", "shape": [3 * 72, 2, n_nouns]}, {"name": "coordinated", "shape": [3 * 36, 2, n_nouns]}]
    assert lock["y2_table"]["layout"] == [{"name": "cue_final", "shape": [3 * 2, 2, n_nouns]}, {"name": "coordinated", "shape": [3 * 1, 2, n_nouns]}]
    assert lock["y1_table"]["gates"]["I5"]["max"] == 0.0 and lock["y1_table"]["gates"]["algebra"]["max"] <= 1e-12
    assert lock["program"]["P1_mask"] == {"cue_final": 14, "coordinated": 30} and lock["semantics"]["scope"] == b0c.SCOPE
    assert all(entry["guard"] == {"min_inclusive": 0.90} for entry in lock["conditions"].values())
    assert runner.output("candidate-preregistration.md").read_text() == b0c.render_preregistration(lock)
    with pytest.raises(b0c.PhaseError, match="lock already written"):
        runner.lock()


def test_a_failed_recheck_at_lock_is_an_incident_and_writes_no_lock(world, base023, calibrated023, sandbox, monkeypatch):
    root = sandbox(calibrated023)
    monkeypatch.setattr(runner_module.Runner, "_recheck", lambda self: {"ok": False, "message": "planted recheck failure"})
    runner, logs = make_runner(root, world)
    assert runner.lock() == 2 and "planted recheck failure" in logs[-1]
    assert not runner.output("candidate-lock.json").exists() and not runner.output("candidate-preregistration.md").exists()
    assert _state(runner)["phases"]["lock"]["incidents"][-1]["commit"] == COMMIT_A
    with pytest.raises(b0c.PhaseError, match="lock identity incident is recorded"):
        runner.lock()


def test_a_forbidden_key_in_the_ledger_is_refused(world, base023, calibrated023, sandbox):
    root = sandbox(calibrated023)
    runner, _ = make_runner(root, world)
    state = _state(runner)
    confirmation_022 = json.loads((root / ul.CONFIRMATION_RELATIVE_PATH).read_text())
    state["executed_prompt_keys"] = [confirmation_022["manifest"]["S2-TARGET"]["Y2"][0]]  # a spent 022 key
    rd.write_results_state(runner.results_path, state)
    with pytest.raises(b0c.PhaseError, match="forbidden keys"):
        runner.lock()


def test_lock_refuses_a_confirmation_file_other_than_the_calibrated_one(world, base023, calibrated023, sandbox, monkeypatch):
    """The calibration read its counts from one committed confirmation file; lock refuses any other, even a fully valid
    freeze, and a results state that binds another one."""
    root = sandbox(calibrated023)
    runner, _ = make_runner(root, world)
    inputs, confirmation_022, sha_022, _, _ = runner._base()
    manifest, small = world["fake"][0], world["fake"][1]
    with monkeypatch.context() as patch:  # another eligible cue first: a different, fully valid freeze
        patch.setitem(b0c.CUE_CANDIDATES, "determiner-like", tuple(reversed(b0c.CUE_CANDIDATES["determiner-like"])))
        payload = b0c.freeze_payload(toy_tokenizer_023(manifest, small), inputs, confirmation_022, sha_022)
    path = root / b0c.CONFIRMATION_RELATIVE_PATH
    original = path.read_bytes()
    record = json.loads((root / b0c.CALIBRATION_RELATIVE_PATH).read_text())
    path.write_text(pm.canonical_json(payload) + "\n")
    assert _confirmation(runner).content_sha256 != record["confirmation_023"]["content_sha256"]  # it loads and verifies as a confirmation file
    spy = ExecutionSpy(monkeypatch)
    with pytest.raises(b0c.PhaseError, match="calibration record binds a confirmation file other than the committed one"):
        runner.lock()
    path.write_bytes(original)
    state = _state(runner)
    state["confirmation_023"] = {**state["confirmation_023"], "file_sha256": "0" * 64}
    rd.write_results_state(runner.results_path, state)
    with pytest.raises(b0c.PhaseError, match="results state binds a confirmation file other than the committed one"):
        runner.lock()
    assert not spy.counts and not runner.output("candidate-lock.json").exists()


def test_confirm_runs_stage1_the_barrier_and_stage2_once_each_and_scores_four_conditions(world, base023, locked023, sandbox, monkeypatch):
    root = sandbox(locked023)
    assert not (root / TABLE_022).exists()
    _refuse_022_table(monkeypatch)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world)
    assert runner.confirm() == 0, logs[-4:]
    state = _state(runner)
    confirmation = _confirmation(runner)
    assert Counter(spy.counts) == Counter({prompt.key: 1 for prompt in confirmation.all_prompts})  # each manifest key exactly once, nothing else
    assert set(state["executed_prompt_keys"]) == confirmation.manifest_keys()  # 023 executes nothing but its own manifest
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["I7"]["bitwise_equal"]
    results = state["confirmation"]
    assert set(results["conditions"]) == set(b0c.CONDITIONS) and results["aggregate_label"] is None
    assert all(entry["result"] in b0c.RESULTS for entry in results["conditions"].values()) and results["kernel_check"]["passed"]
    assert all(results["gates"][name]["max"] <= b0c.TOLERANCES[name] for name in ("I1", "I3", "I4"))
    stage1 = results["stage1"]
    y2 = stage1["y2_table"]
    assert rc.file_sha256(root / y2["data_path"]) == y2["file_sha256"] and stage1["digest"] == b0c.stage_one_digest(stage1)
    assert y2["gates"]["I5"]["max"] == 0.0 and y2["gates"]["algebra"]["max"] <= 1e-12
    tensors = torch.load(runner.stage2_path)
    assert {key: rc.tensor_digest(value) for key, value in tensors.items()} == results["stage2"]["tensors_sha256"]
    lock = json.loads((root / b0c.LOCK_RELATIVE_PATH).read_text())
    tables = {"Y1": ul.read_table(root / b0c.Y1_TABLE_RELATIVE_PATH, json.loads((root / b0c.Y1_TABLE_INDEX_RELATIVE_PATH).read_text())),
              "Y2": ul.read_table(root / y2["data_path"], json.loads((root / y2["index_path"]).read_text()))}
    for condition in b0c.CONDITIONS:  # every result reproduces from the saved measurements and the tables through the canonical path
        population, group = condition.split("/")
        cells = torch.stack([b0c.pair_cells(tensors[f"{population}/{group}/dc"][i], tables[population][group][i, 0], tables[population][group][i, 1],
                                            tensors[f"{population}/{group}/ceiling"][i]) for i in range(int(tensors[f"{population}/{group}/dc"].shape[0]))])
        scored = b0c.score_selection(cells, torch.arange(int(cells.shape[0])), lock["conditions"][condition]["envelope"]["bound"], condition)
        assert scored["result"] == results["conditions"][condition]["result"] and scored["g"] == results["conditions"][condition]["g"]
    assert set(results["descriptives"]["comparators"]) == set(b0c.CONDITIONS)
    with pytest.raises(b0c.PhaseError, match="confirm already started"):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "The four conditions" in report and "no aggregate label" in report and "not a mathematical upper bound" in report and b0c.SCOPE in report
    assert all(f"{condition}: **" in report for condition in b0c.CONDITIONS) and _state(runner)["phases"]["report"]["status"] == "complete"


@pytest.mark.parametrize("error", [RuntimeError("a descriptive comparator failed"), KeyboardInterrupt()], ids=["failure", "interruption"])
def test_a_descriptive_failure_after_scoring_keeps_the_four_results(world, base023, locked023, sandbox, monkeypatch, error):
    """The four results and the completed phase are on disk before any descriptive record runs; a descriptive failure
    is recorded as such (an interruption is recorded and raised) and never touches a result."""
    root = sandbox(locked023)

    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(b0c, "comparators", broken)
    runner, logs = make_runner(root, world)
    if isinstance(error, Exception):
        assert runner.confirm() == 0, logs[-3:]
    else:
        with pytest.raises(KeyboardInterrupt):
            runner.confirm()
    state = _state(runner)
    results = state["confirmation"]
    assert state["phases"]["confirm"]["status"] == "complete" and "incident" not in results
    assert set(results["conditions"]) == set(b0c.CONDITIONS) and all(entry["result"] in b0c.RESULTS for entry in results["conditions"].values())
    descriptives = results["descriptives"]
    assert descriptives["failures"]["comparators"]["type"] == type(error).__name__ and "comparators" not in descriptives
    assert "subsets" in descriptives and "p1_dx3_relative_error" in descriptives and "block0_profile" in descriptives
    with pytest.raises(b0c.PhaseError, match="confirm already started"):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "comparators was not computed" in report and all(f"{condition}: **" in report for condition in b0c.CONDITIONS)


def test_i7_refuses_a_y1_table_that_does_not_reproduce_before_any_fresh_prompt(world, base023, locked023, sandbox, monkeypatch):
    root = sandbox(locked023)
    original = b0c.prediction_tables

    def drifted(*args, **kwargs):
        out = original(*args, **kwargs)
        out["blocks"][0][1][0, 1, 0] += 1e-12  # a one-ulp-scale drift in P1's reconstruction
        return out

    monkeypatch.setattr(b0c, "prediction_tables", drifted)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world)
    assert runner.confirm() == 2 and "I7" in logs[-1]
    confirm = _state(runner)["phases"]["confirm"]
    assert not spy.counts and confirm["status"] == "not_started" and confirm["incidents"][-1]["commit"] == COMMIT_A
    monkeypatch.setattr(b0c, "prediction_tables", original)
    runner, _ = make_runner(root, world, git_state=lambda: {"commit": COMMIT_B, "dirty": False})
    with pytest.raises(b0c.PhaseError, match="I7 incident"):  # never retried, at any commit
        runner.confirm()
    assert not spy.counts


def test_a_y2_table_changed_before_the_barrier_is_an_incident_and_no_target_runs(world, base023, locked023, sandbox, monkeypatch):
    root = sandbox(locked023)
    original = b0c.stage_one

    def tampering(*args, **kwargs):
        record = original(*args, **kwargs)
        data = root / record["y2_table"]["data_path"]
        raw = bytearray(data.read_bytes())
        raw[0] ^= 1  # the table on disk no longer matches what stage 1 recorded
        data.write_bytes(bytes(raw))
        return record

    monkeypatch.setattr(b0c, "stage_one", tampering)
    spy = ExecutionSpy(monkeypatch)
    runner, logs = make_runner(root, world)
    assert runner.confirm() == 2 and "INCIDENT" in logs[-1]
    confirmation = _confirmation(runner)
    assert set(spy.counts) == {prompt.key for prompt in confirmation.stage1_prompts}  # stage 1 only; no S2-TARGET prompt ran
    state = _state(runner)
    assert state["confirmation"]["incident"]["phase"] == "confirm" and (root / b0c.Y2_TABLE_OUTPUT).exists()  # the evidence is preserved
    with pytest.raises(b0c.PhaseError, match="confirm already started"):
        runner.confirm()
