"""Phase isolation, the confirmation freeze, the no-forward-pass lock, and the full state machine for the Experiment 012 runner on a six-layer fake."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import encoding_read as er
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from neural_decompiler.models import PYTHIA_70M
from plural_fakes import TinyPlural
from test_layer_correction import toy_tokenizer_012

ROOT = Path(__file__).parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_012_runner", ROOT / "experiments/012-layer-correction-token-local/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


REPO_FILES = (pm.MANIFEST_RELATIVE_PATH, pm.EXTENSION_RELATIVE_PATH, cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, ht.LOCK_RELATIVE_PATH, er.CONFIRMATION_RELATIVE_PATH, er.LOCK_RELATIVE_PATH)


def _copy_repo_files(root: Path) -> None:
    for relative in REPO_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, root / relative)


@pytest.fixture(scope="module")
def fake_world(tmp_path_factory):
    """Built once per module (immutable: the manifest, the fake 011 lock — copied by the loader — and the two ledgers' JSON text)."""
    tmp_path = tmp_path_factory.mktemp("world")
    _copy_repo_files(tmp_path)
    manifest, manifest_sha256, extension = pm.load_inputs(tmp_path)
    confirmation_006 = cd.load_confirmation(tmp_path / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(tmp_path / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    confirmation_011 = er.load_confirmation(tmp_path / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": confirmation_006.content_sha256, "confirmation_009": confirmation_009.content_sha256, "confirmation_011": confirmation_011.content_sha256}
    pool_010 = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
    pool = lc.build_pool_012(manifest, extension, confirmation_006, confirmation_009, confirmation_011)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
        model = make_fake_model()
        weights = pm.Weights.from_model(model)
        head = ht.HeadWeights.from_model(model)
        cache = pm.PromptCache(model, tuple(pool.nouns))
        axes = cs.stage_axes(cache, weights, pool_010)  # the fake's own axes stand in for the Experiment 011 lock's
        plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
        read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
        fake_lock_011 = {"experiment": "011", "axes_vectors": {"T": axes["T"].direction.double().tolist(), "R0": axes["R0"].direction.double().tolist()}, "sigma_T": axes["T"].sigma, "read_weight": read.weight.tolist(),
                         "denominators": {"denominators": {template: read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}, "sigma_r": er.sigma_r_from_pairs(read, weights, pool_010.frames)},
                         "confirmation_011_sha256": confirmation_011.content_sha256, "content_sha256": "f" * 64}
        # The two extracts: the fake's per-component fractions over 63 × 24 (Experiment 010) and over the 011 tokens' licensed 011 frames (142), computed with the code path Experiment 012 uses.
        entries_010, entries_011 = {}, {}
        frames_011 = {frame.frame_id for frame in confirmation_011.frames}
        words_011 = {token["word"]: token for token in confirmation_011.tokens}
        for frame in pool.frames:
            ref, components = ra.capture_reference(model, head, pool.reference_prompt(frame), pool.single_nouns)
            functional = ra.read_functional(head, ref, axes["T"])
            records = {name: ra.measure_token_010(model, weights, head, ref, components, functional, name, token_id, axes["R0"], axes["T"], pool.single_nouns) for name, token_id in pool.tokens}
            plural = records[pool.plural_cue[frame.template_id]]
            for name, record in records.items():
                entry = lc.ledger_entry(ra.fractions(record, plural, axes["T"], plural.g_E_inner))
                if frame.frame_id not in frames_011 and name not in words_011:
                    entries_010[f"{name}|{frame.frame_id}"] = entry
                elif frame.frame_id in frames_011 and name in words_011 and frame.frame_id in words_011[name]["licensed_frames"]:
                    entries_011[f"{name}|{frame.frame_id}"] = entry
    assert len(entries_010) == 63 * 24 and len(entries_011) == 142
    texts = {}
    for entries, experiment, relative in ((entries_010, "010", lc.INHERITED_010_LEDGER_RELATIVE_PATH), (entries_011, "011", lc.INHERITED_011_LEDGER_RELATIVE_PATH)):
        payload = lc.inherited_ledger_payload(entries, experiment=experiment, source={"path": "fake"}, digests=digests)
        texts[relative] = pm.canonical_json(payload) + "\n"
    return manifest, fake_lock_011, texts


def _populate(root: Path, monkeypatch, fake_world):
    """The repo files, the fake's inherited records and the per-test constants, into ``root``."""
    manifest, fake_lock_011, texts = fake_world
    _copy_repo_files(root)
    for relative, text in texts.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    return root, manifest, fake_lock_011


@pytest.fixture
def sandbox(tmp_path, monkeypatch, fake_world):
    """A fresh writable root per test: the repo files and the fake's two ledgers, written into this test's own directory."""
    return _populate(tmp_path, monkeypatch, fake_world)


@pytest.fixture(scope="module")
def explored_world(tmp_path_factory, fake_world):
    """Built once per module and never modified (tests copy it): a root taken through validate (refused before the freeze), freeze-confirmation (once; a second
    freeze refused), an explore attempt stopped by a synthetic incident at commit 0…0, the refusal to explore again at that commit, the resumed explore at
    commit a…a (complete) and lock with every capture entry point disabled. The module's single full explore is that resumed run; the state-machine test
    asserts its record, and the incident test takes its confirm-incident half from a copy of this post-lock state."""
    root = tmp_path_factory.mktemp("explored")
    with pytest.MonkeyPatch.context() as patch:
        runner, logs = make_runner(_populate(root, patch, fake_world), patch)
        assert runner.validate() == 1
        assert runner.freeze_confirmation() == 0 and runner.freeze_confirmation() == 1 and runner.validate() == 0
        runner.git_state = lambda: {"commit": "0" * 40, "dirty": False}
        with pytest.MonkeyPatch.context() as guard:
            guard.setattr(lc, "analyse_pair", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic identity failure")))
            assert runner.explore() == 2, logs[-3:]
        with pytest.raises(lc.PhaseError, match="incident"):
            runner.explore()
        runner.git_state = lambda: {"commit": "a" * 40, "dirty": False}
        assert runner.explore() == 0, logs[-3:]
        with pytest.MonkeyPatch.context() as guard:
            for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
                if hasattr(pm, name):
                    guard.setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the lock phase ran a forward pass")))
            assert runner.lock() == 0
    return root, tuple(logs)


@pytest.fixture
def explored_sandbox(tmp_path_factory, monkeypatch, fake_world, explored_world):
    """A fresh writable copy of the explored world for one test — its results state, candidate lock and predictions included — in its own directory (distinct from
    the test's ``sandbox``, which stays a pre-freeze root); the world itself is never modified."""
    root, _ = explored_world
    copy = tmp_path_factory.mktemp("explored-copy")
    shutil.copytree(root, copy, dirs_exist_ok=True)
    return _populate(copy, monkeypatch, fake_world)


def make_runner(sandbox, monkeypatch, *, logs=None):
    root, manifest, fake_lock_011 = sandbox
    logs = logs if logs is not None else []
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    original_validity = er.denominator_validity

    def all_defined(denominators, sigma_r):  # the fake's random head can make a denominator tiny or negative; the rule itself is unit-tested in Experiment 011
        verdict = original_validity(denominators, sigma_r)
        verdict["defined"] = {template: True for template in denominators}
        verdict["n_defined"] = len(denominators)
        return verdict

    monkeypatch.setattr(er, "denominator_validity", all_defined)
    runner = runner_module.Runner(root=root, results_path=root / "outputs/experiment-012/results.json", report_path=root / "outputs/experiment-012/report.md",
                                  model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_012(manifest), lock_011_loader=lambda path: dict(fake_lock_011),
                                  git_state=lambda: {"commit": "a" * 40, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True, changed_paths=lambda commit: [],
                                  contract_runner=lambda: {"passed": True, "returncode": 0, "output_tail": "", "output_sha256": "x"}, log=logs.append)
    return runner, logs


def test_parser_has_exactly_the_six_phases_and_no_overrides():
    parser = runner_module.build_parser()
    for phase in ("validate", "freeze-confirmation", "explore", "lock", "confirm", "report"):
        assert parser.parse_args([phase]).phase == phase
    for forbidden in (["calibrate"], ["confirm", "--force"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden)


def test_full_state_machine_and_no_forward_pass_lock(explored_sandbox, monkeypatch):
    runner, logs = make_runner(explored_sandbox, monkeypatch)
    assert runner.freeze_confirmation() == 1 and runner.validate() == 0  # frozen once in the explored world (validate was refused there before the freeze); never overwritten
    pool, pool_010, lock_011, ledgers, confirmation, digests = runner._inputs()
    assert len(pool.tokens) == 87 and len(pool.frames) == 30 and len(confirmation.tokens) == 24 and len(confirmation.frames) == 6
    # explore ran once in the explored world: a synthetic incident at commit 0…0, the refusal to rerun at that commit, then the complete run at a…a; this test asserts that run's record
    state = lc.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["attempts"] == 2 and state["phases"]["explore"]["attempt_commits"] == ["0" * 40, "a" * 40] and exploration["incidents"][-1]["commit"] == "0" * 40 and state["protocol_code_commit"] == "a" * 40
    assert state["phases"]["explore"]["status"] == "complete"
    assert exploration["replication"]["experiment_010"]["max_abs_deviation"] == 0.0 and exploration["replication"]["experiment_011"]["max_abs_deviation"] == 0.0
    assert len(exploration["tokens"]) == 87 and set(exploration["base_states"]) == set(pm.TEMPLATE_ORDER) and all(entry["n_frames"] == 10 for entry in exploration["base_states"].values())
    assert exploration["tolerances"]["tau_c"] >= lc.TAU_MIN and exploration["tolerances"]["tau_P"] >= lc.TAU_MIN and exploration["tolerances"]["calibration"].startswith("leave-one-frame-out")
    assert exploration["exposed_check"]["y1_lofo"]["n"] == 87 and "r2" in exploration["exposed_check"]["y1_lofo"] and exploration["exposed_check"]["r2_lofo_subsamples"]["draws"] == lc.SUBSAMPLE_DRAWS
    assert set(exploration["neurons"]) == set(lc.MLP_KEYS) and len(exploration["neurons"]["L02.MLP"]["per_template"]["cardinal"]["top"]) == min(lc.TOP_NEURONS, 16)
    some_pair = next(iter(exploration["pairs"].values()))
    assert {"c_L", "c_M", "c_H", "c_k", "P1", "P1_prime", "q_T", "g_E", "own", "ladder", "neurons", "lofo"} <= set(some_pair)
    assert not {prompt.key for prompt in confirmation.all_prompts} & set(state["executed_prompt_keys"])
    with pytest.raises(lc.PhaseError):
        runner.explore()
    assert not exploration["summary"]["prediction_undefined"]
    # The lock phase must not be able to run any prompt: every capture or intervention entry point raises while it runs.
    # lock ran once in the explored world with every capture entry point disabled (no forward pass); this test asserts the candidates it wrote
    candidate = runner.results_path.parent / "candidate-lock.json"
    predictions_path = runner.results_path.parent / "candidate-predictions.md"
    lock = json.loads(candidate.read_text())
    assert lock["experiment"] == "012" and set(lock["predictions"]["tokens"]) == {token["word"] for token in confirmation.tokens} and set(lock["base_states"]) == set(pm.TEMPLATE_ORDER)
    entry = lock["predictions"]["tokens"]["half"]
    assert set(entry["by_template"]) == set(pm.TEMPLATE_ORDER) and {"c_hat", "c_mlp1", "c_mlp2", "c_par", "c_perp", "interaction", "g_E", "q_hat_prime", "q_hat"} <= set(entry["by_template"]["cardinal"])
    assert entry["means"]["c_hat"] == pytest.approx(sum(entry["by_template"][t]["c_hat"] for t in pm.TEMPLATE_ORDER) / 3) and len(entry["licensed_frames"]) == 6
    assert lock["floors"]["y1_r2"] == lc.Y1_R2 and lock["lock_011_sha256"] == "f" * 64 and lock["read_weight"] == lock_011["read_weight"]
    text = predictions_path.read_text()
    assert "preregistered predictions" in text and "predicted ĉ" in text
    with pytest.raises(lc.PhaseError):
        runner.confirm()  # not installed
    shutil.copy(candidate, runner.root / lc.LOCK_RELATIVE_PATH)
    (runner.root / lc.PREDICTIONS_RELATIVE_PATH).write_text(text, encoding="utf-8")
    runner.changed_paths = lambda commit: ["src/neural_decompiler/layer_correction.py"]
    with pytest.raises(lc.PhaseError, match="scientific paths"):
        runner.confirm()
    runner.changed_paths = lambda commit: [lc.LOCK_RELATIVE_PATH, lc.PREDICTIONS_RELATIVE_PATH, f"{lc.EXPERIMENT_DIR}/README.md", f"{lc.EXPERIMENT_DIR}/evidence/x.md", "README.md"]
    assert runner.confirm() == 0, logs[-3:]
    state = lc.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["lock_predictions_reproduced_max_difference"] == 0.0
    results = state["confirmation"]
    assert results["outcome"]["label"] == "PRECONDITION_FAILED" or all(part in lc.OUTCOME_Y1 + lc.OUTCOME_Y2 for part in results["outcome"]["label"].split(" | "))
    assert set(results["frames"]) == {frame.frame_id for frame in confirmation.frames} and {prompt.key for prompt in confirmation.all_prompts} <= set(state["executed_prompt_keys"])
    for word, row in results["tokens"].items():
        if row["scored"]:
            assert row["n_valid_frames"] >= lc.MIN_VALID_FRAMES_PER_TOKEN and row["c_hat_locked_mean"] is not None and row["c_L_mean"] is not None and row["base_point_term"] is not None
    if "Y1" in results:
        assert set(results["Y1"]) >= {"spearman", "mae", "r2", "tau_c", "r2_floor", "passed", "failing"} and set(results["Y2"]) >= {"spearman", "mae", "tau_P", "passed"}
        assert set(results["neuron_overlap_exposed_vs_fresh"]["L02.MLP"]) == set(pm.TEMPLATE_ORDER)
    with pytest.raises(lc.PhaseError):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "## Tier A" in report and "## Confirmation" in report and "leave-one-frame-out" in report


def test_validity_is_independent_of_candidates(explored_sandbox, monkeypatch):
    """A frame's validity depends on the template's plural cue and cue pair only; every fresh token in a valid frame is scored."""
    runner, logs = make_runner(explored_sandbox, monkeypatch)
    # freeze, explore and lock ran once per module in the explored world; this test starts from a copy of its post-lock state
    shutil.copy(runner.results_path.parent / "candidate-lock.json", runner.root / lc.LOCK_RELATIVE_PATH)
    shutil.copy(runner.results_path.parent / "candidate-predictions.md", runner.root / lc.PREDICTIONS_RELATIVE_PATH)
    assert runner.confirm() == 0
    results = lc.load_results_state(runner.results_path)["confirmation"]
    pool, pool_010, lock_011, ledgers, confirmation, digests = runner._inputs()
    valid = set(results["valid_frames"])
    for token in confirmation.tokens:
        expected = sorted(frame_id for frame_id in token["licensed_frames"] if frame_id in valid)
        assert results["tokens"][token["word"]]["frames"] == expected  # no candidate dropped by its own values


def test_incidents_are_recorded_and_block_reruns(sandbox, explored_sandbox, monkeypatch):
    """An IncidentError inside explore or confirm is recorded with its commit and the phase cannot be re-run at that commit / in this protocol version."""
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.freeze_confirmation() == 0
    original = lc.analyse_pair
    calls = {"n": 0}

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise pm.IncidentError("synthetic identity failure")
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(lc, "analyse_pair", failing)
        assert runner.explore() == 2
    state = lc.load_results_state(runner.results_path)
    incident = state["exploration"]["incidents"][-1]
    assert incident["phase"] == "explore" and incident["commit"] == "a" * 40 and "synthetic" in incident["message"] and state["phases"]["explore"]["status"] == "running"
    with pytest.raises(lc.PhaseError, match="incident"):
        runner.explore()
    # A committed fix (a new commit) permits explore to run again; then the same failure at confirm is terminal.
    # A committed fix (a new commit) permits explore to run again: the phase starts and records a second attempt at the new commit (a fresh synthetic incident stops it at once);
    # the resumed run's completion and lock run once per module in the explored world, whose record the state-machine test asserts.
    runner.git_state = lambda: {"commit": "b" * 40, "dirty": False}
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(lc, "analyse_pair", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic identity failure")))
        assert runner.explore() == 2
    state = lc.load_results_state(runner.results_path)
    assert state["phases"]["explore"]["attempts"] == 2 and state["phases"]["explore"]["attempt_commits"] == ["a" * 40, "b" * 40] and state["exploration"]["incidents"][-1]["commit"] == "b" * 40
    # The same failure at confirm is terminal: from a fresh copy of the explored world's post-lock state, with its candidates installed.
    runner, logs = make_runner(explored_sandbox, monkeypatch)
    shutil.copy(runner.results_path.parent / "candidate-lock.json", runner.root / lc.LOCK_RELATIVE_PATH)
    shutil.copy(runner.results_path.parent / "candidate-predictions.md", runner.root / lc.PREDICTIONS_RELATIVE_PATH)
    with pytest.MonkeyPatch.context() as guard:  # the plural cue's E-patch runs in every defined fresh frame, valid or not
        guard.setattr(lc, "measure_pair", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic confirm failure")))
        assert runner.confirm() == 2
    state = lc.load_results_state(runner.results_path)
    assert state["confirmation"]["incident"]["phase"] == "confirm" and state["phases"]["confirm"]["status"] == "running"
    with pytest.raises(lc.PhaseError, match="confirm"):
        runner.confirm()
    assert runner.report() == 0 and "## Incidents" in runner.report_path.read_text()
