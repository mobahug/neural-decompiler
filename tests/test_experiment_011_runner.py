"""Phase isolation, the confirmation freeze, the weights-only lock, and the full state machine for the Experiment 011 runner on a six-layer fake."""

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
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M
from plural_fakes import TinyPlural
from test_encoding_read import toy_tokenizer_011

ROOT = Path(__file__).parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_011_runner", ROOT / "experiments/011-encoding-read-prospective/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for relative in (pm.MANIFEST_RELATIVE_PATH, pm.EXTENSION_RELATIVE_PATH, cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, ht.LOCK_RELATIVE_PATH):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    (tmp_path / er.INHERITED_010_EXTRACT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha256, extension = pm.load_inputs(tmp_path)
    confirmation_006 = cd.load_confirmation(tmp_path / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(tmp_path / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    pool = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
    # The 010-style extract: the fake's transport fractions over 63 × 24 (computed with the same code path Experiment 011 uses).
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    axes = cs.stage_axes(cache, weights, pool)
    fractions = {}
    for frame in pool.frames:
        ref, components = ra.capture_reference(model, head, pool.reference_prompt(frame), pool.single_nouns)
        functional = ra.read_functional(head, ref, axes["T"])
        records = {name: ra.measure_token_010(model, weights, head, ref, components, functional, name, token_id, axes["R0"], axes["T"], pool.single_nouns) for name, token_id in pool.tokens}
        plural = records[pool.plural_cue[frame.template_id]]
        for name, record in records.items():
            analysis = ra.fractions(record, plural, axes["T"], plural.g_E_inner)
            fractions[f"{name}|{frame.frame_id}"] = None if analysis is None else analysis["q_T"]
    payload = er.inherited_010_payload(fractions, source={"path": "fake"}, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256,
                                       confirmation_009_sha256=confirmation_009.content_sha256, model={"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision})
    (tmp_path / er.INHERITED_010_EXTRACT_RELATIVE_PATH).write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    return tmp_path, manifest


def make_runner(sandbox, monkeypatch, *, logs=None):
    root, manifest = sandbox
    logs = logs if logs is not None else []
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    monkeypatch.setattr(er, "DENOMINATOR_RELATIVE_FLOOR", 0.0)  # the fake's random head can make a template denominator tiny; the rule itself is unit-tested
    runner = runner_module.Runner(root=root, results_path=root / "outputs/experiment-011/results.json", report_path=root / "outputs/experiment-011/report.md",
                                  model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_011(manifest),
                                  git_state=lambda: {"commit": "a" * 40, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True, changed_paths=lambda commit: [],
                                  contract_runner=lambda: {"passed": True, "returncode": 0, "output_tail": "", "output_sha256": "x"}, baseline_loader=lambda lock: None, log=logs.append)
    return runner, logs


def test_parser_has_exactly_the_six_phases_and_no_overrides():
    parser = runner_module.build_parser()
    for phase in ("validate", "freeze-confirmation", "explore", "lock", "confirm", "report"):
        assert parser.parse_args([phase]).phase == phase
    for forbidden in (["calibrate"], ["confirm", "--force"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden)


def test_full_state_machine_and_weights_only_lock(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.validate() == 1
    assert runner.freeze_confirmation() == 0 and runner.freeze_confirmation() == 1 and runner.validate() == 0
    manifest, manifest_sha256, extension = pm.load_inputs(runner.root)
    confirmation_006 = cd.load_confirmation(runner.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(runner.root / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    confirmation = er.load_confirmation(runner.confirmation_path, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    assert runner.explore() == 0
    state = er.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["status"] == "complete" and exploration["replication"]["experiment_010"]["max_abs_deviation"] == 0.0
    assert len(exploration["tokens"]) == 63 and exploration["tolerances"]["tau_g"] >= er.TAU_MIN and exploration["tolerances"]["tau_M"] >= er.TAU_MIN
    assert not {prompt.key for prompt in confirmation.all_prompts} & set(state["executed_prompt_keys"])
    with pytest.raises(er.PhaseError):
        runner.explore()
    # The lock phase must not be able to run any prompt: every capture or intervention entry point raises while it runs.
    assert not exploration["summary"]["prediction_undefined"]
    with pytest.MonkeyPatch.context() as guard:
        for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, name):
                guard.setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the lock phase ran a forward pass")))
        assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    predictions_path = runner.results_path.parent / "candidate-predictions.md"
    lock = json.loads(candidate.read_text())
    assert lock["experiment"] == "011" and set(lock["predictions"]["tokens"]) == {token["word"] for token in confirmation.tokens}
    an = lock["predictions"]["tokens"]["an"]
    assert set(an["by_template"]) <= set(pm.TEMPLATE_ORDER) and "baseline_009" not in an and len(an["licensed_frames"]) <= 4
    text = predictions_path.read_text()
    assert "preregistered predictions" in text and "predicted g_E" in text
    with pytest.raises(er.PhaseError):
        runner.confirm()  # not installed
    shutil.copy(candidate, runner.root / er.LOCK_RELATIVE_PATH)
    (runner.root / er.PREDICTIONS_RELATIVE_PATH).write_text(text, encoding="utf-8")
    runner.changed_paths = lambda commit: ["src/neural_decompiler/encoding_read.py"]
    with pytest.raises(er.PhaseError, match="scientific paths"):
        runner.confirm()
    runner.changed_paths = lambda commit: [er.LOCK_RELATIVE_PATH, er.PREDICTIONS_RELATIVE_PATH, f"{er.EXPERIMENT_DIR}/README.md"]
    assert runner.confirm() == 0
    state = er.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["lock_predictions_reproduced_max_difference"] == 0.0
    results = state["confirmation"]
    assert results["outcome"]["label"] == "PRECONDITION_FAILED" or all(part in er.OUTCOME_Y1 + er.OUTCOME_Y2 for part in results["outcome"]["label"].split(" | "))
    assert set(results["frames"]) == {frame.frame_id for frame in confirmation.frames} and {prompt.key for prompt in confirmation.all_prompts} <= set(state["executed_prompt_keys"])
    for word, row in results["tokens"].items():
        if row["scored"]:
            assert row["n_valid_frames"] >= er.MIN_VALID_FRAMES_PER_TOKEN and row["g_E_mean"] is not None and row["q_T_mean"] is not None
    with pytest.raises(er.PhaseError):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "## Tier A" in report and "## Confirmation" in report


def test_validity_is_independent_of_candidates(sandbox, monkeypatch):
    """A frame's validity depends on the template's plural cue and cue pair only; every licensed candidate in a valid frame is scored."""
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.freeze_confirmation() == 0 and runner.explore() == 0
    assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    shutil.copy(candidate, runner.root / er.LOCK_RELATIVE_PATH)
    shutil.copy(runner.results_path.parent / "candidate-predictions.md", runner.root / er.PREDICTIONS_RELATIVE_PATH)
    assert runner.confirm() == 0
    results = er.load_results_state(runner.results_path)["confirmation"]
    manifest, manifest_sha256, extension = pm.load_inputs(runner.root)
    confirmation_006 = cd.load_confirmation(runner.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(runner.root / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    confirmation = er.load_confirmation(runner.confirmation_path, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    valid = set(results["valid_frames"])
    for token in confirmation.tokens:
        expected = sorted(frame_id for frame_id in token["licensed_frames"] if frame_id in valid)
        assert results["tokens"][token["word"]]["frames"] == expected  # no candidate dropped by its own q_T
