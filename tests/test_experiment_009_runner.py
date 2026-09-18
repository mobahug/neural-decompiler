"""Phase isolation, the confirmation freeze, and the full state machine for the Experiment 009 runner on a six-layer fake with the real frozen inputs."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import head_transport as ht
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M
from plural_fakes import TinyPlural
from test_head_transport import toy_tokenizer_009

ROOT = Path(__file__).parents[1]
FAKE_CIRCUIT = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L03.H04",), ("L04.MLP", "L05.MLP"))


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_009_runner", ROOT / "experiments/009-head-transport-rule/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


@pytest.fixture
def sandbox(tmp_path):
    for relative in (pm.MANIFEST_RELATIVE_PATH, pm.EXTENSION_RELATIVE_PATH, cd.CONFIRMATION_RELATIVE_PATH, ss.INHERITED_EXTRACT_RELATIVE_PATH, cs.EXPERIMENT_007_LOCK_PATH, ht.PROGRAM_SOURCE):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    (tmp_path / cs.INHERITED_007_EXTRACT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ht.EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha256, extension = pm.load_inputs(tmp_path)
    confirmation_006 = cd.load_confirmation(tmp_path / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    pool = cd.exposed_pool(manifest, extension)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    lock_007 = json.loads((tmp_path / cs.EXPERIMENT_007_LOCK_PATH).read_text())
    model_record = {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}
    cache = pm.PromptCache(model, tuple(pool.nouns))
    responses = cd.measure_epatch_responses(model, weights, cache, pool.frames, pool.tokens, pool.reference_ids, circuit=FAKE_CIRCUIT)
    recorded = {f"{token}|{frame_id}": {"template_id": r.template_id, "mean_shift": pm._mean(list(r.shifts.values())), "delta_norm": float(r.delta_residual.norm()), "circuit_share": r.circuit_share} for (token, frame_id), r in responses.items()}
    payload = ss.inherited_extract_payload(recorded, source={"path": "fake", "file_sha256": "0" * 64, "state_sha256": "1" * 64, "run_id": "fake", "protocol_code_commit": "b" * 40, "explore_completed_at": "t"},
                                           manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256, model=model_record)
    (tmp_path / ss.INHERITED_EXTRACT_RELATIVE_PATH).write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    cache_fresh = pm.PromptCache(model, tuple(confirmation_006.nouns))
    fresh_tokens = [(entry["word"], entry["token_id"]) for entry in confirmation_006.tokens]
    fresh = cd.measure_epatch_responses(model, weights, cache_fresh, confirmation_006.frames, fresh_tokens, confirmation_006.reference_ids, circuit=FAKE_CIRCUIT)
    recorded_007 = {f"{token}|{frame_id}": {"template_id": r.template_id, "mean_shift": pm._mean(list(r.shifts.values()))} for (token, frame_id), r in fresh.items()}
    payload_007 = cs.inherited_007_payload(recorded_007, source={"path": "fake"}, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256,
                                           lock_sha256=lock_007["content_sha256"], model=model_record)
    (tmp_path / cs.INHERITED_007_EXTRACT_RELATIVE_PATH).write_text(pm.canonical_json(payload_007) + "\n", encoding="utf-8")
    e_vectors, reference_vectors = cd._token_vectors(weights, pool.tokens, pool.reference_ids)
    contexts = ss.context_residuals(cache, pool.frames, pool.reference_ids)
    rows = ss.design_rows(e_vectors=e_vectors, reference_vectors=reference_vectors, responses=responses, frames=pool.frames, tokens=[name for name, _ in pool.tokens])
    program_dir = tmp_path / "outputs/fake-007-program"
    ss.export_program(program_dir, weights, ss.fit_supervised(rows, 1, reference_ids=pool.reference_ids), contexts)
    return tmp_path, program_dir, manifest


def make_runner(sandbox, monkeypatch, *, logs=None):
    root, program_dir, manifest = sandbox
    logs = logs if logs is not None else []
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)  # the fake's random head carries little number signal; keep every stage informative
    runner = runner_module.Runner(root=root, results_path=root / "outputs/experiment-009/results.json", report_path=root / "outputs/experiment-009/report.md", parameters_dir=root / "outputs/experiment-009/parameters",
                                  model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_009(manifest),
                                  program_007_loader=lambda: ss.load_linear_program(program_dir, ROOT / ht.PROGRAM_SOURCE),
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


def test_full_state_machine_on_the_fake(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    monkeypatch.setattr(ss, "RANKS", (1, 2))
    assert runner.validate() == 1  # not frozen yet
    with pytest.raises(ht.PhaseError):
        runner.explore()
    assert runner.freeze_confirmation() == 0 and runner.freeze_confirmation() == 1 and runner.validate() == 0
    manifest, manifest_sha256, extension = pm.load_inputs(runner.root)
    confirmation_006 = cd.load_confirmation(runner.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation = ht.load_confirmation(runner.confirmation_path, manifest, manifest_sha256, extension, confirmation_006)
    assert runner.explore() == 0
    state = ht.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["status"] == "complete"
    assert exploration["replication"]["experiment_006"]["max_abs_deviation"] == 0.0 and exploration["replication"]["experiment_007"]["max_abs_deviation"] == 0.0
    assert len(exploration["tokens"]) == 40 and set(exploration["mechanism"]["levels"]) == {"P1", "P2", "P3"}
    assert set(exploration["transport_rule"]["loco"]) == {name for name in exploration["tokens"]} and set(exploration["contrast_rule"]["loco"]) == {"1", "2"}
    assert not {prompt.key for prompt in confirmation.all_prompts} & set(state["executed_prompt_keys"])
    with pytest.raises(ht.PhaseError):
        runner.explore()
    # Force every axis lockable on the fake if the gates or the level rule failed (the state machine is what is under test here).
    forced = ht.load_results_state(runner.results_path)
    for key in ("transport_rule", "contrast_rule"):
        forced["exploration"][key]["gate"]["passed"] = True
    if forced["exploration"]["mechanism"]["locked_level"] is None:
        forced["exploration"]["mechanism"]["locked_level"] = 3
        forced["exploration"]["mechanism"]["tau_M"] = 0.5
    ht.write_results_state(runner.results_path, forced)
    with pytest.raises(ht.PhaseError):
        runner.confirm()  # lock not written
    assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    predictions = runner.results_path.parent / "candidate-predictions.md"
    lock = json.loads(candidate.read_text())
    assert lock["experiment"] == "009" and set(lock["predictions"]["tokens"]) == {token["word"] for token in confirmation.tokens}
    first = next(iter(lock["predictions"]["tokens"].values()))
    assert {"q_T", "contrast", "program_007"} <= set(next(iter(first["frames"].values())))
    text = predictions.read_text()
    assert "preregistered predictions" in text and "either" in text
    with pytest.raises(ht.PhaseError):
        runner.confirm()  # not installed
    shutil.copy(candidate, runner.root / ht.LOCK_RELATIVE_PATH)
    (runner.root / ht.PREDICTIONS_RELATIVE_PATH).write_text(text + "\n", encoding="utf-8")
    with pytest.raises(ht.PhaseError, match="predictions.md"):
        runner.confirm()  # tampered artifact
    (runner.root / ht.PREDICTIONS_RELATIVE_PATH).write_text(text, encoding="utf-8")
    runner.changed_paths = lambda commit: ["src/neural_decompiler/head_transport.py"]
    with pytest.raises(ht.PhaseError, match="scientific paths"):
        runner.confirm()
    runner.changed_paths = lambda commit: [ht.LOCK_RELATIVE_PATH, ht.PREDICTIONS_RELATIVE_PATH, f"{ht.EXPERIMENT_DIR}/README.md"]
    assert runner.confirm() == 0
    state = ht.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["lock_predictions_reproduced_max_difference"] == 0.0
    verdict = state["confirmation"]["outcome"]
    assert verdict["label"] == "CUE_EFFECT_NOT_REPLICATED" or all(part in {"HEAD_MECHANISM_CONFIRMED_P1", "HEAD_MECHANISM_CONFIRMED_P2", "HEAD_MECHANISM_CONFIRMED_P3", "HEAD_MECHANISM_NOT_SUPPORTED", "TRANSPORT_RULE_PREDICTED", "TRANSPORT_RULE_FAILED", "CONTRAST_RULE_PREDICTED", "CONTRAST_RULE_FAILED", "NOT_LOCKED"} for part in verdict["label"].split(" | "))
    assert {prompt.key for prompt in confirmation.all_prompts} <= set(state["executed_prompt_keys"])
    with pytest.raises(ht.PhaseError):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "## Tier A" in report and "## Confirmation" in report
