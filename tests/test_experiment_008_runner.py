"""Phase isolation and the single discovery run of the Experiment 008 runner on a six-layer fake with the real frozen inputs."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M
from plural_fakes import TinyPlural

ROOT = Path(__file__).parents[1]
FAKE_CIRCUIT = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L03.H04",), ("L04.MLP", "L05.MLP"))


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_008_runner", ROOT / "experiments/008-cue-suppression-localization/run.py")
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
    for relative in (pm.MANIFEST_RELATIVE_PATH, pm.EXTENSION_RELATIVE_PATH, cd.CONFIRMATION_RELATIVE_PATH, ss.INHERITED_EXTRACT_RELATIVE_PATH, cs.EXPERIMENT_007_LOCK_PATH, ss.PROGRAM_007_SOURCE if hasattr(ss, "PROGRAM_007_SOURCE") else cs.PROGRAM_007_SOURCE):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    (tmp_path / cs.INHERITED_007_EXTRACT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha256, extension = pm.load_inputs(tmp_path)
    confirmation = cd.load_confirmation(tmp_path / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    pool = cd.exposed_pool(manifest, extension)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    lock = json.loads((tmp_path / cs.EXPERIMENT_007_LOCK_PATH).read_text())
    model_record = {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}
    # The 006-style extract: the fake's exposed responses averaged over the exposed nouns.
    cache = pm.PromptCache(model, tuple(pool.nouns))
    responses = cd.measure_epatch_responses(model, weights, cache, pool.frames, pool.tokens, pool.reference_ids, circuit=FAKE_CIRCUIT)
    recorded = {f"{token}|{frame_id}": {"template_id": r.template_id, "mean_shift": pm._mean(list(r.shifts.values())), "delta_norm": float(r.delta_residual.norm()), "circuit_share": r.circuit_share} for (token, frame_id), r in responses.items()}
    payload = ss.inherited_extract_payload(recorded, source={"path": "fake", "file_sha256": "0" * 64, "state_sha256": "1" * 64, "run_id": "fake", "protocol_code_commit": "b" * 40, "explore_completed_at": "t"},
                                           manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation.content_sha256, model=model_record)
    (tmp_path / ss.INHERITED_EXTRACT_RELATIVE_PATH).write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    # The 007-style extract: the fake's confirmation-token responses on the confirmation frames averaged over the confirmation nouns.
    cache_fresh = pm.PromptCache(model, tuple(confirmation.nouns))
    fresh_tokens = [(entry["word"], entry["token_id"]) for entry in confirmation.tokens]
    fresh = cd.measure_epatch_responses(model, weights, cache_fresh, confirmation.frames, fresh_tokens, confirmation.reference_ids, circuit=FAKE_CIRCUIT)
    recorded_007 = {f"{token}|{frame_id}": {"template_id": r.template_id, "mean_shift": pm._mean(list(r.shifts.values()))} for (token, frame_id), r in fresh.items()}
    payload_007 = cs.inherited_007_payload(recorded_007, source={"path": "fake"}, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation.content_sha256,
                                           lock_sha256=lock["content_sha256"], model=model_record)
    (tmp_path / cs.INHERITED_007_EXTRACT_RELATIVE_PATH).write_text(pm.canonical_json(payload_007) + "\n", encoding="utf-8")
    # A rank-1 supervised program fitted on the fake stands in for the 007 program.
    e_vectors, reference_vectors = cd._token_vectors(weights, pool.tokens, pool.reference_ids)
    contexts = ss.context_residuals(cache, pool.frames, pool.reference_ids)
    rows = ss.design_rows(e_vectors=e_vectors, reference_vectors=reference_vectors, responses=responses, frames=pool.frames, tokens=[name for name, _ in pool.tokens])
    fit = ss.fit_supervised(rows, 1, reference_ids=pool.reference_ids)
    program_dir = tmp_path / "outputs/fake-program"
    ss.export_program(program_dir, weights, fit, contexts)
    return tmp_path, program_dir


def make_runner(sandbox, monkeypatch, *, logs=None):
    root, program_dir = sandbox
    logs = logs if logs is not None else []

    def loader():
        program = ss.load_linear_program(program_dir, ROOT / cs.PROGRAM_007_SOURCE)
        return program, {"parameters_sha256": "fake", "lock_sha256": "fake", "rank": program.rank, "kind": program.kind}

    runner = runner_module.Runner(root=root, results_path=root / "outputs/experiment-008/results.json", report_path=root / "outputs/experiment-008/report.md",
                                  program_loader=loader, model_loader=lambda spec: make_fake_model(), git_state=lambda: {"commit": "a" * 40, "dirty": False},
                                  versions=lambda: {"torch": "test"}, tracked=lambda path: True,
                                  contract_runner=lambda: {"passed": True, "returncode": 0, "output_tail": "", "output_sha256": "x"}, log=logs.append)
    return runner, logs


def test_parser_has_exactly_three_phases():
    parser = runner_module.build_parser()
    for phase in ("validate", "explore", "report"):
        assert parser.parse_args([phase]).phase == phase
    for forbidden in (["lock"], ["confirm"], ["explore", "--force"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden)


def test_validate_and_the_single_discovery_run(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.validate() == 0
    with pytest.raises(cs.PhaseError):
        runner.report()  # nothing explored yet
    assert runner.explore() == 0
    state = cs.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["status"] == "complete"
    assert exploration["replication"]["experiment_006"]["passed"] and exploration["replication"]["experiment_006"]["max_abs_deviation"] == 0.0
    assert exploration["replication"]["experiment_007"]["passed"] and exploration["replication"]["experiment_007"]["max_abs_deviation"] == 0.0
    assert len(exploration["tokens"]) == 40 and all(len(frames) == 18 for frames in exploration["per_frame"].values())
    assert exploration["summary"]["label"] == "AXIS_ARTIFACT" or exploration["summary"]["label"].startswith(("LOCALIZED_", "MIXED", "CONTEXT_LOCALIZED", "PROBE_INVALID"))
    assert exploration["identity"]["max_error"] <= cs.IDENTITY_TOLERANCE
    for row in exploration["tokens"].values():
        assert row["stratum"] in ("suppressed", "amplified", "ordinary") and set(row["fractions"]) == set(cs.VECTOR_STAGES) | {"c"}
    # The ledger holds exactly the pool's prompts: 18 frames × (sg, pl, ref, 40 tokens), and all 80 nouns.
    manifest, manifest_sha256, extension = pm.load_inputs(runner.root)
    confirmation = cd.load_confirmation(runner.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    pool = cs.build_pool(manifest, extension, confirmation)
    expected = {prompt.key for prompt in pm.manifest_prompts(pool.frames)} | {pool.reference_prompt(frame).key for frame in pool.frames} | {pool.token_prompt(frame, name).key for frame in pool.frames for name, _ in pool.tokens}
    assert set(state["executed_prompt_keys"]) == expected and len(state["executed_noun_keys"]) == 80
    with pytest.raises(cs.PhaseError):
        runner.explore()  # once
    assert runner.report() == 0
    text = runner.report_path.read_text()
    assert "## Summary" in text and "## Oriented traces" in text and "## Determiner group" in text


def test_replication_incident_is_recorded_and_blocks_a_rerun_at_the_same_commit(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    path = runner.root / cs.INHERITED_007_EXTRACT_RELATIVE_PATH
    payload = json.loads(path.read_text())
    key = next(iter(payload["responses"]))
    payload["responses"][key]["mean_shift"] += 1e-3
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in payload.items() if k != "content_sha256"}))
    path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    assert runner.explore() == 2
    state = cs.load_results_state(runner.results_path)
    assert "deviates" in state["exploration"]["incidents"][-1]["message"] and state["phases"]["explore"]["status"] == "running"
    with pytest.raises(cs.PhaseError, match="incident is recorded at this commit"):
        runner.explore()
    assert runner.report() == 0 and "## Incidents" in runner.report_path.read_text()
