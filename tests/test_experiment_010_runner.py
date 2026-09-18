"""Phase isolation and the single attribution run of the Experiment 010 runner on a six-layer fake with the real frozen inputs."""

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
from neural_decompiler import read_assembly as ra
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M
from plural_fakes import TinyPlural

ROOT = Path(__file__).parents[1]
FAKE_CIRCUIT = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L03.H04",), ("L04.MLP", "L05.MLP"))


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_010_runner", ROOT / "experiments/010-read-direction-assembly/run.py")
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
    for relative in (pm.MANIFEST_RELATIVE_PATH, pm.EXTENSION_RELATIVE_PATH, cd.CONFIRMATION_RELATIVE_PATH, cs.EXPERIMENT_007_LOCK_PATH, ht.CONFIRMATION_RELATIVE_PATH, ht.LOCK_RELATIVE_PATH):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    for relative in (ss.INHERITED_EXTRACT_RELATIVE_PATH, cs.INHERITED_007_EXTRACT_RELATIVE_PATH, ra.INHERITED_009_EXTRACT_RELATIVE_PATH):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha256, extension = pm.load_inputs(tmp_path)
    confirmation_006 = cd.load_confirmation(tmp_path / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(tmp_path / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    pool = cd.exposed_pool(manifest, extension)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    model_record = {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}
    # 006-style extract (59 exposed nouns, 16 tokens × 12 frames).
    cache = pm.PromptCache(model, tuple(pool.nouns))
    responses = cd.measure_epatch_responses(model, weights, cache, pool.frames, pool.tokens, pool.reference_ids, circuit=FAKE_CIRCUIT)
    recorded = {f"{token}|{frame_id}": {"template_id": r.template_id, "mean_shift": pm._mean(list(r.shifts.values())), "delta_norm": float(r.delta_residual.norm()), "circuit_share": r.circuit_share} for (token, frame_id), r in responses.items()}
    payload = ss.inherited_extract_payload(recorded, source={"path": "fake", "file_sha256": "0" * 64, "state_sha256": "1" * 64, "run_id": "fake", "protocol_code_commit": "b" * 40, "explore_completed_at": "t"},
                                           manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256, model=model_record)
    (tmp_path / ss.INHERITED_EXTRACT_RELATIVE_PATH).write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    # 007-style extract (20 confirmation nouns, 24 tokens × 6 confirmation frames).
    cache_fresh = pm.PromptCache(model, tuple(confirmation_006.nouns))
    fresh_tokens = [(entry["word"], entry["token_id"]) for entry in confirmation_006.tokens]
    fresh = cd.measure_epatch_responses(model, weights, cache_fresh, confirmation_006.frames, fresh_tokens, confirmation_006.reference_ids, circuit=FAKE_CIRCUIT)
    lock_007 = json.loads((tmp_path / cs.EXPERIMENT_007_LOCK_PATH).read_text())
    payload_007 = cs.inherited_007_payload({f"{token}|{frame_id}": {"template_id": r.template_id, "mean_shift": pm._mean(list(r.shifts.values()))} for (token, frame_id), r in fresh.items()},
                                           source={"path": "fake"}, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256, lock_sha256=lock_007["content_sha256"], model=model_record)
    (tmp_path / cs.INHERITED_007_EXTRACT_RELATIVE_PATH).write_text(pm.canonical_json(payload_007) + "\n", encoding="utf-8")
    # 009-style extract (79 nouns, 23 tokens × 6 Experiment 009 frames).
    all_nouns = pool.nouns + confirmation_006.nouns
    cache_all = pm.PromptCache(model, tuple(all_nouns))
    tokens_009 = [(entry["word"], entry["token_id"]) for entry in confirmation_009.tokens]
    fresh_009 = cd.measure_epatch_responses(model, weights, cache_all, confirmation_009.frames, tokens_009, confirmation_009.reference_ids, circuit=FAKE_CIRCUIT)
    lock_009 = json.loads((tmp_path / ht.LOCK_RELATIVE_PATH).read_text())
    payload_009 = ra.inherited_009_payload({f"{token}|{frame_id}": {"template_id": r.template_id, "mean_shift": pm._mean(list(r.shifts.values()))} for (token, frame_id), r in fresh_009.items()},
                                           source={"path": "fake"}, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_009_sha256=confirmation_009.content_sha256, lock_sha256=lock_009["content_sha256"], model=model_record)
    (tmp_path / ra.INHERITED_009_EXTRACT_RELATIVE_PATH).write_text(pm.canonical_json(payload_009) + "\n", encoding="utf-8")
    return tmp_path


def make_runner(sandbox, monkeypatch, *, logs=None):
    logs = logs if logs is not None else []
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)  # the fake's random head carries little number signal
    runner = runner_module.Runner(root=sandbox, results_path=sandbox / "outputs/experiment-010/results.json", report_path=sandbox / "outputs/experiment-010/report.md",
                                  model_loader=lambda spec: make_fake_model(), git_state=lambda: {"commit": "a" * 40, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True,
                                  contract_runner=lambda: {"passed": True, "returncode": 0, "output_tail": "", "output_sha256": "x"}, log=logs.append)
    return runner, logs


def test_parser_has_exactly_three_phases():
    parser = runner_module.build_parser()
    for phase in ("validate", "explore", "report"):
        assert parser.parse_args([phase]).phase == phase
    for forbidden in (["lock"], ["confirm"], ["explore", "--force"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden)


def test_single_attribution_run_on_the_fake(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.validate() == 0
    with pytest.raises(ra.PhaseError):
        runner.report()
    assert runner.explore() == 0
    state = ra.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["status"] == "complete"
    for key in ("experiment_006", "experiment_007", "experiment_009"):
        assert exploration["replication"][key]["passed"] and exploration["replication"][key]["max_abs_deviation"] == 0.0
    assert len(exploration["tokens"]) == 63 and all(len(frames) == 24 for frames in exploration["per_frame"].values())
    assert exploration["identities"]["max_rho_identity_error"] <= ra.IDENTITY_TOLERANCE and exploration["identities"]["max_p1_cross_check"] <= ra.P1_CROSS_CHECK_TOLERANCE
    assert exploration["summary"]["label"].split("+")[0] in ra.SUMMARY_LABELS
    row = exploration["tokens"]["this"]
    assert row["class"] in ra.TOKEN_CLASSES and set(row["means"]["f_k"]) == set(ra.COMPONENT_ORDER) and set(row["neurons"]) == set(pm.TEMPLATE_ORDER)
    assert set(exploration["strata"]) == {"low", "mid", "high", "uninformative", "reference"} and set(exploration["components"]) == {"low", "high"}
    assert set(exploration["strata"]["reference"]) == {"cardinal:sg", "quantifier:sg"} and exploration["tokens"]["cardinal:sg"]["n_informative"] == 8 and exploration["tokens"]["quantifier:sg"]["n_informative"] == 16
    assert "cardinal:sg" not in exploration["strata"]["low"] and exploration["tokens"]["cardinal:sg"]["neurons"]["cardinal"]["n_80"] is None
    assert all(frame_id not in {f for f in exploration["per_frame"]["cardinal:sg"] if exploration["per_frame"]["cardinal:sg"][f] is not None} for frame_id in exploration["per_frame"]["cardinal:sg"] if frame_id.startswith(("cardinal", "coordinated")))
    # Ledger: exactly the clean cue prompts and the reference prompts of the 24 frames; all 80 nouns.
    expected_prompts = 24 * 3
    assert len(state["executed_prompt_keys"]) == expected_prompts and len(state["executed_noun_keys"]) == 80
    with pytest.raises(ra.PhaseError):
        runner.explore()
    assert runner.report() == 0
    text = runner.report_path.read_text()
    assert "## Ledger of the head-readable number signal" in text and "## Component contributions" in text and "## Neurons" in text


def test_replication_incident_blocks_a_rerun_at_the_same_commit(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    path = sandbox / ra.INHERITED_009_EXTRACT_RELATIVE_PATH
    payload = json.loads(path.read_text())
    key = next(iter(payload["responses"]))
    payload["responses"][key]["mean_shift"] += 1e-3
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in payload.items() if k != "content_sha256"}))
    path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    assert runner.explore() == 2
    state = ra.load_results_state(runner.results_path)
    assert "deviates" in state["exploration"]["incidents"][-1]["message"]
    with pytest.raises(ra.PhaseError, match="incident is recorded at this commit"):
        runner.explore()
    assert runner.report() == 0 and "## Incidents" in runner.report_path.read_text()
