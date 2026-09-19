"""Phase isolation, the confirmation freeze, the no-forward-pass lock, the two-stage confirm with its digest barrier, and the full state machine for the Experiment 013 runner on a six-layer fake."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from neural_decompiler import attention_paths as ap
from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import encoding_read as er
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from neural_decompiler.models import PYTHIA_70M
from plural_fakes import TinyPlural
from test_attention_paths import toy_tokenizer_013

ROOT = Path(__file__).parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_013_runner", ROOT / "experiments/013-attention-paths-frozen-pattern/run.py")
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
    for relative in (pm.MANIFEST_RELATIVE_PATH, pm.EXTENSION_RELATIVE_PATH, cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, ht.LOCK_RELATIVE_PATH, er.CONFIRMATION_RELATIVE_PATH, er.LOCK_RELATIVE_PATH,
                     lc.CONFIRMATION_RELATIVE_PATH, lc.LOCK_RELATIVE_PATH):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    (tmp_path / ap.INHERITED_012_LEDGER_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha256, extension = pm.load_inputs(tmp_path)
    confirmation_006 = cd.load_confirmation(tmp_path / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(tmp_path / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    confirmation_011 = er.load_confirmation(tmp_path / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": confirmation_006.content_sha256, "confirmation_009": confirmation_009.content_sha256, "confirmation_011": confirmation_011.content_sha256}
    pool_010 = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
    pool_012 = lc.build_pool_012(manifest, extension, confirmation_006, confirmation_009, confirmation_011)
    confirmation_012 = lc.load_confirmation(tmp_path / lc.CONFIRMATION_RELATIVE_PATH, pool_012, digests)
    digests["confirmation_012"] = confirmation_012.content_sha256
    pool = ap.build_pool_013(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012)
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    axes = cs.stage_axes(cache, weights, pool_010)  # the fake's own axes stand in for the Experiment 011 lock's
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
    fake_lock_011 = {"experiment": "011", "axes_vectors": {"T": axes["T"].direction.double().tolist(), "R0": axes["R0"].direction.double().tolist()}, "sigma_T": axes["T"].sigma, "read_weight": read.weight.tolist(),
                     "denominators": {"denominators": {template: read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}, "sigma_r": er.sigma_r_from_pairs(read, weights, pool_010.frames)},
                     "confirmation_011_sha256": confirmation_011.content_sha256, "content_sha256": "f" * 64}
    # The fake's Experiment 012 model: template-mean base states over the 30 frames exposed then; and its per-pair ledger over the 2724 recorded pairs.
    states = {}
    for frame in pool.frames:
        states[frame.frame_id] = ap.capture_frame_013(model, head, pool.reference_prompt(frame), pool.single_nouns, axes["T"])
    bases_012 = lc.template_bases(pool_012.frames, {fid: (s.x1, s.x2) for fid, s in states.items() if fid in {f.frame_id for f in pool_012.frames}})
    fake_lock_012 = {"experiment": "012", "axes_vectors": fake_lock_011["axes_vectors"], "sigma_T": axes["T"].sigma, "read_weight": read.weight.tolist(),
                     "base_states": lc.bases_to_json(bases_012, {template: len(pool_012.frames_of(template)) for template in pm.TEMPLATE_ORDER}), "defined_templates": list(pm.TEMPLATE_ORDER),
                     "denominators": {"weight_only": {"denominators": {template: read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}}}, "confirmation_012_sha256": confirmation_012.content_sha256,
                     "lock_011_sha256": fake_lock_011["content_sha256"], "content_sha256": "e" * 64}
    fpm = ap.FrozenPatternModel(read, lw, heads, bases_012)
    frames_012 = {frame.frame_id for frame in confirmation_012.frames}
    words_012 = {token["word"] for token in confirmation_012.tokens}
    entries = {}
    for frame in pool.frames:
        state = states[frame.frame_id]
        names = [name for name, _ in pool.tokens if (frame.frame_id not in frames_012 and name not in words_012) or (frame.frame_id in frames_012 and name in words_012)]
        plural_name = pool.plural_cue[frame.template_id]
        plural = ap.measure_pair_013(model, weights, head, state, plural_name, pool.token_id(plural_name), axes["R0"], axes["T"], pool.single_nouns)
        for name in names:
            record = ap.measure_pair_013(model, weights, head, state, name, pool.token_id(name), axes["R0"], axes["T"], pool.single_nouns)
            analysis = ap.analyse_pair_013(record, plural, model=fpm, weights=weights, state=state, axis_T=axes["T"])
            if analysis is not None:
                entries[f"{name}|{frame.frame_id}"] = ap.ledger_entry_from_013(analysis)
    assert len(entries) == ap.EXPECTED_LEDGER_SIZE
    payload = ap.inherited_ledger_payload(entries, source={"path": "fake"}, digests=digests | {"lock_012": fake_lock_012["content_sha256"]})
    (tmp_path / ap.INHERITED_012_LEDGER_RELATIVE_PATH).write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    return tmp_path, manifest, pool, fake_lock_011, fake_lock_012


def make_runner(sandbox, monkeypatch, *, logs=None):
    root, manifest, pool, fake_lock_011, fake_lock_012 = sandbox
    logs = logs if logs is not None else []
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    monkeypatch.setattr(ap, "CALIBRATION_TOLERANCE", 10.0)  # the fake's exposed RMSEs are not the design values; the check itself is unit-level
    runner = runner_module.Runner(root=root, results_path=root / "outputs/experiment-013/results.json", report_path=root / "outputs/experiment-013/report.md",
                                  model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_013(manifest, pool),
                                  lock_011_loader=lambda path: dict(fake_lock_011), lock_012_loader=lambda path: dict(fake_lock_012),
                                  git_state=lambda: {"commit": "a" * 40, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True, changed_paths=lambda commit: [],
                                  contract_runner=lambda: {"passed": True, "returncode": 0, "output_tail": "", "output_sha256": "x"}, log=logs.append)
    return runner, logs


def test_parser_has_exactly_the_six_phases_and_no_overrides():
    parser = runner_module.build_parser()
    for phase in ("validate", "freeze-confirmation", "explore", "lock", "confirm", "report"):
        assert parser.parse_args([phase]).phase == phase
    for forbidden in (["calibrate"], ["confirm", "--stage", "2"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden)


def test_full_state_machine_lock_without_forward_pass_and_stage_barrier(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.validate() == 1
    assert runner.freeze_confirmation() == 0 and runner.freeze_confirmation() == 1 and runner.validate() == 0
    pool, pool_010, lock_011, lock_012, ledger, confirmation, digests = runner._inputs()
    assert len(pool.tokens) == 111 and len(pool.frames) == 36 and len(confirmation.tokens) == 24 and len(confirmation.exposed_frame_prompts) == 24 * 36
    assert runner.explore() == 0, logs[-3:]
    state = ap.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["status"] == "complete" and exploration["replication"]["experiment_012"]["max_abs_deviation"] == 0.0 and exploration["replication"]["experiment_012"]["max_c_012_difference"] == 0.0
    assert len(exploration["pairs"]) == ap.EXPECTED_LEDGER_SIZE and set(exploration["locked_states"]) == {frame.frame_id for frame in pool.frames}
    assert exploration["calibration"]["tau_r"] == ap.TAU_R and exploration["calibration"]["tau_a"] == ap.TAU_A and exploration["exposed_check"]["n_tokens"] == 111
    assert exploration["exposed_check"]["head_identity_max_error"] < ap.OV_IDENTITY_TOLERANCE and "thy" in exploration and len(exploration["thy"]) == 6
    assert not {prompt.key for prompt in confirmation.all_prompts} & set(state["executed_prompt_keys"])
    with pytest.raises(ap.PhaseError):
        runner.explore()
    with pytest.MonkeyPatch.context() as guard:
        for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, name):
                guard.setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the lock phase ran a forward pass")))
        assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    predictions_path = runner.results_path.parent / "candidate-predictions.md"
    lock = json.loads(candidate.read_text())
    assert lock["experiment"] == "013" and len(lock["predictions"]["rows"]) == 24 * 36 and set(lock["predictions"]["token_means"]) == {token["word"] for token in confirmation.tokens}
    assert set(lock["predictions"]["rows"][0]) == set(ap.PREDICTION_COLUMNS) | {"compensation_case"} and lock["tolerances"]["tau_r"] == ap.TAU_R and lock["floors"]["y_r2"] == ap.Y_R2
    assert set(lock["locked_states"]) == {frame.frame_id for frame in pool.frames} and lock["lock_012_sha256"] == "e" * 64
    text = predictions_path.read_text()
    assert "preregistered predictions" in text and "predicted residual" in text
    with pytest.raises(ap.PhaseError):
        runner.confirm()  # not installed
    shutil.copy(candidate, runner.root / ap.LOCK_RELATIVE_PATH)
    (runner.root / ap.PREDICTIONS_RELATIVE_PATH).write_text(text, encoding="utf-8")
    runner.changed_paths = lambda commit: ["src/neural_decompiler/attention_paths.py"]
    with pytest.raises(ap.PhaseError, match="scientific paths"):
        runner.confirm()
    runner.changed_paths = lambda commit: [ap.LOCK_RELATIVE_PATH, ap.PREDICTIONS_RELATIVE_PATH, f"{ap.EXPERIMENT_DIR}/README.md", "README.md"]
    # The stage barrier: stage 2 refuses to run any fresh cue prompt unless stage 1's digested table is in the results state.
    original_stage_two = ap.stage_two
    seen = {}

    def guarded_stage_two(model, pool_, pool_010_, confirmation_, lock_, lock_012_, stage1, **kwargs):
        seen["digest"] = stage1["digest"]
        ap.assert_stage_one_digest(stage1)
        return original_stage_two(model, pool_, pool_010_, confirmation_, lock_, lock_012_, stage1, **kwargs)

    monkeypatch.setattr(ap, "stage_two", guarded_stage_two)
    assert runner.confirm() == 0, logs[-3:]
    state = ap.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["lock_predictions_reproduced_max_difference"] == 0.0
    results = state["confirmation"]
    assert results["stage1"]["digest"] == seen["digest"] == ap.table_digest(results["stage1"]["rows"], results["stage1"]["states"]) and len(results["stage1"]["rows"]) == 24 * 6
    parts = results["outcome"]["label"].split(" | ")
    assert parts[0] in ap.OUTCOME_Y1 and parts[1] in ap.OUTCOME_Y2 and parts[2] in ap.OUTCOME_Y3
    assert len(results["Y1"]["scored_tokens"]) == 24 and all(entry["n_valid_frames"] == 36 for entry in results["Y1"]["tokens"].values())
    assert {prompt.key for prompt in confirmation.exposed_frame_prompts} <= set(state["executed_prompt_keys"])
    with pytest.raises(ap.PhaseError):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "## Tier A" in report and "stage 1" in report and "stage 2" in report
    # A tampered stage-1 table is refused before any fresh cue prompt.
    tampered = json.loads(json.dumps(results["stage1"]))
    tampered["rows"][0]["r_hat"] += 1e-6
    with pytest.raises(ap.PhaseError, match="digest"):
        ap.assert_stage_one_digest(tampered)


def test_incidents_are_recorded_and_block_reruns(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.freeze_confirmation() == 0
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(ap, "capture_frame_013", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic capture failure")))
        assert runner.explore() == 2
    state = ap.load_results_state(runner.results_path)
    assert state["exploration"]["incidents"][-1]["phase"] == "explore" and state["phases"]["explore"]["status"] == "running"
    with pytest.raises(ap.PhaseError, match="incident"):
        runner.explore()
    runner.git_state = lambda: {"commit": "b" * 40, "dirty": False}
    assert runner.explore() == 0 and runner.lock() == 0
    shutil.copy(runner.results_path.parent / "candidate-lock.json", runner.root / ap.LOCK_RELATIVE_PATH)
    shutil.copy(runner.results_path.parent / "candidate-predictions.md", runner.root / ap.PREDICTIONS_RELATIVE_PATH)
    with pytest.MonkeyPatch.context() as guard:  # stage 1 fails on the first fresh frame's reference capture; stage 1's table is never digested
        guard.setattr(ap, "capture_frame_013", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic stage-1 failure")))
        assert runner.confirm() == 2
    state = ap.load_results_state(runner.results_path)
    assert state["confirmation"]["incident"]["phase"] == "confirm" and "stage1" not in state["confirmation"] and state["phases"]["confirm"]["status"] == "running"
    with pytest.raises(ap.PhaseError, match="confirm"):
        runner.confirm()
    assert runner.report() == 0 and "## Incidents" in runner.report_path.read_text()
