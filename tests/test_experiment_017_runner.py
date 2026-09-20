"""Phase isolation, the confirmation freeze, the no-forward-pass lock, the two-stage confirm with its digest barrier and frame guard, and the full state machine for the Experiment 017 runner on a six-layer fake."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from neural_decompiler import attention_paths as ap
from neural_decompiler import attention_patterns as atp
from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import encoding_read as er
from neural_decompiler import frame_channels as fch
from neural_decompiler import head_pattern as hp
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import neuron_feature as nf
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from plural_fakes import TinyPlural
from test_head_pattern import toy_tokenizer_017

ROOT = Path(__file__).parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_017_runner", ROOT / "experiments/017-transport-head-pattern/run.py")
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
                     lc.CONFIRMATION_RELATIVE_PATH, lc.LOCK_RELATIVE_PATH, ap.CONFIRMATION_RELATIVE_PATH, ap.LOCK_RELATIVE_PATH, nf.CONFIRMATION_RELATIVE_PATH, nf.LOCK_RELATIVE_PATH, atp.CONFIRMATION_RELATIVE_PATH, atp.LOCK_RELATIVE_PATH,
                     fch.CONFIRMATION_RELATIVE_PATH, fch.LOCK_RELATIVE_PATH):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    (tmp_path / hp.INHERITED_016_EXTRACT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha256, extension = pm.load_inputs(tmp_path)
    confirmation_006 = cd.load_confirmation(tmp_path / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(tmp_path / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    confirmation_011 = er.load_confirmation(tmp_path / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": confirmation_006.content_sha256, "confirmation_009": confirmation_009.content_sha256, "confirmation_011": confirmation_011.content_sha256}
    pool_010 = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
    pool_012 = lc.build_pool_012(manifest, extension, confirmation_006, confirmation_009, confirmation_011)
    confirmation_012 = lc.load_confirmation(tmp_path / lc.CONFIRMATION_RELATIVE_PATH, pool_012, digests)
    digests["confirmation_012"] = confirmation_012.content_sha256
    pool_013 = ap.build_pool_013(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012)
    confirmation_013 = ap.load_confirmation(tmp_path / ap.CONFIRMATION_RELATIVE_PATH, pool_013, digests)
    digests["confirmation_013"] = confirmation_013.content_sha256
    pool_014 = nf.build_pool_014(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013)
    confirmation_014 = nf.load_confirmation(tmp_path / nf.CONFIRMATION_RELATIVE_PATH, pool_014, digests)
    digests["confirmation_014"] = confirmation_014.content_sha256
    pool_015 = atp.build_pool_015(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014)
    confirmation_015 = atp.load_confirmation(tmp_path / atp.CONFIRMATION_RELATIVE_PATH, pool_015, digests)
    digests["confirmation_015"] = confirmation_015.content_sha256
    pool_016 = fch.build_pool_016(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015)
    confirmation_016 = fch.load_confirmation(tmp_path / fch.CONFIRMATION_RELATIVE_PATH, pool_016, digests)
    digests["confirmation_016"] = confirmation_016.content_sha256
    pool = hp.build_pool_017(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016)
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    programs = atp.programs_from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    axes = cs.stage_axes(cache, weights, pool_010)
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
    fake_lock_011 = {"experiment": "011", "axes_vectors": {"T": axes["T"].direction.double().tolist(), "R0": axes["R0"].direction.double().tolist()}, "sigma_T": axes["T"].sigma, "read_weight": read.weight.tolist(),
                     "denominators": {"denominators": {template: read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}, "sigma_r": er.sigma_r_from_pairs(read, weights, pool_010.frames)},
                     "confirmation_011_sha256": confirmation_011.content_sha256, "content_sha256": "f" * 64}
    states = {frame.frame_id: ap.capture_frame_013(model, head, pool.reference_prompt(frame), pool.single_nouns, axes["T"]) for frame in pool.frames}
    bases_012 = lc.template_bases(pool_012.frames, {fid: (s.x1, s.x2) for fid, s in states.items() if fid in {f.frame_id for f in pool_012.frames}})
    fake_lock_012 = {"experiment": "012", "axes_vectors": fake_lock_011["axes_vectors"], "sigma_T": axes["T"].sigma, "read_weight": read.weight.tolist(),
                     "base_states": lc.bases_to_json(bases_012, {template: len(pool_012.frames_of(template)) for template in pm.TEMPLATE_ORDER}), "defined_templates": list(pm.TEMPLATE_ORDER),
                     "confirmation_012_sha256": confirmation_012.content_sha256, "lock_011_sha256": "f" * 64, "content_sha256": "e" * 64}
    fake_lock_014 = {"experiment": "014", "locked_states": {fid: states[fid].locked_state() for fid in {f.frame_id for f in pool_014.frames}}, "confirmation_014_sha256": confirmation_014.content_sha256, "lock_013_sha256": "d" * 64, "content_sha256": "c" * 64}
    fake_lock_015 = {"experiment": "015", "locked_states": {fid: atp.locked_state(states[fid]) for fid in {f.frame_id for f in pool_015.frames}}, "confirmation_015_sha256": confirmation_015.content_sha256, "lock_014_sha256": "c" * 64, "content_sha256": "b" * 64}
    fake_lock_016 = {"experiment": "016", "locked_states": {fid: atp.locked_state(states[fid]) for fid in {f.frame_id for f in pool_016.frames}}, "confirmation_016_sha256": confirmation_016.content_sha256, "lock_015_sha256": "b" * 64, "content_sha256": "a" * 64}
    stage1_digests = {frame.frame_id: ap.state_digest(atp.locked_state(states[frame.frame_id])) for frame in confirmation_016.frames}
    fpm = ap.FrozenPatternModel(read, lw, heads, bases_012)
    fcm = fch.FrameChannelModel(read, lw, programs, bases_012, axes["R0"].direction.double())
    context = fch.AnalysisContext(fcm, fpm, heads, weights, axes["T"])
    # The fake's Experiment 016 extract, the pairs 016 recorded: Experiment 015's 6180 (014's 4884 plus the 015 tokens in all 54 frames) plus the 016 tokens in all 66 frames.
    frames_012 = {frame.frame_id for frame in confirmation_012.frames}
    frames_013 = {frame.frame_id for frame in confirmation_013.frames}
    frames_014 = {frame.frame_id for frame in confirmation_014.frames}
    frames_015 = {frame.frame_id for frame in confirmation_015.frames}
    frames_016 = {frame.frame_id for frame in confirmation_016.frames}
    words_012 = {token["word"] for token in confirmation_012.tokens}
    words_013 = {token["word"] for token in confirmation_013.tokens}
    words_014 = {token["word"] for token in confirmation_014.tokens}
    words_015 = {token["word"] for token in confirmation_015.tokens}
    words_016 = {token["word"] for token in confirmation_016.tokens}
    entries = {}
    for frame in pool.frames:
        state = states[frame.frame_id]
        rows = atp.reference_rows(programs, state.x1_all, state.x2_all)
        old_frame = frame.frame_id not in frames_012 and frame.frame_id not in frames_013 and frame.frame_id not in frames_014 and frame.frame_id not in frames_015 and frame.frame_id not in frames_016
        if frame.frame_id in frames_016:
            names = [name for name, _ in pool.tokens if name in words_016]
        elif frame.frame_id in frames_015:
            names = [name for name, _ in pool.tokens if name in words_015 or name in words_016]
        else:
            names = [name for name, _ in pool.tokens if name in words_016 or name in words_015 or name in words_014 or (name in words_013 and frame.frame_id not in frames_014) or (name in words_012 and frame.frame_id in frames_012)
                     or (name not in words_012 and name not in words_013 and name not in words_014 and name not in words_015 and name not in words_016 and old_frame)]
        plural_name = pool.plural_cue[frame.template_id]
        plural = fch.measure_pair(model, weights, head, state, plural_name, pool.token_id(plural_name), axes["R0"], axes["T"], pool.single_nouns)
        for name in names:
            record = fch.measure_pair(model, weights, head, state, name, pool.token_id(name), axes["R0"], axes["T"], pool.single_nouns)
            analysis = fch.analyse_pair_016(record, plural, context=context, state=state, rows=rows)
            if analysis is not None:
                entries[f"{name}|{frame.frame_id}"] = hp.extract_entry_016(analysis)
    assert len(entries) == hp.EXPECTED_EXTRACT_SIZE_016
    (tmp_path / hp.INHERITED_016_EXTRACT_RELATIVE_PATH).write_text(pm.canonical_json(hp.inherited_extract_payload(entries, source={"path": "fake"}, digests=digests | {"lock_016": "a" * 64}, stage1_state_digests=stage1_digests)) + "\n", encoding="utf-8")
    return tmp_path, manifest, pool, fake_lock_011, fake_lock_012, fake_lock_014, fake_lock_015, fake_lock_016


def make_runner(sandbox, monkeypatch, *, logs=None):
    root, manifest, pool, fake_lock_011, fake_lock_012, fake_lock_014, fake_lock_015, fake_lock_016 = sandbox
    logs = logs if logs is not None else []
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    fake_lock_013 = {"experiment": "013", "locked_states": {}, "confirmation_013_sha256": json.loads((root / ap.CONFIRMATION_RELATIVE_PATH).read_text())["content_sha256"], "lock_012_sha256": "e" * 64, "content_sha256": "d" * 64}
    runner = runner_module.Runner(root=root, results_path=root / "outputs/experiment-017/results.json", report_path=root / "outputs/experiment-017/report.md",
                                  model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_017(manifest, pool),
                                  lock_011_loader=lambda path: dict(fake_lock_011), lock_012_loader=lambda path: dict(fake_lock_012), lock_013_loader=lambda path: dict(fake_lock_013), lock_014_loader=lambda path: dict(fake_lock_014),
                                  lock_015_loader=lambda path: dict(fake_lock_015), lock_016_loader=lambda path: dict(fake_lock_016),
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
    pool, pool_010, lock_011, lock_012, lock_016, extract, confirmation, digests = runner._inputs()
    assert len(pool.tokens) == 207 and len(pool.frames) == 66 and len(confirmation.tokens) == 24 and len(confirmation.exposed_frame_prompts) == 24 * 66 and len(confirmation.frames) == 12
    assert runner.explore() == 0, logs[-3:]
    state = hp.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["status"] == "complete" and exploration["replication"]["experiment_016"]["max_abs_deviation"] == 0.0
    assert len(exploration["pairs"]) == hp.EXPECTED_EXTRACT_SIZE_016 and set(exploration["locked_states"]) == {frame.frame_id for frame in pool.frames} and all(set(s) == {"p_c", "p_t", "x1_all", "x2_all", "x3_all"} for s in exploration["locked_states"].values())
    assert all(len(s["x3_all"]) == s["p_t"] + 1 == len(s["x1_all"]) for s in exploration["locked_states"].values()) and any(s["p_t"] == s["p_c"] + 1 for s in exploration["locked_states"].values())
    assert set(exploration["bases_3"]) == set(pm.TEMPLATE_ORDER) and exploration["bases_3"]["coordinated-adjective"]["p_t"] is not None and exploration["bases_3"]["cardinal"]["p_t"] is None and exploration["bases_3"]["cardinal"]["n_frames"] == 22
    assert exploration["program"]["3"]["rotary_dim"] == 0 and set(exploration["program"]) == {"1", "2", "3"} and exploration["exposed_check"]["n_tokens"] == 207 and "frozen_cue_final" in exploration["summary"]
    assert all(value < 1e-4 for key, value in exploration["identities"].items()) and exploration["identities"]["head_level1_recovery"] < 1e-9 and exploration["identities"]["level1_recovery"] < 1e-9
    assert {"I4_x3", "I5_head_row", "I6_dT", "I7_split", "I5_reference_head_row", "head_level1_recovery"} <= set(exploration["identities"])
    assert set(exploration["exposed_check"]["split"]["per_template"]) == set(pm.TEMPLATE_ORDER) and exploration["exposed_check"]["split"]["per_template"]["cardinal"]["n_pairs"] > 0
    assert not {prompt.key for prompt in confirmation.all_prompts} & set(state["executed_prompt_keys"])
    with pytest.raises(hp.PhaseError):
        runner.explore()
    with pytest.MonkeyPatch.context() as guard:
        for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, name):
                guard.setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the lock phase ran a forward pass")))
        assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    predictions_path = runner.results_path.parent / "candidate-predictions.md"
    lock = json.loads(candidate.read_text())
    assert lock["experiment"] == "017" and len(lock["predictions"]["rows"]) == 24 * 66 and set(lock["predictions"]["rows"][0]) == set(hp.PREDICTION_COLUMNS) and lock["floors"]["frame_guard_r2"] == hp.FRAME_GUARD_R2
    assert lock["lock_016_sha256"] == "a" * 64 and lock["program"]["3"]["n_heads"] == 8 and "token_means" in lock["predictions"] and set(lock["bases_3"]) == set(pm.TEMPLATE_ORDER) and lock["head"] == "L03.H04"
    text = predictions_path.read_text()
    assert "preregistered predictions" in text and "predicted ΔT̂" in text
    with pytest.raises(hp.PhaseError):
        runner.confirm()  # not installed
    shutil.copy(candidate, runner.root / hp.LOCK_RELATIVE_PATH)
    (runner.root / hp.PREDICTIONS_RELATIVE_PATH).write_text(text, encoding="utf-8")
    runner.changed_paths = lambda commit: ["src/neural_decompiler/head_pattern.py"]
    with pytest.raises(hp.PhaseError, match="scientific paths"):
        runner.confirm()
    runner.changed_paths = lambda commit: [hp.LOCK_RELATIVE_PATH, hp.PREDICTIONS_RELATIVE_PATH, f"{hp.EXPERIMENT_DIR}/README.md", "README.md"]
    seen = {}
    original_stage_two = hp.stage_two

    def guarded_stage_two(model, pool_, pool_010_, confirmation_, lock_, lock_012_, stage1, **kwargs):
        seen["digest"] = stage1["digest"]
        ap.assert_stage_one_digest(stage1)
        return original_stage_two(model, pool_, pool_010_, confirmation_, lock_, lock_012_, stage1, **kwargs)

    monkeypatch.setattr(hp, "stage_two", guarded_stage_two)
    assert runner.confirm() == 0, logs[-3:]
    state = hp.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["lock_predictions_reproduced_max_difference"] == 0.0
    results = state["confirmation"]
    assert results["stage1"]["digest"] == seen["digest"] == ap.table_digest(results["stage1"]["rows"], results["stage1"]["states"]) and len(results["stage1"]["rows"]) == 24 * 12
    assert all(set(s) == {"p_c", "p_t", "x1_all", "x2_all", "x3_all"} for s in results["stage1"]["states"].values())
    parts = results["outcome"]["label"].split(" | ")
    assert parts[0] in hp.OUTCOME_Y1 and parts[1] in hp.OUTCOME_Y2 and parts[2] in hp.OUTCOME_Y3
    assert len(results["Y1"]["scored_tokens"]) == 24 and all(entry["n_valid_frames"] == 66 and entry["n_cue_final"] == 44 for entry in results["Y1"]["tokens"].values())
    assert set(results["Y2"]["per_frame"]) <= set(results["valid_frames"]) and results["Y3"]["cue_final"]["n_pairs"] > 0 and results["Y3"]["coordinated_split"]["n_pairs"] > 0 and results["Y3"]["coordinated_split"]["expected_not_rejected"]
    assert "ladder" in results["Y1"]["descriptive"] and results["Y1"]["descriptive"]["ladder"]["decoded_c_L"]["level1_error_max"] < 1e-3 and results["Y1"]["descriptive"]["ladder"]["rows"]["level1"] == pytest.approx(1.0, abs=1e-6)
    assert all(value < 1e-4 for value in results["identities"].values())
    assert {prompt.key for prompt in confirmation.exposed_frame_prompts} <= set(state["executed_prompt_keys"])
    with pytest.raises(hp.PhaseError):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "## Tier A" in report and "stage 1" in report and "stage 2" in report and "ladder (rows / ΔT / Π R²)" in report and "decoded c_L" in report and "Split — cue-final" in report and "Coordinated split" in report and "Template quantifier" in report
    assert "invalid at stage 1" in report or "- frame " in report  # the twelve fresh frames are reported individually regardless of the outcome


def test_incidents_are_recorded_and_block_reruns(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.freeze_confirmation() == 0
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(hp, "capture_frame_017", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic capture failure")))
        assert runner.explore() == 2
    state = hp.load_results_state(runner.results_path)
    assert state["exploration"]["incidents"][-1]["phase"] == "explore" and state["phases"]["explore"]["status"] == "running"
    with pytest.raises(hp.PhaseError, match="incident"):
        runner.explore()
    runner.git_state = lambda: {"commit": "b" * 40, "dirty": False}
    assert runner.explore() == 0 and runner.lock() == 0
    shutil.copy(runner.results_path.parent / "candidate-lock.json", runner.root / hp.LOCK_RELATIVE_PATH)
    shutil.copy(runner.results_path.parent / "candidate-predictions.md", runner.root / hp.PREDICTIONS_RELATIVE_PATH)
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(hp, "capture_frame_017", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic stage-1 failure")))
        assert runner.confirm() == 2
    state = hp.load_results_state(runner.results_path)
    assert state["confirmation"]["incident"]["phase"] == "confirm" and "stage1" not in state["confirmation"] and state["phases"]["confirm"]["status"] == "running"
    with pytest.raises(hp.PhaseError, match="confirm"):
        runner.confirm()
    assert runner.report() == 0 and "## Incidents" in runner.report_path.read_text()


def test_identity_violation_is_an_incident(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.freeze_confirmation() == 0
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(hp, "X3_IDENTITY_TOLERANCE", 0.0)  # the exact chain is float32-noisy against the capture: the layer-3 identity must stop the phase
        assert runner.explore() == 2
    state = hp.load_results_state(runner.results_path)
    assert "layer-3 residuals" in state["exploration"]["incidents"][-1]["message"]
