"""Phase isolation, the confirmation freeze, the no-forward-pass lock, the ranking and subsets locked from the exposed pool only, the two-stage confirm with its digest barrier and the guards, and the full state machine for the Experiment 018 runner on a six-layer fake with a 2048-neuron block 2."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from neural_decompiler import attention_paths as ap
from neural_decompiler import attention_patterns as atp
from neural_decompiler import block_concentration as bc
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
from test_block_concentration import toy_tokenizer_018

ROOT = Path(__file__).parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_018_runner", ROOT / "experiments/018-block2-concentration/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8, d_mlp=bc.N_NEURONS)


REPO_FILES = (pm.MANIFEST_RELATIVE_PATH, pm.EXTENSION_RELATIVE_PATH, cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, ht.LOCK_RELATIVE_PATH, er.CONFIRMATION_RELATIVE_PATH, er.LOCK_RELATIVE_PATH,
              lc.CONFIRMATION_RELATIVE_PATH, lc.LOCK_RELATIVE_PATH, ap.CONFIRMATION_RELATIVE_PATH, ap.LOCK_RELATIVE_PATH, nf.CONFIRMATION_RELATIVE_PATH, nf.LOCK_RELATIVE_PATH, atp.CONFIRMATION_RELATIVE_PATH, atp.LOCK_RELATIVE_PATH,
              fch.CONFIRMATION_RELATIVE_PATH, fch.LOCK_RELATIVE_PATH, hp.CONFIRMATION_RELATIVE_PATH, hp.LOCK_RELATIVE_PATH, hp.INHERITED_016_EXTRACT_RELATIVE_PATH)


def _copy_repo_files(root: Path) -> None:
    for relative in REPO_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, root / relative)


@pytest.fixture(scope="module")
def fake_world(tmp_path_factory):
    """Built once per module: the pool, the fake locks 011–017 and the fake's Experiment 017 extract (9636 pairs recomputed on the fake)."""
    tmp_path = tmp_path_factory.mktemp("world")
    _copy_repo_files(tmp_path)
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
    pool_017 = hp.build_pool_017(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016)
    confirmation_017 = hp.load_confirmation(tmp_path / hp.CONFIRMATION_RELATIVE_PATH, pool_017, digests)
    digests["confirmation_017"] = confirmation_017.content_sha256
    pool = bc.build_pool_018(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016, confirmation_017)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
        _relax_float32_identities(patch)
        model = make_fake_model()
        weights = pm.Weights.from_model(model)
        head = ht.HeadWeights.from_model(model)
        lw = lc.LayerWeights.from_model(model)
        heads = ap.HeadSet.from_model(model)
        programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in bc.PROGRAM_LAYERS}
        cache = pm.PromptCache(model, tuple(pool.nouns))
        axes = cs.stage_axes(cache, weights, pool_010)
        plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
        read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
        fake_lock_011 = {"experiment": "011", "axes_vectors": {"T": axes["T"].direction.double().tolist(), "R0": axes["R0"].direction.double().tolist()}, "sigma_T": axes["T"].sigma, "read_weight": read.weight.tolist(),
                         "denominators": {"denominators": {template: read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}, "sigma_r": er.sigma_r_from_pairs(read, weights, pool_010.frames)},
                         "confirmation_011_sha256": confirmation_011.content_sha256, "content_sha256": "f" * 64}
        states = {frame.frame_id: hp.capture_frame_017(model, head, pool.reference_prompt(frame), pool.single_nouns, axes["T"]) for frame in pool.frames}
        bases_012 = lc.template_bases(pool_012.frames, {fid: (s.state.x1, s.state.x2) for fid, s in states.items() if fid in {f.frame_id for f in pool_012.frames}})
        fake_lock_012 = {"experiment": "012", "axes_vectors": fake_lock_011["axes_vectors"], "sigma_T": axes["T"].sigma, "read_weight": read.weight.tolist(),
                         "base_states": lc.bases_to_json(bases_012, {template: len(pool_012.frames_of(template)) for template in pm.TEMPLATE_ORDER}), "defined_templates": list(pm.TEMPLATE_ORDER),
                         "confirmation_012_sha256": confirmation_012.content_sha256, "lock_011_sha256": "f" * 64, "content_sha256": "e" * 64}
        fake_locks = {"013": {"experiment": "013", "locked_states": {fid: atp.locked_state(states[fid].state) for fid in {f.frame_id for f in pool_013.frames}}, "confirmation_013_sha256": confirmation_013.content_sha256, "lock_012_sha256": "e" * 64, "content_sha256": "d" * 64},
                      "014": {"experiment": "014", "locked_states": {fid: atp.locked_state(states[fid].state) for fid in {f.frame_id for f in pool_014.frames}}, "confirmation_014_sha256": confirmation_014.content_sha256, "lock_013_sha256": "d" * 64, "content_sha256": "c" * 64},
                      "015": {"experiment": "015", "locked_states": {fid: atp.locked_state(states[fid].state) for fid in {f.frame_id for f in pool_015.frames}}, "confirmation_015_sha256": confirmation_015.content_sha256, "lock_014_sha256": "c" * 64, "content_sha256": "b" * 64},
                      "016": {"experiment": "016", "locked_states": {fid: atp.locked_state(states[fid].state) for fid in {f.frame_id for f in pool_016.frames}}, "confirmation_016_sha256": confirmation_016.content_sha256, "lock_015_sha256": "b" * 64, "content_sha256": "a" * 64}}
        bases_3, counts_3 = hp.layer3_bases(pool_017.frames, states)
        fake_locks["017"] = {"experiment": "017", "locked_states": {frame.frame_id: hp.locked_state(states[frame.frame_id]) for frame in pool_017.frames}, "bases_3": hp.bases_to_json(bases_3, counts_3),
                             "confirmation_017_sha256": confirmation_017.content_sha256, "lock_016_sha256": "a" * 64, "content_sha256": "9" * 64}
        stage1_digests = {frame.frame_id: ap.state_digest(hp.locked_state(states[frame.frame_id])) for frame in confirmation_017.frames}
        fpm = ap.FrozenPatternModel(read, lw, heads, bases_012)
        fcm = fch.FrameChannelModel(read, lw, {layer: programs[layer] for layer in hp.UPSTREAM_LAYERS}, bases_012, axes["R0"].direction.double())
        hcm = hp.HeadChainModel(fcm, programs[hp.HEAD_LAYER], bases_3, axes["T"].direction.double())
        context = hp.AnalysisContext(hcm, fch.AnalysisContext(fcm, fpm, heads, weights, axes["T"]), weights, axes["T"])
        # The fake's Experiment 017 extract: exactly the 9636 pairs the real record licenses (its keys), recomputed on the fake.
        real_keys = set(json.loads((ROOT / bc.INHERITED_017_EXTRACT_RELATIVE_PATH).read_text())["entries"])
        entries = {}
        for frame in pool.frames:
            state = states[frame.frame_id]
            names = [name for name, _ in pool.tokens if f"{name}|{frame.frame_id}" in real_keys]
            plural_name = pool.plural_cue[frame.template_id]
            plural = hp.measure_pair(model, weights, head, state, plural_name, pool.token_id(plural_name), axes["R0"], axes["T"], pool.single_nouns)
            for name in names:
                record = hp.measure_pair(model, weights, head, state, name, pool.token_id(name), axes["R0"], axes["T"], pool.single_nouns)
                analysis = hp.analyse_pair_017(record, plural, context=context, state=state)
                if analysis is not None:
                    entries[f"{name}|{frame.frame_id}"] = bc.extract_entry_017(analysis)
    assert len(entries) == bc.EXPECTED_EXTRACT_SIZE_017
    extract_text = pm.canonical_json(bc.inherited_extract_payload(entries, source={"path": "fake"}, digests=digests | {"lock_017": "9" * 64}, stage1_state_digests=stage1_digests)) + "\n"
    return manifest, pool, fake_lock_011, fake_lock_012, fake_locks, extract_text


def _relax_float32_identities(patch) -> None:
    """The fake runs block 2 with 2048 neurons in float32: its own forward-pass identities (ρ(Δr_c) against the summed component reads) carry ~1e-4 relative noise, above the real model's tolerance. Only the fake's arithmetic is relaxed; nothing scientific."""
    patch.setattr(ra, "IDENTITY_TOLERANCE", 1e-3)
    patch.setattr(ra, "P1_CROSS_CHECK_TOLERANCE", max(ra.P1_CROSS_CHECK_TOLERANCE, 1e-3))
    patch.setattr(ra, "NEURON_SUM_TOLERANCE", max(ra.NEURON_SUM_TOLERANCE, 1e-3))


@pytest.fixture
def sandbox(tmp_path, monkeypatch, fake_world):
    manifest, pool, fake_lock_011, fake_lock_012, fake_locks, extract_text = fake_world
    _copy_repo_files(tmp_path)
    (tmp_path / bc.INHERITED_017_EXTRACT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / bc.INHERITED_017_EXTRACT_RELATIVE_PATH).write_text(extract_text, encoding="utf-8")
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    _relax_float32_identities(monkeypatch)
    return tmp_path, manifest, pool, fake_lock_011, fake_lock_012, fake_locks


def make_runner(sandbox, monkeypatch, *, logs=None):
    root, manifest, pool, fake_lock_011, fake_lock_012, fake_locks = sandbox
    logs = logs if logs is not None else []
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    runner = runner_module.Runner(root=root, results_path=root / "outputs/experiment-018/results.json", report_path=root / "outputs/experiment-018/report.md",
                                  model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_018(manifest, pool),
                                  lock_011_loader=lambda path: dict(fake_lock_011), lock_012_loader=lambda path: dict(fake_lock_012), lock_013_loader=lambda path: dict(fake_locks["013"]), lock_014_loader=lambda path: dict(fake_locks["014"]),
                                  lock_015_loader=lambda path: dict(fake_locks["015"]), lock_016_loader=lambda path: dict(fake_locks["016"]), lock_017_loader=lambda path: dict(fake_locks["017"]),
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
    pool, pool_010, lock_011, lock_012, lock_017, extract, confirmation, digests = runner._inputs()
    assert len(pool.tokens) == 231 and len(pool.frames) == 78 and len(confirmation.tokens) == 24 and len(confirmation.exposed_frame_prompts) == 24 * 78 and len(confirmation.frames) == 12
    assert runner.explore() == 0, logs[-3:]
    state = bc.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["status"] == "complete" and exploration["replication"]["experiment_017"]["max_abs_deviation"] == 0.0
    assert len(exploration["pairs"]) == bc.EXPECTED_EXTRACT_SIZE_017 and set(exploration["locked_states"]) == {frame.frame_id for frame in pool.frames}
    assert exploration["base2_pt"]["n_frames"] == 26 and len(exploration["base2_pt"]["vector"]) == lw_dim(runner) and set(exploration["subsets"]) == set(bc.RUNGS) and len(exploration["scores"]) == bc.N_NEURONS
    assert exploration["ranking_pool"]["n_pairs"] == 9636 and exploration["ranking_pool"]["n_records"] == 9636 + sum(1 for key in exploration["pairs"] if key.split("|")[1].startswith("coordinated"))
    assert set(exploration["frame_subsets"]) == set(exploration["locked_states"]) and all(len(v) == 256 for v in exploration["frame_subsets"].values()) and all(0 <= v["S256"] <= 256 for v in exploration["frame_overlaps"].values())
    assert all(value < 1e-3 for key, value in exploration["identities"].items()) and exploration["identities"]["I8_reference_rung"] < 1e-9 and exploration["identities"]["head_level1_recovery"] < 1e-9  # the fake's own float32 identities sit at ~1e-4
    assert {"I4_x3", "I5_head_row", "I6_dT", "I7_split", "I8_reference_rung"} <= set(exploration["identities"]) and exploration["exposed_check"]["n_tokens"] == 231
    x = exploration["exposed_check"]["pairs"]
    assert set(x["rungs"]) == {*bc.RUNGS, bc.ORACLE} and x["kappa"]["c_L"]["S2048"] == pytest.approx(1.0) and x["kappa"]["c_L"]["S0"] == pytest.approx(0.0) and x["rungs"]["S2048"]["c_L"]["r2"] is not None
    assert set(exploration["exposed_check"]["split"]["per_template"]) == set(pm.TEMPLATE_ORDER) and "orderings" in exploration["exposed_check"] and "kappa_c_L" in exploration["summary"]
    assert not {prompt.key for prompt in confirmation.all_prompts} & set(state["executed_prompt_keys"])
    with pytest.raises(bc.PhaseError):
        runner.explore()
    with pytest.MonkeyPatch.context() as guard:
        for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, name):
                guard.setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the lock phase ran a forward pass")))
        assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    predictions_path = runner.results_path.parent / "candidate-predictions.md"
    lock = json.loads(candidate.read_text())
    assert lock["experiment"] == "018" and len(lock["predictions"]["rows"]) == 24 * 78 and set(lock["predictions"]["rows"][0]) == set(bc.PREDICTION_COLUMNS) and lock["floors"]["kappa_c_L_floor"] == bc.KAPPA_CL_FLOOR
    assert lock["lock_017_sha256"] == "9" * 64 and lock["subsets"] == exploration["subsets"] and lock["frame_subsets"] == exploration["frame_subsets"] and lock["head"] == "L03.H04" and lock["block"] == 2 and "token_means" in lock["predictions"]
    text = predictions_path.read_text()
    assert "preregistered predictions" in text and "ĉ_L S_256" in text
    with pytest.raises(bc.PhaseError):
        runner.confirm()  # not installed
    shutil.copy(candidate, runner.root / bc.LOCK_RELATIVE_PATH)
    (runner.root / bc.PREDICTIONS_RELATIVE_PATH).write_text(text, encoding="utf-8")
    runner.changed_paths = lambda commit: ["src/neural_decompiler/block_concentration.py"]
    with pytest.raises(bc.PhaseError, match="scientific paths"):
        runner.confirm()
    runner.changed_paths = lambda commit: [bc.LOCK_RELATIVE_PATH, bc.PREDICTIONS_RELATIVE_PATH, f"{bc.EXPERIMENT_DIR}/README.md", "README.md"]
    tampered = json.loads(json.dumps(lock))  # a lock whose S_256 is not the frozen rule's: refused before anything fresh runs, even with a recomputed digest
    tampered["subsets"]["S256"] = sorted((set(tampered["subsets"]["S256"]) - {tampered["subsets"]["S256"][0]}) | {tampered["subsets"]["B256"][0]})
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in tampered.items() if key != "content_sha256"}))
    (runner.root / bc.LOCK_RELATIVE_PATH).write_text(pm.canonical_json(tampered) + "\n", encoding="utf-8")
    with pytest.raises(bc.PhaseError):
        runner.confirm()
    shutil.copy(candidate, runner.root / bc.LOCK_RELATIVE_PATH)
    seen = {}
    original_stage_two = bc.stage_two

    def guarded_stage_two(model, pool_, pool_010_, confirmation_, lock_, lock_012_, stage1, **kwargs):
        seen["digest"] = stage1["digest"]
        bc.assert_stage_one_digest(stage1)
        return original_stage_two(model, pool_, pool_010_, confirmation_, lock_, lock_012_, stage1, **kwargs)

    monkeypatch.setattr(bc, "stage_two", guarded_stage_two)
    assert runner.confirm() == 0, logs[-3:]
    state = bc.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["lock_predictions_reproduced_max_difference"] == 0.0 and state["phases"]["confirm"]["ranking_reproduced_max_difference"] == 0.0
    results = state["confirmation"]
    s1 = results["stage1"]
    assert s1["digest"] == seen["digest"] == bc.stage_digest(s1["rows"], s1["states"], s1["frame_subsets"]) and len(s1["rows"]) == 24 * 12 and set(s1["frame_subsets"]) == set(s1["states"]) and all(len(v) == 256 for v in s1["frame_subsets"].values())
    assert all(set(s) == {"p_c", "p_t", "x1_all", "x2_all", "x3_all"} for s in s1["states"].values()) and all(v["S256"] >= 0 for v in s1["frame_overlaps"].values())
    parts = results["outcome"]["label"].split(" | ")
    assert parts[0] in bc.OUTCOME_Y1 and parts[1] in bc.OUTCOME_Y2 and parts[2] in bc.OUTCOME_Y3 and parts[3] in bc.OUTCOME_Y4
    assert len(results["Y1"]["scored_tokens"]) == 24 and all(entry["n_valid_frames"] == 78 and entry["n_cue_final"] == 52 for entry in results["Y1"]["tokens"].values())
    assert set(results["Y2"]["per_frame"]) <= set(results["valid_frames"]) and results["Y3"]["n_pairs"] == results["Y3"]["n_pairs_Y1"] + results["Y3"]["n_pairs_Y2"] and "per_set" in results["Y3"]
    assert set(results["Y4"]["cue_final"]) <= {f"{t}-018-{i}" for t in pm.CUE_FINAL_TEMPLATES for i in (1, 2, 3, 4)} and "reading" in results["Y3"]
    y1 = results["Y1"]
    assert "statistics" in y1 and y1["statistics"]["kappa"]["c_L"]["S2048"] == pytest.approx(1.0) and y1["precondition"]["reference"]["c_L"] is not None and set(y1["precondition"]["checks"]) >= {"reference_c_L", "gap_c_L", "gap_Pi"}
    assert "split_gap_cue_final" in results["Y2"]["precondition"]["checks"] and "split_guard" in results["Y2"]["test"] and "no_harm_guard" in results["Y2"]["test"]
    assert all(value < 1e-3 for value in results["identities"].values())
    assert {prompt.key for prompt in confirmation.exposed_frame_prompts} <= set(state["executed_prompt_keys"])
    with pytest.raises(bc.PhaseError):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "## Tier A" in report and "stage 1" in report and "stage 2" in report and "κ" in report and "Y3 (single neuron" in report and "Y4 (secondary" in report and "predeclared orderings" in report and "Template quantifier" in report


def lw_dim(runner) -> int:
    model = make_fake_model()
    return int(model.cfg.d_model)


def test_incidents_are_recorded_and_block_reruns(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.freeze_confirmation() == 0
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(hp, "capture_frame_017", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic capture failure")))
        assert runner.explore() == 2
    state = bc.load_results_state(runner.results_path)
    assert state["exploration"]["incidents"][-1]["phase"] == "explore" and state["phases"]["explore"]["status"] == "running"
    with pytest.raises(bc.PhaseError, match="incident"):
        runner.explore()
    runner.git_state = lambda: {"commit": "b" * 40, "dirty": False}
    assert runner.explore() == 0 and runner.lock() == 0
    shutil.copy(runner.results_path.parent / "candidate-lock.json", runner.root / bc.LOCK_RELATIVE_PATH)
    shutil.copy(runner.results_path.parent / "candidate-predictions.md", runner.root / bc.PREDICTIONS_RELATIVE_PATH)
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(hp, "capture_frame_017", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic stage-1 failure")))
        assert runner.confirm() == 2
    state = bc.load_results_state(runner.results_path)
    assert state["confirmation"]["incident"]["phase"] == "confirm" and "stage1" not in state["confirmation"] and state["phases"]["confirm"]["status"] == "running"
    with pytest.raises(bc.PhaseError, match="confirm"):
        runner.confirm()
    assert runner.report() == 0 and "## Incidents" in runner.report_path.read_text()
