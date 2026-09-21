"""Phase isolation, the confirmation freeze, the no-forward-pass lock, the selectors locked from the exposed pool only, the two-stage confirm with its digest barrier (no target pair before it), the oracles only after every measurement, and the full state machine for the Experiment 019 runner on a six-layer fake with a 2048-neuron block 2 (a reduced licensed pool: the fake's Experiment 018 extract holds eight tokens' pairs, so that the phases run in minutes)."""

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
from neural_decompiler import block_routing as br
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
from test_block_routing import toy_tokenizer_019
from test_experiment_018_runner import _relax_float32_identities

ROOT = Path(__file__).parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_019_runner", ROOT / "experiments/019-block2-routing/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8, d_mlp=br.N_NEURONS)


REPO_FILES = (pm.MANIFEST_RELATIVE_PATH, pm.EXTENSION_RELATIVE_PATH, cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, ht.LOCK_RELATIVE_PATH, er.CONFIRMATION_RELATIVE_PATH, er.LOCK_RELATIVE_PATH,
              lc.CONFIRMATION_RELATIVE_PATH, lc.LOCK_RELATIVE_PATH, ap.CONFIRMATION_RELATIVE_PATH, ap.LOCK_RELATIVE_PATH, nf.CONFIRMATION_RELATIVE_PATH, nf.LOCK_RELATIVE_PATH, atp.CONFIRMATION_RELATIVE_PATH, atp.LOCK_RELATIVE_PATH,
              fch.CONFIRMATION_RELATIVE_PATH, fch.LOCK_RELATIVE_PATH, hp.CONFIRMATION_RELATIVE_PATH, hp.LOCK_RELATIVE_PATH, bc.CONFIRMATION_RELATIVE_PATH, bc.LOCK_RELATIVE_PATH)
SMALL_QUOTA = 2  # the fake tokenizer exposes two candidates per class: ten fresh tokens, so that the phases run in minutes


def _copy_repo_files(root: Path) -> None:
    for relative in REPO_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, root / relative)


def small_tokenizer(manifest, pool):
    """toy_tokenizer_019 with every candidate beyond the first two eligible of each class mapped to an exposed token id (ineligible by the builder's rule), and the pool's words at their real ids."""
    base = toy_tokenizer_019(manifest, pool)
    vocabulary = dict(base.vocabulary)
    exposed_ids = {token_id for _, token_id in pool.tokens}
    sink = next(token_id for word, token_id in pool.tokens if pool.token_source.get(word) == "confirmation-018")  # never decoded by the confirmation builder
    for words in br.CANDIDATES.values():
        eligible = 0
        for word in words:
            if vocabulary.get(" " + word) in exposed_ids:
                continue
            eligible += 1
            if eligible > SMALL_QUOTA:
                vocabulary[" " + word] = sink
    return type(base)(vocabulary)


@pytest.fixture(scope="module")
def fake_world(tmp_path_factory):
    """Built once per module: the 019 pool, the fake locks 011–018 and the fake's reduced Experiment 018 extract (eight tokens' pairs recomputed on the fake with Experiment 018's chain)."""
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
    pool_018 = bc.build_pool_018(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016, confirmation_017)
    confirmation_018 = bc.load_confirmation(tmp_path / bc.CONFIRMATION_RELATIVE_PATH, pool_018, digests)
    digests["confirmation_018"] = confirmation_018.content_sha256
    pool = br.build_pool_019(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016, confirmation_017, confirmation_018)
    tokens_017 = [name for name, _ in pool.tokens if pool.token_source.get(name) == "confirmation-017"][:4]
    tokens_018 = [name for name, _ in pool.tokens if pool.token_source.get(name) == "confirmation-018"][:4]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
        _relax_float32_identities(patch)
        model = make_fake_model()
        weights = pm.Weights.from_model(model)
        head = ht.HeadWeights.from_model(model)
        lw = lc.LayerWeights.from_model(model)
        heads = ap.HeadSet.from_model(model)
        programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in br.PROGRAM_LAYERS}
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
        base2_pt, n_pt = bc.block2_base_pt(pool_018.frames, states)
        fpm = ap.FrozenPatternModel(read, lw, heads, bases_012)
        fcm = fch.FrameChannelModel(read, lw, {layer: programs[layer] for layer in hp.UPSTREAM_LAYERS}, bases_012, axes["R0"].direction.double())
        hcm = hp.HeadChainModel(fcm, programs[hp.HEAD_LAYER], bases_3, axes["T"].direction.double())
        hp_context = hp.AnalysisContext(hcm, fch.AnalysisContext(fcm, fpm, heads, weights, axes["T"]), weights, axes["T"])
        read_out = bc.read_of_outputs(read, lw)
        locked_018 = {frame.frame_id: hp.locked_state(states[frame.frame_id]) for frame in pool_018.frames}
        # Experiment 018's ranking on the fake over the reduced explore pool (the four 017 tokens in the 78 frames), pooled over positions; its per-frame lists; the stage-1 lists of the twelve 018 frames.
        unmasked = bc.MaskedChainModel(hcm, base2_pt, {name: torch.ones(bc.N_NEURONS, dtype=torch.float64) for name in bc.RUNGS})
        ranking = [(name, frame.frame_id, frame.template_id, pool.token_id(name)) for frame in pool_018.frames for name in tokens_017]
        scores, frame_scores = bc.pooled_ranking(unmasked, weights, read_out, locked_018, ranking)
        subsets = bc.subsets_from_scores(scores)
        frame_subsets = {fid: bc.frame_subset(s) for fid, s in frame_scores.items()}
        fake_locks["018"] = {"experiment": "018", "locked_states": locked_018, "bases_3": hp.bases_to_json(bases_3, counts_3), "base2_pt": {"vector": base2_pt.tolist(), "n_frames": n_pt, "template": "coordinated-adjective"},
                             "scores": scores.tolist(), "subsets": subsets, "frame_subsets": frame_subsets, "confirmation_018_sha256": confirmation_018.content_sha256, "lock_017_sha256": "9" * 64, "content_sha256": "8" * 64}
        stage1_digests, stage1_lists = {}, {}
        for frame in confirmation_018.frames:
            locked = hp.locked_state(states[frame.frame_id])
            stage1_digests[frame.frame_id] = ap.state_digest(locked)
            stage1_lists[frame.frame_id] = bc.frame_subset(bc.frame_ranking(unmasked, weights, read_out, locked, frame.template_id, [(name, pool.token_id(name)) for name in tokens_017]))
        # The fake's reduced Experiment 018 extract: the eight tokens' pairs recomputed on the fake with Experiment 018's masked chain and its fake subsets.
        masked_018 = bc.MaskedChainModel(hcm, base2_pt, bc.masks_from_subsets(subsets, bc.N_NEURONS))
        context_018 = bc.AnalysisContext(masked_018, hp_context, weights, axes["T"])
        entries = {}
        for frame in pool.frames:
            state = states[frame.frame_id]
            plural_name = pool.plural_cue[frame.template_id]
            plural = hp.measure_pair(model, weights, head, state, plural_name, pool.token_id(plural_name), axes["R0"], axes["T"], pool.single_nouns)
            is_018_frame = pool.frame_origin[frame.frame_id] == "confirmation-018"
            for name in (tokens_018 if is_018_frame else tokens_017 + tokens_018):
                record = hp.measure_pair(model, weights, head, state, name, pool.token_id(name), axes["R0"], axes["T"], pool.single_nouns)
                analysis = bc.analyse_pair_018(record, plural, context=context_018, state=state)
                assert analysis is not None
                subset = "Y2" if is_018_frame else ("Y1" if name in tokens_018 else "explore")
                entries[f"{name}|{frame.frame_id}"] = br.extract_entry_018(analysis, subset)
    assert len(entries) == 4 * 78 + 4 * 78 + 4 * 12
    fake_state = {"phases": {"confirm": {"status": "complete", "confirm_commit": "7" * 40}}, "lock": {"content_sha256": "8" * 64}, "state_sha256": "6" * 64, "run_id": "fake018", "protocol_code_commit": "5" * 40,
                  "exploration": {"pairs": {}}, "confirmation": {"per_frame_exposed": {}, "per_frame_fresh": {}, "stage1": {"state_digests": stage1_digests, "frame_subsets": stage1_lists}}}
    payload = br.build_inherited_extract_018(fake_state, fake_locks["018"], digests=digests | {"lock_017": "9" * 64, "lock_018": "8" * 64}, results_state_path="fake", results_state_file_sha256="4" * 64, lock_path="fake", lock_file_sha256="3" * 64)
    payload["entries"] = {key: dict(value) for key, value in sorted(entries.items())}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return manifest, pool, fake_lock_011, fake_lock_012, fake_locks, pm.canonical_json(payload) + "\n"


@pytest.fixture
def sandbox(tmp_path, monkeypatch, fake_world):
    manifest, pool, fake_lock_011, fake_lock_012, fake_locks, extract_text = fake_world
    _copy_repo_files(tmp_path)
    (tmp_path / br.INHERITED_018_EXTRACT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / br.INHERITED_018_EXTRACT_RELATIVE_PATH).write_text(extract_text, encoding="utf-8")
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    monkeypatch.setattr(br, "EXPECTED_EXTRACT_SIZE_018", 4 * 78 + 4 * 78 + 4 * 12)
    monkeypatch.setattr(br, "EXTRACT_SET_SIZES_018", (4 * 78, 4 * 78, 4 * 12))
    monkeypatch.setattr(br, "MIN_SCORED_TOKENS", 8)
    _relax_float32_identities(monkeypatch)
    return tmp_path, manifest, pool, fake_lock_011, fake_lock_012, fake_locks


def make_runner(sandbox, monkeypatch, *, logs=None):
    root, manifest, pool, fake_lock_011, fake_lock_012, fake_locks = sandbox
    logs = logs if logs is not None else []
    runner = runner_module.Runner(root=root, results_path=root / "outputs/experiment-019/results.json", report_path=root / "outputs/experiment-019/report.md",
                                  model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: small_tokenizer(manifest, pool),
                                  lock_011_loader=lambda path: dict(fake_lock_011), lock_012_loader=lambda path: dict(fake_lock_012), lock_013_loader=lambda path: dict(fake_locks["013"]), lock_014_loader=lambda path: dict(fake_locks["014"]),
                                  lock_015_loader=lambda path: dict(fake_locks["015"]), lock_016_loader=lambda path: dict(fake_locks["016"]), lock_017_loader=lambda path: dict(fake_locks["017"]), lock_018_loader=lambda path: dict(fake_locks["018"]),
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
    pool, pool_010, lock_011, lock_012, lock_018, extract, confirmation, digests = runner._inputs()
    assert len(pool.tokens) == 255 and len(pool.frames) == 90 and len(confirmation.tokens) == 10 and len(confirmation.exposed_frame_prompts) == 10 * 90 and len(confirmation.frames) == 18
    assert runner.explore() == 0, logs[-3:]
    state = br.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["status"] == "complete" and exploration["replication"]["experiment_018"]["max_abs_deviation"] == 0.0 and exploration["i9_replication"] == {"passed": True, "n_frames": 90}
    assert len(exploration["pairs"]) == len(extract["entries"]) and set(exploration["locked_states"]) == {frame.frame_id for frame in pool.frames} and set(exploration["frame_meta"]) == set(exploration["locked_states"])
    sel = exploration["selectors"]
    assert set(sel["lists"]) == {"Sp", "T", "E", "G", "Sp1", "E1", "L", "R"} and set(sel["lists"]["E"]) == set(exploration["locked_states"]) and sel["lists"]["L"]["256"] == lock_018["subsets"]["S256"] and sel["n_records"]["licensed"] == len(extract["entries"])
    assert sel["n_records"]["evaluated"] == sum(exploration["licensed_pool"]["tokens_by_template"][frame.template_id] for frame in pool.frames) and exploration["licensed_pool"]["tokens_by_template"] == {t: 8 for t in pm.TEMPLATE_ORDER}
    assert all(value < 1e-3 for key, value in exploration["identities"].items()) and exploration["identities"]["I8_reference_rung"] < 1e-9 and exploration["identities"]["I10_read_identity"] < 1e-9 and exploration["identities"]["I9_frame_lists"] == 0.0
    x = exploration["exposed_check"]["statistics"]["pairs"]
    assert set(x["rungs"]) == set(br.ALL_RUNGS) and (x["kappa"]["c_L"]["S2048"] is None or x["kappa"]["c_L"]["S2048"] == pytest.approx(1.0)) and "Os64" in x["kappa"]["c_L"]  # on the fake channel D may matter little: κ is then undefined
    first_pair = next(iter(exploration["pairs"].values()))
    assert set(exploration["exposed_check"]["oracle_lists"]) == set(exploration["locked_states"]) and "witness_lists" not in first_pair and "oracle" in first_pair and "witness_overlap" in first_pair and "statistics" not in first_pair
    assert "kappa_witness" in exploration["summary"] and set(exploration["exposed_check"]["membership"]["mean"]) == {"E", "G", "Sp", "T"} and set(exploration["exposed_check"]["membership"]["spearman"]["mean"]) == {"E", "G"}
    assert exploration["exposed_check"]["effect_fidelity"]["n"] == len(exploration["pairs"]) and set(exploration["exposed_check"]["witness_overlap_mean"]) == {"E", "G", "Sp"} and set(exploration["selector_overlaps"]) == set(exploration["locked_states"])
    assert not {prompt.key for prompt in confirmation.all_prompts} & set(state["executed_prompt_keys"])
    with pytest.raises(br.PhaseError):
        runner.explore()
    with pytest.MonkeyPatch.context() as guard:
        for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, name):
                guard.setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the lock phase ran a forward pass")))
        assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    predictions_path = runner.results_path.parent / "candidate-predictions.md"
    lock = json.loads(candidate.read_text())
    assert lock["experiment"] == "019" and len(lock["predictions"]["rows"]) == 10 * 90 and set(lock["predictions"]["rows"][0]) == set(br.PREDICTION_COLUMNS) and lock["floors"] == br.frozen_floors()
    assert lock["lock_018_sha256"] == "8" * 64 and lock["selectors"]["lists"] == exploration["selectors"]["lists"] and lock["confirmation_prompt_keys"] == br.prompt_key_manifest(confirmation) and lock["terminology"] == br.WITNESS_TERMINOLOGY
    text = predictions_path.read_text()
    assert "preregistered predictions" in text and "ĉ_L E_64" in text and "witness" in text
    with pytest.raises(br.PhaseError):
        runner.confirm()  # not installed
    shutil.copy(candidate, runner.root / br.LOCK_RELATIVE_PATH)
    (runner.root / br.PREDICTIONS_RELATIVE_PATH).write_text(text, encoding="utf-8")
    runner.changed_paths = lambda commit: ["src/neural_decompiler/block_routing.py"]
    with pytest.raises(br.PhaseError, match="scientific paths"):
        runner.confirm()
    runner.changed_paths = lambda commit: [br.LOCK_RELATIVE_PATH, br.PREDICTIONS_RELATIVE_PATH, f"{br.EXPERIMENT_DIR}/README.md", "README.md"]
    tampered = json.loads(json.dumps(lock))  # a lock whose E_64 list of one frame is not the frozen rule's: refused before anything fresh runs, even with a recomputed digest
    fid = next(iter(tampered["selectors"]["lists"]["E"]))
    pos = next(iter(tampered["selectors"]["lists"]["E"][fid]))
    tampered["selectors"]["lists"]["E"][fid][pos]["64"] = sorted(set(tampered["selectors"]["lists"]["E"][fid][pos]["64"][1:]) | {2047 if 2047 not in tampered["selectors"]["lists"]["E"][fid][pos]["64"] else 2046})
    tampered["selectors"]["digest"] = br.selectors_digest(tampered["selectors"])
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in tampered.items() if key != "content_sha256"}))
    (runner.root / br.LOCK_RELATIVE_PATH).write_text(pm.canonical_json(tampered) + "\n", encoding="utf-8")
    with pytest.raises(br.PhaseError):
        runner.confirm()
    shutil.copy(candidate, runner.root / br.LOCK_RELATIVE_PATH)
    seen = {"stage": None, "measured_in_stage_1": [], "ledger_at_barrier": None}
    original_stage_one, original_stage_two, original_measure = br.stage_one, br.stage_two, br.measure_pair
    fresh_ids = {token["token_id"] for token in confirmation.tokens}

    def spying_measure(model, weights, head, state, name, token_id, *args, **kwargs):
        if seen["stage"] == 1:
            seen["measured_in_stage_1"].append((state.frame.frame_id, name, token_id))
        return original_measure(model, weights, head, state, name, token_id, *args, **kwargs)

    def guarded_stage_one(*args, **kwargs):
        seen["stage"] = 1
        try:
            return original_stage_one(*args, **kwargs)
        finally:
            seen["stage"] = None

    def guarded_stage_two(model, pool_, pool_010_, confirmation_, lock_, lock_012_, stage1, **kwargs):
        seen["digest"] = stage1["digest"]
        br.assert_stage_one_digest(stage1)
        seen["stage"] = 2
        return original_stage_two(model, pool_, pool_010_, confirmation_, lock_, lock_012_, stage1, **kwargs)

    original_assert = br.assert_no_target_pair_executed

    def spying_assert(state_, confirmation_):
        seen["ledger_at_barrier"] = set(state_["executed_prompt_keys"])
        return original_assert(state_, confirmation_)

    monkeypatch.setattr(br, "measure_pair", spying_measure)
    monkeypatch.setattr(br, "stage_one", guarded_stage_one)
    monkeypatch.setattr(br, "stage_two", guarded_stage_two)
    monkeypatch.setattr(br, "assert_no_target_pair_executed", spying_assert)
    assert runner.confirm() == 0, logs[-3:]
    assert seen["measured_in_stage_1"] and not any(token_id in fresh_ids for _, _, token_id in seen["measured_in_stage_1"])  # stage 1 measured only the plural cue pairs of the fresh frames
    state = br.load_results_state(runner.results_path)
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["lock_predictions_reproduced_max_difference"] == 0.0 and state["phases"]["confirm"]["selectors_reproduced_max_difference"] == 0.0
    results = state["confirmation"]
    s1 = results["stage1"]
    assert s1["digest"] == seen["digest"] == br.stage_digest(s1["rows"], s1["states"], s1["frame_selectors"]) and len(s1["rows"]) == 10 * 18 and set(s1["frame_selectors"]) == set(s1["states"])
    assert all(set(s) == {"p_c", "p_t", "x1_all", "x2_all", "x3_all"} for s in s1["states"].values()) and all(set(v) == {"E", "G", "E1", "scores"} for v in s1["frame_selectors"].values()) and not any(field in s1 for field in br.STAGE1_FORBIDDEN_FIELDS)
    targets = {prompt.key for prompt in confirmation.token_prompts} | {prompt.key for prompt in confirmation.exposed_frame_prompts}
    valid_fresh = {fid for fid, e in s1["frames"].items() if e.get("valid")}
    executed_targets = {prompt.key for prompt in confirmation.token_prompts if prompt.frame.frame_id in valid_fresh} | {prompt.key for prompt in confirmation.exposed_frame_prompts}
    assert seen["ledger_at_barrier"] is not None and not (targets & seen["ledger_at_barrier"]) and executed_targets <= set(state["executed_prompt_keys"])  # zero target pairs at the barrier; every valid frame's after
    assert not ((targets - executed_targets) & set(state["executed_prompt_keys"]))  # an invalid fresh frame's target pairs never run
    parts = results["outcome"]["label"].split(" | ")
    assert parts[0] in br.OUTCOME_Y1 and parts[1] in br.OUTCOME_Y2 and parts[2] in br.OUTCOME_Y3 and parts[3] in br.OUTCOME_Y4 and parts[4] in br.OUTCOME_Y5
    y1 = results["Y1"]
    assert len(y1["scored_tokens"]) == 10 and all(entry["n_valid_frames"] == 90 and entry["n_cue_final"] == 60 for entry in y1["tokens"].values()) and set(results["oracle_lists"]) == {"Y1", "Y2"}
    assert set(results["oracle_lists"]["Y1"]) == {frame.frame_id for frame in pool.frames} and all("witness_lists" in a and "oracle" in a and "witness_overlap" in a for a in results["per_frame_exposed"].values()) and set(results["oracle_scores"]) == {"Y1", "Y2"}
    assert y1["effect_fidelity"]["n"] == y1["n_pairs"] and set(y1["witness_overlap_mean"]) == {"E", "G", "Sp"} and (y1.get("membership", {}).get("spearman") is None or set(y1["membership"]["spearman"]["mean"]) == {"E", "G"})
    assert all(len(a["witness_lists"]["64"]) == 64 for a in results["per_frame_exposed"].values())
    if "statistics" in y1:
        k = y1["statistics"]["pairs"]["kappa"]["c_L"]
        assert (k["S2048"] is None or k["S2048"] == pytest.approx(1.0)) and set(y1["precondition"]["checks"]) == {"scored_tokens", "reference_c_L", "reference_rows", "reference_dT", "gap_c_L", "gap_cue_final", "gap_coordinated"}
        assert set(y1["statistics"]["gains"]["64"]) >= {"E64", "G64", "T64", "L64", "O64", "Os64", "E1"} and "witness" in y1["statistics"]["headroom"]
    if y1["precondition"].get("ok") and results["Y2"]["precondition"].get("ok"):
        assert results["pooled"] is not None and "rho" in results["Y3"] and "values" in results["Y4"] and set(results["Y5"]["per_set"]) == {"Y1", "Y2"} and "pooled" in results["descriptive"]
    assert results["terminology"] == br.WITNESS_TERMINOLOGY and all(value < 1e-3 for value in results["identities"].values())
    with pytest.raises(br.PhaseError):
        runner.confirm()
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "## Tier A" in report and "stage 1" in report and "stage 2" in report and "κ" in report and "Y3 (the operating-point-plus-drive rule" in report and "Y4 (the template-family account" in report and "witness" in report and "Y5 (membership" in report


def test_incidents_are_recorded_and_block_reruns(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    assert runner.freeze_confirmation() == 0
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(hp, "capture_frame_017", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic capture failure")))
        assert runner.explore() == 2
    state = br.load_results_state(runner.results_path)
    assert state["exploration"]["incidents"][-1]["phase"] == "explore" and state["phases"]["explore"]["status"] == "running"
    with pytest.raises(br.PhaseError, match="incident"):
        runner.explore()
    runner.git_state = lambda: {"commit": "b" * 40, "dirty": False}
    assert runner.explore() == 0 and runner.lock() == 0
    shutil.copy(runner.results_path.parent / "candidate-lock.json", runner.root / br.LOCK_RELATIVE_PATH)
    shutil.copy(runner.results_path.parent / "candidate-predictions.md", runner.root / br.PREDICTIONS_RELATIVE_PATH)
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(hp, "capture_frame_017", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic stage-1 failure")))
        assert runner.confirm() == 2
    state = br.load_results_state(runner.results_path)
    assert state["confirmation"]["incident"]["phase"] == "confirm" and "stage1" not in state["confirmation"] and state["phases"]["confirm"]["status"] == "running"
    with pytest.raises(br.PhaseError, match="confirm"):
        runner.confirm()
    assert runner.report() == 0 and "## Incident" in runner.report_path.read_text()
