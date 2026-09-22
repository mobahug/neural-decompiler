"""Phase isolation, the single-forward reference capture, computation-time noun freshness, the no-forward-pass lock,
the two-stage barrier (no S2-TARGET prompt before it) and the full state machine for the Experiment 020 runner on a
six-layer fake, with a reduced exposed pool so that the phases run in seconds."""

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
from neural_decompiler import readout_decompilation as rd
from plural_fakes import TinyPlural
from test_readout_decompilation import toy_tokenizer_020

ROOT = Path(__file__).parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_020_runner", ROOT / "experiments/020-readout-decompilation/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


REPO_FILES = (pm.MANIFEST_RELATIVE_PATH, pm.EXTENSION_RELATIVE_PATH, cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, ht.LOCK_RELATIVE_PATH, er.CONFIRMATION_RELATIVE_PATH,
              er.LOCK_RELATIVE_PATH, lc.CONFIRMATION_RELATIVE_PATH, lc.LOCK_RELATIVE_PATH, ap.CONFIRMATION_RELATIVE_PATH, ap.LOCK_RELATIVE_PATH, nf.CONFIRMATION_RELATIVE_PATH, nf.LOCK_RELATIVE_PATH,
              atp.CONFIRMATION_RELATIVE_PATH, atp.LOCK_RELATIVE_PATH, fch.CONFIRMATION_RELATIVE_PATH, fch.LOCK_RELATIVE_PATH, hp.CONFIRMATION_RELATIVE_PATH, hp.LOCK_RELATIVE_PATH,
              bc.CONFIRMATION_RELATIVE_PATH, bc.LOCK_RELATIVE_PATH, br.CONFIRMATION_RELATIVE_PATH, rd.CONFIRMATION_RELATIVE_PATH)
SMALL_TOKENS = 2  # exposed cues measured per frame at explore: the frozen confirmation file pins the exposed pool,
# so the pool itself is the real one (108 frames, 279 cues) and only the exploration budget is reduced here.


def _copy_repo_files(root: Path) -> None:
    for relative in REPO_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, root / relative)


@pytest.fixture(scope="module")
def fake_world(tmp_path_factory):
    """Built once per module: the reduced pool, the fake's 011/012/017 locks and the frozen 020 confirmation set."""
    tmp_path = tmp_path_factory.mktemp("world")
    _copy_repo_files(tmp_path)
    manifest, manifest_sha256, extension = pm.load_inputs(tmp_path)
    c006 = cd.load_confirmation(tmp_path / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    c009 = ht.load_confirmation(tmp_path / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, c006)
    c011 = er.load_confirmation(tmp_path / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, c006, c009)
    digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": c006.content_sha256, "confirmation_009": c009.content_sha256,
               "confirmation_011": c011.content_sha256}
    pool_012 = lc.build_pool_012(manifest, extension, c006, c009, c011)
    c012 = lc.load_confirmation(tmp_path / lc.CONFIRMATION_RELATIVE_PATH, pool_012, digests)
    digests["confirmation_012"] = c012.content_sha256
    c013 = ap.load_confirmation(tmp_path / ap.CONFIRMATION_RELATIVE_PATH, ap.build_pool_013(manifest, extension, c006, c009, c011, c012), digests)
    digests["confirmation_013"] = c013.content_sha256
    c014 = nf.load_confirmation(tmp_path / nf.CONFIRMATION_RELATIVE_PATH, nf.build_pool_014(manifest, extension, c006, c009, c011, c012, c013), digests)
    digests["confirmation_014"] = c014.content_sha256
    c015 = atp.load_confirmation(tmp_path / atp.CONFIRMATION_RELATIVE_PATH, atp.build_pool_015(manifest, extension, c006, c009, c011, c012, c013, c014), digests)
    digests["confirmation_015"] = c015.content_sha256
    c016 = fch.load_confirmation(tmp_path / fch.CONFIRMATION_RELATIVE_PATH, fch.build_pool_016(manifest, extension, c006, c009, c011, c012, c013, c014, c015), digests)
    digests["confirmation_016"] = c016.content_sha256
    pool_017 = hp.build_pool_017(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016)
    c017 = hp.load_confirmation(tmp_path / hp.CONFIRMATION_RELATIVE_PATH, pool_017, digests)
    digests["confirmation_017"] = c017.content_sha256
    pool_018 = bc.build_pool_018(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017)
    c018 = bc.load_confirmation(tmp_path / bc.CONFIRMATION_RELATIVE_PATH, pool_018, digests)
    digests["confirmation_018"] = c018.content_sha256
    pool_019 = br.build_pool_019(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017, c018)
    c019 = br.load_confirmation(tmp_path / br.CONFIRMATION_RELATIVE_PATH, pool_019, digests)
    digests["confirmation_019"] = c019.content_sha256
    pool = rd.build_pool_020(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017, c018, c019)
    small = pool
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
        model = make_fake_model()
        weights = pm.Weights.from_model(model)
        head = ht.HeadWeights.from_model(model)
        lw = lc.LayerWeights.from_model(model, layers=rd.PROGRAM_LAYERS)
        pool_010 = ra.build_pool_010(manifest, extension, c006, c009)
        cache = pm.PromptCache(model, tuple(small.nouns))
        axes = cs.stage_axes(cache, weights, pool_010)
        plural_ids = {template: small.token_id(name) for template, name in small.plural_cue.items()}
        read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(small.reference_ids), plural_ids)
        fake_lock_011 = {"experiment": "011", "axes_vectors": {"T": axes["T"].direction.double().tolist(), "R0": axes["R0"].direction.double().tolist()}, "sigma_T": axes["T"].sigma,
                         "read_weight": read.weight.tolist(),
                         "denominators": {"denominators": {template: read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}, "sigma_r": er.sigma_r_from_pairs(read, weights, pool_010.frames)},
                         "confirmation_011_sha256": c011.content_sha256, "content_sha256": "f" * 64}
        states = {frame.frame_id: hp.capture_frame_017(model, head, small.reference_prompt(frame), small.single_nouns, axes["T"]) for frame in small.frames}
        bases_012 = lc.template_bases(pool_012.frames, {fid: (s.state.x1, s.state.x2) for fid, s in states.items() if fid in {f.frame_id for f in pool_012.frames}})
        fake_lock_012 = {"experiment": "012", "axes_vectors": fake_lock_011["axes_vectors"], "sigma_T": axes["T"].sigma, "read_weight": read.weight.tolist(),
                         "base_states": lc.bases_to_json(bases_012, {template: len(pool_012.frames_of(template)) for template in pm.TEMPLATE_ORDER}), "defined_templates": list(pm.TEMPLATE_ORDER),
                         "confirmation_012_sha256": c012.content_sha256, "lock_011_sha256": "f" * 64, "content_sha256": "e" * 64}
        bases_3, counts_3 = hp.layer3_bases(pool_017.frames, states)
        fake_lock_017 = {"experiment": "017", "locked_states": {frame_id: hp.locked_state(state) for frame_id, state in states.items()}, "bases_3": hp.bases_to_json(bases_3, counts_3),
                         "confirmation_017_sha256": c017.content_sha256, "lock_016_sha256": "a" * 64, "content_sha256": "9" * 64}
    return manifest, small, digests, fake_lock_011, fake_lock_012, fake_lock_017


def _populate(root: Path, monkeypatch, fake_world):
    """The repo files and the per-test constants, into ``root``."""
    manifest, small, digests, lock_011, lock_012, lock_017 = fake_world
    _copy_repo_files(root)
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    monkeypatch.setattr(rd, "EXPLORE_TOKEN_LIMIT", SMALL_TOKENS)
    monkeypatch.setattr(rd, "MIN_SCORED_TOKENS", 4)
    monkeypatch.setattr(rd, "MIN_VALID_EXPOSED_FRAMES", 3)
    monkeypatch.setattr(rd, "MIN_VALID_FRAMES_PER_TOKEN", 1)
    monkeypatch.setattr(rd, "MIN_VALID_FRESH_FRAMES", 3)
    monkeypatch.setattr(rd, "MIN_VALID_COORDINATED_FRAMES", 1)
    monkeypatch.setattr(rd, "MIN_SCORABLE_FRESH_NOUNS", 4)
    return root, manifest, small, digests, lock_011, lock_012, lock_017


@pytest.fixture
def sandbox(tmp_path, monkeypatch, fake_world):
    """A fresh writable root per test."""
    return _populate(tmp_path, monkeypatch, fake_world)


def make_runner(sandbox, monkeypatch, *, logs=None, small_confirmation=True):
    root, manifest, small, digests, lock_011, lock_012, lock_017 = sandbox
    logs = logs if logs is not None else []
    runner = runner_module.Runner(root=root, results_path=root / "outputs/experiment-020/results.json", report_path=root / "outputs/experiment-020/report.md",
                                  model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_020(manifest, small),
                                  lock_011_loader=lambda path: dict(lock_011), lock_012_loader=lambda path: dict(lock_012), lock_017_loader=lambda path: dict(lock_017),
                                  git_state=lambda: {"commit": "a" * 40, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True, changed_paths=lambda commit: [],
                                  contract_runner=lambda: {"passed": True, "returncode": 0, "output_tail": "", "output_sha256": "x"}, log=logs.append)
    return runner, logs


def pool_frame_id(lock):
    return sorted(lock["locked_states"])[0]


def test_parser_has_exactly_the_six_phases_and_no_overrides():
    parser = runner_module.build_parser()
    for phase in ("validate", "freeze-confirmation", "explore", "lock", "confirm", "report"):
        assert parser.parse_args([phase]).phase == phase
    for forbidden in (["calibrate"], ["confirm", "--stage", "2"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden)


class ExecutionSpy:
    """Counts actual forward executions by prompt key (multiplicity, not membership)."""

    def __init__(self, monkeypatch):
        self.counts: dict[str, int] = {}
        original = pm.capture_prompt

        def spy(model, prompt, sites):
            self.counts[prompt.key] = self.counts.get(prompt.key, 0) + 1
            return original(model, prompt, sites)

        for module in (pm, ra.pm, rd.pm):
            monkeypatch.setattr(module, "capture_prompt", spy)

    def total(self) -> int:
        return sum(self.counts.values())


def test_one_reference_execution_per_frame(sandbox, monkeypatch):
    """The S1-REF prompt of a frame is executed exactly once for the complete 017+020 reference state."""
    root, manifest, small, digests, lock_011, lock_012, lock_017 = sandbox
    model = make_fake_model()
    head = ht.HeadWeights.from_model(model)
    weights = pm.Weights.from_model(model)
    nouns = rd.NounSet.build(weights, small.nouns, [])
    axis_T = pm.SiteAxis("T", torch.zeros(int(model.cfg.d_model), dtype=torch.float64), torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64), float(lock_011["sigma_T"]))
    frame = small.frames[0]
    reference = small.reference_prompt(frame)
    executed: list[str] = []
    original = pm.capture_prompt

    def spy(model_, prompt, sites):
        executed.append(prompt.key)
        return original(model_, prompt, sites)

    monkeypatch.setattr(pm, "capture_prompt", spy)
    monkeypatch.setattr(ra.pm, "capture_prompt", spy)
    state = rd.capture_frame_020(model, head, reference, nouns, axis_T)
    assert executed.count(reference.key) == 1 and len(executed) == 1
    assert len(state.x4_all) == frame.p_t + 1 and len(state.x5_all) == frame.p_t + 1 and state.h6.shape == state.x5_all[0].shape
    assert state.rows4.shape[0] == int(model.cfg.n_heads) and state.rows5.shape[1] == frame.p_t + 1
    assert state.c_ref.shape[0] == len(nouns.nouns) and len(state.x3_all) == frame.p_t + 1


def test_explore_never_touches_a_fresh_noun(sandbox, monkeypatch):
    """Computation-time freshness: no fresh noun id reaches a contrast, and the guard refuses a set that carries one."""
    root, manifest, small, digests, lock_011, lock_012, lock_017 = sandbox
    confirmation = rd.load_confirmation(root / rd.CONFIRMATION_RELATIVE_PATH, small, dict(digests) | {"confirmation_019": digests["confirmation_019"]}) if False else None
    payload = json.loads((root / rd.CONFIRMATION_RELATIVE_PATH).read_text())
    fresh_ids = {token for entry in payload["nouns"] for token in (*entry["sg_ids"], *entry["pl_ids"])}
    fresh_keys = {entry["lexical_key"] for entry in payload["nouns"]}
    weights = pm.Weights.from_model(make_fake_model())
    exposed_only = rd.NounSet.build(weights, small.nouns, [])
    assert not ({noun.lexical_key for noun in exposed_only.nouns} & fresh_keys)
    assert not ({token for noun in exposed_only.nouns for token in (*noun.sg_ids, *noun.pl_ids)} & fresh_ids)
    fake_confirmation = type("C", (), {"nouns": tuple(pm.Noun(entry["lexical_key"], type(small.nouns[0].split)(entry["split"]), entry["rule_class"], tuple(entry["sg_ids"]), tuple(entry["pl_ids"]))
                                                     for entry in payload["nouns"])})()
    rd.assert_explore_nouns(exposed_only, fake_confirmation)
    with_fresh = rd.NounSet.build(weights, small.nouns, list(fake_confirmation.nouns[:1]))
    with pytest.raises(rd.PhaseError, match="fresh nouns"):
        rd.assert_explore_nouns(with_fresh, fake_confirmation)
    sneaked = rd.NounSet.build(weights, list(small.nouns) + list(fake_confirmation.nouns[:1]), [])
    with pytest.raises(rd.PhaseError, match="reaches the fresh noun"):
        rd.assert_explore_nouns(sneaked, fake_confirmation)


def test_stage_one_executes_each_manifest_prompt_exactly_once_and_uses_the_frozen_validity_rule(sandbox, monkeypatch):
    """Multiplicity, not membership: one S1-REF and one S1-VALIDITY execution per fresh frame, no S2-TARGET, and the
    validity verdict is the inherited head-informative rule with the frozen cue-effect count."""
    root, manifest, small, digests, lock_011, lock_012, lock_017 = sandbox
    confirmation = rd.load_confirmation(root / rd.CONFIRMATION_RELATIVE_PATH, small, dict(digests) | {"confirmation_020": json.loads((root / rd.CONFIRMATION_RELATIVE_PATH).read_text())["content_sha256"]})
    model = make_fake_model()
    lock = {"content_sha256": "0" * 64, "template_bases": {}, "locked_states": {}}
    weights = pm.Weights.from_model(model)
    lw = lc.LayerWeights.from_model(model, layers=rd.PROGRAM_LAYERS)
    programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in rd.PROGRAM_LAYERS}
    states = {frame.frame_id: hp.capture_frame_017(model, ht.HeadWeights.from_model(model), small.reference_prompt(frame), small.single_nouns,
                                                   pm.SiteAxis("T", torch.zeros(int(model.cfg.d_model), dtype=torch.float64), torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64), float(lock_011["sigma_T"])))
              for frame in small.frames[:1]}
    bases = rd.template_bases_020({}, []) if False else {}
    for template in pm.TEMPLATE_ORDER:  # the comparator's bases; their values do not matter to this test
        bases[template] = {"4": [0.0] * int(model.cfg.d_model), "5": [0.0] * int(model.cfg.d_model)}
    lock["template_bases"] = bases
    spy = ExecutionSpy(monkeypatch)
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    record = rd.stage_one(model, small, confirmation, lock, lock_011, lock_012, lock_017, protocol_code_commit="a" * 40)
    classes = rd.manifest_classes(confirmation)
    for key in classes["S1-REF"] + classes["S1-VALIDITY"]:
        assert spy.counts.get(key) == 1, (key, spy.counts.get(key))
    assert not (set(spy.counts) & set(classes["S2-TARGET"])) and spy.total() == len(classes["S1-REF"]) + len(classes["S1-VALIDITY"])
    for frame_id, entry in record["frames"].items():
        assert set(entry) >= {"valid", "head_informative", "plural_head_change", "cue_effect_positive", "cue_effect_required"}
        assert entry["cue_effect_required"] == pm.exact_count_floor(rd.FRAME_CUE_EFFECT_RATE, len([n for n in small.nouns if n.single_token]))
        assert entry["valid"] == bool(entry["head_informative"] and entry["cue_effect_positive"] >= entry["cue_effect_required"])
    # the frozen head-informative rule is a real gate: a floor above every measured head change invalidates every frame
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 1e6)
    strict = rd.stage_one(model, small, confirmation, lock, lock_011, lock_012, lock_017, protocol_code_commit="a" * 40)
    assert not any(entry["valid"] for entry in strict["frames"].values()) and all(not entry["head_informative"] for entry in strict["frames"].values())
    assert all("dc_fresh_nouns" in row and len(row["dc_fresh_nouns"]) == len(confirmation.nouns) for row in record["rows"])


def test_full_state_machine_lock_without_forward_pass_and_stage_barrier(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    root = runner.root
    assert runner.validate() == 0, logs[-2:]  # the confirmation set is frozen and committed in the repository
    assert runner.explore() == 0, logs[-3:]
    state = rd.load_results_state(runner.results_path)
    exploration = state["exploration"]
    assert state["phases"]["explore"]["status"] == "complete" and exploration["n_pairs"] > 0
    assert exploration["noun_population"]["non_scorable"] == ["peach"] and exploration["noun_population"]["exposed_scorable"] == len(exploration["statistics"]["per_cue"]) * 0 + exploration["statistics"]["n_nouns"]
    assert set(exploration["comparators"]) == {"ceiling_measured_dx3", "no_l5_heads", "template_base_mlps", "dT_only", "standing"}
    assert exploration["comparators"]["standing"]["no_l5_heads"] == "nested simplification comparator"
    assert all(value < 1.0 for key, value in exploration["identities"].items())
    confirmation = rd.load_confirmation(root / rd.CONFIRMATION_RELATIVE_PATH, sandbox[2], dict(sandbox[3]) | {"confirmation_020": json.loads((root / rd.CONFIRMATION_RELATIVE_PATH).read_text())["content_sha256"]})
    assert not {prompt.key for prompt in confirmation.all_prompts} & set(state["executed_prompt_keys"])
    with pytest.raises(rd.PhaseError):
        runner.explore()  # once
    with pytest.MonkeyPatch.context() as guard:
        for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, name):
                guard.setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the lock phase ran a forward pass")))
        assert runner.lock() == 0
    candidate = runner.results_path.parent / "candidate-lock.json"
    lock = json.loads(candidate.read_text())
    assert lock["experiment"] == "020" and lock["floors"] == rd.frozen_floors() and lock["populations"] == {name: dict(value) for name, value in rd.POPULATIONS.items()}
    assert len(lock["predictions"]["rows"]) == len(confirmation.tokens) * len(sandbox[2].frames) and lock["confirmation_prompt_manifest"] == rd.manifest_classes(confirmation)
    assert set(lock["noun_keys"]) and "peach" not in lock["noun_keys"] and lock["fresh_noun_keys"] == [noun.lexical_key for noun in confirmation.nouns]
    # (1) the lock carries a committed prediction for BOTH populations; no row may fall back to exposed values for Y3
    assert lock["n_predicted_nouns"] == {"exposed_scorable": len(lock["noun_keys"]), "fresh": len(confirmation.nouns)}
    for row in lock["predictions"]["rows"]:
        assert len(row["dc"]) == len(lock["noun_keys"]) and len(row["dc_fresh_nouns"]) == len(confirmation.nouns)
        assert row["dc"][: len(confirmation.nouns)] != row["dc_fresh_nouns"]  # the fresh predictions are their own, not the first 24 exposed ones
    # (4) the rank-1 comparator's two frozen objects are in the lock
    assert len(lock["rank1"]["noun_vector"]) == len(lock["noun_keys"]) and len(lock["rank1"]["d_noun"]) == len(lock["locked_states"][pool_frame_id(lock)]["h6"])
    assert lock["rank1"]["exposed_fit_r2"] is not None
    # (5) the lock phase proved its own provenance invariant
    assert rd.load_results_state(runner.results_path)["lock"]["provenance_difference"] <= rd.PROVENANCE_TOLERANCE
    text = (runner.results_path.parent / "candidate-predictions.md").read_text()
    assert "preregistered predictions" in text and "Y1 table" in text
    with pytest.raises(rd.PhaseError):
        runner.confirm()  # not installed
    shutil.copy(candidate, root / rd.LOCK_RELATIVE_PATH)
    (root / rd.PREDICTIONS_RELATIVE_PATH).write_text(text, encoding="utf-8")
    runner.changed_paths = lambda commit: ["src/neural_decompiler/readout_decompilation.py"]
    with pytest.raises(rd.PhaseError, match="scientific paths"):
        runner.confirm()
    runner.changed_paths = lambda commit: [rd.LOCK_RELATIVE_PATH, rd.PREDICTIONS_RELATIVE_PATH, f"{rd.EXPERIMENT_DIR}/README.md"]
    seen = {"stage": None, "stage1_tokens": [], "ledger_at_barrier": None}
    original_stage_one, original_stage_two, original_measure = rd.stage_one, rd.stage_two, rd.measure_pair
    target_ids = {int(token["token_id"]) for token in confirmation.tokens}

    def spying_measure(model, state_, nouns_, word, token_id, **kwargs):
        if seen["stage"] == 1:
            seen["stage1_tokens"].append((state_.frame.frame_id, word, token_id))
        return original_measure(model, state_, nouns_, word, token_id, **kwargs)

    def guarded_stage_one(*args, **kwargs):
        seen["stage"] = 1
        try:
            return original_stage_one(*args, **kwargs)
        finally:
            seen["stage"] = None

    def guarded_stage_two(model, pool_, confirmation_, lock_, l11, l12, l17, stage1, **kwargs):
        rd.assert_stage_one_digest(stage1)
        seen["digest"] = stage1["digest"]
        seen["stage"] = 2
        return original_stage_two(model, pool_, confirmation_, lock_, l11, l12, l17, stage1, **kwargs)

    original_assert = rd.assert_no_target_prompt_executed

    def spying_assert(state_, confirmation_):
        seen["ledger_at_barrier"] = set(state_["executed_prompt_keys"])
        return original_assert(state_, confirmation_)

    monkeypatch.setattr(rd, "measure_pair", spying_measure)
    monkeypatch.setattr(rd, "stage_one", guarded_stage_one)
    monkeypatch.setattr(rd, "stage_two", guarded_stage_two)
    monkeypatch.setattr(rd, "assert_no_target_prompt_executed", spying_assert)
    assert runner.confirm() == 0, logs[-3:]
    assert seen["stage1_tokens"] and not any(token_id in target_ids for _, _, token_id in seen["stage1_tokens"])  # stage 1 measured only the frozen validity cue pairs
    classes = rd.manifest_classes(confirmation)
    assert seen["ledger_at_barrier"] is not None and not (set(classes["S2-TARGET"]) & seen["ledger_at_barrier"])  # zero target prompts at the barrier
    state = rd.load_results_state(runner.results_path)
    results = state["confirmation"]
    assert state["phases"]["confirm"]["status"] == "complete" and state["phases"]["confirm"]["lock_predictions_reproduced_max_difference"] == 0.0
    # (6) the Y2 table was recomputed from the digested states before any target prompt
    assert state["phases"]["confirm"]["stage1_rows_reproduced_max_difference"] == 0.0
    # (5) every frozen identity that a measured pair can carry has a recorded value
    assert {"readout", "logit", "level1", "additive"} <= set(results["identities"]) and set(results["identities"]) <= set(rd.IDENTITY_NAMES)
    assert {"readout", "logit", "level1", "additive", "inherited_017"} <= set(results["stage1"]["identities"])
    assert results["identities"]["additive"] < rd.ADDITIVE_IDENTITY_TOLERANCE  # it runs on the stage-2 target pairs, whose states come from the lock and the digest
    # (4) the rank-1 comparator is scored as the frozen rule, not as the primary program
    for name in ("Y1", "Y2"):
        if name in results["comparators"]:
            assert results["comparators"][name]["rank1_nouns"]["standing"] == "descriptive comparator"
    y3_direct = results["Y3"].get("nouns")
    if y3_direct and results["comparators"].get("Y1", {}).get("rank1_nouns", {}).get("fresh_noun_r2") is not None:
        assert results["comparators"]["Y1"]["rank1_nouns"]["fresh_noun_r2"] != pytest.approx(1.0)  # it is a different predictor
    s1 = results["stage1"]
    assert s1["digest"] == seen["digest"] == rd.stage_digest(s1["rows"], s1["states"], s1["frames"]) and len(s1["rows"]) == len(confirmation.tokens) * len(confirmation.frames)
    parts = results["outcome"]["label"].split(" | ")
    assert parts[0] in rd.OUTCOME_Y1 and parts[1] in rd.OUTCOME_Y2 and parts[2] in rd.OUTCOME_Y3
    assert results["Y1"]["population"]["nouns"] == "exposed_scorable" and results["Y3"]["population"]["nouns"] == "fresh"
    assert set(results["noun_keys"]) == set(lock["noun_keys"]) and results["fresh_noun_keys"] == [noun.lexical_key for noun in confirmation.nouns]
    if results["Y3"].get("nouns"):
        assert set(results["Y3"]["nouns"]) == set(results["fresh_noun_keys"])  # Y3 scores the fresh nouns only
    assert results["joint_fresh_nouns"] is None or results["joint_fresh_nouns"]["n_nouns"] == len(confirmation.nouns)
    invalid = [frame_id for frame_id, entry in s1["frames"].items() if not entry["valid"]]
    executed = set(state["executed_prompt_keys"])
    for frame in confirmation.frames:
        keys = {pm.Prompt(frame, int(token["token_id"]), token["word"]).key for token in confirmation.tokens}
        assert (keys <= executed) is (frame.frame_id not in invalid)  # an invalid frame's target prompts never run
    with pytest.raises(rd.PhaseError):
        runner.confirm()  # once
    assert runner.report() == 0
    report = runner.report_path.read_text()
    assert "## Tier A" in report and "stage 1" in report and "Interpretation limit" in report and "conditional on each fresh frame's stage-1 reference state" in report


def test_stage_two_refuses_a_committed_row_without_a_fresh_noun_prediction(sandbox, monkeypatch):
    """The removed fallback: a row that carries no dc_fresh_nouns has no committed Y3 prediction, and stage 2 refuses
    rather than scoring the fresh nouns against exposed-noun predictions."""
    root, manifest, small, digests, lock_011, lock_012, lock_017 = sandbox
    confirmation = rd.load_confirmation(root / rd.CONFIRMATION_RELATIVE_PATH, small, dict(digests) | {"confirmation_020": json.loads((root / rd.CONFIRMATION_RELATIVE_PATH).read_text())["content_sha256"]})
    model = make_fake_model()
    frame = small.frames[0]
    weights = pm.Weights.from_model(model)
    nouns = rd.NounSet.build(weights, small.nouns, confirmation.nouns)
    program = rd.ReadoutProgram.from_model(model)
    state = rd.capture_frame_020(model, ht.HeadWeights.from_model(model), small.reference_prompt(frame), nouns,
                                 pm.SiteAxis("T", torch.zeros(int(model.cfg.d_model), dtype=torch.float64), torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64), float(lock_011["sigma_T"])))
    exposed_keys = [nouns.nouns[i].lexical_key for i in nouns.exposed_scorable]
    row = {"token": confirmation.tokens[0]["word"], "frame_id": frame.frame_id, "template": frame.template_id,
           "dc": [0.0] * len(exposed_keys), "dc_no_l5_heads": [0.0] * len(exposed_keys), "dc_template_base": [0.0] * len(exposed_keys), "dT": 0.0}
    lock = {"content_sha256": "0" * 64, "template_bases": {}, "locked_states": {frame.frame_id: rd.locked_state(state)}, "predictions": {"rows": [row]}}
    stage1 = {"frames": {}, "states": {}, "rows": []}
    small_one = cs.Pool008((frame,), small.frame_origin, small.tokens, small.token_category, small.token_source, small.nouns, small.noun_source, small.reference_ids, small.plural_cue)
    one_token = type("C", (), {k: getattr(confirmation, k) for k in ("reference_ids", "frames", "exposed_frame_ids", "nouns", "token_prompts", "exposed_frame_prompts", "content_sha256")})()
    one_token.tokens = (confirmation.tokens[0],)
    with pytest.raises(rd.PhaseError, match="no fresh-noun prediction"):
        rd.stage_two(model, small_one, one_token, lock, lock_011, lock_012, lock_017, stage1)


def test_incidents_are_recorded_and_block_reruns(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(rd, "capture_frame_020", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic capture failure")))
        assert runner.explore() == 2
    state = rd.load_results_state(runner.results_path)
    assert state["exploration"]["incidents"][-1]["phase"] == "explore" and state["phases"]["explore"]["status"] == "running"
    with pytest.raises(rd.PhaseError, match="incident"):
        runner.explore()
    runner.git_state = lambda: {"commit": "b" * 40, "dirty": False}
    with pytest.MonkeyPatch.context() as guard:  # a committed fix lets the phase start again; a fresh incident stops it at once
        guard.setattr(rd, "capture_frame_020", lambda *args, **kwargs: (_ for _ in ()).throw(pm.IncidentError("synthetic second failure")))
        assert runner.explore() == 2
    state = rd.load_results_state(runner.results_path)
    assert state["phases"]["explore"]["attempts"] == 2 and state["phases"]["explore"]["attempt_commits"] == ["a" * 40, "b" * 40]
    assert state["exploration"]["incidents"][-1]["commit"] == "b" * 40


def test_an_identity_beyond_its_tolerance_is_an_incident_but_an_invalid_frame_is_not(sandbox, monkeypatch):
    runner, logs = make_runner(sandbox, monkeypatch)
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(rd, "LEVEL1_TOLERANCE", 0.0)  # the float32 captures cannot reproduce the chain exactly: the identity must stop the phase
        assert runner.explore() == 2
    state = rd.load_results_state(runner.results_path)
    assert "level1 failed" in state["exploration"]["incidents"][-1]["message"] and "frozen tolerance" in state["exploration"]["incidents"][-1]["message"]
    # an invalid fresh frame is a scientific fact: the scoring records it and the phase continues
    stage1 = {"frames": {"f1": {"valid": False, "template_id": "cardinal"}, "f2": {"valid": True, "template_id": "cardinal"}}, "rows": [], "states": {}}
    scored = rd.score_confirmation(stage1, {"Y1": {"exposed": None, "fresh": None}, "Y2": {"exposed": None, "fresh": None}}, {"dT_only_noun_vector": None, "rank1": None})
    assert scored["Y1"]["label"] == "PRECONDITION_FAILED_TOKENS" and scored["valid_frames"] == ["f2"] and "incident" not in scored
