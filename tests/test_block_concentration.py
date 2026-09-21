"""Experiment 018: the masked chain against Experiment 017's rungs on the fake (I8), the ranking's independence from measurements and from the confirmation set, the subsets and controls contract, the leak-proof boundary, the closure fraction κ unclipped with its evaluability rule, the scoring with the preconditions, the split and no-harm guards, Y3 pooled and Y4 per frame on synthetic tables, the confirmation policy and the extract."""

from __future__ import annotations

import json
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
from test_head_pattern import toy_tokenizer_017

ROOT = Path(__file__).resolve().parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8, d_mlp=bc.N_NEURONS)


def test_frozen_constants_match_the_design():
    assert bc.KAPPA_CL_FLOOR == 0.70 and bc.KAPPA_PI_FLOOR == 0.50 and bc.SPLIT_GUARD == {"cue_final": 0.60, "coordinated": 0.40} and bc.NO_HARM_MARGIN == 0.05 and bc.GAP_MIN == 0.05
    assert bc.PRECONDITION_REFERENCE == {"c_L": 0.98, "rows": 0.95, "dT": 0.95} and bc.Y3_KAPPA_MAX == 0.50 and bc.Y3_MARGIN == 0.25 and bc.Y4_FROZEN_MAX == 0.90 and bc.Y4_REFERENCE_MIN == 0.95 and bc.Y4_MIN_EVALUABLE == 6
    assert bc.ROW_DIFFUSE_MAX == 0.50 and bc.RANDOM_MARGIN == 0.30 and bc.SUBSET_SIZES == (0, 1, 4, 16, 64, 256, 1024, 2048) and bc.HYPOTHESIS_SIZE == 256 and bc.N_RANDOM_CONTROLS == 3 and bc.BOTTOM_SIZE == 256 and bc.N_NEURONS == 2048
    assert bc.RUNGS == ("S0", "S1", "S4", "S16", "S64", "S256", "S1024", "S2048", "R1", "R2", "R3", "B256") and bc.HYPOTHESIS == "S256" and bc.REFERENCE == "S2048" and bc.TEMPLATE_BASE == "S0" and bc.SINGLE == "S1"
    assert bc.MIN_VALID_CUE_FINAL_FRAMES == 6 and bc.MIN_VALID_COORDINATED_FRAMES == 3 and bc.MIN_VALID_FRAMES_PER_TOKEN == 3 and bc.MIN_SCORED_TOKENS == 16 and bc.I8_TOLERANCE == 1e-9 and bc.REPLICATION_TOLERANCE == 1e-6
    assert bc.RUNTIME_SEED == 20260916 and bc.CONTROL_SEED == 20260924 and len(bc.FRESH_FRAMES) == 12 and bc.EXPECTED_EXTRACT_SIZE_017 == 9636 and sum(bc.QUOTAS.values()) == 24 and bc.FRAME_ID_TAG == "018"
    assert [t for t, _ in bc.FRESH_FRAMES].count("cardinal") == 4 and [t for t, _ in bc.FRESH_FRAMES].count("quantifier") == 4 and [t for t, _ in bc.FRESH_FRAMES].count("coordinated-adjective") == 4
    texts_017 = {text for _, text in hp.FRESH_FRAMES}
    assert not ({text for _, text in bc.FRESH_FRAMES} & texts_017) and bc.CANDIDATES is hp.CANDIDATES
    assert bc.OUTCOME_Y1[0] == "CHANNEL_D_CONCENTRATED_TOKENS" and bc.OUTCOME_Y2[0] == "CHANNEL_D_CONCENTRATED_FRAMES_CONDITIONAL" and bc.OUTCOME_Y3 == ("SINGLE_NEURON_REJECTED", "SINGLE_NEURON_NOT_REJECTED", "SINGLE_NEURON_NOT_EVALUABLE")
    assert bc.OUTCOME_Y4 == ("PATTERN_TERM_NEEDED_WITHIN_FRAMES", "PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES", "PATTERN_TERM_NOT_EVALUABLE_WITHIN_FRAMES")
    assert bc.PREDICTION_COLUMNS[:5] == ("token", "frame_id", "template", "p_c", "p_t") and {"row_S256", "c_L_S256", "F_S2048", "Pi_S1", "dT_R3", "row_B256", "c_L_oracle", "dT_frozen", "c_L_level0F"} <= set(bc.PREDICTION_COLUMNS)
    assert bc.SCIENTIFIC_PATH_PREFIXES[:3] == ("src/", "experiments/018-block2-concentration/", "experiments/017-transport-head-pattern/") and bc.HEAD_KEY == "L03.H04" and bc.BLOCK == 2
    assert set(bc.frozen_floors()) >= {"kappa_c_L_floor", "kappa_Pi_floor", "split_guard", "no_harm_margin", "gap_min", "precondition_reference", "y3_kappa_max", "y3_margin", "y4_frozen_max", "y4_reference_min", "hypothesis_size", "i8_tolerance"}


def test_kappa_is_unclipped_and_undefined_below_the_gap():
    assert bc.kappa(0.9, 0.7, 1.0) == pytest.approx(2.0 / 3.0)
    assert bc.kappa(0.6, 0.7, 1.0) == pytest.approx(-1.0 / 3.0)  # worse than the template base: negative, reported as computed
    assert bc.kappa(1.0, 0.7, 0.97) == pytest.approx(1.0 / 0.9)  # better than the full channel: above one
    assert bc.kappa(0.99, 0.96, 1.0) is None and bc.kappa(0.98, 0.95, 1.0) == pytest.approx(0.6)  # the gap must reach 0.05 (0.05 itself is evaluable)
    assert bc.kappa(None, 0.7, 1.0) is None and bc.kappa(0.9, None, 1.0) is None and bc.kappa(0.9, 0.7, None) is None


def test_subsets_controls_overlaps_and_the_lock_check():
    scores = torch.linspace(0.0, 1.0, bc.N_NEURONS, dtype=torch.float64) * 1e-3
    scores[1987] = 5.0
    scores[10] = 3.0
    scores[11] = 3.0  # a tie: the lower index ranks first
    subsets = bc.subsets_from_scores(scores)
    assert set(subsets) == set(bc.RUNGS) and subsets["S0"] == [] and subsets["S1"] == [1987] and set(subsets["S4"]) == {1987, 10, 11, 2047} and len(subsets["S2048"]) == bc.N_NEURONS
    order = bc.order_of(scores)
    assert order[:3] == [1987, 10, 11] and all(len(subsets[f"S{n}"]) == n for n in bc.SUBSET_SIZES) and set(subsets["S16"]) <= set(subsets["S64"]) <= set(subsets["S256"]) <= set(subsets["S1024"])
    for name in bc.RANDOM_RUNGS:
        assert len(subsets[name]) == 256 == len(set(subsets[name])) and subsets[name] == sorted(subsets[name])
    assert subsets["R1"] != subsets["R2"] != subsets["R3"] and subsets["B256"] == sorted(order[-256:]) and 1987 not in subsets["B256"]
    assert bc.subsets_from_scores(scores) == subsets  # reproduced exactly from the seed
    overlaps = bc.subset_overlaps(subsets)
    assert set(overlaps) == {"S256&R1", "S256&R2", "S256&R3", "S256&B256", "R1&R2", "R1&R3", "R1&B256", "R2&R3", "R2&B256", "R3&B256"} and overlaps["S256&B256"] == 0
    assert all(0 <= overlaps[f"S256&{name}"] <= 256 for name in bc.RANDOM_RUNGS) and sum(overlaps[f"S256&{name}"] for name in bc.RANDOM_RUNGS) > 0  # overlap is expected and recorded, never redrawn
    bc.check_subsets(subsets, scores)
    tampered = dict(subsets)
    tampered["R2"] = sorted(set(subsets["R2"]) - set(subsets["S256"]) | {2047})[:256]
    with pytest.raises(bc.PhaseError, match="frozen rule"):
        bc.check_subsets(tampered, scores)
    with pytest.raises(bc.PhaseError):
        bc.check_subsets({**subsets, "S256": subsets["S256"][:255] + [subsets["S256"][0]]}, scores)
    mask = bc.mask_of(subsets["S4"])
    assert mask.sum() == 4 and mask[1987] == 1.0 and mask[0] == 0.0 and bc.mask_of([]).sum() == 0
    assert bc.frame_subset(scores, 3) == sorted([1987, 10, 11]) and bc.overlap_with([1, 2, 3], [3, 4]) == 1


def test_extract_replication_check():
    entry = {"F": 0.7, "Pi": 0.01, "dT": 0.71, "c_L": 0.1, "row": [0.1, -0.1], "row_level0": [0.1, -0.1], "F_hat": 0.7, "Pi_hat": 0.01, "dT_hat": 0.71, "c_L_level0D": 0.1}
    assert bc.check_extract_replication({"a|f": entry}, {"a|f": entry})["passed"]
    with pytest.raises(pm.IncidentError, match="c_L_level0D"):
        bc.check_extract_replication({"a|f": {**entry, "c_L_level0D": 0.1 + 2e-6}}, {"a|f": entry})
    with pytest.raises(pm.IncidentError, match="row_level0"):
        bc.check_extract_replication({"a|f": {**entry, "row_level0": [0.1, -0.1 + 2e-6]}}, {"a|f": entry})
    with pytest.raises(pm.IncidentError, match="lack"):
        bc.check_extract_replication({}, {"a|f": entry})
    mapped = bc.extract_entry_017({"F": 0.7, "Pi": 0.01, "dT": 0.71, "c_L": 0.1, "row": [0.1, -0.1], "prediction": {"row_level0": [0.1, -0.1], "F_hat": 0.7, "Pi_hat": 0.01, "dT_hat": 0.71, "c_L_level0D": 0.1}})
    assert mapped == entry


def test_outcome_labels():
    base = {"precondition_Y1": {"passed": True}, "precondition_Y2": {"passed": True}, "Y1": {"test": {"passed": True}}, "Y2": {"test": {"passed": False}}, "Y3": {"evaluable": True, "rejected": True}, "Y4": {"evaluable": True, "passed": True}}
    assert bc.outcome(base)["label"] == "CHANNEL_D_CONCENTRATED_TOKENS | CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL | SINGLE_NEURON_REJECTED | PATTERN_TERM_NEEDED_WITHIN_FRAMES"
    assert bc.outcome({**base, "Y3": {"evaluable": False, "rejected": False}})["Y3"] == "SINGLE_NEURON_NOT_EVALUABLE" and bc.outcome({**base, "precondition_Y2": {"passed": False}})["Y2"] == "PRECONDITION_FAILED_FRAMES"
    assert bc.outcome({**base, "Y3": {"evaluable": True, "rejected": False}})["Y3"] == "SINGLE_NEURON_NOT_REJECTED" and bc.outcome({**base, "Y4": {"evaluable": True, "passed": False}})["Y4"] == "PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES"
    assert bc.outcome({**base, "Y4": {"evaluable": False, "passed": False}})["Y4"] == "PATTERN_TERM_NOT_EVALUABLE_WITHIN_FRAMES"


def test_y4_per_frame_boundaries():
    frames = {f"c{i}": {"valid": True, "cue_final": True} for i in range(8)} | {"k1": {"valid": True, "cue_final": False}, "c9": {"valid": False, "cue_final": True}}
    entry = lambda ref, frozen, cue_final=True: {"n_pairs": 24, "template": "cardinal" if cue_final else "coordinated-adjective", "cue_final": cue_final, "dT": {bc.REFERENCE: ref}, "dT_frozen": frozen}  # noqa: E731
    per_frame = {f"c{i}": entry(0.99, 0.5) for i in range(8)} | {"k1": entry(0.99, 0.2, cue_final=False), "c9": entry(0.99, 0.99)}
    y4 = bc.score_y4(per_frame, frames)
    assert y4["evaluable"] and y4["passed"] and y4["n_evaluable"] == 8 and set(y4["coordinated"]) == {"k1"} and "c9" not in y4["cue_final"]  # the invalid frame never enters
    at_boundary = dict(per_frame)
    at_boundary["c3"] = entry(0.99, 0.90)  # exactly at the ceiling: not below → the frame is named and Y4 is not established
    y4 = bc.score_y4(at_boundary, frames)
    assert y4["evaluable"] and not y4["passed"] and y4["frames_at_or_above"] == ["c3"]
    below = dict(per_frame)
    below["c3"] = entry(0.99, 0.8999)
    assert bc.score_y4(below, frames)["passed"]
    excluded = dict(per_frame)
    excluded["c4"] = entry(0.94, 0.99)  # the reference rung fails in this frame: the frame is not interpreted, and the seven others carry the label
    y4 = bc.score_y4(excluded, frames)
    assert y4["passed"] and y4["n_evaluable"] == 7 and not y4["cue_final"]["c4"]["evaluable"] and "c4" not in y4["frames_at_or_above"]
    for i in (4, 5, 6):
        excluded[f"c{i}"] = entry(0.94, 0.99)
    y4 = bc.score_y4(excluded, frames)
    assert not y4["evaluable"] and y4["n_evaluable"] == 5 and not y4["passed"]


@pytest.fixture(scope="module")
def inputs():
    manifest, manifest_sha256, extension = pm.load_inputs(ROOT)
    confirmation_006 = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(ROOT / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    confirmation_011 = er.load_confirmation(ROOT / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": confirmation_006.content_sha256, "confirmation_009": confirmation_009.content_sha256, "confirmation_011": confirmation_011.content_sha256}
    pool_012 = lc.build_pool_012(manifest, extension, confirmation_006, confirmation_009, confirmation_011)
    confirmation_012 = lc.load_confirmation(ROOT / lc.CONFIRMATION_RELATIVE_PATH, pool_012, digests)
    digests["confirmation_012"] = confirmation_012.content_sha256
    digests["lock_012"] = json.loads((ROOT / lc.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    pool_013 = ap.build_pool_013(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012)
    confirmation_013 = ap.load_confirmation(ROOT / ap.CONFIRMATION_RELATIVE_PATH, pool_013, digests)
    digests["confirmation_013"] = confirmation_013.content_sha256
    digests["lock_013"] = json.loads((ROOT / ap.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    pool_014 = nf.build_pool_014(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013)
    confirmation_014 = nf.load_confirmation(ROOT / nf.CONFIRMATION_RELATIVE_PATH, pool_014, digests)
    digests["confirmation_014"] = confirmation_014.content_sha256
    digests["lock_014"] = json.loads((ROOT / nf.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    pool_015 = atp.build_pool_015(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014)
    confirmation_015 = atp.load_confirmation(ROOT / atp.CONFIRMATION_RELATIVE_PATH, pool_015, digests)
    digests["confirmation_015"] = confirmation_015.content_sha256
    digests["lock_015"] = json.loads((ROOT / atp.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    pool_016 = fch.build_pool_016(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015)
    confirmation_016 = fch.load_confirmation(ROOT / fch.CONFIRMATION_RELATIVE_PATH, pool_016, digests)
    digests["confirmation_016"] = confirmation_016.content_sha256
    digests["lock_016"] = json.loads((ROOT / fch.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    pool_017 = hp.build_pool_017(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016)
    confirmation_017 = hp.load_confirmation(ROOT / hp.CONFIRMATION_RELATIVE_PATH, pool_017, digests)
    digests["confirmation_017"] = confirmation_017.content_sha256
    digests["lock_017"] = json.loads((ROOT / hp.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    pool = bc.build_pool_018(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016, confirmation_017)
    return manifest, digests, pool, confirmation_017


def toy_tokenizer_018(manifest, pool):
    base = toy_tokenizer_017(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = []
    for _, text in bc.FRESH_FRAMES:
        for index, piece in enumerate(text.replace("{cue}", "").split()):
            words.append(piece if index == 0 else " " + piece)
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    for word, token_id in pool.tokens:  # Experiment 017's confirmed words carry their real ids in the pool: the toy tokenizer must map them to those ids so the builder skips them
        if pool.token_source.get(word) == "confirmation-017":
            vocabulary[" " + word] = token_id
    known = set(vocabulary.values())
    for frame in pool.frames:
        for token_id in (*frame.prefix_ids, *frame.suffix_ids, *frame.cue_ids.values()):
            if token_id not in known:
                vocabulary[f"⟨{token_id}⟩"] = token_id
                known.add(token_id)
    return type(base)(vocabulary)


def test_pool_018_and_committed_extract(inputs):
    manifest, digests, pool, confirmation_017 = inputs
    assert len(pool.tokens) == 231 and len(pool.frames) == 78 and pool.token_source["further"] == "confirmation-017" and pool.frame_origin["coordinated-adjective-017-4"] == "confirmation-017"
    assert sum(1 for frame in pool.frames if frame.p_t == frame.p_c) == 52 and all(frame.p_t == frame.p_c + 1 for frame in pool.frames_of("coordinated-adjective"))
    extract = bc.load_inherited_extract(ROOT / bc.INHERITED_017_EXTRACT_RELATIVE_PATH, digests=digests, expected_size=bc.EXPECTED_EXTRACT_SIZE_017)
    keys = set(extract["entries"])
    assert "further|cardinal-017-1" in keys and "heavy|cardinal-1" in keys and "further|cardinal-1" in keys and "cardinal:sg|cardinal-1" not in keys and "cardinal:sg|quantifier-1" in keys
    assert set(extract["stage1_state_digests"]) == {f"{t}-017-{i}" for t in pm.TEMPLATE_ORDER for i in (1, 2, 3, 4)} and set(next(iter(extract["entries"].values()))) == {*bc.EXTRACT_SCALARS, *bc.EXTRACT_ROWS}
    with pytest.raises(ValueError):
        bc.load_inherited_extract(ROOT / bc.INHERITED_017_EXTRACT_RELATIVE_PATH, digests=digests, expected_size=10)
    ranking = bc.ranking_pool(extract["entries"], pool)
    assert len(ranking) == 9636 and ranking[0][0] < ranking[-1][0] and all(len(entry) == 4 for entry in ranking)
    by_template = bc.tokens_by_template(ranking)
    assert set(by_template) == set(pm.TEMPLATE_ORDER) and "cardinal:sg" not in dict(by_template["cardinal"]) and "cardinal:sg" in dict(by_template["quantifier"]) and "further" in dict(by_template["cardinal"])
    # The triangular licensing: 85 × 30 + 24 × 6 + 24 × 42 + 24 × 48 + 24 × 54 + 24 × 66 + 24 × 78 + 10 + 20.
    counts = {}
    for word, *_ in ranking:
        counts[word] = counts.get(word, 0) + 1
    assert sorted(counts.values())[-24:] == [78] * 24 and sum(counts.values()) == 9636 and sum(1 for v in counts.values() if v == 30) == 85 and counts["cardinal:sg"] == 10 and counts["quantifier:sg"] == 20
    words_018 = {word for words in bc.CANDIDATES.values() for word in words} - {word for word, _ in pool.tokens}
    assert not ({word for word, *_ in ranking} & words_018)  # no candidate of the fresh set is in the ranking population


def test_confirmation_policy_with_twelve_frames(inputs, tmp_path):
    manifest, digests, pool, confirmation_017 = inputs
    tokenizer = toy_tokenizer_018(manifest, pool)
    payload = bc.build_confirmation_payload(tokenizer, pool, digests)
    words = [token["word"] for token in payload["tokens"]]
    assert len(words) == 24 and words[:5] == ["primary", "secondary", "principal", "opposite", "distinct"] and words[5:9] == ["duo", "solo", "triplet", "tens"] and words[9:14] == ["maximal", "greater", "massive", "immense", "modest"]
    assert words[14:18] == ["ye", "him", "me", "it"] and words[-6:] == ["fancy", "sweet", "bitter", "spicy", "ripe", "raw"]
    assert not (set(words) & {token["word"] for token in confirmation_017.tokens}) and len(payload["token_prompts"]) == 24 * 12 and len(payload["exposed_frame_prompts"]) == 24 * 78
    confirmation = bc.validate_confirmation(payload, pool, digests)
    assert len(confirmation.all_prompts) == 24 + 288 + 1872 and [frame.frame_id for frame in confirmation.frames][:4] == ["cardinal-018-1", "cardinal-018-2", "cardinal-018-3", "cardinal-018-4"]
    assert all(frame.p_t == frame.p_c + 1 for frame in confirmation.frames if frame.template_id == "coordinated-adjective") and all(frame.p_t == frame.p_c for frame in confirmation.frames if frame.template_id != "coordinated-adjective")
    path = tmp_path / "c.json"
    assert bc.freeze_confirmation(path, tokenizer, pool, digests) == payload["content_sha256"]
    with pytest.raises(FileExistsError):
        bc.freeze_confirmation(path, tokenizer, pool, digests)
    tampered = dict(payload)
    tampered["tokens"] = list(reversed(payload["tokens"]))
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in tampered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError):
        bc.validate_confirmation(tampered, pool, digests)


def _fake_setup(pool):
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in bc.PROGRAM_LAYERS}
    small = cs.Pool008(pool.frames[:2] + pool.frames_of("quantifier")[:1] + pool.frames_of("coordinated-adjective")[:2], pool.frame_origin, pool.tokens[:6], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    cache = pm.PromptCache(model, tuple(small.nouns))
    axes = cs.stage_axes(cache, weights, small)
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
    states = {frame.frame_id: hp.capture_frame_017(model, head, small.reference_prompt(frame), small.single_nouns, axes["T"]) for frame in small.frames}
    bases = lc.template_bases(small.frames, {fid: (s.state.x1, s.state.x2) for fid, s in states.items()})
    bases_3, counts = hp.layer3_bases(small.frames, states)
    base2_pt, n_pt = bc.block2_base_pt(small.frames, states)
    fpm = ap.FrozenPatternModel(read, lw, heads, bases)
    fcm = fch.FrameChannelModel(read, lw, {layer: programs[layer] for layer in hp.UPSTREAM_LAYERS}, bases, axes["R0"].direction.double())
    hcm = hp.HeadChainModel(fcm, programs[hp.HEAD_LAYER], bases_3, axes["T"].direction.double())
    hp_context = hp.AnalysisContext(hcm, fch.AnalysisContext(fcm, fpm, heads, weights, axes["T"]), weights, axes["T"])
    return model, weights, head, lw, programs, small, axes, plural_ids, read, states, bases, bases_3, base2_pt, n_pt, hcm, hp_context


def test_masked_chain_reproduces_017_ranking_independence_and_boundary_on_the_fake(inputs, monkeypatch):
    manifest, digests, pool, confirmation_017 = inputs
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model, weights, head, lw, programs, small, axes, plural_ids, read, states, bases, bases_3, base2_pt, n_pt, hcm, hp_context = _fake_setup(pool)
    assert n_pt == 2 and base2_pt.shape == (lw.W_in[2].shape[0],)
    read_out = bc.read_of_outputs(read, lw)
    assert read_out.shape == (bc.N_NEURONS,)
    # The ranking from the locked states, the weights and ΔE only — the same records give the pooled and the per-frame scores.
    locked = {fid: hp.locked_state(s) for fid, s in states.items()}
    unmasked = bc.MaskedChainModel(hcm, base2_pt, {name: torch.ones(bc.N_NEURONS, dtype=torch.float64) for name in bc.RUNGS})
    ranking = [(name, frame.frame_id, frame.template_id, token_id) for frame in small.frames for name, token_id in small.tokens if token_id not in (pool.reference_ids[frame.template_id], plural_ids[frame.template_id])]
    scores, frame_scores = bc.pooled_ranking(unmasked, weights, read_out, locked, ranking)
    assert scores.shape == (bc.N_NEURONS,) and float(scores.min()) >= 0.0 and float(scores.max()) > 0.0 and set(frame_scores) == set(locked)
    with pytest.MonkeyPatch.context() as guard:
        for attr in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, attr):
                guard.setattr(pm, attr, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the ranking touched the network")))
        again, _ = bc.pooled_ranking(unmasked, weights, read_out, locked, ranking)
    assert torch.equal(scores, again)
    frame_id = small.frames[0].frame_id
    own = bc.frame_ranking(unmasked, weights, read_out, locked[frame_id], small.frames[0].template_id, [(n, t) for n, f, _, t in ranking if f == frame_id])
    assert torch.allclose(own, frame_scores[frame_id])
    # The ranking population is the record's KEYS: scrambling every recorded value, or handing over a poisoned confirmation set, changes nothing.
    entries = {f"{name}|{fid}": {"F": 1e9, "Pi": -1e9, "dT": 0.0, "c_L": 1e9, "row": [1e9], "row_level0": [1e9], "F_hat": 1e9, "Pi_hat": 1e9, "dT_hat": 1e9, "c_L_level0D": 1e9} for name, fid, _, _ in ranking}
    pool_pairs = bc.ranking_pool(entries, pool)
    assert [(w, f) for w, f, _, _ in pool_pairs] == sorted((w, f) for w, f, _, _ in ranking) and set(bc.tokens_by_template(pool_pairs)) <= set(pm.TEMPLATE_ORDER)
    scrambled, _ = bc.pooled_ranking(unmasked, weights, read_out, locked, pool_pairs)
    firing = bc.FiringAccumulator(read_out)
    with_firing, _ = bc.pooled_ranking(unmasked, weights, read_out, locked, pool_pairs, firing)
    assert torch.equal(scrambled, with_firing) and bc.subsets_from_scores(scrambled) == bc.subsets_from_scores(bc.pooled_ranking(unmasked, weights, read_out, locked, sorted(ranking, key=lambda r: (r[0], r[1])))[0])
    summary = firing.summary(bc.subsets_from_scores(scores)["S256"])
    assert set(summary) <= set(pm.TEMPLATE_ORDER) and all(len(v["mean_effect"]) == 256 and 0 <= v["n_firing_in_half"] <= 256 and len(v["top"]) == bc.FIRING_TOP for v in summary.values())
    subsets = bc.subsets_from_scores(scores)
    masked = bc.MaskedChainModel(hcm, base2_pt, bc.masks_from_subsets(subsets, bc.N_NEURONS))
    context = bc.AnalysisContext(masked, hp_context, weights, axes["T"])
    for frame in (small.frames[0], small.frames_of("coordinated-adjective")[0]):
        state = states[frame.frame_id]
        template = frame.template_id
        name, token_id = next((n, t) for n, t in small.tokens if t not in (pool.reference_ids[template], plural_ids[template]))
        plural = hp.measure_pair(model, weights, head, state, pool.plural_cue[template], plural_ids[template], axes["R0"], axes["T"], small.single_nouns)
        record = hp.measure_pair(model, weights, head, state, name, token_id, axes["R0"], axes["T"], small.single_nouns)
        oracle = bc.mask_of(bc.frame_subset(frame_scores[frame.frame_id]))
        analysis = bc.analyse_pair_018(record, plural, context=context, state=state, oracle_mask=oracle)
        assert analysis is not None and analysis["cue_final"] == (state.p_t == state.p_c)
        ids = analysis["identities"]
        assert ids["I8_reference_rung"] < 1e-9 and ids["I4_x3"] < 1e-5 and ids["I7_split"] < 1e-5 and ids["head_level1_recovery"] < 1e-9
        p, p17 = analysis["prediction"], analysis["analysis_017"]["prediction"]
        assert set(p) == set(bc.PREDICTION_COLUMNS[3:]) and p["dT_S2048"] == pytest.approx(p17["dT_hat"], abs=1e-12) and p["c_L_S2048"] == pytest.approx(p17["c_L_level0D"], abs=1e-12) and p["dT_frozen"] == p["F_S2048"] and p["c_L_level0F"] == pytest.approx(p17["c_L_level0F"])
        assert p["row_S2048"] == pytest.approx(p17["row_level0"], abs=1e-12) and p["Pi_S2048"] == pytest.approx(p17["Pi_hat"], abs=1e-12)
        for rung in (*bc.RUNGS, bc.ORACLE):
            assert len(p[f"row_{rung}"]) == state.p_t + 1 and sum(p[f"row_{rung}"]) == pytest.approx(0.0, abs=1e-9) and p[f"dT_{rung}"] == pytest.approx(p[f"F_{rung}"] + p[f"Pi_{rung}"], abs=1e-9)
        if state.p_t == state.p_c:
            assert p["dT_S0"] == pytest.approx(p17["dT_no_D"], abs=1e-12) and p["row_S0"] == pytest.approx(p17["row_no_D"], abs=1e-12)  # S_0 at p_c is Experiment 017's −D rung
        else:
            assert p["dT_S0"] != pytest.approx(p17["dT_no_D"], abs=1e-9)  # at p_t the −D rung kept the frame's operating point; S_0 sits at the locked p_t base
        assert p["dT_S0"] != pytest.approx(p["dT_S2048"], abs=1e-9) and p["c_L_S1"] != pytest.approx(p["c_L_S0"], abs=1e-12)  # the subsets are real reductions
        # The masks compose: the reference rung with every neuron own equals channel D whatever the subset; a mask equal to S_2048's is S_2048.
        up = masked.upstream_parts(weights, atp.reference_rows(hcm.fcm.programs, state.x1_all[: state.p_c + 1], state.x2_all[: state.p_c + 1]), state.x1_all, state.x2_all, state.p_c, state.p_t, token_id, template)
        assert set(up.parts) == set(state.positions) and up.effects(state.p_c).shape == (bc.N_NEURONS,) and float(up.effects(state.p_c).abs().max()) > 0.0
        # The boundary: a poisoned patched capture at both positions leaves every column of every rung unchanged (the identities notice it; the predictions never look).
        poisoned = {key: value * 3.0 + 1.0 for key, value in record.extra.items()}
        fields = {f.name: getattr(record, f.name) for f in record.__dataclass_fields__.values()}
        for key, value in list(fields.items()):
            if isinstance(value, float):
                fields[key] = value * 3.0 + 1.0
            elif isinstance(value, dict) and value and all(isinstance(v, float) for v in value.values()):
                fields[key] = {k: v * 3.0 + 1.0 for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                fields[key] = value * 3.0 + 1.0
        fields["extra"] = poisoned
        poisoned_record = ra.Attribution(**fields)
        with pytest.raises(pm.IncidentError):
            bc.analyse_pair_018(poisoned_record, plural, context=context, state=state, oracle_mask=oracle)
        with pytest.MonkeyPatch.context() as guard:
            for module, names in ((hp, ("X3_IDENTITY_TOLERANCE", "ROW_IDENTITY_TOLERANCE", "DT_IDENTITY_TOLERANCE", "SPLIT_IDENTITY_TOLERANCE")), (atp, ("ROW_IDENTITY_TOLERANCE", "CHAIN_IDENTITY_TOLERANCE")),
                                  (ap, ("OV_IDENTITY_TOLERANCE",)), (lc, ("LADDER_IDENTITY_TOLERANCE", "NEURON_IDENTITY_TOLERANCE")), (ra, ("IDENTITY_TOLERANCE", "P1_CROSS_CHECK_TOLERANCE", "NEURON_SUM_TOLERANCE"))):
                for attr in names:
                    guard.setattr(module, attr, float("inf"))
            poisoned_analysis = bc.analyse_pair_018(poisoned_record, plural, context=context, state=state, oracle_mask=oracle)
        assert poisoned_analysis is not None and atp._max_numeric_difference(poisoned_analysis["prediction"], p, "poisoned") == 0.0 and poisoned_analysis["dT"] != analysis["dT"]
        # The table is computed with every capture and intervention entry point disabled.
        with pytest.MonkeyPatch.context() as guard:
            for attr in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
                if hasattr(pm, attr):
                    guard.setattr(pm, attr, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a prediction touched the network")))
            table = bc.prediction_table(masked, weights, locked, {fid: bc.frame_subset(frame_scores[fid]) for fid in locked}, small.frames, [{"word": name, "token_id": token_id}], list(pm.TEMPLATE_ORDER))
        mine = next(row for row in table if row["frame_id"] == frame.frame_id)
        assert len(table) == len(small.frames) and set(mine) == set(bc.PREDICTION_COLUMNS) and atp._max_numeric_difference(mine, bc.prediction_row(name, frame.frame_id, template, p), "row") < 1e-12
        means = bc.token_means_from_table(table, [name])
        assert means[name]["n_frames"] == len(small.frames) and means[name]["n_cue_final"] == 3
        stats = bc.statistics_for([analysis], [p], [name])
        assert set(stats["pairs"]["rungs"]) == {*bc.RUNGS, bc.ORACLE} and frame.frame_id in stats["per_frame"]
    # The masked model from a source record (the exploration record at lock time, the lock at confirm time) reproduces the in-process model.
    source = {"axes_vectors": {"T": axes["T"].direction.double().tolist(), "R0": axes["R0"].direction.double().tolist()}, "read_weight": read.weight.tolist(), "sigma_T": axes["T"].sigma, "defined_templates": list(pm.TEMPLATE_ORDER),
              "bases_3": hp.bases_to_json(bases_3, {t: 1 for t in bases_3}), "base2_pt": {"vector": base2_pt.tolist(), "n_frames": n_pt, "template": "coordinated-adjective"}, "subsets": subsets}
    lock_012 = {"base_states": lc.bases_to_json(bases, {template: len(small.frames_of(template)) for template in pm.TEMPLATE_ORDER}), "defined_templates": list(pm.TEMPLATE_ORDER), "axes_vectors": source["axes_vectors"], "read_weight": read.weight.tolist(), "sigma_T": axes["T"].sigma}
    rebuilt = bc.masked_model_from_source(source, lock_012, lw, programs, pool)
    frame = small.frames[0]
    name, token_id = small.tokens[1]
    a = rebuilt.predict_from_locked(weights, locked[frame.frame_id], token_id, frame.template_id)
    b = masked.predict_from_locked(weights, locked[frame.frame_id], token_id, frame.template_id)
    assert atp._max_numeric_difference(a, b, "rebuilt") < 1e-9


# ---------------------------------------------------------------------------
# Scoring on synthetic tables: predictions of every rung are the measured value plus (1 − α) times a fixed noise, so that κ = 1 − (1 − α)² exactly.


def _noise(seed: int, length: int, scale: float) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(length, generator=generator, dtype=torch.float64)
    return (scale * (raw - raw.mean())).tolist()


DEFAULT_ALPHA = {"S0": 0.0, "S1": 0.2, "S4": 0.3, "S16": 0.4, "S64": 0.5, "S256": 0.6, "S1024": 0.8, "S2048": 0.9, "R1": 0.1, "R2": 0.1, "R3": 0.1, "B256": 0.05, "oracle": 1.0}
ROW_ALPHA = {**DEFAULT_ALPHA, "S256": 0.2, "S1024": 0.4}
K = lambda a: (1.0 - (1.0 - a) ** 2) / (1.0 - (1.0 - DEFAULT_ALPHA["S2048"]) ** 2)  # noqa: E731  # the closure fraction of a rung with blend α against the reference rung's own blend


class _Confirmation:
    def __init__(self, fresh_frames, tokens):
        self.frames = [pm.Frame(t, fid, (1, 2), () if t in pm.CUE_FINAL_TEMPLATES else (7,), {"sg": 5, "pl": 6}, "x {cue}" + ("" if t in pm.CUE_FINAL_TEMPLATES else " y")) for fid, t in fresh_frames]
        self.tokens = [{"word": w, "category": "adjective", "licensed_frames": [fid for fid, _ in fresh_frames]} for w in tokens]


def _synthetic(alpha=None, row_alpha=None, *, frame_alpha=None, frame_row_alpha=None, pi_scale=None, noise=0.55, valid=None):
    """Measured analyses and the two tables. ``alpha[rung]`` blends the rung's scalar predictions toward the measured values; ``frame_alpha[(frame, rung)]`` overrides per frame; ``pi_scale[frame]`` scales the measured pattern term."""
    alpha = {**DEFAULT_ALPHA, **(alpha or {})}
    row_alpha = {**ROW_ALPHA, **(row_alpha or {})}
    frame_alpha = frame_alpha or {}
    frame_row_alpha = frame_row_alpha or {}
    pi_scale = pi_scale or {}
    exposed = [(f"e{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(6)]
    fresh = [(f"n{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(12)]
    words = [f"w{i}" for i in range(18)]
    p_c = 3
    p_t_of = lambda t: p_c if t in pm.CUE_FINAL_TEMPLATES else p_c + 1  # noqa: E731
    measured, rows_locked, rows_stage1 = {}, [], []
    for frames, sink in ((exposed, rows_locked), (fresh, rows_stage1)):
        for fid, t in frames:
            for i, w in enumerate(words):
                seed = 1000 * i + 17 * int(fid[1:]) + (0 if fid[0] == "e" else 500)
                g = torch.Generator().manual_seed(seed)
                r = lambda: float(torch.randn(1, generator=g, dtype=torch.float64))  # noqa: E731
                c_L, F = 0.3 * r() + 0.1, 0.6 * r() + 0.7
                Pi = pi_scale.get(fid, 1.0) * 0.5 * r()
                row = _noise(seed + 1, p_t_of(t) + 1, 0.1)
                n = {"c_L": noise * 0.3 * r(), "F": noise * 0.6 * r(), "Pi": noise * 0.5 * r()}
                n_row = _noise(seed + 2, p_t_of(t) + 1, noise * 0.1)
                measured[f"{w}|{fid}"] = {"token": w, "frame_id": fid, "template": t, "p_c": p_c, "p_t": p_t_of(t), "cue_final": p_c == p_t_of(t), "row": row, "F": F, "Pi": Pi, "dT": F + Pi, "c_L": c_L}
                pred = {"p_c": p_c, "p_t": p_t_of(t)}
                for rung in (*bc.RUNGS, bc.ORACLE):
                    a = frame_alpha.get((fid, rung), alpha[rung])
                    ra_ = frame_row_alpha.get((fid, rung), row_alpha[rung])
                    pred[f"c_L_{rung}"] = c_L + (1 - a) * n["c_L"]
                    pred[f"F_{rung}"] = F + (1 - a) * n["F"]
                    pred[f"Pi_{rung}"] = Pi + (1 - a) * n["Pi"]
                    pred[f"dT_{rung}"] = pred[f"F_{rung}"] + pred[f"Pi_{rung}"]
                    pred[f"row_{rung}"] = [x + (1 - ra_) * e for x, e in zip(row, n_row)]
                pred["dT_frozen"] = pred[f"F_{bc.REFERENCE}"]
                pred["c_L_level0F"] = pred["c_L_S0"]
                sink.append(bc.prediction_row(w, fid, t, pred))
    lock = {"predictions": {"rows": rows_locked}}
    stage1 = {"rows": rows_stage1, "frames": {fid: {"template_id": t, "valid": True if valid is None else fid in valid, "template_defined": True, "cue_final": p_t_of(t) == p_c} for fid, t in fresh}}
    pairs_exposed = {k: v for k, v in measured.items() if k.split("|")[1].startswith("e")}
    pairs_fresh = {k: v for k, v in measured.items() if k.split("|")[1].startswith("n") and (valid is None or k.split("|")[1] in valid)}
    return _Confirmation(fresh, words), lock, stage1, pairs_exposed, pairs_fresh


def test_scoring_floors_guards_preconditions_y3_and_y4_on_synthetic_tables():
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic()
    results = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    y1, y2 = results["Y1"], results["Y2"]
    k1 = y1["statistics"]["kappa"]
    assert k1["c_L"]["S256"] == pytest.approx(K(0.6), abs=1e-9) and k1["c_L"]["S1"] == pytest.approx(K(0.2), abs=1e-9) and k1["c_L"]["S0"] == pytest.approx(0.0, abs=1e-12) and k1["c_L"]["S2048"] == pytest.approx(1.0, abs=1e-12)
    assert k1["c_L"]["oracle"] > 1.0 and k1["rows"]["S256"] == pytest.approx(K(0.2), abs=1e-9) and k1["Pi"]["S256"] == pytest.approx(K(0.6), abs=1e-9)  # unclipped: the oracle exceeds the full channel
    assert y1["test"]["passed"] and y2["test"]["passed"] and results["precondition_Y1"]["passed"] and results["precondition_Y2"]["passed"] and "split_guard" not in y1["test"]
    assert y2["test"]["split_guard"]["cue_final"]["passed"] and y2["test"]["split_guard"]["coordinated"]["passed"] and y2["test"]["no_harm_guard"]["passed"] and len(y2["test"]["no_harm_guard"]["per_frame"]) == 12
    assert y1["precondition"]["reference"]["c_L"] > 0.98 and y1["precondition"]["gaps"]["c_L"] >= 0.05 and set(y2["precondition"]["checks"]) >= {"reference_c_L", "reference_rows", "reference_dT", "gap_c_L", "gap_Pi", "split_gap_cue_final", "split_gap_coordinated"}
    assert results["Y3"]["evaluable"] and results["Y3"]["rejected"] and results["Y3"]["kappa_c_L_single"] == pytest.approx(K(0.2), abs=1e-9) and results["Y3"]["gap"] == pytest.approx(K(0.6) - K(0.2), abs=1e-9) and results["Y3"]["n_pairs"] == 18 * 18
    assert results["Y3"]["per_set"]["Y1"]["kappa_c_L"] == pytest.approx(K(0.2), abs=1e-9) and results["Y3"]["per_set"]["Y2"]["gap_to_hypothesis"] == pytest.approx(K(0.6) - K(0.2), abs=1e-9)
    assert results["Y4"]["evaluable"] and results["Y4"]["passed"] and results["Y4"]["n_evaluable"] == 8 and len(results["Y4"]["coordinated"]) == 4
    assert results["outcome"]["label"] == "CHANNEL_D_CONCENTRATED_TOKENS | CHANNEL_D_CONCENTRATED_FRAMES_CONDITIONAL | SINGLE_NEURON_REJECTED | PATTERN_TERM_NEEDED_WITHIN_FRAMES"
    o = y1["orderings"]
    assert o["row_diffuse"]["holds"] and o["random_margin"]["holds"] and o["c_L_over_Pi"]["holds"] and results["ladder_both_sets"]["kappa"]["c_L"]["S256"] == pytest.approx(K(0.6), abs=1e-9)
    # Split guard: the coordinated fresh frames' S_256 closes little → the guard fails and names the split while the pooled floors still pass.
    coordinated_fresh = [fid for fid, t in [(f"n{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(12)] if t not in pm.CUE_FINAL_TEMPLATES]
    cue_final_fresh = [f"n{i}" for i in range(12) if f"n{i}" not in coordinated_fresh]
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic(frame_alpha={(fid, "S256"): 0.2 for fid in coordinated_fresh} | {(fid, "S256"): 0.8 for fid in cue_final_fresh})
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert verdict["Y1"]["test"]["passed"] and verdict["Y2"]["test"]["kappa_c_L"] >= bc.KAPPA_CL_FLOOR and verdict["Y2"]["test"]["failing"] == ["split_guard_coordinated"] and verdict["Y2"]["test"]["split_guard"]["coordinated"]["kappa_c_L"] == pytest.approx(K(0.2), abs=1e-9) and verdict["outcome"]["Y2"] == "CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL"
    # No-harm guard needs a defined per-frame R²: a frame with no measured c_L spread is a precondition failure, never a guard failure.
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic()
    for key, entry in pairs_fresh.items():
        if key.endswith("|n2"):
            entry["c_L"] = 0.25
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert "no_harm_evaluable" in verdict["precondition_Y2"]["failing"] and verdict["outcome"]["Y2"] == "PRECONDITION_FAILED_FRAMES" and "n2" not in verdict["Y2"]["test"]["no_harm_guard"]["frames_below"]
    # No-harm guard: one fresh frame where S_256 is worse than the template base by more than the margin.
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic(frame_alpha={("n5", "S256"): -0.5})
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert set(verdict["Y2"]["test"]["no_harm_guard"]["frames_below"]) == {"n5"} and "no_harm_guard" in verdict["Y2"]["test"]["failing"] and verdict["Y2"]["per_frame_scored"]["n5"]["kappa_c_L"] < 0.0
    # A tiny gap: channel D does not matter → κ undefined → the precondition fails on both sets, no concentration label, Y3 not evaluable.
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic({"S0": 0.95})
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert verdict["outcome"]["label"].startswith("PRECONDITION_FAILED_TOKENS | PRECONDITION_FAILED_FRAMES | SINGLE_NEURON_NOT_EVALUABLE") and "gap_c_L" in verdict["precondition_Y1"]["failing"] and verdict["Y1"]["statistics"]["kappa"]["c_L"]["S256"] is None
    # A tiny gap in one decision split only: Y1 passes, Y2 is PRECONDITION_FAILED_FRAMES naming the split, never pass/fail.
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic(frame_alpha={(fid, "S0"): 0.95 for fid in coordinated_fresh})
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert verdict["outcome"]["Y1"] == "CHANNEL_D_CONCENTRATED_TOKENS" and verdict["outcome"]["Y2"] == "PRECONDITION_FAILED_FRAMES" and verdict["precondition_Y2"]["failing"] == ["split_gap_coordinated"]
    # The reference rung itself fails the precondition on the fresh set: nothing about concentration is interpreted there.
    fresh_ids = [f"n{i}" for i in range(12)]
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic(frame_alpha={(fid, "S2048"): 0.5 for fid in fresh_ids})
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert verdict["outcome"]["Y2"] == "PRECONDITION_FAILED_FRAMES" and "reference_c_L" in verdict["precondition_Y2"]["failing"] and verdict["outcome"]["Y1"] == "CHANNEL_D_CONCENTRATED_TOKENS"
    # Y3 not rejected when the single neuron carries most of the gap; the floors of Y1/Y2 are untouched.
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic({"S1": 0.6})
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert verdict["outcome"]["Y3"] == "SINGLE_NEURON_NOT_REJECTED" and verdict["Y3"]["kappa_c_L_single"] == pytest.approx(K(0.6), abs=1e-9) and verdict["outcome"]["Y1"] == "CHANNEL_D_CONCENTRATED_TOKENS"
    # Y3 rejected only with the margin: κ(S_1) just below 0.50 but S_256 not 0.25 above it → not rejected.
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic({"S1": 0.28, "S256": 0.43})  # κ 0.486 and 0.676: below 0.50, and less than 0.25 apart
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert verdict["Y3"]["kappa_c_L_single"] < 0.5 and verdict["Y3"]["gap"] < 0.25 and not verdict["Y3"]["rejected"] and verdict["outcome"]["Y1"] == "CHANNEL_D_NOT_CONCENTRATED_TOKENS"
    # Y4: a fresh cue-final frame with a tiny pattern term is named (frozen ≥ 0.90); a frame where the reference rung fails is excluded; too few evaluable → not evaluable.
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic(pi_scale={"n0": 0.02})
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert verdict["outcome"]["Y4"] == "PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES" and verdict["Y4"]["frames_at_or_above"] == ["n0"] and verdict["Y4"]["cue_final"]["n0"]["frozen_dT_r2"] >= 0.90
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic(pi_scale={"n0": 0.02}, frame_alpha={("n0", "S2048"): 0.2})
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert verdict["outcome"]["Y4"] == "PATTERN_TERM_NEEDED_WITHIN_FRAMES" and not verdict["Y4"]["cue_final"]["n0"]["evaluable"] and verdict["Y4"]["n_evaluable"] == 7
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic(frame_alpha={(fid, "S2048"): 0.2 for fid in ("n0", "n1", "n3")})
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert verdict["outcome"]["Y4"] == "PATTERN_TERM_NOT_EVALUABLE_WITHIN_FRAMES" and verdict["Y4"]["n_evaluable"] == 5
    # Y2 validity precondition: five valid cue-final frames → PRECONDITION_FAILED_FRAMES while Y1 is unaffected; invalid frames run no fresh cue.
    five_cue_final = [fid for fid, t in [(f"n{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(12)] if t in pm.CUE_FINAL_TEMPLATES][:5] + coordinated_fresh
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic(valid=five_cue_final)
    verdict = bc.score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    assert verdict["outcome"]["Y2"] == "PRECONDITION_FAILED_FRAMES" and "valid_cue_final" in verdict["precondition_Y2"]["failing"] and verdict["outcome"]["Y1"] == "CHANNEL_D_CONCENTRATED_TOKENS" and set(verdict["Y2"]["per_frame"]) == set(five_cue_final)


def test_lock_reproduction_refusal_ranking_reproduction_and_predictions_rendering():
    confirmation, lock, stage1, pairs_exposed, pairs_fresh = _synthetic()
    row = lock["predictions"]["rows"][0]
    one = {"predictions": {"rows": [row]}}
    assert bc.assert_lock_predictions_reproduced(one, {"rows": [json.loads(json.dumps(row))]}) == 0.0
    perturbed = json.loads(json.dumps(row))
    perturbed["row_R2"][1] += 1e-6
    with pytest.raises(bc.PhaseError, match="nothing was executed"):
        bc.assert_lock_predictions_reproduced(one, {"rows": [perturbed]})
    moved = json.loads(json.dumps(row))
    moved["p_t"] = 4
    with pytest.raises(bc.PhaseError, match="ordered differently"):
        bc.assert_lock_predictions_reproduced(one, {"rows": [moved]})
    scores = torch.rand(bc.N_NEURONS, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    subsets = bc.subsets_from_scores(scores)
    frame_subsets = {"f": bc.frame_subset(scores)}
    full_lock = {"run_id": "r", "protocol_code_commit": "c" * 40, "confirmation_018_sha256": "s" * 64, "lock_017_sha256": "l" * 64, "sigma_T": 1.014, "scores": scores.tolist(), "subsets": subsets, "frame_subsets": frame_subsets, "overlaps": bc.subset_overlaps(subsets),
                 "tokens": [{"word": "w0", "category": "adjective"}], "predictions": {"rows": [row], "token_means": bc.token_means_from_table([row], ["w0"])}}
    assert bc.assert_ranking_reproduced(full_lock, scores, frame_subsets) == 0.0
    with pytest.raises(bc.PhaseError, match="locked subsets"):
        bc.assert_ranking_reproduced(full_lock, scores * -1.0, frame_subsets)
    text = bc.render_predictions(full_lock)
    assert "preregistered predictions" in text and "| w0 | adjective | 1 |" in text and "κ_c_L(S_256) ≥ 0.7" in text and "Y4" in text and "S_1 [" in text
