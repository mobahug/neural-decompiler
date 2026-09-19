"""Experiment 011: the weight-defined encoding read, denominator validity, tolerances, the confirmation policy, floors, and the extract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import encoding_read as er
from neural_decompiler import head_transport as ht
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from neural_decompiler.models import PYTHIA_70M
from plural_fakes import TinyPlural
from test_cue_decompilation import toy_tokenizer_006

ROOT = Path(__file__).resolve().parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


def test_frozen_constants_match_the_design():
    assert er.Y1_SPEARMAN == 0.80 and er.Y2_SPEARMAN == 0.90 and er.TAU_MIN == 0.10 and er.TAU_MULTIPLIER == 3.0 and er.DENOMINATOR_RELATIVE_FLOOR == 0.25
    assert er.MIN_VALID_FRAMES == 4 and er.MIN_VALID_FRAMES_PER_TOKEN == 3 and er.MIN_SCORED_TOKENS == 16 and er.FRAME_CUE_EFFECT_RATE == 108 / 120
    assert er.RUNTIME_SEED == 20260916 and er.CONTROL_SEED == 20260924 and len(er.FRESH_FRAMES) == 6
    assert er.QUOTAS == {"determiner-like": 5, "numeral": 5, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 5} and "an" in er.CANDIDATES["determiner-like"]


def test_denominator_validity_tau_and_outcome():
    validity = er.denominator_validity({"cardinal": 1.0, "quantifier": 0.2, "coordinated-adjective": 1.0}, sigma_r=0.5)
    assert validity["defined"] == {"cardinal": True, "quantifier": False, "coordinated-adjective": True} and validity["n_defined"] == 2
    assert er.denominator_validity({"cardinal": 1.0, "quantifier": 0.3, "coordinated-adjective": 1.0}, sigma_r=2.0)["defined"]["quantifier"] is False  # below 0.25 σ_r
    assert er.tau([0.01, -0.01]) == er.TAU_MIN and er.tau([0.2, -0.2]) == pytest.approx(0.6) and er.tau([]) == er.TAU_MIN
    assert er.token_mean({"a": 0.2, "b": None, "c": 0.4}) == pytest.approx(0.3) and er.token_mean({"a": None}) is None
    base = {"precondition": {"passed": True}, "Y1": {"passed": True}, "Y2": {"passed": False}}
    assert er.outcome(base)["label"] == "ENCODING_READ_PREDICTS_TRANSPORT | HEAD_P1_NOT_REPLICATED"
    assert er.outcome({**base, "precondition": {"passed": False}})["label"] == "PRECONDITION_FAILED"
    assert er.outcome({**base, "Y1": {"passed": False}, "Y2": {"passed": True}})["label"] == "ENCODING_READ_FAILS | HEAD_P1_REPLICATED"


def test_fraction_replication_check():
    recorded = {"a|f": 0.5, "b|f": None}
    assert er.check_fraction_replication({"a|f": 0.5, "b|f": None}, recorded)["passed"]
    with pytest.raises(pm.IncidentError, match="informativeness"):
        er.check_fraction_replication({"a|f": 0.5, "b|f": 0.1}, recorded)
    with pytest.raises(pm.IncidentError, match="deviates"):
        er.check_fraction_replication({"a|f": 0.5 + 2e-6, "b|f": None}, recorded)
    with pytest.raises(pm.IncidentError, match="cover exactly"):
        er.check_fraction_replication({"a|f": 0.5}, recorded)


@pytest.fixture(scope="module")
def inputs():
    manifest, manifest_sha256, extension = pm.load_inputs(ROOT)
    confirmation_006 = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(ROOT / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    return manifest, manifest_sha256, extension, confirmation_006, confirmation_009


def toy_tokenizer_011(manifest):
    base = toy_tokenizer_006(manifest)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in list(ht.CANDIDATES.values()) + list(er.CANDIDATES.values()) for word in words]
    for _, text in list(ht.FRESH_FRAMES) + list(er.FRESH_FRAMES):
        for index, piece in enumerate(text.replace("{cue}", "").split()):
            words.append(piece if index == 0 else " " + piece)
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    return type(base)(vocabulary)


def test_confirmation_policy_and_validation(inputs, tmp_path):
    manifest, manifest_sha256, extension, confirmation_006, confirmation_009 = inputs
    tokenizer = toy_tokenizer_011(manifest)
    payload = er.build_confirmation_payload(tokenizer, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    words = [token["word"] for token in payload["tokens"]]
    assert len(words) == 24 and words[:5] == ["an", "such", "much", "little", "less"] and "expectation" not in payload["tokens"][0]
    an = next(token for token in payload["tokens"] if token["word"] == "an")
    assert an["licensed_frames"] == ["cardinal-011-1", "cardinal-011-2", "quantifier-011-1", "quantifier-011-2"]
    assert all(len(token["licensed_frames"]) == 6 for token in payload["tokens"] if token["word"] != "an")
    assert len(payload["token_prompts"]) == 23 * 6 + 4
    confirmation = er.validate_confirmation(payload, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    assert len(confirmation.frames_for("an")) == 4 and len(confirmation.all_prompts) == 12 + 23 * 6 + 4
    path = tmp_path / "c.json"
    assert er.freeze_confirmation(path, tokenizer, manifest, manifest_sha256, extension, confirmation_006, confirmation_009) == payload["content_sha256"]
    with pytest.raises(FileExistsError):
        er.freeze_confirmation(path, tokenizer, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    tampered = dict(payload)
    tampered["tokens"] = [dict(t, licensed_frames=t["licensed_frames"][:3]) if t["word"] == "such" else t for t in payload["tokens"]]
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in tampered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError):
        er.validate_confirmation(tampered, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)


def test_encoding_read_on_the_fake_is_frame_independent_and_matches_the_functional(inputs):
    manifest, manifest_sha256, extension, confirmation_006, confirmation_009 = inputs
    pool = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    small = cs.Pool008(pool.frames[:2] + pool.frames_of("quantifier")[:1], pool.frame_origin, pool.tokens[:8], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    axes = cs.stage_axes(cache, weights, small)
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = er.EncodingRead(ra.read_weight(head, axes["T"]), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
    for template in pm.TEMPLATE_ORDER:
        plural = read.reads(weights, plural_ids[template], template)
        assert plural["g_E"] == pytest.approx(1.0) and plural["g_par"] + plural["g_perp"] == pytest.approx(1.0)
        reference = read.reads(weights, pool.reference_ids[template], template)
        assert reference == {"g_E": 0.0, "g_par": 0.0, "g_perp": 0.0}
    # g_E is the ratio of the exact functional's encoding reads (the reference scalars cancel) — check against ra's functional on one frame.
    frame = small.frames[0]
    ref, _ = ra.capture_reference(model, head, small.reference_prompt(frame), small.single_nouns)
    functional = ra.read_functional(head, ref, axes["T"])
    token_id = small.tokens[5][1]
    e_ref, e_w, e_pl = (pm.lexicon_vector(weights, i).double() for i in (pool.reference_ids[frame.template_id], token_id, plural_ids[frame.template_id]))
    assert read.reads(weights, token_id, frame.template_id)["g_E"] == pytest.approx(functional(e_w - e_ref) / functional(e_pl - e_ref), rel=1e-9)
    sigma_r = er.sigma_r_from_pairs(read, weights, small.frames)
    assert sigma_r > 0
    validity = er.denominator_validity({template: read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}, sigma_r)
    assert set(validity["defined"]) == set(pm.TEMPLATE_ORDER)


def test_committed_010_extract_matches_the_frozen_inputs(inputs):
    manifest, manifest_sha256, extension, confirmation_006, confirmation_009 = inputs
    extract = er.load_inherited_010(ROOT / er.INHERITED_010_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256, confirmation_009_sha256=confirmation_009.content_sha256)
    pool = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
    assert set(extract["fractions"]) == {f"{name}|{frame.frame_id}" for name, _ in pool.tokens for frame in pool.frames}
    assert sum(1 for value in extract["fractions"].values() if value is None) == 24  # the two reference cues' own-reference frames (16 + 8)
