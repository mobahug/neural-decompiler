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


def _frame(template, valid=True):
    return {"template_id": template, "valid": valid}


def _analysis(q, f_layers=0.0, G=0.3, f_E=0.5, f_perp=-0.1, p1=None):
    return {"q_T": q, "f_total": p1 if p1 is not None else q, "p1_fraction": p1 if p1 is not None else q, "f_E": f_E, "f_perp": f_perp, "f_layers": f_layers, "G": G, "dc": -1.0, "dc_beh": -1.0}


def _lock(tokens, tau_g=0.2, tau_M=0.15, exposed_range=None):
    predictions = {}
    for word, g in tokens.items():
        predictions[word] = {"by_template": {t: {"g_E": g, "g_par": g, "g_perp": 0.0} for t in pm.TEMPLATE_ORDER}, "means": {"g_E": g, "g_par": g, "g_perp": 0.0},
                             "baseline_009": {t: g for t in pm.TEMPLATE_ORDER}, "licensed_frames": []}
    return {"predictions": {"tokens": predictions}, "tolerances": {"tau_g": tau_g, "tau_M": tau_M}, "defined_templates": list(pm.TEMPLATE_ORDER), "exposed_net_layer_range": exposed_range}


class _Confirmation:
    def __init__(self, frames, tokens):
        self.frames = [pm.Frame(t, fid, (1, 2), () if t in pm.CUE_FINAL_TEMPLATES else (7,), {"sg": 5, "pl": 6}, "x {cue}" + ("" if t in pm.CUE_FINAL_TEMPLATES else " y")) for fid, t in frames]
        self.tokens = [{"word": w, "category": "x", "licensed_frames": [fid for fid, _ in frames]} for w in tokens]


def test_scoring_rules_on_synthetic_tables():
    frames = [("c1", "cardinal"), ("c2", "cardinal"), ("q1", "quantifier"), ("q2", "quantifier"), ("a1", "coordinated-adjective"), ("a2", "coordinated-adjective")]
    words = [f"w{i}" for i in range(18)]
    g_values = {w: 0.05 * i for i, w in enumerate(words)}
    confirmation = _Confirmation(frames, words)
    frames_out = {fid: _frame(t) for fid, t in frames}
    # Perfect agreement: both pass.
    per_token = {w: {fid: _analysis(g_values[w]) for fid, _ in frames} for w in words}
    results = er.score_confirmation(frames_out, per_token, confirmation, _lock(g_values, exposed_range={"min": -0.1, "max": 0.1}))
    assert results["precondition"]["passed"] and results["Y1"]["passed"] and results["Y2"]["passed"] and results["outcome"]["label"] == "ENCODING_READ_PREDICTS_TRANSPORT | HEAD_P1_REPLICATED"
    assert results["descriptive"]["net_layer_change_within_exposed_range"]["count"] == 18
    # Y1 fails on Spearman only (a small rank scramble within τ_g) and on MAE only (shifted by more than τ_g).
    scrambled = {w: {fid: _analysis(g_values[words[(i + 9) % 18]]) for fid, _ in frames} for i, w in enumerate(words)}
    verdict = er.score_confirmation(frames_out, scrambled, confirmation, _lock(g_values, tau_g=2.0))
    assert verdict["Y1"]["failing"] == ["spearman"]
    shifted = {w: {fid: _analysis(g_values[w] + 0.5) for fid, _ in frames} for w in words}
    assert er.score_confirmation(frames_out, shifted, confirmation, _lock(g_values))["Y1"]["failing"] == ["mae"]
    # Y2 fails when the P1 fraction does not track q_T.
    bad_p1 = {w: {fid: _analysis(g_values[w], p1=1.0 - g_values[w]) for fid, _ in frames} for w in words}
    assert not er.score_confirmation(frames_out, bad_p1, confirmation, _lock(g_values))["Y2"]["passed"]
    # Precondition: three valid frames → fail; fifteen scored tokens → fail; a token with two valid licensed frames is not scored.
    three = {fid: _frame(t, valid=fid in ("c1", "q1", "a1")) for fid, t in frames}
    partial = {w: {fid: _analysis(g_values[w]) for fid, _ in frames if fid in ("c1", "q1", "a1")} for w in words}
    assert er.score_confirmation(three, partial, confirmation, _lock(g_values))["outcome"]["label"] == "PRECONDITION_FAILED"
    fifteen = {w: {fid: _analysis(g_values[w]) for fid, _ in frames} for w in words[:15]} | {w: {} for w in words[15:]}
    assert er.score_confirmation(frames_out, fifteen, confirmation, _lock(g_values))["precondition"]["scored_tokens"] == 15
    two_frames = {w: {fid: _analysis(g_values[w]) for fid, _ in frames} for w in words[:17]} | {words[17]: {fid: _analysis(0.5) for fid, _ in frames[:2]}}
    scored = er.score_confirmation(frames_out, two_frames, confirmation, _lock(g_values))
    assert not scored["tokens"][words[17]]["scored"] and scored["precondition"]["scored_tokens"] == 17 and scored["precondition"]["passed"]
    # ḡ_E and q̄_T use the same frame set: a token valid only in cardinal frames + one quantifier frame gets the frame-weighted g mean.
    lock = _lock(g_values)
    lock["predictions"]["tokens"][words[0]]["by_template"]["cardinal"]["g_E"] = 0.9
    subset = {w: {fid: _analysis(g_values[w]) for fid, _ in frames} for w in words}
    subset[words[0]] = {"c1": _analysis(0.1), "c2": _analysis(0.1), "q1": _analysis(0.1)}
    row = er.score_confirmation(frames_out, subset, confirmation, lock)["tokens"][words[0]]
    assert row["frames"] == ["c1", "c2", "q1"] and row["g_E_mean"] == pytest.approx((0.9 + 0.9 + 0.0) / 3)


def test_lock_reproduction_refusal_and_enforced_identities():
    lock = {"predictions": {"tokens": {"w": {"by_template": {"cardinal": {"g_E": 0.5, "g_par": 0.4, "g_perp": 0.1}}, "means": {"g_E": 0.5}, "baseline_009": {"cardinal": 0.3}}}}}
    assert er.assert_lock_predictions_reproduced(lock, lock["predictions"]) == 0.0
    perturbed = {"tokens": {"w": {"by_template": {"cardinal": {"g_E": 0.5, "g_par": 0.4, "g_perp": 0.1}}, "means": {"g_E": 0.5}, "baseline_009": {"cardinal": 0.3 + 1e-6}}}}
    with pytest.raises(er.PhaseError, match="nothing was executed"):
        er.assert_lock_predictions_reproduced(lock, perturbed)
    plural = ra.Attribution("pl", 1, "f", "cardinal", 1.0, 1.0, 0.0, {}, 2.0, 0.0, 0.0, False, 2.0, {}, {}, 0.0, 4.0, torch.zeros(2, dtype=torch.float64))
    good = ra.Attribution("w", 2, "f", "cardinal", 0.5, 0.5, 0.0, {}, 0.4, 1e-9, 1e-12, False, 0.3, {}, {}, 1e-9, 1.0, torch.zeros(2, dtype=torch.float64))
    assert set(er.enforce_identities(good, plural, "x")) == {"rho_identity", "p1_cross_check", "neuron_sum"}
    bad = ra.Attribution("w", 2, "f", "cardinal", 0.5, 0.5, 0.0, {}, 0.4, 1e-2, 1e-12, False, 0.3, {}, {}, 1e-9, 1.0, torch.zeros(2, dtype=torch.float64))
    with pytest.raises(pm.IncidentError, match="ρ"):
        er.enforce_identities(bad, plural, "x")
