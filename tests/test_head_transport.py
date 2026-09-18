"""Experiment 009: head weights and the exact OV decomposition, the confirmation policy, the rules, and the measurements on a six-layer fake."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import head_transport as ht
from neural_decompiler import plural_mechanism as pm
from neural_decompiler.candidate_screening import Split
from plural_fakes import TinyPlural
from test_cue_decompilation import toy_tokenizer_006

ROOT = Path(__file__).resolve().parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


def test_frozen_constants_match_the_design():
    assert ht.HEAD_KEY == "L03.H04" and ht.G_MAX == 0.25 and ht.MECHANISM_R2_FLOOR == 0.90 and ht.MECHANISM_MAE_CEILING == 0.10
    assert ht.TAU_MIN == 0.10 and ht.TAU_MULTIPLIER == 3.0 and ht.Y1_SPEARMAN == 0.90 and ht.Y2_SPEARMAN == 0.80
    assert ht.SINGULAR_MAX_Q == 0.35 and ht.PLURAL_MIN_Q == 0.65 and ht.PLURAL_RATE == 0.8 and ht.AN_MIN_VOWEL_NOUNS == 10
    assert ht.RUNTIME_SEED == 20260916 and ht.CONTROL_SEED == 20260922 and ht.QUOTAS == {"singular-selecting": 3, "plural-numeral": 6, "plural-quantity": 5, "number-neutral": 6, "bare-adjective": 4}
    assert "million" not in ht.CANDIDATES["plural-numeral"] and "which" not in ht.CANDIDATES["number-neutral"] and len(ht.FRESH_FRAMES) == 6


# ---------------------------------------------------------------------------
# The exact OV decomposition on planted tensors.


def _random_head(d: int = 8, d_head: int = 2, seed: int = 0) -> ht.HeadWeights:
    g = torch.Generator().manual_seed(seed)
    return ht.HeadWeights(3, 4, 1.0 + 0.1 * torch.randn(d, generator=g, dtype=torch.float64), 0.05 * torch.randn(d, generator=g, dtype=torch.float64), 1e-5,
                          torch.randn(d, d_head, generator=g, dtype=torch.float64) / d**0.5, 0.1 * torch.randn(d_head, generator=g, dtype=torch.float64), torch.randn(d_head, d, generator=g, dtype=torch.float64) / d_head**0.5)


def _levels(head, p_c, A_ref, A_patch, r_ref, r_patch):
    measured = ht.reconstruct_head_result(head, A_patch, r_patch) - ht.reconstruct_head_result(head, A_ref, r_ref)
    return ht.ov_levels(head, p_c=p_c, attention_ref=A_ref, attention_patch=A_patch, residuals_ref=r_ref, residuals_patch=r_patch, delta_head=measured), measured


def test_ov_identity_and_planted_cases():
    head = _random_head()
    g = torch.Generator().manual_seed(1)
    n_keys, p_c = 4, 2
    r_ref = [torch.randn(8, generator=g, dtype=torch.float64) for _ in range(n_keys)]
    A_ref = torch.softmax(torch.randn(n_keys, generator=g, dtype=torch.float64), dim=0)
    # (a) a small zero-mean change orthogonal to the centered residual leaves the LayerNorm scale unchanged to first order:
    #     the fixed-normalization P1 agrees with P2, P3, and the measured change.
    centered = r_ref[p_c] - r_ref[p_c].mean()
    delta = torch.randn(8, generator=g, dtype=torch.float64)
    delta = delta - delta.mean()
    delta = delta - (delta @ centered) / (centered @ centered) * centered
    tiny = [r.clone() for r in r_ref]
    tiny[p_c] = tiny[p_c] + 1e-4 * delta
    out, measured = _levels(head, p_c, A_ref, A_ref, r_ref, tiny)
    assert out["identity_error"] < 1e-12 and out["decomposition_error"] < 1e-12
    for key in ("delta_T1", "delta_T2", "delta_T3"):
        assert torch.allclose(out[key], measured, atol=1e-10, rtol=1e-3)
    # A change along the centered residual changes the scale and is not captured by the fixed-normalization P1 even when small.
    along = [r.clone() for r in r_ref]
    along[p_c] = along[p_c] + 1e-2 * centered
    out_along, measured_along = _levels(head, p_c, A_ref, A_ref, r_ref, along)
    assert torch.allclose(out_along["delta_T2"], measured_along, atol=1e-10) and not torch.allclose(out_along["delta_T1"], measured_along, rtol=1e-3, atol=1e-12)
    # (b) a large change that alters the LayerNorm scale: P2 == P3 == measured, P1 differs.
    big = [r.clone() for r in r_ref]
    big[p_c] = big[p_c] * 3.0 + 1.0
    out, measured = _levels(head, p_c, A_ref, A_ref, r_ref, big)
    assert torch.allclose(out["delta_T2"], measured, atol=1e-10) and torch.allclose(out["delta_T3"], measured, atol=1e-10)
    assert not torch.allclose(out["delta_T1"], measured, atol=1e-3)
    assert out["sigma_patch"] > out["sigma_ref"]
    # (c) an attention-only change: P1 == P2 == 0, P3 == measured.
    A_patch = torch.softmax(torch.randn(n_keys, generator=g, dtype=torch.float64), dim=0)
    out, measured = _levels(head, p_c, A_ref, A_patch, r_ref, r_ref)
    assert float(out["delta_T1"].abs().max()) < 1e-12 and float(out["delta_T2"].abs().max()) < 1e-12
    assert torch.allclose(out["delta_T3"], measured, atol=1e-10) and float(out["remainder"].abs().max()) < 1e-12
    # (d) a change at another position lands in the remainder; the identity still holds.
    other = [r.clone() for r in r_ref]
    other[p_c + 1] = other[p_c + 1] + torch.randn(8, generator=g, dtype=torch.float64)
    out, measured = _levels(head, p_c, A_ref, A_patch, r_ref, other)
    assert out["identity_error"] < 1e-10 and out["decomposition_error"] < 1e-10
    assert float(out["remainder"].abs().max()) > 1e-3 and torch.allclose(out["delta_T3"] + out["remainder"], measured, atol=1e-10)


def test_level_table_rule():
    pairs = [{"q_T": q, "levels": {"P1": q + 0.5, "P2": q + 0.02 * (-1) ** i, "P3": q, "remainder": 0.0}} for i, q in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.2, 0.8])]
    table = ht.level_table(pairs)
    assert table["locked_level"] == 2 and table["levels"]["P1"]["r2"] < ht.MECHANISM_R2_FLOOR and table["tau_M"] == pytest.approx(max(ht.TAU_MIN, 3 * 0.02))
    none = ht.level_table([{"q_T": q, "levels": {"P1": 0.0, "P2": 0.0, "P3": 0.0, "remainder": q}} for q in [0.1, 0.5, 0.9]])
    assert none["locked_level"] is None and none["tau_M"] is None


# ---------------------------------------------------------------------------
# Rules on planted data.


def test_transport_rule_recovers_a_planted_linear_rule_and_gates():
    g = torch.Generator().manual_seed(5)
    d = 8
    v_true = torch.randn(d, generator=g, dtype=torch.float64)
    v_true = v_true / v_true.norm()
    beta_true = {"cardinal": 0.8, "quantifier": 1.1, "coordinated-adjective": 0.9}
    # With orthonormal token vectors Σ x_i x_iᵀ = I, so the cross-moment direction Σ x_i y_i recovers a planted linear rule exactly.
    basis, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    tokens = [f"t{i}" for i in range(d)]
    rows = []
    for i, token in enumerate(tokens):
        x = basis[:, i]
        for template in pm.TEMPLATE_ORDER:
            for _ in range(2):
                rows.append((token, template, x, beta_true[template] * float(x @ v_true)))
    rule = ht.fit_transport_rule(rows, tokens)
    assert abs(abs(float(rule.v @ v_true)) - 1.0) < 1e-6
    for template in pm.TEMPLATE_ORDER:
        assert rule.predict(template, rows[0][2]) == pytest.approx(beta_true[template] * float(rows[0][2] @ v_true), abs=1e-6)
    table = ht.transport_loco(rows, tokens)
    assert set(table) == set(tokens) and all(set(entry) == {"predicted", "measured", "mae"} for entry in table.values())
    gate = ht.gate_scalar(table)
    assert set(gate) >= {"passed", "spearman", "normalized_rmse"}
    assert ht.tau_q({"a": {"mae": 0.01}, "b": {"mae": 0.01}}) == ht.TAU_MIN and ht.tau_q({"a": {"mae": 0.2}, "b": {"mae": 0.2}}) == pytest.approx(0.6)
    ridge = ht.ridge_scalar_loco(rows, tokens[:6], multipliers=(1e-2, 1e-1))
    assert set(ridge["table"]) == set(tokens[:6]) and all(m in (1e-2, 1e-1) for m in ridge["multipliers"].values())
    assert rule.digest() == ht.TransportRule(rule.v.clone(), dict(rule.beta), ()).digest()


# ---------------------------------------------------------------------------
# Confirmation policy (tokenizer-only) with the real frozen inputs and a toy tokenizer.


def toy_tokenizer_009(manifest):
    base = toy_tokenizer_006(manifest)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in ht.CANDIDATES.values() for word in words]
    for _, text in ht.FRESH_FRAMES:
        for index, piece in enumerate(text.replace("{cue}", "").split()):
            words.append(piece if index == 0 else " " + piece)
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    return type(base)(vocabulary)


@pytest.fixture(scope="module")
def inputs():
    manifest, manifest_sha256, extension = pm.load_inputs(ROOT)
    confirmation_006 = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    return manifest, manifest_sha256, extension, confirmation_006


def test_confirmation_policy_and_validation(inputs, tmp_path):
    manifest, manifest_sha256, extension, confirmation_006 = inputs
    tokenizer = toy_tokenizer_009(manifest)
    payload = ht.build_confirmation_payload(tokenizer, manifest, manifest_sha256, extension, confirmation_006)
    words = [token["word"] for token in payload["tokens"]]
    assert "an" not in words  # eight vowel-initial nouns in the exposed pool: fewer than ten
    assert words[:2] == ["either", "neither"] and "million" not in words and "which" not in words
    by_category = {}
    for token in payload["tokens"]:
        by_category.setdefault(token["category"], []).append(token["word"])
    assert len(by_category["plural-numeral"]) == 6 and len(by_category["plural-quantity"]) == 5 and len(by_category["number-neutral"]) == 6 and len(by_category["bare-adjective"]) == 4
    assert all(len(token["licensed_frames"]) == 6 and len(token["licensed_noun_keys"]) == 79 for token in payload["tokens"])
    assert len(payload["token_prompts"]) == 6 * len(words)
    confirmation = ht.validate_confirmation(payload, manifest, manifest_sha256, extension, confirmation_006)
    assert len(confirmation.frames) == 6 and confirmation.frames[0].frame_id == "cardinal-009-1" and len(confirmation.all_prompts) == 12 + 6 * len(words)
    path = tmp_path / "confirmation.json"
    digest = ht.freeze_confirmation(path, tokenizer, manifest, manifest_sha256, extension, confirmation_006)
    assert digest == payload["content_sha256"]
    with pytest.raises(FileExistsError):
        ht.freeze_confirmation(path, tokenizer, manifest, manifest_sha256, extension, confirmation_006)
    loaded = ht.load_confirmation(path, manifest, manifest_sha256, extension, confirmation_006)
    assert loaded.content_sha256 == digest
    tampered = dict(payload)
    tampered["tokens"] = [dict(t, expectation="high") if t["word"] == "either" else t for t in payload["tokens"]]
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in tampered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError, match="frozen candidate lists"):
        ht.validate_confirmation(tampered, manifest, manifest_sha256, extension, confirmation_006)


def test_an_policy_with_enough_vowel_nouns():
    nouns = [pm.Noun(word, Split.DEVELOPMENT, "simple-suffix", (i,), (100 + i,)) for i, word in enumerate(["apple", "engine", "island", "ocean", "uncle", "arch", "elbow", "idea", "oven", "umpire", "stone", "door"])]
    frames = [pm.Frame("cardinal", "cardinal-009-1", (1, 2), (), {"sg": 5, "pl": 6}, "x {cue}"), pm.Frame("coordinated-adjective", "coordinated-adjective-009-1", (1, 2), (7,), {"sg": 5, "pl": 6}, "y {cue} z")]
    assert ht.licensed_frames("an", frames) == ["cardinal-009-1"] and ht.licensed_frames("either", frames) == ["cardinal-009-1", "coordinated-adjective-009-1"]
    assert ht.licensed_noun_keys("an", nouns) == ["apple", "engine", "island", "ocean", "uncle", "arch", "elbow", "idea", "oven", "umpire"]
    assert len(ht.licensed_noun_keys("my", nouns)) == 12


# ---------------------------------------------------------------------------
# Measurements on the fake with the real exposed pool.


@pytest.fixture(scope="module")
def fake_setting(inputs):
    manifest, manifest_sha256, extension, confirmation_006 = inputs
    pool = cs.build_pool(manifest, extension, confirmation_006)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    return pool, model, weights, head, cache


def test_reconstruction_and_measurements_on_the_fake(fake_setting):
    pool, model, weights, head, cache = fake_setting
    small = cs.Pool008(pool.frames[:1] + pool.frames_of("coordinated-adjective")[:1], pool.frame_origin, pool.tokens[:6], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    axes = cs.stage_axes(cache, weights, small)
    nouns = small.single_nouns
    noun_keys = [n.lexical_key for n in nouns]
    for frame in small.frames:
        ref = ht.capture_reference(model, head, cache, small.reference_prompt(frame), nouns)
        assert ref.reconstruction_error < ht.RECONSTRUCTION_TOLERANCE and len(ref.residuals) == frame.p_t + 1
        plural = ht.measure_token(model, weights, head, cache, ref, small.plural_cue[frame.template_id], frame.cue_ids["pl"], axes["R0"], nouns, token_prompt=None)
        plural_norm = float(plural.full.head_delta.norm())
        for name, token_id in small.tokens:
            record = ht.measure_token(model, weights, head, cache, ref, name, token_id, axes["R0"], nouns, token_prompt=None)
            analysis = ht.frame_analysis_009(record, plural, axes, noun_keys)
            ht.check_ov_identities(analysis, plural_norm, f"{frame.frame_id}/{name}")
            assert set(analysis["levels"]) == {"P1", "P2", "P3", "remainder"} and set(analysis["gaps"]) == set(ht.ADDITIVITY_STAGES)
            if token_id == small.reference_ids[frame.template_id]:
                assert analysis["q_T"] == 0.0 or analysis["q_T"] is None
                assert record.dc_beh == 0.0 and all(float(v.abs().max()) == 0.0 for v in record.full.delta.values())
            if frame.template_id in pm.CUE_FINAL_TEMPLATES and analysis["levels"]["remainder"] is not None:
                assert abs(analysis["levels"]["remainder"]) < 1e-6  # no position after the cue can change in a cue-final frame
        plural_analysis = ht.frame_analysis_009(plural, plural, axes, noun_keys)
        if plural_analysis["q_T"] is not None:  # the fake's head may carry no number signal in a frame (uninformative stage)
            assert plural_analysis["q_T"] == pytest.approx(1.0) and plural_analysis["levels"]["P3"] == pytest.approx(1.0 - (plural_analysis["levels"]["remainder"] or 0.0), abs=1e-6)
        assert all(plural_analysis["fractions"]["full"][stage] in (None, pytest.approx(1.0)) for stage in cs.VECTOR_STAGES)


def test_additivity_stage_and_ov_identity_check_and_lock_reproduction():
    # The non-additivity stage is the first stage whose oriented gap exceeds G_MAX; NONE otherwise; the mode ties to the earliest stage.
    assert ht.modal_stage(["R2", "T", "T", "NONE"]) == ("T", 0.5) and ht.modal_stage(["R3", "R1"]) == ("R1", 0.5) and ht.modal_stage(["NONE", "NONE"]) == ("NONE", 1.0)
    with pytest.raises(pm.IncidentError, match="identity error"):
        ht.check_ov_identities({"identity_error": 1e-2, "decomposition_error": 0.0}, 1.0, "x")
    ht.check_ov_identities({"identity_error": 1e-6, "decomposition_error": 1e-6}, 1.0, "x")
    lock = {"predictions": {"tokens": {"w": {"frames": {"f": {"q_T": 0.5, "contrast": {"mean": -1.0, "by_noun": {"n": -1.0}}, "program_007": {"mean": -0.5, "by_noun": {"n": -0.5}}}}}}}}
    assert ht.assert_lock_predictions_reproduced(lock, lock["predictions"]) == 0.0
    perturbed = {"tokens": {"w": {"frames": {"f": {"q_T": 0.5, "contrast": {"mean": -1.0, "by_noun": {"n": -1.0 + 1e-6}}, "program_007": {"mean": -0.5, "by_noun": {"n": -0.5}}}}}}}
    with pytest.raises(ht.PhaseError, match="nothing was executed"):
        ht.assert_lock_predictions_reproduced(lock, perturbed)


def test_outcome_triple_and_not_locked():
    base = {"cue_effect": {"passed": True}, "Y1": {"passed": True, "level": 1}, "Y2": {"passed": False}, "Y3": {"passed": None}}
    assert ht.outcome(base)["label"] == "HEAD_MECHANISM_CONFIRMED_P1 | TRANSPORT_RULE_FAILED | NOT_LOCKED"
    assert ht.outcome({**base, "cue_effect": {"passed": False}})["label"] == "CUE_EFFECT_NOT_REPLICATED"
    assert ht.outcome({**base, "Y1": {"passed": False, "level": 2}, "Y3": {"passed": True}})["label"] == "HEAD_MECHANISM_NOT_SUPPORTED | TRANSPORT_RULE_FAILED | CONTRAST_RULE_PREDICTED"
    assert ht.outcome({**base, "Y1": {"passed": None, "level": None}})["mechanism"] == "NOT_LOCKED"
