"""Experiment 010: the exact read functional and its decompositions, the neuron split, the frozen rules, and the measurements on a six-layer fake."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import head_transport as ht
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from plural_fakes import TinyPlural

ROOT = Path(__file__).resolve().parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


def test_frozen_constants_match_the_design():
    assert ra.S_MIN == 0.35 and ra.S_HIGH == 0.65 and ra.C_MIN == 0.10 and ra.G_MAX == 0.25 and ra.CONSENSUS == 0.75
    assert ra.CONSISTENCY_MIN == 0.10 and ra.N80_CONCENTRATED == 40 and ra.TOP_NEURONS == 20 and ra.NEURON_MASS_FRACTION == 0.8
    assert ra.RUNTIME_SEED == 20260916 and ra.CONTROL_SEED == 20260923
    assert ra.COMPONENT_ORDER[:2] == ("L01.H00", "L01.H01") and ra.COMPONENT_ORDER[8] == "L01.MLP" and ra.COMPONENT_ORDER[-1] == "L02.MLP" and len(ra.COMPONENT_ORDER) == 18


def test_classification_branches():
    base = {"f_E": 0.2, "f_par": 0.5, "f_layers": 0.0, "G": 0.1, "D": 0.05, "f_total": 0.2}
    assert ra.classify_token(base) == "E_BORNE"
    # Large cancelling layer updates are not a relay.
    assert ra.classify_token({**base, "G": 1.4, "D": 0.7}) == "MIXED"
    assert ra.classify_token({"f_E": 0.8, "f_par": 0.8, "f_layers": -0.6, "G": 0.6, "D": 0.6, "f_total": 0.2}) == "LAYER_BORNE"
    assert ra.classify_token({"f_E": 0.9, "f_par": 0.9, "f_layers": 0.02, "G": 0.1, "D": 0.05, "f_total": 0.92}) == "RELAYED"
    assert ra.classify_token({"f_E": 0.6, "f_par": 0.6, "f_layers": 0.3, "G": 0.4, "D": 0.3, "f_total": 0.9}) == "AMPLIFIED"
    assert ra.classify_token({**base, "f_E": None}) == "MIXED"
    assert ra.stratum(0.35) == "low" and ra.stratum(0.65) == "high" and ra.stratum(0.5) == "mid" and ra.stratum(None) == "uninformative"


def test_neuron_concentration_is_deterministic_and_on_absolute_mass():
    terms = torch.tensor([0.5, -0.5, 0.1, -0.1, 0.0, 0.0], dtype=torch.float64)
    out = ra.neuron_concentration(terms)
    assert out["n_80"] == 2 and out["positive_mass"] == pytest.approx(0.6) and out["negative_mass"] == pytest.approx(-0.6)
    assert [entry["neuron"] for entry in out["top"]] == [0, 1, 2, 3, 4, 5]  # ties broken by index
    assert ra.neuron_concentration(torch.zeros(4, dtype=torch.float64))["n_80"] is None  # undefined on zero mass
    assert ra.jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5) and ra.jaccard([], []) == 0.0


def _analysis(f_k_value: float) -> dict:
    return {"f_k": {key: f_k_value for key in ra.COMPONENT_ORDER}}


def test_consistent_components_and_summary():
    per_frame = {"t1": {"f1": _analysis(-0.3), "f2": _analysis(-0.2), "f3": None}, "t2": {"f1": _analysis(-0.15), "f2": _analysis(-0.12)}, "t3": {"f1": _analysis(0.0), "f2": _analysis(0.0)}}
    components = ra.consistent_components(per_frame, ["t1", "t2"])
    assert components["opposers"] == list(ra.COMPONENT_ORDER) and components["supporters"] == []
    assert ra.consistent_components(per_frame, ["t1", "t2", "t3"])["opposers"] == []  # only 2 of 3 tokens oppose: below 75%
    rows = {name: {"token": name, "stratum": "low", "class": "E_BORNE"} for name in ("a", "this", "another")}
    assert ra.summarize(rows, {"opposers": [], "supporters": []}, True)["label"] == "E_BORNE+CONCENTRATED_ENCODING"
    rows["another"]["class"] = "LAYER_BORNE"
    assert ra.summarize(rows, {"opposers": [], "supporters": []}, False)["label"] == "MIXED"
    layer_rows = {name: {"token": name, "stratum": "low", "class": "LAYER_BORNE"} for name in ("a", "this", "another", "every")}
    assert ra.summarize(layer_rows, {"opposers": ["L02.MLP"], "supporters": []}, False)["label"] == "LAYER_BORNE"
    assert ra.summarize(layer_rows, {"opposers": [], "supporters": []}, False)["label"] == "MIXED"  # no consistent opposer
    assert ra.summarize({"x": {"token": "x", "stratum": "high", "class": "RELAYED"}}, {"opposers": [], "supporters": []}, False)["label"] == "MIXED"


def test_fractions_and_predictor_check():
    plural = ra.Attribution("pl", 1, "f", "cardinal", 1.0, 1.0, 0.0, {k: 0.0 for k in ra.COMPONENT_ORDER}, 1.0, 0.0, 0.0, False, 2.0, {}, {}, 0.0, 4.0, torch.zeros(2, dtype=torch.float64))
    components = {k: 0.0 for k in ra.COMPONENT_ORDER}
    components["L01.H03"], components["L02.MLP"] = 0.8, -0.9
    record = ra.Attribution("w", 2, "f", "cardinal", 0.5, 0.7, -0.2, components, 0.4, 0.0, 0.0, False, 0.3, {}, {}, 0.0, 1.0, torch.zeros(2, dtype=torch.float64))
    axis = pm.SiteAxis("T", torch.zeros(1), torch.ones(1), 1.0)
    out = ra.fractions(record, plural, axis, plural.g_E_inner)
    assert out["f_E"] == pytest.approx(0.25) and out["f_total"] == pytest.approx(0.2) and out["q_T"] == pytest.approx(0.15) and out["g_E"] == pytest.approx(0.25)
    assert out["f_L1"] == pytest.approx(0.4) and out["f_L2"] == pytest.approx(-0.45) and out["f_layers"] == pytest.approx(-0.05)
    assert out["G"] == pytest.approx(0.85) and out["D"] == pytest.approx(0.4)  # net small, gross and cumulative large
    assert ra.fractions(record, ra.Attribution("pl", 1, "f", "cardinal", 0.0, 0.0, 0.0, components, 0.0, 0.0, 0.0, False, 0.1, {}, {}, 0.0, 0.0, torch.zeros(2, dtype=torch.float64)), axis, 0.0) is None
    own = ra.Attribution("ref", 3, "f", "cardinal", 0.0, 0.0, 0.0, components, 0.0, 0.0, 0.0, True, 0.0, {}, {}, 0.0, 0.0, torch.zeros(2, dtype=torch.float64))
    assert ra.fractions(own, plural, axis, plural.g_E_inner) is None  # own-reference frames are uninformative for the token
    assert set(out["raw"]) >= {"rho_E", "rho_total", "rho_components", "head_change", "denominator_measured", "denominator_rho_plural", "identity_error"}
    rows = {f"t{i}": {"means": {"f_E": v, "g_E": v * 0.5, "f_total": v, "q_T": v}} for i, v in enumerate([0.1, 0.4, 0.9, 0.7])}
    check = ra.predictor_check(rows)
    assert check["f_E"]["spearman"] == pytest.approx(1.0) and check["f_E"]["mae"] == 0.0 and check["g_E"]["n"] == 4


@pytest.fixture(scope="module")
def fake_setting():
    manifest, manifest_sha256, extension = pm.load_inputs(ROOT)
    confirmation_006 = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(ROOT / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    pool = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    return pool, model, weights, head, cache


def test_pool_010_composition(fake_setting):
    pool, *_ = fake_setting
    assert len(pool.frames) == 24 and len(pool.tokens) == 63 and len(pool.nouns) == 80
    assert sorted(set(pool.frame_origin.values())) == ["confirmation", "confirmation-009", "extension", "manifest"]
    assert pool.token_source["either"] == "confirmation-23" and pool.token_source["this"] == "confirmation-24" and pool.token_source["a"] == "exposed-16"
    assert pool.plural_cue == {"cardinal": "cardinal:pl", "quantifier": "quantifier:pl", "coordinated-adjective": "cardinal:pl"}


def test_read_functional_matches_p1_and_the_neuron_split_on_the_fake(fake_setting):
    pool, model, weights, head, cache = fake_setting
    small = cs.Pool008(pool.frames[:1] + pool.frames_of("coordinated-adjective")[:1], pool.frame_origin, pool.tokens[:6], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    axes = cs.stage_axes(cache, weights, small)
    nouns = small.single_nouns
    for frame in small.frames:
        ref, components = ra.capture_reference(model, head, small.reference_prompt(frame), nouns)
        assert set(components) == set(ra.COMPONENT_ORDER)
        functional = ra.read_functional(head, ref, axes["T"])
        assert functional.scale == pytest.approx(float(ref.attention[frame.p_c]) / head.scale(ref.residuals[frame.p_c]))
        for name, token_id in small.tokens:
            record = ra.measure_token_010(model, weights, head, ref, components, functional, name, token_id, axes["R0"], axes["T"], nouns)
            scale = max(abs(record.head_change), 1e-9)
            assert record.identity_error / scale <= ra.IDENTITY_TOLERANCE  # ρ(Δr_c) = ρ(ΔE) + Σ_k ρ(Δout_k), up to float32 residual-sum rounding
            assert record.p1_cross_check / scale < 1e-8  # equals Experiment 009's P1 projection
            assert record.neuron_sum_error <= ra.NEURON_SUM_TOLERANCE  # Σ_j c_j = ρ(ΔE)
            assert record.rho_E == pytest.approx(record.rho_par + record.rho_perp, abs=1e-9)
            if token_id == small.reference_ids[frame.template_id]:
                assert record.own_reference and record.rho_total == 0.0 and all(value == 0.0 for value in record.rho_components.values())
            else:
                assert not record.own_reference
        # The functional applied to every neuron row reproduces ρ of the encoding change exactly (weight-only path).
        terms = ra.neuron_terms(weights, functional, small.tokens[1][1], ref.reference.cue_token_id)
        delta_e = pm.lexicon_vector(weights, small.tokens[1][1]).double() - pm.lexicon_vector(weights, ref.reference.cue_token_id).double()
        assert float(terms.sum()) == pytest.approx(functional(delta_e), rel=1e-4, abs=1e-6)


def test_inherited_009_extract_roundtrip_and_committed_file(tmp_path):
    responses = {f"t{i}|f{j}": {"template_id": "cardinal", "mean_shift": 0.1 * i} for i in range(23) for j in range(6)}
    from neural_decompiler.models import PYTHIA_70M
    payload = ra.inherited_009_payload(responses, source={"path": "x"}, manifest_sha256="m", extension_sha256="e", confirmation_009_sha256="c", lock_sha256="l", model={"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision})
    path = tmp_path / "x.json"
    path.write_text(pm.canonical_json(payload) + "\n")
    assert ra.load_inherited_009(path, manifest_sha256="m", extension_sha256="e", confirmation_009_sha256="c")["content_sha256"] == payload["content_sha256"]
    with pytest.raises(ValueError):
        ra.load_inherited_009(path, manifest_sha256="m", extension_sha256="e", confirmation_009_sha256="other")
    manifest, manifest_sha256, extension = pm.load_inputs(ROOT)
    confirmation_006 = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(ROOT / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    extract = ra.load_inherited_009(ROOT / ra.INHERITED_009_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_009_sha256=confirmation_009.content_sha256)
    lock = json.loads((ROOT / ht.LOCK_RELATIVE_PATH).read_text())
    assert extract["lock_sha256"] == lock["content_sha256"]
    assert set(extract["responses"]) == {f"{token['word']}|{frame.frame_id}" for token in confirmation_009.tokens for frame in confirmation_009.frames}
