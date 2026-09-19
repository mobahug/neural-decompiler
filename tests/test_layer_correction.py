"""Experiment 012: the token-local MLP model, its exactness for block 1, the neuron and ladder identities, base states, tolerances, the confirmation policy, floors, and the extracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import encoding_read as er
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from plural_fakes import TinyPlural
from test_encoding_read import toy_tokenizer_011

ROOT = Path(__file__).resolve().parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


def test_frozen_constants_match_the_design():
    assert lc.Y1_SPEARMAN == 0.80 and lc.Y1_R2 == 0.50 and lc.Y2_SPEARMAN == 0.90 and lc.TAU_MIN == 0.10 and lc.TAU_MULTIPLIER == 3.0
    assert lc.MIN_VALID_FRAMES == 4 and lc.MIN_VALID_FRAMES_PER_TOKEN == 3 and lc.MIN_SCORED_TOKENS == 16 and lc.FRAME_CUE_EFFECT_RATE == 108 / 120
    assert lc.RUNTIME_SEED == 20260916 and lc.CONTROL_SEED == 20260924 and len(lc.FRESH_FRAMES) == 6 and lc.LAYERS == (1, 2)
    assert lc.QUOTAS == {"determiner-like": 5, "numeral": 5, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 5}
    assert lc.SUBSAMPLE_SIZE == 24 and lc.SUBSAMPLE_DRAWS == 5000 and lc.LOCK_PREDICTION_TOLERANCE == 1e-9 and lc.REPLICATION_TOLERANCE == 1e-6
    assert set(lc.MLP_KEYS) | set(lc.HEAD_KEYS) == set(ra.COMPONENT_ORDER) and len(lc.HEAD_KEYS) == 16
    exposed_011 = {token["word"] for token in json.loads((ROOT / er.CONFIRMATION_RELATIVE_PATH).read_text())["tokens"]}
    assert not (set(sum(lc.CANDIDATES.values(), ())) & exposed_011), "012 candidates must not be Experiment 011's frozen tokens"
    assert lc.OUTCOME_Y1 == ("LAYER_CORRECTION_TOKEN_LOCAL_MLP", "LAYER_CORRECTION_NOT_TOKEN_LOCAL") and lc.OUTCOME_Y2 == ("COMPOSITE_PREDICTS_P1", "COMPOSITE_FAILS_P1")


def test_tau_explained_variance_comparison_subsamples_and_outcome():
    assert lc.tau([0.01, -0.01]) == lc.TAU_MIN and lc.tau([0.2, -0.2]) == pytest.approx(0.6) and lc.tau([]) == lc.TAU_MIN
    measured = [0.0, 0.1, 0.2, 0.3]
    assert lc.explained_variance(measured, measured) == pytest.approx(1.0)
    assert lc.explained_variance([0.15] * 4, measured) == pytest.approx(0.0)
    assert lc.explained_variance([0.5 * m for m in measured], measured) < 0.5  # a half-scale predictor fails the magnitude floor
    assert lc.explained_variance([1.0, 1.0], [0.5, 0.5]) is None and lc.explained_variance([1.0], [0.5]) is None
    c = lc.comparison([0.1, 0.2, 0.4], [0.0, 0.2, 0.3])
    assert c["n"] == 3 and c["spearman"] == pytest.approx(1.0) and c["mae"] == pytest.approx(0.2 / 3) and c["bias"] == pytest.approx(0.2 / 3) and c["r2"] is not None
    assert lc.comparison([], [])["spearman"] is None
    predicted = {f"w{i}": 0.02 * i + (0.01 if i % 2 else -0.01) for i in range(40)}
    measured_by = {f"w{i}": 0.02 * i for i in range(40)}
    distribution = lc.subsample_distribution(predicted, measured_by, draws=200)
    assert distribution["draws"] == 200 and set(distribution["percentiles"]) == {"1", "5", "10", "50"} and all(0.9 < v <= 1.0 for v in distribution["percentiles"].values())
    assert lc.subsample_distribution({"a": 1.0}, {"a": 1.0})["draws"] == 0
    base = {"precondition": {"passed": True}, "Y1": {"passed": True}, "Y2": {"passed": False}}
    assert lc.outcome(base)["label"] == "LAYER_CORRECTION_TOKEN_LOCAL_MLP | COMPOSITE_FAILS_P1"
    assert lc.outcome({**base, "precondition": {"passed": False}})["label"] == "PRECONDITION_FAILED"
    assert lc.outcome({**base, "Y1": {"passed": False}, "Y2": {"passed": True}})["label"] == "LAYER_CORRECTION_NOT_TOKEN_LOCAL | COMPOSITE_PREDICTS_P1"


def _entry(value, **overrides):
    entry = {"f_k": {key: value for key in ra.COMPONENT_ORDER}, "f_E": value, "f_total": value, "f_layers": value, "q_T": value}
    entry.update(overrides)
    return entry


def test_ledger_replication_check():
    recorded = {"a|f": _entry(0.5), "b|f": None}
    assert lc.check_ledger_replication({"a|f": _entry(0.5), "b|f": None, "c|f": _entry(0.1)}, recorded)["passed"]
    with pytest.raises(pm.IncidentError, match="informativeness"):
        lc.check_ledger_replication({"a|f": _entry(0.5), "b|f": _entry(0.1)}, recorded)
    with pytest.raises(pm.IncidentError, match="deviates"):
        lc.check_ledger_replication({"a|f": _entry(0.5, q_T=0.5 + 2e-6), "b|f": None}, recorded)
    bad_component = _entry(0.5)
    bad_component["f_k"]["L02.MLP"] += 2e-6
    with pytest.raises(pm.IncidentError, match="L02.MLP"):
        lc.check_ledger_replication({"a|f": bad_component, "b|f": None}, recorded)
    with pytest.raises(pm.IncidentError, match="lacks"):
        lc.check_ledger_replication({"a|f": _entry(0.5)}, recorded)


@pytest.fixture(scope="module")
def inputs():
    manifest, manifest_sha256, extension = pm.load_inputs(ROOT)
    confirmation_006 = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    confirmation_009 = ht.load_confirmation(ROOT / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
    confirmation_011 = er.load_confirmation(ROOT / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": confirmation_006.content_sha256, "confirmation_009": confirmation_009.content_sha256, "confirmation_011": confirmation_011.content_sha256}
    pool = lc.build_pool_012(manifest, extension, confirmation_006, confirmation_009, confirmation_011)
    return manifest, digests, pool


def toy_tokenizer_012(manifest):
    base = toy_tokenizer_011(manifest)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in lc.CANDIDATES.values() for word in words]
    for _, text in lc.FRESH_FRAMES:
        for index, piece in enumerate(text.replace("{cue}", "").split()):
            words.append(piece if index == 0 else " " + piece)
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    return type(base)(vocabulary)


def test_pool_012_extends_the_exposed_pool(inputs):
    manifest, digests, pool = inputs
    assert len(pool.tokens) == 87 and len(pool.frames) == 30 and all(len(pool.frames_of(template)) == 10 for template in pm.TEMPLATE_ORDER)
    assert pool.token_source["an"] == "confirmation-011" and pool.token_category["eighteen"] == "numeral" and pool.frame_origin["cardinal-011-1"] == "confirmation-011"


def test_committed_ledgers_match_the_frozen_inputs(inputs):
    manifest, digests, pool = inputs
    ledger_010 = lc.load_inherited_ledger(ROOT / lc.INHERITED_010_LEDGER_RELATIVE_PATH, experiment="010", digests=digests, expected_size=63 * 24)
    ledger_011 = lc.load_inherited_ledger(ROOT / lc.INHERITED_011_LEDGER_RELATIVE_PATH, experiment="011", digests=digests, expected_size=142)
    exposed_010 = {f"{name}|{frame.frame_id}" for name, _ in pool.tokens if pool.token_source[name] != "confirmation-011" for frame in pool.frames if pool.frame_origin[frame.frame_id] != "confirmation-011"}
    assert set(ledger_010["entries"]) == exposed_010 and sum(1 for value in ledger_010["entries"].values() if value is None) == 24
    assert all(value is not None and set(value["f_k"]) == set(ra.COMPONENT_ORDER) for value in ledger_011["entries"].values())
    assert all(key.split("|")[1].endswith(("-011-1", "-011-2")) for key in ledger_011["entries"])
    with pytest.raises(ValueError):
        lc.load_inherited_ledger(ROOT / lc.INHERITED_010_LEDGER_RELATIVE_PATH, experiment="011", digests=digests, expected_size=63 * 24)


def test_confirmation_policy_and_validation(inputs, tmp_path):
    manifest, digests, pool = inputs
    tokenizer = toy_tokenizer_012(manifest)
    payload = lc.build_confirmation_payload(tokenizer, pool, digests)
    words = [token["word"] for token in payload["tokens"]]
    assert len(words) == 24 and words[:5] == ["half", "whichever", "whatever", "which", "what"] and "expectation" not in payload["tokens"][0]
    assert all(len(token["licensed_frames"]) == 6 for token in payload["tokens"]) and len(payload["token_prompts"]) == 24 * 6
    assert [frame["frame_id"] for frame in payload["frames"]] == ["cardinal-012-1", "cardinal-012-2", "quantifier-012-1", "quantifier-012-2", "coordinated-adjective-012-1", "coordinated-adjective-012-2"]
    confirmation = lc.validate_confirmation(payload, pool, digests)
    assert len(confirmation.frames_for("half")) == 6 and len(confirmation.all_prompts) == 12 + 24 * 6
    path = tmp_path / "c.json"
    assert lc.freeze_confirmation(path, tokenizer, pool, digests) == payload["content_sha256"]
    with pytest.raises(FileExistsError):
        lc.freeze_confirmation(path, tokenizer, pool, digests)
    tampered = dict(payload)
    tampered["tokens"] = [dict(t, licensed_frames=t["licensed_frames"][:3]) if t["word"] == "half" else t for t in payload["tokens"]]
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in tampered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError):
        lc.validate_confirmation(tampered, pool, digests)
    reordered = dict(payload)
    reordered["tokens"] = list(reversed(payload["tokens"]))
    reordered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in reordered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError):
        lc.validate_confirmation(reordered, pool, digests)


def test_base_states_and_leave_one_frame_out(inputs):
    manifest, digests, pool = inputs
    states = {frame.frame_id: (torch.full((4,), float(i)), torch.full((4,), 10.0 + i)) for i, frame in enumerate(pool.frames)}
    full = lc.template_bases(pool.frames, states)
    cardinal = [i for i, frame in enumerate(pool.frames) if frame.template_id == "cardinal"]
    assert full["cardinal"][0][0].item() == pytest.approx(sum(cardinal) / len(cardinal)) and set(full) == set(pm.TEMPLATE_ORDER)
    first = pool.frames_of("cardinal")[0]
    lofo = lc.template_bases(pool.frames, states, exclude=first.frame_id)
    assert lofo["cardinal"][0][0].item() == pytest.approx((sum(cardinal) - pool.frames.index(first)) / (len(cardinal) - 1))
    assert lofo["quantifier"][1][0].item() == full["quantifier"][1][0].item()
    restored = lc.bases_from_json(lc.bases_to_json(full, {template: len(pool.frames_of(template)) for template in pm.TEMPLATE_ORDER}))
    assert all(torch.equal(restored[t][k], full[t][k]) for t in pm.TEMPLATE_ORDER for k in (0, 1))
    with pytest.raises(lc.PhaseError):
        lc.mean_state([])


def test_token_local_model_on_the_fake_is_exact_for_block_one(inputs, monkeypatch):
    manifest, digests, pool = inputs
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    assert lw.layers == (1, 2) and lw.W_in[1].shape == (8, 16) and lw.W_out[2].shape == (16, 8) and lw.W_in[1].dtype == torch.float64
    small = cs.Pool008(pool.frames[:2] + pool.frames_of("quantifier")[:1], pool.frame_origin, pool.tokens[:6], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    cache = pm.PromptCache(model, tuple(small.nouns))
    axes = cs.stage_axes(cache, weights, small)
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
    frame = small.frames[0]
    state = lc.capture_frame(model, head, small.reference_prompt(frame), small.single_nouns, axes["T"])
    assert torch.equal(state.x1, state.ref.vectors["R0"].double()) and state.x2.shape == (8,)
    template = frame.template_id
    plural = lc.measure_pair(model, weights, head, state, pool.plural_cue[template], plural_ids[template], axes["R0"], axes["T"], small.single_nouns)
    name, token_id = next((n, t) for n, t in small.tokens if t not in (pool.reference_ids[template], plural_ids[template]))
    record = lc.measure_pair(model, weights, head, state, name, token_id, axes["R0"], axes["T"], small.single_nouns)
    assert set(record.extra) == {f"RESID_PRE.L1@{frame.p_c}", f"RESID_PRE.L2@{frame.p_c}"}
    analysis = lc.analyse_pair(record, plural, read=read, lw=lw, weights=weights, state=state, axis_T=axes["T"])
    assert analysis is not None
    # Block 1 is exact at the frame's own base (Δx₁ = ΔE), block 2 is reproduced from its captured input, and the neuron terms sum to the MLP reads.
    assert analysis["ladder"]["mlp1_exact_error"] < 1e-6 and analysis["ladder"]["mlp2_identity_error"] < 1e-6
    assert all(analysis["neurons"][key]["identity_error"] < 1e-6 for key in lc.MLP_KEYS)
    assert analysis["c_L"] == pytest.approx(analysis["c_M"] + analysis["c_H"]) and analysis["P1_prime"] == pytest.approx(analysis["g_E"] + analysis["c_L"], rel=1e-6, abs=1e-6)
    assert analysis["own"]["c_hat"] == pytest.approx(analysis["own"]["c_mlp1"] + analysis["own"]["c_mlp2"])
    assert analysis["ladder"]["attention_input_term"] == pytest.approx(analysis["c_M"] - analysis["own"]["c_hat"])
    # The own-reference pair and the plural cue's composite.
    reference = lc.measure_pair(model, weights, head, state, "ref", pool.reference_ids[template], axes["R0"], axes["T"], small.single_nouns)
    assert reference.own_reference and lc.analyse_pair(reference, plural, read=read, lw=lw, weights=weights, state=state, axis_T=axes["T"]) is None
    total = read.plural_total(weights, lw, (state.x1, state.x2), template)
    plural_prediction = read.predict(weights, lw, (state.x1, state.x2), plural_ids[template], template)
    assert plural_prediction["g_E"] == pytest.approx(1.0) and lc.composite(plural_prediction, total["D_hat"]) == pytest.approx(1.0)
    assert plural_prediction["interaction"] == pytest.approx(plural_prediction["c_hat"] - plural_prediction["c_par"] - plural_prediction["c_perp"])
    # The prediction depends on the base state (nonlinear MLPs): a non-uniform perturbation of the base changes ĉ but not g_E (a uniform shift would be removed by LayerNorm).
    tilt = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
    other = read.predict(weights, lw, (state.x1 + tilt, state.x2 - tilt), token_id, template)
    assert other["g_E"] == pytest.approx(analysis["own"]["g_E"]) and other["c_hat"] != pytest.approx(analysis["own"]["c_hat"])
    with pytest.raises(pm.IncidentError):
        lc.neuron_ledger(read, lw, 2, state.x2, state.x2 + 1.0, read.denominator(weights, template), expected=1e3)
    d1, d2 = lc.propagate(lw, (state.x1, state.x2), torch.zeros(8, dtype=torch.float64))
    assert float(d1.abs().max()) == 0.0 and float(d2.abs().max()) == 0.0


def _analysis(token, template, c_L, *, c_M=None, P1=None, g_E=0.5, own_c=None):
    c_M = c_L if c_M is None else c_M
    return {"token": token, "template": template, "c_L": c_L, "c_M": c_M, "c_H": c_L - c_M, "P1": (g_E + c_L) if P1 is None else P1, "P1_prime": g_E + c_L, "q_T": g_E + c_L, "g_E": g_E,
            "fractions_010": {"f_layers": c_L}, "own": {"c_hat": c_L if own_c is None else own_c, "q_hat": g_E + c_L}, "ladder": {"attention_input_term": 0.0}, "dc": -1.0, "dc_beh": -1.0}


def _lock(tokens, tau_c=0.2, tau_P=0.15, scale=1.0, shift=0.0, g_E=0.5):
    predictions = {}
    for word, c in tokens.items():
        by_template = {t: {"c_hat": scale * c + shift, "c_mlp1": 0.0, "c_mlp2": scale * c + shift, "c_par": 0.0, "c_perp": scale * c + shift, "interaction": 0.0, "g_E": g_E, "q_hat_prime": g_E + scale * c + shift, "q_hat": g_E + scale * c + shift} for t in pm.TEMPLATE_ORDER}
        predictions[word] = {"by_template": by_template, "means": {k: by_template["cardinal"][k] for k in by_template["cardinal"]}, "licensed_frames": [], "category": "x", "token_id": 1}
    return {"predictions": {"tokens": predictions, "plural_totals": {}}, "tolerances": {"tau_c": tau_c, "tau_P": tau_P}, "defined_templates": list(pm.TEMPLATE_ORDER), "exposed_check": {"heads_share": {"share": 0.2}}, "exposed_neurons": {}}


class _Confirmation:
    def __init__(self, frames, tokens):
        self.frames = [pm.Frame(t, fid, (1, 2), () if t in pm.CUE_FINAL_TEMPLATES else (7,), {"sg": 5, "pl": 6}, "x {cue}" + ("" if t in pm.CUE_FINAL_TEMPLATES else " y")) for fid, t in frames]
        self.tokens = [{"word": w, "category": "x", "licensed_frames": [fid for fid, _ in frames]} for w in tokens]


def test_scoring_rules_on_synthetic_tables():
    frames = [("c1", "cardinal"), ("c2", "cardinal"), ("q1", "quantifier"), ("q2", "quantifier"), ("a1", "coordinated-adjective"), ("a2", "coordinated-adjective")]
    templates = dict(frames)
    words = [f"w{i}" for i in range(18)]
    c_values = {w: 0.05 * i for i, w in enumerate(words)}
    confirmation = _Confirmation(frames, words)
    frames_out = {fid: {"template_id": t, "valid": True} for fid, t in frames}
    perfect = {w: {fid: _analysis(w, templates[fid], c_values[w]) for fid, _ in frames} for w in words}
    results = lc.score_confirmation(frames_out, perfect, confirmation, _lock(c_values))
    assert results["precondition"]["passed"] and results["Y1"]["passed"] and results["Y2"]["passed"] and results["Y1"]["r2"] == pytest.approx(1.0)
    assert results["outcome"]["label"] == "LAYER_CORRECTION_TOKEN_LOCAL_MLP | COMPOSITE_PREDICTS_P1" and results["descriptive"]["heads_share"]["share"] == 0.0
    # Spearman only: a small rank scramble with a generous τ_c (R² stays high because the scramble is small).
    scrambled = {w: {fid: _analysis(w, templates[fid], c_values[words[(i + 1) % 18]] if i % 2 == 0 else c_values[words[i - 1]]) for fid, _ in frames} for i, w in enumerate(words)}
    verdict = lc.score_confirmation(frames_out, scrambled, confirmation, _lock(c_values, tau_c=2.0))
    assert verdict["Y1"]["failing"] == ["spearman"] or verdict["Y1"]["failing"] == []  # adjacent swaps may keep Spearman ≥ 0.80
    reversed_values = {w: {fid: _analysis(w, templates[fid], c_values[words[17 - i]]) for fid, _ in frames} for i, w in enumerate(words)}
    assert "spearman" in lc.score_confirmation(frames_out, reversed_values, confirmation, _lock(c_values, tau_c=2.0))["Y1"]["failing"]
    # MAE only: a uniform shift larger than τ_c but small relative to the spread keeps Spearman 1 and R² ≥ 0.5.
    shifted = {w: {fid: _analysis(w, templates[fid], c_values[w] + 0.15) for fid, _ in frames} for w in words}
    assert lc.score_confirmation(frames_out, shifted, confirmation, _lock(c_values, tau_c=0.12))["Y1"]["failing"] == ["mae"]
    # R² only: a perfectly ordered half-scale prediction with a generous τ_c passes Spearman and MAE but fails the magnitude floor.
    compressed = lc.score_confirmation(frames_out, perfect, confirmation, _lock(c_values, tau_c=0.5, scale=0.5))
    assert compressed["Y1"]["failing"] == ["r2"] and compressed["Y1"]["spearman"] == pytest.approx(1.0) and compressed["Y1"]["r2"] < lc.Y1_R2
    assert compressed["outcome"]["Y1"] == "LAYER_CORRECTION_NOT_TOKEN_LOCAL"
    # Y2 fails when the measured P1 fraction does not track the composite.
    bad_p1 = {w: {fid: _analysis(w, templates[fid], c_values[w], P1=1.0 - c_values[w]) for fid, _ in frames} for w in words}
    assert not lc.score_confirmation(frames_out, bad_p1, confirmation, _lock(c_values))["Y2"]["passed"]
    # Precondition: three valid frames → fail; fifteen scored tokens → fail; two valid frames → token not scored.
    three = {fid: {"template_id": t, "valid": fid in ("c1", "q1", "a1")} for fid, t in frames}
    partial = {w: {fid: _analysis(w, templates[fid], c_values[w]) for fid, _ in frames if fid in ("c1", "q1", "a1")} for w in words}
    assert lc.score_confirmation(three, partial, confirmation, _lock(c_values))["outcome"]["label"] == "PRECONDITION_FAILED"
    fifteen = {w: {fid: _analysis(w, templates[fid], c_values[w]) for fid, _ in frames} for w in words[:15]} | {w: {} for w in words[15:]}
    assert lc.score_confirmation(frames_out, fifteen, confirmation, _lock(c_values))["precondition"]["scored_tokens"] == 15
    two_frames = {w: {fid: _analysis(w, templates[fid], c_values[w]) for fid, _ in frames} for w in words[:17]} | {words[17]: {fid: _analysis(words[17], templates[fid], 0.5) for fid, _ in frames[:2]}}
    scored = lc.score_confirmation(frames_out, two_frames, confirmation, _lock(c_values))
    assert not scored["tokens"][words[17]]["scored"] and scored["precondition"]["scored_tokens"] == 17 and scored["precondition"]["passed"]
    # Predicted and measured means use the same frame set: a token valid only in two cardinal frames and one quantifier frame gets the frame-weighted locked mean.
    lock = _lock(c_values)
    lock["predictions"]["tokens"][words[0]]["by_template"]["cardinal"]["c_hat"] = 0.9
    subset = {w: {fid: _analysis(w, templates[fid], c_values[w]) for fid, _ in frames} for w in words}
    subset[words[0]] = {"c1": _analysis(words[0], "cardinal", 0.1), "c2": _analysis(words[0], "cardinal", 0.1), "q1": _analysis(words[0], "quantifier", 0.1)}
    row = lc.score_confirmation(frames_out, subset, confirmation, lock)["tokens"][words[0]]
    assert row["frames"] == ["c1", "c2", "q1"] and row["c_hat_locked_mean"] == pytest.approx((0.9 + 0.9 + 0.0) / 3) and row["c_L_mean"] == pytest.approx(0.1)
    assert row["base_point_term"] == pytest.approx(row["c_hat_own_mean"] - row["c_hat_locked_mean"])


def test_lock_reproduction_refusal():
    prediction = {"c_hat": 0.3, "c_mlp1": 0.1, "c_mlp2": 0.2, "c_par": 0.25, "c_perp": 0.04, "interaction": 0.01, "g_E": 0.5, "q_hat_prime": 0.8, "r_E": 1.0, "r_delta": 0.6, "q_hat": 0.75}
    lock = {"predictions": {"tokens": {"w": {"by_template": {"cardinal": prediction}, "means": {"c_hat": 0.3}}}, "plural_totals": {"cardinal": {"denominator": 2.0, "c_hat_plural": 0.05, "D_hat": 2.1}}}}
    assert lc.assert_lock_predictions_reproduced(lock, lock["predictions"]) == 0.0
    perturbed = json.loads(json.dumps(lock["predictions"]))
    perturbed["plural_totals"]["cardinal"]["D_hat"] += 1e-6
    with pytest.raises(lc.PhaseError, match="nothing was executed"):
        lc.assert_lock_predictions_reproduced(lock, perturbed)
    perturbed = json.loads(json.dumps(lock["predictions"]))
    perturbed["tokens"]["w"]["by_template"]["cardinal"]["c_perp"] += 1e-6
    with pytest.raises(lc.PhaseError):
        lc.assert_lock_predictions_reproduced(lock, perturbed)


def test_locked_axes_and_read_weight_checks():
    direction_T = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    direction_E = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    lock = {"axes_vectors": {"T": direction_T.tolist(), "R0": direction_E.tolist()}, "sigma_T": 2.0}
    recomputed = {"T": pm.SiteAxis("T", torch.zeros(4), direction_T.clone(), 2.0), "R0": pm.SiteAxis("R0", torch.zeros(4), direction_E.clone(), 1.0)}
    axis_T, e_axis = lc.locked_axes(lock, recomputed)
    assert axis_T.sigma == 2.0 and torch.equal(axis_T.direction, direction_T) and torch.equal(e_axis.direction, direction_E)
    with pytest.raises(pm.IncidentError, match="σ_T"):
        lc.locked_axes({**lock, "sigma_T": 2.1}, recomputed)
    tilted = {"T": pm.SiteAxis("T", torch.zeros(4), torch.tensor([0.9, 0.436, 0.0, 0.0], dtype=torch.float64), 2.0), "R0": recomputed["R0"]}
    with pytest.raises(pm.IncidentError, match="directions"):
        lc.locked_axes(lock, tilted)
