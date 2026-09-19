"""Experiment 013: the frozen-pattern head model, the exact per-head split, the ladder, the confirmation policy, the floors with the degenerate-spread guard, and the extract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_decompiler import attention_paths as ap
from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import encoding_read as er
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from plural_fakes import TinyPlural
from test_layer_correction import toy_tokenizer_012

ROOT = Path(__file__).resolve().parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


def test_frozen_constants_match_the_design():
    assert ap.Y_SPEARMAN == 0.80 and ap.Y_R2 == 0.50 and ap.TAU_R == 0.070 and ap.TAU_A == 0.106 and ap.TAU_MIN == 0.05 and ap.TAU_MULTIPLIER == 3.0
    assert ap.DESIGN_RMSE_R == 0.0233 and ap.DESIGN_RMSE_A == 0.0355 and ap.tau_rule(ap.DESIGN_RMSE_R) == pytest.approx(0.0699, abs=1e-4) and ap.tau_rule(ap.DESIGN_RMSE_A) == pytest.approx(0.1065, abs=1e-4)
    assert ap.Y3_SPEARMAN_H == 0.70 and ap.Y3_SPEARMAN_M == 0.90 and ap.C_H_DEGENERATE_SD == 0.017 and ap.COMPENSATION_MIN == 0.05
    assert ap.MIN_VALID_FRAMES == 4 and ap.MIN_VALID_FRAMES_PER_TOKEN == 3 and ap.MIN_SCORED_TOKENS == 16 and ap.FRAME_CUE_EFFECT_RATE == 108 / 120
    assert ap.RUNTIME_SEED == 20260916 and ap.CONTROL_SEED == 20260924 and len(ap.FRESH_FRAMES) == 6 and ap.HEAD_LAYERS == (1, 2) and len(ap.HEAD_KEYS) == 16
    assert ap.QUOTAS == {"determiner-like": 5, "numeral": 5, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 5} and ap.EXPECTED_LEDGER_SIZE == 2724
    exposed = {token["word"] for path in (er.CONFIRMATION_RELATIVE_PATH, lc.CONFIRMATION_RELATIVE_PATH) for token in json.loads((ROOT / path).read_text())["tokens"]}
    assert not (set(sum(ap.CANDIDATES.values(), ())) & exposed), "013 candidates must not be earlier frozen tokens"
    assert ap.OUTCOME_Y3 == ("ATTRIBUTION_PREDICTED", "ATTRIBUTION_NOT_PREDICTED", "ATTRIBUTION_NOT_EVALUABLE")


def test_compensation_rule_spread_and_outcome():
    assert ap.is_compensation_case(0.2, -0.1) and ap.is_compensation_case(-0.06, 0.06) and not ap.is_compensation_case(0.2, 0.1) and not ap.is_compensation_case(0.04, -0.2) and not ap.is_compensation_case(0.2, -0.05)
    assert ap.spread([1.0, 1.0, 1.0]) == 0.0 and ap.spread([1.0]) is None and ap.spread([0.0, 2.0]) == pytest.approx(1.0)
    assert ap.rmse([1.0, 2.0], [1.0, 4.0]) == pytest.approx((2.0) ** 0.5) and ap.rmse([], []) is None
    base = {"precondition_Y1": {"passed": True}, "precondition_Y2": {"passed": True}, "Y1": {"test": {"passed": True}}, "Y2": {"test": {"passed": False}}, "Y3": {"evaluable": True, "passed": True}}
    assert ap.outcome(base)["label"] == "RESIDUAL_PREDICTED_TOKENS | RESIDUAL_NOT_PREDICTED_FRAMES_CONDITIONAL | ATTRIBUTION_PREDICTED"
    assert ap.outcome({**base, "precondition_Y2": {"passed": False}})["Y2"] == "PRECONDITION_FAILED_FRAMES"
    assert ap.outcome({**base, "precondition_Y1": {"passed": False}})["Y1"] == "PRECONDITION_FAILED_TOKENS"
    assert ap.outcome({**base, "Y3": {"evaluable": False, "passed": False}})["Y3"] == "ATTRIBUTION_NOT_EVALUABLE"
    assert ap.outcome({**base, "Y3": {"evaluable": True, "passed": False}})["Y3"] == "ATTRIBUTION_NOT_PREDICTED"


def _entry(value, c_012=0.1):
    return {"c_L": value, "c_M": value, "c_H": 0.0, "c_k": {key: 0.0 for key in ra.COMPONENT_ORDER}, "c_own": value, "c_012": c_012, "P1": value, "q_T": value, "g_E": 0.5}


def test_ledger_replication_check():
    recorded = {"a|f": _entry(0.5)}
    assert ap.check_ledger_replication({"a|f": _entry(0.5), "b|f": _entry(0.1)}, recorded)["passed"]
    with pytest.raises(pm.IncidentError, match="deviates"):
        ap.check_ledger_replication({"a|f": _entry(0.5 + 2e-6)}, recorded)
    with pytest.raises(pm.IncidentError, match="012 model"):
        ap.check_ledger_replication({"a|f": _entry(0.5, c_012=0.1 + 1e-6)}, recorded)
    with pytest.raises(pm.IncidentError, match="lacks"):
        ap.check_ledger_replication({}, recorded)


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
    lock_012 = json.loads((ROOT / lc.LOCK_RELATIVE_PATH).read_text())
    digests["lock_012"] = lock_012["content_sha256"]
    pool = ap.build_pool_013(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012)
    return manifest, digests, pool


def toy_tokenizer_013(manifest, pool):
    """The 012 toy tokenizer plus the 013 candidates and frame words, and a placeholder string for every real token ID of the exposed frames so their prompts round-trip."""
    base = toy_tokenizer_012(manifest)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in ap.CANDIDATES.values() for word in words]
    for _, text in ap.FRESH_FRAMES:
        for index, piece in enumerate(text.replace("{cue}", "").split()):
            words.append(piece if index == 0 else " " + piece)
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    known = set(vocabulary.values())
    for frame in pool.frames:
        for token_id in (*frame.prefix_ids, *frame.suffix_ids, *frame.cue_ids.values()):
            if token_id not in known:
                vocabulary[f"⟨{token_id}⟩"] = token_id
                known.add(token_id)
    return type(base)(vocabulary)


def test_pool_013_and_committed_extract(inputs):
    manifest, digests, pool = inputs
    assert len(pool.tokens) == 111 and len(pool.frames) == 36 and pool.token_source["thy"] == "confirmation-012" and pool.frame_origin["cardinal-012-1"] == "confirmation-012"
    ledger = ap.load_inherited_ledger(ROOT / ap.INHERITED_012_LEDGER_RELATIVE_PATH, digests=digests, expected_size=ap.EXPECTED_LEDGER_SIZE)
    keys = set(ledger["entries"])
    assert sum(1 for key in keys if key.split("|")[1].endswith(("-012-1", "-012-2"))) == 144 and all(set(v["c_k"]) == set(ra.COMPONENT_ORDER) for v in ledger["entries"].values())
    assert "thy|cardinal-012-1" in keys and "cardinal:sg|cardinal-1" not in keys and "cardinal:pl|cardinal-1" in keys  # own-reference pairs were never recorded
    with pytest.raises(ValueError):
        ap.load_inherited_ledger(ROOT / ap.INHERITED_012_LEDGER_RELATIVE_PATH, digests=digests, expected_size=10)


def test_confirmation_policy_with_two_prompt_lists(inputs, tmp_path):
    manifest, digests, pool = inputs
    tokenizer = toy_tokenizer_013(manifest, pool)
    payload = ap.build_confirmation_payload(tokenizer, pool, digests)
    words = [token["word"] for token in payload["tokens"]]
    assert len(words) == 24 and words[:5] == ["same", "own", "last", "next", "first"] and words[5:10] == ["zero", "thousand", "million", "billion", "trillion"]
    assert len(payload["token_prompts"]) == 24 * 6 and len(payload["exposed_frame_prompts"]) == 24 * 36 and payload["exposed_frame_ids"] == [frame.frame_id for frame in pool.frames]
    confirmation = ap.validate_confirmation(payload, pool, digests)
    assert len(confirmation.all_prompts) == 12 + 144 + 864 and len(confirmation.frames_for("same")) == 6
    path = tmp_path / "c.json"
    assert ap.freeze_confirmation(path, tokenizer, pool, digests) == payload["content_sha256"]
    with pytest.raises(FileExistsError):
        ap.freeze_confirmation(path, tokenizer, pool, digests)
    tampered = dict(payload)
    tampered["exposed_frame_prompts"] = payload["exposed_frame_prompts"][:-1]
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in tampered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError):
        ap.validate_confirmation(tampered, pool, digests)


def test_frozen_pattern_model_and_exact_head_split_on_the_fake(inputs, monkeypatch):
    manifest, digests, pool = inputs
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    assert set(heads.heads) == set(ap.HEAD_KEYS) and heads.keys_of(1) == [f"L01.H0{h}" for h in range(8)]
    small = cs.Pool008(pool.frames[:2] + pool.frames_of("quantifier")[:1], pool.frame_origin, pool.tokens[:6], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    cache = pm.PromptCache(model, tuple(small.nouns))
    axes = cs.stage_axes(cache, weights, small)
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
    states = {}
    for frame in small.frames:
        states[frame.frame_id] = ap.capture_frame_013(model, head, small.reference_prompt(frame), small.single_nouns, axes["T"])
    frame = small.frames[0]
    state = states[frame.frame_id]
    assert len(state.x1_all) == frame.p_c + 1 and state.A1.shape[0] == 8 and state.A1.shape[1] > frame.p_c and abs(float(state.A1[0][: frame.p_c + 1].sum()) - 1.0) < 1e-5
    bases_012 = lc.template_bases(small.frames, {fid: (s.x1, s.x2) for fid, s in states.items()})
    fpm = ap.FrozenPatternModel(read, lw, heads, bases_012)
    template = frame.template_id
    plural = ap.measure_pair_013(model, weights, head, state, pool.plural_cue[template], plural_ids[template], axes["R0"], axes["T"], small.single_nouns)
    name, token_id = next((n, t) for n, t in small.tokens if t not in (pool.reference_ids[template], plural_ids[template]))
    record = ap.measure_pair_013(model, weights, head, state, name, token_id, axes["R0"], axes["T"], small.single_nouns)
    assert {f"ATTN_PATTERN.L1@{frame.p_c}", f"ATTN_PATTERN.L2@{frame.p_c}", f"L01.H03@{frame.p_c}", f"L02.H07@{frame.p_c}"} <= set(record.extra)
    # The exact split per head: frozen-pattern term (measured arriving change) + pattern-change term = captured head output change.
    for key in ap.HEAD_KEYS:
        identity = ap.head_identity(heads, key, state, record)
        assert identity["error"] < 1e-5
    analysis = ap.analyse_pair_013(record, plural, model=fpm, weights=weights, state=state, axis_T=axes["T"])
    assert analysis is not None and analysis["head_identity_max_error"] < 1e-5
    p, ladder = analysis["prediction"], analysis["ladder"]
    assert p["c_L_hat"] == pytest.approx(p["c_M_hat"] + p["c_H_hat"]) and p["c_H_hat"] == pytest.approx(sum(p["head_terms"].values()))
    assert p["r_hat"] == pytest.approx(p["c_L_hat"] - p["c_012"]) and ladder["base_point"] + ladder["frozen_attention"] + ladder["remainder"] == pytest.approx(analysis["r"])
    assert ladder["remainder"] == pytest.approx(ladder["remainder_direct_pattern_change"] + ladder["remainder_indirect"]) and analysis["c_L"] == pytest.approx(analysis["c_M"] + analysis["c_H"])
    # Level 1 (the 012 model at the frame's own base) equals the frozen-pattern model with every pattern weight zero, and c_012 is the model at the locked base.
    zero = fpm.predict(weights, state.x1, state.x2, {key: 0.0 for key in ap.HEAD_KEYS}, token_id, template)
    assert zero["c_L_hat"] == pytest.approx(p["c_own"]) and zero["c_H_hat"] == 0.0
    assert p["c_012"] == pytest.approx(read.predict(weights, lw, bases_012[template], token_id, template)["c_hat"])
    # From the locked state the prediction is identical.
    locked = state.locked_state()
    from_locked = fpm.predict_from_locked(weights, locked, token_id, template)
    assert from_locked["c_L_hat"] == pytest.approx(p["c_L_hat"], abs=1e-12) and ap.state_digest(locked) == ap.state_digest(json.loads(json.dumps(locked)))
    row = ap.prediction_row(name, frame.frame_id, template, p)
    assert set(ap.PREDICTION_COLUMNS) | {"compensation_case"} == set(row)
    # The own-reference pair is not analysed; a wrong pattern row makes the identity an incident.
    reference = ap.measure_pair_013(model, weights, head, state, "ref", pool.reference_ids[template], axes["R0"], axes["T"], small.single_nouns)
    assert ap.analyse_pair_013(reference, plural, model=fpm, weights=weights, state=state, axis_T=axes["T"]) is None
    broken = ap.FrameState013(state.ref, state.components, state.functional, state.x1, state.x2, state.x1_all, state.x2_all, torch.roll(state.A1, 1, dims=1), state.A2)
    with pytest.raises(pm.IncidentError, match="do not sum"):
        ap.head_identity(heads, "L01.H00", broken, record)


def _analysis(token, frame_id, template, c_L, c_012, *, c_M=None, c_H=None, c_own=None):
    c_M = c_L if c_M is None else c_M
    c_H = c_L - c_M if c_H is None else c_H
    own = c_L if c_own is None else c_own
    prediction = {"c_012": c_012, "c_own": own, "base_point": own - c_012, "frozen_attention": c_L - own, "r_hat": c_L - c_012, "c_M_hat": c_M, "c_H_hat": c_H, "c_L_hat": c_L, "c_mlp1_hat": 0.0, "c_mlp2_hat": c_M, "g_E": 0.5, "head_terms": {k: 0.0 for k in ap.HEAD_KEYS}}
    return {"token": token, "frame_id": frame_id, "template": template, "c_L": c_L, "c_M": c_M, "c_H": c_H, "c_k": {k: 0.0 for k in ra.COMPONENT_ORDER}, "P1": 0.5 + c_L, "P1_prime": 0.5 + c_L, "q_T": 0.5 + c_L, "g_E": 0.5, "f_layers_010": c_L,
            "prediction": prediction, "r": c_L - c_012, "ladder": {"base_point": own - c_012, "frozen_attention": c_L - own, "remainder": 0.0, "remainder_direct_pattern_change": 0.0, "remainder_indirect": 0.0, "attention_input_term": 0.0, "mlp1_exact_error": 0.0, "mlp2_identity_error": 0.0},
            "head_identity_max_error": 0.0, "delta_A_pc": {}, "identities": {}, "compensation_case": ap.is_compensation_case(c_M, c_H)}


def _row(token, frame_id, template, r_hat, c_012, *, c_M_hat=None, c_H_hat=0.0, scale=1.0, shift=0.0):
    c_L_hat = c_012 + scale * r_hat + shift
    c_M_hat = c_L_hat - c_H_hat if c_M_hat is None else c_M_hat
    prediction = {"c_012": c_012, "c_own": c_012, "base_point": 0.0, "frozen_attention": c_L_hat - c_012, "r_hat": c_L_hat - c_012, "c_M_hat": c_M_hat, "c_H_hat": c_H_hat, "c_L_hat": c_L_hat, "c_mlp1_hat": 0.0, "c_mlp2_hat": c_M_hat, "g_E": 0.5, "head_terms": {k: 0.0 for k in ap.HEAD_KEYS}}
    return ap.prediction_row(token, frame_id, template, prediction)


class _Confirmation:
    def __init__(self, fresh_frames, tokens):
        self.frames = [pm.Frame(t, fid, (1, 2), () if t in pm.CUE_FINAL_TEMPLATES else (7,), {"sg": 5, "pl": 6}, "x {cue}" + ("" if t in pm.CUE_FINAL_TEMPLATES else " y")) for fid, t in fresh_frames]
        self.tokens = [{"word": w, "category": "x", "licensed_frames": [fid for fid, _ in fresh_frames]} for w in tokens]


def test_scoring_rules_and_the_degenerate_guard_on_synthetic_tables():
    exposed = [(f"e{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(6)]
    fresh = [("c1", "cardinal"), ("c2", "cardinal"), ("q1", "quantifier"), ("q2", "quantifier"), ("a1", "coordinated-adjective"), ("a2", "coordinated-adjective")]
    words = [f"w{i}" for i in range(18)]
    r_values = {w: 0.02 * i - 0.1 for i, w in enumerate(words)}  # residual spread comparable to the exposed 0.065
    c_012 = 0.2
    confirmation = _Confirmation(fresh, words)

    def tables(scale=1.0, shift=0.0, c_H_hat=0.0):
        locked = [_row(w, fid, t, r_values[w], c_012, scale=scale, shift=shift, c_H_hat=c_H_hat) for fid, t in exposed for w in words]
        stage1_rows = [_row(w, fid, t, r_values[w], c_012, scale=scale, shift=shift, c_H_hat=c_H_hat) for fid, t in fresh for w in words]
        return {"predictions": {"rows": locked}}, {"rows": stage1_rows, "frames": {fid: {"template_id": t, "valid": True, "template_defined": True} for fid, t in fresh}}

    def measured(frames, *, c_H=None, c_H_by_token=None):
        out = {}
        for fid, t in frames:
            for w in words:
                h = c_H if c_H is not None else (c_H_by_token[w] if c_H_by_token else 0.0)
                out[f"{w}|{fid}"] = _analysis(w, fid, t, c_012 + r_values[w], c_012, c_M=c_012 + r_values[w] - h, c_H=h)
        return out

    lock, stage1 = tables()
    # Perfect residual prediction, zero heads: Y1 and Y2 pass; Y3 is non-evaluable because the measured c_H has no spread.
    results = ap.score_confirmation(stage1, measured(exposed), measured(fresh), confirmation, lock)
    assert results["Y1"]["test"]["passed"] and results["Y2"]["test"]["passed"] and results["Y1"]["test"]["r2"] == pytest.approx(1.0)
    assert not results["Y3"]["evaluable"] and results["outcome"]["label"] == "RESIDUAL_PREDICTED_TOKENS | RESIDUAL_PREDICTED_FRAMES_CONDITIONAL | ATTRIBUTION_NOT_EVALUABLE"
    # Heads with spread and a matching prediction: Y3 passes.
    c_H_by_token = {w: 0.05 * ((i % 4) - 1.5) for i, w in enumerate(words)}
    lock_h = {"predictions": {"rows": [_row(w, fid, t, r_values[w], c_012, c_H_hat=c_H_by_token[w]) for fid, t in exposed for w in words]}}
    stage1_h = {"rows": [_row(w, fid, t, r_values[w], c_012, c_H_hat=c_H_by_token[w]) for fid, t in fresh for w in words], "frames": stage1["frames"]}
    results = ap.score_confirmation(stage1_h, measured(exposed, c_H_by_token=c_H_by_token), measured(fresh, c_H_by_token=c_H_by_token), confirmation, lock_h)
    assert results["Y3"]["evaluable"] and results["Y3"]["passed"] and results["Y3"]["c_H"]["spearman"] == pytest.approx(1.0) and results["Y3"]["n_pairs"] == 18 * 12
    assert results["Y3"]["compensation"]["n"] == 60 and results["Y3"]["compensation"]["fraction_signs_as_predicted"] == 1.0 and results["outcome"]["Y3"] == "ATTRIBUTION_PREDICTED"
    # Y3 fails on the c_H ordering when the predicted heads are anti-ordered.
    anti = {"rows": [_row(w, fid, t, r_values[w], c_012, c_H_hat=-c_H_by_token[w]) for fid, t in fresh for w in words], "frames": stage1["frames"]}
    lock_anti = {"predictions": {"rows": [_row(w, fid, t, r_values[w], c_012, c_H_hat=-c_H_by_token[w]) for fid, t in exposed for w in words]}}
    assert "c_H_spearman" in ap.score_confirmation(anti, measured(exposed, c_H_by_token=c_H_by_token), measured(fresh, c_H_by_token=c_H_by_token), confirmation, lock_anti)["Y3"]["failing"]
    # A compressed residual prediction (0.4 × the true scale) keeps the ordering and stays under τ_r but fails R² (and only R²).
    lock_c, stage1_c = tables(scale=0.4)
    compressed = ap.score_confirmation(stage1_c, measured(exposed), measured(fresh), confirmation, lock_c)
    assert compressed["Y1"]["test"]["failing"] == ["r2"] and compressed["Y1"]["test"]["spearman"] == pytest.approx(1.0) and compressed["Y1"]["test"]["mae"] <= ap.TAU_R
    # A uniform shift just beyond τ_r (small against the spread, so R² stays above 0.50) fails MAE only.
    lock_s, stage1_s = tables(shift=0.072)
    assert ap.score_confirmation(stage1_s, measured(exposed), measured(fresh), confirmation, lock_s)["Y2"]["test"]["failing"] == ["mae"]
    # Reversed ordering fails Spearman.
    reversed_lock = {"predictions": {"rows": [_row(w, fid, t, r_values[words[17 - i]], c_012) for fid, t in exposed for i, w in enumerate(words)]}}
    assert "spearman" in ap.score_confirmation(stage1, measured(exposed), measured(fresh), confirmation, reversed_lock)["Y1"]["test"]["failing"]
    # Y2 precondition: three valid fresh frames → PRECONDITION_FAILED_FRAMES while Y1 is unaffected.
    three = {"rows": stage1["rows"], "frames": {fid: {"template_id": t, "valid": fid in ("c1", "q1", "a1"), "template_defined": True} for fid, t in fresh}}
    partial = {k: v for k, v in measured(fresh).items() if k.split("|")[1] in ("c1", "q1", "a1")}
    verdict = ap.score_confirmation(three, measured(exposed), partial, confirmation, lock)
    assert verdict["outcome"]["Y2"] == "PRECONDITION_FAILED_FRAMES" and verdict["outcome"]["Y1"] == "RESIDUAL_PREDICTED_TOKENS"
    # Identical frame sets: a token measured in only two exposed frames is not scored for Y1.
    two = {k: v for k, v in measured(exposed).items() if not (k.startswith("w0|") and k.split("|")[1] not in ("e0", "e1"))}
    scored = ap.score_confirmation(stage1, two, measured(fresh), confirmation, lock)
    assert not scored["Y1"]["tokens"]["w0"]["scored"] and len(scored["Y1"]["scored_tokens"]) == 17


def test_stage_one_digest_and_lock_reproduction():
    rows = [_row("w", "f", "cardinal", 0.1, 0.2)]
    states = {"f": {"x1": [0.0], "x2": [1.0], "A_pc": {k: 0.1 for k in ap.HEAD_KEYS}, "p_c": 3}}
    stage1 = {"rows": rows, "states": states, "digest": ap.table_digest(rows, states)}
    ap.assert_stage_one_digest(stage1)
    with pytest.raises(ap.PhaseError, match="digest"):
        ap.assert_stage_one_digest({**stage1, "digest": "0" * 64})
    tampered_rows = json.loads(json.dumps(rows))
    tampered_rows[0]["r_hat"] += 1e-9
    with pytest.raises(ap.PhaseError):
        ap.assert_stage_one_digest({**stage1, "rows": tampered_rows})
    lock = {"predictions": {"rows": rows, "c_012_by_template": {"w": {"cardinal": 0.2}}}}
    assert ap.assert_lock_predictions_reproduced(lock, lock["predictions"]) == 0.0
    perturbed = json.loads(json.dumps(lock["predictions"]))
    perturbed["rows"][0]["head_terms"]["L02.H05"] += 1e-6
    with pytest.raises(ap.PhaseError, match="nothing was executed"):
        ap.assert_lock_predictions_reproduced(lock, perturbed)
    with pytest.raises(ap.PhaseError, match="different number"):
        ap.assert_lock_predictions_reproduced(lock, {"rows": [], "c_012_by_template": {}})
