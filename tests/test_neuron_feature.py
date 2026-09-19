"""Experiment 014: neuron 1987's exact computation, the predictor and its Jacobian form, the axis alternative, the invariant, balanced accuracy with its guard, the confirmation policy, floors, and the extract."""

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
from neural_decompiler import neuron_feature as nf
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from plural_fakes import TinyPlural
from test_attention_paths import toy_tokenizer_013

ROOT = Path(__file__).resolve().parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


def test_frozen_constants_match_the_design():
    assert nf.NEURON == 1987 and nf.NEURON_LAYER == 2 and nf.FIRING_THRESHOLD == 0.5
    assert nf.Y_SPEARMAN == 0.80 and nf.Y_R2 == 0.50 and nf.Y_BALANCED_ACCURACY == 0.90 and nf.Y3_R2_MAX == 0.30 and nf.Y3_BALANCED_ACCURACY_MAX == 0.70
    assert nf.MIN_VALID_FRAMES == 4 and nf.MIN_VALID_FRAMES_PER_TOKEN == 3 and nf.MIN_SCORED_TOKENS == 16 and nf.FRAME_CUE_EFFECT_RATE == 108 / 120
    assert nf.RUNTIME_SEED == 20260916 and nf.CONTROL_SEED == 20260924 and len(nf.FRESH_FRAMES) == 6 and nf.EXPECTED_LEDGER_SIZE == 3732
    assert nf.QUOTAS == {"determiner-like": 5, "numeral": 5, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 5}
    exposed = {token["word"] for path in (er.CONFIRMATION_RELATIVE_PATH, lc.CONFIRMATION_RELATIVE_PATH, ap.CONFIRMATION_RELATIVE_PATH) for token in json.loads((ROOT / path).read_text())["tokens"]}
    assert not (set(sum(nf.CANDIDATES.values(), ())) & exposed), "014 candidates must not be earlier frozen tokens"
    assert nf.OUTCOME_Y3 == ("AXIS_ONLY_REJECTED", "AXIS_ONLY_NOT_REJECTED", "AXIS_ONLY_NOT_EVALUABLE") and nf.CLASSIFICATION_NOT_EVALUABLE_SUFFIX == "_CLASSIFICATION_NOT_EVALUABLE"
    assert set(nf.frozen_floors()) == {"y_spearman", "y_r2", "y_balanced_accuracy", "y3_r2_max", "y3_balanced_accuracy_max", "firing_threshold", "min_valid_frames", "min_valid_frames_per_token", "min_scored_tokens", "frame_cue_effect_rate", "head_stage_floor"}


def test_balanced_accuracy_guard_and_outcome():
    c = nf.balanced_accuracy([True, True, False, False, True], [True, False, False, False, True])
    assert c["tp"] == 2 and c["fp"] == 1 and c["tn"] == 2 and c["fn"] == 0 and c["sensitivity"] == 1.0 and c["specificity"] == pytest.approx(2 / 3) and c["balanced_accuracy"] == pytest.approx(5 / 6) and c["evaluable"]
    assert c["raw_accuracy"] == pytest.approx(0.8) and c["majority_baseline"] == pytest.approx(0.6)
    degenerate = nf.balanced_accuracy([True, False], [False, False])
    assert degenerate["balanced_accuracy"] is None and not degenerate["evaluable"] and degenerate["specificity"] == 0.5
    assert nf.gelu(0.0) == 0.0 and nf.gelu(2.0) == pytest.approx(1.9545, abs=1e-3) and nf.fires(0.5) and not nf.fires(0.49)
    base = {"precondition_Y1": {"passed": True}, "precondition_Y2": {"passed": True}, "Y1": {"test": {"passed": True, "classification_evaluable": True}}, "Y2": {"test": {"passed": False, "classification_evaluable": False}},
            "Y3": {"evaluable": True, "rejected": True}}
    o = nf.outcome(base)
    assert o["label"] == "NEURON_FEATURE_PREDICTED_TOKENS | NEURON_FEATURE_NOT_PREDICTED_FRAMES_CONDITIONAL_CLASSIFICATION_NOT_EVALUABLE | AXIS_ONLY_REJECTED"
    assert nf.outcome({**base, "Y3": {"evaluable": False, "rejected": False}})["Y3"] == "AXIS_ONLY_NOT_EVALUABLE" and nf.outcome({**base, "Y3": {"evaluable": True, "rejected": False}})["Y3"] == "AXIS_ONLY_NOT_REJECTED"
    assert nf.outcome({**base, "precondition_Y2": {"passed": False}})["Y2"] == "PRECONDITION_FAILED_FRAMES" and nf.outcome({**base, "precondition_Y1": {"passed": False}})["Y1"] == "PRECONDITION_FAILED_TOKENS"


def _entry(value):
    return {"c_L": value, "c_M": value, "c_H": 0.0, "c_k": {key: 0.0 for key in ra.COMPONENT_ORDER}}


def test_ledger_replication_check():
    recorded = {"a|f": _entry(0.5)}
    assert nf.check_ledger_replication({"a|f": _entry(0.5), "b|f": _entry(0.1)}, recorded)["passed"]
    with pytest.raises(pm.IncidentError, match="deviates"):
        nf.check_ledger_replication({"a|f": _entry(0.5 + 2e-6)}, recorded)
    with pytest.raises(pm.IncidentError, match="lacks"):
        nf.check_ledger_replication({}, recorded)


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
    pool = nf.build_pool_014(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013)
    return manifest, digests, pool


def toy_tokenizer_014(manifest, pool):
    base = toy_tokenizer_013(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in nf.CANDIDATES.values() for word in words]
    for _, text in nf.FRESH_FRAMES:
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


def test_pool_014_and_committed_extract(inputs):
    manifest, digests, pool = inputs
    assert len(pool.tokens) == 135 and len(pool.frames) == 42 and pool.token_source["whom"] == "confirmation-013" and pool.frame_origin["cardinal-013-1"] == "confirmation-013"
    ledger = nf.load_inherited_ledger(ROOT / nf.INHERITED_013_LEDGER_RELATIVE_PATH, digests=digests, expected_size=nf.EXPECTED_LEDGER_SIZE)
    keys = set(ledger["entries"])
    assert "whom|cardinal-013-1" in keys and "same|cardinal-1" in keys and "thy|cardinal-012-1" in keys and "cardinal:sg|cardinal-1" not in keys
    assert sum(1 for key in keys if key.split("|")[1].endswith(("-013-1", "-013-2"))) == 144
    with pytest.raises(ValueError):
        nf.load_inherited_ledger(ROOT / nf.INHERITED_013_LEDGER_RELATIVE_PATH, digests=digests, expected_size=10)


def test_confirmation_policy_with_two_prompt_lists(inputs, tmp_path):
    manifest, digests, pool = inputs
    tokenizer = toy_tokenizer_014(manifest, pool)
    payload = nf.build_confirmation_payload(tokenizer, pool, digests)
    words = [token["word"] for token in payload["tokens"]]
    assert len(words) == 24 and words[:5] == ["second", "third", "former", "latter", "only"] and words[5:10] == ["dozens", "hundreds", "thousands", "millions", "billions"]
    assert len(payload["token_prompts"]) == 24 * 6 and len(payload["exposed_frame_prompts"]) == 24 * 42
    confirmation = nf.validate_confirmation(payload, pool, digests)
    assert len(confirmation.all_prompts) == 12 + 144 + 1008
    path = tmp_path / "c.json"
    assert nf.freeze_confirmation(path, tokenizer, pool, digests) == payload["content_sha256"]
    with pytest.raises(FileExistsError):
        nf.freeze_confirmation(path, tokenizer, pool, digests)
    tampered = dict(payload)
    tampered["tokens"] = list(reversed(payload["tokens"]))
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in tampered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError):
        nf.validate_confirmation(tampered, pool, digests)


def test_neuron_computation_predictor_and_invariant_on_the_fake(inputs, monkeypatch):
    manifest, digests, pool = inputs
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    small = cs.Pool008(pool.frames[:2] + pool.frames_of("quantifier")[:1], pool.frame_origin, pool.tokens[:6], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    cache = pm.PromptCache(model, tuple(small.nouns))
    axes = cs.stage_axes(cache, weights, small)
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
    states = {frame.frame_id: ap.capture_frame_013(model, head, small.reference_prompt(frame), small.single_nouns, axes["T"]) for frame in small.frames}
    bases = lc.template_bases(small.frames, {fid: (s.x1, s.x2) for fid, s in states.items()})
    fpm = ap.FrozenPatternModel(read, lw, heads, bases)
    neuron = nf.NeuronWeights.from_layer_weights(lw, read, layer=2, index=5)  # the fake has 16 neurons per MLP
    d_E = axes["R0"].direction.double()
    frame = small.frames[0]
    state = states[frame.frame_id]
    template = frame.template_id
    # pre/act agree with the exact layer weights; the Jacobian is the first-order LayerNorm derivative.
    assert neuron.pre(state.x2) == pytest.approx(float(pm.exact_layer_norm(state.x2, lw.ln_w[2], lw.ln_b[2], lw.eps) @ lw.W_in[2][:, 5] + lw.b_in[2][5]))
    v = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
    finite = (neuron.pre(state.x2 + 1e-6 * v) - neuron.pre(state.x2)) / 1e-6
    assert neuron.jacobian_pre(state.x2, v) == pytest.approx(finite, rel=1e-4, abs=1e-6)
    uniform = torch.ones(8, dtype=torch.float64)
    assert abs(neuron.jacobian_pre(state.x2, uniform)) < 1e-12 and abs(neuron.pre(state.x2 + uniform) - neuron.pre(state.x2)) < 1e-9  # a uniform shift is removed by the LayerNorm
    plural = nf.measure_pair(model, weights, head, state, pool.plural_cue[template], plural_ids[template], axes["R0"], axes["T"], small.single_nouns)
    name, token_id = next((n, t) for n, t in small.tokens if t not in (pool.reference_ids[template], plural_ids[template]))
    record = nf.measure_pair(model, weights, head, state, name, token_id, axes["R0"], axes["T"], small.single_nouns)
    analysis = nf.analyse_pair_014(record, plural, neuron=neuron, fpm=fpm, weights=weights, state=state, axis_T=axes["T"], d_E=d_E)
    assert analysis is not None and analysis["ledger_da"] == pytest.approx(analysis["da"]) and analysis["fires"] == nf.fires(analysis["da"])
    x2_patch = record.extra[f"RESID_PRE.L2@{frame.p_c}"]
    assert analysis["dpre"] == pytest.approx(neuron.pre(x2_patch) - neuron.pre(state.x2)) and analysis["term"] == pytest.approx(analysis["da"] * neuron.read_out / read.denominator(weights, template))
    p = analysis["prediction"]
    assert p["da_hat"] == pytest.approx(nf.gelu(p["pre_ref"] + p["dpre_hat"]) - nf.gelu(p["pre_ref"])) and p["fires_hat"] == nf.fires(p["da_hat"])
    assert p["dpre_J"] == pytest.approx(p["dpre_J_E"] + p["dpre_J_mlp1"] + p["dpre_J_heads1"]) and p["dpre_J"] == pytest.approx(p["dpre_J_axis"] + p["dpre_J_offaxis"])  # the Jacobian is linear in the change
    assert analysis["remainders"]["pattern_change"] == pytest.approx(analysis["dpre"] - p["dpre_hat"]) and analysis["remainders"]["jacobian"] == pytest.approx(p["dpre_hat"] - p["dpre_J"])
    # The axis alternative equals the predictor when the arriving change lies along d̂_E (a synthetic arriving change c·d̂_E).
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(nf, "arriving_change", lambda *args, **kwargs: {"E": 0.7 * d_E, "mlp1": torch.zeros(8, dtype=torch.float64), "heads1": torch.zeros(8, dtype=torch.float64), "total": 0.7 * d_E})
        along = nf.predict_from_state(neuron, fpm, weights, state, token_id, template, d_E)
    assert along["dpre_hat"] == pytest.approx(along["dpre_axis"], abs=1e-7) and along["da_hat"] == pytest.approx(along["da_axis"], abs=1e-7) and along["fires_hat"] == along["fires_axis"] and abs(along["dpre_J_offaxis"]) < 1e-7  # d̂_E is unit to float32 precision
    assert p["da_J"] == pytest.approx(nf.gelu(p["pre_ref"] + p["dpre_J"]) - nf.gelu(p["pre_ref"])) and p["fires_J"] == nf.fires(p["da_J"])
    # The invariant: the prediction table is computed with every capture and intervention entry point disabled.
    locked = {frame.frame_id: nf.locked_state(states[frame.frame_id], neuron) for frame in small.frames}
    with pytest.MonkeyPatch.context() as guard:
        for attr in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, attr):
                guard.setattr(pm, attr, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a prediction touched the network")))
        rows = nf.prediction_table(neuron, fpm, weights, locked, small.frames, [{"word": name, "token_id": token_id}], list(pm.TEMPLATE_ORDER), d_E)
    assert len(rows) == 3 and rows[0]["da_hat"] == pytest.approx(p["da_hat"]) and set(rows[0]) == set(nf.PREDICTION_COLUMNS)
    assert nf.predict_from_locked(neuron, fpm, weights, locked[frame.frame_id], token_id, template, d_E)["dpre_hat"] == pytest.approx(p["dpre_hat"], abs=1e-12)
    means = nf.token_means_from_table(rows, [name])
    assert means[name]["n_frames"] == 3 and 0.0 <= means[name]["fires_hat_fraction"] <= 1.0


def _analysis(token, frame_id, template, da, pre_ref=-0.5, c_mlp2=0.3):
    dpre = da + 0.1
    return {"token": token, "frame_id": frame_id, "template": template, "pre_ref": pre_ref, "dpre": dpre, "da": da, "fires": nf.fires(da), "term": da * 0.2, "c_mlp2": c_mlp2,
            "c_L": 0.1, "c_M": 0.1, "c_H": 0.0, "c_k": {k: 0.0 for k in ra.COMPONENT_ORDER}, "P1": 0.5, "q_T": 0.5, "g_E": 0.4, "prediction": {}, "remainders": {"pattern_change": 0.0, "jacobian": 0.0}, "identities": {}}


def _row(token, frame_id, template, da_hat, da_axis, pre_ref=-0.5):
    prediction = {"pre_ref": pre_ref, "dpre_hat": da_hat + 0.1, "da_hat": da_hat, "fires_hat": nf.fires(da_hat), "dpre_J": da_hat + 0.1, "da_J": da_hat, "fires_J": nf.fires(da_hat), "dpre_J_E": da_hat, "dpre_J_mlp1": 0.05, "dpre_J_heads1": 0.05,
                  "dpre_J_axis": 0.3, "dpre_J_offaxis": da_hat - 0.2, "dpre_axis": da_axis + 0.1, "da_axis": da_axis, "fires_axis": nf.fires(da_axis), "term_hat": da_hat * 0.2}
    return nf.prediction_row(token, frame_id, template, prediction)


class _Confirmation:
    def __init__(self, fresh_frames, tokens):
        self.frames = [pm.Frame(t, fid, (1, 2), () if t in pm.CUE_FINAL_TEMPLATES else (7,), {"sg": 5, "pl": 6}, "x {cue}" + ("" if t in pm.CUE_FINAL_TEMPLATES else " y")) for fid, t in fresh_frames]
        self.tokens = [{"word": w, "category": "numeral" if i % 2 else "adjective", "licensed_frames": [fid for fid, _ in fresh_frames]} for i, w in enumerate(tokens)]


def test_scoring_rules_the_guard_and_the_axis_rejection_on_synthetic_tables():
    exposed = [(f"e{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(6)]
    fresh = [("c1", "cardinal"), ("c2", "cardinal"), ("q1", "quantifier"), ("q2", "quantifier"), ("a1", "coordinated-adjective"), ("a2", "coordinated-adjective")]
    words = [f"w{i}" for i in range(18)]
    da_values = {w: (1.5 + 0.05 * i if i % 2 else -0.1 + 0.01 * i) for i, w in enumerate(words)}  # bimodal: odd tokens fire, even do not
    confirmation = _Confirmation(fresh, words)
    categories = {token["word"]: token["category"] for token in confirmation.tokens}

    def measured(frames, scale=1.0):
        return {f"{w}|{fid}": _analysis(w, fid, t, scale * da_values[w]) for fid, t in frames for w in words}

    def tables(da_hat, da_axis):
        locked = {"predictions": {"rows": [_row(w, fid, t, da_hat(w), da_axis(w)) for fid, t in exposed for w in words]}}
        stage1 = {"rows": [_row(w, fid, t, da_hat(w), da_axis(w)) for fid, t in fresh for w in words], "frames": {fid: {"template_id": t, "valid": True, "template_defined": True} for fid, t in fresh}}
        return locked, stage1

    # Perfect predictor, axis-only firing for everything: Y1/Y2 pass, the axis alternative is rejected.
    lock, stage1 = tables(lambda w: da_values[w], lambda w: 1.0)
    results = nf.score_confirmation(stage1, measured(exposed), measured(fresh), confirmation, lock, categories)
    assert results["Y1"]["test"]["passed"] and results["Y2"]["test"]["passed"] and results["Y1"]["test"]["classification"]["balanced_accuracy"] == 1.0 and results["Y1"]["test"]["r2"] == pytest.approx(1.0)
    assert results["Y3"]["evaluable"] and results["Y3"]["rejected"] and results["Y3"]["classification"]["specificity"] == 0.0 and results["Y3"]["classification"]["balanced_accuracy"] == 0.5
    assert results["outcome"]["label"] == "NEURON_FEATURE_PREDICTED_TOKENS | NEURON_FEATURE_PREDICTED_FRAMES_CONDITIONAL | AXIS_ONLY_REJECTED"
    assert set(results["firing_sets"]["Y1"]["measured"]["numeral"]) == {w for i, w in enumerate(words) if i % 2} and "adjective" not in results["firing_sets"]["Y1"]["measured"]
    # Axis-only equal to the predictor: not rejected.
    lock_a, stage1_a = tables(lambda w: da_values[w], lambda w: da_values[w])
    assert nf.score_confirmation(stage1_a, measured(exposed), measured(fresh), confirmation, lock_a, categories)["outcome"]["Y3"] == "AXIS_ONLY_NOT_REJECTED"
    # A compressed predictor (0.4 × the true scale) keeps the ordering; the firing tokens fall below 0.5 only when 0.4 × da < 0.5, i.e. never here (da ≥ 1.55) — so the classification stays perfect and R² fails alone.
    lock_c, stage1_c = tables(lambda w: 0.4 * da_values[w], lambda w: 1.0)
    compressed = nf.score_confirmation(stage1_c, measured(exposed), measured(fresh), confirmation, lock_c, categories)
    assert compressed["Y1"]["test"]["failing"] == ["r2"] and compressed["Y1"]["test"]["spearman"] == pytest.approx(1.0)
    # A predictor that fires for everything: balanced accuracy 0.5 fails while the ordering is kept.
    lock_f, stage1_f = tables(lambda w: da_values[w] + 1.0, lambda w: 1.0)
    always = nf.score_confirmation(stage1_f, measured(exposed), measured(fresh), confirmation, lock_f, categories)
    assert "balanced_accuracy" in always["Y1"]["test"]["failing"] and always["Y1"]["test"]["classification"]["specificity"] == 0.0
    # Reversed ordering fails Spearman (the compressed predictor above keeps it).
    lock_r = {"predictions": {"rows": [_row(w, fid, t, da_values[words[17 - i]], 1.0) for fid, t in exposed for i, w in enumerate(words)]}}
    assert "spearman" in nf.score_confirmation(stage1, measured(exposed), measured(fresh), confirmation, lock_r, categories)["Y1"]["test"]["failing"]
    # A descriptive share over an empty filtered list must not crash (every firing pair with |c_mlp2| ≤ 0.05).
    tiny = {k: {**v, "c_mlp2": 0.01} for k, v in measured(exposed).items()}
    assert nf.score_confirmation(stage1, tiny, measured(fresh), confirmation, lock, categories)["Y1"]["descriptive"]["term_share_of_mlp2_firing"] is None
    assert nf.firing_set({"w": {"da_mean": 1.0}}, {"w": "x"}) == {}
    # Guard: when nothing fires in the measured set, the classification is non-evaluable and the label says so; Y3 is non-evaluable too.
    none_fire = nf.score_confirmation(stage1, measured(exposed, scale=0.1), measured(fresh, scale=0.1), confirmation, lock, categories)
    assert not none_fire["Y1"]["test"]["classification_evaluable"] and none_fire["outcome"]["Y1"].endswith(nf.CLASSIFICATION_NOT_EVALUABLE_SUFFIX) and none_fire["outcome"]["Y3"] == "AXIS_ONLY_NOT_EVALUABLE"
    # Y2 precondition: three valid fresh frames → PRECONDITION_FAILED_FRAMES while Y1 is unaffected.
    three = {"rows": stage1["rows"], "frames": {fid: {"template_id": t, "valid": fid in ("c1", "q1", "a1"), "template_defined": True} for fid, t in fresh}}
    partial = {k: v for k, v in measured(fresh).items() if k.split("|")[1] in ("c1", "q1", "a1")}
    verdict = nf.score_confirmation(three, measured(exposed), partial, confirmation, lock, categories)
    assert verdict["outcome"]["Y2"] == "PRECONDITION_FAILED_FRAMES" and verdict["outcome"]["Y1"] == "NEURON_FEATURE_PREDICTED_TOKENS"
    # Identical frame sets: a token measured in two exposed frames is not scored for Y1.
    two = {k: v for k, v in measured(exposed).items() if not (k.startswith("w0|") and k.split("|")[1] not in ("e0", "e1"))}
    scored = nf.score_confirmation(stage1, two, measured(fresh), confirmation, lock, categories)
    assert not scored["Y1"]["tokens"]["w0"]["scored"] and len(scored["Y1"]["scored_tokens"]) == 17


def test_lock_reproduction_refusal():
    rows = [_row("w", "f", "cardinal", 1.2, 0.3)]
    lock = {"predictions": {"rows": rows}}
    assert nf.assert_lock_predictions_reproduced(lock, {"rows": rows}) == 0.0
    perturbed = json.loads(json.dumps(rows))
    perturbed[0]["dpre_J_offaxis"] += 1e-6
    with pytest.raises(nf.PhaseError, match="nothing was executed"):
        nf.assert_lock_predictions_reproduced(lock, {"rows": perturbed})
    flipped = json.loads(json.dumps(rows))
    flipped[0]["fires_axis"] = not flipped[0]["fires_axis"]
    with pytest.raises(nf.PhaseError, match="firing prediction"):
        nf.assert_lock_predictions_reproduced(lock, {"rows": flipped})
