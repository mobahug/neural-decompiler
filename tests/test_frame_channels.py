"""Experiment 016: the channel switches and their recoveries on the fake, the sigma-prime invariant, the table shapes, the pooled statistics with the frame guard, the confirmation policy, floors, Y3 and the extract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_decompiler import attention_paths as ap
from neural_decompiler import attention_patterns as atp
from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import encoding_read as er
from neural_decompiler import frame_channels as fch
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import neuron_feature as nf
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from plural_fakes import TinyPlural
from test_attention_patterns import toy_tokenizer_015

ROOT = Path(__file__).resolve().parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


def test_frozen_constants_match_the_design():
    assert fch.Y_SPEARMAN == 0.90 and fch.Y_R2 == 0.85 and fch.Y_ENTRY_R2 == 0.85 and fch.FRAME_GUARD_R2 == 0.80 and fch.Y3_ENTRY_R2_MAX == 0.85 and fch.Y3_MARGIN == 0.15 and fch.COMPARATOR_MARGIN == 0.10
    assert fch.MIN_VALID_FRAMES == 8 and fch.MIN_VALID_FRAMES_PER_TOKEN == 3 and fch.MIN_SCORED_TOKENS == 16 and fch.FRAME_CUE_EFFECT_RATE == 108 / 120 and fch.RECOVERY_TOLERANCE == 1e-9
    assert fch.RUNTIME_SEED == 20260916 and fch.CONTROL_SEED == 20260924 and len(fch.FRESH_FRAMES) == 12 and fch.EXPECTED_EXTRACT_SIZE_015 == 6180 and sum(fch.QUOTAS.values()) == 24
    assert fch.QUOTAS == {"determiner-like": 5, "ordinal-or-numeral": 4, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 6}
    exposed = {token["word"] for path in (er.CONFIRMATION_RELATIVE_PATH, lc.CONFIRMATION_RELATIVE_PATH, ap.CONFIRMATION_RELATIVE_PATH, nf.CONFIRMATION_RELATIVE_PATH, atp.CONFIRMATION_RELATIVE_PATH) for token in json.loads((ROOT / path).read_text())["tokens"]}
    assert not (set(sum(fch.CANDIDATES.values(), ())) & exposed), "016 candidates must not be earlier frozen tokens"
    assert fch.OUTCOME_Y3 == ("SCALE_ONLY_REJECTED", "SCALE_ONLY_NOT_REJECTED", "SCALE_ONLY_NOT_EVALUABLE") and fch.OUTCOME_Y1[0] == "FRAME_CHANNELS_PREDICTED_TOKENS"
    assert fch.CHANNEL_NAMES == ("operands", "scale", "operating_point") and fch.LEVEL0F.name == "level0F" and fch.SCALE_ONLY.name == "scale_only" and fch.LEVEL0_015.name == "level0_015" and fch.LEVEL1.name == "level1"
    assert {c.name for c in fch.ABLATION_CHANNELS.values()} == set(fch.ABLATIONS) and fch.Channels(operands=True, scale=False, operating_point=False).name == "only_operands"
    assert set(fch.frozen_floors()) >= {"y_spearman", "y_r2", "y_entry_r2", "frame_guard_r2", "y3_entry_r2_max", "y3_margin", "comparator_margin", "recovery_tolerance"}


def test_extract_replication_check():
    entry = {"c_dA": 0.02, "delta_A_pc": {key: 0.1 for key in fch.HEAD_KEYS}, "c_L": 0.1, "c_M": 0.1, "c_H": 0.0}
    assert fch.check_extract_replication({"a|f": entry}, {"a|f": entry})["passed"]
    with pytest.raises(pm.IncidentError, match="c_dA"):
        fch.check_extract_replication({"a|f": {**entry, "c_dA": 0.02 + 2e-6}}, {"a|f": entry})
    with pytest.raises(pm.IncidentError, match="lack"):
        fch.check_extract_replication({}, {"a|f": entry})


def test_outcome_labels():
    base = {"precondition_Y1": {"passed": True}, "precondition_Y2": {"passed": True}, "Y1": {"test": {"passed": True}}, "Y2": {"test": {"passed": False}}, "Y3": {"evaluable": True, "rejected": True}}
    assert fch.outcome(base)["label"] == "FRAME_CHANNELS_PREDICTED_TOKENS | FRAME_CHANNELS_NOT_PREDICTED_FRAMES_CONDITIONAL | SCALE_ONLY_REJECTED"
    assert fch.outcome({**base, "Y3": {"evaluable": False, "rejected": False}})["Y3"] == "SCALE_ONLY_NOT_EVALUABLE" and fch.outcome({**base, "precondition_Y2": {"passed": False}})["Y2"] == "PRECONDITION_FAILED_FRAMES"


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
    pool = fch.build_pool_016(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015)
    return manifest, digests, pool


def toy_tokenizer_016(manifest, pool):
    base = toy_tokenizer_015(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in fch.CANDIDATES.values() for word in words]
    for _, text in fch.FRESH_FRAMES:
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


def test_pool_016_and_committed_extract(inputs):
    manifest, digests, pool = inputs
    assert len(pool.tokens) == 183 and len(pool.frames) == 54 and pool.token_source["tenth"] == "confirmation-015" and pool.frame_origin["quantifier-015-2"] == "confirmation-015"
    extract = fch.load_inherited_extract(ROOT / fch.INHERITED_015_EXTRACT_RELATIVE_PATH, digests=digests, expected_size=fch.EXPECTED_EXTRACT_SIZE_015)
    keys = set(extract["entries"])
    assert "tenth|cardinal-1" in keys and "tenth|quantifier-015-2" in keys and "dozens|cardinal-014-1" in keys and "same|cardinal-1" in keys and "cardinal:sg|cardinal-1" not in keys
    assert set(extract["stage1_state_digests"]) == {f"{t}-015-{i}" for t in pm.TEMPLATE_ORDER for i in (1, 2)} and set(next(iter(extract["entries"].values()))) == {"c_dA", "delta_A_pc", "c_L", "c_M", "c_H"}
    with pytest.raises(ValueError):
        fch.load_inherited_extract(ROOT / fch.INHERITED_015_EXTRACT_RELATIVE_PATH, digests=digests, expected_size=10)


def test_confirmation_policy_with_twelve_frames(inputs, tmp_path):
    manifest, digests, pool = inputs
    tokenizer = toy_tokenizer_016(manifest, pool)
    payload = fch.build_confirmation_payload(tokenizer, pool, digests)
    words = [token["word"] for token in payload["tokens"]]
    assert len(words) == 24 and words[:5] == ["initial", "upper", "respective", "individual", "specific"] and words[5:9] == ["twentieth", "quarter", "twin", "dual"] and words[-6:] == ["bright", "orange", "pink", "brown", "grey", "heavy"]
    assert len(payload["token_prompts"]) == 24 * 12 and len(payload["exposed_frame_prompts"]) == 24 * 54
    confirmation = fch.validate_confirmation(payload, pool, digests)
    assert len(confirmation.all_prompts) == 24 + 288 + 1296 and [frame.frame_id for frame in confirmation.frames][:4] == ["cardinal-016-1", "cardinal-016-2", "cardinal-016-3", "cardinal-016-4"]
    path = tmp_path / "c.json"
    assert fch.freeze_confirmation(path, tokenizer, pool, digests) == payload["content_sha256"]
    with pytest.raises(FileExistsError):
        fch.freeze_confirmation(path, tokenizer, pool, digests)
    tampered = dict(payload)
    tampered["tokens"] = list(reversed(payload["tokens"]))
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in tampered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError):
        fch.validate_confirmation(tampered, pool, digests)


def _fake_setup(pool):
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    programs = atp.programs_from_model(model)
    small = cs.Pool008(pool.frames[:2] + pool.frames_of("quantifier")[:1], pool.frame_origin, pool.tokens[:6], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    cache = pm.PromptCache(model, tuple(small.nouns))
    axes = cs.stage_axes(cache, weights, small)
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
    states = {frame.frame_id: ap.capture_frame_013(model, head, small.reference_prompt(frame), small.single_nouns, axes["T"]) for frame in small.frames}
    bases = lc.template_bases(small.frames, {fid: (s.x1, s.x2) for fid, s in states.items()})
    fpm = ap.FrozenPatternModel(read, lw, heads, bases)
    fcm = fch.FrameChannelModel(read, lw, programs, bases, axes["R0"].direction.double())
    return model, weights, head, lw, heads, programs, small, axes, plural_ids, read, states, bases, fpm, fcm


def test_channel_recoveries_analysis_and_invariant_on_the_fake(inputs, monkeypatch):
    manifest, digests, pool = inputs
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model, weights, head, lw, heads, programs, small, axes, plural_ids, read, states, bases, fpm, fcm = _fake_setup(pool)
    context = fch.AnalysisContext(fcm, fpm, heads, weights, axes["T"])
    frame = small.frames[0]
    state = states[frame.frame_id]
    template = frame.template_id
    rows = atp.reference_rows(programs, state.x1_all, state.x2_all)
    name, token_id = next((n, t) for n, t in small.tokens if t not in (pool.reference_ids[template], plural_ids[template]))
    # The recoveries: all switches off is Experiment 015's Level 0; all channels plus the remainder is Level 1.
    errors = fch.recovery_errors(fcm, weights, rows, state.x1_all, state.x2_all, token_id, template)
    assert errors["level0_015_recovery"] < 1e-12 and errors["level1_recovery"] < 1e-12
    x1, x2 = state.x1_all[frame.p_c].double(), state.x2_all[frame.p_c].double()
    full = fcm.predict_channels(weights, rows, x1, x2, token_id, template, fch.LEVEL0F)
    # Channel B: σ' is the scale of the reference residual plus the PREDICTED change — an algebraic function, checked here against the definition.
    dx2 = full["dx2"]
    assert full["sigma"][2][0] == pytest.approx(float(fch._sigma(x2, programs[2].eps))) and full["sigma"][2][1] == pytest.approx(float(fch._sigma(x2 + dx2, programs[2].eps)))
    assert full["sigma"][1][1] == pytest.approx(float(fch._sigma(x1 + read.encoding_delta(weights, token_id, template), programs[1].eps)))
    # With the frame equal to the template base, every channel is inert: Level 0-F equals Experiment 015's Level 0.
    own = fch.FrameChannelModel(read, lw, programs, {template: (x1, x2)}, fcm.d_E)
    z_own = own.predict_channels(weights, rows, x1, x2, token_id, template, fch.LEVEL0F)
    z_off = own.predict_channels(weights, rows, x1, x2, token_id, template, fch.LEVEL0_015)
    assert all(z_own["rows"][layer] == pytest.approx(z_off["rows"][layer], abs=1e-12) for layer in fch.HEAD_LAYERS)
    # The analysis: identities, statistics, ladder, sigma remainders; the table entry.
    plural = fch.measure_pair(model, weights, head, state, pool.plural_cue[template], plural_ids[template], axes["R0"], axes["T"], small.single_nouns)
    record = fch.measure_pair(model, weights, head, state, name, token_id, axes["R0"], axes["T"], small.single_nouns)
    analysis = fch.analyse_pair_016(record, plural, context=context, state=state, rows=rows)
    assert analysis is not None
    ids = analysis["identities"]
    assert ids["I1_patched_rows"] < 1e-5 and ids["I2_chain"] < 1e-5 and ids["I3_head_split"] < 1e-5 and ids["level0_015_recovery"] < 1e-12 and ids["level1_recovery"] < 1e-12
    assert analysis["ladder_statistics"]["level1"]["2"]["ss_res"] < 1e-10 and analysis["ladder"]["level1_error"] < 1e-4 and analysis["c_level1"] == pytest.approx(analysis["c"], abs=1e-5)
    assert set(analysis["ladder_statistics"]) == {*fch.ABLATIONS, "level0_015", "level1"} and set(analysis["statistics"]) == {"level0F", "scale_only", "diag_L0F"}
    assert abs(analysis["scale_remainder"]["1"]) < 1e-6  # layer 1's change is exact (ΔE), so the predicted σ' equals the measured one
    p = analysis["prediction"]
    assert set(p) == set(fch.PREDICTION_COLUMNS[3:]) and p["c_hat"] == pytest.approx(p["c_hat_1"] + p["c_hat_2"]) and p["c_L_level0F"] == pytest.approx(p["c_M_level0F"] + p["c_H_level0F"])
    for layer in fch.HEAD_LAYERS:
        for key in ("rows_level0F", "rows_scale_only", "rows_diag_L0F"):
            predicted = atp.rows_from_json(p[key][str(layer)], layer)
            assert predicted.shape == (8, frame.p_c + 1) and predicted.sum(-1) == pytest.approx(torch.zeros(8), abs=1e-9)
        assert p["sigma_ratio"][str(layer)] == pytest.approx(full["sigma"][layer][1] / full["sigma"][layer][0])
    # The invariant: the table is computed with every capture and intervention entry point disabled, and a poisoned patched capture cannot change it.
    locked = {frame.frame_id: fch.locked_state(states[frame.frame_id]) for frame in small.frames}
    with pytest.MonkeyPatch.context() as guard:
        for attr in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, attr):
                guard.setattr(pm, attr, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a prediction touched the network")))
        table = fch.prediction_table(fcm, weights, locked, small.frames, [{"word": name, "token_id": token_id}], list(pm.TEMPLATE_ORDER))
    assert len(table) == 3 and set(table[0]) == set(fch.PREDICTION_COLUMNS) and atp._max_numeric_difference(table[0], fch.prediction_row(name, frame.frame_id, template, p), "row") < 1e-12
    poisoned = dict(record.extra)
    for key in poisoned:
        poisoned[key] = poisoned[key] + 10.0
    poisoned_record = ra.Attribution(*[getattr(record, f.name) for f in record.__dataclass_fields__.values() if f.name != "extra"], poisoned)
    with pytest.raises(pm.IncidentError):  # the identities notice the poisoned capture …
        fch.analyse_pair_016(poisoned_record, plural, context=context, state=state, rows=rows)
    assert atp._max_numeric_difference(fcm.predict_from_state(weights, state, token_id, template), p, "again") < 1e-12  # … while the prediction never looks at it
    means = fch.token_means_from_table(table, [name])
    assert means[name]["n_frames"] == 3 and set(means[name]["self_hat_mean"]) == set(fch.HEAD_KEYS)
    stats = fch.statistics_for([analysis], [p], [name])
    assert stats["pairs"]["level0F"]["2"]["n_entries"] == 8 * (frame.p_c + 1) and set(stats["ladder"]["ablation_cost_layer2"]) == set(fch.ABLATIONS) and frame.frame_id in stats["per_frame"]


def _rows(seed, p_c, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for layer in fch.HEAD_LAYERS:
        raw = torch.randn(8, p_c + 1, generator=generator, dtype=torch.float64)
        raw = raw - raw.mean(-1, keepdim=True)
        out[str(layer)] = atp.rows_to_json(scale * raw, layer)
    return out


def _stat(rows_a, rows_b, layer):
    return atp._statistics_of(rows_b, rows_a, layer)


def _measured(word, frame_id, template, p_c, rows, c):
    self_values = {ap.head_key(layer, h): rows[str(layer)][ap.head_key(layer, h)][p_c] for layer in fch.HEAD_LAYERS for h in range(8)}
    unit = {"n": 1, "sum_m": 0.0, "sum_m2": 1.0, "ss_res": 0.0, "tv_err": 0.0, "tv": 1.0}
    return {"token": word, "frame_id": frame_id, "template": template, "p_c": p_c, "rows": rows, "self": self_values, "c_1": c / 2, "c_2": c / 2, "c": c, "c_level1": c,
            "c_L": 0.1, "c_M": 0.1, "c_H": 0.0, "c_k": {k: 0.0 for k in ra.COMPONENT_ORDER}, "P1": 0.5, "q_T": 0.5, "g_E": 0.4, "prediction": {},
            "statistics": {"level0F": {"1": unit, "2": unit}}, "ladder_statistics": {name: {"1": unit, "2": unit} for name in (*fch.ABLATIONS, "level0_015", "level1")},
            "ladder": {key: 0.1 for key in fch.LADDER_KEYS} | {"level1_error": 0.0}, "sigma": {"1": {"ref": 1.0, "after_predicted": 1.1, "after_measured": 1.1}, "2": {"ref": 1.0, "after_predicted": 1.1, "after_measured": 1.1}},
            "scale_remainder": {"1": 0.0, "2": 0.0}, "identities": {}}


def _table_row(word, frame_id, template, p_c, rows, c_hat, rows_scale, c_scale, rows_diag):
    self_values = {ap.head_key(layer, h): rows[str(layer)][ap.head_key(layer, h)][p_c] for layer in fch.HEAD_LAYERS for h in range(8)}
    prediction = {"p_c": p_c, "rows_level0F": rows, "self_level0F": self_values, "c_hat_1": c_hat / 2, "c_hat_2": c_hat / 2, "c_hat": c_hat, "rows_scale_only": rows_scale, "c_scale_only": c_scale, "rows_diag_L0F": rows_diag, "c_diag_L0F": 0.0,
                  "c_ablate_operands": 0.0, "c_ablate_scale": 0.0, "c_ablate_operating_point": 0.0, "c_level0_015": 0.0, "sigma_ratio": {"1": 1.1, "2": 1.1}, "arrival_read": 0.3, "c_L_level0F": 0.1, "c_M_level0F": 0.1, "c_H_level0F": 0.0}
    return fch.prediction_row(word, frame_id, template, prediction)


class _Confirmation:
    def __init__(self, fresh_frames, tokens):
        self.frames = [pm.Frame(t, fid, (1, 2), () if t in pm.CUE_FINAL_TEMPLATES else (7,), {"sg": 5, "pl": 6}, "x {cue}" + ("" if t in pm.CUE_FINAL_TEMPLATES else " y")) for fid, t in fresh_frames]
        self.tokens = [{"word": w, "category": "adjective", "licensed_frames": [fid for fid, _ in fresh_frames]} for w in tokens]


def test_scoring_floors_frame_guard_and_scale_only_rejection_on_synthetic_tables():
    exposed = [(f"e{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(6)]
    fresh = [(f"n{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(12)]
    words = [f"w{i}" for i in range(18)]
    c_values = {w: 0.01 * (i - 9) for i, w in enumerate(words)}
    p_c = 3
    confirmation = _Confirmation(fresh, words)
    true_rows = {(w, fid): _rows(hash((w, fid)) % 10_000, p_c) for fid, _ in exposed + fresh for w in words}
    zero_rows = _rows(0, p_c, scale=0.0)

    def measured(frames):
        return {f"{w}|{fid}": _measured(w, fid, t, p_c, true_rows[(w, fid)], c_values[w]) for fid, t in frames for w in words}

    def tables(rows_hat, c_hat, rows_scale, c_scale, valid=None):
        make = lambda fid, t, w: _table_row(w, fid, t, p_c, rows_hat(w, fid), c_hat(w), rows_scale(w, fid), c_scale(w), zero_rows)  # noqa: E731
        locked = {"predictions": {"rows": [make(fid, t, w) for fid, t in exposed for w in words]}}
        stage1 = {"rows": [make(fid, t, w) for fid, t in fresh for w in words], "frames": {fid: {"template_id": t, "valid": True if valid is None else fid in valid, "template_defined": True} for fid, t in fresh}}
        return locked, stage1

    perfect = lambda w, fid: true_rows[(w, fid)]  # noqa: E731
    nothing = lambda w, fid: zero_rows  # noqa: E731
    lock, stage1 = tables(perfect, lambda w: c_values[w], nothing, lambda w: 0.0)
    results = fch.score_confirmation(stage1, measured(exposed), measured(fresh), confirmation, lock)
    t2 = results["Y2"]["test"]
    assert results["Y1"]["test"]["passed"] and t2["passed"] and t2["frame_guard"]["passed"] and len(t2["frame_guard"]["per_frame_layer2"]) == 12 and "frame_guard" not in results["Y1"]["test"]
    assert results["Y3"]["rejected"] and results["Y3"]["scale_only_entry_r2_layer2"] == pytest.approx(0.0, abs=1e-9) and results["Y3"]["margin"] == pytest.approx(1.0)
    assert results["outcome"]["label"] == "FRAME_CHANNELS_PREDICTED_TOKENS | FRAME_CHANNELS_PREDICTED_FRAMES_CONDITIONAL | SCALE_ONLY_REJECTED" and results["comparator"]["interaction_beyond_self_logit"]
    assert set(results["Y2"]["per_frame"]) == {fid for fid, _ in fresh} and "ladder_both_sets" in results
    # One collapsed fresh frame: the aggregate still passes its floors, the guard fails the family and names the frame.
    collapsed = lambda w, fid: (_rows(hash((w, fid)) % 10_000, p_c, scale=0.1) if fid == "n5" else true_rows[(w, fid)])  # noqa: E731
    lock_c, stage1_c = tables(collapsed, lambda w: c_values[w], nothing, lambda w: 0.0)
    verdict = fch.score_confirmation(stage1_c, measured(exposed), measured(fresh), confirmation, lock_c)
    t2 = verdict["Y2"]["test"]
    assert t2["entry_r2_layer2"] >= fch.Y_ENTRY_R2 and t2["failing"] == ["frame_guard"] and set(t2["frame_guard"]["frames_below"]) == {"n5"} and verdict["outcome"]["Y2"] == "FRAME_CHANNELS_NOT_PREDICTED_FRAMES_CONDITIONAL"
    # Scale-only equal to the truth: not rejected (its R² is above the ceiling); compressed Level 0-F (0.6 ×: R² 0.84) fails both entry floors.
    lock_s, stage1_s = tables(perfect, lambda w: c_values[w], perfect, lambda w: c_values[w])
    assert fch.score_confirmation(stage1_s, measured(exposed), measured(fresh), confirmation, lock_s)["outcome"]["Y3"] == "SCALE_ONLY_NOT_REJECTED"
    compressed = lambda w, fid: _rows(hash((w, fid)) % 10_000, p_c, scale=0.6)  # noqa: E731
    lock_k, stage1_k = tables(compressed, lambda w: c_values[w], nothing, lambda w: 0.0)
    failing = fch.score_confirmation(stage1_k, measured(exposed), measured(fresh), confirmation, lock_k)["Y1"]["test"]
    assert failing["failing"] == ["entry_r2_layer1", "entry_r2_layer2"] and failing["entry_r2_layer1"] == pytest.approx(0.84)
    # Scale-only close to Level 0-F (margin below 0.15): not rejected even though below the ceiling.
    near = lambda w, fid: _rows(hash((w, fid)) % 10_000, p_c, scale=0.9)  # noqa: E731
    lock_n, stage1_n = tables(perfect, lambda w: c_values[w], near, lambda w: 0.9 * c_values[w])
    y3 = fch.score_confirmation(stage1_n, measured(exposed), measured(fresh), confirmation, lock_n)["Y3"]
    assert y3["scale_only_entry_r2_layer2"] == pytest.approx(0.99) and not y3["rejected"]
    # Y2 precondition: seven valid fresh frames → PRECONDITION_FAILED_FRAMES while Y1 is unaffected; invalid frames run no fresh cue.
    seven = [fid for fid, _ in fresh][:7]
    lock_p, stage1_p = tables(perfect, lambda w: c_values[w], nothing, lambda w: 0.0, valid=seven)
    partial = {k: v for k, v in measured(fresh).items() if k.split("|")[1] in seven}
    verdict = fch.score_confirmation(stage1_p, measured(exposed), partial, confirmation, lock_p)
    assert verdict["outcome"]["Y2"] == "PRECONDITION_FAILED_FRAMES" and verdict["outcome"]["Y1"] == "FRAME_CHANNELS_PREDICTED_TOKENS"
    # Reversed reads fail Spearman and R² while the rows are exact.
    lock_r, stage1_r = tables(perfect, lambda w: -c_values[w], nothing, lambda w: 0.0)
    failing = fch.score_confirmation(stage1_r, measured(exposed), measured(fresh), confirmation, lock_r)["Y1"]["test"]["failing"]
    assert "spearman" in failing and "r2" in failing and "entry_r2_layer1" not in failing


def test_lock_reproduction_refusal_and_predictions_rendering():
    row = _table_row("w", "f", "cardinal", 3, _rows(1, 3), 0.02, _rows(2, 3), 0.0, _rows(3, 3))
    lock = {"predictions": {"rows": [row]}}
    assert fch.assert_lock_predictions_reproduced(lock, {"rows": [json.loads(json.dumps(row))]}) == 0.0
    perturbed = json.loads(json.dumps(row))
    perturbed["rows_scale_only"]["2"]["L02.H06"][1] += 1e-6
    with pytest.raises(fch.PhaseError, match="nothing was executed"):
        fch.assert_lock_predictions_reproduced(lock, {"rows": [perturbed]})
    ratio = json.loads(json.dumps(row))
    ratio["sigma_ratio"]["2"] += 1e-6
    with pytest.raises(fch.PhaseError, match="nothing was executed"):
        fch.assert_lock_predictions_reproduced(lock, {"rows": [ratio]})
    full_lock = {"run_id": "r", "protocol_code_commit": "c" * 40, "confirmation_016_sha256": "s" * 64, "lock_015_sha256": "l" * 64, "program": {"1": {"d_head": 64, "rotary_dim": 16, "rotary_base": 10000.0}, "2": {"d_head": 64, "rotary_dim": 16, "rotary_base": 10000.0}},
                 "tokens": [{"word": "w", "category": "adjective"}], "predictions": {"rows": [row], "token_means": fch.token_means_from_table([row], ["w"])}}
    text = fch.render_predictions(full_lock)
    assert "preregistered predictions" in text and "| w | adjective | 1 | **0.0200**" in text and "frame-collapse guard" in text and "L02.H07" in text
