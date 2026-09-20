"""Experiment 017: the exact chain and its identities on the fake (cue-final and coordinated frames), the switch variants and the Level 1 recovery, the leak-proof boundary with a poisoned capture at both positions, the table shapes, the scoring with the frame guard and the cue-final Y3, the confirmation policy and the extract."""

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
from neural_decompiler import head_pattern as hp
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import neuron_feature as nf
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from plural_fakes import TinyPlural
from test_frame_channels import toy_tokenizer_016

ROOT = Path(__file__).resolve().parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


def test_frozen_constants_match_the_design():
    assert hp.Y_ROW_R2 == 0.95 and hp.Y_PI_SPEARMAN == 0.90 and hp.Y_PI_R2 == 0.90 and hp.Y_DT_R2 == 0.95 and hp.FRAME_GUARD_R2 == 0.90 and hp.Y3_R2_MAX == 0.95 and hp.Y3_MARGIN == 0.05
    assert hp.X3_IDENTITY_TOLERANCE == 1e-4 and hp.ROW_IDENTITY_TOLERANCE == 1e-4 and hp.DT_IDENTITY_TOLERANCE == 1e-3 and hp.SPLIT_IDENTITY_TOLERANCE == 1e-4 and hp.RECOVERY_TOLERANCE == 1e-9
    assert hp.MIN_VALID_FRAMES == 8 and hp.MIN_VALID_FRAMES_PER_TOKEN == 3 and hp.MIN_SCORED_TOKENS == 16 and hp.FRAME_CUE_EFFECT_RATE == 108 / 120
    assert hp.RUNTIME_SEED == 20260916 and hp.CONTROL_SEED == 20260924 and len(hp.FRESH_FRAMES) == 12 and hp.EXPECTED_EXTRACT_SIZE_016 == 7764 and sum(hp.QUOTAS.values()) == 24
    assert hp.QUOTAS == {"determiner-like": 5, "ordinal-or-numeral": 4, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 6}
    assert [t for t, _ in hp.FRESH_FRAMES].count("cardinal") == 4 and [t for t, _ in hp.FRESH_FRAMES].count("coordinated-adjective") == 4
    exposed = {token["word"] for path in (er.CONFIRMATION_RELATIVE_PATH, lc.CONFIRMATION_RELATIVE_PATH, ap.CONFIRMATION_RELATIVE_PATH, nf.CONFIRMATION_RELATIVE_PATH, atp.CONFIRMATION_RELATIVE_PATH, fch.CONFIRMATION_RELATIVE_PATH) for token in json.loads((ROOT / path).read_text())["tokens"]}
    assert not (set(sum(hp.CANDIDATES.values(), ())) & exposed), "017 candidates must not be earlier frozen tokens"
    assert hp.OUTCOME_Y3 == ("FROZEN_PATTERN_REJECTED", "FROZEN_PATTERN_NOT_REJECTED", "FROZEN_PATTERN_NOT_EVALUABLE") and hp.OUTCOME_Y1[0] == "HEAD_PATTERN_PREDICTED_TOKENS" and hp.OUTCOME_Y2[0] == "HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL"
    assert hp.HEAD_KEY == "L03.H04" and hp.HEAD_LAYER == 3 and hp.HEAD_INDEX == 4 and hp.PROGRAM_LAYERS == (1, 2, 3)
    assert hp.LEVEL0.upstream is fch.LEVEL0F and hp.LEVEL0.block2_own and hp.LEVEL0.head_channels and not hp.LEVEL0.exact_head
    assert hp.LEVEL1.upstream is fch.LEVEL1 and hp.LEVEL1.exact_head and not hp.NO_D.block2_own and not hp.TEMPLATE_HEAD.head_channels and hp.EXACT_HEAD.exact_head
    assert hp.RUNGS == ("exact_head", "no_D", "template_head") and hp.ABLATIONS == ("no_D", "template_head") and hp.LADDER_KEYS == ("c_L_level0F", "c_L_level0D", "c_L_level1")
    assert hp.PREDICTION_COLUMNS[:5] == ("token", "frame_id", "template", "p_c", "p_t") and {"row_level0", "F_hat", "Pi_hat", "dT_hat", "dT_frozen", "row_template_head", "c_L_level0D"} <= set(hp.PREDICTION_COLUMNS)
    assert set(hp.frozen_floors()) >= {"y_row_r2", "y_pi_spearman", "y_pi_r2", "y_dt_r2", "frame_guard_r2", "y3_r2_max", "y3_margin", "x3_identity_tolerance", "dt_identity_tolerance", "recovery_tolerance"}


def test_extract_replication_check():
    entry = {"c_dA": 0.02, "c_hat_016": 0.02, "delta_A_pc": {key: 0.1 for key in fch.HEAD_KEYS}, "c_L": 0.1, "c_M": 0.1, "c_H": 0.0, "c_L_level0F": 0.1}
    assert hp.check_extract_replication({"a|f": entry}, {"a|f": entry})["passed"]
    with pytest.raises(pm.IncidentError, match="c_L_level0F"):
        hp.check_extract_replication({"a|f": {**entry, "c_L_level0F": 0.1 + 2e-6}}, {"a|f": entry})
    with pytest.raises(pm.IncidentError, match="lack"):
        hp.check_extract_replication({}, {"a|f": entry})
    mapped = hp.extract_entry_016({"c": 0.02, "prediction": {"c_hat": 0.02}, "self": {key: 0.1 for key in fch.HEAD_KEYS}, "c_L": 0.1, "c_M": 0.1, "c_H": 0.0, "ladder": {"c_L_level0F": 0.1}})
    assert mapped == entry


def test_outcome_labels():
    base = {"precondition_Y1": {"passed": True}, "precondition_Y2": {"passed": True}, "Y1": {"test": {"passed": True}}, "Y2": {"test": {"passed": False}}, "Y3": {"evaluable": True, "rejected": True}}
    assert hp.outcome(base)["label"] == "HEAD_PATTERN_PREDICTED_TOKENS | HEAD_PATTERN_NOT_PREDICTED_FRAMES_CONDITIONAL | FROZEN_PATTERN_REJECTED"
    assert hp.outcome({**base, "Y3": {"evaluable": False, "rejected": False}})["Y3"] == "FROZEN_PATTERN_NOT_EVALUABLE" and hp.outcome({**base, "precondition_Y2": {"passed": False}})["Y2"] == "PRECONDITION_FAILED_FRAMES"
    assert hp.outcome({**base, "Y3": {"evaluable": True, "rejected": False}})["Y3"] == "FROZEN_PATTERN_NOT_REJECTED"


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
    pool = hp.build_pool_017(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016)
    return manifest, digests, pool


def toy_tokenizer_017(manifest, pool):
    base = toy_tokenizer_016(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in hp.CANDIDATES.values() for word in words]
    for _, text in hp.FRESH_FRAMES:
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


def test_pool_017_and_committed_extract(inputs):
    manifest, digests, pool = inputs
    assert len(pool.tokens) == 207 and len(pool.frames) == 66 and pool.token_source["heavy"] == "confirmation-016" and pool.frame_origin["coordinated-adjective-016-4"] == "confirmation-016"
    assert sum(1 for frame in pool.frames if frame.p_t == frame.p_c) == 44 and all(frame.p_t == frame.p_c + 1 for frame in pool.frames_of("coordinated-adjective"))
    extract = hp.load_inherited_extract(ROOT / hp.INHERITED_016_EXTRACT_RELATIVE_PATH, digests=digests, expected_size=hp.EXPECTED_EXTRACT_SIZE_016)
    keys = set(extract["entries"])
    assert "heavy|cardinal-1" in keys and "heavy|quantifier-016-3" in keys and "tenth|cardinal-1" in keys and "same|cardinal-1" in keys and "cardinal:sg|cardinal-1" not in keys
    assert set(extract["stage1_state_digests"]) == {f"{t}-016-{i}" for t in pm.TEMPLATE_ORDER for i in (1, 2, 3, 4)} and set(next(iter(extract["entries"].values()))) == {"c_dA", "c_hat_016", "delta_A_pc", "c_L", "c_M", "c_H", "c_L_level0F"}
    with pytest.raises(ValueError):
        hp.load_inherited_extract(ROOT / hp.INHERITED_016_EXTRACT_RELATIVE_PATH, digests=digests, expected_size=10)


def test_confirmation_policy_with_twelve_frames(inputs, tmp_path):
    manifest, digests, pool = inputs
    tokenizer = toy_tokenizer_017(manifest, pool)
    payload = hp.build_confirmation_payload(tokenizer, pool, digests)
    words = [token["word"] for token in payload["tokens"]]
    assert len(words) == 24 and words[:5] == ["further", "usual", "adjacent", "nearby", "leading"] and words[5:9] == ["twice", "once", "score", "trio"] and words[-6:] == ["violet", "metal", "paper", "leather", "blunt", "costly"]
    assert len(payload["token_prompts"]) == 24 * 12 and len(payload["exposed_frame_prompts"]) == 24 * 66
    confirmation = hp.validate_confirmation(payload, pool, digests)
    assert len(confirmation.all_prompts) == 24 + 288 + 1584 and [frame.frame_id for frame in confirmation.frames][:4] == ["cardinal-017-1", "cardinal-017-2", "cardinal-017-3", "cardinal-017-4"]
    assert all(frame.p_t == frame.p_c + 1 for frame in confirmation.frames if frame.template_id == "coordinated-adjective") and all(frame.p_t == frame.p_c for frame in confirmation.frames if frame.template_id != "coordinated-adjective")
    path = tmp_path / "c.json"
    assert hp.freeze_confirmation(path, tokenizer, pool, digests) == payload["content_sha256"]
    with pytest.raises(FileExistsError):
        hp.freeze_confirmation(path, tokenizer, pool, digests)
    tampered = dict(payload)
    tampered["tokens"] = list(reversed(payload["tokens"]))
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in tampered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError):
        hp.validate_confirmation(tampered, pool, digests)


def _fake_setup(pool):
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in hp.PROGRAM_LAYERS}
    small = cs.Pool008(pool.frames[:2] + pool.frames_of("quantifier")[:1] + pool.frames_of("coordinated-adjective")[:2], pool.frame_origin, pool.tokens[:6], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    cache = pm.PromptCache(model, tuple(small.nouns))
    axes = cs.stage_axes(cache, weights, small)
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = lc.CorrectionRead(ra.read_weight(head, axes["T"]).double(), axes["R0"].direction.double(), dict(pool.reference_ids), plural_ids)
    states = {frame.frame_id: hp.capture_frame_017(model, head, small.reference_prompt(frame), small.single_nouns, axes["T"]) for frame in small.frames}
    bases = lc.template_bases(small.frames, {fid: (s.state.x1, s.state.x2) for fid, s in states.items()})
    bases_3, counts = hp.layer3_bases(small.frames, states)
    fpm = ap.FrozenPatternModel(read, lw, heads, bases)
    fcm = fch.FrameChannelModel(read, lw, {layer: programs[layer] for layer in hp.UPSTREAM_LAYERS}, bases, axes["R0"].direction.double())
    hcm = hp.HeadChainModel(fcm, programs[hp.HEAD_LAYER], bases_3, axes["T"].direction.double())
    context = hp.AnalysisContext(hcm, fch.AnalysisContext(fcm, fpm, heads, weights, axes["T"]), weights, axes["T"])
    return model, weights, head, lw, heads, programs, small, axes, plural_ids, read, states, bases, bases_3, hcm, context


def test_exact_chain_variants_identities_and_boundary_on_the_fake(inputs, monkeypatch):
    manifest, digests, pool = inputs
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model, weights, head, lw, heads, programs, small, axes, plural_ids, read, states, bases, bases_3, hcm, context = _fake_setup(pool)
    assert bases_3["coordinated-adjective"]["p_t"] is not None and bases_3["cardinal"]["p_t"] is None
    for frame in (small.frames[0], small.frames_of("coordinated-adjective")[0]):
        state = states[frame.frame_id]
        template = frame.template_id
        assert (state.p_t == state.p_c) == (template != "coordinated-adjective") and len(state.x3_all) == state.p_t + 1 and len(state.x1_all) == state.p_t + 1 and state.A3.shape == (state.p_t + 1,)
        assert hp.check_reference_row(state, programs[3]) < 1e-5
        name, token_id = next((n, t) for n, t in small.tokens if t not in (pool.reference_ids[template], plural_ids[template]))
        plural = hp.measure_pair(model, weights, head, state, pool.plural_cue[template], plural_ids[template], axes["R0"], axes["T"], small.single_nouns)
        record = hp.measure_pair(model, weights, head, state, name, token_id, axes["R0"], axes["T"], small.single_nouns)
        analysis = hp.analyse_pair_017(record, plural, context=context, state=state)
        assert analysis is not None and analysis["cue_final"] == (state.p_t == state.p_c) and analysis["p_t"] == state.p_t
        ids = analysis["identities"]
        # I4–I7: the exact chain reproduces the captured layer-3 residuals at both positions, the head's patched row and the measured ΔT; ΔT = F + Π from the captures.
        assert ids["I4_x3"] < 1e-5 and ids["I5_head_row"] < 1e-5 and ids["I6_dT"] < 1e-4 and ids["I7_split"] < 1e-5 and ids["head_level1_recovery"] < 1e-9
        assert ids["I1_patched_rows"] < 1e-5 and ids["I2_chain"] < 1e-5 and ids["level1_recovery"] < 1e-9  # Experiment 016's recovery keeps its own key
        assert analysis["statistics"]["level1"]["ss_res"] < 1e-9 and analysis["rungs"]["level1"]["dT"] == pytest.approx(analysis["dT"], abs=1e-4) and analysis["rungs"]["level1"]["Pi"] == pytest.approx(analysis["Pi"], abs=1e-4)
        assert analysis["F"] + analysis["Pi"] == pytest.approx(analysis["dT"], abs=1e-5)
        p = analysis["prediction"]
        assert set(p) == set(hp.PREDICTION_COLUMNS[3:]) and p["dT_hat"] == pytest.approx(p["F_hat"] + p["Pi_hat"]) and p["dT_frozen"] == p["F_hat"] and p["self_level0"] == pytest.approx(p["row_level0"][state.p_c])
        for key in ("row_level0", "row_exact_head", "row_no_D", "row_template_head"):
            assert len(p[key]) == state.p_t + 1 and sum(p[key]) == pytest.approx(0.0, abs=1e-9)
        assert set(p["sigma_ratio_3"]) == {str(pos) for pos in state.positions} == set(analysis["x3_remainder"]) == set(analysis["scale_remainder"])
        assert set(analysis["rungs"]) == {"exact_head", "no_D", "template_head", "level1"} and set(analysis["statistics"]) == set(hp.ALL_RUNGS) and set(analysis["ladder"]) == {*hp.LADDER_KEYS, "level1_error"}
        # Channel D: the decoded c_L with block 2 at the frame's operating point reads the predicted layer correction; the −D rung and Level 0 differ only through block 2's MLP.
        parts = hcm.parts(weights, state.x1_all, state.x2_all, state.x3_all, state.p_c, state.p_t, token_id, template, variants=hp.VARIANTS)
        up0, upD = parts["variants"]["level0"]["upstream"], parts["variants"]["no_D"]["upstream"]
        assert up0["dx2"][state.p_c] == pytest.approx(upD["dx2"][state.p_c], abs=1e-12) and p["c_L_level0D"] == pytest.approx(read.inner(up0["dx3"][state.p_c] - up0["delta_e"]) / up0["denominator"])
        # With the layer-3 base equal to the frame's own residuals, the template-head rung is inert: it equals Level 0.
        own = hp.HeadChainModel(hcm.fcm, hcm.program3, {template: {"p_c": state.x3_all[state.p_c], "p_t": state.x3_all[state.p_t] if state.p_t != state.p_c else None}}, hcm.d_T)
        own_entry = own.predict_from_state(weights, state, token_id, template)
        assert own_entry["row_template_head"] == pytest.approx(own_entry["row_level0"], abs=1e-12) and own_entry["dT_template_head"] == pytest.approx(own_entry["dT_hat"], abs=1e-12)
        # Level 1 versus Level 0: the exact chain at the frame's own state is not the reduced model (the reduction is a real reduction on the fake as well).
        assert analysis["rungs"]["level1"]["row"] != pytest.approx(p["row_level0"], abs=1e-12) or analysis["rungs"]["level1"]["dT"] != pytest.approx(p["dT_hat"], abs=1e-12)
        # The boundary: the table is computed with every capture and intervention entry point disabled; a poisoned patched capture at both positions cannot change it.
        locked = {f.frame_id: hp.locked_state(states[f.frame_id]) for f in small.frames}
        with pytest.MonkeyPatch.context() as guard:
            for attr in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
                if hasattr(pm, attr):
                    guard.setattr(pm, attr, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a prediction touched the network")))
            table = hp.prediction_table(hcm, weights, locked, small.frames, [{"word": name, "token_id": token_id}], list(pm.TEMPLATE_ORDER))
        mine = next(row for row in table if row["frame_id"] == frame.frame_id)
        assert len(table) == len(small.frames) and set(mine) == set(hp.PREDICTION_COLUMNS) and atp._max_numeric_difference(mine, hp.prediction_row(name, frame.frame_id, template, p), "row") < 1e-12
        poisoned = {key: value * 3.0 + 1.0 for key, value in record.extra.items()}  # the residuals at p_c and p_t of every layer, the rows and the head sites: scaled and shifted
        fields = {f.name: getattr(record, f.name) for f in record.__dataclass_fields__.values()}
        for key, value in list(fields.items()):  # every numeric patched-run quantity of the record, not only the captures
            if isinstance(value, float):
                fields[key] = value * 3.0 + 1.0
            elif isinstance(value, dict) and value and all(isinstance(v, float) for v in value.values()):
                fields[key] = {k: v * 3.0 + 1.0 for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                fields[key] = value * 3.0 + 1.0
        fields["extra"] = poisoned
        poisoned_record = ra.Attribution(**fields)
        with pytest.raises(pm.IncidentError):  # the identities notice the poisoned capture …
            hp.analyse_pair_017(poisoned_record, plural, context=context, state=state)
        with pytest.MonkeyPatch.context() as guard:  # … while the prediction never looks at it: with the identity checks relaxed, the table entry is bit-identical to the clean one
            for module, names in ((hp, ("X3_IDENTITY_TOLERANCE", "ROW_IDENTITY_TOLERANCE", "DT_IDENTITY_TOLERANCE", "SPLIT_IDENTITY_TOLERANCE")), (atp, ("ROW_IDENTITY_TOLERANCE", "CHAIN_IDENTITY_TOLERANCE")),
                                  (ap, ("OV_IDENTITY_TOLERANCE",)), (lc, ("LADDER_IDENTITY_TOLERANCE", "NEURON_IDENTITY_TOLERANCE")), (ra, ("IDENTITY_TOLERANCE", "P1_CROSS_CHECK_TOLERANCE", "NEURON_SUM_TOLERANCE"))):
                for attr in names:
                    guard.setattr(module, attr, float("inf"))
            poisoned_analysis = hp.analyse_pair_017(poisoned_record, plural, context=context, state=state)
        assert poisoned_analysis is not None and atp._max_numeric_difference(poisoned_analysis["prediction"], p, "poisoned") == 0.0
        assert poisoned_analysis["dT"] != analysis["dT"] and poisoned_analysis["scale_remainder"] != analysis["scale_remainder"] and poisoned_analysis["x3_remainder"] == analysis["x3_remainder"]
        means = hp.token_means_from_table(table, [name])
        assert means[name]["n_frames"] == len(small.frames) and means[name]["n_cue_final"] == 3
        stats = hp.statistics_for([analysis], [p], [name])
        assert stats["pairs"]["rows"]["level0"]["n_entries"] == state.p_t + 1 and set(stats["ladder"]["ablation_cost_rows"]) == set(hp.ABLATIONS) and frame.frame_id in stats["per_frame"]
    # The exact chain on a coordinated frame: the propagation step reproduces the captured residual at p_t, not only at p_c (I4 is the maximum over both positions).
    coordinated = states[small.frames_of("coordinated-adjective")[0].frame_id]
    delta_e = read.encoding_delta(weights, small.tokens[0][1], "coordinated-adjective")
    exact = hp.exact_chain(programs, lw, coordinated.x1_all, coordinated.x2_all, coordinated.x3_all, coordinated.p_c, coordinated.p_t, delta_e)
    assert set(exact["dx3"]) == {coordinated.p_c, coordinated.p_t} and set(exact["dx"][2]) == {coordinated.p_c, coordinated.p_t} and set(exact["dx"][1]) == {coordinated.p_c}


def _rows(seed, length, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(length, generator=generator, dtype=torch.float64)
    return (scale * (raw - raw.mean())).tolist()


def _measured(word, frame_id, template, p_c, p_t, row, F, Pi):
    unit = {"n": 1, "sum_m": 0.0, "sum_m2": 1.0, "ss_res": 0.0, "tv_err": 0.0, "tv": 1.0}
    rung = {"row": row, "F": F, "Pi": Pi, "dT": F + Pi}
    return {"token": word, "frame_id": frame_id, "template": template, "p_c": p_c, "p_t": p_t, "cue_final": p_c == p_t, "row": row, "self": row[p_c], "F": F, "Pi": Pi, "dT": F + Pi,
            "c_dA": 0.0, "c_hat_016": 0.0, "c_L": 0.1, "c_M": 0.1, "c_H": 0.0, "ladder": {"c_L_level0F": 0.1, "c_L_level0D": 0.1, "c_L_level1": 0.1, "level1_error": 0.0}, "prediction": {},
            "rungs": {name: rung for name in ("exact_head", "no_D", "template_head", "level1")}, "statistics": {name: unit for name in hp.ALL_RUNGS},
            "x3_remainder": {str(p_c): 0.01}, "scale_remainder": {str(p_c): 0.0}, "sigma_3": {}, "identities": {}}


def _table_row(word, frame_id, template, p_c, p_t, row, F_hat, Pi_hat, rows_rung=None):
    rows_rung = row if rows_rung is None else rows_rung
    prediction = {"p_c": p_c, "p_t": p_t, "row_level0": row, "self_level0": row[p_c], "F_hat": F_hat, "Pi_hat": Pi_hat, "dT_hat": F_hat + Pi_hat, "dT_frozen": F_hat,
                  **{f"row_{name}": rows_rung for name in hp.RUNGS}, **{f"Pi_{name}": Pi_hat for name in hp.RUNGS}, **{f"dT_{name}": F_hat + Pi_hat for name in hp.RUNGS},
                  "sigma_ratio_3": {str(p_c): 1.1}, "x3_change_norm": {str(p_c): 1.0}, "c_L_level0D": 0.1, "c_L_level0F": 0.1}
    return hp.prediction_row(word, frame_id, template, prediction)


class _Confirmation:
    def __init__(self, fresh_frames, tokens):
        self.frames = [pm.Frame(t, fid, (1, 2), () if t in pm.CUE_FINAL_TEMPLATES else (7,), {"sg": 5, "pl": 6}, "x {cue}" + ("" if t in pm.CUE_FINAL_TEMPLATES else " y")) for fid, t in fresh_frames]
        self.tokens = [{"word": w, "category": "adjective", "licensed_frames": [fid for fid, _ in fresh_frames]} for w in tokens]


def test_scoring_floors_frame_guard_and_cue_final_frozen_rejection_on_synthetic_tables():
    exposed = [(f"e{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(6)]
    fresh = [(f"n{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(12)]
    words = [f"w{i}" for i in range(18)]
    p_c = 3
    p_t_of = lambda t: p_c if t in pm.CUE_FINAL_TEMPLATES else p_c + 1  # noqa: E731
    F_of = {w: 0.1 * (i - 9) for i, w in enumerate(words)}  # the value term (what the frozen pattern carries)
    Pi_of = {(w, fid): 0.15 * ((hash((w, fid)) % 7) - 3) for fid, _ in exposed + fresh for w in words}  # the pattern term: about half the value term's spread, zero on average
    confirmation = _Confirmation(fresh, words)
    true_rows = {(w, fid): _rows(hash((w, fid)) % 10_000, p_t_of(t) + 1) for fid, t in exposed + fresh for w in words}

    def measured(frames, Pi_scale=1.0):
        return {f"{w}|{fid}": _measured(w, fid, t, p_c, p_t_of(t), true_rows[(w, fid)], F_of[w], Pi_scale * Pi_of[(w, fid)]) for fid, t in frames for w in words}

    def tables(rows_hat, Pi_hat, valid=None, F_hat=None):
        F_hat = F_hat or (lambda w: F_of[w])
        make = lambda fid, t, w: _table_row(w, fid, t, p_c, p_t_of(t), rows_hat(w, fid), F_hat(w), Pi_hat(w, fid))  # noqa: E731
        locked = {"predictions": {"rows": [make(fid, t, w) for fid, t in exposed for w in words]}}
        stage1 = {"rows": [make(fid, t, w) for fid, t in fresh for w in words], "frames": {fid: {"template_id": t, "valid": True if valid is None else fid in valid, "template_defined": True} for fid, t in fresh}}
        return locked, stage1

    perfect_rows = lambda w, fid: true_rows[(w, fid)]  # noqa: E731
    perfect_Pi = lambda w, fid: Pi_of[(w, fid)]  # noqa: E731
    lock, stage1 = tables(perfect_rows, perfect_Pi)
    results = hp.score_confirmation(stage1, measured(exposed), measured(fresh), confirmation, lock)
    t1, t2 = results["Y1"]["test"], results["Y2"]["test"]
    assert t1["passed"] and t2["passed"] and t2["frame_guard"]["passed"] and len(t2["frame_guard"]["per_frame_rows"]) == 12 and "frame_guard" not in t1
    assert t1["row_entry_r2"] == pytest.approx(1.0) and t1["dT_pairs"]["r2"] == pytest.approx(1.0) and t1["Pi_token_means"]["r2"] == pytest.approx(1.0)
    y3 = results["Y3"]
    assert y3["evaluable"] and y3["rejected"] and y3["cue_final"]["gap"] == pytest.approx(1.0 - y3["cue_final"]["frozen_dT_r2"]) and y3["cue_final"]["frozen_dT_r2"] < hp.Y3_R2_MAX
    assert y3["cue_final"]["n_pairs"] == 18 * (4 + 8) and y3["coordinated_split"]["n_pairs"] == 18 * (2 + 4) and y3["coordinated_split"]["expected_not_rejected"]
    assert results["outcome"]["label"] == "HEAD_PATTERN_PREDICTED_TOKENS | HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL | FROZEN_PATTERN_REJECTED"
    assert set(results["Y2"]["per_frame"]) == {fid for fid, _ in fresh} and "ladder_both_sets" in results and results["ladder_both_sets"]["rows"]["level1"] == pytest.approx(1.0)
    # One collapsed fresh frame: the aggregate floors still pass, the guard fails the family and names the frame.
    collapsed = lambda w, fid: (_rows(hash((w, fid)) % 10_000, len(true_rows[(w, fid)]), scale=0.5) if fid == "n5" else true_rows[(w, fid)])  # noqa: E731
    lock_c, stage1_c = tables(collapsed, perfect_Pi)
    verdict = hp.score_confirmation(stage1_c, measured(exposed), measured(fresh), confirmation, lock_c)
    t2 = verdict["Y2"]["test"]
    assert t2["row_entry_r2"] >= hp.Y_ROW_R2 and t2["failing"] == ["frame_guard"] and set(t2["frame_guard"]["frames_below"]) == {"n5"} and verdict["outcome"]["Y2"] == "HEAD_PATTERN_NOT_PREDICTED_FRAMES_CONDITIONAL"
    # A tiny pattern term (a hundredth of its size): Level 0 still predicts everything, but the frozen pattern is within 0.05 of it — Y3 is NOT rejected while Y1 and Y2 pass (the independently failable Y3).
    tiny_lock, tiny_stage1 = tables(perfect_rows, lambda w, fid: 0.01 * Pi_of[(w, fid)])
    independent = hp.score_confirmation(tiny_stage1, measured(exposed, Pi_scale=0.01), measured(fresh, Pi_scale=0.01), confirmation, tiny_lock)
    assert independent["Y1"]["test"]["passed"] and independent["Y2"]["test"]["passed"] and not independent["Y3"]["rejected"] and independent["Y3"]["cue_final"]["frozen_dT_r2"] > hp.Y3_R2_MAX
    assert independent["outcome"]["label"].endswith("FROZEN_PATTERN_NOT_REJECTED")
    # Y3 is evaluated on the cue-final pairs only: a pattern term that matters only on coordinated pairs leaves the frozen pattern unrejected, while the split records that the criterion would reject there.
    template_of = dict(exposed + fresh)

    def scaled(frames):
        out = measured(frames)
        for entry in out.values():
            entry["Pi"] = (0.01 if entry["cue_final"] else 5.0) * entry["Pi"]
            entry["dT"] = entry["F"] + entry["Pi"]
        return out

    split_lock, split_stage1 = tables(perfect_rows, lambda w, fid: (0.01 if template_of[fid] in pm.CUE_FINAL_TEMPLATES else 5.0) * Pi_of[(w, fid)])
    split = hp.score_confirmation(split_stage1, scaled(exposed), scaled(fresh), confirmation, split_lock)
    assert split["Y1"]["test"]["passed"] and not split["Y3"]["rejected"] and split["Y3"]["cue_final"]["frozen_dT_r2"] > hp.Y3_R2_MAX and split["Y3"]["coordinated_split"]["would_be_rejected"]
    # Compressed rows (0.6 ×: R² 0.84) fail the row floor and the guard; the Π and ΔT floors are untouched.
    compressed = lambda w, fid: _rows(hash((w, fid)) % 10_000, len(true_rows[(w, fid)]), scale=0.6)  # noqa: E731
    lock_k, stage1_k = tables(compressed, perfect_Pi)
    failing = hp.score_confirmation(stage1_k, measured(exposed), measured(fresh), confirmation, lock_k)
    assert failing["Y1"]["test"]["failing"] == ["row_entry_r2"] and failing["Y1"]["test"]["row_entry_r2"] == pytest.approx(0.84) and failing["Y2"]["test"]["failing"] == ["row_entry_r2", "frame_guard"]
    # Reversed pattern terms fail the Π floors (Spearman and R²) while the row floor is untouched: the floors are separate claims.
    lock_r, stage1_r = tables(perfect_rows, lambda w, fid: -Pi_of[(w, fid)])
    reversed_ = hp.score_confirmation(stage1_r, measured(exposed), measured(fresh), confirmation, lock_r)["Y1"]["test"]
    assert "Pi_spearman" in reversed_["failing"] and "Pi_r2" in reversed_["failing"] and "row_entry_r2" not in reversed_["failing"]
    # Y2 precondition: seven valid fresh frames → PRECONDITION_FAILED_FRAMES while Y1 is unaffected; invalid frames run no fresh cue.
    seven = [fid for fid, _ in fresh][:7]
    lock_p, stage1_p = tables(perfect_rows, perfect_Pi, valid=seven)
    partial = {k: v for k, v in measured(fresh).items() if k.split("|")[1] in seven}
    verdict = hp.score_confirmation(stage1_p, measured(exposed), partial, confirmation, lock_p)
    assert verdict["outcome"]["Y2"] == "PRECONDITION_FAILED_FRAMES" and verdict["outcome"]["Y1"] == "HEAD_PATTERN_PREDICTED_TOKENS"
    # Without cue-final pairs Y3 is not evaluable.
    coordinated_frames = [(fid, t) for fid, t in exposed + fresh if t not in pm.CUE_FINAL_TEMPLATES]
    only_co_lock = {"predictions": {"rows": [r for r in lock["predictions"]["rows"] if r["template"] not in pm.CUE_FINAL_TEMPLATES]}}
    only_co_stage1 = {"rows": [r for r in stage1["rows"] if r["template"] not in pm.CUE_FINAL_TEMPLATES], "frames": {fid: e for fid, e in stage1["frames"].items() if e["template_id"] not in pm.CUE_FINAL_TEMPLATES}}
    only_co = hp.score_confirmation(only_co_stage1, {k: v for k, v in measured(exposed).items() if not v["cue_final"]}, {k: v for k, v in measured(fresh).items() if not v["cue_final"]}, confirmation, only_co_lock)
    assert only_co["outcome"]["Y3"] == "FROZEN_PATTERN_NOT_EVALUABLE" and only_co["Y3"]["cue_final"]["n_pairs"] == 0 and len(coordinated_frames) == 6


def test_lock_reproduction_refusal_and_predictions_rendering():
    row = _table_row("w", "f", "cardinal", 3, 3, _rows(1, 4), 0.2, 0.05)
    lock = {"predictions": {"rows": [row]}}
    assert hp.assert_lock_predictions_reproduced(lock, {"rows": [json.loads(json.dumps(row))]}) == 0.0
    perturbed = json.loads(json.dumps(row))
    perturbed["row_no_D"][1] += 1e-6
    with pytest.raises(hp.PhaseError, match="nothing was executed"):
        hp.assert_lock_predictions_reproduced(lock, {"rows": [perturbed]})
    ratio = json.loads(json.dumps(row))
    ratio["sigma_ratio_3"]["3"] += 1e-6
    with pytest.raises(hp.PhaseError, match="nothing was executed"):
        hp.assert_lock_predictions_reproduced(lock, {"rows": [ratio]})
    moved = json.loads(json.dumps(row))
    moved["p_t"] = 4
    with pytest.raises(hp.PhaseError, match="ordered differently"):
        hp.assert_lock_predictions_reproduced(lock, {"rows": [moved]})
    full_lock = {"run_id": "r", "protocol_code_commit": "c" * 40, "confirmation_017_sha256": "s" * 64, "lock_016_sha256": "l" * 64, "sigma_T": 1.014,
                 "program": {str(layer): {"d_head": 64, "rotary_dim": 16, "rotary_base": 10000.0} for layer in hp.PROGRAM_LAYERS},
                 "tokens": [{"word": "w", "category": "adjective"}], "predictions": {"rows": [row], "token_means": hp.token_means_from_table([row], ["w"])}}
    text = hp.render_predictions(full_lock)
    assert "preregistered predictions" in text and "| w | adjective | 1 | **0.2500**" in text and "frame guard" in text and "cue-final" in text and "L03.H04" in text


def _synthetic_program(seed: int, *, d_model: int = 12, n_heads: int = 6, d_head: int = 8, rotary_dim: int = 4) -> atp.LayerProgram:
    generator = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)  # noqa: E731
    return atp.LayerProgram(3, 1.0 + 0.1 * r(d_model), 0.1 * r(d_model), 1e-5, 0.3 * r(n_heads, d_model, d_head), 0.1 * r(n_heads, d_head), 0.3 * r(n_heads, d_model, d_head), 0.1 * r(n_heads, d_head),
                            0.3 * r(n_heads, d_model, d_head), 0.1 * r(n_heads, d_head), 0.3 * r(n_heads, d_head, d_model), rotary_dim, 10000.0)


@pytest.mark.parametrize("p_c,p_t", [(4, 4), (4, 5)])
def test_reduced_head_at_the_frames_own_base_recovers_the_exact_head_with_rotation_and_biases(p_c, p_t):
    """With rotary rotation and Q/K/V biases (which the fake lacks): the reduced form at a base equal to the frame's own x₃ is the exact head program, at p_t = p_c and one position later."""
    program = _synthetic_program(11)
    generator = torch.Generator().manual_seed(5)
    x3_all = [torch.randn(12, generator=generator, dtype=torch.float64) * 2.0 for _ in range(p_t + 1)]
    dx3 = {pos: torch.randn(12, generator=generator, dtype=torch.float64) for pos in sorted({p_c, p_t})}
    d_T = torch.randn(12, generator=generator, dtype=torch.float64)
    d_T = d_T / d_T.norm()
    bases = {"cardinal": {"p_c": x3_all[p_c], "p_t": x3_all[p_t] if p_t != p_c else None}}
    model = hp.HeadChainModel(None, program, bases, d_T)
    rr3 = atp.ReferenceRow(program, x3_all)
    exact = model.head(rr3, x3_all, dx3, p_c, p_t, "cardinal", hp.EXACT_HEAD)
    reduced = model.head(rr3, x3_all, dx3, p_c, p_t, "cardinal", hp.LEVEL0)
    template_only = model.head(rr3, x3_all, dx3, p_c, p_t, "cardinal", hp.TEMPLATE_HEAD)
    for other in (reduced, template_only):
        assert other["rows"] == pytest.approx(exact["rows"], abs=1e-12) and other["F"] == pytest.approx(exact["F"], abs=1e-12) and other["Pi"] == pytest.approx(exact["Pi"], abs=1e-12) and other["dT"] == pytest.approx(exact["dT"], abs=1e-12)
    assert exact["rows"].shape == (6, p_t + 1) and exact["row"].shape == (p_t + 1,) and float(exact["row"].abs().max()) > 1e-3
    # A different base (a non-uniform perturbation: a uniform shift is removed by the LayerNorm centering) makes the reduction a real reduction, and the reduced rows still sum to one.
    bump = torch.randn(12, generator=generator, dtype=torch.float64)
    shifted = hp.HeadChainModel(None, program, {"cardinal": {"p_c": x3_all[p_c] + bump, "p_t": (x3_all[p_t] + bump) if p_t != p_c else None}}, d_T)
    other = shifted.head(rr3, x3_all, dx3, p_c, p_t, "cardinal", hp.LEVEL0)
    assert other["rows"] != pytest.approx(exact["rows"], abs=1e-9) and other["rows"].sum(-1) == pytest.approx(torch.ones(6), abs=1e-12)


def test_head_terms_separate_the_value_term_from_the_pattern_term():
    """F is the reference row carrying the value change (zero when only the pattern moves); Π is the changed row carrying the changed values (zero when only the values move); ΔT = F + Π."""
    program = _synthetic_program(3)
    generator = torch.Generator().manual_seed(9)
    x3_all = [torch.randn(12, generator=generator, dtype=torch.float64) for _ in range(5)]
    rr3 = atp.ReferenceRow(program, x3_all)
    d_T = torch.randn(12, generator=generator, dtype=torch.float64)
    changed_values = rr3.values.clone()
    changed_values[:, 4] = program.v(program.normalize(x3_all[4] + torch.randn(12, generator=generator, dtype=torch.float64)))
    moved_rows = torch.softmax(rr3.scores_ref + torch.randn(6, 5, generator=generator, dtype=torch.float64), dim=-1)
    value_only = hp.head_terms(program, rr3, rr3.A_ref, changed_values, d_T)
    pattern_only = hp.head_terms(program, rr3, moved_rows, rr3.values, d_T)
    both = hp.head_terms(program, rr3, moved_rows, changed_values, d_T)
    assert value_only["Pi"] == 0.0 and value_only["F"] == pytest.approx(value_only["dT"]) and abs(value_only["dT"]) > 1e-6
    assert pattern_only["F"] == 0.0 and pattern_only["Pi"] == pytest.approx(pattern_only["dT"]) and abs(pattern_only["dT"]) > 1e-6
    assert both["F"] + both["Pi"] == pytest.approx(both["dT"], abs=1e-12) and both["F"] == pytest.approx(value_only["F"]) and both["Pi"] != pytest.approx(pattern_only["Pi"])  # Π carries the changed values
    with_captured_reference = hp.head_terms(program, rr3, moved_rows, changed_values, d_T, row_ref=rr3.A_ref[hp.HEAD_INDEX] + 1e-3)
    assert with_captured_reference["F"] != pytest.approx(both["F"], abs=1e-9)  # the supplied reference row is the one used
