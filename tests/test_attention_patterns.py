"""Experiment 015: the weight-only Q/K/V program and its identities on the fake, the rotation lemma, the token-local model and its companions, the invariant, the pooled statistics, the confirmation policy, floors, the comparator rule, and the extracts."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from neural_decompiler import attention_paths as ap
from neural_decompiler import attention_patterns as atp
from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import encoding_read as er
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import neuron_feature as nf
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from plural_fakes import TinyPlural
from test_neuron_feature import toy_tokenizer_014

ROOT = Path(__file__).resolve().parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


def test_frozen_constants_match_the_design():
    assert atp.Y_SPEARMAN == 0.80 and atp.Y_R2 == 0.50 and atp.Y_ENTRY_R2 == 0.50 and atp.Y3_R2_MAX == 0.30 and atp.Y3_ENTRY_R2_MAX == 0.30 and atp.COMPARATOR_MARGIN == 0.10
    assert atp.MIN_VALID_FRAMES == 4 and atp.MIN_VALID_FRAMES_PER_TOKEN == 3 and atp.MIN_SCORED_TOKENS == 16 and atp.FRAME_CUE_EFFECT_RATE == 108 / 120
    assert atp.ROW_IDENTITY_TOLERANCE == 1e-4 and atp.CHAIN_IDENTITY_TOLERANCE == 1e-4 and atp.REPLICATION_TOLERANCE == 1e-6 and atp.LOCKED_STATE_TOLERANCE == 1e-9 and atp.LOCK_PREDICTION_TOLERANCE == 1e-9
    assert atp.RUNTIME_SEED == 20260916 and atp.CONTROL_SEED == 20260924 and len(atp.FRESH_FRAMES) == 6 and atp.EXPECTED_LEDGER_SIZE_014 == 4884 and atp.EXPECTED_EXTRACT_SIZE_013 == 3732
    assert atp.QUOTAS == {"determiner-like": 5, "ordinal-or-numeral": 5, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 5}
    exposed = {token["word"] for path in (er.CONFIRMATION_RELATIVE_PATH, lc.CONFIRMATION_RELATIVE_PATH, ap.CONFIRMATION_RELATIVE_PATH, nf.CONFIRMATION_RELATIVE_PATH) for token in json.loads((ROOT / path).read_text())["tokens"]}
    assert not (set(sum(atp.CANDIDATES.values(), ())) & exposed), "015 candidates must not be earlier frozen tokens"
    assert atp.OUTCOME_Y3 == ("AXIS_ONLY_REJECTED", "AXIS_ONLY_NOT_REJECTED", "AXIS_ONLY_NOT_EVALUABLE") and atp.OUTCOME_Y1[0] == "PATTERN_CHANGE_PREDICTED_TOKENS" and atp.OUTCOME_Y2[0] == "PATTERN_CHANGE_PREDICTED_FRAMES_CONDITIONAL"
    assert set(atp.frozen_floors()) == {"y_spearman", "y_r2", "y_entry_r2", "y3_r2_max", "y3_entry_r2_max", "comparator_margin", "min_valid_frames", "min_valid_frames_per_token", "min_scored_tokens", "frame_cue_effect_rate", "head_stage_floor",
                                        "row_identity_tolerance", "chain_identity_tolerance"}
    assert atp.PREDICTION_COLUMNS[:4] == ("token", "frame_id", "template", "p_c") and atp.ROW_MODELS == ("level0", "axis", "diag_L0")


def _rotated_program(rotary_dim: int, d_head: int = 16) -> atp.LayerProgram:
    generator = torch.Generator().manual_seed(1)
    z = torch.zeros(8, d_head, dtype=torch.float64)
    return atp.LayerProgram(1, torch.ones(8, dtype=torch.float64), torch.zeros(8, dtype=torch.float64), 1e-5, torch.randn(8, 8, d_head, generator=generator, dtype=torch.float64), z, torch.randn(8, 8, d_head, generator=generator, dtype=torch.float64), z,
                            torch.randn(8, 8, d_head, generator=generator, dtype=torch.float64), z, torch.randn(8, d_head, 8, generator=generator, dtype=torch.float64), rotary_dim, 10000.0)


def test_rotary_rotation_convention_and_the_same_position_lemma():
    program = _rotated_program(8)
    x = torch.randn(8, 16, dtype=torch.float64)
    assert torch.equal(program.rotate(x, 0), x)  # R_0 is the identity
    # HuggingFace half-split: pair (i, i + 4) rotated by p · base^(−2i/8); coordinates 8..15 pass through.
    p = 3
    rotated = program.rotate(x, p)
    for i in range(4):
        theta = torch.tensor(p * 10000.0 ** (-2 * i / 8), dtype=torch.float64)
        assert rotated[:, i] == pytest.approx(x[:, i] * torch.cos(theta) - x[:, i + 4] * torch.sin(theta), abs=1e-12)
        assert rotated[:, i + 4] == pytest.approx(x[:, i] * torch.sin(theta) + x[:, i + 4] * torch.cos(theta), abs=1e-12)
    assert torch.equal(rotated[:, 8:], x[:, 8:])
    a, b = torch.randn(8, 16, dtype=torch.float64), torch.randn(8, 16, dtype=torch.float64)
    for position in (1, 5, 11):
        assert (program.rotate(a, position) * program.rotate(b, position)).sum(-1) == pytest.approx((a * b).sum(-1), abs=1e-12)  # (R_p a)ᵀ(R_p b) = aᵀb
    odd = make_fake_model()
    odd.cfg.rotary_dim = 7
    with pytest.raises(ValueError, match="rotary dimension"):
        atp.LayerProgram.from_model(odd, 1)  # an odd rotary dimension is rejected by the model reader
    adjacent = make_fake_model()
    adjacent.cfg.rotary_adjacent_pairs = True
    with pytest.raises(ValueError, match="half-split"):
        atp.LayerProgram.from_model(adjacent, 1)
    fake = atp.LayerProgram.from_model(make_fake_model(), 1)
    assert fake.rotary_dim == 0 and fake.d_head == 1 and fake.n_heads == 8 and torch.equal(fake.b_Q, torch.zeros(8, 1, dtype=torch.float64)) and fake.record()["rotary_dim"] == 0


def test_level0_diagonal_only_change_equals_proportional_redistribution():
    program = _rotated_program(8)
    residuals = [torch.randn(8, dtype=torch.float64) for _ in range(4)]
    row = atp.ReferenceRow(program, residuals)
    assert row.A_ref.sum(-1) == pytest.approx(torch.ones(8), abs=1e-12) and row.p_c == 3
    ds = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
    moved = row.row_from_changes(torch.zeros(8, 16, dtype=torch.float64), ds)
    assert moved == pytest.approx(row.proportional(moved[:, 3]), abs=1e-12)  # no off-diagonal coupling → the other keys keep their proportions
    assert row.row_from_changes(torch.zeros(8, 16, dtype=torch.float64), torch.zeros(8, dtype=torch.float64)) == pytest.approx(row.A_ref, abs=1e-12)
    assert row.row_from_state(row.normed_pc) == pytest.approx(row.A_ref, abs=1e-12) and row.row_additive(row.normed_pc) == pytest.approx(row.A_ref, abs=1e-12)
    v_pc = row.values[:, 3]
    assert row.output_change(row.A_ref, v_pc) == pytest.approx(torch.zeros(8), abs=1e-12) and row.pattern_change_vectors(row.A_ref, v_pc) == pytest.approx(torch.zeros(8, 8), abs=1e-12)


def test_row_statistics_pooled_r2_and_tv_ratio():
    measured = torch.tensor([[0.1, -0.1, 0.0], [0.3, -0.2, -0.1]], dtype=torch.float64)
    exact = atp.row_statistics(measured, measured)
    assert exact["ss_res"] == 0.0 and exact["tv_err"] == 0.0 and exact["tv"] == pytest.approx(0.4) and exact["n"] == 6
    half = atp.row_statistics(0.5 * measured, measured)
    assert atp.pooled_entry_r2([exact, exact]) == pytest.approx(1.0) and atp.pooled_entry_r2([half]) == pytest.approx(0.75) and atp.tv_ratio([half]) == pytest.approx(0.5)
    zero = atp.row_statistics(torch.zeros_like(measured), measured)
    assert atp.pooled_entry_r2([zero]) == pytest.approx(0.0) and atp.tv_ratio([zero]) == pytest.approx(1.0)
    assert atp.pooled_entry_r2([atp.row_statistics(torch.zeros(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64))]) is None
    summary = atp.pooled_summary([half, exact])
    assert summary["n_pairs"] == 2 and summary["n_entries"] == 12 and 0.75 < summary["entry_r2"] < 1.0


def _entry(value):
    return {"c_L": value, "c_M": value, "c_H": 0.0, "c_k": {key: 0.0 for key in ra.COMPONENT_ORDER}}


def test_replication_checks():
    recorded = {"a|f": _entry(0.5)}
    assert atp.check_ledger_replication({"a|f": _entry(0.5), "b|f": _entry(0.1)}, recorded)["passed"]
    with pytest.raises(pm.IncidentError, match="deviates"):
        atp.check_ledger_replication({"a|f": _entry(0.5 + 2e-6)}, recorded)
    with pytest.raises(pm.IncidentError, match="lacks"):
        atp.check_ledger_replication({}, recorded)
    pattern = {"a|f": {"delta_A_pc": {key: 0.1 for key in atp.HEAD_KEYS}, "direct_pattern_change": 0.02}}
    assert atp.check_extract_replication(pattern, pattern)["passed"]
    off = {"a|f": {"delta_A_pc": {**pattern["a|f"]["delta_A_pc"], "L02.H06": 0.1 + 2e-6}, "direct_pattern_change": 0.02}}
    with pytest.raises(pm.IncidentError, match="L02.H06"):
        atp.check_extract_replication(off, pattern)


def test_outcome_labels():
    base = {"precondition_Y1": {"passed": True}, "precondition_Y2": {"passed": True}, "Y1": {"test": {"passed": True}}, "Y2": {"test": {"passed": False}}, "Y3": {"evaluable": True, "rejected": True}}
    assert atp.outcome(base)["label"] == "PATTERN_CHANGE_PREDICTED_TOKENS | PATTERN_CHANGE_NOT_PREDICTED_FRAMES_CONDITIONAL | AXIS_ONLY_REJECTED"
    assert atp.outcome({**base, "Y3": {"evaluable": False, "rejected": False}})["Y3"] == "AXIS_ONLY_NOT_EVALUABLE" and atp.outcome({**base, "Y3": {"evaluable": True, "rejected": False}})["Y3"] == "AXIS_ONLY_NOT_REJECTED"
    assert atp.outcome({**base, "precondition_Y2": {"passed": False}})["Y2"] == "PRECONDITION_FAILED_FRAMES" and atp.outcome({**base, "precondition_Y1": {"passed": False}})["Y1"] == "PRECONDITION_FAILED_TOKENS"


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
    pool = atp.build_pool_015(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014)
    return manifest, digests, pool


def toy_tokenizer_015(manifest, pool):
    base = toy_tokenizer_014(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in atp.CANDIDATES.values() for word in words]
    for _, text in atp.FRESH_FRAMES:
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


def test_pool_015_and_committed_extracts(inputs):
    manifest, digests, pool = inputs
    assert len(pool.tokens) == 159 and len(pool.frames) == 48 and pool.token_source["dozens"] == "confirmation-014" and pool.frame_origin["cardinal-014-1"] == "confirmation-014"
    ledger = atp.load_inherited_ledger(ROOT / atp.INHERITED_014_LEDGER_RELATIVE_PATH, digests=digests, expected_size=atp.EXPECTED_LEDGER_SIZE_014)
    keys = set(ledger["entries"])
    assert "dozens|cardinal-014-1" in keys and "dozens|cardinal-1" in keys and "whom|cardinal-013-1" in keys and "same|cardinal-1" in keys and "cardinal:sg|cardinal-1" not in keys
    assert sum(1 for key in keys if key.split("|")[1].endswith(("-014-1", "-014-2"))) == 144
    extract = atp.load_inherited_extract(ROOT / atp.INHERITED_013_EXTRACT_RELATIVE_PATH, digests=digests, expected_size=atp.EXPECTED_EXTRACT_SIZE_013)
    assert set(extract["entries"]) < keys and set(next(iter(extract["entries"].values()))["delta_A_pc"]) == set(atp.HEAD_KEYS)
    with pytest.raises(ValueError):
        atp.load_inherited_ledger(ROOT / atp.INHERITED_014_LEDGER_RELATIVE_PATH, digests=digests, expected_size=10)
    with pytest.raises(ValueError):
        atp.load_inherited_extract(ROOT / atp.INHERITED_014_LEDGER_RELATIVE_PATH, digests=digests, expected_size=atp.EXPECTED_LEDGER_SIZE_014)  # the wrong kind


def test_confirmation_policy_with_two_prompt_lists(inputs, tmp_path):
    manifest, digests, pool = inputs
    tokenizer = toy_tokenizer_015(manifest, pool)
    payload = atp.build_confirmation_payload(tokenizer, pool, digests)
    words = [token["word"] for token in payload["tokens"]]
    assert len(words) == 24 and words[:5] == ["fourth", "fifth", "particular", "previous", "final"] and words[5:10] == ["sixth", "seventh", "eighth", "ninth", "tenth"]
    assert len(payload["token_prompts"]) == 24 * 6 and len(payload["exposed_frame_prompts"]) == 24 * 48
    confirmation = atp.validate_confirmation(payload, pool, digests)
    assert len(confirmation.all_prompts) == 12 + 144 + 1152 and [frame.frame_id for frame in confirmation.frames][:2] == ["cardinal-015-1", "cardinal-015-2"]
    path = tmp_path / "c.json"
    assert atp.freeze_confirmation(path, tokenizer, pool, digests) == payload["content_sha256"]
    with pytest.raises(FileExistsError):
        atp.freeze_confirmation(path, tokenizer, pool, digests)
    tampered = dict(payload)
    tampered["tokens"] = list(reversed(payload["tokens"]))
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in tampered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError):
        atp.validate_confirmation(tampered, pool, digests)


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
    tlm = atp.TokenLocalModel(read, lw, programs, bases, axes["R0"].direction.double())
    return model, weights, head, lw, heads, programs, small, axes, plural_ids, read, states, bases, fpm, tlm


def test_program_identities_models_and_invariant_on_the_fake(inputs, monkeypatch):
    manifest, digests, pool = inputs
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model, weights, head, lw, heads, programs, small, axes, plural_ids, read, states, bases, fpm, tlm = _fake_setup(pool)
    context = atp.AnalysisContext(tlm, fpm, heads, weights, axes["T"])
    frame = small.frames[0]
    state = states[frame.frame_id]
    template = frame.template_id
    rows = atp.reference_rows(programs, state.x1_all, state.x2_all)
    assert atp.check_reference_rows(rows, state) < 1e-5  # I1 on the reference run: the program reproduces the fake's captured rows
    # The program's values agree with the head weights Experiment 013 uses.
    key = ap.head_key(1, 3)
    assert rows[1].values[3, frame.p_c] == pytest.approx(heads.heads[key].value(state.x1).double(), abs=1e-12)
    plural = atp.measure_pair(model, weights, head, state, pool.plural_cue[template], plural_ids[template], axes["R0"], axes["T"], small.single_nouns)
    name, token_id = next((n, t) for n, t in small.tokens if t not in (pool.reference_ids[template], plural_ids[template]))
    record = atp.measure_pair(model, weights, head, state, name, token_id, axes["R0"], axes["T"], small.single_nouns)
    analysis = atp.analyse_pair_015(record, plural, context=context, state=state, rows=rows)
    assert analysis is not None
    ids = analysis["identities"]
    assert ids["I1_patched_rows"] < 1e-5 and ids["I2_chain"] < 1e-5 and ids["I3_head_split"] < 1e-5 and ids["I1_level1_rows"] < 1e-5  # Level 1 is the model itself
    assert analysis["c"] == pytest.approx(analysis["c_1"] + analysis["c_2"]) and analysis["c_level1"] == pytest.approx(analysis["c"], abs=1e-5)
    assert analysis["rung_statistics"]["level1"]["2"]["ss_res"] < 1e-10 and analysis["rung_statistics"]["level1"]["2"]["tv_err"] < 1e-5
    # The measured rows sum to zero per head (a change of a probability row) and the self entries are the recorded ΔA_pc.
    for layer in atp.HEAD_LAYERS:
        measured = atp.rows_from_json(analysis["rows"][str(layer)], layer)
        assert measured.sum(-1) == pytest.approx(torch.zeros(8), abs=1e-6)
        assert measured[:, frame.p_c].tolist() == pytest.approx([analysis["self"][ap.head_key(layer, h)] for h in range(8)], abs=1e-6)  # the same captured differences, float32 vs float64 subtraction
    p = analysis["prediction"]
    assert set(p) == set(atp.PREDICTION_COLUMNS[3:]) and p["c_hat"] == pytest.approx(p["c_hat_1"] + p["c_hat_2"]) and p["p_c"] == frame.p_c
    ladder = analysis["ladder"]
    assert ladder["c_L_level1"] == pytest.approx(analysis["c_L"], abs=1e-4) and ladder["level1_error"] < 1e-4 and ladder["c_M_level1"] == pytest.approx(analysis["c_M"], abs=1e-4) and ladder["c_H_level1"] == pytest.approx(analysis["c_H"], abs=1e-4)  # Level 1 accounts for all of c_L
    assert p["c_L_level0"] == pytest.approx(p["c_M_level0"] + p["c_H_level0"]) and ladder["c_L_level0"] == p["c_L_level0"] and ladder["c_013_own"] == pytest.approx(fpm.predict_from_state(weights, state, token_id, template)["c_L_hat"])
    for layer in atp.HEAD_LAYERS:
        for key in ("rows_level0", "rows_axis", "rows_diag_L0"):
            predicted = atp.rows_from_json(p[key][str(layer)], layer)
            assert predicted.shape == (8, frame.p_c + 1) and predicted.sum(-1) == pytest.approx(torch.zeros(8), abs=1e-9)
        assert [p["rows_level0"][str(layer)][ap.head_key(layer, h)][frame.p_c] for h in range(8)] == pytest.approx([p["self_level0"][ap.head_key(layer, h)] for h in range(8)])
        assert [p["rows_diag_L0"][str(layer)][ap.head_key(layer, h)][frame.p_c] for h in range(8)] == pytest.approx([p["self_level0"][ap.head_key(layer, h)] for h in range(8)])  # the comparator keeps Level 0's self weight
    # The statistics stored per pair are those of the table rows against the measured rows.
    assert analysis["statistics"]["level0"]["1"] == atp._statistics_of(analysis["rows"], p["rows_level0"], 1)
    # Level 0 equals Level 1 when the template base is the frame's own state.
    own = atp.TokenLocalModel(read, lw, programs, {template: (state.x1, state.x2)}, tlm.d_E)
    zero, one = own.level_zero(weights, rows, token_id, template), own.level_one(weights, rows, state.x1, state.x2, token_id, template)
    assert all(zero["rows"][layer] == pytest.approx(one["rows"][layer], abs=1e-12) for layer in atp.HEAD_LAYERS) and zero["c"][1] + zero["c"][2] == pytest.approx(one["c"][1] + one["c"][2], abs=1e-12)
    # The axis alternative equals Level 1 at layer 1 when the encoding change lies along d̂_E.
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(lc.CorrectionRead, "encoding_delta", lambda self, *args, **kwargs: 0.7 * tlm.d_E)
        one_axis = tlm.level_one(weights, rows, state.x1, state.x2, token_id, template)
    axis = tlm.axis_only(rows, {1: state.x1.double(), 2: state.x2.double()}, one_axis["dx"], read.denominator(weights, template))
    assert axis["rows"][1] == pytest.approx(one_axis["rows"][1], abs=1e-7)  # d̂_E is unit to float32 precision
    # The invariant: the prediction table is computed with every capture and intervention entry point disabled, and the locked state reproduces it.
    locked = {frame.frame_id: atp.locked_state(states[frame.frame_id]) for frame in small.frames}
    assert set(locked[frame.frame_id]) == {"p_c", "x1_all", "x2_all"} and len(locked[frame.frame_id]["x1_all"]) == frame.p_c + 1
    with pytest.MonkeyPatch.context() as guard:
        for attr in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, attr):
                guard.setattr(pm, attr, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a prediction touched the network")))
        table = atp.prediction_table(tlm, weights, locked, small.frames, [{"word": name, "token_id": token_id}], list(pm.TEMPLATE_ORDER))
    assert len(table) == 3 and set(table[0]) == set(atp.PREDICTION_COLUMNS) and atp._max_numeric_difference(table[0], atp.prediction_row(name, frame.frame_id, template, p), "row") < 1e-12
    means = atp.token_means_from_table(table, [name])
    assert means[name]["n_frames"] == 3 and set(means[name]["self_hat_mean"]) == set(atp.HEAD_KEYS)
    # Statistics over a small exposed set run end to end; the stored form drops the predicted rows and keeps the statistics.
    stats = atp.statistics_for([analysis], [p], [name])
    assert stats["n_pairs"] == 1 and stats["pairs"]["level0"]["1"]["n_entries"] == 8 * (frame.p_c + 1) and set(stats["rungs"]) >= set(atp.RUNGS) and set(stats["ladder"]["pairs"]) == set(atp.LADDER_KEYS)
    compact = atp.compact_analysis(analysis)
    assert set(compact["prediction"]) == set(atp.PREDICTION_COLUMNS[3:]) - {"rows_level0", "rows_axis", "rows_diag_L0"} and compact["statistics"] == analysis["statistics"] and compact["rows"] == analysis["rows"]


def _measured(word, frame_id, template, p_c, rows, c, self_values):
    return {"token": word, "frame_id": frame_id, "template": template, "p_c": p_c, "rows": rows, "self": self_values, "c_1": c / 2, "c_2": c / 2, "c": c, "c_level1": c,
            "c_L": 0.1, "c_M": 0.1, "c_H": 0.0, "c_k": {k: 0.0 for k in ra.COMPONENT_ORDER}, "P1": 0.5, "q_T": 0.5, "g_E": 0.4, "prediction": {}, "statistics": {},
            "rung_statistics": {name: {"1": {"n": 1, "sum_m": 0.0, "sum_m2": 1.0, "ss_res": 0.0, "tv_err": 0.0, "tv": 1.0}, "2": {"n": 1, "sum_m": 0.0, "sum_m2": 1.0, "ss_res": 0.0, "tv_err": 0.0, "tv": 1.0}} for name in atp.RUNGS},
            "rung_reads": {name: {"1": c / 2, "2": c / 2} for name in atp.RUNGS}, "diagonal_terms": {"1": {"dq_k": [1.0], "q_dk": [0.5], "dq_dk": [0.2]}, "2": {"dq_k": [1.0], "q_dk": [0.5], "dq_dk": [0.2]}},
            "ladder": {"c_012": 0.1, "c_013_own": 0.1, "c_L_level0": 0.1, "c_M_level0": 0.1, "c_H_level0": 0.0, "c_L_level1": 0.1, "c_M_level1": 0.1, "c_H_level1": 0.0, "level1_error": 0.0},
            "norm_ratio": {"1": 0.9, "2": 0.8}, "identities": {}}


def _rows(seed, p_c, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for layer in atp.HEAD_LAYERS:
        raw = torch.randn(8, p_c + 1, generator=generator, dtype=torch.float64)
        raw = raw - raw.mean(-1, keepdim=True)
        out[str(layer)] = atp.rows_to_json(scale * raw, layer)
    return out


def _table_row(word, frame_id, template, p_c, rows, c_hat, rows_axis, c_axis, rows_diag, c_diag):
    self_values = {ap.head_key(layer, h): rows[str(layer)][ap.head_key(layer, h)][p_c] for layer in atp.HEAD_LAYERS for h in range(8)}
    prediction = {"p_c": p_c, "rows_level0": rows, "self_level0": self_values, "c_hat_1": c_hat / 2, "c_hat_2": c_hat / 2, "c_hat": c_hat, "rows_axis": rows_axis, "c_axis": c_axis, "rows_diag_L0": rows_diag, "c_diag_L0": c_diag, "arrival_read": 0.3,
                  "c_L_level0": 0.1, "c_M_level0": 0.1, "c_H_level0": 0.0}
    return atp.prediction_row(word, frame_id, template, prediction)


class _Confirmation:
    def __init__(self, fresh_frames, tokens):
        self.frames = [pm.Frame(t, fid, (1, 2), () if t in pm.CUE_FINAL_TEMPLATES else (7,), {"sg": 5, "pl": 6}, "x {cue}" + ("" if t in pm.CUE_FINAL_TEMPLATES else " y")) for fid, t in fresh_frames]
        self.tokens = [{"word": w, "category": "adjective", "licensed_frames": [fid for fid, _ in fresh_frames]} for w in tokens]


def test_scoring_floors_axis_rejection_and_comparator_rule_on_synthetic_tables():
    exposed = [(f"e{i}", pm.TEMPLATE_ORDER[i % 3]) for i in range(6)]
    fresh = [("c1", "cardinal"), ("c2", "cardinal"), ("q1", "quantifier"), ("q2", "quantifier"), ("a1", "coordinated-adjective"), ("a2", "coordinated-adjective")]
    words = [f"w{i}" for i in range(18)]
    c_values = {w: 0.01 * (i - 9) for i, w in enumerate(words)}
    p_c = 3
    confirmation = _Confirmation(fresh, words)
    true_rows = {(w, fid): _rows(hash((w, fid)) % 10_000, p_c) for fid, _ in exposed + fresh for w in words}
    zero_rows = _rows(0, p_c, scale=0.0)

    def self_of(rows):
        return {ap.head_key(layer, h): rows[str(layer)][ap.head_key(layer, h)][p_c] for layer in atp.HEAD_LAYERS for h in range(8)}

    def measured(frames, c_scale=1.0):
        return {f"{w}|{fid}": _measured(w, fid, t, p_c, true_rows[(w, fid)], c_scale * c_values[w], self_of(true_rows[(w, fid)])) for fid, t in frames for w in words}

    def tables(rows_hat, c_hat, rows_axis, c_axis, rows_diag):
        make = lambda fid, t, w: _table_row(w, fid, t, p_c, rows_hat(w, fid), c_hat(w), rows_axis(w, fid), c_axis(w), rows_diag(w, fid), 0.0)  # noqa: E731
        locked = {"predictions": {"rows": [make(fid, t, w) for fid, t in exposed for w in words]}}
        stage1 = {"rows": [make(fid, t, w) for fid, t in fresh for w in words], "frames": {fid: {"template_id": t, "valid": True, "template_defined": True} for fid, t in fresh}}
        return locked, stage1

    perfect = lambda w, fid: true_rows[(w, fid)]  # noqa: E731
    nothing = lambda w, fid: zero_rows  # noqa: E731
    # A perfect Level 0 with a frozen-pattern alternative (no change predicted): Y1/Y2 pass, the axis alternative is rejected, the interaction is beyond the self logit (the comparator predicts no change).
    lock, stage1 = tables(perfect, lambda w: c_values[w], nothing, lambda w: 0.0, nothing)
    results = atp.score_confirmation(stage1, measured(exposed), measured(fresh), confirmation, lock)
    t = results["Y1"]["test"]
    assert t["passed"] and t["r2"] == pytest.approx(1.0) and t["entry_r2_layer1"] == pytest.approx(1.0) and t["entry_r2_layer2"] == pytest.approx(1.0) and results["Y2"]["test"]["passed"]
    assert results["Y3"]["evaluable"] and results["Y3"]["rejected"] and results["Y3"]["entry_r2"] == pytest.approx(0.0, abs=1e-9) and results["Y3"]["r2"] < 0
    assert results["comparator"]["interaction_beyond_self_logit"] and results["comparator"]["margin"]["1"] == pytest.approx(1.0) and results["comparator"]["n_pairs"] == 18 * 12
    assert results["outcome"]["label"] == "PATTERN_CHANGE_PREDICTED_TOKENS | PATTERN_CHANGE_PREDICTED_FRAMES_CONDITIONAL | AXIS_ONLY_REJECTED"
    assert len(results["Y2"]["per_frame"]) == 6 and results["Y2"]["per_frame"]["c1"]["level0_layer1"]["entry_r2"] == pytest.approx(1.0) and set(results["Y1"]["descriptive"]["ladder"]["token_means"]) == set(atp.LADDER_KEYS)
    # The comparator equal to Level 0: no margin → the narrower wording.
    lock_d, stage1_d = tables(perfect, lambda w: c_values[w], nothing, lambda w: 0.0, perfect)
    assert not atp.score_confirmation(stage1_d, measured(exposed), measured(fresh), confirmation, lock_d)["comparator"]["interaction_beyond_self_logit"]
    # An axis alternative equal to the truth is not rejected.
    lock_a, stage1_a = tables(perfect, lambda w: c_values[w], perfect, lambda w: c_values[w], nothing)
    assert atp.score_confirmation(stage1_a, measured(exposed), measured(fresh), confirmation, lock_a)["outcome"]["Y3"] == "AXIS_ONLY_NOT_REJECTED"
    # Compressed rows (0.2 × the truth) keep the read but fail both entry floors (R² = 1 − 0.8² = 0.36).
    compressed = lambda w, fid: _rows(hash((w, fid)) % 10_000, p_c, scale=0.2)  # noqa: E731
    lock_c, stage1_c = tables(compressed, lambda w: c_values[w], nothing, lambda w: 0.0, nothing)
    verdict = atp.score_confirmation(stage1_c, measured(exposed), measured(fresh), confirmation, lock_c)
    assert verdict["Y1"]["test"]["failing"] == ["entry_r2_layer1", "entry_r2_layer2"] and verdict["Y1"]["test"]["entry_r2_layer1"] == pytest.approx(0.36) and verdict["outcome"]["Y1"] == "PATTERN_CHANGE_NOT_PREDICTED_TOKENS"
    # A reversed read ordering fails Spearman and R² while the rows are exact.
    lock_r, stage1_r = tables(perfect, lambda w: -c_values[w], nothing, lambda w: 0.0, nothing)
    failing = atp.score_confirmation(stage1_r, measured(exposed), measured(fresh), confirmation, lock_r)["Y1"]["test"]["failing"]
    assert "spearman" in failing and "r2" in failing and "entry_r2_layer1" not in failing
    # Y2 precondition: three valid fresh frames → PRECONDITION_FAILED_FRAMES while Y1 is unaffected.
    three = {"rows": stage1["rows"], "frames": {fid: {"template_id": t, "valid": fid in ("c1", "q1", "a1"), "template_defined": True} for fid, t in fresh}}
    partial = {k: v for k, v in measured(fresh).items() if k.split("|")[1] in ("c1", "q1", "a1")}
    verdict = atp.score_confirmation(three, measured(exposed), partial, confirmation, lock)
    assert verdict["outcome"]["Y2"] == "PRECONDITION_FAILED_FRAMES" and verdict["outcome"]["Y1"] == "PATTERN_CHANGE_PREDICTED_TOKENS"
    # Identical frame sets: a token measured in two exposed frames is not scored for Y1.
    two = {k: v for k, v in measured(exposed).items() if not (k.startswith("w0|") and k.split("|")[1] not in ("e0", "e1"))}
    scored = atp.score_confirmation(stage1, two, measured(fresh), confirmation, lock)
    assert not scored["Y1"]["tokens"]["w0"]["scored"] and len(scored["Y1"]["scored_tokens"]) == 17
    # Too few scored tokens: Y3 non-evaluable.
    few = {k: v for k, v in measured(exposed).items() if k.startswith("w0|")}
    few_fresh = {k: v for k, v in measured(fresh).items() if k.startswith("w0|")}
    assert atp.score_confirmation(stage1, few, few_fresh, confirmation, lock)["outcome"]["Y3"] == "AXIS_ONLY_NOT_EVALUABLE"


def test_lock_reproduction_refusal():
    rows_hat = _rows(1, 3)
    row = _table_row("w", "f", "cardinal", 3, rows_hat, 0.02, _rows(2, 3), 0.0, _rows(3, 3), 0.01)
    lock = {"predictions": {"rows": [row]}}
    assert atp.assert_lock_predictions_reproduced(lock, {"rows": [json.loads(json.dumps(row))]}) == 0.0
    perturbed = json.loads(json.dumps(row))
    perturbed["rows_level0"]["2"]["L02.H06"][1] += 1e-6
    with pytest.raises(atp.PhaseError, match="nothing was executed"):
        atp.assert_lock_predictions_reproduced(lock, {"rows": [perturbed]})
    shorter = json.loads(json.dumps(row))
    shorter["rows_axis"]["1"]["L01.H00"].pop()
    with pytest.raises(atp.PhaseError, match="different length"):
        atp.assert_lock_predictions_reproduced(lock, {"rows": [shorter]})
    moved = json.loads(json.dumps(row))
    moved["p_c"] = 4
    with pytest.raises(atp.PhaseError, match="ordered differently"):
        atp.assert_lock_predictions_reproduced(lock, {"rows": [moved]})
    with pytest.raises(atp.PhaseError, match="number of rows"):
        atp.assert_lock_predictions_reproduced(lock, {"rows": []})


def test_render_predictions_lists_the_token_means_and_self_weights():
    rows_hat = _rows(1, 3)
    row = _table_row("w", "f", "cardinal", 3, rows_hat, 0.02, _rows(2, 3), 0.0, _rows(3, 3), 0.01)
    lock = {"run_id": "r", "protocol_code_commit": "c" * 40, "confirmation_015_sha256": "s" * 64, "lock_014_sha256": "l" * 64, "program": {"1": {"d_head": 64, "rotary_dim": 16, "rotary_base": 10000.0}, "2": {"d_head": 64, "rotary_dim": 16, "rotary_base": 10000.0}},
            "tokens": [{"word": "w", "category": "adjective"}], "predictions": {"rows": [row], "token_means": atp.token_means_from_table([row], ["w"])}}
    text = atp.render_predictions(lock)
    assert "preregistered predictions" in text and "| w | adjective | 1 | **0.0200**" in text and "L02.H07" in text and "rotary_dim 16" in text


@pytest.mark.pythia_smoke
def test_program_reproduces_the_pinned_model_rows_on_a_neutral_prompt():
    """I1 on the pinned Pythia-70M with the rotary path the fake cannot exercise (cfg.rotary_dim is None on the bridge; the HuggingFace rope parameters give 16 of 64)."""
    import os

    from neural_decompiler.models import PYTHIA_70M, load_model

    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")
    model = load_model(PYTHIA_70M)
    programs = atp.programs_from_model(model)
    assert programs[1].rotary_dim == 16 and programs[1].d_head == 64 and programs[1].rotary_base == 10000.0 and programs[2].n_heads == 8
    ids = tuple(int(i) for i in model.to_tokens("The quick brown fox jumps over", prepend_bos=False, truncate=False)[0])  # a neutral prompt, no experiment frame or cue
    p_c = len(ids) - 1
    frame = pm.Frame("cardinal", "smoke-1", ids[:-1], (), {"sg": ids[-1], "pl": ids[-1]}, "smoke {cue}", origin="extension")  # the template label only shapes the Frame object; the text is not an experiment frame
    prompt = pm.Prompt(frame, ids[-1], "smoke")
    sites = [(f"RESID_PRE.L{layer}", k) for layer in atp.HEAD_LAYERS for k in range(p_c + 1)] + [(f"ATTN_PATTERN.L{layer}", p_c) for layer in atp.HEAD_LAYERS]
    run = pm.capture_prompt(model, prompt, sites)
    for layer in atp.HEAD_LAYERS:
        residuals = [run.vector((f"RESID_PRE.L{layer}", k)).double() for k in range(p_c + 1)]
        captured = run.vector((f"ATTN_PATTERN.L{layer}", p_c))[:, : p_c + 1].double()
        row = atp.ReferenceRow(programs[layer], residuals)
        assert float((row.A_ref - captured).abs().max()) < atp.ROW_IDENTITY_TOLERANCE
        unrotated = atp.ReferenceRow(dataclasses.replace(programs[layer], rotary_dim=0), residuals)
        assert float((unrotated.A_ref - captured).abs().max()) > 0.01  # the rotation is not optional: without it the rows are wrong
