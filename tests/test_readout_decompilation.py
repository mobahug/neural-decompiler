"""Experiment 020 unit tests: the frozen constants, the norm-normalized Level 1 error, the noun populations and the
scorable rule, the readout program and its identities on the fake, the provenance boundary, the scoring branches on
synthetic tables, and the confirmation set (24 cues, 18 frames, 24 nouns, the three manifest classes)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_decompiler import attention_paths as ap
from neural_decompiler import attention_patterns as atp
from neural_decompiler import block_concentration as bc
from neural_decompiler import block_routing as br
from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import encoding_read as er
from neural_decompiler import frame_channels as fch
from neural_decompiler import head_pattern as hp
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import neuron_feature as nf
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_decompilation as rd
from neural_decompiler.candidate_screening import Split
from plural_fakes import TinyPlural
from test_block_routing import toy_tokenizer_019

ROOT = Path(__file__).parents[1]


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


# ---------------------------------------------------------------------------
# Frozen constants.


def test_frozen_constants_match_the_design():
    assert rd.READOUT_LAYERS == (3, 4, 5) and rd.HEAD_LAYER == 3 and rd.RUNTIME_SEED == 20260916 and rd.CONTROL_SEED == 20260924
    assert (rd.READOUT_IDENTITY_TOLERANCE, rd.LOGIT_IDENTITY_TOLERANCE, rd.ADDITIVE_IDENTITY_TOLERANCE) == (2e-2, 2e-2, 1e-4)
    assert (rd.LEVEL1_TOLERANCE, rd.INHERITED_017_TOLERANCE, rd.PREDICTION_REPRODUCTION_TOLERANCE, rd.PROVENANCE_TOLERANCE, rd.NORM_FLOOR) == (1e-3, 1e-6, 0.0, 1e-6, 1e-12)
    assert (rd.Y1_TOKEN_MEAN_R2, rd.Y1_PAIR_MEAN_R2, rd.Y1_CUE_MAE, rd.Y1_CUE_MAE_SHARE, rd.Y1_POOLED_MAE) == (0.80, 0.65, 1.5, 0.80, 1.0)
    assert (rd.Y2_FRAME_MEAN_R2, rd.Y2_FRAME_R2, rd.Y2_FRAME_SHARE, rd.Y2_SPLIT_R2) == (0.55, 0.40, 0.75, 0.40)
    assert (rd.Y3_MEDIAN_R2, rd.Y3_NOUN_R2, rd.Y3_SLOPE_BAND, rd.Y3_BIAS, rd.Y3_SHARE) == (0.70, 0.55, (0.75, 1.15), 0.8, 0.90)
    assert (rd.MIN_SCORED_TOKENS, rd.MIN_VALID_EXPOSED_FRAMES, rd.MIN_VALID_FRESH_FRAMES, rd.MIN_VALID_COORDINATED_FRAMES, rd.MIN_SCORABLE_FRESH_NOUNS) == (16, 60, 12, 4, 18)
    assert sum(rd.QUOTAS.values()) == 24 and len(rd.FRESH_FRAMES) == 18 and rd.NOUN_QUOTA == 8 and len(rd.NOUN_CANDIDATES) == 3
    assert rd.MANIFEST_CLASSES == ("S1-REF", "S1-VALIDITY", "S2-TARGET") and rd.PHASES == ("validate", "freeze-confirmation", "explore", "lock", "confirm", "report")
    assert rd.OUTCOME_Y3[1] == "NOUN_READOUT_NOT_ESTABLISHED"  # the negative never asserts a noun-specific mechanism
    assert set(rd.COMPARATOR_STANDING) == {"dT_only", "template_base_mlps", "no_l5_heads", "rank1_nouns"}
    assert rd.COMPARATOR_STANDING["no_l5_heads"] == "nested simplification comparator" and rd.COMPARATOR_STANDING["rank1_nouns"] == "descriptive comparator"
    assert rd.POPULATIONS["Y1"]["nouns"] == rd.POPULATIONS["Y2"]["nouns"] == "exposed_scorable" and rd.POPULATIONS["Y3"]["nouns"] == "fresh"
    assert [count for _, count in sorted({t: sum(1 for x, _ in rd.FRESH_FRAMES if x == t) for t, _ in rd.FRESH_FRAMES}.items())] == [6, 6, 6]


# ---------------------------------------------------------------------------
# The Level 1 error: norm-normalized, never componentwise.


def test_level1_error_is_norm_normalized_and_cannot_explode_near_zero():
    measured = torch.tensor([1.0, -2.0, 1e-18])
    assert rd.level1_error(measured, measured) == 0.0
    off = measured + torch.tensor([0.0, 0.0, 1e-3])
    assert rd.level1_error(off, measured) == pytest.approx(1e-3 / 2.0)  # normalized by ‖measured‖∞ = 2, not by the 1e-18 component
    componentwise = float(((off - measured).abs() / measured.abs()).max())
    assert componentwise > 1e14 and rd.level1_error(off, measured) < rd.LEVEL1_TOLERANCE  # the rejected definition would stop the experiment here
    zero = torch.zeros(3)
    assert rd.level1_error(torch.tensor([1e-15, 0.0, 0.0]), zero) == pytest.approx(1e-3)  # the floor applies; no division by zero
    assert rd.relative_error(off, measured) == rd.level1_error(off, measured)


def test_enforce_raises_above_the_tolerance_and_records_below_it():
    assert rd.enforce("check", 9e-4, rd.LEVEL1_TOLERANCE) == pytest.approx(9e-4)
    with pytest.raises(pm.IncidentError, match="frozen tolerance"):
        rd.enforce("check", 2e-3, rd.LEVEL1_TOLERANCE)
    with pytest.raises(pm.IncidentError):
        rd.enforce("check", float("nan"), rd.LEVEL1_TOLERANCE)


# ---------------------------------------------------------------------------
# Nouns: the scorable rule and the two populations.


def _noun(key, rule="simple-suffix", sg=(11,), pl=(12,), split=Split.HOLDOUT):
    return pm.Noun(key, split, rule, sg, pl)


def test_noun_set_drops_the_non_single_token_noun_and_keeps_the_populations_disjoint():
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    exposed = [_noun("brickish"), _noun("peachlike", sg=(13, 14), pl=(13, 15)), _noun("boxen", rule="sibilant-es", sg=(16,), pl=(17,))]
    fresh = [_noun("colonial", rule="consonant-y", sg=(18,), pl=(19,), split=Split.FUTURE_RESERVE)]
    nouns = rd.NounSet.build(weights, exposed, fresh)
    assert nouns.non_scorable == ("peachlike",) and len(nouns.exposed_scorable) == 2 and nouns.fresh == (3,)
    assert not set(nouns.exposed_scorable) & set(nouns.fresh)
    assert torch.equal(nouns.dw[1], torch.zeros(weights.W_U.shape[0], dtype=torch.float64))  # a shared first token gives the zero read
    expected = weights.W_U.double()[:, 11] - weights.W_U.double()[:, 12]
    assert torch.allclose(nouns.dw[0], expected)
    delta = torch.randn(weights.W_U.shape[0], dtype=torch.float64)
    assert nouns.read(delta)[0] == pytest.approx(float(expected @ delta))
    assert nouns.population("exposed_scorable") == nouns.exposed_scorable and nouns.population("fresh") == nouns.fresh
    with pytest.raises(ValueError):
        nouns.population("everything")


def test_contrasts_use_the_single_token_forms_only():
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    nouns = rd.NounSet.build(weights, [_noun("a", sg=(11,), pl=(12,)), _noun("multi", sg=(13, 14), pl=(13, 15))], [])
    logits = torch.randn(int(model.cfg.d_vocab))
    values = nouns.contrasts(logits)
    log_probs = logits.double().log_softmax(dim=-1)
    assert values[0] == pytest.approx(float(log_probs[11] - log_probs[12])) and values[1] == 0.0


# ---------------------------------------------------------------------------
# The readout program on the fake: the identities and the provenance boundary.


@pytest.fixture(scope="module")
def fake_frame():
    """One frame of the manifest on the fake: its reference state, a cue pair's measured quantities and the program."""
    model = make_fake_model()
    manifest, manifest_sha256, extension = pm.load_inputs(ROOT)
    c006 = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    pool = cd.exposed_pool(manifest, extension)
    frame = pool.frames[0]
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    nouns = rd.NounSet.build(weights, pool.nouns[:6], [])
    axis_T = pm.SiteAxis("T", torch.zeros(int(model.cfg.d_model), dtype=torch.float64), torch.nn.functional.normalize(torch.arange(1.0, int(model.cfg.d_model) + 1, dtype=torch.float64), dim=0), 1.0)
    reference = pm.Prompt(frame, frame.cue_ids["sg"], "ref")
    state = rd.capture_frame_020(model, head, reference, nouns, axis_T)
    program = rd.ReadoutProgram.from_model(model)
    cue = pm.Prompt(frame, frame.cue_ids["pl"], "pl")
    sites = [("RESID_PRE.L3", state.p_c), ("RESID_PRE.L3", state.p_t), ("RESID_POST.L5", state.p_t)]
    sites += [(f"L0{layer}.MLP", state.p_t) for layer in (3, 4, 5)] + [(f"L0{layer}.H0{h}", state.p_t) for layer in (3, 4, 5) for h in range(int(model.cfg.n_heads))]
    run = pm.capture_prompt(model, cue, sites)
    positions = sorted({state.p_c, state.p_t})
    dx3 = {position: run.vector(("RESID_PRE.L3", position)).double() - state.x3_all[position].double() for position in positions}
    return {"model": model, "weights": weights, "program": program, "state": state, "nouns": nouns, "run": run, "dx3": dx3, "positions": positions}


def test_identities_hold_on_the_fake(fake_frame):
    program, state, nouns, run = fake_frame["program"], fake_frame["state"], fake_frame["nouns"], fake_frame["run"]
    weights, model = fake_frame["weights"], fake_frame["model"]
    h6 = run.vector(("RESID_POST.L5", state.p_t)).double()
    assert rd.logit_identity(program, weights, h6, run.logits) < rd.LOGIT_IDENTITY_TOLERANCE
    assert rd.readout_identity(nouns, program, state, h6, nouns.contrasts(run.logits)) < rd.READOUT_IDENTITY_TOLERANCE
    parts = []
    for layer in (3, 4, 5):
        parts.append(run.vector((f"L0{layer}.MLP", state.p_t)).double() - reference_component(state, model, f"L0{layer}.MLP"))
        for head_index in range(int(model.cfg.n_heads)):
            key = f"L0{layer}.H0{head_index}"
            parts.append(run.vector((key, state.p_t)).double() - reference_component(state, model, key))
    assert rd.additive_identity(h6 - state.h6, fake_frame["dx3"][state.p_t], parts) < rd.ADDITIVE_IDENTITY_TOLERANCE
    e1 = rd.level1_error(program.level1(state, fake_frame["dx3"]), h6 - state.h6)
    assert e1 < rd.LEVEL1_TOLERANCE  # the exact chain reproduces the measured final residual change on the fake too
    assert rd.enforce("level1", e1, rd.LEVEL1_TOLERANCE) == pytest.approx(e1)
    # the parallel residual is what makes it exact: feeding block 5's MLP the post-attention state instead breaks it
    dh4 = program.blocks_3_to_5(state, fake_frame["dx3"])
    sequential = dh4["dh4"][state.p_t] + dh4["parts"]["block5_attention"] + program.mlp_delta(5, state.x5_all[state.p_t], dh4["dh4"][state.p_t] + dh4["parts"]["block5_attention"])
    assert rd.level1_error(sequential, h6 - state.h6) > e1


_REFERENCE_COMPONENTS: dict[tuple[int, str], torch.Tensor] = {}


def reference_component(state, model, key):
    """The reference run's component output at p_t, captured once per (frame, key)."""
    cache_key = (id(state), key)
    if cache_key not in _REFERENCE_COMPONENTS:
        reference = pm.Prompt(state.frame, state.state_017.state.ref.frame.cue_ids["sg"], "ref")
        run = pm.capture_prompt(model, reference, [(key, state.p_t)])
        _REFERENCE_COMPONENTS[cache_key] = run.vector((key, state.p_t)).double()
    return _REFERENCE_COMPONENTS[cache_key]


def test_the_readout_program_reads_only_the_reference_state_and_the_predicted_change(fake_frame):
    """Provenance: with every capture and intervention entry point disabled, the program still produces its prediction."""
    program, state, nouns, dx3 = fake_frame["program"], fake_frame["state"], fake_frame["nouns"], fake_frame["dx3"]
    with pytest.MonkeyPatch.context() as guard:
        for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, name):
                guard.setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the readout program touched the network")))
        out = program.blocks_3_to_5(state, dx3)
        predicted = program.contrast(state, out["dh6"], nouns)
    again = program.contrast(state, program.blocks_3_to_5(state, dx3)["dh6"], nouns)
    assert rd.provenance_difference(predicted, again) <= rd.PROVENANCE_TOLERANCE
    assert set(out["parts"]) == {"block3_attention", "block3_mlp", "block4_mlp", "block5_attention", "block5_mlp"} and out["dh6"].shape == state.h6.shape


def test_the_comparators_change_the_prediction_in_the_declared_direction(fake_frame):
    program, state, nouns, dx3 = fake_frame["program"], fake_frame["state"], fake_frame["nouns"], fake_frame["dx3"]
    full = program.contrast(state, program.blocks_3_to_5(state, dx3)["dh6"], nouns)
    no_l5 = program.contrast(state, program.blocks_3_to_5(state, dx3, l5_heads=False)["dh6"], nouns)
    bases = {4: state.x4_all[state.p_t] * 0.5, 5: state.x5_all[state.p_t] * 0.5}
    at_base = program.contrast(state, program.blocks_3_to_5(state, dx3, mlps_at_base=bases)["dh6"], nouns)
    assert not torch.allclose(full, no_l5) and not torch.allclose(full, at_base)  # both comparators are real reductions
    assert torch.allclose(full, program.contrast(state, program.blocks_3_to_5(state, dx3, l5_heads=True)["dh6"], nouns))


# ---------------------------------------------------------------------------
# Scoring on synthetic tables.


def _table(n_cues=24, n_frames=12, n_nouns=20, *, noise=0.0, cue_noise=None, frame_noise=None, seed=0, templates=("cardinal", "quantifier", "coordinated-adjective")):
    generator = torch.Generator().manual_seed(seed)
    cues, frames, tpls, rows, predicted = [], [], [], [], []
    base = torch.randn(n_nouns, generator=generator, dtype=torch.float64).abs() + 0.5
    for frame_index in range(n_frames):
        template = templates[frame_index % len(templates)]
        for cue_index in range(n_cues):
            score = -4.0 + 0.4 * ((cue_index % 7) - 3) + 0.2 * ((frame_index % 5) - 2)
            measured = score * base
            error = noise
            if cue_noise is not None and cue_index in cue_noise:
                error = cue_noise[cue_index]
            if frame_noise is not None and frame_index in frame_noise:
                error = frame_noise[frame_index]
            perturbation = torch.randn(n_nouns, generator=generator, dtype=torch.float64) * error
            cues.append(f"cue{cue_index}")
            frames.append(f"{template}-{frame_index}")
            tpls.append(template)
            rows.append(measured)
            predicted.append(measured + perturbation)
    return rd.ScoringTable(tuple(cues), tuple(frames), tuple(tpls), torch.stack(rows), torch.stack(predicted), tuple(f"n{i}" for i in range(n_nouns)))


def test_pair_statistics_aggregate_by_cue_frame_and_template():
    table = _table(noise=0.05)
    statistics = rd.pair_statistics(table)
    assert statistics["n_pairs"] == 24 * 12 and statistics["n_nouns"] == 20
    assert statistics["flattened_r2"] > 0.99 and statistics["pooled_mae"] < 0.1
    assert set(statistics["per_cue"]) == {f"cue{i}" for i in range(24)} and len(statistics["per_frame"]) == 12 and set(statistics["per_template"]) == {"cardinal", "quantifier", "coordinated-adjective"}
    assert statistics["cue_final_r2"] is not None and statistics["coordinated_r2"] is not None
    assert statistics["token_mean_r2"] is not None and statistics["frame_mean_r2"] is not None


def test_y1_branches_on_synthetic_tables():
    good = rd.score_y1(_table(noise=0.05), n_valid_frames=90)
    assert good["label"] == "CONTRAST_PREDICTED_TOKENS" and all(good["conditions"].values()) and good["population"] == rd.POPULATIONS["Y1"]
    # a few cues far off: the MAE share guard fails while the pooled statistics stay high
    bad_cues = rd.score_y1(_table(noise=0.05, cue_noise={i: 6.0 for i in range(8)}), n_valid_frames=90)
    assert bad_cues["label"] == "CONTRAST_NOT_PREDICTED_TOKENS" and not bad_cues["conditions"]["cue_mae_share"]
    # everything off: the token-mean and pooled-MAE conditions fail too
    noisy = rd.score_y1(_table(noise=4.0, seed=3), n_valid_frames=90)
    assert noisy["label"] == "CONTRAST_NOT_PREDICTED_TOKENS" and not noisy["conditions"]["pooled_mae"]
    few_frames = rd.score_y1(_table(noise=0.05), n_valid_frames=10)
    assert few_frames["label"] == "PRECONDITION_FAILED_TOKENS" and few_frames["precondition"]["n_valid_frames"] == 10
    few_cues = rd.score_y1(_table(n_cues=8, noise=0.05), n_valid_frames=90)
    assert few_cues["label"] == "PRECONDITION_FAILED_TOKENS" and not few_cues["precondition"]["scored_tokens"]


def test_y2_branches_on_synthetic_tables():
    frames = [f"{t}-{i}" for i, t in enumerate(("cardinal", "quantifier", "coordinated-adjective") * 6)]
    good = rd.score_y2(_table(n_frames=18, noise=0.05), valid_frames=frames, valid_coordinated=6)
    assert good["label"] == "CONTRAST_PREDICTED_FRAMES_CONDITIONAL" and all(good["conditions"].values())
    # a third of the frames badly predicted: the share guard fails
    bad = rd.score_y2(_table(n_frames=18, noise=0.05, frame_noise={i: 12.0 for i in range(6)}), valid_frames=frames, valid_coordinated=6)
    assert bad["label"] == "CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL" and not bad["conditions"]["frame_share"]
    # the coordinated family alone off: its split guard fails
    coordinated = rd.score_y2(_table(n_frames=18, noise=0.05, frame_noise={i: 12.0 for i in range(18) if i % 3 == 2}), valid_frames=frames, valid_coordinated=6)
    assert not coordinated["conditions"]["coordinated_split"] and coordinated["label"] == "CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL"
    too_few = rd.score_y2(_table(n_frames=18, noise=0.05), valid_frames=frames[:6], valid_coordinated=2)
    assert too_few["label"] == "PRECONDITION_FAILED_FRAMES" and not too_few["precondition"]["valid_fresh_frames"]
    no_coordinated = rd.score_y2(_table(n_frames=18, noise=0.05), valid_frames=frames, valid_coordinated=2)
    assert no_coordinated["label"] == "PRECONDITION_FAILED_FRAMES" and not no_coordinated["precondition"]["valid_coordinated"]


def test_y3_branches_and_the_negative_label_on_synthetic_tables():
    table = _table(n_nouns=24, noise=0.05)
    good = rd.score_y3(table)
    assert good["label"] == "NOUN_READOUT_FIXED" and good["median_r2"] > 0.9 and good["population"]["nouns"] == "fresh"
    # one pathological noun does not decide the outcome (no minimum gate), five of 24 do
    one_bad = rd.ScoringTable(table.cues, table.frames, table.templates, table.measured, table.predicted.clone(), table.noun_keys)
    one_bad.predicted[:, 0] = one_bad.measured[:, 0] * 0.1
    assert rd.score_y3(one_bad)["label"] == "NOUN_READOUT_FIXED"
    many_bad = rd.ScoringTable(table.cues, table.frames, table.templates, table.measured, table.predicted.clone(), table.noun_keys)
    for index in range(5):
        many_bad.predicted[:, index] = many_bad.measured[:, index] * 0.1
    result = rd.score_y3(many_bad)
    assert result["label"] == "NOUN_READOUT_NOT_ESTABLISHED" and not result["conditions"]["slope_share"]
    biased = rd.ScoringTable(table.cues, table.frames, table.templates, table.measured, table.predicted + 2.0, table.noun_keys)
    assert rd.score_y3(biased)["label"] == "NOUN_READOUT_NOT_ESTABLISHED" and not rd.score_y3(biased)["conditions"]["bias_share"]
    degenerate = rd.ScoringTable(table.cues, table.frames, table.templates, torch.zeros_like(table.measured), table.predicted, table.noun_keys)
    assert rd.score_y3(degenerate)["label"] == "PRECONDITION_FAILED_NOUNS" and rd.score_y3(degenerate)["precondition"]["n_scorable_nouns"] == 0


def test_every_frozen_identity_has_a_tolerance_and_an_unknown_one_is_a_defect():
    tolerances = rd.identity_tolerances()
    assert set(tolerances) == set(rd.IDENTITY_NAMES)
    assert rd.enforce_all({"readout": 1e-3, "level1": 1e-4, "additive": 1e-6, "logit": 1e-3, "inherited_017": 1e-8, "lock_rows": 0.0, "stage1_rows": 0.0, "provenance": 0.0,
                           "reference_contrast": 1e-3}) is not None
    with pytest.raises(pm.IncidentError, match="no frozen tolerance"):
        rd.enforce_all({"invented": 0.0})
    with pytest.raises(pm.IncidentError, match="frozen tolerance"):
        rd.enforce_all({"level1": 1.0})


def test_the_rank1_comparator_is_the_frozen_rule_not_the_primary_program():
    """Two frozen objects: the exposed noun vector (which defines the pair score) and d_noun (which gives a fresh
    noun's factor from its weights alone). The comparator must differ from the primary prediction."""
    generator = torch.Generator().manual_seed(11)
    n_pairs, n_exposed, n_fresh, d_model = 40, 12, 5, 16
    dw_exposed = torch.randn(n_exposed, d_model, generator=generator, dtype=torch.float64)
    dw_fresh = torch.randn(n_fresh, d_model, generator=generator, dtype=torch.float64)
    delta = torch.randn(n_pairs, d_model, generator=generator, dtype=torch.float64)
    measured_exposed = delta @ dw_exposed.T
    table = rd.ScoringTable(tuple(f"c{i}" for i in range(n_pairs)), tuple("f" for _ in range(n_pairs)), tuple("cardinal" for _ in range(n_pairs)),
                            measured_exposed, measured_exposed, tuple(f"n{i}" for i in range(n_exposed)))
    frozen = rd.rank1_noun_factor(table, dw_exposed)
    assert len(frozen["noun_vector"]) == n_exposed and len(frozen["d_noun"]) == d_model and frozen["exposed_fit_r2"] is not None
    fresh_table = rd.ScoringTable(table.cues, table.frames, table.templates, delta @ dw_fresh.T, delta @ dw_fresh.T, tuple(f"x{i}" for i in range(n_fresh)))
    entry = {"exposed": table, "fresh": fresh_table}
    value = rd.rank1_fresh_r2(entry, {"rank1": frozen}, dw_fresh)
    assert value is not None and value < 1.0  # a rank-1 factorization, not the primary prediction it is compared against
    assert rd.rank1_fresh_r2(entry, {"rank1": {}}, dw_fresh) is None and rd.rank1_fresh_r2(entry, {"rank1": frozen}, None) is None


def test_the_frozen_validity_rule_uses_the_head_change_and_the_cue_effect_count(fake_frame, monkeypatch):
    """The inherited 017–019 rule: |⟨ΔT, d̂_T⟩| ≥ floor · σ_T together with the frozen cue-effect count."""
    program, state, nouns, dx3 = fake_frame["program"], fake_frame["state"], fake_frame["nouns"], fake_frame["dx3"]
    axis = pm.SiteAxis("T", torch.zeros(state.h6.shape[0], dtype=torch.float64), torch.nn.functional.normalize(torch.arange(1.0, state.h6.shape[0] + 1, dtype=torch.float64), dim=0), 1.0)
    measurement = rd.PairMeasurement("pl", 1, state.frame.frame_id, state.frame.template_id, torch.full((len(nouns.nouns),), -1.0, dtype=torch.float64), dx3,
                                     torch.zeros_like(state.h6), state.h6, torch.zeros(3), {})
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    verdict = rd.frame_validity(program, state, nouns, measurement, axis)
    assert verdict["head_informative"] and verdict["cue_effect_positive"] == len(nouns.exposed_scorable) and verdict["valid"]
    assert verdict["plural_head_change"] == pytest.approx(float(rd.head_output_change(program, state, dx3) @ axis.direction))
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 1e9)  # an unreachable floor makes the frame uninformative
    assert not rd.frame_validity(program, state, nouns, measurement, axis)["valid"]
    monkeypatch.setattr(cs, "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    wrong_sign = rd.PairMeasurement("pl", 1, state.frame.frame_id, state.frame.template_id, torch.full((len(nouns.nouns),), +1.0, dtype=torch.float64), dx3,
                                    torch.zeros_like(state.h6), state.h6, torch.zeros(3), {})
    assert rd.frame_validity(program, state, nouns, wrong_sign, axis)["cue_effect_positive"] == 0  # c_sg − c_pl = −Δc


def test_outcome_label_joins_the_three_frozen_labels():
    y1 = {"label": "CONTRAST_PREDICTED_TOKENS"}
    y2 = {"label": "CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL"}
    y3 = {"label": "NOUN_READOUT_FIXED"}
    assert rd.outcome_label(y1, y2, y3)["label"] == "CONTRAST_PREDICTED_TOKENS | CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL | NOUN_READOUT_FIXED"
    with pytest.raises(ValueError):
        rd.outcome_label({"label": "SOMETHING_ELSE"}, y2, y3)


# ---------------------------------------------------------------------------
# The exposed pool and the confirmation set.


@pytest.fixture(scope="module")
def inputs():
    manifest_obj, manifest_sha256, extension = pm.load_inputs(ROOT)
    c006 = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest_obj, manifest_sha256, extension)
    c009 = ht.load_confirmation(ROOT / ht.CONFIRMATION_RELATIVE_PATH, manifest_obj, manifest_sha256, extension, c006)
    c011 = er.load_confirmation(ROOT / er.CONFIRMATION_RELATIVE_PATH, manifest_obj, manifest_sha256, extension, c006, c009)
    digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": c006.content_sha256, "confirmation_009": c009.content_sha256, "confirmation_011": c011.content_sha256}
    pool_012 = lc.build_pool_012(manifest_obj, extension, c006, c009, c011)
    c012 = lc.load_confirmation(ROOT / lc.CONFIRMATION_RELATIVE_PATH, pool_012, digests)
    digests["confirmation_012"] = c012.content_sha256
    digests["lock_012"] = json.loads((ROOT / lc.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    c013 = ap.load_confirmation(ROOT / ap.CONFIRMATION_RELATIVE_PATH, ap.build_pool_013(manifest_obj, extension, c006, c009, c011, c012), digests)
    digests["confirmation_013"] = c013.content_sha256
    digests["lock_013"] = json.loads((ROOT / ap.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    c014 = nf.load_confirmation(ROOT / nf.CONFIRMATION_RELATIVE_PATH, nf.build_pool_014(manifest_obj, extension, c006, c009, c011, c012, c013), digests)
    digests["confirmation_014"] = c014.content_sha256
    digests["lock_014"] = json.loads((ROOT / nf.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    c015 = atp.load_confirmation(ROOT / atp.CONFIRMATION_RELATIVE_PATH, atp.build_pool_015(manifest_obj, extension, c006, c009, c011, c012, c013, c014), digests)
    digests["confirmation_015"] = c015.content_sha256
    digests["lock_015"] = json.loads((ROOT / atp.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    c016 = fch.load_confirmation(ROOT / fch.CONFIRMATION_RELATIVE_PATH, fch.build_pool_016(manifest_obj, extension, c006, c009, c011, c012, c013, c014, c015), digests)
    digests["confirmation_016"] = c016.content_sha256
    c017 = hp.load_confirmation(ROOT / hp.CONFIRMATION_RELATIVE_PATH, hp.build_pool_017(manifest_obj, extension, c006, c009, c011, c012, c013, c014, c015, c016), digests)
    digests["confirmation_017"] = c017.content_sha256
    digests["lock_017"] = json.loads((ROOT / hp.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    pool_018 = bc.build_pool_018(manifest_obj, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017)
    c018 = bc.load_confirmation(ROOT / bc.CONFIRMATION_RELATIVE_PATH, pool_018, digests)
    digests["confirmation_018"] = c018.content_sha256
    digests["lock_018"] = json.loads((ROOT / bc.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    pool_019 = br.build_pool_019(manifest_obj, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017, c018)
    c019 = br.load_confirmation(ROOT / br.CONFIRMATION_RELATIVE_PATH, pool_019, digests)
    digests["confirmation_019"] = c019.content_sha256
    pool = rd.build_pool_020(manifest_obj, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017, c018, c019)
    return manifest_obj, digests, pool, c019


def test_pool_020_is_experiment_019s_pool_plus_its_confirmed_tokens_and_frames(inputs):
    _, _, pool, c019 = inputs
    assert len(pool.tokens) == 279 and len(pool.frames) == 108 and len(pool.nouns) == 80
    assert sum(1 for noun in pool.nouns if noun.single_token) == 79 and [n.lexical_key for n in pool.nouns if not n.single_token] == ["peach"]
    assert sum(1 for frame in pool.frames if pool.frame_origin[frame.frame_id] == "confirmation-019") == 18
    assert {token["word"] for token in c019.tokens} <= {name for name, _ in pool.tokens}
    weights_like = type("W", (), {"W_U": torch.zeros(4, 60000), "b_U": torch.zeros(60000)})()
    nouns = rd.NounSet.build(weights_like, pool.nouns, [])
    assert len(nouns.exposed_scorable) == 79 and nouns.non_scorable == ("peach",)


def toy_tokenizer_020(manifest, pool):
    base = toy_tokenizer_019(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in rd.CANDIDATES.values() for word in words]
    for rule_class, nouns in rd.NOUN_CANDIDATES.items():
        for word in nouns:
            words += [" " + word, " " + rd.plural_form(word, rule_class)]
    for _, text in rd.FRESH_FRAMES:
        for index, piece in enumerate(text.replace("{cue}", "").split()):
            words.append(piece if index == 0 else " " + piece)
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    for word, token_id in pool.tokens:  # exposed words carry their real ids so that the builder skips them
        vocabulary[" " + word] = token_id
    for noun in pool.nouns:  # exposed noun forms too
        vocabulary[" " + noun.lexical_key] = noun.sg_ids[0]
    known = set(vocabulary.values())
    for frame in pool.frames:
        for token_id in (*frame.prefix_ids, *frame.suffix_ids, *frame.cue_ids.values()):
            if token_id not in known:
                vocabulary[f"⟨{token_id}⟩"] = token_id
                known.add(token_id)
    return type(base)(vocabulary)


def test_confirmation_policy_with_24_cues_18_frames_and_24_nouns(inputs, tmp_path):
    manifest, digests, pool, _ = inputs
    tokenizer = toy_tokenizer_020(manifest, pool)
    path = tmp_path / "confirmation-v1.json"
    digest = rd.freeze_confirmation(path, tokenizer, pool, digests)
    confirmation = rd.load_confirmation(path, pool, digests)
    assert confirmation.content_sha256 == digest
    assert len(confirmation.tokens) == 24 and len(confirmation.frames) == 18 and len(confirmation.nouns) == 24
    words = [token["word"] for token in confirmation.tokens]
    assert words[:6] == ["recent", "current", "original", "typical", "ordinary", "identical"]
    assert words[6:12] == ["average", "excessive", "exhaustive", "comprehensive", "thorough", "sweeping"]
    assert words[12:18] == ["anything", "something", "everything", "nothing", "who", "thee"]
    assert words[18:] == ["purple", "yellow", "hidden", "hard", "dirty", "rare"]
    noun_keys = [noun.lexical_key for noun in confirmation.nouns]
    assert noun_keys[:8] == ["brick", "candle", "statue", "barrel", "curtain", "magnet", "puzzle", "pillar"]
    assert noun_keys[8:16] == ["switch", "branch", "ash", "sketch", "batch", "flash", "arch", "crash"]
    assert noun_keys[16:] == ["colony", "gallery", "cavity", "battery", "category", "artery", "boundary", "anomaly"]
    assert "tunnel" not in noun_keys and all(noun.single_token for noun in confirmation.nouns)
    exposed_keys = {noun.lexical_key for noun in pool.nouns}
    exposed_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    assert not (set(noun_keys) & exposed_keys) and not ({token for noun in confirmation.nouns for token in (*noun.sg_ids, *noun.pl_ids)} & exposed_ids)
    assert not ({token["token_id"] for token in confirmation.tokens} & exposed_ids)
    assert all(frame.text_template not in {f.text_template for f in pool.frames} for frame in confirmation.frames)
    assert confirmation.frames[0].frame_id == "cardinal-020-1" and len(confirmation.token_prompts) == 24 * 18 and len(confirmation.exposed_frame_prompts) == 24 * 108
    classes = rd.manifest_classes(confirmation)
    assert len(classes["S1-REF"]) == 18 and len(classes["S1-VALIDITY"]) == 18 and len(classes["S2-TARGET"]) == 24 * 18 + 24 * 108
    assert not (set(classes["S2-TARGET"]) & (set(classes["S1-REF"]) | set(classes["S1-VALIDITY"])))
    assert rd.prompt_key_manifest(confirmation) == sorted(set(classes["S1-REF"]) | set(classes["S1-VALIDITY"]) | set(classes["S2-TARGET"]))
    with pytest.raises(rd.PhaseError):
        rd.freeze_confirmation(path, tokenizer, pool, digests)


def test_confirmation_tampering_is_refused(inputs, tmp_path):
    manifest, digests, pool, _ = inputs
    tokenizer = toy_tokenizer_020(manifest, pool)
    path = tmp_path / "confirmation-v1.json"
    rd.freeze_confirmation(path, tokenizer, pool, digests)
    payload = json.loads(path.read_text())
    for mutate, match in (
        (lambda p: p.update(manifest={**p["manifest"], "S2-TARGET": p["manifest"]["S2-TARGET"][:-1]}), "manifest"),
        (lambda p: p.update(tokens=[dict(t, word="bogus") if t["word"] == "recent" else t for t in p["tokens"]]), "frozen class list"),
        (lambda p: p.update(nouns=[dict(n, lexical_key="tunnel") if n["lexical_key"] == "brick" else n for n in p["nouns"]]), "is an exposed noun"),  # revision 1's error: tunnel is exposed
        (lambda p: p.update(nouns=[dict(n, lexical_key="obelisk") if n["lexical_key"] == "brick" else n for n in p["nouns"]]), "frozen class list"),
        (lambda p: p.update(nouns=p["nouns"][:-1]), "quotas"),
        (lambda p: p.update(frames=p["frames"][:-1]), "frozen literal frames"),
    ):
        tampered = json.loads(json.dumps(payload))
        mutate(tampered)
        tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in tampered.items() if key != "content_sha256"}))
        with pytest.raises(ValueError, match=match):
            rd.validate_confirmation(tampered, pool, digests)


def test_fresh_nouns_may_not_appear_in_an_exploration_record(inputs, tmp_path):
    manifest, digests, pool, _ = inputs
    tokenizer = toy_tokenizer_020(manifest, pool)
    path = tmp_path / "confirmation-v1.json"
    rd.freeze_confirmation(path, tokenizer, pool, digests)
    confirmation = rd.load_confirmation(path, pool, digests)
    rd.assert_fresh_nouns_absent({"exploration": {"pairs": {"recent|cardinal-1": {"c_L": 0.5}}}}, confirmation)
    with pytest.raises(rd.PhaseError, match="names the fresh noun"):
        rd.assert_fresh_nouns_absent({"nouns": {"brick": {"r2": 0.9}}}, confirmation)


def test_the_barrier_refuses_an_executed_target_prompt(inputs, tmp_path):
    manifest, digests, pool, _ = inputs
    tokenizer = toy_tokenizer_020(manifest, pool)
    path = tmp_path / "confirmation-v1.json"
    rd.freeze_confirmation(path, tokenizer, pool, digests)
    confirmation = rd.load_confirmation(path, pool, digests)
    classes = rd.manifest_classes(confirmation)
    state = {"executed_prompt_keys": list(classes["S1-REF"]) + list(classes["S1-VALIDITY"])}
    rd.assert_no_target_prompt_executed(state, confirmation)  # stage 1's own keys are allowed at the barrier
    state["executed_prompt_keys"].append(classes["S2-TARGET"][0])
    with pytest.raises(rd.PhaseError, match="before the barrier"):
        rd.assert_no_target_prompt_executed(state, confirmation)
    with pytest.raises(rd.PhaseError, match="confirmation prompts"):
        rd.assert_confirmation_untouched({"executed_prompt_keys": [classes["S1-REF"][0]]}, confirmation)
