"""Experiment 008: frozen constants, the anomaly score, the collapse rule on oriented traces, the classification and summary rules, and the measurements on a six-layer fake."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M
from plural_fakes import TinyPlural

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_constants_match_the_design():
    assert cs.S_MIN == 0.3 and cs.KAPPA == 0.5 and cs.G_MAX == 0.25 and cs.X_MIN == 0.5 and cs.C_MIN == 2.0
    assert cs.PROBE_FLOOR == 0.5 and cs.CONSENSUS == 0.75 and cs.STAGE_UNINFORMATIVE_FLOOR == 0.25 and cs.CANCELLATION_DENOMINATOR_FLOOR == 0.25
    assert cs.STRATUM_THRESHOLD == 1.5 and cs.IDENTITY_TOLERANCE == 1e-4 and cs.REPLICATION_TOLERANCE == 1e-6
    assert cs.RUNTIME_SEED == 20260916 and cs.CONTROL_SEED == 20260921
    assert cs.RUNNING_STAGES == ("R0", "R1", "R2", "R3", "c") and cs.HEAD_KEY == "L03.H04"


def test_anomaly_score_is_sign_normalized():
    assert cs.anomaly_score(-3.0, 0.0) == -3.0 and cs.anomaly_score(3.0, 0.0) == -3.0
    assert cs.anomaly_score(-3.0, -5.0) == 2.0 and cs.anomaly_score(3.0, 5.0) == 2.0
    assert cs.anomaly_score(0.0, 1.0) == 1.0  # sign(0) = +1
    assert cs.stratum(-1.5) == "suppressed" and cs.stratum(1.5) == "amplified" and cs.stratum(-1.49) == "ordinary"


def test_collapse_rule_on_oriented_traces():
    # A strengthening negative raw signal is not a collapse once oriented: raw −2 → −3 becomes q 2 → 3.
    assert cs.collapse_stage({"R0": 2.0, "R1": 3.0, "R2": 3.0, "R3": 2.5, "c": 2.0})["stage"] == "NO_COLLAPSE"
    # A drop to at most half collapses at that stage.
    assert cs.collapse_stage({"R0": 1.0, "R1": 0.9, "R2": 0.45, "R3": 0.4, "c": 0.4})["stage"] == "R2"
    # A sign flip relative to the encoding signal collapses.
    verdict = cs.collapse_stage({"R0": 1.0, "R1": 0.8, "R2": 0.9, "R3": -0.2, "c": -0.1})
    assert verdict["stage"] == "R3" and verdict["previous"] == "R2"
    # No encoding signal: nothing to trace.
    assert cs.collapse_stage({"R0": 0.2, "R1": 0.0, "R2": 0.0, "R3": 0.0, "c": 0.0})["stage"] == "NO_SIGNAL"
    # Uninformative stages are skipped.
    assert cs.collapse_stage({"R0": 1.0, "R1": None, "R2": 0.9, "R3": 0.3, "c": 0.3})["stage"] == "R3"
    # Transport sub-decision at R2.
    head = cs.collapse_stage({"R0": 1.0, "R1": 1.0, "T": 0.2, "R2": 0.3, "R3": 0.3, "c": 0.3})
    other = cs.collapse_stage({"R0": 1.0, "R1": 1.0, "T": 0.9, "R2": 0.3, "R3": 0.3, "c": 0.3})
    assert head["transport"] == "HEAD_DROPPED" and other["transport"] == "OTHER_LAYER3_CANCELLED"
    # A previous stage below S_MIN cannot be the base of a collapse.
    assert cs.collapse_stage({"R0": 1.0, "R1": 0.25, "R2": 0.1, "R3": 0.1, "c": 0.1})["stage"] == "R1"
    # Revision 4: the rule runs from R0 literally; a weak R0 followed by a strong R1 that then collapses is a collapse at that stage.
    assert cs.collapse_stage({"R0": 0.1, "R1": 0.6, "R2": 0.1, "R3": 0.1, "c": 0.1})["stage"] == "R2"
    # A trace that never reaches S_MIN at any stage before the end has no signal to trace.
    assert cs.collapse_stage({"R0": 0.1, "R1": 0.2, "R2": 0.1, "R3": 0.1, "c": 0.9})["stage"] == "NO_SIGNAL"
    assert cs.collapse_stage({"R0": None, "R1": None, "R2": None, "R3": None, "c": None})["stage"] == "NO_SIGNAL"


def test_encoding_class_branches_and_mode_aggregation():
    assert cs.encoding_class(0.8, -0.7, 0.05, 0.9) == "LINEAR_CANCELLATION"
    assert cs.encoding_class(0.8, 0.0, -0.7, 0.9) == "NONLINEAR_GATING"
    assert cs.encoding_class(0.1, 0.0, 0.1, 0.9) == "AXIS_NOT_SUFFICIENT"
    assert cs.encoding_class(0.9, 0.05, 0.05, 0.9) == "ADDITIVE_ORDINARY"
    assert cs.encoding_class(None, 0.0, 0.0, 0.9) is None
    assert cs._mode_stage(["R2", "R2", "R3", "NO_SIGNAL"]) == ("R2", pytest.approx(2 / 3))
    assert cs._mode_stage(["R3", "R1"]) == ("R1", 0.5)  # ties go to the earliest stage
    assert cs._mode_stage(["NO_SIGNAL"]) == (None, None)


def _row(token, stratum, *, carries=True, stage="R2", cls="NONLINEAR_GATING", context="CONTEXT_NEUTRAL", artifact=False):
    return {"token": token, "stratum": stratum, "carries_signal": carries, "collapse_stage": stage, "encoding_class": cls, "context_class": context, "axis_artifact": artifact}


def test_summary_labels():
    rows = [_row("this", "suppressed"), _row("another", "suppressed"), _row("a", "suppressed"), _row("every", "suppressed", stage="R3"), _row("two", "ordinary")]
    assert cs.summarize(rows, True)["label"] == "LOCALIZED_R2_NONLINEAR_GATING"
    mixed = rows[:2] + [_row("a", "suppressed", stage="R3"), _row("every", "suppressed", stage="R3")]
    assert cs.summarize(mixed, True)["label"] == "MIXED"
    artifact = [_row("this", "suppressed", carries=False, artifact=True), _row("a", "suppressed", carries=False, artifact=True)]
    assert cs.summarize(artifact, True)["label"] == "AXIS_ARTIFACT"
    gated = [_row("this", "suppressed", stage="R2", context="CONTEXT_GATED"), _row("a", "suppressed", stage="R3", context="CONTEXT_GATED"), _row("every", "suppressed", stage="c", context="CONTEXT_GATED")]
    assert cs.summarize(gated, True)["label"] == "CONTEXT_LOCALIZED"
    invalid = cs.summarize(rows, False)
    assert invalid["label"] == "PROBE_INVALID" and invalid["trace_summary"] == "LOCALIZED_R2"
    assert cs.summarize([_row("two", "ordinary")], True)["label"] == "NO_SUPPRESSED_TOKENS"
    # Revision 4: NO_COLLAPSE is not a collapse stage; a gated stratum without a common collapse stage is CONTEXT_LOCALIZED.
    gated_flat = [_row("this", "suppressed", stage="NO_COLLAPSE", context="CONTEXT_GATED"), _row("a", "suppressed", stage="NO_COLLAPSE", context="CONTEXT_GATED")]
    assert cs.summarize(gated_flat, True)["label"] == "CONTEXT_LOCALIZED"
    assert cs.summarize([_row("this", "suppressed", stage="NO_COLLAPSE"), _row("a", "suppressed", stage="NO_COLLAPSE")], True)["label"] == "MIXED"
    # Joint consensus: stage and class must agree on the same tokens.
    split = [_row("t1", "suppressed", stage="R2", cls="A"), _row("t2", "suppressed", stage="R2", cls="A"), _row("t3", "suppressed", stage="R2", cls="B"), _row("t4", "suppressed", stage="R3", cls="A")]
    assert cs.summarize(split, True)["label"] == "MIXED"
    # Probe invalidity takes precedence over every other label, including the axis artifact.
    assert cs.summarize(artifact, False)["label"] == "PROBE_INVALID"


# ---------------------------------------------------------------------------
# Fake-model measurements with the real frozen inputs.


def make_fake_model():
    return TinyPlural(seed=3, d_vocab=60000, n_layers=6, n_heads=8)


@pytest.fixture(scope="module")
def fake_setting():
    manifest, manifest_sha256, extension = pm.load_inputs(ROOT)
    confirmation = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    pool = cs.build_pool(manifest, extension, confirmation)
    model = make_fake_model()
    weights = pm.Weights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    return pool, model, weights, cache


def test_pool_composition(fake_setting):
    pool, *_ = fake_setting
    assert len(pool.frames) == 18 and len(pool.tokens) == 40 and len(pool.nouns) == 80 and len(pool.single_nouns) == 79
    assert sorted(set(pool.frame_origin.values())) == ["confirmation", "extension", "manifest"]
    assert pool.token_source["this"] == "confirmation-24" and pool.token_source["cardinal:sg"] == "exposed-16"
    assert pool.token_category["this"] == "determiner" and pool.token_category["a"] == "extension" and pool.token_category["cardinal:pl"] == "original-cue"
    assert pool.plural_cue == {"cardinal": "cardinal:pl", "quantifier": "quantifier:pl", "coordinated-adjective": "cardinal:pl"}
    assert len(pool.single_nouns_from("exposed-60")) == 59 and len(pool.single_nouns_from("confirmation-20")) == 20


def test_axes_and_measurements_on_the_fake(fake_setting):
    pool, model, weights, cache = fake_setting
    small = cs.Pool008(pool.frames[:2] + pool.frames_of("quantifier")[:1], pool.frame_origin, pool.tokens[:6], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    axes = cs.stage_axes(cache, weights, small)
    assert set(axes) == set(cs.VECTOR_STAGES)
    for axis in axes.values():
        assert float(axis.direction.norm()) == pytest.approx(1.0, abs=1e-6) and axis.sigma > 0
    records = cs.measure_pool(model, weights, cache, small, axes["R0"])
    assert len(records) == len(small.frames) * len(small.tokens)
    frame = small.frames[0]
    plural = records[(small.plural_cue[frame.template_id], frame.frame_id)]
    reference_name = next(name for name, token_id in small.tokens if token_id == small.reference_ids[frame.template_id])
    ref_record = records[(reference_name, frame.frame_id)]
    assert ref_record.dc == 0.0 and ref_record.dc_par == 0.0 and all(float(v.abs().max()) == 0.0 for v in ref_record.delta.values())
    assert ref_record.dc_context_only == 0.0 and ref_record.dc_beh == 0.0
    # R0 changes by exactly ΔE; the component split reconstructs ΔE; direct-effect deltas sum to the shift.
    for record in records.values():
        assert record.identity_error <= cs.IDENTITY_TOLERANCE and abs(record.additivity_check) <= 1e-4
    e_ref = pm.lexicon_vector(weights, small.reference_ids[frame.template_id]).double()
    e_pl = pm.lexicon_vector(weights, plural.token_id).double()
    assert torch.allclose(plural.delta["R0"], e_pl - e_ref, atol=1e-4)
    # The plural cue's own fractions are one at every stage by construction; the x_out endpoint of the plural cue is its clean prompt.
    u1 = torch.zeros(weights.W_E.shape[1], dtype=torch.float64)
    u1[0] = 1.0
    analysis = cs.frame_analysis(plural, plural, axes, u1, axes["R0"], e_pl - e_ref, e_pl - e_ref)
    assert all(value == pytest.approx(1.0) for value in analysis["fractions"].values() if value is not None)
    assert analysis["oriented"]["R0"] == pytest.approx(1.0) and analysis["carries_signal"]
    assert analysis["collapse"]["stage"] == "NO_COLLAPSE"
    other = records[(small.tokens[4][0], frame.frame_id)]
    other_analysis = cs.frame_analysis(other, plural, axes, u1, axes["R0"], pm.lexicon_vector(weights, other.token_id).double() - e_ref, e_pl - e_ref)
    assert set(other_analysis["fractions"]) == set(cs.VECTOR_STAGES) | {"c"}
    assert other_analysis["oriented"]["R0"] is None or other_analysis["oriented"]["R0"] >= 0.0
    assert other_analysis["cancellation_index"] is None or other_analysis["cancellation_index"] >= 1.0 - 1e-9
    assert set(other_analysis["dde_circuit"]) == {"L03.H04", "L04.MLP", "L05.MLP"}


def test_inherited_007_extract_roundtrip(tmp_path):
    responses = {f"t{i}|f{j}": {"template_id": "cardinal", "mean_shift": 0.1 * i} for i in range(24) for j in range(6)}
    payload = cs.inherited_007_payload(responses, source={"path": "x"}, manifest_sha256="m", extension_sha256="e", confirmation_sha256="c", lock_sha256="l",
                                       model={"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision})
    path = tmp_path / "x.json"
    path.write_text(pm.canonical_json(payload) + "\n")
    assert cs.load_inherited_007(path, manifest_sha256="m", extension_sha256="e", confirmation_sha256="c")["content_sha256"] == payload["content_sha256"]
    with pytest.raises(ValueError):
        cs.load_inherited_007(path, manifest_sha256="other", extension_sha256="e", confirmation_sha256="c")
    measured = {key: entry["mean_shift"] for key, entry in payload["responses"].items()}
    assert cs.check_replication(measured, payload["responses"], label="x")["passed"]
    measured["t1|f0"] += 2e-6
    with pytest.raises(pm.IncidentError, match="deviates"):
        cs.check_replication(measured, payload["responses"], label="x")
    with pytest.raises(pm.IncidentError, match="not recomputed"):
        cs.check_replication({}, payload["responses"], label="x")


def test_committed_007_extract_matches_the_frozen_inputs():
    manifest, manifest_sha256, extension = pm.load_inputs(ROOT)
    confirmation = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    extract = cs.load_inherited_007(ROOT / cs.INHERITED_007_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation.content_sha256)
    lock = json.loads((ROOT / cs.EXPERIMENT_007_LOCK_PATH).read_text())
    assert extract["lock_sha256"] == lock["content_sha256"]
    assert set(extract["responses"]) == {f"{token['word']}|{frame.frame_id}" for token in confirmation.tokens for frame in confirmation.frames}


def test_probe_validity_and_token_row_rules(fake_setting):
    pool, model, weights, cache = fake_setting
    frames = pool.frames[:4]
    small = cs.Pool008(frames, pool.frame_origin, pool.tokens[:5], pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    # Synthetic per-frame analyses for the plural cue (valid probe) and one token.
    def analysis(r_par, *, s_r0=0.9, stage="R2", x_in=0.9, C=1.2, g=0.0, r_perp=0.0):
        oriented = {s: (s_r0 if s in ("R0", "R1", "T") else 0.2) for s in cs.VECTOR_STAGES + ("c",)}
        return {"fractions": {s: oriented[s] for s in cs.VECTOR_STAGES + ("c",)}, "oriented": oriented, "s_u1": 0.8, "r_par": r_par, "r_perp": r_perp, "r_full": r_par + r_perp + g, "g": g,
                "x_in": x_in, "x_out": 0.1, "context_only": 0.0, "behavior": 0.1, "attention_fraction": 1.0, "attention_delta": 0.0, "cancellation_index": C,
                "opposing_terms": {}, "dde_circuit": {}, "collapse": {"stage": stage, "transport": "HEAD_DROPPED" if stage == "R2" else None}, "dc": -1.0, "dc_beh": -1.0}
    plural_names = {small.plural_cue[frame.template_id] for frame in frames}
    per_frame_by_token = {name: {frame.frame_id: analysis(0.9 if name in plural_names else 0.8) for frame in frames} for name, _ in small.tokens}
    probe = cs.probe_validity(per_frame_by_token, small)
    assert probe["valid"] and probe["frames_ok"] == 4
    for name in plural_names:
        for frame in frames:
            per_frame_by_token[name][frame.frame_id]["r_par"] = 0.4
    assert not cs.probe_validity(per_frame_by_token, small)["valid"]
    token = "cardinal:sg" if "cardinal:sg" in dict(small.tokens) else small.tokens[0][0]
    scores = {frame.frame_id: {"p": -3.0, "m": 0.0, "a": cs.anomaly_score(-3.0, 0.0)} for frame in frames}
    plural_cancellation = {frame.frame_id: 1.0 for frame in frames}
    row = cs.token_row(token, small, {frame.frame_id: analysis(0.8, C=2.5, x_in=0.2) for frame in frames}, scores, plural_cancellation, True)
    assert row["stratum"] == "suppressed" and row["anomaly_score"] == -3.0 and row["in_sample_for_007"] and row["carries_signal"]
    assert row["encoding_class"] == "ADDITIVE_ORDINARY" and row["collapse_stage"] == "R2" and row["transport_detail"] == "HEAD_DROPPED"
    assert row["context_class"] == "CONTEXT_GATED" and row["late_cancellation"] and not row["axis_artifact"]
    # Class withheld when the probe is invalid or the token carries no signal; axis artifact for a suppressed token without signal.
    assert cs.token_row(token, small, {frame.frame_id: analysis(0.8) for frame in frames}, scores, plural_cancellation, False)["encoding_class"] is None
    weak = cs.token_row(token, small, {frame.frame_id: analysis(0.8, s_r0=0.1, stage="NO_SIGNAL") for frame in frames}, scores, plural_cancellation, True)
    assert weak["encoding_class"] is None and weak["axis_artifact"] and weak["collapse_stage"] is None
    assert set(row["per_template"]) == set(pm.TEMPLATE_ORDER) and "oriented" in row["per_template"]["cardinal"]


def test_m3_conventions_and_identity_incident(fake_setting, monkeypatch):
    pool, model, weights, cache = fake_setting
    frame = pool.frames_of("quantifier")[0]
    names = [pool.plural_cue["quantifier"], next(name for name, token_id in pool.tokens if token_id == pool.reference_ids["quantifier"]), "this"]
    small = cs.Pool008((frame,), pool.frame_origin, tuple((name, pool.token_id(name)) for name in names), pool.token_category, pool.token_source, pool.nouns, pool.noun_source, pool.reference_ids, pool.plural_cue)
    axes = cs.stage_axes(cache, weights, small)
    records = cs.measure_pool(model, weights, cache, small, axes["R0"])
    plural, reference = records[(names[0], frame.frame_id)], records[(names[1], frame.frame_id)]
    assert reference.dc_out == 0.0 and reference.dc_context_only == 0.0  # E(ref) in the plural context is the shared baseline itself
    assert plural.dc_in == pytest.approx(plural.dc_out, abs=1e-9)  # both endpoints of the plural cue are its clean prompt versus E(ref) in that prompt
    assert plural.dc_in == pytest.approx(plural.dc_beh - plural.dc_context_only, abs=1e-9)
    monkeypatch.setattr(cs, "IDENTITY_TOLERANCE", -1.0)
    with pytest.raises(pm.IncidentError, match="does not change by"):
        cs.measure_pool(model, weights, cache, small, axes["R0"])


def test_load_program_007_refuses_a_digest_mismatch(tmp_path):
    root = tmp_path
    (root / cs.EXPERIMENT_007_LOCK_PATH).parent.mkdir(parents=True)
    (root / cs.EXPERIMENT_007_LOCK_PATH).write_text(json.dumps({"parameters": {"selected": "0" * 64}, "content_sha256": "x", "program_source_sha256": "y"}))
    directory = root / "program"
    directory.mkdir()
    (directory / "parameters.json").write_text("{}")
    with pytest.raises(cs.PhaseError, match="does not match"):
        cs.load_program_007(root, parameters_dir=directory)
    with pytest.raises(cs.PhaseError, match="not available"):
        cs.load_program_007(root, parameters_dir=root / "missing")
