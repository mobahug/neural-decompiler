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
    assert cs.summarize([_row("two", "ordinary")], True)["label"] == "MIXED"


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
