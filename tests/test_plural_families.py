"""Floors, bands, band hits, revision rule, contested check, and outcome labels on synthetic results."""

from __future__ import annotations

import random

import pytest

from neural_decompiler import plural_mechanism as pm


def test_exact_count_floors_match_the_spec_numbers():
    assert pm.exact_count_floor(pm.FLOOR_RATES["B1_accuracy"], 120) == 103
    assert pm.exact_count_floor(pm.FLOOR_RATES["B1_flip"], 120) == 96
    assert pm.exact_count_floor(pm.FLOOR_RATES["B2_positive"], 120) == 114
    assert pm.exact_count_floor(pm.FLOOR_RATES["P3_retention"], 120) == 84
    assert pm.exact_count_floor(pm.FLOOR_RATES["X1_positive"], 120) == 108
    assert pm.exact_count_floor(pm.FLOOR_RATES["X1_variables"], 12) == 11
    assert pm.exact_count_floor(pm.FLOOR_RATES["X3_sign_word"], 6) == 5
    assert pm.exact_count_floor(pm.FLOOR_RATES["X3_sign_overall"], 6 * 8) == 44
    assert pm.exact_count_floor(pm.FLOOR_RATES["S3"], 22) == 18
    assert pm.exact_count_floor(pm.FLOOR_RATES["B1_accuracy"], 114) == 98


def _rows(n: int, d_full: float, d_patch: float, template: str = "cardinal", rule: str = "simple-suffix"):
    return [{"frame_id": f"{template}-1", "template_id": template, "lexical_key": f"n{i}", "rule_class": rule, "d_full": d_full, "d_patch": d_patch} for i in range(n)]


def _summary(recovery: float):
    entry = {"cases": 40, "mean_d_full": 5.0, "mean_d_patch": 5.0 * recovery, "denominator_valid": True, "recovery": recovery}
    return {"overall": dict(entry), "templates": {t: dict(entry) for t in pm.TEMPLATE_ORDER}, "rule_classes": {r: dict(entry) for r in pm.RULE_CLASSES}}


def _results(*, recovery=0.9, positive=118, primary=110, flips=100, faithfulness=0.7, retained=100, p4=0.7, p5_single=0.7, blocked=0.7,
             p6_opp=0.7, p6_same=0.1, p7_same=0.1, p7_opp=0.8, loss=0.5, m_t=0.8, m_r=0.8, m_r_frozen=0.2, s1=0.9, n_wrong=4, n_t_matches=4):
    rows = _rows(40, 5.0, 5.0 * recovery)
    return {
        "behavior": {"cases": 120, "primary_correct": primary, "flips": flips, "positive_pairs": positive, "template_mean_d_full": {t: 5.0 for t in pm.TEMPLATE_ORDER},
                     "variables": {"n_c": 12, "n_t": 12, "prompts": 12}, "s3": {"n_wrong": n_wrong, "n_t_matches": n_t_matches}, "rows": [], "wrong_conditions": []},
        "P1": {"summary": _summary(recovery), "rows": rows}, "P2": {"median": 0.02, "reference": 0.05, "recoveries": [], "sets": []},
        "P3": {"summary": _summary(faithfulness), "sign_retention": {"retained": retained, "total": 120}, "rows": _rows(40, 5.0, 5.0 * faithfulness)},
        "P4": {"variant": "P4", "value": p4, "rows": _rows(40, 5.0, 5.0 * p4, "coordinated-adjective")},
        "P5": {"applicable": True, "a_single": p5_single, "a_declared": p5_single + 0.1, "b_unfrozen": 0.9, "b_frozen": 0.9 * (1 - blocked), "b_blocked_fraction": blocked,
               "rows": {"single": _rows(40, 5.0, 5.0 * p5_single, "coordinated-adjective"), "declared": _rows(40, 5.0, 5.0 * (p5_single + 0.1), "coordinated-adjective"),
                        "unfrozen": _rows(40, 5.0, 4.5, "coordinated-adjective"), "frozen": _rows(40, 5.0, 4.5 * (1 - blocked), "coordinated-adjective")}},
        "P6": {"applicable": True, "opposite": p6_opp, "same": p6_same, "rows_opposite": _rows(40, 5.0, 5.0 * p6_opp), "rows_same": _rows(40, 5.0, 5.0 * p6_same)},
        "P7": {"same": p7_same, "opposite": p7_opp, "rows_same": _rows(40, 5.0, 5.0 * p7_same), "rows_opposite": _rows(40, 5.0, 5.0 * p7_opp)},
        "P8": {"loss": loss, "compensation_ratio": 0.1, "rows": _rows(40, 5.0, 5.0 * loss)},
        "P9": {"applicable": True, "m_T": m_t, "m_R": m_r, "m_R_given_T_frozen": m_r_frozen},
        "S1": {"applicable": True, "cosines": {"a|b": s1}, "minimum": s1},
        "program_residual": {"conditions": 240, "rmse": 0.4, "mae": 0.3, "entries": []},
    }


def test_circuit_floors_pass_and_fail_by_family():
    floors = pm.circuit_floors(_results(), hypothesis="H1", n_cases=120)
    assert floors["circuit_passed"] and floors["primary_failures"] == []
    assert floors["B1"]["required"] == {"accuracy": 103, "flips": 96}
    failing = pm.circuit_floors(_results(recovery=0.65, blocked=0.4, n_wrong=5, n_t_matches=3), hypothesis="H1", n_cases=120)
    assert set(failing["primary_failures"]) == {"P1", "P5", "P7"} or set(failing["primary_failures"]) >= {"P1", "P5"}
    assert not failing["S3"]["passed"]
    assert not pm.circuit_floors(_results(p5_single=0.4, blocked=0.4), hypothesis="H2", n_cases=120)["P5"]["passed"]
    assert pm.circuit_floors(_results(p5_single=0.65, blocked=0.4), hypothesis="H2", n_cases=120)["P5"]["passed"]
    precondition = pm.circuit_floors(_results(primary=100), hypothesis="H1", n_cases=120)
    assert not precondition["precondition_passed"] and not precondition["circuit_passed"]


def test_bands_are_deterministic_and_follow_the_width_rule():
    results = _results()
    first, second = pm.calibration_bands(results), pm.calibration_bands(results)
    assert first == second
    band = first["P1"]["overall"]
    assert band["high"] - band["point"] >= pm.BAND_MIN_WIDTH - 1e-12
    assert first["B2"]["cardinal"] == {"point": 5.0, "low": 4.5, "high": 5.5}
    assert first["P9"]["m_T"]["high"] - first["P9"]["m_T"]["low"] == pytest.approx(2 * pm.BAND_MIN_WIDTH)
    generator = random.Random(1)
    rows = [pm.CaseRow("f", "cardinal", f"n{i}", "simple-suffix", 5.0 + (i % 3), 4.0 + (i % 2)) for i in range(20)]
    boot = pm.bootstrap_rows(rows, pm._stat_recovery, generator, resamples=200)
    assert boot["se"] > 0.0 and boot["point"] == pytest.approx(pm._stat_recovery(rows))
    assert pm.tolerance_tau(0.1) == 0.5 and pm.tolerance_tau(0.3) == pytest.approx(0.9)


def test_band_hits_and_outcome_labels():
    results = _results()
    bands = pm.calibration_bands(results)
    hits = pm.band_hits(results, bands)
    assert hits["all_hit"] and hits["missed"] == []
    shifted = _results(recovery=0.6)
    assert "P1:overall" in pm.band_hits(shifted, bands)["missed"]
    circuit = pm.circuit_floors(results, hypothesis="H1", n_cases=120)
    deco_pass = {"program_passed": True, "primary_failures": [], "program_capped": False}
    deco_fail = {"program_passed": False, "primary_failures": ["X3"], "program_capped": False}
    x_hits = {"X3": {"hit": True, "words_hit": 10}, "X4": {"hit": True, "words_hit": 11}}
    assert pm.outcome(circuit=circuit, hits=hits, decompilation=deco_pass, x_hits=x_hits, contested_findings=[])["label"] == "MECHANISM_CONFIRMED"
    assert pm.outcome(circuit=circuit, hits=pm.band_hits(shifted, bands), decompilation=deco_pass, x_hits=x_hits, contested_findings=[])["label"] == "MECHANISM_SUPPORTED_MISCALIBRATED"
    assert pm.outcome(circuit=circuit, hits=hits, decompilation=deco_fail, x_hits=x_hits, contested_findings=[])["label"] == "CIRCUIT_ONLY"
    assert pm.outcome(circuit=circuit, hits=hits, decompilation=deco_pass, x_hits=x_hits, contested_findings=["P5.b_blocked_fraction=0.300 inside H2"])["label"] == "MECHANISM_CONTESTED"
    failed = pm.circuit_floors(_results(recovery=0.5), hypothesis="H1", n_cases=120)
    assert pm.outcome(circuit=failed, hits=hits, decompilation=deco_pass, x_hits=x_hits, contested_findings=[])["label"] == "MECHANISM_NOT_SUPPORTED"
    behavior = pm.circuit_floors(_results(primary=90), hypothesis="H1", n_cases=120)
    assert pm.outcome(circuit=behavior, hits=hits, decompilation=deco_pass, x_hits=x_hits, contested_findings=[])["label"] == "BEHAVIOR_NOT_REPLICATED"
    x_missed = {"X3": {"hit": False, "words_hit": 5}, "X4": {"hit": True, "words_hit": 11}}
    assert pm.outcome(circuit=circuit, hits=hits, decompilation=deco_pass, x_hits=x_missed, contested_findings=[])["label"] == "MECHANISM_SUPPORTED_MISCALIBRATED"


def test_contested_requires_outside_chosen_and_inside_rejected():
    results = _results(blocked=0.3)
    bands = pm.calibration_bands(_results(blocked=0.7))
    findings = pm.contested(results, hypothesis="H1", bands=bands)
    assert any("P5.b_blocked_fraction" in item and "H2" in item for item in findings)
    assert pm.contested(_results(blocked=0.7), hypothesis="H1", bands=bands) == []


def test_revise_version_rules():
    previous = {"version": "M1", "hypothesis": "H1", "mechanism": {"branch": "L00.MLP", "e_keys": ["L00.MLP"], "t_keys": ["L03.H04"], "r_keys": ["L04.MLP"]}}
    a2 = {"rankings": {"p_t_all_templates": ["L00.MLP", "L03.H04", "L04.MLP", "L05.MLP", "L02.MLP", "L05.H07", "L00.H05", "L01.H03"]}}
    calibration = {"floors": {"primary_failures": ["P1"], "precondition_passed": True}}
    assert pm.revise_version(previous, calibration, a2) == {"version": "M2", "action": "extend", "add": ["L05.MLP"], "failed": ["P1"]}
    calibration = {"floors": {"primary_failures": ["P5"], "precondition_passed": True}}
    assert pm.revise_version(previous, calibration, a2)["row"] == "H2"
    calibration = {"floors": {"primary_failures": ["P1"], "precondition_passed": False}}
    assert pm.revise_version(previous, calibration, a2)["action"] == "reject"
    with pytest.raises(pm.PhaseError):
        pm.revise_version({**previous, "version": "M3"}, {"floors": {"primary_failures": ["P1"], "precondition_passed": True}}, a2)
    full = {**previous, "mechanism": {**previous["mechanism"], "t_keys": ["L03.H04", "L05.H07", "L00.H05"], "r_keys": ["L04.MLP", "L05.MLP", "L02.MLP"]}}
    assert pm.revise_version(full, {"floors": {"primary_failures": ["P3"], "precondition_passed": True}}, a2)["action"] == "reject"


def test_spearman_handles_ties_and_monotone_sequences():
    assert pm.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert pm.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert -1.0 <= pm.spearman([1, 1, 2, 3], [3, 1, 2, 2]) <= 1.0
