"""Experiment 006: split-invariant floors, Y families, bands, and the outcome rule."""

from __future__ import annotations

import pytest

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import plural_mechanism as pm
from plural_fakes import make_context


def _summary(recovery: float):
    entry = {"cases": 40, "mean_d_full": 5.0, "mean_d_patch": 5.0 * recovery, "denominator_valid": True, "recovery": recovery}
    return {"overall": dict(entry), "templates": {t: dict(entry) for t in pm.TEMPLATE_ORDER}, "rule_classes": {r: dict(entry) for r in pm.RULE_CLASSES}}


def _results(*, positive=120, recovery=0.9, f=0.8, signs=118, corr=0.95, p4=0.8, single=0.8, blocked=0.8, same=0.05, opp=0.85, loss=0.5, m_t=0.9, m_r=0.95, m_rt=0.1):
    return {"behavior": {"cases": 120, "positive_pairs": positive, "primary_correct": 100, "flips": 80},
            "P1": {"summary": _summary(recovery)}, "P3": {"summary": _summary(f), "sign_agreement": signs, "pairs": 120, "correlation": corr},
            "P4": {"value": p4}, "P5": {"applicable": True, "a_single": single, "b_blocked_fraction": blocked}, "P7": {"same": same, "opposite": opp},
            "P8": {"loss": loss, "compensation_ratio": -1.0}, "P9": {"applicable": True, "m_T": m_t, "m_R": m_r, "m_R_given_T_frozen": m_rt}}


def test_circuit_floors_and_cue_effect_gate():
    floors = cd.circuit_floors(_results(), fresh=False)
    assert floors["passed"] and floors["failures"] == [] and floors["cue_effect"]["required"] == 114
    assert cd.circuit_floors(_results(), fresh=True)["cue_effect"]["required"] == 108
    assert not cd.circuit_floors(_results(positive=113), fresh=False)["cue_effect"]["passed"]
    assert cd.circuit_floors(_results(positive=110), fresh=True)["cue_effect"]["passed"]
    assert cd.circuit_floors(_results(signs=113), fresh=False)["failures"] == ["P3"]
    assert cd.circuit_floors(_results(corr=0.89), fresh=False)["failures"] == ["P3"]
    assert cd.circuit_floors(_results(same=0.3), fresh=False)["failures"] == ["P7"]
    assert cd.circuit_floors(_results(loss=0.2), fresh=False)["failures"] == ["P8"]
    assert cd.circuit_floors(_results(m_rt=0.6), fresh=False)["failures"] == ["P9"]


def _per_token(n_tokens=24, n_frames=6, *, predicted_scale=1.0, noise=0.0):
    per_token = {}
    for i in range(n_tokens):
        base = -4.0 + 8.0 * i / (n_tokens - 1)
        frames = {}
        for j in range(n_frames):
            template = pm.TEMPLATE_ORDER[j // 2]
            measured = base + 0.1 * j
            predicted = predicted_scale * base + noise * ((-1) ** j)
            frames[f"{template}-fresh-{j % 2 + 1}"] = {"template_id": template, "epatch_measured": measured, "behavior_measured": measured + 0.2,
                                                        "predicted:selected": predicted, "epatch_mae:selected": abs(predicted - measured), "behavior_mae:selected": abs(predicted - measured - 0.2)}
        per_token[f"w{i}"] = {"token_id": i, "category": "numeral", "frames": frames}
    return per_token


def test_y_floors_bands_and_outcome():
    y3 = {template: {"mean_measured": 4.0, "mean_predicted:selected": 3.8} for template in pm.TEMPLATE_ORDER}
    y3.update({"positive_pairs": 118, "pairs": 120})
    measurements = {"per_token": _per_token(), "Y3": y3}
    floors = cd.y_floors(measurements, tau=0.6)
    assert floors["passed"] and floors["Y1"]["spearman"] == pytest.approx(1.0)
    bands = cd.y_band_hits(measurements["per_token"], tau=0.6)
    assert bands["hit"] and bands["tokens_hit"] == 24
    weak = cd.y_floors({"per_token": _per_token(predicted_scale=0.4), "Y3": y3}, tau=0.6)
    assert not weak["Y1"]["passed"] and "Y1" in weak["failures"]
    ok = cd.circuit_floors(_results(), fresh=False)
    ok_fresh = cd.circuit_floors(_results(), fresh=True)
    assert cd.outcome(manifest_floors=ok, fresh_floors=ok_fresh, program_floors=floors, bands=bands)["label"] == "DECOMPILED"
    assert cd.outcome(manifest_floors=ok, fresh_floors=ok_fresh, program_floors=floors, bands={"hit": False})["label"] == "DECOMPILED_MISCALIBRATED"
    assert cd.outcome(manifest_floors=ok, fresh_floors=ok_fresh, program_floors=weak, bands=bands)["label"] == "CIRCUIT_ONLY"
    assert cd.outcome(manifest_floors=ok, fresh_floors=cd.circuit_floors(_results(loss=0.1), fresh=True), program_floors=floors, bands=bands)["label"] == "CIRCUIT_NOT_GENERALIZED"
    assert cd.outcome(manifest_floors=cd.circuit_floors(_results(recovery=0.5), fresh=False), fresh_floors=ok_fresh, program_floors=floors, bands=bands)["label"] == "NOT_SUPPORTED"
    assert cd.outcome(manifest_floors=cd.circuit_floors(_results(positive=100), fresh=False), fresh_floors=ok_fresh, program_floors=floors, bands=bands)["label"] == "CUE_EFFECT_NOT_REPLICATED"


def test_p3_fidelity_and_circuit_families_run_on_the_fake(tmp_path):
    ctx = make_context()
    circuit = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L01.H00",), ("L01.MLP",))
    parameters = pm.estimate_program_parameters(ctx, circuit)
    pm.export_program_parameters(tmp_path, ctx.weights, parameters, vocab_size=16)
    axes = pm.site_axes_from_parameters(tmp_path)
    ev = pm.EvalContext(ctx.model, ctx.weights, circuit, None, ctx.frames, ctx.nouns, ctx.cache, axes, "H1")
    fidelity = cd.p3_fidelity(ev)
    assert fidelity["pairs"] == 24 and -1.0 <= fidelity["correlation"] <= 1.0 and 0 <= fidelity["sign_agreement"] <= 24
    results = cd.circuit_families(ev, ctx.universe)
    assert set(results) == {"behavior", "P1", "P3", "P4", "P5", "P7", "P8", "P9"}
    floors = cd.circuit_floors(results, fresh=False)
    assert set(floors["cue_effect"]) >= {"passed", "required", "descriptive"}
