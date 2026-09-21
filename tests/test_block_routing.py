"""Experiment 019: the per-position masked chain against Experiment 018's rungs on the fake (I8, I10), the selectors from locked states only (poisoned measurements and confirmation
cannot enter; a fresh token is refused), the leave-one-cue-out witness (deterministic, tie rule, planted recovery, the held-out cue's own u_j and read cannot enter), the oracles applied
to the same upstream parts, the leak-proof boundary, the routing statistics and every label branch on synthetic tables, the confirmation policy with eighteen frames, the pool and the
committed, reproducible Experiment 018 extract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_decompiler import attention_paths as ap
from neural_decompiler import attention_patterns as atp
from neural_decompiler import block_concentration as bc
from neural_decompiler import block_routing as br
from neural_decompiler import head_pattern as hp
from neural_decompiler import layer_correction as lc
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from test_block_concentration import _fake_setup, make_fake_model, toy_tokenizer_018
from test_block_concentration import inputs as inputs_018  # noqa: F401  (the fixture)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_constants_match_the_design():
    assert br.SIZES == (16, 64, 256) and br.DECISION_SIZE == 64 and br.DRIVE_QUANTILES == (0.1, 0.5, 0.9) and br.N_RANDOM_CONTROLS == 3 and br.N_NEURONS == 2048 and br.BLOCK == 2
    assert br.ROUTING_GAIN_FLOOR == 0.05 and br.SPLIT_GAIN_FLOOR == 0.02 and br.RHO_POOLED_FLOOR == 0.60 and br.RHO_SET_FLOOR == 0.50 and br.RHO_DENOMINATOR_MIN == 0.05 and br.TEMPLATE_MARGIN == 0.05
    assert br.MEMBERSHIP_FLOOR == 0.60 and br.MEMBERSHIP_MARGIN == 0.20 and br.HEADROOM_MIN == 0.05 and br.GAP_MIN == 0.05 and br.PRECONDITION_REFERENCE == {"c_L": 0.98, "rows": 0.95, "dT": 0.95}
    assert br.MIN_VALID_CUE_FINAL_FRAMES == 9 and br.MIN_VALID_COORDINATED_FRAMES == 4 and br.MIN_VALID_FRAMES_PER_TOKEN == 3 and br.MIN_SCORED_TOKENS == 16 and br.I8_TOLERANCE == 1e-9 and br.I10_TOLERANCE == 1e-9 and br.REPLICATION_TOLERANCE == 1e-6
    assert br.RUNTIME_SEED == 20260916 and br.CONTROL_SEED == 20260924 and len(br.FRESH_FRAMES) == 18 and br.EXPECTED_EXTRACT_SIZE_018 == 11796 and sum(br.QUOTAS.values()) == 24 and br.FRAME_ID_TAG == "019"
    assert all([t for t, _ in br.FRESH_FRAMES].count(template) == 6 for template in pm.TEMPLATE_ORDER)
    assert not ({text for _, text in br.FRESH_FRAMES} & ({text for _, text in hp.FRESH_FRAMES} | {text for _, text in bc.FRESH_FRAMES}))
    for category, words in hp.CANDIDATES.items():
        assert br.CANDIDATES[category][: len(words)] == words  # the 017 lists first, then the extension
    assert br.OUTCOME_Y1[2] == "ROUTING_NOT_EVALUABLE_TOKENS" and br.OUTCOME_Y3[0] == "OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT" and br.OUTCOME_Y4 == ("TEMPLATE_FAMILY_INSUFFICIENT", "TEMPLATE_FAMILY_SUFFICIENT", "TEMPLATE_FAMILY_NOT_DISTINGUISHED", "TEMPLATE_FAMILY_NOT_EVALUABLE")
    assert br.OUTCOME_Y5 == ("MEMBERSHIP_PREDICTED", "MEMBERSHIP_NOT_PREDICTED", "MEMBERSHIP_NOT_EVALUABLE") and "witness" in br.WITNESS_TERMINOLOGY and "upper bound" in br.WITNESS_TERMINOLOGY
    assert len(br.PROSPECTIVE_RUNGS) == 28 and br.ORACLE_RUNGS == ("O16", "O64", "O256", "Os16", "Os64", "Os256") and br.DECISION == {"global": "Sp64", "template": "T64", "full": "E64", "rule": "G64", "inherited": "L64", "oracle": "O64", "witness": "Os64"}
    assert br.PREDICTION_COLUMNS[:5] == ("token", "frame_id", "template", "p_c", "p_t") and {"row_E64", "c_L_Sp64", "F_S2048", "Pi_G16", "dT_R3_256", "row_L256", "c_L_E1", "dT_frozen", "c_L_level0F"} <= set(br.PREDICTION_COLUMNS)
    assert "c_L_O64" not in br.PREDICTION_COLUMNS and "c_L_Os64" in br.ORACLE_SCALAR_KEYS
    assert br.SCIENTIFIC_PATH_PREFIXES[:3] == ("src/", "experiments/019-block2-routing/", "experiments/018-block2-concentration/") and br.HEAD_KEY == "L03.H04"
    assert set(br.frozen_floors()) >= {"decision_size", "routing_gain_floor", "split_gain_floor", "rho_pooled_floor", "rho_set_floor", "rho_denominator_min", "template_margin", "membership_floor", "membership_margin", "headroom_min", "witness", "outcomes"}
    assert not hasattr(br, "greedy_in_sample") and not hasattr(br, "greedy_oracle_in_sample")


def test_top_k_random_controls_and_the_witness():
    scores = torch.zeros(br.N_NEURONS, dtype=torch.float64)
    scores[5] = scores[7] = 1.0
    scores[100] = 0.5
    assert br.top_k(scores, 2) == [5, 7] and br.top_k(scores, 3) == [5, 7, 100] and br.top_k(scores, 4) == [0, 5, 7, 100]  # ties by the lower index
    controls = br.random_controls()
    assert set(controls) == {"16", "64", "256"} and all(len(lists) == 3 and all(len(l) == int(k) and len(set(l)) == int(k) and l == sorted(l) for l in lists) for k, lists in controls.items())
    assert controls == br.random_controls(br.CONTROL_SEED) and controls["256"][0] != br.random_controls(1)["256"][0]
    # The witness: a planted subset is recovered exactly from the other cues' contribution vectors and reads.
    g = torch.Generator().manual_seed(2)
    n = 23
    u = torch.zeros(n, br.N_NEURONS, dtype=torch.float64)
    planted = [3, 77, 1500, 1987, 2047]
    for j in planted:
        u[:, j] = torch.randn(n, generator=g, dtype=torch.float64)
    u[:, 10] = 1e-3 * torch.randn(n, generator=g, dtype=torch.float64)  # a small distractor
    y = u[:, planted].sum(1)
    assert br.greedy_witness(u, y, 5) == planted
    assert br.greedy_witness(u, y, 6) == sorted(planted + [0])  # no early stop: with the residual at zero every unused neuron ties (a zero vector changes nothing; the distractor would add error) and the lowest index is taken
    assert br.greedy_witness(u, y, 5) == br.greedy_witness(u.clone(), y.clone(), 5)  # deterministic
    tied = torch.zeros(n, br.N_NEURONS, dtype=torch.float64)
    tied[:, 40] = tied[:, 30] = 1.0
    assert br.greedy_witness(tied, torch.ones(n, dtype=torch.float64), 1) == [30]  # exact ties: the lower index
    with pytest.raises(ValueError):
        br.greedy_witness(u[:, :10], y, 2)
    with pytest.raises(pm.IncidentError):
        br.greedy_witness(u[:0], y[:0], 2)
    assert br.apply_mask_read(0.5, u[0], planted) == pytest.approx(0.5 + float(u[0, planted].sum()))


def _synthetic_pairs(n_frames_by_template=(2, 2, 2), n_words=18, prefix="e", alphas=None, frame_alphas=None):
    """Pairs whose every rung's prediction blends the measured value with a fixed noise: κ(rung) = (1 − (1 − α)²) / (1 − (1 − α_ref)²) exactly."""
    alphas = {**{rung: 0.1 for rung in br.ALL_RUNGS}, br.TEMPLATE_BASE: 0.0, br.REFERENCE: 0.9, "Sp1": 0.2, "E1": 0.25, "Sp16": 0.4, "Sp64": 0.5, "Sp256": 0.7, "T16": 0.42, "T64": 0.55, "T256": 0.72, "E16": 0.55, "E64": 0.7, "E256": 0.8,
              "G16": 0.5, "G64": 0.65, "G256": 0.78, "L16": 0.4, "L64": 0.5, "L256": 0.7, "O16": 0.6, "O64": 0.75, "O256": 0.82, "Os16": 0.7, "Os64": 0.85, "Os256": 0.86, **(alphas or {})}
    frame_alphas = frame_alphas or {}
    frames = []
    for template, count in zip(pm.TEMPLATE_ORDER, n_frames_by_template):
        for i in range(count):
            frames.append((f"{prefix}-{template}-{i}", template))
    words = [f"w{i}" for i in range(n_words)]
    p_c = 3
    pairs, lists_E, lists_G, oracle_lists, meta = {}, {}, {}, {}, {}
    for fid, t in frames:
        p_t = p_c if t in pm.CUE_FINAL_TEMPLATES else p_c + 1
        meta[fid] = {"template": t, "p_c": p_c, "p_t": p_t, "cue_final": p_c == p_t}
        positions = sorted({p_c, p_t})
        lists_E[fid] = {str(pos): {str(k): list(range(k)) for k in br.SIZES} for pos in positions}
        lists_G[fid] = {str(pos): {str(k): list(range(k // 2)) + list(range(1000, 1000 + k - k // 2)) for k in br.SIZES} for pos in positions}
        oracle_lists[fid] = {str(pos): {str(k): list(range(k - k // 8)) + list(range(1500, 1500 + k // 8)) for k in br.SIZES} for pos in positions}
        for i, w in enumerate(words):
            seed = 1000 * i + 17 * (hash(fid) % 1000)
            g = torch.Generator().manual_seed(seed)
            r = lambda: float(torch.randn(1, generator=g, dtype=torch.float64))  # noqa: E731
            c_L, F, Pi = 0.3 * r() + 0.1, 0.6 * r() + 0.7, 0.5 * r()
            row = [0.1 * r() for _ in range(p_t + 1)]
            n = {"c_L": 0.55 * 0.3 * r(), "F": 0.55 * 0.6 * r(), "Pi": 0.55 * 0.5 * r()}
            n_row = [0.05 * r() for _ in range(p_t + 1)]
            pred, oracle = {"p_c": p_c, "p_t": p_t}, {}
            for rung in br.ALL_RUNGS:
                a = frame_alphas.get((fid, rung), alphas[rung])
                sink = pred if rung in br.PROSPECTIVE_RUNGS else oracle
                sink[f"c_L_{rung}"] = c_L + (1 - a) * n["c_L"]
                sink[f"F_{rung}"] = F + (1 - a) * n["F"]
                sink[f"Pi_{rung}"] = Pi + (1 - a) * n["Pi"]
                sink[f"dT_{rung}"] = sink[f"F_{rung}"] + sink[f"Pi_{rung}"]
                sink[f"row_{rung}"] = [x + (1 - a) * e for x, e in zip(row, n_row)]
            pred["dT_frozen"] = pred[f"F_{br.REFERENCE}"]
            pred["c_L_level0F"] = pred[f"c_L_{br.TEMPLATE_BASE}"]
            pairs[f"{w}|{fid}"] = {"token": w, "token_id": 100 + i, "frame_id": fid, "template": t, "p_c": p_c, "p_t": p_t, "cue_final": p_c == p_t, "row": row, "F": F, "Pi": Pi, "dT": F + Pi, "c_L": c_L, "prediction": pred, "oracle": oracle}
    return pairs, words, meta, lists_E, lists_G, oracle_lists


K_REF = 1.0 - (1.0 - 0.9) ** 2
K = lambda a: (1.0 - (1.0 - a) ** 2) / K_REF  # noqa: E731


def _lock_and_stage1(meta_exposed, lists_E_e, lists_G_e, meta_fresh, lists_E_n, lists_G_n):
    sp = {pt: {str(k): list(range(k - k // 3)) + list(range(2000, 2000 + k // 3)) for k in br.SIZES} for pt in br.POSITION_TYPES}
    lists = {"Sp": sp, "T": {t: {pt: {str(k): list(range(k // 2)) + list(range(1700, 1700 + k - k // 2)) for k in br.SIZES} for pt in br.POSITION_TYPES} for t in pm.TEMPLATE_ORDER},
             "E": lists_E_e, "G": lists_G_e, "Sp1": [1987], "E1": {fid: {pos: [0] for pos in v} for fid, v in lists_E_e.items()}, "L": {str(k): list(range(k)) for k in br.SIZES}, "R": br.random_controls()}
    lock = {"selectors": {"lists": lists}, "frame_meta": meta_exposed}
    stage1 = {"frames": {fid: {"template_id": m["template"], "valid": True, "template_defined": True, "p_c": m["p_c"], "p_t": m["p_t"], "cue_final": m["cue_final"]} for fid, m in meta_fresh.items()},
              "frame_selectors": {fid: {"E": lists_E_n[fid], "G": lists_G_n[fid], "E1": {pos: [0] for pos in lists_E_n[fid]}} for fid in meta_fresh}}
    return lock, stage1


class _Confirmation:
    def __init__(self, words):
        self.tokens = [{"word": w, "category": "adjective", "token_id": 100 + i} for i, w in enumerate(words)]


def _score(alphas=None, frame_alphas_fresh=None, frame_alphas_exposed=None, fresh_counts=(6, 6, 6), valid=None):
    pairs_e, words, meta_e, le, ge, ol_e = _synthetic_pairs((3, 3, 3), prefix="e", alphas=alphas, frame_alphas=frame_alphas_exposed)
    pairs_n, _, meta_n, ln, gn, ol_n = _synthetic_pairs(fresh_counts, prefix="n", alphas=alphas, frame_alphas=frame_alphas_fresh)
    lock, stage1 = _lock_and_stage1(meta_e, le, ge, meta_n, ln, gn)
    if valid is not None:
        for fid in stage1["frames"]:
            stage1["frames"][fid]["valid"] = fid in valid
        pairs_n = {k: v for k, v in pairs_n.items() if k.split("|")[1] in valid}
    return br.score_confirmation(stage1, pairs_e, pairs_n, _Confirmation(words), lock, {"Y1": ol_e, "Y2": ol_n})


def test_routing_statistics_and_every_label_branch_on_synthetic_tables():
    results = _score()
    y1, y2, y3, y4, y5 = results["Y1"], results["Y2"], results["Y3"], results["Y4"], results["Y5"]
    kap = y1["statistics"]["pairs"]["kappa"]["c_L"]
    assert kap["E64"] == pytest.approx(K(0.7), abs=1e-9) and kap["Sp64"] == pytest.approx(K(0.5), abs=1e-9) and kap["Os64"] == pytest.approx(K(0.85), abs=1e-9) and kap["S2048"] == pytest.approx(1.0) and kap["S0"] == pytest.approx(0.0)
    g = y1["statistics"]["gains"]["64"]
    assert g["E64"] == pytest.approx(K(0.7) - K(0.5), abs=1e-9) and g["G64"] == pytest.approx(K(0.65) - K(0.5), abs=1e-9) and g["Os64"] == pytest.approx(K(0.85) - K(0.5), abs=1e-9) and g["E1"] == pytest.approx(K(0.25) - K(0.2), abs=1e-9)
    assert y1["statistics"]["rho"]["64"] == pytest.approx((K(0.65) - K(0.5)) / (K(0.7) - K(0.5)), abs=1e-9)
    assert y1["precondition"]["ok"] and y2["precondition"]["ok"] and y1["test"]["label"] == "ROUTING_PREDICTED_TOKENS" and y2["test"]["label"] == "ROUTING_PREDICTED_FRAMES_CONDITIONAL"
    assert y1["test"]["wins"]["wins"] == 9 and y1["test"]["wins"]["n_frames"] == 9 and y2["test"]["wins"]["wins"] == 18
    assert y1["test"]["headroom_witness"] == pytest.approx(K(0.85) - K(0.5), abs=1e-9) and y1["test"]["headroom_ranking_proxy"] == pytest.approx(K(0.75) - K(0.5), abs=1e-9) and y1["test"]["witness_share"] == pytest.approx((K(0.7) - K(0.5)) / (K(0.85) - K(0.5)))
    assert y3["label"] == "OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT" and y3["rho"]["pooled"] == pytest.approx(y3["rho"]["Y1"], abs=1e-6) and set(y3["denominators"]) == {"Y1", "Y2", "pooled"}
    assert y4["label"] == "TEMPLATE_FAMILY_INSUFFICIENT" and y4["values"]["Y2"]["A"] == pytest.approx(K(0.7) - K(0.55), abs=1e-9) and y4["values"]["pooled"]["B"] == pytest.approx(K(0.55) - K(0.5), abs=1e-9)
    assert y5["label"] == "MEMBERSHIP_PREDICTED" and y5["per_set"]["Y1"]["E"] == pytest.approx(56 / 64) and y5["per_set"]["Y1"]["Sp"] == pytest.approx(43 / 64) and y5["per_set"]["Y2"]["G"] == pytest.approx(32 / 64)
    assert results["outcome"]["label"] == "ROUTING_PREDICTED_TOKENS | ROUTING_PREDICTED_FRAMES_CONDITIONAL | OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT | TEMPLATE_FAMILY_INSUFFICIENT | MEMBERSHIP_PREDICTED"
    d = results["descriptive"]["Y1"]
    assert d["i_other_sizes"]["256"]["holds"] and d["iii_random"]["64"]["below_max"] and d["iii_random"]["64"]["margin_holds"] and d["x_witness"]["64"]["kappa_witness"] == pytest.approx(K(0.85), abs=1e-9) and "pooled" in results["descriptive"]
    assert results["terminology"] == br.WITNESS_TERMINOLOGY
    # A negative E with the witness headroom present → NOT_PREDICTED; with no witness headroom and E's gain below the floor → NOT_EVALUABLE.
    weak = _score(alphas={"E16": 0.5, "E64": 0.52, "E256": 0.72})
    assert weak["Y1"]["test"]["label"] == "ROUTING_NOT_PREDICTED_TOKENS" and "gain" in weak["Y1"]["test"]["failed"] and weak["Y1"]["test"]["headroom_present"]
    none = _score(alphas={"E16": 0.5, "E64": 0.52, "E256": 0.72, "Os64": 0.53})
    assert none["Y1"]["test"]["label"] == "ROUTING_NOT_EVALUABLE_TOKENS" and none["Y2"]["test"]["label"] == "ROUTING_NOT_EVALUABLE_FRAMES_CONDITIONAL" and not none["Y1"]["test"]["headroom_present"]
    assert none["Y3"]["label"] == "OPERATING_POINT_PLUS_DRIVE_RULE_NOT_EVALUABLE" and set(none["Y3"]["not_evaluable"]) == {"Y1", "Y2", "pooled"}
    # A guard failure with E's gain at the floor and no witness headroom stays NOT_PREDICTED (E itself demonstrates the headroom); a positive E with no witness headroom stays positive.
    fresh_ids = [f"n-{t}-{i}" for t in pm.TEMPLATE_ORDER for i in range(6)]
    half = {**{(fid, "E64"): 0.49 for fid in fresh_ids[:9]}, **{(fid, "E64"): 0.9 for fid in fresh_ids[9:]}}  # exactly half of the 18 fresh frames below S'_64 (the guard needs strictly more than half) while the pooled gain stays above the floor
    guard = _score(alphas={"Os64": 0.53}, frame_alphas_fresh=half)
    assert guard["Y2"]["test"]["wins"]["wins"] == 9 and guard["Y2"]["test"]["wins"]["n_frames"] == 18 and not guard["Y2"]["test"]["conditions"]["frame_count"]
    assert guard["Y2"]["test"]["label"] == "ROUTING_NOT_PREDICTED_FRAMES_CONDITIONAL" and guard["Y2"]["test"]["gain"] >= br.ROUTING_GAIN_FLOOR and guard["Y2"]["test"]["headroom_witness"] < br.HEADROOM_MIN
    assert guard["Y1"]["test"]["label"] == "ROUTING_PREDICTED_TOKENS" and guard["Y1"]["test"]["headroom_witness"] < br.HEADROOM_MIN
    ten = {(fid, "E64"): 0.45 for fid in fresh_ids[:8]}
    assert _score(frame_alphas_fresh=ten)["Y2"]["test"]["wins"]["wins"] == 10 and _score(frame_alphas_fresh=ten)["Y2"]["test"]["conditions"]["frame_count"]
    # The split guard: the coordinated family below the floor.
    coord = {(fid, "E64"): 0.5 for fid in fresh_ids if "coordinated" in fid}
    split = _score(frame_alphas_fresh=coord)
    assert not split["Y2"]["test"]["conditions"]["split_coordinated"] and split["Y2"]["test"]["label"] == "ROUTING_NOT_PREDICTED_FRAMES_CONDITIONAL" and split["Y2"]["test"]["split_gains"]["coordinated"] == pytest.approx(0.0, abs=1e-9)
    # Y3: ρ at exactly the pooled floor passes; a set below 0.50 fails even when pooled passes; a denominator below 0.05 is not evaluable and names the set.
    def rho_for(g_alpha):
        return (K(g_alpha) - K(0.5)) / (K(0.7) - K(0.5))
    import math
    target = K(0.5) + 0.60 * (K(0.7) - K(0.5))
    alpha_60 = 1.0 - math.sqrt(1.0 - target * K_REF)
    above = _score(alphas={"G64": alpha_60 + 0.01})
    assert above["Y3"]["rho"]["pooled"] > 0.60 and above["Y3"]["label"] == "OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT"
    below = _score(alphas={"G64": alpha_60 - 0.01})
    assert below["Y3"]["rho"]["pooled"] < 0.60 and below["Y3"]["label"] == "OPERATING_POINT_PLUS_DRIVE_RULE_INSUFFICIENT" and "pooled" in below["Y3"]["failed"]
    # The exact boundaries, on hand-made statistics: ρ at exactly 0.60 pooled and exactly 0.50 on a set pass; a denominator at exactly 0.05 is evaluable, below it is not.
    stat = lambda e, g, r: {"gains": {"64": {"E64": e, "G64": g}}, "rho": {"64": r}}  # noqa: E731
    boundary = br.rule_test({"Y1": stat(0.1, 0.05, 0.5), "Y2": stat(0.1, 0.05, 0.5)}, stat(0.1, 0.06, 0.6))
    assert boundary["label"] == "OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT" and boundary["failed"] == []
    assert br.rule_test({"Y1": stat(0.1, 0.05, 0.5), "Y2": stat(0.1, 0.049, 0.49)}, stat(0.1, 0.06, 0.6))["failed"] == ["Y2"]
    assert br.rule_test({"Y1": stat(0.05, 0.03, 0.6), "Y2": stat(0.1, 0.06, 0.6)}, stat(0.1, 0.06, 0.6))["evaluable"]
    assert br.rule_test({"Y1": stat(0.049, 0.03, 0.6), "Y2": stat(0.1, 0.06, 0.6)}, stat(0.1, 0.06, 0.6))["not_evaluable"] == ["Y1"]
    per_set = _score(frame_alphas_fresh={(fid, "G64"): 0.52 for fid in fresh_ids})
    assert per_set["Y3"]["rho"]["Y2"] < 0.5 <= per_set["Y3"]["rho"]["Y1"] and per_set["Y3"]["label"] == "OPERATING_POINT_PLUS_DRIVE_RULE_INSUFFICIENT" and "Y2" in per_set["Y3"]["failed"] and "Y1" not in per_set["Y3"]["failed"]
    denominator = _score(frame_alphas_fresh={(fid, "E64"): 0.52 for fid in fresh_ids})
    assert denominator["Y3"]["label"] == "OPERATING_POINT_PLUS_DRIVE_RULE_NOT_EVALUABLE" and denominator["Y3"]["not_evaluable"] == ["Y2"]
    # Y4: sufficient (T carries the routing, E adds < 0.05), set-dependent, and neither.
    sufficient = _score(alphas={"T64": 0.68, "E64": 0.7})
    assert sufficient["Y4"]["label"] == "TEMPLATE_FAMILY_SUFFICIENT"
    tstat = lambda e, t, sp: {"pairs": {"kappa": {"c_L": {"E64": e, "T64": t, "Sp64": sp}}}, "gains": {"64": {"T64": t - sp, "E64": e - sp}}}  # noqa: E731
    dependent = br.template_test({"Y1": tstat(0.9, 0.85, 0.7), "Y2": tstat(0.9, 0.8, 0.7)}, tstat(0.9, 0.87, 0.7), "Y2", True)
    assert dependent["label"] == "TEMPLATE_FAMILY_NOT_DISTINGUISHED" and "set-dependent" in dependent["reason"] and dependent["insufficient_holds"] == {"Y2": True, "pooled": False}
    assert br.template_test({"Y1": tstat(0.9, 0.85, 0.7), "Y2": tstat(0.9, 0.86, 0.7)}, tstat(0.9, 0.87, 0.7), "Y2", True)["label"] == "TEMPLATE_FAMILY_SUFFICIENT"
    assert br.template_test({"Y1": tstat(0.9, 0.85, 0.7), "Y2": tstat(0.9, 0.86, 0.7)}, tstat(0.9, 0.87, 0.7), "Y2", False)["label"] == "TEMPLATE_FAMILY_NOT_EVALUABLE"
    neither = _score(alphas={"T64": 0.5, "E64": 0.53})
    assert neither["Y4"]["label"] == "TEMPLATE_FAMILY_NOT_DISTINGUISHED" and "neither" in neither["Y4"]["reason"]
    # Y5 fails on one set when its overlap falls; the label names the set.
    pairs_e, words, meta_e, le, ge, ol_e = _synthetic_pairs((3, 3, 3), prefix="e")
    pairs_n, _, meta_n, ln, gn, ol_n = _synthetic_pairs((6, 6, 6), prefix="n")
    lock, stage1 = _lock_and_stage1(meta_e, le, ge, meta_n, ln, gn)
    for fid in ol_n:
        for pos in ol_n[fid]:
            ol_n[fid][pos]["64"] = list(range(1800, 1864))
    y5 = br.score_confirmation(stage1, pairs_e, pairs_n, _Confirmation(words), lock, {"Y1": ol_e, "Y2": ol_n})["Y5"]
    assert y5["label"] == "MEMBERSHIP_NOT_PREDICTED" and y5["failed"] == ["Y2"] and y5["per_set"]["Y2"]["E"] == 0.0
    # Preconditions: too few valid fresh frames; an unscored token; a non-evaluable gap.
    few = _score(valid=[f"n-cardinal-{i}" for i in range(6)] + [f"n-quantifier-{i}" for i in range(2)] + [f"n-coordinated-adjective-{i}" for i in range(4)])
    assert few["Y2"]["test"]["label"] == "PRECONDITION_FAILED_FRAMES" and "valid_fresh_frames" in few["Y2"]["precondition"]["failed"] and few["Y3"]["label"] == "OPERATING_POINT_PLUS_DRIVE_RULE_NOT_EVALUABLE" and few["Y4"]["label"] == "TEMPLATE_FAMILY_NOT_EVALUABLE" and few["Y5"]["label"] == "MEMBERSHIP_NOT_EVALUABLE"
    gap = _score(alphas={br.TEMPLATE_BASE: 0.89})
    assert gap["Y1"]["test"]["label"] == "PRECONDITION_FAILED_TOKENS" and "gap_c_L" in gap["Y1"]["precondition"]["failed"]
    # κ and ρ are unclipped.
    assert br.rho(-0.02, 0.1) == pytest.approx(-0.2) and br.rho(0.2, 0.1) == pytest.approx(2.0) and br.rho(0.02, 0.04) is None and br.rho(None, 0.1) is None
    assert br.frame_wins({"a": {"r2": {"E64": 0.5, "Sp64": 0.5}}, "b": {"r2": {"E64": 0.6, "Sp64": 0.5}}}, "E64", "Sp64") == {"n_frames": 2, "wins": 1, "losses": 0, "ties": 1, "undefined": 0, "winning_frames": ["b"], "holds": False}


def test_stage_one_digest_and_target_pair_guards():
    stage1 = {"rows": [{"a": 1}], "states": {"f": {"p_c": 3}}, "frame_selectors": {"f": {"E": {}}}}
    stage1["digest"] = br.stage_digest(stage1["rows"], stage1["states"], stage1["frame_selectors"])
    br.assert_stage_one_digest(stage1)
    with pytest.raises(br.PhaseError):
        br.assert_stage_one_digest({**stage1, "rows": [{"a": 2}]})
    with pytest.raises(br.PhaseError, match="measured fresh-cue quantity"):
        br.assert_stage_one_digest({**stage1, "effects": {}})
    with pytest.raises(br.PhaseError):
        br.assert_stage_one_digest({})


@pytest.fixture(scope="module")
def inputs(inputs_018):
    manifest, digests, pool_018, confirmation_017 = inputs_018
    digests = dict(digests)
    confirmation_018 = bc.load_confirmation(ROOT / bc.CONFIRMATION_RELATIVE_PATH, pool_018, digests)
    digests["confirmation_018"] = confirmation_018.content_sha256
    lock_018 = json.loads((ROOT / bc.LOCK_RELATIVE_PATH).read_text())
    digests["lock_018"] = lock_018["content_sha256"]
    manifest_obj, _, extension = pm.load_inputs(ROOT)
    pool = br.build_pool_019(*_confirmations(manifest_obj, extension, pool_018, confirmation_017, confirmation_018))
    return manifest, digests, pool, confirmation_018, lock_018


def _confirmations(manifest, extension, pool_018, confirmation_017, confirmation_018):
    from neural_decompiler import cue_decompilation as cd
    from neural_decompiler import encoding_read as er
    from neural_decompiler import frame_channels as fch
    from neural_decompiler import head_transport as ht
    from neural_decompiler import neuron_feature as nf
    manifest, manifest_sha256, extension = pm.load_inputs(ROOT)
    c006 = cd.load_confirmation(ROOT / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    c009 = ht.load_confirmation(ROOT / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, c006)
    c011 = er.load_confirmation(ROOT / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, c006, c009)
    digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": c006.content_sha256, "confirmation_009": c009.content_sha256, "confirmation_011": c011.content_sha256}
    c012 = lc.load_confirmation(ROOT / lc.CONFIRMATION_RELATIVE_PATH, lc.build_pool_012(manifest, extension, c006, c009, c011), digests)
    digests["confirmation_012"] = c012.content_sha256
    digests["lock_012"] = json.loads((ROOT / lc.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    c013 = ap.load_confirmation(ROOT / ap.CONFIRMATION_RELATIVE_PATH, ap.build_pool_013(manifest, extension, c006, c009, c011, c012), digests)
    digests["confirmation_013"] = c013.content_sha256
    digests["lock_013"] = json.loads((ROOT / ap.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    c014 = nf.load_confirmation(ROOT / nf.CONFIRMATION_RELATIVE_PATH, nf.build_pool_014(manifest, extension, c006, c009, c011, c012, c013), digests)
    digests["confirmation_014"] = c014.content_sha256
    digests["lock_014"] = json.loads((ROOT / nf.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    c015 = atp.load_confirmation(ROOT / atp.CONFIRMATION_RELATIVE_PATH, atp.build_pool_015(manifest, extension, c006, c009, c011, c012, c013, c014), digests)
    digests["confirmation_015"] = c015.content_sha256
    digests["lock_015"] = json.loads((ROOT / atp.LOCK_RELATIVE_PATH).read_text())["content_sha256"]
    c016 = fch.load_confirmation(ROOT / fch.CONFIRMATION_RELATIVE_PATH, fch.build_pool_016(manifest, extension, c006, c009, c011, c012, c013, c014, c015), digests)
    return manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016, confirmation_017, confirmation_018


def test_pool_019_and_the_committed_reproducible_extract(inputs):
    manifest, digests, pool, confirmation_018, lock_018 = inputs
    assert len(pool.tokens) == 255 and len(pool.frames) == 90 and pool.token_source["fancy"] == "confirmation-018" and pool.frame_origin["quantifier-018-4"] == "confirmation-018"
    assert sum(1 for frame in pool.frames if frame.p_t == frame.p_c) == 60 and all(frame.p_t == frame.p_c + 1 for frame in pool.frames_of("coordinated-adjective"))
    extract = br.load_inherited_extract_018(ROOT / br.INHERITED_018_EXTRACT_RELATIVE_PATH, digests=digests, expected_size=br.EXPECTED_EXTRACT_SIZE_018)
    keys = set(extract["entries"])
    assert "fancy|cardinal-009-1" in keys and "fancy|quantifier-018-4" in keys and "abundant|cardinal-009-1" in keys and len(keys) == 11796
    assert extract["entries"]["fancy|quantifier-018-4"]["set"] == "Y2" and extract["entries"]["fancy|cardinal-009-1"]["set"] == "Y1" and extract["entries"]["abundant|cardinal-009-1"]["set"] == "explore"
    assert set(extract["stage1_state_digests"]) == {f"{t}-018-{i}" for t in pm.TEMPLATE_ORDER for i in (1, 2, 3, 4)} and len(extract["frame_subsets_explore"]) == 78 and len(extract["frame_subsets_stage1"]) == 12
    assert set(extract["source"]) == set(br.EXTRACT_SOURCE_FIELDS) and extract["source"]["lock_content_sha256"] == lock_018["content_sha256"] and extract["source"]["run_id"] == "86ccb888f52600ef" and extract["source"]["extraction_schema_version"] == 1
    assert extract["frame_subsets_explore"] == {fid: [int(i) for i in v] for fid, v in lock_018["frame_subsets"].items()}
    with pytest.raises(ValueError):
        br.load_inherited_extract_018(ROOT / br.INHERITED_018_EXTRACT_RELATIVE_PATH, digests=digests, expected_size=10)
    with pytest.raises(ValueError):
        br.load_inherited_extract_018(ROOT / br.INHERITED_018_EXTRACT_RELATIVE_PATH, digests={**digests, "lock_018": "0" * 64}, expected_size=br.EXPECTED_EXTRACT_SIZE_018)
    # The builder is deterministic: the same source gives the same content digest; a different source digest changes it and a tampered entry breaks the loader.
    fake_state = {"phases": {"confirm": {"status": "complete", "confirm_commit": "c" * 40}}, "lock": {"content_sha256": "L" * 64}, "state_sha256": "S" * 64, "run_id": "r", "protocol_code_commit": "e" * 40,
                  "exploration": {"pairs": {"w|f": {"F": 1.0, "Pi": 0.5, "dT": 1.5, "c_L": 0.2, "row": [0.1, -0.1], "prediction": {f"{o}_{r}": 0.1 for o in ("c_L", "F", "Pi", "dT") for r in ("S0", "S256", "S2048")} | {"row_S0": [0.0, 0.0], "row_S2048": [0.1, -0.1]}}}},
                  "confirmation": {"per_frame_exposed": {}, "per_frame_fresh": {}, "stage1": {"state_digests": {"n": "d"}, "frame_subsets": {"n": list(range(256))}}}}
    fake_lock = {"content_sha256": "L" * 64, "frame_subsets": {"f": list(range(256))}}
    fake_digests = {key: "0" * 64 for key in br.EXTRACT_018_DIGEST_KEYS}
    a = br.build_inherited_extract_018(fake_state, fake_lock, digests=fake_digests, results_state_path="p", results_state_file_sha256="f" * 64, lock_path="q", lock_file_sha256="g" * 64)
    b = br.build_inherited_extract_018(json.loads(json.dumps(fake_state)), fake_lock, digests=fake_digests, results_state_path="p", results_state_file_sha256="f" * 64, lock_path="q", lock_file_sha256="g" * 64)
    assert a["content_sha256"] == b["content_sha256"] and a["source"]["results_state_sha256"] == "S" * 64 and a["source"]["confirm_commit"] == "c" * 40
    c = br.build_inherited_extract_018(fake_state, fake_lock, digests=fake_digests, results_state_path="p", results_state_file_sha256="h" * 64, lock_path="q", lock_file_sha256="g" * 64)
    assert c["content_sha256"] != a["content_sha256"]
    with pytest.raises(ValueError, match="not the lock"):
        br.build_inherited_extract_018(fake_state, {**fake_lock, "content_sha256": "M" * 64}, digests=fake_digests, results_state_path="p", results_state_file_sha256="f" * 64, lock_path="q", lock_file_sha256="g" * 64)


def toy_tokenizer_019(manifest, pool):
    base = toy_tokenizer_018(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for words in br.CANDIDATES.values() for word in words]
    for _, text in br.FRESH_FRAMES:
        for index, piece in enumerate(text.replace("{cue}", "").split()):
            words.append(piece if index == 0 else " " + piece)
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    for word, token_id in pool.tokens:  # exposed words carry their real ids so that the builder skips them
        vocabulary[" " + word] = token_id
    known = set(vocabulary.values())
    for frame in pool.frames:
        for token_id in (*frame.prefix_ids, *frame.suffix_ids, *frame.cue_ids.values()):
            if token_id not in known:
                vocabulary[f"⟨{token_id}⟩"] = token_id
                known.add(token_id)
    return type(base)(vocabulary)


def test_confirmation_policy_with_eighteen_frames_and_the_prompt_manifest(inputs, tmp_path):
    manifest, digests, pool, confirmation_018, lock_018 = inputs
    tokenizer = toy_tokenizer_019(manifest, pool)
    path = tmp_path / "confirmation-v1.json"
    digest = br.freeze_confirmation(path, tokenizer, pool, digests)
    confirmation = br.load_confirmation(path, pool, digests)
    assert confirmation.content_sha256 == digest and len(confirmation.frames) == 18 and len(confirmation.tokens) == 24 and [t for t, _ in br.FRESH_FRAMES] == [frame.template_id for frame in confirmation.frames]
    words = [token["word"] for token in confirmation.tokens]
    assert words[:5] == ["separate", "remaining", "similar", "different", "prior"] and words[5:9] == ["double", "triple", "couple", "pair"] and words[9:14] == ["generous", "overall", "negligible", "maximum", "minimum"]
    assert words[14:18] == ["she", "he", "they", "we"] and words[18:] == ["rotten", "faded", "stolen", "gentle", "giant", "silver"]
    exposed_ids = {token_id for _, token_id in pool.tokens}
    assert not ({token["token_id"] for token in confirmation.tokens} & exposed_ids) and all(frame.text_template not in {f.text_template for f in pool.frames} for frame in confirmation.frames)
    assert len(confirmation.token_prompts) == 24 * 18 and len(confirmation.exposed_frame_prompts) == 24 * 90 and confirmation.frames[0].frame_id == "cardinal-019-1" and confirmation.frames[17].p_t == confirmation.frames[17].p_c + 1
    payload = json.loads(path.read_text())
    manifest_keys = payload["prompt_key_manifest"]
    assert len(manifest_keys) == len(set(manifest_keys)) and len(manifest_keys) == 24 * 18 + 24 * 90 + 18 + 36 and manifest_keys == br.prompt_key_manifest(confirmation)
    with pytest.raises(FileExistsError):
        br.freeze_confirmation(path, tokenizer, pool, digests)
    tampered = dict(payload)
    tampered["prompt_key_manifest"] = manifest_keys[:-1]
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in tampered.items() if key != "content_sha256"}))
    with pytest.raises(ValueError, match="prompt-key manifest"):
        br.validate_confirmation(tampered, pool, digests)
    tampered = dict(payload)
    tampered["tokens"] = [dict(t, word="bogus") if t["word"] == "prior" else t for t in payload["tokens"]]
    with pytest.raises(ValueError):
        br.validate_confirmation(tampered, pool, digests)


def test_chain_selectors_oracles_and_the_boundary_on_the_fake(inputs, monkeypatch):
    manifest, digests, pool, confirmation_018, lock_018 = inputs
    monkeypatch.setattr(__import__("neural_decompiler.cue_suppression", fromlist=["cs"]), "STAGE_UNINFORMATIVE_FLOOR", 0.0)
    model, weights, head, lw, programs, small, axes, plural_ids, read, states, bases, bases_3, base2_pt, n_pt, hcm, hp_context = _fake_setup(pool)
    read_out = bc.read_of_outputs(read, lw)
    chain = br.RoutingChain(bc.MaskedChainModel(hcm, base2_pt, {}))
    locked = {fid: hp.locked_state(s) for fid, s in states.items()}
    frames = list(small.frames)
    tokens = [(n, t) for n, t in small.tokens]
    tokens_by_template = {template: [(n, t) for n, t in tokens if t not in (pool.reference_ids[template], plural_ids[template])] for template in pm.TEMPLATE_ORDER}
    licensed = {f"{n}|{f.frame_id}" for f in frames for n, t in tokens_by_template[f.template_id]}
    inherited = {f"S{k}": list(range(k)) for k in (16, 64, 256)}
    context = br.AnalysisContext(chain, hp_context, weights, axes["T"], read_out)
    # The selectors from the locked states, the weights and the exposed token ids only: no capture entry point is reachable, and the lists follow from the scores by the frozen rule.
    with pytest.MonkeyPatch.context() as guard:
        for attr in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
            if hasattr(pm, attr):
                guard.setattr(pm, attr, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a selector touched the network")))
        selectors = br.selectors_from_states(chain, weights, read_out, locked, frames, tokens_by_template, licensed, inherited)
    lists = selectors["lists"]
    assert set(lists) == {"Sp", "T", "E", "G", "Sp1", "E1", "L", "R"} and set(lists["Sp"]) == {"p_c", "p_t"} and set(lists["E"]) == set(locked) and len(lists["Sp"]["p_c"]["64"]) == 64 and lists["L"]["64"] == list(range(64))
    assert set(selectors["drive_quantiles"]) == set(pm.TEMPLATE_ORDER) and set(selectors["drive_quantiles"]["coordinated-adjective"]) == {"p_c", "p_t"} and set(selectors["drive_quantiles"]["cardinal"]) == {"p_c"}
    assert all(len(v) == br.N_NEURONS for t in selectors["drive_quantiles"].values() for pt in t.values() for v in pt.values()) and selectors["n_records"]["licensed"] == len(licensed)
    br.check_selector_lists(selectors, inherited)
    bad = json.loads(json.dumps(selectors))
    bad["lists"]["E"][frames[0].frame_id][str(states[frames[0].frame_id].p_c)]["64"][0] = 2047
    with pytest.raises(br.PhaseError):
        br.check_selector_lists(bad, inherited)
    assert selectors["digest"] == br.selectors_digest(selectors) and br.selectors_digest(bad) != selectors["digest"]
    again = br.selectors_from_states(chain, weights, read_out, locked, frames, tokens_by_template, licensed, inherited)
    assert again["digest"] == selectors["digest"]
    # I9 on a cue-final frame: the E rule with Experiment 018's accumulator (positions pooled) is Experiment 018's frame ranking.
    frame = frames[0]
    replication = {"frame_ids": {f.frame_id for f in frames}, "stage1_frame_ids": set(), "token_ids_by_template": {t: {token_id for _, token_id in v} for t, v in tokens_by_template.items()}, "licensed_keys": licensed}
    with_rep = br.selectors_from_states(chain, weights, read_out, locked, frames, tokens_by_template, licensed, inherited, replication=replication)
    unmasked = bc.MaskedChainModel(hcm, base2_pt, {name: torch.ones(bc.N_NEURONS, dtype=torch.float64) for name in bc.RUNGS})
    own_018 = bc.frame_ranking(unmasked, weights, read_out, locked[frame.frame_id], frame.template_id, tokens_by_template[frame.template_id])
    assert with_rep["e_018_lists"][frame.frame_id] == bc.frame_subset(own_018) and with_rep["lists"]["E"][frame.frame_id][str(frame.p_c)]["256"] == bc.frame_subset(own_018)
    # A fresh token is refused by the frame selector; the per-frame selector of a locked state equals the pooled computation's lists.
    with pytest.raises(br.PhaseError, match="fresh token"):
        br.frame_selectors_from_state(chain, weights, read_out, locked[frame.frame_id], frame.template_id, tokens_by_template[frame.template_id], selectors["drive_quantiles"], selectors["denominators"][frame.template_id], {tokens_by_template[frame.template_id][0][1]})
    own = br.frame_selectors_from_state(chain, weights, read_out, locked[frame.frame_id], frame.template_id, tokens_by_template[frame.template_id], selectors["drive_quantiles"], selectors["denominators"][frame.template_id], set())
    assert own["E"] == lists["E"][frame.frame_id] and own["G"] == lists["G"][frame.frame_id] and own["E1"] == lists["E1"][frame.frame_id]
    assert set(br.selector_overlaps(lists, own, frame.template_id, frame.p_c)[str(frame.p_c)]) == {"E&Sp", "E&T", "E&G", "G&Sp", "T&Sp"}
    # The chain: I8 against Experiment 017's Level 0, the all-ones per-position mask equals 018's reference rung and the zero mask its S_0; I10; the measured effects; the oracles on the same upstream parts.
    for frame in (frames[0], small.frames_of("coordinated-adjective")[0]):
        state = states[frame.frame_id]
        template = frame.template_id
        name, token_id = tokens_by_template[template][0]
        masks = br.frame_rung_masks(lists, frame.frame_id, template, state.p_c, state.p_t)
        assert set(masks) == set(br.PROSPECTIVE_RUNGS) and all(set(m) == set(state.positions) for m in masks.values())
        plural = br.measure_pair(model, weights, head, state, pool.plural_cue[template], plural_ids[template], axes["R0"], axes["T"], small.single_nouns)
        record = br.measure_pair(model, weights, head, state, name, token_id, axes["R0"], axes["T"], small.single_nouns)
        assert pm.site_label((f"RESID_PRE.L{br.BLOCK}", state.p_t)) in record.extra and pm.site_label((f"RESID_PRE.L{br.BLOCK}", state.p_c)) in record.extra
        out = br.analyse_pair_019(record, plural, context=context, state=state, rung_masks=masks)
        assert out is not None
        analysis, p = out["analysis"], out["analysis"]["prediction"]
        ids = analysis["identities"]
        assert ids["I8_reference_rung"] < 1e-9 and ids["I10_read_identity"] < 1e-9 and ids["I4_x3"] < 1e-5 and ids["I7_split"] < 1e-5
        masked_018 = bc.MaskedChainModel(hcm, base2_pt, bc.masks_from_subsets(bc.subsets_from_scores(torch.rand(bc.N_NEURONS, generator=torch.Generator().manual_seed(1), dtype=torch.float64)), bc.N_NEURONS))
        entry_018 = masked_018.predict_from_state(weights, state, token_id, template)
        assert p["c_L_S2048"] == pytest.approx(entry_018["c_L_S2048"], abs=1e-12) and p["dT_S2048"] == pytest.approx(entry_018["dT_S2048"], abs=1e-12) and p["c_L_S0"] == pytest.approx(entry_018["c_L_S0"], abs=1e-12) and p["dT_S0"] == pytest.approx(entry_018["dT_S0"], abs=1e-12)
        assert p["row_S0"] == pytest.approx(entry_018["row_S0"], abs=1e-12) and p["dT_frozen"] == p["F_S2048"] and p["c_L_level0F"] == pytest.approx(entry_018["c_L_level0F"])
        for rung in br.PROSPECTIVE_RUNGS:
            assert len(p[f"row_{rung}"]) == state.p_t + 1 and sum(p[f"row_{rung}"]) == pytest.approx(0.0, abs=1e-9) and p[f"dT_{rung}"] == pytest.approx(p[f"F_{rung}"] + p[f"Pi_{rung}"], abs=1e-9)
        assert p["c_L_E64"] != pytest.approx(p["c_L_S0"], abs=1e-12) and p["c_L_Sp1"] != pytest.approx(p["c_L_S0"], abs=1e-12) and set(analysis["effect_fidelity"]) == {str(pos) for pos in state.positions}
        assert all(e.shape == (br.N_NEURONS,) for e in out["effects"].values()) and out["u"].shape == (br.N_NEURONS,) and abs(br.apply_mask_read(p["c_L_S0"], out["u"], lists["E"][frame.frame_id][str(state.p_c)]["64"]) - p["c_L_E64"]) < 1e-9
        # The boundary: a poisoned patched capture at both positions leaves every prospective column unchanged; the identities notice; the measured effects change.
        poisoned = {key: value * 3.0 + 1.0 for key, value in record.extra.items()}
        fields = {f.name: getattr(record, f.name) for f in record.__dataclass_fields__.values()}
        for key, value in list(fields.items()):
            if isinstance(value, float):
                fields[key] = value * 3.0 + 1.0
            elif isinstance(value, dict) and value and all(isinstance(v, float) for v in value.values()):
                fields[key] = {k: v * 3.0 + 1.0 for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                fields[key] = value * 3.0 + 1.0
        fields["extra"] = poisoned
        poisoned_record = ra.Attribution(**fields)
        with pytest.raises(pm.IncidentError):
            br.analyse_pair_019(poisoned_record, plural, context=context, state=state, rung_masks=masks)
        with pytest.MonkeyPatch.context() as guard:
            for module, names in ((hp, ("X3_IDENTITY_TOLERANCE", "ROW_IDENTITY_TOLERANCE", "DT_IDENTITY_TOLERANCE", "SPLIT_IDENTITY_TOLERANCE")), (atp, ("ROW_IDENTITY_TOLERANCE", "CHAIN_IDENTITY_TOLERANCE")),
                                  (ap, ("OV_IDENTITY_TOLERANCE",)), (lc, ("LADDER_IDENTITY_TOLERANCE", "NEURON_IDENTITY_TOLERANCE")), (ra, ("IDENTITY_TOLERANCE", "P1_CROSS_CHECK_TOLERANCE", "NEURON_SUM_TOLERANCE"))):
                for attr in names:
                    guard.setattr(module, attr, float("inf"))
            poisoned_out = br.analyse_pair_019(poisoned_record, plural, context=context, state=state, rung_masks=masks)
        assert poisoned_out is not None and atp._max_numeric_difference(poisoned_out["analysis"]["prediction"], p, "poisoned") == 0.0 and poisoned_out["analysis"]["dT"] != analysis["dT"]
        assert not torch.allclose(poisoned_out["effects"][state.p_c], out["effects"][state.p_c]) and torch.equal(poisoned_out["u"], out["u"])
        # The oracles: the ranking oracle from measured effects and the witness from the OTHER cues; the oracle rungs differ from the prospective rungs only in the mask they name.
        pairs, effects, contributions = {}, {}, {}
        for other_name, other_id in tokens_by_template[template][:4]:
            rec = br.measure_pair(model, weights, head, state, other_name, other_id, axes["R0"], axes["T"], small.single_nouns)
            res = br.analyse_pair_019(rec, plural, context=context, state=state, rung_masks=masks)
            key = f"{other_name}|{frame.frame_id}"
            pairs[key], effects[key], contributions[key] = res["analysis"], res["effects"], res["u"]
        oracle_lists, witness_lists = br.oracles_for_set(pairs, effects, contributions, [n for n, _ in tokens_by_template[template][:4]], read_out.abs(), selectors["denominators"])
        assert set(oracle_lists) == {frame.frame_id} and set(oracle_lists[frame.frame_id]) == {str(pos) for pos in state.positions} and set(witness_lists) == set(pairs)
        key = next(iter(pairs))
        assert all(len(witness_lists[key][str(k)]) == k for k in br.SIZES)
        # The held-out pair's own contribution vector AND its own measured read cannot change its witness mask.
        altered_contributions = dict(contributions)
        altered_contributions[key] = contributions[key] * -7.0 + 1.0
        altered_pairs = json.loads(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "prediction"} | {"prediction": v["prediction"]} for k, v in pairs.items()}))
        altered_pairs[key]["c_L"] = pairs[key]["c_L"] + 123.0
        _, altered_witness = br.oracles_for_set(altered_pairs, effects, altered_contributions, [n for n, _ in tokens_by_template[template][:4]], read_out.abs(), selectors["denominators"])
        assert altered_witness[key] == witness_lists[key]
        other = next(k for k in pairs if k != key)
        assert altered_witness[other] != witness_lists[other] or True  # the other pairs' masks may change (their training set changed); the held-out one must not
        oracle_entry = br.oracle_rungs_for_frame(chain, weights, state, token_id, template, oracle_lists[frame.frame_id], witness_lists[key])
        assert set(oracle_entry) == set(br.ORACLE_SCALAR_KEYS) | set(br.ORACLE_ROW_KEYS)
        expected = br.apply_mask_read(p["c_L_S0"], out["u"], witness_lists[key]["64"])
        assert oracle_entry["c_L_Os64"] == pytest.approx(expected, abs=1e-9)
        with pytest.MonkeyPatch.context() as guard:
            for attr in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
                if hasattr(pm, attr):
                    guard.setattr(pm, attr, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a prediction touched the network")))
            table = br.prediction_table(chain, weights, locked, lists, frames, [{"word": name, "token_id": token_id}], list(pm.TEMPLATE_ORDER))
        mine = next(row for row in table if row["frame_id"] == frame.frame_id)
        assert len(table) == len(frames) and set(mine) == set(br.PREDICTION_COLUMNS) and atp._max_numeric_difference(mine, br.prediction_row(name, frame.frame_id, template, p), "row") < 1e-12
    # The chain from a source record reproduces the in-process chain.
    source = {"axes_vectors": {"T": axes["T"].direction.double().tolist(), "R0": axes["R0"].direction.double().tolist()}, "read_weight": read.weight.tolist(), "sigma_T": axes["T"].sigma, "defined_templates": list(pm.TEMPLATE_ORDER),
              "bases_3": hp.bases_to_json(bases_3, {t: 1 for t in bases_3}), "base2_pt": {"vector": base2_pt.tolist(), "n_frames": n_pt, "template": "coordinated-adjective"}}
    lock_012 = {"base_states": lc.bases_to_json(bases, {template: len(small.frames_of(template)) for template in pm.TEMPLATE_ORDER}), "defined_templates": list(pm.TEMPLATE_ORDER), "axes_vectors": source["axes_vectors"], "read_weight": read.weight.tolist(), "sigma_T": axes["T"].sigma}
    rebuilt = br.chain_from_source(source, lock_012, lw, programs, pool)
    frame = frames[0]
    masks = br.frame_rung_masks(lists, frame.frame_id, frame.template_id, frame.p_c, frame.p_t)
    a = rebuilt.predict_from_locked(weights, locked[frame.frame_id], tokens[1][1], frame.template_id, masks)
    b = chain.predict_from_locked(weights, locked[frame.frame_id], tokens[1][1], frame.template_id, masks)
    assert atp._max_numeric_difference(a, b, "rebuilt") < 1e-9
    assert br.render_predictions({"run_id": "r", "protocol_code_commit": "c" * 40, "confirmation_019_sha256": "s" * 64, "lock_018_sha256": "l" * 64, "selectors": selectors, "tokens": [{"word": "w", "category": "adjective"}],
                                  "predictions": {"rows": [], "token_means": {}}}).startswith("# Experiment 019")
