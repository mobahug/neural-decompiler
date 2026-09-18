"""Experiment 007: the supervised cross-moment SVD subspace, the pseudoinverse readout, the nested ridge, LOCO tables, and the inherited check."""

from __future__ import annotations

import pytest
import torch

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M
from plural_fakes import make_context

FAKE_CIRCUIT = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L01.H00",), ("L01.MLP",))


# ---------------------------------------------------------------------------
# Pure tensor tests on planted structure (no model).


def _orthogonal(n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=generator, dtype=torch.float64))
    return q


def test_cross_moment_svd_recovers_a_planted_supervised_subspace_and_is_nested_and_deterministic():
    d = 8
    X = _orthogonal(d, 1)  # XᵀX = I, so the left singular subspace of XᵀY is the planted response-relevant subspace
    U0 = _orthogonal(d, 2)[:, :3]
    W = torch.randn(3, d, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * torch.tensor([[3.0], [2.0], [1.0]], dtype=torch.float64)
    Y = X @ U0 @ W + 1e-3 * torch.randn(d, d, generator=torch.Generator().manual_seed(4), dtype=torch.float64)  # a small full-rank perturbation keeps rank(C) ≥ 5
    svd = ss.cross_moment_svd(X, Y)
    U3 = svd.basis(3)
    projector = torch.eye(d, dtype=torch.float64) - U3 @ U3.T
    assert float((projector @ U0).abs().max()) < 1e-2
    assert torch.allclose(U3.T @ U3, torch.eye(3, dtype=torch.float64), atol=1e-10)
    assert torch.equal(svd.basis(2), svd.basis(4)[:, :2]) and torch.equal(svd.basis(1), svd.basis(3)[:, :1])
    for column in range(4):
        vector = svd.basis(4)[:, column]
        assert float(vector[vector.abs().argmax()]) > 0.0  # sign convention
    again = ss.cross_moment_svd(X, Y)
    assert torch.equal(again.P, svd.P) and again.gaps == svd.gaps and again.rank_C == svd.rank_C
    assert set(svd.gaps) == set(ss.RANKS) and all(gap >= ss.GAP_MIN for gap in svd.gaps.values())
    assert svd.rank_C >= ss.MIN_CROSS_MOMENT_RANK


def test_cross_moment_svd_raises_on_a_planted_tie_and_on_low_rank():
    d = 8
    P, Q = _orthogonal(d, 5), _orthogonal(d, 6)
    tied = P @ torch.diag(torch.tensor([3.0, 2.0, 2.0, 1.0, 0.7, 0.5, 0.3, 0.2], dtype=torch.float64)) @ Q.T
    with pytest.raises(pm.IncidentError, match="singular gap at rank 2"):
        ss.cross_moment_svd(torch.eye(d, dtype=torch.float64), tied, where="fold t")
    low = P @ torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)) @ Q.T
    with pytest.raises(pm.IncidentError, match="rank"):
        ss.cross_moment_svd(torch.eye(d, dtype=torch.float64), low)
    with pytest.raises(pm.IncidentError):
        ss.cross_moment_svd(torch.eye(d, dtype=torch.float64), torch.zeros(d, d, dtype=torch.float64))


def test_pinv_readout_matches_normal_equations_at_full_rank_and_records_rank():
    generator = torch.Generator().manual_seed(7)
    Z = torch.randn(20, 3, generator=generator, dtype=torch.float64)
    V_true = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    Y = Z @ V_true.T
    V, rank = ss.pinv_readout(Z, Y)
    assert rank == 3 and torch.allclose(V, V_true, atol=1e-8)
    normal = torch.linalg.solve(Z.T @ Z, Z.T @ Y).T
    assert torch.allclose(V, normal, atol=1e-8)
    deficient = torch.cat([Z[:, :2], Z[:, :1]], dim=1)  # third column duplicates the first
    _, rank_deficient = ss.pinv_readout(deficient, Y)
    assert rank_deficient == 2


def _rows_with_deficient_template(d: int = 8, n_tokens: int = 12) -> ss.DesignRows:
    generator = torch.Generator().manual_seed(11)
    tokens = tuple(f"t{i}" for i in range(n_tokens))
    xs, ys, templates, owners = [], [], [], []
    shared_direction = torch.randn(d, generator=generator, dtype=torch.float64)
    for i, token in enumerate(tokens):
        for template in pm.TEMPLATE_ORDER:
            for _frame in range(2):
                x = torch.randn(d, generator=generator, dtype=torch.float64)
                if template == "quantifier":
                    x = (i + 1) * shared_direction  # every quantifier row is a multiple of one direction: rank(Z_T) ≤ 1
                xs.append(x)
                ys.append(torch.randn(d, generator=generator, dtype=torch.float64))
                templates.append(template)
                owners.append(token)
    return ss.DesignRows(tokens, torch.stack(xs), torch.stack(ys), tuple(templates), tuple(owners))


def test_identifiability_rule_is_an_incident_for_the_primary_and_a_flag_for_pca():
    rows = _rows_with_deficient_template()
    references = {template: 0 for template in pm.TEMPLATE_ORDER}
    ok = ss.fit_supervised(rows, 1, reference_ids=references)
    assert ok.diagnostics["rank_Z"]["quantifier"] == 1
    with pytest.raises(pm.IncidentError, match="rank\\(Z_T\\) = 1 < r = 2 for template quantifier"):
        ss.fit_supervised(rows, 2, reference_ids=references, where="final fit")
    e_vectors = {token: torch.randn(8, generator=torch.Generator().manual_seed(i), dtype=torch.float64) for i, token in enumerate(rows.tokens)}
    pca = ss.fit_pca_006(rows, 2, e_vectors=e_vectors, reference_ids=references)
    assert pca.kind == ss.KIND_PCA and pca.diagnostics["rank_deficient_templates"] == ["quantifier"]


def test_ridge_closed_form_and_multiplier_tie_break():
    rows = _rows_with_deficient_template()
    lam = 0.3
    maps = ss.ridge_maps(rows, lam)
    for template in pm.TEMPLATE_ORDER:
        X_T, Y_T = rows.rows_of(template)
        expected = torch.linalg.inv(X_T.T @ X_T + lam * torch.eye(8, dtype=torch.float64)) @ X_T.T @ Y_T
        assert torch.allclose(maps[template].T, expected, atol=1e-8)
    assert ss.choose_multiplier({0.01: 2.0, 0.1: 1.0, 1.0: 1.0, 10.0: 1.5}) == 1.0  # tie → larger multiplier
    assert ss.choose_multiplier({0.01: 0.5, 0.1: 1.0}) == 0.01
    assert ss.trace_scale(torch.eye(8, dtype=torch.float64) * 2.0) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Fake-model tests.


@pytest.fixture(scope="module")
def setting():
    ctx = make_context()
    references = {template: ctx.frames_of(template)[0].cue_ids["sg"] for template in pm.TEMPLATE_ORDER}
    tokens: list[tuple[str, int]] = []
    seen: set[int] = set()
    for frame in ctx.frames:
        for label in ("sg", "pl"):
            if frame.cue_ids[label] not in seen:
                seen.add(frame.cue_ids[label])
                tokens.append((f"{frame.template_id}:{label}", frame.cue_ids[label]))
    for extra in (6, 7, 8, 9, 12, 13):
        if extra not in seen:
            seen.add(extra)
            tokens.append((f"tok{extra}", extra))
    responses = cd.measure_epatch_responses(ctx.model, ctx.weights, ctx.cache, ctx.frames, tokens, references, circuit=FAKE_CIRCUIT)
    e_vectors, reference_vectors = cd._token_vectors(ctx.weights, tokens, references)
    contexts = ss.context_residuals(ctx.cache, ctx.frames, references)
    ev = ss.make_evaluation(ctx.weights, contexts, e_vectors, reference_vectors, responses, ctx.frames, ctx.nouns)
    return ctx, references, tuple(tokens), responses, ev


def test_zero_prediction_at_the_reference_for_every_family(setting):
    ctx, references, tokens, responses, ev = setting
    names = [token for token, _ in tokens]
    rows = ss.design_rows(e_vectors=ev.e_vectors, reference_vectors=ev.reference_vectors, responses=responses, frames=ctx.frames, tokens=names)
    fits = [ss.fit_supervised(rows, 2, reference_ids=references), ss.fit_pca_006(rows, 2, e_vectors=ev.e_vectors, reference_ids=references),
            ss.fit_ridge_full(ev, names, reference_ids=references)]
    for fit in fits:
        for frame in ctx.frames:
            template = frame.template_id
            ref = ev.reference_vectors[template]
            delta = ss.predict_delta(fit, template, ref, ref)
            assert float(delta.abs().max()) == 0.0
            assert float(ss.predict_shifts(ctx.weights, ev.contexts.rho(template, frame.frame_id), delta, ev.u_matrix).abs().max()) == 0.0
    ridge = fits[2]
    assert ridge.kind == ss.KIND_RIDGE and ridge.basis is None and ridge.diagnostics["multiplier"] in ss.RIDGE_MULTIPLIERS
    assert ridge.diagnostics["lambda"] == pytest.approx(ridge.diagnostics["multiplier"] * ss.trace_scale(rows.X))


def test_cue_level_values_agree_with_the_experiment_006_per_noun_readout(setting):
    ctx, references, tokens, responses, ev = setting
    names = [token for token, _ in tokens]
    rows = ss.design_rows(e_vectors=ev.e_vectors, reference_vectors=ev.reference_vectors, responses=responses, frames=ctx.frames, tokens=names)
    fit = ss.fit_supervised(rows, 2, reference_ids=references)
    token = names[3]
    values = ss.cue_level_values(fit, ev, token)
    predicted, measured, errors = [], [], []
    for frame in ctx.frames:
        delta = ss.predict_delta(fit, frame.template_id, ev.e_vectors[token], ev.reference_vectors[frame.template_id])
        rho = ev.contexts.rho_frame[frame.frame_id]
        for noun in ev.nouns:
            gamma, beta = ctx.weights.ln_final_w.double(), ctx.weights.ln_final_b.double()
            value = float(-ctx.weights.u(noun) @ (pm.exact_layer_norm(rho + delta, gamma, beta, ctx.weights.eps) - pm.exact_layer_norm(rho, gamma, beta, ctx.weights.eps)))
            predicted.append(value)
            measured.append(responses[(token, frame.frame_id)].shifts[noun.lexical_key])
            errors.append(abs(value - measured[-1]))
    assert values["predicted"] == pytest.approx(pm._mean(predicted), abs=1e-9)
    assert values["measured"] == pytest.approx(pm._mean(measured), abs=1e-9)
    assert values["mae"] == pytest.approx(pm._mean(errors), abs=1e-9)


def test_ridge_selection_is_leakage_free(setting):
    ctx, references, tokens, responses, ev = setting
    names = [token for token, _ in tokens]
    held_out, training = names[0], names[1:]
    fit = ss.fit_ridge_full(ev, training, reference_ids=references)
    perturbed_vectors = dict(ev.e_vectors)
    perturbed_vectors[held_out] = ev.e_vectors[held_out] * 7.0 + 1.0
    ev_perturbed = ss.make_evaluation(ctx.weights, ev.contexts, perturbed_vectors, ev.reference_vectors, responses, ctx.frames, ctx.nouns)
    refit = ss.fit_ridge_full(ev_perturbed, training, reference_ids=references)
    assert refit.diagnostics == fit.diagnostics
    assert all(torch.equal(refit.maps[template], fit.maps[template]) for template in pm.TEMPLATE_ORDER)
    for template in pm.TEMPLATE_ORDER:
        assert fit.maps[template].shape == (ctx.weights.W_E.shape[1], ctx.weights.W_E.shape[1])


def test_loco_tables_fold_by_whole_token_with_diagnostics(setting):
    ctx, references, tokens, responses, ev = setting
    names = [token for token, _ in tokens]
    tables = ss.loco_tables(ev, names, reference_ids=references, ranks=(1, 2))
    assert set(tables["supervised"]) == {1, 2} and set(tables["supervised"][1]) == set(names) and set(tables["pca-006"][2]) == set(names)
    assert set(tables["ridge-full"]) == set(names) and set(tables["folds"]) == set(names)
    fold = tables["folds"][names[0]]
    assert set(fold["gaps"]) == {"1", "2"} and fold["rank_C"] >= ss.MIN_CROSS_MOMENT_RANK
    assert fold["rank_Z"]["2"] == {template: 2 for template in pm.TEMPLATE_ORDER}
    assert fold["ridge"]["multiplier"] in ss.RIDGE_MULTIPLIERS and set(fold["ridge"]["inner_scores"]) == {f"{m:g}" for m in ss.RIDGE_MULTIPLIERS}
    selection = cd.select_rank(tables["supervised"])
    assert selection["selected"] in (1, 2)
    comparison = ss.comparison_record(tables, selection, e005_error=1.0)
    assert set(comparison["per_rank"]) == {"1", "2"} and comparison["selected_rank"] == selection["selected"]
    pair = comparison["per_rank"]["1"]
    assert pair["prediction_met"] == (pair["supervised_error"] <= 0.8 * pair["pca_006_error"])
    assert comparison["ridge_over_selected"] == pytest.approx(comparison["ridge_full_error"] / comparison["selected_error"])


def test_inherited_check_passes_on_identical_values_and_raises_beyond_tolerance(setting):
    ctx, references, tokens, responses, ev = setting
    recorded = {f"{token}|{frame_id}": {"template_id": r.template_id, "mean_shift": pm._mean(list(r.shifts.values())), "delta_norm": float(r.delta_residual.norm()), "circuit_share": r.circuit_share}
                for (token, frame_id), r in responses.items()}
    extract = ss.inherited_extract_payload(recorded, source={"path": "x", "file_sha256": "0" * 64, "state_sha256": "1" * 64, "run_id": "r", "protocol_code_commit": "c" * 40, "explore_completed_at": "t"},
                                           manifest_sha256="m", extension_sha256="e", confirmation_sha256=ss.INHERITED_CONFIRMATION_SHA256, model={"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision})
    verdict = ss.check_inherited_responses(responses, extract)
    assert verdict["passed"] and verdict["max_abs_deviation"] == 0.0 and verdict["n"] == len(responses)
    key = next(iter(extract["responses"]))
    perturbed = {**extract, "responses": {**extract["responses"], key: {**extract["responses"][key], "mean_shift": extract["responses"][key]["mean_shift"] + 2e-6}}}
    with pytest.raises(pm.IncidentError, match="deviates"):
        ss.check_inherited_responses(responses, perturbed)
    missing = {**extract, "responses": {k: v for k, v in extract["responses"].items() if k != key}}
    with pytest.raises(pm.IncidentError, match="cover exactly"):
        ss.check_inherited_responses(responses, missing)


def test_inherited_extract_loads_only_with_the_frozen_confirmation_digest(tmp_path):
    recorded = {f"t{i}|f{j}": {"template_id": "cardinal", "mean_shift": 0.1 * i, "delta_norm": 1.0, "circuit_share": 0.5} for i in range(16) for j in range(12)}
    source = {"path": "x", "file_sha256": "0" * 64, "state_sha256": "1" * 64, "run_id": "r", "protocol_code_commit": "c" * 40, "explore_completed_at": "t"}
    good = ss.inherited_extract_payload(recorded, source=source, manifest_sha256="m", extension_sha256="e", confirmation_sha256=ss.INHERITED_CONFIRMATION_SHA256, model={"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision})
    path = tmp_path / "extract.json"
    path.write_text(pm.canonical_json(good) + "\n", encoding="utf-8")
    loaded = ss.load_inherited_extract(path, manifest_sha256="m", extension_sha256="e", confirmation_sha256=ss.INHERITED_CONFIRMATION_SHA256)
    assert loaded["content_sha256"] == good["content_sha256"]
    with pytest.raises(ValueError, match="different frozen inputs"):
        ss.load_inherited_extract(path, manifest_sha256="other", extension_sha256="e", confirmation_sha256=ss.INHERITED_CONFIRMATION_SHA256)
    with pytest.raises(ValueError, match="confirmation digest"):
        ss.load_inherited_extract(path, manifest_sha256="m", extension_sha256="e", confirmation_sha256="f" * 64)
    bad = ss.inherited_extract_payload(recorded, source=source, manifest_sha256="m", extension_sha256="e", confirmation_sha256="f" * 64, model={"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision})
    path.write_text(pm.canonical_json(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="confirmation digest"):
        ss.load_inherited_extract(path, manifest_sha256="m", extension_sha256="e", confirmation_sha256=ss.INHERITED_CONFIRMATION_SHA256)
    tampered = dict(good)
    tampered["responses"] = {**good["responses"], "t0|f0": {**good["responses"]["t0|f0"], "mean_shift": 9.0}}
    path.write_text(pm.canonical_json(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        ss.load_inherited_extract(path, manifest_sha256="m", extension_sha256="e", confirmation_sha256=ss.INHERITED_CONFIRMATION_SHA256)


def test_committed_extract_matches_the_frozen_inputs_and_covers_the_exposed_pool():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    manifest, manifest_sha256, extension = pm.load_inputs(root)
    extract = ss.load_inherited_extract(root / ss.INHERITED_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=ss.INHERITED_CONFIRMATION_SHA256)
    pool = cd.exposed_pool(manifest, extension)
    assert set(extract["responses"]) == {f"{token}|{frame.frame_id}" for token, _ in pool.tokens for frame in pool.frames}
    assert all(entry["template_id"] == frame.template_id for frame in pool.frames for token, _ in pool.tokens for entry in [extract["responses"][f"{token}|{frame.frame_id}"]])
    confirmation = cd.load_confirmation(root / ss.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    assert confirmation.content_sha256 == ss.INHERITED_CONFIRMATION_SHA256 == extract["confirmation_sha256"]


def test_outcome_rule_branches():
    passed = {"cue_effect": {"passed": True}}
    failed = {"cue_effect": {"passed": False}}
    program_ok = {"passed": True, "failures": []}
    program_bad = {"passed": False, "failures": ["Y2"]}
    assert ss.outcome(fresh_floors=failed, program_floors=program_ok, bands={"hit": True})["label"] == "CUE_EFFECT_NOT_REPLICATED"
    assert ss.outcome(fresh_floors=passed, program_floors=program_bad, bands={"hit": True})["label"] == "PROGRAM_NOT_SUPPORTED"
    assert ss.outcome(fresh_floors=passed, program_floors=program_ok, bands={"hit": True})["label"] == "DECOMPILED"
    assert ss.outcome(fresh_floors=passed, program_floors=program_ok, bands={"hit": False})["label"] == "DECOMPILED_MISCALIBRATED"
    report = ss.circuit_report({"passed": True, "failures": []}, {"passed": False, "failures": ["P3"]})
    assert not report["c002_review_eligible"] and not report["enters_outcome"]
    assert set(ss.OUTCOME_LABELS) == {"QUALITY_GATE_FAILED", "CUE_EFFECT_NOT_REPLICATED", "DECOMPILED", "DECOMPILED_MISCALIBRATED", "PROGRAM_NOT_SUPPORTED"}


def test_frozen_constants_match_the_design():
    assert ss.RANKS == (1, 2, 3, 4) and ss.GAP_MIN == 1e-8 and ss.MIN_CROSS_MOMENT_RANK == 5
    assert ss.RIDGE_MULTIPLIERS == (1e-2, 1e-1, 1.0, 10.0, 100.0)
    assert ss.INHERITED_EPATCH_TOLERANCE == 1e-6 and ss.RUNTIME_SEED == 20260916 and ss.CONTROL_SEED == 20260920
    assert ss.INHERITED_CONFIRMATION_SHA256 == "dbb8dbcef9ab201cb5555a3c1d988e3ca8ab70f742f10b12f80a856c9ebf5521"
    assert ss.PROGRAM_NAMES == ("selected", "pca-006", "e005-scalar", "ridge-full") and ss.PCA_IMPROVEMENT_PREDICTION == 0.8
    assert cd.RANK_IMPROVEMENT_FACTOR == 0.8 and cd.QUALITY_SPEARMAN_FLOOR == 0.70 and cd.QUALITY_NORMALIZED_RMSE_CEILING == 0.50
    assert cd.P3_CORRELATION == 0.90 and cd.CUE_EFFECT_FRESH == 108 / 120


def test_predictions_are_invariant_to_basis_column_signs(setting):
    ctx, references, tokens, responses, ev = setting
    names = [token for token, _ in tokens]
    rows = ss.design_rows(e_vectors=ev.e_vectors, reference_vectors=ev.reference_vectors, responses=responses, frames=ctx.frames, tokens=names)
    svd = ss.cross_moment_svd(rows.X, rows.Y, ranks=(1, 2))
    flipped = ss.CrossMomentSVD(svd.P * -1.0, svd.singular_values, svd.rank_C, svd.gaps)
    fit = ss.fit_supervised(rows, 2, reference_ids=references, svd=svd)
    fit_flipped = ss.fit_supervised(rows, 2, reference_ids=references, svd=flipped)
    for token in names[:4]:
        assert ss.cue_level_values(fit, ev, token)["predicted"] == pytest.approx(ss.cue_level_values(fit_flipped, ev, token)["predicted"], abs=1e-12)


def test_inner_ridge_grid_is_scaled_by_the_inner_training_rows_only(setting, monkeypatch):
    ctx, references, tokens, responses, ev = setting
    names = [token for token, _ in tokens]
    training = names[1:]
    seen: list[tuple[tuple[str, ...], float]] = []
    original = ss.ridge_maps

    def spy(rows, lam):
        seen.append((rows.tokens, lam))
        return original(rows, lam)

    monkeypatch.setattr(ss, "ridge_maps", spy)
    ss.select_ridge_multiplier(ev, training)
    assert len(seen) == len(training) * len(ss.RIDGE_MULTIPLIERS)
    for inner_tokens, lam in seen:
        assert len(inner_tokens) == len(training) - 1 and names[0] not in inner_tokens
        inner_rows = ss.design_rows(e_vectors=ev.e_vectors, reference_vectors=ev.reference_vectors, responses=responses, frames=ctx.frames, tokens=list(inner_tokens))
        assert any(lam == pytest.approx(multiplier * ss.trace_scale(inner_rows.X)) for multiplier in ss.RIDGE_MULTIPLIERS)


def test_e005_frozen_check_reads_the_lock_statement():
    lock = {"parameters_index_sha256": "x", "statement": "R  [L05.MLP, L04.MLP] at p_t : δ(x) = n_t(x) · v_T with g_R = |v_T|: cardinal 5.158, quantifier 7.900, coordinated-adjective 3.800\nREADOUT",
            "parameters_index": {"k_t": 0.5, "tensors": {"W_E": {"sha256": "a"}, "v_t.cardinal": {"sha256": "b"}}},
            "mechanism": {"mechanism": {"branch": "L00.MLP", "e_keys": ["L00.MLP"], "t_keys": ["L03.H04"], "r_keys": ["L05.MLP", "L04.MLP"]}}}
    record = {"parameters_index_sha256": "y", "gains": {"cardinal": 5.1580, "quantifier": 7.8999, "coordinated-adjective": 3.8005}}
    index = {"k_t": 0.5, "tensors": {"W_E": {"sha256": "a"}, "v_t.cardinal": {"sha256": "c"}}}
    verdict = ss.e005_frozen_check(record, index, lock)
    assert verdict["passed"] and verdict["mismatched_tensors"] == ["v_t.cardinal"] and not verdict["index_sha256_matches_lock"]
    with pytest.raises(pm.IncidentError):
        ss.e005_frozen_check(record, {"k_t": 0.6, "tensors": index["tensors"]}, lock)
    with pytest.raises(pm.IncidentError):
        ss.e005_frozen_check({**record, "gains": {**record["gains"], "cardinal": 5.2}}, index, lock)
    with pytest.raises(pm.IncidentError):
        ss.e005_frozen_check(record, {"k_t": 0.5, "tensors": {"W_E": {"sha256": "z"}, "v_t.cardinal": {"sha256": "c"}}}, lock)
    circuit = ss.e005_refit_circuit(lock, cd.FIXED_CIRCUIT)
    assert circuit.r_keys == ("L05.MLP", "L04.MLP") and set(circuit.r_keys) == set(cd.FIXED_CIRCUIT.r_keys)
    assert ss.e005_refit_circuit({}, cd.FIXED_CIRCUIT) is cd.FIXED_CIRCUIT
    with pytest.raises(pm.IncidentError):
        ss.e005_refit_circuit({"mechanism": {"mechanism": {"branch": "L00.MLP", "e_keys": ["L00.MLP"], "t_keys": ["L02.H01"], "r_keys": ["L05.MLP"]}}}, cd.FIXED_CIRCUIT)
    assert ss.e005_frozen_check(record, index, {}) == {"checked": False, "reason": "no Experiment 005 lock available"}


def test_program_failure_where_and_lock_prediction_check():
    verdict = ss.outcome(fresh_floors={"cue_effect": {"passed": True}}, program_floors={"passed": False, "failures": ["Y1", "Y3"]}, bands={"hit": False})
    assert verdict["program_failure_where"] == ["encoding subspace", "frame context"]
    measurements = {"per_token": {"w": {"frames": {"f": {"predicted:selected": 1.0, "predicted:pca-006": 2.0}}}}}
    lock = {"predictions": {"tokens": {"w": {"frames": {"f": {"predicted": {"selected": {"mean": 1.0}, "pca-006": {"mean": 2.0}}}}}}}}
    ss.check_locked_predictions(measurements, lock)
    lock["predictions"]["tokens"]["w"]["frames"]["f"]["predicted"]["pca-006"]["mean"] = 2.0 + 1e-6
    with pytest.raises(pm.IncidentError, match="preregistered"):
        ss.check_locked_predictions(measurements, lock)
