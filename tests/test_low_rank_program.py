"""Experiment 006: E-patch responses, the low-rank fit, export/load, and the weight-only program."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import plural_mechanism as pm
from neural_decompiler.candidate_screening import Split
from plural_fakes import make_context

ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PATH = ROOT / "experiments/006-low-rank-cue-decompilation/low_rank_program.py"
FAKE_CIRCUIT = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L01.H00",), ("L01.MLP",))


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
    for extra in (6, 7, 8, 9, 12, 13):  # additional "cue tokens" from the fake vocabulary
        if extra not in seen:
            seen.add(extra)
            tokens.append((f"tok{extra}", extra))
    responses = cd.measure_epatch_responses(ctx.model, ctx.weights, ctx.cache, ctx.frames, tokens, references, circuit=FAKE_CIRCUIT)
    e_vectors = {token: pm.lexicon_vector(ctx.weights, token_id) for token, token_id in tokens}
    reference_vectors = {template: pm.lexicon_vector(ctx.weights, token_id) for template, token_id in references.items()}
    return ctx, references, tuple(tokens), responses, e_vectors, reference_vectors


def test_epatch_response_is_zero_for_the_reference_and_captured_for_others(setting):
    ctx, references, tokens, responses, _, _ = setting
    for frame in ctx.frames:
        ref_token = next(token for token, token_id in tokens if token_id == references[frame.template_id])
        assert float(responses[(ref_token, frame.frame_id)].delta_residual.abs().max()) == 0.0
        assert all(value == 0.0 for value in responses[(ref_token, frame.frame_id)].shifts.values())
        other = next(token for token, token_id in tokens if token_id != references[frame.template_id])
        assert float(responses[(other, frame.frame_id)].delta_residual.abs().max()) > 0.0
        assert responses[(other, frame.frame_id)].integrity[0]["outside_max_abs_change"] == 0.0


def test_pca_basis_orientation_and_no_intercept_fit(setting):
    ctx, references, tokens, responses, e_vectors, reference_vectors = setting
    mu, basis, singular = cd.pca_basis([e_vectors[token] for token, _ in tokens], 2)
    assert basis.shape == (ctx.weights.W_E.shape[1], 2)
    assert torch.allclose(basis.T @ basis, torch.eye(2, dtype=torch.float64), atol=1e-8)
    assert singular[0] >= singular[1]
    fit = cd.fit_low_rank(e_vectors=e_vectors, reference_ids=references, reference_vectors=reference_vectors, responses=responses,
                          frames=ctx.frames, cache=ctx.cache, fit_tokens=[token for token, _ in tokens], rank=2)
    for template in pm.TEMPLATE_ORDER:
        assert fit.maps[template].shape == (ctx.weights.W_E.shape[1], 2)
        ref_vector = reference_vectors[template]
        assert float(fit.dz(template, ref_vector, ref_vector).abs().max()) == 0.0
    noun = ctx.nouns[0]
    frame = ctx.frames[0]
    ref_vector = reference_vectors[frame.template_id]
    assert cd.predict_epatch_shift(fit, ctx.weights, frame.template_id, frame.frame_id, ref_vector, ref_vector, noun) == 0.0
    # A planted exactly-linear response is recovered by the no-intercept fit.
    planted = torch.randn(ctx.weights.W_E.shape[1], 2, dtype=torch.float64)
    fake_responses = {}
    for (token, frame_id), response in responses.items():
        dz = fit.dz(response.template_id, e_vectors[token], reference_vectors[response.template_id])
        fake_responses[(token, frame_id)] = cd.EPatchResponse(token, response.token_id, frame_id, response.template_id, (planted @ dz).float(), response.shifts, 1.0, ())
    refit = cd.fit_low_rank(e_vectors=e_vectors, reference_ids=references, reference_vectors=reference_vectors, responses=fake_responses,
                            frames=ctx.frames, cache=ctx.cache, fit_tokens=[token for token, _ in tokens], rank=2)
    assert all(torch.allclose(refit.maps[template], planted, atol=1e-4) for template in pm.TEMPLATE_ORDER)


def test_export_load_and_program_predictions_match_the_fit(setting, tmp_path):
    ctx, references, tokens, responses, e_vectors, reference_vectors = setting
    fit = cd.fit_low_rank(e_vectors=e_vectors, reference_ids=references, reference_vectors=reference_vectors, responses=responses,
                          frames=ctx.frames, cache=ctx.cache, fit_tokens=[token for token, _ in tokens], rank=2)
    index = cd.export_low_rank(tmp_path, ctx.weights, fit)
    assert index["rank"] == 2 and set(index["reference_ids"]) == set(pm.TEMPLATE_ORDER)
    program = cd.load_low_rank_program(tmp_path, PROGRAM_PATH)
    noun = ctx.nouns[1]
    for frame in ctx.frames:
        for token, token_id in tokens:
            expected = cd.predict_epatch_shift(fit, ctx.weights, frame.template_id, frame.frame_id, e_vectors[token], reference_vectors[frame.template_id], noun)
            assert program.predict_epatch_shift(frame.template_id, frame.frame_id, token_id, noun.sg_ids[0], noun.pl_ids[0]) == pytest.approx(expected, abs=1e-6)
            assert torch.allclose(program.E(token_id).float(), pm.lexicon_vector(ctx.weights, token_id), atol=1e-6)
    frame = ctx.frames[0]
    assert program.predict_epatch_shift(frame.template_id, frame.frame_id, references[frame.template_id], noun.sg_ids[0], noun.pl_ids[0]) == 0.0
    assert isinstance(program.predict_pair(frame.template_id, None, frame.cue_ids["sg"], frame.cue_ids["pl"], noun.sg_ids[0], noun.pl_ids[0]), float)
    values = cd.cue_level_values(fit, ctx.weights, e_vectors, reference_vectors, responses, ctx.frames, ctx.nouns, tokens[1][0])
    assert set(values) == {"predicted", "measured", "mae"}
    torch.save(torch.load(tmp_path / "basis.pt", weights_only=True) + 1.0, tmp_path / "basis.pt")
    with pytest.raises(ValueError):
        cd.load_low_rank_program(tmp_path, PROGRAM_PATH)


def test_program_module_is_independent_of_the_network_stack():
    script = (
        "import sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('low_rank_program', {str(PROGRAM_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec); sys.modules['low_rank_program'] = module; spec.loader.exec_module(module)\n"
        "print(','.join(module.forbidden_modules_loaded()))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == ""
    import_lines = [line for line in PROGRAM_PATH.read_text(encoding="utf-8").splitlines() if line.startswith(("import ", "from "))]
    assert import_lines and all("transformer" not in line and "neural_decompiler" not in line for line in import_lines)


def test_e005_scalar_baseline_refits_on_the_fake(setting, tmp_path):
    ctx, *_ = setting
    program, record = cd.e005_scalar_baseline(ctx, parameters_dir=tmp_path / "e005", program_path=ROOT / "experiments/005-regular-plural-mechanism/mechanism_program.py", expected_index_sha256=None, circuit=FAKE_CIRCUIT)
    assert record["matches_experiment_005_lock"] is None and isinstance(record["k_t"], float)
    frame = ctx.frames[0]
    noun = ctx.nouns[0]
    assert isinstance(program.predict_epatch_shift(frame.template_id, frame.frame_id, frame.cue_ids["pl"], frame.cue_ids["sg"], noun.sg_ids[0], noun.pl_ids[0]), float)


def _table(errors_by_rank: dict[int, list[float]], predicted=None, measured=None):
    table = {}
    for rank, errors in errors_by_rank.items():
        table[rank] = {f"t{i}": {"mae": error, "predicted": (predicted or errors)[i], "measured": (measured or errors)[i]} for i, error in enumerate(errors)}
    return table


def test_rank_rule_branches():
    # No rank beats rank 1 by 20%: rank 1 selected regardless of the one-SE rule.
    table = _table({1: [1.0] * 4, 2: [0.9] * 4, 3: [0.85] * 4, 4: [0.81] * 4})
    assert cd.select_rank(table)["selected"] == 1 and cd.select_rank(table)["eligible"] == [1]
    # Ranks 2-4 eligible; best is 4 but within one SE of rank 2 → smallest eligible within threshold.
    table = _table({1: [1.0, 1.0, 1.0, 1.0], 2: [0.70, 0.74, 0.78, 0.74], 3: [0.72, 0.70, 0.74, 0.70], 4: [0.60, 0.80, 0.65, 0.75]})
    verdict = cd.select_rank(table)
    assert verdict["eligible"] == [1, 2, 3, 4] and verdict["r_best"] == 4 and verdict["selected"] == 2
    assert verdict["threshold"] == pytest.approx(0.70 + verdict["summary"][4]["se"])
    # Best rank far ahead: threshold excludes the others.
    table = _table({1: [1.0] * 4, 2: [0.75] * 4, 3: [0.30, 0.30, 0.30, 0.30], 4: [0.31] * 4})
    assert cd.select_rank(table)["selected"] == 3
    # Rank 1 itself is best among eligible (eligible ranks slightly worse than allowed threshold).
    table = _table({1: [0.5] * 4, 2: [0.39, 0.41, 0.40, 0.40], 3: [0.45] * 4, 4: [0.6] * 4})
    verdict = cd.select_rank(table)
    assert verdict["eligible"] == [1, 2] and verdict["selected"] == 2


def test_quality_gate_and_tau():
    predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
    measured = [1.1, 2.1, 2.9, 4.2, 5.0]
    per_token = {f"t{i}": {"predicted": p, "measured": m, "mae": abs(p - m)} for i, (p, m) in enumerate(zip(predicted, measured))}
    summary = {1: {"error": 1.0, "se": 0.1}, 2: {"error": 0.7, "se": 0.1}}
    gate = cd.quality_gate(per_token, selected=2, summary=summary)
    assert gate["passed"] and gate["spearman"] == pytest.approx(1.0) and gate["normalized_rmse"] < 0.1
    bad = cd.quality_gate(per_token, selected=2, summary={1: {"error": 0.75, "se": 0.1}, 2: {"error": 0.7, "se": 0.1}})
    assert not bad["improvement_ok"] and not bad["passed"]
    reversed_ = {f"t{i}": {"predicted": p, "measured": m, "mae": abs(p - m)} for i, (p, m) in enumerate(zip(predicted, reversed(measured)))}
    assert not cd.quality_gate(reversed_, selected=1, summary=summary)["spearman_ok"]
    assert cd.tolerance_tau(per_token) == 0.5
    assert cd.tolerance_tau({"a": {"mae": 0.4}, "b": {"mae": 0.4}}) == pytest.approx(1.2)


def test_loco_runs_on_the_fake(setting):
    ctx, references, tokens, responses, e_vectors, reference_vectors = setting
    table = cd.loco_errors(e_vectors=e_vectors, reference_ids=references, reference_vectors=reference_vectors, responses=responses, frames=ctx.frames,
                           cache=ctx.cache, nouns=ctx.nouns, weights=ctx.weights, tokens=[token for token, _ in tokens], ranks=(1, 2))
    assert set(table) == {1, 2} and set(table[1]) == {token for token, _ in tokens}
    verdict = cd.select_rank(table)
    assert verdict["selected"] in (1, 2)
    gate = cd.quality_gate(table[verdict["selected"]], selected=verdict["selected"], summary=verdict["summary"])
    assert set(gate) >= {"spearman", "normalized_rmse", "improvement_ok", "passed", "cue_level"}
