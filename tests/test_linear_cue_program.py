"""Experiment 007: export/load of the weight-only linear cue program for every kind, and its independence from the network stack."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import supervised_subspace as ss
from plural_fakes import make_context

ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PATH = ROOT / "experiments/007-supervised-cue-subspace/linear_cue_program.py"
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
    for extra in (6, 7, 8, 9, 12, 13):
        if extra not in seen:
            seen.add(extra)
            tokens.append((f"tok{extra}", extra))
    responses = cd.measure_epatch_responses(ctx.model, ctx.weights, ctx.cache, ctx.frames, tokens, references, circuit=FAKE_CIRCUIT)
    e_vectors, reference_vectors = cd._token_vectors(ctx.weights, tokens, references)
    contexts = ss.context_residuals(ctx.cache, ctx.frames, references)
    ev = ss.make_evaluation(ctx.weights, contexts, e_vectors, reference_vectors, responses, ctx.frames, ctx.nouns)
    names = [token for token, _ in tokens]
    rows = ss.design_rows(e_vectors=e_vectors, reference_vectors=reference_vectors, responses=responses, frames=ctx.frames, tokens=names)
    fits = {"selected": ss.fit_supervised(rows, 2, reference_ids=references), "pca-006": ss.fit_pca_006(rows, 2, e_vectors=e_vectors, reference_ids=references),
            "ridge-full": ss.fit_ridge_full(ev, names, reference_ids=references)}
    return ctx, references, tuple(tokens), ev, fits


@pytest.mark.parametrize("name", ["selected", "pca-006", "ridge-full"])
def test_export_load_and_program_predictions_match_the_fit(setting, tmp_path, name):
    ctx, references, tokens, ev, fits = setting
    fit = fits[name]
    index = ss.export_program(tmp_path, ctx.weights, fit, ev.contexts)
    assert index["kind"] == fit.kind and set(index["reference_ids"]) == set(pm.TEMPLATE_ORDER) and ("basis" in index["tensors"]) == (fit.basis is not None)
    program = ss.load_linear_program(tmp_path, PROGRAM_PATH)
    assert program.kind == fit.kind
    for frame in ctx.frames:
        rho = ev.contexts.rho_frame[frame.frame_id]
        for token, token_id in tokens:
            delta = ss.predict_delta(fit, frame.template_id, ev.e_vectors[token], ev.reference_vectors[frame.template_id])
            expected = ss.predict_shifts(ctx.weights, rho, delta, ev.u_matrix)
            for noun, value in zip(ev.nouns, expected.tolist()):
                assert program.predict_epatch_shift(frame.template_id, frame.frame_id, token_id, noun.sg_ids[0], noun.pl_ids[0]) == pytest.approx(value, abs=1e-9)
            assert torch.allclose(program.E(token_id).float(), pm.lexicon_vector(ctx.weights, token_id), atol=1e-6)
    frame = ctx.frames[0]
    noun = ctx.nouns[1]
    template = frame.template_id
    assert program.predict_epatch_shift(template, frame.frame_id, references[template], noun.sg_ids[0], noun.pl_ids[0]) == 0.0
    # A frame outside the exported contexts falls back to the template mean residual.
    token_id = tokens[-1][1]
    delta = ss.predict_delta(fit, template, ev.e_vectors[tokens[-1][0]], ev.reference_vectors[template])
    fallback = ss.predict_shifts(ctx.weights, ev.contexts.rho_template[template], delta, ev.u_matrix)[1]
    assert program.predict_epatch_shift(template, None, token_id, noun.sg_ids[0], noun.pl_ids[0]) == pytest.approx(float(fallback), abs=1e-9)
    assert program.predict_pair(template, None, frame.cue_ids["sg"], frame.cue_ids["pl"], noun.sg_ids[0], noun.pl_ids[0]) == pytest.approx(
        program.predict_epatch_shift(template, None, frame.cue_ids["sg"], noun.sg_ids[0], noun.pl_ids[0]) - program.predict_epatch_shift(template, None, frame.cue_ids["pl"], noun.sg_ids[0], noun.pl_ids[0]))
    target = "basis.pt" if fit.basis is not None else "map.cardinal.pt"
    torch.save(torch.load(tmp_path / target, weights_only=True) + 1.0, tmp_path / target)
    with pytest.raises(ValueError):
        ss.load_linear_program(tmp_path, PROGRAM_PATH)


def test_program_module_is_independent_of_the_network_stack():
    script = (
        "import sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('linear_cue_program', {str(PROGRAM_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec); sys.modules['linear_cue_program'] = module; spec.loader.exec_module(module)\n"
        "print(','.join(module.forbidden_modules_loaded()))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == ""
    import_lines = [line for line in PROGRAM_PATH.read_text(encoding="utf-8").splitlines() if line.startswith(("import ", "from "))]
    assert import_lines and all("transformer" not in line and "neural_decompiler" not in line for line in import_lines)


def test_load_programs_returns_every_locked_program(setting, tmp_path):
    ctx, references, tokens, ev, fits = setting
    for name, fit in fits.items():
        ss.export_program(tmp_path / name, ctx.weights, fit, ev.contexts)
    program_005, _ = cd.e005_scalar_baseline(ctx, parameters_dir=tmp_path / "e005-scalar", program_path=ROOT / "experiments/005-regular-plural-mechanism/mechanism_program.py",
                                             expected_index_sha256=None, circuit=FAKE_CIRCUIT)
    programs = ss.load_programs(tmp_path, PROGRAM_PATH, ROOT / "experiments/005-regular-plural-mechanism/mechanism_program.py", references)
    assert set(programs) == set(ss.PROGRAM_NAMES)
    frame = ctx.frames[0]
    noun = ctx.nouns[0]
    for program in programs.values():
        assert isinstance(program.predict_epatch_shift(frame.template_id, None, tokens[-1][1], noun.sg_ids[0], noun.pl_ids[0]), float)
        assert isinstance(program.predict_pair(frame.template_id, None, frame.cue_ids["sg"], frame.cue_ids["pl"], noun.sg_ids[0], noun.pl_ids[0]), float)
