"""Hypothesis tree, encoding branch, set selection, program parameters, and the weight-only program."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from neural_decompiler import plural_mechanism as pm
from plural_fakes import make_context

ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PATH = ROOT / "experiments/005-regular-plural-mechanism/mechanism_program.py"


@pytest.fixture(scope="module")
def ctx():
    return make_context()


def test_hypothesis_tree_boundaries():
    assert pm.hypothesis_tree(0.9, 0.9, 0.49, 0.70, 0.9) == ("H3", False)
    assert pm.hypothesis_tree(0.9, 0.9, 0.50, 0.70, 0.9) == ("H1", False)
    assert pm.hypothesis_tree(0.50, 0.50, 0.80, 0.10, 0.10) == ("H1", False)
    assert pm.hypothesis_tree(0.50, 0.50, 0.49, 0.69, 0.10) == ("H1", True)
    assert pm.hypothesis_tree(0.49, 0.9, 0.80, 0.10, 0.70) == ("H2", False)
    assert pm.hypothesis_tree(0.49, 0.9, 0.49, 0.10, 0.70) == ("H2", True)
    assert pm.hypothesis_tree(0.49, 0.9, 0.80, 0.10, 0.69) == ("H2", True)
    with pytest.raises(pm.IncidentError):
        pm.hypothesis_tree(float("nan"), 0.5, 0.5, 0.5, 0.5)


def test_encoding_branch_rules():
    base = {"axis_norm": 10.0}
    assert pm.encoding_branch({**base, "entries": {"EMBED": 3.0, "L00.MLP": 5.0, "L01.MLP": 2.0}})["branch"] == "L00.MLP"
    embed = pm.encoding_branch({**base, "entries": {"EMBED": 6.0, "L00.MLP": 2.0, "L01.MLP": 2.0}})
    assert embed["branch"] == "EMBED" and embed["e_keys"] == ["EMBED"]
    shared = pm.encoding_branch({**base, "entries": {"EMBED": 3.0, "L00.MLP": 3.0, "L01.MLP": 3.0, "L00.H00": 1.0}})
    assert shared["e_keys"] == ["L00.MLP", "L01.MLP"] and shared["flag"] is None
    thin = pm.encoding_branch({**base, "entries": {"EMBED": 3.0, "L00.MLP": 2.0, "L01.MLP": 1.0, "L00.H00": 1.0}})
    assert thin["e_keys"] == ["L00.MLP", "L01.MLP"] and thin["flag"] == "ENCODING_SHARE_BELOW_HALF"
    assert pm.program_eligible("L00.MLP") and pm.program_eligible("EMBED")
    assert not pm.program_eligible("L01.MLP") and not pm.program_eligible("L03.H04")


def test_mechanism_set_roles_and_program_eligibility():
    mechanism = pm.MechanismSet("L00.MLP", ("L00.MLP", "L01.MLP"), ("L03.H04",), ("L04.MLP", "L05.MLP"))
    assert mechanism.l_r == 4 and mechanism.l_t == 3
    assert mechanism.e_program_keys == ("L00.MLP",) and mechanism.contextual_e_keys == ("L01.MLP",)
    assert mechanism.role_sites() == (("L00.MLP", "p_c"), ("L01.MLP", "p_c"), ("L03.H04", "p_t"), ("L04.MLP", "p_t"), ("L05.MLP", "p_t"))
    cue_final = pm.Frame("cardinal", "c", (1, 2, 3), (), {"sg": 4, "pl": 5}, "{cue}")
    assert [key for key, _ in mechanism.p_t_sites_for(cue_final)] == ["L00.MLP", "L01.MLP", "L03.H04", "L04.MLP", "L05.MLP"]
    embed = pm.MechanismSet("EMBED", ("EMBED",), ("L03.H04",), ("L04.MLP",))
    assert embed.role_sites() == (("L03.H04", "p_t"), ("L04.MLP", "p_t"))
    assert embed.e_program_keys == ("EMBED",)


def test_program_parameters_export_and_program_predictions(ctx, tmp_path):
    mechanism = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L01.H00",), ("L01.MLP",))
    parameters = pm.estimate_program_parameters(ctx, mechanism)
    assert set(parameters.v_t) == set(pm.TEMPLATE_ORDER) and len(parameters.rho_frame) == 6
    index = pm.export_program_parameters(tmp_path, ctx.weights, parameters, vocab_size=16)
    assert (tmp_path / "parameters.json").exists() and index["k_t"] == pytest.approx(parameters.k_t)
    program = pm.load_program(tmp_path, PROGRAM_PATH)
    for frame in ctx.frames:
        for prompt in pm.frame_prompts(frame):
            expected = pm.lexicon_vector(ctx.weights, prompt.cue_token_id)
            assert torch.allclose(program.e_program_vector(prompt.cue_token_id).float(), expected, atol=1e-6)
    assert program.lexicon_table.shape == (16,)
    sg_values = [program.n_c(frame.cue_ids["sg"]) for frame in ctx.frames]
    pl_values = [program.n_c(frame.cue_ids["pl"]) for frame in ctx.frames]
    assert sum(sg_values) / 6 == pytest.approx(-1.0, abs=1e-6) or sum(pl_values) / 6 == pytest.approx(1.0, abs=1e-6)
    # Exact LayerNorm equals torch's population-variance LayerNorm.
    r = torch.randn(8, dtype=torch.float64)
    reference = torch.nn.functional.layer_norm(r, (8,), program.ln_final_w, program.ln_final_b, program.eps)
    assert torch.allclose(program.exact_layer_norm(r, program.ln_final_w, program.ln_final_b, program.eps) if hasattr(program, "exact_layer_norm") else reference, reference)
    frame = ctx.frames[0]
    noun = ctx.nouns[0]
    d_hat = program.predict_pair(frame.template_id, frame.frame_id, frame.cue_ids["sg"], frame.cue_ids["pl"], noun.sg_ids[0], noun.pl_ids[0])
    assert isinstance(d_hat, float)
    shift = program.predict_shift(frame.template_id, frame.frame_id, frame.cue_ids["pl"], frame.cue_ids["sg"], noun.sg_ids[0], noun.pl_ids[0])
    assert shift == pytest.approx(-d_hat, abs=1e-9)
    assert program.predict_shift(frame.template_id, frame.frame_id, frame.cue_ids["sg"], frame.cue_ids["sg"], noun.sg_ids[0], noun.pl_ids[0]) == 0.0
    assert isinstance(program.predict_epatch_shift(frame.template_id, frame.frame_id, frame.cue_ids["pl"], frame.cue_ids["sg"], noun.sg_ids[0], noun.pl_ids[0]), float)
    # New frames fall back to the template context.
    assert isinstance(program.predict_pair(frame.template_id, None, frame.cue_ids["sg"], frame.cue_ids["pl"], noun.sg_ids[0], noun.pl_ids[0]), float)
    floors = pm.program_development_floors(program, ctx)
    assert set(floors) == {"passed", "sign_failures", "templates"}


def test_program_rejects_tampered_parameters(ctx, tmp_path):
    mechanism = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L01.H00",), ("L01.MLP",))
    pm.export_program_parameters(tmp_path, ctx.weights, pm.estimate_program_parameters(ctx, mechanism), vocab_size=16)
    tensor = torch.load(tmp_path / "W_U.pt", weights_only=True)
    torch.save(tensor + 1.0, tmp_path / "W_U.pt")
    with pytest.raises(ValueError):
        pm.load_program(tmp_path, PROGRAM_PATH)


def test_program_module_is_independent_of_the_network_stack():
    script = (
        "import sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('mechanism_program', {str(PROGRAM_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec); sys.modules['mechanism_program'] = module; spec.loader.exec_module(module)\n"
        "print(','.join(module.forbidden_modules_loaded()))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == ""
    import_lines = [line for line in PROGRAM_PATH.read_text(encoding="utf-8").splitlines() if line.startswith(("import ", "from "))]
    assert import_lines and all("transformer" not in line and "neural_decompiler" not in line for line in import_lines)


def test_select_mechanism_set_and_run_discovery_on_the_fake(ctx, tmp_path):
    state = pm.new_results_state(manifest_sha256="a" * 64, extension_sha256="b" * 64, protocol_code_commit="c" * 40, git_dirty=False, versions={})
    version = pm.run_discovery(ctx, state=state, results_path=tmp_path / "results.json", program_path=PROGRAM_PATH, parameters_dir=tmp_path / "parameters")
    assert version["version"] == "M1"
    assert version["hypothesis"] in pm.HYPOTHESIS_ROWS
    assert state["mechanism_versions"][-1] is version
    loaded = pm.load_results_state(tmp_path / "results.json")
    assert set(loaded["discovery"]) >= {"a2", "candidate_roles", "a3", "a4", "a5_provisional", "encoding_branch", "a6", "hypothesis", "a7", "a8", "a10_selection"}
    attempts = loaded["discovery"]["a10_selection"]["attempts"]
    assert attempts[0]["k"] == 0 and not attempts[0]["roles_floor"]["passed"]
    if version["status"] == "candidate":
        assert "statement" in version and "VARIABLES" in version["statement"]
        assert version["k"] <= pm.MAX_COMPONENTS_AT_P_T
        assert (tmp_path / "parameters/parameters.json").exists()
    else:
        assert version["outcome"] == "NO_COMPACT_MECHANISM"
