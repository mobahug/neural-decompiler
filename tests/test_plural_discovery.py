"""Tier A measurement plumbing and exact-accounting identities on a tiny parallel-residual model."""

from __future__ import annotations

import itertools

import pytest
import torch

from neural_decompiler import plural_mechanism as pm
from neural_decompiler.candidate_screening import Split
from plural_fakes import TinyPlural


_PREFIXES = {
    "cardinal": (((1, 2, 3), ()), ((6, 2, 3), ())),
    "quantifier": (((9, 10, 11), ()), ((9, 6, 11), ())),
    "coordinated-adjective": (((1, 2, 3, 6, 7), (8,)), ((9, 2, 3, 6, 7), (8,))),
}


def _frames(cues: dict[str, tuple[int, int]]) -> tuple[pm.Frame, ...]:
    frames = []
    for template, variants in _PREFIXES.items():
        for index, (prefix, suffix) in enumerate(variants, start=1):
            sg, pl = cues[template]
            frames.append(pm.Frame(template, f"{template}-{index}", prefix, suffix, {"sg": sg, "pl": pl}, f"{template} {index} {{cue}}"))
    return tuple(frames)


def _nouns() -> tuple[pm.Noun, ...]:
    return (pm.Noun("n0", Split.DEVELOPMENT, "simple-suffix", (0,), (1,)),
            pm.Noun("n1", Split.DEVELOPMENT, "simple-suffix", (14,), (15,)),
            pm.Noun("n2", Split.DEVELOPMENT, "sibilant-es", (2,), (3,)),
            pm.Noun("n3", Split.DEVELOPMENT, "consonant-y", (10,), (11,)))


@pytest.fixture(scope="module")
def ctx() -> pm.DiscoveryContext:
    """Pick, per template, the cue pair that moves the contrast most, so denominators clear the floors."""
    model = TinyPlural(seed=3)
    nouns = _nouns()
    cache = pm.PromptCache(model, nouns)
    candidates = [token for token in range(16) if token not in {0, 1, 2, 3, 10, 11, 14, 15}]
    cues: dict[str, tuple[int, int]] = {}
    for template, variants in _PREFIXES.items():
        best, best_value = None, -1.0
        for sg, pl in itertools.permutations(candidates, 2):
            values = []
            for index, (prefix, suffix) in enumerate(variants, start=1):
                frame = pm.Frame(template, f"{template}-{index}", prefix, suffix, {"sg": sg, "pl": pl}, "{cue}")
                a, b = pm.frame_prompts(frame)
                values.extend(cache.c(a)[noun.lexical_key] - cache.c(b)[noun.lexical_key] for noun in nouns)
            mean = sum(values) / len(values)
            if mean > best_value:
                best, best_value = (sg, pl), mean
        assert best is not None and best_value >= 0.6, (template, best_value)
        cues[template] = best
    return pm.DiscoveryContext(model, pm.Weights.from_model(model), None, _frames(cues), nouns, pm.PromptCache(model, nouns))


def test_lexicon_matches_captured_layer0_mlp_everywhere(ctx):
    for frame in ctx.frames:
        for prompt in pm.frame_prompts(frame):
            captured = ctx.cache.run(prompt).vector(("L00.MLP", prompt.p_c))
            assert torch.allclose(pm.lexicon_vector(ctx.weights, prompt.cue_token_id), captured, atol=1e-6)


def test_direct_effects_reconstruct_the_contrast_exactly(ctx):
    for frame in ctx.frames:
        for prompt in pm.frame_prompts(frame):
            effects = pm.direct_effects(ctx.weights, ctx.cache.run(prompt), ctx.nouns, p_t=prompt.p_t, n_layers=ctx.n_layers, universe=ctx.universe)
            assert effects["max_identity_error"] < 1e-9
            assert effects["residual_sum_error"] < 1e-5
            assert effects["max_model_gap"] < 1e-4
            pm.check_direct_effects(effects, where=prompt.key)
            per = effects["per_noun"]["n0"]
            assert set(per) >= {"EMBED", "L00.H00", "L01.MLP", "b_O.L0", "beta", "c_reconstructed", "c_model"}


def test_site_axis_maps_the_development_pair_to_plus_minus_one():
    a = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 3.0])]
    b = [torch.tensor([2.0, 1.0]), torch.tensor([2.0, 3.0])]
    axis = pm.site_axis("x", a, b)
    assert axis.sigma == pytest.approx(1.0)
    assert [round(axis.n(v), 6) for v in a] == [-1.0, -1.0]
    assert [round(axis.n(v), 6) for v in b] == [1.0, 1.0]
    with pytest.raises(pm.IncidentError):
        pm.site_axis("zero", a, a)


def test_a2_position_map_and_a3_profile(ctx):
    a2 = pm.a2_position_map(ctx)
    assert a2["prefix_identity"] == "verified"
    assert set(a2["roles"]) == {"p_c", "p_t"}
    assert len(a2["roles"]["p_t"]) == len(ctx.universe) and len(a2["roles"]["p_c"]) == len(ctx.universe)
    assert a2["roles"]["p_t"]["L00.MLP"]["templates"]["coordinated-adjective"] == pytest.approx(0.0, abs=1e-9)
    assert len(a2["rankings"]["p_t_coordinated"]) == len(ctx.universe)
    a3 = pm.a3_layer_profile(ctx)
    assert a3["recovery_by_layer"]["L0"]["recovery"] == pytest.approx(1.0, abs=1e-6)
    assert "L1" in a3["recovery_by_layer"]


def test_a4_attention_reports_every_head(ctx):
    a4 = pm.a4_attention(ctx)
    assert len(a4["heads"]) == ctx.n_layers * int(ctx.model.cfg.n_heads)
    for entry in a4["heads"].values():
        assert 0.0 <= entry["attention_to_cue"] <= 1.0
        assert len(entry["attention_to_cue_by_prompt"]) == 4


def test_a5_contributions_sum_and_encoding_branch(ctx):
    a5 = pm.a5_axes_and_direct_effects(ctx, e_keys=["L00.MLP"], t_keys=["L01.H00"], r_keys=["L01.MLP"], l_r=1, l_t=1)
    contributions = a5["contributions_at_L_R"]["all"]
    assert contributions["total"] == pytest.approx(contributions["axis_projection"], abs=1e-6)
    assert set(contributions["entries"]) == {"EMBED", "L00.H00", "L00.H01", "L00.MLP"}
    encoding = a5["encoding_decomposition"]
    assert encoding["layer"] == 1
    assert encoding["embedding_share"] + sum(v for k, v in encoding["entries"].items() if k != "EMBED") / encoding["axis_norm"] == pytest.approx(1.0, abs=1e-6)
    assert set(a5["site_axes"]) == {"E", "T", "R_in", "R_out"}
    for label, values in a5["number_variables"].items():
        assert len(values) == (4 if label == "T" else 12)
    shares = a5["direct_effects"]["pair_shares_by_template"]["cardinal"]
    assert "beta" in shares and "EMBED" in shares
    assert "_site_axes" in a5


def test_a6_chain_returns_fractions(ctx):
    a5 = pm.a5_axes_and_direct_effects(ctx, e_keys=["L00.MLP"], t_keys=["L01.H00"], r_keys=["L01.MLP"], l_r=1, l_t=1)
    a6 = pm.a6_chain(ctx, a5["_site_axes"], e_keys=["L00.MLP"], t_keys=["L01.H00"], t_candidates=["L01.H00", "L01.H01", "L00.H00"], r_keys=["L01.MLP"], l_t=1)
    assert set(a6["e_alone"]) >= {"recovery", "m_R", "m_T", "m_T_by_head"}
    assert set(a6["e_alone_t_frozen"]) == {"L01.H00", "L01.H01", "L00.H00", "top3"}
    assert a6["residual_patch"]["recovery_unfrozen"] is not None
    assert set(a6["t_alone"]["cumulative"]) == {"top1", "top2", "top3"}
    assert isinstance(a6["r_alone"]["recovery"], float)


def test_a7_a8_a9_run_end_to_end(ctx):
    a7 = pm.a7_abstractness(ctx, e_keys=["L00.MLP"], t_candidates=["L01.H00"], r_keys=["L01.MLP"])
    assert set(a7["cross_cue"]) == {"cardinal<-quantifier", "quantifier<-cardinal"}
    assert set(a7["cross_frame"]) == {"E", "R", "T:L01.H00"}
    assert set(a7["cross_frame"]["E"]["cardinal"]) == {"opposite_cue", "same_cue"}
    a8 = pm.a8_neutralization(ctx, e_keys=["L00.MLP"], t_candidates=["L01.H00"], r_keys=["L01.MLP"])
    entry = a8["T:L01.H00"]["templates"]["coordinated-adjective"]
    assert set(entry) >= {"loss", "sign_retention", "compensation_ratio", "top_compensators", "clean_direct_effect_removed"}
    assert entry["sign_retention"]["total"] == 8
    assert "candidate_set" in a8
    a9 = pm.a9_isolation(ctx, [("L00.MLP", "p_c"), ("L01.H00", "p_t"), ("L01.MLP", "p_t")])
    assert a9["sign_retention"]["total"] == 24
    assert set(a9["faithfulness"]) == {"overall", "templates", "rule_classes"}


def test_isolation_neutralizes_exactly_the_complement(ctx):
    frame = ctx.coordinated[0]
    runs = pm.isolation_runs(ctx.model, ctx.cache, frame, [("L00.MLP", "p_c"), ("L01.H00", "p_t")])
    touched = {item["site"] for item in runs["sg"].integrity}
    universe = set(ctx.universe)
    expected = {f"{key}@{frame.p_c}" for key in universe - {"L00.MLP"}} | {f"{key}@{frame.p_t}" for key in universe - {"L01.H00"}}
    assert touched == expected


def test_neutralization_rows_measure_loss(ctx):
    frame = ctx.frames[0]
    runs = pm.neutralized_runs(ctx.model, ctx.cache, frame, [("L00.MLP", "p_t")])
    rows = pm.neutralization_rows(ctx.cache, frame, runs)
    sg, pl = pm.frame_prompts(frame)
    for row in rows:
        full = ctx.cache.c(sg)[row.lexical_key] - ctx.cache.c(pl)[row.lexical_key]
        assert row.d_full == pytest.approx(full)
        assert row.d_patch == pytest.approx(full - (pm.contrasts(runs["sg"].logits, ctx.nouns)[row.lexical_key] - pm.contrasts(runs["pl"].logits, ctx.nouns)[row.lexical_key]))
