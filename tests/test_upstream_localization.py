"""Experiment 022's module: the pure core (tier A) — constants, frozen inputs, the Shapley kernel and its direct
check, the exact order-statistic envelopes, the four-way classification, the gap rule, I3, the CDF percentile, the
committed table byte format and the SHA-indexed draws."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest
import torch

from neural_decompiler import upstream_localization as ul

# ---------------------------------------------------------------------------
# Constants and frozen inputs.


def test_constants_are_the_frozen_design():
    assert ul.DESIGN == {"path": "docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md", "revision": 3, "commit": "b0c7382"}
    assert ul.PLAN["revision"] == 2 and ul.PLAN["commit"] == "fb26a23"
    assert ul.FACTORS == ("R", "emb", "Bv", "Bp", "T") and ul.BIT == {"R": 1, "emb": 2, "Bv": 4, "Bp": 8, "T": 16}
    assert ul.N_MASKS == 32 and ul.FULL_MASK == 31 and ul.CUE_FINAL_MASKS == tuple(range(16)) and ul.CUE_FINAL_FULL_MASK == 15
    assert ul.SHAPLEY_WEIGHTS == (Fraction(1, 5), Fraction(1, 20), Fraction(1, 30), Fraction(1, 20), Fraction(1, 5))
    assert ul.B == 10_000 and ul.CROSS_CHECK_DRAWS == 16 and ul.SLOTS == {"cue": 6, "frame": 6}
    assert (ul.GUARD_C1_MIN, ul.GUARD_C2_MAX, ul.GUARD_C3_RANGE, ul.GUARD_C4_EXCLUSIVE_MIN) == (0.50, 0.10, (0.10, 0.90), 0.0)
    assert ul.GAP_MIN == 0.02
    assert ul.TOLERANCES == {"I1": 1e-4, "I2": 1e-12, "I3": 1e-4, "I4": 1e-3, "I5": 0.0, "I6": 1e-12, "R1": 1e-9, "kernel": 1e-10}
    assert ul.I3_FLOOR == 1e-12
    assert ul.RESULTS == ("NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS")
    assert ul.CLAIM_GROUP == {"C1": "cue_final", "C2": "cue_final", "C3": "coordinated", "C4": "coordinated"}
    assert ul.CLAIM_WORDING["C4"] == "block-0 cue→target attention contributes positively to the coordinated gap"
    assert ul.P_C_RANGE == {"cardinal": (3, 5), "quantifier": (3, 5), "coordinated-adjective": (4, 7)}
    assert ul.CUE_QUOTA == 6 and ul.FRAME_QUOTA == 6


def test_candidate_lists_are_the_designs_verbatim():
    assert ul.CUE_CANDIDATES["determiner-like"][:6] == ("alternate", "random", "standard", "regular", "normal", "common")
    assert ul.CUE_CANDIDATES["quantity"][:6] == ("bulk", "spare", "dense", "lengthy", "lots", "loads")
    assert ul.CUE_CANDIDATES["possessive-or-pronoun"] == ("thou", "yourself", "themselves", "ones", "others", "naught", "whatsoever")
    assert ul.CUE_CANDIDATES["adjective"][:6] == ("square", "wild", "brave", "calm", "eager", "fierce")
    assert [len(words) for words in ul.CUE_CANDIDATES.values()] == [11, 12, 7, 18]
    assert ul.FRAME_CANDIDATES["cardinal"][0] == "The pantry keeps {cue}" and len(ul.FRAME_CANDIDATES["cardinal"]) == 10
    assert ul.FRAME_CANDIDATES["quantifier"][5] == "The appendix cites {cue}" and len(ul.FRAME_CANDIDATES["quantifier"]) == 8
    assert ul.FRAME_CANDIDATES["coordinated-adjective"][5] == "Nora and Paul rinsed {cue} dry" and len(ul.FRAME_CANDIDATES["coordinated-adjective"]) == 9


def test_frozen_module_blobs_verify_and_a_changed_module_is_refused(monkeypatch):
    assert ul.assert_frozen_blobs() == ul.FROZEN_BLOBS
    monkeypatch.setitem(ul.FROZEN_BLOBS, "head_pattern.py", "0" * 40)
    with pytest.raises(ul.PhaseError, match="head_pattern.py"):
        ul.assert_frozen_blobs()


# ---------------------------------------------------------------------------
# Coalitions and the five-factor game.


def test_masks_and_the_cue_final_canonicalization():
    assert ul.mask_of(("R", "T")) == 17 and ul.names_of(21) == ("R", "Bv", "T")
    assert all(ul.mask_of(ul.names_of(mask)) == mask for mask in range(32))
    assert [ul.canonical_mask(mask, cue_final=True) for mask in (16, 17, 31)] == [0, 1, 15]
    assert [ul.canonical_mask(mask, cue_final=False) for mask in (16, 17, 31)] == [16, 17, 31]


def _game(values):
    """A game given by v(m) as SSE with SST = 1 (v = 1 − SSE)."""
    v = torch.tensor([values], dtype=torch.float64)
    return ul.group_statistics(1.0 - v, torch.ones(1, dtype=torch.float64))


def _permutation_shapley(values, players):
    phi = {player: 0.0 for player in players}
    orders = list(itertools.permutations(players))
    for order in orders:
        mask = 0
        for player in order:
            bit = ul.BIT[player]
            phi[player] += values[mask | bit] - values[mask]
            mask |= bit
    return {player: value / len(orders) for player, value in phi.items()}


def test_shapley_efficiency_symmetry_and_known_games():
    generator = torch.Generator().manual_seed(22)
    for _ in range(20):
        values = [0.0] + torch.rand(31, generator=generator, dtype=torch.float64).tolist()
        stats = _game(values)
        assert abs(float(stats["phi"].sum()) - float(stats["gap"])) <= 1e-13
        expected = _permutation_shapley(values, ul.FACTORS)
        assert all(abs(float(stats["phi"][0, i]) - expected[name]) <= 1e-13 for i, name in enumerate(ul.FACTORS))
    additive = {"R": 0.3, "emb": 0.01, "Bv": 0.2, "Bp": 0.15, "T": 0.05}
    stats = _game([sum(value for name, value in additive.items() if mask & ul.BIT[name]) for mask in range(32)])
    assert torch.allclose(stats["phi"][0], torch.tensor(list(additive.values()), dtype=torch.float64), atol=1e-15)
    stats = _game([1.0 if mask == 31 else 0.0 for mask in range(32)])  # pure interaction: equal split
    assert torch.allclose(stats["phi"][0], torch.full((5,), 0.2, dtype=torch.float64), atol=1e-15)
    swap = {ul.BIT["Bv"]: ul.BIT["Bp"], ul.BIT["Bp"]: ul.BIT["Bv"]}
    values = [0.0] + torch.rand(31, generator=generator, dtype=torch.float64).tolist()
    symmetric = [0.5 * (values[m] + values[(m & ~12) | sum(swap[b] for b in (4, 8) if m & b)]) for m in range(32)]
    stats = _game(symmetric)
    assert abs(float(stats["phi"][0, 2]) - float(stats["phi"][0, 3])) <= 1e-14  # Bv and Bp are interchangeable


def test_cue_final_t_is_an_exact_null_player_and_the_other_four_are_the_four_player_game():
    generator = torch.Generator().manual_seed(4)
    four = [0.0] + torch.rand(15, generator=generator, dtype=torch.float64).tolist()  # masks 0…15 over R, emb, Bv, Bp
    five = [four[ul.canonical_mask(mask, cue_final=True)] for mask in range(32)]  # the stored number of m | T is that of m
    stats = _game(five)
    assert float(stats["phi"][0, 4]) == 0.0  # φ_T exactly zero, not merely small
    expected = _permutation_shapley(four + [0.0] * 16, ("R", "emb", "Bv", "Bp"))  # the four-player game on masks 0…15
    for i, name in enumerate(("R", "emb", "Bv", "Bp")):
        assert abs(float(stats["phi"][0, i]) - expected[name]) <= 1e-12
    assert abs(float(stats["phi"][0].sum()) - (four[15] - four[0])) <= 1e-13


def _random_tables(generator, pairs=12, nouns=7, offset=0.0):
    measured = offset + torch.randn(pairs, nouns, generator=generator, dtype=torch.float64)
    predicted = measured.unsqueeze(1) + 0.3 * torch.randn(pairs, 32, nouns, generator=generator, dtype=torch.float64) * torch.linspace(1.0, 0.1, 32, dtype=torch.float64).view(1, 32, 1)
    return measured, predicted


def _cells(measured, predicted):
    sse = ((measured.unsqueeze(1) - predicted) ** 2).sum(dim=-1)
    count = torch.full((measured.shape[0],), float(measured.shape[1]), dtype=torch.float64)
    mean = measured.mean(dim=1)
    m2 = ((measured - mean.unsqueeze(1)) ** 2).sum(dim=1)
    return sse, count, mean, m2


@pytest.mark.parametrize("offset", [0.0, 1e4])
def test_the_kernel_agrees_with_the_direct_permutation_formula_with_multiplicities(offset):
    generator = torch.Generator().manual_seed(7)
    measured, predicted = _random_tables(generator, offset=offset)
    sse, count, mean, m2 = _cells(measured, predicted)
    index = torch.tensor([[0, 1, 2, 2, 5, 7, 7, 7, 11]], dtype=torch.int64)  # repeats count with multiplicity
    group_sse, _, _, group_m2 = ul.group_sums(sse, count, mean, m2, index)
    kernel = ul.group_statistics(group_sse, group_m2)
    direct = ul.direct_group_statistics(measured[index[0]], predicted[index[0]])
    assert all(ul.agreement(float(kernel["v"][0, m]), direct["v"][m]) <= 1e-10 for m in range(32))
    assert ul.agreement(float(kernel["gap"][0]), direct["gap"]) <= 1e-10
    assert all(ul.agreement(float(kernel["phi"][0, i]), direct["phi"][i]) <= 1e-10 for i in range(5))
    if direct["interpretable"]:
        assert all(ul.agreement(float(kernel["shares"][0, i]), direct["shares"][i]) <= 1e-10 for i in range(5))
    assert float(kernel["efficiency"][0]) <= 1e-12


def test_group_sums_are_hierarchical_so_pre_aggregated_cells_give_the_same_game():
    generator = torch.Generator().manual_seed(11)
    measured, predicted = _random_tables(generator, pairs=10)
    sse, count, mean, m2 = _cells(measured, predicted)
    flat = ul.group_sums(sse, count, mean, m2, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))
    halves = [ul.group_sums(sse, count, mean, m2, torch.tensor([[0, 1, 2, 3, 4]])), ul.group_sums(sse, count, mean, m2, torch.tensor([[5, 6, 7, 8, 9]]))]
    cells = [torch.cat([h[k] for h in halves]) for k in range(4)]
    nested = ul.group_sums(cells[0], cells[1], cells[2], cells[3], torch.tensor([[0, 1]]))
    for a, b in zip(flat, nested):
        assert torch.allclose(a, b, rtol=1e-13, atol=1e-12)


def test_the_gap_rule_is_on_the_aggregate_and_removes_nothing():
    sse = torch.tensor([[1.0] * 31 + [0.98]], dtype=torch.float64)
    sse_copy = sse.clone()
    exactly = ul.group_statistics(sse, torch.ones(1, dtype=torch.float64))  # G = 0.02 exactly (0.98 = 1 − 0.02)
    assert abs(float(exactly["gap"][0]) - 0.02) < 1e-15
    assert bool(exactly["interpretable"][0]) == (float(exactly["gap"][0]) >= 0.02)
    below = ul.group_statistics(torch.tensor([[1.0] * 31 + [0.99]], dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    assert not bool(below["interpretable"][0]) and torch.isnan(below["shares"][0]).all()
    zero_sst = ul.group_statistics(sse, torch.zeros(1, dtype=torch.float64))
    assert not bool(zero_sst["interpretable"][0])
    assert torch.equal(sse, sse_copy)  # the rule reads the aggregate; it alters no unit


# ---------------------------------------------------------------------------
# Envelopes.


def test_ranks_and_elements_at_b_10000():
    assert (ul.lower_rank(10_000), ul.upper_rank(10_000), ul.c3_low_rank(10_000), ul.c3_high_rank(10_000)) == (250, 9751, 125, 9876)
    values = torch.arange(1, 10_001, dtype=torch.float64)[torch.randperm(10_000, generator=torch.Generator().manual_seed(3))]
    defined = torch.ones(10_000, dtype=torch.bool)
    c1, c2, c3, c4 = (ul.claim_envelope(claim, values, defined) for claim in ul.CLAIMS)
    assert (c1["kind"], c1["rank"], c1["element"], c1["bound"]) == ("lower", 250, 249, 250.0)
    assert (c4["kind"], c4["rank"], c4["element"], c4["bound"]) == ("lower", 250, 249, 250.0)
    assert (c2["kind"], c2["rank"], c2["element"], c2["bound"]) == ("upper", 9751, 9750, 9751.0)
    assert c2["bound"] != c1["bound"]  # C2 is never computed from the lower tail
    assert (c3["kind"], c3["ranks"], c3["elements"], c3["low"], c3["high"]) == ("two-sided", [125, 9876], [124, 9875], 125.0, 9876.0)


def test_undefined_values_count_against_the_envelope_and_the_stop_is_at_exactly_250_and_125():
    values = torch.arange(1, 10_001, dtype=torch.float64)
    assert ul.undefined_threshold("C1", 10_000) == ul.undefined_threshold("C2", 10_000) == ul.undefined_threshold("C4", 10_000) == 250
    assert ul.undefined_threshold("C3", 10_000) == 125
    for count, degenerate in ((249, False), (250, True)):
        defined = torch.ones(10_000, dtype=torch.bool)
        defined[5000:5000 + count] = False
        assert math.isinf(ul.claim_envelope("C1", values, defined)["bound"]) is degenerate
        assert math.isinf(ul.claim_envelope("C2", values, defined)["bound"]) is degenerate
        assert (count >= ul.undefined_threshold("C1", 10_000)) is degenerate
    for count, degenerate in ((124, False), (125, True)):
        defined = torch.ones(10_000, dtype=torch.bool)
        defined[:count] = False
        envelope = ul.claim_envelope("C3", values, defined)
        assert math.isinf(envelope["low"]) is degenerate and math.isinf(envelope["high"]) is degenerate
        assert (count >= ul.undefined_threshold("C3", 10_000)) is degenerate


def test_direction_checks_use_the_median_of_defined_draws_and_catch_a_reversed_tail():
    values = torch.arange(1, 10_001, dtype=torch.float64)
    defined = torch.ones(10_000, dtype=torch.bool)
    defined[-100:] = False  # the median ignores undefined draws
    assert ul.defined_median(values, defined) == 4950.5
    for claim in ul.CLAIMS:
        envelope = ul.claim_envelope(claim, values, defined)
        assert ul.direction_check(claim, envelope, values, defined)["ok"]
    reversed_c2 = {"kind": "upper", "bound": float(torch.sort(values).values[249])}
    assert not ul.direction_check("C2", reversed_c2, values, defined)["ok"]
    assert not ul.direction_check("C1", {"kind": "lower", "bound": 9751.0}, values, defined)["ok"]


# ---------------------------------------------------------------------------
# The four-way result.


def test_classify_precedence_and_every_boundary():
    lower = {"kind": "lower", "bound": 0.4}
    assert ul.classify("C1", 0.2, False, lower) == "NOT_INTERPRETABLE"  # precedence: interpretability first
    assert ul.classify("C1", 0.50, True, lower) == "PASS"  # guard inclusive
    assert ul.classify("C1", math.nextafter(0.50, 0.0), True, lower) == "GUARD_FAILURE"
    assert ul.classify("C1", 0.6, True, {"kind": "lower", "bound": 0.7}) == "ENVELOPE_ONLY_FAILURE"
    assert ul.classify("C1", 0.7, True, {"kind": "lower", "bound": 0.7}) == "PASS"  # envelope inclusive
    assert ul.classify("C1", 0.3, True, {"kind": "lower", "bound": 0.7}) == "GUARD_FAILURE"  # guard before envelope
    upper = {"kind": "upper", "bound": 0.2}
    assert ul.classify("C2", 0.10, True, upper) == "PASS"
    assert ul.classify("C2", math.nextafter(0.10, 1.0), True, upper) == "GUARD_FAILURE"
    assert ul.classify("C2", 0.05, True, {"kind": "upper", "bound": 0.01}) == "ENVELOPE_ONLY_FAILURE"
    assert ul.classify("C2", 0.01, True, {"kind": "upper", "bound": 0.01}) == "PASS"
    band = {"kind": "two-sided", "low": 0.05, "high": 0.95}
    assert ul.classify("C3", 0.10, True, band) == "PASS" and ul.classify("C3", 0.90, True, band) == "PASS"
    assert ul.classify("C3", math.nextafter(0.10, 0.0), True, band) == "GUARD_FAILURE"
    assert ul.classify("C3", math.nextafter(0.90, 1.0), True, band) == "GUARD_FAILURE"
    assert ul.classify("C3", 0.5, True, {"kind": "two-sided", "low": 0.6, "high": 0.8}) == "ENVELOPE_ONLY_FAILURE"
    assert ul.classify("C4", 0.0, True, {"kind": "lower", "bound": -0.1}) == "GUARD_FAILURE"  # strict > 0
    assert ul.classify("C4", 5e-324, True, {"kind": "lower", "bound": -0.1}) == "PASS"
    assert ul.classify("C4", 0.05, True, {"kind": "lower", "bound": 0.08}) == "ENVELOPE_ONLY_FAILURE"
    with pytest.raises(ul.IncidentError):
        ul.classify("C1", float("nan"), True, lower)
    assert ul.classify("C1", None, False, lower) == "NOT_INTERPRETABLE"


def test_guard_bound_is_recorded_only_where_the_guard_is_stricter():
    assert ul.guard_bound("C1", {"bound": 0.45}) and not ul.guard_bound("C1", {"bound": 0.8})
    assert ul.guard_bound("C2", {"bound": 0.2}) and not ul.guard_bound("C2", {"bound": 0.01})
    assert ul.guard_bound("C3", {"low": 0.05, "high": 0.5}) and not ul.guard_bound("C3", {"low": 0.3, "high": 0.6})
    assert ul.guard_bound("C4", {"bound": -0.01}) and not ul.guard_bound("C4", {"bound": 0.05})


# ---------------------------------------------------------------------------
# I3, the CDF percentile, the table format, the draws.


def test_i3_is_the_maximum_per_position_l2_relative_error_with_the_frozen_floor():
    measured = {5: torch.tensor([3.0, 4.0], dtype=torch.float64), 6: torch.tensor([0.0, 2.0], dtype=torch.float64)}
    predicted = {5: torch.tensor([3.0, 4.5], dtype=torch.float64), 6: torch.tensor([0.0, 2.2], dtype=torch.float64)}
    assert ul.i3_error(predicted, measured) == pytest.approx(max(0.5 / 5.0, 0.2 / 2.0))
    assert ul.i3_error({3: torch.tensor([1e-13])}, {3: torch.tensor([0.0])}) == pytest.approx(1e-13 / 1e-12)
    with pytest.raises(ul.IncidentError):
        ul.i3_error({3: torch.zeros(2)}, {3: torch.zeros(2), 4: torch.zeros(2)})


def test_cdf_percentile_counts_defined_draws_at_or_below_the_fresh_value():
    values = torch.tensor([0.1, 0.2, 0.2, 0.4, float("nan")], dtype=torch.float64)
    defined = torch.tensor([True, True, True, True, False])
    assert ul.cdf_percentile(values, defined, 0.2) == {"cdf_percentile": 0.75, "defined": 4, "undefined": 1}
    assert ul.cdf_percentile(values, defined, None)["cdf_percentile"] is None


def test_the_table_byte_format_round_trips_bitwise_and_detects_one_flipped_bit(tmp_path):
    blocks = [("cue_final", torch.randn(4, 16, 3, dtype=torch.float64)), ("coordinated", torch.randn(2, 32, 3, dtype=torch.float64))]
    first = ul.write_table(tmp_path / "t.f64", tmp_path / "t.json", blocks, {"pairs": ["a", "b"]})
    second = ul.write_table(tmp_path / "u.f64", tmp_path / "u.json", blocks, {"pairs": ["a", "b"]})
    assert first["file_sha256"] == second["file_sha256"] and (tmp_path / "t.f64").read_bytes() == (tmp_path / "u.f64").read_bytes()
    assert [entry["offset_bytes"] for entry in first["blocks"]] == [0, 4 * 16 * 3 * 8]
    back = ul.read_table(tmp_path / "t.f64", first)
    assert all(torch.equal(back[name], block) for name, block in blocks)
    data = bytearray((tmp_path / "t.f64").read_bytes())
    data[17] ^= 0x01
    (tmp_path / "t.f64").write_bytes(bytes(data))
    with pytest.raises(ul.IncidentError):
        ul.read_table(tmp_path / "t.f64", first)


def test_the_draw_index_has_golden_values_and_is_reproducible():
    assert [ul.slot_index(b, s, i, n) for b, s, i, n in [(0, "cue/quantity", 0, 45), (9999, "frame/coordinated-adjective", 5, 14), (17, "cue/adjective", 3, 49),
                                                         (4242, "frame/cardinal", 2, 14)]] == [24, 10, 14, 0]
    first = ul.draw_indices({"cue/quantity": 45, "frame/cardinal": 14}, {"cue/quantity": 6, "frame/cardinal": 6}, 50)
    second = ul.draw_indices({"cue/quantity": 45, "frame/cardinal": 14}, {"cue/quantity": 6, "frame/cardinal": 6}, 50)
    assert all(torch.equal(first[key], second[key]) for key in first) and first["cue/quantity"].shape == (50, 6)
    assert int(first["cue/quantity"].max()) < 45 and int(first["frame/cardinal"].max()) < 14


# ---------------------------------------------------------------------------
# The pinned model (tier C; opt-in): exposed pairs only — the new chains against the committed ones, I1–I5, the
# cue-final null player on real compositions, and the R1 pre-calibration check against Experiment 021's table.

import os  # noqa: E402
from pathlib import Path  # noqa: E402

from neural_decompiler import head_pattern as hp  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def pinned():
    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")
    from neural_decompiler.models import PYTHIA_70M, load_model, seed_runtime

    inputs = ul.load_frozen_inputs(ROOT)
    seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
    model = load_model(PYTHIA_70M)
    return model, inputs, ul.ModelPrograms.from_model(model, inputs)


def _exposed_sample(inputs, frames_per_template=2, cues_per_class=2):
    """A few exposed pairs of every template (all in Experiment 020's ledger): frames by frame_id, pool cues by class."""
    pools = rc.production_pools(inputs.pool)
    cues = [entry for members in pools.cues.values() for entry in members[:cues_per_class]]
    frames = []
    for template in ul.TEMPLATES:
        frames += sorted((frame for frame in inputs.pool.frames if frame.template_id == template), key=lambda frame: frame.frame_id)[:frames_per_template]
    locked = inputs.closure["exploration"]["locked_states"]
    for frame in frames:
        state = rd.state_from_locked(locked[frame.frame_id], frame)
        for word, token_id in cues:
            yield frame, state, word, int(token_id)


@pytest.mark.pythia_smoke
def test_the_new_chains_equal_the_committed_ones_bit_for_bit(pinned):
    _, inputs, progs = pinned
    for frame, state, word, token_id in _exposed_sample(inputs):
        ctx = ul.pair_context(progs, frame, state, inputs.pool.reference_ids[frame.template_id], token_id, word)
        s17 = state.state_017
        committed = hp.exact_chain(progs.programs, progs.lw, s17.x1_all, s17.x2_all, s17.x3_all, frame.p_c, frame.p_t, ctx.factors.delta_e)["dx3"]
        mine = ul.exact_chain_multi(progs.programs, progs.lw, s17.x1_all, s17.x2_all, frame.p_c, frame.p_t, {frame.p_c: ctx.factors.delta_e})
        assert set(committed) == set(mine) and all(torch.equal(committed[p], mine[p]) for p in committed)
        level0 = rd.predicted_dx3(progs.chain, progs.weights, state, ctx.rows16, token_id, frame.template_id)
        reduced = ul.reduced_chain(progs.chain, ctx.rows16, s17.x1_all, s17.x2_all, frame.p_c, frame.p_t, frame.template_id, ctx.factors.delta_e)
        assert set(level0) == set(reduced) and all(torch.equal(level0[p], reduced[p]) for p in level0)  # I5, exactly


@pytest.mark.pythia_smoke
def test_the_identity_gates_hold_on_exposed_pairs_and_t_is_null_in_cue_final_frames(pinned):
    model, inputs, progs = pinned
    worst = {"I1": 0.0, "I2": 0.0, "I3": 0.0, "I4": 0.0, "I5": 0.0}
    for frame, state, word, token_id in _exposed_sample(inputs, frames_per_template=1, cues_per_class=2):
        ctx = ul.pair_context(progs, frame, state, inputs.pool.reference_ids[frame.template_id], token_id, word)
        table, composed = ul.pair_compositions(progs, ctx)
        measurement = ul.measure_prompt(model, progs, frame, state, token_id, word)
        gates, ceiling = ul.pair_gates(progs, ctx, table, composed, measurement)
        worst = {name: max(worst[name], gates[name]) for name in worst}
        if ctx.cue_final:
            assert all(torch.equal(table[mask | 16], table[mask]) for mask in range(16))
        sse, count, mean, m2 = ul.pair_cells(measurement["dc"], table)
        assert sse.shape == (32,) and count == 79.0 and m2 > 0.0
    assert worst["I1"] <= ul.TOLERANCES["I1"] and worst["I2"] <= ul.TOLERANCES["I2"] and worst["I3"] <= ul.TOLERANCES["I3"]
    assert worst["I4"] <= ul.TOLERANCES["I4"] and worst["I5"] == 0.0


@pytest.mark.pythia_smoke
def test_r1_the_remeasured_exposed_dc_reproduces_021s_table_at_1e_9(pinned):
    """The pre-calibration check (plan revision 2, Q4): 022's capture site set must reproduce Experiment 021's
    digest-bound exposed measurements. A failure stops for review; the tolerance is never loosened."""
    model, inputs, progs = pinned
    path = ROOT / ul.EXPOSED_TABLE_021_RELATIVE_PATH
    if not path.exists():
        pytest.skip("Experiment 021's local exposed table is absent")
    table = torch.load(path)
    assert rc.tensor_digest(table["measured"]) == ul.INHERITED_021["exposed_measured_sha256"]
    row = {(cue, frame): index for index, (cue, frame) in enumerate(zip(table["cues"], table["frames"]))}
    worst = 0.0
    for frame, state, word, token_id in _exposed_sample(inputs, frames_per_template=2, cues_per_class=2):
        measurement = ul.measure_prompt(model, progs, frame, state, token_id, word)
        worst = max(worst, float((measurement["dc"] - table["measured"][row[(word, frame.frame_id)]]).abs().max()))
    assert worst <= ul.TOLERANCES["R1"], f"R1 pre-check failed: {worst:.3e}"


# ---------------------------------------------------------------------------
# The freeze on a stub tokenizer and a stub frozen world (tier A; the real freeze is a later, authorized phase).

import json  # noqa: E402
import re  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from neural_decompiler import plural_mechanism as pm  # noqa: E402


class StubTokenizer:
    """Whitespace pieces (with their leading space) mapped to ids; a few words split in two."""

    SPLIT = {" thou": (" th", "ou")}

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.pieces: dict[int, str] = {}

    def id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = 1000 + len(self.vocab)
            self.pieces[self.vocab[piece]] = piece
        return self.vocab[piece]

    def encode(self, text, add_special_tokens=False):
        out = []
        for piece in re.findall(r" ?[^ ]+", text):
            out += [self.id(part) for part in self.SPLIT.get(piece, (piece,))]
        return out

    def decode(self, ids):
        return "".join(self.pieces[int(i)] for i in ids)


def _stub_inputs(tokenizer, *, used_words=("alternate",), used_texts=("The studio makes {cue}",), planted=()):
    one, two, each, several = (tokenizer.id(" one"), tokenizer.id(" two"), tokenizer.id(" each"), tokenizer.id(" several"))
    cue_ids = {"cardinal": {"sg": one, "pl": two}, "quantifier": {"sg": each, "pl": several}, "coordinated-adjective": {"sg": one, "pl": two}}
    texts = {"cardinal": "The display contains {cue}", "quantifier": "The catalog lists {cue}", "coordinated-adjective": "Mira and Noah packed {cue} bright"}
    frames = tuple(pm._build_new_frame(tokenizer, template, text, cue_ids[template], f"{template}-1") for template, text in texts.items())
    frames = tuple(pm.Frame(f.template_id, f.frame_id, f.prefix_ids, f.suffix_ids, f.cue_ids, f.text_template, origin="manifest") for f in frames)
    names = {one: "one", two: "two", each: "each", several: "several"}
    tokens = tuple((word, tokenizer.id(" " + word)) for word in used_words)
    pool = SimpleNamespace(tokens=tokens + tuple((name, token_id) for token_id, name in names.items()), frames=frames,
                           frame_origin={frame.frame_id: "manifest" for frame in frames}, reference_ids={"cardinal": one, "quantifier": each, "coordinated-adjective": one},
                           plural_cue={"cardinal": "two", "quantifier": "several", "coordinated-adjective": "two"}, token_id=lambda name: {v: k for k, v in names.items()}[name])
    used_frame = pm._build_new_frame(tokenizer, "cardinal", used_texts[0], cue_ids["cardinal"], "cardinal-used-1") if used_texts else None
    confirmation = SimpleNamespace(tokens=tuple({"word": w, "token_id": tokenizer.id(" " + w)} for w in planted), frames=(used_frame,) if used_frame else (), content_sha256="c" * 64)
    extension = SimpleNamespace(original_frames=frames, new_frames=(), cue_words=(), reference_cue_ids={}, content_sha256="e" * 64)
    return ul.FrozenInputs(pool, {}, {}, {}, None, {}, {}, {"019": confirmation}, None, extension)


def test_the_freeze_takes_the_first_eligible_entries_and_records_every_rejection():
    tokenizer = StubTokenizer()
    inputs = _stub_inputs(tokenizer, planted=("wild",))
    payload = ul.freeze_payload(tokenizer, inputs, model={"model_id": "stub", "revision": "x"})
    words = {cls: [entry["word"] for entry in payload["cues"] if entry["class"] == cls] for cls in ul.CUE_CANDIDATES}
    assert words["determiner-like"] == ["random", "standard", "regular", "normal", "common", "general"]  # alternate already used
    assert words["possessive-or-pronoun"] == ["yourself", "themselves", "ones", "others", "naught", "whatsoever"]  # thou is two tokens here
    assert words["adjective"] == ["square", "brave", "calm", "eager", "fierce", "humble"]  # wild planted in a confirmation
    reasons = {(entry["candidate"], entry["reason"].split(" ")[0]) for entry in payload["rejected"]}
    assert ("alternate", "token") in reasons and ("wild", "token") in reasons and ("thou", "2") in reasons
    assert ("The studio makes {cue}", "text") in reasons
    cardinal = [entry for entry in payload["frames"] if entry["template_id"] == "cardinal"]
    assert [entry["frame_id"] for entry in cardinal] == [f"cardinal-022-{k}" for k in range(1, 7)] and cardinal[1]["text_template"] == "The market trades {cue}"
    coordinated = [entry for entry in payload["frames"] if entry["template_id"] == "coordinated-adjective"]
    assert all(entry["p_t"] == entry["p_c"] + 1 and 4 <= entry["p_c"] <= 7 for entry in coordinated)
    confirmation = ul.confirmation_from_payload(payload, inputs.pool)
    manifest = confirmation.manifest()
    assert len(manifest["S1-REF"]) == 18 and len(manifest["S1-VALIDITY"]) == 18
    assert len(manifest["S2-TARGET"]["Y1"]) == 24 * len(inputs.pool.frames) and len(manifest["S2-TARGET"]["Y2"]) == 24 * 18
    assert payload["counts"] == {"classes": {cls: 6 for cls in ul.CUE_CANDIDATES}, "templates": {template: 6 for template in ul.FRAME_CANDIDATES}}


def test_a_shortfall_writes_nothing_and_stops_for_review():
    tokenizer = StubTokenizer()
    inputs = _stub_inputs(tokenizer, used_words=("yourself", "themselves"))  # thou splits: ones, others, naught, whatsoever remain
    with pytest.raises(ul.FreezeShortfall, match="possessive-or-pronoun"):
        ul.freeze_payload(tokenizer, inputs)


def test_the_exclusion_is_structured_and_covers_every_source():
    tokenizer = StubTokenizer()
    inputs = _stub_inputs(tokenizer, planted=("bulk",))
    exclusion = ul.extract_exclusion(inputs)
    assert tokenizer.id(" bulk") in exclusion["cue_token_ids"] and tokenizer.id(" alternate") in exclusion["cue_token_ids"]
    assert tokenizer.id(" two") in exclusion["cue_token_ids"]  # a frame's own plural cue
    assert "The studio makes {cue}" in exclusion["frame_texts"] and "The display contains {cue}" in exclusion["frame_texts"]
    assert [entry["source"] for entry in exclusion["sources"]] == ["pool-020", "confirmation-019", "extension"]
    assert exclusion["cue_token_ids_sha256"] == pm.sha256_text(pm.canonical_json(exclusion["cue_token_ids"]))


def test_a_tampered_confirmation_file_is_refused(tmp_path, monkeypatch):
    tokenizer = StubTokenizer()
    inputs = _stub_inputs(tokenizer)
    payload = ul.freeze_payload(tokenizer, inputs)
    path = tmp_path / "confirmation-v1.json"
    path.write_text(pm.canonical_json(payload) + "\n")
    assert len(ul.load_confirmation_022(path, inputs).tokens) == 24
    for mutate in (lambda p: p["cues"][0].update(word="other"), lambda p: p["manifest"]["S1-REF"].pop(), lambda p: p.update(counts={})):
        changed = json.loads(path.read_text())
        mutate(changed)
        changed["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in changed.items() if k != "content_sha256"}))
        (tmp_path / "bad.json").write_text(pm.canonical_json(changed) + "\n")
        with pytest.raises(ul.PhaseError):
            ul.load_confirmation_022(tmp_path / "bad.json", inputs)
    stale = json.loads(path.read_text())
    stale["content_sha256"] = "0" * 64
    (tmp_path / "stale.json").write_text(pm.canonical_json(stale) + "\n")
    with pytest.raises(ul.PhaseError, match="digest"):
        ul.load_confirmation_022(tmp_path / "stale.json", inputs)
    later = _stub_inputs(tokenizer, planted=(payload["cues"][0]["word"],))  # a unit that became used after the freeze
    with pytest.raises(ul.PhaseError):
        ul.load_confirmation_022(path, later)


# ---------------------------------------------------------------------------
# The calibration's statistics on synthetic tables (tier A at small sizes; the full-scale dry run is marked slow).
# No model: the re-materialization loop itself is exercised by the runner tests on the fake world.

import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from neural_decompiler import readout_calibration as rc  # noqa: E402,F811

SIX = {"classes": {cls: 6 for cls in ul.CUE_CLASSES}, "templates": {template: 6 for template in ul.TEMPLATES}}


def _synthetic_units(classes=(5, 4, 3, 6), frames_per_template=4, y2_per_template=3, counts=SIX):
    cues_by_class = {cls: [(f"{cls[:3]}{i}", 1000 + 100 * c + (11 * i) % n) for i in range(n)] for c, (cls, n) in enumerate(zip(ul.CUE_CLASSES, classes))}
    frames = [SimpleNamespace(frame_id=f"{template}-{k:02d}", template_id=template) for template in reversed(ul.TEMPLATES) for k in range(frames_per_template)]
    y2 = {template: [f"{template}-{k:02d}" for k in reversed(range(frames_per_template - y2_per_template, frames_per_template))] for template in ul.TEMPLATES}
    return ul.units_from(cues_by_class, frames, y2, counts)


def _synthetic_table(units, nouns=79, seed=0):
    """Measured Δc and 32 coalitions whose error shrinks as factors are switched on; T is absent from cue-final
    frames, whose row m | T is the row of m (the canonical form)."""
    generator = torch.Generator().manual_seed(seed)
    c, f = len(units.cues), len(units.frames)
    coordinated = torch.tensor([frame.template_id == ul.COORDINATED for frame in units.frames])
    dc = torch.randn(c, f, nouns, generator=generator, dtype=torch.float64)
    residual = 0.05 * torch.randn(c, f, nouns, generator=generator, dtype=torch.float64)
    scale = {"R": 0.05, "emb": 0.03, "Bv": 0.3, "Bp": 0.2, "T": 0.25}
    errors = {name: scale[name] * torch.randn(c, f, nouns, generator=generator, dtype=torch.float64) for name in ul.FACTORS}
    errors["R"][:, coordinated] *= 6.0
    dc_hat = torch.empty(c, f, 32, nouns, dtype=torch.float64)
    for mask in range(32):
        dc_hat[:, :, mask] = dc + residual
        for name in ul.FACTORS:
            if not mask & ul.BIT[name] and name != "T":
                dc_hat[:, :, mask] += errors[name]
        if not mask & ul.BIT["T"]:
            dc_hat[:, coordinated, mask] += errors["T"][:, coordinated]
    for mask in range(16):
        dc_hat[:, ~coordinated, mask | 16] = dc_hat[:, ~coordinated, mask]
    sse = torch.empty(c, f, 32, dtype=torch.float64)
    count, mean, m2 = (torch.empty(c, f, dtype=torch.float64) for _ in range(3))
    for ci in range(c):
        for fi in range(f):
            sse[ci, fi], count[ci, fi], mean[ci, fi], m2[ci, fi] = ul.pair_cells(dc[ci, fi], dc_hat[ci, fi])
    gates = {name: torch.zeros(c, f, dtype=torch.float64) for name in ("I1", "I2", "I3", "I4", "I5")}
    return ul.CalibrationTable(dc, dc_hat, sse, count, mean, m2, dc + residual, gates)


def test_units_follow_the_frozen_order_and_the_draws_index_within_their_strata():
    units = _synthetic_units()
    assert [cls for _, _, cls in units.cues] == [cls for cls, n in zip(ul.CUE_CLASSES, (5, 4, 3, 6)) for _ in range(n)]
    for cls in ul.CUE_CLASSES:
        start, n = units.class_offsets[cls]
        ids = [token_id for _, token_id, _ in units.cues[start:start + n]]
        assert ids == sorted(ids)  # token-id order within the class (plan revision 2, Q6)
    assert [frame.frame_id for frame in units.frames] == sorted(frame.frame_id for frame in units.frames)
    assert all(units.frames[i].template_id == template for template, indices in units.y2_frames.items() for i in indices)
    assert [units.frames[i].frame_id for i in units.y2_frames["cardinal"]] == ["cardinal-01", "cardinal-02", "cardinal-03"]  # by frame_id
    assert sorted(units.groups["cue_final"] + units.groups["coordinated"]) == list(range(len(units.frames)))
    draws = ul.draw_units(units, 40)
    for cls in ul.CUE_CLASSES:
        start, n = units.class_offsets[cls]
        block = ul.CUE_CLASSES.index(cls) * 6
        assert all(int(draws["cue_index"][b, block + i]) == start + ul.slot_index(b, f"cue/{cls}", i, n) for b in (0, 7, 39) for i in range(6))
    for template in ul.TEMPLATES:
        pool = units.y2_frames[template]
        assert all(int(draws["frame_index"][template][b, i]) == pool[ul.slot_index(b, f"frame/{template}", i, len(pool))] for b in (0, 39) for i in range(6))
    assert set(draws["digests"]) == set(units.sizes())


def test_the_calibration_kernel_equals_the_direct_recomputation_with_duplicates_and_any_chunking():
    units = _synthetic_units()
    table = _synthetic_table(units)
    draws = ul.draw_units(units, 30)
    kernel = ul.kernel_statistics(table, units, draws, chunk=7)
    again = ul.kernel_statistics(table, units, draws, chunk=1000)
    for population in ul.POPULATIONS:
        for group in ul.GROUPS:
            assert all(torch.equal(kernel["stats"][population][group][key], again["stats"][population][group][key]) for key in ("v", "gap", "phi", "sst"))
    check = ul.kernel_direct_check(table, units, draws, kernel, n_draws=30)
    assert check["n_exceeding"] == 0 and check["max_difference"] <= 1e-10 and check["n_checked"] == 30 * 4 * (32 + 1 + 5 + 5 + 1)
    cues, frames = ul.draw_pairs(units, draws, "Y2", "cue_final", 0)
    assert cues.shape[0] == 24 * 12 and len(set(zip(cues.tolist(), frames.tolist()))) < 24 * 12  # duplicates are kept, with multiplicity
    cues, frames = ul.draw_pairs(units, draws, "Y1", "coordinated", 0)
    assert cues.shape[0] == 24 * len(units.groups["coordinated"])
    assert max(ul.efficiency_maxima(kernel).values()) <= ul.TOLERANCES["I6"]
    phi_t = kernel["stats"]["Y1"]["cue_final"]["phi"][:, 4]
    assert torch.equal(phi_t, torch.zeros_like(phi_t))  # the cue-final null player, exactly


def test_a_planted_kernel_disagreement_is_a_cross_check_incident():
    units = _synthetic_units()
    table = _synthetic_table(units)
    draws = ul.draw_units(units, 5)
    kernel = ul.kernel_statistics(table, units, draws)
    cues, frames = ul.draw_pairs(units, draws, "Y2", "coordinated", 2)
    table.dc_hat[cues[0], frames[0], 9] += 0.5  # the direct side now sees a different table
    with pytest.raises(ul.CrossCheckError) as caught:
        ul.kernel_direct_check(table, units, draws, kernel)
    assert caught.value.details["n_exceeding"] > 0 and "coordinated" in caught.value.details["at"]


def _claims(n, *, undefined=None, nonfinite=None, generator=None):
    generator = generator or torch.Generator().manual_seed(5)
    centers = {"C1": 0.9, "C2": 0.02, "C3": 0.4, "C4": 0.15}
    out = {}
    for population in ul.POPULATIONS:
        out[population] = {}
        for claim in ul.CLAIMS:
            values = centers[claim] + 0.05 * torch.randn(n, generator=generator, dtype=torch.float64)
            interpretable = torch.ones(n, dtype=torch.bool)
            k = (undefined or {}).get(f"{population}/{claim}", 0)
            interpretable[:k] = False
            values[:k] = float("nan")
            if nonfinite == f"{population}/{claim}":
                values[n - 1] = float("inf")
            out[population][claim] = {"value": values, "interpretable": interpretable, "gap": torch.full((n,), 0.1, dtype=torch.float64)}
    return out


def test_the_undefined_stop_is_at_exactly_250_and_125_and_a_non_finite_interpretable_share_is_an_incident():
    for key, count, stop in (("Y1/C1", 249, False), ("Y1/C1", 250, True), ("Y2/C2", 250, True), ("Y2/C4", 250, True), ("Y1/C3", 124, False), ("Y1/C3", 125, True)):
        claims = _claims(10_000, undefined={key: count})
        ul.assert_finite_where_interpretable(claims)
        result = ul.calibration_stop(ul.undefined_counts(claims), 10_000)
        assert result["stop"] is stop and (key in result["offending"]) is stop and result["counts"][key.split("/")[0]][key.split("/")[1]] == count
    with pytest.raises(ul.IncidentError, match="non-finite"):
        ul.assert_finite_where_interpretable(_claims(1000, nonfinite="Y2/C3"))


def test_envelopes_result_rates_and_a_reversed_tail(monkeypatch):
    claims = _claims(1000, undefined={"Y1/C2": 10})
    evaluated = ul.envelopes_and_rates(claims)
    assert evaluated["envelopes"]["Y1"]["C2"]["kind"] == "upper" and evaluated["envelopes"]["Y1"]["C2"]["rank"] == ul.upper_rank(1000)
    assert evaluated["summaries"]["Y1"]["C2"]["undefined"] == 10
    for population in ul.POPULATIONS:
        for claim in ul.CLAIMS:
            rates = evaluated["result_rates"][population][claim]
            assert set(rates) == set(ul.RESULTS) and abs(sum(rates.values()) - 1.0) < 1e-12
    assert evaluated["result_rates"]["Y1"]["C2"]["NOT_INTERPRETABLE"] == 0.01
    assert 0.0 <= evaluated["joint_rates"]["all_eight_pass"] <= min(evaluated["joint_rates"]["Y1_all_four_pass"], evaluated["joint_rates"]["Y2_all_four_pass"])
    original = ul.claim_envelope

    def reversed_c2(claim, values, defined):
        envelope = original(claim, values, defined)
        if claim == "C2":  # the classic slip: the upper bound taken from the lower tail
            envelope = {**envelope, "bound": ul.order_statistic(values, defined, ul.lower_rank(int(values.shape[0])), undefined_at=math.inf)}
        return envelope

    monkeypatch.setattr(ul, "claim_envelope", reversed_c2)
    with pytest.raises(ul.IncidentError, match="direction check"):
        ul.envelopes_and_rates(_claims(1000))


def test_gate_maxima_are_located_and_enforced_at_the_frozen_tolerances():
    units = _synthetic_units()
    shape = (len(units.cues), len(units.frames))
    gates = {name: torch.zeros(shape, dtype=torch.float64) for name in ("I1", "I2", "I3", "I4", "I5")}
    gates["I1"][2, 3] = 1e-4  # exactly at the tolerance passes
    summary = ul.gate_summary(gates, units)
    assert summary["I1"] == {"max": 1e-4, "at": f"{units.cues[2][0]}|{units.frames[3].frame_id}", "tolerance": 1e-4}
    ul.enforce_gates(summary)
    gates["I5"][0, 0] = 5e-324  # I5 is exact equality
    with pytest.raises(ul.IncidentError, match="I5"):
        ul.enforce_gates(ul.gate_summary(gates, units))
    gates["I5"][0, 0] = 0.0
    gates["I3"][1, 1] = float("nan")  # a missing value is never a pass
    summary = ul.gate_summary(gates, units)
    assert summary["I3"]["max"] is None
    with pytest.raises(ul.IncidentError, match="I3"):
        ul.enforce_gates(summary)


def _fake_021(tmp_path, monkeypatch, units, table, noun_keys):
    rows = [(word, frame.frame_id) for word, _, _ in units.cues for frame in units.frames]
    measured = torch.stack([table.dc[ci, fi] for ci in range(len(units.cues)) for fi in range(len(units.frames))])
    exposed = {"cues": [word for word, _ in rows], "frames": [frame for _, frame in rows], "noun_keys": list(noun_keys), "measured": measured.clone()}
    (tmp_path / "outputs").mkdir(exist_ok=True)
    torch.save(exposed, tmp_path / "outputs" / "exposed-table.pt")
    record = {"experiment": "021", "table_sha256": {"measured": rc.tensor_digest(measured)}}
    record["content_sha256"] = rc.content_digest(record)
    (tmp_path / "calibration-021.json").write_text(pm.canonical_json(record) + "\n")
    monkeypatch.setattr(ul, "CALIBRATION_021_RELATIVE_PATH", "calibration-021.json")
    monkeypatch.setattr(ul, "EXPOSED_TABLE_021_RELATIVE_PATH", "outputs/exposed-table.pt")
    monkeypatch.setitem(ul.INHERITED_021, "calibration_file_sha256", rc.file_sha256(tmp_path / "calibration-021.json"))
    monkeypatch.setitem(ul.INHERITED_021, "calibration_content_sha256", record["content_sha256"])
    monkeypatch.setitem(ul.INHERITED_021, "exposed_measured_sha256", record["table_sha256"]["measured"])
    return exposed


def test_r1_verifies_021s_record_and_table_before_comparing(tmp_path, monkeypatch):
    units = _synthetic_units()
    table = _synthetic_table(units, nouns=5)
    keys = [f"noun{i}" for i in range(5)]
    _fake_021(tmp_path, monkeypatch, units, table, keys)
    result = ul.r1_gate(tmp_path, table, units, keys)
    assert result["passed"] and result["max_difference"] == 0.0 and result["n_pairs"] == len(units.cues) * len(units.frames)
    table.dc[1, 2, 3] += 2e-9
    result = ul.r1_gate(tmp_path, table, units, keys)
    assert not result["passed"] and result["at"] == f"{units.cues[1][0]}|{units.frames[2].frame_id}" and result["max_difference"] == pytest.approx(2e-9, rel=1e-3)
    with pytest.raises(ul.IncidentError, match="noun"):
        ul.r1_gate(tmp_path, table, units, list(reversed(keys)))
    exposed = torch.load(tmp_path / "outputs" / "exposed-table.pt")
    exposed["measured"][0, 0] += 1.0  # the local table is not the one 021's record digests
    torch.save(exposed, tmp_path / "outputs" / "exposed-table.pt")
    with pytest.raises(ul.IncidentError, match="local exposed table"):
        ul.r1_gate(tmp_path, table, units, keys)
    (tmp_path / "calibration-021.json").write_text("{}\n")
    with pytest.raises(ul.IncidentError, match="committed calibration record"):
        ul.r1_gate(tmp_path, table, units, keys)


def _calibration_record(units, table, n_draws):
    draws = ul.draw_units(units, n_draws)
    kernel = ul.kernel_statistics(table, units, draws)
    cross = ul.kernel_direct_check(table, units, draws, kernel)
    ul.assert_finite_where_interpretable(kernel["claims"])
    undefined = ul.undefined_counts(kernel["claims"])
    stop = ul.calibration_stop(undefined, n_draws)
    evaluated = ul.envelopes_and_rates(kernel["claims"])
    arrays = ul.draw_arrays(kernel)
    record = ul.calibration_record(run_id="run", protocol_code_commit="c" * 40, digests={"manifest": "m"}, units=units, confirmation_sha256="f" * 64,
                                   gates=ul.gate_summary(table.gates, units), r1={"passed": True}, table_digests=table.digests(), draws=draws, cross_check=cross,
                                   efficiency=ul.efficiency_maxima(kernel), undefined=undefined, evaluated=evaluated, descriptives=ul.exposed_descriptives(table, units),
                                   array_digests={key: rc.tensor_digest(value) for key, value in arrays.items()}, n_draws=n_draws)
    return record, stop, kernel


def test_the_calibration_record_binds_the_frozen_constants_and_verifies():
    units = _synthetic_units()
    table = _synthetic_table(units)
    record, stop, _ = _calibration_record(units, table, 200)
    assert not stop["stop"]
    ul.verify_calibration_record(record, draws=200)
    assert record["constants"]["ranks"] == {"lower": 5, "upper": 196, "c3_low": 3, "c3_high": 198}
    assert record["descriptive_exposed"]["group/cue_final"]["phi"]["T"] == 0.0 and record["descriptive_exposed"]["group/cue_final"]["interpretable"]
    ladder = record["descriptive_exposed"]["group/coordinated"]["ladder"]
    assert list(ladder) == ["Level 0", "R", "R+emb", "R+emb+Bv+Bp", "all"] and ladder["all"] > ladder["Level 0"]
    with pytest.raises(ul.PhaseError, match="constants"):
        ul.verify_calibration_record(record, draws=ul.B)
    for mutate in (lambda r: r["constants"]["guards"].update(C1_min=0.4), lambda r: r["envelopes"]["Y2"].pop("C4"), lambda r: r["direction_checks"]["Y1"]["C1"].update(ok=False)):
        changed = json.loads(pm.canonical_json(record))
        mutate(changed)
        changed["content_sha256"] = rc.content_digest(changed)
        with pytest.raises(ul.PhaseError):
            ul.verify_calibration_record(changed, draws=200)
    stale = json.loads(pm.canonical_json(record))
    stale["envelopes"]["Y1"]["C1"]["bound"] = 0.0
    with pytest.raises(ul.PhaseError, match="digest"):
        ul.verify_calibration_record(stale, draws=200)


@pytest.mark.slow
def test_a_full_scale_synthetic_calibration_dry_run_at_b_10000():
    """Plan revision 2, Task 5: the draws, the kernel, the cross-check, the envelopes and the record at the real
    pool sizes (175 cues 45/45/36/49 × 108 frames, 14 Y2-like frames per template, B = 10,000), with no model."""
    started = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    units = _synthetic_units(classes=(45, 45, 36, 49), frames_per_template=36, y2_per_template=14)
    assert len(units.cues) == 175 and len(units.frames) == 108 and units.sizes() == {**{f"cue/{c}": n for c, n in zip(ul.CUE_CLASSES, (45, 45, 36, 49))},
                                                                                     **{f"frame/{t}": 14 for t in ul.TEMPLATES}}
    table = _synthetic_table(units)
    built = time.perf_counter()
    record, stop, kernel = _calibration_record(units, table, ul.B)
    finished = time.perf_counter()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux
    ul.verify_calibration_record(record)
    assert not stop["stop"] and record["cross_check"]["n_exceeding"] == 0 and record["cross_check"]["n_checked"] == 16 * 4 * 44
    assert kernel["stats"]["Y1"]["cue_final"]["v"].shape == (ul.B, 32) and kernel["stats"]["Y2"]["coordinated"]["phi"].shape == (ul.B, 5)
    assert all(record["direction_checks"][p][c]["ok"] for p in ul.POPULATIONS for c in ul.CLAIMS)
    print(f"\nfull-scale dry run: table {built - started:.1f} s, draws+kernel+check+envelopes+record {finished - built:.1f} s, "
          f"peak RSS {peak * scale / 2**30:.2f} GiB (was {before * scale / 2**30:.2f} GiB)")
    assert finished - built < 600 and peak * scale < 6 * 2**30


# ---------------------------------------------------------------------------
# The confirmation's scoring on synthetic fresh populations (tier A): the one kernel with D = 1, its direct check,
# and every one of the four results reachable through the frozen classification.


def _fresh_population(generator, *, cue_final_scale=1.0, coordinated_r=1.0, nouns=11):
    scale = {"R": 0.05, "emb": 0.03, "Bv": 0.3, "Bp": 0.2, "T": 0.25}
    blocks, tables = {}, {}
    for group, n, width in (("cue_final", 12, 16), ("coordinated", 6, 32)):
        dc = torch.randn(n, nouns, generator=generator, dtype=torch.float64)
        residual = 0.02 * torch.randn(n, nouns, generator=generator, dtype=torch.float64)
        errors = {name: value * torch.randn(n, nouns, generator=generator, dtype=torch.float64) for name, value in scale.items()}
        if group == "cue_final":
            errors = {name: value * cue_final_scale for name, value in errors.items()}
            errors["T"] = torch.zeros_like(errors["T"])
            residual = residual * cue_final_scale
        else:
            errors["R"] = errors["R"] * coordinated_r
        rows = torch.empty(n, width, nouns, dtype=torch.float64)
        for mask in range(width):
            rows[:, mask] = dc + residual
            for name in ul.FACTORS:
                if not mask & ul.BIT[name]:
                    rows[:, mask] += errors[name]
        blocks[group] = {"dc": dc}
        tables[group] = rows
    return {"units": None, "blocks": blocks}, tables


def test_score_022_reaches_every_result_through_the_locked_envelopes_and_guards():
    generator = torch.Generator().manual_seed(9)
    y1_measured, y1_tables = _fresh_population(generator, coordinated_r=4.0)  # Y1: the reductions hold a fair share of the coordinated gap
    y2_measured, y2_tables = _fresh_population(generator, cue_final_scale=1e-3, coordinated_r=40.0)  # Y2: no cue-final gap; R dominates the coordinated gap
    measured, tables = {"Y1": y1_measured, "Y2": y2_measured}, {"Y1": y1_tables, "Y2": y2_tables}
    lower = lambda bound: {"kind": "lower", "rank": 1, "element": 0, "bound": bound}  # noqa: E731
    upper = lambda bound: {"kind": "upper", "rank": 1, "element": 0, "bound": bound}  # noqa: E731
    band = {"kind": "two-sided", "ranks": [1, 1], "elements": [0, 0], "low": -10.0, "high": 10.0}
    envelopes = {"Y1/C1": lower(0.0), "Y1/C2": upper(-1.0), "Y1/C3": band, "Y1/C4": lower(10.0), "Y2/C1": lower(0.0), "Y2/C2": upper(1.0), "Y2/C3": band, "Y2/C4": lower(-10.0)}
    lock = {"conditions": {key: {"envelope": envelope, "guard": dict(ul.GUARDS[key.split("/")[1]])} for key, envelope in envelopes.items()}}
    scored = ul.score_022(measured, tables, lock)
    results = {key: entry["result"] for key, entry in scored["conditions"].items()}
    assert results == {"Y1/C1": "PASS", "Y1/C2": "ENVELOPE_ONLY_FAILURE", "Y1/C3": "PASS", "Y1/C4": "ENVELOPE_ONLY_FAILURE",
                       "Y2/C1": "NOT_INTERPRETABLE", "Y2/C2": "NOT_INTERPRETABLE", "Y2/C3": "GUARD_FAILURE", "Y2/C4": "PASS"}
    assert set(results.values()) == set(ul.RESULTS) and scored["aggregate_label"] is None
    assert scored["conditions"]["Y2/C1"]["value"] is None and scored["conditions"]["Y2/C3"]["value"] > 0.9 and scored["conditions"]["Y2/C3"]["direction"]["larger"].startswith("the layer-1")
    assert scored["cross_check"]["n_exceeding"] == 0 and scored["cross_check"]["n_checked"] == 4 * 44 and max(scored["efficiency_I6"].values()) <= 1e-12
    shares = scored["games"]["Y1"]["cue_final"]["shares"]
    assert shares["T"] == 0.0 and abs(sum(shares.values()) - 1.0) < 1e-12  # the cue-final null player, and efficiency
    y2_rows = y2_tables["coordinated"].clone()
    stats = ul.fresh_game(y2_measured["blocks"]["coordinated"]["dc"], y2_rows, "coordinated")
    assert float(stats["shares"][0, 0]) == scored["conditions"]["Y2/C3"]["value"]  # the kernel with D = 1 is the scored value


def test_verified_y2_blocks_and_the_barrier_refuse_every_mismatch(tmp_path):
    units = ul.table_units([{"word": "b", "token_id": 9, "class": "quantity"}, {"word": "a", "token_id": 3, "class": "adjective"}],
                           [SimpleNamespace(frame_id="quantifier-022-1", template_id="quantifier"), SimpleNamespace(frame_id="coordinated-adjective-022-1", template_id=ul.COORDINATED)])
    assert units.pair_json() == {"cue_final": [["a", 3, "quantifier-022-1"], ["b", 9, "quantifier-022-1"]], "coordinated": [["a", 3, "coordinated-adjective-022-1"], ["b", 9, "coordinated-adjective-022-1"]]}
    meta = ul.table_meta("Y2", units, ["n1", "n2"])
    layout = ul.table_layout(units, 2)
    lock = {"content_sha256": "L", "y2_table": {"data_path": "y2.f64", "index_path": "y2.json", "layout": layout, "meta": meta}}
    blocks = [("cue_final", torch.randn(2, 16, 2, dtype=torch.float64)), ("coordinated", torch.randn(2, 32, 2, dtype=torch.float64))]
    index = ul.write_table(tmp_path / "y2.f64", tmp_path / "y2.json", blocks, meta)
    stage1 = {"y2_table": {"data_path": "y2.f64", "index_path": "y2.json", "file_sha256": index["file_sha256"], "index_sha256": rc.file_sha256(tmp_path / "y2.json")},
              "lock_sha256": "L", "frames": {}}
    stage1["digest"] = ul.stage_one_digest(stage1)
    back = ul.verified_y2_blocks(tmp_path, stage1, lock)
    assert all(torch.equal(back[name], block) for name, block in blocks)
    confirmation = SimpleNamespace(target_prompts=(SimpleNamespace(key="k1"), SimpleNamespace(key="k2")))
    state = {"confirmation": {"stage1": stage1}, "executed_prompt_keys": ["other"]}
    assert set(ul.barrier_022(tmp_path, state, lock, confirmation)) == {"cue_final", "coordinated"}
    with pytest.raises(ul.IncidentError, match="S2-TARGET"):
        ul.barrier_022(tmp_path, {**state, "executed_prompt_keys": ["k2"]}, lock, confirmation)
    with pytest.raises(ul.IncidentError, match="digest"):
        ul.barrier_022(tmp_path, {**state, "confirmation": {"stage1": {**stage1, "frames": {"x": 1}}}}, lock, confirmation)
    with pytest.raises(ul.IncidentError, match="bound layout"):
        ul.verified_y2_blocks(tmp_path, stage1, {**lock, "y2_table": {**lock["y2_table"], "layout": list(reversed(layout))}})
    with pytest.raises(ul.IncidentError, match="noun order"):
        ul.verified_y2_blocks(tmp_path, stage1, {**lock, "y2_table": {**lock["y2_table"], "meta": {**meta, "nouns": ["n2", "n1"]}}})
    (tmp_path / "y2.json").write_text((tmp_path / "y2.json").read_text().replace('"Y2"', '"Y1"'))
    with pytest.raises(ul.IncidentError, match="index changed"):
        ul.verified_y2_blocks(tmp_path, stage1, lock)
