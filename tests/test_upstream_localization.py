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
