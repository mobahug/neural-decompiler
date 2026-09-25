"""Experiment 024's module (tier A unless marked): the constants and the immutable production configuration, the pins
and inherited inputs, the canonical per-cue MSE, the score, Spearman, the line, the SHA indices, the primary
classification, the exact E–N guard and the outcome; the freeze, the calibration, the lock and the confirmation's
pure steps; the pinned-model contracts (tier C, opt-in; spent data only)."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import random
import shutil
from fractions import Fraction
from pathlib import Path

import numpy
import pytest
import torch

from neural_decompiler import block0_completion as b0c
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul

ROOT = Path(__file__).parents[1]


# ---------------------------------------------------------------------------
# Task 1: constants, the production configuration, the pins and the inherited inputs.


def test_constants_are_the_design_values():
    assert rr.DESIGN == {"path": "docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md", "revision": 2, "commit": "9d03dee"}
    assert rr.PLAN == {"path": "docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md", "revision": 1, "commit": "608088c"}
    assert rr.EXPERIMENT == "024" and rr.EXPERIMENT_DIR == "experiments/024-readout-routing-nounness"
    assert rr.CLASSES == ("N", "B", "D", "C", "E") and rr.GROUPS == ("cue_final", "coordinated") and rr.PRONOUN_STRATUM == "possessive-or-pronoun"
    assert (rr.PRIMARY_TAG, rr.NULL_TAG, rr.CONTRAST_TAG) == ("024|primary", "024|null", "024|contrast")
    assert rr.TOLERANCES == {"I1": 1e-4, "I3": 1e-4, "I4": 1e-3, "C_recompute": 0.0, "I7": 0.0, "spearman": 1e-12, "en_float": 1e-12, "mse": 1e-12, "level1": 2e-2}
    assert rr.PRIMARY_RESULTS == ("NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS")
    assert rr.GUARD_RESULTS == ("NOT_INTERPRETABLE", "FAIL", "PASS")
    assert rr.OUTCOMES == ("NOT_INTERPRETABLE", "NOUNNESS_PREDICTION_NOT_ESTABLISHED", "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED",
                           "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS")
    assert "SIMPLE" in rr.OUTCOMES[-1] and "simple" in rr.SEMANTICS["simple"] and "unique causal factor" in rr.SEMANTICS["simple"]


def test_candidate_lists_are_the_design_lists_verbatim():
    assert len(rr.N_CANDIDATES) == 37 and rr.N_CANDIDATES[:10] == ("eager", "fierce", "honest", "polite", "rude", "sleepy", "wise", "lucky", "merry", "nervous")
    assert rr.N_CANDIDATES[-3:] == ("latest", "earliest", "brave")
    assert len(rr.MEASURE_LEMMAS) == 24 and rr.MEASURE_LEMMAS[0] == ("gallon", "gallons") and rr.MEASURE_LEMMAS[10] == ("bunch", "bunches")
    assert rr.MEASURE_LEMMAS[-3:] == (("crate", "crates"), ("basket", "baskets"), ("carton", "cartons"))
    assert len(rr.ORDINARY_LEMMAS) == 28 and rr.ORDINARY_LEMMAS[4] == ("teacher", "teachers") and rr.ORDINARY_LEMMAS[-1] == ("statue", "statues")
    assert all(plural in (singular + "s", singular + "es") for singular, plural in rr.MEASURE_LEMMAS + rr.ORDINARY_LEMMAS)
    assert rr.EXPECTED_PICKS == {"N": ("honest", "polite", "rude", "sleepy", "wise", "lucky", "merry", "nervous"),
                                 "measure": ("gallon", "ounce", "acre", "herd", "crowd", "bundle", "cluster", "litre"),
                                 "ordinary": ("apple", "horse", "doctor", "king", "rabbit", "poet", "dragon", "lion")}


def test_the_production_configuration_is_the_design_and_cannot_be_mutated():
    config = rr.PRODUCTION
    assert (config.name, config.class_quota, config.n_fresh, config.draws, config.null_permutations, config.contrast_resamples) == ("production", 8, 40, 10_000,
                                                                                                                                    100_000, 10_000)
    assert config.calibration_counts == (("determiner-like", 45), ("quantity", 45), ("adjective", 49)) and config.n_calibration == 139
    assert (config.n_pronoun, config.n_frames, config.n_nouns, config.cross_check_draws) == (36, 108, 79, 16)
    assert {key: words for key, words in config.expected_picks} == rr.EXPECTED_PICKS and config.guard is rr.PRODUCTION_GUARD
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.class_quota = 4  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rr.PRODUCTION_GUARD.max_upper = 322  # type: ignore[misc]
    assert json.loads(json.dumps(config.to_json()))["guard"]["max_upper"] == 321


def test_production_e_n_guard_contract_is_derived_independently():
    """Requirement 2: production binds n_E = n_N = 8, C(16, 8) = 12,870 and the largest passing K = 321 — checked against an
    independent enumeration count and the exact 321/322 boundary, not only against the module's own derivation."""
    assert rr.PRODUCTION_GUARD == rr.GuardSpec(n_e=8, n_n=8, assignments=12_870, max_upper=321)
    assert sum(1 for _ in itertools.combinations(range(16), 8)) == 12_870 == math.factorial(16) // (math.factorial(8) ** 2)
    assert Fraction(321, 12_870) <= Fraction(25, 1000) < Fraction(322, 12_870)
    assert rr.GuardSpec.derive(8, 8) == rr.PRODUCTION_GUARD
    assert rr.en_decision(321, rr.PRODUCTION_GUARD) == "PASS" and rr.en_decision(322, rr.PRODUCTION_GUARD) == "FAIL"
    assert rr.GuardSpec.derive(4, 4) == rr.GuardSpec(4, 4, 70, 1)  # the fake world's sizes, derived by the same rule
    with pytest.raises(ValueError, match="inconsistent guard"):
        dataclasses.replace(rr.PRODUCTION, guard=rr.GuardSpec(8, 8, 12_870, 322))
    with pytest.raises(ValueError, match="full E and N classes"):
        dataclasses.replace(rr.PRODUCTION, class_quota=4)
    with pytest.raises(ValueError, match="expected picks"):
        dataclasses.replace(rr.PRODUCTION, guard=rr.GuardSpec.derive(4, 4), class_quota=4)  # the design's picks are eight per list


def test_the_reused_programs_are_pinned_by_blob_and_a_change_refuses(monkeypatch):
    assert rr.module_blobs() == rr.FROZEN_BLOBS
    assert rr.FROZEN_BLOBS["upstream_localization.py"] == "465856962aa380747d1a4f1338d1d2762d03c9f9"
    assert rr.FROZEN_BLOBS["block0_completion.py"] == "16d310fc7fab5599ec83b8bd8162fb9613f8dad2"
    assert {name: blob for name, blob in rr.FROZEN_BLOBS.items() if name not in ("upstream_localization.py", "block0_completion.py")} == ul.FROZEN_BLOBS
    assert {name: blob for name, blob in rr.FROZEN_BLOBS.items() if name != "block0_completion.py"} == b0c.FROZEN_BLOBS
    monkeypatch.setitem(rr.FROZEN_BLOBS, "block0_completion.py", "0" * 40)
    with pytest.raises(rr.PhaseError, match="frozen modules changed"):
        rr.assert_frozen_blobs()


def test_the_pins_are_literal_and_not_inherited(monkeypatch):
    monkeypatch.setitem(b0c.FROZEN_BLOBS, "readout_decompilation.py", "0" * 40)
    monkeypatch.setitem(ul.FROZEN_BLOBS, "readout_decompilation.py", "0" * 40)
    assert rr.FROZEN_BLOBS["readout_decompilation.py"] == "caa73b40192f4c910dc63371bd19db75a3258339" and rr.assert_frozen_blobs()


def _copy_023(tmp_path: Path) -> Path:
    for relative in rr.INHERITED_023_PATHS.values():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, tmp_path / relative)
    return tmp_path


def test_experiment_023_inputs_verify_and_tampering_refuses(tmp_path):
    """Requirement 4: the calibration source is the reviewed 023 artifact, by file and content digest."""
    root = _copy_023(tmp_path)
    digests = rr.verify_023_inputs(root)
    assert set(digests) == set(rr.DIGEST_KEYS[-7:]) and digests["023_cells_data_file"] == "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4"
    assert rr.calibration_source()["index_content_sha256"] == "d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16"
    data = root / b0c.CELLS_DATA_RELATIVE_PATH
    raw = bytearray(data.read_bytes())
    raw[100] ^= 1
    data.write_bytes(bytes(raw))
    with pytest.raises(rr.PhaseError, match="cells data"):
        rr.verify_023_inputs(root)
    shutil.copyfile(ROOT / b0c.CELLS_DATA_RELATIVE_PATH, data)
    index = root / b0c.CELLS_INDEX_RELATIVE_PATH
    payload = json.loads(index.read_text())
    payload["cells_version"] = "resealed"
    payload["content_sha256"] = rc.content_digest(payload)  # verifies on its own, but is not the reviewed file
    index.write_text(json.dumps(payload))
    with pytest.raises(rr.PhaseError, match="cells index"):
        rr.verify_023_inputs(root)
    shutil.copyfile(ROOT / b0c.CELLS_INDEX_RELATIVE_PATH, index)
    (root / b0c.LOCK_RELATIVE_PATH).unlink()
    with pytest.raises(rr.PhaseError, match="lock"):
        rr.verify_023_inputs(root)


# ---------------------------------------------------------------------------
# Task 2: the canonical computations.


def _cells(rows: int, seed: int = 3) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    ys = [torch.randn(79, generator=generator, dtype=torch.float64) for _ in range(rows)]
    cs = [y + 0.3 * torch.randn(79, generator=generator, dtype=torch.float64) for y in ys]
    return torch.stack([rr.fresh_pair_cells(y, c) for y, c in zip(ys, cs)]), ys, cs


def test_the_fresh_cells_are_023s_cell_function_and_the_mse_is_sse_c_over_n():
    cells, ys, cs = _cells(6)
    for row, (y, c) in enumerate(zip(ys, cs)):
        assert torch.equal(cells[row], b0c.pair_cells(y, c, c, c))
        assert cells[row, rr.SSEC] == cells[row, b0c.CELL_COLUMNS.index("SSE0")] == cells[row, b0c.CELL_COLUMNS.index("SSE1")]
        assert cells[row, rr.COUNT] == 79.0 and math.isclose(float(cells[row, rr.SSEC]), float(((y - c) ** 2).sum()), rel_tol=1e-15)
    expected = math.fsum(float(((y - c) ** 2).sum()) for y, c in zip(ys, cs)) / (6 * 79)
    assert math.isclose(rr.cue_mse(cells), expected, rel_tol=1e-14)
    assert rr.cue_mse(cells[torch.randperm(6)]) == rr.cue_mse(cells)  # exactly rounded sums: the row order cannot change it
    assert abs(rr.cue_mse(cells) - rr.cue_mse_torch(cells)) <= 1e-12 * rr.cue_mse(cells)
    hand = torch.zeros(2, 8, dtype=torch.float64)
    hand[:, rr.COUNT], hand[:, rr.SSEC] = torch.tensor([79.0, 79.0], dtype=torch.float64), torch.tensor([7.9, 23.7], dtype=torch.float64)
    assert rr.cue_mse(hand) == pytest.approx(0.2, rel=1e-15)


def test_log_mse_is_defined_only_for_a_finite_positive_value():
    assert rr.log_mse(math.e) == 1.0 and rr.log_mse(1.0) == 0.0
    for bad in (0.0, -1.0, math.nan, math.inf, None):
        assert rr.log_mse(bad) is None


class _Noun:
    def __init__(self, sg, pl, single=True):
        self.sg_ids, self.pl_ids, self.single_token = (sg,), (pl,), single


def test_the_score_is_the_cosine_difference_with_a_direct_leave_one_out_centroid():
    generator = torch.Generator().manual_seed(24)
    W_E = torch.randn(40, 6, generator=generator, dtype=torch.float32)
    pool = type("Pool", (), {"nouns": (_Noun(1, 2), _Noun(3, 4, single=False), _Noun(5, 6))})()
    units = type("Units", (), {"cues": (("a", 10, "determiner-like"), ("p", 11, "possessive-or-pronoun"), ("b", 12, "quantity"), ("c", 13, "adjective"),
                                        ("d", 14, "adjective"))})()
    assert rr.noun_row_ids(pool) == [1, 2, 5, 6] and rr.calibration_cues(units) == [("a", 10, "determiner-like"), ("b", 12, "quantity"), ("c", 13, "adjective"),
                                                                                  ("d", 14, "adjective")]
    bindings = rr.score_bindings(W_E, pool, units)
    assert bindings["calibration_cue_ids"] == [10, 12, 13, 14] and bindings["embedding_sha256"] == rr.embedding_digest(W_E.clone())
    mu_noun = W_E[[1, 2, 5, 6]].double().mean(0)
    loo = rr.calibration_scores(W_E, bindings)
    for k, token_id in enumerate([10, 12, 13, 14]):
        others = [i for i in [10, 12, 13, 14] if i != token_id]
        e = W_E[token_id].double()
        direct = float(e @ mu_noun / (e.norm() * mu_noun.norm())) - float(e @ W_E[others].double().mean(0) / (e.norm() * W_E[others].double().mean(0).norm()))
        assert loo[k] == pytest.approx(direct, abs=1e-15)
        assert loo[k] == rr.nounness(W_E[token_id], mu_noun, rr.centroid(W_E, others))  # the direct mean of the others, bit for bit
    full = rr.full_scores(W_E, bindings, [11, 20])
    assert full[0] == rr.nounness(W_E[11], mu_noun, rr.centroid(W_E, [10, 12, 13, 14]))
    e = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    assert rr.nounness(e, e, torch.tensor([0.0, 1.0, 0, 0, 0, 0], dtype=torch.float64)) == 1.0


def test_spearman_uses_average_ranks_and_is_undefined_for_a_constant_vector():
    assert rr.average_ranks([3.0, 1.0, 2.0, 2.0]) == [4.0, 1.0, 2.5, 2.5]
    assert rr.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8, abs=1e-15)
    assert rr.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(4.5 / math.sqrt(4.5 * 5.0), abs=1e-15)
    assert rr.spearman([1, 1, 1], [1, 2, 3]) is None and rr.spearman([1, 2, 3], [5, 5, 5]) is None
    with pytest.raises(ValueError):
        rr.spearman([1.0, math.nan], [1.0, 2.0])
    rng = random.Random(7)
    worst = 0.0
    for _ in range(200):
        x = [round(rng.gauss(0, 1), 1) for _ in range(40)]  # rounded: plenty of ties
        y = [math.exp(rng.gauss(-2, 0.4)) for _ in range(40)]
        rho = rr.spearman(x, y)
        assert rho == rr.spearman(x, [math.log(v) for v in y])  # a monotone transform leaves ρ unchanged
        order = list(range(40))
        rng.shuffle(order)
        assert rho == rr.spearman([x[i] for i in order], [y[i] for i in order])  # exact sums: the pair order cannot change it
        worst = max(worst, rr.spearman_agreement(rho, rr.spearman_direct(x, y)))
    assert worst <= rr.TOLERANCES["spearman"]
    assert rr.spearman_agreement(None, None) == 0.0 and rr.spearman_agreement(0.1, None) == math.inf


def test_the_line_is_exact_ols_and_no_rounded_design_diagnostic_is_in_the_code():
    line = rr.ols([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.5])
    assert line["slope"] == 2.15 and line["intercept"] == pytest.approx(0.9, abs=1e-15) and line["n"] == 4
    assert line["residual_sd"] == pytest.approx(math.sqrt(0.075 / 2), abs=1e-15)
    rng = numpy.random.default_rng(5)
    x = rng.normal(-0.2, 0.15, 139)
    y = 1.3 * x - 2.5 + rng.normal(0, 0.25, 139)
    line = rr.ols(x.tolist(), y.tolist())
    slope, intercept = numpy.polyfit(x, y, 1)
    assert abs(line["slope"] - slope) <= 1e-12 and abs(line["intercept"] - intercept) <= 1e-12
    assert rr.predict(line, 0.1) == line["intercept"] + line["slope"] * 0.1
    source = Path(rr.__file__).read_text()
    for rounded in ("1.299", "2.481", "0.257", "0.315", "0.530", "0.834", "0.416"):
        assert rounded not in source, rounded  # the exact values come from calibration, never from the design's prose


def test_the_sha_indices_are_deterministic_and_fisher_yates_gives_permutations():
    assert rr.sha_int("024|primary|0|0") % 139 == rr.primary_draw_index(0, 0, 139) == 3
    assert rr.null_permutation(0, 10) == [5, 2, 4, 1, 8, 9, 6, 0, 7, 3]
    for p in range(200):
        assert sorted(rr.null_permutation(p, 40)) == list(range(40))
    assert rr.primary_draw_indices(3, 40, 139) == rr.primary_draw_indices(3, 40, 139) and all(0 <= i < 139 for row in rr.primary_draw_indices(3, 40, 139) for i in row)
    assert all(0 <= rr.contrast_draw_index(b, "E", slot, 8) < 8 for b in range(10) for slot in range(8))
    assert rr.contrast_draw_index(0, "E", 0, 8) != rr.contrast_draw_index(0, "N", 0, 8) or rr.contrast_draw_index(1, "E", 0, 8) != rr.contrast_draw_index(1, "N", 0, 8)


def test_the_ranks_and_order_statistics():
    assert (rr.lower_rank(10_000), rr.null_rank(100_000), rr.lower_rank(40), rr.null_rank(400)) == (250, 97_500, 1, 390)
    assert rr.lower_rank(10_000) - 1 == 249 and 10_000 - rr.lower_rank(10_000) == 9_750  # the bootstrap's two elements
    assert rr.order_statistic([0.3, None, 0.1, 0.2], 1) == -math.inf and rr.order_statistic([0.3, None, 0.1, 0.2], 2) == 0.1
    assert rr.defined_median([0.3, None, 0.1, 0.2]) == 0.2 and rr.defined_median([None]) is None


def test_the_primary_classification_precedence():
    assert rr.classify_primary(None, 0.26, 0.31) == "NOT_INTERPRETABLE"
    assert rr.classify_primary(0.30, 0.26, 0.31) == "GUARD_FAILURE"
    assert rr.classify_primary(0.31, 0.26, 0.31) == "PASS"  # with F < null the null binds and no envelope-only failure can occur
    assert rr.classify_primary(0.40, 0.45, 0.31) == "ENVELOPE_ONLY_FAILURE" and rr.classify_primary(0.45, 0.45, 0.31) == "PASS"
    with pytest.raises(rr.IncidentError):
        rr.classify_primary(0.4, math.nan, 0.31)
    with pytest.raises(rr.IncidentError):
        rr.classify_primary(math.inf, 0.26, 0.31)


def _arranged(values: list[float], rank: int) -> list[float]:
    """The 16 values rearranged so that the observed E assignment is the one whose exact sum has the given descending
    rank (1 = the largest) among all 12,870 assignments."""
    exact = rr.exact_subset_sums(values, rr.PRODUCTION_GUARD)
    order = sorted(range(12_870), key=lambda row: -exact["sums"][row])
    chosen = list(itertools.combinations(range(16), 8))[order[rank - 1]]
    return [values[i] for i in chosen] + [values[i] for i in range(16) if i not in chosen]


def test_the_exact_guard_passes_at_321_and_fails_at_322_through_the_full_computation():
    rng = random.Random(24)
    values = [rng.gauss(-2.4, 0.4) for _ in range(16)]
    for rank, result in ((1, "PASS"), (321, "PASS"), (322, "FAIL"), (12_870, "FAIL")):
        guard = rr.en_guard_logs(_arranged(values, rank), rr.PRODUCTION_GUARD)
        assert (guard["K"], guard["result"]) == (rank, result)
        assert guard["p_exact"] == f"{rank}/12870" and guard["checks"]["assignments"] == 12_870 and guard["checks"]["complement_identity"]
    assert rr.en_guard_logs(_arranged(values, 321), rr.PRODUCTION_GUARD)["threshold_D"] == rr.en_guard_logs(_arranged(values, 321), rr.PRODUCTION_GUARD)["D_EN"]


def test_an_all_tied_population_fails_and_e_below_n_fails():
    tied = rr.en_guard([0.08] * 8, [0.08] * 8, rr.PRODUCTION_GUARD)
    assert (tied["K"], tied["result"], tied["threshold"], tied["D_EN"]) == (12_870, "FAIL", "+inf", 0.0)
    assert rr.en_guard([0.05] * 8, [0.1] * 8, rr.PRODUCTION_GUARD)["K"] == 12_870
    above = rr.en_guard([0.2 + 0.01 * i for i in range(8)], [0.05 + 0.001 * i for i in range(8)], rr.PRODUCTION_GUARD)
    assert (above["K"], above["result"]) == (1, "PASS") and above["ratio_of_geometric_means"] > 1.0
    assert above["groups"]["E"]["mean_mse"] == pytest.approx(0.235) and above["groups"]["N"]["mean_log_mse"] < above["groups"]["E"]["mean_log_mse"]


def test_the_known_float64_adversarial_case_gets_the_exact_tie_count():
    """Requirement 1's regression case: float64 subset sums tie the observed assignment with 6,863 others, 3,432 of which
    are strictly below it; ordinary float summation would count K = 9,867 where the exact count is 6,435."""
    logs = [1.0, 2.0 ** -60] + [0.0] * 6 + [1.0] + [0.0] * 7
    floats = [sum(logs[i] for i in subset) for subset in itertools.combinations(range(16), 8)]
    assert sum(1 for value in floats if value >= floats[0]) == 9_867  # the float route: wrong
    guard = rr.en_guard_logs(logs, rr.PRODUCTION_GUARD)
    assert guard["K"] == 6_435 and guard["result"] == "FAIL"
    exact = rr.exact_subset_sums(logs, rr.PRODUCTION_GUARD)
    assert sum(1 for value in exact["sums"] if value == exact["sums"][0]) == 3_432  # the genuine ties, the observed included


def test_the_guard_size_is_at_most_321_of_12870_with_or_without_ties():
    rng = random.Random(9)
    cases = [[rng.gauss(0, 1) for _ in range(16)], [float(rng.randint(0, 3)) for _ in range(16)], [float(i % 2) for i in range(16)]]
    for index, values in enumerate(cases):
        import bisect

        sums = sorted(rr.exact_subset_sums(values, rr.PRODUCTION_GUARD)["sums"])
        assert rr.upper_count(sums, sums[5_000]) == len(sums) - bisect.bisect_left(sums, sums[5_000])  # the counting rule, by bisection below
        passing = sum(1 for value in sums if rr.en_decision(len(sums) - bisect.bisect_left(sums, value), rr.PRODUCTION_GUARD) == "PASS")
        assert passing <= 321
        if index == 0:
            assert passing == 321  # untied: exactly 321 assignments could pass (size 321/12,870)
            assert rr.en_guard_logs(values, rr.PRODUCTION_GUARD)["checks"]["float_difference"] <= 1e-12


def test_the_guard_refuses_wrong_sizes_and_is_not_interpretable_without_a_positive_mse(monkeypatch):
    with pytest.raises(rr.PhaseError, match="8 \\+ 8"):
        rr.en_guard([0.1] * 7, [0.1] * 9, rr.PRODUCTION_GUARD)
    guard = rr.en_guard([0.1] * 7 + [0.0], [0.1] * 8, rr.PRODUCTION_GUARD)
    assert guard["result"] == "NOT_INTERPRETABLE" and guard["K"] is None and guard["log_mse"][7] is None
    fake = rr.GuardSpec.derive(4, 4)
    assert rr.en_guard([0.3, 0.4, 0.5, 0.6], [0.1, 0.11, 0.12, 0.13], fake)["K"] == 1 and rr.en_guard([0.3, 0.4, 0.5, 0.6], [0.1, 0.11, 0.12, 0.13], fake)["result"] == "PASS"
    monkeypatch.setitem(rr.TOLERANCES, "en_float", -1.0)
    with pytest.raises(rr.GuardCheckError, match="plain float64"):
        rr.en_guard([0.2] * 8, [0.1] * 8, rr.PRODUCTION_GUARD)


def test_the_outcome_table_is_exhaustive():
    table = {("NOT_INTERPRETABLE", guard): "NOT_INTERPRETABLE" for guard in rr.GUARD_RESULTS}
    table.update({(primary, guard): "NOUNNESS_PREDICTION_NOT_ESTABLISHED" for primary in ("GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE") for guard in rr.GUARD_RESULTS})
    table.update({("PASS", "FAIL"): "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED", ("PASS", "NOT_INTERPRETABLE"): "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED",
                  ("PASS", "PASS"): "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS"})
    assert len(table) == 12
    for (primary, guard), expected in table.items():
        assert rr.outcome(primary, guard) == expected
    with pytest.raises(ValueError):
        rr.outcome("PASS", "MAYBE")
    assert set(rr.SEMANTICS["outcomes"]) == set(rr.OUTCOMES) and set(rr.SEMANTICS["primary"]) == set(rr.PRIMARY_RESULTS)
    sentence = rr.EXTRAPOLATION_SENTENCE.format(k=5, n=8, maximum=0.135)
    assert sentence.startswith("5 of the 8 E cues") and "prospective extrapolation test" in sentence and "design motivation only" in sentence


# ---------------------------------------------------------------------------
# Task 3: the tokenizer-only freeze (a stub tokenizer that reproduces the design's eligibility pattern, tier A; the real
# tokenizer, tier C).

MULTI_TOKEN = ("clumsy", "grumpy", "thirsty", "pints", "quarts", "flocks", "swarms", "bunches", "heaps", "mounds", "handfuls", "liters", "crates", "baskets", "carton",
               "cartons", "tigers", "castles", "pencils", "queens", "baker", "bakers", "lemons", "bananas", "violins", "wizards")


def _stub_world(used=("eager", "fierce", "brave", "dozen", "dozens"), noun_words=(("dog", "dogs"), ("cat", "cats"))):
    from types import SimpleNamespace

    from test_upstream_localization import StubTokenizer

    from neural_decompiler import plural_mechanism as pm

    class Tokenizer(StubTokenizer):
        SPLIT = {" " + word: (" " + word[:2], word[2:]) for word in MULTI_TOKEN}

    tokenizer = Tokenizer()
    one, two = tokenizer.id(" one"), tokenizer.id(" two")
    frames = (pm._build_new_frame(tokenizer, "cardinal", "The teacher counted {cue}", {"sg": one, "pl": two}, "cardinal-1"),
              pm._build_new_frame(tokenizer, "quantifier", "The farmer lists {cue}", {"sg": one, "pl": two}, "quantifier-1"),
              pm._build_new_frame(tokenizer, "coordinated-adjective", "The crate and basket held {cue} tight", {"sg": one, "pl": two}, "coordinated-adjective-1"))
    nouns = tuple(_Noun(tokenizer.id(" " + sg), tokenizer.id(" " + pl)) for sg, pl in noun_words)
    pool = SimpleNamespace(nouns=nouns, frames=frames, reference_ids={"cardinal": one, "quantifier": one, "coordinated-adjective": one})
    ids = sorted(tokenizer.id(" " + word) for word in used)
    base = {"cue_token_ids": ids, "cue_token_ids_sha256": pm.sha256_text(pm.canonical_json(ids)), "sources": [{"source": "stub"}]}
    c023 = {"cues": [{"word": "general", "token_id": tokenizer.id(" general")}], "frames": [{"cue_ids": {"sg": one, "pl": two}}],
            "exclusion": {"cue_token_ids_sha256": base["cue_token_ids_sha256"]}, "content_sha256": "3" * 64}
    return tokenizer, pool, base, c023


def _freeze(tokenizer, pool, base, c023, config=rr.PRODUCTION):
    return rr.freeze_payload(tokenizer, pool=pool, exclusion_base=base, confirmation_023=c023, confirmation_023_file_sha256="f" * 64, config=config,
                             model={"model_id": "stub", "revision": "x"})


def test_the_freeze_takes_the_first_eligible_entries_and_records_every_reason(tmp_path):
    from neural_decompiler import plural_mechanism as pm

    tokenizer, pool, base, c023 = _stub_world()
    payload = _freeze(tokenizer, pool, base, c023)
    by_class = {cls: [entry["word"] for entry in payload["cues"] if entry["class"] == cls] for cls in rr.CLASSES}
    assert by_class == {"N": list(rr.EXPECTED_PICKS["N"]), "D": list(rr.EXPECTED_PICKS["measure"]), "B": [w + "s" for w in rr.EXPECTED_PICKS["measure"]],
                        "E": list(rr.EXPECTED_PICKS["ordinary"]), "C": [w + "s" for w in rr.EXPECTED_PICKS["ordinary"]]}
    assert [entry["class"] for entry in payload["cues"]] == [cls for cls in rr.CLASSES for _ in range(8)]
    assert all(entry["lemma"] == entry["word"].removesuffix("s") for entry in payload["cues"] if entry["class"] in ("B", "C"))
    reasons = {entry["candidate"]: entry["reason"] for entry in payload["rejected"]}
    assert "already used" in reasons["eager"] and "2 tokens" in reasons["clumsy"] and "already used" in reasons["brave"]
    assert "exposed frame" in reasons["teacher/teachers"] and "exposed frame" in reasons["farmer/farmers"] and "already used" in reasons["dozen/dozens"]
    assert "exposed frame" in reasons["crate/crates"] and "2 tokens" in reasons["crate/crates"] and "2 tokens" in reasons["tiger/tigers"]
    assert payload["reserves"]["measure"] == ["barrel/barrels", "bucket/buckets", "sack/sacks"]
    assert payload["reserves"]["ordinary"] == ["soldier/soldiers", "sailor/sailors", "priest/priests", "knight/knights", "onion/onions", "carrot/carrots",
                                               "pirate/pirates", "tourist/tourists", "statue/statues"]
    assert payload["reserves"]["N"][0] == "anxious" and len(payload["reserves"]["N"]) == 23
    assert payload["exclusion"]["sources"][-1]["source"] == "confirmation-023" and tokenizer.id(" general") in payload["exclusion"]["cue_token_ids"]
    assert payload["counts"] == {"classes": {cls: 8 for cls in rr.CLASSES}} and len(payload["manifest"]["S2-TARGET"]) == 40 * 3
    assert payload["picks_match_expected"] is True and payload["configuration"] == rr.PRODUCTION.to_json() and payload["rules"] == rr.FREEZE_RULES
    path = tmp_path / "confirmation-v1.json"
    path.write_text(pm.canonical_json(payload) + "\n")
    excluded = rr.exclusion(base, c023, "f" * 64)
    confirmation = rr.load_confirmation_024(path, pool, excluded, rr.PRODUCTION)
    assert len(confirmation.target_prompts) == 120 and confirmation.frames == () and confirmation.manifest_keys() == set(payload["manifest"]["S2-TARGET"])
    assert [token["word"] for token in confirmation.class_tokens("E")] == list(rr.EXPECTED_PICKS["ordinary"])


def test_a_deviation_or_a_shortfall_raises_before_anything_exists_to_write():
    tokenizer, pool, base, c023 = _stub_world(used=("eager", "fierce", "brave", "dozen", "dozens", "honest"))
    with pytest.raises(rr.FreezeDeviation, match="honest"):
        _freeze(tokenizer, pool, base, c023)  # honest used before: N would become polite … anxious — the expected 40 are not what froze
    tokenizer, pool, base, c023 = _stub_world(noun_words=(("apple", "apples"),))
    with pytest.raises(rr.FreezeDeviation, match="ordinary"):
        _freeze(tokenizer, pool, base, c023)  # apple a target-noun form: the ordinary lemmas shift
    tokenizer, pool, base, c023 = _stub_world()
    tokenizer.SPLIT.update({" " + plural: (" " + plural[:2], plural[2:]) for _, plural in rr.MEASURE_LEMMAS})
    with pytest.raises(rr.FreezeShortfall, match="list measure: 0 eligible of 8"):
        _freeze(tokenizer, pool, base, c023)
    tokenizer, pool, base, c023 = _stub_world()
    with pytest.raises(rr.PhaseError, match="does not reproduce"):
        _freeze(tokenizer, pool, base, dict(c023, exclusion={"cue_token_ids_sha256": "0" * 64}))


def test_a_tampered_stale_or_foreign_confirmation_file_is_refused(tmp_path):
    import dataclasses as dc

    from neural_decompiler import plural_mechanism as pm

    tokenizer, pool, base, c023 = _stub_world()
    payload = _freeze(tokenizer, pool, base, c023)
    excluded = rr.exclusion(base, c023, "f" * 64)
    mutations = (lambda p: p["cues"][0].update(word="other"), lambda p: p["manifest"]["S2-TARGET"].pop(), lambda p: p.update(picks_match_expected=False),
                 lambda p: p.update(experiment="023"), lambda p: p["configuration"].update(draws=40), lambda p: p["cues"][0].update(token_id=p["cues"][1]["token_id"]))
    for mutate in mutations:
        changed = json.loads(pm.canonical_json(payload))
        mutate(changed)
        changed["content_sha256"] = rc.content_digest(changed)
        (tmp_path / "bad.json").write_text(pm.canonical_json(changed) + "\n")
        with pytest.raises(rr.PhaseError):
            rr.load_confirmation_024(tmp_path / "bad.json", pool, excluded, rr.PRODUCTION)
    (tmp_path / "ok.json").write_text(pm.canonical_json(payload) + "\n")
    later = rr.exclusion(dict(base, sources=[{"source": "stub"}, {"source": "later"}]), c023, "f" * 64)
    with pytest.raises(rr.PhaseError, match="exclusion recorded at the freeze"):
        rr.load_confirmation_024(tmp_path / "ok.json", pool, later, rr.PRODUCTION)
    test_config = dc.replace(rr.PRODUCTION, name="another")
    with pytest.raises(rr.PhaseError, match="different configuration"):
        rr.load_confirmation_024(tmp_path / "ok.json", pool, excluded, test_config)


def _real_freeze_inputs():
    from transformers import AutoTokenizer

    from neural_decompiler.models import PYTHIA_70M

    tokenizer = AutoTokenizer.from_pretrained(PYTHIA_70M.model_id, revision=PYTHIA_70M.revision, local_files_only=True)
    inputs = ul.load_frozen_inputs(ROOT)
    c022_path = ROOT / ul.CONFIRMATION_RELATIVE_PATH
    c022 = json.loads(c022_path.read_text())
    c023 = json.loads((ROOT / b0c.CONFIRMATION_RELATIVE_PATH).read_text())
    return tokenizer, inputs, b0c.exclusion(inputs, c022, rc.file_sha256(c022_path)), c023


@pytest.mark.pythia_smoke
def test_the_real_tokenizer_freeze_takes_the_designs_expected_picks_and_writes_nothing():
    """Task 3's contract, tokenizer and committed files only, in memory: the mechanical picks are design revision 2's
    expected 40, the reserves and the ✗ marks are the design's, and nothing is written (the freeze phase is a later,
    separately authorized step)."""
    import os

    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")
    target = ROOT / rr.CONFIRMATION_RELATIVE_PATH
    existed = target.exists()
    tokenizer, inputs, base, c023 = _real_freeze_inputs()
    payload = rr.freeze_payload(tokenizer, pool=inputs.pool, exclusion_base=base, confirmation_023=c023,
                                confirmation_023_file_sha256=rc.file_sha256(ROOT / b0c.CONFIRMATION_RELATIVE_PATH), config=rr.PRODUCTION)
    assert target.exists() == existed
    assert payload["picks"] == {key: list(words) for key, words in rr.EXPECTED_PICKS.items()}
    assert len(payload["exclusion"]["cue_token_ids"]) == 351 and len(base["cue_token_ids"]) == 327
    assert len(payload["target_noun_form_ids"]["ids"]) == 161 and len(payload["frame_token_ids"]["ids"]) == 316
    assert payload["reserves"]["measure"] == ["barrel/barrels", "bucket/buckets", "sack/sacks"]
    assert payload["reserves"]["ordinary"] == ["soldier/soldiers", "sailor/sailors", "priest/priests", "knight/knights", "onion/onions", "carrot/carrots",
                                               "pirate/pirates", "tourist/tourists", "statue/statues"]
    assert len(payload["reserves"]["N"]) == 23 and payload["reserves"]["N"][0] == "anxious"
    reasons = {entry["candidate"]: entry["reason"] for entry in payload["rejected"]}
    assert set(reasons) == {"eager", "fierce", "clumsy", "grumpy", "thirsty", "brave", "pint/pints", "quart/quarts", "flock/flocks", "swarm/swarms", "bunch/bunches",
                            "heap/heaps", "mound/mounds", "handful/handfuls", "liter/liters", "dozen/dozens", "crate/crates", "basket/baskets", "carton/cartons",
                            "teacher/teachers", "tiger/tigers", "castle/castles", "pencil/pencils", "farmer/farmers", "queen/queens", "baker/bakers", "lemon/lemons",
                            "banana/bananas", "violin/violins", "wizard/wizards"}
    for lemma in ("teacher/teachers", "farmer/farmers", "crate/crates", "basket/baskets"):
        assert "a token of an exposed frame" in reasons[lemma]
    assert len(payload["manifest"]["S2-TARGET"]) == 4_320 and payload["counts"] == {"classes": {cls: 8 for cls in rr.CLASSES}}


# ---------------------------------------------------------------------------
# Task 4: the calibration from the committed exposed cells and the weights (synthetic, tier A; the real artifact's
# exposed MSE, tier A; a synthetic full-scale dry run, slow).


def toy_configuration(**overrides) -> rr.Configuration:
    """A small, explicit test configuration (the only way a test world changes a size)."""
    values = dict(name="toy", class_quota=4, draws=40, null_permutations=400, contrast_resamples=40, cross_check_draws=4,
                  calibration_counts=(("determiner-like", 5), ("quantity", 5), ("adjective", 6)), n_pronoun=3, n_frames=6, n_nouns=79,
                  expected_picks=(("N", ("a", "b", "c", "d")), ("measure", ("e", "f", "g", "h")), ("ordinary", ("i", "j", "k", "l"))),
                  guard=rr.GuardSpec.derive(4, 4))
    values.update(overrides)
    return rr.Configuration(**values)


def _toy_calibration_world(config, seed=11, constant_mse=False):
    from neural_decompiler import plural_mechanism as pm

    frames = tuple(pm.Frame("cardinal" if i % 3 else "coordinated-adjective", f"frame-{i}", (1, 2, 3), () if i % 3 else (9,), {"sg": 4, "pl": 5}, "t")
                   for i in range(config.n_frames))
    strata = [stratum for stratum, count in config.calibration_counts for _ in range(count)] + [rr.PRONOUN_STRATUM] * config.n_pronoun
    cues = tuple((f"cue{i}", 100 + i, stratum) for i, stratum in enumerate(strata))
    units = b0c.ExposedUnits(cues, frames, {})
    generator = torch.Generator().manual_seed(seed)
    W_E = torch.randn(300, 8, generator=generator, dtype=torch.float32)
    pool = type("Pool", (), {"nouns": tuple(_Noun(10 + 2 * k, 11 + 2 * k) for k in range(4))})()
    bindings = rr.score_bindings(W_E, pool, units)
    cells = torch.zeros(len(cues) * config.n_frames, 8, dtype=torch.float64)
    cells[:, rr.COUNT] = float(config.n_nouns)
    scores = rr.full_scores(W_E, bindings, [token_id for _, token_id, _ in cues])
    for ci, score in enumerate(scores):
        level = 0.05 if constant_mse else math.exp(1.3 * score - 2.5)
        noise = 1.0 if constant_mse else torch.exp(0.2 * torch.randn(config.n_frames, generator=generator, dtype=torch.float64))
        cells[ci * config.n_frames:(ci + 1) * config.n_frames, rr.SSEC] = level * config.n_nouns * noise
    return cells, units, W_E, bindings


def _calibrate(config, cells, units, W_E, bindings):
    population = rr.calibration_population(cells, units, W_E, bindings, config)
    draws = rr.primary_draws(population["calibration"], config)
    null = rr.null_distribution(config)
    checks = {"mse": population["mse_check"], "spearman": rr.spearman_cross_check(population["calibration"], draws, config)}
    evaluated = rr.evaluate_calibration(population["calibration"], population["pronouns"], draws, null, config)
    return population, draws, null, checks, evaluated


def test_the_calibration_computes_the_floor_the_null_and_the_exact_line():
    config = toy_configuration()
    cells, units, W_E, bindings = _toy_calibration_world(config)
    population, draws, null, checks, evaluated = _calibrate(config, cells, units, W_E, bindings)
    entries = population["calibration"]
    assert len(entries) == 16 and len(population["pronouns"]) == 3 and checks["mse"]["passed"] and checks["spearman"]["n_checked"] == 5
    assert [entry["token_id"] for entry in entries] == bindings["calibration_cue_ids"] and all(entry["log_mse"] == math.log(entry["mse"]) for entry in entries)
    values = draws["values"]
    assert len(values) == 40 and len(draws["indices"][0]) == 20 and all(0 <= i < 16 for row in draws["indices"] for i in row)
    floor = evaluated["primary_floor"]
    assert (floor["rank"], floor["element"]) == (1, 0) and floor["F_rho"] == sorted(v for v in values if v is not None)[0]
    assert evaluated["null"]["rank"] == 390 and evaluated["null"]["null_975"] == sorted(null["values"])[389]
    assert evaluated["effective_threshold"]["value"] == max(floor["F_rho"], evaluated["null"]["null_975"])
    line = rr.ols([e["nounness_loo"] for e in entries], [e["log_mse"] for e in entries])
    assert evaluated["line"] == line and 0.5 < line["slope"] < 2.5
    assert math.isclose(sum(evaluated["descriptive"]["draw_rates"].values()), 1.0) and evaluated["descriptive"]["calibration_rho"] > 0
    arrays = rr.calibration_arrays(draws, null)
    record = rr.calibration_record(run_id="r", protocol_code_commit="c" * 40, digests={"x": "y"}, config=config, confirmation={"path": "p"}, bindings=bindings,
                                   dependencies={"calibration_source": rr.calibration_source()}, population=population, evaluated=evaluated, checks=checks,
                                   array_digests=rr.arrays_digests(arrays))
    rr.verify_calibration_record(record, config)
    assert record["configuration"]["name"] == "toy" and record["arrays_sha256"]["draw_indices"] == rc.tensor_digest(arrays["draw_indices"])
    for mutate in (lambda r: r["line"].update(slope=r["line"]["slope"] * (1 + 1e-15)), lambda r: r["primary_floor"].update(element=5),
                   lambda r: r["calibration_cues"].pop(), lambda r: r["effective_threshold"].update(binds="nothing")):
        changed = json.loads(json.dumps(record))
        mutate(changed)
        changed["content_sha256"] = rc.content_digest(changed)
        with pytest.raises(rr.PhaseError):
            rr.verify_calibration_record(changed, config)
    with pytest.raises(rr.PhaseError, match="configuration"):
        rr.verify_calibration_record(record, rr.PRODUCTION)


def test_the_calibration_stops_for_review_on_undefined_draws_or_a_reversed_direction(monkeypatch):
    config = toy_configuration()
    cells, units, W_E, bindings = _toy_calibration_world(config, constant_mse=True)
    with pytest.raises(rr.CalibrationStop, match="undefined draws reach the stop"):
        _calibrate(config, cells, units, W_E, bindings)
    cells, units, W_E, bindings = _toy_calibration_world(config)
    monkeypatch.setattr(rr, "order_statistic", lambda values, rank: 2.0)
    with pytest.raises(rr.CalibrationStop, match="reversed direction"):
        _calibrate(config, cells, units, W_E, bindings)


def test_a_failed_cross_check_is_an_incident(monkeypatch):
    config = toy_configuration()
    cells, units, W_E, bindings = _toy_calibration_world(config)
    monkeypatch.setitem(rr.TOLERANCES, "mse", -1.0)
    with pytest.raises(rr.CrossCheckError, match="mse"):
        rr.calibration_population(cells, units, W_E, bindings, config)
    monkeypatch.setitem(rr.TOLERANCES, "mse", 1e-12)
    monkeypatch.setattr(rr, "spearman_direct", lambda x, y: 2.0)
    with pytest.raises(rr.CrossCheckError, match="spearman"):
        _calibrate(config, cells, units, W_E, bindings)


def test_the_calibration_refuses_a_population_that_is_not_the_configured_one():
    config = toy_configuration()
    cells, units, W_E, bindings = _toy_calibration_world(config)
    with pytest.raises(rr.PhaseError, match="calibration population"):
        rr.calibration_population(cells, units, W_E, bindings, toy_configuration(calibration_counts=(("determiner-like", 6), ("quantity", 5), ("adjective", 5))))
    with pytest.raises(rr.PhaseError, match="frames per cue"):
        rr.calibration_population(cells, units, W_E, bindings, toy_configuration(n_frames=7))
    cells[3, rr.COUNT] = 78.0
    with pytest.raises(rr.PhaseError, match="nouns"):
        rr.calibration_population(cells, units, W_E, bindings, config)


@pytest.mark.slow
def test_the_committed_023_artifact_gives_every_exposed_cue_mse_by_a_direct_loop():
    """The real calibration source, read through 023's reader (exposed values only; no floor is computed here)."""
    inputs = ul.load_frozen_inputs(ROOT)
    noun_keys = [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]
    record_022 = json.loads((ROOT / ul.CALIBRATION_RELATIVE_PATH).read_text())
    cells, units = rr.read_exposed_cells(ROOT, inputs, noun_keys, record_022)
    mse, check = rr.exposed_cue_mse(cells, units)
    assert len(mse) == 175 and check["passed"] and len(units.frames) == 108 and len(noun_keys) == 79
    counts = {stratum: sum(1 for _, _, cls in units.cues if cls == stratum) for stratum in (*b0c.STRATA, rr.PRONOUN_STRATUM)}
    assert counts == {"determiner-like": 45, "quantity": 45, "adjective": 49, rr.PRONOUN_STRATUM: 36}
    raw = numpy.frombuffer((ROOT / b0c.CELLS_DATA_RELATIVE_PATH).read_bytes(), dtype="<f8").reshape(18_900, 8)
    for ci in (0, 44, 90, 174):
        rows = raw[ci * 108:(ci + 1) * 108]
        assert mse[ci] == math.fsum(rows[:, 5]) / math.fsum(rows[:, 0])


@pytest.mark.slow
def test_a_synthetic_full_scale_calibration_dry_run_fits_in_time_and_memory():
    """139 cues × 108 frames, B = 10,000 and P = 100,000 at the production sizes, with synthetic cells and embeddings
    (no model, no real data): the whole calibration path runs and every order statistic is at its production element."""
    import time

    config = dataclasses.replace(rr.PRODUCTION, name="dry-run")
    cells, units, W_E, bindings = _toy_calibration_world(config)
    started = time.time()
    population, draws, null, checks, evaluated = _calibrate(config, cells, units, W_E, bindings)
    assert time.time() - started < 600
    assert evaluated["primary_floor"]["element"] == 249 and evaluated["null"]["element"] == 97_499 and len(null["values"]) == 100_000
    assert 0.28 < evaluated["null"]["null_975"] < 0.35 and checks["spearman"]["passed"]


# ---------------------------------------------------------------------------
# Task 5: the lock (synthetic, tier A).


def _toy_confirmation(config, frames=()):
    tokens = tuple({"word": f"{cls.lower()}{k}", "token_id": 200 + 10 * index + k, "class": cls, "lemma": f"{cls.lower()}{k}", "form": "word"}
                   for index, cls in enumerate(rr.CLASSES) for k in range(config.class_quota))
    return rr.Confirmation024({"cardinal": 4, "quantifier": 4, "coordinated-adjective": 4}, tuple(frames), tokens, "9" * 64)


def _toy_lock(config=None):
    config = config or toy_configuration()
    cells, units, W_E, bindings = _toy_calibration_world(config)
    population, draws, null, checks, evaluated = _calibrate(config, cells, units, W_E, bindings)
    confirmation = _toy_confirmation(config, units.frames)
    dependencies = {"calibration_source": rr.calibration_source(), "module_blobs": dict(rr.FROZEN_BLOBS), "C": rr.C_DEFINITION, "measurement": rr.MEASUREMENT,
                    "readout_020": {"module": "readout_decompilation.py", "exposed_states_sha256": "s" * 64},
                    "model": {"parameters_sha256": "p" * 64, "embedding_sha256": rr.embedding_digest(W_E)}}
    record = rr.calibration_record(run_id="r", protocol_code_commit="c" * 40, digests={"x": "y"}, config=config,
                                   confirmation=rr.confirmation_binding(confirmation, "f" * 64), bindings=bindings, dependencies=dependencies, population=population,
                                   evaluated=evaluated, checks=checks, array_digests={})
    fresh = rr.fresh_quantities(W_E, bindings, confirmation, record)
    lock = rr.build_lock(run_id="r", protocol_code_commit="d" * 40, digests={"x": "y"}, config=config, record=record, record_file_sha256="e" * 64,
                         confirmation=confirmation, confirmation_file_sha256="f" * 64, fresh=fresh, dependencies=dependencies, noun_keys=["n"] * 79)
    return {"config": config, "W_E": W_E, "bindings": bindings, "record": record, "confirmation": confirmation, "fresh": fresh, "lock": lock,
            "dependencies": dependencies}


def test_the_lock_binds_the_scores_the_thresholds_the_guard_and_the_outcome():
    world = _toy_lock()
    lock, record, fresh = world["lock"], world["record"], world["fresh"]
    assert lock["content_sha256"] == rc.content_digest(lock) and lock["configuration"]["name"] == "toy"
    assert (lock["primary"]["F_rho"], lock["primary"]["null_975"]) == (record["primary_floor"]["F_rho"], record["null"]["null_975"])
    assert lock["guard"]["spec"]["max_upper"] == 1 and [unit["word"] for unit in lock["guard"]["units"]["E"]] == ["e0", "e1", "e2", "e3"]
    assert lock["outcome"]["table"][-1][2] == "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS" and lock["semantics"] == rr.SEMANTICS
    assert len(fresh["cues"]) == 20 and all(cue["predicted_log_mse"] == rr.predict(record["line"], cue["nounness"]) for cue in fresh["cues"])
    maximum = max(entry["nounness_loo"] for entry in record["calibration_cues"])
    assert fresh["maximum_calibration_score"] == maximum and all(cue["above_calibration_maximum"] == (cue["nounness"] > maximum) for cue in fresh["cues"])
    assert fresh["extrapolation"]["sentence"].startswith(f"{fresh['extrapolation']['E_above_maximum']} of the 4 E cues")
    again = rr.fresh_quantities(world["W_E"], world["bindings"], world["confirmation"], record)
    assert again == fresh  # the lock's quantities reproduce exactly (what I7 checks before any prompt)
    text = rr.render_preregistration(lock)
    assert text == rr.render_preregistration(json.loads(json.dumps(lock))) and "exact one-sided permutation test" in text and "Extrapolation" in text
    assert repr(lock["primary"]["F_rho"]) in text and "| E | e0 |" in text and "Simple: the guard eliminates" in text


def _validate(world, *, state, lock=None, record=None, prereg=None, config=None, **overrides):
    lock = lock or world["lock"]
    text = prereg if prereg is not None else rr.render_preregistration(lock)
    arguments = dict(state=state, digests={"x": "y"}, config=config or world["config"], record=record or world["record"], record_file_sha256="e" * 64,
                     confirmation=world["confirmation"], confirmation_file_sha256="f" * 64, dependencies=world["dependencies"], noun_keys=["n"] * 79,
                     preregistration_text=text, git_state={"dirty": False}, tracked=True, changed_paths=[])
    arguments.update(overrides)
    return rr.validate_lock(lock, **arguments)


def test_validate_lock_refuses_every_drift():
    from neural_decompiler import plural_mechanism as pm

    world = _toy_lock()
    lock = world["lock"]
    text = rr.render_preregistration(lock)
    state = {"lock": {"content_sha256": lock["content_sha256"], "preregistration_sha256": pm.sha256_text(text)},
             "confirmation_024": rr.confirmation_binding(world["confirmation"], "f" * 64)}
    _validate(world, state=state)
    resealed = json.loads(json.dumps(lock))
    resealed["primary"]["F_rho"] = -1.0
    resealed["content_sha256"] = rc.content_digest(resealed)
    cases = [(dict(lock=resealed, state=dict(state, lock=dict(state["lock"], content_sha256=resealed["content_sha256"]))), "thresholds"),
             (dict(state=state, prereg=text + "\nedited"), "preregistration"),
             (dict(state=state, config=dataclasses.replace(world["config"], name="other")), "configuration"),
             (dict(state=state, dependencies={**world["dependencies"], "C": "another readout"}), "dependencies"),
             (dict(state=state, changed_paths=["src/neural_decompiler/readout_routing.py"]), "scientific paths changed"),
             (dict(state=state, changed_paths=["experiments/023-block0-completion/exposed-cells.f64"]), "scientific paths changed"),
             (dict(state=state, changed_paths=None), "not an ancestor"), (dict(state=state, tracked=False), "tracked"),
             (dict(state=state, git_state={"dirty": True}), "clean"),
             (dict(state=dict(state, confirmation_024={"path": "x"})), "results state binds")]
    for overrides, message in cases:
        with pytest.raises(rr.PhaseError, match=message):
            _validate(world, **overrides)
    assert rr.scientific_changes(["experiments/023-block0-completion/README.md", "experiments/023-block0-completion/evidence/x.md", rr.LOCK_RELATIVE_PATH, "docs/a.md"]) == []
    with pytest.raises(rr.PhaseError, match="the model"):
        rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256="q" * 64, embedding_sha256=lock["dependencies"]["model"]["embedding_sha256"],
                                     what="the lock")


# ---------------------------------------------------------------------------
# Tier C (the pinned model, opt-in): weights-only score facts, and the measurement path on spent exposed pairs only.

# Requirement 3: the only prompts any 024 test may execute — four exposed pairs spent by Experiments 020 and 022 (every
# key is in 020's ledger), one per stratum and per template. No 024 candidate cue or key can reach the model.
SPENT_REMEASURE_KEYS = ("cardinal-009-1|an|271", "quantifier-009-1|least|1878", "coordinated-adjective-009-1|black|2806", "quantifier-new-2|he|344")


def _smoke_enabled() -> None:
    import os

    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")


@pytest.mark.pythia_smoke
def test_the_real_scores_match_the_design_and_five_e_cues_extrapolate(monkeypatch):
    """Weights only (every forward pass refused for the whole test): the 40 expected picks' scores are design revision
    2's (background item 8, three decimals), every E above every N, and 5 of the 8 E cues above the largest calibration
    leave-one-out score. No MSE, draw or floor is computed; nothing is written."""
    _smoke_enabled()
    from neural_decompiler import plural_mechanism as pm
    from neural_decompiler.models import PYTHIA_70M, load_model

    def refuse(*args, **kwargs):
        raise AssertionError("a forward pass was attempted in a weights-only test")

    for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
        if hasattr(pm, name):
            monkeypatch.setattr(pm, name, refuse)

    tokenizer, inputs, base, c023 = _real_freeze_inputs()
    payload = rr.freeze_payload(tokenizer, pool=inputs.pool, exclusion_base=base, confirmation_023=c023,
                                confirmation_023_file_sha256=rc.file_sha256(ROOT / b0c.CONFIRMATION_RELATIVE_PATH), config=rr.PRODUCTION)
    confirmation = rr.confirmation_from_payload(payload, inputs.pool, rr.PRODUCTION)
    W_E = pm.Weights.from_model(load_model(PYTHIA_70M)).W_E
    units = b0c.exposed_units(inputs)
    bindings = rr.score_bindings(W_E, inputs.pool, units)
    assert len(bindings["noun_row_ids"]) == 158 and len(bindings["calibration_cue_ids"]) == 139
    scores = dict(zip((token["word"] for token in confirmation.tokens), rr.full_scores(W_E, bindings, [token["token_id"] for token in confirmation.tokens])))
    e = {word: scores[word] for word in rr.EXPECTED_PICKS["ordinary"]}
    n = {word: scores[word] for word in rr.EXPECTED_PICKS["N"]}
    assert round(e["lion"], 3) == 0.108 and round(e["horse"], 3) == 0.298 and round(sum(e.values()) / 8, 3) == 0.174 and min(e, key=e.get) == "lion"
    assert round(n["honest"], 3) == -0.229 and round(n["nervous"], 3) == -0.045 and round(sum(n.values()) / 8, 3) == -0.146 and max(n, key=n.get) == "nervous"
    assert min(e.values()) > max(n.values())
    maximum = max(rr.calibration_scores(W_E, bindings))
    assert round(maximum, 3) == 0.135 and sum(1 for value in e.values() if value > maximum) == 5


@pytest.mark.pythia_smoke
def test_the_measurement_path_reproduces_the_calibration_source_on_spent_pairs_only(monkeypatch, capsys, tmp_path):
    """The one-canonical-MSE contract between calibration and confirm: re-measured through 022's ``stage_two_022``, saved,
    re-read and passed through 024's cell function, four spent exposed pairs reproduce 023's committed ``n`` and
    ``SSE_C`` bit for bit, ``C`` recomputes from the saved ``Δx3`` bit for bit and I1, I3 and I4 hold. An explicit
    allow-list guards the model: every key must be spent (in 020's ledger), no key or cue id may belong to 024's
    candidates or would-be manifest (checked before the model loads), and any other capture is refused."""
    _smoke_enabled()
    from neural_decompiler import plural_mechanism as pm
    from neural_decompiler.models import PYTHIA_70M, load_model

    tokenizer, inputs, base, c023 = _real_freeze_inputs()
    ledger = set(inputs.closure["ledger"])
    candidate_ids = {ids[0] for word in (*rr.N_CANDIDATES, *(w for pair in rr.MEASURE_LEMMAS + rr.ORDINARY_LEMMAS for w in pair))
                     if len(ids := pm._encode(tokenizer, " " + word)) == 1}
    payload = rr.freeze_payload(tokenizer, pool=inputs.pool, exclusion_base=base, confirmation_023=c023,
                                confirmation_023_file_sha256=rc.file_sha256(ROOT / b0c.CONFIRMATION_RELATIVE_PATH), config=rr.PRODUCTION)
    manifest_024 = set(payload["manifest"]["S2-TARGET"])
    units = b0c.exposed_units(inputs)
    frames = {frame.frame_id: fi for fi, frame in enumerate(units.frames)}
    pairs = []
    for key in SPENT_REMEASURE_KEYS:
        frame_id, word, token_id = key.split("|")
        ci = next(i for i, (w, t, _) in enumerate(units.cues) if w == word and int(t) == int(token_id))
        pairs.append((ci, frames[frame_id]))
        assert key in ledger and key not in manifest_024 and int(token_id) not in candidate_ids  # before any model is loaded
    allowed = set(SPENT_REMEASURE_KEYS)
    original = pm.capture_prompt
    executed_keys: list[str] = []

    def guarded(model, prompt, sites):
        if prompt.key not in allowed or int(prompt.cue_token_id) in candidate_ids:
            raise AssertionError(f"a non-allow-listed prompt reached the model: {prompt.key}")
        executed_keys.append(prompt.key)
        return original(model, prompt, sites)

    monkeypatch.setattr(pm, "capture_prompt", guarded)
    model = load_model(PYTHIA_70M)
    progs = ul.ModelPrograms.from_model(model, inputs)
    raw = numpy.frombuffer((ROOT / b0c.CELLS_DATA_RELATIVE_PATH).read_bytes(), dtype="<f8").reshape(-1, 8)
    locked = inputs.closure["exploration"]["locked_states"]
    for ci, fi in pairs:
        word, token_id, _ = units.cues[ci]
        frame = units.frames[fi]
        spent = rr.Confirmation024(dict(inputs.pool.reference_ids), (frame,), ({"word": word, "token_id": int(token_id), "class": "spent", "lemma": word, "form": "word"},),
                                   "spent")
        states = ul.y1_states(locked, (frame,))
        saved = tmp_path / f"stage2-{ci}-{fi}.pt"
        torch.save(ul.measurement_tensors(ul.stage_two_022(model, progs, spent, {"Y1": states, "Y2": {}}, executed=[])), saved)
        tensors = torch.load(saved)  # confirm's path: the saved measurements, re-read
        units_spent = rr.target_units(spent)
        rr.assert_measurements(tensors, units_spent)
        assert rr.recompute_c(progs, units_spent, states, tensors)["bitwise_equal"]  # C from the saved Δx3, bit for bit
        rr.enforce_gates(rr.target_gates(progs, spent, units_spent, states, tensors))  # I1, I3, I4 on the real model
        cells = rr.fresh_cue_cells(units_spent, tensors, spent)
        row = raw[ci * len(units.frames) + fi]
        assert float(cells[0, 0, rr.COUNT]) == row[0] and float(cells[0, 0, rr.SSEC]) == row[5], (word, frame.frame_id)
    assert executed_keys == list(SPENT_REMEASURE_KEYS)
    with capsys.disabled():
        print("\nspent keys re-measured (020's ledger; none a 024 candidate):", ", ".join(executed_keys))
