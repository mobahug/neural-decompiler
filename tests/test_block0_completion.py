"""Experiment 023's module: constants, frozen dependencies, inherited inputs and the exposed order (tier A); the canonical
scoring path, the draws and the artifact format (tier A); the pinned-model contracts (tier C, opt-in)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from neural_decompiler import block0_completion as b0c
from neural_decompiler import readout_calibration as rc
from neural_decompiler import upstream_localization as ul

ROOT = Path(__file__).parents[1]
TABLE_022 = ROOT / b0c.TABLE_022_RELATIVE_PATH


# ---------------------------------------------------------------------------
# Task 1: constants, frozen dependencies, inherited inputs.


def test_constants_are_the_design_values():
    assert b0c.DESIGN == {"path": "docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md", "revision": 2, "commit": "5b38aba"}
    assert b0c.PLAN == {"path": "docs/superpowers/plans/2026-09-24-experiment-023-block0-completion-plan.md", "revision": 1, "commit": "3a795fb"}
    assert b0c.B == 10_000 and b0c.CROSS_CHECK_DRAWS == 16 and b0c.DRAW_TAG == "023|primary"
    assert b0c.GUARD_MIN == 0.90 and b0c.GAP_MIN == 0.02 and b0c.CEILING_LIMITED_R2 == 0.80
    assert b0c.P0_MASK == 0 and b0c.P1_MASK == {"cue_final": 14, "coordinated": 30} and b0c.FULL_MASK == {"cue_final": 15, "coordinated": 31}
    assert b0c.TOLERANCES == {"E4": 1e-9, "E5": 1e-10, "E6": 1e-10, "kernel": 1e-10, "algebra": 1e-12, "I1": 1e-4, "I3": 1e-4, "I4": 1e-3, "I5": 0.0}
    assert b0c.CELL_COLUMNS == ("n", "S", "Q", "SSE0", "SSE1", "SSEC", "mean", "M2") and b0c.CELLS_VERSION == "023-cells-v1"
    assert b0c.RESULTS == ("NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS")
    assert b0c.CONDITIONS == ("Y1/cue_final", "Y1/coordinated", "Y2/cue_final", "Y2/coordinated")
    assert b0c.STRATA == ("determiner-like", "quantity", "adjective") and b0c.CUE_QUOTA == 8 and b0c.FRAME_QUOTA == 6 and b0c.FRAME_ID_TAG == "023"
    assert b0c.HEAD_ORDER == (5, 0, 7, 6, 3, 2, 4, 1)
    # The 022 masks the program reuses mean what 023 says they mean.
    assert b0c.P1_MASK["cue_final"] == ul.BIT["emb"] | ul.BIT["Bv"] | ul.BIT["Bp"] and b0c.P1_MASK["coordinated"] == b0c.P1_MASK["cue_final"] | ul.BIT["T"]
    assert b0c.FULL_MASK["coordinated"] == ul.FULL_MASK and b0c.FULL_MASK["cue_final"] == ul.CUE_FINAL_FULL_MASK


def test_candidate_lists_are_the_design_lists_verbatim():
    assert b0c.CUE_CANDIDATES["determiner-like"] == ("general", "subsequent", "given", "chosen", "selected", "present", "ultimate", "preceding", "following", "latest",
                                                      "earliest")
    assert b0c.CUE_CANDIDATES["quantity"] == ("tons", "piles", "masses", "stacks", "gross", "net", "scores", "batches", "bundles", "crowds", "herds", "multitude",
                                               "cumulative", "aggregate")
    assert b0c.CUE_CANDIDATES["adjective"] == ("humble", "proud", "shy", "lazy", "busy", "sturdy", "fragile", "shiny", "dusty", "hollow", "noble", "clever", "fuzzy",
                                                "crisp", "stiff", "tender", "polished")
    assert tuple(b0c.CUE_CANDIDATES) == b0c.STRATA and tuple(b0c.FRAME_CANDIDATES) == b0c.TEMPLATES
    assert [len(texts) for texts in b0c.FRAME_CANDIDATES.values()] == [13, 14, 12]
    assert b0c.FRAME_CANDIDATES["cardinal"][:2] == ("The dairy produces {cue}", "The shelter feeds {cue}")
    assert b0c.FRAME_CANDIDATES["coordinated-adjective"][4] == "Rosa and Felix polished {cue} bright"
    assert b0c.CUE_CANDIDATES["adjective"][-1] == "polished"  # last, so that it never repeats a picked coordinated frame's verb


def test_the_reused_022_program_is_pinned_by_blob_and_a_change_refuses(monkeypatch):
    assert b0c.module_blobs() == b0c.FROZEN_BLOBS  # every pinned module, 022's own included, is the committed one
    assert b0c.FROZEN_BLOBS["upstream_localization.py"] == "465856962aa380747d1a4f1338d1d2762d03c9f9"
    assert {name: blob for name, blob in b0c.FROZEN_BLOBS.items() if name != "upstream_localization.py"} == ul.FROZEN_BLOBS
    assert b0c.assert_frozen_blobs() == b0c.FROZEN_BLOBS
    monkeypatch.setitem(b0c.FROZEN_BLOBS, "upstream_localization.py", "0" * 40)
    with pytest.raises(b0c.PhaseError, match="upstream_localization.py"):
        b0c.assert_frozen_blobs()


def test_the_pins_are_literal_and_not_inherited_from_022(monkeypatch):
    monkeypatch.setitem(ul.FROZEN_BLOBS, "models.py", "1" * 40)  # a later edit of 022's table cannot relax 023's pin
    assert b0c.FROZEN_BLOBS["models.py"] == "b1c6f03379af7e0918d0d1a6460a264651603fb2"
    assert b0c.assert_frozen_blobs()["models.py"] == "b1c6f03379af7e0918d0d1a6460a264651603fb2"


def test_experiment_022_inputs_verify_and_a_tampered_file_refuses(tmp_path):
    digests = b0c.verify_022_inputs(ROOT)
    assert digests == {"calibration_022_file": b0c.INHERITED_022["calibration_file_sha256"], "calibration_022_content": b0c.INHERITED_022["calibration_content_sha256"],
                       "confirmation_022_file": b0c.INHERITED_022["confirmation_file_sha256"], "confirmation_022_content": b0c.INHERITED_022["confirmation_content_sha256"],
                       "table_022_file": b0c.INHERITED_022["table_file_sha256"]}
    record = json.loads((ROOT / ul.CALIBRATION_RELATIVE_PATH).read_text())
    assert record["rematerialization"]["table_sha256"]["dc"].startswith("54f0db75")  # the tensor digests the extraction verifies (E1)
    for relative in (ul.CALIBRATION_RELATIVE_PATH, ul.CONFIRMATION_RELATIVE_PATH):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes((ROOT / relative).read_bytes())
    assert b0c.verify_022_inputs(tmp_path) == digests
    target = tmp_path / ul.CONFIRMATION_RELATIVE_PATH
    target.write_text(target.read_text().replace('"experiment":"022"', '"experiment":"022" '))
    with pytest.raises(b0c.PhaseError, match="confirmation"):
        b0c.verify_022_inputs(tmp_path)


def test_the_exposed_order_is_022s_canonical_calibration_order():
    inputs = ul.load_frozen_inputs(ROOT)
    units = b0c.exposed_units(inputs)
    assert len(units.cues) == 175 and len(units.frames) == 108 and units.n_pairs == 18_900
    assert [cls for _, _, cls in units.cues] == sorted((cls for _, _, cls in units.cues), key=ul.CUE_CLASSES.index)
    assert {template: len(indices) for template, indices in units.y2_like.items()} == {template: 14 for template in b0c.TEMPLATES}
    assert [len(units.stratum_cues(stratum)) for stratum in b0c.STRATA] == [45, 45, 49]
    assert len(units.group_frames("cue_final")) == 72 and len(units.group_frames("coordinated")) == 36
    assert units.pair_index(2, 5) == 2 * 108 + 5
    if TABLE_022.exists():  # the local 022 table (read-only here) holds its pairs in exactly this order
        table = torch.load(TABLE_022)
        assert table["cues"] == [word for word, _, _ in units.cues] and table["token_ids"] == [token for _, token, _ in units.cues]
        assert table["classes"] == [cls for _, _, cls in units.cues] and table["frames"] == [frame.frame_id for frame in units.frames]
    reference = ul.units_from(rc.production_pools(inputs.pool).cues, inputs.pool.frames, rc.production_pools(inputs.pool).frames_unscreened,
                              {"classes": {cls: 6 for cls in ul.CUE_CLASSES}, "templates": {template: 6 for template in ul.TEMPLATES}})
    assert tuple(reference.cues) == units.cues and dict(reference.y2_frames) == dict(units.y2_like)  # the counts never change the order


# ---------------------------------------------------------------------------
# Task 2: the canonical scoring path, the draws and the artifact format.


def _frames(templates):
    from types import SimpleNamespace

    return tuple(SimpleNamespace(frame_id=f"{template}-{i:02d}", template_id=template) for i, template in enumerate(templates))


def _toy_units():
    """Four cues per 022 class (the pronoun class included, as in 022's order) and nine frames, three per template."""
    cues = tuple((f"{cls[:3]}{k}", 100 * i + k, cls) for i, cls in enumerate(ul.CUE_CLASSES) for k in range(4))
    frames = _frames(["cardinal"] * 3 + ["coordinated-adjective"] * 3 + ["quantifier"] * 3)
    frames = tuple(sorted(frames, key=lambda frame: frame.frame_id))
    y2_like = {template: tuple(i for i, frame in enumerate(frames) if frame.template_id == template)[:2] for template in b0c.TEMPLATES}
    return b0c.ExposedUnits(cues, frames, y2_like)


def _random_cells(n_pairs, seed=23, n=79, shift=0.7):
    generator = torch.Generator().manual_seed(seed)
    rows, raw = [], []
    for k in range(n_pairs):
        y = torch.randn(n, generator=generator, dtype=torch.float64) * (0.5 + k % 3) - 4.0 + shift * k
        p0 = y + torch.randn(n, generator=generator, dtype=torch.float64)
        p1 = y + 0.1 * torch.randn(n, generator=generator, dtype=torch.float64)
        c = y + 0.09 * torch.randn(n, generator=generator, dtype=torch.float64)
        rows.append(b0c.pair_cells(y, p0, p1, c))
        raw.append((y, p0, p1, c))
    return torch.stack(rows), raw


def test_pair_cells_are_the_sufficient_statistics_computed_by_022s_cell_path():
    y = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    p0, p1, c = torch.zeros(3, dtype=torch.float64), y - 0.5, y + 0.25
    cells = b0c.pair_cells(y, p0, p1, c)
    mean = 7.0 / 3.0
    assert cells[:6].tolist() == [3.0, 7.0, 21.0, 21.0, 0.75, 0.1875]
    assert float(cells[6]) == pytest.approx(mean, abs=1e-15) and float(cells[7]) == pytest.approx(sum((v - mean) ** 2 for v in (1.0, 2.0, 4.0)), abs=1e-14)
    compositions = torch.zeros(32, 3, dtype=torch.float64)
    compositions[14] = p1
    sse, count, mean_022, m2_022 = ul.pair_cells(y, compositions)  # 022's own cell computation on a 32-row table
    assert float(cells[3]) == float(sse[0]) and float(cells[4]) == float(sse[14]) and float(cells[0]) == count
    assert float(cells[6]) == mean_022 and float(cells[7]) == m2_022


def test_pooled_sst_is_the_pooled_variance_with_repeats_never_a_sum_of_pair_variances():
    cells, raw = _random_cells(4)
    index = torch.tensor([0, 2, 2, 3, 2])  # repeats count once per selection
    pooled = b0c.pool(cells, index)
    flat = [torch.cat([raw[i][k] for i in index.tolist()]) for k in range(4)]
    y = flat[0]
    direct_sst = float(((y - y.mean()) ** 2).sum())
    assert float(pooled["N"]) == 5 * 79 and float(pooled["S"]) == pytest.approx(float(y.sum()), rel=1e-14)
    assert float(pooled["SST"]) == pytest.approx(direct_sst, rel=1e-12) and float(pooled["SST_two_pass"]) == pytest.approx(direct_sst, rel=1e-12)
    for key, prediction in (("SSE0", flat[1]), ("SSE1", flat[2]), ("SSEC", flat[3])):
        assert float(pooled[key]) == pytest.approx(float(((y - prediction) ** 2).sum()), rel=1e-13)
    summed_pair_variances = float(cells[index, 7].sum())
    assert direct_sst - summed_pair_variances > 100.0  # the pair means differ: the naive sum is wrong, and the pooled SST is not it
    batched = b0c.pool(cells, torch.stack([index, torch.tensor([1, 1, 1, 0, 3])]))
    assert float(batched["SST"][0]) == float(pooled["SST"]) and float(batched["SSE1"][0]) == float(pooled["SSE1"])


def test_e6_holds_on_consistent_cells_and_raises_on_a_planted_inconsistency():
    cells, _ = _random_cells(4)
    index = torch.tensor([0, 1, 2, 3, 3])
    assert b0c.enforce_e6(b0c.pool(cells, index), "consistent") <= 1e-13
    bad = cells.clone()
    bad[3, 2] += 1e-3  # Q no longer agrees with the two-pass moments
    with pytest.raises(b0c.IncidentError, match="E6"):
        b0c.enforce_e6(b0c.pool(bad, index), "planted")


def test_g_is_unclipped_and_the_gap_rule_decides_interpretability():
    pooled = {"SST": torch.tensor([100.0, 100.0, 100.0, 100.0, 0.0, 100.0, 100.0], dtype=torch.float64),
              "SSE0": torch.tensor([50.0, 50.0, 50.0, 50.0, 1.0, 50.0, 50.0], dtype=torch.float64),
              "SSE1": torch.tensor([5.0, 60.0, 10.0, 12.0, 1.0, 48.0, 48.0], dtype=torch.float64),
              "SSEC": torch.tensor([10.0, 10.0, 10.0, 49.0, 1.0, 48.1, 48.0], dtype=torch.float64)}
    stats = b0c.statistics(pooled)
    assert stats["defined"].tolist() == [True, True, True, False, False, False, True]
    assert float(stats["g"][0]) == pytest.approx(1.125)  # the program beats the ceiling: g > 1 is a valid value
    assert float(stats["g"][1]) == pytest.approx(-0.25)  # g < 0 is a valid value
    assert float(stats["g"][2]) == pytest.approx(1.0) and float(stats["g"][6]) == pytest.approx(1.0)  # exactly at the 0.02 boundary: defined
    assert all(math_isnan(float(stats["g"][i])) for i in (3, 4, 5))
    assert float(stats["gap"][0]) == pytest.approx(0.40) and float(stats["R2_1"][0]) == pytest.approx(0.95) and float(stats["R2_C"][3]) == pytest.approx(0.51)
    assert stats["ceiling_limited"].tolist() == [False, False, False, True, False, True, True]  # R²_C < 0.80, descriptive only


def math_isnan(value: float) -> bool:
    return value != value


def test_the_lower_bound_is_element_249_with_undefined_values_at_minus_infinity_and_250_stop_it():
    assert b0c.lower_rank(10_000) == 250 and b0c.lower_rank(40) == 1
    values = torch.arange(10_000, dtype=torch.float64) / 10_000
    defined = torch.ones(10_000, dtype=torch.bool)
    assert b0c.lower_bound(values, defined) == {"kind": "lower", "rank": 250, "element": 249, "bound": 0.0249}
    defined[5000:5249] = False  # 249 undefined draws count against the tail but leave a finite bound
    assert b0c.lower_bound(values, defined)["bound"] == 0.0  # the 250th value: after the 249 at −∞ comes the smallest defined value, 0.0
    defined[5249] = False  # the 250th undefined draw makes the bound −∞: the calibration stop
    assert b0c.lower_bound(values, defined)["bound"] == float("-inf")
    assert b0c.direction_check({"bound": 0.0249}, values, torch.ones(10_000, dtype=torch.bool))["ok"]
    assert not b0c.direction_check({"bound": 0.9}, values, torch.ones(10_000, dtype=torch.bool))["ok"]


def test_classify_precedence_and_the_effective_threshold():
    assert b0c.classify(None, False, 0.99) == "NOT_INTERPRETABLE"
    assert b0c.classify(0.85, True, 0.50) == "GUARD_FAILURE"  # the guard binds even below a lower envelope
    assert b0c.classify(-0.3, True, 0.99) == "GUARD_FAILURE"
    assert b0c.classify(0.95, True, 0.99) == "ENVELOPE_ONLY_FAILURE"
    assert b0c.classify(0.995, True, 0.99) == "PASS" and b0c.classify(0.90, True, 0.50) == "PASS"
    assert b0c.classify(1.2, True, 0.99) == "PASS"  # g > 1: an ordinary pass, never an incident
    with pytest.raises(b0c.IncidentError, match="non-finite"):
        b0c.classify(float("nan"), True, 0.9)


def test_score_selection_is_the_single_path_from_cells_to_a_result():
    cells, _ = _random_cells(6)
    index = torch.arange(6)
    scored = b0c.score_selection(cells, index, 0.5, "toy")
    pooled = b0c.pool(cells, index)
    stats = b0c.statistics(pooled)
    assert scored["g"] == float(stats["g"]) and scored["SST"] == float(pooled["SST"]) and scored["interpretable"]
    assert scored["result"] == b0c.classify(scored["g"], True, 0.5) and scored["e6"] <= 1e-13
    assert set(scored) == {"N", "SST", "SSE0", "SSE1", "SSEC", "g", "interpretable", "gap", "R2_0", "R2_1", "R2_C", "ceiling_limited", "e6", "result"}
    assert "result" not in b0c.score_selection(cells, index, None, "toy")


def test_draws_are_sha_indexed_with_the_023_tag_and_select_only_the_three_strata():
    import hashlib

    units = _toy_units()
    indices = b0c.draw_indices(units, 7)
    assert {key: tuple(value.shape) for key, value in indices.items()} == {**{f"cue/{s}": (7, 8) for s in b0c.STRATA}, **{f"frame/{t}": (7, 6) for t in b0c.TEMPLATES}}
    expected = int.from_bytes(hashlib.sha256(b"023|primary|3|cue/quantity|5").digest()[:8], "big") % 4
    assert int(indices["cue/quantity"][3, 5]) == expected == b0c.slot_index(3, "cue/quantity", 5, 4)
    assert all(torch.equal(a, b) for a, b in zip(indices.values(), b0c.draw_indices(units, 7).values()))
    stratum_cues = {i for stratum in b0c.STRATA for i in units.stratum_cues(stratum)}
    for population in b0c.POPULATIONS:
        for group in b0c.GROUPS:
            pairs = b0c.draw_pairs(units, indices, population, group, range(7))
            n_frames = len(units.group_frames(group)) if population == "Y1" else (12 if group == "cue_final" else 6)
            assert tuple(pairs.shape) == (7, 24 * n_frames)
            cues, frames = pairs // len(units.frames), pairs % len(units.frames)
            assert set(cues.flatten().tolist()) <= stratum_cues  # never the pronoun class
            assert all(units.group_of(f) == group for f in set(frames.flatten().tolist()))
            if population == "Y2":
                assert set(frames.flatten().tolist()) <= {i for indices_ in units.y2_like.values() for i in indices_}


def test_the_cells_artifact_round_trips_and_refuses_any_mismatch(tmp_path):
    units = _toy_units()
    record = {"rematerialization": {"table_sha256": {"dc": "d" * 64}}}
    meta = b0c.cells_meta(units, ["cat", "dog"], record)
    cells, _ = _random_cells(units.n_pairs)
    data, index = tmp_path / "cells.f64", tmp_path / "cells.json"
    written = b0c.write_cells(data, index, cells, meta, {"module_blob": "x"})
    assert written["total_bytes"] == units.n_pairs * 8 * 8 and written["extraction"] == {"module_blob": "x"}
    read, loaded = b0c.read_cells(data, index, units=units, meta=meta)
    assert torch.equal(read, cells) and loaded["file_sha256"] == rc.file_sha256(data)
    assert loaded["source_022"]["table_file_sha256"] == b0c.INHERITED_022["table_file_sha256"] and loaded["columns"] == list(b0c.CELL_COLUMNS)
    with pytest.raises(b0c.PhaseError, match="construction|order"):
        b0c.read_cells(data, index, units=units, meta={**meta, "nouns": ["cat"]})
    raw = bytearray(data.read_bytes())
    raw[5] ^= 1
    data.write_bytes(bytes(raw))
    with pytest.raises(b0c.PhaseError, match="does not match its index"):
        b0c.read_cells(data, index, units=units, meta=meta)


# ---------------------------------------------------------------------------
# Task 3: the extraction identities, on a synthetic table in 022's format (tier A) and read-only on the real local
# 022 table (tier C, opt-in; no artifact is written and no phase runs).


def _synthetic_table(seed=5):
    """A small table in 022's saved format: 3 cues per 022 class, 3 frames per template, 5 nouns; the per-pair cells
    computed by 022's own ``ul.pair_cells`` exactly as 022's calibration stored them."""
    generator = torch.Generator().manual_seed(seed)
    units = _toy_units()
    cues = units.cues
    n_c, n_f, n_n = len(cues), len(units.frames), 5
    dc = torch.randn(n_c, n_f, n_n, generator=generator, dtype=torch.float64) - 3.0
    dc_hat = dc.unsqueeze(2) + 0.3 * torch.randn(n_c, n_f, 32, n_n, generator=generator, dtype=torch.float64)
    ceiling = dc + 0.05 * torch.randn(n_c, n_f, n_n, generator=generator, dtype=torch.float64)
    sse = torch.empty(n_c, n_f, 32, dtype=torch.float64)
    count, mean, m2 = (torch.empty(n_c, n_f, dtype=torch.float64) for _ in range(3))
    for ci in range(n_c):
        for fi in range(n_f):
            sse[ci, fi], count[ci, fi], mean[ci, fi], m2[ci, fi] = ul.pair_cells(dc[ci, fi], dc_hat[ci, fi])
    gates = {name: torch.rand(n_c, n_f, generator=generator, dtype=torch.float64) for name in ("I1", "I2", "I3", "I4", "I5")}
    table = {"cues": [w for w, _, _ in cues], "token_ids": [t for _, t, _ in cues], "classes": [c for _, _, c in cues], "frames": [f.frame_id for f in units.frames],
             "templates": [f.template_id for f in units.frames], "noun_keys": [f"n{k}" for k in range(n_n)], "dc": dc, "dc_hat": dc_hat, "sse": sse, "count": count,
             "mean": mean, "m2": m2, "ceiling": ceiling, "historical": dc_hat[:, :, 0].clone(), "gates": gates}
    record = {"rematerialization": {"table_sha256": {**{name: rc.tensor_digest(table[name]) for name in ul.CalibrationTable.TENSORS},
                                                     **{f"gate_{name}": rc.tensor_digest(values) for name, values in sorted(gates.items())}}}}
    return table, record, units


def test_the_extraction_identities_hold_on_a_table_in_022s_format():
    table, record, units = _synthetic_table()
    assert b0c.verify_table_022(table, record, units, table["noun_keys"])["passed"]
    cells = b0c.cells_from_table(table, units)
    checks = b0c.verify_cells_against_table(cells, table, units)
    assert checks["E2"]["passed"] and checks["E3"]["passed"]
    fi_coordinated = units.group_frames("coordinated")[0]
    row = units.pair_index(2, fi_coordinated)
    assert float(cells[row, 4]) == float(table["sse"][2, fi_coordinated, 30])  # SSE1 is mask 30 in a coordinated frame, mask 14 in a cue-final one
    assert float(cells[units.pair_index(2, units.group_frames("cue_final")[0]), 4]) == float(table["sse"][2, units.group_frames("cue_final")[0], 14])
    draws = b0c.verify_draws_against_table(cells, table, units, draws=4)
    assert draws["E5"]["passed"] and draws["E5"]["max_difference"] <= 1e-12 and draws["E5"]["n_checked"] == 4 * 4 * 3 and draws["E6"]["passed"]


def test_every_extraction_mismatch_refuses():
    table, record, units = _synthetic_table()
    tampered = dict(table, dc=table["dc"].clone())
    tampered["dc"][0, 0, 0] += 1e-9
    with pytest.raises(b0c.ExtractionMismatch, match="E1.*dc"):
        b0c.verify_table_022(tampered, record, units, table["noun_keys"])
    with pytest.raises(b0c.ExtractionMismatch, match="E1.*frames"):
        b0c.verify_table_022(dict(table, frames=list(reversed(table["frames"]))), record, units, table["noun_keys"])
    with pytest.raises(b0c.ExtractionMismatch, match="E1.*noun_keys"):
        b0c.verify_table_022(table, record, units, list(reversed(table["noun_keys"])))
    cells = b0c.cells_from_table(table, units)
    wrong = cells.clone()
    wrong[3, 4] = float(wrong[3, 4]) * (1 + 1e-15) + 1e-12
    with pytest.raises(b0c.ExtractionMismatch, match="E2.*SSE1"):
        b0c.verify_cells_against_table(wrong, table, units)
    bad = cells.clone()
    bad[1, 2] += 1.0  # Q inconsistent with the two-pass moments: E6 on the single pair
    with pytest.raises(b0c.IncidentError, match="E6"):
        b0c.verify_draws_against_table(bad, table, units, draws=2)


requires_table = pytest.mark.skipif(not TABLE_022.exists(), reason="the local Experiment 022 calibration table is not present")


@pytest.mark.pythia_smoke
@requires_table
def test_real_022_table_extraction_identities_read_only():
    """E1–E3, E5 and E6 on the real local 022 table, read-only: the implementation reproduces 022's stored cells bit for
    bit before any extraction runs. Nothing is written."""
    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")
    inputs = ul.load_frozen_inputs(ROOT)
    units = b0c.exposed_units(inputs)
    record = json.loads((ROOT / ul.CALIBRATION_RELATIVE_PATH).read_text())
    table = torch.load(TABLE_022)
    assert rc.file_sha256(TABLE_022) == b0c.INHERITED_022["table_file_sha256"]
    assert b0c.verify_table_022(table, record, units, table["noun_keys"])["passed"]
    cells = b0c.cells_from_table(table, units)
    checks = b0c.verify_cells_against_table(cells, table, units)
    assert checks["E2"]["passed"] and checks["E3"]["passed"]
    draws = b0c.verify_draws_against_table(cells, table, units)
    assert draws["E5"]["max_difference"] <= 1e-10 and draws["E6"]["max_per_pair"] <= 1e-10 and draws["E6"]["max_per_draw"] <= 1e-10


@pytest.mark.pythia_smoke
@requires_table
def test_real_022_p1_recomputes_bit_for_bit_on_a_handful_of_exposed_pairs():
    """E4 on a deterministic handful of exposed pairs (every template, both groups): 022's own functions, fed the
    pinned weights and 020's locked states, reproduce the stored P1 — and the stored P0 — exactly. Weights only."""
    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")
    from neural_decompiler import readout_decompilation as rd
    from neural_decompiler.models import PYTHIA_70M, load_model

    inputs = ul.load_frozen_inputs(ROOT)
    units = b0c.exposed_units(inputs)
    table = torch.load(TABLE_022)
    progs = ul.ModelPrograms.from_model(load_model(PYTHIA_70M), inputs)
    locked = inputs.closure["exploration"]["locked_states"]
    picks = [(0, 0), (44, 40), (90, 75), (174, 107)] + [(ci, units.group_frames("coordinated")[k]) for ci, k in ((7, 0), (130, 20))]
    for ci, fi in picks:
        frame = units.frames[fi]
        word, token_id, _ = units.cues[ci]
        state = rd.state_from_locked(locked[frame.frame_id], frame)
        ctx = ul.pair_context(progs, frame, state, int(inputs.pool.reference_ids[frame.template_id]), int(token_id), word)
        for mask in (b0c.P0_MASK, b0c.P1_MASK[units.group_of(fi)]):
            recomputed = ul.contrast_of(progs, state, ul.compose_dx3(progs, ctx, mask))
            assert torch.equal(recomputed, table["dc_hat"][ci, fi, mask]), (word, frame.frame_id, mask)


def test_extract_cells_runs_the_identities_in_order_and_writes_a_verifiable_artifact(tmp_path):
    table, record, units = _synthetic_table()
    calls = []

    def recompute():
        calls.append("E4")
        return {"max": 0.0, "at": "", "passed": True}

    result = b0c.extract_cells(table, record, units, table["noun_keys"], recompute=recompute)
    assert calls == ["E4"] and set(result["checks"]) == {"E1", "E2", "E3", "E4", "E5", "E6"}
    data, index = tmp_path / "c.f64", tmp_path / "c.json"
    b0c.write_cells(data, index, result["cells"], result["meta"], {"checks": result["checks"]})
    cells, loaded = b0c.read_cells(data, index, units=units, meta=b0c.cells_meta(units, table["noun_keys"], record))
    assert torch.equal(cells, result["cells"]) and loaded["extraction"]["checks"]["E2"]["passed"]
    with pytest.raises(b0c.ExtractionMismatch, match="E1"):
        b0c.extract_cells(dict(table, noun_keys=["x"] * 5), record, units, table["noun_keys"], recompute=recompute)
    assert calls == ["E4"]  # an E1 failure stops before the model is ever consulted


# ---------------------------------------------------------------------------
# Task 4: the tokenizer-only freeze (a stub tokenizer, tier A; the real tokenizer, tier C).


def _confirmation_022(tokenizer, words=("general",), texts=("The dairy produces {cue}",)):
    """A stand-in for Experiment 022's committed confirmation: its cues and frame texts are excluded from 023."""
    return {"cues": [{"word": word, "token_id": tokenizer.id(" " + word), "class": "determiner-like"} for word in words],
            "frames": [{"text_template": text, "cue_ids": {"sg": tokenizer.id(" one"), "pl": tokenizer.id(" two")}} for text in texts], "content_sha256": "2" * 64}


def test_the_freeze_takes_the_first_eligible_entries_and_excludes_022s_units():
    from test_upstream_localization import StubTokenizer, _stub_inputs

    tokenizer = StubTokenizer()
    inputs = _stub_inputs(tokenizer, planted=("humble",))
    payload = b0c.freeze_payload(tokenizer, inputs, _confirmation_022(tokenizer), "f" * 64, model={"model_id": "stub", "revision": "x"})
    words = {stratum: [entry["word"] for entry in payload["cues"] if entry["class"] == stratum] for stratum in b0c.STRATA}
    assert words["determiner-like"] == ["subsequent", "given", "chosen", "selected", "present", "ultimate", "preceding", "following"]  # general: 022's
    assert words["quantity"] == ["tons", "piles", "masses", "stacks", "gross", "net", "scores", "batches"]
    assert words["adjective"] == ["proud", "shy", "lazy", "busy", "sturdy", "fragile", "shiny", "dusty"]  # humble planted in an earlier confirmation
    cardinal = [entry for entry in payload["frames"] if entry["template_id"] == "cardinal"]
    assert cardinal[0]["text_template"] == "The shelter feeds {cue}" and [entry["frame_id"] for entry in cardinal] == [f"cardinal-023-{k}" for k in range(1, 7)]
    reasons = {(entry["candidate"], entry["reason"].split(" ")[0]) for entry in payload["rejected"]}
    assert ("general", "token") in reasons and ("humble", "token") in reasons and ("The dairy produces {cue}", "text") in reasons
    assert payload["exclusion"]["sources"][-1]["source"] == "confirmation-022" and payload["scope"] == b0c.SCOPE
    assert payload["counts"] == {"classes": {stratum: 8 for stratum in b0c.STRATA}, "templates": {template: 6 for template in b0c.TEMPLATES}}
    coordinated = [entry for entry in payload["frames"] if entry["template_id"] == "coordinated-adjective"]
    assert all(entry["p_t"] == entry["p_c"] + 1 for entry in coordinated)
    confirmation = b0c.confirmation_from_payload(payload, inputs.pool)
    manifest = confirmation.manifest()
    assert len(manifest["S1-REF"]) == 18 and len(manifest["S1-VALIDITY"]) == 18 and len(manifest["S2-TARGET"]["Y2"]) == 24 * 18
    assert len(manifest["S2-TARGET"]["Y1"]) == 24 * len(inputs.pool.frames) and len(confirmation.manifest_keys()) == 36 + 24 * (len(inputs.pool.frames) + 18)


def test_a_freeze_shortfall_writes_nothing_and_is_never_refilled():
    from test_upstream_localization import StubTokenizer, _stub_inputs

    tokenizer = StubTokenizer()
    inputs = _stub_inputs(tokenizer)
    used = _confirmation_022(tokenizer, words=("tons", "piles", "masses", "stacks", "gross", "net", "scores"))  # 7 of the 14 quantity words
    with pytest.raises(b0c.FreezeShortfall, match="quantity: 7 eligible of 8"):
        b0c.freeze_payload(tokenizer, inputs, used, "f" * 64)


def test_a_tampered_or_stale_confirmation_file_is_refused(tmp_path):
    from test_upstream_localization import StubTokenizer, _stub_inputs

    from neural_decompiler import plural_mechanism as pm

    tokenizer = StubTokenizer()
    inputs = _stub_inputs(tokenizer)
    c022 = _confirmation_022(tokenizer)
    payload = b0c.freeze_payload(tokenizer, inputs, c022, "f" * 64)
    path = tmp_path / "confirmation-v1.json"
    path.write_text(pm.canonical_json(payload) + "\n")
    assert len(b0c.load_confirmation_023(path, inputs, c022, "f" * 64).tokens) == 24
    for mutate in (lambda p: p["cues"][0].update(word="other"), lambda p: p["manifest"]["S1-REF"].pop(), lambda p: p.update(counts={}), lambda p: p.update(experiment="022")):
        changed = json.loads(path.read_text())
        mutate(changed)
        changed["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in changed.items() if k != "content_sha256"}))
        (tmp_path / "bad.json").write_text(pm.canonical_json(changed) + "\n")
        with pytest.raises(b0c.PhaseError):
            b0c.load_confirmation_023(tmp_path / "bad.json", inputs, c022, "f" * 64)
    with pytest.raises(b0c.PhaseError, match="sources|exclusion"):
        b0c.load_confirmation_023(path, inputs, c022, "e" * 64)  # 022's confirmation file changed since the freeze
    later = _confirmation_022(tokenizer, words=("general", payload["cues"][0]["word"]))  # a unit that became used after the freeze
    with pytest.raises(b0c.PhaseError):
        b0c.load_confirmation_023(path, inputs, later, "f" * 64)


@pytest.mark.pythia_smoke
def test_the_real_tokenizer_freeze_takes_the_designs_expected_units_and_writes_nothing():
    """Task 4's contract: with the real tokenizer and the real exclusion sources (022's confirmation included), in memory
    only, the picks are the ones design revision 2 documents. Nothing is written; the freeze phase is a later,
    authorized step."""
    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")
    from transformers import AutoTokenizer

    from neural_decompiler.models import PYTHIA_70M

    tokenizer = AutoTokenizer.from_pretrained(PYTHIA_70M.model_id, revision=PYTHIA_70M.revision, local_files_only=True)
    inputs = ul.load_frozen_inputs(ROOT)
    c022_path = ROOT / ul.CONFIRMATION_RELATIVE_PATH
    target = ROOT / b0c.CONFIRMATION_RELATIVE_PATH
    existed = target.exists()
    payload = b0c.freeze_payload(tokenizer, inputs, json.loads(c022_path.read_text()), rc.file_sha256(c022_path))
    assert target.exists() == existed
    expected = {"determiner-like": ["general", "subsequent", "given", "chosen", "selected", "present", "ultimate", "preceding"],
                "quantity": ["tons", "piles", "masses", "stacks", "gross", "net", "scores", "batches"],
                "adjective": ["humble", "proud", "shy", "lazy", "busy", "sturdy", "fragile", "shiny"]}
    assert {stratum: [entry["word"] for entry in payload["cues"] if entry["class"] == stratum] for stratum in b0c.STRATA} == expected
    assert {template: [entry["text_template"] for entry in payload["frames"] if entry["template_id"] == template] for template in b0c.TEMPLATES} == \
        {template: list(texts[:6]) for template, texts in b0c.FRAME_CANDIDATES.items()}
    assert payload["rejected"] == [] and len(payload["exclusion"]["cue_token_ids"]) == 327 and len(payload["exclusion"]["frame_texts"]) == 144
    assert all(entry["p_t"] == entry["p_c"] + (1 if entry["template_id"] == ul.COORDINATED else 0) for entry in payload["frames"])


# ---------------------------------------------------------------------------
# Task 5: the calibration from the cells alone.


def _toy_cells(units, seed=11, gap_scale=1.0):
    cells, _ = _random_cells(units.n_pairs, seed=seed, shift=0.02)
    if gap_scale != 1.0:  # shrink the explainable gap SSE0 − SSEC of every pair
        cells[:, 3] = cells[:, 5] + gap_scale * (cells[:, 3] - cells[:, 5])
    return cells


def test_the_calibration_kernel_is_the_canonical_scoring_path_draw_by_draw():
    units = _toy_units()
    cells = _toy_cells(units)
    indices = b0c.draw_indices(units, 30)
    kernel = b0c.calibration_kernel(cells, units, indices, 30, chunk=7)
    for population in b0c.POPULATIONS:
        for group in b0c.GROUPS:
            pairs = b0c.draw_pairs(units, indices, population, group, range(30))
            for b in (0, 13, 29):
                scored = b0c.score_selection(cells, pairs[b], None, "draw")  # the confirmation's path on the same selection
                entries = kernel["conditions"][f"{population}/{group}"]
                assert scored["interpretable"] == bool(entries["defined"][b]) and scored["SST"] == float(entries["SST"][b])
                assert (scored["g"] if scored["g"] is not None else float("nan")) == pytest.approx(float(entries["g"][b]), abs=0.0, nan_ok=True)
                assert scored["R2_C"] == float(entries["R2_C"][b])
    check = b0c.kernel_loop_check(cells, units, indices, kernel, n_draws=5)
    assert check["passed"] and check["n_checked"] == 4 * 5 * 3 and check["max_difference"] <= 1e-12 and kernel["e6_max"] <= 1e-12
    kernel["conditions"]["Y2/coordinated"]["g"][3] += 1e-6  # a planted kernel error
    with pytest.raises(b0c.KernelCheckError, match="Y2/coordinated/draw 3/g"):
        b0c.kernel_loop_check(cells, units, indices, kernel, n_draws=5)


def test_evaluate_sets_the_element_249_envelope_rates_and_the_joint_rate():
    units = _toy_units()
    cells = _toy_cells(units)
    kernel = b0c.calibration_kernel(cells, units, b0c.draw_indices(units, 400), 400)
    evaluated = b0c.evaluate(kernel)
    for condition in b0c.CONDITIONS:
        entry = evaluated["conditions"][condition]
        values = kernel["conditions"][condition]["g"]
        assert entry["envelope"] == {"kind": "lower", "rank": 10, "element": 9, "bound": float(torch.sort(values).values[9])}
        assert entry["direction_check"]["ok"] and abs(sum(entry["rates"].values()) - 1.0) < 1e-12
        assert entry["guard_bound"] == (entry["envelope"]["bound"] < 0.90)
    assert 0.0 <= evaluated["joint_rates"]["all_four_pass"] <= 1.0 and evaluated["joint_rates"]["descriptive_only"]


def test_too_many_undefined_draws_stop_the_calibration_for_review():
    units = _toy_units()
    cells = _toy_cells(units, gap_scale=1e-4)  # the ceiling barely beats Level 0: the gap rule fails nearly everywhere
    kernel = b0c.calibration_kernel(cells, units, b0c.draw_indices(units, 40), 40)
    counts = b0c.undefined_counts(kernel)
    stop = b0c.calibration_stop(counts, 40)
    assert stop["stop"] and stop["threshold"] == 1 and set(stop["offending"]) == set(b0c.CONDITIONS)
    assert not b0c.calibration_stop({condition: 249 for condition in b0c.CONDITIONS}, 10_000)["stop"]
    assert b0c.calibration_stop({**{condition: 0 for condition in b0c.CONDITIONS}, "Y2/cue_final": 250}, 10_000)["offending"] == {"Y2/cue_final": 250}


def test_the_calibration_record_verifies_and_refuses_tampering():
    units = _toy_units()
    cells = _toy_cells(units)
    indices = b0c.draw_indices(units, 40)
    kernel = b0c.calibration_kernel(cells, units, indices, 40)
    arrays = b0c.draw_arrays(kernel)
    record = b0c.calibration_record(run_id="r", protocol_code_commit="a" * 40, digests={"x": "y"}, units=units, cells_files={"data_sha256": "d", "index_sha256": "i"},
                                    confirmation={"content_sha256": "c"}, index_digests=b0c.draw_index_digests(indices),
                                    kernel_check=b0c.kernel_loop_check(cells, units, indices, kernel, n_draws=2), e6_max=kernel["e6_max"],
                                    undefined=b0c.undefined_counts(kernel), evaluated=b0c.evaluate(kernel),
                                    array_digests={key: rc.tensor_digest(value) for key, value in arrays.items()}, draws=40)
    b0c.verify_calibration_record(record, draws=40)
    assert record["constants"]["lower_rank"] == 1 and record["pools"]["cues"] == {stratum: 4 for stratum in b0c.STRATA}
    assert set(arrays) == {f"{condition}/{key}" for condition in b0c.CONDITIONS for key in b0c.KERNEL_KEYS}
    tampered = json.loads(json.dumps(record))
    tampered["conditions"]["Y1/cue_final"]["envelope"]["bound"] = 0.0
    with pytest.raises(b0c.PhaseError, match="verified"):
        b0c.verify_calibration_record(tampered)
    with pytest.raises(b0c.PhaseError, match="constants"):
        b0c.verify_calibration_record(record, draws=10_000)
