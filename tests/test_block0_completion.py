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
