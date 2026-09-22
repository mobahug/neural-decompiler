"""The tier partition: every collected test lands in exactly one kind and exactly one tier, and the default tier never
reaches a runner, a slow test or the pinned-model contract.

The classification is recorded by ``tests/conftest.py`` for every collected item before any deselection, so these
checks see the whole collection even when only tier A is selected; they are themselves unit tests and run on every
bare ``pytest``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import KINDS, SLOW_TESTS, TIERS_KEY, classify, current_experiment, in_tier

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def record(request) -> dict[str, frozenset[str]]:
    record = request.config.stash.get(TIERS_KEY, None)
    assert record, "tests/conftest.py did not record the collection"
    return record


def test_every_test_has_exactly_one_kind(record):
    for nodeid, markers in record.items():
        kinds = markers & set(KINDS)
        assert len(kinds) == 1, f"{nodeid}: kinds {sorted(kinds)}"


def test_every_runner_test_is_current_xor_historical_and_nothing_else_is(record):
    for nodeid, markers in record.items():
        flags = markers & {"current", "historical"}
        if "runner" in markers:
            assert len(flags) == 1, f"{nodeid}: {sorted(flags)}"
        else:
            assert not flags, f"{nodeid}: {sorted(flags)} without runner"


def test_tiers_partition_the_collection(record):
    a = {n for n, m in record.items() if in_tier("A", m)}
    slow = {n for n, m in record.items() if "unit" in m and "slow" in m}
    current = {n for n, m in record.items() if "current" in m}
    historical = {n for n, m in record.items() if in_tier("D", m)}
    smoke = {n for n, m in record.items() if "pythia_smoke" in m}
    parts = [a, slow, current, historical, smoke]
    for i, left in enumerate(parts):
        for right in parts[i + 1 :]:
            assert not (left & right), sorted(left & right)[:5]
    assert set().union(*parts) == set(record)
    b = {n for n, m in record.items() if in_tier("B", m)}
    assert b == a | slow | current | smoke  # C selects the same node ids as B; the contract test skips itself unless opted in
    assert {n for n, m in record.items() if in_tier("all", m)} == set(record)


def test_the_default_tier_never_reaches_a_runner_a_slow_test_or_the_pinned_model(record):
    for nodeid, markers in record.items():
        if in_tier("A", markers):
            assert not (markers & {"runner", "current", "historical", "slow", "pythia_smoke"}), nodeid
    assert any(in_tier("A", m) for m in record.values())


def test_known_files_land_where_expected():
    assert classify("tests/test_experiment_017_runner.py::test_full_state_machine") >= {"runner", "historical"}
    assert classify("tests/test_experiment_005_runner.py::test_parser") >= {"runner", "historical"}
    assert classify("tests/test_experiment_001.py::test_analyze") >= {"runner", "historical"}
    assert classify("tests/test_behavior_candidate_screening_runner.py::test_cli") >= {"runner", "historical"}
    assert classify("tests/test_block_routing.py::test_frozen_constants_match_the_design") == {"unit"}
    assert classify("tests/test_candidate_screening.py::test_x") == {"unit"}
    assert classify("tests/test_tiers.py::test_x") == {"unit"}
    assert classify("tests/test_pythia_bridge_contract.py::test_pinned") == {"pythia_smoke"}
    assert classify("tests/test_attention_patterns.py::test_program_reproduces_the_pinned_model_rows_on_a_neutral_prompt", frozenset({"pythia_smoke"})) == {"pythia_smoke"}
    assert classify("tests/test_block_routing.py::test_routing_statistics_and_every_label_branch_on_synthetic_tables") == {"unit", "slow"}
    future = classify("tests/test_experiment_042_runner.py::test_x", frozenset({"slow"}))
    assert "runner" in future and "slow" in future and len(future & {"current", "historical"}) == 1  # a runner file is current xor historical whatever CURRENT_EXPERIMENT is


def test_slow_list_names_only_collected_unit_tests(record):
    collected = set(record)
    stale = {nodeid for nodeid in SLOW_TESTS if collected and nodeid.split("::")[0] in {n.split("::")[0] for n in collected} and nodeid not in collected}
    assert not stale, f"SLOW_TESTS entries that no longer exist: {sorted(stale)}"
    for nodeid in SLOW_TESTS:
        assert "unit" in classify(nodeid), nodeid


def test_current_experiment_names_an_existing_runner_test():
    current = current_experiment()
    if current is None:
        return
    assert re.fullmatch(r"\d{3}", current), current
    candidates = [ROOT / "tests" / f"test_experiment_{current}_runner.py", ROOT / "tests" / f"test_experiment_{current}.py"]
    assert any(path.exists() for path in candidates), f"CURRENT_EXPERIMENT={current} but no runner test exists"


def test_the_runner_pattern_matches_every_runner_test_file_on_disk():
    files = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))
    for name in files:
        markers = classify(f"tests/{name}::test_x")
        if name.startswith("test_experiment_") or name.endswith("_runner.py"):
            assert "runner" in markers, name
        elif name == "test_pythia_bridge_contract.py":
            assert markers == {"pythia_smoke"}
        else:
            assert markers == {"unit"}, name  # an in-file pythia_smoke marker moves a single test, never a file, out of unit
