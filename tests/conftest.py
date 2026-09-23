"""Test tiers: classification, selection and the partition record.

Every collected test is classified into exactly one kind — ``pythia_smoke`` (an opt-in pinned-model contract test:
the contract file, or any test carrying that marker in its file), ``runner`` (an experiment runner's state machine on
a fake, by path) or ``unit`` (everything else: pure scoring, digests, loaders, policies, identities on the small fakes,
committed-artifact reproduction, poison boundaries).
A runner test is ``current`` when its experiment number equals ``CURRENT_EXPERIMENT`` and ``historical`` otherwise.
A few unit tests are ``slow`` (tens of seconds or more); they stay in the current-experiment and gate tiers.

Tiers select by these markers::

    A   unit and not slow                       the normal ``pytest`` (default when nothing else selects)
    B   not historical                          the current experiment: every unit test plus the current runner
    C   not historical, plus the pinned-model contract    the gate before explore / lock / confirm
    D   historical                              every closed experiment's runner test; hours; never a routine gate
    all no tier deselection

Bare ``pytest`` runs tier A. The default tier is not applied when the invocation already selects something itself —
explicit paths or node ids, ``-m`` or ``-k`` — so ``pytest tests/test_experiment_017_runner.py`` runs that file.
``--tier`` always applies, intersecting with any other selection. Markers given in the files themselves
(``@pytest.mark.slow`` etc.) are honoured too; the classification here adds, it never removes.

The full classification of every collected item is kept in ``config.stash[TIERS_KEY]`` before any deselection, so
that ``tests/test_tiers.py`` can check the partition on every run.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

# The experiment under development: the three-digit number whose runner test is ``current``; ``None`` while no
# experiment is open. The environment variable NEURAL_DECOMPILER_CURRENT_EXPERIMENT overrides it.
CURRENT_EXPERIMENT: str | None = "021"

TIERS = ("A", "B", "C", "D", "all")
KINDS = ("unit", "runner", "pythia_smoke")

_RUNNER_FILE = re.compile(r"^test_experiment_(\d{3})(?:_runner)?\.py$|^test_behavior_candidate_screening_runner\.py$")
_SMOKE_FILE = "test_pythia_bridge_contract.py"

# Unit tests that take tens of seconds or more. Marking them here keeps the closed experiments' test files untouched;
# new tests should carry ``@pytest.mark.slow`` in the file instead.
SLOW_TESTS = frozenset(
    {
        "tests/test_block_routing.py::test_routing_statistics_and_every_label_branch_on_synthetic_tables",
    }
)

TIERS_KEY = pytest.StashKey[dict[str, frozenset[str]]]()
SELECTED_TIER_KEY = pytest.StashKey[str]()


def current_experiment() -> str | None:
    value = os.environ.get("NEURAL_DECOMPILER_CURRENT_EXPERIMENT")
    if value is not None:
        value = value.strip()
        return value or None
    return CURRENT_EXPERIMENT


def classify(nodeid: str, own_markers: frozenset[str] = frozenset()) -> frozenset[str]:
    """The tier markers of a test from its node id (repo-relative) and the markers already on it."""
    path = nodeid.split("::", 1)[0]
    name = Path(path).name
    markers: set[str] = set(own_markers)
    if name == _SMOKE_FILE or "pythia_smoke" in own_markers:
        markers.add("pythia_smoke")  # a pinned-model contract test wherever it lives (opt-in; skips itself otherwise)
    elif (match := _RUNNER_FILE.match(name)) is not None:
        markers.add("runner")
        number = match.group(1)
        markers.add("current" if number is not None and number == current_experiment() else "historical")
    else:
        markers.add("unit")
    if nodeid in SLOW_TESTS:
        markers.add("slow")
    return frozenset(markers)


def in_tier(tier: str, markers: frozenset[str]) -> bool:
    if tier == "all":
        return True
    if tier == "A":
        return "unit" in markers and "slow" not in markers
    if tier in ("B", "C"):
        return "historical" not in markers
    if tier == "D":
        return "historical" in markers
    raise ValueError(f"unknown tier {tier!r}")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tier",
        choices=TIERS,
        default=None,
        help="select a test tier: A (unit, fast; the default for a bare invocation), B (current experiment), "
        "C (protocol gate: B plus the pinned-model contract), D (historical runner tests), all",
    )


def _explicit_selection(config: pytest.Config) -> bool:
    """True when the invocation already selects tests itself: paths or node ids, ``-m`` or ``-k``."""
    return bool(
        config.args_source == pytest.Config.ArgsSource.ARGS
        or config.getoption("markexpr")
        or config.getoption("keyword")
    )


def effective_tier(config: pytest.Config) -> str | None:
    tier = config.getoption("--tier")
    if tier is not None:
        return tier
    return None if _explicit_selection(config) else "A"


def pytest_configure(config: pytest.Config) -> None:
    tier = effective_tier(config)
    config.stash[SELECTED_TIER_KEY] = tier or "none"
    if tier == "C" and not os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE"):
        os.environ["NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE"] = "1"  # the gate includes the pinned-model contract


def pytest_report_header(config: pytest.Config) -> list[str]:
    tier = config.stash.get(SELECTED_TIER_KEY, "none")
    current = current_experiment() or "none"
    if tier == "none":
        note = "no tier (explicit selection); pass --tier to intersect"
    else:
        note = f"tier {tier}" + (" (default for a bare invocation; pass --tier or select tests explicitly to change)" if config.getoption("--tier") is None else "")
    return [f"tiers: {note}; current experiment: {current}"]


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    record: dict[str, frozenset[str]] = {}
    for item in items:
        own = frozenset(marker.name for marker in item.iter_markers() if marker.name in KINDS + ("current", "historical", "slow"))
        markers = classify(item.nodeid, own)
        record[item.nodeid] = markers
        for name in markers - own:
            item.add_marker(getattr(pytest.mark, name))
    config.stash[TIERS_KEY] = record
    tier = effective_tier(config)
    if tier is None or tier == "all":
        return
    kept = [item for item in items if in_tier(tier, record[item.nodeid])]
    deselected = [item for item in items if not in_tier(tier, record[item.nodeid])]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def pytest_report_collectionfinish(config: pytest.Config, items: list[pytest.Item]) -> list[str]:
    record = config.stash.get(TIERS_KEY, {})
    counts = {tier: sum(1 for markers in record.values() if in_tier(tier, markers)) for tier in ("A", "B", "D")}
    slow = sum(1 for markers in record.values() if "slow" in markers)
    current = sum(1 for markers in record.values() if "current" in markers)
    smoke = sum(1 for markers in record.values() if "pythia_smoke" in markers)
    return [
        f"tiers over the {len(record)} collected: A {counts['A']} | B {counts['B']} (slow {slow}, current {current}) | D historical {counts['D']} | pinned-model contract {smoke}; selected {len(items)} (tier {config.stash.get(SELECTED_TIER_KEY, 'none')})"
    ]
