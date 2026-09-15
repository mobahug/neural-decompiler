"""Frozen protocol and integrity contracts for Experiment 004."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence


MODEL_ID = "EleutherAI/pythia-160m-deduped"
MODEL_REVISION = "582159a2dfe3e712a8d47ae83dec95ae3bde8e7e"
DEVELOPMENT_ARTIFACT = "outputs/experiment-004/development-results.json"


class SelectionLockError(RuntimeError):
    """Raised when the Experiment 004 confirmation boundary is invalid."""


@dataclass(frozen=True)
class CapitalPair:
    country: str
    capital: str


@dataclass(frozen=True)
class PromptFormat:
    id: str
    template: str
    rationale: str


DEVELOPMENT_PAIRS = (
    CapitalPair("Spain", "Madrid"),
    CapitalPair("Austria", "Vienna"),
    CapitalPair("Netherlands", "Amsterdam"),
    CapitalPair("Norway", "Oslo"),
    CapitalPair("Poland", "Warsaw"),
    CapitalPair("Greece", "Athens"),
    CapitalPair("Canada", "Ottawa"),
    CapitalPair("China", "Beijing"),
    CapitalPair("Egypt", "Cairo"),
    CapitalPair("Chile", "Santiago"),
    CapitalPair("Iran", "Tehran"),
    CapitalPair("Switzerland", "Bern"),
)

HELDOUT_PAIRS = (
    CapitalPair("Portugal", "Lisbon"),
    CapitalPair("Belgium", "Brussels"),
    CapitalPair("Denmark", "Copenhagen"),
    CapitalPair("Sweden", "Stockholm"),
    CapitalPair("Hungary", "Budapest"),
    CapitalPair("Ireland", "Dublin"),
    CapitalPair("Japan", "Tokyo"),
    CapitalPair("Russia", "Moscow"),
    CapitalPair("Thailand", "Bangkok"),
    CapitalPair("Iraq", "Baghdad"),
    CapitalPair("Philippines", "Manila"),
    CapitalPair("Syria", "Damascus"),
)

PROMPT_FORMATS = (
    PromptFormat("F1", "The capital of {country} is", "Canonical declarative completion."),
    PromptFormat("F2", "{country}'s capital is", "Country-first possessive completion."),
    PromptFormat("F3", "What is the capital of {country}?", "Direct question syntax."),
    PromptFormat(
        "F4",
        "Question: What is the capital of {country}?\nAnswer:",
        "Explicit question and answer boundary.",
    ),
    PromptFormat(
        "F5",
        "Question: What is the capital of France?\nAnswer: Paris\n\n"
        "Question: What is the capital of Germany?\nAnswer: Berlin\n\n"
        "Question: What is the capital of {country}?\nAnswer:",
        "Fixed two-shot question-answer task induction.",
    ),
)

SELECTION_RULE = (
    "higher intended-target top-1 count",
    "lower median intended-target rank",
    "higher mean correct-minus-three-controls final-logit margin",
    "fixed format order F1 < F2 < F3 < F4 < F5",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _pairs_payload(pairs: Sequence[CapitalPair]) -> list[dict[str, str]]:
    return [asdict(pair) for pair in pairs]


def _formats_payload() -> list[dict[str, str]]:
    return [asdict(item) for item in PROMPT_FORMATS]


def protocol_payload() -> dict[str, object]:
    """Return the complete outcome-independent protocol bound by the lock."""

    return {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "split": {
            "development": _pairs_payload(DEVELOPMENT_PAIRS),
            "heldout": _pairs_payload(HELDOUT_PAIRS),
        },
        "candidate_formats": _formats_payload(),
        "selection_rule": list(SELECTION_RULE),
    }


def format_digest() -> str:
    return hashlib.sha256(_canonical_json(_formats_payload())).hexdigest()


def protocol_digest() -> str:
    return hashlib.sha256(_canonical_json(protocol_payload())).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_prompt(format_id: str, country: str) -> str:
    try:
        prompt_format = next(item for item in PROMPT_FORMATS if item.id == format_id)
    except StopIteration as error:
        raise ValueError(f"Unknown prompt format {format_id!r}") from error
    return prompt_format.template.format(country=country)


def select_winner(
    format_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the frozen lexicographic development selection rule."""

    expected_ids = [item.id for item in PROMPT_FORMATS]
    by_id = {str(row.get("format_id")): row for row in format_summaries}
    if len(by_id) != len(format_summaries) or sorted(by_id) != sorted(expected_ids):
        raise ValueError("Development summaries must contain each frozen format exactly once")
    order = {format_id: index for index, format_id in enumerate(expected_ids)}

    def key(row: Mapping[str, object]) -> tuple[float, float, float, int]:
        format_id = str(row["format_id"])
        top1 = int(row["top1_count"])
        median_rank = float(row["median_target_rank"])
        mean_margin = float(row["mean_final_selectivity_margin"])
        if not 0 <= top1 <= len(DEVELOPMENT_PAIRS):
            raise ValueError("Development top-1 count is outside the frozen set size")
        return (-top1, median_rank, -mean_margin, order[format_id])

    return dict(min(format_summaries, key=key))


def build_selection_lock(
    development_results_path: Path,
    *,
    artifact_path: str,
    protocol_code_commit: str,
    timestamp: str,
) -> dict[str, object]:
    """Build, but do not install, the candidate lock for a development artifact."""

    if artifact_path != DEVELOPMENT_ARTIFACT:
        raise ValueError(f"Development artifact path must be {DEVELOPMENT_ARTIFACT!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", protocol_code_commit):
        raise ValueError("Protocol/code commit must be a full lowercase Git SHA")
    if not timestamp:
        raise ValueError("Selection-lock timestamp is required")
    development = json.loads(development_results_path.read_text())
    winner = select_winner(development["format_summaries"])
    protocol = protocol_payload()
    return {
        "schema_version": 1,
        "experiment": "EXPERIMENT-004",
        **protocol,
        "candidate_formats_digest": format_digest(),
        "protocol_digest": protocol_digest(),
        "development_results": {
            "path": artifact_path,
            "sha256": sha256_file(development_results_path),
        },
        "selected_format": str(winner["format_id"]),
        "protocol_code_commit": protocol_code_commit,
        "created_at_utc": timestamp,
    }


def validate_selection_lock(
    lock_path: Path,
    development_results_path: Path,
    *,
    git_state: Mapping[str, object],
) -> dict[str, object]:
    """Validate every boundary required before held-out execution."""

    if not bool(git_state.get("lock_tracked")) or not bool(git_state.get("lock_clean")):
        raise SelectionLockError("Selection lock must be committed and unchanged")
    if not bool(git_state.get("protocol_commit_exists")):
        raise SelectionLockError("Recorded protocol/code commit is unavailable")
    if not bool(git_state.get("protocol_files_unchanged")):
        raise SelectionLockError("Protocol/code files changed after the recorded commit")
    try:
        lock = json.loads(lock_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionLockError("Selection lock is missing or malformed") from error

    expected = protocol_payload()
    if lock.get("model") != expected["model"]:
        raise SelectionLockError("Selection-lock model does not match the frozen model")
    if lock.get("split") != expected["split"]:
        raise SelectionLockError("Selection-lock split does not match the frozen split")
    if lock.get("candidate_formats") != expected["candidate_formats"]:
        raise SelectionLockError("Selection-lock format definitions do not match")
    if lock.get("candidate_formats_digest") != format_digest():
        raise SelectionLockError("Selection-lock format digest does not match")
    if lock.get("selection_rule") != expected["selection_rule"]:
        raise SelectionLockError("Selection-lock selection rule does not match")
    if lock.get("protocol_digest") != protocol_digest():
        raise SelectionLockError("Selection-lock protocol digest does not match")
    if lock.get("development_results", {}).get("path") != DEVELOPMENT_ARTIFACT:
        raise SelectionLockError("Selection-lock development artifact path does not match")
    try:
        actual_digest = sha256_file(development_results_path)
        development = json.loads(development_results_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionLockError("Development artifact is missing or malformed") from error
    if lock["development_results"].get("sha256") != actual_digest:
        raise SelectionLockError("Development artifact digest does not match the lock")
    try:
        winner = select_winner(development["format_summaries"])
    except (KeyError, TypeError, ValueError) as error:
        raise SelectionLockError("Development artifact cannot reproduce a valid winner") from error
    if lock.get("selected_format") != winner["format_id"]:
        raise SelectionLockError("Selected winner is not reproducible from development results")
    commit = lock.get("protocol_code_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SelectionLockError("Recorded protocol/code commit is malformed")
    if lock.get("schema_version") != 1 or lock.get("experiment") != "EXPERIMENT-004":
        raise SelectionLockError("Selection lock schema is malformed")
    if not isinstance(lock.get("created_at_utc"), str) or not lock["created_at_utc"]:
        raise SelectionLockError("Selection lock timestamp is malformed")
    return lock
