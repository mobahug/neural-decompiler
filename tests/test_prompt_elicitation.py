from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_decompiler.prompt_elicitation import (
    DEVELOPMENT_PAIRS,
    HELDOUT_PAIRS,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_FORMATS,
    SelectionLockError,
    build_selection_lock,
    format_digest,
    render_prompt,
    select_winner,
    sha256_file,
    validate_selection_lock,
)


def _summaries() -> list[dict[str, object]]:
    return [
        {"format_id": "F1", "top1_count": 0, "median_target_rank": 8.0, "mean_final_selectivity_margin": 1.0},
        {"format_id": "F2", "top1_count": 2, "median_target_rank": 6.0, "mean_final_selectivity_margin": 2.0},
        {"format_id": "F3", "top1_count": 2, "median_target_rank": 5.0, "mean_final_selectivity_margin": 1.0},
        {"format_id": "F4", "top1_count": 2, "median_target_rank": 5.0, "mean_final_selectivity_margin": 3.0},
        {"format_id": "F5", "top1_count": 2, "median_target_rank": 5.0, "mean_final_selectivity_margin": 3.0},
    ]


def _write_development(path: Path) -> None:
    path.write_text(json.dumps({"format_summaries": _summaries()}, indent=2) + "\n")


def _git_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "lock_tracked": True,
        "lock_clean": True,
        "protocol_commit_exists": True,
        "protocol_files_unchanged": True,
    }
    state.update(overrides)
    return state


def test_frozen_split_and_formats_render_exact_prompts() -> None:
    assert [pair.country for pair in DEVELOPMENT_PAIRS] == [
        "Spain", "Austria", "Netherlands", "Norway", "Poland", "Greece",
        "Canada", "China", "Egypt", "Chile", "Iran", "Switzerland",
    ]
    assert [pair.country for pair in HELDOUT_PAIRS] == [
        "Portugal", "Belgium", "Denmark", "Sweden", "Hungary", "Ireland",
        "Japan", "Russia", "Thailand", "Iraq", "Philippines", "Syria",
    ]
    assert [item.id for item in PROMPT_FORMATS] == ["F1", "F2", "F3", "F4", "F5"]
    assert render_prompt("F1", "Spain") == "The capital of Spain is"
    assert render_prompt("F2", "Spain") == "Spain's capital is"
    assert render_prompt("F3", "Spain") == "What is the capital of Spain?"
    assert render_prompt("F4", "Spain") == "Question: What is the capital of Spain?\nAnswer:"
    assert render_prompt("F5", "Spain").endswith(
        "Question: What is the capital of Spain?\nAnswer:"
    )
    assert "Answer: Paris" in render_prompt("F5", "Spain")
    assert "Answer: Berlin" in render_prompt("F5", "Spain")


def test_selection_uses_every_predeclared_tie_break() -> None:
    summaries = _summaries()
    assert select_winner(summaries)["format_id"] == "F4"

    summaries[1]["top1_count"] = 3
    assert select_winner(summaries)["format_id"] == "F2"

    summaries = _summaries()
    summaries[2]["median_target_rank"] = 4.0
    assert select_winner(summaries)["format_id"] == "F3"

    summaries = _summaries()
    summaries[4]["mean_final_selectivity_margin"] = 4.0
    assert select_winner(summaries)["format_id"] == "F5"


def test_lock_round_trip_binds_protocol_artifact_and_git_state(tmp_path: Path) -> None:
    development = tmp_path / "development-results.json"
    lock_path = tmp_path / "selection-lock.json"
    _write_development(development)
    lock = build_selection_lock(
        development,
        artifact_path="outputs/experiment-004/development-results.json",
        protocol_code_commit="a" * 40,
        timestamp="2026-09-15T00:00:00+00:00",
    )
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    validated = validate_selection_lock(
        lock_path,
        development,
        git_state=_git_state(),
    )

    assert validated["selected_format"] == "F4"
    assert validated["model"] == {"id": MODEL_ID, "revision": MODEL_REVISION}
    assert validated["candidate_formats_digest"] == format_digest()
    assert validated["development_results"]["sha256"] == sha256_file(development)
    assert validated["protocol_code_commit"] == "a" * 40


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda lock: lock.update(selected_format="F5"), "winner"),
        (lambda lock: lock.update(candidate_formats_digest="0" * 64), "format"),
        (lambda lock: lock["model"].update(revision="wrong"), "model"),
        (lambda lock: lock["split"]["heldout"].append({"country": "France", "capital": "Paris"}), "split"),
    ],
)
def test_lock_rejects_tampered_protocol(
    tmp_path: Path, mutation: object, message: str
) -> None:
    development = tmp_path / "development-results.json"
    lock_path = tmp_path / "selection-lock.json"
    _write_development(development)
    lock = build_selection_lock(
        development,
        artifact_path="outputs/experiment-004/development-results.json",
        protocol_code_commit="b" * 40,
        timestamp="2026-09-15T00:00:00+00:00",
    )
    mutation(lock)
    lock_path.write_text(json.dumps(lock))

    with pytest.raises(SelectionLockError, match=message):
        validate_selection_lock(lock_path, development, git_state=_git_state())


def test_lock_rejects_changed_development_artifact(tmp_path: Path) -> None:
    development = tmp_path / "development-results.json"
    lock_path = tmp_path / "selection-lock.json"
    _write_development(development)
    lock = build_selection_lock(
        development,
        artifact_path="outputs/experiment-004/development-results.json",
        protocol_code_commit="c" * 40,
        timestamp="2026-09-15T00:00:00+00:00",
    )
    lock_path.write_text(json.dumps(lock))
    development.write_text(json.dumps({"format_summaries": list(reversed(_summaries()))}))

    with pytest.raises(SelectionLockError, match="digest"):
        validate_selection_lock(lock_path, development, git_state=_git_state())


@pytest.mark.parametrize(
    "git_state",
    [
        _git_state(lock_tracked=False),
        _git_state(lock_clean=False),
        _git_state(protocol_commit_exists=False),
        _git_state(protocol_files_unchanged=False),
    ],
)
def test_lock_rejects_uncommitted_or_diverged_git_state(
    tmp_path: Path, git_state: dict[str, object]
) -> None:
    development = tmp_path / "development-results.json"
    lock_path = tmp_path / "selection-lock.json"
    _write_development(development)
    lock = build_selection_lock(
        development,
        artifact_path="outputs/experiment-004/development-results.json",
        protocol_code_commit="d" * 40,
        timestamp="2026-09-15T00:00:00+00:00",
    )
    lock_path.write_text(json.dumps(lock))

    with pytest.raises(SelectionLockError, match="committed|changed|commit"):
        validate_selection_lock(lock_path, development, git_state=git_state)
