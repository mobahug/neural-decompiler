#!/usr/bin/env python3
"""Run Experiment 004 through separate development and locked held-out phases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.util
from importlib import metadata
import json
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import torch

from neural_decompiler.prompt_elicitation import (
    DEVELOPMENT_ARTIFACT,
    DEVELOPMENT_PAIRS,
    HELDOUT_PAIRS,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_FORMATS,
    SelectionLockError,
    build_selection_lock,
    render_prompt,
    select_winner,
    sha256_file,
    validate_selection_lock,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-004"
DEVELOPMENT_PATH = ROOT / DEVELOPMENT_ARTIFACT
CANDIDATE_LOCK_PATH = OUTPUT_DIR / "selection-lock.candidate.json"
LOCK_PATH = Path(__file__).resolve().parent / "selection-lock.json"
PROTOCOL_PATHS = (
    "src/neural_decompiler/prompt_elicitation.py",
    "experiments/004-prompt-elicitation/run.py",
    "experiments/004-prompt-elicitation/README.md",
    "docs/superpowers/specs/2026-09-15-experiment-004-prompt-elicitation-design.md",
)


def _load_experiment_002() -> Any:
    path = ROOT / "experiments/002-capital-selectivity/run.py"
    spec = importlib.util.spec_from_file_location("experiment_002_for_004", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared measurement path from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPERIMENT_002 = _load_experiment_002()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "heldout"))
    return parser


def load_model() -> Any:
    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        dtype=torch.float32,
    )


def _resolved_revision(model: Any) -> str | None:
    for candidate in (model, getattr(model, "model", None), getattr(model, "hf_model", None)):
        revision = getattr(getattr(candidate, "config", None), "_commit_hash", None)
        if revision:
            return str(revision)
    return None


def ensure_model_identity(model: Any) -> str:
    resolved = _resolved_revision(model)
    if resolved != MODEL_REVISION:
        raise RuntimeError(
            f"Pinned revision mismatch: expected {MODEL_REVISION}, received {resolved}"
        )
    if bool(getattr(model, "compatibility_mode", False)):
        raise RuntimeError("TransformerBridge compatibility mode must remain disabled")
    return resolved


def _control_map() -> dict[str, tuple[str, ...]]:
    return {
        pair.country: controls
        for pair, controls in zip(
            EXPERIMENT_002.CAPITAL_PAIRS,
            EXPERIMENT_002.control_pairs(),
            strict=True,
        )
    }


def _default_analyzer(
    model: Any,
    pair: object,
    controls: Sequence[str],
    *,
    format_id: str,
    template: str,
) -> dict[str, object]:
    return EXPERIMENT_002.analyze_case(
        model,
        pair,
        controls,
        template_id=format_id,
        template=template,
    )


def _decorate_case(case: dict[str, object], n_layers: int) -> dict[str, object]:
    depths = (0.0, *(index / n_layers for index in range(1, n_layers + 1)))
    correct = case["candidate_stages"]["correct"]
    if len(correct) != len(depths):
        raise ValueError("Residual stage count does not match model layer count")
    for stage, depth in zip(correct, depths, strict=True):
        stage["normalized_depth"] = depth
    for control in case["candidate_stages"]["controls"]:
        for stage, depth in zip(control["stages"], depths, strict=True):
            stage["normalized_depth"] = depth
    for stage, depth in zip(case["selectivity_stages"], depths, strict=True):
        stage["normalized_depth"] = depth
    case["format_id"] = case.pop("template_id", case.get("format_id"))
    case["final_target"] = {
        "logit": float(correct[-1]["target_logit"]),
        "rank": int(correct[-1]["target_rank"]),
        "probability": float(correct[-1]["logit_lens_probability"]),
    }
    case["final_metrics"]["final_block_target_logit_change"] = (
        float(correct[-1]["target_logit"]) - float(correct[-2]["target_logit"])
    )
    return case


Analyzer = Callable[..., dict[str, object]]


def analyze_cases(
    model: Any,
    pairs: Sequence[object],
    format_ids: Sequence[str],
    *,
    analyzer: Analyzer = _default_analyzer,
    n_layers: int | None = None,
) -> list[dict[str, object]]:
    """Analyze a fixed Cartesian product sequentially."""

    layers = int(model.cfg.n_layers) if n_layers is None else n_layers
    if layers <= 0:
        raise ValueError("Layer count must be positive")
    formats = {item.id: item for item in PROMPT_FORMATS}
    controls_by_country = _control_map()
    cases: list[dict[str, object]] = []
    for format_id in format_ids:
        if format_id not in formats:
            raise ValueError(f"Unknown prompt format {format_id!r}")
        for pair in pairs:
            case = analyzer(
                model,
                pair,
                controls_by_country[pair.country],
                format_id=format_id,
                template=formats[format_id].template,
            )
            cases.append(_decorate_case(case, layers))
    return cases


def summarize_formats(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for prompt_format in PROMPT_FORMATS:
        selected = [case for case in cases if case["format_id"] == prompt_format.id]
        if not selected:
            continue
        summaries.append(
            {
                "format_id": prompt_format.id,
                "case_count": len(selected),
                "top1_count": sum(bool(case["target_is_final_top1"]) for case in selected),
                "median_target_rank": float(median(int(case["final_target"]["rank"]) for case in selected)),
                "mean_final_selectivity_margin": float(
                    mean(float(case["final_metrics"]["final_correct_minus_control_logit"]) for case in selected)
                ),
            }
        )
    return summaries


def _dependency_versions() -> dict[str, str]:
    packages = {
        "transformer_lens": "transformer-lens",
        "torch": "torch",
        "transformers": "transformers",
        "huggingface_hub": "huggingface-hub",
        "matplotlib": "matplotlib",
    }
    result = {"python": sys.version.split()[0]}
    result.update({name: metadata.version(package) for name, package in packages.items()})
    return result


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def assemble_development_results(
    cases: list[dict[str, object]],
    *,
    resolved_revision: str,
    timestamp: str,
    git_commit: str,
) -> dict[str, object]:
    summaries = summarize_formats(cases)
    winner = select_winner(summaries)
    return {
        "schema_version": "1.0",
        "experiment": "EXPERIMENT-004",
        "phase": "development",
        "analysis_status": "exploratory",
        "run_timestamp_utc": timestamp,
        "protocol_code_commit": git_commit,
        "model": {
            "id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_revision": resolved_revision,
            "device": "cpu",
            "dtype": "float32",
            "compatibility_mode": False,
        },
        "format_summaries": summaries,
        "selected_format": winner["format_id"],
        "cases": cases,
    }


def write_development_outputs(
    result: dict[str, object], output_dir: Path = OUTPUT_DIR
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "development-results.json"
    candidate = output_dir / "selection-lock.candidate.json"
    artifact.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lock = build_selection_lock(
        artifact,
        artifact_path=DEVELOPMENT_ARTIFACT,
        protocol_code_commit=str(result["protocol_code_commit"]),
        timestamp=str(result["run_timestamp_utc"]),
    )
    candidate.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    return artifact, candidate


def _git_state(lock_path: Path, protocol_commit: str) -> dict[str, object]:
    relative_lock = str(lock_path.relative_to(ROOT))

    def succeeds(arguments: list[str]) -> bool:
        return subprocess.run(
            arguments, cwd=ROOT, capture_output=True, text=True
        ).returncode == 0

    protocol_files_clean = succeeds(["git", "diff", "--quiet", "--", *PROTOCOL_PATHS]) and succeeds(
        ["git", "diff", "--cached", "--quiet", "--", *PROTOCOL_PATHS]
    )
    return {
        "lock_tracked": succeeds(["git", "ls-files", "--error-unmatch", relative_lock]),
        "lock_clean": succeeds(["git", "diff", "--quiet", "--", relative_lock]) and succeeds(
            ["git", "diff", "--cached", "--quiet", "--", relative_lock]
        ),
        "protocol_commit_exists": succeeds(["git", "cat-file", "-e", f"{protocol_commit}^{{commit}}"]),
        "protocol_files_unchanged": protocol_files_clean and succeeds(
            ["git", "diff", "--quiet", protocol_commit, "HEAD", "--", *PROTOCOL_PATHS]
        ),
    }


def load_verified_lock(
    *, lock_path: Path = LOCK_PATH, development_path: Path = DEVELOPMENT_PATH
) -> dict[str, object]:
    if not lock_path.exists():
        raise SelectionLockError("Selection lock is missing")
    try:
        preliminary = json.loads(lock_path.read_text())
        protocol_commit = str(preliminary["protocol_code_commit"])
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise SelectionLockError("Selection lock is malformed") from error
    return validate_selection_lock(
        lock_path,
        development_path,
        git_state=_git_state(lock_path, protocol_commit),
    )


def analyze_locked_heldout(
    model: Any,
    lock: Mapping[str, object],
    *,
    analyzer: Analyzer = _default_analyzer,
    n_layers: int | None = None,
) -> list[dict[str, object]]:
    selected_format = str(lock["selected_format"])
    return analyze_cases(
        model,
        HELDOUT_PAIRS,
        [selected_format],
        analyzer=analyzer,
        n_layers=n_layers,
    )


def run_development() -> None:
    if LOCK_PATH.exists():
        raise SelectionLockError("Tracked selection lock already exists; refusing to regenerate development")
    model = load_model()
    try:
        revision = ensure_model_identity(model)
        with torch.inference_mode():
            cases = analyze_cases(model, DEVELOPMENT_PAIRS, [item.id for item in PROMPT_FORMATS])
        result = assemble_development_results(
            cases,
            resolved_revision=revision,
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit=_git_head(),
        )
        artifact, candidate = write_development_outputs(result)
        print(f"Development cases: {len(cases)}")
        for row in result["format_summaries"]:
            print(
                f"{row['format_id']}: top-1 {row['top1_count']}/12; "
                f"median rank {row['median_target_rank']:.1f}; "
                f"mean margin {row['mean_final_selectivity_margin']:+.6f}"
            )
        print(f"Selected format: {result['selected_format']}")
        print(f"Development artifact: {artifact}")
        print(f"Candidate lock: {candidate}")
    finally:
        del model
        gc.collect()


def run_heldout() -> None:
    lock = load_verified_lock()
    model = load_model()
    try:
        ensure_model_identity(model)
        with torch.inference_mode():
            cases = analyze_locked_heldout(model, lock)
        print(f"Held-out cases: {len(cases)}")
        print(f"Locked format: {lock['selected_format']}")
    finally:
        del model
        gc.collect()


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.phase == "development":
        run_development()
    else:
        run_heldout()


if __name__ == "__main__":
    main()
