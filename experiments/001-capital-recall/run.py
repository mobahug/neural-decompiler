#!/usr/bin/env python3
"""Run Experiment 001: observational capital-recall logit lens."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import torch

from neural_decompiler.logit_lens import StageMetric, analyze_cached_prompt


DEFAULT_MODEL = "EleutherAI/pythia-70m-deduped"
DEFAULT_REVISION = "main"
PROMPTS = (
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Finland is", " Helsinki"),
)
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs/experiment-001"


def load_model() -> Any:
    """Load the sole approved model through the raw TransformerBridge path."""

    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers(DEFAULT_MODEL, device="cpu")


def _token_strings(model: Any, tokens: torch.Tensor) -> list[str]:
    strings = model.to_str_tokens(tokens)
    if strings and isinstance(strings[0], list):
        strings = strings[0]
    return [str(token) for token in strings]


def _transitions(stages: Sequence[StageMetric]) -> list[dict[str, object]]:
    return [
        {
            "from_stage": previous.label,
            "to_stage": current.label,
            "target_logit_delta": current.target_logit - previous.target_logit,
            "target_rank_delta": current.target_rank - previous.target_rank,
        }
        for previous, current in zip(stages, stages[1:])
    ]


def derive_observations(
    target_token: str,
    stages: Sequence[StageMetric],
) -> list[str]:
    """Describe the largest adjacent measured changes without causal language."""

    if len(stages) < 2:
        return []
    transitions = _transitions(stages)
    target = target_token.strip()
    logit_high = max(transitions, key=lambda item: float(item["target_logit_delta"]))
    logit_low = min(transitions, key=lambda item: float(item["target_logit_delta"]))
    rank_best = min(transitions, key=lambda item: int(item["target_rank_delta"]))
    rank_worst = max(transitions, key=lambda item: int(item["target_rank_delta"]))

    observations: list[str] = []
    high_delta = float(logit_high["target_logit_delta"])
    if high_delta > 0:
        observations.append(
            f"The {target} target logit increased most strongly from "
            f"{logit_high['from_stage']} to {logit_high['to_stage']} "
            f"({high_delta:+.6f})."
        )
    else:
        observations.append(
            f"The {target} target logit did not increase between adjacent stages; "
            f"the least negative change was from {logit_high['from_stage']} to "
            f"{logit_high['to_stage']} ({high_delta:+.6f})."
        )

    low_delta = float(logit_low["target_logit_delta"])
    if low_delta < 0:
        observations.append(
            f"The {target} target logit decreased most strongly from "
            f"{logit_low['from_stage']} to {logit_low['to_stage']} "
            f"({low_delta:+.6f})."
        )
    else:
        observations.append(
            f"The {target} target logit did not decrease between adjacent stages; "
            f"the smallest increase was from {logit_low['from_stage']} to "
            f"{logit_low['to_stage']} ({low_delta:+.6f})."
        )

    best_delta = int(rank_best["target_rank_delta"])
    best_from = next(stage for stage in stages if stage.label == rank_best["from_stage"])
    best_to = next(stage for stage in stages if stage.label == rank_best["to_stage"])
    if best_delta < 0:
        observations.append(
            f"The {target} target rank improved most strongly from "
            f"{rank_best['from_stage']} to {rank_best['to_stage']} "
            f"({best_from.target_rank} to {best_to.target_rank})."
        )
    else:
        observations.append(
            f"The {target} target rank did not improve between adjacent stages; "
            f"the smallest worsening was from {rank_best['from_stage']} to "
            f"{rank_best['to_stage']} ({best_from.target_rank} to {best_to.target_rank})."
        )

    worst_delta = int(rank_worst["target_rank_delta"])
    worst_from = next(stage for stage in stages if stage.label == rank_worst["from_stage"])
    worst_to = next(stage for stage in stages if stage.label == rank_worst["to_stage"])
    if worst_delta > 0:
        observations.append(
            f"The {target} target rank worsened most strongly from "
            f"{rank_worst['from_stage']} to {rank_worst['to_stage']} "
            f"({worst_from.target_rank} to {worst_to.target_rank})."
        )
    else:
        observations.append(
            f"The {target} target rank did not worsen between adjacent stages; "
            f"the smallest improvement was from {rank_worst['from_stage']} to "
            f"{rank_worst['to_stage']} ({worst_from.target_rank} to {worst_to.target_rank})."
        )
    return observations


def analyze_prompt(model: Any, prompt: str, target_token: str) -> dict[str, object]:
    """Run and measure one prompt without retaining its cache afterward."""

    try:
        target_token_id = int(model.to_single_token(target_token))
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"Target {target_token!r} must tokenize to exactly one token"
        ) from error

    tokens = model.to_tokens(prompt)
    with torch.inference_mode():
        actual_logits, cache = model.run_with_cache(tokens)
        actual_final_logits = actual_logits[0, -1, :]
        projection = analyze_cached_prompt(
            model,
            cache,
            actual_final_logits,
            target_token_id,
        )

    actual_probabilities = torch.softmax(actual_final_logits.float(), dim=-1)
    top1_token_id = int(actual_final_logits.argmax().item())
    target_is_top1 = top1_token_id == target_token_id
    stages = list(projection.stages)
    return {
        "prompt": prompt,
        "tokenized_prompt": _token_strings(model, tokens),
        "token_ids": [int(token_id) for token_id in tokens[0].tolist()],
        "target_token": {"text": target_token, "id": target_token_id},
        "final_model_prediction": {
            "token": str(model.to_string(top1_token_id)),
            "token_id": top1_token_id,
            "logit": float(actual_final_logits[top1_token_id].item()),
            "probability": float(actual_probabilities[top1_token_id].item()),
        },
        "target_is_final_top1": target_is_top1,
        "stages": [asdict(stage) for stage in stages],
        "transitions": _transitions(stages),
        "final_projection_validation": asdict(projection.validation),
        "observations": derive_observations(target_token, stages),
    }


def _resolved_revision(model: Any) -> str | None:
    candidates = (
        model,
        getattr(model, "model", None),
        getattr(model, "hf_model", None),
    )
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)
    return None


def _dependency_versions() -> dict[str, str]:
    packages = {
        "transformer_lens": "transformer-lens",
        "torch": "torch",
        "transformers": "transformers",
        "huggingface_hub": "huggingface-hub",
        "matplotlib": "matplotlib",
    }
    versions = {"python": sys.version.split()[0]}
    versions.update(
        {
            output_name: metadata.version(distribution_name)
            for output_name, distribution_name in packages.items()
        }
    )
    return versions


def build_results(
    model: Any,
    prompts: Iterable[tuple[str, str]] = PROMPTS,
) -> dict[str, object]:
    """Run all prompt cases sequentially and assemble the result document."""

    dtype = str(getattr(model.cfg, "dtype", torch.float32)).removeprefix("torch.")
    prompt_results = [
        analyze_prompt(model, prompt, target_token)
        for prompt, target_token in prompts
    ]
    return {
        "schema_version": "1.0",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": DEFAULT_MODEL,
            "requested_revision": DEFAULT_REVISION,
            "resolved_revision": _resolved_revision(model),
            "device": "cpu",
            "dtype": dtype,
        },
        "dependencies": _dependency_versions(),
        "prompts": prompt_results,
    }


def write_results_json(result: dict[str, object], path: Path) -> None:
    """Write a stable, human-diffable machine result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_trace(result: dict[str, object]) -> None:
    """Print the required human-readable trace."""

    print(f"Model: {result['model']['name']}")
    for prompt_result in result["prompts"]:
        print(f"\nPrompt: {prompt_result['prompt']}")
        print(f"Tokens: {prompt_result['tokenized_prompt']}")
        print(f"Token IDs: {prompt_result['token_ids']}")
        target = prompt_result["target_token"]
        print(f"Target: {target['text']!r} (ID {target['id']})")
        for stage in prompt_result["stages"]:
            print(f"\n{stage['label']}")
            print(f"  Target logit: {stage['target_logit']:.6f}")
            print(f"  Target rank: {stage['target_rank']}")
            print(
                "  Intermediate logit-lens probability: "
                f"{stage['logit_lens_probability']:.8f}"
            )
        prediction = prompt_result["final_model_prediction"]
        print(
            f"\nActual final prediction: {prediction['token']!r} "
            f"(ID {prediction['token_id']}, probability {prediction['probability']:.8f})"
        )
        validation = prompt_result["final_projection_validation"]
        print(
            "Final-projection validation: "
            f"rank_match={validation['target_rank_matches']}, "
            f"top1_match={validation['top1_token_id_matches']}, "
            f"max_abs_logit_difference={validation['max_abs_logit_difference']:.9g}"
        )
        if not validation["logits_allclose"]:
            print(
                "WARNING: raw final projected logits are outside tolerance; "
                f"mean offset={validation['mean_actual_minus_projected_offset']:.9g}, "
                "maximum difference after removing the offset="
                f"{validation['max_abs_difference_after_offset']:.9g}"
            )
        if not prompt_result["target_is_final_top1"]:
            print(
                f"WARNING: Pythia-70M did not predict the intended target "
                f"{target['text']!r} as its top-1 next token."
            )
        print("Observations:")
        for observation in prompt_result["observations"]:
            print(f"  - {observation}")


def main() -> None:
    model = load_model()
    result = build_results(model)
    write_results_json(result, OUTPUT_DIR / "results.json")
    write_plots(result, OUTPUT_DIR)
    write_html_report(result, OUTPUT_DIR)
    print_trace(result)
    print(f"\nWrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
