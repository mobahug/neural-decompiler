"""Stable provenance records and canonical artifact serialization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from .behavior import validate_json_safe


@dataclass(frozen=True)
class TokenizerProvenance:
    tokenizer_id: str
    revision: str | None
    bos_token_id: int | None
    eos_token_id: int | None
    bos_token: str | None
    eos_token: str | None
    prepend_bos: bool
    add_special_tokens: bool
    padding_side: str
    truncation: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RuntimeProvenance:
    device: str
    backend: str
    dtype: str
    seed: int
    deterministic_algorithms: bool
    atol: float
    rtol: float
    versions: Mapping[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.atol)
            or not math.isfinite(self.rtol)
            or self.atol < 0
            or self.rtol < 0
        ):
            raise ValueError("numerical tolerance values must be finite and nonnegative")
        validate_json_safe(self.versions, path="runtime.versions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "backend": self.backend,
            "dtype": self.dtype,
            "seed": self.seed,
            "deterministic_algorithms": self.deterministic_algorithms,
            "atol": self.atol,
            "rtol": self.rtol,
            "versions": dict(self.versions),
        }


@dataclass(frozen=True)
class BridgeProvenance:
    compatibility_mode: bool
    use_attn_result: bool

    def to_dict(self) -> dict[str, bool]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class InputProvenance:
    text: str | None
    token_ids: tuple[int, ...]
    token_strings: tuple[str, ...]
    target_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_strings):
            raise ValueError("token_ids and token_strings must have equal length")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": list(self.token_ids),
            "token_strings": list(self.token_strings),
            "target_positions": list(self.target_positions),
        }


@dataclass(frozen=True)
class RunProvenance:
    model_id: str
    requested_revision: str
    resolved_revision: str
    tokenizer: TokenizerProvenance
    runtime: RuntimeProvenance
    bridge: BridgeProvenance
    input: InputProvenance
    git_commit: str | None
    git_dirty: bool | None
    capture_definitions: tuple[Mapping[str, Any], ...] = ()
    intervention_definitions: tuple[Mapping[str, Any], ...] = ()
    raw_measurements: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_json_safe(self.to_dict(), path="run_provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "id": self.model_id,
                "requested_revision": self.requested_revision,
                "resolved_revision": self.resolved_revision,
            },
            "tokenizer": self.tokenizer.to_dict(),
            "runtime": self.runtime.to_dict(),
            "bridge": self.bridge.to_dict(),
            "input": self.input.to_dict(),
            "git": {"commit": self.git_commit, "dirty": self.git_dirty},
            "captures": [dict(item) for item in self.capture_definitions],
            "interventions": [
                dict(item) for item in self.intervention_definitions
            ],
            "raw_measurements": dict(self.raw_measurements),
        }


def canonical_json(value: Any) -> str:
    """Serialize one JSON-safe value deterministically without a newline."""

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    validate_json_safe(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json(path: str | Path, value: Any) -> None:
    """Write canonical UTF-8 JSON with one trailing newline."""

    Path(path).write_text(canonical_json(value) + "\n", encoding="utf-8")


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact contiguous CPU bytes."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_digest requires a torch.Tensor")
    value = tensor.detach().contiguous().cpu()
    header = canonical_json(
        {"dtype": str(value.dtype), "shape": list(value.shape)}
    ).encode("utf-8")
    raw = value.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def collect_versions() -> dict[str, str | None]:
    """Collect relevant dependency versions without importing their modules."""

    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in (
        "torch",
        "transformer-lens",
        "transformers",
        "huggingface-hub",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def collect_git_state(path: str | Path | None = None) -> dict[str, Any]:
    """Return the current commit and dirty state when Git is available."""

    cwd = str(path) if path is not None else None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": bool(status.strip())}
