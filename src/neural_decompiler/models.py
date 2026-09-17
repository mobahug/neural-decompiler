"""Pinned TransformerBridge loading and explicit runtime control."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Callable

import torch


_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class ModelSpec:
    """Exact checkpoint and runtime requested for one model load."""

    model_id: str
    revision: str
    device: str = "cpu"
    dtype: str = "float32"
    deterministic_algorithms: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ValueError("revision must be an immutable commit SHA")


@dataclass(frozen=True)
class RuntimeInfo:
    """Validated execution backend and dtype."""

    device: str
    backend: str
    dtype_name: str
    torch_dtype: torch.dtype


PYTHIA_70M = ModelSpec(
    model_id="EleutherAI/pythia-70m-deduped",
    revision="e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
)

PYTHIA_160M = ModelSpec(
    model_id="EleutherAI/pythia-160m-deduped",
    revision="582159a2dfe3e712a8d47ae83dec95ae3bde8e7e",
)


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve the deliberately small Phase 1 dtype vocabulary."""

    try:
        return _DTYPES[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(_DTYPES))
        raise ValueError(f"Unknown dtype {name!r}; expected one of: {allowed}") from exc


def validate_runtime(spec: ModelSpec) -> RuntimeInfo:
    """Validate a device/dtype pair before model weights are loaded."""

    dtype = resolve_dtype(spec.dtype)
    device = spec.device

    if device == "cpu":
        if dtype is not torch.float32:
            raise ValueError("Unsupported device/dtype pair: CPU requires float32")
        return RuntimeInfo(device, "cpu", spec.dtype, dtype)

    if device == "mps":
        if dtype not in (torch.float32, torch.float16):
            raise ValueError(
                "Unsupported device/dtype pair: MPS supports float32 or float16"
            )
        if not torch.backends.mps.is_available():
            raise ValueError("Unsupported device/dtype pair: MPS is unavailable")
        return RuntimeInfo(device, "mps", spec.dtype, dtype)

    if device == "cuda" or re.fullmatch(r"cuda:\d+", device):
        if not torch.cuda.is_available():
            raise ValueError("Unsupported device/dtype pair: CUDA is unavailable")
        if device != "cuda":
            index = int(device.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise ValueError(
                    f"Unsupported device/dtype pair: CUDA index {index} is unavailable"
                )
        return RuntimeInfo(device, "cuda", spec.dtype, dtype)

    raise ValueError(f"Unsupported device/dtype pair: unknown device {device!r}")


def seed_runtime(seed: int, deterministic_algorithms: bool) -> None:
    """Seed Python and PyTorch and set the requested deterministic mode."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def resolved_revision(model: Any) -> str | None:
    """Read the Hugging Face commit hash through common Bridge wrappers."""

    for candidate in (
        model,
        getattr(model, "original_model", None),
        getattr(model, "model", None),
    ):
        config = getattr(candidate, "config", None)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)
    return None


def load_model(
    spec: ModelSpec,
    loader: Callable[..., Any] | None = None,
) -> Any:
    """Load one exact checkpoint through raw TransformerBridge semantics."""

    runtime = validate_runtime(spec)
    if loader is None:
        from transformer_lens.model_bridge import TransformerBridge

        loader = TransformerBridge.boot_transformers

    model = loader(
        spec.model_id,
        revision=spec.revision,
        device=runtime.device,
        dtype=runtime.torch_dtype,
    )
    resolved = resolved_revision(model)
    if resolved != spec.revision:
        raise RuntimeError(
            "Pinned model revision mismatch: "
            f"expected {spec.revision}, received {resolved}"
        )
    if bool(getattr(model, "compatibility_mode", False)):
        raise RuntimeError("TransformerBridge compatibility mode must remain disabled")
    model.eval()
    return model
