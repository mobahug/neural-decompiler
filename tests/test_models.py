from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch

from neural_decompiler.models import (
    PYTHIA_70M,
    ModelSpec,
    load_model,
    resolve_dtype,
    resolved_revision,
    seed_runtime,
    validate_runtime,
)


class FakeBridge:
    def __init__(self, revision: str, *, compatibility_mode: bool = False) -> None:
        self.original_model = SimpleNamespace(
            config=SimpleNamespace(_commit_hash=revision)
        )
        self.compatibility_mode = compatibility_mode
        self.training = True

    def eval(self) -> "FakeBridge":
        self.training = False
        return self


def test_pinned_pythia_spec_is_exact_and_immutable() -> None:
    assert PYTHIA_70M.model_id == "EleutherAI/pythia-70m-deduped"
    assert (
        PYTHIA_70M.revision
        == "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
    )
    with pytest.raises(FrozenInstanceError):
        PYTHIA_70M.device = "mps"  # type: ignore[misc]


def test_load_model_passes_explicit_runtime_and_requires_raw_bridge() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    fake = FakeBridge(PYTHIA_70M.revision)

    def loader(name: str, **kwargs: object) -> FakeBridge:
        calls.append((name, kwargs))
        return fake

    loaded = load_model(PYTHIA_70M, loader=loader)

    assert calls == [
        (
            PYTHIA_70M.model_id,
            {
                "revision": PYTHIA_70M.revision,
                "device": "cpu",
                "dtype": torch.float32,
            },
        )
    ]
    assert loaded is fake
    assert fake.training is False


def test_load_model_rejects_wrong_revision() -> None:
    with pytest.raises(RuntimeError, match="Pinned model revision mismatch"):
        load_model(PYTHIA_70M, loader=lambda *_args, **_kwargs: FakeBridge("wrong"))


def test_load_model_rejects_compatibility_mode() -> None:
    bridge = FakeBridge(PYTHIA_70M.revision, compatibility_mode=True)
    with pytest.raises(RuntimeError, match="compatibility mode"):
        load_model(PYTHIA_70M, loader=lambda *_args, **_kwargs: bridge)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", "float16"), ("mps", "bfloat16"), ("tpu", "float32")],
)
def test_invalid_device_dtype_pairs_fail_before_loading(
    device: str, dtype: str
) -> None:
    with pytest.raises(ValueError, match="device/dtype"):
        validate_runtime(replace(PYTHIA_70M, device=device, dtype=dtype))


def test_model_spec_rejects_mutable_revision_names() -> None:
    with pytest.raises(ValueError, match="immutable commit SHA"):
        ModelSpec("example/model", "main")


def test_dtype_resolution_is_closed() -> None:
    assert resolve_dtype("float32") is torch.float32
    with pytest.raises(ValueError, match="Unknown dtype"):
        resolve_dtype("float64")


def test_resolved_revision_finds_wrapped_hugging_face_config() -> None:
    assert resolved_revision(FakeBridge("abc123")) == "abc123"


def test_seed_runtime_repeats_torch_random_values() -> None:
    seed_runtime(17, deterministic_algorithms=False)
    first = torch.rand(3)
    seed_runtime(17, deterministic_algorithms=False)
    second = torch.rand(3)
    assert torch.equal(first, second)
