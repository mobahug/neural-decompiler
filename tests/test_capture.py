from __future__ import annotations

import pytest
import torch

from instrumentation_fakes import TinyBridge, tiny_tokens
from neural_decompiler.capture import (
    CapturePlan,
    CaptureRequest,
    InstrumentationSettings,
    run_capture,
)
from neural_decompiler.components import ComponentKind, ComponentRef


@pytest.fixture
def tiny_bridge() -> TinyBridge:
    return TinyBridge()


def test_empty_capture_matches_direct_forward_and_uses_inference_mode(
    tiny_bridge: TinyBridge,
) -> None:
    tokens = tiny_tokens()
    direct = tiny_bridge(tokens)

    result = run_capture(tiny_bridge, tokens, CapturePlan(()))

    assert torch.equal(result.logits, direct)
    assert result.activations == {}
    assert tiny_bridge.last_grad_enabled is False
    assert tiny_bridge.training is False


def test_capture_slices_before_storage(tiny_bridge: TinyBridge) -> None:
    tokens = tiny_tokens()
    request = CaptureRequest(
        ComponentRef(ComponentKind.RESID_PRE, layer=1), positions=(2,)
    )

    result = run_capture(tiny_bridge, tokens, CapturePlan((request,)))

    capture = result.activations[request]
    assert capture.original_shape == (1, 4, tiny_bridge.cfg.d_model)
    assert capture.tensor.shape == (1, 1, tiny_bridge.cfg.d_model)
    assert capture.tensor.device.type == "cpu"
    assert tiny_bridge.fired_hooks == {"blocks.1.hook_in"}


def test_capture_preserves_head_axis_and_uses_scoped_setter(
    tiny_bridge: TinyBridge,
) -> None:
    request = CaptureRequest(
        ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=1), positions=(0,)
    )

    result = run_capture(
        tiny_bridge,
        tiny_tokens(),
        CapturePlan((request,), InstrumentationSettings(use_attn_result=True)),
    )

    assert result.activations[request].tensor.shape == (1, 1, 1, 3)
    assert tiny_bridge.setter_calls == [True, False]
    assert tiny_bridge.cfg.use_attn_result is False
    assert result.hook_settings.requested_use_attn_result is True
    assert result.hook_settings.effective_use_attn_result is True


def test_head_capture_refuses_implicit_hook_enablement(
    tiny_bridge: TinyBridge,
) -> None:
    request = CaptureRequest(
        ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=0), positions=(0,)
    )
    with pytest.raises(ValueError, match="use_attn_result"):
        run_capture(tiny_bridge, tiny_tokens(), CapturePlan((request,)))
    assert tiny_bridge.setter_calls == []


def test_capture_uses_query_and_key_filters(tiny_bridge: TinyBridge) -> None:
    request = CaptureRequest(
        ComponentRef(ComponentKind.ATTN_PATTERN, layer=1, head=0),
        positions=(2,),
        key_positions=(0, -1),
    )

    result = run_capture(tiny_bridge, tiny_tokens(), CapturePlan((request,)))

    capture = result.activations[request]
    assert capture.tensor.shape == (1, 1, 1, 2)
    assert capture.axis_names == (
        "batch",
        "head",
        "query_position",
        "key_position",
    )


def test_capture_rejects_duplicate_requests_before_forward(
    tiny_bridge: TinyBridge,
) -> None:
    request = CaptureRequest(
        ComponentRef(ComponentKind.MLP_POST, layer=0), positions=(1,)
    )
    with pytest.raises(ValueError, match="duplicate"):
        CapturePlan((request, request))


def test_capture_fails_when_declared_hook_does_not_fire(
    tiny_bridge: TinyBridge,
) -> None:
    request = CaptureRequest(ComponentRef(ComponentKind.FINAL_NORM), positions=(0,))
    tiny_bridge.hook_dict.pop("ln_final.hook_out")
    with pytest.raises(ValueError, match="unavailable"):
        run_capture(tiny_bridge, tiny_tokens(), CapturePlan((request,)))


def test_capture_rejects_key_positions_for_non_pattern_component() -> None:
    with pytest.raises(ValueError, match="key_positions"):
        CaptureRequest(
            ComponentRef(ComponentKind.RESID_PRE, layer=0),
            positions=(0,),
            key_positions=(0,),
        )
