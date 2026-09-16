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
from neural_decompiler.interventions import (
    Intervention,
    InterventionOperation,
    InterventionPlan,
    ReplacementSource,
    run_interventions,
)


@pytest.fixture
def tiny_bridge() -> TinyBridge:
    return TinyBridge()


def test_empty_intervention_plan_matches_direct_inference(
    tiny_bridge: TinyBridge,
) -> None:
    tokens = tiny_tokens()
    direct = tiny_bridge(tokens)

    result = run_interventions(tiny_bridge, tokens, InterventionPlan(()))

    assert torch.equal(result.logits, direct)
    assert result.executions == ()
    assert tiny_bridge.last_grad_enabled is False


def test_same_activation_replacement_is_exact_noop(tiny_bridge: TinyBridge) -> None:
    tokens = tiny_tokens()
    target = ComponentRef(ComponentKind.MLP_POST, layer=0)
    request = CaptureRequest(target, positions=(1,))
    baseline = run_capture(tiny_bridge, tokens, CapturePlan((request,)))
    replacement = baseline.activations[request].tensor

    result = run_interventions(
        tiny_bridge,
        tokens,
        InterventionPlan(
            (
                Intervention(
                    target,
                    positions=(1,),
                    operation=InterventionOperation.REPLACE,
                    replacement=replacement,
                    source=ReplacementSource.DIRECT,
                ),
            )
        ),
    )

    assert torch.equal(result.logits, baseline.logits)
    assert result.executions[0].outside_max_abs_change == 0.0
    assert torch.equal(result.executions[0].before, result.executions[0].after)


def test_replacement_changes_only_requested_neuron_slice(
    tiny_bridge: TinyBridge,
) -> None:
    target = ComponentRef(ComponentKind.MLP_NEURON, layer=0, neuron=2)
    replacement = torch.tensor([[[99.0]]], dtype=torch.float32)

    result = run_interventions(
        tiny_bridge,
        tiny_tokens(),
        InterventionPlan(
            (
                Intervention(
                    target,
                    positions=(1,),
                    operation=InterventionOperation.REPLACE,
                    replacement=replacement,
                    source=ReplacementSource.REFERENCE,
                ),
            )
        ),
    )

    execution = result.executions[0]
    assert execution.hook_name == "blocks.0.mlp.out.hook_in"
    assert execution.normalized_positions == (1,)
    assert execution.after.shape == (1, 1, 1)
    assert execution.after.item() == 99.0
    assert execution.outside_max_abs_change == 0.0


def test_zero_intervention_records_explicit_operation(tiny_bridge: TinyBridge) -> None:
    target = ComponentRef(ComponentKind.RESID_PRE, layer=0)
    result = run_interventions(
        tiny_bridge,
        tiny_tokens(),
        InterventionPlan(
            (Intervention(target, (0,), InterventionOperation.ZERO),)
        ),
    )
    execution = result.executions[0]
    assert execution.operation is InterventionOperation.ZERO
    assert torch.count_nonzero(execution.after) == 0
    assert not torch.equal(result.logits, TinyBridge()(tiny_tokens()))


def test_negative_positions_normalize_before_overlap_validation(
    tiny_bridge: TinyBridge,
) -> None:
    target = ComponentRef(ComponentKind.RESID_PRE, layer=0)
    plan = InterventionPlan(
        (
            Intervention(target, (-1,), InterventionOperation.ZERO),
            Intervention(target, (3,), InterventionOperation.ZERO),
        )
    )
    with pytest.raises(ValueError, match="overlap"):
        run_interventions(tiny_bridge, tiny_tokens(), plan)
    assert tiny_bridge.last_grad_enabled is None


def test_different_neurons_at_same_position_are_disjoint(
    tiny_bridge: TinyBridge,
) -> None:
    plan = InterventionPlan(
        (
            Intervention(
                ComponentRef(ComponentKind.MLP_NEURON, layer=0, neuron=0),
                (1,),
                InterventionOperation.ZERO,
            ),
            Intervention(
                ComponentRef(ComponentKind.MLP_NEURON, layer=0, neuron=1),
                (1,),
                InterventionOperation.ZERO,
            ),
        )
    )
    result = run_interventions(tiny_bridge, tiny_tokens(), plan)
    assert len(result.executions) == 2


@pytest.mark.parametrize(
    "component",
    [
        ComponentRef(ComponentKind.ATTN_PATTERN, layer=0),
        ComponentRef(ComponentKind.FINAL_NORM),
        ComponentRef(ComponentKind.LOGITS),
    ],
)
def test_capture_only_components_fail_before_forward(
    tiny_bridge: TinyBridge, component: ComponentRef
) -> None:
    with pytest.raises(ValueError, match="capture-only"):
        run_interventions(
            tiny_bridge,
            tiny_tokens(),
            InterventionPlan(
                (Intervention(component, (0,), InterventionOperation.ZERO),)
            ),
        )
    assert tiny_bridge.last_grad_enabled is None


def test_replacement_requires_exact_shape_and_dtype(tiny_bridge: TinyBridge) -> None:
    target = ComponentRef(ComponentKind.MLP_NEURON, layer=0, neuron=0)
    wrong_shape = Intervention(
        target,
        (0,),
        InterventionOperation.REPLACE,
        replacement=torch.zeros(1, 2, 1),
        source=ReplacementSource.MEAN,
    )
    with pytest.raises(ValueError, match="shape"):
        run_interventions(
            tiny_bridge, tiny_tokens(), InterventionPlan((wrong_shape,))
        )

    wrong_dtype = Intervention(
        target,
        (0,),
        InterventionOperation.REPLACE,
        replacement=torch.zeros(1, 1, 1, dtype=torch.float64),
        source=ReplacementSource.MEAN,
    )
    with pytest.raises(ValueError, match="dtype"):
        run_interventions(
            TinyBridge(), tiny_tokens(), InterventionPlan((wrong_dtype,))
        )


def test_replacement_declaration_requires_tensor_and_source() -> None:
    target = ComponentRef(ComponentKind.RESID_PRE, layer=0)
    with pytest.raises(ValueError, match="replacement tensor and source"):
        Intervention(target, (0,), InterventionOperation.REPLACE)
    with pytest.raises(ValueError, match="must not include"):
        Intervention(
            target,
            (0,),
            InterventionOperation.ZERO,
            replacement=torch.zeros(1),
            source=ReplacementSource.DIRECT,
        )


def test_head_intervention_requires_explicit_result_setting(
    tiny_bridge: TinyBridge,
) -> None:
    target = ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=0)
    plan = InterventionPlan(
        (Intervention(target, (0,), InterventionOperation.ZERO),)
    )
    with pytest.raises(ValueError, match="use_attn_result"):
        run_interventions(tiny_bridge, tiny_tokens(), plan)

    enabled = InterventionPlan(
        plan.interventions, InstrumentationSettings(use_attn_result=True)
    )
    result = run_interventions(tiny_bridge, tiny_tokens(), enabled)
    assert result.executions[0].after.shape == (1, 1, 1, 3)
    assert tiny_bridge.setter_calls == [True, False]
