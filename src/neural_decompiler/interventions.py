"""Auditable zero and exact replacement interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from .capture import (
    CapturedActivation,
    CapturePlan,
    CaptureRequest,
    HookSettingsRecord,
    InstrumentationSettings,
    _require_raw_bridge,
    _validate_head_settings,
    scoped_hook_settings,
)
from .components import (
    AxisSelection,
    ComponentKind,
    ComponentRef,
    NormalizedSelection,
    ResolvedComponent,
    normalize_positions,
    normalize_selection,
    resolve_component,
    select_tensor,
)


class InterventionOperation(str, Enum):
    ZERO = "zero"
    REPLACE = "replace"


class ReplacementSource(str, Enum):
    DIRECT = "direct"
    REFERENCE = "reference"
    MEAN = "mean"
    RESAMPLE = "resample"


@dataclass(frozen=True)
class Intervention:
    """One exact mutation at a named component and sequence positions."""

    component: ComponentRef
    positions: tuple[int, ...] | None
    operation: InterventionOperation
    replacement: torch.Tensor | None = None
    source: ReplacementSource | None = None

    def __post_init__(self) -> None:
        AxisSelection(self.positions)
        if self.operation is InterventionOperation.REPLACE:
            if self.replacement is None or self.source is None:
                raise ValueError(
                    "replacement intervention requires replacement tensor and source"
                )
        elif self.replacement is not None or self.source is not None:
            raise ValueError("zero intervention must not include replacement or source")


@dataclass(frozen=True)
class InterventionPlan:
    interventions: tuple[Intervention, ...]
    settings: InstrumentationSettings = field(default_factory=InstrumentationSettings)


@dataclass(frozen=True)
class InterventionExecution:
    """What one hook mutation actually selected and wrote."""

    component: ComponentRef
    hook_name: str
    normalized_positions: tuple[int, ...]
    operation: InterventionOperation
    source: ReplacementSource | None
    shape: tuple[int, ...]
    dtype: str
    device: str
    replacement_transferred: bool
    before: torch.Tensor
    after: torch.Tensor
    outside_max_abs_change: float


@dataclass(frozen=True)
class InterventionResult:
    logits: torch.Tensor
    executions: tuple[InterventionExecution, ...]
    hook_settings: HookSettingsRecord
    activations: dict[CaptureRequest, CapturedActivation] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedIntervention:
    index: int
    declaration: Intervention
    component: ResolvedComponent
    selection: NormalizedSelection


_CAPTURE_ONLY = {
    ComponentKind.ATTN_PATTERN,
    ComponentKind.FINAL_NORM,
    ComponentKind.LOGITS,
}


def _validate_inputs(inputs: torch.Tensor) -> int:
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
        raise ValueError("Phase 1 interventions require [batch, position] token IDs")
    return int(inputs.shape[1])


def _subcomponent_overlaps(
    left: ResolvedComponent, right: ResolvedComponent
) -> bool:
    if left.head_axis is not None or right.head_axis is not None:
        left_head = left.ref.head
        right_head = right.ref.head
        if left_head is not None and right_head is not None and left_head != right_head:
            return False
    if left.neuron_axis is not None or right.neuron_axis is not None:
        left_neuron = left.ref.neuron
        right_neuron = right.ref.neuron
        if (
            left_neuron is not None
            and right_neuron is not None
            and left_neuron != right_neuron
        ):
            return False
    return True


def _prepare(
    model: Any,
    inputs: torch.Tensor,
    plan: InterventionPlan,
) -> tuple[_PreparedIntervention, ...]:
    sequence_length = _validate_inputs(inputs)
    prepared: list[_PreparedIntervention] = []
    needs_head_result = False
    for index, declaration in enumerate(plan.interventions):
        if declaration.component.kind in _CAPTURE_ONLY:
            raise ValueError(
                f"{declaration.component.kind.value} is capture-only in Phase 1"
            )
        component = resolve_component(declaration.component, model)
        positions = normalize_positions(declaration.positions, sequence_length)
        prepared.append(
            _PreparedIntervention(
                index=index,
                declaration=declaration,
                component=component,
                selection=NormalizedSelection(positions),
            )
        )
        needs_head_result |= declaration.component.kind is ComponentKind.ATTN_HEAD

    if needs_head_result and not plan.settings.use_attn_result:
        raise ValueError(
            "attention-head intervention requires use_attn_result=True explicitly"
        )

    for left_index, left in enumerate(prepared):
        for right in prepared[left_index + 1 :]:
            if left.component.hook_name != right.component.hook_name:
                continue
            if not set(left.selection.positions) & set(right.selection.positions):
                continue
            if _subcomponent_overlaps(left.component, right.component):
                raise ValueError(
                    "interventions overlap after position normalization at "
                    f"{left.component.hook_name}"
                )
    return tuple(prepared)


def _assign_selected(
    destination: torch.Tensor,
    replacement: torch.Tensor,
    component: ResolvedComponent,
    selection: NormalizedSelection,
) -> None:
    for source_position, target_position in enumerate(selection.positions):
        target_index: list[int | slice] = [slice(None)] * destination.ndim
        source_index: list[int | slice] = [slice(None)] * replacement.ndim
        target_index[component.position_axis] = target_position
        source_index[component.position_axis] = source_position
        if component.head_axis is not None and component.ref.head is not None:
            target_index[component.head_axis] = component.ref.head
            source_index[component.head_axis] = 0
        if component.neuron_axis is not None and component.ref.neuron is not None:
            target_index[component.neuron_axis] = component.ref.neuron
            source_index[component.neuron_axis] = 0
        destination[tuple(target_index)] = replacement[tuple(source_index)]


def _outside_max_abs_change(
    before_full: torch.Tensor,
    after_full: torch.Tensor,
    before_selected: torch.Tensor,
    component: ResolvedComponent,
    selection: NormalizedSelection,
) -> float:
    restored = after_full.clone()
    _assign_selected(restored, before_selected, component, selection)
    return float((restored - before_full).abs().max().item())


def run_interventions(
    model: Any,
    inputs: torch.Tensor,
    plan: InterventionPlan,
    *,
    prepend_bos: bool = False,
    captures: CapturePlan | None = None,
) -> InterventionResult:
    """Execute validated interventions during one inference-only forward.

    An optional capture plan records activations from the same forward. A
    capture on a hook that is also intervened observes the post-intervention
    value, because capture hooks are registered after intervention hooks.
    """

    _require_raw_bridge(model)
    prepared = _prepare(model, inputs, plan)
    if captures is not None:
        if captures.settings != plan.settings:
            raise ValueError("capture settings must equal the intervention settings")
        _validate_head_settings(captures.requests, captures.settings)
    grouped: dict[str, list[_PreparedIntervention]] = {}
    for item in prepared:
        grouped.setdefault(item.component.hook_name, []).append(item)

    execution_by_index: dict[int, InterventionExecution] = {}
    hooks: list[tuple[str, Any]] = []
    for hook_name, items in grouped.items():

        def intervention_hook(
            activation: torch.Tensor,
            hook: Any,
            *,
            items: tuple[_PreparedIntervention, ...] = tuple(items),
        ) -> torch.Tensor:
            del hook
            updated = activation
            for item in items:
                before_full = updated
                before = select_tensor(
                    before_full, item.component, item.selection
                ).detach().clone()
                transferred = False
                if item.declaration.operation is InterventionOperation.ZERO:
                    replacement = torch.zeros_like(before)
                else:
                    assert item.declaration.replacement is not None
                    replacement = item.declaration.replacement
                    if replacement.shape != before.shape:
                        raise ValueError(
                            "replacement shape mismatch: "
                            f"expected {tuple(before.shape)}, received "
                            f"{tuple(replacement.shape)}"
                        )
                    if replacement.dtype != before.dtype:
                        raise ValueError(
                            "replacement dtype mismatch: "
                            f"expected {before.dtype}, received {replacement.dtype}"
                        )
                    if replacement.device != before.device:
                        replacement = replacement.to(before.device)
                        transferred = True

                updated = before_full.clone()
                _assign_selected(
                    updated, replacement, item.component, item.selection
                )
                after = select_tensor(
                    updated, item.component, item.selection
                ).detach().clone()
                outside_change = _outside_max_abs_change(
                    before_full,
                    updated,
                    before,
                    item.component,
                    item.selection,
                )
                execution_by_index[item.index] = InterventionExecution(
                    component=item.declaration.component,
                    hook_name=item.component.hook_name,
                    normalized_positions=item.selection.positions,
                    operation=item.declaration.operation,
                    source=item.declaration.source,
                    shape=tuple(after.shape),
                    dtype=str(after.dtype),
                    device=str(after.device),
                    replacement_transferred=transferred,
                    before=before.cpu(),
                    after=after.cpu(),
                    outside_max_abs_change=outside_change,
                )
            return updated

        hooks.append((hook_name, intervention_hook))

    captured: dict[CaptureRequest, CapturedActivation] = {}
    if captures is not None:
        for request in captures.requests:
            component = resolve_component(request.component, model)

            def capture_hook(
                activation: torch.Tensor,
                hook: Any,
                *,
                request: CaptureRequest = request,
                component: ResolvedComponent = component,
            ) -> torch.Tensor:
                del hook
                selection = normalize_selection(request.selection, component, activation.shape)
                selected = select_tensor(activation, component, selection).detach().clone()
                if captures.storage_device == "cpu":
                    selected = selected.cpu()
                captured[request] = CapturedActivation(
                    request=request,
                    hook_name=component.hook_name,
                    axis_names=component.axis_names,
                    original_shape=tuple(activation.shape),
                    tensor=selected,
                )
                return activation

            hooks.append((component.hook_name, capture_hook))

    model.eval()
    with scoped_hook_settings(model, plan.settings) as hook_record:
        with torch.inference_mode():
            if hooks:
                logits = model.run_with_hooks(
                    inputs,
                    fwd_hooks=hooks,
                    prepend_bos=prepend_bos,
                )
            else:
                logits = model(inputs, prepend_bos=prepend_bos)

    missing = [item.index for item in prepared if item.index not in execution_by_index]
    if missing:
        raise RuntimeError(f"intervention hooks did not fire for indices {missing}")
    if captures is not None:
        missing_captures = [request for request in captures.requests if request not in captured]
        if missing_captures:
            raise RuntimeError(f"requested capture hooks did not fire: {missing_captures!r}")
    executions = tuple(execution_by_index[index] for index in range(len(prepared)))
    return InterventionResult(logits, executions, hook_record, captured)
