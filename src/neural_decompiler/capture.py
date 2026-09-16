"""Selective, inference-only capture at canonical TransformerBridge hooks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from .components import (
    AxisSelection,
    ComponentKind,
    ComponentRef,
    ResolvedComponent,
    normalize_selection,
    resolve_component,
    select_tensor,
)


@dataclass(frozen=True)
class InstrumentationSettings:
    """Optional Bridge execution structure requested for one run."""

    use_attn_result: bool = False


@dataclass(frozen=True)
class HookSettingsRecord:
    """Requested and effective optional-hook state during a forward."""

    requested_use_attn_result: bool
    effective_use_attn_result: bool
    compatibility_mode: bool


@dataclass(frozen=True)
class CaptureRequest:
    """One component and the input-dependent axes to retain."""

    component: ComponentRef
    positions: tuple[int, ...] | None = None
    key_positions: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (
            self.key_positions is not None
            and self.component.kind is not ComponentKind.ATTN_PATTERN
        ):
            raise ValueError("key_positions are valid only for attention patterns")
        AxisSelection(self.positions, self.key_positions)

    @property
    def selection(self) -> AxisSelection:
        return AxisSelection(self.positions, self.key_positions)


@dataclass(frozen=True)
class CapturePlan:
    """Validated selective capture declaration."""

    requests: tuple[CaptureRequest, ...]
    settings: InstrumentationSettings = field(default_factory=InstrumentationSettings)
    storage_device: str = "cpu"

    def __post_init__(self) -> None:
        if len(self.requests) != len(set(self.requests)):
            raise ValueError("capture plan contains duplicate requests")
        if self.storage_device not in {"cpu", "execution"}:
            raise ValueError("storage_device must be 'cpu' or 'execution'")


@dataclass(frozen=True)
class CapturedActivation:
    """Detached selected tensor plus the live hook's declared shape metadata."""

    request: CaptureRequest
    hook_name: str
    axis_names: tuple[str, ...]
    original_shape: tuple[int, ...]
    tensor: torch.Tensor


@dataclass(frozen=True)
class CaptureResult:
    """Logits and exactly the requested activations from one forward."""

    logits: torch.Tensor
    activations: dict[CaptureRequest, CapturedActivation]
    hook_settings: HookSettingsRecord


def _require_raw_bridge(model: Any) -> None:
    if bool(getattr(model, "compatibility_mode", False)):
        raise ValueError("TransformerBridge compatibility mode must remain disabled")


@contextmanager
def scoped_hook_settings(
    model: Any, settings: InstrumentationSettings
) -> Iterator[HookSettingsRecord]:
    """Apply optional Bridge settings through setters and restore them safely."""

    cfg = getattr(model, "cfg", None)
    if cfg is None or not hasattr(cfg, "use_attn_result"):
        if settings.use_attn_result:
            raise ValueError("model does not expose use_attn_result")
        effective = False
        yield HookSettingsRecord(False, effective, False)
        return

    prior = bool(cfg.use_attn_result)
    target = True if settings.use_attn_result else prior
    changed = target != prior
    if changed:
        setter = getattr(model, "set_use_attn_result", None)
        if setter is None:
            raise ValueError("model does not expose set_use_attn_result")
        setter(target)
    record = HookSettingsRecord(
        requested_use_attn_result=settings.use_attn_result,
        effective_use_attn_result=bool(cfg.use_attn_result),
        compatibility_mode=bool(getattr(model, "compatibility_mode", False)),
    )
    try:
        yield record
    finally:
        if changed:
            model.set_use_attn_result(prior)


def _validate_head_settings(
    requests: tuple[CaptureRequest, ...], settings: InstrumentationSettings
) -> None:
    needs_results = any(
        request.component.kind is ComponentKind.ATTN_HEAD for request in requests
    )
    if needs_results and not settings.use_attn_result:
        raise ValueError(
            "attention-head capture requires use_attn_result=True explicitly"
        )


def run_capture(
    model: Any,
    inputs: torch.Tensor,
    plan: CapturePlan,
    *,
    prepend_bos: bool = False,
) -> CaptureResult:
    """Run one selective capture forward without retaining a full cache."""

    _require_raw_bridge(model)
    _validate_head_settings(plan.requests, plan.settings)
    resolved: list[tuple[CaptureRequest, ResolvedComponent]] = [
        (request, resolve_component(request.component, model))
        for request in plan.requests
    ]
    captures: dict[CaptureRequest, CapturedActivation] = {}
    hooks: list[tuple[str, Any]] = []

    for request, component in resolved:

        def capture_hook(
            activation: torch.Tensor,
            hook: Any,
            *,
            request: CaptureRequest = request,
            component: ResolvedComponent = component,
        ) -> torch.Tensor:
            del hook
            selection = normalize_selection(
                request.selection, component, activation.shape
            )
            selected = select_tensor(activation, component, selection).detach().clone()
            if plan.storage_device == "cpu":
                selected = selected.cpu()
            captures[request] = CapturedActivation(
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

    missing = [request for request in plan.requests if request not in captures]
    if missing:
        raise RuntimeError(f"requested capture hooks did not fire: {missing!r}")
    return CaptureResult(
        logits=logits,
        activations=captures,
        hook_settings=hook_record,
    )
