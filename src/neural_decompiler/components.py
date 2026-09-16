"""Canonical TransformerBridge components, selectors, and decompositions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class ComponentKind(str, Enum):
    """Closed component vocabulary supported by Phase 1."""

    TOKEN_EMBED = "token_embed"
    POSITION_EMBED = "position_embed"
    RESID_PRE = "resid_pre"
    ATTN_OUT = "attn_out"
    ATTN_HEAD = "attn_head"
    ATTN_PATTERN = "attn_pattern"
    MLP_POST = "mlp_post"
    MLP_NEURON = "mlp_neuron"
    MLP_OUT = "mlp_out"
    RESID_POST = "resid_post"
    FINAL_NORM = "final_norm"
    LOGITS = "logits"


_GLOBAL_KINDS = {
    ComponentKind.TOKEN_EMBED,
    ComponentKind.POSITION_EMBED,
    ComponentKind.FINAL_NORM,
    ComponentKind.LOGITS,
}
_LAYER_KINDS = set(ComponentKind) - _GLOBAL_KINDS


@dataclass(frozen=True)
class ComponentRef:
    """One scientific component independent of an input position."""

    kind: ComponentKind
    layer: int | None = None
    head: int | None = None
    neuron: int | None = None

    def __post_init__(self) -> None:
        if self.kind in _GLOBAL_KINDS:
            if self.layer is not None:
                raise ValueError(f"{self.kind.value} does not accept layer")
        elif self.layer is None:
            raise ValueError(f"{self.kind.value} requires layer")
        elif self.layer < 0:
            raise ValueError("layer must be nonnegative")

        if self.kind is ComponentKind.ATTN_HEAD:
            if self.head is None:
                raise ValueError("attn_head requires head")
        elif self.kind is not ComponentKind.ATTN_PATTERN and self.head is not None:
            raise ValueError(f"{self.kind.value} does not accept head")
        if self.head is not None and self.head < 0:
            raise ValueError("head must be nonnegative")

        if self.kind is ComponentKind.MLP_NEURON:
            if self.neuron is None:
                raise ValueError("mlp_neuron requires neuron")
        elif self.neuron is not None:
            raise ValueError(f"{self.kind.value} does not accept neuron")
        if self.neuron is not None and self.neuron < 0:
            raise ValueError("neuron must be nonnegative")


@dataclass(frozen=True)
class ResolvedComponent:
    """Canonical hook plus the axes observed at that hook."""

    ref: ComponentRef
    hook_name: str
    axis_names: tuple[str, ...]
    position_axis: int
    key_position_axis: int | None = None
    head_axis: int | None = None
    neuron_axis: int | None = None


@dataclass(frozen=True)
class AxisSelection:
    """Input-dependent sequence axes selected at a component site."""

    positions: tuple[int, ...] | None = None
    key_positions: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.positions == ():
            raise ValueError("positions must not be empty")
        if self.key_positions == ():
            raise ValueError("key_positions must not be empty")


@dataclass(frozen=True)
class NormalizedSelection:
    """Selection with every negative position resolved to an absolute index."""

    positions: tuple[int, ...]
    key_positions: tuple[int, ...] | None = None


def _hook_names(model: Any) -> set[str]:
    for name in ("hook_dict", "_hook_registry"):
        registry = getattr(model, name, None)
        if registry is not None:
            return set(registry)
    raise ValueError("model does not expose a TransformerBridge hook registry")


def _layer_hook(ref: ComponentRef, suffix: str) -> str:
    assert ref.layer is not None
    return f"blocks.{ref.layer}.{suffix}"


def resolve_component(ref: ComponentRef, model: Any) -> ResolvedComponent:
    """Resolve one declaration to a canonical raw-Bridge hook."""

    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("model does not expose cfg geometry")

    if ref.layer is not None:
        n_layers = int(getattr(cfg, "n_layers", -1))
        if ref.layer >= n_layers:
            raise ValueError(
                f"layer {ref.layer} out of range for {n_layers} layers"
            )
    if ref.head is not None:
        n_heads = int(getattr(cfg, "n_heads", -1))
        if ref.head >= n_heads:
            raise ValueError(f"head {ref.head} out of range for {n_heads} heads")
    if ref.neuron is not None:
        d_mlp = int(getattr(cfg, "d_mlp", -1))
        if ref.neuron >= d_mlp:
            raise ValueError(
                f"neuron {ref.neuron} out of range for MLP width {d_mlp}"
            )

    standard_axes = ("batch", "position", "d_model")
    if ref.kind is ComponentKind.TOKEN_EMBED:
        resolved = ResolvedComponent(ref, "embed.hook_out", standard_axes, 1)
    elif ref.kind is ComponentKind.POSITION_EMBED:
        positional_type = str(
            getattr(cfg, "positional_embedding_type", "unknown")
        ).lower()
        if positional_type == "rotary":
            raise ValueError("model has no additive positional embedding component")
        resolved = ResolvedComponent(ref, "pos_embed.hook_out", standard_axes, 1)
    elif ref.kind is ComponentKind.RESID_PRE:
        resolved = ResolvedComponent(
            ref, _layer_hook(ref, "hook_in"), standard_axes, 1
        )
    elif ref.kind is ComponentKind.ATTN_OUT:
        resolved = ResolvedComponent(
            ref, _layer_hook(ref, "attn.hook_out"), standard_axes, 1
        )
    elif ref.kind is ComponentKind.ATTN_HEAD:
        resolved = ResolvedComponent(
            ref,
            _layer_hook(ref, "attn.hook_result"),
            ("batch", "position", "head", "d_model"),
            1,
            head_axis=2,
        )
    elif ref.kind is ComponentKind.ATTN_PATTERN:
        resolved = ResolvedComponent(
            ref,
            _layer_hook(ref, "attn.hook_pattern"),
            ("batch", "head", "query_position", "key_position"),
            2,
            key_position_axis=3,
            head_axis=1,
        )
    elif ref.kind is ComponentKind.MLP_POST:
        resolved = ResolvedComponent(
            ref,
            _layer_hook(ref, "mlp.out.hook_in"),
            ("batch", "position", "neuron"),
            1,
            neuron_axis=2,
        )
    elif ref.kind is ComponentKind.MLP_NEURON:
        resolved = ResolvedComponent(
            ref,
            _layer_hook(ref, "mlp.out.hook_in"),
            ("batch", "position", "neuron"),
            1,
            neuron_axis=2,
        )
    elif ref.kind is ComponentKind.MLP_OUT:
        resolved = ResolvedComponent(
            ref, _layer_hook(ref, "mlp.hook_out"), standard_axes, 1
        )
    elif ref.kind is ComponentKind.RESID_POST:
        resolved = ResolvedComponent(
            ref, _layer_hook(ref, "hook_out"), standard_axes, 1
        )
    elif ref.kind is ComponentKind.FINAL_NORM:
        resolved = ResolvedComponent(ref, "ln_final.hook_out", standard_axes, 1)
    elif ref.kind is ComponentKind.LOGITS:
        resolved = ResolvedComponent(
            ref, "unembed.hook_out", ("batch", "position", "vocab"), 1
        )
    else:  # pragma: no cover - Enum exhaustiveness guard
        raise ValueError(f"Unsupported component kind: {ref.kind}")

    if resolved.hook_name not in _hook_names(model):
        raise ValueError(
            f"Canonical TransformerBridge hook {resolved.hook_name!r} is unavailable"
        )
    return resolved


def normalize_positions(
    positions: tuple[int, ...] | None, sequence_length: int
) -> tuple[int, ...]:
    """Resolve Python-style negative indices and reject ambiguity."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if positions is None:
        return tuple(range(sequence_length))
    if not positions:
        raise ValueError("positions must not be empty")

    normalized: list[int] = []
    for position in positions:
        absolute = position + sequence_length if position < 0 else position
        if absolute < 0 or absolute >= sequence_length:
            raise ValueError(
                f"position {position} out of range for sequence length {sequence_length}"
            )
        normalized.append(absolute)
    if len(normalized) != len(set(normalized)):
        raise ValueError("positions contain duplicate normalized indices")
    return tuple(normalized)


def normalize_selection(
    selection: AxisSelection,
    component: ResolvedComponent,
    tensor_shape: torch.Size | tuple[int, ...],
) -> NormalizedSelection:
    """Normalize input-dependent axes against an observed activation shape."""

    if len(tensor_shape) != len(component.axis_names):
        raise ValueError(
            f"Hook {component.hook_name} returned rank {len(tensor_shape)}; "
            f"expected axes {component.axis_names}"
        )
    positions = normalize_positions(
        selection.positions, int(tensor_shape[component.position_axis])
    )
    if component.key_position_axis is None:
        if selection.key_positions is not None:
            raise ValueError("key_positions are valid only for attention patterns")
        keys = None
    else:
        keys = normalize_positions(
            selection.key_positions,
            int(tensor_shape[component.key_position_axis]),
        )
    return NormalizedSelection(positions=positions, key_positions=keys)


def select_tensor(
    tensor: torch.Tensor,
    component: ResolvedComponent,
    selection: NormalizedSelection,
) -> torch.Tensor:
    """Select requested axes while preserving every tensor dimension."""

    selected = tensor.index_select(
        component.position_axis,
        torch.tensor(selection.positions, device=tensor.device),
    )
    if component.key_position_axis is not None:
        assert selection.key_positions is not None
        selected = selected.index_select(
            component.key_position_axis,
            torch.tensor(selection.key_positions, device=tensor.device),
        )
    if component.head_axis is not None and component.ref.head is not None:
        selected = selected.index_select(
            component.head_axis,
            torch.tensor((component.ref.head,), device=tensor.device),
        )
    if component.neuron_axis is not None and component.ref.neuron is not None:
        selected = selected.index_select(
            component.neuron_axis,
            torch.tensor((component.ref.neuron,), device=tensor.device),
        )
    return selected


def selections_overlap(
    left: AxisSelection,
    right: AxisSelection,
    sequence_length: int,
) -> bool:
    """Compare positional selections after negative-index normalization."""

    left_positions = set(normalize_positions(left.positions, sequence_length))
    right_positions = set(normalize_positions(right.positions, sequence_length))
    return bool(left_positions & right_positions)


def _require_finite(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("decomposition inputs must be tensors")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("decomposition inputs and measurements must be finite")


def _require_close(
    reconstructed: torch.Tensor,
    measured: torch.Tensor,
    *,
    name: str,
    atol: float,
    rtol: float,
) -> float:
    if reconstructed.shape != measured.shape:
        raise ValueError(
            f"{name} shape mismatch: reconstructed {tuple(reconstructed.shape)}, "
            f"measured {tuple(measured.shape)}"
        )
    error = float((reconstructed - measured).abs().max().item())
    if not torch.allclose(reconstructed, measured, atol=atol, rtol=rtol):
        raise ValueError(
            f"{name} does not reconstruct measured output within "
            f"atol={atol}, rtol={rtol}; max_abs_error={error}"
        )
    return error


@dataclass(frozen=True)
class AttentionDecomposition:
    per_head: torch.Tensor
    head_sum: torch.Tensor
    output_bias: torch.Tensor
    reconstructed: torch.Tensor
    measured: torch.Tensor
    max_abs_error: float


def decompose_attention(
    per_head: torch.Tensor,
    output_bias: torch.Tensor,
    measured: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> AttentionDecomposition:
    """Verify that per-head contributions plus ``b_O`` equal attention output."""

    if output_bias is None:
        raise TypeError("output_bias must be supplied explicitly")
    _require_finite(per_head, output_bias, measured)
    if per_head.ndim < 3:
        raise ValueError("per_head must include head and d_model axes")
    if output_bias.ndim != 1 or output_bias.shape[0] != per_head.shape[-1]:
        raise ValueError("output_bias must have shape [d_model]")
    head_sum = per_head.sum(dim=-2)
    reconstructed = head_sum + output_bias
    error = _require_close(
        reconstructed,
        measured,
        name="attention decomposition",
        atol=atol,
        rtol=rtol,
    )
    return AttentionDecomposition(
        per_head=per_head,
        head_sum=head_sum,
        output_bias=output_bias,
        reconstructed=reconstructed,
        measured=measured,
        max_abs_error=error,
    )


@dataclass(frozen=True)
class BlockDecomposition:
    resid_pre: torch.Tensor
    attention_output: torch.Tensor
    mlp_output: torch.Tensor
    reconstructed: torch.Tensor
    measured: torch.Tensor
    architecture: str
    max_abs_error: float


def decompose_pythia_parallel_block(
    resid_pre: torch.Tensor,
    attention_output: torch.Tensor,
    mlp_output: torch.Tensor,
    measured: torch.Tensor,
    *,
    architecture: str,
    parallel_attn_mlp: bool,
    atol: float,
    rtol: float,
) -> BlockDecomposition:
    """Verify the declared GPT-NeoX parallel-residual block identity."""

    if architecture != "GPTNeoXForCausalLM" or not parallel_attn_mlp:
        raise ValueError(
            "Block decomposition currently supports only parallel GPT-NeoX"
        )
    _require_finite(resid_pre, attention_output, mlp_output, measured)
    if not (
        resid_pre.shape
        == attention_output.shape
        == mlp_output.shape
        == measured.shape
    ):
        raise ValueError("parallel block decomposition inputs must have equal shapes")
    reconstructed = resid_pre + attention_output + mlp_output
    error = _require_close(
        reconstructed,
        measured,
        name="parallel block decomposition",
        atol=atol,
        rtol=rtol,
    )
    return BlockDecomposition(
        resid_pre=resid_pre,
        attention_output=attention_output,
        mlp_output=mlp_output,
        reconstructed=reconstructed,
        measured=measured,
        architecture=architecture,
        max_abs_error=error,
    )
