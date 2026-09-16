from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neural_decompiler.components import (
    AxisSelection,
    ComponentKind,
    ComponentRef,
    decompose_attention,
    decompose_pythia_parallel_block,
    normalize_positions,
    normalize_selection,
    resolve_component,
    select_tensor,
    selections_overlap,
)


def fake_geometry(
    *,
    n_layers: int = 6,
    n_heads: int = 8,
    d_mlp: int = 2048,
    positional_embedding_type: str = "rotary",
) -> SimpleNamespace:
    hooks = {
        "embed.hook_out",
        "ln_final.hook_out",
        "unembed.hook_out",
    }
    for layer in range(n_layers):
        hooks.update(
            {
                f"blocks.{layer}.hook_in",
                f"blocks.{layer}.attn.hook_out",
                f"blocks.{layer}.attn.hook_result",
                f"blocks.{layer}.attn.hook_pattern",
                f"blocks.{layer}.mlp.out.hook_in",
                f"blocks.{layer}.mlp.hook_out",
                f"blocks.{layer}.hook_out",
            }
        )
    return SimpleNamespace(
        cfg=SimpleNamespace(
            n_layers=n_layers,
            n_heads=n_heads,
            d_mlp=d_mlp,
            positional_embedding_type=positional_embedding_type,
        ),
        hook_dict={name: object() for name in hooks},
    )


def test_component_resolution_uses_canonical_bridge_hooks() -> None:
    model = fake_geometry()

    assert (
        resolve_component(ComponentRef(ComponentKind.RESID_PRE, layer=2), model).hook_name
        == "blocks.2.hook_in"
    )
    assert (
        resolve_component(
            ComponentRef(ComponentKind.ATTN_HEAD, layer=2, head=3), model
        ).hook_name
        == "blocks.2.attn.hook_result"
    )
    assert (
        resolve_component(
            ComponentRef(ComponentKind.MLP_NEURON, layer=2, neuron=17), model
        ).hook_name
        == "blocks.2.mlp.out.hook_in"
    )


def test_component_resolution_rejects_missing_rotary_position_embedding() -> None:
    with pytest.raises(ValueError, match="positional embedding"):
        resolve_component(
            ComponentRef(ComponentKind.POSITION_EMBED), fake_geometry()
        )


@pytest.mark.parametrize(
    "component",
    [
        ComponentRef(ComponentKind.RESID_PRE, layer=6),
        ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=8),
        ComponentRef(ComponentKind.MLP_NEURON, layer=0, neuron=2048),
    ],
)
def test_component_resolution_rejects_out_of_range_indices(
    component: ComponentRef,
) -> None:
    with pytest.raises(ValueError, match="out of range"):
        resolve_component(component, fake_geometry())


def test_component_ref_rejects_inappropriate_fields() -> None:
    with pytest.raises(ValueError, match="does not accept layer"):
        ComponentRef(ComponentKind.TOKEN_EMBED, layer=0)
    with pytest.raises(ValueError, match="requires head"):
        ComponentRef(ComponentKind.ATTN_HEAD, layer=0)


def test_negative_positions_normalize_before_overlap() -> None:
    assert normalize_positions((-1,), 5) == (4,)
    left = AxisSelection(positions=(-1,))
    right = AxisSelection(positions=(4,))
    assert selections_overlap(left, right, sequence_length=5)


def test_position_normalization_rejects_duplicates_and_range_errors() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        normalize_positions((0, -3), 3)
    with pytest.raises(ValueError, match="out of range"):
        normalize_positions((3,), 3)


def test_select_tensor_preserves_selected_axes_as_singletons() -> None:
    tensor = torch.arange(2 * 4 * 3 * 5).reshape(2, 4, 3, 5)
    resolved = resolve_component(
        ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=1), fake_geometry()
    )
    selection = normalize_selection(
        AxisSelection(positions=(-1,)), resolved, tensor.shape
    )

    selected = select_tensor(tensor, resolved, selection)

    assert selected.shape == (2, 1, 1, 5)
    assert torch.equal(selected, tensor[:, 3:4, 1:2, :])


def test_attention_pattern_selection_uses_query_and_key_axes() -> None:
    tensor = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    resolved = resolve_component(
        ComponentRef(ComponentKind.ATTN_PATTERN, layer=0, head=2), fake_geometry()
    )
    selection = normalize_selection(
        AxisSelection(positions=(1,), key_positions=(-1,)), resolved, tensor.shape
    )

    selected = select_tensor(tensor, resolved, selection)

    assert selected.shape == (2, 1, 1, 1)
    assert torch.equal(selected, tensor[:, 2:3, 1:2, 3:4])


def test_attention_decomposition_includes_output_bias() -> None:
    heads = torch.arange(24.0).reshape(1, 2, 3, 4)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
    measured = heads.sum(dim=2) + bias

    result = decompose_attention(heads, bias, measured, atol=0.0, rtol=0.0)

    assert torch.equal(result.head_sum, heads.sum(dim=2))
    assert torch.equal(result.output_bias, bias)
    assert torch.equal(result.reconstructed, measured)
    assert result.max_abs_error == 0.0


def test_attention_decomposition_rejects_bias_omission_and_mismatch() -> None:
    heads = torch.zeros(1, 2, 3, 4)
    measured = torch.ones(1, 2, 4)
    with pytest.raises(TypeError, match="output_bias"):
        decompose_attention(heads, None, measured, atol=0.0, rtol=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not reconstruct"):
        decompose_attention(
            heads, torch.zeros(4), measured, atol=0.0, rtol=0.0
        )


def test_parallel_block_decomposition_reconstructs_measured_output() -> None:
    resid = torch.arange(8.0).reshape(1, 2, 4)
    attn = torch.ones_like(resid)
    mlp = torch.full_like(resid, 2.0)
    measured = resid + attn + mlp

    result = decompose_pythia_parallel_block(
        resid,
        attn,
        mlp,
        measured,
        architecture="GPTNeoXForCausalLM",
        parallel_attn_mlp=True,
        atol=0.0,
        rtol=0.0,
    )

    assert torch.equal(result.reconstructed, measured)
    assert result.max_abs_error == 0.0


def test_block_decomposition_rejects_non_parallel_architecture() -> None:
    tensor = torch.zeros(1, 2, 4)
    with pytest.raises(ValueError, match="parallel GPT-NeoX"):
        decompose_pythia_parallel_block(
            tensor,
            tensor,
            tensor,
            tensor,
            architecture="GPTNeoXForCausalLM",
            parallel_attn_mlp=False,
            atol=0.0,
            rtol=0.0,
        )
    with pytest.raises(ValueError, match="parallel GPT-NeoX"):
        decompose_pythia_parallel_block(
            tensor,
            tensor,
            tensor,
            tensor,
            architecture="LlamaForCausalLM",
            parallel_attn_mlp=True,
            atol=0.0,
            rtol=0.0,
        )


def test_decomposition_rejects_non_finite_inputs() -> None:
    heads = torch.zeros(1, 1, 1, 2)
    heads[..., 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        decompose_attention(
            heads,
            torch.zeros(2),
            torch.zeros(1, 1, 2),
            atol=1e-5,
            rtol=1e-5,
        )
