from __future__ import annotations

import os

import pytest
import torch

from neural_decompiler.capture import (
    CapturePlan,
    CaptureRequest,
    InstrumentationSettings,
    run_capture,
)
from neural_decompiler.components import (
    ComponentKind,
    ComponentRef,
    decompose_attention,
    decompose_pythia_parallel_block,
)
from neural_decompiler.interventions import (
    Intervention,
    InterventionOperation,
    InterventionPlan,
    ReplacementSource,
    run_interventions,
)
from neural_decompiler.models import PYTHIA_70M, load_model


pytestmark = pytest.mark.pythia_smoke
ATOL = 1e-5
RTOL = 1e-5


@pytest.fixture(scope="module")
def pythia_bridge():
    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")
    return load_model(PYTHIA_70M)


def test_pinned_pythia_bridge_contract(pythia_bridge) -> None:
    assert pythia_bridge.cfg.n_layers == 6
    assert pythia_bridge.cfg.n_heads == 8
    assert pythia_bridge.cfg.parallel_attn_mlp is True
    tokens = pythia_bridge.to_tokens(
        "The quick brown fox", prepend_bos=False, truncate=False
    )
    with torch.inference_mode():
        direct = pythia_bridge(tokens, prepend_bos=False)

    requests = (
        CaptureRequest(ComponentRef(ComponentKind.TOKEN_EMBED), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.RESID_PRE, layer=0), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.ATTN_OUT, layer=0), (0,)),
        *(
            CaptureRequest(
                ComponentRef(ComponentKind.ATTN_HEAD, layer=0, head=head), (0,)
            )
            for head in range(pythia_bridge.cfg.n_heads)
        ),
        CaptureRequest(
            ComponentRef(ComponentKind.ATTN_PATTERN, layer=0, head=0),
            (0,),
            (0,),
        ),
        CaptureRequest(ComponentRef(ComponentKind.MLP_POST, layer=0), (0,)),
        CaptureRequest(
            ComponentRef(ComponentKind.MLP_NEURON, layer=0, neuron=0), (0,)
        ),
        CaptureRequest(ComponentRef(ComponentKind.MLP_OUT, layer=0), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.RESID_POST, layer=0), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.FINAL_NORM), (0,)),
        CaptureRequest(ComponentRef(ComponentKind.LOGITS), (0,)),
    )
    result = run_capture(
        pythia_bridge,
        tokens,
        CapturePlan(
            requests,
            InstrumentationSettings(use_attn_result=True),
        ),
        prepend_bos=False,
    )

    assert torch.allclose(result.logits, direct, atol=ATOL, rtol=RTOL)
    assert torch.equal(result.logits.argmax(dim=-1), direct.argmax(dim=-1))
    assert result.activations[requests[0]].tensor.shape == (1, 1, 512)
    assert result.activations[requests[1]].tensor.shape == (1, 1, 512)
    assert result.activations[requests[3]].tensor.shape == (1, 1, 1, 512)
    assert result.activations[requests[11]].tensor.shape == (1, 1, 1, 1)
    assert result.activations[requests[12]].tensor.shape == (1, 1, 2048)
    assert result.activations[requests[13]].tensor.shape == (1, 1, 1)
    assert pythia_bridge.cfg.use_attn_result is False

    captured_heads = torch.cat(
        tuple(result.activations[request].tensor for request in requests[3:11]),
        dim=2,
    )
    decompose_attention(
        captured_heads,
        pythia_bridge.blocks[0].attn.b_O.detach().cpu(),
        result.activations[requests[2]].tensor,
        atol=ATOL,
        rtol=RTOL,
    )
    decompose_pythia_parallel_block(
        result.activations[requests[1]].tensor,
        result.activations[requests[2]].tensor,
        result.activations[requests[14]].tensor,
        result.activations[requests[15]].tensor,
        architecture=pythia_bridge.original_model.config.architectures[0],
        parallel_attn_mlp=pythia_bridge.cfg.parallel_attn_mlp,
        atol=ATOL,
        rtol=RTOL,
    )

    noop = run_interventions(
        pythia_bridge,
        tokens,
        InterventionPlan(
            (
                Intervention(
                    ComponentRef(ComponentKind.MLP_POST, layer=0),
                    (0,),
                    InterventionOperation.REPLACE,
                    result.activations[requests[12]].tensor,
                    ReplacementSource.DIRECT,
                ),
            )
        ),
        prepend_bos=False,
    )
    assert torch.allclose(noop.logits, direct, atol=ATOL, rtol=RTOL)
    assert torch.equal(noop.logits.argmax(dim=-1), direct.argmax(dim=-1))
