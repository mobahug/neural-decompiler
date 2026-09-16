from __future__ import annotations

import json

import pytest
import torch

from neural_decompiler.provenance import (
    BridgeProvenance,
    InputProvenance,
    RunProvenance,
    RuntimeProvenance,
    TokenizerProvenance,
    canonical_json,
    collect_git_state,
    collect_versions,
    tensor_digest,
    write_json,
)


def complete_run_provenance() -> RunProvenance:
    return RunProvenance(
        model_id="EleutherAI/pythia-70m-deduped",
        requested_revision="e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
        resolved_revision="e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
        tokenizer=TokenizerProvenance(
            tokenizer_id="EleutherAI/pythia-70m-deduped",
            revision="e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
            bos_token_id=0,
            eos_token_id=0,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            prepend_bos=False,
            add_special_tokens=False,
            padding_side="right",
            truncation=False,
        ),
        runtime=RuntimeProvenance(
            device="cpu",
            backend="cpu",
            dtype="float32",
            seed=17,
            deterministic_algorithms=True,
            atol=1e-5,
            rtol=1e-5,
            versions={"torch": "2.8.0", "transformer-lens": "3.9.0"},
        ),
        bridge=BridgeProvenance(
            compatibility_mode=False,
            use_attn_result=True,
        ),
        input=InputProvenance(
            text="A B",
            token_ids=(32, 347),
            token_strings=("A", " B"),
            target_positions=(1,),
        ),
        git_commit="abc123",
        git_dirty=False,
        capture_definitions=({"kind": "resid_pre", "layer": 0},),
        intervention_definitions=(),
        raw_measurements={"max_abs_error": 0.0},
    )


def test_canonical_provenance_is_stable_and_complete(tmp_path) -> None:
    run = complete_run_provenance()

    first = canonical_json(run.to_dict())
    second = canonical_json(run.to_dict())

    assert first == second
    decoded = json.loads(first)
    assert decoded["tokenizer"]["prepend_bos"] is False
    assert decoded["bridge"]["use_attn_result"] is True
    assert decoded["input"]["token_ids"] == [32, 347]
    path = tmp_path / "run.json"
    write_json(path, run.to_dict())
    assert path.read_bytes().endswith(b"\n")
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_tensor_digest_includes_shape_dtype_and_bytes() -> None:
    float32 = tensor_digest(torch.zeros(2, dtype=torch.float32))
    float64 = tensor_digest(torch.zeros(2, dtype=torch.float64))
    different_shape = tensor_digest(torch.zeros(1, 2, dtype=torch.float32))
    assert len(float32) == 64
    assert len({float32, float64, different_shape}) == 3


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), torch.tensor(1), {1: "x"}, {"x"}],
)
def test_non_json_artifacts_fail_loudly(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_json({"bad": bad})


def test_runtime_tolerances_must_be_finite_and_nonnegative() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        RuntimeProvenance(
            "cpu", "cpu", "float32", 1, False, -1.0, 1e-5, {}
        )


def test_version_and_git_collectors_return_serializable_mappings() -> None:
    versions = collect_versions()
    git_state = collect_git_state()
    assert "torch" in versions
    assert set(git_state) == {"commit", "dirty"}
    canonical_json({"versions": versions, "git": git_state})
