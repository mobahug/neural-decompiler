"""The bound parameters digest is over the pinned checkpoint's values: the bridge's 113 parameters against the raw
safetensors (94 tensors), with the fused query_key_value split per head into q, k, v. Same guards as c_frozen.py."""
from common import install_fs_guard

install_fs_guard()

import json  # noqa: E402

from common import EVENTS, forbidden_101_sequences, install_model_guards, seal_model, write_json  # noqa: E402

SEQ = sorted(set(forbidden_101_sequences().values()))
import torch  # noqa: E402

install_model_guards(SEQ)
import numpy as np  # noqa: E402

import mine  # noqa: E402
from neural_decompiler import models  # noqa: E402

EVENTS["model_loads"] += 1
model = models.load_model(models.PYTHIA_70M)
seal_model(model)
bridge = {name: p.detach().cpu().contiguous().to(torch.float32).numpy() for name, p in model.named_parameters()}
del model
raw = mine.raw_weights()
raw_bytes = {}
for name, arr in raw.items():
    raw_bytes.setdefault((tuple(arr.shape), np.ascontiguousarray(arr, dtype="<f4").tobytes()), []).append(name)
derived = {}
for layer in range(6):
    w = raw[f"gpt_neox.layers.{layer}.attention.query_key_value.weight"].reshape(8, 3, 64, 512)
    b = raw[f"gpt_neox.layers.{layer}.attention.query_key_value.bias"].reshape(8, 3, 64)
    for index, part in enumerate("qkv"):
        derived[(layer, part, "weight")] = w[:, index].reshape(512, 512)
        derived[(layer, part, "bias")] = b[:, index].reshape(512)
result = {"whole_raw_tensor": [], "qkv_slice_of_raw": [], "unmatched": []}
for name, arr in bridge.items():
    key = (tuple(arr.shape), np.ascontiguousarray(arr, dtype="<f4").tobytes())
    if key in raw_bytes:
        result["whole_raw_tensor"].append(name)
        continue
    parts = name.split(".")
    hit = False
    if "attention" in parts and parts[parts.index("attention") + 1] in ("q", "k", "v"):
        layer, part, kind = int(parts[2]), parts[parts.index("attention") + 1], parts[-1]
        ref = derived[(layer, part, kind)]
        hit = ref.shape == arr.shape and np.array_equal(ref.astype("<f4"), arr.astype("<f4"))
    (result["qkv_slice_of_raw"] if hit else result["unmatched"]).append(name)
summary = {key: len(value) for key, value in result.items()}
summary["unmatched_names"] = result["unmatched"]
summary["raw_tensors_not_used_whole"] = sorted(n for n in raw if not any(n in v for k, v in raw_bytes.items() if any(b == n for b in v) and False))[:0]
write_json("d_params.json", {"summary": summary, "events": EVENTS})
print(json.dumps(summary, indent=1))
print("events:", json.dumps({k: v for k, v in EVENTS.items() if k != "writes_outside_repo_not_review"}))
lm_bias = bridge["lm_head._original_component.bias"]
extra = {"lm_head_bias_shape": list(lm_bias.shape), "lm_head_bias_all_zero": bool(np.all(lm_bias == 0.0)), "raw_has_embed_out_bias": "embed_out.bias" in raw}
write_json("d_params.json", {"summary": summary, "lm_head_bias": extra, "events": EVENTS})
print(json.dumps(extra))
