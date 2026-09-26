"""Item 9: control determinism. A separate process seeds (and advances) every random-number generator differently --
Python's random, NumPy's legacy and new generators, torch's CPU generator -- before and between generations, runs under a
PYTHONHASHSEED given on the command line, and recomputes the 280 controls both with the reviewer's own stream and with
cr.control_directions, plus the whole cr.geometry_block, under 1, 4 and 8 torch threads and in a shuffled cue order.
Weights come straight from the checkpoint file (no model object at all)."""
import guard  # noqa: F401

import json
import os
import random
import sys
from pathlib import Path

ROOT = guard.ROOT
HERE = guard.HERE

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

import r2a_stream  # noqa: E402

guard.seal()
guard.refuse_model_load()
seed = int(sys.argv[1])


def scramble(k: int) -> None:
    random.seed(seed * 7919 + k)
    np.random.seed((seed * 104729 + k) % 2 ** 32)
    torch.manual_seed(seed * 1299709 + k)
    random.random(); np.random.rand(17); torch.rand(33); np.random.default_rng(seed + k).normal(size=5)  # advance every stream


scramble(0)
snapshot = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
with safe_open(str(snapshot / "model.safetensors"), framework="pt") as handle:
    W_E = handle.get_tensor("gpt_neox.embed_in.weight").to(torch.float32)
record = json.loads((ROOT / "experiments/024-readout-routing-nounness/calibration-v1.json").read_text(encoding="utf-8"))
freeze = json.loads((ROOT / "experiments/025-nounness-direction-intervention/confirmation-v1.json").read_text(encoding="utf-8"))
lock = json.loads((ROOT / "outputs/experiment-025/candidate-lock.json").read_text(encoding="utf-8"))
inputs = ul.load_frozen_inputs(ROOT)
confirmation = cr.confirmation_from_payload(freeze, inputs.pool, cr.PRODUCTION)
out = {"seed": seed, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "hash_of_str": hash("025|control"), "embedding_digest": rc.tensor_digest(W_E) if False else None}

torch.set_num_threads(4)
d = cr.direction(W_E, record)
mine, theirs = [], []
for k, token in enumerate(confirmation.tokens):
    scramble(1000 + k)
    g = cr.cue_geometry(W_E, int(token["token_id"]), d)
    mine.append(torch.stack([r2a_stream.control(int(token["token_id"]), j, g.E_hat, g.t_hat) for j in range(1, 8)]))
    scramble(2000 + k)
    theirs.append(torch.stack(cr.control_directions(g, 7)))
out["controls_mine_digest"] = rc.tensor_digest(torch.stack(mine))
out["controls_cr_digest"] = rc.tensor_digest(torch.stack(theirs))
out["controls_expected"] = "e12581d4be183798b42e2f673670eda711ddaf87359c28df5350c1c806a37315"
out["per_cue_controls_equal_lock"] = all(rc.tensor_digest(m) == c["controls_sha256"] for m, c in zip(mine, lock["geometry"]["cues"]))

blocks = {}
for threads in (1, 4, 8):
    torch.set_num_threads(threads)
    scramble(3000 + threads)
    block = cr.geometry_block(W_E, record, inputs.pool, confirmation.tokens, cr.PRODUCTION)
    combined = torch.stack([torch.stack([block["vectors32"][int(t["token_id"])][c] for c in cr.PRODUCTION.conditions]) for t in confirmation.tokens])
    blocks[threads] = {"verify_against_lock": cr.verify_geometry_against_lock(block["block"], lock), "vectors32_digest": rc.tensor_digest(combined)}
out["geometry_block_by_threads"] = blocks
torch.set_num_threads(4)

shuffled = list(confirmation.tokens)
random.Random(seed).shuffle(shuffled)
scramble(4000)
block = cr.geometry_block(W_E, record, inputs.pool, shuffled, cr.PRODUCTION)
by_id = {c["token_id"]: c for c in block["block"]["cues"]}
out["shuffled_order_per_cue_equal_lock"] = all(by_id[c["token_id"]] == c for c in lock["geometry"]["cues"])
out["shuffled_order_first_three"] = [t["word"] for t in shuffled[:3]]
out["guard"] = guard.summary()
(HERE / f"r3_determinism_{seed}.out.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
