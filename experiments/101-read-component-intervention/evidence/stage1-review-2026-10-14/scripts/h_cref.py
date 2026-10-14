"""Item 3 support: each stored state's c_ref against the noun read of LN_final(h6) from the same stored state (the readout
program computes LN_final(h6) itself). Raw checkpoint weights (safetensors) only; no model object, no module call, no
token enters any computation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402  (first)

guard.self_test()
import json  # noqa: E402

import numpy  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

ROOT = guard.ROOT
EXP = ROOT / "experiments/101-read-component-intervention"
SNAP = Path("/Users/gaborhorvath-ulenius/.cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c")
lock = json.loads((EXP / "preregistration-lock.json").read_text())
data = numpy.frombuffer((EXP / "stage1/states.f64").read_bytes(), dtype="<f8")
blocks, offset = {}, 0
for entry in lock["stage1"]["states"]["layout"]:
    n = int(numpy.prod(entry["shape"]))
    blocks[entry["name"]] = torch.from_numpy(data[offset: offset + n].reshape(entry["shape"]).copy())
    offset += n
sd = load_file(str(SNAP / "model.safetensors"))
W_U = sd["embed_out.weight"].double()
w, b = sd["gpt_neox.final_layer_norm.weight"].double(), sd["gpt_neox.final_layer_norm.bias"].double()
# The noun list in NounSet order (80 exposed nouns: 79 scorable + 'peach'), from 020's pool through the frozen loader.
from neural_decompiler import upstream_localization as ul  # noqa: E402
import subprocess  # noqa: E402

inputs = ul.load_frozen_inputs(ROOT, tracked=lambda p: subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=ROOT, capture_output=True).returncode == 0)
nouns = list(inputs.pool.nouns)
out = {"frames": {}, "worst_abs": 0.0, "worst_rel": 0.0}
for meta in lock["stage1"]["states"]["meta"]["frames"]:
    fid = meta["frame_id"]
    h6, c_ref = blocks[f"{fid}/h6"], blocks[f"{fid}/c_ref"]
    c = h6 - h6.mean()
    lnf = w * c / torch.sqrt((c * c).mean() + 1e-5) + b
    logits = W_U @ lnf
    pred = torch.tensor([float(logits[n.sg_ids[0]] - logits[n.pl_ids[0]]) if n.single_token else 0.0 for n in nouns], dtype=torch.float64)
    err = float((pred - c_ref).abs().max())
    out["frames"][fid] = {"max_abs": err, "mean_c_ref_scorable": float(c_ref[[i for i, n in enumerate(nouns) if n.single_token]].mean())}
    out["worst_abs"] = max(out["worst_abs"], err)
    out["worst_rel"] = max(out["worst_rel"], err / max(float(c_ref.abs().max()), 1e-12))
out["n_nouns"] = len(nouns)
out["guard"] = guard.summary()
# criterion: the project's frozen logit-identity tolerance (rd.LOGIT_IDENTITY_TOLERANCE = 2e-2); the calibration recorded
# G4 logit maxima of 2.5e-3 to 3.03e-3 on the exposed frames (float32 logits); a frame or noun misalignment gives O(1) errors
out["criterion"] = "max-abs < 2e-2 (rd.LOGIT_IDENTITY_TOLERANCE); calibration G4_logit max 3.03e-3 for comparison"
out["ok"] = out["worst_abs"] < 2e-2 and len(nouns) == 80
(guard.OUT / "h_cref.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"c_ref vs W_U·LN_final(h6): worst max-abs {out['worst_abs']:.3e} (relative {out['worst_rel']:.3e}) over 36 frames × {len(nouns)} nouns; ok={out['ok']}")
means = [v["mean_c_ref_scorable"] for v in out["frames"].values()]
print(f"reference contrast c_ref (mean over the 79 scorable nouns) per frame: min {min(means):+.3f} median {sorted(means)[17]:+.3f} max {max(means):+.3f}")
print("guard:", guard.summary())
