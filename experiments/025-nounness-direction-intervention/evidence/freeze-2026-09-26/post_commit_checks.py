"""Experiment 025: read-only checks after the freeze-install commit. Tokenizer and committed files only: the model
loader, torch module calls, every capture/intervention entry point and every score/geometry function refuse; no file
is written."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import models
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_routing as rr

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
PATH = "experiments/025-nounness-direction-intervention/confirmation-v1.json"
EXPECTED = {"file": "54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a", "content": "6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4",
            "manifest": "0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61"}
refused: list[str] = []


def refuse(name):
    def call(*args, **kwargs):
        refused.append(name)
        raise RuntimeError(f"{name} refused in a read-only check")

    return call


torch.nn.Module.__call__ = refuse("torch.nn.Module.__call__")
models.load_model = refuse("load_model")
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    setattr(pm, name, refuse(f"pm.{name}"))
for module, names in ((rr, ("nounness", "full_scores", "calibration_scores", "centroid", "centroids", "score_bindings")),
                      (cr, ("score", "direction", "centroids", "cue_geometry", "cue_vectors", "geometry_block", "plurality_direction", "nearest_tokens"))):
    for name in names:
        setattr(module, name, refuse(f"{module.__name__.rsplit('.', 1)[1]}.{name}"))


def git(*args: str, text: bool = True):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=text).stdout


spec = importlib.util.spec_from_file_location("experiment_025_runner", ROOT / "experiments/025-nounness-direction-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
logs: list[str] = []
runner = runner_module.Runner(log=logs.append)

committed = git("show", f"HEAD:{PATH}", text=False)
working = (ROOT / PATH).read_bytes()
payload = json.loads(committed)
base = runner._base()
tokenizer = runner.tokenizer_loader(models.PYTHIA_70M)
again = cr.freeze_payload(tokenizer, pool=base.inputs.pool, freeze_024=base.inherited["confirmation"], prior=base.prior, config=cr.PRODUCTION)
rebuilt = (pm.canonical_json(again) + "\n").encode("utf-8")
keys = payload["manifest"]["S2-TARGET"]
untagged = {cr.untagged(key) for key in keys}
checks = {
    "commit_touches_exactly_one_path": git("show", "--name-only", "--format=", "HEAD").split() == [PATH],
    "committed_bytes_equal_the_reviewed_file": committed == working and hashlib.sha256(committed).hexdigest() == EXPECTED["file"],
    "content_sha256": payload["content_sha256"] == EXPECTED["content"] == rc.content_digest(payload),
    "the_40_cues_reconstruct_mechanically": again["cues"] == payload["cues"] and again["picks"] == payload["picks"] and len(payload["cues"]) == 40,
    "the_whole_file_reconstructs_byte_for_byte": rebuilt == committed,
    "manifest_90720_tagged_keys_same_digest": len(keys) == 90720 == len(set(keys)) and again["manifest"] == payload["manifest"]
    and pm.sha256_text(pm.canonical_json(payload["manifest"])) == payload["manifest_sha256"] == EXPECTED["manifest"],
    "spent_set_is_43632": len(base.forbidden) == 43632,
    "zero_collision_with_spent_keys": not (set(keys) | untagged) & base.forbidden,
    "zero_collision_with_patch_path_keys": not (set(keys) | untagged) & set(cr.PATCH_PATH_SPENT_KEYS) and set(cr.PATCH_PATH_SPENT_KEYS) <= base.forbidden,
    "runner_validate_passes": runner.validate() == 0,
    "nothing_refused_was_reached": not refused,
    "no_outputs_experiment_025": not (ROOT / "outputs/experiment-025").exists(),
    "tree_clean": git("status", "--porcelain").strip() == "",
}
print(json.dumps({"head": git("rev-parse", "HEAD").strip(), "checks": checks, "validate_log": logs[-1:], "refused": refused, "all": all(checks.values())}, indent=1))
