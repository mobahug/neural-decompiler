"""Experiment 025: read-only checks before the production lock. No model load, no forward, no write: the model loader,
torch module calls and every capture/intervention entry point refuse. The tokenizer is not needed either."""
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
EXPECTED = {"head": "d3ecbd6feb1bd2aefe08dc0f8d799815be47c519", "file": "54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a",
            "content": "6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4", "manifest": "0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61"}
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


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


spec = importlib.util.spec_from_file_location("experiment_025_runner", ROOT / f"{cr.EXPERIMENT_DIR}/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
logs: list[str] = []
runner = runner_module.Runner(log=logs.append)

data = (ROOT / cr.CONFIRMATION_RELATIVE_PATH).read_bytes()
payload = json.loads(data)
keys = payload["manifest"]["S2-TARGET"]
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
lock_024 = base.inherited["lock"]
from huggingface_hub import constants as hf_constants  # noqa: E402

snapshot = Path(hf_constants.HF_HUB_CACHE) / "models--EleutherAI--pythia-70m-deduped" / "snapshots" / models.PYTHIA_70M.revision
checks = {
    "head_equals_origin_equals_remote": git("rev-parse", "HEAD") == git("rev-parse", "origin/main") == git("ls-remote", "origin", "refs/heads/main").split()[0] == EXPECTED["head"],
    "tree_clean": git("status", "--porcelain", "--untracked-files=all") == "",
    "stock_validate_passes": runner.validate() == 0,
    "freeze_file_sha256": hashlib.sha256(data).hexdigest() == EXPECTED["file"] == confirmation_sha,
    "freeze_content_sha256": payload["content_sha256"] == EXPECTED["content"] == rc.content_digest(payload) == confirmation.content_sha256,
    "manifest_90720": len(keys) == 90720 == len(set(keys)) == len(confirmation.tagged_keys()),
    "manifest_digest": pm.sha256_text(pm.canonical_json(payload["manifest"])) == EXPECTED["manifest"] == payload["manifest_sha256"],
    "freeze_tracked": runner.tracked(ROOT / cr.CONFIRMATION_RELATIVE_PATH),
    "no_outputs_experiment_025": not (ROOT / "outputs/experiment-025").exists(),
    "no_results_state_or_ledger": not (ROOT / "outputs/experiment-025/results.json").exists(),
    "no_lock_or_preregistration": not (ROOT / cr.LOCK_RELATIVE_PATH).exists() and not (ROOT / cr.PREREGISTRATION_RELATIVE_PATH).exists(),
    "pinned_module_blobs": cr.assert_frozen_blobs() == cr.FROZEN_BLOBS,
    "committed_020_024_inputs_recheck": runner._recheck() == {"ok": True},
    "runtime_and_versions_equal_020_explore": runner._check_runtime(base.inputs.closure) is not None,
    "model_binding_freeze": payload["model"] == {"model_id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision},
    "model_binding_024_lock": (lock_024["dependencies"]["model"]["model_id"], lock_024["dependencies"]["model"]["revision"])
    == (models.PYTHIA_70M.model_id, models.PYTHIA_70M.revision),
    "local_checkpoint_snapshot_present": snapshot.is_dir() and any(snapshot.iterdir()),
    "nothing_refused_was_reached": not refused,
}
print(json.dumps({"head": git("rev-parse", "HEAD"), "checks": checks, "all": all(checks.values()), "validate_log": logs[:1], "refused": refused,
                  "model": {"model_id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision}, "snapshot": str(snapshot)}, indent=1))
