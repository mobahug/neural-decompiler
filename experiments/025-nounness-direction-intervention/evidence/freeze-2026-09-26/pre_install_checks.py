"""Experiment 025: read-only checks before installing the reviewed freeze (no model, no tokenizer, no write)."""
import hashlib
import json
import subprocess
from pathlib import Path

from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
PATH = "experiments/025-nounness-direction-intervention/confirmation-v1.json"
EXPECTED = {"head": "c765148fdb612ae249e21ef3c830e4ceb1d12dd0", "file": "54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a",
            "content": "6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4", "manifest": "0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


data = (ROOT / PATH).read_bytes()
payload = json.loads(data)
keys = payload["manifest"]["S2-TARGET"]
checks = {
    "head_equals_origin_equals_remote": git("rev-parse", "HEAD") == git("rev-parse", "origin/main") == git("ls-remote", "origin", "refs/heads/main").split()[0] == EXPECTED["head"],
    "tree_is_exactly_the_untracked_freeze": git("status", "--porcelain=v1", "--untracked-files=all").splitlines() == [f"?? {PATH}"],
    "no_outputs_experiment_025": not (ROOT / "outputs/experiment-025").exists(),
    "no_lock_or_preregistration": not (ROOT / "experiments/025-nounness-direction-intervention/preregistration-lock.json").exists()
    and not (ROOT / "experiments/025-nounness-direction-intervention/preregistration.md").exists(),
    "file_sha256": hashlib.sha256(data).hexdigest() == EXPECTED["file"],
    "content_sha256": payload["content_sha256"] == EXPECTED["content"] == rc.content_digest(payload),
    "manifest_count_90720_unique": len(keys) == 90720 == len(set(keys)),
    "manifest_digest": pm.sha256_text(pm.canonical_json(payload["manifest"])) == payload["manifest_sha256"] == EXPECTED["manifest"],
    "no_gitattributes": not (ROOT / ".gitattributes").exists(),
    "no_active_git_hooks": not [p.name for p in (ROOT / ".git/hooks").iterdir() if not p.name.endswith(".sample")],
    "autocrlf_unset_or_false": subprocess.run(["git", "config", "--get", "core.autocrlf"], cwd=ROOT, capture_output=True, text=True).stdout.strip() in ("", "false"),
}
print(json.dumps({"head": git("rev-parse", "HEAD"), "checks": checks, "all": all(checks.values())}, indent=1))
