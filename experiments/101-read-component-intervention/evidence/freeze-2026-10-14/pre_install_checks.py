"""Experiment 101 freeze: the checks before the confirmation file is committed (read-only; no model, no forward)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HERE = Path(__file__).resolve().parent
PATH = "experiments/101-read-component-intervention/confirmation-v1.json"


def git(*args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


record = json.loads((HERE / "freeze_run.json").read_text())
data = (ROOT / PATH).read_bytes()
payload = json.loads(data)
checks = {
    "head_is_the_freeze_head": git("rev-parse", "HEAD") == record["environment"]["head"],
    "branch_is_exp101_phases": git("rev-parse", "--abbrev-ref", "HEAD") == "exp101-phases",
    "tree_holds_exactly_the_untracked_freeze_file": git("status", "--porcelain", "--untracked-files=all") == f"?? {PATH}",
    "no_outputs_experiment_101": not (ROOT / "outputs/experiment-101").exists(),
    "no_lock_calibration_or_stage1": not any((ROOT / "experiments/101-read-component-intervention" / name).exists()
                                             for name in ("preregistration-lock.json", "preregistration.md", "calibration-v1.json", "stage1")),
    "file_sha256_is_the_recorded": hashlib.sha256(data).hexdigest() == record["post_write"]["file_sha256"] == record["facts"]["expected_file_sha256"],
    "content_sha256_is_the_recorded": payload["content_sha256"] == record["post_write"]["content_sha256"] == record["facts"]["content_sha256"],
    "manifest_counts": len(payload["manifest"]["S1-REF"]) == 36 and len(payload["manifest"]["S2-TARGET"]) == 1980,
    "manifest_sha256_is_the_recorded": payload["manifest_sha256"] == record["facts"]["manifest_sha256"],
    "no_gitattributes": not (ROOT / ".gitattributes").exists(),
    "no_active_git_hook": not [p.name for p in (ROOT / ".git/hooks").iterdir() if p.is_file() and not p.name.endswith(".sample")],
    "autocrlf_unset_or_false": git("config", "--get", "core.autocrlf") in ("", "false"),
    "ends_with_one_newline": data.endswith(b"\n") and not data.endswith(b"\n\n"),
}
print(json.dumps({"checks": checks, "all_true": all(checks.values()), "head": git("rev-parse", "HEAD"), "file_sha256": hashlib.sha256(data).hexdigest(),
                  "bytes": len(data)}, indent=1))
sys.exit(0 if all(checks.values()) else 1)
