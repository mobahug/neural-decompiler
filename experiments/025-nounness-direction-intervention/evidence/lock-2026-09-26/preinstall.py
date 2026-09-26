"""Experiment 025: read-only checks before installing the reviewed candidate lock and preregistration (no model, no
tokenizer, no write)."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

from neural_decompiler import cue_rotation as cr
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
os.chdir(ROOT)
OUT = ROOT / "outputs/experiment-025"
EXPECTED = {"head": "d3ecbd6feb1bd2aefe08dc0f8d799815be47c519", "lock_file": "1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be",
            "lock_content": "8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb", "prereg": "e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919",
            "state": "8eeb8184faf556211e68579d4682c85f545c877051641afe07dd5eac2b701dad", "freeze_file": "54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a",
            "freeze_content": "6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4"}


def git(*args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


lock_bytes, prereg_bytes = (OUT / "candidate-lock.json").read_bytes(), (OUT / "candidate-preregistration.md").read_bytes()
lock = json.loads(lock_bytes)
state = rd.load_results_state(OUT / "results.json")
freeze = (ROOT / cr.CONFIRMATION_RELATIVE_PATH).read_bytes()
checks = {
    "cwd_is_repo_root": Path.cwd().resolve() == ROOT,
    "head_equals_origin_equals_remote": git("rev-parse", "HEAD") == git("rev-parse", "origin/main") == git("ls-remote", "origin", "refs/heads/main").split()[0] == EXPECTED["head"],
    "tree_clean": git("status", "--porcelain", "--untracked-files=all") == "",
    "freeze_unchanged": hashlib.sha256(freeze).hexdigest() == EXPECTED["freeze_file"] and json.loads(freeze)["content_sha256"] == EXPECTED["freeze_content"],
    "lock_complete_once_no_incident": state["phases"]["lock"]["status"] == "complete" and not state["phases"]["lock"].get("incidents") and state["lock"]["content_sha256"] == EXPECTED["lock_content"],
    "confirm_and_report_not_started": state["phases"]["confirm"] == {"status": "not_started"} and state["phases"]["report"] == {"status": "not_started"} and state["confirmation"] is None,
    "ledger_empty": state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [],
    "only_the_three_candidate_files": sorted(p.name for p in OUT.iterdir()) == ["candidate-lock.json", "candidate-preregistration.md", "results.json"],
    "candidate_lock_file_sha256": hashlib.sha256(lock_bytes).hexdigest() == EXPECTED["lock_file"],
    "candidate_lock_content_sha256": lock["content_sha256"] == EXPECTED["lock_content"] == rc.content_digest(lock),
    "candidate_preregistration_sha256": hashlib.sha256(prereg_bytes).hexdigest() == EXPECTED["prereg"] == state["lock"]["preregistration_sha256"],
    "results_state_sha256": state["state_sha256"] == EXPECTED["state"],
    "install_targets_absent": not (ROOT / cr.LOCK_RELATIVE_PATH).exists() and not (ROOT / cr.PREREGISTRATION_RELATIVE_PATH).exists(),
    "install_targets_not_ignored": all(subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 1
                                       for path in (cr.LOCK_RELATIVE_PATH, cr.PREREGISTRATION_RELATIVE_PATH)),  # one path per call
    "no_gitattributes_or_autocrlf": not (ROOT / ".gitattributes").exists()
    and subprocess.run(["git", "config", "--get", "core.autocrlf"], cwd=ROOT, capture_output=True, text=True).stdout.strip() in ("", "false"),
}
print(json.dumps({"head": git("rev-parse", "HEAD"), "checks": checks, "all": all(checks.values())}, indent=1))
