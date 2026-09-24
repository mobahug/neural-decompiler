"""Experiment 023 pre-calibrate check: strictly read-only."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HEAD = "4e5deebd021bc284143b01b76b45efd28e1a1344"
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def tracked(relative):
    return subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True).returncode == 0


check("HEAD", git("rev-parse", "HEAD") == HEAD, git("rev-parse", "HEAD"))
check("origin/main (ls-remote)", git("ls-remote", "origin", "refs/heads/main").split()[0] == HEAD)
check("tree clean incl. untracked", git("status", "--porcelain", "--untracked-files=all") == "")
state = rd.load_results_state(ROOT / "outputs/experiment-023/results.json")
extract = state["phases"]["extract"]
check("extract complete once, no incident/stop", extract["status"] == "complete" and "incidents" not in extract and "stop" not in extract, extract["commit"][:7])
check("calibrate/lock/confirm/report not_started", all(state["phases"][p]["status"] == "not_started" for p in ("calibrate", "lock", "confirm", "report")))
check("no calibration/lock/confirmation records", state["calibration"] == {} and state["lock"] is None and state["confirmation"] is None and state["confirmation_023"] is None)
check("both ledgers empty", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
check("results state unchanged since extract", state["state_sha256"] == "fcb2c0305a02c0c42a9ce39bc5a9c19c2c4f30c298d3a198c2644a751abb9e4d")
out = ROOT / "outputs/experiment-023"
check("no calibration outputs yet", not (out / "candidate-calibration.json").exists() and not (out / "draw-values.pt").exists() and not (ROOT / b0c.CALIBRATION_RELATIVE_PATH).exists())
for relative, digest in ((b0c.CELLS_DATA_RELATIVE_PATH, "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4"),
                         (b0c.CELLS_INDEX_RELATIVE_PATH, "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe"),
                         (b0c.CONFIRMATION_RELATIVE_PATH, "5fadfa503f4fe35308cb4473220a1d9cd6f7a46e3852f638594b8081909825f4")):
    check(f"{relative} committed and exact", tracked(relative) and hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
          == hashlib.sha256(subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, check=True).stdout).hexdigest())
check("state's candidate digests == the installed artifact", (state["extract"]["data_sha256"], state["extract"]["index_sha256"]) ==
      ("d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4", "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe"))
computed = b0c.module_blobs()
check("all 11 pins == HEAD blobs", all(computed[n] == git("rev-parse", f"HEAD:src/neural_decompiler/{n}") == p for n, p in b0c.FROZEN_BLOBS.items()))
check("block0_completion.py == extract's blob", git("rev-parse", "HEAD:src/neural_decompiler/block0_completion.py") == "16d310fc7fab5599ec83b8bd8162fb9613f8dad2")
changed = git("diff", "--no-renames", "--name-only", extract["commit"], "HEAD").split()
check("no scientific path changed since extract", b0c.scientific_changes(changed) == [], f"{len(changed)} paths changed")
check("B, tag, rank", (b0c.B, b0c.DRAW_TAG, b0c.lower_rank(b0c.B), b0c.GUARD_MIN, b0c.GAP_MIN) == (10_000, "023|primary", 250, 0.90, 0.02))
print("\nPRE-CALIBRATE:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
