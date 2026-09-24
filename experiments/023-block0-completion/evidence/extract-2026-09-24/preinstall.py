"""Experiment 023 pre-install check: strictly read-only."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HEAD = "2111271892a8237f69c3ef18d6eec774e798ac5c"
EXPECTED = {"candidate-exposed-cells.f64": "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4",
            "candidate-exposed-cells.json": "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe"}
CONTENT = "d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16"
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


check("HEAD", git("rev-parse", "HEAD") == HEAD, git("rev-parse", "HEAD"))
remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
check("origin/main (ls-remote)", remote == HEAD, remote)
check("tree clean incl. untracked", git("status", "--porcelain", "--untracked-files=all") == "")
out = ROOT / "outputs/experiment-023"
for name, digest in EXPECTED.items():
    actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
    check(f"{name} sha256", actual == digest, actual)
index = json.loads((out / "candidate-exposed-cells.json").read_text())
check("index content_sha256", index["content_sha256"] == CONTENT == rc.content_digest(index), index["content_sha256"])
state = rd.load_results_state(out / "results.json")
extract = state["phases"]["extract"]
check("extract complete once, no incident/stop", extract["status"] == "complete" and "incidents" not in extract and "stop" not in extract
      and extract["commit"] == HEAD, f"{extract['started_at']} → {extract['completed_at']}")
check("state candidate digests == reviewed", (state["extract"]["data_sha256"], state["extract"]["index_sha256"]) == tuple(EXPECTED.values()))
check("later phases not_started", all(state["phases"][p]["status"] == "not_started" for p in ("calibrate", "lock", "confirm", "report")),
      json.dumps({p: e["status"] for p, e in state["phases"].items()}))
check("prompt ledgers empty", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
check("no calibration/lock/confirmation records in state", state["calibration"] == {} and state["lock"] is None and state["confirmation"] is None
      and state["confirmation_023"] is None)
check("results.json sha256 unchanged since extract", rc.file_sha256(out / "results.json") == "9bb1e33594670343a41e5c08f81cca3a78f50d1f1c4e04882708879e508a5ea7")
check("install targets absent", not (ROOT / b0c.CELLS_DATA_RELATIVE_PATH).exists() and not (ROOT / b0c.CELLS_INDEX_RELATIVE_PATH).exists())
check("install targets not ignored", subprocess.run(["git", "check-ignore", "-q", b0c.CELLS_DATA_RELATIVE_PATH], cwd=ROOT).returncode == 1
      and subprocess.run(["git", "check-ignore", "-q", b0c.CELLS_INDEX_RELATIVE_PATH], cwd=ROOT).returncode == 1)
attrs = git("check-attr", "-a", b0c.CELLS_DATA_RELATIVE_PATH, b0c.CELLS_INDEX_RELATIVE_PATH)
check("no git attributes on the targets", attrs == "", attrs or "none")
print("\nPRE-INSTALL:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
