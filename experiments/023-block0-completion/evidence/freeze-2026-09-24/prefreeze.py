"""Experiment 023 pre-freeze check: strictly read-only."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HEAD = "89536b004b2e4536e83185cdfec31df4e7206982"
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
state = rd.load_results_state(ROOT / "outputs/experiment-023/results.json")
extract = state["phases"]["extract"]
check("extract complete once, no incident/stop", extract["status"] == "complete" and "incidents" not in extract and "stop" not in extract,
      f"{extract['started_at']} → {extract['completed_at']} at {extract['commit'][:7]}")
check("freeze not run: no confirmation file, never committed", not (ROOT / b0c.CONFIRMATION_RELATIVE_PATH).exists()
      and git("log", "--all", "--format=%H", "--", b0c.CONFIRMATION_RELATIVE_PATH) == "")
check("calibrate/lock/confirm/report not_started", all(state["phases"][p]["status"] == "not_started" for p in ("calibrate", "lock", "confirm", "report")),
      json.dumps({p: e["status"] for p, e in state["phases"].items()}))
check("both prompt ledgers empty", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
check("no calibration/lock/confirmation records", state["calibration"] == {} and state["lock"] is None and state["confirmation"] is None and state["confirmation_023"] is None)
check("results state unchanged since extract", state["state_sha256"] == "fcb2c0305a02c0c42a9ce39bc5a9c19c2c4f30c298d3a198c2644a751abb9e4d"
      and rc.file_sha256(ROOT / "outputs/experiment-023/results.json") == "9bb1e33594670343a41e5c08f81cca3a78f50d1f1c4e04882708879e508a5ea7")
data, index = ROOT / b0c.CELLS_DATA_RELATIVE_PATH, ROOT / b0c.CELLS_INDEX_RELATIVE_PATH
check("installed exposed-cells.f64", hashlib.sha256(data.read_bytes()).hexdigest() == "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4")
check("installed exposed-cells.json file", hashlib.sha256(index.read_bytes()).hexdigest() == "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe")
loaded = json.loads(index.read_text())
check("installed exposed-cells.json content", loaded["content_sha256"] == rc.content_digest(loaded) == "d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16")
computed = b0c.module_blobs()
for name, pinned in b0c.FROZEN_BLOBS.items():
    check(f"pin {name}", computed[name] == git("rev-parse", f"HEAD:src/neural_decompiler/{name}") == pinned, pinned)
check("block0_completion.py unchanged since extract", git("rev-parse", "HEAD:src/neural_decompiler/block0_completion.py") == b0c.own_blob()
      == "16d310fc7fab5599ec83b8bd8162fb9613f8dad2")
changed = git("diff", "--no-renames", "--name-only", "2111271892a8237f69c3ef18d6eec774e798ac5c", "HEAD").split()
check("no scientific path changed since extract", b0c.scientific_changes(changed) == [], json.dumps(changed))
print("\nPRE-FREEZE:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
