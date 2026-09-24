"""Experiment 023 pre-lock check: strictly read-only; uses the stock runner's own installed-artifact readers."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HEAD = "13b8d392f9e88844e2673c530a37228337cc64b7"
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
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


check("HEAD == origin", git("rev-parse", "HEAD") == HEAD == git("ls-remote", "origin", "refs/heads/main").split()[0], HEAD)
check("tree clean incl. untracked", git("status", "--porcelain", "--untracked-files=all") == "")
runner = run023.Runner(log=lambda message: print("   runner:", message))
check("stock validate", runner.validate() == 0)
inputs, confirmation_022, sha_022, digests, forbidden = runner._base()
confirmation, confirmation_sha = runner._confirmation(inputs, confirmation_022, sha_022, forbidden)
state = rd.load_results_state(runner.results_path)
check("state: extract + calibrate complete once; lock/confirm/report not_started; no lock incident", state["phases"]["extract"]["status"] == "complete"
      and state["phases"]["calibrate"]["status"] == "complete" and all(state["phases"][p]["status"] == "not_started" for p in ("lock", "confirm", "report"))
      and not state["phases"]["lock"].get("incidents") and state["lock"] is None, json.dumps({p: e["status"] for p, e in state["phases"].items()}))
check("ledgers empty", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
b0c.assert_phase_allowed("lock", state)
check("lock allowed by the phase rules", True)
record, record_sha = runner._installed_record(state, digests)  # tracked, == the candidate this run wrote, strict verification, same inputs
check("installed calibration record == this run's candidate (strict verification)", record_sha == "94db6df26156d7a134e51f8afbf10707ca18030525f62dd08f02fb298e9ba03a")
b0c.verify_confirmation_binding(record, state, b0c.confirmation_binding(confirmation, confirmation_sha))
check("record and state bind the committed confirmation", confirmation_sha == "5fadfa503f4fe35308cb4473220a1d9cd6f7a46e3852f638594b8081909825f4")
runner._cells_unchanged(record["exposed_cells"])
check("installed exposed cells == the record's binding", True)
changed = runner.changed_paths(state["phases"]["calibrate"]["commit"])
check("no scientific change since calibrate", changed is not None and b0c.scientific_changes(changed) == [], json.dumps(changed))
check("runtime and versions == 020's explore", bool(runner._check_runtime(inputs.closure)))
out = ROOT / "outputs/experiment-023"
check("no lock candidates yet", not any((out / name).exists() for name in ("candidate-lock.json", "candidate-preregistration.md", "candidate-locked-y1-table.f64",
                                                                       "candidate-locked-y1-table.json"))
      and not any((ROOT / p).exists() for p in (b0c.LOCK_RELATIVE_PATH, b0c.PREREGISTRATION_RELATIVE_PATH, b0c.Y1_TABLE_RELATIVE_PATH, b0c.Y1_TABLE_INDEX_RELATIVE_PATH)))
check("020 results file is the pinned one (and the archive restores it)", rc.file_sha256(ROOT / "outputs/experiment-020/results.json") == rc.RESULTS_020_FILE_SHA256
      == digests["results_020_file"] and hashlib.sha256(subprocess.run(["gunzip", "-c", str(ROOT / "experiments/023-block0-completion/evidence/archive/experiment-020-results.json.gz")],
                                                                          capture_output=True, check=True).stdout).hexdigest() == rc.RESULTS_020_FILE_SHA256)
check("pins", b0c.module_blobs() == b0c.FROZEN_BLOBS and git("rev-parse", "HEAD:src/neural_decompiler/block0_completion.py") == "16d310fc7fab5599ec83b8bd8162fb9613f8dad2")
print("\nPRE-LOCK:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
