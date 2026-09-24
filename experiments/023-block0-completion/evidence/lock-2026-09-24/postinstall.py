"""Experiment 023 post-install checks at the lock commit: strictly read-only. Runs confirm's own pre-prompt validation
(the runner's readers and b0c.validate_lock) without loading a model, executing a prompt or writing anything."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402


def refuse(*args, **kwargs):
    raise AssertionError("a model was loaded during a read-only check")


failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


runner = run023.Runner(model_loader=refuse, log=lambda message: print("   runner:", message))
results_before = rc.file_sha256(runner.results_path)
check("stock validate", runner.validate() == 0)
inputs, confirmation_022, sha_022, digests, forbidden = runner._base()
confirmation, confirmation_sha = runner._confirmation(inputs, confirmation_022, sha_022, forbidden)
state = runner._state_for("confirm", digests)  # clean tree at a committed SHA, versions, and confirm allowed by the phase rules
check("confirm allowed by the phase rules", state["phases"]["confirm"]["status"] == "not_started")
b0c.assert_ledger_isolated(state["executed_prompt_keys"], forbidden | confirmation.manifest_keys(), "the ledger before confirm")
check("ledger isolated and empty", state["executed_prompt_keys"] == [])
paths = {name: ROOT / relative for name, relative in (("lock", b0c.LOCK_RELATIVE_PATH), ("preregistration", b0c.PREREGISTRATION_RELATIVE_PATH),
                                                        ("y1_data", b0c.Y1_TABLE_RELATIVE_PATH), ("y1_index", b0c.Y1_TABLE_INDEX_RELATIVE_PATH))}
lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
record, record_sha = runner._installed_record(state, digests)
runner._cells_unchanged(record["exposed_cells"])
git = runner.git_state()
changed = runner.changed_paths(lock["protocol_code_commit"])
y1_index = b0c.validate_lock(lock, state=state, digests=digests, record=record, record_file_sha256=record_sha, confirmation=confirmation,
                             confirmation_file_sha256=confirmation_sha, cells_files=record["exposed_cells"], noun_keys=runner._noun_keys(inputs),
                             exposed_states=inputs.closure["exploration"]["locked_states"], preregistration_text=paths["preregistration"].read_text(encoding="utf-8"),
                             y1_index_text=paths["y1_index"].read_text(encoding="utf-8"), y1_file_sha256=rc.file_sha256(paths["y1_data"]), git_state=git,
                             tracked=all(runner.tracked(path) for path in paths.values()), changed_paths=changed)
check("validate_lock (confirm's pre-prompt validation) accepts the installed lock", y1_index["file_sha256"] == "37603f058500e056214a8a6ae413ef2b27eb94c5e1d618caaa8c9a61121a2f96",
      f"changed since lock commit: {changed}")
check("no Y2 table on disk", not any((ROOT / lock["y2_table"][key]).exists() for key in ("data_path", "index_path")))
check("runtime and versions == 020's explore (as confirm requires)", bool(runner._check_runtime(inputs.closure)))
check("results state untouched by these checks", rc.file_sha256(runner.results_path) == results_before)
check("tree clean", subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True).stdout == "")
print("\nPOST-INSTALL:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
