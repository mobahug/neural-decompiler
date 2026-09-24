"""Experiment 023 pre-confirm check at c7efec7: strictly read-only. Ends with confirm's own pre-prompt validation
(validate_lock and the rest), run without a model or a prompt."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HEAD = "c7efec7f32709dc20ecb83cae16bb73927a14366"
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


def refuse(*args, **kwargs):
    raise AssertionError("a model was loaded during a read-only check")


check("HEAD == origin == c7efec7", git("rev-parse", "HEAD") == HEAD == git("ls-remote", "origin", "refs/heads/main").split()[0])
check("tree clean incl. untracked", git("status", "--porcelain", "--untracked-files=all") == "")
E = ROOT / "experiments/023-block0-completion"
expected = {"preregistration-lock.json": "4bd5a5b14627768d49398c273fa1ce387db2e4739d7d31b7258072ef48beac9c", "preregistration.md": "dc198785b6350d20165f589236741f5f08ffc845da33840ca156c7d71780dc07",
            "locked-y1-table.f64": "37603f058500e056214a8a6ae413ef2b27eb94c5e1d618caaa8c9a61121a2f96", "locked-y1-table.json": "3bf85e186db7dfbf8c80f45557275058317e259cb5995dcd2f73e327dfd9644d",
            "calibration-v1.json": "94db6df26156d7a134e51f8afbf10707ca18030525f62dd08f02fb298e9ba03a", "confirmation-v1.json": "5fadfa503f4fe35308cb4473220a1d9cd6f7a46e3852f638594b8081909825f4",
            "exposed-cells.f64": "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4", "exposed-cells.json": "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe"}
for name, digest in expected.items():
    committed = subprocess.run(["git", "show", f"HEAD:experiments/023-block0-completion/{name}"], cwd=ROOT, capture_output=True, check=True).stdout
    check(f"{name}: installed == committed == reviewed", hashlib.sha256((E / name).read_bytes()).hexdigest() == hashlib.sha256(committed).hexdigest() == digest)
state = rd.load_results_state(ROOT / "outputs/experiment-023/results.json")
check("state: extract/calibrate/lock complete; confirm/report not_started; no incident", [state["phases"][p]["status"] for p in b0c.STATE_PHASES]
      == ["complete", "complete", "complete", "not_started", "not_started"] and not state["phases"]["confirm"].get("incidents") and state["confirmation"] is None,
      state["state_sha256"])
check("both ledgers empty", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
check("state lock == installed lock", state["lock"]["content_sha256"] == json.loads((E / "preregistration-lock.json").read_text())["content_sha256"]
      == "97ca520f342e212d5172b1476e9d5a80c6c2622f80a4f99f4e15e0a399007117")
out = ROOT / "outputs/experiment-023"
check("no Y2 table, stage-2 file or report yet", not any((out / n).exists() for n in ("y2-table.f64", "y2-table.json", "stage2-measurements.pt", "report.md")))
check("pins", b0c.module_blobs() == b0c.FROZEN_BLOBS and git("rev-parse", "HEAD:src/neural_decompiler/block0_completion.py") == "16d310fc7fab5599ec83b8bd8162fb9613f8dad2")
check("020 results file == the pinned one", rc.file_sha256(ROOT / "outputs/experiment-020/results.json") == rc.RESULTS_020_FILE_SHA256)
# confirm's own pre-prompt sequence, read-only
runner = run023.Runner(model_loader=refuse, log=lambda message: print("   runner:", message))
check("stock validate", runner.validate() == 0)
inputs, confirmation_022, sha_022, digests, forbidden = runner._base()
confirmation, confirmation_sha = runner._confirmation(inputs, confirmation_022, sha_022, forbidden)
state = runner._state_for("confirm", digests)
b0c.assert_ledger_isolated(state["executed_prompt_keys"], forbidden | confirmation.manifest_keys(), "the ledger before confirm")
paths = {n: ROOT / r for n, r in (("lock", b0c.LOCK_RELATIVE_PATH), ("preregistration", b0c.PREREGISTRATION_RELATIVE_PATH), ("y1_data", b0c.Y1_TABLE_RELATIVE_PATH),
                                   ("y1_index", b0c.Y1_TABLE_INDEX_RELATIVE_PATH))}
lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
record, record_sha = runner._installed_record(state, digests)
runner._cells_unchanged(record["exposed_cells"])
y1_index = b0c.validate_lock(lock, state=state, digests=digests, record=record, record_file_sha256=record_sha, confirmation=confirmation,
                             confirmation_file_sha256=confirmation_sha, cells_files=record["exposed_cells"], noun_keys=runner._noun_keys(inputs),
                             exposed_states=inputs.closure["exploration"]["locked_states"], preregistration_text=paths["preregistration"].read_text(encoding="utf-8"),
                             y1_index_text=paths["y1_index"].read_text(encoding="utf-8"), y1_file_sha256=rc.file_sha256(paths["y1_data"]), git_state=runner.git_state(),
                             tracked=all(runner.tracked(p) for p in paths.values()), changed_paths=runner.changed_paths(lock["protocol_code_commit"]))
check("validate_lock accepts the installed lock", y1_index["file_sha256"] == "37603f058500e056214a8a6ae413ef2b27eb94c5e1d618caaa8c9a61121a2f96")
check("runtime and versions == 020's explore", bool(runner._check_runtime(inputs.closure)))
manifest = confirmation.manifest()
keys = manifest["S1-REF"] + manifest["S1-VALIDITY"] + manifest["S2-TARGET"]["Y1"] + manifest["S2-TARGET"]["Y2"]
check("frozen manifest: 18 + 18 + 2,592 + 432 = 3,060 unique keys; none forbidden", len(keys) == len(set(keys)) == 3060 and not set(keys) & forbidden)
print("\nPRE-CONFIRM:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
