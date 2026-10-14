"""Check 2, the reads a correct confirm makes that no rehearsal reaches, judged by the launcher's REAL hook on the REAL
repository (ROOT = the repository, working directory = the repository root, as in production) — read-only:
- the hook runs under REHEARSAL rules, which refuse every repository write (reads are judged identically in both
  modes: the read branch of ``decide`` does not depend on the mode), and my guard, installed first, refuses every
  write outside this review's directory; child git processes run with GIT_OPTIONAL_LOCKS=0;
- ``runner._base()``, ``_confirmation``, ``_calibration``, ``_installed_lock``, ``rd.load_results_state``;
- ``runner._committed_stage1(lock, confirmation, record)`` — the barrier's and G6-before-result's re-read of the three
  stage-1 files (untracked now; it checks them against the record only; no model, no table recomputation);
- ``runner._recheck()`` — called by confirm on the SUCCESS path before the result write (run.py:859; a False is an
  incident there) and in every incident recording.
Nothing of the runner's phases runs; no model is loaded. Output: out/f_reads_real_repo.json.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, check_launcher, extract_launcher, install_guard, write_json  # noqa: E402

assert os.environ.get("GIT_OPTIONAL_LOCKS") == "0", "run with GIT_OPTIONAL_LOCKS=0"
check_launcher()
recorded: list = []
install_guard(recorded)
os.chdir(REPO)
ns = extract_launcher(REPO, "rehearsal")
sys.addaudithook(ns["_hook"])  # the real hook: raises PermissionError on a refusal

from neural_decompiler import readout_decompilation as rd  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment_101_runner_review", REPO / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
logs: list = []
runner = runner_module.Runner(log=logs.append)
steps: dict = {}


def step(name, fn):
    t0 = time.time()
    try:
        value = fn()
        steps[name] = {"ok": True, "seconds": round(time.time() - t0, 3)}
        return value
    except BaseException as error:  # noqa: BLE001
        steps[name] = {"ok": False, "seconds": round(time.time() - t0, 3), "error": f"{type(error).__name__}: {error}"}
        return None


base = step("_base", runner._base)
confirmation_pair = step("_confirmation", lambda: runner._confirmation(base))
confirmation, confirmation_sha = confirmation_pair
calibration_pair = step("_calibration", lambda: runner._calibration(confirmation, confirmation_sha))
lock_pair = step("_installed_lock", runner._installed_lock)
state = step("load_results_state", lambda: rd.load_results_state(runner.results_path))
committed = step("_committed_stage1 (the barrier's and G6's re-read)", lambda: runner._committed_stage1(lock_pair[0], confirmation, state["stage1"]))
recheck = step("_recheck (success path before the result; incident recording)", runner._recheck)
steps["_recheck_value"] = recheck
outputs_paths = sorted({p for r in recorded if r["event"] == "open" for p in (r["paths"] or []) if isinstance(p, str) and p.startswith(str(REPO / "outputs") + "/")})
repo_writes = [r for r in recorded if r["writing"] and any(isinstance(p, str) and p.startswith(str(REPO) + "/") for p in (r["paths"] or []))]
out = {"steps": steps, "hook_refusals": {k: ns["events"][k] for k in ("forbidden_reads", "forbidden_writes")}, "outputs_paths_opened": outputs_paths,
       "hook_outputs_reads": ns["events"]["outputs_reads"], "repo_writes": repo_writes, "n_events": len(recorded),
       "committed_states": len(committed[0]) if committed else None, "logs": logs}
write_json("f_reads_real_repo.json", out)
for name, value in steps.items():
    print(f"{name:<64} {value}")
print("hook refusals:", out["hook_refusals"])
print("outputs paths opened:", outputs_paths)
print("repo writes:", repo_writes)
