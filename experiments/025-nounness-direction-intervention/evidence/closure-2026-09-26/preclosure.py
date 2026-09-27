"""Pre-closure verification of Experiment 025 (read-only; no model, no prompt, no phase; nothing written).

It checks:
- the repository and the remote at the confirm-evidence commit `c8d9fad`;
- all three state phases complete, with no incident and no descriptive failure;
- the measurements, the generated report and the final state;
- reverting the two report entries gives the pre-report (confirm-final) state, and removing the descriptive records
  from that gives the result-write state;
- the installed freeze, lock and preregistration, and the direct module binding;
- the runner's validate.

Any mismatch exits non-zero, and the closure does not proceed.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
os.chdir(ROOT)
REFUSED = []


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    raw = os.fsdecode(args[0])
    if not writing or not os.path.isabs(raw):
        return
    path = Path(raw).resolve()
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
        REFUSED.append(str(path))
        raise PermissionError(f"the pre-closure verification may not write into the repository: {path}")


sys.addaudithook(_hook)

from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402

GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
COMMIT = "c8d9fad08ab7181460cdd58bee6075668796fd40"
OUT = ROOT / "outputs/experiment-025"
failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, env=GIT_ENV).stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


check("head, origin and remote at the confirm-evidence commit", git("rev-parse", "HEAD") == git("rev-parse", "origin/main") == git("ls-remote", "origin", "refs/heads/main").split()[0] == COMMIT)
check("tree clean (no untracked file), no stash", git("status", "--porcelain", "--untracked-files=all") == "" and git("stash", "list") == "")
state = rd.load_results_state(OUT / "results.json")
unsigned = {k: v for k, v in state.items() if k != "state_sha256"}
check("final state", state["state_sha256"] == "534dad3907d94fa78c3a88a9e74bd88051b6ada57a09daff8007d2a3495bcdb8" and sha(OUT / "results.json") == "fef818dced09bbf4faae43e68bc7da8c97bfeaed5190637099132968f4408134")
check("all state phases complete", {k: v["status"] for k, v in state["phases"].items()} == {p: "complete" for p in cr.STATE_PHASES})
check("no incident", not any(v.get("incidents") for v in state["phases"].values()) and "incident" not in state["confirmation"])
check("no descriptive failure", not (state["confirmation"].get("descriptives") or {}).get("failures"))
check("report", sha(OUT / "report.md") == "ad9f85e62e74bf10b22060dc6f49ab75ddf3e79b7c81814d29ac6aa19fb75407" == state["report"]["sha256"])
check("measurements", sha(OUT / "stage2-measurements.pt") == "87284bddfba16a1b7709d7d33f98aa1a0898b1d85767956a524d2950d80a7788")
confirm_final = {**unsigned, "report": None, "phases": {**unsigned["phases"], "report": {"status": "not_started"}}}
confirm_final_sha = pm.sha256_text(pm.canonical_json(confirm_final))
confirm_final_file = hashlib.sha256((pm.canonical_json({**confirm_final, "state_sha256": confirm_final_sha}) + "\n").encode("utf-8")).hexdigest()
check("reverting the two report entries gives the confirm-final state", (confirm_final_sha, confirm_final_file)
      == ("1d5d37f035bc3153fec6efac527b088f64cd42fcd74e59680f4388daea98b33c", "c827bb69292f11ff6baf2194b4d8e34ccb0fbad9058c76d1abf2a3581724efb4"))
result_write = {**{k: v for k, v in confirm_final.items() if k != "confirmation"}, "confirmation": {k: v for k, v in confirm_final["confirmation"].items() if k != "descriptives"}}
check("removing the descriptive records gives the result-write state", pm.sha256_text(pm.canonical_json(result_write)) == "aa3db023c4a4afbe730a0e55df00d6d378184bcf8fdf77c967f4d0aad11907c4")
results = state["confirmation"]["results"]
check("the official result", results["outcome"]["label"] == "CAUSAL_EFFECT_NOT_ESTABLISHED" and (results["A"]["positive"], results["B"]["positive"], results["G"]["positive"]) == (26, 23, 37)
      and {results[n]["threshold"] for n in "ABG"} == {27} and results["criterion"]["reference_tail"]["value"] == 0.01923865414210013)
check("ledger and accounting", len(state["executed_prompt_keys"]) == 90720 and state["confirmation"]["accounting"] == {"equal": True, "executed": 90720, "ledger": 90720, "manifest": 90720})
lock = json.loads((ROOT / cr.LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
check("installed lock and preregistration", sha(ROOT / cr.LOCK_RELATIVE_PATH) == "1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be"
      and lock["content_sha256"] == rc.content_digest(lock) == state["lock"]["content_sha256"] == state["confirmation"]["lock_sha256"]
      and sha(ROOT / cr.PREREGISTRATION_RELATIVE_PATH) == "e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919")
check("installed freeze", sha(ROOT / cr.CONFIRMATION_RELATIVE_PATH) == "54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a")
for relative in (cr.CONFIRMATION_RELATIVE_PATH, cr.LOCK_RELATIVE_PATH, cr.PREREGISTRATION_RELATIVE_PATH):
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True, capture_output=True, env=GIT_ENV).stdout
    check(f"{relative} equals its HEAD blob byte for byte", committed == (ROOT / relative).read_bytes())
for phase in ("lock", "confirm"):
    try:
        cr.assert_phase_allowed(phase, state)
        check(f"the phase rule refuses another {phase}", False, "allowed")
    except cr.PhaseError as refusal:
        check(f"the phase rule refuses another {phase}", True, refusal)
check("the freeze file exists and is tracked, so the runner's freeze refuses a second run", (ROOT / cr.CONFIRMATION_RELATIVE_PATH).exists()
      and git("ls-files", cr.CONFIRMATION_RELATIVE_PATH) == cr.CONFIRMATION_RELATIVE_PATH)
print("note: the phase rule permits re-rendering the report (idempotent by design); the one-time report launcher's sentinel refuses a second run", flush=True)
since_lock = [line for line in git("diff", "--name-only", "d3ecbd6feb1bd2aefe08dc0f8d799815be47c519", "HEAD").splitlines() if line]
check("no scientific path changed since the lock commit", cr.scientific_changes(since_lock) == [], f"{len(since_lock)} paths changed, all non-scientific")
check("module binding", lock["module"]["blob"] == cr.own_blob() == git("rev-parse", "HEAD:src/neural_decompiler/cue_rotation.py"))
spec = importlib.util.spec_from_file_location("experiment_025_runner", ROOT / f"{cr.EXPERIMENT_DIR}/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
logs = []
runner = runner_module.Runner(log=logs.append, model_loader=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")),
                              tokenizer_loader=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no tokenizer")))
check("the runner's validate", runner.validate() == 0, logs[-1] if logs else "")
check("nothing written into the repository", not REFUSED)
print(f"{'PASS' if not failures else 'FAIL'}: {len(failures)} failure(s) {failures}")
sys.exit(1 if failures else 0)
