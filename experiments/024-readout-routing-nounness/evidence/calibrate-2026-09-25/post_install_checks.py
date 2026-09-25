"""Post-install checks of Experiment 024's calibration record (read-only; no model, no prompt; the lock is NOT run).

The tracked record's bytes and hashes; the stock `validate`; the record checker; the freeze binding in the record and
the state; the lock's own preconditions evaluated without running the lock (the phase rule, the changed paths since
calibrate, `_installed_record`, the confirmation binding); and the calibrated values preserved exactly.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
REFUSED = []


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if not writing:
        return
    try:
        path = Path(os.fsdecode(args[0])).resolve()
    except OSError:
        return
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
        REFUSED.append(str(path))
        raise PermissionError(f"the post-install checks may not write into the repository: {path}")


sys.addaudithook(_hook)

from neural_decompiler import models  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402


def refuse(*args, **kwargs):
    raise RuntimeError("no model or tokenizer may load in the post-install checks")


models.load_model = refuse
spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)

failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


path = ROOT / rr.CALIBRATION_RELATIVE_PATH
head_bytes = subprocess.run(["git", "show", f"HEAD:{rr.CALIBRATION_RELATIVE_PATH}"], cwd=ROOT, check=True, capture_output=True).stdout
record = json.loads(path.read_text(encoding="utf-8"))
check("tracked bytes == working bytes == candidate bytes", head_bytes == path.read_bytes() == (ROOT / "outputs/experiment-024/candidate-calibration.json").read_bytes())
check("file sha256", rc.file_sha256(path) == "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d")
check("content sha256", record["content_sha256"] == rc.content_digest(record) == "09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8")
logs: list[str] = []
runner = runner_module.Runner(log=logs.append, model_loader=refuse, tokenizer_loader=refuse)
check("stock validate", runner.validate() == 0, logs[-1] if logs else "")
rr.verify_calibration_record(record, rr.PRODUCTION)
check("installed record checker (rr.verify_calibration_record, production)", True)
state = rd.load_results_state(runner.results_path)
check("state digest unchanged", state["state_sha256"] == "5ed26580be4c28310d072bf2c451dbeba01e4b1f8710b65072d25c9281d31252")
freeze = {"path": rr.CONFIRMATION_RELATIVE_PATH, "file_sha256": "68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60",
          "content_sha256": "87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87"}
check("freeze binding in the record and the state", record["confirmation_024"] == freeze == state["confirmation_024"])
check("calibration source bound", record["exposed_cells"] == rr.calibration_source() == record["dependencies"]["calibration_source"])
# the lock's preconditions, evaluated without running the lock
rr.assert_phase_allowed("lock", state)
check("phase rule allows lock (calibrate complete, lock not started, no lock incident)", True)
changed = runner.changed_paths(state["phases"]["calibrate"]["commit"])
check("changed paths since calibrate are all non-scientific", changed is not None and rr.scientific_changes(changed) == [], changed)
base = runner._base()
installed, installed_sha = runner._installed_record(state, base.digests)
check("lock reads the installed record (tracked, the candidate this run wrote, verified, same inputs)", installed == record and installed_sha == rc.file_sha256(path))
confirmation, confirmation_sha = runner._confirmation(base)
rr.verify_confirmation_binding(record, state, rr.confirmation_binding(confirmation, confirmation_sha))
check("the committed freeze binding verifies against the record and the state", True)
line, floor, null, effective = record["line"], record["primary_floor"], record["null"], record["effective_threshold"]
preserved = {"slope": (line["slope"], 1.2992017464639374), "intercept": (line["intercept"], -2.481260357420486),
             "residual_sd": (line["residual_sd"], 0.2570037337349923), "exposed Spearman": (record["descriptive"]["calibration_rho"], 0.5296617364493499),
             "F_rho": (floor["F_rho"], 0.24411074612857814), "null_975": (null["null_975"], 0.3136960600375234), "effective": (effective["value"], 0.3136960600375234),
             "binds": (effective["binds"], "null_975"), "undefined draws": (floor["undefined"], 0), "draw indices": (record["arrays_sha256"]["draw_indices"][:8], "955930ce"),
             "draw values": (record["arrays_sha256"]["draw_rho"][:8], "7c03376c"), "draw flags": (record["arrays_sha256"]["draw_defined"][:8], "76b5b970"),
             "null values": (record["arrays_sha256"]["null_rho"][:8], "92e63029"), "permutations": (record["arrays_sha256"]["null_permutations"][:8], "a68f7291"),
             "parameters": (record["dependencies"]["model"]["parameters_sha256"][:8], "fd953f1c"), "embedding": (record["score"]["embedding_sha256"][:8], "9cd6f39b"),
             "mu_noun": (record["score"]["mu_noun_sha256"][:8], "87930f01"), "mu_cue": (record["score"]["mu_cue_sha256"][:8], "f8cca954")}
for name, (value, expected) in preserved.items():
    check(f"preserved {name}", value == expected, value)
check("E–N rule bound, not evaluated", record["constants"]["configuration"]["guard"]["max_upper"] == 321 and not any(key in json.dumps(record) for key in ('"K":', '"D_EN":')))
check("no fresh score in the record", "fresh" not in record and len(record["calibration_cues"]) == 139)
check("ledger empty; lock/confirm/report not started", state["executed_prompt_keys"] == [] and all(state["phases"][p]["status"] == "not_started" for p in ("lock", "confirm", "report")))
check("no repository write", not REFUSED, REFUSED)
print("FAILURES:", failures if failures else "none", flush=True)
sys.exit(1 if failures else 0)
