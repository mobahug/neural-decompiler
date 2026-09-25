"""Pre-lock checks for Experiment 024 (read-only; committed files and weights only; no prompt; the lock is NOT run).

Part 1: the repository and protocol state, the stock `validate`, the committed freeze and calibration bytes, the state,
the pins and 023's artifact. Part 2: the lock's weight-derived values reproduced with the canonical module against the
installed calibration record (weights only; module calls during the load counted, every forward refused afterwards):
the 40-score digest, the calibration maximum, 5 of 8 E above it, E > N. Any mismatch exits non-zero: lock must not run.
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
        raise PermissionError(f"the pre-lock checks may not write into the repository: {path}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


git("fetch", "-q", "origin")
head, origin = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
check("HEAD == origin/main == be74d23", head == origin and head.startswith("be74d23"), head)
check("working tree clean", git("status", "--porcelain") == "")
spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)


def refuse(*args, **kwargs):
    raise RuntimeError("no model or tokenizer in the validate check")


logs: list[str] = []
runner = runner_module.Runner(log=logs.append, model_loader=refuse, tokenizer_loader=refuse)
check("configuration is the frozen production one", runner.config is rr.PRODUCTION)
check("stock validate passes", runner.validate() == 0, logs[-1] if logs else "")
freeze, calibration = ROOT / rr.CONFIRMATION_RELATIVE_PATH, ROOT / rr.CALIBRATION_RELATIVE_PATH
record = json.loads(calibration.read_text(encoding="utf-8"))
check("freeze bytes", rc.file_sha256(freeze) == "68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60"
      and json.loads(freeze.read_text())["content_sha256"] == "87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87")
check("calibration bytes", rc.file_sha256(calibration) == "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d"
      and record["content_sha256"] == rc.content_digest(record) == "09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8")
state = rd.load_results_state(runner.results_path)
check("state digest", state["state_sha256"] == "5ed26580be4c28310d072bf2c451dbeba01e4b1f8710b65072d25c9281d31252")
check("calibrate complete once; lock/confirm/report not started", state["phases"]["calibrate"]["status"] == "complete" and not state["calibration"].get("incidents")
      and all(state["phases"][p]["status"] == "not_started" for p in ("lock", "confirm", "report")))
check("prompt ledger empty; no stage-2 file; no lock files", state["executed_prompt_keys"] == [] and not (ROOT / "outputs/experiment-024/stage2-measurements.pt").exists()
      and not (ROOT / rr.LOCK_RELATIVE_PATH).exists() and not (ROOT / "outputs/experiment-024/candidate-lock.json").exists())
check("pins", rr.assert_frozen_blobs() == rr.FROZEN_BLOBS)
check("023 committed artifacts", len(rr.verify_023_inputs(ROOT)) == 7)
check("022 committed inputs", len(b0c.verify_022_inputs(ROOT)) == 5)
changed = runner.changed_paths(state["phases"]["calibrate"]["commit"])
check("no scientific path changed since calibrate", changed is not None and rr.scientific_changes(changed) == [], f"{len(changed)} non-scientific paths")
base = runner._base()
installed, _ = runner._installed_record(state, base.digests)
confirmation, confirmation_sha = runner._confirmation(base)
rr.verify_confirmation_binding(installed, state, rr.confirmation_binding(confirmation, confirmation_sha))
check("the lock's record and confirmation preconditions", installed == record)

calls = {"during_load": 0, "after_load": 0}
original_call = torch.nn.Module.__call__
phase = {"loading": True}


def guarded_call(self, *args, **kwargs):
    if phase["loading"]:
        calls["during_load"] += 1
        return original_call(self, *args, **kwargs)
    calls["after_load"] += 1
    raise RuntimeError("a forward pass was attempted in the pre-lock check")


torch.nn.Module.__call__ = guarded_call
model = models.load_model(models.PYTHIA_70M)
phase["loading"] = False
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capture refused")))
check("parameter digest == the record's", rr.parameters_digest(model) == record["dependencies"]["model"]["parameters_sha256"])
W_E = pm.Weights.from_model(model).W_E
del model
check("score bindings recompute == the record's", rr.score_bindings(W_E, base.inputs.pool, b0c.exposed_units(base.inputs)) == record["score"])
fresh = rr.fresh_quantities(W_E, record["score"], confirmation, record)
e = [cue for cue in fresh["cues"] if cue["class"] == "E"]
n = [cue for cue in fresh["cues"] if cue["class"] == "N"]
check("40-score digest a703ac16…", fresh["scores_sha256"] == "a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995", fresh["scores_sha256"])
check("calibration maximum", fresh["maximum_calibration_score"] == 0.13502027836111233, fresh["maximum_calibration_score"])
check("5 of 8 E above the maximum", fresh["extrapolation"]["E_above_maximum"] == 5 and [c["word"] for c in e if c["above_calibration_maximum"]] == ["apple", "horse", "doctor", "poet", "dragon"])
check("every E above every N", min(c["nounness"] for c in e) > max(c["nounness"] for c in n))
check("the extrapolation sentence", fresh["extrapolation"]["sentence"].startswith("5 of the 8 E cues lie above the calibration population's maximum nounness (+0.135020)"),
      fresh["extrapolation"]["sentence"])
check("module calls: 0 during the load, 0 after", calls == {"during_load": 0, "after_load": 0}, calls)
check("no repository write", not REFUSED, REFUSED)
print("FAILURES:", failures if failures else "none", flush=True)
sys.exit(1 if failures else 0)
