"""Item 3: the stock validate, loaded via importlib, with refusing model/tokenizer loaders. No other phase method is called."""
import guard  # noqa: F401
from guard import REPO, Seal, summary

import importlib.util
import os
import time

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


RUN_PY = os.path.join(REPO, "experiments/024-readout-routing-nounness/run.py")
spec = importlib.util.spec_from_file_location("run024_postinstall_review", RUN_PY)
run = importlib.util.module_from_spec(spec)
import sys  # noqa: E402

sys.modules[spec.name] = run  # dataclasses (with postponed annotations) look the module up by name
spec.loader.exec_module(run)
check("run.py loaded from the repository file", os.path.realpath(run.__file__) == os.path.realpath(RUN_PY), run.__file__)

seal = Seal()
refused = seal.refuse_captures()
print("capture entry points refused:", refused)
calls = {"model_loader": 0, "tokenizer_loader": 0}


def refuse_model(spec):
    calls["model_loader"] += 1
    raise RuntimeError("postinstall review: the model loader is refused in validate")


def refuse_tokenizer(spec):
    calls["tokenizer_loader"] += 1
    raise RuntimeError("postinstall review: the tokenizer loader is refused in validate")


logged = []
runner = run.Runner(log=logged.append, model_loader=refuse_model, tokenizer_loader=refuse_tokenizer)
check("runner.config is rr.PRODUCTION", runner.config is run.rr.PRODUCTION)
check("runner paths are the production ones", runner.results_path == run.RESULTS_PATH and runner.root == run.ROOT, str(runner.results_path))
t0 = time.time()
code = runner.validate()
print(f"validate() returned {code} in {time.time() - t0:.1f} s")
for line in logged:
    print("  LOG:", line)
check("validate() returned 0", code == 0, code)
check("log reports the lock and preregistration verified", any("lock and preregistration verified" in line for line in logged))
check("log reports the calibration record verified", any("calibration record verified" in line for line in logged))
check("log reports the confirmation (40 cues, 4320 keys)", any("(40 cues, 4320 keys)" in line for line in logged))
check("log reports 12 module blobs verified", any("module blobs 12 verified" in line for line in logged))
check("model loader never called", calls["model_loader"] == 0)
check("tokenizer loader never called", calls["tokenizer_loader"] == 0)
check("no capture entry point reached", seal.counts["capture_hits"] == 0, seal.capture_hits_by_name)
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"ITEM3 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
