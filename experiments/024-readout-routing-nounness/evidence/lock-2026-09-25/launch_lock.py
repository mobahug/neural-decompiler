"""Guarded launcher for Experiment 024's production `lock` (run exactly once).

Before the runner module is imported:
- every capture and intervention entry point of `plural_mechanism` refuses (no prompt can run);
- `torch.nn.Module.__call__` is counted while the weights are loaded and the programs are built (the runner's
  `_weights`: `load_model`, then `ModelPrograms.from_model`), and refused from the moment the parameter digest is taken
  (the first step after them) to the end;
- `load_model` is counted (one load expected);
- an audit hook refuses opening Experiment 022's local calibration table or anything under `outputs/experiment-023/`,
  and refuses any repository write outside `outputs/experiment-024/`.
Every event is counted; the counts, the start and end times (UTC) and the exit status go to this scratchpad directory.
"""
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
OUTPUTS = (ROOT / "outputs/experiment-024").resolve()
events = {"load_model": 0, "module_calls_while_loading": 0, "module_calls_refused": 0, "capture_calls": 0, "forbidden_reads": [], "forbidden_writes": [],
          "output_writes": []}
phase = {"sealed": False}


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        path = Path(os.fsdecode(args[0])).resolve()
    except OSError:
        return
    text = str(path)
    if text.endswith("calibration-table.pt") or "/outputs/experiment-023/" in text:
        events["forbidden_reads"].append(text)
        raise PermissionError(f"lock may not open {text}")
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if writing and text.startswith(str(ROOT)) and "/.venv/" not in text:
        if text.startswith(str(OUTPUTS) + os.sep):
            events["output_writes"].append(Path(text).name)
            return
        events["forbidden_writes"].append(text)
        raise PermissionError(f"lock may not write {text}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

original_call = torch.nn.Module.__call__
original_load = models.load_model
original_digest = rr.parameters_digest


def _guarded_call(self, *args, **kwargs):
    if not phase["sealed"]:
        events["module_calls_while_loading"] += 1
        return original_call(self, *args, **kwargs)
    events["module_calls_refused"] += 1
    raise RuntimeError("a torch module was called during lock after the weights were read")


def _counted_load(spec):
    events["load_model"] += 1
    return original_load(spec)


def _sealing_digest(model):
    phase["sealed"] = True  # the weights are loaded and the programs built: from here on, no module may be called
    return original_digest(model)


def _refuse_capture(*args, **kwargs):
    events["capture_calls"] += 1
    raise RuntimeError("a capture or intervention entry point was reached during lock")


torch.nn.Module.__call__ = _guarded_call
models.load_model = _counted_load
rr.parameters_digest = _sealing_digest
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, _refuse_capture)

spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
assert runner_module.load_model is _counted_load

logs: list[str] = []


def log(message: str) -> None:
    print(message, flush=True)
    logs.append(message)


runner = runner_module.Runner(log=log)
assert runner.config is rr.PRODUCTION
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
status = None
error = None
try:
    status = runner.lock()
except BaseException as exc:  # recorded, never retried
    error = f"{type(exc).__name__}: {exc}"
ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
record = {"started_at": started, "ended_at": ended, "exit_status": status, "error": error, "invocations": 1, "sealed": phase["sealed"], "events": events, "logs": logs,
          "config": runner.config.to_json()["name"]}
(HERE / "lock_run.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({key: record[key] for key in ("started_at", "ended_at", "exit_status", "error", "invocations", "sealed")}), flush=True)
print(json.dumps(events), flush=True)
sys.exit(0 if status == 0 and error is None else 1)
