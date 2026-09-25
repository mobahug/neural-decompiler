"""Guarded launcher for Experiment 024's production `freeze` (run exactly once).

Before the runner module is imported:
- `neural_decompiler.models.load_model` is replaced by a refusal (the freeze is tokenizer-only; the model is never
  loaded, so no weight, logit, hidden state or attention can exist);
- `torch.nn.Module.__call__` refuses (no forward of any module);
- every capture and intervention entry point of `plural_mechanism` refuses;
- an audit hook refuses opening Experiment 022's local calibration table or anything under `outputs/experiment-023/`,
  and refuses any write under the repository except the one confirmation file the freeze writes.
Every refusal is counted; the counts, the start and end times (UTC) and the exit status are written to this
scratchpad directory only.
"""
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
CONFIRMATION = (ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json").resolve()
events = {"module_calls": 0, "load_model": 0, "capture_calls": 0, "forbidden_reads": [], "forbidden_writes": [], "confirmation_writes": 0}


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
        raise PermissionError(f"the freeze may not open {text}")
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if writing and text.startswith(str(ROOT)) and "/.venv/" not in text:
        if path == CONFIRMATION:
            events["confirmation_writes"] += 1
            return
        events["forbidden_writes"].append(text)
        raise PermissionError(f"the freeze may not write {text}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402


def _refuse_module_call(self, *args, **kwargs):
    events["module_calls"] += 1
    raise RuntimeError("a torch module was called during the freeze")


def _refuse_load(*args, **kwargs):
    events["load_model"] += 1
    raise RuntimeError("the model was loaded during the freeze")


def _refuse_capture(*args, **kwargs):
    events["capture_calls"] += 1
    raise RuntimeError("a capture or intervention entry point was reached during the freeze")


torch.nn.Module.__call__ = _refuse_module_call
models.load_model = _refuse_load
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, _refuse_capture)

spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
assert runner_module.load_model is _refuse_load  # the runner's default model loader is the refusal

logs: list[str] = []


def log(message: str) -> None:
    print(message, flush=True)
    logs.append(message)


runner = runner_module.Runner(log=log)
assert runner.config is runner_module.rr.PRODUCTION
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
status = None
error = None
try:
    status = runner.freeze()
except BaseException as exc:  # recorded, then re-raised after the record is written
    error = f"{type(exc).__name__}: {exc}"
ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
record = {"started_at": started, "ended_at": ended, "exit_status": status, "error": error, "invocations": 1, "events": events, "logs": logs,
          "config": runner.config.to_json()["name"]}
(HERE / "freeze_run.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({key: record[key] for key in ("started_at", "ended_at", "exit_status", "error", "invocations")}), flush=True)
print(json.dumps(events), flush=True)
sys.exit(0 if status == 0 and error is None else 1)
