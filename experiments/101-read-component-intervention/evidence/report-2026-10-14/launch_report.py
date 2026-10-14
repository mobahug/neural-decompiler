"""Guarded launcher for Experiment 101's `report` (025's report-launcher pattern: no model forward, no prompt). It runs
from a scratchpad directory outside the repository and writes its records there (O_EXCL, never overwritten).

`report` recomputes the prediction table from the committed stage-1 states and the weights (G6 at the report; weights
only) and renders `outputs/experiment-101/report.md` from the state; it may re-render. The guards:
- `torch.nn.Module.__call__` refuses for the whole run; every module's and model class's `forward` is sealed after the
  counted load; every capture and intervention entry point (and alias) refuses;
- every measurement-side function refuses (`ri.run_arm`, `ri.capture_reference`, `ri.measure_frame`, `ri.stage_runs`,
  `ri.run_gates`, `ri.frame_values`, `ri.primary_statistics`, `ri.secondary_statistics`, `ri.calibration_results`);
  the program functions the G6 recomputation needs stay available;
- the audit hook (fail closed, the stage-1 launcher version 2's resolution rules) allows repository writes only to
  `outputs/experiment-101/report.md` and the results state (through its temporary files and renames).
Mode `check` runs the same guards and computes G6 without writing; mode `production` calls `runner.report()` once.
"""
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

signal.signal(signal.SIGHUP, signal.SIG_IGN)
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
OUTPUTS = (ROOT / "outputs/experiment-101").resolve()
MODE = os.environ.get("REPORT101_MODE", "")
if MODE not in ("check", "production"):
    raise SystemExit("set REPORT101_MODE to 'check' or 'production'")
LAUNCHER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
WRITABLE = {OUTPUTS / "report.md", OUTPUTS / "results.json"}
READABLE_OUTPUTS = (ROOT / "outputs/experiment-020/results.json", OUTPUTS)
os.chdir(ROOT)
events = {"mode": MODE, "load_model": 0, "refused": [], "forbidden_reads": [], "forbidden_writes": [], "outputs_reads": [], "report_invocations": 0}
UNRESOLVED = Path("/<unresolved>")


def _path(value, dir_fd=None):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    raw = os.fsdecode(value)
    if not os.path.isabs(raw):
        if isinstance(dir_fd, int) and not isinstance(dir_fd, bool) and dir_fd >= 0:
            try:
                base = fcntl.fcntl(dir_fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0].decode()
            except (OSError, ValueError):
                return UNRESOLVED
        else:
            base = os.getcwd()
        raw = os.path.join(base, raw)
    try:
        return Path(raw).resolve()
    except (OSError, ValueError):
        return UNRESOLVED


def _inside(path: Path, root: Path) -> bool:
    return path == root or str(path).startswith(str(root) + os.sep)


def _in_repo(path: Path) -> bool:
    return path == UNRESOLVED or (_inside(path, ROOT) and not _inside(path, ROOT / ".venv"))


def _temp_state(path: Path) -> bool:
    return MODE == "production" and path.parent == OUTPUTS and path.name.startswith(".results-") and path.name.endswith(".json")


def write_allowed(path: Path) -> bool:
    return MODE == "production" and (path in WRITABLE or _temp_state(path))


def decide(event, args):
    if event == "open":
        path = _path(args[0]) if args else None
        if path is None:
            return None
        mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
        if not writing:
            if path != UNRESOLVED and _inside(path, ROOT / "outputs") and not any(_inside(path, allowed) for allowed in READABLE_OUTPUTS):
                return ("forbidden_reads", str(path))
            return None
        if _in_repo(path) and not write_allowed(path):
            return ("forbidden_writes", str(path))
        return None
    if event in ("os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir", "os.chmod"):
        dir_fd = args[2] if event in ("os.mkdir", "os.chmod") and len(args) > 2 else args[1] if event in ("os.remove", "os.rmdir", "shutil.rmtree") and len(args) > 1 else None
        path = _path(args[0], dir_fd) if args else UNRESOLVED
        if path is None or not _in_repo(path):
            return None
        if event == "os.remove" and _temp_state(path):
            return None
        if event == "os.mkdir" and MODE == "production" and path == OUTPUTS:
            return None
        return ("forbidden_writes", f"{event} {path}")
    if event in ("os.rename", "os.link", "os.symlink"):
        if event == "os.symlink":
            paths = [_path(args[1], args[2] if len(args) > 2 else None)] if len(args) > 1 else [UNRESOLVED]
        else:
            paths = [_path(args[0], args[2] if len(args) > 2 else None), _path(args[1], args[3] if len(args) > 3 else None)] if len(args) > 1 else [UNRESOLVED]
        paths = [p for p in paths if p is not None]
        if not any(_in_repo(p) for p in paths):
            return None
        if event == "os.rename" and len(paths) == 2 and _temp_state(paths[0]) and paths[1] == OUTPUTS / "results.json":
            return None
        return ("forbidden_writes", f"{event} {' -> '.join(str(p) for p in paths)}")
    return None


def _hook(event, args):
    decision = decide(event, args)
    if event == "open" and decision is None and args:
        path = _path(args[0])
        if path is not None and path != UNRESOLVED and _inside(path, ROOT / "outputs") and str(path) not in events["outputs_reads"]:
            events["outputs_reads"].append(str(path))
    if decision is not None:
        kind, detail = decision
        events[kind].append(detail)
        raise PermissionError(f"report launcher ({MODE}): {kind}: {detail}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import capture, interventions, models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _refusal(kind: str, name: str):
    def refuse(*args, **kwargs):
        events[kind].append(name)
        raise RuntimeError(f"{name} refused during report")

    return refuse


def _refuse_module_call(self, *args, **kwargs):
    events["refused"].append(f"torch.nn.Module.__call__ ({type(self).__name__})")
    raise RuntimeError("a torch module was called during report")


torch.nn.Module.__call__ = _refuse_module_call
ENTRY = {id(capture.run_capture): ("capture.run_capture", capture.run_capture),
         id(interventions.run_interventions): ("interventions.run_interventions", interventions.run_interventions),
         id(pm.capture_prompt): ("pm.capture_prompt", pm.capture_prompt), id(pm.run_patched): ("pm.run_patched", pm.run_patched),
         id(rd.capture_frame_020): ("rd.capture_frame_020", rd.capture_frame_020)}
REFUSALS = {key: _refusal("refused", name) for key, (name, _) in ENTRY.items()}


def _sweep_aliases() -> list[str]:
    replaced = []
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name.startswith("neural_decompiler") or module_name.startswith("experiment_101")):
            continue
        for attribute, value in list(vars(module).items()):
            if id(value) in ENTRY and value is ENTRY[id(value)][1]:
                setattr(module, attribute, REFUSALS[id(value)])
                replaced.append(f"{module_name}.{attribute}")
    return replaced


aliases = _sweep_aliases()
for name in ("run_arm", "capture_reference", "measure_frame", "stage_runs", "run_gates", "frame_values", "primary_statistics", "secondary_statistics",
             "calibration_results"):
    setattr(ri, name, _refusal("refused", f"ri.{name}"))
original_load = models.load_model


def _counted_load(spec, *args, **kwargs):
    events["load_model"] += 1
    model = original_load(spec, *args, **kwargs)
    for module_name, module in model.named_modules():
        object.__setattr__(module, "forward", _refusal("refused", f"{module_name or type(module).__name__}.forward"))
    for cls in {type(module) for module in model.modules()}:
        for method in ("forward", "run_with_hooks", "run_with_cache", "generate"):
            if method in cls.__dict__ and callable(cls.__dict__[method]):
                setattr(cls, method, _refusal("refused", f"{cls.__qualname__}.{method}"))
    return model


models.load_model = _counted_load
spec = importlib.util.spec_from_file_location("experiment_101_runner", ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
assert runner_module.load_model is _counted_load
aliases += _sweep_aliases()
logs: list[str] = []


def log(message: str) -> None:
    print(message, flush=True)
    logs.append(message)


runner = runner_module.Runner(log=log)
record: dict = {"mode": MODE, "launcher_sha256": LAUNCHER_SHA256, "head": _git("rev-parse", "HEAD"), "tree_before": _git("status", "--porcelain", "--untracked-files=all"),
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "aliases": sorted(set(aliases))}
status, error = None, None
try:
    if not sys.flags.dont_write_bytecode or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("run with python -B and HF_HUB_OFFLINE=1")
    if record["tree_before"]:
        raise RuntimeError("the tree is not clean")
    state = rd.load_results_state(OUTPUTS / "results.json")
    record["state_before"] = state["state_sha256"]
    if MODE == "check":
        base = runner._base()
        confirmation, _ = runner._confirmation(base)
        lock, _ = runner._installed_lock()
        runner._check_runtime(base.inputs.closure)
        record["g6"] = runner._g6_at_report(base, lock, confirmation)
        status = 0 if record["g6"].get("bitwise_equal") else 1
    else:
        events["report_invocations"] += 1
        status = runner.report()
        after = rd.load_results_state(OUTPUTS / "results.json")
        record["report_block"] = after.get("report")
        record["report_sha256"] = hashlib.sha256((OUTPUTS / "report.md").read_bytes()).hexdigest()
        record["state_after"] = after["state_sha256"]
    record["logs"] = logs
except BaseException as exc:
    error = f"{type(exc).__name__}: {exc}"
    record["logs"] = logs
record.update({"ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_status": status, "error": error, "events": events})
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
fd = os.open(HERE / f"{MODE}_run_{stamp}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
os.write(fd, (json.dumps(record, indent=1, default=str) + "\n").encode("utf-8"))
os.fsync(fd)
os.close(fd)
print(json.dumps({k: record.get(k) for k in ("mode", "exit_status", "error", "state_before", "state_after", "report_sha256")}, default=str), flush=True)
print(json.dumps({k: v for k, v in events.items() if k != "outputs_reads"}, default=str), flush=True)
sys.exit(0 if status == 0 and error is None and not events["refused"] and not events["forbidden_writes"] else 1)
