"""Guarded launcher for Experiment 024's production `report` (run exactly once).

Usage: ``launch_report.py --check`` (the read-only pre-report check only; `report` is not invoked) or
``launch_report.py --report-once`` (the check, then the one production invocation).

- **A sentinel** is created exclusively per mode before anything else, so each mode runs once.
- **A record-only audit hook** records repository writes outside `outputs/experiment-024/`, the writes inside it, and
  any open of Experiment 022's local table or Experiment 023's local outputs.
- **No model may run.** The runner is built with refusing model and tokenizer loaders. Every prompt, capture,
  intervention, measurement and ledger entry point refuses, and so does `torch.nn.Module.__call__`; each attempt is
  counted. `report` needs none of them: it renders the completed confirmation state and the installed calibration
  record.
- **The pre-report check** (read-only):
  - HEAD, origin and the remote are the evidence commit, and the tree is clean;
  - the state is `076ab9f9…` / `e636210c…`, and the measurements `b21babe1…`;
  - confirm is complete and report not started, with no incident;
  - there is no `report.md`, and the phase rule allows report;
  - the installed lock and preregistration are unchanged.

  A mismatch stops before `report` is invoked. The exact pre-report state bytes are copied to this directory, for the
  reversal check afterwards.
- **One `runner.report()`**: its outcome is recorded, never retried.
"""
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) == 2 else ""
if MODE not in ("--check", "--report-once"):
    raise SystemExit("usage: launch_report.py --check | --report-once")
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
PREFIX = "check" if MODE == "--check" else "report"
SENTINEL = HERE / ("CHECK_LAUNCHED" if MODE == "--check" else "REPORT_LAUNCHED")
RECORD = HERE / f"{PREFIX}_record.json"
COMMIT = "b80409b1464c8ab59f82d026e104f4510247c69e"
EXPECTED = {
    "state": "076ab9f984f3ba6244b4df767d8c8fd7ebd40a634521a46a3b881d7da1604e92",
    "state_file": "e636210c84a4f3f329e9a5fde5baa493d730b6bf77cb419273056e6c89085a7a",
    "stage2": "b21babe189d1cccb9ae359cfabf74baaa3a8337cac04612bc0d4aaa232b4dc93",
    "lock_file": "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d",
    "prereg_file": "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54",
    "calibration_file": "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d",
}
GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


try:
    descriptor = os.open(SENTINEL, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
except FileExistsError:
    print(f"REFUSED: {SENTINEL} exists; this mode has already run once", flush=True)
    sys.exit(4)
os.write(descriptor, f"{MODE} {utc_now()} pid {os.getpid()}\n".encode())
os.close(descriptor)

ROOT_TEXT = str(ROOT) + os.sep
OUTPUTS_TEXT = str(ROOT / "outputs/experiment-024") + os.sep
TABLE_022 = str(ROOT / "outputs/experiment-022/calibration-table.pt")
OUTPUTS_023 = str(ROOT / "outputs/experiment-023") + os.sep
events = {"repository_writes_outside_outputs": [], "output_writes": {}, "forbidden_reads": [], "relative_opens": 0, "hook_errors": 0}


def _hook(event, args):
    if event != "open":
        return
    try:
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        raw = os.fsdecode(args[0])
        if not os.path.isabs(raw):
            events["relative_opens"] += 1
            return
        text = os.path.normpath(raw)
        if text == TABLE_022 or text.startswith(OUTPUTS_023):
            events["forbidden_reads"].append(text)
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND)))
        if writing and text.startswith(ROOT_TEXT) and "/.venv/" not in text:
            if text.startswith(OUTPUTS_TEXT):
                name = os.path.basename(text)
                events["output_writes"][name] = events["output_writes"].get(name, 0) + 1
            else:
                events["repository_writes_outside_outputs"].append(text)
    except Exception:  # noqa: BLE001
        events["hook_errors"] += 1


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

counts = {"report_invocations": 0, "model_loads_refused": 0, "tokenizer_loads_refused": 0, "entry_points_refused": 0, "module_calls_refused": 0}
record = {"mode": MODE, "launcher_started_at": utc_now(), "pid": os.getpid(), "commit_expected": COMMIT}


def write_record() -> None:
    record.update({"counts": counts, "events": events, "record_written_at": utc_now()})
    RECORD.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")


def stop(reason: str) -> None:
    record["stopped_before_report"] = reason
    write_record()
    print(f"STOPPED before report (report not invoked): {reason}", flush=True)
    sys.exit(3)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, env=GIT_ENV).stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- no model may run -------------------------------------------------------------------------------------------------
def _refuse(kind):
    def refuse(*args, **kwargs):
        counts[kind] += 1
        raise RuntimeError(f"report may not reach a {kind.replace('_', ' ')}")
    return refuse


for module, names in ((pm, ("capture_prompt", "run_capture", "run_patched", "run_interventions", "record_execution", "measure_prompt")),
                      (ul, ("stage_two_022", "one_022"))):
    for name in names:
        if hasattr(module, name):
            setattr(module, name, _refuse("entry_points_refused"))
torch.nn.Module.__call__ = _refuse("module_calls_refused")

# -- the pre-report check (read-only) ----------------------------------------------------------------------------------
out = ROOT / "outputs/experiment-024"
state_path = out / "results.json"
before = rd.load_results_state(state_path)
check = {
    "head": git("rev-parse", "HEAD"), "origin": git("rev-parse", "origin/main"), "remote": git("ls-remote", "origin", "refs/heads/main").split()[0],
    "tree": git("status", "--porcelain", "--untracked-files=all"),
    "state": before["state_sha256"], "state_file": sha(state_path), "stage2": sha(out / "stage2-measurements.pt"),
    "lock_file": sha(ROOT / rr.LOCK_RELATIVE_PATH), "prereg_file": sha(ROOT / rr.PREREGISTRATION_RELATIVE_PATH),
    "calibration_file": sha(ROOT / rr.CALIBRATION_RELATIVE_PATH),
    "phases": {name: entry["status"] for name, entry in before["phases"].items()},
    "incident": (before.get("confirmation") or {}).get("incident"), "phase_incidents": {name: entry.get("incidents") for name, entry in before["phases"].items() if entry.get("incidents")},
    "report_md_exists": (out / "report.md").exists(), "ledger": len(before["executed_prompt_keys"]), "outputs": sorted(path.name for path in out.iterdir()),
}
try:
    rr.assert_phase_allowed("report", before)
    check["phase_rule_allows_report"] = True
except rr.PhaseError as error:
    check["phase_rule_allows_report"] = str(error)
record["pre_report_check"] = check
print(f"pre-report check: {json.dumps(check)}", flush=True)
problems = []
if not (check["head"] == check["origin"] == check["remote"] == COMMIT) or check["tree"] != "":
    problems.append("HEAD/origin/remote are not the evidence commit, or the tree is not clean")
for key in ("state", "state_file", "stage2", "lock_file", "prereg_file", "calibration_file"):
    if check[key] != EXPECTED[key]:
        problems.append(f"{key} differs")
if check["phases"] != {"calibrate": "complete", "lock": "complete", "confirm": "complete", "report": "not_started"} or check["incident"] or check["phase_incidents"]:
    problems.append("the phase state is not confirm complete / report not started without incident")
if check["report_md_exists"] or check["phase_rule_allows_report"] is not True or check["ledger"] != 4_320:
    problems.append("a report exists, the phase rule refuses report, or the ledger is not the 4,320 keys")
if check["outputs"] != ["calibration-arrays.pt", "candidate-calibration.json", "candidate-lock.json", "candidate-preregistration.md", "results.json", "stage2-measurements.pt"]:
    problems.append("the outputs are not the six post-confirm files")
if problems:
    stop("; ".join(problems))
if MODE == "--check":
    write_record()
    print("CHECK complete: every pre-report condition holds; report was NOT invoked", flush=True)
    sys.exit(0)

shutil.copyfile(state_path, HERE / "results-before-report.json")  # the exact confirm-final bytes, for the reversal check
record["results_before_report_copy_sha256"] = sha(HERE / "results-before-report.json")

# -- the one production report -----------------------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
logs: list[str] = []


def log(message: str) -> None:
    print(message, flush=True)
    logs.append(message)


runner = runner_module.Runner(log=log, model_loader=_refuse("model_loads_refused"), tokenizer_loader=_refuse("tokenizer_loads_refused"))
if runner.config is not rr.PRODUCTION or Path(runner.root).resolve() != ROOT:
    stop("the runner is not the production runner at the repository root")
record["report_started_at"] = utc_now()
status, error, trace = None, None, None
try:
    counts["report_invocations"] += 1
    status = runner.report()
except BaseException as exc:  # recorded, never retried
    error = f"{type(exc).__name__}: {exc}"
    trace = traceback.format_exc()
finally:
    record.update({"report_ended_at": utc_now(), "exit_status": status, "error": error, "traceback": trace, "logs": logs})
    try:
        after = rd.load_results_state(state_path)
        record["after"] = {"state": after["state_sha256"], "state_file": sha(state_path), "report": after.get("report"),
                           "phases": {name: entry["status"] for name, entry in after["phases"].items()},
                           "report_md_sha256": sha(out / "report.md") if (out / "report.md").exists() else None,
                           "stage2": sha(out / "stage2-measurements.pt"), "outputs": sorted(path.name for path in out.iterdir())}
    finally:
        write_record()
print(json.dumps({key: record.get(key) for key in ("report_started_at", "report_ended_at", "exit_status", "error")}), flush=True)
print(json.dumps({"counts": counts, "events": events, "after": record.get("after")}, default=str), flush=True)
sys.exit(0 if status == 0 and error is None else 1)
