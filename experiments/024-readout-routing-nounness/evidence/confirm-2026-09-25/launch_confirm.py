"""Guarded launcher for Experiment 024's production `confirm`: run exactly once, never retried.

Usage: ``launch_confirm.py --dry-run`` (every pre-confirm step, then stop without invoking confirm or loading the model)
or ``launch_confirm.py --confirm-once`` (the one production invocation).

Order of the production invocation:
1. A sentinel file is created exclusively before anything else. If it exists, the launcher refuses, so it can never run
   confirm twice.
2. A record-only audit hook is installed. It never raises, so it can never interrupt the run. It records repository
   writes outside `outputs/experiment-024/`, the writes inside it, and any open of Experiment 022's local calibration
   table or Experiment 023's local outputs (confirm needs neither).
3. **The direct module-blob assertion, before the model loads** (the reviewer's decision on lock-review note 1).
   `rr.own_blob()`, the installed lock's `module.blob`, `git hash-object` of the imported file, git's HEAD blob and an
   own sha1 must all equal `e6cb3776…`, and `rr` must be imported from the repository file. A mismatch stops before the
   runner is built, the model loads, the ledger is written or any prompt runs.
4. The environment:
   - HEAD `af160ce` and a clean tree;
   - the preserved hashes;
   - 4 torch threads, `HF_HUB_OFFLINE=1` and the repository's `.venv`;
   - the pinned revision's cached checkpoint, sha256 `3da38833…`.

   Anything else also stops before the runner is built.
5. Prompt accounting. Every `plural_mechanism.capture_prompt` call is logged to `prompts.jsonl` (its key, before the
   call) and then delegated unchanged. `run_patched`, `run_capture`, `run_interventions` and `load_model` are counted
   and delegated.
6. The stock runner (`Runner`, production configuration) and one `runner.confirm()`. Its outcome — status, exception
   and traceback — is recorded and never retried.
"""
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) == 2 else ""
if MODE not in ("--dry-run", "--confirm-once"):
    raise SystemExit("usage: launch_confirm.py --dry-run | --confirm-once")
DRY = MODE == "--dry-run"
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
PREFIX = "dry_run" if DRY else "confirm"
SENTINEL = HERE / ("DRY_RUN_LAUNCHED" if DRY else "CONFIRM_LAUNCHED")
RECORD = HERE / f"{PREFIX}_record.json"
PROMPT_LOG = HERE / f"{PREFIX}_prompts.jsonl"
COMMIT = "af160cee0389bcdaf12bcbd92a8616f142777a52"
EXPECTED_BLOB = "e6cb37767d1d06c6ff40804a88eab569723afdb5"
EXPECTED = {
    "lock_file": "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d",
    "lock_content": "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739",
    "prereg_file": "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54",
    "state": "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d",
    "state_file": "14459cdde6ec2d832c3965d5b0ebaadaf06cef6aa0b0460c3a31b29dd442b262",
    "checkpoint": "3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd",
}
GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# -- 1. the sentinel: exclusive, before anything else -------------------------------------------------------------------
try:
    descriptor = os.open(SENTINEL, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
except FileExistsError:
    print(f"REFUSED: {SENTINEL} exists; this launcher has already run in this mode and never runs again", flush=True)
    sys.exit(4)
os.write(descriptor, f"{MODE} {utc_now()} pid {os.getpid()}\n".encode())
os.close(descriptor)

# -- 2. the record-only audit hook (never raises) -----------------------------------------------------------------------
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
            events["relative_opens"] += 1  # dir_fd-relative probes (filelock in the system temp directory)
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
    except Exception:  # noqa: BLE001 — the hook may never interrupt the run
        events["hook_errors"] += 1


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

record = {"mode": MODE, "launcher_started_at": utc_now(), "pid": os.getpid(), "commit_expected": COMMIT, "sentinel": str(SENTINEL)}
counts = {"confirm_invocations": 0, "load_model": 0, "capture_prompt": 0, "run_patched": 0, "run_capture": 0, "run_interventions": 0, "prompt_log_errors": 0}


def write_record() -> None:
    record.update({"counts": counts, "events": events, "record_written_at": utc_now()})
    RECORD.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")


def stop(reason: str) -> None:
    record["stopped_before_confirm"] = reason
    write_record()
    print(f"STOPPED before confirm (no runner built, no model loaded, no ledger, no prompt): {reason}", flush=True)
    sys.exit(3)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, env=GIT_ENV).stdout.strip()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- 3. the direct module-blob assertion, before the model loads --------------------------------------------------------
lock_path = ROOT / rr.LOCK_RELATIVE_PATH
installed_lock = json.loads(lock_path.read_text(encoding="utf-8"))
module_path = Path(rr.__file__).resolve()
spec_origin = Path(importlib.util.find_spec("neural_decompiler.readout_routing").origin).resolve()
module_bytes = module_path.read_bytes()
assertion = {
    "expected_lock_blob": installed_lock["module"]["blob"],
    "expected_constant": EXPECTED_BLOB,
    "observed_rr_own_blob": rr.own_blob(),
    "git_hash_object_of_imported_file": git("hash-object", "--", str(module_path)),
    "git_head_blob": git("rev-parse", "HEAD:src/neural_decompiler/readout_routing.py"),
    "own_sha1_of_imported_file": hashlib.sha1(b"blob %d\0" % len(module_bytes) + module_bytes).hexdigest(),
    "imported_path": str(module_path),
    "import_spec_origin": str(spec_origin),
    "repository_path": str((ROOT / "src/neural_decompiler/readout_routing.py").resolve()),
    "lock_module_path": installed_lock["module"]["path"],
    "asserted_at": utc_now(),
}
blobs = {assertion[key] for key in ("expected_lock_blob", "expected_constant", "observed_rr_own_blob", "git_hash_object_of_imported_file", "git_head_blob",
                                    "own_sha1_of_imported_file")}
assertion["agree"] = blobs == {EXPECTED_BLOB} and assertion["imported_path"] == assertion["import_spec_origin"] == assertion["repository_path"] \
    and assertion["lock_module_path"] == "src/neural_decompiler/readout_routing.py"
record["module_blob_assertion"] = assertion
print(f"module-blob assertion: {'AGREE' if assertion['agree'] else 'MISMATCH'} {json.dumps(assertion)}", flush=True)
if not assertion["agree"]:
    stop("the direct module-blob assertion failed")

# -- 4. the environment, the repository and the preserved hashes -------------------------------------------------------
checkpoint = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots" / models.PYTHIA_70M.revision / "model.safetensors"
state_path = ROOT / "outputs/experiment-024/results.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
environment = {
    "python": platform.python_version(), "platform": platform.platform(), "executable": sys.executable, "prefix": sys.prefix,
    "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "transformer-lens", "huggingface-hub", "numpy", "safetensors", "tokenizers")},
    "torch_num_threads": torch.get_num_threads(),
    "env": {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "PYTHONDONTWRITEBYTECODE", "GIT_OPTIONAL_LOCKS")},
    "model": {"id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision},
    "checkpoint": {"path": str(checkpoint), "blob": checkpoint.resolve().name if checkpoint.exists() else None,
                   "sha256": sha(checkpoint.read_bytes()) if checkpoint.exists() else None},
    "git": {"head": git("rev-parse", "HEAD"), "status_porcelain": git("status", "--porcelain", "--untracked-files=all")},
    "hashes": {"lock_file": sha(lock_path.read_bytes()), "lock_content": installed_lock["content_sha256"],
               "prereg_file": sha((ROOT / rr.PREREGISTRATION_RELATIVE_PATH).read_bytes()), "state": state["state_sha256"], "state_file": sha(state_path.read_bytes())},
    "state_before": {"phases": {name: entry["status"] for name, entry in state["phases"].items()}, "ledger": len(state["executed_prompt_keys"]),
                     "noun_ledger": len(state["executed_noun_keys"]), "stage2_exists": (ROOT / "outputs/experiment-024/stage2-measurements.pt").exists()},
}
record["environment"] = environment
problems = []
if environment["torch_num_threads"] != 4:
    problems.append("torch threads are not 4")
if environment["env"]["HF_HUB_OFFLINE"] != "1" or environment["env"]["PYTHONDONTWRITEBYTECODE"] != "1":
    problems.append("HF_HUB_OFFLINE=1 and PYTHONDONTWRITEBYTECODE=1 are required")
if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    problems.append("the interpreter is not the repository's .venv")
if environment["checkpoint"]["sha256"] != EXPECTED["checkpoint"] or environment["checkpoint"]["blob"] != EXPECTED["checkpoint"]:
    problems.append("the cached checkpoint is not the pinned one")
if environment["git"]["head"] != COMMIT or environment["git"]["status_porcelain"] != "":
    problems.append("HEAD is not af160ce or the tree is not clean")
if environment["hashes"]["lock_file"] != EXPECTED["lock_file"] or environment["hashes"]["lock_content"] != EXPECTED["lock_content"] \
        or rc.content_digest(installed_lock) != EXPECTED["lock_content"] or environment["hashes"]["prereg_file"] != EXPECTED["prereg_file"]:
    problems.append("the installed lock or preregistration hashes differ")
if environment["hashes"]["state"] != EXPECTED["state"] or environment["hashes"]["state_file"] != EXPECTED["state_file"]:
    problems.append("the results state differs from 58891c24… / 14459cdd…")
if environment["state_before"] != {"phases": {"calibrate": "complete", "lock": "complete", "confirm": "not_started", "report": "not_started"}, "ledger": 0,
                                   "noun_ledger": 0, "stage2_exists": False}:
    problems.append("the phase state, the ledgers or the stage-2 file are not the pre-confirm ones")
print(f"environment: {json.dumps(environment, default=str)}", flush=True)
if problems:
    stop("; ".join(problems))

# -- 5. prompt accounting (logged, then delegated unchanged) ------------------------------------------------------------
original = {name: getattr(pm, name) for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions") if hasattr(pm, name)}
prompt_log = PROMPT_LOG.open("x", encoding="utf-8")


def _capture_prompt(*args, **kwargs):
    counts["capture_prompt"] += 1
    try:
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt")
        prompt_log.write(json.dumps({"n": counts["capture_prompt"], "key": getattr(prompt, "key", None), "t": time.time()}) + "\n")
        prompt_log.flush()
    except Exception:  # noqa: BLE001 — accounting may never interrupt the run
        counts["prompt_log_errors"] += 1
    return original["capture_prompt"](*args, **kwargs)


def _counted(name):
    def wrapper(*args, **kwargs):
        counts[name] += 1
        return original[name](*args, **kwargs)
    return wrapper


pm.capture_prompt = _capture_prompt
for name in ("run_patched", "run_capture", "run_interventions"):
    if name in original:
        setattr(pm, name, _counted(name))


def _counted_load(spec):
    counts["load_model"] += 1
    return models.load_model(spec)


# -- 6. the stock runner, one confirm ----------------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
logs: list[str] = []


def log(message: str) -> None:
    print(message, flush=True)
    logs.append(message)


runner = runner_module.Runner(log=log, model_loader=_counted_load)
if runner.config is not rr.PRODUCTION or Path(runner.root).resolve() != ROOT:
    stop("the runner is not the production runner at the repository root")
record["runner"] = {"config": runner.config.to_json()["name"], "root": str(runner.root), "results_path": str(runner.results_path),
                    "stage2_path": str(runner.stage2_path)}

if DRY:
    # exercise the accounting path with a stub (never the real entry point), then restore it; no confirm, no model
    real = original["capture_prompt"]
    original["capture_prompt"] = lambda *args, **kwargs: "stub"
    class _P:  # noqa: E306
        key = "dry-run|probe|0"
    probe = pm.capture_prompt(None, _P(), ())
    original["capture_prompt"] = real
    counts["capture_prompt"] = 0
    record["dry_run"] = {"probe_returned": probe, "prompt_log_lines": len(PROMPT_LOG.read_text(encoding="utf-8").splitlines()) if PROMPT_LOG.exists() else 0,
                         "wrappers_installed": {name: getattr(pm, name) is not original.get(name) for name in original}}
    prompt_log.close()
    write_record()
    print(f"DRY RUN complete: every pre-confirm step passed; confirm was NOT invoked and no model was loaded. {json.dumps(record['dry_run'])}", flush=True)
    sys.exit(0)

record["confirm_started_at"] = utc_now()
print(f"confirm: the one production invocation begins at {record['confirm_started_at']}", flush=True)
status, error, trace = None, None, None
try:
    counts["confirm_invocations"] += 1
    status = runner.confirm()
except BaseException as exc:  # recorded, never retried; the runner has recorded any incident in its own state
    error = f"{type(exc).__name__}: {exc}"
    trace = traceback.format_exc()
finally:
    record.update({"confirm_ended_at": utc_now(), "exit_status": status, "error": error, "traceback": trace, "log_lines": len(logs)})
    try:
        prompt_log.close()
    finally:
        write_record()
print(json.dumps({key: record[key] for key in ("confirm_started_at", "confirm_ended_at", "exit_status", "error")}), flush=True)
print(json.dumps({"counts": counts, "events": {key: (value if key != "output_writes" else value) for key, value in events.items()}}), flush=True)
sys.exit(0 if status == 0 and error is None else 1)
