"""Guarded launcher for Experiment 101's production `lock` (run exactly once; weights only; 025's lock-launcher
pattern). It runs from a scratchpad directory outside the repository and writes its own records there; they are copied
byte-identically into `experiments/101-read-component-intervention/evidence/lock-2026-10-14/` afterwards.

Before any model is loaded, it records the environment (HEAD, branch, `origin/main`, the tree state, this launcher's
sha256, the interpreter and package versions, the torch thread count, the checkpoint's revision, snapshot path and file
digests, the environment flags). Then it installs these guards, every event counted or logged:
- **`torch.nn.Module.__call__` refuses for the whole run** (the lock is weights only: the programs read weights and
  never call a module).
- **Every capture and intervention entry point refuses** (`capture.run_capture`, `interventions.run_interventions`,
  `plural_mechanism.capture_prompt` and `run_patched`, `readout_decompilation.capture_frame_020`), with every alias of
  one of them found by identity in any loaded `neural_decompiler` module or in the runner.
- **Every measurement-side computation refuses:** `ri.run_arm`, `ri.capture_reference`, `ri.measure_frame`,
  `ri.stage_runs`, the gates (`ri.frame_gates`, `ri.run_gates`), the run records, `ri.frame_values`, the statistics
  (`ri.primary_statistics`, `ri.secondary_statistics`, `ri.calibration_results`), `ri.head_delta`,
  `ri.assert_measurements`, `ul.i3_error`, `rd.NounSet.contrasts` (a measured logits read) and
  `rd.ReadoutProgram.level1_detail` (a gate). The program's own functions (`ul.pair_context`, `ul.compose_dx3`,
  `ul.contrast_of`, the readout's `blocks_3_to_5` and `contrast`) stay available: the lock recomputes the exposed
  prediction table and compares its digest with the calibration's (plan N2), on spent exposed states only.
- **The model load is counted.** Right after it, every module of the loaded model gets a refusing instance `forward`,
  and the model classes' `forward`, `run_with_hooks`, `run_with_cache` and `generate` refuse.
- **An audit hook** refuses every repository write, delete, rename, link or mkdir outside `outputs/experiment-101/`,
  and every read under `outputs/` except 020's results state and 101's own outputs.
- **Dry-run mode (`LOCK101_DRY_RUN=1`):** it loads the weights and builds the programs under every guard, takes the
  parameter and embedding digests, and recomputes the exposed table's digest; it does not call `runner.lock()`.
- **Production** requires `LOCK101_CALIBRATION=<the installed calibration's content sha256>` and an exclusive
  sentinel; it calls `runner.lock()` once.
"""
import datetime
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
OUTPUTS = (ROOT / "outputs/experiment-101").resolve()
DRY_RUN = os.environ.get("LOCK101_DRY_RUN") == "1"
LAUNCHER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
READABLE_OUTPUTS = (ROOT / "outputs/experiment-020/results.json", OUTPUTS)
os.chdir(ROOT)
events = {"load_model": 0, "modules_sealed": 0, "module_calls_refused": 0, "forward_refused": [], "model_method_refused": [], "capture_or_intervention_refused": [],
          "measurement_refused": [], "aliases_patched": [], "forbidden_reads": [], "forbidden_writes": [], "forbidden_deletes_renames_links_mkdirs": [],
          "output_writes": [], "output_renames": [], "output_mkdirs": [], "outputs_reads": [], "lock_invocations": 0}


def _path(value, dir_fd=None):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    try:
        raw = os.fsdecode(value)
        if isinstance(dir_fd, int) and not isinstance(dir_fd, bool) and not os.path.isabs(raw):
            raw = os.path.join(fcntl.fcntl(dir_fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0].decode(), raw)
        return Path(raw).resolve()
    except (OSError, ValueError):
        return None


def _inside(path: Path, root: Path) -> bool:
    return path == root or str(path).startswith(str(root) + os.sep)


def _in_repo(path: Path) -> bool:
    return _inside(path, ROOT) and not _inside(path, ROOT / ".venv")


def _hook(event, args):
    if event == "open":
        path = _path(args[0]) if args else None
        if path is None:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if _inside(path, ROOT / "outputs") and not writing:
            if not any(_inside(path, allowed) for allowed in READABLE_OUTPUTS):
                events["forbidden_reads"].append(str(path))
                raise PermissionError(f"lock may not open {path}")
            if str(path) not in events["outputs_reads"]:
                events["outputs_reads"].append(str(path))
        if not writing:
            return
        if _in_repo(path):
            if _inside(path, OUTPUTS):
                events["output_writes"].append(path.name)
                return
            events["forbidden_writes"].append(str(path))
            raise PermissionError(f"lock may not write {path}")
        return
    if event in ("os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir", "os.chmod"):
        # per event (Python 3.12 audit signatures): os.mkdir and os.chmod (path, mode, dir_fd); os.remove, os.rmdir, shutil.rmtree (path, dir_fd);
        # os.truncate (path, length) has none. A mode or a length is never taken for a descriptor (freeze review N1).
        dir_fd = args[2] if event in ("os.mkdir", "os.chmod") and len(args) > 2 else args[1] if event in ("os.remove", "os.rmdir", "shutil.rmtree") and len(args) > 1 else None
        path = _path(args[0], dir_fd) if args else None
        if path is None or not _in_repo(path):
            return
        if _inside(path, OUTPUTS):
            events["output_mkdirs" if event == "os.mkdir" else "output_writes"].append(f"{event} {path.name}")
            return
        events["forbidden_deletes_renames_links_mkdirs"].append(f"{event} {path}")
        raise PermissionError(f"lock may not {event} {path}")
    if event in ("os.rename", "os.link", "os.symlink"):
        if event == "os.symlink":
            candidates = (_path(args[1], args[2] if len(args) > 2 else None),)
        else:
            candidates = (_path(args[0], args[2] if len(args) > 2 else None), _path(args[1], args[3] if len(args) > 3 else None))
        paths = [p for p in candidates if p is not None]
        if not any(_in_repo(p) for p in paths):
            return
        if all(_inside(p, OUTPUTS) for p in paths):
            events["output_renames"].append(f"{event} {' -> '.join(p.name for p in paths)}")
            return
        events["forbidden_deletes_renames_links_mkdirs"].append(f"{event} {' -> '.join(str(p) for p in paths)}")
        raise PermissionError(f"lock may not {event} {paths}")


sys.addaudithook(_hook)

import numpy  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
import huggingface_hub  # noqa: E402
from huggingface_hub import constants as hf_constants  # noqa: E402

from neural_decompiler import capture, interventions, models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


SNAPSHOT = Path(hf_constants.HF_HUB_CACHE) / "models--EleutherAI--pythia-70m-deduped" / "snapshots" / models.PYTHIA_70M.revision
ENV_FLAGS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM",
             "CUDA_VISIBLE_DEVICES", "VIRTUAL_ENV", "NEURAL_DECOMPILER_CURRENT_EXPERIMENT", "LOCK101_DRY_RUN")
environment = {
    "head": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "origin_main": _git("rev-parse", "origin/main"),
    "commits_ahead_of_origin_main": int(_git("rev-list", "--count", "origin/main..HEAD")), "tree_porcelain": _git("status", "--porcelain", "--untracked-files=all"),
    "repository": str(ROOT), "launcher_path": str(Path(__file__).resolve()), "launcher_sha256": LAUNCHER_SHA256, "dry_run": DRY_RUN,
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__,
    "transformer_lens": importlib.metadata.version("transformer-lens"), "huggingface_hub": huggingface_hub.__version__, "numpy": numpy.__version__,
    "torch_num_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "model": {"model_id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision, "device": models.PYTHIA_70M.device, "dtype": models.PYTHIA_70M.dtype},
    "checkpoint": {"snapshot": str(SNAPSHOT), "files": {p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size} for p in sorted(SNAPSHOT.iterdir()) if p.is_file()}},
    "env_flags": {name: os.environ.get(name) for name in ENV_FLAGS},
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}


def _refusal(kind: str, name: str):
    def refuse(*args, **kwargs):
        events[kind].append(name)
        raise RuntimeError(f"{name} refused during lock")

    return refuse


def _refuse_module_call(self, *args, **kwargs):
    events["module_calls_refused"] += 1
    raise RuntimeError("a torch module was called during lock")


torch.nn.Module.__call__ = _refuse_module_call  # for the whole run
ENTRY_POINTS = {id(capture.run_capture): ("capture.run_capture", capture.run_capture),
                id(interventions.run_interventions): ("interventions.run_interventions", interventions.run_interventions),
                id(pm.capture_prompt): ("pm.capture_prompt", pm.capture_prompt), id(pm.run_patched): ("pm.run_patched", pm.run_patched),
                id(rd.capture_frame_020): ("rd.capture_frame_020", rd.capture_frame_020)}
REFUSALS = {key: _refusal("capture_or_intervention_refused", name) for key, (name, _) in ENTRY_POINTS.items()}


def _sweep_aliases() -> None:
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name.startswith("neural_decompiler") or module_name.startswith("experiment_101")):
            continue
        for attribute, value in list(vars(module).items()):
            key = id(value)
            if key in ENTRY_POINTS and value is ENTRY_POINTS[key][1]:
                setattr(module, attribute, REFUSALS[key])
                events["aliases_patched"].append(f"{module_name}.{attribute}")


_sweep_aliases()
for name in ("run_arm", "capture_reference", "measure_frame", "stage_runs", "frame_gates", "run_gates", "run_records", "frame_values", "primary_statistics",
             "secondary_statistics", "calibration_results", "head_delta", "assert_measurements"):
    setattr(ri, name, _refusal("measurement_refused", f"ri.{name}"))
ul.i3_error = _refusal("measurement_refused", "ul.i3_error")
rd.ReadoutProgram.level1_detail = _refusal("measurement_refused", "rd.ReadoutProgram.level1_detail")
rd.NounSet.contrasts = _refusal("measurement_refused", "rd.NounSet.contrasts")

original_load = models.load_model


def _counted_load(spec, *args, **kwargs):
    events["load_model"] += 1
    model = original_load(spec, *args, **kwargs)
    for module_name, module in model.named_modules():
        object.__setattr__(module, "forward", _refusal("forward_refused", f"{module_name or type(module).__name__}.forward"))
        events["modules_sealed"] += 1
    for cls in {type(module) for module in model.modules()}:
        for method in ("forward", "run_with_hooks", "run_with_cache", "generate"):
            if method in cls.__dict__ and callable(cls.__dict__[method]):
                setattr(cls, method, _refusal("model_method_refused", f"{cls.__qualname__}.{method}"))
    return model


models.load_model = _counted_load

spec = importlib.util.spec_from_file_location("experiment_101_runner", ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
assert runner_module.load_model is _counted_load  # the runner's default model loader is the counted, sealing one
_sweep_aliases()  # the runner's namespace too

logs: list[str] = []


def log(message: str) -> None:
    print(message, flush=True)
    logs.append(message)


runner = runner_module.Runner(log=log)
assert runner.config is ri.PRODUCTION
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
status = None
error = None
dry: dict = {}
preflight: dict = {}
sentinel = HERE / "lock.sentinel"
try:
    preflight["tree_clean"] = environment["tree_porcelain"] == ""
    calibration_path = ROOT / ri.CALIBRATION_RELATIVE_PATH
    preflight["calibration_committed"] = _git("ls-files", "--error-unmatch", ri.CALIBRATION_RELATIVE_PATH) == ri.CALIBRATION_RELATIVE_PATH
    calibration = json.loads(calibration_path.read_text())
    preflight["no_lock_installed"] = not (ROOT / ri.LOCK_RELATIVE_PATH).exists() and not (ROOT / ri.PREREGISTRATION_RELATIVE_PATH).exists()
    state = rd.load_results_state(OUTPUTS / "results.json")
    preflight["state_lock_not_started"] = state["phases"]["lock"] == {"status": "not_started"} and state["phases"]["calibrate"]["status"] == "complete"
    preflight["state_calibration_is_installed"] = state["calibration"]["content_sha256"] == calibration["content_sha256"]
    if not all(preflight.values()):
        raise RuntimeError(f"preflight failed: {sorted(k for k, v in preflight.items() if not v)}")
    if DRY_RUN:
        base = runner._base()
        model = models.load_model(models.PYTHIA_70M)
        progs = ul.ModelPrograms.from_model(model, base.inputs)
        noun_keys = runner._check_nouns(progs, base.inputs)
        geometry = ri.read_geometry(base.inputs.lock_011)
        exposed = runner._exposed_frames(base.inputs)
        locked = base.inputs.closure["exploration"]["locked_states"]
        states = {frame.frame_id: rd.state_from_locked(locked[frame.frame_id], frame) for frame in exposed}
        table = ri.prediction_table(progs, exposed, states, geometry, ri.PRODUCTION, noun_keys, fractions_for_weak=True)
        dry = {"parameters_sha256": rr.parameters_digest(model), "embedding_sha256": rr.embedding_digest(progs.weights.W_E), "scorable_nouns": len(progs.scorable),
               "exposed_table_sha256": table["content_sha256"], "exposed_table_equals_calibration": table["content_sha256"] == calibration["table_content_sha256"]}
        status = 0 if dry["exposed_table_equals_calibration"] else 1
    else:
        if os.environ.get("LOCK101_CALIBRATION") != calibration["content_sha256"]:
            raise RuntimeError("the production token does not name the installed calibration")
        fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, LAUNCHER_SHA256.encode())
        os.close(fd)
        events["lock_invocations"] += 1
        status = runner.lock()  # the one production lock
except BaseException as exc:  # recorded, never retried
    error = f"{type(exc).__name__}: {exc}"
ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
outputs = {}
if OUTPUTS.exists():
    for path in sorted(OUTPUTS.iterdir()):
        if path.is_file():
            outputs[path.name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
record = {"dry_run": DRY_RUN, "started_at": started, "ended_at": ended, "exit_status": status, "error": error, "lock_invocations": events["lock_invocations"],
          "environment": environment, "preflight": preflight, "events": events, "dry_run_digests": dry, "outputs": outputs, "logs": logs,
          "config": runner.config.to_json()["name"]}
(HERE / ("dryrun_run.json" if DRY_RUN else "lock_run.json")).write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({key: record[key] for key in ("dry_run", "started_at", "ended_at", "exit_status", "error", "lock_invocations")}), flush=True)
print(json.dumps({key: value for key, value in events.items() if key not in ("output_writes", "aliases_patched", "outputs_reads")}), flush=True)
print(json.dumps({"preflight": preflight, "dry_run_digests": dry, "outputs": outputs}), flush=True)
sys.exit(0 if status == 0 and error is None else 1)
