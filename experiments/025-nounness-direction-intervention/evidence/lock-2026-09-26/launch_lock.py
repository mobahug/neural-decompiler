"""Guarded launcher for Experiment 025's production `lock` (run exactly once; weights only).

Before any model is loaded, it records the environment: the full HEAD, the tree state, the repository path, this
launcher's sha256, the Python, torch, transformers, transformer-lens, huggingface-hub and numpy versions, the torch
thread count, the checkpoint's revision, snapshot path and file digests, the tokenizer revision and the relevant
environment flags.

Then it installs these guards, every event counted or logged:
- **`torch.nn.Module.__call__` refuses for the whole run.** 024's lock saw zero module calls while the weights were
  loaded and the programs built.
- **Every capture and intervention entry point refuses:** `capture.run_capture`, `interventions.run_interventions`,
  `plural_mechanism.capture_prompt` and `plural_mechanism.run_patched`, and every alias of one of them found by
  identity in any loaded `neural_decompiler` module or in the runner.
- **Every confirm-only computation refuses:** the patched measurement, stage 2, the patch-path check, the gates, `C`,
  `D_attn`, ℓ, A/B/G, the descriptives, `ul.measure_prompt`, `ul.contrast_of`, `NounSet.contrasts`, and
  `ReadoutProgram.level1_detail`.
- **The model load is counted.** Right after it, every module of the loaded model gets a refusing instance `forward`,
  which blocks direct `.forward()` routes, and the model class's `forward`, `run_with_hooks`, `run_with_cache` and
  `generate` refuse.
- **An audit hook** refuses:
  - reads of 022's calibration table and of `outputs/experiment-023/` and `outputs/experiment-024/`;
  - any repository write, delete, rename, link or mkdir outside `outputs/experiment-025/`.

  Writes outside the repository are logged.
- **In dry-run mode (`LOCK025_DRY_RUN=1`):**
  - it loads the weights and builds the programs under every guard, and takes the parameter and embedding digests;
  - it refuses every 025 geometry function (no 025 cue geometry exists before the lock);
  - it does not call `runner.lock()`.

The records go to this scratchpad directory only.
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
OUTPUTS = (ROOT / "outputs/experiment-025").resolve()
DRY_RUN = os.environ.get("LOCK025_DRY_RUN") == "1"
LAUNCHER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
events = {"load_model": 0, "modules_sealed": 0, "module_calls_refused": 0, "forward_refused": [], "model_method_refused": [], "capture_or_intervention_refused": [],
          "confirm_only_refused": [], "geometry_refused_in_dry_run": [], "aliases_patched": [], "forbidden_reads": [], "forbidden_writes": [],
          "forbidden_deletes_renames_links_mkdirs": [], "output_writes": [], "output_renames": [], "output_mkdirs": [], "outside_repo_writes": [], "lock_invocations": 0}


def _path(value, dir_fd=None):
    """The absolute path an audit event names: a relative name with a ``dir_fd`` is resolved against that directory
    (macOS ``F_GETPATH``), not the working directory. The first dry run showed why: filelock's import-time probe removes
    ``probe-source`` inside a temporary directory by ``dir_fd``."""
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    try:
        raw = os.fsdecode(value)
        if isinstance(dir_fd, int) and not os.path.isabs(raw):
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
        text = str(path)
        if text.endswith("calibration-table.pt") or "/outputs/experiment-023/" in text or "/outputs/experiment-024/" in text:
            events["forbidden_reads"].append(text)
            raise PermissionError(f"lock may not open {text}")
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if not writing:
            return
        if _in_repo(path):
            if _inside(path, OUTPUTS):
                events["output_writes"].append(path.name)
                return
            events["forbidden_writes"].append(text)
            raise PermissionError(f"lock may not write {text}")
        if not _inside(path, HERE):
            events["outside_repo_writes"].append(text)
        return
    if event in ("os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir"):
        # os.remove, os.rmdir, shutil.rmtree: (path, dir_fd); os.mkdir: (path, mode, dir_fd); os.truncate: (path, length)
        dir_fd = args[2] if event == "os.mkdir" and len(args) > 2 else args[1] if event in ("os.remove", "os.rmdir", "shutil.rmtree") and len(args) > 1 else None
        path = _path(args[0], dir_fd) if args else None
        if path is None or not _in_repo(path):
            return
        if _inside(path, OUTPUTS):
            events["output_mkdirs" if event == "os.mkdir" else "output_writes"].append(f"{event} {path.name}")
            return
        events["forbidden_deletes_renames_links_mkdirs"].append(f"{event} {path}")
        raise PermissionError(f"lock may not {event} {path}")
    if event in ("os.rename", "os.link", "os.symlink"):
        # os.rename, os.link: (src, dst, src_dir_fd, dst_dir_fd); os.symlink: (src, dst, dir_fd), src being the link's text
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
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
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
             "CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK", "HF_HOME", "HF_HUB_CACHE", "VIRTUAL_ENV", "NEURAL_DECOMPILER_CURRENT_EXPERIMENT",
             "NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE", "LOCK025_DRY_RUN")
environment = {
    "head": _git("rev-parse", "HEAD"), "origin_main": _git("rev-parse", "origin/main"), "tree_porcelain": _git("status", "--porcelain", "--untracked-files=all"),
    "repository": str(ROOT), "launcher_path": str(Path(__file__).resolve()), "launcher_sha256": LAUNCHER_SHA256, "dry_run": DRY_RUN,
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__,
    "transformer_lens": importlib.metadata.version("transformer-lens"), "huggingface_hub": huggingface_hub.__version__, "numpy": numpy.__version__,
    "torch_num_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "model": {"model_id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision, "device": models.PYTHIA_70M.device, "dtype": models.PYTHIA_70M.dtype},
    "tokenizer_revision": models.PYTHIA_70M.revision,
    "checkpoint": {"snapshot": str(SNAPSHOT), "files": {path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size} for path in sorted(SNAPSHOT.iterdir()) if path.is_file()}},
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

ENTRY_POINTS = {id(capture.run_capture): ("capture.run_capture", capture.run_capture), id(interventions.run_interventions): ("interventions.run_interventions", interventions.run_interventions),
                id(pm.capture_prompt): ("pm.capture_prompt", pm.capture_prompt), id(pm.run_patched): ("pm.run_patched", pm.run_patched)}
REFUSALS = {key: _refusal("capture_or_intervention_refused", name) for key, (name, _) in ENTRY_POINTS.items()}


def _sweep_aliases() -> None:
    """Every module attribute that *is* an entry point (the definition or any alias or copy) is replaced by its refusal."""
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name.startswith("neural_decompiler") or module_name.startswith("experiment_025")):
            continue
        for attribute, value in list(vars(module).items()):
            key = id(value)
            if key in ENTRY_POINTS and value is ENTRY_POINTS[key][1]:
                setattr(module, attribute, REFUSALS[key])
                events["aliases_patched"].append(f"{module_name}.{attribute}")


_sweep_aliases()
for name in ("measure_rotated", "stage_two", "patch_path_check", "identity_gates", "level1_gates", "run_gates", "recompute_c", "routing_distance", "per_cue", "statistics",
             "descriptives", "ladder", "vector_factors", "lexicon_of", "assert_measurements", "enforce_gates"):
    setattr(cr, name, _refusal("confirm_only_refused", f"cr.{name}"))
for name in ("measure_prompt", "contrast_of", "compose_dx3", "i3_error"):
    setattr(ul, name, _refusal("confirm_only_refused", f"ul.{name}"))
for name in ("cue_mse", "fresh_pair_cells", "cue_mse_torch"):
    setattr(rr, name, _refusal("confirm_only_refused", f"rr.{name}"))
rd.ReadoutProgram.level1_detail = _refusal("confirm_only_refused", "rd.ReadoutProgram.level1_detail")
rd.NounSet.contrasts = _refusal("confirm_only_refused", "rd.NounSet.contrasts")
if DRY_RUN:
    for name in ("geometry_block", "cue_vectors", "cue_geometry", "direction", "centroids", "plurality_direction", "control_directions", "nearest_tokens", "condition_vectors",
                 "geometry_checks", "score"):
        setattr(cr, name, _refusal("geometry_refused_in_dry_run", f"cr.{name}"))

original_load = models.load_model


def _counted_load(spec, *args, **kwargs):
    events["load_model"] += 1
    model = original_load(spec, *args, **kwargs)
    for module_name, module in model.named_modules():  # direct .forward() routes refuse from here on
        object.__setattr__(module, "forward", _refusal("forward_refused", f"{module_name or type(module).__name__}.forward"))
        events["modules_sealed"] += 1
    for method in ("forward", "run_with_hooks", "run_with_cache", "generate"):
        if hasattr(type(model), method):
            setattr(type(model), method, _refusal("model_method_refused", f"{type(model).__name__}.{method}"))
    return model


models.load_model = _counted_load

spec = importlib.util.spec_from_file_location("experiment_025_runner", ROOT / "experiments/025-nounness-direction-intervention/run.py")
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
assert runner.config is cr.PRODUCTION
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
status = None
error = None
dry: dict = {}
try:
    if DRY_RUN:
        base = runner._base()
        model = models.load_model(models.PYTHIA_70M)
        progs = ul.ModelPrograms.from_model(model, base.inputs)
        dry = {"parameters_sha256": rr.parameters_digest(model), "embedding_sha256": rr.embedding_digest(progs.weights.W_E), "scorable_nouns": len(progs.scorable)}
        status = 0
    else:
        events["lock_invocations"] += 1
        status = runner.lock()  # the one production lock
except BaseException as exc:  # recorded, never retried
    error = f"{type(exc).__name__}: {exc}"
ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
outputs = {}
if OUTPUTS.exists():
    for path in sorted(OUTPUTS.iterdir()):
        outputs[path.name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
record = {"dry_run": DRY_RUN, "started_at": started, "ended_at": ended, "exit_status": status, "error": error, "lock_invocations": events["lock_invocations"],
          "environment": environment, "events": events, "dry_run_digests": dry, "outputs": outputs, "logs": logs, "config": runner.config.to_json()["name"]}
(HERE / ("dryrun_run.json" if DRY_RUN else "lock_run.json")).write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({key: record[key] for key in ("dry_run", "started_at", "ended_at", "exit_status", "error", "lock_invocations")}), flush=True)
print(json.dumps({key: value for key, value in events.items() if key not in ("output_writes", "aliases_patched")}), flush=True)
print(json.dumps({"aliases_patched": events["aliases_patched"], "outputs": outputs, "dry_run_digests": dry}), flush=True)
sys.exit(0 if status == 0 and error is None else 1)
