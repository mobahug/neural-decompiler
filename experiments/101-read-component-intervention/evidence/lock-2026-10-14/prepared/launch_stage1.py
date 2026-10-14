"""Guarded launcher for Experiment 101's `stage1` (plan § The prediction freeze and § One-shot protections; 025's
confirm-launcher v3 pattern, with the freeze review's N1 applied to the audit hook). It runs from a scratchpad directory
outside the repository and writes its own records there; they are copied byte-identically into the experiment's
`evidence/stage1-<date>/` afterwards. The mode must be given explicitly (`STAGE1101_MODE`); anything else refuses.

PREPARED, NOT YET EXERCISED: it can run only after the user has installed and committed the lock and the
preregistration (design §9: "The user commits the lock"). Version 2: the lock review's S1–S3 and N8 applied (the
hook fails closed and its decisions are self-tested in every mode; the sentinel follows the preflight; records are never
overwritten; `-B` required; the post-run checks assert the ledger and the tree). Version 2 has not been re-reviewed.

**`rehearsal`** — never calls `stage1`, allows no forward, writes nothing in the repository. It runs the preflight:
- N1: the repository root; HEAD, branch and `origin/main` recorded; a clean tree; the runner's own git state;
- N2: the environment against the lock run's record;
- the installed lock and preregistration (both hashes; the preregistration re-rendered from the lock), the module
  binding (every pinned blob; `ri`'s own blob equal to the lock's and to the git blob at HEAD), the full
  `ri.validate_lock` through the runner (`_validated_lock`), stage1 eligibility (`runner._state_for("stage1")`), the
  020 runtime, the empty confirmation ledger, the absence of the three stage-1 files, the 36 S1 keys;
- the stage-1 write path under the hook: `hook_self_test` decides, under the production rules, the event tuples Python
  3.12 raises for every stage-1 write (the lock review's recording), with refused controls (a sibling file, another
  experiment file, a source file, relative names at the root, an unresolvable descriptor); nothing is created.

**`production`** — the one stage 1. It requires `STAGE1101_LOCK=<the installed lock's content sha256>` and an exclusive
sentinel. After the same preflight:
- **N3.** A forward runs only inside `ri.capture_reference`, wrapped by identity, allowed only for one of the 36 S1 keys
  `frame|ref|581|S1` of the frozen confirmation, each once, and only once the S1 keys are in the ledger on disk.
  `ri.run_arm` refuses outright (no arm runs in stage 1). The guarded routes are those of the calibrate launcher.
- **N4.** The audit hook allows repository writes only to `outputs/experiment-101/results.json` (through its
  `.results-*.json` temporary files and one rename each), to exactly the three stage-1 files, and the `mkdir` of
  `stage1/` (and of `outputs/experiment-101/`). The runner's `write_check` is the launcher's predicate.
- After the phase: the three files exist, their sha256s equal the stage-1 record's, and the record's digest
  reproduces. The launcher prints the paths for the user to commit (the prediction freeze); it commits nothing.
"""
import datetime
import fcntl
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
OUTPUTS = (ROOT / "outputs/experiment-101").resolve()
EXPERIMENT_DIR = (ROOT / "experiments/101-read-component-intervention").resolve()
STAGE1_DIR = EXPERIMENT_DIR / "stage1"
STAGE1_FILES = {STAGE1_DIR / "states.f64", STAGE1_DIR / "states.json", STAGE1_DIR / "table.json"}
MODE = os.environ.get("STAGE1101_MODE", "")
if MODE not in ("rehearsal", "production"):
    raise SystemExit("set STAGE1101_MODE to 'rehearsal' or 'production'")
LAUNCHER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LOCK_RUN_RECORD = EXPERIMENT_DIR / "evidence/lock-2026-10-14/lock_run.json"

os.chdir(ROOT)
if Path.cwd().resolve() != ROOT:
    raise SystemExit(f"cwd {Path.cwd()} is not the repository root {ROOT}")

events = {"mode": MODE, "load_model": 0, "refused": [], "forbidden_reads": [], "forbidden_writes": [], "output_writes": [], "output_renames": [], "stage1_writes": [],
          "stage1_mkdirs": 0, "outputs_reads": [], "sanctioned_runs": 0, "sanctioned_forward_calls": 0, "stage1_invocations": 0}
READABLE_OUTPUTS = (ROOT / "outputs/experiment-020/results.json", OUTPUTS)


UNRESOLVED = Path("/<unresolved>")


def _path(value, dir_fd=None):
    """The absolute path an audit event names, or None when the argument is not a path (an open on a descriptor). A
    relative name is resolved against its directory descriptor when one is given (``dir_fd`` >= 0; Python 3.12 passes
    -1 for none) and against the working directory otherwise. A name that cannot be resolved is UNRESOLVED, which every
    write-type check refuses: the hook fails closed (the lock review's S2)."""
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


def _temp_state(path: Path, production: bool) -> bool:
    return production and path.parent == OUTPUTS and path.name.startswith(".results-") and path.name.endswith(".json")


def write_allowed(path: Path, production: bool | None = None) -> bool:
    """The launcher's write predicate (the runner's write_check uses it too): in production, the results state and its
    temporary files and exactly the three stage-1 files; in rehearsal, nothing (the lock review's N8)."""
    production = (MODE == "production") if production is None else production
    return production and (path == OUTPUTS / "results.json" or path in STAGE1_FILES or _temp_state(path, production))


def decide(event, args, production):
    """A pure decision for one audit event: None to allow, or (kind, detail) to refuse. The real hook and the
    rehearsal's self-test share it (the lock review's S1)."""
    if event == "open":
        path = _path(args[0]) if args else None
        if path is None:
            return None
        mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if not writing:
            if path != UNRESOLVED and _inside(path, ROOT / "outputs") and not any(_inside(path, allowed) for allowed in READABLE_OUTPUTS):
                return ("forbidden_reads", str(path))
            return None
        if _in_repo(path) and not write_allowed(path, production):
            return ("forbidden_writes", str(path))
        return None
    if event in ("os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir", "os.chmod"):
        # per event (Python 3.12 audit signatures): os.mkdir and os.chmod (path, mode, dir_fd); os.remove, os.rmdir,
        # shutil.rmtree (path, dir_fd); os.truncate (path, length) has none
        dir_fd = args[2] if event in ("os.mkdir", "os.chmod") and len(args) > 2 else args[1] if event in ("os.remove", "os.rmdir", "shutil.rmtree") and len(args) > 1 else None
        path = _path(args[0], dir_fd) if args else UNRESOLVED
        if path is None or not _in_repo(path):
            return None
        if event == "os.remove" and _temp_state(path, production):
            return None
        if event == "os.mkdir" and production and path in (OUTPUTS, STAGE1_DIR):
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
        if event == "os.rename" and len(paths) == 2 and _temp_state(paths[0], production) and paths[1] == OUTPUTS / "results.json":
            return None
        return ("forbidden_writes", f"{event} {' -> '.join(str(p) for p in paths)}")
    return None


def _hook(event, args):
    decision = decide(event, args, MODE == "production")
    if event == "open" and decision is None and args:
        path = _path(args[0])
        if path is not None and path != UNRESOLVED and _inside(path, ROOT / "outputs") and str(path) not in events["outputs_reads"]:
            events["outputs_reads"].append(str(path))
        if path in STAGE1_FILES and MODE == "production":
            mode = args[1] if len(args) > 1 else None
            if isinstance(mode, str) and "w" in mode:
                events["stage1_writes"].append(path.name)
    if event == "os.mkdir" and decision is None and args and _path(args[0], args[2] if len(args) > 2 else None) == STAGE1_DIR:
        events["stage1_mkdirs"] += 1
    if decision is not None:
        kind, detail = decision
        events[kind].append(detail)
        raise PermissionError(f"stage1 launcher ({MODE}): {kind}: {detail}")


def hook_self_test() -> dict:
    """The rehearsal's exercise of the stage-1 write path under the hook (the lock review's S1): the event tuples Python
    3.12 raises for the stage-1 writes (recorded by the lock review, ``out/f_audit_events.json``), decided under the
    production rules, with refused controls; nothing is created."""
    flags_w = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0)
    tmp = OUTPUTS / ".results-selftest.json"
    allowed = {
        "mkdir stage1": ("os.mkdir", (str(STAGE1_DIR), 511, -1)),
        "mkdir outputs": ("os.mkdir", (str(OUTPUTS), 511, -1)),
        "write states.f64": ("open", (str(STAGE1_DIR / "states.f64"), "w", flags_w)),
        "write states.json": ("open", (str(STAGE1_DIR / "states.json"), "w", flags_w)),
        "write table.json": ("open", (str(STAGE1_DIR / "table.json"), "w", flags_w)),
        "state temporary": ("open", (str(tmp), None, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0))),
        "state rename": ("os.rename", (str(tmp), str(OUTPUTS / "results.json"), -1, -1)),
        "state temporary removed": ("os.remove", (str(tmp), -1)),
        "read stage1 file": ("open", (str(STAGE1_DIR / "states.f64"), "r", getattr(os, "O_CLOEXEC", 0))),
    }
    refused = {
        "sibling in stage1": ("open", (str(STAGE1_DIR / "other.json"), "w", flags_w)),
        "experiment file": ("open", (str(EXPERIMENT_DIR / "table.json"), "w", flags_w)),
        "source file": ("open", (str(ROOT / "src/neural_decompiler/read_intervention.py"), "w", flags_w)),
        "relative mkdir at the root": ("os.mkdir", ("x", 511, -1)),
        "relative remove at the root": ("os.remove", ("x", -1)),
        "relative rename at the root": ("os.rename", ("a", "b", -1, -1)),
        "rename into the experiment": ("os.rename", (str(tmp), str(EXPERIMENT_DIR / "x.json"), -1, -1)),
        "unresolvable dir_fd": ("os.remove", ("x", 10 ** 6)),
        "mkdir elsewhere in the experiment": ("os.mkdir", (str(EXPERIMENT_DIR / "other"), 511, -1)),
        "read another experiment's outputs": ("open", (str(ROOT / "outputs/experiment-025/results.json"), "r", 0)),
    }
    out = {"allowed": {name: decide(event, args, True) is None for name, (event, args) in allowed.items()},
           "refused": {name: decide(event, args, True) is not None for name, (event, args) in refused.items()},
           "rehearsal_refuses_stage1_writes": decide(*allowed["write states.f64"], False) is not None}
    out["ok"] = all(out["allowed"].values()) and all(out["refused"].values()) and out["rehearsal_refuses_stage1_writes"]
    return out
sys.addaudithook(_hook)

import numpy  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
import huggingface_hub  # noqa: E402
from huggingface_hub import constants as hf_constants  # noqa: E402

from neural_decompiler import capture, interventions, models  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler.provenance import collect_git_state  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


HEAD = _git("rev-parse", "HEAD")
SNAPSHOT = Path(hf_constants.HF_HUB_CACHE) / "models--EleutherAI--pythia-70m-deduped" / "snapshots" / models.PYTHIA_70M.revision
environment = {
    "head": HEAD, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "origin_main": _git("rev-parse", "origin/main"),
    "commits_ahead_of_origin_main": int(_git("rev-list", "--count", "origin/main..HEAD")), "repository": str(ROOT), "launcher_path": str(Path(__file__).resolve()),
    "launcher_sha256": LAUNCHER_SHA256, "mode": MODE, "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "torch": torch.__version__, "transformers": transformers.__version__, "transformer_lens": importlib.metadata.version("transformer-lens"),
    "huggingface_hub": huggingface_hub.__version__, "numpy": numpy.__version__, "torch_num_threads": torch.get_num_threads(),
    "model": {"model_id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision, "device": models.PYTHIA_70M.device, "dtype": models.PYTHIA_70M.dtype},
    "checkpoint": {"snapshot": str(SNAPSHOT), "files": {p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size} for p in sorted(SNAPSHOT.iterdir()) if p.is_file()}},
    "env_flags": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "OMP_NUM_THREADS",
                                                 "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "VIRTUAL_ENV", "STAGE1101_MODE")},
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}

inside: list[str] = []
allowed_keys: set[str] = set()
done_keys: set[str] = set()
state_flags = {"ledger_verified": False}
original_call = torch.nn.Module.__call__
run_log = open(HERE / "production_runs.jsonl", "a", buffering=1, encoding="utf-8") if MODE == "production" else None


def _deny(name: str):
    events["refused"].append(name)
    raise RuntimeError(f"{name} refused by the stage1 launcher ({MODE})")


def _guarded_call(self, *args, **kwargs):
    if not inside:
        _deny(f"torch.nn.Module.__call__ outside a sanctioned call ({type(self).__name__})")
    events["sanctioned_forward_calls"] += 1
    return original_call(self, *args, **kwargs)


torch.nn.Module.__call__ = _guarded_call
LOW_LEVEL = {id(capture.run_capture): ("capture.run_capture", capture.run_capture),
             id(interventions.run_interventions): ("interventions.run_interventions", interventions.run_interventions),
             id(pm.capture_prompt): ("pm.capture_prompt", pm.capture_prompt), id(pm.run_patched): ("pm.run_patched", pm.run_patched),
             id(rd.capture_frame_020): ("rd.capture_frame_020", rd.capture_frame_020)}


def _low_level(name, original):
    @functools.wraps(original)
    def guarded(*args, **kwargs):
        if not inside:
            _deny(f"{name} outside a sanctioned call")
        return original(*args, **kwargs)

    return guarded


def _ledger_on_disk() -> bool:
    if MODE != "production":
        return False
    if not state_flags["ledger_verified"]:
        state = rd.load_results_state(OUTPUTS / "results.json")
        ledger = set(state["executed_prompt_keys"])
        state_flags["ledger_verified"] = ledger == allowed_keys and state["phases"]["stage1"]["status"] == "running"
    return state_flags["ledger_verified"]


def _sanctioned_capture_reference(original):
    @functools.wraps(original)
    def guarded(model, progs, prompt, tag="S1"):
        key = cr.tagged_key(prompt, tag)
        if tag != "S1" or key not in allowed_keys or key in done_keys or not _ledger_on_disk():
            _deny(f"ri.capture_reference for {key}")
        done_keys.add(key)
        events["sanctioned_runs"] += 1
        run_log.write(json.dumps({"n": events["sanctioned_runs"], "key": key}) + "\n")
        inside.append(key)
        try:
            return original(model, progs, prompt, tag)
        finally:
            inside.pop()

    return guarded


def _refuse_run_arm(*args, **kwargs):
    _deny("ri.run_arm in stage 1")


WRAPPERS = {key: _low_level(name, original) for key, (name, original) in LOW_LEVEL.items()}
WRAPPERS[id(ri.capture_reference)] = _sanctioned_capture_reference(ri.capture_reference)
WRAPPERS[id(ri.run_arm)] = _refuse_run_arm
ORIGINALS = {**{k: v[1] for k, v in LOW_LEVEL.items()}, id(ri.capture_reference): ri.capture_reference, id(ri.run_arm): ri.run_arm}


def _sweep_aliases() -> list[str]:
    replaced = []
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name.startswith("neural_decompiler") or module_name.startswith("experiment_101")):
            continue
        for attribute, value in list(vars(module).items()):
            if id(value) in ORIGINALS and value is ORIGINALS[id(value)]:
                setattr(module, attribute, WRAPPERS[id(value)])
                replaced.append(f"{module_name}.{attribute}")
    return replaced


def _seal_model(model) -> dict:
    sealed = {"instances": 0, "classes": [], "signatures_kept": True}

    def signature(obj):
        try:
            return str(inspect.signature(obj))
        except (TypeError, ValueError):
            return None

    for name, module in model.named_modules():
        original = module.forward
        before = signature(original)

        def make(original=original, name=name):
            @functools.wraps(original)
            def checked(*args, **kwargs):
                if not inside:
                    _deny(f"{name or 'model'}.forward outside a sanctioned call")
                return original(*args, **kwargs)

            return checked

        object.__setattr__(module, "forward", make())
        sealed["signatures_kept"] = sealed["signatures_kept"] and signature(module.forward) == before
        sealed["instances"] += 1
    for cls in sorted({type(module) for module in model.modules()}, key=lambda c: c.__qualname__):
        for method in ("forward", "run_with_hooks", "run_with_cache", "generate"):
            if method in cls.__dict__ and callable(cls.__dict__[method]):
                original = cls.__dict__[method]
                before = signature(original)

                def make(original=original, label=f"{cls.__qualname__}.{method}"):
                    @functools.wraps(original)
                    def checked(*args, **kwargs):
                        if not inside:
                            _deny(f"{label} outside a sanctioned call")
                        return original(*args, **kwargs)

                    return checked

                setattr(cls, method, make())
                sealed["signatures_kept"] = sealed["signatures_kept"] and signature(cls.__dict__[method]) == before
                sealed["classes"].append(f"{cls.__qualname__}.{method}")
    return sealed


original_load = models.load_model
sealing: dict = {}


def _counted_load(spec, *args, **kwargs):
    events["load_model"] += 1
    if events["load_model"] > 1:
        _deny("a second model load")
    model = original_load(spec, *args, **kwargs)
    sealing.update(_seal_model(model))
    return model


models.load_model = _counted_load
aliases = _sweep_aliases()


def write_check(paths) -> None:
    """The runner's write_check under this launcher: each path is one of the three stage-1 files, allowed by the hook's
    predicate, and does not exist yet; the stage-1 directory exists."""
    for path in paths:
        path = Path(path).resolve()
        if path not in STAGE1_FILES or not write_allowed(path) or path.exists() or not path.parent.is_dir():
            raise ri.PhaseError(f"the launcher's write predicate refuses {path}")


def preflight() -> dict:
    checks: dict = {}

    def need(name: str, ok: bool, detail=None) -> None:
        checks[name] = bool(ok) if detail is None else {"ok": bool(ok), "detail": detail}
        if not ok:
            raise RuntimeError(f"preflight failed: {name} ({detail})")

    need("N1_cwd_is_repo_root", Path.cwd().resolve() == ROOT)
    need("N1_tree_clean", _git("status", "--porcelain", "--untracked-files=all") == "")
    git_state = collect_git_state()
    need("N1_runner_git_state_at_root", git_state == {"commit": HEAD, "dirty": False}, git_state)
    recorded = json.loads(LOCK_RUN_RECORD.read_text())["environment"]
    identity = {key: (environment[key], recorded[key]) for key in ("python", "executable", "platform", "torch", "transformers", "transformer_lens", "huggingface_hub",
                                                                    "numpy", "torch_num_threads", "model")}
    identity["checkpoint_files"] = (environment["checkpoint"]["files"], recorded["checkpoint"]["files"])
    differing = sorted(key for key, (now, then) in identity.items() if now != then)
    need("N2_environment_identity_with_lock_run", not differing, differing)
    lock_bytes, prereg_bytes = (ROOT / ri.LOCK_RELATIVE_PATH).read_bytes(), (ROOT / ri.PREREGISTRATION_RELATIVE_PATH).read_bytes()
    lock = json.loads(lock_bytes)
    need("installed_lock_verifies", lock["content_sha256"] == rc.content_digest(lock) and ri.render_preregistration(lock).encode("utf-8") == prereg_bytes)
    need("lock_and_preregistration_committed", all(_git("ls-files", "--error-unmatch", p) == p for p in (ri.LOCK_RELATIVE_PATH, ri.PREREGISTRATION_RELATIVE_PATH)))
    if MODE == "production":
        need("production_token_names_the_installed_lock", os.environ.get("STAGE1101_LOCK") == lock["content_sha256"])
    blobs = {"lock": lock["module"]["blob"], "git_head": _git("rev-parse", "HEAD:src/neural_decompiler/read_intervention.py"),
             "imported_file": rc.program_blob_sha1(Path(ri.__file__)), "own_blob": ri.own_blob()}
    need("module_binding_ri", len(set(blobs.values())) == 1, blobs)
    need("all_pinned_blobs", ri.assert_frozen_blobs() == ri.FROZEN_BLOBS == lock["module_blobs"])

    spec = importlib.util.spec_from_file_location("experiment_101_runner", ROOT / "experiments/101-read-component-intervention/run.py")
    runner_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner_module
    spec.loader.exec_module(runner_module)
    if runner_module.load_model is not models.load_model:
        raise RuntimeError("the runner's model loader is not the launcher's counted loader")
    logs: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        logs.append(message)

    runner = runner_module.Runner(log=log, write_check=write_check)
    need("production_configuration", runner.config is ri.PRODUCTION and Path(runner.root).resolve() == ROOT)
    base = runner._base()
    confirmation, confirmation_sha = runner._confirmation(base)
    calibration, calibration_sha = runner._calibration(confirmation, confirmation_sha)
    state = runner._state_for("stage1", base.digests)
    runner._validated_lock(state, base, confirmation, confirmation_sha, calibration, calibration_sha)
    checks["full_validate_lock"] = True
    need("stage1_not_started_ledger_empty", state["phases"]["stage1"] == {"status": "not_started"} and state["executed_prompt_keys"] == [])
    need("no_stage1_file_yet", not any(path.exists() for path in STAGE1_FILES))
    need("runtime_and_versions_equal_020_explore", runner._check_runtime(base.inputs.closure) is not None)
    keys = confirmation.s1_keys()
    need("s1_keys_36", len(keys) == 36 == len(set(keys)) and all(key.endswith("|ref|581|S1") for key in keys))
    need("s1_keys_not_forbidden", not {cr.untagged(key) for key in keys} & base.forbidden)
    probes = {"allowed": [str(p) for p in sorted(STAGE1_FILES)], "refused": [str(STAGE1_DIR / "other.json"), str(EXPERIMENT_DIR / "table.json")]}
    predicate = {path: write_allowed(Path(path)) for path in probes["allowed"] + probes["refused"]}
    if MODE == "production":
        need("write_predicate_allows_exactly_the_stage1_files", all(predicate[p] for p in probes["allowed"]) and not any(predicate[p] for p in probes["refused"]), predicate)
    else:
        need("write_predicate_refuses_stage1_files_in_rehearsal", not any(predicate.values()), predicate)
    return {"checks": checks, "runner": runner, "confirmation": confirmation, "lock": lock, "logs": logs, "keys": keys}


record: dict = {"mode": MODE, "environment": environment, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
status, error = None, None
sentinel = HERE / "stage1.sentinel"
try:
    if not sys.flags.dont_write_bytecode:
        raise RuntimeError("run with python -B or PYTHONDONTWRITEBYTECODE=1 (the lock review's N8)")
    pre = preflight()
    aliases += _sweep_aliases()
    record["preflight"] = pre["checks"]
    record["aliases_wrapped"] = sorted(set(aliases))
    record["hook_self_test"] = hook_self_test()
    if not record["hook_self_test"]["ok"]:
        raise RuntimeError(f"the hook's self-test failed: {record['hook_self_test']}")
    if MODE == "rehearsal":
        status = 0
    else:
        # the sentinel after the preflight, immediately before the one stage 1 (the lock review's S3): a refusal in the
        # preflight leaves no sentinel and can be retried once its cause is fixed
        fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, LAUNCHER_SHA256.encode())
        os.close(fd)
        allowed_keys.update(pre["keys"])
        events["stage1_invocations"] += 1
        status = pre["runner"].stage1()  # the one stage 1
        state = rd.load_results_state(OUTPUTS / "results.json")
        stage1 = state.get("stage1") or {}
        expected_tree = sorted(f"?? {p.relative_to(ROOT)}" for p in STAGE1_FILES)
        tree = sorted(line for line in _git("status", "--porcelain", "--untracked-files=all").splitlines() if line)
        record["post"] = {"keys_done": len(done_keys), "all_36_ran_once": done_keys == allowed_keys,
                          "ledger_is_exactly_the_36_s1_keys": sorted(state["executed_prompt_keys"]) == sorted(allowed_keys),
                          "tree_is_exactly_the_three_untracked_stage1_files": tree == expected_tree,
                          "files": {str(p.relative_to(ROOT)): _sha256_file(p) for p in sorted(STAGE1_FILES) if p.exists()},
                          "record_digest_reproduces": bool(stage1) and stage1.get("digest") == b0c.stage_one_digest(stage1),
                          "files_equal_record": bool(stage1) and all(_sha256_file(ROOT / entry["path"]) == entry["sha256"] for entry in stage1["files"].values()),
                          "table_content_sha256": stage1.get("table_content_sha256"), "commit_these": sorted(str(p.relative_to(ROOT)) for p in STAGE1_FILES)}
    record["logs"] = pre["logs"]
except BaseException as exc:
    error = f"{type(exc).__name__}: {exc}"
record.update({"ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_status": status, "error": error, "events": events, "sealing": sealing,
               "tree_after": _git("status", "--porcelain", "--untracked-files=all")})
if run_log is not None:
    run_log.close()
# never overwrite an earlier record (the lock review's S3): each invocation writes a new file
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
fd = os.open(HERE / f"{MODE}_run_{stamp}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
os.write(fd, (json.dumps(record, indent=1, default=str) + "\n").encode("utf-8"))
os.close(fd)
print(json.dumps({key: record.get(key) for key in ("mode", "started_at", "ended_at", "exit_status", "error")}), flush=True)
print(json.dumps({key: value for key, value in events.items() if key != "outputs_reads"}, default=str), flush=True)
post_ok = MODE == "production" and status == 0 and error is None and all(record["post"][k] for k in ("all_36_ran_once", "ledger_is_exactly_the_36_s1_keys",
                                                                                                     "tree_is_exactly_the_three_untracked_stage1_files",
                                                                                                     "record_digest_reproduces", "files_equal_record"))
if post_ok:
    print("Stage 1 complete. The user commits exactly these files, unchanged, before confirm: " + ", ".join(record["post"]["commit_these"]), flush=True)
sys.exit(0 if status == 0 and error is None and (MODE != "production" or post_ok) else (status if isinstance(status, int) and status else 1))
