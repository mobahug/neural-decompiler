"""Guarded launcher for Experiment 101's `confirm` (plan § The prediction freeze, § One-shot protections and guarded
launchers; 025's confirm-launcher v3 pattern). The forward guards, the sealing and the counted load are the calibrate
launcher's, which ran in production; the audit hook is the stage-1 launcher's version 2 (fail-closed path resolution,
one decision function, a self-test), which the stage-1 re-check found READY. It runs from a scratchpad directory outside
the repository and writes its own records there (each with O_EXCL, never overwritten); they are copied byte-identically
into the experiment's `evidence/` afterwards. The mode must be given explicitly (`CONFIRM101_MODE`).

PREPARED, NOT YET EXERCISED IN PRODUCTION. Confirm can run only after the user has committed the three stage-1 files.

**`rehearsal`** — never calls `confirm`, allows no forward, writes nothing in the repository. The preflight:
- N1: the repository root; a clean tree; the runner's own git state; HEAD, the branch and `origin/exp101-phases`
  recorded;
- N2: the environment against the lock run's record;
- `-B` and `HF_HUB_OFFLINE=1` required;
- the installed lock and preregistration (hashes, rendering, committed), the module binding (the 20 pinned blobs; `ri`'s
  own blob equal to the lock's and the git blob at HEAD);
- **the stage-1 files**: committed, their sha256s equal to the state's stage-1 record, the table's content sha256 the
  record's; the state's `phases.stage1` is `complete`, with `awaiting_commit`, and **no incident entry** (the stage-1
  re-check's N1: the runner's own confirm rule does not read `phases.stage1.incidents`); the ledger is exactly the 36
  S1 keys; confirm is `not_started` with no confirmation block;
- the runner's `_state_for("confirm")` and `_validated_lock` (with the stage-1 files as extra tracked paths); the 020
  runtime and versions;
- the 1,980 S2 keys: distinct, none in the ledger, none forbidden;
- the hook's self-test under the production rules;
- then, weights only (the review's N7): the model loaded under every guard, its digests against the lock, the noun keys,
  the stage-1 binding, the geometry against the lock bit for bit, and the runner's own barrier (the committed stage-1
  files re-read and the table recomputed bit for bit), exactly as `confirm` will run them before its ledger. Nothing is
  written; no forward pass.

**`smoke`** — engineering only: the same preflight, then exactly the production guards with one exposed (spent) frame's
55 run keys (`coordinated-adjective-1` of 020's pool, through the same `ri.stage_runs`), compared bit for bit with the
calibrate launcher's unguarded reference (`../calibrate/smoke_reference.pt`, its 71 tensors and logits digests), and
each unsanctioned route refused (a model call, `model.forward`, `pm.capture_prompt`, `ri.capture_reference` — refused
outright in confirm —, a key outside the allowed set, a second run). Nothing in the repository is written.

**`production`** — the one confirm. It requires `CONFIRM101_LOCK=<the lock's content sha256>` and
`CONFIRM101_TABLE=<the stage-1 table's content sha256>`. After the preflight it creates an exclusive sentinel and calls
`runner.confirm()` once:
- **N3.** A forward runs only inside `ri.run_arm`, wrapped by identity, allowed only for one of the 1,980 S2 keys, each
  once, and only once the S2 ledger is on disk with the barrier passed (`confirmation.barrier.passed`, `phases.confirm`
  running). `ri.capture_reference` refuses outright. The guarded routes are those of the calibrate launcher.
- **N4.** The audit hook allows repository writes only to `outputs/experiment-101/results.json` (through its
  `.results-*.json` temporary files and their renames), `stage2-measurements.pt` and `stage2-partial.pt`, and the
  `mkdir` of `outputs/experiment-101/`; it fails closed on any path it cannot resolve.
- **Signals** (the review's S1 and N1). SIGHUP is ignored for the whole run, so a closed terminal cannot interrupt it
  (`nohup` semantics kept). During `confirm()` only, SIGTERM raises `KeyboardInterrupt`, so that the runner records an
  interruption after the ledger as an incident; once `confirm()` returns or raises, SIGTERM is ignored while the record is
  written. Production refuses to start when stdout or stderr is a terminal: launch it detached (`nohup`), with its output
  to a file (never through a pipe or `tee`), and do not interrupt it.
- **Logging.** Every sanctioned run is appended, with its key and its replacement's digest, to
  `production_runs.jsonl` here, at admission.
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
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
OUTPUTS = (ROOT / "outputs/experiment-101").resolve()
EXPERIMENT_DIR = (ROOT / "experiments/101-read-component-intervention").resolve()
STAGE1_FILES = {EXPERIMENT_DIR / "stage1" / name for name in ("states.f64", "states.json", "table.json")}
MODE = os.environ.get("CONFIRM101_MODE", "")
if MODE not in ("rehearsal", "smoke", "production"):
    raise SystemExit("set CONFIRM101_MODE to 'rehearsal', 'smoke' or 'production'")
LAUNCHER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LOCK_RUN_RECORD = EXPERIMENT_DIR / "evidence/lock-2026-10-14/lock_run.json"
SMOKE_REFERENCE = HERE.parent / "calibrate" / "smoke_reference.pt"
SMOKE_REFERENCE_SHA256 = "ba4698a55bd58f3638311f853743860a248e2d61caf3c80af9414531859df204"  # as committed in evidence/calibrate-2026-10-14/SHA256SUMS
WRITABLE = {OUTPUTS / "results.json", OUTPUTS / "stage2-measurements.pt", OUTPUTS / "stage2-partial.pt"}
READABLE_OUTPUTS = (ROOT / "outputs/experiment-020/results.json", OUTPUTS)

signal.signal(signal.SIGHUP, signal.SIG_IGN)  # the review's S1: a closed terminal never interrupts the run
os.chdir(ROOT)
if Path.cwd().resolve() != ROOT:
    raise SystemExit(f"cwd {Path.cwd()} is not the repository root {ROOT}")

events = {"mode": MODE, "load_model": 0, "refused": [], "forbidden_reads": [], "forbidden_writes": [], "outputs_reads": [], "sanctioned_runs": 0,
          "sanctioned_forward_calls": 0, "confirm_invocations": 0, "signals": []}

UNRESOLVED = Path("/<unresolved>")


def _path(value, dir_fd=None):
    """The absolute path an audit event names, or None when the argument is not a path. A relative name resolves against
    its directory descriptor when one is given (``dir_fd`` >= 0; Python 3.12 passes -1 for none) and against the working
    directory otherwise; a name that cannot be resolved is UNRESOLVED, which every write-type check refuses."""
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


def write_allowed(path: Path, production: bool) -> bool:
    return production and (path in WRITABLE or _temp_state(path, production))


def decide(event, args, production):
    """A pure decision for one audit event: None to allow, or (kind, detail) to refuse."""
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
        if _in_repo(path) and not write_allowed(path, production):
            return ("forbidden_writes", str(path))
        return None
    if event in ("os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir", "os.chmod"):
        dir_fd = args[2] if event in ("os.mkdir", "os.chmod") and len(args) > 2 else args[1] if event in ("os.remove", "os.rmdir", "shutil.rmtree") and len(args) > 1 else None
        path = _path(args[0], dir_fd) if args else UNRESOLVED
        if path is None or not _in_repo(path):
            return None
        if event == "os.remove" and _temp_state(path, production):
            return None
        if event == "os.mkdir" and production and path == OUTPUTS:
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
    if decision is not None:
        kind, detail = decision
        events[kind].append(detail)
        raise PermissionError(f"confirm launcher ({MODE}): {kind}: {detail}")


def hook_self_test() -> dict:
    """The recorded Python 3.12 event tuples of confirm's writes, decided under the production rules, with refused
    controls; nothing is created."""
    cloexec = getattr(os, "O_CLOEXEC", 0)
    flags_w = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | cloexec
    tmp = OUTPUTS / ".results-selftest.json"
    allowed = {
        "mkdir outputs": ("os.mkdir", (str(OUTPUTS), 511, -1)),
        "state temporary": ("open", (str(tmp), None, os.O_RDWR | os.O_CREAT | os.O_EXCL | cloexec)),
        "state rename": ("os.rename", (str(tmp), str(OUTPUTS / "results.json"), -1, -1)),
        "state temporary removed": ("os.remove", (str(tmp), -1)),
        "measurements": ("open", (str(OUTPUTS / "stage2-measurements.pt"), "wb", flags_w)),
        "partial measurements": ("open", (str(OUTPUTS / "stage2-partial.pt"), "wb", flags_w)),
        "read the measurements": ("open", (str(OUTPUTS / "stage2-measurements.pt"), "rb", cloexec)),
        "read a stage-1 file": ("open", (str(EXPERIMENT_DIR / "stage1/table.json"), "r", cloexec)),
        "read 020's results": ("open", (str(ROOT / "outputs/experiment-020/results.json"), "r", cloexec)),
    }
    refused = {
        "a stage-1 file written": ("open", (str(EXPERIMENT_DIR / "stage1/table.json"), "w", flags_w)),
        "the lock written": ("open", (str(EXPERIMENT_DIR / "preregistration-lock.json"), "w", flags_w)),
        "a source file": ("open", (str(ROOT / "src/neural_decompiler/read_intervention.py"), "w", flags_w)),
        "another output file": ("open", (str(OUTPUTS / "other.pt"), "wb", flags_w)),
        "a truncating read-only open": ("open", (str(EXPERIMENT_DIR / "stage1/table.json"), None, os.O_RDONLY | os.O_TRUNC)),
        "relative mkdir at the root": ("os.mkdir", ("x", 511, -1)),
        "relative remove at the root": ("os.remove", ("x", -1)),
        "relative rename at the root": ("os.rename", ("a", "b", -1, -1)),
        "rename onto a stage-1 file": ("os.rename", (str(tmp), str(EXPERIMENT_DIR / "stage1/table.json"), -1, -1)),
        "unresolvable dir_fd": ("os.remove", ("x", 10 ** 6)),
        "remove a stage-1 file": ("os.remove", (str(EXPERIMENT_DIR / "stage1/states.f64"), -1)),
        "read another experiment's outputs": ("open", (str(ROOT / "outputs/experiment-025/results.json"), "r", 0)),
    }
    out = {"allowed": {name: decide(event, args, True) is None for name, (event, args) in allowed.items()},
           "refused": {name: decide(event, args, True) is not None for name, (event, args) in refused.items()},
           "rehearsal_refuses_the_measurements": decide(*allowed["measurements"], False) is not None}
    out["ok"] = all(out["allowed"].values()) and all(out["refused"].values()) and out["rehearsal_refuses_the_measurements"]
    return out


sys.addaudithook(_hook)

import numpy  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
import huggingface_hub  # noqa: E402
from huggingface_hub import constants as hf_constants  # noqa: E402

from neural_decompiler import capture, interventions, models  # noqa: E402
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
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
    "head": HEAD, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    "origin_branch": subprocess.run(["git", "--no-optional-locks", "rev-parse", "origin/exp101-phases"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    "repository": str(ROOT), "launcher_path": str(Path(__file__).resolve()), "launcher_sha256": LAUNCHER_SHA256, "mode": MODE, "python": sys.version,
    "executable": sys.executable, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__,
    "transformer_lens": importlib.metadata.version("transformer-lens"), "huggingface_hub": huggingface_hub.__version__, "numpy": numpy.__version__,
    "torch_num_threads": torch.get_num_threads(),
    "model": {"model_id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision, "device": models.PYTHIA_70M.device, "dtype": models.PYTHIA_70M.dtype},
    "checkpoint": {"snapshot": str(SNAPSHOT), "files": {p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size} for p in sorted(SNAPSHOT.iterdir()) if p.is_file()}},
    "env_flags": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "OMP_NUM_THREADS",
                                                 "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "VIRTUAL_ENV", "GIT_OPTIONAL_LOCKS", "CONFIRM101_MODE")},
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}

# ---------------------------------------------------------------------------
# Guards (the calibrate launcher's, which ran in production).

inside: list[str] = []
allowed_keys: set[str] = set()
done_keys: set[str] = set()
state_flags = {"ledger_verified": False}
original_call = torch.nn.Module.__call__
run_log = open(HERE / "production_runs.jsonl", "a", buffering=1, encoding="utf-8") if MODE == "production" else None


def _deny(name: str):
    events["refused"].append(name)
    raise RuntimeError(f"{name} refused by the confirm launcher ({MODE})")


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
        return True
    if not state_flags["ledger_verified"]:
        state = rd.load_results_state(OUTPUTS / "results.json")
        confirmation = state.get("confirmation") or {}
        state_flags["ledger_verified"] = (set(state["executed_prompt_keys"]) >= allowed_keys and (confirmation.get("barrier") or {}).get("passed") is True
                                          and state["phases"]["confirm"]["status"] == "running")
    return state_flags["ledger_verified"]


def _vector_digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().to("cpu", torch.float32).contiguous().reshape(-1).numpy().tobytes()).hexdigest()


def _sanctioned_run_arm(original):
    @functools.wraps(original)
    def guarded(model, prompt, tag, *, replacements=None, capture_sites=()):
        key = cr.tagged_key(prompt, tag)
        if key not in allowed_keys:
            _deny(f"ri.run_arm for {key}: not a key of this mode's manifest")
        if key in done_keys:
            _deny(f"ri.run_arm for {key}: a second time")
        if not _ledger_on_disk():
            _deny(f"ri.run_arm for {key}: the S2 ledger is not on disk with the barrier passed")
        done_keys.add(key)
        events["sanctioned_runs"] += 1
        if run_log is not None:
            run_log.write(json.dumps({"n": events["sanctioned_runs"], "key": key, "replacements": {f"{site}@{pos}": _vector_digest(value)
                                                                                                  for (site, pos), value in (replacements or {}).items()}}) + "\n")
        inside.append(key)
        try:
            return original(model, prompt, tag, replacements=replacements, capture_sites=capture_sites)
        finally:
            inside.pop()

    return guarded


def _refuse_capture_reference(*args, **kwargs):
    _deny("ri.capture_reference in confirm")


WRAPPERS = {key: _low_level(name, original) for key, (name, original) in LOW_LEVEL.items()}
WRAPPERS[id(ri.run_arm)] = _sanctioned_run_arm(ri.run_arm)
WRAPPERS[id(ri.capture_reference)] = _refuse_capture_reference
ORIGINALS = {**{k: v[1] for k, v in LOW_LEVEL.items()}, id(ri.run_arm): ri.run_arm, id(ri.capture_reference): ri.capture_reference}


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


def preflight() -> dict:
    checks: dict = {}

    def need(name: str, ok: bool, detail=None) -> None:
        checks[name] = bool(ok) if detail is None else {"ok": bool(ok), "detail": detail}
        if not ok:
            raise RuntimeError(f"preflight failed: {name} ({detail})")

    need("bytecode_writes_off", bool(sys.flags.dont_write_bytecode))
    need("hf_hub_offline", os.environ.get("HF_HUB_OFFLINE") == "1")
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
    blobs = {"lock": lock["module"]["blob"], "git_head": _git("rev-parse", "HEAD:src/neural_decompiler/read_intervention.py"),
             "imported_file": rc.program_blob_sha1(Path(ri.__file__)), "own_blob": ri.own_blob()}
    need("module_binding_ri", len(set(blobs.values())) == 1, blobs)
    need("all_pinned_blobs", ri.assert_frozen_blobs() == ri.FROZEN_BLOBS == lock["module_blobs"])
    state = rd.load_results_state(OUTPUTS / "results.json")
    record = state.get("stage1") or {}
    phase1 = state["phases"]["stage1"]
    need("stage1_complete_awaiting_commit_no_incident", phase1.get("status") == "complete" and phase1.get("awaiting_commit") is True
         and not phase1.get("incidents") and not record.get("incident"), {k: v for k, v in phase1.items() if k in ("status", "awaiting_commit", "incidents")})
    need("stage1_record_digest_reproduces", bool(record) and record.get("digest") == b0c.stage_one_digest(record) and record.get("lock_sha256") == lock["content_sha256"])
    files_ok = {}
    for name, entry in record.get("files", {}).items():
        path = ROOT / entry["path"]
        files_ok[entry["path"]] = (_git("ls-files", "--error-unmatch", entry["path"]) == entry["path"] and _sha256_file(path) == entry["sha256"]
                                   and _git("rev-parse", f"HEAD:{entry['path']}") != "")
    need("stage1_files_committed_and_equal_to_the_record", len(files_ok) == 3 and all(files_ok.values()) and {ROOT / p for p in files_ok} == STAGE1_FILES, files_ok)
    need("confirm_not_started", state["phases"]["confirm"] == {"status": "not_started"} and state.get("confirmation") is None)
    if MODE == "production":
        need("production_token_names_the_lock", os.environ.get("CONFIRM101_LOCK") == lock["content_sha256"])
        need("production_token_names_the_table", os.environ.get("CONFIRM101_TABLE") == record.get("table_content_sha256"))

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

    runner = runner_module.Runner(log=log)
    need("production_configuration", runner.config is ri.PRODUCTION and Path(runner.root).resolve() == ROOT)
    base = runner._base()
    confirmation, confirmation_sha = runner._confirmation(base)
    calibration, calibration_sha = runner._calibration(confirmation, confirmation_sha)
    eligible = runner._state_for("confirm", base.digests)
    runner._validated_lock(eligible, base, confirmation, confirmation_sha, calibration, calibration_sha, extra_tracked=sorted(STAGE1_FILES))
    checks["full_validate_lock_with_the_stage1_files"] = True
    need("runtime_and_versions_equal_020_explore", runner._check_runtime(base.inputs.closure) is not None)
    s1 = set(confirmation.s1_keys())
    need("ledger_is_exactly_the_36_s1_keys", set(state["executed_prompt_keys"]) == s1 and len(state["executed_prompt_keys"]) == 36)
    s2 = confirmation.s2_keys()
    need("s2_keys_1980_distinct", len(s2) == 1980 == len(set(s2)))
    need("s2_keys_not_in_ledger_not_forbidden", not set(s2) & set(state["executed_prompt_keys"]) and not {cr.untagged(k) for k in s2} & base.forbidden)
    test = hook_self_test()
    need("hook_self_test", test["ok"], test)
    return {"checks": checks, "runner": runner, "base": base, "confirmation": confirmation, "logs": logs, "s2": s2}


def barrier_dry_run(pre: dict) -> dict:
    """Weights only, nothing written: everything confirm runs before its ledger, with the runner's own functions (the
    model load counted and sealed; no forward pass)."""
    runner, base, confirmation = pre["runner"], pre["base"], pre["confirmation"]
    out = {}
    lock, _ = runner._installed_lock()
    model = models.load_model(models.PYTHIA_70M)
    progs = ul.ModelPrograms.from_model(model, base.inputs)
    runner._model_dependencies(lock, model, progs)
    out["model_dependencies"] = True
    noun_keys = runner._check_nouns(progs, base.inputs)
    geometry = ri.read_geometry(base.inputs.lock_011)
    runner._check_stage1_binding(lock, confirmation, progs, model, noun_keys)
    out["stage1_binding"] = True
    out["geometry_against_lock"] = runner._geometry_against_lock(lock, progs, geometry, [frame.frame_id for frame in confirmation.frames])
    barrier = runner._barrier(lock, confirmation, progs, geometry, noun_keys)
    out["barrier"] = {"passed": True, "stage1_digest": barrier.record["digest"], "table_content_sha256": barrier.table["content_sha256"]}
    out["forward_calls"] = events["sanctioned_forward_calls"]
    out["ok"] = out["geometry_against_lock"].get("bitwise_equal") is True and out["forward_calls"] == 0
    return out


def smoke(pre: dict) -> dict:
    runner, base = pre["runner"], pre["base"]
    frames = runner._exposed_frames(base.inputs)
    schedule = runner._calibration_schedules(base.inputs, frames)[0]
    allowed_keys.update(run.key for run in schedule)
    models.seed_runtime(rd.RUNTIME_SEED, models.PYTHIA_70M.deterministic_algorithms)
    model = models.load_model(models.PYTHIA_70M)
    progs = ul.ModelPrograms.from_model(model, base.inputs)
    geometry = ri.read_geometry(base.inputs.lock_011)
    out: dict = {}
    logits: dict = {}
    executed: list = []
    ri.stage_runs(model, progs, [schedule], geometry, ri.PRODUCTION, executed=executed, out=out, logits=logits)
    if _sha256_file(SMOKE_REFERENCE) != SMOKE_REFERENCE_SHA256:
        raise RuntimeError("the smoke reference is not the committed one")
    reference = torch.load(SMOKE_REFERENCE)
    if set(reference["keys"][1:]) != {run.key for run in schedule}:
        raise RuntimeError("the smoke reference's runs are not this frame's 55 keys")
    mine = {name: rc.tensor_digest(value) for name, value in out.items()}
    result = {"frame": schedule[0].frame.frame_id, "runs": len(executed), "tensors_equal_unguarded": mine == reference["tensors_sha256"],
              "logits_equal_unguarded": logits == reference["logits_sha256"], "sealing": dict(sealing)}
    refusals = {}
    other = base.inputs.pool.reference_prompt(frames[1])  # a spent exposed prompt outside the allowed set; never a 101 prompt, not even as a probe
    for label, attempt in (("model_call", lambda: model(torch.tensor([[1, 2, 3]]))), ("model_forward", lambda: model.forward(torch.tensor([[1, 2, 3]]))),
                           ("pm.capture_prompt", lambda: pm.capture_prompt(model, schedule[0].prompt, [])),
                           ("ri.capture_reference", lambda: ri.capture_reference(model, progs, schedule[0].prompt, "S1")),
                           ("ri.run_arm_outside_the_allowed_set", lambda: ri.run_arm(model, other, "REF")),
                           ("ri.run_arm_second_time", lambda: ri.run_arm(model, schedule[0].prompt, schedule[0].tag))):
        before = events["sanctioned_forward_calls"]
        try:
            attempt()
            refusals[label] = {"refused": False}
        except RuntimeError as exc:
            refusals[label] = {"refused": "refused by the confirm launcher" in str(exc), "forward_calls_during": events["sanctioned_forward_calls"] - before}
    result["refusals"] = refusals
    result["ok"] = (result["tensors_equal_unguarded"] and result["logits_equal_unguarded"] and result["runs"] == 55 and sealing.get("signatures_kept") is True
                    and all(entry["refused"] and entry.get("forward_calls_during") == 0 for entry in refusals.values()))
    return result


def _signal_to_interrupt(signum, frame):
    events["signals"].append(signum)
    raise KeyboardInterrupt(f"signal {signum}")


record: dict = {"mode": MODE, "environment": environment, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
status, error = None, None
sentinel = HERE / "confirm.sentinel"
try:
    pre = preflight()
    aliases += _sweep_aliases()
    record["preflight"] = pre["checks"]
    record["aliases_wrapped"] = sorted(set(aliases))
    if MODE == "rehearsal":
        record["barrier_dry_run"] = barrier_dry_run(pre)
        status = 0 if record["barrier_dry_run"]["ok"] else 1
    elif MODE == "smoke":
        record["smoke"] = smoke(pre)
        status = 0 if record["smoke"]["ok"] else 1
    else:
        if os.isatty(1) or os.isatty(2):
            raise RuntimeError("production refuses a terminal on stdout or stderr: launch it detached with its output to a file")
        fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)  # after the preflight, immediately before the one confirm
        os.write(fd, LAUNCHER_SHA256.encode())
        os.close(fd)
        allowed_keys.update(pre["s2"])
        events["confirm_invocations"] += 1
        signal.signal(signal.SIGTERM, _signal_to_interrupt)  # during confirm only: an interruption after the ledger becomes an incident
        try:
            status = pre["runner"].confirm()  # the one confirm
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)  # the record is written whatever happens next
        state = rd.load_results_state(OUTPUTS / "results.json")
        confirmation_block = state.get("confirmation") or {}
        record["post"] = {"phase": state["phases"]["confirm"].get("status"), "incident": confirmation_block.get("incident"),
                          "label": ((confirmation_block.get("results") or {}).get("outcome") or {}).get("label"),
                          "keys_done": len(done_keys), "all_1980_ran_once": done_keys == allowed_keys,
                          "accounting": confirmation_block.get("accounting"), "gate_failures": (confirmation_block.get("gates") or {}).get("failures"),
                          "secondary_failures": (confirmation_block.get("secondary") or {}).get("failures"), "state_sha256": state.get("state_sha256")}
    record["logs"] = pre["logs"]
except BaseException as exc:
    error = f"{type(exc).__name__}: {exc}"
record.update({"ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_status": status, "error": error, "events": events, "sealing": sealing})
try:
    record["tree_after"] = _git("status", "--porcelain", "--untracked-files=all")
except Exception as exc:  # the record is never lost to a git failure
    record["tree_after"] = f"unavailable: {exc}"
if run_log is not None:
    run_log.close()
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
fd = os.open(HERE / f"{MODE}_run_{stamp}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
os.write(fd, (json.dumps(record, indent=1, default=str) + "\n").encode("utf-8"))
os.fsync(fd)
os.close(fd)
print(json.dumps({key: record.get(key) for key in ("mode", "started_at", "ended_at", "exit_status", "error")}), flush=True)
print(json.dumps({key: value for key, value in events.items() if key != "outputs_reads"}, default=str), flush=True)
if MODE == "production" and record.get("post"):
    print(json.dumps({k: v for k, v in record["post"].items() if k not in ("accounting",)}, default=str), flush=True)
post = record.get("post") or {}
post_ok = MODE != "production" or (post.get("phase") == "complete" and post.get("all_1980_ran_once") is True and not events["refused"]
                                   and not events["forbidden_writes"] and not events["forbidden_reads"])
sys.exit(0 if status == 0 and error is None and post_ok else (status if isinstance(status, int) and status else 1))
