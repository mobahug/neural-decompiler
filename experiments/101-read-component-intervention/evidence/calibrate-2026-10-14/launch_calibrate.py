"""Guarded launcher for Experiment 101's `calibrate` (exposed, exploratory; plan § One-shot protections and guarded
launchers, on 025's confirm-launcher v3 pattern). It runs from a scratchpad directory outside the repository, writes its
own records there, and is copied byte-identically into `experiments/101-read-component-intervention/evidence/` after the
run. The mode must be given explicitly (`CALIBRATE101_MODE`); anything else refuses.

**`rehearsal`** — never calls `calibrate`, allows no forward, writes nothing in the repository. It runs the provenance
preflight:
- N1: `chdir` to the repository root; HEAD, the branch and `origin/main` recorded; a clean tree (untracked files
  included); the runner's own `collect_git_state()` from that root;
- N2: the environment record (interpreter, packages, threads, the pinned checkpoint's file hashes), compared with the
  freeze run's record;
- the module binding: every pinned blob, and `ri.own_blob()` equal to the git blob at HEAD and to the imported file;
- the committed confirmation file against the freeze record (file and content sha256), verified by the runner;
- no 101 results state yet; calibrate eligible (`runner._state_for("calibrate")`, read-only);
- the 020 runtime and versions (`runner._check_runtime`);
- the 2,016 calibration keys: distinct, each untagged key in 020's exposed ledger and in no confirmation manifest, and
  none of them a 101 frame prompt.

**`smoke`** — engineering only, before production: the same preflight, then exactly the production guards with one
exposed coordinated frame's 56 keys (its S1 re-capture and its 55 runs; spent exposed prompts, as tier C uses). Through
the sanctioned wrappers it runs that frame with `ri.capture_reference` and `ri.stage_runs`, then `ri.run_gates` and
`ri.prediction_table` for that frame. It compares every saved tensor and every logits digest bit for bit with an
unguarded process (`smoke_reference.py`), checks that each sealed `forward` keeps its signature, and checks that each
unsanctioned route is refused. `calibrate` is never called; nothing in the repository is written.

**`production`** — the one calibrate. It requires `CALIBRATE101_CONFIRMATION=<the committed confirmation's content
sha256>` and an exclusive sentinel named by the commit (the plan's rerun rule: a rerun is possible only at a new commit,
after a recorded incident). After the same preflight:
- **N3.** A forward runs only inside a sanctioned call: `ri.run_arm` or `ri.capture_reference`, wrapped by identity,
  allowed only for a key `prompt.key|tag` among the 2,016 calibration keys, each once, and only once the calibration
  ledger (`calibration_executed` and its planned digest) is on disk. The guarded routes are `torch.nn.Module.__call__`,
  every module instance's `forward`, the class-level `forward`, `run_with_hooks`, `run_with_cache` and `generate` of
  every module class in the model, and the low-level capture and intervention entry points with all their aliases
  (`pm.capture_prompt`, `pm.run_patched`, `capture.run_capture`, `interventions.run_interventions`,
  `rd.capture_frame_020`). One model load.
- **N4.** An audit hook allows repository writes only to `outputs/experiment-101/results.json` (through its
  `.results-*.json` temporary files and one rename each), `calibration-measurements.pt`, `calibration-partial.pt` and
  `candidate-calibration-v1.json`, and the `mkdir` of `outputs/experiment-101/`. It refuses every other write, delete,
  rename, link and mkdir in the repository, and every read under `outputs/` except 020's results state and 101's own
  outputs.
- **Logging.** Every sanctioned run is appended, with its key and its replacement's digest, to `production_runs.jsonl`
  here. The launcher calls `runner.calibrate()` once and records whatever happens, with the exit status.
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
MODE = os.environ.get("CALIBRATE101_MODE", "")
if MODE not in ("rehearsal", "smoke", "production"):
    raise SystemExit("set CALIBRATE101_MODE to 'rehearsal', 'smoke' or 'production'")
LAUNCHER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FREEZE_RECORD = ROOT / "experiments/101-read-component-intervention/evidence/freeze-2026-10-14/freeze_run.json"
SMOKE_REFERENCE = HERE / "smoke_reference.pt"

# N1: the working directory is the repository root before anything else runs.
os.chdir(ROOT)
if Path.cwd().resolve() != ROOT:
    raise SystemExit(f"cwd {Path.cwd()} is not the repository root {ROOT}")

events = {"mode": MODE, "load_model": 0, "refused": [], "forbidden_reads": [], "forbidden_writes": [], "output_writes": [], "output_renames": [], "outputs_reads": [],
          "sanctioned_runs": 0, "sanctioned_forward_calls": 0, "calibrate_invocations": 0}
ALLOWED_OUTPUTS = {OUTPUTS / name for name in ("results.json", "calibration-measurements.pt", "calibration-partial.pt", "candidate-calibration-v1.json")} \
    if MODE == "production" else set()
READABLE_OUTPUTS = (ROOT / "outputs/experiment-020/results.json", OUTPUTS)


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


def _temp_state(path: Path) -> bool:
    return MODE == "production" and path.parent == OUTPUTS and path.name.startswith(".results-") and path.name.endswith(".json")


def _refuse_event(kind: str, detail: str):
    events[kind].append(detail)
    raise PermissionError(f"calibrate launcher ({MODE}): {kind}: {detail}")


def _hook(event, args):
    if event == "open":
        path = _path(args[0]) if args else None
        if path is None:
            return
        mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if _inside(path, ROOT / "outputs") and not writing:
            if not any(_inside(path, allowed) for allowed in READABLE_OUTPUTS):
                _refuse_event("forbidden_reads", str(path))
            if str(path) not in events["outputs_reads"]:
                events["outputs_reads"].append(str(path))
        if not writing:
            return
        if _in_repo(path):
            if path in ALLOWED_OUTPUTS or _temp_state(path):
                events["output_writes"].append(path.name)
                return
            _refuse_event("forbidden_writes", str(path))
        return
    if event in ("os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir", "os.chmod"):
        # per event (Python 3.12 audit signatures): os.mkdir and os.chmod (path, mode, dir_fd); os.remove, os.rmdir, shutil.rmtree (path, dir_fd);
        # os.truncate (path, length) has none. A mode or a length is never taken for a descriptor (freeze review N1).
        dir_fd = args[2] if event in ("os.mkdir", "os.chmod") and len(args) > 2 else args[1] if event in ("os.remove", "os.rmdir", "shutil.rmtree") and len(args) > 1 else None
        path = _path(args[0], dir_fd) if args else None
        if path is None or not _in_repo(path):
            return
        if (event == "os.remove" and _temp_state(path)) or (event == "os.mkdir" and path == OUTPUTS and MODE == "production"):
            events["output_writes"].append(f"{event} {path.name}")
            return
        _refuse_event("forbidden_writes", f"{event} {path}")
    if event in ("os.rename", "os.link", "os.symlink"):
        if event == "os.symlink":
            paths = [p for p in (_path(args[1], args[2] if len(args) > 2 else None),) if p is not None]
        else:
            paths = [p for p in (_path(args[0], args[2] if len(args) > 2 else None), _path(args[1], args[3] if len(args) > 3 else None)) if p is not None]
        if not any(_in_repo(p) for p in paths):
            return
        if event == "os.rename" and len(paths) == 2 and _temp_state(paths[0]) and paths[1] == OUTPUTS / "results.json":
            events["output_renames"].append(f"{paths[0].name} -> results.json")
            return
        _refuse_event("forbidden_writes", f"{event} {' -> '.join(str(p) for p in paths)}")


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
    "head": HEAD, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "origin_main": _git("rev-parse", "origin/main"),
    "head_is_descendant_of_origin_main": subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=ROOT).returncode == 0,
    "commits_ahead_of_origin_main": int(_git("rev-list", "--count", "origin/main..HEAD")),
    "repository": str(ROOT), "cwd": str(Path.cwd().resolve()), "launcher_path": str(Path(__file__).resolve()), "launcher_sha256": LAUNCHER_SHA256, "mode": MODE,
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__,
    "transformer_lens": importlib.metadata.version("transformer-lens"), "huggingface_hub": huggingface_hub.__version__, "numpy": numpy.__version__,
    "torch_num_threads": torch.get_num_threads(),
    "model": {"model_id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision, "device": models.PYTHIA_70M.device, "dtype": models.PYTHIA_70M.dtype},
    "checkpoint": {"snapshot": str(SNAPSHOT), "files": {p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size} for p in sorted(SNAPSHOT.iterdir()) if p.is_file()}},
    "env_flags": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                 "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES", "VIRTUAL_ENV", "CALIBRATE101_MODE")},
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}


def preflight() -> dict:
    """Everything that must hold before a model load, with the runner's own objects; raises on the first failure."""
    checks: dict = {}

    def need(name: str, ok: bool, detail=None) -> None:
        checks[name] = bool(ok) if detail is None else {"ok": bool(ok), "detail": detail}
        if not ok:
            raise RuntimeError(f"preflight failed: {name} ({detail})")

    need("N1_cwd_is_repo_root", Path.cwd().resolve() == ROOT)
    need("N1_tree_clean", _git("status", "--porcelain", "--untracked-files=all") == "")
    git_state = collect_git_state()
    need("N1_runner_git_state_at_root", git_state == {"commit": HEAD, "dirty": False}, git_state)
    need("N1_head_descends_from_origin_main", environment["head_is_descendant_of_origin_main"], environment["commits_ahead_of_origin_main"])
    freeze = json.loads(FREEZE_RECORD.read_text())
    need("freeze_record_is_the_production_freeze", freeze["dry_run"] is False and freeze["exit_status"] == 0 and freeze["error"] is None
         and freeze["freeze_invocations"] == 1 and freeze["post_write"]["bytes_equal_pre_verified_payload"] is True)
    recorded = freeze["environment"]
    identity = {"python": (environment["python"], recorded["python"]), "platform": (environment["platform"], recorded["platform"]),
                "torch": (environment["torch"], recorded["torch"]), "transformers": (environment["transformers"], recorded["transformers"])}
    differing = sorted(key for key, (now, then) in identity.items() if now != then)
    need("N2_environment_identity_with_freeze_run", not differing, differing)
    need("all_pinned_blobs", ri.assert_frozen_blobs() == ri.FROZEN_BLOBS)
    module_path = (ROOT / "src/neural_decompiler/read_intervention.py").resolve()
    blobs = {"git_head": _git("rev-parse", "HEAD:src/neural_decompiler/read_intervention.py"), "imported_file": rc.program_blob_sha1(Path(ri.__file__)),
             "own_blob": ri.own_blob()}
    need("module_binding_ri", len(set(blobs.values())) == 1 and Path(ri.__file__).resolve() == module_path, blobs)
    runner_blobs = {"git_head": _git("rev-parse", "HEAD:experiments/101-read-component-intervention/run.py"),
                    "file": rc.program_blob_sha1(ROOT / "experiments/101-read-component-intervention/run.py")}
    need("runner_binding", len(set(runner_blobs.values())) == 1, runner_blobs)

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
    need("confirmation_is_the_frozen_file", confirmation_sha == freeze["post_write"]["file_sha256"] and confirmation.content_sha256 == freeze["post_write"]["content_sha256"],
         {"file": confirmation_sha, "content": confirmation.content_sha256})
    need("confirmation_committed_at_head", _git("ls-files", "--error-unmatch", ri.CONFIRMATION_RELATIVE_PATH) == ri.CONFIRMATION_RELATIVE_PATH)
    if MODE == "production":
        need("production_token_names_the_confirmation", os.environ.get("CALIBRATE101_CONFIRMATION") == confirmation.content_sha256)
    need("no_results_state_yet", not runner.results_path.exists())
    need("outputs_dir_holds_nothing", not OUTPUTS.exists() or not any(OUTPUTS.iterdir()), sorted(p.name for p in OUTPUTS.iterdir()) if OUTPUTS.exists() else [])
    need("calibrate_eligible", runner._state_for("calibrate", base.digests)["phases"]["calibrate"]["status"] == "not_started")
    need("runtime_and_versions_equal_020_explore", runner._check_runtime(base.inputs.closure) is not None)
    frames = runner._exposed_frames(base.inputs)
    schedules = runner._calibration_schedules(base.inputs, frames)
    references = [(frame, base.inputs.pool.reference_prompt(frame)) for frame in frames]
    keys = [cr.tagged_key(prompt, "S1") for _, prompt in references] + [run.key for schedule in schedules for run in schedule]
    untagged = {cr.untagged(key) for key in keys}
    ledger_020 = frozenset(base.inputs.closure["ledger"])
    need("calibration_keys_2016_distinct", len(keys) == 2016 == len(set(keys)), len(keys))
    need("calibration_keys_spent_exposed_only", untagged <= ledger_020 and not untagged & base.manifests)
    need("calibration_keys_disjoint_from_101", not untagged & confirmation.untagged_keys())
    sequences = {tuple(frame.prompt_ids(cue_id)) for frame in confirmation.frames for cue_id in (581, 767, 1264, 1142)}
    calibration_sequences = {tuple(run.prompt.frame.prompt_ids(run.prompt.cue_token_id)) for schedule in schedules for run in schedule}
    need("no_calibration_prompt_is_a_101_token_sequence", not sequences & calibration_sequences)
    return {"checks": checks, "runner": runner, "base": base, "confirmation": confirmation, "logs": logs, "keys": keys, "schedules": schedules, "references": references,
            "frames": frames}


# ---------------------------------------------------------------------------
# Guards.

inside: list[str] = []  # the sanctioned call in progress, if any
allowed_keys: set[str] = set()
done_keys: set[str] = set()
state_flags = {"ledger_verified": False, "planned_sha256": None}
original_call = torch.nn.Module.__call__
run_log = open(HERE / "production_runs.jsonl", "a", buffering=1, encoding="utf-8") if MODE == "production" else None


def _deny(name: str):
    events["refused"].append(name)
    raise RuntimeError(f"{name} refused by the calibrate launcher ({MODE})")


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
SANCTIONED = {id(ri.run_arm): ("ri.run_arm", ri.run_arm), id(ri.capture_reference): ("ri.capture_reference", ri.capture_reference)}


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
        planned = state.get("calibration", {}).get("planned", {})
        state_flags["ledger_verified"] = (set(state["calibration_executed"]) >= allowed_keys and planned.get("count") == len(allowed_keys)
                                          and planned.get("sha256") == pm.sha256_text(pm.canonical_json(sorted(allowed_keys))) and state["executed_prompt_keys"] == [])
    return state_flags["ledger_verified"]


def _admit(key: str, label: str) -> None:
    if key not in allowed_keys:
        _deny(f"{label} for {key}: not a key of this mode's manifest")
    if key in done_keys:
        _deny(f"{label} for {key}: a second time")
    if not _ledger_on_disk():
        _deny(f"{label} for {key}: the calibration ledger is not on disk")
    done_keys.add(key)
    events["sanctioned_runs"] += 1


def _vector_digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().to("cpu", torch.float32).contiguous().reshape(-1).numpy().tobytes()).hexdigest()


def _sanctioned_run_arm(original):
    @functools.wraps(original)
    def guarded(model, prompt, tag, *, replacements=None, capture_sites=()):
        key = cr.tagged_key(prompt, tag)
        _admit(key, "ri.run_arm")
        if run_log is not None:
            run_log.write(json.dumps({"n": events["sanctioned_runs"], "key": key, "replacements": {f"{site}@{pos}": _vector_digest(value)
                                                                                                  for (site, pos), value in (replacements or {}).items()}}) + "\n")
        inside.append(key)
        try:
            return original(model, prompt, tag, replacements=replacements, capture_sites=capture_sites)
        finally:
            inside.pop()

    return guarded


def _sanctioned_capture_reference(original):
    @functools.wraps(original)
    def guarded(model, progs, prompt, tag="S1"):
        key = cr.tagged_key(prompt, tag)
        if tag != "S1":
            _deny(f"ri.capture_reference with tag {tag}")
        _admit(key, "ri.capture_reference")
        if run_log is not None:
            run_log.write(json.dumps({"n": events["sanctioned_runs"], "key": key, "replacements": {}}) + "\n")
        inside.append(key)
        try:
            return original(model, progs, prompt, tag)
        finally:
            inside.pop()

    return guarded


WRAPPERS: dict = {}
for key, (name, original) in LOW_LEVEL.items():
    WRAPPERS[key] = _low_level(name, original)
WRAPPERS[id(ri.run_arm)] = _sanctioned_run_arm(ri.run_arm)
WRAPPERS[id(ri.capture_reference)] = _sanctioned_capture_reference(ri.capture_reference)
ORIGINALS = {**{k: v[1] for k, v in LOW_LEVEL.items()}, **{k: v[1] for k, v in SANCTIONED.items()}}


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


def smoke(pre: dict) -> dict:
    """One exposed frame through the production guards; every tensor and logits digest compared with the unguarded
    reference process; every unsanctioned route refused."""
    runner, base = pre["runner"], pre["base"]
    schedule, (frame, reference) = pre["schedules"][0], pre["references"][0]
    allowed_keys.update([cr.tagged_key(reference, "S1")] + [run.key for run in schedule])
    models.seed_runtime(rd.RUNTIME_SEED, models.PYTHIA_70M.deterministic_algorithms)  # as the runner's calibrate does
    model = models.load_model(models.PYTHIA_70M)
    progs = ul.ModelPrograms.from_model(model, base.inputs)
    geometry = ri.read_geometry(base.inputs.lock_011)
    out: dict = {}
    logits: dict = {}
    executed: list = []
    recaptured = {frame.frame_id: ri.capture_reference(model, progs, reference, "S1")}
    ri.stage_runs(model, progs, [schedule], geometry, ri.PRODUCTION, executed=executed, out=out, logits=logits)
    locked = base.inputs.closure["exploration"]["locked_states"]
    program_states = {frame.frame_id: rd.state_from_locked(locked[frame.frame_id], frame)}
    book, ceilings = ri.run_gates(progs, [schedule], out, logits, program_states, recaptured, geometry, ri.PRODUCTION)
    table = ri.prediction_table(progs, [frame], program_states, geometry, ri.PRODUCTION, runner._noun_keys(base.inputs), fractions_for_weak=True)
    reference_record = torch.load(SMOKE_REFERENCE)
    mine = {name: rc.tensor_digest(value) for name, value in out.items()}
    result = {"frame": frame.frame_id, "runs": len(executed), "keys_allowed": len(allowed_keys), "keys_done": len(done_keys),
              "tensors_equal_unguarded": mine == reference_record["tensors_sha256"], "logits_equal_unguarded": logits == reference_record["logits_sha256"],
              "state_digest_equal_unguarded": rd.state_digest(rd.locked_state(recaptured[frame.frame_id])) == reference_record["state_digest"],
              "gate_failures_this_frame": list(book.failures()), "table_frames": sorted(table["frames"]), "sealing": dict(sealing)}
    refusals = {}
    for label, attempt in (("model_call", lambda: model(torch.tensor([[1, 2, 3]]))), ("model_forward", lambda: model.forward(torch.tensor([[1, 2, 3]]))),
                           ("pm.capture_prompt", lambda: pm.capture_prompt(model, reference, [])),
                           ("ri.run_arm_outside_manifest", lambda: ri.run_arm(model, pre["references"][1][1], "REF")),
                           ("ri.run_arm_second_time", lambda: ri.run_arm(model, schedule[0].prompt, schedule[0].tag)),
                           ("ri.capture_reference_second_time", lambda: ri.capture_reference(model, progs, reference, "S1")),
                           ("rd.capture_frame_020_direct", lambda: rd.capture_frame_020(model, progs.head, reference, progs.nouns, progs.axis_T))):
        before = events["sanctioned_forward_calls"]
        try:
            attempt()
            refusals[label] = {"refused": False}
        except RuntimeError as exc:
            refusals[label] = {"refused": "refused by the calibrate launcher" in str(exc), "forward_calls_during": events["sanctioned_forward_calls"] - before}
    result["refusals"] = refusals
    result["ok"] = (result["tensors_equal_unguarded"] and result["logits_equal_unguarded"] and result["state_digest_equal_unguarded"] and result["runs"] == 55
                    and all(entry["refused"] and entry.get("forward_calls_during") == 0 for entry in refusals.values()) and sealing.get("signatures_kept") is True)
    return result


# ---------------------------------------------------------------------------
# Run.

record: dict = {"mode": MODE, "environment": environment, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
status, error = None, None
sentinel = HERE / f"calibrate-{HEAD}.sentinel"
try:
    if MODE == "production":
        fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # a second production launch at this commit refuses here
        os.write(fd, LAUNCHER_SHA256.encode())
        os.close(fd)
    pre = preflight()
    aliases += _sweep_aliases()  # the runner's namespace, imported during the preflight
    record["preflight"] = pre["checks"]
    record["aliases_wrapped"] = sorted(set(aliases))
    if MODE == "rehearsal":
        status = 0
    elif MODE == "smoke":
        record["smoke"] = smoke(pre)
        status = 0 if record["smoke"]["ok"] else 1
    else:
        allowed_keys.update(pre["keys"])
        events["calibrate_invocations"] += 1
        status = pre["runner"].calibrate()  # the one production calibrate
        record["post"] = {"keys_done": len(done_keys), "all_keys_ran_once": done_keys == allowed_keys,
                          "outputs": sorted(p.name for p in OUTPUTS.iterdir()) if OUTPUTS.exists() else []}
    record["logs"] = pre["logs"]
except BaseException as exc:
    error = f"{type(exc).__name__}: {exc}"
record.update({"ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_status": status, "error": error, "events": events,
               "sealing": sealing, "tree_after": _git("status", "--porcelain", "--untracked-files=all")})
if run_log is not None:
    run_log.close()
(HERE / f"{MODE}_run.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
print(json.dumps({key: record.get(key) for key in ("mode", "started_at", "ended_at", "exit_status", "error")}), flush=True)
print(json.dumps({key: value for key, value in events.items() if key != "outputs_reads"}, default=str), flush=True)
sys.exit(0 if status == 0 and error is None else (status if isinstance(status, int) and status else 1))
