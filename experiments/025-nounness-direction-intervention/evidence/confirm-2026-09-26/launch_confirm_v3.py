"""Guarded launcher for Experiment 025's `confirm`: a read-only rehearsal mode, and the production mode (exactly once).

The mode must be given explicitly (`CONFIRM025_MODE`); anything else refuses.

**`rehearsal`** — the post-install confirmation rehearsal. It never calls `confirm`, never runs the patch-path check,
allows no forward of any kind, and writes nothing in the repository. It runs:
- the provenance preflight:
  - N1: `chdir` to the repository root, assert `cwd`, then HEAD, `origin/main` and the remote, a clean tree, and the
    runner's own `collect_git_state()` from that root;
  - N2: the environment identity with the lock run's record;
  - the installed lock and preregistration hashes;
  - the direct module binding (the lock's blob, the git blob at HEAD, the imported file's blob, `cr.own_blob()`);
  - the full `cr.validate_lock`, and scientific-path immutability;
  - the 020 runtime and version check;
  - the results state;
  - manifest freshness and the spent-set collisions;
  - confirm eligibility (`runner._state_for("confirm")`, read-only);
- then a weights-only I7′:
  - the geometry recomputed and compared with the installed lock bit for bit;
  - the model digests, and the combined control and vector digests;
  - the neutrality of the random controls on the actual float32 patches;
  - the digest → (cue, condition) map production uses to tag every run.

**`smoke`** — the pre-confirm guard smoke test (engineering only; version 2 of this launcher added it; version 3 fixes
its refusal loop, which reused the module-level name `error` and so crashed the record write). It runs the
same preflight and installs exactly the production guards, with three spent reference prompts in place of the four
patch-path keys (`SMOKE_KEYS`: one per template family, each in 020's ledger and none a patch-path or manifest key).
Through the sanctioned wrappers it runs one plain capture and one θ = 0 patched run (the model's own row) per key. It
compares every logit and captured slice bit for bit with the unguarded reference process (`smoke_reference.py`),
checks that every sealed `forward` keeps its original `inspect.signature`, and checks that each unsanctioned route is
refused before any forward. No ℓ, MSE, A, B, G, D_attn, Level-1, Δx3 or Δc is computed; no rotated vector is used;
`confirm` and the patch-path check are never called; nothing in the repository is written.

**`production`** — the one confirm. It additionally requires `CONFIRM025_LOCK=<installed lock content sha256>` and
creates an exclusive sentinel, so a second launch refuses: confirm is never retried. After the same preflight:
- **N3.** It allows a forward only inside a sanctioned call; everything else is refused and logged. The sanctioned
  calls are:
  - the four spent patch-path keys: one plain capture each, and one θ = 0 patched run with the model's own row, and
    only before any fresh run;
  - fresh runs, only once the 90,720-key ledger is on disk: each run's replacement vector must be one of the lock's
    patched vectors for that prompt's cue (a digest map built from I7′'s own recomputation), and each tagged key
    runs once.

  The guarded routes are `torch.nn.Module.__call__`, every module instance's `forward`, the class-level `forward` of
  every module class in the model, and the model's `run_with_hooks`, `run_with_cache` and `generate`. The capture and
  intervention entry points and all their aliases are wrapped by identity. The wrappers pass through unchanged inside
  a sanctioned call (`functools.wraps`, so `inspect.signature` still sees the originals).
- **N4.** An audit hook allows repository writes only to `outputs/experiment-025/results.json` (through its
  `.results-*.json` temporary files and one rename), `stage2-measurements.pt` and `stage2-partial.pt`. It blocks
  every other write, delete, rename, link and mkdir, and every read of 023's or 024's outputs or 022's calibration
  table.
- **Ledger isolation.** Every state write is checked: no spent patch-path key and no untagged key may enter the ledger.
- **Logging.** Every fresh run is appended, with its tagged key, to `production_runs.jsonl` here. The spent runs are
  recorded separately.
- **The order is the runner's own:** I7′ → the patch-path check → the ledger → stage 2 → the save → accounting → C →
  the gates → the scores → the re-check → the one atomic result write → the descriptives. An incident is written
  before any partial save. The launcher calls `runner.confirm()` once and records whatever happens.
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
OUTPUTS = (ROOT / "outputs/experiment-025").resolve()
MODE = os.environ.get("CONFIRM025_MODE", "")
if MODE not in ("rehearsal", "production", "smoke"):
    raise SystemExit("set CONFIRM025_MODE to 'rehearsal', 'smoke' or 'production'")
LAUNCHER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LOCK_RUN_RECORD = ROOT / "experiments/025-nounness-direction-intervention/evidence/lock-2026-09-26/lock_run.json"
EXPECTED = {"lock_file": "1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be", "lock_content": "8e5300de76357ff5df9a3bff1473a85a79d2a3677769b18921bba8e5fb40c2fb",
            "prereg": "e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919", "state": "8eeb8184faf556211e68579d4682c85f545c877051641afe07dd5eac2b701dad",
            "controls280": "e12581d4be183798b42e2f673670eda711ddaf87359c28df5350c1c806a37315", "vectors840": "30a35c0050421b0b07579ee5827e23386dde95ec777aba8ac36a940551df7ca9"}
SMOKE_KEYS = ("cardinal-009-1|ref|581", "quantifier-009-1|ref|1016", "coordinated-adjective-009-1|ref|581")  # smoke mode only: spent reference prompts

# N1: the working directory is the repository root before anything else runs.
os.chdir(ROOT)
if Path.cwd().resolve() != ROOT:
    raise SystemExit(f"cwd {Path.cwd()} is not the repository root {ROOT}")

events = {"mode": MODE, "load_model": 0, "refused": [], "forbidden_reads": [], "forbidden_writes": [], "output_writes": [], "output_renames": [], "outside_repo_writes": 0,
          "spent_runs": [], "fresh_runs": 0, "sanctioned_forward_calls": 0, "confirm_invocations": 0, "ledger_checks": 0}
ALLOWED_OUTPUTS = {OUTPUTS / "results.json", OUTPUTS / "stage2-measurements.pt", OUTPUTS / "stage2-partial.pt"} if MODE == "production" else set()


def _path(value, dir_fd=None):
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


def _temp_state(path: Path) -> bool:
    return MODE == "production" and path.parent == OUTPUTS and path.name.startswith(".results-") and path.name.endswith(".json")


def _refuse_event(kind: str, detail: str):
    events[kind].append(detail)
    raise PermissionError(f"confirm launcher ({MODE}): {kind}: {detail}")


def _hook(event, args):
    if event == "open":
        path = _path(args[0]) if args else None
        if path is None:
            return
        text = str(path)
        if text.endswith("calibration-table.pt") or "/outputs/experiment-023/" in text or "/outputs/experiment-024/" in text:
            _refuse_event("forbidden_reads", text)
        mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if not writing:
            return
        if _in_repo(path):
            if path in ALLOWED_OUTPUTS or _temp_state(path):
                events["output_writes"].append(path.name)
                return
            _refuse_event("forbidden_writes", text)
        if not _inside(path, HERE):
            events["outside_repo_writes"] += 1
        return
    if event in ("os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir"):
        dir_fd = args[2] if event == "os.mkdir" and len(args) > 2 else args[1] if event in ("os.remove", "os.rmdir", "shutil.rmtree") and len(args) > 1 else None
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
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
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


SNAPSHOT = Path(hf_constants.HF_HUB_CACHE) / "models--EleutherAI--pythia-70m-deduped" / "snapshots" / models.PYTHIA_70M.revision
environment = {
    "head": _git("rev-parse", "HEAD"), "repository": str(ROOT), "cwd": str(Path.cwd().resolve()), "launcher_path": str(Path(__file__).resolve()),
    "launcher_sha256": LAUNCHER_SHA256, "mode": MODE, "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "torch": torch.__version__, "transformers": transformers.__version__, "transformer_lens": importlib.metadata.version("transformer-lens"),
    "huggingface_hub": huggingface_hub.__version__, "numpy": numpy.__version__, "torch_num_threads": torch.get_num_threads(),
    "model": {"model_id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision, "device": models.PYTHIA_70M.device, "dtype": models.PYTHIA_70M.dtype},
    "tokenizer_revision": models.PYTHIA_70M.revision,
    "checkpoint": {"snapshot": str(SNAPSHOT), "files": {p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size} for p in sorted(SNAPSHOT.iterdir()) if p.is_file()}},
    "env_flags": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                 "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES", "VIRTUAL_ENV", "CONFIRM025_MODE")},
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
    head = _git("rev-parse", "HEAD")
    need("N1_head_origin_remote_equal", head == _git("rev-parse", "origin/main") == _git("ls-remote", "origin", "refs/heads/main").split()[0], head)
    need("N1_tree_clean", _git("status", "--porcelain", "--untracked-files=all") == "")
    git_state = collect_git_state()  # the runner's own provenance call, from the asserted root
    need("N1_runner_git_state_at_root", git_state == {"commit": head, "dirty": False}, git_state)
    recorded = json.loads(LOCK_RUN_RECORD.read_text())["environment"]
    identity = {key: (environment[key], recorded[key]) for key in ("python", "executable", "platform", "torch", "transformers", "transformer_lens", "huggingface_hub", "numpy",
                                                                    "torch_num_threads", "model", "tokenizer_revision")}
    identity["checkpoint_files"] = (environment["checkpoint"]["files"], recorded["checkpoint"]["files"])
    differing = sorted(key for key, (now, then) in identity.items() if now != then)
    need("N2_environment_identity_with_lock_run", not differing, differing)

    lock_bytes, prereg_bytes = (ROOT / cr.LOCK_RELATIVE_PATH).read_bytes(), (ROOT / cr.PREREGISTRATION_RELATIVE_PATH).read_bytes()
    lock = json.loads(lock_bytes)
    need("installed_lock_hashes", hashlib.sha256(lock_bytes).hexdigest() == EXPECTED["lock_file"] and lock["content_sha256"] == EXPECTED["lock_content"] == rc.content_digest(lock))
    need("installed_preregistration_hash", hashlib.sha256(prereg_bytes).hexdigest() == EXPECTED["prereg"] and cr.render_preregistration(lock).encode("utf-8") == prereg_bytes)
    if MODE == "production":
        need("production_token_names_the_installed_lock", os.environ.get("CONFIRM025_LOCK") == lock["content_sha256"])
    module_path = (ROOT / "src/neural_decompiler/cue_rotation.py").resolve()
    blobs = {"lock": lock["module"]["blob"], "git_head": _git("rev-parse", "HEAD:src/neural_decompiler/cue_rotation.py"),
             "imported_file": rc.program_blob_sha1(Path(cr.__file__)), "own_blob": cr.own_blob()}
    need("module_binding_cr", len(set(blobs.values())) == 1 and Path(cr.__file__).resolve() == module_path and lock["module"]["path"] == "src/neural_decompiler/cue_rotation.py", blobs)
    rr_blobs = {"lock": lock["module_blobs"]["readout_routing.py"], "git_head": _git("rev-parse", "HEAD:src/neural_decompiler/readout_routing.py"),
                "imported_file": rc.program_blob_sha1(Path(rr.__file__))}
    need("module_binding_rr", len(set(rr_blobs.values())) == 1, rr_blobs)
    need("all_pinned_blobs", cr.assert_frozen_blobs() == cr.FROZEN_BLOBS == lock["module_blobs"])

    spec = importlib.util.spec_from_file_location("experiment_025_runner", ROOT / f"{cr.EXPERIMENT_DIR}/run.py")
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
    need("production_configuration", runner.config is cr.PRODUCTION and Path(runner.root).resolve() == ROOT)
    base = runner._base()
    confirmation, confirmation_sha = runner._confirmation(base)
    state = rd.load_results_state(runner.results_path)
    need("results_state_is_the_lock_state", state["state_sha256"] == EXPECTED["state"] and state["phases"]["lock"]["status"] == "complete"
         and state["phases"]["confirm"] == {"status": "not_started"} and state["confirmation"] is None, state["state_sha256"])
    need("ledger_empty", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
    changed = runner.changed_paths(lock["protocol_code_commit"])
    need("scientific_paths_unchanged_since_lock", changed is not None and cr.scientific_changes(changed) == [], changed)
    placeholder = cr.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
    cr.validate_lock(lock, state=state, digests=base.digests, config=runner.config, confirmation=confirmation, confirmation_file_sha256=confirmation_sha,
                     dependencies=placeholder, noun_keys=runner._noun_keys(base.inputs), preregistration_text=prereg_bytes.decode("utf-8"), git_state=git_state,
                     tracked=all(runner.tracked(ROOT / p) for p in (cr.LOCK_RELATIVE_PATH, cr.PREREGISTRATION_RELATIVE_PATH)), changed_paths=changed)
    checks["full_validate_lock"] = True
    need("runtime_and_versions_equal_020_explore", runner._check_runtime(base.inputs.closure) is not None)
    tagged, untagged = confirmation.tagged_keys(), confirmation.manifest_keys()
    need("manifest_fresh_90720", len(tagged) == 90720 and lock["confirmation_025"]["manifest_sha256"] == pm.sha256_text(pm.canonical_json(confirmation.manifest())))
    need("zero_spent_collisions", len(base.forbidden) == 43632 and not (tagged | untagged) & base.forbidden)
    need("patch_path_keys_spent_and_outside_manifest", set(cr.PATCH_PATH_SPENT_KEYS) <= base.forbidden and not set(cr.PATCH_PATH_SPENT_KEYS) & (tagged | untagged)
         and lock["patch_path_spent_keys"] == list(cr.PATCH_PATH_SPENT_KEYS))
    eligible_state = runner._state_for("confirm", base.digests)  # read-only: raises if confirm is not allowed
    need("confirm_eligible_exactly_once", eligible_state["phases"]["confirm"]["status"] == "not_started" and not eligible_state["phases"]["confirm"].get("incidents"))
    need("only_the_lock_outputs_exist", sorted(p.name for p in OUTPUTS.iterdir()) == ["candidate-lock.json", "candidate-preregistration.md", "results.json"])
    return {"checks": checks, "runner": runner, "base": base, "confirmation": confirmation, "lock": lock, "logs": logs}


# ---------------------------------------------------------------------------
# Guards.

inside: list[str] = []  # the sanctioned call in progress, if any
vector_map: dict[str, tuple[int, str]] = {}  # sha256 of a patched float32 vector -> (cue token id, condition)
state_flags = {"ledger_verified": False}
original_call = torch.nn.Module.__call__


def _deny(name: str):
    events["refused"].append(name)
    raise RuntimeError(f"{name} refused by the confirm launcher ({MODE})")


def _guarded_call(self, *args, **kwargs):
    if not inside:
        _deny(f"torch.nn.Module.__call__ outside a sanctioned call ({type(self).__name__})")
    events["sanctioned_forward_calls"] += 1
    return original_call(self, *args, **kwargs)


torch.nn.Module.__call__ = _guarded_call
ENTRY = {id(capture.run_capture): ("capture.run_capture", capture.run_capture), id(interventions.run_interventions): ("interventions.run_interventions", interventions.run_interventions),
         id(pm.capture_prompt): ("pm.capture_prompt", pm.capture_prompt), id(pm.run_patched): ("pm.run_patched", pm.run_patched)}
spent_keys = set(SMOKE_KEYS) if MODE == "smoke" else set(cr.PATCH_PATH_SPENT_KEYS)
spent_done: dict[str, int] = {}
fresh_done: set[str] = set()
run_log = open(HERE / "production_runs.jsonl", "a", buffering=1, encoding="utf-8") if MODE == "production" else None


def _low_level(name, original):
    @functools.wraps(original)
    def guarded(*args, **kwargs):
        if not inside:
            _deny(f"{name} outside a sanctioned call")
        return original(*args, **kwargs)

    return guarded


def _vector_digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().to("cpu", torch.float32).contiguous().reshape(-1).numpy().tobytes()).hexdigest()


def _ledger_on_disk() -> bool:
    if not state_flags["ledger_verified"]:
        state = rd.load_results_state(OUTPUTS / "results.json")
        state_flags["ledger_verified"] = len(state["executed_prompt_keys"]) == 90720 and not set(state["executed_prompt_keys"]) & spent_keys
    return state_flags["ledger_verified"]


def _guarded_capture_prompt(original):
    @functools.wraps(original)
    def guarded(model, prompt, sites):
        if MODE not in ("production", "smoke") or prompt.key not in spent_keys or events["fresh_runs"] or spent_done.get(("plain", prompt.key), 0):
            _deny(f"pm.capture_prompt({prompt.key})")
        spent_done[("plain", prompt.key)] = 1
        events["spent_runs"].append({"kind": "plain", "key": prompt.key})
        inside.append(f"spent-plain {prompt.key}")
        try:
            return original(model, prompt, sites)
        finally:
            inside.pop()

    return guarded


def _guarded_run_patched(original):
    @functools.wraps(original)
    def guarded(model, prompt, replacements, sources, *, capture_sites=()):
        if MODE not in ("production", "smoke") or set(replacements) != {("EMBED", prompt.p_c)}:
            _deny(f"pm.run_patched({prompt.key})")
        vector = replacements[("EMBED", prompt.p_c)]
        if prompt.key in spent_keys:
            own_row = model.embed.W_E[int(prompt.cue_token_id)].detach().to("cpu", torch.float32)
            if events["fresh_runs"] or spent_done.get(("patched", prompt.key), 0) or not torch.equal(vector.detach().to("cpu", torch.float32).reshape(-1), own_row):
                _deny(f"pm.run_patched spent {prompt.key}")
            spent_done[("patched", prompt.key)] = 1
            events["spent_runs"].append({"kind": "patched-theta0", "key": prompt.key})
            label = f"spent-patched {prompt.key}"
        else:
            found = vector_map.get(_vector_digest(vector))
            if found is None or found[0] != int(prompt.cue_token_id) or not _ledger_on_disk():
                _deny(f"pm.run_patched fresh {prompt.key}: vector {'unknown' if found is None else found} or ledger not on disk")
            tagged_key = f"{prompt.key}|{found[1]}"
            if tagged_key in fresh_done:
                _deny(f"pm.run_patched fresh {tagged_key} a second time")
            fresh_done.add(tagged_key)
            events["fresh_runs"] += 1
            run_log.write(json.dumps({"n": events["fresh_runs"], "key": tagged_key}) + "\n")
            label = f"fresh {tagged_key}"
        inside.append(label)
        try:
            return original(model, prompt, replacements, sources, capture_sites=capture_sites)
        finally:
            inside.pop()

    return guarded


def _sweep_aliases() -> list[str]:
    replaced = []
    wrappers = {}
    for key, (name, original) in ENTRY.items():
        if name == "pm.capture_prompt":
            wrappers[key] = _guarded_capture_prompt(original)
        elif name == "pm.run_patched":
            wrappers[key] = _guarded_run_patched(original)
        else:
            wrappers[key] = _low_level(name, original)
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name.startswith("neural_decompiler") or module_name.startswith("experiment_025")):
            continue
        for attribute, value in list(vars(module).items()):
            if id(value) in ENTRY and value is ENTRY[id(value)][1]:
                setattr(module, attribute, wrappers[id(value)])
                replaced.append(f"{module_name}.{attribute}")
    return replaced


def _seal_model(model) -> dict:
    """Every module instance's forward and every module class's forward pass through only inside a sanctioned call;
    the model's hook, cache and generate routes likewise. functools.wraps keeps the original signatures visible."""
    sealed = {"instances": 0, "classes": [], "original_instance_signatures": {}, "wrapped_instance_signatures": {}, "original_class_signatures": {},
              "wrapped_class_signatures": {}}

    def signature(obj):
        try:
            return str(inspect.signature(obj))
        except (TypeError, ValueError):
            return None

    for name, module in model.named_modules():
        original = module.forward
        sealed["original_instance_signatures"][name or "<root>"] = signature(original)

        def make(original=original, name=name):
            @functools.wraps(original)
            def checked(*args, **kwargs):
                if not inside:
                    _deny(f"{name or 'model'}.forward outside a sanctioned call")
                return original(*args, **kwargs)

            return checked

        object.__setattr__(module, "forward", make())
        sealed["wrapped_instance_signatures"][name or "<root>"] = signature(module.forward)
        sealed["instances"] += 1
    classes = {type(module) for module in model.modules()}
    for cls in sorted(classes, key=lambda c: c.__qualname__):
        for method in ("forward", "run_with_hooks", "run_with_cache", "generate"):
            if method in cls.__dict__ and callable(cls.__dict__[method]):
                original = cls.__dict__[method]

                def make(original=original, label=f"{cls.__qualname__}.{method}"):
                    @functools.wraps(original)
                    def checked(*args, **kwargs):
                        if not inside:
                            _deny(f"{label} outside a sanctioned call")
                        return original(*args, **kwargs)

                    return checked

                sealed["original_class_signatures"][f"{cls.__qualname__}.{method}"] = signature(original)
                setattr(cls, method, make())
                sealed["wrapped_class_signatures"][f"{cls.__qualname__}.{method}"] = signature(cls.__dict__[method])
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
if MODE in ("rehearsal", "smoke"):
    for name in ("measure_rotated", "stage_two", "patch_path_check", "identity_gates", "level1_gates", "run_gates", "recompute_c", "routing_distance", "per_cue", "statistics",
                 "descriptives", "ladder", "vector_factors"):
        setattr(cr, name, lambda *a, _n=name, **k: _deny(f"cr.{_n} in the rehearsal"))

# ---------------------------------------------------------------------------
# Run.

record: dict = {"mode": MODE, "environment": environment, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
status, error = None, None
sentinel = HERE / "confirm.sentinel"
try:
    if MODE == "production":
        fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # a second production launch refuses here: never retried
        os.write(fd, LAUNCHER_SHA256.encode())
        os.close(fd)
    pre = preflight()
    aliases += _sweep_aliases()  # the runner's namespace, imported during the preflight
    record["preflight"] = pre["checks"]
    runner, base, confirmation, lock = pre["runner"], pre["base"], pre["confirmation"], pre["lock"]
    original_geometry_block = cr.geometry_block

    def recording_geometry_block(*args, **kwargs):
        out = original_geometry_block(*args, **kwargs)
        for token_id, vectors in out["vectors32"].items():
            for condition, vector in vectors.items():
                vector_map[_vector_digest(vector)] = (int(token_id), condition)
        return out

    cr.geometry_block = recording_geometry_block
    if MODE == "rehearsal":
        model = models.load_model(models.PYTHIA_70M)
        progs = ul.ModelPrograms.from_model(model, base.inputs)
        W_E = progs.weights.W_E
        rehearsal = {"parameters_sha256_equal": rr.parameters_digest(model) == lock["dependencies"]["model"]["parameters_sha256"],
                     "embedding_sha256_equal": rr.embedding_digest(W_E) == lock["dependencies"]["model"]["embedding_sha256"]}
        recomputed = cr.geometry_block(W_E, base.inherited["calibration"], base.inputs.pool, confirmation.tokens, cr.PRODUCTION)
        rehearsal["I7_bitwise"] = cr.verify_geometry_against_lock(recomputed["block"], lock)
        d = cr.direction(W_E, base.inherited["calibration"])
        p_hat = cr.plurality_direction(W_E, base.inputs.pool)
        controls, vectors32, neutral32, neutral64, d_dot = [], [], 0.0, 0.0, 0.0
        for token in confirmation.tokens:
            entry = cr.cue_vectors(W_E, d, p_hat, int(token["token_id"]), cr.PRODUCTION)
            controls.append(torch.stack(entry["controls"]))
            vectors32.append(torch.stack([entry["vectors32"][c] for c in cr.PRODUCTION.conditions]))
            for j, u in enumerate(entry["controls"], start=1):
                p, m = f"rand{j}+0.32", f"rand{j}-0.32"
                neutral32 = max(neutral32, abs(cr.score(d, entry["vectors32"][p].double()) - cr.score(d, entry["vectors32"][m].double())))
                neutral64 = max(neutral64, abs(cr.score(d, entry["vectors64"][p]) - cr.score(d, entry["vectors64"][m])))
                d_dot = max(d_dot, abs(float(d @ u)))
        rehearsal["direction_sha256_equal"] = rc.tensor_digest(d) == lock["geometry"]["direction_sha256"]
        rehearsal["controls280_sha256"] = rc.tensor_digest(torch.stack(controls))
        rehearsal["vectors840_sha256"] = rc.tensor_digest(torch.stack(vectors32))
        rehearsal["combined_digests_equal"] = (rehearsal["controls280_sha256"], rehearsal["vectors840_sha256"]) == (EXPECTED["controls280"], EXPECTED["vectors840"])
        rehearsal["random_control_neutrality"] = {"float32_max": neutral32, "float64_max": neutral64, "d_dot_u_max": d_dot, "within_1e-6": neutral32 <= 1e-6}
        rehearsal["tolerances_equal"] = lock["tolerances"] == cr.TOLERANCES and lock["geometry"]["tolerances"] == recomputed["block"]["tolerances"]
        spent_rows = {_vector_digest(W_E[int(key.split("|")[2])]) for key in cr.PATCH_PATH_SPENT_KEYS}
        rehearsal["run_tag_map"] = {"entries": len(vector_map), "unique_for_840": len(vector_map) == 840, "spent_rows_not_in_map": not spent_rows & set(vector_map)}
        rehearsal["model_loads"] = events["load_model"]
        rehearsal["sealed"] = {"instances": sealing.get("instances"), "class_methods": len(sealing.get("classes", []))}
        record["rehearsal"] = rehearsal
        ok = (rehearsal["parameters_sha256_equal"] and rehearsal["embedding_sha256_equal"] and rehearsal["I7_bitwise"] == {"bitwise_equal": True, "differing": []}
              and rehearsal["direction_sha256_equal"] and rehearsal["combined_digests_equal"] and rehearsal["random_control_neutrality"]["within_1e-6"]
              and rehearsal["tolerances_equal"] and rehearsal["run_tag_map"]["unique_for_840"] and rehearsal["run_tag_map"]["spent_rows_not_in_map"])
        status = 0 if ok else 1
    elif MODE == "smoke":
        reference = json.loads((HERE / "smoke_reference.json").read_text())
        reference_bytes = (HERE / "smoke_reference.pt").read_bytes()
        if reference["head"] != environment["head"] or reference["keys"] != list(SMOKE_KEYS) or not reference["outputs_unchanged"] \
                or hashlib.sha256(reference_bytes).hexdigest() != reference["reference_tensors_sha256"]:
            raise RuntimeError("the unguarded reference is not this commit's, not these keys', or not intact")
        ref_tensors = torch.load(HERE / "smoke_reference.pt")
        model = models.load_model(models.PYTHIA_70M)  # counted; every forward route sealed right after the load
        W_E32 = pm.Weights.from_model(model).W_E
        frames = {frame.frame_id: frame for frame in base.inputs.pool.frames}
        smoke = {"keys": list(SMOKE_KEYS), "per_key": {}}
        for key in SMOKE_KEYS:
            frame_id, word, token_id = key.split("|")
            frame = frames[frame_id]
            prompt = pm.Prompt(frame, int(token_id), word)
            site = ("EMBED", frame.p_c)
            sites = ul.measured_sites(frame) + cr.attention_sites(frame) + [site, ("RESID_PRE.L0", frame.p_c)]
            plain = pm.capture_prompt(model, prompt, sites)  # through the sanctioned, guarded route
            patched = pm.run_patched(model, prompt, {site: W_E32[int(token_id)].detach().reshape(1, 1, -1).to(torch.float32)}, {site: pm.ReplacementSource.DIRECT},
                                     capture_sites=sites)
            result = {}
            for kind, run in (("plain", plain), ("patched", patched)):
                labels = sorted(run.slices)
                ref_labels = sorted(k[len(f"{key}/{kind}/"):] for k in ref_tensors if k.startswith(f"{key}/{kind}/") and not k.endswith("/logits"))
                slice_digests = {label: rc.tensor_digest(run.slices[label].double()) for label in labels}
                result[kind] = {"logits_equal_unguarded": bool(torch.equal(run.logits, ref_tensors[f"{key}/{kind}/logits"])), "same_sites_as_unguarded": labels == ref_labels,
                                "slices_equal_unguarded": labels == ref_labels and all(torch.equal(run.slices[label], ref_tensors[f"{key}/{kind}/{label}"]) for label in labels),
                                "digests_equal_unguarded": rc.tensor_digest(run.logits.double()) == reference["digests"][key][kind]["logits_sha256"]
                                and slice_digests == reference["digests"][key][kind]["slices_sha256"],
                                "shapes_equal_unguarded": {label: list(run.slices[label].shape) for label in labels} == reference["digests"][key][kind]["shapes"],
                                "dtypes": sorted({str(v.dtype) for v in run.slices.values()}), "logits_sha256": rc.tensor_digest(run.logits.double())}
            result["plain_equals_patched_theta0"] = bool(torch.equal(plain.logits, patched.logits)) and set(plain.slices) == set(patched.slices) \
                and all(torch.equal(plain.slices[label], patched.slices[label]) for label in plain.slices)
            result["embed_equals_resid_pre0"] = all(bool(torch.equal(run.vector(site), run.vector(("RESID_PRE.L0", frame.p_c)))) for run in (plain, patched))
            result["theta0_landed_exactly"] = len(patched.integrity) == 1 and patched.integrity[0]["outside_max_abs_change"] == 0.0 \
                and bool(torch.equal(patched.vector(site).to(torch.float32), W_E32[int(token_id)]))
            smoke["per_key"][key] = result
        smoke["signatures"] = {
            "instance_wrapped_equal_original": sealing["original_instance_signatures"] == sealing["wrapped_instance_signatures"],
            "class_wrapped_equal_original": sealing["original_class_signatures"] == sealing["wrapped_class_signatures"],
            "instance_original_equal_unguarded_process": sealing["original_instance_signatures"] == reference["instance_signatures"],
            "class_original_equal_unguarded_process": sealing["original_class_signatures"] == reference["class_signatures"],
            "instances": len(sealing["wrapped_instance_signatures"]), "class_methods": len(sealing["wrapped_class_signatures"]),
            "unavailable": sum(1 for v in sealing["wrapped_instance_signatures"].values() if v is None)}
        guarded_calls_after_runs = events["sanctioned_forward_calls"]
        dummy = torch.tensor([[1, 2, 3]])
        first_frame = frames[SMOKE_KEYS[0].split("|")[0]]
        first_prompt = pm.Prompt(first_frame, int(SMOKE_KEYS[0].split("|")[2]), "ref")
        synthetic = pm.Prompt(first_frame, 50000, "synthetic")  # neither spent nor in the manifest
        zeros = torch.zeros(1, 1, int(W_E32.shape[1]))
        submodule = next(module for name, module in model.named_modules() if name)
        attempts = {
            "module_call": lambda: model(dummy),
            "root_instance_forward": lambda: model.forward(dummy),
            "root_class_forward": lambda: type(model).forward(model, dummy),
            "run_with_cache": lambda: model.run_with_cache(dummy),
            "run_with_hooks": lambda: model.run_with_hooks(dummy),
            "submodule_instance_forward": lambda: submodule.forward(dummy),
            "capture_prompt_unlisted_key": lambda: pm.capture_prompt(model, synthetic, [("EMBED", first_frame.p_c)]),
            "capture_prompt_second_time": lambda: pm.capture_prompt(model, first_prompt, [("EMBED", first_frame.p_c)]),
            "run_patched_spent_key_not_own_row": lambda: pm.run_patched(model, first_prompt, {("EMBED", first_frame.p_c): zeros}, {("EMBED", first_frame.p_c): pm.ReplacementSource.DIRECT}),
            "run_patched_fresh_route_unmapped": lambda: pm.run_patched(model, synthetic, {("EMBED", first_frame.p_c): zeros}, {("EMBED", first_frame.p_c): pm.ReplacementSource.DIRECT}),
            "low_level_run_capture": lambda: capture.run_capture(model, dummy, None),
            "low_level_run_interventions": lambda: interventions.run_interventions(model, dummy, None),
        }
        refusals = {}
        for name, attempt in attempts.items():
            before = len(events["refused"])
            try:
                attempt()
                refusals[name] = "NOT REFUSED"
            except Exception as refusal:  # the guard's own RuntimeError, recorded in events["refused"] (never `as error`: that name is the run's own)
                refusals[name] = "refused" if len(events["refused"]) == before + 1 and "refused by the confirm launcher" in str(refusal) else f"OTHER {type(refusal).__name__}: {refusal}"
        smoke["refusals"] = refusals
        smoke["forwards_during_refusal_attempts"] = events["sanctioned_forward_calls"] - guarded_calls_after_runs
        smoke["refusals_during_sanctioned_runs"] = len(events["refused"]) - len(attempts)
        smoke["guarded_runs"] = len(events["spent_runs"])
        record["smoke"] = smoke
        ok = all(all(v is True for k, v in r[kind].items() if k.endswith("_unguarded")) for r in smoke["per_key"].values() for kind in ("plain", "patched")) \
            and all(r["plain_equals_patched_theta0"] and r["embed_equals_resid_pre0"] and r["theta0_landed_exactly"] for r in smoke["per_key"].values()) \
            and all(smoke["signatures"][k] for k in ("instance_wrapped_equal_original", "class_wrapped_equal_original", "instance_original_equal_unguarded_process",
                                                    "class_original_equal_unguarded_process")) \
            and all(v == "refused" for v in refusals.values()) and smoke["forwards_during_refusal_attempts"] == 0 and smoke["refusals_during_sanctioned_runs"] == 0 \
            and smoke["guarded_runs"] == 6 and events["fresh_runs"] == 0 and events["load_model"] == 1
        status = 0 if ok else 1
    else:
        original_write = type(runner)._write

        def checked_write(self, state):
            events["ledger_checks"] += 1
            ledger = state.get("executed_prompt_keys") or []
            bad = [key for key in ledger if key in spent_keys or len(key.split("|")) != 4]
            if bad:
                _deny(f"a spent or untagged key in the ledger: {bad[:2]}")
            return original_write(self, state)

        type(runner)._write = checked_write
        events["confirm_invocations"] += 1
        status = runner.confirm()  # the one production confirm; never retried
except BaseException as exc:  # recorded, never retried
    error = f"{type(exc).__name__}: {exc}"
record.update({"ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_status": status, "error": error, "events": events, "aliases": aliases,
               "sealing": {"instances": sealing.get("instances"), "class_methods": sealing.get("classes")},
               "outputs": {p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size} for p in sorted(OUTPUTS.iterdir())} if OUTPUTS.exists() else {},
               "tree_after": _git("status", "--porcelain", "--untracked-files=all"), "head_after": _git("rev-parse", "HEAD")})
if run_log is not None:
    run_log.close()
(HERE / f"{MODE}_run.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
print(json.dumps({k: record[k] for k in ("mode", "started_at", "ended_at", "exit_status", "error")}), flush=True)
print(json.dumps({k: v for k, v in events.items() if k not in ("output_writes",)}, default=str), flush=True)
sys.exit(0 if status == 0 and error is None else 1)
