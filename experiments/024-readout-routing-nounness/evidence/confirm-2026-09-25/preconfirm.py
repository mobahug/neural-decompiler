"""Pre-confirm check for Experiment 024 (read-only; no model load, no tokenizer, no prompt; confirm is NOT run here).

Before the one production `confirm`: the repository and remote state; the stock `validate`; confirm's own pre-model path
up to and including the full `validate_lock` and the runtime check (replicated statement by statement from
`Runner.confirm`, which is never called); the phase rule; the ledgers; the absence of any fresh measurement or outcome;
the manifest's isolation; the preserved hashes; the direct module-blob assertion; and the environment (threads,
versions, the cached checkpoint). Any mismatch exits non-zero: confirm must not run.
"""
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
REFUSED = []


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if not writing:
        return
    raw = os.fsdecode(args[0])
    if not os.path.isabs(raw):
        return  # dir_fd-relative probes (filelock in the system temp directory); nothing here writes relative to the repository
    path = Path(raw).resolve()
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
        REFUSED.append(str(path))
        raise PermissionError(f"the pre-confirm check may not write into the repository: {path}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

COMMIT = "af160cee0389bcdaf12bcbd92a8616f142777a52"
EXPECTED = {
    "lock_file": "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d",
    "lock_content": "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739",
    "prereg_file": "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54",
    "state": "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d",
    "state_file": "14459cdde6ec2d832c3965d5b0ebaadaf06cef6aa0b0460c3a31b29dd442b262",
    "module_blob": "e6cb37767d1d06c6ff40804a88eab569723afdb5",
    "manifest": "fcc437fca9730b8599333eeac95807786570f0d03bd01e49f52daec76ed4e32d",
    "checkpoint": "3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd",
}
CHECKPOINT = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots" / models.PYTHIA_70M.revision / "model.safetensors"
GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


def git(*args, text=True):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=text, env=GIT_ENV).stdout


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- repository and remote --------------------------------------------------------------------------------------------
head, origin = git("rev-parse", "HEAD").strip(), git("rev-parse", "origin/main").strip()
remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
check("HEAD == origin/main == remote == af160ce", head == origin == remote == COMMIT, f"{head} / {origin} / {remote}")
check("working tree clean, no untracked file", git("status", "--porcelain", "--untracked-files=all") == "")
check("no stash", git("stash", "list") == "")

# -- the stock validate --------------------------------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)


def refuse(*args, **kwargs):
    raise RuntimeError("no model or tokenizer in the pre-confirm check")


logs: list[str] = []
runner = runner_module.Runner(log=logs.append, model_loader=refuse, tokenizer_loader=refuse)
check("configuration is the frozen production one", runner.config is rr.PRODUCTION)
check("stock validate passes (the installed lock and preregistration verified)", runner.validate() == 0 and "lock and preregistration verified" in logs[-1], logs[-1])

# -- confirm's pre-model path, statement by statement (Runner.confirm is never called) ----------------------------------
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
state = runner._state_for("confirm", base.digests)  # clean tree at a commit; recorded inputs, versions, configuration; the phase rule
rr.assert_phase_allowed("confirm", state)
check("the phase rule permits confirm", True)
rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.manifest_keys(), "Experiment 024's ledger before confirm")
paths = {"lock": ROOT / rr.LOCK_RELATIVE_PATH, "preregistration": ROOT / rr.PREREGISTRATION_RELATIVE_PATH}
lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
record, record_sha = runner._installed_record(state, base.digests)
git_state = runner.git_state()
placeholder = rr.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
tracked = all(runner.tracked(path) for path in paths.values())
changed = runner.changed_paths(lock["protocol_code_commit"])
rr.validate_lock(lock, state=state, digests=base.digests, config=runner.config, record=record, record_file_sha256=record_sha, confirmation=confirmation,
                 confirmation_file_sha256=confirmation_sha, dependencies=placeholder, noun_keys=runner._noun_keys(base.inputs),
                 preregistration_text=paths["preregistration"].read_text(encoding="utf-8"), git_state=git_state, tracked=tracked, changed_paths=changed)
check("full validate_lock passes (tracked, clean, no scientific change since be74d23)", tracked and not git_state["dirty"] and rr.scientific_changes(changed) == [],
      f"{len(changed)} changed paths, none scientific")
runtime = runner._check_runtime(base.inputs.closure)
check("confirm's runtime check passes (020's explore runtime and versions)", runtime["torch_num_threads"] == 4, runtime)

# -- phases, ledgers, no fresh measurement or outcome -------------------------------------------------------------------
phases = state["phases"]
check("calibrate and lock complete once, no incident", phases["calibrate"]["status"] == "complete" and phases["lock"]["status"] == "complete"
      and not phases["lock"].get("incidents") and not state["calibration"].get("incidents"))
check("confirm and report not started, no incident", phases["confirm"] == {"status": "not_started"} and phases["report"] == {"status": "not_started"})
check("both ledgers empty", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
out = ROOT / "outputs/experiment-024"
listing = sorted(path.name for path in out.iterdir())
check("no stage2-measurements.pt, no report, no temp file (exactly the five pre-confirm files)",
      listing == ["calibration-arrays.pt", "candidate-calibration.json", "candidate-lock.json", "candidate-preregistration.md", "results.json"], listing)
check("no fresh outcome value: the state's confirmation and report are null", state["confirmation"] is None and state["report"] is None)
fresh_keys = {key for cue in lock["fresh"]["cues"] for key in cue}
check("the lock's fresh cues carry no observed quantity", not fresh_keys & {"mse", "log_mse", "rho", "K", "D_EN", "delta_c", "delta_x3"}, sorted(fresh_keys))
keys = confirmation.manifest()["S2-TARGET"]
check("manifest: 4,320 unique keys, digest fcc437fc…", len(keys) == len(set(keys)) == 4_320 and pm.sha256_text(pm.canonical_json(confirmation.manifest())) == EXPECTED["manifest"])
check("zero collision with the 39,312 spent keys", len(base.forbidden) == 39_312 and not (set(keys) & base.forbidden))

# -- preserved hashes -----------------------------------------------------------------------------------------------------
check("installed lock file sha256", sha(paths["lock"].read_bytes()) == EXPECTED["lock_file"])
check("lock content sha256", lock["content_sha256"] == rc.content_digest(lock) == EXPECTED["lock_content"])
check("preregistration sha256", sha(paths["preregistration"].read_bytes()) == EXPECTED["prereg_file"])
check("results-state digest 58891c24…", state["state_sha256"] == EXPECTED["state"])
check("results file sha256 14459cdd…", sha((out / "results.json").read_bytes()) == EXPECTED["state_file"])

# -- the direct module-blob assertion (the launcher repeats it immediately before the load) -----------------------------
module_path = Path(rr.__file__).resolve()
origin_path = Path(importlib.util.find_spec("neural_decompiler.readout_routing").origin).resolve()
data = module_path.read_bytes()
own = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
hash_object = git("hash-object", "--", str(module_path)).strip()
at_head = git("rev-parse", "HEAD:src/neural_decompiler/readout_routing.py").strip()
check("module blob: lock == rr.own_blob() == own sha1 == git hash-object == git HEAD == e6cb3776…",
      lock["module"]["blob"] == rr.own_blob() == own == hash_object == at_head == EXPECTED["module_blob"], {"expected": lock["module"]["blob"], "observed": rr.own_blob()})
check("rr imported from the repository file", module_path == origin_path == (ROOT / "src/neural_decompiler/readout_routing.py").resolve(), module_path)

# -- the environment -------------------------------------------------------------------------------------------------------
check("HF_HUB_OFFLINE=1 and PYTHONDONTWRITEBYTECODE=1", os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1")
check("4 torch threads", torch.get_num_threads() == 4, torch.get_num_threads())
check("the cached checkpoint of the pinned revision: sha256 == its LFS blob id 3da38833…",
      CHECKPOINT.exists() and sha(CHECKPOINT.read_bytes()) == CHECKPOINT.resolve().name == EXPECTED["checkpoint"], CHECKPOINT)
check("the interpreter is the repository's .venv", Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), sys.prefix)
print("environment:", json.dumps({"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                                  "versions": dict(runner.versions()), "runtime": runtime, "model": {"id": models.PYTHIA_70M.model_id,
                                                                                                     "revision": models.PYTHIA_70M.revision}}), flush=True)
check("no repository write", not REFUSED, REFUSED)
print("FAILURES:", failures if failures else "none", flush=True)
sys.exit(1 if failures else 0)
