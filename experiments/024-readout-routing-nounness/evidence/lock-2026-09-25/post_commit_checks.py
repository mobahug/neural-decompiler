"""Post-commit checks of Experiment 024's installed lock (read-only; no model, no tokenizer, no prompt; confirm NOT run).

After the two-file lock artifact commit: the commit's paths; the committed and working bytes; the rendering; the stock
`validate`; confirm's own pre-model path up to and including the full `validate_lock` with tracked-file enforcement
(replicated statement by statement from `Runner.confirm`, without calling it); the paths changed since the lock-run
commit; the direct module-blob assertion; and every lock value the reviewer named, preserved exactly.
"""
import hashlib
import importlib.util
import json
import math
import os
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
    try:
        path = Path(os.fsdecode(args[0])).resolve()
    except OSError:
        return
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
        REFUSED.append(str(path))
        raise PermissionError(f"the post-commit checks may not write into the repository: {path}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

LOCK_RUN = "be74d23d086bd12d894db1f73e6a1b4160aeebd5"
LOCK_PATH, PREREG_PATH = rr.LOCK_RELATIVE_PATH, rr.PREREGISTRATION_RELATIVE_PATH
EXPECTED = {
    "lock_file": "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d",
    "lock_content": "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739",
    "prereg_file": "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54",
    "state": "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d",
    "state_file": "14459cdde6ec2d832c3965d5b0ebaadaf06cef6aa0b0460c3a31b29dd442b262",
    "module_blob": "e6cb37767d1d06c6ff40804a88eab569723afdb5",
    "scores": "a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995",
    "manifest": "fcc437fca9730b8599333eeac95807786570f0d03bd01e49f52daec76ed4e32d",
}
failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


def git(*args, text=True):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=text).stdout


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# -- the commit ------------------------------------------------------------------------------------------------------
head = git("rev-parse", "HEAD").strip()
print(f"HEAD {head}; parent {git('rev-parse', 'HEAD^').strip()}", flush=True)
check("the artifact commit's parent is the lock-run commit be74d23", git("rev-parse", "HEAD^").strip() == LOCK_RUN)
touched = git("show", "--name-only", "--format=", "HEAD").split()
check("the commit touches exactly the two lock paths", sorted(touched) == sorted([LOCK_PATH, PREREG_PATH]), touched)
check("working tree clean", git("status", "--porcelain", "--untracked-files=all") == "")
committed = {path: git("show", f"HEAD:{path}", text=False) for path in (LOCK_PATH, PREREG_PATH)}
working = {path: (ROOT / path).read_bytes() for path in (LOCK_PATH, PREREG_PATH)}
candidates = {LOCK_PATH: (ROOT / "outputs/experiment-024/candidate-lock.json").read_bytes(),
              PREREG_PATH: (ROOT / "outputs/experiment-024/candidate-preregistration.md").read_bytes()}
for path, expected in ((LOCK_PATH, EXPECTED["lock_file"]), (PREREG_PATH, EXPECTED["prereg_file"])):
    check(f"git show HEAD:{path.rsplit('/', 1)[1]} == working == reviewed candidate, sha256 exact",
          committed[path] == working[path] == candidates[path] and sha(committed[path]) == expected, sha(committed[path]))
lock = json.loads(committed[LOCK_PATH].decode("utf-8"))
check("installed lock content sha256 (stored == recomputed == expected)", lock["content_sha256"] == rc.content_digest(lock) == EXPECTED["lock_content"])
check("render_preregistration(installed lock) == committed preregistration.md, byte for byte", rr.render_preregistration(lock).encode("utf-8") == committed[PREREG_PATH])

# -- the stock validate -------------------------------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)


def refuse(*args, **kwargs):
    raise RuntimeError("no model or tokenizer in the post-commit checks")


logs: list[str] = []
runner = runner_module.Runner(log=logs.append, model_loader=refuse, tokenizer_loader=refuse)
check("configuration is the frozen production one", runner.config is rr.PRODUCTION)
check("stock validate passes (and verifies the installed lock and preregistration)", runner.validate() == 0 and "lock and preregistration verified" in logs[-1], logs[-1])

# -- confirm's pre-model path, statement by statement (Runner.confirm is never called) ----------------------------------
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
state = runner._state_for("confirm", base.digests)  # a clean tree at a commit; the recorded inputs, versions and configuration; the phase rule
check("confirm's state preconditions and phase rule pass (_state_for('confirm'))", True)
rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.manifest_keys(), "Experiment 024's ledger before confirm")
check("the ledger is isolated from the spent keys and the manifest (and is empty)", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
paths = {"lock": ROOT / LOCK_PATH, "preregistration": ROOT / PREREG_PATH}
installed_lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
record, record_sha = runner._installed_record(state, base.digests)
git_state = runner.git_state()
placeholder = rr.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
tracked = all(runner.tracked(path) for path in paths.values())
changed = runner.changed_paths(installed_lock["protocol_code_commit"])
arguments = dict(state=state, digests=base.digests, config=runner.config, record=record, record_file_sha256=record_sha, confirmation=confirmation,
                 confirmation_file_sha256=confirmation_sha, dependencies=placeholder, noun_keys=runner._noun_keys(base.inputs),
                 preregistration_text=paths["preregistration"].read_text(encoding="utf-8"), git_state=git_state)
rr.validate_lock(installed_lock, **arguments, tracked=tracked, changed_paths=changed)
check("full validate_lock passes at the new commit (tracked, clean, lock commit an ancestor, no scientific change)", tracked and not git_state["dirty"], git_state)
try:
    rr.validate_lock(installed_lock, **arguments, tracked=False, changed_paths=changed)
    check("tracked-file enforcement is live (untracked lock files are refused)", False)
except rr.PhaseError as error:
    check("tracked-file enforcement is live (untracked lock files are refused)", "tracked" in str(error), error)
check("the lock-run commit is be74d23", installed_lock["protocol_code_commit"] == LOCK_RUN)
check("the only paths changed since be74d23 are the two installed lock files", sorted(changed or []) == sorted([LOCK_PATH, PREREG_PATH]) and rr.scientific_changes(changed) == [], changed)
runtime = runner._check_runtime(base.inputs.closure)
check("confirm's runtime check passes (020's explore runtime and versions)", True, runtime)

# -- the direct module-blob assertion (the reviewer's note 1; the future confirm launcher repeats it before the load) ----
module_file = Path(rr.__file__).resolve()
own = blob_sha1(module_file.read_bytes())
at_head = git("rev-parse", "HEAD:src/neural_decompiler/readout_routing.py").strip()
check("rr is imported from the repository file", module_file == (ROOT / "src/neural_decompiler/readout_routing.py").resolve(), module_file)
check("module blob: own sha1 == rr.own_blob() == git HEAD == installed lock module.blob == e6cb3776…",
      own == rr.own_blob() == at_head == installed_lock["module"]["blob"] == EXPECTED["module_blob"] and installed_lock["module"]["path"] == "src/neural_decompiler/readout_routing.py",
      {"expected": installed_lock["module"]["blob"], "observed": own})

# -- the lock values, preserved exactly ------------------------------------------------------------------------------
cues = installed_lock["fresh"]["cues"]
scores = [cue["nounness"] for cue in cues]
check("40 scores in the frozen order", [(c["word"], c["token_id"], c["class"]) for c in cues] == [(t["word"], t["token_id"], t["class"]) for t in json.loads(
    (ROOT / rr.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))["cues"]] and len(cues) == 40)
check("40-score digest a703ac16…", installed_lock["fresh"]["scores_sha256"] == rc.tensor_digest(torch.tensor(scores, dtype=torch.float64)) == EXPECTED["scores"])
maximum = installed_lock["fresh"]["maximum_calibration_score"]
check("calibration maximum 0.13502027836111233 (== the record's largest leave-one-out score)",
      maximum == 0.13502027836111233 == max(entry["nounness_loo"] for entry in record["calibration_cues"]), maximum)
check("above-maximum flags == score > maximum", all(c["above_calibration_maximum"] == (c["nounness"] > maximum) for c in cues))
e_above = [c["word"] for c in cues if c["class"] == "E" and c["above_calibration_maximum"]]
check("exactly 5 of 8 E above it: apple, horse, doctor, poet, dragon", e_above == ["apple", "horse", "doctor", "poet", "dragon"]
      and installed_lock["fresh"]["extrapolation"]["E_above_maximum"] == 5, e_above)
e_scores = [c["nounness"] for c in cues if c["class"] == "E"]
n_scores = [c["nounness"] for c in cues if c["class"] == "N"]
check("every E score above every N score", min(e_scores) > max(n_scores), f"{min(e_scores):+.6f} > {max(n_scores):+.6f}")
check("predictions == the frozen line", all(c["predicted_log_mse"] == rr.predict(record["line"], c["nounness"]) for c in cues))
primary = installed_lock["primary"]
check("primary: F_rho 0.24411074612857814, null 0.3136960600375234, threshold 0.3136960600375234 bound by the null",
      primary["F_rho"] == 0.24411074612857814 and primary["null_975"] == 0.3136960600375234
      and primary["effective_threshold"] == {"value": 0.3136960600375234, "binds": "null_975"}, primary["effective_threshold"])
t = primary["effective_threshold"]["value"]
check("threshold rule: ρ = t passes, the next float below fails (pure rule, no data)",
      rr.classify_primary(t, primary["F_rho"], primary["null_975"]) == "PASS" and rr.classify_primary(math.nextafter(t, -math.inf), primary["F_rho"], primary["null_975"]) == "GUARD_FAILURE")
guard = installed_lock["guard"]["spec"]
check("E–N guard: 8 + 8, 12,870 assignments, K ≤ 321 passes, ties against the guard, exact arithmetic",
      (guard["n_E"], guard["n_N"], guard["assignments"], guard["max_upper"]) == (8, 8, math.comb(16, 8), 321) and guard == rr.PRODUCTION.guard.to_json()
      and "ties count against the guard" in guard["rule"] and "exact" in guard["arithmetic"], guard["rule"])
check("E–N decision: K = 321 PASS, K = 322 FAIL, K = 12,870 (all tied) FAIL (pure rule, no data)",
      [rr.en_decision(k, rr.PRODUCTION.guard) for k in (321, 322, 12_870)] == ["PASS", "FAIL", "FAIL"])
check("guard units: the frozen 8 E then 8 N", [u["word"] for u in installed_lock["guard"]["units"]["E"]] == list(rr.EXPECTED_PICKS["ordinary"])
      and [u["word"] for u in installed_lock["guard"]["units"]["N"]] == list(rr.EXPECTED_PICKS["N"]))
REVIEWED_TABLE = [
    ["an incident", "—", "no result is written"],
    ["NOT_INTERPRETABLE", "any", "NOT_INTERPRETABLE"],
    ["GUARD_FAILURE or ENVELOPE_ONLY_FAILURE", "any (descriptive only)", "NOUNNESS_PREDICTION_NOT_ESTABLISHED"],
    ["PASS", "FAIL or NOT_INTERPRETABLE", "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED"],
    ["PASS", "PASS", "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS"],
]
check("outcome hierarchy exactly as reviewed (the lock's table == rr.OUTCOME_TABLE == the reviewed rows)",
      installed_lock["outcome"]["table"] == [list(row) for row in rr.OUTCOME_TABLE] == REVIEWED_TABLE and installed_lock["semantics"] == rr.SEMANTICS)


def reviewed(primary_result, guard_result):
    if primary_result == "NOT_INTERPRETABLE":
        return "NOT_INTERPRETABLE"
    if primary_result in ("GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE"):
        return "NOUNNESS_PREDICTION_NOT_ESTABLISHED"
    return REVIEWED_TABLE[4][2] if guard_result == "PASS" else REVIEWED_TABLE[3][2]


pairs = [(p, g) for p in rr.PRIMARY_RESULTS for g in rr.GUARD_RESULTS]
check("rr.outcome == the reviewed hierarchy on all 12 (primary, guard) pairs", len(pairs) == 12 and all(rr.outcome(p, g) == reviewed(p, g) for p, g in pairs))
manifest = confirmation.manifest()
keys = manifest["S2-TARGET"]
check("manifest: 4,320 unique keys, digest fcc437fc… (lock == recomputed from the committed freeze)",
      len(keys) == len(set(keys)) == 4_320 and installed_lock["confirmation_024"]["manifest_sha256"] == pm.sha256_text(pm.canonical_json(manifest)) == EXPECTED["manifest"])
check("zero overlap with the 39,312 spent keys", len(base.forbidden) == 39_312 and not (set(keys) & base.forbidden))

# -- zero fresh execution ---------------------------------------------------------------------------------------------
out = ROOT / "outputs/experiment-024"
check("results state unchanged by the install (digest and file)", state["state_sha256"] == EXPECTED["state"] and sha((out / "results.json").read_bytes()) == EXPECTED["state_file"])
check("confirm and report not started; no confirmation or report", state["phases"]["confirm"] == {"status": "not_started"} and state["phases"]["report"] == {"status": "not_started"}
      and state["confirmation"] is None and state["report"] is None)
check("zero fresh-key execution: ledger empty; no stage-2 file; no report", state["executed_prompt_keys"] == [] and not (out / "stage2-measurements.pt").exists()
      and not (out / "report.md").exists())
check("no repository write", not REFUSED, REFUSED)
print("FAILURES:", failures if failures else "none", flush=True)
sys.exit(1 if failures else 0)
