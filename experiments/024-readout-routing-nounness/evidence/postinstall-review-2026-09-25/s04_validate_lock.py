"""Item 4: the full validate_lock exactly as Runner.confirm calls it before the model loads (run.py 510-526), replicated
statement by statement WITHOUT calling confirm; then the runtime check; then tracked-file enforcement and other refusals."""
import guard  # noqa: F401
from guard import REPO, Seal, git, summary

import copy
import importlib.util
import json
import os
import sys

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


RUN_PY = os.path.join(REPO, "experiments/024-readout-routing-nounness/run.py")
spec = importlib.util.spec_from_file_location("run024_postinstall_review", RUN_PY)
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)
rr, PhaseError = run.rr, run.PhaseError
seal = Seal()
seal.refuse_captures()


def refuse(spec):
    raise RuntimeError("postinstall review: no model/tokenizer load in item 4")


runner = run.Runner(log=lambda m: print("  LOG:", m), model_loader=refuse, tokenizer_loader=refuse)

# ---- run.py 510-524, statement by statement --------------------------------------------------------------------------
base = runner._base()                                                                                   # 510
confirmation, confirmation_sha = runner._confirmation(base)                                            # 511
state = runner._state_for("confirm", base.digests)                                                     # 512
rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.manifest_keys(),
                          "Experiment 024's ledger before confirm")                                     # 513
paths = {"lock": runner.root / rr.LOCK_RELATIVE_PATH, "preregistration": runner.root / rr.PREREGISTRATION_RELATIVE_PATH}  # 514
if not all(path.exists() for path in paths.values()):                                                  # 515
    raise PhaseError("the committed lock and preregistration must both exist")                          # 516
lock = json.loads(paths["lock"].read_text(encoding="utf-8"))                                           # 517
record, record_sha = runner._installed_record(state, base.digests)                                     # 518
git_state = runner.git_state()                                                                         # 519
placeholder = rr.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")       # 520
tracked_values = {name: runner.tracked(path) for name, path in paths.items()}
changed = runner.changed_paths(lock["protocol_code_commit"])
kwargs = dict(state=state, digests=base.digests, config=runner.config, record=record, record_file_sha256=record_sha, confirmation=confirmation,
              confirmation_file_sha256=confirmation_sha, dependencies=placeholder, noun_keys=runner._noun_keys(base.inputs),
              preregistration_text=paths["preregistration"].read_text(encoding="utf-8"), git_state=git_state,
              tracked=all(runner.tracked(path) for path in paths.values()), changed_paths=runner.changed_paths(lock["protocol_code_commit"]))
rr.validate_lock(lock, **kwargs)                                                                       # 521-524
commit = str(git_state.get("commit"))                                                                  # 525
runtime = runner._check_runtime(base.inputs.closure)                                                   # 526
# ---------------------------------------------------------------------------------------------------------------------
check("validate_lock (full, as confirm calls it) accepted the installed lock", True)
check("the runtime check passed", True, runtime)
head = git("rev-parse", "HEAD").strip()
check("git_state: commit == HEAD, not dirty", git_state == {"commit": head, "dirty": False}, git_state)
check("commit (confirm_commit) == af160ce", commit == "af160cee0389bcdaf12bcbd92a8616f142777a52", commit)
check("tracked(lock) and tracked(preregistration) both True", tracked_values == {"lock": True, "preregistration": True}, tracked_values)
check("changed_paths(lock.protocol_code_commit) is 32 paths, none scientific", changed is not None and len(changed) == 32 and rr.scientific_changes(changed) == [],
      None if changed is None else len(changed))
check("record_sha == installed calibration-v1.json sha256 81fb499e…", record_sha == "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d")
check("confirmation_sha == committed confirmation-v1.json sha256 68510e1b…", confirmation_sha.startswith("68510e1b"), confirmation_sha)
check("forbidden (spent) keys = 39,312", len(base.forbidden) == 39312, len(base.forbidden))
check("state ledger empty (prompt keys, noun keys)", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
recorded = base.inputs.closure["state"]["phases"]["explore"].get("runtime")
print("  runtime now:", runtime)
print("  020 explore runtime:", recorded)
check("placeholder dependencies differ from the lock's only under 'model'",
      sorted(k for k in set(placeholder) | set(lock["dependencies"]) if placeholder.get(k) != lock["dependencies"].get(k)) == ["model"])


def refused(name, **override):
    args = dict(kwargs)
    target = override.pop("_lock", lock)
    args.update(override)
    try:
        rr.validate_lock(target, **args)
    except PhaseError as error:
        check(f"refused: {name}", True, str(error)[:110])
        return
    check(f"refused: {name}", False, "ACCEPTED")


# tracked-file enforcement, live
refused("tracked=False", tracked=False)
candidate_paths = [runner.output("candidate-lock.json"), runner.output("candidate-preregistration.md")]
live = [runner.tracked(p) for p in candidate_paths]
check("live: runner.tracked(candidate files) is False (gitignored, same bytes)", live == [False, False], live)
refused("tracked computed live from the untracked candidate paths", tracked=all(runner.tracked(p) for p in candidate_paths))
# other branches (pure calls on in-memory copies)
refused("dirty tree", git_state={"commit": head, "dirty": True})
refused("lock commit not an ancestor (changed_paths=None)", changed_paths=None)
refused("a scientific path changed (readout_routing.py)", changed_paths=list(changed) + ["src/neural_decompiler/readout_routing.py"])
refused("a scientific path changed (run.py)", changed_paths=list(changed) + ["experiments/024-readout-routing-nounness/run.py"])
refused("023 cells changed", changed_paths=list(changed) + ["experiments/023-block0-completion/exposed-cells.f64"])
refused("edited preregistration", preregistration_text=kwargs["preregistration_text"].replace("0.3136960600375234", "0.3136960600375235"))
forged = copy.deepcopy(lock)
forged["primary"]["F_rho"] = 0.0
refused("lock with a changed threshold (digest broken)", _lock=forged)
resealed = copy.deepcopy(lock)
resealed["primary"]["null_975"] = 0.1
resealed["content_sha256"] = run.rc.content_digest(resealed)
refused("resealed lock with a replaced threshold", _lock=resealed)
state2 = copy.deepcopy(state)
state2["lock"]["content_sha256"] = resealed["content_sha256"]
refused("resealed lock with a forged state", _lock=resealed, state=state2)
refused("different noun order", noun_keys=list(reversed(kwargs["noun_keys"])))
try:
    run.Runner(config=run.rr.Configuration(**{**{f: getattr(rr.PRODUCTION, f) for f in rr.PRODUCTION.__dataclass_fields__}, "name": "test"}))
    check("a non-production configuration against the real repository is refused", False)
except PhaseError as error:
    check("a non-production configuration against the real repository is refused", True, str(error)[:80])
check("no capture entry point reached", seal.counts["capture_hits"] == 0)
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"ITEM4 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
