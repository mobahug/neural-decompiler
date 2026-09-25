"""Item 6: I7 rehearsal, replicating Runner.confirm (run.py 510-537) statement by statement WITHOUT calling confirm, under
no-forward / no-capture guards; then an own float64 recomputation of the 40 scores (numerical agreement, not bytes)."""
import guard  # noqa: F401
from guard import REPO, Seal, canonical, sha256_bytes, summary

import hashlib
import importlib.util
import json
import math
import os
import sys
import time

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


RUN_PY = os.path.join(REPO, "experiments/024-readout-routing-nounness/run.py")
spec = importlib.util.spec_from_file_location("run024_postinstall_review", RUN_PY)
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)
rr, rd, ul, PhaseError = run.rr, run.rd, run.ul, run.PhaseError
from neural_decompiler import models as models_module  # noqa: E402
from neural_decompiler.models import PYTHIA_70M, load_model, seed_runtime  # noqa: E402

seal = Seal()
refused_names = seal.refuse_captures()  # stricter than required: refused from the start, before anything runs
print("refused from the start:", refused_names)


def refuse_loader(spec):
    raise RuntimeError("postinstall review: the runner's own loaders are not used; the model is loaded explicitly below")


runner = run.Runner(log=lambda m: print("  LOG:", m), model_loader=refuse_loader, tokenizer_loader=refuse_loader)

# ---- run.py 510-526 -------------------------------------------------------------------------------------------------
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
state = runner._state_for("confirm", base.digests)
rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.manifest_keys(), "Experiment 024's ledger before confirm")
paths = {"lock": runner.root / rr.LOCK_RELATIVE_PATH, "preregistration": runner.root / rr.PREREGISTRATION_RELATIVE_PATH}
if not all(path.exists() for path in paths.values()):
    raise PhaseError("the committed lock and preregistration must both exist")
lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
record, record_sha = runner._installed_record(state, base.digests)
git = runner.git_state()
placeholder = rr.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
rr.validate_lock(lock, state=state, digests=base.digests, config=runner.config, record=record, record_file_sha256=record_sha, confirmation=confirmation,
                 confirmation_file_sha256=confirmation_sha, dependencies=placeholder, noun_keys=runner._noun_keys(base.inputs),
                 preregistration_text=paths["preregistration"].read_text(encoding="utf-8"), git_state=git,
                 tracked=all(runner.tracked(path) for path in paths.values()), changed_paths=runner.changed_paths(lock["protocol_code_commit"]))
commit = str(git.get("commit"))
check("pre-model block (510-525) incl. the full validate_lock passed", True, commit)
runner._check_runtime(base.inputs.closure)                                                             # 526
check("runtime check passed", True)
seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)                                     # 527
# ---- 528: the model load, Module.__call__ counted ----------------------------------------------------------------------
load_invocations = 0
seal.count_module_calls()
t0 = time.time()
load_invocations += 1
model = load_model(PYTHIA_70M)                                                                         # 528 (the default model_loader)
load_seconds = time.time() - t0
calls_during_load = seal.counts["module_calls_during_load"]
# ---- immediately after the load: seal every module call and every forward -----------------------------------------------
stubbed_classes = seal.seal_model(model)
print(f"model loaded in {load_seconds:.1f} s; module classes stubbed ({len(stubbed_classes)}): {stubbed_classes}")
print(f"__call__ overrides found in module class MROs: {seal.counts['call_overrides_found']}")
# ---- 530-537 ------------------------------------------------------------------------------------------------------------
parameters_sha = rr.parameters_digest(model)
rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256=parameters_sha,
                             embedding_sha256=lock["dependencies"]["model"]["embedding_sha256"], what="the lock")      # 530-531
check("verify_model_dependencies(parameters_digest(model)) passed", True, parameters_sha)
progs = ul.ModelPrograms.from_model(model, base.inputs)                                                # 532
runner._check_nouns(progs, base.inputs)                                                                # 533
check("_check_nouns passed (79 scorable exposed nouns in pool order)", len(progs.scorable) == 79, len(progs.scorable))
with run.pytest_free_guard():                                                                          # 534
    embedding_sha = rr.embedding_digest(progs.weights.W_E)
    rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256=lock["dependencies"]["model"]["parameters_sha256"],
                                 embedding_sha256=embedding_sha, what="the lock")                     # 535-536
    reproduced = rr.reproduce_lock_quantities(progs.weights.W_E, record, confirmation, lock)          # 537
check("embedding digest check passed", embedding_sha == lock["dependencies"]["model"]["embedding_sha256"] == record["score"]["embedding_sha256"], embedding_sha)
check("I7: reproduce_lock_quantities bitwise_equal True", reproduced["bitwise_equal"] is True, reproduced)
check("I7: nothing differing", reproduced["differing"] == [])
check("I7: scores_sha256 == lock.fresh.scores_sha256 (a703ac16…)", reproduced["scores_sha256"] == lock["fresh"]["scores_sha256"], reproduced["scores_sha256"])
# STOP: nothing after run.py 541 is replicated (no units, no ledger, no record_execution, no state write, no stage_two_022).
counts = dict(seal.counts)
print("GUARD COUNTS:", json.dumps({"load_model_invocations": load_invocations, "module_calls_during_load": calls_during_load,
                                   "module_call_attempts_after_seal": counts["module_call_attempts_after_seal"], "forward_stub_hits": counts["forward_stub_hits"],
                                   "forward_classes_stubbed": counts["forward_classes_stubbed"], "instance_forwards_stubbed": counts["instance_forwards_stubbed"],
                                   "capture_hits": counts["capture_hits"], "capture_hits_by_name": seal.capture_hits_by_name}))
check("0 module calls after the seal, 0 forward stub hits, 0 capture hits",
      counts["module_call_attempts_after_seal"] == 0 and counts["forward_stub_hits"] == 0 and counts["capture_hits"] == 0)
check("state untouched in memory: ledger empty, confirm not_started", state["executed_prompt_keys"] == [] and state["phases"]["confirm"] == {"status": "not_started"})

# ---- own recomputation of the 40 scores -----------------------------------------------------------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402

embed_params = [(n, p) for n, p in model.named_parameters() if tuple(p.shape) == (50304, 512) and "embed" in n and "unembed" not in n]
print("embedding parameters found:", [n for n, _ in embed_params])
W = embed_params[0][1].detach().to("cpu", torch.float32).contiguous()
check("own-selected embedding parameter is bitwise progs.weights.W_E", all(torch.equal(p.detach().cpu().float(), progs.weights.W_E) for _, p in embed_params))
own_emb_digest = hashlib.sha256(json.dumps(list(W.shape)).encode("ascii") + b"|<f4|" + W.numpy().astype("<f4").tobytes()).hexdigest()
check("own embedding digest == lock/record embedding_sha256", own_emb_digest == lock["dependencies"]["model"]["embedding_sha256"], own_emb_digest[:12])
own_param = hashlib.sha256()
for name, p in sorted(model.named_parameters(), key=lambda item: item[0]):
    a = p.detach().cpu().contiguous().to(torch.float32)
    own_param.update(name.encode() + b"|" + json.dumps(list(a.shape)).encode("ascii") + b"|" + a.numpy().astype("<f4").tobytes())
check("own parameters digest == lock parameters_sha256", own_param.hexdigest() == lock["dependencies"]["model"]["parameters_sha256"], own_param.hexdigest()[:12])
E = W.numpy().astype(np.float64)
score = record["score"]
noun_ids, cue_ids = [int(i) for i in score["noun_row_ids"]], [int(i) for i in score["calibration_cue_ids"]]
check("record ids: 158 noun rows, 139 calibration cue rows; own canonical digests match",
      len(noun_ids) == 158 and len(cue_ids) == 139 and sha256_bytes(canonical(noun_ids).encode()) == score["noun_row_ids_sha256"]
      and sha256_bytes(canonical(cue_ids).encode()) == score["calibration_cue_ids_sha256"])
check("lock.score == record.score", lock["score"] == record["score"])
check("calibration cue ids == the record's 139 calibration_cues token ids in order", [int(e["token_id"]) for e in record["calibration_cues"]] == cue_ids)
with open(os.path.join(REPO, rr.CONFIRMATION_RELATIVE_PATH), encoding="utf-8") as h:
    frozen = json.load(h)
fresh = [(c["word"], int(c["token_id"]), c["class"]) for c in frozen["cues"]]
lock_cues = lock["fresh"]["cues"]
check("lock fresh cues == committed freeze cues (word, id, class, order)", [(c["word"], c["token_id"], c["class"]) for c in lock_cues] == fresh)


def cos_np(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


mu_noun, mu_cue = E[noun_ids].mean(axis=0), E[cue_ids].mean(axis=0)
own_np = [cos_np(E[t], mu_noun) - cos_np(E[t], mu_cue) for _, t, _ in fresh]


def mean_fsum(ids):
    return [math.fsum(E[i, d] for i in ids) / len(ids) for d in range(E.shape[1])]


def cos_fsum(a, b):
    return math.fsum(x * y for x, y in zip(a, b)) / (math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(y * y for y in b)))


mn, mc = mean_fsum(noun_ids), mean_fsum(cue_ids)
own_fs = [cos_fsum(list(E[t]), mn) - cos_fsum(list(E[t]), mc) for _, t, _ in fresh]
canon = [float(c["nounness"]) for c in lock_cues]
d_np = max(abs(a - b) for a, b in zip(own_np, canon))
d_fs = max(abs(a - b) for a, b in zip(own_fs, canon))
order_lock = sorted(range(40), key=lambda i: canon[i])
order_np = sorted(range(40), key=lambda i: own_np[i])
order_fs = sorted(range(40), key=lambda i: own_fs[i])
gaps = sorted(canon)
min_gap = min(b - a for a, b in zip(gaps, gaps[1:]))
print(f"own float64 numpy route: max |own - lock| = {d_np:.3e}; own fsum route: {d_fs:.3e}; smallest gap between neighbouring lock scores {min_gap:.3e}")
check("own numpy route agrees numerically (max abs diff < 1e-12)", d_np < 1e-12, f"{d_np:.3e}")
check("own fsum route agrees numerically (max abs diff < 1e-12)", d_fs < 1e-12, f"{d_fs:.3e}")
check("ordering identical (numpy route)", order_np == order_lock)
check("ordering identical (fsum route)", order_fs == order_lock)
bitwise_np = sum(1 for a, b in zip(own_np, canon) if a == b)
bitwise_fs = sum(1 for a, b in zip(own_fs, canon) if a == b)
print(f"bit-identical scores: numpy route {bitwise_np}/40, fsum route {bitwise_fs}/40 (not required; numerical agreement is expected)")


def f64_digest(values):
    arr = np.asarray(values, dtype="<f8")
    return hashlib.sha256(json.dumps([len(values)]).encode("ascii") + b"|<f8|" + arr.tobytes()).hexdigest()


check("own tensor digest of the lock's 40 scores == lock.fresh.scores_sha256", f64_digest(canon) == lock["fresh"]["scores_sha256"], f64_digest(canon)[:12])
print(f"digest of own numpy-route scores {f64_digest(own_np)[:12]}…, own fsum-route {f64_digest(own_fs)[:12]}… (numerical routes; byte identity not expected)")
# calibration maximum, above-maximum flags, class separation, the line's predictions
maximum = max(float(e["nounness_loo"]) for e in record["calibration_cues"])
check("maximum calibration score == lock.fresh.maximum_calibration_score == 0.13502027836111233",
      maximum == lock["fresh"]["maximum_calibration_score"] == 0.13502027836111233, repr(maximum))
own_loo = []
for k, t in enumerate(cue_ids):
    others = cue_ids[:k] + cue_ids[k + 1:]
    own_loo.append(cos_np(E[t], mu_noun) - cos_np(E[t], E[others].mean(axis=0)))
d_loo = max(abs(a - float(e["nounness_loo"])) for a, e in zip(own_loo, record["calibration_cues"]))
check("own leave-one-out calibration scores agree numerically (< 1e-12)", d_loo < 1e-12, f"{d_loo:.3e}")
flags_lock = [bool(c["above_calibration_maximum"]) for c in lock_cues]
flags_own = [s > max(own_loo) for s in own_np]
check("above-maximum flags: own == lock", flags_own == flags_lock)
e_above = [c["word"] for c in lock_cues if c["class"] == "E" and c["above_calibration_maximum"]]
check("5 of 8 E above the maximum: apple, horse, doctor, poet, dragon", e_above == ["apple", "horse", "doctor", "poet", "dragon"], e_above)
by_class = {cls: [s for s, (_, _, c) in zip(canon, fresh) if c == cls] for cls in "NBDCE"}
check("every E above every N (min E > max N)", min(by_class["E"]) > max(by_class["N"]), f"{min(by_class['E']):+.6f} > {max(by_class['N']):+.6f}")
above = {cls: sum(1 for c in lock_cues if c["class"] == cls and c["above_calibration_maximum"]) for cls in "NBDCE"}
check("23 of 40 above the maximum (6 B, 4 D, 8 C, 5 E, 0 N)", above == {"N": 0, "B": 6, "D": 4, "C": 8, "E": 5}, above)
pred_ok = all(float(c["predicted_log_mse"]) == float(lock["line"]["intercept"]) + float(lock["line"]["slope"]) * float(c["nounness"]) for c in lock_cues)
check("every predicted_log_mse == intercept + slope * nounness exactly in float64", pred_ok)
nearest = min(abs(s - maximum) for s in canon)
print(f"nearest fresh score to the calibration maximum: {nearest:.3e}")
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"ITEM6 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
