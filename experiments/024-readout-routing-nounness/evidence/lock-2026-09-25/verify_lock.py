"""Post-lock verification of Experiment 024 (read-only; weights only for the I7 rehearsal; no prompt; confirm NOT run).

The candidate lock and preregistration against the committed freeze and calibration record, the frozen semantics and
the configuration; the 40 bound scores (order, values, digest, per-class summary, the extrapolation facts); the manifest;
the state; and a rehearsal of confirm's I7 (the canonical module regenerating the lock's weight-derived quantities from
the weights, compared bit for bit), with every forward refused.
"""
import json
import os
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
        raise PermissionError(f"the verification may not write into the repository: {path}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


out = ROOT / "outputs/experiment-024"
lock_path, prereg_path = out / "candidate-lock.json", out / "candidate-preregistration.md"
lock = json.loads(lock_path.read_text(encoding="utf-8"))
record = json.loads((ROOT / rr.CALIBRATION_RELATIVE_PATH).read_text(encoding="utf-8"))
state = rd.load_results_state(out / "results.json")
text = prereg_path.read_text(encoding="utf-8")
print(f"candidate lock: file sha256 {rc.file_sha256(lock_path)}; content sha256 {lock['content_sha256']}")
print(f"candidate preregistration: file sha256 {rc.file_sha256(prereg_path)}; {len(text.splitlines())} lines")
print(f"state sha256 {state['state_sha256']}; phases {({k: v['status'] for k, v in state['phases'].items()})}; lock commit {state['phases']['lock'].get('commit', '')[:7]}; "
      f"ledger {len(state['executed_prompt_keys'])}")
check("lock content digest", lock["content_sha256"] == rc.content_digest(lock))
check("state binds this candidate lock and preregistration", state["lock"]["content_sha256"] == lock["content_sha256"] and state["lock"]["preregistration_sha256"] == pm.sha256_text(text))
check("preregistration is the one this lock renders", text == rr.render_preregistration(lock))
check("lock/confirm/report: lock complete once, confirm and report not started; ledger empty",
      state["phases"]["lock"]["status"] == "complete" and not state["phases"]["lock"].get("incidents") and state["phases"]["confirm"]["status"] == "not_started"
      and state["phases"]["report"]["status"] == "not_started" and state["executed_prompt_keys"] == [])
check("design, plan, configuration, constants, module blobs", lock["design"] == rr.DESIGN and lock["plan"] == rr.PLAN and lock["configuration"] == rr.PRODUCTION.to_json()
      and lock["constants"] == rr.record_constants(rr.PRODUCTION) and lock["module_blobs"] == rr.FROZEN_BLOBS)
check("the lock binds the rr module blob at the run commit", lock["module"] == {"path": "src/neural_decompiler/readout_routing.py", "blob": rr.own_blob()}, lock["module"]["blob"])
check("calibration binding", lock["calibration"] == {"path": rr.CALIBRATION_RELATIVE_PATH, "file_sha256": "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d",
                                                     "content_sha256": "09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8"})
binding = lock["confirmation_024"]
check("freeze binding, counts and manifest", {k: binding[k] for k in ("path", "file_sha256", "content_sha256")} == record["confirmation_024"]
      and binding["counts"] == {"classes": {cls: 8 for cls in rr.CLASSES}} and binding["manifest_size"] == 4_320 and binding["manifest_sha256"].startswith("fcc437fc"),
      binding["manifest_sha256"])
check("calibration source and dependencies == the record's", lock["exposed_cells"] == rr.calibration_source() and lock["dependencies"] == record["dependencies"])
check("score bindings == the record's", lock["score"] == record["score"])
primary = lock["primary"]
check("primary: F_rho, null_975, effective threshold (null binds), line", (primary["F_rho"], primary["null_975"]) == (0.24411074612857814, 0.3136960600375234)
      and primary["effective_threshold"] == {"value": 0.3136960600375234, "binds": "null_975"} and lock["line"] == record["line"]
      and (lock["line"]["slope"], lock["line"]["intercept"], lock["line"]["residual_sd"]) == (1.2992017464639374, -2.481260357420486, 0.2570037337349923))
check("primary statistic text and precedence", primary["statistic"] == rr.STATISTIC and primary["precedence"] == list(rr.PRIMARY_RESULTS))
guard = lock["guard"]["spec"]
check("E–N guard spec", (guard["n_E"], guard["n_N"], guard["assignments"], guard["max_upper"]) == (8, 8, 12_870, 321) and "ties count against the guard" in guard["rule"]
      and "exact" in guard["arithmetic"], guard["rule"])
units = lock["guard"]["units"]
check("E and N cues in order", [u["word"] for u in units["E"]] == list(rr.EXPECTED_PICKS["ordinary"]) and [u["word"] for u in units["N"]] == list(rr.EXPECTED_PICKS["N"]))
check("outcome table and readings; semantics", lock["outcome"]["table"] == [list(row) for row in rr.OUTCOME_TABLE] and lock["outcome"]["readings"] == rr.SEMANTICS["outcomes"]
      and lock["semantics"] == rr.SEMANTICS)
check("secondary definitions bound (semantics text + constants + module blob)", "never rescue, alter or redefine" in lock["semantics"]["secondary"]
      and lock["constants"]["tags"]["contrast"] == "024|contrast" and lock["constants"]["ranks"]["contrast_lower"] == 250 and lock["constants"]["ranks"]["contrast_upper_element"] == 9750)
fresh = lock["fresh"]
cues = fresh["cues"]
freeze = json.loads((ROOT / rr.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))
check("40 cues in the canonical frozen order", [(c["word"], c["token_id"], c["class"]) for c in cues] == [(t["word"], t["token_id"], t["class"]) for t in freeze["cues"]])
check("score digest a703ac16…", fresh["scores_sha256"] == rc.tensor_digest(torch.tensor([c["nounness"] for c in cues], dtype=torch.float64))
      == "a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995")
check("predictions = the frozen line", all(c["predicted_log_mse"] == rr.predict(record["line"], c["nounness"]) for c in cues))
maximum = fresh["maximum_calibration_score"]
check("calibration maximum", maximum == 0.13502027836111233 == max(e["nounness_loo"] for e in record["calibration_cues"]))
check("flags = score > maximum", all(c["above_calibration_maximum"] == (c["nounness"] > maximum) for c in cues))
e_above = [c["word"] for c in cues if c["class"] == "E" and c["above_calibration_maximum"]]
check("5 of 8 E above; the sentence", fresh["extrapolation"]["E_above_maximum"] == 5 and e_above == ["apple", "horse", "doctor", "poet", "dragon"]
      and fresh["extrapolation"]["sentence"].startswith("5 of the 8 E cues"))
e_scores = [c["nounness"] for c in cues if c["class"] == "E"]
n_scores = [c["nounness"] for c in cues if c["class"] == "N"]
check("every E above every N", min(e_scores) > max(n_scores), f"{min(e_scores):+.6f} > {max(n_scores):+.6f}")
for cls in rr.CLASSES:
    values = [c["nounness"] for c in cues if c["class"] == cls]
    above = sum(1 for c in cues if c["class"] == cls and c["above_calibration_maximum"])
    print(f"  {cls}: min {min(values):+.6f} max {max(values):+.6f} mean {sum(values) / len(values):+.6f}; above the maximum {above} of 8", flush=True)
print(f"  descriptive: {sum(1 for c in cues if c['above_calibration_maximum'])} of 40 above the maximum (no outcome force)", flush=True)
inputs = ul.load_frozen_inputs(ROOT)
c022_path = ROOT / ul.CONFIRMATION_RELATIVE_PATH
c022 = json.loads(c022_path.read_text())
c023_path = ROOT / b0c.CONFIRMATION_RELATIVE_PATH
excluded = rr.exclusion(b0c.exclusion(inputs, c022, rc.file_sha256(c022_path)), json.loads(c023_path.read_text()), rc.file_sha256(c023_path))
confirmation = rr.load_confirmation_024(ROOT / rr.CONFIRMATION_RELATIVE_PATH, inputs.pool, excluded, rr.PRODUCTION)
manifest = confirmation.manifest()
check("manifest recomputed: 4,320 unique keys, digest bound", len(confirmation.manifest_keys()) == 4_320 and pm.sha256_text(pm.canonical_json(manifest)) == binding["manifest_sha256"])
# the I7 rehearsal: exactly confirm's check, from the weights, every forward refused
calls = {"during_load": 0}
original_call = torch.nn.Module.__call__


def counting(self, *args, **kwargs):
    calls["during_load"] += 1
    return original_call(self, *args, **kwargs)


torch.nn.Module.__call__ = counting
model = models.load_model(models.PYTHIA_70M)
torch.nn.Module.__call__ = lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("forward refused"))
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("capture refused")))
parameters = rr.parameters_digest(model)
W_E = pm.Weights.from_model(model).W_E
del model
rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256=parameters, embedding_sha256=rr.embedding_digest(W_E), what="the candidate lock")
check("model dependencies (parameters, embedding) reproduce", True)
reproduced = rr.reproduce_lock_quantities(W_E, record, confirmation, lock)
check("I7 rehearsal: the lock's quantities regenerate bit for bit", reproduced["bitwise_equal"], reproduced)
check("module calls during the load", calls["during_load"] == 0, calls)
check("no repository write", not REFUSED, REFUSED)
print("FAILURES:", failures if failures else "none", flush=True)
sys.exit(1 if failures else 0)
