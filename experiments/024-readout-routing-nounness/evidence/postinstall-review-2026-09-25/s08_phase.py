"""Item 8: confirm phase eligibility on the byte-verified results state; the phase table; once-only rules (pure calls)."""
import guard  # noqa: F401
from guard import REPO, canonical, sha256_bytes, summary

import json
import os

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


PATH = os.path.join(REPO, "outputs/experiment-024/results.json")
with open(PATH, "rb") as h:
    raw = h.read()
state = json.loads(raw)
payload = {k: v for k, v in state.items() if k != "state_sha256"}
own = sha256_bytes(canonical(payload).encode("utf-8"))
check("file sha256 == 14459cdd…", sha256_bytes(raw) == "14459cdde6ec2d832c3965d5b0ebaadaf06cef6aa0b0460c3a31b29dd442b262", sha256_bytes(raw))
check("own state digest == recorded state_sha256 == 58891c24…", own == state["state_sha256"] == "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d", own)
check("file == canonical JSON (with state_sha256) + one newline (the atomic writer's format)", raw == (canonical(state) + "\n").encode("utf-8"))

from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

loaded = rd.load_results_state(__import__("pathlib").Path(PATH))
check("rd.load_results_state verifies it", loaded["state_sha256"] == own)
rr.assert_phase_allowed("confirm", loaded)
check("rr.assert_phase_allowed('confirm', state) passes", True)
phases = loaded["phases"]
for name, entry in phases.items():
    print(f"   {name}: {json.dumps(entry, sort_keys=True)}")
LOCK_RUN = "be74d23d086bd12d894db1f73e6a1b4160aeebd5"
check("calibrate: complete, no incidents", phases["calibrate"]["status"] == "complete" and not phases["calibrate"].get("incidents"))
check("lock: complete at be74d23, no incident", phases["lock"]["status"] == "complete" and phases["lock"].get("commit") == LOCK_RUN and not phases["lock"].get("incidents"))
check("confirm: not_started, no incident", phases["confirm"] == {"status": "not_started"})
check("report: not_started", phases["report"] == {"status": "not_started"})
check("phase keys exactly {calibrate, lock, confirm, report} (canonical JSON stores them sorted)", set(phases) == {"calibrate", "lock", "confirm", "report"} and len(phases) == 4)
cal = loaded["calibration"]
check("calibration: no incidents, no stop, no cross-check failure", not cal.get("incidents") and "stop" not in cal and "cross_check" not in cal, sorted(cal))
with open(os.path.join(REPO, rr.CALIBRATION_RELATIVE_PATH), "rb") as h:
    cal_bytes = h.read()
check("calibration.record_sha256 == installed calibration-v1.json sha256", cal["record_sha256"] == sha256_bytes(cal_bytes))
with open(os.path.join(REPO, rr.CONFIRMATION_RELATIVE_PATH), "rb") as h:
    conf_bytes = h.read()
binding = {"path": rr.CONFIRMATION_RELATIVE_PATH, "file_sha256": sha256_bytes(conf_bytes), "content_sha256": json.loads(conf_bytes)["content_sha256"]}
check("state.confirmation_024 binding == the committed freeze (path, file, content)", loaded["confirmation_024"] == binding, binding["file_sha256"][:12])
check("state.configuration == PRODUCTION; module_blobs == FROZEN_BLOBS", loaded["configuration"] == rr.PRODUCTION.to_json() and loaded["module_blobs"] == rr.FROZEN_BLOBS)
print(f"   state.protocol_code_commit {loaded['protocol_code_commit']}; run_id {loaded['run_id']}; git_dirty {loaded['git_dirty']}")
check("state.git_dirty False", loaded["git_dirty"] is False)
with open(os.path.join(REPO, rr.LOCK_RELATIVE_PATH), encoding="utf-8") as h:
    lock = json.load(h)
check("state.run_id == lock.run_id", loaded["run_id"] == lock["run_id"], lock["run_id"])
for phase in ("calibrate", "lock"):
    try:
        rr.assert_phase_allowed(phase, loaded)
        check(f"{phase} refused now (runs once)", False, "ALLOWED")
    except rr.PhaseError as error:
        check(f"{phase} refused now (runs once)", True, str(error)[:90])
import copy  # noqa: E402

for label, mutate in (("confirm running", lambda s: s["phases"]["confirm"].update(status="running")),
                      ("confirm with an I7 incident", lambda s: s["phases"]["confirm"].update(incidents=[{"x": 1}])),
                      ("lock not complete", lambda s: s["phases"]["lock"].update(status="not_started"))):
    probe = copy.deepcopy(loaded)
    mutate(probe)
    try:
        rr.assert_phase_allowed("confirm", probe)
        check(f"confirm refused on a state with {label}", False, "ALLOWED")
    except rr.PhaseError as error:
        check(f"confirm refused on a state with {label}", True, str(error)[:80])
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"ITEM8 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
