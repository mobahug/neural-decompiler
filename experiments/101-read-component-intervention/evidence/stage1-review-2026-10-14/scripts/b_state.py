"""Item 2: the results state after stage 1 (no model, no forward; reads only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402  (first)

guard.self_test()
import hashlib  # noqa: E402
import json  # noqa: E402

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler.provenance import collect_versions  # noqa: E402

ROOT = guard.ROOT
EXP = ROOT / ri.EXPERIMENT_DIR
out: dict = {"checks": {}, "facts": {}}


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}


state = rd.load_results_state(ROOT / "outputs/experiment-101/results.json")  # verifies the state digest
check("state_digest_verifies", True, state["state_sha256"])
out["facts"]["state_sha256"] = state["state_sha256"]
out["facts"]["results_json_file_sha256"] = hashlib.sha256((ROOT / "outputs/experiment-101/results.json").read_bytes()).hexdigest()
out["facts"]["top_keys"] = sorted(state)
check("experiment_and_configuration", state["experiment"] == "101" and state["configuration"] == ri.PRODUCTION.to_json())
lock = json.loads((EXP / "preregistration-lock.json").read_text(encoding="utf-8"))
check("lock_content_digest_verifies", lock["content_sha256"] == rc.content_digest(lock), lock["content_sha256"])
payload = json.loads((EXP / "confirmation-v1.json").read_text(encoding="utf-8"))
confirmation = ri.confirmation_from_payload(payload, ri.PRODUCTION)  # verify=True
s1 = confirmation.s1_keys()
s2 = set(confirmation.s2_keys())

phases = state["phases"]
out["facts"]["phases"] = phases
statuses = {name: entry["status"] for name, entry in phases.items()}
check("phase_statuses", statuses == {"calibrate": "complete", "lock": "complete", "stage1": "complete", "confirm": "not_started", "report": "not_started"}, statuses)
check("stage1_awaiting_commit", phases["stage1"].get("awaiting_commit") is True)
check("stage1_phase_commit", phases["stage1"].get("commit") == "961e81408e5c4da4fffdf9dfab0343e449a4e1f2", phases["stage1"].get("commit"))
check("stage1_geometry_bitwise", phases["stage1"].get("geometry") == {"bitwise_equal": True, "differing": []}, phases["stage1"].get("geometry"))
check("confirm_not_started_exactly", phases["confirm"] == {"status": "not_started"}, phases["confirm"])
check("report_not_started_exactly", phases["report"] == {"status": "not_started"}, phases["report"])
incidents = {name: entry.get("incidents") for name, entry in phases.items() if entry.get("incidents")}
check("no_phase_incidents", not incidents, incidents)
check("no_calibration_incident", not (state.get("calibration") or {}).get("incidents"), (state.get("calibration") or {}).get("incidents"))
check("no_stage1_record_incident", "incident" not in (state.get("stage1") or {}))
check("confirmation_block_absent", state.get("confirmation") is None, type(state.get("confirmation")).__name__)
check("report_block_absent", state.get("report") is None)
check("lock_phase_before", phases["lock"]["status"] == "complete" and not phases["lock"].get("incidents"))

ledger = state["executed_prompt_keys"]
check("ledger_exactly_36_s1_keys", sorted(ledger) == sorted(s1) and len(ledger) == 36 == len(set(ledger)), len(ledger))
check("ledger_has_no_s2_key", not (set(ledger) & s2) and all(key.endswith("|ref|581|S1") for key in ledger))
check("ledger_sorted_as_written", ledger == sorted(ledger))
cal_keys = state["calibration_executed"]
check("calibration_ledger_2016_no_101_frame", len(cal_keys) == 2016 == len(set(cal_keys)) and not any("-101-" in key for key in cal_keys), len(cal_keys))
check("s1_s2_disjoint_untagged_s1_only_ref", all(cr.untagged(k).endswith("|ref|581") for k in s1))

record = state["stage1"]
out["facts"]["stage1_record_keys"] = sorted(record)
check("record_keys", sorted(record) == sorted(["frames", "state_digests", "files", "table_content_sha256", "lock_sha256", "commit", "noun_keys", "executed", "digest"]), sorted(record))
check("record_digest_reproduces", record["digest"] == b0c.stage_one_digest(record), record["digest"])
check("record_frames_are_frozen_frames", record["frames"] == [{"frame_id": f.frame_id, "p_c": f.p_c, "p_t": f.p_t} for f in confirmation.frames])
check("record_frames_coordinated_structure", all(f["p_t"] == f["p_c"] + 1 and 4 <= f["p_c"] <= 7 for f in record["frames"]))
check("record_state_digests_36", sorted(record["state_digests"]) == sorted(f.frame_id for f in confirmation.frames) and len(record["state_digests"]) == 36)
expected_files = {"states_data": ("experiments/101-read-component-intervention/stage1/states.f64", "1c7e51c049e1c61cf669f1783736054956073dc5af6947354b1d13c058ac5b66"),
                  "states_index": ("experiments/101-read-component-intervention/stage1/states.json", "2b2c3a61b4a2ce05eba0dcf875c19bfdb1c2627da1ccb1e8d49710af95a873ca"),
                  "table": ("experiments/101-read-component-intervention/stage1/table.json", "71d339cd1d7151ad57949384388679593264ac07dec8c9c944c121d9918795fa")}
check("record_files", {k: (v["path"], v["sha256"]) for k, v in record["files"].items()} == expected_files, record["files"])
check("record_paths_equal_lock_paths", {k: v["path"] for k, v in record["files"].items()} == lock["stage1"]["paths"])
check("record_table_content_sha256", record["table_content_sha256"] == "9191800440af37909d2068459e360d7772191fc66bf2ad5fb9cf2c133246ac86")
check("record_lock_sha256_is_installed_lock", record["lock_sha256"] == lock["content_sha256"] == "8da97e35556d0baa4739899cb841e86cbd59cced21611c732fdd9d8c9edd9061", record["lock_sha256"])
check("record_commit", record["commit"] == "961e81408e5c4da4fffdf9dfab0343e449a4e1f2", record["commit"])
check("record_noun_keys_equal_lock", record["noun_keys"] == lock["noun_keys"] == lock["stage1"]["table"]["noun_keys"] and len(record["noun_keys"]) == 79)
check("record_executed_36", record["executed"] == 36)
check("state_lock_block", state["lock"]["content_sha256"] == lock["content_sha256"]
      and state["lock"]["preregistration_sha256"] == pm.sha256_text((EXP / "preregistration.md").read_text(encoding="utf-8")), state["lock"])
check("state_versions_equal_now", dict(state["versions"]) == dict(collect_versions()), {"state": state["versions"], "now": collect_versions()})
check("state_protocol_commit_is_calibrate_commit", state["protocol_code_commit"] == "8f65062627c77815adf9289c3fe60191d9fc3c17", state["protocol_code_commit"])
out["facts"]["stage1_phase"] = phases["stage1"]
out["facts"]["record_digest"] = record["digest"]

# The phase rules on this state: confirm is allowed; stage1, lock and calibrate are refused.
try:
    ri.assert_phase_allowed("confirm", state)
    check("confirm_phase_rule_allows", True)
except Exception as error:
    check("confirm_phase_rule_allows", False, str(error))
for phase in ("stage1", "lock", "calibrate"):
    try:
        ri.assert_phase_allowed(phase, state)
        check(f"{phase}_phase_rule_refuses", False, "allowed")
    except Exception as error:
        check(f"{phase}_phase_rule_refuses", True, str(error))
# The latent rule gap (prior re-check N1): confirm's rule ignores phases.stage1.incidents. Show it on a copy, in memory.
probe = json.loads(json.dumps(state))
probe["phases"]["stage1"]["incidents"] = [{"message": "synthetic", "after_ledger": True}]
try:
    ri.assert_phase_allowed("confirm", probe)
    out["facts"]["N1_confirm_rule_ignores_stage1_phase_incidents"] = True
except Exception:
    out["facts"]["N1_confirm_rule_ignores_stage1_phase_incidents"] = False
out["guard"] = guard.summary()
out["all_ok"] = all(c["ok"] for c in out["checks"].values())
(guard.OUT / "b_state.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
for name, c in out["checks"].items():
    print(("PASS " if c["ok"] else "FAIL ") + name + ("" if c["ok"] else f"  {c['detail']}"))
print(json.dumps({k: out["facts"][k] for k in ("state_sha256", "results_json_file_sha256", "record_digest", "stage1_phase", "N1_confirm_rule_ignores_stage1_phase_incidents")}, indent=1, default=str))
print("ALL OK" if out["all_ok"] else "SOME CHECKS FAILED")
