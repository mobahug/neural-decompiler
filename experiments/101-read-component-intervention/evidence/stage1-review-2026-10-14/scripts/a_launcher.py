"""Item 1: the launcher's evidence, tested as claims (no model, no forward; reads only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402  (first)

guard.self_test()
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

ROOT = guard.ROOT
L = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/stage1")
EXP = ROOT / "experiments/101-read-component-intervention"
out: dict = {"checks": {}, "facts": {}}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}


LAUNCHER_SHA = "13b1992f7f0feb652207ca835df044244f20faea750e92f42d8d51bced7c3e1c"
check("launcher_sha256", sha(L / "launch_stage1.py") == LAUNCHER_SHA, sha(L / "launch_stage1.py"))
prepared = EXP / "evidence/lock-2026-10-14/prepared/launch_stage1.py"
check("launcher_equals_committed_prepared_copy", prepared.exists() and sha(prepared) == LAUNCHER_SHA, sha(prepared) if prepared.exists() else None)
committed_blob = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD:experiments/101-read-component-intervention/evidence/lock-2026-10-14/prepared/launch_stage1.py"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip()
own_blob = hashlib.sha1(b"blob " + str(len((L / "launch_stage1.py").read_bytes())).encode() + b"\0" + (L / "launch_stage1.py").read_bytes()).hexdigest()
check("launcher_equals_git_blob_at_HEAD", committed_blob == own_blob, {"head_blob": committed_blob, "file_blob": own_blob})
check("launcher_sha256_at_production_txt", (L / "launcher_sha256_at_production.txt").read_text().split()[0] == LAUNCHER_SHA, (L / "launcher_sha256_at_production.txt").read_text().strip())
check("sentinel_names_launcher", (L / "stage1.sentinel").read_text() == LAUNCHER_SHA, (L / "stage1.sentinel").read_text())
check("production_exit_0", (L / "production.exit").read_text().strip() == "exit 0", (L / "production.exit").read_text().strip())

records = sorted(L.glob("production_run_*.json"))
rehearsals = sorted(L.glob("rehearsal_run_*.json"))
check("exactly_one_production_record", len(records) == 1, [p.name for p in records])
check("exactly_one_rehearsal_record", len(rehearsals) == 1, [p.name for p in rehearsals])
rec = json.loads(records[0].read_text())
ev = rec["events"]
env = rec["environment"]
check("record_mode_production", rec["mode"] == "production" and env["mode"] == "production")
check("record_head_is_lock_commit", env["head"] == "961e81408e5c4da4fffdf9dfab0343e449a4e1f2", env["head"])
check("record_branch", env["branch"] == "exp101-phases", env["branch"])
check("record_launcher_sha", env["launcher_sha256"] == LAUNCHER_SHA)
check("record_env_flags", env["env_flags"]["HF_HUB_OFFLINE"] == "1" and env["env_flags"]["PYTHONDONTWRITEBYTECODE"] == "1", env["env_flags"])
check("record_exit_0_no_error", rec["exit_status"] == 0 and rec["error"] is None, {"exit_status": rec["exit_status"], "error": rec["error"]})
check("one_stage1_invocation", ev["stage1_invocations"] == 1, ev["stage1_invocations"])
check("one_model_load", ev["load_model"] == 1, ev["load_model"])
check("nothing_refused", ev["refused"] == [] and ev["forbidden_reads"] == [] and ev["forbidden_writes"] == [],
      {k: ev[k] for k in ("refused", "forbidden_reads", "forbidden_writes")})
check("sanctioned_runs_36", ev["sanctioned_runs"] == 36, ev["sanctioned_runs"])
check("forward_calls_36x340", ev["sanctioned_forward_calls"] == 36 * 340, {"calls": ev["sanctioned_forward_calls"], "per_run": ev["sanctioned_forward_calls"] / 36})
check("stage1_writes_exactly_three", sorted(ev["stage1_writes"]) == ["states.f64", "states.json", "table.json"], ev["stage1_writes"])
check("stage1_mkdirs_two_events", ev["stage1_mkdirs"] == 2, ev["stage1_mkdirs"])
reads = ev["outputs_reads"]
allowed_reads = all(r == str(ROOT / "outputs/experiment-020/results.json") or r == str(ROOT / "outputs/experiment-101") or r.startswith(str(ROOT / "outputs/experiment-101") + "/")
                    for r in reads)
check("outputs_reads_only_020_results_and_101_state", allowed_reads, reads)
check("outputs_101_reads_only_state_and_temporaries",
      all(Path(r).name in ("results.json", "experiment-101") or (Path(r).name.startswith(".results-") and r.endswith(".json")) for r in reads if "experiment-101" in r), reads)
check("vestigial_output_writes_and_renames_empty", ev["output_writes"] == [] and ev["output_renames"] == [], "never filled by v2 (prior re-check N7): the state writes are not counted")
pre = rec["preflight"]
pre_ok = {k: (v if isinstance(v, bool) else v.get("ok")) for k, v in pre.items()}
check("preflight_all_true", all(pre_ok.values()), pre_ok)
reh_pre = json.loads(sorted(L.glob("rehearsal_run_*.json"))[0].read_text())["preflight"]
expected_keys = (set(reh_pre) - {"write_predicate_refuses_stage1_files_in_rehearsal"}) | {"production_token_names_the_installed_lock", "write_predicate_allows_exactly_the_stage1_files"}
check("preflight_17_checks_rehearsal_plus_token", len(pre_ok) == 17 and set(pre_ok) == expected_keys and len(reh_pre) == 16, {"production": len(pre_ok), "rehearsal": len(reh_pre)})
check("hook_self_test_ok", rec["hook_self_test"]["ok"] is True)
post = rec["post"]
check("post_checks_all_true", all(post[k] for k in ("all_36_ran_once", "ledger_is_exactly_the_36_s1_keys", "tree_is_exactly_the_three_untracked_stage1_files",
                                                     "record_digest_reproduces", "files_equal_record")) and post["keys_done"] == 36,
      {k: post[k] for k in ("keys_done", "all_36_ran_once", "ledger_is_exactly_the_36_s1_keys", "tree_is_exactly_the_three_untracked_stage1_files",
                            "record_digest_reproduces", "files_equal_record")})
expected_files = {"experiments/101-read-component-intervention/stage1/states.f64": "1c7e51c049e1c61cf669f1783736054956073dc5af6947354b1d13c058ac5b66",
                  "experiments/101-read-component-intervention/stage1/states.json": "2b2c3a61b4a2ce05eba0dcf875c19bfdb1c2627da1ccb1e8d49710af95a873ca",
                  "experiments/101-read-component-intervention/stage1/table.json": "71d339cd1d7151ad57949384388679593264ac07dec8c9c944c121d9918795fa"}
check("post_files_equal_brief", post["files"] == expected_files, post["files"])
check("post_table_content_sha256", post["table_content_sha256"] == "9191800440af37909d2068459e360d7772191fc66bf2ad5fb9cf2c133246ac86", post["table_content_sha256"])
check("tree_after_three_untracked", sorted(rec["tree_after"].splitlines()) == sorted(f"?? {p}" for p in expected_files), rec["tree_after"])
check("sealing_signatures_kept", rec["sealing"]["signatures_kept"] is True and rec["sealing"]["instances"] == 430, rec["sealing"]["instances"])
check("aliases_include_capture_reference_and_run_arm",
      {"neural_decompiler.read_intervention.capture_reference", "neural_decompiler.read_intervention.run_arm"} <= set(rec["aliases_wrapped"]), rec["aliases_wrapped"])

# The run log: 36 lines, n = 1..36, each S1 key once, in frozen order.
conf = json.loads((EXP / "confirmation-v1.json").read_text())
s1_frozen_order = [f"{f['frame_id']}|ref|581|S1" for f in conf["frames"]]
lines = [json.loads(line) for line in (L / "production_runs.jsonl").read_text().splitlines() if line.strip()]
check("run_log_36_lines_in_order", [e["n"] for e in lines] == list(range(1, 37)) and [e["key"] for e in lines] == s1_frozen_order, len(lines))
check("run_log_equals_manifest_S1", sorted(e["key"] for e in lines) == sorted(conf["manifest"]["S1-REF"]))
check("no_s2_key_in_run_log", not any(not e["key"].endswith("|ref|581|S1") for e in lines))
# The rehearsal record: no model, no writes, no sanctioned run.
reh = json.loads(rehearsals[0].read_text())
check("rehearsal_clean", reh["mode"] == "rehearsal" and reh["events"]["load_model"] == 0 and reh["events"]["sanctioned_runs"] == 0 and reh["events"]["stage1_writes"] == []
      and reh["exit_status"] == 0 and reh["error"] is None and reh["environment"]["head"] == env["head"], {k: reh["events"][k] for k in ("load_model", "sanctioned_runs", "stage1_writes")})
check("rehearsal_launcher_sha", reh["environment"]["launcher_sha256"] == LAUNCHER_SHA, reh["environment"]["launcher_sha256"])

# The log: the 36 'captured' lines in frozen order, one completion line naming the table digest.
log = (L / "production.log").read_text()
captured = [line.strip().split(":")[0] for line in log.splitlines() if line.strip().endswith("reference state captured")]
check("log_36_captures_in_order", captured == [f["frame_id"] for f in conf["frames"]], len(captured))
check("log_names_table_digest", "table 9191800440af37909d2068459e360d7772191fc66bf2ad5fb9cf2c133246ac86" in log)
check("log_no_incident", "INCIDENT" not in log and "Traceback" not in log)

# Timeline (UTC): started file, record, state, files' mtimes.
state = json.loads((ROOT / "outputs/experiment-101/results.json").read_text())
st = state["phases"]["stage1"]


def mtime(path: Path) -> str:
    return datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat()


timeline = {"production_started_utc.txt": (L / "production_started_utc.txt").read_text().strip(), "record.environment.recorded_at": env["recorded_at"],
            "record.started_at": rec["started_at"], "state.phases.stage1.started_at": st.get("started_at"), "state.phases.stage1.completed_at": st.get("completed_at"),
            "record.ended_at": rec["ended_at"], "mtime.production_runs.jsonl": mtime(L / "production_runs.jsonl"),
            "mtime.states.f64": mtime(EXP / "stage1/states.f64"), "mtime.states.json": mtime(EXP / "stage1/states.json"), "mtime.table.json": mtime(EXP / "stage1/table.json"),
            "mtime.results.json": mtime(ROOT / "outputs/experiment-101/results.json"), "mtime.sentinel": mtime(L / "stage1.sentinel"),
            "mtime.record": mtime(records[0]), "mtime.production.exit": mtime(L / "production.exit"),
            "mtime.launcher_sha_txt": mtime(L / "launcher_sha256_at_production.txt"), "mtime.launch_stage1.py": mtime(L / "launch_stage1.py")}
out["facts"]["timeline_utc"] = timeline
t = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
check("timeline_ordered", t(timeline["production_started_utc.txt"]) <= t(rec["started_at"]) <= t(st["started_at"]) <= t(st["completed_at"]) <= t(rec["ended_at"]),
      "started file <= record start <= stage1 running <= stage1 complete <= record end (state times have 1 s resolution)")
check("files_written_inside_the_run", all(t(rec["started_at"]) <= t(timeline[k]) <= t(rec["ended_at"]) for k in ("mtime.states.f64", "mtime.states.json", "mtime.table.json")))
check("launcher_not_modified_after_production_sha", t(timeline["mtime.launch_stage1.py"]) <= t(timeline["mtime.launcher_sha_txt"]))
# Cross-run constancy of the per-run module-call count (calibrate smoke and production, stage 1).
cal = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/calibrate")
cal_prod = json.loads((cal / "production_run.json").read_text())["events"]
cal_smoke = json.loads((cal / "smoke_run.json").read_text())["events"]
per_run = {"calibrate_production": cal_prod["sanctioned_forward_calls"] / cal_prod["sanctioned_runs"], "calibrate_smoke": cal_smoke["sanctioned_forward_calls"] / cal_smoke["sanctioned_runs"],
           "stage1": ev["sanctioned_forward_calls"] / ev["sanctioned_runs"]}
out["facts"]["per_run_module_calls"] = {**per_run, "calibrate_production_runs": cal_prod["sanctioned_runs"], "calibrate_smoke_runs": cal_smoke["sanctioned_runs"]}
check("per_run_count_constant_340", set(per_run.values()) == {340.0}, per_run)
out["guard"] = guard.summary()
ok = all(c["ok"] for c in out["checks"].values())
out["all_ok"] = ok
(guard.OUT / "a_launcher.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
for name, c in out["checks"].items():
    print(("PASS " if c["ok"] else "FAIL ") + name + ("" if c["ok"] else f"  {c['detail']}"))
print(json.dumps(out["facts"], indent=1))
print("ALL OK" if ok else "SOME CHECKS FAILED")
