"""Review items 1, 2 (records part), 3 (keys, log, ledger, pairing), 13 (write order and digests).

Own code only: no module statistic is called. No model, no artifact. Reads only the freeze, the lock, the results
state, the launcher's records and file metadata. Writes r1_records.json here.
"""
import guard  # noqa: F401

import datetime
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(guard.ROOT)
HERE = Path(guard.HERE)
RUNS = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm025")
EXP = ROOT / "experiments/025-nounness-direction-intervention"
OUT = ROOT / "outputs/experiment-025"
out: dict = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> str:  # the project's canonical JSON, restated
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------- item 1: provenance
launcher = RUNS / "launch_confirm_v3.py"
prov = {"launcher_v3_sha256_today": sha256_file(launcher), "launcher_v2_sha256": sha256_file(RUNS / "launch_confirm_v2.py"),
        "launcher_v1_sha256": sha256_file(RUNS / "launch_confirm.py")}
prov["launcher_v3_matches_58b760be"] = prov["launcher_v3_sha256_today"] == "58b760bea9d8a7bf66bef4352deb5c7153f7aea46c264de02f49c9ab96d1e307"
cmd = (RUNS / "production_command.txt").read_text()
prov["production_command"] = cmd
prov["production_exit"] = (RUNS / "production.exit").read_text()
run = json.loads((RUNS / "production_run.json").read_text())
ev = run["events"]
prov["run_record"] = {"mode": run["mode"], "head": run["environment"]["head"], "launcher_sha256": run["environment"]["launcher_sha256"],
                      "cwd": run["environment"]["cwd"], "started_at": run["started_at"], "ended_at": run["ended_at"], "exit_status": run["exit_status"],
                      "error": run["error"], "load_model": ev["load_model"], "refused": ev["refused"], "forbidden_reads": ev["forbidden_reads"],
                      "forbidden_writes": ev["forbidden_writes"], "fresh_runs": ev["fresh_runs"], "confirm_invocations": ev["confirm_invocations"],
                      "ledger_checks": ev["ledger_checks"], "spent_runs": ev["spent_runs"], "output_renames": len(ev["output_renames"]),
                      "sanctioned_forward_calls": ev["sanctioned_forward_calls"], "tree_after": run["tree_after"], "head_after": run["head_after"],
                      "outputs": run["outputs"], "env_flags": run["environment"]["env_flags"], "preflight_all_true": all(
                          (v is True) or (isinstance(v, dict) and v.get("ok") is True) for v in run["preflight"].values())}
prov["forward_calls_per_run"] = ev["sanctioned_forward_calls"] / (ev["fresh_runs"] + len(ev["spent_runs"]))
prov["sentinel_content"] = (RUNS / "confirm.sentinel").read_text()
prov["sentinel_is_launcher_sha"] = prov["sentinel_content"] == prov["launcher_v3_sha256_today"]
log_text = (RUNS / "production.log").read_text()
prov["log_confirm_line"] = [line for line in log_text.splitlines() if line.startswith("confirm complete")]
prov["log_frames_measured"] = len(re.findall(r"^  frame \d+/108 \(", log_text, flags=re.M))
prov["log_has_traceback_or_incident"] = ("Traceback" in log_text) or ("INCIDENT" in log_text)

# the exit-127 event
shell = {name: (RUNS / name).read_text() for name in ("shell_error_production_command.txt", "shell_error_production.log", "shell_error_production.exit")}
meta = {}
for name in ("launch_confirm_v3.py", "rehearsal_run.json", "smoke_run.json", "shell_error_production_command.txt", "shell_error_production.log",
             "shell_error_production.exit", "production_command.txt", "production.log", "production_runs.jsonl", "confirm.sentinel", "production.exit",
             "production_run.json"):
    s = os.stat(RUNS / name)
    meta[name] = {"birth": utc(s.st_birthtime), "mtime": utc(s.st_mtime), "ctime": utc(s.st_ctime), "inode": s.st_ino, "bytes": s.st_size}
for name in ("candidate-lock.json", "candidate-preregistration.md", "results.json", "stage2-measurements.pt"):
    s = os.stat(OUT / name)
    meta["outputs/" + name] = {"birth": utc(s.st_birthtime), "mtime": utc(s.st_mtime), "ctime": utc(s.st_ctime), "inode": s.st_ino, "bytes": s.st_size}
shell_event = {"files": shell, "meta": meta}
b = lambda n: os.stat(RUNS / n).st_birthtime  # noqa: E731
m = lambda n: os.stat(RUNS / n).st_mtime  # noqa: E731
started = datetime.datetime.fromisoformat(run["started_at"]).timestamp()
recorded_at = datetime.datetime.fromisoformat(run["environment"]["recorded_at"]).timestamp()
shell_event["event_duration_s"] = m("shell_error_production.exit") - b("shell_error_production_command.txt")
shell_event["shell_command_identical_to_production_command_except_time"] = [l for l in shell["shell_error_production_command.txt"].splitlines() if not l.startswith("launched_at")] == [
    l for l in cmd.splitlines() if not l.startswith("launched_at")]
shell_event["zsh_whole_string_not_found"] = shell["shell_error_production.log"].startswith("(eval):1: no such file or directory: caffeinate -i -s env ")
shell_event["run_log_born_after_production_launch"] = b("production_runs.jsonl") > b("production_command.txt")
shell_event["run_log_born_between_recorded_at_and_started_at"] = recorded_at <= b("production_runs.jsonl") <= started
shell_event["sentinel_born_after_started_at"] = b("confirm.sentinel") >= started
shell_event["sentinel_minus_started_at_s"] = b("confirm.sentinel") - started
shell_event["sentinel_inode_after_production_command_inode"] = os.stat(RUNS / "confirm.sentinel").st_ino > os.stat(RUNS / "production_command.txt").st_ino
shell_event["renamed_at_ctime"] = utc(os.stat(RUNS / "shell_error_production.log").st_ctime)
out["item1_provenance"] = prov
out["item1_shell_event"] = shell_event

# ---------------------------------------------------------------- the state (digest, phases, timestamps)
raw = (OUT / "results.json").read_bytes()
state = json.loads(raw)
recorded = state.pop("state_sha256")
st = {"file_sha256": hashlib.sha256(raw).hexdigest(), "state_sha256_recorded": recorded, "state_sha256_recomputed": sha_text(canonical(state)),
      "created_at": state["created_at"], "lock_phase": {k: v for k, v in state["phases"]["lock"].items() if k != "runtime"},
      "confirm_phase": {k: v for k, v in state["phases"]["confirm"].items() if k not in ("I7", "patch_path_check")}, "report": state["report"],
      "confirmation_keys": sorted(state["confirmation"]), "incident": state["confirmation"].get("incident"),
      "phase_incidents": {k: v.get("incidents") for k, v in state["phases"].items() if v.get("incidents")}}
st["verifies"] = st["state_sha256_recorded"] == st["state_sha256_recomputed"] == "1d5d37f035bc3153fec6efac527b088f64cd42fcd74e59680f4388daea98b33c"
out["state"] = st

# ---------------------------------------------------------------- item 13: the result write digest
result_state = json.loads(json.dumps(state))
removed = result_state["confirmation"].pop("descriptives")
w13 = {"removed_keys": sorted(removed), "result_write_digest_recomputed": sha_text(canonical(result_state))}
w13["equals_aa3db023"] = w13["result_write_digest_recomputed"] == "aa3db023c4a4afbe730a0e55df00d6d378184bcf8fdf77c967f4d0aad11907c4"
w13["equals_log_digest"] = any(w13["result_write_digest_recomputed"] in line for line in prov["log_confirm_line"])
# the descriptive writes changed nothing else: the only difference between the two states is confirmation.descriptives
w13["only_difference_is_descriptives"] = {k: v for k, v in state.items() if k != "confirmation"} == {k: v for k, v in result_state.items() if k != "confirmation"} and {
    k: v for k, v in state["confirmation"].items() if k != "descriptives"} == result_state["confirmation"]
w13["n_state_writes_launcher"] = len(ev["output_renames"])
w13["expected_state_writes"] = ["ledger", "stage2 record", "accounting", "c_recompute", "gates", "result (atomic)", "descriptives.records", "descriptives.ladder"]
out["item13_write"] = w13

# ---------------------------------------------------------------- item 3: accounting, rebuilt from the freeze
freeze = json.loads((EXP / "confirmation-v1.json").read_text())
lock = json.loads((EXP / "preregistration-lock.json").read_text())
conditions = list(freeze["conditions"])
cues = [(c["word"], int(c["token_id"]), c["stratum"]) for c in freeze["cues"]]
frame_ids = list(freeze["exposed_frame_ids"])
own_keys = [f"{fid}|{w}|{t}|{c}" for fid in frame_ids for (w, t, _) in cues for c in conditions]
acc = {"n_cues": len(cues), "n_frames": len(frame_ids), "n_conditions": len(conditions), "own_keys": len(own_keys), "own_unique": len(set(own_keys))}
own = set(own_keys)
manifest = freeze["manifest"]["S2-TARGET"]
acc["freeze_manifest"] = {"n": len(manifest), "unique": len(set(manifest)), "equals_own": set(manifest) == own, "sorted": manifest == sorted(manifest)}
acc["conditions_equal_lock_config"] = conditions == lock["configuration"]["conditions"]
lines = [json.loads(line) for line in (RUNS / "production_runs.jsonl").read_text().splitlines()]
log_keys = [line["key"] for line in lines]
counts = Counter(log_keys)
acc["run_log"] = {"lines": len(lines), "unique": len(counts), "duplicates": sum(c - 1 for c in counts.values() if c > 1), "missing": len(own - set(counts)),
                  "extra": len(set(counts) - own), "n_field_contiguous": [line["n"] for line in lines] == list(range(1, len(lines) + 1)),
                  "per_condition": dict(sorted(Counter(k.rsplit("|", 1)[1] for k in log_keys).items()))}
ledger = state["executed_prompt_keys"]
lc = Counter(ledger)
acc["ledger"] = {"n": len(ledger), "unique": len(lc), "duplicates": sum(c - 1 for c in lc.values() if c > 1), "missing": len(own - set(lc)), "extra": len(set(lc) - own),
                 "per_condition": dict(sorted(Counter(k.rsplit("|", 1)[1] for k in ledger).items()))}
acc["per_condition_all_4320"] = all(v == 4320 for v in acc["run_log"]["per_condition"].values()) and all(v == 4320 for v in acc["ledger"]["per_condition"].values()) \
    and len(acc["run_log"]["per_condition"]) == 21
acc["state_accounting_record"] = state["confirmation"]["accounting"]
acc["stage2_n_executed"] = state["confirmation"]["stage2"]["n_executed"]

# the pairing proof: the runner's loop order (frames by frame_id, cues by token id, conditions in frozen order)
loop = [f"{fid}|{w}|{t}|{c}" for fid in sorted(frame_ids) for (w, t, _) in sorted(cues, key=lambda x: x[1]) for c in conditions]
mismatch = [i for i, (a, b2) in enumerate(zip(log_keys, loop)) if a != b2]
acc["pairing"] = {"loop_len": len(loop), "log_len": len(log_keys), "all_equal": len(loop) == len(log_keys) and not mismatch, "first_mismatch": mismatch[:3],
                  "first_log_keys": log_keys[:3], "last_log_key": log_keys[-1], "frame_order_matches_production_log": None}
frames_logged = re.findall(r"^  frame (\d+)/108 \((.+?)\) measured", log_text, flags=re.M)
acc["pairing"]["frame_order_matches_production_log"] = [fid for _, fid in frames_logged] == sorted(frame_ids)
# the conditions in the log per (frame, cue) block appear in frozen order, each block complete
blocks_ok = all(log_keys[i:i + 21] and [k.rsplit("|", 1)[1] for k in log_keys[i:i + 21]] == conditions for i in range(0, len(log_keys), 21))
acc["pairing"]["every_21_block_in_frozen_condition_order"] = blocks_ok
out["item3_accounting"] = acc

# ---------------------------------------------------------------- item 2: the patch-path keys outside the ledger and the accounting
spent = list(lock["patch_path_spent_keys"])
pp = state["phases"]["confirm"]["patch_path_check"]
iso = {"spent_keys": spent, "passed": pp["passed"], "per_key": {k: {"plain_equals_patched_theta0": v["plain_equals_patched_theta0"],
                                                                    "embed_equals_resid_pre0": v["embed_equals_resid_pre0"]} for k, v in pp["checks"].items()},
       "keys_recorded": pp["keys"] == spent,
       "spent_in_ledger_untagged": [k for k in spent if k in set(ledger)],
       "spent_in_ledger_as_prefix": [k for k in spent if any(x.startswith(k + "|") for x in ledger)],
       "spent_in_run_log": [k for k in spent if any(x == k or x.startswith(k + "|") for x in log_keys)],
       "spent_in_manifest": [k for k in spent if any(x.startswith(k + "|") for x in manifest)],
       "ledger_all_tagged_4_fields": all(len(k.split("|")) == 4 for k in ledger),
       "spent_runs_in_launcher_record": ev["spent_runs"], "I7": state["phases"]["confirm"]["I7"]}
iso["spent_words"] = sorted({k.split("|")[1] for k in spent})
iso["spent_token_ids_not_025_cues"] = not ({int(k.split("|")[2]) for k in spent} & {t for _, t, _ in cues})
out["item2_isolation"] = iso

(HERE / "r1_records.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(json.dumps({"launcher_ok": prov["launcher_v3_matches_58b760be"], "state_verifies": st["verifies"], "aa3db023": w13["equals_aa3db023"],
                  "only_desc_diff": w13["only_difference_is_descriptives"], "accounting": {k: acc[k] for k in ("own_keys", "own_unique")},
                  "manifest_eq": acc["freeze_manifest"]["equals_own"], "log": acc["run_log"], "ledger": {k: v for k, v in acc["ledger"].items() if k != "per_condition"},
                  "pairing": acc["pairing"], "iso": {k: v for k, v in iso.items() if k not in ("spent_runs_in_launcher_record",)}, "shell": {
                      k: v for k, v in shell_event.items() if k not in ("files", "meta")}, "guard": guard.EVENTS}, indent=1, ensure_ascii=False))
