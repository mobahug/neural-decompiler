"""Item 2: one-shot execution evidence and the state transition against the pre-confirm state (read-only)."""
import rguard  # noqa: F401  (first)
from rguard import REPO

import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(REPO)
EV = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm024")
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail != '' else ''}", flush=True)


def canon(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(state):
    return hashlib.sha256(canon({k: v for k, v in state.items() if k != "state_sha256"}).encode("utf-8")).hexdigest()


def file_bytes(state):
    payload = {k: v for k, v in state.items() if k != "state_sha256"}
    return (canon({**payload, "state_sha256": digest(payload)}) + "\n").encode("utf-8")


def ts(text):
    return datetime.datetime.fromisoformat(text).timestamp()


# --- the launcher source: one confirm, no retry (AST) ------------------------------------------------------------------
import ast  # noqa: E402

src = (EV / "launch_confirm.py").read_text(encoding="utf-8")
tree = ast.parse(src)
parents = {}
for node in ast.walk(tree):
    for child in ast.iter_child_nodes(node):
        parents[child] = node


def ancestors(node):
    while node in parents:
        node = parents[node]
        yield node


confirm_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "confirm"
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "runner"]
check("launcher AST: exactly one runner.confirm() call", len(confirm_calls) == 1, [n.lineno for n in confirm_calls])
loops = [a for n in confirm_calls for a in ancestors(n) if isinstance(a, (ast.For, ast.While, ast.AsyncFor, ast.FunctionDef, ast.Lambda))]
check("launcher AST: the call is at module level (inside try only), in no loop and no function (so it cannot be re-invoked)", loops == [],
      [type(a).__name__ for a in loops])
trys = [a for n in confirm_calls for a in ancestors(n) if isinstance(a, ast.Try)]
handlers_retry = [h for t in trys for h in t.handlers for sub in ast.walk(h) if isinstance(sub, ast.Call) and getattr(sub.func, "attr", "") == "confirm"]
check("launcher AST: no confirm call inside an except/finally handler (no retry)", handlers_retry == [])
dry_if = [n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "DRY"]
exits_in_dry = [s2 for s2 in ast.walk(dry_if[0]) if isinstance(s2, ast.Call) and getattr(s2.func, "attr", "") == "exit"] if dry_if else []
check("launcher AST: the `if DRY:` block ends in sys.exit(0) and precedes the confirm call", len(dry_if) == 1 and len(exits_in_dry) == 1
      and dry_if[0].lineno < confirm_calls[0].lineno, [n.lineno for n in dry_if])
check("launcher source: sentinel created with O_CREAT|O_EXCL before the audit hook and before torch is imported",
      "os.O_CREAT | os.O_EXCL" in src and src.index("O_EXCL") < src.index("sys.addaudithook(_hook)") < src.index("import torch"))
check("launcher source: model loaded only through the counted loader (Runner(model_loader=_counted_load)); one load_model call site",
      src.count("models.load_model(") == 1 and "model_loader=_counted_load" in src)

# --- sentinels, records, logs ------------------------------------------------------------------------------------------
cs = (EV / "CONFIRM_LAUNCHED").read_text()
ds = (EV / "DRY_RUN_LAUNCHED").read_text()
print("   CONFIRM_LAUNCHED:", cs.strip(), "| DRY_RUN_LAUNCHED:", ds.strip())
check("both sentinels exist, one line each, distinct pids", cs.count("\n") == 1 and ds.count("\n") == 1 and cs.split()[-1] != ds.split()[-1])
rec = json.loads((EV / "confirm_record.json").read_text())
dry = json.loads((EV / "dry_run_record.json").read_text())
check("confirm record: pid == sentinel pid", str(rec["pid"]) == cs.split()[-1])
check("dry-run record: pid == sentinel pid", str(dry["pid"]) == ds.split()[-1])
c = rec["counts"]
check("confirm record: confirm_invocations 1, load_model 1", c["confirm_invocations"] == 1 and c["load_model"] == 1, c)
check("confirm record: capture_prompt 4320 == run_capture 4320 (capture_prompt's inner call), run_patched 0, run_interventions 0, log errors 0",
      c["capture_prompt"] == 4320 and c["run_capture"] == 4320 and c["run_patched"] == 0 and c["run_interventions"] == 0 and c["prompt_log_errors"] == 0)
check("confirm record: exit_status 0, no error, no traceback", rec["exit_status"] == 0 and rec["error"] is None and rec["traceback"] is None)
check("confirm record: module-blob assertion agreed (before the model load)", rec["module_blob_assertion"]["agree"] is True
      and ts(rec["module_blob_assertion"]["asserted_at"]) < ts(rec["confirm_started_at"]))
env = rec["environment"]
check("confirm record: 4 torch threads, HF offline, no bytecode, repo .venv", env["torch_num_threads"] == 4 and env["env"]["HF_HUB_OFFLINE"] == "1"
      and env["env"]["PYTHONDONTWRITEBYTECODE"] == "1" and env["prefix"] == str(ROOT / ".venv"))
check("confirm record: pinned model + revision + checkpoint sha256 3da38833…", env["model"] == {"id": "EleutherAI/pythia-70m-deduped", "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"}
      and env["checkpoint"]["sha256"] == env["checkpoint"]["blob"] == "3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd")
check("confirm record: HEAD af160ce, clean tree, pre-confirm state hashes 58891c24… / 14459cdd…",
      env["git"] == {"head": "af160cee0389bcdaf12bcbd92a8616f142777a52", "status_porcelain": ""}
      and env["hashes"]["state"] == "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d"
      and env["hashes"]["state_file"] == "14459cdde6ec2d832c3965d5b0ebaadaf06cef6aa0b0460c3a31b29dd442b262")
check("confirm record: state_before phases/ledgers/no stage2", env["state_before"] == {"phases": {"calibrate": "complete", "confirm": "not_started", "lock": "complete", "report": "not_started"},
                                                                                        "ledger": 0, "noun_ledger": 0, "stage2_exists": False})
ev = rec["events"]
check("confirm record: no repository write outside outputs, no forbidden read, no hook error", ev["repository_writes_outside_outputs"] == [] and ev["forbidden_reads"] == [] and ev["hook_errors"] == 0)
dc = dry["counts"]
check("dry run: confirm_invocations 0, load_model 0, every counter 0, no confirm_started_at", all(v == 0 for v in dc.values()) and "confirm_started_at" not in dry and "exit_status" not in dry, dc)
check("dry run: no output write at all", dry["events"]["output_writes"] == {} and dry["events"]["repository_writes_outside_outputs"] == [])
dlog = (EV / "dry_run_prompts.jsonl").read_text().splitlines()
check("dry run prompt log: only the stub probe line", len(dlog) == 1 and json.loads(dlog[0])["key"] == "dry-run|probe|0" and dry["dry_run"]["probe_returned"] == "stub", dlog)
out_c = (EV / "confirm.out").read_text()
out_d = (EV / "dry_run.out").read_text()
check("confirm.out: exactly one 'confirm: the one production invocation begins' line and one 'confirm complete' line",
      out_c.count("the one production invocation begins") == 1 and out_c.count("confirm complete:") == 1)
check("confirm.out: exactly one weight-loading progress bar (one model load)", out_c.count("Loading weights:   0%") == 1, out_c.count("Loading weights:   0%"))
check("dry_run.out: no load, no confirm", "Loading weights" not in out_d and "production invocation" not in out_d and "DRY RUN complete" in out_d)
check("confirm.exit: launcher exit 0", (EV / "confirm.exit").read_text().startswith("launcher exit 0"))
check("dry run finished before the confirm launch", ts(dry["record_written_at"]) < datetime.datetime.fromisoformat(cs.split()[1]).timestamp())

# --- the prompt log timing: the first capture after the ledger write, nothing before --------------------------------
plog = [json.loads(line) for line in (EV / "confirm_prompts.jsonl").read_text().splitlines()]
check("prompt log: 4320 lines numbered 1..4320", [e["n"] for e in plog] == list(range(1, 4321)))
state = json.loads((ROOT / "outputs/experiment-024/results.json").read_text())
ph = state["phases"]["confirm"]
t_first, t_last = plog[0]["t"], plog[-1]["t"]
print(f"   launcher start {rec['launcher_started_at']} | confirm_started_at {rec['confirm_started_at']} | phase started_at {ph['started_at']} | "
      f"first capture {datetime.datetime.fromtimestamp(t_first, datetime.timezone.utc).isoformat()} | last capture "
      f"{datetime.datetime.fromtimestamp(t_last, datetime.timezone.utc).isoformat()} | completed_at {ph['completed_at']} | confirm_ended_at {rec['confirm_ended_at']}")
check("first capture after confirm start and not before the phase's started_at (seconds resolution)", ts(rec["confirm_started_at"]) < t_first and ts(ph["started_at"]) <= t_first)
check("prompt log monotone in time", all(a["t"] <= b["t"] for a, b in zip(plog, plog[1:])))
st2 = ROOT / "outputs/experiment-024/stage2-measurements.pt"
m_stage2 = os.stat(st2).st_mtime
m_results = os.stat(ROOT / "outputs/experiment-024/results.json").st_mtime
print(f"   stage2 mtime {datetime.datetime.fromtimestamp(m_stage2, datetime.timezone.utc).isoformat()} | results.json mtime "
      f"{datetime.datetime.fromtimestamp(m_results, datetime.timezone.utc).isoformat()}")
check("stage2 written after the last capture and before completed_at (result write); results.json last written before confirm_ended_at",
      t_last < m_stage2 < ts(ph["completed_at"]) + 1 and m_results < ts(rec["confirm_ended_at"]))

# --- 020's explore runtime (the runner's _check_runtime reference) has 4 threads ----------------------------------------
cal_rt = state["phases"]["calibrate"]["runtime"]
check("calibrate runtime (same check) records 4 torch threads", cal_rt["torch_num_threads"] == 4, cal_rt)

# --- the checkpoint now ------------------------------------------------------------------------------------------------
ck = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c/model.safetensors"
h = hashlib.sha256(ck.read_bytes()).hexdigest()
check("cached checkpoint sha256 now == 3da38833… == its blob name", h == ck.resolve().name == "3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd", h)

# --- the state transition: revert only confirm's fields --------------------------------------------------------------
final = json.loads((ROOT / "outputs/experiment-024/results.json").read_text())
final.pop("state_sha256")
before = json.loads(json.dumps(final))
before["phases"]["confirm"] = {"status": "not_started"}
before["executed_prompt_keys"] = []
before["executed_noun_keys"] = []
before["confirmation"] = None
d_before = digest(before)
b_before = file_bytes(before)
check("revert {phases.confirm, executed_prompt_keys, executed_noun_keys, confirmation} -> digest 58891c24…", d_before == "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d", d_before)
check("... and the file bytes sha256 14459cdd… exactly", hashlib.sha256(b_before).hexdigest() == "14459cdde6ec2d832c3965d5b0ebaadaf06cef6aa0b0460c3a31b29dd442b262",
      hashlib.sha256(b_before).hexdigest())
# which top-level keys differ between the reconstructed pre-confirm state and the final one
diff_keys = sorted(k for k in set(before) | set(final) if before.get(k) != final.get(k))
check("only these four fields differ (top level: confirmation, executed_*; within phases only 'confirm')", diff_keys == ["confirmation", "executed_noun_keys", "executed_prompt_keys", "phases"]
      and [k for k in final["phases"] if final["phases"][k] != before["phases"][k]] == ["confirm"], diff_keys)
# the committed lock-phase evidence records the same pre-confirm digests
hits = subprocess.run(["git", "grep", "-l", "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d", "HEAD", "--", "experiments/024-readout-routing-nounness/evidence"],
                      cwd=ROOT, capture_output=True, text=True).stdout.split()
check("the committed lock evidence (HEAD) records 58891c24… as the post-lock state", len(hits) >= 1, hits)
# the noun ledger
nk = final["executed_noun_keys"]
print(f"   executed_noun_keys: {len(nk)} keys, e.g. {nk[:3]}")
print(f"S02 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
