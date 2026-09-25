"""Item 11: the result-state write chronology and the pre-descriptive result write (read-only)."""
import rguard  # noqa: F401  (first)
from rguard import REPO

import ast
import datetime
import hashlib
import json
import os
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


final = json.loads((ROOT / "outputs/experiment-024/results.json").read_text())
final.pop("state_sha256")
conf = final["confirmation"]
print("   confirmation keys (final):", sorted(conf))
check("final confirmation holds stage2, accounting, c_recompute, gates, results, lock_sha256, completed_at, descriptives — and nothing else (no incident, no cross_check)",
      sorted(conf) == ["accounting", "c_recompute", "completed_at", "descriptives", "gates", "lock_sha256", "results", "stage2"], sorted(conf))

# --- the pre-descriptive result write --------------------------------------------------------------------------------
pre = json.loads(json.dumps(final))
del pre["confirmation"]["descriptives"]
d_pre = digest(pre)
check("final state minus confirmation.descriptives → digest 37f2f88b… (the digest logged right after the one result write)",
      d_pre == "37f2f88b7f0fdde73ebbedc319fc5d9edcab59c35133a8cfa8b24c4e3201f9fd", d_pre)
out = (EV / "confirm.out").read_text()
check("confirm.out logs 'results sha256 37f2f88b…' on the 'confirm complete' line (run.py:606-607, after the write at :597)",
      "confirm complete: NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS (primary PASS, ρ 0.6108818011257036; E–N guard PASS, K 1); results sha256 37f2f88b7f0fdde73ebbedc319fc5d9edcab59c35133a8cfa8b24c4e3201f9fd" in out)
check("that reconstructed result-write state already holds: results (ρ, guard, outcome), phases.confirm complete, gates, c_recompute, accounting",
      pre["phases"]["confirm"]["status"] == "complete" and pre["confirmation"]["results"]["outcome"]["label"].startswith("NOUNNESS_PREDICTS")
      and "gates" in pre["confirmation"] and "c_recompute" in pre["confirmation"] and "accounting" in pre["confirmation"])
check("results.completed_at == phases.confirm.completed_at (same second: set in one statement block before the one write)",
      conf["completed_at"] == final["phases"]["confirm"]["completed_at"], conf["completed_at"])

# --- the code: the write chronology in Runner.confirm and _descriptives ----------------------------------------------
src = (ROOT / "experiments/024-readout-routing-nounness/run.py").read_text()
tree = ast.parse(src)
confirm_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "confirm")
desc_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_descriptives")
writes = sorted(n.lineno for n in ast.walk(confirm_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("_write",))
incident_writes = sorted(n.lineno for n in ast.walk(confirm_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("_record_incident", "_record_phase_incident"))
saves = sorted(n.lineno for n in ast.walk(confirm_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "save_durably")
results_assign = [n.lineno for n in ast.walk(confirm_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "update"
                  and any(isinstance(k, ast.Constant) and k.value == "results" for a in n.args if isinstance(a, ast.Dict) for k in a.keys)]
score_call = [n.lineno for n in ast.walk(confirm_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "score"]
desc_call = [n.lineno for n in ast.walk(confirm_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_descriptives"]
print(f"   Runner.confirm: state writes at {writes}; incident writers at {incident_writes}; save_durably at {saves}; score at {score_call}; results assigned at {results_assign}; "
      f"_descriptives at {desc_call}")
check("the only assignment of confirmation['results'] is at :594, after score (:577) and before the single result write (:597); _descriptives (:608) after it",
      results_assign == [594] and score_call == [577] and 597 in writes and desc_call == [608] and max(w for w in writes if w < 594) == 574)
check("the normal path's writes: 547 (ledger+running), 555 (stage2 digests), 560 (accounting), 569 (C recompute), 574 (gates), 597 (result + complete)",
      [w for w in writes if w in (547, 555, 560, 569, 574, 597)] == [547, 555, 560, 569, 574, 597])
desc_writes = sorted(n.lineno for n in ast.walk(desc_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_write")
targets = [ast.unparse(n) for n in ast.walk(desc_fn) if isinstance(n, (ast.Assign, ast.AugAssign))]
print(f"   _descriptives: writes at {desc_writes}; assignments {targets}")
check("_descriptives assigns only into state['confirmation']['descriptives'] (the setdefault dict) and its 'failures' entry",
      all(t.startswith(("descriptives = state['confirmation'].setdefault('descriptives', {})", "descriptives[name] = ", "per_cue = state['confirmation']['results']['per_cue']",
                                "descriptives.setdefault('failures', {})[name] = ")) for t in targets) and len(targets) == 4, targets)

# --- the evidence: the launcher's first-write order and counts --------------------------------------------------------
rec = json.loads((EV / "confirm_record.json").read_text())
ow = rec["events"]["output_writes"]
names = list(ow)
check("launcher: 9 atomic results writes + 1 stage2 file, each opened exactly once, stage2 second", len(names) == 10 and names[1] == "stage2-measurements.pt"
      and sum(1 for n in names if n.startswith(".results-")) == 9 and set(ow.values()) == {1})
mapping = dict(zip(names, ["547 ledger + running", "552 save_durably(stage2)", "555 stage2 digests", "560 accounting", "569 C recompute", "574 gates",
                           "597 ONE result write (ρ, guard, outcome, complete)", "633 descriptives: secondary", "633 descriptives: contrasts", "633 descriptives: ladder"]))
for n, where in mapping.items():
    print(f"   {n} → run.py:{where}")
check("9 results writes == 6 (normal path to the result) + 3 (secondary, contrasts, ladder) — no incident writer (+2 each) and no failure write ran",
      sum(1 for n in names if n.startswith(".results-")) == 9)
# the temporary files were renamed away (os.replace); none remains
left = sorted(p.name for p in (ROOT / "outputs/experiment-024").iterdir() if p.name.startswith(".results-"))
check("no temporary .results-*.json left in outputs/experiment-024 (every write completed its os.replace)", left == [], left)
check("rr.write_state_atomic: mkstemp in the same directory, fsync, os.replace, directory fsync", all(s in (ROOT / "src/neural_decompiler/readout_routing.py").read_text() for s in
      ("tempfile.mkstemp(prefix=\".results-\", suffix=\".json\", dir=path.parent)", "os.fsync(stream.fileno())", "os.replace(temporary, path)", "fsync_directory(path.parent)")))

# --- timestamps -----------------------------------------------------------------------------------------------------------
ts = lambda s: datetime.datetime.fromisoformat(s).timestamp()  # noqa: E731
st2 = os.stat(ROOT / "outputs/experiment-024/stage2-measurements.pt").st_mtime
res = os.stat(ROOT / "outputs/experiment-024/results.json").st_mtime
plog = (EV / "confirm_prompts.jsonl").read_text().splitlines()
t_last = json.loads(plog[-1])["t"]
ph = final["phases"]["confirm"]
print(f"   started_at {ph['started_at']} | last capture {datetime.datetime.fromtimestamp(t_last, datetime.timezone.utc).isoformat()} | stage2 mtime "
      f"{datetime.datetime.fromtimestamp(st2, datetime.timezone.utc).isoformat()} | result write (completed_at) {ph['completed_at']} | results.json mtime (last descriptive write) "
      f"{datetime.datetime.fromtimestamp(res, datetime.timezone.utc).isoformat()} | launcher end {rec['confirm_ended_at']}")
check("chronology: last capture < stage2 durable < result write (completed_at, second resolution) < last descriptive write < launcher end",
      t_last < st2 < ts(ph["completed_at"]) + 1 and ts(ph["completed_at"]) < res < ts(rec["confirm_ended_at"]))
# no incident coexists with the result
inc = [k for k in ("incident", "cross_check") if k in conf] + [p for p, e in final["phases"].items() if e.get("incidents")] + (["calibration incidents"] if final["calibration"].get("incidents") else [])
check("no incident coexists with the result (confirmation.incident / cross_check / phase incidents / calibration incidents all absent)", inc == [], inc)
print(f"S11 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
