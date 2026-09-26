"""Post-report verification of Experiment 024 (read-only; no model, no prompt; nothing repaired; closure NOT declared).

The report is exactly the frozen renderer's output for the confirm-final state and the installed calibration record. The
report run changed only the state's report entries: reverting them reproduces the confirm-final state byte for byte.
The ledger and the measurements are unchanged. The report states the frozen result exactly. Also checked: the evidence
SHA256SUMS and the repository.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
REFUSED = []


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if not writing:
        return
    raw = os.fsdecode(args[0])
    if not os.path.isabs(raw):
        return
    path = Path(raw).resolve()
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
        REFUSED.append(str(path))
        raise PermissionError(f"the post-report verification may not write into the repository: {path}")


sys.addaudithook(_hook)

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, env=GIT_ENV).stdout.strip()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


out = ROOT / "outputs/experiment-024"
record = json.loads((HERE / "report_record.json").read_text(encoding="utf-8"))
print(f"launcher: report {record['report_started_at']} → {record['report_ended_at']}; exit {record['exit_status']}; error {record['error']}", flush=True)
print(f"counts {record['counts']}; events {record['events']}", flush=True)
check("report invoked exactly once; the check-only run did not invoke it", record["counts"]["report_invocations"] == 1
      and json.loads((HERE / "check_record.json").read_text())["counts"]["report_invocations"] == 0)
check("zero model or tokenizer loads, prompts, captures, measurements, ledger entries or module calls attempted",
      all(record["counts"][key] == 0 for key in ("model_loads_refused", "tokenizer_loads_refused", "entry_points_refused", "module_calls_refused")))
check("writes: report.md once and one atomic state write; nothing else in the repository; no forbidden read",
      set(record["events"]["output_writes"]) - {"report.md"} and all(name.startswith(".results-") for name in set(record["events"]["output_writes"]) - {"report.md"})
      and record["events"]["output_writes"].get("report.md") == 1 and len(record["events"]["output_writes"]) == 2
      and record["events"]["repository_writes_outside_outputs"] == [] and record["events"]["forbidden_reads"] == [] and record["events"]["hook_errors"] == 0)

before_bytes = (HERE / "results-before-report.json").read_bytes()
check("the pre-report byte copy is the confirm-final state file e636210c…", sha(before_bytes) == "e636210c84a4f3f329e9a5fde5baa493d730b6bf77cb419273056e6c89085a7a")
before = rd.load_results_state(HERE / "results-before-report.json")
after_bytes = (out / "results.json").read_bytes()
after = rd.load_results_state(out / "results.json")
report_bytes = (out / "report.md").read_bytes()
record_path = ROOT / rr.CALIBRATION_RELATIVE_PATH
calibration = json.loads(record_path.read_text(encoding="utf-8"))
print(f"report.md: {len(report_bytes)} bytes, sha256 {sha(report_bytes)}; state after {after['state_sha256']} (file {sha(after_bytes)})", flush=True)
check("report.md == rr.render_report(the confirm-final state, the installed calibration record), byte for byte",
      report_bytes == rr.render_report(before, calibration).encode("utf-8"))
check("the state records this report (path, sha256) and the report phase complete",
      after["report"]["sha256"] == sha(report_bytes) == pm.sha256_text(report_bytes.decode("utf-8")) and after["phases"]["report"]["status"] == "complete")
reverted = {key: value for key, value in after.items() if key != "state_sha256"}
reverted["report"] = None
reverted["phases"] = {**reverted["phases"], "report": {"status": "not_started"}}
digest = pm.sha256_text(pm.canonical_json(reverted))
file_bytes = (pm.canonical_json({**reverted, "state_sha256": digest}) + "\n").encode("utf-8")
check("reverting only the report entries reproduces the confirm-final state 076ab9f9… and its file bytes e636210c…",
      digest == "076ab9f984f3ba6244b4df767d8c8fd7ebd40a634521a46a3b881d7da1604e92" and file_bytes == before_bytes)
diff = sorted(key for key in set(after) | set(before) if after.get(key) != before.get(key) and key != "state_sha256")
check("the only top-level differences are report and phases (report only)", diff == ["phases", "report"]
      and {k for k in after["phases"] if after["phases"][k] != before["phases"][k]} == {"report"}, diff)
check("the ledger is unchanged: 4,320 keys, the same sorted digest 5f386b53…", after["executed_prompt_keys"] == before["executed_prompt_keys"]
      and len(after["executed_prompt_keys"]) == 4_320 and pm.sha256_text(pm.canonical_json(sorted(after["executed_prompt_keys"]))).startswith("5f386b53"))
check("the measurements are unchanged (b21babe1…, 76,314,087 bytes)", sha((out / "stage2-measurements.pt").read_bytes()) == "b21babe189d1cccb9ae359cfabf74baaa3a8337cac04612bc0d4aaa232b4dc93"
      and (out / "stage2-measurements.pt").stat().st_size == 76_314_087)
check("the confirmation block and the result are unchanged", after["confirmation"] == before["confirmation"])

text = report_bytes.decode("utf-8")
results = after["confirmation"]["results"]
wanted = {"ρ": "ρ = 0.6108818011257036", "null₉₇.₅": "null₉₇.₅ 0.3136960600375234", "F_ρ": "F_ρ 0.24411074612857814",
          "K": "K = 1 of 12870 assignments", "bound": "(bound 321)", "p": "exact p = 1/12870", "primary PASS": "**PASS** — ρ ≥ max(F_ρ, null₉₇.₅)",
          "D_EN": "D_EN = 0.5680373703974243", "label": "**`NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS`**",
          "threshold": "Effective threshold max(F_ρ, null₉₇.₅) = 0.3136960600375234, bound by null_975"}
for name, fragment in wanted.items():
    check(f"the report states {name}", fragment in text, fragment)
check("the recorded result is the frozen one", (results["primary"]["rho"], results["guard"]["K"], results["outcome"]["label"])
      == (0.6108818011257036, 1, "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS"))
flag = "at least as strongly as the exposed-like relationship"
print(f"note: the frozen renderer's PASS reading contains {flag!r}: {flag in text} (the formal interpretation is ρ ≥ max(F_ρ, null₉₇.₅); see the README)", flush=True)
check("the report carries the frozen disclaimers (not shown: causation, linguistic nounhood, other E/N properties)",
      "that nounness causes the attention change" in text and "that the score measures linguistic nounhood" in text
      and "that no other property that differs between ordinary nouns and these adjectives explains E > N" in text)

evidence = ROOT / "experiments/024-readout-routing-nounness/evidence"
for directory in ("confirm-2026-09-25", "postinstall-review-2026-09-25", "confirmation-review-2026-09-25"):
    lines = (evidence / directory / "SHA256SUMS").read_text().splitlines()
    ok = all(sha((evidence / directory / line.split("  ", 1)[1]).read_bytes()) == line.split("  ", 1)[0] for line in lines)
    check(f"evidence {directory}/SHA256SUMS: {len(lines)} entries verify", ok)
head, origin = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
tree = git("status", "--porcelain", "--untracked-files=all")
print(f"HEAD {head}; origin {origin}; remote {remote}; tree {'clean' if tree == '' else tree}", flush=True)
check("HEAD == origin == remote == b80409b; tree clean", head == origin == remote and head.startswith("b80409b") and tree == "")
check("no report.md or results.json is tracked (gitignored outputs)", git("ls-files", "outputs/experiment-024") == "")
check("no repository write by this verification", not REFUSED, REFUSED)
print("FAILURES:", failures if failures else "none", flush=True)
sys.exit(1 if failures else 0)
