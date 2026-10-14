"""Runs each main-block scenario (``e_main_block_scenario.py``) in its own process, each with a fresh scratch git
repository and launcher directory under ``main_block/<scenario>/``, and tabulates exit code, sentinel, records, the state
on disk and the console. ``sighup_under_nohup`` is started through ``/usr/bin/nohup``, as the launcher's docstring
advises; ``nohup_control`` shows the same SIGHUP against a process that keeps nohup's SIG_IGN. Output:
out/e_main_block.json.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REVIEW, write_json  # noqa: E402

PY = "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python"
SCENARIO = Path(__file__).resolve().parent / "e_main_block_scenario.py"
SCENARIOS = ["correct", "incident_after_ledger", "sigterm_after_ledger", "sighup_under_nohup", "sigterm_during_tree_after", "refusal_before_ledger",
             "preflight_refuses", "sentinel_exists", "earlier_records", "git_fails_after", "keys_not_all_run", "refusal_swallowed"]
env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0", "TMPDIR": str(REVIEW / "tmpdir")}


def setup(name):
    base = REVIEW / "main_block" / name
    if base.exists():
        shutil.rmtree(base)
    (base / "launcher_dir").mkdir(parents=True)
    repo = base / "repo"
    (repo / "outputs/experiment-101").mkdir(parents=True)
    (repo / ".gitignore").write_text("outputs/experiment-101/*\n")
    (repo / "README.md").write_text("scratch\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=review", "-c", "user.email=review@example.invalid", "commit", "-q", "-m", "scratch"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    state_code = ("import sys; from pathlib import Path; from neural_decompiler import readout_routing as rr; "
                  "rr.write_state_atomic(Path(sys.argv[1]), {'experiment': '101', 'protocol_code_commit': '0'*40, "
                  "'phases': {'stage1': {'status': 'complete', 'awaiting_commit': True}, 'confirm': {'status': 'not_started'}}, "
                  "'executed_prompt_keys': ['s1|ref|581|S1'], 'confirmation': None})")
    subprocess.run([PY, "-B", "-c", state_code, str(repo / "outputs/experiment-101/results.json")], check=True, env=env)
    if name == "sentinel_exists":
        (base / "launcher_dir/confirm.sentinel").write_text("an earlier launch")
    if name == "earlier_records":
        (base / "launcher_dir/production_run_20261014T000000000000Z.json").write_text("{}\n")
    return base


rows = {}
for name in SCENARIOS:
    base = setup(name)
    command = [PY, "-B", str(SCENARIO), name]
    if name == "sighup_under_nohup":
        command = ["/usr/bin/nohup", *command]
    proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    here = base / "launcher_dir"
    records = sorted(p.name for p in here.glob("production_run_*.json"))
    newest = [json.loads((here / r).read_text()) for r in records if (here / r).stat().st_size > 3]
    state_path = base / "repo/outputs/experiment-101/results.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    notes = json.loads((base / "notes.json").read_text()) if (base / "notes.json").exists() else {}
    record = newest[-1] if newest else None
    rows[name] = {"exit": proc.returncode, "sentinel": (here / "confirm.sentinel").exists(), "sentinel_content": (here / "confirm.sentinel").read_text()[:16] if (here / "confirm.sentinel").exists() else None,
                  "records": records, "record_error": record.get("error") if record else None, "record_exit_status": record.get("exit_status") if record else None,
                  "record_post": record.get("post") if record else None, "record_has_logs": ("logs" in record) if record else None,
                  "record_signals": record["events"].get("signals") if record else None, "record_tree_after": record.get("tree_after") if record else None,
                  "state_confirm_status": state["phases"]["confirm"]["status"] if state else None,
                  "state_incident": (state.get("confirmation") or {}).get("incident", {}).get("message") if state else None,
                  "state_partial": bool((state.get("confirmation") or {}).get("partial")) if state else None,
                  "partial_file": (base / "repo/outputs/experiment-101/stage2-partial.pt").exists(),
                  "notes": notes, "stdout_tail": proc.stdout[-700:], "stderr_tail": proc.stderr[-700:]}
    print(f"{name:<28} exit={proc.returncode:<4} sentinel={rows[name]['sentinel']!s:<5} records={len(records)} record_error={str(rows[name]['record_error'])[:60]!s:<60} "
          f"state={rows[name]['state_confirm_status']} incident={str(rows[name]['state_incident'])[:40]} partial={rows[name]['partial_file']}")

# the counterfactual: nohup's SIG_IGN kept (no handler installed), the same SIGHUP is harmless
control = ("import os, signal, time; print('SIGHUP disposition:', signal.getsignal(signal.SIGHUP), flush=True); os.kill(os.getpid(), signal.SIGHUP); "
           "time.sleep(0.5); print('still running after SIGHUP', flush=True)")
proc = subprocess.run(["/usr/bin/nohup", PY, "-B", "-c", control], env=env, capture_output=True, text=True, timeout=60)
rows["nohup_control"] = {"exit": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr[-300:]}
print("nohup_control", proc.returncode, proc.stdout.strip().replace("\n", " | "))
write_json("e_main_block.json", rows)
