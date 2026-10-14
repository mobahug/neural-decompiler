"""Runs each v2 main-block scenario in its own process with a fresh scratch git repository and launcher directory under
``v2/main_block/<scenario>/``, and tabulates exit code, sentinel (and its mode), records, the state on disk and notes.
``sighup_under_nohup`` starts through /usr/bin/nohup; the two TTY scenarios start under ``script -q /dev/null`` (a pty).
Output: out/v2_e2_main_block.json."""
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import REVIEW, write_json  # noqa: E402

PY = "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/.venv/bin/python"
SCENARIO = Path(__file__).resolve().parent / "e2_main_block_scenario.py"
SCENARIOS = ["correct", "incident_after_ledger", "sigterm_after_ledger", "sighup_under_nohup", "sighup_without_nohup", "sigterm_before_ledger",
             "sigterm_during_secondary", "sigterm_in_post_block", "sigterm_during_tree_after", "tty_stdout", "tty_stderr_only", "refusal_before_ledger",
             "preflight_refuses", "sentinel_exists", "git_fails_after", "keys_not_all_run", "refusal_swallowed", "forbidden_write_recorded"]
env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0", "TMPDIR": str(REVIEW / "tmpdir"), "CONFIRM101_MODE": "production"}


def setup(name):
    base = REVIEW / "v2/main_block" / name
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
    return base


rows = {}
for name in SCENARIOS:
    base = setup(name)
    command = [PY, "-B", str(SCENARIO), name]
    stdout_target = subprocess.PIPE
    if name == "sighup_under_nohup":
        command = ["/usr/bin/nohup", *command]
    if name == "tty_stdout":
        command = ["/usr/bin/script", "-q", "/dev/null", *command]
    if name == "tty_stderr_only":
        command = ["/usr/bin/script", "-q", "/dev/null", "/bin/sh", "-c", f'exec "$0" "$@" > "{base}/stdout.log"', *command]
    proc = subprocess.run(command, env=env, stdout=stdout_target, stderr=subprocess.PIPE, text=True, timeout=300, stdin=subprocess.DEVNULL)
    here = base / "launcher_dir"
    records = sorted(p.name for p in here.glob("production_run_*.json"))
    record = json.loads((here / records[-1]).read_text()) if records else None
    state_path = base / "repo/outputs/experiment-101/results.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    notes = json.loads((base / "notes.json").read_text()) if (base / "notes.json").exists() else {}
    sentinel = here / "confirm.sentinel"
    rows[name] = {"exit": proc.returncode, "sentinel": sentinel.exists(), "sentinel_mode": oct(stat.S_IMODE(sentinel.stat().st_mode)) if sentinel.exists() else None,
                  "records": records, "record_error": record.get("error") if record else None, "record_exit_status": record.get("exit_status") if record else None,
                  "record_post": record.get("post") if record else None, "record_signals": record["events"].get("signals") if record else None,
                  "state_confirm_status": state["phases"]["confirm"]["status"] if state else None,
                  "state_incident": (state.get("confirmation") or {}).get("incident", {}).get("message") if state else None,
                  "state_secondary_failures": ((state.get("confirmation") or {}).get("secondary") or {}).get("failures") if state else None,
                  "partial_file": (base / "repo/outputs/experiment-101/stage2-partial.pt").exists(), "notes": notes,
                  "stdout_tail": (proc.stdout or "")[-500:], "stderr_tail": (proc.stderr or "")[-500:]}
    r = rows[name]
    print(f"{name:<26} exit={r['exit']:<3} sentinel={str(r['sentinel']):<5} mode={str(r['sentinel_mode']):<6} records={len(records)} "
          f"error={str(r['record_error'])[:52]:<52} state={r['state_confirm_status']} incident={str(r['state_incident'])[:28]} "
          f"secondary_failures={'yes' if r['state_secondary_failures'] else 'no'}")
write_json("v2_e2_main_block.json", rows)
