"""Pre-install verification for Experiment 024's lock (read-only; no model, no tokenizer, no prompt; confirm NOT run).

Before the reviewed candidates are copied byte for byte into the repository: the repository state, the candidate bytes
and digests, the results state, the committed freeze and calibration bytes, the phase state, the ledgers and the absence
of any fresh measurement. Any mismatch exits non-zero: nothing may be installed.
"""
import hashlib
import json
import os
import subprocess
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
        raise PermissionError(f"the pre-install check may not write into the repository: {path}")


sys.addaudithook(_hook)

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

LOCK_RUN = "be74d23d086bd12d894db1f73e6a1b4160aeebd5"
EXPECTED = {
    "lock_file": "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d",
    "lock_content": "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739",
    "prereg_file": "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54",
    "state": "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d",
    "state_file": "14459cdde6ec2d832c3965d5b0ebaadaf06cef6aa0b0460c3a31b29dd442b262",
    "freeze_file": "68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60",
    "freeze_content": "87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87",
    "calibration_file": "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d",
    "calibration_content": "09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8",
    "arrays_file": "a4cd548fd9c6084d58224f072d6c529cf4e81f17650fb50ccbc1d2c97f20aece",
}
failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


def git(*args, text=True):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=text).stdout


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


git("fetch", "-q", "origin")
head, origin = git("rev-parse", "HEAD").strip(), git("rev-parse", "origin/main").strip()
remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
check("HEAD == origin/main == ls-remote == be74d23", head == origin == remote == LOCK_RUN, f"{head} / {origin} / {remote}")
check("working tree clean (no untracked file either)", git("status", "--porcelain", "--untracked-files=all") == "")

out = ROOT / "outputs/experiment-024"
lock_bytes = (out / "candidate-lock.json").read_bytes()
prereg_bytes = (out / "candidate-preregistration.md").read_bytes()
lock = json.loads(lock_bytes.decode("utf-8"))
check("candidate lock file sha256", sha(lock_bytes) == EXPECTED["lock_file"], sha(lock_bytes))
check("candidate lock content sha256 (stored == recomputed == expected)", lock["content_sha256"] == rc.content_digest(lock) == EXPECTED["lock_content"], lock["content_sha256"])
check("candidate lock is canonical JSON plus one newline", lock_bytes == (pm.canonical_json(lock) + "\n").encode("utf-8"))
check("candidate preregistration sha256", sha(prereg_bytes) == EXPECTED["prereg_file"], sha(prereg_bytes))
check("candidate preregistration == render_preregistration(candidate lock)", prereg_bytes == rr.render_preregistration(lock).encode("utf-8"))

state_path = out / "results.json"
state = rd.load_results_state(state_path)
check("results state digest", state["state_sha256"] == EXPECTED["state"], state["state_sha256"])
check("results state file sha256 (unchanged since the lock run)", sha(state_path.read_bytes()) == EXPECTED["state_file"], sha(state_path.read_bytes()))
check("the state binds these candidates", state["lock"]["content_sha256"] == EXPECTED["lock_content"] and state["lock"]["preregistration_sha256"] == EXPECTED["prereg_file"])

for name, relative in (("freeze", rr.CONFIRMATION_RELATIVE_PATH), ("calibration", rr.CALIBRATION_RELATIVE_PATH)):
    data = (ROOT / relative).read_bytes()
    payload = json.loads(data.decode("utf-8"))
    at_head = git("show", f"HEAD:{relative}", text=False)
    check(f"{name}: installed bytes unchanged (file, content, == HEAD blob)",
          sha(data) == EXPECTED[f"{name}_file"] and payload["content_sha256"] == rc.content_digest(payload) == EXPECTED[f"{name}_content"] and at_head == data,
          sha(data))
    commits = git("log", "--format=%h", "--", relative).split()
    check(f"{name}: committed exactly once", len(commits) == 1, commits)

phases = state["phases"]
check("lock complete exactly once, at be74d23, no incident", phases["lock"] == {"commit": LOCK_RUN, "completed_at": phases["lock"]["completed_at"], "status": "complete"}
      and not phases["lock"].get("incidents"), phases["lock"])
try:
    rr.assert_phase_allowed("lock", state)
    check("a second lock is refused", False)
except rr.PhaseError as error:
    check("a second lock is refused", "lock already written" in str(error), error)
check("calibrate complete, no incident or stop", phases["calibrate"]["status"] == "complete" and not state["calibration"].get("incidents") and not state["calibration"].get("stop"))
check("confirm and report not started; no confirm incident", phases["confirm"] == {"status": "not_started"} and phases["report"] == {"status": "not_started"}, (phases["confirm"], phases["report"]))
check("prompt ledger and noun ledger empty", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
check("no confirmation or report in the state", state["confirmation"] is None and state["report"] is None)
listing = sorted(path.name for path in out.iterdir())
check("outputs/experiment-024 holds exactly the five pre-confirm files (no stage-2 file, no report)",
      listing == ["calibration-arrays.pt", "candidate-calibration.json", "candidate-lock.json", "candidate-preregistration.md", "results.json"], listing)
check("calibration arrays unchanged", sha((out / "calibration-arrays.pt").read_bytes()) == EXPECTED["arrays_file"])
check("the install targets do not exist yet", not (ROOT / rr.LOCK_RELATIVE_PATH).exists() and not (ROOT / rr.PREREGISTRATION_RELATIVE_PATH).exists())
check("no repository write", not REFUSED, REFUSED)
print("FAILURES:", failures if failures else "none", flush=True)
sys.exit(1 if failures else 0)
