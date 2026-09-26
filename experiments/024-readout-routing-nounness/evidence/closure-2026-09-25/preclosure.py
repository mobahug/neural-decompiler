"""Pre-closure verification of Experiment 024 (read-only; no model, no prompt, no phase; nothing written).

Checked: the repository and remote at `b80409b`; all four phases complete with no incident; the phase rule; the
measurements, the generated report and the final state; the reversal of the two report entries to the confirm-final
state; and the installed freeze, calibration, lock and preregistration. Any mismatch exits non-zero, and the closure
does not proceed.
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
    raw = os.fsdecode(args[0])
    if not os.path.isabs(raw):
        return
    path = Path(raw).resolve()
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
        REFUSED.append(str(path))
        raise PermissionError(f"the pre-closure verification may not write into the repository: {path}")


sys.addaudithook(_hook)

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
COMMIT = "b80409b1464c8ab59f82d026e104f4510247c69e"
failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, env=GIT_ENV).stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


head, origin = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
check("HEAD == origin/main == remote == b80409b", head == origin == remote == COMMIT, f"{head} / {origin} / {remote}")
check("tree clean (no untracked file), no stash", git("status", "--porcelain", "--untracked-files=all") == "" and git("stash", "list") == "")

out = ROOT / "outputs/experiment-024"
state = rd.load_results_state(out / "results.json")
phases = {name: entry["status"] for name, entry in state["phases"].items()}
check("all four phases complete", phases == {"calibrate": "complete", "lock": "complete", "confirm": "complete", "report": "complete"}, phases)
incidents = {name: entry.get("incidents") for name, entry in state["phases"].items() if entry.get("incidents")}
check("no incident anywhere (phases, calibration, confirmation); no stop; no descriptive failure",
      not incidents and not (state.get("calibration") or {}).get("incidents") and not (state.get("calibration") or {}).get("stop")
      and not state["confirmation"].get("incident") and not (state["confirmation"].get("descriptives") or {}).get("failures"))
for phase in ("calibrate", "lock", "confirm"):
    try:
        rr.assert_phase_allowed(phase, state)
        check(f"the phase rule refuses another {phase}", False)
    except rr.PhaseError as error:
        check(f"the phase rule refuses another {phase}", True, error)
try:
    rr.assert_phase_allowed("report", state)
    print("note: the phase rule permits re-rendering the report (idempotent by design); the one-time report launcher's sentinel refuses a second run",
          flush=True)
except rr.PhaseError as error:
    print(f"note: the phase rule refuses report: {error}", flush=True)

stage2, report = out / "stage2-measurements.pt", out / "report.md"
check("stage-2 measurements: 76,314,087 bytes, sha256 b21babe1…", stage2.stat().st_size == 76_314_087
      and sha(stage2) == "b21babe189d1cccb9ae359cfabf74baaa3a8337cac04612bc0d4aaa232b4dc93")
check("generated report: 8,310 bytes, sha256 8234ed9b…", report.stat().st_size == 8_310
      and sha(report) == "8234ed9b41533c5aa90172cfb1bc0b70c656848eaeff234e9dcfc9a5d09a8193" == state["report"]["sha256"])
check("final results state: digest 20ac6095…, file edbb6dc9…",
      state["state_sha256"] == "20ac6095a7e8534ff933a6405ae74b5f707aafd2ec9ac42e0ef5ffb1c5267376"
      and sha(out / "results.json") == "edbb6dc9c198265202cb567aef7a4d44761c19d658f277e687995759e454ea0c")
reverted = {key: value for key, value in state.items() if key != "state_sha256"}
reverted["report"] = None
reverted["phases"] = {**reverted["phases"], "report": {"status": "not_started"}}
digest = pm.sha256_text(pm.canonical_json(reverted))
file_sha = hashlib.sha256((pm.canonical_json({**reverted, "state_sha256": digest}) + "\n").encode("utf-8")).hexdigest()
check("removing only the two report entries reproduces the confirm-final state 076ab9f9… (file e636210c…)",
      digest == "076ab9f984f3ba6244b4df767d8c8fd7ebd40a634521a46a3b881d7da1604e92" and file_sha == "e636210c84a4f3f329e9a5fde5baa493d730b6bf77cb419273056e6c89085a7a")
installed = {
    rr.CONFIRMATION_RELATIVE_PATH: ("68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60", "87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87"),
    rr.CALIBRATION_RELATIVE_PATH: ("81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d", "09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8"),
    rr.LOCK_RELATIVE_PATH: ("5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d", "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739"),
    rr.PREREGISTRATION_RELATIVE_PATH: ("007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54", None),
}
for relative, (file_hash, content_hash) in installed.items():
    path = ROOT / relative
    ok = sha(path) == file_hash and subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, env=GIT_ENV).stdout == path.read_bytes()
    if content_hash:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ok = ok and payload["content_sha256"] == rc.content_digest(payload) == content_hash
    check(f"installed {relative.rsplit('/', 1)[1]} unchanged (file{', content' if content_hash else ''}, == HEAD)", ok)
check("the module blob is still e6cb3776…", rr.own_blob() == "e6cb37767d1d06c6ff40804a88eab569723afdb5" == json.loads((ROOT / rr.LOCK_RELATIVE_PATH).read_text())["module"]["blob"])
check("no scientific path changed since the lock", rr.scientific_changes(git("diff", "--no-renames", "--name-only", "be74d23", "HEAD").splitlines()) == [])
check("no repository write", not REFUSED, REFUSED)
print("FAILURES:", failures if failures else "none", flush=True)
sys.exit(1 if failures else 0)
