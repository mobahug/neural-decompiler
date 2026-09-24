"""Final integrity: every file under outputs/experiment-023/ and every committed 023 artifact re-hashed and compared with
the values bound by the task, the pre-review snapshot, the state and the HEAD blobs; mtimes unchanged; HEAD == origin;
the tree clean (untracked included). Read-only git only."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirmreview023")
import guard  # noqa: E402  FIRST

import hashlib  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(guard.ROOT)
EV = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm023")
HEAD = "c7efec7f32709dc20ecb83cae16bb73927a14366"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail != "" else ""), flush=True)
    if not ok:
        FAIL.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# the pre-review snapshot (hashes and output mtimes), parsed as data
snap_hash, snap_mtime = {}, {}
for line in (EV / "snapshot_before_review.txt").read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if len(parts) == 2 and len(parts[0]) == 64:
        snap_hash[parts[1]] = parts[0]
    elif len(parts) == 3 and parts[0].startswith("outputs/"):
        snap_mtime[parts[0]] = int(parts[2])
task = {"outputs/experiment-023/results.json": "328be2b688fd8cef34a523c308dc4dd5790c703a90ab0cc6f66ac377a0ec21f2",
        "outputs/experiment-023/y2-table.f64": "4623adde5eb53e14b0368e5b4d8b91bd6f2513bd522ba074bc55ab747970d2c2",
        "outputs/experiment-023/y2-table.json": "e87091ef", "outputs/experiment-023/stage2-measurements.pt": "13b7da17",
        "experiments/023-block0-completion/preregistration-lock.json": "4bd5a5b1", "experiments/023-block0-completion/locked-y1-table.f64": "37603f05",
        "experiments/023-block0-completion/locked-y1-table.json": "3bf85e18", "experiments/023-block0-completion/calibration-v1.json": "94db6df2",
        "experiments/023-block0-completion/confirmation-v1.json": "5fadfa50"}
outputs = sorted(p for p in (ROOT / "outputs/experiment-023").iterdir())
committed = sorted(p for p in (ROOT / "experiments/023-block0-completion").rglob("*") if p.is_file())
for p in outputs + committed:
    rel = str(p.relative_to(ROOT))
    h = sha(p)
    ok = True
    notes = []
    if rel in snap_hash:
        ok &= h == snap_hash[rel]
        notes.append("== pre-review snapshot")
    if rel in snap_mtime:
        ok &= int(os.stat(p).st_mtime) == snap_mtime[rel]
        notes.append("mtime unchanged")
    if rel in task:
        ok &= h.startswith(task[rel])
        notes.append("== task-bound digest")
    if rel.startswith("experiments/"):
        ok &= git("show", f"HEAD:{rel}") == p.read_bytes()
        notes.append("== HEAD blob")
    check(f"{rel}", ok and bool(notes), f"{h[:16]} ({', '.join(notes)})")
check("every output file was in the pre-review snapshot (nothing added)", {str(p.relative_to(ROOT)) for p in outputs} == {k for k in snap_hash if k.startswith("outputs/experiment-023/")})
check("HEAD == c7efec7 == origin/main (git ls-remote, no fetch)", git("rev-parse", "HEAD").decode().strip() == HEAD == git("ls-remote", "origin", "refs/heads/main").decode().split()[0])
status = git("status", "--porcelain", "--untracked-files=all").decode()
check("git status --porcelain --untracked-files=all is empty", status == "", repr(status[:200]))
print("\nFINAL:", "ALL PASS" if not FAIL else f"FAILED {FAIL}")
