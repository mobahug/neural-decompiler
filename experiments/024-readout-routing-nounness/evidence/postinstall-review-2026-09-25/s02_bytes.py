"""Item 2: installed bytes == HEAD blobs == 3affe55 blobs == gitignored candidates; expected sha256s; own canonical JSON of the
lock; preregistration == rr.render_preregistration(lock) byte for byte; the state's lock entry binds the same digests."""
import guard  # noqa: F401
from guard import REPO, canonical, git, git_blob, sha256_bytes, summary

import json
import math
import os
import subprocess

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


LOCK = "experiments/024-readout-routing-nounness/preregistration-lock.json"
PREREG = "experiments/024-readout-routing-nounness/preregistration.md"
CAND = {LOCK: "outputs/experiment-024/candidate-lock.json", PREREG: "outputs/experiment-024/candidate-preregistration.md"}
EXPECTED = {LOCK: "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d", PREREG: "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54"}
CONTENT = "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739"


def show(rev_path: str) -> bytes:
    return subprocess.run(["git", "show", rev_path], cwd=REPO, check=True, capture_output=True).stdout


data = {}
for path in (LOCK, PREREG):
    with open(os.path.join(REPO, path), "rb") as h:
        wt = h.read()
    with open(os.path.join(REPO, CAND[path]), "rb") as h:
        cand = h.read()
    head, inst = show(f"HEAD:{path}"), show(f"3affe55:{path}")
    blob_head = git("rev-parse", f"HEAD:{path}").strip()
    data[path] = wt
    check(f"{os.path.basename(path)}: working tree sha256 == expected", sha256_bytes(wt) == EXPECTED[path], sha256_bytes(wt))
    check(f"{os.path.basename(path)}: working tree == git show HEAD:", wt == head, f"{len(wt)} bytes")
    check(f"{os.path.basename(path)}: working tree == git show 3affe55:", wt == inst)
    check(f"{os.path.basename(path)}: working tree == gitignored candidate {CAND[path]}", wt == cand, sha256_bytes(cand))
    check(f"{os.path.basename(path)}: own git blob of bytes == HEAD tree blob", git_blob(wt) == blob_head, blob_head)
    check(f"{os.path.basename(path)}: no CR bytes, ends with exactly one newline", b"\r" not in wt and wt.endswith(b"\n") and not wt.endswith(b"\n\n"))
    check(f"{os.path.basename(path)}: path absent at be74d23 (first added by 3affe55)",
          subprocess.run(["git", "cat-file", "-e", f"be74d23:{path}"], cwd=REPO, capture_output=True).returncode != 0)
    ignored = subprocess.run(["git", "ls-files", "--error-unmatch", CAND[path]], cwd=REPO, capture_output=True).returncode
    check(f"{CAND[path]} is untracked (gitignored candidate)", ignored != 0)

# Own canonical-JSON check of the lock.
text = data[LOCK].decode("utf-8")
lock = json.loads(text)
floats = []


def walk(value):
    if isinstance(value, float):
        floats.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            walk(item)
    elif isinstance(value, list):
        for item in value:
            walk(item)


walk(lock)
check("lock parses; every float finite", all(math.isfinite(v) for v in floats), f"{len(floats)} floats")
check("lock file == own canonical JSON (sorted keys, compact, UTF-8) + one newline", text == canonical(lock) + "\n")
unsigned = {k: v for k, v in lock.items() if k != "content_sha256"}
own_content = sha256_bytes(canonical(unsigned).encode("utf-8"))
check("own content digest (content_sha256 key removed) == expected a39668bf…", own_content == CONTENT, own_content)
check("lock.content_sha256 field == own content digest", lock["content_sha256"] == own_content)
check("no NaN/Infinity tokens in the lock text", "NaN" not in text and "Infinity" not in text)

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

check("canonical route rc.content_digest(lock) == own content digest", rc.content_digest(lock) == own_content)
check("canonical route pm.canonical_json(lock)+'\\n' == file text", pm.canonical_json(lock) + "\n" == text)
rendered = rr.render_preregistration(lock)
check("preregistration.md == rr.render_preregistration(installed lock), byte for byte", rendered.encode("utf-8") == data[PREREG], sha256_bytes(rendered.encode("utf-8")))

with open(os.path.join(REPO, "outputs/experiment-024/results.json"), "rb") as h:
    state_bytes = h.read()
state = json.loads(state_bytes)
entry = state["lock"]
check("state.lock.content_sha256 == lock content digest", entry["content_sha256"] == own_content)
check("state.lock.preregistration_sha256 == sha256(preregistration.md)", entry["preregistration_sha256"] == sha256_bytes(data[PREREG]))
check("state.lock.calibration_content_sha256 == lock.calibration.content_sha256", entry["calibration_content_sha256"] == lock["calibration"]["content_sha256"])
check("state.lock.extrapolation == lock.fresh.extrapolation", entry["extrapolation"] == lock["fresh"]["extrapolation"])
print("state.lock keys:", sorted(entry.keys()))
# The lock binds the committed calibration record and freeze by file sha256.
for key, rel in (("calibration", "experiments/024-readout-routing-nounness/calibration-v1.json"), ("confirmation_024", "experiments/024-readout-routing-nounness/confirmation-v1.json")):
    with open(os.path.join(REPO, rel), "rb") as h:
        b = h.read()
    check(f"lock.{key}.file_sha256 == sha256 of committed {os.path.basename(rel)}", lock[key]["file_sha256"] == sha256_bytes(b), sha256_bytes(b)[:12])
    check(f"lock.{key}.content_sha256 == own content digest of committed {os.path.basename(rel)}",
          lock[key]["content_sha256"] == sha256_bytes(canonical({k: v for k, v in json.loads(b).items() if k != "content_sha256"}).encode()))
with open(os.path.join(REPO, "outputs/experiment-024/candidate-calibration.json"), "rb") as h:
    cc = h.read()
with open(os.path.join(REPO, "experiments/024-readout-routing-nounness/calibration-v1.json"), "rb") as h:
    ci = h.read()
check("installed calibration-v1.json == candidate-calibration.json (bytes)", cc == ci, sha256_bytes(ci))
print("lock top-level keys:", sorted(lock.keys()))
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"ITEM2 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
