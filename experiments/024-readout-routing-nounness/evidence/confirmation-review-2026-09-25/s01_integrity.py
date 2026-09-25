"""Item 1: repository / phase integrity (read-only)."""
import rguard  # noqa: F401  (first)
from rguard import REPO

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(REPO)
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail != '' else ''}", flush=True)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canon(v):  # own canonical JSON (same definition as the protocol's)
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_digest(payload):
    return hashlib.sha256(canon({k: v for k, v in payload.items() if k != "content_sha256"}).encode("utf-8")).hexdigest()


def own_blob(p):
    data = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


HEAD = "af160cee0389bcdaf12bcbd92a8616f142777a52"
head = git("rev-parse", "HEAD")
origin = git("rev-parse", "origin/main")
remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
status = git("status", "--porcelain", "--untracked-files=all")
check("HEAD == af160ce", head == HEAD, head)
check("origin/main == HEAD", origin == HEAD, origin)
check("ls-remote origin refs/heads/main == HEAD", remote == HEAD, remote)
check("tree clean (status --porcelain --untracked-files=all empty)", status == "", repr(status))
print("   log:", git("log", "--oneline", "-6").replace("\n", " | "))

X = ROOT / "experiments/024-readout-routing-nounness"
expected = {
    "confirmation-v1.json": ("68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60", "87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87"),
    "calibration-v1.json": ("81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d", "09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8"),
    "preregistration-lock.json": ("5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d", "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739"),
}
payloads = {}
for name, (fsha, csha) in expected.items():
    p = X / name
    payload = json.loads(p.read_text(encoding="utf-8"))
    payloads[name] = payload
    fs, cs_rec, cs_own = sha256_file(p), payload.get("content_sha256"), content_digest(payload)
    check(f"{name}: file sha256", fs == fsha, fs)
    check(f"{name}: recorded content_sha256 == expected == own recomputation", cs_rec == csha == cs_own, f"{cs_rec} / own {cs_own}")
    tracked = git("ls-files", "--error-unmatch", str(p.relative_to(ROOT)))
    blob_head = git("rev-parse", f"HEAD:{p.relative_to(ROOT)}")
    check(f"{name}: tracked and HEAD blob == own blob of working file", bool(tracked) and blob_head == own_blob(p), blob_head)
pr = X / "preregistration.md"
check("preregistration.md file sha256", sha256_file(pr) == "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54", sha256_file(pr))
check("preregistration.md tracked and HEAD blob == own blob", git("rev-parse", f"HEAD:{pr.relative_to(ROOT)}") == own_blob(pr))
# candidates in outputs are byte-identical to the installed files
out = ROOT / "outputs/experiment-024"
for cand, inst in (("candidate-calibration.json", "calibration-v1.json"), ("candidate-lock.json", "preregistration-lock.json"), ("candidate-preregistration.md", "preregistration.md")):
    check(f"outputs/{cand} byte-identical to installed {inst}", (out / cand).read_bytes() == (X / inst).read_bytes())

lock = payloads["preregistration-lock.json"]
calib = payloads["calibration-v1.json"]
conf = payloads["confirmation-v1.json"]
check("lock binds the installed calibration (file & content)", lock["calibration"] == {"path": "experiments/024-readout-routing-nounness/calibration-v1.json",
      "file_sha256": expected["calibration-v1.json"][0], "content_sha256": expected["calibration-v1.json"][1]}, lock["calibration"])
check("lock binds the installed confirmation (file & content)", lock["confirmation_024"]["file_sha256"] == expected["confirmation-v1.json"][0]
      and lock["confirmation_024"]["content_sha256"] == expected["confirmation-v1.json"][1], {k: lock["confirmation_024"][k] for k in ("file_sha256", "content_sha256", "manifest_size", "manifest_sha256")})
check("lock protocol_code_commit == be74d23…", lock["protocol_code_commit"].startswith("be74d23"), lock["protocol_code_commit"])
check("lock module path/blob", lock["module"] == {"path": "src/neural_decompiler/readout_routing.py", "blob": "e6cb37767d1d06c6ff40804a88eab569723afdb5"}, lock["module"])

# --- the results state (own digest verification) ---------------------------------------------------------------------
sp = out / "results.json"
raw = sp.read_bytes()
state = json.loads(raw.decode("utf-8"))
recorded = state.pop("state_sha256")
own = hashlib.sha256(canon(state).encode("utf-8")).hexdigest()
check("results.json file sha256 e636210c…", hashlib.sha256(raw).hexdigest() == "e636210c84a4f3f329e9a5fde5baa493d730b6bf77cb419273056e6c89085a7a", hashlib.sha256(raw).hexdigest())
check("state_sha256 recorded == own recomputation == 076ab9f9…", recorded == own == "076ab9f984f3ba6244b4df767d8c8fd7ebd40a634521a46a3b881d7da1604e92", f"{recorded} / {own}")
check("file bytes == canonical_json(state + digest) + newline (own)", raw == (canon({**state, "state_sha256": own}) + "\n").encode("utf-8"))
phases = {k: v for k, v in state["phases"].items()}
print("   phases:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "I7"} for k, v in phases.items()}))
check("confirm complete", phases["confirm"]["status"] == "complete")
check("report not started (phase, state['report'], no report.md)", phases["report"] == {"status": "not_started"} and state.get("report") is None and not (out / "report.md").exists())
check("calibrate complete, lock complete", phases["calibrate"]["status"] == "complete" and phases["lock"]["status"] == "complete")


def find_incidents(v, path=""):
    hits = []
    if isinstance(v, dict):
        for k, x in v.items():
            if "incident" in k.lower() or k in ("cross_check", "failures", "stop"):
                hits.append(f"{path}/{k}")
            hits += find_incidents(x, f"{path}/{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            hits += find_incidents(x, f"{path}[{i}]")
    return hits


inc = find_incidents(state)
check("no incident / cross_check / stop / descriptive failure key anywhere in the state", inc == [], inc)
check("confirm entry has exactly one start and one completion; confirm_commit == af160ce",
      phases["confirm"].get("confirm_commit") == HEAD and "started_at" in phases["confirm"] and "completed_at" in phases["confirm"], {k: v for k, v in phases["confirm"].items() if k != "I7"})
check("I7 bitwise_equal True with nothing differing", phases["confirm"]["I7"]["bitwise_equal"] is True and phases["confirm"]["I7"]["differing"] == [], phases["confirm"]["I7"])
check("state protocol_code_commit is the calibrate commit, not changed", state["protocol_code_commit"] == phases["calibrate"]["commit"], state["protocol_code_commit"])
check("state lock.content_sha256 == installed lock content", state["lock"]["content_sha256"] == lock["content_sha256"])
check("state confirmation.lock_sha256 == lock content", state["confirmation"]["lock_sha256"] == lock["content_sha256"] and phases["confirm"]["lock_sha256"] == lock["content_sha256"])

# --- the canonical module: imported from the repository file; blob bindings ---------------------------------------------
sys.path.insert(0, str(ROOT / "src"))
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402

mod_file = Path(rr.__file__).resolve()
check("rr imported from the repository file", mod_file == (ROOT / "src/neural_decompiler/readout_routing.py").resolve(), str(mod_file))
b_lock = lock["module"]["blob"]
b_own = own_blob(ROOT / "src/neural_decompiler/readout_routing.py")
b_git = git("rev-parse", "HEAD:src/neural_decompiler/readout_routing.py")
b_rr = rr.own_blob()
check("module blob: lock == own sha1 == git HEAD == rr.own_blob() == e6cb3776…", len({b_lock, b_own, b_git, b_rr, "e6cb37767d1d06c6ff40804a88eab569723afdb5"}) == 1,
      {"lock": b_lock, "own": b_own, "git": b_git, "rr.own_blob": b_rr})
frozen_now = rr.assert_frozen_blobs()
own_frozen = {name: own_blob(ROOT / "src/neural_decompiler" / name) for name in rr.FROZEN_BLOBS}
check("the 12 pinned modules: rr.assert_frozen_blobs passes and own sha1 of each == FROZEN_BLOBS == lock.module_blobs",
      frozen_now == rr.FROZEN_BLOBS == own_frozen == lock["module_blobs"], len(own_frozen))

# --- the canonical loader agrees; confirm is refused now --------------------------------------------------------------
st = rd.load_results_state(sp)
try:
    rr.assert_phase_allowed("confirm", st)
    refused = None
except rr.PhaseError as e:  # noqa: PERF203
    refused = str(e)
check("rr.assert_phase_allowed('confirm', state) refuses now", refused is not None, refused)
try:
    rr.assert_phase_allowed("lock", st)
    lock_refused = None
except rr.PhaseError as e:
    lock_refused = str(e)
check("lock also refused (already complete)", lock_refused is not None, lock_refused)
try:
    rr.assert_phase_allowed("report", st)
    report_ok = True
except rr.PhaseError as e:
    report_ok = str(e)
check("report is allowed next", report_ok is True, report_ok)

# --- no scientific path changed after the lock -------------------------------------------------------------------------
lock_commit = lock["protocol_code_commit"]
anc = subprocess.run(["git", "merge-base", "--is-ancestor", lock_commit, "HEAD"], cwd=ROOT, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).returncode
changed = [line for line in git("diff", "--no-renames", "--name-only", lock_commit, "HEAD").splitlines() if line.strip()]
print(f"   changed since the lock commit {lock_commit[:7]}: {changed}")
sci = rr.scientific_changes(changed)
check("lock commit is an ancestor of HEAD", anc == 0)
check("rr.scientific_changes(diff be74d23..HEAD) == []", sci == [], sci)
own_class = [p for p in changed if not (p.startswith("experiments/024-readout-routing-nounness/evidence/")
                                        or p in ("experiments/024-readout-routing-nounness/README.md", "experiments/024-readout-routing-nounness/preregistration-lock.json",
                                                 "experiments/024-readout-routing-nounness/preregistration.md"))]
check("own classification: only the installed lock/preregistration, the README and evidence changed", own_class == [], own_class)
src_changes = [line for line in git("diff", "--no-renames", "--name-only", lock_commit, "HEAD", "--", "src", "experiments/024-readout-routing-nounness/run.py", "tests").splitlines() if line]
check("no change under src/, tests/ or run.py since the lock commit", src_changes == [], src_changes)
calib_commit = phases["calibrate"]["commit"]
changed_c = [line for line in git("diff", "--no-renames", "--name-only", calib_commit, "HEAD").splitlines() if line.strip()]
check("rr.scientific_changes since the calibrate commit == []", rr.scientific_changes(changed_c) == [], (calib_commit[:7], rr.scientific_changes(changed_c)))
print(f"S01 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
