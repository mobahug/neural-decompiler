"""Item 5: the direct module-blob assertion for the future confirm launcher; the 12 pinned module blobs; import origin."""
import guard  # noqa: F401
from guard import REPO, git, git_blob, summary

import importlib.util
import json
import os

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


EXPECTED = "e6cb37767d1d06c6ff40804a88eab569723afdb5"
REL = "src/neural_decompiler/readout_routing.py"
with open(os.path.join(REPO, REL), "rb") as h:
    data = h.read()
own = git_blob(data)
spec_origin = importlib.util.find_spec("neural_decompiler.readout_routing").origin
from neural_decompiler import readout_routing as rr  # noqa: E402

observed = {"own sha1(b'blob <len>\\0'+bytes)": own, "rr.own_blob()": rr.own_blob(), "git rev-parse HEAD:" + REL: git("rev-parse", f"HEAD:{REL}").strip()}
with open(os.path.join(REPO, rr.LOCK_RELATIVE_PATH), encoding="utf-8") as h:
    lock = json.load(h)
observed["installed_lock['module']['blob']"] = lock["module"]["blob"]
print(f"expected: {EXPECTED}")
for k, v in observed.items():
    print(f"observed: {v}  <- {k}")
check("all four equal the expected blob e6cb3776…", all(v == EXPECTED for v in observed.values()))
check("installed_lock['module']['path'] == src/neural_decompiler/readout_routing.py", lock["module"]["path"] == REL, lock["module"]["path"])
check("rr.__file__ resolves to the repository file", os.path.realpath(rr.__file__) == os.path.realpath(os.path.join(REPO, REL)), rr.__file__)
check("find_spec origin == the repository file (no shadow copy)", os.path.realpath(spec_origin) == os.path.realpath(os.path.join(REPO, REL)), spec_origin)
for rev in ("be74d23", "44c3c2b"):
    b = git("rev-parse", f"{rev}:{REL}").strip()
    check(f"blob at {rev} == expected (unchanged since)", b == EXPECTED, b)
log = git("log", "--format=%h %s", "44c3c2b..HEAD", "--", REL).strip()
check("no commit touched readout_routing.py after 44c3c2b", log == "", log)
run_head, run_lock = git("rev-parse", "HEAD:experiments/024-readout-routing-nounness/run.py").strip(), git("rev-parse", "be74d23:experiments/024-readout-routing-nounness/run.py").strip()
with open(os.path.join(REPO, "experiments/024-readout-routing-nounness/run.py"), "rb") as h:
    run_bytes = h.read()
check("run.py: own blob == HEAD blob == be74d23 blob", git_blob(run_bytes) == run_head == run_lock, run_head)

# The 12 pinned modules: own sha1 of the imported file == FROZEN_BLOBS == git HEAD == the lock's and the record's module_blobs.
with open(os.path.join(REPO, rr.CALIBRATION_RELATIVE_PATH), encoding="utf-8") as h:
    record = json.load(h)
bad = []
for name, module in rr._FROZEN_MODULES.items():
    path = os.path.realpath(module.__file__)
    with open(path, "rb") as h:
        blob = git_blob(h.read())
    head = git("rev-parse", f"HEAD:src/neural_decompiler/{name}").strip()
    same = blob == rr.FROZEN_BLOBS[name] == head == lock["module_blobs"][name] == record["module_blobs"][name] and path == os.path.join(REPO, "src/neural_decompiler", name)
    print(f"   {name:28s} {blob} {'ok' if same else 'MISMATCH'}")
    if not same:
        bad.append(name)
check("12 pinned modules: own blob == FROZEN_BLOBS == HEAD == lock == record, imported from the repo", not bad and len(rr._FROZEN_MODULES) == 12, bad)
check("lock.module_blobs == rr.FROZEN_BLOBS (12 keys)", lock["module_blobs"] == rr.FROZEN_BLOBS and len(lock["module_blobs"]) == 12)
check("rr.assert_frozen_blobs() passes", rr.assert_frozen_blobs() == rr.FROZEN_BLOBS)
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"ITEM5 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
