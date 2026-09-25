"""Item 1: HEAD/origin/ls-remote/tree; the two commits' exact path lists; parents; scientific classification."""
import guard  # noqa: F401  (installs the audit hook first)
from guard import git, summary

import json
import os

EXPECTED_HEAD = "af160cee0389bcdaf12bcbd92a8616f142777a52"
LOCK_RUN = "be74d23d086bd12d894db1f73e6a1b4160aeebd5"
INSTALL = "3affe55cca991b8adbbff3b4e1b1c15339f4f058"
LOCK_PATHS = ["experiments/024-readout-routing-nounness/preregistration-lock.json", "experiments/024-readout-routing-nounness/preregistration.md"]
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


head = git("rev-parse", "HEAD").strip()
origin = git("rev-parse", "refs/remotes/origin/main").strip()
remote = git("ls-remote", "origin", "refs/heads/main").split()
porcelain = git("status", "--porcelain", "--untracked-files=all")
stash = git("for-each-ref", "refs/stash").strip()
check("HEAD == expected", head == EXPECTED_HEAD, head)
check("origin/main == expected", origin == EXPECTED_HEAD, origin)
check("ls-remote refs/heads/main == expected", remote[:2] == [EXPECTED_HEAD, "refs/heads/main"], remote)
check("tree clean (porcelain incl. untracked empty)", porcelain == "", repr(porcelain[:200]))
check("no stash", stash == "", repr(stash))
check("branch main", git("rev-parse", "--abbrev-ref", "HEAD").strip() == "main")

full_install = git("rev-parse", "3affe55").strip()
full_lock = git("rev-parse", "be74d23").strip()
full_ev = git("rev-parse", "af160ce").strip()
check("3affe55 full sha", full_install == INSTALL, full_install)
check("be74d23 full sha", full_lock == LOCK_RUN, full_lock)
check("af160ce == HEAD", full_ev == EXPECTED_HEAD)
parents_install = git("rev-list", "--parents", "-n", "1", INSTALL).split()
parents_ev = git("rev-list", "--parents", "-n", "1", EXPECTED_HEAD).split()
check("3affe55 has exactly one parent, be74d23", parents_install == [INSTALL, LOCK_RUN], parents_install)
check("af160ce has exactly one parent, 3affe55", parents_ev == [EXPECTED_HEAD, INSTALL], parents_ev)


def paths_of(commit):
    rows = [line.split("\t") for line in git("show", "--no-renames", "--name-status", "--format=", commit).splitlines() if line.strip()]
    return [(r[0], r[1]) for r in rows]


install_paths = paths_of(INSTALL)
ev_paths = paths_of(EXPECTED_HEAD)
print("3affe55 paths:", install_paths)
check("3affe55 touches exactly the two lock paths, both added", sorted(install_paths) == sorted([("A", p) for p in LOCK_PATHS]), install_paths)
print(f"af160ce: {len(ev_paths)} paths")
for status, path in ev_paths:
    print(f"   {status}\t{path}")
ev_ok_shape = all(p == "experiments/024-readout-routing-nounness/README.md" or p.startswith("experiments/024-readout-routing-nounness/evidence/lock-2026-09-25/")
                  or p.startswith("experiments/024-readout-routing-nounness/evidence/lock-review-2026-09-25/") for _, p in ev_paths)
check("af160ce paths are only README + evidence/lock-2026-09-25/ + evidence/lock-review-2026-09-25/", ev_ok_shape)
check("af160ce: README modified, everything else added", all((s == "M") == (p.endswith("024-readout-routing-nounness/README.md")) for s, p in ev_paths))

from neural_decompiler import readout_routing as rr  # noqa: E402

sci_ev = rr.scientific_changes([p for _, p in ev_paths])
sci_install = rr.scientific_changes([p for _, p in install_paths])
check("rr.scientific_changes(af160ce paths) == []", sci_ev == [], sci_ev)
check("rr.scientific_changes(3affe55 paths) == []", sci_install == [], sci_install)
diff = [line.strip() for line in git("diff", "--no-renames", "--name-only", LOCK_RUN, "HEAD").splitlines() if line.strip()]
sci_diff = rr.scientific_changes(diff)
print(f"git diff --no-renames --name-only be74d23 HEAD: {len(diff)} paths")
check("diff be74d23..HEAD == union of the two commits' paths", sorted(diff) == sorted({p for _, p in install_paths} | {p for _, p in ev_paths}))
check("rr.scientific_changes(diff be74d23..HEAD) == []", sci_diff == [], sci_diff)
anc = __import__("subprocess").run(["git", "merge-base", "--is-ancestor", LOCK_RUN, "HEAD"], cwd=guard.REPO).returncode
check("be74d23 is an ancestor of HEAD", anc == 0, anc)

# Own reading of the classifier + it is live on constructed paths.
print("SCIENTIFIC_PATH_PREFIXES:", rr.SCIENTIFIC_PATH_PREFIXES)
print("NON_SCIENTIFIC_PATHS:", rr.NON_SCIENTIFIC_PATHS)
print("NON_SCIENTIFIC_PREFIXES:", rr.NON_SCIENTIFIC_PREFIXES)


def own_classify(path):
    return path.startswith(rr.SCIENTIFIC_PATH_PREFIXES) and path not in rr.NON_SCIENTIFIC_PATHS and not path.startswith(rr.NON_SCIENTIFIC_PREFIXES)


probes = {"src/neural_decompiler/readout_routing.py": True, "experiments/024-readout-routing-nounness/run.py": True,
          "experiments/024-readout-routing-nounness/preregistration-lock.json": False, "experiments/024-readout-routing-nounness/README.md": False,
          "experiments/024-readout-routing-nounness/evidence/x/y.py": False, "experiments/023-block0-completion/exposed-cells.f64": True,
          "experiments/023-block0-completion/confirmation-v1.json": True, "experiments/022-upstream-error-localization/confirmation-v1.json": True,
          "experiments/020-readout-decompilation/closure.json": True, "docs/superpowers/specs/x.md": False}
for path, expected in probes.items():
    got = bool(rr.scientific_changes([path]))
    same = got == own_classify(path)
    if expected is None:
        check(f"classifier agrees with own reading on {path} (scientific={got})", same)
    else:
        check(f"classifier on {path} -> scientific={got} (expected {expected})", same and got == expected)
# the full classification of every changed path, printed
for path in diff:
    assert not own_classify(path)
check("own reading classifies every diff path as non-scientific", not any(own_classify(p) for p in diff))

state = json.load(open(os.path.join(guard.REPO, "outputs/experiment-024/results.json"), encoding="utf-8"))
lock = json.load(open(os.path.join(guard.REPO, LOCK_PATHS[0]), encoding="utf-8"))
check("lock.protocol_code_commit == be74d23 (full)", lock["protocol_code_commit"] == LOCK_RUN, lock["protocol_code_commit"])
check("state.phases.lock.commit == be74d23 (full)", state["phases"]["lock"].get("commit") == LOCK_RUN, state["phases"]["lock"].get("commit"))

s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"ITEM1 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
