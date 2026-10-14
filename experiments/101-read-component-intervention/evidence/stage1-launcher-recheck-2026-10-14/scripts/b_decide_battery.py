"""Part A: the launcher v2's pure decisions (``decide``) on realistic and adversarial Python 3.12 audit-event tuples.

ROOT is a mirror tree in this re-check's scratch directory (same relative layout as the repository); the decisions are
judged as the production process would make them, with the working directory at ROOT (``fake_cwd``), so this process
never changes into the repository. Real descriptors are opened on mirror directories for the ``dir_fd`` cases. Nothing
is written outside scratch (the guard refuses it). Output: out/b_decide_battery.json.
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAUNCHER_V2, LAUNCHER_V2_SHA256, RECHECK, REPO, extract_launcher, install_guard, write_json  # noqa: E402

assert hashlib.sha256(LAUNCHER_V2.read_bytes()).hexdigest() == LAUNCHER_V2_SHA256, "the launcher under review changed"

M = RECHECK / "mirror"
if M.exists():
    shutil.rmtree(M)
ROOTM = (M / "repo")
OUTSIDE = (M / "outside")
for d in (ROOTM / "outputs/experiment-101", ROOTM / "outputs/experiment-020", ROOTM / "outputs/experiment-025",
          ROOTM / "experiments/101-read-component-intervention", ROOTM / "src/neural_decompiler", ROOTM / ".venv/lib", OUTSIDE):
    d.mkdir(parents=True)
for f in (ROOTM / "outputs/experiment-020/results.json", ROOTM / "outputs/experiment-025/results.json", ROOTM / "src/neural_decompiler/read_intervention.py",
          ROOTM / "experiments/101-read-component-intervention/README.md", ROOTM / "README.md", OUTSIDE / "target.txt", ROOTM / ".venv/lib/x.txt"):
    f.write_text("x\n")
os.symlink("../outside/target.txt", ROOTM / "link_out")  # a symlink inside the repo pointing outside
os.symlink("../repo/experiments/101-read-component-intervention", OUTSIDE / "link_in")  # outside, pointing into the repo
ROOTM = ROOTM.resolve()
OUTSIDE = OUTSIDE.resolve()

install_guard()  # from here on, this process may write only under the re-check's scratch directory

P = extract_launcher(ROOTM, "production", fake_cwd=str(ROOTM))
R = extract_launcher(ROOTM, "rehearsal", fake_cwd=str(ROOTM))
EXP = P["EXPERIMENT_DIR"]
S1 = P["STAGE1_DIR"]
OUT = P["OUTPUTS"]
F64, SJ, TJ = S1 / "states.f64", S1 / "states.json", S1 / "table.json"
tmp = str(OUT / ".results-3iv5jg38.json")
W = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC          # 16778753: write_bytes / write_text
MK = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC  # 16780034: tempfile.mkstemp
RD = os.O_CLOEXEC                                                  # 16777216: read
fd_exp = os.open(EXP, os.O_RDONLY)
fd_out = os.open(OUTSIDE, os.O_RDONLY)
fd_file = os.open(OUTSIDE / "target.txt", os.O_RDONLY)
HERE = OUTSIDE  # the launcher's own directory, outside the repository

# (name, event, args, expected) -- expected: "allow" or "refuse" under production rules
cases = [
    # --- the events a correct stage 1 raises (real argument types: pathlib objects where Python passes them) ---
    ("mkdir stage1/ (Path.mkdir, parents, exist_ok)", "os.mkdir", (S1, 511, -1), "allow"),
    ("mkdir outputs/experiment-101 (write_state_atomic)", "os.mkdir", (OUT, 511, -1), "allow"),
    ("write states.f64 (Path.write_bytes)", "open", (F64, "w", W), "allow"),
    ("write states.json (Path.write_text)", "open", (SJ, "w", W), "allow"),
    ("write table.json (Path.write_text)", "open", (TJ, "w", W), "allow"),
    ("tempfile.mkstemp event", "tempfile.mkstemp", (tmp,), "allow"),
    ("state temporary os.open (mkstemp, with O_NOFOLLOW)", "open", (tmp, None, MK), "allow"),
    ("fdopen of the temporary", "open", (3, "w", W), "allow"),
    ("os.replace(temporary, results.json)", "os.rename", (tmp, OUT / "results.json", -1, -1), "allow"),
    ("fsync_directory os.open(outputs dir)", "open", (OUT, None, RD), "allow"),
    ("cleanup os.unlink(temporary)", "os.remove", (tmp, -1), "allow"),
    ("read results.json (load_results_state)", "open", (OUT / "results.json", "r", RD), "allow"),
    ("read 020 results.json", "open", (ROOTM / "outputs/experiment-020/results.json", "r", RD), "allow"),
    ("read states.f64 (file_sha256)", "open", (F64, "r", RD), "allow"),
    ("read the lock (relative, cwd = ROOT)", "open", ("experiments/101-read-component-intervention/README.md", "r", RD), "allow"),
    ("sentinel O_CREAT|O_EXCL in the launcher dir", "open", (str(HERE / "stage1.sentinel"), None, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "allow"),
    ("record O_CREAT|O_EXCL in the launcher dir", "open", (str(HERE / "production_run_x.json"), None, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "allow"),
    ("production_runs.jsonl append", "open", (HERE / "production_runs.jsonl", "a", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC), "allow"),
    ("mkdir outside the repo (e.g. a cache)", "os.mkdir", (str(OUTSIDE / "cache"), 511, -1), "allow"),
    ("write inside .venv (excluded from the repository)", "open", (ROOTM / ".venv/lib/x.txt", "w", W), "allow"),
    ("listdir (unhandled read)", "os.listdir", (str(ROOTM),), "allow"),
    ("dir_fd >= 0: mkdir 'stage1' relative to the experiment dir", "os.mkdir", ("stage1", 511, fd_exp), "allow"),
    ("dir_fd >= 0: mkdir 'x' relative to an outside dir", "os.mkdir", ("x", 511, fd_out), "allow"),
    ("dir_fd >= 0: remove 'x' relative to an outside dir", "os.remove", ("x", fd_out), "allow"),
    ("relative write naming results.json from ROOT", "open", ("outputs/experiment-101/results.json", "w", W), "allow"),
    ("relative mkdir naming stage1/ from ROOT", "os.mkdir", ("experiments/101-read-component-intervention/stage1", 511, -1), "allow"),
    ("write of a .results-*.json temporary by open()", "open", (str(OUT / ".results-anything.json"), "w", W), "allow"),
    ("direct (non-atomic) write of results.json", "open", (OUT / "results.json", "a", os.O_WRONLY | os.O_APPEND), "allow"),
    # --- writes that must be refused ---
    ("write a repo file", "open", (ROOTM / "README.md", "w", W), "refuse"),
    ("write a source file", "open", (ROOTM / "src/neural_decompiler/read_intervention.py", "w", W), "refuse"),
    ("write a sibling in stage1/", "open", (S1 / "other.json", "w", W), "refuse"),
    ("write another 101 output (report.md)", "open", (OUT / "report.md", "w", W), "refuse"),
    ("write candidate-lock.json", "open", (OUT / "candidate-lock.json", "w", W), "refuse"),
    ("write 025's outputs", "open", (ROOTM / "outputs/experiment-025/results.json", "w", W), "refuse"),
    ("relative write at the root", "open", ("x", "w", W), "refuse"),
    ("relative os.open O_CREAT at the root", "open", ("x", None, os.O_WRONLY | os.O_CREAT), "refuse"),
    ("relative mkdir at the root", "os.mkdir", ("x", 511, -1), "refuse"),
    ("relative mkdir elsewhere in the experiment", "os.mkdir", ("experiments/101-read-component-intervention/other", 511, -1), "refuse"),
    ("relative remove", "os.remove", ("x", -1), "refuse"),
    ("relative rmdir", "os.rmdir", ("x", -1), "refuse"),
    ("relative chmod", "os.chmod", ("x", 0o644, -1), "refuse"),
    ("relative rename", "os.rename", ("a", "b", -1, -1), "refuse"),
    ("relative link", "os.link", ("a", "b", -1, -1), "refuse"),
    ("relative symlink", "os.symlink", ("/etc/hosts", "x", -1), "refuse"),
    ("relative shutil.rmtree", "shutil.rmtree", ("x", None), "refuse"),
    ("relative truncate", "os.truncate", ("x", 0), "refuse"),
    ("remove states.f64", "os.remove", (F64, -1), "refuse"),
    ("chmod states.f64", "os.chmod", (F64, 0o444, -1), "refuse"),
    ("truncate states.f64 by path", "os.truncate", (F64, 0), "refuse"),
    ("rmdir stage1/", "os.rmdir", (S1, -1), "refuse"),
    ("rmtree stage1/", "shutil.rmtree", (S1, None), "refuse"),
    ("mkdir inside stage1/", "os.mkdir", (S1 / "sub", 511, -1), "refuse"),
    ("remove results.json", "os.remove", (OUT / "results.json", -1), "refuse"),
    ("rename results.json away", "os.rename", (OUT / "results.json", tmp, -1, -1), "refuse"),
    ("rename the temporary into the experiment", "os.rename", (tmp, EXP / "x.json", -1, -1), "refuse"),
    ("rename a file from outside onto states.f64", "os.rename", (OUTSIDE / "target.txt", F64, -1, -1), "refuse"),
    ("hard-link a file from outside onto states.f64", "os.link", (OUTSIDE / "target.txt", F64, -1, -1), "refuse"),
    ("symlink onto states.f64", "os.symlink", (str(OUTSIDE / "target.txt"), F64, -1), "refuse"),
    ("rename a repo file to outside", "os.rename", (ROOTM / "README.md", OUTSIDE / "r", -1, -1), "refuse"),
    ("dir_fd >= 0: mkdir 'other' relative to the experiment dir", "os.mkdir", ("other", 511, fd_exp), "refuse"),
    ("dir_fd >= 0: remove 'README.md' relative to the experiment dir", "os.remove", ("README.md", fd_exp), "refuse"),
    ("unresolvable dir_fd (closed descriptor number)", "os.mkdir", ("x", 511, 10 ** 6), "refuse"),
    ("unresolvable src dir_fd with an outside dst", "os.rename", ("a", OUTSIDE / "b", 10 ** 6, -1), "refuse"),
    ("NUL byte in a write path", "open", ("a\0b", "w", W), "refuse"),
    ("NUL byte in a remove path", "os.remove", ("a\0b", -1), "refuse"),
    ("write through an outside symlink into the repo", "open", (OUTSIDE / "link_in" / "x.json", "w", W), "refuse"),
    ("mkdir with too few args (fail closed)", "os.mkdir", (), "refuse"),
    ("rename with one arg (fail closed)", "os.rename", ("a",), "refuse"),
    # --- reads ---
    ("read another experiment's outputs", "open", (ROOTM / "outputs/experiment-025/results.json", "r", RD), "refuse"),
    ("relative read of another experiment's outputs", "open", ("outputs/experiment-025/results.json", "r", RD), "refuse"),
]

# probes whose decision is reported (not asserted): the gaps a reviewer should know about
probes = [
    ("unhandled: os.utime on states.f64", "os.utime", (F64, None, None, -1)),
    ("unhandled: os.utime on a source file", "os.utime", (ROOTM / "src/neural_decompiler/read_intervention.py", None, None, -1)),
    ("unhandled: os.chflags on a source file", "os.chflags", (ROOTM / "src/neural_decompiler/read_intervention.py", 2)),
    ("unhandled: os.setxattr on a source file", "os.setxattr", (ROOTM / "src/neural_decompiler/read_intervention.py", "user.x", b"1", 0)),
    ("unhandled: os.chown on a source file", "os.chown", (ROOTM / "src/neural_decompiler/read_intervention.py", -1, -1, -1)),
    ("unhandled: shutil.copyfile event (its open of dst is judged separately)", "shutil.copyfile", (OUTSIDE / "target.txt", ROOTM / "x")),
    ("unhandled: subprocess.Popen (e.g. git)", "subprocess.Popen", ("git", ["git", "stash"], None, None)),
    ("symlink in the repo pointing outside: remove the link entry", "os.remove", (ROOTM / "link_out", -1)),
    ("symlink in the repo pointing outside: rename the link entry out", "os.rename", (ROOTM / "link_out", OUTSIDE / "moved", -1, -1)),
    ("os.open with dir_fd outside but relative name that names results.json from ROOT (event has no dir_fd)", "open",
     ("outputs/experiment-101/results.json", None, os.O_WRONLY | os.O_CREAT)),
    ("os.open with dir_fd outside and relative name 'x' (event has no dir_fd; judged against ROOT)", "open", ("x", None, os.O_WRONLY | os.O_CREAT)),
    ("dir_fd that is a regular file's descriptor", "os.mkdir", ("x", 511, fd_file)),
    ("read with a NUL byte", "open", ("a\0b", "r", RD)),
]

results, mismatches = [], []
for name, event, args, expected in cases:
    try:
        decision = P["decide"](event, args, True)
        outcome = "allow" if decision is None else "refuse"
        detail = decision
    except Exception as error:  # an exception inside the hook fails the operation: closed
        outcome, detail = "refuse", f"EXCEPTION {type(error).__name__}: {error}"
    rehearsal = None
    try:
        rd_dec = R["decide"](event, args, False)
        rehearsal = "allow" if rd_dec is None else "refuse"
    except Exception as error:
        rehearsal = f"refuse (EXCEPTION {type(error).__name__})"
    row = {"case": name, "event": event, "expected": expected, "production": outcome, "detail": str(detail), "rehearsal": rehearsal}
    results.append(row)
    if outcome != expected:
        mismatches.append(row)

probe_rows = []
for name, event, args in probes:
    try:
        decision = P["decide"](event, args, True)
        outcome = "allow" if decision is None else "refuse"
    except Exception as error:
        outcome, decision = "refuse (exception)", f"{type(error).__name__}: {error}"
    probe_rows.append({"probe": name, "event": event, "production": outcome, "detail": str(decision)})

# the rehearsal rules: every repository write refused, outside writes allowed
WRITE_OPEN_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND


def repo_write_target(event, args):
    """True when the event is a write whose target (as the launcher resolves it) is inside the repository."""
    if event == "open":
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return False
        mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
        if not ((isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & WRITE_OPEN_FLAGS)):
            return False
        return P["_in_repo"](P["_path"](args[0]))
    if event == "os.mkdir":
        return bool(args) and P["_in_repo"](P["_path"](args[0], args[2] if len(args) > 2 else None))
    if event in ("os.remove", "os.rmdir"):
        return bool(args) and P["_in_repo"](P["_path"](args[0], args[1] if len(args) > 1 else None))
    if event == "os.rename":
        return len(args) > 1 and any(P["_in_repo"](P["_path"](a, d)) for a, d in ((args[0], args[2] if len(args) > 2 else None), (args[1], args[3] if len(args) > 3 else None)))
    return False


rehearsal_repo_writes = [r["case"] for r, (name, event, args, expected) in zip(results, cases) if repo_write_target(event, args) and r["rehearsal"] == "allow"]
rehearsal_checked = [name for name, event, args, expected in cases if repo_write_target(event, args)]

# dir_fd at the edge: a huge integer
edge = {}
for label, value in (("dir_fd 10**20", 10 ** 20), ("dir_fd -2", -2), ("dir_fd True", True)):
    try:
        d = P["decide"]("os.mkdir", ("x", 511, value), True)
        edge[label] = "allow" if d is None else f"refuse {d}"
    except Exception as error:
        edge[label] = f"EXCEPTION {type(error).__name__}: {error}"

# the operator's self-test, recomputed here on the mirror
self_test = P["hook_self_test"]()
for fd in (fd_exp, fd_out, fd_file):
    os.close(fd)
out = {"launcher_sha256": LAUNCHER_V2_SHA256, "mirror_root": str(ROOTM), "n_cases": len(results), "mismatches": mismatches, "cases": results,
       "probes": probe_rows, "edge": edge, "rehearsal_repo_write_cases_checked": rehearsal_checked, "rehearsal_allows_repo_writes": rehearsal_repo_writes,
       "hook_self_test_on_mirror": self_test}
write_json("b_decide_battery.json", out)
print(f"cases {len(results)}, mismatches {len(mismatches)}")
for row in mismatches:
    print("MISMATCH", row)
print(f"rehearsal rules: {len(rehearsal_checked)} repo-write cases checked; allowed in rehearsal: {rehearsal_repo_writes}")
for row in probe_rows:
    print("PROBE", row["production"], "|", row["probe"], "|", row["detail"][:120])
print("EDGE", edge)
print("self-test on mirror ok:", self_test["ok"])
