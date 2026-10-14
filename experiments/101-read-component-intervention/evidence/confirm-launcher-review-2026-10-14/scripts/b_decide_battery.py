"""Check 2, decision level: the confirm launcher's ``decide`` (extracted with ast from the reviewed copy) on a battery of
event tuples, judged against a scratch MIRROR tree (ROOT = mirror, the working directory taken as the mirror root),
plus the operator's own ``hook_self_test`` re-run under both the mirror and the real repository path (pure decisions;
``_path`` only resolves paths). Nothing is created anywhere except the mirror tree in this review's directory.
Output: out/b_decide_battery.json.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, REPO, REVIEW, check_launcher, extract_launcher, install_guard, write_json  # noqa: E402

check_launcher()
install_guard()
mirror = REVIEW / "mirror_b"
for rel in ("outputs/experiment-101", "outputs/experiment-020", "outputs/experiment-025", "experiments/101-read-component-intervention/stage1",
            "experiments/101-read-component-intervention/evidence", "src/neural_decompiler", ".venv/lib", "docs"):
    (mirror / rel).mkdir(parents=True, exist_ok=True)
for rel in ("outputs/experiment-101/results.json", "outputs/experiment-101/calibration-measurements.pt", "outputs/experiment-020/results.json",
            "outputs/experiment-020/level1-diagnostic-pass1.json", "outputs/experiment-025/results.json",
            "experiments/101-read-component-intervention/stage1/states.f64", "experiments/101-read-component-intervention/stage1/states.json",
            "experiments/101-read-component-intervention/stage1/table.json", "experiments/101-read-component-intervention/preregistration-lock.json",
            "experiments/101-read-component-intervention/run.py", "src/neural_decompiler/read_intervention.py", "pyproject.toml"):
    p = mirror / rel
    if not p.exists():
        p.write_text("x\n")
link = mirror / "outputs/experiment-101/stage2-measurements.pt"
if not link.is_symlink() and not link.exists():
    pass  # a symlink probe is created separately below

ns = extract_launcher(mirror, "production", fake_cwd=str(mirror))
decide = ns["decide"]
OUTPUTS, EXP = ns["OUTPUTS"], ns["EXPERIMENT_DIR"]
cloexec = getattr(os, "O_CLOEXEC", 0)
MKSTEMP = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | cloexec  # tempfile._bin_openflags on macOS
W = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | cloexec
tmpdir = Path(tempfile.mkdtemp(dir=REVIEW / "tmpdir"))
(tmpdir / "probe-source").write_text("x")
fd_tmp = os.open(tmpdir, os.O_RDONLY)
fd_outputs = os.open(OUTPUTS, os.O_RDONLY)
fd_src = os.open(mirror / "src", os.O_RDONLY)
s = str

allowed = {
    # the results state through write_state_atomic
    "mkdir outputs (Path.mkdir exist_ok)": ("os.mkdir", (s(OUTPUTS), 511, -1)),
    "mkdir outputs as a PosixPath": ("os.mkdir", (OUTPUTS, 511, -1)),
    "mkstemp temporary (real flags)": ("open", (s(OUTPUTS / ".results-abc123.json"), None, MKSTEMP)),
    "fdopen of the temporary (an int)": ("open", (5, "w", os.O_WRONLY | cloexec)),
    "replace temporary -> results.json": ("os.rename", (s(OUTPUTS / ".results-abc123.json"), OUTPUTS / "results.json", -1, -1)),
    "unlink the temporary on failure": ("os.remove", (s(OUTPUTS / ".results-abc123.json"), -1)),
    "fsync_directory open of outputs": ("open", (OUTPUTS, None, os.O_RDONLY)),
    # the measurements
    "save_durably measurements": ("open", (s(OUTPUTS / "stage2-measurements.pt"), "wb", W)),
    "save_durably partial": ("open", (s(OUTPUTS / "stage2-partial.pt"), "wb", W)),
    "torch.load measurements": ("open", (s(OUTPUTS / "stage2-measurements.pt"), "rb", cloexec)),
    "load_results_state": ("open", (s(OUTPUTS / "results.json"), "r", cloexec)),
    "a direct write of results.json (allowed by design)": ("open", (s(OUTPUTS / "results.json"), "w", W)),
    "results.json opened 'a' (allowed by design)": ("open", (s(OUTPUTS / "results.json"), "a", os.O_WRONLY | os.O_CREAT | os.O_APPEND | cloexec)),
    "a '..' spelling of results.json": ("open", (s(OUTPUTS / "../experiment-101/results.json"), "w", W)),
    # reads
    "020's results": ("open", (s(mirror / "outputs/experiment-020/results.json"), "r", cloexec)),
    "a stage-1 file read": ("open", (s(EXP / "stage1/states.f64"), "rb", cloexec)),
    "the lock read": ("open", (s(EXP / "preregistration-lock.json"), "r", cloexec)),
    "a relative read at the root": ("open", ("pyproject.toml", "r", cloexec)),
    "open_code style read": ("open", (s(mirror / "src/neural_decompiler/read_intervention.py"), "r", 0)),
    # outside the repository, and .venv
    "a write in TMPDIR": ("open", (s(tmpdir / "f"), "w", W)),
    "mkdtemp in TMPDIR": ("os.mkdir", (s(tmpdir / "sub"), 448, -1)),
    "filelock probe link": ("os.link", (s(tmpdir / "probe-source"), s(tmpdir / "probe-target"), -1, -1)),
    "filelock probe remove by dir_fd": ("os.remove", ("probe-source", fd_tmp)),
    "rmtree outside": ("shutil.rmtree", (s(tmpdir), None)),
    "a .venv write": ("open", (s(mirror / ".venv/lib/x.pyc"), "wb", W)),
    "/dev/null r+": ("open", ("/dev/null", "r+", os.O_RDWR | cloexec)),
    "an event with no arguments": ("open", ()),
}
refused = {
    "a stage-1 file written": ("open", (s(EXP / "stage1/table.json"), "w", W)),
    "states.f64 written": ("open", (s(EXP / "stage1/states.f64"), "wb", W)),
    "the lock written": ("open", (s(EXP / "preregistration-lock.json"), "w", W)),
    "the runner written": ("open", (s(EXP / "run.py"), "w", W)),
    "a source file": ("open", (s(mirror / "src/neural_decompiler/read_intervention.py"), "w", W)),
    "evidence written": ("open", (s(EXP / "evidence/x.json"), "w", W)),
    "report.md (not confirm's)": ("open", (s(OUTPUTS / "report.md"), "w", W)),
    "calibration measurements overwritten": ("open", (s(OUTPUTS / "calibration-measurements.pt"), "wb", W)),
    "another output file": ("open", (s(OUTPUTS / "other.pt"), "wb", W)),
    "a temporary-like name in another directory": ("open", (s(EXP / ".results-x.json"), None, MKSTEMP)),
    "a temporary-like name not .json": ("open", (s(OUTPUTS / ".results-x.pt"), None, MKSTEMP)),
    "O_RDONLY|O_TRUNC on a stage-1 file (N6 fix)": ("open", (s(EXP / "stage1/table.json"), None, os.O_RDONLY | os.O_TRUNC)),
    "rename temporary -> measurements": ("os.rename", (s(OUTPUTS / ".results-x.json"), s(OUTPUTS / "stage2-measurements.pt"), -1, -1)),
    "rename measurements -> results.json": ("os.rename", (s(OUTPUTS / "stage2-measurements.pt"), s(OUTPUTS / "results.json"), -1, -1)),
    "rename results.json away": ("os.rename", (s(OUTPUTS / "results.json"), s(tmpdir / "r.json"), -1, -1)),
    "rename onto a stage-1 file": ("os.rename", (s(OUTPUTS / ".results-x.json"), s(EXP / "stage1/table.json"), -1, -1)),
    "remove results.json": ("os.remove", (s(OUTPUTS / "results.json"), -1)),
    "remove the measurements": ("os.remove", (s(OUTPUTS / "stage2-measurements.pt"), -1)),
    "remove a stage-1 file": ("os.remove", (s(EXP / "stage1/states.f64"), -1)),
    "rmdir outputs": ("os.rmdir", (s(OUTPUTS), -1)),
    "rmtree stage1": ("shutil.rmtree", (s(EXP / "stage1"), None)),
    "mkdir under outputs": ("os.mkdir", (s(OUTPUTS / "sub"), 511, -1)),
    "mkdir stage1 (stage 1's, not confirm's)": ("os.mkdir", (s(EXP / "stage1"), 511, -1)),
    "mkdir another outputs directory": ("os.mkdir", (s(mirror / "outputs/experiment-102"), 511, -1)),
    "chmod results.json": ("os.chmod", (s(OUTPUTS / "results.json"), 0o644, -1)),
    "truncate results.json": ("os.truncate", (s(OUTPUTS / "results.json"), 0)),
    "symlink into outputs": ("os.symlink", (s(tmpdir / "f"), s(OUTPUTS / "stage2-partial.pt"), -1)),
    "hard link onto results.json": ("os.link", (s(tmpdir / "probe-source"), s(OUTPUTS / "results.json"), -1, -1)),
    "relative mkdir at the root": ("os.mkdir", ("x", 511, -1)),
    "relative remove at the root": ("os.remove", ("x", -1)),
    "relative rename at the root": ("os.rename", ("a", "b", -1, -1)),
    "relative write at the root": ("open", ("x.json", "w", W)),
    "relative remove under src by dir_fd": ("os.remove", ("read_intervention.py", fd_src)),
    "unresolvable dir_fd (huge)": ("os.remove", ("x", 10 ** 6)),
    "a closed descriptor number": ("os.remove", ("x", 987)),
    "missing arguments (os.mkdir)": ("os.mkdir", ()),
    "missing arguments (os.rename)": ("os.rename", ("a",)),
    "read another experiment's outputs": ("open", (s(mirror / "outputs/experiment-025/results.json"), "r", 0)),
    "read another 020 output": ("open", (s(mirror / "outputs/experiment-020/level1-diagnostic-pass1.json"), "r", 0)),
    "read the outputs directory itself": ("open", (s(mirror / "outputs"), None, os.O_RDONLY)),
}
probes = {  # known permissive gaps (re-check N6), reported, not asserted
    "relative write naming an allowed file, resolved from the working directory": ("open", ("outputs/experiment-101/results.json", "w", W)),
    "case variant of the repository root (APFS case-insensitive)": ("open", (s(mirror).replace("mirror_b", "MIRROR_B") + "/src/neural_decompiler/x.py", "w", W)),
    "an unhandled event (os.utime) on a stage-1 file": ("os.utime", (s(EXP / "stage1/table.json"), None, None, -1)),
    "torch.save to a path raises no open event": ("(none)", ()),
}


def verdict(event, args, production=True):
    try:
        d = decide(event, args, production)
        return "allow" if d is None else f"refuse {d[0]}"
    except Exception as error:
        return f"refuse (exception {type(error).__name__}: {error})"


rows = {"allowed": {}, "refused": {}, "rehearsal": {}, "probes": {}}
for name, (event, args) in allowed.items():
    rows["allowed"][name] = verdict(event, args)
for name, (event, args) in refused.items():
    rows["refused"][name] = verdict(event, args)
repo_writes = {name: spec for name, spec in {**allowed, **refused}.items() if spec[0] != "open" or (len(spec[1]) > 2 and isinstance(spec[1][2], int) and spec[1][2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)) or (len(spec[1]) > 1 and isinstance(spec[1][1], str) and any(c in spec[1][1] for c in "wax+"))}
for name, (event, args) in repo_writes.items():
    rows["rehearsal"][name] = verdict(event, args, production=False)
for name, (event, args) in probes.items():
    rows["probes"][name] = verdict(event, args) if event != "(none)" else "not judged: C++ file I/O (confirm never calls torch.save with a path)"
os.close(fd_tmp)
os.close(fd_outputs)
os.close(fd_src)

ok_allowed = all(v == "allow" for v in rows["allowed"].values())
ok_refused = all(v.startswith("refuse") for v in rows["refused"].values())
in_repo_rehearsal = {k: v for k, v in rows["rehearsal"].items() if not any(word in k for word in ("TMPDIR", "outside", ".venv", "/dev/null", "fdopen", "filelock", "unhandled", "no arguments"))}
ok_rehearsal = all(v.startswith("refuse") for v in in_repo_rehearsal.values())

self_test_real = extract_launcher(REPO, "production", fake_cwd=str(REPO))["hook_self_test"]()
self_test_mirror = ns["hook_self_test"]()
out = {"mirror": str(mirror), "ok_allowed": ok_allowed, "ok_refused": ok_refused, "ok_rehearsal_refuses_every_repo_write": ok_rehearsal,
       "n_allowed": len(rows["allowed"]), "n_refused": len(rows["refused"]), "n_rehearsal_repo_writes": len(in_repo_rehearsal),
       "operator_self_test_real_root": self_test_real, "operator_self_test_mirror_root": self_test_mirror, "rows": rows}
write_json("b_decide_battery.json", out)
print(f"allowed {sum(v == 'allow' for v in rows['allowed'].values())}/{len(rows['allowed'])}; refused {sum(v.startswith('refuse') for v in rows['refused'].values())}/{len(rows['refused'])}; "
      f"rehearsal refuses {sum(v.startswith('refuse') for v in in_repo_rehearsal.values())}/{len(in_repo_rehearsal)} repo writes")
for group in ("allowed", "refused"):
    for name, v in rows[group].items():
        expected = (v == "allow") if group == "allowed" else v.startswith("refuse")
        if not expected:
            print("UNEXPECTED", group, name, v)
for name, v in rows["probes"].items():
    print("probe:", name, "->", v)
print("self-test (real root) ok:", self_test_real["ok"], "| (mirror root) ok:", self_test_mirror["ok"])
