"""Part B: the launcher v2's real audit hook (``_hook``, extracted with ``ast``) installed in this process with ROOT =
the mirror tree in scratch, driving the REAL library functions stage 1 uses (``rr.write_state_atomic``,
``Path.mkdir(parents=True, exist_ok=True)``, the launcher's ``write_check``, ``ul.write_table``, ``Path.write_text``,
``rc.file_sha256``, ``rd.load_results_state``) and negative controls, with the working directory at the mirror root.
Usage: c_real_ops.py production|rehearsal. My own guard is installed first and refuses any write outside scratch.
Output: out/c_real_ops_<mode>.json.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAUNCHER_V2, LAUNCHER_V2_SHA256, RECHECK, extract_launcher, install_guard, write_json  # noqa: E402

MODE = sys.argv[1]
assert MODE in ("production", "rehearsal")
assert hashlib.sha256(LAUNCHER_V2.read_bytes()).hexdigest() == LAUNCHER_V2_SHA256

import numpy  # noqa: E402
import torch  # noqa: E402

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

M = RECHECK / f"mirror_{MODE}"
if M.exists():
    shutil.rmtree(M)
ROOTM, OUTSIDE = M / "repo", M / "outside"
for d in (ROOTM / "outputs/experiment-101", ROOTM / "outputs/experiment-020", ROOTM / "outputs/experiment-025", ROOTM / "experiments/101-read-component-intervention",
          ROOTM / "src/neural_decompiler", ROOTM / "src/outputs/experiment-101", OUTSIDE):
    d.mkdir(parents=True)
for f in (ROOTM / "outputs/experiment-020/results.json", ROOTM / "outputs/experiment-025/results.json", ROOTM / "README.md", OUTSIDE / "target.txt"):
    f.write_text("x\n")
os.symlink("../outside/target.txt", ROOTM / "link_out")
ROOTM, OUTSIDE = ROOTM.resolve(), OUTSIDE.resolve()

recorded: list = []
install_guard(recorded)
L = extract_launcher(ROOTM, MODE)
os.chdir(ROOTM)  # the launcher's working directory is the repository root
sys.addaudithook(L["_hook"])

OUT, S1 = L["OUTPUTS"], L["STAGE1_DIR"]
F64, SJ, TJ = S1 / "states.f64", S1 / "states.json", S1 / "table.json"
log: list = []


def step(name, fn, expect):
    """expect: 'ok' or an exception class name."""
    start = len(recorded)
    try:
        value = fn()
        outcome = "ok"
        detail = None if value is None else str(value)[:160]
    except BaseException as error:  # noqa: BLE001
        outcome = type(error).__name__
        detail = str(error)[:300]
    entry = {"step": name, "expect": expect, "outcome": outcome, "detail": detail, "as_expected": outcome == expect,
             "events": [{"event": r["event"], "args": r["args"], "types": r["arg_types"]} for r in recorded[start:] if r["event"] not in ("fcntl.fcntl", "os.listdir", "os.scandir")]}
    log.append(entry)
    print(("OK  " if entry["as_expected"] else "XX  ") + f"{name}: {outcome} (expected {expect})" + (f" — {detail}" if outcome != "ok" and detail else ""))
    return entry


state0 = {"schema_version": 1, "experiment": "101", "phases": {"stage1": {"status": "running"}}, "executed_prompt_keys": ["f|ref|581|S1"]}
blocks = [("a", torch.arange(6, dtype=torch.float64).reshape(2, 3)), ("b", torch.ones(4, dtype=torch.float64))]
production = MODE == "production"
ok_or_perm = "ok" if production else "PermissionError"

# --- the stage-1 write path, in the runner's order ---
step("ledger write: rr.write_state_atomic(results.json)", lambda: rr.write_state_atomic(OUT / "results.json", state0), ok_or_perm)
step("read back: rd.load_results_state", lambda: rd.load_results_state(OUT / "results.json")["executed_prompt_keys"], "ok" if production else "FileNotFoundError")
step("runner: stage1/ mkdir(parents=True, exist_ok=True)", lambda: S1.mkdir(parents=True, exist_ok=True), ok_or_perm)
step("launcher write_check on the three paths", lambda: L["write_check"]([F64, SJ, TJ]), "ok" if production else "PhaseError")
step("ul.write_table(states.f64, states.json)", lambda: ul.write_table(F64, SJ, blocks, {"meta": 1})["file_sha256"], ok_or_perm)
step("table.json write_text", lambda: TJ.write_text(pm.canonical_json({"x": 1}) + "\n", encoding="utf-8"), ok_or_perm)
for p in (F64, SJ, TJ):
    step(f"rc.file_sha256({p.name})", lambda p=p: rc.file_sha256(p), "ok" if production else "FileNotFoundError")
step("final state write: rr.write_state_atomic", lambda: rr.write_state_atomic(OUT / "results.json", {**state0, "phases": {"stage1": {"status": "complete"}}}), ok_or_perm)
real_replace = os.replace


def failing_replace(*args, **kwargs):
    raise OSError("simulated replace failure")


def atomic_with_failure():
    os.replace = failing_replace
    try:
        return rr.write_state_atomic(OUT / "results.json", state0)
    finally:
        os.replace = real_replace


step("cleanup path: replace fails, the temporary is unlinked", atomic_with_failure, "OSError" if production else "PermissionError")
step("temporaries left in outputs", lambda: sorted(p.name for p in OUT.iterdir() if p.name.startswith(".results-")) or None, "ok")
step("stage1/ mkdir again (exists)", lambda: S1.mkdir(parents=True, exist_ok=True), "ok" if production else "PermissionError")
step("write_check after the files exist", lambda: L["write_check"]([F64, SJ, TJ]), "PhaseError")

# --- negative controls (all inside the mirror) ---
step("write a repo file", lambda: open(ROOTM / "README.md", "w").close(), "PermissionError")
step("write a sibling in stage1/", lambda: (S1 / "other.json").write_text("x"), "PermissionError")
step("relative mkdir at the root", lambda: os.mkdir("relative_dir"), "PermissionError")
step("relative remove at the root", lambda: os.remove("README.md"), "PermissionError")
step("relative rename at the root", lambda: os.rename("README.md", "R.md"), "PermissionError")
step("relative chmod at the root", lambda: os.chmod("README.md", 0o600), "PermissionError")
step("remove states.f64", lambda: os.remove(F64), "PermissionError" if production else "PermissionError")
step("chmod states.f64", lambda: os.chmod(F64, 0o444), "PermissionError")
step("truncate states.f64", lambda: os.truncate(F64, 0), "PermissionError")
step("rename results.json away", lambda: os.rename(OUT / "results.json", OUT / "r2.json"), "PermissionError")
step("rmtree stage1/", lambda: shutil.rmtree(S1), "PermissionError")
step("numpy.save into the repo", lambda: numpy.save(ROOTM / "x.npy", numpy.zeros(2)), "PermissionError")
step("read another experiment's outputs", lambda: (ROOTM / "outputs/experiment-025/results.json").read_text(), "PermissionError")

# --- probes: writes the hook cannot see or does not judge (reported, not asserted) ---
probes = []


def probe(name, fn, target):
    before = os.path.lexists(target)
    entry = step("PROBE " + name, fn, "ok")
    entry["target_exists_before"] = before
    entry["target_exists_after"] = os.path.lexists(target)
    probes.append(entry)


probe("torch.save to a repo path", lambda: torch.save(torch.zeros(2), ROOTM / "x.pt"), ROOTM / "x.pt")


def sqlite_write():
    import sqlite3
    con = sqlite3.connect(ROOTM / "db.sqlite")
    con.execute("create table t (x)")
    con.commit()
    con.close()


probe("sqlite3 database in the repo", sqlite_write, ROOTM / "db.sqlite")
probe("os.mkfifo in the repo", lambda: os.mkfifo(ROOTM / "fifo"), ROOTM / "fifo")


def safetensors_write():
    from safetensors.torch import save_file
    save_file({"a": torch.zeros(2)}, str(ROOTM / "x.safetensors"))


probe("safetensors save_file in the repo", safetensors_write, ROOTM / "x.safetensors")
probe("os.utime on a repo file", lambda: os.utime(ROOTM / "README.md", (0, 0)), ROOTM / "README.md")
probe("unlink a repo symlink that points outside", lambda: (ROOTM / "link_out").unlink(), ROOTM / "link_out")


def dirfd_open():
    fd_src = os.open(ROOTM / "src", os.O_RDONLY)
    try:
        fd = os.open("outputs/experiment-101/results.json", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=fd_src)
        os.write(fd, b"{}")
        os.close(fd)
    finally:
        os.close(fd_src)


probe("os.open(dir_fd=src/) of a relative name that names results.json from ROOT", dirfd_open, ROOTM / "src/outputs/experiment-101/results.json")

os.chdir(RECHECK)
tree = sorted(str(p.relative_to(M)) + ("/" if p.is_dir() and not p.is_symlink() else "") for p in M.rglob("*"))
out = {"mode": MODE, "launcher_sha256": LAUNCHER_V2_SHA256, "mirror": str(M), "steps": log, "launcher_events": L["events"], "tree_after": tree,
       "unexpected": [e["step"] for e in log if not e["as_expected"] and not e["step"].startswith("PROBE")],
       "probes": [{"probe": p["step"], "outcome": p["outcome"], "detail": p["detail"], "created_or_changed": p["target_exists_after"], "events": p["events"]} for p in probes]}
write_json(f"c_real_ops_{MODE}.json", out)
print("launcher events:", json.dumps({k: v for k, v in L["events"].items() if v}, default=str)[:1500])
print("unexpected:", out["unexpected"])
