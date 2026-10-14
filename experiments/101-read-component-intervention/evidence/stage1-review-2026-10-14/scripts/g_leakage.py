"""Item 6: no S2 prompt ran and no target result exists anywhere (reads only; no model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402  (first)

guard.self_test()
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402

import torch  # noqa: E402

ROOT = guard.ROOT
EXP = ROOT / "experiments/101-read-component-intervention"
SCRATCH = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101")
out: dict = {"checks": {}, "facts": {}}


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"), flush=True)


listing = sorted(p.name for p in (ROOT / "outputs/experiment-101").iterdir())
out["facts"]["outputs_101"] = listing
check("outputs_101_holds_no_stage2_or_report", listing == ["calibration-measurements.pt", "candidate-calibration-v1.json", "candidate-lock.json",
                                                           "candidate-preregistration.md", "results.json"], listing)
state = json.loads((ROOT / "outputs/experiment-101/results.json").read_text())
check("state_has_no_confirmation_block", state.get("confirmation") is None and state["phases"]["confirm"] == {"status": "not_started"})
conf = json.loads((EXP / "confirmation-v1.json").read_text())
s2 = set(conf["manifest"]["S2-TARGET"])
s2_untagged = {k.rsplit("|", 1)[0] for k in s2} - {k.rsplit("|", 1)[0] for k in conf["manifest"]["S1-REF"]}  # the two/three/many prompts
check("ledger_disjoint_from_s2_and_target_prompts", not (set(state["executed_prompt_keys"]) & s2)
      and not ({k.rsplit("|", 1)[0] for k in state["executed_prompt_keys"]} & s2_untagged))
out["facts"]["target_prompt_keys"] = len(s2_untagged)

# Every file (repository without .git/.venv/.worktrees, and the exp101 scratchpad) that names a 101 S2 key or a 101
# target prompt key; only the confirmation manifest may.
pattern = re.compile(rb"coordinated-adjective-101-\d\d\|(?:two\|767|three\|1264|many\|1142|ref\|581\|REF)")
hits = {}
skipped: list[str] = []
roots = [ROOT, SCRATCH]
for base in roots:
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", ".worktrees", "node_modules", "__pycache__")]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):  # FIFOs, sockets, devices, symlinks: never opened (a FIFO would block)
                    skipped.append(str(path))
                    continue
                if info.st_size > 200_000_000:
                    skipped.append(str(path))
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            found = pattern.findall(data)
            if found:
                hits[str(path)] = len(found)
out["facts"]["files_naming_s2_or_target_prompt_keys"] = hits
out["facts"]["skipped_non_regular_or_huge"] = skipped
# Classification of every hit: key strings (the manifest, the freeze review's derivations, tier-A key-format asserts, a
# git clone of the repository) or the concurrent confirm-launcher review's stub scenarios (StubRunner, torch.zeros(4)).
CLR = SCRATCH / "confirm-launcher-review"
classes = {}
for path in hits:
    if path == str(EXP / "confirmation-v1.json"):
        classes[path] = "the S2 manifest (by design)"
    elif path.endswith(("independent_derivation.json", "cross_check_ri.json")) and ("freeze-review" in path):
        classes[path] = "freeze review: key derivation (key strings only)"
    elif path.endswith("tests/test_read_intervention.py"):
        classes[path] = "tier-A key-format assertions (no model)"
    elif path.startswith(str(CLR / "clone") + "/"):
        classes[path] = "concurrent review: a git clone of the repository (same files)"
    elif path.startswith(str(CLR / "main_block") + "/"):
        classes[path] = "concurrent review: StubRunner scenario (stub state / zero tensors)"
    elif path.startswith(str(guard.REVIEW)):
        classes[path] = "this review"
    else:
        classes[path] = "UNEXPECTED"
out["facts"]["hit_classes"] = classes
check("every_hit_is_a_key_string_or_a_stub", "UNEXPECTED" not in classes.values(), {p: c for p, c in classes.items() if c == "UNEXPECTED"})
stub_tensors = {}
for path in hits:
    if classes[path].startswith("concurrent review: StubRunner") and path.endswith(".pt"):
        t = torch.load(path, map_location="cpu", weights_only=True)
        stub_tensors[path] = {k: [list(v.shape), float(v.abs().sum())] for k, v in t.items()}
out["facts"]["stub_partial_tensors"] = stub_tensors
check("stub_partials_are_zero_vectors", all(shape == [4] and total == 0.0 for d in stub_tensors.values() for shape, total in d.values()) and len(stub_tensors) == 3,
      stub_tensors)
stub_labels = {}
for path in hits:
    if classes[path].startswith("concurrent review: StubRunner") and path.endswith("results.json"):
        found = re.findall(rb'"label":"([A-Z_]+)"', Path(path).read_bytes())
        if found:
            stub_labels[path] = sorted({f.decode() for f in found})
out["facts"]["stub_results_with_fabricated_labels"] = stub_labels
# The files skipped for size: chunked scan (they predate 101's frames).
for path in [p for p in skipped if Path(p).is_file() and not Path(p).is_symlink()]:
    count, tail = 0, b""
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            buf = tail + chunk
            count += len(pattern.findall(buf))
            tail = buf[-80:]
    out["facts"].setdefault("chunk_scanned", {})[path] = count
check("chunk_scanned_large_files_have_no_101_keys", all(v == 0 for v in out["facts"].get("chunk_scanned", {}).values()), out["facts"].get("chunk_scanned"))

# The calibration measurements hold exposed frames only.
measurements = torch.load(ROOT / "outputs/experiment-101/calibration-measurements.pt", map_location="cpu", weights_only=True)
names = list(measurements)
check("calibration_measurements_hold_no_101_frame", not any("-101-" in n for n in names), [n for n in names if "-101-" in n][:3])
out["facts"]["calibration_measurement_tensors"] = len(names)
# Git: no stage-2 or report file was ever committed; outputs are ignored.
log = subprocess.run(["git", "--no-optional-locks", "log", "--all", "--name-only", "--format=%H"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
committed_stage2 = sorted({line for line in log.splitlines() if "stage2" in line or line.startswith("outputs/experiment-101")})
check("no_stage2_or_101_outputs_in_git_history", not committed_stage2, committed_stage2)
ignored = subprocess.run(["git", "--no-optional-locks", "check-ignore", "-q", "outputs/experiment-101/stage2-measurements.pt"], cwd=ROOT).returncode == 0
out["facts"]["stage2_measurements_path_ignored"] = ignored
# (the process check is done from the shell: the guard refuses non-git children)
out["guard"] = guard.summary()
out["all_ok"] = all(c["ok"] for c in out["checks"].values())
(guard.OUT / "g_leakage.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["facts"], indent=1))
print("ALL OK" if out["all_ok"] else "SOME CHECKS FAILED")
