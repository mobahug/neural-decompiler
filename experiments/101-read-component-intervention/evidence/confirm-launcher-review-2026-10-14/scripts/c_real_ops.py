"""Check 2, real operations: the confirm launcher's real ``_hook`` (extracted with ast, ROOT = a fresh scratch MIRROR,
the process working directory = the mirror root as in production) installed as an audit hook, then confirm's own
write and read functions driven on the mirror's paths exactly as the runner calls them:
- ``rr.write_state_atomic`` (twice, the first creating ``outputs/experiment-101``; then with ``os.replace`` failing once,
  which exercises its cleanup ``os.unlink`` of the temporary);
- ``rr.save_durably`` of the measurements and of the partial file (``torch.save`` through a handle), ``torch.load``,
  ``rd.load_results_state``, ``rc.file_sha256``, ``Path.read_bytes`` of a stage-1 file, ``rr.fsync_directory``;
then refused controls. Every audit event is recorded with the hook's decision. Run with argv[1] = production|rehearsal.
Output: out/c_real_ops_<mode>.json.
"""
import os
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REVIEW, check_launcher, extract_launcher, install_guard, write_json  # noqa: E402

check_launcher()
mode = sys.argv[1]
recorded: list = []
install_guard(recorded)
mirror = REVIEW / f"mirror_c_{mode}"
if mirror.exists():
    shutil.rmtree(mirror)
for rel in ("outputs/experiment-020", "outputs/experiment-025", "experiments/101-read-component-intervention/stage1", "src/neural_decompiler"):
    (mirror / rel).mkdir(parents=True)
for rel in ("outputs/experiment-020/results.json", "outputs/experiment-025/results.json", "experiments/101-read-component-intervention/stage1/states.f64",
            "experiments/101-read-component-intervention/stage1/table.json", "src/neural_decompiler/read_intervention.py"):
    (mirror / rel).write_text("{}\n")
os.chdir(mirror)

import torch  # noqa: E402

from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

ns = extract_launcher(mirror, mode)  # the real working directory (= the mirror root) resolves relative names
OUTPUTS, EXP = ns["OUTPUTS"], ns["EXPERIMENT_DIR"]
decisions: list = []
real_hook = ns["_hook"]
depth = {"n": 0}


def recording_hook(event, args):
    if depth["n"]:
        return real_hook(event, args)
    depth["n"] += 1
    try:
        try:
            decision = ns["decide"](event, args, mode == "production")
        except Exception as error:
            decision = ("exception", repr(error))
        if event in ("open", "os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.link", "os.symlink", "os.chmod", "os.truncate", "shutil.rmtree",
                     "tempfile.mkstemp", "os.utime"):
            shown = [os.fsdecode(a) if isinstance(a, (str, bytes, os.PathLike)) else a for a in args][:4]
            if any(isinstance(a, str) and str(mirror) in a for a in shown) or (event == "open" and isinstance(args[0], int) if args else False):
                decisions.append({"step": step["name"], "event": event, "args": shown, "decision": "allow" if decision is None else decision})
    finally:
        depth["n"] -= 1
    return real_hook(event, args)


step = {"name": "setup"}
sys.addaudithook(recording_hook)
results: dict = {}


def run(name, fn, expect_ok=True):
    step["name"] = name
    try:
        value = fn()
        results[name] = {"ok": True, "expected_ok": expect_ok, "value": value if isinstance(value, (str, int, float, bool, type(None))) else type(value).__name__}
    except BaseException as error:  # noqa: BLE001 - a refusal is the expected outcome of the controls
        results[name] = {"ok": False, "expected_ok": expect_ok, "error": f"{type(error).__name__}: {error}", "trace": traceback.format_exc(limit=3)[-600:]}


state = {"experiment": "101", "phases": {"confirm": {"status": "running"}}, "executed_prompt_keys": ["a|ref|581|REF"], "confirmation": {"barrier": {"passed": True}}}
run("write_state_atomic (creates outputs/experiment-101)", lambda: rr.write_state_atomic(OUTPUTS / "results.json", state))
run("write_state_atomic (again)", lambda: rr.write_state_atomic(OUTPUTS / "results.json", {**state, "n": 2}))
original_replace = os.replace


def failing_replace(*args, **kwargs):
    raise OSError("injected failure of os.replace")


def write_with_failure():
    os.replace = failing_replace
    try:
        return rr.write_state_atomic(OUTPUTS / "results.json", {**state, "n": 3})
    finally:
        os.replace = original_replace


run("write_state_atomic with os.replace failing (cleanup unlink)", write_with_failure, expect_ok=False)
run("no temporary left behind", lambda: sorted(p.name for p in OUTPUTS.iterdir()) if OUTPUTS.exists() else None)
tensors = {"coordinated-adjective-101-01/x3_pc": torch.zeros(55, 512), "coordinated-adjective-101-01/positions": torch.tensor([5, 6])}
run("save_durably measurements", lambda: rr.save_durably(tensors, OUTPUTS / "stage2-measurements.pt"))
run("torch.load measurements", lambda: sorted(torch.load(OUTPUTS / "stage2-measurements.pt")))
run("save_durably partial", lambda: rr.save_durably(tensors, OUTPUTS / "stage2-partial.pt"))
run("load_results_state", lambda: rd.load_results_state(OUTPUTS / "results.json")["state_sha256"])
run("file_sha256 of a stage-1 file", lambda: rc.file_sha256(EXP / "stage1/states.f64"))
run("read_bytes of a stage-1 file", lambda: len((EXP / "stage1/table.json").read_bytes()))
run("read 020's results", lambda: len((mirror / "outputs/experiment-020/results.json").read_text()))
run("fsync_directory of outputs", lambda: rr.fsync_directory(OUTPUTS))
# controls: each must be refused
run("CONTROL write a stage-1 file", lambda: (EXP / "stage1/table.json").write_text("y"), expect_ok=False)
run("CONTROL write another output", lambda: rr.save_durably(tensors, OUTPUTS / "other.pt"), expect_ok=False)
run("CONTROL write a source file", lambda: (mirror / "src/neural_decompiler/read_intervention.py").write_text("y"), expect_ok=False)
run("CONTROL remove the measurements", lambda: os.remove(OUTPUTS / "stage2-measurements.pt"), expect_ok=False)
run("CONTROL rename the measurements onto results.json", lambda: os.replace(OUTPUTS / "stage2-measurements.pt", OUTPUTS / "results.json"), expect_ok=False)
run("CONTROL mkdir under outputs", lambda: (OUTPUTS / "sub").mkdir(), expect_ok=False)
run("CONTROL read another experiment's outputs", lambda: (mirror / "outputs/experiment-025/results.json").read_text(), expect_ok=False)
run("CONTROL O_RDONLY|O_TRUNC of a stage-1 file", lambda: os.close(os.open(EXP / "stage1/table.json", os.O_RDONLY | os.O_TRUNC)), expect_ok=False)
run("CONTROL relative write at the root", lambda: Path("stray.json").write_text("y"), expect_ok=False)
run("PROBE torch.save to a path (C++ I/O, not judged)", lambda: torch.save(tensors, str(OUTPUTS / "probe-torch-save.pt")), expect_ok=False)
step["name"] = "end"
os.chdir(REVIEW)
summary = {"mode": mode, "mirror": str(mirror), "steps": results, "decisions": decisions,
           "unexpected": {k: v for k, v in results.items() if v["ok"] != v["expected_ok"] and not k.startswith("PROBE")},
           "refusals_recorded_by_the_hook": {k: ns["events"][k] for k in ("forbidden_writes", "forbidden_reads")},
           "outputs_after": sorted(p.name for p in OUTPUTS.iterdir()) if OUTPUTS.exists() else None}
write_json(f"c_real_ops_{mode}.json", summary)
for name, v in results.items():
    print(f"{'ok ' if v['ok'] else 'ERR'} expected_ok={v['expected_ok']!s:5} {name}: {v.get('value', v.get('error'))!s:.150}")
print("unexpected:", list(summary["unexpected"]))
print("outputs after:", summary["outputs_after"])
print("decided events in the mirror:", len(decisions), "refused:", sum(1 for d in decisions if d["decision"] != "allow"))
