"""Run the stock Experiment 023 runner's `confirm` phase once, unmodified, with prompt accounting: every call of
plural_mechanism.capture_prompt (the one forward-pass entry point) is appended to a JSONL log with its prompt key, and
the calls of run_capture (one forward each), run_patched and run_interventions are counted. An audit hook refuses any
open of Experiment 022's calibration table, which confirm never needs."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
LOG, SUMMARY = Path(sys.argv[1]), Path(sys.argv[2])
TABLE = (ROOT / "outputs/experiment-022/calibration-table.pt").resolve()
TABLE_INODE = os.stat(TABLE).st_ino
refused = []


def hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    raw = os.fsdecode(args[0])
    try:
        path = Path(raw).resolve()
        inode = os.stat(path).st_ino if path.exists() else None
    except OSError:
        path, inode = Path(raw), None
    if path == TABLE or inode == TABLE_INODE or os.path.basename(raw) == "calibration-table.pt":
        refused.append(str(path))
        raise PermissionError(f"refused during confirm: {path}")


sys.addaudithook(hook)
from neural_decompiler import plural_mechanism as pm  # noqa: E402

counts = {"capture_prompt": 0, "run_capture": 0, "run_patched": 0, "run_interventions": 0}
log = LOG.open("a", encoding="utf-8")
original = {name: getattr(pm, name) for name in counts if hasattr(pm, name)}


def counted(name):
    def wrapper(*args, **kwargs):
        counts[name] += 1
        return original[name](*args, **kwargs)
    return wrapper


def logged_capture_prompt(model, prompt, sites):
    counts["capture_prompt"] += 1
    log.write(json.dumps({"n": counts["capture_prompt"], "key": prompt.key, "t": time.time()}) + "\n")
    log.flush()
    return original["capture_prompt"](model, prompt, sites)


pm.capture_prompt = logged_capture_prompt
for name in ("run_capture", "run_patched", "run_interventions"):
    if name in original:
        setattr(pm, name, counted(name))
try:
    open(TABLE, "rb")
    raise SystemExit("guard not live")
except PermissionError:
    refused.clear()
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
code, error = None, None
try:
    code = run023.main(["confirm"])
except BaseException as exc:  # recorded here too; the runner has already recorded any incident in its state
    error = f"{type(exc).__name__}: {exc}"
    raise
finally:
    log.close()
    SUMMARY.write_text(json.dumps({"exit": code, "error": error, "counts": counts, "refused_during_run": refused}, indent=1))
sys.exit(code)
