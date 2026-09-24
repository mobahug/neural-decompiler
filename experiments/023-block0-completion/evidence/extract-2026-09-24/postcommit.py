"""Experiment 023 post-install checks at the artifact commit: read-only. The 022 table is made unavailable to this
process (an audit hook refusing any open of it, and torch.load refusing it) before anything is imported."""
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
TABLE = (ROOT / "outputs/experiment-022/calibration-table.pt").resolve()
TABLE_INODE = os.stat(TABLE).st_ino
blocked = []


def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            path = Path(os.fsdecode(args[0])).resolve()
            same = path == TABLE or (path.exists() and os.stat(path).st_ino == TABLE_INODE)
        except OSError:
            same = False
        if same:
            blocked.append(str(path))
            raise PermissionError(f"the Experiment 022 table is unavailable to this process: {path}")


sys.addaudithook(hook)

import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

import torch  # noqa: E402

_load = torch.load


def guarded_load(f, *args, **kwargs):
    if isinstance(f, (str, os.PathLike)) and Path(f).resolve() == TABLE:
        blocked.append("torch.load")
        raise PermissionError("torch.load of the Experiment 022 table refused")
    return _load(f, *args, **kwargs)


torch.load = guarded_load
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


try:  # prove the guard is live
    open(TABLE, "rb")
    check("guard live (deliberate open refused)", False)
except PermissionError:
    check("guard live (deliberate open refused)", True)
probes = len(blocked)

spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


head = git("rev-parse", "HEAD")
data, index = ROOT / b0c.CELLS_DATA_RELATIVE_PATH, ROOT / b0c.CELLS_INDEX_RELATIVE_PATH
check("installed f64 sha256", hashlib.sha256(data.read_bytes()).hexdigest() == "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4")
check("installed json file sha256", hashlib.sha256(index.read_bytes()).hexdigest() == "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe")
loaded_index = json.loads(index.read_text())
check("installed json content_sha256", loaded_index["content_sha256"] == rc.content_digest(loaded_index) == "d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16")
for relative, path in ((b0c.CELLS_DATA_RELATIVE_PATH, data), (b0c.CELLS_INDEX_RELATIVE_PATH, index)):
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True, capture_output=True).stdout
    check(f"git show HEAD:{relative} == installed bytes == candidate bytes", committed == path.read_bytes() == (ROOT / "outputs/experiment-023" / ("candidate-" + path.name)).read_bytes())
    check(f"{relative} tracked", subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True).returncode == 0)

# The stock runner, in this guarded process: validate, then the installed-artifact reader calibrate uses.
runner = run023.Runner(log=lambda message: print("   runner:", message))
check("stock validate (in-process, table blocked)", runner.validate() == 0)
state = rd.load_results_state(runner.results_path)
inputs = runner._inputs()
cells, units, files = runner._installed_cells(state, inputs)
check("installed-artifact reader accepts the committed pair (== this run's extracted candidate, extraction commit == extract phase)",
      tuple(cells.shape) == (18_900, 8), json.dumps(files))
changed = runner.changed_paths(state["phases"]["extract"]["commit"])
check("calibrate precondition: no scientific change since extract", changed is not None and b0c.scientific_changes(changed) == [], json.dumps(changed))
# Capability only (not the calibrate phase; no envelope): the calibration kernel and its loop check on the first 16 draws.
indices = b0c.draw_indices(units, 16)
kernel = b0c.calibration_kernel(cells, units, indices, 16)
loop = b0c.kernel_loop_check(cells, units, indices, kernel, n_draws=16)
check("calibration kernel on the committed artifact (16 draws × 4 conditions; E6 enforced)", kernel["e6_max"] <= 1e-10 and loop["passed"],
      f"E6 max {kernel['e6_max']:.2e}; loop check {loop['max_difference']:.2e} over {loop['n_checked']}")
check("no access to the 022 table beyond the deliberate probe", len(blocked) == probes, f"{len(blocked) - probes} blocked attempts")
check("HEAD, tree", head == git("rev-parse", "HEAD") and git("status", "--porcelain", "--untracked-files=all") == "", head)
check("results.json unchanged", rc.file_sha256(runner.results_path) == "9bb1e33594670343a41e5c08f81cca3a78f50d1f1c4e04882708879e508a5ea7")
print("\nPOST-COMMIT:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
