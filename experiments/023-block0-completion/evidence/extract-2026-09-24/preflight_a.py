"""Experiment 023 pre-extract preflight, part A: strictly read-only (no model, no prompt, writes nothing in the repo)."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
EXPECTED_HEAD = "2111271892a8237f69c3ef18d6eec774e798ac5c"
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


# 1. HEAD / origin / tree / no 023 outputs
subprocess.run(["git", "fetch", "-q", "origin"], cwd=ROOT, check=True)
head, origin = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
check("HEAD is the approved commit", head == EXPECTED_HEAD, head)
check("origin/main == HEAD", origin == head, origin)
check("working tree clean (incl. untracked)", git("status", "--porcelain", "--untracked-files=all") == "", "")
check("no outputs/experiment-023/", not (ROOT / "outputs/experiment-023").exists())
check("no results state", not (ROOT / "outputs/experiment-023/results.json").exists())
check("no installed artifact / confirmation / record / lock", not any((ROOT / p).exists() for p in (
    b0c.CELLS_DATA_RELATIVE_PATH, b0c.CELLS_INDEX_RELATIVE_PATH, b0c.CONFIRMATION_RELATIVE_PATH, b0c.CALIBRATION_RELATIVE_PATH, b0c.LOCK_RELATIVE_PATH)))

# 2. stock validate (the unmodified runner's own phase)
validate = subprocess.run([sys.executable, str(ROOT / "experiments/023-block0-completion/run.py"), "validate"], cwd=ROOT, capture_output=True, text=True,
                          env={"HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
print("validate stdout:", validate.stdout.strip())
if validate.stderr.strip():
    print("validate stderr:", validate.stderr.strip()[-2000:])
check("stock validate exit 0", validate.returncode == 0, f"exit {validate.returncode}")

# 3. literal pins: computed blob == git's blob at HEAD == git hash-object of the working file == the literal pin
computed = b0c.module_blobs()
for name, pinned in b0c.FROZEN_BLOBS.items():
    at_head = git("rev-parse", f"HEAD:src/neural_decompiler/{name}")
    on_disk = git("hash-object", f"src/neural_decompiler/{name}")
    check(f"pin {name}", computed[name] == at_head == on_disk == pinned, pinned)
check("assert_frozen_blobs", b0c.assert_frozen_blobs() == b0c.FROZEN_BLOBS, f"{len(b0c.FROZEN_BLOBS)} modules")
check("block0_completion.py blob at HEAD", git("rev-parse", "HEAD:src/neural_decompiler/block0_completion.py") == b0c.own_blob(), b0c.own_blob())

# 4. Experiment 020 closure and Experiment 022 committed inputs
inputs = ul.load_frozen_inputs(ROOT)
rc.verify_020_closure(ROOT, inputs.confirmation_020)
check("Experiment 020 closure verifies", True)
digests_022 = b0c.verify_022_inputs(ROOT)
for kind, relative in (("calibration", ul.CALIBRATION_RELATIVE_PATH), ("confirmation", ul.CONFIRMATION_RELATIVE_PATH)):
    payload = json.loads((ROOT / relative).read_text())
    check(f"022 {kind} file sha256", rc.file_sha256(ROOT / relative) == b0c.INHERITED_022[f"{kind}_file_sha256"], rc.file_sha256(ROOT / relative))
    check(f"022 {kind} content sha256", payload["content_sha256"] == rc.content_digest(payload) == b0c.INHERITED_022[f"{kind}_content_sha256"], payload["content_sha256"])
    check(f"022 {kind} tracked", subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True).returncode == 0)

# 5. the local 022 table: present, the exact bound file
table = ROOT / b0c.TABLE_022_RELATIVE_PATH
size = table.stat().st_size if table.exists() else None
digest = hashlib.sha256(table.read_bytes()).hexdigest() if table.exists() else None
check("022 calibration-table.pt present", table.exists(), f"{size} bytes")
check("022 table file sha256 == the bound one", digest == b0c.INHERITED_022["table_file_sha256"] == "04659d5e8b20a70526e5862eb2b2de952b33496f25e3875de6c6254df2b8fd81", str(digest))
record = json.loads((ROOT / ul.CALIBRATION_RELATIVE_PATH).read_text())
print("   022 record binds", len(record["rematerialization"]["table_sha256"]), "table tensor digests (E1 checks them at extract)")

# 6. runtime and dependency versions against 020's explore (the runner's own check), and the exposed pool
runner = run023.Runner()
runtime = runner._check_runtime(inputs.closure)
check("runtime and versions equal 020's explore", True, json.dumps(runtime))
units = b0c.exposed_units(inputs)
check("exposed pool 175 cues × 108 frames = 18,900 pairs", (len(units.cues), len(units.frames), units.n_pairs) == (175, 108, 18_900))
check("scorable nouns 79", len(runner._noun_keys(inputs)) == 79, str(len(runner._noun_keys(inputs))))

print("\nPREFLIGHT A:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
