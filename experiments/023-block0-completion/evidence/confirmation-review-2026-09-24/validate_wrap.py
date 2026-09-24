"""The stock read-only `validate` phase, run inside the review guard (022 table refused, no forward pass possible),
with the results state's digest and mtime checked before and after (validate must write nothing)."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirmreview023")
import guard  # noqa: E402  FIRST

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

print(guard.forbid_forward(), flush=True)  # validate needs no model; any forward would raise
ROOT = Path(guard.ROOT)
RESULTS = ROOT / "outputs/experiment-023/results.json"


def snap():
    return hashlib.sha256(RESULTS.read_bytes()).hexdigest(), os.stat(RESULTS).st_mtime_ns, sorted(p.name for p in RESULTS.parent.iterdir())


before = snap()
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
code = run023.main(["validate"])
after = snap()
print(f"validate exit {code}; results.json unchanged: {before == after} ({after[0][:16]}); refusals during validate: {guard.REFUSED}")
sys.exit(0 if code == 0 and before == after and not guard.REFUSED else 1)
