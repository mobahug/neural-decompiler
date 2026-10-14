"""An isolated test of the prepared stage-1 launcher's audit-hook decisions (version 2): the hook's functions are
extracted from the launcher's source with `ast` and executed in a bare namespace, so nothing of the launcher runs (no
audit hook installed, no import of neural_decompiler, no model). `hook_self_test()` decides the recorded Python 3.12
event tuples of the stage-1 writes under the production rules, with refused controls."""
import ast
import fcntl
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "launch_stage1.py").read_text()
tree = ast.parse(src)
names = {"_path", "_inside", "_in_repo", "_temp_state", "write_allowed", "decide", "hook_self_test"}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "UNRESOLVED" for t in n.targets)]
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
ns = {"os": os, "fcntl": fcntl, "Path": Path, "ROOT": ROOT, "OUTPUTS": ROOT / "outputs/experiment-101",
      "EXPERIMENT_DIR": ROOT / "experiments/101-read-component-intervention", "MODE": "production",
      "READABLE_OUTPUTS": (ROOT / "outputs/experiment-020/results.json", ROOT / "outputs/experiment-101")}
ns["STAGE1_DIR"] = ns["EXPERIMENT_DIR"] / "stage1"
ns["STAGE1_FILES"] = {ns["STAGE1_DIR"] / n for n in ("states.f64", "states.json", "table.json")}
os.chdir(ROOT)  # the launcher's working directory
exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch_stage1_extract", "exec"), ns)
result = ns["hook_self_test"]()
print(json.dumps(result, indent=1))
sys.exit(0 if result["ok"] else 1)
