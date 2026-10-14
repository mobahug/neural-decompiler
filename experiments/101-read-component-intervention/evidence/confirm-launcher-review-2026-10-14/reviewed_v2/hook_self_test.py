"""An isolated test of the confirm launcher's audit-hook decisions: the hook's functions are extracted from the
launcher's source with `ast` and executed in a bare namespace (nothing of the launcher runs)."""
import ast
import fcntl
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
tree = ast.parse((HERE / "launch_confirm.py").read_text())
names = {"_path", "_inside", "_in_repo", "_temp_state", "write_allowed", "decide", "hook_self_test"}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "UNRESOLVED" for t in n.targets)]
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
OUTPUTS = ROOT / "outputs/experiment-101"
ns = {"os": os, "fcntl": fcntl, "Path": Path, "ROOT": ROOT, "OUTPUTS": OUTPUTS, "EXPERIMENT_DIR": ROOT / "experiments/101-read-component-intervention",
      "MODE": "production", "READABLE_OUTPUTS": (ROOT / "outputs/experiment-020/results.json", OUTPUTS),
      "WRITABLE": {OUTPUTS / "results.json", OUTPUTS / "stage2-measurements.pt", OUTPUTS / "stage2-partial.pt"}}
os.chdir(ROOT)
exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch_confirm_extract", "exec"), ns)
result = ns["hook_self_test"]()
print(json.dumps(result, indent=1))
sys.exit(0 if result["ok"] else 1)
