"""Compare the confirm launcher's top-level functions with the calibrate launcher's and the stage-1 launcher v2's, by
ast.dump (docstrings stripped). Read-only; prints a table."""
import ast
import json
import sys
from pathlib import Path

EXP = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101")
files = {"confirm": EXP / "confirm-launcher-review/reviewed/launch_confirm.py", "calibrate": EXP / "calibrate/launch_calibrate.py",
         "stage1v2": EXP / "stage1/launch_stage1.py"}


def functions(path):
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                node = ast.FunctionDef(name=node.name, args=node.args, body=body[1:] or [ast.Pass()], decorator_list=node.decorator_list, returns=node.returns,
                                       type_comment=None, type_params=[])
            out[node.name] = ast.dump(node, include_attributes=False)
    return out


fns = {name: functions(path) for name, path in files.items()}
rows = []
for name in fns["confirm"]:
    rows.append({"function": name, "same_as_calibrate": fns["calibrate"].get(name) == fns["confirm"][name] if name in fns["calibrate"] else None,
                 "same_as_stage1v2": fns["stage1v2"].get(name) == fns["confirm"][name] if name in fns["stage1v2"] else None})
print(json.dumps(rows, indent=1))
