"""v1 (as reviewed) against v2: which top-level functions and module-level statements changed (ast.dump, docstrings
stripped). Read-only; prints a table."""
import ast
import json
from pathlib import Path

R = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/confirm-launcher-review")
v1 = ast.parse((R / "reviewed/launch_confirm.py").read_text())
v2 = ast.parse((R / "reviewed_v2/launch_confirm_v2.py").read_text())


def strip(node):
    if isinstance(node, ast.FunctionDef) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
        node = ast.FunctionDef(name=node.name, args=node.args, body=node.body[1:] or [ast.Pass()], decorator_list=node.decorator_list, returns=node.returns,
                               type_comment=None, type_params=[])
    return ast.dump(node, include_attributes=False)


f1 = {n.name: strip(n) for n in v1.body if isinstance(n, ast.FunctionDef)}
f2 = {n.name: strip(n) for n in v2.body if isinstance(n, ast.FunctionDef)}
rows = {name: ("added" if name not in f1 else "removed" if name not in f2 else "same" if f1[name] == f2[name] else "CHANGED") for name in sorted(set(f1) | set(f2))}
s1 = [ast.dump(n) for n in v1.body if not isinstance(n, ast.FunctionDef)]
s2 = [ast.dump(n) for n in v2.body if not isinstance(n, ast.FunctionDef)]
added_statements = [ast.unparse(n)[:160] for n in v2.body if not isinstance(n, ast.FunctionDef) and ast.dump(n) not in s1]
removed_statements = [ast.unparse(n)[:160] for n in v1.body if not isinstance(n, ast.FunctionDef) and ast.dump(n) not in s2]
out = {"functions": rows, "module_statements_added_or_changed_in_v2": added_statements, "module_statements_removed_or_changed_from_v1": removed_statements}
(R / "out/v2_a2_ast_diff.json").write_text(json.dumps(out, indent=1) + "\n")
for name, verdict in rows.items():
    print(f"{verdict:<8} {name}")
print("module-level statements added/changed in v2:")
for s in added_statements:
    print("   +", s.replace("\n", " ")[:160])
print("module-level statements removed/changed from v1:")
for s in removed_statements:
    print("   -", s.replace("\n", " ")[:160])
