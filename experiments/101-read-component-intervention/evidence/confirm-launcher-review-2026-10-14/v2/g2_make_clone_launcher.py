"""The v2 clone copy: the reviewed v2 source with exactly two textual substitutions — ROOT = the scratch clone, and the
one ``pre["runner"].confirm()`` call replaced by a refusal (inside v2's try/finally). Writes v2/clone_launcher/."""
import ast
import difflib
from pathlib import Path

R = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/confirm-launcher-review")
src = (R / "reviewed_v2/launch_confirm_v2.py").read_text(encoding="utf-8")
root_line = 'ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()'
call = 'status = pre["runner"].confirm()  # the one confirm'
assert src.count(root_line) == 1 and src.count(call) == 1 and src.count('"].confirm()') == 1
patched = src.replace(root_line, f'ROOT = Path("{R / "clone"}").resolve()').replace(call, 'raise SystemExit("review clone: confirm is never called (patched out)")')
calls = [n for n in ast.walk(ast.parse(patched)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "confirm"]
assert not calls and '"].confirm()' not in patched and "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler" not in patched
out = R / "v2/clone_launcher"
out.mkdir(parents=True, exist_ok=True)
(out / "launch_confirm.py").write_text(patched, encoding="utf-8")
diff = "".join(difflib.unified_diff(src.splitlines(True), patched.splitlines(True), "reviewed_v2/launch_confirm_v2.py", "v2/clone_launcher/launch_confirm.py"))
(out / "PATCH.txt").write_text(diff, encoding="utf-8")
print(diff)
