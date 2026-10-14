"""Make the clone copy of the launcher: the reviewed source with exactly two textual substitutions —
(1) ROOT = the scratch clone (never the repository), and
(2) the one ``pre["runner"].confirm()`` call replaced by a refusal, so that even production mode cannot reach confirm.
Writes clone_launcher/launch_confirm.py and clone_launcher/PATCH.txt (the unified diff)."""
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAUNCHER, REVIEW, check_launcher  # noqa: E402

check_launcher()
src = LAUNCHER.read_text(encoding="utf-8")
root_line = 'ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()'
call = 'status = pre["runner"].confirm()  # the one confirm'
assert src.count(root_line) == 1 and src.count(call) == 1 and src.count('"].confirm()') == 1
patched = src.replace(root_line, f'ROOT = Path("{REVIEW / "clone"}").resolve()')
patched = patched.replace(call, 'raise SystemExit("review clone: confirm is never called (patched out)")  # the one confirm, patched out')
import ast
calls = [n for n in ast.walk(ast.parse(patched)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "confirm"]
assert not calls and '"].confirm()' not in patched and "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler" not in patched
out_dir = REVIEW / "clone_launcher"
out_dir.mkdir(exist_ok=True)
(out_dir / "launch_confirm.py").write_text(patched, encoding="utf-8")
diff = "".join(difflib.unified_diff(src.splitlines(True), patched.splitlines(True), "reviewed/launch_confirm.py", "clone_launcher/launch_confirm.py"))
(out_dir / "PATCH.txt").write_text(diff, encoding="utf-8")
print(diff)
