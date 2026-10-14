"""S2 support: the audit-event arguments of relative-path filesystem calls (Python 3.12), and fcntl F_GETPATH on -1.
Run from this review's scratch directory only (it changes into out/audit_event_probe/rel and cleans up after itself)."""
import fcntl
import os
import shutil
import sys
from pathlib import Path

here = Path(__file__).resolve().parent.parent / "out" / "audit_event_probe" / "rel"
here.mkdir(parents=True, exist_ok=True)
os.chdir(here)
seen = []


def hook(event, args):
    if event in ("os.mkdir", "os.chmod", "os.rmdir", "os.remove", "os.rename", "shutil.rmtree"):
        seen.append((event, tuple(x if not isinstance(x, (str, bytes)) else os.fsdecode(x) for x in args)))


sys.addaudithook(hook)
os.mkdir("d1"); os.chmod("d1", 0o755); open("f1", "w").close(); os.rename("f1", "f2"); os.remove("f2"); os.rmdir("d1")  # noqa: E702
os.mkdir("d2"); shutil.rmtree("d2")  # noqa: E702
for item in seen:
    print(item)
try:
    fcntl.fcntl(-1, fcntl.F_GETPATH, bytes(1024))
    print("F_GETPATH(-1): no error")
except (OSError, ValueError) as error:
    print(f"F_GETPATH(-1): {type(error).__name__}: {error}")
os.chdir(here.parent)
here.rmdir()
