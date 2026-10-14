"""Item 11 support (the stage-1 launcher is NOT run or imported): the audit events Python 3.12 raises for exactly the
file operations stage 1 performs (Path.mkdir(parents=True, exist_ok=True), Path.write_bytes, Path.write_text,
Path.read_bytes, rr.write_state_atomic's mkstemp/fdopen/replace/fsync_directory, its cleanup unlink), performed by me
on scratch paths in this review's directory, so that the launcher's hook can be read against the real argument shapes."""
from common import install_fs_guard

install_fs_guard()

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

from common import OUT, write_json  # noqa: E402

base = OUT / "audit_event_probe"
recorded: list = []
interesting = {"open", "os.mkdir", "os.rename", "os.remove", "os.rmdir", "tempfile.mkstemp", "os.chmod", "os.utime", "os.link", "os.symlink", "os.truncate"}


def recorder(event, args):
    if event in interesting:
        shown = []
        for a in args:
            if isinstance(a, (str, bytes, os.PathLike)):
                s = os.fsdecode(a)
                shown.append(s.replace(str(base), "<BASE>") if str(base) in s else ("<path outside probe>" if s.startswith("/") else s))
            else:
                shown.append(a)
        if any(isinstance(x, str) and "<BASE>" in x for x in shown) or (event == "open" and isinstance(args[0], int)):
            recorded.append({"event": event, "args": shown})


import shutil  # noqa: E402

if base.exists():
    shutil.rmtree(base)
base.mkdir(parents=True)
(base / "experiment").mkdir()
sys.addaudithook(recorder)
steps = {}


def step(name, fn):
    start = len(recorded)
    fn()
    steps[name] = recorded[start:]


stage1 = base / "experiment" / "stage1"
step("mkdir parents exist_ok (new)", lambda: stage1.mkdir(parents=True, exist_ok=True))
step("mkdir parents exist_ok (exists)", lambda: stage1.mkdir(parents=True, exist_ok=True))
step("write_bytes states.f64", lambda: (stage1 / "states.f64").write_bytes(b"\0" * 16))
step("write_text states.json", lambda: (stage1 / "states.json").write_text("{}\n", encoding="utf-8"))
step("write_text table.json", lambda: (stage1 / "table.json").write_text("{}\n", encoding="utf-8"))
step("read_bytes (file_sha256)", lambda: (stage1 / "states.f64").read_bytes())
outputs = base / "outputs"


def atomic():
    outputs.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".results-", suffix=".json", dir=outputs)
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write("{}\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, outputs / "results.json")
    descriptor = os.open(outputs, os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)


step("write_state_atomic pattern", atomic)


def cleanup():
    handle, temporary = tempfile.mkstemp(prefix=".results-", suffix=".json", dir=outputs)
    os.close(handle)
    os.unlink(temporary)


step("write_state_atomic cleanup unlink", cleanup)
out = {"python": sys.version, "steps": steps}
write_json("f_audit_events.json", out)
print(json.dumps(out, indent=1))
