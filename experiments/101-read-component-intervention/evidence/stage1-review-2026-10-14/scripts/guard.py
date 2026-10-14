"""The review's guard (AI review of Experiment 101 stage 1, read-only). Imported FIRST by every review script.

1. An audit hook refuses every write-type event that resolves inside the repository (any spelling: case-insensitive,
   with or without /System/Volumes/Data), and every child process that is not a read-only git command. Writes outside
   the repository are allowed and recorded (this directory, the scratch TMPDIR, the HF cache's lock files).
2. ``install_forward_guards()`` (call after importing neural_decompiler, before any model load):
   - ``torch.nn.Module.__call__`` refuses EVERY module call (this review needs no forward at all); before refusing, it
     scans every argument for a forbidden 101 token sequence (each of the 60 candidate texts with each of the four cues,
     plus the 36 frozen frames' prefix + cue + suffix ids) and records a ``token_guard`` hit if one is found;
   - every forward entry point (pm.capture_prompt, pm.run_patched, rd.capture_frame_020, ri.capture_reference,
     ri.run_arm, capture.run_capture, interventions.run_interventions, ra.capture_reference) is replaced, by identity in
     every loaded neural_decompiler module, with a refusal.
3. ``seal(model)`` wraps every module's forward and the model classes' forward/run_with_hooks/run_with_cache/generate
   with refusals.
4. At exit the guard's record is written to out/<script>_guard.json: refusals, token-guard hits, repository write
   attempts, child processes, writes outside the repository.
"""
from __future__ import annotations

import atexit
import fcntl
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
REVIEW = Path(__file__).resolve().parent.parent
OUT = REVIEW / "out"
TMP = REVIEW / "tmp"
TMP.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(TMP)
os.environ["GIT_OPTIONAL_LOCKS"] = "0"
import tempfile  # noqa: E402

tempfile.tempdir = None

SCRIPT = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "interactive"
RECORD = {"script": SCRIPT, "python_flags_dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
          "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "PYTHONDONTWRITEBYTECODE", "GIT_OPTIONAL_LOCKS", "TMPDIR")},
          "repo_write_refusals": [], "subprocess": [], "subprocess_refusals": [], "writes_outside_repo": [], "module_call_refusals": 0,
          "module_call_refusal_examples": [], "entry_point_refusals": [], "token_guard_hits": [], "forbidden_sequences": 0, "sealed": None,
          "forward_guards_installed": False}
if not sys.flags.dont_write_bytecode:
    raise SystemExit("run with python -B")
if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
    raise SystemExit("set HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1")

_ROOT_KEY = str(ROOT).lower()
_DATA_PREFIX = "/system/volumes/data"
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_GIT_READ_ONLY = {"rev-parse", "status", "ls-files", "diff", "merge-base", "ls-tree", "log", "show", "cat-file", "check-ignore", "check-attr",
                  "rev-list", "branch", "describe", "for-each-ref"}


def _resolve(value, dir_fd=None):
    if isinstance(value, int) and not isinstance(value, bool):
        return None  # a descriptor: its open was judged
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    raw = os.fsdecode(value)
    if not os.path.isabs(raw):
        if isinstance(dir_fd, int) and not isinstance(dir_fd, bool) and dir_fd >= 0:
            try:
                base = fcntl.fcntl(dir_fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0].decode()
            except (OSError, ValueError):
                return "<unresolved>"
        else:
            base = os.getcwd()
        raw = os.path.join(base, raw)
    try:
        return os.path.realpath(raw)
    except (OSError, ValueError):
        return "<unresolved>"


def _in_repo(path: str) -> bool:
    if path == "<unresolved>":
        return True  # fail closed
    key = path.lower()
    if key.startswith(_DATA_PREFIX + "/"):
        key = key[len(_DATA_PREFIX):]
    return key == _ROOT_KEY or key.startswith(_ROOT_KEY + "/")


def _note_outside(event, path):
    if path and not path.startswith(str(REVIEW)) and len(RECORD["writes_outside_repo"]) < 200:
        RECORD["writes_outside_repo"].append(f"{event} {path}")


def _refuse(event, detail):
    RECORD["repo_write_refusals"].append(f"{event} {detail}")
    raise PermissionError(f"review guard: repository write refused: {event} {detail}")


_PATH_EVENTS = {  # event -> list of (path index, dir_fd index or None)
    "os.mkdir": [(0, 2)], "os.rmdir": [(0, 1)], "os.remove": [(0, 1)], "shutil.rmtree": [(0, 1)], "os.truncate": [(0, None)],
    "os.chmod": [(0, 2)], "os.chown": [(0, 3)], "os.utime": [(0, 3)], "os.chflags": [(0, None)], "os.setxattr": [(0, None)],
    "os.removexattr": [(0, None)], "os.rename": [(0, 2), (1, 3)], "os.link": [(0, 2), (1, 3)], "os.symlink": [(1, 2)],
    "shutil.copyfile": [(1, None)], "shutil.copymode": [(1, None)], "shutil.copystat": [(1, None)], "shutil.copytree": [(1, None)],
    "shutil.move": [(0, None), (1, None)], "tempfile.mkstemp": [(0, None)], "tempfile.mkdtemp": [(0, None)], "sqlite3.connect": [(0, None)],
    "os.mkfifo": [(0, 2)], "os.mknod": [(0, 3)],
}


def _hook(event, args):
    if event == "open":
        if not args:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        if not writing:
            return
        path = _resolve(args[0])
        if path is None:
            return
        if _in_repo(path):
            _refuse(event, f"{path} mode={mode!r} flags={flags!r}")
        _note_outside(event, path)
        return
    if event in _PATH_EVENTS:
        for index, fd_index in _PATH_EVENTS[event]:
            if len(args) <= index:
                continue
            dir_fd = args[fd_index] if fd_index is not None and len(args) > fd_index else None
            path = _resolve(args[index], dir_fd)
            if path is None:
                continue
            if _in_repo(path):
                _refuse(event, path)
            _note_outside(event, path)
        return
    if event == "subprocess.Popen":
        executable, argv = (args[0] if args else None), (args[1] if len(args) > 1 else None)
        argv_list = [os.fsdecode(a) for a in argv] if isinstance(argv, (list, tuple)) else [str(argv)]
        RECORD["subprocess"].append(argv_list)
        name = os.path.basename(argv_list[0]) if argv_list else ""
        sub = [a for a in argv_list[1:] if not a.startswith("-")]
        if name != "git" or not sub or sub[0] not in _GIT_READ_ONLY:
            RECORD["subprocess_refusals"].append(argv_list)
            raise PermissionError(f"review guard: child process refused: {argv_list}")
        return
    if event in ("os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "pty.spawn"):
        RECORD["subprocess_refusals"].append([event, repr(args)[:200]])
        raise PermissionError(f"review guard: {event} refused")


sys.addaudithook(_hook)


def _write_record():
    try:
        OUT.mkdir(exist_ok=True)
        (OUT / f"{SCRIPT}_guard.json").write_text(json.dumps(RECORD, indent=1, default=str) + "\n", encoding="utf-8")
    except Exception as error:  # never mask the script's own outcome
        print(f"guard record not written: {error}", file=sys.stderr)


atexit.register(_write_record)


# ---------------------------------------------------------------------------------------------------------------------
# Forward guards (installed after neural_decompiler is imported, before any model load).

FORBIDDEN: list[tuple[int, ...]] = []
FORBIDDEN_TEXTS: list[str] = []
CUE_WORDS = {"one": 581, "two": 767, "three": 1264, "many": 1142}


def build_forbidden(tokenizer) -> int:
    """Each of the 60 candidate texts with each of the four cues (tokenizer only, no forward), and the 36 frozen frames'
    prefix + cue + suffix ids from the committed confirmation file."""
    from neural_decompiler import read_intervention as ri

    seqs: set[tuple[int, ...]] = set()
    for text in ri.FRAME_CANDIDATES:
        for word in CUE_WORDS:
            filled = text.format(cue=word)
            FORBIDDEN_TEXTS.append(filled)
            seqs.add(tuple(int(t) for t in tokenizer.encode(filled)))
            seqs.add(tuple(int(t) for t in tokenizer.encode(" " + filled)))
    payload = json.loads((ROOT / ri.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))
    for frame in payload["frames"]:
        for cue_id in CUE_WORDS.values():
            seqs.add(tuple(int(t) for t in (*frame["prefix_ids"], cue_id, *frame["suffix_ids"])))
    FORBIDDEN[:] = sorted(seqs)
    RECORD["forbidden_sequences"] = len(FORBIDDEN)
    RECORD["forbidden_texts"] = len(FORBIDDEN_TEXTS)
    return len(FORBIDDEN)


def _contains(haystack: list[int], needle: tuple[int, ...]) -> bool:
    n = len(needle)
    return any(tuple(haystack[i:i + n]) == needle for i in range(len(haystack) - n + 1))


def _scan(value, depth=0) -> str | None:
    import torch

    if depth > 4:
        return None
    if isinstance(value, str):
        for text in FORBIDDEN_TEXTS:
            if text in value:
                return f"text {text!r}"
        return None
    if isinstance(value, torch.Tensor):
        if value.dtype in (torch.int64, torch.int32) and value.numel() <= 100_000:
            rows = value.reshape(-1, value.shape[-1]).tolist() if value.dim() >= 1 and value.numel() else []
            for row in rows:
                for seq in FORBIDDEN:
                    if _contains(row, seq):
                        return f"token ids {list(seq)}"
        return None
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            for seq in FORBIDDEN:
                if _contains(list(value), seq):
                    return f"token ids {list(seq)}"
            return None
        for item in value:
            hit = _scan(item, depth + 1)
            if hit:
                return hit
        return None
    if isinstance(value, dict):
        for item in value.values():
            hit = _scan(item, depth + 1)
            if hit:
                return hit
    return None


def install_forward_guards() -> None:
    import torch

    from neural_decompiler import capture, interventions
    from neural_decompiler import plural_mechanism as pm
    from neural_decompiler import read_assembly as ra
    from neural_decompiler import read_intervention as ri
    from neural_decompiler import readout_decompilation as rd

    if not FORBIDDEN:
        raise RuntimeError("build_forbidden(tokenizer) first")

    def refused_call(self, *args, **kwargs):
        hit = _scan(args) or _scan(kwargs)
        if hit:
            RECORD["token_guard_hits"].append(f"{type(self).__name__}: {hit}")
        RECORD["module_call_refusals"] += 1
        if len(RECORD["module_call_refusal_examples"]) < 10:
            RECORD["module_call_refusal_examples"].append(type(self).__name__)
        raise RuntimeError(f"review guard: torch.nn.Module.__call__ refused ({type(self).__name__}); this review runs no forward pass")

    torch.nn.Module.__call__ = refused_call
    originals = {id(f): (name, f) for name, f in (("pm.capture_prompt", pm.capture_prompt), ("pm.run_patched", pm.run_patched),
                                                     ("rd.capture_frame_020", rd.capture_frame_020), ("ri.capture_reference", ri.capture_reference),
                                                     ("ri.run_arm", ri.run_arm), ("capture.run_capture", capture.run_capture),
                                                     ("interventions.run_interventions", interventions.run_interventions),
                                                     ("ra.capture_reference", ra.capture_reference))}

    def make(name):
        def refused(*args, **kwargs):
            hit = _scan(args) or _scan(kwargs)
            if hit:
                RECORD["token_guard_hits"].append(f"{name}: {hit}")
            RECORD["entry_point_refusals"].append(name)
            raise RuntimeError(f"review guard: {name} refused; this review runs no forward pass")
        return refused

    wrappers = {key: make(name) for key, (name, _) in originals.items()}
    replaced = []
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name.startswith("neural_decompiler") or module_name.startswith("experiment_101")):
            continue
        for attribute, value in list(vars(module).items()):
            if id(value) in originals and value is originals[id(value)][1]:
                setattr(module, attribute, wrappers[id(value)])
                replaced.append(f"{module_name}.{attribute}")
    RECORD["entry_points_replaced"] = sorted(replaced)
    RECORD["forward_guards_installed"] = True


def seal(model) -> dict:
    count = 0
    for name, module in model.named_modules():
        def make(label=name or "model"):
            def refused(*args, **kwargs):
                RECORD["entry_point_refusals"].append(f"{label}.forward")
                raise RuntimeError(f"review guard: {label}.forward refused")
            return refused
        object.__setattr__(module, "forward", make())
        count += 1
    classes = []
    for cls in {type(m) for m in model.modules()}:
        for method in ("forward", "run_with_hooks", "run_with_cache", "generate"):
            if method in cls.__dict__ and callable(cls.__dict__[method]):
                def make(label=f"{cls.__qualname__}.{method}"):
                    def refused(*args, **kwargs):
                        RECORD["entry_point_refusals"].append(label)
                        raise RuntimeError(f"review guard: {label} refused")
                    return refused
                setattr(cls, method, make())
                classes.append(f"{cls.__qualname__}.{method}")
    RECORD["sealed"] = {"instances": count, "class_methods": sorted(classes)}
    return RECORD["sealed"]


def summary() -> dict:
    return {key: RECORD[key] for key in ("repo_write_refusals", "subprocess_refusals", "module_call_refusals", "entry_point_refusals", "token_guard_hits",
                                         "forbidden_sequences", "forward_guards_installed")}


def self_test() -> dict:
    """Decision-level test of the audit hook on synthetic event tuples (no real I/O touches the repository): repository
    writes in several spellings are refused; reads and writes outside the repository pass; non-git children refused."""
    saved = {key: list(RECORD[key]) for key in ("repo_write_refusals", "subprocess", "subprocess_refusals", "writes_outside_repo")}
    w = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    must_refuse = {
        "open w results": ("open", (str(ROOT / "outputs/experiment-101/results.json"), "w", w)),
        "open w stage1 file": ("open", (str(ROOT / "experiments/101-read-component-intervention/stage1/table.json"), "w", w)),
        "os.open O_RDONLY|O_TRUNC": ("open", (str(ROOT / "experiments/101-read-component-intervention/stage1/states.f64"), None, os.O_RDONLY | os.O_TRUNC)),
        "open a+ case variant": ("open", (str(ROOT).upper() + "/X", "a+", os.O_RDWR | os.O_CREAT | os.O_APPEND)),
        "open data-volume alias": ("open", ("/System/Volumes/Data" + str(ROOT) + "/y", "w", w)),
        "mkdir in repo": ("os.mkdir", (str(ROOT / "experiments/101-read-component-intervention/zz"), 511, -1)),
        "rename into repo": ("os.rename", (str(TMP / "a"), str(ROOT / "b"), -1, -1)),
        "remove in repo": ("os.remove", (str(ROOT / "outputs/experiment-101/results.json"), -1)),
        "relative open w with cwd at root": ("open", ("relative-name.json", "w", w)) if Path.cwd().resolve() == ROOT else ("open", (str(ROOT / "r.json"), "w", w)),
        "unresolvable dir_fd": ("os.remove", ("x", 10 ** 6)),
        "chmod in repo": ("os.chmod", (str(ROOT / "README.md"), 0o644, -1)),
        "utime in repo": ("os.utime", (str(ROOT / "README.md"), None, None, -1)),
        "child python": ("subprocess.Popen", (None, ["python", "-c", "1"], None, None)),
        "git commit": ("subprocess.Popen", (None, ["git", "commit", "-m", "x"], None, None)),
        "git add": ("subprocess.Popen", (None, ["git", "--no-optional-locks", "add", "x"], None, None)),
    }
    must_pass = {
        "read results": ("open", (str(ROOT / "outputs/experiment-101/results.json"), "r", os.O_RDONLY)),
        "write review out": ("open", (str(OUT / "selftest.json"), "w", w)),
        "write tmp": ("open", (str(TMP / "t"), "w", w)),
        "git status": ("subprocess.Popen", (None, ["git", "--no-optional-locks", "status", "--porcelain"], None, None)),
        "git ls-files": ("subprocess.Popen", (None, ["git", "ls-files", "--error-unmatch", "x"], None, None)),
    }
    out = {"refused": {}, "passed": {}}
    for name, (event, args) in must_refuse.items():
        try:
            _hook(event, args)
            out["refused"][name] = False
        except PermissionError:
            out["refused"][name] = True
    for name, (event, args) in must_pass.items():
        try:
            _hook(event, args)
            out["passed"][name] = True
        except PermissionError:
            out["passed"][name] = False
    out["ok"] = all(out["refused"].values()) and all(out["passed"].values())
    for key, value in saved.items():
        RECORD[key] = value
    RECORD["self_test"] = out
    if not out["ok"]:
        raise SystemExit(f"guard self-test failed: {out}")
    return out
