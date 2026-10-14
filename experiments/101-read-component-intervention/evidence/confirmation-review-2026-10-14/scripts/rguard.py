"""Guard for the Experiment 101 confirmation review (AI review, independent and read-only). Imported FIRST by every
review script.

1. An audit hook refuses every write-type event that resolves inside the repository (case-insensitive, with or without
   /System/Volumes/Data) and any child process except read-only git commands. Reads of outputs/ other than
   outputs/experiment-101/ and outputs/experiment-020/results.json are refused (the brief's rule).
2. ``install_forward_guards(tokenizer)`` (after importing neural_decompiler, before any model load):
   - builds the forbidden token sequences: each of the 60 candidate texts of design §11 with each of the four cues
     (one, two, three, many), with and without a leading space, and the 36 frozen frames' prefix + cue + suffix ids;
   - replaces ``torch.nn.Module.__call__`` with a refusal of EVERY module call (this review needs no forward pass);
     before refusing it scans the arguments for any forbidden sequence and records a token-guard hit;
   - replaces every forward entry point (pm.capture_prompt, pm.run_patched, pm.run_capture, pm.run_interventions,
     capture.run_capture, interventions.run_interventions, rd.capture_frame_020, ri.run_arm, ri.capture_reference,
     ra.capture_reference when present) by identity in every loaded neural_decompiler module with a refusal that also
     scans its arguments.
3. ``seal(model)`` wraps every submodule's forward, and the model classes' forward/run_with_hooks/run_with_cache/
   generate, with refusals.
4. At exit the record is written to out/<script>_guard.json: refusals, token-guard hits, write attempts, children.
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
RECORD: dict = {"script": SCRIPT, "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
                "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "PYTHONDONTWRITEBYTECODE", "GIT_OPTIONAL_LOCKS", "TMPDIR")},
                "repo_write_refusals": [], "outputs_read_refusals": [], "outputs_reads": [], "subprocess": [], "subprocess_refusals": [],
                "module_call_refusals": 0, "entry_point_refusals": [], "token_guard_hits": [], "forbidden_sequences": 0,
                "forward_guards_installed": False, "sealed": None, "model_loads": 0}
if not sys.flags.dont_write_bytecode:
    raise SystemExit("run with python -B")
if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
    raise SystemExit("set HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1")

_ROOT_KEY = str(ROOT).lower()
_DATA = "/system/volumes/data"
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_GIT_RO = {"rev-parse", "status", "ls-files", "diff", "merge-base", "ls-tree", "log", "show", "cat-file", "check-ignore", "rev-list",
           "branch", "describe", "for-each-ref", "hash-object"}
_OUTPUTS_OK = (str(ROOT / "outputs/experiment-101").lower(), str(ROOT / "outputs/experiment-020/results.json").lower())


def _resolve(value, dir_fd=None):
    if isinstance(value, int) and not isinstance(value, bool):
        return None
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


def _key(path: str) -> str:
    key = path.lower()
    if key.startswith(_DATA + "/"):
        key = key[len(_DATA):]
    return key


def _in_repo(path: str) -> bool:
    if path == "<unresolved>":
        return True
    key = _key(path)
    return key == _ROOT_KEY or key.startswith(_ROOT_KEY + "/")


def _refuse(kind: str, detail: str):
    RECORD[kind].append(detail)
    raise PermissionError(f"review guard: {kind}: {detail}")


_PATH_EVENTS = {"os.mkdir": [(0, 2)], "os.rmdir": [(0, 1)], "os.remove": [(0, 1)], "shutil.rmtree": [(0, 1)], "os.truncate": [(0, None)],
                "os.chmod": [(0, 2)], "os.chown": [(0, 3)], "os.utime": [(0, 3)], "os.chflags": [(0, None)], "os.setxattr": [(0, None)],
                "os.removexattr": [(0, None)], "os.rename": [(0, 2), (1, 3)], "os.link": [(0, 2), (1, 3)], "os.symlink": [(1, 2)],
                "shutil.copyfile": [(1, None)], "shutil.copymode": [(1, None)], "shutil.copystat": [(1, None)], "shutil.copytree": [(1, None)],
                "shutil.move": [(0, None), (1, None)], "tempfile.mkstemp": [(0, None)], "tempfile.mkdtemp": [(0, None)],
                "sqlite3.connect": [(0, None)], "os.mkfifo": [(0, 2)], "os.mknod": [(0, 3)]}


def _hook(event, args):
    if event == "open":
        if not args:
            return
        path = _resolve(args[0])
        if path is None:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        if writing:
            if _in_repo(path):
                _refuse("repo_write_refusals", f"open {path} mode={mode!r} flags={flags!r}")
            return
        key = _key(path)
        if key.startswith(_ROOT_KEY + "/outputs/"):
            if not any(key == ok or key.startswith(ok + "/") for ok in _OUTPUTS_OK):
                _refuse("outputs_read_refusals", path)
            if path not in RECORD["outputs_reads"]:
                RECORD["outputs_reads"].append(path)
        return
    if event in _PATH_EVENTS:
        for index, fd_index in _PATH_EVENTS[event]:
            if len(args) <= index:
                continue
            dir_fd = args[fd_index] if fd_index is not None and len(args) > fd_index else None
            path = _resolve(args[index], dir_fd)
            if path is not None and _in_repo(path):
                _refuse("repo_write_refusals", f"{event} {path}")
        return
    if event == "subprocess.Popen":
        argv = args[1] if len(args) > 1 else None
        argv_list = [os.fsdecode(a) for a in argv] if isinstance(argv, (list, tuple)) else [str(argv)]
        RECORD["subprocess"].append(argv_list)
        name = os.path.basename(argv_list[0]) if argv_list else ""
        sub = [a for a in argv_list[1:] if not a.startswith("-")]
        if name != "git" or not sub or sub[0] not in _GIT_RO:
            _refuse("subprocess_refusals", repr(argv_list))
        return
    if event in ("os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "pty.spawn"):
        _refuse("subprocess_refusals", f"{event} {repr(args)[:200]}")


sys.addaudithook(_hook)


def _write_record():
    try:
        OUT.mkdir(exist_ok=True)
        (OUT / f"{SCRIPT}_guard.json").write_text(json.dumps(RECORD, indent=1, default=str) + "\n", encoding="utf-8")
    except Exception as error:  # never mask the script's own outcome
        print(f"guard record not written: {error}", file=sys.stderr)


atexit.register(_write_record)

# ---------------------------------------------------------------------------------------------------------------------
FORBIDDEN: list[tuple[int, ...]] = []
CUE_WORDS = {"one": 581, "two": 767, "three": 1264, "many": 1142}


def build_forbidden(tokenizer) -> int:
    from neural_decompiler import read_intervention as ri

    seqs: set[tuple[int, ...]] = set()
    lock = json.loads((ROOT / ri.LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
    candidates = list(lock["configuration"]["frame_candidates"])
    if candidates != list(ri.FRAME_CANDIDATES) or len(candidates) != 60:
        raise SystemExit("the candidate list is not the lock's 60 texts")
    for text in candidates:
        for word, cue_id in CUE_WORDS.items():
            filled = text.format(cue=word)
            for variant in (filled, " " + filled):
                ids = tuple(int(t) for t in tokenizer.encode(variant))
                seqs.add(ids)
    payload = json.loads((ROOT / ri.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))
    for frame in payload["frames"]:
        for cue_id in CUE_WORDS.values():
            seqs.add(tuple(int(t) for t in (*frame["prefix_ids"], cue_id, *frame["suffix_ids"])))
            seqs.add(tuple(int(t) for t in (*frame["prefix_ids"], cue_id)))  # a truncated prompt up to the cue
    FORBIDDEN[:] = sorted(seqs)
    RECORD["forbidden_sequences"] = len(FORBIDDEN)
    return len(FORBIDDEN)


def _contains(haystack: list[int], needle: tuple[int, ...]) -> bool:
    n = len(needle)
    return any(tuple(haystack[i:i + n]) == needle for i in range(len(haystack) - n + 1))


def _scan(value, depth=0):
    import torch

    if depth > 4:
        return None
    if isinstance(value, torch.Tensor):
        if value.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8, torch.long) and value.numel() <= 100_000:
            rows = value.reshape(-1, value.shape[-1]).tolist() if value.dim() >= 1 else [[int(value)]]
            for row in rows:
                for needle in FORBIDDEN:
                    if _contains([int(v) for v in row], needle):
                        return f"tensor contains {needle}"
        return None
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            for needle in FORBIDDEN:
                if _contains(list(value), needle):
                    return f"sequence contains {needle}"
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
    for attribute in ("tokens", "token_ids", "input_ids", "ids"):
        if hasattr(value, attribute):
            try:
                hit = _scan(getattr(value, attribute), depth + 1)
            except Exception:
                hit = None
            if hit:
                return hit
    return None


def _scan_all(args, kwargs):
    for value in (*args, *kwargs.values()):
        hit = _scan(value)
        if hit:
            return hit
    return None


def install_forward_guards(tokenizer) -> None:
    import torch

    import neural_decompiler
    from neural_decompiler import capture, interventions
    from neural_decompiler import plural_mechanism as pm
    from neural_decompiler import read_intervention as ri
    from neural_decompiler import readout_decompilation as rd

    build_forbidden(tokenizer)

    def refused_call(self, *args, **kwargs):
        hit = _scan_all(args, kwargs)
        if hit:
            RECORD["token_guard_hits"].append(f"module call {type(self).__name__}: {hit}")
        RECORD["module_call_refusals"] += 1
        raise RuntimeError(f"review guard: module call refused ({type(self).__name__})")

    torch.nn.Module.__call__ = refused_call
    originals = {}
    for module, name in ((pm, "capture_prompt"), (pm, "run_patched"), (pm, "run_capture"), (pm, "run_interventions"), (capture, "run_capture"),
                         (interventions, "run_interventions"), (rd, "capture_frame_020"), (ri, "run_arm"), (ri, "capture_reference")):
        if hasattr(module, name):
            originals[id(getattr(module, name))] = f"{module.__name__}.{name}"
    try:
        from neural_decompiler import read_assembly as ra

        if hasattr(ra, "capture_reference"):
            originals[id(ra.capture_reference)] = "read_assembly.capture_reference"
    except Exception:
        pass

    def make(label):
        def refused(*args, **kwargs):
            hit = _scan_all(args, kwargs)
            if hit:
                RECORD["token_guard_hits"].append(f"{label}: {hit}")
            RECORD["entry_point_refusals"].append(label)
            raise RuntimeError(f"review guard: {label} refused")

        return refused

    replaced = []
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("neural_decompiler"):
            continue
        for attribute, value in list(vars(module).items()):
            if callable(value) and id(value) in originals:
                setattr(module, attribute, make(originals[id(value)]))
                replaced.append(f"{module_name}.{attribute}")
    RECORD["entry_points_replaced"] = sorted(replaced)
    RECORD["forward_guards_installed"] = True


def seal(model) -> dict:
    import functools

    sealed = {"instances": 0, "classes": []}

    def make(label, original):
        @functools.wraps(original)
        def refused(*args, **kwargs):
            hit = _scan_all(args, kwargs)
            if hit:
                RECORD["token_guard_hits"].append(f"{label}: {hit}")
            RECORD["entry_point_refusals"].append(label)
            raise RuntimeError(f"review guard: {label} refused")

        return refused

    for name, module in model.named_modules():
        object.__setattr__(module, "forward", make(f"{name or 'model'}.forward", module.forward))
        sealed["instances"] += 1
    for cls in sorted({type(module) for module in model.modules()}, key=lambda c: c.__qualname__):
        for method in ("forward", "run_with_hooks", "run_with_cache", "generate"):
            if method in cls.__dict__ and callable(cls.__dict__[method]):
                setattr(cls, method, make(f"{cls.__qualname__}.{method}", cls.__dict__[method]))
                sealed["classes"].append(f"{cls.__qualname__}.{method}")
    RECORD["sealed"] = sealed
    RECORD["model_loads"] += 1
    return sealed


def self_test(tokenizer=None) -> dict:
    """The guard refuses a repository write and a module call (no model involved)."""
    results = {}
    try:
        open(ROOT / "outputs/experiment-101/.review-guard-selftest", "w")
        results["repo_write_refused"] = False
    except PermissionError:
        results["repo_write_refused"] = True
    try:
        open(ROOT / "outputs/experiment-025/results.json", "r")
        results["other_outputs_read_refused"] = False
    except PermissionError:
        results["other_outputs_read_refused"] = True
    except FileNotFoundError:
        results["other_outputs_read_refused"] = False
    # the self-test's own refusals are expected; keep them out of the run's record
    RECORD["repo_write_refusals"] = [r for r in RECORD["repo_write_refusals"] if ".review-guard-selftest" not in r]
    RECORD["outputs_read_refusals"] = [r for r in RECORD["outputs_read_refusals"] if "experiment-025" not in r]
    RECORD["self_test"] = results
    if not all(results.values()):
        raise SystemExit(f"guard self-test failed: {results}")
    return results
