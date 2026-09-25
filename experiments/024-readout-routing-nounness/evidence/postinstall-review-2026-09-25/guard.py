"""Post-install review 024 guard (own code). Import FIRST in every script.

An audit hook that refuses every write / create / remove / rename / metadata change whose target resolves under the
repository root, logs every such event elsewhere (system temp directory, HF cache, this scratch directory), and allows
subprocesses only for git with a read-only subcommand (plus sysctl/uname probes). dir_fd-relative operations are resolved
through the directory descriptor (F_GETPATH), so filelock's dir_fd probes in the system temp directory are judged by
where they really are. The hook is self-tested with sys.audit (event dispatch only: no file operation happens).
Also: forward/capture sealing helpers for the model rehearsal.
"""

from __future__ import annotations

import atexit
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading

REPO = os.path.realpath("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
SCRATCH = os.path.realpath("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/"
                           "8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/postinstall_review024")
_REPO_KEY = REPO.lower()  # APFS is case-insensitive: compare case-folded (over-refusal is the safe side)
os.environ["GIT_OPTIONAL_LOCKS"] = "0"  # git status must not refresh the index as a side effect

STATS: dict = {"refused": [], "outside": {}, "subprocesses": [], "selftest": {}, "open_dir_fd_wrapped": False}
_local = threading.local()
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | os.O_EXCL
GIT_READ_ONLY = {"rev-parse", "status", "ls-files", "merge-base", "diff", "show", "cat-file", "log", "ls-tree", "ls-remote",
                 "rev-list", "for-each-ref", "version", "config"}
OTHER_ALLOWED = {"sysctl", "uname"}


class RepoWriteRefused(PermissionError):
    pass


def _fd_path(fd: int) -> str | None:
    try:
        raw = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))
        return raw.split(b"\0", 1)[0].decode()
    except Exception:
        return None


def _candidates(path, dir_fd=None) -> list[str] | None:
    """Absolute candidates (lexical, parent-realpath, full realpath) for a path argument; None if unresolvable."""
    if isinstance(path, int):
        p = _fd_path(path)
        return None if p is None else [p, os.path.realpath(p)]
    try:
        p = os.fsdecode(os.fspath(path))
    except Exception:
        return None
    if not os.path.isabs(p):
        if isinstance(dir_fd, int) and dir_fd >= 0:
            base = _fd_path(dir_fd)
            if base is None:
                return None
        else:
            base = os.getcwd()
        p = os.path.join(base, p)
    lexical = os.path.normpath(p)
    return [lexical, os.path.join(os.path.realpath(os.path.dirname(lexical)), os.path.basename(lexical)), os.path.realpath(lexical)]


def inside_repo(candidates: list[str]) -> bool:
    return any(c.lower() == _REPO_KEY or c.lower().startswith(_REPO_KEY + os.sep) for c in candidates)


def _judge(event: str, items) -> None:
    for path, dir_fd in items:
        if path is None:
            continue
        cands = _candidates(path, dir_fd)
        if cands is None or inside_repo(cands):
            entry = {"event": event, "path": repr(path), "dir_fd": dir_fd, "resolved": cands}
            if getattr(_local, "selftest", False):
                STATS["selftest"].setdefault("refused", []).append(entry)
            else:
                STATS["refused"].append(entry)
            raise RepoWriteRefused(f"postinstall_review024 guard refused {event} on {cands or path!r} (under the repository or unresolvable)")
        key = "selftest_outside" if getattr(_local, "selftest", False) else event
        (STATS["selftest"] if key == "selftest_outside" else STATS["outside"]).setdefault(key, set()).add(cands[-1])


def _command_ok(argv) -> bool:
    try:
        args = [os.fsdecode(a) for a in argv]
    except Exception:
        return False
    if not args:
        return False
    exe = os.path.basename(args[0])
    if exe in OTHER_ALLOWED:
        return True
    if exe != "git":
        return False
    i = 1
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-C", "-c") else 1
    if i >= len(args):
        return False
    sub = args[i]
    if sub == "config":  # only reads
        return any(a in ("--get", "--get-all", "--list", "-l") for a in args[i + 1:])
    return sub in GIT_READ_ONLY


def _spawn(event: str, argv) -> None:
    ok = _command_ok(argv if not isinstance(argv, (str, bytes)) else [argv])
    STATS["subprocesses"].append({"event": event, "argv": [os.fsdecode(a) for a in argv] if not isinstance(argv, (str, bytes)) else [os.fsdecode(argv)], "allowed": ok})
    if not ok:
        raise RepoWriteRefused(f"postinstall_review024 guard refused the subprocess {argv!r}")


def _hook(event: str, args) -> None:
    if getattr(_local, "busy", False):
        return
    handler = _HANDLERS.get(event)
    if handler is None:
        return
    _local.busy = True
    try:
        handler(event, args)
    finally:
        _local.busy = False


def _open(event, args):
    path, mode, flags = args
    writes = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and bool(flags & WRITE_FLAGS))
    if writes:
        _judge(event, [(path, getattr(_local, "open_dir_fd", None))])


_HANDLERS = {
    "open": _open,
    "os.rename": lambda e, a: _judge(e, [(a[0], a[2]), (a[1], a[3])]),
    "os.remove": lambda e, a: _judge(e, [(a[0], a[1])]),
    "os.rmdir": lambda e, a: _judge(e, [(a[0], a[1])]),
    "os.mkdir": lambda e, a: _judge(e, [(a[0], a[2])]),
    "os.link": lambda e, a: _judge(e, [(a[0], a[2]), (a[1], a[3])]),
    "os.symlink": lambda e, a: _judge(e, [(a[1], a[2])]),
    "os.truncate": lambda e, a: _judge(e, [(a[0], None)]),
    "os.chmod": lambda e, a: _judge(e, [(a[0], a[2] if len(a) > 2 else None)]),
    "os.chown": lambda e, a: _judge(e, [(a[0], a[3] if len(a) > 3 else None)]),
    "os.utime": lambda e, a: _judge(e, [(a[0], a[3] if len(a) > 3 else None)]),
    "os.chflags": lambda e, a: _judge(e, [(a[0], None)]),
    "os.lchflags": lambda e, a: _judge(e, [(a[0], None)]),
    "os.setxattr": lambda e, a: _judge(e, [(a[0], None)]),
    "os.removexattr": lambda e, a: _judge(e, [(a[0], None)]),
    "os.mkfifo": lambda e, a: _judge(e, [(a[0], a[2] if len(a) > 2 else None)]),
    "os.mknod": lambda e, a: _judge(e, [(a[0], a[3] if len(a) > 3 else None)]),
    "shutil.copyfile": lambda e, a: _judge(e, [(a[1], None)]),
    "shutil.copymode": lambda e, a: _judge(e, [(a[1], None)]),
    "shutil.copystat": lambda e, a: _judge(e, [(a[1], None)]),
    "shutil.copytree": lambda e, a: _judge(e, [(a[1], None)]),
    "shutil.move": lambda e, a: _judge(e, [(a[0], None), (a[1], None)]),
    "shutil.rmtree": lambda e, a: _judge(e, [(a[0], a[1] if len(a) > 1 else None)]),
    "shutil.chown": lambda e, a: _judge(e, [(a[0], None)]),
    "shutil.make_archive": lambda e, a: _judge(e, [(a[0], None)]),
    "shutil.unpack_archive": lambda e, a: _judge(e, [(a[1], None)]),
    "tempfile.mkstemp": lambda e, a: _judge(e, [(a[0], None)]),
    "tempfile.mkdtemp": lambda e, a: _judge(e, [(a[0], None)]),
    "subprocess.Popen": lambda e, a: _spawn(e, a[1]),
    "os.posix_spawn": lambda e, a: _spawn(e, a[1]),
    "os.spawn": lambda e, a: _spawn(e, a[2]),
    "os.exec": lambda e, a: _spawn(e, a[1]),
    "os.system": lambda e, a: _spawn(e, [a[0]]),
}

# os.open with dir_fd: the "open" audit event carries no dir_fd, so remember it for the hook while the call runs.
_real_os_open = os.open


def _guarded_os_open(path, flags, mode=0o777, *, dir_fd=None):
    previous = getattr(_local, "open_dir_fd", None)
    _local.open_dir_fd = dir_fd
    try:
        return _real_os_open(path, flags, mode, dir_fd=dir_fd)
    finally:
        _local.open_dir_fd = previous


def _selftest() -> None:
    """Event dispatch only (sys.audit performs no operation): the hook must refuse repo targets and allow temp targets."""
    import tempfile

    tmp = os.path.realpath(tempfile.gettempdir())
    results = {}
    _local.selftest = True
    try:
        cases = [
            ("repo open w", "refuse", ("open", os.path.join(REPO, "outputs/experiment-024/results.json"), "w", 0)),
            ("repo os.open O_CREAT", "refuse", ("open", os.path.join(REPO, "zz_guard_probe"), None, os.O_CREAT | os.O_WRONLY)),
            ("repo rename (os.replace)", "refuse", ("os.rename", os.path.join(tmp, "a"), os.path.join(REPO, "outputs/experiment-024/results.json"), -1, -1)),
            ("repo remove", "refuse", ("os.remove", os.path.join(REPO, "outputs/experiment-024/results.json"), -1)),
            ("repo mkdir", "refuse", ("os.mkdir", os.path.join(REPO, "zz_dir"), 0o777, -1)),
            ("repo relative remove (cwd)", "refuse", ("os.remove", "zz_relative_probe", -1)),
            ("repo case-variant", "refuse", ("open", REPO.upper() + "/X", "w", 0)),
            ("repo read", "allow", ("open", os.path.join(REPO, "README.md"), "r", os.O_RDONLY)),
            ("temp open w", "allow", ("open", os.path.join(tmp, "zz_probe"), "w", 0)),
            ("scratch open w", "allow", ("open", os.path.join(SCRATCH, "zz_probe"), "w", 0)),
        ]
        dfd = _real_os_open(tmp, os.O_RDONLY)
        try:
            cases.append(("temp dir_fd remove", "allow", ("os.remove", "zz_probe", dfd)))
            for name, expect, event in cases:
                try:
                    sys.audit(*event)
                    got = "allow"
                except RepoWriteRefused:
                    got = "refuse"
                results[name] = {"expected": expect, "got": got}
            _local.open_dir_fd = dfd
            try:
                sys.audit("open", "zz_probe_rel", None, os.O_CREAT | os.O_WRONLY)
                results["temp dir_fd os.open"] = {"expected": "allow", "got": "allow"}
            except RepoWriteRefused:
                results["temp dir_fd os.open"] = {"expected": "allow", "got": "refuse"}
            finally:
                _local.open_dir_fd = None
        finally:
            os.close(dfd)
        for argv, expect in ((["git", "rev-parse", "HEAD"], True), (["git", "-C", REPO, "status", "--porcelain"], True), (["git", "commit", "-m", "x"], False),
                             (["git", "push"], False), (["git", "config", "user.name", "x"], False), (["rm", "-rf", "x"], False)):
            results["cmd " + " ".join(argv)] = {"expected": "allow" if expect else "refuse", "got": "allow" if _command_ok(argv) else "refuse"}
    finally:
        _local.selftest = False
    STATS["selftest"]["cases"] = results
    bad = {k: v for k, v in results.items() if v["expected"] != v["got"]}
    if bad:
        raise SystemExit(f"GUARD SELFTEST FAILED: {bad}")


def install() -> None:
    if getattr(sys, "_postinstall024_guard", False):
        return
    if os.environ.get("HF_HUB_OFFLINE") != "1" or not sys.dont_write_bytecode:
        raise SystemExit("run with HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1")
    sys.addaudithook(_hook)
    os.open = _guarded_os_open
    try:
        os.supports_dir_fd.add(_guarded_os_open)  # keep `os.open in os.supports_dir_fd` capability probes truthful
    except Exception:
        pass
    STATS["open_dir_fd_wrapped"] = True
    sys._postinstall024_guard = True
    _selftest()
    atexit.register(_report)


def summary() -> dict:
    return {"refused": len(STATS["refused"]), "refused_entries": STATS["refused"],
            "outside_writes": {k: sorted(v) for k, v in STATS["outside"].items()},
            "subprocesses": [" ".join(s["argv"][:6]) + (" ..." if len(s["argv"]) > 6 else "") + ("" if s["allowed"] else "  [REFUSED]") for s in STATS["subprocesses"]],
            "selftest_cases": len(STATS["selftest"].get("cases", {})), "selftest_all_as_expected": True}


def _report() -> None:
    s = summary()
    print(f"GUARD SUMMARY: refused={s['refused']} selftest_cases={s['selftest_cases']} (all as expected) "
          f"subprocesses={len(s['subprocesses'])} outside_write_events={ {k: len(v) for k, v in s['outside_writes'].items()} }", flush=True)
    for k, v in s["outside_writes"].items():
        for p in v[:12]:
            print(f"GUARD outside write [{k}]: {p}", flush=True)
    uniq = sorted(set(s["subprocesses"]))
    for c in uniq:
        print(f"GUARD subprocess: {c}", flush=True)
    if s["refused"]:
        print(f"GUARD REFUSED ENTRIES: {s['refused_entries']}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Small own helpers.

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def git(*args: str, text: bool = True):
    out = subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=text)
    return out.stdout


def canonical(value) -> str:
    """Own canonical JSON: sorted keys, compact separators, UTF-8 (no ASCII escaping), NaN refused."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


class Seal:
    """Counting / refusing wrappers for torch module calls, forwards and capture entry points."""

    def __init__(self):
        self.counts = {"module_calls_during_load": 0, "module_call_attempts_after_seal": 0, "forward_stub_hits": 0, "capture_hits": 0,
                       "forward_classes_stubbed": 0, "instance_forwards_stubbed": 0, "call_overrides_found": []}
        self.capture_hits_by_name: dict[str, int] = {}
        self._saved = []

    def count_module_calls(self):
        import torch

        original = torch.nn.Module.__call__
        seal = self

        def counting(module, *args, **kwargs):
            seal.counts["module_calls_during_load"] += 1
            return original(module, *args, **kwargs)

        torch.nn.Module.__call__ = counting
        self._original_call = original

    def seal_model(self, model):
        import torch

        seal = self

        def refuse_call(module, *args, **kwargs):
            seal.counts["module_call_attempts_after_seal"] += 1
            raise RuntimeError(f"sealed: a torch module call was attempted ({type(module).__name__})")

        torch.nn.Module.__call__ = refuse_call

        def make_stub(cls_name):
            def refuse_forward(*args, **kwargs):
                seal.counts["forward_stub_hits"] += 1
                raise RuntimeError(f"sealed: {cls_name}.forward was reached")
            return refuse_forward

        classes = {type(m) for m in model.modules()}
        for cls in classes:
            for base in cls.__mro__:
                if base is torch.nn.Module or base is object:
                    continue
                if "__call__" in vars(base):
                    self.counts["call_overrides_found"].append(f"{base.__module__}.{base.__qualname__}")
                    setattr(base, "__call__", refuse_call)
            cls.forward = make_stub(cls.__name__)
            self.counts["forward_classes_stubbed"] += 1
        for m in model.modules():
            if "forward" in vars(m):
                m.forward = make_stub(type(m).__name__ + "(instance)")
                self.counts["instance_forwards_stubbed"] += 1
        return sorted(f"{c.__module__}.{c.__qualname__}" for c in classes)

    def refuse_captures(self):
        from neural_decompiler import capture as capture_module
        from neural_decompiler import interventions as interventions_module
        from neural_decompiler import plural_mechanism as pm
        from neural_decompiler import upstream_localization as ul

        seal = self
        targets = [(pm, n) for n in ("capture_prompt", "run_patched", "run_capture", "run_interventions", "record_execution")]
        targets += [(ul, "stage_two_022"), (ul, "measure_prompt"), (capture_module, "run_capture"), (interventions_module, "run_interventions")]
        refused = []
        for module, name in targets:
            if not hasattr(module, name):
                continue

            def make(label):
                def refuse(*args, **kwargs):
                    seal.counts["capture_hits"] += 1
                    seal.capture_hits_by_name[label] = seal.capture_hits_by_name.get(label, 0) + 1
                    raise RuntimeError(f"sealed: {label} was reached")
                return refuse

            label = f"{module.__name__.rsplit('.', 1)[-1]}.{name}"
            setattr(module, name, make(label))
            refused.append(label)
        return refused


install()
