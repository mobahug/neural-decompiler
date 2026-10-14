"""The lock review's own guards and digest functions (written for this review; imports no neural_decompiler code).

- `install_fs_guard()` adds an audit hook BEFORE anything else is imported by a script. It refuses (raises
  PermissionError) and records every write, create, delete, rename, link, chmod, truncate or mkdir inside the
  repository, and every read under `outputs/` other than `outputs/experiment-020/results.json` and
  `outputs/experiment-101/`. Writes outside the repository are allowed only under this review directory; any other
  write outside the repository (a cache, a temporary file) is recorded, not refused.
- `forbidden_101_sequences()` tokenizes the 60 candidate texts of design §11 (parsed from the design markdown, not
  from `ri`) with each of the four cues, using the pinned `tokenizer.json` through the `tokenizers` library.
- `install_model_guards()` makes `torch.nn.Module.__call__` refuse every call (after checking any integer input
  against the forbidden sequences, for the record); `seal_model()` makes every module instance's `forward`, and the
  root's `run_with_hooks`, `run_with_cache` and `generate`, refuse; `wrap_entry_points()` replaces every capture and
  intervention entry point (and every alias of one, by identity, in any loaded module) with a refusing wrapper that
  checks the prompt's token ids.
- My own canonical JSON, sha256, tensor digest and git blob functions.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
REVIEW = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/lock-review").resolve()
OUT = REVIEW / "out"
OUTPUTS_ROOT = ROOT / "outputs"
READABLE_OUTPUTS = (ROOT / "outputs/experiment-020/results.json", ROOT / "outputs/experiment-101")
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"

EVENTS: dict = {"refused_repo_writes": [], "refused_output_reads": [], "outputs_reads": [], "writes_outside_repo_not_review": [],
                "module_calls_refused": 0, "module_calls_with_101_sequence": 0, "entry_points_refused": [], "entry_point_101_hits": 0,
                "model_loads": 0, "sealed_instances": 0, "aliases_wrapped": []}

WRITE_EVENTS_1 = {"os.remove": 1, "os.rmdir": 1, "shutil.rmtree": 1, "os.mkdir": 2, "os.chmod": 2, "os.chown": 3, "os.utime": 3, "os.truncate": None,
                  "os.chflags": None, "os.lchflags": None, "os.setxattr": None, "os.removexattr": None, "tempfile.mkstemp": None, "tempfile.mkdtemp": None}
WRITE_EVENTS_2 = ("os.rename", "os.link", "shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.move")


def _inside(path: Path, root: Path) -> bool:
    return path == root or str(path).startswith(str(root) + os.sep)


def _resolve(value, dir_fd=None):
    if isinstance(value, bool) or not isinstance(value, (str, bytes, os.PathLike)):
        return None
    try:
        raw = os.fsdecode(value)
        if isinstance(dir_fd, int) and not isinstance(dir_fd, bool) and dir_fd >= 0 and not os.path.isabs(raw):
            base = fcntl.fcntl(dir_fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0].decode()
            raw = os.path.join(base, raw)
        return Path(os.path.abspath(raw)).resolve()
    except (OSError, ValueError):
        return Path("<unresolvable>")


def _check_write(event: str, path) -> None:
    if path is None:
        return
    if str(path) == "<unresolvable>" or _inside(path, ROOT):
        EVENTS["refused_repo_writes"].append(f"{event} {path}")
        raise PermissionError(f"lock review: {event} refused on {path}")
    if not _inside(path, REVIEW):
        EVENTS["writes_outside_repo_not_review"].append(f"{event} {path}")


def _hook(event, args):
    if event == "open":
        path = _resolve(args[0]) if args else None
        if path is None:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)))
        if writing:
            _check_write("open-for-write", path)
            return
        if str(path) != "<unresolvable>" and _inside(path, OUTPUTS_ROOT):
            if not any(_inside(path, allowed) for allowed in READABLE_OUTPUTS):
                EVENTS["refused_output_reads"].append(str(path))
                raise PermissionError(f"lock review: reading {path} refused")
            if str(path) not in EVENTS["outputs_reads"]:
                EVENTS["outputs_reads"].append(str(path))
        return
    if event in WRITE_EVENTS_1:
        slot = WRITE_EVENTS_1[event]
        dir_fd = args[slot] if slot is not None and len(args) > slot else None
        _check_write(event, _resolve(args[0], dir_fd) if args else None)
        return
    if event in WRITE_EVENTS_2:
        src_fd = args[2] if event in ("os.rename", "os.link") and len(args) > 2 else None
        dst_fd = args[3] if event in ("os.rename", "os.link") and len(args) > 3 else None
        _check_write(event, _resolve(args[0], src_fd) if args else None)
        _check_write(event, _resolve(args[1], dst_fd) if len(args) > 1 else None)
        return
    if event == "os.symlink":
        _check_write(event, _resolve(args[1], args[2] if len(args) > 2 else None) if len(args) > 1 else None)


def install_fs_guard() -> None:
    sys.addaudithook(_hook)


# ---------------------------------------------------------------------------
# My own digests.

def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def content_digest(payload: dict) -> str:
    return sha256_text(canonical({k: v for k, v in payload.items() if k != "content_sha256"}))


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def f64_digest(values, shape) -> str:
    """sha256 over json(shape) | '<f8' | little-endian float64 bytes (row-major), from a flat Python float list."""
    import struct
    data = struct.pack("<" + "d" * len(values), *values)
    return sha256_bytes(json.dumps(list(shape)).encode("ascii") + b"|<f8|" + data)


def git(*args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True).returncode == 0


def git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True).stdout


# ---------------------------------------------------------------------------
# The 101 candidate texts, parsed from design §11 (the committed markdown), and their token sequences.

DESIGN_PATH = ROOT / "docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md"
CUES = (("one", 581), ("two", 767), ("three", 1264), ("many", 1142))


def design_candidates() -> list[str]:
    rows = {}
    for line in DESIGN_PATH.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"\| (\d+) \| (.+? \{cue\} \S+) \| (\d+) \| (.+? \{cue\} \S+) \|", line.strip())
        if m:
            rows[int(m.group(1))] = m.group(2)
            rows[int(m.group(3))] = m.group(4)
    if sorted(rows) != list(range(1, 61)):
        raise SystemExit(f"design §11 parse failed: {sorted(rows)}")
    return [rows[i] for i in range(1, 61)]


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json"))


def forbidden_101_sequences() -> dict[str, tuple[int, ...]]:
    tok = tokenizer()
    out = {}
    for number, text in enumerate(design_candidates(), start=1):
        for word, cue_id in CUES:
            ids = tuple(tok.encode(text.format(cue=word), add_special_tokens=False).ids)
            if cue_id not in ids:
                raise SystemExit(f"candidate {number} with {word}: the cue id {cue_id} is not in {ids}")
            out[f"{number:02d}|{word}"] = ids
    return out


def _rows_of(value):
    """Integer token rows found in an argument (tensor, list, tuple, or a Prompt-like object)."""
    rows = []
    try:
        import torch
        if isinstance(value, torch.Tensor):
            if not value.is_floating_point() and value.dim() in (1, 2):
                rows += [tuple(int(t) for t in row) for row in (value.reshape(1, -1) if value.dim() == 1 else value).tolist()]
            return rows
    except ImportError:
        pass
    if hasattr(value, "token_ids"):
        try:
            rows.append(tuple(int(t) for t in value.token_ids))
        except Exception:
            pass
    if isinstance(value, (list, tuple)) and value and all(isinstance(t, int) and not isinstance(t, bool) for t in value):
        rows.append(tuple(value))
    elif isinstance(value, (list, tuple)):
        for item in value:
            rows += _rows_of(item)
    return rows


def hits_101(args, kwargs, sequences) -> bool:
    for row in [r for value in (*args, *kwargs.values()) for r in _rows_of(value)]:
        for seq in sequences:
            n, m = len(row), len(seq)
            if any(row[i:i + m] == seq for i in range(0, max(0, n - m) + 1)) or (n >= 3 and seq[:n] == row):
                return True
    return False


def install_model_guards(sequences) -> None:
    import torch

    def refuse_call(self, *args, **kwargs):
        EVENTS["module_calls_refused"] += 1
        if hits_101(args, kwargs, sequences):
            EVENTS["module_calls_with_101_sequence"] += 1
        raise RuntimeError(f"lock review: torch module call refused ({type(self).__name__}); no forward pass is run")

    torch.nn.Module.__call__ = refuse_call


def seal_model(model) -> None:
    def refusing(label):
        def refuse(*args, **kwargs):
            EVENTS["entry_points_refused"].append(label)
            raise RuntimeError(f"lock review: {label} refused")
        return refuse

    for name, module in model.named_modules():
        object.__setattr__(module, "forward", refusing(f"{name or 'model'}.forward"))
        EVENTS["sealed_instances"] += 1
    for method in ("run_with_hooks", "run_with_cache", "generate"):
        if hasattr(model, method):
            object.__setattr__(model, method, refusing(f"model.{method}"))


def wrap_entry_points(sequences, functions: dict) -> None:
    """functions: label -> function object. Every attribute of any loaded module that IS one of them is replaced."""
    def make(label):
        def refuse(*args, **kwargs):
            EVENTS["entry_points_refused"].append(label)
            if hits_101(args, kwargs, sequences):
                EVENTS["entry_point_101_hits"] += 1
            raise RuntimeError(f"lock review: {label} refused")
        return refuse

    by_id = {id(f): (label, f, make(label)) for label, f in functions.items()}
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name.startswith("neural_decompiler") or module_name.startswith("experiment_101") or module_name == "__main__"):
            continue
        for attribute, value in list(vars(module).items()):
            entry = by_id.get(id(value))
            if entry is not None and value is entry[1]:
                setattr(module, attribute, entry[2])
                EVENTS["aliases_wrapped"].append(f"{module_name}.{attribute}")


def write_json(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path
