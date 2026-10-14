"""Shared, review-only helpers. Imports no neural_decompiler code. Installs a write guard: any attempt to write, delete,
rename or mkdir inside the repository raises (the review is read-only), and records the attempt."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
REVIEW = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/calibration-review").resolve()
OUT = REVIEW / "out"
CAL = REVIEW.parent / "calibrate"
SNAPSHOT = Path("/Users/gaborhorvath-ulenius/.cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c")
OUTPUTS_101 = ROOT / "outputs/experiment-101"
RESULTS_020 = ROOT / "outputs/experiment-020/results.json"

WRITE_ATTEMPTS: list[str] = []
FORBIDDEN_READS: list[str] = []


def _inside(path: Path, root: Path) -> bool:
    return path == root or str(path).startswith(str(root) + os.sep)


def _resolve(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    try:
        return Path(os.fsdecode(value)).resolve()
    except (OSError, ValueError):
        return None


def _hook(event, args):
    if event == "open":
        path = _resolve(args[0]) if args else None
        if path is None:
            return
        mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if _inside(path, ROOT) and not _inside(path, ROOT / ".venv"):
            if writing:
                WRITE_ATTEMPTS.append(f"open-write {path}")
                raise PermissionError(f"review is read-only: refused write to {path}")
            if _inside(path, ROOT / "outputs") and not (_inside(path, OUTPUTS_101) or path == RESULTS_020 or path == ROOT / "outputs"):
                FORBIDDEN_READS.append(str(path))
                raise PermissionError(f"review may not read {path}")
        return
    if event in ("os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir", "os.chmod", "os.rename", "os.link", "os.symlink", "os.utime"):
        paths = [p for p in (_resolve(a) for a in args[:2]) if p is not None]
        if any(_inside(p, ROOT) and not _inside(p, ROOT / ".venv") for p in paths):
            WRITE_ATTEMPTS.append(f"{event} {paths}")
            raise PermissionError(f"review is read-only: refused {event} {paths}")


sys.addaudithook(_hook)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> str:
    """My own canonical JSON: sorted keys, compact separators, UTF-8, no NaN."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def my_tensor_digest(t) -> str:
    """sha256 over json(shape) | kind | little-endian bytes (int64 for integer tensors, float64 otherwise)."""
    import numpy as np
    a = t.detach().cpu().contiguous().numpy()
    kind = "<i8" if a.dtype.kind in "iu" else "<f8"
    data = a.astype(kind, copy=False).tobytes()
    return hashlib.sha256(json.dumps(list(a.shape)).encode("ascii") + b"|" + kind.encode("ascii") + b"|" + data).hexdigest()


def float32_digest(t) -> str:
    """The launcher's run-log digest: sha256 of the flattened float32 bytes."""
    import numpy as np
    return hashlib.sha256(t.detach().cpu().float().contiguous().reshape(-1).numpy().astype("<f4", copy=False).tobytes()).hexdigest()


def dump(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    return path


# --- My own arm and tag list (from plan § Constants, "Arms and order"), written independently of ri. -----------------
PRIMARY = "two"
CUES = (("two", 767, "cardinal:pl"), ("three", 1264, "three"), ("many", 1142, "many"))  # (cue label, id, 020 prompt label)
REF_ID, REF_LABEL = 581, "ref"
MAIN = ["PL", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF", "PC-SWAP-PL", "PC-SWAP-REF"]
PRIMARY_ARMS = (["PL", "NOOP"] + MAIN[1:] + [f"NEC-ISO-{j}" for j in range(1, 9)] + [f"SUF-ISO-{j}" for j in range(1, 9)]
                + ["HEAD-NEC", "HEAD-SUF", "HEAD-NEC-V", "HEAD-SUF-V"])
REF_PROMPT_ARMS = {"REF", "SUF-READ", "SUF-OFF", "SUF-NEG", "SUF-HALF", "PC-SWAP-REF", "HEAD-SUF", "HEAD-SUF-V"} | {f"SUF-ISO-{j}" for j in range(1, 9)}


def frame_runs():
    """(tag, arm, cue label or None, runs on the reference prompt?) in the plan's run order within one frame."""
    runs = [("REF", "REF", None, True)]
    for cue, _, _ in CUES:
        arms = PRIMARY_ARMS if cue == PRIMARY else MAIN
        for arm in arms:
            runs.append((f"{arm}@{cue}", arm, cue, arm in REF_PROMPT_ARMS))
    return runs


def arm_kind(arm: str) -> str:
    if arm in ("REF", "PL"):
        return "plain"
    if arm == "NOOP":
        return "noop"
    if arm in ("NEC-READ", "SUF-READ", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF"):
        return "read"
    if arm in ("NEC-OFF", "SUF-OFF"):
        return "off"
    if arm.startswith(("NEC-ISO-", "SUF-ISO-")):
        return "iso"
    if arm.startswith("PC-SWAP"):
        return "swap"
    if arm.startswith("HEAD-"):
        return "head"
    raise ValueError(arm)


READ_COEFF = {"NEC-READ": -1.0, "SUF-READ": 1.0, "NEC-AMP": 1.0, "SUF-NEG": -1.0, "NEC-HALF": -0.5, "SUF-HALF": 0.5}
