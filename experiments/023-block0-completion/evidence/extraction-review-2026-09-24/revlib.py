"""Independent reviewer helpers for the Experiment 023 exposed-cells artifact.

Everything here is written from the definitions, not imported from the project:
- canonical JSON (pm.canonical_json's definition: sort_keys, compact separators, ensure_ascii=False, allow_nan=False);
- the tensor digest (rc.tensor_digest's definition: sha256(json.dumps(shape) | kind | little-endian row-major bytes));
- git's blob id (sha1(b"blob <len>\\0" + bytes));
- a strict JSON loader that refuses duplicate keys.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
SCRATCH = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/extractreview023")

OUT023 = REPO / "outputs/experiment-023"
CAND_DATA = OUT023 / "candidate-exposed-cells.f64"
CAND_INDEX = OUT023 / "candidate-exposed-cells.json"
RESULTS = OUT023 / "results.json"
TABLE022 = REPO / "outputs/experiment-022/calibration-table.pt"
RECORD022 = REPO / "experiments/022-upstream-error-localization/calibration-v1.json"
CONF022 = REPO / "experiments/022-upstream-error-localization/confirmation-v1.json"

EXPECTED = {
    "head": "2111271892a8237f69c3ef18d6eec774e798ac5c",
    "f64_sha256": "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4",
    "json_sha256": "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe",
    "json_content": "d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16",
    "table_size": 424131002,
    "table_sha256": "04659d5e8b20a70526e5862eb2b2de952b33496f25e3875de6c6254df2b8fd81",
    "record022_file_prefix": "db745653",
    "record022_content_prefix": "46985fd5",
    "conf022_content_prefix": "af848ae4",
}


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def content_digest_without(payload: dict, key: str) -> str:
    return sha256_bytes(canonical({k: v for k, v in payload.items() if k != key}).encode("utf-8"))


def _no_dupes(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f"duplicate JSON key {k!r}")
        seen[k] = v
    return seen


def strict_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_no_dupes)


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def git(*args: str) -> bytes:
    import os
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, env=env).stdout


def tensor_digest_own(tensor) -> str:
    """rc.tensor_digest's definition, re-implemented: int64/int32 -> '<i8', everything else -> '<f8'."""
    import numpy as np
    import torch

    t = tensor.detach().cpu().contiguous()
    kind = "<i8" if t.dtype in (torch.int64, torch.int32) else "<f8"
    if t.dtype == torch.bool:
        arr = t.to(torch.float64).numpy()
    else:
        arr = t.numpy()
    data = np.ascontiguousarray(arr.astype(kind)).tobytes()
    header = json.dumps(list(t.shape)).encode("ascii") + b"|" + kind.encode("ascii") + b"|"
    return hashlib.sha256(header + data).hexdigest()


class Report:
    def __init__(self, name: str):
        self.name = name
        self.lines: list[str] = []
        self.fails: list[str] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        line = f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else "")
        print(line, flush=True)
        self.lines.append(line)
        if not ok:
            self.fails.append(label)
        return ok

    def note(self, text: str) -> None:
        print(f"[NOTE] {text}", flush=True)
        self.lines.append(f"[NOTE] {text}")

    def summary(self) -> str:
        text = f"== {self.name}: {len(self.lines)} lines, {len(self.fails)} failures: {self.fails}"
        print(text, flush=True)
        return text
