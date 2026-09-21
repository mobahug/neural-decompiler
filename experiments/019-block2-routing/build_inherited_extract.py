#!/usr/bin/env python3
"""Build Experiment 019's inherited Experiment 018 extract deterministically from the closed 018 results state and the 018 lock (no model run).

Reads ``outputs/experiment-018/results.json`` (a completed confirmation) and the committed 018 preregistration lock, and writes
``experiments/019-block2-routing/inherited/experiment-018-pair-extract.json`` whose ``source`` names the exact results-state file
sha256 and ``state_sha256``, the lock file sha256 and content sha256, the 018 run id, its explore and confirm commits and the
extraction schema version. Rebuilding from the same source gives a byte-identical file. Refuses to overwrite unless ``--check`` is
given, in which case the rebuilt content digest is compared with the committed file's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/018-block2-concentration"))
sys.path.insert(0, str(ROOT / "src"))

import run as run018  # noqa: E402
from neural_decompiler import block_concentration as bc  # noqa: E402
from neural_decompiler import block_routing as br  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402

RESULTS_018 = ROOT / "outputs/experiment-018/results.json"
LOCK_018 = ROOT / bc.LOCK_RELATIVE_PATH
TARGET = ROOT / br.INHERITED_018_EXTRACT_RELATIVE_PATH


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    runner = run018.Runner()
    pool_018, _, _, _, lock_017, _, digests = runner._base_inputs()
    confirmation_018 = bc.load_confirmation(ROOT / bc.CONFIRMATION_RELATIVE_PATH, pool_018, digests)
    digests = dict(digests) | {"confirmation_018": confirmation_018.content_sha256}
    lock_018 = run018._load_lock(LOCK_018, "018")
    required = ("locked_states", "bases_3", "base2_pt", "scores", "subsets", "frame_subsets", "confirmation_018_sha256", "lock_017_sha256", "content_sha256")
    if any(key not in lock_018 for key in required) or lock_018["confirmation_018_sha256"] != confirmation_018.content_sha256 or lock_018["lock_017_sha256"] != lock_017["content_sha256"]:
        raise ValueError("the Experiment 018 lock does not carry the locked record for the frozen 018 confirmation set and the 017 lock")
    digests["lock_018"] = lock_018["content_sha256"]
    results_state = bc.load_results_state(RESULTS_018)
    return br.build_inherited_extract_018(results_state, lock_018, digests=digests, results_state_path=str(RESULTS_018.relative_to(ROOT)), results_state_file_sha256=file_sha256(RESULTS_018),
                                          lock_path=str(LOCK_018.relative_to(ROOT)), lock_file_sha256=file_sha256(LOCK_018))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="rebuild and compare the content digest with the committed extract instead of writing")
    args = parser.parse_args(argv)
    payload = build()
    text = pm.canonical_json(payload) + "\n"
    if args.check:
        committed = json.loads(TARGET.read_text(encoding="utf-8"))
        same = committed["content_sha256"] == payload["content_sha256"] and TARGET.read_text(encoding="utf-8") == text
        print(f"rebuilt content sha256 {payload['content_sha256']}; committed {committed['content_sha256']}; {'IDENTICAL' if same else 'DIFFERENT'}")
        return 0 if same else 1
    if TARGET.exists():
        print(f"refusing to overwrite {TARGET}; use --check to compare")
        return 1
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET} ({len(payload['entries'])} pairs; content sha256 {payload['content_sha256']}; source run {payload['source']['run_id']}, results state {payload['source']['results_state_sha256'][:16]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
