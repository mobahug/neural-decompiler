"""Item 8 (extra): the four results in the final state are exactly those written at completion. The runner logged the
state digest of the write that recorded the four results and the completed phase ("results sha256 2cba1ece…" in
confirm.out); after it, _descriptives only added descriptives.subsets and descriptives.comparators. Removing exactly
those two keys from the final state must reproduce that digest. No model."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirmreview023")
import guard  # noqa: E402  FIRST

import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(guard.ROOT)
EV = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm023")
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail != "" else ""), flush=True)
    if not ok:
        FAIL.append(name)


def cj(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


state = json.loads((ROOT / "outputs/experiment-023/results.json").read_text(encoding="utf-8"))
state.pop("state_sha256")
log = (EV / "confirm.out").read_text(encoding="utf-8")
logged = re.search(r"confirm complete \(four conditions, no aggregate label\): (.*); results sha256 ([0-9a-f]{64})", log)
at_completion = copy.deepcopy(state)
removed = [k for k in ("subsets", "comparators") if at_completion["confirmation"]["descriptives"].pop(k, None) is not None]
digest = hashlib.sha256(cj(at_completion).encode("utf-8")).hexdigest()
check("final state minus descriptives.{subsets, comparators} reproduces the digest logged at completion (2cba1ece…)", logged is not None and removed == ["subsets", "comparators"]
      and digest == logged.group(2), f"{digest[:16]} vs logged {logged.group(2)[:16] if logged else None}")
check("the logged results line == the state's labels and g (repr)", {k: (v, float(g)) for k, v, g in re.findall(r"(Y[12]/\w+) (\w+) \(g ([0-9.eE+-]+)\)", logged.group(1))}
      == {k: (e["result"], e["g"]) for k, e in state["confirmation"]["conditions"].items()})
check("log: exit 0, barrier message present, 2592 + 432 target pairs, no INCIDENT", "exit 0 end 2026-09-24T17:49:23Z" in log and "barrier passed" in log and "Y1: 2592 target pairs measured" in log
      and "Y2: 432 target pairs measured" in log and "INCIDENT" not in log)
print("\nEXTRA:", "ALL PASS" if not FAIL else f"FAILED {FAIL}")
