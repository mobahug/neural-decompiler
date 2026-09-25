"""Item 14 and the closing checks: cross-experiment ledger isolation, 020's runtime, the evidence inventory with hashes,
and the final repository / outputs state (read-only)."""
import rguard  # noqa: F401  (first)
from rguard import REPO

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(REPO)
SP = Path(rguard.SCRATCH).parent
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail != '' else ''}", flush=True)


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True, text=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --- every executed ledger on this machine vs the 024 manifest --------------------------------------------------------
freeze = json.loads((ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json").read_text())
keys = set(freeze["manifest"]["S2-TARGET"])
pairs = {(k.split("|")[0], k.split("|")[-1]) for k in keys}
report = {}
for d in sorted((ROOT / "outputs").iterdir()):
    p = d / "results.json"
    if d.name == "experiment-024" or not p.exists():
        continue
    try:
        s = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        report[d.name] = f"unreadable: {e}"
        continue
    ledger = s.get("executed_prompt_keys")
    if not isinstance(ledger, list):
        report[d.name] = "no executed_prompt_keys"
        continue
    hits = keys & set(ledger)
    phits = pairs & {(k.split("|")[0], k.split("|")[-1]) for k in ledger if isinstance(k, str) and "|" in k}
    report[d.name] = {"ledger": len(ledger), "key_hits": len(hits), "frame_token_hits": len(phits)}
for k, v in report.items():
    print(f"   {k}: {v}")
check("no 024 manifest key (nor any (frame, token id) pair) in any executed ledger of experiments 005–023 on this machine",
      all(isinstance(v, str) or (v["key_hits"] == 0 and v["frame_token_hits"] == 0) for v in report.values()))

r020 = json.loads((ROOT / "outputs/experiment-020/results.json").read_text())
rt = r020["phases"]["explore"]["runtime"]
check("020's explore runtime (the confirm runtime check's reference) has torch_num_threads 4", rt.get("torch_num_threads") == 4, rt)

# --- evidence inventory with hashes -----------------------------------------------------------------------------------
for folder in ("confirm024", "postinstall_review024", "confirmation_review024"):
    print(f"   --- {folder}/")
    for p in sorted((SP / folder).iterdir()):
        if p.is_file():
            print(f"   {sha(p)}  {p.stat().st_size:>9}  {folder}/{p.name}")

# --- the final repository and outputs state ----------------------------------------------------------------------------
HEAD = "af160cee0389bcdaf12bcbd92a8616f142777a52"
check("git status --porcelain --untracked-files=all is empty", git("status", "--porcelain", "--untracked-files=all") == "")
check("HEAD == origin/main == ls-remote == af160ce (unchanged)", git("rev-parse", "HEAD") == git("rev-parse", "origin/main") == git("ls-remote", "origin", "refs/heads/main").split()[0] == HEAD)
expected = {"calibration-arrays.pt": "a4cd548fd9c6084d58224f072d6c529cf4e81f17650fb50ccbc1d2c97f20aece",
            "candidate-calibration.json": "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d",
            "candidate-lock.json": "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d",
            "candidate-preregistration.md": "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54",
            "results.json": "e636210c84a4f3f329e9a5fde5baa493d730b6bf77cb419273056e6c89085a7a",
            "stage2-measurements.pt": "b21babe189d1cccb9ae359cfabf74baaa3a8337cac04612bc0d4aaa232b4dc93"}
out = ROOT / "outputs/experiment-024"
actual = {p.name: sha(p) for p in sorted(out.iterdir())}
for n, h in actual.items():
    print(f"   {h}  outputs/experiment-024/{n}")
check("outputs/experiment-024 holds exactly the six files, each sha256 unchanged", actual == expected, {k: v[:8] for k, v in actual.items()})
mt = {p.name: p.stat().st_mtime for p in out.iterdir()}
check("results.json and stage2-measurements.pt mtimes unchanged since confirm (19:38:44.949 / 19:35:01.288 UTC)",
      abs(mt["results.json"] - 1790365124.949506147) < 1e-3 and abs(mt["stage2-measurements.pt"] - 1790364901.288494707) < 1e-3)
print(f"S14 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
