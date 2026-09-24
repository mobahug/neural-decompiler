"""Item 1: protocol and repository state (read-only; no neural_decompiler import)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HEAD = "89536b004b2e4536e83185cdfec31df4e7206982"
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        fails.append(name)


def git(*args, text=True):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=text).stdout


def my_canonical(value):
    # independent re-implementation of the repository's canonical JSON contract
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(b):
    return hashlib.sha256(b).hexdigest()


check("HEAD", git("rev-parse", "HEAD").strip() == HEAD, git("rev-parse", "HEAD").strip())
check("origin/main (tracking ref)", git("rev-parse", "origin/main").strip() == HEAD)
remote = [line.split("\t") for line in git("ls-remote", "origin").splitlines()]
remote_main = [sha1 for sha1, ref in remote if ref == "refs/heads/main"]
check("origin main (ls-remote)", remote_main == [HEAD], str(remote_main))
porcelain = git("status", "--porcelain", "--untracked-files=all")
check("status porcelain == exactly the untracked freeze file", porcelain == "?? experiments/023-block0-completion/confirmation-v1.json\n", repr(porcelain))
check("confirmation file never committed on any ref", git("log", "--all", "--format=%H", "--", "experiments/023-block0-completion/confirmation-v1.json").strip() == "")

raw = (ROOT / "outputs/experiment-023/results.json").read_bytes()
state = json.loads(raw)
recorded = state.pop("state_sha256")
recomputed = sha(my_canonical(state).encode("utf-8"))
check("results.json state_sha256 recomputes (own canonical JSON)", recorded == recomputed, recomputed)
check("results.json bytes are canonical JSON + newline", raw == (my_canonical({**state, "state_sha256": recorded}) + "\n").encode("utf-8"))
print("      results.json file sha256", sha(raw))
phases = state["phases"]
ex = phases["extract"]
check("extract complete, no incidents / stop keys", ex.get("status") == "complete" and not any(k in ex for k in ("incidents", "stop", "incident")),
      json.dumps({k: v for k, v in ex.items() if k != "runtime"}))
check("extract started/completed exactly once (single start/complete record)", isinstance(ex.get("started_at"), str) and isinstance(ex.get("completed_at"), str))
check("calibrate/lock/confirm/report not_started", all(phases[p] == {"status": "not_started"} for p in ("calibrate", "lock", "confirm", "report")),
      json.dumps({p: phases[p] for p in ("calibrate", "lock", "confirm", "report")}))
check("phase set", sorted(phases) == sorted(["extract", "calibrate", "lock", "confirm", "report"]), str(sorted(phases)))
check("executed_prompt_keys empty", state["executed_prompt_keys"] == [])
check("executed_noun_keys empty", state["executed_noun_keys"] == [])
check("no calibration / lock / confirmation / confirmation_023 / report record", state["calibration"] == {} and state["lock"] is None and state["confirmation"] is None
      and state["confirmation_023"] is None and state["report"] is None)
check("no incident anywhere in the state text", b"incident" not in raw.lower(), "")
check("state experiment 023, not dirty, protocol commit = extract commit", state["experiment"] == "023" and state["git_dirty"] is False
      and state["protocol_code_commit"] == ex["commit"], state["protocol_code_commit"])

for rel, expected in (("experiments/023-block0-completion/exposed-cells.f64", "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4"),
                      ("experiments/023-block0-completion/exposed-cells.json", "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe")):
    disk = (ROOT / rel).read_bytes()
    head = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, check=True, capture_output=True).stdout
    check(f"{rel} sha256 == expected", sha(disk) == expected, sha(disk))
    check(f"{rel} == git show HEAD bytes", disk == head, f"{len(disk)} bytes")
index = json.loads((ROOT / "experiments/023-block0-completion/exposed-cells.json").read_text())
content = index.pop("content_sha256")
check("exposed-cells.json content digest recomputes (own canonical JSON)", content == sha(my_canonical(index).encode("utf-8")), content)
print("ITEM 1:", "ALL PASS" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
