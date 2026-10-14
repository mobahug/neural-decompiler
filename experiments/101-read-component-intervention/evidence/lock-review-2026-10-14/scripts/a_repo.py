"""Item 1 (and the evidence snapshot), independent: git state, the installed freeze and calibration, the outputs, the
results state, the lock evidence files. No neural_decompiler import; read-only git (--no-optional-locks)."""
from common import install_fs_guard

install_fs_guard()

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from common import EVENTS, ROOT, canonical, content_digest, file_sha256, git, git_bytes, git_ok, sha256_text, write_json  # noqa: E402

EXP = "experiments/101-read-component-intervention"
LOCK_DIR = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/lock")
STAGE1_LAUNCHER = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/stage1/launch_stage1.py")
out: dict = {}

# --- git
out["head"] = git("rev-parse", "HEAD").strip()
out["branch"] = git("rev-parse", "--abbrev-ref", "HEAD").strip()
out["head_is_expected"] = out["head"] == "3621eec4b6a901119a4b96d4fdc07fa3b19652ab"
out["porcelain_untracked_all"] = git("status", "--porcelain=v1", "--untracked-files=all")
out["tree_clean"] = out["porcelain_untracked_all"] == ""
out["origin_main"] = git("rev-parse", "origin/main").strip()
out["ahead_of_origin_main"] = int(git("rev-list", "--count", "origin/main..HEAD").strip())
out["log"] = git("log", "--format=%H %s", "-12").splitlines()
out["index_mtime"] = os.stat(ROOT / ".git/index").st_mtime

# --- the installed freeze and calibration
installed = {}
for name, commit in (("confirmation-v1.json", "a857f9a2326ff61cae8485deef2ab9337f5d6e02"), ("calibration-v1.json", "d502777c869eb7345dc05af0f5bc8445877e9b27")):
    rel = f"{EXP}/{name}"
    data = (ROOT / rel).read_bytes()
    head_blob = git_bytes("show", f"HEAD:{rel}")
    commits = git("log", "--format=%H", "--", rel).split()
    changed_in_commit = git("diff-tree", "--no-commit-id", "--name-status", "-r", commit).splitlines()
    payload = json.loads(data)
    installed[name] = {"file_sha256": hashlib.sha256(data).hexdigest(), "head_blob_sha256": hashlib.sha256(head_blob).hexdigest(),
                       "working_equals_head": data == head_blob, "commits_touching": commits, "install_commit_changes": changed_in_commit,
                       "committed_alone": changed_in_commit == [f"A\t{rel}"], "content_sha256_recorded": payload.get("content_sha256"),
                       "content_sha256_recomputed": content_digest(payload), "file_is_canonical_plus_newline": data.decode("utf-8") == canonical(payload) + "\n"}
cand_cal = (ROOT / "outputs/experiment-101/candidate-calibration-v1.json").read_bytes()
installed["calibration_equals_candidate_bytes"] = cand_cal == (ROOT / f"{EXP}/calibration-v1.json").read_bytes()
out["installed"] = installed

# --- what changed since the calibrate commit (the lock refuses scientific changes since then)
out["changed_since_calibrate_8f65062"] = git("diff", "--no-renames", "--name-only", "8f65062627c77815adf9289c3fe60191d9fc3c17", "HEAD").split()
out["changed_since_calibrate_outside_evidence"] = [p for p in out["changed_since_calibrate_8f65062"] if not p.startswith(f"{EXP}/evidence/")]
out["changed_since_impl_40829d2_in_src_tests_run"] = git("diff", "--name-only", "40829d2", "HEAD", "--", "src", "tests", f"{EXP}/run.py").split()

# --- the design and the plan at their decision commits
for key, path, decision in (("design", "docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md", "516df20"),
                            ("plan", "docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md", "1e48c1c")):
    out[key] = {"commits_touching": git("log", "--format=%h %s", "--", path).splitlines(),
                "unchanged_since_decision": git("diff", "--name-only", decision, "HEAD", "--", path).strip() == ""}
out["design_a87e284_is_revision3"] = "revision 3" in git("log", "-1", "--format=%s", "a87e284")
out["plan_96f0cc8_is_revision3"] = "revision 3" in git("log", "-1", "--format=%s", "96f0cc8")

# --- module blobs at HEAD and at 516df20 (git ls-tree), and read_intervention.py
def ls_tree(rev):
    rows = {}
    for line in git("ls-tree", rev, "src/neural_decompiler/").splitlines():
        meta, path = line.split("\t")
        rows[path.rsplit("/", 1)[1]] = meta.split()[2]
    return rows
out["ls_tree_head"] = ls_tree("HEAD")
out["ls_tree_516df20"] = ls_tree("516df20")
out["read_intervention_blob_head"] = out["ls_tree_head"]["read_intervention.py"]

# --- outputs/experiment-101 (listing via os.scandir: names only, then hashing the 101 files, which this review may read)
listing = sorted(entry.name for entry in os.scandir(ROOT / "outputs/experiment-101"))
out["outputs_101_listing"] = listing
out["outputs_101_sha256"] = {name: file_sha256(ROOT / "outputs/experiment-101" / name) for name in listing if (ROOT / "outputs/experiment-101" / name).is_file()}
out["outputs_101_ignored_status"] = git("status", "--porcelain=v1", "--ignored", "--untracked-files=all", "--", "outputs/experiment-101").splitlines()

# --- the results state
raw = (ROOT / "outputs/experiment-101/results.json").read_bytes()
state = json.loads(raw)
recorded = state.pop("state_sha256")
state_check = {"file_sha256": hashlib.sha256(raw).hexdigest(), "state_sha256_recorded": recorded, "state_sha256_recomputed": sha256_text(canonical(state)),
               "file_is_canonical_plus_newline": raw.decode("utf-8") == canonical({**state, "state_sha256": recorded}) + "\n",
               "run_id": state["run_id"], "phases": state["phases"], "executed_prompt_keys": state["executed_prompt_keys"],
               "calibration_executed_count": len(state["calibration_executed"]), "lock_block": state["lock"], "stage1": state["stage1"],
               "confirmation": state["confirmation"], "report": state["report"], "protocol_code_commit": state["protocol_code_commit"],
               "configuration_name": state["configuration"]["name"]}


def find_incidents(value, path=""):
    hits = []
    if isinstance(value, dict):
        for key, item in value.items():
            if "incident" in key.lower() or key in ("stopped", "partial"):
                hits.append(f"{path}/{key}")
            hits += find_incidents(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value[:5]):
            hits += find_incidents(item, f"{path}[{index}]")
    return hits


state_check["incident_like_keys"] = find_incidents(state)
cand_lock = (ROOT / "outputs/experiment-101/candidate-lock.json").read_bytes()
cand_prereg = (ROOT / "outputs/experiment-101/candidate-preregistration.md").read_bytes()
lock = json.loads(cand_lock)
state_check["lock_block_matches_candidates"] = {"content_sha256": state["lock"]["content_sha256"] == lock["content_sha256"] == content_digest(lock),
                                                "preregistration_sha256": state["lock"]["preregistration_sha256"] == hashlib.sha256(cand_prereg).hexdigest()}
state_check["calibration_block"] = {key: state["calibration"].get(key) for key in ("content_sha256", "file_sha256", "tau", "candidate_path")}
state_check["calibration_block_has"] = sorted(state["calibration"])
state_check["confirmation_101_binding"] = state["confirmation_101"]
out["state"] = state_check

# --- the lock evidence (operator's records), snapshot of digests now
out["lock_evidence_sha256"] = {p.name: file_sha256(p) for p in sorted(LOCK_DIR.iterdir()) if p.is_file()}
out["stage1_launcher_sha256"] = file_sha256(STAGE1_LAUNCHER)
out["fs_events"] = EVENTS
path = write_json("a_repo.json", out)
print(json.dumps({k: out[k] for k in ("head", "branch", "head_is_expected", "tree_clean", "origin_main", "ahead_of_origin_main")}, indent=1))
print("installed:", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("commits_touching",)}) for k, v in installed.items()}, indent=1))
print("changed since 8f65062 outside evidence:", out["changed_since_calibrate_outside_evidence"])
print("changed since 40829d2 in src/tests/run.py:", out["changed_since_impl_40829d2_in_src_tests_run"])
print("design:", out["design"], "plan:", out["plan"], out["design_a87e284_is_revision3"], out["plan_96f0cc8_is_revision3"])
print("read_intervention blob at HEAD:", out["read_intervention_blob_head"])
print("outputs:", json.dumps(out["outputs_101_sha256"], indent=1))
print("state:", json.dumps({k: v for k, v in state_check.items() if k not in ("phases",)}, indent=1, default=str)[:4000])
print("phases:", json.dumps(state_check["phases"]))
print("fs events:", json.dumps(EVENTS))
print("wrote", path)
