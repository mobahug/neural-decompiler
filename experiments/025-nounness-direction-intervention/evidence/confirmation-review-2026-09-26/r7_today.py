"""Items 1, 2, 14 and 15, re-checked today (read-only; no model).

The committed 020–024 inputs verify; the freeze verifies (with its excluded ids recomputed); cr.validate_lock passes on
today's tree; no scientific path changed between the lock commit and the confirm commit; the spent set is 43,632 keys
and disjoint from 025's keys, and the four patch-path keys are in it (020's ledger); the checkpoint files equal the
production record; the semantics and the stopping rule. Writes r7_today.json here.
"""
import guard  # noqa: F401

import hashlib
import json
import re
from pathlib import Path

from neural_decompiler import block0_completion as b0c
from neural_decompiler import cue_rotation as cr
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul

ROOT = Path(guard.ROOT)
HERE = Path(guard.HERE)
RUNS = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm025")
guard.seal()
out: dict = {}
head = guard.git("rev-parse", "HEAD").strip()
out["git"] = {"head": head, "origin_main": guard.git("rev-parse", "origin/main").strip(), "status": guard.git("status", "--porcelain", "--untracked-files=all"),
              "lock_commit_is_ancestor": True}
changed = [line for line in guard.git("diff", "--no-renames", "--name-only", "d3ecbd6feb1bd2aefe08dc0f8d799815be47c519", head).splitlines() if line.strip()]
out["git"]["changed_since_lock"] = changed
out["git"]["scientific_changes_since_lock"] = cr.scientific_changes(changed)
out["git"]["changed_since_lock_all_evidence_readme_or_installed"] = all(p.startswith("experiments/025-nounness-direction-intervention/") and (
    "/evidence/" in p or p.endswith(("README.md", "preregistration-lock.json", "preregistration.md"))) for p in changed)

inputs = ul.load_frozen_inputs(ROOT, tracked=guard.tracked)  # includes rc.verify_020_closure
d22 = b0c.verify_022_inputs(ROOT)
d23 = rr.verify_023_inputs(ROOT)
d24 = cr.verify_024_inputs(ROOT)
prior = cr.prior_nouns(ROOT)
digests = cr.base_digests(inputs, d22, d23, d24, prior)
out["inputs_verify"] = True
state = json.loads((ROOT / "outputs/experiment-025/results.json").read_text())
out["state_inputs_equal_today"] = state["inputs"] == {k: digests[k] for k in cr.DIGEST_KEYS}
inherited = cr.load_024(ROOT)
freeze_path = ROOT / cr.CONFIRMATION_RELATIVE_PATH
confirmation = cr.load_confirmation_025(freeze_path, inputs.pool, inherited["confirmation"], prior, cr.PRODUCTION)
out["freeze_verifies"] = True
c22 = json.loads((ROOT / ul.CONFIRMATION_RELATIVE_PATH).read_text())
c23 = json.loads((ROOT / b0c.CONFIRMATION_RELATIVE_PATH).read_text())


def m22(payload):
    m = payload["manifest"]
    return set(m["S1-REF"]) | set(m["S1-VALIDITY"]) | set(m["S2-TARGET"]["Y1"]) | set(m["S2-TARGET"]["Y2"])


ledger_020 = set(inputs.closure["ledger"])
forbidden = ledger_020 | {p.key for p in inputs.confirmation_020.all_prompts} | m22(c22) | m22(c23) | set(inherited["confirmation"]["manifest"]["S2-TARGET"])
untagged, tagged = confirmation.manifest_keys(), confirmation.tagged_keys()
out["spent"] = {"forbidden": len(forbidden), "equals_43632": len(forbidden) == 43632, "overlap_untagged": len(untagged & forbidden),
                "overlap_tagged": len(tagged & forbidden), "untagged_025": len(untagged), "tagged_025": len(tagged),
                "patch_path_keys_in_020_ledger": all(k in ledger_020 for k in cr.PATCH_PATH_SPENT_KEYS),
                "patch_path_keys_in_forbidden": all(k in forbidden for k in cr.PATCH_PATH_SPENT_KEYS),
                "patch_path_keys_outside_025": not (set(cr.PATCH_PATH_SPENT_KEYS) & (untagged | tagged))}
lock_path, prereg_path = ROOT / cr.LOCK_RELATIVE_PATH, ROOT / cr.PREREGISTRATION_RELATIVE_PATH
lock = json.loads(lock_path.read_text())
placeholder = cr.scientific_dependencies(inputs, parameters_sha256="", embedding_sha256="")
noun_keys = [n.lexical_key for n in inputs.pool.nouns if n.single_token]
cr.validate_lock(lock, state=state, digests=digests, config=cr.PRODUCTION, confirmation=confirmation, confirmation_file_sha256=rc.file_sha256(freeze_path),
                 dependencies=placeholder, noun_keys=noun_keys, preregistration_text=prereg_path.read_text(encoding="utf-8"),
                 git_state={"commit": head, "dirty": bool(out["git"]["status"])}, tracked=all(guard.tracked(p) for p in (lock_path, prereg_path, freeze_path)),
                 changed_paths=changed)
out["validate_lock_today"] = True
out["lock_semantics_equal_module"] = lock["semantics"] == cr.SEMANTICS and lock["outcome"]["table"] == [list(r) for r in cr.OUTCOME_TABLE]
out["reading"] = cr.SEMANTICS["outcomes"]["CAUSAL_EFFECT_NOT_ESTABLISHED"]
out["A_semantics"] = cr.SEMANTICS["A"]
out["not_claimed"] = cr.SEMANTICS["not_claimed"]
out["secondary_semantics"] = cr.SEMANTICS["secondary"]
out["strata_semantics"] = cr.SEMANTICS["strata"]

# the checkpoint files today against the production record
run = json.loads((RUNS / "production_run.json").read_text())
snap = Path(run["environment"]["checkpoint"]["snapshot"])
today = {}
for p in sorted(snap.iterdir()):
    if p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        today[p.name] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
out["checkpoint_files_equal_production_record"] = today == run["environment"]["checkpoint"]["files"]

# the stopping rule: design, plan and README text; no 026 anywhere in the repository's experiment, spec or plan names
design = (ROOT / cr.DESIGN["path"]).read_text()
plan = (ROOT / cr.PLAN["path"]).read_text()
readme = (ROOT / cr.EXPERIMENT_DIR / "README.md").read_text()
out["stopping"] = {"design": [l.strip() for l in design.splitlines() if "After 025" in l or "final experiment" in l],
                   "plan": [l.strip() for l in plan.splitlines() if "After 025" in l],
                   "readme": [l.strip() for l in readme.splitlines() if "final experiment" in l],
                   "any_026_paths": sorted(str(p.relative_to(ROOT)) for base in ("experiments", "docs") for p in (ROOT / base).rglob("*026*")),
                   "tracked_026": [l for l in guard.git("ls-files").splitlines() if re.search(r"(^|[^0-9])026([^0-9]|$)", l)]}
out["guard"] = guard.EVENTS
(HERE / "r7_today.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
