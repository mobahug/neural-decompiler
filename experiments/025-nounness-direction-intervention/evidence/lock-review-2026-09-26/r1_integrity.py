"""Items 1, 2, 4 (file-level bindings), 17 and part of 20: repository/phase integrity, candidate integrity, dependency
hashes, the manifest and the spent-key isolation. No model is loaded; every forward and capture refuses."""
import guard  # noqa: F401  (the audit hook first)

import hashlib
import json
import math
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = guard.ROOT
HERE = guard.HERE
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler.provenance import collect_versions  # noqa: E402

guard.seal()
guard.refuse_model_load()
out: dict = {}


def canonical(value) -> str:  # my own canonical JSON (the repository's definition, written independently)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def content_digest(payload: dict) -> str:
    return sha256_text(canonical({k: v for k, v in payload.items() if k != "content_sha256"}))


def git(*args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def strict_load(text: str):
    def bad(name):
        raise ValueError(f"non-finite constant {name}")
    return json.loads(text, parse_constant=bad)


def walk_numbers(value, path="$"):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from walk_numbers(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk_numbers(v, f"{path}[{i}]")


# ---------------------------------------------------------------------------------------------------------------------
# Item 1: repository and phase integrity.
head = git("rev-parse", "HEAD")
origin = git("rev-parse", "origin/main")
porcelain = git("status", "--porcelain=v1", "--untracked-files=all")
try:
    remote = subprocess.run(["git", "--no-optional-locks", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, check=True, capture_output=True, text=True,
                            timeout=60).stdout.split()[0]
except Exception as error:  # network may be unavailable
    remote = f"unavailable: {error}"
freeze_path = ROOT / cr.CONFIRMATION_RELATIVE_PATH
freeze_bytes = freeze_path.read_bytes()
freeze_blob_head = git("rev-parse", f"HEAD:{cr.CONFIRMATION_RELATIVE_PATH}")
freeze_blob_install = git("rev-parse", f"57f1f98:{cr.CONFIRMATION_RELATIVE_PATH}")
freeze_diff = git("diff", "--name-only", "57f1f98", "HEAD", "--", cr.CONFIRMATION_RELATIVE_PATH)
lock_history = git("log", "--all", "--oneline", "--", cr.LOCK_RELATIVE_PATH, cr.PREREGISTRATION_RELATIVE_PATH)
outputs_dir = ROOT / "outputs/experiment-025"
listing = sorted(p.name for p in outputs_dir.iterdir())
state_text = (outputs_dir / "results.json").read_text(encoding="utf-8")
state = strict_load(state_text)
recorded_state_sha = state.pop("state_sha256")
state_sha_mine = sha256_text(canonical(state))
state_file_canonical = state_text == canonical({**state, "state_sha256": recorded_state_sha}) + "\n"
state_loaded = rd.load_results_state(outputs_dir / "results.json")
phase_rules = {}
for phase in ("lock", "confirm", "report"):
    try:
        cr.assert_phase_allowed(phase, state_loaded)
        phase_rules[phase] = "allowed by the phase rules"
    except cr.PhaseError as error:
        phase_rules[phase] = f"refused: {error}"
incidents = {name: entry.get("incidents") for name, entry in state["phases"].items()}
out["item1"] = {
    "head": head, "origin_main": origin, "remote_main": remote, "all_equal_d3ecbd6": head == origin == remote == "d3ecbd6feb1bd2aefe08dc0f8d799815be47c519",
    "tree_porcelain_empty": porcelain == "", "freeze_file_sha256": sha256_bytes(freeze_bytes), "freeze_blob_head": freeze_blob_head,
    "freeze_blob_at_install_57f1f98": freeze_blob_install, "freeze_blob_mine": blob_sha1(freeze_bytes), "freeze_changed_since_install": freeze_diff,
    "lock_or_prereg_ever_committed": lock_history, "lock_installed_now": (ROOT / cr.LOCK_RELATIVE_PATH).exists() or (ROOT / cr.PREREGISTRATION_RELATIVE_PATH).exists(),
    "outputs_025_listing_incl_hidden": listing, "state_sha256_recorded": recorded_state_sha, "state_sha256_mine": state_sha_mine,
    "state_file_is_canonical_json": state_file_canonical, "phases": state["phases"], "executed_prompt_keys": len(state["executed_prompt_keys"]),
    "executed_noun_keys": len(state["executed_noun_keys"]), "confirmation": state["confirmation"], "report": state["report"], "incidents": incidents,
    "phase_rules_on_current_state": phase_rules, "state_lock_block": state["lock"], "state_run_id": state["run_id"], "state_created_at": state["created_at"],
}

# ---------------------------------------------------------------------------------------------------------------------
# Item 2: candidate integrity.
lock_path = outputs_dir / "candidate-lock.json"
prereg_path = outputs_dir / "candidate-preregistration.md"
lock_bytes = lock_path.read_bytes()
lock_text = lock_bytes.decode("utf-8")
lock = strict_load(lock_text)
prereg_bytes = prereg_path.read_bytes()
prereg_text = prereg_bytes.decode("utf-8")
numbers = list(walk_numbers(lock))
non_finite = [p for p, v in numbers if isinstance(v, float) and not math.isfinite(v)]
rendered = cr.render_preregistration(json.loads(lock_text))
out["item2"] = {
    "lock_file_sha256": sha256_bytes(lock_bytes), "lock_file_expected": "1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be",
    "lock_content_sha256_recorded": lock["content_sha256"], "lock_content_sha256_mine": content_digest(lock), "lock_content_sha256_rc": rc.content_digest(lock),
    "lock_bytes_equal_my_canonical_plus_newline": lock_text == canonical(lock) + "\n", "lock_bytes_equal_pm_canonical_plus_newline": lock_text == pm.canonical_json(lock) + "\n",
    "my_canonical_equals_pm_canonical": canonical(lock) == pm.canonical_json(lock), "numbers_in_lock": len(numbers), "non_finite": non_finite,
    "prereg_file_sha256": sha256_bytes(prereg_bytes), "prereg_expected": "e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919",
    "prereg_equals_render_bytewise": prereg_bytes == rendered.encode("utf-8"), "prereg_sha_in_state": state["lock"]["preregistration_sha256"],
    "state_lock_content_matches": state["lock"]["content_sha256"] == lock["content_sha256"], "lock_run_id_equals_state": lock["run_id"] == state["run_id"],
    "lock_commit": lock["protocol_code_commit"], "state_lock_phase_commit": state["phases"]["lock"]["commit"],
}

# Header numbers of the preregistration, checked against the lock and against independent values.
tail = Fraction(sum(math.comb(40, k) for k in range(27, 41)), 2 ** 40)
tail26 = Fraction(sum(math.comb(40, k) for k in range(26, 41)), 2 ** 40)
threshold = next(k for k in range(41) if Fraction(sum(math.comb(40, j) for j in range(k, 41)), 2 ** 40) <= Fraction(1, 40))
header_expect = {
    "lock digest": f"Lock `{lock['content_sha256']}`", "run": f"run `{lock['run_id']}` at `{head}`",
    "design": "design revision 1 (`c0885e5`, corrected `26c9925`), plan revision 1 (`7d90d28`)",
    "module blob": f"at blob `{blob_sha1((ROOT / 'src/neural_decompiler/cue_rotation.py').read_bytes())}`; configuration `production`",
    "freeze": f"The freeze `{content_digest(json.loads(freeze_bytes))}`", "manifest": f"manifest `{sha256_text(canonical(json.loads(freeze_bytes)['manifest']))}`",
    "runs": "(90720 condition-tagged runs: 40 cues × 108 frames × 21 conditions)", "odd": "exactly ±0.32 (primary) and ±0.16 (secondary)",
    "controls": "7 nounness-neutral random tangent controls per cue (tag `025|control|{token_id}|{j}`)",
    "tail": f"P(X ≥ 27) = {tail.numerator}/{tail.denominator} = {float(tail)!r}", "count": "PASS iff at least 27 of 40 values are strictly positive",
    "patch keys": "cardinal-009-1|an|271, quantifier-009-1|least|1878, coordinated-adjective-009-1|black|2806, quantifier-new-2|he|344",
    "tolerances": "I1, I3, I4 on every run (1e-04, 1e-04 relative, 1e-03); C recomputed bit for bit; the Level-1 identity ≤ 0.02 on every outcome-bearing",
}
out["item2"]["header_numbers_present"] = {name: (text in prereg_text) for name, text in header_expect.items()}
out["item2"]["binomial"] = {"tail27_exact": f"{tail.numerator}/{tail.denominator}", "tail27": float(tail), "tail27_matches_design": float(tail) == 0.01923865414210013,
                            "tail26": float(tail26), "tail26_exceeds_0.025": tail26 > Fraction(1, 40), "derived_threshold": threshold,
                            "lock_reference_tail": lock["configuration"]["reference_tail"]}
dnorm_line = re.search(r"\|d\| ([0-9.]+)\); cos\(d̂, p̂\) = ([0-9.+-]+)", prereg_text)
out["item2"]["prereg_d_line"] = {"d_norm": dnorm_line.group(1), "lock_d_norm_repr": repr(lock["geometry"]["d_norm"]), "cos": dnorm_line.group(2),
                                 "lock_cos_repr": repr(lock["geometry"]["cos_d_plurality"])}
maxima_line = [line for line in prereg_text.splitlines() if line.startswith("- Geometry check maxima: ")][0]
mine_maxima = "- Geometry check maxima: " + ", ".join(f"{name} {value:.3e}" for name, value in sorted(lock["geometry"]["check_maxima"].items()))
out["item2"]["prereg_maxima_line_matches_lock"] = maxima_line == mine_maxima
tol_line = [line for line in prereg_text.splitlines() if line.startswith("- I7′:")][0]
out["item2"]["prereg_tolerance_line"] = tol_line
out["item2"]["prereg_cue_rows"] = sum(1 for line in prereg_text.splitlines() if line.startswith("| adjective |") or line.startswith("| noun |"))

# ---------------------------------------------------------------------------------------------------------------------
# Item 4 (file level): every committed dependency, hashed by my own code.
base = None
from importlib import util as _util  # noqa: E402

spec = _util.spec_from_file_location("experiment_025_runner", ROOT / "experiments/025-nounness-direction-intervention/run.py")
runner_module = _util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
guard.sweep()  # the runner's aliases too
guard.refuse_model_load()
runner_logs: list[str] = []
runner = runner_module.Runner(log=runner_logs.append, model_loader=guard._refusal("runner.model_loader"),
                              tokenizer_loader=guard._refusal("runner.tokenizer_loader"))
validate_status = runner.validate()
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)

blobs = {}
for name, module in cr._FROZEN_MODULES.items():
    path = Path(module.__file__).resolve()
    data = path.read_bytes()
    blobs[name] = {"path_in_repo": str(path).startswith(str(ROOT / "src/neural_decompiler")), "mine": blob_sha1(data),
                   "git_head": git("rev-parse", f"HEAD:src/neural_decompiler/{name}"), "pinned": cr.FROZEN_BLOBS[name], "lock": lock["module_blobs"][name],
                   "lock_dependencies": lock["dependencies"]["module_blobs"][name], "state": state["module_blobs"][name]}
    blobs[name]["all_equal"] = len({blobs[name][k] for k in ("mine", "git_head", "pinned", "lock", "lock_dependencies", "state")}) == 1
cr_path = Path(cr.__file__).resolve()
cr_data = cr_path.read_bytes()
own = {"imported_file": str(cr_path), "is_repo_file": cr_path == (ROOT / "src/neural_decompiler/cue_rotation.py").resolve(), "mine": blob_sha1(cr_data),
       "git_head": git("rev-parse", "HEAD:src/neural_decompiler/cue_rotation.py"), "cr_own_blob": cr.own_blob(), "lock_module": lock["module"],
       "unchanged_since_lock_commit": git("diff", "--name-only", lock["protocol_code_commit"], "HEAD", "--", "src/neural_decompiler/cue_rotation.py") == ""}
own["all_equal"] = len({own["mine"], own["git_head"], own["cr_own_blob"], lock["module"]["blob"]}) == 1
run_py = (ROOT / cr.EXPERIMENT_DIR / "run.py").read_bytes()
inherited = {}
for kind, relative in cr.INHERITED_024_PATHS.items():
    data = (ROOT / relative).read_bytes()
    payload = strict_load(data.decode("utf-8"))
    inherited[kind] = {"path": relative, "file_mine": sha256_bytes(data), "content_mine": content_digest(payload), "content_recorded": payload["content_sha256"],
                       "lock_binding": lock["inherited_024"][kind], "lock_inputs_file": lock["inputs"][f"024_{kind}_file"],
                       "lock_inputs_content": lock["inputs"][f"024_{kind}_content"]}
    inherited[kind]["all_equal"] = (inherited[kind]["file_mine"] == lock["inherited_024"][kind]["file_sha256"] == lock["inputs"][f"024_{kind}_file"]
                                    and inherited[kind]["content_mine"] == inherited[kind]["content_recorded"] == lock["inherited_024"][kind]["content_sha256"]
                                    == lock["inputs"][f"024_{kind}_content"] and relative == lock["inherited_024"][kind]["path"])
prior_data = (ROOT / cr.PRIOR_NOUNS_020["path"]).read_bytes()
prior_payload = strict_load(prior_data.decode("utf-8"))
prior_ids = sorted({int(t) for noun in prior_payload["nouns"] for t in (*noun["sg_ids"], *noun["pl_ids"])})
freeze_payload = strict_load(freeze_bytes.decode("utf-8"))
out["item4_files"] = {
    "freeze": {"file_mine": sha256_bytes(freeze_bytes), "content_mine": content_digest(freeze_payload), "content_recorded": freeze_payload["content_sha256"],
               "manifest_mine": sha256_text(canonical(freeze_payload["manifest"])), "manifest_recorded": freeze_payload["manifest_sha256"],
               "manifest_keys": len(freeze_payload["manifest"]["S2-TARGET"]), "lock_binding": lock["confirmation_025"], "runner_confirmation_sha": confirmation_sha},
    "module_blobs": blobs, "module_blobs_all_equal": all(entry["all_equal"] for entry in blobs.values()), "cue_rotation_own": own,
    "run_py_blob_mine": blob_sha1(run_py), "run_py_blob_head": git("rev-parse", f"HEAD:{cr.EXPERIMENT_DIR}/run.py"),
    "inherited_024": inherited, "prior_nouns_020": {"file_mine": sha256_bytes(prior_data), "pinned": cr.PRIOR_NOUNS_020["file_sha256"],
                                                     "lock_inputs": lock["inputs"]["020_prior_nouns_file"], "nouns": len(prior_payload["nouns"]),
                                                     "ids": len(prior_ids), "ids_sha256_mine": sha256_text(canonical(prior_ids)),
                                                     "freeze_blocked_prior_sha256": freeze_payload["blocked"]["prior_nouns"]["sha256"]},
    "lock_inputs_equal_base_digests": lock["inputs"] == base.digests, "state_inputs_equal_base_digests": state["inputs"] == {k: base.digests[k] for k in cr.DIGEST_KEYS},
    "lock_dependency_frozen_input_digests_equal": lock["dependencies"]["readout_020"]["frozen_input_digests"] == {k: base.inputs.digests[k] for k in rc.DIGEST_KEYS},
    "lock_dependencies_C_measurement": {"C_equals_rr": lock["dependencies"]["C"] == rr.C_DEFINITION, "measurement_equals_cr": lock["dependencies"]["measurement"] == cr.MEASUREMENT},
    "validate_status": validate_status, "validate_log": runner_logs,
}

# The results-state reconstruction: remove the lock-phase fields; compare with a fresh cr.new_results_state.
fresh = cr.new_results_state(digests=base.digests, protocol_code_commit=head, git_dirty=False, versions=collect_versions(), config=cr.PRODUCTION)
stripped = json.loads(json.dumps(state))
stripped["confirmation_025"] = None
stripped["lock"] = None
stripped["phases"]["lock"] = {"status": "not_started"}
differing = sorted(k for k in set(fresh) | set(stripped) if k not in ("run_id", "created_at") and fresh.get(k) != stripped.get(k))
out["item1"]["fresh_state_reconstruction_differs_in"] = differing
out["item1"]["versions_now_equal_state"] = collect_versions() == state["versions"]

# ---------------------------------------------------------------------------------------------------------------------
# Item 17: the manifest rebuilt from the freeze, and the spent set recomputed from its sources.
cues = [(c["word"], int(c["token_id"])) for c in freeze_payload["cues"]]
frames = freeze_payload["exposed_frame_ids"]
conditions = ["base", "noun+0.32", "noun-0.32", "noun+0.16", "noun-0.16"] + [f"rand{j}{s}0.32" for j in range(1, 8) for s in "+-"] + ["plur+0.32", "plur-0.32"]
bearing = ["noun+0.32", "noun-0.32"] + [f"rand{j}{s}0.32" for j in range(1, 8) for s in "+-"]
keys = sorted(f"{frame}|{word}|{tid}|{cond}" for frame in frames for word, tid in cues for cond in conditions)
untagged = {f"{frame}|{word}|{tid}" for frame in frames for word, tid in cues}
results020 = strict_load((ROOT / rc.RESULTS_020_RELATIVE_PATH).read_text(encoding="utf-8"))
ledger020 = set(results020["executed_prompt_keys"])
m20 = strict_load((ROOT / rd.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))["manifest"]
spent_020_conf = set(m20["S1-REF"]) | set(m20["S1-VALIDITY"]) | set(m20["S2-TARGET"])


def manifest_022_like(relative):
    m = strict_load((ROOT / relative).read_text(encoding="utf-8"))["manifest"]
    return set(m["S1-REF"]) | set(m["S1-VALIDITY"]) | set(m["S2-TARGET"]["Y1"]) | set(m["S2-TARGET"]["Y2"])


spent_022 = manifest_022_like(ul.CONFIRMATION_RELATIVE_PATH)
spent_023 = manifest_022_like(b0c.CONFIRMATION_RELATIVE_PATH)
spent_024 = set(strict_load((ROOT / rr.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))["manifest"]["S2-TARGET"])
spent = ledger020 | spent_020_conf | spent_022 | spent_023 | spent_024
patch = set(cr.PATCH_PATH_SPENT_KEYS)
runner_020_conf = {p.key for p in base.inputs.confirmation_020.all_prompts}
out["item17"] = {
    "keys": len(keys), "unique": len(set(keys)), "equals_freeze_manifest_in_order": keys == freeze_payload["manifest"]["S2-TARGET"],
    "manifest_sha256_mine": sha256_text(canonical({"S2-TARGET": keys})), "conditions": len(conditions), "conditions_equal_freeze": conditions == freeze_payload["conditions"],
    "outcome_bearing": len(bearing), "outcome_bearing_equal_config": bearing == list(cr.PRODUCTION.outcome_bearing),
    "tagged_keys_equal_cr": set(keys) == set(confirmation.tagged_keys()), "untagged": len(untagged),
    "spent_sizes": {"020_ledger": len(ledger020), "020_confirmation": len(spent_020_conf), "022": len(spent_022), "023": len(spent_023), "024": len(spent_024), "union": len(spent)},
    "spent_equals_runner_forbidden": spent == set(base.forbidden), "020_conf_equals_runner_all_prompts": spent_020_conf == runner_020_conf,
    "collisions_untagged_with_spent": len(untagged & spent), "collisions_tagged_with_spent": len(set(keys) & spent),
    "collisions_untagged_with_patch_keys": len(untagged & patch), "collisions_tagged_untagged_form_with_spent": len({cr.untagged(k) for k in keys} & spent),
    "patch_keys_in_020_ledger": {k: k in ledger020 for k in cr.PATCH_PATH_SPENT_KEYS}, "patch_keys_in_spent": all(k in spent for k in patch),
    "patch_key_frames_in_exposed_pool": {k: k.split("|")[0] in frames for k in cr.PATCH_PATH_SPENT_KEYS},
    "patch_keys_in_manifest": len(patch & set(keys)), "ledger_empty": state["executed_prompt_keys"] == [],
}

# ---------------------------------------------------------------------------------------------------------------------
out["guard"] = guard.summary()
(HERE / "r1_integrity.out.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False)[:20000])
