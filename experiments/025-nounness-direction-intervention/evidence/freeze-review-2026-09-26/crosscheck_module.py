"""Cross-check (after the independent derivation) against the module and the runner, in memory only.

Calls, under guards: Runner()._base() (for `forbidden`), cr.blocked_sets, cr.select_cues, cr.freeze_payload (in memory;
never the runner's freeze), cr.confirmation_from_payload(verify=True) and cr.load_confirmation_025 on the committed-to-be
file. Records every neural_decompiler function called and every file opened during _base() and during freeze_payload.
Nothing is written under the repository (an audit hook refuses it). No model, no weight file, no forward, no score.
"""
import os
import sys

ROOT = "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler"
HERE = os.path.dirname(os.path.abspath(__file__))
OPENED = {"_base": [], "freeze_payload": [], "other": []}
PHASE = ["other"]
REFUSED = []


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        path = os.path.realpath(os.fsdecode(args[0]))
    except Exception:
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (
        isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    bad = ("/outputs/experiment-023/" in path or "/outputs/experiment-024/" in path or path.endswith("calibration-table.pt")
           or path.endswith(".safetensors") or path.endswith("pytorch_model.bin") or path.endswith(".ckpt"))
    if bad:
        REFUSED.append(path)
        raise PermissionError(f"reviewer guard: may not open {path}")
    if writing and path.startswith(ROOT + "/") and "/.venv/" not in path:
        REFUSED.append("WRITE " + path)
        raise PermissionError(f"reviewer guard: may not write {path}")
    if "/.venv/" not in path and not path.endswith(".py") and not path.endswith(".pyc"):
        OPENED[PHASE[0]].append(path)


sys.addaudithook(_hook)

import collections  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

GUARD_HITS = []


def _named(name):
    def refuse(*a, **k):
        GUARD_HITS.append(name)
        raise RuntimeError(f"reviewer guard: {name} refused")
    return refuse


torch.nn.Module.__call__ = _named("torch.nn.Module.__call__")
models.load_model = _named("models.load_model")
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, _named(f"pm.{name}"))
LAUNCHER_RR = ("nounness", "full_scores", "calibration_scores", "centroid", "centroids", "score_bindings")
LAUNCHER_CR = ("score", "direction", "centroids", "cue_geometry", "cue_vectors", "geometry_block", "plurality_direction", "nearest_tokens", "control_directions",
               "measure_rotated", "stage_two", "per_cue", "statistics")
OUT = {"launcher_guard_names_exist": {"rr": {n: hasattr(rr, n) for n in LAUNCHER_RR}, "cr": {n: hasattr(cr, n) for n in LAUNCHER_CR}}}
# Additionally refuse every other geometry / measurement / scoring entry point of cr (broader than the launcher).
EXTRA_CR = ("unit", "angle", "theta_for", "rotate", "sha_uniforms", "sha_gaussians", "project_off", "plurality_tangent", "condition_directions", "condition_vectors",
            "geometry_checks", "routing_distance", "patch_path_check", "recompute_c", "lexicon_of", "vector_factors", "identity_gates", "level1_gates", "run_gates",
            "ladder", "descriptives", "count_positive", "changes_from_raw")
for module, names in ((rr, LAUNCHER_RR + ("cosine", "embedding_digest", "parameters_digest")), (cr, LAUNCHER_CR + EXTRA_CR)):
    for name in names:
        if hasattr(module, name):
            setattr(module, name, _named(f"{module.__name__.rsplit('.', 1)[1]}.{name}"))

spec = importlib.util.spec_from_file_location("experiment_025_runner_review", Path(ROOT) / "experiments/025-nounness-direction-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)

CALLS = {"_base": collections.Counter(), "freeze_payload": collections.Counter()}


def profiler(frame, event, arg):
    if event == "call":
        mod = frame.f_globals.get("__name__", "")
        if mod.startswith("neural_decompiler") or mod.startswith("experiment_025"):
            CALLS[PHASE[0]][f"{mod}.{frame.f_code.co_name}"] += 1


runner = runner_module.Runner()
OUT["runner_config_is_production"] = runner.config is cr.PRODUCTION
PHASE[0] = "_base"
sys.setprofile(profiler)
base = runner._base()
sys.setprofile(None)
PHASE[0] = "other"

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(models.PYTHIA_70M.model_id, revision=models.PYTHIA_70M.revision, local_files_only=True)
runner_tok = runner.tokenizer_loader(models.PYTHIA_70M)  # the runner's own loader (HF_HUB_OFFLINE=1 in this environment)

PHASE[0] = "freeze_payload"
sys.setprofile(profiler)
payload = cr.freeze_payload(tok, pool=base.inputs.pool, freeze_024=base.inherited["confirmation"], prior=base.prior, config=cr.PRODUCTION)
sys.setprofile(None)
PHASE[0] = "other"
payload_runner_tok = cr.freeze_payload(runner_tok, pool=base.inputs.pool, freeze_024=base.inherited["confirmation"], prior=base.prior, config=cr.PRODUCTION)

path = Path(ROOT) / cr.CONFIRMATION_RELATIVE_PATH
file_bytes = path.read_bytes()
file_payload = json.loads(file_bytes)
OUT["module_payload_bytes_equal_file"] = (pm.canonical_json(payload) + "\n").encode("utf-8") == file_bytes
OUT["module_payload_with_runner_tokenizer_equal_file"] = (pm.canonical_json(payload_runner_tok) + "\n").encode("utf-8") == file_bytes
OUT["module_payload_equal_file_object"] = payload == file_payload

blocked = cr.blocked_sets(base.inputs.pool, base.inherited["confirmation"], base.prior)
OUT["cr_blocked_equal_file"] = blocked == file_payload["blocked"]
sel = cr.select_cues(tok, {k: frozenset(v["ids"]) for k, v in blocked.items()}, cr.PRODUCTION)
OUT["cr_select_equal_file"] = sel["cues"] == file_payload["cues"] and sel["rejected"] == file_payload["rejected"] and sel["picks"] == file_payload["picks"]
confirmation = cr.confirmation_from_payload(file_payload, base.inputs.pool, cr.PRODUCTION)  # verify=True
OUT["confirmation_from_payload_verify_passes"] = True
loaded = cr.load_confirmation_025(path, base.inputs.pool, base.inherited["confirmation"], base.prior, cr.PRODUCTION)
OUT["load_confirmation_025_passes"] = True
OUT["counts"] = confirmation.counts()
OUT["manifest_keys_overlap_forbidden"] = len(confirmation.manifest_keys() & base.forbidden)
OUT["tagged_keys_overlap_forbidden"] = len(confirmation.tagged_keys() & base.forbidden)
cr.assert_ledger_isolated(sorted(confirmation.tagged_keys()), base.forbidden, "review")
OUT["cr_assert_ledger_isolated_passes"] = True
OUT["forbidden_size"] = len(base.forbidden)
OUT["forbidden_sha256"] = hashlib.sha256(pm.canonical_json(sorted(base.forbidden)).encode()).hexdigest()
OUT["patch_path_keys_equal_plan"] = list(cr.PATCH_PATH_SPENT_KEYS) == ["cardinal-009-1|an|271", "quantifier-009-1|least|1878", "coordinated-adjective-009-1|black|2806",
                                                                       "quantifier-new-2|he|344"]
OUT["patch_path_keys_in_forbidden"] = set(cr.PATCH_PATH_SPENT_KEYS) <= base.forbidden
OUT["patch_path_keys_in_020_ledger"] = set(cr.PATCH_PATH_SPENT_KEYS) <= set(base.inputs.closure["ledger"])
OUT["patch_path_keys_in_manifest"] = len(set(cr.PATCH_PATH_SPENT_KEYS) & (confirmation.tagged_keys() | confirmation.manifest_keys()))
OUT["production_to_json_equal_file_configuration"] = cr.PRODUCTION.to_json() == file_payload["configuration"]
OUT["cr_design_plan_equal_file"] = dict(cr.DESIGN) == file_payload["design"] and dict(cr.PLAN) == file_payload["plan"]
OUT["cr_freeze_rules_equal_file"] = dict(cr.FREEZE_RULES) == file_payload["rules"]
F024 = base.inherited["confirmation"]
OUT["cr_adjective_reserves_equal_024"] = list(cr.ADJECTIVE_RESERVES) == F024["reserves"]["N"]
OUT["cr_ordinary_reserves_equal_024"] = ["/".join(p) for p in cr.ORDINARY_RESERVES] == F024["reserves"]["ordinary"]
design = (Path(ROOT) / cr.DESIGN["path"]).read_text(encoding="utf-8")
mm = re.search(r"\*\*The new candidate list, frozen in alphabetical textual order\*\*[^\n]*\n\n((?:> [^\n]*\n)+)", design)
OUT["cr_new_list_equal_design"] = list(cr.NEW_NOUN_LIST) == " ".join(line[2:] for line in mm.group(1).splitlines()).split()
OUT["cr_regular_plural_is_plus_s_for_new_list"] = all(cr.regular_plural(w) == w + "s" for w in cr.NEW_NOUN_LIST)
OUT["base_digest_keys"] = len(base.digests)
OUT["prior"] = {"n_ids": len(base.prior["ids"]), "nouns": len(base.prior["nouns"]), "file_sha256": base.prior["file_sha256"]}
OUT["guard_hits"] = GUARD_HITS
OUT["refused"] = REFUSED
OUT["calls_freeze_payload"] = dict(sorted(CALLS["freeze_payload"].items()))
OUT["calls_base_modules"] = dict(sorted(collections.Counter(k.rsplit(".", 1)[0] for k in CALLS["_base"]).items()))
OUT["calls_base_functions"] = sorted(CALLS["_base"])
OUT["opened_base"] = sorted(set(OPENED["_base"]))
OUT["opened_freeze_payload"] = sorted(set(OPENED["freeze_payload"]))
(Path(HERE) / "crosscheck_module.out.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("written")
