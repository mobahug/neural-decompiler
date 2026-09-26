"""Guarded launcher for Experiment 025's production `freeze` (run exactly once).

Before the runner module is imported:
- `neural_decompiler.models.load_model` is replaced by a refusal: the freeze is tokenizer-only, and the model is never
  loaded, so no weight, logit, hidden state or attention can exist;
- `torch.nn.Module.__call__` refuses: no forward of any module;
- every capture and intervention entry point of `plural_mechanism` refuses;
- every score and geometry function refuses: 024's nounness score and centroids, and 025's direction, geometry,
  plurality direction and nearest tokens. The geometry belongs to the lock, and no score may select or reorder a cue;
- an audit hook refuses opening 022's local calibration table, or anything under `outputs/experiment-023/` or
  `outputs/experiment-024/`. It also refuses any write under the repository except the one confirmation file the
  freeze writes.

The pre-write verification (in memory, before the one freeze invocation) checks every point the reviewer listed:
- the 40 picks;
- single tokens with a leading space;
- distinct ids;
- each exclusion set, with statue and barrel under the prior-noun rule;
- the frame tokens;
- the 90,720 condition-tagged keys: unique, canonical, tagged, and colliding with no spent or patch-path key.

Only if every check passes does it call `runner.freeze()`, exactly once. Afterwards it verifies that the written file is
the pre-verified payload, byte for byte. Every refusal is counted. The counts, the checks, the start and end times
(UTC) and the exit status are written to this scratchpad directory only.
"""
import datetime
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
DRY_RUN = os.environ.get("FREEZE025_DRY_RUN") == "1"  # every guard and pre-write check, in memory; freeze is not called, nothing is written
CONFIRMATION = (ROOT / "experiments/025-nounness-direction-intervention/confirmation-v1.json").resolve()
events = {"module_calls": 0, "load_model": 0, "capture_calls": 0, "score_or_geometry_calls": [], "forbidden_reads": [], "forbidden_writes": [],
          "confirmation_writes": 0, "freeze_invocations": 0}


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        path = Path(os.fsdecode(args[0])).resolve()
    except OSError:
        return
    text = str(path)
    if text.endswith("calibration-table.pt") or "/outputs/experiment-023/" in text or "/outputs/experiment-024/" in text:
        events["forbidden_reads"].append(text)
        raise PermissionError(f"the freeze may not open {text}")
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if writing and text.startswith(str(ROOT)) and "/.venv/" not in text:
        if path == CONFIRMATION:
            events["confirmation_writes"] += 1
            return
        events["forbidden_writes"].append(text)
        raise PermissionError(f"the freeze may not write {text}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402


def _refuse_module_call(self, *args, **kwargs):
    events["module_calls"] += 1
    raise RuntimeError("a torch module was called during the freeze")


def _refuse_load(*args, **kwargs):
    events["load_model"] += 1
    raise RuntimeError("the model was loaded during the freeze")


def _refuse_capture(*args, **kwargs):
    events["capture_calls"] += 1
    raise RuntimeError("a capture or intervention entry point was reached during the freeze")


def _refuse_named(name):
    def refuse(*args, **kwargs):
        events["score_or_geometry_calls"].append(name)
        raise RuntimeError(f"{name} was called during the freeze (no score or geometry may exist at the freeze)")

    return refuse


torch.nn.Module.__call__ = _refuse_module_call
models.load_model = _refuse_load
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, _refuse_capture)
for module, names in ((rr, ("nounness", "full_scores", "calibration_scores", "centroid", "centroids", "score_bindings")),
                      (cr, ("score", "direction", "centroids", "cue_geometry", "cue_vectors", "geometry_block", "plurality_direction", "nearest_tokens",
                            "control_directions", "measure_rotated", "stage_two", "per_cue", "statistics"))):
    for name in names:
        if hasattr(module, name):
            setattr(module, name, _refuse_named(f"{module.__name__.rsplit('.', 1)[1]}.{name}"))

spec = importlib.util.spec_from_file_location("experiment_025_runner", ROOT / "experiments/025-nounness-direction-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
assert runner_module.load_model is _refuse_load  # the runner's default model loader is the refusal

logs: list[str] = []


def log(message: str) -> None:
    print(message, flush=True)
    logs.append(message)


EXPECTED = {
    "adjective": ["anxious", "cheerful", "curious", "jealous", "lonely", "nasty", "careful", "careless", "famous", "friendly", "gorgeous", "hungry", "weary",
                  "wicked", "ugly", "vivid", "vague", "rapid", "rigid", "clever"],
    "noun": ["soldier", "sailor", "priest", "knight", "onion", "carrot", "pirate", "tourist", "author", "bishop", "dancer", "duck", "goat", "guitar", "hunter",
             "lawyer", "monk", "nurse", "painter", "prince"],
}


def verify_before_writing(runner) -> tuple[dict, dict, bytes]:
    """Every pre-write check, in memory, from the committed files and the tokenizer; returns the checks, the payload and
    the exact bytes the freeze must write."""
    from neural_decompiler.models import PYTHIA_70M

    base = runner._base()
    tokenizer = runner.tokenizer_loader(PYTHIA_70M)
    payload = cr.freeze_payload(tokenizer, pool=base.inputs.pool, freeze_024=base.inherited["confirmation"], prior=base.prior, config=cr.PRODUCTION)
    confirmation = cr.confirmation_from_payload(payload, base.inputs.pool, cr.PRODUCTION)
    blocked = {key: set(value["ids"]) for key, value in payload["blocked"].items()}
    cues = payload["cues"]
    checks: dict = {}
    checks["picks_are_the_approved_40"] = payload["picks"] == EXPECTED and payload["picks_match_expected"] is True
    single = []
    for cue in cues:  # pm._encode returns a tuple of ints
        single.append(tuple(pm._encode(tokenizer, " " + cue["word"])) == (int(cue["token_id"]),))
        if cue["stratum"] == "noun":
            plural_ok = tuple(pm._encode(tokenizer, " " + cue["plural"])) == (int(cue["plural_token_id"]),)
            if cue["source"] == "new-list":
                plural_ok = plural_ok and cue["plural"] == cr.regular_plural(cue["word"])
            single.append(plural_ok)
    checks["single_token_with_leading_space"] = all(single) and len(single) == 60
    cue_ids = [cue["token_id"] for cue in cues]
    plural_ids = [cue["plural_token_id"] for cue in cues if cue["stratum"] == "noun"]
    all_ids = cue_ids + plural_ids
    checks["token_ids_distinct"] = len(set(cue_ids)) == 40 and len(set(all_ids)) == 60
    checks["none_in_earlier_cues"] = not set(all_ids) & blocked["earlier_cues"]
    checks["none_in_pool_noun_forms"] = not set(all_ids) & blocked["target_forms"]
    checks["none_in_prior_noun_forms"] = not set(all_ids) & blocked["prior_nouns"]
    checks["none_in_exposed_frame_tokens"] = not set(all_ids) & blocked["frame_tokens"]
    prior_file = json.loads((ROOT / cr.PRIOR_NOUNS_020["path"]).read_text())["nouns"]
    statue_barrel = {noun["lexical_key"]: sorted({*noun["sg_ids"], *noun["pl_ids"]}) for noun in prior_file if noun["lexical_key"] in ("statue", "barrel")}
    checks["statue_and_barrel_excluded_by_prior_noun_rule"] = set(statue_barrel) == {"statue", "barrel"} and all(
        set(ids) <= blocked["prior_nouns"] for ids in statue_barrel.values()) and not {"statue", "barrel"} & set(EXPECTED["noun"])
    statue_reason = [entry["reason"] for entry in payload["rejected"] if entry["candidate"] == "statue/statues"]
    checks["statue_rejected_for_the_prior_noun_rule"] = len(statue_reason) == 1 and "020's confirmation list" in statue_reason[0]
    keys = payload["manifest"]["S2-TARGET"]
    untagged = {cr.untagged(key) for key in keys}
    conditions = set(cr.PRODUCTION.conditions)
    checks["manifest_90720_unique_sorted"] = len(keys) == 90720 == len(set(keys)) and keys == sorted(keys)
    checks["every_key_tagged_with_a_frozen_condition"] = all(len(key.split("|")) == 4 and key.rsplit("|", 1)[1] in conditions for key in keys)
    checks["untagged_keys_are_the_4320_prompts"] = untagged == set(confirmation.manifest_keys()) and len(untagged) == 4320
    checks["no_untagged_key_is_a_tagged_key"] = not untagged & set(keys) and all(len(key.split("|")) == 3 for key in untagged)
    checks["no_collision_with_spent_keys"] = not (untagged | set(keys)) & base.forbidden
    checks["no_collision_with_patch_path_keys"] = not (untagged | set(keys)) & set(cr.PATCH_PATH_SPENT_KEYS) and set(cr.PATCH_PATH_SPENT_KEYS) <= base.forbidden
    checks["manifest_digest_recomputes"] = payload["manifest_sha256"] == pm.sha256_text(pm.canonical_json(payload["manifest"]))
    checks["configuration_is_production"] = payload["configuration"] == cr.PRODUCTION.to_json() and payload["configuration"]["count_threshold"] == 27 \
        and payload["configuration"]["reference_tail"]["value"] == 0.01923865414210013 and len(payload["conditions"]) == 21
    checks["content_digest_verifies"] = payload["content_sha256"] == rc.content_digest(payload)
    lists = {"adjective-reserve": len(cr.ADJECTIVE_RESERVES), "ordinary-reserve": len(cr.ORDINARY_RESERVES), "new-list": len(cr.NEW_NOUN_LIST)}
    rejected = {name: sum(1 for entry in payload["rejected"] if entry["list"] == name) for name in lists}
    picked = {name: sum(1 for cue in cues if cue["source"] == name) for name in lists}
    facts = {"forbidden_keys": len(base.forbidden), "blocked_sizes": {key: len(value) for key, value in blocked.items()},
             "candidates_by_list": lists, "rejected_by_list": rejected, "eligible_by_list": {name: lists[name] - rejected[name] for name in lists},
             "picked_by_list": picked, "unpicked_eligible_reserves_by_list": {name: lists[name] - rejected[name] - picked[name] for name in lists},
             "rejected": payload["rejected"],
             "cues": [[cue["stratum"], cue["word"], cue["token_id"], cue.get("plural"), cue.get("plural_token_id"), cue["source"], cue["rank"]] for cue in cues],
             "manifest_sha256": payload["manifest_sha256"], "content_sha256": payload["content_sha256"]}
    data = (pm.canonical_json(payload) + "\n").encode("utf-8")
    facts["expected_file_sha256"] = hashlib.sha256(data).hexdigest()
    return checks, facts, data


runner = runner_module.Runner(log=log)
assert runner.config is runner_module.cr.PRODUCTION
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
status = None
error = None
checks: dict = {}
facts: dict = {}
post: dict = {}
try:
    assert not CONFIRMATION.exists(), "the confirmation file already exists"
    checks, facts, expected_bytes = verify_before_writing(runner)
    failing = sorted(name for name, ok in checks.items() if not ok)
    if failing:
        raise RuntimeError(f"pre-write verification failed: {failing}; nothing was written")
    if DRY_RUN:
        status = 0
        post = {"dry_run": True, "confirmation_exists": CONFIRMATION.exists()}
    else:
        events["freeze_invocations"] += 1
        status = runner.freeze()  # the one production freeze
        written = CONFIRMATION.read_bytes()
        post = {"file_sha256": hashlib.sha256(written).hexdigest(), "bytes_equal_pre_verified_payload": written == expected_bytes,
                "content_sha256": json.loads(written)["content_sha256"]}
except BaseException as exc:  # recorded, then the exit status says so
    error = f"{type(exc).__name__}: {exc}"
ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
record = {"dry_run": DRY_RUN, "started_at": started, "ended_at": ended, "exit_status": status, "error": error, "freeze_invocations": events["freeze_invocations"],
          "events": events, "pre_write_checks": checks, "facts": facts, "post_write": post, "logs": logs, "config": runner.config.to_json()["name"]}
(HERE / ("dryrun_run.json" if DRY_RUN else "freeze_run.json")).write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({key: record[key] for key in ("started_at", "ended_at", "exit_status", "error", "freeze_invocations")}), flush=True)
print(json.dumps(events), flush=True)
print(json.dumps({"checks_all_true": bool(checks) and all(checks.values()), "post_write": post}), flush=True)
ok = status == 0 and error is None and (post.get("confirmation_exists") is False if DRY_RUN else post.get("bytes_equal_pre_verified_payload") is True)
sys.exit(0 if ok else 1)
