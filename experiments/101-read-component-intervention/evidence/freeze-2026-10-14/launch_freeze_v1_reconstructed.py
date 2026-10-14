"""Guarded launcher for Experiment 101's production `freeze` (run exactly once; 025's freeze-launcher pattern).

It runs from a scratchpad directory outside the repository and writes its own records there; the files are copied
byte-identically into `experiments/101-read-component-intervention/evidence/freeze-2026-10-14/` afterwards.

Before the runner module is imported:
- `neural_decompiler.models.load_model` and the transformer-lens and transformers model constructors refuse: the
  freeze is tokenizer-only, so no weight, logit, hidden state or attention can exist;
- `torch.nn.Module.__call__` refuses: no forward of any module;
- every capture and intervention entry point refuses (`plural_mechanism`'s four, `capture.run_capture`,
  `interventions.run_interventions`, `readout_decompilation.capture_frame_020`);
- every geometry, prediction, measurement and outcome function of `read_intervention` refuses, and so do the frozen
  program's `upstream_localization.compose_dx3` and `contrast_of`: the geometry belongs to calibration and the lock,
  and no model quantity may select, order or exclude a frame;
- an audit hook refuses every repository write, delete, rename and directory creation except the one confirmation
  file the freeze writes (and the `mkdir(exist_ok=True)` of its existing directory); it records every open under
  `outputs/` (integrity reads of 020's closure, as 025's reviewer accepted for 025's freeze, note N3).

The pre-write verification (in memory, before the one freeze invocation) re-derives the 36 frames from the raw
tokenizer, independently of `pm._build_new_frame`, and checks every eligibility rule of design §11 and plan § The frame
freeze against the committed exclusion sources, the S1 and S2 manifests, their isolation from every spent or reserved
key, and the payload's digests. Only if every check passes does it call `runner.freeze()`, exactly once (production
mode), and then it verifies that the written file is the pre-verified payload, byte for byte.

Dry-run mode (`FREEZE101_DRY_RUN=1`): every guard and pre-write check; `freeze` is not called and nothing is written.
"""
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
DRY_RUN = os.environ.get("FREEZE101_DRY_RUN") == "1"
EXPERIMENT_DIR = (ROOT / "experiments/101-read-component-intervention").resolve()
CONFIRMATION = (EXPERIMENT_DIR / "confirmation-v1.json").resolve()
SELF_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
events = {"module_calls": 0, "load_model": 0, "model_constructors": [], "capture_calls": [], "refused_functions": [], "forbidden_writes": [],
          "forbidden_fs_ops": [], "outputs_reads": [], "confirmation_writes": 0, "allowed_mkdir": 0, "freeze_invocations": 0}


def _in_repo(text: str) -> bool:
    return text.startswith(str(ROOT) + "/") and "/.venv/" not in text


def _hook(event, args):
    if event == "open":
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        try:
            path = Path(os.fsdecode(args[0])).resolve()
        except OSError:
            return
        text = str(path)
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if text.startswith(str(ROOT / "outputs") + "/") and not writing:
            if text not in events["outputs_reads"]:
                events["outputs_reads"].append(text)
        if writing and _in_repo(text):
            if path == CONFIRMATION:
                events["confirmation_writes"] += 1
                return
            events["forbidden_writes"].append(text)
            raise PermissionError(f"the freeze may not write {text}")
        return
    if event in ("os.remove", "os.unlink", "os.rename", "os.replace", "os.rmdir", "shutil.rmtree", "shutil.move", "os.mkdir", "os.truncate", "os.chmod",
                 "os.symlink", "os.link"):
        texts = []
        for arg in args:
            if isinstance(arg, (str, bytes, os.PathLike)):
                try:
                    texts.append(str(Path(os.fsdecode(arg)).resolve()))
                except OSError:
                    texts.append(os.fsdecode(arg))
        if any(_in_repo(text) for text in texts):
            if event == "os.mkdir" and texts and Path(texts[0]) == EXPERIMENT_DIR:
                events["allowed_mkdir"] += 1
                return
            events["forbidden_fs_ops"].append([event, texts])
            raise PermissionError(f"the freeze may not {event} {texts}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import capture as capture_module  # noqa: E402
from neural_decompiler import interventions as interventions_module  # noqa: E402
from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402


def _refuse_module_call(self, *args, **kwargs):
    events["module_calls"] += 1
    raise RuntimeError("a torch module was called during the freeze")


def _refuse_load(*args, **kwargs):
    events["load_model"] += 1
    raise RuntimeError("the model was loaded during the freeze")


def _refuse_named(kind, name):
    def refuse(*args, **kwargs):
        events[kind].append(name)
        raise RuntimeError(f"{name} was called during the freeze (tokenizer only)")

    return refuse


torch.nn.Module.__call__ = _refuse_module_call
models.load_model = _refuse_load
try:
    import transformer_lens  # noqa: E402

    for name in ("from_pretrained", "from_pretrained_no_processing"):
        if hasattr(transformer_lens.HookedTransformer, name):
            setattr(transformer_lens.HookedTransformer, name, _refuse_named("model_constructors", f"HookedTransformer.{name}"))
except ImportError:
    pass
import transformers  # noqa: E402

for cls_name in ("AutoModel", "AutoModelForCausalLM", "GPTNeoXForCausalLM", "GPTNeoXModel"):
    cls = getattr(transformers, cls_name, None)
    if cls is not None:
        cls.from_pretrained = _refuse_named("model_constructors", f"transformers.{cls_name}.from_pretrained")
for module, names in ((pm, ("capture_prompt", "run_patched", "run_capture", "run_interventions")), (capture_module, ("run_capture",)),
                      (interventions_module, ("run_interventions",)), (rd, ("capture_frame_020",))):
    for name in names:
        if hasattr(module, name):
            setattr(module, name, _refuse_named("capture_calls", f"{module.__name__.rsplit('.', 1)[1]}.{name}"))
REFUSED_RI = ("read_geometry", "g1_read_weight", "iso_direction", "iso_directions", "basis_checks", "geometry_block", "pair_quantities", "edit_vector",
              "residual_replacement", "predict_cue", "prediction_table", "run_arm", "capture_reference", "head_delta", "measure_frame", "stage_runs",
              "frame_gates", "run_gates", "run_records", "frame_values", "primary_statistics", "secondary_statistics", "calibration_results")
for name in REFUSED_RI:
    setattr(ri, name, _refuse_named("refused_functions", f"ri.{name}"))
for name in ("compose_dx3", "contrast_of", "reference_rows_017", "pair_context"):
    setattr(ul, name, _refuse_named("refused_functions", f"ul.{name}"))

spec = importlib.util.spec_from_file_location("experiment_101_runner", ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
assert runner_module.load_model is _refuse_load  # the runner's default model loader is the refusal

logs: list[str] = []


def log(message: str) -> None:
    print(message, flush=True)
    logs.append(message)


CUE_WORDS = {"ref": ("one", 581), "two": ("two", 767), "three": ("three", 1264), "many": ("many", 1142)}


def independent_frame(tokenizer, text: str) -> dict:
    """The frame re-derived from the raw tokenizer: each cue word in the slot, the cue's single occurrence, the shared
    prefix and suffix, the adjective as the final single token."""
    out = {"ok": True, "problems": []}
    tokenized = {}
    for label, (word, token_id) in CUE_WORDS.items():
        if tuple(int(t) for t in tokenizer.encode(" " + word, add_special_tokens=False)) != (token_id,):
            out["problems"].append(f"' {word}' is not the single token {token_id}")
        ids = [int(t) for t in tokenizer.encode(text.replace("{cue}", word), add_special_tokens=False)]
        where = [i for i, t in enumerate(ids) if t == token_id]
        if len(where) != 1:
            out["problems"].append(f"{label}: cue occurs {len(where)} times")
            continue
        tokenized[label] = (ids, where[0])
    if len(tokenized) == 4:
        (ref_ids, p_c) = tokenized["ref"]
        for label, (ids, position) in tokenized.items():
            if position != p_c or ids[:position] != ref_ids[:p_c] or ids[position + 1:] != ref_ids[p_c + 1:]:
                out["problems"].append(f"{label}: prefix or suffix differs from the reference prompt's")
        suffix = ref_ids[p_c + 1:]
        adjective_word = text.rsplit(" ", 1)[1]
        adjective_ids = [int(t) for t in tokenizer.encode(" " + adjective_word, add_special_tokens=False)]
        if len(suffix) != 1 or suffix != adjective_ids:
            out["problems"].append(f"the adjective {adjective_word!r} is not one final token (suffix {suffix}, own ids {adjective_ids})")
        if not 4 <= p_c <= 7:
            out["problems"].append(f"p_c {p_c} outside 4–7")
        out.update({"prefix_ids": ref_ids[:p_c], "suffix_ids": suffix, "p_c": p_c, "p_t": p_c + len(suffix), "adjective": adjective_word,
                    "adjective_id": suffix[0] if suffix else None})
    out["ok"] = not out["problems"]
    return out


def verify_before_writing(runner) -> tuple[dict, dict, bytes]:
    from neural_decompiler.models import PYTHIA_70M

    base = runner._base()
    tokenizer = runner.tokenizer_loader(PYTHIA_70M)
    payload = ri.freeze_payload(tokenizer, exclusions=base.exclusions, config=ri.PRODUCTION)
    confirmation = ri.confirmation_from_payload(payload, ri.PRODUCTION)
    checks: dict = {}
    facts: dict = {}
    checks["picks_are_candidates_1_to_36"] = payload["picks"] == list(range(1, 37)) == payload["expected_picks"] and payload["picks_match_expected"] is True
    # The committed exclusion sources, read directly from the committed files (not from ri.Exclusions).
    c023 = json.loads((ROOT / "experiments/023-block0-completion/confirmation-v1.json").read_text())
    c024 = json.loads((ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json").read_text())
    c025 = json.loads((ROOT / "experiments/025-nounness-direction-intervention/confirmation-v1.json").read_text())
    texts = set(c023["exclusion"]["frame_texts"]) | {frame["text_template"] for frame in c023["frames"]}
    earlier = {int(i) for i in c024["exclusion"]["cue_token_ids"]} | {int(cue["token_id"]) for cue in c024["cues"]} | {int(cue["token_id"]) for cue in c025["cues"]}
    words_025 = {str(w).lower() for w in c025["candidates"]["adjective-reserve"]} | {str(w).lower() for w in c025["candidates"]["new-list"]} \
        | {str(w).lower() for pair in c025["candidates"]["ordinary-reserve"] for w in pair}
    ids_025 = set()
    for word in words_025:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            ids_025.add(int(ids[0]))
    targets = {int(i) for i in c024["target_noun_form_ids"]["ids"]}
    prior = {int(i) for i in c025["blocked"]["prior_nouns"]["ids"]}
    facts["exclusion_sizes"] = {"frame_texts": len(texts), "earlier_cue_ids": len(earlier), "candidate_words_025": len(words_025),
                                "candidate_ids_025": len(ids_025), "target_form_ids": len(targets), "prior_noun_ids": len(prior)}
    checks["exclusion_sizes_as_planned"] = facts["exclusion_sizes"]["earlier_cue_ids"] == 431 and len(targets) == 161 and len(prior) == 48
    checks["exclusion_record_matches_committed_sources"] = (payload["exclusions"]["frame_texts"]["count"] == len(texts)
                                                            and payload["exclusions"]["earlier_cue_ids"]["count"] == len(earlier)
                                                            and payload["exclusions"]["target_form_ids"]["count"] == len(targets)
                                                            and payload["exclusions"]["prior_noun_ids"]["count"] == len(prior))
    rederived = {}
    eligible_numbers = []
    adjectives_seen = {}
    for number, text in enumerate(ri.FRAME_CANDIDATES, start=1):
        frame = independent_frame(tokenizer, text)
        reasons = list(frame["problems"])
        if text in texts:
            reasons.append("text is an earlier frame text")
        adjective_id = frame.get("adjective_id")
        adjective = frame.get("adjective")
        if adjective_id is not None:
            if adjective_id in earlier:
                reasons.append("earlier cue")
            if adjective.lower() in words_025 or adjective_id in ids_025:
                reasons.append("025 candidate word")
            if adjective_id in targets or adjective_id in prior:
                reasons.append("target-noun form")
            if adjective.lower() in ri.NUMBER_WORDS:
                reasons.append("number word")
            if adjective_id in adjectives_seen:
                reasons.append(f"adjective of candidate {adjectives_seen[adjective_id]}")
        if not reasons:
            adjectives_seen[adjective_id] = number
            eligible_numbers.append(number)
        rederived[number] = {"text": text, "eligible": not reasons, "reasons": reasons, **{k: frame.get(k) for k in ("p_c", "p_t", "adjective", "adjective_id")}}
    facts["independent_eligible_numbers"] = eligible_numbers
    checks["independent_first_36_eligible_are_1_to_36"] = eligible_numbers[:36] == list(range(1, 37))
    payload_status = {entry["number"]: entry for entry in payload["candidates"]}
    checks["payload_eligibility_equals_independent_for_all_60"] = all(bool(payload_status[n]["eligible"]) == rederived[n]["eligible"] for n in range(1, 61))
    frames_ok = []
    for entry, number in zip(payload["frames"], payload["picks"]):
        mine = independent_frame(tokenizer, ri.FRAME_CANDIDATES[number - 1])
        frames_ok.append(entry["text_template"] == ri.FRAME_CANDIDATES[number - 1] and list(entry["prefix_ids"]) == mine["prefix_ids"]
                         and list(entry["suffix_ids"]) == mine["suffix_ids"] and dict(entry["cue_ids"]) == {"sg": 581, "pl": 767}
                         and entry["frame_id"] == f"coordinated-adjective-101-{number:02d}" and entry["template_id"] == "coordinated-adjective")
    checks["frames_equal_independent_tokenization"] = len(frames_ok) == 36 and all(frames_ok)
    adjectives = [rederived[n]["adjective"] for n in range(1, 37)]
    checks["adjectives_distinct"] = len(set(adjectives)) == 36
    facts["adjectives"] = adjectives
    facts["p_c_counts"] = {str(p): sum(1 for n in range(1, 37) if rederived[n]["p_c"] == p) for p in range(4, 8)}
    facts["exposed_frame_token_picks"] = [a for a in adjectives if a in ("soft", "tiny", "clean", "damp", "wet", "thin", "sour", "pale", "neat", "dry", "cool", "loose")]
    s1, s2 = payload["manifest"]["S1-REF"], payload["manifest"]["S2-TARGET"]
    checks["s1_is_36_reference_keys"] = len(s1) == 36 == len(set(s1)) and all(key.endswith("|ref|581|S1") for key in s1)
    checks["s2_is_1980_distinct_keys"] = len(s2) == 1980 == len(set(s2)) and s2 == sorted(s2)
    checks["s1_s2_disjoint"] = not set(s1) & set(s2)
    tags = list(ri.PRODUCTION.tags())
    per_frame = {}
    for key in s2:
        frame_id, label, token, tag = key.split("|")
        per_frame.setdefault(frame_id, []).append((label, token, tag))
    checks["s2_55_per_frame_with_the_frozen_tags"] = len(per_frame) == 36 and all(sorted(t for _, _, t in runs) == sorted(tags) for runs in per_frame.values())
    label_token = {"ref": "581", "two": "767", "three": "1264", "many": "1142"}
    checks["s2_prompt_labels_match_tokens"] = all(label_token.get(label) == token for runs in per_frame.values() for label, token, _ in runs)
    untagged = {cr.untagged(key) for key in s1 + s2}
    checks["untagged_prompts_are_144"] = len(untagged) == 144
    checks["no_collision_with_forbidden"] = not (untagged | set(s1) | set(s2)) & base.forbidden
    facts["forbidden_keys"] = len(base.forbidden)
    checks["manifest_digest_recomputes"] = payload["manifest_sha256"] == pm.sha256_text(pm.canonical_json(payload["manifest"]))
    checks["content_digest_recomputes"] = payload["content_sha256"] == rc.content_digest(payload)
    checks["configuration_is_production"] = payload["configuration"] == ri.PRODUCTION.to_json() and payload["configuration"]["holm_thresholds"] == [26, 25] \
        and payload["configuration"]["runs_per_frame"] == 55
    checks["counts"] = payload["counts"] == {"frames": 36, "runs_per_frame": 55, "S1-REF": 36, "S2-TARGET": 1980, "untagged_prompts": 144}
    checks["confirmation_reverifies"] = confirmation.content_sha256 == payload["content_sha256"]
    facts["rederived"] = rederived
    facts["manifest_sha256"] = payload["manifest_sha256"]
    facts["content_sha256"] = payload["content_sha256"]
    data = (pm.canonical_json(payload) + "\n").encode("utf-8")
    facts["expected_file_sha256"] = hashlib.sha256(data).hexdigest()
    facts["expected_bytes"] = len(data)
    return checks, facts, data


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


runner = runner_module.Runner(log=log)
assert runner.config is ri.PRODUCTION
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
environment = {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "status_porcelain": git("status", "--porcelain"),
               "origin_main": git("rev-parse", "origin/main"), "launcher_sha256": SELF_SHA256, "python": sys.version, "platform": platform.platform(),
               "torch": torch.__version__, "transformers": transformers.__version__, "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
               "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"), "dry_run": DRY_RUN}
status = None
error = None
checks: dict = {}
facts: dict = {}
post: dict = {}
try:
    assert not CONFIRMATION.exists(), "the confirmation file already exists"
    assert environment["status_porcelain"] == "", "the tree is not clean before the freeze"
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
        post = {"file_sha256": hashlib.sha256(written).hexdigest(), "bytes": len(written), "bytes_equal_pre_verified_payload": written == expected_bytes,
                "content_sha256": json.loads(written)["content_sha256"], "status_after": git("status", "--porcelain")}
except BaseException as exc:  # recorded, then the exit status says so
    error = f"{type(exc).__name__}: {exc}"
ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
record = {"dry_run": DRY_RUN, "started_at": started, "ended_at": ended, "exit_status": status, "error": error, "freeze_invocations": events["freeze_invocations"],
          "environment": environment, "events": events, "pre_write_checks": checks, "facts": facts, "post_write": post, "logs": logs,
          "config": runner.config.to_json()["name"]}
(HERE / ("dryrun_run.json" if DRY_RUN else "freeze_run.json")).write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({key: record[key] for key in ("started_at", "ended_at", "exit_status", "error", "freeze_invocations")}), flush=True)
print(json.dumps({k: v for k, v in events.items() if k != "outputs_reads"}), flush=True)
print(json.dumps({"checks_all_true": bool(checks) and all(checks.values()), "failing": sorted(k for k, v in checks.items() if not v), "post_write": post}), flush=True)
ok = status == 0 and error is None and (post.get("confirmation_exists") is False if DRY_RUN else post.get("bytes_equal_pre_verified_payload") is True)
sys.exit(0 if ok else 1)
