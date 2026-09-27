#!/usr/bin/env python3
"""Build the committed Experiment 025 confirmation-record extract from the gitignored final results state.

Adapted from Experiment 024's builder. It is read-only on the state and on every artifact, and writes only the
extract. An audit hook refuses any other repository write and any open of 022's local table or of 023's or 024's local
outputs; every model, capture, intervention, measurement and gate entry point, and `torch.nn.Module.__call__`, refuse.
It follows the 022–024 closure extract:
- canonical JSON plus a newline;
- content digest = sha256(canonical_json(record without content_sha256));
- the 90,720-key prompt ledger is replaced by its counts and digests.

It adds:
- the accounting of the one confirm run: the launcher's run log against the manifest and the ledger;
- the confirmation isolation against every earlier spent set, and the four spent patch-path keys;
- the provenance of the exit-127 shell event before the launch of record;
- a summary of the official result, copied from the recorded state, never recomputed or reinterpreted.

No model, no prompt, no phase.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
os.chdir(ROOT)
EXPERIMENT_DIR = "experiments/025-nounness-direction-intervention"
EVIDENCE = ROOT / EXPERIMENT_DIR / "evidence"
TARGET = EVIDENCE / "confirmation-record-2026-09-26.json"
OUT = ROOT / "outputs/experiment-025"
TABLE_022 = str(ROOT / "outputs/experiment-022/calibration-table.pt")
OUTPUTS_023 = str(ROOT / "outputs/experiment-023") + os.sep
OUTPUTS_024 = str(ROOT / "outputs/experiment-024") + os.sep


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    raw = os.fsdecode(args[0])
    text = os.path.normpath(raw if os.path.isabs(raw) else os.path.join(os.getcwd(), raw))
    if text == TABLE_022 or text.startswith(OUTPUTS_023) or text.startswith(OUTPUTS_024):
        raise PermissionError(f"the extract builder may not open {text}")
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if not writing:
        return
    path = Path(text).resolve()
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path) and path != TARGET.resolve():
        raise PermissionError(f"the extract builder may write only {TARGET}: {path}")


sys.addaudithook(_hook)

import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("run025", ROOT / EXPERIMENT_DIR / "run.py")
run025 = importlib.util.module_from_spec(spec)
sys.modules["run025"] = run025
spec.loader.exec_module(run025)

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import capture, interventions  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402


def refuse(*args, **kwargs):
    raise RuntimeError("the extract builder never loads a model or a tokenizer, and never runs a forward")


for module, names in ((pm, ("capture_prompt", "run_capture", "run_patched", "run_interventions", "record_execution")), (ul, ("stage_two_022", "measure_prompt")),
                      (capture, ("run_capture",)), (interventions, ("run_interventions",)),
                      (cr, ("measure_rotated", "stage_two", "patch_path_check", "identity_gates", "level1_gates", "run_gates", "recompute_c"))):
    for name in names:
        setattr(module, name, refuse)
torch.nn.Module.__call__ = refuse

GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return pm.sha256_text(pm.canonical_json(value))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, env=GIT_ENV).stdout.strip()


def evidence_json(relative: str) -> dict:
    return json.loads((EVIDENCE / relative).read_text(encoding="utf-8"))


state_path = OUT / "results.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
assert canonical_sha(unsigned) == state["state_sha256"], "the results state's own digest does not verify"
assert state["experiment"] == cr.EXPERIMENT and not state["git_dirty"]
assert {k: v["status"] for k, v in state["phases"].items()} == {p: "complete" for p in cr.STATE_PHASES}, "every state phase must be complete"
assert not any(entry.get("incidents") for entry in state["phases"].values()) and "incident" not in state["confirmation"], "an incident is recorded"
assert not (state["confirmation"].get("descriptives") or {}).get("failures"), "a descriptive failure is recorded"

# the confirm-final state: the final state with only the two report entries reverted (the report run added nothing else)
confirm_final = {**unsigned, "report": None, "phases": {**unsigned["phases"], "report": {"status": "not_started"}}}
confirm_final_sha = canonical_sha(confirm_final)
confirm_final_file_sha = hashlib.sha256((pm.canonical_json({**confirm_final, "state_sha256": confirm_final_sha}) + "\n").encode("utf-8")).hexdigest()
report_run = evidence_json("report-2026-09-26/report_record.json")
assert confirm_final_sha == report_run["pre_report_check"]["state"] and confirm_final_file_sha == report_run["pre_report_check"]["state_file"], \
    "the confirm-final state recorded before the report run is not reproduced"
assert report_run["after"]["state"] == state["state_sha256"] and report_run["after"]["state_file"] == file_sha(state_path)
# the state at the result write: the confirm-final state without the descriptive records, which were written after it
result_write = {**{k: v for k, v in confirm_final.items() if k != "confirmation"},
                "confirmation": {k: v for k, v in confirm_final["confirmation"].items() if k != "descriptives"}}
result_write_sha = canonical_sha(result_write)
assert result_write_sha == "aa3db023c4a4afbe730a0e55df00d6d378184bcf8fdf77c967f4d0aad11907c4", "the result-write state is not reproduced"

runner = run025.Runner(log=lambda message: None, model_loader=refuse, tokenizer_loader=refuse)
base = runner._base()  # the frozen inputs; no model, no prompt
confirmation_set, confirmation_file_sha = runner._confirmation(base)
assert state["inputs"] == {key: base.digests[key] for key in cr.DIGEST_KEYS}, "the state's input digests are not the current frozen inputs"
lock = json.loads((ROOT / cr.LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
assert rc.content_digest(lock) == lock["content_sha256"] == state["lock"]["content_sha256"] == state["confirmation"]["lock_sha256"] \
    == state["phases"]["confirm"]["lock_sha256"]
assert lock["confirmation_025"] == cr.confirmation_binding(confirmation_set, confirmation_file_sha) == state["confirmation_025"]
assert lock["module"] == {"path": "src/neural_decompiler/cue_rotation.py", "blob": cr.own_blob()} == {"path": "src/neural_decompiler/cue_rotation.py",
                                                                                                     "blob": git("rev-parse", "HEAD:src/neural_decompiler/cue_rotation.py")}

# the accounting of the one confirm run (the launcher's run log and counts, against the manifest and the ledger)
executed = list(state["executed_prompt_keys"])
manifest = confirmation_set.manifest()["S2-TARGET"]
manifest_keys = set(manifest)
assert canonical_sha(confirmation_set.manifest()) == lock["confirmation_025"]["manifest_sha256"]
run_log = EVIDENCE / "confirm-2026-09-26/production_runs.jsonl"
calls = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines()]
launcher = evidence_json("confirm-2026-09-26/production_run.json")
call_keys = [call["key"] for call in calls]
assert len(set(executed)) == len(executed) == len(manifest_keys) == 90_720 and set(executed) == manifest_keys
assert Counter(call_keys) == Counter({key: 1 for key in manifest_keys}) and [call["n"] for call in calls] == list(range(1, 90_721))
assert executed == sorted(executed), "the ledger is written as one sorted set"
per_condition = dict(sorted(Counter(key.rsplit("|", 1)[1] for key in call_keys).items()))
assert list(per_condition) == sorted(confirmation_set.conditions) and set(per_condition.values()) == {4_320}
assert launcher["exit_status"] == 0 and launcher["error"] is None and launcher["events"]["fresh_runs"] == 90_720 and launcher["events"]["refused"] == []
assert launcher["events"]["confirm_invocations"] == 1 and launcher["events"]["load_model"] == 1
assert launcher["events"]["forbidden_reads"] == [] and launcher["events"]["forbidden_writes"] == []
assert launcher["environment"]["head"] == launcher["head_after"] == state["phases"]["confirm"]["confirm_commit"] and launcher["tree_after"] == ""
assert all(value is True or (isinstance(value, dict) and value.get("ok") is True) for value in launcher["preflight"].values())
accounting = {
    "rule": "expected manifest = executed fresh runs (the launcher's run log) = ledger spend",
    "manifest_keys": len(manifest_keys), "executed_runs": len(call_keys), "executed_unique": len(set(call_keys)), "ledger_keys": len(executed),
    "extra": len(set(call_keys) - manifest_keys), "missing": len(manifest_keys - set(call_keys)), "duplicated": sum(1 for n in Counter(call_keys).values() if n > 1),
    "per_condition": per_condition,
    "ledger_is_sorted": executed == sorted(executed),
    "order": ("the runner writes the ledger as one sorted set before the first fresh run; the run log is in execution order, and the independent "
              "confirmation review verified that its n-th key is the runner's loop-order n-th key for all n"),
    "runner_accounting": state["confirmation"]["accounting"],
    "launcher": {"path": f"{EXPERIMENT_DIR}/evidence/confirm-2026-09-26/launch_confirm_v3.py", "file_sha256": file_sha(EVIDENCE / "confirm-2026-09-26/launch_confirm_v3.py"),
                 "recorded_sha256": launcher["environment"]["launcher_sha256"], "mode": launcher["mode"], "head": launcher["environment"]["head"],
                 "head_after": launcher["head_after"], "tree_after": launcher["tree_after"], "started_at": launcher["started_at"], "ended_at": launcher["ended_at"],
                 "exit": launcher["exit_status"], "error": launcher["error"],
                 "record": f"{EXPERIMENT_DIR}/evidence/confirm-2026-09-26/production_run.json",
                 "record_sha256": file_sha(EVIDENCE / "confirm-2026-09-26/production_run.json")},
    "launcher_preflight": launcher["preflight"],
    "launcher_events": launcher["events"],
    "launcher_sealing": {"instances": launcher["sealing"]["instances"], "class_methods": len(launcher["sealing"]["class_methods"])},
    "run_log": {"path": f"{EXPERIMENT_DIR}/evidence/confirm-2026-09-26/production_runs.jsonl", "file_sha256": file_sha(run_log),
                "sequence_sha256": canonical_sha(call_keys)},
}
assert accounting["launcher"]["file_sha256"] == accounting["launcher"]["recorded_sha256"] == "58b760bea9d8a7bf66bef4352deb5c7153f7aea46c264de02f49c9ab96d1e307"

# the confirmation isolation: every tagged ledger key, and its untagged prompt key, against every earlier spent set
ledger_020 = frozenset(base.inputs.closure["ledger"])
spent_021 = frozenset(prompt.key for prompt in base.inputs.confirmation_020.all_prompts)
manifest_022 = run025._manifest_keys_022(json.loads((ROOT / ul.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8")))
manifest_023 = run025._manifest_keys_022(json.loads((ROOT / b0c.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8")))
manifest_024 = frozenset(base.inherited["confirmation"]["manifest"]["S2-TARGET"])
assert (ledger_020 | spent_021 | manifest_022 | manifest_023 | manifest_024) == base.forbidden and len(base.forbidden) == 43_632
tagged = frozenset(executed)
prompts = frozenset(cr.untagged(key) for key in executed)
assert prompts == confirmation_set.manifest_keys() and len(prompts) == 4_320


def overlap(keys: frozenset) -> dict:
    return {"keys": len(keys), "executed_tagged": len(tagged & keys), "executed_prompts": len(prompts & keys)}


spent_runs = launcher["events"]["spent_runs"]
isolation = {
    "confirmation_025_content_sha256": confirmation_set.content_sha256,
    "manifest": {"tagged_keys": len(manifest_keys), "prompts": len(confirmation_set.manifest_keys()), "executed_tagged": len(tagged & manifest_keys),
                 "sha256": lock["confirmation_025"]["manifest_sha256"]},
    "experiment_020_ledger": overlap(ledger_020),
    "experiment_021_spent_set": overlap(spent_021),
    "experiment_022_manifest": overlap(manifest_022),
    "experiment_023_manifest": overlap(manifest_023),
    "experiment_024_manifest": overlap(manifest_024),
    "spent_union": overlap(base.forbidden),
    "keys_outside_the_manifest": len(tagged - manifest_keys),
    "patch_path_spent_keys": {"keys": list(cr.PATCH_PATH_SPENT_KEYS), "all_in_the_spent_union": all(key in base.forbidden for key in cr.PATCH_PATH_SPENT_KEYS),
                              "in_the_manifest_prompts": len(set(cr.PATCH_PATH_SPENT_KEYS) & confirmation_set.manifest_keys()),
                              "in_the_ledger": len(set(cr.PATCH_PATH_SPENT_KEYS) & (tagged | prompts)),
                              "launcher_spent_runs": len(spent_runs), "launcher_spent_runs_by_kind": dict(sorted(Counter(run["kind"] for run in spent_runs).items()))},
}
assert all(entry["executed_tagged"] == entry["executed_prompts"] == 0 for name, entry in isolation.items() if name.startswith(("experiment_", "spent_union")))
assert isolation["patch_path_spent_keys"]["all_in_the_spent_union"] and isolation["patch_path_spent_keys"]["in_the_manifest_prompts"] == 0 \
    and isolation["patch_path_spent_keys"]["in_the_ledger"] == 0 and {run["key"] for run in spent_runs} == set(cr.PATCH_PATH_SPENT_KEYS)

# the exit-127 shell event before the launch of record (provenance only; it never started Python)
shell_files = ("shell_error_production.log", "shell_error_production.exit", "shell_error_production_command.txt")
shell_exit = (EVIDENCE / "confirm-2026-09-26/shell_error_production.exit").read_text(encoding="utf-8")
assert shell_exit.startswith("exit 127\n") and "ended_at_utc 2026-09-26T19:23:09Z" in shell_exit
assert (EVIDENCE / "confirm-2026-09-26/shell_error_production.log").read_text(encoding="utf-8").startswith("(eval):1: no such file or directory: caffeinate -i -s env ")
shell_event = {
    "exit_status": 127, "ended_at_utc": "2026-09-26T19:23:09Z",
    "what": ("the first launch command failed in the shell: the command had been stored in a shell variable, and zsh does not word-split an unquoted "
             "variable, so it looked for a program named after the whole string ('no such file or directory')"),
    "python_started": False, "launcher_code_ran": False, "runner_confirm_invoked": False, "scientific_attempt": False,
    "state_output_or_ledger_changed": False,
    "launch_of_record": {"command": f"{EXPERIMENT_DIR}/evidence/confirm-2026-09-26/production_command.txt", "started_at": launcher["started_at"],
                         "sentinel": f"{EXPERIMENT_DIR}/evidence/confirm-2026-09-26/confirm.sentinel",
                         "sentinel_content": (EVIDENCE / "confirm-2026-09-26/confirm.sentinel").read_text(encoding="utf-8")},
    "independent_review": f"{EXPERIMENT_DIR}/evidence/confirmation-review-2026-09-26/REVIEW.md (provenance)",
    "files": {f"{EXPERIMENT_DIR}/evidence/confirm-2026-09-26/{name}": file_sha(EVIDENCE / "confirm-2026-09-26" / name) for name in shell_files},
}

committed = {path: {"file_sha256": file_sha(ROOT / path)} for path in (cr.CONFIRMATION_RELATIVE_PATH, cr.LOCK_RELATIVE_PATH, cr.PREREGISTRATION_RELATIVE_PATH)}
committed[cr.CONFIRMATION_RELATIVE_PATH].update({"content_sha256": confirmation_set.content_sha256, "installed_in": git("rev-parse", "57f1f98")})
committed[cr.LOCK_RELATIVE_PATH].update({"content_sha256": lock["content_sha256"], "installed_in": git("rev-parse", "5ee7a74")})
committed[cr.PREREGISTRATION_RELATIVE_PATH]["installed_in"] = git("rev-parse", "5ee7a74")
assert committed[cr.PREREGISTRATION_RELATIVE_PATH]["file_sha256"] == state["lock"]["preregistration_sha256"]
local = ("results.json", "stage2-measurements.pt", "report.md", "candidate-lock.json", "candidate-preregistration.md")
assert sorted(path.name for path in OUT.iterdir()) == sorted(local), "the local outputs are not the five expected files"
assert (OUT / "candidate-lock.json").read_bytes() == (ROOT / cr.LOCK_RELATIVE_PATH).read_bytes()
assert (OUT / "candidate-preregistration.md").read_bytes() == (ROOT / cr.PREREGISTRATION_RELATIVE_PATH).read_bytes()
copies = {"final-report-2026-09-26.md": "report.md"}
for name, source in copies.items():
    assert (EVIDENCE / name).read_bytes() == (OUT / source).read_bytes(), f"{name} is not byte-identical to outputs/experiment-025/{source}"


def run_of(relative: str, started: str, ended: str, commit: str) -> dict:
    entry = evidence_json(relative)
    return {"record": f"{EXPERIMENT_DIR}/evidence/{relative}", "record_sha256": file_sha(EVIDENCE / relative), "commit": commit,
            "started": entry[started], "ended": entry[ended], "exit": entry["exit_status"], "error": entry["error"]}


results = state["confirmation"]["results"]
confirm_commit = state["phases"]["confirm"]["confirm_commit"]
report_commit = report_run["pre_report_check"]["head"]
stage2 = OUT / "stage2-measurements.pt"
official = {  # copied from the recorded state and the evidence, never recomputed
    "experiment": cr.EXPERIMENT, "confirmation_commit": confirm_commit, "report_commit": report_commit,
    "stage2_measurements": {"path": "outputs/experiment-025/stage2-measurements.pt", "file_sha256": file_sha(stage2), "bytes": stage2.stat().st_size,
                            "tensors": len(state["confirmation"]["stage2"]["tensors_sha256"]), "n_executed": state["confirmation"]["stage2"]["n_executed"]},
    "runs": len(executed), "manifest_sha256": lock["confirmation_025"]["manifest_sha256"],
    "outcome": results["outcome"]["label"],
    "outcome_reading": results["outcome"]["reading"],
    "not_claimed": results["outcome"]["not_claimed"],
    "statistics": {name: {key: results[name][key] for key in ("positive", "n", "threshold", "result")} for name in ("A", "B", "G")},
    "count_rule": lock["statistics"]["count_rule"],
    "reference_tail": results["criterion"]["reference_tail"],
    "label_table": lock["outcome"]["table"],
    "gates": state["confirmation"]["gates"], "c_recompute": state["confirmation"]["c_recompute"],
    "incident": None,
    "descriptive_only": [
        "the adjective/noun stratum counts (results.A/B/G.by_stratum, descriptives.records.strata): they cannot change the label",
        "G = 37/40: a recorded PASS of that statistic; under the frozen hierarchy it cannot change the label, because A failed",
        "the causal fraction (descriptives.records.causal_fraction, 0.341)",
        "the block-4/5 ladder (descriptives.ladder)",
        "the magnitudes, the even components and baselines, the half dose, the plurality control, nMSE, G against the random directions and the "
        "per-template counts (descriptives.records)",
        "any correlation, such as Spearman(A, s0) in the independent confirmation review; none is a preregistered test",
    ],
    "confirm_final_state_sha256": confirm_final_sha, "confirm_final_state_file_sha256": confirm_final_file_sha,
    "result_write_state_sha256": result_write_sha,
    "report_state_sha256": state["state_sha256"], "report_state_file_sha256": file_sha(state_path),
    "report_sha256": state["report"]["sha256"],
}
assert (official["outcome"], {name: entry["positive"] for name, entry in official["statistics"].items()}) == ("CAUSAL_EFFECT_NOT_ESTABLISHED", {"A": 26, "B": 23, "G": 37})
assert {entry["threshold"] for entry in official["statistics"].values()} == {27} and official["reference_tail"]["value"] == 0.01923865414210013
assert {name: entry["result"] for name, entry in official["statistics"].items()} == {"A": "FAIL", "B": "FAIL", "G": "PASS"}
assert abs(state["confirmation"]["descriptives"]["records"]["causal_fraction"]["fraction"] - 0.341) < 5e-4

extract = {
    "experiment": cr.EXPERIMENT,
    "kind": ("confirmation record extract: the recorded lock, confirmation and report of the gitignored final results state, with the prompt ledger "
             "replaced by its counts and digests, the accounting of the one confirm run, the confirmation isolation, the provenance of the exit-127 "
             "shell event, and a summary of the official result copied from the state"),
    "official_result": official,
    "source": {
        "path": "outputs/experiment-025/results.json", "file_sha256": file_sha(state_path), "state_sha256": state["state_sha256"], "run_id": state["run_id"],
        "created_at": state["created_at"], "schema_version": state["schema_version"], "git_dirty": state["git_dirty"],
        "state_protocol_code_commit": state["protocol_code_commit"],
        "freeze_commit": git("rev-parse", "c765148"), "freeze_install_commit": git("rev-parse", "57f1f98"), "lock_commit": lock["protocol_code_commit"],
        "lock_install_commit": git("rev-parse", "5ee7a74"), "confirm_commit": confirm_commit, "report_commit": report_commit,
        "phase_runs": {
            "freeze": run_of("freeze-2026-09-26/freeze_run.json", "started_at", "ended_at", git("rev-parse", "c765148")),
            "lock": run_of("lock-2026-09-26/lock_run.json", "started_at", "ended_at", lock["protocol_code_commit"]),
            "confirm": run_of("confirm-2026-09-26/production_run.json", "started_at", "ended_at", confirm_commit),
            "report": run_of("report-2026-09-26/report_record.json", "report_started_at", "report_ended_at", report_commit)},
        "design": state["design"], "plan": state["plan"], "configuration": state["configuration"], "model": state["model"], "versions": state["versions"],
        "module_blobs": state["module_blobs"], "analysis_module_blob": lock["module"], "input_digests": state["inputs"], "committed_artifacts": committed,
        "local_outputs": {name: {"file_sha256": file_sha(OUT / name), "bytes": (OUT / name).stat().st_size} for name in local},
    },
    "evidence_copies": {f"{EXPERIMENT_DIR}/evidence/{name}": {"file_sha256": file_sha(EVIDENCE / name), "byte_identical_to": f"outputs/experiment-025/{source}"}
                        for name, source in copies.items()},
    "phases": state["phases"],
    "ledger": {"executed_prompt_keys": len(executed), "executed_prompt_keys_sha256": canonical_sha(executed),
               "executed_prompt_keys_sorted_sha256": canonical_sha(sorted(executed)), "prompts": len(prompts), "prompts_sorted_sha256": canonical_sha(sorted(prompts)),
               "executed_noun_keys": len(state["executed_noun_keys"]), "executed_noun_keys_sha256": canonical_sha(state["executed_noun_keys"])},
    "prompt_accounting": accounting,
    "confirmation_isolation": isolation,
    "shell_event_before_the_launch_of_record": shell_event,
    "confirmation_025": state["confirmation_025"],
    "lock": state["lock"],
    "confirmation": json.loads(pm.canonical_json(state["confirmation"])),
    "report": state["report"],
}
extract["content_sha256"] = canonical_sha(extract)
TARGET.write_text(pm.canonical_json(extract) + "\n", encoding="utf-8")
check = json.loads(TARGET.read_text(encoding="utf-8"))
assert rc.content_digest(check) == check["content_sha256"] == extract["content_sha256"]
print(f"wrote {TARGET.relative_to(ROOT)}: {TARGET.stat().st_size} bytes, content sha256 {extract['content_sha256']}, file sha256 {file_sha(TARGET)}")
print("official", json.dumps({k: v for k, v in official.items() if k not in ("stage2_measurements", "not_claimed", "label_table", "descriptive_only", "gates")},
                             ensure_ascii=False))
print("accounting", json.dumps({k: v for k, v in accounting.items() if k in ("manifest_keys", "executed_runs", "executed_unique", "ledger_keys", "extra", "missing",
                                                                              "duplicated", "ledger_is_sorted", "runner_accounting")}))
print("isolation", json.dumps(isolation))
print("shell event", json.dumps({k: v for k, v in shell_event.items() if k != "what"}))
print("phase runs", json.dumps(extract["source"]["phase_runs"]))
