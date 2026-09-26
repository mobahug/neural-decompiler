#!/usr/bin/env python3
"""Build the committed Experiment 024 confirmation-record extract from the gitignored final results state.

It is read-only on the state and on every artifact, and writes only the extract; an audit hook refuses any other
repository write. It follows Experiment 022's and 023's closure extract:
- canonical JSON plus a newline;
- content digest = sha256(canonical_json(record without content_sha256));
- the prompt ledger is replaced by its counts and digests.

It adds the prompt accounting of the one confirm run and a summary of the official result. The summary is copied from
the recorded state, not recomputed or reinterpreted. No model, no prompt, no phase.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "experiments/024-readout-routing-nounness/evidence"
TARGET = EVIDENCE / "confirmation-record-2026-09-25.json"
OUT = ROOT / "outputs/experiment-024"


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    raw = os.fsdecode(args[0])
    if not writing or not os.path.isabs(raw):
        return
    path = Path(raw).resolve()
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path) and path != TARGET.resolve():
        raise PermissionError(f"the extract builder may write only {TARGET}: {path}")


sys.addaudithook(_hook)
spec = importlib.util.spec_from_file_location("run024", ROOT / "experiments/024-readout-routing-nounness/run.py")
run024 = importlib.util.module_from_spec(spec)
sys.modules["run024"] = run024
spec.loader.exec_module(run024)

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    return pm.sha256_text(pm.canonical_json(value))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, env=GIT_ENV).stdout.strip()


def evidence_json(relative: str) -> dict:
    return json.loads((EVIDENCE / relative).read_text(encoding="utf-8"))


def refuse(*args, **kwargs):
    raise RuntimeError("the extract builder never loads a model or a tokenizer")


state_path = OUT / "results.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
assert canonical_sha(unsigned) == state["state_sha256"], "the results state's own digest does not verify"
assert {k: v["status"] for k, v in state["phases"].items()} == {p: "complete" for p in rr.STATE_PHASES}, "every state phase must be complete"
assert not any(entry.get("incidents") for entry in state["phases"].values()) and "incident" not in state["confirmation"], "an incident is recorded"
assert not (state["calibration"].get("incidents") or state["calibration"].get("stop")), "a calibration incident or stop is recorded"
assert not (state["confirmation"].get("descriptives") or {}).get("failures"), "a descriptive failure is recorded"

# the confirm-final state: the final state with only the two report entries reverted (the report run added nothing else)
confirm_final = {**unsigned, "report": None, "phases": {**unsigned["phases"], "report": {"status": "not_started"}}}
confirm_final_sha = canonical_sha(confirm_final)
confirm_final_file_sha = hashlib.sha256((pm.canonical_json({**confirm_final, "state_sha256": confirm_final_sha}) + "\n").encode("utf-8")).hexdigest()
report_run = evidence_json("report-2026-09-25/report_record.json")
assert confirm_final_sha == report_run["pre_report_check"]["state"] and confirm_final_file_sha == report_run["pre_report_check"]["state_file"], \
    "the confirm-final state recorded before the report run is not reproduced"

runner = run024.Runner(log=lambda message: None, model_loader=refuse, tokenizer_loader=refuse)
base = runner._base()  # the frozen inputs; no model, no prompt
confirmation_set, confirmation_file_sha = runner._confirmation(base)
assert state["inputs"] == {key: base.digests[key] for key in rr.DIGEST_KEYS}, "the state's input digests are not the current frozen inputs"
lock = json.loads((ROOT / rr.LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
record = json.loads((ROOT / rr.CALIBRATION_RELATIVE_PATH).read_text(encoding="utf-8"))
assert state["lock"]["content_sha256"] == lock["content_sha256"] and state["confirmation"]["lock_sha256"] == lock["content_sha256"]

# the prompt accounting of the one confirm run (the launcher's prompt log and counts, against the manifest and the ledger)
executed = list(state["executed_prompt_keys"])
manifest = confirmation_set.manifest()["S2-TARGET"]
manifest_keys = set(manifest)
calls = [json.loads(line) for line in (EVIDENCE / "confirm-2026-09-25/confirm_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
launcher = evidence_json("confirm-2026-09-25/confirm_record.json")
call_keys = [call["key"] for call in calls]
assert len(set(executed)) == len(executed) == len(manifest_keys) == 4_320 and set(executed) == manifest_keys
assert Counter(call_keys) == Counter({key: 1 for key in manifest_keys}) and [call["n"] for call in calls] == list(range(1, 4_321))
accounting = {
    "rule": "expected manifest = executed capture_prompt calls = ledger spend",
    "manifest_keys": len(manifest_keys), "executed_calls": len(call_keys), "executed_unique": len(set(call_keys)), "ledger_keys": len(executed),
    "extra": len(set(call_keys) - manifest_keys), "missing": len(manifest_keys - set(call_keys)), "duplicated": sum(1 for n in Counter(call_keys).values() if n > 1),
    "runner_accounting": state["confirmation"]["accounting"], "launcher_counts": launcher["counts"], "launcher_exit": launcher["exit_status"],
    "launcher_error": launcher["error"], "launcher_events": {key: value for key, value in launcher["events"].items() if key != "output_writes"},
    "launcher_output_writes_in_first_write_order": list(launcher["events"]["output_writes"]),
    "module_blob_assertion": launcher["module_blob_assertion"],
    "prompts_log": {"path": "experiments/024-readout-routing-nounness/evidence/confirm-2026-09-25/confirm_prompts.jsonl",
                    "file_sha256": file_sha(EVIDENCE / "confirm-2026-09-25/confirm_prompts.jsonl"), "sequence_sha256": canonical_sha(call_keys)},
}
ledger_020 = set(base.inputs.closure["ledger"])
spent_021 = {prompt.key for prompt in base.inputs.confirmation_020.all_prompts}
manifest_022 = run024._manifest_keys_022(json.loads((ROOT / ul.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8")))
manifest_023 = run024._manifest_keys_022(base.confirmation_023)
assert (ledger_020 | spent_021 | manifest_022 | manifest_023) == base.forbidden and len(base.forbidden) == 39_312
isolation = {
    "confirmation_024_content_sha256": confirmation_set.content_sha256,
    "manifest": {"keys": len(manifest_keys), "executed": len(set(executed) & manifest_keys), "sha256": canonical_sha(confirmation_set.manifest())},
    "experiment_020_ledger": {"keys": len(ledger_020), "executed": len(set(executed) & ledger_020)},
    "experiment_021_spent_set": {"keys": len(spent_021), "executed": len(set(executed) & spent_021)},
    "experiment_022_manifest": {"keys": len(manifest_022), "executed": len(set(executed) & manifest_022)},
    "experiment_023_manifest": {"keys": len(manifest_023), "executed": len(set(executed) & manifest_023)},
    "spent_union": {"keys": len(base.forbidden), "executed": len(set(executed) & base.forbidden)},
    "keys_outside_the_manifest": len(set(executed) - manifest_keys),
}

committed = {path: {"file_sha256": file_sha(ROOT / path)} for path in (rr.CONFIRMATION_RELATIVE_PATH, rr.CALIBRATION_RELATIVE_PATH, rr.LOCK_RELATIVE_PATH,
                                                                   rr.PREREGISTRATION_RELATIVE_PATH)}
committed[rr.CONFIRMATION_RELATIVE_PATH]["content_sha256"] = confirmation_set.content_sha256
committed[rr.CALIBRATION_RELATIVE_PATH]["content_sha256"] = record["content_sha256"]
committed[rr.LOCK_RELATIVE_PATH]["content_sha256"] = lock["content_sha256"]
local = ("results.json", "stage2-measurements.pt", "report.md", "calibration-arrays.pt", "candidate-calibration.json", "candidate-lock.json", "candidate-preregistration.md")
copies = {"final-report-2026-09-25.md": "report.md"}
for name, source in copies.items():
    assert (EVIDENCE / name).read_bytes() == (OUT / source).read_bytes(), f"{name} is not byte-identical to outputs/experiment-024/{source}"


def run_of(relative: str, started: str, ended: str, commit: str) -> dict:
    entry = evidence_json(relative)
    return {"record": f"experiments/024-readout-routing-nounness/evidence/{relative}", "record_sha256": file_sha(EVIDENCE / relative), "commit": commit,
            "started": entry[started], "ended": entry[ended], "exit": entry["exit_status"], "error": entry["error"]}


results = state["confirmation"]["results"]
primary, guard = results["primary"], results["guard"]
confirm_commit = state["phases"]["confirm"]["confirm_commit"]
report_commit = report_run["pre_report_check"]["head"]
official = {  # copied from the recorded state and the evidence, never recomputed
    "experiment": rr.EXPERIMENT, "confirmation_commit": confirm_commit, "report_commit": report_commit,
    "stage2_measurements": {"path": "outputs/experiment-024/stage2-measurements.pt", "file_sha256": file_sha(OUT / "stage2-measurements.pt"),
                            "bytes": (OUT / "stage2-measurements.pt").stat().st_size, "tensors_sha256": state["confirmation"]["stage2"]["tensors_sha256"],
                            "n_executed": state["confirmation"]["stage2"]["n_executed"]},
    "prompts": len(executed), "manifest_sha256": lock["confirmation_024"]["manifest_sha256"],
    "primary": {"rho": primary["rho"], "F_rho": primary["F_rho"], "null_975": primary["null_975"], "effective_threshold": primary["effective_threshold"],
                "result": primary["result"]},
    "guard": {"D_EN": guard["D_EN"], "D_EN_exact": guard["D_EN_exact"], "K": guard["K"], "assignments": guard["assignments"], "max_upper": guard["max_upper"],
              "p_exact": guard["p_exact"], "result": guard["result"]},
    "outcome": results["outcome"]["label"],
    "incident": None,
    "confirm_final_state_sha256": confirm_final_sha, "confirm_final_state_file_sha256": confirm_final_file_sha,
    "result_write_state_sha256": canonical_sha({k: v for k, v in confirm_final.items() if k != "confirmation"}
                                               | {"confirmation": {k: v for k, v in confirm_final["confirmation"].items() if k != "descriptives"}}),
    "report_state_sha256": state["state_sha256"], "report_state_file_sha256": file_sha(state_path),
    "report_sha256": state["report"]["sha256"],
    "pass_rule": "ρ ≥ max(F_ρ, null₉₇.₅); for this realized calibration the chance-level null bound (null₉₇.₅ > F_ρ)",
}
assert official["result_write_state_sha256"] == "37f2f88b7f0fdde73ebbedc319fc5d9edcab59c35133a8cfa8b24c4e3201f9fd", "the result-write state is not reproduced"

extract = {
    "experiment": rr.EXPERIMENT,
    "kind": ("confirmation record extract: the recorded calibration, lock, confirmation and report of the gitignored final results state, with the prompt "
             "ledger replaced by its counts and digests, the prompt accounting of the one confirm run, and a summary of the official result copied "
             "from the state"),
    "official_result": official,
    "source": {
        "path": "outputs/experiment-024/results.json", "file_sha256": file_sha(state_path), "state_sha256": state["state_sha256"], "run_id": state["run_id"],
        "created_at": state["created_at"], "state_protocol_code_commit": state["protocol_code_commit"],
        "freeze_commit": git("rev-parse", "44c3c2b"), "calibrate_commit": state["phases"]["calibrate"]["commit"], "lock_commit": lock["protocol_code_commit"],
        "confirm_commit": confirm_commit, "report_commit": report_commit,
        "phase_runs": {
            "freeze": run_of("freeze-2026-09-25/freeze_run.json", "started_at", "ended_at", git("rev-parse", "44c3c2b")),
            "calibrate": run_of("calibrate-2026-09-25/calibrate_run.json", "started_at", "ended_at", state["phases"]["calibrate"]["commit"]),
            "lock": run_of("lock-2026-09-25/lock_run.json", "started_at", "ended_at", lock["protocol_code_commit"]),
            "confirm": run_of("confirm-2026-09-25/confirm_record.json", "confirm_started_at", "confirm_ended_at", confirm_commit),
            "report": run_of("report-2026-09-25/report_record.json", "report_started_at", "report_ended_at", report_commit)},
        "design": state["design"], "plan": state["plan"], "configuration": state["configuration"], "model": state["model"], "versions": state["versions"],
        "module_blobs": state["module_blobs"], "analysis_module_blob": lock["module"], "input_digests": state["inputs"], "committed_artifacts": committed,
        "local_outputs": {name: {"file_sha256": file_sha(OUT / name), "bytes": (OUT / name).stat().st_size} for name in local},
    },
    "evidence_copies": {f"experiments/024-readout-routing-nounness/evidence/{name}": {"file_sha256": file_sha(EVIDENCE / name),
                                                                                      "byte_identical_to": f"outputs/experiment-024/{source}"}
                        for name, source in copies.items()},
    "phases": state["phases"],
    "ledger": {"executed_prompt_keys": len(executed), "executed_prompt_keys_sha256": canonical_sha(executed),
               "executed_prompt_keys_sorted_sha256": canonical_sha(sorted(executed)),
               "executed_noun_keys": len(state["executed_noun_keys"]), "executed_noun_keys_sha256": canonical_sha(state["executed_noun_keys"])},
    "prompt_accounting": accounting,
    "confirmation_isolation": isolation,
    "calibration": state["calibration"],
    "confirmation_024": state["confirmation_024"],
    "lock": state["lock"],
    "confirmation": json.loads(pm.canonical_json(state["confirmation"])),
    "report": state["report"],
}
extract["content_sha256"] = canonical_sha(extract)
TARGET.write_text(pm.canonical_json(extract) + "\n", encoding="utf-8")
check = json.loads(TARGET.read_text(encoding="utf-8"))
assert rc.content_digest(check) == check["content_sha256"] == extract["content_sha256"]
print(f"wrote {TARGET.relative_to(ROOT)}: {TARGET.stat().st_size} bytes, content sha256 {extract['content_sha256']}, file sha256 {file_sha(TARGET)}")
print("official", json.dumps({k: v for k, v in official.items() if k != "stage2_measurements"}, ensure_ascii=False))
print("accounting", json.dumps({k: v for k, v in accounting.items() if k not in ("prompts_log", "module_blob_assertion", "runner_accounting")}))
print("isolation", json.dumps(isolation))
print("phase runs", json.dumps(extract["source"]["phase_runs"]))
