#!/usr/bin/env python3
"""Build the committed Experiment 023 confirmation-record extract from the gitignored results state. Read-only on the
state and on every artifact; it writes only the extract. It follows Experiment 022's closure extract:
- canonical JSON plus a newline;
- content digest = sha256(canonical_json(record without content_sha256));
- the 18 bulky stage-1 reference states are replaced by one digest of the full mapping (their per-frame digests stay in
  the stage-1 record, and the stage-1 digest covers them);
- the prompt ledger is replaced by its counts and digest.
It adds the prompt accounting of the one confirm run. No model, no prompt."""
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402

OUT = ROOT / "outputs/experiment-023"
EVIDENCE = ROOT / b0c.EXPERIMENT_DIR / "evidence"
TARGET = EVIDENCE / "confirmation-record-2026-09-24.json"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    return pm.sha256_text(pm.canonical_json(value))


def phase_run(relative: str) -> dict:
    text = (EVIDENCE / relative).read_text(encoding="utf-8")
    start = re.search(r"^start (\S+) at ([0-9a-f]{40})", text, re.M)
    end = re.search(r"^exit (\d+) end (\S+)$", text, re.M)
    return {"log": f"{b0c.EXPERIMENT_DIR}/evidence/{relative}", "started": start.group(1), "commit": start.group(2), "exit": int(end.group(1)), "ended": end.group(2)}


state_path = OUT / "results.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
assert pm.sha256_text(pm.canonical_json(unsigned)) == state["state_sha256"], "the results state's own digest does not verify"
stage1_full = state["confirmation"]["stage1"]
assert stage1_full["digest"] == b0c.stage_one_digest(stage1_full), "the stage-1 record does not reproduce its digest"
assert {k: v["status"] for k, v in state["phases"].items()} == {p: "complete" for p in b0c.STATE_PHASES}, "every state phase must be complete"
assert "incident" not in state["confirmation"], "a confirmation incident is recorded"

runner = run023.Runner()
inputs, confirmation_022, sha_022, digests, forbidden = runner._base()  # no model, no prompt
confirmation_set, confirmation_sha = runner._confirmation(inputs, confirmation_022, sha_022, forbidden)
assert state["inputs"] == {key: digests[key] for key in b0c.DIGEST_KEYS}, "the state's input digests are not the current frozen inputs"
lock = json.loads((ROOT / b0c.LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
record = json.loads((ROOT / b0c.CALIBRATION_RELATIVE_PATH).read_text(encoding="utf-8"))

confirmation = json.loads(pm.canonical_json(state["confirmation"]))  # a deep copy
stage1 = confirmation["stage1"]
states = stage1.pop("states")
assert stage1["state_digests"] == {frame_id: rd.state_digest(entry) for frame_id, entry in sorted(states.items())}
stage1["states_sha256"] = canonical_sha(states)
stage1["n_states"] = len(states)

executed = list(state["executed_prompt_keys"])
manifest = confirmation_set.manifest()
stage1_keys = set(manifest["S1-REF"]) | set(manifest["S1-VALIDITY"])
y1_keys, y2_keys = set(manifest["S2-TARGET"]["Y1"]), set(manifest["S2-TARGET"]["Y2"])
manifest_keys = stage1_keys | y1_keys | y2_keys
ledger_020 = set(inputs.closure["ledger"])
spent_021 = frozenset(prompt.key for prompt in inputs.confirmation_020.all_prompts)
manifest_022 = run023._manifest_keys_022(confirmation_022)
calls = [json.loads(line) for line in (EVIDENCE / "confirm-2026-09-24/prompts.jsonl").read_text(encoding="utf-8").splitlines()]
launcher = json.loads((EVIDENCE / "confirm-2026-09-24/confirm_summary.json").read_text(encoding="utf-8"))
call_keys = [call["key"] for call in calls]
assert len(set(executed)) == len(executed) == 3060 and set(executed) == manifest_keys
assert Counter(call_keys) == Counter({key: 1 for key in manifest_keys}) and [call["n"] for call in calls] == list(range(1, 3061))
first_target = min(i for i, key in enumerate(call_keys) if key not in stage1_keys)
accounting = {
    "rule": "expected manifest = executed capture_prompt calls = ledger spend",
    "manifest_keys": len(manifest_keys), "executed_calls": len(call_keys), "executed_unique": len(set(call_keys)), "ledger_keys": len(executed),
    "extra": len(set(call_keys) - manifest_keys), "missing": len(manifest_keys - set(call_keys)), "duplicated": sum(1 for n in Counter(call_keys).values() if n > 1),
    "stage1_calls_before_first_target": first_target, "launcher_counts": launcher["counts"], "launcher_exit": launcher["exit"],
    "prompts_log": {"path": f"{b0c.EXPERIMENT_DIR}/evidence/confirm-2026-09-24/prompts.jsonl", "file_sha256": file_sha(EVIDENCE / "confirm-2026-09-24/prompts.jsonl")},
    "table_022_accesses_refused_during_run": launcher["refused_during_run"],
}
isolation = {
    "confirmation_023_sha256": confirmation_set.content_sha256,
    "manifest": {"keys": len(manifest_keys), "executed": len(set(executed) & manifest_keys)},
    "stage1": {"keys": len(stage1_keys), "executed": len(set(executed) & stage1_keys)},
    "s2_target_split": {"Y1_exposed_frames": {"keys": len(y1_keys), "executed": len(set(executed) & y1_keys)},
                        "Y2_new_frames": {"keys": len(y2_keys), "executed": len(set(executed) & y2_keys)}},
    "experiment_020_ledger": {"keys": len(ledger_020), "executed": len(set(executed) & ledger_020)},
    "experiment_021_spent_set": {"keys": len(spent_021), "executed": len(set(executed) & spent_021)},
    "experiment_022_manifest": {"keys": len(manifest_022), "executed": len(set(executed) & manifest_022)},
    "keys_outside_the_manifest": len(set(executed) - manifest_keys),
    "checks_passed_in_confirm": ["validate_lock (tracked, clean tree, no scientific change since the lock; the calibration record and state bind the committed confirmation)",
                                 "Y2 table absent before confirm", "runtime and versions equal to Experiment 020's explore",
                                 "I7 (the committed Y1 table rebuilt bit for bit before any fresh prompt)",
                                 "stage 1 executed exactly the 18 S1-REF and 18 S1-VALIDITY prompts once each", "Y2 table gates I5 and block-0 algebra at stage 1",
                                 "barrier (stage-1 digest reproduced and equal to the in-memory one, lock, no S2-TARGET key in the ledger, Y2 table re-read and verified)",
                                 "stage 2 executed exactly the 3,024 S2-TARGET prompts once each", "every stage-2 measurement finite and saved before any gate",
                                 "Y2 table unchanged after stage 2", "I1, I3, I4 (stage 2)", "kernel/direct agreement",
                                 "Experiment 020 closure and Experiment 022 committed files re-check before any result was written"],
}
committed = {path: {"file_sha256": file_sha(ROOT / path)} for path in (b0c.CONFIRMATION_RELATIVE_PATH, b0c.CALIBRATION_RELATIVE_PATH, b0c.LOCK_RELATIVE_PATH,
                                                                   b0c.PREREGISTRATION_RELATIVE_PATH, b0c.Y1_TABLE_RELATIVE_PATH, b0c.Y1_TABLE_INDEX_RELATIVE_PATH,
                                                                   b0c.CELLS_DATA_RELATIVE_PATH, b0c.CELLS_INDEX_RELATIVE_PATH)}
committed[b0c.CONFIRMATION_RELATIVE_PATH]["content_sha256"] = confirmation_set.content_sha256
committed[b0c.CALIBRATION_RELATIVE_PATH]["content_sha256"] = record["content_sha256"]
committed[b0c.LOCK_RELATIVE_PATH]["content_sha256"] = lock["content_sha256"]
committed[b0c.CELLS_INDEX_RELATIVE_PATH]["content_sha256"] = json.loads((ROOT / b0c.CELLS_INDEX_RELATIVE_PATH).read_text(encoding="utf-8"))["content_sha256"]
local = ("results.json", "draw-values.pt", "stage2-measurements.pt", "y2-table.f64", "y2-table.json", "report.md", "candidate-exposed-cells.f64", "candidate-exposed-cells.json",
         "candidate-calibration.json", "candidate-lock.json", "candidate-preregistration.md", "candidate-locked-y1-table.f64", "candidate-locked-y1-table.json")
copies = {"final-report-2026-09-24.md": "report.md", "y2-table.f64": "y2-table.f64", "y2-table.json": "y2-table.json"}
for name, source in copies.items():
    assert (EVIDENCE / name).read_bytes() == (OUT / source).read_bytes(), f"{name} is not byte-identical to outputs/experiment-023/{source}"

extract = {
    "experiment": "023",
    "kind": ("confirmation record extract: the recorded extraction, calibration, lock, confirmation and report of the gitignored results state, with the 18 "
             "stage-1 reference states replaced by one digest (their per-frame digests stay in the stage-1 record), the prompt ledger by its counts and digest, "
             "and the prompt accounting of the one confirm run"),
    "source": {
        "path": "outputs/experiment-023/results.json", "file_sha256": file_sha(state_path), "state_sha256": state["state_sha256"], "run_id": state["run_id"],
        "created_at": state["created_at"], "state_protocol_code_commit": state["protocol_code_commit"], "extract_commit": state["phases"]["extract"]["commit"],
        "calibrate_commit": state["phases"]["calibrate"]["commit"], "lock_commit": lock["protocol_code_commit"], "confirm_commit": state["phases"]["confirm"]["confirm_commit"],
        "phase_runs": {"extract": phase_run("extract-2026-09-24/extract.out"), "freeze": phase_run("freeze-2026-09-24/freeze.out"),
                       "calibrate": phase_run("calibrate-2026-09-24/calibrate.out"), "lock": phase_run("lock-2026-09-24/lock.out"),
                       "confirm": phase_run("confirm-2026-09-24/confirm.out"), "report": phase_run("report-2026-09-24/report.out")},
        "design": state["design"], "plan": state["plan"], "model": state["model"], "versions": state["versions"], "module_blobs": state["module_blobs"],
        "input_digests": state["inputs"], "committed_artifacts": committed, "local_outputs": {name: file_sha(OUT / name) for name in local},
    },
    "evidence_copies": {f"{b0c.EXPERIMENT_DIR}/evidence/{name}": {"file_sha256": file_sha(EVIDENCE / name), "byte_identical_to": f"outputs/experiment-023/{source}"}
                        for name, source in copies.items()},
    "phases": state["phases"],
    "ledger": {"executed_prompt_keys": len(executed), "executed_prompt_keys_sha256": canonical_sha(executed),
               "executed_noun_keys": len(state["executed_noun_keys"]), "executed_noun_keys_sha256": canonical_sha(state["executed_noun_keys"])},
    "prompt_accounting": accounting,
    "confirmation_isolation": isolation,
    "extract": state["extract"],
    "calibration": state["calibration"],
    "confirmation_023": state["confirmation_023"],
    "lock": state["lock"],
    "confirmation": confirmation,
    "report": state["report"],
}
extract["content_sha256"] = canonical_sha(extract)
TARGET.write_text(pm.canonical_json(extract) + "\n", encoding="utf-8")
check = json.loads(TARGET.read_text(encoding="utf-8"))
assert rc.content_digest(check) == check["content_sha256"] == extract["content_sha256"]
print(f"wrote {TARGET.relative_to(ROOT)}: {TARGET.stat().st_size} bytes, content sha256 {extract['content_sha256']}, file sha256 {file_sha(TARGET)}")
print("accounting", json.dumps({k: v for k, v in accounting.items() if k != "prompts_log"}))
print("isolation", json.dumps({k: v for k, v in isolation.items() if k != "checks_passed_in_confirm"}))
print("phase runs", json.dumps(extract["source"]["phase_runs"]))
