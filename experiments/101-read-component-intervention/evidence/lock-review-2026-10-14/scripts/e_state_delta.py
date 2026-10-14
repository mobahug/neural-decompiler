"""The lock changed only the state's `lock` block and `phases.lock`: undo exactly those two and the state must be the
calibrated state the calibration review recorded (state_sha256 729c9cb5…, file sha256 935a0df2…, also the operator's
state_before_dryrun.txt)."""
from common import install_fs_guard

install_fs_guard()
import hashlib, json  # noqa: E402,E401
from common import ROOT, canonical, sha256_text, write_json, EVENTS  # noqa: E402

state = json.loads((ROOT / "outputs/experiment-101/results.json").read_bytes())
state.pop("state_sha256")
before = dict(state)
before["lock"] = None
before["phases"] = {**state["phases"], "lock": {"status": "not_started"}}
digest = sha256_text(canonical(before))
file_text = canonical({**before, "state_sha256": digest}) + "\n"
extract = json.loads((ROOT / "experiments/101-read-component-intervention/evidence/calibrate-2026-10-14/state_extract.json").read_text())
out = {"reconstructed_pre_lock_state_sha256": digest, "reconstructed_pre_lock_file_sha256": hashlib.sha256(file_text.encode("utf-8")).hexdigest(),
       "operator_state_before_dryrun": (ROOT.parent / "x").name and open("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/lock/state_before_dryrun.txt").read().split()[0],
       "calibration_review_state_sha256_prefix": "729c9cb5", "state_extract_keys": sorted(extract)[:20]}
for key in ("state_sha256", "results_state_sha256", "results_file_sha256", "file_sha256"):
    if key in extract:
        out[f"extract_{key}"] = extract[key]
out["events"] = EVENTS
write_json("e_state_delta.json", out)
print(json.dumps(out, indent=1))
