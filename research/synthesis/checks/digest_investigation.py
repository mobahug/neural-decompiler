"""Why do Experiment 010's and 012's README results-state digests differ from the digests other records carry?

Read-only: it reads the local, gitignored final results states of 010 and 012 and committed files, and writes
nothing into the repository (an audit hook refuses any write there). No model, no prompt, no phase.

Each state digest is sha256(canonical_json(state without state_sha256)), as write_results_state writes it. The script
rebuilds earlier states of the same run from the final state, by undoing only what later phases are known (from the
runner code) to have written, and compares their digests with every recorded value.
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
os.chdir(ROOT)


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        raw = os.fsdecode(args[0])
        path = Path(raw if os.path.isabs(raw) else os.path.join(os.getcwd(), raw)).resolve()
        if writing and str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
            raise PermissionError(f"read-only investigation: {path}")


sys.addaudithook(_hook)
from neural_decompiler import plural_mechanism as pm  # noqa: E402


def digest(state):
    return pm.state_digest(state)


def show(label, value, expected=None):
    mark = "" if expected is None else ("  == " + expected[:12] + "…  MATCH" if value == expected else "  != " + expected[:12] + "…  no match")
    print(f"  {label}: {value}{mark}")
    return value == expected


print("Experiment 010 (read_assembly.write_results_state; report() rewrites the state with phases.report complete)")
final = json.loads((ROOT / "outputs/experiment-010/results.json").read_text(encoding="utf-8"))
import re  # noqa: E402


def logged_digest(experiment: str) -> str:
    """The digest the explore phase printed at its end, read from the local, gitignored explore log."""
    log = (ROOT / f"outputs/experiment-{experiment}/explore.log").read_text(encoding="utf-8")
    return re.findall(r"explore complete: .*results sha256 ([0-9a-f]{64})", log)[-1]


readme_010 = "a9f8d7031207becd4d0390fb0e7ea4d7c37d28e6a500950fed0033d69a27c3da"  # 010 README:50
print(f"  010 explore log digest: {logged_digest('010')}  ({'equals' if logged_digest('010') == readme_010 else 'differs from'} the README value)")
extract_010 = {rel: json.loads((ROOT / rel).read_text(encoding="utf-8"))["source"]["state_sha256"] for rel in (
    "experiments/011-encoding-read-prospective/inherited/experiment-010-transport-fractions.json",
    "experiments/012-layer-correction-token-local/inherited/experiment-010-component-fractions.json")}
ok = [show("final state, recorded", final["state_sha256"]), show("final state, recomputed", digest(final), final["state_sha256"])]
for rel, value in extract_010.items():
    ok.append(show(f"extract {rel.split('/')[-1]}", value, final["state_sha256"]))
print(f"  final phases.report: {final['phases']['report']}")
before_report = copy.deepcopy(final)
before_report.pop("state_sha256")
before_report["phases"]["report"] = {"status": "not_started"}
ok.append(show("final state with phases.report reverted to not_started", digest(before_report), readme_010))
print(f"  010 verdict: {'EXPLAINED' if all(ok[1:]) else 'NOT EXPLAINED'}")

print("\nExperiment 012 (layer_correction.write_results_state; lock() writes state['lock'] and phases.lock; confirm() writes the")
print("fresh ledger keys, phases.confirm and state['confirmation']; report() does not write the state)")
final = json.loads((ROOT / "outputs/experiment-012/results.json").read_text(encoding="utf-8"))
readme_012 = "069a6306d1325dcd6efd7f2aeac330f26baa2c765021e65d5a4480f1252f381a"  # 012 README:130-131
log_012 = logged_digest("012")  # read from outputs/experiment-012/explore.log
lock_012 = json.loads((ROOT / "experiments/012-layer-correction-token-local/preregistration-lock.json").read_text(encoding="utf-8"))["tier_a_results_sha256"]
confirmation = json.loads((ROOT / "experiments/012-layer-correction-token-local/confirmation-v1.json").read_text(encoding="utf-8"))
fresh_frames = {frame["frame_id"] for frame in confirmation["frames"]}
ok = [show("final state, recomputed", digest(final), final["state_sha256"]), show("installed lock tier_a_results_sha256", lock_012, log_012)]
tier_a_keys = [key for key in final["executed_prompt_keys"] if key.split("|")[0] not in fresh_frames]
print(f"  ledger: final {len(final['executed_prompt_keys'])} keys; without the {len(fresh_frames)} fresh frames' keys {len(tier_a_keys)} (README:133-134 says 90)")
after_lock = copy.deepcopy(final)
after_lock.pop("state_sha256")
after_lock["phases"]["confirm"] = {"status": "not_started"}
after_lock["confirmation"] = None
after_lock["executed_prompt_keys"] = tier_a_keys
explore_end = copy.deepcopy(after_lock)
explore_end["phases"]["lock"] = {"status": "not_started"}
explore_end["lock"] = None
ok.append(len(tier_a_keys) == 90)
ok.append(show("rebuilt state at the end of explore", digest(explore_end), log_012))
ok.append(show("rebuilt state after the lock phase", digest(after_lock), readme_012))
print(f"  state['lock'] (written by lock, unchanged by confirm): {json.dumps(final['lock'])[:200]}")
print(f"  012 verdict: {'EXPLAINED' if all(ok) else 'NOT EXPLAINED'}")
print("\nfile sha256 of the local states (for the record):",
      {exp: hashlib.sha256((ROOT / f'outputs/experiment-{exp}/results.json').read_bytes()).hexdigest() for exp in ("010", "012")})
