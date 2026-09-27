"""Experiment 025: read-only verification of the one production report (no model, no prompt, no phase; nothing
written in the repository).

It checks:
- `report.md` and the state's report entry;
- removing only the two report entries reproduces the pre-report state `1d5d37f0…` / `c827bb69…` byte for byte;
- the frozen renderer, applied to that state, reproduces `report.md` byte for byte;
- the measurements, ledger, lock, preregistration and freeze are unchanged;
- the constrained wording.
"""
import hashlib
import json
import os
from pathlib import Path

from neural_decompiler import cue_rotation as cr
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_decompilation as rd

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
os.chdir(ROOT)
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/experiment-025"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


report_bytes = (OUT / "report.md").read_bytes()
final = rd.load_results_state(OUT / "results.json")
unsigned = {k: v for k, v in final.items() if k != "state_sha256"}
before = {**unsigned, "report": None, "phases": {**unsigned["phases"], "report": {"status": "not_started"}}}
before_sha = pm.sha256_text(pm.canonical_json(before))
before_file = (pm.canonical_json({**before, "state_sha256": before_sha}) + "\n").encode("utf-8")
text = report_bytes.decode("utf-8")
checks = {
    "report_sha256_in_state": final["report"]["sha256"] == hashlib.sha256(report_bytes).hexdigest() == pm.sha256_text(text),
    "phases_all_complete": {k: v["status"] for k, v in final["phases"].items()} == {"lock": "complete", "confirm": "complete", "report": "complete"},
    "no_incident": not (final.get("confirmation") or {}).get("incident") and not any(v.get("incidents") for v in final["phases"].values()),
    "reversal_reproduces_pre_report_state": before_sha == "1d5d37f035bc3153fec6efac527b088f64cd42fcd74e59680f4388daea98b33c"
    and hashlib.sha256(before_file).hexdigest() == "c827bb69292f11ff6baf2194b4d8e34ccb0fbad9058c76d1abf2a3581724efb4",
    "reversal_equals_launcher_copy": before_file == (HERE / "results-before-report.json").read_bytes(),
    "frozen_renderer_reproduces_report": cr.render_report({**before, "state_sha256": before_sha}).encode("utf-8") == report_bytes,
    "only_report_entries_changed": {k: v for k, v in unsigned.items() if k not in ("report", "phases")} == {k: v for k, v in before.items() if k not in ("report", "phases")}
    and {k: v for k, v in unsigned["phases"].items() if k != "report"} == {k: v for k, v in before["phases"].items() if k != "report"},
    "measurements_unchanged": sha(OUT / "stage2-measurements.pt") == "87284bddfba16a1b7709d7d33f98aa1a0898b1d85767956a524d2950d80a7788",
    "ledger_unchanged_90720": len(final["executed_prompt_keys"]) == 90720 and final["confirmation"]["accounting"]["equal"] is True,
    "installed_artifacts_unchanged": sha(ROOT / cr.LOCK_RELATIVE_PATH) == "1881a79008f905f0331a4396757d69722b6cc22a9b10c06184d79c97ebc843be"
    and sha(ROOT / cr.PREREGISTRATION_RELATIVE_PATH) == "e4b12656b877fe1d3dbc2eec314c03c6b2e228d3acadcd146eb179bbf02df919"
    and sha(ROOT / cr.CONFIRMATION_RELATIVE_PATH) == "54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a",
    "wording_label_and_reading": "**`CAUSAL_EFFECT_NOT_ESTABLISHED`** — A failed: the +nounness intervention did not consistently produce greater frozen-readout "
                                 "approximation error than the −nounness intervention." in text,
    "wording_counts": all(row in text for row in ("| A | 26 | 40 | 27 | **FAIL** | 20 | 6 |", "| B | 23 | 40 | 27 | **FAIL** | 20 | 3 |",
                                                  "| G | 37 | 40 | 27 | **PASS** | 20 | 17 |")),
    "wording_tail": "reference tail 5288280983/274877906944 = 0.01923865414210013" in text,
    "wording_no_forbidden_phrases": not any(phrase in text.lower() for phrase in ("no causal effect", "near miss", "near-miss")),
    "descriptives_marked_no_outcome_force": "## Descriptive records (no outcome force)" in text,
}
out = {"checks": checks, "all": all(checks.values()), "report_md": {"bytes": len(report_bytes), "sha256": hashlib.sha256(report_bytes).hexdigest()},
       "final_state": {"state_sha256": final["state_sha256"], "file_sha256": sha(OUT / "results.json")}, "pre_report_state": before_sha}
print(json.dumps(out, indent=1))
