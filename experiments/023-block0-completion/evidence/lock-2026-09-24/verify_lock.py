"""Experiment 023 post-lock verification: strictly read-only (no model)."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HEAD = "13b8d392f9e88844e2673c530a37228337cc64b7"
D = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/lock023")
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


guard = json.loads((D / "lock_files.json").read_text())
check("guard: one model load, forward refusal installed, 022 table never touched", guard["exit"] == 0 and guard["model_loads"] == ["EleutherAI/pythia-70m-deduped"]
      and guard["forward_refusal_installed"] and guard["refused_during_run"] == [])
hf_blobs = sorted(Path(p).name for p in guard["opened"] if "/huggingface/" in p and "/blobs/" in p)
print("   HF blobs opened:", hf_blobs)

out = ROOT / "outputs/experiment-023"
state = rd.load_results_state(out / "results.json")
lock_state = state["lock"]
print("state_sha256", state["state_sha256"], "| results.json file", rc.file_sha256(out / "results.json"))
check("lock complete once at HEAD; confirm/report not_started; no incident; ledgers empty", state["phases"]["lock"]["status"] == "complete"
      and state["phases"]["lock"]["commit"] == HEAD and not state["phases"]["lock"].get("incidents") and state["phases"]["confirm"]["status"] == "not_started"
      and state["phases"]["report"]["status"] == "not_started" and state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [],
      state["phases"]["lock"]["completed_at"])

paths = {"lock": out / "candidate-lock.json", "prereg": out / "candidate-preregistration.md", "y1": out / "candidate-locked-y1-table.f64", "y1i": out / "candidate-locked-y1-table.json"}
digest = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
sizes = {k: p.stat().st_size for k, p in paths.items()}
for k in paths:
    print(f"   {paths[k].name}: {sizes[k]} bytes, sha256 {digest[k]}")
lock = json.loads(paths["lock"].read_text())
check("lock content digest == state", lock["content_sha256"] == rc.content_digest(lock) == lock_state["content_sha256"], lock["content_sha256"])
check("lock file is canonical JSON + newline", paths["lock"].read_text() == pm.canonical_json(lock) + "\n")
check("preregistration == render(lock), sha == state", paths["prereg"].read_text() == b0c.render_preregistration(lock) and pm.sha256_text(paths["prereg"].read_text())
      == lock_state["preregistration_sha256"])
check("Y1 table digests == state and lock", digest["y1"] == lock_state["y1_file_sha256"] == lock["y1_table"]["file_sha256"] and digest["y1i"] == lock_state["y1_index_sha256"]
      == lock["y1_table"]["index_sha256"])
runner = run023.Runner()
inputs, confirmation_022, sha_022, digests, forbidden = runner._base()
confirmation, confirmation_sha = runner._confirmation(inputs, confirmation_022, sha_022, forbidden)
record = json.loads((ROOT / b0c.CALIBRATION_RELATIVE_PATH).read_text())
noun_keys = runner._noun_keys(inputs)
check("lock header: experiment, run, commit, inputs, pins, design, plan, constants", (lock["experiment"], lock["run_id"], lock["protocol_code_commit"]) == ("023", state["run_id"], HEAD)
      and lock["inputs"] == dict(digests) and lock["module_blobs"] == b0c.FROZEN_BLOBS and lock["design"] == b0c.DESIGN and lock["plan"] == b0c.PLAN
      and lock["constants"] == b0c.record_constants(b0c.B))
check("lock binds the committed calibration record", lock["calibration"] == {"path": b0c.CALIBRATION_RELATIVE_PATH,
      "file_sha256": "94db6df26156d7a134e51f8afbf10707ca18030525f62dd08f02fb298e9ba03a", "content_sha256": record["content_sha256"]})
check("lock binds the committed exposed cells", lock["exposed_cells"] == record["exposed_cells"] == {"data_path": b0c.CELLS_DATA_RELATIVE_PATH, "index_path": b0c.CELLS_INDEX_RELATIVE_PATH,
      "data_sha256": "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4", "index_sha256": "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe"})
check("lock binds the committed confirmation (+ counts, manifest sizes)", lock["confirmation_023"] == {**b0c.confirmation_binding(confirmation, confirmation_sha),
      "counts": confirmation.counts(), "manifest_sizes": {"S1-REF": 18, "S1-VALIDITY": 18, "Y1": 2592, "Y2": 432}})
check("conditions == the record's envelopes + the guard", lock["conditions"] == b0c.lock_conditions(record))
check("semantics, program, nouns, exposed states", lock["semantics"] == b0c.SEMANTICS and lock["program"] == {"P0_mask": 0, "P1_mask": {"cue_final": 14, "coordinated": 30},
      "construction": b0c.TABLE_CONSTRUCTION} and lock["noun_keys"] == noun_keys and len(noun_keys) == 79
      and lock["exposed_states_sha256"] == ul.exposed_states_digest(inputs.closure["exploration"]["locked_states"]))
check("Y2 specification == the frozen construction for the committed confirmation", lock["y2_table"] == b0c.y2_table_spec(confirmation, noun_keys),
      json.dumps(lock["y2_table"]["layout"]))
units = ul.table_units(confirmation.tokens, confirmation.exposed_frames)
index = json.loads(paths["y1i"].read_text())
ul.verify_table_index(index, layout=lock["y1_table"]["layout"], meta=b0c.table_meta("Y1", units, noun_keys), error=b0c.PhaseError)
blocks = ul.read_table(paths["y1"], index)
check("Y1 table: layout [1728, 2, 79] + [864, 2, 79], 3,276,288 bytes, bytes == index, meta == frozen", lock["y1_table"]["layout"] ==
      [{"name": "cue_final", "shape": [1728, 2, 79]}, {"name": "coordinated", "shape": [864, 2, 79]}] and sizes["y1"] == 3_276_288 == index["total_bytes"]
      and all(bool(v.isfinite().all()) for v in blocks.values()))
gates = lock["y1_table"]["gates"]
check("I5 exactly 0; block-0 algebra ≤ 1e-12", gates["I5"]["max"] == 0.0 and gates["algebra"]["max"] <= 1e-12, json.dumps(gates))
print("   conditions:", json.dumps({c: e["envelope"]["bound"] for c, e in lock["conditions"].items()}))
check("HEAD unchanged; tree clean; nothing installed", git("rev-parse", "HEAD") == HEAD and git("status", "--porcelain", "--untracked-files=all") == ""
      and not any((ROOT / p).exists() for p in (b0c.LOCK_RELATIVE_PATH, b0c.PREREGISTRATION_RELATIVE_PATH, b0c.Y1_TABLE_RELATIVE_PATH, b0c.Y1_TABLE_INDEX_RELATIVE_PATH)))
check("no Y2 table and no stage-2 file yet", not (ROOT / b0c.Y2_TABLE_OUTPUT).exists() and not (out / "stage2-measurements.pt").exists())
print("\nVERIFY LOCK:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
