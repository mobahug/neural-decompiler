"""Experiment 023 post-extract verification: strictly read-only (no model, no prompt, writes nothing in the repo)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
EXPECTED_HEAD = "2111271892a8237f69c3ef18d6eec774e798ac5c"
from neural_decompiler import block0_completion as b0c  # noqa: E402
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


out = ROOT / "outputs/experiment-023"
results = out / "results.json"
data, index_path = out / "candidate-exposed-cells.f64", out / "candidate-exposed-cells.json"
print("outputs/experiment-023:", sorted(p.name for p in out.iterdir()))

# The results state
state = rd.load_results_state(results)  # verifies state_sha256
check("results state digest verifies", True, state["state_sha256"])
check("results.json file sha256", True, rc.file_sha256(results))
extract = state["phases"]["extract"]
check("extract status complete", extract["status"] == "complete", extract["status"])
check("extract commit == approved HEAD", extract["commit"] == state["protocol_code_commit"] == EXPECTED_HEAD, extract["commit"])
check("no extract incident or stop", not extract.get("incidents") and not extract.get("stop"), json.dumps(extract.get("incidents")))
print("   extract started", extract["started_at"], "completed", extract["completed_at"], "runtime", json.dumps(extract["runtime"]))
check("other phases not started", all(state["phases"][p]["status"] == "not_started" for p in ("calibrate", "lock", "confirm", "report")),
      json.dumps({p: state["phases"][p]["status"] for p in state["phases"]}))
check("zero prompts in the ledger", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
check("state inputs are the frozen digests", state["inputs"]["table_022_file"] == b0c.INHERITED_022["table_file_sha256"]
      and state["inputs"]["calibration_022_content"] == b0c.INHERITED_022["calibration_content_sha256"], "")
check("state module blobs == pins", state["module_blobs"] == b0c.FROZEN_BLOBS)
print("   run_id", state["run_id"], "created_at", state["created_at"])

# The candidate files
data_sha, index_sha = rc.file_sha256(data), rc.file_sha256(index_path)
check("data sha256 == state", data_sha == state["extract"]["data_sha256"], data_sha)
check("index file sha256 == state", index_sha == state["extract"]["index_sha256"], index_sha)
check("data size 18,900 × 8 × 8 = 1,209,600 bytes", data.stat().st_size == 1_209_600, str(data.stat().st_size))
index = json.loads(index_path.read_text())
check("index content_sha256 verifies", index["content_sha256"] == rc.content_digest(index), index["content_sha256"])
check("index binds the data bytes", index["file_sha256"] == data_sha and index["total_bytes"] == 1_209_600)
check("index format and layout", (index["format"], index["dtype"], index["blocks"]) == (ul.TABLE_FORMAT, ul.TABLE_DTYPE,
      [{"name": "cells", "shape": [18900, 8], "offset_bytes": 0, "nbytes": 1_209_600}]), json.dumps(index["blocks"]))

# Every bound meta field against the one recomputed now from the frozen inputs and 022's committed record
inputs = ul.load_frozen_inputs(ROOT)
units = b0c.exposed_units(inputs)
noun_keys = [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]
record_022 = json.loads((ROOT / ul.CALIBRATION_RELATIVE_PATH).read_text())
meta = b0c.cells_meta(units, noun_keys, record_022)
cells, loaded = b0c.read_cells(data, index_path, units=units, meta=meta)
check("read_cells verifies every bound field", True, f"{len(meta)} meta keys: {sorted(meta)}")
check("columns", index["columns"] == ["n", "S", "Q", "SSE0", "SSE1", "SSEC", "mean", "M2"], json.dumps(index["columns"]))
check("pair order = 175 cues (022 order) × 108 frames, cue-major", len(index["cues"]) == 175 and len(index["frames"]) == 108, index["pair_order"])
strata = {}
for _, _, cls in index["cues"]:
    strata[cls] = strata.get(cls, 0) + 1
groups = {}
for _, template, group in index["frames"]:
    groups[(template, group)] = groups.get((template, group), 0) + 1
print("   cue strata", strata, "; frame template/group", {f"{t}/{g}": n for (t, g), n in groups.items()})
print("   y2-like frames per template", {t: len(v) for t, v in index["y2_like_frames"].items()}, "; nouns", len(index["nouns"]))
print("   program", json.dumps(index["program"]))
print("   source_022", json.dumps({k: v for k, v in index["source_022"].items() if k != "table_tensor_sha256"}))
check("source_022 tensor digests == 022 record's", index["source_022"]["table_tensor_sha256"] == record_022["rematerialization"]["table_sha256"],
      f"{len(index['source_022']['table_tensor_sha256'])} digests")
extraction = index["extraction"]
check("extraction module blob == block0_completion.py at HEAD", extraction["module_blob"] == git("rev-parse", "HEAD:src/neural_decompiler/block0_completion.py"),
      extraction["module_blob"])
check("extraction commit, version, run", (extraction["protocol_code_commit"], extraction["cells_version"], extraction["run_id"]) == (EXPECTED_HEAD, b0c.CELLS_VERSION, state["run_id"]),
      f"{extraction['protocol_code_commit']} {extraction['cells_version']} {extraction['run_id']} at {extraction['extracted_at']}")
check("extraction frozen blobs == pins", extraction["frozen_blobs"] == b0c.FROZEN_BLOBS)
checks = extraction["checks"]
check("extraction checks == state checks", checks == state["extract"]["checks"])
print("   E1", json.dumps(checks["E1"]))
print("   E2", json.dumps(checks["E2"]))
print("   E3", json.dumps(checks["E3"]))
print("   E4", json.dumps(checks["E4"]), "tolerance", b0c.TOLERANCES["E4"])
print("   E5", json.dumps(checks["E5"]), "tolerance", b0c.TOLERANCES["E5"])
print("   E6", json.dumps(checks["E6"]))
check("E1–E6 all passed", all(checks[k].get("passed") for k in ("E1", "E4", "E5", "E6")) and checks["E2"]["passed"] and checks["E3"]["passed"])

# The artifact reproduces all 18,900 pairs exactly, in canonical order (022's table read read-only)
table = torch.load(ROOT / b0c.TABLE_022_RELATIVE_PATH)
recomputed = b0c.cells_from_table(table, units)
check("artifact == cells recomputed from 022's table, bit for bit", torch.equal(cells, recomputed), f"{int((cells != recomputed).any(dim=1).sum())} pairs differ")
n_pairs = units.n_pairs
sse = table["sse"].reshape(n_pairs, -1)
p1 = torch.tensor([b0c.P1_MASK[units.group_of(i % len(units.frames))] for i in range(n_pairs)])
stored = {"n": table["count"].reshape(-1), "SSE0": sse[:, 0], "SSE1": sse[torch.arange(n_pairs), p1], "mean": table["mean"].reshape(-1), "M2": table["m2"].reshape(-1)}
for name, values in stored.items():
    check(f"artifact {name} == 022 stored (E2)", torch.equal(cells[:, b0c.CELL_COLUMNS.index(name)], values.double()))
independent = b0c.independent_sums(table, units)
for k, name in enumerate(b0c.E3_COLUMNS):
    check(f"artifact {name} == independent whole-table (E3)", torch.equal(cells[:, b0c.CELL_COLUMNS.index(name)], independent[:, k]))
check("artifact finite", bool(torch.isfinite(cells).all()))
check("n == 79 for every pair", bool((cells[:, 0] == 79).all()))
check("per-pair E6", b0c.enforce_e6(b0c.pool(cells, torch.arange(n_pairs).unsqueeze(1)), "verification") <= 1e-10)

# Nothing else moved
check("022 table file unchanged", rc.file_sha256(ROOT / b0c.TABLE_022_RELATIVE_PATH) == b0c.INHERITED_022["table_file_sha256"])
b0c.verify_022_inputs(ROOT)
rc.verify_020_closure(ROOT, inputs.confirmation_020)
check("022 committed files and 020 closure verify", True)
check("HEAD unchanged, tree clean", git("rev-parse", "HEAD") == EXPECTED_HEAD and git("status", "--porcelain", "--untracked-files=all") == "")
check("outputs/experiment-023 ignored by git", subprocess.run(["git", "check-ignore", "-q", str(results)], cwd=ROOT).returncode == 0)
check("no installed artifact yet", not (ROOT / b0c.CELLS_DATA_RELATIVE_PATH).exists() and not (ROOT / b0c.CELLS_INDEX_RELATIVE_PATH).exists())
print("\nVERIFY:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
