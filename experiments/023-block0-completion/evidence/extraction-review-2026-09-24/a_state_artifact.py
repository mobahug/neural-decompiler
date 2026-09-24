"""Items 1, 2 and 4 (file-level provenance): repository/protocol state, artifact integrity, 022 record and
confirmation digests, git blobs. Pure hashlib/json/numpy; no project code at all."""

from __future__ import annotations

import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/extractreview023")

import json

import numpy as np

from revlib import (CAND_DATA, CAND_INDEX, CONF022, EXPECTED, RECORD022, REPO, RESULTS, TABLE022, Report, canonical,
                    content_digest_without, git, git_blob_sha1, sha256_bytes, sha256_file, strict_json)

rep = Report("items 1, 2, 4 (files)")

# ---------------------------------------------------------------------------------------------------------------------
# Item 1: repository and protocol state.
head = git("rev-parse", "HEAD").decode().strip()
remote = git("ls-remote", "origin").decode()
remote_main = [line.split()[0] for line in remote.splitlines() if line.endswith("refs/heads/main")]
porcelain = git("status", "--porcelain", "--untracked-files=all").decode()
rep.check("HEAD == expected 2111271…", head == EXPECTED["head"], head)
rep.check("origin/main (ls-remote) == HEAD", remote_main == [head], str(remote_main))
rep.check("clean tree incl. untracked (porcelain empty)", porcelain == "", repr(porcelain[:200]))

raw_results = RESULTS.read_text(encoding="utf-8")
state = strict_json(RESULTS)  # refuses duplicate keys
rep.check("results.json has no duplicate keys and parses", True)
recomputed = sha256_bytes(canonical({k: v for k, v in state.items() if k != "state_sha256"}).encode("utf-8"))
rep.check("results.json state_sha256 recomputes (own canonical JSON)", recomputed == state["state_sha256"], f"{recomputed}")
rep.check("results.json bytes == canonical JSON + newline", raw_results == canonical(state) + "\n")
phases = state["phases"]
ext = phases["extract"]
rep.check("phases.extract.status == complete", ext.get("status") == "complete", str(ext.get("status")))
rep.check("extract has exactly one start and one completion, no incidents/stop",
          set(ext) == {"commit", "completed_at", "runtime", "started_at", "status"} and "incidents" not in ext and "stop" not in ext,
          f"keys {sorted(ext)}")
rep.check("extract phase commit == HEAD", ext.get("commit") == head)
for name in ("calibrate", "lock", "confirm", "report"):
    rep.check(f"phases.{name} == not_started (no other keys)", phases[name] == {"status": "not_started"}, str(phases[name]))
rep.check("phase set is exactly extract/calibrate/lock/confirm/report", sorted(phases) == sorted(["extract", "calibrate", "lock", "confirm", "report"]))
rep.check("executed_prompt_keys empty", state["executed_prompt_keys"] == [])
rep.check("executed_noun_keys empty", state["executed_noun_keys"] == [])
rep.check("calibration {} / lock, confirmation, confirmation_023, report null",
          state["calibration"] == {} and state["lock"] is None and state["confirmation"] is None and state["confirmation_023"] is None and state["report"] is None)
rep.check("state protocol_code_commit == HEAD, git_dirty false", state["protocol_code_commit"] == head and state["git_dirty"] is False)
data_sha = sha256_file(CAND_DATA)
index_sha = sha256_file(CAND_INDEX)
rep.check("state extract.data_sha256 == candidate f64 file sha256", state["extract"]["data_sha256"] == data_sha, data_sha)
rep.check("state extract.index_sha256 == candidate json file sha256", state["extract"]["index_sha256"] == index_sha, index_sha)
rep.check("state extract.commit == HEAD", state["extract"]["commit"] == head)
rep.check("state extract paths name the candidate files",
          state["extract"]["candidate_data_path"] == str(CAND_DATA) and state["extract"]["candidate_index_path"] == str(CAND_INDEX))
rep.check("no 'incident' / 'incidents' / 'stop' key anywhere in the state",
          all(tok not in canonical(state) for tok in ('"incident"', '"incidents"', '"stop"')))
rep.note(f"state run_id {state['run_id']}; extract started {ext['started_at']} completed {ext['completed_at']}; runtime {ext['runtime']}")

# ---------------------------------------------------------------------------------------------------------------------
# Item 2: artifact integrity.
rep.check("f64 sha256 == expected d2ee71e5…", data_sha == EXPECTED["f64_sha256"], data_sha)
rep.check("json file sha256 == expected 62818147…", index_sha == EXPECTED["json_sha256"], index_sha)
raw = CAND_DATA.read_bytes()
rep.check("f64 is exactly 1,209,600 bytes = 151,200 float64 = 18,900 × 8", len(raw) == 1_209_600 and len(raw) // 8 == 151_200 and len(raw) % 8 == 0, str(len(raw)))
cells = np.frombuffer(raw, dtype="<f8").reshape(18900, 8)
rep.check("all 151,200 values finite", bool(np.isfinite(cells).all()), f"non-finite {int((~np.isfinite(cells)).sum())}")
rep.check("n == 79 in every row", bool((cells[:, 0] == 79.0).all()), f"distinct n {sorted(set(cells[:, 0].tolist()))}")
# the byte order is really little-endian: a big-endian reading gives nonsense for n
rep.check("big-endian reading of column n is not 79 (endianness sanity)", not bool((np.frombuffer(raw, dtype=">f8").reshape(18900, 8)[:, 0] == 79.0).any()))
rep.check("M2 >= 0 in every row; SSE columns >= 0", bool((cells[:, 7] >= 0).all()) and bool((cells[:, 3:6] >= 0).all()))

index_text = CAND_INDEX.read_text(encoding="utf-8")
index = strict_json(CAND_INDEX)
rep.check("index has no duplicate keys", True)
rep.check("index bytes == canonical JSON + newline", index_text == canonical(index) + "\n")
content = content_digest_without(index, "content_sha256")
rep.check("index content_sha256 recomputes (own canonical JSON)", content == index["content_sha256"] == EXPECTED["json_content"], content)
rep.check("index file_sha256 == data sha256", index["file_sha256"] == data_sha)
rep.check("index total_bytes == data length", index["total_bytes"] == len(raw))
rep.check("index blocks == [cells, [18900, 8], offset 0, nbytes 1,209,600]",
          index["blocks"] == [{"name": "cells", "shape": [18900, 8], "offset_bytes": 0, "nbytes": 1209600}], json.dumps(index["blocks"]))
rep.check("index dtype '<f8' and format string", index["dtype"] == "<f8" and index["format"] == "raw IEEE-754 float64, little-endian, C order; blocks back to back in the order listed")
rep.check("index columns == n,S,Q,SSE0,SSE1,SSEC,mean,M2", index["columns"] == ["n", "S", "Q", "SSE0", "SSE1", "SSEC", "mean", "M2"])
rep.check("index experiment/kind/version/schema", (index["experiment"], index["kind"], index["version"], index["schema_version"]) == ("023", "exposed-cells", "023-cells-v1", 1))
rep.check("index design == revision 2 at 5b38aba", index["design"] == {"commit": "5b38aba", "path": "docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md", "revision": 2})
rep.check("index program masks P0 0, P1 14/30", index["program"]["P0_mask"] == 0 and index["program"]["P1_mask"] == {"cue_final": 14, "coordinated": 30})
rep.check("index extraction.run_id == state run_id", index["extraction"]["run_id"] == state["run_id"])
rep.check("index extraction checks == state extract checks", index["extraction"]["checks"] == state["extract"]["checks"])
rep.note(f"index extraction checks: E4 {index['extraction']['checks']['E4']}, E5 {index['extraction']['checks']['E5']}, E6 {index['extraction']['checks']['E6']}")

# ---------------------------------------------------------------------------------------------------------------------
# Item 4 (file-level): the 022 table file, the 022 record, the 022 confirmation, the module blobs.
size = TABLE022.stat().st_size
table_sha = sha256_file(TABLE022)
rep.check("022 table size == 424,131,002", size == EXPECTED["table_size"], str(size))
rep.check("022 table sha256 == 04659d5e…", table_sha == EXPECTED["table_sha256"], table_sha)
src = index["source_022"]
rep.check("index source_022.table_file_sha256 == table sha256", src["table_file_sha256"] == table_sha)
rep.check("index source_022.table_path", src["table_path"] == "outputs/experiment-022/calibration-table.pt")

record = strict_json(RECORD022)
record_file = sha256_file(RECORD022)
record_content = content_digest_without(record, "content_sha256")
rep.check("022 record file sha256 starts db745653 and == index", record_file.startswith(EXPECTED["record022_file_prefix"]) and record_file == src["calibration_record_file_sha256"], record_file)
rep.check("022 record content digest recomputes, starts 46985fd5, == index",
          record_content == record["content_sha256"] and record_content.startswith(EXPECTED["record022_content_prefix"]) and record_content == src["calibration_record_content_sha256"], record_content)
rep.check("022 record path in index", src["calibration_record_path"] == "experiments/022-upstream-error-localization/calibration-v1.json")
bound = record["rematerialization"]["table_sha256"]
rep.check("022 record binds 13 tensor digests incl. 5 gates", len(bound) == 13 and sum(k.startswith("gate_") for k in bound) == 5, str(sorted(bound)))
rep.check("index source_022.table_tensor_sha256 == 022 record's table_sha256", src["table_tensor_sha256"] == bound)
head_record = git("show", f"HEAD:experiments/022-upstream-error-localization/calibration-v1.json")
rep.check("022 record working file == HEAD blob", head_record == RECORD022.read_bytes())

conf = strict_json(CONF022)
conf_content = content_digest_without(conf, "content_sha256")
conf_file = sha256_file(CONF022)
rep.check("022 confirmation content digest recomputes (af848ae4…) == index binding",
          conf_content == conf["content_sha256"] and conf_content.startswith(EXPECTED["conf022_content_prefix"]) and conf_content == src["confirmation_022_content_sha256"], conf_content)
rep.note(f"022 confirmation file sha256 {conf_file}; state inputs confirmation_022_file {state['inputs']['confirmation_022_file']}")
rep.check("state inputs: 022 record file/content, confirmation file/content, table file",
          state["inputs"]["calibration_022_file"] == record_file and state["inputs"]["calibration_022_content"] == record_content
          and state["inputs"]["confirmation_022_file"] == conf_file and state["inputs"]["confirmation_022_content"] == conf_content
          and state["inputs"]["table_022_file"] == table_sha)

extraction = index["extraction"]
module_path = "src/neural_decompiler/block0_completion.py"
head_bytes = git("show", f"HEAD:{module_path}")
own_blob = git_blob_sha1(head_bytes)
git_blob = git("rev-parse", f"HEAD:{module_path}").decode().strip()
rep.check("extraction module path", extraction["module"] == module_path)
rep.check("extraction module_blob == own sha1 of HEAD's block0_completion.py == git rev-parse", extraction["module_blob"] == own_blob == git_blob, own_blob)
rep.check("working-tree block0_completion.py == HEAD bytes", (REPO / module_path).read_bytes() == head_bytes)
rep.check("extraction protocol_code_commit == HEAD", extraction["protocol_code_commit"] == head)
rep.check("extraction cells_version", extraction["cells_version"] == "023-cells-v1")
bad = {}
for name, blob in extraction["frozen_blobs"].items():
    path = f"src/neural_decompiler/{name}"
    mine = git_blob_sha1(git("show", f"HEAD:{path}"))
    if mine != blob or git_blob_sha1((REPO / path).read_bytes()) != blob:
        bad[name] = (blob, mine)
rep.check(f"frozen_blobs ({len(extraction['frozen_blobs'])}) == git blobs at HEAD and working tree", not bad and len(extraction["frozen_blobs"]) == 11, str(bad))
rep.check("frozen_blobs == state module_blobs", extraction["frozen_blobs"] == state["module_blobs"])
rep.check("upstream_localization.py blob is 022's 46585696…", extraction["frozen_blobs"]["upstream_localization.py"] == "465856962aa380747d1a4f1338d1d2762d03c9f9")
rep.check("the extraction commit is HEAD and is on origin/main", extraction["protocol_code_commit"] in remote)

# the candidate files are gitignored outputs; the install targets are not ignored
ignored = git("check-ignore", "--no-index", "-q", "outputs/experiment-023/candidate-exposed-cells.f64") == b""
rep.note("candidate files are under the gitignored outputs/experiment-023/ (git check-ignore exit 0)")
try:
    git("check-ignore", "--no-index", "-q", "experiments/023-block0-completion/exposed-cells.f64")
    install_ignored = True
except Exception:
    install_ignored = False
try:
    git("check-ignore", "--no-index", "-q", "experiments/023-block0-completion/exposed-cells.json")
    install_ignored = True or install_ignored
except Exception:
    pass
rep.check("install targets experiments/023-block0-completion/exposed-cells.{f64,json} are not gitignored", not install_ignored)
rep.check("install targets do not exist yet", not (REPO / "experiments/023-block0-completion/exposed-cells.f64").exists() and not (REPO / "experiments/023-block0-completion/exposed-cells.json").exists())
rep.summary()
