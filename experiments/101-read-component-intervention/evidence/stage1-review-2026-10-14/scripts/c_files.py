"""Item 3: the three stage-1 files against the record and the lock (no model, no forward; reads only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402  (first)

guard.self_test()
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy  # noqa: E402

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

ROOT = guard.ROOT
EXP = ROOT / ri.EXPERIMENT_DIR
out: dict = {"checks": {}, "facts": {}}


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}


state = rd.load_results_state(ROOT / "outputs/experiment-101/results.json")
record = state["stage1"]
lock = json.loads((EXP / "preregistration-lock.json").read_text(encoding="utf-8"))
confirmation = ri.confirmation_from_payload(json.loads((EXP / "confirmation-v1.json").read_text(encoding="utf-8")), ri.PRODUCTION)
paths = {name: ROOT / entry for name, entry in lock["stage1"]["paths"].items()}
raw = {name: path.read_bytes() for name, path in paths.items()}
for name, data in raw.items():
    check(f"sha256_{name}_equals_record", hashlib.sha256(data).hexdigest() == record["files"][name]["sha256"], hashlib.sha256(data).hexdigest())
    check(f"path_{name}_equals_record", str(paths[name].relative_to(ROOT)) == record["files"][name]["path"])
out["facts"]["sizes"] = {name: len(data) for name, data in raw.items()}

# --- states.json, the index -------------------------------------------------------------------------------------------
index = json.loads(raw["states_index"].decode("utf-8"))
check("index_is_canonical_json_plus_newline", raw["states_index"].decode("utf-8") == pm.canonical_json(index) + "\n")
try:
    ul.verify_table_index(index, layout=lock["stage1"]["states"]["layout"], meta=lock["stage1"]["states"]["meta"])
    check("index_verifies_against_lock_layout_and_meta", True)
except Exception as error:
    check("index_verifies_against_lock_layout_and_meta", False, f"{type(error).__name__}: {error}")
check("index_file_sha256_names_states_f64", index["file_sha256"] == hashlib.sha256(raw["states_data"]).hexdigest() and index["total_bytes"] == len(raw["states_data"]),
      {"index": index["file_sha256"], "total_bytes": index["total_bytes"]})
check("index_format", index["format"] == ul.TABLE_FORMAT and index["dtype"] == "<f8", {"format": index["format"], "dtype": index["dtype"]})
meta_keys = set(lock["stage1"]["states"]["meta"])
extra = sorted(set(index) - meta_keys - {"format", "dtype", "blocks", "total_bytes", "file_sha256"})
check("index_has_no_unbound_keys", not extra, extra)
layout = lock["stage1"]["states"]["layout"]
offset, my_offsets = 0, []
for entry in layout:
    nbytes = int(numpy.prod(entry["shape"])) * 8
    my_offsets.append({"name": entry["name"], "shape": entry["shape"], "offset_bytes": offset, "nbytes": nbytes})
    offset += nbytes
check("index_offsets_equal_contiguous_layout", index["blocks"] == my_offsets and offset == index["total_bytes"] == 5240064, {"total": offset})
check("layout_324_blocks", len(layout) == 324 == 36 * len(ri.STATE_FIELDS))

# --- states.f64, read back through ul.read_table and independently with numpy ------------------------------------------
blocks = ul.read_table(paths["states_data"], index)  # verifies the file sha256 and size against the index
arr = numpy.frombuffer(raw["states_data"], dtype="<f8")
independent_equal = all(numpy.array_equal(arr[e["offset_bytes"] // 8: (e["offset_bytes"] + e["nbytes"]) // 8].reshape(e["shape"]), blocks[e["name"]].numpy())
                        for e in my_offsets)
check("independent_numpy_parse_equals_read_table", independent_equal)
check("all_values_finite", bool(numpy.isfinite(arr).all()), int((~numpy.isfinite(arr)).sum()))
locked = ri.states_from_blocks(blocks, confirmation.frames)
digests = {frame_id: rd.state_digest(entry) for frame_id, entry in sorted(locked.items())}
check("state_digests_equal_record", digests == record["state_digests"], sum(digests[k] != record["state_digests"].get(k) for k in digests))
# The same digests from my own reconstruction of the entries (not ri.states_from_blocks): the locked_state keys.
my_entries = {}
for frame in confirmation.frames:
    entry = {"p_c": frame.p_c, "p_t": frame.p_t}
    for name in ("x1_all", "x2_all", "x3_all", "x4_all", "x5_all", "h6", "rows4", "rows5", "c_ref"):
        e = next(b for b in my_offsets if b["name"] == f"{frame.frame_id}/{name}")
        entry[name] = arr[e["offset_bytes"] // 8: (e["offset_bytes"] + e["nbytes"]) // 8].reshape(e["shape"]).tolist()
    my_entries[frame.frame_id] = entry
my_digests = {fid: hashlib.sha256(json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
              for fid, entry in sorted(my_entries.items())}
check("independent_state_digests_equal_record", my_digests == record["state_digests"])
pos_ok, shapes_ok, rows_ok, rows_dev = True, True, True, 0.0
for frame in confirmation.frames:
    entry = locked[frame.frame_id]
    pos_ok &= (entry["p_c"], entry["p_t"]) == (frame.p_c, frame.p_t) and frame.p_t == frame.p_c + 1
    keys = frame.p_t + 1
    shapes_ok &= all(len(entry[f]) == keys and all(len(x) == 512 for x in entry[f]) for f in ("x1_all", "x2_all", "x3_all", "x4_all", "x5_all"))
    n_c_ref = next(b["shape"][0] for b in layout if b["name"] == f"{frame.frame_id}/c_ref")
    shapes_ok &= len(entry["h6"]) == 512 and len(entry["rows4"]) == 8 and len(entry["rows5"]) == 8 and len(entry["c_ref"]) == n_c_ref
    for rows in (entry["rows4"], entry["rows5"]):
        for row in rows:
            rows_dev = max(rows_dev, abs(sum(row) - 1.0))
            rows_ok &= len(row) == keys and min(row) >= 0.0
check("positions_equal_frozen_frames", pos_ok)
check("shapes_match_frames", shapes_ok)
check("attention_rows_are_distributions", rows_ok and rows_dev < 1e-5, rows_dev)
out["facts"]["attention_row_sum_max_deviation"] = rows_dev
out["facts"]["c_ref_length"] = len(locked[confirmation.frames[0].frame_id]["c_ref"])
out["facts"]["p_c_distribution"] = {str(k): sum(1 for f in confirmation.frames if f.p_c == k) for k in range(4, 8)}

# --- table.json ---------------------------------------------------------------------------------------------------------
text = raw["table"].decode("utf-8")


def reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


table = json.loads(text, parse_constant=reject_constant)
check("table_is_canonical_json_plus_newline", text == pm.canonical_json(table) + "\n")
check("table_content_digest_verifies", table["content_sha256"] == rc.content_digest(table) == record["table_content_sha256"], table["content_sha256"])
bound = lock["stage1"]["table"]
check("table_metadata_equals_lock_binding", {key: table.get(key) for key in bound} == dict(bound),
      sorted(key for key in bound if table.get(key) != bound[key]))
check("table_top_keys", set(table) == set(bound) | {"frames", "content_sha256"}, sorted(set(table) - set(bound)))
check("table_fractions_for_weak_off", table["fractions_for_weak"] is False)
check("table_frames_are_the_frozen_frames", set(table["frames"]) == {f.frame_id for f in confirmation.frames} and len(table["frames"]) == 36)
struct_ok, nonfinite, weak = True, 0, {}
for frame in confirmation.frames:
    entry = table["frames"][frame.frame_id]
    struct_ok &= entry["p_c"] == frame.p_c and entry["p_t"] == frame.p_t and set(entry["cues"]) == {"two", "three", "many"} and set(entry) == {"p_c", "p_t", "cues"}
    for label, cue in entry["cues"].items():
        struct_ok &= set(cue) == {"pair", "dc_full", "weak", "fractions", "arms"} and set(cue["arms"]) == set(ri.PREDICTED_ARMS) and len(cue["arms"]) == 11
        struct_ok &= all(len(a["dc"]) == 79 for a in cue["arms"].values())
        struct_ok &= cue["dc_full"] == cue["arms"]["PL"]["mean"] and cue["weak"] == (cue["dc_full"] > -1.0)
        struct_ok &= (cue["fractions"] is None) == cue["weak"]
        if cue["weak"]:
            weak.setdefault(label, []).append(frame.frame_id)
        values = [cue["dc_full"], *cue["pair"].values(), *[x for a in cue["arms"].values() for x in (a["mean"], *a["dc"])]]
        values += list((cue["fractions"] or {}).values())
        nonfinite += sum(1 for v in values if isinstance(v, float) and not math.isfinite(v))
        struct_ok &= all(v is not None for v in values)
check("table_structure", struct_ok)
check("table_no_nonfinite", nonfinite == 0, nonfinite)
out["facts"]["predicted_weak_frames"] = weak
out["guard"] = guard.summary()
out["all_ok"] = all(c["ok"] for c in out["checks"].values())
(guard.OUT / "c_files.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
for name, c in out["checks"].items():
    print(("PASS " if c["ok"] else "FAIL ") + name + ("" if c["ok"] else f"  {c['detail']}"))
print(json.dumps(out["facts"], indent=1))
print("ALL OK" if out["all_ok"] else "SOME CHECKS FAILED")
