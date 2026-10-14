"""Compare the independent derivation with the frozen confirmation file (still no neural_decompiler import)."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HERE = Path(__file__).resolve().parent
PATH = ROOT / "experiments/101-read-component-intervention/confirmation-v1.json"
data = PATH.read_bytes()
payload = json.loads(data)
mine = json.load(open(HERE / "independent_derivation.json"))
o = mine["out"]


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


r: dict = {}
# --- integrity ---
r["file_sha256"] = hashlib.sha256(data).hexdigest()
r["bytes"] = len(data)
r["file_is_canonical_json_plus_newline"] = data == (canonical(payload) + "\n").encode("utf-8")
r["ends_with_single_newline"] = data.endswith(b"\n") and not data.endswith(b"\n\n")
r["content_sha256_recomputes"] = payload["content_sha256"] == sha(canonical({k: v for k, v in payload.items() if k != "content_sha256"}))
r["manifest_sha256_recomputes"] = payload["manifest_sha256"] == sha(canonical(payload["manifest"]))
r["manifest_sha256_equals_independent"] = payload["manifest_sha256"] == o["manifest_sha256"]
r["manifest_equals_independent"] = payload["manifest"] == mine["manifest"]
r["experiment"] = payload["experiment"]
r["schema_version"] = payload["schema_version"]
r["design"] = payload["design"]
r["plan"] = payload["plan"]
r["model"] = payload["model"]
r["top_level_keys"] = sorted(payload)

# --- configuration (as recorded; compared against the plan's constants by hand) ---
cfg = payload["configuration"]
r["configuration_summary"] = {k: v for k, v in cfg.items() if k not in ("frame_candidates", "tags", "expected_picks")}
r["configuration_frame_candidates_equal_design"] = cfg["frame_candidates"] == o["candidates_from_design"]
r["configuration_expected_picks"] = cfg["expected_picks"] == list(range(1, 37))
r["configuration_tags_equal_independent_run_order"] = cfg["tags"] == o["run_order_tags_frame1"]
r["run_order_equals_independent"] = payload["run_order"] == o["run_order_tags_frame1"]
r["cue_order"] = payload["cue_order"]

# --- exclusions ---
ex = payload["exclusions"]


def dig(values):
    return {"count": len(values), "sha256": sha(canonical(sorted(values)))}


mine_ex = {"frame_texts": dig(mine["TEXTS"]), "earlier_cue_ids": dig(mine["EARLIER"]), "candidate_words_025": dig(mine["WORDS_025"]),
           "target_form_ids": dig(mine["TARGETS"]), "prior_noun_ids": dig(mine["PRIOR"])}
r["exclusions_compare"] = {name: {"file": ex[name], "mine": mine_ex[name], "equal": ex[name] == mine_ex[name]} for name in mine_ex}
r["candidate_ids_025_compare"] = {"file": payload["candidate_ids_025"], "mine": dig(mine["IDS_025"]), "equal": payload["candidate_ids_025"] == dig(mine["IDS_025"])}
r["number_words_equal"] = ex["number_words"] == mine["NUMBER_WORDS"]
r["earlier_cue_ids_024"] = ex["earlier_cue_ids_024"]
r["exclusion_sources"] = ex["sources"]


def git_bytes(rev_path):
    return subprocess.run(["git", "--no-optional-locks", "show", rev_path], cwd=ROOT, capture_output=True, check=True).stdout


src_checks = {}
for name, entry in ex["sources"].items():
    b = git_bytes(f"HEAD:{entry['path']}")
    j = json.loads(b)
    if "content_sha256" in entry:
        src_checks[name] = {"recorded": entry["content_sha256"], "file_content_sha256": j.get("content_sha256"),
                            "recomputed": sha(canonical({k: v for k, v in j.items() if k != "content_sha256"})),
                            "equal": entry["content_sha256"] == j.get("content_sha256") == sha(canonical({k: v for k, v in j.items() if k != "content_sha256"}))}
    if "file_sha256" in entry:
        src_checks[name] = {"recorded": entry["file_sha256"], "recomputed": hashlib.sha256(b).hexdigest(), "equal": entry["file_sha256"] == hashlib.sha256(b).hexdigest()}
r["exclusion_source_digests"] = src_checks

# --- candidates ---
cand_diffs = []
walk = {w["number"]: w for w in o["walk"]}
for entry in payload["candidates"]:
    n = entry["number"]
    w = walk[n]
    for key in ("text", "frame_id", "adjective", "adjective_id", "p_c", "p_t", "eligible", "reasons"):
        if entry.get(key) != w.get(key):
            cand_diffs.append([n, key, entry.get(key), w.get(key)])
    extra = set(entry) - {"number", "text", "frame_id", "adjective", "adjective_id", "p_c", "p_t", "eligible", "reasons"}
    if extra:
        cand_diffs.append([n, "extra_keys", sorted(extra)])
r["candidates_count"] = len(payload["candidates"])
r["candidates_numbers_in_order"] = [e["number"] for e in payload["candidates"]] == list(range(1, 61))
r["candidate_diffs"] = cand_diffs

# --- frames ---
frame_diffs = []
r["picks"] = payload["picks"]
r["picks_equal_independent"] = payload["picks"] == o["first_36"]
r["expected_picks_equal"] = payload["expected_picks"] == list(range(1, 37))
r["picks_match_expected"] = payload["picks_match_expected"]
for entry, n in zip(payload["frames"], o["first_36"]):
    w = walk[n]
    checks = {"number": entry.get("number") == n, "frame_id": entry["frame_id"] == w["frame_id"], "text_template": entry["text_template"] == w["text"],
              "prefix_ids": entry["prefix_ids"] == w["prefix_ids"], "suffix_ids": entry["suffix_ids"] == w["suffix_ids"],
              "p_c": entry["p_c"] == w["p_c"], "p_t": entry["p_t"] == w["p_t"], "cue_ids": entry["cue_ids"] == {"sg": 581, "pl": 767},
              "template_id": entry["template_id"] == "coordinated-adjective",
              "prompt_ids_reconstruct": all(entry["prefix_ids"] + [tid] + entry["suffix_ids"] == w["token_ids"][label]
                                            for label, tid in (("ref", 581), ("two", 767), ("three", 1264), ("many", 1142)))}
    if not all(checks.values()):
        frame_diffs.append([n, {k: v for k, v in checks.items() if not v}])
    extra = set(entry) - {"number", "frame_id", "text_template", "prefix_ids", "suffix_ids", "p_c", "p_t", "cue_ids", "template_id", "origin"}
    if extra:
        frame_diffs.append([n, "extra_keys", sorted(extra)])
r["frames_count"] = len(payload["frames"])
r["frame_origins"] = sorted({e.get("origin") for e in payload["frames"]})
r["frame_diffs"] = frame_diffs

# --- manifests ---
s1, s2 = payload["manifest"]["S1-REF"], payload["manifest"]["S2-TARGET"]
r["manifest_counts"] = {"S1": len(s1), "S1_distinct": len(set(s1)), "S2": len(s2), "S2_distinct": len(set(s2)), "S1_sorted": s1 == sorted(s1),
                        "S2_sorted": s2 == sorted(s2), "disjoint": not set(s1) & set(s2)}
r["counts_field"] = payload["counts"]
untagged = {"|".join(k.split("|")[:3]) for k in s1 + s2}
r["untagged_from_file"] = len(untagged)
r["untagged_equal_independent"] = sorted(untagged) == mine["untagged"]
# every reference-prompt arm carries ref|581
bad_role = []
REF_ROLE = {"REF", "SUF-READ", "SUF-OFF", "SUF-NEG", "SUF-HALF", "PC-SWAP-REF", "HEAD-SUF", "HEAD-SUF-V"} | {f"SUF-ISO-{j}" for j in range(1, 9)}
for k in s2:
    fid, label, tid, tag = k.split("|")
    arm, _, cue = tag.partition("@")
    if arm in REF_ROLE:
        ok = (label, tid) == ("ref", "581") and (cue in ("two", "three", "many") or tag == "REF")
    else:
        ok = (label, tid) == (cue, {"two": "767", "three": "1264", "many": "1142"}[cue])
    if not ok:
        bad_role.append(k)
r["s2_role_violations"] = bad_role
per_frame = {}
for k in s2:
    per_frame.setdefault(k.split("|")[0], []).append(k)
r["s2_per_frame_sizes"] = sorted({len(v) for v in per_frame.values()})
r["s2_frames"] = len(per_frame)
print(json.dumps(r, indent=1, ensure_ascii=False))
json.dump(r, open(HERE / "compare_with_file.json", "w"), indent=1, ensure_ascii=False)
