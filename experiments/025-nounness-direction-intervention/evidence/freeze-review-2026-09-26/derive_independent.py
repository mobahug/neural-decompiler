"""Independent re-derivation of Experiment 025's freeze (read-only reviewer script).

Uses only: json on committed files, the pinned tokenizer (transformers AutoTokenizer, local_files_only, plus the raw
`tokenizers` library as a second route), ul.load_frozen_inputs(ROOT) for the pool (80 nouns, 108 frames) as the brief
allows, and 020's closed local results state for 020's ledger (the same file the runner's _base() reads).

It never imports neural_decompiler.cue_rotation, never loads the model, never opens a weight file, never computes a
score, centroid, direction or rotation. An audit hook installed before any import refuses: any write under the
repository, any open under outputs/experiment-023/ or outputs/experiment-024/, any calibration-table.pt, and any
model weight file. Every file opened under the repository is recorded.
"""
import os
import sys

ROOT = "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler"
HERE = os.path.dirname(os.path.abspath(__file__))
OPENED = []
REFUSED = []


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        path = os.path.realpath(os.fsdecode(args[0]))
    except Exception:
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (
        isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    bad = ("/outputs/experiment-023/" in path or "/outputs/experiment-024/" in path or path.endswith("calibration-table.pt")
           or path.endswith(".safetensors") or path.endswith("pytorch_model.bin") or path.endswith(".ckpt"))
    if bad:
        REFUSED.append(path)
        raise PermissionError(f"reviewer guard: may not open {path}")
    if writing and path.startswith(ROOT + "/") and "/.venv/" not in path:
        REFUSED.append("WRITE " + path)
        raise PermissionError(f"reviewer guard: may not write {path}")
    if path.startswith(ROOT + "/") and "/.venv/" not in path and "/src/neural_decompiler/" not in path:
        OPENED.append(path[len(ROOT) + 1:])


sys.addaudithook(_hook)

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402


def _refuse(*a, **k):
    raise RuntimeError("reviewer guard: a torch module forward was reached")


torch.nn.Module.__call__ = _refuse

from neural_decompiler import models  # noqa: E402

models.load_model = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reviewer guard: model load refused"))

R = Path(ROOT)
OUT = {}


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fsha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the artifact and its integrity
ART_REL = "experiments/025-nounness-direction-intervention/confirmation-v1.json"
art_bytes = (R / ART_REL).read_bytes()
art = json.loads(art_bytes.decode("utf-8"))
OUT["file_sha256"] = hashlib.sha256(art_bytes).hexdigest()
OUT["file_size"] = len(art_bytes)
OUT["content_sha256_recorded"] = art["content_sha256"]
OUT["content_sha256_recomputed"] = sha(canon({k: v for k, v in art.items() if k != "content_sha256"}))
OUT["file_is_canonical_json_plus_newline"] = art_bytes == (canon(art) + "\n").encode("utf-8")
OUT["manifest_sha256_recorded"] = art["manifest_sha256"]
OUT["manifest_sha256_recomputed"] = sha(canon(art["manifest"]))
OUT["top_level_keys"] = sorted(art)
OUT["experiment"] = art["experiment"]
OUT["schema_version"] = art["schema_version"]
OUT["design"] = art["design"]
OUT["plan"] = art["plan"]
OUT["model"] = art["model"]
OUT["model_matches_pinned_spec"] = art["model"] == {"model_id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision}

# ---------------------------------------------------------------- the design's and plan's frozen text
design = (R / "docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md").read_text(encoding="utf-8")
plan = (R / "docs/superpowers/plans/2026-09-26-experiment-025-nounness-direction-intervention-plan.md").read_text(encoding="utf-8")
m = re.search(r"\*\*The new candidate list, frozen in alphabetical textual order\*\*[^\n]*\n\n((?:> [^\n]*\n)+)", design)
NEW_LIST = " ".join(line[2:] for line in m.group(1).splitlines()).split()
rows = {}
for label, key in (("20 adjectives", "adj"), ("8 ordinary nouns (singular)", "ord"), ("12 new ordinary nouns (singular)", "new")):
    row = re.search(r"\| \*\*" + re.escape(label) + r"\*\* \|[^|]*\| ([^|]*) \|", design)
    rows[key] = [w.strip() for w in row.group(1).split(",")]
DESIGN_PICKS = {"adjective": rows["adj"], "noun": rows["ord"] + rows["new"]}
PATCH_KEYS = re.findall(r"- `([a-z0-9-]+\|[a-z]+\|\d+)`;?", plan.split("PATCH_PATH_SPENT_KEYS")[1][:800])
OUT["design_new_list"] = {"n": len(NEW_LIST), "words": NEW_LIST, "alphabetical": NEW_LIST == sorted(NEW_LIST)}
OUT["design_picks"] = DESIGN_PICKS
OUT["plan_patch_path_keys"] = PATCH_KEYS

# Conditions, by the design's table and the plan's fixed order.
P, H = "0.32", "0.16"
CONDITIONS = ["base", f"noun+{P}", f"noun-{P}", f"noun+{H}", f"noun-{H}"] + [f"rand{j}{s}{P}" for j in range(1, 8) for s in "+-"] + [f"plur+{P}", f"plur-{P}"]
OUTCOME_BEARING = [f"noun+{P}", f"noun-{P}"] + [f"rand{j}{s}{P}" for j in range(1, 8) for s in "+-"]


def tail(n, k):
    return Fraction(sum(math.comb(n, j) for j in range(k, n + 1)), 2 ** n)


threshold = next(k for k in range(41) if tail(40, k) <= Fraction(1, 40))
t27 = tail(40, 27)
expected_config = {"name": "production", "n_adjectives": 20, "n_nouns": 20, "n_cues": 40, "k_controls": 7, "primary_odd": 0.32, "half_odd": 0.16,
                   "count_threshold": threshold, "reference_tail": {"exact": f"{t27.numerator}/{t27.denominator}", "value": float(t27)}, "n_frames": 108,
                   "n_scored_nouns": 79, "conditions": CONDITIONS, "outcome_bearing": OUTCOME_BEARING, "expected_picks": DESIGN_PICKS}
OUT["config_expected_equals_file"] = art["configuration"] == expected_config
OUT["config_diff"] = {k: (art["configuration"].get(k), v) for k, v in expected_config.items() if art["configuration"].get(k) != v}
OUT["threshold_derived"] = threshold
OUT["tail_27"] = {"exact": f"{t27.numerator}/{t27.denominator}", "value": float(t27)}
OUT["tail_26"] = float(tail(40, 26))
OUT["conditions_equal"] = art["conditions"] == CONDITIONS and art["configuration"]["conditions"] == CONDITIONS
OUT["n_conditions"] = len(CONDITIONS)
OUT["n_outcome_bearing"] = len(art["configuration"]["outcome_bearing"])

# ---------------------------------------------------------------- the pool (ul.load_frozen_inputs, as the brief allows)
from neural_decompiler import upstream_localization as ul  # noqa: E402

inputs = ul.load_frozen_inputs(R)
pool = inputs.pool
frames = list(pool.frames)
nouns = list(pool.nouns)
OUT["pool"] = {"n_nouns": len(nouns), "n_scorable_single_token": sum(1 for n in nouns if len(n.sg_ids) == 1), "n_frames": len(frames),
               "multi_token_nouns": [[n.lexical_key, list(n.sg_ids), list(n.pl_ids)] for n in nouns if len(n.sg_ids) != 1],
               "reference_ids": dict(pool.reference_ids)}

F024 = load("experiments/024-readout-routing-nounness/confirmation-v1.json")
F020 = load("experiments/020-readout-decompilation/confirmation-v1.json")
F022 = load("experiments/022-upstream-error-localization/confirmation-v1.json")
F023 = load("experiments/023-block0-completion/confirmation-v1.json")
OUT["f024_file_sha256"] = fsha(R / "experiments/024-readout-routing-nounness/confirmation-v1.json")
OUT["f024_content_recomputes"] = F024["content_sha256"] == sha(canon({k: v for k, v in F024.items() if k != "content_sha256"}))
OUT["f024_content_sha256"] = F024["content_sha256"]
OUT["f020_file_sha256"] = fsha(R / "experiments/020-readout-decompilation/confirmation-v1.json")
OUT["f020_content_recomputes"] = F020["content_sha256"] == sha(canon({k: v for k, v in F020.items() if k != "content_sha256"}))

frame_ids = [f.frame_id for f in frames]
OUT["frame_ids_equal_file"] = art["exposed_frame_ids"] == frame_ids
OUT["frame_ids_equal_024"] = F024["exposed_frame_ids"] == frame_ids
OUT["frame_ids_equal_020"] = F020["exposed_frame_ids"] == frame_ids
OUT["frame_ids_unique"] = len(set(frame_ids)) == 108
OUT["reference_ids_equal_file"] = art["reference_cue_ids"] == {k: int(v) for k, v in pool.reference_ids.items()} == F024["reference_cue_ids"] == F020["reference_cue_ids"]

# The frames' token sequences against 020's committed exposed-frame prompts (an independent committed record).
by_id = {f.frame_id: f for f in frames}
bad_prompt = 0
for pr in F020["exposed_frame_prompts"]:
    f = by_id[pr["frame_id"]]
    if pr["token_ids"] != list(f.prefix_ids) + [pr["token_id"]] + list(f.suffix_ids):
        bad_prompt += 1
OUT["frames_match_020_exposed_prompts"] = {"prompts": len(F020["exposed_frame_prompts"]), "mismatches": bad_prompt}

# ---------------------------------------------------------------- the four exclusion sets, from their sources
earlier = set(int(i) for i in F024["exclusion"]["cue_token_ids"]) | {int(c["token_id"]) for c in F024["cues"]}
target = {int(t) for n in nouns for t in (*n.sg_ids, *n.pl_ids)}
prior_nouns = F020["nouns"]
prior = {int(t) for n in prior_nouns for t in (*n["sg_ids"], *n["pl_ids"])}
frame_tok = {int(t) for f in frames for t in (*f.prefix_ids, *f.suffix_ids)}
SETS = {"earlier_cues": earlier, "target_forms": target, "prior_nouns": prior, "frame_tokens": frame_tok}
OUT["exclusion_sizes"] = {k: len(v) for k, v in SETS.items()}
OUT["exclusion_parts"] = {"024_exclusion_cue_token_ids": len(set(F024["exclusion"]["cue_token_ids"])), "024_cues": len({c["token_id"] for c in F024["cues"]}),
                          "024_cues_already_in_exclusion": len({c["token_id"] for c in F024["cues"]} & set(F024["exclusion"]["cue_token_ids"])),
                          "020_prior_nouns": len(prior_nouns), "020_prior_noun_keys": sorted(n["lexical_key"] for n in prior_nouns)}
OUT["exclusion_equal_file"] = {k: sorted(v) == art["blocked"][k]["ids"] for k, v in SETS.items()}
OUT["exclusion_sha_recomputes"] = {k: art["blocked"][k]["sha256"] == sha(canon(art["blocked"][k]["ids"])) for k in SETS}
OUT["exclusion_file_keys"] = sorted(art["blocked"])
# Cross-checks against other committed records.
OUT["target_equals_024_committed_target_noun_form_ids"] = sorted(target) == F024["target_noun_form_ids"]["ids"]
OUT["frame_tokens_equal_024_committed_frame_token_ids"] = sorted(frame_tok) == F024["frame_token_ids"]["ids"]
OUT["statue_barrel"] = {n["lexical_key"]: {"ids": sorted({*n["sg_ids"], *n["pl_ids"]}), "in_prior": set(n["sg_ids"] + n["pl_ids"]) <= prior,
                                           "in_other_sets": {k: sorted(set(n["sg_ids"] + n["pl_ids"]) & v) for k, v in SETS.items() if k != "prior_nouns"}}
                        for n in prior_nouns if n["lexical_key"] in ("statue", "barrel")}
frame_cue_ids = {int(t) for f in frames for t in f.cue_ids.values()}
OUT["exposed_frame_cue_ids_subset_of_earlier"] = frame_cue_ids <= earlier
OUT["020_exposed_token_ids_subset_of_earlier"] = set(F020["exposed_token_ids"]) <= earlier
OUT["020_fresh_cues_subset_of_earlier"] = {t["token_id"] for t in F020["tokens"]} <= earlier
OUT["022_cues_subset_of_earlier"] = {c["token_id"] for c in F022["cues"]} <= earlier
OUT["023_cues_subset_of_earlier"] = {c["token_id"] for c in F023["cues"]} <= earlier

# ---------------------------------------------------------------- the tokenizer (two routes)
from transformers import AutoTokenizer  # noqa: E402
import tokenizers  # noqa: E402

tok = AutoTokenizer.from_pretrained(models.PYTHIA_70M.model_id, revision=models.PYTHIA_70M.revision, local_files_only=True)
snap = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots" / models.PYTHIA_70M.revision
raw = tokenizers.Tokenizer.from_file(str(snap / "tokenizer.json"))
OUT["tokenizer"] = {"class": type(tok).__name__, "name_or_path": tok.name_or_path, "revision": models.PYTHIA_70M.revision,
                    "tokenizer_json_sha256": fsha(snap / "tokenizer.json")}
route_mismatch = []


def encode(word):
    a = [int(t) for t in tok.encode(" " + word, add_special_tokens=False)]
    b = [int(t) for t in raw.encode(" " + word, add_special_tokens=False).ids]
    if a != b:
        route_mismatch.append(word)
    return a


def regular_plural(word):
    # the standard English regular plural
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


ORDER = ("earlier_cues", "target_forms", "prior_nouns", "frame_tokens")


def status(word, picked):
    ids = encode(word)
    if len(ids) != 1:
        return {"word": word, "ok": False, "kind": "multi", "n": len(ids), "ids": ids}
    t = ids[0]
    hits = [k for k in ORDER if t in SETS[k]]
    if hits:
        return {"word": word, "ok": False, "kind": "blocked", "id": t, "hits": hits}
    if t in picked:
        return {"word": word, "ok": False, "kind": "picked", "id": t}
    return {"word": word, "ok": True, "kind": "eligible", "id": t}


ADJ_LIST = list(F024["reserves"]["N"])
ORD_LIST = [tuple(pair.split("/")) for pair in F024["reserves"]["ordinary"]]
OUT["lists"] = {"adjective_reserves": ADJ_LIST, "ordinary_reserves": ORD_LIST}
OUT["candidates_equal_file"] = art["candidates"] == {"adjective-reserve": ADJ_LIST, "ordinary-reserve": [list(p) for p in ORD_LIST], "new-list": NEW_LIST}
OUT["new_list_plurals_are_plus_s"] = all(regular_plural(w) == w + "s" for w in NEW_LIST)

picked = set()
my_cues, my_rejected, walk = [], [], []
counts = {name: {"candidates": 0, "rejected": 0, "eligible": 0, "picked": 0, "unpicked_eligible": 0} for name in ("adjective-reserve", "ordinary-reserve", "new-list")}
n_adj = 0
for rank, word in enumerate(ADJ_LIST):
    st = status(word, picked)
    c = counts["adjective-reserve"]
    c["candidates"] += 1
    entry = {"list": "adjective-reserve", "rank": rank, "candidate": word, "status": [st]}
    if not st["ok"]:
        c["rejected"] += 1
        my_rejected.append(entry)
    else:
        c["eligible"] += 1
        if n_adj < 20:
            my_cues.append({"rank": rank, "source": "adjective-reserve", "stratum": "adjective", "token_id": st["id"], "word": word})
            picked.add(st["id"])
            n_adj += 1
            c["picked"] += 1
        else:
            c["unpicked_eligible"] += 1
    walk.append(entry)
n_noun = 0
noun_sources = [("ordinary-reserve", r, sg, pl) for r, (sg, pl) in enumerate(ORD_LIST)] + [("new-list", r, w, regular_plural(w)) for r, w in enumerate(NEW_LIST)]
for source, rank, sg, pl in noun_sources:
    s1, s2 = status(sg, picked), status(pl, picked)
    c = counts[source]
    c["candidates"] += 1
    entry = {"list": source, "rank": rank, "candidate": f"{sg}/{pl}", "status": [s1, s2]}
    if not (s1["ok"] and s2["ok"] and s1["id"] != s2["id"]):
        c["rejected"] += 1
        my_rejected.append(entry)
    else:
        c["eligible"] += 1
        if n_noun < 20:
            my_cues.append({"plural": pl, "plural_token_id": s2["id"], "rank": rank, "source": source, "stratum": "noun", "token_id": s1["id"], "word": sg})
            picked |= {s1["id"], s2["id"]}
            n_noun += 1
            c["picked"] += 1
        else:
            c["unpicked_eligible"] += 1
    walk.append(entry)

OUT["route_mismatch"] = route_mismatch
OUT["per_list_counts"] = counts
OUT["my_picks"] = {"adjective": [c["word"] for c in my_cues if c["stratum"] == "adjective"], "noun": [c["word"] for c in my_cues if c["stratum"] == "noun"]}
OUT["my_picks_equal_design"] = OUT["my_picks"] == DESIGN_PICKS
OUT["cues_equal_file_exactly"] = my_cues == art["cues"]
OUT["cue_table"] = [[c["stratum"], c["word"], c["token_id"], c.get("plural"), c.get("plural_token_id"), c["source"], c["rank"]] for c in my_cues]
OUT["cue_key_sets"] = sorted({tuple(sorted(c)) for c in art["cues"]})
OUT["picks_fields_equal_design"] = art["picks"] == DESIGN_PICKS and art["expected_picks"] == DESIGN_PICKS and art["picks_match_expected"] is True
all_ids = [c["token_id"] for c in my_cues] + [c["plural_token_id"] for c in my_cues if c["stratum"] == "noun"]
OUT["picked_ids"] = {"n_forms": len(all_ids), "distinct": len(set(all_ids)), "in_any_exclusion": sorted(set(all_ids) & set().union(*SETS.values()))}

# Rejection reasons, parsed and checked against my own derivation.
REASON_MAP = (("already used as a cue", "earlier_cues"), ("pool target noun", "target_forms"), ("020's confirmation list", "prior_nouns"),
              ("exposed frame", "frame_tokens"), ("already picked", "picked"))


def parse(part):
    form, reason = part.split(": ", 1)
    if reason == "eligible":
        return form, {"kind": "eligible"}
    mm = re.fullmatch(r"(\d+) tokens with a leading space", reason)
    if mm:
        return form, {"kind": "multi", "n": int(mm.group(1))}
    mm = re.fullmatch(r"token id (\d+): (.+)", reason)
    if mm:
        named = [key for text, key in REASON_MAP if text in mm.group(2)]
        return form, {"kind": "blocked", "id": int(mm.group(1)), "named": named}
    return form, {"kind": "unparsed", "text": reason}


mine_by = {(e["list"], e["rank"]): e for e in my_rejected}
file_by = {(e["list"], e["rank"]): e for e in art["rejected"]}
reason_checks = []
for key in sorted(set(mine_by) | set(file_by), key=lambda k: (["adjective-reserve", "ordinary-reserve", "new-list"].index(k[0]), k[1])):
    mine, theirs = mine_by.get(key), file_by.get(key)
    rec = {"list": key[0], "rank": key[1], "in_mine": mine is not None, "in_file": theirs is not None}
    if mine and theirs:
        rec["candidate_equal"] = mine["candidate"] == theirs["candidate"]
        parts = theirs["reason"].split("; ") if key[0] != "adjective-reserve" else [f"{theirs['candidate']}: {theirs['reason']}"]
        ok = len(parts) == len(mine["status"])
        details = []
        for part, st in zip(parts, mine["status"]):
            form, parsed = parse(part)
            good = form == st["word"]
            if parsed["kind"] == "eligible":
                good = good and st["ok"]
            elif parsed["kind"] == "multi":
                good = good and st["kind"] == "multi" and st["n"] == parsed["n"]
            elif parsed["kind"] == "blocked":
                if parsed["named"] == ["picked"]:
                    good = good and st["kind"] == "picked" and st["id"] == parsed["id"]
                else:
                    good = good and st["kind"] == "blocked" and st["id"] == parsed["id"] and len(parsed["named"]) == 1 and parsed["named"][0] == st["hits"][0]
            else:
                good = False
            ok = ok and good
            details.append({"form": form, "file": parsed, "mine": {k: v for k, v in st.items() if k != "word"}, "agree": good})
        rec["reason_agrees"] = ok
        rec["details"] = details
        rec["file_reason"] = theirs["reason"]
    reason_checks.append(rec)
OUT["rejections"] = reason_checks
OUT["rejections_all_agree"] = all(r.get("reason_agrees") and r.get("candidate_equal") and r["in_mine"] and r["in_file"] for r in reason_checks)
OUT["n_rejected_file"] = len(art["rejected"])
OUT["n_rejected_mine"] = len(my_rejected)
OUT["rejected_order_equal"] = [(e["list"], e["rank"]) for e in art["rejected"]] == [(e["list"], e["rank"]) for e in my_rejected]
OUT["rejected_entry_key_sets"] = sorted({tuple(sorted(e)) for e in art["rejected"]})
new_rej = [r for r in reason_checks if r["list"] == "new-list"]
OUT["new_list_rejection_breakdown"] = {
    "multi_token_entries": sum(1 for r in new_rej if any(d["mine"]["kind"] == "multi" for d in r["details"])),
    "blocked_entries": sorted({r["details"][0]["form"] + "/" + r["details"][1]["form"]: [d["mine"].get("hits") for d in r["details"]] for r in new_rej
                               if not any(d["mine"]["kind"] == "multi" for d in r["details"])}.items()),
}
OUT["statue_rejection"] = [r for r in reason_checks if r["list"] == "ordinary-reserve"]
OUT["unpicked_eligible"] = {"adjective-reserve": [w["candidate"] for w in walk if w["list"] == "adjective-reserve" and all(s["ok"] for s in w["status"])][20:],
                            "new-list": [w["candidate"] for w in walk if w["list"] == "new-list" and all(s["ok"] for s in w["status"])
                                         and w["status"][0]["id"] != w["status"][1]["id"]][12:]}

# ---------------------------------------------------------------- the manifest
tagged = [f"{fid}|{c['word']}|{c['token_id']}|{cond}" for fid in frame_ids for c in my_cues for cond in CONDITIONS]
untagged_mine = {f"{fid}|{c['word']}|{c['token_id']}" for fid in frame_ids for c in my_cues}
file_keys = art["manifest"]["S2-TARGET"]
OUT["manifest"] = {
    "n_mine": len(tagged), "n_mine_unique": len(set(tagged)), "n_file": len(file_keys), "n_file_unique": len(set(file_keys)),
    "file_sorted": file_keys == sorted(file_keys), "equal_to_mine_sorted": file_keys == sorted(tagged),
    "only_S2_TARGET": list(art["manifest"]) == ["S2-TARGET"],
    "all_tagged_4_fields_frozen_condition": all(len(k.split("|")) == 4 and k.split("|")[3] in CONDITIONS for k in file_keys),
    "untagged_file": len({"|".join(k.split("|")[:3]) for k in file_keys}),
    "untagged_equal_4320_prompts": {"|".join(k.split("|")[:3]) for k in file_keys} == untagged_mine and len(untagged_mine) == 4320,
    "per_condition_counts_all_4320": all(v == 4320 for v in __import__("collections").Counter(k.split("|")[3] for k in file_keys).values()),
    "sha256_of_mine": sha(canon({"S2-TARGET": sorted(tagged)})),
    "outcome_bearing_runs": 40 * 108 * len(OUTCOME_BEARING),
}
OUT["counts_field_equal"] = art["counts"] == {"conditions": 21, "frames": 108, "runs": 90720, "strata": {"adjective": 20, "noun": 20}}

# ---------------------------------------------------------------- the spent set, recomputed from its sources
closure = load("experiments/020-readout-decompilation/closure.json")
extract = load("experiments/020-readout-decompilation/evidence/exploration-record-2026-09-22.json")
res_path = R / "outputs/experiment-020/results.json"
res_bytes = res_path.read_bytes()
state = json.loads(res_bytes.decode("utf-8"))
ledger020 = list(state["executed_prompt_keys"])
del state
OUT["020_results_file_sha256"] = hashlib.sha256(res_bytes).hexdigest()
OUT["020_results_file_sha_matches_closure"] = OUT["020_results_file_sha256"] == closure["explore"]["results_file_sha256"]
del res_bytes
cands = {"canon_sorted": sha(canon(sorted(ledger020))), "canon_as_stored": sha(canon(ledger020))}
OUT["020_ledger"] = {"n": len(ledger020), "n_unique": len(set(ledger020)), "extract_n": extract["ledger"]["executed_prompt_keys"],
                     "extract_sha256": extract["ledger"]["executed_prompt_keys_sha256"],
                     "digest_match": [k for k, v in cands.items() if v == extract["ledger"]["executed_prompt_keys_sha256"]]}
conf020 = set(F020["manifest"]["S1-REF"]) | set(F020["manifest"]["S1-VALIDITY"]) | set(F020["manifest"]["S2-TARGET"])
conf020_prompts = {p["key"] for p in F020["exposed_frame_prompts"]} | {p["key"] for p in F020["token_prompts"]}
OUT["020_confirmation"] = {"n": len(conf020), "S2_TARGET_equals_prompt_lists": set(F020["manifest"]["S2-TARGET"]) == conf020_prompts}


def m22(p):
    mm = p["manifest"]
    return set(mm["S1-REF"]) | set(mm["S1-VALIDITY"]) | set(mm["S2-TARGET"]["Y1"]) | set(mm["S2-TARGET"]["Y2"])


parts = {"020_ledger": set(ledger020), "020_confirmation": conf020, "022_manifest": m22(F022), "023_manifest": m22(F023), "024_manifest": set(F024["manifest"]["S2-TARGET"])}
spent = set().union(*parts.values())
names = list(parts)
OUT["spent"] = {"sizes": {k: len(v) for k, v in parts.items()}, "union": len(spent), "sum": sum(len(v) for v in parts.values()),
                "pairwise_overlaps": {f"{a}&{b}": len(parts[a] & parts[b]) for i, a in enumerate(names) for b in names[i + 1:] if parts[a] & parts[b]},
                "field_counts": dict(__import__("collections").Counter(len(k.split("|")) for k in spent))}
all25 = set(file_keys) | untagged_mine
OUT["collisions"] = {"untagged_in_spent": len(untagged_mine & spent), "tagged_in_spent": len(set(file_keys) & spent),
                     "patch_keys_in_manifest_or_untagged": len(set(PATCH_KEYS) & all25),
                     "patch_keys_in_spent": {k: [n for n, v in parts.items() if k in v] for k in PATCH_KEYS}}
# Stronger isolation: prompt identity (frame_id, cue token id) and the cue token ids of every spent key.
spent_pairs = {(k.split("|")[0], k.split("|")[2]) for k in spent}
mine_pairs = {(fid, str(c["token_id"])) for fid in frame_ids for c in my_cues}
spent_cue_ids = {int(k.split("|")[2]) for k in spent}
cue_ids_25 = {c["token_id"] for c in my_cues}
OUT["collisions"]["frame_tokenid_pairs_in_spent"] = len(spent_pairs & mine_pairs)
OUT["collisions"]["cue_ids_ever_prompted_in_spent_keys"] = sorted(cue_ids_25 & spent_cue_ids)
OUT["collisions"]["plural_ids_ever_prompted_in_spent_keys"] = sorted({c["plural_token_id"] for c in my_cues if c["stratum"] == "noun"} & spent_cue_ids)
OUT["collisions"]["words_ever_labels_in_spent_keys"] = sorted({c["word"] for c in my_cues} & {k.split("|")[1] for k in spent})

# ---------------------------------------------------------------- sources and rules
OUT["sources"] = art["sources"]
OUT["sources_ok"] = (art["sources"]["freeze_024"] == {"path": "experiments/024-readout-routing-nounness/confirmation-v1.json", "content_sha256": F024["content_sha256"]}
                     and art["sources"]["prior_nouns_020"] == {"path": "experiments/020-readout-decompilation/confirmation-v1.json", "file_sha256": OUT["f020_file_sha256"],
                                                                "nouns": sorted(n["lexical_key"] for n in prior_nouns)})
OUT["rules"] = art["rules"]
OUT["kind"] = art["kind"]
OUT["cue_rotation_imported"] = "neural_decompiler.cue_rotation" in sys.modules
OUT["opened_files_under_repo"] = sorted(set(OPENED))
OUT["refused"] = REFUSED

(Path(HERE) / "derive_independent.out.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("written", Path(HERE) / "derive_independent.out.json")
