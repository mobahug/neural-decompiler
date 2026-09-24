"""Items 2-5 and 7: clean-room reconstruction of Experiment 023's freeze.

Independence contract: this script never imports ``neural_decompiler`` (asserted at the end). Inputs:
  * the design text (git blob at 5b38aba, checked equal to HEAD and the working tree) -> candidate lists, quotas,
    p_c ranges, expected picks;
  * the tokenizer (AutoTokenizer, pinned revision parsed from models.py text, local_files_only);
  * raw committed JSON files (confirmations 006/009/011-020/022, the extension, 020's exposed ids) and 020's local
    results state (its ledger);
  * frozen constant STRINGS parsed with ``ast`` from the frozen source text (kind, rule texts, loader names, scope,
    design/plan binding, frame origin) -- never evaluated by importing the code.
The confirmation file is read only at the very end, for comparison.
"""
import ast
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
OUT = Path(__file__).resolve().parent
DESIGN_PATH = "docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md"
CONF_023 = ROOT / "experiments/023-block0-completion/confirmation-v1.json"
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        fails.append(name)


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_show(rev, rel):
    return subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=ROOT, check=True, capture_output=True).stdout


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def content_ok(payload):
    return payload.get("content_sha256") == sha_text(canon({k: v for k, v in payload.items() if k != "content_sha256"}))


# ---------------------------------------------------------------------------------------------------------------
# A. The design text: candidate lists, quotas, ranges, expected picks.
print("== A. design revision 2 text")
design = git_show("5b38aba", DESIGN_PATH).decode("utf-8")
check("design text at 5b38aba == HEAD == working tree", design.encode() == git_show("HEAD", DESIGN_PATH) == (ROOT / DESIGN_PATH).read_bytes())
check("design says Revision 2", "**Status:** Revision 2." in design)
cue_rows = re.findall(r"^\| `(determiner-like|quantity|adjective)` \| (.+?) \|$", design, re.M)
frame_rows = re.findall(r"^\| (cardinal|quantifier|coordinated-adjective) \| (.+?) \|$", design, re.M)
D_CUES = {name: [w.strip() for w in row.split(",")] for name, row in cue_rows}
D_FRAMES = {name: [t.strip() for t in row.split(" · ")] for name, row in frame_rows}
check("design cue table: 3 strata in order", [n for n, _ in cue_rows] == ["determiner-like", "quantity", "adjective"], str([n for n, _ in cue_rows]))
check("design frame table: 3 templates in order", [n for n, _ in frame_rows] == ["cardinal", "quantifier", "coordinated-adjective"], str([n for n, _ in frame_rows]))
print("      list lengths", {k: len(v) for k, v in D_CUES.items()}, {k: len(v) for k, v in D_FRAMES.items()})
CUE_QUOTA = int(re.search(r"`freeze` takes, per stratum, the first (\d+) entries", design).group(1))
FRAME_QUOTA = int(re.search(r"`freeze` takes, per template, the first (\d+) texts", design).group(1))
rng = re.search(r"cue-final (\d+)–(\d+), coordinated (\d+)–(\d+)", design)
P_C_RANGE = {"cardinal": [int(rng.group(1)), int(rng.group(2))], "quantifier": [int(rng.group(1)), int(rng.group(2))],
             "coordinated-adjective": [int(rng.group(3)), int(rng.group(4))]}
EXPECTED_PICKS = {name: [w.strip() for w in row.rstrip(";.").split(",")] for name, row in re.findall(r"^- `(determiner-like|quantity|adjective)`: (.+)$", design, re.M)}
print(f"      quotas {CUE_QUOTA}/{FRAME_QUOTA}; p_c ranges {P_C_RANGE}")
check("design text: frame ids are <template>-023-<k>", "Frame ids are `<template>-023-<k>`." in design)
check("design text: coordinated p_t = p_c + 1", "in coordinated frames, `p_t = p_c + 1`" in design)
check("design text: frames built by pm._build_new_frame", "the frame is built by `pm._build_new_frame`" in design)

# ---------------------------------------------------------------------------------------------------------------
# B. Frozen constant strings, parsed from source text with ast (no import).
print("== B. frozen constants (ast over the frozen source text; no import)")
b0c_src = git_show("HEAD", "src/neural_decompiler/block0_completion.py").decode()
ul_src = git_show("HEAD", "src/neural_decompiler/upstream_localization.py").decode()
pm_src = git_show("HEAD", "src/neural_decompiler/plural_mechanism.py").decode()
models_src = git_show("HEAD", "src/neural_decompiler/models.py").decode()
for rel, text in (("block0_completion.py", b0c_src), ("upstream_localization.py", ul_src), ("plural_mechanism.py", pm_src), ("models.py", models_src)):
    check(f"working-tree {rel} == HEAD blob", (ROOT / "src/neural_decompiler" / rel).read_text() == text)
b0c_tree, ul_tree, pm_tree = ast.parse(b0c_src), ast.parse(ul_src), ast.parse(pm_src)


def module_constant(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def dict_string_values(fn, key):
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == key and isinstance(v, ast.Constant) and isinstance(v.value, str):
                    out.append(v.value)
    return out


DESIGN_BIND = module_constant(b0c_tree, "DESIGN")
PLAN_BIND = module_constant(b0c_tree, "PLAN")
SCOPE = module_constant(b0c_tree, "SCOPE")
EXPERIMENT = module_constant(b0c_tree, "EXPERIMENT")
SCHEMA = module_constant(b0c_tree, "CONFIRMATION_SCHEMA_VERSION")
freeze_fn = function(b0c_tree, "freeze_payload")
KIND = dict_string_values(freeze_fn, "kind")
KIND = [k for k in KIND if k.startswith("the new cues")]
RULE_CUE = [v for v in dict_string_values(freeze_fn, "cue") if v.startswith("a single token")]
RULE_FRAME = [v for v in dict_string_values(freeze_fn, "frame") if v.startswith("a new text")]
LOADER_022 = dict_string_values(function(b0c_tree, "exclusion"), "loader")
lfi = function(ul_tree, "load_frozen_inputs")
loaders_node = next(n for n in ast.walk(lfi) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "loaders" for t in n.targets))
LOADERS = {k.value: v.elts[0].value for k, v in zip(loaders_node.value.keys, loaders_node.value.values)}
POOL_LOADER = dict_string_values(lfi, "loader")
EXCLUSION_CONFIRMATIONS = list(module_constant(ul_tree, "EXCLUSION_CONFIRMATIONS"))
MANIFEST_REL = module_constant(pm_tree, "MANIFEST_RELATIVE_PATH")
EXTENSION_REL = module_constant(pm_tree, "EXTENSION_RELATIVE_PATH")
bnf = function(pm_tree, "_build_new_frame")
ORIGIN = [kw.value.value for n in ast.walk(bnf) if isinstance(n, ast.Call) for kw in n.keywords if kw.arg == "origin"]
m = re.search(r'PYTHIA_70M = ModelSpec\(\s*model_id="([^"]+)",\s*revision="([0-9a-f]{40})",?\s*\)', models_src)
MODEL = {"model_id": m.group(1), "revision": m.group(2)}
print("      DESIGN", DESIGN_BIND, "PLAN", PLAN_BIND)
print("      KIND", KIND, "\n      RULE_CUE", RULE_CUE, "\n      RULE_FRAME", RULE_FRAME)
print("      LOADERS", LOADERS, "\n      POOL/EXT loaders", POOL_LOADER, "022 loader", LOADER_022, "origin", ORIGIN, "\n      MODEL", MODEL)
check("single kind / rule strings / 022 loader / origin found", len(KIND) == 1 and len(RULE_CUE) == 1 and len(RULE_FRAME) == 1 and len(LOADER_022) == 1 and ORIGIN == ["extension"])
check("design binding: the design commit is where revision 2 was written", subprocess.run(["git", "log", "--format=%h", "-1", DESIGN_BIND["commit"], "--", DESIGN_PATH], cwd=ROOT,
      capture_output=True, text=True).stdout.strip() == DESIGN_BIND["commit"] and DESIGN_BIND["revision"] == 2 and DESIGN_BIND["path"] == DESIGN_PATH)
check("plan binding: plan commit 3a795fb touched the plan", PLAN_BIND["commit"] in subprocess.run(["git", "log", "--format=%h", "--", PLAN_BIND["path"]], cwd=ROOT, capture_output=True,
      text=True).stdout.split())
check("scope: new determiner-like, quantity and adjective cues and new frames only; no possessive/pronoun claim",
      "new determiner-like, quantity and adjective cues and new sentence frames only" in SCOPE and "no new prospective claim for possessive or pronoun cues" in SCOPE
      and "no claim about all grammatical-number cue classes" in SCOPE, SCOPE)

# ---------------------------------------------------------------------------------------------------------------
# C. The tokenizer (only the tokenizer).
print("== C. tokenizer")
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL["model_id"], revision=MODEL["revision"], local_files_only=True)
print("      tokenizer", type(tok).__name__, "vocab", len(tok), "init path", getattr(tok, "name_or_path", None))


def enc(text):
    return [int(t) for t in tok.encode(text, add_special_tokens=False)]


# ---------------------------------------------------------------------------------------------------------------
# D. The exclusion, rebuilt from raw committed files.
print("== D. exclusion from raw files")
conf_paths = {}
for key in EXCLUSION_CONFIRMATIONS:
    matches = sorted(glob.glob(str(ROOT / f"experiments/{key}-*/confirmation-v1.json")))
    check(f"confirmation {key}: exactly one committed file", len(matches) == 1 and subprocess.run(["git", "ls-files", "--error-unmatch", matches[0]], cwd=ROOT,
          capture_output=True).returncode == 0, matches[0] if matches else "")
    conf_paths[key] = os.path.relpath(matches[0], ROOT)
confs = {key: load(rel) for key, rel in conf_paths.items()}
ext = load(EXTENSION_REL)
c22_rel = "experiments/022-upstream-error-localization/confirmation-v1.json"
c22 = load(c22_rel)
for key, payload in {**confs, "extension": ext, "022": c22}.items():
    check(f"content_sha256 of {key} recomputes (own canonical JSON)", content_ok(payload), payload["content_sha256"][:16])
c20 = confs["020"]
pool_token_ids = [int(t) for t in c20["exposed_token_ids"]]
pool_frame_ids = list(c20["exposed_frame_ids"])
check("020 exposed pool: 279 token ids (distinct), 108 frame ids (distinct)", len(pool_token_ids) == len(set(pool_token_ids)) == 279 and len(pool_frame_ids) == len(set(pool_frame_ids)) == 108)
# every exposed frame id resolved from the raw frame records of the extension and 006..019
frame_records = {}
for source, frames in [("extension-original", ext["original_frames"]), ("extension-new", ext["new_frames"])] + [(k, confs[k]["frames"]) for k in EXCLUSION_CONFIRMATIONS]:
    for f in frames:
        if f["frame_id"] in frame_records:
            check(f"frame id {f['frame_id']} defined once", False, source)
        frame_records[f["frame_id"]] = {**f, "_source": source}
pool_frames = [frame_records[fid] for fid in pool_frame_ids]
check("every 020 exposed frame id resolves to a committed frame record (extension or 006..019)", all(f["_source"] != "020" for f in pool_frames),
      str(sorted({f["_source"] for f in pool_frames})))
check("the 108 exposed frames = extension 12 + confirmations 006..019 frames",
      sorted(pool_frame_ids) == sorted([f["frame_id"] for f in ext["original_frames"] + ext["new_frames"]] + [f["frame_id"] for k in EXCLUSION_CONFIRMATIONS if k != "020" for f in confs[k]["frames"]]))
template_cue_ids = {}
for f in ext["original_frames"]:
    template_cue_ids.setdefault(f["template_id"], dict(f["cue_ids"]))
    check(f"template cue ids consistent within {f['template_id']}", template_cue_ids[f["template_id"]] == dict(f["cue_ids"]))
reference_ids = {k: int(v) for k, v in ext["reference_cue_ids"].items()}
check("reference ids identical in the extension and every confirmation 006..022", all({k: int(v) for k, v in p["reference_cue_ids"].items()} == reference_ids for p in list(confs.values()) + [c22]),
      str(reference_ids))
check("reference ids are the template singular cues", all(reference_ids[t] == template_cue_ids[t]["sg"] for t in template_cue_ids))
plural_ids = {t: int(c["pl"]) for t, c in template_cue_ids.items()}
print("      template cue ids", template_cue_ids, "decoded", {t: {k: tok.decode([v]) for k, v in c.items()} for t, c in template_cue_ids.items()})

ids_022 = set(pool_token_ids) | set(reference_ids.values()) | set(plural_ids.values())
texts_022 = {f["text_template"] for f in pool_frames}
all_frames = list(pool_frames)
for key in EXCLUSION_CONFIRMATIONS:
    ids_022 |= {int(t["token_id"]) for t in confs[key]["tokens"]}
    all_frames += confs[key]["frames"]
all_frames += ext["original_frames"] + ext["new_frames"]
ids_022 |= {int(e["token_id"]) for e in ext["cue_words"]} | set(reference_ids.values())
for f in all_frames:
    ids_022 |= {int(v) for v in f["cue_ids"].values()}
texts_022 |= {f["text_template"] for f in all_frames}
check("022-style exclusion rebuilt = 022's own recorded exclusion (303 ids / 126 texts)",
      sorted(ids_022) == c22["exclusion"]["cue_token_ids"] and sorted(texts_022) == c22["exclusion"]["frame_texts"], f"{len(ids_022)} ids / {len(texts_022)} texts")
ids_023 = set(ids_022) | {int(c["token_id"]) for c in c22["cues"]} | {int(v) for f in c22["frames"] for v in f["cue_ids"].values()}
texts_023 = set(texts_022) | {f["text_template"] for f in c22["frames"]}
EXCL_IDS, EXCL_TEXTS = sorted(ids_023), sorted(texts_023)
check("exclusion: 327 cue token ids, 144 frame texts", (len(EXCL_IDS), len(EXCL_TEXTS)) == (327, 144), f"{len(EXCL_IDS)} / {len(EXCL_TEXTS)}")


def source_entry(name, loader, rels, extra):
    return {"source": name, "loader": loader, "files": [{"path": rel, "file_sha256": file_sha(ROOT / rel)} for rel in rels], **extra}


SOURCES = [source_entry("pool-020", POOL_LOADER[0], [], {"tokens": len(pool_token_ids), "frames": len(pool_frame_ids)})]
for key in EXCLUSION_CONFIRMATIONS:
    SOURCES.append(source_entry(f"confirmation-{key}", LOADERS[key], [conf_paths[key]],
                                {"content_sha256": confs[key]["content_sha256"], "tokens": len(confs[key]["tokens"]), "frames": len(confs[key]["frames"])}))
SOURCES.append(source_entry("extension", POOL_LOADER[1], [MANIFEST_REL, EXTENSION_REL],
                            {"content_sha256": ext["content_sha256"], "frames": len(ext["original_frames"]) + len(ext["new_frames"]), "tokens": len(ext["cue_words"])}))
SOURCES.append(source_entry("confirmation-022", LOADER_022[0], [c22_rel], {"content_sha256": c22["content_sha256"], "tokens": len(c22["cues"]), "frames": len(c22["frames"])}))
check("15 sources", len(SOURCES) == 15, str([s["source"] for s in SOURCES]))
EXCLUSION = {"cue_token_ids": EXCL_IDS, "cue_token_ids_sha256": sha_text(canon(EXCL_IDS)), "frame_texts": EXCL_TEXTS, "frame_texts_sha256": sha_text(canon(EXCL_TEXTS)),
             "sources": SOURCES}

# ---------------------------------------------------------------------------------------------------------------
# E. Cue selection: every candidate evaluated in order.
print("== E. cue selection (every candidate, in order)")
cues, rejected_before_quota = [], []
cue_eval = {}
for stratum, words in D_CUES.items():
    taken, rows = 0, []
    for rank, word in enumerate(words):
        ids = enc(" " + word)
        bare = enc(word)
        single = len(ids) == 1
        tid = ids[0] if single else None
        used = single and tid in ids_023
        dup = single and tid in {c["token_id"] for c in cues}
        eligible = single and not used and not dup
        reason = "eligible" if eligible else ("%d tokens with a leading space" % len(ids) if not single else ("token id already used" if used else "duplicate"))
        rows.append({"rank": rank, "word": word, "ids": ids, "piece": tok.convert_ids_to_tokens(ids), "decoded": tok.decode(ids), "bare_ids": bare,
                     "bare_pieces": tok.convert_ids_to_tokens(bare), "eligible": eligible, "reason": reason})
        if taken < CUE_QUOTA:
            if eligible:
                cues.append({"word": word, "token_id": tid, "class": stratum, "candidate_rank": rank})
                taken += 1
            else:
                rejected_before_quota.append({"kind": "cue", "class": stratum, "candidate": word, "rank": rank, "reason": reason})
    cue_eval[stratum] = rows
    for r in rows:
        print(f"      {stratum:16s} r{r['rank']:>2} {r['word']:12s} ' '+w -> {r['ids']} {r['piece']}  bare -> {r['bare_ids']} {r['bare_pieces']}  {r['reason']}")
    check(f"{stratum}: {CUE_QUOTA} taken = ranks 0..{CUE_QUOTA - 1}", taken == CUE_QUOTA and [c["candidate_rank"] for c in cues if c["class"] == stratum] == list(range(CUE_QUOTA)))
    check(f"{stratum}: picks == the design's expected picks", [c["word"] for c in cues if c["class"] == stratum] == EXPECTED_PICKS[stratum])
    check(f"{stratum}: every entry of the list eligible (design pre-check)", all(r["eligible"] for r in rows), f"{sum(r['eligible'] for r in rows)}/{len(rows)}")
EXPECTED_IDS = {"general": 2087, "subsequent": 6774, "given": 1677, "chosen": 6777, "selected": 4236, "present": 1246, "ultimate": 12553, "preceding": 17691, "tons": 16298,
                "piles": 41019, "masses": 11843, "stacks": 34577, "gross": 13711, "net": 2036, "scores": 7363, "batches": 39657, "humble": 26896, "proud": 9979, "shy": 23478,
                "lazy": 22658, "busy": 10000, "sturdy": 41789, "fragile": 28304, "shiny": 30006}
check("the 24 selected (word, id) == the expected list", {c["word"]: c["token_id"] for c in cues} == EXPECTED_IDS and len(cues) == 24)
check("24 distinct ids", len({c["token_id"] for c in cues}) == 24)
check("each selected id decodes to ' '+word", all(tok.decode([c["token_id"]]) == " " + c["word"] for c in cues))
print("      leading-space rule matters for:", sorted(r["word"] for rows in cue_eval.values() for r in rows[:CUE_QUOTA] if len(r["bare_ids"]) != 1 or r["bare_ids"][0] != r["ids"][0]),
      "| bare word multi-token:", sorted(r["word"] for rows in cue_eval.values() for r in rows[:CUE_QUOTA] if len(r["bare_ids"]) != 1))

# ---------------------------------------------------------------------------------------------------------------
# F. Frame selection: own structural rule.
print("== F. frame selection (every candidate, in order; own structural rule)")


def own_frame(template, text, cue_ids):
    """Returns (frame dict | None, reasons)."""
    reasons = []
    if text.count("{cue}") != 1:
        reasons.append("cue slot count != 1")
        return None, reasons
    built = {}
    for label in ("sg", "pl"):
        cue_word = tok.decode([cue_ids[label]]).strip()
        ids = enc(text.replace("{cue}", cue_word))
        occ = [i for i, t in enumerate(ids) if t == cue_ids[label]]
        if len(occ) != 1:
            reasons.append(f"{label}: cue token occurs {len(occ)} times")
            continue
        built[label] = (ids, occ[0])
    if len(built) < 2:
        return None, reasons
    (sg, ps), (pl, pp) = built["sg"], built["pl"]
    if ps != pp or sg[:ps] != pl[:pp] or sg[ps + 1:] != pl[pp + 1:]:
        reasons.append("sg/pl prompts do not share prefix/suffix")
        return None, reasons
    prefix, suffix = sg[:ps], sg[ps + 1:]
    coordinated = template == "coordinated-adjective"
    if coordinated and len(suffix) != 1:
        reasons.append(f"coordinated suffix has {len(suffix)} tokens")
    if not coordinated and suffix:
        reasons.append("cue-final frame has tokens after the cue")
    p_c, p_t = len(prefix), len(prefix) + len(suffix)
    lo, hi = P_C_RANGE[template]
    if not lo <= p_c <= hi:
        reasons.append(f"p_c {p_c} outside {lo}-{hi}")
    if p_t != p_c + (1 if coordinated else 0):
        reasons.append(f"p_t {p_t} vs p_c {p_c}")
    return {"prefix_ids": prefix, "suffix_ids": suffix, "p_c": p_c, "p_t": p_t}, reasons


frames, frame_eval = [], {}
for template, texts in D_FRAMES.items():
    taken, rows = 0, []
    for rank, text in enumerate(texts):
        used = text in texts_023
        dup = text in {f["text_template"] for f in frames}
        built, reasons = own_frame(template, text, template_cue_ids[template])
        if used:
            reasons.insert(0, "text already used")
        if dup:
            reasons.insert(0, "duplicate text")
        eligible = not reasons
        rows.append({"rank": rank, "text": text, "eligible": eligible, "reasons": reasons, **(built or {})})
        if taken < FRAME_QUOTA:
            if eligible:
                taken += 1
                frames.append({"template_id": template, "frame_id": f"{template}-023-{taken}", "prefix_ids": built["prefix_ids"], "suffix_ids": built["suffix_ids"],
                               "cue_ids": dict(template_cue_ids[template]), "text_template": text, "origin": ORIGIN[0], "p_c": built["p_c"], "p_t": built["p_t"],
                               "candidate_rank": rank})
            else:
                rejected_before_quota.append({"kind": "frame", "template": template, "candidate": text, "rank": rank, "reason": "; ".join(reasons)})
    frame_eval[template] = rows
    for r in rows:
        print(f"      {template:22s} r{r['rank']:>2} p_c {r.get('p_c')} p_t {r.get('p_t')} prefix {r.get('prefix_ids')} suffix {r.get('suffix_ids')}  {r['text']!r}  "
              f"{'eligible' if r['eligible'] else r['reasons']}")
    check(f"{template}: {FRAME_QUOTA} taken = ranks 0..{FRAME_QUOTA - 1}", taken == FRAME_QUOTA and [f["candidate_rank"] for f in frames if f["template_id"] == template] == list(range(FRAME_QUOTA)))
    check(f"{template}: every candidate structurally eligible (design pre-check)", all(r["eligible"] for r in rows), f"{sum(r['eligible'] for r in rows)}/{len(rows)}")
check("rejected (before quota) is empty", rejected_before_quota == [])
check("18 distinct new frame texts", len({f["text_template"] for f in frames}) == 18)
p_cs = {t: sorted({r["p_c"] for r in rows}) for t, rows in frame_eval.items()}
print("      p_c over each full candidate list", p_cs)

# descriptive: every Y2 / Y1 prompt text re-tokenizes to prefix + cue + suffix
roundtrip = {"Y2": [0, 0], "Y1": [0, 0]}
for population, frame_list in (("Y2", frames), ("Y1", pool_frames)):
    for f in frame_list:
        for c in cues:
            ids = enc(f["text_template"].replace("{cue}", c["word"]))
            roundtrip[population][0] += 1
            roundtrip[population][1] += ids == list(f["prefix_ids"]) + [c["token_id"]] + list(f["suffix_ids"])
print("      descriptive round trip (text with the cue word re-tokenizes to prefix+cue+suffix):", roundtrip)
in_context = sorted({(c["word"], f["frame_id"]) for c in cues for f in pool_frames + frames if c["token_id"] in list(f["prefix_ids"]) + list(f["suffix_ids"])})
print("      descriptive: new cue ids occurring inside any exposed/new frame's prefix or suffix:", in_context)

# ---------------------------------------------------------------------------------------------------------------
# G. Freshness beyond the exclusion.
print("== G. freshness")
new_ids = {c["token_id"] for c in cues}
new_texts = {f["text_template"] for f in frames}
check("0 overlap of the 24 ids with the 327 excluded ids", not (new_ids & ids_023))
check("0 overlap of the 18 texts with the 144 excluded texts", not (new_texts & texts_023))
cells = load("experiments/023-block0-completion/exposed-cells.json")
cells_ids, cells_frames = {int(c[1]) for c in cells["cues"]}, {f[0] for f in cells["frames"]}
cells_texts = {frame_records[fid]["text_template"] for fid in cells_frames}
check("022's calibration units (exposed-cells index): 175 cues / 108 frames; 0 overlap", len(cells_ids) == 175 and len(cells_frames) == 108 and not (new_ids & cells_ids)
      and not (new_texts & cells_texts) and cells_frames == set(pool_frame_ids))
c20_ids = {int(t["token_id"]) for t in c20["tokens"]} | {int(v) for f in c20["frames"] for v in f["cue_ids"].values()}
c20_texts = {f["text_template"] for f in c20["frames"]}
c20_prompt_ids = {int(p["token_id"]) for p in c20["token_prompts"] + c20["exposed_frame_prompts"]}
check("020's confirmation set (021's spent set): 0 overlap (tokens, frame cue ids, prompt token ids, 18 frame texts)",
      not (new_ids & (c20_ids | c20_prompt_ids)) and not (new_texts & c20_texts), f"{len(c20_ids | c20_prompt_ids)} ids, {len(c20_texts)} texts")
c22_ids = {int(c["token_id"]) for c in c22["cues"]} | {int(v) for f in c22["frames"] for v in f["cue_ids"].values()}
c22_texts = {f["text_template"] for f in c22["frames"]}
check("022's frozen units: 0 overlap", not (new_ids & c22_ids) and not (new_texts & c22_texts), f"{len(c22_ids)} ids, {len(c22_texts)} texts")
new_frame_ids = {f["frame_id"] for f in frames}
check("new frame ids are not any earlier frame id", not (new_frame_ids & (set(frame_records) | {f["frame_id"] for f in c20["frames"] + c22["frames"]})))

# ---------------------------------------------------------------------------------------------------------------
# H. The manifest, from the frozen key format "<frame_id>|<label>|<token id>".
print("== H. manifest")


def key(frame_id, label, token_id):
    return f"{frame_id}|{label}|{int(token_id)}"


MANIFEST = {"S1-REF": sorted(key(f["frame_id"], "ref", reference_ids[f["template_id"]]) for f in frames),
            "S1-VALIDITY": sorted(key(f["frame_id"], "pl", f["cue_ids"]["pl"]) for f in frames),
            "S2-TARGET": {"Y1": sorted(key(fid, c["word"], c["token_id"]) for fid in pool_frame_ids for c in cues),
                          "Y2": sorted(key(f["frame_id"], c["word"], c["token_id"]) for f in frames for c in cues)}}
all_keys = MANIFEST["S1-REF"] + MANIFEST["S1-VALIDITY"] + MANIFEST["S2-TARGET"]["Y1"] + MANIFEST["S2-TARGET"]["Y2"]
sizes = (len(MANIFEST["S1-REF"]), len(MANIFEST["S1-VALIDITY"]), len(MANIFEST["S2-TARGET"]["Y1"]), len(MANIFEST["S2-TARGET"]["Y2"]))
check("sizes 18 / 18 / 2592 / 432; 3060 unique", sizes == (18, 18, 2592, 432) and len(set(all_keys)) == len(all_keys) == 3060, f"{sizes} unique {len(set(all_keys))}")
ledger_020_state = load("outputs/experiment-020/results.json")
recorded = ledger_020_state.pop("state_sha256")
check("020 results state digest recomputes (own canonical JSON) and is 020's closed state", recorded == sha_text(canon(ledger_020_state))
      == "2e5485dccb39c018eb02ee8d0a3086004399690a4dfd9c114008f057b23bf8f5" and file_sha(ROOT / "outputs/experiment-020/results.json")
      == "da63b8c29f9553a9da62bea7a442e11cef999ccddc71a7a332ac7c938abc9e00")
ledger_020 = set(ledger_020_state["executed_prompt_keys"])
m20 = c20["manifest"]
keys_020_conf = set(m20["S1-REF"]) | set(m20["S1-VALIDITY"]) | set(m20["S2-TARGET"])
m22 = c22["manifest"]
keys_022 = set(m22["S1-REF"]) | set(m22["S1-VALIDITY"]) | set(m22["S2-TARGET"]["Y1"]) | set(m22["S2-TARGET"]["Y2"])
forbidden = ledger_020 | keys_020_conf | keys_022
print(f"      020 ledger {len(ledger_020)}, 020 confirmation manifest {len(keys_020_conf)}, 022 manifest {len(keys_022)}, forbidden union {len(forbidden)}")
check("0 collisions with 020 ledger / 020 confirmation set / 022 manifest", not (set(all_keys) & forbidden))
ledger_023 = load("outputs/experiment-023/results.json")["executed_prompt_keys"]
check("0 collisions with the 023 ledger (empty)", ledger_023 == [] and not (set(all_keys) & set(ledger_023)))
nonconforming = sorted(k for k in forbidden if len(k.split("|")) != 3 or not k.rsplit("|", 1)[1].isdigit())
check("every forbidden key has the <frame_id>|<label>|<token id> shape", not nonconforming, str(nonconforming[:3]))
f_ids = {int(k.rsplit("|", 1)[1]) for k in forbidden if k not in nonconforming}
f_frames = {k.split("|", 1)[0] for k in forbidden}
check("token level: no forbidden key carries a new cue id; no forbidden key uses a new frame id", not (f_ids & new_ids) and not (f_frames & new_frame_ids))
check("token level: every token id in 020's ledger lies in the exclusion", {int(k.rsplit("|", 1)[1]) for k in ledger_020} <= ids_023)
check("every S2-TARGET key has a new cue id or a new frame; every stage-1 key has a new frame",
      all(int(k.rsplit("|", 1)[1]) in new_ids or k.split("|", 1)[0] in new_frame_ids for k in MANIFEST["S2-TARGET"]["Y1"] + MANIFEST["S2-TARGET"]["Y2"])
      and all(k.split("|", 1)[0] in new_frame_ids for k in MANIFEST["S1-REF"] + MANIFEST["S1-VALIDITY"]))

# ---------------------------------------------------------------------------------------------------------------
# I. The full object, serialized, against the file.
print("== I. full object, byte for byte")
payload = {"experiment": EXPERIMENT, "schema_version": SCHEMA, "kind": KIND[0], "design": DESIGN_BIND, "plan": PLAN_BIND, "scope": SCOPE, "model": MODEL,
           "candidates": {"cues": D_CUES, "frames": D_FRAMES},
           "rules": {"cue_quota": CUE_QUOTA, "frame_quota": FRAME_QUOTA, "p_c_range": P_C_RANGE, "cue": RULE_CUE[0], "frame": RULE_FRAME[0]},
           "exclusion": EXCLUSION, "reference_cue_ids": reference_ids, "exposed_frame_ids": pool_frame_ids, "cues": cues, "frames": frames, "rejected": rejected_before_quota,
           "counts": {"classes": {s: sum(c["class"] == s for c in cues) for s in D_CUES}, "templates": {t: sum(f["template_id"] == t for f in frames) for t in D_FRAMES}},
           "manifest": MANIFEST}
payload["content_sha256"] = sha_text(canon(payload))
mine = (canon(payload) + "\n").encode("utf-8")
disk = CONF_023.read_bytes()
print(f"      mine: {len(mine)} bytes, sha256 {hashlib.sha256(mine).hexdigest()}, content {payload['content_sha256']}")
print(f"      file: {len(disk)} bytes, sha256 {hashlib.sha256(disk).hexdigest()}")
check("byte-identical to confirmation-v1.json", mine == disk)
check("file sha256 5fadfa50... / content 4e64d4c2... / 128228 bytes", hashlib.sha256(disk).hexdigest() == "5fadfa503f4fe35308cb4473220a1d9cd6f7a46e3852f638594b8081909825f4"
      and payload["content_sha256"] == "4e64d4c2c85ae8f9c710171372a4bf28f0964a8e8864a0326182edba44aa14ed" and len(disk) == 128228)
filed = json.loads(disk)
if mine != disk:
    for k in sorted(set(filed) | set(payload)):
        if filed.get(k) != payload.get(k):
            print("      DIFF in field", k)
check("file content_sha256 recomputes from the file itself", content_ok(filed))
check("file is canonical JSON + newline", disk == (canon(filed) + "\n").encode("utf-8"))
(OUT / "r2_reconstruction.json").write_text(canon({"payload": payload, "cue_eval": cue_eval, "frame_eval": frame_eval}) + "\n", encoding="utf-8")
check("never imported neural_decompiler", not any(name.startswith("neural_decompiler") for name in sys.modules))
print("ITEMS 2-5,7 (independent):", "ALL PASS" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
