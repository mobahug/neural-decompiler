"""Independent re-derivation of Experiment 101's frame freeze (AI reviewer, read-only).

Uses ONLY: the design text at its decided commit, the committed confirmation files (read through `git show HEAD:`),
the pinned tokenizer (AutoTokenizer, cross-checked against the raw `tokenizers` library on the snapshot's
tokenizer.json), and the canonical-JSON digest rule (json.dumps, sort_keys, compact separators, ensure_ascii=False).
It does NOT import neural_decompiler at all (no ri, no pm, no rc), loads no model and opens no weight file.
It writes nothing but its own JSON output beside itself.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HERE = Path(__file__).resolve().parent
REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots" / REVISION
assert os.environ.get("HF_HUB_OFFLINE") == "1"


def git_show(rev_path: str) -> bytes:
    return subprocess.run(["git", "--no-optional-locks", "show", rev_path], cwd=ROOT, capture_output=True, check=True).stdout


def committed_json(path: str) -> dict:
    data = git_show(f"HEAD:{path}")
    assert data == (ROOT / path).read_bytes(), f"working copy of {path} differs from HEAD"
    return json.loads(data)


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


out: dict = {"order_note": "independent derivation; written before any use of ri.freeze_payload / ri.candidate_status / ri.exclusions_from_committed"}

# ---------------------------------------------------------------------------------------------------------------
# 1. The 60 candidates, parsed from design §11 at the decided commit 516df20 (and compared with HEAD's copy).
DESIGN_PATH = "docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md"
design_decided = git_show(f"516df20:{DESIGN_PATH}").decode("utf-8")
design_head = git_show(f"HEAD:{DESIGN_PATH}").decode("utf-8")
out["design_unchanged_since_516df20"] = design_decided == design_head
section = design_decided.split("## 11. The frames", 1)[1].split("## 12.", 1)[0]
rows = re.findall(r"^\| (\d+) \| (.+?) \| (\d+) \| (.+?) \|$", section, flags=re.M)
cands = {}
for a, ta, b, tb in rows:
    cands[int(a)] = ta.strip()
    cands[int(b)] = tb.strip()
assert sorted(cands) == list(range(1, 61)), sorted(cands)
CANDIDATES = [cands[n] for n in range(1, 61)]
out["candidates_from_design"] = CANDIDATES

# ---------------------------------------------------------------------------------------------------------------
# 2. Two tokenizer implementations.
from transformers import AutoTokenizer  # noqa: E402
import tokenizers  # noqa: E402

hf = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision=REVISION, local_files_only=True)
raw = tokenizers.Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json"))
out["tokenizer_json_sha256"] = hashlib.sha256((SNAPSHOT / "tokenizer.json").read_bytes()).hexdigest()
disagreements = []


def enc(text: str) -> list[int]:
    a = [int(t) for t in hf.encode(text, add_special_tokens=False)]
    b = [int(t) for t in raw.encode(text, add_special_tokens=False).ids]
    if a != b:
        disagreements.append([text, a, b])
    return b


# ---------------------------------------------------------------------------------------------------------------
# 3. The exclusion sources, from the committed files.
c020 = committed_json("experiments/020-readout-decompilation/confirmation-v1.json")
c022 = committed_json("experiments/022-upstream-error-localization/confirmation-v1.json")
c023 = committed_json("experiments/023-block0-completion/confirmation-v1.json")
c024 = committed_json("experiments/024-readout-routing-nounness/confirmation-v1.json")
c025 = committed_json("experiments/025-nounness-direction-intervention/confirmation-v1.json")

texts_023_exclusion = list(c023["exclusion"]["frame_texts"])
texts_023_frames = [f["text_template"] for f in c023["frames"]]
TEXTS = set(texts_023_exclusion) | set(texts_023_frames)
out["sizes"] = {"texts_023_exclusion": len(texts_023_exclusion), "texts_023_exclusion_distinct": len(set(texts_023_exclusion)),
                "texts_023_frames": len(texts_023_frames), "frame_texts_union": len(TEXTS)}
# 023's exclusion digest self-check
out["c023_frame_texts_sha256_recomputes"] = sha(canonical(sorted(texts_023_exclusion))) == c023["exclusion"]["frame_texts_sha256"] or \
    sha(canonical(texts_023_exclusion)) == c023["exclusion"]["frame_texts_sha256"]
# broader text sources (not part of the rule; for the reviewer's novelty check)
broader_texts = set(TEXTS) | set(c022["exclusion"]["frame_texts"]) | {f["text_template"] for f in c022["frames"]} | {f["text_template"] for f in c020["frames"]}
for d in (c022, c023):
    for lst in d["candidates"]["frames"].values():
        broader_texts |= set(lst)
out["sizes"]["broader_texts_incl_unpicked_022_023_candidates"] = len(broader_texts)

cue_024_excl = [int(i) for i in c024["exclusion"]["cue_token_ids"]]
cue_024 = [int(c["token_id"]) for c in c024["cues"]]
cue_025 = [int(c["token_id"]) for c in c025["cues"]]
EARLIER = set(cue_024_excl) | set(cue_024) | set(cue_025)
cue_023_excl = [int(i) for i in c023["exclusion"]["cue_token_ids"]]
cue_023 = [int(c["token_id"]) for c in c023["cues"]]
cue_022_excl = [int(i) for i in c022["exclusion"]["cue_token_ids"]]
cue_022 = [int(c["token_id"]) for c in c022["cues"]]
cue_020 = [int(t["token_id"]) for t in c020["tokens"]]
out["sizes"].update({"cue_024_exclusion": len(cue_024_excl), "cue_024_exclusion_distinct": len(set(cue_024_excl)), "cue_024": len(set(cue_024)),
                     "cue_025": len(set(cue_025)), "earlier_cue_ids": len(EARLIER), "earlier_024_plus_cues": len(set(cue_024_excl) | set(cue_024))})
out["cross"] = {
    "024_exclusion_eq_023_exclusion_plus_023_cues": set(cue_024_excl) == set(cue_023_excl) | set(cue_023),
    "023_exclusion_contains_022_exclusion_and_022_cues": (set(cue_022_excl) | set(cue_022)) <= set(cue_023_excl),
    "022_exclusion_contains_020_tokens": set(cue_020) <= set(cue_022_excl),
    "025_blocked_earlier_cues_eq_024_excl_plus_024_cues": set(int(i) for i in c025["blocked"]["earlier_cues"]["ids"]) == set(cue_024_excl) | set(cue_024),
    "025_blocked_earlier_cues_size": len(c025["blocked"]["earlier_cues"]["ids"]),
    "024_target_forms_eq_025_blocked_target_forms": set(c024["target_noun_form_ids"]["ids"]) == set(c025["blocked"]["target_forms"]["ids"]),
    "024_frame_token_ids_eq_025_blocked_frame_tokens": set(c024["frame_token_ids"]["ids"]) == set(c025["blocked"]["frame_tokens"]["ids"]),
}

words_025 = [w for w in c025["candidates"]["adjective-reserve"]] + [w for w in c025["candidates"]["new-list"]] + \
    [w for pair in c025["candidates"]["ordinary-reserve"] for w in pair]
WORDS_025 = {str(w).lower() for w in words_025}
IDS_025 = set()
for w in sorted(WORDS_025):
    ids = enc(" " + w)
    if len(ids) == 1:
        IDS_025.add(ids[0])
# a wider reading (reviewer's extra): every form 025 named, including the new list's regular plurals and the cues' plurals
wider_words_025 = set(WORDS_025) | {w + "s" for w in c025["candidates"]["new-list"]} | {str(c.get("plural")).lower() for c in c025["cues"] if c.get("plural")}
wider_ids_025 = set(IDS_025)
for w in sorted(wider_words_025):
    ids = enc(" " + w)
    if len(ids) == 1:
        wider_ids_025.add(ids[0])
wider_ids_025 |= {int(c["plural_token_id"]) for c in c025["cues"] if c.get("plural_token_id") is not None}
out["sizes"].update({"words_025": len(words_025), "words_025_distinct": len(WORDS_025), "ids_025_single_token": len(IDS_025),
                     "wider_words_025": len(wider_words_025), "wider_ids_025": len(wider_ids_025)})

TARGETS = {int(i) for i in c024["target_noun_form_ids"]["ids"]}
# 020's confirmation nouns, recomputed from 020's committed file and the tokenizer
prior_from_file = set()
prior_from_tokenizer = set()
noun_rows = []
for noun in c020["nouns"]:
    prior_from_file |= {int(i) for i in noun["sg_ids"]} | {int(i) for i in noun["pl_ids"]}
    noun_rows.append([noun["lexical_key"], noun["sg_ids"], noun["pl_ids"], noun.get("single_token")])
PRIOR = prior_from_file
out["sizes"].update({"target_form_ids": len(TARGETS), "prior_noun_ids": len(PRIOR), "n_020_nouns": len(c020["nouns"])})
out["cross"]["prior_eq_025_blocked_prior_nouns"] = PRIOR == {int(i) for i in c025["blocked"]["prior_nouns"]["ids"]}
out["cross"]["prior_all_single_token_nouns"] = all(r[3] for r in noun_rows)

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
                "hundred", "thousand", "million", "billion", "dozen", "half", "single", "double", "both", "several", "few", "many", "couple", "pair"]
out["sizes"]["number_words"] = len(NUMBER_WORDS)
FRAME_TOKENS = {int(i) for i in c025["blocked"]["frame_tokens"]["ids"]}
out["sizes"]["frame_tokens_025"] = len(FRAME_TOKENS)

# ---------------------------------------------------------------------------------------------------------------
# 4. The mechanical walk.
CUES = {"ref": ("one", 581), "two": ("two", 767), "three": ("three", 1264), "many": ("many", 1142)}
for label, (word, tid) in CUES.items():
    assert enc(" " + word) == [tid], (word, enc(" " + word))

walk = []
taken: dict[int, int] = {}
eligible_numbers = []
for number, text in enumerate(CANDIDATES, start=1):
    reasons = []
    info = {"number": number, "text": text, "frame_id": f"coordinated-adjective-101-{number:02d}"}
    assert text.count("{cue}") == 1
    head, tail = text.split("{cue}")
    seqs = {}
    for label, (word, tid) in CUES.items():
        ids = enc(text.replace("{cue}", word))
        where = [i for i, t in enumerate(ids) if t == tid]
        if len(where) != 1:
            reasons.append(f"{label}: cue occurs {len(where)} times")
            continue
        p = where[0]
        # piecewise tokenization must agree with the whole-text tokenization (no merge across the slot)
        piece = enc(head.rstrip(" ")) + [tid] + enc(tail)
        if piece != ids:
            reasons.append(f"{label}: piecewise tokenization differs")
        seqs[label] = (ids, p)
    if len(seqs) == 4:
        ref_ids, p_c = seqs["ref"]
        for label, (ids, p) in seqs.items():
            if p != p_c or ids[:p] != ref_ids[:p_c] or ids[p + 1:] != ref_ids[p_c + 1:]:
                reasons.append(f"{label}: prefix/suffix differ from the reference prompt")
        suffix = ref_ids[p_c + 1:]
        adjective = tail.strip()
        own = enc(" " + adjective)
        if not (len(suffix) == 1 and suffix == own and tail == " " + adjective):
            reasons.append(f"adjective {adjective!r} is not one final token (suffix {suffix}, own {own})")
        if not 4 <= p_c <= 7:
            reasons.append(f"p_c {p_c} outside 4-7")
        p_t = p_c + len(suffix)
        if p_t != p_c + 1:
            reasons.append("p_t != p_c + 1")
        adj_id = suffix[0] if len(suffix) == 1 else None
        info.update({"prefix_ids": ref_ids[:p_c], "suffix_ids": suffix, "p_c": p_c, "p_t": p_t, "adjective": adjective, "adjective_id": adj_id,
                     "token_ids": {label: seqs[label][0] for label in CUES}})
        if text in TEXTS:
            reasons.append("text is an earlier frame text")
        if adj_id is not None:
            if adj_id in EARLIER:
                reasons.append("earlier cue")
            if adjective.lower() in WORDS_025 or adj_id in IDS_025:
                reasons.append("025 candidate word")
            if adj_id in TARGETS:
                reasons.append("target-noun form (pool)")
            if adj_id in PRIOR:
                reasons.append("target-noun form (020 confirmation nouns)")
            if adjective.lower() in NUMBER_WORDS:
                reasons.append("number word")
            if adj_id in taken:
                reasons.append(f"adjective of candidate {taken[adj_id]}")
        # reviewer's extras (not rules): broader text novelty, wider 025 forms, frame tokens
        info["extra"] = {"in_broader_texts": text in broader_texts, "in_wider_025": (adjective.lower() in wider_words_025) or (adj_id in wider_ids_025),
                         "in_frame_tokens_025": adj_id in FRAME_TOKENS,
                         "case_insensitive_text_match": any(text.lower() == t.lower() for t in broader_texts)}
    info["reasons"] = reasons
    info["eligible"] = not reasons
    if not reasons:
        taken[info["adjective_id"]] = number
        eligible_numbers.append(number)
    walk.append(info)

out["eligible_numbers"] = eligible_numbers
out["first_36"] = eligible_numbers[:36]
out["first_36_are_1_to_36"] = eligible_numbers[:36] == list(range(1, 37))
out["walk"] = walk
picks = eligible_numbers[:36]
out["frame_token_blocked_picks"] = [(n, walk[n - 1]["adjective"]) for n in picks if walk[n - 1]["extra"]["in_frame_tokens_025"]]
out["frame_token_blocked_all60"] = [(w["number"], w["adjective"]) for w in walk if w.get("extra", {}).get("in_frame_tokens_025")]
out["wider_025_hits"] = [(w["number"], w["adjective"]) for w in walk if w.get("extra", {}).get("in_wider_025")]
out["broader_text_hits"] = [w["number"] for w in walk if w.get("extra", {}).get("in_broader_texts") or w.get("extra", {}).get("case_insensitive_text_match")]

# ---------------------------------------------------------------------------------------------------------------
# 5. The manifests (plan § Constants: CUE_ORDER and arm orders; § Ledger: keys).
REF_ARMS_ROLE = {"REF", "SUF-READ", "SUF-OFF", "SUF-NEG", "SUF-HALF", "PC-SWAP-REF", "HEAD-SUF", "HEAD-SUF-V"} | {f"SUF-ISO-{j}" for j in range(1, 9)}
TWO_ORDER = ["REF", "PL", "NOOP", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF", "PC-SWAP-PL", "PC-SWAP-REF"] + \
    [f"NEC-ISO-{j}" for j in range(1, 9)] + [f"SUF-ISO-{j}" for j in range(1, 9)] + ["HEAD-NEC", "HEAD-SUF", "HEAD-NEC-V", "HEAD-SUF-V"]
STRATUM_ORDER = ["PL", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF", "PC-SWAP-PL", "PC-SWAP-REF"]
assert len(TWO_ORDER) == 33 and len(STRATUM_ORDER) == 11
CUE_ORDER = ["two", "three", "many"]
frame_ids = [f"coordinated-adjective-101-{n:02d}" for n in picks]
s1 = [f"{fid}|ref|581|S1" for fid in frame_ids]
s2_ordered = []
run_order_tags = []
for fid in frame_ids:
    for cue in CUE_ORDER:
        word, tid = CUES[cue]
        arms = TWO_ORDER if cue == "two" else STRATUM_ORDER
        for arm in arms:
            tag = "REF" if arm == "REF" else f"{arm}@{cue}"
            prompt = f"{fid}|ref|581" if arm in REF_ARMS_ROLE else f"{fid}|{cue}|{tid}"
            s2_ordered.append(f"{prompt}|{tag}")
            if fid == frame_ids[0]:
                run_order_tags.append(tag)
out["manifest_counts"] = {"S1": len(s1), "S1_distinct": len(set(s1)), "S2": len(s2_ordered), "S2_distinct": len(set(s2_ordered)),
                          "s1_s2_disjoint": not set(s1) & set(s2_ordered)}
manifest = {"S1-REF": sorted(s1), "S2-TARGET": sorted(s2_ordered)}
out["manifest_sha256"] = sha(canonical(manifest))
untagged = {"|".join(k.split("|")[:3]) for k in s1 + s2_ordered}
out["untagged_count"] = len(untagged)
out["run_order_tags_frame1"] = run_order_tags
out["s2_ordered_first_frame"] = s2_ordered[:55]

json.dump({"out": out, "manifest": manifest, "untagged": sorted(untagged), "tokenizer_disagreements": disagreements,
           "EARLIER": sorted(EARLIER), "TEXTS": sorted(TEXTS), "WORDS_025": sorted(WORDS_025), "IDS_025": sorted(IDS_025),
           "TARGETS": sorted(TARGETS), "PRIOR": sorted(PRIOR), "NUMBER_WORDS": NUMBER_WORDS, "FRAME_TOKENS": sorted(FRAME_TOKENS)},
          open(HERE / "independent_derivation.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k not in ("walk", "candidates_from_design", "s2_ordered_first_frame")}, indent=1, ensure_ascii=False))
print("tokenizer disagreements:", len(disagreements))
print("modules imported from neural_decompiler:", [m for m in sys.modules if m.startswith("neural_decompiler")])
