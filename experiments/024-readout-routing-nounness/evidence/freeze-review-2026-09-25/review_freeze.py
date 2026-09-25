"""Independent, read-only review of Experiment 024's production freeze (tokenizer, text and committed files only).

No model is loaded: torch.nn.Module.__call__, neural_decompiler.models.load_model, every plural_mechanism capture or
intervention entry point, torch.load and the safetensors loaders all refuse (and count). An audit hook refuses any
write under the repository and logs every file opened for reading, per stage. Output goes to this scratch directory.
"""
import rguard  # noqa: F401  (first: the audit hook)

import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(rguard.ROOT)
HERE = Path(__file__).resolve().parent
CONF = ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json"
C023 = ROOT / "experiments/023-block0-completion/confirmation-v1.json"
C022 = ROOT / "experiments/022-upstream-error-localization/confirmation-v1.json"
DESIGN = ROOT / "docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md"
PLAN = ROOT / "docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md"
CELLS_INDEX = ROOT / "experiments/023-block0-completion/exposed-cells.json"
EXPECTED_FILE_SHA = "68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60"
EXPECTED_CONTENT_SHA = "87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87"

RESULTS = {}
FAILS = []
LOG = []


def say(text=""):
    print(text, flush=True)
    LOG.append(str(text))


def check(name, condition, detail=""):
    RESULTS.setdefault("checks", {})[name] = bool(condition)
    if not condition:
        FAILS.append(f"{name}: {detail}")
    say(f"  [{'ok' if condition else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}")


def my_canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_hex(data: bytes):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- guard self-test (safe probe: the directory does not exist)
rguard.stage("selftest")
probe = ROOT / "__reviewer_nonexistent_dir__" / "probe.txt"
assert not probe.parent.exists()
try:
    open(probe, "w")
    raise SystemExit("guard self-test failed: no refusal")
except PermissionError:
    pass
say(f"guard self-test: a write under the repository is refused (refusals so far: {len(rguard.STATE['refused'])})")

# ---------------------------------------------------------------- block every model path before any project import
rguard.stage("imports")
import torch  # noqa: E402

COUNTS = Counter()


def _refuser(name):
    def refuse(*args, **kwargs):
        COUNTS[name] += 1
        raise RuntimeError(f"reviewer guard: {name} refused")
    return refuse


torch.nn.Module.__call__ = _refuser("torch.nn.Module.__call__")
torch.load = _refuser("torch.load")
try:
    import safetensors.torch as _st  # noqa: E402
    _st.load_file = _refuser("safetensors.torch.load_file")
    _st.load = _refuser("safetensors.torch.load")
    import safetensors as _s  # noqa: E402
    _s.safe_open = _refuser("safetensors.safe_open")
except ImportError:
    pass

from neural_decompiler import models  # noqa: E402

models.load_model = _refuser("models.load_model")
from neural_decompiler import plural_mechanism as pm  # noqa: E402

for _name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, _name):
        setattr(pm, _name, _refuser(f"pm.{_name}"))
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler.models import PYTHIA_70M  # noqa: E402

say(f"load_model bound in rr/ul/b0c namespaces: {[m.__name__ for m in (rr, ul, b0c) if getattr(m, 'load_model', None) is not None]}")


def git_tracked(path):
    try:
        subprocess.run(["git", "--no-optional-locks", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


# ================================================================ Item 2/3 raw inputs
rguard.stage("raw_files")
raw_024 = CONF.read_bytes()
file_sha = sha_hex(raw_024)
c024 = json.loads(raw_024)
c023_raw = C023.read_bytes()
c023 = json.loads(c023_raw)
c022_raw = C022.read_bytes()
c022 = json.loads(c022_raw)
sha023, sha022 = sha_hex(c023_raw), sha_hex(c022_raw)
say("\n# Item 7 (part): the file on disk")
check("file sha256 == expected", file_sha == EXPECTED_FILE_SHA, file_sha)
content_sha = sha_hex(my_canonical({k: v for k, v in c024.items() if k != "content_sha256"}).encode("utf-8"))
check("content sha256 (own canonical JSON without the key) == recorded == expected", content_sha == c024["content_sha256"] == EXPECTED_CONTENT_SHA, content_sha)
check("file bytes == own canonical JSON + newline", raw_024 == (my_canonical(c024) + "\n").encode("utf-8"))
RESULTS["file"] = {"file_sha256": file_sha, "content_sha256": content_sha, "bytes": len(raw_024)}

# ================================================================ frozen inputs (as Runner._base does, replicated; no Runner method called)
rguard.stage("frozen_inputs")
blobs = rr.assert_frozen_blobs()
inputs = ul.load_frozen_inputs(ROOT, tracked=git_tracked)
pool = inputs.pool
rguard.stage("verify_022_023")
digests_022 = b0c.verify_022_inputs(ROOT)
digests_023 = rr.verify_023_inputs(ROOT)
confirmation_023_obj = b0c.load_confirmation_023(C023, inputs, c022, sha022)
say(f"\nfrozen inputs loaded; 022 and 023 committed inputs verified ({len(digests_022)} + {len(digests_023)} digests); 023 confirmation loaded")

# ================================================================ Item 3: exclusion sets, independently from committed files
rguard.stage("exclusions")
say("\n# Item 3: exclusion sets (own derivation from committed JSON and the frozen pool)")
ex022 = [int(i) for i in c022["exclusion"]["cue_token_ids"]]
ex023 = [int(i) for i in c023["exclusion"]["cue_token_ids"]]
check("022 exclusion: 303 distinct sorted ids, digest self-consistent", len(ex022) == len(set(ex022)) == 303 and ex022 == sorted(ex022)
      and c022["exclusion"]["cue_token_ids_sha256"] == sha_hex(my_canonical(ex022).encode()), len(ex022))
check("023 exclusion: 327 distinct sorted ids, digest self-consistent", len(ex023) == len(set(ex023)) == 327 and ex023 == sorted(ex023)
      and c023["exclusion"]["cue_token_ids_sha256"] == sha_hex(my_canonical(ex023).encode()), len(ex023))
cues022 = {int(c["token_id"]) for c in c022["cues"]}
fcues022 = {int(v) for f in c022["frames"] for v in f["cue_ids"].values()}
chain023 = set(ex022) | cues022 | fcues022
check("chain: 022 exclusion ∪ 022's 24 cues ∪ 022 frames' cue ids == 023's 327", chain023 == set(ex023), f"{len(chain023)}; 022 cues {len(cues022)}, frame cue ids {sorted(fcues022)}")
cues023 = {int(c["token_id"]) for c in c023["cues"]}
fcues023 = {int(v) for f in c023["frames"] for v in f["cue_ids"].values()}
earlier = set(ex023) | cues023 | fcues023
check("earlier cue ids = 023's 327 ∪ 023's 24 cues ∪ 023 frames' cue ids: 351", len(earlier) == 351 and len(cues023) == 24 and not (cues023 & set(ex023)),
      f"{len(earlier)}; 023 cues {len(cues023)}; 023 frame cue ids {sorted(fcues023)} (already in 327: {fcues023 <= set(ex023)})")
# the code's claim: 022's and 023's exclusions reproduce from the frozen inputs now
base022_now = ul.extract_exclusion(inputs)
check("022's 303 reproduce from the frozen inputs (ul.extract_exclusion)", base022_now["cue_token_ids"] == ex022 and base022_now["cue_token_ids_sha256"] == c022["exclusion"]["cue_token_ids_sha256"])
base023_now = b0c.exclusion(inputs, c022, sha022)
check("023's 327 reproduce from the frozen inputs (b0c.exclusion), digest and sources", base023_now["cue_token_ids"] == ex023
      and base023_now["cue_token_ids_sha256"] == c023["exclusion"]["cue_token_ids_sha256"] and base023_now["sources"] == c023["exclusion"]["sources"])
file_ex = c024["exclusion"]
check("file exclusion ids == own 351 (sorted), digest == sha256(canonical ids)", file_ex["cue_token_ids"] == sorted(earlier)
      and file_ex["cue_token_ids_sha256"] == sha_hex(my_canonical(sorted(earlier)).encode()))
last_source = file_ex["sources"][-1]
check("file exclusion sources = 023's sources + a confirmation-023 source with 023's file sha and content sha",
      file_ex["sources"][:-1] == c023["exclusion"]["sources"] and last_source["source"] == "confirmation-023"
      and last_source["files"] == [{"path": "experiments/023-block0-completion/confirmation-v1.json", "file_sha256": sha023}]
      and last_source["content_sha256"] == c023["content_sha256"] and last_source["tokens"] == 24 and last_source["frames"] == 18,
      f"{len(file_ex['sources'])} sources: {[s['source'] for s in file_ex['sources']]}")
# the exclusion sources' file digests are the current committed files
stale = []
for source in file_ex["sources"]:
    for entry in source.get("files", []):
        if entry.get("file_sha256") is not None and sha_hex((ROOT / entry["path"]).read_bytes()) != entry["file_sha256"]:
            stale.append(entry["path"])
check("every exclusion source file digest recorded in the file matches the committed file now", not stale, stale)

nouns = list(pool.nouns)
forms = set()
for noun in nouns:
    forms |= {int(i) for i in noun.sg_ids} | {int(i) for i in noun.pl_ids}
single = [n for n in nouns if len(n.sg_ids) == 1]
check("target-noun form ids: 161 (every sg/pl piece of every pool noun)", len(forms) == 161,
      f"{len(forms)} from {len(nouns)} nouns ({len(single)} single-token, {len(nouns) - len(single)} multi-token: {[n.lexical_key for n in nouns if len(n.sg_ids) != 1]})")
check("file target_noun_form_ids == own (sorted) with digest", c024["target_noun_form_ids"] == {"ids": sorted(forms), "sha256": sha_hex(my_canonical(sorted(forms)).encode())})
frames = list(pool.frames)
ftoks = set()
for frame in frames:
    ftoks |= {int(i) for i in frame.prefix_ids} | {int(i) for i in frame.suffix_ids}
template_counts = Counter(f.template_id for f in frames)
check("108 exposed frames: 36 cardinal, 36 quantifier, 36 coordinated-adjective (72 cue-final + 36 coordinated)",
      len(frames) == 108 and len({f.frame_id for f in frames}) == 108 and template_counts == Counter({"cardinal": 36, "quantifier": 36, "coordinated-adjective": 36}),
      dict(template_counts))
check("exposed-frame token ids: 316 (every prefix/suffix id of the 108)", len(ftoks) == 316, len(ftoks))
check("file frame_token_ids == own (sorted) with digest", c024["frame_token_ids"] == {"ids": sorted(ftoks), "sha256": sha_hex(my_canonical(sorted(ftoks)).encode())})
index = json.loads(CELLS_INDEX.read_text(encoding="utf-8"))
check("the 108 pool frames are 023's committed exposed-cells frames and 023's committed exposed_frame_ids",
      sorted(f.frame_id for f in frames) == sorted(fid for fid, _, _ in index["frames"]) and [f.frame_id for f in frames] == c023["exposed_frame_ids"])
check("file exposed_frame_ids == pool order; reference_cue_ids == pool's", c024["exposed_frame_ids"] == [f.frame_id for f in frames]
      and c024["reference_cue_ids"] == {k: int(v) for k, v in pool.reference_ids.items()}, c024["reference_cue_ids"])
RESULTS["sets"] = {"earlier": len(earlier), "ex023": len(ex023), "forms": len(forms), "frame_tokens": len(ftoks), "nouns": len(nouns), "single_token_nouns": len(single)}

# ================================================================ Item 2: candidate lists vs the design text
rguard.stage("design_text")
say("\n# Item 2: the candidate lists against the design document's table")
doc = DESIGN.read_text(encoding="utf-8")
rows = {}
for line in doc.splitlines():
    m = re.match(r"^\| (N|B/D lemmas|C/E lemmas) \| (.*) \|$", line)
    if m:
        rows[m.group(1)] = m.group(2)
check("the design table has the three ordered rows", set(rows) == {"N", "B/D lemmas", "C/E lemmas"}, list(rows))


def parse_row(row):
    items, bold = [], False
    for raw in row.split(","):
        s = raw.strip()
        starts = s.startswith("**")
        if starts:
            bold = True
            s = s[2:]
        ends = s.endswith("**")
        if ends:
            s = s[:-2]
        is_bold = bold
        if ends:
            bold = False
        note = None
        m = re.search(r"\((.*?)\)", s)
        if m:
            note = m.group(1)
            s = (s[:m.start()] + s[m.end():]).strip()
        crossed = "✗" in s
        s = s.replace("✗", "").strip()
        assert re.fullmatch(r"[a-z]+", s), s
        items.append({"word": s, "bold": is_bold, "crossed": crossed, "note": note})
    return items


design_N, design_BD, design_CE = (parse_row(rows[k]) for k in ("N", "B/D lemmas", "C/E lemmas"))
check("design N list == rr.N_CANDIDATES (order and content)", [i["word"] for i in design_N] == list(rr.N_CANDIDATES), len(design_N))
check("design B/D lemmas == singulars of rr.MEASURE_LEMMAS", [i["word"] for i in design_BD] == [s for s, _ in rr.MEASURE_LEMMAS], len(design_BD))
check("design C/E lemmas == singulars of rr.ORDINARY_LEMMAS", [i["word"] for i in design_CE] == [s for s, _ in rr.ORDINARY_LEMMAS], len(design_CE))
# the plurals: written out in the plan; checked against the plan text and the English regular-plural rule
plan = PLAN.read_text(encoding="utf-8")
m_block = plan[plan.index("`MEASURE_LEMMAS`:"):plan.index("`ORDINARY_LEMMAS`:")]
o_block = plan[plan.index("`ORDINARY_LEMMAS`:"):plan.index("B is the plural and D the singular")]
plan_measure = re.findall(r"([a-z]+)/([a-z]+)", m_block)
plan_ordinary = re.findall(r"([a-z]+)/([a-z]+)", o_block)
check("plan's written-out measure pairs == rr.MEASURE_LEMMAS", tuple(plan_measure) == tuple(rr.MEASURE_LEMMAS), len(plan_measure))
check("plan's written-out ordinary pairs == rr.ORDINARY_LEMMAS", tuple(plan_ordinary) == tuple(rr.ORDINARY_LEMMAS), len(plan_ordinary))
plan_n = re.findall(r"[a-z]+", plan[plan.index("`N_CANDIDATES`:") + len("`N_CANDIDATES`:"):plan.index("`MEASURE_LEMMAS`:")])
check("plan's N_CANDIDATES text == rr.N_CANDIDATES", plan_n == list(rr.N_CANDIDATES), len(plan_n))


def regular_plural(s):
    return s + "es" if s.endswith(("s", "x", "z", "ch", "sh")) else s + "s"


check("every listed plural is the regular English plural of its lemma", all(p == regular_plural(s) for s, p in rr.MEASURE_LEMMAS + rr.ORDINARY_LEMMAS),
      [(s, p) for s, p in rr.MEASURE_LEMMAS + rr.ORDINARY_LEMMAS if p != regular_plural(s)])
check("file candidates == rr lists verbatim", c024["candidates"] == {"N": list(rr.N_CANDIDATES), "measure": [list(p) for p in rr.MEASURE_LEMMAS],
                                                                    "ordinary": [list(p) for p in rr.ORDINARY_LEMMAS]})
all_words = list(rr.N_CANDIDATES) + [w for p in rr.MEASURE_LEMMAS + rr.ORDINARY_LEMMAS for w in p]
check("no duplicate word across the three lists", len(all_words) == len(set(all_words)))
# expected picks / reserves text
exp_text = doc[doc.index("**Expected picks:**"):doc.index("**Reserves**")]
exp_N = re.findall(r"[a-z]+", exp_text[exp_text.index("**N:**") + 6:exp_text.index("**B / D:**")])
exp_BD = re.findall(r"([a-z]+)\s*/\s*([a-z]+)", exp_text[exp_text.index("**B / D:**"):exp_text.index("**C / E:**")])
exp_CE = re.findall(r"([a-z]+)\s*/\s*([a-z]+)", exp_text[exp_text.index("**C / E:**"):])
check("design expected picks == rr.EXPECTED_PICKS", exp_N == list(rr.EXPECTED_PICKS["N"]) and [s for _, s in exp_BD] == list(rr.EXPECTED_PICKS["measure"])
      and [s for _, s in exp_CE] == list(rr.EXPECTED_PICKS["ordinary"]) and all(p == regular_plural(s) for p, s in exp_BD + exp_CE))
res_text = doc[doc.index("**Reserves**"):doc.index("The manifest is 40 cues")]

# ================================================================ Item 2: own mechanical selection with the pinned tokenizer
rguard.stage("tokenizer_load")
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision=PYTHIA_70M.revision, local_files_only=True)
say(f"\ntokenizer: {type(tok).__name__}, revision {PYTHIA_70M.revision}, vocab {len(tok)}")
rguard.stage("selection")


def enc(word):
    return [int(t) for t in tok.encode(" " + word, add_special_tokens=False)]


def status(word, picked):
    ids = enc(word)
    facts = {"ids": ids}
    if len(ids) != 1:
        return None, f"{len(ids)} tokens with a leading space", facts
    t = ids[0]
    facts.update({"earlier": t in earlier, "target_form": t in forms, "frame_token": t in ftoks, "picked": t in picked})
    # the code's precedence for the recorded reason: exclusion, target form, frame token, already picked
    if t in earlier:
        return None, f"token id {t}: already used as a cue by Experiments 005–023", facts
    if t in forms:
        return None, f"token id {t}: a form of a target noun", facts
    if t in ftoks:
        return None, f"token id {t}: a token of an exposed frame", facts
    if t in picked:
        return None, f"token id {t}: already picked", facts
    return t, "eligible", facts


say("\n# Item 2: own mechanical selection (first 8 eligible per list)")
picked = set()
my_rejected, my_reserves, facts_all = [], {"N": [], "measure": [], "ordinary": []}, {}
my_N = []
for rank, word in enumerate(rr.N_CANDIDATES):
    t, reason, facts = status(word, picked)
    facts_all[word] = facts
    if t is None:
        my_rejected.append({"candidate": word, "list": "N", "rank": rank, "reason": reason})
    elif len(my_N) < 8:
        my_N.append((word, t, rank))
        picked.add(t)
    else:
        my_reserves["N"].append(word)
my_lemmas = {"measure": [], "ordinary": []}
for name, pairs in (("measure", rr.MEASURE_LEMMAS), ("ordinary", rr.ORDINARY_LEMMAS)):
    for rank, (s, p) in enumerate(pairs):
        ts, rs, fs = status(s, picked)
        tp, rp, fp = status(p, picked)
        facts_all[s], facts_all[p] = fs, fp
        if ts is None or tp is None or ts == tp:
            my_rejected.append({"candidate": f"{s}/{p}", "list": name, "rank": rank, "reason": f"{s}: {rs}; {p}: {rp}"})
        elif len(my_lemmas[name]) < 8:
            my_lemmas[name].append((s, ts, p, tp, rank))
            picked |= {ts, tp}
        else:
            my_reserves[name].append(f"{s}/{p}")
my_cues = [{"candidate_rank": r, "class": "N", "form": "word", "lemma": w, "token_id": t, "word": w} for w, t, r in my_N]
for cls, name, plural in (("B", "measure", True), ("D", "measure", False), ("C", "ordinary", True), ("E", "ordinary", False)):
    my_cues += [{"candidate_rank": r, "class": cls, "form": "plural" if plural else "singular", "lemma": s, "token_id": tp if plural else ts,
                 "word": p if plural else s} for s, ts, p, tp, r in my_lemmas[name]]
EXPECTED_IDS = {
    "N": [("honest", 8274), ("polite", 30405), ("rude", 30446), ("sleepy", 48849), ("wise", 15822), ("lucky", 13476), ("merry", 49570), ("nervous", 11219)],
    "B": [("gallons", 42616), ("ounces", 28409), ("acres", 20046), ("herds", 47862), ("crowds", 24597), ("bundles", 25663), ("clusters", 9959), ("litres", 47026)],
    "D": [("gallon", 46740), ("ounce", 38831), ("acre", 36982), ("herd", 33361), ("crowd", 9539), ("bundle", 13204), ("cluster", 7368), ("litre", 43803)],
    "C": [("apples", 28580), ("horses", 12074), ("doctors", 11576), ("kings", 25346), ("rabbits", 29948), ("poets", 32976), ("dragons", 41705), ("lions", 44536)],
    "E": [("apple", 19126), ("horse", 8815), ("doctor", 7345), ("king", 6963), ("rabbit", 17876), ("poet", 17502), ("dragon", 22159), ("lion", 27405)],
}
mine_by_class = {cls: [(c["word"], c["token_id"]) for c in my_cues if c["class"] == cls] for cls in rr.CLASSES}
for cls in rr.CLASSES:
    say(f"  {cls}: " + ", ".join(f"{w} {t}" for w, t in mine_by_class[cls]))
check("own picks == the 40 expected words and ids (task list)", mine_by_class == EXPECTED_IDS)
check("own picks == the file's cues (all fields, order N,B,D,C,E)", my_cues == c024["cues"])
check("own rejected (30) == the file's rejected (list, candidate, rank, exact reason text)",
      sorted(my_rejected, key=lambda e: (e["list"], e["rank"])) == sorted(c024["rejected"], key=lambda e: (e["list"], e["rank"])) and len(my_rejected) == 30,
      f"{len(my_rejected)} ({Counter(e['list'] for e in my_rejected)})")
check("own rejected in the file's exact order", my_rejected == c024["rejected"])
check("own reserves == the file's reserves", my_reserves == c024["reserves"], {k: len(v) for k, v in my_reserves.items()})
design_crossed = {"N": {i["word"] for i in design_N if i["crossed"]}, "measure": {i["word"] for i in design_BD if i["crossed"]},
                  "ordinary": {i["word"] for i in design_CE if i["crossed"]}}
mine_crossed = {"N": {e["candidate"] for e in my_rejected if e["list"] == "N"},
                "measure": {e["candidate"].split("/")[0] for e in my_rejected if e["list"] == "measure"},
                "ordinary": {e["candidate"].split("/")[0] for e in my_rejected if e["list"] == "ordinary"}}
check("design ✗ marks == own ineligible entries, list by list", design_crossed == mine_crossed,
      {k: sorted(design_crossed[k] ^ mine_crossed[k]) for k in design_crossed})
design_bold = {"N": [i["word"] for i in design_N if i["bold"]], "measure": [i["word"] for i in design_BD if i["bold"]],
               "ordinary": [i["word"] for i in design_CE if i["bold"]]}
check("design bold entries == own picks, in order", design_bold == {"N": [w for w, _, _ in my_N], "measure": [e[0] for e in my_lemmas["measure"]],
                                                                    "ordinary": [e[0] for e in my_lemmas["ordinary"]]}, design_bold)
annotated = {i["word"]: i["note"] for i in design_BD + design_CE + design_N if i["note"]}
check("design '(frame token)' annotations are exactly teacher and farmer, both frame tokens here",
      annotated == {"teacher": "frame token", "farmer": "frame token"} and facts_all["teacher"].get("frame_token") and facts_all["farmer"].get("frame_token"), annotated)
for w in ("teacher", "farmer", "crate", "basket"):
    f = facts_all[w]
    say(f"    {w}: ids {f['ids']}, earlier {f.get('earlier')}, target_form {f.get('target_form')}, frame_token {f.get('frame_token')}; "
        f"plural {regular_plural(w)}: ids {facts_all[regular_plural(w)]['ids']}")
check("teacher, farmer, crate and basket singulars are single tokens that are exposed-frame tokens (and not earlier cues / target forms)",
      all(len(facts_all[w]["ids"]) == 1 and facts_all[w]["frame_token"] and not facts_all[w]["earlier"] and not facts_all[w]["target_form"]
          for w in ("teacher", "farmer", "crate", "basket")))
check("crates and baskets are multi-token; teachers and farmers are eligible single tokens",
      len(facts_all["crates"]["ids"]) > 1 and len(facts_all["baskets"]["ids"]) > 1 and len(facts_all["teachers"]["ids"]) == 1 and len(facts_all["farmers"]["ids"]) == 1
      and not any(facts_all[w][k] for w in ("teachers", "farmers") for k in ("earlier", "target_form", "frame_token")))
# which frames hold teacher/farmer/crate/basket
where = {}
for w in ("teacher", "farmer", "crate", "basket"):
    t = facts_all[w]["ids"][0]
    where[w] = [f.frame_id for f in frames if t in f.prefix_ids or t in f.suffix_ids]
say(f"    frames holding them: {where}")
res_N_first = re.findall(r"N from ([a-z]+)", res_text)
res_BD = re.findall(r"[a-z]+", res_text[res_text.index("B/D") + 3:res_text.index("C/E")])
res_CE = re.findall(r"[a-z]+", res_text[res_text.index("C/E") + 3:])
check("design reserves text == own reserves (N from anxious; B/D and C/E lists)",
      res_N_first == [my_reserves["N"][0]] and res_BD == [x.split("/")[0] for x in my_reserves["measure"]]
      and res_CE == [x.split("/")[0] for x in my_reserves["ordinary"]], (res_N_first, res_BD, res_CE))
check("no reserve was needed: every list filled its quota before its first reserve; picks are the design's expected picks",
      len(my_N) == 8 and all(len(v) == 8 for v in my_lemmas.values()) and [w for w, _, _ in my_N] == list(rr.EXPECTED_PICKS["N"])
      and [e[0] for e in my_lemmas["measure"]] == list(rr.EXPECTED_PICKS["measure"]) and [e[0] for e in my_lemmas["ordinary"]] == list(rr.EXPECTED_PICKS["ordinary"]))
# the "already picked" rule never fired, and the picks do not depend on it
check("the 'already picked' rule never decided anything", all(not f.get("picked") for f in facts_all.values()))
# 'last picked rank' < every reserve's rank (mechanical first-8)
say(f"  last picked candidate ranks: N {my_N[-1][2]}, measure {my_lemmas['measure'][-1][4]}, ordinary {my_lemmas['ordinary'][-1][4]}")

# ================================================================ Item 3: eligibility of the 40 directly
rguard.stage("eligibility")
say("\n# Item 3: eligibility of the 40 (direct)")
ids40 = [c["token_id"] for c in my_cues]
check("40 distinct ids", len(set(ids40)) == 40)
check("each cue: exactly one id with the leading space; decodes back to ' ' + word; the no-space form is a different string",
      all(enc(c["word"]) == [c["token_id"]] and tok.decode([c["token_id"]]) == " " + c["word"] for c in my_cues))
check("no selected id is an earlier cue id / target-noun form / exposed-frame token",
      not (set(ids40) & earlier) and not (set(ids40) & forms) and not (set(ids40) & ftoks),
      {"earlier": len(set(ids40) & earlier), "forms": len(set(ids40) & forms), "frames": len(set(ids40) & ftoks)})
# lemma pairing: B/D and C/E position by position
pair_ok = True
for pl_cls, sg_cls, lemmas in (("B", "D", rr.MEASURE_LEMMAS), ("C", "E", rr.ORDINARY_LEMMAS)):
    plural_of = dict(lemmas)
    P = [c for c in my_cues if c["class"] == pl_cls]
    S = [c for c in my_cues if c["class"] == sg_cls]
    for p, s in zip(P, S):
        pair_ok &= (p["lemma"] == s["lemma"] == s["word"] and p["word"] == plural_of[s["word"]] and p["form"] == "plural" and s["form"] == "singular"
                    and p["candidate_rank"] == s["candidate_rank"])
check("B/D and C/E are position-wise lemma pairs (same lemma, same candidate rank, plural = listed plural)", pair_ok)
check("class assignment only from the curated lists: N ⊂ N_CANDIDATES; B/D lemmas ⊂ MEASURE_LEMMAS; C/E ⊂ ORDINARY_LEMMAS",
      all(c["word"] in rr.N_CANDIDATES for c in my_cues if c["class"] == "N")
      and all(c["lemma"] in dict(rr.MEASURE_LEMMAS) for c in my_cues if c["class"] in "BD")
      and all(c["lemma"] in dict(rr.ORDINARY_LEMMAS) for c in my_cues if c["class"] in "CE"))
# textual occurrence of the 40 words (any case, any tokenization) in the 108 exposed frames and their text templates
frame_texts = {f.frame_id: tok.decode(list(f.prefix_ids)) + " {cue}" + tok.decode(list(f.suffix_ids)) for f in frames}
hits = {}
for c in my_cues:
    pattern = re.compile(r"(?<![A-Za-z])" + re.escape(c["word"]) + r"(?![A-Za-z])", re.IGNORECASE)
    for fid, text in frame_texts.items():
        if pattern.search(text) or pattern.search(next(f.text_template for f in frames if f.frame_id == fid)):
            hits.setdefault(c["word"], []).append(fid)
check("none of the 40 words occurs as a word (any case/tokenization) in any exposed frame's decoded text or template", not hits, hits)
# the lemma forms of N and of the picked lemmas as substrings of frame tokens' decoded pieces (e.g. ' King' or 'kings')
piece_hits = {}
for t in sorted(ftoks):
    piece = tok.decode([t]).strip().lower()
    for c in my_cues:
        if piece == c["word"].lower():
            piece_hits.setdefault(c["word"], []).append((t, tok.decode([t])))
check("no exposed-frame token decodes (case-insensitively, stripped) to one of the 40 words", not piece_hits, piece_hits)
# target-noun lexical keys vs the 40 lemmas
noun_keys = {n.lexical_key.lower() for n in nouns}
check("no cue lemma or word is a pool noun's lexical key", not ({c["lemma"] for c in my_cues} | {c["word"] for c in my_cues}) & noun_keys,
      sorted(({c["lemma"] for c in my_cues} | {c["word"] for c in my_cues}) & noun_keys))

# ================================================================ Item 4: the manifest
rguard.stage("manifest")
say("\n# Item 4: the manifest")
keys = [f"{f.frame_id}|{c['word']}|{c['token_id']}" for f in frames for c in my_cues]
check("4,320 keys, unique", len(keys) == len(set(keys)) == 4320)
check("own key format == pm.Prompt.key", keys == [pm.Prompt(f, int(c["token_id"]), c["word"]).key for f in frames for c in my_cues])
manifest = {"S2-TARGET": sorted(keys)}
check("file manifest == own (canonical sorted order as stored)", c024["manifest"] == manifest and c024["manifest"]["S2-TARGET"] == sorted(c024["manifest"]["S2-TARGET"]))
manifest_sha = sha_hex(my_canonical(manifest).encode("utf-8"))
check("manifest sha256 == pm.sha256_text(pm.canonical_json(manifest)) and starts fcc437fc", manifest_sha == pm.sha256_text(pm.canonical_json(manifest))
      and manifest_sha.startswith("fcc437fc"), manifest_sha)
check("file counts == {classes: 8 each}", c024["counts"] == {"classes": {cls: 8 for cls in rr.CLASSES}})
ledger020 = set(inputs.closure["ledger"])
conf020 = {p.key for p in inputs.confirmation_020.all_prompts}


def manifest_keys(payload):
    m = payload["manifest"]
    return set(m["S1-REF"]) | set(m["S1-VALIDITY"]) | set(m["S2-TARGET"]["Y1"]) | set(m["S2-TARGET"]["Y2"])


m022, m023 = manifest_keys(c022), manifest_keys(c023)
check("023's raw-JSON manifest keys == the verified 023 object's manifest_keys()", m023 == set(confirmation_023_obj.manifest_keys()))
spent = ledger020 | conf020 | m022 | m023
sizes = {"020_ledger": len(ledger020), "020_confirmation_set": len(conf020), "022_manifest": len(m022), "023_manifest": len(m023), "union": len(spent)}
check("spent/forbidden union == 39,312 keys", len(spent) == 39312, sizes)
check("zero overlap between the 4,320 keys and the spent union", not (set(keys) & spent), len(set(keys) & spent))
spent_ids = {int(k.rsplit("|", 1)[1]) for k in spent}
spent_labels = {k.split("|")[1] for k in spent}
check("no spent key carries one of the 40 cue ids or words as its cue", not (set(ids40) & spent_ids) and not ({c["word"] for c in my_cues} & spent_labels))
check("every 024 key: an exposed frame × a frozen 024 cue", all(k.split("|")[0] in {f.frame_id for f in frames} and (k.split("|")[1], int(k.split("|")[2])) in
                                                                {(c["word"], c["token_id"]) for c in my_cues} for k in keys))
# the forbidden set exactly as Runner._base computes it (replicated) and the overlap check of Runner.freeze
forbidden_runner = (frozenset(inputs.closure["ledger"]) | frozenset(p.key for p in inputs.confirmation_020.all_prompts)
                    | frozenset(c022["manifest"]["S1-REF"]) | frozenset(c022["manifest"]["S1-VALIDITY"]) | frozenset(c022["manifest"]["S2-TARGET"]["Y1"])
                    | frozenset(c022["manifest"]["S2-TARGET"]["Y2"]) | confirmation_023_obj.manifest_keys())
check("Runner._base's forbidden set (replicated) == own spent union", set(forbidden_runner) == spent)
# broad scan: every prompt-key-like string in every tracked file of the repository
rguard.stage("tracked_scan")
tracked = subprocess.run(["git", "--no-optional-locks", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout.split(b"\0")
key_re = re.compile(rb'"([A-Za-z0-9_\-]+)\|([^|"\\\s]+)\|(\d+)"')
scan_ids, scan_labels, scanned = Counter(), Counter(), 0
for rel in tracked:
    if not rel:
        continue
    path = ROOT / rel.decode()
    if not path.is_file() or path.stat().st_size > 200_000_000:
        continue
    data = path.read_bytes()
    if b"|" not in data:
        continue
    scanned += 1
    for m in key_re.finditer(data):
        scan_ids[int(m.group(3))] += 1
        scan_labels[m.group(2).decode("utf-8", "replace")] += 1
hit_ids = {i: scan_ids[i] for i in ids40 if scan_ids[i]}
hit_labels = {c["word"]: scan_labels[c["word"]] for c in my_cues if scan_labels[c["word"]]}
check("no prompt key in any tracked file uses one of the 40 ids or words as its cue", not hit_ids and not hit_labels,
      f"{scanned} tracked files with '|' scanned, {sum(scan_ids.values())} key strings; hits ids {hit_ids} labels {hit_labels}")
RESULTS["manifest"] = {"sha256": manifest_sha, "keys": len(keys), "spent": sizes}

# ================================================================ Item 6: bound configuration and contents
rguard.stage("contents")
say("\n# Item 6: the bound configuration and contents")
cfg = c024["configuration"]
assignments = math.comb(16, 8)
max_upper = (25 * assignments) // 1000
check("C(16,8) = 12,870 and ⌊0.025·12,870⌋ = 321 (own arithmetic)", assignments == 12870 and max_upper == 321 and 321 / 12870 <= 0.025 < 322 / 12870)
expected_cfg = {"name": "production", "class_quota": 8, "n_fresh": 40, "draws": 10000, "null_permutations": 100000, "contrast_resamples": 10000,
                "cross_check_draws": 16, "calibration_counts": {"determiner-like": 45, "quantity": 45, "adjective": 49}, "n_calibration": 139, "n_pronoun": 36,
                "n_frames": 108, "n_nouns": 79,
                "expected_picks": {"N": ["honest", "polite", "rude", "sleepy", "wise", "lucky", "merry", "nervous"],
                                   "measure": ["gallon", "ounce", "acre", "herd", "crowd", "bundle", "cluster", "litre"],
                                   "ordinary": ["apple", "horse", "doctor", "king", "rabbit", "poet", "dragon", "lion"]}}
check("configuration literal values (own expectation)", {k: cfg[k] for k in expected_cfg} == expected_cfg, {k: cfg[k] for k in expected_cfg if cfg[k] != expected_cfg[k]})
g = cfg["guard"]
check("guard: n_E 8, n_N 8, assignments 12,870, max_upper 321; rule 'PASS iff K ≤ max_upper', ties count against, one-sided 0.025",
      g["n_E"] == 8 and g["n_N"] == 8 and g["assignments"] == 12870 and g["max_upper"] == 321 and "PASS iff K ≤ max_upper" in g["rule"]
      and "ties count against the guard" in g["rule"] and g["alpha"] == "0.025, one-sided" and "E first, then N" in g["order"], g)
check("configuration == rr.PRODUCTION.to_json()", cfg == rr.PRODUCTION.to_json())
check("class order N×8, B×8, D×8, C×8, E×8", [c["class"] for c in c024["cues"]] == [k for k in "NBDCE" for _ in range(8)])
check("rules == rr.FREEZE_RULES; classes == rr.CLASS_CONTENT; design/plan/model pinned",
      c024["rules"] == rr.FREEZE_RULES and c024["classes"] == rr.CLASS_CONTENT and c024["design"] == {"path": str(DESIGN.relative_to(ROOT)), "revision": 2, "commit": "9d03dee"}
      and c024["plan"]["revision"] == 1 and c024["plan"]["commit"] == "608088c"
      and c024["model"] == {"model_id": "EleutherAI/pythia-70m-deduped", "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"})
check("picks/expected_picks/picks_match_expected", c024["picks"] == expected_cfg["expected_picks"] == c024["expected_picks"] and c024["picks_match_expected"] is True)
check("top-level keys are exactly the planned schema", set(c024) == {"experiment", "schema_version", "kind", "design", "plan", "model", "configuration", "classes",
                                                                    "candidates", "rules", "exclusion", "target_noun_form_ids", "frame_token_ids",
                                                                    "reference_cue_ids", "exposed_frame_ids", "cues", "reserves", "rejected", "picks",
                                                                    "expected_picks", "picks_match_expected", "counts", "manifest", "content_sha256"}, sorted(c024))


def walk(value, path="$"):
    if isinstance(value, float):
        yield ("float", path, value)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield ("key", f"{path}.{k}", k)
            yield from walk(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk(v, f"{path}[{i}]")


items = list(walk(c024))
floats = [x for x in items if x[0] == "float"]
suspicious = [x[1] for x in items if x[0] == "key" and x[1] != "$.rules.score" and re.search(r"noun(ness)?_?score|nounness|mse|embedding|logit|hidden|delta|centroid|cosine|score",
                                                                    str(x[2]), re.IGNORECASE)]
check("the file holds no floating-point value at all", not floats, floats[:5])
say(f"  (the only score-like key is the rule text $.rules.score: {c024['rules']['score']!r})")
check("no key names a score, embedding, logit, hidden state, MSE or centroid (nounness absent by design)", not suspicious, suspicious)

# ================================================================ Item 7: byte-identical in-memory reconstruction through the production function
say("\n# Item 7: in-memory reconstruction through rr.freeze_payload (nothing written)")
before = CONF.stat()
sha023_now = rc.file_sha256(C023)
rguard.stage("freeze_payload")
payload = rr.freeze_payload(tok, pool=pool, exclusion_base=base023_now, confirmation_023=c023, confirmation_023_file_sha256=sha023_now,
                            config=rr.PRODUCTION)
rguard.stage("after_payload")
rebuilt = (pm.canonical_json(payload) + "\n").encode("utf-8")
check("rebuilt bytes == file bytes", rebuilt == raw_024, f"{len(rebuilt)} vs {len(raw_024)}")
check("rebuilt file sha256 and content sha256 == expected", sha_hex(rebuilt) == EXPECTED_FILE_SHA and payload["content_sha256"] == EXPECTED_CONTENT_SHA
      and rc.content_digest(payload) == EXPECTED_CONTENT_SHA)
confirmation_rt = rr.confirmation_from_payload(payload, pool, rr.PRODUCTION)
check("Runner.freeze's overlap check (replicated): rebuilt manifest ∩ forbidden = ∅", not (confirmation_rt.manifest_keys() & forbidden_runner))
rguard.stage("loader")
excluded_now = rr.exclusion(b0c.exclusion(inputs, c022, rc.file_sha256(C022)), c023, rc.file_sha256(C023))
loaded = rr.load_confirmation_024(CONF, pool, excluded_now, rr.PRODUCTION)
check("rr.load_confirmation_024(file, pool, exclusion recomputed now, rr.PRODUCTION) verifies", loaded.content_sha256 == EXPECTED_CONTENT_SHA
      and len(loaded.tokens) == 40 and len(loaded.manifest_keys()) == 4320 and loaded.frames == ())
check("loaded tokens == own picks (word, id, class, lemma, form)", [(t["word"], t["token_id"], t["class"], t["lemma"], t["form"]) for t in loaded.tokens]
      == [(c["word"], c["token_id"], c["class"], c["lemma"], c["form"]) for c in my_cues])
after = CONF.stat()
check("the file was not touched by the review (size, mtime_ns, inode unchanged)", (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino))

# ================================================================ Item 5: what the replicated freeze path opened
say("\n# Item 5: files opened per stage of the replicated freeze path (reads; writes refused)")
weight_like = re.compile(r"(\.safetensors|\.bin|\.pt|\.pth|\.ckpt|\.npy|\.npz|\.f64|calibration-table|/outputs/)")
for s in ("frozen_inputs", "verify_022_023", "exclusions", "tokenizer_load", "selection", "eligibility", "manifest", "freeze_payload", "loader"):
    paths = rguard.reads_in(s)
    shown = [p.replace(str(ROOT) + "/", "") for p in paths if not p.endswith(".py") and "/site-packages/" not in p and "/lib/python" not in p]
    say(f"  {s}: {len(paths)} files; non-code: {shown}")
    flagged = [p for p in paths if weight_like.search(p)]
    if flagged:
        say(f"    weight/array/outputs-like reads in {s}: {flagged}")
RESULTS["reads"] = {s: rguard.reads_in(s) for s in ("frozen_inputs", "verify_022_023", "exclusions", "tokenizer_load", "selection", "freeze_payload", "loader")}
hf_reads = sorted({p for _, p in rguard.STATE["reads"] if "huggingface" in p})
say(f"  Hugging Face cache files opened: {hf_reads}")
check("no model-weight file opened anywhere in this run (no .safetensors/.bin/.pt/.pth/.ckpt)",
      not any(re.search(r"\.(safetensors|bin|pt|pth|ckpt)$", p) for _, p in rguard.STATE["reads"]))
check("no model path was reached (Module.__call__, load_model, capture entry points, torch.load, safetensors)", sum(COUNTS.values()) == 0, dict(COUNTS))
check("freeze_payload/select_cues/_status stage opened no file at all", not rguard.reads_in("freeze_payload") and not rguard.reads_in("selection"),
      rguard.reads_in("freeze_payload") + rguard.reads_in("selection"))
check("no write under the repository was attempted after the self-test", len(rguard.STATE["refused"]) == 1, rguard.STATE["refused"])
# tokenizer files: git blob id of each cached non-LFS file equals its blob name (the pinned snapshot's content)
snap = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots" / PYTHIA_70M.revision
tok_files = {}
for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
    target = os.path.realpath(snap / name)
    data = Path(target).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    tok_files[name] = {"blob_name": Path(target).name, "git_blob_sha1": blob, "sha256": sha_hex(data), "ok": blob == Path(target).name}
check("pinned snapshot's tokenizer/config files are intact (git blob id == cache blob name)", all(v["ok"] for v in tok_files.values()),
      {k: v["sha256"][:16] for k, v in tok_files.items()})
RESULTS["tokenizer_files"] = tok_files

RESULTS["fails"] = FAILS
RESULTS["counts_refused_model_paths"] = dict(COUNTS)
(HERE / "review_freeze.json").write_text(json.dumps(RESULTS, indent=1, default=str) + "\n")
(HERE / "review_freeze.log").write_text("\n".join(LOG) + "\n")
say(f"\nFAILS: {len(FAILS)}")
for f in FAILS:
    say(f"  {f}")
sys.exit(1 if FAILS else 0)
