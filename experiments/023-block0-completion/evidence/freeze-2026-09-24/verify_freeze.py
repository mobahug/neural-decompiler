"""Experiment 023 post-freeze verification: strictly read-only (tokenizer only; no model)."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HEAD = "89536b004b2e4536e83185cdfec31df4e7206982"
spec = importlib.util.spec_from_file_location("run023", ROOT / "experiments/023-block0-completion/run.py")
run023 = importlib.util.module_from_spec(spec)
sys.modules["run023"] = run023
spec.loader.exec_module(run023)
from transformers import AutoTokenizer  # noqa: E402

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler.models import PYTHIA_70M  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def canonical(obj):  # independent re-implementation, compared with pm.canonical_json below
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


path = ROOT / b0c.CONFIRMATION_RELATIVE_PATH
raw = path.read_bytes()
payload = json.loads(raw)
file_sha = hashlib.sha256(raw).hexdigest()
print("file", path.relative_to(ROOT), len(raw), "bytes, sha256", file_sha)
check("file is canonical JSON + newline", raw.decode("utf-8") == pm.canonical_json(payload) + "\n")
body = {k: v for k, v in payload.items() if k != "content_sha256"}
check("content_sha256 recomputes (pm and independent canonical JSON)", payload["content_sha256"] == pm.sha256_text(pm.canonical_json(body))
      == hashlib.sha256(canonical(body).encode("utf-8")).hexdigest(), payload["content_sha256"])

runner = run023.Runner()
inputs, confirmation_022, sha_022, digests, forbidden = runner._base()
confirmation = b0c.load_confirmation_023(path, inputs, confirmation_022, sha_022)  # digest, manifest, composition, exclusion re-extracted, no overlap
check("load_confirmation_023 verifies (digest, manifest, 8/8/8 + 6/6/6, exclusion re-extracted, no reuse)", True)
check("schema, design, plan, scope", (payload["experiment"], payload["schema_version"], payload["design"], payload["plan"], payload["scope"])
      == ("023", 1, b0c.DESIGN, b0c.PLAN, b0c.SCOPE), payload["scope"][:80] + "…")
check("model binding", payload["model"] == {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, json.dumps(payload["model"]))

# The selection: the first eligible entries of the frozen ordered lists
print("\ncues:")
for stratum in b0c.STRATA:
    picked = [entry for entry in payload["cues"] if entry["class"] == stratum]
    print(f"  {stratum:16s}", ", ".join(f"{e['word']} ({e['token_id']}, rank {e['candidate_rank']})" for e in picked))
    check(f"{stratum}: 8 = the first 8 of the frozen list", [e["word"] for e in picked] == list(b0c.CUE_CANDIDATES[stratum][:8])
          and [e["candidate_rank"] for e in picked] == list(range(8)))
print("frames:")
for template in b0c.TEMPLATES:
    picked = [entry for entry in payload["frames"] if entry["template_id"] == template]
    for e in picked:
        print(f"  {e['frame_id']:28s} rank {e['candidate_rank']:2d}  p_c {e['p_c']} p_t {e['p_t']}  {e['text_template']!r}")
    check(f"{template}: 6 = the first 6 of the frozen list, ids {template}-023-1…6", [e["text_template"] for e in picked] == list(b0c.FRAME_CANDIDATES[template][:6])
          and [e["frame_id"] for e in picked] == [f"{template}-023-{k}" for k in range(1, 7)] and [e["candidate_rank"] for e in picked] == list(range(6)))
check("rejections: none (every first candidate eligible)", payload["rejected"] == [], str(len(payload["rejected"])))
check("counts", payload["counts"] == {"classes": {s: 8 for s in b0c.STRATA}, "templates": {t: 6 for t in b0c.TEMPLATES}}, json.dumps(payload["counts"]))
check("no possessive-or-pronoun cue", all(e["class"] in b0c.STRATA for e in payload["cues"]) and "possessive-or-pronoun" not in payload["counts"]["classes"])

# Freshness: the exclusion sets and explicit overlaps with 020/021/022 units
excluded = payload["exclusion"]
print(f"\nexclusion: {len(excluded['cue_token_ids'])} earlier cue token ids, {len(excluded['frame_texts'])} earlier frame texts, sources:",
      [s["source"] for s in excluded["sources"]])
ids = {e["token_id"] for e in payload["cues"]}
texts = {e["text_template"] for e in payload["frames"]}
check("24 distinct cue ids, 18 distinct frame texts", len(ids) == 24 and len(texts) == 18)
check("0 overlap with the excluded cue ids / frame texts", not ids & set(excluded["cue_token_ids"]) and not texts & set(excluded["frame_texts"]))
pool = inputs.pool
check("0 overlap with 020's exposed pool (175 cues / 108 frames; 022's calibration units)", not ids & {int(t) for _, t, _ in b0c.exposed_units(inputs).cues}
      and not texts & {f.text_template for f in pool.frames})
c020 = inputs.confirmation_020
c020_ids = {int(p.cue_token_id) for p in c020.all_prompts}
c020_texts = {p.frame.text_template for p in c020.all_prompts}
check("0 overlap with 020's confirmation set (021's spent set)", not ids & c020_ids and not texts & c020_texts, f"{len(c020_ids)} ids, {len(c020_texts)} texts")
c022_ids = {int(c["token_id"]) for c in confirmation_022["cues"]} | {int(v) for f in confirmation_022["frames"] for v in f["cue_ids"].values()}
c022_texts = {f["text_template"] for f in confirmation_022["frames"]}
check("0 overlap with 022's frozen units", not ids & c022_ids and not texts & c022_texts, f"{len(c022_ids)} ids, {len(c022_texts)} texts")
check("the exclusion re-extracted now equals the recorded one", b0c.exclusion(inputs, confirmation_022, sha_022) == excluded)

# Tokenizer eligibility and structural validity (tokenizer only)
tokenizer = AutoTokenizer.from_pretrained(PYTHIA_70M.model_id, revision=PYTHIA_70M.revision, local_files_only=True)
for e in payload["cues"]:
    check(f"cue ' {e['word']}' is one token {e['token_id']}", tokenizer.encode(" " + e["word"], add_special_tokens=False) == [e["token_id"]])
cue_ids_by_template = {f.template_id: dict(f.cue_ids) for f in pool.frames if pool.frame_origin[f.frame_id] == "manifest"}
stable = 0
for e in payload["frames"]:
    rebuilt = pm._build_new_frame(tokenizer, e["template_id"], e["text_template"], cue_ids_by_template[e["template_id"]], e["frame_id"]).to_dict()
    low, high = ul.P_C_RANGE[e["template_id"]]
    coordinated = e["template_id"] == ul.COORDINATED
    ok = ({k: rebuilt[k] for k in rebuilt} == {k: e[k] for k in rebuilt} and low <= e["p_c"] <= high and e["p_c"] == len(e["prefix_ids"])
          and e["p_t"] == e["p_c"] + (1 if coordinated else 0) and (not coordinated or len(e["suffix_ids"]) == 1))
    check(f"frame {e['frame_id']} structurally valid (rebuilt = stored; p_c {e['p_c']} in {low}–{high}; p_t)", ok)
    for c in payload["cues"]:
        stable += tokenizer.encode(e["text_template"].replace("{cue}", c["word"]), add_special_tokens=False) == e["prefix_ids"] + [c["token_id"]] + e["suffix_ids"]
print(f"descriptive: {stable} of {24 * 18} cue × frame texts tokenize exactly as prefix + cue + suffix")

# The manifest
manifest = payload["manifest"]
sizes = {"S1-REF": len(manifest["S1-REF"]), "S1-VALIDITY": len(manifest["S1-VALIDITY"]), "Y1": len(manifest["S2-TARGET"]["Y1"]), "Y2": len(manifest["S2-TARGET"]["Y2"])}
print("\nmanifest sizes", sizes)
check("Y1 = 24 × 108 = 2,592 and Y2 = 24 × 18 = 432; stage 1 = 18 + 18", sizes == {"S1-REF": 18, "S1-VALIDITY": 18, "Y1": 2592, "Y2": 432})
all_keys = manifest["S1-REF"] + manifest["S1-VALIDITY"] + manifest["S2-TARGET"]["Y1"] + manifest["S2-TARGET"]["Y2"]
check("every key unique; total 3,060", len(all_keys) == len(set(all_keys)) == 3060 == len(confirmation.manifest_keys()), str(len(set(all_keys))))
check("each list in canonical (sorted) order", all(lst == sorted(lst) for lst in (manifest["S1-REF"], manifest["S1-VALIDITY"], manifest["S2-TARGET"]["Y1"], manifest["S2-TARGET"]["Y2"])))
check("manifest == the one the cues and frames define", manifest == confirmation.manifest())
state = rd.load_results_state(ROOT / "outputs/experiment-023/results.json")
check("0 keys in the 023 prompt ledger", not set(all_keys) & set(state["executed_prompt_keys"]), f"ledger {len(state['executed_prompt_keys'])}")
check("0 collisions with forbidden keys (020 ledger ∪ 020 confirmation set ∪ 022 manifest)", not set(all_keys) & forbidden, f"{len(forbidden)} forbidden keys")
check("every key carries a new cue token or a new frame", all(p.cue_token_id in ids or p.frame.text_template in texts for p in confirmation.all_prompts if p.cue_label not in ("ref", "pl"))
      and all(p.frame.text_template in texts for p in confirmation.stage1_prompts))

# Deterministic: the selection recomputed in memory (tokenizer only, writes nothing) equals the file byte for byte
again = b0c.freeze_payload(tokenizer, inputs, confirmation_022, sha_022)
check("in-memory recomputation == the frozen file, byte for byte", (pm.canonical_json(again) + "\n").encode("utf-8") == raw)

# Nothing else moved
check("results state unchanged (freeze keeps no state)", state["state_sha256"] == "fcb2c0305a02c0c42a9ce39bc5a9c19c2c4f30c298d3a198c2644a751abb9e4d"
      and rc.file_sha256(ROOT / "outputs/experiment-023/results.json") == "9bb1e33594670343a41e5c08f81cca3a78f50d1f1c4e04882708879e508a5ea7")
check("phases", all(state["phases"][p]["status"] == "not_started" for p in ("calibrate", "lock", "confirm", "report")) and state["phases"]["extract"]["status"] == "complete",
      json.dumps({p: e["status"] for p, e in state["phases"].items()}))
check("HEAD unchanged; only the freeze file is new (untracked)", git("rev-parse", "HEAD") == HEAD
      and git("status", "--porcelain", "--untracked-files=all") == "?? experiments/023-block0-completion/confirmation-v1.json")
check("installed exposed cells unchanged", rc.file_sha256(ROOT / b0c.CELLS_DATA_RELATIVE_PATH) == "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4")
print("\nVERIFY FREEZE:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
