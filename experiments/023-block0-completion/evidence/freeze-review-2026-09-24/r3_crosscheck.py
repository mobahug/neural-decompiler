"""Cross-checks against the frozen code (after the independent reconstruction), plus item-6 run-evidence
reproduction: which files a tokenizer-only load and the runner's `_base()` inputs open (audit hook), with the model
loader and nn.Module.__call__ refusing. Read-only: nothing under the repository is written; no model is loaded."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
OUT = Path(__file__).resolve().parent
HF = (Path.home() / ".cache/huggingface").resolve()
BLOBS = HF / "hub/models--EleutherAI--pythia-70m-deduped/blobs"
fails = []
stage = {"name": "imports"}
opened = {}


def hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    raw = os.fsdecode(args[0])
    try:
        path = str(Path(raw).resolve())
    except OSError:
        path = raw
    if (path.startswith(str(ROOT)) and "/.venv/" not in path) or path.startswith(str(HF)):
        mode = args[1] if len(args) > 1 else None
        opened.setdefault(stage["name"], []).append((path, str(mode)))
        if path.startswith(str(ROOT)) and isinstance(mode, str) and any(c in mode for c in "wax+"):
            raise PermissionError(f"write refused: {path}")


sys.addaudithook(hook)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        fails.append(name)


import torch  # noqa: E402


def refuse_call(self, *a, **k):
    raise RuntimeError("nn.Module called")


torch.nn.Module.__call__ = refuse_call
from neural_decompiler import models  # noqa: E402


def refuse_load(*a, **k):
    raise RuntimeError("model load attempted")


models.load_model = refuse_load
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

rec = json.loads((OUT / "r2_reconstruction.json").read_text())
mine = rec["payload"]
# canonical JSON sorted the dict keys on disk; the list ORDER of strata/templates is the design table order, which
# r2 verified against the design text: determiner-like, quantity, adjective / cardinal, quantifier, coordinated-adjective
STRATA_ORDER = ["determiner-like", "quantity", "adjective"]
TEMPLATE_ORDER = ["cardinal", "quantifier", "coordinated-adjective"]
design_cues = {k: mine["candidates"]["cues"][k] for k in STRATA_ORDER}
design_frames = {k: mine["candidates"]["frames"][k] for k in TEMPLATE_ORDER}
check("reconstruction cue order / frame order follow the design tables", [c["class"] for c in mine["cues"]] == [s for s in STRATA_ORDER for _ in range(8)]
      and [f["template_id"] for f in mine["frames"]] == [t for t in TEMPLATE_ORDER for _ in range(6)])

print("== constants against the design text")
check("b0c.CUE_CANDIDATES == design lists (order included)", {k: list(v) for k, v in b0c.CUE_CANDIDATES.items()} == design_cues and list(b0c.CUE_CANDIDATES) == list(design_cues))
check("b0c.FRAME_CANDIDATES == design lists (order included)", {k: list(v) for k, v in b0c.FRAME_CANDIDATES.items()} == design_frames and list(b0c.FRAME_CANDIDATES) == list(design_frames))
check("ul.P_C_RANGE == design ranges", {k: list(v) for k, v in ul.P_C_RANGE.items()} == mine["rules"]["p_c_range"], str(ul.P_C_RANGE))
check("quotas 8/6, strata/templates order", (b0c.CUE_QUOTA, b0c.FRAME_QUOTA) == (8, 6) and list(b0c.STRATA) == list(design_cues) and list(b0c.TEMPLATES) == list(design_frames))
check("model binding == models.PYTHIA_70M", mine["model"] == {"model_id": models.PYTHIA_70M.model_id, "revision": models.PYTHIA_70M.revision}, str(mine["model"]))
check("frozen blobs verify (the code the freeze ran is the pinned code)", b0c.assert_frozen_blobs() == b0c.FROZEN_BLOBS)
check("block0_completion.py blob unchanged since extract (16d310fc…)", b0c.own_blob() == "16d310fc7fab5599ec83b8bd8162fb9613f8dad2" ==
      subprocess.run(["git", "rev-parse", "2111271:src/neural_decompiler/block0_completion.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip())

print("== tokenizer-only load (audited)")
stage["name"] = "tokenizer"
atime_before = {p.name: p.stat().st_atime for p in BLOBS.iterdir()}
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(models.PYTHIA_70M.model_id, revision=models.PYTHIA_70M.revision, local_files_only=True)
atime_after = {p.name: p.stat().st_atime for p in BLOBS.iterdir()}
stage["name"] = "base"

print("== the runner's _base() inputs (audited; same loaders and tracked check as Runner._base)")


def tracked(path):
    return subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True).returncode == 0


inputs = ul.load_frozen_inputs(ROOT, tracked=tracked)
digests_022 = b0c.verify_022_inputs(ROOT)
c22 = json.loads((ROOT / ul.CONFIRMATION_RELATIVE_PATH).read_text())
stage["name"] = "after"
pool = inputs.pool
check("pool tokens == 020's raw exposed_token_ids (279)", sorted(int(t) for _, t in pool.tokens) == sorted(json.loads((ROOT / "experiments/020-readout-decompilation/confirmation-v1.json").read_text())["exposed_token_ids"]))
check("pool frame order == file/reconstruction exposed_frame_ids", [f.frame_id for f in pool.frames] == mine["exposed_frame_ids"])
excl_texts = set(mine["exclusion"]["frame_texts"])
check("pool frame texts all inside the exclusion; pool reference ids / plural cue ids", {f.text_template for f in pool.frames} <= excl_texts
      and {k: int(v) for k, v in pool.reference_ids.items()} == mine["reference_cue_ids"]
      and {int(pool.token_id(n)) for n in pool.plural_cue.values()} == {767, 2067}, str(dict(pool.plural_cue)))
tmpl = {f.template_id: dict(f.cue_ids) for f in pool.frames if pool.frame_origin[f.frame_id] == "manifest"}
check("manifest-origin template cue ids == the ones the reconstruction used", all(tmpl[f["template_id"]] == f["cue_ids"] for f in mine["frames"]), str(tmpl))
units = b0c.exposed_units(inputs)
check("022's calibration units: 175 cues / 108 frames, disjoint from the 24 / 18", len(units.cues) == 175 and len(units.frames) == 108
      and not ({int(c[1]) for c in units.cues} & {c["token_id"] for c in mine["cues"]}) and not ({f.text_template for f in units.frames} & {f["text_template"] for f in mine["frames"]}))
spent_ids = {int(p.cue_token_id) for p in inputs.confirmation_020.all_prompts}
spent_texts = {p.frame.text_template for p in inputs.confirmation_020.all_prompts}
check("020 confirmation set via the frozen loader: disjoint", not (spent_ids & {c["token_id"] for c in mine["cues"]}) and not (spent_texts & {f["text_template"] for f in mine["frames"]}),
      f"{len(spent_ids)} ids, {len(spent_texts)} texts")

print("== pm._build_new_frame against the reconstruction (every candidate)")
for template, texts in design_frames.items():
    k = 0
    for rank, text in enumerate(texts):
        k += 1
        frame = pm._build_new_frame(tok, template, text, tmpl[template], f"{template}-023-{k}")
        lo, hi = ul.P_C_RANGE[template]
        ok_rule = lo <= frame.p_c <= hi and frame.p_t == frame.p_c + (1 if template == ul.COORDINATED else 0)
        row = next(r for r in rec["frame_eval"][template] if r["rank"] == rank)
        same = [frame.p_c, frame.p_t, list(frame.prefix_ids), list(frame.suffix_ids)] == [row["p_c"], row["p_t"], row["prefix_ids"], row["suffix_ids"]]
        if rank < 6:
            same = same and {**frame.to_dict(), "candidate_rank": rank} == mine["frames"][list(design_frames).index(template) * 6 + rank]
        if not (same and ok_rule):
            check(f"{template} rank {rank}", False, text)
check("pm._build_new_frame == own rule for all 39 candidates; the 18 picked frame dicts identical", not any(f.startswith(("cardinal", "quantifier", "coordinated")) for f in fails))

print("== the runner's forbidden set and the frozen manifest code")
forbidden = frozenset(inputs.closure["ledger"]) | frozenset(p.key for p in inputs.confirmation_020.all_prompts) | frozenset(
    c22["manifest"]["S1-REF"] + c22["manifest"]["S1-VALIDITY"] + c22["manifest"]["S2-TARGET"]["Y1"] + c22["manifest"]["S2-TARGET"]["Y2"])
conf = b0c.confirmation_from_payload(mine, pool, verify=True)
check("frozen Confirmation023.manifest() of the reconstruction == reconstruction manifest", conf.manifest() == mine["manifest"])
check("Runner._base forbidden set: 36252 keys, 0 overlap with the 3060", len(forbidden) == 36252 and not (conf.manifest_keys() & forbidden), str(len(forbidden)))
check("020 raw confirmation manifest == frozen loader's all_prompts keys", set(json.loads((ROOT / "experiments/020-readout-decompilation/confirmation-v1.json").read_text())["manifest"]["S1-REF"]
      + json.loads((ROOT / "experiments/020-readout-decompilation/confirmation-v1.json").read_text())["manifest"]["S1-VALIDITY"]
      + json.loads((ROOT / "experiments/020-readout-decompilation/confirmation-v1.json").read_text())["manifest"]["S2-TARGET"]) == {p.key for p in inputs.confirmation_020.all_prompts})

print("== descriptive: noun-form and frame-token coincidences")
new_ids = {c["token_id"]: c["word"] for c in mine["cues"]}
noun_hits = sorted((n.lexical_key, n.sg_ids, n.pl_ids) for n in pool.nouns if set(n.sg_ids + n.pl_ids) & set(new_ids))
fresh_noun_hits = sorted((n.lexical_key, n.sg_ids, n.pl_ids, [new_ids[t] for t in set(n.sg_ids + n.pl_ids) & set(new_ids)]) for n in inputs.confirmation_020.nouns if set(n.sg_ids + n.pl_ids) & set(new_ids))
print("      exposed nouns whose sg/pl token is a new cue id:", noun_hits, "| 020 fresh nouns:", fresh_noun_hits)
suffix_hits = [(f.frame_id, f.text_template, list(f.prefix_ids), list(f.suffix_ids)) for f in pool.frames if set(f.prefix_ids + f.suffix_ids) & set(new_ids)]
print("      exposed frames containing a new cue id in prefix/suffix:", suffix_hits)

print("== audit of opens")
for name in ("tokenizer", "base"):
    paths = sorted({p for p, _ in opened.get(name, [])})
    print(f"   [{name}] {len(paths)} files")
    for p in paths:
        print("      ", p.replace(str(HF), "~HF").replace(str(ROOT), "REPO"))
snap = HF / f"hub/models--EleutherAI--pythia-70m-deduped/snapshots/{models.PYTHIA_70M.revision}"
names = {os.path.basename(os.path.realpath(p)): p.name for p in snap.iterdir()}
freeze_log = json.loads(Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/freeze023/freeze_files.json").read_text())
freeze_hf = sorted(p for p in freeze_log["opened"] if p.startswith(str(HF)))
mine_hf = sorted({p for p, _ in opened.get("tokenizer", []) if p.startswith(str(HF))})
print("      freeze-run HF opens mapped:", [(os.path.basename(p), names.get(os.path.basename(p), "(not a snapshot file)")) for p in freeze_hf])
check("the freeze's HF opens == a tokenizer-only load's HF opens", freeze_hf == mine_hf, f"freeze {len(freeze_hf)} / tokenizer-only {len(mine_hf)}")
check("model.safetensors blob 3da38833… absent from the freeze log", not any("3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd" in p for p in freeze_log["opened"]))
repo_freeze = {p for p in freeze_log["opened"] if p.startswith(str(ROOT)) and "/__pycache__/" not in p and not p.endswith(".py") and ".egg-info" not in p}
repo_base = {p for p, _ in opened.get("base", []) if "/__pycache__/" not in p and not p.endswith(".py") and ".egg-info" not in p}
print("      freeze repo data files not opened by _base():", sorted(repo_freeze - repo_base))
print("      _base() repo data files not in the freeze log:", sorted(repo_base - repo_freeze))
check("every repo data file the freeze opened is explained by _base() or the confirmation write/re-read", repo_freeze - repo_base <= {str(ROOT / b0c.CONFIRMATION_RELATIVE_PATH)})
print("      blob atime changed by the tokenizer-only load:", {k: (atime_before[k] != atime_after[k]) for k in atime_before})
print("      modules imported: transformer_lens" if any(m.startswith("transformer_lens") for m in sys.modules) else "      transformer_lens not imported",
      "| safetensors imported" if "safetensors" in sys.modules else "| safetensors not imported")
print("CROSS-CHECKS:", "ALL PASS" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
