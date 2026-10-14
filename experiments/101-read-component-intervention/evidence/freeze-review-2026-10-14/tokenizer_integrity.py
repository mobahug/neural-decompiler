"""Tokenizer integrity: re-tokenize earlier experiments' committed frames/prompts and compare with their committed ids."""
import json, subprocess
from pathlib import Path
import tokenizers
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
SNAP = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c/tokenizer.json"
tok = tokenizers.Tokenizer.from_file(str(SNAP))
enc = lambda t: tok.encode(t, add_special_tokens=False).ids
def cj(p): return json.loads(subprocess.run(["git","--no-optional-locks","show",f"HEAD:{p}"],cwd=ROOT,capture_output=True,check=True).stdout)
bad = []; n = 0
for p in ["experiments/020-readout-decompilation/confirmation-v1.json","experiments/022-upstream-error-localization/confirmation-v1.json","experiments/023-block0-completion/confirmation-v1.json"]:
    c = cj(p)
    for f in c["frames"]:
        for lab, cid in f["cue_ids"].items():
            word = tok.decode([cid]).strip()
            ids = enc(f["text_template"].replace("{cue}", word)); n += 1
            if ids != f["prefix_ids"] + [cid] + f["suffix_ids"]: bad.append([p, f["frame_id"], lab])
c20 = cj("experiments/020-readout-decompilation/confirmation-v1.json")
for pr in c20["exposed_frame_prompts"] + c20["token_prompts"]:
    n += 1
    if enc(pr["text"]) != pr["token_ids"]: bad.append(["020 prompt", pr["key"]])
print(json.dumps({"checked": n, "mismatches": len(bad), "first": bad[:5]}))
