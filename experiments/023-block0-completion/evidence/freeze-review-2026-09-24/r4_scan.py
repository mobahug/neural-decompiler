"""Beyond the frozen rule (falsification attempt): were any of the 24 new cue ids or 18 new frames ever used anywhere?
  (1) every local results ledger 005-022: prompt keys <frame>|<label>|<id> (raw byte regex, no JSON parse);
  (2) every committed JSON under experiments/ and screening/: "token_id": <id> fields, prompt keys, the frame sentences;
  (3) the screening manifest: any prompt token list containing a new id.
Read-only."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
rec = json.loads((Path(__file__).resolve().parent / "r2_reconstruction.json").read_text())["payload"]
ids = {c["token_id"]: c["word"] for c in rec["cues"]}
frames = rec["frames"]
sentences = {f["text_template"].split("{cue}")[0].strip(): f["frame_id"] for f in frames}  # e.g. "The dairy produces"
key_re = re.compile(rb'"([^"|\\]{1,80})\|([^"|\\]{1,80})\|(\d{1,6})"')

print("== (1) local results ledgers (prompt keys anywhere in the file)")
for n in range(5, 23):
    for path in sorted((ROOT / f"outputs/experiment-{n:03d}").glob("*.json")):
        data = path.read_bytes()
        keys = set(key_re.findall(data))
        hit_ids = sorted({(int(t), f.decode(), l.decode()) for f, l, t in keys if int(t) in ids})
        hit_frames = sorted({f.decode() for f, _, _ in keys if f.decode().endswith(tuple(f"-023-{k}" for k in range(1, 7)))})
        hit_text = sorted(s for s in sentences if s.encode() in data)
        print(f"   {path.relative_to(ROOT)}: {len(keys)} distinct keys; new-id keys {hit_ids[:5]}{'…' if len(hit_ids) > 5 else ''} ({len(hit_ids)}); "
              f"new frame ids {hit_frames}; new sentences {hit_text}")

print("== (2) committed JSON files (excluding 023's own confirmation)")
tracked = subprocess.run(["git", "ls-files", "experiments", "screening"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
for rel in tracked:
    if not rel.endswith(".json") or rel == "experiments/023-block0-completion/confirmation-v1.json":
        continue
    data = (ROOT / rel).read_bytes()
    tid_hits = sorted({int(m) for m in re.findall(rb'"token_id":\s*(\d+)', data) if int(m) in ids})
    key_hits = sorted({(f.decode(), l.decode(), int(t)) for f, l, t in key_re.findall(data) if int(t) in ids})
    txt_hits = sorted(s for s in sentences if s.encode() in data)
    word_keys = sorted({w for w in ids.values() if f'|{w}|'.encode() in data})
    if tid_hits or key_hits or txt_hits or word_keys:
        print(f"   {rel}: token_id fields {tid_hits}; keys {key_hits[:4]}; sentences {txt_hits}; '|word|' labels {word_keys}")

print("== (3) screening manifest prompts containing a new id at any position")
manifest = json.loads((ROOT / "screening/behavior-candidates/manifest-v1.json").read_text())
hits = {}
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        if o and all(isinstance(x, int) and not isinstance(x, bool) for x in o) and path.endswith("_ids"):
            for x in set(o) & set(ids):
                hits.setdefault(ids[x], set()).add(path.split(".")[-1])
        else:
            for v in o:
                walk(v, path)
walk(manifest)
print("   new ids found in manifest id lists:", {w: sorted(p) for w, p in hits.items()})
cands = sorted({c.get("candidate_id") for c in manifest.get("cases", [])}) if isinstance(manifest, dict) else []
print("   manifest candidate ids:", cands)
