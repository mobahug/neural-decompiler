"""Extra: do the 40 fresh ids or words appear as a cue in any local (gitignored) output of Experiments 001–021?
022's and 023's local outputs are deliberately NOT opened (024's protocol never opens them); their executed sets are
their committed manifests, checked elsewhere."""
import rguard  # noqa: F401

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(rguard.ROOT)
c024 = json.loads((ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json").read_bytes())
ids = {int(c["token_id"]): c["word"] for c in c024["cues"]}
words = set(ids.values())
key_re = re.compile(rb'\|([A-Za-z][A-Za-z\-]*)\|(\d+)"')
field_re = re.compile(rb'"(?:token_id|cue_token_id|cue_id|cue_ids|cue)"\s*:\s*(\d+)')
hits, scanned, keys_seen = Counter(), 0, 0
for directory in sorted((ROOT / "outputs").glob("experiment-*")):
    number = int(directory.name.split("-")[1])
    if number >= 22:
        continue
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.stat().st_size > 2_000_000_000:
            continue
        data = path.read_bytes()
        scanned += 1
        for m in key_re.finditer(data):
            keys_seen += 1
            label, token = m.group(1).decode(), int(m.group(2))
            if token in ids or label in words:
                hits[(path.relative_to(ROOT).as_posix(), label, token)] += 1
        for m in field_re.finditer(data):
            if int(m.group(1)) in ids:
                hits[(path.relative_to(ROOT).as_posix(), "field", int(m.group(1)))] += 1
print(f"scanned {scanned} files under outputs/experiment-001..021; key strings seen {keys_seen}; hits {len(hits)}")
for key, count in sorted(hits.items())[:40]:
    print("  ", key, count)
print("reads outside outputs/experiment-0(01..21):", sorted({p for _, p in rguard.STATE["reads"] if "/outputs/" in p and not re.search(r"/outputs/experiment-0(0\d|1\d|2[01])/", p)}))
print("refused:", rguard.STATE["refused"])
