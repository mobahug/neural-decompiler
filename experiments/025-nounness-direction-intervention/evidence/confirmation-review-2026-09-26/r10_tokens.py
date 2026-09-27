"""Item 11: every cue's token id is ' ' + word under the pinned tokenizer (single token); every noun's plural id too;
decoding each id gives the word back. Tokenizer only (no model). Writes r10_tokens.json here."""
import guard  # noqa: F401
import json
from pathlib import Path
from transformers import AutoTokenizer
from neural_decompiler import models
ROOT = Path(guard.ROOT); HERE = Path(guard.HERE)
tok = AutoTokenizer.from_pretrained(models.PYTHIA_70M.model_id, revision=models.PYTHIA_70M.revision)
freeze = json.loads((ROOT / "experiments/025-nounness-direction-intervention/confirmation-v1.json").read_text())
bad = []
for c in freeze["cues"]:
    ids = tok.encode(" " + c["word"], add_special_tokens=False)
    if ids != [int(c["token_id"])] or tok.decode([int(c["token_id"])]) != " " + c["word"]:
        bad.append((c["word"], ids, c["token_id"]))
    if c["stratum"] == "noun":
        pids = tok.encode(" " + c["plural"], add_special_tokens=False)
        if pids != [int(c["plural_token_id"])] or c["plural"] != c["word"] + ("es" if c["word"].endswith(("s", "x", "sh", "ch")) else "s"):
            bad.append((c["plural"], pids, c["plural_token_id"]))
out = {"cues": len(freeze["cues"]), "mismatches": bad, "strata": {s: [c["word"] for c in freeze["cues"] if c["stratum"] == s] for s in ("adjective", "noun")},
       "guard": guard.EVENTS}
(HERE / "r10_tokens.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "strata"}))
