"""Which of the launcher's imports raises filelock's probe and torch's inductor mkdir (both in TMPDIR): the launcher
imports numpy, torch, transformers and huggingface_hub right after installing its hook, so these events are judged by
the real hook in every mode, the rehearsal included. Prints the write-type events per import step."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import install_guard  # noqa: E402

recorded = []
install_guard(recorded)
steps = {}
for name in ("numpy", "torch", "transformers", "huggingface_hub", "neural_decompiler.read_intervention", "transformer_lens.model_bridge"):
    start = len(recorded)
    __import__(name)
    steps[name] = [(r["event"], [str(a)[-60:] if isinstance(a, str) else a for a in r["args"][:3]]) for r in recorded[start:] if r["writing"] or r["event"] in ("tempfile.mkdtemp", "shutil.rmtree", "os.link")]
for name, rows in steps.items():
    print(name, "->", rows if rows else "no write-type event")
