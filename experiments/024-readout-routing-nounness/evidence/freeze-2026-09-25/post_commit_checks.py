"""Post-commit checks of the installed Experiment 024 freeze (read-only; tokenizer and committed files only).

Guards: any write under the repository refused (audit hook); `load_model`, `torch.nn.Module.__call__` and every capture
entry point refused. Checks: the tracked file's blob and hashes; the stock `validate` phase on the tracked freeze; the
freeze rebuilt mechanically in memory with the real tokenizer, byte-identical to the committed file; the manifest's
4,320 unique keys with zero overlap with every spent set.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
REFUSED = []


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if not writing:
        return
    try:
        path = Path(os.fsdecode(args[0])).resolve()
    except OSError:
        return
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
        REFUSED.append(str(path))
        raise PermissionError(f"the post-commit checks may not write into the repository: {path}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402


def refuse(*args, **kwargs):
    raise RuntimeError("a model load or forward was attempted during the post-commit checks")


torch.nn.Module.__call__ = refuse
models.load_model = refuse
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, refuse)

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)

path = ROOT / rr.CONFIRMATION_RELATIVE_PATH
blob = subprocess.run(["git", "ls-files", "-s", str(path)], cwd=ROOT, check=True, capture_output=True, text=True).stdout.split()[1]
committed = subprocess.run(["git", "show", f"HEAD:{rr.CONFIRMATION_RELATIVE_PATH}"], cwd=ROOT, check=True, capture_output=True).stdout
payload = json.loads(path.read_text(encoding="utf-8"))
print(f"tracked blob {blob}; HEAD bytes == working bytes: {committed == path.read_bytes()}; file sha256 {rc.file_sha256(path)}; content sha256 "
      f"{payload['content_sha256']} (recomputed {rc.content_digest(payload)})")

logs: list[str] = []
runner = runner_module.Runner(log=logs.append)
status = runner.validate()
print(f"stock validate: exit {status}; {logs[-1]}")

base = runner._base()
tokenizer = runner_module._load_tokenizer(models.PYTHIA_70M)
rebuilt = rr.freeze_payload(tokenizer, pool=base.inputs.pool, exclusion_base=base.exclusion_base, confirmation_023=base.confirmation_023,
                            confirmation_023_file_sha256=rc.file_sha256(ROOT / b0c.CONFIRMATION_RELATIVE_PATH), config=rr.PRODUCTION)
rebuilt_bytes = (pm.canonical_json(rebuilt) + "\n").encode("utf-8")
picks = {cls: [f"{t['word']} {t['token_id']}" for t in rebuilt["cues"] if t["class"] == cls] for cls in rr.CLASSES}
print(f"mechanical rebuild in memory byte-identical to the committed file: {rebuilt_bytes == committed}")
for cls in rr.CLASSES:
    print(f"  {cls}: {', '.join(picks[cls])}")
confirmation, confirmation_sha = runner._confirmation(base)  # the stock loader, on the tracked file
manifest = payload["manifest"]["S2-TARGET"]
print(f"manifest: {len(manifest)} keys, {len(set(manifest))} unique, sha256 {pm.sha256_text(pm.canonical_json(payload['manifest']))}, overlap with the "
      f"{len(base.forbidden)} spent keys {len(set(manifest) & base.forbidden)}; every key a frozen 024 cue: "
      f"{all(int(key.split('|')[2]) in {int(t['token_id']) for t in confirmation.tokens} for key in manifest)}")
print(f"outputs/experiment-024 exists: {(ROOT / 'outputs/experiment-024').exists()}; repository writes refused: {REFUSED}")
