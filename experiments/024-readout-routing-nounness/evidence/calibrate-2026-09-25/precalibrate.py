"""Pre-calibration checks for Experiment 024 (read-only; tokenizer, committed files and weights only; no prompt).

Part 1: the repository state; the stock `validate`; the committed freeze's hashes; its mechanical rebuild; the manifest.
Part 2: every expected weight-derived value reproduced with the canonical module (weights only; module calls during the
load counted, every forward refused afterwards), compared with the full digests recorded after the freeze. Any
mismatch exits non-zero: calibration must not run.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
REFUSED = []
EXPECTED = {"parameters_sha256": "fd953f1c745299bedfd1f242fc15c7fe31211f79d8bc0d3ca46372962e7fc0e5",
            "embedding_sha256": "9cd6f39b3adfdb74cffe046af459634982684bff9aef2a981760326f1748eaaf",
            "noun_row_ids_sha256": "08fb9b4faf7e8a807c57fc98bd45887f237f90a4675922ec4c98ce9c79d79871",
            "calibration_cue_ids_sha256": "106cfeccd7b9a10362fb7aef1e3558506e2a7299e4bcfa4ec42aa0f46f774dbf",
            "mu_noun_sha256": "87930f0150e7263963d593303defd63f87f601f6892717c61d2f20c704f151d2",
            "mu_cue_sha256": "f8cca95464092089dae6cd668d3e29bd26a89131b0f5994c35f6503a282121c4",
            "scores_sha256": "a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995", "calibration_maximum": 0.13502027836111233,
            "E_above": ["apple", "horse", "doctor", "poet", "dragon"]}


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
        raise PermissionError(f"the pre-calibration checks may not write into the repository: {path}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


git("fetch", "-q", "origin")
head, origin = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
check("HEAD == origin/main == bf0049c", head == origin and head.startswith("bf0049c"), head)
check("working tree clean", git("status", "--porcelain") == "")
check("no outputs/experiment-024", not (ROOT / "outputs/experiment-024").exists())

spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)


def refuse(*args, **kwargs):
    raise RuntimeError("no model or tokenizer in the validate check")


logs: list[str] = []
runner = runner_module.Runner(log=logs.append, model_loader=refuse, tokenizer_loader=refuse)
check("configuration is the frozen production one", runner.config is rr.PRODUCTION)
check("stock validate passes", runner.validate() == 0, logs[-1] if logs else "")
check("pins", rr.assert_frozen_blobs() == rr.FROZEN_BLOBS)
check("023 committed artifacts (exposed cells, confirmation, lock)", len(rr.verify_023_inputs(ROOT)) == 7)
check("022 committed inputs", len(b0c.verify_022_inputs(ROOT)) == 5)
path = ROOT / rr.CONFIRMATION_RELATIVE_PATH
payload = json.loads(path.read_text(encoding="utf-8"))
check("freeze file sha256", rc.file_sha256(path) == "68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60")
check("freeze content sha256", payload["content_sha256"] == rc.content_digest(payload) == "87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87")
base = runner._base()
confirmation, _ = runner._confirmation(base)
tokenizer = runner_module._load_tokenizer(models.PYTHIA_70M)
rebuilt = rr.freeze_payload(tokenizer, pool=base.inputs.pool, exclusion_base=base.exclusion_base, confirmation_023=base.confirmation_023,
                            confirmation_023_file_sha256=rc.file_sha256(ROOT / b0c.CONFIRMATION_RELATIVE_PATH), config=rr.PRODUCTION)
check("mechanical rebuild byte-identical (same 40 cues)", (pm.canonical_json(rebuilt) + "\n").encode("utf-8") == path.read_bytes())
manifest = payload["manifest"]["S2-TARGET"]
check("manifest 4,320 unique keys, digest fcc437fc…, zero spent overlap",
      len(set(manifest)) == len(manifest) == 4_320 and pm.sha256_text(pm.canonical_json(payload["manifest"])).startswith("fcc437fc") and not set(manifest) & base.forbidden,
      f"spent keys {len(base.forbidden)}")

calls = {"during_load": 0, "after_load": 0}
original_call = torch.nn.Module.__call__
phase = {"loading": True}


def guarded_call(self, *args, **kwargs):
    if phase["loading"]:
        calls["during_load"] += 1
        return original_call(self, *args, **kwargs)
    calls["after_load"] += 1
    raise RuntimeError("a forward pass was attempted in the pre-calibration check")


torch.nn.Module.__call__ = guarded_call
model = models.load_model(models.PYTHIA_70M)
phase["loading"] = False
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capture refused")))
parameters = rr.parameters_digest(model)
W_E = pm.Weights.from_model(model).W_E
del model
bindings = rr.score_bindings(W_E, base.inputs.pool, b0c.exposed_units(base.inputs))
scores = rr.full_scores(W_E, bindings, [token["token_id"] for token in confirmation.tokens])
maximum = max(rr.calibration_scores(W_E, bindings))
e = [(token["word"], score) for token, score in zip(confirmation.tokens, scores) if token["class"] == "E"]
n = [score for token, score in zip(confirmation.tokens, scores) if token["class"] == "N"]
now = {"parameters_sha256": parameters, "embedding_sha256": bindings["embedding_sha256"], "noun_row_ids_sha256": bindings["noun_row_ids_sha256"],
       "calibration_cue_ids_sha256": bindings["calibration_cue_ids_sha256"], "mu_noun_sha256": bindings["mu_noun_sha256"], "mu_cue_sha256": bindings["mu_cue_sha256"],
       "scores_sha256": rc.tensor_digest(torch.tensor(scores, dtype=torch.float64)), "calibration_maximum": maximum, "E_above": [w for w, s in e if s > maximum]}
for key, value in EXPECTED.items():
    check(f"expected {key}", now[key] == value, now[key])
check("158 noun-form ids and 139 calibration-cue ids", len(bindings["noun_row_ids"]) == 158 and len(bindings["calibration_cue_ids"]) == 139)
check("every E above every N", min(s for _, s in e) > max(n), f"{min(s for _, s in e):+.6f} > {max(n):+.6f}")
check("module calls: 0 during the load, 0 after", calls == {"during_load": 0, "after_load": 0}, calls)
check("no repository write", not REFUSED, REFUSED)
print("FAILURES:", failures if failures else "none", flush=True)
sys.exit(1 if failures else 0)
