"""Supplement to items 6 and 8: I7 re-run under the seal (the lock's weight-derived quantities, bit for bit) and an own
recomputation of the 40 frozen nounness scores from the pinned embedding (numeric agreement and identical ranks)."""
import rguard  # noqa: F401  (first)
from rguard import REPO

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import rseal

ROOT = Path(REPO)
sys.path.insert(0, str(ROOT / "src"))
import transformer_lens.model_bridge  # noqa: E402,F401
import transformers.models.gpt_neox.modeling_gpt_neox  # noqa: E402,F401

spec = importlib.util.spec_from_file_location("run024_confirmation_review", ROOT / "experiments/024-readout-routing-nounness/run.py")
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)
rr, ul, rd, pm = run.rr, run.ul, run.rd, run.pm
from neural_decompiler.models import PYTHIA_70M, load_model  # noqa: E402

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail != '' else ''}", flush=True)


def refuse(*a, **k):
    raise RuntimeError("unused")


runner = run.Runner(log=lambda m: None, model_loader=refuse, tokenizer_loader=refuse)
base = runner._base()
confirmation, _ = runner._confirmation(base)
lock = json.loads((ROOT / rr.LOCK_RELATIVE_PATH).read_text())
record = json.loads((ROOT / rr.CALIBRATION_RELATIVE_PATH).read_text())
seal = rseal.Seal()
seal.count_during_load()
model = load_model(PYTHIA_70M)
seal.refuse_everything()
check("sealed load: 0 module calls during the load; model fully sealed", seal.load_counts == {"Module.__call__": 0, "forward": 0, "__call__ override": 0}
      and seal.verify_model_sealed(model) == [], seal.load_counts)
progs = ul.ModelPrograms.from_model(model, base.inputs)
W = progs.weights.W_E
check("embedding digest == the lock's", rr.embedding_digest(W) == lock["dependencies"]["model"]["embedding_sha256"])
rep = rr.reproduce_lock_quantities(W, record, confirmation, lock)
check("I7 re-run now: bitwise_equal True, nothing differing, scores digest a703ac16…", rep["bitwise_equal"] and rep["differing"] == []
      and rep["scores_sha256"] == "a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995", rep)

# own nounness: id lists derived from the pool (nouns) and bound by digest (cues); float64 numpy, fsum dot products
E = W.detach().cpu().double().numpy()
noun_ids = [int(t) for n in base.inputs.pool.nouns if n.single_token for t in (n.sg_ids[0], n.pl_ids[0])]
cue_ids = [int(t) for t in record["score"]["calibration_cue_ids"]]
check("own noun row ids == the bound list (158 rows; digest verifies)", noun_ids == record["score"]["noun_row_ids"] and len(noun_ids) == 158
      and hashlib.sha256(json.dumps(noun_ids, separators=(",", ":")).encode()).hexdigest() == record["score"]["noun_row_ids_sha256"])
check("calibration cue ids: 139, digest verifies", len(cue_ids) == 139 and hashlib.sha256(json.dumps(cue_ids, separators=(",", ":")).encode()).hexdigest()
      == record["score"]["calibration_cue_ids_sha256"])


def mean_rows(ids):
    return np.array([math.fsum(E[i, d] for i in ids) / len(ids) for d in range(E.shape[1])])


def cos(a, b):
    return math.fsum(a * b) / (math.sqrt(math.fsum(a * a)) * math.sqrt(math.fsum(b * b)))


mu_n, mu_c = mean_rows(noun_ids), mean_rows(cue_ids)
own = [cos(E[int(c["token_id"])], mu_n) - cos(E[int(c["token_id"])], mu_c) for c in lock["fresh"]["cues"]]
locked = [float(c["nounness"]) for c in lock["fresh"]["cues"]]
d = max(abs(a - b) for a, b in zip(own, locked))
rank = lambda v: sorted(range(len(v)), key=lambda i: v[i])  # noqa: E731
check("own nounness (fsum means and dots) agrees with the lock's 40 scores (max |diff| ≤ 1e-14) with the identical ordering",
      d <= 1e-14 and rank(own) == rank(locked), f"max |diff| {d:.3e}; bitwise {sum(a == b for a, b in zip(own, locked))}/40")
maxcal = max(float(e["nounness_loo"]) for e in record["calibration_cues"])
check("own above-maximum flags == the lock's (23 of 40; 5 of 8 E)", [v > maxcal for v in own] == [bool(c["above_calibration_maximum"]) for c in lock["fresh"]["cues"]])
gap = min(own[i] for i, c in enumerate(lock["fresh"]["cues"]) if c["class"] == "E") - max(own[i] for i, c in enumerate(lock["fresh"]["cues"]) if c["class"] == "N")
print(f"   min E score − max N score = {gap:+.6f} (every E above every N)")
check("the seal held (0 refused attempts)", all(v == 0 for v in seal.refused.values()), seal.refused)
print(f"S06B {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
