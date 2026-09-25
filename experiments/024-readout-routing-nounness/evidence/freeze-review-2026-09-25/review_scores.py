"""Item 8: a read-only, weights-only diagnostic (NOT freeze contents): the values calibration and lock should reproduce.

torch.nn.Module.__call__ is counted during the load and refused afterwards; every plural_mechanism capture or
intervention entry point is refused throughout. The score is also computed by an own numpy float64 implementation.
"""
import rguard  # noqa: F401

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(rguard.ROOT)
HERE = Path(__file__).resolve().parent
CONF = ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json"
CELLS_INDEX = ROOT / "experiments/023-block0-completion/exposed-cells.json"
EXPECTED = {"parameters": "fd953f1c", "embedding": "9cd6f39b", "noun_ids": "08fb9b4f", "mu_noun": "87930f01", "cue_ids": "106cfecc", "mu_cue": "f8cca954",
            "scores": "a703ac16"}
FAILS, OUT = [], {}


def check(name, condition, detail=""):
    if not condition:
        FAILS.append(f"{name}: {detail}")
    print(f"  [{'ok' if condition else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


rguard.stage("imports")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402

COUNTS = Counter()


def refuse_capture(*args, **kwargs):
    COUNTS["pm_capture"] += 1
    raise RuntimeError("reviewer guard: a capture/intervention entry point was reached")


for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, refuse_capture)
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler.models import PYTHIA_70M  # noqa: E402

c024 = json.loads(CONF.read_bytes())
rguard.stage("frozen_inputs")
inputs = ul.load_frozen_inputs(ROOT)
pool = inputs.pool
index = json.loads(CELLS_INDEX.read_text(encoding="utf-8"))

# ---------------------------------------------------------------- the weights: Module.__call__ counted during the load, refused after
rguard.stage("load")
original_call = torch.nn.Module.__call__


def counting(self, *args, **kwargs):
    COUNTS["module_calls_during_load"] += 1
    return original_call(self, *args, **kwargs)


torch.nn.Module.__call__ = counting
model = models.load_model(PYTHIA_70M)


def refuse_call(self, *args, **kwargs):
    COUNTS["module_calls_after_load"] += 1
    raise RuntimeError("reviewer guard: a forward was attempted after the load")


torch.nn.Module.__call__ = refuse_call
rguard.stage("weights")
print(f"model loaded: revision {models.resolved_revision(model)}; module calls during the load: {COUNTS['module_calls_during_load']}")
check("zero module calls during the load", COUNTS["module_calls_during_load"] == 0, COUNTS["module_calls_during_load"])
parameters_sha = rr.parameters_digest(model)
W_E = pm.Weights.from_model(model).W_E
embedding_sha = rr.embedding_digest(W_E)
n_params = len(rr.named_tensors(model))
del model
check("parameters digest (rr.parameters_digest) starts fd953f1c", parameters_sha.startswith(EXPECTED["parameters"]), f"{parameters_sha} over {n_params} named tensors")
check("embedding digest (rr.embedding_digest of pm.Weights.from_model(model).W_E) starts 9cd6f39b", embedding_sha.startswith(EXPECTED["embedding"]),
      f"{embedding_sha}; W_E {tuple(W_E.shape)} {W_E.dtype}")
# own embedding digest recomputation (shape + '|<f4|' + little-endian float32 bytes)
own_emb = hashlib.sha256(json.dumps(list(W_E.shape)).encode("ascii") + b"|<f4|" + W_E.detach().cpu().contiguous().numpy().astype("<f4").tobytes()).hexdigest()
check("own embedding digest == rr.embedding_digest", own_emb == embedding_sha)
# the safetensors blob in the cache is the LFS object its name says
snap = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots" / PYTHIA_70M.revision / "model.safetensors"
blob = Path(os.path.realpath(snap))
h = hashlib.sha256()
with open(blob, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 22), b""):
        h.update(chunk)
check("model.safetensors content sha256 == its LFS blob name", h.hexdigest() == blob.name, blob.name[:16])

# ---------------------------------------------------------------- the id lists: own derivation, cross-checked with rr
nouns = [n for n in pool.nouns if len(n.sg_ids) == 1]
noun_ids = [int(i) for n in nouns for i in (n.sg_ids[0], n.pl_ids[0])]
check("79 scorable nouns (pool order) == 023 exposed-cells index nouns (order)", [n.lexical_key for n in nouns] == index["nouns"] and len(nouns) == 79)
check("158 noun-row ids == rr.noun_row_ids(pool)", noun_ids == rr.noun_row_ids(pool) and len(noun_ids) == 158 and len(set(noun_ids)) == 158)
noun_ids_sha = hashlib.sha256(json.dumps(noun_ids, separators=(",", ":")).encode()).hexdigest()
check("noun-row id list digest starts 08fb9b4f", noun_ids_sha.startswith(EXPECTED["noun_ids"]) and noun_ids_sha == pm.sha256_text(pm.canonical_json(noun_ids)), noun_ids_sha)
cal = [(w, int(t), cls) for w, t, cls in index["cues"] if cls != rr.PRONOUN_STRATUM]
cue_ids = [t for _, t, _ in cal]
units = b0c.exposed_units(inputs)
check("139 calibration cues (index order, 3 strata) == rr.calibration_cues(b0c.exposed_units(inputs))", cal == rr.calibration_cues(units) and len(cal) == 139,
      dict(Counter(cls for _, _, cls in cal)))
cue_ids_sha = hashlib.sha256(json.dumps(cue_ids, separators=(",", ":")).encode()).hexdigest()
check("calibration-cue id list digest starts 106cfecc", cue_ids_sha.startswith(EXPECTED["cue_ids"]) and cue_ids_sha == pm.sha256_text(pm.canonical_json(cue_ids)), cue_ids_sha)
fresh = [(c["word"], int(c["token_id"]), c["class"]) for c in c024["cues"]]
check("no fresh cue id is in μ_noun's or μ_cue's rows", not ({t for _, t, _ in fresh} & (set(noun_ids) | set(cue_ids))))

# ---------------------------------------------------------------- centroids and scores: rr (for the digests) and own numpy float64
bindings = rr.score_bindings(W_E, pool, units)
mu_noun_rr, mu_cue_rr = rr.centroids(W_E, bindings)
check("μ_noun digest (rc.tensor_digest) starts 87930f01", bindings["mu_noun_sha256"].startswith(EXPECTED["mu_noun"]) and rc.tensor_digest(mu_noun_rr) == bindings["mu_noun_sha256"],
      bindings["mu_noun_sha256"])
check("μ_cue digest (rc.tensor_digest) starts f8cca954", bindings["mu_cue_sha256"].startswith(EXPECTED["mu_cue"]) and rc.tensor_digest(mu_cue_rr) == bindings["mu_cue_sha256"],
      bindings["mu_cue_sha256"])
E = W_E.detach().cpu().numpy().astype(np.float64)
mu_noun = E[noun_ids].mean(axis=0)
mu_cue = E[cue_ids].mean(axis=0)
exact_noun = np.array([math.fsum(E[noun_ids, j]) / len(noun_ids) for j in range(E.shape[1])])
check("own centroids within 1e-15 of rr's and of an exactly rounded (fsum) mean", np.max(np.abs(mu_noun - mu_noun_rr.numpy())) < 1e-15
      and np.max(np.abs(mu_cue - mu_cue_rr.numpy())) < 1e-15 and np.max(np.abs(mu_noun - exact_noun)) < 1e-15,
      f"{np.max(np.abs(mu_noun - mu_noun_rr.numpy())):.1e}, {np.max(np.abs(mu_cue - mu_cue_rr.numpy())):.1e}")


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def own_score(e, m_noun, m_cue):
    return cos(e, m_noun) - cos(e, m_cue)


rr_scores = rr.full_scores(W_E, bindings, [t for _, t, _ in fresh])
own_scores = [own_score(E[t], mu_noun, mu_cue) for _, t, _ in fresh]
diff = max(abs(a - b) for a, b in zip(rr_scores, own_scores))
check("own 40 scores agree with rr.full_scores within 1e-12", diff < 1e-12, f"max |Δ| {diff:.2e}")
scores_sha = rc.tensor_digest(torch.tensor(rr_scores, dtype=torch.float64))
own_scores_sha = rc.tensor_digest(torch.tensor(own_scores, dtype=torch.float64))
check("40-score digest (rc.tensor_digest of rr's float64 vector, file cue order) starts a703ac16", scores_sha.startswith(EXPECTED["scores"]),
      f"{scores_sha} (own implementation's vector digest {own_scores_sha[:16]}…, bit-identical: {own_scores_sha == scores_sha})")
rr_loo = rr.calibration_scores(W_E, bindings)
own_loo = []
for k in range(len(cue_ids)):
    rest = cue_ids[:k] + cue_ids[k + 1:]
    own_loo.append(own_score(E[cue_ids[k]], mu_noun, E[rest].mean(axis=0)))
diff_loo = max(abs(a - b) for a, b in zip(rr_loo, own_loo))
check("own 139 leave-one-out scores agree with rr.calibration_scores within 1e-12", diff_loo < 1e-12, f"max |Δ| {diff_loo:.2e}")
maximum = max(rr_loo)
arg = cal[rr_loo.index(maximum)]
check("calibration maximum (max LOO over the 139) ≈ +0.135020", abs(maximum - 0.135020) < 5e-7 and abs(max(own_loo) - maximum) < 1e-12, f"{maximum:+.9f} ({arg[0]}, {arg[2]})")
by_class = {cls: [(w, s) for (w, _, c), s in zip(fresh, rr_scores) if c == cls] for cls in rr.CLASSES}
e_above = [w for w, s in by_class["E"] if s > maximum]
check("exactly 5 of 8 E cues above the calibration maximum", len(e_above) == 5, e_above)
check("every E above every N", min(s for _, s in by_class["E"]) > max(s for _, s in by_class["N"]),
      f"min E {min(s for _, s in by_class['E']):+.6f} vs max N {max(s for _, s in by_class['N']):+.6f}")
above_all = [(w, cls) for (w, _, cls), s in zip(fresh, rr_scores) if s > maximum]
print(f"  fresh cues above the calibration maximum: {len(above_all)} of 40: {above_all}")
for cls in rr.CLASSES:
    values = [s for _, s in by_class[cls]]
    print(f"  {cls}: min {min(values):+.6f} max {max(values):+.6f} mean {sum(values) / 8:+.6f} | " + ", ".join(f"{w} {s:+.6f}" for w, s in by_class[cls]))
# design background item 8 (rounded)
e_vals, n_vals = dict(by_class["E"]), dict(by_class["N"])
check("design item 8: E +0.108 (lion) … +0.298 (horse), mean +0.174; N −0.229 (honest) … −0.045 (nervous), mean −0.146",
      round(e_vals["lion"], 3) == 0.108 and round(e_vals["horse"], 3) == 0.298 and round(sum(e_vals.values()) / 8, 3) == 0.174
      and min(e_vals, key=e_vals.get) == "lion" and max(e_vals, key=e_vals.get) == "horse"
      and round(n_vals["honest"], 3) == -0.229 and round(n_vals["nervous"], 3) == -0.045 and round(sum(n_vals.values()) / 8, 3) == -0.146
      and min(n_vals, key=n_vals.get) == "honest" and max(n_vals, key=n_vals.get) == "nervous")
gap = min(e_vals.values()) - max(n_vals.values())
mean_diff = sum(e_vals.values()) / 8 - sum(n_vals.values()) / 8
cos_noun = {cls: np.mean([cos(E[t], mu_noun) for _, t, c in fresh if c == cls]) for cls in ("E", "N")}
cos_cue = {cls: np.mean([cos(E[t], mu_cue) for _, t, c in fresh if c == cls]) for cls in ("E", "N")}
shift_bd = np.mean([s_p - s_s for (_, s_p), (_, s_s) in zip(by_class["B"], by_class["D"])])
shift_ce = np.mean([s_p - s_s for (_, s_p), (_, s_s) in zip(by_class["C"], by_class["E"])])
print(f"  gap min E − max N {gap:+.6f} (design +0.154); mean difference {mean_diff:+.6f} (design +0.320); cos to μ_noun E {cos_noun['E']:+.3f} N {cos_noun['N']:+.3f} "
      f"(design +0.220/+0.046); cos to μ_cue E {cos_cue['E']:+.3f} N {cos_cue['N']:+.3f} (design +0.046/+0.192); plural shift B−D {shift_bd:+.3f} (design +0.10), "
      f"C−E {shift_ce:+.3f} (design +0.12)")
check("no forward and no capture after the load", COUNTS["module_calls_after_load"] == 0 and COUNTS["pm_capture"] == 0, dict(COUNTS))
check("no write under the repository attempted", not rguard.STATE["refused"], rguard.STATE["refused"])
OUT.update({"parameters_sha256": parameters_sha, "embedding_sha256": embedding_sha, "noun_row_ids_sha256": noun_ids_sha, "calibration_cue_ids_sha256": cue_ids_sha,
            "mu_noun_sha256": bindings["mu_noun_sha256"], "mu_cue_sha256": bindings["mu_cue_sha256"], "scores_sha256": scores_sha,
            "own_scores_sha256": own_scores_sha, "scores": {cls: by_class[cls] for cls in rr.CLASSES}, "calibration_maximum": maximum,
            "calibration_maximum_cue": arg, "E_above": e_above, "counts": dict(COUNTS), "max_abs_diff_scores": diff, "max_abs_diff_loo": diff_loo,
            "fails": FAILS})
(HERE / "review_scores.json").write_text(json.dumps(OUT, indent=1, default=float) + "\n")
print(f"FAILS: {len(FAILS)}")
for f in FAILS:
    print("  ", f)
sys.exit(1 if FAILS else 0)
