"""Independent review of Experiment 024's calibrate — part 2 (weights only; every forward refused): the model and
embedding digests, the 158 noun rows and 139 cue rows, the centroids, the leave-one-out scores in exact arithmetic
against the record, and the expected future lock metadata (the 40 fresh scores) with the canonical module. Read-only."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/calibration_review024")
import rguard  # noqa: E402  (first: the audit hook)

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from decimal import Decimal, getcontext  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

getcontext().prec = 60
ROOT = Path(rguard.ROOT)
SCRATCH = Path(__file__).resolve().parent
results = {"checks": [], "failures": []}


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail != "" else ""), flush=True)
    results["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:2000]})
    if not ok:
        results["failures"].append(name)


def cjson(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def own_tensor_digest_f64(values):
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(json.dumps(list(array.shape)).encode("ascii") + b"|<f8|" + array.tobytes()).hexdigest()


# --- the forward guard: count module calls during the load, refuse every call afterwards; refuse capture entry points
calls = {"during_load": 0, "after_load_refused": 0, "capture_refused": 0}
phase = {"loading": True}
_original_call = torch.nn.Module.__call__


def _guarded_call(self, *args, **kwargs):
    if phase["loading"]:
        calls["during_load"] += 1
        return _original_call(self, *args, **kwargs)
    calls["after_load_refused"] += 1
    raise RuntimeError("review: a torch module was called after the weights were loaded")


torch.nn.Module.__call__ = _guarded_call
from neural_decompiler import plural_mechanism as pm  # noqa: E402


def _refuse_capture(*args, **kwargs):
    calls["capture_refused"] += 1
    raise RuntimeError("review: a capture or intervention entry point was reached")


refused_names = [name for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions") if hasattr(pm, name)]
for name in refused_names:
    setattr(pm, name, _refuse_capture)
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import models  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

print("refused capture entry points:", refused_names)
model = models.load_model(models.PYTHIA_70M)
phase["loading"] = False
print("module calls during the load:", calls["during_load"])
# static proof, without calling anything: the model and every submodule resolve __call__ to the refusing wrapper
import inspect  # noqa: E402
overriding = sorted({type(m).__name__ for m in model.modules() if type(m).__call__ is not _guarded_call})
delegating = {name: "super().__call__" in inspect.getsource(next(type(m) for m in model.modules() if type(m).__name__ == name).__call__) for name in overriding}
check("forward refusal armed: the model resolves __call__ to the refusing wrapper; every submodule class either does too or overrides __call__ only to delegate to super().__call__",
      type(model).__call__ is _guarded_call and all(delegating.values()), f"{sum(1 for _ in model.modules())} modules; overriding classes {delegating}")
record = json.loads((ROOT / "outputs/experiment-024/candidate-calibration.json").read_text(encoding="utf-8"))
freeze = json.loads((ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json").read_text(encoding="utf-8"))
index = json.loads((ROOT / "experiments/023-block0-completion/exposed-cells.json").read_text(encoding="utf-8"))

print("== Item 2 (model part): model and embedding digests ==")
check("pinned revision loaded", models.resolved_revision(model) == models.PYTHIA_70M.revision == record["dependencies"]["model"]["revision"], models.resolved_revision(model))
h = hashlib.sha256()
named = sorted(model.named_parameters(), key=lambda kv: kv[0])
for name, parameter in named:
    array = parameter.detach().cpu().contiguous().to(torch.float32).numpy().astype("<f4", copy=False)
    h.update(name.encode("utf-8") + b"|" + json.dumps(list(array.shape)).encode("ascii") + b"|" + array.tobytes())
own_params = h.hexdigest()
canon_params = rr.parameters_digest(model)
check("parameter digest: own implementation == canonical == record == fd953f1c…", own_params == canon_params == record["dependencies"]["model"]["parameters_sha256"]
      == "fd953f1c745299bedfd1f242fc15c7fe31211f79d8bc0d3ca46372962e7fc0e5", f"{own_params} over {len(named)} named parameters")
W_E = pm.Weights.from_model(model).W_E
direct = model.embed.W_E.detach().cpu().to(torch.float32)
hf = [(n, p) for n, p in model.named_parameters() if "emb" in n.lower() and tuple(p.shape) == (50304, 512)]
check("W_E (pm.Weights) == model.embed.W_E == the named [50304, 512] embedding parameter, float32", W_E.dtype == torch.float32 and tuple(W_E.shape) == (50304, 512)
      and torch.equal(W_E, direct) and len(hf) >= 1 and all(torch.equal(W_E, p.detach().cpu().to(torch.float32)) for _, p in hf), [n for n, _ in hf])
emb = np.ascontiguousarray(W_E.numpy().astype("<f4", copy=False))
own_emb = hashlib.sha256(json.dumps(list(emb.shape)).encode("ascii") + b"|<f4|" + emb.tobytes()).hexdigest()
check("embedding digest: own implementation == canonical == record == 9cd6f39b…", own_emb == rr.embedding_digest(W_E) == record["score"]["embedding_sha256"]
      == record["dependencies"]["model"]["embedding_sha256"] == "9cd6f39b3adfdb74cffe046af459634982684bff9aef2a981760326f1748eaaf", own_emb)
del model  # nothing below needs the model object; W_E is a detached copy

print("== Item 2 (canonical part): frozen inputs, 020 readout pin, exposed-states digest, dependencies ==")
inputs = ul.load_frozen_inputs(ROOT)
states_digest = ul.exposed_states_digest(inputs.closure["exploration"]["locked_states"])
check("exposed locked-states digest (ul.exposed_states_digest) == record 26c21d63…", states_digest == record["dependencies"]["readout_020"]["exposed_states_sha256"], states_digest)
check("frozen-input digests (inputs.digests over rc.DIGEST_KEYS) == record readout_020.frozen_input_digests",
      {k: inputs.digests[k] for k in rc.DIGEST_KEYS} == record["dependencies"]["readout_020"]["frozen_input_digests"])
digests = rr.base_digests(inputs, b0c.verify_022_inputs(ROOT), rr.verify_023_inputs(ROOT))
check("record inputs == rr.base_digests recomputed now (020 chain, 022, 023)", digests == record["inputs"])
check("record dependencies == rr.scientific_dependencies recomputed now", rr.scientific_dependencies(inputs, parameters_sha256=own_params, embedding_sha256=own_emb) == record["dependencies"])
check("module blobs on disk == rr.FROZEN_BLOBS (rr.assert_frozen_blobs)", rr.assert_frozen_blobs() == rr.FROZEN_BLOBS == record["module_blobs"])
try:
    rr.verify_calibration_record(record, rr.PRODUCTION)
    ok, detail = True, "verified"
except Exception as error:  # noqa: BLE001
    ok, detail = False, repr(error)
check("canonical verify_calibration_record(record, PRODUCTION) passes", ok, detail)

print("== Item 3: independent nounness construction ==")
pool = inputs.pool
nouns = [noun for noun in pool.nouns if len(noun.sg_ids) == 1]
check("79 scorable (single-token) exposed nouns, in the 023 index's noun order", len(nouns) == 79 and [n.lexical_key for n in nouns] == index["nouns"])
noun_ids = [int(i) for n in nouns for i in (n.sg_ids[0], n.pl_ids[0])]
own_noun_digest = hashlib.sha256(("[" + ",".join(str(i) for i in noun_ids) + "]").encode("utf-8")).hexdigest()
check("158 noun-form ids == record; own digest == canonical == 08fb9b4f…", len(noun_ids) == 158 and noun_ids == record["score"]["noun_row_ids"] == rr.noun_row_ids(pool)
      and own_noun_digest == pm.sha256_text(pm.canonical_json(noun_ids)) == record["score"]["noun_row_ids_sha256"] == "08fb9b4faf7e8a807c57fc98bd45887f237f90a4675922ec4c98ce9c79d79871", own_noun_digest)
check("158 noun-form ids distinct (sg ≠ pl for every noun, no shared ids)", len(set(noun_ids)) == 158)
cal_from_index = [(w, int(t), s) for w, t, s in index["cues"] if s in b0c.STRATA]
units = b0c.exposed_units(inputs)
cal_canonical = [(w, int(t), s) for w, t, s in units.cues if s in b0c.STRATA]
cue_ids = [t for _, t, _ in cal_from_index]
own_cue_digest = hashlib.sha256(("[" + ",".join(str(i) for i in cue_ids) + "]").encode("utf-8")).hexdigest()
check("139 cue ids: 023 index order == b0c.exposed_units order == record; own digest == canonical == 106cfecc…", cal_from_index == cal_canonical and len(cue_ids) == 139
      and cue_ids == record["score"]["calibration_cue_ids"] and own_cue_digest == pm.sha256_text(pm.canonical_json(cue_ids)) == record["score"]["calibration_cue_ids_sha256"]
      == "106cfeccd7b9a10362fb7aef1e3558506e2a7299e4bcfa4ec42aa0f46f774dbf", own_cue_digest)
check("139 cue ids distinct; disjoint from the noun rows", len(set(cue_ids)) == 139 and not (set(cue_ids) & set(noun_ids)))
bindings_now = rr.score_bindings(W_E, pool, units)
check("canonical rr.score_bindings recomputed now == record score (all keys)", bindings_now == record["score"])
tokenizer = None
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(models.PYTHIA_70M.model_id, revision=models.PYTHIA_70M.revision)
except Exception as error:  # noqa: BLE001
    print("tokenizer unavailable:", repr(error))
if tokenizer is not None:
    bad_cues = [(w, t, tokenizer.decode([t])) for w, t, _ in cal_from_index if tokenizer.decode([t]) != " " + w]
    bad_nouns = [(n.lexical_key, tokenizer.decode([n.sg_ids[0]])) for n in nouns if tokenizer.decode([n.sg_ids[0]]).strip() != n.lexical_key]
    bad_fresh = [(c["word"], c["token_id"]) for c in freeze["cues"] if tokenizer.encode(" " + c["word"]) != [int(c["token_id"])]]
    check("tokenizer: each calibration cue id decodes to ' ' + word; each noun's singular id decodes to its key; each fresh ' ' + word encodes to its single id",
          not bad_cues and not bad_nouns and not bad_fresh, (bad_cues[:3], bad_nouns[:3], bad_fresh[:3]))

# canonical centroids and their digests (canonical byte identity), and my own tensor-digest routine on the same bytes
mu_noun_t, mu_cue_t = rr.centroid(W_E, noun_ids), rr.centroid(W_E, cue_ids)
check("μ_noun digest: canonical rc.tensor_digest == own digest routine == record == 87930f01…", rc.tensor_digest(mu_noun_t) == own_tensor_digest_f64(mu_noun_t.numpy())
      == record["score"]["mu_noun_sha256"] == "87930f0150e7263963d593303defd63f87f601f6892717c61d2f20c704f151d2")
check("μ_cue digest: canonical rc.tensor_digest == own digest routine == record == f8cca954…", rc.tensor_digest(mu_cue_t) == own_tensor_digest_f64(mu_cue_t.numpy())
      == record["score"]["mu_cue_sha256"] == "f8cca95464092089dae6cd668d3e29bd26a89131b0f5994c35f6503a282121c4")

# my own exact route: every float32 value as an integer on one dyadic scale; sums exact; the scale cancels in the cosine
E64 = W_E.double().numpy()
relevant = sorted(set(noun_ids) | set(cue_ids) | {int(c["token_id"]) for c in freeze["cues"]})
scale_exp = 0
for i in relevant:
    for v in E64[i].tolist():
        if v != 0.0:
            scale_exp = max(scale_exp, int(math.log2(float(v).as_integer_ratio()[1])))
SCALE = 1 << scale_exp


def zrow(i):
    out = []
    for v in E64[i].tolist():
        num, den = float(v).as_integer_ratio()
        out.append(num * (SCALE // den))
    return out


Z = {i: zrow(i) for i in relevant}
check("exact encoding round-trips every relevant float32 value", all(float(Z[i][d]) / SCALE == E64[i][d] for i in relevant[:50] for d in range(512)), f"scale 2^{scale_exp}")
Zn = [sum(Z[i][d] for i in noun_ids) for d in range(512)]
Zc = [sum(Z[i][d] for i in cue_ids) for d in range(512)]
Cn = sum(v * v for v in Zn)
Cc = sum(v * v for v in Zc)


def cos_exact(z, zm, cm):
    a = sum(p * q for p, q in zip(z, zm))
    b = sum(p * p for p in z)
    return Decimal(a) / (Decimal(b) * Decimal(cm)).sqrt()


# exact centroids (rounded once to float64) against the canonical torch means: numerical disagreement only
exact_mu_noun = np.array([float(Decimal(v) / Decimal(158 * SCALE)) for v in Zn])
exact_mu_cue = np.array([float(Decimal(v) / Decimal(139 * SCALE)) for v in Zc])
dn = np.abs(exact_mu_noun - mu_noun_t.numpy())
dc = np.abs(exact_mu_cue - mu_cue_t.numpy())
check("centroids: canonical torch means vs exactly rounded means (numerical disagreement only)", dn.max() < 1e-15 and dc.max() < 1e-15,
      f"μ_noun max |Δ| {dn.max():.2e} ({int((dn > 0).sum())}/512 coords differ); μ_cue max |Δ| {dc.max():.2e} ({int((dc > 0).sum())}/512 coords differ)")

entries = record["calibration_cues"]
exact_loo = []
for k, t in enumerate(cue_ids):
    zl = [Zc[d] - Z[t][d] for d in range(512)]  # exact: the integer sum of the other 138 rows
    cl = sum(v * v for v in zl)
    exact_loo.append(cos_exact(Z[t], Zn, Cn) - cos_exact(Z[t], zl, cl))
rec_loo = [e["nounness_loo"] for e in entries]
errors = [abs(Decimal(r) - x) for r, x in zip(rec_loo, exact_loo)]
own_float = [float(x) for x in exact_loo]
check("all 139 record LOO scores vs own exact route (numerical disagreement only)", max(errors) < Decimal("1e-14"),
      f"max |Δ| {float(max(errors)):.3e}; bitwise equal to the exactly rounded value {sum(1 for r, o in zip(rec_loo, own_float) if r == o)}/139")
canon_loo = rr.calibration_scores(W_E, record["score"])
check("canonical rr.calibration_scores recomputed now == record nounness_loo bit for bit (139)", canon_loo == rec_loo)
# the subtraction LOO variant is never used; show that the canonical direct mean is what the record holds
order_rec = sorted(range(139), key=lambda i: rec_loo[i])
order_own = sorted(range(139), key=lambda i: exact_loo[i])
gaps = sorted(exact_loo[order_own[i + 1]] - exact_loo[order_own[i]] for i in range(138))
check("the 139 LOO scores are distinct and ordered identically by the record and the exact route", order_rec == order_own and len(set(rec_loo)) == 139,
      f"smallest gap between consecutive exact scores {float(gaps[0]):.3e} vs max disagreement {float(max(errors)):.3e}")
imax = max(range(139), key=lambda i: rec_loo[i])
second = sorted(rec_loo)[-2]
check("calibration maximum = 0.13502027836111233, the same cue in both routes", rec_loo[imax] == 0.13502027836111233 == record["descriptive"]["maximum_calibration_score"]
      and imax == max(range(139), key=lambda i: exact_loo[i]), f"{entries[imax]['word']} ({entries[imax]['stratum']}); second {second!r}")
pron = record["pronoun_cues"]
pron_exact = [cos_exact(zrow(p["token_id"]), Zn, Cn) - cos_exact(zrow(p["token_id"]), Zc, Cc) for p in pron]
check("36 pronoun scores (full μ_cue, descriptive) vs own exact route", max(abs(Decimal(p["nounness"]) - x) for p, x in zip(pron, pron_exact)) < Decimal("1e-14"),
      f"max |Δ| {float(max(abs(Decimal(p['nounness']) - x) for p, x in zip(pron, pron_exact))):.3e}")

print("== Item 10: the fresh-score boundary (weights only; expected future lock metadata; not a lock) ==")
fresh = freeze["cues"]
fresh_ids = [int(c["token_id"]) for c in fresh]
canon_scores = rr.full_scores(W_E, record["score"], fresh_ids)
scores_digest = rc.tensor_digest(torch.tensor(canon_scores, dtype=torch.float64))
check("40-score digest: rc.tensor_digest of rr.full_scores in the freeze file's cue order == a703ac16…", scores_digest == "a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995"
      and own_tensor_digest_f64(canon_scores) == scores_digest, scores_digest)
fresh_exact = [cos_exact(Z[t], Zn, Cn) - cos_exact(Z[t], Zc, Cc) for t in fresh_ids]
check("40 fresh scores: canonical vs own exact route (numerical disagreement only)", max(abs(Decimal(s) - x) for s, x in zip(canon_scores, fresh_exact)) < Decimal("1e-14"),
      f"max |Δ| {float(max(abs(Decimal(s) - x) for s, x in zip(canon_scores, fresh_exact))):.3e}")
maximum = max(rec_loo)
E = [(c["word"], s) for c, s in zip(fresh, canon_scores) if c["class"] == "E"]
N = [(c["word"], s) for c, s in zip(fresh, canon_scores) if c["class"] == "N"]
above = [w for w, s in E if s > maximum]
check("5 of the 8 E cues above the calibration maximum 0.13502027836111233: apple, horse, doctor, poet, dragon", above == ["apple", "horse", "doctor", "poet", "dragon"], above)
margins = sorted(abs(s - maximum) for _, s in E)
check("the extrapolation flags are robust to the numerical disagreement", margins[0] > 1e-6, f"closest E to the maximum is {margins[0]:.4f} away")
check("every E above every N", min(s for _, s in E) > max(s for _, s in N), f"min E {min(E, key=lambda x: x[1])} > max N {max(N, key=lambda x: x[1])}")
by_class = {k: [round(s, 6) for c, s in zip(fresh, canon_scores) if c["class"] == k] for k in "NBDCE"}
print("    fresh scores by class (rounded for display):", by_class)
print("    E mean", sum(s for _, s in E) / 8, "N mean", sum(s for _, s in N) / 8)
check("no fresh token id is in μ_cue or μ_noun", not (set(fresh_ids) & (set(cue_ids) | set(noun_ids))))

json.dump({"cue_ids": cue_ids, "loo_exact_float": own_float, "loo_record": rec_loo, "fresh_exact_float": [float(x) for x in fresh_exact], "fresh_canonical": canon_scores,
           "pron_exact_float": [float(x) for x in pron_exact]}, open(SCRATCH / "own_scores.json", "w"))
check("forward guard: no module call during the load, none after it, no capture entry point reached",
      calls["during_load"] == 0 and calls["after_load_refused"] == 0 and calls["capture_refused"] == 0, calls)
print("guard events:", rguard.EVENTS)
check("review guard: no repository write attempted, no forbidden read attempted", not rguard.EVENTS["refused_writes"] and not rguard.EVENTS["refused_reads"])
json.dump(results, open(SCRATCH / "review_weights.json", "w"), indent=1)
print("FAILURES:", results["failures"] or "none")
