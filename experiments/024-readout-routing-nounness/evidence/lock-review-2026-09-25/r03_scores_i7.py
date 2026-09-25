"""Items 4 and 10 (and the model side of item 3) of the independent lock review.

Weights only: the pinned checkpoint is read twice — (a) its raw safetensors bytes (no model object at all), and (b) the
runner's own loader (TransformerBridge), with torch.nn.Module.__call__ counted during the load and refused afterwards,
every module class's forward replaced by a refusing stub after the load, and every capture / intervention /
measurement entry point refused. The 40 fresh scores and the 139 leave-one-out calibration scores are reconstructed
EXACTLY (integer arithmetic on the fp16 checkpoint values, one mpmath evaluation at 60 digits per cosine) and with an
independent float64 numpy route; the canonical module is called only for byte identity (fresh_quantities, the digests)
and for the I7 rehearsal and the validate_lock dry run.
"""
import copy
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_guard  # noqa: E402

ROOT = review_guard.ROOT
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail != "" else ""), flush=True)
    if not ok:
        FAIL.append(name)


import mpmath  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

guard = review_guard.ForwardGuard()  # capture entry points refused from here on; Module.__call__ counted until sealed

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

mpmath.mp.dps = 60
OUT = ROOT / "outputs/experiment-024"
lock_bytes = (OUT / "candidate-lock.json").read_bytes()
lock = json.loads(lock_bytes)
prereg_text = (OUT / "candidate-preregistration.md").read_bytes().decode("utf-8")
state = json.loads((OUT / "results.json").read_bytes())
record_path = ROOT / rr.CALIBRATION_RELATIVE_PATH
record = json.loads(record_path.read_bytes())
freeze_path = ROOT / rr.CONFIRMATION_RELATIVE_PATH
freeze = json.loads(freeze_path.read_bytes())
fresh_tokens = [(c["word"], int(c["token_id"]), c["class"]) for c in freeze["cues"]]
noun_ids = [int(i) for i in record["score"]["noun_row_ids"]]
cue_ids = [int(i) for i in record["score"]["calibration_cue_ids"]]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canon(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def float64_digest(values) -> str:
    return sha256_bytes(json.dumps([len(values)]).encode("ascii") + b"|<f8|" + struct.pack(f"<{len(values)}d", *values))


print("=" * 100)
print("inputs: frozen-input chain (read only; the runner's _base/_confirmation replicated without a Runner)")
inputs = ul.load_frozen_inputs(ROOT)
digests_022 = b0c.verify_022_inputs(ROOT)
digests_023 = rr.verify_023_inputs(ROOT)
path_022 = ROOT / ul.CONFIRMATION_RELATIVE_PATH
confirmation_022 = json.loads(path_022.read_text(encoding="utf-8"))
sha_022 = rc.file_sha256(path_022)
path_023 = ROOT / b0c.CONFIRMATION_RELATIVE_PATH
confirmation_023 = b0c.load_confirmation_023(path_023, inputs, confirmation_022, sha_022)
payload_023 = json.loads(path_023.read_text(encoding="utf-8"))
exclusion_base = b0c.exclusion(inputs, confirmation_022, sha_022)
excluded = rr.exclusion(exclusion_base, payload_023, rc.file_sha256(path_023))
digests = rr.base_digests(inputs, digests_022, digests_023)
confirmation = rr.load_confirmation_024(freeze_path, inputs.pool, excluded, rr.PRODUCTION)
confirmation_sha = rc.file_sha256(freeze_path)
check("lock.inputs == the frozen-input digests computed now (rr.base_digests over 020/022/023)", lock["inputs"] == {k: digests[k] for k in rr.DIGEST_KEYS} and set(lock["inputs"]) == set(rr.DIGEST_KEYS))
check("020 readout binding: exposed-states digest and frozen-input digests recomputed now", lock["dependencies"]["readout_020"]["exposed_states_sha256"]
      == ul.exposed_states_digest(inputs.closure["exploration"]["locked_states"]) == "26c21d6386ebfaf45cd33f1e9ceaae08a225d5c4ae3a5a9476e28bf1683b0f83"
      and lock["dependencies"]["readout_020"]["frozen_input_digests"] == {k: inputs.digests[k] for k in rc.DIGEST_KEYS}
      and lock["dependencies"]["readout_020"]["blob"] == rr.FROZEN_BLOBS["readout_decompilation.py"])
placeholder = rr.scientific_dependencies(inputs, parameters_sha256="", embedding_sha256="")
check("non-model dependencies recomputed now == the lock's", {k: v for k, v in placeholder.items() if k != "model"} == {k: v for k, v in lock["dependencies"].items() if k != "model"})
check("the confirmation object's cues == the freeze's, in order", [(t["word"], t["token_id"], t["class"]) for t in confirmation.tokens] == fresh_tokens)
noun_keys = [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]
check("lock.noun_keys == the 79 scorable exposed nouns in pool order", lock["noun_keys"] == noun_keys and len(noun_keys) == 79)
pool_noun_ids = [int(t) for noun in inputs.pool.nouns if noun.single_token for t in (noun.sg_ids[0], noun.pl_ids[0])]
units = b0c.exposed_units(inputs)
pool_cue_ids = [int(t) for _, t, cls in units.cues if cls in b0c.STRATA]
check("score id lists: record == pool-derived (158 noun rows, 139 calibration cues) == calibration entries' ids", pool_noun_ids == noun_ids and len(noun_ids) == 158
      and pool_cue_ids == cue_ids and len(cue_ids) == 139 and [e["token_id"] for e in record["calibration_cues"]] == cue_ids)
check("id-list digests (own canonical sha256) == lock/record", sha256_bytes(canon(noun_ids).encode()) == lock["score"]["noun_row_ids_sha256"]
      and sha256_bytes(canon(cue_ids).encode()) == lock["score"]["calibration_cue_ids_sha256"])

print("=" * 100)
print("tokenizer (no model): every fresh and calibration cue word is ' ' + w -> exactly [its token id]")
from transformers import AutoTokenizer  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped", revision="e93a9faa9c77e5d09219f6c868bfc7a1bd65593c")
bad = [(w, t, tokenizer.encode(" " + w, add_special_tokens=False)) for w, t, _ in fresh_tokens if tokenizer.encode(" " + w, add_special_tokens=False) != [t]]
bad += [(e["word"], e["token_id"]) for e in record["calibration_cues"] if tokenizer.encode(" " + e["word"], add_special_tokens=False) != [e["token_id"]]]
check("40 fresh + 139 calibration words tokenize to their bound single ids", not bad, bad)

print("=" * 100)
print("ITEM 4 (a): exact reconstruction from the raw checkpoint bytes (no model object)")
snapshot = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c/model.safetensors"
raw = snapshot.read_bytes()
check("checkpoint file sha256 == its content-addressed LFS blob id 3da38833… (pinned revision's snapshot)", sha256_bytes(raw) == snapshot.resolve().name == "3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd")
header_len = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8:8 + header_len])
entry = header["gpt_neox.embed_in.weight"]
start, end = entry["data_offsets"]
emb16 = np.frombuffer(raw[8 + header_len + start: 8 + header_len + end], dtype="<f2").reshape(entry["shape"])
check("embed_in.weight: F16 [50304, 512]", entry["dtype"] == "F16" and list(emb16.shape) == [50304, 512])
del raw
# exact integers: every fp16 value times 2^24 is an integer (the smallest fp16 subnormal is 2^-24)
scaled = emb16.astype(np.float64) * float(2 ** 24)
check("fp16 × 2^24 is integral everywhere (exact integer representation)", bool(np.all(scaled == np.round(scaled))))
Z = scaled.astype(np.int64)


def ivec(token_id):
    return [int(v) for v in Z[int(token_id)]]


def isum(ids):
    total = np.zeros(Z.shape[1], dtype=object)
    for i in ids:
        total = total + Z[int(i)].astype(object)
    return [int(v) for v in total]


def idot(a, b):
    return sum(x * y for x, y in zip(a, b))


S_noun = isum(noun_ids)
S_cue = isum(cue_ids)
nn_noun = idot(S_noun, S_noun)
nn_cue = idot(S_cue, S_cue)


def cosine_exact(dot, na, nb):
    return mpmath.mpf(dot) / mpmath.sqrt(mpmath.mpf(na) * mpmath.mpf(nb))


def exact_fresh(token_id):
    e = ivec(token_id)
    ee = idot(e, e)
    return cosine_exact(idot(e, S_noun), ee, nn_noun) - cosine_exact(idot(e, S_cue), ee, nn_cue)


def exact_loo(token_id):
    e = ivec(token_id)
    ee = idot(e, e)
    de = idot(e, S_cue)
    # μ_cue without this cue ∝ S_cue − e: ⟨e, S − e⟩ = de − ee; ‖S − e‖² = ‖S‖² − 2 de + ee (exact integers)
    return cosine_exact(idot(e, S_noun), ee, nn_noun) - cosine_exact(de - ee, ee, nn_cue - 2 * de + ee)


exact_scores = [exact_fresh(t) for _, t, _ in fresh_tokens]
exact_loo_scores = [exact_loo(t) for t in cue_ids]
lock_scores = [c["nounness"] for c in lock["fresh"]["cues"]]
check("lock fresh cues == the freeze's 40, in the freeze's order", [(c["word"], c["token_id"], c["class"]) for c in lock["fresh"]["cues"]] == fresh_tokens)
diffs = [abs(mpmath.mpf(l) - x) for l, x in zip(lock_scores, exact_scores)]
ulps = [float(d / mpmath.mpf(math.ulp(l))) for d, l in zip(diffs, lock_scores)]
rounded = [float(x) for x in exact_scores]
print(f"   lock vs exact (60 digits): max |diff| {float(max(diffs)):.3e}; max {max(ulps):.2f} ulp; {sum(1 for r, l in zip(rounded, lock_scores) if r == l)}/40 lock values equal the correctly rounded exact value")
check("every lock score within 1e-15 of the exact score", float(max(diffs)) <= 1e-15)
loo_diffs = [abs(mpmath.mpf(e["nounness_loo"]) - x) for e, x in zip(record["calibration_cues"], exact_loo_scores)]
print(f"   record LOO scores vs exact: max |diff| {float(max(loo_diffs)):.3e}")
check("every record leave-one-out score within 1e-15 of the exact value", float(max(loo_diffs)) <= 1e-15)
order_lock = sorted(range(40), key=lambda i: lock_scores[i])
order_exact = sorted(range(40), key=lambda i: exact_scores[i])
gaps = sorted(float(exact_scores[order_exact[i + 1]] - exact_scores[order_exact[i]]) for i in range(39))
check("ordering of the 40 lock scores == ordering of the exact scores (identical permutation)", order_lock == order_exact, f"smallest gap between neighbours {gaps[0]:.3e}")
kmax = max(range(139), key=lambda k: exact_loo_scores[k])
exact_max = exact_loo_scores[kmax]
second = sorted(exact_loo_scores)[-2]
print(f"   exact calibration maximum {mpmath.nstr(exact_max, 25)} at '{record['calibration_cues'][kmax]['word']}' (id {cue_ids[kmax]}); next {mpmath.nstr(second, 12)}")
check("calibration maximum: record max == 0.13502027836111233 == float(exact max) (within 1e-16)", max(e["nounness_loo"] for e in record["calibration_cues"]) == 0.13502027836111233
      == lock["fresh"]["maximum_calibration_score"] and abs(float(exact_max) - 0.13502027836111233) <= 1e-16)
flags_exact = [x > exact_max for x in exact_scores]
flags_lock = [c["above_calibration_maximum"] for c in lock["fresh"]["cues"]]
margin = min(abs(float(x - exact_max)) for x in exact_scores)
check("above-calibration flags: lock == exact comparison, for all 40", flags_lock == flags_exact, f"closest fresh score to the maximum is {margin:.3e} away")
e_above = [w for (w, _, cls), f in zip(fresh_tokens, flags_exact) if cls == "E" and f]
check("exactly 5 of 8 E above it: apple, horse, doctor, poet, dragon", e_above == ["apple", "horse", "doctor", "poet", "dragon"] and lock["fresh"]["extrapolation"]["E_above_maximum"] == 5)
e_scores = [x for (w, _, cls), x in zip(fresh_tokens, exact_scores) if cls == "E"]
n_scores = [x for (w, _, cls), x in zip(fresh_tokens, exact_scores) if cls == "N"]
check("every E above every N (exact)", min(e_scores) > max(n_scores), f"min E {mpmath.nstr(min(e_scores), 8)} > max N {mpmath.nstr(max(n_scores), 8)}")
check("descriptively 23 of 40 above the maximum (no outcome force)", sum(flags_exact) == 23 == sum(flags_lock))
for cls in "NBDCE":
    vals = [l for (w, _, c), l in zip(fresh_tokens, lock_scores) if c == cls]
    ab = sum(1 for (w, _, c), f in zip(fresh_tokens, flags_lock) if c == cls and f)
    print(f"   class {cls}: min {min(vals):+.6f} max {max(vals):+.6f} mean {sum(vals) / 8:+.6f}; above the maximum {ab} of 8")
print("   per cue (word, class, lock nounness, exact − lock, above):")
for (w, t, cls), l, x, f in zip(fresh_tokens, lock_scores, exact_scores, flags_lock):
    print(f"     {cls} {w:9s} {t:6d} {l:+.17f} {float(x - mpmath.mpf(l)):+.2e} {'above' if f else ''}")

print("=" * 100)
print("ITEM 4 (b): independent float64 numpy route on the raw checkpoint")
W64 = emb16.astype(np.float64)
mu_noun64 = W64[noun_ids].mean(axis=0)
mu_cue64 = W64[cue_ids].mean(axis=0)


def cos64(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


np_scores = [cos64(W64[t], mu_noun64) - cos64(W64[t], mu_cue64) for _, t, _ in fresh_tokens]
np_diff = max(abs(a - b) for a, b in zip(np_scores, lock_scores))
print(f"   numpy float64 route vs lock: max |diff| {np_diff:.3e}; exactly equal {sum(1 for a, b in zip(np_scores, lock_scores) if a == b)}/40")
check("numpy float64 route agrees with the lock within 1e-14 and orders identically", np_diff <= 1e-14 and sorted(range(40), key=lambda i: np_scores[i]) == order_lock)

print("=" * 100)
print("ITEM 4 (c) / ITEM 10: the runner's loader, sealed; canonical byte identity; the I7 rehearsal")
from neural_decompiler import models  # noqa: E402

model = guard.load()
print("   guard counts after the load:", guard.counts, "; sealed module classes:", len(guard.sealed_classes))
check("torch.nn.Module.__call__ during the load: 0", guard.counts["module_call_during_load"] == 0, guard.counts)
parameters_sha = rr.parameters_digest(model)
own = hashlib.sha256()
named = sorted(model.named_parameters(), key=lambda item: item[0])
for name, parameter in named:
    array = parameter.detach().cpu().contiguous().to(torch.float32).numpy().astype("<f4", copy=False)
    own.update(name.encode("utf-8") + b"|" + json.dumps(list(array.shape)).encode("ascii") + b"|" + array.tobytes())
check("parameters digest: canonical == own implementation == bound fd953f1c…", parameters_sha == own.hexdigest() == lock["dependencies"]["model"]["parameters_sha256"]
      == "fd953f1c745299bedfd1f242fc15c7fe31211f79d8bc0d3ca46372962e7fc0e5", f"{len(named)} named parameters")
W_E = pm.Weights.from_model(model).W_E
own_emb = sha256_bytes(json.dumps(list(W_E.shape)).encode("ascii") + b"|<f4|" + W_E.contiguous().numpy().astype("<f4", copy=False).tobytes())
check("embedding digest: canonical == own == bound 9cd6f39b…", rr.embedding_digest(W_E) == own_emb == lock["dependencies"]["model"]["embedding_sha256"] == "9cd6f39b3adfdb74cffe046af459634982684bff9aef2a981760326f1748eaaf")
check("the loader's W_E (float32) == the raw fp16 checkpoint rows, bit for bit", W_E.dtype == torch.float32 and torch.equal(W_E, torch.from_numpy(emb16.astype(np.float32))))
progs = ul.ModelPrograms.from_model(model, inputs)  # sealed: any module call here would be refused
check("ModelPrograms.from_model ran with every forward refused; its W_E == Weights.from_model W_E", torch.equal(progs.weights.W_E, W_E), guard.counts)
prog_keys = [progs.nouns.nouns[i].lexical_key for i in progs.scorable]
check("_check_nouns replicated: scorable nouns == pool single-token nouns (79)", prog_keys == noun_keys and len(prog_keys) == 79)
bindings = rr.score_bindings(progs.weights.W_E, inputs.pool, units)
check("rr.score_bindings recomputed from the weights == record.score == lock.score", bindings == record["score"] == lock["score"])
fresh_now = json.loads(pm.canonical_json(rc.json_safe(rr.fresh_quantities(progs.weights.W_E, record["score"], confirmation, record))))
check("canonical rr.fresh_quantities now == lock.fresh (JSON round trip, every key)", fresh_now == lock["fresh"], sorted(k for k in set(fresh_now) | set(lock["fresh"]) if fresh_now.get(k) != lock["fresh"].get(k)))
check("score digest: canonical rc.tensor_digest == own float64 digest of the lock values == a703ac16…",
      rc.tensor_digest(torch.tensor(lock_scores, dtype=torch.float64)) == float64_digest(lock_scores) == lock["fresh"]["scores_sha256"]
      == "a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995")
print(f"   own float64 digest of the correctly rounded EXACT scores: {float64_digest(rounded)} ({'equal' if rounded == lock_scores else 'differs: numerical, not byte, identity'})")
check("centroid digests (mu_noun, mu_cue) recomputed canonically == lock", fresh_now["mu_noun_sha256"] == lock["fresh"]["mu_noun_sha256"] == record["score"]["mu_noun_sha256"]
      and fresh_now["mu_cue_sha256"] == lock["fresh"]["mu_cue_sha256"] == record["score"]["mu_cue_sha256"])
# the I7 rehearsal, exactly as Runner.confirm does it (the model digest outside, then embedding + reproduce under the guard)
rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256=rr.parameters_digest(model), embedding_sha256=lock["dependencies"]["model"]["embedding_sha256"], what="the lock")
rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256=lock["dependencies"]["model"]["parameters_sha256"], embedding_sha256=rr.embedding_digest(progs.weights.W_E), what="the lock")
reproduced = rr.reproduce_lock_quantities(progs.weights.W_E, record, confirmation, lock)
check("I7 rehearsal: rr.reproduce_lock_quantities bitwise_equal, nothing differing", reproduced["bitwise_equal"] and reproduced["differing"] == [] and reproduced["scores_sha256"] == lock["fresh"]["scores_sha256"], reproduced)
check("I7 rehearsal covers predictions, flags, maximum, 5/8 and the sentence (all inside lock.fresh)", fresh_now["cues"] == lock["fresh"]["cues"] and fresh_now["maximum_calibration_score"] == 0.13502027836111233
      and fresh_now["extrapolation"] == lock["fresh"]["extrapolation"])
# sensitivity (pure functions, no model call): a one-ulp change in a bound score, or a wrong parameter digest, is refused
drifted = copy.deepcopy(lock)
drifted["fresh"]["cues"][0]["nounness"] = math.nextafter(drifted["fresh"]["cues"][0]["nounness"], 1.0)
again = rr.reproduce_lock_quantities(progs.weights.W_E, record, confirmation, drifted)
check("I7 detects a one-ulp drift in a bound score (bitwise_equal False, differing ['cues'])", again["bitwise_equal"] is False and again["differing"] == ["cues"], again)
try:
    rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256="0" * 64, embedding_sha256=lock["dependencies"]["model"]["embedding_sha256"], what="the lock")
    check("a wrong parameter digest is refused", False)
except rr.PhaseError as error:
    check("a wrong parameter digest is refused (PhaseError before any prompt)", "different scientific dependencies" in str(error), str(error)[:120])
del model

print("=" * 100)
print("ITEM 3 / 6: validate_lock dry run on the candidate as if installed byte-identically (pure call; no install, no confirm)")
record_sha = rc.file_sha256(record_path)
base_kwargs = dict(state=state, digests=digests, config=rr.PRODUCTION, record=record, record_file_sha256=record_sha, confirmation=confirmation,
                   confirmation_file_sha256=confirmation_sha, dependencies=placeholder, noun_keys=noun_keys, preregistration_text=prereg_text,
                   git_state={"commit": "x" * 40, "dirty": False}, tracked=True, changed_paths=[])


def attempt(label, lock_obj, expect_ok, message="", **overrides):
    kwargs = {**base_kwargs, **overrides}
    try:
        rr.validate_lock(lock_obj, **kwargs)
        outcome, text = True, "accepted"
    except rr.PhaseError as error:
        outcome, text = False, str(error)
    ok = outcome == expect_ok and (expect_ok or message in text)
    check(f"validate_lock {label}: {'accepted' if expect_ok else 'refused'}", ok, text[:140])


attempt("the exact candidate, clean tree, no change since the lock", lock, True)
attempt("with the installation commit's own paths changed (lock, preregistration, README, evidence)", lock, True,
        changed_paths=[rr.LOCK_RELATIVE_PATH, rr.PREREGISTRATION_RELATIVE_PATH, f"{rr.EXPERIMENT_DIR}/README.md", f"{rr.EXPERIMENT_DIR}/evidence/lock-review/REVIEW.md"])
attempt("src/…/readout_routing.py changed since the lock", lock, False, "scientific paths changed since the lock", changed_paths=["src/neural_decompiler/readout_routing.py"])
attempt("the runner run.py changed since the lock", lock, False, "scientific paths changed since the lock", changed_paths=[f"{rr.EXPERIMENT_DIR}/run.py"])
attempt("023's exposed cells changed since the lock", lock, False, "scientific paths changed since the lock", changed_paths=["experiments/023-block0-completion/exposed-cells.f64"])
attempt("a renamed-away scientific file (both paths listed)", lock, False, "scientific paths changed since the lock", changed_paths=["src/neural_decompiler/readout_routing.py", "docs/moved.py"])
attempt("lock commit not an ancestor", lock, False, "not an ancestor", changed_paths=None)
attempt("dirty tree", lock, False, "clean Git tree", git_state={"commit": "x" * 40, "dirty": True})
attempt("untracked lock files", lock, False, "tracked and committed", tracked=False)
attempt("an edited preregistration", lock, False, "preregistration", preregistration_text=prereg_text + "\nedited")
resealed = copy.deepcopy(lock)
resealed["primary"]["null_975"] = 0.2
resealed["primary"]["effective_threshold"] = {"value": 0.24411074612857814, "binds": "F_rho"}
resealed["content_sha256"] = rc.content_digest(resealed)
attempt("a resealed lock with a replaced threshold", resealed, False, "not the candidate this run wrote")
forged_state = copy.deepcopy(state)
forged_state["lock"]["content_sha256"] = resealed["content_sha256"]
forged_state["lock"]["preregistration_sha256"] = pm.sha256_text(rr.render_preregistration(resealed))
attempt("… even with a forged state that binds it", resealed, False, "thresholds or line are not the committed record's", state=forged_state,
        preregistration_text=rr.render_preregistration(resealed))
forged_record = copy.deepcopy(record)
forged_record["null"]["null_975"] = 0.2
forged_record["effective_threshold"] = {"value": 0.24411074612857814, "binds": "F_rho"}
forged_record["content_sha256"] = rc.content_digest(forged_record)
attempt("… or with a forged record carrying the same threshold (record file digest no longer matches)", resealed, False, "", state=forged_state, record=forged_record,
        preregistration_text=rr.render_preregistration(resealed))
attempt("under a non-production configuration", lock, False, "configuration",
        config=rr.Configuration(name="x", class_quota=8, draws=10_000, null_permutations=100_000, contrast_resamples=10_000, cross_check_draws=16,
                                calibration_counts=(("determiner-like", 45), ("quantity", 45), ("adjective", 49)), n_pronoun=36, n_frames=108, n_nouns=79,
                                expected_picks=rr.PRODUCTION.expected_picks, guard=rr.PRODUCTION_GUARD))

print("=" * 100)
print("guard counts:", guard.counts)
check("no module call, forward or capture was attempted after the load", guard.counts["module_call_refused"] == 0 and guard.counts["forward_refused"] == 0 and guard.counts["capture_refused"] == 0)
print("audit events:", review_guard.report_guard())
check("no write under the repository was attempted", not review_guard.EVENTS["refused_writes"])
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
