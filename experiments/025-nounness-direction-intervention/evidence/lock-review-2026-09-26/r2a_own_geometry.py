"""Items 4 (model digests), 5-9: the reviewer's OWN reconstruction of d, the 40 cue geometries, the rotations, the 280
SHA-256 counter-mode controls, the plurality control and the I7' checks -- written from the design, with every
cue_rotation geometry function refused (guard.refuse_cr_geometry) so that none can be used. Weights only: the model is
loaded once to read its parameters; every module call, forward, capture and intervention refuses.

Writes r2a_own.pt (tensors) and r2a_own.json (scalars, digests, checks). Nothing is compared with cr here; the lock is
read only at the very end for a first comparison of digests and scalars (the full cross-check is r2b)."""
import guard  # noqa: F401

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = guard.ROOT
HERE = guard.HERE

import mpmath  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402  (constants only; its geometry functions refuse)
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

guard.seal()
guard.refuse_cr_geometry()
torch.set_num_threads(4)
out: dict = {"torch_threads": torch.get_num_threads()}

# ---------------------------------------------------------------------------------------------------------------------
# Digests, written independently of rc.tensor_digest / rr.embedding_digest / rr.parameters_digest.


def digest_f64(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy()).astype("<f8")
    return hashlib.sha256(json.dumps(list(tensor.shape)).encode("ascii") + b"|<f8|" + array.tobytes()).hexdigest()


def digest_f4(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy()).astype("<f4")
    return hashlib.sha256(json.dumps(list(tensor.shape)).encode("ascii") + b"|<f4|" + array.tobytes()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


# ---------------------------------------------------------------------------------------------------------------------
# Inputs: 024's committed calibration record, 025's committed freeze, the pinned checkpoint.
record = json.loads((ROOT / "experiments/024-readout-routing-nounness/calibration-v1.json").read_text(encoding="utf-8"))
freeze = json.loads((ROOT / "experiments/025-nounness-direction-intervention/confirmation-v1.json").read_text(encoding="utf-8"))
noun_ids = [int(i) for i in record["score"]["noun_row_ids"]]
cue_ids = [int(i) for i in record["score"]["calibration_cue_ids"]]
cues = [(c["word"], int(c["token_id"]), c["stratum"]) for c in freeze["cues"]]
out["inputs"] = {"noun_row_ids": len(noun_ids), "calibration_cue_ids": len(cue_ids), "cues": len(cues),
                 "noun_row_ids_sha256_mine": hashlib.sha256(canonical(noun_ids).encode()).hexdigest(), "noun_row_ids_sha256_recorded": record["score"]["noun_row_ids_sha256"],
                 "cue_ids_sha256_mine": hashlib.sha256(canonical(cue_ids).encode()).hexdigest(), "cue_ids_sha256_recorded": record["score"]["calibration_cue_ids_sha256"]}

snapshot = Path.home() / ".cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
with safe_open(str(snapshot / "model.safetensors"), framework="pt") as handle:
    raw_names = list(handle.keys())
    W_raw = handle.get_tensor("gpt_neox.embed_in.weight").clone()
model = models.load_model(models.PYTHIA_70M)
sealed = guard.seal_model_forwards(model)
weights = pm.Weights.from_model(model)
W32 = weights.W_E
out["model"] = {"sealed_modules": sealed, "resolved_revision": models.resolved_revision(model), "W_E_shape": list(W32.shape), "W_E_dtype": str(W32.dtype),
                "raw_embed_equals_model_W_E_bitwise": bool(torch.equal(W_raw.to(torch.float32), W32)), "raw_dtype": str(W_raw.dtype)}
emb_mine = digest_f4(W32)
param_digest = hashlib.sha256()
named = sorted(((name, p) for name, p in model.named_parameters()), key=lambda item: item[0])
for name, parameter in named:
    array = np.ascontiguousarray(parameter.detach().cpu().to(torch.float32).numpy()).astype("<f4")
    param_digest.update(name.encode("utf-8") + b"|" + json.dumps(list(parameter.shape)).encode("ascii") + b"|" + array.tobytes())
out["model"]["embedding_sha256_mine"] = emb_mine
out["model"]["embedding_sha256_raw_file"] = digest_f4(W_raw.to(torch.float32))
out["model"]["parameters_sha256_mine"] = param_digest.hexdigest()
out["model"]["n_named_parameters"] = len(named)
out["model"]["embedding_sha256_rr"] = rr.embedding_digest(W32)
out["model"]["parameters_sha256_rr"] = rr.parameters_digest(model)
del model

W64 = W32.double()

# ---------------------------------------------------------------------------------------------------------------------
# Item 5: d = unit(mean noun rows) - unit(mean calibration-cue rows), float64 means of the float32 rows.
mu_noun = W64[noun_ids].mean(dim=0)
mu_cue = W64[cue_ids].mean(dim=0)
mu_noun_hat = mu_noun / torch.linalg.vector_norm(mu_noun)
mu_cue_hat = mu_cue / torch.linalg.vector_norm(mu_cue)
d = mu_noun_hat - mu_cue_hat
out["direction"] = {"mu_noun_sha256_mine": digest_f64(mu_noun), "mu_noun_sha256_recorded": record["score"]["mu_noun_sha256"],
                    "mu_cue_sha256_mine": digest_f64(mu_cue), "mu_cue_sha256_recorded": record["score"]["mu_cue_sha256"],
                    "d_sha256_mine": digest_f64(d), "d_norm": float(torch.linalg.vector_norm(d)), "cos_mu_noun_mu_cue": float(torch.dot(mu_noun_hat, mu_cue_hat))}

# High-precision reference (mpmath, 40 digits): exact means of the float32 rows, exact unit vectors.
mpmath.mp.dps = 40
MP = mpmath.mpf


def mp_vec(t: torch.Tensor):
    return [MP(float(x)) for x in t.tolist()]


def mp_mean(rows):
    n = len(rows)
    return [mpmath.fsum(col) / n for col in zip(*rows)]


def mp_norm(v):
    return mpmath.sqrt(mpmath.fsum(x * x for x in v))


def mp_dot(a, b):
    return mpmath.fsum(x * y for x, y in zip(a, b))


W_rows = {i: mp_vec(W64[i]) for i in set(noun_ids) | set(cue_ids)}
hp_mu_noun = mp_mean([W_rows[i] for i in noun_ids])
hp_mu_cue = mp_mean([W_rows[i] for i in cue_ids])
nn_, nc_ = mp_norm(hp_mu_noun), mp_norm(hp_mu_cue)
hp_d = [a / nn_ - b / nc_ for a, b in zip(hp_mu_noun, hp_mu_cue)]
out["direction"]["hp_max_abs_err_mu_noun"] = float(max(abs(MP(x) - y) for x, y in zip(mu_noun.tolist(), hp_mu_noun)))
out["direction"]["hp_max_abs_err_mu_cue"] = float(max(abs(MP(x) - y) for x, y in zip(mu_cue.tolist(), hp_mu_cue)))
out["direction"]["hp_max_abs_err_d"] = float(max(abs(MP(x) - y) for x, y in zip(d.tolist(), hp_d)))
out["direction"]["hp_d_norm"] = float(mp_norm(hp_d))

# ---------------------------------------------------------------------------------------------------------------------
# The plurality direction: the 79 scorable nouns' plural minus singular rows. The record's noun rows interleave
# singular and plural (rr.noun_row_ids); the pairing is also checked against the pool below.
sg_ids, pl_ids = noun_ids[0::2], noun_ids[1::2]
inputs = ul.load_frozen_inputs(ROOT)
pool_pairs = [(int(n.sg_ids[0]), int(n.pl_ids[0])) for n in inputs.pool.nouns if n.single_token]
p_raw = W64[pl_ids].mean(dim=0) - W64[sg_ids].mean(dim=0)
p_hat = p_raw / torch.linalg.vector_norm(p_raw)
out["plurality"] = {"pairs": len(sg_ids), "pairing_equals_pool": pool_pairs == list(zip(sg_ids, pl_ids)), "p_hat_sha256_mine": digest_f64(p_hat),
                    "cos_d_hat_p_hat": float(torch.dot(d / torch.linalg.vector_norm(d), p_hat))}

# ---------------------------------------------------------------------------------------------------------------------
# Item 8/9: the frozen control stream, from the design text (no library RNG anywhere).


def uniforms(tag: str, n: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < n:
        digest = hashlib.sha256(f"{tag}|{counter}".encode("utf-8")).digest()
        for k in range(4):
            word = int.from_bytes(digest[8 * k:8 * k + 8], byteorder="big", signed=False) >> 11
            values.append((word + 1) / 2 ** 53)
        counter += 1
    return values[:n]


def gaussians(tag: str, n: int) -> list[float]:
    u = uniforms(tag, n + (n % 2))
    z: list[float] = []
    for i in range(0, len(u), 2):
        r = math.sqrt(-2.0 * math.log(u[i]))
        phi = 2.0 * math.pi * u[i + 1]
        z.append(r * math.cos(phi))
        z.append(r * math.sin(phi))
    return z[:n]


def orthogonalize(v: torch.Tensor, basis) -> torch.Tensor:
    for _ in range(2):
        for b in basis:
            v = v - torch.dot(v, b) * b
    return v


CONDITIONS = ["base", "noun+0.32", "noun-0.32", "noun+0.16", "noun-0.16"] + [f"rand{j}{s}0.32" for j in range(1, 8) for s in "+-"] + ["plur+0.32", "plur-0.32"]


def angle_between(a: torch.Tensor, b: torch.Tensor) -> float:
    """atan2(|b_perp|, a.b) with unit vectors: a formula different from cr.angle's 2*atan2(|a-b|, |a+b|)."""
    ua, ub = a / torch.linalg.vector_norm(a), b / torch.linalg.vector_norm(b)
    c = float(torch.dot(ua, ub))
    perp = ub - c * ua
    return math.atan2(float(torch.linalg.vector_norm(perp)), c)


def s_of(v: torch.Tensor) -> float:
    return float(torch.dot(d, v / torch.linalg.vector_norm(v)))


geo: list[dict] = []
tensors: dict[str, torch.Tensor] = {"d": d, "mu_noun": mu_noun, "mu_cue": mu_cue, "p_hat": p_hat}
all32, all_controls = [], []
hp_worst = {"s0": 0.0, "tau": 0.0, "theta_primary": 0.0, "t_hat": 0.0, "vec64": 0.0}
for index, (word, token_id, stratum) in enumerate(cues):
    E = W64[token_id]
    norm = float(torch.linalg.vector_norm(E))
    E_hat = E / norm
    s0 = float(torch.dot(d, E_hat))
    t = d - s0 * E_hat
    tau = float(torch.linalg.vector_norm(t))
    t_hat = t / tau
    theta_p = math.asin(0.32 / tau)
    theta_h = math.asin(0.16 / tau)
    controls = []
    for j in range(1, 8):
        g = torch.tensor(gaussians(f"025|control|{token_id}|{j}", 512), dtype=torch.float64)
        u = orthogonalize(g, (E_hat, t_hat))
        controls.append(u / torch.linalg.vector_norm(u))
    p_prime = orthogonalize(p_hat.clone(), (E_hat, t_hat))
    p_prime = p_prime / torch.linalg.vector_norm(p_prime)
    directions = {"base": (t_hat, 0.0)}
    for sign, sym in ((1, "+"), (-1, "-")):
        directions[f"noun{sym}0.32"] = (sign * t_hat, theta_p)
        directions[f"noun{sym}0.16"] = (sign * t_hat, theta_h)
        for j in range(1, 8):
            directions[f"rand{j}{sym}0.32"] = (sign * controls[j - 1], theta_p)
        directions[f"plur{sym}0.32"] = (sign * p_prime, theta_p)
    v64 = {c: norm * (math.cos(directions[c][1]) * E_hat + math.sin(directions[c][1]) * directions[c][0]) for c in CONDITIONS}
    v32 = {c: v64[c].to(torch.float32) for c in CONDITIONS}
    stack64 = torch.stack([v64[c] for c in CONDITIONS])
    stack32 = torch.stack([v32[c] for c in CONDITIONS])
    ctrl = torch.stack(controls)
    all32.append(stack32)
    all_controls.append(ctrl)
    tensors[f"{token_id}/t_hat"] = t_hat
    tensors[f"{token_id}/controls"] = ctrl
    tensors[f"{token_id}/p_prime"] = p_prime
    tensors[f"{token_id}/v64"] = stack64
    tensors[f"{token_id}/v32"] = stack32

    # ---- I7' checks, computed my way (float64 and on the float32 casts).
    E32 = W32[token_id]
    rows = {}
    rows["unit_t"] = abs(float(torch.linalg.vector_norm(t_hat)) - 1.0)
    rows["unit_controls"] = max(abs(float(torch.linalg.vector_norm(u)) - 1.0) for u in controls)
    rows["unit_plurality"] = abs(float(torch.linalg.vector_norm(p_prime)) - 1.0)
    rows["orth_E_t"] = abs(float(torch.dot(E_hat, t_hat)))
    rows["orth_E_controls"] = max(abs(float(torch.dot(E_hat, u))) for u in controls)
    rows["orth_t_controls"] = max(abs(float(torch.dot(t_hat, u))) for u in controls)
    rows["orth_E_plurality"] = abs(float(torch.dot(E_hat, p_prime)))
    rows["orth_t_plurality"] = abs(float(torch.dot(t_hat, p_prime)))
    rows["d_dot_controls"] = max(abs(float(torch.dot(d, u))) for u in controls)
    rows["d_dot_plurality"] = abs(float(torch.dot(d, p_prime)))
    neutral_pairs = [(f"rand{j}+0.32", f"rand{j}-0.32") for j in range(1, 8)] + [("plur+0.32", "plur-0.32")]
    rows["neutral64"] = max(abs(s_of(v64[a]) - s_of(v64[b])) for a, b in neutral_pairs)
    rows["neutral32"] = max(abs(s_of(v32[a].double()) - s_of(v32[b].double())) for a, b in neutral_pairs)
    rows["neutral32_controls_only"] = max(abs(s_of(v32[a].double()) - s_of(v32[b].double())) for a, b in neutral_pairs[:7])
    rows["odd64"] = max(abs(0.5 * (s_of(v64[f"noun+{x}"]) - s_of(v64[f"noun-{x}"])) - float(x)) for x in ("0.32", "0.16"))
    rows["odd32"] = max(abs(0.5 * (s_of(v32[f"noun+{x}"].double()) - s_of(v32[f"noun-{x}"].double())) - float(x)) for x in ("0.32", "0.16"))
    rows["length64"] = max(abs(float(torch.linalg.vector_norm(v)) / norm - 1.0) for v in v64.values())
    rows["length32"] = max(abs(float(torch.linalg.vector_norm(v.double())) / norm - 1.0) for v in v32.values())
    rows["angle64"] = max(abs(angle_between(E, v64[c]) - directions[c][1]) for c in CONDITIONS if c != "base")
    rows["angle32"] = max(abs(angle_between(E, v32[c].double()) - directions[c][1]) for c in CONDITIONS if c != "base")
    rows["base_equals_model_row"] = bool(torch.equal(v32["base"], E32))
    # The complete change s0(cos th - 1) +- odd, and the controls' score (the same even term only).
    change = []
    for x, th in (("0.32", theta_p), ("0.16", theta_h)):
        for sign, sym in ((1, "+"), (-1, "-")):
            expected = s0 * (math.cos(th) - 1.0) + sign * float(x)
            change.append(abs((s_of(v64[f"noun{sym}{x}"]) - s0) - expected))
            change.append(abs((s_of(v32[f"noun{sym}{x}"].double()) - s0) - expected))
    rows["complete_change_max"] = max(change)
    rows["controls_score_minus_s0cos_max32"] = max(abs(s_of(v32[f"rand{j}{sym}0.32"].double()) - s0 * math.cos(theta_p)) for j in range(1, 8) for sym in "+-")
    rows["plur_score_minus_s0cos_max32"] = max(abs(s_of(v32[f"plur{sym}0.32"].double()) - s0 * math.cos(theta_p)) for sym in "+-")
    # The float32-patched pair's actual odd displacement u_eff = (v32(+) - v32(-)) / (2|E| sin th): its components along
    # d, E_hat and t_hat (the direction the patched runs really move along).
    eff = []
    for j in range(1, 8):
        u_eff = (v32[f"rand{j}+0.32"].double() - v32[f"rand{j}-0.32"].double()) / (2.0 * norm * math.sin(theta_p))
        eff.append((abs(float(torch.dot(d, u_eff))), abs(float(torch.dot(E_hat, u_eff))), abs(float(torch.dot(t_hat, u_eff))),
                    abs(float(torch.linalg.vector_norm(u_eff)) - 1.0), float(torch.dot(u_eff, controls[j - 1]))))
    rows["u_eff32_d_dot_max"] = max(e[0] for e in eff)
    rows["u_eff32_E_dot_max"] = max(e[1] for e in eff)
    rows["u_eff32_t_dot_max"] = max(e[2] for e in eff)
    rows["u_eff32_unit_dev_max"] = max(e[3] for e in eff)
    rows["u_eff32_cos_with_u_min"] = min(e[4] for e in eff)
    rows["angle32_controls_vs_noun_max"] = max(abs(angle_between(E, v32[f"rand{j}{sym}0.32"].double()) - angle_between(E, v32[f"noun{sym}0.32"].double()))
                                               for j in range(1, 8) for sym in "+-")
    gram = ctrl @ ctrl.T
    rows["controls_max_offdiag_cos"] = float((gram - torch.eye(7, dtype=torch.float64)).abs().max())
    rows["controls_abs_cos_with_p_prime_max"] = float((ctrl @ p_prime).abs().max())
    # Randomness sanity of the raw Gaussians (one per cue: j = 1): mean and variance.
    g1 = torch.tensor(gaussians(f"025|control|{token_id}|1", 512), dtype=torch.float64)
    rows["gauss1_mean"] = float(g1.mean())
    rows["gauss1_var"] = float(g1.var(unbiased=False))

    # ---- High-precision reference for this cue (mpmath): s0, tau, theta, t_hat and the float64 vectors.
    hE = mp_vec(E)
    hn = mp_norm(hE)
    hEh = [x / hn for x in hE]
    hs0 = mp_dot(hp_d, hEh)
    ht = [a - hs0 * b for a, b in zip(hp_d, hEh)]
    htau = mp_norm(ht)
    hth = [x / htau for x in ht]
    htheta = mpmath.asin(MP("0.32") / htau)
    hp_worst["s0"] = max(hp_worst["s0"], float(abs(MP(s0) - hs0)))
    hp_worst["tau"] = max(hp_worst["tau"], float(abs(MP(tau) - htau)))
    hp_worst["theta_primary"] = max(hp_worst["theta_primary"], float(abs(MP(theta_p) - htheta)))
    hp_worst["t_hat"] = max(hp_worst["t_hat"], float(max(abs(MP(x) - y) for x, y in zip(t_hat.tolist(), hth))))
    hv = [hn * (mpmath.cos(htheta) * a + mpmath.sin(htheta) * b) for a, b in zip(hEh, hth)]  # noun+0.32, exact
    hp_worst["vec64"] = max(hp_worst["vec64"], float(max(abs(MP(x) - y) for x, y in zip(v64["noun+0.32"].tolist(), hv))))

    geo.append({"word": word, "token_id": token_id, "stratum": stratum, "norm": norm, "s0": s0, "tau": tau, "theta_primary": theta_p, "theta_half": theta_h,
                "theta_primary_deg": math.degrees(theta_p), "theta_half_deg": math.degrees(theta_h), "even_primary": s0 * (math.cos(theta_p) - 1.0),
                "even_half": s0 * (math.cos(theta_h) - 1.0), "arcsine_args": [0.32 / tau, 0.16 / tau], "t_hat_sha256": digest_f64(t_hat),
                "controls_sha256": digest_f64(ctrl), "plurality_sha256": digest_f64(p_prime), "vectors64_sha256": digest_f64(stack64),
                "vectors32_sha256": digest_f64(stack32), "checks": rows})

combined32 = torch.stack(all32)  # [40, 21, 512] float32
combined_controls = torch.stack(all_controls)  # [40, 7, 512] float64
tensors["combined32"] = combined32
tensors["combined_controls"] = combined_controls
out["combined"] = {"vectors32_40x21x512_sha256": digest_f64(combined32), "vectors32_840x512_sha256": digest_f64(combined32.reshape(840, 512)),
                   "controls_40x7x512_sha256": digest_f64(combined_controls)}
out["hp_worst"] = hp_worst
out["cues"] = geo
names = [k for k in geo[0]["checks"] if not isinstance(geo[0]["checks"][k], bool)]
out["check_maxima_mine"] = {k: max(float(g["checks"][k]) for g in geo) for k in names if not k.startswith(("gauss1", "u_eff32_cos", "controls_max"))}
out["check_minima_mine"] = {"u_eff32_cos_with_u_min": min(g["checks"]["u_eff32_cos_with_u_min"] for g in geo)}
out["controls_max_offdiag_cos_max"] = max(g["checks"]["controls_max_offdiag_cos"] for g in geo)
out["gauss_mean_range"] = [min(g["checks"]["gauss1_mean"] for g in geo), max(g["checks"]["gauss1_mean"] for g in geo)]
out["gauss_var_range"] = [min(g["checks"]["gauss1_var"] for g in geo), max(g["checks"]["gauss1_var"] for g in geo)]
out["base_equals_model_row_all"] = all(g["checks"]["base_equals_model_row"] for g in geo)
out["finite_all"] = all(math.isfinite(v) for g in geo for v in (g["norm"], g["s0"], g["tau"], g["theta_primary"], g["theta_half"])) and bool(
    torch.isfinite(combined32).all()) and bool(torch.isfinite(combined_controls).all())
out["arcsine_args_valid"] = all(0.0 < a < 1.0 for g in geo for a in g["arcsine_args"])
out["ranges"] = {key: [min(g[key] for g in geo), max(g[key] for g in geo)] for key in ("norm", "s0", "tau", "theta_primary_deg", "theta_half_deg", "even_primary", "even_half")}

# Nearest tokens (descriptive, as the lock records them): base and noun+-0.32.
normalized = W64 / torch.linalg.vector_norm(W64, dim=1, keepdim=True)
nearest = {}
for g in geo:
    token_id = g["token_id"]
    entry = {}
    for c in ("base", "noun+0.32", "noun-0.32"):
        v = tensors[f"{token_id}/v32"][CONDITIONS.index(c)].double()
        sims = normalized @ (v / torch.linalg.vector_norm(v))
        order = torch.argsort(sims, descending=True)
        own_first = int(order[0]) == token_id
        other = int(order[1]) if own_first else int(order[0])
        entry[c] = {"nearest_is_own": own_first, "angle_to_nearest_other_deg": math.degrees(math.acos(max(-1.0, min(1.0, float(sims[other])))))}
    nearest[str(token_id)] = entry
out["nearest_mine"] = nearest
out["nearest_min_angle_deg"] = min(e[c]["angle_to_nearest_other_deg"] for e in nearest.values() for c in e)
out["nearest_all_own"] = all(e[c]["nearest_is_own"] for e in nearest.values() for c in e)

torch.save(tensors, HERE / "r2a_own.pt")
out["guard"] = guard.summary()
(HERE / "r2a_own.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("cues", "nearest_mine")}, indent=1))
