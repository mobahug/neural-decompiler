"""Review items 3 (digest → condition map), 5 (the patched vectors), 11 (± mapping, t̂ sign, vector lookup).

Weights only (Module.__call__, every class forward and every capture/intervention entry point refused; the module's
geometry functions refused too, so the reconstruction is my own). Rebuilds d, s0, t̂, τ, θ, the 280 SHA-256 controls,
the plurality tangent and all 840 patched vectors from the design's formulas; compares them with the lock's per-cue
scalars and digests; checks the digest → (cue, condition) map is injective; checks s(noun+) − s(noun−) = +0.64 for
every cue. Writes r3_geometry.json and vectors32_own.pt here.
"""
import guard  # noqa: F401

import hashlib
import json
import math
from pathlib import Path

import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import models
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul

ROOT = Path(guard.ROOT)
HERE = Path(guard.HERE)
EXP = ROOT / "experiments/025-nounness-direction-intervention"
guard.seal()
saved_cr = guard.refuse_functions(cr, ("direction", "centroids", "cue_geometry", "theta_for", "rotate", "sha_uniforms", "sha_gaussians", "project_off",
                                       "control_directions", "plurality_direction", "plurality_tangent", "condition_directions", "condition_vectors",
                                       "cue_vectors", "geometry_block", "geometry_checks", "score", "angle", "unit"), "cr")
out: dict = {}
freeze = json.loads((EXP / "confirmation-v1.json").read_text())
lock = json.loads((EXP / "preregistration-lock.json").read_text())
CONDITIONS = list(freeze["conditions"])
CUES = [{"word": c["word"], "token_id": int(c["token_id"]), "stratum": c["stratum"]} for c in freeze["cues"]]

inputs = ul.load_frozen_inputs(ROOT, tracked=guard.tracked)
model = models.load_model(models.PYTHIA_70M)
out["sealed_model_routes"] = len(guard.seal_model(model))
W_E = pm.Weights.from_model(model).W_E  # float32 [vocab, 512]
out["parameters_sha256_equal_lock"] = rr.parameters_digest(model) == lock["dependencies"]["model"]["parameters_sha256"]
out["embedding_sha256_equal_lock"] = rr.embedding_digest(W_E) == lock["dependencies"]["model"]["embedding_sha256"]
del model

cal_path = ROOT / "experiments/024-readout-routing-nounness/calibration-v1.json"
cal_bytes = cal_path.read_bytes()
out["calibration_024_file_sha256_ok"] = hashlib.sha256(cal_bytes).hexdigest() == "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d"
cal = json.loads(cal_bytes)


def unit(v):
    v = v.double()
    return v / torch.linalg.vector_norm(v)


# the direction: μ_noun (158 noun rows) and μ_cue (139 exposed cue rows), each a float64 mean, unit-normalized
noun_ids, cue_ids = cal["score"]["noun_row_ids"], cal["score"]["calibration_cue_ids"]
mu_noun = torch.stack([W_E[int(i)].double() for i in noun_ids]).mean(0)
mu_cue = torch.stack([W_E[int(i)].double() for i in cue_ids]).mean(0)
out["n_noun_rows"], out["n_cue_rows"] = len(noun_ids), len(cue_ids)
out["mu_digests_equal_024_record"] = rc.tensor_digest(mu_noun) == cal["score"]["mu_noun_sha256"] and rc.tensor_digest(mu_cue) == cal["score"]["mu_cue_sha256"]
d = unit(mu_noun) - unit(mu_cue)
g_lock = lock["geometry"]
out["direction_digest_equal_lock"] = rc.tensor_digest(d) == g_lock["direction_sha256"]
out["d_norm"], out["d_norm_lock"] = float(torch.linalg.vector_norm(d)), g_lock["d_norm"]
out["cos_mu"] = float(torch.dot(unit(mu_noun), unit(mu_cue)))

# the plurality direction: the scorable pool nouns' own singular and plural rows
nouns = [n for n in inputs.pool.nouns if n.single_token]
p_hat = unit(torch.stack([W_E[int(n.pl_ids[0])].double() for n in nouns]).mean(0) - torch.stack([W_E[int(n.sg_ids[0])].double() for n in nouns]).mean(0))
out["plurality_digest_equal_lock"] = rc.tensor_digest(p_hat) == g_lock["plurality_sha256"]
out["cos_d_plurality"], out["cos_d_plurality_lock"] = float(torch.dot(unit(d), p_hat)), g_lock["cos_d_plurality"]


def uniforms(tag, count):  # SHA-256 counter mode: 4 × 64-bit big-endian words per block, top 53 bits, (v + 1) / 2^53
    values, counter = [], 0
    while len(values) < count:
        block = hashlib.sha256(f"{tag}|{counter}".encode("utf-8")).digest()
        values += [((int.from_bytes(block[8 * k: 8 * k + 8], "big") >> 11) + 1) / 2.0 ** 53 for k in range(4)]
        counter += 1
    return values[:count]


def gaussians(tag, count):  # Box–Muller in float64
    u = uniforms(tag, 2 * ((count + 1) // 2))
    g = []
    for i in range(0, len(u), 2):
        r, phi = math.sqrt(-2.0 * math.log(u[i])), 2.0 * math.pi * u[i + 1]
        g += [r * math.cos(phi), r * math.sin(phi)]
    return torch.tensor(g[:count], dtype=torch.float64)


def off(v, basis):  # project off an orthonormal basis, twice
    v = v.double()
    for _ in range(2):
        for b in basis:
            v = v - torch.dot(v, b) * b
    return v


def score(v):
    return float(torch.dot(d, unit(v)))


def raw_digest(t):  # the launcher's run-tag digest: sha256 of the float32 bytes
    return hashlib.sha256(t.detach().to(torch.float32).contiguous().reshape(-1).numpy().tobytes()).hexdigest()


per_cue, vectors32_all, cmp_fail = [], {}, []
tag_map = {}
for c in CUES:
    tid = c["token_id"]
    E = W_E[tid].double()
    norm = float(torch.linalg.vector_norm(E))
    E_hat = E / norm
    s0 = float(torch.dot(d, E_hat))
    t = d - s0 * E_hat
    tau = float(torch.linalg.vector_norm(t))
    t_hat = t / tau
    th_p, th_h = math.asin(0.32 / tau), math.asin(0.16 / tau)
    controls = [unit(off(gaussians(f"025|control|{tid}|{j}", 512), (E_hat, t_hat))) for j in range(1, 8)]
    p_prime = unit(off(p_hat, (E_hat, t_hat)))
    vec64 = {}
    for cond in CONDITIONS:
        if cond == "base":
            direction, theta = t_hat, 0.0
        else:
            kind, sign = cond[:4], (1.0 if "+" in cond else -1.0)
            dose = float(cond[-4:])
            if kind == "noun":
                direction, theta = t_hat, math.asin(dose / tau)
            elif kind == "plur":
                direction, theta = p_prime, th_p
            else:
                direction, theta = controls[int(cond[4]) - 1], th_p
            direction = sign * direction
        vec64[cond] = norm * (math.cos(theta) * E_hat + math.sin(theta) * direction)
    vec32 = {k: v.to(torch.float32) for k, v in vec64.items()}
    L = next(x for x in g_lock["cues"] if x["token_id"] == tid)
    entry = {"word": c["word"], "token_id": tid, "stratum": c["stratum"], "s0": s0, "tau": tau, "theta_primary_deg": math.degrees(th_p),
             "d_dot_t_hat": float(torch.dot(d, t_hat)), "scalars_equal_lock": all(a == b for a, b in (
                 (s0, L["s0"]), (tau, L["tau"]), (norm, L["norm"]), (th_p, L["theta_primary"]), (th_h, L["theta_half"]),
                 (s0 * (math.cos(th_p) - 1.0), L["even_primary"]), (s0 * (math.cos(th_h) - 1.0), L["even_half"]))),
             "t_hat_digest_equal": rc.tensor_digest(t_hat) == L["t_hat_sha256"],
             "controls_digest_equal": rc.tensor_digest(torch.stack(controls)) == L["controls_sha256"],
             "plurality_digest_equal": rc.tensor_digest(p_prime) == L["plurality_sha256"],
             "vectors64_digest_equal": rc.tensor_digest(torch.stack([vec64[k] for k in CONDITIONS])) == L["vectors64_sha256"],
             "vectors32_digest_equal": rc.tensor_digest(torch.stack([vec32[k] for k in CONDITIONS])) == L["vectors32_sha256"],
             "stratum_equal_lock": L["stratum"] == c["stratum"] and L["word"] == c["word"],
             "base_is_model_row": bool(torch.equal(vec32["base"], W_E[tid]))}
    # the ± mapping on the actual float32 patches (item 11)
    sp, sm = score(vec32["noun+0.32"].double()), score(vec32["noun-0.32"].double())
    hp_, hm_ = score(vec32["noun+0.16"].double()), score(vec32["noun-0.16"].double())
    s_base = score(vec32["base"].double())
    entry.update({"s_plus": sp, "s_minus": sm, "s_base32": s_base, "s_plus_minus_s_minus": sp - sm, "odd_dev_primary": abs(0.5 * (sp - sm) - 0.32),
                  "odd_dev_half": abs(0.5 * (hp_ - hm_) - 0.16), "plus_raises": sp > s_base > sm,
                  "rand_neutral32_max": max(abs(score(vec32[f"rand{j}+0.32"].double()) - score(vec32[f"rand{j}-0.32"].double())) for j in range(1, 8)),
                  "plur_neutral32": abs(score(vec32["plur+0.32"].double()) - score(vec32["plur-0.32"].double())),
                  "d_dot_controls_max": max(abs(float(torch.dot(d, u))) for u in controls)})
    entry["all_equal"] = all(entry[k] for k in ("scalars_equal_lock", "t_hat_digest_equal", "controls_digest_equal", "plurality_digest_equal",
                                                "vectors64_digest_equal", "vectors32_digest_equal", "stratum_equal_lock", "base_is_model_row"))
    if not entry["all_equal"]:
        cmp_fail.append(c["word"])
    per_cue.append(entry)
    vectors32_all[tid] = torch.stack([vec32[k] for k in CONDITIONS])
    for cond in CONDITIONS:
        tag_map.setdefault(raw_digest(vec32[cond]), []).append((tid, cond))

out["per_cue"] = per_cue
out["all_cues_bitwise_equal_lock"] = not cmp_fail
out["failing"] = cmp_fail
out["tag_map"] = {"entries": sum(len(v) for v in tag_map.values()), "unique_digests": len(tag_map), "collisions": [v for v in tag_map.values() if len(v) > 1]}
spent_rows = {raw_digest(W_E[int(k.split("|")[2])]) for k in lock["patch_path_spent_keys"]}
out["tag_map"]["spent_rows_in_map"] = sorted(spent_rows & set(tag_map))
for s in ("adjective", "noun"):
    rows = [e for e in per_cue if e["stratum"] == s]
    out[f"summary_{s}"] = {"n": len(rows), "s0_min": min(e["s0"] for e in rows), "s0_max": max(e["s0"] for e in rows),
                           "s0_mean": sum(e["s0"] for e in rows) / len(rows), "s_plus_minus_s_minus_min": min(e["s_plus_minus_s_minus"] for e in rows),
                           "s_plus_minus_s_minus_max": max(e["s_plus_minus_s_minus"] for e in rows),
                           "odd_dev_primary_max": max(e["odd_dev_primary"] for e in rows), "all_plus_raises": all(e["plus_raises"] for e in rows),
                           "d_dot_t_hat_minus_tau_max": max(abs(e["d_dot_t_hat"] - e["tau"]) for e in rows), "tau_min": min(e["tau"] for e in rows),
                           "theta_primary_deg_range": [min(e["theta_primary_deg"] for e in rows), max(e["theta_primary_deg"] for e in rows)]}
out["rand_neutral32_max"] = max(e["rand_neutral32_max"] for e in per_cue)
out["d_dot_controls_max"] = max(e["d_dot_controls_max"] for e in per_cue)
torch.save({"conditions": CONDITIONS, "vectors32": vectors32_all}, HERE / "vectors32_own.pt")
out["guard"] = guard.EVENTS
(HERE / "r3_geometry.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "per_cue"}, indent=1))
