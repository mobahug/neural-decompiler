"""Items 5-8 cross-check (run only after r2a wrote the reviewer's own values): the reviewer's reconstruction against the
cue_rotation functions and against every stored lock value; the I7' recomputation (cr.geometry_block) against the lock
bit for bit, as confirm will do it; the preregistration's cue table re-rendered from the reviewer's own values.
Weights only; every module call, forward, capture and intervention refuses."""
import guard  # noqa: F401

import hashlib
import json
import math

ROOT = guard.ROOT
HERE = guard.HERE

import numpy as np  # noqa: E402
import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

guard.seal()
torch.set_num_threads(4)
mine = json.loads((HERE / "r2a_own.json").read_text(encoding="utf-8"))
own = torch.load(HERE / "r2a_own.pt")
lock = json.loads((ROOT / "outputs/experiment-025/candidate-lock.json").read_text(encoding="utf-8"))
prereg = (ROOT / "outputs/experiment-025/candidate-preregistration.md").read_text(encoding="utf-8")
record = json.loads((ROOT / "experiments/024-readout-routing-nounness/calibration-v1.json").read_text(encoding="utf-8"))
freeze = json.loads((ROOT / "experiments/025-nounness-direction-intervention/confirmation-v1.json").read_text(encoding="utf-8"))
inputs = ul.load_frozen_inputs(ROOT)
confirmation = cr.confirmation_from_payload(freeze, inputs.pool, cr.PRODUCTION)
model = models.load_model(models.PYTHIA_70M)
guard.seal_model_forwards(model)
W_E = pm.Weights.from_model(model).W_E
del model
out: dict = {}


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


def digest_f64(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy()).astype("<f8")
    return hashlib.sha256(json.dumps(list(tensor.shape)).encode("ascii") + b"|<f8|" + array.tobytes()).hexdigest()


# My digest function against rc.tensor_digest.
out["digest_function_equal_rc"] = all(digest_f64(t) == rc.tensor_digest(t) for t in (own["d"], own["combined_controls"], own["combined32"], own["p_hat"]))

# ---- d and p_hat.
d_cr = cr.direction(W_E, record)
p_cr = cr.plurality_direction(W_E, inputs.pool)
out["d"] = {"bitwise": bool(torch.equal(d_cr, own["d"])), "maxdiff": maxdiff(d_cr, own["d"]), "cr_digest": rc.tensor_digest(d_cr),
            "lock_digest": lock["geometry"]["direction_sha256"], "mine_digest": mine["direction"]["d_sha256_mine"],
            "d_norm_equal_lock": mine["direction"]["d_norm"] == lock["geometry"]["d_norm"], "d_norm": mine["direction"]["d_norm"],
            "mu_digests_equal_lock": (mine["direction"]["mu_noun_sha256_mine"] == lock["geometry"]["mu_noun_sha256"] and mine["direction"]["mu_cue_sha256_mine"] == lock["geometry"]["mu_cue_sha256"])}
out["p_hat"] = {"bitwise": bool(torch.equal(p_cr, own["p_hat"])), "maxdiff": maxdiff(p_cr, own["p_hat"]), "lock_digest_equal_mine": mine["plurality"]["p_hat_sha256_mine"] == lock["geometry"]["plurality_sha256"],
                "cos_equal_lock": mine["plurality"]["cos_d_hat_p_hat"] == lock["geometry"]["cos_d_plurality"]}

# ---- Per cue: every cr geometry function against mine; every lock value against mine.
fields = ("norm", "s0", "tau", "theta_primary", "theta_half", "even_primary", "even_half")
digests = ("t_hat_sha256", "controls_sha256", "plurality_sha256", "vectors64_sha256", "vectors32_sha256")
worst = {f"lock_{k}": 0.0 for k in fields}
worst.update({"cr_t_hat": 0.0, "cr_controls": 0.0, "cr_p_prime": 0.0, "cr_v64": 0.0, "cr_v32": 0.0, "cr_uniforms": 0.0, "cr_gaussians": 0.0})
exact = {f"lock_{k}": 0 for k in fields}
exact.update({f"lock_{k}": 0 for k in digests})
bitwise = {"cr_t_hat": 0, "cr_controls": 0, "cr_p_prime": 0, "cr_v64": 0, "cr_v32": 0, "cr_uniforms": 0, "cr_gaussians": 0}
order_ok = [c["token_id"] for c in lock["geometry"]["cues"]] == [c["token_id"] for c in mine["cues"]] == [t["token_id"] for t in confirmation.tokens]
check_disagreement: dict = {}
for mcue, lcue, token in zip(mine["cues"], lock["geometry"]["cues"], confirmation.tokens):
    tid = int(token["token_id"])
    for k in fields:
        worst[f"lock_{k}"] = max(worst[f"lock_{k}"], abs(mcue[k] - lcue[k]))
        exact[f"lock_{k}"] += int(mcue[k] == lcue[k])
    for k in digests:
        exact[f"lock_{k}"] += int(mcue[k] == lcue[k])
    for k, v in lcue["checks"].items():
        if isinstance(v, bool):
            check_disagreement.setdefault(k, set()).add(v == mcue["checks"][k])
        else:
            check_disagreement[k] = max(check_disagreement.get(k, 0.0), abs(v - mcue["checks"][k]))
    g = cr.cue_geometry(W_E, tid, d_cr)
    entry = cr.cue_vectors(W_E, d_cr, p_cr, tid, cr.PRODUCTION)
    for name, a, b in (("cr_t_hat", g.t_hat, own[f"{tid}/t_hat"]), ("cr_controls", torch.stack(entry["controls"]), own[f"{tid}/controls"]),
                       ("cr_p_prime", entry["p_prime"], own[f"{tid}/p_prime"]),
                       ("cr_v64", torch.stack([entry["vectors64"][c] for c in cr.PRODUCTION.conditions]), own[f"{tid}/v64"]),
                       ("cr_v32", torch.stack([entry["vectors32"][c] for c in cr.PRODUCTION.conditions]), own[f"{tid}/v32"])):
        bitwise[name] += int(torch.equal(a, b))
        worst[name] = max(worst[name], maxdiff(a, b))
    # the raw stream, j = 1..7
    import r2a_stream  # noqa: E402  (the reviewer's stream functions, re-exported without side effects)
    for j in range(1, 8):
        tag = f"025|control|{tid}|{j}"
        u_cr, u_me = cr.sha_uniforms(tag, 512), r2a_stream.uniforms(tag, 512)
        g_cr, g_me = cr.sha_gaussians(tag, 512), torch.tensor(r2a_stream.gaussians(tag, 512), dtype=torch.float64)
        bitwise["cr_uniforms"] += int(u_cr == u_me)
        worst["cr_uniforms"] = max(worst["cr_uniforms"], max(abs(a - b) for a, b in zip(u_cr, u_me)))
        bitwise["cr_gaussians"] += int(torch.equal(g_cr, g_me))
        worst["cr_gaussians"] = max(worst["cr_gaussians"], maxdiff(g_cr, g_me))
    # theta_for
    assert cr.theta_for(0.32, g.tau) == mcue["theta_primary"] and cr.theta_for(0.16, g.tau) == mcue["theta_half"]
out["per_cue"] = {"order_ok": order_ok, "max_abs_disagreement": worst, "exact_equal_counts_of_40": exact, "bitwise_counts": bitwise,
                  "check_disagreement": {k: (sorted(v) if isinstance(v, set) else v) for k, v in check_disagreement.items()}}

# ---- The I7' recomputation, exactly as confirm will run it, against the lock's geometry block (bit for bit).
block = cr.geometry_block(W_E, record, inputs.pool, confirmation.tokens, cr.PRODUCTION)
i7 = cr.verify_geometry_against_lock(block["block"], lock)
combined32 = torch.stack([torch.stack([block["vectors32"][int(t["token_id"])][c] for c in cr.PRODUCTION.conditions]) for t in confirmation.tokens])
out["i7_recompute"] = {"verify_geometry_against_lock": i7, "block_equals_lock_geometry": block["block"] == lock["geometry"],
                       "vectors32_bitwise_equal_mine": bool(torch.equal(combined32, own["combined32"])), "combined32_digest_rc": rc.tensor_digest(combined32)}
nearest_cr = cr.nearest_tokens(W_E, block["vectors32"], ("base", "noun+0.32", "noun-0.32"))
near_diff = 0.0
near_flags = True
for tid, entry in lock["nearest_tokens"].items():
    for c, v in entry.items():
        near_diff = max(near_diff, abs(v["angle_to_nearest_other_deg"] - mine["nearest_mine"][tid][c]["angle_to_nearest_other_deg"]))
        near_flags &= v["nearest_is_own"] == mine["nearest_mine"][tid][c]["nearest_is_own"]
out["nearest"] = {"lock_equals_cr_now": nearest_cr == lock["nearest_tokens"], "max_angle_disagreement_mine_vs_lock_deg": near_diff, "flags_equal": near_flags,
                  "min_angle_lock": min(v["angle_to_nearest_other_deg"] for e in lock["nearest_tokens"].values() for v in e.values())}

# ---- The preregistration's cue table re-rendered from MY values.
rows_mine = [f"| {c['stratum']} | {c['word']} | {c['token_id']} | {c['s0']:+.6f} | {c['tau']:.6f} | {math.degrees(c['theta_primary']):.4f} | "
             f"{math.degrees(c['theta_half']):.4f} | {c['even_primary']:+.6f} |" for c in mine["cues"]]
rows_prereg = [line for line in prereg.splitlines() if line.startswith("| adjective |") or line.startswith("| noun |")]
out["prereg_table_equals_mine"] = rows_mine == rows_prereg
maxima_mine = {k: v for k, v in mine["check_maxima_mine"].items() if k in lock["geometry"]["check_maxima"]}
out["check_maxima_mine_vs_lock"] = {k: {"mine": maxima_mine[k], "lock": lock["geometry"]["check_maxima"][k]} for k in sorted(maxima_mine)}
out["tolerances_lock"] = lock["geometry"]["tolerances"]
out["guard"] = guard.summary()
(HERE / "r2b_crosscheck.out.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
