"""Experiment 025: read-only verification of the production candidate lock, weights only.

Guards: torch module calls, direct `.forward()` on the loaded model, every capture/intervention entry point and every
confirm-only computation refuse. No prompt runs and nothing in the repository is written.

The verification covers:
- the candidate files' digests, the rendering and the results state;
- the geometry recomputed in this separate process and compared with the lock bit for bit;
- the neutrality of the 280 random controls, checked separately on their actual float32 patched vectors;
- the nounness doses and angles, and the complete score change;
- the bindings: the freeze, the manifest and its collisions, 024 and 020, the model, the statistics, the outcome
  and the tolerances.

It also computes the combined digests of all cue geometries, all 280 controls and all 840 patched vectors.
"""
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from neural_decompiler import capture, interventions, models
from neural_decompiler import cue_rotation as cr
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
OUT = ROOT / "outputs/experiment-025"
refused: list[str] = []
loads = {"count": 0}


def refusal(name):
    def refuse(*args, **kwargs):
        refused.append(name)
        raise RuntimeError(f"{name} refused in the lock verification")

    return refuse


torch.nn.Module.__call__ = refusal("torch.nn.Module.__call__")
for module, names in ((pm, ("capture_prompt", "run_patched", "run_capture", "run_interventions")), (capture, ("run_capture",)), (interventions, ("run_interventions",)),
                      (cr, ("measure_rotated", "stage_two", "patch_path_check", "identity_gates", "level1_gates", "run_gates", "recompute_c", "routing_distance", "per_cue",
                            "statistics", "descriptives", "ladder", "vector_factors")), (ul, ("measure_prompt", "contrast_of", "compose_dx3")), (rr, ("cue_mse", "fresh_pair_cells"))):
    for name in names:
        setattr(module, name, refusal(f"{module.__name__.rsplit('.', 1)[1]}.{name}"))
rd.ReadoutProgram.level1_detail = refusal("ReadoutProgram.level1_detail")
original_load = models.load_model


def counted_load(spec, *args, **kwargs):
    loads["count"] += 1
    model = original_load(spec, *args, **kwargs)
    for name, module in model.named_modules():
        object.__setattr__(module, "forward", refusal(f"{name or 'model'}.forward"))
    return model


models.load_model = counted_load


def git(*args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


spec = importlib.util.spec_from_file_location("experiment_025_runner", ROOT / f"{cr.EXPERIMENT_DIR}/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
logs: list[str] = []
runner = runner_module.Runner(log=logs.append)

lock_bytes = (OUT / "candidate-lock.json").read_bytes()
prereg_text = (OUT / "candidate-preregistration.md").read_text(encoding="utf-8")
lock = json.loads(lock_bytes)
state = rd.load_results_state(OUT / "results.json")
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
checks: dict = {}

# 1. The candidate files and the results state.
checks["lock_content_digest"] = lock["content_sha256"] == rc.content_digest(lock)
checks["preregistration_renders_from_the_lock_file"] = prereg_text == cr.render_preregistration(lock)
checks["state_binds_lock_and_preregistration"] = state["lock"]["content_sha256"] == lock["content_sha256"] and state["lock"]["preregistration_sha256"] == pm.sha256_text(prereg_text)
checks["state_phase_lock_complete_others_not_started"] = (state["phases"]["lock"]["status"], state["phases"]["confirm"]["status"], state["phases"]["report"]["status"]) \
    == ("complete", "not_started", "not_started") and not state["phases"]["lock"].get("incidents")
checks["state_ledger_empty_no_confirmation"] = state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [] and state["confirmation"] is None and state["report"] is None
checks["state_commit_and_config"] = state["protocol_code_commit"] == lock["protocol_code_commit"] == state["phases"]["lock"]["commit"] == git("rev-parse", "HEAD") \
    and state["configuration"] == lock["configuration"] == cr.PRODUCTION.to_json() and state["git_dirty"] is False
checks["state_inputs_are_the_base_digests"] = state["inputs"] == {key: base.digests[key] for key in cr.DIGEST_KEYS} == lock["inputs"]
try:
    cr.assert_phase_allowed("lock", state)
    checks["a_second_lock_is_refused"] = False
except cr.PhaseError as error:
    checks["a_second_lock_is_refused"] = "lock already written" in str(error)

# 2. The bindings.
checks["module_and_blobs"] = lock["module"] == {"path": "src/neural_decompiler/cue_rotation.py", "blob": cr.own_blob()} and lock["module_blobs"] == cr.FROZEN_BLOBS == cr.assert_frozen_blobs()
checks["design_and_plan"] = lock["design"] == cr.DESIGN and lock["plan"] == cr.PLAN
checks["inherited_024"] = lock["inherited_024"] == cr.binding_024()
checks["freeze_binding"] = lock["confirmation_025"] == cr.confirmation_binding(confirmation, confirmation_sha) and lock["confirmation_025"]["file_sha256"] == \
    "54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a" and lock["confirmation_025"]["content_sha256"] == "6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4" \
    and lock["confirmation_025"]["manifest_sha256"] == "0e1f068b4fe792ea3d9ea23221b17bc11f6c992b14036a5b6e5ead77f24aba61" and lock["confirmation_025"]["counts"]["runs"] == 90720
tagged = confirmation.tagged_keys()
untagged = confirmation.manifest_keys()
checks["manifest_40_108_21_90720"] = (len(confirmation.tokens), len(confirmation.exposed_frames), len(confirmation.conditions), len(tagged), len(untagged)) == (40, 108, 21, 90720, 4320)
checks["zero_spent_collisions"] = len(base.forbidden) == 43632 and not (tagged | untagged) & base.forbidden
checks["zero_patch_path_collisions"] = not (tagged | untagged) & set(cr.PATCH_PATH_SPENT_KEYS) and set(cr.PATCH_PATH_SPENT_KEYS) <= base.forbidden \
    and lock["patch_path_spent_keys"] == list(cr.PATCH_PATH_SPENT_KEYS)
checks["noun_keys_79_in_order"] = lock["noun_keys"] == runner._noun_keys(base.inputs) and len(lock["noun_keys"]) == 79
checks["statistics_outcome_semantics_tolerances"] = lock["semantics"] == cr.SEMANTICS and lock["outcome"]["table"] == [list(row) for row in cr.OUTCOME_TABLE] \
    and lock["outcome"]["labels"] == list(cr.OUTCOMES) and lock["tolerances"] == cr.TOLERANCES
config = lock["configuration"]
checks["threshold_27_tail_and_conditions"] = config["count_threshold"] == 27 and config["reference_tail"] == {"exact": f"{cr.binomial_tail(40, 27).numerator}/{cr.binomial_tail(40, 27).denominator}",
                                                                                                            "value": 0.01923865414210013} \
    and len(config["conditions"]) == 21 and config["outcome_bearing"] == ["noun+0.32", "noun-0.32", *[f"rand{j}{s}0.32" for j in range(1, 8) for s in "+-"]]
checks["readout_020_states_bound"] = lock["dependencies"]["readout_020"]["exposed_states_sha256"] == ul.exposed_states_digest(base.inputs.closure["exploration"]["locked_states"])
lock_024 = base.inherited["lock"]
checks["model_digests_equal_024s_lock"] = {k: lock["dependencies"]["model"][k] for k in ("model_id", "revision", "parameters_sha256", "embedding_sha256")} \
    == {k: lock_024["dependencies"]["model"][k] for k in ("model_id", "revision", "parameters_sha256", "embedding_sha256")}

# 3. The geometry recomputed from the weights in this process, bit for bit, plus the checks on the actual vectors.
model = models.load_model(models.PYTHIA_70M)
progs = ul.ModelPrograms.from_model(model, base.inputs)
W_E = progs.weights.W_E
checks["model_digests_recompute"] = rr.parameters_digest(model) == lock["dependencies"]["model"]["parameters_sha256"] and rr.embedding_digest(W_E) == lock["dependencies"]["model"]["embedding_sha256"]
recomputed = cr.geometry_block(W_E, base.inherited["calibration"], base.inputs.pool, confirmation.tokens, cr.PRODUCTION)
checks["geometry_recomputes_bit_for_bit"] = cr.verify_geometry_against_lock(recomputed["block"], lock) == {"bitwise_equal": True, "differing": []}
record = base.inherited["calibration"]
d = cr.direction(W_E, record)
p_hat = cr.plurality_direction(W_E, base.inputs.pool)
checks["direction_digests"] = rc.tensor_digest(d) == lock["geometry"]["direction_sha256"] and lock["geometry"]["mu_noun_sha256"] == record["score"]["mu_noun_sha256"] \
    and lock["geometry"]["mu_cue_sha256"] == record["score"]["mu_cue_sha256"]
stats = {"random": {"unit64": 0.0, "E_dot_u": 0.0, "t_dot_u": 0.0, "d_dot_u": 0.0, "neutral64": 0.0, "neutral32": 0.0, "angle64": 0.0, "angle32": 0.0, "length32": 0.0},
         "plurality": {"neutral64": 0.0, "neutral32": 0.0, "d_dot": 0.0}, "noun": {"odd64_primary": 0.0, "odd32_primary": 0.0, "odd64_half": 0.0, "odd32_half": 0.0,
                                                                                    "angle32": 0.0, "complete_change64": 0.0, "length32": 0.0},
         "base_equals_model_row": True}
all_controls, all_vectors32, per_cue = [], [], []
for token, cue in zip(confirmation.tokens, lock["geometry"]["cues"]):
    token_id = int(token["token_id"])
    entry = cr.cue_vectors(W_E, d, p_hat, token_id, cr.PRODUCTION)
    g, v64, v32 = entry["geometry"], entry["vectors64"], entry["vectors32"]
    theta, theta_half = cr.theta_for(0.32, g.tau), cr.theta_for(0.16, g.tau)
    assert cue["token_id"] == token_id and cue["word"] == token["word"] and math.isfinite(g.tau) and 0.32 < g.tau and cue["theta_primary"] == theta
    for j, u in enumerate(entry["controls"], start=1):
        s = stats["random"]
        s["unit64"] = max(s["unit64"], abs(float(torch.linalg.vector_norm(u)) - 1.0))
        s["E_dot_u"] = max(s["E_dot_u"], abs(float(g.E_hat @ u)))
        s["t_dot_u"] = max(s["t_dot_u"], abs(float(g.t_hat @ u)))
        s["d_dot_u"] = max(s["d_dot_u"], abs(float(d @ u)))
        plus, minus = f"rand{j}+0.32", f"rand{j}-0.32"
        s["neutral64"] = max(s["neutral64"], abs(cr.score(d, v64[plus]) - cr.score(d, v64[minus])))
        s["neutral32"] = max(s["neutral32"], abs(cr.score(d, v32[plus].double()) - cr.score(d, v32[minus].double())))
        for c in (plus, minus):
            s["angle64"] = max(s["angle64"], abs(cr.angle(g.E, v64[c]) - theta))
            s["angle32"] = max(s["angle32"], abs(cr.angle(g.E, v32[c].double()) - theta))
            s["length32"] = max(s["length32"], abs(float(torch.linalg.vector_norm(v32[c].double())) / g.norm - 1.0))
    s = stats["plurality"]
    s["neutral64"] = max(s["neutral64"], abs(cr.score(d, v64["plur+0.32"]) - cr.score(d, v64["plur-0.32"])))
    s["neutral32"] = max(s["neutral32"], abs(cr.score(d, v32["plur+0.32"].double()) - cr.score(d, v32["plur-0.32"].double())))
    s["d_dot"] = max(s["d_dot"], abs(float(d @ entry["p_prime"])))
    s = stats["noun"]
    for dose, th, key in ((0.32, theta, "primary"), (0.16, theta_half, "half")):
        p, m = f"noun+{dose:.2f}", f"noun-{dose:.2f}"
        s[f"odd64_{key}"] = max(s[f"odd64_{key}"], abs(0.5 * (cr.score(d, v64[p]) - cr.score(d, v64[m])) - dose))
        s[f"odd32_{key}"] = max(s[f"odd32_{key}"], abs(0.5 * (cr.score(d, v32[p].double()) - cr.score(d, v32[m].double())) - dose))
        for c, sign in ((p, 1), (m, -1)):
            s["angle32"] = max(s["angle32"], abs(cr.angle(g.E, v32[c].double()) - th))
            s["complete_change64"] = max(s["complete_change64"], abs((cr.score(d, v64[c]) - g.s0) - (g.s0 * (math.cos(th) - 1.0) + sign * dose)))
            s["length32"] = max(s["length32"], abs(float(torch.linalg.vector_norm(v32[c].double())) / g.norm - 1.0))
    stats["base_equals_model_row"] = stats["base_equals_model_row"] and bool(torch.equal(v32["base"], W_E[token_id]))
    all_controls.append(torch.stack(entry["controls"]))
    all_vectors32.append(torch.stack([v32[c] for c in cr.PRODUCTION.conditions]))
    per_cue.append({"word": cue["word"], "token_id": cue["token_id"], "stratum": cue["stratum"], "s0": cue["s0"], "tau": cue["tau"],
                    "theta_primary_deg": math.degrees(cue["theta_primary"]), "theta_half_deg": math.degrees(cue["theta_half"]), "even_primary": cue["even_primary"],
                    "t_hat_sha256": cue["t_hat_sha256"], "controls_sha256": cue["controls_sha256"], "plurality_sha256": cue["plurality_sha256"],
                    "vectors64_sha256": cue["vectors64_sha256"], "vectors32_sha256": cue["vectors32_sha256"]})
del model
limits = {"random": {"unit64": 1e-12, "E_dot_u": 1e-12, "t_dot_u": 1e-12, "d_dot_u": 1e-12, "neutral64": 1e-12, "neutral32": 1e-6, "angle64": 1e-12, "angle32": 1e-6, "length32": 1e-6},
          "plurality": {"neutral64": 1e-12, "neutral32": 1e-6, "d_dot": 1e-12},
          "noun": {"odd64_primary": 1e-12, "odd32_primary": 1e-6, "odd64_half": 1e-12, "odd32_half": 1e-6, "angle32": 1e-6, "complete_change64": 1e-12, "length32": 1e-6}}
checks["random_controls_nounness_neutral_and_equal_angle"] = all(stats["random"][k] <= v for k, v in limits["random"].items())
checks["plurality_control_neutral"] = all(stats["plurality"][k] <= v for k, v in limits["plurality"].items())
checks["nounness_doses_angles_complete_change"] = all(stats["noun"][k] <= v for k, v in limits["noun"].items())
checks["base_vectors_equal_model_rows"] = stats["base_equals_model_row"]
checks["geometry_maxima_within_tolerances"] = all(lock["geometry"]["check_maxima"][name] <= cr.TOLERANCES[kind] for name, kind in cr.GEOMETRY_LIMITS.items())
nearest = lock["nearest_tokens"]
checks["nearest_token_is_own_for_base_and_primary"] = all(entry["nearest_is_own"] for cue in nearest.values() for entry in cue.values())
checks["nothing_refused_one_model_load"] = not refused and loads["count"] == 1
checks["git_head_origin_remote_tree"] = git("rev-parse", "HEAD") == git("rev-parse", "origin/main") == git("ls-remote", "origin", "refs/heads/main").split()[0] \
    == "d3ecbd6feb1bd2aefe08dc0f8d799815be47c519" and git("status", "--porcelain", "--untracked-files=all") == ""
checks["validate_passes"] = runner.validate() == 0
checks["no_installed_lock"] = not (ROOT / cr.LOCK_RELATIVE_PATH).exists() and not (ROOT / cr.PREREGISTRATION_RELATIVE_PATH).exists()

digests = {
    "candidate_lock_file_sha256": hashlib.sha256(lock_bytes).hexdigest(), "candidate_lock_content_sha256": lock["content_sha256"],
    "candidate_preregistration_sha256": hashlib.sha256(prereg_text.encode("utf-8")).hexdigest(),
    "results_file_sha256": hashlib.sha256((OUT / "results.json").read_bytes()).hexdigest(), "results_state_sha256": state["state_sha256"],
    "direction_sha256": lock["geometry"]["direction_sha256"], "plurality_direction_sha256": lock["geometry"]["plurality_sha256"],
    "all_40_cue_geometry_records_sha256": pm.sha256_text(pm.canonical_json([{k: cue[k] for k in ("word", "token_id", "t_hat_sha256", "controls_sha256", "plurality_sha256",
                                                                                                 "vectors64_sha256", "vectors32_sha256")} for cue in lock["geometry"]["cues"]])),
    "all_280_random_controls_sha256": rc.tensor_digest(torch.stack(all_controls)), "all_840_patched_float32_vectors_sha256": rc.tensor_digest(torch.stack(all_vectors32)),
    "exposed_020_states_sha256": lock["dependencies"]["readout_020"]["exposed_states_sha256"], "parameters_sha256": lock["dependencies"]["model"]["parameters_sha256"],
    "embedding_sha256": lock["dependencies"]["model"]["embedding_sha256"],
}
ranges = {name: [min(c[name] for c in per_cue), max(c[name] for c in per_cue)] for name in ("s0", "tau", "theta_primary_deg", "theta_half_deg", "even_primary")}
angle_other = [entry["angle_to_nearest_other_deg"] for cue in nearest.values() for entry in cue.values()]
result = {"checks": checks, "all": all(checks.values()), "refused": refused, "model_loads": loads["count"], "digests": digests, "d_norm": lock["geometry"]["d_norm"],
          "cos_d_plurality": lock["geometry"]["cos_d_plurality"], "ranges": ranges, "nearest_other_angle_deg_min": min(angle_other),
          "check_maxima": lock["geometry"]["check_maxima"], "random_control_stats": stats["random"], "plurality_stats": stats["plurality"], "noun_stats": stats["noun"],
          "statistics": lock["statistics"], "outcome_table": lock["outcome"]["table"], "tolerances": lock["tolerances"], "per_cue": per_cue, "validate_log": logs[-1:]}
print(json.dumps(result, indent=1))
