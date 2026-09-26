"""Items 4 (drift refusal), 11, 12, 14, 15, 16, 19 (logic): SYNTHETIC unit tests of the pure cue_rotation functions.
No model is loaded (the loader refuses), no forward/capture/intervention can run, and no real measurement exists: every
ell, D_attn, A, B, G or Level-1 number below is made up by this script to probe the code's arithmetic and control flow.
validate_lock is exercised with the real candidate lock and synthetic git states / changed-path lists."""
import guard  # noqa: F401

import copy
import json
import math
import random
from fractions import Fraction
from importlib import util as _util

ROOT = guard.ROOT
HERE = guard.HERE

import torch  # noqa: E402

from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

guard.seal(confirm_only=False)
guard.refuse_model_load()
out: dict = {}


def raises(fn, *exc):
    try:
        fn()
    except exc as error:
        return f"{type(error).__name__}: {str(error)[:160]}"
    return None


# ---- Item 11: the exact tail and the threshold.
tail = Fraction(sum(math.comb(40, k) for k in range(27, 41)), 2 ** 40)
out["tail"] = {"cr_binomial_tail_equals_mine": cr.binomial_tail(40, 27) == tail, "derive_threshold": cr.derive_threshold(40),
               "production_reference_tail": float(cr.PRODUCTION.reference_tail()), "mine": float(tail), "26": float(Fraction(sum(math.comb(40, k) for k in range(26, 41)), 2 ** 40))}

# ---- count_positive: strictly positive; ties, zeros, NaN (and non-finite) count against.
probe = [1e-300, 0.0, -0.0, -1e-300, float("nan"), float("inf"), float("-inf"), None, 5.0]
out["count_positive"] = {"values": [repr(v) for v in probe], "count": cr.count_positive(probe), "expected_strictly_positive_finite": 2}

# ---- Item 12: the outcome hierarchy, every A x B x G.
out["outcome_label"] = {f"A{int(a)}B{int(b)}G{int(g)}": cr.outcome_label(a, b, g) for a in (False, True) for b in (False, True) for g in (False, True)}

# ---- Item 11: cr.statistics on synthetic per-cue values (the real 40 tokens' names only).
freeze = json.loads((ROOT / cr.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))
runner_spec = _util.spec_from_file_location("experiment_025_runner", ROOT / "experiments/025-nounness-direction-intervention/run.py")
runner_module = _util.module_from_spec(runner_spec)
import sys  # noqa: E402

sys.modules[runner_spec.name] = runner_module
runner_spec.loader.exec_module(runner_module)
guard.sweep()
guard.refuse_model_load()
runner = runner_module.Runner(log=lambda m: None, model_loader=guard._refusal("runner.model_loader"), tokenizer_loader=guard._refusal("runner.tokenizer_loader"))
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
P = cr.PRODUCTION.primary
conditions = cr.PRODUCTION.conditions


def synthetic_values(a_signs, b_margin, g_signs, seed=0):
    """ell and d_attn per cue and condition, made up: A_i = a_signs[i]*0.1 (0 -> exact tie), random-control |A_ij| equal
    to 0.1 - b_margin[i] (so B_i = A_i - 0.1 + b_margin[i]), G_i = g_signs[i]*0.01."""
    rng = random.Random(seed)
    values = {}
    for i, token in enumerate(confirmation.tokens):
        entry = {c: {"ell": -3.0, "d_attn": 0.2, "mse": 0.05, "nmse": 0.5, "frames": 108, "ell_by_template": {}} for c in conditions}
        a = a_signs[i] * 0.1
        entry[f"noun+{P}"]["ell"], entry[f"noun-{P}"]["ell"] = -3.0 + a, -3.0 - a
        size = 0.1 - b_margin[i]
        for j in range(1, 8):
            sign = rng.choice((1.0, -1.0))  # signed random effects: B must use their absolute values
            entry[f"rand{j}+{P}"]["ell"], entry[f"rand{j}-{P}"]["ell"] = -3.0 + sign * size, -3.0 - sign * size
        g = g_signs[i] * 0.01
        entry[f"noun+{P}"]["d_attn"], entry[f"noun-{P}"]["d_attn"] = 0.2 + g, 0.2 - g
        values[int(token["token_id"])] = entry
    return values


scenarios = {
    "A27_B26_G40": ([1] * 27 + [0] * 13, [0.01] * 26 + [-0.01] * 14, [1] * 40),
    "A26_B40_G40": ([1] * 26 + [-1] * 14, [0.01] * 40, [1] * 40),
    "A40_B27_G26": ([1] * 40, [0.01] * 27 + [-0.01] * 13, [1] * 26 + [0] * 14),
    "A40_B27_G27": ([1] * 40, [0.01] * 27 + [-0.01] * 13, [1] * 27 + [-1] * 13),
}
stats_out = {}
for name, (a_signs, margins, g_signs) in scenarios.items():
    values = synthetic_values(a_signs, margins, g_signs)
    result = cr.statistics(values, confirmation, cr.PRODUCTION)
    # recompute A, B, G by my own formula from the same synthetic values
    mine_ok = True
    for row in result["per_cue"]:
        v = values[row["token_id"]]
        a = 0.5 * (v[f"noun+{P}"]["ell"] - v[f"noun-{P}"]["ell"])
        aj = [0.5 * (v[f"rand{j}+{P}"]["ell"] - v[f"rand{j}-{P}"]["ell"]) for j in range(1, 8)]
        b = a - math.fsum(abs(x) for x in aj) / 7
        g = 0.5 * (v[f"noun+{P}"]["d_attn"] - v[f"noun-{P}"]["d_attn"])
        mine_ok &= row["A"] == a and row["A_random"] == aj and row["B"] == b and row["G"] == g
    stats_out[name] = {"A": [result["A"]["positive"], result["A"]["result"]], "B": [result["B"]["positive"], result["B"]["result"]],
                       "G": [result["G"]["positive"], result["G"]["result"]], "label": result["outcome"]["label"], "per_cue_equals_my_formulas": mine_ok,
                       "by_stratum_A": result["A"]["by_stratum"], "reference_tail": result["criterion"]["reference_tail"]}
out["statistics_synthetic"] = stats_out

# ---- D_attn definition on synthetic rows (not from any run).


class FakeState:
    def __init__(self, rows4, rows5):
        self.rows4, self.rows5 = rows4, rows5


gen = torch.Generator().manual_seed(7)
K, KMAX = 5, 9
ref4 = torch.distributions.Dirichlet(torch.ones(K)).sample((8,)).double()
ref5 = torch.distributions.Dirichlet(torch.ones(K)).sample((8,)).double()
cap4 = torch.zeros(8, KMAX, dtype=torch.float32)
cap5 = torch.zeros(8, KMAX, dtype=torch.float32)
cap4[:, :K] = torch.distributions.Dirichlet(torch.ones(K)).sample((8,)).float()
cap5[:, :K] = torch.distributions.Dirichlet(torch.ones(K)).sample((8,)).float()
cap4_garbage = cap4.clone()
cap4_garbage[:, K:] = 0.5  # anything beyond p_t + 1 must be ignored
tv = [0.5 * float((cap4.double()[h, :K] - ref4[h]).abs().sum()) for h in range(8)] + [0.5 * float((cap5.double()[h, :K] - ref5[h]).abs().sum()) for h in range(8)]
mine = sum(tv) / 16
state = FakeState(ref4, ref5)
out["d_attn_synthetic"] = {"cr": cr.routing_distance(cap4, cap5, state), "mine_equal_mean_of_16_head_TVs": mine,
                           "abs_diff": abs(cr.routing_distance(cap4, cap5, state) - mine),
                           "padding_ignored": cr.routing_distance(cap4_garbage, cap5, state) == cr.routing_distance(cap4, cap5, state),
                           "too_few_keys_refused": raises(lambda: cr.routing_distance(cap4[:, :K - 1], cap5, state), pm.IncidentError)}

# ---- per_cue on synthetic tensors: ell = log(sum SSE / sum n) over the cue's frames; D_attn = the frame mean.
frames = sorted(base.inputs.pool.frames, key=lambda f: f.frame_id)
pick = [f for f in frames if f.template_id != ul.COORDINATED][:2] + [f for f in frames if f.template_id == ul.COORDINATED][:1]
tokens = [{"word": "synthA", "token_id": 11, "stratum": "adjective"}, {"word": "synthB", "token_id": 7, "stratum": "noun"}]
units = ul.table_units(tokens, pick)
conds = ("base", "x")
tensors = {}
fake_states = {}
torch.manual_seed(3)
for f in pick:
    fake_states[f.frame_id] = FakeState(torch.distributions.Dirichlet(torch.ones(f.p_t + 1)).sample((8,)).double(),
                                        torch.distributions.Dirichlet(torch.ones(f.p_t + 1)).sample((8,)).double())
for group, pairs in units.pairs.items():
    kmax = max(units.frames[f].p_t + 1 for _, f in pairs)
    for c in conds:
        tensors[cr.tensor_name(group, c, "dc")] = torch.randn(len(pairs), 79, dtype=torch.float64)
        tensors[cr.tensor_name(group, c, "C")] = torch.randn(len(pairs), 79, dtype=torch.float64)
        r4 = torch.zeros(len(pairs), 8, kmax)
        r5 = torch.zeros(len(pairs), 8, kmax)
        for row, (t, f) in enumerate(pairs):
            k = units.frames[f].p_t + 1
            r4[row, :, :k] = torch.distributions.Dirichlet(torch.ones(k)).sample((8,))
            r5[row, :, :k] = torch.distributions.Dirichlet(torch.ones(k)).sample((8,))
        tensors[cr.tensor_name(group, c, "rows4")], tensors[cr.tensor_name(group, c, "rows5")] = r4, r5
values = cr.per_cue(units, tensors, fake_states, conds)
ok = True
for t, token in enumerate(units.tokens):
    for c in conds:
        sse, n, dist = [], [], []
        for group, pairs in units.pairs.items():
            for row, (tt, f) in enumerate(pairs):
                if tt != t:
                    continue
                y, cc = tensors[cr.tensor_name(group, c, "dc")][row], tensors[cr.tensor_name(group, c, "C")][row]
                sse.append(float(((y - cc) ** 2).sum()))
                n.append(79.0)
                st = fake_states[units.frames[f].frame_id]
                k = units.frames[f].p_t + 1
                heads = [0.5 * float((tensors[cr.tensor_name(group, c, "rows4")][row].double()[h, :k] - st.rows4[h]).abs().sum()) for h in range(8)]
                heads += [0.5 * float((tensors[cr.tensor_name(group, c, "rows5")][row].double()[h, :k] - st.rows5[h]).abs().sum()) for h in range(8)]
                dist.append(sum(heads) / 16)
        ell = math.log(math.fsum(sse) / math.fsum(n))
        got = values[int(token["token_id"])][c]
        ok &= abs(got["ell"] - ell) < 1e-12 and abs(got["d_attn"] - sum(dist) / len(dist)) < 1e-12 and got["frames"] == len(pick)
out["per_cue_synthetic_matches_definition"] = ok

# ---- Item 14: Level-1 routing on a fake readout; the gate covers exactly the 16 outcome-bearing conditions.


class FakeReadout:
    def level1_detail(self, state, dx3, l4_heads=True):
        assert l4_heads is True
        return {"dh6": torch.zeros(512, dtype=torch.float64)}

    def contrast(self, state, dh6, nouns):
        return torch.zeros(90, dtype=torch.float64)


class FakeProgs:
    readout = FakeReadout()
    nouns = None
    scorable = list(range(79))


class FakeState3:
    def __init__(self, p_t):
        self.x3_all = [torch.zeros(512, dtype=torch.float64) for _ in range(p_t + 1)]


tokens1 = [{"word": "synthA", "token_id": 11, "stratum": "adjective"}]
units1 = ul.table_units(tokens1, pick)
t1 = {}
for group, pairs in units1.pairs.items():
    for k, c in enumerate(conditions):
        t1[cr.tensor_name(group, c, "x3")] = torch.zeros(len(pairs), 2, 512, dtype=torch.float32)
        t1[cr.tensor_name(group, c, "dc")] = torch.full((len(pairs), 79), -(k + 1) * 1e-3, dtype=torch.float64)  # |L1 - dc| = (k + 1) e-3
states1 = {f.frame_id: FakeState3(f.p_t) for f in pick}
gates = cr.level1_gates(FakeProgs(), units1, states1, t1, conditions, cr.PRODUCTION)
bearing_idx = [conditions.index(c) for c in cr.PRODUCTION.outcome_bearing]
secondary_idx = [i for i in range(len(conditions)) if i not in bearing_idx]
out["level1_routing"] = {"outcome_bearing_conditions": list(cr.PRODUCTION.outcome_bearing), "n": len(cr.PRODUCTION.outcome_bearing), "gates": gates,
                         "bearing_max_expected": (max(bearing_idx) + 1) * 1e-3, "secondary_max_expected": (max(secondary_idx) + 1) * 1e-3,
                         "bearing_at_condition": gates["level1_outcome_bearing"]["at"].split("|")[-1], "secondary_at_condition": gates["level1_secondary"]["at"].split("|")[-1]}


def gate_set(i1=0.0, i3=0.0, i4=0.0, l1=0.0, l1s=0.0):
    return {"I1": {"max": i1, "at": "x"}, "I3": {"max": i3, "at": "x"}, "I4": {"max": i4, "at": "x"}, "level1_outcome_bearing": {"max": l1, "at": "x"},
            "level1_secondary": {"max": l1s, "at": "x"}}


out["enforce_gates"] = {
    "all_at_tolerance_pass": raises(lambda: cr.enforce_gates(gate_set(1e-4, 1e-4, 1e-3, 2e-2)), pm.IncidentError),
    "I1_above": raises(lambda: cr.enforce_gates(gate_set(i1=1.0000001e-4)), pm.IncidentError),
    "I3_above": raises(lambda: cr.enforce_gates(gate_set(i3=1.0000001e-4)), pm.IncidentError),
    "I4_above": raises(lambda: cr.enforce_gates(gate_set(i4=1.0000001e-3)), pm.IncidentError),
    "level1_bearing_above": raises(lambda: cr.enforce_gates(gate_set(l1=0.0200001)), pm.IncidentError),
    "level1_bearing_undefined": raises(lambda: cr.enforce_gates(gate_set(l1=None)), pm.IncidentError),
    "level1_secondary_huge_not_enforced": raises(lambda: cr.enforce_gates(gate_set(l1s=99.0)), pm.IncidentError),
    "precedence_I1_before_level1": raises(lambda: cr.enforce_gates(gate_set(i1=1.0, l1=1.0)), pm.IncidentError),
}
worst = {"max": 0.0, "at": ""}
ul._worse(worst, float("nan"), "somewhere")
ul._worse(worst, 5.0, "later")
out["nan_is_never_displaced"] = worst

# ---- I7' enforcement on the lock's own checks.
lock_text = (ROOT / "outputs/experiment-025/candidate-lock.json").read_text(encoding="utf-8")
lock = json.loads(lock_text)
checks = dict(lock["geometry"]["cues"][0]["checks"])
bad_base = dict(checks, base_equals_model_row=False)
bad_neutral = dict(checks, neutral32=2e-6)
bad_nan = dict(checks, angle32=float("nan"))
out["enforce_geometry"] = {"lock_checks_pass_all_40": all(raises(lambda c=c: cr.enforce_geometry(c["checks"], c["word"]), pm.IncidentError) is None for c in lock["geometry"]["cues"]),
                           "base_row_false": raises(lambda: cr.enforce_geometry(bad_base, "x"), pm.IncidentError),
                           "neutral32_2e-6": raises(lambda: cr.enforce_geometry(bad_neutral, "x"), pm.IncidentError),
                           "nan": raises(lambda: cr.enforce_geometry(bad_nan, "x"), pm.IncidentError)}

# ---- Phase rules on synthetic states.
state = rd.load_results_state(ROOT / "outputs/experiment-025/results.json")


def with_(mutate):
    s = copy.deepcopy(state)
    mutate(s)
    return s


def phase(p, s):
    return raises(lambda: cr.assert_phase_allowed(p, s), cr.PhaseError) or "allowed"


out["phase_rules"] = {
    "now_lock": phase("lock", state), "now_confirm": phase("confirm", state), "now_report": phase("report", state),
    "confirm_running_confirm": phase("confirm", with_(lambda s: s["phases"].__setitem__("confirm", {"status": "running"}))),
    "confirm_complete_confirm": phase("confirm", with_(lambda s: s["phases"].__setitem__("confirm", {"status": "complete"}))),
    "pre_ledger_incident_confirm": phase("confirm", with_(lambda s: s["phases"]["confirm"].__setitem__("incidents", [{"message": "x"}]))),
    "pre_ledger_incident_report": phase("report", with_(lambda s: s["phases"]["confirm"].__setitem__("incidents", [{"message": "x"}]))),
    "confirmation_incident_report": phase("report", with_(lambda s: s.__setitem__("confirmation", {"incident": {"message": "x"}}))),
    "no_state_lock": phase("lock", None), "no_state_confirm": phase("confirm", None),
}

# ---- Item 4: validate_lock with the real candidate (as if installed byte for byte) and synthetic git situations.
prereg = (ROOT / "outputs/experiment-025/candidate-preregistration.md").read_text(encoding="utf-8")
placeholder = cr.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
noun_keys = runner._noun_keys(base.inputs)
install = [cr.LOCK_RELATIVE_PATH, cr.PREREGISTRATION_RELATIVE_PATH, f"{cr.EXPERIMENT_DIR}/README.md", f"{cr.EXPERIMENT_DIR}/evidence/lock-review/REVIEW.md"]


def vl(*, lock_=lock, state_=state, digests=None, text=prereg, git_state=None, tracked=True, changed=install):
    return raises(lambda: cr.validate_lock(lock_, state=state_, digests=digests or base.digests, config=cr.PRODUCTION, confirmation=confirmation,
                                           confirmation_file_sha256=confirmation_sha, dependencies=placeholder, noun_keys=noun_keys, preregistration_text=text,
                                           git_state=git_state or {"commit": "f" * 40, "dirty": False}, tracked=tracked, changed_paths=changed), cr.PhaseError)


edited = copy.deepcopy(lock)
edited["geometry"]["cues"][0]["s0"] = edited["geometry"]["cues"][0]["s0"] + 1e-15
edited["content_sha256"] = rc.content_digest(edited)
digests_changed = dict(base.digests)
digests_changed["024_calibration_file"] = "0" * 64
real_own = cr.own_blob
cr.own_blob = lambda: "0" * 40
changed_module = vl()
cr.own_blob = real_own
out["validate_lock"] = {
    "installed_byte_for_byte_clean_tracked_noscience": vl(), "noun_keys_equal_lock": noun_keys == lock["noun_keys"],
    "dirty_tree": vl(git_state={"commit": "f" * 40, "dirty": True}), "untracked": vl(tracked=False), "not_ancestor": vl(changed=None),
    "src_change": vl(changed=install + ["src/neural_decompiler/capture.py"]), "runner_change": vl(changed=install + [f"{cr.EXPERIMENT_DIR}/run.py"]),
    "input_024_change": vl(changed=install + ["experiments/024-readout-routing-nounness/calibration-v1.json"]),
    "input_digest_change": vl(digests=digests_changed), "running_module_blob_differs": changed_module,
    "prereg_one_char": vl(text=prereg.replace("0.32", "0.33", 1)), "lock_edited_resigned": vl(lock_=edited),
}

# ---- Item 16: the patch-path keys: a key that is not spent is refused before any forward.
frames_by_id = {f.frame_id: f for f in base.inputs.pool.frames}
out["patch_path"] = {"keys": list(cr.PATCH_PATH_SPENT_KEYS), "lock_keys_equal": lock["patch_path_spent_keys"] == list(cr.PATCH_PATH_SPENT_KEYS),
                     "unspent_key_refused_before_forward": raises(lambda: cr.patch_path_check(None, frames_by_id, None, frozenset(), keys=[cr.PATCH_PATH_SPENT_KEYS[0]]), pm.IncidentError),
                     "all_in_forbidden": all(k in base.forbidden for k in cr.PATCH_PATH_SPENT_KEYS)}
tagged_spent = f"{cr.PATCH_PATH_SPENT_KEYS[0]}|base"
out["ledger_isolation"] = {"tagged_key_of_spent_prompt_refused": raises(lambda: cr.assert_ledger_isolated([tagged_spent], base.forbidden, "x"), cr.PhaseError),
                           "fresh_manifest_passes": raises(lambda: cr.assert_ledger_isolated(sorted(confirmation.tagged_keys()), base.forbidden, "x"), cr.PhaseError)}

# ---- Item 12/13: render_report on synthetic states.
synthetic_results = cr.statistics(synthetic_values(*scenarios["A40_B27_G27"]), confirmation, cr.PRODUCTION)
s_ok = with_(lambda s: (s["phases"].__setitem__("confirm", {"status": "complete"}), s.__setitem__("confirmation", {"results": synthetic_results, "gates": {}})))
s_inc = with_(lambda s: s.__setitem__("confirmation", {"incident": {"commit": "c", "message": "m"}}))
s_pre = with_(lambda s: s["phases"]["confirm"].__setitem__("incidents", [{"commit": "c", "message": "m"}]))
report_ok = cr.render_report(s_ok)
out["render_report"] = {"incident": "NOT_INTERPRETABLE" in cr.render_report(s_inc) and synthetic_results["outcome"]["label"] not in cr.render_report(s_inc),
                        "pre_ledger_incident": "NOT_INTERPRETABLE" in cr.render_report(s_pre),
                        "result_label_rendered": synthetic_results["outcome"]["label"] in report_ok, "A_semantics_rendered": cr.SEMANTICS["A"] in report_ok,
                        "A_semantics_is_plus_vs_minus": "+nounness intervention produced greater frozen-readout approximation error than the matched −nounness" in cr.SEMANTICS["A"],
                        "lock_semantics_equal_module": lock["semantics"] == cr.SEMANTICS}

# ---- Configuration immutability.
try:
    cr.PRODUCTION.count_threshold = 26
    frozen = False
except Exception:
    frozen = True
other = cr.Configuration(name="test", n_adjectives=1, n_nouns=1, k_controls=2, primary_odd=0.32, half_odd=0.16, count_threshold=1, n_frames=108, n_scored_nouns=79,
                         expected_picks=(("adjective", ("a",)), ("noun", ("b",))))
out["configuration"] = {"production_frozen": frozen, "non_production_refused_at_real_root": raises(lambda: runner_module.Runner(config=other), cr.PhaseError)}

# ---- Item 7: the complete change, float64 and float32 separately (from the reviewer's own saved vectors).
own = torch.load(HERE / "r2a_own.pt")
mine = json.loads((HERE / "r2a_own.json").read_text(encoding="utf-8"))
d = own["d"]


def s_of(v):
    v = v.double()
    return float(torch.dot(d, v / torch.linalg.vector_norm(v)))


w64 = w32 = 0.0
for g in mine["cues"]:
    v64, v32 = own[f"{g['token_id']}/v64"], own[f"{g['token_id']}/v32"]
    for x, th in (("0.32", g["theta_primary"]), ("0.16", g["theta_half"])):
        for sign, sym in ((1, "+"), (-1, "-")):
            idx = list(conditions).index(f"noun{sym}{x}")
            expected = g["s0"] * (math.cos(th) - 1.0) + sign * float(x)
            w64 = max(w64, abs((s_of(v64[idx]) - g["s0"]) - expected))
            w32 = max(w32, abs((s_of(v32[idx]) - g["s0"]) - expected))
out["complete_change"] = {"max_dev_float64": w64, "max_dev_float32": w32}

out["guard"] = guard.summary()
(HERE / "r5_logic.out.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
