"""Items 5, 6 (and the ladder of item 12): C rebuilt from the saved Δx3, the identity gates and the Level-1 identity,
under a sealed weights-only load. No prompt, no forward pass; nothing recorded in the results state is read here except
for the final comparison printed at the end (clearly separated)."""
import rguard  # noqa: F401  (first)
from rguard import REPO

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

import rseal

ROOT = Path(REPO)
SCR = Path(rguard.SCRATCH)
sys.path.insert(0, str(ROOT / "src"))
# import the heavy model modules first so that their classes exist (and are counted) during the load
import transformer_lens.model_bridge  # noqa: E402,F401
import transformers.models.gpt_neox.modeling_gpt_neox  # noqa: E402,F401

spec = importlib.util.spec_from_file_location("run024_confirmation_review", ROOT / "experiments/024-readout-routing-nounness/run.py")
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)
rr, ul, rd, b0c, pm = run.rr, run.ul, run.rd, run.b0c, run.pm
from neural_decompiler.models import PYTHIA_70M, load_model, seed_runtime  # noqa: E402

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail != '' else ''}", flush=True)


def refuse(*a, **k):
    raise RuntimeError("the runner's loaders are never used here")


runner = run.Runner(log=lambda m: None, model_loader=refuse, tokenizer_loader=refuse)
base = runner._base()
confirmation, _ = runner._confirmation(base)
lock = json.loads((ROOT / rr.LOCK_RELATIVE_PATH).read_text())
freeze = json.loads((ROOT / rr.CONFIRMATION_RELATIVE_PATH).read_text())
T = torch.load(ROOT / "outputs/experiment-024/stage2-measurements.pt", weights_only=True)

# --- own row mapping (from reading ul.table_units / ul.stage_two_022): tokens by token id, frames by frame_id; per group
# block, token-major then frame (cue-final = every non-coordinated template) -------------------------------------------
tokens = sorted(({"word": c["word"], "token_id": int(c["token_id"]), "class": c["class"]} for c in freeze["cues"]), key=lambda c: c["token_id"])
frames = sorted(base.inputs.pool.frames, key=lambda f: f.frame_id)
check("pool frames == the freeze's 108 exposed frame ids", sorted(f.frame_id for f in frames) == sorted(freeze["exposed_frame_ids"]) and len(frames) == 108)
COORD = pm.COORDINATED_TEMPLATE
mine = {"cue_final": [(t, f) for t in range(40) for f, fr in enumerate(frames) if fr.template_id != COORD],
        "coordinated": [(t, f) for t in range(40) for f, fr in enumerate(frames) if fr.template_id == COORD]}
units = rr.target_units(confirmation)
check("own pair mapping == rr.target_units(confirmation).pairs; tokens and frames in the same order",
      {g: list(p) for g, p in units.pairs.items()} == mine and [int(t["token_id"]) for t in units.tokens] == [t["token_id"] for t in tokens]
      and [f.frame_id for f in units.frames] == [f.frame_id for f in frames] and list(units.pairs) == ["cue_final", "coordinated"])
check("group sizes 2,880 / 1,440 == the saved row counts", len(mine["cue_final"]) == T["Y1/cue_final/dc"].shape[0] and len(mine["coordinated"]) == T["Y1/coordinated/dc"].shape[0])
pos_ok = all(T[f"Y1/{g}/positions"][row].tolist() == [frames[f].p_c, frames[f].p_t] for g, pairs in mine.items() for row, (_, f) in enumerate(pairs))
check("saved positions[row] == (p_c, p_t) of the mapped frame for all 4,320 rows", pos_ok)
states = ul.y1_states(base.inputs.closure["exploration"]["locked_states"], frames)

# --- the sealed weights-only load ------------------------------------------------------------------------------------
check("torch threads == 4 (as at confirm)", torch.get_num_threads() == 4, torch.get_num_threads())
seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
seal = rseal.Seal()
seal.count_during_load()
t0 = time.time()
model = load_model(PYTHIA_70M)
print(f"   load {time.time() - t0:.1f}s; counts during the load {seal.load_counts}", flush=True)
n_sealed = seal.refuse_everything()
bad = seal.verify_model_sealed(model)
print(f"   sealed {n_sealed} module classes; entry points refused: {len(seal.entry_points)}: {sorted(set(e.split('.')[-1] for e in seal.entry_points))}", flush=True)
check("the load made 0 Module.__call__ / forward / __call__-override calls", seal.load_counts == {"Module.__call__": 0, "forward": 0, "__call__ override": 0}, seal.load_counts)
check("every module class of the loaded model now resolves forward and __call__ to a refusing stub", bad == [], bad)
check("pm.capture_prompt, pm.run_capture, pm.record_execution, ul.stage_two_022, ul.measure_prompt refused",
      all(getattr(getattr(m, n), "__name__", "") == "refusing_stub" for m, n in ((pm, "capture_prompt"), (pm, "run_capture"), (pm, "run_patched"), (pm, "run_interventions"),
                                                                                (pm, "record_execution"), (ul, "stage_two_022"), (ul, "measure_prompt"))))
params = rr.parameters_digest(model)
check("parameters digest == the lock's (the pinned weights)", params == lock["dependencies"]["model"]["parameters_sha256"], params)
progs = ul.ModelPrograms.from_model(model, base.inputs)
check("embedding digest == the lock's", rr.embedding_digest(progs.weights.W_E) == lock["dependencies"]["model"]["embedding_sha256"])
check("progs.scorable: the 79 scorable exposed nouns in the lock's order", [progs.nouns.nouns[i].lexical_key for i in progs.scorable] == lock["noun_keys"])


def changes(tensor, row, frame):
    return {frame.p_c: tensor[row, 0]} if frame.p_t == frame.p_c else {frame.p_c: tensor[row, 0], frame.p_t: tensor[row, 1]}


def worse(entry, value, where):
    if math.isnan(value):
        entry.update({"max": None, "at": where, "nan": True})
    elif entry.get("max") is not None and value > entry["max"]:
        entry.update({"max": value, "at": where})


# --- item 5: C from the saved Δx3 ---------------------------------------------------------------------------------------
t0 = time.time()
c_stats = {"pairs": 0, "bitwise_equal_rows": 0, "max": 0.0, "at": "", "unequal": []}
for group, pairs in mine.items():
    dx3, saved = T[f"Y1/{group}/dx3"], T[f"Y1/{group}/ceiling"]
    for row, (t, f) in enumerate(pairs):
        frame = frames[f]
        again = ul.contrast_of(progs, states[frame.frame_id], changes(dx3, row, frame))
        c_stats["pairs"] += 1
        if torch.equal(again, saved[row]):
            c_stats["bitwise_equal_rows"] += 1
        else:
            c_stats["unequal"].append(f"{tokens[t]['word']}|{frame.frame_id}")
        worse(c_stats, float((again - saved[row]).abs().max()), f"{tokens[t]['word']}|{frame.frame_id}")
print(f"   C recompute {time.time() - t0:.1f}s", flush=True)
check("C rebuilt from the saved Δx3 == saved C bit for bit on all 4,320 pairs (max |diff| 0.0)", c_stats["pairs"] == 4320 and c_stats["bitwise_equal_rows"] == 4320
      and c_stats["max"] == 0.0, {k: v for k, v in c_stats.items() if k != "unequal"})

# --- item 6 (own loop, 023's formulas) and the Level-1 identity / ladder (item 12) ---------------------------------------
t0 = time.time()
gates = {name: {"max": 0.0, "at": ""} for name in ("I1", "I3", "I4")}
level1 = {"max": 0.0, "at": ""}
classes = {t["token_id"]: t["class"] for t in tokens}
ladder_py = {cls: {"C": 0.0, "C5": 0.0, "L1": 0.0} for cls in "NBDCE"}  # python float accumulation in pair order (rr.ladder's)
ladder_terms = {cls: {"C": [], "C5": [], "L1": []} for cls in "NBDCE"}  # for an order-free math.fsum
per_group_level1 = {"cue_final": 0.0, "coordinated": 0.0}
rows16 = {}
for group, pairs in mine.items():
    dx1_t, dx3_t, saved, dc = T[f"Y1/{group}/dx1"], T[f"Y1/{group}/dx3"], T[f"Y1/{group}/ceiling"], T[f"Y1/{group}/dc"]
    for row, (t, f) in enumerate(pairs):
        frame, token = frames[f], tokens[t]
        state = states[frame.frame_id]
        where = f"{token['word']}|{frame.frame_id}"
        if frame.frame_id not in rows16:
            rows16[frame.frame_id] = ul.reference_rows_017(progs.programs, state)
        ctx = ul.pair_context(progs, frame, state, int(confirmation.reference_ids[frame.template_id]), token["token_id"], token["word"], rows16[frame.frame_id])
        fac = ctx.factors
        dx1, dx3 = changes(dx1_t, row, frame), changes(dx3_t, row, frame)
        i1 = float((fac.d_emb + fac.delta_e + fac.block0_pc.total - dx1[frame.p_c]).abs().max())
        if fac.block0_pt is not None:
            i1 = max(i1, float((fac.block0_pt.total - dx1[frame.p_t]).abs().max()))
        worse(gates["I1"], i1, where)
        full = ul.compose_dx3(progs, ctx, b0c.FULL_MASK[group])
        # I3 as 023/022 define it: max over positions of ||Δx̂3 − Δx3||₂ / max(||Δx3||₂, 1e-12), own arithmetic
        i3 = max(float((full[p].double() - dx3[p].double()).norm()) / max(float(dx3[p].double().norm()), 1e-12) for p in sorted(dx3))
        check_i3 = ul.i3_error(full, dx3)
        if i3 != check_i3:
            print(f"   I3 own vs ul.i3_error differ at {where}: {i3} vs {check_i3}", flush=True)
        worse(gates["I3"], i3, where)
        worse(gates["I4"], float((ul.contrast_of(progs, state, full) - saved[row]).abs().max()), where)
        y, c = dc[row].double(), saved[row].double()
        c5 = progs.readout.contrast(state, progs.readout.level1_detail(state, dict(dx3), l4_heads=False)["dh6"], progs.nouns)[progs.scorable].double()
        l1 = progs.readout.contrast(state, progs.readout.level1_detail(state, dict(dx3), l4_heads=True)["dh6"], progs.nouns)[progs.scorable].double()
        cls = classes[token["token_id"]]
        for key, pred in (("C", c), ("C5", c5), ("L1", l1)):
            sq = (y - pred) ** 2
            ladder_py[cls][key] += float(sq.sum())
            ladder_terms[cls][key] += sq.tolist()
        l1_err = float((l1 - y).abs().max())
        worse(level1, l1_err, where)
        per_group_level1[group] = max(per_group_level1[group], l1_err)
    print(f"   {group} done ({time.time() - t0:.1f}s)", flush=True)
TOL = {"I1": 1e-4, "I3": 1e-4, "I4": 1e-3}
for name in ("I1", "I3", "I4"):
    check(f"{name} max {gates[name]['max']!r} at {gates[name]['at']} ≤ {TOL[name]:.0e}", gates[name]["max"] is not None and gates[name]["max"] <= TOL[name])
check(f"Level-1 identity max |L1 − Δc| {level1['max']!r} at {level1['at']} ≤ 2e-2 (descriptive, not a gate)", level1["max"] <= 2e-2, per_group_level1)
shares = {}
for cls in "NBDCE":
    s = ladder_py[cls]
    shares[cls] = {"block5_attention": (s["C"] - s["C5"]) / s["C"], "block4_attention": (s["C5"] - s["L1"]) / s["C"], "remaining_after_L1": s["L1"] / s["C"],
                   "sse": dict(s), "sse_fsum": {k: math.fsum(v) for k, v in ladder_terms[cls].items()}}
    print(f"   ladder {cls}: block-5 attention exact removes {shares[cls]['block5_attention']:.6f} of Σ(Δc−C)², block-4 attention then {shares[cls]['block4_attention']:.6f}; "
          f"left after L1 {shares[cls]['remaining_after_L1']:.6f}; SSE_C {s['C']:.6f} (fsum {shares[cls]['sse_fsum']['C']:.6f})", flush=True)

# --- the canonical gate function on the same saved tensors (a second route) ------------------------------------------
t0 = time.time()
canonical_gates = rr.target_gates(progs, confirmation, units, states, T)
print(f"   rr.target_gates {time.time() - t0:.1f}s: {canonical_gates}", flush=True)
check("rr.target_gates maxima and locations == own loop's (exactly)", all(canonical_gates[n]["max"] == gates[n]["max"] and canonical_gates[n]["at"] == gates[n]["at"] for n in gates))
check("the seal held: 0 refused module calls / entry points during programs, C, gates, Level-1", all(v == 0 for v in seal.refused.values()), seal.refused)

out = {"c_recompute": {k: v for k, v in c_stats.items()}, "gates": gates, "canonical_gates": canonical_gates, "level1_identity": {**level1, "per_group": per_group_level1},
       "ladder": shares, "seal": {"load_counts": seal.load_counts, "refused": seal.refused, "sealed_classes": n_sealed, "entry_points": seal.entry_points},
       "parameters_sha256": params}
(SCR / "s05_results.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

# --- only now: the recorded values, for comparison (after everything above was derived) ----------------------------------
state = json.loads((ROOT / "outputs/experiment-024/results.json").read_text())
rec_c, rec_g, rec_lad = state["confirmation"]["c_recompute"], state["confirmation"]["gates"], state["confirmation"]["descriptives"]["ladder"]
print(f"   recorded c_recompute {rec_c}")
print(f"   recorded gates {rec_g}")
check("recorded c_recompute agrees (bitwise_equal True, max 0.0, 4,320 pairs)", rec_c.get("bitwise_equal") is True and rec_c.get("max") == 0.0 and rec_c.get("pairs") == 4320)
check("recorded gates == own maxima and locations exactly", all(rec_g[n]["max"] == gates[n]["max"] and rec_g[n]["at"] == gates[n]["at"] for n in gates))
check("recorded Level-1 identity == own", rec_lad["level1_identity"]["max"] == level1["max"] and rec_lad["level1_identity"]["at"] == level1["at"], rec_lad["level1_identity"])
lad_eq = all(rec_lad["per_class"][c]["block5_attention"] == shares[c]["block5_attention"] and rec_lad["per_class"][c]["block4_attention"] == shares[c]["block4_attention"]
             for c in "NBDCE")
check("recorded ladder shares == own (bitwise, same accumulation order)", lad_eq,
      {c: (rec_lad["per_class"][c]["block5_attention"] - shares[c]["block5_attention"], rec_lad["per_class"][c]["block4_attention"] - shares[c]["block4_attention"]) for c in "NBDCE"})
print(f"S05 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
