"""Cross-check with the frozen functions, run AFTER my own scripts a, b and c. The model is loaded for its weights only.

Guard (installed before any neural_decompiler import): torch.nn.Module.__call__ refuses every call, after checking any
integer input for a 101 token sequence (the 60 design candidates × one/two/three/many, from script a); the capture and
intervention entry points and the sanctioned ri entry points refuse every call after checking the prompt's token ids.
The record says whether the guard fired. The repository write guard of common.py is active as well.
"""
import importlib.util
import json
import math
import sys
import time

import torch

from common import OUT, OUTPUTS_101, ROOT, WRITE_ATTEMPTS, FORBIDDEN_READS, canonical, dump, sha256_text, sha256_file

FORBIDDEN = {tuple(s) for s in json.loads((OUT / "forbidden_101_sequences.json").read_text())}
assert len(FORBIDDEN) == 240
GUARD = {"installed_at": time.time(), "module_calls_refused": 0, "entry_points_refused": [], "sequence_101_hits": []}


def _check_ids(ids, where):
    seq = tuple(int(t) for t in ids)
    if seq in FORBIDDEN or any(seq[:len(f)] == f for f in FORBIDDEN):
        GUARD["sequence_101_hits"].append(where)
        raise RuntimeError(f"review guard: 101 token sequence at {where}")


def _refuse_call(self, *args, **kwargs):
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.Tensor) and not a.is_floating_point():
            for row in a.reshape(-1, a.shape[-1]) if a.dim() > 0 else []:
                _check_ids(row.tolist(), f"Module.__call__({type(self).__name__})")
    GUARD["module_calls_refused"] += 1
    raise RuntimeError(f"review guard: forward refused ({type(self).__name__})")


torch.nn.Module.__call__ = _refuse_call

from neural_decompiler import capture, interventions, models  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402


def _refuse_entry(name):
    def inner(*args, **kwargs):
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, pm.Prompt):
                _check_ids(a.token_ids, name)
        GUARD["entry_points_refused"].append(name)
        raise RuntimeError(f"review guard: {name} refused")
    return inner


for mod, names in ((pm, ("capture_prompt", "run_patched", "run_capture", "run_interventions")), (capture, ("run_capture",)),
                   (interventions, ("run_interventions",)), (rd, ("capture_frame_020", "measure_pair", "measure_pair_with_boundaries")),
                   (ri, ("run_arm", "capture_reference", "stage_runs", "measure_frame"))):
    for name in names:
        if hasattr(mod, name):
            setattr(mod, name, _refuse_entry(f"{mod.__name__}.{name}"))
for module_name, module in list(sys.modules.items()):  # every alias of the refused entry points
    if module_name.startswith("neural_decompiler") and module is not None:
        for attr in ("capture_prompt", "run_patched", "run_capture", "run_interventions", "capture_frame_020"):
            if attr in vars(module) and callable(vars(module)[attr]) and getattr(vars(module)[attr], "__name__", "") != "inner":
                setattr(module, attr, _refuse_entry(f"{module_name}.{attr}"))

torch.set_grad_enabled(False)
out: dict = {"order": "run after a_keys_ledger.py, b_state_measurements.py and c_exact_chain.py (see the file times in out/)"}
index_mtime_before = (ROOT / ".git/index").stat().st_mtime

spec = importlib.util.spec_from_file_location("experiment_101_runner_review", ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
runner = runner_module.Runner(log=lambda message: None)
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
frames = runner._exposed_frames(base.inputs)
schedules = runner._calibration_schedules(base.inputs, frames)
references = [(frame, base.inputs.pool.reference_prompt(frame)) for frame in frames]
keys = [cr.tagged_key(prompt, "S1") for _, prompt in references] + [run.key for schedule in schedules for run in schedule]
mine_a = json.loads((OUT / "a_keys_ledger.json").read_text())
out["keys"] = {"frozen_count": len(keys), "frozen_sha256": pm.sha256_text(pm.canonical_json(sorted(keys))), "mine_sha256": mine_a["my_keys"]["planned_sha256_mine"],
               "frames_order": [f.frame_id for f in frames] == [k.split("|")[0] for k in keys[:36]]}
# the 60 design texts against ri.FRAME_CANDIDATES (token sequences: script a's set rebuilt from ri)
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file("/Users/gaborhorvath-ulenius/.cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c/tokenizer.json")
ri_seqs = {tuple(tok.encode(text.replace("{cue}", w), add_special_tokens=False).ids) for text in ri.FRAME_CANDIDATES for w in ("one", "two", "three", "many")}
out["keys"]["ri_candidates_sequences_equal_design"] = ri_seqs == FORBIDDEN
untagged = {cr.untagged(k) for k in keys}
out["keys"]["frozen_isolation"] = {"in_020_ledger": untagged <= frozenset(base.inputs.closure["ledger"]), "manifest_overlap": len(untagged & base.manifests),
                                   "forbidden_overlap_101": len(confirmation.untagged_keys() & base.forbidden)}

model = models.load_model(models.PYTHIA_70M)
progs = ul.ModelPrograms.from_model(model, base.inputs)
noun_keys = runner._check_nouns(progs, base.inputs)
out["nouns"] = {"count": len(noun_keys), "equal_020_order": noun_keys == [k for k in json.loads((ROOT / "experiments/020-readout-decompilation/confirmation-v1.json").read_text())["exposed_noun_keys"] if k != "peach"]}
geometry = ri.read_geometry(base.inputs.lock_011)
block = ri.geometry_block(geometry, progs.head, progs.axis_T, [f.frame_id for f in frames], ri.PRODUCTION.k_iso)
cand = json.loads((OUTPUTS_101 / "candidate-calibration-v1.json").read_text())
state = json.loads((OUTPUTS_101 / "results.json").read_text())
out["g1"] = {"recomputed": block["g1"], "candidate": cand["g1"], "equal": json.loads(canonical(block["g1"])) == cand["g1"], "w_c_norm": geometry.norm,
             "enforce_g1": "ok"}
ri.enforce_g1(block)

locked = base.inputs.closure["exploration"]["locked_states"]
program_states = {f.frame_id: rd.state_from_locked(locked[f.frame_id], f) for f in frames}
t0 = time.time()
table = ri.prediction_table(progs, frames, program_states, geometry, ri.PRODUCTION, noun_keys, fractions_for_weak=True)
table_off = ri.prediction_table(progs, frames, program_states, geometry, ri.PRODUCTION, noun_keys, fractions_for_weak=False)
out["table"] = {"content_sha256": table["content_sha256"], "candidate_table_content_sha256": cand["table_content_sha256"],
                "reproduces_across_processes": table["content_sha256"] == cand["table_content_sha256"], "fractions_for_weak_off_digest": table_off["content_sha256"],
                "seconds": time.time() - t0}

saved = torch.load(OUTPUTS_101 / "calibration-measurements.pt", weights_only=True)
logits = state["calibration"]["measurements"]["logits_sha256"]
ri.assert_measurements(saved, schedules)
values = [ri.frame_values(s, saved, ri.PRODUCTION, fractions_for_weak=True) for s in schedules]
values_by_cue = {label: [entry[label] for entry in values] for label in ri.PRODUCTION.cue_labels}
mine_b = json.loads((OUT / "b_values.json").read_text())
diff = 0.0
for f, entry in zip(frames, values):
    for label in ri.PRODUCTION.cue_labels:
        m = mine_b["values"][f.frame_id][label]
        fr = entry[label]["fractions"]
        for k in ("F_rem", "F_rem_off", "F_ins", "F_ins_off", "N", "S", "F_half_rem", "F_half_ins", "F_swap_pl", "F_swap_ref", "additivity"):
            diff = max(diff, abs(fr[k] - m[k]))
        diff = max(diff, abs(entry[label]["dc_full"] - m["dc_full"]), abs(entry[label]["A_amp"] - m["A_amp"]), abs(entry[label]["A_neg"] - m["A_neg"]))
        if label == "two":
            for k in ("iso_rem_margin", "iso_ins_margin", "F_head_rem", "F_head_rem_v", "F_head_ins", "F_head_ins_v"):
                diff = max(diff, abs(fr[k] - m[k]))
            for k in ("head_removal_share", "head_removal_share_v", "head_install_share", "head_install_share_v"):
                if (fr[k] is None) != (m[k] is None):
                    diff = math.inf
                elif fr[k] is not None:
                    diff = max(diff, abs(fr[k] - m[k]))
out["frame_values_max_abs_diff_vs_mine"] = diff

# the gates with the frozen functions; G6 against 020's locked states (the re-captures are not saved; drift 0.0)
book, ceilings = ri.run_gates(progs, schedules, saved, logits, program_states, program_states, geometry, ri.PRODUCTION)
gates = {**book.to_json(), "g1": []}
rec = state["calibration"]["gates"]
out["gates"] = {"equal_recorded_exactly": json.loads(canonical(gates)) == rec, "failures": gates["failures"],
                "ceilings_equal": json.loads(canonical(rc.json_safe(ceilings))) == state["calibration"]["ceilings"]}
if not out["gates"]["equal_recorded_exactly"]:
    out["gates"]["differences"] = {k: (gates.get(k), rec.get(k)) for k in set(gates) | set(rec) if json.loads(canonical(gates.get(k))) != rec.get(k)}

# per-run frozen Level-1, I4, readout (item 11), each G4 run, computed exactly as ri.frame_gates does
per_run = {}
for schedule in schedules:
    frame = schedule[0].frame
    fid, p_c, p_t = frame.frame_id, frame.p_c, frame.p_t
    st = program_states[fid]
    x3pc, x3pt, C, h6 = saved[f"{fid}/x3_pc"], saved[f"{fid}/x3_pt"], saved[f"{fid}/contrasts"], saved[f"{fid}/h6"]
    index = {run.tag: row for row, run in enumerate(schedule)}
    ref_row = index["REF"]
    reference_x3 = [x.double() for x in st.x3_all]
    rows16 = ul.reference_rows_017(progs.programs, st)
    full, i3 = {}, {}
    for label, cue_id in ri.PRODUCTION.cues:
        ctx = ul.pair_context(progs, frame, st, 581, int(cue_id), label, rows16)
        full[label] = ul.compose_dx3(progs, ctx, ri.b0c.FULL_MASK["coordinated"])
        row = index[f"PL@{label}"]
        i3[label] = ul.i3_error(full[label], {p_c: x3pc[row].double() - reference_x3[p_c], p_t: x3pt[row].double() - reference_x3[p_t]})
    for row, run in enumerate(schedule):
        arm, tag = run.arm, run.tag
        if not (arm.residual_edit or (arm.kind == "plain" and tag != "REF")):
            continue
        dx3_run = {p_c: x3pc[row].double() - reference_x3[p_c], p_t: x3pt[row].double() - reference_x3[p_t]}
        zero = torch.zeros(512, dtype=torch.float64)
        if arm.prompt == "ref":
            proxy = {p_c: x3pc[row].double() - reference_x3[p_c], p_t: zero}
        else:
            proxy = {p_c: full[run.label][p_c].double() + (x3pc[row].double() - x3pc[index[f"PL@{run.label}"]].double()), p_t: full[run.label][p_t].double()}
        ceiling = ul.contrast_of(progs, st, dx3_run).double()
        i4 = float((ul.contrast_of(progs, st, proxy).double() - ceiling).abs().max())
        dc_run = C[row].double() - C[ref_row].double()
        detail = progs.readout.level1_detail(st, dict(dx3_run), l4_heads=True)
        level1_vec = progs.readout.contrast(st, detail["dh6"], progs.nouns)[progs.scorable].double() - dc_run
        per_run[run.key] = {"frame": fid, "tag": tag, "arm": arm.name, "cue": run.label, "kind": arm.kind, "p_c": p_c, "level1": float(level1_vec.abs().max()),
                            "level1_noun": noun_keys[int(level1_vec.abs().argmax())], "level1_signed_at_max": float(level1_vec[int(level1_vec.abs().argmax())]),
                            "i4": i4, "i3_pair": i3[run.label], "dc_mean": float(dc_run.mean()),
                            "blocks_inf": {str(k): float(v.abs().max()) for k, v in detail["blocks"].items()}}
mine_c = json.loads((OUT / "c_per_run.json").read_text())["per_run"]
out["per_run_level1_vs_mine"] = {"n": len(per_run), "keys_equal": set(per_run) == {k for k, v in mine_c.items() if v["g4"]},
                                 "max_abs_diff": max(abs(per_run[k]["level1"] - mine_c[k]["level1"]) for k in per_run)}
dump("d_per_run_frozen.json", per_run)

environment = cand["environment"]
results_mine_pairs = ri.calibration_results([f.frame_id for f in frames], values_by_cue, table, mine_b["pairs"], book, environment, ri.PRODUCTION, extra_failures=[])
# the pairs above are my own (script b); recompute them with ri as the runner does, for the payload
pairs = {}
for schedule in schedules:
    fid = schedule[0].frame.frame_id
    index = {run.tag: row for row, run in enumerate(schedule)}
    x3 = saved[f"{fid}/x3_pc"]
    pairs[fid] = {label: ri.pair_quantities(x3[index[ri.tag_of("PL", label)]].double() - x3[index["REF"]].double(), geometry).record() for label in ri.PRODUCTION.cue_labels}
results = ri.calibration_results([f.frame_id for f in frames], values_by_cue, table, pairs, book, environment, ri.PRODUCTION, extra_failures=[])
payload = ri.calibration_payload(config=ri.PRODUCTION, confirmation_binding_=ri.confirmation_binding(confirmation, confirmation_sha), results=results, gates=gates,
                                 environment=environment, g1=block["g1"], keys_sha256=state["calibration"]["planned"]["sha256"], n_keys=len(keys),
                                 tensors_sha256=state["calibration"]["measurements"]["tensors_sha256"], protocol_code_commit=state["protocol_code_commit"])
out["payload"] = {"content_sha256": payload["content_sha256"], "candidate": cand["content_sha256"], "reproduced_bit_for_bit": payload["content_sha256"] == cand["content_sha256"],
                  "canonical_equal": pm.canonical_json(payload) == pm.canonical_json(cand)}
if not out["payload"]["reproduced_bit_for_bit"]:
    out["payload"]["differing_keys"] = sorted(k for k in set(payload) | set(cand) if json.loads(pm.canonical_json(payload.get(k))) != cand.get(k))
out["tau"] = results["tau"]
out["results_with_my_pairs_equal"] = pm.canonical_json(results_mine_pairs) == pm.canonical_json(results)
out["triggers"] = results["triggers"]
out["stopped"] = results["stopped"]
loaded = ri.load_calibration_101(OUTPUTS_101 / "candidate-calibration-v1.json", ri.PRODUCTION, confirmation, confirmation_sha)
out["load_calibration_101_accepts_candidate"] = loaded["content_sha256"] == cand["content_sha256"]
out["tolerances_equal_ri"] = cand["tolerances"] == json.loads(pm.canonical_json(ri.TOLERANCES))
out["module_blob_equal"] = cand["module_blob"] == ri.own_blob()
out["confirmation_binding_equal"] = cand["confirmation_101"] == ri.confirmation_binding(confirmation, confirmation_sha)
out["predicted"] = {f.frame_id: {"F_rem_hat": table["frames"][f.frame_id]["cues"]["two"]["fractions"]["F_rem"], "F_ins_hat": table["frames"][f.frame_id]["cues"]["two"]["fractions"]["F_ins"],
                                 "N_hat": table["frames"][f.frame_id]["cues"]["two"]["fractions"]["N"], "S_hat": table["frames"][f.frame_id]["cues"]["two"]["fractions"]["S"],
                                 "dc_full_hat": table["frames"][f.frame_id]["cues"]["two"]["dc_full"], "weak_hat": table["frames"][f.frame_id]["cues"]["two"]["weak"],
                                 "A_amp_hat": table["frames"][f.frame_id]["cues"]["two"]["fractions"]["A_amp"], "A_neg_hat": table["frames"][f.frame_id]["cues"]["two"]["fractions"]["A_neg"],
                                 "stratum": {c: {"dc_full_hat": table["frames"][f.frame_id]["cues"][c]["dc_full"], "weak_hat": table["frames"][f.frame_id]["cues"][c]["weak"]}
                                             for c in ("three", "many")}}
                    for f in frames}
out["guard"] = GUARD
out["write_attempts"] = WRITE_ATTEMPTS
out["forbidden_reads"] = FORBIDDEN_READS
out["git_index_mtime_unchanged"] = (ROOT / ".git/index").stat().st_mtime == index_mtime_before
dump("d_frozen_crosscheck.json", out)
print(json.dumps({k: v for k, v in out.items() if k != "predicted"}, indent=1, default=str))
