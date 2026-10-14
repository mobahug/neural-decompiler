"""The frozen cross-check (run after b_independent.py): `ri` and the runner's own functions, under my guards.

Guards, installed in this order before any neural_decompiler import: the filesystem audit hook; the 240 forbidden 101
token sequences; `torch.nn.Module.__call__` refusing every call; then, after the imports, every capture and
intervention entry point (and alias) refusing; `models.load_model` replaced by a counting loader that seals the model
it returns. No runner phase other than `validate` is called. No forward pass is needed or run.
"""
from common import install_fs_guard

install_fs_guard()

import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from common import EVENTS, OUT as OUTDIR, ROOT, canonical, content_digest, design_candidates, forbidden_101_sequences, git, git_ok, install_model_guards, seal_model, \
    sha256_text, wrap_entry_points, write_json  # noqa: E402

SEQUENCES = forbidden_101_sequences()
SEQ_LIST = sorted(set(SEQUENCES.values()))
import torch  # noqa: E402

install_model_guards(SEQ_LIST)
import numpy as np  # noqa: E402

import mine  # noqa: E402
from neural_decompiler import attention_paths as ap  # noqa: E402
from neural_decompiler import capture, interventions, models  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import head_pattern as hp  # noqa: E402
from neural_decompiler import head_transport as ht  # noqa: E402
from neural_decompiler import layer_correction as lc  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_assembly as ra  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

ENTRY = {"pm.capture_prompt": pm.capture_prompt, "pm.run_patched": pm.run_patched, "capture.run_capture": capture.run_capture,
         "interventions.run_interventions": interventions.run_interventions, "rd.capture_frame_020": rd.capture_frame_020, "rd.measure_pair": rd.measure_pair,
         "rd.measure_pair_with_boundaries": rd.measure_pair_with_boundaries, "ri.run_arm": ri.run_arm, "ri.capture_reference": ri.capture_reference,
         "ri.measure_frame": ri.measure_frame, "ri.stage_runs": ri.stage_runs, "ra.capture_reference": ra.capture_reference, "ht.capture_reference": ht.capture_reference,
         "hp.capture_frame_017": hp.capture_frame_017, "ap.capture_frame_013": ap.capture_frame_013, "lc.capture_frame": lc.capture_frame, "cr.measure_rotated": cr.measure_rotated}
wrap_entry_points(SEQ_LIST, ENTRY)
_original_load = models.load_model


def counted_sealed_load(spec, *args, **kwargs):
    EVENTS["model_loads"] += 1
    model = _original_load(spec, *args, **kwargs)
    seal_model(model)
    return model


models.load_model = counted_sealed_load
spec = importlib.util.spec_from_file_location("experiment_101_runner", ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
assert runner_module.load_model is counted_sealed_load
wrap_entry_points(SEQ_LIST, ENTRY)

OUT: dict = {"checks": {}, "numbers": {}}
problems: list[str] = []
LOGS: list[str] = []


def check(name, ok, detail=None):
    OUT["checks"][name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
    if not ok:
        problems.append(name)
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or detail is None else f"  -> {json.dumps(detail, default=str)[:800]}"), flush=True)


def my_tracked(path) -> bool:
    return git_ok("ls-files", "--error-unmatch", str(Path(path).resolve().relative_to(ROOT)))


def my_git_state():
    return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain").strip())}


def my_changed_paths(commit):
    if not git_ok("merge-base", "--is-ancestor", commit, "HEAD"):
        return None
    return [line.strip() for line in git("diff", "--no-renames", "--name-only", commit, "HEAD").splitlines() if line.strip()]


OUT["torch_num_threads"] = torch.get_num_threads()
runner = runner_module.Runner(tracked=my_tracked, git_state=my_git_state, changed_paths=my_changed_paths, log=LOGS.append)
check("runner_is_production", runner.config is ri.PRODUCTION)
lock_path = ROOT / "outputs/experiment-101/candidate-lock.json"
lock_bytes = lock_path.read_bytes()
lock = json.loads(lock_bytes)
prereg_text = (ROOT / "outputs/experiment-101/candidate-preregistration.md").read_text(encoding="utf-8")

# --- validate (allowed), and the base.
t0 = time.time()
status = runner.validate()
OUT["validate"] = {"exit": status, "log": list(LOGS), "seconds": round(time.time() - t0, 1)}
check("validate_exits_0", status == 0, LOGS[-1:])
base = runner._base()
wrap_entry_points(SEQ_LIST, ENTRY)  # modules imported by the frozen inputs' loaders
check("inputs_equal_base_digests", lock["inputs"] == {k: base.digests[k] for k in ri.DIGEST_KEYS} and set(lock["inputs"]) == set(ri.DIGEST_KEYS),
      sorted(k for k in set(lock["inputs"]) | set(base.digests) if lock["inputs"].get(k) != base.digests.get(k)))
check("lock_digest_keys_42", len(ri.DIGEST_KEYS) == 42 == len(lock["inputs"]))
OUT["numbers"]["forbidden_keys"] = len(base.forbidden)
confirmation, conf_sha = runner._confirmation(base)
calibration, cal_sha = runner._calibration(confirmation, conf_sha)
check("confirmation_binding_frozen", lock["confirmation_101"] == ri.confirmation_binding(confirmation, conf_sha))
check("calibration_binding_frozen", lock["calibration_101"] == ri.calibration_binding(calibration, cal_sha))
check("frame_candidates_equal_design", list(ri.FRAME_CANDIDATES) == design_candidates())
check("untagged_101_keys_not_forbidden", not (confirmation.untagged_keys() & base.forbidden) and len(confirmation.untagged_keys()) == 144)

# --- the state and the phase rules (no phase is run).
state = rd.load_results_state(ROOT / "outputs/experiment-101/results.json")
rules = {}
for phase in ("calibrate", "lock", "stage1", "confirm", "report"):
    try:
        ri.assert_phase_allowed(phase, state)
        rules[phase] = "allowed"
    except ri.PhaseError as error:
        rules[phase] = f"refused: {error}"
OUT["phase_rules"] = rules
check("second_lock_refused_by_assert_phase_allowed", rules["lock"].startswith("refused") and rules["stage1"] == "allowed" and rules["confirm"].startswith("refused")
      and rules["report"].startswith("refused") and rules["calibrate"].startswith("refused"), rules)
try:
    runner._state_for("lock", base.digests)
    second = "allowed"
except ri.PhaseError as error:
    second = f"refused: {error}"
check("second_lock_refused_through_runner_state_for", second.startswith("refused: lock already ran"), second)
check("state_inputs_equal_base", state["inputs"] == {k: base.digests[k] for k in ri.DIGEST_KEYS})
check("state_versions_equal_now", dict(state["versions"]) == dict(runner.versions()), [state["versions"], runner.versions()])

# --- module, configuration, wording, constants, tolerances (the lock against ri's constants).
check("own_blob", lock["module"] == {"path": "src/neural_decompiler/read_intervention.py", "blob": ri.own_blob()})
check("frozen_blobs", ri.assert_frozen_blobs() == ri.FROZEN_BLOBS == lock["module_blobs"])
check("configuration_equals_PRODUCTION", lock["configuration"] == json.loads(pm.canonical_json(ri.PRODUCTION.to_json())))
check("statistics_text", lock["statistics"] == ri.STATISTICS)
check("constants", lock["constants"] == json.loads(pm.canonical_json(ri.statistics_constants())))
check("tolerances_equal_ri_TOLERANCES (rule 1)", lock["tolerances"] == json.loads(pm.canonical_json(ri.TOLERANCES)))
check("semantics", lock["semantics"] == json.loads(pm.canonical_json(ri.SEMANTICS)))
check("outcome", lock["outcome"] == {"table": [list(r) for r in ri.OUTCOME_TABLE], "labels": list(ri.OUTCOMES)})
check("run_and_cue_order", lock["run_order"] == list(ri.PRODUCTION.tags()) and lock["cue_order"] == list(ri.PRODUCTION.cue_labels))
check("inherited_025", lock["inherited_025"] == {"path": ri.INHERITED_025_PATH, **ri.INHERITED_025})

# --- the weights: load once, sealed; digests; then the model is deleted.
t0 = time.time()
model = models.load_model(models.PYTHIA_70M)
progs = ul.ModelPrograms.from_model(model, base.inputs)
wrap_entry_points(SEQ_LIST, ENTRY)
h = __import__("hashlib").sha256()
named = sorted(model.named_parameters(), key=lambda item: item[0])
for name, parameter in named:
    array = parameter.detach().cpu().contiguous().to(torch.float32).numpy().astype("<f4", copy=False)
    h.update(name.encode("utf-8") + b"|" + json.dumps(list(array.shape)).encode("ascii") + b"|" + array.tobytes())
mine_params = h.hexdigest()
frozen_params = rr.parameters_digest(model)
W_E = progs.weights.W_E.detach().cpu().contiguous().to(torch.float32).numpy().astype("<f4", copy=False)
mine_emb = __import__("hashlib").sha256(json.dumps(list(W_E.shape)).encode("ascii") + b"|<f4|" + W_E.tobytes()).hexdigest()
OUT["numbers"]["parameters"] = {"mine": mine_params, "frozen": frozen_params, "lock": lock["dependencies"]["model"]["parameters_sha256"], "n_named": len(named),
                                "names_first": [n for n, _ in named[:6]]}
check("parameters_sha256_mine_equals_frozen_equals_lock", mine_params == frozen_params == lock["dependencies"]["model"]["parameters_sha256"])
check("embedding_sha256_mine_equals_frozen_equals_lock", mine_emb == rr.embedding_digest(progs.weights.W_E) == lock["dependencies"]["model"]["embedding_sha256"])
# the bridge's parameters against the raw checkpoint: every bridge tensor equals one raw tensor bit for bit
raw = mine.raw_weights()
raw_by_bytes = {}
for name, arr in raw.items():
    raw_by_bytes.setdefault((tuple(arr.shape), np.ascontiguousarray(arr, dtype="<f4").tobytes()), []).append(name)
unmatched = [n for n, p in named if (tuple(p.shape), p.detach().cpu().contiguous().to(torch.float32).numpy().astype("<f4", copy=False).tobytes()) not in raw_by_bytes]
OUT["numbers"]["bridge_params_matching_raw_tensors"] = {"bridge": len(named), "raw": len(raw), "unmatched": unmatched[:10]}
check("bridge_parameters_are_raw_checkpoint_tensors", not unmatched, unmatched[:10])
try:
    runner._model_dependencies(lock, model, progs)
    check("runner_model_dependencies_accepts_lock", True)
except Exception as error:  # noqa: BLE001
    check("runner_model_dependencies_accepts_lock", False, str(error))
n_heads = int(model.cfg.n_heads)
d_model_from_WE = int(progs.weights.W_E.shape[1])
del model
OUT["numbers"]["model_seconds"] = round(time.time() - t0, 1)

noun_keys = runner._check_nouns(progs, base.inputs)
check("noun_keys_frozen_equal_lock", noun_keys == lock["noun_keys"] == runner._noun_keys(base.inputs) and len(noun_keys) == 79 and len(progs.nouns.nouns) == 80)

# --- the geometry: ri against mine, numerically; the block twice; against the lock bit for bit.
geometry = ri.read_geometry(base.inputs.lock_011)
l011 = mine.lock_011()
wc_np, norm_np = mine.wc_numpy(l011["read_weight"])
wc_t, norm_t, what_t = mine.wc_torch(l011["read_weight"])
OUT["numbers"]["w_c_max_abs"] = {"ri_vs_mine_torch_arith": float((geometry.w_c - wc_t).abs().max()), "ri_vs_mine_numpy_fsum": float(np.max(np.abs(geometry.w_c.numpy() - wc_np))),
                                 "w_hat_ri_vs_mine": float((geometry.w_hat - what_t).abs().max()), "norm_ri_minus_mine": geometry.norm - norm_np,
                                 "norm2_ri_minus_mine_sq": geometry.norm2 - norm_np ** 2, "d_T_ri_vs_011": float((geometry.d_T - torch.tensor(l011["axes_vectors"]["T"], dtype=torch.float64)).abs().max())}
check("read_geometry_equals_mine", OUT["numbers"]["w_c_max_abs"]["ri_vs_mine_torch_arith"] == 0.0 and OUT["numbers"]["w_c_max_abs"]["ri_vs_mine_numpy_fsum"] <= 1e-16,
      OUT["numbers"]["w_c_max_abs"])
raw_rw = mine.read_weight_from_raw(raw, np.array(l011["axes_vectors"]["T"], dtype=np.float64))["read_weight"]
frozen_rw = ra.read_weight(progs.head, progs.axis_T).double().numpy()
OUT["numbers"]["read_weight_frozen_vs_raw_mine"] = float(np.max(np.abs(frozen_rw - raw_rw)))
OUT["numbers"]["g1_frozen_value"] = ri.g1_read_weight(progs.head, progs.axis_T, geometry)
frame_ids = [frame.frame_id for frame in confirmation.frames]
with runner_module.pytest_free_guard():
    block = ri.geometry_block(geometry, progs.head, progs.axis_T, frame_ids, ri.PRODUCTION.k_iso)
    again = ri.geometry_block(geometry, progs.head, progs.axis_T, frame_ids, ri.PRODUCTION.k_iso)
check("geometry_block_reproduces_twice_bit_for_bit", pm.canonical_json(ri.encode_nonfinite(block)) == pm.canonical_json(ri.encode_nonfinite(again)))
vg = ri.verify_geometry_against_lock(block, lock)
check("geometry_block_equals_lock_bit_for_bit", vg["bitwise_equal"], vg)
try:
    ri.enforce_g1(block)
    check("enforce_g1_passes", True)
except Exception as error:  # noqa: BLE001
    check("enforce_g1_passes", False, str(error))
worst_iso = 0.0
for fid in frame_ids:
    for j in range(1, 9):
        worst_iso = max(worst_iso, float((ri.iso_direction(geometry, fid, j) - mine.iso_torch(what_t, fid, j)).abs().max()),
                        float(np.max(np.abs(ri.iso_direction(geometry, fid, j).numpy() - mine.iso_numpy(wc_np / norm_np, fid, j)))))
OUT["numbers"]["iso_max_abs_ri_vs_mine_any_arith"] = worst_iso
check("iso_direction_equals_mine", worst_iso <= 1e-15, worst_iso)

# --- the exposed prediction table (spent data; fractions_for_weak on, as calibration computed it), in this process.
exposed = runner._exposed_frames(base.inputs)
locked020 = base.inputs.closure["exploration"]["locked_states"]
states = {frame.frame_id: rd.state_from_locked(locked020[frame.frame_id], frame) for frame in exposed}
t0 = time.time()
with runner_module.pytest_free_guard():
    table_true = ri.prediction_table(progs, exposed, states, geometry, ri.PRODUCTION, noun_keys, fractions_for_weak=True)
OUT["numbers"]["exposed_table_true"] = {"content_sha256": table_true["content_sha256"], "calibration": calibration["table_content_sha256"], "seconds": round(time.time() - t0, 1)}
check("exposed_table_reproduces_calibration_digest", table_true["content_sha256"] == calibration["table_content_sha256"] == "ce055e09c73613b74caae31222192f95528e6f58e101469917138786b2d9758f",
      OUT["numbers"]["exposed_table_true"])
# the confirm setting (fractions_for_weak off) on the same spent states: the bound metadata and the JSON round trip the barrier relies on
with runner_module.pytest_free_guard():
    table_false = ri.prediction_table(progs, exposed, states, geometry, ri.PRODUCTION, noun_keys)
round_trip = json.loads(pm.canonical_json(table_false) + "\n")
check("table_false_meta_equals_lock_binding_after_round_trip", {k: round_trip.get(k) for k in lock["stage1"]["table"]} == lock["stage1"]["table"])
check("table_false_round_trip_canonical_equal", pm.canonical_json(round_trip) == pm.canonical_json(table_false) and round_trip["content_sha256"] == rc.content_digest(round_trip))
OUT["numbers"]["exposed_table_false_digest"] = table_false["content_sha256"]
OUT["numbers"]["exposed_predicted_weak"] = sum(1 for f in table_false["frames"].values() for c in f["cues"].values() if c["weak"])

# --- the stage-1 binding, the dependencies, the lock rebuilt and the preregistration rendered.
stage1_lock_style = ri.stage1_binding(confirmation.frames, ri.PRODUCTION, d_model=geometry.d_model, n_heads=n_heads, n_nouns=len(progs.nouns.nouns), noun_keys=noun_keys)
stage1_stage_style = ri.stage1_binding(confirmation.frames, ri.PRODUCTION, d_model=d_model_from_WE, n_heads=n_heads, n_nouns=len(progs.nouns.nouns), noun_keys=noun_keys)
my_layout = mine.my_layout([json.loads(json.dumps(f.to_dict())) for f in confirmation.frames], 512, 8, 80)
check("stage1_binding_frozen_equals_lock", pm.canonical_json(stage1_lock_style) == pm.canonical_json(lock["stage1"]) == pm.canonical_json(stage1_stage_style))
check("stage1_layout_frozen_equals_mine", stage1_lock_style["states"]["layout"] == my_layout)
deps = ri.scientific_dependencies(base.inputs, parameters_sha256=frozen_params, embedding_sha256=rr.embedding_digest(progs.weights.W_E))
check("dependencies_frozen_equal_lock", json.loads(pm.canonical_json(deps)) == lock["dependencies"])
rebuilt = ri.build_lock(run_id=lock["run_id"], protocol_code_commit=lock["protocol_code_commit"], digests=base.digests, config=ri.PRODUCTION, confirmation=confirmation,
                        confirmation_file_sha256=conf_sha, calibration=calibration, calibration_file_sha256=cal_sha, geometry=block, dependencies=deps,
                        noun_keys=noun_keys, stage1=stage1_lock_style)
check("build_lock_reproduces_candidate_byte_for_byte", (pm.canonical_json(rebuilt) + "\n").encode("utf-8") == lock_bytes, rebuilt["content_sha256"])
check("render_preregistration_byte_for_byte", ri.render_preregistration(lock) == prereg_text and ri.render_preregistration(rebuilt) == prereg_text
      and sha256_text(prereg_text) == state["lock"]["preregistration_sha256"])

# --- validate_lock on the candidate as if installed and committed (simulated git facts), with negative controls.
placeholder = ri.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
installed_changes = ["experiments/101-read-component-intervention/preregistration-lock.json", "experiments/101-read-component-intervention/preregistration.md",
                     "experiments/101-read-component-intervention/README.md", "experiments/101-read-component-intervention/evidence/lock-2026-10-14/lock_run.json",
                     "experiments/101-read-component-intervention/evidence/lock-review-2026-10-14/REVIEW.md", "README.md"]


def validate(candidate, *, text=prereg_text, st=state, changed=installed_changes, dirty=False, tracked=True, nouns=None):
    try:
        ri.validate_lock(candidate, state=st, digests=base.digests, config=ri.PRODUCTION, confirmation=confirmation, confirmation_file_sha256=conf_sha, calibration=calibration,
                         calibration_file_sha256=cal_sha, dependencies=placeholder, noun_keys=nouns or noun_keys, preregistration_text=text,
                         git_state={"commit": "x" * 40, "dirty": dirty}, tracked=tracked, changed_paths=changed)
        return "accepted"
    except ri.PhaseError as error:
        return f"refused: {error}"


sims = {"as_installed": validate(lock)}
sims["scientific_change"] = validate(lock, changed=installed_changes + ["src/neural_decompiler/read_intervention.py"])
sims["run_py_change"] = validate(lock, changed=installed_changes + ["experiments/101-read-component-intervention/run.py"])
sims["stage1_files_committed"] = validate(lock, changed=installed_changes + list(ri.STAGE1_PATHS))
sims["dirty_tree"] = validate(lock, dirty=True)
sims["untracked"] = validate(lock, tracked=False)
sims["lock_commit_not_ancestor"] = validate(lock, changed=None)
sims["preregistration_edited"] = validate(lock, text=prereg_text.replace("at least 27 of 36", "at least 26 of 36"))
widened = copy.deepcopy(lock)
widened["tolerances"]["G4_level1"] = 0.03
widened["content_sha256"] = rc.content_digest(widened)
st_w = copy.deepcopy(state)
st_w["lock"]["content_sha256"] = widened["content_sha256"]
sims["tolerance_widened_and_redigested"] = validate(widened, text=ri.render_preregistration(widened), st={**st_w, "lock": {**st_w["lock"], "preregistration_sha256": sha256_text(ri.render_preregistration(widened))}})
sims["tolerance_widened_state_not_updated"] = validate(widened, text=ri.render_preregistration(widened))
sims["noun_order_changed"] = validate(lock, nouns=list(reversed(noun_keys)))
OUT["validate_lock_simulations"] = sims
check("validate_lock_accepts_candidate_as_installed", sims["as_installed"] == "accepted", sims["as_installed"])
check("validate_lock_refuses_every_negative_control", all(v.startswith("refused") for k, v in sims.items() if k not in ("as_installed", "stage1_files_committed")), sims)
check("validate_lock_allows_committed_stage1_files", sims["stage1_files_committed"] == "accepted", sims["stage1_files_committed"])

# --- the stage-1 write path against the lock's layout, with SYNTHETIC values (no model output), in this review's scratch directory.
scratch = OUTDIR / "stage1_writer_test"
scratch.mkdir(parents=True, exist_ok=True)
for name in ("states.f64", "states.json"):
    if (scratch / name).exists():
        (scratch / name).unlink()
generator = torch.Generator().manual_seed(101)
synthetic = {}
for frame in confirmation.frames:
    keys = frame.p_t + 1
    t = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)  # noqa: E731
    synthetic[frame.frame_id] = {"p_c": frame.p_c, "p_t": frame.p_t, "x1_all": t(keys, 512).tolist(), "x2_all": t(keys, 512).tolist(), "x3_all": t(keys, 512).tolist(),
                                 "x4_all": t(keys, 512).tolist(), "x5_all": t(keys, 512).tolist(), "h6": t(512).tolist(), "rows4": t(8, keys).tolist(),
                                 "rows5": t(8, keys).tolist(), "c_ref": t(80).tolist()}
index = ul.write_table(scratch / "states.f64", scratch / "states.json", ri.states_blocks(synthetic, confirmation.frames), lock["stage1"]["states"]["meta"])
try:
    ul.verify_table_index(index, layout=lock["stage1"]["states"]["layout"], meta=lock["stage1"]["states"]["meta"])
    reread = json.loads((scratch / "states.json").read_text(encoding="utf-8"))
    ul.verify_table_index(reread, layout=lock["stage1"]["states"]["layout"], meta=lock["stage1"]["states"]["meta"])
    back = ri.states_from_blocks(ul.read_table(scratch / "states.f64", reread), confirmation.frames)
    same = {fid: rd.state_digest(entry) for fid, entry in sorted(back.items())} == {fid: rd.state_digest(entry) for fid, entry in sorted(synthetic.items())}
    rebuilt_states = {frame.frame_id: rd.state_from_locked(back[frame.frame_id], frame) for frame in confirmation.frames}
    check("stage1_writer_and_reader_accept_the_locks_layout (synthetic values)", same and len(rebuilt_states) == 36)
except Exception as error:  # noqa: BLE001
    check("stage1_writer_and_reader_accept_the_locks_layout (synthetic values)", False, f"{type(error).__name__}: {error}")

OUT["fs_events"] = EVENTS
OUT["problems"] = problems
write_json("c_frozen.json", OUT)
print(json.dumps(OUT["numbers"], indent=1, default=str))
print("phase rules:", json.dumps(rules, indent=1))
print("validate_lock simulations:", json.dumps(sims, indent=1))
print("fs/model events:", json.dumps(EVENTS, default=str))
print("PROBLEMS:", problems)
