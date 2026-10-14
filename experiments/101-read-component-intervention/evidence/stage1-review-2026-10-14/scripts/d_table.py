"""Item 4 (and 6): the prediction table recomputed in this process from the stage-1 files and the weights, bit for bit
against table.json (G6); the barrier's own read-only methods run on the files as they are; confirm's pre-ledger checks
that can run before the commit. Weights only: every torch.nn.Module.__call__ and every forward entry point is refused,
and a token-sequence guard watches for any 101 prompt. Nothing in the repository is written (audit hook)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402  (first)

guard.self_test()
import gc  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler.models import PYTHIA_70M, seed_runtime  # noqa: E402

ROOT = guard.ROOT
EXP = ROOT / ri.EXPERIMENT_DIR
out: dict = {"checks": {}, "facts": {}, "timings_s": {}}


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"), flush=True)


def diff(a, b, path="", acc=None, limit=20):
    acc = [] if acc is None else acc
    if len(acc) >= limit:
        return acc
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                acc.append(f"{path}/{key}: missing on one side")
            else:
                diff(a[key], b[key], f"{path}/{key}", acc, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc.append(f"{path}: lengths {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", acc, limit)
    elif type(a) is not type(b) or a != b:
        acc.append(f"{path}: {a!r} vs {b!r}")
    return acc


# --- the runner module, loaded as the launcher loads it; then the forward guards --------------------------------------
spec = importlib.util.spec_from_file_location("experiment_101_runner", EXP / "run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
tokenizer = runner_module._load_tokenizer(PYTHIA_70M)  # tokenizer only: to build the forbidden token sequences
guard.build_forbidden(tokenizer)
guard.install_forward_guards()
check("forward_guards_installed", guard.RECORD["forward_guards_installed"] and guard.RECORD["forbidden_sequences"] >= 240,
      {"forbidden_sequences": guard.RECORD["forbidden_sequences"], "replaced": len(guard.RECORD["entry_points_replaced"])})


def refuse_writes(paths):
    raise PermissionError("review: no stage-1 write")


runner = runner_module.Runner(log=lambda message: print("  runner:", message, flush=True), write_check=refuse_writes)
check("runner_production_configuration", runner.config is ri.PRODUCTION and Path(runner.root).resolve() == ROOT)
t0 = time.time()
base = runner._base()
confirmation, confirmation_sha = runner._confirmation(base)
calibration, calibration_sha = runner._calibration(confirmation, confirmation_sha)
out["timings_s"]["base"] = round(time.time() - t0, 2)
lock, lock_paths = runner._installed_lock()
state = rd.load_results_state(runner.results_path)
check("state_inputs_equal_base_digests", state["inputs"] == {key: base.digests[key] for key in ri.DIGEST_KEYS})
check("lock_inputs_equal_base_digests", lock["inputs"] == dict(base.digests))
check("confirmation_sha_equals_lock_binding", lock["confirmation_101"]["file_sha256"] == confirmation_sha, confirmation_sha)
check("calibration_sha_equals_lock_binding", lock["calibration_101"]["file_sha256"] == calibration_sha, calibration_sha)
try:
    ri.assert_phase_allowed("confirm", state)
    check("confirm_phase_rule_allows_now", True)
except Exception as error:
    check("confirm_phase_rule_allows_now", False, str(error))

# --- validate_lock as confirm will call it, now (the tree holds the three untracked files) and simulated post-commit ---
try:
    runner._validated_lock(state, base, confirmation, confirmation_sha, calibration, calibration_sha, extra_tracked=list(runner.stage1_paths().values()))
    check("validated_lock_now_refuses_before_commit", False, "passed although the stage-1 files are untracked")
except Exception as error:
    check("validated_lock_now_refuses_before_commit", True, f"{type(error).__name__}: {error}")
head = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
changed_now = runner.changed_paths(lock["protocol_code_commit"])
out["facts"]["changed_paths_since_lock_protocol_commit"] = changed_now
simulated_changed = sorted(set(changed_now) | set(ri.STAGE1_PATHS))
check("stage1_paths_are_non_scientific", ri.scientific_changes(list(ri.STAGE1_PATHS)) == [] and ri.scientific_changes(simulated_changed) == [],
      ri.scientific_changes(simulated_changed))
evidence_probe = [f"{ri.EXPERIMENT_DIR}/evidence/stage1-2026-10-14/production.log", f"{ri.EXPERIMENT_DIR}/README.md", "outputs/experiment-101/results.json"]
risky_probe = [f"{ri.EXPERIMENT_DIR}/launch_confirm.py", f"{ri.EXPERIMENT_DIR}/stage1/notes.md", "src/neural_decompiler/read_intervention.py", "tests/test_read_intervention.py"]
out["facts"]["scientific_changes_probe"] = {"evidence_readme_outputs": ri.scientific_changes(evidence_probe), "risky": ri.scientific_changes(risky_probe)}
placeholder = ri.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
try:
    ri.validate_lock(lock, state=state, digests=base.digests, config=runner.config, confirmation=confirmation, confirmation_file_sha256=confirmation_sha,
                     calibration=calibration, calibration_file_sha256=calibration_sha, dependencies=placeholder, noun_keys=runner._noun_keys(base.inputs),
                     preregistration_text=lock_paths["preregistration"].read_text(encoding="utf-8"), git_state={"commit": head, "dirty": False},
                     tracked=True, changed_paths=simulated_changed)
    check("validate_lock_passes_simulated_post_commit", True, "git_state clean, tracked, changed paths + the three stage-1 paths")
except Exception as error:
    check("validate_lock_passes_simulated_post_commit", False, f"{type(error).__name__}: {error}")

# --- the runtime, the model (weights only), the programs and confirm's pre-barrier checks ------------------------------
runtime = runner._check_runtime(base.inputs.closure)
out["facts"]["runtime"] = runtime
check("runtime_equals_020_and_stage1", runtime == {**state["phases"]["stage1"]["runtime"]}, {"now": runtime, "stage1": state["phases"]["stage1"]["runtime"]})
seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
t0 = time.time()
model = runner.model_loader(PYTHIA_70M)
guard.seal(model)
progs = ul.ModelPrograms.from_model(model, base.inputs)
runner._model_dependencies(lock, model, progs)
check("model_parameters_and_embedding_digests_equal_lock", True)
noun_keys = runner._check_nouns(progs, base.inputs)
geometry = ri.read_geometry(base.inputs.lock_011)
runner._check_stage1_binding(lock, confirmation, progs, model, noun_keys)
check("stage1_binding_rederived_equals_lock", True)
g = runner._geometry_against_lock(lock, progs, geometry, [frame.frame_id for frame in confirmation.frames])
check("G1_and_geometry_digests_bitwise", g == {"bitwise_equal": True, "differing": []}, g)
del model
gc.collect()
out["timings_s"]["load_and_checks"] = round(time.time() - t0, 2)

# --- the barrier, exactly as confirm calls it (read-only), on the files as they are ------------------------------------
t0 = time.time()
try:
    barrier = runner._barrier(lock, confirmation, progs, geometry, noun_keys)
    check("barrier_passes_on_current_files", True, {"stage1_digest": barrier.record["digest"], "table_content_sha256": barrier.table["content_sha256"]})
except Exception as error:
    check("barrier_passes_on_current_files", False, f"{type(error).__name__}: {error}")
out["timings_s"]["barrier"] = round(time.time() - t0, 2)

# --- my own explicit recompute (the same path, spelled out), bit for bit and byte for byte -----------------------------
paths = runner.stage1_paths()
index = json.loads(paths["states_index"].read_text(encoding="utf-8"))
ul.verify_table_index(index, layout=lock["stage1"]["states"]["layout"], meta=lock["stage1"]["states"]["meta"])
locked = ri.states_from_blocks(ul.read_table(paths["states_data"], index), confirmation.frames)
check("state_digests_from_files_equal_record", {fid: rd.state_digest(e) for fid, e in sorted(locked.items())} == state["stage1"]["state_digests"])
states = {frame.frame_id: rd.state_from_locked(locked[frame.frame_id], frame) for frame in confirmation.frames}
committed_text = paths["table"].read_text(encoding="utf-8")
committed = json.loads(committed_text)
results = {}
for label, threads in (("threads_4_first", 4), ("threads_4_second", 4), ("threads_1", 1), ("threads_8", 8)):
    torch.set_num_threads(threads)
    t0 = time.time()
    with runner_module.pytest_free_guard():
        recomputed = ri.prediction_table(progs, confirmation.frames, states, geometry, ri.PRODUCTION, noun_keys)
    elapsed = round(time.time() - t0, 2)
    differences = diff(json.loads(pm.canonical_json(recomputed)), committed)
    results[label] = {"threads": threads, "content_sha256": recomputed["content_sha256"], "canonical_equal": pm.canonical_json(recomputed) == pm.canonical_json(committed),
                      "bytes_equal": (pm.canonical_json(recomputed) + "\n").encode("utf-8") == paths["table"].read_bytes(), "n_differences": len(differences),
                      "first_differences": differences[:5], "seconds": elapsed}
    print(f"  recompute {label}: {results[label]['content_sha256']} equal={results[label]['canonical_equal']} ({elapsed}s)", flush=True)
torch.set_num_threads(4)
out["facts"]["recomputes"] = results
check("table_recomputed_bit_for_bit_threads_4", results["threads_4_first"]["canonical_equal"] and results["threads_4_first"]["bytes_equal"]
      and results["threads_4_first"]["content_sha256"] == committed["content_sha256"] == state["stage1"]["table_content_sha256"], results["threads_4_first"])
check("table_recompute_deterministic_in_process", results["threads_4_second"]["canonical_equal"])
out["facts"]["thread_sensitivity"] = {label: r["canonical_equal"] for label, r in results.items()}
check("recompute_fractions_for_weak_off", recomputed["fractions_for_weak"] is False and committed["fractions_for_weak"] is False)
out["guard"] = guard.summary()
check("guard_never_fired", not guard.RECORD["token_guard_hits"] and guard.RECORD["module_call_refusals"] == 0 and not guard.RECORD["entry_point_refusals"]
      and not guard.RECORD["repo_write_refusals"] and not guard.RECORD["subprocess_refusals"], guard.summary())
out["all_ok"] = all(c["ok"] for c in out["checks"].values())
(guard.OUT / "d_table.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
print(json.dumps({k: out["facts"][k] for k in ("thread_sensitivity", "scientific_changes_probe", "runtime")}, indent=1))
print(json.dumps(out["timings_s"]))
print("ALL OK" if out["all_ok"] else "SOME CHECKS FAILED")
