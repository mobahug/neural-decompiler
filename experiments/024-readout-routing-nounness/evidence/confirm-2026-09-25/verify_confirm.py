"""Post-confirm verification of Experiment 024 (read-only; no model, no prompt; report NOT run; nothing repaired).

It reads the launcher's record, the prompt log, the results state and the saved stage-2 measurements, and reports:
the invocation; the module-blob assertion; the environment; I7; the accounting (ledger, prompt log and manifest); the
measurement artifact (path, size, sha256, tensor digests re-derived from disk); the identity gates; the primary ρ; the
exact E–N guard; the outcome; the per-class MSE; the descriptive records; incidents; the phases; the state digest; the
repository. The per-cue MSE, ρ, the guard and the outcome are re-derived from the saved measurements with the canonical
functions and compared bit for bit with the recorded result. This is a consistency check, not a new analysis.
"""
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
REFUSED = []


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if not writing:
        return
    raw = os.fsdecode(args[0])
    if not os.path.isabs(raw):
        return
    path = Path(raw).resolve()
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
        REFUSED.append(str(path))
        raise PermissionError(f"the post-confirm verification may not write into the repository: {path}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
failures = []
summary = {}


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, env=GIT_ENV).stdout.strip()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def section(title):
    print("=" * 100 + f"\n{title}", flush=True)


# -- the launcher's record ------------------------------------------------------------------------------------------------
section("the invocation")
record = json.loads((HERE / "confirm_record.json").read_text(encoding="utf-8"))
print(f"launcher: started {record['launcher_started_at']}; confirm {record.get('confirm_started_at')} → {record.get('confirm_ended_at')}; exit status "
      f"{record.get('exit_status')}; error {record.get('error')}", flush=True)
print(f"counts {record['counts']}", flush=True)
print(f"events {record['events']}", flush=True)
print(f"launcher exit file: {(HERE / 'confirm.exit').read_text().strip() if (HERE / 'confirm.exit').exists() else 'missing'}", flush=True)
check("confirm invoked exactly once by the launcher; one model load", record["counts"]["confirm_invocations"] == 1 and record["counts"]["load_model"] == 1)
check("the launcher's sentinel exists (a second launch is refused)", (HERE / "CONFIRM_LAUNCHED").exists(), (HERE / "CONFIRM_LAUNCHED").read_text().strip())
check("no repository write outside outputs/experiment-024; no forbidden read; no hook error",
      record["events"]["repository_writes_outside_outputs"] == [] and record["events"]["forbidden_reads"] == [] and record["events"]["hook_errors"] == 0)
assertion = record["module_blob_assertion"]
check("the direct module-blob assertion agreed before the load", assertion["agree"] and assertion["asserted_at"] < record["confirm_started_at"],
      {key: assertion[key] for key in ("expected_lock_blob", "observed_rr_own_blob", "git_hash_object_of_imported_file", "imported_path")})
environment = record["environment"]
check("environment: 4 threads, offline, the pinned checkpoint, af160ce clean",
      environment["torch_num_threads"] == 4 and environment["env"]["HF_HUB_OFFLINE"] == "1" and environment["checkpoint"]["sha256"].startswith("3da38833")
      and environment["git"]["head"].startswith("af160ce") and environment["git"]["status_porcelain"] == "", environment["packages"])
summary["invocation"] = {key: record.get(key) for key in ("confirm_started_at", "confirm_ended_at", "exit_status", "error")}

# -- the state ------------------------------------------------------------------------------------------------------------
section("the results state")
state_path = ROOT / "outputs/experiment-024/results.json"
state = rd.load_results_state(state_path)
phases = {name: entry["status"] for name, entry in state["phases"].items()}
confirm_phase = state["phases"]["confirm"]
print(f"state sha256 {state['state_sha256']}; file sha256 {sha(state_path.read_bytes())}; phases {phases}", flush=True)
print(f"confirm phase: { {key: value for key, value in confirm_phase.items() if key != 'I7'} }", flush=True)
confirmation_state = state["confirmation"] or {}
incident = confirmation_state.get("incident")
phase_incidents = confirm_phase.get("incidents")
print(f"incident: {incident}; phase incidents: {phase_incidents}", flush=True)
summary["state"] = {"state_sha256": state["state_sha256"], "file_sha256": sha(state_path.read_bytes()), "phases": phases, "incident": incident,
                    "phase_incidents": phase_incidents}
check("report not started", phases["report"] == "not_started")
try:
    rr.assert_phase_allowed("confirm", state)
    check("a second confirm is refused by the phase rule", False)
except rr.PhaseError as error:
    check("a second confirm is refused by the phase rule", True, error)
I7 = confirm_phase.get("I7")
check("I7 reproduced bit for bit before any prompt", bool(I7) and I7.get("bitwise_equal") is True and I7.get("differing") == [], I7)
check("confirm ran at af160ce", str(confirm_phase.get("confirm_commit", "")).startswith("af160ce"), confirm_phase.get("confirm_commit"))

# -- the runner's inputs (read-only) ----------------------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("experiment_024_runner", ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)


def refuse(*args, **kwargs):
    raise RuntimeError("no model in the post-confirm verification")


runner = runner_module.Runner(log=lambda message: None, model_loader=refuse, tokenizer_loader=refuse)
base = runner._base()
confirmation, _ = runner._confirmation(base)
lock = json.loads((ROOT / rr.LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
manifest = confirmation.manifest()["S2-TARGET"]

# -- the accounting -----------------------------------------------------------------------------------------------------------
section("the prompt accounting")
ledger = state["executed_prompt_keys"]
log_lines = [json.loads(line) for line in (HERE / "confirm_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
log_keys = [line["key"] for line in log_lines]
accounting = confirmation_state.get("accounting")
print(f"state accounting: {accounting}", flush=True)
counts_log = Counter(log_keys)
check("the ledger holds exactly the 4,320 manifest keys (no missing, duplicate, extra or outside key)",
      len(ledger) == len(set(ledger)) == 4_320 and set(ledger) == set(manifest), f"{len(ledger)} keys")
check("the launcher's prompt log: 4,320 captures, each manifest key exactly once, numbered 1…4,320",
      len(log_keys) == 4_320 and max(counts_log.values()) == 1 and set(log_keys) == set(manifest) and [line["n"] for line in log_lines] == list(range(1, 4_321)),
      f"{len(log_keys)} lines, {len(counts_log)} distinct")
check("capture count == the prompt log == 4,320 (run_capture is capture_prompt's own inner call); no patched run or intervention",
      record["counts"]["capture_prompt"] == 4_320 and record["counts"].get("run_capture", 0) in (0, 4_320) and record["counts"]["run_patched"] == 0
      and record["counts"]["run_interventions"] == 0, record["counts"])
check("the runner's accounting is equal (manifest 4,320, executed 4,320, ledger 4,320)",
      accounting == {"manifest": 4_320, "executed": 4_320, "ledger": 4_320, "equal": True}, accounting)
digests = {"manifest_sha256": pm.sha256_text(pm.canonical_json({"S2-TARGET": manifest})), "ledger_sorted_sha256": pm.sha256_text(pm.canonical_json(sorted(ledger))),
           "prompt_log_sequence_sha256": pm.sha256_text(pm.canonical_json(log_keys)), "manifest_sorted_sha256": pm.sha256_text(pm.canonical_json(sorted(manifest)))}
print(f"accounting digests: {digests}", flush=True)
check("ledger (sorted) digest == manifest (sorted) digest", digests["ledger_sorted_sha256"] == digests["manifest_sorted_sha256"])
check("the manifest digest is the lock's fcc437fc…", digests["manifest_sha256"] == lock["confirmation_024"]["manifest_sha256"])
print(f"descriptive: the prompt log follows the manifest's stored order: {log_keys == manifest}", flush=True)
check("zero collision of the executed keys with the 39,312 spent keys", not (set(ledger) & base.forbidden))
print(f"noun ledger: {len(state['executed_noun_keys'])} nouns", flush=True)
first, last = log_lines[0], log_lines[-1]
print(f"prompt log: first {first['key']} at {first['t']:.3f}, last {last['key']} at {last['t']:.3f} ({last['t'] - first['t']:.1f} s)", flush=True)
summary["accounting"] = {"state": accounting, "digests": digests, "prompt_log_lines": len(log_keys), "noun_ledger": len(state["executed_noun_keys"])}

# -- the measurement artifact -------------------------------------------------------------------------------------------------
section("the stage-2 measurement artifact")
stage2 = confirmation_state.get("stage2") or {}
stage2_path = Path(stage2.get("path", ROOT / "outputs/experiment-024/stage2-measurements.pt"))
raw = stage2_path.read_bytes()
tensors = torch.load(stage2_path)
recomputed = {key: rc.tensor_digest(value) for key, value in tensors.items()}
shapes = {key: list(value.shape) for key, value in tensors.items()}
print(f"path {stage2_path}; bytes {len(raw)}; file sha256 {sha(raw)}; n_executed {stage2.get('n_executed')}", flush=True)
for key in sorted(tensors):
    print(f"  {key}: shape {shapes[key]} dtype {tensors[key].dtype} digest {recomputed[key]}", flush=True)
check("the tensor digests re-derived from disk == the state's", recomputed == stage2.get("tensors_sha256"))
units = rr.target_units(confirmation)
pairs = {group: len(rows) for group, rows in units.pairs.items()}
check("pairs: 4,320 in total across the groups; every tensor row count matches its group",
      sum(pairs.values()) == 4_320 and all(shapes[key][0] == pairs[key.split("/")[1]] for key in tensors if key.startswith("Y1/")), pairs)
check("every saved measurement is finite", all(bool(torch.isfinite(value).all()) for value in tensors.values() if value.is_floating_point()))
summary["stage2"] = {"path": str(stage2_path), "bytes": len(raw), "file_sha256": sha(raw), "tensors_sha256": recomputed, "pairs": pairs, "n_executed": stage2.get("n_executed")}

# -- the identity gates ---------------------------------------------------------------------------------------------------------
section("the validity and identity checks")
c_check = confirmation_state.get("c_recompute")
gates = confirmation_state.get("gates")
print(f"C recomputed from the saved Δx3: {c_check}", flush=True)
for name in ("I1", "I3", "I4"):
    if gates and name in gates:
        print(f"  {name}: max {gates[name]['max']!r} at {gates[name]['at']} (tolerance {rr.TOLERANCES[name]:.0e})", flush=True)
check("C recomputed from the saved Δx3 bit for bit, all 4,320 pairs", bool(c_check) and c_check.get("bitwise_equal") is True and c_check.get("pairs") == 4_320, c_check)
check("I1, I3 and I4 within their tolerances", bool(gates) and all(gates[name]["max"] is not None and gates[name]["max"] <= rr.TOLERANCES[name] for name in ("I1", "I3", "I4")))
summary["gates"] = {"c_recompute": c_check, "gates": gates, "tolerances": {name: rr.TOLERANCES[name] for name in ("I1", "I3", "I4", "spearman", "en_float")}}

# -- the result ---------------------------------------------------------------------------------------------------------------
section("the result (re-derived from the saved measurements with the canonical functions, compared bit for bit)")
results = confirmation_state.get("results")
if results is None:
    check("a result is recorded", False, "no result in the state")
else:
    cells = rr.fresh_cue_cells(units, tensors, confirmation)
    again = rr.score(cells, confirmation, lock, rr.PRODUCTION)
    check("per-cue MSE, ρ, the guard and the outcome re-derived from disk == the recorded result, bit for bit",
          json.loads(pm.canonical_json(again)) == json.loads(pm.canonical_json(results)))
    primary, guard, label = results["primary"], results["guard"], results["outcome"]["label"]
    threshold = primary["effective_threshold"]["value"]
    print(f"PRIMARY: ρ = {primary['rho']!r} (Spearman across 40 cues; cross-check difference {results['checks']['spearman_direct_difference']!r}); locked threshold "
          f"{threshold!r} (bound by {primary['effective_threshold']['binds']}); F_ρ {primary['F_rho']!r}; result {primary['result']}", flush=True)
    check("the primary comparison is against the locked threshold only", threshold == 0.3136960600375234 == lock["primary"]["effective_threshold"]["value"]
          and primary["result"] == rr.classify_primary(primary["rho"], lock["primary"]["F_rho"], lock["primary"]["null_975"]))
    print(f"E–N GUARD: D_EN = {guard.get('D_EN')!r} (exact {guard.get('D_EN_exact')}); K = {guard.get('K')} of {guard.get('assignments')} "
          f"(K/12,870 = {guard.get('p_value')!r}); passes iff K ≤ {guard.get('max_upper')}; result {guard['result']}; ratio of geometric means "
          f"{guard.get('ratio_of_geometric_means')!r}; float check {guard.get('checks')}", flush=True)
    print(f"OUTCOME: {label}", flush=True)
    check("the outcome is the installed function of the two results", label == rr.outcome(primary["result"], guard["result"]))
    per_cue = results["per_cue"]
    print("per-cue (frozen order): class word nounness MSE log MSE predicted log MSE", flush=True)
    for cue in per_cue:
        print(f"  {cue['class']} {cue['word']:<9} {cue['nounness']:+.6f} {cue['mse']:.6f} {cue['log_mse']:+.6f} {cue['predicted_log_mse']:+.6f}", flush=True)
    by_class = {}
    for cls in rr.CLASSES:
        values = [cue["mse"] for cue in per_cue if cue["class"] == cls]
        logs = [cue["log_mse"] for cue in per_cue if cue["class"] == cls]
        by_class[cls] = {"n": len(values), "min": min(values), "max": max(values), "mean": math.fsum(values) / len(values), "mean_log": math.fsum(logs) / len(logs)}
        print(f"  {cls}: MSE min {min(values):.6f} max {max(values):.6f} mean {by_class[cls]['mean']:.6f}; mean log MSE {by_class[cls]['mean_log']:+.6f}", flush=True)
    summary["result"] = {"rho": primary["rho"], "threshold": threshold, "primary": primary["result"], "D_EN": guard.get("D_EN"), "D_EN_exact": guard.get("D_EN_exact"),
                         "K": guard.get("K"), "p_value": guard.get("p_value"), "guard": guard["result"], "label": label, "by_class": by_class}

# -- the descriptive records --------------------------------------------------------------------------------------------------
section("the descriptive records (no outcome force)")
descriptives = confirmation_state.get("descriptives") or {}
print(f"records: {sorted(descriptives)}; failures: {descriptives.get('failures')}", flush=True)
print(json.dumps(descriptives, indent=1, default=str)[:20000], flush=True)
summary["descriptives_keys"] = sorted(descriptives)

# -- the repository and the outputs ---------------------------------------------------------------------------------------------
section("the repository and the outputs")
head, origin = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
tree = git("status", "--porcelain", "--untracked-files=all")
listing = sorted(path.name for path in (ROOT / "outputs/experiment-024").iterdir())
print(f"HEAD {head}; origin/main {origin}; remote {remote}; tree {'clean' if tree == '' else tree}", flush=True)
print(f"outputs/experiment-024: {listing}", flush=True)
check("HEAD == origin == remote == af160ce; tree clean", head == origin == remote and head.startswith("af160ce") and tree == "")
check("the installed lock and preregistration are unchanged", sha((ROOT / rr.LOCK_RELATIVE_PATH).read_bytes()).startswith("5c2a9b90")
      and sha((ROOT / rr.PREREGISTRATION_RELATIVE_PATH).read_bytes()).startswith("007c9e6c"))
check("no report was written", "report.md" not in listing and state["report"] is None)
summary["repository"] = {"head": head, "origin": origin, "remote": remote, "clean": tree == "", "outputs": listing}
check("no repository write by this verification", not REFUSED, REFUSED)
(HERE / "verify_confirm.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
print("FAILURES:", failures if failures else "none", flush=True)
sys.exit(1 if failures else 0)
