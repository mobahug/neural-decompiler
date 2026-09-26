"""Does the launcher's guard prefix (everything before it imports the runner) catch a forward, a model load, a capture,
a score or geometry call, and a forbidden read or write?

Executes only the launcher's prefix (up to the line that builds the runner's import spec): the audit hook and the
patches. The runner's freeze is never called; the runner module is only imported to check its bound model loader.
Forbidden paths are never opened: the hook function is called directly with synthetic arguments. The toy module used
for the forward test is torch.nn.Identity, not the model. Nothing is written under the repository.
"""
import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

LAUNCHER = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/freeze025/launch_freeze.py")
HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
src = LAUNCHER.read_text(encoding="utf-8")
OUT = {"launcher_sha256": hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()}

# 1. Static order of the top-level statements.
tree = ast.parse(src)
order = []
for node in tree.body:
    text = ast.get_source_segment(src, node).splitlines()[0][:110]
    order.append((node.lineno, text))
OUT["top_level_order"] = order


def first_line(pred):
    return next(lineno for lineno, text in order if pred(text))


lines = {
    "addaudithook": first_line(lambda t: t.startswith("sys.addaudithook(_hook)")),
    "import_torch": first_line(lambda t: t.startswith("import torch")),
    "import_cr": first_line(lambda t: t.startswith("from neural_decompiler import cue_rotation")),
    "patch_module_call": first_line(lambda t: t.startswith("torch.nn.Module.__call__ =")),
    "patch_load_model": first_line(lambda t: t.startswith("models.load_model =")),
    "patch_pm_loop": first_line(lambda t: t.startswith("for name in (\"capture_prompt\"")),
    "patch_score_loop": first_line(lambda t: t.startswith("for module, names in ((rr,")),
    "runner_spec": first_line(lambda t: t.startswith("spec = importlib.util.spec_from_file_location")),
    "runner_exec": first_line(lambda t: t.startswith("spec.loader.exec_module(runner_module)")),
    "assert_refusal_bound": first_line(lambda t: t.startswith("assert runner_module.load_model is _refuse_load")),
}
OUT["lines"] = lines
OUT["guards_before_runner_import"] = max(lines[k] for k in ("addaudithook", "patch_module_call", "patch_load_model", "patch_pm_loop", "patch_score_loop")) < lines["runner_exec"]
OUT["hook_before_any_project_import"] = lines["addaudithook"] < lines["import_torch"] < lines["import_cr"]
OUT["freeze_call_sites"] = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "freeze"]

# 2. Execute the prefix only (the hook and the patches), in its own namespace.
prefix = "\n".join(src.splitlines()[: lines["runner_spec"] - 1])
from neural_decompiler import cue_rotation as cr0  # noqa: E402  (imported first so the original functions can be kept)
from neural_decompiler import readout_routing as rr0  # noqa: E402

orig = {"cr.direction": cr0.direction, "cr.centroids": cr0.centroids, "cr.cue_vectors": cr0.cue_vectors}
ns = {"__name__": "launcher_prefix", "__file__": str(LAUNCHER)}
exec(compile(prefix, str(LAUNCHER), "exec"), ns)
events = ns["events"]
import torch  # noqa: E402

from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

results = {}


def attempt(name, fn):
    try:
        fn()
        results[name] = "NOT REFUSED"
    except BaseException as error:  # noqa: BLE001
        results[name] = f"{type(error).__name__}: {error}"


# The runner module's bound loader (import only; no phase is run).
spec = importlib.util.spec_from_file_location("experiment_025_runner_guardtest", ROOT / "experiments/025-nounness-direction-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
OUT["runner_load_model_is_refusal"] = runner_module.load_model is ns["_refuse_load"]
OUT["runner_default_model_loader_is_refusal"] = runner_module.Runner.__dataclass_fields__["model_loader"].default is ns["_refuse_load"]

attempt("models.load_model", lambda: models.load_model(models.PYTHIA_70M))
attempt("runner.model_loader", lambda: runner_module.Runner.__dataclass_fields__["model_loader"].default(models.PYTHIA_70M))
attempt("toy nn.Module forward", lambda: torch.nn.Identity()(torch.zeros(1)))
attempt("pm.run_patched", lambda: pm.run_patched(None, None, {}, {}))
attempt("pm.capture_prompt", lambda: pm.capture_prompt(None, None, []))
z = torch.zeros(4, dtype=torch.float64)
W = torch.zeros(3, 4)
attempt("rr.nounness", lambda: rr.nounness(z, z, z))
attempt("rr.centroid", lambda: rr.centroid(W, [0]))
attempt("rr.full_scores", lambda: rr.full_scores(W, {}, [0]))
attempt("cr.score", lambda: cr.score(z, z))
attempt("cr.geometry_block", lambda: cr.geometry_block(W, {}, None, [], cr.PRODUCTION))
# Internal calls through module globals: the ORIGINAL (unpatched) functions reach the patched ones.
attempt("original cr.direction -> centroids (internal)", lambda: orig["cr.direction"](W, {"score": {"noun_row_ids": [0], "calibration_cue_ids": [1]}}))
attempt("original cr.centroids -> rr.centroid (internal)", lambda: orig["cr.centroids"](W, {"score": {"noun_row_ids": [0], "calibration_cue_ids": [1]}}))

# The audit hook, called directly with synthetic arguments (no forbidden path is opened).
hook = ns["_hook"]
attempt("hook read outputs/experiment-023", lambda: hook("open", (str(ROOT / "outputs/experiment-023/any.pt"), "rb", 0)))
attempt("hook read outputs/experiment-024", lambda: hook("open", (str(ROOT / "outputs/experiment-024/results.json"), "r", 0)))
attempt("hook read calibration-table.pt", lambda: hook("open", (str(ROOT / "outputs/experiment-022/calibration-table.pt"), "rb", 0)))
attempt("hook write README (mode w)", lambda: hook("open", (str(ROOT / "README.md"), "w", 0)))
attempt("hook write via os.open flags", lambda: hook("open", (str(ROOT / "experiments/x.json"), None, os.O_WRONLY | os.O_CREAT)))
attempt("hook write confirmation (allowed, counted)", lambda: hook("open", (str(ROOT / "experiments/025-nounness-direction-intervention/confirmation-v1.json"), "w", 0)))
attempt("hook read README (allowed)", lambda: hook("open", (str(ROOT / "README.md"), "r", 0)))
OUT["attempts"] = results
OUT["events_after"] = {k: (list(v) if isinstance(v, list) else v) for k, v in events.items()}

# 3. Does Path.write_text / read_bytes raise exactly one "open" audit event with a detectable mode? (scratchpad only)
seen = []
probe = HERE / "audit_probe.txt"


def recorder(event, args):
    if event == "open" and args and isinstance(args[0], (str, os.PathLike)) and os.fsdecode(args[0]) == str(probe):
        seen.append([event, str(args[1]), args[2] if len(args) > 2 else None])


sys.addaudithook(recorder)
probe.write_text("x", encoding="utf-8")
write_events = list(seen)
seen.clear()
probe.read_bytes()
read_events = list(seen)
probe.unlink()
OUT["audit_open_event_for_write_text"] = write_events
OUT["audit_open_event_for_read_bytes"] = read_events
(HERE / "guard_test.out.json").write_text(json.dumps(OUT, indent=1) + "\n", encoding="utf-8")
print(json.dumps(OUT, indent=1)[:6000])
