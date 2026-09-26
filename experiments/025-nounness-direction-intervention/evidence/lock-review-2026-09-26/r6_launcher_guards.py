"""Item 3: would the production launcher's guards (launch_lock.py, sha256 f66c12f2...) have caught a violation?

The launcher is NOT run. Its guard functions are extracted verbatim from its source with `ast` and exercised here:
- its audit hook `_hook` is CALLED DIRECTLY with synthetic audit-event arguments (no file is opened, written, renamed or
  deleted by this test);
- its refusal/sealing logic (`_refusal`, `_refuse_module_call`, `_sweep_aliases`, `_counted_load`) is applied to a toy
  torch module and to a toy alias module -- never to the Pythia model, and no prompt exists here."""
import guard  # noqa: F401  (the reviewer's own repository protection stays on)

import ast
import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

HERE = guard.HERE
LAUNCHER = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/lock025/launch_lock.py")
source = LAUNCHER.read_text(encoding="utf-8")
out = {"launcher_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}

import torch  # noqa: E402

from neural_decompiler import capture, interventions  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402

tree = ast.parse(source)
wanted = {"_path", "_inside", "_in_repo", "_hook", "_refusal", "_refuse_module_call", "_sweep_aliases", "_counted_load"}
chunks = [ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
out["extracted_functions"] = sorted(node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted)
ns: dict = {"os": os, "sys": sys, "Path": Path, "fcntl": __import__("fcntl")}
ns["ROOT"] = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
ns["HERE"] = LAUNCHER.parent
ns["OUTPUTS"] = (ns["ROOT"] / "outputs/experiment-025").resolve()
ns["events"] = {"load_model": 0, "modules_sealed": 0, "module_calls_refused": 0, "forward_refused": [], "model_method_refused": [], "capture_or_intervention_refused": [],
                "confirm_only_refused": [], "geometry_refused_in_dry_run": [], "aliases_patched": [], "forbidden_reads": [], "forbidden_writes": [],
                "forbidden_deletes_renames_links_mkdirs": [], "output_writes": [], "output_renames": [], "output_mkdirs": [], "outside_repo_writes": [], "lock_invocations": 0}
for chunk in chunks:
    exec(compile(chunk, str(LAUNCHER), "exec"), ns)
hook = ns["_hook"]
R = str(ns["ROOT"])


def probe(event, args):
    try:
        hook(event, args)
        return "allowed"
    except PermissionError as error:
        return f"refused: {str(error)[:120]}"


tmpdir = tempfile.mkdtemp(dir=str(HERE))
fd = os.open(tmpdir, os.O_RDONLY)
cases = {
    "read 022 calibration table": probe("open", (f"{R}/outputs/experiment-022/calibration-table.pt", "rb", os.O_RDONLY)),
    "read outputs/experiment-024": probe("open", (f"{R}/outputs/experiment-024/results.json", "r", os.O_RDONLY)),
    "read outputs/experiment-023": probe("open", (f"{R}/outputs/experiment-023/results.json", "r", os.O_RDONLY)),
    "read a source file": probe("open", (f"{R}/src/neural_decompiler/cue_rotation.py", "r", os.O_RDONLY)),
    "write a source file": probe("open", (f"{R}/src/neural_decompiler/cue_rotation.py", "w", os.O_WRONLY | os.O_CREAT | os.O_TRUNC)),
    "os.open O_CREAT in experiments/025": probe("open", (f"{R}/experiments/025-nounness-direction-intervention/preregistration-lock.json", None, os.O_WRONLY | os.O_CREAT)),
    "write outputs/experiment-025 candidate": probe("open", (f"{R}/outputs/experiment-025/candidate-lock.json", "w", os.O_WRONLY | os.O_CREAT)),
    "write outputs/experiment-025 stage2 (a file lock never writes)": probe("open", (f"{R}/outputs/experiment-025/stage2-measurements.pt", "wb", os.O_WRONLY | os.O_CREAT)),
    "remove a repository file": probe("os.remove", (f"{R}/README.md", None)),
    "rename inside outputs/experiment-025": probe("os.rename", (f"{R}/outputs/experiment-025/.results-x.json", f"{R}/outputs/experiment-025/results.json", None, None)),
    "rename out of outputs into src": probe("os.rename", (f"{R}/outputs/experiment-025/x", f"{R}/src/x", None, None)),
    "mkdir in the repository": probe("os.mkdir", (f"{R}/newdir", 0o777, None)),
    "symlink into the repository": probe("os.symlink", ("/tmp/x", f"{R}/link", None)),
    "relative remove with a temp dir_fd (dry-run-1 false positive case)": probe("os.remove", ("probe-source", fd)),
    "write outside the repository": probe("open", ("/private/var/folders/xx/T/tmpfile", "w", os.O_WRONLY)),
}
os.close(fd)
os.rmdir(tmpdir)
out["audit_hook_cases"] = cases
out["audit_hook_events"] = {k: v for k, v in ns["events"].items() if v}

# The refusal machinery on a toy module (never the model).
ns["events"]["module_calls_refused"] = 0
original_call = torch.nn.Module.__call__
torch.nn.Module.__call__ = ns["_refuse_module_call"]
toy = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
x = torch.zeros(1, 2)
try:
    toy(x)
    module_call = "NOT refused"
except RuntimeError as error:
    module_call = f"refused: {error}"
ns["original_load"] = lambda spec, *a, **k: toy
ns["_counted_load"](None)
try:
    toy.forward(x)
    direct_forward = "NOT refused"
except RuntimeError as error:
    direct_forward = f"refused: {error}"
try:
    toy[0].forward(x)
    sub_forward = "NOT refused"
except RuntimeError as error:
    sub_forward = f"refused: {error}"
class_route = "not sealed at class level (a type(sub).forward(sub, x) call would bypass the instance seal; Module.__call__ still refuses)"
torch.nn.Module.__call__ = original_call

# The alias sweep: an alias module holding copies of the entry points, found by identity.
alias = types.ModuleType("neural_decompiler.review_alias_probe")
alias.cap = pm.capture_prompt
alias.patched = pm.run_patched
alias.rc = capture.run_capture
alias.ri = interventions.run_interventions
sys.modules[alias.__name__] = alias
ns["ENTRY_POINTS"] = {id(capture.run_capture): ("capture.run_capture", capture.run_capture), id(interventions.run_interventions): ("interventions.run_interventions", interventions.run_interventions),
                      id(pm.capture_prompt): ("pm.capture_prompt", pm.capture_prompt), id(pm.run_patched): ("pm.run_patched", pm.run_patched)}
ns["REFUSALS"] = {key: ns["_refusal"]("capture_or_intervention_refused", name) for key, (name, _) in ns["ENTRY_POINTS"].items()}
ns["_sweep_aliases"]()
swept = [name for name in ("cap", "patched", "rc", "ri") if getattr(alias, name) is not getattr(pm, "__never__", None) and getattr(alias, name) in ns["REFUSALS"].values()]
try:
    alias.cap(None, None, [])
    alias_call = "NOT refused"
except RuntimeError as error:
    alias_call = f"refused: {error}"
out["toy_refusals"] = {"module_call": module_call, "direct_instance_forward": direct_forward, "submodule_direct_forward": sub_forward, "class_level_route": class_route,
                       "aliases_swept": swept, "alias_call": alias_call, "events": {k: v for k, v in ns["events"].items() if v and k not in ("forbidden_reads",)}}
out["guard"] = guard.summary()
(HERE / "r6_launcher_guards.out.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
