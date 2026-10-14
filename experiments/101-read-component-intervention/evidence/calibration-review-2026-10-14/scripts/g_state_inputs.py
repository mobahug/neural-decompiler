"""Check 3's remaining fields with the frozen code (no model, every forward refused): the state's inputs against the
runner's digests now, its versions against collect_versions(), its configuration and blobs; the phase rules' view of the
state (calibrate refuses a rerun, lock is eligible); the rehearsal's preflight count. Runs after a–f."""
import importlib.util
import json
import sys

import torch

from common import CAL, OUTPUTS_101, ROOT, WRITE_ATTEMPTS, dump


def _refuse(*a, **k):
    raise RuntimeError("review guard: forward refused")


torch.nn.Module.__call__ = _refuse
from neural_decompiler import models  # noqa: E402
models.load_model = _refuse
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler.provenance import collect_versions  # noqa: E402

for name in ("capture_prompt", "run_patched"):
    setattr(pm, name, _refuse)
index_mtime = (ROOT / ".git/index").stat().st_mtime
spec = importlib.util.spec_from_file_location("experiment_101_runner_review_g", ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
runner = runner_module.Runner(log=lambda m: None)
base = runner._base()
state = json.loads((OUTPUTS_101 / "results.json").read_text())
out = {"inputs_equal_runner_digests": state["inputs"] == {k: base.digests[k] for k in ri.DIGEST_KEYS}, "n_inputs": len(state["inputs"]),
       "versions_equal_now": dict(state["versions"]) == dict(collect_versions()), "configuration_equal": state["configuration"] == ri.PRODUCTION.to_json(),
       "module_blobs_equal_frozen": state["module_blobs"] == ri.FROZEN_BLOBS, "assert_frozen_blobs": ri.assert_frozen_blobs() == ri.FROZEN_BLOBS,
       "design_plan": state["design"] == ri.DESIGN and state["plan"] == ri.PLAN, "model": state["model"]}
try:
    ri.assert_phase_allowed("calibrate", state)
    out["calibrate_rerun_refused"] = False
except ri.PhaseError as error:
    out["calibrate_rerun_refused"] = str(error)
try:
    ri.assert_phase_allowed("lock", state)
    out["lock_eligible"] = True
except ri.PhaseError as error:
    out["lock_eligible"] = str(error)
for mode in ("rehearsal", "smoke", "production"):
    record = json.loads((CAL / f"{mode}_run.json").read_text())
    checks = record["preflight"]
    out[f"preflight_{mode}"] = {"count": len(checks), "all_true": all((v if isinstance(v, bool) else v["ok"]) for v in checks.values())}
out["git_index_mtime_unchanged"] = (ROOT / ".git/index").stat().st_mtime == index_mtime
out["write_attempts"] = WRITE_ATTEMPTS
dump("g_state_inputs.json", out)
print(json.dumps(out, indent=1))
