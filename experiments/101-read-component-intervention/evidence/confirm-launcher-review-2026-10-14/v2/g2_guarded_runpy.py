"""Run the v2 CLONE copy of the launcher (argv[1]) as __main__, after installing, before it runs:
- my guard (every write outside this review's directory refused) with an event recorder;
- a refusal of every forward route the launcher captures as its "originals" (``torch.nn.Module.__call__``,
  ``capture.run_capture``, ``interventions.run_interventions``, ``pm.capture_prompt``, ``pm.run_patched``,
  ``rd.capture_frame_020`` and their ``pm`` aliases): no token sequence of any kind can reach a forward in this process
  (stronger than a token-sequence guard). The model LOAD is allowed, so that ``barrier_dry_run`` runs as in production.
- optionally (REVIEW_NEGATIVE_CONTROL=geometry|table) a falsified recomputation, to show that a barrier failure in the
  dry run is a refusal with nothing written (no phase incident).
At exit, the recorded write-type events and the clone's state digest are written to out/v2_g2_<label>_events.json."""
import atexit
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import REVIEW, install_guard  # noqa: E402

target = Path(sys.argv[1]).resolve()
label = sys.argv[2]
assert target.parent == REVIEW / "v2/clone_launcher", target
src = target.read_text(encoding="utf-8")
import ast  # noqa: E402
assert not [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "confirm"]
assert '"].confirm()' not in src and "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler" not in src
clone = REVIEW / "clone"
state_path = clone / "outputs/experiment-101/results.json"
before = hashlib.sha256(state_path.read_bytes()).hexdigest()
recorded: list = []
install_guard(recorded)
sys.path = [p for p in sys.path if p != str(Path(__file__).resolve().parents[1] / "scripts")]

import torch  # noqa: E402

from neural_decompiler import capture, interventions, models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402

assert Path(models.__file__).resolve().is_relative_to(clone), models.__file__
attempts = {"forward_attempts": 0}


def never(name):
    def refuse(*args, **kwargs):
        attempts["forward_attempts"] += 1
        raise RuntimeError(f"review wrapper: {name} refused; no forward may run in the clone rehearsal")

    refuse.__name__ = refuse.__qualname__ = f"never_{name.replace('.', '_')}"
    return refuse


torch.nn.Module.__call__ = never("torch.nn.Module.__call__")
capture.run_capture = pm.run_capture = never("run_capture")
interventions.run_interventions = pm.run_interventions = never("run_interventions")
pm.capture_prompt = never("pm.capture_prompt")
pm.run_patched = never("pm.run_patched")
rd.capture_frame_020 = never("rd.capture_frame_020")
control = os.environ.get("REVIEW_NEGATIVE_CONTROL")
if control == "geometry":
    ri.verify_geometry_against_lock = lambda recomputed, lock: {"bitwise_equal": False, "differing": ["review-negative-control"]}
elif control == "table":
    real_table = ri.prediction_table

    def falsified_table(*args, **kwargs):
        table = real_table(*args, **kwargs)
        return {**table, "review_negative_control": True}

    ri.prediction_table = falsified_table


def dump():
    after = hashlib.sha256(state_path.read_bytes()).hexdigest()
    writes = [r for r in recorded if r["writing"]]
    clone_writes = [r for r in writes if any(isinstance(p, str) and p.startswith(str(clone) + "/") for p in (r["paths"] or []))]
    out = {"label": label, "control": control, "state_sha256_before": before, "state_sha256_after": after, "state_unchanged": before == after,
           "forward_attempts_refused_by_wrapper": attempts["forward_attempts"], "write_events_total": len(writes), "write_events_in_clone": clone_writes,
           "write_event_paths": sorted({p for r in writes for p in (r["paths"] or []) if isinstance(p, str)})}
    (REVIEW / f"out/v2_g2_{label}_events.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


atexit.register(dump)
sys.argv = [str(target)]
runpy.run_path(str(target), run_name="__main__")
