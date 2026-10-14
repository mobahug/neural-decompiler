"""The unguarded reference for the calibrate launcher's smoke mode (engineering only): the first exposed coordinated
frame's S1 re-capture and its 55 runs, through the same `ri` functions with no launcher guard, in a separate process.
It writes only `smoke_reference.pt` beside itself (outside the repository): every tensor's digest, the logits digests
and the re-captured state's digest. No gate, table, fraction or statistic is computed."""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)

import importlib.util  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment_101_runner", ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
runner = runner_module.Runner(log=lambda message: None)
base = runner._base()
frames = runner._exposed_frames(base.inputs)
schedule = runner._calibration_schedules(base.inputs, frames)[0]
frame = frames[0]
reference = base.inputs.pool.reference_prompt(frame)
models.seed_runtime(rd.RUNTIME_SEED, models.PYTHIA_70M.deterministic_algorithms)
model = models.load_model(models.PYTHIA_70M)
progs = ul.ModelPrograms.from_model(model, base.inputs)
geometry = ri.read_geometry(base.inputs.lock_011)
state = ri.capture_reference(model, progs, reference, "S1")
out: dict = {}
logits: dict = {}
executed: list = []
ri.stage_runs(model, progs, [schedule], geometry, ri.PRODUCTION, executed=executed, out=out, logits=logits)
record = {"frame": frame.frame_id, "keys": [cr.tagged_key(reference, "S1")] + executed, "tensors_sha256": {name: rc.tensor_digest(value) for name, value in out.items()},
          "logits_sha256": logits, "state_digest": rd.state_digest(rd.locked_state(state))}
torch.save(record, HERE / "smoke_reference.pt")
print(json.dumps({"frame": frame.frame_id, "runs": len(executed), "tensors": len(out), "state_digest": record["state_digest"]}))
