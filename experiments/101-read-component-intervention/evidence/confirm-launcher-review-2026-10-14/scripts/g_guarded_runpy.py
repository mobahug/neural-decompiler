"""Run the CLONE copy of the launcher (argv[1]) as __main__, after installing, before it runs:
- my guard (every write outside this review's directory refused; the clone is inside it);
- a refusal of every model load (``models.load_model``) and of every forward entry point and ``Module.__call__``
  (captured by the launcher as its "originals"), so that no forward of any prompt can happen in this process.
The launcher's own hook and guards are then installed by the launcher itself on top."""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REVIEW, install_guard  # noqa: E402

target = Path(sys.argv[1]).resolve()
assert target.parent == REVIEW / "clone_launcher", target
src = target.read_text(encoding="utf-8")
import ast  # noqa: E402
assert not [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "confirm"]
assert '"].confirm()' not in src and "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler" not in src
install_guard()
sys.path = [p for p in sys.path if p != str(Path(__file__).resolve().parent)]

import torch  # noqa: E402

from neural_decompiler import capture, interventions, models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402

assert Path(models.__file__).resolve().is_relative_to(REVIEW / "clone"), models.__file__


def never(name):
    def refuse(*args, **kwargs):
        raise RuntimeError(f"review wrapper: {name} refused; no model load or forward may run in the clone rehearsal")

    refuse.__name__ = refuse.__qualname__ = f"never_{name.replace('.', '_')}"
    return refuse


models.load_model = never("models.load_model")
torch.nn.Module.__call__ = never("torch.nn.Module.__call__")
capture.run_capture = pm.run_capture = never("run_capture")
interventions.run_interventions = pm.run_interventions = never("run_interventions")
pm.capture_prompt = never("pm.capture_prompt")
pm.run_patched = never("pm.run_patched")
rd.capture_frame_020 = never("rd.capture_frame_020")
sys.argv = [str(target)]
runpy.run_path(str(target), run_name="__main__")
