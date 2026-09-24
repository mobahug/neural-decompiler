"""Probe: does loading the pinned model execute any module? The refusal of nn.Module.__call__ (and of every capture /
intervention entry point) is installed BEFORE the load; nothing is computed."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/lockreview023")
import guard  # noqa: E402  (first: audit hook + torch.load patch)

import gc  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402

print("guard live:", json.dumps(guard.prove_live()), flush=True)
print("refusal installed before load:", guard.refuse_execution(), flush=True)
from neural_decompiler.models import PYTHIA_70M, load_model  # noqa: E402

try:
    model = load_model(PYTHIA_70M)
    print("load succeeded under the refusal (the load executed no module)", flush=True)
except Exception as error:  # noqa: BLE001
    print("load FAILED under the refusal:", type(error).__name__, error, flush=True)
    raise SystemExit(3)
modules = list(model.modules())
not_refused = sorted({type(m).__name__ for m in modules if type(m).__call__ is not guard.REFUSE_CALL})
print(f"modules {len(modules)}; classes whose __call__ is not the refusal: {not_refused}", flush=True)
print("execution_refused():", guard.execution_refused(), flush=True)
hf = sorted({p for p in guard.OPENED if p.startswith(str(guard.HF))})
print("HF files opened (Python-level):", json.dumps(hf, indent=1), flush=True)
del model, modules
gc.collect()
alive = [type(o).__name__ for o in gc.get_objects() if isinstance(o, torch.nn.Module)]
print(f"nn.Module objects alive after del + gc: {len(alive)} {sorted(set(alive))[:10]}", flush=True)
print("refused during probe:", guard.REFUSED, flush=True)
