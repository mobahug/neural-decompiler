"""Cross-check (after the independent part): the 161 target-noun form ids and the 316 frame tokens from 020's pool,
via the runner's base (module code), against the committed lists. Same guards as cross_check_ri.py."""
import importlib.util, json, os, sys
from pathlib import Path
ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
refused = []
def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        text = str(Path(os.fsdecode(args[0])).resolve())
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if text.endswith((".safetensors", "pytorch_model.bin", ".ckpt")) or (writing and text.startswith(str(ROOT) + "/") and "/.venv/" not in text):
            refused.append(text); raise PermissionError(text)
    elif event in ("os.remove", "os.unlink", "os.rename", "os.replace", "os.rmdir", "shutil.rmtree", "os.mkdir"):
        for a in args:
            if isinstance(a, (str, bytes, os.PathLike)) and os.path.isabs(os.fsdecode(a)) and str(Path(os.fsdecode(a)).resolve()).startswith(str(ROOT) + "/"):
                refused.append([event, os.fsdecode(a)]); raise PermissionError(event)
sys.addaudithook(_hook)
import torch
torch.nn.Module.__call__ = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forward refused"))
from neural_decompiler import models
models.load_model = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("load refused"))
from neural_decompiler import readout_routing as rr
spec = importlib.util.spec_from_file_location("r101", ROOT / "experiments/101-read-component-intervention/run.py")
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
base = m.Runner(log=lambda s: None)._base()
pool = base.inputs.pool
c024 = json.loads((ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json").read_text())
c025 = json.loads((ROOT / "experiments/025-nounness-direction-intervention/confirmation-v1.json").read_text())
t = rr.target_noun_form_ids(pool); f = rr.frame_token_ids(pool)
print(json.dumps({"pool_nouns": len(pool.nouns), "pool_frames": len(pool.frames), "target_forms_from_pool": len(t),
                  "equal_024_committed": t == sorted(c024["target_noun_form_ids"]["ids"]),
                  "frame_tokens_from_pool": len(f), "equal_025_blocked_frame_tokens": f == sorted(c025["blocked"]["frame_tokens"]["ids"]),
                  "refused": refused}))
