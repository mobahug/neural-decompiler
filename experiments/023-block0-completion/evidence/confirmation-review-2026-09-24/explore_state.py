import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirmreview023")
import guard  # noqa: E402  (installs the audit hook and the torch.load refusal, proves them live)
import json, hashlib
import torch
ROOT = guard.ROOT
state = json.loads(open(f"{ROOT}/outputs/experiment-023/results.json", encoding="utf-8").read())

def shape(v, depth=0, maxdepth=3):
    if isinstance(v, dict):
        if depth >= maxdepth:
            return f"dict[{len(v)}]"
        return {k: shape(x, depth + 1, maxdepth) for k, x in v.items()}
    if isinstance(v, list):
        return f"list[{len(v)}]"
    return v if not isinstance(v, str) or len(v) < 90 else v[:90] + "..."

top = {k: shape(v, 0, 1) for k, v in state.items()}
print(json.dumps(top, indent=1)[:6000])
print("phases:", json.dumps(state["phases"], indent=1)[:4000])
conf = state["confirmation"]
print("confirmation keys:", list(conf))
print("stage1 keys:", list(conf["stage1"]))
print("stage2:", json.dumps({k: (v if k != "tensors_sha256" else {kk: vv[:12] for kk, vv in v.items()}) for k, v in conf["stage2"].items()}, indent=1))
print("gates:", json.dumps(conf["gates"], indent=1))
print("kernel_check:", conf.get("kernel_check"))
print("aggregate_label:", conf.get("aggregate_label"), "lock_sha256", conf.get("lock_sha256"), "completed_at", conf.get("completed_at"))
for c, e in conf["conditions"].items():
    print(c, json.dumps(e)[:1500])
print("descriptives keys:", list(conf["descriptives"]), "failures:", conf["descriptives"].get("failures"))
m = torch.load(f"{ROOT}/outputs/experiment-023/stage2-measurements.pt")
for k, v in m.items():
    print(k, tuple(v.shape), v.dtype)
