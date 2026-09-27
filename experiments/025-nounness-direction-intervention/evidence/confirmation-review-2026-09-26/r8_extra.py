"""Extras: (a) the smoke keys are spent (020's ledger) and outside 025; (b) the 020 reference rows are aligned to their own
frame: for the base runs, each frame's captured L4/L5 rows are nearer (TV) to its own locked rows than to any other
same-length frame's; (c) cr.render_report(state) rendered in memory only (a pure function; no report phase, nothing
written anywhere but this JSON), to check what `report` will state. No model. Writes r8_extra.json here.
"""
import guard  # noqa: F401

import json
from pathlib import Path

import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import upstream_localization as ul

ROOT = Path(guard.ROOT)
HERE = Path(guard.HERE)
guard.seal()
out: dict = {}
inputs = ul.load_frozen_inputs(ROOT, tracked=guard.tracked)
freeze = json.loads((ROOT / cr.CONFIRMATION_RELATIVE_PATH).read_text())
untagged = {f"{fid}|{c['word']}|{c['token_id']}" for fid in freeze["exposed_frame_ids"] for c in freeze["cues"]}
smoke = ("cardinal-009-1|ref|581", "quantifier-009-1|ref|1016", "coordinated-adjective-009-1|ref|581")
ledger = set(inputs.closure["ledger"])
out["smoke_keys"] = {"in_020_ledger": all(k in ledger for k in smoke), "in_025_manifest": any(k in untagged for k in smoke),
                     "are_patch_path_keys": any(k in cr.PATCH_PATH_SPENT_KEYS for k in smoke)}

# (b) frame alignment of the reference rows
frames = {f.frame_id: f for f in inputs.pool.frames}
FRAME_IDS = sorted(frames)
locked = inputs.closure["exploration"]["locked_states"]
saved = torch.load(ROOT / "outputs/experiment-025/stage2-measurements.pt", mmap=True, weights_only=True)
group_frames = {g: [f for f in FRAME_IDS if (frames[f].template_id == "coordinated-adjective") == (g == "coordinated")] for g in ("cue_final", "coordinated")}
tokens = sorted(int(c["token_id"]) for c in freeze["cues"])
ref = {f: (torch.tensor(locked[f]["rows4"], dtype=torch.float64), torch.tensor(locked[f]["rows5"], dtype=torch.float64)) for f in FRAME_IDS}


def mean_tv(g, fid, other):
    k = frames[fid].p_t + 1
    total = 0.0
    for t_rank in range(len(tokens)):
        row = t_rank * len(group_frames[g]) + group_frames[g].index(fid)
        a4 = saved[f"{g}/base/rows4"][row, :, :k].double()
        a5 = saved[f"{g}/base/rows5"][row, :, :k].double()
        total += float(torch.cat([0.5 * (a4 - ref[other][0]).abs().sum(-1), 0.5 * (a5 - ref[other][1]).abs().sum(-1)]).mean())
    return total / len(tokens)


own_best, gaps = 0, []
n_frames = 0
for g, fids in group_frames.items():
    for fid in fids:
        same = [o for o in FRAME_IDS if o != fid and frames[o].p_t == frames[fid].p_t]
        if not same:
            continue
        n_frames += 1
        mine = mean_tv(g, fid, fid)
        others = min(mean_tv(g, fid, o) for o in same)
        own_best += mine < others
        gaps.append(others - mine)
out["reference_row_frame_alignment"] = {"frames_tested": n_frames, "own_frame_nearest": own_best, "min_gap": min(gaps), "median_gap": sorted(gaps)[len(gaps) // 2]}

# (c) what `report` will render (in memory only)
state = json.loads((ROOT / "outputs/experiment-025/results.json").read_text())
text = cr.render_report(state)
lines = text.splitlines()
out["report_preview"] = {"label_line": next(l for l in lines if l.startswith("**`")), "table": [l for l in lines if l.startswith("| A |") or l.startswith("| B |") or l.startswith("| G |")],
                         "concentration_line": [l for l in lines if "reporting trigger" in l or "Concentration" in l],
                         "says_no_causal_effect": "no causal effect" in text.lower(), "chars": len(text),
                         "validity_line": [l for l in lines if l.startswith("- I1 max")], "c_line": [l for l in lines if l.startswith("- C recomputed")],
                         "patch_path_line": [l for l in lines if l.startswith("- The patch-path check: passed")]}
out["guard"] = guard.EVENTS
(HERE / "r8_extra.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=1, ensure_ascii=False))
