"""A descriptive breakdown (no gate): the ceiling error per arm — 020's frozen readout fed the run's measured Δx3 (the
ceiling) against the run's measured Δc, per noun (max) and in the mean over the 79 nouns — from the saved measurements
and the committed states (weights only; the review's guard; no forward pass)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rguard  # noqa: E402  (first)

rguard.self_test()
import json  # noqa: E402

import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = rguard.ROOT
EXP = ROOT / ri.EXPERIMENT_DIR
OUT = rguard.OUT
tokenizer = AutoTokenizer.from_pretrained(models.PYTHIA_70M.model_id, revision=models.PYTHIA_70M.revision)
rguard.install_forward_guards(tokenizer)
inputs = ul.load_frozen_inputs(ROOT)
model = models.load_model(models.PYTHIA_70M)
rguard.seal(model)
progs = ul.ModelPrograms.from_model(model, inputs)
confirmation = ri.confirmation_from_payload(json.loads((EXP / "confirmation-v1.json").read_text(encoding="utf-8")), ri.PRODUCTION)
index = json.loads((EXP / "stage1/states.json").read_text(encoding="utf-8"))
locked = ri.states_from_blocks(ul.read_table(EXP / "stage1/states.f64", index), confirmation.frames)
states = {f.frame_id: rd.state_from_locked(locked[f.frame_id], f) for f in confirmation.frames}
tensors = torch.load(ROOT / "outputs/experiment-101/stage2-measurements.pt")
by_arm: dict = {}
for frame in confirmation.frames:
    schedule = confirmation.schedule(frame)
    fid, p_c, p_t = frame.frame_id, frame.p_c, frame.p_t
    st = states[fid]
    x3pc, x3pt = tensors[f"{fid}/x3_pc"], tensors[f"{fid}/x3_pt"]
    con = tensors[f"{fid}/contrasts"].double()
    ref_row = [r.tag for r in schedule].index("REF")
    for row, run in enumerate(schedule):
        if not (run.arm.residual_edit or (run.arm.kind == "plain" and run.tag != "REF")):
            continue
        dx3 = {p_c: x3pc[row].double() - st.x3_all[p_c].double(), p_t: x3pt[row].double() - st.x3_all[p_t].double()}
        ceiling = ul.contrast_of(progs, st, dx3).double()
        dc = con[row] - con[ref_row]
        name = run.tag.split("@")[0]
        key = ("NEC-ISO" if name.startswith("NEC-ISO") else "SUF-ISO" if name.startswith("SUF-ISO") else name) + "@" + run.tag.split("@")[1]
        by_arm.setdefault(key, []).append((float((ceiling - dc).abs().max()), float(ceiling.mean() - dc.mean()), float(dc.mean()), fid))
summary = {}
for k, vals in sorted(by_arm.items()):
    worst = max(vals, key=lambda v: v[0])
    summary[k] = {"per_noun_max": worst[0], "at": worst[3], "mean_error_max_abs": max(abs(v[1]) for v in vals),
                  "mean_error_median": sorted(v[1] for v in vals)[len(vals) // 2], "median_measured_mean_dc": sorted(v[2] for v in vals)[len(vals) // 2]}
print(json.dumps(summary, indent=1))
rec = {"summary": summary, "guard": {"module_call_refusals": rguard.RECORD["module_call_refusals"], "entry_point_refusals": rguard.RECORD["entry_point_refusals"],
                                     "token_guard_hits": rguard.RECORD["token_guard_hits"]}}
(OUT / "g_ceiling_by_arm.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
print("guard fired:", bool(rguard.RECORD["module_call_refusals"] or rguard.RECORD["entry_point_refusals"] or rguard.RECORD["token_guard_hits"]))
