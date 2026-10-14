"""A descriptive breakdown (no gate): 020's normalized E₁ (recorded, not gated by the reviewer's G4 decision) per arm, with
its denominator ‖Δh6_meas‖∞, and the Level-1 contrast-form error per arm, from the saved measurements and the committed
states through the frozen readout (weights only; the review's guard; no forward pass)."""
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
lock = json.loads((EXP / "preregistration-lock.json").read_text(encoding="utf-8"))
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
    x3pc, x3pt, h6 = tensors[f"{fid}/x3_pc"], tensors[f"{fid}/x3_pt"], tensors[f"{fid}/h6"]
    con = tensors[f"{fid}/contrasts"].double()
    ref_row = [r.tag for r in schedule].index("REF")
    for row, run in enumerate(schedule):
        if not (run.arm.residual_edit or (run.arm.kind == "plain" and run.tag != "REF")):
            continue
        dx3 = {p_c: x3pc[row].double() - st.x3_all[p_c].double(), p_t: x3pt[row].double() - st.x3_all[p_t].double()}
        detail = progs.readout.level1_detail(st, dict(dx3), l4_heads=True)
        dh6_meas = h6[row].double() - st.h6.double()
        e1 = rd.level1_error(detail["dh6"], dh6_meas)
        level1 = progs.readout.contrast(st, detail["dh6"], progs.nouns)[progs.scorable].double()
        l1 = float((level1 - (con[row] - con[ref_row])).abs().max())
        name = run.tag.split("@")[0]
        key = ("NEC-ISO" if name.startswith("NEC-ISO") else "SUF-ISO" if name.startswith("SUF-ISO") else name) + "@" + run.tag.split("@")[1]
        by_arm.setdefault(key, []).append((e1, float(dh6_meas.abs().max()), l1))
summary = {k: {"E1_max": max(v[0] for v in vals), "denominator_at_E1_max": max(vals)[1], "denominator_min": min(v[1] for v in vals),
               "level1_max": max(v[2] for v in vals)} for k, vals in sorted(by_arm.items())}
print(json.dumps(summary, indent=1))
rec = {"summary": summary, "guard": {"module_call_refusals": rguard.RECORD["module_call_refusals"], "entry_point_refusals": rguard.RECORD["entry_point_refusals"],
                                     "token_guard_hits": rguard.RECORD["token_guard_hits"]}}
(OUT / "f_e1_by_arm.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
print("guard fired:", bool(rguard.RECORD["module_call_refusals"] or rguard.RECORD["entry_point_refusals"] or rguard.RECORD["token_guard_hits"]))
