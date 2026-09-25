"""Item 4: the stage-2 measurement artifact, re-read from disk (read-only)."""
import rguard  # noqa: F401  (first)
from rguard import REPO

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(REPO)
EV = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm024")
X = ROOT / "experiments/024-readout-routing-nounness"
P = ROOT / "outputs/experiment-024/stage2-measurements.pt"
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail != '' else ''}", flush=True)


raw = P.read_bytes()
check("stage2-measurements.pt: 76,314,087 bytes", len(raw) == 76_314_087, len(raw))
check("stage2-measurements.pt: sha256 b21babe1…", hashlib.sha256(raw).hexdigest() == "b21babe189d1cccb9ae359cfabf74baaa3a8337cac04612bc0d4aaa232b4dc93", hashlib.sha256(raw).hexdigest())
T = torch.load(P, weights_only=True)
check("torch.load(weights_only=True): a dict of 20 tensors", isinstance(T, dict) and len(T) == 20 and all(isinstance(v, torch.Tensor) for v in T.values()), len(T))
expected_shapes = {}
for pop, rows in (("Y1", {"cue_final": 2880, "coordinated": 1440}), ("Y2", {"cue_final": 0, "coordinated": 0})):
    for group, n in rows.items():
        expected_shapes[f"{pop}/{group}/dc"] = ([n, 79], torch.float64)
        expected_shapes[f"{pop}/{group}/ceiling"] = ([n, 79], torch.float64)
        expected_shapes[f"{pop}/{group}/dx1"] = ([n, 2, 512], torch.float64)
        expected_shapes[f"{pop}/{group}/dx3"] = ([n, 2, 512], torch.float64)
        expected_shapes[f"{pop}/{group}/positions"] = ([n, 2], torch.int64)
actual = {k: (list(v.shape), v.dtype) for k, v in T.items()}
for k in sorted(T):
    print(f"   {k}: shape {list(T[k].shape)} dtype {T[k].dtype}")
check("tensor names, shapes and dtypes are exactly the expected 20", actual == expected_shapes)
check("4,320 pairs: 2,880 cue_final + 1,440 coordinated; Y2 empty", T["Y1/cue_final/dc"].shape[0] + T["Y1/coordinated/dc"].shape[0] == 4320)
fin = {k: bool(torch.isfinite(v).all()) for k, v in T.items() if v.dtype == torch.float64}
check("every float tensor entirely finite (no NaN left from the NaN-initialised blocks)", all(fin.values()), [k for k, v in fin.items() if not v])
# cue-final rows: p_t == p_c, slot 1 duplicates slot 0 exactly (stage_two_022 writes the same position twice)
cf = T["Y1/cue_final/positions"]
check("cue-final positions: p_t == p_c in every row", bool((cf[:, 0] == cf[:, 1]).all()))
check("cue-final dx1/dx3 slot 1 == slot 0 bit for bit", torch.equal(T["Y1/cue_final/dx1"][:, 0], T["Y1/cue_final/dx1"][:, 1]) and torch.equal(T["Y1/cue_final/dx3"][:, 0], T["Y1/cue_final/dx3"][:, 1]))
co = T["Y1/coordinated/positions"]
check("coordinated positions: p_t == p_c + 1 in every row", bool((co[:, 1] == co[:, 0] + 1).all()))


def canonical_digest(t):  # own re-implementation of readout_calibration.tensor_digest's definition
    a = t.detach().cpu().contiguous()
    kind = "<i8" if a.dtype in (torch.int64, torch.int32) else "<f8"
    return hashlib.sha256(json.dumps(list(a.shape)).encode("ascii") + b"|" + kind.encode("ascii") + b"|" + a.numpy().astype(kind, copy=False).tobytes()).hexdigest()


def own_digest(name, t):  # a different definition: name, dtype string, shape, raw native bytes via numpy tobytes(order C) + a sha512
    a = np.ascontiguousarray(t.detach().cpu().numpy())
    return hashlib.sha512(f"{name}|{a.dtype.str}|{a.shape}|".encode() + a.tobytes(order="C")).hexdigest()


state = json.loads((ROOT / "outputs/experiment-024/results.json").read_text())
recorded = state["confirmation"]["stage2"]["tensors_sha256"]
sys.path.insert(0, str(ROOT / "src"))
from neural_decompiler import readout_calibration as rc  # noqa: E402

mine = {k: canonical_digest(v) for k, v in T.items()}
canon_fn = {k: rc.tensor_digest(v) for k, v in T.items()}
check("own re-implementation of the canonical tensor digest == rc.tensor_digest == the state's recorded digests (20/20)", mine == canon_fn == recorded,
      sorted(k for k in recorded if recorded[k] != mine.get(k)))
check("the state's stage2 path is this file", state["confirmation"]["stage2"]["path"] == str(P))
for k in sorted(T):
    if k.startswith("Y1"):
        print(f"   {k}: canonical {mine[k][:16]}… own-sha512 {own_digest(k, T[k])[:24]}…")
combined = hashlib.sha256("".join(f"{k}={own_digest(k, T[k])}\n" for k in sorted(T)).encode()).hexdigest()
print(f"   own combined digest over the 20 tensors (sha256 of 'name=sha512' lines): {combined}")

# value ranges (descriptive)
for k in ("Y1/cue_final/dc", "Y1/coordinated/dc", "Y1/cue_final/ceiling", "Y1/coordinated/ceiling"):
    v = T[k]
    print(f"   {k}: min {float(v.min()):+.6f} max {float(v.max()):+.6f} mean {float(v.mean()):+.6f}")

# --- durability and order: code + evidence -----------------------------------------------------------------------------
run_src = (X / "run.py").read_text().splitlines()


def line_of(text, start=0):
    for i, line in enumerate(run_src[start:], start=start):
        if text in line:
            return i + 1
    return None


l_measure = line_of("measured = ul.stage_two_022(")
l_save = line_of("rr.save_durably(tensors, self.stage2_path)")
l_write_stage2 = line_of("self._write(state)  # every fresh measurement is on disk")
l_accounting = line_of("state[\"confirmation\"][\"accounting\"] = accounting")
l_reload = line_of("saved = torch.load(self.stage2_path)")
l_recompute = line_of("c_check = rr.recompute_c(progs, units, states, saved)")
l_gates = line_of("gates = rr.target_gates(progs, confirmation, units, states, saved)")
l_cells = line_of("cells = rr.fresh_cue_cells(units, saved, confirmation)")
l_score = line_of("results = rr.score(cells, confirmation, lock, self.config)")
order = [l_measure, l_save, l_write_stage2, l_accounting, l_reload, l_recompute, l_gates, l_cells, l_score]
print(f"   run.py lines: measure {l_measure}, save_durably {l_save}, state write (stage2 digests) {l_write_stage2}, accounting {l_accounting}, "
      f"torch.load re-read {l_reload}, recompute_c {l_recompute}, target_gates {l_gates}, fresh_cue_cells(saved) {l_cells}, score {l_score}")
check("run.py: measure → save_durably → state write → accounting → re-read from disk → C recompute → gates → cells from the re-read tensors → score",
      all(order) and order == sorted(order))
rr_src = (ROOT / "src/neural_decompiler/readout_routing.py").read_text()
check("rr.save_durably: flush + os.fsync(file) + fsync of the directory", "handle.flush()" in rr_src and "os.fsync(handle.fileno())" in rr_src and "fsync_directory(path.parent)" in rr_src)
rec = json.loads((EV / "confirm_record.json").read_text())
writes = list(rec["events"]["output_writes"])
check("launcher's first-write order: one results temp (ledger write) → stage2-measurements.pt → 8 more results temps (each file opened once)",
      writes[1] == "stage2-measurements.pt" and writes[0].startswith(".results-") and all(w.startswith(".results-") for w in writes[2:]) and len(writes) == 10
      and all(v == 1 for v in rec["events"]["output_writes"].values()), writes)
st = os.stat(P)
import datetime  # noqa: E402

ph = state["phases"]["confirm"]
check("stage2 mtime (19:35:01.29Z) precedes the result write's completed_at (19:37:04Z)", st.st_mtime < datetime.datetime.fromisoformat(ph["completed_at"]).timestamp(),
      datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat())
print(f"S04 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
