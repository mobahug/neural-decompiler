"""Descriptive: the design's exposed-spike numbers (pool-wide, per group) recomputed from the artifact columns alone."""
import json, math, sys
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/extractreview023")
from revlib import CAND_DATA, CAND_INDEX
idx = json.load(open(CAND_INDEX))
c = np.frombuffer(CAND_DATA.read_bytes(), dtype="<f8").reshape(18900, 8)
grp = [f[2] for f in idx["frames"]]
cls = [q[2] for q in idx["cues"]]
for label, keep_pron in (("all 175 cues (the spike's pool)", True), ("without the pronoun stratum", False)):
    for g in ("cue_final", "coordinated"):
        rows = [ci * 108 + fi for ci in range(175) for fi in range(108) if grp[fi] == g and (keep_pron or cls[ci] != "possessive-or-pronoun")]
        r = c[rows]
        N, S, Q = (math.fsum(r[:, k].tolist()) for k in (0, 1, 2))
        s0, s1, sc = (math.fsum(r[:, k].tolist()) for k in (3, 4, 5))
        sst = Q - S * S / N
        print(f"{label:34s} {g:12s} pairs {len(rows):6d}  R2_0 {1-s0/sst:.4f}  R2_1 {1-s1/sst:.4f}  R2_C {1-sc/sst:.4f}  g {(s0-s1)/(s0-sc):.4f}  gap {(s0-sc)/sst:.4f}  mean {S/N:.3f} sd {math.sqrt(sst/N):.3f}")
