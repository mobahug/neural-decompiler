"""Items 4 and 7: the four results recomputed exactly from the saved measurements and the two tables with my own code
(numpy float64 per-pair pooling, flattened two-pass, math.fsum, and exact integer/rational arithmetic on the float64
inputs); my own classification against the lock's full-precision floors; the ' shiny' secondary number. No model."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirmreview023")
import guard  # noqa: E402  FIRST

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(guard.ROOT)
E = ROOT / "experiments/023-block0-completion"
OUT = ROOT / "outputs/experiment-023"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail != "" else ""), flush=True)
    if not ok:
        FAIL.append(name)


def cj(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def my_tensor_digest(t):  # my reimplementation of readout_calibration.tensor_digest
    a = t.detach().cpu().contiguous().numpy()
    kind = "<i8" if a.dtype in (np.int64, np.int32) else "<f8"
    data = np.ascontiguousarray(a.astype(kind)).tobytes()
    return hashlib.sha256(json.dumps(list(a.shape)).encode("ascii") + b"|" + kind.encode("ascii") + b"|" + data).hexdigest()


def read_f64_table(data_path, blocks):  # my own reader: raw little-endian float64, blocks back to back
    raw = Path(data_path).read_bytes()
    out, offset = {}, 0
    for name, shape in blocks:
        count = int(np.prod(shape))
        out[name] = np.frombuffer(raw, dtype="<f8", count=count, offset=offset).reshape(shape).astype(np.float64)
        offset += 8 * count
    assert offset == len(raw), (offset, len(raw))
    return out, hashlib.sha256(raw).hexdigest()


raw_state = (OUT / "results.json").read_text(encoding="utf-8")
state = json.loads(raw_state)
lock = json.loads((E / "preregistration-lock.json").read_text(encoding="utf-8"))
conf = json.loads((E / "confirmation-v1.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- inputs, verified by digests
m = torch.load(OUT / "stage2-measurements.pt")
rec = state["confirmation"]["stage2"]["tensors_sha256"]
mine = {k: my_tensor_digest(v) for k, v in m.items()}
check("stage-2 tensors: my tensor digests == the state's (20 tensors)", mine == rec and len(mine) == 20)
check("stage-2 dc/ceiling/dx1/dx3 all finite", all(bool(torch.isfinite(v).all()) for k, v in m.items() if not k.endswith("/positions")))
y1, y1_sha = read_f64_table(E / "locked-y1-table.f64", [("cue_final", (1728, 2, 79)), ("coordinated", (864, 2, 79))])
check("committed Y1 table bytes sha256 == lock-bound 37603f05…", y1_sha == lock["y1_table"]["file_sha256"] == "37603f058500e056214a8a6ae413ef2b27eb94c5e1d618caaa8c9a61121a2f96")
y2, y2_sha = read_f64_table(OUT / "y2-table.f64", [("cue_final", (288, 2, 79)), ("coordinated", (144, 2, 79))])
check("on-disk Y2 table bytes sha256 == stage-1 record 4623adde…", y2_sha == state["confirmation"]["stage1"]["y2_table"]["file_sha256"] == "4623adde5eb53e14b0368e5b4d8b91bd6f2513bd522ba074bc55ab747970d2c2")
tables = {"Y1": y1, "Y2": y2}

# my own pair order (cues by token id; frames by frame_id; per group) to check the positions tensor and locate ' shiny'
from neural_decompiler import upstream_localization as ul  # noqa: E402  (only the frozen input loader: the 020 pool frames)

inputs = ul.load_frozen_inputs(ROOT)
exposed = sorted(inputs.pool.frames, key=lambda f: f.frame_id)
new = sorted([(e["frame_id"], e["template_id"], e["p_c"], e["p_t"]) for e in conf["frames"]])
tok = sorted([(int(c["token_id"]), c["word"], c["class"]) for c in conf["cues"]])
frames = {"Y1": [(f.frame_id, f.template_id, f.p_c, f.p_t) for f in exposed], "Y2": new}
order = {}
for pop in ("Y1", "Y2"):
    for group in ("cue_final", "coordinated"):
        fr = [f for f in frames[pop] if (f[1] == "coordinated-adjective") == (group == "coordinated")]
        order[f"{pop}/{group}"] = [(t, w, cls, f) for t, w, cls in tok for f in fr]
y1_index = json.loads((E / "locked-y1-table.json").read_text(encoding="utf-8"))
y2_index = json.loads((OUT / "y2-table.json").read_text(encoding="utf-8"))
check("my pair order == the Y1 and Y2 indexes' pair lists", all([[w, t, f[0]] for t, w, _, f in order[f"Y1/{g}"]] == y1_index["pairs"][g] for g in ("cue_final", "coordinated"))
      and all([[w, t, f[0]] for t, w, _, f in order[f"Y2/{g}"]] == y2_index["pairs"][g] for g in ("cue_final", "coordinated")))
check("stage-2 positions tensors == (p_c, p_t) of each row's frame in my order", all(m[f"{k}/positions"].tolist() == [[f[2], f[3]] for _, _, _, f in order[k]] for k in order))

# ---------------------------------------------------------------- the statistic, four independent ways
GUARD = 0.90
GAP = 0.02


def exact_ints(arrays):
    """Every float64 of the given arrays as an integer numerator over the common denominator 2**K (exact)."""
    ratios = [[v.as_integer_ratio() for v in a.reshape(-1).tolist()] for a in arrays]
    K = max((den.bit_length() - 1) for r in ratios for _, den in r)
    return [[num << (K - (den.bit_length() - 1)) for num, den in r] for r in ratios], K


def stats_exact(y, p0, p1, c):
    (Y, P0, P1, C), K = exact_ints([y, p0, p1, c])
    n = len(Y)
    SY, QY = sum(Y), sum(v * v for v in Y)
    A0 = sum((a - b) ** 2 for a, b in zip(Y, P0))
    A1 = sum((a - b) ** 2 for a, b in zip(Y, P1))
    AC = sum((a - b) ** 2 for a, b in zip(Y, C))
    scale = Fraction(1, 1 << (2 * K))
    sst = Fraction(n * QY - SY * SY, n) * scale  # Q − S²/N, exact
    sse = {k: Fraction(v) * scale for k, v in (("SSE0", A0), ("SSE1", A1), ("SSEC", AC))}
    out = {"N": n, "S": Fraction(SY, 1 << K), "Q": Fraction(QY) * scale, "SST": sst, **sse}
    out["defined"] = sst > 0 and sse["SSE0"] - sse["SSEC"] >= Fraction(GAP) * sst  # 0.02 as the double the code uses
    out["g"] = Fraction(A0 - A1, A0 - AC)
    out["gap"] = (sse["SSE0"] - sse["SSEC"]) / sst
    out.update({f"R2_{k}": 1 - sse[f"SSE{k}"] / sst for k in ("0", "1", "C")})
    return out


def stats_pairs_np(y, p0, p1, c):
    """Per-pair cells (n, S, Q, SSE0, SSE1, SSEC) then pooled sums; SST = Q − S²/N (the frozen form, my numpy code)."""
    n = np.full(y.shape[0], y.shape[1], dtype=np.float64)
    S, Q = y.sum(axis=1), (y * y).sum(axis=1)
    e0, e1, ec = ((y - p0) ** 2).sum(axis=1), ((y - p1) ** 2).sum(axis=1), ((y - c) ** 2).sum(axis=1)
    N, SS, QQ = n.sum(), S.sum(), Q.sum()
    sst = QQ - SS * SS / N
    E0, E1, EC = e0.sum(), e1.sum(), ec.sum()
    mean_i = S / n
    m2_i = ((y - mean_i[:, None]) ** 2).sum(axis=1)
    two_pass = m2_i.sum() + (n * (mean_i - SS / N) ** 2).sum()
    return {"N": N, "S": SS, "Q": QQ, "SST": sst, "SSE0": E0, "SSE1": E1, "SSEC": EC, "g": (E0 - E1) / (E0 - EC), "gap": (E0 - EC) / sst,
            "R2_0": 1 - E0 / sst, "R2_1": 1 - E1 / sst, "R2_C": 1 - EC / sst, "e6": abs(sst - two_pass) / abs(two_pass)}


def stats_flat_np(y, p0, p1, c):
    yf = y.reshape(-1)
    sst = float(((yf - yf.mean()) ** 2).sum())
    e = [float(((yf - k.reshape(-1)) ** 2).sum()) for k in (p0, p1, c)]
    return {"SST": sst, "SSE0": e[0], "SSE1": e[1], "SSEC": e[2], "g": (e[0] - e[1]) / (e[0] - e[2])}


def stats_fsum(y, p0, p1, c):
    yl = y.reshape(-1).tolist()
    N = len(yl)
    S = math.fsum(yl)
    mean = S / N
    sst = math.fsum((v - mean) ** 2 for v in yl)
    e = [math.fsum((a - b) ** 2 for a, b in zip(yl, k.reshape(-1).tolist())) for k in (p0, p1, c)]
    return {"SST": sst, "SSE0": e[0], "SSE1": e[1], "SSEC": e[2], "g": (e[0] - e[1]) / (e[0] - e[2])}


def classify(g, defined, F):
    if not defined:
        return "NOT_INTERPRETABLE"
    if g < GUARD:
        return "GUARD_FAILURE"
    if g < F:
        return "ENVELOPE_ONLY_FAILURE"
    return "PASS"


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


print()
print("| condition | pairs | g (state) | g (mine, pooled) | g exact (to 20 d.p.) | F (lock, exact double) | g − F (float) | g − F (exact) | my label | state label | gap | R²₀ | R²₁ | R²_C |")
worst = {}
results = {}
for key in ("Y1/cue_final", "Y1/coordinated", "Y2/cue_final", "Y2/coordinated"):
    pop, group = key.split("/")
    y = m[f"{key}/dc"].numpy().astype(np.float64)
    c = m[f"{key}/ceiling"].numpy().astype(np.float64)
    p0, p1 = tables[pop][group][:, 0, :], tables[pop][group][:, 1, :]
    rec = state["confirmation"]["conditions"][key]
    F = lock["conditions"][key]["envelope"]["bound"]
    check(f"{key}: the lock's envelope is the frozen element [249] (rank 250, lower) and == the state's", lock["conditions"][key]["envelope"] == rec["envelope"] and
          (lock["conditions"][key]["envelope"]["element"], lock["conditions"][key]["envelope"]["rank"], lock["conditions"][key]["envelope"]["kind"]) == (249, 250, "lower"))
    ex = stats_exact(y, p0, p1, c)
    pr = stats_pairs_np(y, p0, p1, c)
    fl = stats_flat_np(y, p0, p1, c)
    fs = stats_fsum(y, p0, p1, c)
    g_exact = ex["g"]
    label_float = classify(pr["g"], pr["SST"] > 0 and pr["SSE0"] - pr["SSEC"] >= GAP * pr["SST"], F)
    label_exact = classify(g_exact, ex["defined"], Fraction(F))
    # the frozen rule's quantities against the state
    diffs = {name: rel(pr[name], rec[name]) for name in ("N", "SST", "SSE0", "SSE1", "SSEC", "g", "gap", "R2_0", "R2_1", "R2_C")}
    diffs_exact = {name: rel(float(ex[name]), rec[name]) for name in ("SST", "SSE0", "SSE1", "SSEC", "g", "gap", "R2_0", "R2_1", "R2_C")}
    worst[key] = max(diffs.values())
    check(f"{key}: my pooled float64 statistics == state (N exact; others ≤ 1e-12 relative)", pr["N"] == rec["N"] == 79 * rec["n_pairs"] == 79 * y.shape[0] and max(diffs.values()) <= 1e-12,
          f"max rel diff {max(diffs.values()):.2e}")
    check(f"{key}: exact arithmetic == state within 1e-12 (g |Δ| {abs(float(g_exact) - rec['g']):.2e})", max(diffs_exact.values()) <= 1e-12, f"max rel diff {max(diffs_exact.values()):.2e}")
    check(f"{key}: flattened two-pass and fsum agree with exact (g ≤ 1e-12)", abs(fl["g"] - float(g_exact)) <= 1e-12 and abs(fs["g"] - float(g_exact)) <= 1e-12,
          f"flat {fl['g'] - float(g_exact):+.2e}, fsum {fs['g'] - float(g_exact):+.2e}")
    check(f"{key}: E6 (SST one-pass vs pooled two-pass) ≤ 1e-10", pr["e6"] <= 1e-10, f"{pr['e6']:.2e}")
    check(f"{key}: interpretable (SST > 0, gap ≥ 0.02) exactly", ex["defined"] and rec["interpretable"], f"gap exact {float(ex['gap']):.6f}")
    check(f"{key}: my label (float and exact) == state label == expected", label_float == label_exact == rec["result"], f"{label_exact}")
    results[key] = {"label": label_exact, "g": float(g_exact), "F": F, "margin_exact": g_exact - Fraction(F), "margin_float": pr["g"] - F,
                    "g_float_err": abs(pr["g"] - float(g_exact)), "g_state_err": abs(rec["g"] - float(g_exact))}
    print(f"| {key} | {y.shape[0]} | {rec['g']!r} | {pr['g']!r} | {float(g_exact):.17f} ({g_exact.numerator.bit_length()}/{g_exact.denominator.bit_length()} bits) | {F!r} | {pr['g'] - F:+.17e} | "
          f"{float(g_exact - Fraction(F)):+.17e} | {label_exact} | {rec['result']} | {float(ex['gap']):.4f} | {float(ex['R2_0']):.4f} | {float(ex['R2_1']):.4f} | {float(ex['R2_C']):.4f} |", flush=True)

print()
for key, r in results.items():
    print(f"{key}: g − F exact = {float(r['margin_exact']):+.17e}; |g_state − g_exact| = {r['g_state_err']:.2e}; |g_mine − g_exact| = {r['g_float_err']:.2e}; "
          f"margin / numerical error = {abs(float(r['margin_exact'])) / max(r['g_state_err'], 1e-300):.3e}; g − 0.90 = {r['g'] - 0.9:+.6f}")
# Y1/coordinated in detail
r = results["Y1/coordinated"]
check("Y1/coordinated: g ≥ F exactly (exact rational comparison with the lock's double) and the float comparison agrees", r["margin_exact"] > 0 and r["margin_float"] > 0,
      f"g − F = {float(r['margin_exact']):.6e} exact, {r['margin_float']:.6e} float; numerical error {r['g_state_err']:.1e}")
# the display value: the classification is the same whether the exact bound or the 6-d.p. display value is used, and score() reads the exact one
check("labels unchanged against the preregistration's rounded display bounds (robustness only)", all(classify(results[k]["g"], True, round(results[k]["F"], 6)) == results[k]["label"] for k in results))
b0c_src = (ROOT / "src/neural_decompiler/block0_completion.py").read_text(encoding="utf-8")
score_src = b0c_src[b0c_src.index("def score("):b0c_src.index("def fresh_descriptives(")]
check("score() classifies with lock['conditions'][c]['envelope']['bound'] (the lock JSON's full-precision double), not a rendered value",
      'entry = lock["conditions"][condition]' in score_src and 'entry["envelope"]["bound"], condition)' in score_src and "round(" not in score_src and ":.6f" not in score_src)

# ---------------------------------------------------------------- item 7: ' shiny' × coordinated-adjective-009-1
rows = [i for i, (t, w, cls, f) in enumerate(order["Y1/coordinated"]) if t == 30006 and f[0] == "coordinated-adjective-009-1"]
frame009 = next(f for f in inputs.pool.frames if f.frame_id == "coordinated-adjective-009-1")
check("' shiny' (30006) × coordinated-adjective-009-1 is exactly one Y1/coordinated row; that frame's p_t token is 30006 (same token at p_c and p_t)",
      len(rows) == 1 and tuple(frame009.suffix_ids) == (30006,) and y1_index["pairs"]["coordinated"][rows[0]] == ["shiny", 30006, "coordinated-adjective-009-1"],
      f"row {rows[0]}; frame text {frame009.text_template!r}, suffix {frame009.suffix_ids}")
check("the primary Y1/coordinated result includes all 864 pairs (n_pairs 864, N 68,256)", state["confirmation"]["conditions"]["Y1/coordinated"]["n_pairs"] == 864 and state["confirmation"]["conditions"]["Y1/coordinated"]["N"] == 68256.0)
keep = [i for i in range(864) if i != rows[0]]
y = m["Y1/coordinated/dc"].numpy()[keep]
c = m["Y1/coordinated/ceiling"].numpy()[keep]
p0, p1 = y1["coordinated"][keep, 0, :], y1["coordinated"][keep, 1, :]
ex = stats_exact(y, p0, p1, c)
pr = stats_pairs_np(y, p0, p1, c)
F = lock["conditions"]["Y1/coordinated"]["envelope"]["bound"]
lab = classify(ex["g"], ex["defined"], Fraction(F))
check("SECONDARY (descriptive only): Y1/coordinated without that pair (863 pairs) keeps the label", lab == "PASS" == results["Y1/coordinated"]["label"],
      f"g {float(ex['g']):.16f} (float {pr['g']:.16f}); g − F {float(ex['g'] - Fraction(F)):+.3e}; Δg vs primary {float(ex['g']) - results['Y1/coordinated']['g']:+.3e}")
# the pair itself
i = rows[0]
yy, cc = m["Y1/coordinated/dc"].numpy()[i], m["Y1/coordinated/ceiling"].numpy()[i]
pp0, pp1 = y1["coordinated"][i, 0], y1["coordinated"][i, 1]
print(f"INFO  the shiny pair: SSE0 {((yy - pp0) ** 2).sum():.4f}, SSE1 {((yy - pp1) ** 2).sum():.4f}, SSEC {((yy - cc) ** 2).sum():.4f}; mean Δc {yy.mean():.4f}")

# ---------------------------------------------------------------- the floors against the calibration draws (bonus)
cal = json.loads((E / "calibration-v1.json").read_text(encoding="utf-8"))
draws = torch.load(OUT / "draw-values.pt")
check("draw-values.pt: my tensor digests == the calibration record's draw_arrays_sha256", {k: my_tensor_digest(v) for k, v in draws.items()} == cal["draw_arrays_sha256"])
for key in results:
    g = draws[f"{key}/g"].numpy().astype(np.float64)
    d = draws[f"{key}/defined"].numpy().astype(bool)
    v = np.sort(np.where(d, g, -np.inf), kind="stable")
    check(f"{key}: F == the 250th ascending of the 10,000 draws (undefined at −inf) == lock", len(v) == 10000 and float(v[249]) == lock["conditions"][key]["envelope"]["bound"],
          f"undefined {int((~d).sum())}; CDF of fresh g among defined draws {float(((g <= results[key]['g']) & d).sum() / d.sum()):.4f}")
print("\nSCORE:", "ALL PASS" if not FAIL else f"FAILED {FAIL}")
