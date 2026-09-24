"""Item 8: calibration independence (a capability check, NOT the calibrate phase).

Run as its own Python process. Before anything else is imported, an audit hook refuses every "open" of Experiment
022's calibration table (by resolved path and by inode) and torch.load is patched to refuse that path. The guard is
proven live with deliberate probes. Then, from the candidate artifact alone: b0c.read_cells, b0c.draw_indices(units,
16), b0c.draw_pairs for the four conditions, and b0c.pool -> b0c.statistics -> b0c.result_codes with an ILLUSTRATIVE
bound of 0.95 (NOT the calibrated envelope), each quantity cross-checked against this reviewer's own implementation from
the artifact's raw columns. Finally: the number of guard triggers besides the deliberate probes (expected 0)."""

import os
import sys

REPO = "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler"
SCRATCH = "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/extractreview023"
TABLE = os.path.realpath(os.path.join(REPO, "outputs/experiment-022/calibration-table.pt"))
_TABLE_STAT = os.stat(TABLE)
TABLE_ID = (_TABLE_STAT.st_dev, _TABLE_STAT.st_ino)

GUARD = {"audit": 0, "torch_load": 0, "armed": True}
OUTPUT_OPENS = set()


def _is_table(path) -> bool:
    if isinstance(path, int):
        return False
    try:
        text = os.fsdecode(os.fspath(path))
    except TypeError:
        return False
    try:
        if os.path.realpath(text) == TABLE:
            return True
    except (OSError, ValueError):
        pass
    try:
        st = os.stat(text)
        return (st.st_dev, st.st_ino) == TABLE_ID
    except (OSError, ValueError):
        return False


def _audit(event, args):
    if event == "open" and args:
        path = args[0]
        if _is_table(path):
            GUARD["audit"] += 1
            raise PermissionError(f"audit guard: an open of Experiment 022's calibration table was refused ({path!r})")
        if isinstance(path, (str, bytes)) and "/outputs/" in os.fsdecode(path):
            OUTPUT_OPENS.add(os.fsdecode(path))


sys.addaudithook(_audit)

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_orig_torch_load = torch.load


def _guarded_load(f, *args, **kwargs):
    if _is_table(f):
        GUARD["torch_load"] += 1
        raise PermissionError(f"torch.load guard: Experiment 022's calibration table was refused ({f!r})")
    return _orig_torch_load(f, *args, **kwargs)


torch.load = _guarded_load

LINES = []
FAILS = []


def check(label, ok, detail=""):
    line = f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else "")
    print(line, flush=True)
    LINES.append(line)
    if not ok:
        FAILS.append(label)


# --- the deliberate probes: every route must be refused ------------------------------------------------------------------
link = os.path.join(SCRATCH, "table-symlink-probe.pt")
if os.path.lexists(link):
    os.remove(link)
os.symlink(TABLE, link)
probes = {
    "builtins.open(abs)": lambda: open(TABLE, "rb"),
    "builtins.open(relative to repo cwd)": lambda: open("outputs/experiment-022/calibration-table.pt", "rb"),
    "os.open": lambda: os.open(TABLE, os.O_RDONLY),
    "pathlib.read_bytes": lambda: Path(TABLE).read_bytes(),
    "numpy.fromfile": lambda: np.fromfile(TABLE, dtype=np.uint8, count=8),
    "torch.load (patched)": lambda: torch.load(TABLE),
    "torch.load (original, audit only)": lambda: _orig_torch_load(TABLE),
    "open via a symlink in the scratchpad": lambda: open(link, "rb"),
}
refused = {}
for name, attempt in probes.items():
    try:
        handle = attempt()
        refused[name] = False
        try:
            (os.close(handle) if isinstance(handle, int) else handle.close())
        except Exception:  # noqa: BLE001
            pass
    except PermissionError:
        refused[name] = True
os.remove(link)
probe_counts = dict(GUARD)
check("guard live: every deliberate probe refused", all(refused.values()), f"{refused}; triggers during probes {probe_counts}")
GUARD["audit"] = 0
GUARD["torch_load"] = 0

# --- the capability: the artifact alone ------------------------------------------------------------------------------
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

inputs = ul.load_frozen_inputs(Path(REPO))
units = b0c.exposed_units(inputs)
noun_keys = [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]
record_022 = json.loads(Path(REPO, "experiments/022-upstream-error-localization/calibration-v1.json").read_text(encoding="utf-8"))
meta = b0c.cells_meta(units, noun_keys, record_022)
data_path = Path(REPO, "outputs/experiment-023/candidate-exposed-cells.f64")
index_path = Path(REPO, "outputs/experiment-023/candidate-exposed-cells.json")
cells_t, loaded = b0c.read_cells(data_path, index_path, units=units, meta=meta)
own_cells = np.frombuffer(data_path.read_bytes(), dtype="<f8").reshape(18900, 8)
check("b0c.read_cells accepted the candidate (content digest, layout, meta recomputed now, bytes digest)", True)
check("b0c.read_cells tensor == own raw little-endian read, bit for bit", np.array_equal(cells_t.numpy(), own_cells) and cells_t.dtype == torch.float64)

# own draws and pair indices, from the index JSON alone
index = json.loads(index_path.read_text(encoding="utf-8"))
cues, frames = index["cues"], index["frames"]
fids = [f[0] for f in frames]
STRATA = ("determiner-like", "quantity", "adjective")
strata_rows = {s: [i for i, c in enumerate(cues) if c[2] == s] for s in STRATA}
y2_rows = {t: [fids.index(fid) for fid in index["y2_like_frames"][t]] for t in ("cardinal", "quantifier", "coordinated-adjective")}
group_frames = {"cue_final": [i for i, f in enumerate(frames) if f[2] == "cue_final"], "coordinated": [i for i, f in enumerate(frames) if f[2] == "coordinated"]}


def own_slot(b, stratum, slot, n):
    import hashlib

    return int.from_bytes(hashlib.sha256(f"023|primary|{b}|{stratum}|{slot}".encode()).digest()[:8], "big") % n


def own_pairs(b, population, group):
    cue_rows = [strata_rows[s][own_slot(b, f"cue/{s}", i, len(strata_rows[s]))] for s in STRATA for i in range(8)]
    if population == "Y1":
        frame_rows = group_frames[group]
    else:
        temps = ["coordinated-adjective"] if group == "coordinated" else ["cardinal", "quantifier"]
        frame_rows = [y2_rows[t][own_slot(b, f"frame/{t}", i, len(y2_rows[t]))] for t in temps for i in range(6)]
    return [c * 108 + f for c in cue_rows for f in frame_rows]


BOUND = 0.95  # ILLUSTRATIVE ONLY — an arbitrary bound, NOT the calibrated envelope F (calibrate has not run)
RESULTS = ("NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS")
indices = b0c.draw_indices(units, 16)
worst = {"value": 0.0, "at": ""}
flag_mismatch = 0
code_mismatch = 0
pair_mismatch = 0
e6_worst = 0.0
tally = {}
rows_out = []
for population in ("Y1", "Y2"):
    for group in ("cue_final", "coordinated"):
        condition = f"{population}/{group}"
        pairs = b0c.draw_pairs(units, indices, population, group, range(16))
        pooled = b0c.pool(cells_t, pairs)
        stats = b0c.statistics(pooled)
        codes = b0c.result_codes(stats["g"], stats["defined"], BOUND)
        for b in range(16):
            sel = own_pairs(b, population, group)
            if sel != pairs[b].tolist():
                pair_mismatch += 1
            rows = own_cells[sel]
            N, S, Q = (math.fsum(rows[:, k].tolist()) for k in (0, 1, 2))
            SSE0, SSE1, SSEC = (math.fsum(rows[:, k].tolist()) for k in (3, 4, 5))
            SST = Q - S * S / N
            two_pass = math.fsum(rows[:, 7].tolist()) + math.fsum((rows[:, 0] * (rows[:, 6] - S / N) ** 2).tolist())
            e6_worst = max(e6_worst, abs(SST - two_pass) / abs(two_pass))
            defined = SST > 0 and (SSE0 - SSEC) >= 0.02 * SST
            g = (SSE0 - SSE1) / (SSE0 - SSEC) if defined else math.nan
            mine = {"N": N, "S": S, "Q": Q, "SSE0": SSE0, "SSE1": SSE1, "SSEC": SSEC, "SST": SST, "g": g, "gap": (SSE0 - SSEC) / SST,
                    "R2_0": 1 - SSE0 / SST, "R2_1": 1 - SSE1 / SST, "R2_C": 1 - SSEC / SST}
            code = "NOT_INTERPRETABLE" if not defined else "GUARD_FAILURE" if g < 0.90 else "ENVELOPE_ONLY_FAILURE" if g < BOUND else "PASS"
            theirs = {**{k: float(pooled[k][b]) for k in ("N", "S", "Q", "SSE0", "SSE1", "SSEC", "SST")},
                      **{k: float(stats[k][b]) for k in ("g", "gap", "R2_0", "R2_1", "R2_C")}}
            for key, d in mine.items():
                k = theirs[key]
                if math.isnan(d) and math.isnan(k):
                    diff = 0.0
                elif math.isnan(d) != math.isnan(k):
                    diff = math.inf
                else:
                    diff = abs(k - d) / max(1.0, abs(d))
                if diff > worst["value"]:
                    worst = {"value": diff, "at": f"{condition}/draw {b}/{key}"}
            if bool(stats["defined"][b]) != defined or bool(stats["ceiling_limited"][b]) != (mine["R2_C"] < 0.80):
                flag_mismatch += 1
            if RESULTS[int(codes[b])] != code:
                code_mismatch += 1
            tally.setdefault(condition, {}).setdefault(code, 0)
            tally[condition][code] += 1
            rows_out.append((condition, b, round(g, 6), round(mine["gap"], 4), round(mine["R2_0"], 4), round(mine["R2_1"], 4), round(mine["R2_C"], 4), code))

check("own pair indices == b0c.draw_pairs for the 16 draws × 4 conditions", pair_mismatch == 0, f"mismatching draws {pair_mismatch}")
check("b0c pool/statistics vs own fsum implementation: max |k − d| / max(1, |d|) <= 1e-10", worst["value"] <= 1e-10, f"{worst['value']:.3e} at {worst['at']}")
check("interpretability and ceiling_limited flags identical (64 draw-conditions)", flag_mismatch == 0, f"mismatches {flag_mismatch}")
check("four-way result codes identical under the ILLUSTRATIVE bound 0.95", code_mismatch == 0, f"mismatches {code_mismatch}")
check("E6 on these draws (own one-pass vs own two-pass merge) <= 1e-10", e6_worst <= 1e-10, f"{e6_worst:.3e}")
print(f"[NOTE] ILLUSTRATIVE-bound (0.95, not F) code tally over the 16 draws: {tally}", flush=True)
gs = {}
for condition, b, g, gap, r0, r1, rc_, code in rows_out:
    gs.setdefault(condition, []).append(g)
print("[NOTE] g over the 16 draws (min / max): " + "; ".join(f"{c} {min(v):.4f} / {max(v):.4f}" for c, v in gs.items()), flush=True)
after = dict(GUARD)
check("guard triggers other than the deliberate probes == 0", after["audit"] == 0 and after["torch_load"] == 0, f"{after}")
print(f"[NOTE] files opened under outputs/ during the run: {sorted(OUTPUT_OPENS)}", flush=True)
print(f"== item 8: {len(LINES)} checks, {len(FAILS)} failures: {FAILS}", flush=True)
sys.exit(1 if FAILS else 0)
