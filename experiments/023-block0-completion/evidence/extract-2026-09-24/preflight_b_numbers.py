"""Experiment 023 pre-extract preflight, part B: E1, E2, E3, E5, E6 on the real local 022 table with the frozen
implementation's own functions, read-only; explicit per-column counts over all 18,900 exposed pairs. Writes nothing."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

t0 = time.time()
inputs = ul.load_frozen_inputs(ROOT)
units = b0c.exposed_units(inputs)
noun_keys = [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]
record = json.loads((ROOT / ul.CALIBRATION_RELATIVE_PATH).read_text())
table = torch.load(ROOT / b0c.TABLE_022_RELATIVE_PATH)
failures = []
e1 = b0c.verify_table_022(table, record, units, noun_keys)
print(f"E1 passed: {len(e1['tensors'])} tensor digests, orders {e1['orders']}")
cells = b0c.cells_from_table(table, units)
n_pairs = units.n_pairs
sse = table["sse"].reshape(n_pairs, -1)
p1 = torch.tensor([b0c.P1_MASK[units.group_of(i % len(units.frames))] for i in range(n_pairs)])
stored = {"n": table["count"].reshape(-1), "SSE0": sse[:, b0c.P0_MASK], "SSE1": sse[torch.arange(n_pairs), p1], "mean": table["mean"].reshape(-1), "M2": table["m2"].reshape(-1)}
for name, values in stored.items():
    differing = int((cells[:, b0c.CELL_COLUMNS.index(name)] != values.double()).sum())
    print(f"E2 {name}: {differing} of {n_pairs} pairs differ from 022's stored cells")
    failures += [f"E2 {name}"] if differing else []
independent = b0c.independent_sums(table, units)
for k, name in enumerate(b0c.E3_COLUMNS):
    differing = int((cells[:, b0c.CELL_COLUMNS.index(name)] != independent[:, k]).sum())
    print(f"E3 {name}: {differing} of {n_pairs} pairs differ from the independent whole-table computation")
    failures += [f"E3 {name}"] if differing else []
checks = b0c.verify_cells_against_table(cells, table, units)
print("verify_cells_against_table:", json.dumps(checks))
draws = b0c.verify_draws_against_table(cells, table, units)
print("E5/E6:", json.dumps(draws))
contiguous = {name: bool(table[name].is_contiguous()) for name in ("dc", "ceiling", "dc_hat")}
print("tensor layout contiguous:", contiguous, "dtypes:", {name: str(table[name].dtype) for name in ("dc", "ceiling")})
print(f"cells shape {tuple(cells.shape)}; {time.time() - t0:.1f}s")
print("PREFLIGHT B NUMBERS:", "ALL PASS" if not failures else f"FAILED {failures}")
sys.exit(1 if failures else 0)
