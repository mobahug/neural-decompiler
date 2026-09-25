"""Post-calibration verification of Experiment 024 (read-only; weights only for the expected lock values; no prompt).

The candidate record verified by the frozen checker; its bindings against the expected digests; F_ρ, the null and the
effective threshold recomputed from scratch from the record's own calibration entries and the SHA schemes, and from
the saved arrays; the state; and — as expected values for the lock, not a lock — the 40 scores and the extrapolation
count computed in memory against this record.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
REFUSED = []


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if not writing:
        return
    try:
        path = Path(os.fsdecode(args[0])).resolve()
    except OSError:
        return
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
        REFUSED.append(str(path))
        raise PermissionError(f"the verification may not write into the repository: {path}")


sys.addaudithook(_hook)

import torch  # noqa: E402

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

out = ROOT / "outputs/experiment-024"
record_path = out / "candidate-calibration.json"
record = json.loads(record_path.read_text(encoding="utf-8"))
rr.verify_calibration_record(record, rr.PRODUCTION)
print(f"record {record_path.name}: file sha256 {rc.file_sha256(record_path)}; content sha256 {record['content_sha256']} (recomputed {rc.content_digest(record)}); "
      f"verify_calibration_record: ok")
state = rd.load_results_state(out / "results.json")
print(f"state sha256 {state['state_sha256']}; phases {({k: v['status'] for k, v in state['phases'].items()})}; ledger {len(state['executed_prompt_keys'])} keys; "
      f"calibrate commit {state['phases']['calibrate']['commit'][:7]}; configuration {state['configuration']['name']}")
print(f"state records the candidate: {state['calibration']['record_sha256'] == rc.file_sha256(record_path)}; confirmation binding {record['confirmation_024']}")
score = record["score"]
dependencies = record["dependencies"]
print(f"bindings: embedding {score['embedding_sha256']}; noun ids {score['noun_row_ids_sha256']} ({len(score['noun_row_ids'])}); cue ids {score['calibration_cue_ids_sha256']} "
      f"({len(score['calibration_cue_ids'])}); mu_noun {score['mu_noun_sha256']}; mu_cue {score['mu_cue_sha256']}; parameters {dependencies['model']['parameters_sha256']}")
print(f"calibration source {record['exposed_cells']}")
entries = record["calibration_cues"]
assert len(entries) == 139 and all(entry["stratum"] in b0c.STRATA for entry in entries)
line = record["line"]
print(f"line: slope {line['slope']!r}, intercept {line['intercept']!r}, residual sd {line['residual_sd']!r}, n {line['n']}")
descriptive = record["descriptive"]
print(f"calibration rho {descriptive['calibration_rho']!r}; within strata {descriptive['within_stratum_rho']}; pronoun check {descriptive['pronoun_check']}; "
      f"calibration maximum {descriptive['maximum_calibration_score']!r}; draw rates {descriptive['draw_rates']}")
floor, null = record["primary_floor"], record["null"]
print(f"F_rho {floor['F_rho']!r} (element [{floor['element']}] of {floor['draws']}); undefined {floor['undefined']}; direction {floor['direction_check']}; tails {floor['tails']}")
print(f"null_975 {null['null_975']!r} (element [{null['element']}] of {null['permutations']}); median {null['median']!r}; permutations sha256 {null['permutations_sha256']}")
print(f"effective threshold {record['effective_threshold']}; checks: mse {record['checks']['mse']['max_difference']!r}, spearman {record['checks']['spearman']['max_difference']!r} "
      f"over {record['checks']['spearman']['n_checked']}")
# the saved arrays against the record, and the two order statistics recomputed from them
arrays = torch.load(out / "calibration-arrays.pt")
assert rr.arrays_digests(arrays) == record["arrays_sha256"]
draws = sorted(float(v) for v, d in zip(arrays["draw_rho"].tolist(), arrays["draw_defined"].tolist()) if d)
nulls = sorted(arrays["null_rho"].tolist())
assert draws[249] == floor["F_rho"] and nulls[97_499] == null["null_975"] and len(nulls) == 100_000
ties = sum(1 for v in nulls if v == null["null_975"])
print(f"arrays: digests match the record; draws min {draws[0]!r} median {rr._median(draws)!r} max {draws[-1]!r}; null min {nulls[0]!r} max {nulls[-1]!r}; "
      f"null values equal to the threshold {ties} (the null is discrete; the order statistic is well defined)")
# from scratch: the draws and the null by the frozen SHA schemes, from the record's own entries
config = rr.PRODUCTION
again = rr.primary_draws(entries, config)
assert again["values"] == [None if not d else v for v, d in zip(arrays["draw_rho"].tolist(), arrays["draw_defined"].tolist())]
assert rr.primary_floor(again["values"], config) == floor
null_again = rr.null_distribution(config)
assert null_again["permutations_sha256"] == null["permutations_sha256"] and sorted(null_again["values"])[97_499] == null["null_975"]
assert rr.ols([e["nounness_loo"] for e in entries], [e["log_mse"] for e in entries]) == {**line}
print("from scratch: the 10,000 draws, F_rho, the 100,000 permutations, null_975 and the line all reproduce exactly")
# expected lock values against this record (in memory; weights only; every forward refused)
calls = {"during_load": 0}
original_call = torch.nn.Module.__call__
torch.nn.Module.__call__ = lambda self, *a, **k: (calls.__setitem__("during_load", calls["during_load"] + 1), original_call(self, *a, **k))[1]
model = models.load_model(models.PYTHIA_70M)
torch.nn.Module.__call__ = lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("forward refused"))
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("capture refused")))
assert rr.parameters_digest(model) == dependencies["model"]["parameters_sha256"]
W_E = pm.Weights.from_model(model).W_E
del model
inputs = ul.load_frozen_inputs(ROOT)
assert rr.score_bindings(W_E, inputs.pool, b0c.exposed_units(inputs)) == score
c022_path = ROOT / ul.CONFIRMATION_RELATIVE_PATH
c022 = json.loads(c022_path.read_text())
c023_path = ROOT / b0c.CONFIRMATION_RELATIVE_PATH
excluded = rr.exclusion(b0c.exclusion(inputs, c022, rc.file_sha256(c022_path)), json.loads(c023_path.read_text()), rc.file_sha256(c023_path))
confirmation = rr.load_confirmation_024(ROOT / rr.CONFIRMATION_RELATIVE_PATH, inputs.pool, excluded, config)
fresh = rr.fresh_quantities(W_E, score, confirmation, record)
print(f"expected lock values (in memory, not a lock): scores sha256 {fresh['scores_sha256']}; calibration maximum {fresh['maximum_calibration_score']!r}; "
      f"E above {fresh['extrapolation']['E_above_maximum']} of {fresh['extrapolation']['E']}; module calls during the load {calls['during_load']}")
print(f"repository writes refused: {REFUSED}")
