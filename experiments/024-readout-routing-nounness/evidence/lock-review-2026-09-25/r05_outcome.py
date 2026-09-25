"""Item 8: the frozen hierarchy, labels only (no outcome data): rr.outcome over every (primary, guard) pair against an
own transcription of the design's table; classify_primary's precedence at the bound thresholds on boundary ρ values
(threshold arithmetic only, no fresh data); secondary paths cannot reach the label."""
import ast, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_guard  # noqa
from neural_decompiler import readout_routing as rr
ROOT = review_guard.ROOT

def design(primary, guard):  # own transcription of design rev 2 "Outcome (frozen)"
    if primary == "NOT_INTERPRETABLE":
        return "NOT_INTERPRETABLE"
    if primary in ("GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE"):
        return "NOUNNESS_PREDICTION_NOT_ESTABLISHED"
    return "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS" if guard == "PASS" else "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED"

pairs = [(p, g) for p in rr.PRIMARY_RESULTS for g in rr.GUARD_RESULTS]
bad = [(p, g, rr.outcome(p, g), design(p, g)) for p, g in pairs if rr.outcome(p, g) != design(p, g)]
print("12 (primary, guard) pairs; disagreements with the design table:", bad)
lock = json.loads((ROOT / "outputs/experiment-024/candidate-lock.json").read_bytes())
F, N = lock["primary"]["F_rho"], lock["primary"]["null_975"]
import math
probe = {"just below null": math.nextafter(N, -1.0), "at null": N, "at F (below null)": F, "None": None}
print("classify_primary at the bound thresholds:", {k: rr.classify_primary(v, F, N) for k, v in probe.items()})
print("envelope-only band empty (F < null):", F < N)
src = (ROOT / "src/neural_decompiler/readout_routing.py").read_text()
tree = ast.parse(src)
callers = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) in ("outcome", "classify_primary", "en_guard", "en_decision"):
                callers.setdefault(sub.func.id, []).append(node.name)
print("callers inside rr:", callers)
run_src = (ROOT / "experiments/024-readout-routing-nounness/run.py").read_text()
print("run.py references to outcome/classify/en_guard/label assignment:", [l.strip()[:110] for l in run_src.splitlines() if "rr.outcome" in l or "classify_primary" in l or "en_guard" in l or '["label"] =' in l])
desc = [l.strip()[:120] for l in run_src.splitlines()[613:634]]
print("_descriptives writes only descriptives/failures:", all("results\"] =" not in l and "\"outcome\"" not in l for l in desc))
ok = not bad and rr.classify_primary(math.nextafter(N, -1.0), F, N) == "GUARD_FAILURE" and rr.classify_primary(N, F, N) == "PASS" and callers.get("outcome") == ["score"]
print("RESULT:", "ok" if ok else "FAIL", "; guard:", review_guard.EVENTS)
