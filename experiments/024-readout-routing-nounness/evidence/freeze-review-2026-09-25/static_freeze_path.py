"""Static (AST, no execution of run.py) inspection of the production freeze code path at HEAD: every name, attribute
and call reachable from Runner.freeze through rr's freeze functions, against a list of model/outcome quantities."""
import rguard  # noqa: F401

import ast
import subprocess
from pathlib import Path

ROOT = Path(rguard.ROOT)
RR = ROOT / "src/neural_decompiler/readout_routing.py"
RUN = ROOT / "experiments/024-readout-routing-nounness/run.py"
B0C = ROOT / "src/neural_decompiler/block0_completion.py"
UL = ROOT / "src/neural_decompiler/upstream_localization.py"

# the committed blobs at HEAD equal the working tree files (so the inspected code is the run commit's code)
for path in (RR, RUN, B0C, UL):
    head = subprocess.run(["git", "--no-optional-locks", "rev-parse", f"HEAD:{path.relative_to(ROOT)}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    work = subprocess.run(["git", "--no-optional-locks", "hash-object", "--no-filters", str(path)], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    print(f"{path.relative_to(ROOT)}: HEAD blob {head[:12]} working {work[:12]} {'same' if head == work else 'DIFFERENT'}")


def functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, []).append(node)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.setdefault(f"{node.name}.{item.name}", []).append(item)
    return out


def names_in(node):
    found = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)
            if isinstance(sub.value, ast.Name):
                found.add(f"{sub.value.id}.{sub.attr}")
    return found


rr_f, run_f, b0c_f, ul_f = functions(RR), functions(RUN), functions(B0C), functions(UL)
path_funcs = {
    "run.Runner.freeze": run_f["Runner.freeze"], "run.Runner._base": run_f["Runner._base"], "run.Runner._inputs": run_f["Runner._inputs"],
    "run._load_tokenizer": run_f["_load_tokenizer"], "run._manifest_keys_022": run_f["_manifest_keys_022"],
    "rr.freeze_payload": rr_f["freeze_payload"], "rr.exclusion": rr_f["exclusion"], "rr.target_noun_form_ids": rr_f["target_noun_form_ids"],
    "rr.frame_token_ids": rr_f["frame_token_ids"], "rr._digested": rr_f["_digested"], "rr._status": rr_f["_status"], "rr.select_cues": rr_f["select_cues"],
    "rr.confirmation_from_payload": rr_f["confirmation_from_payload"], "rr.load_confirmation_024": rr_f["load_confirmation_024"],
    "rr.Confirmation024.manifest": rr_f["Confirmation024.manifest"], "rr.Confirmation024.counts": rr_f["Confirmation024.counts"],
    "rr.Confirmation024.target_prompts": rr_f["Confirmation024.target_prompts"], "rr.Configuration.to_json": rr_f["Configuration.to_json"],
    "rr.verify_023_inputs": rr_f["verify_023_inputs"], "b0c.exclusion": b0c_f["exclusion"], "b0c.load_confirmation_023": b0c_f["load_confirmation_023"],
    "b0c.verify_022_inputs": b0c_f["verify_022_inputs"], "ul.extract_exclusion": ul_f["extract_exclusion"],
}
FORBIDDEN = {"nounness", "cosine", "centroid", "centroids", "score_bindings", "full_scores", "calibration_scores", "embedding_digest", "parameters_digest",
             "W_E", "Weights", "from_model", "load_model", "model_loader", "capture_prompt", "run_patched", "run_capture", "run_interventions", "ModelPrograms",
             "cue_mse", "read_exposed_cells", "read_cells", "exposed_cue_mse", "calibration_population", "contrast_of", "stage_two_022", "measure_prompt",
             "fresh_quantities", "line_predictions", "extrapolation", "ols", "predict", "spearman", "logits", "hidden", "dx3", "Δx3", "delta_c",
             "pair_cells", "SSEC", "locked_states", "exploration", "torch", "numpy", "np"}
total_hits = {}
for name, nodes in path_funcs.items():
    used = set().union(*(names_in(n) for n in nodes))
    hits = sorted(x for x in used if x.split(".")[-1] in FORBIDDEN or x in FORBIDDEN)
    print(f"{name}: {len(used)} names; forbidden hits: {hits}")
    if hits:
        total_hits[name] = hits
print("\nforbidden names on the freeze path:", total_hits or "none")
# what Runner.freeze calls, in order (attribute calls)
freeze = run_f["Runner.freeze"][0]
calls = [ast.unparse(n.func) for n in ast.walk(freeze) if isinstance(n, ast.Call)]
print("\nRunner.freeze calls:", calls)
