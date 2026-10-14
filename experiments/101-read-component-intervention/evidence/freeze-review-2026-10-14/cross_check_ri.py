"""AFTER the independent derivation: cross-check with ri.exclusions_from_committed / ri.freeze_payload / Runner()._base().

Guards (installed before neural_decompiler is imported): an audit hook that refuses every write, delete, rename or
mkdir inside the repository and every open of a weight file anywhere, and logs every repository path opened for
reading; torch.nn.Module.__call__ and models.load_model and the HF model constructors refuse. Nothing is written
except this script's own JSON beside it (outside the repository).
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
WEIGHT_SUFFIXES = (".safetensors", "pytorch_model.bin", ".ckpt")
log = {"repo_reads": set(), "outputs_reads": set(), "refused": [], "weight_open_attempts": []}


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            text = str(Path(os.fsdecode(args[0])).resolve())
        except OSError:
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if text.endswith(WEIGHT_SUFFIXES):
            log["weight_open_attempts"].append(text)
            raise PermissionError(f"weight file refused: {text}")
        if text.startswith(str(ROOT) + "/") and "/.venv/" not in text:
            if writing:
                log["refused"].append(["write", text])
                raise PermissionError(f"repository write refused: {text}")
            log["repo_reads"].add(text)
            if text.startswith(str(ROOT / "outputs") + "/"):
                log["outputs_reads"].add(text)
    elif event in ("os.remove", "os.unlink", "os.rename", "os.replace", "os.rmdir", "shutil.rmtree", "shutil.move", "os.mkdir", "os.truncate", "os.chmod"):
        for arg in args:
            if isinstance(arg, (str, bytes, os.PathLike)):
                raw = os.fsdecode(arg)
                if os.path.isabs(raw) and str(Path(raw).resolve()).startswith(str(ROOT) + "/") and "/.venv/" not in raw:
                    log["refused"].append([event, raw])
                    raise PermissionError(f"{event} refused: {raw}")


sys.addaudithook(_hook)
import torch  # noqa: E402


def _refuse(name):
    def inner(*a, **k):
        log["refused"].append(["call", name])
        raise RuntimeError(f"{name} refused in the review cross-check")
    return inner


torch.nn.Module.__call__ = _refuse("torch.nn.Module.__call__")
from neural_decompiler import models  # noqa: E402

models.load_model = _refuse("models.load_model")
import transformers  # noqa: E402

for cls_name in ("AutoModel", "AutoModelForCausalLM", "GPTNeoXForCausalLM", "GPTNeoXModel"):
    cls = getattr(transformers, cls_name, None)
    if cls is not None:
        cls.from_pretrained = _refuse(f"transformers.{cls_name}.from_pretrained")
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment_101_runner_review", ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
runner_module.load_model = _refuse("runner.load_model")

out = {}
file_bytes = (ROOT / ri.CONFIRMATION_RELATIVE_PATH).read_bytes()
file_payload = json.loads(file_bytes)
out["configuration_equals_PRODUCTION"] = file_payload["configuration"] == ri.PRODUCTION.to_json()
out["holm_thresholds_recomputed"] = list(ri.holm_thresholds(36))
out["holm_tails"] = [float(ri.binomial_tail(36, 26)), float(ri.binomial_tail(36, 25)), float(ri.binomial_tail(36, 24))]
out["tau_rank_recomputed"] = ri.nearest_rank(36, 90)
out["design_plan_constants"] = {"DESIGN": ri.DESIGN, "PLAN": ri.PLAN}
out["design_plan_equal_file"] = file_payload["design"] == ri.DESIGN and file_payload["plan"] == ri.PLAN
out["frame_candidates_equal_design_parse"] = list(ri.FRAME_CANDIDATES) == json.load(open(HERE / "independent_derivation.json"))["out"]["candidates_from_design"]
out["number_words_equal"] = list(ri.NUMBER_WORDS) == json.load(open(HERE / "independent_derivation.json"))["NUMBER_WORDS"]

runner = runner_module.Runner(log=lambda m: None)
base = runner._base()
out["runner_forbidden_size"] = len(base.forbidden)
mine_forbidden_sha = json.load(open(HERE / "spent_set.json"))["forbidden_sha256"]
out["runner_forbidden_equals_independent"] = hashlib.sha256(pm.canonical_json(sorted(base.forbidden)).encode()).hexdigest() == mine_forbidden_sha
out["runner_manifests_size"] = len(base.manifests)
out["exclusions_record_equals_file"] = dict(base.exclusions.record) == file_payload["exclusions"]

tokenizer = runner.tokenizer_loader(models.PYTHIA_70M)
payload = ri.freeze_payload(tokenizer, exclusions=base.exclusions, config=ri.PRODUCTION)
regen = (pm.canonical_json(payload) + "\n").encode("utf-8")
out["regenerated_payload_equals_file_payload"] = payload == file_payload
out["regenerated_bytes_equal_file_bytes"] = regen == file_bytes
out["regenerated_file_sha256"] = hashlib.sha256(regen).hexdigest()
conf = ri.confirmation_from_payload(file_payload, ri.PRODUCTION, verify=True)
out["confirmation_from_payload_verify_ok"] = conf.content_sha256 == file_payload["content_sha256"]
loaded = ri.load_confirmation_101(ROOT / ri.CONFIRMATION_RELATIVE_PATH, base.exclusions, ri.PRODUCTION)
out["load_confirmation_101_ok"] = loaded.content_sha256 == file_payload["content_sha256"]
out["untagged_keys"] = len(conf.untagged_keys())
out["collisions_with_runner_forbidden"] = len((conf.untagged_keys() | set(conf.s1_keys()) | set(conf.s2_keys())) & base.forbidden)
out["s2_key_order_frame1"] = conf.s2_keys()[:55]
out["guard_log"] = {"refused": log["refused"], "weight_open_attempts": log["weight_open_attempts"], "outputs_reads": sorted(log["outputs_reads"]),
                    "n_repo_reads": len(log["repo_reads"])}
out["repo_reads"] = sorted(p.replace(str(ROOT) + "/", "") for p in log["repo_reads"])
json.dump(out, open(HERE / "cross_check_ri.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("s2_key_order_frame1", "repo_reads")}, indent=1))
