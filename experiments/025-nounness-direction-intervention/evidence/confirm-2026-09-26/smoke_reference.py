"""Experiment 025 guard smoke test — the UNGUARDED reference process (engineering only, spent prompts only).

It uses no launcher, no forward wrappers and no class or instance sealing. Only a key allow-list stands in front of
the model: the three spent reference prompts, one per template family, each in 020's ledger.

For each prompt it runs:
- a plain capture;
- a θ = 0 patched run, which replaces EMBED@p_c with the model's own float32 row (the permitted patched route).

Both capture exactly the sites confirm's patch path uses: the measured residuals at p_c and p_t, ATTN_PATTERN.L4/L5 at
p_t, EMBED@p_c and RESID_PRE.L0@p_c.

It records the logits and every captured slice, as tensors and digests, and the original `inspect.signature` of every
module instance's `forward` and of the class methods the launcher seals. Nothing scientific is computed: no ℓ, MSE,
A, B, G, D_attn, Level-1, Δx3 or Δc. Nothing in the repository is written; the outputs go to this scratchpad
directory.
"""
import gzip
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import models
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import upstream_localization as ul

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)
assert Path.cwd().resolve() == ROOT
SMOKE_KEYS = ("cardinal-009-1|ref|581", "quantifier-009-1|ref|1016", "coordinated-adjective-009-1|ref|581")
EXPECTED_HEAD = "6f72d7d8afcaa079ee38256b7c1250352b1b1bdd"


def git(*args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert git("rev-parse", "HEAD") == git("rev-parse", "origin/main") == git("ls-remote", "origin", "refs/heads/main").split()[0] == EXPECTED_HEAD
assert git("status", "--porcelain", "--untracked-files=all") == ""
outputs_before = {p.name: sha256_file(p) for p in sorted((ROOT / "outputs/experiment-025").iterdir())}

# The keys are spent (020's ledger), and none is a patch-path key or an Experiment 025 manifest key.
inputs = ul.load_frozen_inputs(ROOT)
ledger = inputs.closure["ledger"]
archive = ROOT / "experiments/023-block0-completion/evidence/archive/experiment-020-results.json.gz"
archived_ledger = set(json.loads(gzip.decompress(archive.read_bytes()))["executed_prompt_keys"])
freeze = json.loads((ROOT / cr.CONFIRMATION_RELATIVE_PATH).read_text())
manifest_prompts = {cr.untagged(key) for key in freeze["manifest"]["S2-TARGET"]}
proof = {key: {"in_020_ledger": key in ledger, "in_committed_020_archive": key in archived_ledger, "patch_path_key": key in cr.PATCH_PATH_SPENT_KEYS,
               "in_025_manifest": key in manifest_prompts} for key in SMOKE_KEYS}
assert all(v["in_020_ledger"] and v["in_committed_020_archive"] and not v["patch_path_key"] and not v["in_025_manifest"] for v in proof.values()), proof

allowed = set(SMOKE_KEYS)
executed: list[tuple[str, str]] = []
original_capture, original_patched = pm.capture_prompt, pm.run_patched


def guarded_capture(model, prompt, sites):
    if prompt.key not in allowed:
        raise AssertionError(f"a non-allow-listed prompt: {prompt.key}")
    executed.append(("plain", prompt.key))
    return original_capture(model, prompt, sites)


def guarded_patched(model, prompt, replacements, sources, *, capture_sites=()):
    own = model.embed.W_E[int(prompt.cue_token_id)].detach().to("cpu", torch.float32)
    if prompt.key not in allowed or set(replacements) != {("EMBED", prompt.p_c)} or not torch.equal(replacements[("EMBED", prompt.p_c)].detach().to("cpu").reshape(-1), own):
        raise AssertionError(f"a non-allow-listed patched run: {prompt.key}")
    executed.append(("patched-theta0", prompt.key))
    return original_patched(model, prompt, replacements, sources, capture_sites=capture_sites)


pm.capture_prompt, pm.run_patched = guarded_capture, guarded_patched

model = models.load_model(models.PYTHIA_70M)


def signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


instance_signatures = {name or "<root>": signature(module.forward) for name, module in model.named_modules()}
class_signatures = {}
for cls in sorted({type(m) for m in model.modules()}, key=lambda c: c.__qualname__):
    for method in ("forward", "run_with_hooks", "run_with_cache", "generate"):
        if method in cls.__dict__ and callable(cls.__dict__[method]):
            class_signatures[f"{cls.__qualname__}.{method}"] = signature(cls.__dict__[method])
W_E32 = pm.Weights.from_model(model).W_E
frames = {frame.frame_id: frame for frame in inputs.pool.frames}
tensors: dict[str, torch.Tensor] = {}
digests: dict[str, dict] = {}
for key in SMOKE_KEYS:
    frame_id, word, token_id = key.split("|")
    frame = frames[frame_id]
    prompt = pm.Prompt(frame, int(token_id), word)
    site = ("EMBED", frame.p_c)
    sites = ul.measured_sites(frame) + cr.attention_sites(frame) + [site, ("RESID_PRE.L0", frame.p_c)]
    plain = pm.capture_prompt(model, prompt, sites)
    patched = pm.run_patched(model, prompt, {site: W_E32[int(token_id)].detach().reshape(1, 1, -1).to(torch.float32)}, {site: pm.ReplacementSource.DIRECT},
                             capture_sites=sites)
    entry = {}
    for kind, run in (("plain", plain), ("patched", patched)):
        tensors[f"{key}/{kind}/logits"] = run.logits.clone()
        for label, value in run.slices.items():
            tensors[f"{key}/{kind}/{label}"] = value.clone()
        entry[kind] = {"logits_sha256": rc.tensor_digest(run.logits.double()), "slices": sorted(run.slices),
                       "slices_sha256": {label: rc.tensor_digest(value.double()) for label, value in sorted(run.slices.items())},
                       "shapes": {label: list(value.shape) for label, value in run.slices.items()}, "dtypes": sorted({str(v.dtype) for v in run.slices.values()})}
    entry["plain_equals_patched_theta0"] = torch.equal(plain.logits, patched.logits) and set(plain.slices) == set(patched.slices) \
        and all(torch.equal(plain.slices[label], patched.slices[label]) for label in plain.slices)
    entry["embed_equals_resid_pre0"] = all(torch.equal(run.vector(site), run.vector(("RESID_PRE.L0", frame.p_c))) for run in (plain, patched))
    entry["patch_integrity"] = [dict(item) for item in patched.integrity]
    digests[key] = entry
torch.save(tensors, HERE / "smoke_reference.pt")
outputs_after = {p.name: sha256_file(p) for p in sorted((ROOT / "outputs/experiment-025").iterdir())}
record = {"head": git("rev-parse", "HEAD"), "keys": list(SMOKE_KEYS), "spent_proof": proof, "ledger_source": {
    "020_results_state": "outputs/experiment-020/results.json (executed_prompt_keys), verified by rc.verify_020_closure against the committed closure's file and state digests",
    "committed_archive": str(archive.relative_to(ROOT)), "committed_archive_sha256": sha256_file(archive)},
    "executed": executed, "forwards": len(executed), "digests": digests, "instance_signatures": instance_signatures, "class_signatures": class_signatures,
    "reference_tensors_sha256": sha256_file(HERE / "smoke_reference.pt"), "outputs_unchanged": outputs_before == outputs_after, "outputs": outputs_after,
    "tree_after": git("status", "--porcelain", "--untracked-files=all")}
(HERE / "smoke_reference.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
print(json.dumps({"keys": list(SMOKE_KEYS), "forwards": len(executed), "theta0_identity": {k: v["plain_equals_patched_theta0"] and v["embed_equals_resid_pre0"] for k, v in digests.items()},
                  "outputs_unchanged": record["outputs_unchanged"], "tree_clean": record["tree_after"] == ""}), flush=True)
