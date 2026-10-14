"""Checks 1 and 2 for the parts no rehearsal reaches, judged LIVE (at event time) by the confirm launcher's ``decide``
under production rules (ROOT = the real repository, the working directory taken as the repository root; pure
decisions, nothing of the repository is touched: my guard refuses every write outside this review's directory).

The launcher's own guard machinery (``_deny``, ``_guarded_call``, ``LOW_LEVEL``, ``_low_level``, ``_ledger_on_disk``,
``_vector_digest``, ``_sanctioned_run_arm``, ``_refuse_capture_reference``, ``WRAPPERS``, ``ORIGINALS``,
``_sweep_aliases``, ``_seal_model``, ``_counted_load``) is extracted from the reviewed source with ``ast`` and executed
with MODE = "production" and OUTPUTS = a scratch mirror (so ``_ledger_on_disk`` reads a mirror state, never the real one).

Steps: ``seed_runtime``; the counted, sealed model load; ``ul.load_frozen_inputs`` and ``ModelPrograms.from_model``;
then, on ONE NEUTRAL sentence that is no 101 prompt ("The quick brown fox jumps over the lazy dog."), with a
token-sequence guard on the innermost entry points that refuses every other token sequence:
  (a) a sanctioned ``ri.run_arm`` BEFORE the mirror ledger exists (must be refused, 0 forwards);
  (b) the mirror ledger written (keys ⊇ allowed, barrier passed, confirm running), then three sanctioned runs through
      ``ri.run_arm`` exactly as ``measure_frame`` calls it: a plain REF run, a residual REPLACE at RESID_PRE.L3@p_c
      (NOOP-like, the plain run's own residual) and a head REPLACE at L03.H04@p_t;
  (c) unsanctioned routes (a model call, model.forward, pm.capture_prompt, ri.capture_reference, a second run, a key
      outside the allowed set, a second model load), each refused with 0 forwards;
  (d) confirm's post-run computations OUTSIDE any sanctioned call, with every guard active: ``ri.frame_values``,
      ``ri.primary_statistics``, ``ri.run_records``, ``ri.secondary_statistics`` on SYNTHETIC tensors for SYNTHETIC
      frames (no 101 and no exposed data), ``rc.tensor_digest``, ``rc.json_safe``, ``ri.encode_nonfinite``; and
      ``rr.save_durably``/``torch.load`` of the synthetic tensors in the mirror.
Output: out/d_live_guards_load_forward.json.
"""
import ast
import collections
import functools
import gc
import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAUNCHER, REPO, REVIEW, check_launcher, extract_launcher, install_guard, make_live_judge, write_json  # noqa: E402

check_launcher()
work = REVIEW / "work_d"
work.mkdir(exist_ok=True)
assert not any(work.iterdir()), "the working directory must start empty"
os.chdir(work)
recorded: list = []
counts = collections.Counter()
phase = {"name": "imports"}


def counter(event, args):
    counts[(phase["name"], event)] += 1


install_guard(recorded)
sys.addaudithook(counter)
P = extract_launcher(REPO, "production", fake_cwd=str(REPO))
live: list = []
make_live_judge(P, True, live, phase)

import torch  # noqa: E402

from neural_decompiler import capture, interventions, models  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# ---- the token-sequence guard (innermost): only the neutral sentence may reach a forward
TEXT = "The quick brown fox jumps over the lazy dog."
phase["name"] = "tokenizer"
tokenizer = AutoTokenizer.from_pretrained(models.PYTHIA_70M.model_id, revision=models.PYTHIA_70M.revision)
NEUTRAL = tuple(int(i) for i in tokenizer(TEXT)["input_ids"])
token_guard = {"checked": 0, "refused": 0}


def _tokens_of(value):
    if isinstance(value, torch.Tensor):
        return tuple(int(i) for i in value.reshape(-1).tolist())
    return None


def token_guarded(name, original, kind):
    @functools.wraps(original)
    def guarded(*args, **kwargs):
        token_guard["checked"] += 1
        if kind == "prompt":
            seq = tuple(int(i) for i in args[1].token_ids)
        else:
            seq = _tokens_of(args[1])
        if seq != NEUTRAL:
            token_guard["refused"] += 1
            raise RuntimeError(f"review token guard: {name} with a non-neutral token sequence refused")
        return original(*args, **kwargs)

    return guarded


raw = {"capture.run_capture": capture.run_capture, "interventions.run_interventions": interventions.run_interventions,
       "pm.capture_prompt": pm.capture_prompt, "pm.run_patched": pm.run_patched}
assert pm.run_capture is capture.run_capture and pm.run_interventions is interventions.run_interventions
g_capture = token_guarded("run_capture", capture.run_capture, "tokens")
g_interv = token_guarded("run_interventions", interventions.run_interventions, "tokens")
capture.run_capture = pm.run_capture = g_capture
interventions.run_interventions = pm.run_interventions = g_interv
pm.capture_prompt = token_guarded("pm.capture_prompt", raw["pm.capture_prompt"], "prompt")
pm.run_patched = token_guarded("pm.run_patched", raw["pm.run_patched"], "prompt")
for module_name, module in list(sys.modules.items()):  # any other alias of the raw entry points also gets the token guard
    if module is None or not module_name.startswith("neural_decompiler"):
        continue
    for attribute, value in list(vars(module).items()):
        if value is raw["capture.run_capture"]:
            setattr(module, attribute, g_capture)
        elif value is raw["interventions.run_interventions"]:
            setattr(module, attribute, g_interv)

# ---- the launcher's guard machinery, extracted from the reviewed source
tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
start = next(n.lineno for n in tree.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "inside")
stop = next(n.lineno for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "aliases" for t in n.targets))
nodes = [n for n in tree.body if start <= n.lineno <= stop and not (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "run_log" for t in n.targets))]
mirror = REVIEW / "mirror_d"
(mirror / "outputs/experiment-101").mkdir(parents=True, exist_ok=True)
for stale in (mirror / "outputs/experiment-101").iterdir():
    stale.unlink()
run_log_path = REVIEW / "out/d_production_runs_mirror.jsonl"
G = {"torch": torch, "functools": functools, "inspect": inspect, "json": json, "hashlib": hashlib, "sys": sys, "capture": capture, "interventions": interventions,
     "pm": pm, "rd": rd, "ri": ri, "cr": cr, "models": models, "MODE": "production", "OUTPUTS": mirror / "outputs/experiment-101",
     "events": P["events"], "run_log": open(run_log_path, "w", buffering=1, encoding="utf-8")}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch_confirm_guards", "exec"), G)
guards = {"aliases_at_import": list(G["aliases"]), "module_call_guarded": torch.nn.Module.__call__ is G["_guarded_call"]}
events = G["events"]

# ---- seed, counted load, programs
phase["name"] = "seed_runtime"
models.seed_runtime(rd.RUNTIME_SEED, models.PYTHIA_70M.deterministic_algorithms)
phase["name"] = "load_model (counted, sealed)"
t0 = time.time()
model = models.load_model(models.PYTHIA_70M)
load_seconds = time.time() - t0
phase["name"] = "frozen inputs + ModelPrograms + parameters digest"
inputs = ul.load_frozen_inputs(REPO)
progs = ul.ModelPrograms.from_model(model, inputs)
parameters_sha256 = rr.parameters_digest(model)
geometry = ri.read_geometry(inputs.lock_011)
n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)


class NeutralPrompt:
    """Duck type: what ri.run_arm, cr.tagged_key, pm.capture_prompt and pm.run_patched read of a prompt."""

    def __init__(self, label):
        self.cue_label = label
        self.key = f"neutral-review|{label}|0"
        self.token_ids = NEUTRAL
        self.p_c, self.p_t = 3, len(NEUTRAL) - 1


class NeutralFrame:
    p_c, p_t = 3, len(NEUTRAL) - 1


ref_prompt, cue_prompt = NeutralPrompt("ref"), NeutralPrompt("two")
keys = {"REF": cr.tagged_key(ref_prompt, "REF"), "NOOP@two": cr.tagged_key(cue_prompt, "NOOP@two"), "HEAD-NEC@two": cr.tagged_key(cue_prompt, "HEAD-NEC@two")}
G["allowed_keys"].update(keys.values())
plain_sites = ri.capture_sites(NeutralFrame(), n_layers, n_heads, plain=True)
edit_sites = ri.capture_sites(NeutralFrame(), n_layers, n_heads, plain=False)
results: dict = {}


def attempt(label, fn):
    before_calls, before_refused = events["sanctioned_forward_calls"], len(events["refused"])
    try:
        value = fn()
        results[label] = {"refused": False, "forward_calls": events["sanctioned_forward_calls"] - before_calls}
        return value
    except RuntimeError as error:
        results[label] = {"refused": "refused by the confirm launcher" in str(error), "message": str(error)[:160],
                          "forward_calls": events["sanctioned_forward_calls"] - before_calls, "refusals_logged": len(events["refused"]) - before_refused}
        return None


phase["name"] = "(a) run_arm before the mirror ledger"
pre_ledger = {"experiment": "101", "phases": {"confirm": {"status": "not_started"}}, "executed_prompt_keys": ["s1-only|ref|581|S1"], "confirmation": None}
rr.write_state_atomic(mirror / "outputs/experiment-101/results.json", pre_ledger)  # the state as confirm finds it: S1 keys only
attempt("(a) REF before the ledger is on disk", lambda: ri.run_arm(model, ref_prompt, "REF", capture_sites=plain_sites))
barrier_only = {"experiment": "101", "phases": {"confirm": {"status": "running"}}, "executed_prompt_keys": ["s1-only|ref|581|S1"],
                "confirmation": {"barrier": {"passed": True}}}
rr.write_state_atomic(mirror / "outputs/experiment-101/results.json", barrier_only)  # a barrier without the keys must not satisfy the check
attempt("(a2) REF with the barrier passed but the keys absent", lambda: ri.run_arm(model, ref_prompt, "REF", capture_sites=plain_sites))
keys_only = {"experiment": "101", "phases": {"confirm": {"status": "not_started"}}, "executed_prompt_keys": sorted(G["allowed_keys"]), "confirmation": None}
rr.write_state_atomic(mirror / "outputs/experiment-101/results.json", keys_only)  # keys without the barrier must not satisfy it either
attempt("(a3) REF with the keys but no barrier", lambda: ri.run_arm(model, ref_prompt, "REF", capture_sites=plain_sites))
G["done_keys"].discard(keys["REF"])  # (a) was refused before admission; nothing ran
phase["name"] = "(b) mirror ledger"
state = {"experiment": "101", "phases": {"confirm": {"status": "running"}}, "executed_prompt_keys": sorted(G["allowed_keys"]),
         "confirmation": {"barrier": {"passed": True}}}
rr.write_state_atomic(mirror / "outputs/experiment-101/results.json", state)
phase["name"] = "(b) sanctioned runs"
plain = attempt("(b1) REF plain run", lambda: ri.run_arm(model, ref_prompt, "REF", capture_sites=plain_sites))
x_ref = plain.vector((ri.RESID_L3, 3))
noop = attempt("(b2) residual REPLACE at RESID_PRE.L3@p_c", lambda: ri.run_arm(model, cue_prompt, "NOOP@two", replacements={(ri.RESID_L3, 3): x_ref.detach().reshape(1, 1, -1).to(torch.float32)},
                                                                                  capture_sites=edit_sites))
captured = plain.slice((ri.HEAD_SITE_KEY, NeutralFrame.p_t))
head = attempt("(b3) head REPLACE at L03.H04@p_t", lambda: ri.run_arm(model, cue_prompt, "HEAD-NEC@two", replacements={(ri.HEAD_SITE_KEY, NeutralFrame.p_t): captured.clone()},
                                                                       capture_sites=edit_sites))
landing = {"noop_logits_equal_plain_bitwise": bool(torch.equal(noop.logits, plain.logits)) if noop is not None else None,
           "head_noop_logits_equal_plain_bitwise": bool(torch.equal(head.logits, plain.logits)) if head is not None else None}
phase["name"] = "(c) refused routes"
attempt("(c) model call", lambda: model(torch.tensor([list(NEUTRAL)])))
attempt("(c) model.forward", lambda: model.forward(torch.tensor([list(NEUTRAL)])))
attempt("(c) pm.capture_prompt outside", lambda: pm.capture_prompt(model, ref_prompt, plain_sites))
attempt("(c) pm.run_patched outside", lambda: pm.run_patched(model, cue_prompt, {(ri.RESID_L3, 3): x_ref.reshape(1, 1, -1)}, {(ri.RESID_L3, 3): pm.ReplacementSource.DIRECT}))
attempt("(c) ri.capture_reference", lambda: ri.capture_reference(model, progs, ref_prompt, "S1"))
attempt("(c) ri.run_arm a second time", lambda: ri.run_arm(model, ref_prompt, "REF", capture_sites=plain_sites))
attempt("(c) ri.run_arm outside the allowed set", lambda: ri.run_arm(model, NeutralPrompt("ref"), "SUF-READ@two", capture_sites=edit_sites))
attempt("(c) rd.capture_frame_020 directly", lambda: rd.capture_frame_020(model, progs.head, ref_prompt, progs.nouns, progs.axis_T))
attempt("(c) a second model load", lambda: models.load_model(models.PYTHIA_70M))

# ---- (d) confirm's post-run computations outside any sanctioned call, all guards active, on synthetic data only
phase["name"] = "(d) post-run computations (synthetic)"
calls_before_d = events["sanctioned_forward_calls"]
refused_before_d = len(events["refused"])
config = ri.PRODUCTION
d_model = int(progs.weights.W_E.shape[1])
n_scorable = len(progs.scorable)
generator = torch.Generator().manual_seed(7)
synthetic_frames, schedules, tensors = [], [], {}
for i in range(config.n_frames):
    frame = pm.Frame(ri.TEMPLATE, f"synthetic-review-{i:02d}", (10, 11, 12, 13, 14), (15,), {"sg": 581, "pl": 767}, "synthetic")
    reference = pm.Prompt(frame, 581, "ref")
    cues = {label: pm.Prompt(frame, cue_id, label) for label, cue_id in config.cues}
    schedule = ri.frame_schedule(frame, config, reference, cues)
    rows = len(schedule)
    contrasts = torch.randn(rows, n_scorable, generator=generator, dtype=torch.float64)
    for row, run in enumerate(schedule):
        if run.arm.name == "REF":
            contrasts[row] = 0.0
        elif run.arm.name == "PL":
            contrasts[row] = -5.0 + 0.1 * contrasts[row]
    tensors[f"{frame.frame_id}/contrasts"] = contrasts
    tensors[f"{frame.frame_id}/x3_pc"] = torch.randn(rows, d_model, generator=generator)
    tensors[f"{frame.frame_id}/T_pt"] = torch.randn(rows, d_model, generator=generator)
    synthetic_frames.append(frame)
    schedules.append(schedule)
frame_ids = [frame.frame_id for frame in synthetic_frames]
post: dict = {}
try:
    values = [ri.frame_values(schedule, tensors, config, fractions_for_weak=False) for schedule in schedules]
    values_by_cue = {label: [entry[label] for entry in values] for label in config.cue_labels}
    predicted = [{"weak": False, "fractions": {"F_rem": 0.0, "F_ins": 1.0, "N": 1.0, "S": 1.0}} for _ in frame_ids]
    primary = ri.primary_statistics(frame_ids, values_by_cue["two"], predicted, 0.025, 0.035, config)
    post["primary_statistics"] = {"label": primary["outcome"]["label"], "R": primary["R"]["count"], "I": primary["I"]["count"]}
    records = ri.run_records(progs, schedules, tensors, geometry, config)
    post["run_records_frames"] = len(records)
    secondary = ri.secondary_statistics(frame_ids, values_by_cue, primary, records, config)
    post["secondary_statistics_keys"] = sorted(secondary)
    post["encode_nonfinite"] = bool(ri.encode_nonfinite({label: list(entries) for label, entries in values_by_cue.items()}))
    post["tensor_digest"] = rc.tensor_digest(tensors[f"{frame_ids[0]}/contrasts"])
    post["json_safe"] = bool(rc.json_safe({"x": 1.0}))
    rr.save_durably(tensors, mirror / "outputs/experiment-101/stage2-measurements.pt")
    loaded = torch.load(mirror / "outputs/experiment-101/stage2-measurements.pt")
    post["save_load_digests_equal"] = {k: rc.tensor_digest(v) for k, v in loaded.items()} == {k: rc.tensor_digest(v) for k, v in tensors.items()}
    post["ok"] = True
except Exception as error:  # a guard refusal here would be an incident in production
    post["ok"] = False
    post["error"] = f"{type(error).__name__}: {error}"
post["forward_calls_during"] = events["sanctioned_forward_calls"] - calls_before_d
post["refusals_during"] = len(events["refused"]) - refused_before_d

phase["name"] = "end"
G["run_log"].close()
del model
gc.collect()
live_refusals = [row for row in live if row["decision_at_event_time"] != "allow"]
repo_writes = [r for r in recorded if r["writing"] and any(isinstance(p, str) and p.startswith(str(REPO) + "/") for p in (r["paths"] or []))]
outputs_paths = sorted({p for r in recorded for p in (r["paths"] or []) if isinstance(p, str) and p.startswith(str(REPO / "outputs") + "/")})
out = {"launcher": str(LAUNCHER), "tmpdir": os.environ.get("TMPDIR"), "text": TEXT, "neutral_ids": NEUTRAL, "token_guard": token_guard, "guards": guards,
       "sealing": {k: (v if k != "classes" else len(v)) for k, v in G["sealing"].items()}, "load_seconds": load_seconds, "parameters_sha256": parameters_sha256,
       "attempts": results, "landing": landing, "post_run_computations": post,
       "events": {k: (v if k != "outputs_reads" else len(v)) for k, v in events.items()}, "done_keys": sorted(G["done_keys"]),
       "live_judged": live, "live_refusals": live_refusals, "repo_writes": repo_writes, "repo_outputs_paths_touched": outputs_paths,
       "event_counts": {f"{k[0]} | {k[1]}": v for k, v in sorted(counts.items()) if k[1] in ("open", "os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.link",
                                                                                           "shutil.rmtree", "tempfile.mkstemp", "tempfile.mkdtemp", "os.utime",
                                                                                           "os.chmod", "subprocess.Popen", "pickle.find_class", "torch.load")}}
write_json("d_live_guards_load_forward.json", out)
print("token guard:", token_guard, "| sealing:", out["sealing"], "| load s:", round(load_seconds, 2))
for label, value in results.items():
    print(f"  {label:<48} {value}")
print("landing:", landing)
print("post-run computations:", {k: v for k, v in post.items() if k != "secondary_statistics_keys"})
print("events:", out["events"])
print(f"live-judged events: {len(live)}; refusals at event time: {len(live_refusals)}")
for row in live:
    print("   ", row["phase"][:42], row["event"], str(row["args"][:3])[:150], "->", row["decision_at_event_time"][:100])
print("repo writes:", repo_writes)
print("repo outputs paths touched:", outputs_paths)
