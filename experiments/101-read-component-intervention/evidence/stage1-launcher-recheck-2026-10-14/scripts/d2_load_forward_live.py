"""The audit events Python 3.12 raises for the parts of stage 1 that no rehearsal reaches: ``seed_runtime``, the model
load (``models.load_model`` → TransformerBridge), ``ul.ModelPrograms.from_model``, ``rr.parameters_digest`` and ONE
forward through ``pm.capture_prompt`` with stage 1's site set (``ri.capture_sites(..., plain=True)``), on a NEUTRAL
sentence that is not a 101 frame ("The quick brown fox jumps over the lazy dog."). No 101 prompt runs; nothing of the
runner runs. The working directory is an empty scratch directory, so any relative-path event stays relative and can be
judged afterwards as production would (from the repository root). My guard refuses any write outside scratch.
Each recorded write-type or relative-path event is then decided by the launcher v2's ``decide`` under production rules
with ROOT = the repository and the working directory taken as the repository root (pure decision; no chdir).
Output: out/d_load_forward_events.json.
"""
import collections
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAUNCHER_V2, LAUNCHER_V2_SHA256, RECHECK, REPO, WRITE_EVENTS, extract_launcher, install_guard, write_json  # noqa: E402

assert hashlib.sha256(LAUNCHER_V2.read_bytes()).hexdigest() == LAUNCHER_V2_SHA256
work = RECHECK / "work_load"
work.mkdir(exist_ok=True)
assert not any(work.iterdir()), "the working directory must start empty"
os.chdir(work)

recorded: list = []
counts = collections.Counter()
phase = {"name": "imports"}
marks: list = []


def counter(event, args):
    counts[(phase["name"], event)] += 1


install_guard(recorded)
sys.addaudithook(counter)
P_LIVE = extract_launcher(REPO, "production", fake_cwd=str(REPO))
JUDGED = {"open", "os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir", "os.chmod", "os.rename", "os.link", "os.symlink"}
live: list = []
jdepth = {"n": 0}


def judge(event, args):
    if jdepth["n"] or event not in JUDGED:
        return
    jdepth["n"] += 1
    try:
        try:
            decision = P_LIVE["decide"](event, args, True)
            outcome = "allow" if decision is None else f"REFUSE {decision}"
        except Exception as error:  # an exception in the real hook fails the operation
            outcome = f"REFUSE (exception {type(error).__name__}: {error})"
        writing = event != "open" or (len(args) > 2 and isinstance(args[2], int) and bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))) \
            or (len(args) > 1 and isinstance(args[1], str) and any(ch in args[1] for ch in "wax+"))
        if outcome != "allow" or writing:
            live.append({"phase": phase["name"], "event": event, "args": [os.fsdecode(a) if isinstance(a, (str, bytes, os.PathLike)) else a for a in args][:4],
                         "production_decision_at_event_time": outcome})
    finally:
        jdepth["n"] -= 1


sys.addaudithook(judge)


def mark(name):
    phase["name"] = name
    marks.append((name, len(recorded)))


import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

mark("seed_runtime")
models.seed_runtime(rd.RUNTIME_SEED, models.PYTHIA_70M.deterministic_algorithms)
mark("load_model")
model = models.load_model(models.PYTHIA_70M)
mark("frozen_inputs (read-only, as _base does)")
inputs = ul.load_frozen_inputs(REPO)
mark("ModelPrograms.from_model + parameters_digest")
progs = ul.ModelPrograms.from_model(model, inputs)
digest = rr.parameters_digest(model)
mark("tokenize the neutral sentence")
text = "The quick brown fox jumps over the lazy dog."
ids = tuple(int(i) for i in model.tokenizer(text)["input_ids"])


class NeutralFrame:  # duck type: ri.capture_sites reads p_c and p_t only
    p_c, p_t = 3, 4


class NeutralPrompt:  # duck type: pm.capture_prompt reads token_ids and p_t only
    token_ids, p_t = ids, 4


sites = ri.capture_sites(NeutralFrame(), int(model.cfg.n_layers), int(model.cfg.n_heads), plain=True)
mark("one forward: pm.capture_prompt (neutral sentence, stage 1's plain site set)")
run = pm.capture_prompt(model, NeutralPrompt(), sites)
mark("second forward (steady state)")
run2 = pm.capture_prompt(model, NeutralPrompt(), sites)
mark("gc")
import gc  # noqa: E402

del model
gc.collect()
mark("end")

# --- judge every recorded write-type or relative-path event as production would ---
P = extract_launcher(REPO, "production", fake_cwd=str(REPO))
bounds = marks + [("<end>", len(recorded))]
by_phase = {}
for (name, start), (_, stop) in zip(bounds, bounds[1:]):
    by_phase[name] = recorded[start:stop]
by_phase = {"imports": recorded[: marks[0][1]], **by_phase}

judged = []
for name, rows in by_phase.items():
    for row in rows:
        event, args = row["event"], row["args"]
        path_args = args[:1] if event not in ("os.rename", "os.link", "os.symlink") else (args[:2] if event != "os.symlink" else args[1:2])
        relative = event in (WRITE_EVENTS | {"open"}) and any(isinstance(a, str) and a and not a.startswith("/") for a in path_args)
        if not (row["writing"] or event in WRITE_EVENTS or relative):
            continue
        # rebuild the argument tuple with the recorded values (paths as str, ints as int)
        try:
            decision = P["decide"](event, tuple(args), True)
            outcome = "allow" if decision is None else f"REFUSE {decision}"
        except Exception as error:
            outcome = f"REFUSE (exception {type(error).__name__}: {error})"
        judged.append({"phase": name, "event": event, "args": args, "cwd": row["cwd"], "writing": row["writing"], "relative": relative,
                       "production_decision": outcome})

refusals = [j for j in judged if j["production_decision"] != "allow"]
repo_touch = [r for r in recorded if any(isinstance(p, str) and p.startswith(str(REPO) + "/") and not p.startswith(str(REPO / ".venv") + "/") for p in (r["paths"] or []))]
repo_reads_outside_src = sorted({p for r in repo_touch for p in r["paths"] if isinstance(p, str) and not p.startswith(str(REPO / "src") + "/")})
outputs_reads = sorted({p for r in repo_touch for p in r["paths"] if isinstance(p, str) and p.startswith(str(REPO / "outputs") + "/")})
live_refusals = [row for row in live if row["production_decision_at_event_time"] != "allow"]
out = {"tmpdir": os.environ.get("TMPDIR"), "live_judged_writes": live, "live_refusals": live_refusals, "launcher_sha256": LAUNCHER_V2_SHA256, "python": sys.version, "cwd": str(work), "text": text, "n_tokens": len(ids), "n_sites": len(sites),
       "parameters_digest": digest, "work_dir_after": sorted(p.name for p in work.iterdir()),
       "event_counts": {f"{k[0]} | {k[1]}": v for k, v in sorted(counts.items())},
       "write_or_relative_events": judged, "refused_by_v2_production": refusals,
       "repo_paths_touched_outside_src": repo_reads_outside_src, "outputs_paths_touched": outputs_reads,
       "repo_writes": [r for r in repo_touch if r["writing"]]}
write_json(os.environ.get("D_OUT", "d_load_forward_events.json"), out)
print("event names by phase (watched + unwatched):")
for key, value in sorted(counts.items()):
    if key[1] in ("open", "os.listdir", "os.scandir", "fcntl.fcntl", "object.__getattr__", "object.__setattr__", "builtins.id", "exec", "compile", "import",
                  "marshal.loads", "sys._getframe", "code.__new__", "function.__new__", "os.stat", "sys._getframemodulename", "ctypes.dlsym", "ctypes.dlopen",
                  "cpython._PySys_ClearAuditHooks", "sys.excepthook", "socket.gethostname", "time.sleep"):
        continue
    print(f"   {key[0]:<70} {key[1]:<28} {value}")
print(f"write-type or relative events: {len(judged)}; refused by v2 production rules: {len(refusals)}")
for row in judged:
    print("   ", row["phase"][:40], row["event"], [a for a in row["args"][:3]], "->", row["production_decision"][:140])
print("repo paths touched outside src/:", repo_reads_outside_src[:20])
print("outputs paths touched:", outputs_reads)
print("repo writes:", out["repo_writes"])
print("work dir after:", out["work_dir_after"])
print(f"LIVE (at event time): {len(live)} write-type or refused events judged; refusals: {len(live_refusals)}")
for row in live:
    print("   LIVE", row["phase"][:40], row["event"], row["args"][:3], "->", row["production_decision_at_event_time"][:120])
