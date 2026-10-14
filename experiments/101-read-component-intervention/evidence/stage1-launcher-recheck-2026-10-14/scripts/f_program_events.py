"""The audit events of stage 1's pure computations, judged live by the launcher v2's ``decide`` under production rules
(ROOT = the repository, the working directory taken as the repository root): ``ri.read_geometry``, ``ri.geometry_block``
+ ``ri.enforce_g1``, ``ri.prediction_table`` and ``ri.states_blocks`` + ``ul.table_bytes`` + ``rd.state_digest``. As the
lock phase did, they run on the 36 EXPOSED (spent) coordinated frames and 020's locked states: no forward pass, no 101
frame, no 101 prompt. The exposed table's digest is compared with the calibration's (``ce055e09…``). TMPDIR and the
working directory are in scratch; my guard refuses any write outside scratch. Output: out/f_program_events.json.
"""
import collections
import gc
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAUNCHER_V2, LAUNCHER_V2_SHA256, RECHECK, REPO, extract_launcher, install_guard, write_json  # noqa: E402

assert hashlib.sha256(LAUNCHER_V2.read_bytes()).hexdigest() == LAUNCHER_V2_SHA256
work = RECHECK / "work_program"
work.mkdir(exist_ok=True)
assert not any(work.iterdir())
os.chdir(work)
recorded: list = []
install_guard(recorded)
P_LIVE = extract_launcher(REPO, "production", fake_cwd=str(REPO))
JUDGED = {"open", "os.remove", "os.rmdir", "shutil.rmtree", "os.truncate", "os.mkdir", "os.chmod", "os.rename", "os.link", "os.symlink"}
phase = {"name": "imports"}
live, counts, jdepth = [], collections.Counter(), {"n": 0}


def judge(event, args):
    counts[(phase["name"], event)] += 1
    if jdepth["n"] or event not in JUDGED:
        return
    jdepth["n"] += 1
    try:
        try:
            decision = P_LIVE["decide"](event, args, True)
            outcome = "allow" if decision is None else f"REFUSE {decision}"
        except Exception as error:
            outcome = f"REFUSE (exception {type(error).__name__}: {error})"
        writing = event != "open" or (len(args) > 2 and isinstance(args[2], int) and bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))) \
            or (len(args) > 1 and isinstance(args[1], str) and any(ch in args[1] for ch in "wax+"))
        if outcome != "allow" or writing:
            live.append({"phase": phase["name"], "event": event, "args": [os.fsdecode(a) if isinstance(a, (str, bytes, os.PathLike)) else a for a in args][:4],
                         "production_decision_at_event_time": outcome})
    finally:
        jdepth["n"] -= 1


sys.addaudithook(judge)

from neural_decompiler import models  # noqa: E402
from neural_decompiler import read_intervention as ri  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

phase["name"] = "seed_runtime + load_model + inputs + programs"
models.seed_runtime(rd.RUNTIME_SEED, models.PYTHIA_70M.deterministic_algorithms)
model = models.load_model(models.PYTHIA_70M)
inputs = ul.load_frozen_inputs(REPO)
progs = ul.ModelPrograms.from_model(model, inputs)
n_heads = int(model.cfg.n_heads)
del model
gc.collect()
config = ri.PRODUCTION
exposed = [frame for frame in inputs.pool.frames if frame.template_id == ri.TEMPLATE]
noun_keys = [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]
phase["name"] = "read_geometry + geometry_block (exposed ids) + enforce_g1"
geometry = ri.read_geometry(inputs.lock_011)
block = ri.geometry_block(geometry, progs.head, progs.axis_T, [frame.frame_id for frame in exposed], config.k_iso)
ri.enforce_g1(block)
phase["name"] = "prediction_table (exposed, spent states)"
locked = inputs.closure["exploration"]["locked_states"]
states = {frame.frame_id: rd.state_from_locked(locked[frame.frame_id], frame) for frame in exposed}
table = ri.prediction_table(progs, exposed, states, geometry, config, noun_keys, fractions_for_weak=True)
phase["name"] = "states_blocks + table_bytes + state_digest"
entries = {frame.frame_id: locked[frame.frame_id] for frame in exposed}
blocks = ri.states_blocks(entries, exposed)
data = ul.table_bytes(blocks)
digests = {frame_id: rd.state_digest(entry) for frame_id, entry in sorted(entries.items())}
phase["name"] = "end"

refusals = [row for row in live if row["production_decision_at_event_time"] != "allow"]
out = {"n_exposed_frames": len(exposed), "exposed_table_content_sha256": table["content_sha256"],
       "equals_calibration_digest_prefix_ce055e09": table["content_sha256"].startswith("ce055e09"), "states_bytes": len(data), "n_blocks": len(blocks),
       "live_judged_writes": live, "live_refusals": refusals, "work_dir_after": sorted(p.name for p in work.iterdir()),
       "repo_writes": [r for r in recorded if r["writing"] and any(isinstance(p, str) and p.startswith(str(REPO) + "/") for p in (r["paths"] or []))],
       "event_counts": {f"{k[0]} | {k[1]}": v for k, v in sorted(counts.items()) if k[1] not in ("open", "object.__getattr__", "builtins.id", "exec", "compile", "import",
                                                                                                   "marshal.loads", "sys._getframe", "os.listdir", "os.scandir",
                                                                                                   "fcntl.fcntl", "code.__new__", "function.__new__",
                                                                                                   "sys._getframemodulename", "object.__setattr__", "object.__delattr__",
                                                                                                   "array.__new__", "ctypes.dlsym", "ctypes.dlopen")}}
write_json("f_program_events.json", out)
print("exposed table", table["content_sha256"], "equals calibration (ce055e09…):", out["equals_calibration_digest_prefix_ce055e09"])
print("states bytes", len(data), "blocks", len(blocks))
print(f"LIVE: {len(live)} write-type or refused events; refusals {len(refusals)}")
for row in live:
    print("   LIVE", row["phase"][:50], row["event"], row["args"][:3], "->", row["production_decision_at_event_time"][:100])
print("event counts (excluding reads/imports):")
for k, v in out["event_counts"].items():
    print("  ", k, v)
print("work dir after:", out["work_dir_after"], "| repo writes:", out["repo_writes"])
