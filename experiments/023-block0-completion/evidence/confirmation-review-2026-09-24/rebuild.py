"""Items 3 and 5 (+ an independent I7): weights only. The Y2 table rebuilt from the stage-1 reference states recorded in
the state and the weights, serialized by my own code and compared byte for byte; its index rebuilt; the Y1 table rebuilt
likewise against the committed bytes; I1/I3/I4 recomputed on all 3,024 target pairs from the saved measurements; the
ceiling C recomputed from the saved Δx3. The model is loaded only to extract its weights; immediately after the load
torch.nn.Module.__call__ and plural_mechanism's capture/intervention entry points raise. No prompt, no forward pass."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirmreview023")
import guard  # noqa: E402  FIRST: audit hook + torch.load refusal, proven live

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(guard.ROOT)
E = ROOT / "experiments/023-block0-completion"
OUT = ROOT / "outputs/experiment-023"
FAIL = []
T0 = time.time()


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail != "" else ""), flush=True)
    if not ok:
        FAIL.append(name)


def cj(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(b):
    return hashlib.sha256(b).hexdigest()


from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler.models import PYTHIA_70M, load_model, seed_runtime  # noqa: E402

state = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
lock = json.loads((E / "preregistration-lock.json").read_text(encoding="utf-8"))
conf = json.loads((E / "confirmation-v1.json").read_text(encoding="utf-8"))
stage1 = state["confirmation"]["stage1"]
inputs = ul.load_frozen_inputs(ROOT)
print(f"frozen inputs loaded ({time.time() - T0:.0f} s); torch threads {torch.get_num_threads()}", flush=True)
check("runtime threads == the recorded runtime (4)", torch.get_num_threads() == state["phases"]["extract"]["runtime"]["torch_num_threads"] == 4)

# ---------------------------------------------------------------- weights only
calls = {"n": 0}
_orig_call = torch.nn.Module.__call__


def _counting_call(self, *args, **kwargs):
    calls["n"] += 1
    return _orig_call(self, *args, **kwargs)


torch.nn.Module.__call__ = _counting_call
seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
model = load_model(PYTHIA_70M)
torch.nn.Module.__call__ = _orig_call
print(guard.forbid_forward(), flush=True)  # immediately after the load
check("the model load itself ran no module call (counted during the load)", calls["n"] == 0, f"{calls['n']} calls")
check("no class in the model's MRO overrides __call__ (the Module.__call__ guard covers model(...))", [c.__name__ for c in type(model).__mro__ if "__call__" in c.__dict__] == ["Module"])
progs = ul.ModelPrograms.from_model(model, inputs)
del model
gc.collect()
print(f"weights extracted, model deleted ({time.time() - T0:.0f} s)", flush=True)

# ---------------------------------------------------------------- units, states and my order
noun_keys = [progs.nouns.nouns[i].lexical_key for i in progs.scorable]
check("scorable nouns == the lock's 79 noun keys", noun_keys == lock["noun_keys"] and len(noun_keys) == 79)
ref_ids = {k: int(v) for k, v in conf["reference_cue_ids"].items()}
new_frames = {e["frame_id"]: pm.Frame(e["template_id"], e["frame_id"], tuple(e["prefix_ids"]), tuple(e["suffix_ids"]), e["cue_ids"], e["text_template"], origin="extension")
              for e in conf["frames"]}
check("new frames rebuilt from confirmation-v1.json reproduce its p_c/p_t", all((new_frames[e["frame_id"]].p_c, new_frames[e["frame_id"]].p_t) == (e["p_c"], e["p_t"]) for e in conf["frames"]))
exposed_frames = {f.frame_id: f for f in inputs.pool.frames}
locked = inputs.closure["exploration"]["locked_states"]
check("exposed states digest == the lock's exposed_states_sha256", ul.exposed_states_digest(locked) == lock["exposed_states_sha256"])
check("stage-1 states: rd.state_digest == recorded state_digests (18)", all(rd.state_digest(stage1["states"][f]) == stage1["state_digests"][f] for f in new_frames) and len(stage1["states"]) == 18)
states = {"Y1": {fid: rd.state_from_locked(locked[fid], frame) for fid, frame in exposed_frames.items()},
          "Y2": {fid: rd.state_from_locked(stage1["states"][fid], frame) for fid, frame in new_frames.items()}}
check("rebuilt stage-1 states carry the frames' p_c/p_t", all((s.p_c, s.p_t) == (new_frames[f].p_c, new_frames[f].p_t) for f, s in states["Y2"].items()))
frames = {"Y1": exposed_frames, "Y2": new_frames}
tokens = sorted([(int(c["token_id"]), c["word"]) for c in conf["cues"]])
GROUPS = ("cue_final", "coordinated")
P1_MASK = {"cue_final": 14, "coordinated": 30}
FULL = {"cue_final": 15, "coordinated": 31}
check("ul's full mask canonicalizes to 15 (cue-final) / 31 (coordinated); P1 masks = emb|Bv|Bp (+T)",
      ul.canonical_mask(ul.FULL_MASK, True) == 15 and ul.canonical_mask(ul.FULL_MASK, False) == 31 and ul.BIT["emb"] | ul.BIT["Bv"] | ul.BIT["Bp"] == 14 and 14 | ul.BIT["T"] == 30)


def my_pairs(pop):
    ordered = sorted(frames[pop])
    return {g: [(t, w, fid) for t, w in tokens for fid in ordered if (frames[pop][fid].template_id == "coordinated-adjective") == (g == "coordinated")] for g in GROUPS}


pairs = {pop: my_pairs(pop) for pop in ("Y1", "Y2")}
check("pair counts: Y1 1728/864, Y2 288/144", [len(pairs[p][g]) for p in ("Y1", "Y2") for g in GROUPS] == [1728, 864, 288, 144])

# ---------------------------------------------------------------- the loop: predictions (P0, P1), I5, identities, ceiling
meas = torch.load(OUT / "stage2-measurements.pt")
gates = {name: {"max": 0.0, "at": ""} for name in ("I1", "I3", "I3_ul", "I4", "I5", "C_recomputed")}
blocks = {pop: {} for pop in ("Y1", "Y2")}
cf_slot_equal = True


def worse(name, value, where):
    if math.isnan(value) or value > gates[name]["max"]:
        gates[name].update({"max": value, "at": where})


def my_i3(pred, measured):  # max_p ‖Δx̂3(p) − Δx3(p)‖₂ / max(‖Δx3(p)‖₂, 1e-12), my numpy code
    out = 0.0
    for p in sorted(measured):
        a, b = pred[p].double().numpy(), measured[p].double().numpy()
        out = max(out, float(np.sqrt(((a - b) ** 2).sum())) / max(float(np.sqrt((b * b).sum())), 1e-12))
    return out


for pop in ("Y1", "Y2"):
    for g in GROUPS:
        out = np.empty((len(pairs[pop][g]), 2, 79), dtype=np.float64)
        rows16 = {}
        dx1_all, dx3_all, ceil_all = (meas[f"{pop}/{g}/{name}"] for name in ("dx1", "dx3", "ceiling"))
        for row, (token_id, word, fid) in enumerate(pairs[pop][g]):
            frame, st = frames[pop][fid], states[pop][fid]
            if fid not in rows16:
                rows16[fid] = ul.reference_rows_017(progs.programs, st)
            ctx = ul.pair_context(progs, frame, st, ref_ids[frame.template_id], token_id, word, rows16[fid])
            empty = ul.compose_dx3(progs, ctx, 0)
            out[row, 0] = ul.contrast_of(progs, st, empty).numpy()
            out[row, 1] = ul.contrast_of(progs, st, ul.compose_dx3(progs, ctx, P1_MASK[g])).numpy()
            where = f"{pop}|{word}|{fid}"
            committed = rd.predicted_dx3(progs.chain, progs.weights, st, ctx.rows16, token_id, frame.template_id)
            worse("I5", math.inf if set(committed) != set(empty) else max(float((empty[p] - committed[p]).abs().max()) for p in committed), where)
            # measured changes at the changed positions
            if g == "cue_final":
                cf_slot_equal &= bool(torch.equal(dx1_all[row, 0], dx1_all[row, 1]) and torch.equal(dx3_all[row, 0], dx3_all[row, 1]))
                dx1, dx3 = {frame.p_c: dx1_all[row, 0]}, {frame.p_c: dx3_all[row, 0]}
            else:
                dx1, dx3 = {frame.p_c: dx1_all[row, 0], frame.p_t: dx1_all[row, 1]}, {frame.p_c: dx3_all[row, 0], frame.p_t: dx3_all[row, 1]}
            f = ctx.factors
            i1 = float((f.d_emb + f.delta_e + f.block0_pc.total - dx1[frame.p_c]).abs().max())
            if frame.p_t != frame.p_c:
                i1 = max(i1, float((f.block0_pt.total - dx1[frame.p_t]).abs().max()))
            worse("I1", i1, where)
            full = ul.compose_dx3(progs, ctx, FULL[g])
            worse("I3", my_i3(full, dx3), where)
            worse("I3_ul", ul.i3_error(full, dx3), where)
            worse("I4", float((ul.contrast_of(progs, st, full) - ceil_all[row]).abs().max()), where)
            worse("C_recomputed", float((ul.contrast_of(progs, st, dx3) - ceil_all[row]).abs().max()), where)
        blocks[pop][g] = out
        print(f"  {pop}/{g}: {len(pairs[pop][g])} pairs ({time.time() - T0:.0f} s)", flush=True)

rec = state["confirmation"]["gates"]
print("my gates:", json.dumps(gates), flush=True)
print("state gates:", json.dumps(rec), flush=True)
check("cue-final rows: both Δx1/Δx3 slots hold the same p_c vector", cf_slot_equal)
check("I1 ≤ 1e-4 on all 3,024 targets and == the recorded maximum (value and location)", gates["I1"]["max"] <= 1e-4 and gates["I1"]["max"] == rec["I1"]["max"] and gates["I1"]["at"] == rec["I1"]["at"],
      f"{gates['I1']['max']:.6e} at {gates['I1']['at']}")
check("I3 (my relative-error code) ≤ 1e-4 and == ul.i3_error's maximum ≈ the recorded one", gates["I3"]["max"] <= 1e-4 and abs(gates["I3"]["max"] - gates["I3_ul"]["max"]) <= 1e-15 and abs(gates["I3_ul"]["max"] - rec["I3"]["max"]) == 0.0
      and gates["I3"]["at"] == rec["I3"]["at"], f"mine {gates['I3']['max']:.6e}, ul {gates['I3_ul']['max']:.6e} at {gates['I3']['at']}")
check("I4 ≤ 1e-3 and == the recorded maximum", gates["I4"]["max"] <= 1e-3 and gates["I4"]["max"] == rec["I4"]["max"] and gates["I4"]["at"] == rec["I4"]["at"], f"{gates['I4']['max']:.6e} at {gates['I4']['at']}")
check("I5: P0's Δx̂3 == the committed Level 0 exactly (all 3,024)", gates["I5"]["max"] == 0.0)
check("C: the saved ceiling == the frozen readout fed the saved measured Δx3, bit for bit (all 3,024)", gates["C_recomputed"]["max"] == 0.0, f"max |Δ| {gates['C_recomputed']['max']}")

# ---------------------------------------------------------------- Y2 table: my serialization, byte for byte; the index
def my_bytes(block_list):
    parts = []
    for name, arr in block_list:
        flat = arr.reshape(-1).tolist()
        parts.append(struct.pack(f"<{len(flat)}d", *flat))
    return b"".join(parts)


y2_mine = my_bytes([("cue_final", blocks["Y2"]["cue_final"]), ("coordinated", blocks["Y2"]["coordinated"])])
y2_disk = (OUT / "y2-table.f64").read_bytes()
check("Y2 table rebuilt from the stage-1 states and the weights == y2-table.f64 byte for byte (4623adde…)", y2_mine == y2_disk and sha(y2_mine) == "4623adde5eb53e14b0368e5b4d8b91bd6f2513bd522ba074bc55ab747970d2c2",
      f"{len(y2_mine)} bytes, sha {sha(y2_mine)[:16]}")
check("my struct serialization == numpy '<f8' C-order bytes (two independent serializers)", y2_mine == b"".join(np.ascontiguousarray(blocks["Y2"][g], dtype="<f8").tobytes() for g in GROUPS))


def my_index(kind, pop, data):
    meta = {"experiment": "023", "kind": kind, "schema_version": 1, "construction": lock["y2_table"]["meta"]["construction"], "design": lock["y2_table"]["meta"]["design"],
            "pair_order": "cue token id, then frame_id", "pairs": {g: [[w, t, fid] for t, w, fid in pairs[pop][g]] for g in GROUPS}, "columns": ["P0", "P1"],
            "masks": {"P0": 0, "P1": {"cue_final": 14, "coordinated": 30}}, "nouns": noun_keys}
    entries, offset = [], 0
    for g in GROUPS:
        shape = list(blocks[pop][g].shape)
        nbytes = int(np.prod(shape)) * 8
        entries.append({"name": g, "shape": shape, "offset_bytes": offset, "nbytes": nbytes})
        offset += nbytes
    return {"format": "raw IEEE-754 float64, little-endian, C order; blocks back to back in the order listed", "dtype": "<f8", "blocks": entries, "total_bytes": offset,
            "file_sha256": sha(data), **meta}, meta


y2_index_mine, y2_meta_mine = my_index("Y2", "Y2", y2_mine)
y2_index_text = (OUT / "y2-table.json").read_text(encoding="utf-8")
check("Y2 index rebuilt (my pairs, masks, nouns, layout, digest) == y2-table.json byte for byte (e87091ef…)", cj(y2_index_mine) + "\n" == y2_index_text and sha(y2_index_text.encode("utf-8")).startswith("e87091ef"),
      sha(y2_index_text.encode('utf-8'))[:16])
check("the lock's Y2 specification == my layout and meta (construction, orders, masks, nouns; paths)", lock["y2_table"]["layout"] == [{"name": e["name"], "shape": e["shape"]} for e in y2_index_mine["blocks"]]
      and lock["y2_table"]["meta"] == y2_meta_mine and (lock["y2_table"]["data_path"], lock["y2_table"]["index_path"]) == ("outputs/experiment-023/y2-table.f64", "outputs/experiment-023/y2-table.json")
      and lock["y2_table"]["format"] == y2_index_mine["format"] and lock["y2_table"]["dtype"] == "<f8")
from neural_decompiler import block0_completion as b0c  # noqa: E402  (a constant only: the construction text the lock binds)
check("the construction text the lock binds is b0c.TABLE_CONSTRUCTION (P0 mask 0; P1 mask 14/30)", lock["y2_table"]["meta"]["construction"] == b0c.TABLE_CONSTRUCTION)

# ---------------------------------------------------------------- Y1 table (I7, independently): my serialization against the committed bytes
y1_mine = my_bytes([("cue_final", blocks["Y1"]["cue_final"]), ("coordinated", blocks["Y1"]["coordinated"])])
y1_disk = (E / "locked-y1-table.f64").read_bytes()
check("Y1 table rebuilt from 020's locked states and the weights == committed locked-y1-table.f64 byte for byte (37603f05…)", y1_mine == y1_disk and sha(y1_mine) == lock["y1_table"]["file_sha256"],
      f"{len(y1_mine)} bytes")
y1_index_mine, _ = my_index("Y1", "Y1", y1_mine)
check("Y1 index rebuilt == committed locked-y1-table.json byte for byte (3bf85e18…)", cj(y1_index_mine) + "\n" == (E / "locked-y1-table.json").read_text(encoding="utf-8"))
print(f"\nREBUILD ({time.time() - T0:.0f} s):", "ALL PASS" if not FAIL else f"FAILED {FAIL}")
