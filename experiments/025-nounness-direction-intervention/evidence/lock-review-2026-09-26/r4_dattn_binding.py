"""Item 10: the D_attn binding. (a) 020's locked rows4/rows5 for the 108 exposed frames: frame and reference
association, shape [8, p_t + 1], non-negative, rows summing to 1, the prompt ending at p_t; (b) the exposed-states digest
recomputed by the reviewer's own code and shown to cover rows4/rows5; (c) rd.state_from_locked / ul.y1_states hold
exactly those rows; (d) at most 6 guarded plain captures of spent 020 reference prompts (two per template family, key
frame_id|ref|<reference id>, each in 020's ledger), recording ONLY whether the captured ATTN_PATTERN.L4/L5@p_t rows,
taken through the very PromptRun.vector path confirm uses, equal the locked rows (no distance statistic, no logits
used, nothing else captured)."""
import guard  # noqa: F401

import hashlib
import json

ROOT = guard.ROOT
HERE = guard.HERE

import torch  # noqa: E402

from neural_decompiler import models  # noqa: E402
from neural_decompiler import cue_rotation as cr  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402

guard.seal()
torch.set_num_threads(4)


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


inputs = ul.load_frozen_inputs(ROOT)
locked = inputs.closure["exploration"]["locked_states"]
ledger = frozenset(inputs.closure["ledger"])
lock = json.loads((ROOT / "outputs/experiment-025/candidate-lock.json").read_text(encoding="utf-8"))
freeze = json.loads((ROOT / cr.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))
frames = {frame.frame_id: frame for frame in inputs.pool.frames}
out: dict = {"exposed_frames": len(frames), "locked_state_frames": len(locked), "frames_equal": sorted(frames) == sorted(locked) == sorted(freeze["exposed_frame_ids"])}

# (a) the locked rows
worst_sum, min_value, shape_ok, pos_ok, keys_all = 0.0, 1.0, True, True, set()
entry_keys = set()
for frame_id, frame in frames.items():
    entry = locked[frame_id]
    entry_keys |= set(entry)
    pos_ok &= entry["p_c"] == frame.p_c and entry["p_t"] == frame.p_t and len(frame.prompt_ids(inputs.pool.reference_ids[frame.template_id])) == frame.p_t + 1
    for name in ("rows4", "rows5"):
        rows = torch.tensor(entry[name], dtype=torch.float64)
        shape_ok &= tuple(rows.shape) == (8, frame.p_t + 1)
        keys_all.add(rows.shape[1])
        worst_sum = max(worst_sum, float((rows.sum(dim=1) - 1.0).abs().max()))
        min_value = min(min_value, float(rows.min()))
out["rows"] = {"shape_8_by_pt_plus_1": shape_ok, "positions_match_frames_and_prompt_ends_at_p_t": pos_ok, "key_lengths_seen": sorted(keys_all),
               "max_abs_row_sum_minus_1": worst_sum, "min_entry": min_value, "entry_fields": sorted(entry_keys)}

# (b) the exposed-states digest, my own code
per_frame = {frame_id: sha(canonical(entry)) for frame_id, entry in sorted(locked.items())}
mine = sha(canonical(per_frame))
extract = inputs.closure["extract"]["exploration"]["locked_state_digests"]
tampered = json.loads(json.dumps(locked[sorted(locked)[0]]))
tampered["rows5"][0][0] = tampered["rows5"][0][0] + 1e-9
out["digest"] = {"mine": mine, "lock": lock["dependencies"]["readout_020"]["exposed_states_sha256"], "ul": ul.exposed_states_digest(locked),
                 "per_frame_equal_committed_extract": per_frame == dict(extract), "rows_are_hashed_fields": all("rows4" in e and "rows5" in e for e in locked.values()),
                 "a_rows5_change_changes_the_frame_digest": sha(canonical(tampered)) != per_frame[sorted(locked)[0]],
                 "results_020_file_sha256_mine": hashlib.sha256((ROOT / rc.RESULTS_020_RELATIVE_PATH).read_bytes()).hexdigest(),
                 "results_020_file_pinned": rc.RESULTS_020_FILE_SHA256}

# (c) the states confirm builds
units_frames = tuple(sorted(inputs.pool.frames, key=lambda f: f.frame_id))
states = ul.y1_states(locked, units_frames)
same = all(torch.equal(states[f].rows4, torch.tensor(locked[f]["rows4"], dtype=torch.float64)) and torch.equal(states[f].rows5, torch.tensor(locked[f]["rows5"], dtype=torch.float64))
           and states[f].p_t == frames[f].p_t and states[f].frame is frames[f] for f in frames)
out["states"] = {"y1_states_rows_equal_locked": same, "frames_bound_to_states": all(states[f].frame.frame_id == f for f in frames)}

# (d) the guarded spent reference-prompt captures
model = models.load_model(models.PYTHIA_70M)
guard.seal_model_forwards(model, door=True)
untagged_025 = frozenset(cr.untagged(k) for k in freeze["manifest"]["S2-TARGET"])
budget = {"total": 0}
by_template: dict[str, list] = {}
for frame in units_frames:
    by_template.setdefault(frame.template_id, []).append(frame)
chosen = [frame for template in sorted(by_template) for frame in by_template[template][:2]]
captures = []
for frame in chosen:
    reference_id = int(inputs.pool.reference_ids[frame.template_id])
    prompt = pm.Prompt(frame, reference_id, "ref")
    sites = [("ATTN_PATTERN.L4", frame.p_t), ("ATTN_PATTERN.L5", frame.p_t)]
    run = guard.allow_spent_reference_capture(model, prompt, sites, ledger=ledger, forbidden_prefixes=untagged_025, budget=budget)
    state = states[frame.frame_id]
    record = {"key": prompt.key, "template": frame.template_id, "in_020_ledger": prompt.key in ledger, "p_t": frame.p_t}
    for name, reference in (("ATTN_PATTERN.L4", state.rows4), ("ATTN_PATTERN.L5", state.rows5)):
        vector = run.vector((name, frame.p_t))  # the path cr.measure_rotated takes: PromptRun.vector -> [heads, keys]
        as_confirm = vector.detach().to(torch.float32).double()[:, : int(reference.shape[1])]  # cr.stage_two / cr.routing_distance slicing
        record[name] = {"captured_shape": list(vector.shape), "reference_shape": list(reference.shape), "bitwise_equal_locked": bool(torch.equal(as_confirm, reference)),
                        "allclose_1e-7": bool(torch.allclose(as_confirm, reference, rtol=0.0, atol=1e-7)), "captured_dtype": str(vector.dtype)}
    del run
    captures.append(record)
# A key that is not a spent reference prompt (here a spent, non-reference 020 key) is refused before it runs.
frame_id, word, token_id = cr.PATCH_PATH_SPENT_KEYS[0].split("|")
probe = pm.Prompt(frames[frame_id], int(token_id), word)
try:
    guard.allow_spent_reference_capture(model, probe, [("ATTN_PATTERN.L4", probe.frame.p_t), ("ATTN_PATTERN.L5", probe.frame.p_t)], ledger=ledger,
                                        forbidden_prefixes=untagged_025, budget=budget)
    refused_probe = False
except RuntimeError:
    refused_probe = True
out["captures"] = captures
out["captures_count"] = budget
out["non_spent_025_key_refused_before_running"] = refused_probe
del model
out["guard"] = guard.summary()
(HERE / "r4_dattn_binding.out.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
