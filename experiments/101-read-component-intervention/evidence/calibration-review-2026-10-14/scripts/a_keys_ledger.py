"""Check 1 (launcher evidence: keys, ledger, run log) — independent: no neural_decompiler import, tokenizer only.

- the 60 candidate texts parsed from design §11 (the committed design file), × the cues one/two/three/many, tokenized
  with the pinned tokenizer.json (the `tokenizers` library) → the 240 forbidden 101 token sequences;
- the 36 exposed coordinated frames from 020's committed confirmation file (`exposed_frame_prompts` token ids) and 020's
  locked states (p_c, p_t); each frame re-tokenized from its text to confirm the prefix and suffix;
- my own reconstruction of the 2,016 calibration keys, compared with the state's `calibration_executed`, its planned
  count and digest, the launcher's run log (order, count, uniqueness) and the record's counters;
- every untagged key in 020's ledger and in no confirmation manifest (020, 022, 023, 024, 025); no calibration prompt's
  token sequence is a 101 sequence.
"""
import json
import re
import subprocess
import sys
from collections import Counter

from common import (CAL, CUES, OUTPUTS_101, REF_ID, REF_LABEL, RESULTS_020, ROOT, SNAPSHOT, WRITE_ATTEMPTS, canonical, dump, frame_runs,
                    sha256_text)

assert not [m for m in sys.modules if m.startswith("neural_decompiler")]
from tokenizers import Tokenizer  # noqa: E402

out: dict = {}
tok = Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json"))


def encode(text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


# --- the 60 design texts (committed design at HEAD) -------------------------------------------------------------
design = subprocess.run(["git", "--no-optional-locks", "show", "HEAD:docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md"],
                        cwd=ROOT, check=True, capture_output=True, text=True).stdout
section = design.split("**Candidates, in order.**", 1)[1].split("If fewer than 36", 1)[0]
cands = {}
for line in section.splitlines():
    m = re.match(r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*$", line)
    if m:
        cands[int(m.group(1))] = m.group(2)
        cands[int(m.group(3))] = m.group(4)
assert sorted(cands) == list(range(1, 61)), sorted(cands)
cue_words = {"one": 581, "two": 767, "three": 1264, "many": 1142}
forbidden_seqs = {}
for number, text in sorted(cands.items()):
    for word, cid in cue_words.items():
        ids = encode(text.replace("{cue}", word))
        assert ids.count(cid) == 1, (number, word, ids)
        forbidden_seqs[tuple(ids)] = f"101 candidate {number} with {word}"
out["design_candidates"] = {"count": len(cands), "sha256_of_texts": sha256_text(canonical([cands[i] for i in range(1, 61)])), "sequences": len(forbidden_seqs)}

# cross-check against the committed confirmation file's 36 frozen frames (prefix + cue + suffix)
conf = json.loads((ROOT / "experiments/101-read-component-intervention/confirmation-v1.json").read_text())
frozen_ok = 0
for i, fr in enumerate(conf["frames"], start=1):
    for word, cid in cue_words.items():
        seq = tuple(fr["prefix_ids"]) + (cid,) + tuple(fr["suffix_ids"])
        assert forbidden_seqs.get(seq) == f"101 candidate {i} with {word}", (i, word)
        frozen_ok += 1
out["frozen_frames_match_design_tokenization"] = frozen_ok  # 36 × 4

# --- 020's exposed coordinated frames ---------------------------------------------------------------------------
c020 = json.loads((ROOT / "experiments/020-readout-decompilation/confirmation-v1.json").read_text())
state020 = json.loads(RESULTS_020.read_text())
locked = state020["exploration"]["locked_states"]
ledger020 = frozenset(state020["executed_prompt_keys"])
coord_ids = [f for f in c020["exposed_frame_ids"] if f.startswith("coordinated-adjective")]
assert len(coord_ids) == 36
frames = {}
for fid in coord_ids:
    p_c, p_t = locked[fid]["p_c"], locked[fid]["p_t"]
    prompts = [e for e in c020["exposed_frame_prompts"] if e["frame_id"] == fid]
    shapes = {(tuple(e["token_ids"][:p_c]), tuple(e["token_ids"][p_c + 1:])) for e in prompts}
    assert len(shapes) == 1 and all(e["token_ids"][p_c] == e["token_id"] for e in prompts), fid
    prefix, suffix = shapes.pop()
    assert p_t == p_c + 1 and len(prefix) == p_c and len(suffix) == 1
    # the frame text, from one committed prompt, and an independent re-tokenization with each calibration cue
    e = prompts[0]
    words = e["text"].split(" ")
    assert e["word"] in words
    template = " ".join("{cue}" if w == e["word"] else w for w in words)
    for word, cid in cue_words.items():
        assert tuple(encode(template.replace("{cue}", word))) == prefix + (cid,) + suffix, (fid, word)
    frames[fid] = {"prefix": prefix, "suffix": suffix, "p_c": p_c, "p_t": p_t, "text": template}
out["exposed_frames"] = {"count": len(frames), "p_c_histogram": dict(Counter(f["p_c"] for f in frames.values())),
                         "retokenized_ok": True}

# --- my reconstruction of the calibration keys ------------------------------------------------------------------
labels = {cue: (label, cid) for cue, cid, label in CUES}
keys_s1, keys_runs, seqs = [], [], {}
for fid in coord_ids:
    keys_s1.append(f"{fid}|{REF_LABEL}|{REF_ID}|S1")
    seqs[f"{fid}|{REF_LABEL}|{REF_ID}"] = frames[fid]["prefix"] + (REF_ID,) + frames[fid]["suffix"]
    for tag, arm, cue, on_ref in frame_runs():
        if on_ref:
            prompt = f"{fid}|{REF_LABEL}|{REF_ID}"
        else:
            label, cid = labels[cue]
            prompt = f"{fid}|{label}|{cid}"
            seqs[prompt] = frames[fid]["prefix"] + (cid,) + frames[fid]["suffix"]
        keys_runs.append(f"{prompt}|{tag}")
my_keys = keys_s1 + keys_runs
assert len(frame_runs()) == 55
out["my_keys"] = {"count": len(my_keys), "distinct": len(set(my_keys)), "s1": len(keys_s1), "runs": len(keys_runs),
                  "planned_sha256_mine": sha256_text(canonical(sorted(my_keys)))}

state = json.loads((OUTPUTS_101 / "results.json").read_text())
cal = state["calibration"]
out["state_ledger"] = {"calibration_executed_count": len(state["calibration_executed"]),
                       "equal_to_mine_as_sets": set(state["calibration_executed"]) == set(my_keys),
                       "calibration_executed_sorted": state["calibration_executed"] == sorted(my_keys),
                       "planned": cal["planned"], "planned_sha256_equal_mine": cal["planned"]["sha256"] == out["my_keys"]["planned_sha256_mine"],
                       "candidate_keys": json.loads((OUTPUTS_101 / "candidate-calibration-v1.json").read_text())["keys"],
                       "executed_prompt_keys": state["executed_prompt_keys"], "accounting": cal["accounting"],
                       "measurements_n_executed": cal["measurements"]["n_executed"]}

# --- the launcher's run log ------------------------------------------------------------------------------------
lines = [json.loads(line) for line in (CAL / "production_runs.jsonl").read_text().splitlines()]
log_keys = [entry["key"] for entry in lines]
expected_order = keys_s1 + keys_runs  # S1 re-captures first (calibrate()'s reference loop), then frames in pool order, runs in plan order
out["run_log"] = {"lines": len(lines), "n_sequence_ok": [e["n"] for e in lines] == list(range(1, len(lines) + 1)), "distinct": len(set(log_keys)),
                  "equal_to_mine_as_sets": set(log_keys) == set(my_keys), "order_equals_my_expected_order": log_keys == expected_order,
                  "max_count": max(Counter(log_keys).values())}
for mode in ("rehearsal", "smoke", "production"):
    record = json.loads((CAL / f"{mode}_run.json").read_text())
    out[f"record_{mode}"] = {"exit_status": record["exit_status"], "error": record["error"], "events": {k: v for k, v in record["events"].items() if k != "outputs_reads"},
                             "outputs_reads": record["events"].get("outputs_reads"), "tree_after": record["tree_after"],
                             "launcher_sha256": record["environment"]["launcher_sha256"], "head": record["environment"]["head"],
                             "post": record.get("post"), "started_at": record["started_at"], "ended_at": record["ended_at"]}
out["launcher_sha256_file"] = __import__("hashlib").sha256((CAL / "launch_calibrate.py").read_bytes()).hexdigest()
out["production_exit_file"] = (CAL / "production.exit").read_text().strip()

# --- isolation: 020's ledger, the confirmation manifests, the 101 token sequences --------------------------------
untagged = sorted({k.rsplit("|", 1)[0] for k in my_keys})
c022 = json.loads((ROOT / "experiments/022-upstream-error-localization/confirmation-v1.json").read_text())
c023 = json.loads((ROOT / "experiments/023-block0-completion/confirmation-v1.json").read_text())
c024 = json.loads((ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json").read_text())
c025 = json.loads((ROOT / "experiments/025-nounness-direction-intervention/confirmation-v1.json").read_text())


def manifest_keys(payload):
    m = payload["manifest"]
    keys = set()
    for value in m.values():
        if isinstance(value, dict):
            for v in value.values():
                keys |= set(v)
        else:
            keys |= set(value)
    return keys


manifests = {"020": manifest_keys(c020), "022": manifest_keys(c022), "023": manifest_keys(c023), "024": manifest_keys(c024),
             "025_untagged": {k.rsplit("|", 1)[0] for k in c025["manifest"]["S2-TARGET"]}}
c101 = conf
k101 = set(c101["manifest"]["S1-REF"]) | set(c101["manifest"]["S2-TARGET"])
untagged101 = {k.rsplit("|", 1)[0] for k in k101}
out["isolation"] = {"untagged_prompts": len(untagged), "all_in_020_ledger": all(k in ledger020 for k in untagged),
                    "in_any_manifest": {name: len(set(untagged) & keys) for name, keys in manifests.items()},
                    "manifest_sizes": {name: len(keys) for name, keys in manifests.items()},
                    "tagged_overlap_with_101_manifest": len(set(my_keys) & k101), "untagged_overlap_with_101": len(set(untagged) & untagged101)}
calib_seqs = {seqs[p] for p in untagged}
out["token_sequences"] = {"calibration_prompt_sequences": len(calib_seqs), "forbidden_101_sequences": len(forbidden_seqs),
                          "overlap": len(calib_seqs & set(forbidden_seqs)),
                          "prefix_relations": sum(1 for a in calib_seqs for b in forbidden_seqs if a[:len(b)] == b or b[:len(a)] == a)}
# The smoke's keys (frame coordinated-adjective-1) are calibration keys run outside the phase, and in the reference process
smoke_keys = [k for k in my_keys if k.startswith("coordinated-adjective-1|")]
out["smoke_frame_keys"] = len(smoke_keys)
out["write_attempts"] = WRITE_ATTEMPTS
dump("a_keys_ledger.json", out)
# export my sequences and forbidden set for the guard of later scripts
dump("forbidden_101_sequences.json", sorted([list(s) for s in forbidden_seqs]))
dump("exposed_frames.json", {fid: {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in f.items()}} for fid, f in frames.items()})
print(json.dumps({k: v for k, v in out.items() if not k.startswith("record_")}, indent=1, default=str))
