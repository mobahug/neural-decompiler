"""Post-freeze verification of Experiment 024 (read-only; writes only to this scratchpad directory).

Part 1, tokenizer and committed files only: the frozen file re-read and verified by 024's own loader; then
independent checks of the 40 cues (classes, lemma pairs, single tokens, exclusions, distinct ids), the counts, the
manifest (4,320 unique keys, canonical order, zero collision with every spent set) and the bound configuration.

Part 2, weights only (every forward refused after the load; module calls during the load counted): the nounness
metadata the lock will bind — the embedding and parameter digests, the 158 and 139 ids, the centroid digests, the 40
scores in the frozen cue order, per-class minima and maxima, and the extrapolation count. Computed after the freeze
file exists, from its cues; it selects nothing.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler").resolve()
HERE = Path(__file__).resolve().parent
REFUSED = []


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
    if not writing:
        return
    try:
        path = Path(os.fsdecode(args[0])).resolve()
    except OSError:
        return
    if str(path).startswith(str(ROOT)) and "/.venv/" not in str(path):
        REFUSED.append(str(path))
        raise PermissionError(f"the verification may not write into the repository: {path}")


sys.addaudithook(_hook)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler.models import PYTHIA_70M, load_model  # noqa: E402

out = {}
lines = []


def say(text=""):
    print(text, flush=True)
    lines.append(text)


# ---------------------------------------------------------------- Part 1: tokenizer and committed files
path = ROOT / rr.CONFIRMATION_RELATIVE_PATH
payload = json.loads(path.read_text(encoding="utf-8"))
inputs = ul.load_frozen_inputs(ROOT)
c022_path = ROOT / ul.CONFIRMATION_RELATIVE_PATH
c022 = json.loads(c022_path.read_text())
c023_path = ROOT / b0c.CONFIRMATION_RELATIVE_PATH
c023 = json.loads(c023_path.read_text())
confirmation_023 = b0c.load_confirmation_023(c023_path, inputs, c022, rc.file_sha256(c022_path))
base = b0c.exclusion(inputs, c022, rc.file_sha256(c022_path))
excluded = rr.exclusion(base, c023, rc.file_sha256(c023_path))
confirmation = rr.load_confirmation_024(path, inputs.pool, excluded, rr.PRODUCTION)  # 024's own loader: digest, configuration, manifest, picks, rules' inputs
file_sha, content_sha = rc.file_sha256(path), payload["content_sha256"]
assert content_sha == rc.content_digest(payload) and confirmation.content_sha256 == content_sha
say(f"# Experiment 024 post-freeze verification")
say()
say(f"- file {rr.CONFIRMATION_RELATIVE_PATH}: sha256 `{file_sha}`, content sha256 `{content_sha}`; verified by rr.load_confirmation_024")

tokenizer = AutoTokenizer.from_pretrained(PYTHIA_70M.model_id, revision=PYTHIA_70M.revision, local_files_only=True)
tokens = confirmation.tokens
assert [t["class"] for t in tokens] == [cls for cls in rr.CLASSES for _ in range(8)]
by_class = {cls: [(t["word"], int(t["token_id"])) for t in tokens if t["class"] == cls] for cls in rr.CLASSES}
expected = {"N": list(rr.EXPECTED_PICKS["N"]), "D": list(rr.EXPECTED_PICKS["measure"]), "E": list(rr.EXPECTED_PICKS["ordinary"]),
            "B": [dict(rr.MEASURE_LEMMAS)[w] for w in rr.EXPECTED_PICKS["measure"]], "C": [dict(rr.ORDINARY_LEMMAS)[w] for w in rr.EXPECTED_PICKS["ordinary"]]}
assert {cls: [w for w, _ in words] for cls, words in by_class.items()} == expected, "the picks are not the design's expected 40"
for cls in rr.CLASSES:
    say(f"- {cls}: " + ", ".join(f"{w} ({t})" for w, t in by_class[cls]))
# lemma pairing: B/D and C/E share their lemma, position by position, and each plural is the list's plural of that lemma
pairs_ok = True
for plural_cls, singular_cls, lemmas in (("B", "D", rr.MEASURE_LEMMAS), ("C", "E", rr.ORDINARY_LEMMAS)):
    plural_of = dict(lemmas)
    for p, s in zip([t for t in tokens if t["class"] == plural_cls], [t for t in tokens if t["class"] == singular_cls]):
        ok = p["lemma"] == s["lemma"] == s["word"] and p["word"] == plural_of[s["word"]] and p["form"] == "plural" and s["form"] == "singular"
        pairs_ok &= ok
assert pairs_ok
say(f"- lemma pairs: B/D and C/E share their lemma position by position; every plural is its lemma's listed plural: {pairs_ok}")
# single tokens with the leading space, decoded back
single = all(list(pm._encode(tokenizer, " " + t["word"])) == [int(t["token_id"])] and tokenizer.decode([int(t["token_id"])]) == " " + t["word"] for t in tokens)
assert single
ids = [int(t["token_id"]) for t in tokens]
assert len(set(ids)) == 40
say(f"- every cue is one token with its leading space and decodes back: {single}; 40 distinct ids: {len(set(ids)) == 40}")
# exclusions, recomputed independently of the file
forms = set(rr.target_noun_form_ids(inputs.pool))
frame_tokens = set(rr.frame_token_ids(inputs.pool))
earlier = set(excluded["cue_token_ids"])
earlier_direct = set(c023["exclusion"]["cue_token_ids"]) | {int(c["token_id"]) for c in c023["cues"]} | {int(v) for f in c023["frames"] for v in f["cue_ids"].values()}
assert earlier == earlier_direct and len(earlier) == 351 and len(set(c023["exclusion"]["cue_token_ids"])) == 327
assert len(forms) == 161 and len(frame_tokens) == 316
overlaps = {"earlier_cue_ids": len(set(ids) & earlier), "target_noun_forms": len(set(ids) & forms), "exposed_frame_tokens": len(set(ids) & frame_tokens)}
assert overlaps == {"earlier_cue_ids": 0, "target_noun_forms": 0, "exposed_frame_tokens": 0}
say(f"- exclusions: {len(earlier)} earlier cue ids (023's 327 + 023's own cues and frame cue ids), {len(forms)} target-noun form ids, "
    f"{len(frame_tokens)} exposed-frame token ids; overlap with the 40 ids {overlaps}")
assert payload["exclusion"]["cue_token_ids"] == sorted(earlier) and payload["target_noun_form_ids"]["ids"] == sorted(forms) and payload["frame_token_ids"]["ids"] == sorted(frame_tokens)
reasons = {}
for entry in payload["rejected"]:
    key = entry["list"]
    reasons.setdefault(key, []).append(entry["candidate"])
say(f"- rejected: {len(payload['rejected'])} ({', '.join(f'{k} {len(v)}' for k, v in reasons.items())}); reserves: " +
    ", ".join(f"{k} {len(v)}" for k, v in payload["reserves"].items()))
for key, names in reasons.items():
    say(f"  - {key}: " + "; ".join(f"{e['candidate']} ({e['reason']})" for e in payload["rejected"] if e["list"] == key))
say(f"- reserves (recorded, never used): " + " | ".join(f"{k}: {', '.join(v)}" for k, v in payload["reserves"].items()))
assert payload["picks_match_expected"] is True and payload["configuration"] == rr.PRODUCTION.to_json()
# the manifest
manifest = payload["manifest"]["S2-TARGET"]
frames = {frame.frame_id for frame in inputs.pool.frames}
parsed = [key.split("|") for key in manifest]
spent = (frozenset(inputs.closure["ledger"]) | frozenset(prompt.key for prompt in inputs.confirmation_020.all_prompts)
         | frozenset(c022["manifest"]["S1-REF"]) | frozenset(c022["manifest"]["S1-VALIDITY"]) | frozenset(c022["manifest"]["S2-TARGET"]["Y1"])
         | frozenset(c022["manifest"]["S2-TARGET"]["Y2"]) | confirmation_023.manifest_keys())
checks = {"count": len(manifest), "unique": len(set(manifest)), "canonical_sorted": manifest == sorted(manifest), "equals_the_cues_times_frames": set(manifest) == confirmation.manifest_keys(),
          "every_key_a_fresh_cue": all(int(t) in set(ids) for _, _, t in parsed), "every_frame_exposed": {f for f, _, _ in parsed} == frames and len(frames) == 108,
          "collision_with_spent_sets": len(set(manifest) & spent), "spent_keys_checked": len(spent),
          "no_024_ledger": not (ROOT / "outputs/experiment-024").exists()}
assert checks["count"] == checks["unique"] == 4_320 and checks["canonical_sorted"] and checks["equals_the_cues_times_frames"] and checks["every_key_a_fresh_cue"]
assert checks["every_frame_exposed"] and checks["collision_with_spent_sets"] == 0 and checks["no_024_ledger"]
manifest_sha = pm.sha256_text(pm.canonical_json(payload["manifest"]))
say(f"- manifest: {checks}; manifest sha256 `{manifest_sha}`")
guard = payload["configuration"]["guard"]
say(f"- bound configuration: {payload['configuration']['name']}; guard n_E {guard['n_E']}, n_N {guard['n_N']}, assignments {guard['assignments']}, max_upper "
    f"{guard['max_upper']}; B {payload['configuration']['draws']}, P {payload['configuration']['null_permutations']}, contrasts "
    f"{payload['configuration']['contrast_resamples']}")
out["part1"] = {"file_sha256": file_sha, "content_sha256": content_sha, "cues": by_class, "overlaps": overlaps, "manifest": checks, "manifest_sha256": manifest_sha,
                "rejected": len(payload["rejected"]), "reserves": {k: len(v) for k, v in payload["reserves"].items()}}

# ---------------------------------------------------------------- Part 2: the nounness metadata, weights only
calls = {"during_load": 0}
original_call = torch.nn.Module.__call__


def counting(self, *args, **kwargs):
    calls["during_load"] += 1
    return original_call(self, *args, **kwargs)


torch.nn.Module.__call__ = counting
model = load_model(PYTHIA_70M)


def refuse(*args, **kwargs):
    raise RuntimeError("a forward pass was attempted during the verification")


torch.nn.Module.__call__ = refuse
for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
    if hasattr(pm, name):
        setattr(pm, name, refuse)
parameters_sha = rr.parameters_digest(model)
W_E = pm.Weights.from_model(model).W_E
del model
units = b0c.exposed_units(inputs)
bindings = rr.score_bindings(W_E, inputs.pool, units)
scores = rr.full_scores(W_E, bindings, ids)
loo = rr.calibration_scores(W_E, bindings)
maximum = max(loo)
by_class_scores = {cls: [(t["word"], s) for t, s in zip(tokens, scores) if t["class"] == cls] for cls in rr.CLASSES}
e_above = [w for w, s in by_class_scores["E"] if s > maximum]
say()
say(f"- weights: {calls['during_load']} module calls during the load, every forward refused afterwards; parameters sha256 `{parameters_sha}`; "
    f"embedding sha256 `{bindings['embedding_sha256']}`")
say(f"- μ_noun: {len(bindings['noun_row_ids'])} ids (sha256 `{bindings['noun_row_ids_sha256']}`), centroid `{bindings['mu_noun_sha256']}`; μ_cue: "
    f"{len(bindings['calibration_cue_ids'])} ids (sha256 `{bindings['calibration_cue_ids_sha256']}`), centroid `{bindings['mu_cue_sha256']}`")
say(f"- 40 scores (full μ_cue, the frozen cue order) sha256 `{rc.tensor_digest(torch.tensor(scores, dtype=torch.float64))}`")
for cls in rr.CLASSES:
    values = [s for _, s in by_class_scores[cls]]
    say(f"  - {cls}: min {min(values):+.6f}, max {max(values):+.6f}, mean {sum(values) / 8:+.6f}: " + ", ".join(f"{w} {s:+.6f}" for w, s in by_class_scores[cls]))
say(f"- calibration maximum (leave-one-out, 139 cues) {maximum:+.6f}; E above it: {len(e_above)} of 8 ({', '.join(e_above)}); every E above every N: "
    f"{min(s for _, s in by_class_scores['E']) > max(s for _, s in by_class_scores['N'])}")
assert len(e_above) == 5
out["part2"] = {"parameters_sha256": parameters_sha, "embedding_sha256": bindings["embedding_sha256"], "noun_row_ids_sha256": bindings["noun_row_ids_sha256"],
                "calibration_cue_ids_sha256": bindings["calibration_cue_ids_sha256"], "mu_noun_sha256": bindings["mu_noun_sha256"], "mu_cue_sha256": bindings["mu_cue_sha256"],
                "scores": {cls: by_class_scores[cls] for cls in rr.CLASSES}, "scores_sha256": rc.tensor_digest(torch.tensor(scores, dtype=torch.float64)),
                "calibration_maximum": maximum, "E_above": e_above, "module_calls_during_load": calls["during_load"]}
(HERE / "verify_freeze.json").write_text(json.dumps(out, indent=1) + "\n")
(HERE / "verify_freeze.md").write_text("\n".join(lines) + "\n")
print("repository writes refused:", REFUSED)
