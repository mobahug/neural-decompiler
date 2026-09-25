"""Item 9: no fresh prompt or outcome exists anywhere: outputs listing, state nulls and hashes, no outcome keys in the lock
(the preregistration is its byte-exact rendering), and no tracked file carries a fresh 024 prompt key besides the freeze."""
import guard  # noqa: F401
from guard import REPO, git, sha256_file, summary

import json
import os

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


OUT = os.path.join(REPO, "outputs/experiment-024")
EXPECTED = {"calibration-arrays.pt": "a4cd548fd9c6084d58224f072d6c529cf4e81f17650fb50ccbc1d2c97f20aece",
            "candidate-calibration.json": "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d",
            "candidate-lock.json": "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d",
            "candidate-preregistration.md": "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54",
            "results.json": "14459cdde6ec2d832c3965d5b0ebaadaf06cef6aa0b0460c3a31b29dd442b262"}
listing = sorted(os.listdir(OUT))  # includes dotfiles
print("outputs/experiment-024:", listing)
check("exactly the five pre-confirm files (no stage2-measurements.pt, no report.md, no .results-*.json temp)", listing == sorted(EXPECTED), listing)
check("no subdirectories", all(os.path.isfile(os.path.join(OUT, f)) for f in listing))
for name, digest in EXPECTED.items():
    check(f"{name} sha256 unchanged", sha256_file(os.path.join(OUT, name)) == digest)

with open(os.path.join(OUT, "results.json"), encoding="utf-8") as h:
    state = json.load(h)
check("state.confirmation is null", state["confirmation"] is None)
check("state.report is null", state["report"] is None)
check("state_sha256 == 58891c24…", state["state_sha256"] == "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d")
check("state ledgers empty", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])

OUTCOME_KEYS = {"mse", "log_mse", "rho", "K", "D_EN", "D_EN_exact", "p_value", "p_exact", "dc", "delta_c", "dx1", "dx3", "ceiling", "per_cue", "results", "label",
                "accounting", "gates", "c_recompute", "stage2", "descriptives", "measured", "SSEC", "sse", "incident"}


def keys_of(value, prefix=""):
    out = []
    if isinstance(value, dict):
        for k, v in value.items():
            out.append(f"{prefix}.{k}" if prefix else k)
            out += keys_of(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out += keys_of(v, f"{prefix}[]")
    return out


with open(os.path.join(REPO, "experiments/024-readout-routing-nounness/preregistration-lock.json"), encoding="utf-8") as h:
    lock = json.load(h)
lock_paths = set(keys_of(lock))
bad_lock = sorted(p for p in lock_paths if p.rsplit(".", 1)[-1].replace("[]", "") in OUTCOME_KEYS)
from neural_decompiler import readout_routing as rr  # noqa: E402


def at(obj, dotted):
    for part in dotted.split("."):
        obj = obj[part]
    return obj


def definitional(obj, path):
    """A flagged key is acceptable only when its value is a definition, a label vocabulary or a tolerance."""
    value = at(obj, path)
    leaf = path.rsplit(".", 1)[-1]
    if leaf == "p_value":
        return value == "K / assignments (exact)"
    if path == "constants.results":
        return value == {"primary": list(rr.PRIMARY_RESULTS), "guard": list(rr.GUARD_RESULTS), "outcomes": list(rr.OUTCOMES)}
    if path == "constants.tolerances.mse":
        return value == rr.TOLERANCES["mse"]
    if path == "calibration.checks.mse":  # the calibration's own exposed-cue MSE cross-check (canonical vs torch route)
        return set(value) == {"check", "max_difference", "at", "tolerance", "passed"} and value["check"] == "mse" and value["passed"] is True
    return False


for p in bad_lock:
    print(f"   flagged lock path {p} = {json.dumps(at(lock, p))[:120]}")
check("every lock key named like an outcome is definitional (spec text, label vocabulary, tolerance)", all(definitional(lock, p) for p in bad_lock), bad_lock)
print("   lock.fresh keys:", sorted(lock["fresh"]), "; per-cue keys:", sorted(lock["fresh"]["cues"][0]))
check("lock.fresh.cues carry only weight-derived fields", sorted(lock["fresh"]["cues"][0]) == ["above_calibration_maximum", "class", "form", "lemma", "nounness",
                                                                                              "predicted_log_mse", "token_id", "word"])
mse_like = sorted(p for p in lock_paths if "mse" in p.lower() or "rho" in p.lower())
print("   lock key paths mentioning mse/rho:", mse_like)
bad_state = sorted(p for p in set(keys_of(state)) if p.rsplit(".", 1)[-1].replace("[]", "") in OUTCOME_KEYS)
for p in bad_state:
    print(f"   flagged state path {p} = {json.dumps(at(state, p))[:160]}")
check("every state key named like an outcome is definitional or the calibration's exposed cross-check",
      all(definitional(state, p) for p in bad_state), bad_state)

# tracked files: which carry a fresh 024 prompt key (frame|word|id) or a fresh cue's "|word|id" suffix?
with open(os.path.join(REPO, "experiments/024-readout-routing-nounness/confirmation-v1.json"), encoding="utf-8") as h:
    frozen = json.load(h)
suffixes = [f"|{c['word']}|{c['token_id']}".encode() for c in frozen["cues"]]
tracked = [p for p in git("ls-files", "-z").split("\0") if p]
carriers = []
for rel in tracked:
    path = os.path.join(REPO, rel)
    if not os.path.isfile(path) or os.path.getsize(path) > 200_000_000:
        continue
    with open(path, "rb") as h:
        data = h.read()
    if any(s in data for s in suffixes):
        carriers.append(rel)
print(f"   tracked files scanned: {len(tracked)}; carrying a fresh 024 prompt-key suffix: {carriers}")
check("the only tracked file with fresh 024 prompt keys is the freeze (confirmation-v1.json)",
      carriers == ["experiments/024-readout-routing-nounness/confirmation-v1.json"], carriers)
exp_files = [p for p in tracked if p.startswith("experiments/024-readout-routing-nounness/") and "/evidence/" not in p]
print("   tracked 024 files outside evidence/:", exp_files)
check("no tracked 024 results, stage-2 or report file", not any(("stage2" in p or "report" in p.lower() or "results" in p) for p in exp_files))
with open(os.path.join(REPO, "experiments/024-readout-routing-nounness/calibration-v1.json"), encoding="utf-8") as h:
    record = json.load(h)
fresh_ids = {int(c["token_id"]) for c in frozen["cues"]}
rec_ids = {int(e["token_id"]) for e in record["calibration_cues"]} | {int(e["token_id"]) for e in record["pronoun_cues"]}
check("the calibration record holds no fresh cue's MSE (no fresh id among its 139 + 36 cues)", not (rec_ids & fresh_ids) and len(rec_ids) == 175)
exposed_words = {e["word"] for e in record["calibration_cues"]} | {e["word"] for e in record["pronoun_cues"]}
where = state["calibration"]["checks"]["mse"]["at"]
check("the state's MSE cross-check location is an exposed cue, not a fresh one", where in exposed_words and where not in {c["word"] for c in frozen["cues"]}, where)
stage2 = []
for root, _, files in os.walk(os.path.join(REPO, "outputs")):
    stage2 += [os.path.join(root, f) for f in files if "stage2" in f and "024" in root]
check("no stage-2 file for 024 anywhere under outputs/", not stage2, stage2)
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"ITEM9 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
