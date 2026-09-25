"""Item 3: prompt accounting, reconstructed independently from the installed freeze/lock (read-only)."""
import rguard  # noqa: F401  (first)
from rguard import REPO

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(REPO)
EV = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm024")
X = ROOT / "experiments/024-readout-routing-nounness"
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail != '' else ''}", flush=True)


def canon(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def h(v):
    return hashlib.sha256(canon(v).encode("utf-8")).hexdigest()


freeze = json.loads((X / "confirmation-v1.json").read_text())
lock = json.loads((X / "preregistration-lock.json").read_text())
state = json.loads((ROOT / "outputs/experiment-024/results.json").read_text())

cues = [(c["word"], int(c["token_id"]), c["class"]) for c in freeze["cues"]]
frames = list(freeze["exposed_frame_ids"])
check("freeze: 40 cues (8 per class N,B,D,C,E in that order), 40 distinct token ids", len(cues) == 40 and len({t for _, t, _ in cues}) == 40
      and [c for _, _, c in cues] == [k for k in "NBDCE" for _ in range(8)])
check("freeze: 108 distinct exposed frame ids", len(frames) == 108 == len(set(frames)))
lock_cues = [(c["word"], int(c["token_id"]), c["class"]) for c in lock["fresh"]["cues"]]
check("lock fresh cues == freeze cues (word, token id, class, order)", lock_cues == cues)
check("lock guard units: E then N in the frozen order", [(u["word"], u["token_id"]) for u in lock["guard"]["units"]["E"]] == [(w, t) for w, t, c in cues if c == "E"]
      and [(u["word"], u["token_id"]) for u in lock["guard"]["units"]["N"]] == [(w, t) for w, t, c in cues if c == "N"])

keys = [f"{f}|{w}|{t}" for f in frames for (w, t, _) in cues]
manifest = {"S2-TARGET": sorted(keys)}
md = h(manifest)
check("own manifest: 4,320 keys, all distinct", len(keys) == 4320 == len(set(keys)))
check("own manifest digest == fcc437fc… == lock.confirmation_024.manifest_sha256", md == lock["confirmation_024"]["manifest_sha256"] == "fcc437fca9730b8599333eeac95807786570f0d03bd01e49f52daec76ed4e32d", md)
check("own manifest == the freeze's stored manifest (sorted)", manifest == freeze["manifest"])

ledger = state["executed_prompt_keys"]
check("ledger: 4,320 entries, sorted, no duplicates", len(ledger) == 4320 and ledger == sorted(ledger) and len(set(ledger)) == 4320)
check("ledger set == manifest set (no missing, no extra)", set(ledger) == set(keys), (len(set(keys) - set(ledger)), len(set(ledger) - set(keys))))
plog = [json.loads(line) for line in (EV / "confirm_prompts.jsonl").read_text().splitlines()]
pkeys = [e["key"] for e in plog]
cnt = Counter(pkeys)
check("prompt log: 4,320 captures; every manifest key exactly once; no extra key", len(pkeys) == 4320 and cnt == Counter(keys) and max(cnt.values()) == 1)
# execution order predicted from ul.stage_two_022: frames by frame_id, then cues by token id
by_frame = sorted(frames)
by_token = sorted(cues, key=lambda c: c[1])
predicted = [f"{f}|{w}|{t}" for f in by_frame for (w, t, _) in by_token]
check("prompt log order == frames by frame_id, then cues by token id (stage_two_022's loop order)", pkeys == predicted)
acc = state["confirmation"]["accounting"]
check("state accounting {manifest 4320, executed 4320, ledger 4320, equal True}", acc == {"equal": True, "executed": 4320, "ledger": 4320, "manifest": 4320}, acc)
check("state stage2.n_executed == 4320", state["confirmation"]["stage2"]["n_executed"] == 4320)
rec = json.loads((EV / "confirm_record.json").read_text())
check("launcher: capture_prompt == 4320 == prompt-log lines; run_patched 0; run_interventions 0", rec["counts"]["capture_prompt"] == 4320 == len(plog)
      and rec["counts"]["run_patched"] == 0 and rec["counts"]["run_interventions"] == 0)

# --- the spent keys: own union from the committed/pinned files --------------------------------------------------------
r020_path = ROOT / "outputs/experiment-020/results.json"
r020_bytes = r020_path.read_bytes()
check("020 results file sha256 == da63b8c2… (pinned)", hashlib.sha256(r020_bytes).hexdigest() == "da63b8c29f9553a9da62bea7a442e11cef999ccddc71a7a332ac7c938abc9e00")
r020 = json.loads(r020_bytes)
rec020 = r020.pop("state_sha256")
check("020 results state digest verifies (own) == 2e5485dc…", rec020 == h(r020) == "2e5485dccb39c018eb02ee8d0a3086004399690a4dfd9c114008f057b23bf8f5")
ledger020 = set(r020["executed_prompt_keys"])
m020 = json.loads((ROOT / "experiments/020-readout-decompilation/confirmation-v1.json").read_text())["manifest"]
set020c = set(m020["S1-REF"]) | set(m020["S1-VALIDITY"]) | set(m020["S2-TARGET"])
m022 = json.loads((ROOT / "experiments/022-upstream-error-localization/confirmation-v1.json").read_text())["manifest"]
set022 = set(m022["S1-REF"]) | set(m022["S1-VALIDITY"]) | set(m022["S2-TARGET"]["Y1"]) | set(m022["S2-TARGET"]["Y2"])
m023 = json.loads((ROOT / "experiments/023-block0-completion/confirmation-v1.json").read_text())["manifest"]
set023 = set(m023["S1-REF"]) | set(m023["S1-VALIDITY"]) | set(m023["S2-TARGET"]["Y1"]) | set(m023["S2-TARGET"]["Y2"])
sizes = {"020 ledger": len(ledger020), "020 confirmation set": len(set020c), "022 manifest": len(set022), "023 manifest": len(set023)}
spent = ledger020 | set020c | set022 | set023
print("   spent-key sources:", sizes, "union", len(spent))
check("spent sources: 30,132 + 3,060 + 3,060 + 3,060, pairwise disjoint, union 39,312", sizes == {"020 ledger": 30132, "020 confirmation set": 3060, "022 manifest": 3060, "023 manifest": 3060}
      and len(spent) == 39312)
check("zero collision of the 4,320 manifest keys (and the ledger, and the prompt log) with the 39,312 spent keys",
      not (set(keys) & spent) and not (set(ledger) & spent) and not (set(pkeys) & spent))
# also no collision by (frame, token id) ignoring the label, the stronger form
spent_pairs = {(k.split("|")[0], k.split("|")[2]) for k in spent}
check("zero collision even on (frame_id, token_id) ignoring the word label", not ({(k.split("|")[0], k.split("|")[2]) for k in keys} & spent_pairs))
fresh_ids = {t for _, t, _ in cues}
spent_ids = {int(k.split("|")[2]) for k in spent}
check("no fresh cue token id appears anywhere in the 39,312 spent keys", not (fresh_ids & spent_ids), sorted(fresh_ids & spent_ids))

# --- the canonical forbidden set (frozen loaders; no model, no tokenizer) agrees ----------------------------------------
sys.path.insert(0, str(ROOT / "src"))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("run024_confirmation_review", ROOT / "experiments/024-readout-routing-nounness/run.py")
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)


def refuse(*a, **k):
    raise RuntimeError("no model or tokenizer in this review step")


runner = run.Runner(log=lambda m: None, model_loader=refuse, tokenizer_loader=refuse)
base = runner._base()
check("canonical forbidden set (Runner._base) == own union (39,312)", base.forbidden == frozenset(spent), (len(base.forbidden), len(spent)))
confirmation, confirmation_sha = runner._confirmation(base)
check("canonical manifest keys == own manifest", confirmation.manifest_keys() == frozenset(keys) and confirmation.manifest() == manifest)

# --- the noun ledger ---------------------------------------------------------------------------------------------------
nouns = state["executed_noun_keys"]
pool_keys = sorted(n.key for n in base.inputs.pool.nouns)
scorable = [n.lexical_key for n in base.inputs.pool.nouns if n.single_token]
check("noun ledger == every pool noun's key (80, record_execution over pool.nouns); 79 of them scorable",
      nouns == pool_keys and len(nouns) == 80 and len(scorable) == 79 == len(lock["noun_keys"]) and scorable == lock["noun_keys"],
      [n.key for n in base.inputs.pool.nouns if not n.single_token])
check("noun ledger holds no fresh noun of 020 (020's fresh nouns never scored)", not ({n.key for n in base.inputs.confirmation_020.nouns} & set(nouns)))

# --- own accounting digests ---------------------------------------------------------------------------------------------
digests = {"manifest (canonical {'S2-TARGET': sorted})": md, "manifest sorted list": h(sorted(keys)), "ledger sorted list": h(ledger),
           "prompt-log key sequence": h(pkeys), "prompt-log sorted": h(sorted(pkeys)), "spent-key union sorted": h(sorted(spent)),
           "noun ledger": h(nouns)}
for k, v in digests.items():
    print(f"   digest {k}: {v}")
check("own digests: ledger sorted == manifest sorted == prompt log sorted", digests["manifest sorted list"] == digests["ledger sorted list"] == digests["prompt-log sorted"])
print(f"S03 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
