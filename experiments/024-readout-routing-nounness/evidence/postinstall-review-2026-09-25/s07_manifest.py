"""Item 7: the 4,320 keys reconstructed from the committed freeze; unique; digest == lock manifest_sha256; zero overlap with
the 39,312 spent keys (runner._base().forbidden and own count by source); the state's ledgers are empty; wider scan of
every experiment ledger on this machine for the fresh ids."""
import guard  # noqa: F401
from guard import REPO, Seal, canonical, sha256_bytes, summary

import glob
import importlib.util
import json
import os
import sys

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


def load(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as h:
        return json.load(h)


frozen = load("experiments/024-readout-routing-nounness/confirmation-v1.json")
lock = load("experiments/024-readout-routing-nounness/preregistration-lock.json")
cues = [(c["word"], int(c["token_id"])) for c in frozen["cues"]]
frames = list(frozen["exposed_frame_ids"])
keys = [f"{frame}|{word}|{tid}" for frame in frames for word, tid in cues]
check("40 cues x 108 exposed frames", len(cues) == 40 and len(frames) == 108 and len(set(frames)) == 108)
check("4,320 keys, all unique", len(keys) == 4320 and len(set(keys)) == 4320, len(set(keys)))
check("own sorted keys == the committed manifest's S2-TARGET (stored order)", sorted(keys) == frozen["manifest"]["S2-TARGET"] and list(frozen["manifest"]) == ["S2-TARGET"])
own_digest = sha256_bytes(canonical({"S2-TARGET": sorted(keys)}).encode("utf-8"))
check("own sha256(canonical {'S2-TARGET': keys}) == lock.confirmation_024.manifest_sha256 (fcc437fc…)", own_digest == lock["confirmation_024"]["manifest_sha256"], own_digest)
check("lock.confirmation_024.manifest_size == 4320", lock["confirmation_024"]["manifest_size"] == 4320)
fresh_ids = {tid for _, tid in cues}
check("40 distinct fresh token ids", len(fresh_ids) == 40)

# own spent-key count by source
ledger_020 = set(load("outputs/experiment-020/results.json")["executed_prompt_keys"])
m020 = load("experiments/020-readout-decompilation/confirmation-v1.json")["manifest"]
conf_020 = set(m020["S1-REF"]) | set(m020["S1-VALIDITY"]) | set(m020["S2-TARGET"])
m022 = load("experiments/022-upstream-error-localization/confirmation-v1.json")["manifest"]
man_022 = set(m022["S1-REF"]) | set(m022["S1-VALIDITY"]) | set(m022["S2-TARGET"]["Y1"]) | set(m022["S2-TARGET"]["Y2"])
m023 = load("experiments/023-block0-completion/confirmation-v1.json")["manifest"]
man_023 = set(m023["S1-REF"]) | set(m023["S1-VALIDITY"]) | set(m023["S2-TARGET"]["Y1"]) | set(m023["S2-TARGET"]["Y2"])
sources = {"020 ledger": ledger_020, "020 confirmation set": conf_020, "022 manifest": man_022, "023 manifest": man_023}
for name, s in sources.items():
    print(f"   {name}: {len(s)} keys; overlap with 024's 4,320: {len(s & set(keys))}")
own_union = set().union(*sources.values())
check("own count by source: 30,132 + 3,060 + 3,060 + 3,060 = 39,312, pairwise disjoint",
      [len(s) for s in sources.values()] == [30132, 3060, 3060, 3060] and len(own_union) == 39312)
check("zero overlap between the 4,320 keys and the own spent union", not (own_union & set(keys)))
spent_ids = {int(k.rsplit("|", 1)[1]) for k in own_union}
check("no fresh token id appears in any spent key", not (spent_ids & fresh_ids), sorted(spent_ids & fresh_ids)[:5])

RUN_PY = os.path.join(REPO, "experiments/024-readout-routing-nounness/run.py")
spec = importlib.util.spec_from_file_location("run024_postinstall_review", RUN_PY)
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)
Seal().refuse_captures()


def refuse(spec):
    raise RuntimeError("no loaders in item 7")


runner = run.Runner(log=print, model_loader=refuse, tokenizer_loader=refuse)
base = runner._base()
confirmation, _ = runner._confirmation(base)
check("runner._base().forbidden has 39,312 keys and equals the own union", len(base.forbidden) == 39312 and set(base.forbidden) == own_union)
check("canonical confirmation.manifest_keys() == own keys", set(confirmation.manifest_keys()) == set(keys))
check("forbidden & manifest == empty (canonical sets)", not (base.forbidden & confirmation.manifest_keys()))
state = run.rd.load_results_state(runner.results_path)
check("state.executed_prompt_keys == []", state["executed_prompt_keys"] == [])
check("state.executed_noun_keys == []", state["executed_noun_keys"] == [])
run.rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.manifest_keys(), "Experiment 024's ledger before confirm")
check("rr.assert_ledger_isolated(empty ledger, forbidden | manifest) passes", True)

# wider: every experiment results state on this machine, any key carrying a fresh id or equal to a fresh key
hits = {}
for path in sorted(glob.glob(os.path.join(REPO, "outputs/experiment-*/results.json"))):
    with open(path, encoding="utf-8") as h:
        payload = json.load(h)
    ledger = payload.get("executed_prompt_keys") if isinstance(payload, dict) else None
    if not ledger:
        continue
    bad = [k for k in ledger if k in set(keys) or (k.rsplit("|", 1)[-1].isdigit() and int(k.rsplit("|", 1)[-1]) in fresh_ids)]
    hits[os.path.relpath(path, REPO)] = (len(ledger), len(bad))
for p, (n, b) in hits.items():
    print(f"   {p}: {n} executed keys, {b} with a fresh 024 id")
check("no experiment ledger on this machine holds a key with a fresh 024 token id", all(b == 0 for _, b in hits.values()), len(hits))
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"ITEM7 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
