"""Independent reconstruction of the spent (forbidden) key set and the 101 collision check (no neural_decompiler import).

Reads: the committed confirmation files (via git show HEAD:), and outputs/experiment-020/results.json (020's ledger,
the one outputs/ file the brief allows). Writes only its JSON beside itself.
"""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler")
HERE = Path(__file__).resolve().parent


def committed_json(path):
    b = subprocess.run(["git", "--no-optional-locks", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    assert b == (ROOT / path).read_bytes()
    return json.loads(b)


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def untag(key: str) -> str:
    return "|".join(key.split("|")[:3])


r = {}
results_path = ROOT / "outputs/experiment-020/results.json"
raw = results_path.read_bytes()
r["results_020_file_sha256"] = hashlib.sha256(raw).hexdigest()
state = json.loads(raw)
recorded = state.pop("state_sha256", None)
r["results_020_state_sha256_recorded"] = recorded
r["results_020_state_sha256_verifies"] = recorded == hashlib.sha256(canonical(state).encode("utf-8")).hexdigest()
L020 = set(state["executed_prompt_keys"])
del state, raw

c020 = committed_json("experiments/020-readout-decompilation/confirmation-v1.json")
c022 = committed_json("experiments/022-upstream-error-localization/confirmation-v1.json")
c023 = committed_json("experiments/023-block0-completion/confirmation-v1.json")
c024 = committed_json("experiments/024-readout-routing-nounness/confirmation-v1.json")
c025 = committed_json("experiments/025-nounness-direction-intervention/confirmation-v1.json")
c101 = json.loads((ROOT / "experiments/101-read-component-intervention/confirmation-v1.json").read_bytes())


def m022(c):
    m = c["manifest"]
    return set(m["S1-REF"]) | set(m["S1-VALIDITY"]) | set(m["S2-TARGET"]["Y1"]) | set(m["S2-TARGET"]["Y2"])


C020 = set(c020["manifest"]["S1-REF"]) | set(c020["manifest"]["S1-VALIDITY"]) | set(c020["manifest"]["S2-TARGET"])
C020_prompts = {p["key"] for p in c020["exposed_frame_prompts"]} | {p["key"] for p in c020["token_prompts"]}
M022, M023 = m022(c022), m022(c023)
M024 = set(c024["manifest"]["S2-TARGET"])
U025 = {untag(k) for k in c025["manifest"]["S2-TARGET"]}
F025 = L020 | C020 | M022 | M023 | M024
FORBIDDEN = F025 | U025
r["sizes"] = {"L020": len(L020), "C020": len(C020), "C020_S2_eq_prompt_lists": set(c020["manifest"]["S2-TARGET"]) == C020_prompts, "M022": len(M022),
              "M023": len(M023), "M024": len(M024), "U025": len(U025), "F025_025s_own_set": len(F025), "FORBIDDEN": len(FORBIDDEN)}
r["tagged_keys_in_forbidden"] = sum(1 for k in FORBIDDEN if k.count("|") >= 3)
s1, s2 = c101["manifest"]["S1-REF"], c101["manifest"]["S2-TARGET"]
keys101 = set(s1) | set(s2)
untagged101 = {untag(k) for k in keys101}
r["keys101"] = len(keys101)
r["untagged101"] = len(untagged101)
r["collisions_tagged"] = sorted(keys101 & FORBIDDEN)
r["collisions_untagged"] = sorted(untagged101 & FORBIDDEN)
r["collisions_with_untagged_forms_of_forbidden"] = sorted(untagged101 & {untag(k) for k in FORBIDDEN})
r["forbidden_frame_ids_with_101"] = sorted({k.split("|")[0] for k in FORBIDDEN if "-101-" in k.split("|")[0]})
r["forbidden_frame_ids_coordinated"] = len({k.split("|")[0] for k in FORBIDDEN if k.startswith("coordinated-adjective")})
# 025's full tagged manifest too (beyond the rule)
T025 = set(c025["manifest"]["S2-TARGET"])
r["collisions_with_025_tagged"] = len(keys101 & T025)
r["frame_id_prefix_101_in_any_earlier_manifest"] = any(k.startswith("coordinated-adjective-101-") for k in (L020 | C020 | M022 | M023 | M024 | T025))
json.dump({"r": r, "forbidden_sha256": hashlib.sha256(canonical(sorted(FORBIDDEN)).encode()).hexdigest()}, open(HERE / "spent_set.json", "w"), indent=1)
r["forbidden_sha256_sorted_canonical"] = hashlib.sha256(canonical(sorted(FORBIDDEN)).encode()).hexdigest()
print(json.dumps(r, indent=1))
