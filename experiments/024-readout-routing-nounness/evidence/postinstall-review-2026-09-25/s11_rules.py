"""Supplement (items 2/8/10): the bound decision rules as pure functions on the lock's values and integers only (no data):
the outcome table over all 12 pairs, the primary classification at the threshold edge (the envelope-only band is
empty), and the E–N decision at K = 321 / 322."""
import guard  # noqa: F401
from guard import REPO, summary

import json
import math
import os
from fractions import Fraction

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


from neural_decompiler import readout_routing as rr  # noqa: E402

with open(os.path.join(REPO, rr.LOCK_RELATIVE_PATH), encoding="utf-8") as h:
    lock = json.load(h)


def own_outcome(primary, guard_result):  # read off the lock's own table rows
    table = lock["outcome"]["table"]
    if primary == "NOT_INTERPRETABLE":
        return table[1][2]
    if primary in ("GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE"):
        return table[2][2]
    return table[4][2] if guard_result == "PASS" else table[3][2]


pairs = [(p, g) for p in rr.PRIMARY_RESULTS for g in rr.GUARD_RESULTS]
check("rr.outcome matches the lock's outcome table on all 12 (primary, guard) pairs", all(rr.outcome(p, g) == own_outcome(p, g) for p, g in pairs), len(pairs))
F, N, T = lock["primary"]["F_rho"], lock["primary"]["null_975"], lock["primary"]["effective_threshold"]["value"]
check("threshold == max(F_ρ, null₉₇.₅) == null₉₇.₅ (null binds; F_ρ < null)", T == max(F, N) == N and F < N, (F, N))
below = math.nextafter(T, -math.inf)
cases = {"ρ = threshold": (T, "PASS"), "ρ = threshold − 1 ulp": (below, "GUARD_FAILURE"), "ρ = F_ρ": (F, "GUARD_FAILURE"), "ρ = 1": (1.0, "PASS"),
         "ρ = −1": (-1.0, "GUARD_FAILURE"), "ρ undefined": (None, "NOT_INTERPRETABLE")}
for label, (rho, expected) in cases.items():
    got = rr.classify_primary(rho, F, N)
    check(f"classify_primary({label}) == {expected}", got == expected, got)
check("ENVELOPE_ONLY_FAILURE unreachable for these thresholds (null₉₇.₅ > F_ρ)", not (N <= F))
spec = rr.PRODUCTION_GUARD
check("lock guard spec == PRODUCTION_GUARD.to_json() (8 + 8, 12,870, max_upper 321)", lock["guard"]["spec"] == spec.to_json() and (spec.n_e, spec.n_n, spec.assignments, spec.max_upper) == (8, 8, 12870, 321))
check("321/12,870 ≤ 0.025 < 322/12,870 (exact)", Fraction(321, 12870) <= Fraction(1, 40) < Fraction(322, 12870))
check("en_decision(321) PASS, en_decision(322) FAIL, en_decision(12,870) FAIL",
      (rr.en_decision(321, spec), rr.en_decision(322, spec), rr.en_decision(12870, spec)) == ("PASS", "FAIL", "FAIL"))
check("guard units: 8 E then 8 N, the freeze's order", [u["word"] for u in lock["guard"]["units"]["E"]] == ["apple", "horse", "doctor", "king", "rabbit", "poet", "dragon", "lion"]
      and len(lock["guard"]["units"]["N"]) == 8, [u["word"] for u in lock["guard"]["units"]["N"]])
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"RULES {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
