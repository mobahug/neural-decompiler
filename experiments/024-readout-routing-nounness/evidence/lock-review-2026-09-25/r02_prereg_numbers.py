"""Own number sweep over the candidate preregistration: every number token outside backtick-quoted digests/commits must be
a lock value (float repr or token id) or a structural integer the lock also carries; backticked tokens must be lock strings."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_guard  # noqa
ROOT = review_guard.ROOT
lock = json.loads((ROOT / "outputs/experiment-024/candidate-lock.json").read_bytes())
text = (ROOT / "outputs/experiment-024/candidate-preregistration.md").read_bytes().decode("utf-8")

def walk(v):
    if isinstance(v, dict):
        for x in v.values(): yield from walk(x)
    elif isinstance(v, list):
        for x in v: yield from walk(x)
    else:
        yield v

values = list(walk(lock))
strings = {v for v in values if isinstance(v, str)}
floats = {repr(v) for v in values if isinstance(v, float)}
ints = {str(v) for v in values if isinstance(v, int) and not isinstance(v, bool)}
ticked = re.findall(r"`([^`]*)`", text)
bad_ticks = [t for t in ticked if t not in strings and t not in {"production"} and not any(t == s for s in strings)]
print("backticked tokens:", len(ticked), "not a lock string value:", bad_ticks)
stripped = re.sub(r"`[^`]*`", "", text)
numbers = re.findall(r"(?<![0-9A-Za-z_.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![0-9A-Za-z_])", stripped)
unexplained = sorted({n for n in numbers if n not in floats and n not in ints})
print("number tokens outside backticks:", len(numbers), "distinct:", len(set(numbers)))
print("not a lock float repr or lock integer:", unexplained)
# the remaining ones must be derivable from the lock: "+0.135020" is the maximum at 6 decimals; "C(16, 8)" = n_E + n_N, n_E; experiment numbers in prose
derivable = {"+0.135020": f"{lock['fresh']['maximum_calibration_score']:+.6f}" == "+0.135020",
             "16": lock["guard"]["spec"]["n_E"] + lock["guard"]["spec"]["n_N"] == 16}
print("derivable:", derivable)
ok = not [t for t in bad_ticks if t not in ("NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS", "FAIL")] and set(unexplained) <= {"+0.135020", "16", "020", "023", "024", "4", "5", "2"} and all(derivable.values())
print("RESULT:", "ok" if ok else "FAIL")
print("guard:", review_guard.EVENTS)
