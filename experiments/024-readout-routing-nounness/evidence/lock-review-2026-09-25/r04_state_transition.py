"""Reconstruct the pre-lock results state from the current one by undoing exactly the two fields Runner.lock writes
(state["lock"], state["phases"]["lock"]) and compare with the pre-lock digests committed in the calibration review
evidence (state digest 5ed26580…, file sha256 4b319d62…)."""
import copy, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_guard  # noqa
ROOT = review_guard.ROOT
state = json.loads((ROOT / "outputs/experiment-024/results.json").read_bytes())
canon = lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
pre = copy.deepcopy(state)
pre.pop("state_sha256")
pre["lock"] = None
pre["phases"]["lock"] = {"status": "not_started"}
digest = hashlib.sha256(canon(pre).encode()).hexdigest()
file_sha = hashlib.sha256((canon({**pre, "state_sha256": digest}) + "\n").encode()).hexdigest()
review = (ROOT / "experiments/024-readout-routing-nounness/evidence/calibration-review-2026-09-25/REVIEW.md").read_text()
print("reconstructed pre-lock state digest:", digest)
print("reconstructed pre-lock file sha256:  ", file_sha)
print("committed evidence names 5ed26580… and file 4b319d62…:", "5ed26580" in review, "4b319d62" in review)
ok = digest == "5ed26580be4c28310d072bf2c451dbeba01e4b1f8710b65072d25c9281d31252" and file_sha.startswith("4b319d62")
print("RESULT:", "ok — the lock run changed only state['lock'] and state['phases']['lock']" if ok else "FAIL")
print("guard:", review_guard.EVENTS)
