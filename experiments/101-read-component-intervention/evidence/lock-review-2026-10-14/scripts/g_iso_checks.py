"""G1's float64 checks, computed my way (math.fsum over exact float products) on the 288 directions that my
torch-arithmetic implementation reproduces bit for bit (their digests equal the lock's), plus the mutual overlap."""
from common import install_fs_guard

install_fs_guard()
import json, math  # noqa: E402,E401
import mine  # noqa: E402
from common import EVENTS, write_json  # noqa: E402

l011 = mine.lock_011()
wc_t, norm_t, what_t = mine.wc_torch(l011["read_weight"])
conf = mine.confirmation()
lock = json.loads((mine.ROOT / "outputs/experiment-101/candidate-lock.json").read_text())
w_c = wc_t.tolist()
worst = {"abs_norm_minus_1": 0.0, "abs_mean": 0.0, "abs_dot_w_c": 0.0, "abs_dot_w_hat": 0.0, "max_abs_offdiag_cos": 0.0}
mismatch = 0
for frame in conf["frames"]:
    fid = frame["frame_id"]
    vs = []
    for j in range(1, 9):
        v = mine.iso_torch(what_t, fid, j)
        mismatch += mine.tensor_digest_torch(v) != lock["geometry"]["iso_sha256"][fid][j - 1]
        x = v.tolist()
        vs.append(x)
        worst["abs_norm_minus_1"] = max(worst["abs_norm_minus_1"], abs(math.sqrt(math.fsum(a * a for a in x)) - 1.0))
        worst["abs_mean"] = max(worst["abs_mean"], abs(math.fsum(x) / len(x)))
        worst["abs_dot_w_c"] = max(worst["abs_dot_w_c"], abs(math.fsum(a * b for a, b in zip(x, w_c))))
        worst["abs_dot_w_hat"] = max(worst["abs_dot_w_hat"], abs(math.fsum(a * b for a, b in zip(x, what_t.tolist()))))
    for a in range(8):
        for b in range(a + 1, 8):
            worst["max_abs_offdiag_cos"] = max(worst["max_abs_offdiag_cos"], abs(math.fsum(p * q for p, q in zip(vs[a], vs[b]))))
out = {"n": 36 * 8, "digest_mismatches": mismatch, "worst": worst, "w_c_norm": norm_t, "events": EVENTS}
write_json("g_iso_checks.json", out)
print(json.dumps(out, indent=1))
