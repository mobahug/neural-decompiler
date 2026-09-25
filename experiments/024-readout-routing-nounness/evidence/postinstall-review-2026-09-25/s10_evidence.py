"""Item 10: the new README text and evidence/lock-2026-09-25/README.md against verified facts; lock_run.json's sha256;
SHA256SUMS; archived evidence == the scratchpad originals; the pre-lock state reconstructed from today's state; the
README's 'correctly rounded exact values give e494736a…' claim by an own exact-integer route from the pinned checkpoint."""
import guard  # noqa: F401
from guard import REPO, canonical, sha256_bytes, sha256_file, summary

import hashlib
import json
import os
import struct
from decimal import Decimal, getcontext

import numpy as np

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + str(detail)) if detail else ''}", flush=True)


def read(path):
    with open(path, "rb") as h:
        return h.read()


EXP = os.path.join(REPO, "experiments/024-readout-routing-nounness")
EV = os.path.join(EXP, "evidence/lock-2026-09-25")
RV = os.path.join(EXP, "evidence/lock-review-2026-09-25")
SCR = os.path.dirname(guard.SCRATCH)
ORIG = {"lock": os.path.join(SCR, "lock024"), "install": os.path.join(SCR, "install_lock024"), "review": os.path.join(SCR, "lock_review024")}

# ---- A. SHA256SUMS, lock_run.json, byte identity with the originals ----------------------------------------------------
sums = {}
for line in read(os.path.join(EV, "SHA256SUMS")).decode().splitlines():
    digest, name = line.split("  ", 1)
    sums[name] = digest
archived = sorted(f for f in os.listdir(EV))
check("SHA256SUMS lists every archived file of lock-2026-09-25 except README.md and itself",
      sorted(sums) == sorted(set(archived) - {"README.md", "SHA256SUMS"}), sorted(set(archived) - set(sums)))
for name, digest in sums.items():
    check(f"SHA256SUMS entry {name} == sha256(archived file)", sha256_file(os.path.join(EV, name)) == digest)
LOCK_RUN_SHA = "b7a51118ddb998fa005d7dc32cba7bce2a18278a2ec79f94ea19db973be5f5e2"
check("lock_run.json: archived sha256 == b7a51118… == SHA256SUMS == scratch original",
      sha256_file(os.path.join(EV, "lock_run.json")) == LOCK_RUN_SHA == sums["lock_run.json"] == sha256_file(os.path.join(ORIG["lock"], "lock_run.json")))
origin_of = {n: ("lock" if os.path.exists(os.path.join(ORIG["lock"], n)) else "install") for n in sums}
for name in sorted(sums):
    src = os.path.join(ORIG[origin_of[name]], name)
    check(f"archived {name} == original {origin_of[name]}_…/{name} (bytes)", os.path.exists(src) and read(src) == read(os.path.join(EV, name)))
originals = sorted(os.listdir(ORIG["lock"])) + sorted(os.listdir(ORIG["install"]))
check("every lock024/ and install_lock024/ original is archived", sorted(originals) == sorted(sums), sorted(set(originals) ^ set(sums)))
review_arch = sorted(os.listdir(RV))
review_orig = sorted(os.listdir(ORIG["review"]))
check("lock-review archive == lock_review024/ originals + REVIEW.md", sorted(set(review_arch) - {"REVIEW.md"}) == review_orig, sorted(set(review_arch) ^ set(review_orig)))
for name in review_orig:
    check(f"archived review {name} == original (bytes)", read(os.path.join(RV, name)) == read(os.path.join(ORIG["review"], name)))
print("   note: REVIEW.md has no scratch original (the reviewer's report with the decisions appended); checked for content below")

# ---- B. README claims against verified facts ----------------------------------------------------------------------------
ev_readme = read(os.path.join(EV, "README.md")).decode()
exp_readme = read(os.path.join(EXP, "README.md")).decode()
review = read(os.path.join(RV, "REVIEW.md")).decode()
lock = json.loads(read(os.path.join(EXP, "preregistration-lock.json")))
state = json.loads(read(os.path.join(REPO, "outputs/experiment-024/results.json")))
run_record = json.loads(read(os.path.join(EV, "lock_run.json")))
facts = {
    "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d": sha256_file(os.path.join(EXP, "preregistration-lock.json")),
    "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739": lock["content_sha256"],
    "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54": sha256_file(os.path.join(EXP, "preregistration.md")),
    "e6cb37767d1d06c6ff40804a88eab569723afdb5": lock["module"]["blob"],
    "a703ac16c01f0960100ce1e9db470dfd04c0ce4251319d135fd57e38197d4995": lock["fresh"]["scores_sha256"],
    LOCK_RUN_SHA: sha256_file(os.path.join(EV, "lock_run.json")),
    "0.3136960600375234": repr(lock["primary"]["effective_threshold"]["value"]),
    "0.24411074612857814": repr(lock["primary"]["F_rho"]),
    "0.13502027836111233": repr(lock["fresh"]["maximum_calibration_score"]),
}
for text, name in ((exp_readme, "experiment README"), (ev_readme, "evidence README")):
    for claim, verified in facts.items():
        if claim in text:
            check(f"{name} states {claim[:16]}… == verified value", claim == verified, verified[:16])
for claim in ("58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d", "fcc437fca9730b8599333eeac95807786570f0d03bd01e49f52daec76ed4e32d"):
    check(f"evidence README states {claim[:12]}…", claim in ev_readme)
check("58891c24… == the state's digest now", state["state_sha256"] == "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d")
check("fcc437fc… == the lock's manifest_sha256", lock["confirmation_024"]["manifest_sha256"].startswith("fcc437fca9730b85"))
check("both READMEs name the effective threshold bound by the null", lock["primary"]["effective_threshold"]["binds"] == "null_975" and "bound by the null" in ev_readme and "bound by the null" in exp_readme)
spec = lock["guard"]["spec"]
check("E–N guard: 12,870 assignments, PASS iff K ≤ 321 (README) == lock spec", spec["assignments"] == 12870 and spec["max_upper"] == 321 and "K ≤ 321" in ev_readme and "12,870" in ev_readme)
check("lock_run.json: started 17:13:39.160, ended 17:14:05.116, exit 0, 1 load, 0 module calls, 0 refused, 0 captures",
      run_record["started_at"].startswith("2026-09-25T17:13:39.160") and run_record["ended_at"].startswith("2026-09-25T17:14:05.116") and run_record["exit_status"] == 0
      and run_record["events"]["load_model"] == 1 and run_record["events"]["module_calls_while_loading"] == 0 and run_record["events"]["module_calls_refused"] == 0
      and run_record["events"]["capture_calls"] == 0 and not run_record["events"]["forbidden_writes"])
check("README's run window and exit status match lock_run.json", "17:13:39.160 to 17:14:05.116 UTC" in ev_readme and ("exit\nstatus 0" in ev_readme or "exit status 0" in ev_readme))
writes = run_record["events"]["output_writes"]
check("lock_run.json writes: the two candidates + one atomic temp state (.results-*.json)",
      writes[:2] == ["candidate-lock.json", "candidate-preregistration.md"] and len(writes) == 3 and writes[2].startswith(".results-") and writes[2].endswith(".json"))
check("state.phases.lock.completed_at (17:14:05) within the run window", state["phases"]["lock"]["completed_at"] == "2026-09-25T17:14:05+00:00")
check("README: 3affe55 touches exactly the two paths; review PASS WITH NOTES, no blockers",
      "3affe55" in ev_readme and "PASS WITH NOTES" in ev_readme and "no blockers" in ev_readme and review.startswith("# Independent lock review") and "PASS WITH NOTES, no blockers" in review)
check("README: 5 of 8 E (apple, horse, doctor, poet, dragon); 23 of 40 (6 B, 4 D, 8 C, 5 E, 0 N)",
      "(apple, horse, doctor, poet, dragon)" in ev_readme and "23 of the 40 cues" in ev_readme and "(6 B, 4 D, 8 C, 5 E, 0 N)" in ev_readme)
check("README: status 'confirm next, only when separately authorized'; confirm and report not run", "only when separately authorized" in exp_readme and "**Not run:** `confirm` and `report`" in exp_readme)
check("README: the confirm launcher must assert rr.own_blob() == installed_lock['module']['blob'] before the model loads",
      "rr.own_blob() == installed_lock[\"module\"][\"blob\"]" in exp_readme and "before the model" in exp_readme)

# ---- C. the pre-lock state, reconstructed by undoing only the lock's entries ---------------------------------------------
pre = {k: v for k, v in state.items() if k != "state_sha256"}
pre["lock"] = None
pre["phases"] = {**pre["phases"], "lock": {"status": "not_started"}}
pre_digest = sha256_bytes(canonical(pre).encode("utf-8"))
pre_file = sha256_bytes((canonical({**pre, "state_sha256": pre_digest}) + "\n").encode("utf-8"))
print(f"   reconstructed pre-lock state digest {pre_digest}; file sha256 {pre_file}")
calib_readme = read(os.path.join(EXP, "evidence/calibrate-2026-09-25/README.md")).decode()
check("undoing only the lock's entries reproduces the committed pre-lock state digest 5ed26580… (calibrate evidence, be74d23)",
      pre_digest == "5ed26580be4c28310d072bf2c451dbeba01e4b1f8710b65072d25c9281d31252" and pre_digest in calib_readme)
check("... and its file sha256 4b319d62… (as the READMEs state)", pre_file.startswith("4b319d62") and "4b319d62" in ev_readme)

# ---- D. the exact route from the pinned checkpoint (own safetensors parser; no model object) ------------------------------
CKPT = os.path.expanduser("~/.cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c/model.safetensors")
blob = os.path.realpath(CKPT)
raw = read(blob)
check("checkpoint bytes sha256 == its LFS blob id 3da38833…", hashlib.sha256(raw).hexdigest() == os.path.basename(blob) and os.path.basename(blob).startswith("3da38833"))
n = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8:8 + n])
meta = header["gpt_neox.embed_in.weight"]
start, end = meta["data_offsets"]
emb16 = np.frombuffer(raw[8 + n + start:8 + n + end], dtype="<f2").reshape(meta["shape"])
emb32 = emb16.astype(np.float32)
own_emb = hashlib.sha256(json.dumps(list(emb32.shape)).encode("ascii") + b"|<f4|" + emb32.astype("<f4").tobytes()).hexdigest()
check("own embedding digest of the fp16 checkpoint rows (as float32) == lock embedding_sha256 9cd6f39b…",
      own_emb == lock["dependencies"]["model"]["embedding_sha256"], own_emb[:12])
record = json.loads(read(os.path.join(EXP, "calibration-v1.json")))
noun_ids, cue_ids = record["score"]["noun_row_ids"], record["score"]["calibration_cue_ids"]
fresh = [(c["word"], int(c["token_id"])) for c in lock["fresh"]["cues"]]
scale = 2 ** 24  # fp16 values are integer multiples of 2^-24
ints = {}
for t in set(noun_ids) | set(cue_ids) | {t for _, t in fresh}:
    row = emb16[t].astype(np.float64) * scale
    assert np.all(row == np.round(row))
    ints[t] = [int(v) for v in row]
S_n = [sum(ints[i][d] for i in noun_ids) for d in range(512)]
S_c = [sum(ints[i][d] for i in cue_ids) for d in range(512)]
C_n, C_c = sum(v * v for v in S_n), sum(v * v for v in S_c)
getcontext().prec = 90
exact = []
for _, t in fresh:
    e = ints[t]
    B = sum(v * v for v in e)
    A_n, A_c = sum(a * b for a, b in zip(e, S_n)), sum(a * b for a, b in zip(e, S_c))
    value = Decimal(A_n) / Decimal(B * C_n).sqrt() - Decimal(A_c) / Decimal(B * C_c).sqrt()
    exact.append(float(value))  # correctly rounded to binary64 (90-digit intermediate)
canon = [float(c["nounness"]) for c in lock["fresh"]["cues"]]
d = max(abs(a - b) for a, b in zip(exact, canon))
same = sum(1 for a, b in zip(exact, canon) if a == b)
ulps = max(abs(a - b) / np.spacing(abs(b)) for a, b in zip(exact, canon))
digest = hashlib.sha256(json.dumps([40]).encode("ascii") + b"|<f8|" + np.asarray(exact, dtype="<f8").tobytes()).hexdigest()
print(f"   exact route: max |exact - lock| = {d:.3e} ({ulps:.2f} ulp); bit-identical {same}/40; digest of the correctly rounded exact scores {digest}")
check("exact route agrees numerically with the lock (max abs diff < 1e-15)", d < 1e-15, f"{d:.3e}")
check("exact route: identical ordering", sorted(range(40), key=lambda i: exact[i]) == sorted(range(40), key=lambda i: canon[i]))
check("README claim: correctly rounded exact values give e494736a…", digest.startswith("e494736a") and "e494736a" in exp_readme, digest[:12])
s = summary()
check("guard: 0 refused", s["refused"] == 0)
print(f"ITEM10 {'PASS' if all(ok for _, ok in checks) else 'FAIL'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
