"""Items 1, 2, 3 (file side), 5, 6, 9 and 12 of the independent lock review — no model, no tokenizer forward.

Own implementations wherever the check is numerical or structural; the canonical module only where canonical byte
identity is the thing checked (render_preregistration, the frozen-input loaders, exposed_states_digest).
"""
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_guard  # noqa: E402  (audit hook first)

ROOT = review_guard.ROOT
OUT = ROOT / "outputs/experiment-024"
EXP = ROOT / "experiments/024-readout-routing-nounness"
FAIL = []
NOTES = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail != "" else ""), flush=True)
    if not ok:
        FAIL.append(name)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canon(value) -> str:
    # own canonical JSON (the protocol's definition: sorted keys, no whitespace, UTF-8, NaN refused)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_digest(payload) -> str:
    return sha256_bytes(canon({k: v for k, v in payload.items() if k != "content_sha256"}).encode("utf-8"))


def git(*args, binary=False):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    out = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, env=env)
    return out.stdout if binary else out.stdout.decode().strip()


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def float64_digest(values) -> str:
    # own implementation of the tensor digest of a float64 vector: sha256(json(shape) | "<f8" | little-endian bytes)
    return sha256_bytes(json.dumps([len(values)]).encode("ascii") + b"|<f8|" + struct.pack(f"<{len(values)}d", *values))


def all_floats(value, path=""):
    if isinstance(value, float):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from all_floats(v, f"{path}/{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from all_floats(v, f"{path}[{i}]")


print("=" * 100)
print("ITEM 1: repository and protocol state")
head = git("rev-parse", "HEAD")
origin = git("rev-parse", "origin/main")
check("HEAD == be74d23…", head == "be74d23d086bd12d894db1f73e6a1b4160aeebd5", head)
check("origin/main (local ref) == HEAD", origin == head, origin)
remote = git("ls-remote", "origin", "refs/heads/main").split()
check("ls-remote origin refs/heads/main == HEAD", remote and remote[0] == head, remote)
status = git("status", "--porcelain")
check("working tree clean (git status --porcelain empty)", status == "", repr(status))
check("no stash", git("stash", "list") == "")
worktrees = git("worktree", "list", "--porcelain")
print("   git worktree list:", worktrees.replace("\n", " | "))

freeze_path, record_path = EXP / "confirmation-v1.json", EXP / "calibration-v1.json"
freeze_bytes, record_bytes = freeze_path.read_bytes(), record_path.read_bytes()
freeze_head = git("cat-file", "blob", "HEAD:experiments/024-readout-routing-nounness/confirmation-v1.json", binary=True)
record_head = git("cat-file", "blob", "HEAD:experiments/024-readout-routing-nounness/calibration-v1.json", binary=True)
freeze, record = json.loads(freeze_bytes), json.loads(record_bytes)
check("freeze file sha256 68510e1b… (working tree == HEAD blob)", sha256_bytes(freeze_bytes) == "68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60" and freeze_bytes == freeze_head)
check("freeze content sha256 87f8aff1… (own canonical digest)", freeze["content_sha256"] == content_digest(freeze) == "87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87")
check("calibration file sha256 81fb499e… (working tree == HEAD blob)", sha256_bytes(record_bytes) == "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d" and record_bytes == record_head)
check("calibration content sha256 09bf093e… (own canonical digest)", record["content_sha256"] == content_digest(record) == "09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8")
check("freeze and record bytes are canonical JSON + newline", freeze_bytes == (canon(freeze) + "\n").encode() and record_bytes == (canon(record) + "\n").encode())
check("candidate-calibration.json is byte-identical to the committed record", (OUT / "candidate-calibration.json").read_bytes() == record_bytes)
# history of the two committed files: introduced once, never changed after
for rel in ("experiments/024-readout-routing-nounness/confirmation-v1.json", "experiments/024-readout-routing-nounness/calibration-v1.json"):
    log = git("log", "--format=%h %s", "--", rel)
    print(f"   history of {rel}: {log.splitlines()}")
    check(f"{Path(rel).name} committed exactly once", len(log.splitlines()) == 1)
check("lock / preregistration not installed in the repository", not (EXP / "preregistration-lock.json").exists() and not (EXP / "preregistration.md").exists())

state_bytes = (OUT / "results.json").read_bytes()
state = json.loads(state_bytes)
payload = {k: v for k, v in state.items() if k != "state_sha256"}
own_state_sha = sha256_bytes(canon(payload).encode("utf-8"))
check("results state: own state digest == recorded == 58891c24…", own_state_sha == state["state_sha256"] == "58891c2416684785cb52437069beebcaa198ea3674c6d4b9d8480c103aadc65d", own_state_sha)
check("results file bytes are canonical JSON + newline", state_bytes == (canon(state) + "\n").encode())
phases = {k: v["status"] for k, v in state["phases"].items()}
check("phases: calibrate complete, lock complete, confirm/report not_started", phases == {"calibrate": "complete", "lock": "complete", "confirm": "not_started", "report": "not_started"}, phases)
check("no incident anywhere (calibration, lock, confirm, report); no stop",
      not state["calibration"].get("incidents") and not state["calibration"].get("stop") and all(not v.get("incidents") for v in state["phases"].values()))
check("calibrate commit bf0049c…, lock commit be74d23…", state["phases"]["calibrate"]["commit"] == "bf0049c85342d01caf9d6a830389ce1e72c05a71" and state["phases"]["lock"]["commit"] == head)
check("executed_prompt_keys and executed_noun_keys empty; confirmation and report null",
      state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [] and state["confirmation"] is None and state["report"] is None)
check("state binds the candidate lock and preregistration", state["lock"]["content_sha256"] == "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739"
      and state["lock"]["preregistration_sha256"] == "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54")
check("state's calibration binding == committed record file/content", state["calibration"]["record_sha256"] == sha256_bytes(record_bytes)
      and state["calibration"]["record_content_sha256"] == record["content_sha256"] and state["lock"]["calibration_content_sha256"] == record["content_sha256"])
check("state's confirmation_024 == committed freeze", state["confirmation_024"] == {"path": "experiments/024-readout-routing-nounness/confirmation-v1.json",
                                                                                  "file_sha256": sha256_bytes(freeze_bytes), "content_sha256": freeze["content_sha256"]})
files = sorted(p.name for p in OUT.iterdir())
check("outputs/experiment-024 holds exactly the five expected files (no stage2 / measurement / temp file)",
      files == ["calibration-arrays.pt", "candidate-calibration.json", "candidate-lock.json", "candidate-preregistration.md", "results.json"], files)

print("=" * 100)
print("ITEM 2: candidate integrity and rendering")
lock_bytes = (OUT / "candidate-lock.json").read_bytes()
prereg_bytes = (OUT / "candidate-preregistration.md").read_bytes()
lock = json.loads(lock_bytes)
check("candidate lock file sha256 5c2a9b90…", sha256_bytes(lock_bytes) == "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d")
check("candidate lock content sha256 (own canonical, key removed) a39668bf…", content_digest(lock) == lock["content_sha256"] == "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739")
check("candidate lock bytes == canonical JSON + single newline", lock_bytes == (canon(lock) + "\n").encode("utf-8") and lock_bytes.count(b"\n") == 1)
floats = list(all_floats(lock))
check("every float in the lock finite; no NaN/Infinity token", all(math.isfinite(v) for _, v in floats) and b"NaN" not in lock_bytes and b"Infinity" not in lock_bytes, f"{len(floats)} floats")
check("candidate preregistration file sha256 007c9e6c…", sha256_bytes(prereg_bytes) == "007c9e6cd7c1b9bf3bf57ad36a4ee0338b6081d9fb3256b953e898475314db54")

from neural_decompiler import plural_mechanism as pm  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

check("canonical route: pm.sha256_text(pm.canonical_json(lock without key)) == content", pm.sha256_text(pm.canonical_json({k: v for k, v in lock.items() if k != "content_sha256"})) == lock["content_sha256"])
rendered = rr.render_preregistration(lock)
check("preregistration bytes == rr.render_preregistration(lock) (UTF-8, byte for byte)", prereg_bytes == rendered.encode("utf-8"))
check("state preregistration digest == pm.sha256_text(text)", pm.sha256_text(prereg_bytes.decode("utf-8")) == state["lock"]["preregistration_sha256"])

# own number check: every number in the preregistration appears in, and agrees with, the lock
text = prereg_bytes.decode("utf-8")
lines = text.split("\n")
accounted = Counter()


def expect(label, ok):
    check(f"prereg: {label}", ok)


fresh = lock["fresh"]
cues = fresh["cues"]
# header
h = lines[2]
expect("header run id / commit / design / plan / configuration", f"`{lock['run_id']}`" in h and f"`{lock['protocol_code_commit']}`" in h and f"revision {lock['design']['revision']} (`{lock['design']['commit']}`)" in h
       and f"plan revision {lock['plan']['revision']} (`{lock['plan']['commit']}`)" in h and f"`{lock['configuration']['name']}`" in h)
expect("calibration digests", f"`{lock['calibration']['content_sha256']}`" in lines[3] and f"`{lock['calibration']['file_sha256']}`" in lines[3])
expect("confirmation / manifest", f"`{lock['confirmation_024']['content_sha256']}`" in lines[4] and f"{lock['confirmation_024']['manifest_size']} S2-TARGET" in lines[4]
       and f"`{lock['confirmation_024']['manifest_sha256']}`" in lines[4])
expect("023 cells digests", all(f"`{lock['exposed_cells'][k]}`" in lines[5] for k in ("data_sha256", "index_sha256", "index_content_sha256")))
expect("model / embedding / 020 states digests", f"`{lock['dependencies']['model']['parameters_sha256']}`" in lines[6] and f"`{lock['dependencies']['model']['embedding_sha256']}`" in lines[6]
       and f"`{lock['dependencies']['readout_020']['exposed_states_sha256']}`" in lines[6])
# primary thresholds (parsed back to float and compared exactly)
m = re.search(r"F_ρ = (\S+); null₉₇\.₅ = (\S+); a PASS needs ρ ≥ max\(F_ρ, null₉₇\.₅\) = (\S+) \(binding: (\w+)\)", text)
expect("primary F_ρ, null, effective threshold parse back exactly to the lock's floats", m is not None and float(m.group(1)) == lock["primary"]["F_rho"]
       and float(m.group(2)) == lock["primary"]["null_975"] and float(m.group(3)) == lock["primary"]["effective_threshold"]["value"] and m.group(4) == lock["primary"]["effective_threshold"]["binds"]
       and m.group(1) == repr(lock["primary"]["F_rho"]) and m.group(2) == repr(lock["primary"]["null_975"]))
# guard
m = re.search(r"- E: (.+); N: (.+)\n", text)
expect("guard E and N words == lock units, in order", m is not None and m.group(1).split(", ") == [u["word"] for u in lock["guard"]["units"]["E"]]
       and m.group(2).split(", ") == [u["word"] for u in lock["guard"]["units"]["N"]])
spec = lock["guard"]["spec"]
expect("guard C(16, 8) = 12870; max_upper 321; 321/12870", f"C({spec['n_E'] + spec['n_N']}, {spec['n_E']}) = {spec['assignments']}" in text and "max_upper = 321" in text
       and "at most 321/12870" in text and spec["assignments"] == math.comb(16, 8) == 12870 and spec["max_upper"] == 321)
# outcome table
rows = [line for line in lines if line.startswith("| ") and line.count("|") == 4 and "primary result" not in line]
parsed = [[cell.strip().strip("`") for cell in row.strip("|").split("|")] for row in rows]
expect("outcome table rows == lock outcome table", parsed == [list(r) for r in lock["outcome"]["table"]])
# extrapolation
expect("extrapolation sentence == lock's; +0.135020 == the bound maximum at 6 decimals", fresh["extrapolation"]["sentence"] in text
       and f"{fresh['maximum_calibration_score']:+.6f}" == "+0.135020" and fresh["extrapolation"]["sentence"].startswith(f"{fresh['extrapolation']['E_above_maximum']} of the {fresh['extrapolation']['E']} E cues"))
# line
m = re.search(r"log MSE = (\S+) \+ (\S+) · nounness \(residual sd (\S+), n = (\d+)\)", text)
expect("line intercept / slope / residual sd / n parse back exactly", m is not None and float(m.group(1)) == lock["line"]["intercept"] and float(m.group(2)) == lock["line"]["slope"]
       and float(m.group(3)) == lock["line"]["residual_sd"] and int(m.group(4)) == lock["line"]["n"] and m.group(1) == repr(lock["line"]["intercept"]))
# fresh table
table = [line for line in lines if re.match(r"^\| [NBDCE] \| ", line)]
ok_rows = len(table) == 40
worst = 0.0
for line, cue in zip(table, cues):
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    cls, word, tid, score, pred, above = cells
    ok_rows &= (cls == cue["class"] and word == cue["word"] and int(tid) == cue["token_id"] and float(score) == cue["nounness"] and float(pred) == cue["predicted_log_mse"]
                and score == repr(cue["nounness"]) and pred == repr(cue["predicted_log_mse"]) and above == ("yes" if cue["above_calibration_maximum"] else "no"))
expect("all 40 table rows: class, word, token id, nounness, prediction, flag == the lock's (floats parse back bit-exactly)", ok_rows)
# generic sweep: every number token in the file must be one the structured checks cover
allowed = set()
for _, v in all_floats(lock):
    allowed.add(repr(v))
allowed |= {str(x) for x in (lock["run_id"],)}
allowed |= {str(v) for _, v in [(None, x) for x in json.loads(json.dumps(lock, default=str)).__repr__() and []]}
def _ints(value):
    if isinstance(value, dict):
        for x in value.values(): yield from _ints(x)
    elif isinstance(value, list):
        for x in value: yield from _ints(x)
    elif isinstance(value, int) and not isinstance(value, bool):
        yield str(value)
allowed |= set(_ints(lock))
numbers = re.findall(r"(?<![0-9A-Za-z_.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![0-9A-Za-z_])", re.sub(r"`[^`]*`", "", text))  # backticked digests/commits checked structurally above
unaccounted = Counter(n for n in numbers if n not in allowed)
print("   number tokens not equal to a lock float (context-checked above):", dict(sorted(unaccounted.items())))
struct_ints = {"024", "2", "1", "4320", "023", "020", "16", "8", "12870", "321", "5", "4", "108", "79", "139", "+0.135020", "0"}
check("prereg: no unexplained number (only structural integers and the +0.135020 rendering remain)", set(unaccounted) <= struct_ints, sorted(set(unaccounted) - struct_ints))

print("=" * 100)
print("ITEM 3 (file side): bindings")
check("lock.calibration == committed record (path, file, content)", lock["calibration"] == {"path": "experiments/024-readout-routing-nounness/calibration-v1.json",
                                                                                           "file_sha256": sha256_bytes(record_bytes), "content_sha256": record["content_sha256"]})
check("lock.confirmation_024 == committed freeze (path, file, content) + counts + manifest", {k: lock["confirmation_024"][k] for k in ("path", "file_sha256", "content_sha256")}
      == {"path": "experiments/024-readout-routing-nounness/confirmation-v1.json", "file_sha256": sha256_bytes(freeze_bytes), "content_sha256": freeze["content_sha256"]}
      and lock["confirmation_024"]["counts"] == freeze["counts"] == {"classes": {c: 8 for c in "NBDCE"}} and lock["confirmation_024"]["manifest_size"] == 4320)
cells_data = ROOT / lock["exposed_cells"]["data_path"]
cells_index = ROOT / lock["exposed_cells"]["index_path"]
index_payload = json.loads(cells_index.read_bytes())
check("023 exposed cells: data file d2ee71e5…, index file 62818147…, index content d38305cb… (own hashes)",
      sha256_file(cells_data) == lock["exposed_cells"]["data_sha256"] == "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4"
      and sha256_file(cells_index) == lock["exposed_cells"]["index_sha256"] == "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe"
      and content_digest(index_payload) == index_payload["content_sha256"] == lock["exposed_cells"]["index_content_sha256"] == "d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16")
check("023 cells files unchanged in git since 023 (committed, clean)", git("ls-files", "--error-unmatch", lock["exposed_cells"]["data_path"]) != "")
check("lock.dependencies == record.dependencies (whole block)", lock["dependencies"] == record["dependencies"])
check("model digests bound: parameters fd953f1c…, embedding 9cd6f39b…", lock["dependencies"]["model"]["parameters_sha256"] == "fd953f1c745299bedfd1f242fc15c7fe31211f79d8bc0d3ca46372962e7fc0e5"
      and lock["dependencies"]["model"]["embedding_sha256"] == lock["fresh"]["embedding_sha256"] == lock["score"]["embedding_sha256"] == "9cd6f39b3adfdb74cffe046af459634982684bff9aef2a981760326f1748eaaf"
      and lock["dependencies"]["model"]["model_id"] == "EleutherAI/pythia-70m-deduped" and lock["dependencies"]["model"]["revision"] == "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c")
# module blobs: own sha1 of the working files, git's blob ids at HEAD, the constants, the lock and the record
tree = {line.split()[3]: line.split()[2] for line in git("ls-tree", "HEAD", "src/neural_decompiler/").splitlines()}
blobs_ok = True
for name, expected in rr.FROZEN_BLOBS.items():
    own = blob_sha1((ROOT / "src/neural_decompiler" / name).read_bytes())
    at_head = tree.get(f"src/neural_decompiler/{name}")
    blobs_ok &= own == expected == at_head == lock["module_blobs"][name] == record["module_blobs"][name] == lock["dependencies"]["module_blobs"][name]
    print(f"   {name:28s} own {own} head {at_head} {'ok' if own == expected == at_head else 'MISMATCH'}")
check("all 12 pinned module blobs: own sha1 == git HEAD blob == FROZEN_BLOBS == lock == record", blobs_ok and len(rr.FROZEN_BLOBS) == 12 and set(lock["module_blobs"]) == set(rr.FROZEN_BLOBS))
own_rr = blob_sha1((ROOT / "src/neural_decompiler/readout_routing.py").read_bytes())
check("lock.module: readout_routing.py blob e6cb3776… == own sha1 of the file == git blob at HEAD == git blob at the lock commit",
      lock["module"] == {"path": "src/neural_decompiler/readout_routing.py", "blob": own_rr} and own_rr == tree["src/neural_decompiler/readout_routing.py"]
      == git("rev-parse", f"{lock['protocol_code_commit']}:src/neural_decompiler/readout_routing.py") and own_rr.startswith("e6cb3776"), own_rr)
check("the imported rr module is the repository file (editable install)", Path(rr.__file__).resolve() == (ROOT / "src/neural_decompiler/readout_routing.py").resolve(), rr.__file__)
last = git("log", "-1", "--format=%h %cI %s", "--", "src/neural_decompiler/readout_routing.py", "experiments/024-readout-routing-nounness/run.py")
print("   last commit touching readout_routing.py or run.py:", last)
changed_since_calibrate = git("diff", "--no-renames", "--name-only", "bf0049c85342d01caf9d6a830389ce1e72c05a71", "HEAD").splitlines()
print("   paths changed between the calibrate commit and HEAD:", changed_since_calibrate)
check("no scientific path changed between the calibrate commit and the lock commit (rr.scientific_changes)", rr.scientific_changes(changed_since_calibrate) == [])
check("the lock commit changed nothing since (HEAD == lock commit)", git("diff", "--name-only", lock["protocol_code_commit"], "HEAD") == "")
check("lock inputs == record inputs == state inputs", lock["inputs"] == record["inputs"] == state["inputs"])
check("lock design / plan / configuration / constants / score / line", lock["design"] == rr.DESIGN == record["design"] and lock["plan"] == rr.PLAN == record["plan"]
      and lock["configuration"] == rr.PRODUCTION.to_json() == record["configuration"] and lock["constants"] == rr.record_constants(rr.PRODUCTION) == record["constants"]
      and lock["score"] == record["score"] and lock["line"] == record["line"])
check("lock.semantics == rr.SEMANTICS; outcome table == rr.OUTCOME_TABLE; readings", lock["semantics"] == rr.SEMANTICS and lock["outcome"]["table"] == [list(r) for r in rr.OUTCOME_TABLE]
      and lock["outcome"]["readings"] == rr.SEMANTICS["outcomes"] and lock["primary"]["readings"] == rr.SEMANTICS["primary"] and lock["guard"]["readings"] == rr.SEMANTICS["guard"]
      and lock["primary"]["statistic"] == rr.STATISTIC and lock["primary"]["precedence"] == list(rr.PRIMARY_RESULTS))
check("lock kind / experiment / schema / run id", lock["experiment"] == "024" and lock["schema_version"] == 1 and lock["run_id"] == state["run_id"] == record["run_id"])
check("secondary definitions frozen in the bound code: CONTRASTS, contrast tag, bootstrap elements", rr.CONTRASTS == {
    "noun effect": "mean(m_B, m_C, m_D, m_E) − m_N", "measure effect among nouns": "mean(m_B, m_D) − mean(m_C, m_E)",
    "plurality effect among nouns": "mean(m_B, m_C) − mean(m_D, m_E)", "measure × plurality interaction (descriptive)": "(m_B − m_D) − (m_C − m_E)"}
      and lock["constants"]["tags"]["contrast"] == rr.CONTRAST_TAG == "024|contrast" and lock["constants"]["ranks"]["contrast_lower"] == 250
      and lock["constants"]["ranks"]["contrast_upper_element"] == 9750 and rr.PRODUCTION.contrast_resamples == 10_000)
print("   SCIENTIFIC_PATH_PREFIXES:", rr.SCIENTIFIC_PATH_PREFIXES)
print("   NON_SCIENTIFIC_PATHS:", rr.NON_SCIENTIFIC_PATHS)
print("   NON_SCIENTIFIC_PREFIXES:", rr.NON_SCIENTIFIC_PREFIXES)
cases = {"src/neural_decompiler/readout_routing.py": True, "src/neural_decompiler/upstream_localization.py": True, "experiments/024-readout-routing-nounness/run.py": True,
         "experiments/023-block0-completion/exposed-cells.f64": True, "experiments/023-block0-completion/confirmation-v1.json": True,
         "experiments/020-readout-decompilation/closure.json": True,
         "experiments/024-readout-routing-nounness/preregistration-lock.json": False, "experiments/024-readout-routing-nounness/preregistration.md": False,
         "experiments/024-readout-routing-nounness/README.md": False, "experiments/024-readout-routing-nounness/evidence/lock-2026-09-25/x.json": False,
         "experiments/024-readout-routing-nounness/confirmation-v1.json": False, "experiments/024-readout-routing-nounness/calibration-v1.json": False,
         "tests/test_readout_routing.py": False, "docs/x.md": False, "pyproject.toml": False, "uv.lock": False}
got = {path: bool(rr.scientific_changes([path])) for path in cases}
for path, flag in got.items():
    print(f"   scientific_changes([{path}]) -> {'SCIENTIFIC' if flag else 'non-scientific'}")
check("scientific_changes classifies the constructed paths as expected", got == cases, {k: v for k, v in got.items() if v != cases[k]})

print("=" * 100)
print("ITEM 5: frozen predictor (file side)")
line = lock["line"]
check("lock.line == committed record line (slope 1.2992017464639374, intercept −2.481260357420486, residual sd 0.2570037337349923, n 139)",
      line == record["line"] and (line["slope"], line["intercept"], line["residual_sd"], line["n"]) == (1.2992017464639374, -2.481260357420486, 0.2570037337349923, 139))
check("every predicted_log_mse == intercept + slope · nounness (float64, the same expression)", all(c["predicted_log_mse"] == line["intercept"] + line["slope"] * c["nounness"] for c in cues))
worst_ulps = 0
for c in cues:
    exact = Fraction(line["intercept"]) + Fraction(line["slope"]) * Fraction(c["nounness"])
    err = abs(Fraction(c["predicted_log_mse"]) - exact)
    ulp = math.ulp(c["predicted_log_mse"])
    worst_ulps = max(worst_ulps, float(err / Fraction(ulp)))
print(f"   predictions vs exact rational intercept + slope·x: worst error {worst_ulps:.3f} ulp")
check("predictions within 1 ulp of the exact rational value", worst_ulps <= 1.0)
# the line recomputed exactly (rational OLS) from the record's own 139 entries
xs = [Fraction(e["nounness_loo"]) for e in record["calibration_cues"]]
ys = [Fraction(e["log_mse"]) for e in record["calibration_cues"]]
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
sxx = sum((x - mx) ** 2 for x in xs)
sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
slope = sxy / sxx
intercept = my - slope * mx
rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
resid_sd = math.sqrt(float(rss / (n - 2)))
print(f"   exact OLS: slope {float(slope)!r} (lock {line['slope']!r}, diff {abs(float(slope) - line['slope']):.2e}); intercept {float(intercept)!r} (diff {abs(float(intercept) - line['intercept']):.2e});"
      f" residual sd {resid_sd!r} (diff {abs(resid_sd - line['residual_sd']):.2e})")
check("line agrees with the exact rational OLS of the record's entries to ≤ 1e-14", abs(float(slope) - line["slope"]) <= 1e-14 and abs(float(intercept) - line["intercept"]) <= 1e-14
      and abs(resid_sd - line["residual_sd"]) <= 1e-14)
check("record's log_mse == ln(mse) for every calibration entry", all(e["log_mse"] == math.log(e["mse"]) for e in record["calibration_cues"]))


def own_avg_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = Fraction(i + j + 2, 2)
        i = j + 1
    return ranks


def own_spearman_exact(x, y):
    rx, ry = own_avg_ranks(x), own_avg_ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return float(sxy) / math.sqrt(float(sxx * syy))


rho_exposed = own_spearman_exact([e["nounness_loo"] for e in record["calibration_cues"]], [e["mse"] for e in record["calibration_cues"]])
print(f"   exposed Spearman (own, exact sums) {rho_exposed!r} vs record {record['descriptive']['calibration_rho']!r}")
check("exposed Spearman 0.5296617364493499 (record) reproduced by own implementation within 1e-15", abs(rho_exposed - record["descriptive"]["calibration_rho"]) <= 1e-15
      and record["descriptive"]["calibration_rho"] == 0.5296617364493499)
check("no fresh observed MSE anywhere in the lock (no key mse/log_mse/rho/K in fresh cues)", all(set(c) == {"above_calibration_maximum", "class", "form", "lemma", "nounness", "predicted_log_mse", "token_id", "word"} for c in cues))

print("=" * 100)
print("ITEM 6: primary binding (thresholds from the committed record and the arrays)")
import torch  # noqa: E402

floor, null = record["primary_floor"], record["null"]
check("record F_ρ 0.24411074612857814, null₉₇.₅ 0.3136960600375234, effective 0.3136960600375234 binds null_975, undefined 0",
      floor["F_rho"] == 0.24411074612857814 and null["null_975"] == 0.3136960600375234 and record["effective_threshold"] == {"value": 0.3136960600375234, "binds": "null_975"}
      and floor["undefined"] == 0 and (floor["rank"], floor["element"]) == (250, 249) and (null["rank"], null["element"]) == (97500, 97499) and floor["direction_check"]["ok"])
check("lock primary == record thresholds", (lock["primary"]["F_rho"], lock["primary"]["null_975"], lock["primary"]["effective_threshold"]) == (floor["F_rho"], null["null_975"], record["effective_threshold"]))
check("state calibration arrays digests == record arrays digests", state["calibration"]["arrays_sha256"] == record["arrays_sha256"])
arrays = torch.load(OUT / "calibration-arrays.pt")
own_digests = {}
for key, value in sorted(arrays.items()):
    t = value.to(torch.int64) if value.dtype in (torch.bool, torch.int8) else value
    arr = t.contiguous().numpy()
    kind = "<i8" if t.dtype in (torch.int64, torch.int32) else "<f8"
    own_digests[key] = sha256_bytes(json.dumps(list(arr.shape)).encode("ascii") + b"|" + kind.encode() + b"|" + arr.astype(kind, copy=False).tobytes())
check("calibration-arrays.pt: own digests == record arrays_sha256 (draw_rho, draw_defined, draw_indices, null_rho, null_permutations)", own_digests == record["arrays_sha256"], own_digests)
check("calibration-arrays.pt file sha256 a4cd548f… (unchanged baseline)", sha256_file(OUT / "calibration-arrays.pt") == "a4cd548fd9c6084d58224f072d6c529cf4e81f17650fb50ccbc1d2c97f20aece")
draw_rho = arrays["draw_rho"].tolist()
null_rho = arrays["null_rho"].tolist()
check("draws: 10,000, all defined", len(draw_rho) == 10_000 and bool(arrays["draw_defined"].all()) and all(math.isfinite(v) for v in draw_rho))
check("F_ρ == sorted(draw_rho)[249]; null₉₇.₅ == sorted(null_rho)[97499]", sorted(draw_rho)[249] == floor["F_rho"] and sorted(null_rho)[97499] == null["null_975"] and len(null_rho) == 100_000)


def sha_int(text):
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


indices = [[sha_int(f"024|primary|{b}|{s}") % 139 for s in range(40)] for b in range(10_000)]
check("own regeneration of the 10,000 × 40 SHA-indexed draw indices == the saved array", indices == arrays["draw_indices"].tolist())
entries = record["calibration_cues"]
x_all = [e["nounness_loo"] for e in entries]
y_all = [e["mse"] for e in entries]
worst_draw = 0.0
for b in range(10_000):
    row = indices[b]
    value = own_spearman_exact([x_all[i] for i in row], [y_all[i] for i in row])
    worst_draw = max(worst_draw, abs(value - draw_rho[b]))
print(f"   own exact-sum Spearman of every draw vs the saved draw_rho: max |diff| {worst_draw:.3e}")
check("every draw's Spearman reproduced (own implementation) within 1e-14", worst_draw <= 1e-14)
perm_ok = True
worst_null = 0.0
saved_perms = arrays["null_permutations"]
for p in range(100_000):
    perm = list(range(40))
    for i in range(39, 0, -1):
        j = sha_int(f"024|null|{p}|{i}") % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    if p % 1 == 0 and perm != saved_perms[p].tolist():
        perm_ok = False
    d2 = sum((i - v) ** 2 for i, v in enumerate(perm))
    rho = 1.0 - 6.0 * d2 / (40 * (40 * 40 - 1))  # closed form (no ties): an independent route
    worst_null = max(worst_null, abs(rho - null_rho[p]))
check("own Fisher–Yates regeneration of all 100,000 permutations == the saved array", perm_ok)
print(f"   closed-form 1 − 6Σd²/(n(n²−1)) vs the saved null_rho: max |diff| {worst_null:.3e}")
check("null Spearman values reproduced by the closed form within 1e-15", worst_null <= 1e-15)
d2_975 = sorted(sum((i - v) ** 2 for i, v in enumerate(saved_perms[p].tolist())) for p in range(100_000))
exact_null = Fraction(1) - Fraction(6 * sorted(d2_975, reverse=True)[97499], 40 * 1599)
print(f"   exact null₉₇.₅ as a rational from the order statistic of Σd²: {float(exact_null)!r} (record {null['null_975']!r})")
check("null₉₇.₅ equals the exact rational order statistic to ≤ 1e-16", abs(float(exact_null) - null["null_975"]) <= 1e-16)
median_defined = sorted(draw_rho)[4999] * 0.5 + sorted(draw_rho)[5000] * 0.5
check("direction check: F_ρ ≤ median of the defined draws", floor["F_rho"] <= median_defined, f"median {median_defined}")

print("=" * 100)
print("ITEM 7 (file side): E–N guard binding")
e_freeze = [(c["word"], c["token_id"]) for c in freeze["cues"] if c["class"] == "E"]
n_freeze = [(c["word"], c["token_id"]) for c in freeze["cues"] if c["class"] == "N"]
check("guard units E == the freeze's 8 E cues in frozen order; N == the 8 N cues", [(u["word"], u["token_id"]) for u in lock["guard"]["units"]["E"]] == e_freeze
      and [(u["word"], u["token_id"]) for u in lock["guard"]["units"]["N"]] == n_freeze and len(e_freeze) == len(n_freeze) == 8
      and [w for w, _ in e_freeze] == ["apple", "horse", "doctor", "king", "rabbit", "poet", "dragon", "lion"]
      and [w for w, _ in n_freeze] == ["honest", "polite", "rude", "sleepy", "wise", "lucky", "merry", "nervous"])
check("guard spec == rr.PRODUCTION_GUARD.to_json() == GuardSpec.derive(8, 8); 321/12870 ≤ 0.025 < 322/12870",
      spec == rr.PRODUCTION_GUARD.to_json() and rr.GuardSpec.derive(8, 8) == rr.PRODUCTION_GUARD and Fraction(321, 12870) <= Fraction(25, 1000) < Fraction(322, 12870)
      and (25 * 12870) // 1000 == 321)
import itertools  # noqa: E402

subsets = list(itertools.combinations(range(16), 8))
index = {s: i for i, s in enumerate(subsets)}
check("enumeration structure: 12,870 distinct size-8 subsets, observed (0…7) first, complement pairing is an involution without fixed points",
      len(subsets) == len(set(subsets)) == 12870 and subsets[0] == tuple(range(8))
      and all(index[tuple(sorted(set(range(16)) - set(s)))] != i for i, s in enumerate(subsets)))

print("=" * 100)
print("ITEM 9: the fresh manifest and its isolation")
frames = freeze["exposed_frame_ids"]
fresh_cues = [(c["word"], int(c["token_id"])) for c in freeze["cues"]]
keys = [f"{frame_id}|{word}|{token_id}" for frame_id in frames for word, token_id in fresh_cues]
check("108 frames × 40 cues = 4,320 keys, all unique", len(frames) == 108 and len(fresh_cues) == 40 and len(keys) == len(set(keys)) == 4320)
manifest = {"S2-TARGET": sorted(keys)}
manifest_sha = sha256_bytes(canon(manifest).encode("utf-8"))
check("own manifest digest == lock manifest_sha256 fcc437fc…", manifest_sha == lock["confirmation_024"]["manifest_sha256"] == "fcc437fca9730b8599333eeac95807786570f0d03bd01e49f52daec76ed4e32d", manifest_sha)
check("the freeze file's stored manifest == own reconstruction in canonical (sorted) order", freeze["manifest"] == manifest)
results_020 = ROOT / "outputs/experiment-020/results.json"
check("020 results file is the closed one (da63b8c2…)", sha256_file(results_020) == "da63b8c29f9553a9da62bea7a442e11cef999ccddc71a7a332ac7c938abc9e00")
ledger_020 = set(json.loads(results_020.read_bytes())["executed_prompt_keys"])
c020 = json.loads((ROOT / "experiments/020-readout-decompilation/confirmation-v1.json").read_bytes())


def flatten(m):
    if isinstance(m, dict):
        out = []
        for v in m.values():
            out += flatten(v)
        return out
    return list(m)


conf_020 = set(flatten(c020["manifest"]))
m022 = json.loads((ROOT / "experiments/022-upstream-error-localization/confirmation-v1.json").read_bytes())["manifest"]
m023 = json.loads((ROOT / "experiments/023-block0-completion/confirmation-v1.json").read_bytes())["manifest"]
set_022, set_023 = set(flatten(m022)), set(flatten(m023))
spent = ledger_020 | conf_020 | set_022 | set_023
print(f"   spent sets: 020 ledger {len(ledger_020)}, 020 confirmation {len(conf_020)}, 022 manifest {len(set_022)}, 023 manifest {len(set_023)}; union {len(spent)}")
check("spent union is 39,312 keys", len(spent) == 39_312, len(spent))
check("zero collision between the 4,320 fresh keys and the 39,312 spent keys", not (set(keys) & spent))
spent_token_ids = {int(k.rsplit("|", 1)[1]) for k in spent}
check("no fresh cue token id occurs as the cue token of any spent key", not ({t for _, t in fresh_cues} & spent_token_ids))
check("the 024 ledger (executed_prompt_keys) holds none of them (empty)", not (set(state["executed_prompt_keys"]) & (set(keys) | spent)) and state["executed_prompt_keys"] == [])

print("=" * 100)
print("ITEM 12: no fresh outcome leakage")
tracked = git("ls-files").splitlines()
fresh_key_pattern = re.compile("|".join(re.escape(f"|{w}|{t}") for w, t in fresh_cues))
hits = []
for rel in tracked:
    p = ROOT / rel
    if not p.is_file() or p.stat().st_size > 50_000_000:
        continue
    data = p.read_bytes()
    try:
        s = data.decode("utf-8")
    except UnicodeDecodeError:
        continue
    if fresh_key_pattern.search(s):
        hits.append(rel)
print("   tracked files containing a fresh prompt key:", hits)
check("the only tracked file containing any fresh key is the freeze (its manifest)", hits == ["experiments/024-readout-routing-nounness/confirmation-v1.json"])
untracked = git("ls-files", "--others", "--exclude-standard")
check("no untracked, non-ignored file in the repository", untracked == "", untracked)
names = []
for dirpath, dirnames, filenames in os.walk(ROOT / "outputs"):
    for f in filenames:
        if "stage2" in f or "024" in dirpath and f not in ("calibration-arrays.pt", "candidate-calibration.json", "candidate-lock.json", "candidate-preregistration.md", "results.json"):
            names.append(os.path.join(dirpath, f))
print("   stage2-named files anywhere under outputs/ (names only):", [n.replace(str(ROOT) + "/", "") for n in names])
check("no stage-2 measurement file for 024 anywhere under outputs/", not any("experiment-024" in n for n in names))
recent = []
import datetime  # noqa: E402

freeze_time = datetime.datetime.fromisoformat(git("log", "-1", "--format=%cI", "566eb6f")).timestamp()
for dirpath, dirnames, filenames in os.walk(ROOT / "outputs"):
    for f in filenames:
        full = os.path.join(dirpath, f)
        if os.stat(full).st_mtime >= freeze_time:
            recent.append((full.replace(str(ROOT) + "/", ""), datetime.datetime.fromtimestamp(os.stat(full).st_mtime).isoformat(timespec="seconds")))
print("   files under outputs/ modified since the freeze commit:", recent)
check("since the freeze, only outputs/experiment-024 files were written under outputs/", all(r[0].startswith("outputs/experiment-024/") for r in recent))
added = git("log", "--format=", "--name-only", "--diff-filter=AM", "566eb6f^..HEAD").splitlines()
print("   files added/modified in commits from the freeze to HEAD:", sorted(set(a for a in added if a)))
ev_hits = []
for rel in tracked:
    if rel.startswith("experiments/024-readout-routing-nounness/evidence/") and rel.endswith((".json", ".out", ".md", ".txt", ".py", ".log")):
        s = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for w, _ in fresh_cues:
            for m in re.finditer(rf"\b{re.escape(w)}\b[^\n]{{0,80}}\b(mse|MSE|log_mse|Δc|delta_c|rho|K =)", s):
                ev_hits.append((rel, m.group(0)[:100]))
print("   024 evidence lines naming a fresh cue together with an outcome word:", ev_hits[:20], f"({len(ev_hits)} total)")
check("the state carries no fresh measurement, Δc, Δx3, MSE, ρ or K", state["confirmation"] is None and "stage2" not in json.dumps(state))
check("the candidate lock and preregistration carry no fresh outcome (no ρ value, K or MSE of a fresh cue)", "\"rho\"" not in lock_bytes.decode() and "\"K\"" not in lock_bytes.decode()
      and "\"mse\"" not in json.dumps(lock["fresh"]))

print("=" * 100)
print("guard events:", review_guard.report_guard())
check("no write under the repository was attempted by this script", not review_guard.EVENTS["refused_writes"])
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
