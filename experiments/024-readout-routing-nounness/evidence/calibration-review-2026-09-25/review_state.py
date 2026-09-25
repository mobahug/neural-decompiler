"""Independent review of Experiment 024's calibrate — part 1: repository/protocol state, record/state digests, the
non-model bindings, the per-cue MSE from 023's raw committed artifact, the E–N binding and the fresh-measurement
boundary. No model, no tokenizer, no prompt. Read-only (the guard refuses any repository write)."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/calibration_review024")
import rguard  # noqa: E402  (first: the audit hook)

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(rguard.ROOT)
SCRATCH = Path(__file__).resolve().parent
OUT = ROOT / "outputs/experiment-024"
HEAD_EXPECTED = "bf0049c85342d01caf9d6a830389ce1e72c05a71"
EXPECT = {
    "candidate_file": "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d",
    "candidate_content": "09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8",
    "state": "5ed26580be4c28310d072bf2c451dbeba01e4b1f8710b65072d25c9281d31252",
    "freeze_file": "68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60",
    "freeze_content": "87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87",
    "cells_data": "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4",
    "cells_index_file": "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe",
    "cells_index_content": "d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16",
}
STRATA = ("determiner-like", "quantity", "adjective")
results = {"checks": [], "failures": []}


def check(name, ok, detail=""):
    tag = "ok" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f": {detail}" if detail != "" else ""), flush=True)
    results["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:2000]})
    if not ok:
        results["failures"].append(name)


def cjson(value):
    """My canonical JSON (the protocol's convention: sorted keys, compact separators, UTF-8, no NaN)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def content_digest(payload, key="content_sha256"):
    return sha256_bytes(cjson({k: v for k, v in payload.items() if k != key}).encode("utf-8"))


def blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def git(*args, timeout=60):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------------------------------------------
print("== Item 1: repository and protocol state ==")
head = git("rev-parse", "HEAD").stdout.strip()
origin = git("rev-parse", "origin/main").stdout.strip()
check("HEAD is bf0049c", head == HEAD_EXPECTED, head)
check("local origin/main is bf0049c", origin == HEAD_EXPECTED, origin)
remote = git("ls-remote", "origin", "refs/heads/main", timeout=60)
remote_sha = remote.stdout.split()[0] if remote.returncode == 0 and remote.stdout.strip() else None
check("remote origin main (ls-remote, read-only) is bf0049c", remote_sha == HEAD_EXPECTED, remote_sha or remote.stderr.strip()[:200])
porcelain = git("status", "--porcelain").stdout
check("working tree clean (git status --porcelain empty)", porcelain == "", repr(porcelain[:300]))
untracked_ignored = git("status", "--porcelain", "--ignored", "--", "outputs/experiment-024").stdout
check("outputs/experiment-024 is ignored, not tracked", untracked_ignored.strip() == "!! outputs/experiment-024/", repr(untracked_ignored))
freeze_path = ROOT / "experiments/024-readout-routing-nounness/confirmation-v1.json"
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
check("freeze file sha256", sha256_file(freeze_path) == EXPECT["freeze_file"], sha256_file(freeze_path))
check("freeze content digest recomputes (own canonical JSON) and equals 87f8aff1…",
      freeze["content_sha256"] == content_digest(freeze) == EXPECT["freeze_content"], content_digest(freeze))
check("freeze file is canonical JSON + newline", freeze_path.read_bytes() == (cjson(freeze) + "\n").encode("utf-8"))
freeze_log = git("log", "--format=%h %s", "--", str(freeze_path.relative_to(ROOT))).stdout.strip().splitlines()
check("freeze file committed once (566eb6f) and unchanged since", len(freeze_log) == 1 and freeze_log[0].startswith("566eb6f"), freeze_log)
check("freeze file identical to HEAD's blob", git("diff", "--quiet", "HEAD", "--", str(freeze_path.relative_to(ROOT))).returncode == 0)

state_path, record_path, arrays_path = OUT / "results.json", OUT / "candidate-calibration.json", OUT / "calibration-arrays.pt"
listing = sorted(os.listdir(OUT))
check("outputs/experiment-024 holds exactly the three calibrate outputs", listing == ["calibration-arrays.pt", "candidate-calibration.json", "results.json"], listing)
state_bytes = state_path.read_bytes()
state = json.loads(state_bytes)
payload = {k: v for k, v in state.items() if k != "state_sha256"}
own_state_digest = sha256_bytes(cjson(payload).encode("utf-8"))
check("results state verifies its own digest (own canonical JSON)", own_state_digest == state["state_sha256"], own_state_digest)
check("results state digest is 5ed26580…", state["state_sha256"] == EXPECT["state"])
check("results state file is canonical JSON + newline", state_bytes == (cjson(state) + "\n").encode("utf-8"))
phases = {name: entry.get("status") for name, entry in state["phases"].items()}
check("phases: calibrate complete; lock, confirm, report not_started", phases == {"calibrate": "complete", "lock": "not_started", "confirm": "not_started", "report": "not_started"}, phases)
cal_phase = state["phases"]["calibrate"]
check("calibrate phase: one commit bf0049c, no incidents key", cal_phase.get("commit") == HEAD_EXPECTED and "incidents" not in cal_phase, {k: v for k, v in cal_phase.items() if k != "runtime"})
check("calibrate phase keys are exactly status/started_at/completed_at/commit/runtime", set(cal_phase) == {"status", "started_at", "completed_at", "commit", "runtime"}, sorted(cal_phase))
check("state protocol_code_commit bf0049c, git_dirty false", state["protocol_code_commit"] == HEAD_EXPECTED and state["git_dirty"] is False)
cal = state["calibration"]
check("state calibration: no incidents, no stop, no cross_check failure", not any(k in cal for k in ("incidents", "stop", "cross_check")), sorted(cal))
check("executed_prompt_keys and executed_noun_keys empty", state["executed_prompt_keys"] == [] and state["executed_noun_keys"] == [])
check("lock, confirmation and report are null", state["lock"] is None and state["confirmation"] is None and state["report"] is None)
for name in ("lock", "confirm", "report"):
    check(f"phase {name} has only a status", state["phases"][name] == {"status": "not_started"}, state["phases"][name])
check("created_at == started_at (single fresh state)", state["created_at"] == cal_phase["started_at"], (state["created_at"], cal_phase["started_at"], cal_phase["completed_at"]))

record_bytes = record_path.read_bytes()
record = json.loads(record_bytes)
check("candidate file sha256 81fb499e…", sha256_bytes(record_bytes) == EXPECT["candidate_file"], sha256_bytes(record_bytes))
check("candidate content digest recomputes (own canonical JSON) and is 09bf093e…", record["content_sha256"] == content_digest(record) == EXPECT["candidate_content"], content_digest(record))
check("candidate file is canonical JSON + newline (byte-identical re-serialization)", record_bytes == (cjson(record) + "\n").encode("utf-8"))
check("state binds this candidate (record_sha256, content, path)", cal.get("record_sha256") == EXPECT["candidate_file"] and cal.get("record_content_sha256") == EXPECT["candidate_content"]
      and cal.get("candidate_path") == str(record_path))
check("state's F_rho, null_975, line, arrays digests, dependencies equal the record's",
      cal["F_rho"] == record["primary_floor"]["F_rho"] and cal["null_975"] == record["null"]["null_975"] and cal["line"] == record["line"]
      and cal["arrays_sha256"] == record["arrays_sha256"] and cal["dependencies"] == record["dependencies"] and cal["checks"] == record["checks"])
check("record and state share run_id, commit, inputs, module blobs, configuration, design, plan",
      record["run_id"] == state["run_id"] and record["protocol_code_commit"] == HEAD_EXPECTED and record["inputs"] == state["inputs"]
      and record["module_blobs"] == state["module_blobs"] and record["configuration"] == state["configuration"] and record["design"] == state["design"] and record["plan"] == state["plan"])
check("state binds the freeze (confirmation_024)", state["confirmation_024"] == record["confirmation_024"])
expected_keys = {"experiment", "schema_version", "kind", "design", "plan", "run_id", "protocol_code_commit", "inputs", "module_blobs", "constants", "configuration",
                 "exposed_cells", "confirmation_024", "dependencies", "score", "calibration_cues", "pronoun_cues", "line", "primary_floor", "null", "effective_threshold",
                 "checks", "descriptive", "arrays_sha256", "content_sha256"}
check("record top-level keys are exactly the planned ones", set(record) == expected_keys, sorted(set(record) ^ expected_keys))
for rel, commit in ((record["design"]["path"], record["design"]["commit"]), (record["plan"]["path"], record["plan"]["commit"])):
    anc = git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0
    changed = git("diff", "--name-only", commit, "HEAD", "--", rel).stdout.strip()
    check(f"{rel.split('/')[-1]} at {commit} is an ancestor and unchanged to HEAD", anc and changed == "", changed)

# ---------------------------------------------------------------------------------------------------------------
print("== Item 2: scientific dependency bindings (non-model part) ==")
check("record binds the freeze: path, file and content digests", record["confirmation_024"] == {"path": "experiments/024-readout-routing-nounness/confirmation-v1.json",
                                                                                                  "file_sha256": EXPECT["freeze_file"], "content_sha256": EXPECT["freeze_content"]})
cells_path, index_path = ROOT / "experiments/023-block0-completion/exposed-cells.f64", ROOT / "experiments/023-block0-completion/exposed-cells.json"
cells_bytes = cells_path.read_bytes()
index = json.loads(index_path.read_text(encoding="utf-8"))
check("023 exposed-cells data file sha256 d2ee71e5…", sha256_bytes(cells_bytes) == EXPECT["cells_data"])
check("023 exposed-cells index file sha256 62818147…", sha256_file(index_path) == EXPECT["cells_index_file"])
check("023 index content digest recomputes and is d38305cb…", index["content_sha256"] == content_digest(index) == EXPECT["cells_index_content"])
check("023 index binds the data file digest and the byte layout", index["file_sha256"] == EXPECT["cells_data"] and index["total_bytes"] == len(cells_bytes) == 18900 * 8 * 8
      and index["dtype"] == "<f8" and index["blocks"] == [{"name": "cells", "nbytes": 1209600, "offset_bytes": 0, "shape": [18900, 8]}])
expected_source = {"cells_version": "023-cells-v1", "columns": ["n", "S", "Q", "SSE0", "SSE1", "SSEC", "mean", "M2"], "data_path": "experiments/023-block0-completion/exposed-cells.f64",
                   "data_sha256": EXPECT["cells_data"], "index_content_sha256": EXPECT["cells_index_content"], "index_path": "experiments/023-block0-completion/exposed-cells.json",
                   "index_sha256": EXPECT["cells_index_file"]}
check("record exposed_cells and dependencies.calibration_source bind exactly these", record["exposed_cells"] == expected_source == record["dependencies"]["calibration_source"])
check("023 index columns and version agree", index["columns"] == expected_source["columns"] and index["version"] == "023-cells-v1")
for kind, rel, file_key, content_key in (("confirmation", "experiments/023-block0-completion/confirmation-v1.json", "023_confirmation_file", "023_confirmation_content"),
                                         ("lock", "experiments/023-block0-completion/preregistration-lock.json", "023_lock_file", "023_lock_content")):
    p = ROOT / rel
    obj = json.loads(p.read_text(encoding="utf-8"))
    check(f"023 {kind}: file and content digests on disk equal the record's inputs", sha256_file(p) == record["inputs"][file_key] and obj["content_sha256"] == content_digest(obj) == record["inputs"][content_key])
check("record inputs bind the 023 cells digests", record["inputs"]["023_cells_data_file"] == EXPECT["cells_data"] and record["inputs"]["023_cells_index_file"] == EXPECT["cells_index_file"]
      and record["inputs"]["023_cells_index_content"] == EXPECT["cells_index_content"])
for kind, rel in (("calibration", "experiments/022-upstream-error-localization/calibration-v1.json"), ("confirmation", "experiments/022-upstream-error-localization/confirmation-v1.json")):
    p = ROOT / rel
    obj = json.loads(p.read_text(encoding="utf-8"))
    check(f"022 {kind}: file and content digests on disk equal the record's inputs", sha256_file(p) == record["inputs"][f"{kind}_022_file"]
          and obj["content_sha256"] == content_digest(obj) == record["inputs"][f"{kind}_022_content"])
check("table_022_file is bound as a digest only (the table is never opened here)", record["inputs"]["table_022_file"] == "04659d5e8b20a70526e5862eb2b2de952b33496f25e3875de6c6254df2b8fd81"
      and index["source_022"]["table_file_sha256"] == record["inputs"]["table_022_file"])

# every other bound input digest located independently among the tracked files (file sha256 or self-verifying content digest)
tracked = [line for line in git("ls-files").stdout.splitlines() if line]
by_file, by_content = {}, {}
for rel in tracked:
    p = ROOT / rel
    if not p.is_file():
        continue
    data = p.read_bytes()
    by_file.setdefault(sha256_bytes(data), []).append(rel)
    if rel.endswith(".json") and len(data) < 50_000_000:
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("content_sha256"), str) and obj["content_sha256"] == content_digest(obj):
            by_content.setdefault(obj["content_sha256"], []).append(rel)
located = {}
for key, value in sorted(record["inputs"].items()):
    located[key] = {"file": by_file.get(value, []), "content": by_content.get(value, [])}
unlocated = sorted(k for k, v in located.items() if not v["file"] and not v["content"])
for key, where in sorted(located.items()):
    print(f"    input {key}: {record['inputs'][key][:12]}… file={where['file'][:2]} content={where['content'][:2]}")
check("every bound input digest except 022's local table (never opened) and 020's results state (gz archive, below) is a tracked file's sha256 or self-verifying content digest",
      set(unlocated) <= {"table_022_file", "results_020_state", "results_020_file"}, unlocated)
check("readout_020.frozen_input_digests equals the record's inputs on those keys",
      all(record["inputs"][k] == v for k, v in record["dependencies"]["readout_020"]["frozen_input_digests"].items()) and len(record["dependencies"]["readout_020"]["frozen_input_digests"]) == 21,
      len(record["dependencies"]["readout_020"]["frozen_input_digests"]))
# results_020_state / results_020_file: the gz-archived 020 results state (023's evidence) and the local hash-pinned copy
import gzip  # noqa: E402
archived = gzip.decompress((ROOT / "experiments/023-block0-completion/evidence/archive/experiment-020-results.json.gz").read_bytes())
archived_state = json.loads(archived)
archived_digest = sha256_bytes(cjson({k: v for k, v in archived_state.items() if k != "state_sha256"}).encode("utf-8"))
local_020 = ROOT / "outputs/experiment-020/results.json"
closure_020 = json.loads((ROOT / "experiments/020-readout-decompilation/closure.json").read_text(encoding="utf-8"))
check("results_020_file = sha256 of the archived (gz) and the local 020 results state; results_020_state its self-verifying digest; both equal 020's closure",
      sha256_bytes(archived) == record["inputs"]["results_020_file"] == closure_020["explore"]["results_file_sha256"] and local_020.read_bytes() == archived
      and archived_digest == archived_state["state_sha256"] == record["inputs"]["results_020_state"] == closure_020["explore"]["results_state_sha256"],
      (sha256_bytes(archived)[:12], archived_digest[:12]))

# module blobs: my own sha1 of b"blob <len>\0" + bytes, against the record, the state and git's tree at HEAD
names = sorted(record["module_blobs"])
own_blobs = {name: blob_sha1(ROOT / "src/neural_decompiler" / name) for name in names}
tree = {}
for line in git("ls-tree", "HEAD", "src/neural_decompiler/").stdout.splitlines():
    meta, path = line.split("\t")
    tree[path.split("/")[-1]] = meta.split()[2]
check("twelve pinned module blobs", len(names) == 12, names)
check("own blob sha1 == record module_blobs == dependencies.module_blobs == state module_blobs == git tree at HEAD",
      all(own_blobs[n] == record["module_blobs"][n] == record["dependencies"]["module_blobs"][n] == state["module_blobs"][n] == tree[n] for n in names),
      {n: own_blobs[n][:10] for n in names})
check("readout_020 pins readout_decompilation.py at caa73b40…", record["dependencies"]["readout_020"]["module"] == "readout_decompilation.py"
      and record["dependencies"]["readout_020"]["blob"] == own_blobs["readout_decompilation.py"] == "caa73b40192f4c910dc63371bd19db75a3258339")
own_rr = blob_sha1(ROOT / "src/neural_decompiler/readout_routing.py")
check("readout_routing.py on disk is HEAD's blob (the code that ran at bf0049c)", own_rr == tree["readout_routing.py"], own_rr)
runner_blob = blob_sha1(ROOT / "experiments/024-readout-routing-nounness/run.py")
runner_tree = git("ls-tree", "HEAD", "experiments/024-readout-routing-nounness/run.py").stdout.split()[2]
check("run.py on disk is HEAD's blob", runner_blob == runner_tree, runner_blob)

# ---------------------------------------------------------------------------------------------------------------
print("== Item 4: exposed per-cue MSE from the committed artifact alone ==")
raw = np.frombuffer(cells_bytes, dtype="<f8").reshape(18900, 8)
check("artifact decodes as [18900, 8] little-endian float64, all finite", raw.shape == (18900, 8) and bool(np.isfinite(raw).all()))
cues_all = [tuple(c) for c in index["cues"]]
frames = [tuple(f) for f in index["frames"]]
check("index: 175 cues, 108 frames (72 cue-final, 36 coordinated), 79 nouns", len(cues_all) == 175 and len(frames) == 108 and len(index["nouns"]) == 79
      and sum(1 for f in frames if f[2] == "cue_final") == 72 and sum(1 for f in frames if f[2] == "coordinated") == 36)
check("frames are in frame_id order", [f[0] for f in frames] == sorted(f[0] for f in frames))
strata_seq = []
for _, _, s in cues_all:
    if not strata_seq or strata_seq[-1] != s:
        strata_seq.append(s)
ul_src = (ROOT / "src/neural_decompiler/upstream_localization.py").read_text(encoding="utf-8")
ul_classes = re.findall(r'"([a-z\-]+)": \(', re.search(r"CUE_CANDIDATES = \{.*?\n\}", ul_src, re.S).group(0))
check("cue order: classes contiguous in 022's order (ul.CUE_CANDIDATES), token id ascending within a class",
      strata_seq == ul_classes == ["determiner-like", "quantity", "possessive-or-pronoun", "adjective"]
      and all([t for _, t, s in cues_all if s == st] == sorted(t for _, t, s in cues_all if s == st) for st in strata_seq), (strata_seq, ul_classes))
check("175 distinct token ids", len({t for _, t, _ in cues_all}) == 175)
col = {name: i for i, name in enumerate(index["columns"])}
check("every exposed cell counts 79 nouns", bool((raw[:, col["n"]] == 79.0).all()))
own_mse_fsum, own_mse_exact, own_mse_numpy = [], [], []
for ci in range(175):
    rows = raw[ci * 108:(ci + 1) * 108]
    ssec, n = rows[:, col["SSEC"]].tolist(), rows[:, col["n"]].tolist()
    own_mse_fsum.append(math.fsum(ssec) / math.fsum(n))
    own_mse_exact.append(float(sum(Fraction(v) for v in ssec) / sum(Fraction(v) for v in n)))
    own_mse_numpy.append(float(np.sum(rows[:, col["SSEC"]]) / np.sum(rows[:, col["n"]])))
cal_idx = [i for i, c in enumerate(cues_all) if c[2] in STRATA]
pron_idx = [i for i, c in enumerate(cues_all) if c[2] == "possessive-or-pronoun"]
check("139 calibration cues (45/45/49), 36 pronoun cues", len(cal_idx) == 139 and len(pron_idx) == 36
      and [sum(1 for i in cal_idx if cues_all[i][2] == s) for s in STRATA] == [45, 45, 49])
entries = record["calibration_cues"]
check("record: 139 entries, in the artifact's canonical order (word, token id, stratum), no pronoun",
      [(e["word"], e["token_id"], e["stratum"]) for e in entries] == [cues_all[i] for i in cal_idx] and all(e["stratum"] in STRATA for e in entries))
check("record score.calibration_cue_ids equal the entries' token ids", record["score"]["calibration_cue_ids"] == [e["token_id"] for e in entries])
d_fsum = max(abs(e["mse"] - own_mse_fsum[i]) for e, i in zip(entries, cal_idx))
eq_fsum = sum(1 for e, i in zip(entries, cal_idx) if e["mse"] == own_mse_fsum[i])
d_exact = max(abs(e["mse"] - own_mse_exact[i]) / own_mse_exact[i] for e, i in zip(entries, cal_idx))
eq_exact = sum(1 for e, i in zip(entries, cal_idx) if e["mse"] == own_mse_exact[i])
d_numpy = max(abs(e["mse"] - own_mse_numpy[i]) / own_mse_numpy[i] for e, i in zip(entries, cal_idx))
check("record MSE == own fsum MSE bit for bit (all 139)", eq_fsum == 139, f"max |Δ| {d_fsum:.3e}, bitwise equal {eq_fsum}/139")
check("record MSE vs exactly-rounded rational MSE", d_exact <= 1e-15, f"max rel {d_exact:.3e}, bitwise equal {eq_exact}/139")
check("record MSE vs numpy pairwise-sum MSE (numerical disagreement only)", d_numpy <= 1e-12, f"max rel {d_numpy:.3e}")
check("record log_mse == math.log(record mse) bit for bit", all(e["log_mse"] == math.log(e["mse"]) for e in entries))
check("every calibration MSE finite and positive", all(math.isfinite(e["mse"]) and e["mse"] > 0 for e in entries), f"min {min(e['mse'] for e in entries)!r} max {max(e['mse'] for e in entries)!r}")
check("139 calibration MSE values distinct (no rank ties between distinct cues)", len({e["mse"] for e in entries}) == 139)
pron = record["pronoun_cues"]
check("record pronoun cues: the 36 in canonical order, MSE bit-identical to own fsum", [(p["word"], p["token_id"]) for p in pron] == [cues_all[i][:2] for i in pron_idx]
      and all(p["mse"] == own_mse_fsum[i] for p, i in zip(pron, pron_idx)))
check("the calibration population excludes every pronoun id", not ({p["token_id"] for p in pron} & set(record["score"]["calibration_cue_ids"])))
json.dump({"cal_cues": [list(cues_all[i]) for i in cal_idx], "mse_fsum": [own_mse_fsum[i] for i in cal_idx], "mse_exact": [own_mse_exact[i] for i in cal_idx],
           "pron_cues": [list(cues_all[i]) for i in pron_idx], "pron_mse_fsum": [own_mse_fsum[i] for i in pron_idx]},
          open(SCRATCH / "own_mse.json", "w"))

# ---------------------------------------------------------------------------------------------------------------
print("== Item 5 (static part): rounded prose values are not code inputs ==")
for rel in ("src/neural_decompiler/readout_routing.py", "experiments/024-readout-routing-nounness/run.py"):
    text = (ROOT / rel).read_text(encoding="utf-8")
    hits = {pattern: [m.start() for m in re.finditer(pattern, text)] for pattern in (r"1\.299", r"2\.481", r"0\.257", r"0\.315", r"0\.53\d?\b", r"0\.135", r"99\.5|0\.995|99_?500")}
    check(f"{rel}: no rounded design diagnostic or 99.5 % literal", not any(hits.values()), {k: v for k, v in hits.items() if v})
line = record["line"]
check("record line is full precision (not the rounded prose)", line["slope"] != 1.299 and line["intercept"] != -2.481 and line["residual_sd"] != 0.257, line)

# ---------------------------------------------------------------------------------------------------------------
print("== Item 7 (static part): no 99.5 % value in the record ==")
record_text = record_bytes.decode("utf-8")
check("record text has no 99.5 / 0.995 / 99500 element", not re.search(r"99\.5|0\.995|99500|99_500", record_text))
check("record null element is [97499] of 100000 (rank 97500), tag 024|null", (record["null"]["element"], record["null"]["rank"], record["null"]["permutations"], record["null"]["n"], record["null"]["tag"])
      == (97499, 97500, 100000, 40, "024|null"))
check("record floor element is [249] of 10000 (rank 250), stop at 250, tag 024|primary", (record["primary_floor"]["element"], record["primary_floor"]["rank"], record["primary_floor"]["draws"],
                                                                                             record["primary_floor"]["stop_at"], record["primary_floor"]["tag"]) == (249, 250, 10000, 250, "024|primary"))

# ---------------------------------------------------------------------------------------------------------------
print("== Item 9: the E–N guard is bound, not evaluated ==")
guard = record["configuration"]["guard"]
check("guard spec 8 + 8, C(16,8) = 12,870 assignments, max_upper 321 = ⌊0.025·12,870⌋", (guard["n_E"], guard["n_N"], guard["assignments"], guard["max_upper"]) == (8, 8, 12870, 321)
      and math.comb(16, 8) == 12870 and (25 * 12870) // 1000 == 321 and Fraction(321, 12870) <= Fraction(1, 40) < Fraction(322, 12870))
check("guard rule text: K counts ties and the observed; PASS iff K ≤ max_upper; exact dyadic arithmetic; natural log; E first",
      "ties count against the guard" in guard["rule"] and "PASS iff K ≤ max_upper" in guard["rule"] and "the observed one included" in guard["rule"]
      and "exact" in guard["arithmetic"] and "as_integer_ratio" in guard["arithmetic"] and guard["log"] == "natural" and guard["order"].startswith("E first"))
check("record constants carry the same configuration", record["constants"]["configuration"] == record["configuration"])


def all_keys(value, acc):
    if isinstance(value, dict):
        for k, v in value.items():
            acc.add(k)
            all_keys(v, acc)
    elif isinstance(value, list):
        for v in value:
            all_keys(v, acc)
    return acc


keys = all_keys(record, set())
check("no guard evaluation in the record (no K, D_EN, t, exact p, groups, guard result)", not (keys & {"K", "D_EN", "D_EN_exact", "threshold_D", "p_exact", "groups", "result"}),
      sorted(keys & {"K", "D_EN", "D_EN_exact", "threshold_D", "p_exact", "groups", "result"}))
check("the only 'p_value' fields are the guard spec's rule text", record["configuration"]["guard"]["p_value"] == record["constants"]["configuration"]["guard"]["p_value"]
      == "K / assignments (exact)" and record_text.count('"p_value"') == 2)

# ---------------------------------------------------------------------------------------------------------------
print("== Item 10 (record part) and item 11: fresh-score boundary, no fresh measurement ==")
fresh = freeze["cues"]
fresh_ids = [int(c["token_id"]) for c in fresh]
check("freeze: 40 distinct fresh cues, classes N,B,D,C,E × 8 in that order", len(set(fresh_ids)) == 40 and [c["class"] for c in fresh] == [k for k in "NBDCE" for _ in range(8)])
check("no fresh id among the calibration cues, the pronoun cues or the noun rows (a fresh cue is never in μ_cue)",
      not (set(fresh_ids) & (set(record["score"]["calibration_cue_ids"]) | {p["token_id"] for p in pron} | set(record["score"]["noun_row_ids"]))))
check("no fresh id among the 175 exposed artifact cues", not (set(fresh_ids) & {t for _, t, _ in cues_all}))
check("record binds no fresh score (no 'fresh' key, no scores digest, no a703ac16…)", "fresh" not in keys and "scores_sha256" not in keys and "a703ac16" not in record_text)
check("record has no per-fresh-cue entry (no fresh word as an entry word)", not ({c["word"] for c in fresh} & {e["word"] for e in entries + pron}))
for name in ("stage2-measurements.pt", "candidate-lock.json", "candidate-preregistration.md", "report.md"):
    check(f"no {name} in outputs/experiment-024", not (OUT / name).exists())
for rel in ("experiments/024-readout-routing-nounness/calibration-v1.json", "experiments/024-readout-routing-nounness/preregistration-lock.json",
            "experiments/024-readout-routing-nounness/preregistration.md"):
    check(f"{rel} not installed yet", not (ROOT / rel).exists())
manifest = freeze["manifest"]["S2-TARGET"]
check("freeze manifest: 4,320 unique S2-TARGET keys; none in the state's ledger", len(manifest) == len(set(manifest)) == 4320 and not (set(manifest) & set(state["executed_prompt_keys"])))
state_text = state_bytes.decode("utf-8")
check("no manifest key appears anywhere in the state or the record text", not any(k in state_text or k in record_text for k in manifest))
print("guard events:", rguard.EVENTS)
check("review guard: no repository write attempted, no forbidden read attempted", not rguard.EVENTS["refused_writes"] and not rguard.EVENTS["refused_reads"])
json.dump(results, open(SCRATCH / "review_state.json", "w"), indent=1)
print("FAILURES:", results["failures"] or "none")
