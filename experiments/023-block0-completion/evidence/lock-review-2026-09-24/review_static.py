"""Independent lock review, no-model part (items 1, 2, 5, 6, 8; the calibration floors from the draws and from the
committed exposed cells). Experiment 022's table is blocked (proven live). No model is loaded.

For items 2 and 6 the expected lock is rebuilt field by field from the committed inputs with my own code; the only b0c
objects used are constants (STATISTIC, SCOPE, SEMANTICS, GUARD, TABLE_CONSTRUCTION, DESIGN, PLAN, path constants, the
string literals of y2_table_spec / build_lock read through ``ast``), ``b0c.scientific_changes`` (item 1, allowed) and
``b0c.render_preregistration`` (text-only check, allowed).
"""
import sys
SCRATCH = "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/lockreview023"
sys.path.insert(0, SCRATCH)
import guard  # noqa: E402  (first)

import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = guard.ROOT
OUT = Path(SCRATCH)
PROOF = guard.prove_live()
RESULTS: list[tuple[str, bool, str]] = []
DATA: dict = {"guard_proof": PROOF}
HEAD_EXPECTED = "13b8d392f9e88844e2673c530a37228337cc64b7"
CAL_COMMIT = "4e5deebd021bc284143b01b76b45efd28e1a1344"
EXTRACT_COMMIT = "2111271892a8237f69c3ef18d6eec774e798ac5c"
E23 = "experiments/023-block0-completion"
O23 = ROOT / "outputs/experiment-023"


def check(name: str, ok: bool, detail="") -> bool:
    RESULTS.append((name, bool(ok), str(detail)[:600]))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {str(detail)[:600]}", flush=True)
    return bool(ok)


def cj(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path) -> str:
    return sha(Path(path).read_bytes())


def content_digest(payload: dict) -> str:
    return sha(cj({k: v for k, v in payload.items() if k != "content_sha256"}).encode("utf-8"))


def blob_sha1(path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    """My reimplementation of readout_calibration.tensor_digest: sha256(json(shape) | kind | little-endian bytes)."""
    array = tensor.detach().cpu().contiguous()
    kind = "<i8" if array.dtype in (torch.int64, torch.int32) else "<f8"
    data = np.asarray(array.numpy()).astype(kind).tobytes()
    return sha(json.dumps(list(array.shape)).encode("ascii") + b"|" + kind.encode("ascii") + b"|" + data)


def git(*args, binary=False):
    out = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode("utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# =====================================================================================================================
print("== item 1: provenance", flush=True)
head = git("rev-parse", "HEAD").strip()
remote = git("ls-remote", "origin", "refs/heads/main").split()
check("1 HEAD == 13b8d39…", head == HEAD_EXPECTED, head)
check("1 origin/main (ls-remote) == HEAD", bool(remote) and remote[0] == head, remote[:1])
status = git("status", "--porcelain", "--untracked-files=all")
check("1 clean tree incl. untracked", status == "", repr(status[:200]))
check("1 parents: 13b8d39^ = 51b5c7d, 51b5c7d^ = 4e5deeb (calibrate commit)",
      git("rev-parse", "13b8d39^").strip().startswith("51b5c7d") and git("rev-parse", "51b5c7d^").strip() == CAL_COMMIT)
files_51 = [line for line in git("show", "--name-only", "--format=", "51b5c7d").splitlines() if line.strip()]
check("1 calibration-v1.json committed alone in 51b5c7d", files_51 == [f"{E23}/calibration-v1.json"], files_51)
committed_cal = git("show", f"51b5c7d:{E23}/calibration-v1.json", binary=True)
head_cal = git("show", f"HEAD:{E23}/calibration-v1.json", binary=True)
cand_cal = (O23 / "candidate-calibration.json").read_bytes()
tree_cal = (ROOT / E23 / "calibration-v1.json").read_bytes()
check("1 committed calibration-v1.json (51b5c7d, HEAD, tree) byte-identical to candidate-calibration.json",
      committed_cal == cand_cal == head_cal == tree_cal, sha(cand_cal))

state_path = O23 / "results.json"
state_text = state_path.read_text(encoding="utf-8")
state = json.loads(state_text)
my_state_sha = sha(cj({k: v for k, v in state.items() if k != "state_sha256"}).encode("utf-8"))
check("1 results.json state_sha256 recomputed (own canonical JSON)", my_state_sha == state["state_sha256"], my_state_sha)
check("1 results.json file is canonical JSON + newline", state_text == cj(state) + "\n", sha_file(state_path))
phases = state["phases"]
check("1 extract complete once at 2111271, no incident", phases["extract"]["status"] == "complete" and phases["extract"]["commit"] == EXTRACT_COMMIT
      and "incidents" not in phases["extract"], {k: phases["extract"][k] for k in ("status", "commit")})
check("1 calibrate complete once at 4e5deeb, no incident/stop", phases["calibrate"]["status"] == "complete" and phases["calibrate"]["commit"] == CAL_COMMIT
      and "incidents" not in phases["calibrate"] and "incidents" not in state["calibration"] and "stop" not in state["calibration"], phases["calibrate"])
check("1 lock complete once at 13b8d39, no incident", phases["lock"]["status"] == "complete" and phases["lock"]["commit"] == HEAD_EXPECTED
      and set(phases["lock"]) == {"status", "completed_at", "commit"}, phases["lock"])
check("1 confirm and report not_started", phases["confirm"] == {"status": "not_started"} and phases["report"] == {"status": "not_started"}, {k: phases[k] for k in ("confirm", "report")})
lock_path, prereg_path = O23 / "candidate-lock.json", O23 / "candidate-preregistration.md"
y1_path, y1_index_path = O23 / "candidate-locked-y1-table.f64", O23 / "candidate-locked-y1-table.json"
lock_text = lock_path.read_text(encoding="utf-8")
lock = json.loads(lock_text)
prereg_bytes = prereg_path.read_bytes()
sl = state["lock"]
record_path = ROOT / E23 / "calibration-v1.json"
record = load(record_path)
check("1 state lock entry digests == the candidate files", sl["content_sha256"] == content_digest(lock) == lock["content_sha256"]
      and sl["preregistration_sha256"] == sha(prereg_bytes) and sl["y1_file_sha256"] == sha_file(y1_path) and sl["y1_index_sha256"] == sha_file(y1_index_path)
      and sl["calibration_content_sha256"] == content_digest(record)
      and [sl[k] for k in ("candidate_path", "preregistration_path", "y1_data_path", "y1_index_path")] == [str(p) for p in (lock_path, prereg_path, y1_path, y1_index_path)],
      {k: sl[k] for k in ("content_sha256", "preregistration_sha256", "y1_file_sha256", "y1_index_sha256")})
check("1 state lock gates == candidate lock gates", sl["gates"] == lock["y1_table"]["gates"], sl["gates"])
check("1 git merge-base: calibrate commit is an ancestor of HEAD", subprocess.run(["git", "--no-optional-locks", "merge-base", "--is-ancestor", CAL_COMMIT, "HEAD"], cwd=ROOT).returncode == 0)
changed = [line for line in git("diff", "--no-renames", "--name-only", CAL_COMMIT, "HEAD").splitlines() if line.strip()]
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler import readout_calibration as rc  # noqa: E402
check("1 no scientific path changed since calibrate (b0c.scientific_changes)", b0c.scientific_changes(changed) == [], {"changed": changed, "scientific": b0c.scientific_changes(changed)})

# =====================================================================================================================
print("== item 2 / 6: lock reconstruction", flush=True)
inputs = ul.load_frozen_inputs(ROOT)
# --- floors: from the record, and element [249] of the draws (digests verified with my own tensor digest)
arrays = torch.load(O23 / "draw-values.pt")
my_array_digests = {key: tensor_digest(value) for key, value in arrays.items()}
check("2 draw-values.pt: every tensor digest (own reimplementation) == the record's draw_arrays_sha256", my_array_digests == record["draw_arrays_sha256"],
      f"{len(my_array_digests)} arrays")
CONDITIONS = ["Y1/cue_final", "Y1/coordinated", "Y2/cue_final", "Y2/coordinated"]
floors = {}
for c in CONDITIONS:
    g = arrays[f"{c}/g"].double().numpy()
    defined = arrays[f"{c}/defined"].bool().numpy()
    placed = np.where(defined, g, -np.inf)
    f_np = float(np.sort(placed, kind="stable")[249])
    f_py = sorted(float(v) for v in placed)[249]
    floors[c] = f_np
    bound = record["conditions"][c]["envelope"]["bound"]
    check(f"2 floor {c}: element [249] of the ascending 10,000 (undefined at -inf) == record == lock", f_np == f_py == bound == lock["conditions"][c]["envelope"]["bound"],
          f"F {f_np!r}; undefined {int((~defined).sum())}; n {g.size}")
DATA["floors"] = floors

# --- extra: the whole calibration from the committed exposed cells (own draws, own pooling)
cells_index = load(ROOT / E23 / "exposed-cells.json")
cells_bytes = (ROOT / E23 / "exposed-cells.f64").read_bytes()
check("2x committed exposed cells: bytes == index file_sha256; index content digest verifies", sha(cells_bytes) == cells_index["file_sha256"] and content_digest(cells_index) == cells_index["content_sha256"])
cells = np.frombuffer(cells_bytes, dtype="<f8").reshape(len(cells_index["cues"]) * len(cells_index["frames"]), 8)
COORD = "coordinated-adjective"
STRATA = ["determiner-like", "quantity", "adjective"]
TEMPLATES = ["cardinal", "quantifier", COORD]
cues = cells_index["cues"]
frames_x = cells_index["frames"]
NF = len(frames_x)
stratum_cues = {s: [i for i, c in enumerate(cues) if c[2] == s] for s in STRATA}
check("2x exposed cue order: token id within each stratum", all([cues[i][1] for i in idx] == sorted(cues[i][1] for i in idx) for idx in stratum_cues.values()),
      {s: len(v) for s, v in stratum_cues.items()})
fid = {fr[0]: i for i, fr in enumerate(frames_x)}
y2_like = {t: [fid[f] for f in cells_index["y2_like_frames"][t]] for t in TEMPLATES}
check("2x Y2-like frames by frame_id, 14 per template", all(cells_index["y2_like_frames"][t] == sorted(cells_index["y2_like_frames"][t]) and len(y2_like[t]) == 14 for t in TEMPLATES))
group_frames = {g: [i for i, fr in enumerate(frames_x) if (fr[1] == COORD) == (g == "coordinated")] for g in ("cue_final", "coordinated")}
B = 10_000
slots = {**{f"cue/{s}": 8 for s in STRATA}, **{f"frame/{t}": 6 for t in TEMPLATES}}
sizes = {**{f"cue/{s}": len(stratum_cues[s]) for s in STRATA}, **{f"frame/{t}": len(y2_like[t]) for t in TEMPLATES}}
draw_idx = {k: np.array([[int.from_bytes(hashlib.sha256(f"023|primary|{b}|{k}|{i}".encode("utf-8")).digest()[:8], "big") % sizes[k] for i in range(slots[k])]
                         for b in range(B)], dtype=np.int64) for k in slots}
my_idx_digests = {k: tensor_digest(torch.from_numpy(v)) for k, v in sorted(draw_idx.items())}
check("2x draw indices (own SHA draws) == the record's index digests", my_idx_digests == record["draws"]["index_sha256"])
cue_pick = np.concatenate([np.array(stratum_cues[s])[draw_idx[f"cue/{s}"]] for s in STRATA], axis=1)  # [B, 24]
my_g = {}
for pop in ("Y1", "Y2"):
    for group in ("cue_final", "coordinated"):
        if pop == "Y1":
            fr = np.broadcast_to(np.array(group_frames[group]), (B, len(group_frames[group])))
        else:
            fr = np.concatenate([np.array(y2_like[t])[draw_idx[f"frame/{t}"]] for t in TEMPLATES if (t == COORD) == (group == "coordinated")], axis=1)
        sums = np.zeros((B, 6))
        for start in range(0, B, 1000):
            pairs_ = (cue_pick[start:start + 1000, :, None] * NF + fr[start:start + 1000, None, :]).reshape(min(1000, B - start), -1)
            sums[start:start + 1000] = cells[pairs_][:, :, :6].sum(axis=1)
        n, s_, q, sse0, sse1, ssec = sums.T
        sst = q - s_ * s_ / n
        defined = (sst > 0) & (sse0 - ssec >= 0.02 * sst)
        g = np.where(defined, (sse0 - sse1) / np.where(defined, sse0 - ssec, 1.0), np.nan)
        c = f"{pop}/{group}"
        ref = arrays[f"{c}/g"].double().numpy()
        same_defined = bool(np.array_equal(defined, arrays[f"{c}/defined"].bool().numpy()))
        diff = float(np.nanmax(np.abs(g - ref)))
        f_mine = float(np.sort(np.where(defined, g, -np.inf))[249])
        my_g[c] = {"max_abs_diff_vs_draw_values": diff, "defined_equal": same_defined, "F_from_cells": f_mine, "F_diff": abs(f_mine - floors[c])}
        check(f"2x {c}: g of all 10,000 draws recomputed from the committed cells", same_defined and diff < 1e-12 and abs(f_mine - floors[c]) < 1e-12, my_g[c])
DATA["calibration_from_cells"] = my_g
# rates and the joint share (descriptive) from the draws
passes = []
for c in CONDITIONS:
    g = arrays[f"{c}/g"].double().numpy()
    defined = arrays[f"{c}/defined"].bool().numpy()
    F = floors[c]
    codes = np.where(~defined, 0, np.where(g < 0.90, 1, np.where(g < F, 2, 3)))
    rates = {name: float(np.mean(codes == k)) for k, name in enumerate(["NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS"])}
    passes.append(codes == 3)
    check(f"6 {c}: result rates recomputed == record", rates == record["conditions"][c]["rates"], rates)
joint = float(np.mean(np.all(np.stack(passes), axis=0)))
check("6 joint all-four PASS share recomputed == record 0.9178 (descriptive only)", joint == record["joint_rates"]["all_four_pass"] == 0.9178 and record["joint_rates"]["descriptive_only"] is True, joint)

# --- masks
BIT = ul.BIT
P1 = {"cue_final": BIT["emb"] | BIT["Bv"] | BIT["Bp"], "coordinated": BIT["emb"] | BIT["Bv"] | BIT["Bp"] | BIT["T"]}
check("2 masks from ul.BIT: P0 = 0, P1 = 14 / 30 (R off)", BIT == {"R": 1, "emb": 2, "Bv": 4, "Bp": 8, "T": 16} and P1 == {"cue_final": 14, "coordinated": 30}
      and lock["program"]["P0_mask"] == 0 and lock["program"]["P1_mask"] == P1 and lock["constants"]["p0_mask"] == 0 and lock["constants"]["p1_mask"] == P1
      and lock["y2_table"]["meta"]["masks"] == {"P0": 0, "P1": P1}, P1)
# --- the 79 nouns
noun_keys = [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]
y1_index = load(y1_index_path)
check("2 79-noun order: pool single-token nouns == lock == Y1 index == Y2 meta == exposed-cells index", len(noun_keys) == 79 and noun_keys == lock["noun_keys"]
      == y1_index["nouns"] == lock["y2_table"]["meta"]["nouns"] == cells_index["nouns"], noun_keys[:4])
# --- exposed states digest (read directly from Experiment 020's results file)
r020 = load(ROOT / "outputs/experiment-020/results.json")
locked_states = r020["exploration"]["locked_states"]
per_frame = {frame_id: sha(cj(entry).encode("utf-8")) for frame_id, entry in sorted(locked_states.items())}
exposed_digest = sha(cj(per_frame).encode("utf-8"))
extract_020 = load(ROOT / "experiments/020-readout-decompilation/evidence/exploration-record-2026-09-22.json")
check("2 exposed-state digest (own) == lock; per-frame digests == 020's committed extract", exposed_digest == lock["exposed_states_sha256"]
      and per_frame == extract_020["exploration"]["locked_state_digests"] and len(per_frame) == 108, exposed_digest)
# --- the confirmation binding
conf_path = ROOT / E23 / "confirmation-v1.json"
conf = load(conf_path)
counts = {"classes": {s: sum(1 for c in conf["cues"] if c["class"] == s) for s in STRATA}, "templates": {t: sum(1 for f in conf["frames"] if f["template_id"] == t) for t in TEMPLATES}}
sizes_m = {"S1-REF": len(conf["frames"]), "S1-VALIDITY": len(conf["frames"]), "Y1": len(conf["cues"]) * len(conf["exposed_frame_ids"]), "Y2": len(conf["cues"]) * len(conf["frames"])}
expected_conf = {"path": f"{E23}/confirmation-v1.json", "file_sha256": sha_file(conf_path), "content_sha256": content_digest(conf), "counts": counts, "manifest_sizes": sizes_m}
check("2 confirmation binding (path, file, content, counts 8/8/8 + 6/6/6, sizes 18/18/2592/432) == lock", expected_conf == lock["confirmation_023"]
      and conf["content_sha256"] == expected_conf["content_sha256"] and sizes_m == {"S1-REF": 18, "S1-VALIDITY": 18, "Y1": 2592, "Y2": 432}
      and counts == {"classes": {s: 8 for s in STRATA}, "templates": {t: 6 for t in TEMPLATES}}
      and [len(conf["manifest"]["S1-REF"]), len(conf["manifest"]["S1-VALIDITY"]), len(conf["manifest"]["S2-TARGET"]["Y1"]), len(conf["manifest"]["S2-TARGET"]["Y2"])] == [18, 18, 2592, 432]
      and {k: expected_conf[k] for k in ("path", "file_sha256", "content_sha256")} == state["confirmation_023"] == record["confirmation_023"], expected_conf["file_sha256"])
# --- the calibration binding and the exposed-cells binding
expected_cal = {"path": f"{E23}/calibration-v1.json", "file_sha256": sha_file(record_path), "content_sha256": content_digest(record)}
check("2 calibration binding (path, file 94db6df2…, content a391500c…) == lock", expected_cal == lock["calibration"] and record["content_sha256"] == expected_cal["content_sha256"], expected_cal)
expected_cells = {"data_path": f"{E23}/exposed-cells.f64", "index_path": f"{E23}/exposed-cells.json", "data_sha256": sha(cells_bytes), "index_sha256": sha_file(ROOT / E23 / "exposed-cells.json")}
check("2 exposed-cells binding == lock == record == state", expected_cells == lock["exposed_cells"] == record["exposed_cells"] == state["calibration"]["exposed_cells"], expected_cells)
# --- the Y2 specification
src = Path(b0c.__file__).read_text(encoding="utf-8")
tree = ast.parse(src)


def dict_literals(function_name: str) -> dict:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    out = {}
                    for key, value in zip(sub.keys, sub.values):
                        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                            out[key.value] = value.value
                    if out:
                        return out
    return {}


y2_literals = dict_literals("y2_table_spec")
kind_literal = dict_literals("build_lock").get("kind")
conf_tokens = sorted(({"word": c["word"], "token_id": int(c["token_id"])} for c in conf["cues"]), key=lambda c: c["token_id"])
conf_frames = sorted(conf["frames"], key=lambda f: f["frame_id"])
y2_pairs = {g: [[t["word"], t["token_id"], f["frame_id"]] for t in conf_tokens for f in conf_frames if (f["template_id"] == COORD) == (g == "coordinated")] for g in ("cue_final", "coordinated")}
DESIGN = {"path": "docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md", "revision": 2, "commit": "5b38aba"}
PLAN = {"path": "docs/superpowers/plans/2026-09-24-experiment-023-block0-completion-plan.md", "revision": 1, "commit": "3a795fb"}
check("2 design/plan identities (5b38aba = design rev. 2, 3a795fb = plan rev. 1) == b0c constants", DESIGN == b0c.DESIGN and PLAN == b0c.PLAN
      and "Revision 2" in git("show", f"5b38aba:{DESIGN['path']}")[:400] and "Plan revision 1" in git("show", f"3a795fb:{PLAN['path']}")[:600])
expected_y2 = {"data_path": "outputs/experiment-023/y2-table.f64", "index_path": "outputs/experiment-023/y2-table.json", "format": ul.TABLE_FORMAT, "dtype": "<f8",
               "layout": [{"name": "cue_final", "shape": [len(y2_pairs["cue_final"]), 2, 79]}, {"name": "coordinated", "shape": [len(y2_pairs["coordinated"]), 2, 79]}],
               "meta": {"experiment": "023", "kind": "Y2", "schema_version": 1, "construction": b0c.TABLE_CONSTRUCTION, "design": DESIGN, "pair_order": "cue token id, then frame_id",
                        "pairs": y2_pairs, "columns": ["P0", "P1"], "masks": {"P0": 0, "P1": P1}, "nouns": noun_keys},
               "states": y2_literals.get("states"), "freeze": y2_literals.get("freeze")}
check("2 Y2 specification (layout [288,2,79] + [144,2,79], pair order, nouns, construction) == lock", expected_y2 == lock["y2_table"]
      and expected_y2["layout"][0]["shape"] == [288, 2, 79] and expected_y2["layout"][1]["shape"] == [144, 2, 79],
      {k: expected_y2[k] == lock["y2_table"].get(k) for k in expected_y2})
# --- the inputs digests
d022 = {}
for kind_, rel in (("calibration", "experiments/022-upstream-error-localization/calibration-v1.json"), ("confirmation", "experiments/022-upstream-error-localization/confirmation-v1.json")):
    payload = load(ROOT / rel)
    d022[f"{kind_}_022_file"] = sha_file(ROOT / rel)
    d022[f"{kind_}_022_content"] = content_digest(payload)
    check(f"2 022 {kind_} content digest (own) == its stored content_sha256", payload["content_sha256"] == d022[f"{kind_}_022_content"])
rec022 = load(ROOT / "experiments/022-upstream-error-localization/evidence/confirmation-record-2026-09-24.json")
table_digests_022 = {m for m in json.dumps(rec022).split('"') if len(m) == 64 and m.startswith("04659d5e")}
d022["table_022_file"] = b0c.INHERITED_022["table_file_sha256"]
check("2 table_022_file (not hashed: the table stays blocked) == 022's committed confirmation record == the committed cells index", table_digests_022 == {d022["table_022_file"]}
      and cells_index["source_022"]["table_file_sha256"] == d022["table_022_file"], d022["table_022_file"])
expected_inputs = {**{k: inputs.digests[k] for k in rc.DIGEST_KEYS}, **d022}
own = {"results_020_file": sha_file(ROOT / "outputs/experiment-020/results.json"),
       "results_020_state": sha(cj({k: v for k, v in r020.items() if k != "state_sha256"}).encode("utf-8")),
       "closure_020": content_digest(load(ROOT / "experiments/020-readout-decompilation/closure.json")), "extract_020": content_digest(extract_020),
       "confirmation_020": content_digest(load(ROOT / "experiments/020-readout-decompilation/confirmation-v1.json")),
       "manifest": load(ROOT / "screening/behavior-candidates/manifest-v1.json")["content_sha256"]}
for key, rel in (("lock_011", "experiments/011-encoding-read-prospective/preregistration-lock.json"), ("lock_012", "experiments/012-layer-correction-token-local/preregistration-lock.json"),
                 ("lock_017", "experiments/017-transport-head-pattern/preregistration-lock.json")):
    payload = load(ROOT / rel)
    own[key] = content_digest(payload) if payload["content_sha256"] == content_digest(payload) else "MISMATCH"
own_ok = {k: own[k] == expected_inputs[k] for k in own}
check("2 inputs == state == record == frozen-input loader + own 022 digests; own recomputation of 020/lock/manifest digests agrees",
      lock["inputs"] == expected_inputs == state["inputs"] == record["inputs"] and all(own_ok.values()), own_ok)
# --- module blobs
blobs = {name: blob_sha1(ROOT / "src/neural_decompiler" / name) for name in b0c.FROZEN_BLOBS}
git_blobs = {name: git("rev-parse", f"HEAD:src/neural_decompiler/{name}").strip() for name in b0c.FROZEN_BLOBS}
check("2 module blobs (own sha1 = git HEAD blobs) == lock == record == state; ul = 465856962a…", blobs == git_blobs == lock["module_blobs"] == record["module_blobs"] == state["module_blobs"]
      and len(blobs) == 11 and blobs["upstream_localization.py"] == "465856962aa380747d1a4f1338d1d2762d03c9f9", len(blobs))
# --- constants (from the design / plan)
expected_constants = {"B": 10000, "draw_tag": "023|primary", "cross_check_draws": 16, "lower_rank": 250, "guard_min": 0.9, "gap_min": 0.02, "ceiling_limited_r2": 0.8,
                      "tolerances": {"E4": 1e-9, "E5": 1e-10, "E6": 1e-10, "kernel": 1e-10, "algebra": 1e-12, "I1": 1e-4, "I3": 1e-4, "I4": 1e-3, "I5": 0.0},
                      "cell_columns": ["n", "S", "Q", "SSE0", "SSE1", "SSEC", "mean", "M2"], "cells_version": "023-cells-v1", "strata": STRATA, "slots": slots,
                      "p0_mask": 0, "p1_mask": P1, "results": ["NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS"], "statistic": b0c.STATISTIC,
                      "conditions": CONDITIONS, "scope": b0c.SCOPE}
check("2 constants (from the design) == lock == record", expected_constants == lock["constants"] == record["constants"],
      {k: expected_constants[k] == lock["constants"].get(k) for k in expected_constants})
# --- item 6: conditions, guard, semantics
expected_conditions = {c: {"population": c.split("/")[0], "group": c.split("/")[1], "statistic": b0c.STATISTIC,
                           "envelope": {"kind": "lower", "rank": 250, "element": 249, "bound": floors[c]}, "guard": {"min_inclusive": 0.9},
                           "guard_bound": floors[c] < 0.9} for c in CONDITIONS}
check("6 the four conditions == record envelopes (own F) + guard {min_inclusive 0.90}; guard_bound == (F < 0.90) == record", expected_conditions == lock["conditions"]
      and all(record["conditions"][c]["envelope"] == expected_conditions[c]["envelope"] and record["conditions"][c]["guard_bound"] == expected_conditions[c]["guard_bound"] for c in CONDITIONS)
      and b0c.GUARD == {"min_inclusive": 0.9}, {c: floors[c] for c in CONDITIONS})
DATA["effective_pass_threshold"] = {c: max(floors[c], 0.90) for c in CONDITIONS}
sem = lock["semantics"]
check("6 semantics: marginal conditions, no aggregate label, no all-pass requirement, Y1/Y2 never pooled; precedence; == b0c.SEMANTICS",
      sem == b0c.SEMANTICS and sem["aggregate"].startswith("none:") and "each read on its own" in sem["aggregate"] and "no all-pass requirement" in sem["aggregate"]
      and "never pooled" in sem["aggregate"] and sem["precedence"] == ["NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS"]
      and "joint" not in lock_text and "all_four" not in lock_text, sem["aggregate"])
# --- program, top level, y1 companion (the model-derived digests are filled in by the model review)
expected_program = {"P0_mask": 0, "P1_mask": P1, "construction": b0c.TABLE_CONSTRUCTION}
check("2 program == {P0 0, P1 14/30, 022's construction}", expected_program == lock["program"] and "ul.pair_context, ul.compose_dx3, ul.contrast_of" in b0c.TABLE_CONSTRUCTION)
y1_bytes = y1_path.read_bytes()
expected_y1 = {"data_path": f"{E23}/locked-y1-table.f64", "index_path": f"{E23}/locked-y1-table.json", "file_sha256": sha(y1_bytes), "index_sha256": sha_file(y1_index_path),
               "total_bytes": (1728 + 864) * 2 * 79 * 8, "layout": [{"name": "cue_final", "shape": [1728, 2, 79]}, {"name": "coordinated", "shape": [864, 2, 79]}],
               "p0_dx3_sha256": None, "factors_sha256": None, "gates": None}
check("2 y1_table: paths, file/index digests, 3,276,288 bytes, layout == lock", all(expected_y1[k] == lock["y1_table"][k] for k in ("data_path", "index_path", "file_sha256", "index_sha256", "total_bytes", "layout"))
      and len(y1_bytes) == 3276288 and set(lock["y1_table"]) == set(expected_y1), {k: lock["y1_table"][k] for k in ("file_sha256", "index_sha256", "total_bytes")})
expected_lock = {"experiment": "023", "schema_version": 1, "kind": kind_literal, "design": DESIGN, "plan": PLAN, "run_id": state["run_id"], "protocol_code_commit": head,
                 "inputs": expected_inputs, "module_blobs": blobs, "constants": expected_constants, "calibration": expected_cal, "exposed_cells": expected_cells,
                 "confirmation_023": expected_conf, "conditions": expected_conditions, "semantics": dict(b0c.SEMANTICS), "program": expected_program, "noun_keys": noun_keys,
                 "exposed_states_sha256": exposed_digest, "y1_table": expected_y1, "y2_table": expected_y2}
check("2 nothing else in the lock: top-level keys == expected (+ content_sha256)", set(lock) == set(expected_lock) | {"content_sha256"}, sorted(set(lock) ^ (set(expected_lock) | {"content_sha256"})))
differing = sorted(k for k in expected_lock if k != "y1_table" and expected_lock[k] != lock.get(k))
check("2 every lock field except the y1 model digests == my reconstruction", differing == [] and kind_literal == "the preregistration lock of Experiment 023 (design revision 2, plan revision 1)"
      and state["run_id"] == lock["run_id"] == record["run_id"], differing)
check("2 lock content_sha256 == own canonical digest == state; file == canonical JSON + newline", content_digest(lock) == lock["content_sha256"] == sl["content_sha256"]
      == "97ca520f342e212d5172b1476e9d5a80c6c2622f80a4f99f4e15e0a399007117" and lock_text == cj(lock) + "\n")
prereg = prereg_bytes.decode("utf-8")
check("2 preregistration == b0c.render_preregistration(lock) (text-only), sha == state", prereg == b0c.render_preregistration(lock) and sha(prereg_bytes) == sl["preregistration_sha256"],
      sha(prereg_bytes))
json.dump(expected_lock, open(OUT / "expected_lock_partial.json", "w"), ensure_ascii=False, indent=1)

# =====================================================================================================================
print("== item 5: 022 isolation (lock run)", flush=True)
lock_files = load(Path(SCRATCH).parent / "lock023/lock_files.json")
check("5 lock run: exit 0, one model load, forward refusal installed, refused_during_run empty, no 022 table in the opened list",
      lock_files["exit"] == 0 and lock_files["model_loads"] == ["EleutherAI/pythia-70m-deduped"] and lock_files["forward_refusal_installed"] is True
      and lock_files["refused_during_run"] == [] and not any("calibration-table" in p for p in lock_files["opened"]), len(lock_files["opened"]))
weights_blob = "3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd"
check("4 lock run's file-open log lists no weights blob (safetensors opens natively, below the audit hook)", not any(weights_blob in p for p in lock_files["opened"]),
      [p.rsplit("/", 1)[-1] for p in lock_files["opened"] if "/blobs/" in p])

# =====================================================================================================================
print("== item 8: stage isolation", flush=True)
present = sorted(p.name for p in O23.iterdir())
check("8 no y2-table.* and no stage2-measurements.pt / report.md in outputs/experiment-023", not any(n.startswith("y2-table") or n in ("stage2-measurements.pt", "report.md") for n in present), present)
check("8 state: confirmation None, report None, both ledgers empty, confirm not_started", state["confirmation"] is None and state["report"] is None and state["executed_prompt_keys"] == []
      and state["executed_noun_keys"] == [] and phases["confirm"]["status"] == "not_started")
installed = [n for n in ("preregistration-lock.json", "preregistration.md", "locked-y1-table.f64", "locked-y1-table.json") if (ROOT / E23 / n).exists()]
tracked = [line for line in git("ls-files", E23).splitlines() if any(line.endswith(n) for n in ("preregistration-lock.json", "preregistration.md", "locked-y1-table.f64", "locked-y1-table.json"))]
check("8 nothing installed or tracked in experiments/023-block0-completion (lock, preregistration, Y1 table)", installed == [] and tracked == [], {"installed": installed, "tracked": tracked})

DATA["guard_end"] = {"refused_during_run": list(guard.REFUSED), "opened_022": sorted({p for p in guard.OPENED if "experiment-022" in p})}
check("5 this review process: no refusal during the run; nothing opened under outputs/experiment-022", guard.REFUSED == [] and not any("outputs/experiment-022" in p for p in guard.OPENED),
      DATA["guard_end"])
DATA["results"] = RESULTS
(OUT / "review_static.json").write_text(json.dumps(DATA, indent=1, default=str), encoding="utf-8")
failed = [r for r in RESULTS if not r[1]]
print(f"\nSTATIC REVIEW: {len(RESULTS) - len(failed)} PASS, {len(failed)} FAIL", flush=True)
for r in failed:
    print("  FAIL:", r[0], r[2][:300])
