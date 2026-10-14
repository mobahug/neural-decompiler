"""Review item 1 (execution integrity), independent of the runner's code: the state's digests, the result write's digest
reconstructed, the measurement file and its tensors' digests, the keys (manifest, ledger, the launcher's per-run log, the
measurement rows), the replacement digests logged at admission against the saved replacements, and the landing of every
saved replacement. No model, no neural_decompiler import."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rguard  # noqa: E402  (first)

rguard.self_test()
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402

ROOT = rguard.ROOT
CONFIRM = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/1e9b7b17-f314-4497-8f93-aab17cd53f53/scratchpad/exp101/confirm")
EXP = ROOT / "experiments/101-read-component-intervention"
OUT = rguard.OUT
out = {"checks": {}, "facts": {}}


def check(name, ok, detail=None):
    out["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if ok and detail is None else f"  {detail}"), flush=True)


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_digest(t: torch.Tensor) -> str:  # own implementation of the documented rule: shape json | kind | LE bytes
    a = t.detach().cpu().contiguous()
    kind = "<i8" if a.dtype in (torch.int64, torch.int32) else "<f8"
    data = a.numpy().astype(kind, copy=False).tobytes()
    return hashlib.sha256(json.dumps(list(a.shape)).encode("ascii") + b"|" + kind.encode("ascii") + b"|" + data).hexdigest()


# --- the state ----------------------------------------------------------------------------------------------------
results_path = ROOT / "outputs/experiment-101/results.json"
raw = json.loads(results_path.read_text(encoding="utf-8"))
recorded = raw.pop("state_sha256")
check("state_digest_reproduces", sha_text(canonical(raw)) == recorded, recorded)
check("state_digest_is_the_launcher_post_record", recorded == "1e3eaeb38ff59941ce7f6a788751d779691ca4fcfa8a4dfedfa0c0e35bd2f37a")
check("results_file_sha256_is_outputs_after_confirm", sha_file(results_path) == "adc83b1ae030fa8a104c9af3c4cfea90c6e1e89624a8fa5bb239f1d3c93c42e0")
# the result write (before the secondary step): the same state without confirmation.secondary
at_result = json.loads(json.dumps(raw))
secondary = at_result["confirmation"].pop("secondary")
check("result_write_digest_reconstructed_04221aa2", sha_text(canonical(at_result)) == "04221aa2f53eae384d2a49e7cc9ddea73a0063e6599a09bdffa93b257684ccfc",
      sha_text(canonical(at_result)))
check("secondary_has_no_failures", "failures" not in secondary, sorted(secondary))
c = raw["confirmation"]
check("confirm_phase_complete", raw["phases"]["confirm"]["status"] == "complete" and raw["phases"]["confirm"]["confirm_commit"] == "59d8ec49e9a34cb40f23b7c3928e8b36522972fe",
      raw["phases"]["confirm"])
check("no_incident_anywhere", "incident" not in c and not raw["phases"]["confirm"].get("incidents") and not raw["phases"]["stage1"].get("incidents"))
check("report_not_started", raw["phases"]["report"] == {"status": "not_started"} and raw.get("report") is None)
check("lock_and_barrier", c["lock_sha256"] == "8da97e35556d0baa4739899cb841e86cbd59cced21611c732fdd9d8c9edd9061" and c["barrier"]["passed"] is True
      and c["barrier"]["table_content_sha256"] == "9191800440af37909d2068459e360d7772191fc66bf2ad5fb9cf2c133246ac86"
      and c["barrier"]["stage1_digest"] == raw["stage1"]["digest"])
out["facts"]["phases"] = raw["phases"]

# --- the measurement file ---------------------------------------------------------------------------------------------
pt = ROOT / "outputs/experiment-101/stage2-measurements.pt"
check("measurements_file_sha256_is_outputs_after_confirm", sha_file(pt) == "4e9a3a638b67391f63b228185483086389583b703b213a28b6947851b86e6678")
check("no_partial_file", not (ROOT / "outputs/experiment-101/stage2-partial.pt").exists())
tensors = torch.load(pt)
digests_now = {k: tensor_digest(v) for k, v in tensors.items()}
check("tensor_digests_equal_state", digests_now == c["stage2"]["tensors_sha256"], len(digests_now))
check("n_executed_1980", c["stage2"]["n_executed"] == 1980)

# --- keys ---------------------------------------------------------------------------------------------------------
lock = json.loads((EXP / "preregistration-lock.json").read_text(encoding="utf-8"))
conf = json.loads((EXP / "confirmation-v1.json").read_text(encoding="utf-8"))
run_order = lock["run_order"]
check("run_order_55", len(run_order) == 55 and run_order == conf["run_order"] == lock["configuration"]["tags"])
REF_PROMPT_ARMS = {"REF", "SUF-READ", "SUF-OFF", "SUF-NEG", "SUF-HALF", "PC-SWAP-REF", "HEAD-SUF", "HEAD-SUF-V"} | {f"SUF-ISO-{j}" for j in range(1, 9)}
CUES = {"two": 767, "three": 1264, "many": 1142}


def prompt_key(frame_id: str, tag: str) -> str:
    name, _, label = tag.partition("@")
    if name in REF_PROMPT_ARMS:
        return f"{frame_id}|ref|581|{tag}"
    return f"{frame_id}|{label}|{CUES[label]}|{tag}"


frames = [f["frame_id"] for f in conf["frames"]]
check("frames_are_01_to_36_in_order", frames == [f"coordinated-adjective-101-{i:02d}" for i in range(1, 37)])
expected_order = [prompt_key(fid, tag) for fid in frames for tag in run_order]
check("expected_s2_keys_distinct_1980", len(expected_order) == 1980 == len(set(expected_order)))
check("manifest_s2_equals_expected", sorted(conf["manifest"]["S2-TARGET"]) == sorted(expected_order) and conf["manifest"]["S2-TARGET"] == sorted(conf["manifest"]["S2-TARGET"]))
s1 = [f"{fid}|ref|581|S1" for fid in frames]
check("manifest_s1_equals_expected", sorted(conf["manifest"]["S1-REF"]) == sorted(s1))
ledger = raw["executed_prompt_keys"]
check("ledger_is_s1_plus_s2_exactly", len(ledger) == 2016 == len(set(ledger)) and set(ledger) == set(s1) | set(expected_order))
rows = [json.loads(line) for line in (CONFIRM / "production_runs.jsonl").read_text(encoding="utf-8").splitlines()]
check("run_log_1980_rows_numbered", len(rows) == 1980 and [r["n"] for r in rows] == list(range(1, 1981)))
check("run_log_order_is_frozen_order", [r["key"] for r in rows] == expected_order)
check("run_log_no_s1_key", not any(r["key"].endswith("|S1") for r in rows))
check("run_log_each_key_once", collections.Counter(r["key"] for r in rows) == collections.Counter(expected_order))

# --- the logged replacement digests against the saved replacements; landing -----------------------------------------
positions = {fid: tensors[f"{fid}/positions"].tolist() for fid in frames}
mismatch, landing_fail, kinds = [], [], collections.Counter()
for r in rows:
    fid, label, tok, tag = r["key"].split("|")
    p_c, p_t = positions[fid]
    name = tag.partition("@")[0]
    row = run_order.index(tag)
    repl = r["replacements"]
    if name in ("REF", "PL"):
        kinds["plain"] += 1
        if repl:
            mismatch.append((r["key"], "plain run with a replacement"))
        continue
    if name.startswith("HEAD-"):
        kinds["head"] += 1
        saved = tensors[f"{fid}/head_replacement/{tag}"]
        site = f"L03.H04@{p_t}"
        captured = tensors[f"{fid}/T_pt"][row]
    else:
        kinds["residual"] += 1
        saved = tensors[f"{fid}/replacement/{tag}"]
        site = f"RESID_PRE.L3@{p_c}"
        captured = tensors[f"{fid}/x3_pc"][row]
    digest = hashlib.sha256(saved.detach().to(torch.float32).contiguous().reshape(-1).numpy().tobytes()).hexdigest()
    if list(repl) != [site] or repl[site] != digest:
        mismatch.append((r["key"], list(repl), site))
    if not torch.equal(captured.reshape(-1), saved.reshape(-1)):
        landing_fail.append(r["key"])
check("logged_replacement_digests_equal_saved_replacements", not mismatch, mismatch[:3])
check("every_saved_replacement_landed_bitwise", not landing_fail, landing_fail[:3])
out["facts"]["run_kinds"] = dict(kinds)
check("run_kinds", dict(kinds) == {"plain": 36 * 4, "residual": 36 * 47, "head": 36 * 4}, dict(kinds))

# --- the stage-2 tensors' structure -----------------------------------------------------------------------------------
expected_names = set()
for fid in frames:
    expected_names |= {f"{fid}/{n}" for n in ("contrasts", "logit_identity", "x3_pc", "x3_pt", "x4_pc", "heads_pc", "mlp_pc", "T_pt", "attn_pt", "h6",
                                              "positions", "c_ref_pm")}
    for tag in run_order:
        name = tag.partition("@")[0]
        if name in ("REF", "PL"):
            expected_names.add(f"{fid}/x3_all/{tag}")
        elif name.startswith("HEAD-"):
            expected_names |= {f"{fid}/delta/{tag}", f"{fid}/head_replacement/{tag}"}
        else:
            expected_names.add(f"{fid}/replacement/{tag}")
check("tensor_names_exactly_expected", set(tensors) == expected_names, (len(tensors), len(expected_names)))
nonfinite = [k for k, v in tensors.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
check("all_tensors_finite", not nonfinite, nonfinite[:3])
shape_bad = [fid for fid in frames if tensors[f"{fid}/contrasts"].shape != (55, 79)]
check("contrasts_shape_55x79", not shape_bad)
pos_ok = all(positions[f["frame_id"]] == [f["p_c"], f["p_t"]] for f in conf["frames"])
check("positions_equal_frozen_frames", pos_ok)

# --- the logits' digests per row: plain cue runs differ from REF; NOOP equals PL ----------------------------------------
logit_d = c["stage2"]["logits_sha256"]
check("logit_digests_55_per_frame", all(len(logit_d[fid]) == 55 for fid in frames))
dup = []
for fid in frames:
    seen = collections.defaultdict(list)
    for i, d in enumerate(logit_d[fid]):
        seen[d].append(run_order[i])
    for d, tags in seen.items():
        if len(tags) > 1 and sorted(tags) != sorted(["PL@two", "NOOP@two"]):
            dup.append((fid, tags))
check("only_noop_and_pl_share_logits", not dup, dup[:5])

# --- the launcher's records -----------------------------------------------------------------------------------------
prod = json.loads((CONFIRM / "production_run_20261014T174234317359Z.json").read_text(encoding="utf-8"))
ev = prod["events"]
check("launcher_one_confirm_one_load", ev["confirm_invocations"] == 1 and ev["load_model"] == 1)
check("launcher_no_refusal_no_forbidden", not ev["refused"] and not ev["forbidden_reads"] and not ev["forbidden_writes"] and not ev["signals"])
check("launcher_1980_sanctioned_runs_340_calls_each", ev["sanctioned_runs"] == 1980 and ev["sanctioned_forward_calls"] == 1980 * 340)
check("launcher_sha256", prod["environment"]["launcher_sha256"] == "56a687f372605e5d6e16e2cc62bc9fd1197f21fdd8dccf0dda6d6ac75723309f"
      == sha_file(CONFIRM / "launch_confirm.py") == (CONFIRM / "confirm.sentinel").read_text().strip())
check("launcher_head_and_tree", prod["environment"]["head"] == "59d8ec49e9a34cb40f23b7c3928e8b36522972fe" and prod["tree_after"] == ""
      and prod["preflight"]["N1_tree_clean"] is True)
check("launcher_exit_0", prod["exit_status"] == 0 and prod["error"] is None and (CONFIRM / "production.exit").read_text().strip() == "exit 0")
writes = [p for p in ev["outputs_reads"] if "/outputs/" in p]
out["facts"]["outputs_paths_touched_by_open"] = writes
temps = [p for p in writes if Path(p).name.startswith(".results-")]
check("nine_atomic_state_writes", len(temps) == 9, len(temps))
check("no_temp_state_left", not any(Path(p).exists() for p in temps))
check("only_allowed_outputs", all(Path(p).name in ("results.json", "stage2-measurements.pt") or Path(p).name.startswith(".results-") or p.endswith("experiment-101")
                                  or p.endswith("experiment-020/results.json") for p in writes), writes)
out["facts"]["confirm_times"] = {"launcher_start": prod["started_at"], "confirm_started": raw["phases"]["confirm"]["started_at"],
                                 "confirm_completed": raw["phases"]["confirm"]["completed_at"], "launcher_end": prod["ended_at"]}
mt = {p.name: p.stat().st_mtime for p in (ROOT / "outputs/experiment-101").iterdir()}
out["facts"]["outputs_mtimes"] = mt
(OUT / "a_integrity.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
print("failed:", [k for k, v in out["checks"].items() if not v["ok"]])
