"""Items 1, 2, 6, 8 (and the file-level parts of 3/5): prompt accounting, ordering and the barrier, bindings and
isolation, state consistency and one-shot evidence. No model, no forward pass, read-only."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirmreview023")
import guard  # noqa: E402  FIRST: audit hook + torch.load refusal, proven live

import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(guard.ROOT)
EV = Path("/private/tmp/claude-501/-Users-gaborhorvath-ulenius-myprojects-neural-decompiler/8109f22f-55d4-4c73-8b05-dafc2d319c0b/scratchpad/confirm023")
E = ROOT / "experiments/023-block0-completion"
OUT = ROOT / "outputs/experiment-023"
HEAD = "c7efec7f32709dc20ecb83cae16bb73927a14366"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail != "" else ""), flush=True)
    if not ok:
        FAIL.append(name)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cj(value):  # my own canonical JSON (the definition in plural_mechanism.canonical_json, rewritten)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iso(t):
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat(timespec="milliseconds")


def parse_utc(s):
    return datetime.datetime.fromisoformat(s).timestamp()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


# ---------------------------------------------------------------- inputs (raw JSON, own digests)
conf_bytes = (E / "confirmation-v1.json").read_bytes()
conf = json.loads(conf_bytes)
check("confirmation-v1.json file sha256 == 5fadfa50…", sha(E / "confirmation-v1.json") == "5fadfa503f4fe35308cb4473220a1d9cd6f7a46e3852f638594b8081909825f4")
check("confirmation-v1.json content digest recomputes (own canonical JSON)", sha_text(cj({k: v for k, v in conf.items() if k != "content_sha256"})) == conf["content_sha256"]
      == "4e64d4c2c85ae8f9c710171372a4bf28f0964a8e8864a0326182edba44aa14ed", conf["content_sha256"][:16])
lock = json.loads((E / "preregistration-lock.json").read_text(encoding="utf-8"))
check("lock file sha256 == 4bd5a5b1…", sha(E / "preregistration-lock.json") == "4bd5a5b14627768d49398c273fa1ce387db2e4739d7d31b7258072ef48beac9c")
check("lock content digest recomputes == 97ca520f…", sha_text(cj({k: v for k, v in lock.items() if k != "content_sha256"})) == lock["content_sha256"]
      == "97ca520f342e212d5172b1476e9d5a80c6c2622f80a4f99f4e15e0a399007117")

# the 108 exposed frames: from Experiment 020's closure (locked states) via the frozen loader, not from 023's file
from neural_decompiler import upstream_localization as ul  # noqa: E402
from neural_decompiler import plural_mechanism as pm  # noqa: E402
import inspect  # noqa: E402

inputs = ul.load_frozen_inputs(ROOT)
pool_frames = [frame.frame_id for frame in inputs.pool.frames]
locked_ids = sorted(inputs.closure["exploration"]["locked_states"])
check("108 exposed frames: 020 pool == 020 locked states == confirmation-v1.json exposed_frame_ids", len(pool_frames) == 108 and sorted(pool_frames) == locked_ids
      and conf["exposed_frame_ids"] == pool_frames, f"{len(pool_frames)} frames")
check("reference cue ids: confirmation-v1.json == 020 pool", {k: int(v) for k, v in conf["reference_cue_ids"].items()} == {k: int(v) for k, v in inputs.pool.reference_ids.items()},
      conf["reference_cue_ids"])
# the key format, read from the source of pm.Prompt.key (then reimplemented below)
src = inspect.getsource(pm.Prompt)
check("pm.Prompt.key format is '{frame_id}|{cue_label}|{cue_token_id}'", 'return f"{self.frame.frame_id}|{self.cue_label}|{self.cue_token_id}"' in src)


def key(frame_id, label, token_id):
    return f"{frame_id}|{label}|{int(token_id)}"


cues = [(c["word"], int(c["token_id"]), c["class"]) for c in conf["cues"]]
new_frames = [(f["frame_id"], f["template_id"], {k: int(v) for k, v in f["cue_ids"].items()}) for f in conf["frames"]]
ref_ids = {k: int(v) for k, v in conf["reference_cue_ids"].items()}
check("24 cues (8/8/8), 18 new frames (6/6/6), all ids distinct", len(cues) == 24 and len({t for _, t, _ in cues}) == 24 and Counter(c for _, _, c in cues) == Counter({"determiner-like": 8, "quantity": 8, "adjective": 8})
      and len(new_frames) == 18 and Counter(t for _, t, _ in new_frames) == Counter({"cardinal": 6, "quantifier": 6, "coordinated-adjective": 6}))
M = {"S1-REF": sorted(key(fid, "ref", ref_ids[tpl]) for fid, tpl, _ in new_frames),
     "S1-VALIDITY": sorted(key(fid, "pl", ids["pl"]) for fid, _, ids in new_frames),
     "Y1": sorted(key(efid, w, t) for efid in pool_frames for w, t, _ in cues),
     "Y2": sorted(key(fid, w, t) for fid, _, _ in new_frames for w, t, _ in cues)}
sizes = {k: len(v) for k, v in M.items()}
allkeys = M["S1-REF"] + M["S1-VALIDITY"] + M["Y1"] + M["Y2"]
check("my manifest: S1-REF 18, S1-VALIDITY 18, Y1 2,592, Y2 432 = 3,060 unique", sizes == {"S1-REF": 18, "S1-VALIDITY": 18, "Y1": 2592, "Y2": 432} and len(set(allkeys)) == 3060, sizes)
check("confirmation-v1.json manifest == my manifest", conf["manifest"] == {"S1-REF": M["S1-REF"], "S1-VALIDITY": M["S1-VALIDITY"], "S2-TARGET": {"Y1": M["Y1"], "Y2": M["Y2"]}})
check("lock manifest sizes == mine", lock["confirmation_023"]["manifest_sizes"] == sizes)

# ---------------------------------------------------------------- executed prompts (the launcher's log)
lines = [json.loads(line) for line in (EV / "prompts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
ns = [entry["n"] for entry in lines]
keys = [entry["key"] for entry in lines]
ts = [entry["t"] for entry in lines]
cnt = Counter(keys)
dups = sorted(k for k, c in cnt.items() if c > 1)
extra = sorted(set(keys) - set(allkeys))
missing = sorted(set(allkeys) - set(keys))
check("prompts.jsonl: 3,060 lines, n = 1..3060 contiguous (one launcher run)", len(lines) == 3060 and ns == list(range(1, 3061)))
check("executed keys == my manifest: each exactly once; no extra, omission or duplicate", not dups and not extra and not missing and len(keys) == 3060,
      f"extra {len(extra)} missing {len(missing)} dup {len(dups)}")
check("timestamps non-decreasing", all(b >= a for a, b in zip(ts, ts[1:])))
summary = json.loads((EV / "confirm_summary.json").read_text(encoding="utf-8"))
c = summary["counts"]
check("launcher counts: capture_prompt == run_capture == 3060; run_patched == run_interventions == 0; exit 0; no error", c == {"capture_prompt": 3060, "run_capture": 3060, "run_patched": 0, "run_interventions": 0}
      and summary["exit"] == 0 and summary["error"] is None, c)
check("launcher: the 022 table was never opened during the run (refused_during_run empty)", summary["refused_during_run"] == [])

# ---------------------------------------------------------------- the state
raw = (OUT / "results.json").read_bytes()
check("results.json file sha256 == 328be2b6…", hashlib.sha256(raw).hexdigest() == "328be2b688fd8cef34a523c308dc4dd5790c703a90ab0cc6f66ac377a0ec21f2")
state = json.loads(raw)
recorded = state.pop("state_sha256")
check("state_sha256 recomputes with my canonical JSON", sha_text(cj(state)) == recorded == "938cb2469d6445b4ffcf04ca5937e8d26f4e80c80e7b96ad5458102f70fa5876", recorded[:16])
check("the file is exactly canonical JSON + newline (no hand edit)", raw == (cj({**state, "state_sha256": recorded}) + "\n").encode("utf-8"))
ledger = state["executed_prompt_keys"]
check("ledger (executed_prompt_keys) == my manifest, sorted, no duplicates", ledger == sorted(allkeys) and len(ledger) == len(set(ledger)) == 3060)
expected_nouns = sorted({f"{n.split.value}:{n.lexical_key}" for n in inputs.pool.nouns})
fresh_020 = {f"{n.split.value}:{n.lexical_key}" for n in inputs.confirmation_020.nouns} if hasattr(inputs.confirmation_020, "nouns") else set()
scorable = [n.lexical_key for n in inputs.pool.nouns if n.single_token]
check("executed_noun_keys == every exposed pool noun key (pm.record_execution over inputs.pool.nouns); no 020 fresh noun", state["executed_noun_keys"] == expected_nouns and not (set(expected_nouns) & fresh_020),
      f"{len(expected_nouns)} keys ({len(scorable)} scorable single-token, {len(expected_nouns) - len(scorable)} multi-token); fresh-020 overlap {len(set(expected_nouns) & fresh_020)} of {len(fresh_020)}")
check("scored nouns == the lock's 79 noun keys", scorable == lock["noun_keys"] and len(scorable) == 79)
# forbidden: 020 ledger, 020 confirmation keys, 022 manifest
conf022 = json.loads((ROOT / "experiments/022-upstream-error-localization/confirmation-v1.json").read_text(encoding="utf-8"))
m22 = conf022["manifest"]
forbidden = set(inputs.closure["ledger"]) | {p.key for p in inputs.confirmation_020.all_prompts} | set(m22["S1-REF"]) | set(m22["S1-VALIDITY"]) | set(m22["S2-TARGET"]["Y1"]) | set(m22["S2-TARGET"]["Y2"])
check("no executed key was executed or reserved before (020 ledger, 020 confirmation, 022 manifest)", not (set(keys) & forbidden), f"{len(forbidden)} forbidden keys")

# ---------------------------------------------------------------- phases, one-shot
ph = state["phases"]
check("phases: extract, calibrate, lock complete; confirm complete; report not_started", [ph[p]["status"] for p in ("extract", "calibrate", "lock", "confirm", "report")]
      == ["complete", "complete", "complete", "complete", "not_started"])
check("confirm: commit c7efec7, lock 97ca520f…, I7 bitwise_equal with the Y1 file digest", ph["confirm"]["confirm_commit"] == HEAD and ph["confirm"]["lock_sha256"] == lock["content_sha256"]
      and ph["confirm"]["I7"] == {"bitwise_equal": True, "file_sha256": "37603f058500e056214a8a6ae413ef2b27eb94c5e1d618caaa8c9a61121a2f96"}, f"{ph['confirm']['started_at']} → {ph['confirm']['completed_at']}")
check("no incident anywhere (no phase 'incidents', no confirmation 'incident', no calibration incidents)", not any(ph[p].get("incidents") for p in ph) and "incident" not in state["confirmation"]
      and not (state.get("calibration") or {}).get("incidents"))
check("report: state['report'] is None and no report.md", state.get("report") is None and not (OUT / "report.md").exists())
d = state["confirmation"]["descriptives"]
check("descriptives present (subsets, comparators after the results; p1_dx3_relative_error, block0_profile from the gates); no failures", set(d) == {"subsets", "comparators", "p1_dx3_relative_error", "block0_profile"} and not d.get("failures"))
check("no aggregate label; confirmation.lock_sha256 == lock", state["confirmation"]["aggregate_label"] is None and state["confirmation"]["lock_sha256"] == lock["content_sha256"])
check("stage 2: n_executed == 3024 and path is the bound stage-2 file", state["confirmation"]["stage2"]["n_executed"] == 3024 and Path(state["confirmation"]["stage2"]["path"]) == OUT / "stage2-measurements.pt")

# ---------------------------------------------------------------- the stage-1 record and the barrier
s1 = state["confirmation"]["stage1"]
check("stage-1 record reproduces its digest (own canonical JSON)", sha_text(cj({k: v for k, v in s1.items() if k != "digest"})) == s1["digest"], s1["digest"][:16])
check("stage-1 record names the lock and the commit", s1["lock_sha256"] == lock["content_sha256"] and s1["commit"] == HEAD)
check("stage-1 state digests: each of the 18 recorded states hashes (own canonical JSON) to its recorded digest", sorted(s1["states"]) == sorted(f for f, _, _ in new_frames)
      and all(sha_text(cj(s1["states"][f])) == s1["state_digests"][f] for f in s1["states"]))
y2 = s1["y2_table"]
check("stage-1 Y2 digests == the files on disk (data 4623adde…, index e87091ef…)", y2["file_sha256"] == sha(OUT / "y2-table.f64") == "4623adde5eb53e14b0368e5b4d8b91bd6f2513bd522ba074bc55ab747970d2c2"
      and y2["index_sha256"] == sha(OUT / "y2-table.json") and y2["index_sha256"].startswith("e87091ef") and y2["total_bytes"] == (OUT / "y2-table.f64").stat().st_size == 546048)
check("stage-1 Y2 paths == the lock's", (y2["data_path"], y2["index_path"]) == (lock["y2_table"]["data_path"], lock["y2_table"]["index_path"]))
check("stage-1 noun keys == lock", s1["noun_keys"] == lock["noun_keys"])
check("stage-1 frames: 18, p_c/p_t == the frozen frames; validity descriptive", all((s1["frames"][f]["p_c"], s1["frames"][f]["p_t"]) == (next(e["p_c"] for e in conf["frames"] if e["frame_id"] == f), next(e["p_t"] for e in conf["frames"] if e["frame_id"] == f))
                                                                                   and s1["frames"][f]["selects"].startswith("nothing") for f in s1["frames"]), f"validity all True: {all(s1['frames'][f]['validity'].get('valid') for f in s1['frames'])}")
check("stage-1 gates (I5, algebra) within tolerance", s1["y2_table"]["gates"]["I5"]["max"] == 0.0 and s1["y2_table"]["gates"]["algebra"]["max"] <= 1e-12, s1["y2_table"]["gates"])

# ordering: stage-1 keys first, in the stage_one order (frames by frame_id, ref then pl)
s1_keys = set(M["S1-REF"]) | set(M["S1-VALIDITY"])
pos1 = [i for i, k in enumerate(keys) if k in s1_keys]
pos2 = [i for i, k in enumerate(keys) if k not in s1_keys]
check("the 36 stage-1 keys are lines 1..36 (by order) and precede every stage-2 key", pos1 == list(range(36)) and min(pos2) == 36)
t1_last, t2_first = max(ts[i] for i in pos1), min(ts[i] for i in pos2)
check("by timestamp: last stage-1 < first stage-2", t1_last < t2_first, f"last S1 {iso(t1_last)}, first S2 {iso(t2_first)}")
order1 = [x for fid in sorted(f for f, _, _ in new_frames) for x in (key(fid, "ref", ref_ids[dict((f, t) for f, t, _ in new_frames)[fid]]), key(fid, "pl", dict((f, i) for f, _, i in new_frames)[fid]["pl"]))]
check("stage-1 order == frames by frame_id, S1-REF then S1-VALIDITY", keys[:36] == order1)
# stage-2 order: Y1 (exposed frames in table order: frame_id sorted; cues by token id), then Y2
tok_sorted = sorted(cues, key=lambda c: c[1])
exp_order = [key(f, w, t) for f in sorted(pool_frames) for w, t, _ in tok_sorted] + [key(f, w, t) for f in sorted(fid for fid, _, _ in new_frames) for w, t, _ in tok_sorted]
check("stage-2 order == Y1 then Y2, frames by frame_id, cues by token id (ul.stage_two_022)", keys[36:] == exp_order)
# file times of the Y2 table (materialized once, after stage 1 and before the targets)
for name in ("y2-table.f64", "y2-table.json"):
    st = os.stat(OUT / name)
    birth = getattr(st, "st_birthtime", None)
    check(f"{name}: created and last modified after the last stage-1 prompt and before the first S2-TARGET prompt; never rewritten (birth ≈ mtime ≈ ctime)",
          t1_last < birth <= st.st_mtime < t2_first and abs(st.st_ctime - st.st_mtime) < 1.0 and st.st_mtime - birth < 1.0,
          f"birth {iso(birth)} mtime {iso(st.st_mtime)} ctime {iso(st.st_ctime)}; window ({iso(t1_last)}, {iso(t2_first)})")
st2 = os.stat(OUT / "stage2-measurements.pt")
check("stage2-measurements.pt written once, after the last S2 prompt", st2.st_birthtime >= ts[-1] and st2.st_mtime - st2.st_birthtime < 5, f"birth {iso(st2.st_birthtime)} mtime {iso(st2.st_mtime)} last prompt {iso(ts[-1])}")
st3 = os.stat(OUT / "results.json")
run_start, run_end = parse_utc("2026-09-24T17:38:14+00:00"), parse_utc("2026-09-24T17:49:23+00:00") + 1.0
check("results.json last modified within the confirm run (≤ its end 17:49:23Z) and after completion", parse_utc(ph["confirm"]["completed_at"]) <= st3.st_mtime <= run_end, f"mtime {iso(st3.st_mtime)}")
check("confirm started_at (the I7 record, second resolution) ≤ first capture", parse_utc(ph["confirm"]["started_at"]) <= ts[0] < parse_utc(ph["confirm"]["started_at"]) + 1.0 and run_start <= ts[0],
      f"started_at {ph['confirm']['started_at']}, first capture {iso(ts[0])}")
check("confirm completed_at after the last capture", parse_utc(ph["confirm"]["completed_at"]) >= ts[-1] - 1.0, f"{ph['confirm']['completed_at']} vs last capture {iso(ts[-1])}")
out_files = sorted(p.name for p in OUT.iterdir())
check("outputs/experiment-023 holds only the expected files (no report, no second-run leftovers)", out_files == sorted(["candidate-calibration.json", "candidate-exposed-cells.f64", "candidate-exposed-cells.json", "candidate-lock.json",
                                                                                                                      "candidate-locked-y1-table.f64", "candidate-locked-y1-table.json", "candidate-preregistration.md", "draw-values.pt", "results.json",
                                                                                                                      "stage2-measurements.pt", "y2-table.f64", "y2-table.json"]), out_files)
st_sum = os.stat(EV / "confirm_summary.json")
check("confirm_summary.json written once at the run's end (a later launcher run would rewrite it)", abs(st_sum.st_mtime - st3.st_mtime) < 5 and st_sum.st_birthtime >= run_start, f"mtime {iso(st_sum.st_mtime)}")
# the 022 table: never opened here; its atime relative to the run window (weak evidence, stat only)
st22 = os.stat(guard.TABLE_022_RESOLVED)
print(f"INFO  022 table stat (not opened): atime {iso(st22.st_atime)} mtime {iso(st22.st_mtime)}; confirm window {iso(run_start)} → {iso(run_end)}; atime before window: {st22.st_atime < run_start}")

# ---------------------------------------------------------------- bindings (item 6)
committed = {n: E / n for n in ("preregistration-lock.json", "preregistration.md", "locked-y1-table.f64", "locked-y1-table.json", "calibration-v1.json", "confirmation-v1.json",
                                "exposed-cells.f64", "exposed-cells.json")}
expected_sha = {"preregistration-lock.json": "4bd5a5b1", "locked-y1-table.f64": "37603f05", "locked-y1-table.json": "3bf85e18", "calibration-v1.json": "94db6df2", "confirmation-v1.json": "5fadfa50"}
for n, p in committed.items():
    blob = git("show", f"HEAD:experiments/023-block0-completion/{n}")
    check(f"{n}: working tree == HEAD blob" + (f" == {expected_sha[n]}…" if n in expected_sha else ""), p.read_bytes() == blob and (n not in expected_sha or sha(p).startswith(expected_sha[n])), sha(p)[:16])
cal = json.loads(committed["calibration-v1.json"].read_text(encoding="utf-8"))
check("state binds the committed confirmation file (path, file and content digests)", state["confirmation_023"] == {"path": "experiments/023-block0-completion/confirmation-v1.json", "file_sha256": sha(committed["confirmation-v1.json"]),
                                                                                                                   "content_sha256": conf["content_sha256"]})
check("state lock == committed lock (content), preregistration and Y1 digests == committed files", state["lock"]["content_sha256"] == lock["content_sha256"] and state["lock"]["preregistration_sha256"] == sha(committed["preregistration.md"])
      and state["lock"]["y1_file_sha256"] == sha(committed["locked-y1-table.f64"]) and state["lock"]["y1_index_sha256"] == sha(committed["locked-y1-table.json"]))
check("state calibration record digests == committed calibration-v1.json (file, content)", state["calibration"]["record_sha256"] == sha(committed["calibration-v1.json"])
      and state["calibration"]["record_content_sha256"] == cal["content_sha256"] == sha_text(cj({k: v for k, v in cal.items() if k != "content_sha256"})) == lock["calibration"]["content_sha256"]
      and lock["calibration"]["file_sha256"] == sha(committed["calibration-v1.json"]))
check("state/lock/record bind the committed exposed cells", state["extract"]["data_sha256"] == sha(committed["exposed-cells.f64"]) == lock["exposed_cells"]["data_sha256"] == cal["exposed_cells"]["data_sha256"]
      and state["extract"]["index_sha256"] == sha(committed["exposed-cells.json"]) == lock["exposed_cells"]["index_sha256"])
check("lock binds the committed Y1 table (file 37603f05…, index 3bf85e18…)", lock["y1_table"]["file_sha256"] == sha(committed["locked-y1-table.f64"]) and lock["y1_table"]["index_sha256"] == sha(committed["locked-y1-table.json"]))
check("state inputs == lock inputs == record inputs", state["inputs"] == lock["inputs"] == cal["inputs"])
check("lock envelopes == calibration record envelopes (full precision)", all(lock["conditions"][k]["envelope"] == cal["conditions"][k]["envelope"] for k in lock["conditions"]))
check("state condition envelopes == lock envelopes (full precision)", all(state["confirmation"]["conditions"][k]["envelope"] == lock["conditions"][k]["envelope"] for k in lock["conditions"]))

# ---------------------------------------------------------------- the code path: no 022 table, no patched run, one-shot refusal
run_src = (E / "run.py").read_text(encoding="utf-8")
b0c_src = (ROOT / "src/neural_decompiler/block0_completion.py").read_text(encoding="utf-8")
confirm_src = run_src[run_src.index("    def confirm(self)"):run_src.index("    def _descriptives(")]
descr_src = run_src[run_src.index("    def _descriptives("):run_src.index("    def _record_for_report(")]
check("confirm() and _descriptives() never name the 022 table or call torch.load", "TABLE_022" not in confirm_src + descr_src and "calibration-table" not in confirm_src + descr_src and "torch.load" not in confirm_src + descr_src)
check("run.py opens the 022 table only in extract()", [m.start() for m in re.finditer(r"TABLE_022_RELATIVE_PATH", run_src)] and all(run_src.rfind("    def ", 0, m.start()) == run_src.index("    def extract(") for m in re.finditer(r"TABLE_022_RELATIVE_PATH", run_src)))
check("block0_completion.py: TABLE_022_RELATIVE_PATH only a constant and a string in cells_source (no open/torch.load of it)", len(re.findall(r"TABLE_022_RELATIVE_PATH", b0c_src)) == 2 and "torch.load" not in b0c_src)
check("no run_patched / run_interventions in the confirm path sources (run.py confirm, b0c, ul stage_two/measure_prompt)", "run_patched" not in confirm_src + b0c_src and "run_interventions" not in confirm_src + b0c_src)
shiny_lines = [(name, i + 1, line.strip()[:60]) for name, text in (("run.py", run_src), ("block0_completion.py", b0c_src)) for i, line in enumerate(text.splitlines())
               if "shiny" in line or "30006" in line or "coordinated-adjective-009" in line]
check("no 'shiny' / 30006 / coordinated-adjective-009 exclusion in run.py or block0_completion.py (the only hit is the frozen CUE_CANDIDATES list)",
      shiny_lines == [("block0_completion.py", 153, shiny_lines[0][2])] and shiny_lines[0][2].startswith('"adjective": ("humble"'), shiny_lines)
from neural_decompiler import block0_completion as b0c  # noqa: E402
try:
    b0c.assert_phase_allowed("confirm", {**state, "state_sha256": recorded})
    refused = False
except pm.PhaseError as error:
    refused = "already started" in str(error)
check("a second confirm is refused by assert_phase_allowed on this state (called read-only)", refused)
st_now = os.stat(OUT / "results.json")
check("results.json untouched by this review (mtime unchanged)", st_now.st_mtime == st3.st_mtime)
print("\nACCT:", "ALL PASS" if not FAIL else f"FAILED {FAIL}")
