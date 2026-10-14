"""The launcher v2's production main block (from ``record: dict = ...`` to the end, extracted with ``ast`` and executed
unchanged) against a STUB runner, in a scratch git repository with the repository's ignore rules for the stage-1
paths, to check the post-run checks, the record (O_EXCL), the sentinel and the exit code in each scenario. Nothing of
the real runner, model or repository is used; the real ``rr.write_state_atomic``, ``rd.load_results_state`` and
``b0c.stage_one_digest`` are. Output: out/e_main_block.json.
"""
import ast
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAUNCHER_V2, LAUNCHER_V2_SHA256, RECHECK, extract_launcher, install_guard, write_json  # noqa: E402

assert hashlib.sha256(LAUNCHER_V2.read_bytes()).hexdigest() == LAUNCHER_V2_SHA256
from neural_decompiler import block0_completion as b0c  # noqa: E402
from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402

install_guard()
src = LAUNCHER_V2.read_text(encoding="utf-8")
tree = ast.parse(src)
start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "record")
main_block = compile(ast.Module(body=tree.body[start:], type_ignores=[]), "launch_stage1_main", "exec")
git_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_git")
sha_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sha256_file")

BASE = RECHECK / "main_block"
if BASE.exists():
    shutil.rmtree(BASE)
KEYS = sorted(f"101-{i:02d}|ref|581|S1" for i in range(36))


class Exit(Exception):
    def __init__(self, code):
        self.code = code


def git(root, *args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def make_repo(name):
    root = (BASE / name / "repo")
    (root / "outputs/experiment-101").mkdir(parents=True)
    (root / "experiments/101-read-component-intervention").mkdir(parents=True)
    (root / ".gitignore").write_text(".DS_Store\noutputs/experiment-101/*\n")
    (root / "experiments/101-read-component-intervention/README.md").write_text("x\n")
    git(root, "init", "-q")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    root = root.resolve()
    here = (BASE / name / "here")
    here.mkdir()
    rr.write_state_atomic(root / "outputs/experiment-101/results.json",
                          {"experiment": "101", "phases": {"stage1": {"status": "not_started"}}, "executed_prompt_keys": [], "stage1": None})
    return root, here.resolve()


def run_scenario(name, behaviour, *, preflight_raises=None, self_test_ok=True, sentinel_exists=False, extra_file=False, tamper=None, records_before=0):
    root, here = make_repo(name)
    L = extract_launcher(root, "production")
    outputs, stage1_dir, files = L["OUTPUTS"], L["STAGE1_DIR"], L["STAGE1_FILES"]
    allowed_keys, done_keys = set(), set()
    calls = {"stage1": 0}

    class StubRunner:
        def stage1(self):
            calls["stage1"] += 1
            if behaviour == "refuse_before_ledger":
                raise RuntimeError("PhaseError (stub): a refusal before the ledger")
            state = rd.load_results_state(outputs / "results.json")
            state = {k: v for k, v in state.items() if k != "state_sha256"}
            state["executed_prompt_keys"] = sorted(KEYS)
            state["phases"]["stage1"] = {"status": "running"}
            rr.write_state_atomic(outputs / "results.json", state)  # the ledger
            done_keys.update(allowed_keys)  # the 36 sanctioned captures
            if behaviour == "interrupt_after_ledger":
                state["phases"]["stage1"] = {**state["phases"]["stage1"], "incidents": [{"message": "KeyboardInterrupt", "after_ledger": True}]}
                rr.write_state_atomic(outputs / "results.json", state)
                raise KeyboardInterrupt()
            if behaviour == "incident_before_files":
                state["phases"]["stage1"] = {**state["phases"]["stage1"], "incidents": [{"message": "PermissionError (stub)", "after_ledger": True}]}
                rr.write_state_atomic(outputs / "results.json", state)
                return 2
            stage1_dir.mkdir(exist_ok=True)
            payloads = {"states_data": (stage1_dir / "states.f64", b"\0" * 64), "states_index": (stage1_dir / "states.json", b"{}\n"),
                        "table": (stage1_dir / "table.json", b'{"content_sha256": "t"}\n')}
            for path, data in payloads.values():
                path.write_bytes(data)
            record = {"files": {k: {"path": str(p.relative_to(root)), "sha256": hashlib.sha256(d).hexdigest()} for k, (p, d) in payloads.items()},
                      "table_content_sha256": "t", "executed": 36}
            record["digest"] = b0c.stage_one_digest(record)
            if tamper == "digest":
                record["digest"] = "0" * 64
            state["stage1"] = record
            state["phases"]["stage1"] = {**state["phases"]["stage1"], "status": "complete", "awaiting_commit": True}
            if behaviour == "complete_then_incident":  # the final write failed, then the incident write succeeded (run.py:721-724)
                state["phases"]["stage1"] = {**state["phases"]["stage1"], "incidents": [{"message": "OSError (stub): final write", "after_ledger": True}]}
                rr.write_state_atomic(outputs / "results.json", state)
                return 2
            rr.write_state_atomic(outputs / "results.json", state)
            if tamper == "file":
                (stage1_dir / "table.json").write_bytes(b"changed\n")
            if extra_file:
                (root / "experiments/101-read-component-intervention/stray.txt").write_text("x\n")
            return 0

    def preflight():
        if preflight_raises:
            raise RuntimeError(preflight_raises)
        return {"checks": {"stub": True}, "runner": StubRunner(), "logs": ["stub log"], "keys": list(KEYS)}

    printed = []
    fake_sys = types.SimpleNamespace(flags=types.SimpleNamespace(dont_write_bytecode=1), exit=lambda code: (_ for _ in ()).throw(Exit(code)))
    ns = {"MODE": "production", "HERE": here, "ROOT": root, "OUTPUTS": outputs, "STAGE1_FILES": files, "STAGE1_DIR": stage1_dir, "events": L["events"],
          "sealing": {}, "environment": {"stub": True}, "LAUNCHER_SHA256": LAUNCHER_V2_SHA256, "done_keys": done_keys, "allowed_keys": allowed_keys,
          "run_log": None, "aliases": [], "preflight": preflight, "_sweep_aliases": lambda: [],
          "hook_self_test": lambda: {"ok": self_test_ok}, "rd": rd, "b0c": b0c, "datetime": datetime, "json": json, "os": os, "sys": fake_sys,
          "subprocess": subprocess, "hashlib": hashlib, "Path": Path, "print": lambda *a, **k: printed.append(" ".join(str(x) for x in a))}
    exec(compile(ast.Module(body=[git_fn, sha_fn], type_ignores=[]), "fns", "exec"), ns)
    if sentinel_exists:
        (here / "stage1.sentinel").write_text("earlier")
    for i in range(records_before):
        (here / f"production_run_2026101{i}T000000000000Z.json").write_text("earlier record\n")
    code = None
    try:
        exec(main_block, ns)
    except Exit as e:
        code = e.code
    records = sorted(p.name for p in here.glob("production_run_*.json"))
    record = json.loads((here / records[-1]).read_text()) if records and records[-1].endswith(".json") and records[-1] != "production_run_20261010T000000000000Z.json" else None
    return {"scenario": name, "exit_code": code, "stage1_calls": calls["stage1"], "sentinel": (here / "stage1.sentinel").exists(),
            "sentinel_content_is_launcher_sha": (here / "stage1.sentinel").exists() and (here / "stage1.sentinel").read_text() == LAUNCHER_V2_SHA256,
            "records": records, "error": record and record.get("error"), "exit_status": record and record.get("exit_status"),
            "post": record and record.get("post"), "logs_in_record": record and record.get("logs"), "printed_complete": any("Stage 1 complete" in p for p in printed),
            "record_mode": oct((here / records[-1]).stat().st_mode & 0o777) if records else None,
            "git_status": git(root, "status", "--porcelain", "--untracked-files=all")}


scenarios = [
    run_scenario("a_correct", "correct"),
    run_scenario("b_incident_after_ledger_return2", "incident_before_files"),
    run_scenario("c_refusal_inside_stage1_before_ledger", "refuse_before_ledger"),
    run_scenario("d_interrupt_after_ledger", "interrupt_after_ledger"),
    run_scenario("e_correct_but_stray_file", "correct", extra_file=True),
    run_scenario("f_digest_tampered", "correct", tamper="digest"),
    run_scenario("g_file_changed_after_record", "correct", tamper="file"),
    run_scenario("h_complete_then_incident", "complete_then_incident"),
    run_scenario("i_sentinel_exists", "correct", sentinel_exists=True),
    run_scenario("j_preflight_refuses", "correct", preflight_raises="preflight failed: N1_tree_clean (stub)"),
    run_scenario("k_self_test_fails", "correct", self_test_ok=False),
    run_scenario("l_earlier_records_kept", "correct", records_before=2),
]
write_json("e_main_block.json", scenarios)
for s in scenarios:
    post = s["post"] or {}
    flags = {k: post.get(k) for k in ("all_36_ran_once", "ledger_is_exactly_the_36_s1_keys", "tree_is_exactly_the_three_untracked_stage1_files",
                                      "record_digest_reproduces", "files_equal_record")} if post else None
    print(f"{s['scenario']:<42} exit={s['exit_code']!s:<4} stage1_calls={s['stage1_calls']} sentinel={s['sentinel']!s:<5} "
          f"records={len(s['records'])} complete_msg={s['printed_complete']!s:<5} error={str(s['error'])[:60]!s}")
    if flags:
        print(f"{'':<42} post={flags} logs={'yes' if s['logs_in_record'] else 'no'}")
