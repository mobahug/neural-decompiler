"""v2 diff-check, check 4: the v2 launcher's module-level SIGHUP statement and its main block (``_signal_to_interrupt``
and everything from ``record: dict = ...`` to the end), extracted with ast from the reviewed v2 copy and executed
UNCHANGED in a bare namespace — the SIGHUP statement first, as at the top of the launcher. The stub ``preflight``'s runner
is a ``Runner`` of the real ``run.py`` pointed at a scratch git repository (its real ``_write``, ``_record_incident`` and
``_preserve_partial``; ``_recheck`` stubbed fast); its stub ``confirm`` follows the real runner's structure (the barrier
region before the ledger, the ledger, the runs' ``try``/``except BaseException``, the result write, a ``_secondary``-like
step that records a failure and re-raises an interruption). Nothing runs a model; nothing touches the repository.
One scenario per process (argv[1]); the process exit code is the launcher's.
"""
import ast
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import REVIEW, install_guard  # noqa: E402

V2 = REVIEW / "reviewed_v2/launch_confirm_v2.py"
scenario = sys.argv[1]
base = REVIEW / "v2/main_block" / scenario
here = base / "launcher_dir"
root = base / "repo"
outputs = root / "outputs/experiment-101"
notes = {"initial_sighup": str(signal.getsignal(signal.SIGHUP)), "initial_sigterm": str(signal.getsignal(signal.SIGTERM)),
         "stdout_isatty": os.isatty(1), "stderr_isatty": os.isatty(2)}
tree = ast.parse(V2.read_text(encoding="utf-8"))
sighup_nodes = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and "SIGHUP" in ast.unparse(n)]
assert len(sighup_nodes) == 1, [ast.unparse(n) for n in sighup_nodes]
exec(compile(ast.Module(body=sighup_nodes, type_ignores=[]), "launch_confirm_v2_top", "exec"), {"signal": signal})  # as at launch_confirm.py:83
notes["sighup_after_top_statement"] = str(signal.getsignal(signal.SIGHUP))
install_guard()

import torch  # noqa: E402

from neural_decompiler import readout_decompilation as rd  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment_101_runner_review", Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/experiments/101-read-component-intervention/run.py"))
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)

S2 = [f"coordinated-adjective-101-{i:02d}|ref|581|REF" for i in range(1, 4)]  # labels only; nothing runs
logs: list = []


def self_signal(signum):
    os.kill(os.getpid(), signum)
    time.sleep(1.5)  # a Python handler raises here; an ignored signal just lets the sleep finish


class StubRunner(runner_module.Runner):
    def _recheck(self):
        return {"ok": True}

    def confirm(self):
        notes["sighup_inside_confirm"] = str(signal.getsignal(signal.SIGHUP))
        notes["sigterm_inside_confirm"] = str(signal.getsignal(signal.SIGTERM))
        if scenario == "refusal_before_ledger":
            raise runner_module.PhaseError("confirm already started or recorded an incident")
        state = rd.load_results_state(self.results_path)
        if scenario == "sigterm_before_ledger":
            self_signal(signal.SIGTERM)  # the model load / barrier region: no except catches it here (run.py:797-809)
        state["phases"]["confirm"] = {"status": "running", "confirm_commit": "0" * 40}
        state["executed_prompt_keys"] = sorted(set(state["executed_prompt_keys"]) | set(S2))
        state["confirmation"] = {"barrier": {"passed": True}}
        self._write(state)  # the ledger
        executed, tensors = [], {}
        try:
            for key in S2:
                G["done_keys"].add(key)
                executed.append(key)
                tensors[key] = torch.zeros(4)
                if len(executed) == 2:
                    if scenario == "sigterm_after_ledger":
                        self_signal(signal.SIGTERM)
                    elif scenario in ("sighup_under_nohup", "sighup_without_nohup"):
                        self_signal(signal.SIGHUP)
            if scenario == "incident_after_ledger":
                raise runner_module.IncidentError("integrity failure after the measurements: gates ['G4_level1'] failed")
        except BaseException as error:
            self._record_incident(state, "confirm", error)
            self._preserve_partial(state, tensors, executed, "stage2-partial.pt", state["confirmation"])
            if isinstance(error, Exception):
                return 2
            raise
        state["confirmation"].update({"results": {"outcome": {"label": "READ_COMPONENT_EFFECTS_AS_PREDICTED"}}, "accounting": {"equal": True}})
        state["phases"]["confirm"] = {**state["phases"]["confirm"], "status": "complete"}
        self._write(state)  # the result
        secondary = state["confirmation"].setdefault("secondary", {})
        try:  # the _secondary pattern (run.py:897-907)
            if scenario == "sigterm_during_secondary":
                self_signal(signal.SIGTERM)
            secondary["statistics"] = {"stub": True}
        except BaseException as error:
            secondary.setdefault("failures", {})["statistics"] = {"type": type(error).__name__, "message": str(error)}
            self._write(state)
            if not isinstance(error, Exception):
                raise
        self._write(state)
        if scenario == "refusal_swallowed":
            G["events"]["refused"].append("a refusal some code swallowed")
        if scenario == "forbidden_write_recorded":
            G["events"]["forbidden_writes"].append("a refused write some code swallowed")
        if scenario == "keys_not_all_run":
            G["done_keys"].discard(S2[-1])
        if scenario == "sigterm_in_post_block":
            notes["post_block_signal"] = "SIGTERM sent while the launcher reads the state for its post block"
        return 0


def preflight():
    if scenario == "preflight_refuses":
        raise RuntimeError("preflight failed: N1_tree_clean (None)")
    runner = StubRunner(root=root, results_path=outputs / "results.json", log=lambda m: (print(m, flush=True), logs.append(m)),
                        git_state=lambda: {"commit": "0" * 40, "dirty": False}, versions=lambda: {})
    return {"checks": {"stub": True}, "runner": runner, "logs": logs, "s2": list(S2)}


def barrier_dry_run(pre):  # rehearsal is not exercised here (see v2/g2_*); a stub keeps the namespace complete
    return {"ok": True, "stub": True}


first = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.FunctionDef) and n.name == "_signal_to_interrupt")
git_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_git")
sha_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sha256_file")
rd_proxy = types.SimpleNamespace(load_results_state=rd.load_results_state)
if scenario == "sigterm_in_post_block":
    def load_with_signal(path):
        if notes.get("post_block_signal"):
            self_signal(signal.SIGTERM)  # after confirm() returned: SIGTERM is SIG_IGN now
        return rd.load_results_state(path)
    rd_proxy.load_results_state = load_with_signal
G = {"__name__": "launch_confirm_v2_main_block", "datetime": datetime, "json": json, "os": os, "sys": sys, "signal": signal, "Path": Path, "subprocess": subprocess,
     "hashlib": __import__("hashlib"), "MODE": os.environ.get("CONFIRM101_MODE", "production"), "ROOT": root, "HERE": here, "OUTPUTS": outputs,
     "LAUNCHER_SHA256": "f" * 64, "environment": {"stub": True},
     "events": {"mode": "production", "load_model": 1, "refused": [], "forbidden_reads": [], "forbidden_writes": [], "outputs_reads": [], "sanctioned_runs": 0,
                "sanctioned_forward_calls": 0, "confirm_invocations": 0, "signals": []},
     "allowed_keys": set(), "done_keys": set(), "sealing": {}, "aliases": [], "_sweep_aliases": lambda: [], "preflight": preflight, "smoke": None,
     "barrier_dry_run": barrier_dry_run, "rd": rd_proxy, "run_log": open(here / "production_runs.jsonl", "a", buffering=1, encoding="utf-8")}
exec(compile(ast.Module(body=[git_def, sha_def], type_ignores=[]), "launch_confirm_v2_git", "exec"), G)
real_git = G["_git"]
if scenario == "git_fails_after":
    def failing_git(*args):
        raise subprocess.CalledProcessError(128, ["git", *args])
    G["_git"] = failing_git
if scenario == "sigterm_during_tree_after":
    def git_with_signal(*args):
        if args and args[0] == "status":
            self_signal(signal.SIGTERM)
        return real_git(*args)
    G["_git"] = git_with_signal
main_only = compile(ast.Module(body=tree.body[first:], type_ignores=[]), "launch_confirm_v2_main", "exec")
(base / "notes.json").write_text(json.dumps(notes, indent=1) + "\n")
try:
    exec(main_only, G)
finally:
    notes["sigterm_at_exit"] = str(signal.getsignal(signal.SIGTERM))
    (base / "notes.json").write_text(json.dumps(notes, indent=1) + "\n")
