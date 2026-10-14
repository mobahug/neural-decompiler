"""Check 4: the launcher's main block (``_signal_to_interrupt`` and everything from ``record: dict = ...`` to the end),
extracted with ast from the reviewed copy and executed UNCHANGED in a bare namespace, with a stub ``preflight`` whose
runner is a ``Runner`` of the real ``run.py`` pointed at a scratch git repository (its real ``_write``,
``_record_incident`` and ``_preserve_partial``; ``_recheck`` stubbed fast), and a stub ``confirm`` that follows the real
runner's structure after its ledger (run.py:810–887): ledger write, a ``try`` whose ``except BaseException`` records the
incident, preserves the partial tensors and re-raises an interruption. ``_git`` is the launcher's own, at the scratch
repository. One scenario per process (argv[1]); the process exit code is the launcher's. Nothing touches the repository.
"""
import ast
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LAUNCHER, REVIEW, check_launcher, install_guard  # noqa: E402

check_launcher()
scenario = sys.argv[1]
base = REVIEW / "main_block" / scenario
here = base / "launcher_dir"
root = base / "repo"
outputs = root / "outputs/experiment-101"
initial_sighup = signal.getsignal(signal.SIGHUP)
install_guard()

import torch  # noqa: E402

from neural_decompiler import readout_decompilation as rd  # noqa: E402
from neural_decompiler import readout_routing as rr  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment_101_runner_review", Path("/Users/gaborhorvath-ulenius/myprojects/neural-decompiler/experiments/101-read-component-intervention/run.py"))
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)

S2 = [f"coordinated-adjective-101-{i:02d}|ref|581|REF" for i in range(1, 4)]  # labels only; nothing runs
logs: list = []
notes = {"initial_sighup": str(initial_sighup)}


class StubRunner(runner_module.Runner):
    def _recheck(self):
        return {"ok": True}

    def confirm(self):
        notes["sighup_inside_confirm"] = str(signal.getsignal(signal.SIGHUP))
        notes["sigterm_inside_confirm"] = str(signal.getsignal(signal.SIGTERM))
        if scenario == "refusal_before_ledger":
            raise runner_module.PhaseError("confirm already started or recorded an incident")
        state = rd.load_results_state(self.results_path)
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
                if scenario in ("sigterm_after_ledger", "sighup_under_nohup") and len(executed) == 2:
                    os.kill(os.getpid(), signal.SIGTERM if scenario == "sigterm_after_ledger" else signal.SIGHUP)
                    time.sleep(2)  # the handler raises here
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
        self._write(state)
        if scenario == "refusal_swallowed":
            G["events"]["refused"].append("a refusal some code swallowed")
        if scenario == "keys_not_all_run":
            G["done_keys"].discard(S2[-1])
        return 0


def git(*args):
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def preflight():
    if scenario == "preflight_refuses":
        raise RuntimeError("preflight failed: N1_tree_clean (None)")
    runner = StubRunner(root=root, results_path=outputs / "results.json", log=lambda m: (print(m, flush=True), logs.append(m)),
                        git_state=lambda: {"commit": "0" * 40, "dirty": False}, versions=lambda: {})
    return {"checks": {"stub": True}, "runner": runner, "logs": logs, "s2": list(S2)}


tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
first = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.FunctionDef) and n.name == "_signal_to_interrupt")
git_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_git")
main_nodes = [git_def] + tree.body[first:]
G = {"__name__": "launch_confirm_main_block", "datetime": datetime, "json": json, "os": os, "sys": sys, "signal": signal, "Path": Path, "subprocess": subprocess,
     "MODE": "production", "ROOT": root, "HERE": here, "OUTPUTS": outputs, "LAUNCHER_SHA256": "f" * 64, "environment": {"stub": True},
     "events": {"mode": "production", "load_model": 1, "refused": [], "forbidden_reads": [], "forbidden_writes": [], "outputs_reads": [], "sanctioned_runs": 0,
                "sanctioned_forward_calls": 0, "confirm_invocations": 0, "signals": []},
     "allowed_keys": set(), "done_keys": set(), "sealing": {}, "aliases": [], "_sweep_aliases": lambda: [], "preflight": preflight, "smoke": None, "rd": rd,
     "run_log": open(here / "production_runs.jsonl", "a", buffering=1, encoding="utf-8")}
if scenario == "git_fails_after":
    def failing_git(*args):
        raise subprocess.CalledProcessError(128, ["git", *args])
    G["_git_override"] = failing_git
if scenario == "sigterm_during_tree_after":
    notes["note"] = "SIGTERM arrives while the launcher runs `git status` for tree_after (after confirm returned 0)"
code = compile(ast.Module(body=main_nodes, type_ignores=[]), "launch_confirm_main", "exec")
# _git is defined by the extracted source; for two scenarios wrap it (after definition) through a tiny trampoline
exec(compile(ast.Module(body=[git_def], type_ignores=[]), "launch_confirm_git", "exec"), G)
real_git = G["_git"]
if scenario == "git_fails_after":
    G["_git"] = G["_git_override"]
if scenario == "sigterm_during_tree_after":
    def git_with_signal(*args):
        if args and args[0] == "status":
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(2)
        return real_git(*args)
    G["_git"] = git_with_signal
main_only = compile(ast.Module(body=tree.body[first:], type_ignores=[]), "launch_confirm_main", "exec")
(base / "notes.json").write_text(json.dumps(notes, indent=1) + "\n")
try:
    exec(main_only, G)
finally:
    (base / "notes.json").write_text(json.dumps(notes, indent=1) + "\n")
