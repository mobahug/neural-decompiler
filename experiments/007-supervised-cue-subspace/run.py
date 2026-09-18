#!/usr/bin/env python3
"""Run Experiment 007 through isolated phases.

``validate`` checks the frozen inputs without a model: the manifest, the Experiment 005
extension, Experiment 006's ``confirmation-v1.json`` (read in place; its digest must equal
the frozen constant), and the committed extract of Experiment 006's exposed E-patch
responses. The scientific phases ``explore``, ``calibrate``, ``lock``, ``confirm``, and
``report`` are enforced in order through the results state; fresh nouns, fresh frames, and
fresh cue tokens are executed only by ``confirm`` after the committed preregistration lock
validates. There is no freeze phase: the confirmation set is inherited.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs/experiment-007"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
PARAMETERS_DIR = OUTPUT_DIR / "parameters"
PROGRAM_PATH = EXPERIMENT_DIR / "linear_cue_program.py"
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")
PROGRAM_005_PATH = ROOT / "experiments/005-regular-plural-mechanism/mechanism_program.py"
LOCK_005_PATH = ROOT / cd.EXPERIMENT_005_LOCK_PATH
RESULTS_006_PATH = ROOT / "outputs/experiment-006/results.json"
PHASES = ("validate", "explore", "calibrate", "lock", "confirm", "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the manifest, extension, inherited confirmation set, and inherited response extract without a model"),
                             ("explore", "Tier A: E-patch responses replicated against Experiment 006, LOCO tables, rank selection, quality gate"),
                             ("calibrate", "Tier B: record τ and the comparison (refuses if the quality gate failed)"),
                             ("lock", "write outputs/experiment-007/candidate-lock.json for review"),
                             ("confirm", "Tier C: the single inherited-set run; requires the committed lock"),
                             ("report", "render outputs/experiment-007/report.md")):
        phases.add_parser(phase, help=help_text)
    return parser


def _git_tracked(path: Path) -> bool:
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _git_changed_paths(commit: str) -> list[str] | None:
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True, capture_output=True)
        diff = subprocess.run(["git", "diff", "--name-only", commit, "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in diff.stdout.splitlines() if line.strip()]


def _run_contract_test() -> dict[str, Any]:
    environment = dict(os.environ, NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE="1")
    completed = subprocess.run([sys.executable, "-m", "pytest", *CONTRACT_TEST], cwd=ROOT, env=environment, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    return {"returncode": completed.returncode, "passed": completed.returncode == 0, "output_tail": output[-4000:], "output_sha256": pm.sha256_text(output)}


def runtime_record(spec: ModelSpec) -> dict[str, Any]:
    runtime = validate_runtime(spec)
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": ss.RUNTIME_SEED, "deterministic_algorithms": spec.deterministic_algorithms,
            "torch_num_threads": int(torch.get_num_threads())}


@dataclass
class Runner:
    root: Path = ROOT
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    parameters_dir: Path = PARAMETERS_DIR
    program_path: Path = PROGRAM_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    git_state: Callable[[], Mapping[str, Any]] = field(default_factory=lambda: lambda: collect_git_state(ROOT))
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    changed_paths: Callable[[str], list[str] | None] = _git_changed_paths
    contract_runner: Callable[[], Mapping[str, Any]] = _run_contract_test
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))
    program_005_path: Path = PROGRAM_005_PATH
    lock_005_path: Path = LOCK_005_PATH
    results_006_path: Path = RESULTS_006_PATH
    circuit: pm.MechanismSet = cd.FIXED_CIRCUIT

    @property
    def confirmation_path(self) -> Path:
        return self.root / ss.CONFIRMATION_RELATIVE_PATH

    @property
    def inherited_path(self) -> Path:
        return self.root / ss.INHERITED_EXTRACT_RELATIVE_PATH

    # -- inputs ---------------------------------------------------------------

    def _inputs(self):
        manifest, manifest_sha256, extension = pm.load_inputs(self.root)
        if not self.confirmation_path.exists():
            raise ss.PhaseError(f"missing inherited confirmation set {self.confirmation_path}")
        if not self.tracked(self.confirmation_path) or not self.tracked(self.inherited_path):
            raise ss.PhaseError("the inherited confirmation set and the inherited response extract must be tracked and committed")
        confirmation = cd.load_confirmation(self.confirmation_path, manifest, manifest_sha256, extension)
        if confirmation.content_sha256 != ss.INHERITED_CONFIRMATION_SHA256:
            raise ss.PhaseError(f"the confirmation set digest {confirmation.content_sha256} is not the frozen Experiment 006 digest")
        inherited = ss.load_inherited_extract(self.inherited_path, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation.content_sha256)
        return manifest, manifest_sha256, extension, confirmation, inherited

    def _local_source_record(self, inherited: Mapping[str, Any]) -> dict[str, Any]:
        """Informational: whether the gitignored Experiment 006 results file on this machine is the extract's source."""
        if not self.results_006_path.exists():
            return {"present": False}
        digest = hashlib.sha256(self.results_006_path.read_bytes()).hexdigest()
        return {"present": True, "file_sha256": digest, "matches_extract_source": digest == inherited["source"]["file_sha256"]}

    def validate(self) -> int:
        try:
            manifest, manifest_sha256, extension, confirmation, inherited = self._inputs()
        except (ss.PhaseError, ValueError) as error:
            self.log(f"validation failed: {error}")
            return 1
        pool = cd.exposed_pool(manifest, extension)
        self.log(f"exposed pool: {len(pool.frames)} frames, {len(pool.tokens)} cue tokens, {len(pool.nouns)} nouns")
        self.log(f"inherited confirmation ok: sha256 {confirmation.content_sha256}; {len(confirmation.nouns)} nouns; {len(confirmation.frames)} frames; {len(confirmation.tokens)} tokens")
        local = self._local_source_record(inherited)
        self.log(f"inherited extract ok: sha256 {inherited['content_sha256']}; {len(inherited['responses'])} responses; source state {inherited['source']['state_sha256'][:16]}…; local source {local}")
        return 0

    # -- scientific phases ---------------------------------------------------

    def _provenance(self) -> dict[str, Any]:
        git = self.git_state()
        return {"protocol_code_commit": str(git.get("commit") or ""), "git_dirty": bool(git.get("dirty", True)), "versions": dict(self.versions())}

    def _state_for(self, phase: str, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str) -> dict[str, Any]:
        provenance = self._provenance()
        if self.results_path.exists():
            state = cd.load_results_state(self.results_path)
            cd.assert_provenance_identical(state, manifest_sha256=manifest_sha256, extension_sha256=extension_sha256, confirmation_sha256=confirmation_sha256, **provenance)
        else:
            if phase != "explore":
                raise ss.PhaseError(f"{phase} requires an existing results state from explore")
            state = cd.new_results_state(manifest_sha256=manifest_sha256, extension_sha256=extension_sha256, confirmation_sha256=confirmation_sha256, **provenance)
        cd.assert_phase_allowed(phase, state)
        return state

    def _lock_005(self) -> dict[str, Any]:
        return json.loads(self.lock_005_path.read_text(encoding="utf-8")) if self.lock_005_path.exists() else {}

    def _record_incident(self, state: dict[str, Any], phase: str, error: Exception) -> None:
        entry = {"message": str(error), "at": pm.utc_now(), "phase": phase, "commit": self._provenance()["protocol_code_commit"]}
        if getattr(error, "payload", None):
            entry["invalidated"] = dict(error.payload)  # the invalidated measurements stay with the record
        if phase == "explore":
            state["exploration"].setdefault("incidents", []).append(entry)
        else:
            state["confirmation"] = {"incident": entry}
        cd.write_results_state(self.results_path, state)
        self.log(f"INCIDENT ({phase}): {error}")

    def explore(self) -> int:
        manifest, manifest_sha256, extension, confirmation, inherited = self._inputs()
        state = self._state_for("explore", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        commit = self._provenance()["protocol_code_commit"]
        incidents = state["exploration"].get("incidents", [])
        if incidents and incidents[-1]["commit"] == commit:
            raise ss.PhaseError("an explore incident is recorded at this commit; a committed fix or documented amendment is required before explore runs again")
        contract = dict(self.contract_runner())
        state["exploration"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            cd.write_results_state(self.results_path, state)
            self.log("A0 contract test failed; refusing to explore")
            return 1
        pool = cd.exposed_pool(manifest, extension)
        state["exploration"]["inherited_source_local"] = self._local_source_record(inherited)
        state["protocol_code_commit"] = commit  # the commit of the attempt that produced the recorded data
        state["phases"]["explore"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime_record(PYTHIA_70M),
                                      "attempts": int(state["phases"]["explore"].get("attempts", 0)) + 1,
                                      "attempt_commits": list(state["phases"]["explore"].get("attempt_commits", [])) + [commit]}
        pm.record_execution(state, pm.manifest_prompts(pool.frames) + tuple(pool.reference_prompt(frame) for frame in pool.frames), pool.nouns)
        cd.assert_confirmation_untouched(state, confirmation)
        cd.write_results_state(self.results_path, state)
        seed_runtime(ss.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            development = pm.nouns_for(manifest, pm.Split.DEVELOPMENT)
            factory = lambda: pm.DiscoveryContext(model, pm.Weights.from_model(model), manifest, pool.manifest_frames, development, pm.PromptCache(model, development))
            try:
                ss.run_exploration(model, pool, state=state, results_path=self.results_path, parameters_dir=self.parameters_dir, program_path=self.program_path,
                                   program_005_path=self.program_005_path, lock_005=self._lock_005(), inherited=inherited, circuit=self.circuit,
                                   development_ctx_factory=factory, log=self.log)
            except pm.IncidentError as error:
                self._record_incident(state, "explore", error)
                return 2
        finally:
            del model
            gc.collect()
        cd.assert_confirmation_untouched(state, confirmation)
        state["phases"]["explore"] = {**state["phases"]["explore"], "status": "complete", "completed_at": pm.utc_now()}
        digest = cd.write_results_state(self.results_path, state)
        selection, gate = state["exploration"]["selection"], state["exploration"]["quality_gate"]
        self.log(f"explore complete: selected rank {selection['selected']}, quality gate {'passed' if gate['passed'] else 'FAILED'}, tau {state['exploration']['tau']:.3f}; results sha256 {digest}")
        return 0

    def calibrate(self) -> int:
        manifest, manifest_sha256, extension, confirmation, inherited = self._inputs()
        state = self._state_for("calibrate", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        exploration = state["exploration"]
        if not exploration["quality_gate"]["passed"]:
            raise ss.PhaseError("the pre-lock quality gate failed; Experiment 007 ended at Tier A with QUALITY_GATE_FAILED")
        state["calibration"] = {"tau": exploration["tau"], "quality_gate_passed": True, "comparison": exploration["comparison"], "completed_at": pm.utc_now()}
        state["phases"]["calibrate"] = {"status": "complete", "completed_at": pm.utc_now()}
        cd.write_results_state(self.results_path, state)
        self.log(f"calibration recorded: tau {exploration['tau']:.3f}")
        return 0

    def lock(self) -> int:
        manifest, manifest_sha256, extension, confirmation, inherited = self._inputs()
        state = self._state_for("lock", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        programs = ss.load_programs(self.parameters_dir, self.program_path, self.program_005_path, confirmation.reference_ids)
        lock = ss.build_candidate_lock(state=state, manifest=manifest, manifest_sha256=manifest_sha256, extension=extension, confirmation=confirmation, inherited=inherited, programs=programs,
                                       parameters_dir=self.parameters_dir, program_path=self.program_path, program_005_path=self.program_005_path,
                                       protocol_code_commit=self._provenance()["protocol_code_commit"])
        candidate_path = self.results_path.parent / "candidate-lock.json"
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        state["lock"] = {"candidate_path": str(candidate_path), "content_sha256": lock["content_sha256"], "rank": lock["rank"], "tau": lock["tau"], "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now()}
        cd.write_results_state(self.results_path, state)
        self.log(f"candidate lock written to {candidate_path} (sha256 {lock['content_sha256']}). Install it as {ss.LOCK_RELATIVE_PATH} and commit before confirm.")
        return 0

    def confirm(self) -> int:
        manifest, manifest_sha256, extension, confirmation, inherited = self._inputs()
        state = self._state_for("confirm", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        lock_path = self.root / ss.LOCK_RELATIVE_PATH
        if not lock_path.exists():
            raise ss.PhaseError(f"missing committed lock {lock_path}")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        git = self.git_state()
        ss.validate_lock(lock, state=state, manifest_sha256=manifest_sha256, extension=extension, confirmation=confirmation, inherited=inherited, parameters_dir=self.parameters_dir,
                         program_path=self.program_path, program_005_path=self.program_005_path, git_state=git, tracked=self.tracked(lock_path), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        pool = cd.exposed_pool(manifest, extension)
        programs = ss.load_programs(self.parameters_dir, self.program_path, self.program_005_path, confirmation.reference_ids)
        reproduced = ss.assert_lock_predictions_reproduced(programs, confirmation, lock)  # PhaseError before anything fresh runs
        state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": str(git.get("commit")),
                                      "lock_predictions_reproduced_max_difference": reproduced}
        pm.record_execution(state, confirmation.all_prompts + tuple(confirmation.reference_prompt(frame) for frame in confirmation.frames) + pm.manifest_prompts(pool.manifest_frames), confirmation.nouns)
        cd.write_results_state(self.results_path, state)
        seed_runtime(ss.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            try:
                results = ss.run_confirmation(model, pool, confirmation, programs, lock, circuit=self.circuit, axes_dir=self.parameters_dir / "e005-scalar", log=self.log)
            except pm.IncidentError as error:
                self._record_incident(state, "confirm", error)  # the phase stays "running": confirm never re-runs in this protocol version
                return 2
        finally:
            del model
            gc.collect()
        state["confirmation"] = {**results, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}
        state["phases"]["confirm"] = {**state["phases"]["confirm"], "status": "complete", "completed_at": pm.utc_now()}
        digest = cd.write_results_state(self.results_path, state)
        self.log(f"confirm complete: {results['outcome']['label']}; results sha256 {digest}")
        return 0

    def report(self) -> int:
        manifest, manifest_sha256, extension, confirmation, inherited = self._inputs()
        if self.results_path.exists():
            state = cd.load_results_state(self.results_path)
            if state["phases"]["explore"]["status"] != "complete" and state["exploration"].get("incidents"):
                self.report_path.parent.mkdir(parents=True, exist_ok=True)
                self.report_path.write_text(ss.render_report(state), encoding="utf-8")
                self.log(f"incident report written to {self.report_path}")
                return 0
        state = self._state_for("report", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(ss.render_report(state), encoding="utf-8")
        self.log(f"report written to {self.report_path}")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    if args.phase in PHASES:
        return getattr(runner, args.phase)()
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
