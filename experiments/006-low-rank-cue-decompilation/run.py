#!/usr/bin/env python3
"""Run Experiment 006 through isolated phases.

``validate`` checks the frozen inputs without a model. ``freeze-confirmation``
builds ``confirmation-v1.json`` from tokenizer rules only and refuses to
overwrite it. The scientific phases ``explore``, ``calibrate``, ``lock``,
``confirm``, and ``report`` are enforced in order through the results state;
fresh nouns, fresh frames, and fresh cue tokens are executed only by ``confirm``
after the committed preregistration lock validates.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import plural_mechanism as pm
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs/experiment-006"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
PARAMETERS_DIR = OUTPUT_DIR / "parameters"
PROGRAM_PATH = EXPERIMENT_DIR / "low_rank_program.py"
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")
PROGRAM_005_PATH = ROOT / "experiments/005-regular-plural-mechanism/mechanism_program.py"
LOCK_005_PATH = ROOT / cd.EXPERIMENT_005_LOCK_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("validate", help="check the manifest, extension, and frozen confirmation set without a model")
    phases.add_parser("freeze-confirmation", help="build confirmation-v1.json from tokenizer rules only (refuses to overwrite)")
    for phase, help_text in (("explore", "Tier A: circuit verification, E-patch responses, rank selection, quality gate"), ("calibrate", "Tier B: bands from the LOCO residuals"),
                             ("lock", "write outputs/experiment-006/candidate-lock.json for review"), ("confirm", "Tier C: the single fresh-set run; requires the committed lock"),
                             ("report", "render outputs/experiment-006/report.md")):
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
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": cd.RUNTIME_SEED, "deterministic_algorithms": spec.deterministic_algorithms}


def _load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


@dataclass
class Runner:
    root: Path = ROOT
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    parameters_dir: Path = PARAMETERS_DIR
    program_path: Path = PROGRAM_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    tokenizer_loader: Callable[[ModelSpec], Any] = _load_tokenizer
    git_state: Callable[[], Mapping[str, Any]] = field(default_factory=lambda: lambda: collect_git_state(ROOT))
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    changed_paths: Callable[[str], list[str] | None] = _git_changed_paths
    contract_runner: Callable[[], Mapping[str, Any]] = _run_contract_test
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))
    program_005_path: Path = PROGRAM_005_PATH
    lock_005_path: Path = LOCK_005_PATH
    circuit: pm.MechanismSet = cd.FIXED_CIRCUIT

    @property
    def confirmation_path(self) -> Path:
        return self.root / cd.CONFIRMATION_RELATIVE_PATH

    def _base_inputs(self):
        return pm.load_inputs(self.root)

    def validate(self) -> int:
        manifest, manifest_sha256, extension = self._base_inputs()
        pool = cd.exposed_pool(manifest, extension)
        self.log(f"exposed pool: {len(pool.frames)} frames, {len(pool.tokens)} cue tokens, {len(pool.nouns)} nouns, {len(pool.cue_word_prompts)} cue-word prompts")
        if not self.confirmation_path.exists():
            self.log("confirmation set not frozen yet: run freeze-confirmation")
            return 1
        confirmation = cd.load_confirmation(self.confirmation_path, manifest, manifest_sha256, extension)
        tracked = self.tracked(self.confirmation_path)
        self.log(f"confirmation ok: sha256 {confirmation.content_sha256}; {len(confirmation.nouns)} nouns; {len(confirmation.frames)} frames; "
                 f"{len(confirmation.tokens)} tokens; {len(confirmation.token_prompts)} token prompts; tracked={tracked}")
        return 0 if tracked else 1

    def freeze_confirmation(self) -> int:
        manifest, manifest_sha256, extension = self._base_inputs()
        if self.confirmation_path.exists():
            self.log(f"refusing to overwrite frozen confirmation set {self.confirmation_path}")
            return 1
        digest = cd.freeze_confirmation(self.confirmation_path, self.tokenizer_loader(PYTHIA_70M), manifest, manifest_sha256, extension)
        self.log(f"froze confirmation set {self.confirmation_path} sha256 {digest}; commit it before any scientific phase")
        return 0


    # -- scientific phases ---------------------------------------------------

    def _provenance(self) -> dict[str, Any]:
        git = self.git_state()
        return {"protocol_code_commit": str(git.get("commit") or ""), "git_dirty": bool(git.get("dirty", True)), "versions": dict(self.versions())}

    def _inputs(self):
        manifest, manifest_sha256, extension = self._base_inputs()
        if not self.confirmation_path.exists() or not self.tracked(self.confirmation_path):
            raise cd.PhaseError("the confirmation set must be frozen and committed before any scientific phase")
        confirmation = cd.load_confirmation(self.confirmation_path, manifest, manifest_sha256, extension)
        return manifest, manifest_sha256, extension, confirmation

    def _state_for(self, phase: str, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str) -> dict[str, Any]:
        provenance = self._provenance()
        if self.results_path.exists():
            state = cd.load_results_state(self.results_path)
            cd.assert_provenance_identical(state, manifest_sha256=manifest_sha256, extension_sha256=extension_sha256, confirmation_sha256=confirmation_sha256, **provenance)
        else:
            if phase != "explore":
                raise cd.PhaseError(f"{phase} requires an existing results state from explore")
            state = cd.new_results_state(manifest_sha256=manifest_sha256, extension_sha256=extension_sha256, confirmation_sha256=confirmation_sha256, **provenance)
        cd.assert_phase_allowed(phase, state)
        return state

    def _lock_005(self) -> dict[str, Any]:
        return json.loads(self.lock_005_path.read_text(encoding="utf-8")) if self.lock_005_path.exists() else {}

    def explore(self) -> int:
        manifest, manifest_sha256, extension, confirmation = self._inputs()
        state = self._state_for("explore", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        contract = dict(self.contract_runner())
        state["exploration"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            self.log("A0 contract test failed; refusing to explore")
            return 1
        pool = cd.exposed_pool(manifest, extension)
        state["phases"]["explore"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime_record(PYTHIA_70M),
                                      "attempts": int(state["phases"]["explore"].get("attempts", 0)) + 1}
        cd.write_results_state(self.results_path, state)
        seed_runtime(cd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            pm.record_execution(state, pm.manifest_prompts(pool.frames) + tuple(pool.reference_prompt(frame) for frame in pool.frames), pool.nouns)
            cd.assert_confirmation_untouched(state, confirmation)
            development = pm.nouns_for(manifest, pm.Split.DEVELOPMENT)
            factory = lambda: pm.DiscoveryContext(model, pm.Weights.from_model(model), manifest, pool.manifest_frames, development, pm.PromptCache(model, development))
            lock_005 = self._lock_005()
            cd.run_exploration(model, pool, state=state, results_path=self.results_path, parameters_dir=self.parameters_dir, program_path=self.program_path,
                               program_005_path=self.program_005_path, e005_index_sha256=lock_005.get("parameters_index_sha256"), circuit=self.circuit,
                               development_ctx_factory=factory, log=self.log)
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
        manifest, manifest_sha256, extension, confirmation = self._inputs()
        state = self._state_for("calibrate", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        exploration = state["exploration"]
        bands_005 = self._lock_005().get("bands", {})
        state["calibration"] = {"tau": exploration["tau"], "circuit_bands_005": {family: bands_005.get(family) for family in ("P1", "P3", "P4", "P5", "P7", "P8", "P9")},
                                "quality_gate_passed": exploration["quality_gate"]["passed"], "completed_at": pm.utc_now()}
        state["phases"]["calibrate"] = {"status": "complete", "completed_at": pm.utc_now()}
        cd.write_results_state(self.results_path, state)
        self.log(f"calibration recorded: tau {exploration['tau']:.3f}; quality gate {'passed' if exploration['quality_gate']['passed'] else 'FAILED'}")
        return 0

    def lock(self) -> int:
        manifest, manifest_sha256, extension, confirmation = self._inputs()
        state = self._state_for("lock", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        if not state["exploration"]["quality_gate"]["passed"]:
            raise cd.PhaseError("the pre-lock quality gate failed; no lock may be written")
        programs = cd.load_programs(self.parameters_dir, self.program_path, self.program_005_path, confirmation.reference_ids)
        lock = cd.build_candidate_lock(state=state, manifest=manifest, manifest_sha256=manifest_sha256, extension=extension, confirmation=confirmation, programs=programs,
                                       parameters_dir=self.parameters_dir, program_path=self.program_path, program_005_path=self.program_005_path,
                                       protocol_code_commit=self._provenance()["protocol_code_commit"], bands_005=self._lock_005().get("bands", {}))
        candidate_path = self.results_path.parent / "candidate-lock.json"
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        state["lock"] = {"candidate_path": str(candidate_path), "content_sha256": lock["content_sha256"], "rank": lock["rank"], "tau": lock["tau"], "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now()}
        cd.write_results_state(self.results_path, state)
        self.log(f"candidate lock written to {candidate_path} (sha256 {lock['content_sha256']}). Install it as {cd.LOCK_RELATIVE_PATH} and commit before confirm.")
        return 0

    def confirm(self) -> int:
        manifest, manifest_sha256, extension, confirmation = self._inputs()
        state = self._state_for("confirm", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        lock_path = self.root / cd.LOCK_RELATIVE_PATH
        if not lock_path.exists():
            raise cd.PhaseError(f"missing committed lock {lock_path}")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        git = self.git_state()
        cd.validate_lock(lock, state=state, manifest_sha256=manifest_sha256, extension=extension, confirmation=confirmation, parameters_dir=self.parameters_dir,
                         program_path=self.program_path, program_005_path=self.program_005_path, git_state=git, tracked=self.tracked(lock_path), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        pool = cd.exposed_pool(manifest, extension)
        programs = cd.load_programs(self.parameters_dir, self.program_path, self.program_005_path, confirmation.reference_ids)
        state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": str(git.get("commit"))}
        pm.record_execution(state, confirmation.all_prompts + tuple(confirmation.reference_prompt(frame) for frame in confirmation.frames) + pm.manifest_prompts(pool.manifest_frames), confirmation.nouns)
        cd.write_results_state(self.results_path, state)
        seed_runtime(cd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            results = cd.run_confirmation(model, pool, confirmation, programs, lock, circuit=self.circuit, axes_dir=self.parameters_dir / "e005-scalar", log=self.log)
        finally:
            del model
            gc.collect()
        state["confirmation"] = {**results, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}
        state["phases"]["confirm"] = {**state["phases"]["confirm"], "status": "complete", "completed_at": pm.utc_now()}
        digest = cd.write_results_state(self.results_path, state)
        self.log(f"confirm complete: {results['outcome']['label']}; results sha256 {digest}")
        return 0

    def report(self) -> int:
        manifest, manifest_sha256, extension, confirmation = self._inputs()
        state = self._state_for("report", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(cd.render_report(state), encoding="utf-8")
        self.log(f"report written to {self.report_path}")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    if args.phase == "validate":
        return runner.validate()
    if args.phase == "freeze-confirmation":
        return runner.freeze_confirmation()
    if args.phase in {"explore", "calibrate", "lock", "confirm", "report"}:
        return getattr(runner, args.phase)()
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
