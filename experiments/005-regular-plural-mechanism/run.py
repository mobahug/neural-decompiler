#!/usr/bin/env python3
"""Run Experiment 005 through isolated phases.

``validate`` checks the manifest and the frozen extension without a model.
``freeze-extension`` builds ``extension-v1.json`` from tokenizer rules only and
refuses to overwrite it. The scientific phases ``discover``, ``calibrate``,
``revise``, ``lock``, ``confirm``, and ``report`` are enforced in order through
the results state; reserve nouns and extension prompts are executed only by
``confirm`` after the committed preregistration lock validates.
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

from neural_decompiler import plural_mechanism as pm
from neural_decompiler.candidate_screening import load_manifest
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs/experiment-005"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
PARAMETERS_DIR = OUTPUT_DIR / "parameters"
PROGRAM_PATH = EXPERIMENT_DIR / "mechanism_program.py"
SCREENING_RESULTS_CANDIDATES = (
    ROOT / "outputs/behavior-candidate-screening/results.json",
    ROOT / ".worktrees/behavior-candidate-screening/outputs/behavior-candidate-screening/results.json",
)
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("validate", help="check the manifest and frozen extension without loading a model")
    phases.add_parser("freeze-extension", help="build extension-v1.json from tokenizer rules only (refuses to overwrite)")
    discover = phases.add_parser("discover", help="Tier A: run A0-A11 once on development data")
    discover.add_argument("--screening-results", default=None, metavar="PATH", help="screening results.json for the A1 per-case comparison")
    phases.add_parser("continue", help="revision 5: adopt the recorded circuit as protocol v2 after a NO_COMPACT_MECHANISM discover")
    phases.add_parser("calibrate", help="Tier B: evaluate the current mechanism version on holdout nouns (at most twice)")
    phases.add_parser("revise", help="mechanical revision after a failed first calibration pass")
    phases.add_parser("lock", help="write outputs/experiment-005/candidate-lock.json for review")
    phases.add_parser("confirm", help="Tier C: the single reserve + extension run; requires the committed lock")
    phases.add_parser("report", help="render outputs/experiment-005/report.md")
    return parser


def _git_tracked(path: Path) -> bool:
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def runtime_record(spec: ModelSpec) -> dict[str, Any]:
    runtime = validate_runtime(spec)
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name,
            "seed": pm.RUNTIME_SEED, "deterministic_algorithms": spec.deterministic_algorithms}


def _load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


def _git_changed_paths(commit: str) -> list[str] | None:
    """Paths changed between ``commit`` and HEAD, or None when ``commit`` is not an ancestor."""
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True, capture_output=True)
        diff = subprocess.run(["git", "diff", "--name-only", commit, "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in diff.stdout.splitlines() if line.strip()]


def _run_contract_test() -> dict[str, Any]:
    """A0: the pinned Bridge contract test on this runtime, recorded verbatim."""
    environment = dict(os.environ, NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE="1")
    completed = subprocess.run([sys.executable, "-m", "pytest", *CONTRACT_TEST], cwd=ROOT, env=environment, capture_output=True, text=True)
    output = (completed.stdout + completed.stderr)[-4000:]
    return {"returncode": completed.returncode, "passed": completed.returncode == 0, "output_tail": output,
            "output_sha256": pm.sha256_text(completed.stdout + completed.stderr)}


@dataclass
class Runner:
    """Phase orchestration with injectable model, tokenizer, Git, and version providers."""

    root: Path = ROOT
    experiment_dir: Path = EXPERIMENT_DIR
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    tokenizer_loader: Callable[[ModelSpec], Any] = _load_tokenizer
    git_state: Callable[[], Mapping[str, Any]] = field(default_factory=lambda: lambda: collect_git_state(ROOT))
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    contract_runner: Callable[[], Mapping[str, Any]] = _run_contract_test
    changed_paths: Callable[[str], list[str] | None] = _git_changed_paths
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))
    parameters_dir: Path = PARAMETERS_DIR
    program_path: Path = PROGRAM_PATH

    @property
    def manifest_path(self) -> Path:
        return self.root / pm.MANIFEST_RELATIVE_PATH

    @property
    def extension_path(self) -> Path:
        return self.root / pm.EXTENSION_RELATIVE_PATH

    def _manifest(self):
        manifest = load_manifest(self.manifest_path)
        return manifest, pm.manifest_digest_from_path(self.manifest_path)

    def validate(self) -> int:
        manifest, manifest_sha256 = self._manifest()
        frames = pm.derive_frames(manifest)
        nouns = {split: pm.nouns_for(manifest, split) for split in pm.Split}
        self.log(f"manifest ok: sha256 {manifest_sha256}; {len(frames)} frames; "
                 + ", ".join(f"{split.value}={len(items)} nouns" for split, items in nouns.items()))
        if not self.extension_path.exists():
            self.log("extension not frozen yet: run freeze-extension")
            return 1
        extension = pm.load_extension(self.extension_path, manifest, manifest_sha256)
        tracked = self.tracked(self.extension_path)
        self.log(f"extension ok: sha256 {extension.content_sha256}; {len(extension.new_frames)} new frames; "
                 f"{len(extension.cue_words)} cue words; {len(extension.cue_word_prompts)} cue-word prompts; tracked={tracked}")
        return 0 if tracked else 1

    def freeze_extension(self) -> int:
        manifest, manifest_sha256 = self._manifest()
        if self.extension_path.exists():
            self.log(f"refusing to overwrite frozen extension {self.extension_path}")
            return 1
        tokenizer = self.tokenizer_loader(PYTHIA_70M)
        digest = pm.freeze_extension(self.extension_path, tokenizer, manifest, manifest_sha256)
        self.log(f"froze extension {self.extension_path} sha256 {digest}; commit it before any scientific phase")
        return 0


    # -- scientific phases ---------------------------------------------------

    def _provenance(self) -> dict[str, Any]:
        git = self.git_state()
        return {"protocol_code_commit": str(git.get("commit") or ""), "git_dirty": bool(git.get("dirty", True)), "versions": dict(self.versions())}

    def _inputs(self):
        manifest, manifest_sha256 = self._manifest()
        if not self.extension_path.exists() or not self.tracked(self.extension_path):
            raise pm.PhaseError("the extension set must be frozen and committed before any scientific phase")
        extension = pm.load_extension(self.extension_path, manifest, manifest_sha256)
        return manifest, manifest_sha256, extension

    def _state_for(self, phase: str, manifest_sha256: str, extension_sha256: str) -> dict[str, Any]:
        provenance = self._provenance()
        if self.results_path.exists():
            state = pm.load_results_state(self.results_path)
            pm.assert_provenance_identical(state, manifest_sha256=manifest_sha256, extension_sha256=extension_sha256, **provenance)
        else:
            if phase != "discover":
                raise pm.PhaseError(f"{phase} requires an existing results state from discover")
            state = pm.new_results_state(manifest_sha256=manifest_sha256, extension_sha256=extension_sha256, **provenance)
        pm.assert_phase_allowed(phase, state)
        return state

    def _screening_results(self, override: str | None) -> Path | None:
        if override:
            path = Path(override)
            if not path.exists():
                raise pm.PhaseError(f"screening results not found: {path}")
            return path
        for candidate in SCREENING_RESULTS_CANDIDATES:
            if candidate.exists():
                return candidate
        return None

    def discover(self, *, screening_results: str | None = None) -> int:
        manifest, manifest_sha256, extension = self._inputs()
        state = self._state_for("discover", manifest_sha256, extension.content_sha256)
        reserve = pm.nouns_for(manifest, pm.Split.FUTURE_RESERVE)
        contract = dict(self.contract_runner())
        state["discovery"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            self.log("A0 contract test failed; refusing to discover")
            self.log(str(contract.get("output_tail", ""))[-1500:])
            return 1
        screening_path = self._screening_results(screening_results)
        self.log(f"A1 comparison source: {screening_path if screening_path else 'report constants only (no results.json found)'}")
        state["discovery"]["a1_screening_results_path"] = str(screening_path) if screening_path else None
        attempts = int(state["phases"]["discover"].get("attempts", 0)) + 1
        if attempts > 1:
            self.log(f"discover attempt {attempts}: the previous attempt did not conclude; recomputing the deterministic measurements")
            state["discovery"] = {key: value for key, value in state["discovery"].items() if key in {"a0_contract_test", "a1_screening_results_path"}}
        state["phases"]["discover"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime_record(PYTHIA_70M), "attempts": attempts}
        pm.write_results_state(self.results_path, state)
        seed_runtime(pm.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            development = pm.nouns_for(manifest, pm.Split.DEVELOPMENT)
            holdout = pm.nouns_for(manifest, pm.Split.HOLDOUT)
            ctx = pm.new_discovery_context(model, manifest, development)
            pm.record_execution(state, pm.manifest_prompts(ctx.frames), tuple(development) + tuple(holdout))
            pm.assert_not_executed(state, extension=extension, reserve=reserve)
            version = pm.run_discovery(ctx, state=state, results_path=self.results_path, program_path=self.program_path,
                                       parameters_dir=self.parameters_dir, screening_results_path=screening_path, log=self.log)
        finally:
            del model
            gc.collect()
        pm.assert_not_executed(state, extension=extension, reserve=reserve)
        state["phases"]["discover"] = {**state["phases"]["discover"], "status": "complete", "completed_at": pm.utc_now()}
        digest = pm.write_results_state(self.results_path, state)
        self.log(f"discover complete: version {version['version']} status {version['status']} hypothesis {version['hypothesis']}"
                 f"{' (ambiguous)' if version.get('hypothesis_flagged') else ''}; results sha256 {digest}")
        if version["status"] == "candidate":
            self.log(version["statement"])
        else:
            self.log(f"outcome {version.get('outcome')}: no lock will be written")
        return 0


    # -- Tier B, lock, Tier C, report ------------------------------------------

    def _current_version(self, state: Mapping[str, Any]) -> dict[str, Any]:
        candidates = [entry for entry in state["mechanism_versions"] if entry.get("status") == "candidate"]
        if not candidates:
            raise pm.PhaseError("no candidate mechanism version exists")
        return dict(candidates[-1])

    def _eval_context(self, model: Any, version: Mapping[str, Any], frames, nouns) -> pm.EvalContext:
        mechanism = pm.MechanismSet(version["mechanism"]["branch"], tuple(version["mechanism"]["e_keys"]), tuple(version["mechanism"]["t_keys"]), tuple(version["mechanism"]["r_keys"]))
        program = pm.load_program(self.parameters_dir, self.program_path)
        axes = pm.site_axes_from_parameters(self.parameters_dir)
        return pm.EvalContext(model, pm.Weights.from_model(model), mechanism, program, tuple(frames), tuple(nouns), pm.PromptCache(model, tuple(nouns)), axes, version["hypothesis"])

    def continue_v2(self) -> int:
        manifest, manifest_sha256, extension = self._inputs()
        state = self._state_for("continue", manifest_sha256, extension.content_sha256)
        reserve = pm.nouns_for(manifest, pm.Split.FUTURE_RESERVE)
        development = pm.nouns_for(manifest, pm.Split.DEVELOPMENT)
        state["phases"]["continue"] = {"status": "running", "started_at": pm.utc_now(), "protocol_version": 2, "commit": self._provenance()["protocol_code_commit"]}
        pm.write_results_state(self.results_path, state)
        seed_runtime(pm.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            ctx = pm.new_discovery_context(model, manifest, development)
            pm.assert_not_executed(state, extension=extension, reserve=reserve)
            version = pm.adopt_continuation(ctx, state=state, results_path=self.results_path, program_path=self.program_path, parameters_dir=self.parameters_dir, log=self.log)
        finally:
            del model
            gc.collect()
        pm.assert_not_executed(state, extension=extension, reserve=reserve)
        state["phases"]["continue"] = {**state["phases"]["continue"], "status": "complete", "completed_at": pm.utc_now()}
        digest = pm.write_results_state(self.results_path, state)
        self.log(f"continuation adopted: version {version['version']} (k={version['k']}), PROGRAM_CAPPED; results sha256 {digest}")
        self.log(version["statement"])
        return 0

    def calibrate(self) -> int:
        manifest, manifest_sha256, extension = self._inputs()
        state = self._state_for("calibrate", manifest_sha256, extension.content_sha256)
        version = self._current_version(state)
        reserve = pm.nouns_for(manifest, pm.Split.FUTURE_RESERVE)
        holdout = tuple(noun for noun in pm.nouns_for(manifest, pm.Split.HOLDOUT) if noun.single_token)
        frames = pm.derive_frames(manifest)
        state["phases"]["calibrate"] = {"status": "running", "started_at": pm.utc_now()}
        pm.write_results_state(self.results_path, state)
        seed_runtime(pm.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            ev = self._eval_context(model, version, frames, holdout)
            universe = pm.universe_keys(model)
            pm.record_execution(state, pm.manifest_prompts(frames), holdout)
            pm.assert_not_executed(state, extension=extension, reserve=reserve)
            self.log(f"calibrating {version['version']} on {len(holdout)} holdout nouns")
            results = pm.evaluate_circuit_families(ev, universe)
        finally:
            del model
            gc.collect()
        n_cases = results["behavior"]["cases"]
        floors = pm.circuit_floors(results, hypothesis=version["hypothesis"], n_cases=n_cases)
        bands = pm.calibration_bands(results)
        residual = results["program_residual"]
        entry = {"version": version["version"], "n_cases": n_cases, "results": {key: value for key, value in results.items() if key != "program_residual"},
                 "program_residual": {"rmse": residual["rmse"], "mae": residual["mae"], "conditions": residual["conditions"]},
                 "floors": floors, "bands": bands, "floors_passed": floors["circuit_passed"], "completed_at": pm.utc_now()}
        state["calibration"]["passes"].append(entry)
        state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "complete", "completed_at": pm.utc_now()}
        digest = pm.write_results_state(self.results_path, state)
        self.log(f"calibration pass {len(state['calibration']['passes'])}: floors {'passed' if floors['circuit_passed'] else 'FAILED ' + str(floors['primary_failures'] or ['B1'])}; "
                 f"RMSE_B {residual['rmse']:.4f}; tau {pm.tolerance_tau(residual['rmse']):.3f}; results sha256 {digest}")
        return 0

    def revise(self) -> int:
        manifest, manifest_sha256, extension = self._inputs()
        state = self._state_for("revise", manifest_sha256, extension.content_sha256)
        version = self._current_version(state)
        instruction = pm.revise_version(version, state["calibration"]["passes"][-1], state["discovery"]["a2"])
        state["phases"]["revise"] = {"status": "running", "started_at": pm.utc_now(), "instruction": instruction}
        pm.write_results_state(self.results_path, state)
        if instruction["action"] == "reject":
            state["mechanism_versions"].append({"version": instruction["version"], "status": "rejected", "outcome": "NO_COMPACT_MECHANISM", "reason": instruction["reason"]})
            state["phases"]["revise"] = {**state["phases"]["revise"], "status": "complete", "completed_at": pm.utc_now()}
            pm.write_results_state(self.results_path, state)
            self.log(f"revision rejected: {instruction['reason']}")
            return 1
        reserve = pm.nouns_for(manifest, pm.Split.FUTURE_RESERVE)
        development = pm.nouns_for(manifest, pm.Split.DEVELOPMENT)
        seed_runtime(pm.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            ctx = pm.new_discovery_context(model, manifest, development)
            pm.assert_not_executed(state, extension=extension, reserve=reserve)
            previous = version["mechanism"]
            hypothesis = version["hypothesis"]
            t_keys, r_keys = list(previous["t_keys"]), list(previous["r_keys"])
            if instruction["action"] == "extend":
                for key in instruction["add"]:
                    (t_keys if pm.is_head_key(key) else r_keys).append(key)
            else:
                hypothesis = instruction["row"]
            mechanism = pm.MechanismSet(previous["branch"], tuple(previous["e_keys"]), tuple(t_keys), tuple(r_keys))
            evaluation = pm.evaluate_candidate_set(ctx, mechanism, program_path=self.program_path, parameters_dir=self.parameters_dir)
            new_version = {**version, "version": instruction["version"], "mechanism": mechanism.to_dict(), "hypothesis": hypothesis,
                           "program_capped": evaluation["program_capped"], "program_floors": evaluation["program_floors"], "recovery": evaluation["recovery"],
                           "isolation": evaluation["isolation"]["faithfulness"], "sign_retention": evaluation["isolation"]["sign_retention"], "revision": instruction}
            if evaluation["passed"]:
                parameters = evaluation["_parameters"]
                index = pm.export_program_parameters(self.parameters_dir, ctx.weights, parameters, vocab_size=int(ctx.weights.W_E.shape[0]))
                program = pm.load_program(self.parameters_dir, self.program_path)
                new_version["parameters_index_sha256"] = pm.sha256_text(pm.canonical_json(index))
                new_version["lexicon_sha256"] = program.lexicon_digest()
                new_version["statement"] = pm.render_mechanism_statement(mechanism, parameters, hypothesis=hypothesis, flagged=version["hypothesis_flagged"],
                                                                        capped=evaluation["program_capped"], encoding=version["encoding_branch"])
                new_version["status"] = "candidate"
            else:
                new_version["status"] = "rejected"
                new_version["outcome"] = "NO_COMPACT_MECHANISM"
        finally:
            del model
            gc.collect()
        state["mechanism_versions"].append(new_version)
        state["phases"]["revise"] = {**state["phases"]["revise"], "status": "complete", "completed_at": pm.utc_now()}
        pm.write_results_state(self.results_path, state)
        self.log(f"revision {new_version['version']}: {new_version['status']} ({instruction['action']})")
        return 0 if new_version["status"] == "candidate" else 1

    def lock(self) -> int:
        manifest, manifest_sha256, extension = self._inputs()
        state = self._state_for("lock", manifest_sha256, extension.content_sha256)
        version = self._current_version(state)
        calibration = state["calibration"]["passes"][-1]
        if calibration["version"] != version["version"]:
            raise pm.PhaseError("the last calibration pass does not belong to the current mechanism version")
        program = pm.load_program(self.parameters_dir, self.program_path)
        lock = pm.build_candidate_lock(state=state, version=version, calibration=calibration, manifest=manifest, manifest_sha256=manifest_sha256, extension=extension,
                                       program=program, parameters_dir=self.parameters_dir, program_path=self.program_path, protocol_code_commit=self._provenance()["protocol_code_commit"])
        candidate_path = self.results_path.parent / "candidate-lock.json"
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        state["lock"] = {"candidate_path": str(candidate_path), "content_sha256": lock["content_sha256"], "version": version["version"], "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now()}
        pm.write_results_state(self.results_path, state)
        self.log(f"candidate lock written to {candidate_path} (sha256 {lock['content_sha256']}).")
        self.log(f"Review it, then install it as {pm.LOCK_RELATIVE_PATH} and commit before running confirm.")
        return 0

    def confirm(self) -> int:
        manifest, manifest_sha256, extension = self._inputs()
        state = self._state_for("confirm", manifest_sha256, extension.content_sha256)
        lock_path = self.root / pm.LOCK_RELATIVE_PATH
        if not lock_path.exists():
            raise pm.PhaseError(f"missing committed lock {lock_path}")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        git = self.git_state()
        pm.validate_lock(lock, state=state, manifest=manifest, manifest_sha256=manifest_sha256, extension=extension, parameters_dir=self.parameters_dir,
                         program_path=self.program_path, git_state=git, tracked=self.tracked(lock_path), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        version = self._current_version(state)
        reserve = pm.nouns_for(manifest, pm.Split.FUTURE_RESERVE)
        frames = pm.derive_frames(manifest)
        state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": str(git.get("commit"))}
        pm.write_results_state(self.results_path, state)
        seed_runtime(pm.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            universe = pm.universe_keys(model)
            # The intent to execute is recorded before any reserve or extension forward runs.
            pm.record_execution(state, pm.manifest_prompts(frames) + extension.new_frame_prompts() + extension.cue_word_prompts, reserve)
            pm.write_results_state(self.results_path, state)
            self.log("Tier C: reserve nouns on the twelve manifest prompts")
            ev = self._eval_context(model, version, frames, reserve)
            circuit_results = pm.evaluate_circuit_families(ev, universe)
            self.log("Tier C: new frames")
            ev_new = pm.EvalContext(model, ev.weights, ev.mechanism, ev.program, extension.new_frames, reserve, pm.PromptCache(model, reserve), ev.site_axes, version["hypothesis"])
            new_frame_results = pm.evaluate_new_frame_families(ev_new, universe)
            self.log("Tier C: cue words")
            locked_full_shift = {template: band["point"] for template, band in lock["bands"]["B2"].items()}
            cue_word_results = pm.evaluate_cue_word_families(ev, extension, locked_full_shift=locked_full_shift)
            cue_word_results["locked_full_shift"] = locked_full_shift
        finally:
            del model
            gc.collect()
        n_cases = circuit_results["behavior"]["cases"]
        circuit = pm.circuit_floors(circuit_results, hypothesis=version["hypothesis"], n_cases=n_cases)
        hits = pm.band_hits(circuit_results, lock["bands"])
        decompilation = pm.decompilation_floors(new_frame_results, cue_word_results, hypothesis=version["hypothesis"], capped=bool(lock["mechanism"]["program_capped"]))
        x_hits = pm.x_band_hits(cue_word_results, {"X3": lock["x_predictions"]["X3_band"], "X4": lock["x_predictions"]["X4_band"]})
        contested = pm.contested(circuit_results, hypothesis=version["hypothesis"], bands=lock["bands"])
        verdict = pm.outcome(circuit=circuit, hits=hits, decompilation=decompilation, x_hits=x_hits, contested_findings=contested)
        state["confirmation"] = {
            "lock_sha256": lock["content_sha256"], "n_cases": n_cases,
            "reserve": {key: value for key, value in circuit_results.items() if key != "program_residual"},
            "reserve_program_residual": {key: value for key, value in circuit_results["program_residual"].items() if key != "entries"},
            "new_frames": {key: value for key, value in new_frame_results.items() if key != "program_residual"},
            "new_frames_program_residual": {key: value for key, value in new_frame_results["program_residual"].items() if key != "entries"},
            "cue_words": cue_word_results, "circuit_floors": circuit, "band_hits": hits, "decompilation_floors": decompilation, "x_band_hits": x_hits,
            "contested": contested, "outcome": verdict, "completed_at": pm.utc_now(),
        }
        state["phases"]["confirm"] = {**state["phases"]["confirm"], "status": "complete", "completed_at": pm.utc_now()}
        digest = pm.write_results_state(self.results_path, state)
        self.log(f"confirm complete: {verdict['label']} ({verdict['axes']}); results sha256 {digest}")
        return 0

    def report(self) -> int:
        manifest, manifest_sha256, extension = self._inputs()
        state = self._state_for("report", manifest_sha256, extension.content_sha256)
        text = pm.render_report(state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(text, encoding="utf-8")
        self.log(f"report written to {self.report_path}")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    if args.phase == "validate":
        return runner.validate()
    if args.phase == "freeze-extension":
        return runner.freeze_extension()
    if args.phase == "discover":
        return runner.discover(screening_results=args.screening_results)
    if args.phase == "continue":
        return runner.continue_v2()
    if args.phase in {"calibrate", "revise", "lock", "confirm", "report"}:
        return getattr(runner, args.phase)()
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
