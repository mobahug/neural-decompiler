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
        state["phases"]["discover"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime_record(PYTHIA_70M)}
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    if args.phase == "validate":
        return runner.validate()
    if args.phase == "freeze-extension":
        return runner.freeze_extension()
    if args.phase == "discover":
        return runner.discover(screening_results=args.screening_results)
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
