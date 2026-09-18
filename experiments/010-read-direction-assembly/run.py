#!/usr/bin/env python3
"""Run Experiment 010 through isolated phases.

``validate`` checks the frozen inputs without a model: the manifest, the Experiment 005 extension,
the Experiment 006 and 009 confirmation sets (read in place; digests checked), the three committed
response extracts, and the Experiment 009 lock. ``explore`` runs the single exact-attribution
measurement over the 63 exposed tokens × 24 exposed frames and records every executed prompt and
noun; ``report`` renders the ledger. There is no lock and no confirm phase.
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

import torch

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import cue_suppression as cs
from neural_decompiler import head_transport as ht
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-010"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")
PHASES = ("validate", "explore", "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the frozen inputs and extracts without a model"),
                             ("explore", "the single exact-attribution measurement over the exposed pool"),
                             ("report", "render outputs/experiment-010/report.md")):
        phases.add_parser(phase, help=help_text)
    return parser


def _git_tracked(path: Path) -> bool:
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _run_contract_test() -> dict[str, Any]:
    environment = dict(os.environ, NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE="1")
    completed = subprocess.run([sys.executable, "-m", "pytest", *CONTRACT_TEST], cwd=ROOT, env=environment, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    return {"returncode": completed.returncode, "passed": completed.returncode == 0, "output_tail": output[-4000:], "output_sha256": pm.sha256_text(output)}


def runtime_record(spec: ModelSpec) -> dict[str, Any]:
    runtime = validate_runtime(spec)
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": ra.RUNTIME_SEED, "control_seed": ra.CONTROL_SEED,
            "deterministic_algorithms": spec.deterministic_algorithms, "torch_num_threads": int(torch.get_num_threads())}


@dataclass
class Runner:
    root: Path = ROOT
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    git_state: Callable[[], Mapping[str, Any]] = field(default_factory=lambda: lambda: collect_git_state(ROOT))
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    contract_runner: Callable[[], Mapping[str, Any]] = _run_contract_test
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))

    def _inputs(self):
        manifest, manifest_sha256, extension = pm.load_inputs(self.root)
        confirmation_006 = cd.load_confirmation(self.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
        if confirmation_006.content_sha256 != ss.INHERITED_CONFIRMATION_SHA256:
            raise ra.PhaseError("the Experiment 006 confirmation set digest is not the frozen constant")
        confirmation_009 = ht.load_confirmation(self.root / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
        lock_009 = json.loads((self.root / ra.EXPERIMENT_009_LOCK_PATH).read_text(encoding="utf-8"))
        if lock_009["confirmation_sha256"] != confirmation_009.content_sha256:
            raise ra.PhaseError("the Experiment 009 lock does not carry the Experiment 009 confirmation digest")
        for relative in (cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, ss.INHERITED_EXTRACT_RELATIVE_PATH, cs.INHERITED_007_EXTRACT_RELATIVE_PATH, ra.INHERITED_009_EXTRACT_RELATIVE_PATH, ra.EXPERIMENT_009_LOCK_PATH):
            if not (self.root / relative).exists() or not self.tracked(self.root / relative):
                raise ra.PhaseError(f"{relative} must exist and be tracked and committed")
        inherited_006 = ss.load_inherited_extract(self.root / ss.INHERITED_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256)
        inherited_007 = cs.load_inherited_007(self.root / cs.INHERITED_007_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256)
        inherited_009 = ra.load_inherited_009(self.root / ra.INHERITED_009_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_009_sha256=confirmation_009.content_sha256)
        if inherited_009["lock_sha256"] != lock_009["content_sha256"]:
            raise ra.PhaseError("the inherited 009 extract does not carry the committed 009 lock digest")
        pool = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
        return manifest, manifest_sha256, extension, confirmation_006, confirmation_009, pool, inherited_006, inherited_007, inherited_009

    def validate(self) -> int:
        try:
            *_, pool, inherited_006, inherited_007, inherited_009 = self._inputs()
        except (ra.PhaseError, ValueError) as error:
            self.log(f"validation failed: {error}")
            return 1
        self.log(f"pool: {len(pool.frames)} frames, {len(pool.tokens)} tokens, {len(pool.nouns)} nouns ({len(pool.single_nouns)} single-token); extracts {len(inherited_006['responses'])} + {len(inherited_007['responses'])} + {len(inherited_009['responses'])}")
        return 0

    def _provenance(self) -> dict[str, Any]:
        git = self.git_state()
        return {"protocol_code_commit": str(git.get("commit") or ""), "git_dirty": bool(git.get("dirty", True)), "versions": dict(self.versions())}

    def _record_incident(self, state: dict[str, Any], error: Exception) -> None:
        state["exploration"].setdefault("incidents", []).append({"message": str(error), "at": pm.utc_now(), "phase": "explore", "commit": self._provenance()["protocol_code_commit"]})
        ra.write_results_state(self.results_path, state)
        self.log(f"INCIDENT (explore): {error}")

    def explore(self) -> int:
        manifest, manifest_sha256, extension, confirmation_006, confirmation_009, pool, inherited_006, inherited_007, inherited_009 = self._inputs()
        provenance = self._provenance()
        if self.results_path.exists():
            state = ra.load_results_state(self.results_path)
            if provenance["git_dirty"] or not pm._COMMIT_SHA.fullmatch(provenance["protocol_code_commit"]):
                raise ra.PhaseError("scientific execution requires a clean Git tree at a committed SHA")
            if (state["manifest_sha256"], state["extension_sha256"], state["confirmation_sha256"], state["confirmation_009_sha256"]) != (manifest_sha256, extension.content_sha256, confirmation_006.content_sha256, confirmation_009.content_sha256) or dict(state["versions"]) != provenance["versions"]:
                raise ra.PhaseError("frozen inputs or dependency versions differ from the recorded run")
        else:
            state = ra.new_results_state(manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256, confirmation_009_sha256=confirmation_009.content_sha256, **provenance)
        ra.assert_phase_allowed("explore", state)
        commit = provenance["protocol_code_commit"]
        incidents = state["exploration"].get("incidents", [])
        if incidents and incidents[-1]["commit"] == commit:
            raise ra.PhaseError("an explore incident is recorded at this commit; a committed fix or documented amendment is required before explore runs again")
        contract = dict(self.contract_runner())
        state["exploration"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            ra.write_results_state(self.results_path, state)
            self.log("A0 contract test failed; refusing to explore")
            return 1
        state["protocol_code_commit"] = commit
        state["phases"]["explore"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime_record(PYTHIA_70M), "attempts": int(state["phases"]["explore"].get("attempts", 0)) + 1,
                                      "attempt_commits": list(state["phases"]["explore"].get("attempt_commits", [])) + [commit]}
        prompts = pm.manifest_prompts(pool.frames) + tuple(pool.reference_prompt(frame) for frame in pool.frames)
        pm.record_execution(state, prompts, pool.nouns)
        ra.write_results_state(self.results_path, state)
        seed_runtime(ra.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            try:
                ra.run_exploration(model, pool, state=state, results_path=self.results_path, inherited_006=inherited_006, inherited_007=inherited_007, inherited_009=inherited_009, log=self.log)
            except pm.IncidentError as error:
                self._record_incident(state, error)
                return 2
            except Exception as error:
                self._record_incident(state, error)
                raise
        finally:
            del model
            gc.collect()
        state["phases"]["explore"] = {**state["phases"]["explore"], "status": "complete", "completed_at": pm.utc_now()}
        digest = ra.write_results_state(self.results_path, state)
        self.log(f"explore complete: summary {state['exploration']['summary']['label']}; results sha256 {digest}")
        return 0

    def report(self) -> int:
        self._inputs()
        state = ra.load_results_state(self.results_path)
        ra.assert_phase_allowed("report", state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(ra.render_report(state), encoding="utf-8")
        if state["phases"]["explore"]["status"] == "complete":
            state["phases"]["report"] = {"status": "complete", "completed_at": pm.utc_now()}
            ra.write_results_state(self.results_path, state)
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
