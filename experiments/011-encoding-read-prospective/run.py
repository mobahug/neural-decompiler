#!/usr/bin/env python3
"""Run Experiment 011 through isolated phases.

``validate`` checks the frozen inputs without a model. ``freeze-confirmation`` builds
``confirmation-v1.json`` from tokenizer rules (refuses to overwrite). ``explore`` (once) measures
the exposed pool, replicates Experiment 010, fixes the axes and tolerances. ``lock`` computes every
prediction from the weights and the locked axes — it performs no forward pass and has no capture or
intervention call path — and writes the candidate lock and the prediction artifact. ``confirm`` (once)
validates the committed artifacts, reproduces every prediction before any fresh prompt, runs the fresh
set, and scores it against the committed numbers. ``report`` renders the report.
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
from neural_decompiler import encoding_read as er
from neural_decompiler import head_transport as ht
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-011"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")
PHASES = ("validate", "freeze-confirmation", "explore", "lock", "confirm", "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the frozen inputs and the confirmation set without a model"), ("freeze-confirmation", "build confirmation-v1.json from tokenizer rules only (refuses to overwrite)"),
                             ("explore", "Tier A: exposed-pool transport fractions, replication, axes, tolerances"), ("lock", "compute every prediction from the weights; write the candidate lock and predictions"),
                             ("confirm", "Tier C: the single fresh-set run scored against the committed predictions"), ("report", "render outputs/experiment-011/report.md")):
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
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": er.RUNTIME_SEED, "control_seed": er.CONTROL_SEED, "deterministic_algorithms": spec.deterministic_algorithms,
            "torch_num_threads": int(torch.get_num_threads())}


def _load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


@dataclass
class Runner:
    root: Path = ROOT
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    tokenizer_loader: Callable[[ModelSpec], Any] = _load_tokenizer
    git_state: Callable[[], Mapping[str, Any]] = field(default_factory=lambda: lambda: collect_git_state(ROOT))
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    changed_paths: Callable[[str], list[str] | None] = _git_changed_paths
    contract_runner: Callable[[], Mapping[str, Any]] = _run_contract_test
    baseline_loader: Callable[[Mapping[str, Any]], ht.TransportRule | None] | None = None  # default: the Experiment 009 lock's rank-1 rule; tests inject None
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))

    @property
    def confirmation_path(self) -> Path:
        return self.root / er.CONFIRMATION_RELATIVE_PATH

    def _baseline(self, lock_009: Mapping[str, Any]) -> ht.TransportRule | None:
        if self.baseline_loader is not None:
            return self.baseline_loader(lock_009)
        return ht.TransportRule(torch.tensor(lock_009["transport_rule"]["v"], dtype=torch.float64), dict(lock_009["transport_rule"]["beta"]), ())

    # -- inputs ---------------------------------------------------------------

    def _base_inputs(self):
        manifest, manifest_sha256, extension = pm.load_inputs(self.root)
        confirmation_006 = cd.load_confirmation(self.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
        if confirmation_006.content_sha256 != ss.INHERITED_CONFIRMATION_SHA256:
            raise er.PhaseError("the Experiment 006 confirmation set digest is not the frozen constant")
        confirmation_009 = ht.load_confirmation(self.root / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
        lock_009 = json.loads((self.root / er.EXPERIMENT_009_LOCK_PATH).read_text(encoding="utf-8"))
        if lock_009["confirmation_sha256"] != confirmation_009.content_sha256 or lock_009.get("transport_rule") is None or "v" not in lock_009["transport_rule"]:
            raise er.PhaseError("the Experiment 009 lock does not carry the 009 confirmation digest and the transport rule's direction")
        for relative in (cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, er.EXPERIMENT_009_LOCK_PATH, er.INHERITED_010_EXTRACT_RELATIVE_PATH):
            if not (self.root / relative).exists() or not self.tracked(self.root / relative):
                raise er.PhaseError(f"{relative} must exist and be tracked and committed")
        inherited_010 = er.load_inherited_010(self.root / er.INHERITED_010_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256, confirmation_009_sha256=confirmation_009.content_sha256)
        pool = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
        digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": confirmation_006.content_sha256, "confirmation_009": confirmation_009.content_sha256}
        return manifest, manifest_sha256, extension, confirmation_006, confirmation_009, pool, inherited_010, lock_009, digests

    def _inputs(self):
        manifest, manifest_sha256, extension, confirmation_006, confirmation_009, pool, inherited_010, lock_009, digests = self._base_inputs()
        if not self.confirmation_path.exists() or not self.tracked(self.confirmation_path):
            raise er.PhaseError("the Experiment 011 confirmation set must be frozen and committed before any scientific phase")
        confirmation = er.load_confirmation(self.confirmation_path, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
        digests = dict(digests) | {"confirmation_011": confirmation.content_sha256}
        return manifest, extension, confirmation_006, confirmation_009, confirmation, pool, inherited_010, lock_009, digests

    def validate(self) -> int:
        try:
            manifest, manifest_sha256, extension, confirmation_006, confirmation_009, pool, inherited_010, lock_009, digests = self._base_inputs()
        except (er.PhaseError, ValueError) as error:
            self.log(f"validation failed: {error}")
            return 1
        self.log(f"exposed pool: {len(pool.frames)} frames, {len(pool.tokens)} tokens; 010 extract {len(inherited_010['fractions'])} pairs; 009 lock carries the rank-1 rule")
        if not self.confirmation_path.exists():
            self.log("confirmation set not frozen yet: run freeze-confirmation")
            return 1
        try:
            confirmation = er.load_confirmation(self.confirmation_path, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
        except ValueError as error:
            self.log(f"validation failed: {error}")
            return 1
        tracked = self.tracked(self.confirmation_path)
        self.log(f"confirmation ok: sha256 {confirmation.content_sha256}; {len(confirmation.tokens)} tokens {[t['word'] for t in confirmation.tokens]}; {len(confirmation.frames)} frames; {len(confirmation.token_prompts)} prompts; tracked={tracked}")
        return 0 if tracked else 1

    def freeze_confirmation(self) -> int:
        manifest, manifest_sha256, extension, confirmation_006, confirmation_009, *_ = self._base_inputs()
        if self.confirmation_path.exists():
            self.log(f"refusing to overwrite frozen confirmation set {self.confirmation_path}")
            return 1
        digest = er.freeze_confirmation(self.confirmation_path, self.tokenizer_loader(PYTHIA_70M), manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
        self.log(f"froze confirmation set {self.confirmation_path} sha256 {digest}; commit it before any scientific phase")
        return 0

    # -- scientific phases ---------------------------------------------------

    def _provenance(self) -> dict[str, Any]:
        git = self.git_state()
        return {"protocol_code_commit": str(git.get("commit") or ""), "git_dirty": bool(git.get("dirty", True)), "versions": dict(self.versions())}

    def _state_for(self, phase: str, digests: Mapping[str, str]) -> dict[str, Any]:
        provenance = self._provenance()
        if self.results_path.exists():
            state = er.load_results_state(self.results_path)
            if provenance["git_dirty"] or not pm._COMMIT_SHA.fullmatch(provenance["protocol_code_commit"]):
                raise er.PhaseError("scientific execution requires a clean Git tree at a committed SHA")
            recorded = (state["manifest_sha256"], state["extension_sha256"], state["confirmation_sha256"], state["confirmation_009_sha256"], state["confirmation_011_sha256"])
            if recorded != (digests["manifest"], digests["extension"], digests["confirmation_006"], digests["confirmation_009"], digests["confirmation_011"]) or dict(state["versions"]) != provenance["versions"]:
                raise er.PhaseError("frozen inputs or dependency versions differ from the recorded run")
        else:
            if phase != "explore":
                raise er.PhaseError(f"{phase} requires an existing results state from explore")
            state = er.new_results_state(digests=digests, **provenance)
        er.assert_phase_allowed(phase, state)
        return state

    def _record_incident(self, state: dict[str, Any], phase: str, error: Exception) -> None:
        entry = {"message": str(error), "at": pm.utc_now(), "phase": phase, "commit": self._provenance()["protocol_code_commit"]}
        if phase == "explore":
            state["exploration"].setdefault("incidents", []).append(entry)
        else:
            state["confirmation"] = {"incident": entry}
        er.write_results_state(self.results_path, state)
        self.log(f"INCIDENT ({phase}): {error}")

    def explore(self) -> int:
        manifest, extension, confirmation_006, confirmation_009, confirmation, pool, inherited_010, lock_009, digests = self._inputs()
        state = self._state_for("explore", digests)
        commit = self._provenance()["protocol_code_commit"]
        incidents = state["exploration"].get("incidents", [])
        if incidents and incidents[-1]["commit"] == commit:
            raise er.PhaseError("an explore incident is recorded at this commit; a committed fix or documented amendment is required before explore runs again")
        contract = dict(self.contract_runner())
        state["exploration"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            er.write_results_state(self.results_path, state)
            self.log("A0 contract test failed; refusing to explore")
            return 1
        state["protocol_code_commit"] = commit
        state["phases"]["explore"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime_record(PYTHIA_70M), "attempts": int(state["phases"]["explore"].get("attempts", 0)) + 1,
                                      "attempt_commits": list(state["phases"]["explore"].get("attempt_commits", [])) + [commit]}
        pm.record_execution(state, pm.manifest_prompts(pool.frames) + tuple(pool.reference_prompt(frame) for frame in pool.frames), pool.nouns)
        er.assert_confirmation_untouched(state, confirmation)
        er.write_results_state(self.results_path, state)
        seed_runtime(er.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            try:
                er.run_exploration(model, pool, state=state, results_path=self.results_path, inherited_010=inherited_010, log=self.log)
            except pm.IncidentError as error:
                self._record_incident(state, "explore", error)
                return 2
            except Exception as error:
                self._record_incident(state, "explore", error)
                raise
        finally:
            del model
            gc.collect()
        er.assert_confirmation_untouched(state, confirmation)
        state["phases"]["explore"] = {**state["phases"]["explore"], "status": "complete", "completed_at": pm.utc_now()}
        digest = er.write_results_state(self.results_path, state)
        summary = state["exploration"]["summary"]
        self.log(f"explore complete: defined templates {summary['defined_templates']}, tau_g {summary['tau_g']:.3f}, tau_M {summary['tau_M']:.3f}; results sha256 {digest}")
        return 0

    def _weights_only(self) -> tuple[pm.Weights, ht.HeadWeights]:
        """The lock phase's only access to the model: its parameters. No prompt is run and no capture API is reachable from here."""
        model = self.model_loader(PYTHIA_70M)
        try:
            return pm.Weights.from_model(model), ht.HeadWeights.from_model(model)
        finally:
            del model
            gc.collect()

    def _predictions(self, weights: pm.Weights, exploration: Mapping[str, Any], confirmation: er.Confirmation011, pool: cs.Pool008, lock_009: Mapping[str, Any]) -> dict[str, Any]:
        plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
        read = er.read_from_lock_inputs(exploration, pool.reference_ids, plural_ids)  # at confirm, ``exploration`` is the lock itself (locked axes and read weight)
        return er.lock_predictions(weights, read, confirmation, exploration["denominators"]["defined"], self._baseline(lock_009))

    def lock(self) -> int:
        manifest, extension, confirmation_006, confirmation_009, confirmation, pool, inherited_010, lock_009, digests = self._inputs()
        state = self._state_for("lock", digests)
        changed = self.changed_paths(state["protocol_code_commit"])
        scientific = [path for path in (changed or []) if path.startswith(er.SCIENTIFIC_PATH_PREFIXES) and path not in er.NON_SCIENTIFIC_PATHS and not path.startswith(er.NON_SCIENTIFIC_PREFIXES)]
        if changed is None:
            raise er.PhaseError("the explore commit is not an ancestor of the current commit; lock must be written at the explore protocol")
        if scientific:
            raise er.PhaseError(f"scientific paths changed since explore: {scientific}; lock must be written at the explore protocol")
        weights, _ = self._weights_only()
        predictions = self._predictions(weights, state["exploration"], confirmation, pool, lock_009)
        lock = er.build_candidate_lock(state=state, digests=digests, confirmation=confirmation, predictions=predictions, protocol_code_commit=self._provenance()["protocol_code_commit"])
        candidate_path = self.results_path.parent / "candidate-lock.json"
        predictions_path = self.results_path.parent / "candidate-predictions.md"
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        predictions_text = er.render_predictions(lock)
        predictions_path.write_text(predictions_text, encoding="utf-8")
        state["lock"] = {"candidate_path": str(candidate_path), "predictions_path": str(predictions_path), "content_sha256": lock["content_sha256"], "predictions_sha256": pm.sha256_text(predictions_text), "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now()}
        er.write_results_state(self.results_path, state)
        self.log(f"candidate lock written to {candidate_path} (sha256 {lock['content_sha256']}) and {predictions_path} (sha256 {state['lock']['predictions_sha256']}). Install both as {er.LOCK_RELATIVE_PATH} and {er.PREDICTIONS_RELATIVE_PATH} and commit before confirm.")
        return 0

    def confirm(self) -> int:
        manifest, extension, confirmation_006, confirmation_009, confirmation, pool, inherited_010, lock_009, digests = self._inputs()
        state = self._state_for("confirm", digests)
        lock_path, predictions_path = self.root / er.LOCK_RELATIVE_PATH, self.root / er.PREDICTIONS_RELATIVE_PATH
        if not lock_path.exists() or not predictions_path.exists():
            raise er.PhaseError("missing committed lock or predictions artifact")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        git = self.git_state()
        er.validate_lock(lock, state=state, digests=digests, confirmation=confirmation, predictions_text=predictions_path.read_text(encoding="utf-8"), git_state=git,
                         tracked=self.tracked(lock_path) and self.tracked(predictions_path), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        seed_runtime(er.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            weights = pm.Weights.from_model(model)
            reproduced = er.assert_lock_predictions_reproduced(lock, self._predictions(weights, lock, confirmation, pool, lock_009))  # from the locked axes; PhaseError before anything fresh runs
            state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": str(git.get("commit")), "lock_predictions_reproduced_max_difference": reproduced}
            defined_frames = [frame for frame in confirmation.frames if frame.template_id in lock["defined_templates"]]
            prompts = tuple(prompt for prompt in confirmation.all_prompts if prompt.frame.template_id in lock["defined_templates"]) + tuple(confirmation.reference_prompt(frame) for frame in defined_frames)
            pm.record_execution(state, prompts, pool.nouns)
            er.write_results_state(self.results_path, state)
            try:
                results = er.run_confirmation(model, pool, confirmation, lock, log=self.log)
            except pm.IncidentError as error:
                self._record_incident(state, "confirm", error)
                return 2
            except Exception as error:
                self._record_incident(state, "confirm", error)
                raise
        finally:
            del model
            gc.collect()
        state["confirmation"] = {**results, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}
        state["phases"]["confirm"] = {**state["phases"]["confirm"], "status": "complete", "completed_at": pm.utc_now()}
        digest = er.write_results_state(self.results_path, state)
        self.log(f"confirm complete: {results['outcome']['label']}; results sha256 {digest}")
        return 0

    def report(self) -> int:
        self._inputs()
        state = er.load_results_state(self.results_path)
        er.assert_phase_allowed("report", state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(er.render_report(state), encoding="utf-8")
        self.log(f"report written to {self.report_path}")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    if args.phase in PHASES:
        return getattr(runner, args.phase.replace("-", "_"))()
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
