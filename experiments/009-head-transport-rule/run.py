#!/usr/bin/env python3
"""Run Experiment 009 through isolated phases.

``validate`` checks the frozen inputs without a model. ``freeze-confirmation`` builds
``confirmation-v1.json`` from tokenizer rules under the frozen grammatical-compatibility policy
and refuses to overwrite it. ``explore`` (once) measures the exposed pool, fits and gates the
rules, and locks the mechanism level. ``lock`` writes the candidate lock and the human-readable
prediction artifact without running any fresh prompt. ``confirm`` (once) validates the committed
lock and predictions, reproduces every locked prediction before any fresh prompt, then runs the
fresh set and scores it only against the locked numbers. ``report`` renders the report.
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
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-009"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
PARAMETERS_DIR = OUTPUT_DIR / "parameters"
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")
PHASES = ("validate", "freeze-confirmation", "explore", "lock", "confirm", "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the frozen inputs and the confirmation set without a model"),
                             ("freeze-confirmation", "build confirmation-v1.json from tokenizer rules only (refuses to overwrite)"),
                             ("explore", "Tier A: exposed-pool measurements, OV decomposition, rules, gates"),
                             ("lock", "write the candidate lock and predictions artifact without running any fresh prompt"),
                             ("confirm", "Tier C: the single fresh-set run; requires the committed lock and predictions"),
                             ("report", "render outputs/experiment-009/report.md")):
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
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": ht.RUNTIME_SEED, "control_seed": ht.CONTROL_SEED,
            "deterministic_algorithms": spec.deterministic_algorithms, "torch_num_threads": int(torch.get_num_threads())}


def _load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


@dataclass
class Runner:
    root: Path = ROOT
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    parameters_dir: Path = PARAMETERS_DIR
    model_loader: Callable[[ModelSpec], Any] = load_model
    tokenizer_loader: Callable[[ModelSpec], Any] = _load_tokenizer
    program_007_loader: Callable[[], Any | None] | None = None  # default: cs.load_program_007(root); tests inject
    git_state: Callable[[], Mapping[str, Any]] = field(default_factory=lambda: lambda: collect_git_state(ROOT))
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    changed_paths: Callable[[str], list[str] | None] = _git_changed_paths
    contract_runner: Callable[[], Mapping[str, Any]] = _run_contract_test
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))

    @property
    def confirmation_path(self) -> Path:
        return self.root / ht.CONFIRMATION_RELATIVE_PATH

    @property
    def program_source(self) -> Path:
        return self.root / ht.PROGRAM_SOURCE

    # -- inputs ---------------------------------------------------------------

    def _base_inputs(self):
        manifest, manifest_sha256, extension = pm.load_inputs(self.root)
        confirmation_006 = cd.load_confirmation(self.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
        if confirmation_006.content_sha256 != ss.INHERITED_CONFIRMATION_SHA256:
            raise ht.PhaseError("the Experiment 006 confirmation set digest is not the frozen constant")
        for relative in (cd.CONFIRMATION_RELATIVE_PATH, ss.INHERITED_EXTRACT_RELATIVE_PATH, cs.INHERITED_007_EXTRACT_RELATIVE_PATH):
            if not self.tracked(self.root / relative):
                raise ht.PhaseError(f"{relative} must be tracked and committed")
        inherited_006 = ss.load_inherited_extract(self.root / ss.INHERITED_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256)
        inherited_007 = cs.load_inherited_007(self.root / cs.INHERITED_007_EXTRACT_RELATIVE_PATH, manifest_sha256=manifest_sha256, extension_sha256=extension.content_sha256, confirmation_sha256=confirmation_006.content_sha256)
        pool = cs.build_pool(manifest, extension, confirmation_006)
        return manifest, manifest_sha256, extension, confirmation_006, pool, inherited_006, inherited_007

    def _inputs(self):
        manifest, manifest_sha256, extension, confirmation_006, pool, inherited_006, inherited_007 = self._base_inputs()
        if not self.confirmation_path.exists() or not self.tracked(self.confirmation_path):
            raise ht.PhaseError("the Experiment 009 confirmation set must be frozen and committed before any scientific phase")
        confirmation = ht.load_confirmation(self.confirmation_path, manifest, manifest_sha256, extension, confirmation_006)
        return manifest, manifest_sha256, extension, confirmation, pool, inherited_006, inherited_007

    def _program_007(self) -> Any | None:
        if self.program_007_loader is not None:
            return self.program_007_loader()
        program, _ = cs.load_program_007(self.root)
        return program

    def validate(self) -> int:
        try:
            manifest, manifest_sha256, extension, confirmation_006, pool, inherited_006, inherited_007 = self._base_inputs()
        except (ht.PhaseError, ValueError) as error:
            self.log(f"validation failed: {error}")
            return 1
        self.log(f"exposed pool: {len(pool.frames)} frames, {len(pool.tokens)} tokens, {len(pool.nouns)} nouns; extracts {len(inherited_006['responses'])} + {len(inherited_007['responses'])}")
        if not self.confirmation_path.exists():
            self.log("confirmation set not frozen yet: run freeze-confirmation")
            return 1
        try:
            confirmation = ht.load_confirmation(self.confirmation_path, manifest, manifest_sha256, extension, confirmation_006)
        except ValueError as error:
            self.log(f"validation failed: {error}")
            return 1
        tracked = self.tracked(self.confirmation_path)
        self.log(f"confirmation ok: sha256 {confirmation.content_sha256}; {len(confirmation.tokens)} tokens {[t['word'] for t in confirmation.tokens]}; {len(confirmation.frames)} frames; {len(confirmation.token_prompts)} token prompts; tracked={tracked}")
        return 0 if tracked else 1

    def freeze_confirmation(self) -> int:
        manifest, manifest_sha256, extension, confirmation_006, *_ = self._base_inputs()
        if self.confirmation_path.exists():
            self.log(f"refusing to overwrite frozen confirmation set {self.confirmation_path}")
            return 1
        digest = ht.freeze_confirmation(self.confirmation_path, self.tokenizer_loader(PYTHIA_70M), manifest, manifest_sha256, extension, confirmation_006)
        self.log(f"froze confirmation set {self.confirmation_path} sha256 {digest}; commit it before any scientific phase")
        return 0

    # -- scientific phases ---------------------------------------------------

    def _provenance(self) -> dict[str, Any]:
        git = self.git_state()
        return {"protocol_code_commit": str(git.get("commit") or ""), "git_dirty": bool(git.get("dirty", True)), "versions": dict(self.versions())}

    def _state_for(self, phase: str, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str) -> dict[str, Any]:
        provenance = self._provenance()
        if self.results_path.exists():
            state = ht.load_results_state(self.results_path)
            if provenance["git_dirty"] or not pm._COMMIT_SHA.fullmatch(provenance["protocol_code_commit"]):
                raise ht.PhaseError("scientific execution requires a clean Git tree at a committed SHA")
            if (state["manifest_sha256"], state["extension_sha256"], state["confirmation_sha256"]) != (manifest_sha256, extension_sha256, confirmation_sha256) or dict(state["versions"]) != provenance["versions"]:
                raise ht.PhaseError("frozen inputs or dependency versions differ from the recorded run")
        else:
            if phase != "explore":
                raise ht.PhaseError(f"{phase} requires an existing results state from explore")
            state = ht.new_results_state(manifest_sha256=manifest_sha256, extension_sha256=extension_sha256, confirmation_sha256=confirmation_sha256, **provenance)
        ht.assert_phase_allowed(phase, state)
        return state

    def _record_incident(self, state: dict[str, Any], phase: str, error: Exception) -> None:
        entry = {"message": str(error), "at": pm.utc_now(), "phase": phase, "commit": self._provenance()["protocol_code_commit"]}
        if phase == "explore":
            state["exploration"].setdefault("incidents", []).append(entry)
        else:
            state["confirmation"] = {"incident": entry}
        ht.write_results_state(self.results_path, state)
        self.log(f"INCIDENT ({phase}): {error}")

    def explore(self) -> int:
        manifest, manifest_sha256, extension, confirmation, pool, inherited_006, inherited_007 = self._inputs()
        state = self._state_for("explore", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        commit = self._provenance()["protocol_code_commit"]
        incidents = state["exploration"].get("incidents", [])
        if incidents and incidents[-1]["commit"] == commit:
            raise ht.PhaseError("an explore incident is recorded at this commit; a committed fix or documented amendment is required before explore runs again")
        contract = dict(self.contract_runner())
        state["exploration"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            ht.write_results_state(self.results_path, state)
            self.log("A0 contract test failed; refusing to explore")
            return 1
        program_007 = self._program_007()
        state["protocol_code_commit"] = commit
        state["phases"]["explore"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime_record(PYTHIA_70M), "attempts": int(state["phases"]["explore"].get("attempts", 0)) + 1,
                                      "attempt_commits": list(state["phases"]["explore"].get("attempt_commits", [])) + [commit]}
        prompts = pm.manifest_prompts(pool.frames) + tuple(pool.reference_prompt(frame) for frame in pool.frames) + tuple(pool.token_prompt(frame, name) for frame in pool.frames for name, _ in pool.tokens)
        pm.record_execution(state, prompts, pool.nouns)
        ht.assert_confirmation_untouched(state, confirmation)
        ht.write_results_state(self.results_path, state)
        seed_runtime(ht.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            try:
                ht.run_exploration(model, pool, state=state, results_path=self.results_path, parameters_dir=self.parameters_dir, program_source=self.program_source, program_007=program_007,
                                   inherited_006=inherited_006, inherited_007=inherited_007, log=self.log)
            except pm.IncidentError as error:
                self._record_incident(state, "explore", error)
                return 2
            except Exception as error:
                self._record_incident(state, "explore", error)
                raise
        finally:
            del model
            gc.collect()
        ht.assert_confirmation_untouched(state, confirmation)
        state["phases"]["explore"] = {**state["phases"]["explore"], "status": "complete", "completed_at": pm.utc_now()}
        digest = ht.write_results_state(self.results_path, state)
        summary = state["exploration"]["summary"]
        self.log(f"explore complete: mechanism level {summary['locked_level']}, transport gate {summary['transport_gate']}, contrast gate {summary['contrast_gate']} (rank {summary['contrast_rank']}); results sha256 {digest}")
        return 0

    def _rules(self, exploration: Mapping[str, Any], reference_ids: Mapping[str, int]):
        transport = ht.load_transport_rule(self.parameters_dir, exploration) if exploration["transport_rule"]["gate"]["passed"] else None
        contrast = None
        if exploration["contrast_rule"]["gate"]["passed"]:
            index_text = (self.parameters_dir / "contrast" / "parameters.json").read_text(encoding="utf-8")
            if pm.sha256_text(pm.canonical_json(json.loads(index_text))) != exploration["contrast_rule"]["parameters_sha256"]:
                raise ht.PhaseError("the contrast rule parameters on disk do not match the recorded digest")
            contrast = ss.load_linear_program(self.parameters_dir / "contrast", self.program_source)
        return transport, contrast

    def _predictions(self, model_weights: pm.Weights, confirmation: ht.Confirmation009, pool: cs.Pool008, exploration: Mapping[str, Any]) -> dict[str, Any]:
        transport, contrast = self._rules(exploration, pool.reference_ids)
        program_007 = self._program_007()
        return ht.lock_predictions(weights=model_weights, confirmation=confirmation, transport=transport, contrast_program=contrast, program_007=program_007, nouns_by_key={noun.lexical_key: noun for noun in pool.single_nouns})

    def lock(self) -> int:
        manifest, manifest_sha256, extension, confirmation, pool, *_ = self._inputs()
        state = self._state_for("lock", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        exploration = state["exploration"]
        if exploration["mechanism"]["locked_level"] is None and not exploration["transport_rule"]["gate"]["passed"] and not exploration["contrast_rule"]["gate"]["passed"]:
            raise ht.PhaseError("no axis is lockable: no mechanism level qualified and both rule gates failed")
        model = self.model_loader(PYTHIA_70M)  # weights only: E(w) for the fresh tokens; no prompt is run
        try:
            weights = pm.Weights.from_model(model)
        finally:
            del model
            gc.collect()
        changed = self.changed_paths(state["protocol_code_commit"])
        scientific = [path for path in (changed or []) if path.startswith(ht.SCIENTIFIC_PATH_PREFIXES) and path not in ht.NON_SCIENTIFIC_PATHS and not path.startswith(ht.NON_SCIENTIFIC_PREFIXES)]
        if changed is None or scientific:
            raise ht.PhaseError(f"scientific paths changed since explore: {scientific if changed is not None else 'explore commit is not an ancestor'}; lock must be written at the explore protocol")
        predictions = self._predictions(weights, confirmation, pool, exploration)
        transport, _ = self._rules(exploration, pool.reference_ids)
        lock = ht.build_candidate_lock(state=state, manifest_sha256=manifest_sha256, extension=extension, confirmation=confirmation, predictions=predictions, parameters_dir=self.parameters_dir,
                                       program_source=self.program_source, protocol_code_commit=self._provenance()["protocol_code_commit"], axes_T_direction=exploration["axes_vectors"]["T"],
                                       read_direction=exploration["head"]["read_direction"], transport_v=(transport.v.tolist() if transport is not None else None))
        candidate_path = self.results_path.parent / "candidate-lock.json"
        predictions_path = self.results_path.parent / "candidate-predictions.md"
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        predictions_text = ht.render_predictions(lock)
        predictions_path.write_text(predictions_text, encoding="utf-8")
        state["lock"] = {"candidate_path": str(candidate_path), "predictions_path": str(predictions_path), "content_sha256": lock["content_sha256"], "predictions_sha256": pm.sha256_text(predictions_text), "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now()}
        ht.write_results_state(self.results_path, state)
        self.log(f"candidate lock written to {candidate_path} (sha256 {lock['content_sha256']}) and {predictions_path} (sha256 {state['lock']['predictions_sha256']}). Install both as {ht.LOCK_RELATIVE_PATH} and {ht.PREDICTIONS_RELATIVE_PATH} and commit before confirm.")
        return 0

    def confirm(self) -> int:
        manifest, manifest_sha256, extension, confirmation, pool, *_ = self._inputs()
        state = self._state_for("confirm", manifest_sha256, extension.content_sha256, confirmation.content_sha256)
        lock_path = self.root / ht.LOCK_RELATIVE_PATH
        predictions_path = self.root / ht.PREDICTIONS_RELATIVE_PATH
        if not lock_path.exists() or not predictions_path.exists():
            raise ht.PhaseError("missing committed lock or predictions artifact")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        git = self.git_state()
        ht.validate_lock(lock, state=state, manifest_sha256=manifest_sha256, extension=extension, confirmation=confirmation, predictions_text=predictions_path.read_text(encoding="utf-8"), git_state=git,
                         tracked=self.tracked(lock_path) and self.tracked(predictions_path), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        seed_runtime(ht.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            weights = pm.Weights.from_model(model)
            recomputed = self._predictions(weights, confirmation, pool, state["exploration"])
            reproduced = ht.assert_lock_predictions_reproduced(lock, recomputed)  # PhaseError before anything fresh runs
            state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": str(git.get("commit")), "lock_predictions_reproduced_max_difference": reproduced}
            pm.record_execution(state, confirmation.all_prompts + tuple(confirmation.reference_prompt(frame) for frame in confirmation.frames), pool.nouns)
            ht.write_results_state(self.results_path, state)
            try:
                results = ht.run_confirmation(model, pool, confirmation, lock, log=self.log)
            except pm.IncidentError as error:
                self._record_incident(state, "confirm", error)
                return 2
            except Exception as error:  # a software defect is an incident too: record it, then surface it
                self._record_incident(state, "confirm", error)
                raise
        finally:
            del model
            gc.collect()
        state["confirmation"] = {**results, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}
        state["phases"]["confirm"] = {**state["phases"]["confirm"], "status": "complete", "completed_at": pm.utc_now()}
        digest = ht.write_results_state(self.results_path, state)
        self.log(f"confirm complete: {results['outcome']['label']}; results sha256 {digest}")
        return 0

    def report(self) -> int:
        self._inputs()
        state = ht.load_results_state(self.results_path)
        ht.assert_phase_allowed("report", state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(ht.render_report(state), encoding="utf-8")
        self.log(f"report written to {self.report_path}")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    method = args.phase.replace("-", "_")
    if args.phase in PHASES:
        return getattr(runner, method)()
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
