#!/usr/bin/env python3
"""Run Experiment 012 through isolated phases.

``validate`` checks the frozen inputs without a model. ``freeze-confirmation`` builds ``confirmation-v1.json``
from tokenizer rules (refuses to overwrite). ``explore`` (once) measures the exposed pool (87 tokens × 30
frames) with the residuals before blocks 1 and 2 captured, replicates Experiments 010 and 011, locks the base
states, and calibrates the tolerances leave-one-frame-out. ``lock`` computes every prediction from the weights,
the Experiment 011 lock's axes, and the locked base states — it performs no forward pass and has no capture or
intervention call path — and writes the candidate lock and the prediction artifact. ``confirm`` (once) validates
the committed artifacts, reproduces every prediction before any fresh prompt, runs the fresh set, and scores it
against the committed numbers. ``report`` renders the report.
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
from neural_decompiler import encoding_read as er
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-012"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")
PHASES = ("validate", "freeze-confirmation", "explore", "lock", "confirm", "report")
LOCK_011_REQUIRED_KEYS = ("axes_vectors", "read_weight", "sigma_T", "denominators", "confirmation_011_sha256", "content_sha256")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the frozen inputs and the confirmation set without a model"), ("freeze-confirmation", "build confirmation-v1.json from tokenizer rules only (refuses to overwrite)"),
                             ("explore", "Tier A: exposed-pool measurements, replication, base states, leave-one-frame-out calibration"), ("lock", "compute every prediction from the weights and locked states; write the candidate lock and predictions"),
                             ("confirm", "Tier C: the single fresh-set run scored against the committed predictions"), ("report", "render outputs/experiment-012/report.md")):
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
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": lc.RUNTIME_SEED, "control_seed": lc.CONTROL_SEED, "deterministic_algorithms": spec.deterministic_algorithms,
            "torch_num_threads": int(torch.get_num_threads())}


def _load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


def _load_lock_011(path: Path) -> dict[str, Any]:
    """The committed Experiment 011 lock, its content digest re-verified."""
    lock = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("experiment") != "011" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise lc.PhaseError("the Experiment 011 lock's content digest does not verify")
    return lock


@dataclass
class Runner:
    root: Path = ROOT
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    tokenizer_loader: Callable[[ModelSpec], Any] = _load_tokenizer
    lock_011_loader: Callable[[Path], dict[str, Any]] = _load_lock_011  # tests inject a lock built from the fake's own axes
    git_state: Callable[[], Mapping[str, Any]] = field(default_factory=lambda: lambda: collect_git_state(ROOT))
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    changed_paths: Callable[[str], list[str] | None] = _git_changed_paths
    contract_runner: Callable[[], Mapping[str, Any]] = _run_contract_test
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))

    @property
    def confirmation_path(self) -> Path:
        return self.root / lc.CONFIRMATION_RELATIVE_PATH

    # -- inputs ---------------------------------------------------------------

    def _base_inputs(self):
        manifest, manifest_sha256, extension = pm.load_inputs(self.root)
        confirmation_006 = cd.load_confirmation(self.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
        if confirmation_006.content_sha256 != ss.INHERITED_CONFIRMATION_SHA256:
            raise lc.PhaseError("the Experiment 006 confirmation set digest is not the frozen constant")
        confirmation_009 = ht.load_confirmation(self.root / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
        confirmation_011 = er.load_confirmation(self.root / lc.EXPERIMENT_011_CONFIRMATION_PATH, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
        for relative in (cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, lc.EXPERIMENT_011_CONFIRMATION_PATH, lc.EXPERIMENT_011_LOCK_PATH, lc.INHERITED_010_LEDGER_RELATIVE_PATH, lc.INHERITED_011_LEDGER_RELATIVE_PATH):
            if not (self.root / relative).exists() or not self.tracked(self.root / relative):
                raise lc.PhaseError(f"{relative} must exist and be tracked and committed")
        lock_011 = self.lock_011_loader(self.root / lc.EXPERIMENT_011_LOCK_PATH)
        if any(key not in lock_011 for key in LOCK_011_REQUIRED_KEYS) or lock_011["confirmation_011_sha256"] != confirmation_011.content_sha256 or not {"denominators", "sigma_r"} <= set(lock_011["denominators"]):
            raise lc.PhaseError("the Experiment 011 lock does not carry the locked axes, read weight, σ_T, denominators and σ_r for the frozen 011 confirmation set")
        digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": confirmation_006.content_sha256, "confirmation_009": confirmation_009.content_sha256,
                   "confirmation_011": confirmation_011.content_sha256, "lock_011": lock_011["content_sha256"]}
        ledgers = {"010": lc.load_inherited_ledger(self.root / lc.INHERITED_010_LEDGER_RELATIVE_PATH, experiment="010", digests=digests, expected_size=63 * 24),
                   "011": lc.load_inherited_ledger(self.root / lc.INHERITED_011_LEDGER_RELATIVE_PATH, experiment="011", digests=digests, expected_size=len(confirmation_011.token_prompts))}
        digests |= {"ledger_010": ledgers["010"]["content_sha256"], "ledger_011": ledgers["011"]["content_sha256"]}
        pool_010 = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
        pool = lc.build_pool_012(manifest, extension, confirmation_006, confirmation_009, confirmation_011)
        return pool, pool_010, lock_011, ledgers, digests

    def _inputs(self):
        pool, pool_010, lock_011, ledgers, digests = self._base_inputs()
        if not self.confirmation_path.exists() or not self.tracked(self.confirmation_path):
            raise lc.PhaseError("the Experiment 012 confirmation set must be frozen and committed before any scientific phase")
        confirmation = lc.load_confirmation(self.confirmation_path, pool, digests)
        digests = dict(digests) | {"confirmation_012": confirmation.content_sha256}
        return pool, pool_010, lock_011, ledgers, confirmation, digests

    def validate(self) -> int:
        try:
            pool, pool_010, lock_011, ledgers, digests = self._base_inputs()
        except (lc.PhaseError, ValueError) as error:
            self.log(f"validation failed: {error}")
            return 1
        self.log(f"exposed pool: {len(pool.frames)} frames, {len(pool.tokens)} tokens; ledgers 010 {len(ledgers['010']['entries'])} pairs, 011 {len(ledgers['011']['entries'])} pairs; 011 lock {digests['lock_011'][:12]}…")
        if not self.confirmation_path.exists():
            self.log("confirmation set not frozen yet: run freeze-confirmation")
            return 1
        try:
            confirmation = lc.load_confirmation(self.confirmation_path, pool, digests)
        except ValueError as error:
            self.log(f"validation failed: {error}")
            return 1
        tracked = self.tracked(self.confirmation_path)
        self.log(f"confirmation ok: sha256 {confirmation.content_sha256}; {len(confirmation.tokens)} tokens {[t['word'] for t in confirmation.tokens]}; {len(confirmation.frames)} frames; {len(confirmation.token_prompts)} prompts; tracked={tracked}")
        return 0 if tracked else 1

    def freeze_confirmation(self) -> int:
        pool, pool_010, lock_011, ledgers, digests = self._base_inputs()
        if self.confirmation_path.exists():
            self.log(f"refusing to overwrite frozen confirmation set {self.confirmation_path}")
            return 1
        digest = lc.freeze_confirmation(self.confirmation_path, self.tokenizer_loader(PYTHIA_70M), pool, digests)
        self.log(f"froze confirmation set {self.confirmation_path} sha256 {digest}; commit it before any scientific phase")
        return 0

    # -- scientific phases ---------------------------------------------------

    def _provenance(self) -> dict[str, Any]:
        git = self.git_state()
        return {"protocol_code_commit": str(git.get("commit") or ""), "git_dirty": bool(git.get("dirty", True)), "versions": dict(self.versions())}

    def _state_for(self, phase: str, digests: Mapping[str, str]) -> dict[str, Any]:
        provenance = self._provenance()
        if self.results_path.exists():
            state = lc.load_results_state(self.results_path)
            if provenance["git_dirty"] or not pm._COMMIT_SHA.fullmatch(provenance["protocol_code_commit"]):
                raise lc.PhaseError("scientific execution requires a clean Git tree at a committed SHA")
            if lc.state_digests(state) != lc.digest_tuple(digests) or dict(state["versions"]) != provenance["versions"]:
                raise lc.PhaseError("frozen inputs or dependency versions differ from the recorded run")
        else:
            if phase != "explore":
                raise lc.PhaseError(f"{phase} requires an existing results state from explore")
            state = lc.new_results_state(digests=digests, **provenance)
        lc.assert_phase_allowed(phase, state)
        return state

    def _record_incident(self, state: dict[str, Any], phase: str, error: Exception) -> None:
        entry = {"message": str(error), "at": pm.utc_now(), "phase": phase, "commit": self._provenance()["protocol_code_commit"]}
        if phase == "explore":
            state["exploration"].setdefault("incidents", []).append(entry)
        else:
            state["confirmation"] = {"incident": entry}
        lc.write_results_state(self.results_path, state)
        self.log(f"INCIDENT ({phase}): {error}")

    def explore(self) -> int:
        pool, pool_010, lock_011, ledgers, confirmation, digests = self._inputs()
        state = self._state_for("explore", digests)
        commit = self._provenance()["protocol_code_commit"]
        incidents = state["exploration"].get("incidents", [])
        if incidents and incidents[-1]["commit"] == commit:
            raise lc.PhaseError("an explore incident is recorded at this commit; a committed fix or documented amendment is required before explore runs again")
        contract = dict(self.contract_runner())
        state["exploration"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            lc.write_results_state(self.results_path, state)
            self.log("A0 contract test failed; refusing to explore")
            return 1
        state["protocol_code_commit"] = commit
        state["phases"]["explore"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime_record(PYTHIA_70M), "attempts": int(state["phases"]["explore"].get("attempts", 0)) + 1,
                                      "attempt_commits": list(state["phases"]["explore"].get("attempt_commits", [])) + [commit]}
        pm.record_execution(state, pm.manifest_prompts(pool.frames) + tuple(pool.reference_prompt(frame) for frame in pool.frames), pool.nouns)
        lc.assert_confirmation_untouched(state, confirmation)
        lc.write_results_state(self.results_path, state)
        seed_runtime(lc.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            try:
                lc.run_exploration(model, pool, pool_010, lock_011=lock_011, inherited=ledgers, state=state, results_path=self.results_path, log=self.log)
            except pm.IncidentError as error:
                self._record_incident(state, "explore", error)
                return 2
            except Exception as error:
                self._record_incident(state, "explore", error)
                raise
        finally:
            del model
            gc.collect()
        lc.assert_confirmation_untouched(state, confirmation)
        state["phases"]["explore"] = {**state["phases"]["explore"], "status": "complete", "completed_at": pm.utc_now()}
        digest = lc.write_results_state(self.results_path, state)
        summary = state["exploration"]["summary"]
        self.log(f"explore complete: defined templates {summary['defined_templates']}, tau_c {summary['tau_c']:.3f}, tau_P {summary['tau_P']:.3f}, R²_LOFO {summary['r2_lofo']:.3f}; results sha256 {digest}")
        return 0

    def _weights_only(self) -> tuple[pm.Weights, lc.LayerWeights]:
        """The lock phase's only access to the model: its parameters. No prompt is run and no capture API is reachable from here."""
        model = self.model_loader(PYTHIA_70M)
        try:
            return pm.Weights.from_model(model), lc.LayerWeights.from_model(model)
        finally:
            del model
            gc.collect()

    def _predictions(self, weights: pm.Weights, lw: lc.LayerWeights, source: Mapping[str, Any], confirmation: lc.Confirmation012, pool: Any) -> dict[str, Any]:
        """``source`` is the exploration record at lock time and the lock itself at confirm time (both carry the axes, read weight, base states and validity)."""
        plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
        read = lc.read_from_lock_011(source, pool.reference_ids, plural_ids)
        return lc.lock_predictions(weights, lw, read, lc.bases_from_json(source["base_states"]), confirmation, source["denominators"]["defined"])

    def lock(self) -> int:
        pool, pool_010, lock_011, ledgers, confirmation, digests = self._inputs()
        state = self._state_for("lock", digests)
        changed = self.changed_paths(state["protocol_code_commit"])
        if changed is None:
            raise lc.PhaseError("the explore commit is not an ancestor of the current commit; lock must be written at the explore protocol")
        scientific = [path for path in changed if path.startswith(lc.SCIENTIFIC_PATH_PREFIXES) and path not in lc.NON_SCIENTIFIC_PATHS and not path.startswith(lc.NON_SCIENTIFIC_PREFIXES)]
        if scientific:
            raise lc.PhaseError(f"scientific paths changed since explore: {scientific}; lock must be written at the explore protocol")
        weights, lw = self._weights_only()
        predictions = self._predictions(weights, lw, state["exploration"], confirmation, pool)
        lock = lc.build_candidate_lock(state=state, digests=digests, confirmation=confirmation, predictions=predictions, protocol_code_commit=self._provenance()["protocol_code_commit"])
        candidate_path = self.results_path.parent / "candidate-lock.json"
        predictions_path = self.results_path.parent / "candidate-predictions.md"
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        predictions_text = lc.render_predictions(lock)
        predictions_path.write_text(predictions_text, encoding="utf-8")
        state["lock"] = {"candidate_path": str(candidate_path), "predictions_path": str(predictions_path), "content_sha256": lock["content_sha256"], "predictions_sha256": pm.sha256_text(predictions_text), "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now()}
        lc.write_results_state(self.results_path, state)
        self.log(f"candidate lock written to {candidate_path} (sha256 {lock['content_sha256']}) and {predictions_path} (sha256 {state['lock']['predictions_sha256']}). Install both as {lc.LOCK_RELATIVE_PATH} and {lc.PREDICTIONS_RELATIVE_PATH} and commit before confirm.")
        return 0

    def confirm(self) -> int:
        pool, pool_010, lock_011, ledgers, confirmation, digests = self._inputs()
        state = self._state_for("confirm", digests)
        lock_path, predictions_path = self.root / lc.LOCK_RELATIVE_PATH, self.root / lc.PREDICTIONS_RELATIVE_PATH
        if not lock_path.exists() or not predictions_path.exists():
            raise lc.PhaseError("missing committed lock or predictions artifact")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        git = self.git_state()
        lc.validate_lock(lock, state=state, digests=digests, confirmation=confirmation, predictions_text=predictions_path.read_text(encoding="utf-8"), git_state=git,
                         tracked=self.tracked(lock_path) and self.tracked(predictions_path), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        seed_runtime(lc.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            weights, lw = pm.Weights.from_model(model), lc.LayerWeights.from_model(model)
            reproduced = lc.assert_lock_predictions_reproduced(lock, self._predictions(weights, lw, lock, confirmation, pool))  # from the locked axes and base states; PhaseError before anything fresh runs
            state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": str(git.get("commit")), "lock_predictions_reproduced_max_difference": reproduced}
            defined_frames = [frame for frame in confirmation.frames if frame.template_id in lock["defined_templates"]]
            prompts = tuple(prompt for prompt in confirmation.all_prompts if prompt.frame.template_id in lock["defined_templates"]) + tuple(confirmation.reference_prompt(frame) for frame in defined_frames)
            pm.record_execution(state, prompts, pool.nouns)
            lc.write_results_state(self.results_path, state)
            try:
                results = lc.run_confirmation(model, pool, pool_010, confirmation, lock, log=self.log)
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
        digest = lc.write_results_state(self.results_path, state)
        self.log(f"confirm complete: {results['outcome']['label']}; results sha256 {digest}")
        return 0

    def report(self) -> int:
        self._inputs()
        state = lc.load_results_state(self.results_path)
        lc.assert_phase_allowed("report", state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(lc.render_report(state), encoding="utf-8")
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
