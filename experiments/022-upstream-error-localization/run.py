#!/usr/bin/env python3
"""Run Experiment 022 through isolated phases (design revision 4, ``219cdc5``; plan revision 3, ``e6d8299``).

``validate`` checks the frozen inputs, the pinned module blobs, Experiment 020's closure, Experiment 021's committed
calibration record and local exposed table, and — once they exist — the confirmation file and the results state,
without a model. ``freeze`` (tokenizer only) writes the new cues and frames to ``confirmation-v1.json``, which is
committed by hand before anything else. ``calibrate`` (once) re-measures the 18,900 exposed pairs of Experiment 021's
pools against Experiment 020's locked states, composes every coalition, enforces I1–I5 and R1, draws 10,000 SHA-indexed
exposed-like populations, runs the kernel with its direct cross-check, applies the undefined-draw stop, and writes the
candidate calibration record; it is installed byte-identical and committed, and the floors are reviewed, before
``lock`` — which runs no forward pass — writes the Y1 table companion, the candidate lock and the preregistration.
``confirm`` (once, never resumed) validates the lock, reconstructs the Y1 table bit for bit (I7) before any fresh
prompt, runs stage 1 (each new frame's S1-REF and S1-VALIDITY prompts) and writes the Y2 table, re-reads and verifies
both at the barrier, then runs stage 2, saves every measurement, enforces the gates and scores the eight conditions.
``report`` renders the report; ``replicate-021`` (only after it) is the exploratory replication on Experiment 021's
spent set.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import upstream_localization as ul
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-022"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")
PHASES = ul.PHASES
PhaseError = ul.PhaseError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the frozen inputs, the module blobs, 020's closure and 021's record without a model"),
                             ("freeze", "tokenizer only: freeze the new cues and frames into confirmation-v1.json (commit it by hand)"),
                             ("calibrate", "once: the exposed re-materialization, the gates, R1, the draws, the kernel check and the candidate record"),
                             ("lock", "no forward pass: the Y1 table companion, the candidate lock and the preregistration"),
                             ("confirm", "once, never resumed: I7, stage 1 and the Y2 table, the barrier, stage 2, the gates and the eight conditions"),
                             ("report", "render the report from the results state"),
                             ("replicate-021", "after report only: the exploratory replication on Experiment 021's spent set")):
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


def _load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


def runtime_record(spec: ModelSpec) -> dict[str, Any]:
    runtime = validate_runtime(spec)
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": rd.RUNTIME_SEED, "control_seed": rd.CONTROL_SEED,
            "deterministic_algorithms": spec.deterministic_algorithms, "torch_num_threads": int(torch.get_num_threads())}


class pytest_free_guard:
    """Disable every capture and intervention entry point for the duration of a block: the executed proof that the
    freeze, the lock's compositions, the I7 reconstruction and the replication cannot reach a forward pass."""

    _NAMES = ("capture_prompt", "run_patched", "run_capture", "run_interventions")

    def __enter__(self):
        self._saved = {name: getattr(pm, name) for name in self._NAMES if hasattr(pm, name)}

        def refuse(*args, **kwargs):
            raise PhaseError("a forward pass was reached where none may run")

        for name in self._saved:
            setattr(pm, name, refuse)
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(pm, name, value)
        return False


@dataclass
class Runner:
    root: Path = ROOT
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    tokenizer_loader: Callable[[ModelSpec], Any] = _load_tokenizer
    lock_011_loader: Callable[[Path], Mapping[str, Any]] | None = None  # tests inject locks built from the fake
    lock_012_loader: Callable[[Path], Mapping[str, Any]] | None = None
    lock_017_loader: Callable[[Path], Mapping[str, Any]] | None = None
    git_state: Callable[[], Mapping[str, Any]] = collect_git_state
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    changed_paths: Callable[[str], list[str] | None] = _git_changed_paths
    contract_runner: Callable[[], Mapping[str, Any]] = _run_contract_test
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))

    # -- paths ----------------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        return self.results_path.parent

    def output(self, name: str) -> Path:
        return self.output_dir / name

    @property
    def candidate_calibration_path(self) -> Path:
        return self.output("candidate-calibration.json")

    @property
    def table_path(self) -> Path:
        return self.output("calibration-table.pt")

    @property
    def draws_path(self) -> Path:
        return self.output("draw-values.pt")

    @property
    def stage2_path(self) -> Path:
        return self.output("stage2-measurements.pt")

    @property
    def replication_path(self) -> Path:
        return self.output("replicate-021.json")

    # -- inputs ---------------------------------------------------------------

    def _inputs(self) -> ul.FrozenInputs:
        inputs = ul.load_frozen_inputs(self.root, lock_011_loader=self.lock_011_loader, lock_012_loader=self.lock_012_loader, lock_017_loader=self.lock_017_loader,
                                       tracked=self.tracked)
        ul.assert_frozen_blobs()
        return inputs

    def _confirmation(self, inputs: ul.FrozenInputs) -> tuple[ul.Confirmation022, str]:
        path = self.root / ul.CONFIRMATION_RELATIVE_PATH
        if not path.exists() or not self.tracked(path):
            raise PhaseError(f"{ul.CONFIRMATION_RELATIVE_PATH} must be frozen and committed first (the freeze phase)")
        return ul.load_confirmation_022(path, inputs), rc.file_sha256(path)

    def _all(self):
        """Every frozen input, the committed confirmation file, Experiment 021's committed record, and the forbidden
        keys (022's manifest and Experiment 020's confirmation set), which Experiment 020's ledger must not hold."""
        inputs = self._inputs()
        confirmation, confirmation_sha = self._confirmation(inputs)
        digests = ul.input_digests(inputs, confirmation, confirmation_sha, ul.verify_021_record(self.root))
        forbidden = confirmation.manifest_keys() | frozenset(prompt.key for prompt in inputs.confirmation_020.all_prompts)
        ul.assert_ledger_isolated(inputs.closure["state"]["executed_prompt_keys"], forbidden, "Experiment 020's ledger")
        return inputs, confirmation, confirmation_sha, digests, forbidden

    def _noun_keys(self, inputs: ul.FrozenInputs) -> list[str]:
        """The scorable exposed nouns in ``NounSet`` order (no weight is needed: scorability is single-tokenness)."""
        return [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]

    # -- non-scientific phases ------------------------------------------------

    def validate(self) -> int:
        try:
            inputs = self._inputs()
            pools = rc.production_pools(inputs.pool)
            digests_021 = ul.verify_021_record(self.root)
            table_021 = self.root / ul.EXPOSED_TABLE_021_RELATIVE_PATH
            if not table_021.exists() or rc.tensor_digest(torch.load(table_021)["measured"]) != digests_021["exposed_measured_021"]:
                raise PhaseError(f"{ul.EXPOSED_TABLE_021_RELATIVE_PATH} is missing or not the table Experiment 021's record digests")
            path = self.root / ul.CONFIRMATION_RELATIVE_PATH
            frozen = "not frozen yet"
            if path.exists():
                inputs, confirmation, confirmation_sha, digests, forbidden = self._all()
                frozen = f"confirmation {confirmation.content_sha256[:12]}… ({len(confirmation.tokens)} cues, {len(confirmation.frames)} frames, {len(confirmation.manifest_keys())} keys)"
                if self.results_path.exists():
                    state = rd.load_results_state(self.results_path)
                    if state["inputs"] != digests:
                        raise PhaseError("the results state was written against different frozen inputs")
                    if state["phases"]["confirm"]["status"] == "not_started":
                        ul.assert_ledger_isolated(state["executed_prompt_keys"], forbidden, "Experiment 022's ledger before confirm")
        except (PhaseError, rd.PhaseError, pm.IncidentError, ValueError, KeyError) as error:
            self.log(f"validation failed: {error}")
            return 1
        self.log(f"module blobs {len(ul.FROZEN_BLOBS)} verified; Experiment 020 closed (ledger {len(inputs.closure['ledger'])} keys); Experiment 021 record "
                 f"{digests_021['calibration_021_content'][:12]}… and exposed table verified; pools: cues {', '.join(f'{c} {len(v)}' for c, v in pools.cues.items())}, "
                 f"Y2-like frames {', '.join(f'{t} {len(v)}' for t, v in pools.frames_unscreened.items())}; {frozen}")
        return 0

    def freeze(self) -> int:
        inputs = self._inputs()
        path = self.root / ul.CONFIRMATION_RELATIVE_PATH
        if path.exists():
            raise PhaseError(f"{ul.CONFIRMATION_RELATIVE_PATH} already exists; the freeze runs once")
        if self.results_path.exists():
            raise PhaseError("a results state exists; the freeze precedes calibrate")
        tokenizer = self.tokenizer_loader(PYTHIA_70M)
        try:
            with pytest_free_guard():
                payload = ul.freeze_payload(tokenizer, inputs)
        except ul.FreezeShortfall as error:
            self.log(f"freeze shortfall: {error}; nothing was written; stopped for review")
            return 3
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
        confirmation = ul.load_confirmation_022(path, inputs)  # re-read and verified from disk
        self.log(f"froze {len(confirmation.tokens)} cues and {len(confirmation.frames)} frames to {path} (content sha256 {confirmation.content_sha256}, file sha256 "
                 f"{rc.file_sha256(path)}); commit it by hand before calibrate")
        return 0

    # -- scientific phases ----------------------------------------------------

    def _provenance(self) -> dict[str, Any]:
        git = self.git_state()
        return {"protocol_code_commit": str(git.get("commit") or ""), "git_dirty": bool(git.get("dirty", True)), "versions": dict(self.versions())}

    def _state_for(self, phase: str, digests: Mapping[str, str]) -> dict[str, Any]:
        provenance = self._provenance()
        if self.results_path.exists():
            state = rd.load_results_state(self.results_path)
            if provenance["git_dirty"] or not pm._COMMIT_SHA.fullmatch(provenance["protocol_code_commit"]):
                raise PhaseError("scientific execution requires a clean Git tree at a committed SHA")
            if state.get("experiment") != ul.EXPERIMENT or state["inputs"] != {key: digests[key] for key in ul.DIGEST_KEYS} or dict(state["versions"]) != provenance["versions"]:
                raise PhaseError("frozen inputs or dependency versions differ from the recorded run")
            if phase == "calibrate":
                self._reconcile_unrecorded_attempt(state)
        else:
            if phase != "calibrate":
                raise PhaseError(f"{phase} requires an existing results state from calibrate")
            state = ul.new_results_state(digests=digests, **provenance)
        ul.assert_phase_allowed(phase, state)
        return state

    def _write(self, state: Mapping[str, Any]) -> str:
        return rd.write_results_state(self.results_path, state)

    def _reconcile_unrecorded_attempt(self, state: dict[str, Any]) -> None:
        """An attempt that ended without writing its incident is recorded now as an incident of its own commit."""
        phase = state["phases"]["calibrate"]
        attempts = list(phase.get("attempt_commits", []))
        recorded = {entry["commit"] for entry in state["calibration"].get("incidents", [])}
        if phase.get("status") == "running" and attempts and attempts[-1] not in recorded and not state["calibration"].get("record_sha256"):
            state["calibration"].setdefault("incidents", []).append({"message": f"the calibrate attempt at {attempts[-1]} ended without recording an incident (killed or interrupted)",
                                                                     "type": "UnrecordedTermination", "at": pm.utc_now(), "phase": "calibrate", "commit": attempts[-1]})
            self._write(state)
            self.log(f"recorded the unrecorded end of the calibrate attempt at {attempts[-1]}")

    def _recheck(self, inputs: ul.FrozenInputs) -> dict[str, Any]:
        """Experiment 020's closure and Experiment 021's record, verified again after a phase (or an incident)."""
        try:
            rc.verify_020_closure(self.root, inputs.confirmation_020)
            ul.verify_021_record(self.root)
        except Exception as error:  # recorded; never allowed to hide the incident being recorded
            return {"ok": False, "message": str(error)}
        return {"ok": True}

    def _record_incident(self, state: dict[str, Any], phase: str, error: BaseException, inputs: ul.FrozenInputs | None = None) -> None:
        entry_phase = state.get("phases", {}).get(phase, {})
        commit = str(entry_phase.get("commit") or entry_phase.get("confirm_commit") or state.get("protocol_code_commit") or "")
        entry = {"message": str(error) or type(error).__name__, "type": type(error).__name__, "at": pm.utc_now(), "phase": phase, "commit": commit}
        if phase == "calibrate":
            state["calibration"] = rc.json_safe(state["calibration"])
            state["calibration"].setdefault("incidents", []).append(entry)
        else:
            state["confirmation"] = rc.json_safe(state.get("confirmation") or {})
            state["confirmation"]["incident"] = entry
        self._write(state)  # the incident is on disk before anything slow runs
        self.log(f"INCIDENT ({phase}): {error}")
        if inputs is not None:
            entry["recheck_020_021"] = self._recheck(inputs)
            self._write(state)

    def _record_phase_incident(self, state: dict[str, Any], phase: str, error: BaseException, inputs: ul.FrozenInputs) -> None:
        """An identity failure before any fresh prompt (lock's I2/I5 and provenance, confirm's I7): recorded with its
        commit before anything else, after which the phase is refused (never retried until it passes)."""
        entry = {"message": str(error) or type(error).__name__, "type": type(error).__name__, "at": pm.utc_now(), "phase": phase,
                 "commit": self._provenance()["protocol_code_commit"]}
        state["phases"][phase] = {**state["phases"][phase], "incidents": list(state["phases"][phase].get("incidents", [])) + [entry]}
        self._write(state)
        self.log(f"INCIDENT ({phase}): {error}")
        entry["recheck_020_021"] = self._recheck(inputs)
        self._write(state)

    def _check_runtime(self, closure: Mapping[str, Any]) -> dict[str, Any]:
        """The runtime and the dependency versions of Experiment 020's explore (021's check, unchanged)."""
        runtime = runtime_record(PYTHIA_70M)
        recorded = closure["state"]["phases"]["explore"].get("runtime") or {}
        if {k: v for k, v in runtime.items() if k != "seed"} != {k: v for k, v in recorded.items() if k != "seed"}:
            raise PhaseError(f"the runtime {runtime} differs from Experiment 020's explore runtime {recorded}")
        if dict(self.versions()) != dict(closure["state"]["versions"]):
            raise PhaseError(f"the dependency versions {dict(self.versions())} differ from Experiment 020's {closure['state']['versions']}")
        return runtime

    def calibrate(self) -> int:
        inputs, confirmation, confirmation_sha, digests, forbidden = self._all()
        state = self._state_for("calibrate", digests)
        commit = self._provenance()["protocol_code_commit"]
        if any(entry["commit"] == commit for entry in state["calibration"].get("incidents", [])):
            raise PhaseError("a calibrate incident is recorded at this commit; a committed fix is required before calibrate runs again")
        if commit in state["phases"]["calibrate"].get("attempt_commits", []):
            raise PhaseError("calibrate was already attempted at this commit; a new commit is required")
        runtime = self._check_runtime(inputs.closure)
        contract = dict(self.contract_runner())
        state["calibration"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            self._write(state)
            self.log("A0 contract test failed; refusing to calibrate")
            return 1
        units = ul.calibration_units(inputs.pool, confirmation.counts())  # the confirmation file's counts are all the calibration reads of it
        expected = {pm.Prompt(frame, token_id, word).key for word, token_id, _ in units.cues for frame in units.frames}
        state["protocol_code_commit"] = commit
        previous = state["phases"]["calibrate"]
        state["phases"]["calibrate"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime, "commit": commit, "attempts": int(previous.get("attempts", 0)) + 1,
                                        "attempt_commits": list(previous.get("attempt_commits", [])) + [commit]}
        self._write(state)
        model = None
        executed: list[pm.Prompt] = []
        calibration = state["calibration"]
        try:
            try:
                seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
                model = self.model_loader(PYTHIA_70M)
                progs = ul.ModelPrograms.from_model(model, inputs)
                noun_keys = [progs.nouns.nouns[index].lexical_key for index in progs.scorable]
                def before_frame(prompts) -> None:  # a frame's keys reach the ledger on disk before any of them runs
                    pm.record_execution(state, prompts, inputs.pool.nouns)
                    self._write(state)

                try:
                    table = ul.calibration_rematerialize(model, progs, inputs, units, forbidden_keys=forbidden, log=self.log, executed=executed, before_frame=before_frame)
                finally:
                    pm.record_execution(state, executed, inputs.pool.nouns)
                    self._write(state)
                gates = ul.gate_summary(table.gates, units)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(table.saved(units, noun_keys), self.table_path)
                calibration.update({"gates": gates, "table_sha256": table.digests(), "units": units.to_json()})
                self._write(state)  # the maxima and the table reach disk before anything can stop the phase
                counts = Counter(prompt.key for prompt in executed)
                if set(counts) != expected or any(count != 1 for count in counts.values()) or not expected <= inputs.closure["ledger"]:
                    raise pm.IncidentError("the re-materialization did not execute exactly the calibration keys, once each, all in Experiment 020's ledger")
                ul.assert_measurements_finite(table)
                ul.enforce_gates(gates)
                r1 = ul.r1_gate(self.root, table, units, noun_keys)
                calibration["r1"] = r1
                self._write(state)
                if not r1["passed"]:
                    raise pm.IncidentError(f"R1 failed: the re-measured exposed Δc differs from Experiment 021's table by {r1['max_difference']:.3e} at {r1['at']}")
                self.log(f"gates: {', '.join(f'{name} {entry['max']:.1e}' for name, entry in gates.items())}; R1 {r1['max_difference']:.1e}")
                draws = ul.draw_units(units, ul.B)
                kernel = ul.kernel_statistics(table, units, draws)
                cross_check = ul.kernel_direct_check(table, units, draws, kernel)
                efficiency = ul.efficiency_maxima(kernel)
                calibration.update({"cross_check": cross_check, "efficiency_I6": efficiency})
                self._write(state)
                ul.enforce_efficiency(efficiency)
                ul.assert_finite_where_interpretable(kernel["claims"])
                undefined = ul.undefined_counts(kernel["claims"])
                arrays = ul.draw_arrays(kernel)
                array_digests = {key: rc.tensor_digest(values) for key, values in arrays.items()}
                torch.save(arrays, self.draws_path)
                calibration.update({"undefined_counts": undefined, "draw_arrays_sha256": array_digests, "draw_index_sha256": draws["digests"]})
                stop = ul.calibration_stop(undefined, ul.B)
                if stop["stop"]:
                    calibration["stop"] = stop
                    state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "stopped_for_review", "stopped_at": pm.utc_now()}
                    self._write(state)
                    self.log(f"calibration stop (design revision 3): undefined draws {stop['offending']} at or above {stop['thresholds']}; no floor table is written; stopped for review")
                    return 3
                evaluated = ul.envelopes_and_rates(kernel["claims"])
                record = ul.calibration_record(run_id=state["run_id"], protocol_code_commit=commit, digests=digests, units=units, confirmation_sha256=confirmation.content_sha256,
                                               gates=gates, r1=r1, table_digests=table.digests(), draws=draws, cross_check=cross_check, efficiency=efficiency,
                                               undefined=undefined, evaluated=evaluated, descriptives=ul.exposed_descriptives(table, units), array_digests=array_digests,
                                               n_draws=ul.B)
                recheck = self._recheck(inputs)
                if not recheck["ok"]:
                    raise pm.IncidentError(f"Experiment 020's closure or Experiment 021's record no longer verifies: {recheck['message']}")
                text = pm.canonical_json(record) + "\n"
                self.candidate_calibration_path.write_text(text, encoding="utf-8")
                calibration.update({"record_sha256": pm.sha256_text(text), "record_content_sha256": record["content_sha256"], "candidate_path": str(self.candidate_calibration_path),
                                    "envelopes": record["envelopes"], "guard_bound": record["guard_bound"], "joint_rates": record["joint_rates"]})
                self._write(state)  # the record's digest reaches disk with the record; nothing after this can fail the phase
            except ul.CrossCheckError as error:
                calibration["cross_check"] = rc.json_safe(error.details)
                self._record_incident(state, "calibrate", error, inputs)
                return 2
            except pm.IncidentError as error:
                self._record_incident(state, "calibrate", error, inputs)
                return 2
            except BaseException as error:  # a protocol failure or an interruption: recorded, then raised
                self._record_incident(state, "calibrate", error, inputs)
                raise
        finally:
            if model is not None:
                del model
            gc.collect()
        state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "complete", "completed_at": pm.utc_now()}
        digest = self._write(state)
        self.log(f"calibrate complete: candidate record {self.candidate_calibration_path} (sha256 {calibration['record_sha256']}); results sha256 {digest}. "
                 f"Install it byte-identical as {ul.CALIBRATION_RELATIVE_PATH}, commit it, and stop for the floor review before lock.")
        return 0

    def _weights_only(self, inputs: ul.FrozenInputs) -> ul.ModelPrograms:
        """The only access of lock and replicate-021 to the model: its parameters. No prompt runs."""
        model = self.model_loader(PYTHIA_70M)
        try:
            return ul.ModelPrograms.from_model(model, inputs)
        finally:
            del model
            gc.collect()

    def _installed_record(self, state: Mapping[str, Any], digests: Mapping[str, str]) -> tuple[dict[str, Any], str]:
        path = self.root / ul.CALIBRATION_RELATIVE_PATH
        if not path.exists() or not self.tracked(path):
            raise PhaseError(f"install the candidate calibration record byte-identical as {ul.CALIBRATION_RELATIVE_PATH} and commit it before lock")
        sha = rc.file_sha256(path)
        if sha != state["calibration"].get("record_sha256"):
            raise PhaseError("the installed calibration record is not the candidate this run wrote")
        record = json.loads(path.read_text(encoding="utf-8"))
        ul.verify_calibration_record(record)
        if record["inputs"] != dict(digests):
            raise PhaseError("the calibration record was computed from different inputs")
        return record, sha

    def lock(self) -> int:
        inputs, confirmation, confirmation_sha, digests, forbidden = self._all()
        state = self._state_for("lock", digests)
        ul.assert_ledger_isolated(state["executed_prompt_keys"], forbidden, "Experiment 022's ledger")
        changed = self.changed_paths(state["phases"]["calibrate"]["commit"])
        if changed is None:
            raise PhaseError("the calibrate commit is not an ancestor of the current commit; lock must be written at the calibrated protocol")
        scientific = ul.scientific_changes(changed)
        if scientific:
            raise PhaseError(f"scientific paths changed since calibrate: {scientific}; lock must be written at the calibrated protocol")
        record, record_sha = self._installed_record(state, digests)
        self._check_runtime(inputs.closure)  # the locked table must reproduce bit for bit at confirm, on 020's runtime
        progs = self._weights_only(inputs)
        noun_keys = [progs.nouns.nouns[index].lexical_key for index in progs.scorable]
        if noun_keys != self._noun_keys(inputs):
            raise PhaseError("the scorable exposed nouns are not the pool's single-token nouns in order")
        units = ul.table_units(confirmation.tokens, confirmation.exposed_frames)
        states = ul.y1_states(inputs.closure["exploration"]["locked_states"], units.frames)
        data_path, index_path = self.output("candidate-locked-y1-table.f64"), self.output("candidate-locked-y1-table.json")
        try:
            with pytest_free_guard():  # the provenance invariant, executed rather than asserted
                tables = ul.composition_tables(progs, units, states, confirmation.reference_ids, log=self.log)
                ul.enforce_table_gates(tables["gates"])
                index = ul.write_table(data_path, index_path, tables["blocks"], ul.table_meta("Y1", units, noun_keys))
                again = ul.composition_tables(progs, units, states, confirmation.reference_ids)
            if ul.table_bytes(again["blocks"]) != data_path.read_bytes() or (again["level0_dx3_sha256"], again["factors_sha256"]) != (tables["level0_dx3_sha256"], tables["factors_sha256"]):
                raise pm.IncidentError("provenance: the Y1 table does not reproduce bit for bit from the weights and the locked inputs")
        except pm.IncidentError as error:
            self._record_phase_incident(state, "lock", error, inputs)
            return 2
        lock = ul.build_lock(run_id=state["run_id"], protocol_code_commit=self._provenance()["protocol_code_commit"], digests=digests, record=record, record_file_sha256=record_sha,
                             confirmation=confirmation, confirmation_file_sha256=confirmation_sha, y1_index=index, y1_index_sha256=rc.file_sha256(index_path), y1_tables=tables,
                             noun_keys=noun_keys, exposed_states=inputs.closure["exploration"]["locked_states"])
        candidate_path, preregistration_path = self.output("candidate-lock.json"), self.output("candidate-preregistration.md")
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        preregistration = ul.render_preregistration(lock)
        preregistration_path.write_text(preregistration, encoding="utf-8")
        recheck = self._recheck(inputs)
        if not recheck["ok"]:
            raise PhaseError(f"Experiment 020's closure or Experiment 021's record no longer verifies after lock: {recheck['message']}")
        state["lock"] = {"candidate_path": str(candidate_path), "preregistration_path": str(preregistration_path), "y1_data_path": str(data_path), "y1_index_path": str(index_path),
                         "content_sha256": lock["content_sha256"], "preregistration_sha256": pm.sha256_text(preregistration), "y1_file_sha256": index["file_sha256"],
                         "y1_index_sha256": rc.file_sha256(index_path), "calibration_content_sha256": record["content_sha256"], "gates": tables["gates"], "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now()}
        self._write(state)
        self.log(f"candidate lock {candidate_path} (content sha256 {lock['content_sha256']}), {preregistration_path} and the Y1 table {data_path} ({index['total_bytes']} bytes, "
                 f"sha256 {index['file_sha256']}) with its index. Install all four byte-identical as {ul.LOCK_RELATIVE_PATH}, {ul.PREREGISTRATION_RELATIVE_PATH}, "
                 f"{ul.Y1_TABLE_RELATIVE_PATH} and {ul.Y1_TABLE_INDEX_RELATIVE_PATH}, commit, and stop for the independent lock review before confirm.")
        return 0

    def confirm(self) -> int:
        inputs, confirmation, confirmation_sha, digests, forbidden = self._all()
        state = self._state_for("confirm", digests)
        ul.assert_ledger_isolated(state["executed_prompt_keys"], forbidden, "Experiment 022's ledger")
        paths = {name: self.root / relative for name, relative in (("lock", ul.LOCK_RELATIVE_PATH), ("preregistration", ul.PREREGISTRATION_RELATIVE_PATH),
                                                                     ("y1_data", ul.Y1_TABLE_RELATIVE_PATH), ("y1_index", ul.Y1_TABLE_INDEX_RELATIVE_PATH))}
        if not all(path.exists() for path in paths.values()):
            raise PhaseError("the committed lock, preregistration and Y1 table companion must all exist")
        lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
        record, record_sha = self._installed_record(state, digests)
        git = self.git_state()
        y1_index = ul.validate_lock(lock, state=state, digests=digests, record=record, record_file_sha256=record_sha, confirmation=confirmation, confirmation_file_sha256=confirmation_sha,
                                    noun_keys=self._noun_keys(inputs), exposed_states=inputs.closure["exploration"]["locked_states"],
                                    preregistration_text=paths["preregistration"].read_text(encoding="utf-8"), y1_index_text=paths["y1_index"].read_text(encoding="utf-8"),
                                    y1_file_sha256=rc.file_sha256(paths["y1_data"]), git_state=git, tracked=all(self.tracked(path) for path in paths.values()),
                                    changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        if any((self.root / lock["y2_table"][key]).exists() for key in ("data_path", "index_path")):
            raise PhaseError("a Y2 table is already on disk before confirm; the Y2 table is written once, at stage 1")
        commit = str(git.get("commit"))
        self._check_runtime(inputs.closure)
        seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            progs = ul.ModelPrograms.from_model(model, inputs)
            units_y1 = ul.table_units(confirmation.tokens, confirmation.exposed_frames)
            states = {"Y1": ul.y1_states(inputs.closure["exploration"]["locked_states"], units_y1.frames)}
            with pytest_free_guard():  # I7: the committed Y1 table, rebuilt from the weights and the locked inputs, bit for bit, before any fresh prompt
                rebuilt = ul.composition_tables(progs, units_y1, states["Y1"], confirmation.reference_ids, log=self.log)
            if ul.table_bytes(rebuilt["blocks"]) != paths["y1_data"].read_bytes() or rebuilt["level0_dx3_sha256"] != lock["y1_table"]["level0_dx3_sha256"] \
                    or rebuilt["factors_sha256"] != lock["y1_table"]["factors_sha256"]:
                self._record_phase_incident(state, "confirm", pm.IncidentError("I7 failed: the committed Y1 table does not reproduce bit for bit; no fresh prompt has run"), inputs)
                return 2
            y1_blocks = ul.read_table(paths["y1_data"], y1_index)
            state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": commit,
                                          "I7": {"bitwise_equal": True, "file_sha256": lock["y1_table"]["file_sha256"]}}
            pm.record_execution(state, confirmation.stage1_prompts, inputs.pool.nouns)
            self._write(state)
            try:
                executed_1: list[pm.Prompt] = []
                stage1 = ul.stage_one_022(model, progs, confirmation, lock, root=self.root, protocol_code_commit=commit, executed=executed_1, log=self.log)
                expected = ul.stage_one_expectations(stage1)  # kept in memory: the barrier compares the re-read artifacts with what stage 1 wrote
                state["confirmation"] = {"stage1": stage1}
                self._write(state)  # the stage-1 record, with the Y2 table's digests, is on disk
                if Counter(prompt.key for prompt in executed_1) != Counter(prompt.key for prompt in confirmation.stage1_prompts):
                    raise pm.IncidentError("stage 1 did not execute exactly the S1-REF and S1-VALIDITY prompts, once each")
                ul.enforce_table_gates(stage1["y2_table"]["gates"])
                # Hard boundary: the digested stage-1 record and the Y2 table are re-read from disk and verified before any S2-TARGET prompt.
                state = rd.load_results_state(self.results_path)
                y2_blocks = ul.barrier_022(self.root, state, lock, confirmation, expected)
                stage1 = state["confirmation"]["stage1"]
                states["Y2"] = {frame.frame_id: rd.state_from_locked(stage1["states"][frame.frame_id], frame) for frame in confirmation.frames}
                self.log("barrier passed: the stage-1 record and the Y2 table verified from disk; no S2-TARGET prompt has run; stage 2 begins")
                pm.record_execution(state, confirmation.target_prompts, inputs.pool.nouns)
                self._write(state)
                executed_2: list[pm.Prompt] = []
                measured = ul.stage_two_022(model, progs, confirmation, states, executed=executed_2, log=self.log)
                tensors = ul.measurement_tensors(measured)
                torch.save(tensors, self.stage2_path)
                state["confirmation"]["stage2"] = {"path": str(self.stage2_path), "tensors_sha256": {key: rc.tensor_digest(value) for key, value in tensors.items()},
                                                   "n_executed": len(executed_2)}
                self._write(state)  # every fresh measurement is on disk before any gate or score can stop the phase
                if Counter(prompt.key for prompt in executed_2) != Counter(prompt.key for prompt in confirmation.target_prompts):
                    raise pm.IncidentError("stage 2 did not execute exactly the S2-TARGET prompts, once each")
                for key, value in tensors.items():
                    if key.endswith(("/dc", "/ceiling", "/dx1", "/dx3")) and not bool(torch.isfinite(value).all()):
                        raise pm.IncidentError(f"a missing or non-finite stage-2 measurement in {key}")
                after = ul.verified_y2_blocks(self.root, stage1, lock, expected)  # any change after the barrier is an incident
                if any(not torch.equal(after[name], y2_blocks[name]) for name in y2_blocks):
                    raise pm.IncidentError("the Y2 table changed after the barrier")
                tables = {"Y1": y1_blocks, "Y2": y2_blocks}
                checked = ul.target_gates(progs, confirmation, measured, tables, states)
                state["confirmation"]["gates"] = checked["gates"]
                self._write(state)
                ul.enforce_target_gates(checked["gates"])
                results = ul.score_022(measured, tables, lock)
                descriptives = {"subsets": ul.fresh_descriptives(measured, tables), "dx3_relative_error": checked["dx3_relative_error"], "block0_profile": checked["block0_profile"],
                                "historical_comparator": checked["historical_comparator"]}
                recheck = self._recheck(inputs)
                if not recheck["ok"]:
                    raise pm.IncidentError(f"Experiment 020's closure or Experiment 021's record no longer verifies after confirm: {recheck['message']}")
            except ul.CrossCheckError as error:
                state["confirmation"] = rc.json_safe(state.get("confirmation") or {})
                state["confirmation"]["cross_check"] = rc.json_safe(error.details)
                self._record_incident(state, "confirm", error, inputs)
                return 2
            except pm.IncidentError as error:
                self._record_incident(state, "confirm", error, inputs)
                return 2
            except BaseException as error:  # a protocol failure or an interruption: recorded, then raised; confirm never resumes
                self._record_incident(state, "confirm", error, inputs)
                raise
        finally:
            del model
            gc.collect()
        state["confirmation"] = {**state["confirmation"], **results, "descriptives": rc.json_safe(descriptives), "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}
        state["phases"]["confirm"] = {**state["phases"]["confirm"], "status": "complete", "completed_at": pm.utc_now()}
        digest = self._write(state)
        self.log("confirm complete (eight conditions, no aggregate label): " + "; ".join(f"{key} {entry['result']}" for key, entry in results["conditions"].items())
                 + f"; results sha256 {digest}")
        return 0

    def _record_for_report(self) -> dict[str, Any] | None:
        installed = self.root / ul.CALIBRATION_RELATIVE_PATH
        source = installed if installed.exists() else self.candidate_calibration_path
        return json.loads(source.read_text(encoding="utf-8")) if source.exists() else None

    def report(self) -> int:
        self._all()
        state = rd.load_results_state(self.results_path)
        ul.assert_phase_allowed("report", state)
        record = self._record_for_report()
        arrays = ul.draw_values_verified(torch.load(self.draws_path), record) if record is not None and self.draws_path.exists() else None
        text = ul.render_report(state, record, arrays)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(text, encoding="utf-8")
        if state["phases"]["confirm"]["status"] == "complete":
            state["report"] = {"path": str(self.report_path), "sha256": pm.sha256_text(text), "written_at": pm.utc_now()}
            if state["phases"]["report"]["status"] != "complete":
                state["phases"]["report"] = {"status": "complete", "completed_at": pm.utc_now()}
            self._write(state)
        self.log(f"report written to {self.report_path}")
        return 0

    def replicate_021(self) -> int:
        inputs, *_ = self._all()
        state = rd.load_results_state(self.results_path)
        ul.assert_phase_allowed("replicate-021", state)
        self._check_runtime(inputs.closure)
        progs = self._weights_only(inputs)
        with pytest_free_guard():
            record = ul.replicate_021(self.root, progs, inputs, log=self.log)
        text = pm.canonical_json(record) + "\n"
        self.replication_path.write_text(text, encoding="utf-8")
        state["replication"] = {"path": str(self.replication_path), "sha256": pm.sha256_text(text), "content_sha256": record["content_sha256"], "kind": record["kind"],
                                "checks": record["checks"], "populations": record["populations"]}
        state["phases"]["replicate-021"] = {"status": "complete", "completed_at": pm.utc_now()}
        self._write(state)
        self.replication_path.with_suffix(".md").write_text(ul.render_report(state, None, None), encoding="utf-8")
        self.log(f"exploratory replication written to {self.replication_path} (no result names; it affects no 022 result)")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    if args.phase not in PHASES:
        raise SystemExit(f"unknown phase {args.phase}")
    return getattr(runner, args.phase.replace("-", "_"))()


if __name__ == "__main__":
    sys.exit(main())
