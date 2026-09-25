#!/usr/bin/env python3
"""Run Experiment 024 through isolated phases (design revision 2, ``9d03dee``; plan revision 1, ``608088c``).

``validate`` checks the pinned modules, the frozen inputs, Experiment 022's and 023's committed files (023's reviewed
exposed cells included) and — once they exist — the confirmation file, the calibration record, the lock and its
preregistration, and the results state, without a model. ``freeze`` (tokenizer only) writes the 40 fresh cues to ``confirmation-v1.json``, committed by hand; a
shortfall or a deviation from the design's expected picks writes nothing. ``calibrate`` (once; weights only; no forward
pass) reads 023's committed exposed cells and the embedding matrix and writes the candidate record (F_ρ, null₉₇.₅, the
line). ``lock`` (weights only) binds the 40 scores, the thresholds, the E–N guard and the outcome. ``confirm`` (once,
never resumed) validates the lock, reproduces its weight-derived quantities before any prompt (I7), runs the 4,320
S2-TARGET prompts once each, saves every measurement, checks the accounting, recomputes ``C`` from the saved ``Δx3``,
enforces the identities and scores the result — written in one atomic write with the completed phase before anything
descriptive runs. ``report`` renders the report.

The production configuration is fixed: the parser takes no option, and ``main`` runs ``rr.PRODUCTION`` only.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from neural_decompiler import block0_completion as b0c
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-024"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
PHASES = rr.PHASES
PhaseError = rr.PhaseError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the pinned modules, the frozen inputs, 022's and 023's committed files and the installed artifacts without a model"),
                             ("freeze", "tokenizer only: freeze the 40 fresh cues into confirmation-v1.json (commit it by hand)"),
                             ("calibrate", "once, weights only: F_ρ, null₉₇.₅ and the line from 023's committed exposed cells"),
                             ("lock", "weights only: the 40 scores, the thresholds, the E–N guard, the outcome and the preregistration"),
                             ("confirm", "once, never resumed: I7, the 4,320 prompts, the accounting, C, the identities, the result"),
                             ("report", "render the report from the results state")):
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
        # --no-renames: a moved file lists both its old and its new path, so a scientific file cannot leave by a rename.
        diff = subprocess.run(["git", "diff", "--no-renames", "--name-only", commit, "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in diff.stdout.splitlines() if line.strip()]


def _load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


def runtime_record(spec: ModelSpec) -> dict[str, Any]:
    runtime = validate_runtime(spec)
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": rd.RUNTIME_SEED, "control_seed": rd.CONTROL_SEED,
            "deterministic_algorithms": spec.deterministic_algorithms, "torch_num_threads": int(torch.get_num_threads())}


class pytest_free_guard:
    """Disable every capture and intervention entry point for the duration of a block: the executed proof that the
    freeze, the calibration, the lock and I7 cannot reach a forward pass."""

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


def _manifest_keys_022(payload: Mapping[str, Any]) -> frozenset[str]:
    manifest = payload["manifest"]
    return frozenset(manifest["S1-REF"]) | frozenset(manifest["S1-VALIDITY"]) | frozenset(manifest["S2-TARGET"]["Y1"]) | frozenset(manifest["S2-TARGET"]["Y2"])


@dataclass
class Base:
    inputs: ul.FrozenInputs
    digests: dict[str, str]
    confirmation_023: dict[str, Any]
    exclusion_base: dict[str, Any]
    excluded: dict[str, Any]
    forbidden: frozenset[str]


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
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))
    config: rr.Configuration = rr.PRODUCTION  # only a test world passes another one, explicitly

    def __post_init__(self) -> None:
        if self.config is not rr.PRODUCTION and Path(self.root).resolve() == ROOT.resolve():
            raise PhaseError("a non-production configuration may never run against the real repository")

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
    def arrays_path(self) -> Path:
        return self.output("calibration-arrays.pt")

    @property
    def stage2_path(self) -> Path:
        return self.output("stage2-measurements.pt")

    # -- inputs ---------------------------------------------------------------

    def _inputs(self) -> ul.FrozenInputs:
        rr.assert_frozen_blobs()  # the pins first: nothing is loaded through a changed module
        return ul.load_frozen_inputs(self.root, lock_011_loader=self.lock_011_loader, lock_012_loader=self.lock_012_loader, lock_017_loader=self.lock_017_loader,
                                     tracked=self.tracked)

    def _base(self) -> Base:
        """The frozen inputs; 022's and 023's committed files (verified); the exclusion; the forbidden keys: 020's ledger
        (which holds 022's exposed calibration keys), 020's confirmation set (021's spent set), 022's manifest and 023's
        manifest."""
        inputs = self._inputs()
        digests_022 = b0c.verify_022_inputs(self.root)
        digests_023 = rr.verify_023_inputs(self.root)
        path_022 = self.root / ul.CONFIRMATION_RELATIVE_PATH
        confirmation_022 = json.loads(path_022.read_text(encoding="utf-8"))
        sha_022 = rc.file_sha256(path_022)
        path_023 = self.root / b0c.CONFIRMATION_RELATIVE_PATH
        confirmation_023 = b0c.load_confirmation_023(path_023, inputs, confirmation_022, sha_022)
        payload_023 = json.loads(path_023.read_text(encoding="utf-8"))
        base = b0c.exclusion(inputs, confirmation_022, sha_022)
        excluded = rr.exclusion(base, payload_023, rc.file_sha256(path_023))
        forbidden = (frozenset(inputs.closure["ledger"]) | frozenset(prompt.key for prompt in inputs.confirmation_020.all_prompts)
                     | _manifest_keys_022(confirmation_022) | confirmation_023.manifest_keys())
        return Base(inputs, rr.base_digests(inputs, digests_022, digests_023), payload_023, base, excluded, forbidden)

    def _confirmation(self, base: Base) -> tuple[rr.Confirmation024, str]:
        path = self.root / rr.CONFIRMATION_RELATIVE_PATH
        if not path.exists() or not self.tracked(path):
            raise PhaseError(f"{rr.CONFIRMATION_RELATIVE_PATH} must be frozen and committed first (the freeze phase)")
        confirmation = rr.load_confirmation_024(path, base.inputs.pool, base.excluded, self.config)
        overlap = confirmation.manifest_keys() & base.forbidden
        if overlap:
            raise PhaseError(f"{len(overlap)} of 024's manifest keys were executed or reserved before, e.g. {sorted(overlap)[:2]}")
        return confirmation, rc.file_sha256(path)

    def _noun_keys(self, inputs: ul.FrozenInputs) -> list[str]:
        """The scorable exposed nouns in ``NounSet`` order (scorability is single-tokenness; no weight is needed)."""
        return [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]

    def _record_022(self) -> dict[str, Any]:
        return json.loads((self.root / ul.CALIBRATION_RELATIVE_PATH).read_text(encoding="utf-8"))

    # -- non-scientific phases ------------------------------------------------

    def validate(self) -> int:
        try:
            base = self._base()
            cells, units = rr.read_exposed_cells(self.root, base.inputs, self._noun_keys(base.inputs), self._record_022())
            frozen = "not frozen yet"
            if (self.root / rr.CONFIRMATION_RELATIVE_PATH).exists():
                confirmation, _ = self._confirmation(base)
                frozen = f"confirmation {confirmation.content_sha256[:12]}… ({len(confirmation.tokens)} cues, {len(confirmation.manifest_keys())} keys)"
            record = "no calibration record installed"
            record_path = self.root / rr.CALIBRATION_RELATIVE_PATH
            if record_path.exists():
                rr.verify_calibration_record(json.loads(record_path.read_text(encoding="utf-8")), self.config)
                record = "calibration record verified"
            lock_path = self.root / rr.LOCK_RELATIVE_PATH
            if lock_path.exists():
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                if lock.get("experiment") != rr.EXPERIMENT or lock.get("content_sha256") != rc.content_digest(lock) or lock.get("configuration") != self.config.to_json():
                    raise PhaseError("the installed lock does not verify against its digest and configuration")
                if not record_path.exists() or lock["calibration"]["file_sha256"] != rc.file_sha256(record_path):
                    raise PhaseError("the installed lock binds a calibration record other than the installed one")
                preregistration = self.root / rr.PREREGISTRATION_RELATIVE_PATH
                if not preregistration.exists() or preregistration.read_text(encoding="utf-8") != rr.render_preregistration(lock):
                    raise PhaseError("the installed preregistration is not the one the installed lock renders")
                record += "; lock and preregistration verified"
            if self.results_path.exists():
                state = rd.load_results_state(self.results_path)
                if state["inputs"] != {key: base.digests[key] for key in rr.DIGEST_KEYS} or state.get("configuration") != self.config.to_json():
                    raise PhaseError("the results state was written against different frozen inputs or another configuration")
                rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden, "Experiment 024's ledger")
        except (PhaseError, rd.PhaseError, pm.IncidentError, ValueError, KeyError) as error:
            self.log(f"validation failed: {error}")
            return 1
        self.log(f"module blobs {len(rr.FROZEN_BLOBS)} verified (022's and 023's programs included); 022's and 023's committed files verified; the calibration "
                 f"source {int(cells.shape[0])} exposed pairs ({len(units.cues)} cues × {len(units.frames)} frames); {frozen}; {record}")
        return 0

    def freeze(self) -> int:
        base = self._base()
        path = self.root / rr.CONFIRMATION_RELATIVE_PATH
        if path.exists():
            raise PhaseError(f"{rr.CONFIRMATION_RELATIVE_PATH} already exists; the freeze runs once")
        tokenizer = self.tokenizer_loader(PYTHIA_70M)
        try:
            with pytest_free_guard():
                payload = rr.freeze_payload(tokenizer, pool=base.inputs.pool, exclusion_base=base.exclusion_base, confirmation_023=base.confirmation_023,
                                            confirmation_023_file_sha256=rc.file_sha256(self.root / b0c.CONFIRMATION_RELATIVE_PATH), config=self.config)
        except (rr.FreezeShortfall, rr.FreezeDeviation) as error:
            self.log(f"freeze stopped for review: {error}; nothing was written and no state was created")
            return 3
        confirmation = rr.confirmation_from_payload(payload, base.inputs.pool, self.config)
        overlap = confirmation.manifest_keys() & base.forbidden
        if overlap:
            raise PhaseError(f"{len(overlap)} of the frozen manifest keys were executed or reserved before; nothing was written")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
        confirmation = rr.load_confirmation_024(path, base.inputs.pool, base.excluded, self.config)  # re-read and verified from disk
        self.log(f"froze {len(confirmation.tokens)} cues to {path} (content sha256 {confirmation.content_sha256}, file sha256 {rc.file_sha256(path)}); commit it by hand "
                 "and stop for the freeze review before calibrate")
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
            if state.get("experiment") != rr.EXPERIMENT or state["inputs"] != {key: digests[key] for key in rr.DIGEST_KEYS} \
                    or dict(state["versions"]) != provenance["versions"] or state.get("configuration") != self.config.to_json():
                raise PhaseError("frozen inputs, dependency versions or the configuration differ from the recorded run")
        else:
            if phase != "calibrate":
                raise PhaseError(f"{phase} requires an existing results state from calibrate")
            state = rr.new_results_state(digests=digests, config=self.config, **provenance)
        rr.assert_phase_allowed(phase, state)
        return state

    def _write(self, state: Mapping[str, Any]) -> str:
        return rr.write_state_atomic(self.results_path, state)

    def _recheck(self) -> dict[str, Any]:
        """Experiment 020's closure and Experiment 022's and 023's committed files, verified again."""
        try:
            inputs = self._inputs()
            rc.verify_020_closure(self.root, inputs.confirmation_020)
            b0c.verify_022_inputs(self.root)
            rr.verify_023_inputs(self.root)
        except Exception as error:  # recorded; never allowed to hide the incident being recorded
            return {"ok": False, "message": str(error)}
        return {"ok": True}

    def _record_incident(self, state: dict[str, Any], phase: str, error: BaseException) -> None:
        """An incident of a running calibrate or confirm: on disk before anything slow runs, with its commit."""
        entry_phase = state.get("phases", {}).get(phase, {})
        commit = str(entry_phase.get("commit") or entry_phase.get("confirm_commit") or state.get("protocol_code_commit") or "")
        entry = {"message": str(error) or type(error).__name__, "type": type(error).__name__, "at": pm.utc_now(), "phase": phase, "commit": commit}
        if phase == "calibrate":
            state["calibration"] = rc.json_safe(state["calibration"])
            state["calibration"].setdefault("incidents", []).append(entry)
        else:
            state["confirmation"] = rc.json_safe(state.get("confirmation") or {})
            state["confirmation"]["incident"] = entry
        self._write(state)
        self.log(f"INCIDENT ({phase}): {error}")
        entry["recheck_020_022_023"] = self._recheck()
        self._write(state)

    def _record_phase_incident(self, state: dict[str, Any], phase: str, error: BaseException) -> None:
        """An identity failure of lock or of confirm's I7: recorded with its commit; the phase is then refused until the
        reviewer decides."""
        entry = {"message": str(error) or type(error).__name__, "type": type(error).__name__, "at": pm.utc_now(), "phase": phase,
                 "commit": self._provenance()["protocol_code_commit"]}
        state["phases"][phase] = {**state["phases"][phase], "incidents": list(state["phases"][phase].get("incidents", [])) + [entry]}
        self._write(state)
        self.log(f"INCIDENT ({phase}): {error}")
        entry["recheck_020_022_023"] = self._recheck()
        self._write(state)

    def _check_runtime(self, closure: Mapping[str, Any]) -> dict[str, Any]:
        """The runtime and the dependency versions of Experiment 020's explore (021's, 022's and 023's check, unchanged)."""
        runtime = runtime_record(PYTHIA_70M)
        recorded = closure["state"]["phases"]["explore"].get("runtime") or {}
        if {k: v for k, v in runtime.items() if k != "seed"} != {k: v for k, v in recorded.items() if k != "seed"}:
            raise PhaseError(f"the runtime {runtime} differs from Experiment 020's explore runtime {recorded}")
        if dict(self.versions()) != dict(closure["state"]["versions"]):
            raise PhaseError(f"the dependency versions {dict(self.versions())} differ from Experiment 020's {closure['state']['versions']}")
        return runtime

    def _weights(self, inputs: ul.FrozenInputs) -> tuple[ul.ModelPrograms, str]:
        """The only access of calibrate and lock to the model: its parameters (and their digest). No prompt runs."""
        model = self.model_loader(PYTHIA_70M)
        try:
            return ul.ModelPrograms.from_model(model, inputs), rr.parameters_digest(model)
        finally:
            del model
            gc.collect()

    def _check_nouns(self, progs: ul.ModelPrograms, inputs: ul.FrozenInputs) -> list[str]:
        noun_keys = [progs.nouns.nouns[index].lexical_key for index in progs.scorable]
        if noun_keys != self._noun_keys(inputs) or len(noun_keys) != self.config.n_nouns:
            raise PhaseError("the scorable exposed nouns are not the pool's single-token nouns in order")
        return noun_keys

    def calibrate(self) -> int:
        base = self._base()
        confirmation, confirmation_sha = self._confirmation(base)
        state = self._state_for("calibrate", base.digests)
        rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.manifest_keys(), "Experiment 024's ledger before confirm")
        commit = self._provenance()["protocol_code_commit"]
        incidents = (state.get("calibration") or {}).get("incidents", [])
        if any(entry["commit"] == commit for entry in incidents):
            raise PhaseError("a calibrate incident is recorded at this commit; a committed fix is required before calibrate runs again")
        if incidents:
            changed = self.changed_paths(incidents[-1]["commit"])
            if changed is None or rr.scientific_changes(changed):
                raise PhaseError("a rerun after a calibrate incident must change no scientific path since the incident's commit (a code fix is a new protocol version)")
        cells, units = rr.read_exposed_cells(self.root, base.inputs, self._noun_keys(base.inputs), self._record_022())
        runtime = self._check_runtime(base.inputs.closure)
        state["protocol_code_commit"] = commit
        state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "running", "started_at": pm.utc_now(), "commit": commit, "runtime": runtime}
        self._write(state)
        calibration = state["calibration"]
        try:
            progs, parameters_sha = self._weights(base.inputs)
            self._check_nouns(progs, base.inputs)
            with pytest_free_guard():
                W_E = progs.weights.W_E
                bindings = rr.score_bindings(W_E, base.inputs.pool, units)
                dependencies = rr.scientific_dependencies(base.inputs, parameters_sha256=parameters_sha, embedding_sha256=bindings["embedding_sha256"])
                population = rr.calibration_population(cells, units, W_E, bindings, self.config)
                draws = rr.primary_draws(population["calibration"], self.config)
                null = rr.null_distribution(self.config)
                checks = {"mse": population["mse_check"], "spearman": rr.spearman_cross_check(population["calibration"], draws, self.config)}
            arrays = rr.calibration_arrays(draws, null)
            array_digests = rr.arrays_digests(arrays)
            rr.save_durably(arrays, self.arrays_path)
            calibration.update({"arrays_sha256": array_digests, "checks": rc.json_safe(checks), "dependencies": dependencies})
            self._write(state)
            try:
                evaluated = rr.evaluate_calibration(population["calibration"], population["pronouns"], draws, null, self.config)
            except rr.CalibrationStop as stop:
                calibration["stop"] = rc.json_safe(stop.details)
                state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "stopped_for_review", "stopped_at": pm.utc_now()}
                self._write(state)
                self.log(f"calibration stopped for review: {stop}; no record was written")
                return 3
            binding = rr.confirmation_binding(confirmation, confirmation_sha)
            record = rr.calibration_record(run_id=state["run_id"], protocol_code_commit=commit, digests=base.digests, config=self.config, confirmation=binding,
                                           bindings=bindings, dependencies=dependencies, population=population, evaluated=evaluated, checks=checks,
                                           array_digests=array_digests)
            recheck = self._recheck()
            if not recheck["ok"]:
                raise pm.IncidentError(f"Experiment 020's closure or Experiment 022's or 023's committed files no longer verify: {recheck['message']}")
            text = pm.canonical_json(record) + "\n"
            self.candidate_calibration_path.write_text(text, encoding="utf-8")
            calibration.update({"record_sha256": pm.sha256_text(text), "record_content_sha256": record["content_sha256"], "candidate_path": str(self.candidate_calibration_path),
                                "F_rho": record["primary_floor"]["F_rho"], "null_975": record["null"]["null_975"], "line": record["line"]})
            state["confirmation_024"] = binding
            self._write(state)
        except rr.CrossCheckError as error:
            calibration["cross_check"] = rc.json_safe(error.details)
            self._record_incident(state, "calibrate", error)
            return 2
        except pm.IncidentError as error:
            self._record_incident(state, "calibrate", error)
            return 2
        except BaseException as error:
            self._record_incident(state, "calibrate", error)
            raise
        state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "complete", "completed_at": pm.utc_now()}
        digest = self._write(state)
        self.log(f"calibrate complete: candidate record {self.candidate_calibration_path} (sha256 {calibration['record_sha256']}; F_ρ {calibration['F_rho']}, null₉₇.₅ "
                 f"{calibration['null_975']}); results sha256 {digest}. Install it byte-identical as {rr.CALIBRATION_RELATIVE_PATH}, commit it, and stop for the "
                 "floor review before lock.")
        return 0

    def _installed_record(self, state: Mapping[str, Any], digests: Mapping[str, str]) -> tuple[dict[str, Any], str]:
        path = self.root / rr.CALIBRATION_RELATIVE_PATH
        if not path.exists() or not self.tracked(path):
            raise PhaseError(f"install the candidate calibration record byte-identical as {rr.CALIBRATION_RELATIVE_PATH} and commit it before lock")
        sha = rc.file_sha256(path)
        if sha != (state.get("calibration") or {}).get("record_sha256"):
            raise PhaseError("the installed calibration record is not the candidate this run wrote")
        record = json.loads(path.read_text(encoding="utf-8"))
        rr.verify_calibration_record(record, self.config)
        if record["inputs"] != dict(digests):
            raise PhaseError("the calibration record was computed from different inputs")
        return record, sha

    def lock(self) -> int:
        base = self._base()
        confirmation, confirmation_sha = self._confirmation(base)
        state = self._state_for("lock", base.digests)
        rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.manifest_keys(), "Experiment 024's ledger before confirm")
        changed = self.changed_paths(state["phases"]["calibrate"]["commit"])
        if changed is None:
            raise PhaseError("the calibrate commit is not an ancestor of the current commit; lock must be written at the calibrated protocol")
        scientific = rr.scientific_changes(changed)
        if scientific:
            raise PhaseError(f"scientific paths changed since calibrate: {scientific}")
        record, record_sha = self._installed_record(state, base.digests)
        rr.verify_confirmation_binding(record, state, rr.confirmation_binding(confirmation, confirmation_sha))
        self._check_runtime(base.inputs.closure)
        progs, parameters_sha = self._weights(base.inputs)
        noun_keys = self._check_nouns(progs, base.inputs)
        with pytest_free_guard():
            W_E = progs.weights.W_E
            bindings = rr.score_bindings(W_E, base.inputs.pool, b0c.exposed_units(base.inputs))
            if bindings != record["score"]:
                raise PhaseError("the score's bindings (embedding, id lists, centroids) are not the calibration record's")
            dependencies = rr.scientific_dependencies(base.inputs, parameters_sha256=parameters_sha, embedding_sha256=bindings["embedding_sha256"])
            rr.verify_dependencies(record["dependencies"], dependencies, "the calibration record")
        try:
            with pytest_free_guard():  # the provenance invariant, executed rather than asserted
                fresh = rr.fresh_quantities(W_E, record["score"], confirmation, record)
                again = rr.fresh_quantities(W_E, record["score"], confirmation, record)
            if fresh != again:
                raise pm.IncidentError("provenance: the lock's weight-derived quantities do not reproduce bit for bit")
            recheck = self._recheck()
            if not recheck["ok"]:
                raise pm.IncidentError(f"Experiment 020's closure or Experiment 022's or 023's committed files no longer verify at lock: {recheck['message']}; "
                                       "no lock was written")
        except pm.IncidentError as error:
            self._record_phase_incident(state, "lock", error)
            return 2
        commit = self._provenance()["protocol_code_commit"]
        lock = rr.build_lock(run_id=state["run_id"], protocol_code_commit=commit, digests=base.digests, config=self.config, record=record, record_file_sha256=record_sha,
                             confirmation=confirmation, confirmation_file_sha256=confirmation_sha, fresh=fresh, dependencies=dependencies, noun_keys=noun_keys)
        candidate_path, preregistration_path = self.output("candidate-lock.json"), self.output("candidate-preregistration.md")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        preregistration = rr.render_preregistration(lock)
        preregistration_path.write_text(preregistration, encoding="utf-8")
        state["lock"] = {"candidate_path": str(candidate_path), "preregistration_path": str(preregistration_path), "content_sha256": lock["content_sha256"],
                         "preregistration_sha256": pm.sha256_text(preregistration), "calibration_content_sha256": record["content_sha256"],
                         "extrapolation": lock["fresh"]["extrapolation"], "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now(), "commit": commit}
        self._write(state)
        self.log(f"candidate lock {candidate_path} (content sha256 {lock['content_sha256']}) and {preregistration_path}. Install both byte-identical as "
                 f"{rr.LOCK_RELATIVE_PATH} and {rr.PREREGISTRATION_RELATIVE_PATH}, commit, and stop for the independent lock review before confirm.")
        return 0

    def confirm(self) -> int:
        base = self._base()
        confirmation, confirmation_sha = self._confirmation(base)
        state = self._state_for("confirm", base.digests)
        rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.manifest_keys(), "Experiment 024's ledger before confirm")
        paths = {"lock": self.root / rr.LOCK_RELATIVE_PATH, "preregistration": self.root / rr.PREREGISTRATION_RELATIVE_PATH}
        if not all(path.exists() for path in paths.values()):
            raise PhaseError("the committed lock and preregistration must both exist")
        lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
        record, record_sha = self._installed_record(state, base.digests)
        git = self.git_state()
        placeholder = rr.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
        rr.validate_lock(lock, state=state, digests=base.digests, config=self.config, record=record, record_file_sha256=record_sha, confirmation=confirmation,
                         confirmation_file_sha256=confirmation_sha, dependencies=placeholder, noun_keys=self._noun_keys(base.inputs),
                         preregistration_text=paths["preregistration"].read_text(encoding="utf-8"), git_state=git,
                         tracked=all(self.tracked(path) for path in paths.values()), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        commit = str(git.get("commit"))
        self._check_runtime(base.inputs.closure)
        seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256=rr.parameters_digest(model),
                                         embedding_sha256=lock["dependencies"]["model"]["embedding_sha256"], what="the lock")
            progs = ul.ModelPrograms.from_model(model, base.inputs)
            self._check_nouns(progs, base.inputs)
            with pytest_free_guard():  # I7: the lock's weight-derived quantities, before any fresh prompt, bit for bit
                rr.verify_model_dependencies(lock["dependencies"]["model"], parameters_sha256=lock["dependencies"]["model"]["parameters_sha256"],
                                             embedding_sha256=rr.embedding_digest(progs.weights.W_E), what="the lock")
                reproduced = rr.reproduce_lock_quantities(progs.weights.W_E, record, confirmation, lock)
            if not reproduced["bitwise_equal"]:
                self._record_phase_incident(state, "confirm", pm.IncidentError(f"I7 failed: the lock's {reproduced['differing']} do not reproduce bit for bit; no fresh "
                                                                               "prompt has run"))
                return 2
            units = rr.target_units(confirmation)
            states = ul.y1_states(base.inputs.closure["exploration"]["locked_states"], units.frames)
            state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": commit, "I7": reproduced}
            pm.record_execution(state, confirmation.target_prompts, base.inputs.pool.nouns)
            state["confirmation"] = {}
            self._write(state)  # the 4,320 keys are in the ledger on disk before any of them runs
            try:
                executed: list[pm.Prompt] = []
                measured = ul.stage_two_022(model, progs, confirmation, {"Y1": states, "Y2": {}}, executed=executed, log=self.log)
                tensors = ul.measurement_tensors(measured)
                rr.save_durably(tensors, self.stage2_path)
                digests = {key: rc.tensor_digest(value) for key, value in tensors.items()}
                state["confirmation"]["stage2"] = {"path": str(self.stage2_path), "tensors_sha256": digests, "n_executed": len(executed)}
                self._write(state)  # every fresh measurement is on disk before any gate or score can stop the phase
                expected = Counter(prompt.key for prompt in confirmation.target_prompts)
                accounting = {"manifest": len(expected), "executed": len(executed), "ledger": len(state["executed_prompt_keys"]),
                              "equal": Counter(prompt.key for prompt in executed) == expected and set(state["executed_prompt_keys"]) == set(expected)}
                state["confirmation"]["accounting"] = accounting
                self._write(state)
                if not accounting["equal"]:
                    raise pm.IncidentError(f"the prompt accounting failed: manifest {accounting['manifest']}, executed {accounting['executed']}, ledger {accounting['ledger']}")
                saved = torch.load(self.stage2_path)
                if {key: rc.tensor_digest(value) for key, value in saved.items()} != digests:
                    raise pm.IncidentError("the saved measurements do not reproduce their digests")
                rr.assert_measurements(saved, units)
                c_check = rr.recompute_c(progs, units, states, saved)
                state["confirmation"]["c_recompute"] = c_check
                self._write(state)
                if not c_check["bitwise_equal"]:
                    raise pm.IncidentError(f"C recomputed from the saved Δx3 differs from the saved C: {c_check['max']} at {c_check['at']}")
                gates = rr.target_gates(progs, confirmation, units, states, saved)
                state["confirmation"]["gates"] = gates
                self._write(state)  # the gate values (never the result) before enforcement
                rr.enforce_gates(gates)
                cells = rr.fresh_cue_cells(units, saved, confirmation)
                results = rr.score(cells, confirmation, lock, self.config)
                recheck = self._recheck()  # before any result is written: an incident carries no result
                if not recheck["ok"]:
                    raise pm.IncidentError(f"Experiment 020's closure or Experiment 022's or 023's committed files no longer verify after confirm: {recheck['message']}; "
                                           "the scored results are void")
            except rr.CrossCheckError as error:
                state["confirmation"] = rc.json_safe(state.get("confirmation") or {})
                state["confirmation"]["cross_check"] = rc.json_safe(error.details)
                self._record_incident(state, "confirm", error)
                return 2
            except pm.IncidentError as error:
                self._record_incident(state, "confirm", error)
                return 2
            except BaseException as error:  # a protocol failure or an interruption: recorded, then raised; confirm never resumes
                self._record_incident(state, "confirm", error)
                raise
            running = dict(state["phases"]["confirm"])
            state["confirmation"].update({"results": results, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()})
            state["phases"]["confirm"] = {**running, "status": "complete", "completed_at": pm.utc_now()}
            try:
                digest = self._write(state)  # one atomic write: the primary result, the guard, the outcome and the completed phase
            except BaseException as error:  # the atomic write left the previous state (no result) on disk: record an incident, never a result
                for key in ("results", "lock_sha256", "completed_at"):
                    state["confirmation"].pop(key, None)
                state["phases"]["confirm"] = running
                self._record_incident(state, "confirm", error)
                if isinstance(error, Exception):
                    return 2
                raise
            self.log(f"confirm complete: {results['outcome']['label']} (primary {results['primary']['result']}, ρ {results['primary']['rho']}; E–N guard "
                     f"{results['guard']['result']}, K {results['guard'].get('K')}); results sha256 {digest}")
            self._descriptives(state, progs, confirmation, units, states, saved, cells, lock)
        finally:
            del model
            gc.collect()
        return 0

    def _descriptives(self, state: dict[str, Any], progs: ul.ModelPrograms, confirmation: rr.Confirmation024, units: ul.TableUnits, states: Mapping[str, Any],
                      tensors: Mapping[str, torch.Tensor], cells: torch.Tensor, lock: Mapping[str, Any]) -> None:
        """The descriptive records (no outcome force), computed only after the result and the completed phase are on disk:
        a failure is recorded as such and the phase stays complete; an interruption is recorded and raised. Neither can
        touch the result."""
        descriptives = state["confirmation"].setdefault("descriptives", {})
        per_cue = state["confirmation"]["results"]["per_cue"]
        for name, compute in (("secondary", lambda: rr.secondary(cells, units, tensors, confirmation, lock)),
                              ("contrasts", lambda: rr.contrasts(per_cue, self.config)),
                              ("ladder", lambda: rr.ladder(progs, units, states, tensors, confirmation))):
            try:
                descriptives[name] = rc.json_safe(compute())
            except BaseException as error:
                descriptives.setdefault("failures", {})[name] = {"type": type(error).__name__, "message": str(error), "at": pm.utc_now()}
                self._write(state)
                self.log(f"the descriptive record {name} failed (no outcome force; the result stands): {error}")
                if not isinstance(error, Exception):
                    raise
                continue
            self._write(state)

    def _record_for_report(self) -> dict[str, Any] | None:
        installed = self.root / rr.CALIBRATION_RELATIVE_PATH
        source = installed if installed.exists() else self.candidate_calibration_path
        return json.loads(source.read_text(encoding="utf-8")) if source.exists() else None

    def report(self) -> int:
        self._base()
        state = rd.load_results_state(self.results_path)
        rr.assert_phase_allowed("report", state)
        text = rr.render_report(state, self._record_for_report())
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(text, encoding="utf-8")
        if state["phases"]["confirm"]["status"] == "complete":
            state["report"] = {"path": str(self.report_path), "sha256": pm.sha256_text(text), "written_at": pm.utc_now()}
            if state["phases"]["report"]["status"] != "complete":
                state["phases"]["report"] = {"status": "complete", "completed_at": pm.utc_now()}
            self._write(state)
        self.log(f"report written to {self.report_path}")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()  # always rr.PRODUCTION
    if args.phase not in PHASES:
        raise SystemExit(f"unknown phase {args.phase}")
    return getattr(runner, args.phase)()


if __name__ == "__main__":
    sys.exit(main())
