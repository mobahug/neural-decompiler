#!/usr/bin/env python3
"""Run Experiment 023 through isolated phases (design revision 2, ``5b38aba``; plan revision 1, ``3a795fb``).

``validate`` checks the pinned modules (Experiment 022's program included), the frozen inputs, Experiment 022's
committed calibration record and confirmation file, and — once they exist — the installed exposed-cells artifact, the
confirmation file and the results state, without a model. ``extract`` (once; weights only; no prompt) is the only phase
that opens Experiment 022's local calibration table: it verifies it (E1), extracts the per-pair sufficient statistics
of the 18,900 exposed pairs (E2, E3), recomputes ``P1`` from the weights (E4), checks the cells against the per-noun
table (E5) and the pooled ``SST`` (E6), and writes the candidate artifact, installed byte-identically and committed.
``freeze`` (tokenizer only) writes the new cues and frames to ``confirmation-v1.json``, committed by hand.
``calibrate`` (once; no model) reads only the committed artifact and the confirmation's counts. ``lock`` (no forward
pass) writes the Y1 prediction table, the candidate lock and the preregistration. ``confirm`` (once, never resumed)
validates the lock, rebuilds the Y1 table bit for bit (I7) before any fresh prompt, runs stage 1 and writes the Y2
prediction table once, re-reads and verifies both at the barrier, runs stage 2, saves every measurement, enforces the
identities and scores the four conditions through the one canonical path, writing them before any descriptive record is
computed. ``report`` renders the report.
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
from neural_decompiler import upstream_localization as ul
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-023"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
PHASES = b0c.PHASES
PhaseError = b0c.PhaseError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the pinned modules, the frozen inputs, 022's committed files and the installed artifacts without a model"),
                             ("extract", "once, weights only: the exposed cells from 022's verified table (E1–E6); the only phase that opens that table"),
                             ("freeze", "tokenizer only: freeze the new cues and frames into confirmation-v1.json (commit it by hand)"),
                             ("calibrate", "once, no model: the draws, the envelopes and the candidate record from the committed exposed cells"),
                             ("lock", "no forward pass: the Y1 prediction table, the candidate lock and the preregistration"),
                             ("confirm", "once, never resumed: I7, stage 1 and the Y2 table, the barrier, stage 2, the identities and the four conditions"),
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
        diff = subprocess.run(["git", "diff", "--name-only", commit, "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
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
    extraction, the freeze, the lock's predictions and the I7 reconstruction cannot reach a forward pass."""

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

    # -- paths ----------------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        return self.results_path.parent

    def output(self, name: str) -> Path:
        return self.output_dir / name

    @property
    def candidate_cells_paths(self) -> tuple[Path, Path]:
        return self.output("candidate-exposed-cells.f64"), self.output("candidate-exposed-cells.json")

    @property
    def candidate_calibration_path(self) -> Path:
        return self.output("candidate-calibration.json")

    @property
    def draws_path(self) -> Path:
        return self.output("draw-values.pt")

    @property
    def stage2_path(self) -> Path:
        return self.output("stage2-measurements.pt")

    # -- inputs ---------------------------------------------------------------

    def _inputs(self) -> ul.FrozenInputs:
        inputs = ul.load_frozen_inputs(self.root, lock_011_loader=self.lock_011_loader, lock_012_loader=self.lock_012_loader, lock_017_loader=self.lock_017_loader,
                                       tracked=self.tracked)
        b0c.assert_frozen_blobs()
        return inputs

    def _base(self):
        """The frozen inputs, Experiment 022's committed files (verified), the state's input digests and the forbidden
        keys: 020's ledger (which holds 022's exposed calibration keys), 020's confirmation set (021's spent set) and
        022's manifest."""
        inputs = self._inputs()
        digests_022 = b0c.verify_022_inputs(self.root)
        path_022 = self.root / ul.CONFIRMATION_RELATIVE_PATH
        confirmation_022 = json.loads(path_022.read_text(encoding="utf-8"))
        digests = b0c.base_digests(inputs, digests_022)
        forbidden = frozenset(inputs.closure["ledger"]) | frozenset(prompt.key for prompt in inputs.confirmation_020.all_prompts) | _manifest_keys_022(confirmation_022)
        return inputs, confirmation_022, rc.file_sha256(path_022), digests, forbidden

    def _confirmation(self, inputs: ul.FrozenInputs, confirmation_022: Mapping[str, Any], confirmation_022_sha: str,
                      forbidden: frozenset[str]) -> tuple[b0c.Confirmation023, str]:
        path = self.root / b0c.CONFIRMATION_RELATIVE_PATH
        if not path.exists() or not self.tracked(path):
            raise PhaseError(f"{b0c.CONFIRMATION_RELATIVE_PATH} must be frozen and committed first (the freeze phase)")
        confirmation = b0c.load_confirmation_023(path, inputs, confirmation_022, confirmation_022_sha)
        overlap = confirmation.manifest_keys() & forbidden
        if overlap:
            raise PhaseError(f"{len(overlap)} of 023's manifest keys were executed or reserved before, e.g. {sorted(overlap)[:2]}")
        return confirmation, rc.file_sha256(path)

    def _noun_keys(self, inputs: ul.FrozenInputs) -> list[str]:
        """The scorable exposed nouns in ``NounSet`` order (scorability is single-tokenness; no weight is needed)."""
        return [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]

    def _record_022(self) -> dict[str, Any]:
        return json.loads((self.root / ul.CALIBRATION_RELATIVE_PATH).read_text(encoding="utf-8"))

    def _read_installed_cells(self, inputs: ul.FrozenInputs) -> tuple[torch.Tensor, b0c.ExposedUnits, dict[str, str], dict[str, Any]]:
        """The committed artifact re-read and verified against the meta recomputed from the frozen inputs and 022's
        committed record now. Experiment 022's calibration table is never opened here."""
        data, index = self.root / b0c.CELLS_DATA_RELATIVE_PATH, self.root / b0c.CELLS_INDEX_RELATIVE_PATH
        if not data.exists() or not index.exists() or not self.tracked(data) or not self.tracked(index):
            raise PhaseError(f"install the candidate exposed cells byte-identically as {b0c.CELLS_DATA_RELATIVE_PATH} and {b0c.CELLS_INDEX_RELATIVE_PATH} and commit them")
        units = b0c.exposed_units(inputs)
        cells, loaded = b0c.read_cells(data, index, units=units, meta=b0c.cells_meta(units, self._noun_keys(inputs), self._record_022()))
        files = {"data_path": b0c.CELLS_DATA_RELATIVE_PATH, "index_path": b0c.CELLS_INDEX_RELATIVE_PATH, "data_sha256": rc.file_sha256(data), "index_sha256": rc.file_sha256(index)}
        return cells, units, files, loaded

    # -- non-scientific phases ------------------------------------------------

    def validate(self) -> int:
        try:
            inputs, confirmation_022, confirmation_022_sha, digests, forbidden = self._base()
            units = b0c.exposed_units(inputs)
            artifact = "not extracted yet"
            if (self.root / b0c.CELLS_INDEX_RELATIVE_PATH).exists():
                cells, _, files, loaded = self._read_installed_cells(inputs)
                artifact = f"exposed cells {files['data_sha256'][:12]}… ({int(cells.shape[0])} pairs, extracted at {loaded['extraction'].get('protocol_code_commit', '')[:7]})"
            frozen = "not frozen yet"
            if (self.root / b0c.CONFIRMATION_RELATIVE_PATH).exists():
                confirmation, _ = self._confirmation(inputs, confirmation_022, confirmation_022_sha, forbidden)
                frozen = f"confirmation {confirmation.content_sha256[:12]}… ({len(confirmation.tokens)} cues, {len(confirmation.frames)} frames, {len(confirmation.manifest_keys())} keys)"
            if self.results_path.exists():
                state = rd.load_results_state(self.results_path)
                if state["inputs"] != {key: digests[key] for key in b0c.DIGEST_KEYS}:
                    raise PhaseError("the results state was written against different frozen inputs")
                b0c.assert_ledger_isolated(state["executed_prompt_keys"], forbidden, "Experiment 023's ledger")
        except (PhaseError, rd.PhaseError, pm.IncidentError, ValueError, KeyError) as error:
            self.log(f"validation failed: {error}")
            return 1
        self.log(f"module blobs {len(b0c.FROZEN_BLOBS)} verified (022's program included); Experiment 022's record and confirmation verified; exposed pool "
                 f"{len(units.cues)} cues × {len(units.frames)} frames; {artifact}; {frozen}")
        return 0

    def freeze(self) -> int:
        inputs, confirmation_022, confirmation_022_sha, digests, forbidden = self._base()
        path = self.root / b0c.CONFIRMATION_RELATIVE_PATH
        if path.exists():
            raise PhaseError(f"{b0c.CONFIRMATION_RELATIVE_PATH} already exists; the freeze runs once")
        tokenizer = self.tokenizer_loader(PYTHIA_70M)
        try:
            with pytest_free_guard():
                payload = b0c.freeze_payload(tokenizer, inputs, confirmation_022, confirmation_022_sha)
        except b0c.FreezeShortfall as error:
            self.log(f"freeze shortfall: {error}; nothing was written; stopped for review")
            return 3
        confirmation = b0c.confirmation_from_payload(payload, inputs.pool)
        overlap = confirmation.manifest_keys() & forbidden
        if overlap:
            raise PhaseError(f"{len(overlap)} of the frozen manifest keys were executed or reserved before; nothing was written")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
        confirmation = b0c.load_confirmation_023(path, inputs, confirmation_022, confirmation_022_sha)  # re-read and verified from disk
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
            if state.get("experiment") != b0c.EXPERIMENT or state["inputs"] != {key: digests[key] for key in b0c.DIGEST_KEYS} or dict(state["versions"]) != provenance["versions"]:
                raise PhaseError("frozen inputs or dependency versions differ from the recorded run")
        else:
            if phase != "extract":
                raise PhaseError(f"{phase} requires an existing results state from extract")
            state = b0c.new_results_state(digests=digests, **provenance)
        b0c.assert_phase_allowed(phase, state)
        return state

    def _write(self, state: Mapping[str, Any]) -> str:
        return rd.write_results_state(self.results_path, state)

    def _recheck(self) -> dict[str, Any]:
        """Experiment 020's closure and Experiment 022's committed files, verified again after a phase or an incident."""
        try:
            inputs = self._inputs()
            rc.verify_020_closure(self.root, inputs.confirmation_020)
            b0c.verify_022_inputs(self.root)
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
        entry["recheck_020_022"] = self._recheck()
        self._write(state)

    def _record_phase_incident(self, state: dict[str, Any], phase: str, error: BaseException) -> None:
        """An identity failure of extract, lock or confirm's I7: recorded with its commit; the phase is then refused at
        that commit (extract) or until the reviewer decides (lock, confirm)."""
        entry = {"message": str(error) or type(error).__name__, "type": type(error).__name__, "at": pm.utc_now(), "phase": phase,
                 "commit": self._provenance()["protocol_code_commit"]}
        state["phases"][phase] = {**state["phases"][phase], "incidents": list(state["phases"][phase].get("incidents", [])) + [entry]}
        self._write(state)
        self.log(f"INCIDENT ({phase}): {error}")
        entry["recheck_020_022"] = self._recheck()
        self._write(state)

    def _check_runtime(self, closure: Mapping[str, Any]) -> dict[str, Any]:
        """The runtime and the dependency versions of Experiment 020's explore (021's and 022's check, unchanged)."""
        runtime = runtime_record(PYTHIA_70M)
        recorded = closure["state"]["phases"]["explore"].get("runtime") or {}
        if {k: v for k, v in runtime.items() if k != "seed"} != {k: v for k, v in recorded.items() if k != "seed"}:
            raise PhaseError(f"the runtime {runtime} differs from Experiment 020's explore runtime {recorded}")
        if dict(self.versions()) != dict(closure["state"]["versions"]):
            raise PhaseError(f"the dependency versions {dict(self.versions())} differ from Experiment 020's {closure['state']['versions']}")
        return runtime

    def _weights_only(self, inputs: ul.FrozenInputs) -> ul.ModelPrograms:
        """The only access of extract and lock to the model: its parameters. No prompt runs."""
        model = self.model_loader(PYTHIA_70M)
        try:
            return ul.ModelPrograms.from_model(model, inputs)
        finally:
            del model
            gc.collect()

    def extract(self) -> int:
        inputs, confirmation_022, confirmation_022_sha, digests, forbidden = self._base()
        if (self.root / b0c.CELLS_DATA_RELATIVE_PATH).exists() or (self.root / b0c.CELLS_INDEX_RELATIVE_PATH).exists():
            raise PhaseError("the exposed-cells artifact is already installed; extract runs once")
        candidate_data, candidate_index = self.candidate_cells_paths
        if candidate_data.exists() or candidate_index.exists():
            raise PhaseError("a candidate exposed-cells artifact already exists; extract runs once")
        state = self._state_for("extract", digests)
        commit = self._provenance()["protocol_code_commit"]
        if any(entry["commit"] == commit for entry in state["phases"]["extract"].get("incidents", [])):
            raise PhaseError("an extract incident is recorded at this commit; a committed fix is required before extract runs again")
        record_022 = self._record_022()
        runtime = self._check_runtime(inputs.closure)
        state["protocol_code_commit"] = commit
        state["phases"]["extract"] = {**state["phases"]["extract"], "status": "running", "started_at": pm.utc_now(), "commit": commit, "runtime": runtime}
        self._write(state)
        table_path = self.root / b0c.TABLE_022_RELATIVE_PATH
        try:
            if not table_path.exists() or rc.file_sha256(table_path) != b0c.INHERITED_022["table_file_sha256"]:
                raise b0c.ExtractionMismatch(f"E1: {b0c.TABLE_022_RELATIVE_PATH} is missing or not the file Experiment 022's closure binds")
            progs = self._weights_only(inputs)
            units = b0c.exposed_units(inputs)
            noun_keys = [progs.nouns.nouns[index].lexical_key for index in progs.scorable]
            if noun_keys != self._noun_keys(inputs):
                raise PhaseError("the scorable exposed nouns are not the pool's single-token nouns in order")
            with pytest_free_guard():
                table = torch.load(table_path)
                result = b0c.extract_cells(table, record_022, units, noun_keys, recompute=lambda: b0c.recompute_p1(progs, inputs, units, table, log=self.log), log=self.log)
            del table
        except b0c.ExtractionMismatch as error:
            state["phases"]["extract"] = {**state["phases"]["extract"], "status": "stopped_for_review", "stopped_at": pm.utc_now(), "stop": str(error)}
            self._write(state)
            self.log(f"extraction stopped for review: {error}; nothing was written")
            return 3
        except pm.IncidentError as error:
            self._record_phase_incident(state, "extract", error)
            return 2
        except BaseException as error:  # a protocol failure or an interruption: recorded, then raised
            self._record_phase_incident(state, "extract", error)
            raise
        extraction = {"module": "src/neural_decompiler/block0_completion.py", "module_blob": b0c.own_blob(), "cells_version": b0c.CELLS_VERSION, "protocol_code_commit": commit,
                      "run_id": state["run_id"], "extracted_at": pm.utc_now(), "frozen_blobs": b0c.module_blobs(), "checks": result["checks"]}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        index = b0c.write_cells(candidate_data, candidate_index, result["cells"], result["meta"], extraction)
        recheck = self._recheck()
        if not recheck["ok"]:
            raise PhaseError(f"Experiment 020's closure or Experiment 022's committed files no longer verify after extract: {recheck['message']}")
        state["extract"] = {"candidate_data_path": str(candidate_data), "candidate_index_path": str(candidate_index), "data_sha256": index["file_sha256"],
                            "index_sha256": rc.file_sha256(candidate_index), "checks": result["checks"], "commit": commit, "written_at": pm.utc_now()}
        state["phases"]["extract"] = {**state["phases"]["extract"], "status": "complete", "completed_at": pm.utc_now()}
        digest = self._write(state)
        self.log(f"extract complete: candidate {candidate_data} (sha256 {index['file_sha256']}) and {candidate_index}; results sha256 {digest}. Install both "
                 f"byte-identical as {b0c.CELLS_DATA_RELATIVE_PATH} and {b0c.CELLS_INDEX_RELATIVE_PATH}, commit them, and stop for the artifact's check.")
        return 0

    def _installed_cells(self, state: Mapping[str, Any], inputs: ul.FrozenInputs) -> tuple[torch.Tensor, b0c.ExposedUnits, dict[str, str]]:
        cells, units, files, loaded = self._read_installed_cells(inputs)
        if (files["data_sha256"], files["index_sha256"]) != (state["extract"].get("data_sha256"), state["extract"].get("index_sha256")):
            raise PhaseError("the installed exposed-cells artifact is not the candidate this run extracted")
        if loaded["extraction"].get("protocol_code_commit") != state["phases"]["extract"].get("commit"):
            raise PhaseError("the artifact's extraction record names a commit other than the extract phase's")
        return cells, units, files

    def calibrate(self) -> int:
        inputs, confirmation_022, confirmation_022_sha, digests, forbidden = self._base()
        confirmation, confirmation_sha = self._confirmation(inputs, confirmation_022, confirmation_022_sha, forbidden)
        state = self._state_for("calibrate", digests)
        b0c.assert_ledger_isolated(state["executed_prompt_keys"], forbidden | confirmation.manifest_keys(), "Experiment 023's ledger before confirm")
        commit = self._provenance()["protocol_code_commit"]
        if any(entry["commit"] == commit for entry in (state.get("calibration") or {}).get("incidents", [])):
            raise PhaseError("a calibrate incident is recorded at this commit; a committed fix is required before calibrate runs again")
        changed = self.changed_paths(state["phases"]["extract"]["commit"])
        if changed is None:
            raise PhaseError("the extract commit is not an ancestor of the current commit; calibrate must run at the extracted protocol")
        scientific = b0c.scientific_changes(changed)
        if scientific:
            raise PhaseError(f"scientific paths changed since extract: {scientific}")
        cells, units, cells_files = self._installed_cells(state, inputs)
        state["protocol_code_commit"] = commit
        state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "running", "started_at": pm.utc_now(), "commit": commit}
        self._write(state)
        calibration = state["calibration"]
        try:
            indices = b0c.draw_indices(units, b0c.B)
            kernel = b0c.calibration_kernel(cells, units, indices, b0c.B, log=self.log)
            kernel_check = b0c.kernel_loop_check(cells, units, indices, kernel)
            undefined = b0c.undefined_counts(kernel)
            arrays = b0c.draw_arrays(kernel)
            array_digests = {key: rc.tensor_digest(values) for key, values in arrays.items()}
            self.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(arrays, self.draws_path)
            calibration.update({"kernel_check": kernel_check, "e6_max": kernel["e6_max"], "undefined_counts": undefined, "draw_arrays_sha256": array_digests,
                                "draw_index_sha256": b0c.draw_index_digests(indices), "exposed_cells": cells_files})
            self._write(state)
            stop = b0c.calibration_stop(undefined, b0c.B)
            if stop["stop"]:
                calibration["stop"] = stop
                state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "stopped_for_review", "stopped_at": pm.utc_now()}
                self._write(state)
                self.log(f"calibration stop: undefined draws {stop['offending']} at or above {stop['threshold']}; no envelope is written; stopped for review")
                return 3
            evaluated = b0c.evaluate(kernel)
            binding = b0c.confirmation_binding(confirmation, confirmation_sha)
            record = b0c.calibration_record(run_id=state["run_id"], protocol_code_commit=commit, digests=digests, units=units, cells_files=cells_files,
                                            confirmation=binding, index_digests=b0c.draw_index_digests(indices), kernel_check=kernel_check, e6_max=kernel["e6_max"], undefined=undefined,
                                            evaluated=evaluated, array_digests=array_digests, draws=b0c.B)
            recheck = self._recheck()
            if not recheck["ok"]:
                raise pm.IncidentError(f"Experiment 020's closure or Experiment 022's committed files no longer verify: {recheck['message']}")
            text = pm.canonical_json(record) + "\n"
            self.candidate_calibration_path.write_text(text, encoding="utf-8")
            calibration.update({"record_sha256": pm.sha256_text(text), "record_content_sha256": record["content_sha256"], "candidate_path": str(self.candidate_calibration_path),
                                "envelopes": {condition: record["conditions"][condition]["envelope"] for condition in b0c.CONDITIONS}, "joint_rates": record["joint_rates"]})
            state["confirmation_023"] = binding
            self._write(state)
        except b0c.KernelCheckError as error:
            calibration["kernel_check"] = rc.json_safe(error.details)
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
        self.log(f"calibrate complete: candidate record {self.candidate_calibration_path} (sha256 {calibration['record_sha256']}); results sha256 {digest}. "
                 f"Install it byte-identical as {b0c.CALIBRATION_RELATIVE_PATH}, commit it, and stop for the floor review before lock.")
        return 0

    def _installed_record(self, state: Mapping[str, Any], digests: Mapping[str, str]) -> tuple[dict[str, Any], str]:
        path = self.root / b0c.CALIBRATION_RELATIVE_PATH
        if not path.exists() or not self.tracked(path):
            raise PhaseError(f"install the candidate calibration record byte-identical as {b0c.CALIBRATION_RELATIVE_PATH} and commit it before lock")
        sha = rc.file_sha256(path)
        if sha != state["calibration"].get("record_sha256"):
            raise PhaseError("the installed calibration record is not the candidate this run wrote")
        record = json.loads(path.read_text(encoding="utf-8"))
        b0c.verify_calibration_record(record)
        if record["inputs"] != dict(digests):
            raise PhaseError("the calibration record was computed from different inputs")
        return record, sha

    def _cells_unchanged(self, cells_files: Mapping[str, str]) -> None:
        """The installed artifact's files are still the ones the record binds (by digest; the 022 table is not read)."""
        for key in ("data", "index"):
            path = self.root / cells_files[f"{key}_path"]
            if not path.exists() or rc.file_sha256(path) != cells_files[f"{key}_sha256"]:
                raise PhaseError(f"the installed exposed-cells {key} file is not the one the calibration record binds")

    def lock(self) -> int:
        inputs, confirmation_022, confirmation_022_sha, digests, forbidden = self._base()
        confirmation, confirmation_sha = self._confirmation(inputs, confirmation_022, confirmation_022_sha, forbidden)
        state = self._state_for("lock", digests)
        b0c.assert_ledger_isolated(state["executed_prompt_keys"], forbidden | confirmation.manifest_keys(), "Experiment 023's ledger before confirm")
        changed = self.changed_paths(state["phases"]["calibrate"]["commit"])
        if changed is None:
            raise PhaseError("the calibrate commit is not an ancestor of the current commit; lock must be written at the calibrated protocol")
        scientific = b0c.scientific_changes(changed)
        if scientific:
            raise PhaseError(f"scientific paths changed since calibrate: {scientific}")
        record, record_sha = self._installed_record(state, digests)
        b0c.verify_confirmation_binding(record, state, b0c.confirmation_binding(confirmation, confirmation_sha))
        self._cells_unchanged(record["exposed_cells"])
        self._check_runtime(inputs.closure)  # the locked table must reproduce bit for bit at confirm, on 020's runtime
        progs = self._weights_only(inputs)
        noun_keys = [progs.nouns.nouns[index].lexical_key for index in progs.scorable]
        if noun_keys != self._noun_keys(inputs):
            raise PhaseError("the scorable exposed nouns are not the pool's single-token nouns in order")
        units = ul.table_units(confirmation.tokens, confirmation.exposed_frames)
        states = ul.y1_states(inputs.closure["exploration"]["locked_states"], units.frames)
        data_path, index_path = self.output("candidate-locked-y1-table.f64"), self.output("candidate-locked-y1-table.json")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with pytest_free_guard():  # the provenance invariant, executed rather than asserted
                tables = b0c.prediction_tables(progs, units, states, confirmation.reference_ids, log=self.log)
                b0c.enforce_table_gates(tables["gates"])
                index = ul.write_table(data_path, index_path, tables["blocks"], b0c.table_meta("Y1", units, noun_keys))
                again = b0c.prediction_tables(progs, units, states, confirmation.reference_ids)
            if ul.table_bytes(again["blocks"]) != data_path.read_bytes() or (again["p0_dx3_sha256"], again["factors_sha256"]) != (tables["p0_dx3_sha256"], tables["factors_sha256"]):
                raise pm.IncidentError("provenance: the Y1 table does not reproduce bit for bit from the weights and the locked inputs")
        except pm.IncidentError as error:
            self._record_phase_incident(state, "lock", error)
            return 2
        lock = b0c.build_lock(run_id=state["run_id"], protocol_code_commit=self._provenance()["protocol_code_commit"], digests=digests, record=record, record_file_sha256=record_sha,
                              confirmation=confirmation, confirmation_file_sha256=confirmation_sha, cells_files=record["exposed_cells"], y1_index=index,
                              y1_index_sha256=rc.file_sha256(index_path), y1_tables=tables, noun_keys=noun_keys, exposed_states=inputs.closure["exploration"]["locked_states"])
        candidate_path, preregistration_path = self.output("candidate-lock.json"), self.output("candidate-preregistration.md")
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        preregistration = b0c.render_preregistration(lock)
        preregistration_path.write_text(preregistration, encoding="utf-8")
        recheck = self._recheck()
        if not recheck["ok"]:
            raise PhaseError(f"Experiment 020's closure or Experiment 022's committed files no longer verify after lock: {recheck['message']}")
        state["lock"] = {"candidate_path": str(candidate_path), "preregistration_path": str(preregistration_path), "y1_data_path": str(data_path), "y1_index_path": str(index_path),
                         "content_sha256": lock["content_sha256"], "preregistration_sha256": pm.sha256_text(preregistration), "y1_file_sha256": index["file_sha256"],
                         "y1_index_sha256": rc.file_sha256(index_path), "calibration_content_sha256": record["content_sha256"], "gates": tables["gates"], "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now(), "commit": self._provenance()["protocol_code_commit"]}
        self._write(state)
        self.log(f"candidate lock {candidate_path} (content sha256 {lock['content_sha256']}), {preregistration_path} and the Y1 table {data_path} ({index['total_bytes']} bytes, "
                 f"sha256 {index['file_sha256']}) with its index. Install all four byte-identical as {b0c.LOCK_RELATIVE_PATH}, {b0c.PREREGISTRATION_RELATIVE_PATH}, "
                 f"{b0c.Y1_TABLE_RELATIVE_PATH} and {b0c.Y1_TABLE_INDEX_RELATIVE_PATH}, commit, and stop for the independent lock review before confirm.")
        return 0

    def confirm(self) -> int:
        inputs, confirmation_022, confirmation_022_sha, digests, forbidden = self._base()
        confirmation, confirmation_sha = self._confirmation(inputs, confirmation_022, confirmation_022_sha, forbidden)
        state = self._state_for("confirm", digests)
        b0c.assert_ledger_isolated(state["executed_prompt_keys"], forbidden | confirmation.manifest_keys(), "Experiment 023's ledger before confirm")
        paths = {name: self.root / relative for name, relative in (("lock", b0c.LOCK_RELATIVE_PATH), ("preregistration", b0c.PREREGISTRATION_RELATIVE_PATH),
                                                                     ("y1_data", b0c.Y1_TABLE_RELATIVE_PATH), ("y1_index", b0c.Y1_TABLE_INDEX_RELATIVE_PATH))}
        if not all(path.exists() for path in paths.values()):
            raise PhaseError("the committed lock, preregistration and Y1 table must all exist")
        lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
        record, record_sha = self._installed_record(state, digests)
        self._cells_unchanged(record["exposed_cells"])
        git = self.git_state()
        y1_index = b0c.validate_lock(lock, state=state, digests=digests, record=record, record_file_sha256=record_sha, confirmation=confirmation,
                                     confirmation_file_sha256=confirmation_sha, cells_files=record["exposed_cells"], noun_keys=self._noun_keys(inputs),
                                     exposed_states=inputs.closure["exploration"]["locked_states"], preregistration_text=paths["preregistration"].read_text(encoding="utf-8"),
                                     y1_index_text=paths["y1_index"].read_text(encoding="utf-8"), y1_file_sha256=rc.file_sha256(paths["y1_data"]), git_state=git,
                                     tracked=all(self.tracked(path) for path in paths.values()), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
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
                rebuilt = b0c.prediction_tables(progs, units_y1, states["Y1"], confirmation.reference_ids, log=self.log)
            if ul.table_bytes(rebuilt["blocks"]) != paths["y1_data"].read_bytes() or rebuilt["p0_dx3_sha256"] != lock["y1_table"]["p0_dx3_sha256"] \
                    or rebuilt["factors_sha256"] != lock["y1_table"]["factors_sha256"]:
                self._record_phase_incident(state, "confirm", pm.IncidentError("I7 failed: the committed Y1 table does not reproduce bit for bit; no fresh prompt has run"))
                return 2
            y1_blocks = ul.read_table(paths["y1_data"], y1_index)
            state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": commit,
                                          "I7": {"bitwise_equal": True, "file_sha256": lock["y1_table"]["file_sha256"]}}
            pm.record_execution(state, confirmation.stage1_prompts, inputs.pool.nouns)
            self._write(state)
            try:
                executed_1: list[pm.Prompt] = []
                stage1 = b0c.stage_one(model, progs, confirmation, lock, root=self.root, protocol_code_commit=commit, executed=executed_1, log=self.log)
                expected = b0c.stage_one_expectations(stage1)  # kept in memory: the barrier compares the re-read artifacts with what stage 1 wrote
                state["confirmation"] = {"stage1": stage1}
                self._write(state)  # the stage-1 record, with the Y2 table's digests, is on disk
                if Counter(prompt.key for prompt in executed_1) != Counter(prompt.key for prompt in confirmation.stage1_prompts):
                    raise pm.IncidentError("stage 1 did not execute exactly the S1-REF and S1-VALIDITY prompts, once each")
                b0c.enforce_table_gates(stage1["y2_table"]["gates"])
                # Hard boundary: the digested stage-1 record and the Y2 table are re-read from disk and verified before any S2-TARGET prompt.
                state = rd.load_results_state(self.results_path)
                y2_blocks = b0c.barrier(self.root, state, lock, confirmation, expected)
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
                after = b0c.verified_y2_blocks(self.root, stage1, lock, expected)  # any change after the barrier is an incident
                if any(not torch.equal(after[name], y2_blocks[name]) for name in y2_blocks):
                    raise pm.IncidentError("the Y2 table changed after the barrier")
                tables = {"Y1": y1_blocks, "Y2": y2_blocks}
                checked = b0c.target_gates(progs, confirmation, measured, states)
                state["confirmation"]["gates"] = checked["gates"]
                state["confirmation"]["descriptives"] = {"p1_dx3_relative_error": checked["p1_dx3_relative_error"], "block0_profile": checked["block0_profile"]}
                self._write(state)
                b0c.enforce_target_gates(checked["gates"])
                results = b0c.score(measured, tables, lock)
                state["confirmation"].update({**results, "lock_sha256": lock["content_sha256"], "scored_at": pm.utc_now()})
                self._write(state)  # the four results are on disk before anything else runs
                recheck = self._recheck()
                if not recheck["ok"]:
                    raise pm.IncidentError(f"Experiment 020's closure or Experiment 022's committed files no longer verify after confirm: {recheck['message']}")
            except b0c.KernelCheckError as error:
                state["confirmation"] = rc.json_safe(state.get("confirmation") or {})
                state["confirmation"]["kernel_check"] = rc.json_safe(error.details)
                self._record_incident(state, "confirm", error)
                return 2
            except pm.IncidentError as error:
                self._record_incident(state, "confirm", error)
                return 2
            except BaseException as error:  # a protocol failure or an interruption: recorded, then raised; confirm never resumes
                self._record_incident(state, "confirm", error)
                raise
            state["confirmation"]["completed_at"] = pm.utc_now()
            state["phases"]["confirm"] = {**state["phases"]["confirm"], "status": "complete", "completed_at": pm.utc_now()}
            digest = self._write(state)
            self.log("confirm complete (four conditions, no aggregate label): " + "; ".join(f"{key} {entry['result']} (g {entry['g']})" for key, entry in results["conditions"].items())
                     + f"; results sha256 {digest}")
            self._descriptives(state, progs, confirmation, measured, states, tables)
        finally:
            del model
            gc.collect()
        return 0

    def _descriptives(self, state: dict[str, Any], progs: ul.ModelPrograms, confirmation: b0c.Confirmation023, measured: Mapping[str, Any],
                      states: Mapping[str, Any], tables: Mapping[str, Any]) -> None:
        """The descriptive records (no outcome force), computed only after the four results and the completed phase are on
        disk: a failure is recorded as such and the phase stays complete; an interruption is recorded and raised. Neither
        can touch a result."""
        descriptives = state["confirmation"].setdefault("descriptives", {})
        for name, compute in (("subsets", lambda: b0c.fresh_descriptives(measured, tables)),
                              ("comparators", lambda: b0c.comparators(progs, confirmation, measured, states, tables))):
            try:
                descriptives[name] = rc.json_safe(compute())
            except BaseException as error:
                descriptives.setdefault("failures", {})[name] = {"type": type(error).__name__, "message": str(error), "at": pm.utc_now()}
                self._write(state)
                self.log(f"the descriptive record {name} failed (no outcome force; the four results stand): {error}")
                if not isinstance(error, Exception):
                    raise
                continue
            self._write(state)

    def _record_for_report(self) -> dict[str, Any] | None:
        installed = self.root / b0c.CALIBRATION_RELATIVE_PATH
        source = installed if installed.exists() else self.candidate_calibration_path
        return json.loads(source.read_text(encoding="utf-8")) if source.exists() else None

    def report(self) -> int:
        self._base()
        state = rd.load_results_state(self.results_path)
        b0c.assert_phase_allowed("report", state)
        record = self._record_for_report()
        arrays = b0c.draw_values_verified(torch.load(self.draws_path), record) if record is not None and self.draws_path.exists() else None
        text = b0c.render_report(state, record, arrays)
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
    runner = Runner()
    if args.phase not in PHASES:
        raise SystemExit(f"unknown phase {args.phase}")
    return getattr(runner, args.phase)()


if __name__ == "__main__":
    sys.exit(main())
