#!/usr/bin/env python3
"""Run Experiment 025 through isolated phases (design revision 1, ``c0885e5``, corrected ``26c9925``; plan revision 1,
``7d90d28``).

- **``validate``** checks, without a model:
  - the pinned modules and the frozen inputs;
  - Experiment 022's, 023's and 024's committed files, and 020's confirmation file (the prior-noun rule's source);
  - once they exist, the confirmation file, the lock and its preregistration, and the results state.
- **``freeze``** (tokenizer only) writes the 40 fresh cues and the 90,720-key condition-tagged manifest to
  ``confirmation-v1.json``, which is committed by hand. A shortfall or a deviation from the design's expected picks
  writes nothing.
- **``lock``** (weights only; every forward refused) computes the frozen direction, each cue's geometry, the controls,
  the plurality tangent and every patched vector. It enforces the geometry gates, and writes the candidate lock and
  preregistration.
- **``confirm``** (once, never resumed):
  1. validates the lock, including the direct module-blob check;
  2. recomputes the geometry bit for bit and re-enforces every geometry gate (I7′);
  3. runs the patch-path check on four already-executed keys, signed off by the reviewer;
  4. records the ledger, then runs the 90,720 runs once each;
  5. saves the measurements, checks the accounting, recomputes ``C`` from the saved ``Δx3``, and enforces I1, I3, I4
     and the Level-1 identity;
  6. scores A, B and G, writing the result and the completed phase in one atomic write before anything descriptive
     runs.
- **``report``** renders the report.

The production configuration is fixed: the parser takes no option, and ``main`` runs ``cr.PRODUCTION`` only.
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
from neural_decompiler import cue_rotation as cr
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-025"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
PHASES = cr.PHASES
PhaseError = cr.PhaseError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the pinned modules, the frozen inputs, 022–024's committed files and the installed artifacts without a model"),
                             ("freeze", "tokenizer only: freeze the 40 fresh cues and the 90,720-key manifest into confirmation-v1.json (commit it by hand)"),
                             ("lock", "weights only: the direction, the geometry, the controls, the patched vectors, the gates and the preregistration"),
                             ("confirm", "once, never resumed: I7′, the patch-path check, the 90,720 runs, the accounting, C, the gates, the result"),
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
    freeze, the lock and I7′ cannot reach a forward pass."""

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
    inherited: dict[str, dict[str, Any]]  # 024's calibration record, lock and freeze
    prior: dict[str, Any]
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
    config: cr.Configuration = cr.PRODUCTION  # only a test world passes another one, explicitly

    def __post_init__(self) -> None:
        if self.config is not cr.PRODUCTION and Path(self.root).resolve() == ROOT.resolve():
            raise PhaseError("a non-production configuration may never run against the real repository")

    # -- paths ----------------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        return self.results_path.parent

    def output(self, name: str) -> Path:
        return self.output_dir / name

    @property
    def stage2_path(self) -> Path:
        return self.output("stage2-measurements.pt")

    # -- inputs ---------------------------------------------------------------

    def _inputs(self) -> ul.FrozenInputs:
        cr.assert_frozen_blobs()  # the pins first: nothing is loaded through a changed module
        return ul.load_frozen_inputs(self.root, lock_011_loader=self.lock_011_loader, lock_012_loader=self.lock_012_loader, lock_017_loader=self.lock_017_loader,
                                     tracked=self.tracked)

    def _base(self) -> Base:
        """The frozen inputs; 022's, 023's and 024's committed files and 020's confirmation file (verified); the forbidden
        keys: 020's ledger, 020's confirmation set, and the 022, 023 and 024 manifests."""
        inputs = self._inputs()
        digests_022 = b0c.verify_022_inputs(self.root)
        digests_023 = rr.verify_023_inputs(self.root)
        digests_024 = cr.verify_024_inputs(self.root)
        inherited = cr.load_024(self.root)
        prior = cr.prior_nouns(self.root)
        confirmation_022 = json.loads((self.root / ul.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))
        confirmation_023 = json.loads((self.root / b0c.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))
        forbidden = (frozenset(inputs.closure["ledger"]) | frozenset(prompt.key for prompt in inputs.confirmation_020.all_prompts)
                     | _manifest_keys_022(confirmation_022) | _manifest_keys_022(confirmation_023) | frozenset(inherited["confirmation"]["manifest"]["S2-TARGET"]))
        return Base(inputs, cr.base_digests(inputs, digests_022, digests_023, digests_024, prior), inherited, prior, forbidden)

    def _confirmation(self, base: Base) -> tuple[cr.Confirmation025, str]:
        path = self.root / cr.CONFIRMATION_RELATIVE_PATH
        if not path.exists() or not self.tracked(path):
            raise PhaseError(f"{cr.CONFIRMATION_RELATIVE_PATH} must be frozen and committed first (the freeze phase)")
        confirmation = cr.load_confirmation_025(path, base.inputs.pool, base.inherited["confirmation"], base.prior, self.config)
        overlap = confirmation.manifest_keys() & base.forbidden
        if overlap:
            raise PhaseError(f"{len(overlap)} of 025's prompt keys were executed or reserved before, e.g. {sorted(overlap)[:2]}")
        return confirmation, rc.file_sha256(path)

    def _noun_keys(self, inputs: ul.FrozenInputs) -> list[str]:
        """The scorable exposed nouns in ``NounSet`` order (scorability is single-tokenness; no weight is needed)."""
        return [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]

    # -- non-scientific phases ------------------------------------------------

    def validate(self) -> int:
        try:
            base = self._base()
            frozen = "not frozen yet"
            if (self.root / cr.CONFIRMATION_RELATIVE_PATH).exists():
                confirmation, _ = self._confirmation(base)
                frozen = f"confirmation {confirmation.content_sha256[:12]}… ({len(confirmation.tokens)} cues, {len(confirmation.tagged_keys())} condition-tagged keys)"
            installed = "no lock installed"
            lock_path = self.root / cr.LOCK_RELATIVE_PATH
            if lock_path.exists():
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                if lock.get("experiment") != cr.EXPERIMENT or lock.get("content_sha256") != rc.content_digest(lock) or lock.get("configuration") != self.config.to_json():
                    raise PhaseError("the installed lock does not verify against its digest and configuration")
                preregistration = self.root / cr.PREREGISTRATION_RELATIVE_PATH
                if not preregistration.exists() or preregistration.read_text(encoding="utf-8") != cr.render_preregistration(lock):
                    raise PhaseError("the installed preregistration is not the one the installed lock renders")
                installed = "lock and preregistration verified"
            if self.results_path.exists():
                state = rd.load_results_state(self.results_path)
                if state["inputs"] != {key: base.digests[key] for key in cr.DIGEST_KEYS} or state.get("configuration") != self.config.to_json():
                    raise PhaseError("the results state was written against different frozen inputs or another configuration")
                rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden, "Experiment 025's ledger")
        except (PhaseError, rd.PhaseError, pm.IncidentError, ValueError, KeyError) as error:
            self.log(f"validation failed: {error}")
            return 1
        self.log(f"module blobs {len(cr.FROZEN_BLOBS)} verified (024's included); 022–024's committed files and 020's confirmation file verified; "
                 f"{frozen}; {installed}")
        return 0

    def freeze(self) -> int:
        base = self._base()
        path = self.root / cr.CONFIRMATION_RELATIVE_PATH
        if path.exists():
            raise PhaseError(f"{cr.CONFIRMATION_RELATIVE_PATH} already exists; the freeze runs once")
        tokenizer = self.tokenizer_loader(PYTHIA_70M)
        try:
            with pytest_free_guard():
                payload = cr.freeze_payload(tokenizer, pool=base.inputs.pool, freeze_024=base.inherited["confirmation"], prior=base.prior, config=self.config)
        except (cr.FreezeShortfall, cr.FreezeDeviation) as error:
            self.log(f"freeze stopped for review: {error}; nothing was written and no state was created")
            return 3
        confirmation = cr.confirmation_from_payload(payload, base.inputs.pool, self.config)
        overlap = confirmation.manifest_keys() & base.forbidden
        if overlap:
            raise PhaseError(f"{len(overlap)} of the frozen prompt keys were executed or reserved before; nothing was written")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
        confirmation = cr.load_confirmation_025(path, base.inputs.pool, base.inherited["confirmation"], base.prior, self.config)  # re-read, verified
        self.log(f"froze {len(confirmation.tokens)} cues and {len(confirmation.tagged_keys())} condition-tagged keys to {path} (content sha256 "
                 f"{confirmation.content_sha256}, file sha256 {rc.file_sha256(path)}); commit it by hand and stop for the freeze review before lock")
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
            if state.get("experiment") != cr.EXPERIMENT or state["inputs"] != {key: digests[key] for key in cr.DIGEST_KEYS} \
                    or dict(state["versions"]) != provenance["versions"] or state.get("configuration") != self.config.to_json():
                raise PhaseError("frozen inputs, dependency versions or the configuration differ from the recorded run")
            cr.assert_phase_allowed(phase, state)
            return state
        cr.assert_phase_allowed(phase, None)
        return cr.new_results_state(digests=digests, config=self.config, **provenance)

    def _write(self, state: Mapping[str, Any]) -> str:
        return rr.write_state_atomic(self.results_path, state)

    def _recheck(self) -> dict[str, Any]:
        """Experiment 020's closure and 022's, 023's and 024's committed files and 020's confirmation file, verified
        again."""
        try:
            inputs = self._inputs()
            rc.verify_020_closure(self.root, inputs.confirmation_020)
            b0c.verify_022_inputs(self.root)
            rr.verify_023_inputs(self.root)
            cr.verify_024_inputs(self.root)
            cr.prior_nouns(self.root)
        except Exception as error:  # recorded; never allowed to hide the incident being recorded
            return {"ok": False, "message": str(error)}
        return {"ok": True}

    def _record_incident(self, state: dict[str, Any], phase: str, error: BaseException) -> None:
        """An incident of a running confirm: on disk before anything slow runs, with its commit."""
        entry_phase = state.get("phases", {}).get(phase, {})
        commit = str(entry_phase.get("commit") or entry_phase.get("confirm_commit") or state.get("protocol_code_commit") or "")
        entry = {"message": str(error) or type(error).__name__, "type": type(error).__name__, "at": pm.utc_now(), "phase": phase, "commit": commit}
        state["confirmation"] = rc.json_safe(state.get("confirmation") or {})
        state["confirmation"]["incident"] = entry
        self._write(state)
        self.log(f"INCIDENT ({phase}): {error}")
        entry["recheck_020_024"] = self._recheck()
        self._write(state)

    def _record_phase_incident(self, state: dict[str, Any], phase: str, error: BaseException) -> None:
        """An identity failure of lock, or of confirm before the ledger (I7′, the patch path): recorded with its commit;
        the phase is then refused until the reviewer decides."""
        entry = {"message": str(error) or type(error).__name__, "type": type(error).__name__, "at": pm.utc_now(), "phase": phase,
                 "commit": self._provenance()["protocol_code_commit"]}
        state["phases"][phase] = {**state["phases"][phase], "incidents": list(state["phases"][phase].get("incidents", [])) + [entry]}
        self._write(state)
        self.log(f"INCIDENT ({phase}): {error}")
        entry["recheck_020_024"] = self._recheck()
        self._write(state)

    def _preserve_partial(self, state: dict[str, Any], tensors: Mapping[str, torch.Tensor], executed: list[str]) -> None:
        """After a stage-2 incident is on disk (the full measurements were never saved): the runs measured so far are
        saved durably beside the ledger and recorded with their count and last key. A failed save is recorded as such;
        an interruption of the save is recorded and raised. The incident is already written either way."""
        confirmation = state.get("confirmation")
        if confirmation is None or "stage2" in confirmation or not tensors:
            return
        path = self.output("stage2-partial.pt")
        try:
            rr.save_durably(dict(tensors), path)
        except BaseException as error:
            confirmation["stage2_partial"] = {"path": str(path), "save_failed": f"{type(error).__name__}: {error}", "n_executed": len(executed)}
            self._write(state)
            if not isinstance(error, Exception):
                raise
            return
        confirmation["stage2_partial"] = {"path": str(path), "tensors_sha256": {key: rc.tensor_digest(value) for key, value in tensors.items()},
                                          "n_executed": len(executed), "last_key": executed[-1] if executed else None}
        self._write(state)

    def _check_runtime(self, closure: Mapping[str, Any]) -> dict[str, Any]:
        """The runtime and the dependency versions of Experiment 020's explore (021–024's check, unchanged)."""
        runtime = runtime_record(PYTHIA_70M)
        recorded = closure["state"]["phases"]["explore"].get("runtime") or {}
        if {k: v for k, v in runtime.items() if k != "seed"} != {k: v for k, v in recorded.items() if k != "seed"}:
            raise PhaseError(f"the runtime {runtime} differs from Experiment 020's explore runtime {recorded}")
        if dict(self.versions()) != dict(closure["state"]["versions"]):
            raise PhaseError(f"the dependency versions {dict(self.versions())} differ from Experiment 020's {closure['state']['versions']}")
        return runtime

    def _check_nouns(self, progs: ul.ModelPrograms, inputs: ul.FrozenInputs) -> list[str]:
        noun_keys = [progs.nouns.nouns[index].lexical_key for index in progs.scorable]
        if noun_keys != self._noun_keys(inputs) or len(noun_keys) != self.config.n_scored_nouns:
            raise PhaseError("the scorable exposed nouns are not the pool's single-token nouns in order")
        return noun_keys

    def _model_dependencies(self, lock_dependencies: Mapping[str, Any], parameters_sha256: str, embedding_sha256: str) -> None:
        recorded = dict(lock_dependencies)
        cr.verify_dependencies(recorded, {**recorded, "parameters_sha256": parameters_sha256, "embedding_sha256": embedding_sha256}, "the lock (the model)")

    def lock(self) -> int:
        base = self._base()
        confirmation, confirmation_sha = self._confirmation(base)
        state = self._state_for("lock", base.digests)
        rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.manifest_keys() | confirmation.tagged_keys(),
                                  "Experiment 025's ledger before confirm")
        runtime = self._check_runtime(base.inputs.closure)
        commit = self._provenance()["protocol_code_commit"]
        model = self.model_loader(PYTHIA_70M)
        try:
            progs, parameters_sha = ul.ModelPrograms.from_model(model, base.inputs), rr.parameters_digest(model)
        finally:
            del model
            gc.collect()
        noun_keys = self._check_nouns(progs, base.inputs)
        try:
            with pytest_free_guard():
                W_E = progs.weights.W_E
                geometry = cr.geometry_block(W_E, base.inherited["calibration"], base.inputs.pool, confirmation.tokens, self.config)
                again = cr.geometry_block(W_E, base.inherited["calibration"], base.inputs.pool, confirmation.tokens, self.config)
                if geometry["block"] != again["block"]:
                    raise pm.IncidentError("the geometry does not reproduce bit for bit within the lock")
                nearest = cr.nearest_tokens(W_E, geometry["vectors32"], ("base", f"noun+{self.config.primary}", f"noun-{self.config.primary}"))
                dependencies = cr.scientific_dependencies(base.inputs, parameters_sha256=parameters_sha, embedding_sha256=rr.embedding_digest(W_E))
            recheck = self._recheck()
            if not recheck["ok"]:
                raise pm.IncidentError(f"the committed inputs no longer verify at lock: {recheck['message']}; no lock was written")
        except pm.IncidentError as error:
            state["phases"]["lock"] = {**state["phases"]["lock"], "status": "running", "commit": commit}
            self._record_phase_incident(state, "lock", error)
            return 2
        lock = cr.build_lock(run_id=state["run_id"], protocol_code_commit=commit, digests=base.digests, config=self.config, confirmation=confirmation,
                             confirmation_file_sha256=confirmation_sha, geometry=geometry["block"], nearest=nearest, dependencies=dependencies, noun_keys=noun_keys,
                             root=self.root)
        candidate_path, preregistration_path = self.output("candidate-lock.json"), self.output("candidate-preregistration.md")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        preregistration = cr.render_preregistration(json.loads(candidate_path.read_text(encoding="utf-8")))  # rendered from the file that is installed
        preregistration_path.write_text(preregistration, encoding="utf-8")
        state["confirmation_025"] = cr.confirmation_binding(confirmation, confirmation_sha)
        state["lock"] = {"candidate_path": str(candidate_path), "preregistration_path": str(preregistration_path), "content_sha256": lock["content_sha256"],
                         "preregistration_sha256": pm.sha256_text(preregistration), "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now(), "commit": commit, "runtime": runtime}
        self._write(state)
        self.log(f"candidate lock {candidate_path} (content sha256 {lock['content_sha256']}) and {preregistration_path}. Install both byte-identical as "
                 f"{cr.LOCK_RELATIVE_PATH} and {cr.PREREGISTRATION_RELATIVE_PATH}, commit, and stop for the independent lock review before confirm.")
        return 0

    def confirm(self) -> int:
        base = self._base()
        confirmation, confirmation_sha = self._confirmation(base)
        state = self._state_for("confirm", base.digests)
        rr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.manifest_keys() | confirmation.tagged_keys(),
                                  "Experiment 025's ledger before confirm")
        paths = {"lock": self.root / cr.LOCK_RELATIVE_PATH, "preregistration": self.root / cr.PREREGISTRATION_RELATIVE_PATH}
        if not all(path.exists() for path in paths.values()):
            raise PhaseError("the committed lock and preregistration must both exist")
        lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
        git = self.git_state()
        placeholder = cr.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
        cr.validate_lock(lock, state=state, digests=base.digests, config=self.config, confirmation=confirmation, confirmation_file_sha256=confirmation_sha,
                         dependencies=placeholder, noun_keys=self._noun_keys(base.inputs), preregistration_text=paths["preregistration"].read_text(encoding="utf-8"),
                         git_state=git, tracked=all(self.tracked(path) for path in paths.values()), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        commit = str(git.get("commit"))
        self._check_runtime(base.inputs.closure)
        seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            self._model_dependencies(lock["dependencies"]["model"], rr.parameters_digest(model), lock["dependencies"]["model"]["embedding_sha256"])
            progs = ul.ModelPrograms.from_model(model, base.inputs)
            self._check_nouns(progs, base.inputs)
            W_E = progs.weights.W_E
            self._model_dependencies(lock["dependencies"]["model"], lock["dependencies"]["model"]["parameters_sha256"], rr.embedding_digest(W_E))
            try:
                with pytest_free_guard():  # I7′: the geometry and every patched vector, before any prompt, bit for bit
                    recomputed = cr.geometry_block(W_E, base.inherited["calibration"], base.inputs.pool, confirmation.tokens, self.config)
                i7 = cr.verify_geometry_against_lock(recomputed["block"], lock)
                if not i7["bitwise_equal"]:
                    raise pm.IncidentError(f"I7′ failed: the lock's geometry fields {i7['differing']} do not reproduce bit for bit; no fresh prompt has run")
                frames_by_id = {frame.frame_id: frame for frame in base.inputs.pool.frames}
                try:
                    path_check = cr.patch_path_check(model, frames_by_id, W_E, base.forbidden)
                except pm.IncidentError:
                    raise
                except Exception as error:  # the check could not run: an incident before the ledger, as a mismatch is
                    raise pm.IncidentError(f"the patch-path check could not run: {type(error).__name__}: {error}; no fresh prompt has run") from error
                if not path_check["passed"]:
                    state["phases"]["confirm"] = {**state["phases"]["confirm"], "patch_path_check": path_check}  # equality flags and digests only
                    raise pm.IncidentError(f"the patch-path check failed on the spent keys: {path_check['checks']}; no fresh prompt has run")
            except pm.IncidentError as error:
                self._record_phase_incident(state, "confirm", error)
                return 2
            units = cr.target_units(confirmation)
            states = ul.y1_states(base.inputs.closure["exploration"]["locked_states"], units.frames)
            state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": commit,
                                          "I7": i7, "patch_path_check": path_check}
            state["executed_prompt_keys"] = sorted(set(state["executed_prompt_keys"]) | confirmation.tagged_keys())
            cr.record_nouns(state, base.inputs.pool.nouns)
            state["confirmation"] = {}
            self._write(state)  # the 90,720 keys are in the ledger on disk before any of them runs
            executed: list[str] = []
            tensors: dict[str, torch.Tensor] = {}
            try:
                cr.stage_two(model, progs, confirmation, states, recomputed["vectors32"], executed=executed, log=self.log, out=tensors)
                rr.save_durably(tensors, self.stage2_path)
                digests = {key: rc.tensor_digest(value) for key, value in tensors.items()}
                state["confirmation"]["stage2"] = {"path": str(self.stage2_path), "tensors_sha256": digests, "n_executed": len(executed)}
                self._write(state)  # every measurement is on disk before any gate or score can stop the phase
                tensors.clear()  # the re-read copy below is the one every later step uses
                expected = Counter(confirmation.tagged_keys())
                accounting = {"manifest": len(expected), "executed": len(executed), "ledger": len(state["executed_prompt_keys"]),
                              "equal": Counter(executed) == expected and set(state["executed_prompt_keys"]) == set(expected)}
                state["confirmation"]["accounting"] = accounting
                self._write(state)
                if not accounting["equal"]:
                    raise pm.IncidentError(f"the accounting failed: manifest {accounting['manifest']}, executed {accounting['executed']}, ledger {accounting['ledger']}")
                saved = torch.load(self.stage2_path)
                if {key: rc.tensor_digest(value) for key, value in saved.items()} != digests:
                    raise pm.IncidentError("the saved measurements do not reproduce their digests")
                cr.assert_measurements(saved, units, confirmation.conditions, len(progs.scorable))
                c_check = cr.recompute_c(progs, units, states, saved, confirmation.conditions)
                state["confirmation"]["c_recompute"] = c_check
                self._write(state)
                if not c_check["bitwise_equal"]:
                    raise pm.IncidentError(f"C recomputed from the saved Δx3 differs from the saved C: {c_check['max']} at {c_check['at']}")
                gates = cr.run_gates(progs, confirmation, units, states, saved, recomputed["vectors32"], self.config)
                state["confirmation"]["gates"] = gates
                self._write(state)  # the gate values (never the result) before enforcement
                cr.enforce_gates(gates)
                values = cr.per_cue(units, saved, states, confirmation.conditions)
                results = cr.statistics(values, confirmation, self.config)
                recheck = self._recheck()  # before any result is written: an incident carries no result
                if not recheck["ok"]:
                    raise pm.IncidentError(f"the committed inputs no longer verify after confirm: {recheck['message']}; the scored results are void")
            except pm.IncidentError as error:
                self._record_incident(state, "confirm", error)
                self._preserve_partial(state, tensors, executed)
                return 2
            except BaseException as error:  # a protocol failure or an interruption: recorded, then raised; confirm never resumes
                self._record_incident(state, "confirm", error)
                self._preserve_partial(state, tensors, executed)
                raise
            running = dict(state["phases"]["confirm"])
            state["confirmation"].update({"results": results, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()})
            state["phases"]["confirm"] = {**running, "status": "complete", "completed_at": pm.utc_now()}
            try:
                digest = self._write(state)  # one atomic write: A, B, G, the outcome and the completed phase
            except BaseException as error:  # the atomic write left the previous state (no result) on disk: record an incident, never a result
                for key in ("results", "lock_sha256", "completed_at"):
                    state["confirmation"].pop(key, None)
                state["phases"]["confirm"] = running
                self._record_incident(state, "confirm", error)
                if isinstance(error, Exception):
                    return 2
                raise
            self.log(f"confirm complete: {results['outcome']['label']} (A {results['A']['positive']}/{results['A']['n']}, B {results['B']['positive']}/"
                     f"{results['B']['n']}, G {results['G']['positive']}/{results['G']['n']}; threshold {self.config.count_threshold}); results sha256 {digest}")
            self._descriptives(state, progs, confirmation, units, states, saved, values, results, base)
        finally:
            del model
            gc.collect()
        return 0

    def _descriptives(self, state: dict[str, Any], progs: ul.ModelPrograms, confirmation: cr.Confirmation025, units: ul.TableUnits, states: Mapping[str, Any],
                      tensors: Mapping[str, torch.Tensor], values: Mapping[int, Any], results: Mapping[str, Any], base: Base) -> None:
        """The descriptive records (no outcome force), computed only after the result and the completed phase are on disk:
        a failure is recorded as such and the phase stays complete; an interruption is recorded and raised. Neither can
        touch the result."""
        descriptives = state["confirmation"].setdefault("descriptives", {})
        p = self.config.primary
        for name, compute in (("records", lambda: cr.descriptives(values, results, confirmation, self.config, float(base.inherited["calibration"]["line"]["slope"]))),
                              ("ladder", lambda: cr.ladder(progs, units, states, tensors, ("base", f"noun+{p}", f"noun-{p}")))):
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

    def report(self) -> int:
        self._base()
        state = rd.load_results_state(self.results_path)
        cr.assert_phase_allowed("report", state)
        text = cr.render_report(state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(text, encoding="utf-8")
        state["report"] = {"path": str(self.report_path), "sha256": pm.sha256_text(text), "written_at": pm.utc_now()}
        if state["phases"]["report"]["status"] != "complete":
            state["phases"]["report"] = {"status": "complete", "completed_at": pm.utc_now()}
        self._write(state)
        self.log(f"report written to {self.report_path}")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()  # always cr.PRODUCTION
    if args.phase not in PHASES:
        raise SystemExit(f"unknown phase {args.phase}")
    return getattr(runner, args.phase)()


if __name__ == "__main__":
    sys.exit(main())
