#!/usr/bin/env python3
"""Run Experiment 101 through isolated phases (design revision 3, ``a87e284``, decided ``516df20``; plan revision 3,
``96f0cc8``, G4 decided ``1e48c1c``).

- **``validate``** checks the pinned modules, the frozen inputs, 022–025's committed files and every installed
  artifact, without a model.
- **``freeze``** (tokenizer only) writes the 36 new coordinated frames and the S1 and S2 manifests to
  ``confirmation-v1.json``, which the user commits alone. A shortfall or a deviation from the expected picks writes
  nothing.
- **``calibrate``** (exposed, exploratory) re-captures the 36 exposed coordinated reference states, runs every arm on
  them, computes G1–G6, ``τ_rem`` and ``τ_ins``, and evaluates the four revision triggers. A trigger stops for review
  and writes no calibration file.
- **``lock``** (weights only; every forward refused) binds the frames, ``w_c``, every ``v_j``, the calibration, the
  stage-1 files' layout and the statistics, and writes the candidate lock and preregistration.
- **``stage1``** runs only the 36 reference prompts, seals the model, computes the program's table from the states and
  the weights, and writes the stage-1 files for the user to commit.
- **``confirm``** (once, never resumed) passes the barrier, writes the ledger, runs every S2 key once, saves the
  measurements, checks the accounting, computes G2–G6 (a failure is an integrity incident), scores R, I, D and P, and
  writes the result in one atomic write before anything secondary runs.
- **``report``** recomputes the table (G6) and renders the report; it may re-render.

The production configuration is fixed: the parser takes no option, and ``main`` runs ``ri.PRODUCTION`` only.
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
from typing import Any, Callable, Mapping, Sequence

import torch

from neural_decompiler import block0_completion as b0c
from neural_decompiler import cue_rotation as cr
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_intervention as ri
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-101"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
PHASES = ri.PHASES
PhaseError = ri.PhaseError
IncidentError = ri.IncidentError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the pinned modules, the frozen inputs, 022–025's committed files and the installed artifacts without a model"),
                             ("freeze", "tokenizer only: freeze the 36 frames and the S1/S2 manifests into confirmation-v1.json (commit it by hand)"),
                             ("calibrate", "exposed, exploratory: re-capture, run every arm, G1–G6, τ_rem and τ_ins, the revision triggers"),
                             ("lock", "weights only: the geometry, the bindings, the stage-1 layout, the statistics and the preregistration"),
                             ("stage1", "once: the 36 reference prompts, the sealed model, the program's table, the stage-1 files"),
                             ("confirm", "once, never resumed: the barrier, the ledger, every S2 key, the accounting, G2–G6, the result"),
                             ("report", "G6 at the report and the rendered report")):
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
    freeze, the lock, the barrier and G6 cannot reach a forward pass."""

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


def _default_write_check(paths: Sequence[Path]) -> None:
    """Outside a launcher: the stage-1 directory exists and is a directory, and none of the files exists yet."""
    for path in paths:
        if not path.parent.is_dir():
            raise PhaseError(f"the stage-1 directory {path.parent} cannot be created")
        if path.exists():
            raise PhaseError(f"{path} already exists; the stage-1 files are written exactly once")


@dataclass
class Base:
    inputs: ul.FrozenInputs
    digests: dict[str, str]
    prior: dict[str, Any]
    forbidden: frozenset[str]  # every key executed or reserved before 101 (untagged forms included)
    manifests: frozenset[str]  # every confirmation manifest's untagged prompt keys (calibration may run none of them)
    exclusions: ri.Exclusions


@dataclass
class Barrier:
    states: dict[str, rd.FrameState020]
    table: dict[str, Any]
    record: dict[str, Any]


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
    write_check: Callable[[Sequence[Path]], None] = _default_write_check  # the stage-1 launcher installs its own hook's predicate
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))
    config: ri.Configuration = ri.PRODUCTION  # only a test world passes another one, explicitly

    def __post_init__(self) -> None:
        if self.config is not ri.PRODUCTION and Path(self.root).resolve() == ROOT.resolve():
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

    @property
    def calibration_path(self) -> Path:
        return self.output("calibration-measurements.pt")

    def stage1_paths(self) -> dict[str, Path]:
        return {"states_data": self.root / ri.STAGE1_STATES_DATA, "states_index": self.root / ri.STAGE1_STATES_INDEX, "table": self.root / ri.STAGE1_TABLE}

    # -- inputs ---------------------------------------------------------------

    def _inputs(self) -> ul.FrozenInputs:
        ri.assert_frozen_blobs()  # the pins first: nothing is loaded through a changed module
        return ul.load_frozen_inputs(self.root, lock_011_loader=self.lock_011_loader, lock_012_loader=self.lock_012_loader, lock_017_loader=self.lock_017_loader,
                                     tracked=self.tracked)

    def _base(self) -> Base:
        """The frozen inputs; 022's, 023's, 024's and 025's committed files and 020's confirmation file (verified); the
        forbidden keys: 025's (020's ledger, 020's confirmation set, the 022–024 manifests) plus the untagged forms of
        025's manifest; the freeze's exclusion sources."""
        inputs = self._inputs()
        digests_022 = b0c.verify_022_inputs(self.root)
        digests_023 = rr.verify_023_inputs(self.root)
        digests_024 = cr.verify_024_inputs(self.root)
        digests_025 = ri.verify_025_inputs(self.root)
        inherited_024 = cr.load_024(self.root)
        confirmation_025 = ri.load_025(self.root)
        prior = cr.prior_nouns(self.root)
        confirmation_022 = json.loads((self.root / ul.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))
        confirmation_023 = json.loads((self.root / b0c.CONFIRMATION_RELATIVE_PATH).read_text(encoding="utf-8"))
        manifest_025 = frozenset(cr.untagged(key) for key in confirmation_025["manifest"]["S2-TARGET"])
        reserved = (frozenset(prompt.key for prompt in inputs.confirmation_020.all_prompts) | _manifest_keys_022(confirmation_022) | _manifest_keys_022(confirmation_023)
                    | frozenset(inherited_024["confirmation"]["manifest"]["S2-TARGET"]))
        forbidden = frozenset(inputs.closure["ledger"]) | reserved | manifest_025
        manifests = frozenset(cr.untagged(key) for key in reserved) | manifest_025
        exclusions = ri.exclusions_from_committed(confirmation_023=confirmation_023, confirmation_024=inherited_024["confirmation"],
                                                  confirmation_025=confirmation_025, prior=prior)
        digests = ri.base_digests(inputs, digests_022, digests_023, digests_024, prior, digests_025)
        return Base(inputs, digests, prior, forbidden, manifests, exclusions)

    def _confirmation(self, base: Base) -> tuple[ri.Confirmation101, str]:
        path = self.root / ri.CONFIRMATION_RELATIVE_PATH
        if not path.exists() or not self.tracked(path):
            raise PhaseError(f"{ri.CONFIRMATION_RELATIVE_PATH} must be frozen and committed first (the freeze phase)")
        confirmation = ri.load_confirmation_101(path, base.exclusions, self.config)
        overlap = confirmation.untagged_keys() & base.forbidden
        if overlap:
            raise PhaseError(f"{len(overlap)} of 101's prompt keys were executed or reserved before, e.g. {sorted(overlap)[:2]}")
        return confirmation, rc.file_sha256(path)

    def _calibration(self, confirmation: ri.Confirmation101, confirmation_sha: str) -> tuple[dict[str, Any], str]:
        path = self.root / ri.CALIBRATION_RELATIVE_PATH
        if not path.exists() or not self.tracked(path):
            raise PhaseError(f"{ri.CALIBRATION_RELATIVE_PATH} must be installed and committed first (the calibrate phase and its review)")
        return ri.load_calibration_101(path, self.config, confirmation, confirmation_sha), rc.file_sha256(path)

    def _installed_lock(self) -> tuple[dict[str, Any], dict[str, Path]]:
        paths = {"lock": self.root / ri.LOCK_RELATIVE_PATH, "preregistration": self.root / ri.PREREGISTRATION_RELATIVE_PATH}
        if not all(path.exists() for path in paths.values()):
            raise PhaseError("the committed lock and preregistration must both exist")
        return json.loads(paths["lock"].read_text(encoding="utf-8")), paths

    def _validated_lock(self, state: Mapping[str, Any], base: Base, confirmation: ri.Confirmation101, confirmation_sha: str, calibration: Mapping[str, Any],
                        calibration_sha: str, extra_tracked: Sequence[Path] = ()) -> dict[str, Any]:
        lock, paths = self._installed_lock()
        placeholder = ri.scientific_dependencies(base.inputs, parameters_sha256="", embedding_sha256="")
        ri.validate_lock(lock, state=state, digests=base.digests, config=self.config, confirmation=confirmation, confirmation_file_sha256=confirmation_sha,
                         calibration=calibration, calibration_file_sha256=calibration_sha, dependencies=placeholder, noun_keys=self._noun_keys(base.inputs),
                         preregistration_text=paths["preregistration"].read_text(encoding="utf-8"), git_state=self.git_state(),
                         tracked=all(self.tracked(path) for path in (*paths.values(), *extra_tracked)), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        return lock

    def _noun_keys(self, inputs: ul.FrozenInputs) -> list[str]:
        """The scorable exposed nouns in ``NounSet`` order (scorability is single-tokenness; no weight is needed)."""
        return [noun.lexical_key for noun in inputs.pool.nouns if noun.single_token]

    def _exposed_frames(self, inputs: ul.FrozenInputs) -> list[pm.Frame]:
        frames = [frame for frame in inputs.pool.frames if frame.template_id == ri.TEMPLATE]
        if len(frames) != self.config.exposed_frames:
            raise PhaseError(f"the exposed pool has {len(frames)} coordinated frames, not {self.config.exposed_frames}")
        return frames

    def _calibration_schedules(self, inputs: ul.FrozenInputs, frames: Sequence[pm.Frame]) -> list[tuple[ri.RunSpec, ...]]:
        """The calibration's runs: the same arms and tags as the confirmation, on the exposed frames, with 020's own
        prompt labels (``pool.reference_prompt`` and ``pool.token_prompt``)."""
        pool = inputs.pool
        if int(pool.reference_ids[ri.TEMPLATE]) != int(self.config.reference[1]):
            raise PhaseError("020's coordinated reference cue is not the configuration's reference cue")
        names = {}
        for label, cue_id in self.config.cues:
            matches = [name for name, token_id in pool.tokens if int(token_id) == int(cue_id)]
            if len(matches) != 1:
                raise PhaseError(f"020's pool names the cue id {cue_id} {len(matches)} times")
            names[label] = matches[0]
        return [ri.frame_schedule(frame, self.config, pool.reference_prompt(frame), {label: pool.token_prompt(frame, names[label]) for label in self.config.cue_labels})
                for frame in frames]

    # -- non-scientific phases ------------------------------------------------

    def validate(self) -> int:
        try:
            base = self._base()
            frozen = "not frozen yet"
            if (self.root / ri.CONFIRMATION_RELATIVE_PATH).exists():
                confirmation, confirmation_sha = self._confirmation(base)
                frozen = f"confirmation {confirmation.content_sha256[:12]}… ({len(confirmation.frames)} frames, {len(confirmation.s2_keys())} S2 keys)"
                if (self.root / ri.CALIBRATION_RELATIVE_PATH).exists():
                    calibration, _ = self._calibration(confirmation, confirmation_sha)
                    frozen += f"; calibration τ_rem {calibration['tau']['rem']!r}, τ_ins {calibration['tau']['ins']!r}"
            installed = "no lock installed"
            lock_path = self.root / ri.LOCK_RELATIVE_PATH
            if lock_path.exists():
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                if lock.get("experiment") != ri.EXPERIMENT or lock.get("content_sha256") != rc.content_digest(lock) or lock.get("configuration") != self.config.to_json():
                    raise PhaseError("the installed lock does not verify against its digest and configuration")
                preregistration = self.root / ri.PREREGISTRATION_RELATIVE_PATH
                if not preregistration.exists() or preregistration.read_text(encoding="utf-8") != ri.render_preregistration(lock):
                    raise PhaseError("the installed preregistration is not the one the installed lock renders")
                installed = "lock and preregistration verified"
            if self.results_path.exists():
                state = rd.load_results_state(self.results_path)
                if state["inputs"] != {key: base.digests[key] for key in ri.DIGEST_KEYS} or state.get("configuration") != self.config.to_json():
                    raise PhaseError("the results state was written against different frozen inputs or another configuration")
                cr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden, "Experiment 101's ledger")
        except (PhaseError, rd.PhaseError, IncidentError, ValueError, KeyError) as error:
            self.log(f"validation failed: {error}")
            return 1
        self.log(f"module blobs {len(ri.FROZEN_BLOBS)} verified; 022–025's committed files and 020's confirmation file verified; {frozen}; {installed}")
        return 0

    def freeze(self) -> int:
        base = self._base()
        path = self.root / ri.CONFIRMATION_RELATIVE_PATH
        if path.exists():
            raise PhaseError(f"{ri.CONFIRMATION_RELATIVE_PATH} already exists; the freeze runs once")
        tokenizer = self.tokenizer_loader(PYTHIA_70M)
        try:
            with pytest_free_guard():
                payload = ri.freeze_payload(tokenizer, exclusions=base.exclusions, config=self.config)
        except (ri.FreezeShortfall, ri.FreezeDeviation) as error:
            self.log(f"freeze stopped for review: {error}; nothing was written and no state was created")
            return 3
        confirmation = ri.confirmation_from_payload(payload, self.config)
        overlap = confirmation.untagged_keys() & base.forbidden
        if overlap:
            raise PhaseError(f"{len(overlap)} of the frozen prompt keys were executed or reserved before; nothing was written")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
        confirmation = ri.load_confirmation_101(path, base.exclusions, self.config)  # re-read, verified
        self.log(f"froze {len(confirmation.frames)} frames, {len(confirmation.s1_keys())} S1 and {len(confirmation.s2_keys())} S2 keys to {path} (content sha256 "
                 f"{confirmation.content_sha256}, file sha256 {rc.file_sha256(path)}); commit it alone and stop for the freeze review before calibrate")
        return 0

    # -- the state ------------------------------------------------------------

    def _provenance(self) -> dict[str, Any]:
        git = self.git_state()
        return {"protocol_code_commit": str(git.get("commit") or ""), "git_dirty": bool(git.get("dirty", True)), "versions": dict(self.versions())}

    def _state_for(self, phase: str, digests: Mapping[str, str]) -> dict[str, Any]:
        provenance = self._provenance()
        if self.results_path.exists():
            state = rd.load_results_state(self.results_path)
            if provenance["git_dirty"] or not pm._COMMIT_SHA.fullmatch(provenance["protocol_code_commit"]):
                raise PhaseError("scientific execution requires a clean Git tree at a committed SHA")
            if state.get("experiment") != ri.EXPERIMENT or state["inputs"] != {key: digests[key] for key in ri.DIGEST_KEYS} \
                    or dict(state["versions"]) != provenance["versions"] or state.get("configuration") != self.config.to_json():
                raise PhaseError("frozen inputs, dependency versions or the configuration differ from the recorded run")
            ri.assert_phase_allowed(phase, state)
            return state
        ri.assert_phase_allowed(phase, None)
        return ri.new_results_state(digests=digests, config=self.config, **provenance)

    def _write(self, state: Mapping[str, Any]) -> str:
        return rr.write_state_atomic(self.results_path, state)

    def _recheck(self) -> dict[str, Any]:
        """020's closure and 022's–025's committed files and 020's confirmation file, verified again."""
        try:
            inputs = self._inputs()
            rc.verify_020_closure(self.root, inputs.confirmation_020)
            b0c.verify_022_inputs(self.root)
            rr.verify_023_inputs(self.root)
            cr.verify_024_inputs(self.root)
            ri.verify_025_inputs(self.root)
            cr.prior_nouns(self.root)
        except Exception as error:  # recorded; never allowed to hide the incident being recorded
            return {"ok": False, "message": str(error)}
        return {"ok": True}

    def _incident_entry(self, phase: str, error: BaseException, commit: str | None = None) -> dict[str, Any]:
        return {"message": str(error) or type(error).__name__, "type": type(error).__name__, "at": pm.utc_now(), "phase": phase,
                "commit": commit or self._provenance()["protocol_code_commit"]}

    def _record_incident(self, state: dict[str, Any], phase: str, error: BaseException) -> None:
        """An incident of a running confirm (after its ledger): on disk before anything slow runs, with its commit."""
        entry_phase = state.get("phases", {}).get(phase, {})
        entry = self._incident_entry(phase, error, str(entry_phase.get("commit") or entry_phase.get("confirm_commit") or state.get("protocol_code_commit") or ""))
        state["confirmation"] = rc.json_safe(state.get("confirmation") or {})
        state["confirmation"]["incident"] = entry
        self._write(state)
        self.log(f"INCIDENT ({phase}): {error}")
        entry["recheck"] = self._recheck()
        self._write(state)

    def _record_phase_incident(self, state: dict[str, Any], phase: str, error: BaseException, *, after_ledger: bool = False) -> None:
        """An incident of lock, stage1 or confirm before its ledger (or of stage1 after it, which spends the frames):
        recorded with its commit; the phase is then refused for good."""
        entry = {**self._incident_entry(phase, error), "after_ledger": after_ledger}
        state["phases"][phase] = {**state["phases"][phase], "incidents": list(state["phases"][phase].get("incidents", [])) + [entry]}
        self._write(state)
        self.log(f"INCIDENT ({phase}): {error}")
        entry["recheck"] = self._recheck()
        self._write(state)

    def _record_calibration_incident(self, state: dict[str, Any], error: BaseException) -> None:
        calibration = state.get("calibration") or {}
        calibration["incidents"] = list(calibration.get("incidents", [])) + [self._incident_entry("calibrate", error)]
        state["calibration"] = calibration
        self._write(state)
        self.log(f"INCIDENT (calibrate): {error}")

    def _preserve_partial(self, state: dict[str, Any], tensors: Mapping[str, torch.Tensor], executed: Sequence[str], name: str, block: dict[str, Any]) -> None:
        """After an incident is on disk: the runs measured so far are saved durably beside the ledger and recorded with
        their count and last key. A failed save is recorded as such; an interruption of the save is recorded and raised."""
        if not tensors:
            return
        path = self.output(name)
        try:
            rr.save_durably(dict(tensors), path)
        except BaseException as error:
            block["partial"] = {"path": str(path), "save_failed": f"{type(error).__name__}: {error}", "n_executed": len(executed)}
            self._write(state)
            if not isinstance(error, Exception):
                raise
            return
        block["partial"] = {"path": str(path), "tensors_sha256": {key: rc.tensor_digest(value) for key, value in tensors.items()}, "n_executed": len(executed),
                            "last_key": executed[-1] if executed else None}
        self._write(state)

    def _check_runtime(self, closure: Mapping[str, Any]) -> dict[str, Any]:
        """The runtime and the dependency versions of Experiment 020's explore (021–025's check, unchanged)."""
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

    def _model_dependencies(self, lock: Mapping[str, Any], model: Any, progs: ul.ModelPrograms) -> None:
        recorded = dict(lock["dependencies"]["model"])
        now = {**recorded, "parameters_sha256": rr.parameters_digest(model), "embedding_sha256": rr.embedding_digest(progs.weights.W_E)}
        cr.verify_dependencies(recorded, now, "the lock (the model)")

    def _check_stage1_binding(self, lock: Mapping[str, Any], confirmation: ri.Confirmation101, progs: ul.ModelPrograms, model: Any, noun_keys: Sequence[str]) -> None:
        """The lock's stage-1 binding re-derived from the frozen frames and the loaded model (a refusal on any difference)."""
        expected = ri.stage1_binding(confirmation.frames, self.config, d_model=int(progs.weights.W_E.shape[1]), n_heads=int(model.cfg.n_heads),
                                     n_nouns=len(progs.nouns.nouns), noun_keys=noun_keys)
        if pm.canonical_json(expected) != pm.canonical_json(lock["stage1"]):
            raise PhaseError("the lock's stage-1 binding is not the one the frozen frames and the model give")

    def _geometry_against_lock(self, lock: Mapping[str, Any], progs: ul.ModelPrograms, geometry: ri.ReadGeometry, frame_ids: Sequence[str]) -> dict[str, Any]:
        """G1 and the digests of ``w_c`` and every ``v_j``, recomputed from the weights and compared with the lock bit for
        bit (an incident on any difference)."""
        with pytest_free_guard():
            block = ri.geometry_block(geometry, progs.head, progs.axis_T, frame_ids, self.config.k_iso)
        ri.enforce_g1(block)
        check = ri.verify_geometry_against_lock(block, lock)
        if not check["bitwise_equal"]:
            raise IncidentError(f"the lock's geometry fields {check['differing']} do not reproduce bit for bit; no 101 prompt may run")
        return check

    # -- calibrate --------------------------------------------------------------

    def calibrate(self) -> int:
        base = self._base()
        confirmation, confirmation_sha = self._confirmation(base)
        state = self._state_for("calibrate", base.digests)
        commit = self._provenance()["protocol_code_commit"]
        incidents = (state.get("calibration") or {}).get("incidents", [])
        if any(entry["commit"] == commit for entry in incidents):
            raise PhaseError("a calibrate incident is recorded at this commit; a committed fix is required before calibrate runs again")
        if incidents:
            changed = self.changed_paths(incidents[-1]["commit"])
            if changed is None or ri.scientific_changes(changed):
                raise PhaseError("a rerun after a calibrate incident must change no scientific path since the incident's commit (a code fix is a new protocol version)")
        frames = self._exposed_frames(base.inputs)
        schedules = self._calibration_schedules(base.inputs, frames)
        references = [(frame, base.inputs.pool.reference_prompt(frame)) for frame in frames]
        keys = [cr.tagged_key(prompt, "S1") for _, prompt in references] + [run.key for schedule in schedules for run in schedule]
        untagged = sorted({cr.untagged(key) for key in keys})
        foreign = [key for key in untagged if key not in frozenset(base.inputs.closure["ledger"])]
        reserved = [key for key in untagged if key in base.manifests]
        if foreign or reserved or len(set(keys)) != len(keys):
            raise PhaseError(f"the calibration keys are not spent exposed keys only: {len(foreign)} outside 020's ledger (e.g. {foreign[:2]}), {len(reserved)} in a "
                             f"confirmation manifest, {len(keys) - len(set(keys))} repeated")
        runtime = self._check_runtime(base.inputs.closure)
        state["protocol_code_commit"] = commit
        state["confirmation_101"] = ri.confirmation_binding(confirmation, confirmation_sha)
        calibration_block = state.get("calibration") or {}
        calibration_block["planned"] = {"count": len(keys), "sha256": pm.sha256_text(pm.canonical_json(sorted(keys)))}
        state["calibration"] = calibration_block
        state["calibration_executed"] = sorted(set(state["calibration_executed"]) | set(keys))
        state["phases"]["calibrate"] = {"status": "running", "started_at": pm.utc_now(), "commit": commit, "runtime": runtime}
        self._write(state)  # the calibration ledger is on disk before any calibration prompt runs
        seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        executed: list[str] = []
        tensors: dict[str, torch.Tensor] = {}
        logits: dict[str, list[str]] = {}
        try:
            try:
                progs = ul.ModelPrograms.from_model(model, base.inputs)
                noun_keys = self._check_nouns(progs, base.inputs)
                geometry = ri.read_geometry(base.inputs.lock_011)
                with pytest_free_guard():
                    block = ri.geometry_block(geometry, progs.head, progs.axis_T, [frame.frame_id for frame in frames], self.config.k_iso)
                g1_failures = []
                try:
                    ri.enforce_g1(block)
                except IncidentError as error:
                    g1_failures.append(str(error))
                recaptured: dict[str, rd.FrameState020] = {}
                for frame, prompt in references:
                    recaptured[frame.frame_id] = ri.capture_reference(model, progs, prompt, "S1")
                    executed.append(cr.tagged_key(prompt, "S1"))
                    if (recaptured[frame.frame_id].p_c, recaptured[frame.frame_id].p_t) != (frame.p_c, frame.p_t):
                        raise IncidentError(f"{frame.frame_id}: the re-capture's positions break the frame's structure")
                locked = base.inputs.closure["exploration"]["locked_states"]
                drift = {frame.frame_id: ri.state_drift(rd.locked_state(recaptured[frame.frame_id]), locked[frame.frame_id]) for frame in frames}
                worst = max(drift.items(), key=lambda item: item[1])
                environment = {"max_state_drift": worst[1], "at": worst[0], "tolerance": ri.TOLERANCES["environment"],
                               "passed": bool(worst[1] <= ri.TOLERANCES["environment"])}
                ri.stage_runs(model, progs, schedules, geometry, self.config, executed=executed, out=tensors, logits=logits, log=self.log)
                rr.save_durably(tensors, self.calibration_path)
                digests = {key: rc.tensor_digest(value) for key, value in tensors.items()}
                state["calibration"]["measurements"] = {"path": str(self.calibration_path), "tensors_sha256": digests, "logits_sha256": logits, "n_executed": len(executed)}
                self._write(state)
                tensors.clear()
                accounting = {"planned": len(keys), "executed": len(executed), "equal": Counter(executed) == Counter(keys)}
                state["calibration"]["accounting"] = accounting
                self._write(state)
                if not accounting["equal"]:
                    raise IncidentError(f"the calibration accounting failed: planned {len(keys)}, executed {len(executed)}")
                saved = torch.load(self.calibration_path)
                if {key: rc.tensor_digest(value) for key, value in saved.items()} != digests:
                    raise IncidentError("the saved calibration measurements do not reproduce their digests")
                ri.assert_measurements(saved, schedules)
                program_states = {frame.frame_id: rd.state_from_locked(locked[frame.frame_id], frame) for frame in frames}
                table = ri.prediction_table(progs, frames, program_states, geometry, self.config, noun_keys, fractions_for_weak=True)
                book, ceilings = ri.run_gates(progs, schedules, saved, logits, program_states, recaptured, geometry, self.config)
                values = [ri.frame_values(schedule, saved, self.config, fractions_for_weak=True) for schedule in schedules]
                values_by_cue = {label: [entry[label] for entry in values] for label in self.config.cue_labels}
                pairs = {}
                for schedule in schedules:
                    frame_id = schedule[0].frame.frame_id
                    index = {run.tag: row for row, run in enumerate(schedule)}
                    x3 = saved[f"{frame_id}/x3_pc"]
                    pairs[frame_id] = {label: ri.pair_quantities(x3[index[ri.tag_of("PL", label)]].double() - x3[index["REF"]].double(), geometry).record()
                                       for label in self.config.cue_labels}
                gates = {**book.to_json(), "g1": g1_failures}
                if g1_failures:
                    gates["failures"] = list(gates["failures"]) + ["G1"]
                results = ri.calibration_results([frame.frame_id for frame in frames], values_by_cue, table, pairs, book, environment, self.config,
                                                 extra_failures=["G1"] if g1_failures else [])
                state["calibration"]["gates"] = gates
                state["calibration"]["ceilings"] = rc.json_safe(ceilings)
                self._write(state)
            except BaseException as error:
                self._record_calibration_incident(state, error)
                self._preserve_partial(state, tensors, executed, "calibration-partial.pt", state["calibration"])
                if isinstance(error, Exception):
                    return 2
                raise
        finally:
            del model
            gc.collect()
        payload = ri.calibration_payload(config=self.config, confirmation_binding_=ri.confirmation_binding(confirmation, confirmation_sha), results=results, gates=gates,
                                         environment=environment, g1=block["g1"], keys_sha256=state["calibration"]["planned"]["sha256"], n_keys=len(keys),
                                         tensors_sha256=digests, protocol_code_commit=commit)
        if results["stopped"]:
            state["calibration"]["stopped"] = {"triggers": results["triggers"], "tau": results["tau"], "content_sha256": payload["content_sha256"]}
            state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "stopped_for_review", "stopped_at": pm.utc_now()}
            self._write(state)
            fired = [name for name, entry in results["triggers"].items() if entry["fired"]]
            self.log(f"calibration stopped for review: revision triggers {fired}; no calibration file was written; the values are in the state")
            return 3
        candidate = self.output("candidate-calibration-v1.json")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidate.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
        state["calibration"].update({"candidate_path": str(candidate), "content_sha256": payload["content_sha256"], "file_sha256": rc.file_sha256(candidate),
                                     "tau": payload["tau"]})
        state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "complete", "completed_at": pm.utc_now()}
        self._write(state)
        self.log(f"calibration complete: τ_rem {payload['tau']['rem']!r}, τ_ins {payload['tau']['ins']!r} (P evaluable {payload['tau']['evaluable']}); candidate "
                 f"{candidate}. Install it byte-identical as {ri.CALIBRATION_RELATIVE_PATH}, commit, and stop for the calibration review before lock.")
        return 0

    # -- lock -------------------------------------------------------------------

    def lock(self) -> int:
        base = self._base()
        confirmation, confirmation_sha = self._confirmation(base)
        calibration, calibration_sha = self._calibration(confirmation, confirmation_sha)
        state = self._state_for("lock", base.digests)
        if (state.get("calibration") or {}).get("content_sha256") != calibration["content_sha256"]:
            raise PhaseError("the installed calibration file is not the candidate this run's calibration wrote")
        changed = self.changed_paths(state["phases"]["calibrate"]["commit"])
        if changed is None or ri.scientific_changes(changed):
            raise PhaseError(f"scientific paths changed since the calibration ({changed}); τ and the G4 record bind the code they were computed with")
        cr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.untagged_keys(), "Experiment 101's ledger before stage 1")
        runtime = self._check_runtime(base.inputs.closure)
        commit = self._provenance()["protocol_code_commit"]
        model = self.model_loader(PYTHIA_70M)
        try:
            progs, parameters_sha, n_heads = ul.ModelPrograms.from_model(model, base.inputs), rr.parameters_digest(model), int(model.cfg.n_heads)
        finally:
            del model
            gc.collect()
        noun_keys = self._check_nouns(progs, base.inputs)
        try:
            with pytest_free_guard():
                geometry = ri.read_geometry(base.inputs.lock_011)
                frame_ids = [frame.frame_id for frame in confirmation.frames]
                block = ri.geometry_block(geometry, progs.head, progs.axis_T, frame_ids, self.config.k_iso)
                again = ri.geometry_block(geometry, progs.head, progs.axis_T, frame_ids, self.config.k_iso)
                if pm.canonical_json(ri.encode_nonfinite(block)) != pm.canonical_json(ri.encode_nonfinite(again)):
                    raise IncidentError("the geometry does not reproduce bit for bit within the lock")
                ri.enforce_g1(block)
                dependencies = ri.scientific_dependencies(base.inputs, parameters_sha256=parameters_sha, embedding_sha256=rr.embedding_digest(progs.weights.W_E))
                stage1 = ri.stage1_binding(confirmation.frames, self.config, d_model=geometry.d_model, n_heads=n_heads, n_nouns=len(progs.nouns.nouns),
                                           noun_keys=noun_keys)
                exposed = self._exposed_frames(base.inputs)
                locked = base.inputs.closure["exploration"]["locked_states"]
                states = {frame.frame_id: rd.state_from_locked(locked[frame.frame_id], frame) for frame in exposed}
                table = ri.prediction_table(progs, exposed, states, geometry, self.config, noun_keys, fractions_for_weak=True)
                if table["content_sha256"] != calibration["table_content_sha256"]:
                    raise IncidentError("the exposed prediction table does not reproduce across processes bit for bit (the calibration's digest); "
                                        "no 101 prompt has run")
            recheck = self._recheck()
            if not recheck["ok"]:
                raise IncidentError(f"the committed inputs no longer verify at lock: {recheck['message']}; no lock was written")
        except IncidentError as error:
            state["phases"]["lock"] = {**state["phases"]["lock"], "status": "running", "commit": commit}
            self._record_phase_incident(state, "lock", error)
            return 2
        lock = ri.build_lock(run_id=state["run_id"], protocol_code_commit=commit, digests=base.digests, config=self.config, confirmation=confirmation,
                             confirmation_file_sha256=confirmation_sha, calibration=calibration, calibration_file_sha256=calibration_sha, geometry=block,
                             dependencies=dependencies, noun_keys=noun_keys, stage1=stage1)
        candidate_path, preregistration_path = self.output("candidate-lock.json"), self.output("candidate-preregistration.md")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        preregistration = ri.render_preregistration(json.loads(candidate_path.read_text(encoding="utf-8")))  # rendered from the file that is installed
        preregistration_path.write_text(preregistration, encoding="utf-8")
        state["lock"] = {"candidate_path": str(candidate_path), "preregistration_path": str(preregistration_path), "content_sha256": lock["content_sha256"],
                         "preregistration_sha256": pm.sha256_text(preregistration), "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now(), "commit": commit, "runtime": runtime}
        self._write(state)
        self.log(f"candidate lock {candidate_path} (content sha256 {lock['content_sha256']}) and {preregistration_path}. Install both byte-identical as "
                 f"{ri.LOCK_RELATIVE_PATH} and {ri.PREREGISTRATION_RELATIVE_PATH}, commit, and stop for the independent lock review before stage1.")
        return 0

    # -- stage 1 ----------------------------------------------------------------

    def stage1(self) -> int:
        base = self._base()
        confirmation, confirmation_sha = self._confirmation(base)
        calibration, calibration_sha = self._calibration(confirmation, confirmation_sha)
        state = self._state_for("stage1", base.digests)
        lock = self._validated_lock(state, base, confirmation, confirmation_sha, calibration, calibration_sha)
        cr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden | confirmation.untagged_keys(), "Experiment 101's ledger before stage 1")
        paths = self.stage1_paths()
        if any(path.exists() for path in paths.values()):
            raise PhaseError("a stage-1 file already exists; the stage-1 files are written exactly once")
        runtime = self._check_runtime(base.inputs.closure)
        commit = self._provenance()["protocol_code_commit"]
        seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            progs = ul.ModelPrograms.from_model(model, base.inputs)
            self._model_dependencies(lock, model, progs)
            noun_keys = self._check_nouns(progs, base.inputs)
            geometry = ri.read_geometry(base.inputs.lock_011)
            self._check_stage1_binding(lock, confirmation, progs, model, noun_keys)
            try:
                check = self._geometry_against_lock(lock, progs, geometry, [frame.frame_id for frame in confirmation.frames])
            except IncidentError as error:
                self._record_phase_incident(state, "stage1", error)
                return 2
            paths["table"].parent.mkdir(parents=True, exist_ok=True)
            self.write_check(list(paths.values()))  # a refusal: nothing is recorded and no prompt has run
            executed: list[str] = []
            state["executed_prompt_keys"] = sorted(set(state["executed_prompt_keys"]) | set(confirmation.s1_keys()))
            state["phases"]["stage1"] = {"status": "running", "started_at": pm.utc_now(), "commit": commit, "runtime": runtime, "geometry": check}
            self._write(state)  # the 36 S1 keys are in the ledger on disk before the first reference prompt
            try:
                locked: dict[str, dict[str, Any]] = {}
                for frame in confirmation.frames:
                    prompt = confirmation.reference_prompt(frame)
                    captured = ri.capture_reference(model, progs, prompt, "S1")
                    executed.append(cr.tagged_key(prompt, "S1"))
                    if (captured.p_c, captured.p_t) != (frame.p_c, frame.p_t) or frame.p_t != frame.p_c + 1:
                        raise IncidentError(f"{frame.frame_id}: the reference capture's positions ({captured.p_c}, {captured.p_t}) break the frame's frozen structure")
                    locked[frame.frame_id] = rd.locked_state(captured)
                    self.log(f"  {frame.frame_id}: reference state captured")
                del model  # sealed: no forward pass is possible from here on
                model = None
                gc.collect()
                if Counter(executed) != Counter(confirmation.s1_keys()):
                    raise IncidentError("the stage-1 accounting failed")
                states = {frame.frame_id: rd.state_from_locked(locked[frame.frame_id], frame) for frame in confirmation.frames}
                with pytest_free_guard():
                    table = ri.prediction_table(progs, confirmation.frames, states, geometry, self.config, noun_keys)
                if any(path.exists() for path in paths.values()):
                    raise IncidentError("a stage-1 file appeared after the ledger; nothing is overwritten")
                index = ul.write_table(paths["states_data"], paths["states_index"], ri.states_blocks(locked, confirmation.frames), lock["stage1"]["states"]["meta"])
                ul.verify_table_index(index, layout=lock["stage1"]["states"]["layout"], meta=lock["stage1"]["states"]["meta"])
                paths["table"].write_text(pm.canonical_json(table) + "\n", encoding="utf-8")
                record = {"frames": [{"frame_id": frame.frame_id, "p_c": frame.p_c, "p_t": frame.p_t} for frame in confirmation.frames],
                          "state_digests": {frame_id: rd.state_digest(entry) for frame_id, entry in sorted(locked.items())},
                          "files": {name: {"path": str(path.relative_to(self.root)), "sha256": rc.file_sha256(path)} for name, path in paths.items()},
                          "table_content_sha256": table["content_sha256"], "lock_sha256": lock["content_sha256"], "commit": commit, "noun_keys": list(noun_keys),
                          "executed": len(executed)}
                record["digest"] = b0c.stage_one_digest(record)
                state["stage1"] = record
                state["phases"]["stage1"] = {**state["phases"]["stage1"], "status": "complete", "completed_at": pm.utc_now(), "awaiting_commit": True}
                self._write(state)
            except BaseException as error:
                self._record_phase_incident(state, "stage1", error, after_ledger=True)
                if isinstance(error, Exception):
                    return 2
                raise
        finally:
            del model
            gc.collect()
        self.log(f"stage 1 complete: {len(executed)} reference states; table {table['content_sha256']}; files {[str(path.relative_to(self.root)) for path in paths.values()]}. "
                 f"Commit them exactly as written, then confirm.")
        return 0

    # -- confirm ----------------------------------------------------------------

    def _committed_stage1(self, lock: Mapping[str, Any], confirmation: ri.Confirmation101, record: Mapping[str, Any]) -> tuple[dict[str, rd.FrameState020], dict[str, Any]]:
        """The committed stage-1 files re-read from disk against the stage-1 record: each file's sha256 and path, the
        states table against the lock's layout and metadata, every state's digest, the table's digest and its bound
        construction. Any failure is an incident (the caller records it)."""
        paths = self.stage1_paths()
        for name, path in paths.items():
            if record["files"][name]["path"] != str(path.relative_to(self.root)) or rc.file_sha256(path) != record["files"][name]["sha256"]:
                raise IncidentError(f"the committed stage-1 file {name} is not the one stage 1 wrote")
        index = json.loads(paths["states_index"].read_text(encoding="utf-8"))
        ul.verify_table_index(index, layout=lock["stage1"]["states"]["layout"], meta=lock["stage1"]["states"]["meta"])
        locked = ri.states_from_blocks(ul.read_table(paths["states_data"], index), confirmation.frames)
        if {frame_id: rd.state_digest(entry) for frame_id, entry in sorted(locked.items())} != record["state_digests"]:
            raise IncidentError("a committed stage-1 state does not reproduce its digest")
        states = {frame.frame_id: rd.state_from_locked(locked[frame.frame_id], frame) for frame in confirmation.frames}
        table = json.loads(paths["table"].read_text(encoding="utf-8"))
        if table.get("content_sha256") != record["table_content_sha256"] or table["content_sha256"] != rc.content_digest(table):
            raise IncidentError("the committed table does not reproduce its digest")
        if {key: table.get(key) for key in lock["stage1"]["table"]} != dict(lock["stage1"]["table"]):
            raise IncidentError("the committed table's construction is not the bound one")
        return states, table

    def _recompute_table(self, progs: ul.ModelPrograms, confirmation: ri.Confirmation101, states: Mapping[str, rd.FrameState020], geometry: ri.ReadGeometry,
                         noun_keys: Sequence[str], committed: Mapping[str, Any]) -> bool:
        with pytest_free_guard():
            recomputed = ri.prediction_table(progs, confirmation.frames, states, geometry, self.config, noun_keys)
        return pm.canonical_json(recomputed) == pm.canonical_json(committed)

    def _barrier(self, lock: Mapping[str, Any], confirmation: ri.Confirmation101, progs: ul.ModelPrograms, geometry: ri.ReadGeometry,
                 noun_keys: Sequence[str]) -> Barrier:
        """The hard boundary, on the state re-read from disk: the stage-1 record reproduces its digest and names this
        lock; the committed files are the ones stage 1 wrote (``_committed_stage1``); the table recomputed from the
        committed states and the weights equals the committed table bit for bit; no S2 key is in the ledger. Any failure
        is an incident."""
        state = rd.load_results_state(self.results_path)
        record = state.get("stage1")
        if record is None or record.get("digest") != b0c.stage_one_digest(record):
            raise IncidentError("the stage-1 record does not reproduce its digest; no S2 prompt may run")
        if record.get("lock_sha256") != lock["content_sha256"]:
            raise IncidentError("the stage-1 record was written under a different lock")
        states, committed = self._committed_stage1(lock, confirmation, record)
        if not self._recompute_table(progs, confirmation, states, geometry, noun_keys, committed):
            raise IncidentError("the table recomputed from the committed states and the weights differs from the committed table")
        overlap = set(confirmation.s2_keys()) & set(state["executed_prompt_keys"])
        if overlap:
            raise IncidentError(f"{len(overlap)} S2 keys are in the ledger before the barrier, e.g. {sorted(overlap)[:2]}")
        return Barrier(states, committed, record)

    def confirm(self) -> int:
        base = self._base()
        confirmation, confirmation_sha = self._confirmation(base)
        calibration, calibration_sha = self._calibration(confirmation, confirmation_sha)
        state = self._state_for("confirm", base.digests)
        paths = self.stage1_paths()
        if not all(path.exists() for path in paths.values()) or not all(self.tracked(path) for path in paths.values()):
            raise PhaseError("the stage-1 files must exist and be committed before confirm")
        lock = self._validated_lock(state, base, confirmation, confirmation_sha, calibration, calibration_sha, extra_tracked=list(paths.values()))
        cr.assert_ledger_isolated(state["executed_prompt_keys"], base.forbidden, "Experiment 101's ledger before confirm")
        runtime = self._check_runtime(base.inputs.closure)
        commit = self._provenance()["protocol_code_commit"]
        seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            progs = ul.ModelPrograms.from_model(model, base.inputs)
            self._model_dependencies(lock, model, progs)
            noun_keys = self._check_nouns(progs, base.inputs)
            geometry = ri.read_geometry(base.inputs.lock_011)
            self._check_stage1_binding(lock, confirmation, progs, model, noun_keys)
            try:
                self._geometry_against_lock(lock, progs, geometry, [frame.frame_id for frame in confirmation.frames])
                barrier = self._barrier(lock, confirmation, progs, geometry, noun_keys)
            except IncidentError as error:
                self._record_phase_incident(state, "confirm", error)
                return 2
            state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": commit,
                                          "runtime": runtime}
            state["executed_prompt_keys"] = sorted(set(state["executed_prompt_keys"]) | set(confirmation.s2_keys()))
            state["confirmation"] = {"barrier": {"passed": True, "stage1_digest": barrier.record["digest"], "table_content_sha256": barrier.table["content_sha256"]}}
            self._write(state)  # the 1,980 S2 keys are in the ledger on disk before any of them runs
            executed: list[str] = []
            tensors: dict[str, torch.Tensor] = {}
            logits: dict[str, list[str]] = {}
            schedules = [confirmation.schedule(frame) for frame in confirmation.frames]
            frame_ids = [frame.frame_id for frame in confirmation.frames]
            try:
                ri.stage_runs(model, progs, schedules, geometry, self.config, executed=executed, out=tensors, logits=logits, log=self.log)
                rr.save_durably(tensors, self.stage2_path)
                digests = {key: rc.tensor_digest(value) for key, value in tensors.items()}
                state["confirmation"]["stage2"] = {"path": str(self.stage2_path), "tensors_sha256": digests, "logits_sha256": logits, "n_executed": len(executed)}
                self._write(state)  # every measurement is on disk before any gate or score can stop the phase
                tensors.clear()
                expected = Counter(confirmation.s2_keys())
                ledger_s2 = {key for key in state["executed_prompt_keys"] if key.rsplit("|", 1)[1] != "S1"}
                accounting = {"manifest": len(expected), "executed": len(executed), "ledger": len(ledger_s2),
                              "equal": Counter(executed) == expected and ledger_s2 == set(expected)}
                state["confirmation"]["accounting"] = accounting
                self._write(state)
                if not accounting["equal"]:
                    raise IncidentError(f"the accounting failed: manifest {accounting['manifest']}, executed {accounting['executed']}, ledger {accounting['ledger']}")
                saved = torch.load(self.stage2_path)
                if {key: rc.tensor_digest(value) for key, value in saved.items()} != digests:
                    raise IncidentError("the saved measurements do not reproduce their digests")
                ri.assert_measurements(saved, schedules)
                book, ceilings = ri.run_gates(progs, schedules, saved, logits, barrier.states, barrier.states, geometry, self.config)
                state["confirmation"]["gates"] = book.to_json()
                state["confirmation"]["ceilings"] = rc.json_safe(ceilings)
                self._write(state)  # the gate values (never a result) before enforcement
                failures = book.failures()
                if failures:
                    raise IncidentError(f"integrity failure after the measurements: gates {failures} failed; outcome scoring stops, no frame is dropped or "
                                        f"replaced, and the failure is reviewed under the incident procedure")
                reread_states, reread_table = self._committed_stage1(lock, confirmation, barrier.record)
                g6 = {"bitwise_equal": pm.canonical_json(reread_table) == pm.canonical_json(barrier.table)
                      and self._recompute_table(progs, confirmation, reread_states, geometry, noun_keys, reread_table)}
                state["confirmation"]["g6_before_result"] = g6
                self._write(state)
                if not g6["bitwise_equal"]:
                    raise IncidentError("G6 failed before the result write: the table no longer reproduces from its committed inputs")
                values = [ri.frame_values(schedule, saved, self.config, fractions_for_weak=False) for schedule in schedules]
                values_by_cue = {label: [entry[label] for entry in values] for label in self.config.cue_labels}
                predicted = [barrier.table["frames"][frame_id]["cues"][self.config.primary[0]] for frame_id in frame_ids]
                tau = lock["calibration_101"]
                results = ri.primary_statistics(frame_ids, values_by_cue[self.config.primary[0]], predicted, float(tau["tau_rem"]), float(tau["tau_ins"]), self.config)
                recheck = self._recheck()  # before any result is written: an incident carries no result
                if not recheck["ok"]:
                    raise IncidentError(f"the committed inputs no longer verify after confirm: {recheck['message']}; the scored results are void")
            except BaseException as error:  # an incident or an interruption: recorded; confirm never resumes
                self._record_incident(state, "confirm", error)
                self._preserve_partial(state, tensors, executed, "stage2-partial.pt", state["confirmation"])
                if isinstance(error, Exception):
                    return 2
                raise
            running = dict(state["phases"]["confirm"])
            state["confirmation"].update({"results": results, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()})
            state["phases"]["confirm"] = {**running, "status": "complete", "completed_at": pm.utc_now()}
            try:
                digest = self._write(state)  # one atomic write: R, I, D, P, the label and the completed phase
            except BaseException as error:  # the atomic write left the previous state (no result) on disk: an incident, never a result
                for key in ("results", "lock_sha256", "completed_at"):
                    state["confirmation"].pop(key, None)
                state["phases"]["confirm"] = running
                self._record_incident(state, "confirm", error)
                if isinstance(error, Exception):
                    return 2
                raise
            self.log(f"confirm complete: {results['outcome']['label']} (R {results['R']['count']}/{results['n']} {results['R']['result']}, I {results['I']['count']}/"
                     f"{results['n']} {results['I']['result']}, D {results['D']['result']}, P {results['P']['result']}); results sha256 {digest}")
            self._secondary(state, progs, schedules, saved, geometry, frame_ids, values_by_cue, results)
        finally:
            del model
            gc.collect()
        return 0

    def _secondary(self, state: dict[str, Any], progs: ul.ModelPrograms, schedules: Sequence[Sequence[ri.RunSpec]], tensors: Mapping[str, torch.Tensor],
                   geometry: ri.ReadGeometry, frame_ids: Sequence[str], values_by_cue: Mapping[str, Any], results: Mapping[str, Any]) -> None:
        """The secondary records (no label force), computed only after the result and the completed phase are on disk:
        a failure is recorded as such and the phase stays complete; an interruption is recorded and raised."""
        secondary = state["confirmation"].setdefault("secondary", {})
        steps = (("values", lambda: ri.encode_nonfinite({label: list(entries) for label, entries in values_by_cue.items()})),
                 ("records", lambda: ri.run_records(progs, schedules, tensors, geometry, self.config)),
                 ("statistics", lambda: ri.secondary_statistics(frame_ids, values_by_cue, results, secondary["records"], self.config)))
        for name, compute in steps:
            try:
                secondary[name] = compute()
            except BaseException as error:
                secondary.setdefault("failures", {})[name] = {"type": type(error).__name__, "message": str(error), "at": pm.utc_now()}
                self._write(state)
                self.log(f"the secondary record {name} failed (no label force; the result stands): {error}")
                if not isinstance(error, Exception):
                    raise
                continue
            self._write(state)

    # -- report -----------------------------------------------------------------

    def report(self) -> int:
        base = self._base()
        state = rd.load_results_state(self.results_path)
        ri.assert_phase_allowed("report", state)
        results = (state.get("confirmation") or {}).get("results")
        block = dict(state.get("report") or {})
        if results:
            confirmation, _ = self._confirmation(base)
            lock, _ = self._installed_lock()
            self._check_runtime(base.inputs.closure)  # the recompute runs under the recorded runtime, or the report refuses
            block["g6"] = self._g6_at_report(base, lock, confirmation)
        state["report"] = block
        text = ri.render_report(state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(text, encoding="utf-8")
        state["report"] = {**block, "path": str(self.report_path), "sha256": pm.sha256_text(text), "written_at": pm.utc_now(),
                           "renders": int(block.get("renders", 0)) + 1}
        if state["phases"]["report"]["status"] != "complete":
            state["phases"]["report"] = {"status": "complete", "completed_at": pm.utc_now()}
        self._write(state)
        self.log(f"report written to {self.report_path}")
        return 0

    def _g6_at_report(self, base: Base, lock: Mapping[str, Any], confirmation: ri.Confirmation101) -> dict[str, Any]:
        """G6 at the report: the committed stage-1 files against the stage-1 record, and the table recomputed from the
        committed states and the weights (no forward pass) against the committed table. A mismatch is recorded, and the
        report then renders NOT_INTERPRETABLE; the result is never rewritten."""
        model = self.model_loader(PYTHIA_70M)
        try:
            progs = ul.ModelPrograms.from_model(model, base.inputs)
        finally:
            del model
            gc.collect()
        try:
            state = rd.load_results_state(self.results_path)
            noun_keys = self._check_nouns(progs, base.inputs)
            geometry = ri.read_geometry(base.inputs.lock_011)
            states, committed = self._committed_stage1(lock, confirmation, state["stage1"])
            equal = self._recompute_table(progs, confirmation, states, geometry, noun_keys, committed)
            return {"bitwise_equal": equal, "message": "" if equal else "the recomputed table differs", "at": pm.utc_now()}
        except Exception as error:
            return {"bitwise_equal": False, "message": f"{type(error).__name__}: {error}", "at": pm.utc_now()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()  # always ri.PRODUCTION
    if args.phase not in PHASES:
        raise SystemExit(f"unknown phase {args.phase}")
    return getattr(runner, args.phase)()


if __name__ == "__main__":
    sys.exit(main())
