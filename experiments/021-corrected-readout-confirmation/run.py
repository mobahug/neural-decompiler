#!/usr/bin/env python3
"""Run Experiment 021 through isolated phases (design revision 3, ``0ac46aa``).

``validate`` checks the frozen inputs, the unchanged readout program, Experiment 020's closure, its evidence extract,
its local results state and the confirmation set's isolation — without a model. ``calibrate`` (once) re-executes
exactly Experiment 020's exposed ledger with 020's own functions in 020's order, checks the re-captured reference
states, enforces the identities and the ``1e-9`` reproduction gate, applies the frozen validity screen and the
calibration precondition, and only then draws: 10,000 SHA-indexed base draws, the statistics of every admissible row,
the kernel/direct cross-check, the floors and the descriptives, written as the candidate calibration record. The
record is installed byte-identical and committed by hand, and the floors are reviewed, before ``lock`` — which runs no
forward pass — writes the Y1 prediction table and the floor tables. ``confirm`` (once) validates the lock, runs stage 1
(each fresh frame's S1-REF and S1-VALIDITY prompts only), selects and digests the Y2 floor row from the validity
verdicts, re-reads the record at the barrier, then runs stage 2 and scores the outcome on the selected rows through
the one pass predicate. ``report`` renders the report.
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

from neural_decompiler import attention_paths as ap
from neural_decompiler import attention_patterns as atp
from neural_decompiler import block_concentration as bc
from neural_decompiler import block_routing as br
from neural_decompiler import cue_decompilation as cd
from neural_decompiler import encoding_read as er
from neural_decompiler import frame_channels as fch
from neural_decompiler import head_pattern as hp
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import neuron_feature as nf
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-021"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")
PHASES = rc.PHASES
LOCK_011_REQUIRED_KEYS = ("axes_vectors", "read_weight", "sigma_T", "denominators", "confirmation_011_sha256", "content_sha256")
LOCK_012_REQUIRED_KEYS = ("axes_vectors", "read_weight", "sigma_T", "base_states", "defined_templates", "confirmation_012_sha256", "lock_011_sha256", "content_sha256")
LOCK_017_REQUIRED_KEYS = ("locked_states", "bases_3", "confirmation_017_sha256", "lock_016_sha256", "content_sha256")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the frozen inputs, the program blob and Experiment 020's closure without a model"),
                             ("calibrate", "once: re-materialize 020's exposed table, the gate, the screen, the draws and the floors (exposed prompts only)"),
                             ("lock", "write the Y1 prediction table and the floor tables (no forward pass)"),
                             ("confirm", "once: stage 1, the Y2 row, the barrier, stage 2 and the scoring"),
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


def _run_contract_test() -> dict[str, Any]:
    environment = dict(os.environ, NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE="1")
    completed = subprocess.run([sys.executable, "-m", "pytest", *CONTRACT_TEST], cwd=ROOT, env=environment, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    return {"returncode": completed.returncode, "passed": completed.returncode == 0, "output_tail": output[-4000:], "output_sha256": pm.sha256_text(output)}


def runtime_record(spec: ModelSpec) -> dict[str, Any]:
    runtime = validate_runtime(spec)
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": rd.RUNTIME_SEED, "control_seed": rd.CONTROL_SEED,
            "deterministic_algorithms": spec.deterministic_algorithms, "torch_num_threads": int(torch.get_num_threads())}


class pytest_free_guard:
    """Disable every capture and intervention entry point for the duration of a block: the lock phase's own proof
    that its predictions cannot reach a forward pass. Independent of the test framework."""

    _NAMES = ("capture_prompt", "run_patched", "run_capture", "run_interventions")

    def __enter__(self):
        self._saved = {name: getattr(pm, name) for name in self._NAMES if hasattr(pm, name)}

        def refuse(*args, **kwargs):
            raise rd.PhaseError("the lock phase reached a forward pass")

        for name in self._saved:
            setattr(pm, name, refuse)
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(pm, name, value)
        return False


def _load_lock(path: Path, experiment: str) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("experiment") != experiment or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise rd.PhaseError(f"the Experiment {experiment} lock's content digest does not verify")
    return lock


@dataclass
class Runner:
    root: Path = ROOT
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    lock_011_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "011")  # tests inject locks built from the fake
    lock_012_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "012")
    lock_017_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "017")
    git_state: Callable[[], Mapping[str, Any]] = collect_git_state
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    changed_paths: Callable[[str], list[str] | None] = _git_changed_paths
    contract_runner: Callable[[], Mapping[str, Any]] = _run_contract_test
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))

    @property
    def output_dir(self) -> Path:
        return self.results_path.parent

    @property
    def candidate_calibration_path(self) -> Path:
        return self.output_dir / "candidate-calibration.json"

    @property
    def table_path(self) -> Path:
        return self.output_dir / "exposed-table.pt"

    @property
    def draws_path(self) -> Path:
        return self.output_dir / "draw-values.pt"

    @property
    def stage2_path(self) -> Path:
        return self.output_dir / "stage2-tables.pt"

    # -- inputs ---------------------------------------------------------------

    def _base_inputs(self):
        """Experiment 020's input chain, unchanged: every confirmation set 006–019 and the 011/012/017 locks."""
        manifest, manifest_sha256, extension = pm.load_inputs(self.root)
        c006 = cd.load_confirmation(self.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
        if c006.content_sha256 != ss.INHERITED_CONFIRMATION_SHA256:
            raise rd.PhaseError("the Experiment 006 confirmation set digest is not the frozen constant")
        c009 = ht.load_confirmation(self.root / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, c006)
        c011 = er.load_confirmation(self.root / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, c006, c009)
        for relative in (cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, er.CONFIRMATION_RELATIVE_PATH, rd.EXPERIMENT_011_LOCK_PATH, lc.CONFIRMATION_RELATIVE_PATH,
                         rd.EXPERIMENT_012_LOCK_PATH, hp.EXPERIMENT_013_CONFIRMATION_PATH, hp.EXPERIMENT_014_CONFIRMATION_PATH, hp.EXPERIMENT_015_CONFIRMATION_PATH,
                         hp.EXPERIMENT_016_CONFIRMATION_PATH, bc.EXPERIMENT_017_CONFIRMATION_PATH, rd.EXPERIMENT_017_LOCK_PATH, br.EXPERIMENT_018_CONFIRMATION_PATH,
                         rd.EXPERIMENT_019_CONFIRMATION_PATH, rd.CONFIRMATION_RELATIVE_PATH, rc.CLOSURE_020_RELATIVE_PATH, rc.EXTRACT_020_RELATIVE_PATH):
            if not (self.root / relative).exists() or not self.tracked(self.root / relative):
                raise rd.PhaseError(f"{relative} must exist and be tracked and committed")
        digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": c006.content_sha256, "confirmation_009": c009.content_sha256,
                   "confirmation_011": c011.content_sha256}
        lock_011 = self.lock_011_loader(self.root / rd.EXPERIMENT_011_LOCK_PATH)
        if any(key not in lock_011 for key in LOCK_011_REQUIRED_KEYS) or lock_011["confirmation_011_sha256"] != c011.content_sha256:
            raise rd.PhaseError("the Experiment 011 lock does not carry the locked axes and read weight for the frozen 011 confirmation set")
        digests["lock_011"] = lock_011["content_sha256"]
        pool_012 = lc.build_pool_012(manifest, extension, c006, c009, c011)
        c012 = lc.load_confirmation(self.root / lc.CONFIRMATION_RELATIVE_PATH, pool_012, digests)
        digests["confirmation_012"] = c012.content_sha256
        lock_012 = self.lock_012_loader(self.root / rd.EXPERIMENT_012_LOCK_PATH)
        if any(key not in lock_012 for key in LOCK_012_REQUIRED_KEYS) or lock_012["confirmation_012_sha256"] != c012.content_sha256 or lock_012["lock_011_sha256"] != lock_011["content_sha256"]:
            raise rd.PhaseError("the Experiment 012 lock does not carry the locked bases for the frozen 012 confirmation set and the 011 lock")
        digests["lock_012"] = lock_012["content_sha256"]
        c013 = ap.load_confirmation(self.root / hp.EXPERIMENT_013_CONFIRMATION_PATH, ap.build_pool_013(manifest, extension, c006, c009, c011, c012), digests)
        digests["confirmation_013"] = c013.content_sha256
        c014 = nf.load_confirmation(self.root / hp.EXPERIMENT_014_CONFIRMATION_PATH, nf.build_pool_014(manifest, extension, c006, c009, c011, c012, c013), digests)
        digests["confirmation_014"] = c014.content_sha256
        c015 = atp.load_confirmation(self.root / hp.EXPERIMENT_015_CONFIRMATION_PATH, atp.build_pool_015(manifest, extension, c006, c009, c011, c012, c013, c014), digests)
        digests["confirmation_015"] = c015.content_sha256
        c016 = fch.load_confirmation(self.root / hp.EXPERIMENT_016_CONFIRMATION_PATH, fch.build_pool_016(manifest, extension, c006, c009, c011, c012, c013, c014, c015), digests)
        digests["confirmation_016"] = c016.content_sha256
        pool_017 = hp.build_pool_017(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016)
        c017 = hp.load_confirmation(self.root / bc.EXPERIMENT_017_CONFIRMATION_PATH, pool_017, digests)
        digests["confirmation_017"] = c017.content_sha256
        lock_017 = self.lock_017_loader(self.root / rd.EXPERIMENT_017_LOCK_PATH)
        if any(key not in lock_017 for key in LOCK_017_REQUIRED_KEYS) or lock_017["confirmation_017_sha256"] != c017.content_sha256:
            raise rd.PhaseError("the Experiment 017 lock does not carry the locked layer-3 bases for the frozen 017 confirmation set")
        digests["lock_017"] = lock_017["content_sha256"]
        rc.assert_lock_digests(digests)  # the exact inherited objects, bound into every 021 state, record and lock
        pool_018 = bc.build_pool_018(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017)
        c018 = bc.load_confirmation(self.root / br.EXPERIMENT_018_CONFIRMATION_PATH, pool_018, digests)
        digests["confirmation_018"] = c018.content_sha256
        pool_019 = br.build_pool_019(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017, c018)
        c019 = br.load_confirmation(self.root / rd.EXPERIMENT_019_CONFIRMATION_PATH, pool_019, digests)
        digests["confirmation_019"] = c019.content_sha256
        pool = rd.build_pool_020(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017, c018, c019)
        return pool, lock_011, lock_012, lock_017, digests

    def _inputs(self):
        pool, lock_011, lock_012, lock_017, digests = self._base_inputs()
        rc.assert_program_blob()
        confirmation = rd.load_confirmation(self.root / rd.CONFIRMATION_RELATIVE_PATH, pool, digests)
        closure = rc.verify_020_closure(self.root, confirmation)
        digests = dict(digests) | dict(closure["digests"])
        return pool, lock_011, lock_012, lock_017, confirmation, digests, closure

    # -- non-scientific phase -------------------------------------------------

    def validate(self) -> int:
        try:
            pool, lock_011, lock_012, lock_017, confirmation, digests, closure = self._inputs()
            pools = rc.production_pools(pool)
            rc.assert_no_manifest_key(closure["state"]["executed_prompt_keys"], confirmation, "Experiment 020's ledger")
        except (rd.PhaseError, br.PhaseError, bc.PhaseError, ValueError, KeyError) as error:
            self.log(f"validation failed: {error}")
            return 1
        classes = rd.manifest_classes(confirmation)
        self.log(f"program blob {rc.PROGRAM_BLOB_SHA1}; confirmation {confirmation.content_sha256} ({', '.join(f'{name} {len(keys)}' for name, keys in classes.items())}); "
                 f"Experiment 020 closed: closure {digests['closure_020'][:12]}…, extract {digests['extract_020'][:12]}…, results {digests['results_020_state'][:12]}…, "
                 f"ledger {len(closure['ledger'])} keys, 0 manifest keys")
        self.log(f"pools: cues {', '.join(f'{c} {len(v)}' for c, v in pools.cues.items())}; frames before the screen {', '.join(f'{t} {len(v)}' for t, v in pools.frames_unscreened.items())}; "
                 f"nouns {', '.join(f'{r} {len(v)}' for r, v in pools.nouns.items())}; rows Y2 {len(rc.y2_rows())}, Y3 {len(rc.y3_rows())}")
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
                raise rd.PhaseError("scientific execution requires a clean Git tree at a committed SHA")
            if rc.state_digests(state) != rc.digest_tuple(digests) or dict(state["versions"]) != provenance["versions"]:
                raise rd.PhaseError("frozen inputs or dependency versions differ from the recorded run")
            if phase == "calibrate":
                self._reconcile_unrecorded_attempt(state)
        else:
            if phase != "calibrate":
                raise rd.PhaseError(f"{phase} requires an existing results state from calibrate")
            state = rc.new_results_state(digests=digests, **provenance)
        rc.assert_phase_allowed(phase, state)
        return state

    def _write(self, state: Mapping[str, Any]) -> str:
        return rd.write_results_state(self.results_path, state)

    def _reconcile_unrecorded_attempt(self, state: dict[str, Any]) -> None:
        """An attempt that ended without writing its incident (a kill, an out-of-memory stop, a second interrupt)
        is recorded now as an incident of its own commit, so the phase can resume only at a new commit."""
        phase = state["phases"]["calibrate"]
        attempts = list(phase.get("attempt_commits", []))
        recorded = {entry["commit"] for entry in state["calibration"].get("incidents", [])}
        if phase.get("status") == "running" and attempts and attempts[-1] not in recorded and not state["calibration"].get("record_sha256"):
            state["calibration"].setdefault("incidents", []).append({"message": f"the calibrate attempt at {attempts[-1]} ended without recording an incident (killed or interrupted)",
                                                                     "type": "UnrecordedTermination", "at": pm.utc_now(), "phase": "calibrate", "commit": attempts[-1]})
            self._write(state)
            self.log(f"recorded the unrecorded end of the calibrate attempt at {attempts[-1]}")

    def _recheck_020(self, confirmation) -> dict[str, Any]:
        """Experiment 020's closure, extract and results state, verified again after a phase (or an incident)."""
        try:
            rc.verify_020_closure(self.root, confirmation)
        except Exception as error:  # recorded, never allowed to hide the incident being recorded
            return {"ok": False, "message": str(error)}
        return {"ok": True}

    def _record_incident(self, state: dict[str, Any], phase: str, error: BaseException, confirmation=None) -> None:
        commit = str(state.get("phases", {}).get(phase, {}).get("commit") or state.get("phases", {}).get(phase, {}).get("confirm_commit") or state.get("protocol_code_commit") or "")
        entry = {"message": str(error) or type(error).__name__, "type": type(error).__name__, "at": pm.utc_now(), "phase": phase, "commit": commit}
        if phase == "calibrate":
            state["calibration"] = rc.json_safe(state["calibration"])
            state["calibration"].setdefault("incidents", []).append(entry)
        else:
            state["confirmation"] = rc.json_safe(state.get("confirmation") or {})
            state["confirmation"]["incident"] = entry  # the same object the re-check result is added to below
        self._write(state)  # the incident is on disk before anything slow runs
        self.log(f"INCIDENT ({phase}): {error}")
        if confirmation is not None:
            entry["closure_020_recheck"] = self._recheck_020(confirmation)
            self._write(state)

    def _check_runtime(self, closure: Mapping[str, Any]) -> dict[str, Any]:
        """The runtime and the dependency versions of Experiment 020's explore: the re-captured states must reproduce."""
        runtime = runtime_record(PYTHIA_70M)
        recorded = closure["state"]["phases"]["explore"].get("runtime") or {}
        if {k: v for k, v in runtime.items() if k != "seed"} != {k: v for k, v in recorded.items() if k != "seed"}:
            raise rd.PhaseError(f"the runtime {runtime} differs from Experiment 020's explore runtime {recorded}")
        if dict(self.versions()) != dict(closure["state"]["versions"]):
            raise rd.PhaseError(f"the dependency versions {dict(self.versions())} differ from Experiment 020's {closure['state']['versions']}")
        return runtime

    def calibrate(self) -> int:
        pool, lock_011, lock_012, lock_017, confirmation, digests, closure = self._inputs()
        state = self._state_for("calibrate", digests)
        commit = self._provenance()["protocol_code_commit"]
        incidents = state["calibration"].get("incidents", [])
        if any(entry["commit"] == commit for entry in incidents):
            raise rd.PhaseError("a calibrate incident is recorded at this commit; a committed fix is required before calibrate runs again")
        if commit in state["phases"]["calibrate"].get("attempt_commits", []):
            raise rd.PhaseError("calibrate was already attempted at this commit; a new commit is required")
        runtime = self._check_runtime(closure)
        rc.assert_no_manifest_key(closure["state"]["executed_prompt_keys"], confirmation, "Experiment 020's ledger")
        contract = dict(self.contract_runner())
        state["calibration"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            self._write(state)
            self.log("A0 contract test failed; refusing to calibrate")
            return 1
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
                try:
                    manifest_keys = frozenset(prompt.key for prompt in confirmation.all_prompts)
                    table, context = rc.rematerialize(model, pool, lock_011=lock_011, lock_012=lock_012, lock_017=lock_017, exploration_020=closure["exploration"],
                                                      ledger_020=closure["ledger"], manifest_keys=manifest_keys, confirmation=confirmation, log=self.log, executed=executed)
                finally:
                    pm.record_execution(state, executed, pool.nouns)
                    self._write(state)
                calibration.update({"environment": context["environment"], "identities": context["identities"], "identity_tolerances": rd.identity_tolerances(),
                                    "level1_worst": context["level1_worst"]})
                self._write(state)  # every maximum reaches disk before any of them can stop the phase
                if set(state["executed_prompt_keys"]) != set(closure["ledger"]):
                    raise pm.IncidentError("the re-materialization did not execute exactly Experiment 020's ledger")
                rd.enforce_all(context["identities"])
                gate = rc.reproduction_gate(table, closure["exploration"])
                calibration["gate"] = gate
                self._write(state)
                rc.enforce_gate(gate)
                self.log(f"reproduction gate passed: max difference {gate['max_difference']:.1e} over {gate['n_pairs']} pairs × {gate['n_nouns']} nouns")
                screen = rc.validity_screen(context["program"], context["states"], context["nouns"], context["plural"], context["axis_T"])
                pools = rc.production_pools(pool).screened({frame_id: bool(verdict["valid"]) for frame_id, verdict in screen.items()})
                precondition = rc.screen_precondition(pools)
                table_digests = table.digests()
                self.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(table.tensors(), self.table_path)
                calibration.update({"screen": screen, "precondition": precondition, "pools": pools.to_json(), "table_sha256": table_digests})
                if not precondition["ok"]:
                    state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "stopped_for_review", "stopped_at": pm.utc_now()}
                    self._write(state)
                    self.log(f"calibration precondition failed {precondition['counts']}: no floor is computed; stopped for review")
                    return 3
                self._write(state)
                result = rc.run_calibration(table, pools, log=self.log)
                arrays = result["arrays"]
                array_digests = {outcome: {key: rc.tensor_digest(values) for key, values in entries.items()} for outcome, entries in arrays.items()}
                torch.save(arrays, self.draws_path)
                rematerialization = {"n_reference_captures": len(pool.frames), "n_cue_prompts": int(table.measured.shape[0]), "executed_keys": len(state["executed_prompt_keys"]),
                                     "ledger_equal_020": True, "environment": context["environment"], "identities": context["identities"],
                                     "identity_tolerances": rd.identity_tolerances(), "level1_worst": context["level1_worst"]}
                record = rc.calibration_record(body=result["body"], run_id=state["run_id"], protocol_code_commit=commit, digests=digests, pools=pools, screen=screen,
                                               precondition=precondition, rematerialization=rematerialization, gate=gate, table_digests=table_digests, array_digests=array_digests)
                rd.assert_fresh_nouns_absent(record, confirmation)
                rc.verify_020_closure(self.root, confirmation)  # Experiment 020's files are unchanged by this phase
                rd.assert_fresh_nouns_absent(state["calibration"], confirmation)
                text = pm.canonical_json(record) + "\n"
                self.candidate_calibration_path.write_text(text, encoding="utf-8")
                calibration.update({"record_sha256": pm.sha256_text(text), "record_content_sha256": record["content_sha256"], "candidate_path": str(self.candidate_calibration_path),
                                    "draw_arrays_sha256": array_digests, "cross_check": result["body"]["cross_check"],
                                    "record_summary": {"rows": {outcome: len(entries) for outcome, entries in record["rows"].items()},
                                                       "clamped": {outcome: sorted({name for entry in entries for name, flag in entry["clamped"].items() if flag}) for outcome, entries in record["rows"].items()},
                                                       "joint_pass_rate_all_outcomes": record["full_rows"]["joint_pass_rate_all_outcomes"]}})
                self._write(state)  # the record's digest reaches disk with the record; nothing after this can fail the phase
            except rc.CrossCheckError as error:
                calibration["cross_check"] = error.details
                self._record_incident(state, "calibrate", error, confirmation)
                return 2
            except rc.EnvironmentIncident as error:
                calibration["environment"] = error.environment
                self._record_incident(state, "calibrate", error, confirmation)
                return 2
            except pm.IncidentError as error:
                self._record_incident(state, "calibrate", error, confirmation)
                return 2
            except BaseException as error:  # a protocol failure or an interruption: recorded, then raised
                self._record_incident(state, "calibrate", error, confirmation)
                raise
        finally:
            if model is not None:
                del model
            gc.collect()
        state["phases"]["calibrate"] = {**state["phases"]["calibrate"], "status": "complete", "completed_at": pm.utc_now()}
        digest = self._write(state)
        self.log(f"calibrate complete: candidate record {self.candidate_calibration_path} (sha256 {calibration['record_sha256']}); results sha256 {digest}. "
                 f"Install it byte-identical as {rc.CALIBRATION_RELATIVE_PATH}, commit it, and stop for the floor review before lock.")
        return 0

    def _weights_only(self):
        """The lock phase's only access to the model: its parameters. No prompt runs and no capture API is reachable."""
        model = self.model_loader(PYTHIA_70M)
        try:
            weights = pm.Weights.from_model(model)
            lw = lc.LayerWeights.from_model(model, layers=rd.PROGRAM_LAYERS)
            programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in rd.PROGRAM_LAYERS}
            return weights, lw, programs, rd.attention_bias_sum(model)
        finally:
            del model
            gc.collect()

    def _prediction_rows(self, weights, lw, programs, source: Mapping[str, Any], lock_011, lock_012, lock_017, confirmation, pool, *, bias_sum=None):
        program = rd.ReadoutProgram(lw, {layer: programs[layer] for layer in rd.READOUT_LAYERS}, weights.ln_final_w.double(), weights.ln_final_b.double(), float(weights.eps), bias_sum)
        chain = rd.chain_from_locks(lock_011, lock_012, lock_017, lw, programs, pool)
        nouns = rd.NounSet.build(weights, pool.nouns, confirmation.nouns)
        frames_by_id = {frame.frame_id: frame for frame in pool.frames}
        states = {frame_id: rd.state_from_locked(entry, frames_by_id[frame_id]) for frame_id, entry in source["locked_states"].items()}
        rows16 = {frame_id: atp.reference_rows(programs, state.state_017.x1_all, state.state_017.x2_all) for frame_id, state in states.items()}
        bases = {template: {int(layer): torch.tensor(vector, dtype=torch.float64) for layer, vector in entry.items()} for template, entry in source["template_bases"].items()}
        axis_T = torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64)
        return rd.prediction_rows(program, chain, weights, states, rows16, nouns, list(pool.frames), list(confirmation.tokens), bases, axis_T), nouns

    def _installed_record(self, state: Mapping[str, Any], digests: Mapping[str, str]) -> tuple[dict[str, Any], str]:
        path = self.root / rc.CALIBRATION_RELATIVE_PATH
        if not path.exists() or not self.tracked(path):
            raise rd.PhaseError(f"install the candidate calibration record byte-identical as {rc.CALIBRATION_RELATIVE_PATH} and commit it before lock")
        sha = rc.file_sha256(path)
        if sha != state["calibration"].get("record_sha256"):
            raise rd.PhaseError("the installed calibration record is not the candidate this run wrote")
        record = json.loads(path.read_text(encoding="utf-8"))
        rc.verify_calibration_record(record)
        rc.assert_record_inputs(record, digests)
        return record, sha

    def lock(self) -> int:
        pool, lock_011, lock_012, lock_017, confirmation, digests, closure = self._inputs()
        state = self._state_for("lock", digests)
        rc.assert_no_manifest_key(closure["state"]["executed_prompt_keys"], confirmation, "Experiment 020's ledger")
        rc.assert_no_manifest_key(state["executed_prompt_keys"], confirmation, "Experiment 021's ledger")
        changed = self.changed_paths(state["phases"]["calibrate"]["commit"])
        if changed is None:
            raise rd.PhaseError("the calibrate commit is not an ancestor of the current commit; lock must be written at the calibrated protocol")
        scientific = rc.scientific_changes(changed)
        if scientific:
            raise rd.PhaseError(f"scientific paths changed since calibrate: {scientific}; lock must be written at the calibrated protocol")
        record, record_sha = self._installed_record(state, digests)
        self._check_runtime(closure)  # the locked rows must reproduce exactly at confirm, on 020's runtime
        weights, lw, programs, bias_sum = self._weights_only()
        rows, nouns = self._prediction_rows(weights, lw, programs, closure["exploration"], lock_011, lock_012, lock_017, confirmation, pool, bias_sum=bias_sum)
        with pytest_free_guard():  # the provenance invariant, executed rather than asserted
            again, _ = self._prediction_rows(weights, lw, programs, closure["exploration"], lock_011, lock_012, lock_017, confirmation, pool, bias_sum=bias_sum)
        provenance = max((atp._max_numeric_difference(dict(b), dict(a), "provenance") for a, b in zip(rows, again)), default=0.0)
        rd.enforce("provenance invariant", provenance, rd.PROVENANCE_TOLERANCE)
        noun_keys = [nouns.nouns[index].lexical_key for index in nouns.exposed_scorable]
        lock = rc.build_candidate_lock(run_id=state["run_id"], protocol_code_commit=self._provenance()["protocol_code_commit"], digests=digests, confirmation=confirmation, rows=rows,
                                       noun_keys=noun_keys, exploration_020=closure["exploration"], record=record, record_file_sha256=record_sha)
        candidate_path, predictions_path = self.output_dir / "candidate-lock.json", self.output_dir / "candidate-predictions.md"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        predictions_text = rc.render_predictions(lock)
        predictions_path.write_text(predictions_text, encoding="utf-8")
        recheck = self._recheck_020(confirmation)
        if not recheck["ok"]:
            raise rd.PhaseError(f"Experiment 020's closure no longer verifies after lock: {recheck['message']}")
        state["lock"] = {"candidate_path": str(candidate_path), "predictions_path": str(predictions_path), "content_sha256": lock["content_sha256"], "provenance_difference": provenance,
                         "predictions_sha256": pm.sha256_text(predictions_text), "written_at": pm.utc_now(), "n_rows": len(rows), "n_nouns": len(noun_keys),
                         "calibration_content_sha256": record["content_sha256"]}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now()}
        self._write(state)
        self.log(f"candidate lock written to {candidate_path} (sha256 {lock['content_sha256']}, {len(rows)} rows) and {predictions_path}. Install both as "
                 f"{rc.LOCK_RELATIVE_PATH} and {rc.PREDICTIONS_RELATIVE_PATH}, commit, and stop for the sign-off before confirm.")
        return 0

    def confirm(self) -> int:
        pool, lock_011, lock_012, lock_017, confirmation, digests, closure = self._inputs()
        state = self._state_for("confirm", digests)
        rc.assert_no_manifest_key(closure["state"]["executed_prompt_keys"], confirmation, "Experiment 020's ledger")
        rc.assert_no_manifest_key(state["executed_prompt_keys"], confirmation, "Experiment 021's ledger")
        lock_path, predictions_path = self.root / rc.LOCK_RELATIVE_PATH, self.root / rc.PREDICTIONS_RELATIVE_PATH
        if not lock_path.exists() or not predictions_path.exists():
            raise rd.PhaseError("missing committed lock or predictions artifact")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        record, record_sha = self._installed_record(state, digests)
        git = self.git_state()
        rc.validate_lock(lock, state=state, digests=digests, confirmation=confirmation, record=record, record_file_sha256=record_sha,
                         predictions_text=predictions_path.read_text(encoding="utf-8"), git_state=git, tracked=self.tracked(lock_path) and self.tracked(predictions_path),
                         changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        commit = str(git.get("commit"))
        self._check_runtime(closure)
        seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            weights = pm.Weights.from_model(model)
            lw = lc.LayerWeights.from_model(model, layers=rd.PROGRAM_LAYERS)
            programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in rd.PROGRAM_LAYERS}
            rows, _ = self._prediction_rows(weights, lw, programs, lock, lock_011, lock_012, lock_017, confirmation, pool, bias_sum=rd.attention_bias_sum(model))
            reproduced = rd.assert_lock_predictions_reproduced(lock, rows)  # PhaseError before anything fresh runs
            state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": commit,
                                          "lock_predictions_reproduced_max_difference": reproduced}
            pm.record_execution(state, confirmation.stage1_prompts, pool.nouns)
            self._write(state)
            try:
                stage1 = rd.stage_one(model, pool, confirmation, lock, lock_011, lock_012, lock_017, protocol_code_commit=commit, log=self.log)
                selection = rc.select_y2_row(stage1["frames"], lock["floor_tables"])  # the validity verdicts only
                stage1["y2_selection"] = selection
                stage1["y2_selection_sha256"] = rc.selection_digest(selection, stage1["digest"])
                state["confirmation"] = {"stage1": stage1, "tokens_meta": [dict(token) for token in confirmation.tokens]}
                rd.assert_no_target_prompt_executed(state, confirmation)
                self._write(state)
                # Hard boundary: the digested stage-1 record and the Y2 row are re-read from disk before any fresh cue prompt runs.
                state = rd.load_results_state(self.results_path)
                rd.assert_stage_one_digest(state["confirmation"]["stage1"])
                rc.assert_y2_selection(state["confirmation"]["stage1"], lock["floor_tables"])
                rd.assert_no_target_prompt_executed(state, confirmation)
                valid = [frame_id for frame_id, entry in state["confirmation"]["stage1"]["frames"].items() if entry["valid"]]
                stage1_reproduced = rd.reproduce_stage_one_rows(model, pool, confirmation, lock, lock_011, lock_012, lock_017, state["confirmation"]["stage1"])
                state["phases"]["confirm"]["stage1_rows_reproduced_max_difference"] = stage1_reproduced
                rd.assert_no_target_prompt_executed(state, confirmation)
                self.log(f"stage 1 digested: {len(valid)} valid fresh frames, composition {selection['composition']} → Y2 row {selection['row']}; no fresh cue prompt has run; stage 2 begins")
                targets = tuple(confirmation.exposed_frame_prompts) + tuple(prompt for prompt in confirmation.token_prompts if prompt.frame.frame_id in valid)
                pm.record_execution(state, targets, pool.nouns)
                self._write(state)
                measured = rc.stage_two_021(model, pool, confirmation, lock, lock_011, lock_012, lock_017, state["confirmation"]["stage1"], log=self.log, enforce=False)
                saved = {name: {part: ({"cues": list(value.cues), "frames": list(value.frames), "templates": list(value.templates), "noun_keys": list(value.noun_keys),
                                        "measured": value.measured, "predicted": value.predicted} if isinstance(value, rd.ScoringTable) else value)
                                for part, value in entry.items()} for name, entry in measured["tables"].items()}
                torch.save(saved, self.stage2_path)
                state["confirmation"]["stage2"] = {"path": str(self.stage2_path), "identities": rc.json_safe(measured["identities"]),
                                                   "tables_sha256": {name: {part: {key: rc.tensor_digest(value[key]) for key in ("measured", "predicted")} if isinstance(value, dict) else
                                                                           (rc.tensor_digest(value) if isinstance(value, torch.Tensor) else None) for part, value in entry.items()}
                                                                     for name, entry in saved.items()}}
                self._write(state)  # every fresh measurement is on disk before any check or score can stop the phase
                rd.enforce_all(measured["identities"])
                results = rc.score_021(state["confirmation"]["stage1"], measured, lock)
                results["identities"] = measured["identities"]
                results["noun_keys"] = measured["noun_keys"]
                results["fresh_noun_keys"] = measured["fresh_noun_keys"]
                recheck = self._recheck_020(confirmation)
                if not recheck["ok"]:
                    raise pm.IncidentError(f"Experiment 020's closure no longer verifies after confirm: {recheck['message']}")
            except pm.IncidentError as error:
                self._record_incident(state, "confirm", error, confirmation)
                return 2
            except BaseException as error:  # a protocol failure or an interruption: recorded, then raised
                self._record_incident(state, "confirm", error, confirmation)
                raise
        finally:
            del model
            gc.collect()
        state["confirmation"] = {**state["confirmation"], **results, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}
        state["phases"]["confirm"] = {**state["phases"]["confirm"], "status": "complete", "completed_at": pm.utc_now()}
        digest = self._write(state)
        self.log(f"confirm complete: {results['outcome']['label']}; results sha256 {digest}")
        return 0

    def _draw_values(self, state: Mapping[str, Any]) -> dict[str, dict[str, list[list[float | None]]]] | None:
        """The per-row draw values, verified against the digests the calibration recorded."""
        if not self.draws_path.exists():
            return None
        arrays = torch.load(self.draws_path)
        record_path = self.root / rc.CALIBRATION_RELATIVE_PATH
        source = self.candidate_calibration_path if not record_path.exists() else record_path
        record = json.loads(source.read_text(encoding="utf-8"))
        out = {}
        for outcome, entries in arrays.items():
            out[outcome] = {}
            for key, values in entries.items():
                if rc.tensor_digest(values) != record["draw_arrays_sha256"][outcome][key]:
                    raise rd.PhaseError(f"the draw values of {outcome} row {key} do not match the calibration record")
                out[outcome][key] = [[value if value == value and abs(value) != float("inf") else None for value in row] for row in values.tolist()]
        return out

    def report(self) -> int:
        self._inputs()
        state = rd.load_results_state(self.results_path)
        rc.assert_phase_allowed("report", state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        record_path = self.root / rc.CALIBRATION_RELATIVE_PATH
        source = record_path if record_path.exists() else self.candidate_calibration_path
        record = json.loads(source.read_text(encoding="utf-8")) if source.exists() else None
        self.report_path.write_text(rc.render_report(state, self._draw_values(state), record), encoding="utf-8")
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
