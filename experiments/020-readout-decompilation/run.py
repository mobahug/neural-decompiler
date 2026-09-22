#!/usr/bin/env python3
"""Run Experiment 020 through isolated phases.

``validate`` checks the frozen inputs and the confirmation set without a model. ``freeze-confirmation`` builds
``confirmation-v1.json`` from tokenizer rules (refuses to overwrite). ``explore`` (once) captures the 108 exposed
frames' reference states — one forward each — measures every exposed cue in them, runs the decoded chain's predicted
layer-3 input change through the Experiment 020 readout program to the singular-versus-plural contrast, checks the
identities (readout, Level 1 with the norm-normalized ``E₁``, the inherited Experiment 017 reproduction), records the
comparators and the exposed statistics, and computes nothing whatever of a fresh noun. ``lock`` computes the Y1
prediction table for the 24 fresh cues in the 108 exposed frames from the weights, the inherited locks and the locked
reference states — no forward pass, no capture or intervention call path — and writes the candidate lock and the
predictions artifact. ``confirm`` (once) validates the artifacts, reproduces every locked prediction, then runs stage 1
(each fresh frame's S1-REF and S1-VALIDITY prompts only, its validity, its reference state and the frame-conditional
prediction table, serialized and digested into the results state) and, only after re-reading that digest from disk,
stage 2 (the S2-TARGET prompts of the Y1 block and of the valid fresh frames' Y2 block, then the scoring on the frozen
populations). ``report`` renders the report. ``diagnose`` is not a phase: it never opens the results state and it
enforces no identity — it measures a template-complete subset of the *exposed* pool, records each pair's Level-1
breakdown, and writes ``level1-diagnostic.json``, so that a Level-1 failure can be placed.
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
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-020"
RESULTS_PATH = OUTPUT_DIR / "results.json"
DIAGNOSTIC_PATH = OUTPUT_DIR / "level1-diagnostic.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")
PHASES = rd.PHASES
LOCK_011_REQUIRED_KEYS = ("axes_vectors", "read_weight", "sigma_T", "denominators", "confirmation_011_sha256", "content_sha256")
LOCK_012_REQUIRED_KEYS = ("axes_vectors", "read_weight", "sigma_T", "base_states", "defined_templates", "confirmation_012_sha256", "lock_011_sha256", "content_sha256")
LOCK_017_REQUIRED_KEYS = ("locked_states", "bases_3", "confirmation_017_sha256", "lock_016_sha256", "content_sha256")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the frozen inputs and the confirmation set without a model"),
                             ("freeze-confirmation", "build confirmation-v1.json from tokenizer rules (once)"),
                             ("explore", "Tier A: the exposed pool's reference states, measurements, predictions, identities and comparators"),
                             ("lock", "write the Y1 prediction table and the candidate lock (no forward pass)"),
                             ("confirm", "Tier C: stage 1 (the fresh frames' reference states, digested) then stage 2 (the fresh cues, the scoring)"),
                             ("report", "render the report from the results state")):
        phases.add_parser(phase, help=help_text)
    # Not a phase: it writes its own record, never the results state, and it enforces no identity.
    diagnose = phases.add_parser("diagnose", help="exposed-only Level-1 diagnostic: place a Level-1 identity failure without enforcing anything")
    diagnose.add_argument("--frames-per-template", type=int, default=1, help="exposed frames per template, taken in pool order (default 1)")
    diagnose.add_argument("--cues", type=int, default=0, help="exposed cues per frame, 0 for every one of them (default 0)")
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


def _load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


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
    diagnostic_path: Path = DIAGNOSTIC_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    tokenizer_loader: Callable[[ModelSpec], Any] = _load_tokenizer
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
    def confirmation_path(self) -> Path:
        return self.root / rd.CONFIRMATION_RELATIVE_PATH

    # -- inputs ---------------------------------------------------------------

    def _base_inputs(self):
        manifest, manifest_sha256, extension = pm.load_inputs(self.root)
        c006 = cd.load_confirmation(self.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
        if c006.content_sha256 != ss.INHERITED_CONFIRMATION_SHA256:
            raise rd.PhaseError("the Experiment 006 confirmation set digest is not the frozen constant")
        c009 = ht.load_confirmation(self.root / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, c006)
        c011 = er.load_confirmation(self.root / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, c006, c009)
        for relative in (cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, er.CONFIRMATION_RELATIVE_PATH, rd.EXPERIMENT_011_LOCK_PATH, lc.CONFIRMATION_RELATIVE_PATH,
                         rd.EXPERIMENT_012_LOCK_PATH, hp.EXPERIMENT_013_CONFIRMATION_PATH, hp.EXPERIMENT_014_CONFIRMATION_PATH, hp.EXPERIMENT_015_CONFIRMATION_PATH,
                         hp.EXPERIMENT_016_CONFIRMATION_PATH, bc.EXPERIMENT_017_CONFIRMATION_PATH, rd.EXPERIMENT_017_LOCK_PATH, br.EXPERIMENT_018_CONFIRMATION_PATH,
                         rd.EXPERIMENT_019_CONFIRMATION_PATH):
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
        if not self.confirmation_path.exists() or not self.tracked(self.confirmation_path):
            raise rd.PhaseError("the Experiment 020 confirmation set must be frozen and committed before any scientific phase")
        confirmation = rd.load_confirmation(self.confirmation_path, pool, digests)
        digests = dict(digests) | {"confirmation_020": confirmation.content_sha256}
        return pool, lock_011, lock_012, lock_017, confirmation, digests

    # -- non-scientific phases ------------------------------------------------

    def validate(self) -> int:
        try:
            pool, lock_011, lock_012, lock_017, digests = self._base_inputs()
        except (rd.PhaseError, br.PhaseError, bc.PhaseError, ValueError) as error:
            self.log(f"validation failed: {error}")
            return 1
        scorable = sum(1 for noun in pool.nouns if noun.single_token)
        self.log(f"exposed pool: {len(pool.frames)} frames, {len(pool.tokens)} cue tokens, {len(pool.nouns)} nouns ({scorable} scorable); locks 011 {lock_011['content_sha256'][:12]}…, 012 {lock_012['content_sha256'][:12]}…, 017 {lock_017['content_sha256'][:12]}…")
        if not self.confirmation_path.exists():
            self.log("confirmation set not frozen yet: run freeze-confirmation")
            return 1
        try:
            confirmation = rd.load_confirmation(self.confirmation_path, pool, digests)
        except ValueError as error:
            self.log(f"validation failed: {error}")
            return 1
        classes = rd.manifest_classes(confirmation)
        tracked = self.tracked(self.confirmation_path)
        self.log(f"confirmation ok: sha256 {confirmation.content_sha256}; {len(confirmation.tokens)} cues {[t['word'] for t in confirmation.tokens]}; {len(confirmation.frames)} fresh frames; "
                 f"{len(confirmation.nouns)} fresh nouns {[n.lexical_key for n in confirmation.nouns]}; manifest " + ", ".join(f"{name} {len(classes[name])}" for name in rd.MANIFEST_CLASSES) + f"; tracked={tracked}")
        return 0 if tracked else 1

    def freeze_confirmation(self) -> int:
        pool, lock_011, lock_012, lock_017, digests = self._base_inputs()
        if self.confirmation_path.exists():
            self.log(f"refusing to overwrite frozen confirmation set {self.confirmation_path}")
            return 1
        digest = rd.freeze_confirmation(self.confirmation_path, self.tokenizer_loader(PYTHIA_70M), pool, digests)
        self.log(f"froze confirmation set {self.confirmation_path} sha256 {digest}; commit it before any scientific phase")
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
            if rd.state_digests(state) != rd.digest_tuple(digests) or dict(state["versions"]) != provenance["versions"]:
                raise rd.PhaseError("frozen inputs or dependency versions differ from the recorded run")
        else:
            if phase != "explore":
                raise rd.PhaseError(f"{phase} requires an existing results state from explore")
            state = rd.new_results_state(digests={key: digests[key] for key in rd.CONFIRMATION_DIGEST_KEYS} | {"confirmation_020": digests["confirmation_020"]}, **provenance)
        rd.assert_phase_allowed(phase, state)
        return state

    def _record_incident(self, state: dict[str, Any], phase: str, error: Exception) -> None:
        entry = {"message": str(error), "at": pm.utc_now(), "phase": phase, "commit": self._provenance()["protocol_code_commit"]}
        if phase == "explore":
            state["exploration"].setdefault("incidents", []).append(entry)
        else:
            state["confirmation"] = {**(state.get("confirmation") or {}), "incident": entry}
        rd.write_results_state(self.results_path, state)
        self.log(f"INCIDENT ({phase}): {error}")

    def explore(self) -> int:
        pool, lock_011, lock_012, lock_017, confirmation, digests = self._inputs()
        state = self._state_for("explore", digests)
        commit = self._provenance()["protocol_code_commit"]
        incidents = state["exploration"].get("incidents", [])
        if incidents and incidents[-1]["commit"] == commit:
            raise rd.PhaseError("an explore incident is recorded at this commit; a committed fix or documented amendment is required before explore runs again")
        contract = dict(self.contract_runner())
        state["exploration"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            rd.write_results_state(self.results_path, state)
            self.log("A0 contract test failed; refusing to explore")
            return 1
        state["protocol_code_commit"] = commit
        state["phases"]["explore"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime_record(PYTHIA_70M), "attempts": int(state["phases"]["explore"].get("attempts", 0)) + 1,
                                      "attempt_commits": list(state["phases"]["explore"].get("attempt_commits", [])) + [commit]}
        rd.assert_confirmation_untouched(state, confirmation)
        rd.write_results_state(self.results_path, state)
        seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            try:
                rd.run_exploration(model, pool, lock_011=lock_011, lock_012=lock_012, lock_017=lock_017, confirmation=confirmation, state=state, results_path=self.results_path, log=self.log)
            except pm.IncidentError as error:
                self._record_incident(state, "explore", error)
                return 2
            except Exception as error:
                self._record_incident(state, "explore", error)
                raise
        finally:
            del model
            gc.collect()
        rd.assert_confirmation_untouched(state, confirmation)
        rd.assert_fresh_nouns_absent(state["exploration"], confirmation)
        state["phases"]["explore"] = {**state["phases"]["explore"], "status": "complete", "completed_at": pm.utc_now()}
        digest = rd.write_results_state(self.results_path, state)
        self.log(f"explore complete: {state['exploration']['n_pairs']} exposed pairs; results sha256 {digest}")
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

    def lock(self) -> int:
        pool, lock_011, lock_012, lock_017, confirmation, digests = self._inputs()
        state = self._state_for("lock", digests)
        changed = self.changed_paths(state["protocol_code_commit"])
        if changed is None:
            raise rd.PhaseError("the explore commit is not an ancestor of the current commit; lock must be written at the explore protocol")
        scientific = [path for path in changed if path.startswith(rd.SCIENTIFIC_PATH_PREFIXES) and path not in rd.NON_SCIENTIFIC_PATHS and not path.startswith(rd.NON_SCIENTIFIC_PREFIXES)]
        if scientific:
            raise rd.PhaseError(f"scientific paths changed since explore: {scientific}; lock must be written at the explore protocol")
        weights, lw, programs, bias_sum = self._weights_only()
        rows, nouns = self._prediction_rows(weights, lw, programs, state["exploration"], lock_011, lock_012, lock_017, confirmation, pool, bias_sum=bias_sum)
        # the provenance invariant, executed rather than asserted: the same rows with every capture entry point disabled
        guard = pytest_free_guard()
        with guard:
            again, _ = self._prediction_rows(weights, lw, programs, state["exploration"], lock_011, lock_012, lock_017, confirmation, pool, bias_sum=bias_sum)
        provenance = max((atp._max_numeric_difference(dict(b), dict(a), "provenance") for a, b in zip(rows, again)), default=0.0)
        rd.enforce("provenance invariant", provenance, rd.PROVENANCE_TOLERANCE)
        noun_keys = [nouns.nouns[index].lexical_key for index in nouns.exposed_scorable]
        lock = rd.build_candidate_lock(state=state, digests=digests, confirmation=confirmation, rows=rows, noun_keys=noun_keys, protocol_code_commit=self._provenance()["protocol_code_commit"])
        candidate_path = self.results_path.parent / "candidate-lock.json"
        predictions_path = self.results_path.parent / "candidate-predictions.md"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        predictions_text = rd.render_predictions(lock)
        predictions_path.write_text(predictions_text, encoding="utf-8")
        state["lock"] = {"candidate_path": str(candidate_path), "predictions_path": str(predictions_path), "content_sha256": lock["content_sha256"], "provenance_difference": provenance,
                         "predictions_sha256": pm.sha256_text(predictions_text), "written_at": pm.utc_now(), "n_rows": len(rows), "n_nouns": len(noun_keys)}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now()}
        rd.write_results_state(self.results_path, state)
        self.log(f"candidate lock written to {candidate_path} (sha256 {lock['content_sha256']}, {len(rows)} rows × {len(noun_keys)} nouns) and {predictions_path}. Install both as "
                 f"{rd.LOCK_RELATIVE_PATH} and {rd.PREDICTIONS_RELATIVE_PATH} and commit before confirm.")
        return 0

    def confirm(self) -> int:
        pool, lock_011, lock_012, lock_017, confirmation, digests = self._inputs()
        state = self._state_for("confirm", digests)
        lock_path, predictions_path = self.root / rd.LOCK_RELATIVE_PATH, self.root / rd.PREDICTIONS_RELATIVE_PATH
        if not lock_path.exists() or not predictions_path.exists():
            raise rd.PhaseError("missing committed lock or predictions artifact")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        git = self.git_state()
        rd.validate_lock(lock, state=state, digests=digests, confirmation=confirmation, predictions_text=predictions_path.read_text(encoding="utf-8"), git_state=git,
                         tracked=self.tracked(lock_path) and self.tracked(predictions_path), changed_paths=self.changed_paths(lock["protocol_code_commit"]))
        commit = str(git.get("commit"))
        runtime = runtime_record(PYTHIA_70M)
        recorded_runtime = state["phases"]["explore"].get("runtime")
        if recorded_runtime is not None and {k: v for k, v in runtime.items() if k != "seed"} != {k: v for k, v in recorded_runtime.items() if k != "seed"}:
            raise rd.PhaseError(f"the confirm runtime {runtime} differs from the explore runtime {recorded_runtime}; the re-captured reference states must be bitwise reproducible")
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
            rd.write_results_state(self.results_path, state)
            try:
                stage1 = rd.stage_one(model, pool, confirmation, lock, lock_011, lock_012, lock_017, protocol_code_commit=commit, log=self.log)
                state["confirmation"] = {"stage1": stage1, "tokens_meta": [dict(token) for token in confirmation.tokens]}
                rd.assert_no_target_prompt_executed(state, confirmation)
                rd.write_results_state(self.results_path, state)
                # Hard boundary: the digested stage-1 record is re-read from disk before any fresh cue prompt runs.
                state = rd.load_results_state(self.results_path)
                rd.assert_stage_one_digest(state["confirmation"]["stage1"])
                rd.assert_no_target_prompt_executed(state, confirmation)
                valid = [frame_id for frame_id, entry in state["confirmation"]["stage1"]["frames"].items() if entry["valid"]]
                stage1_reproduced = rd.reproduce_stage_one_rows(model, pool, confirmation, lock, lock_011, lock_012, lock_017, state["confirmation"]["stage1"])
                state["phases"]["confirm"]["stage1_rows_reproduced_max_difference"] = stage1_reproduced
                rd.assert_no_target_prompt_executed(state, confirmation)
                self.log(f"stage 1 digested: {len(state['confirmation']['stage1']['rows'])} rows, digest {state['confirmation']['stage1']['digest'][:16]}…; {len(valid)} valid fresh frames; "
                         f"every row recomputed from the digested states with difference {stage1_reproduced:.1e}; no fresh cue prompt has run; stage 2 begins")
                targets = tuple(confirmation.exposed_frame_prompts) + tuple(prompt for prompt in confirmation.token_prompts if prompt.frame.frame_id in valid)
                pm.record_execution(state, targets, pool.nouns)
                rd.write_results_state(self.results_path, state)
                measured = rd.stage_two(model, pool, confirmation, lock, lock_011, lock_012, lock_017, state["confirmation"]["stage1"], log=self.log)
                results = rd.score_confirmation(state["confirmation"]["stage1"], measured["tables"], lock, measured["dw_fresh"])
                results["identities"] = measured["identities"]
                results["noun_keys"] = measured["noun_keys"]
                results["fresh_noun_keys"] = measured["fresh_noun_keys"]
            except pm.IncidentError as error:
                self._record_incident(state, "confirm", error)
                return 2
            except Exception as error:
                self._record_incident(state, "confirm", error)
                raise
        finally:
            del model
            gc.collect()
        state["confirmation"] = {**state["confirmation"], **results, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}
        state["phases"]["confirm"] = {**state["phases"]["confirm"], "status": "complete", "completed_at": pm.utc_now()}
        digest = rd.write_results_state(self.results_path, state)
        self.log(f"confirm complete: {results['outcome']['label']}; results sha256 {digest}")
        return 0

    def diagnose(self, *, frames_per_template: int = 1, cues: int | None = None) -> int:
        """The Level-1 diagnostic. Not a phase: it never opens the results state, so the recorded run, its ledger and
        its incident are untouched, and it enforces no identity — the failure under investigation must be measured, not
        re-raised. Exposed frames, exposed cues and exposed scorable nouns only."""
        pool, lock_011, lock_012, lock_017, confirmation, digests = self._inputs()
        provenance = self._provenance()
        seed_runtime(rd.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            record = rd.run_level1_diagnostic(model, pool, lock_011=lock_011, lock_012=lock_012, lock_017=lock_017, confirmation=confirmation,
                                              frames_per_template=frames_per_template, cues_per_frame=cues, log=self.log)
        finally:
            del model
            gc.collect()
        recorded = rd.load_results_state(self.results_path) if self.results_path.exists() else None
        record = {**record, "protocol_code_commit": provenance["protocol_code_commit"], "git_dirty": provenance["git_dirty"],
                  "versions": provenance["versions"], "runtime": runtime_record(PYTHIA_70M),
                  "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
                  "created_at": pm.utc_now(),
                  "investigating": {"run_id": recorded["run_id"], "incidents": recorded["exploration"].get("incidents", [])} if recorded else None}
        self.diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**record, "content_sha256": pm.sha256_text(pm.canonical_json(record))}
        self.diagnostic_path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
        self.log(f"diagnostic written to {self.diagnostic_path} sha256 {payload['content_sha256']}")
        if self.results_path.exists() and rd.load_results_state(self.results_path)["state_sha256"] != (recorded or {}).get("state_sha256"):
            raise rd.PhaseError("the diagnostic changed the recorded results state")
        return 0

    def report(self) -> int:
        self._inputs()
        state = rd.load_results_state(self.results_path)
        rd.assert_phase_allowed("report", state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(rd.render_report(state), encoding="utf-8")
        self.log(f"report written to {self.report_path}")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    if args.phase == "diagnose":
        return runner.diagnose(frames_per_template=int(args.frames_per_template), cues=int(args.cues) or None)
    if args.phase in PHASES:
        return getattr(runner, args.phase.replace("-", "_"))()
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
