#!/usr/bin/env python3
"""Run Experiment 019 through isolated phases.

``validate`` checks the frozen inputs without a model. ``freeze-confirmation`` builds ``confirmation-v1.json`` from
tokenizer rules (refuses to overwrite). ``explore`` (once) captures the 90 exposed frames' reference states (checked
against the Experiment 018 lock and its stage-1 digests), computes every prospective selector — the matched global
``S'``, the template ``T``, the drive-quantile tables and the full evaluation ``E`` and the operating-point-plus-drive
rule ``G`` of every exposed frame and position, the controls — from the locked states, the weights and the exposed
cues' ΔE before any pair is measured, checks I9 against Experiment 018's recorded per-frame lists, re-measures the
11 796 licensed pairs with the layer-3 residuals, the head's row and the block-2 input captured, replicates the 018
extract, checks I1–I10, computes the exposed oracles and records every rung's exposed statistics. ``lock`` computes the
prediction table for the fresh tokens in the 90 exposed frames from the weights, the locks, the locked states and
lists — no forward pass, no capture or intervention call path — and writes the candidate lock and the prediction
artifact. ``confirm`` (once) validates the artifacts, reproduces the locked selectors and every locked prediction, then
runs stage 1 (the fresh frames' reference and cue-pair prompts only, validity, their own E and G lists, the
frame-conditional prediction table, serialized and digested into the results state) and, only after re-reading that
digest from disk, stage 2 (every fresh cue in both sets, the oracles from the measurements alone, scoring). ``report``
renders the report.
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
from neural_decompiler import read_assembly as ra
from neural_decompiler import supervised_subspace as ss
from neural_decompiler.models import PYTHIA_70M, ModelSpec, load_model, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiment-019"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
CONTRACT_TEST = ("tests/test_pythia_bridge_contract.py", "-m", "pythia_smoke", "-q")
PHASES = ("validate", "freeze-confirmation", "explore", "lock", "confirm", "report")
LOCK_011_REQUIRED_KEYS = ("axes_vectors", "read_weight", "sigma_T", "denominators", "confirmation_011_sha256", "content_sha256")
LOCK_012_REQUIRED_KEYS = ("axes_vectors", "read_weight", "sigma_T", "base_states", "defined_templates", "confirmation_012_sha256", "lock_011_sha256", "content_sha256")
LOCK_REQUIRED_KEYS = {"013": ("locked_states", "confirmation_013_sha256", "lock_012_sha256", "content_sha256"), "014": ("locked_states", "confirmation_014_sha256", "lock_013_sha256", "content_sha256"),
                      "015": ("locked_states", "confirmation_015_sha256", "lock_014_sha256", "content_sha256"), "016": ("locked_states", "confirmation_016_sha256", "lock_015_sha256", "content_sha256"),
                      "017": ("locked_states", "bases_3", "confirmation_017_sha256", "lock_016_sha256", "content_sha256"),
                      "018": ("locked_states", "bases_3", "base2_pt", "scores", "subsets", "frame_subsets", "confirmation_018_sha256", "lock_017_sha256", "content_sha256")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, help_text in (("validate", "check the frozen inputs and the confirmation set without a model"), ("freeze-confirmation", "build confirmation-v1.json from tokenizer rules only (refuses to overwrite)"),
                             ("explore", "Tier A: reference states, the selectors (before any measurement), re-measurement, replication, identities, the exposed oracles and statistics"), ("lock", "compute every prospective rung's prediction table for fresh tokens × exposed frames; write the candidate lock and predictions"),
                             ("confirm", "Tier C: stage 1 (fresh frames' reference states, their E and G lists, digested predictions) then stage 2 (fresh cues, the oracles, scoring)"), ("report", "render outputs/experiment-019/report.md")):
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
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": br.RUNTIME_SEED, "control_seed": br.CONTROL_SEED, "deterministic_algorithms": spec.deterministic_algorithms,
            "torch_num_threads": int(torch.get_num_threads())}


def _load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)


def _load_lock(path: Path, experiment: str) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("experiment") != experiment or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise br.PhaseError(f"the Experiment {experiment} lock's content digest does not verify")
    return lock


@dataclass
class Runner:
    root: Path = ROOT
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    tokenizer_loader: Callable[[ModelSpec], Any] = _load_tokenizer
    lock_011_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "011")  # tests inject locks built from the fake
    lock_012_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "012")
    lock_013_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "013")
    lock_014_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "014")
    lock_015_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "015")
    lock_016_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "016")
    lock_017_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "017")
    lock_018_loader: Callable[[Path], dict[str, Any]] = lambda path: _load_lock(path, "018")
    git_state: Callable[[], Mapping[str, Any]] = field(default_factory=lambda: lambda: collect_git_state(ROOT))
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    changed_paths: Callable[[str], list[str] | None] = _git_changed_paths
    contract_runner: Callable[[], Mapping[str, Any]] = _run_contract_test
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))

    @property
    def confirmation_path(self) -> Path:
        return self.root / br.CONFIRMATION_RELATIVE_PATH

    # -- inputs ---------------------------------------------------------------

    def _base_inputs(self):
        manifest, manifest_sha256, extension = pm.load_inputs(self.root)
        confirmation_006 = cd.load_confirmation(self.root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
        if confirmation_006.content_sha256 != ss.INHERITED_CONFIRMATION_SHA256:
            raise br.PhaseError("the Experiment 006 confirmation set digest is not the frozen constant")
        confirmation_009 = ht.load_confirmation(self.root / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006)
        confirmation_011 = er.load_confirmation(self.root / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
        for relative in (cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, er.CONFIRMATION_RELATIVE_PATH, hp.EXPERIMENT_011_LOCK_PATH, lc.CONFIRMATION_RELATIVE_PATH, hp.EXPERIMENT_012_LOCK_PATH,
                         hp.EXPERIMENT_013_CONFIRMATION_PATH, hp.EXPERIMENT_013_LOCK_PATH, hp.EXPERIMENT_014_CONFIRMATION_PATH, hp.EXPERIMENT_014_LOCK_PATH, hp.EXPERIMENT_015_CONFIRMATION_PATH, hp.EXPERIMENT_015_LOCK_PATH,
                         hp.EXPERIMENT_016_CONFIRMATION_PATH, hp.EXPERIMENT_016_LOCK_PATH, bc.EXPERIMENT_017_CONFIRMATION_PATH, bc.EXPERIMENT_017_LOCK_PATH, br.EXPERIMENT_018_CONFIRMATION_PATH, br.EXPERIMENT_018_LOCK_PATH,
                         br.INHERITED_018_EXTRACT_RELATIVE_PATH):  # the 016 and 017 extracts are bound by the 017 and 018 locks and not read here
            if not (self.root / relative).exists() or not self.tracked(self.root / relative):
                raise br.PhaseError(f"{relative} must exist and be tracked and committed")
        lock_011 = self.lock_011_loader(self.root / hp.EXPERIMENT_011_LOCK_PATH)
        if any(key not in lock_011 for key in LOCK_011_REQUIRED_KEYS) or lock_011["confirmation_011_sha256"] != confirmation_011.content_sha256:
            raise br.PhaseError("the Experiment 011 lock does not carry the locked axes and read weight for the frozen 011 confirmation set")
        digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": confirmation_006.content_sha256, "confirmation_009": confirmation_009.content_sha256,
                   "confirmation_011": confirmation_011.content_sha256, "lock_011": lock_011["content_sha256"]}
        pool_010 = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
        pool_012 = lc.build_pool_012(manifest, extension, confirmation_006, confirmation_009, confirmation_011)
        confirmation_012 = lc.load_confirmation(self.root / lc.CONFIRMATION_RELATIVE_PATH, pool_012, digests)
        digests["confirmation_012"] = confirmation_012.content_sha256
        lock_012 = self.lock_012_loader(self.root / hp.EXPERIMENT_012_LOCK_PATH)
        if any(key not in lock_012 for key in LOCK_012_REQUIRED_KEYS) or lock_012["confirmation_012_sha256"] != confirmation_012.content_sha256 or lock_012["lock_011_sha256"] != lock_011["content_sha256"]:
            raise br.PhaseError("the Experiment 012 lock does not carry the frozen model for the frozen 012 confirmation set and the 011 lock")
        digests["lock_012"] = lock_012["content_sha256"]
        pool_013 = ap.build_pool_013(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012)
        confirmation_013 = ap.load_confirmation(self.root / hp.EXPERIMENT_013_CONFIRMATION_PATH, pool_013, digests)
        digests["confirmation_013"] = confirmation_013.content_sha256
        lock_013 = self.lock_013_loader(self.root / hp.EXPERIMENT_013_LOCK_PATH)
        self._check_chain_lock(lock_013, "013", confirmation_013, lock_012)
        digests["lock_013"] = lock_013["content_sha256"]
        pool_014 = nf.build_pool_014(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013)
        confirmation_014 = nf.load_confirmation(self.root / hp.EXPERIMENT_014_CONFIRMATION_PATH, pool_014, digests)
        digests["confirmation_014"] = confirmation_014.content_sha256
        lock_014 = self.lock_014_loader(self.root / hp.EXPERIMENT_014_LOCK_PATH)
        self._check_chain_lock(lock_014, "014", confirmation_014, lock_013)
        digests["lock_014"] = lock_014["content_sha256"]
        pool_015 = atp.build_pool_015(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014)
        confirmation_015 = atp.load_confirmation(self.root / hp.EXPERIMENT_015_CONFIRMATION_PATH, pool_015, digests)
        digests["confirmation_015"] = confirmation_015.content_sha256
        lock_015 = self.lock_015_loader(self.root / hp.EXPERIMENT_015_LOCK_PATH)
        self._check_chain_lock(lock_015, "015", confirmation_015, lock_014)
        digests["lock_015"] = lock_015["content_sha256"]
        pool_016 = fch.build_pool_016(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015)
        confirmation_016 = fch.load_confirmation(self.root / hp.EXPERIMENT_016_CONFIRMATION_PATH, pool_016, digests)
        digests["confirmation_016"] = confirmation_016.content_sha256
        lock_016 = self.lock_016_loader(self.root / hp.EXPERIMENT_016_LOCK_PATH)
        self._check_chain_lock(lock_016, "016", confirmation_016, lock_015)
        digests["lock_016"] = lock_016["content_sha256"]
        pool_017 = hp.build_pool_017(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016)
        confirmation_017 = hp.load_confirmation(self.root / bc.EXPERIMENT_017_CONFIRMATION_PATH, pool_017, digests)
        digests["confirmation_017"] = confirmation_017.content_sha256
        lock_017 = self.lock_017_loader(self.root / bc.EXPERIMENT_017_LOCK_PATH)
        self._check_chain_lock(lock_017, "017", confirmation_017, lock_016)
        digests["lock_017"] = lock_017["content_sha256"]
        pool_018 = bc.build_pool_018(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016, confirmation_017)
        confirmation_018 = bc.load_confirmation(self.root / br.EXPERIMENT_018_CONFIRMATION_PATH, pool_018, digests)
        digests["confirmation_018"] = confirmation_018.content_sha256
        lock_018 = self.lock_018_loader(self.root / br.EXPERIMENT_018_LOCK_PATH)
        self._check_chain_lock(lock_018, "018", confirmation_018, lock_017)
        digests["lock_018"] = lock_018["content_sha256"]
        pool = br.build_pool_019(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016, confirmation_017, confirmation_018)
        extract = br.load_inherited_extract_018(self.root / br.INHERITED_018_EXTRACT_RELATIVE_PATH, digests=digests, expected_size=br.EXPECTED_EXTRACT_SIZE_018)
        digests["extract_018"] = extract["content_sha256"]
        return pool, pool_010, lock_011, lock_012, lock_018, extract, digests

    @staticmethod
    def _check_chain_lock(lock: Mapping[str, Any], experiment: str, confirmation: Any, previous: Mapping[str, Any]) -> None:
        keys = LOCK_REQUIRED_KEYS[experiment]
        previous_field = f"lock_{int(experiment) - 1:03d}_sha256"
        if any(key not in lock for key in keys) or lock[f"confirmation_{experiment}_sha256"] != confirmation.content_sha256 or lock[previous_field] != previous["content_sha256"]:
            raise br.PhaseError(f"the Experiment {experiment} lock does not carry the locked record for the frozen {experiment} confirmation set and the previous lock")

    def _inputs(self):
        pool, pool_010, lock_011, lock_012, lock_018, extract, digests = self._base_inputs()
        if not self.confirmation_path.exists() or not self.tracked(self.confirmation_path):
            raise br.PhaseError("the Experiment 019 confirmation set must be frozen and committed before any scientific phase")
        confirmation = br.load_confirmation(self.confirmation_path, pool, digests)
        digests = dict(digests) | {"confirmation_019": confirmation.content_sha256}
        return pool, pool_010, lock_011, lock_012, lock_018, extract, confirmation, digests

    def validate(self) -> int:
        try:
            pool, pool_010, lock_011, lock_012, lock_018, extract, digests = self._base_inputs()
        except (br.PhaseError, bc.PhaseError, ValueError) as error:
            self.log(f"validation failed: {error}")
            return 1
        self.log(f"exposed pool: {len(pool.frames)} frames, {len(pool.tokens)} tokens; 018 extract {len(extract['entries'])} pairs with {len(extract['stage1_state_digests'])} stage-1 state digests (source run {extract['source']['run_id']}); 018 lock {digests['lock_018'][:12]}…")
        if not self.confirmation_path.exists():
            self.log("confirmation set not frozen yet: run freeze-confirmation")
            return 1
        try:
            confirmation = br.load_confirmation(self.confirmation_path, pool, digests)
        except ValueError as error:
            self.log(f"validation failed: {error}")
            return 1
        tracked = self.tracked(self.confirmation_path)
        self.log(f"confirmation ok: sha256 {confirmation.content_sha256}; {len(confirmation.tokens)} tokens {[t['word'] for t in confirmation.tokens]}; {len(confirmation.frames)} fresh frames; {len(confirmation.token_prompts)} fresh-frame prompts + {len(confirmation.exposed_frame_prompts)} exposed-frame prompts; tracked={tracked}")
        return 0 if tracked else 1

    def freeze_confirmation(self) -> int:
        pool, pool_010, lock_011, lock_012, lock_018, extract, digests = self._base_inputs()
        if self.confirmation_path.exists():
            self.log(f"refusing to overwrite frozen confirmation set {self.confirmation_path}")
            return 1
        digest = br.freeze_confirmation(self.confirmation_path, self.tokenizer_loader(PYTHIA_70M), pool, digests)
        self.log(f"froze confirmation set {self.confirmation_path} sha256 {digest}; commit it before any scientific phase")
        return 0

    # -- scientific phases ---------------------------------------------------

    def _provenance(self) -> dict[str, Any]:
        git = self.git_state()
        return {"protocol_code_commit": str(git.get("commit") or ""), "git_dirty": bool(git.get("dirty", True)), "versions": dict(self.versions())}

    def _state_for(self, phase: str, digests: Mapping[str, str]) -> dict[str, Any]:
        provenance = self._provenance()
        if self.results_path.exists():
            state = br.load_results_state(self.results_path)
            if provenance["git_dirty"] or not pm._COMMIT_SHA.fullmatch(provenance["protocol_code_commit"]):
                raise br.PhaseError("scientific execution requires a clean Git tree at a committed SHA")
            if br.state_digests(state) != br.digest_tuple(digests) or dict(state["versions"]) != provenance["versions"]:
                raise br.PhaseError("frozen inputs or dependency versions differ from the recorded run")
        else:
            if phase != "explore":
                raise br.PhaseError(f"{phase} requires an existing results state from explore")
            state = br.new_results_state(digests=digests, **provenance)
        br.assert_phase_allowed(phase, state)
        return state

    def _record_incident(self, state: dict[str, Any], phase: str, error: Exception) -> None:
        entry = {"message": str(error), "at": pm.utc_now(), "phase": phase, "commit": self._provenance()["protocol_code_commit"]}
        if phase == "explore":
            state["exploration"].setdefault("incidents", []).append(entry)
        else:
            state["confirmation"] = {**(state.get("confirmation") or {}), "incident": entry}
        br.write_results_state(self.results_path, state)
        self.log(f"INCIDENT ({phase}): {error}")

    def explore(self) -> int:
        pool, pool_010, lock_011, lock_012, lock_018, extract, confirmation, digests = self._inputs()
        state = self._state_for("explore", digests)
        commit = self._provenance()["protocol_code_commit"]
        incidents = state["exploration"].get("incidents", [])
        if incidents and incidents[-1]["commit"] == commit:
            raise br.PhaseError("an explore incident is recorded at this commit; a committed fix or documented amendment is required before explore runs again")
        contract = dict(self.contract_runner())
        state["exploration"]["a0_contract_test"] = contract
        if not contract.get("passed"):
            br.write_results_state(self.results_path, state)
            self.log("A0 contract test failed; refusing to explore")
            return 1
        state["protocol_code_commit"] = commit
        state["phases"]["explore"] = {"status": "running", "started_at": pm.utc_now(), "runtime": runtime_record(PYTHIA_70M), "attempts": int(state["phases"]["explore"].get("attempts", 0)) + 1,
                                      "attempt_commits": list(state["phases"]["explore"].get("attempt_commits", [])) + [commit]}
        pm.record_execution(state, pm.manifest_prompts(pool.frames) + tuple(pool.reference_prompt(frame) for frame in pool.frames), pool.nouns)
        br.assert_confirmation_untouched(state, confirmation)
        br.write_results_state(self.results_path, state)
        seed_runtime(br.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            try:
                br.run_exploration(model, pool, pool_010, lock_011=lock_011, lock_012=lock_012, lock_018=lock_018, inherited_extract=extract, confirmation_token_ids={token["token_id"] for token in confirmation.tokens},
                                   state=state, results_path=self.results_path, log=self.log)
            except pm.IncidentError as error:
                self._record_incident(state, "explore", error)
                return 2
            except Exception as error:
                self._record_incident(state, "explore", error)
                raise
        finally:
            del model
            gc.collect()
        br.assert_confirmation_untouched(state, confirmation)
        state["phases"]["explore"] = {**state["phases"]["explore"], "status": "complete", "completed_at": pm.utc_now()}
        digest = br.write_results_state(self.results_path, state)
        summary = state["exploration"]["summary"]
        f = cd._f
        self.log(f"explore complete: exposed κ_c_L at k=64 — S' {f(summary['kappa_global'], 3)}, T {f(summary['kappa_template'], 3)}, E {f(summary['kappa_full'], 3)}, G {f(summary['kappa_rule'], 3)}, O* {f(summary['kappa_witness'], 3)}; ρ {f(summary['rho'], 3)}; reference rung c_L R² {f(summary['reference_c_L_r2'], 3)}; results sha256 {digest}")
        return 0

    def _weights_only(self) -> tuple[pm.Weights, lc.LayerWeights, dict[int, atp.LayerProgram]]:
        """The lock phase's only access to the model: its parameters. No prompt is run and no capture API is reachable from here."""
        model = self.model_loader(PYTHIA_70M)
        try:
            return pm.Weights.from_model(model), lc.LayerWeights.from_model(model), {layer: atp.LayerProgram.from_model(model, layer) for layer in br.PROGRAM_LAYERS}
        finally:
            del model
            gc.collect()

    def _predictions(self, weights: pm.Weights, lw: lc.LayerWeights, programs: Mapping[int, atp.LayerProgram], source: Mapping[str, Any], lock_012: Mapping[str, Any], confirmation: br.Confirmation019, pool: Any) -> dict[str, Any]:
        """``source`` is the exploration record at lock time and the lock itself at confirm time: it carries the axes, the read weight, the layer-3 bases, the p_t base, the locked states and the selector lists."""
        chain = br.chain_from_source(source, lock_012, lw, programs, pool)
        return br.lock_predictions(chain, weights, source["locked_states"], source["selectors"]["lists"], pool, confirmation, source["defined_templates"])

    def lock(self) -> int:
        pool, pool_010, lock_011, lock_012, lock_018, extract, confirmation, digests = self._inputs()
        state = self._state_for("lock", digests)
        changed = self.changed_paths(state["protocol_code_commit"])
        if changed is None:
            raise br.PhaseError("the explore commit is not an ancestor of the current commit; lock must be written at the explore protocol")
        scientific = [path for path in changed if path.startswith(br.SCIENTIFIC_PATH_PREFIXES) and path not in br.NON_SCIENTIFIC_PATHS and not path.startswith(br.NON_SCIENTIFIC_PREFIXES)]
        if scientific:
            raise br.PhaseError(f"scientific paths changed since explore: {scientific}; lock must be written at the explore protocol")
        weights, lw, programs = self._weights_only()
        predictions = self._predictions(weights, lw, programs, state["exploration"], lock_012, confirmation, pool)
        lock = br.build_candidate_lock(state=state, digests=digests, confirmation=confirmation, predictions=predictions, protocol_code_commit=self._provenance()["protocol_code_commit"])
        candidate_path = self.results_path.parent / "candidate-lock.json"
        predictions_path = self.results_path.parent / "candidate-predictions.md"
        candidate_path.write_text(pm.canonical_json(lock) + "\n", encoding="utf-8")
        predictions_text = br.render_predictions(lock)
        predictions_path.write_text(predictions_text, encoding="utf-8")
        state["lock"] = {"candidate_path": str(candidate_path), "predictions_path": str(predictions_path), "content_sha256": lock["content_sha256"], "predictions_sha256": pm.sha256_text(predictions_text), "written_at": pm.utc_now()}
        state["phases"]["lock"] = {"status": "complete", "completed_at": pm.utc_now()}
        br.write_results_state(self.results_path, state)
        self.log(f"candidate lock written to {candidate_path} (sha256 {lock['content_sha256']}) and {predictions_path} (sha256 {state['lock']['predictions_sha256']}). Install both as {br.LOCK_RELATIVE_PATH} and {br.PREDICTIONS_RELATIVE_PATH} and commit before confirm.")
        return 0

    def confirm(self) -> int:
        pool, pool_010, lock_011, lock_012, lock_018, extract, confirmation, digests = self._inputs()
        state = self._state_for("confirm", digests)
        lock_path, predictions_path = self.root / br.LOCK_RELATIVE_PATH, self.root / br.PREDICTIONS_RELATIVE_PATH
        if not lock_path.exists() or not predictions_path.exists():
            raise br.PhaseError("missing committed lock or predictions artifact")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        git = self.git_state()
        br.validate_lock(lock, state=state, digests=digests, confirmation=confirmation, predictions_text=predictions_path.read_text(encoding="utf-8"), git_state=git,
                         tracked=self.tracked(lock_path) and self.tracked(predictions_path), changed_paths=self.changed_paths(lock["protocol_code_commit"]), inherited_subsets=lock_018["subsets"])
        commit = str(git.get("commit"))
        runtime = runtime_record(PYTHIA_70M)
        recorded_runtime = state["phases"]["explore"].get("runtime")
        if recorded_runtime is not None and {k: v for k, v in runtime.items() if k != "seed"} != {k: v for k, v in recorded_runtime.items() if k != "seed"}:
            raise br.PhaseError(f"the confirm runtime {runtime} differs from the explore runtime {recorded_runtime}; the re-captured reference states must be bitwise reproducible")
        seed_runtime(br.RUNTIME_SEED, PYTHIA_70M.deterministic_algorithms)
        model = self.model_loader(PYTHIA_70M)
        try:
            weights, lw, programs = pm.Weights.from_model(model), lc.LayerWeights.from_model(model), {layer: atp.LayerProgram.from_model(model, layer) for layer in br.PROGRAM_LAYERS}
            chain = br.chain_from_source(lock, lock_012, lw, programs, pool)
            licensed = br.licensed_pool(extract["entries"], pool)
            recomputed = br.selectors_from_states(chain, weights, bc.read_of_outputs(chain.hcm.fcm.read, lw), lock["locked_states"], pool.frames, bc.tokens_by_template(licensed), {f"{w}|{fid}" for w, fid, _, _ in licensed}, lock_018["subsets"])
            selectors_difference = br.assert_selectors_reproduced(lock, recomputed)  # PhaseError before anything fresh runs
            reproduced = br.assert_lock_predictions_reproduced(lock, self._predictions(weights, lw, programs, lock, lock_012, confirmation, pool))
            state["phases"]["confirm"] = {"status": "running", "started_at": pm.utc_now(), "lock_sha256": lock["content_sha256"], "confirm_commit": commit, "lock_predictions_reproduced_max_difference": reproduced, "selectors_reproduced_max_difference": selectors_difference}
            defined_frames = [frame for frame in confirmation.frames if frame.template_id in lock["defined_templates"]]
            pm.record_execution(state, tuple(confirmation.reference_prompt(frame) for frame in defined_frames) + tuple(prompt for prompt in confirmation.frame_prompts() if prompt.frame.template_id in lock["defined_templates"]), pool.nouns)
            br.write_results_state(self.results_path, state)
            try:
                stage1 = br.stage_one(model, pool, pool_010, confirmation, lock, lock_012, extract, protocol_code_commit=commit, log=self.log)
                state["confirmation"] = {"stage1": stage1, "tokens_meta": [dict(token) for token in confirmation.tokens]}
                br.assert_no_target_pair_executed(state, confirmation)
                br.write_results_state(self.results_path, state)
                # Hard boundary: the digested stage-1 record is re-read from disk before any fresh cue prompt runs.
                state = br.load_results_state(self.results_path)
                br.assert_stage_one_digest(state["confirmation"]["stage1"])
                br.assert_no_target_pair_executed(state, confirmation)
                self.log(f"stage 1 digested: {len(state['confirmation']['stage1']['rows'])} rows, digest {state['confirmation']['stage1']['digest'][:16]}…; no fresh cue prompt has run; stage 2 begins")
                fresh_prompts = tuple(prompt for prompt in confirmation.token_prompts if prompt.frame.template_id in lock["defined_templates"] and state["confirmation"]["stage1"]["frames"][prompt.frame.frame_id]["valid"])
                exposed_prompts = tuple(prompt for prompt in confirmation.exposed_frame_prompts if prompt.frame.template_id in lock["defined_templates"])
                pm.record_execution(state, fresh_prompts + exposed_prompts, pool.nouns)
                br.write_results_state(self.results_path, state)
                results = br.stage_two(model, pool, pool_010, confirmation, lock, lock_012, state["confirmation"]["stage1"], log=self.log)
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
        digest = br.write_results_state(self.results_path, state)
        self.log(f"confirm complete: {results['outcome']['label']}; results sha256 {digest}")
        return 0

    def report(self) -> int:
        self._inputs()
        state = br.load_results_state(self.results_path)
        br.assert_phase_allowed("report", state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(br.render_report(state), encoding="utf-8")
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
