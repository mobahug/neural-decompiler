#!/usr/bin/env python3
"""Run the frozen behavior candidate screen through four isolated phases.

``validate`` proves manifest integrity without model output. ``behavioral``
performs the one allowed behavioral screen on pinned Pythia-70M and repeats it
on pinned Pythia-160M only when zero candidates pass. ``compactness`` probes
only behavioral passers with development data. ``report`` renders the single
Markdown matrix and, after finalist audits, records the zero/one-target
decision. No flag selects a candidate, scale, threshold, or template.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from neural_decompiler import candidate_screening as cs
from neural_decompiler.models import ModelSpec, load_model, resolved_revision, seed_runtime, validate_runtime
from neural_decompiler.provenance import collect_git_state, collect_versions


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / cs.MANIFEST_RELATIVE_PATH
OUTPUT_DIR = ROOT / "outputs/behavior-candidate-screening"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
RUNTIME_SEED = 20260916


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("validate", help="check the committed manifest without loading a model")
    for phase in ("behavioral", "compactness"):
        scientific = phases.add_parser(phase)
        scientific.add_argument("--resume", action="store_true", help="continue an interrupted, provenance-identical running phase")
        scientific.add_argument("--incident-note", default=None, metavar="TRACKED_PATH", help="invalidate a completed screen after a committed software fix")
    report = phases.add_parser("report")
    report.add_argument("--audit-json", default=None, metavar="JSON", help="finalist prior-art audit entries")
    return parser


def _git_tracked(path: Path) -> bool:
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def runtime_record(spec: ModelSpec) -> dict[str, Any]:
    runtime = validate_runtime(spec)
    return {"device": runtime.device, "backend": runtime.backend, "dtype": runtime.dtype_name, "seed": RUNTIME_SEED,
            "deterministic_algorithms": spec.deterministic_algorithms}


@dataclass
class Runner:
    """Phase orchestration with injectable model, Git, and version providers."""

    manifest_path: Path = MANIFEST_PATH
    results_path: Path = RESULTS_PATH
    report_path: Path = REPORT_PATH
    model_loader: Callable[[ModelSpec], Any] = load_model
    screen: Callable[..., Any] = cs.run_behavioral_screen
    probe: Callable[..., Any] = cs.run_compactness_probe
    git_state: Callable[[], Mapping[str, Any]] = field(default_factory=lambda: lambda: collect_git_state(ROOT))
    versions: Callable[[], Mapping[str, Any]] = collect_versions
    tracked: Callable[[Path], bool] = _git_tracked
    log: Callable[[str], None] = field(default_factory=lambda: lambda message: print(message, flush=True))
    write_every: int = 20

    # -- shared helpers -----------------------------------------------------

    def _write(self, state: Mapping[str, Any]) -> str:
        return cs.write_results_state(self.results_path, state)

    def _current_provenance(self) -> dict[str, Any]:
        git = dict(self.git_state())
        return {"protocol_code_commit": git.get("commit") or "", "git_dirty": bool(git.get("dirty")), "versions": dict(self.versions())}

    def _load_state_for(self, phase: str, manifest: cs.ScreeningManifest, *, resume: bool, incident_note: str | None) -> dict[str, Any]:
        provenance = self._current_provenance()
        invalidated = False
        if self.results_path.exists():
            state = cs.load_results_state(self.results_path)
            if incident_note is not None:
                state = self._apply_incident(state, incident_note, phase)
                invalidated = True
        elif phase == "behavioral" and not resume and incident_note is None:
            state = cs.new_results_state(manifest_sha256=manifest.content_sha256 or "", **provenance)
        else:
            raise cs.PhaseError(f"{phase} requires an existing results state at {self.results_path}")
        cs.assert_phase_allowed(phase, state, resume=resume)
        cs.assert_provenance_identical(state, manifest_sha256=manifest.content_sha256 or "", **provenance)
        if invalidated:
            # Persist the invalidation only once every refusal check has passed.
            self._write(state)
        return state

    def _apply_incident(self, state: Mapping[str, Any], note_path: str, phase: str) -> dict[str, Any]:
        path = Path(note_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.suffix != ".md" or not path.exists() or not self.tracked(path):
            raise cs.PhaseError("incident note must be an existing tracked Markdown file")
        note = cs.parse_incident_note(path.read_text(encoding="utf-8"))
        if note["phase"] != phase:
            raise cs.PhaseError(f"incident note is for the {note['phase']} phase, not {phase}")
        artifact_sha256 = cs.sha256_text(self.results_path.read_text(encoding="utf-8"))
        relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        return cs.invalidate_run_with_incident(state, note, note_path=relative, artifact_sha256=artifact_sha256)

    # -- phases -------------------------------------------------------------

    def validate(self) -> int:
        manifest = cs.load_manifest(self.manifest_path)
        for candidate in manifest.candidates:
            for split in cs.Split:
                cases = manifest.cases_for(candidate.candidate_id, split)
                self.log(f"{candidate.candidate_id} {split.value}: {len(cases)} cases across {len(candidate.template_ids)} templates")
        development_single = all(condition.single_token for candidate_id in manifest.candidate_ids
                                 for case in manifest.cases_for(candidate_id, cs.Split.DEVELOPMENT) for condition in (case.x_a, case.x_b, case.s_a, case.s_b))
        self.log(f"manifest sha256 {manifest.content_sha256}; {len(manifest.cases)} cases; development single-token: {development_single}")
        self.log("future-reserve cases are enumerable and not executable")
        return 0

    def behavioral(self, *, resume: bool = False, incident_note: str | None = None) -> int:
        manifest = cs.load_manifest(self.manifest_path)
        state = self._load_state_for("behavioral", manifest, resume=resume, incident_note=incident_note)
        phase = state["phases"]["behavioral"]
        state["phases"]["behavioral"] = {**phase, "status": "running", "started_at": phase.get("started_at") or cs.utc_now()}
        self._write(state)
        expected_ids = [case.case_id for candidate_id in manifest.candidate_ids for split in cs.EXECUTABLE_SPLITS
                        for case in cs.executable_cases(manifest, candidate_id, split)]
        for spec in cs.MODEL_ORDER:
            screen = state["behavioral"].get(spec.model_id)
            if screen is not None and screen["status"] == "complete":
                if screen["passing_candidate_ids"]:
                    break
                continue
            runtime = runtime_record(spec)
            if screen is None:
                screen = {"status": "running", "model_id": spec.model_id, "revision": spec.revision, "resolved_revision": None, "runtime": runtime,
                          "started_at": cs.utc_now(), "completed_at": None, "measurements": {}, "measurement_digests": {}, "executed_case_ids": [],
                          "candidates": {}, "passing_candidate_ids": []}
                state["behavioral"][spec.model_id] = screen
            pending = cs.resume_pending_case_ids(screen, expected_ids, revision=spec.revision, runtime=runtime)
            self.log(f"behavioral {spec.model_id}: {len(pending)} of {len(expected_ids)} executable cases pending")
            seed_runtime(RUNTIME_SEED, spec.deterministic_algorithms)
            model = self.model_loader(spec)
            screen["resolved_revision"] = resolved_revision(model) or spec.revision
            completed = {case_id: cs.CaseMeasurement.from_dict(value) for case_id, value in screen["measurements"].items()}
            progress = {"count": 0}

            def on_case(measurement: cs.CaseMeasurement, *, screen: dict[str, Any] = screen) -> None:
                value = measurement.to_dict()
                screen["measurements"][measurement.case_id] = value
                screen["measurement_digests"][measurement.case_id] = cs.measurement_digest(value)
                screen["executed_case_ids"].append(measurement.case_id)
                progress["count"] += 1
                if progress["count"] % self.write_every == 0:
                    self._write(state)
                    self.log(f"behavioral {spec.model_id}: {len(screen['executed_case_ids'])} cases recorded")

            try:
                results = self.screen(model, manifest, None, on_case=on_case, completed=completed)
            finally:
                self._write(state)
            for candidate_id, per_split in results.items():
                screen["candidates"][candidate_id] = cs.behavioral_result_for_candidate(
                    manifest, candidate_id, per_split[cs.Split.DEVELOPMENT.value], per_split[cs.Split.HOLDOUT.value])
            screen["passing_candidate_ids"] = list(cs.behavioral_passers(state, spec.model_id))
            screen["status"], screen["completed_at"] = "complete", cs.utc_now()
            del model
            gc.collect()
            self.log(f"behavioral {spec.model_id}: passers {screen['passing_candidate_ids'] or 'none'}")
            self._write(state)
            if screen["passing_candidate_ids"]:
                state["selection_model_id"] = spec.model_id
                break
        state["phases"]["behavioral"] = {**state["phases"]["behavioral"], "status": "complete", "completed_at": cs.utc_now()}
        self._write(state)
        return 0

    def compactness(self, *, resume: bool = False, incident_note: str | None = None) -> int:
        manifest = cs.load_manifest(self.manifest_path)
        state = self._load_state_for("compactness", manifest, resume=resume, incident_note=incident_note)
        phase = state["phases"]["compactness"]
        state["phases"]["compactness"] = {**phase, "status": "running", "started_at": phase.get("started_at") or cs.utc_now()}
        model_id = state["selection_model_id"]
        if model_id is None:
            self.log("compactness: no candidate passed behavioral screening at either scale; nothing to probe")
            state["phases"]["compactness"] = {**state["phases"]["compactness"], "status": "complete", "completed_at": cs.utc_now(), "note": "no behavioral passer"}
            self._write(state)
            return 0
        spec = next(item for item in cs.MODEL_ORDER if item.model_id == model_id)
        passers = cs.behavioral_passers(state, model_id)
        entry = state["compactness"].get(model_id)
        if entry is None:
            entry = {"status": "running", "model_id": model_id, "revision": spec.revision, "runtime": runtime_record(spec), "started_at": cs.utc_now(),
                     "completed_at": None, "candidates": {}, "executed_case_ids": []}
            state["compactness"][model_id] = entry
        elif entry["revision"] != spec.revision or entry["runtime"] != runtime_record(spec):
            raise cs.PhaseError("model revision or runtime differs from the recorded compactness run")
        self._write(state)
        seed_runtime(RUNTIME_SEED, spec.deterministic_algorithms)
        model = self.model_loader(spec)
        for candidate_id in passers:
            if candidate_id in entry["candidates"]:
                continue
            self.log(f"compactness {model_id}: probing {candidate_id}")

            def on_progress(stage: str, done: int, total: int) -> None:
                if done == total or done % 10 == 0:
                    self.log(f"compactness {candidate_id}: {stage} {done}/{total}")

            try:
                result = self.probe(model, manifest, candidate_id, on_progress=on_progress)
            except cs.CompactnessIntegrityError as error:
                gate = cs.CompactnessGateResult({"denominators": False, "overall_recovery": False, "beats_random": False, "positive_template_recovery": False}, False)
                result = {"candidate_id": candidate_id, "integrity_error": str(error), "gate": gate.to_dict()}
            entry["candidates"][candidate_id] = result
            entry["executed_case_ids"].extend(result.get("discovery_case_ids", []) + result.get("validation_case_ids", []))
            self._write(state)
        del model
        gc.collect()
        entry["status"], entry["completed_at"] = "complete", cs.utc_now()
        state["phases"]["compactness"] = {**state["phases"]["compactness"], "status": "complete", "completed_at": cs.utc_now()}
        self._write(state)
        return 0

    def report(self, *, audit_json: str | None = None) -> int:
        manifest = cs.load_manifest(self.manifest_path)
        state = cs.load_results_state(self.results_path)
        if state["manifest_sha256"] != manifest.content_sha256:
            raise cs.PhaseError("manifest digest differs from the recorded run")
        if audit_json is not None:
            audits = cs.parse_audit_json(audit_json)
            cs.validate_finalist_audits(state, audits)
            state["audits"].update(audits)
        state["selection"] = cs.finalize_selection(state)
        final = state["selection"]["status"] != "PENDING_FINALIST_AUDIT"
        state["phases"]["report"] = {"status": "complete" if final else "running", "rendered_at": cs.utc_now()}
        self._write(state)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(cs.render_markdown_report(state, manifest), encoding="utf-8")
        self.log(f"report: {state['selection']['status']} -> {self.report_path}")
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner()
    try:
        if args.phase == "validate":
            return runner.validate()
        if args.phase == "behavioral":
            return runner.behavioral(resume=args.resume, incident_note=args.incident_note)
        if args.phase == "compactness":
            return runner.compactness(resume=args.resume, incident_note=args.incident_note)
        return runner.report(audit_json=args.audit_json)
    except (cs.PhaseError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
