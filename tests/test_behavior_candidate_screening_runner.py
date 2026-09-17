"""Phase isolation, model order, reserve non-execution, resume, and refusal tests for the screening runner."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from neural_decompiler import candidate_screening as cs
from neural_decompiler.models import PYTHIA_160M, PYTHIA_70M


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / cs.MANIFEST_RELATIVE_PATH
COMMIT = "a" * 40
FIX_COMMIT = "b" * 40


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("screening_runner", ROOT / "screening/behavior-candidates/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()
manifest = cs.load_manifest(MANIFEST_PATH)


class FakeModel:
    def __init__(self, spec) -> None:
        self.spec = spec
        self.cfg = type("Cfg", (), {"device": "cpu", "n_layers": 1, "n_heads": 1})()
        self.original_model = type("HF", (), {"config": type("C", (), {"_commit_hash": spec.revision})()})()


def measurement(case: cs.ScreeningCase, *, correct: bool) -> cs.CaseMeasurement:
    good = {"A": -1.0, "B": -3.0}
    bad = {"A": -3.0, "B": -1.0}
    if case.primary_orientation == "A":
        x_a, x_b = (good if correct else bad), bad
    else:
        x_a, x_b = good, (bad if correct else good)
    return cs.measure_case({"x_a": x_a, "x_b": x_b, "s_a": x_b, "s_b": x_a}, case.primary_orientation, case_id=case.case_id, template_id=case.template_id)


def make_screen(pass_map: dict[str, dict[str, bool]], *, crash_after: int | None = None, calls: list[str] | None = None):
    """Fake behavioral screen: pass_map[model_id][candidate_id] decides whether a candidate passes."""

    def screen(model, manifest_, candidate_ids, *, on_case, completed):
        def scorer(model_, case):
            if calls is not None:
                calls.append(case.case_id)
            if crash_after is not None and len(calls or []) > crash_after:
                raise RuntimeError("simulated crash")
            passes = pass_map[model.spec.model_id][case.candidate_id]
            index = int(case.case_id[-2:])
            return measurement(case, correct=passes or index % 2 == 0)

        return cs.run_behavioral_screen(model, manifest_, candidate_ids, on_case=on_case, scorer=scorer, completed=completed)

    return screen


def make_runner(tmp_path: Path, *, pass_map, loads: list[str], probe=None, git=None, crash_after=None, calls=None, tracked=None):
    def loader(spec):
        loads.append(spec.model_id)
        return FakeModel(spec)

    def default_probe(model, manifest_, candidate_id, *, on_progress=None):
        cases = manifest_.cases_for(candidate_id, cs.Split.DEVELOPMENT)
        gate = cs.CompactnessGateResult({"denominators": True, "overall_recovery": True, "beats_random": True, "positive_template_recovery": True}, True, 3, {})
        return {"candidate_id": candidate_id, "seed": cs.COMPACTNESS_SEED, "max_k": 12, "random_sets_per_k": 100, "component_universe": ["L00.H00"],
                "discovery_case_ids": [case.case_id for case in cases[:60]], "validation_case_ids": [case.case_id for case in cases[60:]],
                "unpatched": {}, "validation_denominators": cs.evaluate_compactness_denominators(1.0, {"t": 1.0}).to_dict(),
                "singleton_discovery": {"L00.H00": {"mean_d_patch": 0.5, "effects": []}}, "ranking": ["L00.H00"],
                "top_k": {"3": {"components": ["L00.H00"], "measurements": []}}, "random": {"3": {"sets": [], "recoveries": [0.1], "median": 0.1, "reference": 0.1}},
                "gate": gate.to_dict()}

    return runner.Runner(
        manifest_path=MANIFEST_PATH, results_path=tmp_path / "results.json", report_path=tmp_path / "report.md",
        model_loader=loader, screen=make_screen(pass_map, crash_after=crash_after, calls=calls), probe=probe or default_probe,
        git_state=git or (lambda: {"commit": COMMIT, "dirty": False}), versions=lambda: {"python": "3.12.0"},
        tracked=tracked or (lambda path: True), log=lambda message: None, write_every=1 if crash_after is not None else 20)


ALL_PASS_70M = {PYTHIA_70M.model_id: {"regular-plural": True, "ordinal-suffix": True}, PYTHIA_160M.model_id: {"regular-plural": True, "ordinal-suffix": True}}
ORDINAL_PASS_70M = {PYTHIA_70M.model_id: {"regular-plural": False, "ordinal-suffix": True}, PYTHIA_160M.model_id: {"regular-plural": True, "ordinal-suffix": True}}
NONE_70M_PLURAL_160M = {PYTHIA_70M.model_id: {"regular-plural": False, "ordinal-suffix": False}, PYTHIA_160M.model_id: {"regular-plural": True, "ordinal-suffix": False}}
NONE_ANYWHERE = {PYTHIA_70M.model_id: {"regular-plural": False, "ordinal-suffix": False}, PYTHIA_160M.model_id: {"regular-plural": False, "ordinal-suffix": False}}


def test_cli_exposes_only_four_phases_and_no_candidate_or_threshold_override() -> None:
    parser = runner.build_parser()
    for phase in ("validate", "behavioral", "compactness", "report"):
        assert parser.parse_args([phase]).phase == phase
    assert parser.parse_args(["behavioral", "--resume"]).resume is True
    for bad in (["behavioral", "--candidate", "ordinal-suffix"], ["behavioral", "--threshold", "0.80"], ["compactness", "--model", "160m"], ["validate", "--resume"], ["report", "--resume"], ["screen"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_validate_phase_never_needs_a_model_or_results_state(tmp_path) -> None:
    loads: list[str] = []
    lines: list[str] = []
    run = make_runner(tmp_path, pass_map=ALL_PASS_70M, loads=loads)
    run.log = lines.append
    assert run.validate() == 0
    assert loads == [] and not (tmp_path / "results.json").exists()
    assert any("720 cases" in line for line in lines) and any("not executable" in line for line in lines)


def test_phase_order_and_single_execution_rules() -> None:
    state = cs.new_results_state(manifest_sha256="m", protocol_code_commit=COMMIT, git_dirty=False, versions={})
    with pytest.raises(cs.PhaseError, match="behavioral"):
        cs.assert_phase_allowed("compactness", state)
    with pytest.raises(cs.PhaseError, match="--resume is only"):
        cs.assert_phase_allowed("behavioral", state, resume=True)
    state["phases"]["behavioral"]["status"] = "running"
    with pytest.raises(cs.PhaseError, match="already running"):
        cs.assert_phase_allowed("behavioral", state)
    cs.assert_phase_allowed("behavioral", state, resume=True)
    state["phases"]["behavioral"]["status"] = "complete"
    with pytest.raises(cs.PhaseError, match="already complete"):
        cs.assert_phase_allowed("behavioral", state)
    cs.assert_phase_allowed("compactness", state)
    cs.assert_phase_allowed("report", state)
    with pytest.raises(cs.PhaseError, match="clean Git tree"):
        cs.new_results_state(manifest_sha256="m", protocol_code_commit=COMMIT, git_dirty=True, versions={})
    with pytest.raises(cs.PhaseError, match="40-character"):
        cs.new_results_state(manifest_sha256="m", protocol_code_commit="abc", git_dirty=False, versions={})


def test_behavioral_stops_after_70m_when_any_candidate_passes(tmp_path) -> None:
    loads: list[str] = []
    run = make_runner(tmp_path, pass_map=ORDINAL_PASS_70M, loads=loads)
    assert run.behavioral() == 0
    assert loads == [PYTHIA_70M.model_id]
    state = cs.load_results_state(run.results_path)
    assert state["selection_model_id"] == PYTHIA_70M.model_id
    assert state["behavioral"][PYTHIA_70M.model_id]["passing_candidate_ids"] == ["ordinal-suffix"]
    assert state["behavioral"][PYTHIA_70M.model_id]["candidates"]["regular-plural"]["gates"]["passed"] is False
    assert PYTHIA_160M.model_id not in state["behavioral"]
    assert state["phases"]["behavioral"]["status"] == "complete"


def test_behavioral_runs_160m_only_when_zero_70m_candidates_pass(tmp_path) -> None:
    loads: list[str] = []
    run = make_runner(tmp_path, pass_map=NONE_70M_PLURAL_160M, loads=loads)
    assert run.behavioral() == 0
    assert loads == [PYTHIA_70M.model_id, PYTHIA_160M.model_id]
    state = cs.load_results_state(run.results_path)
    assert state["selection_model_id"] == PYTHIA_160M.model_id
    assert state["behavioral"][PYTHIA_70M.model_id]["passing_candidate_ids"] == []
    assert state["behavioral"][PYTHIA_160M.model_id]["passing_candidate_ids"] == ["regular-plural"]
    assert len(state["behavioral"][PYTHIA_160M.model_id]["executed_case_ids"]) == 480


def test_future_reserve_ids_never_execute_and_executed_ids_are_exactly_development_and_holdout(tmp_path) -> None:
    loads: list[str] = []
    run = make_runner(tmp_path, pass_map=NONE_ANYWHERE, loads=loads)
    run.behavioral()
    state = cs.load_results_state(run.results_path)
    reserve = {case.case_id for candidate_id in manifest.candidate_ids for case in manifest.cases_for(candidate_id, cs.Split.FUTURE_RESERVE)}
    expected = [case.case_id for candidate_id in manifest.candidate_ids for split in cs.EXECUTABLE_SPLITS for case in manifest.cases_for(candidate_id, split)]
    for model_id in (PYTHIA_70M.model_id, PYTHIA_160M.model_id):
        executed = state["behavioral"][model_id]["executed_case_ids"]
        assert executed == expected and not (set(executed) & reserve)
    assert state["selection_model_id"] is None
    assert run.compactness() == 0
    state = cs.load_results_state(run.results_path)
    assert state["phases"]["compactness"]["status"] == "complete" and state["compactness"] == {}
    assert run.report() == 0
    state = cs.load_results_state(run.results_path)
    assert state["selection"]["status"] == "ZERO_TARGET"
    text = run.report_path.read_text()
    assert "`ZERO_TARGET`" in text and "reserve IDs executed: 0" in text and "No Experiment 005 exists" in text


def test_crash_then_resume_completes_exactly_the_remaining_cases(tmp_path) -> None:
    uninterrupted_loads: list[str] = []
    reference = make_runner(tmp_path / "reference", pass_map=ORDINAL_PASS_70M, loads=uninterrupted_loads)
    reference.behavioral()
    reference_state = cs.load_results_state(reference.results_path)

    calls: list[str] = []
    loads: list[str] = []
    crashing = make_runner(tmp_path / "crash", pass_map=ORDINAL_PASS_70M, loads=loads, crash_after=17, calls=calls)
    with pytest.raises(RuntimeError, match="simulated crash"):
        crashing.behavioral()
    state = cs.load_results_state(crashing.results_path)
    assert state["phases"]["behavioral"]["status"] == "running"
    assert len(state["behavioral"][PYTHIA_70M.model_id]["executed_case_ids"]) == 17
    with pytest.raises(cs.PhaseError, match="already running"):
        crashing.behavioral()

    resumed_calls: list[str] = []
    resumed = make_runner(tmp_path / "crash", pass_map=ORDINAL_PASS_70M, loads=loads, calls=resumed_calls)
    assert resumed.behavioral(resume=True) == 0
    expected = [case.case_id for candidate_id in manifest.candidate_ids for split in cs.EXECUTABLE_SPLITS for case in manifest.cases_for(candidate_id, split)]
    assert resumed_calls == expected[17:]
    final = cs.load_results_state(resumed.results_path)
    volatile = {"run_id", "created_at", "state_sha256", "phases"}
    for key in set(final) - volatile:
        if key == "behavioral":
            left, right = final[key][PYTHIA_70M.model_id], reference_state[key][PYTHIA_70M.model_id]
            for field in set(left) - {"started_at", "completed_at"}:
                assert left[field] == right[field], field
        else:
            assert final[key] == reference_state[key], key


def test_tampering_is_rejected_before_any_model_load(tmp_path) -> None:
    loads: list[str] = []
    run = make_runner(tmp_path, pass_map=ORDINAL_PASS_70M, loads=loads, crash_after=5, calls=[])
    with pytest.raises(RuntimeError):
        run.behavioral()
    original = json.loads(run.results_path.read_text())
    loads.clear()

    def rewrite(state: dict[str, Any]) -> None:
        state["state_sha256"] = cs.state_digest(state)
        run.results_path.write_text(json.dumps(state))

    tampered = copy.deepcopy(original)
    tampered["behavioral"][PYTHIA_70M.model_id]["measurements"][original["behavioral"][PYTHIA_70M.model_id]["executed_case_ids"][0]]["x_a"]["logp_a"] = -0.5
    run.results_path.write_text(json.dumps(tampered))
    with pytest.raises(cs.PhaseError, match="modified outside the runner"):
        run.behavioral(resume=True)
    rewrite(tampered)
    with pytest.raises(cs.PhaseError, match="does not match its digest"):
        run.behavioral(resume=True)
    for field, value, message in (("manifest_sha256", "0" * 64, "manifest digest"), ("protocol_code_commit", FIX_COMMIT, "commit differs")):
        broken = copy.deepcopy(original)
        broken[field] = value
        rewrite(broken)
        with pytest.raises(cs.PhaseError, match=message):
            run.behavioral(resume=True)
    broken = copy.deepcopy(original)
    broken["behavioral"][PYTHIA_70M.model_id]["revision"] = FIX_COMMIT
    rewrite(broken)
    with pytest.raises(cs.PhaseError, match="revision or runtime"):
        run.behavioral(resume=True)
    assert loads == []
    rewrite(copy.deepcopy(original))
    dirty = make_runner(tmp_path, pass_map=ORDINAL_PASS_70M, loads=loads, git=lambda: {"commit": COMMIT, "dirty": True})
    with pytest.raises(cs.PhaseError, match="clean Git tree"):
        dirty.behavioral(resume=True)
    assert loads == []


def test_second_completed_run_requires_a_valid_tracked_incident_note(tmp_path) -> None:
    loads: list[str] = []
    run = make_runner(tmp_path, pass_map=ORDINAL_PASS_70M, loads=loads)
    run.behavioral()
    with pytest.raises(cs.PhaseError, match="already complete"):
        run.behavioral()
    state = cs.load_results_state(run.results_path)
    digest = cs.sha256_text(run.results_path.read_text())
    note = tmp_path / "incident.md"
    note.write_text("# Incident\n\n- run_id: WRONG\n- phase: behavioral\n- model_id: EleutherAI/pythia-70m-deduped\n- defect: scorer bug\n"
                    f"- invalid_artifact_sha256: {digest}\n- fix_commit: {FIX_COMMIT}\n- decision: full-rerun\n")
    with pytest.raises(cs.PhaseError, match="different run ID"):
        run.behavioral(incident_note=str(note))
    note.write_text(note.read_text().replace("WRONG", state["run_id"]))
    untracked = make_runner(tmp_path, pass_map=ORDINAL_PASS_70M, loads=loads, tracked=lambda path: False)
    with pytest.raises(cs.PhaseError, match="tracked Markdown"):
        untracked.behavioral(incident_note=str(note))
    with pytest.raises(cs.PhaseError, match="commit differs"):
        run.behavioral(incident_note=str(note))  # HEAD is still COMMIT, not the fix commit
    fixed = make_runner(tmp_path, pass_map=ORDINAL_PASS_70M, loads=loads, git=lambda: {"commit": FIX_COMMIT, "dirty": False})
    assert fixed.behavioral(incident_note=str(note)) == 0
    rerun = cs.load_results_state(run.results_path)
    assert rerun["run_id"] != state["run_id"] and rerun["protocol_code_commit"] == FIX_COMMIT
    assert len(rerun["invalidated_runs"]) == 1 and rerun["invalidated_runs"][0]["run_id"] == state["run_id"]
    assert rerun["invalidated_runs"][0]["invalid_artifact_sha256"] == digest
    assert rerun["behavioral"][PYTHIA_70M.model_id]["status"] == "complete"
    assert loads == [PYTHIA_70M.model_id, PYTHIA_70M.model_id]
    with pytest.raises(cs.PhaseError, match="already complete"):
        fixed.behavioral()


def test_compactness_probes_only_behavioral_passers_of_the_selection_model(tmp_path) -> None:
    loads: list[str] = []
    probed: list[str] = []

    def probe(model, manifest_, candidate_id, *, on_progress=None):
        probed.append((model.spec.model_id, candidate_id))
        raise cs.CompactnessIntegrityError("L00.H00: outside change")

    run = make_runner(tmp_path, pass_map=NONE_70M_PLURAL_160M, loads=loads, probe=probe)
    with pytest.raises(cs.PhaseError, match="requires an existing results state"):
        run.compactness()
    run.behavioral()
    assert run.compactness() == 0
    assert probed == [(PYTHIA_160M.model_id, "regular-plural")]
    state = cs.load_results_state(run.results_path)
    entry = state["compactness"][PYTHIA_160M.model_id]
    assert entry["status"] == "complete" and entry["candidates"]["regular-plural"]["gate"]["passed"] is False
    assert "outside change" in entry["candidates"]["regular-plural"]["integrity_error"]
    with pytest.raises(cs.PhaseError, match="already complete"):
        run.compactness()
    run.report()
    state = cs.load_results_state(run.results_path)
    assert state["selection"]["status"] == "ZERO_TARGET"
    assert "Intervention integrity failure" in run.report_path.read_text()


def test_report_pending_audit_then_deterministic_selection(tmp_path) -> None:
    loads: list[str] = []
    run = make_runner(tmp_path, pass_map=ALL_PASS_70M, loads=loads)
    run.behavioral()
    run.compactness()
    assert run.report() == 0
    state = cs.load_results_state(run.results_path)
    assert state["selection"]["status"] == "PENDING_FINALIST_AUDIT"
    assert state["selection"]["pending_audits"] == ["regular-plural", "ordinal-suffix"]
    assert state["phases"]["report"]["status"] == "running"
    text = run.report_path.read_text()
    assert "pending finalist audit" in text and "`PENDING_FINALIST_AUDIT`" in text

    def entry(status: str, sources: list[str]) -> dict[str, Any]:
        return {"novelty_status": status, "searched_through": "2026-09-17", "behavior_variants": ["x"], "model_families": ["Pythia"],
                "explanation_levels": list(cs.EXPLANATION_LEVELS), "validation_modes": list(cs.VALIDATION_MODES),
                "primary_sources": sources, "explicit_gap": "No close naturally learned mechanism was located."}

    with pytest.raises(ValueError, match="non-finalists"):
        run.report(audit_json=json.dumps({"degree-inflection": entry("NO_CLOSE_MECHANISM_LOCATED", [])}))
    with pytest.raises(ValueError, match="unknown"):
        run.report(audit_json=json.dumps({"ordinal-suffix": {**entry("NO_CLOSE_MECHANISM_LOCATED", []), "extra": 1}}))
    with pytest.raises(ValueError, match="unrecognized novelty status"):
        run.report(audit_json=json.dumps({"ordinal-suffix": entry("NOVEL", [])}))
    with pytest.raises(ValueError, match="primary source"):
        run.report(audit_json=json.dumps({"ordinal-suffix": entry("PARTIAL_OVERLAP_WITH_EXPLICIT_GAP", [])}))
    run.report(audit_json=json.dumps({"regular-plural": entry("PARTIAL_OVERLAP_WITH_EXPLICIT_GAP", ["https://example.org/plural"])}))
    state = cs.load_results_state(run.results_path)
    assert state["selection"]["status"] == "PENDING_FINALIST_AUDIT" and state["selection"]["pending_audits"] == ["ordinal-suffix"]
    run.report(audit_json=json.dumps({"ordinal-suffix": entry("NO_CLOSE_MECHANISM_LOCATED", [])}))
    state = cs.load_results_state(run.results_path)
    assert state["selection"]["status"] == "ONE_PROPOSED_TARGET"
    assert state["selection"]["selected_candidate_id"] == "ordinal-suffix"
    assert state["selection"]["ranking"] == ["ordinal-suffix", "regular-plural"]
    assert state["phases"]["report"]["status"] == "complete"
    text = run.report_path.read_text()
    assert "`ONE_PROPOSED_TARGET`" in text and "proposed target" in text and "No Experiment 005 exists" in text
    for name in cs._GATE_NAMES:
        assert f"`{name}`" in text
    assert "Baselines: majority 50.0%, lexical-prior" in text and "reserve IDs executed: 0" in text and "exploratory" in text
    assert "| 1. `overall_accuracy` |" in text and "| 10. `integrity` |" in text
    assert all("entry_sha256" in audit and "recorded_at" in audit for audit in state["audits"].values())
    # Recorded audits are write-once: a silent re-audit that would flip the decision is refused.
    with pytest.raises(cs.PhaseError, match="already recorded"):
        run.report(audit_json=json.dumps({"ordinal-suffix": entry("SUBSTANTIALLY_MAPPED", ["https://example.org/ordinal"])}))
    state = cs.load_results_state(run.results_path)
    assert state["selection"]["status"] == "ONE_PROPOSED_TARGET"
    # A tracked report-phase incident note moves the audits and decision into history first.
    digest = cs.sha256_text(run.results_path.read_text())
    note = tmp_path / "audit-incident.md"
    note.write_text(f"- run_id: {state['run_id']}\n- phase: report\n- model_id: {PYTHIA_70M.model_id}\n- defect: missed a close prior mechanism\n"
                    f"- invalid_artifact_sha256: {digest}\n- fix_commit: {COMMIT}\n- decision: full-rerun\n")
    run.report(incident_note=str(note), audit_json=json.dumps({"ordinal-suffix": entry("SUBSTANTIALLY_MAPPED", ["https://example.org/ordinal"]),
                                                              "regular-plural": entry("SUBSTANTIALLY_MAPPED", ["https://example.org/plural"])}))
    state = cs.load_results_state(run.results_path)
    assert state["selection"]["status"] == "ZERO_TARGET" and state["selection"]["selected_candidate_id"] is None
    assert len(state["invalidated_runs"]) == 1 and state["invalidated_runs"][0]["phase"] == "report"
    assert state["invalidated_runs"][0]["selection"]["status"] == "ONE_PROPOSED_TARGET"
    assert state["behavioral"][PYTHIA_70M.model_id]["status"] == "complete" and state["compactness"][PYTHIA_70M.model_id]["status"] == "complete"
    assert "Invalidated runs" in run.report_path.read_text() and "missed a close prior mechanism" in run.report_path.read_text()


def test_resume_after_crash_between_completion_writes_restores_selection_model(tmp_path) -> None:
    """A screen recorded complete with passers must select that model even if the crash hit before the phase closed."""
    loads: list[str] = []
    run = make_runner(tmp_path, pass_map=ORDINAL_PASS_70M, loads=loads)
    run.behavioral()
    state = cs.load_results_state(run.results_path)
    state["selection_model_id"] = None
    state["phases"]["behavioral"] = {"status": "running", "started_at": state["phases"]["behavioral"]["started_at"]}
    state["state_sha256"] = cs.state_digest(state)
    run.results_path.write_text(json.dumps(state))
    assert run.behavioral(resume=True) == 0
    resumed = cs.load_results_state(run.results_path)
    assert resumed["selection_model_id"] == PYTHIA_70M.model_id
    assert loads == [PYTHIA_70M.model_id]


def test_selection_orders_by_worst_template_after_novelty() -> None:
    def summary(worst: float, overall: float, flip: float) -> dict[str, Any]:
        return {"accuracy": overall, "template_accuracies": {"t1": worst, "t2": 0.95, "t3": 0.95}, "contrast_flip_rate": flip}

    def state_with(a_status: str, b_status: str, a_k: int = 2, b_k: int = 2) -> dict[str, Any]:
        return {
            "selection_model_id": "m", "phases": {"behavioral": {"status": "complete"}, "compactness": {"status": "complete"}, "report": {"status": "not_started"}},
            "behavioral": {"m": {"candidates": {"alpha": {"holdout": summary(0.90, 0.95, 0.9)}, "beta": {"holdout": summary(0.925, 0.93, 0.8)}}}},
            "compactness": {"m": {"status": "complete", "candidates": {"alpha": {"gate": {"passed": True, "selected_k": a_k}}, "beta": {"gate": {"passed": True, "selected_k": b_k}}}}},
            "audits": {"alpha": {"novelty_status": a_status}, "beta": {"novelty_status": b_status}},
        }

    assert cs.select_finalist(state_with("NO_CLOSE_MECHANISM_LOCATED", "NO_CLOSE_MECHANISM_LOCATED")).selected_candidate_id == "beta"
    assert cs.select_finalist(state_with("NO_CLOSE_MECHANISM_LOCATED", "PARTIAL_OVERLAP_WITH_EXPLICIT_GAP")).selected_candidate_id == "alpha"
    assert cs.select_finalist(state_with("SUBSTANTIALLY_MAPPED", "SUBSTANTIALLY_MAPPED")).selected_candidate_id is None
    assert cs.select_finalist(state_with("SUBSTANTIALLY_MAPPED", "PARTIAL_OVERLAP_WITH_EXPLICIT_GAP")).ranking == ("beta",)
