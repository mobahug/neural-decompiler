"""Experiment 025's runner on the six-layer fake, inside the fake world of Experiment 024's runner test.

020 is explored and closed on the fake; 022 is frozen and calibrated; 023 is extracted, frozen, calibrated and locked;
024 is frozen, calibrated and locked on the fake by its own runner. So 024's calibration record, lock and freeze exist
in 024's committed formats. The 025 constants that are facts about the world are pointed at the fake world's:
- the digests of 024's reviewed files (``INHERITED_024``) and of 020's confirmation file (``PRIOR_NOUNS_020``);
- the patch-path keys (``PATCH_PATH_SPENT_KEYS``): the fake's 020 ledger has other cues, so one already-executed key
  per production key's frame.

The world uses an explicit test configuration, never a patched production constant: ``FAKE``, with 2 adjectives, 2
nouns, 2 random controls (11 conditions) and a threshold of 3 of 4. On that world it checks:
- every phase and its refusals, and the freeze's stops;
- the no-forward-pass lock;
- I7′ and the patch-path check (each an incident before any fresh key);
- the accounting and the saved measurements, and `C` recomputed bit for bit;
- the gate incidents, and the single result write;
- the descriptive records' failures, and the report.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import head_pattern as hp
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul
from test_experiment_020_runner import fake_world, make_fake_model  # noqa: F401 — fake_world is a fixture
from test_experiment_022_runner import COMMIT_A
from test_experiment_022_runner import calibrated, frozen, world  # noqa: F401 — fixtures
from test_experiment_023_runner import base023, calibrated023, extracted, frozen023, locked023  # noqa: F401 — fixtures
from test_experiment_024_runner import FAKE as FAKE_024
from test_experiment_024_runner import _apply as apply_024
from test_experiment_024_runner import _install, toy_tokenizer_024
from test_experiment_024_runner import base024, calibrated024, frozen024, locked024  # noqa: F401 — fixtures
from test_experiment_024_runner import make_runner as make_runner_024

ROOT = Path(__file__).parents[1]


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_025_runner", ROOT / "experiments/025-nounness-direction-intervention/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner_module()

FAKE = cr.Configuration(name="fake-world", n_adjectives=2, n_nouns=2, k_controls=2, primary_odd=0.32, half_odd=0.16, count_threshold=3, n_frames=108,
                        n_scored_nouns=79, expected_picks=(("adjective", ("anxious", "cheerful")), ("noun", ("soldier", "sailor"))))


def toy_tokenizer_025(manifest, pool):
    """024's toy tokenizer plus every 025 candidate word (fresh ids below the fake's vocabulary)."""
    base = toy_tokenizer_024(manifest, pool)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words = [" " + word for word in cr.ADJECTIVE_RESERVES] + [" " + word for pair in cr.ORDINARY_RESERVES for word in pair]
    words += [" " + word for word in cr.NEW_NOUN_LIST] + [" " + cr.regular_plural(word) for word in cr.NEW_NOUN_LIST]
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    assert next_id < 60000
    return type(base)(vocabulary)


def fake_spent_keys(ledger) -> tuple[str, ...]:
    """The fake world's patch-path keys: for each production key's frame, the first key of that frame in the fake's 020
    ledger whose cue is a word (the fake's 020 cues are not the real ones)."""
    keys = []
    for real in cr.PATCH_PATH_SPENT_KEYS:
        frame_id = real.split("|")[0]
        candidates = sorted(key for key in ledger if key.split("|")[0] == frame_id and key.split("|")[1].isalpha() and key.split("|")[1] != "ref")
        assert candidates, frame_id
        keys.append(candidates[0])
    return tuple(keys)


@pytest.fixture(scope="module")
def base025(world, base023, base024, locked024, tmp_path_factory):
    """024 locked on the fake; the 025 world constants for that world: the inherited digests of its 024 files and of
    020's confirmation file, and its patch-path keys (already-executed keys of the fake's 020 ledger)."""
    root = tmp_path_factory.mktemp("base025")
    shutil.copytree(locked024, root, dirs_exist_ok=True)
    inherited = {}
    for kind, relative in cr.INHERITED_024_PATHS.items():
        inherited[f"{kind}_file_sha256"] = rc.file_sha256(root / relative)
        inherited[f"{kind}_content_sha256"] = json.loads((root / relative).read_text())["content_sha256"]
    prior_path = root / cr.PRIOR_NOUNS_020["path"]
    prior = {"file_sha256": rc.file_sha256(prior_path), "nouns": len(json.loads(prior_path.read_text())["nouns"])}
    with pytest.MonkeyPatch.context() as patch:
        apply_024(patch, world, base023, base024)
        ledger = make_runner_024(root, world, FAKE_024)[0]._inputs().closure["ledger"]
    return {"root": root, "inherited": inherited, "prior": prior, "spent": fake_spent_keys(ledger)}


def _apply(patch, world, base, base24, base25) -> None:
    apply_024(patch, world, base, base24)
    for key, value in base25["inherited"].items():
        patch.setitem(cr.INHERITED_024, key, value)
    for key, value in base25["prior"].items():
        patch.setitem(cr.PRIOR_NOUNS_020, key, value)
    patch.setattr(cr, "PATCH_PATH_SPENT_KEYS", base25["spent"])


def _unreadable_outputs(patch) -> None:
    """022's calibration table and 023's and 024's local outputs can never be opened by a 025 phase."""
    original_load, original_read_bytes, original_read_text = torch.load, Path.read_bytes, Path.read_text

    def forbidden(path) -> bool:
        text = str(path)
        return text.endswith("calibration-table.pt") or "outputs/experiment-023/" in text or "outputs/experiment-024/" in text

    def load(path, *args, **kwargs):
        if forbidden(path):
            raise AssertionError(f"a forbidden file was opened by a 025 phase: {path}")
        return original_load(path, *args, **kwargs)

    def read_bytes(self):
        if forbidden(self):
            raise AssertionError(f"a forbidden file was read by a 025 phase: {self}")
        return original_read_bytes(self)

    def read_text(self, *args, **kwargs):
        if forbidden(self):
            raise AssertionError(f"a forbidden file was read by a 025 phase: {self}")
        return original_read_text(self, *args, **kwargs)

    patch.setattr(torch, "load", load)
    patch.setattr(Path, "read_bytes", read_bytes)
    patch.setattr(Path, "read_text", read_text)


def make_runner(root: Path, world, config, **overrides):
    manifest, small, digests, lock_011, lock_012, lock_017 = world["fake"]
    logs: list[str] = []
    arguments = dict(root=root, results_path=root / "outputs/experiment-025/results.json", report_path=root / "outputs/experiment-025/report.md",
                     model_loader=lambda spec: make_fake_model(), tokenizer_loader=lambda spec: toy_tokenizer_025(manifest, small),
                     lock_011_loader=lambda path: dict(lock_011), lock_012_loader=lambda path: dict(lock_012), lock_017_loader=lambda path: dict(lock_017),
                     git_state=lambda: {"commit": COMMIT_A, "dirty": False}, versions=lambda: {"torch": "test"}, tracked=lambda path: True,
                     changed_paths=lambda commit: [], log=logs.append, config=config)
    arguments.update(overrides)
    return runner_module.Runner(**arguments), logs


def _stage(world, base, base24, base25, tmp_path_factory, name: str, source: Path, step) -> Path:
    root = tmp_path_factory.mktemp(name)
    shutil.copytree(source, root, dirs_exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        _apply(patch, world, base, base24, base25)
        _unreadable_outputs(patch)
        runner, logs = make_runner(root, world, FAKE)
        step(runner, logs, root, patch)
    return root


class ForwardSpy:
    """Counts every forward by its condition-tagged key (patched runs) or plain key (captures)."""

    def __init__(self, monkeypatch):
        self.patched: Counter[str] = Counter()
        self.plain: Counter[str] = Counter()
        original_patched, original_plain = pm.run_patched, pm.capture_prompt

        def patched(model, prompt, replacements, sources, *, capture_sites=()):
            self.patched[prompt.key] += 1
            return original_patched(model, prompt, replacements, sources, capture_sites=capture_sites)

        def plain(model, prompt, sites):
            self.plain[prompt.key] += 1
            return original_plain(model, prompt, sites)

        monkeypatch.setattr(pm, "run_patched", patched)
        monkeypatch.setattr(pm, "capture_prompt", plain)


@pytest.fixture(scope="module")
def frozen025(world, base023, base024, base025, tmp_path_factory):
    def step(runner, logs, root, patch):
        assert runner.freeze() == 0, logs[-2:]

    return _stage(world, base023, base024, base025, tmp_path_factory, "frozen025", base025["root"], step)


@pytest.fixture(scope="module")
def locked025(world, base023, base024, base025, frozen025, tmp_path_factory):
    def step(runner, logs, root, patch):
        assert runner.lock() == 0, logs[-4:]
        _install(runner.output("candidate-lock.json"), root / cr.LOCK_RELATIVE_PATH)
        _install(runner.output("candidate-preregistration.md"), root / cr.PREREGISTRATION_RELATIVE_PATH)

    return _stage(world, base023, base024, base025, tmp_path_factory, "locked025", frozen025, step)


@pytest.fixture(scope="module")
def confirmed025(world, base023, base024, base025, locked025, tmp_path_factory):
    """One real confirmation on the fake, with every forward counted and every state write observed."""
    counts: dict[str, Counter[str]] = {}
    writes: list[dict] = []

    def step(runner, logs, root, patch):
        spy = ForwardSpy(patch)
        original_write = runner_module.Runner._write

        def recording(self, state):
            confirmation = state.get("confirmation") or {}
            writes.append({"confirmation": sorted(confirmation), "results": sorted(confirmation.get("results") or {}),
                           "status": state["phases"]["confirm"]["status"], "ledger": len(state["executed_prompt_keys"])})
            return original_write(self, state)

        patch.setattr(runner_module.Runner, "_write", recording)
        assert runner.confirm() == 0, logs[-4:]
        counts["patched"], counts["plain"] = spy.patched, spy.plain

    root = _stage(world, base023, base024, base025, tmp_path_factory, "confirmed025", locked025, step)
    return {"root": root, "counts": counts, "writes": writes}


@pytest.fixture
def sandbox(world, base023, base024, base025, tmp_path, monkeypatch):
    """A per-test copy of a stage directory, with every patch the fake needs and the forbidden outputs unreadable."""
    _apply(monkeypatch, world, base023, base024, base025)
    _unreadable_outputs(monkeypatch)

    def copy(source: Path) -> Path:
        root = tmp_path / "root"
        shutil.copytree(source, root)
        return root

    return copy


def _state(runner):
    return rd.load_results_state(runner.results_path)


def _refuse(*args, **kwargs):
    raise AssertionError("a model or tokenizer was loaded where none may be")


# ---------------------------------------------------------------------------
# The parser, validate, freeze.


def test_parser_has_exactly_the_five_phases_no_option_and_production_only():
    parser = runner_module.build_parser()
    for phase in cr.PHASES:
        assert parser.parse_args([phase]).phase == phase
    for forbidden_args in (["calibrate"], ["confirm", "--config", "fake"], ["freeze", "--quota", "4"]):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden_args)
    assert runner_module.PHASES == ("validate", "freeze", "lock", "confirm", "report")
    assert runner_module.Runner().config is cr.PRODUCTION
    with pytest.raises(cr.PhaseError, match="non-production configuration"):
        runner_module.Runner(config=FAKE)
    assert len(FAKE.conditions) == 11 and len(FAKE.outcome_bearing) == 6


def test_validate_loads_no_model_and_refuses_a_tampered_024_file_or_pin(world, base025, sandbox, monkeypatch):
    root = sandbox(base025["root"])
    runner, logs = make_runner(root, world, FAKE, model_loader=_refuse, tokenizer_loader=_refuse)
    assert runner.validate() == 0, logs
    assert "not frozen yet" in logs[-1] and "no lock installed" in logs[-1]
    for relative in (rr.LOCK_RELATIVE_PATH, rr.CALIBRATION_RELATIVE_PATH, cr.PRIOR_NOUNS_020["path"]):
        path = root / relative
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        assert runner.validate() == 1 and ("not the reviewed file" in logs[-1] or "not the pinned file" in logs[-1]), logs[-1]
        path.write_bytes(original)
    assert runner.validate() == 0
    monkeypatch.setitem(cr.FROZEN_BLOBS, "readout_routing.py", "0" * 40)
    assert runner.validate() == 1 and "frozen modules" in logs[-1]


def test_freeze_is_tokenizer_only_and_its_stops_write_nothing(world, base025, sandbox):
    root = sandbox(base025["root"])
    target = root / cr.CONFIRMATION_RELATIVE_PATH
    deviating = dataclasses.replace(FAKE, expected_picks=(("adjective", ("cheerful", "anxious")), ("noun", ("soldier", "sailor"))))
    runner, logs = make_runner(root, world, deviating, model_loader=_refuse)
    assert runner.freeze() == 3 and "differ from the design's expected picks" in logs[-1]
    assert not target.exists() and not runner.results_path.exists()
    too_many = dataclasses.replace(FAKE, n_adjectives=30, expected_picks=(("adjective", tuple(f"w{k}" for k in range(30))), ("noun", ("soldier", "sailor"))))
    runner, logs = make_runner(root, world, too_many, model_loader=_refuse)
    assert runner.freeze() == 3 and "eligible adjectives 23 of 30" in logs[-1] and not target.exists()
    runner, logs = make_runner(root, world, FAKE, model_loader=_refuse)
    assert runner.freeze() == 0, logs
    payload = json.loads(target.read_text())
    assert payload["picks"] == {"adjective": ["anxious", "cheerful"], "noun": ["soldier", "sailor"]} and payload["picks_match_expected"] is True
    assert payload["configuration"] == FAKE.to_json() and len(payload["manifest"]["S2-TARGET"]) == 4 * 108 * 11
    assert payload["conditions"] == list(FAKE.conditions) and payload["manifest_sha256"] == pm.sha256_text(pm.canonical_json(payload["manifest"]))
    assert all(key.rsplit("|", 1)[1] in FAKE.conditions for key in payload["manifest"]["S2-TARGET"])
    assert not runner.results_path.exists()  # the freeze creates no results state
    with pytest.raises(cr.PhaseError, match="runs once"):
        runner.freeze()
    assert runner.validate() == 0 and "4752 condition-tagged keys" in logs[-1]
    other, logs = make_runner(root, world, dataclasses.replace(FAKE, k_controls=3), model_loader=_refuse)
    assert other.validate() == 1 and "different configuration" in logs[-1]
    payload["cues"][0]["token_id"] += 1
    target.write_text(pm.canonical_json(payload) + "\n")
    assert runner.validate() == 1


def test_the_tightened_prior_noun_rule_excludes_020s_confirmation_nouns(world, base025, sandbox):
    """``statue`` (one of 020's confirmation nouns, read by 021) is rejected by the prior-noun rule and never picked; the
    ninth noun is then the new list's first eligible word. Without the rule, statue would be picked."""
    root = sandbox(base025["root"])
    prior = cr.prior_nouns(root)
    statue = next(noun for noun in json.loads((root / cr.PRIOR_NOUNS_020["path"]).read_text())["nouns"] if noun["lexical_key"] == "statue")
    manifest, small = world["fake"][0], world["fake"][1]
    vocabulary = dict(toy_tokenizer_025(manifest, small).vocabulary)
    vocabulary[" statue"], vocabulary[" statues"] = int(statue["sg_ids"][0]), int(statue["pl_ids"][0])  # the pinned tokenizer's ids, as in 020's file
    tokenizer = type(toy_tokenizer_025(manifest, small))(vocabulary)
    runner, _ = make_runner(root, world, FAKE)
    base = runner._base()
    blocked = {key: frozenset(value["ids"]) for key, value in cr.blocked_sets(base.inputs.pool, base.inherited["confirmation"], prior).items()}
    ordinary = [singular for singular, _ in cr.ORDINARY_RESERVES if singular != "statue"]
    nine = dataclasses.replace(FAKE, n_nouns=9, expected_picks=(("adjective", ("anxious", "cheerful")), ("noun", (*ordinary, "author"))))
    selection = cr.select_cues(tokenizer, blocked, nine)
    reasons = [entry["reason"] for entry in selection["rejected"] if entry["candidate"] == "statue/statues"]
    assert reasons == [f"statue: token id {statue['sg_ids'][0]}: a form of a noun of 020's confirmation list (read by 021); statues: token id "
                       f"{statue['pl_ids'][0]}: a form of a noun of 020's confirmation list (read by 021)"]
    loose = {**blocked, "prior_nouns": frozenset()}
    with pytest.raises(cr.FreezeDeviation, match="statue"):
        cr.select_cues(tokenizer, loose, nine)


# ---------------------------------------------------------------------------
# Lock.


def test_lock_is_weights_only_enforces_the_geometry_and_runs_once(world, base025, frozen025, sandbox, monkeypatch):
    root = sandbox(frozen025)
    spy = ForwardSpy(monkeypatch)
    runner, logs = make_runner(root, world, FAKE, tokenizer_loader=_refuse)
    assert runner.lock() == 0, logs[-4:]
    assert not spy.patched and not spy.plain  # weights only
    state = _state(runner)
    lock = json.loads(runner.output("candidate-lock.json").read_text())
    assert lock["content_sha256"] == rc.content_digest(lock) and state["lock"]["content_sha256"] == lock["content_sha256"]
    assert state["phases"]["lock"]["status"] == "complete" and state["executed_prompt_keys"] == []
    assert lock["module"] == {"path": "src/neural_decompiler/cue_rotation.py", "blob": cr.own_blob()} and lock["module_blobs"] == cr.FROZEN_BLOBS
    assert lock["patch_path_spent_keys"] == list(base025["spent"]) and lock["configuration"] == FAKE.to_json()
    geometry = lock["geometry"]
    assert [cue["word"] for cue in geometry["cues"]] == ["anxious", "cheerful", "soldier", "sailor"] and geometry["conditions"] == list(FAKE.conditions)
    for name, limit in cr.GEOMETRY_LIMITS.items():
        assert geometry["check_maxima"][name] <= cr.TOLERANCES[limit], name
    assert all(cue["checks"]["base_equals_model_row"] is True for cue in geometry["cues"])
    assert all(abs(cue["tau"] * math.sin(cue["theta_primary"]) - 0.32) < 1e-12 for cue in geometry["cues"])
    assert runner.output("candidate-preregistration.md").read_text() == cr.render_preregistration(lock)
    with pytest.raises(cr.PhaseError, match="lock already written"):
        runner.lock()


def test_a_geometry_gate_failure_at_lock_is_an_incident_and_writes_no_lock(world, base025, frozen025, sandbox, monkeypatch):
    root = sandbox(frozen025)
    monkeypatch.setitem(cr.TOLERANCES, "patched", 0.0)  # the float32 casts can never meet a zero tolerance
    runner, logs = make_runner(root, world, FAKE)
    assert runner.lock() == 2 and "I7′" in logs[-1]
    state = _state(runner)
    assert not runner.output("candidate-lock.json").exists() and state["phases"]["lock"]["incidents"][-1]["commit"] == COMMIT_A
    with pytest.raises(cr.PhaseError, match="lock incident is recorded"):
        runner.lock()
    assert runner.report() == 0 and "a lock incident is recorded; no lock was written" in runner.report_path.read_text()


def test_lock_refuses_an_uncommitted_freeze_and_a_forbidden_key_in_the_ledger(world, base025, frozen025, sandbox):
    root = sandbox(frozen025)
    confirmation_path = root / cr.CONFIRMATION_RELATIVE_PATH
    runner, _ = make_runner(root, world, FAKE, model_loader=_refuse, tracked=lambda path: path != confirmation_path)
    with pytest.raises(cr.PhaseError, match="frozen and committed first"):
        runner.lock()
    runner, _ = make_runner(root, world, FAKE)
    assert runner.lock() == 0
    state = _state(runner)
    freeze_024 = json.loads((root / rr.CONFIRMATION_RELATIVE_PATH).read_text())
    for planted in (freeze_024["manifest"]["S2-TARGET"][0], freeze_024["manifest"]["S2-TARGET"][0] + "|noun+0.32"):  # a spent 024 key, plain or tagged
        state["executed_prompt_keys"] = [planted]
        rr.write_state_atomic(runner.results_path, state)
        with pytest.raises(cr.PhaseError, match="forbidden keys"):
            runner.confirm()
        assert runner.validate() == 1


# ---------------------------------------------------------------------------
# Confirm.


def test_confirm_runs_every_key_once_and_writes_the_result_in_one_write(world, base025, confirmed025, sandbox):
    root = confirmed025["root"]
    runner, logs = make_runner(root, world, FAKE)
    state = _state(runner)
    confirmation = runner._confirmation(runner._base())[0]
    tagged = confirmation.tagged_keys()
    assert state["phases"]["confirm"]["status"] == "complete" and "incident" not in state["confirmation"]
    assert set(state["executed_prompt_keys"]) == set(tagged) and len(tagged) == 4 * 108 * 11
    fresh = confirmation.manifest_keys()
    spent = base025["spent"]
    assert confirmed025["counts"]["patched"] == Counter({**{key: 11 for key in fresh}, **{key: 1 for key in spent}})
    assert confirmed025["counts"]["plain"] == Counter({key: 1 for key in spent})  # no fresh key ever runs unpatched
    check = state["phases"]["confirm"]["patch_path_check"]
    assert check["passed"] and check["keys"] == list(spent) and state["phases"]["confirm"]["I7"] == {"bitwise_equal": True, "differing": []}
    assert all(entry["plain_equals_patched_theta0"] and entry["embed_equals_resid_pre0"] for entry in check["checks"].values())
    assert not {"ell", "A", "B", "G", "d_attn", "level1"} & {key for entry in check["checks"].values() for key in entry}
    writes = confirmed025["writes"]
    first = next(index for index, write in enumerate(writes) if write["results"])
    assert writes[first]["status"] == "complete" and all(write["status"] == "running" for write in writes[:first])
    assert writes[first - 1]["results"] == [] and "gates" in writes[first - 1]["confirmation"]
    assert writes[0]["ledger"] == len(tagged) and writes[0]["confirmation"] == []  # the ledger is on disk before any measurement
    results = state["confirmation"]["results"]
    assert results["outcome"]["label"] in cr.OUTCOMES[1:] and {results[name]["threshold"] for name in "ABG"} == {3}
    assert [row["word"] for row in results["per_cue"]] == ["anxious", "cheerful", "soldier", "sailor"]
    for row, response in zip(results["per_cue"], results["responses"]):
        assert row["B"] == row["A"] - math.fsum(abs(value) for value in row["A_random"]) / 2
        by_condition = response["conditions"]
        assert response["word"] == row["word"] and list(by_condition) == list(FAKE.conditions) and all(entry["frames"] == 108 for entry in by_condition.values())
        assert row["A"] == 0.5 * (by_condition["noun+0.32"]["ell"] - by_condition["noun-0.32"]["ell"])
        assert row["G"] == 0.5 * (by_condition["noun+0.32"]["d_attn"] - by_condition["noun-0.32"]["d_attn"])
        assert row["A_random"] == [0.5 * (by_condition[f"rand{j}+0.32"]["ell"] - by_condition[f"rand{j}-0.32"]["ell"]) for j in (1, 2)]
    assert state["confirmation"]["c_recompute"] == {"max": 0.0, "at": "", "runs": len(tagged), "bitwise_equal": True}
    gates = state["confirmation"]["gates"]
    for name, tolerance in (("I1", "I1"), ("I3", "I3"), ("I4", "I4"), ("level1_outcome_bearing", "level1")):
        assert gates[name]["max"] is not None and gates[name]["max"] <= cr.TOLERANCES[tolerance]
    assert state["confirmation"]["accounting"] == {"manifest": len(tagged), "executed": len(tagged), "ledger": len(tagged), "equal": True}
    assert set(state["confirmation"]["descriptives"]) == {"records", "ladder"}
    saved = torch.load(runner.stage2_path)
    assert {key: rc.tensor_digest(value) for key, value in saved.items()} == state["confirmation"]["stage2"]["tensors_sha256"]


def test_the_base_condition_is_the_unpatched_forward_and_every_rotation_lands(world, confirmed025, sandbox):
    """The θ = 0 run's saved captures are a plain forward's, bit for bit (on the fake, where fresh keys are free); every
    rotated condition's cue-position capture differs from it."""
    root = sandbox(confirmed025["root"])
    runner, _ = make_runner(root, world, FAKE)
    confirmation = runner._confirmation(runner._base())[0]
    units = cr.target_units(confirmation)
    saved = torch.load(runner.stage2_path)
    model = make_fake_model()
    for group, pairs in units.pairs.items():
        for row in (0, len(pairs) - 1):
            t, f = pairs[row]
            frame, token = units.frames[f], units.tokens[t]
            plain = pm.capture_prompt(model, pm.Prompt(frame, int(token["token_id"]), token["word"]), ul.measured_sites(frame))
            for name, layer in (("x1", "RESID_PRE.L1"), ("x3", f"RESID_PRE.L{hp.HEAD_LAYER}")):
                expected = torch.stack([plain.vector((layer, position)).detach().to(torch.float32) for position in (frame.p_c, frame.p_t)])
                assert torch.equal(saved[cr.tensor_name(group, "base", name)][row], expected), (group, row, name)
            for condition in confirmation.conditions[1:]:
                assert not torch.equal(saved[cr.tensor_name(group, condition, "x1")][row, 0], saved[cr.tensor_name(group, "base", "x1")][row, 0]), condition


def test_the_scores_are_recomputed_from_the_saved_measurements(world, base025, confirmed025, sandbox, monkeypatch):
    """A, B and G recomputed independently of the runner from the saved tensors reproduce the recorded ones exactly."""
    root = sandbox(confirmed025["root"])
    runner, _ = make_runner(root, world, FAKE)
    state = _state(runner)
    base = runner._base()
    confirmation = runner._confirmation(base)[0]
    units = cr.target_units(confirmation)
    states = ul.y1_states(base.inputs.closure["exploration"]["locked_states"], units.frames)
    saved = torch.load(runner.stage2_path)
    values = cr.per_cue(units, saved, states, confirmation.conditions)
    assert cr.statistics(values, confirmation, FAKE) == state["confirmation"]["results"]


def test_a_second_confirm_is_refused(world, confirmed025, sandbox):
    root = sandbox(confirmed025["root"])
    runner, _ = make_runner(root, world, FAKE, model_loader=_refuse)
    with pytest.raises(cr.PhaseError, match="confirm already started"):
        runner.confirm()


def test_an_i7_drift_is_an_incident_before_any_fresh_key(world, locked025, sandbox, monkeypatch):
    root = sandbox(locked025)
    spy = ForwardSpy(monkeypatch)
    original = cr.geometry_block

    def drifted(*args, **kwargs):
        out = original(*args, **kwargs)
        out["block"]["cues"][0]["s0"] = out["block"]["cues"][0]["s0"] + 1e-12
        return out

    monkeypatch.setattr(cr, "geometry_block", drifted)
    runner, logs = make_runner(root, world, FAKE)
    assert runner.confirm() == 2 and "I7′ failed" in logs[-1]
    state = _state(runner)
    assert state["executed_prompt_keys"] == [] and state["phases"]["confirm"]["incidents"][-1]["commit"] == COMMIT_A
    assert state["confirmation"] is None and not spy.patched and not spy.plain
    with pytest.raises(cr.PhaseError, match="incident is recorded"):
        runner.confirm()
    assert runner.report() == 0 and "confirm incident" in runner.report_path.read_text() and "`NOT_INTERPRETABLE`" in runner.report_path.read_text()


def test_a_patch_path_failure_is_an_incident_before_any_fresh_key(world, base025, locked025, sandbox, monkeypatch):
    root = sandbox(locked025)
    spy = ForwardSpy(monkeypatch)
    original = pm.run_patched

    def perturbed(model, prompt, replacements, sources, *, capture_sites=()):
        run = original(model, prompt, replacements, sources, capture_sites=capture_sites)
        return dataclasses.replace(run, logits=run.logits + 1e-6) if prompt.key in base025["spent"] else run

    monkeypatch.setattr(pm, "run_patched", perturbed)
    runner, logs = make_runner(root, world, FAKE)
    assert runner.confirm() == 2 and "patch-path check failed" in logs[-1]
    state = _state(runner)
    assert state["executed_prompt_keys"] == [] and state["confirmation"] is None
    recorded = state["phases"]["confirm"]["patch_path_check"]
    assert recorded["passed"] is False and recorded["keys"] == list(base025["spent"]) and not any(entry["plain_equals_patched_theta0"] for entry in recorded["checks"].values())
    assert set(spy.patched) <= set(base025["spent"]) and set(spy.plain) <= set(base025["spent"])
    with pytest.raises(cr.PhaseError, match="incident is recorded"):
        runner.confirm()


def test_an_exception_inside_the_patch_path_check_is_an_incident_before_any_fresh_key(world, locked025, sandbox, monkeypatch):
    root = sandbox(locked025)
    monkeypatch.setattr(cr, "patch_path_check", lambda *args, **kwargs: {}["missing frame"])
    spy = ForwardSpy(monkeypatch)
    runner, logs = make_runner(root, world, FAKE)
    assert runner.confirm() == 2 and "the patch-path check could not run: KeyError" in logs[-1]
    state = _state(runner)
    assert state["executed_prompt_keys"] == [] and state["phases"]["confirm"]["incidents"] and not spy.patched and not spy.plain
    with pytest.raises(cr.PhaseError, match="incident is recorded"):
        runner.confirm()


def test_a_patch_path_key_that_was_never_executed_is_an_incident(world, base025, locked025, sandbox, monkeypatch):
    root = sandbox(locked025)
    runner, _ = make_runner(root, world, FAKE)
    confirmation = runner._confirmation(runner._base())[0]
    fresh_key = sorted(confirmation.manifest_keys())[0]
    monkeypatch.setattr(cr, "PATCH_PATH_SPENT_KEYS", (fresh_key,))
    spy = ForwardSpy(monkeypatch)
    lock = json.loads((root / cr.LOCK_RELATIVE_PATH).read_text())
    with pytest.raises(cr.PhaseError, match="patch-path key"):
        runner.confirm()  # the lock binds the four keys: a changed key list is refused before the model
    assert lock["patch_path_spent_keys"] == list(base025["spent"]) and not spy.patched
    frames = {frame.frame_id: frame for frame in runner._base().inputs.pool.frames}
    with pytest.raises(pm.IncidentError, match="not an already-executed key"):
        cr.patch_path_check(None, frames, torch.zeros(1), frozenset(base025["spent"]), (fresh_key,))
    assert not spy.patched and not spy.plain


def _raise(*args, **kwargs):
    raise RuntimeError("forced descriptive failure")


def _reuse_gates(monkeypatch, world, confirmed025, failing: str | None = None) -> None:
    """The failure-path tests reuse the confirmed fake run's gates (the fake is deterministic: the same world gives the
    same values) instead of recomputing I1, I3, I4 and the Level-1 identity over every run; ``failing`` forces one."""
    gates = _state(make_runner(confirmed025["root"], world, FAKE)[0])["confirmation"]["gates"]
    if failing is not None:
        gates = {**gates, failing: {"max": 1.0, "at": "forced"}}
    monkeypatch.setattr(cr, "run_gates", lambda *args, **kwargs: json.loads(json.dumps(gates)))


@pytest.mark.parametrize("failure", ["accounting", "c_recompute", "gate", "level1", "recheck"])
def test_a_post_measurement_incident_keeps_every_measurement_and_carries_no_result(world, locked025, confirmed025, sandbox, monkeypatch, failure):
    root = sandbox(locked025)
    runner, logs = make_runner(root, world, FAKE)
    if failure == "accounting":
        original = cr.stage_two

        def dropped(*args, executed, **kwargs):
            out = original(*args, executed=executed, **kwargs)
            executed.pop()
            return out

        monkeypatch.setattr(cr, "stage_two", dropped)
    elif failure == "c_recompute":
        monkeypatch.setattr(cr, "recompute_c", lambda *args, **kwargs: {"max": 1.0, "at": "forced", "runs": 1, "bitwise_equal": False})
    elif failure in ("gate", "level1"):
        _reuse_gates(monkeypatch, world, confirmed025, "I3" if failure == "gate" else "level1_outcome_bearing")
    else:
        _reuse_gates(monkeypatch, world, confirmed025)
        monkeypatch.setattr(runner_module.Runner, "_recheck", lambda self: {"ok": False, "message": "forced"})
    assert runner.confirm() == 2, logs[-3:]
    state = _state(runner)
    assert state["confirmation"]["incident"]["commit"] == COMMIT_A and "results" not in state["confirmation"]
    assert state["phases"]["confirm"]["status"] == "running" and runner.stage2_path.exists() and state["confirmation"]["stage2"]["n_executed"] > 0
    assert len(state["executed_prompt_keys"]) == 4 * 108 * 11
    with pytest.raises(cr.PhaseError, match="confirm already started"):
        runner.confirm()
    assert runner.report() == 0 and "NOT_INTERPRETABLE" in runner.report_path.read_text()


def test_a_per_run_integrity_failure_during_stage_two_keeps_the_runs_measured_so_far(world, locked025, sandbox, monkeypatch):
    root = sandbox(locked025)
    runner, logs = make_runner(root, world, FAKE)
    original, calls = cr.measure_rotated, []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 100:
            raise pm.IncidentError("forced: EMBED@3: inexact replacement")
        return original(*args, **kwargs)

    monkeypatch.setattr(cr, "measure_rotated", failing)
    writes: list[dict] = []
    original_write = runner_module.Runner._write

    def recording(self, state):
        confirmation = state.get("confirmation") or {}
        writes.append({"incident": "incident" in confirmation, "partial": "stage2_partial" in confirmation})
        return original_write(self, state)

    monkeypatch.setattr(runner_module.Runner, "_write", recording)
    assert runner.confirm() == 2 and "inexact replacement" in logs[-1]
    first_partial = next(index for index, write in enumerate(writes) if write["partial"])
    assert writes[first_partial - 1] == {"incident": True, "partial": False}  # the incident is on disk before the slow save
    state = _state(runner)
    partial = state["confirmation"]["stage2_partial"]
    assert state["confirmation"]["incident"]["type"] == "IncidentError" and "results" not in state["confirmation"] and "stage2" not in state["confirmation"]
    assert partial["n_executed"] == 99 and partial["last_key"].endswith("|" + FAKE.conditions[(99 - 1) % 11]) and len(state["executed_prompt_keys"]) == 4 * 108 * 11
    saved = torch.load(runner.output("stage2-partial.pt"))
    assert {key: rc.tensor_digest(value) for key, value in saved.items()} == partial["tensors_sha256"] and not runner.stage2_path.exists()
    assert runner.report() == 0 and "NOT_INTERPRETABLE" in runner.report_path.read_text()


def test_an_interruption_during_stage_two_or_its_partial_save_is_recorded_and_raised(world, locked025, sandbox, monkeypatch):
    root = sandbox(locked025)
    runner, _ = make_runner(root, world, FAKE)
    original, calls = cr.measure_rotated, []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 50:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(cr, "measure_rotated", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.confirm()
    state = _state(runner)
    assert state["confirmation"]["incident"]["type"] == "KeyboardInterrupt" and state["confirmation"]["stage2_partial"]["n_executed"] == 49
    with pytest.raises(cr.PhaseError, match="confirm already started"):
        runner.confirm()


def test_an_interrupted_partial_save_keeps_the_incident(world, locked025, sandbox, monkeypatch):
    root = sandbox(locked025)
    runner, _ = make_runner(root, world, FAKE)
    original, calls = cr.measure_rotated, []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 30:
            raise pm.IncidentError("forced: EMBED@3: inexact replacement")
        return original(*args, **kwargs)

    original_save = rr.save_durably

    def save(value, path):
        if Path(path).name == "stage2-partial.pt":
            raise KeyboardInterrupt
        return original_save(value, path)

    monkeypatch.setattr(cr, "measure_rotated", failing)
    monkeypatch.setattr(rr, "save_durably", save)
    with pytest.raises(KeyboardInterrupt):
        runner.confirm()
    state = _state(runner)
    assert "inexact replacement" in state["confirmation"]["incident"]["message"] and "results" not in state["confirmation"]
    assert state["confirmation"]["stage2_partial"] == {"path": str(runner.output("stage2-partial.pt")), "save_failed": "KeyboardInterrupt: ", "n_executed": 29}


def test_a_failed_result_write_is_an_incident_with_no_result(world, locked025, confirmed025, sandbox, monkeypatch):
    root = sandbox(locked025)
    runner, logs = make_runner(root, world, FAKE)
    _reuse_gates(monkeypatch, world, confirmed025)
    original_write = runner_module.Runner._write

    def failing(self, state):
        if (state.get("confirmation") or {}).get("results") and state["phases"]["confirm"]["status"] == "complete":
            raise OSError("forced: the disk refused the result write")
        return original_write(self, state)

    monkeypatch.setattr(runner_module.Runner, "_write", failing)
    assert runner.confirm() == 2
    state = _state(runner)
    assert "forced" in state["confirmation"]["incident"]["message"] and "results" not in state["confirmation"]
    assert state["phases"]["confirm"]["status"] == "running"


def test_a_descriptive_failure_keeps_the_result_unchanged(world, locked025, confirmed025, sandbox, monkeypatch):
    root = sandbox(locked025)
    runner, logs = make_runner(root, world, FAKE)
    _reuse_gates(monkeypatch, world, confirmed025)
    monkeypatch.setattr(cr, "descriptives", _raise)
    assert runner.confirm() == 0
    state = _state(runner)
    reference = _state(make_runner(confirmed025["root"], world, FAKE)[0])
    assert state["phases"]["confirm"]["status"] == "complete" and state["confirmation"]["results"] == reference["confirmation"]["results"]
    assert state["confirmation"]["descriptives"]["failures"]["records"]["message"] == "forced descriptive failure"
    assert "ladder" in state["confirmation"]["descriptives"] and "records" not in state["confirmation"]["descriptives"]
    assert runner.report() == 0 and "was not computed" in runner.report_path.read_text()


def test_confirm_refuses_a_tampered_uncommitted_or_changed_lock_before_the_model(world, locked025, sandbox, monkeypatch):
    root = sandbox(locked025)
    lock_path = root / cr.LOCK_RELATIVE_PATH
    original = lock_path.read_text()
    for runner_overrides, expected in ((dict(tracked=lambda path: path.name != "preregistration-lock.json"), "tracked"),
                                       (dict(git_state=lambda: {"commit": COMMIT_A, "dirty": True}), "clean"),
                                       (dict(changed_paths=lambda commit: ["src/neural_decompiler/cue_rotation.py"]), "scientific paths changed"),
                                       (dict(changed_paths=lambda commit: [f"{cr.EXPERIMENT_DIR}/run.py"]), "scientific paths changed"),
                                       (dict(changed_paths=lambda commit: [rr.CALIBRATION_RELATIVE_PATH]), "scientific paths changed"),
                                       (dict(changed_paths=lambda commit: None), "ancestor")):
        runner, _ = make_runner(root, world, FAKE, model_loader=_refuse, **runner_overrides)
        with pytest.raises(cr.PhaseError, match=expected):
            runner.confirm()
    runner, _ = make_runner(root, world, FAKE, model_loader=_refuse, changed_paths=lambda commit: [f"{cr.EXPERIMENT_DIR}/README.md", "docs/x.md",
                                                                                                     f"{cr.EXPERIMENT_DIR}/evidence/lock/REVIEW.md"])
    with pytest.raises(AssertionError, match="a model or tokenizer was loaded"):
        runner.confirm()  # non-scientific changes pass every check up to the model load
    lock = json.loads(original)
    lock["module"]["blob"] = "0" * 40
    lock["content_sha256"] = rc.content_digest(lock)
    lock_path.write_text(pm.canonical_json(lock) + "\n")
    runner, _ = make_runner(root, world, FAKE, model_loader=_refuse)
    with pytest.raises(cr.PhaseError, match="not the candidate this run wrote"):
        runner.confirm()
    lock_path.write_text(original)
    monkeypatch.setattr(cr, "own_blob", lambda: "0" * 40)  # the running module is not the one the lock binds
    with pytest.raises(cr.PhaseError, match="direct module-blob check"):
        runner.confirm()
    assert _state(runner)["executed_prompt_keys"] == [] and _state(runner)["phases"]["confirm"]["status"] == "not_started"


def test_confirm_refuses_drifted_weights_before_any_prompt(world, locked025, sandbox, monkeypatch):
    root = sandbox(locked025)
    spy = ForwardSpy(monkeypatch)
    monkeypatch.setattr(rr, "parameters_digest", lambda model: "0" * 64)
    runner, _ = make_runner(root, world, FAKE)
    with pytest.raises(cr.PhaseError, match="scientific dependencies differ"):
        runner.confirm()
    assert not spy.patched and not spy.plain and _state(runner)["executed_prompt_keys"] == []


# ---------------------------------------------------------------------------
# Report.


def test_report_renders_the_outcome_and_the_counts(world, confirmed025, sandbox):
    root = sandbox(confirmed025["root"])
    runner, logs = make_runner(root, world, FAKE, model_loader=_refuse, tokenizer_loader=_refuse)
    assert runner.report() == 0
    text = runner.report_path.read_text()
    state = _state(runner)
    assert f"**`{state['confirmation']['results']['outcome']['label']}`**" in text and "| A |" in text and "| G |" in text
    assert cr.SEMANTICS["A"] in text and cr.SEMANTICS["B"] in text and cr.SEMANTICS["G"] in text and cr.SEMANTICS["patch_path"] in text
    assert cr.SEMANTICS["secondary"] in text and ("Concentration:" in text or "No stratum is below its reporting trigger" in text)
    assert "The patch-path check: passed True" in text and state["phases"]["report"]["status"] == "complete"
    assert state["report"]["sha256"] == pm.sha256_text(text)
    assert runner.validate() == 0 and "lock and preregistration verified" in logs[-1]


def test_report_before_confirm_is_refused(world, locked025, sandbox):
    root = sandbox(locked025)
    runner, _ = make_runner(root, world, FAKE, model_loader=_refuse)
    with pytest.raises(cr.PhaseError, match="report requires"):
        runner.report()
