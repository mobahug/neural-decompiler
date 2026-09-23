"""Experiment 021's pure core, without a model: the frozen constants, the program blob, Experiment 020's closure
verification, the provenance pools, the admissible rows, the SHA-indexed draws, the statistics kernel against
Experiment 020's own statistics, the floor rule, the one pass predicate, the row selections, the scoring branches, the
reproduction gate and the calibration record (including a failing kernel/direct cross-check)."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import shutil
import sys
from pathlib import Path

import pytest
import torch

from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd

ROOT = Path(__file__).parents[1]
RESULTS_020 = ROOT / rc.RESULTS_020_RELATIVE_PATH


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("experiment_021_runner_for_units", ROOT / "experiments/021-corrected-readout-confirmation/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frozen():
    """The real exposed pool, digests and confirmation set, from the committed inputs only (no model)."""
    runner = _load_runner_module().Runner()
    pool, lock_011, lock_012, lock_017, digests = runner._base_inputs()
    confirmation = rd.load_confirmation(ROOT / rd.CONFIRMATION_RELATIVE_PATH, pool, digests)
    return pool, digests, confirmation


# ---------------------------------------------------------------------------
# Synthetic worlds: a table over every cue × frame pair, and matching pools.


def synthetic_table(*, cues_per_class=3, frames_per_template=4, nouns_per_rule=4, seed=0) -> tuple[rc.ExposedTable, rc.CalibrationPools]:
    generator = torch.Generator().manual_seed(seed)
    cues = {cls: [f"{cls[:4]}{i}" for i in range(cues_per_class)] for cls in rc.CUE_CLASSES}
    frames = {template: [f"{template}-s-{i}" for i in range(frames_per_template)] for template in rc.Y2_AXES}
    nouns = {rule: [f"{rule[:4]}{i}" for i in range(nouns_per_rule)] for rule in rc.RULE_CLASSES}
    cue_list = [cue for cls in rc.CUE_CLASSES for cue in cues[cls]]
    frame_list = [(frame, template) for template in rc.Y2_AXES for frame in frames[template]]
    noun_list = [noun for rule in rc.RULE_CLASSES for noun in nouns[rule]]
    pairs = [(cue, frame, template) for frame, template in frame_list for cue in cue_list]
    n = len(pairs)
    signal = torch.randn(n, 1, generator=generator, dtype=torch.float64) * 2.0
    loadings = 1.0 + 0.3 * torch.randn(1, len(noun_list), generator=generator, dtype=torch.float64)
    measured = signal * loadings + 0.4 * torch.randn(n, len(noun_list), generator=generator, dtype=torch.float64)
    level0 = 0.85 * measured + 0.5 * torch.randn(n, len(noun_list), generator=generator, dtype=torch.float64)
    table = rc.ExposedTable(tuple(p[0] for p in pairs), tuple(p[1] for p in pairs), tuple(p[2] for p in pairs), tuple(noun_list), measured, level0,
                            measured + 0.05 * torch.randn(n, len(noun_list), generator=generator, dtype=torch.float64), 0.7 * measured, 0.6 * measured,
                            torch.randn(n, generator=generator, dtype=torch.float64))
    cue_entries = {cls: tuple((cue, 1000 + 10 * k + i) for i, cue in enumerate(cues[cls])) for k, cls in enumerate(rc.CUE_CLASSES)}
    pools = rc.CalibrationPools(cue_entries, {cls: cue_entries[cls][:1] for cls in rc.CUE_CLASSES}, {"cohort-a": cue_entries["quantity"], "cohort-b": cue_entries["adjective"]},
                                {template: tuple(frames[template][:3]) for template in rc.Y2_AXES}, {template: tuple(frames[template]) for template in rc.Y2_AXES},
                                {rule: tuple(nouns[rule]) for rule in rc.RULE_CLASSES})
    return table, pools.screened({frame: True for template in rc.Y2_AXES for frame in frames[template]})


def recorded_exploration(table: rc.ExposedTable) -> dict:
    """What Experiment 020 recorded, recomputed from a table and passed through JSON as 020's state was."""
    scoring = table.scoring("level0")
    return json.loads(pm.canonical_json({
        "statistics": rd.pair_statistics(scoring), "nouns": rd.noun_statistics(scoring),
        "comparators": {"ceiling_measured_dx3": rd.pair_statistics(table.scoring("ceiling"))["flattened_r2"], "no_l5_heads": rd.pair_statistics(table.scoring("no_l5"))["flattened_r2"],
                        "template_base_mlps": rd.pair_statistics(table.scoring("base"))["flattened_r2"], "dT_only": rd.dT_only_fit(table.dT, scoring), "standing": dict(rd.COMPARATOR_STANDING)}}))


def write_closed_020(root: Path, *, state: dict, confirmation) -> dict[str, str]:
    """A closure record, an evidence extract and a results state in the committed format; returns their digests."""
    results_path = root / rc.RESULTS_020_RELATIVE_PATH
    state_digest = rd.write_results_state(results_path, state)
    loaded = rd.load_results_state(results_path)
    extract = {"experiment": "020", "exploration": {**{k: v for k, v in loaded["exploration"].items() if k != "locked_states"},
                                                    "locked_state_digests": {frame_id: rd.state_digest(entry) for frame_id, entry in sorted(loaded["exploration"]["locked_states"].items())}}}
    extract["content_sha256"] = rc.content_digest(extract)
    (root / rc.EXTRACT_020_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / rc.EXTRACT_020_RELATIVE_PATH).write_text(pm.canonical_json(extract) + "\n", encoding="utf-8")
    closure = {"experiment": "020", "status": "closed", "labels": None, "lock": None, "confirmation": None,
               "explore": {"results_state_sha256": state_digest, "results_file_sha256": rc.file_sha256(results_path)},
               "confirmation_set": {"content_sha256": confirmation.content_sha256, "executed_keys": 0},
               "evidence": {"exploration_record_content_sha256": extract["content_sha256"]}}
    closure["content_sha256"] = rc.content_digest(closure)
    (root / rc.CLOSURE_020_RELATIVE_PATH).write_text(json.dumps(closure, indent=2) + "\n", encoding="utf-8")
    return {"CLOSURE_020_SHA256": closure["content_sha256"], "EXTRACT_020_SHA256": extract["content_sha256"], "RESULTS_020_FILE_SHA256": rc.file_sha256(results_path),
            "RESULTS_020_STATE_SHA256": state_digest}


# ---------------------------------------------------------------------------
# 1–3: constants, the program blob, Experiment 020's closure.


def test_frozen_constants_are_design_revision_3():
    assert rc.DESIGN == {"path": "docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md", "revision": 3, "commit": "0ac46aa"}
    assert rc.B == 10_000 and rc.tail_index(rc.B) == 250 and rc.ALPHA_PER_MILLE == 25 and rc.tail_index(40) == 1
    assert rc.SHARES == {"cue": 80, "frame": 75, "noun": 90} and rc.SLOTS == {"cue": 6, "frame": 6, "noun": 8}
    assert rc.VARIANTS == ("primary", "all-frames", "lineage") and rc.FULL_ROWS == {"Y1": (), "Y2": (6, 6, 6), "Y3": (8, 8, 8)}
    assert rc.RECONSTRUCTION_TOLERANCE == 1e-9 and rc.LOCKED_STATE_TOLERANCE == 1e-9 and rc.MIN_SCREENED_FRAMES_PER_TEMPLATE == 6
    assert rc.STATISTIC_AGREEMENT_TOLERANCE == 1e-10 and rc.CROSS_CHECK_DRAWS == 16  # implementation-only guards
    assert rc.R2_STATISTICS == {"token_mean_r2", "pair_mean_r2", "frame_mean_r2", "frame_r2_k75", "cue_final_r2", "coordinated_r2", "noun_median_r2", "noun_r2_k90"}
    assert rc.ERROR_STATISTICS == {"cue_mae_k80", "pooled_mae", "noun_slope_dev_k90", "noun_bias_k90"} and not (rc.R2_STATISTICS & rc.ERROR_STATISTICS)
    assert rc.PROGRAM_BLOB_SHA1 == "caa73b40192f4c910dc63371bd19db75a3258339" and rc.CONFIRMATION_020_SHA256.startswith("e098e2b4")
    assert (rc.CLOSURE_020_SHA256[:8], rc.EXTRACT_020_SHA256[:8], rc.RESULTS_020_FILE_SHA256[:8], rc.RESULTS_020_STATE_SHA256[:8]) == ("f2b1b5e7", "99f25ee6", "da63b8c2", "2e5485dc")
    # every inherited rule is Experiment 020's own constant, never redefined here
    for name in ("MIN_SCORED_TOKENS", "MIN_VALID_EXPOSED_FRAMES", "MIN_VALID_FRESH_FRAMES", "MIN_VALID_COORDINATED_FRAMES", "MIN_SCORABLE_FRESH_NOUNS", "LEVEL1_TOLERANCE",
                 "OUTCOME_Y1", "OUTCOME_Y2", "OUTCOME_Y3", "POPULATIONS", "COMPARATOR_STANDING"):
        assert not hasattr(rc, name), name
    assert (rd.MIN_VALID_FRESH_FRAMES, rd.MIN_VALID_COORDINATED_FRAMES, rd.MIN_SCORABLE_FRESH_NOUNS, rd.LEVEL1_TOLERANCE) == (12, 4, 18, 7e-3)


def test_the_inherited_locks_are_bound_by_their_exact_digests(frozen):
    pool, digests, confirmation = frozen
    for key, path in (("lock_011", rd.EXPERIMENT_011_LOCK_PATH), ("lock_012", rd.EXPERIMENT_012_LOCK_PATH), ("lock_017", rd.EXPERIMENT_017_LOCK_PATH)):
        committed = json.loads((ROOT / path).read_text())
        assert committed["content_sha256"] == rc.content_digest(committed) == rc.LOCK_DIGESTS[key] == digests[key]  # the real input chain binds them
    rc.assert_lock_digests(digests)
    for key in rc.LOCK_DIGESTS:  # a replaced lock is refused, however self-consistent its own digest
        with pytest.raises(rd.PhaseError, match=key):
            rc.assert_lock_digests(dict(digests, **{key: "1" * 64}))
    assert {"lock_011", "lock_012", "lock_017"} <= set(rc.DIGEST_KEYS)
    full = {key: digests.get(key, f"{key}-digest") for key in rc.DIGEST_KEYS}
    state = rc.new_results_state(digests=full, protocol_code_commit="a" * 40, git_dirty=False, versions={})
    assert all(state[f"{key}_sha256"] == rc.LOCK_DIGESTS[key] for key in rc.LOCK_DIGESTS) and rc.state_digests(state) == rc.digest_tuple(full)
    assert rc.state_digests(state) != rc.digest_tuple(dict(full, lock_017="1" * 64))  # a later phase would refuse the recorded state
    record = {"inputs": dict(full)}
    rc.assert_record_inputs(record, full)
    for key in rc.LOCK_DIGESTS:
        with pytest.raises(rd.PhaseError, match="different inputs"):
            rc.assert_record_inputs(record, dict(full, **{key: "1" * 64}))


def test_the_program_is_experiment_020s_blob(tmp_path):
    assert rc.program_blob_sha1(Path(rd.__file__)) == rc.PROGRAM_BLOB_SHA1 == rc.assert_program_blob()
    copy = tmp_path / "readout_decompilation.py"
    copy.write_bytes(Path(rd.__file__).read_bytes() + b"\n")
    with pytest.raises(rd.PhaseError, match="blob"):
        rc.assert_program_blob(copy)
    data = Path(rd.__file__).read_bytes()
    assert rc.program_blob_sha1(Path(rd.__file__)) == hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_the_committed_closure_verifies_and_every_tampering_is_refused(tmp_path, frozen, monkeypatch):
    pool, digests, confirmation = frozen
    closure = json.loads((ROOT / rc.CLOSURE_020_RELATIVE_PATH).read_text())
    extract = json.loads((ROOT / rc.EXTRACT_020_RELATIVE_PATH).read_text())
    assert closure["content_sha256"] == rc.content_digest(closure) == rc.CLOSURE_020_SHA256
    assert extract["content_sha256"] == rc.content_digest(extract) == rc.EXTRACT_020_SHA256
    if RESULTS_020.exists():  # gitignored: present on the machine that ran Experiment 020
        verified = rc.verify_020_closure(ROOT, confirmation)
        assert len(verified["ledger"]) == 30_132 and verified["digests"]["results_020_state"] == rc.RESULTS_020_STATE_SHA256
    # a small closed world in the committed format, with the frozen digest constants pointed at it
    state = rd.new_results_state(digests={key: digests[key] for key in rd.CONFIRMATION_DIGEST_KEYS} | {"confirmation_020": confirmation.content_sha256},
                                 protocol_code_commit="a" * 40, git_dirty=False, versions={"torch": "test"})
    state["phases"]["explore"] = {"status": "complete"}
    state["exploration"] = {"locked_states": {"f1": {"p_c": 1, "h6": [0.5, 0.25]}}, "statistics": {}}
    state["executed_prompt_keys"] = ["f1|a|247"]
    state["executed_noun_keys"] = [pool.nouns[0].key]
    constants = write_closed_020(tmp_path, state=state, confirmation=confirmation)
    for name, value in constants.items():
        monkeypatch.setattr(rc, name, value)
    assert rc.verify_020_closure(tmp_path, confirmation)["ledger"] == frozenset({"f1|a|247"})
    for broken, message in ((dict(state, executed_prompt_keys=["f1|a|247", confirmation.all_prompts[0].key]), "confirmation prompts"),
                            (dict(state, executed_noun_keys=[confirmation.nouns[0].key]), "fresh noun"),
                            (dict(state, lock={"content_sha256": "x"}), "without lock")):
        constants = write_closed_020(tmp_path, state=dict(broken), confirmation=confirmation)
        for name, value in constants.items():
            monkeypatch.setattr(rc, name, value)
        with pytest.raises(rd.PhaseError, match=message):
            rc.verify_020_closure(tmp_path, confirmation)
    constants = write_closed_020(tmp_path, state=state, confirmation=confirmation)
    for name, value in constants.items():
        monkeypatch.setattr(rc, name, value)
    for path, mutate in ((rc.CLOSURE_020_RELATIVE_PATH, lambda d: d.update(status="open")), (rc.EXTRACT_020_RELATIVE_PATH, lambda d: d.update(experiment="x"))):
        original = (tmp_path / path).read_text()
        payload = json.loads(original)
        mutate(payload)
        (tmp_path / path).write_text(json.dumps(payload))
        with pytest.raises(rd.PhaseError):
            rc.verify_020_closure(tmp_path, confirmation)
        (tmp_path / path).write_text(original)
    (tmp_path / rc.RESULTS_020_RELATIVE_PATH).write_text((tmp_path / rc.RESULTS_020_RELATIVE_PATH).read_text().replace("f1|a|247", "f1|a|248"))
    with pytest.raises(rd.PhaseError, match="results file"):
        rc.verify_020_closure(tmp_path, confirmation)


# ---------------------------------------------------------------------------
# 4–8: pools, rows, k, draws, the prefix construction.


def test_pools_are_the_frozen_provenance_pools(frozen):
    pool, digests, confirmation = frozen
    pools = rc.production_pools(pool)
    reference_or_plural = set(pool.reference_ids.values()) | {pool.token_id(name) for name in pool.plural_cue.values()}
    for cls, entries in pools.cues.items():
        assert [token_id for _, token_id in entries] == sorted(token_id for _, token_id in entries)
        assert all(pool.token_category[word] == cls and pool.token_source[word] in rc.CUE_COHORTS and token_id not in reference_or_plural for word, token_id in entries)
    excluded_quantity = {word for word, _ in pool.tokens if pool.token_category[word] == "quantity" and pool.token_source[word] not in rc.CUE_COHORTS}
    assert excluded_quantity == {"dozen", "countless", "various", "fewer"} and not excluded_quantity & {word for word, _ in pools.cues["quantity"]}
    assert {cls: len(v) for cls, v in pools.lineage_cues.items()} == {"determiner-like": 15, "quantity": 15, "possessive-or-pronoun": 12, "adjective": 18}
    assert sorted(len(v) for v in pools.cohorts.values()) == [19] * 5 + [20] * 4 and sum(len(v) for v in pools.cohorts.values()) == 175
    origins = {pool.frame_origin[frame_id] for ids in pools.frames_unscreened.values() for frame_id in ids}
    assert origins == {"confirmation-017", "confirmation-018", "confirmation-019"} and all(list(ids) == sorted(ids) for ids in pools.frames_unscreened.values())
    assert {template: len(ids) for template, ids in pools.frames_all_unscreened.items()} == {template: 36 for template in rc.Y2_AXES}
    assert "peach" not in {key for keys in pools.nouns.values() for key in keys} and all(list(keys) == sorted(keys) for keys in pools.nouns.values())
    assert not {key for keys in pools.nouns.values() for key in keys} & {noun.lexical_key for noun in confirmation.nouns}
    with pytest.raises(rd.PhaseError, match="validity screen"):
        pools.frames()  # the frame pools exist only after the screen


def test_rows_are_exactly_the_admissible_compositions():
    y2, y3 = rc.y2_rows(), rc.y3_rows()
    assert len(y2) == 64 and len(y3) == 84 and len(set(y2)) == 64 and len(set(y3)) == 84
    assert [sum(1 for row in y2 if sum(row) == n) for n in range(12, 19)] == [18, 15, 12, 9, 6, 3, 1]
    assert [sum(1 for row in y3 if sum(row) == n) for n in range(18, 25)] == [28, 21, 15, 10, 6, 3, 1]
    every = [(a, b, c) for a in range(7) for b in range(7) for c in range(7)]
    assert set(y2) == {row for row in every if row[2] >= 4 and sum(row) >= 12}
    assert set(y3) == {(a, b, c) for a in range(9) for b in range(9) for c in range(9) if a + b + c >= 18}
    assert rc.row_key(()) == "all" and rc.row_key((6, 5, 4)) == "6/5/4"


def test_k_is_the_integer_form_of_the_share_conditions():
    assert rc.k_of(80, 24) == 20 and [rc.k_of(75, n) for n in range(12, 19)] == [9, 10, 11, 12, 12, 13, 14]
    assert [rc.k_of(90, n) for n in range(18, 25)] == [17, 18, 18, 19, 20, 21, 22]
    for percent in (80, 75, 90):
        for n in range(1, 41):
            assert rc.k_of(percent, n) == math.ceil(percent * n / 100)  # exact by construction; the float form agrees for these shares
            for count in range(n + 1):
                flags = [True] * count + [False] * (n - count)
                assert (count >= rc.k_of(percent, n)) == (rd._share(flags) >= percent / 100) == (100 * count >= percent * n)


PINNED_INDEX_DIGESTS = {
    "cue/determiner-like": "50f2462d447fcb36c173c1a6226bf360c47af7d2fddb753d2fde515639bbf099",
    "cue/quantity": "ac9bdd6c64b0e28bc908173feb59bd75bcd6bca1dac5609905830f24f9003d33",
    "cue/possessive-or-pronoun": "fa287323809854fd8f0e380f00423453b2ef8802dd06161814b6afd486044bc8",
    "cue/adjective": "85938232c4a740016f270d502824381e089f85d33f3ca317b418faaa1c91729a",
    "frame/cardinal": "9d958f4af9197c5a439f361fbd2cf8a86a48dc8e11b26b38ae0ce2286bc6a851",
    "frame/quantifier": "a2801deb7af6608394b96218a12d624516d718393497faa4b2a262396782e7ee",
    "frame/coordinated-adjective": "72056ac0ab3cc114f06da67c08b749af9af148bdf22171cd719a3a759cd1026c",
    "noun/simple-suffix": "024c882acd921e8f0f054e83e76d3c73605ae102f93353cc769157ba1ace2250",
    "noun/sibilant-es": "e6e4e4bce9995765e0a71f84866a1802c98ac605a3011a76cd2e0b9a6a6b59d0",
    "noun/consonant-y": "18a44e4a8d29978700291f9fbd9e49d96f92745c27d47a171c6bc209f19e2587",
}


def test_draws_are_sha_indexed_deterministic_and_pinned():
    expected = int.from_bytes(hashlib.sha256("021|primary|0|cue/adjective|0".encode("utf-8")).digest()[:8], "big") % 49
    assert rc.slot_index("primary", 0, "cue/adjective", 0, 49) == expected
    assert rc.slot_index("primary", 9999, "noun/consonant-y", 7, 20) == int.from_bytes(hashlib.sha256(b"021|primary|9999|noun/consonant-y|7").digest()[:8], "big") % 20
    sizes = {**{f"cue/{c}": n for c, n in rc.EXPECTED_POOL_COUNTS["cues"].items()}, **{f"frame/{t}": 14 for t in rc.Y2_AXES},
             **{f"noun/{r}": n for r, n in rc.EXPECTED_POOL_COUNTS["nouns"].items()}}
    assert set(sizes) == set(PINNED_INDEX_DIGESTS)  # exactly the ten frozen stratum strings
    slots = {stratum: rc.SLOTS[stratum.split("/")[0]] for stratum in sizes}
    indices = rc.draw_indices("primary", sizes, slots, rc.B)
    assert {stratum: rc.tensor_digest(t) for stratum, t in indices.items()} == PINNED_INDEX_DIGESTS
    assert all(tuple(t.shape) == (rc.B, slots[s]) and int(t.min()) >= 0 and int(t.max()) < sizes[s] for s, t in indices.items())
    again = rc.draw_indices("primary", {"cue/adjective": 49}, {"cue/adjective": 6}, 50)
    assert torch.equal(again["cue/adjective"], indices["cue/adjective"][:50])
    lineage = rc.draw_indices("lineage", {"cue/adjective": 49}, {"cue/adjective": 6}, 50)
    assert not torch.equal(lineage["cue/adjective"], again["cue/adjective"])


def test_every_row_is_a_prefix_of_the_base_draw():
    assert rc.prefix_columns((2, 0, 3), 6) == [0, 1, 12, 13, 14] and rc.prefix_columns((8, 8, 8), 8) == list(range(24))
    for rows, slots in ((rc.y2_rows(), 6), (rc.y3_rows(), 8)):
        for composition in rows:
            columns = rc.prefix_columns(composition, slots)
            assert len(columns) == sum(composition)
            assert [sum(1 for c in columns if c // slots == block) for block in range(3)] == list(composition)
            assert all(c % slots < composition[c // slots] for c in columns)


# ---------------------------------------------------------------------------
# 9–11: the kernel against 020's statistics, the floor rule, the pass predicate.


def test_the_kernel_equals_experiment_020s_own_statistics_with_duplicated_units():
    table, pools = synthetic_table(seed=1)
    scoring = table.scoring("level0")
    cues = [cue for cls in rc.CUE_CLASSES for cue, _ in pools.cues[cls]]
    frames = [frame for template in rc.Y2_AXES for frame in pools.frames_all_unscreened[template]]
    nouns = [key for rule in rc.RULE_CLASSES for key in pools.nouns[rule]]
    generator = torch.Generator().manual_seed(7)
    cue_index = torch.randint(0, len(cues), (6, 9), generator=generator)
    cue_index[:, 1] = cue_index[:, 0]  # duplicates are units of their own
    y1 = rc.y1_statistics(rc.cue_sums(scoring, cues), cue_index)
    grid = rc.grid_sums(scoring, cues, frames)
    frame_index = torch.randint(0, len(frames), (6, 7), generator=generator)
    frame_index[:, 3] = frame_index[:, 2]
    slots = rc.y2_slot_sums(grid, cue_index, frame_index)
    noun_sums = rc.noun_cue_sums(scoring, cues, nouns)
    noun_index = torch.randint(0, len(nouns), (6, 10), generator=generator)
    noun_index[:, 5] = noun_index[:, 4]
    slot_statistics = rc.y3_slot_statistics(noun_sums, noun_index, cue_index)
    for b in range(6):
        units = [cues[int(i)] for i in cue_index[b]]
        kernel = {name: rc.as_values(y1[name][b : b + 1])[0] for name in rc.Y1_STATISTICS}
        assert rc.agreement(kernel, rc.direct_y1(scoring, units))[0] <= rc.STATISTIC_AGREEMENT_TOLERANCE
        for columns in ([0, 1, 2, 3, 4, 5, 6], [0, 2, 3], [1, 4, 5, 6]):
            frame_units = [frames[int(frame_index[b][c])] for c in columns]
            y2 = rc.y2_statistics(slots, columns, [grid.templates[int(frame_index[b][c])] for c in columns])
            kernel = {name: rc.as_values(y2[name][b : b + 1])[0] for name in rc.Y2_STATISTICS}
            assert rc.agreement(kernel, rc.direct_y2(scoring, units, frame_units))[0] <= rc.STATISTIC_AGREEMENT_TOLERANCE, (b, columns)
        for columns in (list(range(10)), [0, 1, 4, 5, 9], [2, 3, 6, 7, 8]):
            noun_units = [nouns[int(noun_index[b][c])] for c in columns]
            y3 = rc.y3_statistics(slot_statistics, columns)
            kernel = {name: rc.as_values(y3[name][b : b + 1])[0] for name in rc.Y3_STATISTICS}
            assert rc.agreement(kernel, rc.direct_y3(scoring, units, noun_units))[0] <= rc.STATISTIC_AGREEMENT_TOLERANCE, (b, columns)
    # an undefined statistic is undefined on both sides (a noun whose measured contrast is constant)
    flat = rc.ExposedTable(table.cues, table.frames, table.templates, table.noun_keys, torch.cat([torch.full((len(table.cues), 1), 0.5, dtype=torch.float64), table.measured[:, 1:]], dim=1),
                           table.level0, table.ceiling, table.no_l5, table.base, table.dT)
    flat_scoring = flat.scoring("level0")
    constant_sums = rc.noun_cue_sums(flat_scoring, cues, [nouns[0], nouns[1]])
    values = rc.y3_slot_statistics(constant_sums, torch.tensor([[0, 1]]), torch.tensor([[0, 1, 2]]))
    assert math.isnan(float(values["r2"][0, 0])) and rc.direct_y3(flat_scoring, cues[:3], [nouns[0], nouns[1]])["noun_r2_k90"] is None


def test_the_kernel_pools_two_pass_moments_so_a_large_offset_cannot_cost_it_precision():
    """With every value shifted by 1e4 a one-pass Σy² − (Σy)²/n pooling drifts to ~1e-10 on the pair-mean R²; the
    two-pass group moments stay at reassociation level. This pins the pooling, not only the definitions."""
    table, pools = synthetic_table(seed=5)
    shifted = rc.ExposedTable(table.cues, table.frames, table.templates, table.noun_keys, table.measured + 1e4, table.level0 + 1e4, table.ceiling + 1e4, table.no_l5 + 1e4,
                              table.base + 1e4, table.dT)
    scoring = shifted.scoring("level0")
    cues = [cue for cls in rc.CUE_CLASSES for cue, _ in pools.cues[cls]]
    frames = [frame for template in rc.Y2_AXES for frame in pools.frames_all_unscreened[template]]
    nouns = [key for rule in rc.RULE_CLASSES for key in pools.nouns[rule]]
    cue_index = torch.randint(0, len(cues), (4, 12), generator=torch.Generator().manual_seed(11))
    y1 = rc.y1_statistics(rc.cue_sums(scoring, cues), cue_index)
    grid = rc.grid_sums(scoring, cues, frames)
    frame_index = torch.arange(len(frames)).unsqueeze(0).repeat(4, 1)
    y2 = rc.y2_statistics(rc.y2_slot_sums(grid, cue_index, frame_index), list(range(len(frames))), list(grid.templates))
    noun_index = torch.arange(len(nouns)).unsqueeze(0).repeat(4, 1)
    y3 = rc.y3_statistics(rc.y3_slot_statistics(rc.noun_cue_sums(scoring, cues, nouns), noun_index, cue_index), list(range(len(nouns))))
    for b in range(4):
        units = [cues[int(i)] for i in cue_index[b]]
        for kernel, direct in (({n: rc.as_values(y1[n][b : b + 1])[0] for n in rc.Y1_STATISTICS}, rc.direct_y1(scoring, units)),
                               ({n: rc.as_values(y2[n][b : b + 1])[0] for n in rc.Y2_STATISTICS}, rc.direct_y2(scoring, units, frames)),
                               ({n: rc.as_values(y3[n][b : b + 1])[0] for n in rc.Y3_STATISTICS}, rc.direct_y3(scoring, units, nouns))):
            assert rc.agreement(kernel, direct)[0] <= 1e-11  # an order of magnitude inside the guard


def test_the_floor_rule_is_the_directional_tail_with_the_zero_skill_clamp():
    values = [i / 10_000 for i in range(10_000)][::-1]
    assert rc.tail_floor(values, "r2") == {"floor": 249 / 10_000, "raw": 249 / 10_000, "clamped": False}
    assert rc.tail_floor(values, "error")["floor"] == 9_750 / 10_000  # the 250th largest
    negative = [value - 0.5 for value in values]
    assert rc.tail_floor(negative, "r2") == {"floor": 0.0, "raw": 249 / 10_000 - 0.5, "clamped": True}
    undefined = [None] * 300 + values[300:]
    assert rc.tail_floor(undefined, "r2") == {"floor": 0.0, "raw": None, "clamped": True}
    assert rc.tail_floor(undefined, "error")["floor"] is None  # an uninformative (+∞) error ceiling is recorded as such
    exact = [0.1234567890123456789] * 10_000
    assert rc.tail_floor(exact, "r2")["floor"] == 0.1234567890123456789  # full precision, no rounding


def test_the_pass_predicate_at_its_boundaries():
    smallest = math.nextafter(0.0, 1.0)
    assert smallest == 5e-324
    for floor in (0.0, 0.37):
        assert not rc.passes("r2", -1e-12, floor) and not rc.passes("r2", 0.0, floor)
        assert not rc.passes("r2", None, floor) and not rc.passes("r2", math.nan, floor) and not rc.passes("r2", math.inf, floor) and not rc.passes("r2", -math.inf, floor)
    assert rc.passes("r2", smallest, 0.0)  # any positive skill passes a clamped floor
    assert rc.passes("r2", 0.37, 0.37) and not rc.passes("r2", math.nextafter(0.37, -math.inf), 0.37)
    assert rc.passes("error", 0.8, 0.8) and not rc.passes("error", math.nextafter(0.8, math.inf), 0.8)
    assert not rc.passes("error", None, 0.8) and not rc.passes("error", math.nan, 0.8) and not rc.passes("error", math.inf, 0.8)
    assert rc.passes("error", 1e6, None)  # the recorded uninformative ceiling
    with pytest.raises(ValueError):
        rc.passes("r2", 0.5, None)
    # 020's truthiness idiom failed an exact 0.0 by accident; 021 fails it by the explicit positive-skill rule
    assert ((0.0 or -1.0) >= 0.0) is False and rc.passes("r2", 0.0, 0.0) is False


# ---------------------------------------------------------------------------
# 12–15: one predicate, the selections, the scoring branches.


def _floor_tables(y1=0.1, y2=0.1, y3=0.1, error=5.0):
    floors = lambda names: {name: (error if rc.KIND[name] == "error" else value) for name, value in ((n, {"Y1": y1, "Y2": y2, "Y3": y3}[o]) for o, ns in rc.STATISTICS.items() for n in ns) if name in names}  # noqa: E731
    return {"Y1": floors(rc.Y1_STATISTICS), "Y2": {rc.row_key(row): floors(rc.Y2_STATISTICS) for row in rc.y2_rows()}, "Y3": {rc.row_key(row): floors(rc.Y3_STATISTICS) for row in rc.y3_rows()}}


def test_the_y2_row_comes_from_the_validity_verdicts_alone():
    tables = _floor_tables()

    class OnlyVerdicts(dict):
        def __getitem__(self, key):
            if key not in ("valid", "template_id"):
                raise AssertionError(f"the Y2 selection read {key}")
            return super().__getitem__(key)

    def frames(valid: dict[str, int]):
        out = {}
        for template in rc.Y2_AXES:
            for i in range(6):
                out[f"{template}-{i}"] = OnlyVerdicts({"valid": i < valid[template], "template_id": template, "rows": "never read"})
        return out

    selection = rc.select_y2_row(frames({"cardinal": 6, "quantifier": 5, "coordinated-adjective": 4}), tables)
    assert selection["composition"] == [6, 5, 4] and selection["row"] == "6/5/4" and selection["floors"] == tables["Y2"]["6/5/4"] and len(selection["valid_frames"]) == 15
    for valid in ({"cardinal": 6, "quantifier": 5, "coordinated-adjective": 0}, {"cardinal": 4, "quantifier": 4, "coordinated-adjective": 3}, {"cardinal": 3, "quantifier": 4, "coordinated-adjective": 4},
                  {"cardinal": 6, "quantifier": 6, "coordinated-adjective": 3}):  # the last: 15 valid frames, failing on the coordinated clause alone
        failed = rc.select_y2_row(frames(valid), tables)
        assert failed["precondition"]["ok"] is False and failed["row"] is None and failed["floors"] is None
    stage1 = {"frames": {k: dict(v) for k, v in frames({"cardinal": 6, "quantifier": 6, "coordinated-adjective": 6}).items()}, "digest": "d" * 64}
    stage1["y2_selection"] = rc.select_y2_row(stage1["frames"], tables)
    stage1["y2_selection_sha256"] = rc.selection_digest(stage1["y2_selection"], stage1["digest"])
    assert rc.assert_y2_selection(stage1, tables)["row"] == "6/6/6"
    for tamper in (lambda s: s["y2_selection"].update(row="6/6/5"), lambda s: s["frames"]["cardinal-0"].update(valid=False), lambda s: s.update(digest="e" * 64)):
        broken = json.loads(json.dumps(stage1))
        tamper(broken)
        with pytest.raises(rd.PhaseError):
            rc.assert_y2_selection(broken, tables)


def test_the_y3_row_comes_from_measured_scorability_alone(frozen):
    pool, digests, confirmation = frozen
    tables = _floor_tables()
    assert list(inspect.signature(rc.select_y3_row).parameters) == ["measured_fresh", "nouns", "tables"]  # no prediction can reach it
    measured = torch.randn(40, len(confirmation.nouns), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    full = rc.select_y3_row(measured, confirmation.nouns, tables)
    assert full["composition"] == [8, 8, 8] and full["row"] == "8/8/8" and len(full["scorable"]) == 24
    constant = measured.clone()
    constant[:, 9] = 0.25  # a sibilant noun whose measured contrast has zero variance
    dropped = rc.select_y3_row(constant, confirmation.nouns, tables)
    assert dropped["composition"] == [8, 7, 8] and dropped["row"] == "8/7/8" and confirmation.nouns[9].lexical_key not in dropped["scorable"]
    few = measured.clone()
    few[:, :7] = 1.0
    assert rc.select_y3_row(few, confirmation.nouns, tables)["precondition"]["ok"] is False


def _scoring_world(*, y2_valid: int = 6, seed: int = 0, noise: float = 0.3):
    """Synthetic stage-2 tables and a lock carrying floor tables: small, with 020's preconditions lowered by the caller."""
    generator = torch.Generator().manual_seed(seed)
    cues = [f"cue{i}" for i in range(6)]
    exposed_frames = [(f"{template}-e{i}", template) for template in rc.Y2_AXES for i in range(3)]
    fresh_frames = [(f"{template}-f{i}", template) for template in rc.Y2_AXES for i in range(y2_valid)]
    nouns, fresh_nouns = [f"n{i}" for i in range(5)], [pm.Noun(f"fresh{i}", rd.FRESH_NOUN_SPLIT, rc.RULE_CLASSES[i % 3], (100 + i,), (200 + i,)) for i in range(6)]

    def table(frames, keys, offset):
        pairs = [(cue, frame, template) for frame, template in frames for cue in cues]
        y = torch.randn(len(pairs), len(keys), generator=generator, dtype=torch.float64) + offset
        return rd.ScoringTable(tuple(p[0] for p in pairs), tuple(p[1] for p in pairs), tuple(p[2] for p in pairs), y,
                               y + noise * torch.randn(len(pairs), len(keys), generator=generator, dtype=torch.float64), tuple(keys))

    y1, y1_fresh, y2, y2_fresh = table(exposed_frames, nouns, 0.0), table(exposed_frames, [n.lexical_key for n in fresh_nouns], 0.0), table(fresh_frames, nouns, 0.0), table(fresh_frames, [n.lexical_key for n in fresh_nouns], 0.0)
    y1_fresh = rd.ScoringTable(y1.cues, y1.frames, y1.templates, y1_fresh.measured, y1_fresh.predicted, y1_fresh.noun_keys)
    same = lambda t: rd.ScoringTable(t.cues, t.frames, t.templates, t.measured, t.predicted, t.noun_keys)  # noqa: E731
    tables = {name: {"exposed": base, "fresh": fresh, "no_l5": same(base), "base": same(base), "ceiling": same(base), "dT": torch.randn(len(base.cues), dtype=torch.float64)}
              for name, base, fresh in (("Y1", y1, y1_fresh), ("Y2", y2, y2_fresh))}
    stage1_frames = {frame: {"valid": True, "template_id": template} for frame, template in fresh_frames}
    return tables, stage1_frames, fresh_nouns


def test_every_label_branch_through_the_one_predicate(monkeypatch):
    for name, value in (("MIN_SCORED_TOKENS", 2), ("MIN_VALID_EXPOSED_FRAMES", 2), ("MIN_VALID_FRESH_FRAMES", 3), ("MIN_VALID_COORDINATED_FRAMES", 1), ("MIN_SCORABLE_FRESH_NOUNS", 3),
                        ("MIN_VALID_FRAMES_PER_TOKEN", 1)):
        monkeypatch.setattr(rd, name, value)
    tables, stage1_frames, fresh_nouns = _scoring_world(y2_valid=2)
    calls = []
    original = rc.passes
    monkeypatch.setattr(rc, "passes", lambda kind, value, floor: calls.append(("confirm", kind)) or original(kind, value, floor))

    def scored(floors):
        stage1 = {"frames": stage1_frames, "digest": "d" * 64}
        stage1["y2_selection"] = rc.select_y2_row(stage1_frames, floors)
        lock = {"floor_tables": floors, "dT_only_noun_vector": None, "rank1": None}
        return rc.score_021(stage1, {"tables": tables, "fresh_nouns": fresh_nouns, "dw_fresh": None}, lock)

    easy = scored(_floor_tables(y1=0.01, y2=0.01, y3=0.01, error=10.0))
    assert easy["outcome"]["labels"] == [rd.OUTCOME_Y1[0], rd.OUTCOME_Y2[0], rd.OUTCOME_Y3[0]] and calls
    assert easy["Y2"]["row"] == "2/2/2" and easy["Y3"]["row"] == "2/2/2" and easy["agreement"]["max_difference"] <= rc.STATISTIC_AGREEMENT_TOLERANCE
    assert easy["ceiling"]["Y1"]["error_split"] is not None and easy["joint_fresh_nouns"]["n_nouns"] == len(fresh_nouns)
    hard = scored(_floor_tables(y1=0.999, y2=0.999, y3=0.999, error=1e-6))
    assert hard["outcome"]["labels"] == [rd.OUTCOME_Y1[1], rd.OUTCOME_Y2[1], rd.OUTCOME_Y3[1]]
    monkeypatch.setattr(rc, "passes", lambda kind, value, floor: False)  # the only decision function: replacing it changes every label
    assert scored(_floor_tables(y1=0.01, y2=0.01, y3=0.01, error=10.0))["outcome"]["labels"] == [rd.OUTCOME_Y1[1], rd.OUTCOME_Y2[1], rd.OUTCOME_Y3[1]]
    monkeypatch.setattr(rc, "passes", original)
    # preconditions: too few valid frames, no Y1 table, too few scorable nouns
    monkeypatch.setattr(rd, "MIN_VALID_FRESH_FRAMES", 12)
    assert scored(_floor_tables())["Y2"]["label"] == rd.OUTCOME_Y2[2]
    monkeypatch.setattr(rd, "MIN_VALID_FRESH_FRAMES", 3)
    monkeypatch.setattr(rd, "MIN_SCORABLE_FRESH_NOUNS", 30)
    assert scored(_floor_tables())["Y3"]["label"] == rd.OUTCOME_Y3[2]


def test_the_report_ranks_the_fresh_ceiling_against_the_ceiling_column():
    state = {"run_id": "r", "confirmation_020_sha256": "c", "program_blob_sha1": "p", "protocol_code_commit": "a" * 40, "phases": {}, "executed_prompt_keys": [], "executed_noun_keys": [],
             "calibration": {}, "confirmation": {"outcome": {"label": "x"}, "Y1": {"label": "L", "row": None}, "Y2": {"label": "L", "row": "6/6/6"}, "Y3": {"label": "L"},
                                                 "ceiling": {"Y1": {"flattened_r2": 0.5, "level0_flattened_r2": 0.2, "error_split": None}}}}
    draws = {"ceiling:Y1": {"all": [[0.9, 0.1], [0.9, 0.2], [0.9, 0.3], [0.9, 0.7]]}}  # column 0 is Level 0, column 1 the ceiling
    text = rc.render_report(state, draws)
    assert "percentile among the row's exposed-like draws 0.750" in text  # 3 of 4 ceiling draws ≤ 0.5; ranked on Level 0 it would be 0.000


def test_json_safety_and_the_descriptive_percentile():
    assert rc.json_safe({"a": [1.0, math.inf, {"b": math.nan}], "c": -math.inf, "d": "x"}) == {"a": [1.0, None, {"b": None}], "c": None, "d": "x"}
    assert rc.percentile_of(0.5, [None, 0.2, 0.6, 0.5], "r2") == 0.75  # an undefined draw ranks as the worst
    assert rc.percentile_of(0.5, [None, 0.2, 0.6, 0.5], "error") == 0.75 and rc.percentile_of(None, [0.1], "r2") is None


def test_calibration_pass_rates_use_the_one_predicate(monkeypatch):
    calls = []
    original = rc.passes
    monkeypatch.setattr(rc, "passes", lambda kind, value, floor: calls.append(kind) or original(kind, value, floor))
    rates = rc.pass_rates({"token_mean_r2": [0.5, 0.0, None, 0.9], "pooled_mae": [0.2, 0.3, 0.4, 0.5]}, {"token_mean_r2": 0.0, "pooled_mae": 0.35})
    assert len(calls) == 8 and rates["conditions"] == {"token_mean_r2": 0.5, "pooled_mae": 0.5} and rates["joint"] == 0.25 and rates["per_draw"] == [True, False, False, False]


# ---------------------------------------------------------------------------
# 16–18: the reproduction gate, the calibration record, the cross-check hardening, the precondition.


def test_the_reproduction_gate_passes_exactly_and_refuses_any_planted_difference():
    table, _ = synthetic_table(seed=2)
    recorded = recorded_exploration(table)
    gate = rc.reproduction_gate(table, recorded)
    assert gate["passed"] and gate["max_difference"] == 0.0 and set(gate["sections"]) == {"statistics", "nouns", "comparators"}
    rc.enforce_gate(gate)
    for path in (("statistics", "per_cue", table.cues[0], "mae"), ("statistics", "per_frame", table.frames[0], "r2"), ("nouns", table.noun_keys[0], "slope"),
                 ("comparators", "ceiling_measured_dx3")):
        planted = json.loads(json.dumps(recorded))
        target = planted
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = target[path[-1]] + 1e-8
        failed = rc.reproduction_gate(table, planted)
        assert not failed["passed"] and failed["max_difference"] == pytest.approx(1e-8, rel=1e-3)
        with pytest.raises(pm.IncidentError, match="reproduction gate failed"):
            rc.enforce_gate(failed)
    missing = json.loads(json.dumps(recorded))
    del missing["statistics"]["per_cue"][table.cues[0]]
    assert rc.reproduction_gate(table, missing)["passed"] is False


def test_the_calibration_record_on_a_synthetic_table(monkeypatch, frozen):
    pool, digests, confirmation = frozen
    monkeypatch.setattr(rc, "B", 40)
    monkeypatch.setattr(rc, "CROSS_CHECK_DRAWS", 2)
    table, pools = synthetic_table(seed=3)
    seen = {"Y1": 0, "Y2": 0, "Y3": 0}
    for outcome, name in (("Y1", "score_y1"), ("Y2", "score_y2"), ("Y3", "score_y3")):
        original = getattr(rd, name)
        monkeypatch.setattr(rd, name, lambda *args, _o=outcome, _f=original, **kwargs: seen.__setitem__(_o, seen[_o] + 1) or _f(*args, **kwargs))
    result = rc.run_calibration(table, pools)
    body = result["body"]
    assert seen == {"Y1": 40, "Y2": 40, "Y3": 40}  # 020's own scoring runs on the full rows only, once per base draw
    assert [len(body["rows"][o]) for o in ("Y1", "Y2", "Y3")] == [1, 64, 84] and body["draws"]["B"] == 40 and body["draws"]["tail_index"] == 1
    assert body["cross_check"]["n_checked"] == 2 * (1 + 64 + 84) and body["cross_check"]["max_difference"] <= rc.STATISTIC_AGREEMENT_TOLERANCE
    for outcome, entries in body["rows"].items():
        for entry in entries:
            assert set(entry["floors"]) == set(rc.STATISTICS[outcome])
            assert all(entry["floors"][name] >= 0.0 for name in rc.STATISTICS[outcome] if rc.KIND[name] == "r2")
            assert all(0.0 <= rate <= 1.0 for rate in entry["pass_rates"].values()) and 0.0 <= entry["joint_pass_rate"] <= 1.0
            assert entry["draw_values_sha256"] == rc.tensor_digest(result["arrays"][outcome][entry["key"]])
    assert set(body["full_rows"]["draw_values"]) == {"Y1", "Y2", "Y3"} and all(len(values) == 40 for o in body["full_rows"]["draw_values"].values() for values in o.values())
    assert set(body["sensitivity"]) == {"all-frames", "lineage", "cohorts"} and body["sensitivity"]["all-frames"]["rows"]["Y2"]["composition"] == [6, 6, 6]
    array_digests = {o: {k: rc.tensor_digest(v) for k, v in entries.items()} for o, entries in result["arrays"].items()}
    record = rc.calibration_record(body=body, run_id="r", protocol_code_commit="a" * 40, digests={"confirmation_020": confirmation.content_sha256}, pools=pools, screen={},
                                   precondition={"ok": True}, rematerialization={}, gate={"passed": True}, table_digests=table.digests(), array_digests=array_digests)
    assert record["content_sha256"] == rc.content_digest(record)
    rd.assert_fresh_nouns_absent(record, confirmation)  # nothing of a fresh noun, and the index arrays only as digests
    assert "local" not in json.dumps(record["draws"]) and set(record["draws"]["index_digests"]) == {f"cue/{c}" for c in rc.CUE_CLASSES} | {f"frame/{t}" for t in rc.Y2_AXES} | {f"noun/{r}" for r in rc.RULE_CLASSES}
    rc.verify_calibration_record(record)  # computed and verified under the same constants (B patched to 40 here)
    monkeypatch.setattr(rc, "B", 10_000)
    with pytest.raises(rd.PhaseError, match="frozen constants"):  # a record from other constants is refused
        rc.verify_calibration_record(record)
    monkeypatch.setattr(rc, "B", 40)
    tampered = dict(record, program_blob_sha1="0" * 40)
    tampered["content_sha256"] = rc.content_digest(tampered)
    with pytest.raises(rd.PhaseError, match="program"):
        rc.verify_calibration_record(tampered)
    for key, value in (("statistic_agreement_tolerance", 1e-6), ("cross_check_draws", 1), ("reconstruction_tolerance", 1e-6), ("preconditions", {})):
        other = json.loads(json.dumps(record))
        other["constants"][key] = value
        other["content_sha256"] = rc.content_digest(other)
        with pytest.raises(rd.PhaseError, match="frozen constants"):
            rc.verify_calibration_record(other)
    tables = rc.floor_tables(record)
    assert set(tables["Y2"]) == {rc.row_key(row) for row in rc.y2_rows()} and set(tables["Y3"]) == {rc.row_key(row) for row in rc.y3_rows()}


def test_a_failing_cross_check_stops_before_any_floor_with_its_worst_location(monkeypatch):
    monkeypatch.setattr(rc, "B", 40)
    monkeypatch.setattr(rc, "CROSS_CHECK_DRAWS", 2)
    table, pools = synthetic_table(seed=4)
    original = rc.direct_y2
    target = rc.row_key(rc.y2_rows()[5])
    calls = {"n": 0}

    def planted(*args, **kwargs):
        calls["n"] += 1
        values = original(*args, **kwargs)
        return {**values, "frame_r2_k75": values["frame_r2_k75"] + (7.0 if calls["n"] == 2 * 5 + 2 else 1e-3)}  # every Y2 row off by 1e-3; row 5, draw 1 by 7

    monkeypatch.setattr(rc, "direct_y2", planted)
    floors_computed = []
    monkeypatch.setattr(rc, "tail_floor", lambda *args, **kwargs: floors_computed.append(1))
    with pytest.raises(rc.CrossCheckError) as caught:
        rc.run_calibration(table, pools)
    details = caught.value.details
    assert details["outcome"] == "Y2" and details["statistic"] == "frame_r2_k75" and details["row"] == target and details["draw"] == 1  # the maximum, not the first
    assert details["max_difference"] == pytest.approx(7.0 / max(1.0, abs(details["direct"])), rel=1e-6) and details["n_exceeding"] == 2 * len(rc.y2_rows())
    assert details["first_exceeding"]["row"] == rc.row_key(rc.y2_rows()[0]) and details["first_exceeding"]["draw"] == 0 and details["n_checked"] == 2 * (1 + 64 + 84)
    assert not floors_computed  # the whole check ran, and still no floor exists
    json.dumps(details, allow_nan=False)
    monkeypatch.setattr(rc, "direct_y2", lambda *args, **kwargs: {**original(*args, **kwargs), "cue_final_r2": None})  # undefined on one side only
    with pytest.raises(rc.CrossCheckError) as caught:
        rc.run_calibration(table, pools)
    assert caught.value.details["max_difference"] is None and "undefined on one side" in str(caught.value)
    json.dumps(caught.value.details, allow_nan=False)


def test_the_calibration_precondition_needs_six_screened_frames_per_template():
    table, pools = synthetic_table(frames_per_template=7)
    frames = {frame: True for ids in pools.frames_all_unscreened.values() for frame in ids}
    assert rc.screen_precondition(pools.screened(frames)) == {"minimum": 6, "counts": {t: 3 for t in rc.Y2_AXES}, "ok": False}  # the synthetic 017–019 pools hold 3
    wide = rc.CalibrationPools(pools.cues, pools.lineage_cues, pools.cohorts, pools.frames_all_unscreened, pools.frames_all_unscreened, pools.nouns)
    assert rc.screen_precondition(wide.screened(frames))["ok"] is True
    one_short = dict(frames)
    one_short[pools.frames_all_unscreened["quantifier"][0]] = False
    one_short[pools.frames_all_unscreened["quantifier"][1]] = False
    result = rc.screen_precondition(wide.screened(one_short))
    assert result["ok"] is False and result["counts"]["quantifier"] == 5
