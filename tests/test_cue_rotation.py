"""Experiment 025's module.

Tier A checks, without a model:
- the pins and the inherited files;
- the population lists, the count threshold, the configuration and its conditions;
- the geometry, the generator and the plurality pairing on synthetic embeddings;
- the selection rules, the scores, the statistics and the outcome, the paths and the preregistration rendering.

Tier C (the pinned model, opt-in) holds the contracts on spent prompts only:
- the real freeze picks (tokenizer only, nothing written);
- the frozen direction and every geometry gate on 024's spent cues (weights only);
- the patch path on the four spent keys;
- the θ = 0 measurement against ``ul.measure_prompt``, and the vector gate factors against the token ones;
- the θ = 0 save, re-read, tagged keys, ``C`` recomputation and I1/I3/I4 against 024's token gates;
- rotated patches landing, with ``EMBED`` as the only capture.

No tier-C test computes ``ℓ``, ``A``, ``B``, ``G``, ``D_attn`` or the Level-1 identity for any run; a guard refuses
them (and, around rotated runs, the frozen readout and every gate).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul
from test_plural_mechanism import ToyTokenizer

ROOT = Path(__file__).parents[1]

SMALL = cr.Configuration(name="synthetic", n_adjectives=2, n_nouns=2, k_controls=3, primary_odd=0.32, half_odd=0.16, count_threshold=3, n_frames=2,
                         n_scored_nouns=3, expected_picks=(("adjective", ("anxious", "cheerful")), ("noun", ("soldier", "sailor"))))


# ---------------------------------------------------------------------------
# Pins, inherited files, the population lists.


def test_the_pinned_modules_and_the_inherited_files_are_the_reviewed_ones():
    assert cr.assert_frozen_blobs() == cr.FROZEN_BLOBS
    assert {name: blob for name, blob in cr.FROZEN_BLOBS.items() if name != "readout_routing.py"} == rr.FROZEN_BLOBS
    lock_024 = json.loads((ROOT / rr.LOCK_RELATIVE_PATH).read_text())
    assert lock_024["module"] == {"path": "src/neural_decompiler/readout_routing.py", "blob": cr.FROZEN_BLOBS["readout_routing.py"]}
    assert cr.verify_024_inputs(ROOT) == {f"024_{kind}_{what}": cr.INHERITED_024[f"{kind}_{what}_sha256"] for kind in cr.INHERITED_024_PATHS for what in ("file", "content")}
    assert lock_024["content_sha256"] == cr.INHERITED_024["lock_content_sha256"]
    assert cr.DIGEST_KEYS == (*rr.DIGEST_KEYS, *(f"024_{kind}_{what}" for kind in ("calibration", "lock", "confirmation") for what in ("file", "content")),
                              "020_prior_nouns_file")
    prior = cr.prior_nouns(ROOT)
    assert len(prior["nouns"]) == 24 and {"statue", "barrel"} <= set(prior["nouns"]) and prior["file_sha256"] == cr.PRIOR_NOUNS_020["file_sha256"]
    assert prior["sha256"] == pm.sha256_text(pm.canonical_json(prior["ids"])) and prior["ids"] == sorted(set(prior["ids"]))
    assert cr.DESIGN["commit"] == "c0885e5" and cr.DESIGN["correction"] == "26c9925" and cr.PLAN["commit"] == "7d90d28"
    assert (ROOT / cr.DESIGN["path"]).exists() and (ROOT / cr.PLAN["path"]).exists()


def test_a_changed_inherited_file_or_pin_is_refused(tmp_path, monkeypatch):
    for kind, relative in cr.INHERITED_024_PATHS.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    assert cr.verify_024_inputs(tmp_path)
    lock_path = tmp_path / rr.LOCK_RELATIVE_PATH
    lock_path.write_bytes(lock_path.read_bytes() + b"\n")
    with pytest.raises(cr.PhaseError, match="not the reviewed file"):
        cr.verify_024_inputs(tmp_path)
    with pytest.raises(cr.PhaseError, match="020's committed confirmation file"):
        cr.prior_nouns(tmp_path)
    monkeypatch.setitem(cr.FROZEN_BLOBS, "models.py", "0" * 40)
    with pytest.raises(cr.PhaseError, match=r"frozen modules changed: \['models.py'\]"):
        cr.assert_frozen_blobs()


def test_the_population_lists_are_024s_reserves_and_the_frozen_new_list():
    freeze_024 = json.loads((ROOT / rr.CONFIRMATION_RELATIVE_PATH).read_text())
    assert list(cr.ADJECTIVE_RESERVES) == freeze_024["reserves"]["N"]
    assert [f"{singular}/{plural}" for singular, plural in cr.ORDINARY_RESERVES] == freeze_024["reserves"]["ordinary"]
    assert len(cr.NEW_NOUN_LIST) == 39 and list(cr.NEW_NOUN_LIST) == sorted(set(cr.NEW_NOUN_LIST))
    assert cr.EXPECTED_PICKS["adjective"] == cr.ADJECTIVE_RESERVES[:20]
    assert cr.EXPECTED_PICKS["noun"][:8] == tuple(singular for singular, _ in cr.ORDINARY_RESERVES if singular != "statue")
    new = cr.EXPECTED_PICKS["noun"][8:]
    assert new == ("author", "bishop", "dancer", "duck", "goat", "guitar", "hunter", "lawyer", "monk", "nurse", "painter", "prince")
    assert list(new) == [word for word in cr.NEW_NOUN_LIST if word in new]  # the new list's textual order
    used_024 = {cue["word"] for cue in freeze_024["cues"]}
    assert not used_024 & {*cr.EXPECTED_PICKS["adjective"], *cr.EXPECTED_PICKS["noun"], *(cr.regular_plural(word) for word in new)}
    assert tuple(dict(cr.PRODUCTION.expected_picks)[key] for key in cr.STRATA) == (cr.EXPECTED_PICKS["adjective"], cr.EXPECTED_PICKS["noun"])
    assert [cr.regular_plural(word) for word in ("author", "church", "box", "bus", "dish", "wizard")] == ["authors", "churches", "boxes", "buses", "dishes", "wizards"]


# ---------------------------------------------------------------------------
# The count rule and the configuration.


def test_the_count_threshold_and_the_reference_tail():
    assert cr.binomial_tail(40, 27) == Fraction(sum(math.comb(40, j) for j in range(27, 41)), 2 ** 40)
    assert float(cr.binomial_tail(40, 27)) == 0.01923865414210013 and cr.binomial_tail(40, 27) <= cr.ALPHA < cr.binomial_tail(40, 26)
    assert round(float(cr.binomial_tail(40, 26)), 4) == 0.0403
    assert cr.derive_threshold(40) == 27 == cr.PRODUCTION.count_threshold and cr.PRODUCTION.reference_tail() == cr.binomial_tail(40, 27)
    assert cr.binomial_tail(4, 0) == 1 and cr.binomial_tail(4, 5) == 0 and cr.derive_threshold(4) == 5  # no count of 4 reaches 0.025
    power = math.fsum(math.comb(40, j) * 0.722 ** j * 0.278 ** (40 - j) for j in range(27, 41))
    assert 0.79 < power < 0.81  # the design's power statement: q ≈ 0.722 for 80 %


def test_the_production_configuration_and_its_conditions():
    config = cr.PRODUCTION
    assert (config.n_cues, config.primary, config.half, config.k_controls, config.n_frames, config.n_scored_nouns) == (40, "0.32", "0.16", 7, 108, 79)
    randoms = tuple(f"rand{j}{sign}0.32" for j in range(1, 8) for sign in "+-")
    assert config.conditions == ("base", "noun+0.32", "noun-0.32", "noun+0.16", "noun-0.16", *randoms, "plur+0.32", "plur-0.32")
    assert config.outcome_bearing == ("noun+0.32", "noun-0.32", *randoms) and len(config.conditions) == 21 and len(config.outcome_bearing) == 16
    assert config.n_cues * config.n_frames * len(config.conditions) == 90720 and config.n_cues * config.n_frames * len(config.outcome_bearing) == 69120
    assert config.spec("base") == {"kind": "base", "sign": 0, "odd": 0.0, "j": None}
    assert config.spec("noun-0.16") == {"kind": "noun", "sign": -1, "odd": 0.16, "j": None}
    assert config.spec("rand7+0.32") == {"kind": "rand", "sign": 1, "odd": 0.32, "j": 7}
    assert config.spec("plur-0.32") == {"kind": "plur", "sign": -1, "odd": 0.32, "j": None}
    for unknown in ("rand8+0.32", "noun+0.33", "plur+0.16", "rand1+0.16", "noun"):
        with pytest.raises(ValueError, match="unknown condition"):
            config.spec(unknown)
    record = config.to_json()
    assert Fraction(record["reference_tail"]["exact"]) == config.reference_tail() and record["reference_tail"]["value"] == 0.01923865414210013
    assert record["conditions"] == list(config.conditions) and record["expected_picks"] == {key: list(value) for key, value in cr.EXPECTED_PICKS.items()}


def test_an_inconsistent_configuration_is_refused():
    picks = cr.PRODUCTION.expected_picks
    for kwargs, message in ((dict(count_threshold=26), "production threshold"), (dict(n_adjectives=19), "quotas"),
                            (dict(half_odd=0.32), "doses"), (dict(count_threshold=41), "count threshold"), (dict(k_controls=0), "control count"),
                            (dict(expected_picks=tuple(reversed(picks))), "adjective then noun")):
        arguments = dict(name="production", n_adjectives=20, n_nouns=20, k_controls=7, primary_odd=0.32, half_odd=0.16, count_threshold=27, n_frames=108,
                         n_scored_nouns=79, expected_picks=picks)
        arguments.update(kwargs)
        with pytest.raises(ValueError, match=message):
            cr.Configuration(**arguments)
    assert cr.Configuration(**{**dict(name="test-world", n_adjectives=20, n_nouns=20, k_controls=7, primary_odd=0.32, half_odd=0.16, count_threshold=26,
                                      n_frames=108, n_scored_nouns=79, expected_picks=picks)}).count_threshold == 26  # only production derives it


# ---------------------------------------------------------------------------
# The geometry on synthetic embeddings.


def _synthetic(seed: int = 0, vocab: int = 96, dim: int = 24):
    generator = torch.Generator().manual_seed(seed)
    W_E = torch.randn(vocab, dim, generator=generator)  # float32, as the model's
    nouns = tuple(SimpleNamespace(sg_ids=(2 * i,), pl_ids=(2 * i + 1,), single_token=True) for i in range(10))
    noun_row_ids = [token for noun in nouns for token in (noun.sg_ids[0], noun.pl_ids[0])]  # interleaved, as 024's record
    cue_ids = list(range(40, 60))
    record = {"score": {"noun_row_ids": noun_row_ids, "calibration_cue_ids": cue_ids, "mu_noun_sha256": rc.tensor_digest(rr.centroid(W_E, noun_row_ids)),
                        "mu_cue_sha256": rc.tensor_digest(rr.centroid(W_E, cue_ids))}}
    tokens = ({"word": "anxious", "token_id": 70, "stratum": "adjective"}, {"word": "cheerful", "token_id": 71, "stratum": "adjective"},
              {"word": "soldier", "token_id": 72, "stratum": "noun"}, {"word": "sailor", "token_id": 73, "stratum": "noun"})
    return W_E, record, SimpleNamespace(nouns=nouns), tokens


def test_the_direction_is_024s_score_and_its_centroids_must_reproduce_the_record():
    W_E, record, _, _ = _synthetic()
    d = cr.direction(W_E, record)
    mu_noun, mu_cue = cr.centroids(W_E, record)
    for token_id in range(60, 80):
        assert abs(cr.score(d, W_E[token_id]) - rr.nounness(W_E[token_id], mu_noun, mu_cue)) < 1e-14  # 024's score, as d·Ê
    tampered = json.loads(json.dumps(record))
    tampered["score"]["calibration_cue_ids"] = tampered["score"]["calibration_cue_ids"][1:]
    with pytest.raises(pm.IncidentError, match="do not reproduce 024's calibration record"):
        cr.direction(W_E, tampered)


def test_the_rotation_has_the_exact_odd_component_preserves_the_length_and_controls_are_neutral():
    W_E, record, pool, _ = _synthetic()
    d = cr.direction(W_E, record)
    p_hat = cr.plurality_direction(W_E, pool)
    for token_id in (70, 71, 72, 73):
        entry = cr.cue_vectors(W_E, d, p_hat, token_id, SMALL)
        g = entry["geometry"]
        assert torch.allclose(g.s0 * g.E_hat + g.tau * g.t_hat, d, rtol=0, atol=1e-14) and abs(float(g.E_hat @ g.t_hat)) < 1e-15
        for dose in (0.32, 0.16):
            theta = cr.theta_for(dose, g.tau)
            plus, minus = entry["vectors64"][f"noun+{dose:.2f}"], entry["vectors64"][f"noun-{dose:.2f}"]
            assert abs(0.5 * (cr.score(d, plus) - cr.score(d, minus)) - dose) < 1e-14
            assert abs(cr.score(d, plus) - g.s0 - (g.s0 * (math.cos(theta) - 1.0) + dose)) < 1e-14  # the complete change
            assert abs(cr.angle(g.E, plus) - theta) < 1e-12 and abs(float(torch.linalg.vector_norm(plus)) - g.norm) < 1e-12
        theta = cr.theta_for(0.32, g.tau)
        for condition in SMALL.conditions:
            if SMALL.spec(condition)["kind"] in ("rand", "plur"):
                assert abs(cr.score(d, entry["vectors64"][condition]) - g.s0 * math.cos(theta)) < 1e-14, condition  # nounness-neutral
        for u in (*entry["controls"], entry["p_prime"]):
            assert abs(float(u @ g.E_hat)) < 1e-15 and abs(float(u @ g.t_hat)) < 1e-15 and abs(float(u @ d)) < 1e-14
        checks = cr.geometry_checks(d, g, entry["controls"], entry["p_prime"], entry["vectors64"], entry["vectors32"], W_E[token_id], SMALL)
        cr.enforce_geometry(checks, str(token_id))
        assert checks["base_equals_model_row"] is True and torch.equal(entry["vectors32"]["base"], W_E[token_id])
    with pytest.raises(pm.IncidentError, match="not reachable"):
        cr.theta_for(1.5, 1.4)


def test_the_geometry_block_is_reproducible_bit_for_bit_and_json_exact():
    W_E, record, pool, tokens = _synthetic()
    first, second = cr.geometry_block(W_E, record, pool, tokens, SMALL), cr.geometry_block(W_E, record, pool, tokens, SMALL)
    assert first["block"] == second["block"] and json.loads(pm.canonical_json(first["block"])) == first["block"]
    assert cr.verify_geometry_against_lock(second["block"], {"geometry": first["block"]}) == {"bitwise_equal": True, "differing": []}
    drifted = json.loads(json.dumps(first["block"]))
    drifted["cues"][2]["tau"] = math.nextafter(drifted["cues"][2]["tau"], 2.0)
    assert cr.verify_geometry_against_lock(drifted, {"geometry": first["block"]}) == {"bitwise_equal": False, "differing": ["cues"]}
    assert [cue["word"] for cue in first["block"]["cues"]] == ["anxious", "cheerful", "soldier", "sailor"] and first["block"]["conditions"] == list(SMALL.conditions)
    for name, limit in cr.GEOMETRY_LIMITS.items():
        assert first["block"]["check_maxima"][name] <= cr.TOLERANCES[limit]
    for token in tokens:
        vectors = first["vectors32"][token["token_id"]]
        assert list(vectors) == list(SMALL.conditions) and all(vector.dtype == torch.float32 for vector in vectors.values())


def test_a_geometry_gate_failure_raises_an_incident():
    W_E, record, pool, tokens = _synthetic()
    d = cr.direction(W_E, record)
    entry = cr.cue_vectors(W_E, d, cr.plurality_direction(W_E, pool), 70, SMALL)
    g = entry["geometry"]
    checks = cr.geometry_checks(d, g, entry["controls"], entry["p_prime"], entry["vectors64"], entry["vectors32"], W_E[70], SMALL)
    with pytest.raises(pm.IncidentError, match="not the model's embedding row"):
        cr.enforce_geometry({**checks, "base_equals_model_row": False}, "x")
    with pytest.raises(pm.IncidentError, match="odd32"):
        cr.enforce_geometry({**checks, "odd32": 2e-6}, "x")
    with pytest.raises(pm.IncidentError, match="neutral64"):
        cr.enforce_geometry({**checks, "neutral64": float("nan")}, "x")
    wrong_row = W_E[70].clone()
    wrong_row[0] = torch.nextafter(wrong_row[0], torch.tensor(10.0))  # one float32 ulp
    assert cr.geometry_checks(d, g, entry["controls"], entry["p_prime"], entry["vectors64"], entry["vectors32"], wrong_row, SMALL)["base_equals_model_row"] is False


def test_the_angle_is_well_conditioned():
    a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert cr.angle(a, a) == 0.0 and abs(cr.angle(a, -a) - math.pi) < 1e-15 and abs(cr.angle(a, torch.tensor([0.0, 2.0, 0.0])) - math.pi / 2) < 1e-15
    for theta in (1e-9, 1e-4, 0.241, 1.3):
        b = torch.tensor([math.cos(theta), math.sin(theta), 0.0], dtype=torch.float64) * 3.0
        assert abs(cr.angle(a, b) - theta) < 1e-15 * max(1.0, theta) + 1e-18


def test_the_plurality_direction_pairs_each_nouns_own_singular_and_plural():
    generator = torch.Generator().manual_seed(4)
    W_E = torch.randn(40, 12, generator=generator)
    shift = torch.randn(12, generator=generator)
    nouns = tuple(SimpleNamespace(sg_ids=(2 * i,), pl_ids=(2 * i + 1,), single_token=True) for i in range(8))
    for noun in nouns:
        W_E[noun.pl_ids[0]] = W_E[noun.sg_ids[0]] + shift
    split = SimpleNamespace(sg_ids=(30, 31), pl_ids=(32,), single_token=False)  # a multi-token noun is skipped
    W_E[30] = 1e3
    p_hat = cr.plurality_direction(W_E, SimpleNamespace(nouns=(*nouns, split)))
    assert torch.allclose(p_hat, cr.unit(shift.double()), rtol=0, atol=1e-6)


def test_the_control_generator_is_sha256_counter_mode_box_muller():
    """An independent re-implementation of the frozen generator reproduces it bit for bit."""

    def uniforms(tag: str, count: int) -> list[float]:
        out = []
        for counter in range((count + 3) // 4):
            out += [((word >> 11) + 1) * 2.0 ** -53 for word in struct.unpack(">4Q", hashlib.sha256(f"{tag}|{counter}".encode()).digest())]
        return out[:count]

    tag = cr.CONTROL_TAG.format(token_id=8274, j=3)
    assert tag == "025|control|8274|3"
    assert cr.sha_uniforms(tag, 9) == uniforms(tag, 9) and all(0.0 < u <= 1.0 for u in cr.sha_uniforms(tag, 400))
    u = uniforms(tag, 8)
    expected = []
    for i in range(4):
        r, phi = math.sqrt(-2.0 * math.log(u[2 * i])), 2.0 * math.pi * u[2 * i + 1]
        expected += [r * math.cos(phi), r * math.sin(phi)]
    assert cr.sha_gaussians(tag, 8).tolist() == expected and cr.sha_gaussians(tag, 7).tolist() == expected[:7]
    assert cr.sha_gaussians(tag, 512).dtype == torch.float64 and not torch.equal(cr.sha_gaussians(tag, 16), cr.sha_gaussians("025|control|8274|4", 16))
    # the pinned digest: 512 Gaussians of the first control of token 8274 (the generator can never change silently)
    assert rc.tensor_digest(cr.sha_gaussians("025|control|8274|1", 512)) == "85cd0b694890ac2d0c8dfc7800e1ec3a61d6372cd61c2ab1eb71291f77c277cc"
    W_E, record, pool, _ = _synthetic()
    d = cr.direction(W_E, record)
    g = cr.cue_geometry(W_E, 70, d)
    assert all(torch.equal(a, b) for a, b in zip(cr.control_directions(g, 3), cr.control_directions(g, 3)))
    assert torch.equal(cr.control_directions(g, 3)[2], cr.unit(cr.project_off(cr.sha_gaussians("025|control|70|3", 24), (g.E_hat, g.t_hat))))


# ---------------------------------------------------------------------------
# The freeze's selection on a synthetic tokenizer.


def _selection_tokenizer(**overrides: int) -> ToyTokenizer:
    words = [*cr.ADJECTIVE_RESERVES, *(w for pair in cr.ORDINARY_RESERVES for w in pair), *cr.NEW_NOUN_LIST, *(cr.regular_plural(w) for w in cr.NEW_NOUN_LIST)]
    vocabulary = {" " + word: 1000 + index for index, word in enumerate(dict.fromkeys(words))}
    vocabulary.update({" " + word: token for word, token in overrides.items()})
    return ToyTokenizer(vocabulary)


def test_the_selection_applies_every_rule_mechanically():
    tokenizer = _selection_tokenizer()
    ids = {word.strip(): token for word, token in tokenizer.vocabulary.items()}
    blocked = {"earlier_cues": frozenset({ids["curious"]}), "target_forms": frozenset({ids["soldiers"]}), "prior_nouns": frozenset({ids["statue"]}),
               "frame_tokens": frozenset({ids["author"]})}
    config = cr.Configuration(name="synthetic", n_adjectives=3, n_nouns=9, k_controls=1, primary_odd=0.32, half_odd=0.16, count_threshold=1, n_frames=1,
                              n_scored_nouns=1, expected_picks=(("adjective", ("anxious", "cheerful", "jealous")),
                                                                ("noun", ("sailor", "priest", "knight", "onion", "carrot", "pirate", "tourist", "baker", "bishop"))))
    selection = cr.select_cues(tokenizer, blocked, config)
    assert selection["picks"] == {"adjective": ["anxious", "cheerful", "jealous"], "noun": list(config.expected("noun"))}
    reasons = {entry["candidate"]: entry["reason"] for entry in selection["rejected"]}
    assert reasons["curious"] == f"token id {ids['curious']}: already used as a cue by Experiments 005–024"
    assert reasons["soldier/soldiers"] == f"soldier: eligible; soldiers: token id {ids['soldiers']}: a form of a pool target noun"
    assert "statue: token id" in reasons["statue/statues"] and "020's confirmation list" in reasons["statue/statues"]
    assert "a token of an exposed frame" in reasons["author/authors"]
    assert all(cue["plural_token_id"] == ids[cue["plural"]] for cue in selection["cues"] if cue["stratum"] == "noun")
    with pytest.raises(cr.FreezeDeviation, match="expected picks"):
        cr.select_cues(tokenizer, {key: frozenset() for key in blocked}, config)
    with pytest.raises(cr.FreezeShortfall, match="eligible adjectives 22 of 23"):
        cr.select_cues(tokenizer, blocked, cr.Configuration(name="synthetic", n_adjectives=23, n_nouns=1, k_controls=1, primary_odd=0.32, half_odd=0.16,
                                                            count_threshold=1, n_frames=1, n_scored_nouns=1,
                                                            expected_picks=(("adjective", tuple(f"w{k}" for k in range(23))), ("noun", ("sailor",)))))


def test_a_multi_token_word_a_reused_id_and_a_shared_form_are_rejected():
    tokenizer = _selection_tokenizer()
    vocabulary = dict(tokenizer.vocabulary)
    del vocabulary[" lonely"]
    vocabulary.update({" lone": 7, "ly": 8, " nasty": vocabulary[" anxious"], " sailors": vocabulary[" sailor"]})
    tokenizer = ToyTokenizer(vocabulary)
    empty = {key: frozenset() for key in ("earlier_cues", "target_forms", "prior_nouns", "frame_tokens")}
    config = cr.Configuration(name="synthetic", n_adjectives=5, n_nouns=2, k_controls=1, primary_odd=0.32, half_odd=0.16, count_threshold=1, n_frames=1,
                              n_scored_nouns=1, expected_picks=(("adjective", ("anxious", "cheerful", "curious", "jealous", "careful")),
                                                                ("noun", ("soldier", "priest"))))
    reasons = {entry["candidate"]: entry["reason"] for entry in cr.select_cues(tokenizer, empty, config)["rejected"]}
    assert reasons["lonely"] == "2 tokens with a leading space" and reasons["nasty"].endswith("already picked")
    assert reasons["sailor/sailors"] == "sailor: eligible; sailors: eligible"  # one id for both forms: not a noun pair


def test_the_manifest_is_condition_tagged():
    frames = (pm.Frame("cardinal", "cardinal-9", (1, 2), (), {"sg": 5, "pl": 6}, "x {cue}"),
              pm.Frame("coordinated-adjective", "coordinated-adjective-9", (1, 2), (7,), {"sg": 5, "pl": 6}, "y {cue} z"))
    tokens = ({"word": "anxious", "token_id": 70, "stratum": "adjective"}, {"word": "soldier", "token_id": 72, "stratum": "noun"})
    confirmation = cr.Confirmation025({"cardinal": 3, "coordinated-adjective": 3}, frames, tokens, SMALL.conditions, "c")
    tagged = confirmation.tagged_keys()
    assert len(tagged) == 2 * 2 * len(SMALL.conditions) and "cardinal-9|anxious|70|noun+0.32" in tagged and "cardinal-9|anxious|70" in confirmation.manifest_keys()
    assert confirmation.manifest() == {"S2-TARGET": sorted(tagged)} and not confirmation.manifest_keys() & tagged
    assert confirmation.counts() == {"strata": {"adjective": 1, "noun": 1}, "frames": 2, "conditions": len(SMALL.conditions), "runs": len(tagged)}
    assert cr.tagged_key(pm.Prompt(frames[0], 70, "anxious"), "base") == "cardinal-9|anxious|70|base"
    assert cr.untagged("cardinal-9|anxious|70|noun-0.32") == "cardinal-9|anxious|70" == cr.untagged("cardinal-9|anxious|70")
    cr.assert_ledger_isolated(sorted(tagged), frozenset({"cardinal-9|other|71"}), "x")
    for planted in ("cardinal-9|anxious|70", "cardinal-9|anxious|70|rand1+0.32"):
        with pytest.raises(cr.PhaseError, match="forbidden keys"):
            cr.assert_ledger_isolated([planted], frozenset({"cardinal-9|anxious|70"}), "x")


# ---------------------------------------------------------------------------
# The scores, the statistics, the outcome.


def test_the_count_rule_counts_ties_zeros_and_nan_against():
    assert cr.count_positive([1e-300, 0.0, -0.0, -1.0, float("nan"), float("inf"), None, 2.0]) == 2  # a non-finite value counts against too


def test_the_outcome_hierarchy():
    assert cr.outcome_label(False, True, True) == "CAUSAL_EFFECT_NOT_ESTABLISHED"
    assert cr.outcome_label(True, False, True) == "DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC"
    assert cr.outcome_label(True, True, False) == "READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED"
    assert cr.outcome_label(True, True, True) == "NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR"
    assert [label for _, label in cr.OUTCOME_TABLE] == list(cr.OUTCOMES) and set(cr.SEMANTICS["outcomes"]) == set(cr.OUTCOMES)


def _values(config: cr.Configuration, tokens, a_signs, random_size: float, g_signs):
    """Per-cue ℓ and D_attn with a chosen A sign, a chosen random |A| and a chosen G sign."""
    values = {}
    for index, token in enumerate(tokens):
        entry = {condition: {"ell": -2.0, "d_attn": 0.1, "nmse": 0.5, "ell_by_template": {"cardinal": -2.0}} for condition in config.conditions}
        a = 0.01 * (index + 1) * a_signs[index]
        entry[f"noun+{config.primary}"] = {**entry["base"], "ell": -2.0 + a, "d_attn": 0.1 + 0.001 * g_signs[index]}
        entry[f"noun-{config.primary}"] = {**entry["base"], "ell": -2.0 - a, "d_attn": 0.1 - 0.001 * g_signs[index]}
        for j in range(1, config.k_controls + 1):
            entry[f"rand{j}+{config.primary}"] = {**entry["base"], "ell": -2.0 + random_size * (-1) ** j}
            entry[f"rand{j}-{config.primary}"] = {**entry["base"], "ell": -2.0 - random_size * (-1) ** j}
        values[int(token["token_id"])] = entry
    return values


def test_the_statistics_the_counts_and_the_labels():
    tokens = ({"word": "a1", "token_id": 70, "stratum": "adjective"}, {"word": "a2", "token_id": 71, "stratum": "adjective"},
              {"word": "n1", "token_id": 72, "stratum": "noun"}, {"word": "n2", "token_id": 73, "stratum": "noun"})
    confirmation = cr.Confirmation025({}, (), tokens, SMALL.conditions, "c")
    results = cr.statistics(_values(SMALL, tokens, (1, 1, 1, -1), 0.005, (1, 1, 1, 1)), confirmation, SMALL)
    assert [row["A"] for row in results["per_cue"]] == pytest.approx([0.01, 0.02, 0.03, -0.04], abs=1e-14)
    assert results["per_cue"][0]["A_random"] == pytest.approx([-0.005, 0.005, -0.005], abs=1e-14)
    assert [row["B"] for row in results["per_cue"]] == pytest.approx([0.005, 0.015, 0.025, -0.045], abs=1e-14)
    assert (results["A"]["positive"], results["B"]["positive"], results["G"]["positive"]) == (3, 3, 4)
    assert results["A"]["by_stratum"] == {"adjective": 2, "noun": 1} and results["outcome"]["label"] == "NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR"
    assert cr.statistics(_values(SMALL, tokens, (1, 1, 1, -1), 0.015, (1, 1, 1, 1)), confirmation, SMALL)["outcome"]["label"] == "DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC"
    assert cr.statistics(_values(SMALL, tokens, (1, 1, -1, -1), 0.0, (1, 1, 1, 1)), confirmation, SMALL)["outcome"]["label"] == "CAUSAL_EFFECT_NOT_ESTABLISHED"
    assert cr.statistics(_values(SMALL, tokens, (1, 1, 1, 1), 0.0, (1, 0, 1, -1)), confirmation, SMALL)["outcome"]["label"] == "READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED"
    assert results["criterion"]["reference_tail"] == {"exact": "5/16", "value": 0.3125}
    assert [response["word"] for response in results["responses"]] == ["a1", "a2", "n1", "n2"] and list(results["responses"][3]["conditions"]) == list(SMALL.conditions)
    assert results["responses"][3]["conditions"]["noun-0.32"]["ell"] == pytest.approx(-2.0 + 0.04)


def test_the_descriptive_records_on_the_production_configuration():
    tokens = tuple({"word": f"w{k}", "token_id": 100 + k, "stratum": "adjective" if k < 20 else "noun"} for k in range(40))
    confirmation = cr.Confirmation025({}, (), tokens, cr.PRODUCTION.conditions, "c")
    signs = tuple(1 if k < 13 or k >= 20 else -1 for k in range(40))  # A: 13 of 20 adjectives, 20 of 20 nouns
    values = _values(cr.PRODUCTION, tokens, signs, 0.0, signs)
    results = cr.statistics(values, confirmation, cr.PRODUCTION)
    assert results["A"]["positive"] == 33 and results["A"]["result"] == "PASS" and results["A"]["by_stratum"] == {"adjective": 13, "noun": 20}
    records = cr.descriptives(values, results, confirmation, cr.PRODUCTION, 1.2992017464639374)
    assert records["strata_trigger"] == {"adjective": 14, "noun": 14}
    assert records["strata"]["A"]["adjective"] == {"positive": 13, "of": 20, "below_trigger": True} and records["strata"]["A"]["noun"]["below_trigger"] is False
    assert records["causal_fraction"]["fraction"] == pytest.approx(records["magnitudes"]["A"]["median"] / (1.2992017464639374 * 0.32))
    assert records["counts"]["A_half_positive"] == 0 and records["even_noun_mean"] == pytest.approx(0.0, abs=1e-15)


def test_the_routing_distance_is_the_equal_head_mean_total_variation():
    reference4 = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    reference5 = torch.tensor([[0.2, 0.3, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32)
    state = SimpleNamespace(rows4=reference4, rows5=reference5)
    padded = lambda rows: torch.cat([rows, torch.zeros(rows.shape[0], 2)], dim=1)  # the saved rows' zero padding beyond p_t
    assert cr.routing_distance(padded(reference4), padded(reference5), state) == 0.0
    captured4 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    captured5 = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.0, 0.5]])
    assert cr.routing_distance(padded(captured4), padded(captured5), state) == pytest.approx((1.0 + 1.0 + 0.0 + 0.5) / 4, abs=1e-7)
    with pytest.raises(pm.IncidentError, match="attention rows"):
        cr.routing_distance(captured4[:1], captured5, state)


def test_the_cue_log_mse_is_the_pooled_sse_over_the_pooled_count():
    frames = (pm.Frame("cardinal", "cardinal-9", (1, 2), (), {"sg": 5, "pl": 6}, "x {cue}"),
              pm.Frame("coordinated-adjective", "coordinated-adjective-9", (1, 2), (7,), {"sg": 5, "pl": 6}, "y {cue} z"))
    units = ul.table_units(({"word": "w", "token_id": 70, "stratum": "noun"},), frames)
    generator = torch.Generator().manual_seed(2)
    tensors, states = {}, {}
    for frame in frames:
        keys = frame.p_t + 1
        states[frame.frame_id] = SimpleNamespace(rows4=torch.full((2, keys), 1.0 / keys), rows5=torch.full((2, keys), 1.0 / keys))
    for group in ("cue_final", "coordinated"):
        for condition in ("base",):
            tensors[cr.tensor_name(group, condition, "dc")] = torch.randn(1, 5, generator=generator, dtype=torch.float64)
            tensors[cr.tensor_name(group, condition, "C")] = torch.randn(1, 5, generator=generator, dtype=torch.float64)
            tensors[cr.tensor_name(group, condition, "rows4")] = torch.full((1, 2, 4), 0.25)
            tensors[cr.tensor_name(group, condition, "rows5")] = torch.full((1, 2, 4), 0.25)
    values = cr.per_cue(units, tensors, states, ("base",))[70]["base"]
    sse = math.fsum(float(((tensors[cr.tensor_name(g, "base", "dc")] - tensors[cr.tensor_name(g, "base", "C")]) ** 2).sum()) for g in ("cue_final", "coordinated"))
    assert values["ell"] == pytest.approx(math.log(sse / 10), rel=1e-13) and values["frames"] == 2 and set(values["ell_by_template"]) == {"cardinal", "coordinated-adjective"}
    # cue-final: 3 keys, reference 1/3 each, captured 1/4 each: TV ½·3·(1/12) = 1/8 per head; coordinated: 4 keys, equal rows: 0
    assert values["d_attn"] == pytest.approx(0.5 * (0.125 + 0.0), abs=1e-7)


# ---------------------------------------------------------------------------
# Paths, the preregistration's rendering, the phase rules.


def test_the_scientific_paths():
    changed = ["src/neural_decompiler/cue_rotation.py", f"{cr.EXPERIMENT_DIR}/run.py", rr.CALIBRATION_RELATIVE_PATH, rr.LOCK_RELATIVE_PATH,
               rr.CONFIRMATION_RELATIVE_PATH, "experiments/023-block0-completion/exposed-cells.f64"]
    assert cr.scientific_changes(changed) == changed
    assert cr.scientific_changes([cr.CONFIRMATION_RELATIVE_PATH, cr.LOCK_RELATIVE_PATH, cr.PREREGISTRATION_RELATIVE_PATH, f"{cr.EXPERIMENT_DIR}/README.md",
                                  f"{cr.EXPERIMENT_DIR}/evidence/lock/REVIEW.md", f"{rr.EXPERIMENT_DIR}/README.md", f"{rr.EXPERIMENT_DIR}/evidence/x.md",
                                  "docs/superpowers/plans/x.md", "tests/test_cue_rotation.py"]) == []


def test_the_preregistration_rendering_does_not_depend_on_the_key_order():
    W_E, record, pool, tokens = _synthetic()
    frames = (pm.Frame("cardinal", "cardinal-9", (1, 2), (), {"sg": 5, "pl": 6}, "x {cue}"),)
    confirmation = cr.Confirmation025({"cardinal": 3}, frames, tokens, SMALL.conditions, "c" * 64)
    lock = cr.build_lock(run_id="r", protocol_code_commit="a" * 40, digests={"x": "y"}, config=SMALL, confirmation=confirmation, confirmation_file_sha256="f" * 64,
                         geometry=cr.geometry_block(W_E, record, pool, tokens, SMALL)["block"], nearest={}, dependencies={}, noun_keys=["n"])
    canonical = json.loads(pm.canonical_json(lock))
    assert list(canonical["statistics"]) != list(lock["statistics"])  # the installed file's order differs from the built one's
    assert cr.render_preregistration(canonical) == cr.render_preregistration(lock)
    text = cr.render_preregistration(lock)
    assert "## The intervention" in text and "≥" in text and all(key in text for key in cr.PATCH_PATH_SPENT_KEYS) and lock["content_sha256"] == rc.content_digest(lock)


def test_the_phase_rules():
    state = cr.new_results_state(digests={key: "d" for key in cr.DIGEST_KEYS}, protocol_code_commit="a" * 40, git_dirty=False, versions={}, config=SMALL)
    cr.assert_phase_allowed("lock", None)
    cr.assert_phase_allowed("lock", state)
    for phase, message in (("confirm", "requires the lock phase"), ("report", "report requires"), ("freeze", "unknown phase")):
        with pytest.raises(cr.PhaseError, match=message):
            cr.assert_phase_allowed(phase, state)
    with pytest.raises(cr.PhaseError, match="requires the results state"):
        cr.assert_phase_allowed("confirm", None)
    locked_out = {**state, "phases": {**state["phases"], "lock": {"status": "running", "incidents": [{"message": "m", "commit": "a" * 40}]}}}
    cr.assert_phase_allowed("report", locked_out)  # a lock incident alone can be reported
    assert "a lock incident is recorded" in cr.render_report({**locked_out, "run_id": "r"})


def test_the_report_names_a_stratum_below_its_trigger_on_a_global_pass():
    tokens = tuple({"word": f"w{k}", "token_id": 100 + k, "stratum": "adjective" if k < 20 else "noun"} for k in range(40))
    confirmation = cr.Confirmation025({}, (), tokens, cr.PRODUCTION.conditions, "c")
    signs = tuple(1 if k < 13 or k >= 20 else -1 for k in range(40))
    results = cr.statistics(_values(cr.PRODUCTION, tokens, signs, 0.0, signs), confirmation, cr.PRODUCTION)
    state = cr.new_results_state(digests={key: "d" for key in cr.DIGEST_KEYS}, protocol_code_commit="a" * 40, git_dirty=False, versions={},
                                 config=cr.PRODUCTION)
    state["confirmation"] = {"results": results, "gates": {}, "c_recompute": {}}
    text = cr.render_report(state)
    assert "A passes globally, but the adjective stratum has 13 of 20 positive, below the reporting trigger of 14" in text
    assert "B passes globally, but the adjective stratum has 13 of 20" in text and "noun stratum has" not in text
    assert cr.SEMANTICS["A"] in text and cr.SEMANTICS["patch_path"] in text
    with pytest.raises(cr.PhaseError, match="clean Git tree"):
        cr.new_results_state(digests={key: "d" for key in cr.DIGEST_KEYS}, protocol_code_commit="a" * 40, git_dirty=True, versions={}, config=SMALL)


# ---------------------------------------------------------------------------
# Tier C (the pinned model, opt-in): spent prompts only; no outcome of any run is computed.


def _smoke_enabled() -> None:
    import os

    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")


def _runner():
    spec = importlib.util.spec_from_file_location("experiment_025_runner_tier_c", ROOT / f"{cr.EXPERIMENT_DIR}/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Runner()


def _refuse_forwards(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a forward pass was attempted in a weights-only or tokenizer-only test")

    for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
        if hasattr(pm, name):
            monkeypatch.setattr(pm, name, refuse)


def _refuse_outcomes(monkeypatch, *, rotated: bool = True) -> None:
    """The effect-inspection guard: no ``ℓ``, statistic, routing distance, Level-1 value, ladder or descriptive can be
    computed. For a test that runs rotated vectors, the frozen readout (``ul.contrast_of``, ``level1_detail``) and
    every gate are refused too."""
    from neural_decompiler import readout_decompilation as rd

    def refuse(*args, **kwargs):
        raise AssertionError("an outcome quantity was computed in an engineering test")

    for name in ("per_cue", "statistics", "descriptives", "ladder", "routing_distance", "level1_gates", "run_gates"):
        monkeypatch.setattr(cr, name, refuse)
    for name in ("cue_mse", "fresh_pair_cells", "cue_mse_torch"):
        monkeypatch.setattr(rr, name, refuse)
    monkeypatch.setattr(rd.ReadoutProgram, "level1_detail", refuse)
    if rotated:
        for name in ("identity_gates", "recompute_c", "stage_two"):
            monkeypatch.setattr(cr, name, refuse)
        monkeypatch.setattr(ul, "contrast_of", refuse)


def _frozen_in_memory(runner):
    """The freeze's computation with the real tokenizer and the committed files, in memory: nothing is written."""
    from neural_decompiler.models import PYTHIA_70M

    base = runner._base()
    tokenizer = runner.tokenizer_loader(PYTHIA_70M)
    payload = cr.freeze_payload(tokenizer, pool=base.inputs.pool, freeze_024=base.inherited["confirmation"], prior=base.prior, config=cr.PRODUCTION)
    return tokenizer, base, payload, cr.confirmation_from_payload(payload, base.inputs.pool, cr.PRODUCTION)


def _candidate_ids(tokenizer) -> set[int]:
    words = [*cr.ADJECTIVE_RESERVES, *(w for pair in cr.ORDINARY_RESERVES for w in pair), *cr.NEW_NOUN_LIST, *(cr.regular_plural(w) for w in cr.NEW_NOUN_LIST)]
    return {ids[0] for word in words if len(ids := pm._encode(tokenizer, " " + word)) == 1}


@pytest.mark.pythia_smoke
def test_the_real_tokenizer_freeze_takes_the_designs_expected_picks_and_writes_nothing(monkeypatch):
    _smoke_enabled()
    _refuse_forwards(monkeypatch)
    target = ROOT / cr.CONFIRMATION_RELATIVE_PATH
    existed = target.exists()
    tokenizer, base, payload, confirmation = _frozen_in_memory(_runner())
    assert payload["picks"] == {key: list(value) for key, value in cr.EXPECTED_PICKS.items()} and payload["picks_match_expected"] is True
    assert payload["counts"] == {"strata": {"adjective": 20, "noun": 20}, "frames": 108, "conditions": 21, "runs": 90720}
    assert len(payload["manifest"]["S2-TARGET"]) == 90720 and not confirmation.manifest_keys() & base.forbidden
    reasons = {entry["candidate"]: entry["reason"] for entry in payload["rejected"]}
    assert "020's confirmation list" in reasons["statue/statues"]
    blocked = {key: frozenset(value["ids"]) for key, value in payload["blocked"].items()}
    eligible = [word for word in cr.NEW_NOUN_LIST if all(cr._status(tokenizer, form, blocked, set())[0] is not None for form in (word, cr.regular_plural(word)))]
    assert len(eligible) == 17 and eligible[:12] == list(cr.EXPECTED_PICKS["noun"][8:])
    assert len(payload["blocked"]["prior_nouns"]["ids"]) >= 48 and target.exists() == existed


@pytest.mark.pythia_smoke
def test_the_direction_is_024s_and_every_geometry_gate_holds_on_024s_spent_cues(monkeypatch):
    """Weights only: the direction reproduces 024's centroid digests and 024's locked scores; the geometry block on
    024's 40 spent cues passes every gate and reproduces bit for bit."""
    _smoke_enabled()
    _refuse_forwards(monkeypatch)
    from neural_decompiler.models import PYTHIA_70M, load_model

    runner = _runner()
    base = runner._base()
    W_E = pm.Weights.from_model(load_model(PYTHIA_70M)).W_E
    record, lock_024 = base.inherited["calibration"], base.inherited["lock"]
    d = cr.direction(W_E, record)
    mu_noun, mu_cue = cr.centroids(W_E, record)
    assert round(float(torch.linalg.vector_norm(d)), 3) == 1.358 and round(rr.cosine(mu_noun, mu_cue), 3) == 0.078
    assert round(float(cr.unit(d) @ cr.plurality_direction(W_E, base.inputs.pool)), 3) == 0.235
    for cue in lock_024["fresh"]["cues"]:
        assert abs(cr.score(d, W_E[cue["token_id"]]) - cue["nounness"]) < 1e-12, cue["word"]
    tokens = [{"word": cue["word"], "token_id": cue["token_id"], "stratum": "adjective" if cue["class"] == "N" else "noun"}
              for cue in base.inherited["confirmation"]["cues"]]
    first = cr.geometry_block(W_E, record, base.inputs.pool, tokens, cr.PRODUCTION)
    second = cr.geometry_block(W_E, record, base.inputs.pool, tokens, cr.PRODUCTION)
    assert first["block"] == second["block"] and len(first["block"]["cues"]) == 40
    for name, limit in cr.GEOMETRY_LIMITS.items():
        assert first["block"]["check_maxima"][name] <= cr.TOLERANCES[limit], name
    assert all(12.0 < math.degrees(cue["theta_primary"]) < 16.0 and abs(cue["even_primary"]) <= 0.017 for cue in first["block"]["cues"])


def _spent_setup(monkeypatch):
    """The four spent keys' frames, states and prompts, checked before the model loads: every key in 020's ledger, none
    a 025 manifest key, no cue a 025 candidate id. Then the model, with an allow-list on every forward."""
    from neural_decompiler.models import PYTHIA_70M, load_model

    runner = _runner()
    tokenizer, base, payload, confirmation = _frozen_in_memory(runner)
    ledger = set(base.inputs.closure["ledger"])
    candidates = _candidate_ids(tokenizer)
    tagged = set(payload["manifest"]["S2-TARGET"])
    frames = {frame.frame_id: frame for frame in base.inputs.pool.frames}
    for key in cr.PATCH_PATH_SPENT_KEYS:
        frame_id, _, token_id = key.split("|")
        assert key in ledger and key not in confirmation.manifest_keys() and not any(t.startswith(key + "|") for t in tagged)
        assert int(token_id) not in candidates and frame_id in frames
    allowed = set(cr.PATCH_PATH_SPENT_KEYS)
    executed: list[tuple[str, str, tuple]] = []
    inside: list[str] = []  # the low-level forwards run only inside an allow-listed call
    original_capture, original_patched = pm.capture_prompt, pm.run_patched
    original_run_capture, original_run_interventions = pm.run_capture, pm.run_interventions

    def guarded_capture(model, prompt, sites):
        if prompt.key not in allowed or int(prompt.cue_token_id) in candidates:
            raise AssertionError(f"a non-allow-listed prompt reached the model: {prompt.key}")
        executed.append(("plain", prompt.key, tuple(sites)))
        inside.append(prompt.key)
        try:
            return original_capture(model, prompt, sites)
        finally:
            inside.pop()

    def guarded_patched(model, prompt, replacements, sources, *, capture_sites=()):
        if prompt.key not in allowed or int(prompt.cue_token_id) in candidates or set(replacements) != {("EMBED", prompt.p_c)}:
            raise AssertionError(f"a non-allow-listed patched run reached the model: {prompt.key}")
        executed.append(("patched", prompt.key, tuple(capture_sites)))
        inside.append(prompt.key)
        try:
            return original_patched(model, prompt, replacements, sources, capture_sites=capture_sites)
        finally:
            inside.pop()

    def low_level(original):
        def guarded(*args, **kwargs):
            if not inside:
                raise AssertionError("a forward outside an allow-listed prompt")
            return original(*args, **kwargs)

        return guarded

    monkeypatch.setattr(pm, "capture_prompt", guarded_capture)
    monkeypatch.setattr(pm, "run_patched", guarded_patched)
    monkeypatch.setattr(pm, "run_capture", low_level(original_run_capture))
    monkeypatch.setattr(pm, "run_interventions", low_level(original_run_interventions))
    model = load_model(PYTHIA_70M)
    progs = ul.ModelPrograms.from_model(model, base.inputs)
    return SimpleNamespace(runner=runner, base=base, frames=frames, ledger=ledger, model=model, progs=progs, executed=executed, W_E=progs.weights.W_E)


@pytest.mark.pythia_smoke
def test_the_patch_path_holds_on_the_four_spent_keys(monkeypatch, capsys):
    _smoke_enabled()
    world = _spent_setup(monkeypatch)
    _refuse_outcomes(monkeypatch)
    check = cr.patch_path_check(world.model, world.frames, world.W_E, frozenset(world.ledger) | world.base.forbidden)
    assert check["passed"] and check["keys"] == list(cr.PATCH_PATH_SPENT_KEYS)
    assert all(set(entry) == {"plain_equals_patched_theta0", "embed_equals_resid_pre0", "logits_sha256", "slices_sha256"} for entry in check["checks"].values())
    assert [(kind, key) for kind, key, _ in world.executed] == [(kind, key) for key in cr.PATCH_PATH_SPENT_KEYS for kind in ("plain", "patched")]
    with capsys.disabled():
        print("\npatch path on the spent keys:", ", ".join(cr.PATCH_PATH_SPENT_KEYS))


@pytest.mark.pythia_smoke
def test_the_theta_zero_measurement_is_measure_prompt_and_the_vector_factors_are_the_token_ones(monkeypatch):
    """On the spent keys: ``measure_rotated`` with the model's own row reproduces ``ul.measure_prompt``'s Δx1, Δx3 and
    Δc bit for bit, and ``C`` from its Δx3; the vector gate factors equal ``ul.pair_factors`` bit for bit. No gate,
    distance or score is computed."""
    _smoke_enabled()
    world = _spent_setup(monkeypatch)
    _refuse_outcomes(monkeypatch, rotated=False)
    locked = world.base.inputs.closure["exploration"]["locked_states"]
    for key in cr.PATCH_PATH_SPENT_KEYS:
        frame_id, word, token_id = key.split("|")
        frame, token_id = world.frames[frame_id], int(token_id)
        state = ul.y1_states(locked, (frame,))[frame_id]
        plain = ul.measure_prompt(world.model, world.progs, frame, state, token_id, word)
        rotated = cr.measure_rotated(world.model, world.progs, frame, state, token_id, word, world.W_E[token_id])
        dx1 = cr.changes_from_raw(rotated["x1"], state.state_017.x1_all, frame)
        dx3 = cr.changes_from_raw(rotated["x3"], state.x3_all, frame)
        assert set(dx1) == set(plain["dx1"]) and all(torch.equal(dx1[p], plain["dx1"][p]) for p in dx1), key
        assert all(torch.equal(dx3[p], plain["dx3"][p]) for p in dx3) and torch.equal(rotated["dc"], plain["dc"]), key
        reference_id = int(world.base.inputs.pool.reference_ids[frame.template_id])
        x0_all = ul.reference_embeddings(world.progs.weights, frame, reference_id)
        token, vector = ul.pair_factors(world.progs, x0_all, frame, token_id, reference_id), cr.vector_factors(world.progs, x0_all, frame, world.W_E[token_id], reference_id)
        assert torch.equal(token.delta_e, vector.delta_e) and torch.equal(token.d_emb, vector.d_emb) and torch.equal(token.block0_pc.total, vector.block0_pc.total)
        assert (token.block0_pt is None) == (vector.block0_pt is None) and (token.block0_pt is None or torch.equal(token.block0_pt.total, vector.block0_pt.total))
    assert all(kind == "plain" or sites[-2:] == tuple(cr.attention_sites(world.frames[key.split("|")[0]])) for kind, key, sites in world.executed)


@pytest.mark.pythia_smoke
def test_the_vector_identity_gates_at_theta_zero_are_024s_token_gates_on_spent_keys(monkeypatch, tmp_path):
    """Saving, re-reading, the condition-tagged keys, ``C`` recomputed bit for bit, and I1, I3 and I4 through the vector
    path at θ = 0 against ``rr.target_gates``'s token path (``ul.stage_two_022``), bit for bit, on the four spent keys.
    The Level-1 identity, ``ℓ`` and ``D_attn`` are refused."""
    _smoke_enabled()
    world = _spent_setup(monkeypatch)
    _refuse_outcomes(monkeypatch, rotated=False)
    locked = world.base.inputs.closure["exploration"]["locked_states"]
    for key in cr.PATCH_PATH_SPENT_KEYS:
        frame_id, word, token_id = key.split("|")
        frame, token = world.frames[frame_id], {"word": word, "token_id": int(token_id), "stratum": "spent"}
        states = ul.y1_states(locked, (frame,))
        spent = cr.Confirmation025(dict(world.base.inputs.pool.reference_ids), (frame,), (token,), ("base",), "spent")
        executed: list[str] = []
        vectors32 = {int(token_id): {"base": world.W_E[int(token_id)]}}
        saved = tmp_path / f"{frame_id}.pt"
        rr.save_durably(cr.stage_two(world.model, world.progs, spent, states, vectors32, executed=executed), saved)
        tensors = torch.load(saved)
        units = cr.target_units(spent)
        assert executed == [f"{key}|base"] == sorted(spent.tagged_keys())
        cr.assert_measurements(tensors, units, ("base",), len(world.progs.scorable))
        assert cr.recompute_c(world.progs, units, states, tensors, ("base",))["bitwise_equal"]
        vector = cr.identity_gates(world.progs, spent, units, states, tensors, vectors32)
        spent_024 = rr.Confirmation024(dict(world.base.inputs.pool.reference_ids), (frame,), ({**token, "class": "spent", "lemma": word, "form": "word"},), "spent")
        measured = ul.measurement_tensors(ul.stage_two_022(world.model, world.progs, spent_024, {"Y1": states, "Y2": {}}, executed=[]))
        token_gates = rr.target_gates(world.progs, spent_024, rr.target_units(spent_024), states, measured)
        assert {name: vector[name]["max"] for name in ("I1", "I3", "I4")} == {name: token_gates[name]["max"] for name in ("I1", "I3", "I4")}, key
        assert all(vector[name]["max"] <= cr.TOLERANCES[name] for name in ("I1", "I3", "I4"))


@pytest.mark.pythia_smoke
def test_rotated_patches_land_exactly_with_embed_as_the_only_capture(monkeypatch):
    """On the spent keys, every rotated vector of the spent cue replaces ``EMBED@p_c`` exactly (the integrity record and
    the post-intervention capture); nothing but ``EMBED`` is captured and the logits are never read."""
    _smoke_enabled()
    world = _spent_setup(monkeypatch)
    _refuse_outcomes(monkeypatch)
    record = world.base.inherited["calibration"]
    tokens = [{"word": key.split("|")[1], "token_id": int(key.split("|")[2]), "stratum": "spent"} for key in cr.PATCH_PATH_SPENT_KEYS]
    vectors = cr.geometry_block(world.W_E, record, world.base.inputs.pool, tokens, cr.PRODUCTION)["vectors32"]
    for key, token in zip(cr.PATCH_PATH_SPENT_KEYS, tokens):
        frame = world.frames[key.split("|")[0]]
        site = ("EMBED", frame.p_c)
        for condition in ("noun+0.32", "noun-0.32", "rand1+0.32", "plur-0.32"):
            vector = vectors[token["token_id"]][condition]
            run = pm.run_patched(world.model, pm.Prompt(frame, token["token_id"], token["word"]), {site: vector.reshape(1, 1, -1)},
                                 {site: pm.ReplacementSource.DIRECT}, capture_sites=[site])
            assert len(run.integrity) == 1 and run.integrity[0]["outside_max_abs_change"] == 0.0
            assert set(run.slices) == {pm.site_label(site)} and torch.equal(run.vector(site).to(torch.float32), vector)
            assert not torch.equal(vector, world.W_E[token["token_id"]])
    assert all(sites == (("EMBED", world.frames[key.split("|")[0]].p_c),) for kind, key, sites in world.executed)
