"""Experiment 101's module (``read_intervention``): tier A (unit), plus tier C (the pinned model, opt-in, spent prompts
only; no 101 frame prompt and no outcome of any run is ever computed here)."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from neural_decompiler import cue_rotation as cr
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_intervention as ri
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import upstream_localization as ul
from test_plural_mechanism import ToyTokenizer

ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / ri.DESIGN["path"]


# ---------------------------------------------------------------------------
# Pins, constants and the configuration.


def test_the_pinned_blobs_are_the_design_commits():
    assert ri.assert_frozen_blobs() == ri.FROZEN_BLOBS and len(ri.FROZEN_BLOBS) == 20
    listing = subprocess.run(["git", "ls-tree", "516df20", "src/neural_decompiler/"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    at_commit = {line.split()[3].rsplit("/", 1)[1]: line.split()[2] for line in listing.splitlines()}
    assert {name: at_commit[name] for name in ri.FROZEN_BLOBS} == ri.FROZEN_BLOBS
    assert ri.verify_025_inputs(ROOT) == {"025_confirmation_file": ri.INHERITED_025["confirmation_file_sha256"],
                                          "025_confirmation_content": ri.INHERITED_025["confirmation_content_sha256"]}


def test_a_changed_pin_is_refused(monkeypatch):
    monkeypatch.setitem(ri.FROZEN_BLOBS, "capture.py", "0" * 40)
    with pytest.raises(ri.PhaseError, match="capture.py"):
        ri.assert_frozen_blobs()


def test_the_frame_candidates_are_design_section_11_verbatim():
    text = DESIGN_PATH.read_text(encoding="utf-8")
    section = text[text.index("## 11. The frames"):text.index("## 12.")]
    candidates = {}
    for line in section.splitlines():
        match = re.match(r"^\| (\d+) \| (.+?) \| (\d+) \| (.+?) \|$", line)
        if match:
            candidates[int(match.group(1))], candidates[int(match.group(3))] = match.group(2).strip(), match.group(4).strip()
    assert tuple(candidates[number] for number in range(1, 61)) == ri.FRAME_CANDIDATES
    adjectives = [text.rsplit(" ", 1)[1] for text in ri.FRAME_CANDIDATES]
    assert len(set(adjectives)) == 60 and not set(adjectives) & set(ri.NUMBER_WORDS)


def test_the_holm_thresholds_the_tails_and_the_rank_are_design_section_7s():
    assert ri.holm_thresholds(36) == (26, 25)
    assert ri.binomial_tail(36, 26) == Fraction(sum(math.comb(36, k) for k in range(26, 37)), 2 ** 36)
    assert round(float(ri.binomial_tail(36, 26)), 6) == 0.005665 and round(float(ri.binomial_tail(36, 25)), 6) == 0.014408
    assert float(ri.binomial_tail(36, 25)) > 0.0125 and float(ri.binomial_tail(36, 24)) > 0.025
    assert ri.nearest_rank(36, 90) == 33 and ri.interval_ranks(10_000) == (250, 9750)


def test_the_production_configuration_and_its_run_order():
    config = ri.PRODUCTION
    assert (config.n_frames, config.k_iso, config.reference, config.cues) == (36, 8, ("ref", 581), (("two", 767), ("three", 1264), ("many", 1142)))
    assert (config.d_threshold, config.p_threshold, config.precondition, config.tau_rank) == (25, 27, 30, 33)
    tags = config.tags()
    assert len(tags) == config.runs_per_frame == 55 and len(set(tags)) == 55
    assert tags[:4] == ("REF", "PL@two", "NOOP@two", "NEC-READ@two")
    assert tags[13:21] == tuple(f"NEC-ISO-{j}@two" for j in range(1, 9)) and tags[21:29] == tuple(f"SUF-ISO-{j}@two" for j in range(1, 9))
    assert tags[29:33] == ("HEAD-NEC@two", "HEAD-SUF@two", "HEAD-NEC-V@two", "HEAD-SUF-V@two")
    assert tags[33] == "PL@three" and tags[44] == "PL@many" and tags[-1] == "PC-SWAP-REF@many"
    assert sum(1 for tag in tags if tag.endswith("@two")) == 32 and sum(1 for tag in tags if tag.endswith("@three")) == 11
    assert ri.PREDICTED_ARMS == ("PL", "NEC-READ", "SUF-READ", "NEC-OFF", "SUF-OFF", "NEC-AMP", "SUF-NEG", "NEC-HALF", "SUF-HALF", "PC-SWAP-PL", "PC-SWAP-REF")


def test_an_inconsistent_configuration_is_refused():
    fields = dict(name="production", n_frames=36, frame_candidates=ri.FRAME_CANDIDATES, expected_picks=ri.EXPECTED_PICKS, reference=("ref", 581),
                  cues=(("two", 767), ("three", 1264), ("many", 1142)), k_iso=8, n_scored_nouns=79, exposed_frames=36, d_threshold=25, p_threshold=27,
                  precondition=30, tau_rank=33, max_overshoot_frames=6, max_weak_frames=6)
    assert ri.Configuration(**fields).to_json() == ri.PRODUCTION.to_json()
    for change in ({"p_threshold": 26}, {"precondition": 29}, {"tau_rank": 32}, {"cues": (("two", 767), ("two", 1264))}, {"reference": ("one", 581)},
                   {"expected_picks": tuple(range(2, 38))}, {"k_iso": 0}):
        with pytest.raises(ValueError):
            ri.Configuration(**{**fields, **change})


def test_the_tolerances_are_the_decided_ones_and_no_code_path_changes_them():
    assert ri.TOLERANCES == {"G1_read_weight": 1e-9, "G1_basis": 1e-12, "G2_read": 1e-4, "G2_length": 1e-4, "G2_direction": 1e-4, "G2_u_off32": 1e-6,
                             "G3_pass_through": 1e-4, "G4_I3": 1e-4, "G4_I4": 1e-3, "G4_level1": 2e-2, "G4_readout": 2e-2, "G4_logit": 2e-2,
                             "G5_row": 1e-4, "G5_output": 1e-3, "environment": 1e-9}
    source = (ROOT / "src/neural_decompiler/read_intervention.py").read_text(encoding="utf-8")
    assert not re.findall(r"TOLERANCES\[[^\]]+\]\s*=[^=]", source) and "TOLERANCES.update" not in source and "TOLERANCES.setdefault" not in source
    assert set(ri.TOLERANCE_GATES.values()) <= set(ri.TOLERANCES)


def test_the_tags_parse_and_fit_their_prompts():
    frame = _frames(1)[0]
    reference, cue = pm.Prompt(frame, 581, "ref"), pm.Prompt(frame, 767, "two")
    assert ri.parse_tag("REF") == ("REF", None) and ri.parse_tag("SUF-ISO-3@two") == ("SUF-ISO-3", "two")
    assert ri.arm_role("SUF-ISO-3") == "ref" and ri.arm_role("NEC-ISO-8") == "cue" and ri.arm_role("HEAD-SUF-V") == "ref" and ri.arm_role("PC-SWAP-PL") == "cue"
    for bad in ("NEC-READ", "@two", "NEC-READ@"):
        with pytest.raises(ri.PhaseError):
            ri.parse_tag(bad)
    with pytest.raises(ri.PhaseError, match="does not fit"):
        ri.run_arm(None, cue, "SUF-READ@two")
    with pytest.raises(ri.PhaseError, match="does not fit"):
        ri.run_arm(None, reference, "NEC-READ@two")
    with pytest.raises(ri.PhaseError):
        ri.capture_reference(None, None, cue)


# ---------------------------------------------------------------------------
# The keys.


def _frames(n: int, *, p_c: int = 5) -> list[pm.Frame]:
    return [pm.Frame(ri.TEMPLATE, ri.frame_id_for(k), tuple(range(100, 100 + p_c)), (5000 + k,), {"sg": 581, "pl": 767}, f"text {k} {{cue}} adj",
                     origin="extension") for k in range(1, n + 1)]


def test_the_s2_keys_are_1980_distinct_each_once_and_disjoint_from_s1():
    confirmation = ri.Confirmation101(tuple(_frames(36)), ri.PRODUCTION, "")
    s1, s2 = confirmation.s1_keys(), confirmation.s2_keys()
    assert len(s1) == 36 and len(s2) == 1980 and len(set(s2)) == 1980 and not set(s1) & set(s2)
    assert all(count == 1 for count in __import__("collections").Counter(s2).values())
    assert len(confirmation.untagged_keys()) == 144
    first = [key for key in s2 if key.startswith(ri.frame_id_for(1) + "|")]
    assert first[0] == "coordinated-adjective-101-01|ref|581|REF" and "coordinated-adjective-101-01|ref|581|SUF-READ@three" in first
    assert "coordinated-adjective-101-01|three|1264|PL@three" in first and len(first) == 55
    assert {cr.untagged(key) for key in first} == {"coordinated-adjective-101-01|ref|581", "coordinated-adjective-101-01|two|767",
                                                    "coordinated-adjective-101-01|three|1264", "coordinated-adjective-101-01|many|1142"}


# ---------------------------------------------------------------------------
# Geometry and edits (synthetic vectors).


def _geometry(d: int = 16, seed: int = 7) -> ri.ReadGeometry:
    generator = torch.Generator().manual_seed(seed)
    lock = {"read_weight": (torch.randn(d, generator=generator, dtype=torch.float64) + 0.3).tolist(),
            "axes_vectors": {"T": cr.unit(torch.randn(d, generator=generator, dtype=torch.float64)).tolist()}}
    return ri.read_geometry(lock)


def test_the_read_covector_is_the_centred_read_weight():
    geometry = _geometry()
    assert abs(float(geometry.w_c.mean())) < 1e-15 and abs(float(torch.linalg.vector_norm(geometry.w_hat)) - 1.0) < 1e-15
    assert geometry.norm2 == float(geometry.w_c @ geometry.w_c) and geometry.d_model == 16
    with pytest.raises(ri.IncidentError):
        ri.read_geometry({"read_weight": [1.0] * 4, "axes_vectors": {"T": [1.0, 0.0, 0.0, 0.0]}})


def test_the_isotropic_directions_are_orthonormal_to_the_basis_and_pinned():
    geometry = _geometry()
    directions = {"f": ri.iso_directions(geometry, "f", 8)}
    checks = ri.basis_checks(geometry, directions)
    assert all(value <= 1e-12 for value in checks.values()), checks
    v = directions["f"][0]
    assert torch.equal(v, ri.iso_direction(geometry, "f", 1)) and not torch.equal(v, ri.iso_direction(geometry, "g", 1))
    assert ri.ISO_TAG.format(frame_id="f", j=1) == "101|iso|f|1"
    assert rc.tensor_digest(ri.iso_direction(_geometry(), "coordinated-adjective-101-01", 1)) == ISO_DIGEST


ISO_DIGEST = "0d45b2b2e3d4bc0a8da7e0384278b71d1edd40572aa125e1b91a3a3bd23766da"  # the regression pin of the SHA-256 counter-mode generator


def _pair(geometry: ri.ReadGeometry, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    x_ref = (torch.randn(geometry.d_model, generator=generator) * 4.0).to(torch.float32)
    x_cue = (x_ref.double() + torch.randn(geometry.d_model, generator=generator, dtype=torch.float64) * 1.5).to(torch.float32)
    return x_ref, x_cue, ri.pair_quantities(x_cue.double() - x_ref.double(), geometry)


def test_the_pair_quantities_and_every_edit_have_their_exact_geometry():
    geometry = _geometry()
    x_ref, x_cue, pair = _pair(geometry)
    assert abs(float(pair.read @ geometry.w_c) - pair.dr) < 1e-12 and abs(float(pair.u_off @ geometry.w_c)) < 1e-12 and abs(float(pair.u_off.mean())) < 1e-15
    assert abs(float(torch.linalg.vector_norm(pair.u_off)) - 1.0) < 1e-12 and pair.s == abs(pair.dr) / geometry.norm
    assert pair.record()["overshoot"] == (pair.off_norm < pair.s)
    v = ri.iso_direction(geometry, "f", 1)
    arms = {arm.name: arm for arm in ri.primary_arms(2)}
    for name, arm in arms.items():
        if not arm.residual_edit:
            continue
        replacement = ri.residual_replacement(arm, x_ref, x_cue, pair, v if arm.kind == "iso" else None)
        assert replacement.dtype == torch.float32 and tuple(replacement.shape) == (1, 1, geometry.d_model)
        values = ri.g2_values(arm, x_ref, x_cue, replacement.reshape(-1), pair, geometry)
        assert all(value <= 1e-5 for value in values.values()), (name, values)
        if arm.kind in ("noop", "swap"):
            expected = x_cue if (arm.kind == "noop" or arm.prompt == "ref") else x_ref
            assert torch.equal(replacement.reshape(-1), expected)
    half = ri.residual_replacement(arms["NEC-HALF"], x_ref, x_cue, pair).reshape(-1).double()
    assert abs(float(half @ geometry.w_c) - (float(x_cue.double() @ geometry.w_c) - pair.dr / 2)) < 1e-5
    off = ri.residual_replacement(arms["NEC-OFF"], x_ref, x_cue, pair).reshape(-1).double() - x_cue.double()
    assert abs(float(torch.linalg.vector_norm(off - off.mean())) - pair.s) / pair.s < 1e-5
    assert max(ri.u_off32_values(pair, geometry).values()) <= 1e-6


def test_a_wrong_edit_fails_its_g2_check():
    geometry = _geometry()
    x_ref, x_cue, pair = _pair(geometry)
    arm = next(arm for arm in ri.MAIN_ARMS if arm.name == "NEC-READ")
    wrong = (x_cue.double() - 0.9 * pair.read).to(torch.float32)
    values = ri.g2_values(arm, x_ref, x_cue, wrong, pair, geometry)
    assert values["r_error"] > 1e-4 and values["length_error"] > 1e-4
    off = next(arm for arm in ri.MAIN_ARMS if arm.name == "SUF-OFF")
    tilted = (x_ref.double() + pair.s * cr.unit(pair.u_off + 0.1 * geometry.w_hat)).to(torch.float32)
    assert ri.g2_values(off, x_ref, x_cue, tilted, pair, geometry)["direction_error"] > 1e-4


def test_the_predicted_changes_are_design_section_5s_table():
    geometry = _geometry()
    frame = _frames(1)[0]
    generator = torch.Generator().manual_seed(11)
    dx = {frame.p_c: torch.randn(16, generator=generator, dtype=torch.float64), frame.p_t: torch.randn(16, generator=generator, dtype=torch.float64)}
    pair = ri.pair_quantities(dx[frame.p_c], geometry)
    zero = torch.zeros(16, dtype=torch.float64)
    for arm in ri.PREDICTED_ARMS:
        change = ri.predicted_change(arm, dx, pair, frame)
        assert set(change) == {frame.p_c, frame.p_t} and all(t.dtype == torch.float64 for t in change.values())
    assert torch.equal(ri.predicted_change("SUF-READ", dx, pair, frame)[frame.p_t], zero)
    assert torch.equal(ri.predicted_change("PC-SWAP-PL", dx, pair, frame)[frame.p_c], zero)
    assert torch.equal(ri.predicted_change("NEC-READ", dx, pair, frame)[frame.p_c], dx[frame.p_c] - pair.read)
    assert torch.equal(ri.predicted_change("SUF-OFF", dx, pair, frame)[frame.p_c], pair.s * pair.u_off)


def test_the_layer_norm_records():
    from neural_decompiler import head_transport as ht

    d = 16
    head = ht.HeadWeights(3, 4, torch.ones(d, dtype=torch.float64) * 2.0, torch.zeros(d, dtype=torch.float64), 1e-5, torch.eye(d, 4, dtype=torch.float64),
                          torch.zeros(4, dtype=torch.float64), torch.eye(4, d, dtype=torch.float64))
    x = torch.arange(d, dtype=torch.float64)
    e = torch.ones(d, dtype=torch.float64)
    record = ri.layer_norm_records(head, torch.ones(d, dtype=torch.float64) * 3.0, x, e)
    sigma = float(torch.sqrt(((x - x.mean()) ** 2).mean() + 1e-5))
    assert abs(record["sigma"] - sigma) < 1e-12 and abs(record["gamma3_e_over_sigma"] - 2.0 * 4.0 / sigma) < 1e-12
    assert abs(record["gamma_ln2_e_over_sigma"] - 3.0 * 4.0 / sigma) < 1e-12


# ---------------------------------------------------------------------------
# The freeze, on a toy tokenizer.


def toy_tokenizer(extra: dict[str, int] | None = None) -> ToyTokenizer:
    vocabulary = {" one": 581, " two": 767, " three": 1264, " many": 1142, " anxious": 900_001, " author": 900_002, " soldier": 900_003, " soldiers": 900_004}
    next_id = 10_000
    for text in ri.FRAME_CANDIDATES:
        for index, piece in enumerate(text.replace("{cue}", "").split()):
            word = piece if index == 0 else " " + piece
            if word not in vocabulary:
                vocabulary[word] = next_id
                next_id += 1
    vocabulary.update(extra or {})
    return ToyTokenizer(vocabulary)


def _exclusions(*, earlier=(900,), candidates=("anxious",), targets=(902,), prior=(903,), texts=("Old {cue} x",), frame_tokens=(10_004,)) -> ri.Exclusions:
    c23 = {"exclusion": {"frame_texts": list(texts)}, "frames": [{"text_template": "Another {cue} y"}], "content_sha256": "a" * 64}
    c24 = {"exclusion": {"cue_token_ids": list(earlier)}, "cues": [{"token_id": 901}], "target_noun_form_ids": {"ids": list(targets)}, "content_sha256": "b" * 64}
    c25 = {"blocked": {"earlier_cues": {"ids": sorted({*earlier, 901})}, "target_forms": {"ids": list(targets)}, "prior_nouns": {"ids": list(prior)},
                       "frame_tokens": {"ids": list(frame_tokens)}},
           "cues": [{"token_id": 904}], "candidates": {"adjective-reserve": list(candidates), "new-list": ["author"], "ordinary-reserve": [["soldier", "soldiers"]]},
           "content_sha256": "c" * 64}
    return ri.exclusions_from_committed(confirmation_023=c23, confirmation_024=c24, confirmation_025=c25,
                                       prior={"ids": list(prior), "path": "p.json", "file_sha256": "d" * 64})


def test_the_freeze_takes_the_first_36_eligible_candidates_and_its_manifests():
    payload = ri.freeze_payload(toy_tokenizer(), exclusions=_exclusions(), config=ri.PRODUCTION, model={"model_id": "toy", "revision": "0"})
    assert payload["picks"] == list(range(1, 37)) and payload["picks_match_expected"] is True and len(payload["candidates"]) == 60
    assert all(entry["eligible"] for entry in payload["candidates"]) and payload["counts"]["S2-TARGET"] == 1980
    confirmation = ri.confirmation_from_payload(payload, ri.PRODUCTION)
    assert [frame.frame_id for frame in confirmation.frames] == [ri.frame_id_for(k) for k in range(1, 37)]
    assert payload["manifest"] == {"S1-REF": sorted(confirmation.s1_keys()), "S2-TARGET": sorted(confirmation.s2_keys())}
    assert payload["content_sha256"] == rc.content_digest(payload)
    tampered = json.loads(json.dumps(payload))
    tampered["frames"][0]["frame_id"] = ri.frame_id_for(40)
    tampered["content_sha256"] = rc.content_digest(tampered)
    with pytest.raises(ri.PhaseError):
        ri.confirmation_from_payload(tampered, ri.PRODUCTION)


def test_soft_and_dry_stay_eligible_although_they_are_tokens_of_exposed_frames():
    tokenizer = toy_tokenizer()
    exclusions = _exclusions(frame_tokens=(tokenizer.vocabulary[" soft"], tokenizer.vocabulary[" dry"]))
    payload = ri.freeze_payload(tokenizer, exclusions=exclusions, config=ri.PRODUCTION)
    status = {entry["adjective"]: entry for entry in payload["candidates"]}
    assert status["soft"]["eligible"] and status["dry"]["eligible"]
    assert "frame_tokens" in ri.FREEZE_RULES["not_applied"]


def test_an_excluded_adjective_shifts_the_picks_and_the_freeze_stops_with_nothing_to_write():
    tokenizer = toy_tokenizer()
    large = tokenizer.vocabulary[" large"]
    for exclusions in (_exclusions(earlier=(900, large)), _exclusions(targets=(large,)), _exclusions(candidates=("large",)), _exclusions(prior=(large,))):
        with pytest.raises(ri.FreezeDeviation):
            ri.freeze_payload(tokenizer, exclusions=exclusions, config=ri.PRODUCTION)
    with pytest.raises(ri.FreezeDeviation):
        ri.freeze_payload(tokenizer, exclusions=_exclusions(texts=(ri.FRAME_CANDIDATES[0],)), config=ri.PRODUCTION)
    statuses = []
    try:
        ri.freeze_payload(tokenizer, exclusions=_exclusions(earlier=(900, large)), config=ri.PRODUCTION)
    except ri.FreezeDeviation as error:
        statuses.append(str(error))
    assert "2, 3" in statuses[0] and "37" in statuses[0]


def test_a_shortfall_and_a_structural_failure_are_recorded():
    tokenizer = toy_tokenizer()
    adjectives = [tokenizer.vocabulary[" " + text.rsplit(" ", 1)[1]] for text in ri.FRAME_CANDIDATES[:25]]
    with pytest.raises(ri.FreezeShortfall):
        ri.freeze_payload(tokenizer, exclusions=_exclusions(earlier=(900, *adjectives)), config=ri.PRODUCTION)
    # a longer piece swallows the stratum cue in candidate 1: its structure fails for many, and the picks shift by one
    split = toy_tokenizer({" many large": 30_000})
    with pytest.raises(ri.FreezeDeviation):
        ri.freeze_payload(split, exclusions=_exclusions(), config=ri.PRODUCTION)
    shifted = ri.Configuration(**{**_fields(), "expected_picks": tuple(range(2, 38))})
    payload = ri.freeze_payload(split, exclusions=_exclusions(), config=shifted)
    first = payload["candidates"][0]
    assert not first["eligible"] and any(reason.startswith("structure (many)") for reason in first["reasons"]) and payload["picks"] == list(range(2, 38))


def _fields() -> dict:
    return dict(name="toy", n_frames=36, frame_candidates=ri.FRAME_CANDIDATES, expected_picks=ri.EXPECTED_PICKS, reference=("ref", 581),
                cues=(("two", 767), ("three", 1264), ("many", 1142)), k_iso=8, n_scored_nouns=79, exposed_frames=36, d_threshold=25, p_threshold=27,
                precondition=30, tau_rank=33, max_overshoot_frames=6, max_weak_frames=6)


def test_the_committed_exclusions_cross_check_their_sources():
    with pytest.raises(ri.PhaseError, match="earlier-cue"):
        ri.exclusions_from_committed(
            confirmation_023={"exclusion": {"frame_texts": []}, "frames": [], "content_sha256": "a" * 64},
            confirmation_024={"exclusion": {"cue_token_ids": [1]}, "cues": [], "target_noun_form_ids": {"ids": []}, "content_sha256": "b" * 64},
            confirmation_025={"blocked": {"earlier_cues": {"ids": [2]}, "target_forms": {"ids": []}, "prior_nouns": {"ids": []}}, "cues": [],
                              "candidates": {"adjective-reserve": [], "new-list": [], "ordinary-reserve": []}, "content_sha256": "c" * 64},
            prior={"ids": [], "path": "p", "file_sha256": "d" * 64})


def test_the_real_committed_exclusions_have_the_reviewed_counts():
    exclusions = _real_exclusions()
    record = exclusions.record
    assert (record["frame_texts"]["count"], record["earlier_cue_ids"]["count"], record["earlier_cue_ids_024"], record["target_form_ids"]["count"],
            record["prior_noun_ids"]["count"]) == (162, 431, 391, 161, 48)


def _real_exclusions() -> ri.Exclusions:
    from neural_decompiler import block0_completion as b0c
    from neural_decompiler import readout_routing as rr

    return ri.exclusions_from_committed(confirmation_023=json.loads((ROOT / b0c.CONFIRMATION_RELATIVE_PATH).read_text()),
                                        confirmation_024=json.loads((ROOT / rr.CONFIRMATION_RELATIVE_PATH).read_text()),
                                        confirmation_025=ri.load_025(ROOT), prior=cr.prior_nouns(ROOT))


# ---------------------------------------------------------------------------
# The stage-1 files.


def _locked_entry(frame: pm.Frame, d: int = 8, heads: int = 8, nouns: int = 80, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    keys = frame.p_t + 1
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)  # noqa: E731
    return {"p_c": frame.p_c, "p_t": frame.p_t, "x1_all": rand(keys, d).tolist(), "x2_all": rand(keys, d).tolist(), "x3_all": rand(keys, d).tolist(),
            "x4_all": rand(keys, d).tolist(), "x5_all": rand(keys, d).tolist(), "h6": rand(d).tolist(), "rows4": rand(heads, keys).tolist(),
            "rows5": rand(heads, keys).tolist(), "c_ref": rand(nouns).tolist()}


def test_the_stage1_states_round_trip_through_the_table_with_their_digests(tmp_path):
    frames = _frames(3)
    locked = {frame.frame_id: _locked_entry(frame, seed=i) for i, frame in enumerate(frames)}
    layout, meta = ri.states_layout(frames, 8, 8, 80), ri.states_meta(frames)
    index = ul.write_table(tmp_path / "s.f64", tmp_path / "s.json", ri.states_blocks(locked, frames), meta)
    ul.verify_table_index(index, layout=layout, meta=meta)
    back = ri.states_from_blocks(ul.read_table(tmp_path / "s.f64", index), frames)
    assert {k: rd.state_digest(v) for k, v in back.items()} == {k: rd.state_digest(v) for k, v in locked.items()}
    with pytest.raises(ri.IncidentError):
        ul.verify_table_index(index, layout=ri.states_layout(frames, 8, 4, 80), meta=meta)


# ---------------------------------------------------------------------------
# The statistics.


def _value(*, full=-4.0, N=0.3, S=0.3, A_amp=-0.5, A_neg=0.5, F_rem=0.4, F_ins=0.4, weak=None):
    weak = full > ri.WEAK_SHIFT if weak is None else weak
    fractions = None if weak else {"F_rem": F_rem, "F_rem_off": F_rem + N, "F_ins": F_ins, "F_ins_off": F_ins - S, "N": N, "S": S}
    return {"dc_full": full, "weak": weak, "A_amp": A_amp, "A_neg": A_neg, "fractions": fractions}


def _prediction(*, F_rem=0.4, F_ins=0.4, weak=False):
    return {"weak": weak, "fractions": None if weak else {"F_rem": F_rem, "F_ins": F_ins, "N": 0.3, "S": 0.3}}


def _stats(values, predictions=None, tau=(0.05, 0.05), config=ri.PRODUCTION):
    predictions = predictions or [_prediction() for _ in values]
    return ri.primary_statistics([f"f{i}" for i in range(len(values))], values, predictions, tau[0], tau[1], config)


def test_every_label_row():
    good = [_value() for _ in range(36)]
    assert _stats(good)["outcome"]["label"] == "READ_COMPONENT_EFFECTS_AS_PREDICTED"
    assert _stats(good, tau=(0.25, 0.05))["outcome"]["label"] == "READ_COMPONENT_EFFECTS_PROGRAM_NOT_EVALUABLE"
    assert _stats(good, [_prediction(F_rem=0.9) for _ in range(36)])["outcome"]["label"] == "READ_COMPONENT_EFFECTS_PROGRAM_NOT_CONFIRMED"
    assert _stats([_value(A_amp=0.1) for _ in range(36)])["outcome"]["label"] == "READ_COMPONENT_EFFECTS_DIRECTION_NOT_ESTABLISHED"
    assert _stats([_value(S=-0.1) for _ in range(36)])["outcome"]["label"] == "READ_REMOVAL_EFFECT_INSTALLATION_NOT_ESTABLISHED"
    assert _stats([_value(N=-0.1) for _ in range(36)])["outcome"]["label"] == "READ_INSTALLATION_EFFECT_REMOVAL_NOT_ESTABLISHED"
    assert _stats([_value(N=-0.1, S=-0.1) for _ in range(36)])["outcome"]["label"] == "READ_COMPONENT_EFFECTS_NOT_ESTABLISHED"
    weak = [_value(full=-0.5) for _ in range(7)] + [_value() for _ in range(29)]
    assert _stats(weak)["outcome"]["label"] == "BEHAVIOR_PRECONDITION_NOT_MET"
    assert _stats([_value(full=-1.0)] * 6 + [_value()] * 30)["precondition"]["met"] is True  # exactly −1 is not weak


def _counted(r: int, i: int, **extra) -> list[dict]:
    return [_value(N=0.3 if k < r else -0.3, S=0.3 if k < i else -0.3, **extra) for k in range(36)]


def test_holm_the_thresholds_and_a_failed_floor():
    stats = _stats(_counted(26, 25))
    assert (stats["R"]["count"], stats["I"]["count"]) == (26, 25) and stats["R"]["result"] == stats["I"]["result"] == "PASS"
    stats = _stats(_counted(25, 25))
    assert not stats["R"]["holm_rejected"] and not stats["I"]["holm_rejected"] and stats["R"]["result"] == stats["I"]["result"] == "FAIL"
    stats = _stats(_counted(26, 24))
    assert stats["R"]["result"] == "PASS" and stats["I"]["result"] == "FAIL" and stats["outcome"]["label"] == "READ_REMOVAL_EFFECT_INSTALLATION_NOT_ESTABLISHED"
    low = [_value(N=0.01, S=0.3) for _ in range(36)]
    stats = _stats(low)
    assert stats["R"]["holm_rejected"] and stats["R"]["result"] == "FAIL" and stats["I"]["result"] == "PASS"  # a failed floor leaves I's threshold alone
    assert ri.holm_decisions({"R": Fraction(1, 100), "I": Fraction(2, 100)}) == {"R": True, "I": True}
    assert ri.holm_decisions({"R": Fraction(2, 100), "I": Fraction(2, 100)}) == {"R": False, "I": False}
    assert ri.holm_decisions({"R": Fraction(1, 80), "I": Fraction(1, 40)}) == {"R": True, "I": True}


def test_weak_frames_fail_every_count_and_take_the_worst_value_and_ties_fail():
    values = [_value(full=-0.2) for _ in range(18)] + [_value() for _ in range(18)]
    stats = _stats(values)
    assert stats["R"]["count"] == 18 and stats["R"]["median"] is None and stats["R"]["median_nonfinite"] == "-inf"
    assert stats["D"]["A_amp_negative"] == 18 and stats["D"]["median_A_amp"] is None and stats["D"]["median_A_amp_nonfinite"] == "+inf"
    pm.canonical_json(stats)  # many weak frames: the result is still writable
    ties = [_value(N=0.0, S=0.0, A_amp=0.0, A_neg=0.0) for _ in range(36)]
    stats = _stats(ties)
    assert (stats["R"]["count"], stats["I"]["count"], stats["D"]["A_amp_negative"], stats["D"]["A_neg_positive"]) == (0, 0, 0, 0)
    predicted_weak = [_prediction(weak=True) for _ in range(36)]
    assert _stats([_value() for _ in range(36)], predicted_weak)["P"]["agreements"] == 0


def test_p_uses_the_floor_the_cap_and_less_or_equal():
    values = [_value(F_rem=0.25, F_ins=0.25) for _ in range(36)]
    within_floor = [_prediction(F_rem=0.25 + 2 ** -6, F_ins=0.25 - 2 ** -6) for _ in range(36)]  # 0.015625 ≤ the 0.02 floor
    assert _stats(values, within_floor, tau=(0.001, 0.001))["P"]["agreements"] == 36
    at_tau = [_prediction(F_rem=0.25 + 2 ** -5, F_ins=0.25) for _ in range(36)]  # exactly τ = 0.03125: ≤ agrees
    assert _stats(values, at_tau, tau=(2 ** -5, 0.001))["P"]["agreements"] == 36
    assert _stats(values, at_tau, tau=(0.001, 0.001))["P"]["agreements"] == 0  # 0.03125 > the floor
    assert _stats(values, tau=(0.20, 0.20))["P"]["evaluable"] is True and _stats(values, tau=(0.2000001, 0.1))["P"]["evaluable"] is False
    with pytest.raises(ri.IncidentError):
        _stats(values[:35])


def test_the_median_rule_and_the_bootstrap():
    assert ri.median([1.0, 2.0, 3.0, 4.0]) == 2.5 and ri.median([3.0, 1.0, 2.0]) == 2.0 and ri.median([]) is None
    assert ri.median([-math.inf] * 18 + [1.0] * 18) == -math.inf and ri.median([-math.inf] * 17 + [1.0] * 19) == 1.0
    assert math.isnan(ri.median([-math.inf, math.inf]))
    index = ri.bootstrap_index(36)
    assert index.shape == (10_000, 36) and torch.equal(index, ri.bootstrap_index(36))
    values = [float(v) for v in torch.linspace(-1, 1, 36)]
    identity = torch.arange(36).unsqueeze(0)
    assert float(ri.row_medians(torch.tensor(values, dtype=torch.float64)[identity])[0]) == ri.median(values)
    with_none = [None if i % 5 == 0 else v for i, v in enumerate(values)]
    entry = ri.median_with_interval(with_none, index)
    assert entry["n"] == 28 and entry["excluded"] == 8 and entry["interval"][0] <= entry["median"] <= entry["interval"][1]
    infinite = ri.median_with_interval([-math.inf] * 30 + [0.5] * 6, index)
    assert infinite["median"] == -math.inf and infinite["interval"][1] == -math.inf


def test_the_nonfinite_encoding():
    encoded = ri.encode_nonfinite({"a": -math.inf, "b": [1.0, math.inf, math.nan], "c": {"d": math.nan}, "e": 2.0})
    assert encoded == {"a": None, "a_nonfinite": "-inf", "b": [1.0, None, None], "b_nonfinite_at": {"1": "+inf", "2": "nan"}, "c": {"d": None, "d_nonfinite": "nan"},
                       "e": 2.0}
    pm.canonical_json(encoded)
    assert ri.ratio(1.0, 0.0) == math.inf and ri.ratio(-1.0, 0.0) == -math.inf and math.isnan(ri.ratio(0.0, 0.0)) and ri.ratio(1.0, -0.0) == -math.inf


def test_the_measured_values_the_weak_switch_and_the_head_guards():
    config = ri.Configuration(**{**_fields(), "k_iso": 2})
    frame = _frames(1)[0]
    schedule = ri.Confirmation101((frame,), config, "").schedule(frame)
    index = {run.tag: row for row, run in enumerate(schedule)}
    contrasts = torch.zeros(len(schedule), 79, dtype=torch.float64)
    shifts = {"PL@two": -4.0, "NEC-READ@two": -2.0, "NEC-OFF@two": -3.0, "SUF-READ@two": -2.0, "SUF-OFF@two": -1.0, "NEC-AMP@two": -5.0, "SUF-NEG@two": 0.5,
              "NEC-HALF@two": -3.0, "SUF-HALF@two": -1.0, "PC-SWAP-PL@two": -0.5, "PC-SWAP-REF@two": -3.5, "NEC-ISO-1@two": -3.0, "NEC-ISO-2@two": -4.0,
              "SUF-ISO-1@two": -1.0, "SUF-ISO-2@two": 0.0, "HEAD-NEC@two": -3.0, "HEAD-NEC-V@two": -3.5, "HEAD-SUF@two": -1.0, "HEAD-SUF-V@two": -0.05,
              "PL@three": -0.5, "NEC-READ@three": -0.25, "PL@many": -2.0, "SUF-READ@many": -0.05}
    for tag, value in shifts.items():
        contrasts[index[tag]] = value
    tensors = {f"{frame.frame_id}/contrasts": contrasts}
    values = ri.frame_values(schedule, tensors, config, fractions_for_weak=False)
    two = values["two"]["fractions"]
    assert values["two"]["dc_full"] == -4.0 and not values["two"]["weak"] and values["two"]["A_amp"] == -1.0 and values["two"]["A_neg"] == 0.5
    assert (two["F_rem"], two["F_rem_off"], two["F_ins"], two["F_ins_off"], two["N"], two["S"]) == (0.5, 0.75, 0.5, 0.25, 0.25, 0.25)
    assert two["iso_rem_margin"] == pytest.approx(0.875 - 0.5) and two["iso_ins_margin"] == pytest.approx(0.5 - 0.125)
    assert two["head_removal_share"] == 0.5 and two["head_install_share"] == 0.5 and two["head_install_share_v"] == pytest.approx(0.025)
    assert values["three"]["weak"] and values["three"]["fractions"] is None and values["three"]["reason"] == "weak"
    switched = ri.frame_values(schedule, tensors, config, fractions_for_weak=True)
    assert switched["three"]["fractions"]["F_rem"] == 0.5
    assert values["many"]["fractions"]["F_ins"] == pytest.approx(0.025)
    guarded = dict(shifts, **{"SUF-READ@two": -0.1})
    for tag, value in guarded.items():
        contrasts[index[tag]] = value
    assert ri.frame_values(schedule, tensors, config, fractions_for_weak=False)["two"]["fractions"]["head_install_share"] is None


def _calibration_inputs(n=36, *, weak=0, overshoot=0, nonfinite=False):
    config = ri.PRODUCTION
    frame_ids = [f"f{i:02d}" for i in range(n)]
    values, predicted, pairs = [], [], {}
    for i, frame_id in enumerate(frame_ids):
        full = -0.5 if i < weak else -4.0
        error = (i + 1) / 1000.0  # |F̂_rem − F_rem| = (i + 1)/1000: the 33rd of 36 sorted is 0.033
        fractions = {"F_rem": 0.5, "F_rem_off": 0.8, "F_ins": 0.5, "F_ins_off": 0.2, "N": 0.3, "S": 0.3, "F_half_rem": 0.7, "F_half_ins": 0.3,
                     "F_swap_pl": 0.1, "F_swap_ref": 0.9, "additivity": 1.0}
        if nonfinite and i == 3:
            fractions["F_rem"] = math.inf
        values.append({"dc_full": full, "weak": full > ri.WEAK_SHIFT, "A_amp": -0.5, "A_neg": 0.5, "fractions": fractions})
        predicted.append({"weak": False, "fractions": {"F_rem": 0.5 + error, "F_ins": 0.5 - error / 2}})
        pairs[frame_id] = {label: {"s": 2.0 + i / 100, "overshoot": i < overshoot} for label in config.cue_labels}
    table = {"content_sha256": "t" * 64, "frames": {frame_id: {"cues": {label: predicted[i] for label in config.cue_labels}} for i, frame_id in enumerate(frame_ids)}}
    values_by_cue = {label: values for label in config.cue_labels}
    return frame_ids, values_by_cue, table, pairs


def test_calibration_tau_is_the_33rd_of_all_36_and_the_triggers_fire_above_six():
    frame_ids, values_by_cue, table, pairs = _calibration_inputs()
    results = ri.calibration_results(frame_ids, values_by_cue, table, pairs, ri.GateBook(), {"passed": True}, ri.PRODUCTION)
    assert results["tau"]["rem"] == pytest.approx(0.033) and results["tau"]["ins"] == pytest.approx(0.0165) and results["tau"]["evaluable"] is True
    assert not results["stopped"] and results["min_s"]["two"] == {"s": 2.0, "at": "f00"} and results["table_content_sha256"] == "t" * 64
    for weak, fired in ((6, False), (7, True)):
        results = ri.calibration_results(*_calibration_inputs(weak=weak), ri.GateBook(), {"passed": True}, ri.PRODUCTION)
        assert results["triggers"]["weak"]["fired"] is fired and results["stopped"] is fired
        assert results["tau"]["rem"] is not None  # weak frames' fractions enter τ (calibration's switch is on)
    for overshoot, fired in ((6, False), (7, True)):
        results = ri.calibration_results(*_calibration_inputs(overshoot=overshoot), ri.GateBook(), {"passed": True}, ri.PRODUCTION)
        assert results["triggers"]["overshoot"]["fired"] is fired
    results = ri.calibration_results(*_calibration_inputs(nonfinite=True), ri.GateBook(), {"passed": True}, ri.PRODUCTION)
    assert results["triggers"]["nonfinite_fraction"]["fired"] and results["tau"]["rem"] is None and results["tau"]["evaluable"] is False
    book = ri.GateBook()
    book.tol("G4_level1", 0.03, "f01|two|767|NEC-AMP@two", "NEC-AMP@two")
    results = ri.calibration_results(*_calibration_inputs(), book, {"passed": False}, ri.PRODUCTION, extra_failures=["G1"])
    assert results["triggers"]["gate_failure"]["gates"] == ["G4_level1", "G1", "environment"] and results["stopped"]


def test_the_statistics_constants_design_and_plan_are_pinned_literally():
    assert ri.statistics_constants() == {
        "weak_shift": -1.0, "holm_alpha": "1/40", "median_floor": 0.05, "tau_cap": 0.2, "tau_floor": 0.02, "head_share_guard": 0.05, "bootstrap": 10000,
        "bootstrap_seed": 20261001, "interval_ranks": [250, 9750],
        "worst": {"N": None, "N_nonfinite": "-inf", "S": None, "S_nonfinite": "-inf", "A_amp": None, "A_amp_nonfinite": "+inf", "A_neg": None,
                  "A_neg_nonfinite": "-inf", "head_share": None, "head_share_nonfinite": "-inf", "iso_margin": None, "iso_margin_nonfinite": "-inf",
                  "F_head_rem": None, "F_head_rem_nonfinite": "+inf", "F_head_ins": None, "F_head_ins_nonfinite": "-inf"},
        "direction_free": ["F_half_rem", "F_half_ins", "F_swap_pl", "F_swap_ref", "additivity", "P_error_rem", "P_error_ins"],
        "head_counts": {"F_head_rem": ["F_head_rem", -1, 1.0], "F_head_rem_v": ["F_head_rem", -1, 1.0], "F_head_ins": ["F_head_ins", 1, 0.0],
                        "F_head_ins_v": ["F_head_ins", 1, 0.0]}}
    assert ri.DESIGN == {"path": "docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md", "revision": 3, "commit": "a87e284",
                         "decision": "516df20"}
    assert ri.PLAN == {"path": "docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md", "revision": 3, "commit": "96f0cc8",
                       "decision": "1e48c1c"}
    assert len(ri.NUMBER_WORDS) == 42 and {"one", "two", "dozen", "several", "pair"} <= set(ri.NUMBER_WORDS)


# ---------------------------------------------------------------------------
# The gate book, the phase rules, the scientific paths, the rendering.


def test_the_gate_book_keeps_the_worst_value_and_names_failures():
    book = ri.GateBook()
    book.tol("G4_I4", 1e-4, "a", "NEC-READ@two")
    book.tol("G4_I4", 5e-4, "b", "NEC-READ@two")
    book.tol("G4_I4", 2e-4, "c", "SUF-READ@two")
    assert book.values["G4_I4"] == {"max": 5e-4, "at": "b"} and book.failures() == []
    book.tol("G4_level1", math.nan, "d", "PL@two")
    book.tol("G4_level1", 1.0, "e", "PL@two")
    book.bit("G6_state", False, "f")
    assert book.values["G4_level1"]["at"] == "d" and set(book.failures()) == {"G4_level1", "G6_state"}
    pm.canonical_json(book.to_json())


def test_the_phase_rules():
    state = {"phases": {name: {"status": "not_started"} for name in ri.STATE_PHASES}, "calibration": None, "stage1": None, "confirmation": None}
    ri.assert_phase_allowed("calibrate", None)
    for phase in ("lock", "stage1", "confirm", "report"):
        with pytest.raises(ri.PhaseError):
            ri.assert_phase_allowed(phase, state)
    state["phases"]["calibrate"]["status"] = "running"
    state["calibration"] = {"incidents": [{"commit": "a"}]}
    ri.assert_phase_allowed("calibrate", state)  # a rerun after a recorded incident (the runner checks the commit and the paths)
    state["phases"]["calibrate"]["status"] = "stopped_for_review"
    with pytest.raises(ri.PhaseError):
        ri.assert_phase_allowed("calibrate", state)
    state["phases"]["calibrate"]["status"] = "complete"
    ri.assert_phase_allowed("lock", state)
    state["phases"]["lock"]["status"] = "complete"
    ri.assert_phase_allowed("stage1", state)
    with pytest.raises(ri.PhaseError):
        ri.assert_phase_allowed("confirm", state)
    state["phases"]["stage1"] = {"status": "complete", "awaiting_commit": True}
    ri.assert_phase_allowed("confirm", state)
    state["phases"]["stage1"]["incidents"] = [{"after_ledger": True}]
    with pytest.raises(ri.PhaseError):
        ri.assert_phase_allowed("stage1", state)
    ri.assert_phase_allowed("report", state)  # a recorded incident allows the report
    state["phases"]["confirm"]["incidents"] = [{"x": 1}]
    with pytest.raises(ri.PhaseError):
        ri.assert_phase_allowed("confirm", state)


def test_the_scientific_paths():
    assert ri.scientific_changes(["src/neural_decompiler/read_intervention.py", f"{ri.EXPERIMENT_DIR}/run.py", cr.CONFIRMATION_RELATIVE_PATH]) == [
        "src/neural_decompiler/read_intervention.py", f"{ri.EXPERIMENT_DIR}/run.py", cr.CONFIRMATION_RELATIVE_PATH]
    assert ri.scientific_changes([*ri.STAGE1_PATHS, ri.CONFIRMATION_RELATIVE_PATH, ri.CALIBRATION_RELATIVE_PATH, ri.LOCK_RELATIVE_PATH,
                                  f"{ri.EXPERIMENT_DIR}/README.md", f"{ri.EXPERIMENT_DIR}/evidence/x.py", "docs/x.md", "tests/test_x.py"]) == []
    assert ri.scientific_changes([f"{ri.STAGE1_DIR}/other.json"]) == [f"{ri.STAGE1_DIR}/other.json"]


def test_the_report_of_an_incident_and_of_a_result():
    state = ri.new_results_state(digests={key: "0" for key in ri.DIGEST_KEYS}, protocol_code_commit="a" * 40, git_dirty=False, versions={}, config=ri.PRODUCTION)
    state["confirmation"] = {"incident": {"commit": "a" * 40, "message": "integrity failure after the measurements: gates ['G4_I4']"},
                             "gates": {"failures": ["G4_I4"]}}
    text = ri.render_report(state)
    assert "NOT_INTERPRETABLE" in text and "G4_I4" in text
    results = _stats([_value() for _ in range(36)])
    state["confirmation"] = {"results": results, "gates": {"failures": []}}
    state["phases"]["confirm"]["status"] = "complete"
    text = ri.render_report(state)
    assert "READ_COMPONENT_EFFECTS_AS_PREDICTED" in text and ri.SEMANTICS["scope"] in text and "NOT_INTERPRETABLE" not in text
    state["report"] = {"g6": {"bitwise_equal": False, "message": "the recomputed table differs"}}
    text = ri.render_report(state)
    assert "NOT_INTERPRETABLE" in text and "void" in text


def test_the_preregistration_renders_from_a_lock():
    lock = {"content_sha256": "c" * 64, "run_id": "r", "protocol_code_commit": "a" * 40, "design": dict(ri.DESIGN), "plan": dict(ri.PLAN),
            "module": {"path": "src/neural_decompiler/read_intervention.py", "blob": "b" * 40}, "configuration": ri.PRODUCTION.to_json(),
            "confirmation_101": {"content_sha256": "d" * 64, "manifest_sha256": "e" * 64, "counts": {"S1-REF": 36, "S2-TARGET": 1980}},
            "calibration_101": {"content_sha256": "f" * 64, "tau_rem": 0.05, "tau_ins": 0.06, "evaluable": True},
            "geometry": {"w_c_sha256": "g", "w_c_norm": 0.7659, "k_iso": 8, "iso_tag": ri.ISO_TAG, "u_off_rule": ri.U_OFF_RULE},
            "cue_order": ["two", "three", "many"], "run_order": list(ri.PRODUCTION.tags()), "statistics": dict(ri.STATISTICS),
            "constants": ri.statistics_constants(), "semantics": json.loads(pm.canonical_json(ri.SEMANTICS)), "tolerances": dict(ri.TOLERANCES),
            "outcome": {"table": [list(row) for row in ri.OUTCOME_TABLE]},
            "stage1": {"freeze": "x", "paths": {"states_data": ri.STAGE1_STATES_DATA, "states_index": ri.STAGE1_STATES_INDEX, "table": ri.STAGE1_TABLE}}}
    text = ri.render_preregistration(lock)
    assert "Holm thresholds 26 and 25 of 36" in text and "READ_COMPONENT_EFFECTS_AS_PREDICTED" in text and ri.STAGE1_TABLE in text
    assert text == ri.render_preregistration(json.loads(json.dumps(lock, sort_keys=True)))


# ---------------------------------------------------------------------------
# Tier C (the pinned model, opt-in): spent exposed prompts only; no 101 frame prompt, no outcome.


def _smoke_enabled() -> None:
    import os

    if os.environ.get("NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE") != "1":
        pytest.skip("set NEURAL_DECOMPILER_RUN_PYTHIA_SMOKE=1 to run")


def _runner():
    spec = importlib.util.spec_from_file_location("experiment_101_runner_tier_c", ROOT / f"{ri.EXPERIMENT_DIR}/run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Runner()


def _refuse_forwards(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a forward pass was attempted in a tokenizer-only test")

    for name in ("capture_prompt", "run_patched", "run_capture", "run_interventions"):
        if hasattr(pm, name):
            monkeypatch.setattr(pm, name, refuse)


def _candidate_sequences(tokenizer) -> set[tuple[int, ...]]:
    """Every 101 candidate frame's token sequence with every cue: what the guard refuses, by tokens, not by frame id."""
    sequences = set()
    for number, text in enumerate(ri.FRAME_CANDIDATES, start=1):
        for cue_id in (ri.PRODUCTION.reference[1], *(cue for _, cue in ri.PRODUCTION.cues)):
            try:
                frame = pm._build_new_frame(tokenizer, ri.TEMPLATE, text, {"sg": ri.PRODUCTION.reference[1], "pl": cue_id}, ri.frame_id_for(number))
            except ValueError:
                continue
            sequences.add(frame.prompt_ids(ri.PRODUCTION.reference[1]))
            sequences.add(frame.prompt_ids(cue_id))
    return sequences


def _guard_101_prompts_and_outcomes(monkeypatch, tokenizer) -> None:
    """No 101 frame prompt can run (identified by its token sequence) and no outcome quantity can be computed."""
    sequences = _candidate_sequences(tokenizer)
    original_patched, original_plain = pm.run_patched, pm.capture_prompt

    def patched(model, prompt, replacements, sources, *, capture_sites=()):
        assert tuple(prompt.token_ids) not in sequences, f"a 101 frame prompt was run: {prompt.key}"
        return original_patched(model, prompt, replacements, sources, capture_sites=capture_sites)

    def plain(model, prompt, sites):
        assert tuple(prompt.token_ids) not in sequences, f"a 101 frame prompt was run: {prompt.key}"
        return original_plain(model, prompt, sites)

    monkeypatch.setattr(pm, "run_patched", patched)
    monkeypatch.setattr(pm, "capture_prompt", plain)

    def refuse(*args, **kwargs):
        raise AssertionError("an outcome quantity was computed in an engineering test")

    for name in ("frame_values", "main_fractions", "primary_statistics", "secondary_statistics", "calibration_results", "prediction_table", "predict_cue"):
        monkeypatch.setattr(ri, name, refuse)
    original_contrast = ul.contrast_of

    def contrast_of(progs, state, dx3):
        assert not state.frame.frame_id.startswith("coordinated-adjective-101-"), "the frozen readout was applied to a 101 frame"
        return original_contrast(progs, state, dx3)

    monkeypatch.setattr(ul, "contrast_of", contrast_of)


@pytest.fixture(scope="module")
def pythia():
    """The pinned model, its programs, the frozen inputs and one spent exposed coordinated frame's calibration schedule
    (every key of it is in 020's ledger)."""
    _smoke_enabled()
    from neural_decompiler.models import PYTHIA_70M, load_model

    runner = _runner()
    base = runner._base()
    model = load_model(PYTHIA_70M)
    progs = ul.ModelPrograms.from_model(model, base.inputs)
    frames = runner._exposed_frames(base.inputs)
    schedules = runner._calibration_schedules(base.inputs, frames[:2])
    ledger = frozenset(base.inputs.closure["ledger"])
    assert all(cr.untagged(run.key) in ledger for schedule in schedules for run in schedule)
    tokenizer = runner.tokenizer_loader(PYTHIA_70M)
    return {"runner": runner, "base": base, "model": model, "progs": progs, "frames": frames[:2], "schedules": schedules, "tokenizer": tokenizer,
            "geometry": ri.read_geometry(base.inputs.lock_011)}


@pytest.mark.pythia_smoke
def test_the_real_tokenizer_freeze_takes_the_expected_picks_and_writes_nothing(monkeypatch):
    _smoke_enabled()
    _refuse_forwards(monkeypatch)
    from neural_decompiler.models import PYTHIA_70M

    runner = _runner()
    target = ROOT / ri.CONFIRMATION_RELATIVE_PATH
    existed = target.exists()
    base = runner._base()
    payload = ri.freeze_payload(runner.tokenizer_loader(PYTHIA_70M), exclusions=base.exclusions, config=ri.PRODUCTION)
    confirmation = ri.confirmation_from_payload(payload, ri.PRODUCTION)
    assert payload["picks"] == list(ri.EXPECTED_PICKS) and all(entry["eligible"] for entry in payload["candidates"])
    assert payload["counts"] == {"frames": 36, "runs_per_frame": 55, "S1-REF": 36, "S2-TARGET": 1980, "untagged_prompts": 144}
    assert not confirmation.untagged_keys() & base.forbidden and target.exists() == existed
    assert sorted({frame.p_c for frame in confirmation.frames}) == [4, 5, 6]


@pytest.mark.pythia_smoke
def test_a_residual_replace_lands_and_noop_reproduces_the_plain_run_on_a_spent_frame(pythia, monkeypatch):
    _guard_101_prompts_and_outcomes(monkeypatch, pythia["tokenizer"])
    schedule = pythia["schedules"][0]
    frame = schedule[0].frame
    model, progs = pythia["model"], pythia["progs"]
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    by_tag = {run.tag: run for run in schedule}
    pl = ri.run_arm(model, by_tag["PL@two"].prompt, "PL@two", capture_sites=ri.capture_sites(frame, n_layers, n_heads, plain=True))
    x_pl = pl.slice((ri.RESID_L3, frame.p_c))
    noop = ri.run_arm(model, by_tag["NOOP@two"].prompt, "NOOP@two", replacements={(ri.RESID_L3, frame.p_c): x_pl.clone()},
                      capture_sites=ri.capture_sites(frame, n_layers, n_heads, plain=False))
    assert torch.equal(noop.logits, pl.logits) and all(torch.equal(noop.slices[label], pl.slices[label]) for label in noop.slices)
    assert noop.integrity and noop.integrity[0]["outside_max_abs_change"] == 0.0


@pytest.mark.pythia_smoke
def test_two_spent_frames_pass_every_gate_and_their_maxima_are_recorded(pythia, monkeypatch, capsys):
    """Every arm on two spent exposed frames, through the production code path; G2–G6 must hold (a failure here is a
    stopping condition before the freeze). Only the gates' maxima are printed; no outcome is computed."""
    _guard_101_prompts_and_outcomes(monkeypatch, pythia["tokenizer"])
    model, progs, geometry = pythia["model"], pythia["progs"], pythia["geometry"]
    out, logits, executed = {}, {}, []
    ri.stage_runs(model, progs, pythia["schedules"], geometry, ri.PRODUCTION, executed=executed, out=out, logits=logits)
    assert Counter_like(executed) == Counter_like([run.key for schedule in pythia["schedules"] for run in schedule])
    ri.assert_measurements(out, pythia["schedules"])
    locked = pythia["base"].inputs.closure["exploration"]["locked_states"]
    states = {frame.frame_id: rd.state_from_locked(locked[frame.frame_id], frame) for frame in pythia["frames"]}
    recaptured = {frame.frame_id: ri.capture_reference(model, progs, pythia["base"].inputs.pool.reference_prompt(frame)) for frame in pythia["frames"]}
    book, _ = ri.run_gates(progs, pythia["schedules"], out, logits, states, recaptured, geometry, ri.PRODUCTION)
    report = book.to_json()
    with capsys.disabled():
        print("\n101 tier-C gate maxima on two spent exposed frames:")
        for name, entry in report["tolerance"].items():
            print(f"  {name}: {entry['max']!r} (tolerance {entry['tolerance']!r}) at {entry['at']}")
        print(f"  bitwise: {report['bitwise']}; recorded: {report['recorded']}")
    assert report["failures"] == [], report["failures"]


def Counter_like(keys):
    from collections import Counter

    return Counter(keys)


@pytest.mark.pythia_smoke
def test_a_head_replace_moves_only_the_head_output_by_the_replaced_difference(pythia, monkeypatch):
    _guard_101_prompts_and_outcomes(monkeypatch, pythia["tokenizer"])
    schedule = pythia["schedules"][0]
    frame = schedule[0].frame
    model = pythia["model"]
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    prompt = next(run.prompt for run in schedule if run.tag == "PL@two")
    sites = ri.capture_sites(frame, n_layers, n_heads, plain=True) + [("RESID_PRE.L4", frame.p_t)]
    plain = ri.run_arm(model, prompt, "PL@two", capture_sites=sites)
    captured = plain.slice((ri.HEAD_SITE_KEY, frame.p_t))
    same = ri.run_arm(model, prompt, "HEAD-NEC@two", replacements={(ri.HEAD_SITE_KEY, frame.p_t): captured.clone()}, capture_sites=sites)
    assert torch.equal(same.logits, plain.logits) and all(torch.equal(same.slices[label], plain.slices[label]) for label in same.slices)
    delta = torch.zeros_like(captured)
    delta.reshape(-1)[:] = torch.linspace(-0.5, 0.5, captured.numel())
    moved = ri.run_arm(model, prompt, "HEAD-NEC@two", replacements={(ri.HEAD_SITE_KEY, frame.p_t): (captured + delta).to(torch.float32)}, capture_sites=sites)
    change = moved.vector(("RESID_PRE.L4", frame.p_t)).double() - plain.vector(("RESID_PRE.L4", frame.p_t)).double()
    assert float((change - delta.reshape(-1).double()).abs().max()) < 1e-5
    assert torch.equal(moved.vector((ri.RESID_L3, frame.p_c)), plain.vector((ri.RESID_L3, frame.p_c)))


@pytest.mark.pythia_smoke
def test_the_reference_run_is_deterministic_across_capture_paths(pythia, monkeypatch):
    """G6's comparisons on a spent reference prompt: the stage-1 capture path against a stage-2 REF run."""
    _guard_101_prompts_and_outcomes(monkeypatch, pythia["tokenizer"])
    frame = pythia["frames"][1]
    model, progs = pythia["model"], pythia["progs"]
    prompt = pythia["base"].inputs.pool.reference_prompt(frame)
    state = ri.capture_reference(model, progs, prompt)
    run = ri.run_arm(model, prompt, "REF", capture_sites=ri.capture_sites(frame, int(model.cfg.n_layers), int(model.cfg.n_heads), plain=True))
    assert torch.equal(ri.arranged_pm_contrasts(progs, run.logits), state.c_ref.double())
    assert torch.equal(run.vector((ri.RESID_L3, frame.p_c)).double(), state.x3_all[frame.p_c].double())
    assert torch.equal(run.vector((ri.RESID_L3, frame.p_t)).double(), state.x3_all[frame.p_t].double())
    assert torch.equal(run.vector((f"RESID_POST.L{int(model.cfg.n_layers) - 1}", frame.p_t)).double(), state.h6.double())


@pytest.mark.pythia_smoke
def test_the_guards_refuse_a_101_prompt_and_every_outcome_function(pythia, monkeypatch):
    _guard_101_prompts_and_outcomes(monkeypatch, pythia["tokenizer"])
    frame = pm._build_new_frame(pythia["tokenizer"], ri.TEMPLATE, ri.FRAME_CANDIDATES[0], {"sg": 581, "pl": 767}, "renamed-to-dodge-a-frame-id-guard")
    with pytest.raises(AssertionError, match="101 frame prompt"):
        ri.run_arm(pythia["model"], pm.Prompt(frame, 581, "ref"), "REF", capture_sites=[])
    for name in ("frame_values", "primary_statistics", "prediction_table"):
        with pytest.raises(AssertionError, match="outcome"):
            getattr(ri, name)()
