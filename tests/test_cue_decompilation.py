"""Experiment 006: exposed pool, confirmation set, results state, and phase isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_decompiler import cue_decompilation as cd
from neural_decompiler import plural_mechanism as pm
from neural_decompiler.candidate_screening import load_manifest, regular_plural

from test_plural_mechanism import ToyTokenizer, _toy_tokenizer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / pm.MANIFEST_RELATIVE_PATH
EXTENSION_PATH = ROOT / pm.EXTENSION_RELATIVE_PATH
CONFIRMATION_PATH = ROOT / cd.CONFIRMATION_RELATIVE_PATH


@pytest.fixture(scope="module")
def inputs():
    manifest = load_manifest(MANIFEST_PATH)
    manifest_sha256 = pm.manifest_digest_from_path(MANIFEST_PATH)
    extension = pm.load_extension(EXTENSION_PATH, manifest, manifest_sha256)
    return manifest, manifest_sha256, extension


def toy_tokenizer_006(manifest) -> ToyTokenizer:
    """The Experiment 005 toy tokenizer plus every fresh noun, frame word, and cue token as single tokens."""
    base = _toy_tokenizer(manifest)
    vocabulary = dict(base.vocabulary)
    next_id = max(vocabulary.values()) + 1
    words: list[str] = []
    for pool in cd.NOUN_POOLS.values():
        for word in pool:
            words += [" " + word, " " + regular_plural(word)]
    for words_ in cd.TOKEN_CATEGORIES.values():
        words += [" " + word for word in words_]
    words += [" " + word for word in cd.INHERITED_TOKENS]
    for _, text in cd.FRESH_FRAMES:
        for index, piece in enumerate(text.replace("{cue}", "").split()):
            words.append(piece if index == 0 else " " + piece)
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    return ToyTokenizer(vocabulary)


def test_exposed_pool_has_twelve_frames_sixteen_tokens_sixty_nouns(inputs):
    manifest, _, extension = inputs
    pool = cd.exposed_pool(manifest, extension)
    assert len(pool.frames) == 12 and len(pool.manifest_frames) == 6 and len(pool.extension_frames) == 6
    assert len(pool.tokens) == 16 and len({token_id for _, token_id in pool.tokens}) == 16
    assert [word for word, _ in pool.tokens][:4] == ["cardinal:sg", "cardinal:pl", "quantifier:sg", "quantifier:pl"]
    assert len(pool.nouns) == 60 and len(pool.cue_word_prompts) == 72
    frame = pool.frames_of("quantifier")[0]
    assert pool.reference_prompt(frame).cue_token_id == frame.cue_ids["sg"]


def test_confirmation_payload_quotas_and_disjointness(inputs):
    manifest, manifest_sha256, extension = inputs
    payload = cd.build_confirmation_payload(toy_tokenizer_006(manifest), manifest, manifest_sha256, extension)
    assert [entry["rule_class"] for entry in payload["nouns"]].count("simple-suffix") == 10
    assert [entry["rule_class"] for entry in payload["nouns"]].count("sibilant-es") == 5
    assert [entry["rule_class"] for entry in payload["nouns"]].count("consonant-y") == 5
    exposed_keys = {noun.lexical_key for noun in cd.exposed_pool(manifest, extension).nouns}
    assert not {entry["lexical_key"] for entry in payload["nouns"]} & exposed_keys
    categories = [entry["category"] for entry in payload["tokens"]]
    assert categories.count("inherited") == 8 and all(categories.count(name) == 4 for name in cd.TOKEN_CATEGORIES)
    assert [entry["word"] for entry in payload["tokens"]][:8] == list(cd.INHERITED_TOKENS)
    assert [entry["word"] for entry in payload["tokens"] if entry["category"] == "control"] == ["big", "red", "old", "fresh"]
    assert len(payload["frames"]) == 6 and len(payload["token_prompts"]) == 24 * 6
    confirmation = cd.validate_confirmation(payload, manifest, manifest_sha256, extension)
    assert len(confirmation.all_prompts) == 12 + 144
    assert all(noun.single_token for noun in confirmation.nouns)


def test_confirmation_validation_rejects_tampering(inputs):
    manifest, manifest_sha256, extension = inputs
    payload = cd.build_confirmation_payload(toy_tokenizer_006(manifest), manifest, manifest_sha256, extension)
    tampered = json.loads(json.dumps(payload))
    tampered["tokens"][8]["word"] = "seven"
    with pytest.raises(ValueError):
        cd.validate_confirmation(tampered, manifest, manifest_sha256, extension)
    tampered = json.loads(json.dumps(payload))
    tampered["nouns"][0]["lexical_key"] = "cat"
    tampered["content_sha256"] = pm.sha256_text(pm.canonical_json({k: v for k, v in tampered.items() if k != "content_sha256"}))
    with pytest.raises(ValueError):
        cd.validate_confirmation(tampered, manifest, manifest_sha256, extension)


def test_freeze_refuses_overwrite(tmp_path, inputs):
    manifest, manifest_sha256, extension = inputs
    target = tmp_path / "confirmation-v1.json"
    cd.freeze_confirmation(target, toy_tokenizer_006(manifest), manifest, manifest_sha256, extension)
    with pytest.raises(FileExistsError):
        cd.freeze_confirmation(target, toy_tokenizer_006(manifest), manifest, manifest_sha256, extension)


@pytest.mark.skipif(not CONFIRMATION_PATH.exists(), reason="confirmation set not frozen yet")
def test_committed_confirmation_validates(inputs):
    manifest, manifest_sha256, extension = inputs
    confirmation = cd.load_confirmation(CONFIRMATION_PATH, manifest, manifest_sha256, extension)
    assert len(confirmation.tokens) == 24 and len(confirmation.nouns) == 20 and len(confirmation.frames) == 6


def _state():
    return cd.new_results_state(manifest_sha256="a" * 64, extension_sha256="b" * 64, confirmation_sha256="c" * 64, protocol_code_commit="d" * 40, git_dirty=False, versions={})


def test_phase_order_and_ledger(inputs, tmp_path):
    manifest, manifest_sha256, extension = inputs
    state = _state()
    cd.assert_phase_allowed("explore", state)
    for phase in ("calibrate", "lock", "confirm", "report"):
        with pytest.raises(cd.PhaseError):
            cd.assert_phase_allowed(phase, state)
    state["phases"]["explore"]["status"] = "complete"
    cd.assert_phase_allowed("calibrate", state)
    with pytest.raises(cd.PhaseError):
        cd.assert_phase_allowed("lock", state)
    state["phases"]["calibrate"]["status"] = "complete"
    cd.assert_phase_allowed("lock", state)
    state["phases"]["lock"]["status"] = "complete"
    cd.assert_phase_allowed("confirm", state)
    state["phases"]["confirm"]["status"] = "complete"
    with pytest.raises(cd.PhaseError):
        cd.assert_phase_allowed("confirm", state)
    path = tmp_path / "results.json"
    cd.write_results_state(path, state)
    assert cd.load_results_state(path)["phases"]["confirm"]["status"] == "complete"
    confirmation = cd.validate_confirmation(cd.build_confirmation_payload(toy_tokenizer_006(manifest), manifest, manifest_sha256, extension), manifest, manifest_sha256, extension)
    pool = cd.exposed_pool(manifest, extension)
    pm.record_execution(state, pm.manifest_prompts(pool.frames), pool.nouns)
    cd.assert_confirmation_untouched(state, confirmation)
    pm.record_execution(state, confirmation.token_prompts[:1], ())
    with pytest.raises(cd.PhaseError):
        cd.assert_confirmation_untouched(state, confirmation)
    state = _state()
    pm.record_execution(state, (), confirmation.nouns[:1])
    with pytest.raises(cd.PhaseError):
        cd.assert_confirmation_untouched(state, confirmation)
