from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_decompiler import plural_mechanism as pm
from neural_decompiler.candidate_screening import Split, load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / pm.MANIFEST_RELATIVE_PATH
EXTENSION_PATH = ROOT / pm.EXTENSION_RELATIVE_PATH


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST_PATH)


@pytest.fixture(scope="module")
def manifest_sha256():
    return pm.manifest_digest_from_path(MANIFEST_PATH)


class ToyTokenizer:
    """Greedy longest-match tokenizer over a fixed vocabulary of BPE-style strings."""

    def __init__(self, vocabulary: dict[str, int]) -> None:
        self.vocabulary = dict(vocabulary)
        self.reverse = {token: word for word, token in vocabulary.items()}
        self.longest = max(len(word) for word in vocabulary)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            for length in range(min(self.longest, len(text) - index), 0, -1):
                piece = text[index:index + length]
                if piece in self.vocabulary:
                    ids.append(self.vocabulary[piece])
                    index += length
                    break
            else:
                raise KeyError(text[index:])
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.reverse[token] for token in ids)


# ---------------------------------------------------------------------------
# Manifest-derived structure


def test_manifest_yields_six_frames_with_frozen_positions(manifest):
    frames = pm.derive_frames(manifest)
    assert [frame.frame_id for frame in frames] == [
        "cardinal-1", "cardinal-2", "quantifier-1", "quantifier-2",
        "coordinated-adjective-1", "coordinated-adjective-2",
    ]
    for frame in frames:
        if frame.template_id == "coordinated-adjective":
            assert (frame.p_c, frame.p_t) == (5, 6)
            assert len(frame.suffix_ids) == 1
        else:
            assert (frame.p_c, frame.p_t) == (3, 3)
            assert frame.suffix_ids == ()
    assert frames[0].text_template == "The display contains {cue}"
    assert frames[4].text_template == "Mira and Noah packed {cue} bright"
    assert len(pm.manifest_prompts(frames)) == 12


def test_manifest_prompts_reproduce_case_token_ids(manifest):
    frames = pm.derive_frames(manifest)
    prompts = {prompt.token_ids for prompt in pm.manifest_prompts(frames)}
    for case in pm.regular_plural_cases(manifest):
        assert case.x_a.prompt_token_ids in prompts
        assert case.x_b.prompt_token_ids in prompts
    assert len(prompts) == 12


def test_nouns_per_split_match_the_frozen_rule_class_counts(manifest):
    expected = {
        Split.DEVELOPMENT: {"simple-suffix": 10, "sibilant-es": 6, "consonant-y": 4},
        Split.HOLDOUT: {"simple-suffix": 10, "sibilant-es": 5, "consonant-y": 5},
        Split.FUTURE_RESERVE: {"simple-suffix": 10, "sibilant-es": 4, "consonant-y": 6},
    }
    for split, counts in expected.items():
        nouns = pm.nouns_for(manifest, split)
        assert len(nouns) == 20
        assert {rule: sum(noun.rule_class == rule for noun in nouns) for rule in pm.RULE_CLASSES} == counts
    holdout = pm.nouns_for(manifest, Split.HOLDOUT)
    assert [noun.lexical_key for noun in holdout if not noun.single_token] == ["peach"]
    assert all(noun.single_token for noun in pm.nouns_for(manifest, Split.FUTURE_RESERVE))


def test_frame_rejects_wrong_suffix_structure():
    with pytest.raises(ValueError):
        pm.Frame("cardinal", "x", (1, 2), (3,), {"sg": 4, "pl": 5}, "t {cue} s")
    with pytest.raises(ValueError):
        pm.Frame("coordinated-adjective", "x", (1, 2), (), {"sg": 4, "pl": 5}, "t {cue}")


# ---------------------------------------------------------------------------
# Extension set


def _toy_tokenizer(manifest) -> ToyTokenizer:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    vocabulary = {string.replace("Ġ", " "): int(token) for token, string in payload["token_string_by_id"].items()}
    next_id = max(vocabulary.values()) + 1
    words = ["The", " basket", " carries", " museum", " owns", " menu", " offers", " report", " cites",
             "Ana", " and", " Luis", " painted", " tiny", "Kai", " Sara", " carried", " heavy"]
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = next_id
            next_id += 1
    for word in pm.EXTENSION_CUE_WORDS:
        if " " + word not in vocabulary:
            vocabulary[" " + word] = next_id
            next_id += 1
    return ToyTokenizer(vocabulary)


def test_extension_payload_structure_and_digest(manifest, manifest_sha256):
    payload = pm.build_extension_payload(_toy_tokenizer(manifest), manifest, manifest_sha256)
    assert payload["content_sha256"] == pm.extension_content_digest(payload)
    assert [entry["frame_id"] for entry in payload["new_frames"]] == [
        "cardinal-new-1", "cardinal-new-2", "quantifier-new-1", "quantifier-new-2",
        "coordinated-adjective-new-1", "coordinated-adjective-new-2",
    ]
    assert [entry["word"] for entry in payload["cue_words"]] == list(pm.EXTENSION_CUE_WORDS[:12])
    assert payload["cue_words"][-1]["word"] == "every"
    assert len(payload["cue_word_prompts"]) == 72
    for entry in payload["new_frames"]:
        gap = 1 if entry["template_id"] == "coordinated-adjective" else 0
        assert entry["p_t"] == entry["p_c"] + gap
        assert entry["p_t"] == len(entry["prompts"]["sg"]["token_ids"]) - 1
    assert "nouns" not in payload
    extension = pm.validate_extension(payload, manifest, manifest_sha256)
    assert len(extension.new_frame_prompts()) == 12
    assert len(extension.cue_word_prompts) == 72
    assert extension.reference_cue_ids["cardinal"] == extension.original_frames[0].cue_ids["sg"]


def test_extension_validation_rejects_tampering(manifest, manifest_sha256):
    payload = pm.build_extension_payload(_toy_tokenizer(manifest), manifest, manifest_sha256)
    tampered = json.loads(json.dumps(payload))
    tampered["cue_words"][0]["word"] = "many"
    with pytest.raises(ValueError):
        pm.validate_extension(tampered, manifest, manifest_sha256)
    tampered = json.loads(json.dumps(payload))
    tampered["new_frames"][0]["prompts"]["sg"]["token_ids"][0] += 1
    tampered["content_sha256"] = pm.extension_content_digest(tampered)
    with pytest.raises(ValueError):
        pm.validate_extension(tampered, manifest, manifest_sha256)
    with pytest.raises(ValueError):
        pm.validate_extension(payload, manifest, "0" * 64)


def test_freeze_refuses_to_overwrite(tmp_path, manifest, manifest_sha256):
    target = tmp_path / "extension-v1.json"
    digest = pm.freeze_extension(target, _toy_tokenizer(manifest), manifest, manifest_sha256)
    assert json.loads(target.read_text())["content_sha256"] == digest
    with pytest.raises(FileExistsError):
        pm.freeze_extension(target, _toy_tokenizer(manifest), manifest, manifest_sha256)


@pytest.mark.skipif(not EXTENSION_PATH.exists(), reason="extension not frozen yet")
def test_committed_extension_validates_against_manifest(manifest, manifest_sha256):
    extension = pm.load_extension(EXTENSION_PATH, manifest, manifest_sha256)
    assert [word for word, _ in extension.cue_words] == list(pm.EXTENSION_CUE_WORDS[:12])
    assert all(frame.origin == "extension" for frame in extension.new_frames)
    assert len(extension.cue_word_prompts) == 72


# ---------------------------------------------------------------------------
# Results state and phase isolation


def _state():
    return pm.new_results_state(manifest_sha256="a" * 64, extension_sha256="b" * 64,
                                protocol_code_commit="c" * 40, git_dirty=False, versions={"torch": "2"})


def test_state_requires_clean_committed_tree():
    with pytest.raises(pm.PhaseError):
        pm.new_results_state(manifest_sha256="a" * 64, extension_sha256="b" * 64, protocol_code_commit="c" * 40, git_dirty=True, versions={})
    with pytest.raises(pm.PhaseError):
        pm.new_results_state(manifest_sha256="a" * 64, extension_sha256="b" * 64, protocol_code_commit="short", git_dirty=False, versions={})


def test_state_round_trip_and_tamper_detection(tmp_path):
    state = _state()
    path = tmp_path / "results.json"
    pm.write_results_state(path, state)
    loaded = pm.load_results_state(path)
    assert loaded["phases"]["discover"]["status"] == "not_started"
    raw = json.loads(path.read_text())
    raw["executed_noun_keys"] = ["future-reserve:anchor"]
    path.write_text(json.dumps(raw))
    with pytest.raises(pm.PhaseError):
        pm.load_results_state(path)


def test_phase_order_is_enforced():
    state = _state()
    pm.assert_phase_allowed("discover", state)
    for phase in ("calibrate", "revise", "lock", "confirm", "report"):
        with pytest.raises(pm.PhaseError):
            pm.assert_phase_allowed(phase, state)
    state["phases"]["discover"]["status"] = "complete"
    with pytest.raises(pm.PhaseError):
        pm.assert_phase_allowed("discover", state)
    pm.assert_phase_allowed("calibrate", state)
    with pytest.raises(pm.PhaseError):
        pm.assert_phase_allowed("lock", state)
    state["calibration"]["passes"].append({"floors_passed": False})
    with pytest.raises(pm.PhaseError):
        pm.assert_phase_allowed("calibrate", state)  # needs revise first
    pm.assert_phase_allowed("revise", state)
    state["phases"]["revise"]["status"] = "complete"
    pm.assert_phase_allowed("calibrate", state)
    state["calibration"]["passes"].append({"floors_passed": True})
    with pytest.raises(pm.PhaseError):
        pm.assert_phase_allowed("calibrate", state)  # third pass refused
    with pytest.raises(pm.PhaseError):
        pm.assert_phase_allowed("revise", state)
    pm.assert_phase_allowed("lock", state)
    state["phases"]["lock"]["status"] = "complete"
    pm.assert_phase_allowed("confirm", state)
    state["phases"]["confirm"]["status"] = "complete"
    with pytest.raises(pm.PhaseError):
        pm.assert_phase_allowed("confirm", state)
    pm.assert_phase_allowed("report", state)


def test_execution_ledger_flags_reserve_and_extension(manifest, manifest_sha256):
    extension = pm.validate_extension(pm.build_extension_payload(_toy_tokenizer(manifest), manifest, manifest_sha256), manifest, manifest_sha256)
    reserve = pm.nouns_for(manifest, Split.FUTURE_RESERVE)
    state = _state()
    frames = pm.derive_frames(manifest)
    pm.record_execution(state, pm.manifest_prompts(frames), pm.nouns_for(manifest, Split.DEVELOPMENT))
    pm.assert_not_executed(state, extension=extension, reserve=reserve)
    pm.record_execution(state, (), reserve[:1])
    with pytest.raises(pm.PhaseError):
        pm.assert_not_executed(state, extension=extension, reserve=reserve)
    state = _state()
    pm.record_execution(state, extension.cue_word_prompts[:1], ())
    with pytest.raises(pm.PhaseError):
        pm.assert_not_executed(state, extension=extension, reserve=reserve)
