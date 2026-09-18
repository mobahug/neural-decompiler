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


# ---------------------------------------------------------------------------
# Prompt-level execution primitives (TinyBridge)

import torch

from instrumentation_fakes import TinyBridge
from neural_decompiler.interventions import ReplacementSource


def _tiny_frame() -> pm.Frame:
    return pm.Frame("cardinal", "tiny-1", (1, 2, 3), (), {"sg": 4, "pl": 7}, "a b c {cue}")


def _tiny_nouns() -> tuple[pm.Noun, ...]:
    return (pm.Noun("n0", Split.DEVELOPMENT, "simple-suffix", (0,), (1,)),
            pm.Noun("n1", Split.DEVELOPMENT, "sibilant-es", (2,), (3,)),
            pm.Noun("multi", Split.HOLDOUT, "consonant-y", (0, 1), (2, 3)))


def test_component_keys_round_trip():
    for key in ("L00.H01", "L01.MLP", "EMBED", "RESID_PRE.L1", "RESID_POST.L0", "ATTN_PATTERN.L1"):
        assert pm.component_key(pm.component_ref(key)) == key
    assert pm.is_head_key("L00.H01") and not pm.is_head_key("L00.MLP")
    assert pm.key_layer("L03.H04") == 3 and pm.key_layer("RESID_PRE.L2") == 2
    with pytest.raises(ValueError):
        pm.component_ref("L00.NEURON")
    assert len(pm.universe_keys(TinyBridge())) == 6  # 2 layers × (2 heads + MLP)


def test_capture_prompt_returns_final_logits_and_requested_sites():
    model = TinyBridge()
    frame = _tiny_frame()
    prompt = pm.Prompt(frame, frame.cue_ids["sg"], "sg")
    run = pm.capture_prompt(model, prompt, [("L00.MLP", 3), ("L01.H01", 3), ("RESID_PRE.L1", 0), ("ATTN_PATTERN.L0", 3)])
    direct = model(torch.tensor([prompt.token_ids]))
    assert torch.equal(run.logits, direct[0, -1])
    assert run.slice(("L00.MLP", 3)).shape == (1, 1, 3)
    assert run.slice(("L01.H01", 3)).shape == (1, 1, 1, 3)
    assert run.vector(("L01.H01", 3)).shape == (3,)
    assert run.vector(("ATTN_PATTERN.L0", 3)).shape == (2, 4)
    with pytest.raises(ValueError):
        pm.capture_prompt(model, prompt, [("L00.MLP", 4)])


def test_run_patched_is_exact_and_captures_post_patch_values():
    model = TinyBridge()
    frame = _tiny_frame()
    sg, pl = pm.Prompt(frame, 4, "sg"), pm.Prompt(frame, 7, "pl")
    site = ("L00.MLP", 3)
    run_sg = pm.capture_prompt(model, sg, [site, ("RESID_POST.L1", 3)])
    run_pl = pm.capture_prompt(model, pl, [site])
    same = pm.run_patched(model, sg, {site: run_sg.slice(site)}, {site: ReplacementSource.REFERENCE}, capture_sites=[("RESID_POST.L1", 3)])
    assert torch.equal(same.logits, run_sg.logits)
    assert torch.equal(same.slice(("RESID_POST.L1", 3)), run_sg.slice(("RESID_POST.L1", 3)))
    assert same.integrity[0]["outside_max_abs_change"] == 0.0
    patched = pm.run_patched(model, sg, {site: run_pl.slice(site)}, {site: ReplacementSource.REFERENCE}, capture_sites=[site, ("RESID_POST.L1", 3)])
    assert not torch.equal(patched.logits, run_sg.logits)
    assert torch.equal(patched.slice(site), run_pl.slice(site))
    assert not torch.equal(patched.slice(("RESID_POST.L1", 3)), run_sg.slice(("RESID_POST.L1", 3)))
    with pytest.raises(ValueError):
        pm.run_patched(model, sg, {site: run_pl.slice(site)}, {})


def test_contrasts_and_pair_centered():
    logits = torch.tensor([1.0, 0.0, 2.0, 2.0, -1.0])
    values = pm.contrasts(logits, _tiny_nouns())
    assert values == {"n0": pytest.approx(1.0), "n1": pytest.approx(0.0)}
    a, b = torch.tensor([[1.0, 3.0]]), torch.tensor([[3.0, 5.0]])
    assert torch.equal(pm.pair_centered(a, b), torch.tensor([[2.0, 4.0]]))
    with pytest.raises(ValueError):
        pm.pair_centered(a, torch.tensor([1.0, 2.0, 3.0]))


def test_prefix_identity_assertion():
    model = TinyBridge()
    frame = _tiny_frame()
    sg, pl = pm.Prompt(frame, 4, "sg"), pm.Prompt(frame, 7, "pl")
    sites = [("RESID_POST.L1", 0), ("L00.MLP", 2)]
    run_sg, run_pl = pm.capture_prompt(model, sg, sites), pm.capture_prompt(model, pl, sites)
    pm.assert_prefix_identical(run_sg, run_pl, sites)
    cue_sites = [("L00.MLP", 3)]
    with pytest.raises(pm.IncidentError):
        pm.assert_prefix_identical(pm.capture_prompt(model, sg, cue_sites), pm.capture_prompt(model, pl, cue_sites), cue_sites)


def test_case_rows_and_stratified_recovery():
    frame = _tiny_frame()
    nouns = _tiny_nouns()
    c_a, c_b = {"n0": 2.0, "n1": 1.0}, {"n0": -2.0, "n1": -1.0}
    rows = pm.case_rows(frame, nouns, c_a, c_b, {"n0": -1.0, "n1": 0.0}, {"n0": 1.0, "n1": 0.0})
    assert [row.lexical_key for row in rows] == ["n0", "n1"]  # multi-token noun excluded
    assert rows[0].d_full == pytest.approx(4.0) and rows[0].d_patch == pytest.approx(3.0)
    assert rows[1].d_full == pytest.approx(2.0) and rows[1].d_patch == pytest.approx(1.0)
    summary = pm.stratified_recovery(rows)
    assert summary["overall"]["recovery"] == pytest.approx(4.0 / 6.0)
    assert summary["templates"]["cardinal"]["cases"] == 2
    assert summary["rule_classes"]["sibilant-es"]["recovery"] == pytest.approx(0.5)
    tiny = pm.stratified_recovery([pm.CaseRow("f", "cardinal", "n", "simple-suffix", 0.2, 0.1)])
    assert tiny["overall"]["denominator_valid"] is False and tiny["overall"]["recovery"] is None


def test_random_sets_are_deterministic_and_ordered():
    universe = tuple(f"C{i}" for i in range(10))
    first = pm.deterministic_random_sets(universe, 3, count=5)
    second = pm.deterministic_random_sets(universe, 3, count=5)
    assert first == second and len(first) == 5
    assert all(len(set(s)) == 3 and list(s) == sorted(s, key=universe.index) for s in first)
    assert pm.deterministic_random_sets(universe, 3, count=5, seed=1) != first
