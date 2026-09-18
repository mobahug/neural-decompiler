"""Experiment 006: low-rank, weight-only decompilation of the layer-0 MLP cue encoding.

The circuit is inherited from Experiment 005 and never searched. Every Experiment
005 prompt, noun, and cue token is exploratory input; a fresh confirmation set
(nouns, frames, cue tokens) is frozen by tokenizer rules before any model run and
executed exactly once after the committed lock. Constants are copied from the
approved design (revision 4).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import plural_mechanism as pm
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest, Split, regular_plural
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 4).

EXPERIMENT_DIR = "experiments/006-low-rank-cue-decompilation"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
EXPERIMENT_005_LOCK_PATH = "experiments/005-regular-plural-mechanism/preregistration-lock.json"
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260919
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
PHASES = ("explore", "calibrate", "lock", "confirm", "report")

FIXED_CIRCUIT = pm.MechanismSet("L00.MLP", ("L00.MLP",), ("L03.H04",), ("L04.MLP", "L05.MLP"))
RANKS = (1, 2, 3, 4)
RANK_IMPROVEMENT_FACTOR = 0.8  # r > 1 is eligible only if error_r <= 0.8 * error_1 (the 20% rule)
QUALITY_SPEARMAN_FLOOR = 0.70
QUALITY_NORMALIZED_RMSE_CEILING = 0.50
TAU_MIN_NATS = 0.5
TAU_RMSE_MULTIPLIER = 3.0

NOUN_QUOTAS = {"simple-suffix": 10, "sibilant-es": 5, "consonant-y": 5}
NOUN_POOLS = {
    "simple-suffix": ("stone", "field", "road", "door", "window", "bridge", "tower", "planet", "letter", "wheel", "candle", "branch"),
    "sibilant-es": ("patch", "coach", "church", "tax", "boss", "lens", "arch", "flash"),
    "consonant-y": ("body", "copy", "duty", "spy", "study", "memory", "theory", "entry", "colony", "galaxy"),
}
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The crate holds {cue}"),
    ("cardinal", "The gallery shows {cue}"),
    ("quantifier", "The index lists {cue}"),
    ("quantifier", "The archive keeps {cue}"),
    ("coordinated-adjective", "Ravi and Elena sorted {cue} plain"),
    ("coordinated-adjective", "Nora and Felix stacked {cue} heavy"),
)
INHERITED_TOKENS: tuple[str, ...] = pm.EXTENSION_CUE_WORDS[pm.CUE_WORD_COUNT:]  # the eight never executed in Experiment 005
TOKEN_QUOTA = 4
TOKEN_CATEGORIES: dict[str, tuple[str, ...]] = {
    "numeral": ("six", "seven", "eight", "nine", "twenty", "fifty", "thousand"),
    "quantity": ("dozen", "countless", "various", "fewer", "more", "most", "enough", "plenty"),
    "determiner": ("this", "that", "these", "those", "either", "neither", "an", "my", "our", "their"),
    "control": ("big", "red", "old", "fresh", "only", "very"),
}

# Floors (design revision 4).
CUE_EFFECT_MANIFEST = 114 / 120
CUE_EFFECT_FRESH = 108 / 120
P3_SIGN_RATE = 114 / 120
P3_CORRELATION = 0.90
P3_F_OVERALL, P3_F_TEMPLATE = 0.50, 0.40
Y1_SPEARMAN, Y2_SPEARMAN = 0.80, 0.70
Y2_TAU_FACTOR = 1.5
Y1_CONFIDENT_NATS = 1.0
Y1_SIGN_FRAMES = 5  # of 6
Y3_POSITIVE_RATE = 108 / 120
Y_BAND_TOKENS, Y_BAND_FRAMES = 18, 5


class PhaseError(pm.PhaseError):
    pass


# ---------------------------------------------------------------------------
# Exposed pool.


@dataclass(frozen=True)
class ExposedPool:
    frames: tuple[pm.Frame, ...]  # six manifest + six extension frames
    tokens: tuple[tuple[str, int], ...]  # sixteen exposed cue tokens (word, id)
    nouns: tuple[pm.Noun, ...]  # sixty
    reference_ids: Mapping[str, int]  # template -> singular reference cue id
    cue_word_prompts: tuple[pm.Prompt, ...]  # the 72 Experiment 005 cue-word prompts

    @property
    def manifest_frames(self) -> tuple[pm.Frame, ...]:
        return tuple(frame for frame in self.frames if frame.origin == "manifest")

    @property
    def extension_frames(self) -> tuple[pm.Frame, ...]:
        return tuple(frame for frame in self.frames if frame.origin == "extension")

    def frames_of(self, template_id: str) -> tuple[pm.Frame, ...]:
        return tuple(frame for frame in self.frames if frame.template_id == template_id)

    def reference_prompt(self, frame: pm.Frame) -> pm.Prompt:
        return pm.Prompt(frame, self.reference_ids[frame.template_id], "ref")


def exposed_pool(manifest: ScreeningManifest, extension: pm.Extension, tokenizer_strings: Mapping[str, str] | None = None) -> ExposedPool:
    frames = pm.derive_frames(manifest)
    tokens: list[tuple[str, int]] = []
    seen: set[int] = set()
    for template in pm.TEMPLATE_ORDER:
        frame = next(frame for frame in frames if frame.template_id == template)
        for label in ("sg", "pl"):
            token_id = frame.cue_ids[label]
            if token_id not in seen:
                seen.add(token_id)
                tokens.append((f"{template}:{label}", token_id))
    for word, token_id in extension.cue_words:
        if token_id in seen:
            raise ValueError(f"extension word {word} duplicates an original cue token")
        seen.add(token_id)
        tokens.append((word, token_id))
    if len(tokens) != 16:
        raise ValueError(f"expected sixteen exposed cue tokens, found {len(tokens)}")
    nouns = tuple(noun for split in pm.Split for noun in pm.nouns_for(manifest, split))
    return ExposedPool(frames + extension.new_frames, tuple(tokens), nouns, dict(extension.reference_cue_ids), extension.cue_word_prompts)


# ---------------------------------------------------------------------------
# Confirmation set: tokenizer-only construction, digest, loading, validation.


def _rule_class_of(word: str) -> str:
    if word.endswith("y") and word[-2] not in "aeiou":
        return "consonant-y"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return "sibilant-es"
    return "simple-suffix"


def fresh_nouns(tokenizer: Any, excluded_keys: set[str], excluded_ids: set[int]) -> list[dict[str, Any]]:
    """First eligible nouns per category quota; both forms single tokens; disjoint from every exposed noun."""
    chosen: list[dict[str, Any]] = []
    for rule_class, quota in NOUN_QUOTAS.items():
        count = 0
        for word in NOUN_POOLS[rule_class]:
            if _rule_class_of(word) != rule_class:
                raise ValueError(f"{word} is listed under the wrong rule class")
            plural = regular_plural(word)
            sg, pl = pm._encode(tokenizer, " " + word), pm._encode(tokenizer, " " + plural)
            eligible = len(sg) == 1 and len(pl) == 1 and word not in excluded_keys and sg[0] not in excluded_ids and pl[0] not in excluded_ids
            if not eligible:
                continue
            chosen.append({"lexical_key": word, "rule_class": rule_class, "sg_id": sg[0], "pl_id": pl[0], "sg_text": " " + word, "pl_text": " " + plural})
            count += 1
            if count == quota:
                break
        if count != quota:
            raise ValueError(f"{rule_class}: only {count} eligible fresh nouns, quota {quota}")
    return chosen


def fresh_tokens(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
    """The eight inherited tokens plus the first four eligible tokens of each category."""
    chosen: list[dict[str, Any]] = []
    seen: set[int] = set(excluded_ids)
    for word in INHERITED_TOKENS:
        ids = pm._encode(tokenizer, " " + word)
        if len(ids) != 1 or ids[0] in seen:
            raise ValueError(f"inherited token {word} is not eligible")
        seen.add(ids[0])
        chosen.append({"word": word, "token_id": ids[0], "category": "inherited"})
    for category, words in TOKEN_CATEGORIES.items():
        count = 0
        for word in words:
            ids = pm._encode(tokenizer, " " + word)
            if len(ids) != 1 or ids[0] in seen:
                continue
            seen.add(ids[0])
            chosen.append({"word": word, "token_id": ids[0], "category": category})
            count += 1
            if count == TOKEN_QUOTA:
                break
        if count != TOKEN_QUOTA:
            raise ValueError(f"{category}: only {count} eligible tokens, quota {TOKEN_QUOTA}")
    return chosen


def build_confirmation_payload(tokenizer: Any, manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension) -> dict[str, Any]:
    pool = exposed_pool(manifest, extension)
    excluded_keys = {noun.lexical_key for noun in pool.nouns}
    excluded_ids = {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    nouns = fresh_nouns(tokenizer, excluded_keys, excluded_ids)
    tokens = fresh_tokens(tokenizer, {token_id for _, token_id in pool.tokens})
    frames: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    cue_ids_by_template = {frame.template_id: dict(frame.cue_ids) for frame in pool.manifest_frames}
    for template_id, text_template in FRESH_FRAMES:
        counters[template_id] = counters.get(template_id, 0) + 1
        frame = pm._build_new_frame(tokenizer, template_id, text_template, cue_ids_by_template[template_id], f"{template_id}-fresh-{counters[template_id]}")
        frames.append({"frame_id": frame.frame_id, "template_id": template_id, "text_template": text_template, "prefix_ids": list(frame.prefix_ids),
                       "suffix_ids": list(frame.suffix_ids), "cue_ids": dict(frame.cue_ids), "p_c": frame.p_c, "p_t": frame.p_t,
                       "prompts": {label: {"text": pm._decode(tokenizer, frame.prompt_ids(frame.cue_ids[label])), "token_ids": list(frame.prompt_ids(frame.cue_ids[label]))} for label in ("sg", "pl")}})
    token_prompts = []
    for entry in frames:
        frame = pm.Frame(entry["template_id"], entry["frame_id"], tuple(entry["prefix_ids"]), tuple(entry["suffix_ids"]), entry["cue_ids"], entry["text_template"], origin="extension")
        for token in tokens:
            ids = frame.prompt_ids(token["token_id"])
            text = pm._decode(tokenizer, ids)
            if pm._encode(tokenizer, text) != ids:
                raise ValueError(f"{frame.frame_id}/{token['word']}: text does not round-trip to the constructed token IDs")
            token_prompts.append({"frame_id": frame.frame_id, "word": token["word"], "token_id": token["token_id"], "text": text, "token_ids": list(ids), "p_c": frame.p_c, "p_t": frame.p_t})
    payload = {
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256, "extension_sha256": extension.content_sha256,
        "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
        "reference_cue_ids": dict(pool.reference_ids),
        "exposed_token_ids": sorted(token_id for _, token_id in pool.tokens),
        "nouns": nouns, "frames": frames, "tokens": tokens, "token_prompts": token_prompts,
        "construction": "tokenizer-only; quotas 10/5/5 nouns, 8 inherited + 4 per category tokens; no model output",
    }
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


@dataclass(frozen=True)
class Confirmation:
    manifest_sha256: str
    extension_sha256: str
    reference_ids: Mapping[str, int]
    nouns: tuple[pm.Noun, ...]
    frames: tuple[pm.Frame, ...]
    tokens: tuple[dict[str, Any], ...]
    token_prompts: tuple[pm.Prompt, ...]
    content_sha256: str

    def frame_prompts(self) -> tuple[pm.Prompt, ...]:
        return pm.manifest_prompts(self.frames)

    def reference_prompt(self, frame: pm.Frame) -> pm.Prompt:
        return pm.Prompt(frame, self.reference_ids[frame.template_id], "ref")

    @property
    def all_prompts(self) -> tuple[pm.Prompt, ...]:
        return self.frame_prompts() + self.token_prompts


def validate_confirmation(payload: Mapping[str, Any], manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension) -> Confirmation:
    pm._require_exact_keys(payload, {"schema_version", "manifest_sha256", "extension_sha256", "model", "reference_cue_ids", "exposed_token_ids", "nouns", "frames",
                                     "tokens", "token_prompts", "construction", "content_sha256"}, "confirmation")
    if payload["schema_version"] != CONFIRMATION_SCHEMA_VERSION:
        raise ValueError("confirmation schema version is not frozen")
    if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("confirmation content_sha256 does not match canonical payload")
    if payload["manifest_sha256"] != manifest_sha256 or payload["extension_sha256"] != extension.content_sha256:
        raise ValueError("confirmation was built against different frozen inputs")
    pool = exposed_pool(manifest, extension)
    if dict(payload["reference_cue_ids"]) != dict(pool.reference_ids):
        raise ValueError("reference cue IDs disagree with the exposed pool")
    if list(payload["exposed_token_ids"]) != sorted(token_id for _, token_id in pool.tokens):
        raise ValueError("exposed token IDs disagree with the exposed pool")
    exposed_keys = {noun.lexical_key for noun in pool.nouns}
    exposed_ids = {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    nouns = []
    counts = {rule: 0 for rule in NOUN_QUOTAS}
    for entry in payload["nouns"]:
        pm._require_exact_keys(entry, {"lexical_key", "rule_class", "sg_id", "pl_id", "sg_text", "pl_text"}, "noun")
        if entry["lexical_key"] in exposed_keys or entry["sg_id"] in exposed_ids or entry["pl_id"] in exposed_ids:
            raise ValueError(f"fresh noun {entry['lexical_key']} overlaps an exposed noun")
        if entry["lexical_key"] not in NOUN_POOLS[entry["rule_class"]] or _rule_class_of(entry["lexical_key"]) != entry["rule_class"]:
            raise ValueError(f"fresh noun {entry['lexical_key']} is not in its frozen pool")
        counts[entry["rule_class"]] += 1
        nouns.append(pm.Noun(entry["lexical_key"], Split.FUTURE_RESERVE, entry["rule_class"], (int(entry["sg_id"]),), (int(entry["pl_id"]),)))
    if counts != NOUN_QUOTAS:
        raise ValueError(f"fresh noun quotas violated: {counts}")
    frames = tuple(pm._parse_frame(entry, "extension") for entry in payload["frames"])
    expected = []
    counters: dict[str, int] = {}
    for template_id, text_template in FRESH_FRAMES:
        counters[template_id] = counters.get(template_id, 0) + 1
        expected.append((f"{template_id}-fresh-{counters[template_id]}", template_id, text_template))
    if [(frame.frame_id, frame.template_id, frame.text_template) for frame in frames] != expected:
        raise ValueError("fresh frames are not the frozen literal frames")
    cue_ids_by_template = {frame.template_id: dict(frame.cue_ids) for frame in pool.manifest_frames}
    for frame in frames:
        if dict(frame.cue_ids) != cue_ids_by_template[frame.template_id]:
            raise ValueError(f"{frame.frame_id}: fresh frames must use the template's original cue tokens")
    tokens = []
    exposed_token_ids = {token_id for _, token_id in pool.tokens}
    for entry in payload["tokens"]:
        pm._require_exact_keys(entry, {"word", "token_id", "category"}, "token")
        if entry["token_id"] in exposed_token_ids:
            raise ValueError(f"fresh token {entry['word']} is an exposed cue token")
        tokens.append(dict(entry))
    inherited = [entry["word"] for entry in tokens if entry["category"] == "inherited"]
    if inherited != list(INHERITED_TOKENS):
        raise ValueError("inherited tokens must be exactly the eight never-executed Experiment 005 words")
    for category, words in TOKEN_CATEGORIES.items():
        chosen = [entry["word"] for entry in tokens if entry["category"] == category]
        if len(chosen) != TOKEN_QUOTA or any(word not in words for word in chosen) or [words.index(word) for word in chosen] != sorted(words.index(word) for word in chosen):
            raise ValueError(f"{category}: token quota or order violated")
    if len(tokens) != len(INHERITED_TOKENS) + TOKEN_QUOTA * len(TOKEN_CATEGORIES) or len({entry["token_id"] for entry in tokens}) != len(tokens):
        raise ValueError("fresh tokens must be twenty-four distinct tokens")
    frames_by_id = {frame.frame_id: frame for frame in frames}
    token_ids = {entry["word"]: entry["token_id"] for entry in tokens}
    prompts = []
    for entry in payload["token_prompts"]:
        pm._require_exact_keys(entry, {"frame_id", "word", "token_id", "text", "token_ids", "p_c", "p_t"}, "token_prompt")
        frame = frames_by_id[entry["frame_id"]]
        if token_ids.get(entry["word"]) != entry["token_id"] or tuple(entry["token_ids"]) != frame.prompt_ids(entry["token_id"]) or (entry["p_c"], entry["p_t"]) != (frame.p_c, frame.p_t):
            raise ValueError(f"{entry['frame_id']}/{entry['word']}: stored prompt disagrees with the frame")
        prompts.append(pm.Prompt(frame, int(entry["token_id"]), entry["word"]))
    if [(prompt.frame.frame_id, prompt.cue_label) for prompt in prompts] != [(frame.frame_id, entry["word"]) for frame in frames for entry in tokens]:
        raise ValueError("token prompts must cover every fresh frame and token in order")
    return Confirmation(payload["manifest_sha256"], payload["extension_sha256"], dict(payload["reference_cue_ids"]), tuple(nouns), frames, tuple(tokens), tuple(prompts), payload["content_sha256"])


def freeze_confirmation(path: Path, tokenizer: Any, manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; the confirmation set is frozen and cannot be rebuilt")
    payload = build_confirmation_payload(tokenizer, manifest, manifest_sha256, extension)
    validate_json_safe(payload, path="confirmation")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    return payload["content_sha256"]


def load_confirmation(path: Path, manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension) -> Confirmation:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, manifest, manifest_sha256, extension)


# ---------------------------------------------------------------------------
# Results state and phase isolation.

_STATE_KEYS = {"schema_version", "run_id", "created_at", "manifest_sha256", "extension_sha256", "confirmation_sha256", "protocol_code_commit", "git_dirty",
               "model", "versions", "phases", "executed_prompt_keys", "executed_noun_keys", "exploration", "calibration", "lock", "confirmation", "invalidated_runs", "state_sha256"}


def new_results_state(*, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str, protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "run_id": pm.sha256_text(manifest_sha256 + extension_sha256 + confirmation_sha256 + protocol_code_commit + pm.utc_now())[:16],
        "created_at": pm.utc_now(), "manifest_sha256": manifest_sha256, "extension_sha256": extension_sha256, "confirmation_sha256": confirmation_sha256,
        "protocol_code_commit": protocol_code_commit, "git_dirty": False,
        "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "versions": dict(versions),
        "phases": {phase: {"status": "not_started"} for phase in PHASES},
        "executed_prompt_keys": [], "executed_noun_keys": [],
        "exploration": {}, "calibration": None, "lock": None, "confirmation": None, "invalidated_runs": [],
    }


def write_results_state(path: Path, state: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    validate_json_safe(payload, path="results_state")
    payload["state_sha256"] = pm.state_digest(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload["state_sha256"]


def load_results_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhaseError(f"could not read results state: {path}") from error
    if not isinstance(state, Mapping) or state.get("schema_version") != RESULTS_SCHEMA_VERSION or set(state) != _STATE_KEYS:
        raise PhaseError("results state schema is not recognized")
    if state["state_sha256"] != pm.state_digest(state):
        raise PhaseError("results state digest mismatch: the artifact was modified outside the runner")
    if set(state["phases"]) != set(PHASES) or any(entry.get("status") not in pm.PHASE_STATUSES for entry in state["phases"].values()):
        raise PhaseError("results state phase records are invalid")
    return dict(state)


def assert_provenance_identical(state: Mapping[str, Any], *, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str, protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> None:
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if (state["manifest_sha256"], state["extension_sha256"], state["confirmation_sha256"]) != (manifest_sha256, extension_sha256, confirmation_sha256):
        raise PhaseError("frozen input digests differ from the recorded run")
    if dict(state["versions"]) != dict(versions):
        raise PhaseError("dependency versions differ from the recorded run")


def assert_phase_allowed(phase: str, state: Mapping[str, Any]) -> None:
    if phase not in PHASES:
        raise PhaseError(f"unknown phase {phase}")
    status = {name: entry["status"] for name, entry in state["phases"].items()}
    if phase == "explore":
        if status["explore"] == "running" and not state["exploration"].get("selection"):
            return  # crash recovery before any selection was concluded
        if status["explore"] != "not_started":
            raise PhaseError("explore already ran; exploration is never re-run in one protocol version")
    elif phase == "calibrate":
        if status["explore"] != "complete":
            raise PhaseError("calibrate requires the completed explore phase")
        if status["calibrate"] == "complete":
            raise PhaseError("calibrate already ran")
    elif phase == "lock":
        if status["calibrate"] != "complete":
            raise PhaseError("lock requires the completed calibrate phase")
        if status["lock"] == "complete":
            raise PhaseError("lock already written; a new candidate lock requires a new protocol version")
    elif phase == "confirm":
        if status["lock"] != "complete":
            raise PhaseError("confirm requires the lock phase")
        if status["confirm"] != "not_started":
            raise PhaseError("confirm already ran; a second scientific attempt requires a new protocol version")
    elif phase == "report":
        if status["explore"] != "complete":
            raise PhaseError("report requires at least the completed explore phase")


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation) -> None:
    executed_prompts = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed_prompts & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")
    if set(state["executed_noun_keys"]) & {noun.key for noun in confirmation.nouns}:
        raise PhaseError("fresh nouns were scored before the confirmation boundary")


def load_inputs(root: Path) -> tuple[ScreeningManifest, str, pm.Extension, Confirmation]:
    manifest, manifest_sha256, extension = pm.load_inputs(root)
    confirmation = load_confirmation(root / CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    return manifest, manifest_sha256, extension, confirmation
