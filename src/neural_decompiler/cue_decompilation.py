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


# ---------------------------------------------------------------------------
# E-patch residual response on exposed prompts, and the low-rank fit.

import torch  # noqa: E402

from .interventions import ReplacementSource  # noqa: E402


def e_slice(weights: pm.Weights, token_id: int) -> torch.Tensor:
    """The weight-only E(w) shaped as the L00.MLP replacement slice [1, 1, d_model]."""
    return pm.lexicon_vector(weights, token_id).reshape(1, 1, -1).to(torch.float32)


@dataclass(frozen=True)
class EPatchResponse:
    token: str
    token_id: int
    frame_id: str
    template_id: str
    delta_residual: torch.Tensor  # [d_model] final pre-LayerNorm residual change
    shifts: Mapping[str, float]  # noun -> E-patch contrast shift
    circuit_share: float  # fraction of |delta_residual|^2 explained by the named T ∪ R (and E at p_t) outputs' change
    integrity: tuple[dict[str, Any], ...]


def measure_epatch_responses(model: Any, weights: pm.Weights, cache: pm.PromptCache, frames: Sequence[pm.Frame], tokens: Sequence[tuple[str, int]],
                             reference_ids: Mapping[str, int], circuit: pm.MechanismSet = FIXED_CIRCUIT) -> dict[tuple[str, str], EPatchResponse]:
    """One intervention forward per (token, frame): replace E in the reference prompt by E(w)."""
    n_layers = int(model.cfg.n_layers)
    final_key = f"RESID_POST.L{n_layers - 1}"
    responses: dict[tuple[str, str], EPatchResponse] = {}
    for frame in frames:
        reference = pm.Prompt(frame, reference_ids[frame.template_id], "ref")
        clean = cache.run(reference)
        clean_final = clean.vector((final_key, reference.p_t))
        circuit_sites = circuit.p_t_sites_for(frame)
        clean_circuit = pm.summed_vector(clean, circuit_sites)
        c_ref = cache.c(reference)
        captured_reference = clean.vector(("L00.MLP", reference.p_c))
        lexicon_gap = float((pm.lexicon_vector(weights, reference.cue_token_id) - captured_reference).abs().max())
        if lexicon_gap > 1e-5:
            raise pm.IncidentError(f"{frame.frame_id}: weight-only E(ref) differs from the captured L00.MLP output by {lexicon_gap:.2e}")
        for token, token_id in tokens:
            if token_id == reference.cue_token_id:
                # Replacing E(ref) by E(ref) is the identity: the response is zero by definition, no forward is run.
                responses[(token, frame.frame_id)] = EPatchResponse(token, token_id, frame.frame_id, frame.template_id, torch.zeros_like(clean_final),
                                                                    {noun: 0.0 for noun in c_ref}, 1.0, ())
                continue
            site = ("L00.MLP", reference.p_c)
            patched = pm.run_patched(model, reference, {site: e_slice(weights, token_id)}, {site: ReplacementSource.RESAMPLE},
                                     capture_sites=[(final_key, reference.p_t)] + list(circuit_sites))
            delta = patched.vector((final_key, reference.p_t)) - clean_final
            circuit_delta = pm.summed_vector(patched, circuit_sites) - clean_circuit
            norm = float(delta.double().norm() ** 2)
            share = float((delta.double() @ circuit_delta.double()) / norm) if norm > 0 else 0.0
            c_patched = pm.contrasts(patched.logits, cache.nouns)
            responses[(token, frame.frame_id)] = EPatchResponse(token, token_id, frame.frame_id, frame.template_id, delta.clone(),
                                                                {noun: c_patched[noun] - c_ref[noun] for noun in c_patched}, share, patched.integrity)
    return responses


@dataclass(frozen=True)
class LowRankFit:
    rank: int
    mu: torch.Tensor  # [d_model] float64
    basis: torch.Tensor  # U_r [d_model, r] float64
    maps: Mapping[str, torch.Tensor]  # template -> V_T [d_model, r] float64
    reference_ids: Mapping[str, int]
    rho_frame: Mapping[str, torch.Tensor]  # frame_id -> final reference residual (float64)
    rho_template: Mapping[str, torch.Tensor]
    fit_tokens: tuple[str, ...]
    singular_values: tuple[float, ...]

    def z(self, e_vector: torch.Tensor) -> torch.Tensor:
        return self.basis.T @ (e_vector.double() - self.mu)

    def dz(self, template: str, e_vector: torch.Tensor, e_reference: torch.Tensor) -> torch.Tensor:
        return self.basis.T @ (e_vector.double() - e_reference.double())


def pca_basis(e_vectors: Sequence[torch.Tensor], rank: int) -> tuple[torch.Tensor, torch.Tensor, tuple[float, ...]]:
    """Centered PCA: rows of X are centered E vectors; U_r = first r right singular vectors (columns, in R^d)."""
    matrix = torch.stack([vector.double().reshape(-1) for vector in e_vectors])
    mu = matrix.mean(dim=0)
    centered = matrix - mu
    _, singular, vt = torch.linalg.svd(centered, full_matrices=False)
    if rank > vt.shape[0]:
        raise ValueError(f"rank {rank} exceeds the number of centered vectors")
    return mu, vt[:rank].T.contiguous(), tuple(float(value) for value in singular[:rank])


def fit_low_rank(*, e_vectors: Mapping[str, torch.Tensor], reference_ids: Mapping[str, int], reference_vectors: Mapping[str, torch.Tensor],
                 responses: Mapping[tuple[str, str], EPatchResponse], frames: Sequence[pm.Frame], cache: pm.PromptCache, fit_tokens: Sequence[str], rank: int) -> LowRankFit:
    """μ_E and U_r from the fit tokens' E vectors; V_T by no-intercept least squares on Δr_Epatch ≈ V_T Δz_T."""
    mu, basis, singular = pca_basis([e_vectors[token] for token in fit_tokens], rank)
    n_layers = int(cache.model.cfg.n_layers)
    final_key = f"RESID_POST.L{n_layers - 1}"
    maps: dict[str, torch.Tensor] = {}
    rho_frame: dict[str, torch.Tensor] = {}
    rho_template: dict[str, torch.Tensor] = {}
    for template in pm.TEMPLATE_ORDER:
        template_frames = [frame for frame in frames if frame.template_id == template]
        reference_vector = reference_vectors[template].double()
        features, targets = [], []
        for frame in template_frames:
            reference = pm.Prompt(frame, reference_ids[template], "ref")
            rho_frame[frame.frame_id] = cache.run(reference).vector((final_key, reference.p_t)).double()
            for token in fit_tokens:
                response = responses[(token, frame.frame_id)]
                features.append(basis.T @ (e_vectors[token].double() - reference_vector))
                targets.append(response.delta_residual.double())
        rho_template[template] = torch.stack([rho_frame[frame.frame_id] for frame in template_frames]).mean(dim=0)
        Z = torch.stack(features)  # [n, r]
        R = torch.stack(targets)  # [n, d]
        solution = torch.linalg.lstsq(Z, R).solution  # [r, d]
        maps[template] = solution.T.contiguous()  # [d, r]
    return LowRankFit(rank, mu, basis, maps, dict(reference_ids), rho_frame, rho_template, tuple(fit_tokens), singular)


def predict_epatch_shift(fit: LowRankFit, weights: pm.Weights, template: str, frame_id: str | None, e_vector: torch.Tensor, e_reference: torch.Tensor, noun: pm.Noun) -> float:
    """Δĉ_N(w, f) = −u_N · [LN(ρ + V_T Δz_T(w)) − LN(ρ)] with the exact final LayerNorm (float64)."""
    rho = fit.rho_frame[frame_id] if frame_id is not None and frame_id in fit.rho_frame else fit.rho_template[template]
    delta = fit.maps[template] @ fit.dz(template, e_vector, e_reference)
    gamma, beta = weights.ln_final_w.double(), weights.ln_final_b.double()
    u = weights.u(noun)
    return float(-u @ (pm.exact_layer_norm(rho + delta, gamma, beta, weights.eps) - pm.exact_layer_norm(rho, gamma, beta, weights.eps)))


def cue_level_values(fit: LowRankFit, weights: pm.Weights, e_vectors: Mapping[str, torch.Tensor], reference_vectors: Mapping[str, torch.Tensor],
                     responses: Mapping[tuple[str, str], EPatchResponse], frames: Sequence[pm.Frame], nouns: Sequence[pm.Noun], token: str) -> dict[str, float]:
    """For one token: mean predicted and measured E-patch shift over frames and nouns, and the mean absolute error."""
    predicted, measured, errors = [], [], []
    for frame in frames:
        response = responses[(token, frame.frame_id)]
        for noun in nouns:
            if not noun.single_token:
                continue
            value = predict_epatch_shift(fit, weights, frame.template_id, frame.frame_id, e_vectors[token], reference_vectors[frame.template_id], noun)
            predicted.append(value)
            measured.append(response.shifts[noun.lexical_key])
            errors.append(abs(value - response.shifts[noun.lexical_key]))
    return {"predicted": pm._mean(predicted), "measured": pm._mean(measured), "mae": pm._mean(errors)}


# ---------------------------------------------------------------------------
# Export / load of the weight-only program; the E005-scalar baseline.


def export_low_rank(directory: Path, weights: pm.Weights, fit: LowRankFit) -> dict[str, Any]:
    from .provenance import tensor_digest

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {
        "W_E": weights.W_E, "W_U": weights.W_U, "ln_final_w": weights.ln_final_w, "ln_final_b": weights.ln_final_b,
        "ln2_0_w": weights.ln2_0_w, "ln2_0_b": weights.ln2_0_b, "mlp0_W_in": weights.mlp0_W_in, "mlp0_b_in": weights.mlp0_b_in,
        "mlp0_W_out": weights.mlp0_W_out, "mlp0_b_out": weights.mlp0_b_out,
        "mu": fit.mu, "basis": fit.basis,
    }
    for template, matrix in fit.maps.items():
        tensors[f"map.{template}"] = matrix
    for frame_id, vector in fit.rho_frame.items():
        tensors[f"rho_frame.{frame_id}"] = vector
    for template, vector in fit.rho_template.items():
        tensors[f"rho_template.{template}"] = vector
    digests = {}
    for name, tensor in tensors.items():
        path = directory / f"{name}.pt"
        torch.save(tensor.detach().cpu().contiguous(), path)
        digests[name] = {"file": path.name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_digest(tensor)}
    index = {"eps": weights.eps, "act_fn": weights.act_fn, "rank": fit.rank, "templates": list(pm.TEMPLATE_ORDER),
             "reference_ids": dict(fit.reference_ids), "fit_tokens": list(fit.fit_tokens), "singular_values": list(fit.singular_values),
             "frames": sorted(fit.rho_frame), "tensors": digests}
    (directory / "parameters.json").write_text(pm.canonical_json(index) + "\n", encoding="utf-8")
    return index


def load_low_rank_program(parameters_dir: Path, program_path: Path) -> Any:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("low_rank_program", str(program_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LowRankProgram.load(Path(parameters_dir))


def e005_scalar_baseline(ctx: pm.DiscoveryContext, *, parameters_dir: Path, program_path: Path, expected_index_sha256: str | None, circuit: pm.MechanismSet = FIXED_CIRCUIT) -> tuple[Any, dict[str, Any]]:
    """Refit the frozen Experiment 005 M3 program on development data and record whether its digest matches the 005 lock."""
    parameters = pm.estimate_program_parameters(ctx, circuit)
    index = pm.export_program_parameters(parameters_dir, ctx.weights, parameters, vocab_size=int(ctx.weights.W_E.shape[0]))
    digest = pm.sha256_text(pm.canonical_json(index))
    program = pm.load_program(parameters_dir, program_path)
    return program, {"parameters_index_sha256": digest, "matches_experiment_005_lock": (digest == expected_index_sha256) if expected_index_sha256 else None,
                     "k_t": parameters.k_t, "gains": parameters.gains()}


# ---------------------------------------------------------------------------
# Leave-one-cue-out rank selection, the pre-lock quality gate, and τ.


def loco_errors(*, e_vectors: Mapping[str, torch.Tensor], reference_ids: Mapping[str, int], reference_vectors: Mapping[str, torch.Tensor],
                responses: Mapping[tuple[str, str], EPatchResponse], frames: Sequence[pm.Frame], cache: pm.PromptCache, nouns: Sequence[pm.Noun],
                weights: pm.Weights, tokens: Sequence[str], ranks: Sequence[int] = RANKS) -> dict[int, dict[str, dict[str, float]]]:
    """For every rank and held-out token: cue-level predicted/measured values and mean absolute error."""
    table: dict[int, dict[str, dict[str, float]]] = {}
    for rank in ranks:
        per_token = {}
        for held_out in tokens:
            fit_tokens = [token for token in tokens if token != held_out]
            fit = fit_low_rank(e_vectors=e_vectors, reference_ids=reference_ids, reference_vectors=reference_vectors, responses=responses,
                               frames=frames, cache=cache, fit_tokens=fit_tokens, rank=rank)
            per_token[held_out] = cue_level_values(fit, weights, e_vectors, reference_vectors, responses, frames, nouns, held_out)
        table[rank] = per_token
    return table


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    mean = pm._mean(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def select_rank(table: Mapping[int, Mapping[str, Mapping[str, float]]]) -> dict[str, Any]:
    """The design's executable rule; the cue token is the unit of the standard error."""
    summary = {}
    for rank, per_token in table.items():
        errors = [entry["mae"] for entry in per_token.values()]
        mean, se = _mean_se(errors)
        summary[int(rank)] = {"error": mean, "se": se, "n": len(errors)}
    error_1 = summary[1]["error"]
    eligible = [1] + [rank for rank in sorted(summary) if rank > 1 and summary[rank]["error"] <= RANK_IMPROVEMENT_FACTOR * error_1]
    r_best = min(eligible, key=lambda rank: (summary[rank]["error"], rank))
    threshold = summary[r_best]["error"] + summary[r_best]["se"]
    selected = min(rank for rank in eligible if summary[rank]["error"] <= threshold)
    return {"summary": summary, "eligible": eligible, "r_best": r_best, "threshold": threshold, "selected": selected, "rule": CONTINUATION_RULE_TEXT}


CONTINUATION_RULE_TEXT = ("eligible = {1} ∪ {r > 1 : error_r ≤ 0.8 × error_1}; r_best = argmin_{eligible} error_r; "
                          "threshold = error_{r_best} + SE_{r_best}; selected = min {r ∈ eligible : error_r ≤ threshold}")


def quality_gate(per_token: Mapping[str, Mapping[str, float]], *, selected: int, summary: Mapping[int, Mapping[str, float]]) -> dict[str, Any]:
    """Pre-lock gate on the sixteen cue-level values of the selected rank; independent of τ."""
    tokens = list(per_token)
    predicted = [per_token[token]["predicted"] for token in tokens]
    measured = [per_token[token]["measured"] for token in tokens]
    spearman = pm.spearman(predicted, measured)
    rmse = math.sqrt(pm._mean([(p - m) ** 2 for p, m in zip(predicted, measured)]))
    rms_measured = math.sqrt(pm._mean([m * m for m in measured]))
    normalized = rmse / rms_measured if rms_measured > 0 else float("inf")
    improvement_ok = selected == 1 or summary[selected]["error"] <= RANK_IMPROVEMENT_FACTOR * summary[1]["error"]
    return {"spearman": spearman, "spearman_ok": spearman >= QUALITY_SPEARMAN_FLOOR, "normalized_rmse": normalized,
            "normalized_rmse_ok": normalized <= QUALITY_NORMALIZED_RMSE_CEILING, "improvement_ok": improvement_ok,
            "passed": spearman >= QUALITY_SPEARMAN_FLOOR and normalized <= QUALITY_NORMALIZED_RMSE_CEILING and improvement_ok,
            "cue_level": {token: {"predicted": per_token[token]["predicted"], "measured": per_token[token]["measured"], "mae": per_token[token]["mae"]} for token in tokens}}


def tolerance_tau(per_token: Mapping[str, Mapping[str, float]]) -> float:
    """τ = max(0.5 nats, 3 × RMSE of the cue-level errors of the selected rank)."""
    errors = [entry["mae"] for entry in per_token.values()]
    return max(TAU_MIN_NATS, TAU_RMSE_MULTIPLIER * math.sqrt(pm._mean([error * error for error in errors])))
