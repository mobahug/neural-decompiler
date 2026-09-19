"""Experiment 011: does the weight-defined encoding read predict ``L03.H04``'s transport of new cues?

The quantity under test is ``g_E(w, T) = r(E(w) − E(ref_T)) / r(E(pl_T) − E(ref_T))`` with
``r(x) = ⟨x − mean(x), γ₃ ⊙ m⟩`` and ``m = W_V W_O d̂_T``: a zero-parameter read functional, fixed by
earlier experiments, applied to the cue's weight-only layer-0 encoding. Predictions for a frozen set of
new cue tokens and new frames are committed before any fresh forward pass (the lock phase touches
weights and the exposed results only); the single confirmation measures the head's transport of those
cues and scores it against the committed numbers (Y1) and against the frozen P1 equation (Y2), with
validity rules that never look at a candidate's own outcome. Constants are copied from design revision 2.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import cue_decompilation as cd
from . import cue_suppression as cs
from . import head_transport as ht
from . import plural_mechanism as pm
from . import read_assembly as ra
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 2).

EXPERIMENT_DIR = "experiments/011-encoding-read-prospective"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
INHERITED_010_EXTRACT_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-010-transport-fractions.json"
EXPERIMENT_009_LOCK_PATH = ht.LOCK_RELATIVE_PATH
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260924
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
INHERITED_010_SCHEMA_VERSION = 1
PHASES = ("explore", "lock", "confirm", "report")

Y1_SPEARMAN = 0.80
Y2_SPEARMAN = 0.90
TAU_MIN = 0.10
TAU_MULTIPLIER = 3.0
DENOMINATOR_RELATIVE_FLOOR = 0.25  # of the largest template denominator, and of σ_r
MIN_VALID_FRAMES = 4  # of 6
MIN_VALID_FRAMES_PER_TOKEN = 3
MIN_SCORED_TOKENS = 16
FRAME_CUE_EFFECT_RATE = 108 / 120
LOCK_PREDICTION_TOLERANCE = 1e-9
REPLICATION_TOLERANCE = 1e-6

CANDIDATES: dict[str, tuple[str, ...]] = {
    "determiner-like": ("an", "such", "much", "little", "less", "half", "whichever", "whatever"),
    "numeral": ("eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "zero"),
    "quantity": ("additional", "extra", "sufficient", "ample", "abundant", "scarce", "fewest", "least", "myriad", "manifold"),
    "possessive-or-pronoun": ("its", "whose", "mine", "yours", "hers", "theirs"),
    "adjective": ("green", "cheap", "warm", "dark", "huge", "black", "white", "young", "empty", "wooden"),
}
QUOTAS = {"determiner-like": 5, "numeral": 5, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 5}
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The cabinet stores {cue}"),
    ("cardinal", "The vendor sells {cue}"),
    ("quantifier", "The survey covers {cue}"),
    ("quantifier", "The monitor tracks {cue}"),
    ("coordinated-adjective", "Leo and Maya labeled {cue} round"),
    ("coordinated-adjective", "Priya and Jonas bundled {cue} soft"),
)
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", "experiments/010-read-direction-assembly/", "experiments/009-head-transport-rule/", "experiments/008-cue-suppression-localization/",
                            "experiments/007-supervised-cue-subspace/", "experiments/006-low-rank-cue-decompilation/", "experiments/005-regular-plural-mechanism/", "screening/behavior-candidates/manifest-v1.json")
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)
OUTCOME_Y1 = ("ENCODING_READ_PREDICTS_TRANSPORT", "ENCODING_READ_FAILS")
OUTCOME_Y2 = ("HEAD_P1_REPLICATED", "HEAD_P1_NOT_REPLICATED")


class PhaseError(pm.PhaseError):
    pass


# ---------------------------------------------------------------------------
# The weight-defined encoding read.


@dataclass(frozen=True)
class EncodingRead:
    """r(x) = ⟨x − mean(x), γ₃ ⊙ m⟩ with the frozen axes; g_E, g_∥, g_⊥ per token and template; the template denominators."""

    weight: torch.Tensor  # γ₃ ⊙ m (float64)
    e_direction: torch.Tensor  # d̂_E (float64)
    reference_ids: Mapping[str, int]
    plural_ids: Mapping[str, int]

    def inner(self, x: torch.Tensor) -> float:
        x = x.double()
        return float((x - x.mean()) @ self.weight)

    def denominator(self, weights: pm.Weights, template: str) -> float:
        return self.inner(pm.lexicon_vector(weights, self.plural_ids[template]).double() - pm.lexicon_vector(weights, self.reference_ids[template]).double())

    def reads(self, weights: pm.Weights, token_id: int, template: str) -> dict[str, float]:
        delta = pm.lexicon_vector(weights, token_id).double() - pm.lexicon_vector(weights, self.reference_ids[template]).double()
        parallel = (delta @ self.e_direction) * self.e_direction
        denominator = self.denominator(weights, template)
        g_E = self.inner(delta) / denominator
        g_par = self.inner(parallel) / denominator
        return {"g_E": g_E, "g_par": g_par, "g_perp": g_E - g_par}


def denominator_validity(denominators: Mapping[str, float], sigma_r: float) -> dict[str, Any]:
    """A template is defined iff its (signed) denominator is at least 0.25 × max_T |denominator| and at least 0.25 σ_r (a negative denominator is undefined)."""
    largest = max(abs(value) for value in denominators.values())
    defined = {template: value >= DENOMINATOR_RELATIVE_FLOOR * largest and value >= DENOMINATOR_RELATIVE_FLOOR * sigma_r for template, value in denominators.items()}
    return {"denominators": dict(denominators), "sigma_r": sigma_r, "largest": largest, "defined": defined, "n_defined": sum(defined.values())}


def sigma_r_from_pairs(read: EncodingRead, weights: pm.Weights, frames: Sequence[pm.Frame]) -> float:
    """The site-axis scale of r over the exposed frames' clean cue pairs (weight-only encodings): 0.5 × |mean of r(E(pl)) − r(E(sg))| (a scale, hence non-negative)."""
    differences = [read.inner(pm.lexicon_vector(weights, frame.cue_ids["pl"]).double()) - read.inner(pm.lexicon_vector(weights, frame.cue_ids["sg"]).double()) for frame in frames]
    return 0.5 * abs(pm._mean(differences))


def token_mean(values_by_frame: Mapping[str, float | None]) -> float | None:
    return cs._mean_or_none(list(values_by_frame.values()))


def tau(residuals: Sequence[float]) -> float:
    return max(TAU_MIN, TAU_MULTIPLIER * math.sqrt(pm._mean([value * value for value in residuals]))) if residuals else TAU_MIN


# ---------------------------------------------------------------------------
# Confirmation set (tokenizer-only; classes carry no expectation).


def licensed_frames(word: str, frames: Sequence[pm.Frame]) -> list[str]:
    if word == "an":
        return [frame.frame_id for frame in frames if frame.template_id in pm.CUE_FINAL_TEMPLATES]
    return [frame.frame_id for frame in frames]


def fresh_tokens_011(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen = set(excluded_ids)
    for category, words in CANDIDATES.items():
        count = 0
        for word in words:
            ids = pm._encode(tokenizer, " " + word)
            if len(ids) != 1 or ids[0] in seen:
                continue
            seen.add(ids[0])
            chosen.append({"word": word, "token_id": ids[0], "category": category})
            count += 1
            if count == QUOTAS[category]:
                break
    return chosen


def build_confirmation_payload(tokenizer: Any, manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009) -> dict[str, Any]:
    pool = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_011(tokenizer, excluded_ids)
    exposed_texts = {frame.text_template for frame in pool.frames}
    cue_ids_by_template = {frame.template_id: dict(frame.cue_ids) for frame in pool.frames if pool.frame_origin[frame.frame_id] == "manifest"}
    frames: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for template_id, text_template in FRESH_FRAMES:
        if text_template in exposed_texts:
            raise ValueError(f"fresh frame text {text_template!r} is an exposed frame")
        counters[template_id] = counters.get(template_id, 0) + 1
        frame = pm._build_new_frame(tokenizer, template_id, text_template, cue_ids_by_template[template_id], f"{template_id}-011-{counters[template_id]}")
        frames.append({"frame_id": frame.frame_id, "template_id": template_id, "text_template": text_template, "prefix_ids": list(frame.prefix_ids), "suffix_ids": list(frame.suffix_ids),
                       "cue_ids": dict(frame.cue_ids), "p_c": frame.p_c, "p_t": frame.p_t,
                       "prompts": {label: {"text": pm._decode(tokenizer, frame.prompt_ids(frame.cue_ids[label])), "token_ids": list(frame.prompt_ids(frame.cue_ids[label]))} for label in ("sg", "pl")}})
    frame_objects = [pm.Frame(e["template_id"], e["frame_id"], tuple(e["prefix_ids"]), tuple(e["suffix_ids"]), e["cue_ids"], e["text_template"], origin="extension") for e in frames]
    for token in tokens:
        token["licensed_frames"] = licensed_frames(token["word"], frame_objects)
    token_prompts = []
    for frame in frame_objects:
        for token in tokens:
            if frame.frame_id not in token["licensed_frames"]:
                continue
            ids = frame.prompt_ids(token["token_id"])
            text = pm._decode(tokenizer, ids)
            if pm._encode(tokenizer, text) != ids:
                raise ValueError(f"{frame.frame_id}/{token['word']}: text does not round-trip to the constructed token IDs")
            token_prompts.append({"frame_id": frame.frame_id, "word": token["word"], "token_id": token["token_id"], "text": text, "token_ids": list(ids), "p_c": frame.p_c, "p_t": frame.p_t})
    payload = {"schema_version": CONFIRMATION_SCHEMA_VERSION, "manifest_sha256": manifest_sha256, "extension_sha256": extension.content_sha256,
               "confirmation_006_sha256": confirmation_006.content_sha256, "confirmation_009_sha256": confirmation_009.content_sha256,
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "reference_cue_ids": dict(pool.reference_ids),
               "exposed_token_ids": sorted(token_id for _, token_id in pool.tokens), "policy": {"an_frames": "cue-final only", "quotas": dict(QUOTAS), "expectations": "none: the committed g_E values are the only predictions"},
               "tokens": tokens, "frames": frames, "token_prompts": token_prompts, "construction": "tokenizer-only; first eligible candidates per lexical class; classes carry no expectation; no model output"}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


@dataclass(frozen=True)
class Confirmation011:
    manifest_sha256: str
    extension_sha256: str
    reference_ids: Mapping[str, int]
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

    def frames_for(self, word: str) -> tuple[pm.Frame, ...]:
        licensed = next(token["licensed_frames"] for token in self.tokens if token["word"] == word)
        return tuple(frame for frame in self.frames if frame.frame_id in licensed)


def validate_confirmation(payload: Mapping[str, Any], manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009) -> Confirmation011:
    pm._require_exact_keys(payload, {"schema_version", "manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "model", "reference_cue_ids", "exposed_token_ids", "policy",
                                     "tokens", "frames", "token_prompts", "construction", "content_sha256"}, "confirmation")
    if payload["schema_version"] != CONFIRMATION_SCHEMA_VERSION:
        raise ValueError("confirmation schema version is not frozen")
    if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("confirmation content_sha256 does not match canonical payload")
    if (payload["manifest_sha256"], payload["extension_sha256"], payload["confirmation_006_sha256"], payload["confirmation_009_sha256"]) != (manifest_sha256, extension.content_sha256, confirmation_006.content_sha256, confirmation_009.content_sha256):
        raise ValueError("confirmation was built against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}:
        raise ValueError("confirmation was built for a different pinned model")
    pool = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
    if dict(payload["reference_cue_ids"]) != dict(pool.reference_ids) or list(payload["exposed_token_ids"]) != sorted(token_id for _, token_id in pool.tokens):
        raise ValueError("reference cues or exposed token IDs disagree with the exposed pool")
    exposed_texts = {frame.text_template for frame in pool.frames}
    frames = tuple(pm._parse_frame(entry, "extension") for entry in payload["frames"])
    expected, counters = [], {}
    for template_id, text_template in FRESH_FRAMES:
        counters[template_id] = counters.get(template_id, 0) + 1
        expected.append((f"{template_id}-011-{counters[template_id]}", template_id, text_template))
    if [(frame.frame_id, frame.template_id, frame.text_template) for frame in frames] != expected:
        raise ValueError("fresh frames are not the frozen literal frames")
    cue_ids_by_template = {frame.template_id: dict(frame.cue_ids) for frame in pool.frames if pool.frame_origin[frame.frame_id] == "manifest"}
    for frame in frames:
        if frame.text_template in exposed_texts or dict(frame.cue_ids) != cue_ids_by_template[frame.template_id]:
            raise ValueError(f"{frame.frame_id}: fresh frame text exposed or original cue tokens missing")
    exposed_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = []
    for entry in payload["tokens"]:
        pm._require_exact_keys(entry, {"word", "token_id", "category", "licensed_frames"}, "token")
        if entry["category"] not in CANDIDATES:
            raise ValueError(f"fresh token {entry['word']} has an unknown class {entry['category']}")
        if entry["token_id"] in exposed_ids or entry["word"] not in CANDIDATES[entry["category"]] or list(entry["licensed_frames"]) != licensed_frames(entry["word"], frames):
            raise ValueError(f"fresh token {entry['word']} violates the frozen candidate lists or the licensing policy")
        tokens.append(dict(entry))
    for category, words in CANDIDATES.items():
        chosen = [entry["word"] for entry in tokens if entry["category"] == category]
        if len(chosen) > QUOTAS[category] or [words.index(word) for word in chosen] != sorted(words.index(word) for word in chosen):
            raise ValueError(f"{category}: quota or candidate order violated")
    if len({entry["token_id"] for entry in tokens}) != len(tokens) or not tokens:
        raise ValueError("fresh tokens must be distinct and non-empty")
    frames_by_id = {frame.frame_id: frame for frame in frames}
    prompts = []
    for entry in payload["token_prompts"]:
        pm._require_exact_keys(entry, {"frame_id", "word", "token_id", "text", "token_ids", "p_c", "p_t"}, "token_prompt")
        frame = frames_by_id[entry["frame_id"]]
        token = next(token for token in tokens if token["word"] == entry["word"])
        if token["token_id"] != entry["token_id"] or frame.frame_id not in token["licensed_frames"] or tuple(entry["token_ids"]) != frame.prompt_ids(entry["token_id"]) or (entry["p_c"], entry["p_t"]) != (frame.p_c, frame.p_t):
            raise ValueError(f"{entry['frame_id']}/{entry['word']}: stored prompt disagrees with the frame or the policy")
        prompts.append(pm.Prompt(frame, int(entry["token_id"]), entry["word"]))
    expected_prompts = [(frame.frame_id, token["word"]) for frame in frames for token in tokens if frame.frame_id in token["licensed_frames"]]
    if [(prompt.frame.frame_id, prompt.cue_label) for prompt in prompts] != expected_prompts:
        raise ValueError("token prompts must cover exactly the licensed (frame, token) pairs in order")
    return Confirmation011(payload["manifest_sha256"], payload["extension_sha256"], dict(payload["reference_cue_ids"]), frames, tuple(tokens), tuple(prompts), payload["content_sha256"])


def freeze_confirmation(path: Path, tokenizer: Any, manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; the confirmation set is frozen and cannot be rebuilt")
    payload = build_confirmation_payload(tokenizer, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    validate_confirmation(payload, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)
    validate_json_safe(payload, path="confirmation")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    return payload["content_sha256"]


def load_confirmation(path: Path, manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009) -> Confirmation011:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, manifest, manifest_sha256, extension, confirmation_006, confirmation_009)


# ---------------------------------------------------------------------------
# Inherited Experiment 010 transport fractions (derived extract).


def inherited_010_payload(fractions: Mapping[str, float | None], *, source: Mapping[str, Any], manifest_sha256: str, extension_sha256: str, confirmation_sha256: str, confirmation_009_sha256: str, model: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"schema_version": INHERITED_010_SCHEMA_VERSION,
               "description": "Derived extract of Experiment 010's recorded transport fractions q_T per (token, frame) over the 63 × 24 exposed pool (null where the frame was uninformative for the token). Experiment 011 recomputes these and requires agreement within 1e-6.",
               "source": dict(source), "manifest_sha256": manifest_sha256, "extension_sha256": extension_sha256, "confirmation_sha256": confirmation_sha256, "confirmation_009_sha256": confirmation_009_sha256,
               "model": dict(model), "fractions": {key: (None if value is None else float(value)) for key, value in sorted(fractions.items())}}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_010(path: Path, *, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str, confirmation_009_sha256: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited 010 extract: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "description", "source", "manifest_sha256", "extension_sha256", "confirmation_sha256", "confirmation_009_sha256", "model", "fractions", "content_sha256"}, "inherited 010 extract")
    if payload["schema_version"] != INHERITED_010_SCHEMA_VERSION or payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("inherited 010 extract schema or digest is not frozen")
    if (payload["manifest_sha256"], payload["extension_sha256"], payload["confirmation_sha256"], payload["confirmation_009_sha256"]) != (manifest_sha256, extension_sha256, confirmation_sha256, confirmation_009_sha256):
        raise ValueError("inherited 010 extract was recorded against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision} or len(payload["fractions"]) != 63 * 24:
        raise ValueError("inherited 010 extract model or size is not frozen")
    return payload


def check_fraction_replication(measured: Mapping[str, float | None], recorded: Mapping[str, float | None]) -> dict[str, Any]:
    if set(measured) != set(recorded):
        raise pm.IncidentError("recomputed transport fractions do not cover exactly the recorded (token, frame) pairs")
    worst_key, worst = None, 0.0
    for key, value in recorded.items():
        other = measured[key]
        if (value is None) != (other is None):
            raise pm.IncidentError(f"{key}: informativeness differs from Experiment 010's record")
        if value is not None:
            deviation = abs(value - other)
            if deviation > worst:
                worst_key, worst = key, deviation
    if worst > REPLICATION_TOLERANCE:
        raise pm.IncidentError(f"{worst_key}: recomputed transport fraction deviates from Experiment 010 by {worst:.3e}")
    return {"passed": True, "n": len(recorded), "max_abs_deviation": worst, "worst_key": worst_key}


# ---------------------------------------------------------------------------
# Results state (four phases) and phase isolation.

_STATE_KEYS = {"schema_version", "run_id", "created_at", "manifest_sha256", "extension_sha256", "confirmation_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "protocol_code_commit", "git_dirty",
               "model", "versions", "phases", "executed_prompt_keys", "executed_noun_keys", "exploration", "lock", "confirmation", "invalidated_runs", "state_sha256"}


def new_results_state(*, digests: Mapping[str, str], protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {"schema_version": RESULTS_SCHEMA_VERSION, "run_id": pm.sha256_text("".join(digests[key] for key in sorted(digests)) + protocol_code_commit + pm.utc_now())[:16], "created_at": pm.utc_now(),
            "manifest_sha256": digests["manifest"], "extension_sha256": digests["extension"], "confirmation_sha256": digests["confirmation_006"], "confirmation_009_sha256": digests["confirmation_009"], "confirmation_011_sha256": digests["confirmation_011"],
            "protocol_code_commit": protocol_code_commit, "git_dirty": False, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "versions": dict(versions),
            "phases": {phase: {"status": "not_started"} for phase in PHASES}, "executed_prompt_keys": [], "executed_noun_keys": [], "exploration": {}, "lock": None, "confirmation": None, "invalidated_runs": []}


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


def assert_phase_allowed(phase: str, state: Mapping[str, Any]) -> None:
    """The same four-phase rules as Experiment 009, raised as this experiment's PhaseError."""
    try:
        ht.assert_phase_allowed(phase, state)
    except pm.PhaseError as error:
        raise PhaseError(str(error)) from None


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation011) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


# ---------------------------------------------------------------------------
# The inherited identity checks (Experiment 010's rules, enforced here in both phases).


def enforce_identities(record: ra.Attribution, plural: ra.Attribution, where: str) -> dict[str, float]:
    scale = max(abs(plural.rho_total), 1e-12)
    errors = {"rho_identity": record.identity_error / scale, "p1_cross_check": record.p1_cross_check / scale, "neuron_sum": record.neuron_sum_error}
    if errors["rho_identity"] > ra.IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{where}: ρ(Δr_c) ≠ ρ(ΔE) + Σ_k ρ(Δout_k) (relative error {errors['rho_identity']:.2e})")
    if errors["p1_cross_check"] > ra.P1_CROSS_CHECK_TOLERANCE:
        raise pm.IncidentError(f"{where}: ρ(Δr_c) does not equal the P1 prediction (relative error {errors['p1_cross_check']:.2e})")
    if errors["neuron_sum"] > ra.NEURON_SUM_TOLERANCE:
        raise pm.IncidentError(f"{where}: the neuron terms do not sum to ρ(ΔE) (error {errors['neuron_sum']:.2e})")
    return errors


def _max_errors(accumulator: dict[str, float], errors: Mapping[str, float]) -> None:
    for key, value in errors.items():
        accumulator[key] = max(accumulator.get(key, 0.0), value)


# ---------------------------------------------------------------------------
# Exploration (exposed pool: replication, axes, g_E for the 63, tolerances).


def run_exploration(model: Any, pool: cs.Pool008, *, state: dict[str, Any], results_path: Path | None, inherited_010: Mapping[str, Any], log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    nouns = pool.single_nouns
    exploration = state["exploration"]
    say("stage axes from the 24 exposed frames' clean cue pairs")
    axes = cs.stage_axes(cache, weights, pool)
    e_axis, axis_T = axes["R0"], axes["T"]
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = EncodingRead(ra.read_weight(head, axis_T), e_axis.direction.double(), dict(pool.reference_ids), plural_ids)
    denominators = {template: read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}
    validity = denominator_validity(denominators, sigma_r_from_pairs(read, weights, pool.frames))
    exploration["axes"] = {stage: {"sigma": axis.sigma} for stage, axis in axes.items()}
    exploration["axes_vectors"] = {"T": axis_T.direction.double().tolist(), "R0": e_axis.direction.double().tolist()}
    exploration["read_weight"] = read.weight.tolist()
    exploration["denominators"] = validity
    say(f"template denominators {denominators}; defined {validity['defined']}")
    say("measurements: 63 tokens × 24 frames (transport fractions and the P1 fractions)")
    records: dict[tuple[str, str], ra.Attribution] = {}
    for frame in pool.frames:
        ref, components = ra.capture_reference(model, head, pool.reference_prompt(frame), nouns)
        functional = ra.read_functional(head, ref, axis_T)
        for name, token_id in pool.tokens:
            records[(name, frame.frame_id)] = ra.measure_token_010(model, weights, head, ref, components, functional, name, token_id, e_axis, axis_T, nouns)
    q_by_pair: dict[str, float | None] = {}
    p1_residuals: list[float] = []
    per_token: dict[str, dict[str, Any]] = {}
    identities: dict[str, float] = {}
    for name, token_id in pool.tokens:
        q_frames, g_frames, layers = {}, {}, []
        for frame in pool.frames:
            record = records[(name, frame.frame_id)]
            plural = records[(pool.plural_cue[frame.template_id], frame.frame_id)]
            if not record.own_reference:
                _max_errors(identities, enforce_identities(record, plural, f"{frame.frame_id}/{name}"))
            analysis = ra.fractions(record, plural, axis_T, plural.g_E_inner)
            q_by_pair[f"{name}|{frame.frame_id}"] = None if analysis is None else analysis["q_T"]
            # F(w): the token's informative frames whose template has a defined denominator — the same set for ḡ_E, q̄_T, and the P1 residuals.
            if analysis is not None and validity["defined"][frame.template_id]:
                q_frames[frame.frame_id] = analysis["q_T"]
                g_frames[frame.frame_id] = read.reads(weights, token_id, frame.template_id)["g_E"]
                p1_residuals.append(analysis["f_total"] - analysis["q_T"])
                layers.append(analysis["f_layers"])
        g_mean, q_mean = token_mean(g_frames), token_mean(q_frames)
        per_token[name] = {"token": name, "token_id": token_id, "category": pool.token_category[name], "source": pool.token_source[name], "n_frames": len(q_frames),
                           "g_E_mean": g_mean, "q_T_mean": q_mean, "f_layers_mean": pm._mean(layers) if layers else None,
                           "reads": {template: read.reads(weights, token_id, template) for template in pm.TEMPLATE_ORDER if validity["defined"][template]}}
    exploration["identities"] = identities
    exploration["replication"] = {"experiment_010": check_fraction_replication(q_by_pair, inherited_010["fractions"])}
    say(f"replication against Experiment 010: max deviation {exploration['replication']['experiment_010']['max_abs_deviation']:.2e}")
    residuals = [row["g_E_mean"] - row["q_T_mean"] for row in per_token.values() if row["g_E_mean"] is not None and row["q_T_mean"] is not None]
    tau_g = tau(residuals)
    tau_M = tau(p1_residuals)
    pairs = [(row["g_E_mean"], row["q_T_mean"]) for row in per_token.values() if row["g_E_mean"] is not None and row["q_T_mean"] is not None]
    exploration["tokens"] = per_token
    exploration["tolerances"] = {"tau_g": tau_g, "tau_M": tau_M, "n_tokens": len(residuals), "n_pairs": len(p1_residuals)}
    layer_values = [row["f_layers_mean"] for row in per_token.values() if row["f_layers_mean"] is not None]
    exploration["exposed_net_layer_range"] = {"min": min(layer_values), "max": max(layer_values)} if layer_values else None
    exploration["sigma_T"] = axis_T.sigma
    exploration["exposed_check"] = {"g_E_vs_q_T": {"spearman": pm.spearman([g for g, _ in pairs], [q for _, q in pairs]), "mae": pm._mean([abs(g - q) for g, q in pairs]), "n": len(pairs)},
                                    "p1_vs_q_T": {"mae": pm._mean([abs(r) for r in p1_residuals]), "n": len(p1_residuals)}}
    exploration["summary"] = {"defined_templates": [t for t, ok in validity["defined"].items() if ok], "tau_g": tau_g, "tau_M": tau_M, "exposed_spearman": exploration["exposed_check"]["g_E_vs_q_T"]["spearman"],
                              "prediction_undefined": validity["n_defined"] < 2}
    say(f"tau_g {tau_g:.3f}, tau_M {tau_M:.3f}; exposed g_E vs q_T Spearman {exploration['exposed_check']['g_E_vs_q_T']['spearman']:.3f}")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


# ---------------------------------------------------------------------------
# Lock (weights only), the prediction artifact, validation, reproduction.


def read_from_lock_inputs(exploration: Mapping[str, Any], reference_ids: Mapping[str, int], plural_ids: Mapping[str, int]) -> EncodingRead:
    return EncodingRead(torch.tensor(exploration["read_weight"], dtype=torch.float64), torch.tensor(exploration["axes_vectors"]["R0"], dtype=torch.float64), dict(reference_ids), dict(plural_ids))


def lock_predictions(weights: pm.Weights, read: EncodingRead, confirmation: Confirmation011, defined: Mapping[str, bool], baseline: ht.TransportRule | None) -> dict[str, Any]:
    """Every fresh token's g_E, g_∥, g_⊥ per defined template, its licensed-frame mean, and the 009 baseline — from weights, no forward pass."""
    tokens = {}
    for token in confirmation.tokens:
        word, token_id = token["word"], token["token_id"]
        frames = confirmation.frames_for(word)
        by_template = {template: read.reads(weights, token_id, template) for template in pm.TEMPLATE_ORDER if defined[template]}
        licensed = [frame for frame in frames if defined[frame.template_id]]
        means = {key: pm._mean([by_template[frame.template_id][key] for frame in licensed]) for key in ("g_E", "g_par", "g_perp")} if licensed else {}
        entry = {"token_id": token_id, "category": token["category"], "licensed_frames": [frame.frame_id for frame in licensed], "by_template": by_template, "means": means}
        if baseline is not None:
            entry["baseline_009"] = {template: baseline.predict(template, pm.lexicon_vector(weights, token_id).double() - pm.lexicon_vector(weights, read.reference_ids[template]).double()) for template in by_template}
        tokens[word] = entry
    return {"tokens": tokens}


def render_predictions(lock: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 011 — preregistered predictions (zero-parameter encoding read)", "", f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_011_sha256']}`",
             f"- τ_g {f(lock['tolerances']['tau_g'], 3)} (Y1 MAE ceiling, token means); τ_M {f(lock['tolerances']['tau_M'], 3)} (Y2, pairs); floors Spearman ≥ {Y1_SPEARMAN} (Y1), ≥ {Y2_SPEARMAN} (Y2)",
             f"- Defined templates: {lock['defined_templates']}; denominators {lock['denominators']['denominators']}", "",
             "| token | class | template | predicted g_E | g_∥ | g_⊥ | 009 rank-1 rule |", "|---|---|---|---|---|---|---|"]
    for word, entry in lock["predictions"]["tokens"].items():
        for template, reads in entry["by_template"].items():
            baseline = (entry.get("baseline_009") or {}).get(template)
            lines.append(f"| {word} | {entry['category']} | {template} | {f(reads['g_E'], 3)} | {f(reads['g_par'], 3)} | {f(reads['g_perp'], 3)} | {f(baseline, 3)} |")
        lines.append(f"| **{word}** | | **mean over {len(entry['licensed_frames'])} licensed frames** | **{f(entry['means'].get('g_E'), 3)}** | {f(entry['means'].get('g_par'), 3)} | {f(entry['means'].get('g_perp'), 3)} | |")
    lines.append("")
    return "\n".join(lines)


def build_candidate_lock(*, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation011, predictions: Mapping[str, Any], protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    if exploration["summary"]["prediction_undefined"]:
        raise PhaseError("fewer than two templates have a defined denominator: PREDICTION_UNDEFINED, no lock")
    lock = {"schema_version": 1, "experiment": "011", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit,
            "manifest_sha256": digests["manifest"], "extension_sha256": digests["extension"], "confirmation_sha256": digests["confirmation_006"], "confirmation_009_sha256": digests["confirmation_009"], "confirmation_011_sha256": confirmation.content_sha256,
            "confirmation_prompt_keys": sorted(prompt.key for prompt in confirmation.all_prompts), "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED},
            "head": ht.HEAD_KEY, "axes_vectors": exploration["axes_vectors"], "sigma_T": exploration["sigma_T"], "read_weight": exploration["read_weight"], "denominators": exploration["denominators"], "defined_templates": exploration["summary"]["defined_templates"],
            "exposed_net_layer_range": exploration.get("exposed_net_layer_range"),
            "tolerances": exploration["tolerances"], "exposed_check": exploration["exposed_check"],
            "floors": {"y1_spearman": Y1_SPEARMAN, "y2_spearman": Y2_SPEARMAN, "min_valid_frames": MIN_VALID_FRAMES, "min_valid_frames_per_token": MIN_VALID_FRAMES_PER_TOKEN, "min_scored_tokens": MIN_SCORED_TOKENS,
                       "frame_cue_effect_rate": FRAME_CUE_EFFECT_RATE, "head_stage_floor": cs.STAGE_UNINFORMATIVE_FLOOR},
            "predictions": dict(predictions), "tier_a_results_sha256": state.get("state_sha256")}
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation011, predictions_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "011" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked or git_state.get("dirty", True):
        raise PhaseError("confirm requires the committed lock and predictions on a clean Git tree")
    if (lock["manifest_sha256"], lock["extension_sha256"], lock["confirmation_sha256"], lock["confirmation_009_sha256"], lock["confirmation_011_sha256"]) != (digests["manifest"], digests["extension"], digests["confirmation_006"], digests["confirmation_009"], confirmation.content_sha256):
        raise PhaseError("lock was built against different frozen inputs")
    if not state.get("lock") or lock["content_sha256"] != state["lock"]["content_sha256"] or lock["run_id"] != state["run_id"]:
        raise PhaseError("the installed lock is not the candidate lock written by the lock phase")
    if pm.sha256_text(predictions_text) != state["lock"]["predictions_sha256"]:
        raise PhaseError("the installed predictions.md is not the artifact written by the lock phase")
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = [path for path in changed_paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and path not in NON_SCIENTIFIC_PATHS and not path.startswith(NON_SCIENTIFIC_PREFIXES)]
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock commit: {scientific}")
    assert_confirmation_untouched(state, confirmation)


def assert_lock_predictions_reproduced(lock: Mapping[str, Any], recomputed: Mapping[str, Any]) -> float:
    worst = 0.0
    for word, entry in lock["predictions"]["tokens"].items():
        fresh = recomputed["tokens"][word]
        for template, reads in entry["by_template"].items():
            for key, value in reads.items():
                worst = max(worst, abs(value - fresh["by_template"][template][key]))
        for key, value in entry["means"].items():
            worst = max(worst, abs(value - fresh["means"][key]))
        for template, value in (entry.get("baseline_009") or {}).items():
            worst = max(worst, abs(value - fresh["baseline_009"][template]))
    if worst > LOCK_PREDICTION_TOLERANCE:
        raise PhaseError(f"the weights and locked axes do not reproduce the preregistered predictions (max difference {worst:.3e}); nothing was executed")
    return worst


# ---------------------------------------------------------------------------
# Confirmation (Tier C), validity, floors, outcome, report.


def run_confirmation(model: Any, pool: cs.Pool008, confirmation: Confirmation011, lock: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    nouns = pool.single_nouns
    cache = pm.PromptCache(model, tuple(pool.nouns))
    say("stage axes (exposed frames only)")
    axes = cs.stage_axes(cache, weights, pool)
    if pm.cosine(axes["T"].direction, torch.tensor(lock["axes_vectors"]["T"], dtype=torch.float64)) < 1.0 - 1e-6 or pm.cosine(axes["R0"].direction, torch.tensor(lock["axes_vectors"]["R0"], dtype=torch.float64)) < 1.0 - 1e-6:
        raise pm.IncidentError("the stage axes differ from the locked directions")
    if abs(axes["T"].sigma - lock["sigma_T"]) > 1e-9:
        raise pm.IncidentError("the head-stage scale σ_T differs from the locked value")
    e_axis, axis_T = axes["R0"], axes["T"]
    frames_out: dict[str, Any] = {}
    per_token_frames: dict[str, dict[str, dict[str, Any]]] = {token["word"]: {} for token in confirmation.tokens}
    identities: dict[str, float] = {}
    say("fresh frames: reference, plural cue (validity and normalization), cue pairs")
    for frame in confirmation.frames:
        template = frame.template_id
        if template not in lock["defined_templates"]:
            frames_out[frame.frame_id] = {"template_id": template, "valid": False, "template_defined": False, "note": "template excluded before the lock; nothing run"}
            say(f"  {frame.frame_id}: template {template} undefined; skipped")
            continue
        reference = confirmation.reference_prompt(frame)
        ref, components = ra.capture_reference(model, head, reference, nouns)
        functional = ra.read_functional(head, ref, axis_T)
        sg, pl = pm.frame_prompts(frame)
        c_a, c_b = cache.c(sg), cache.c(pl)
        positive = sum(1 for noun in nouns if c_a[noun.lexical_key] - c_b[noun.lexical_key] > 0)
        required = pm.exact_count_floor(FRAME_CUE_EFFECT_RATE, len(nouns))
        plural = ra.measure_token_010(model, weights, head, ref, components, functional, pool.plural_cue[template], pl.cue_token_id, e_axis, axis_T, nouns)
        _max_errors(identities, enforce_identities(plural, plural, f"{frame.frame_id}/{pool.plural_cue[template]}"))
        head_informative = abs(plural.head_change) >= cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma
        valid = head_informative and positive >= required
        frames_out[frame.frame_id] = {"template_id": template, "valid": valid, "head_informative": head_informative, "plural_head_change": plural.head_change, "cue_effect_positive": positive, "cue_effect_required": required,
                                      "template_defined": True, "reconstruction_error": ref.reconstruction_error, "attention_ref": functional.attention_ref, "sigma_ref": functional.sigma_ref}
        if not valid:
            say(f"  {frame.frame_id}: INVALID (head informative {head_informative}, cue effect {positive}/{required})")
            continue
        for token in confirmation.tokens:
            if frame.frame_id not in token["licensed_frames"]:
                continue
            record = ra.measure_token_010(model, weights, head, ref, components, functional, token["word"], token["token_id"], e_axis, axis_T, nouns)
            _max_errors(identities, enforce_identities(record, plural, f"{frame.frame_id}/{token['word']}"))
            analysis = ra.fractions(record, plural, axis_T, plural.g_E_inner)
            if analysis is None:
                raise pm.IncidentError(f"{frame.frame_id}/{token['word']}: no fractions although the frame is valid")
            c_word = cache.c(pm.Prompt(frame, token["token_id"], token["word"]))
            analysis["dc_beh"] = pm._mean([c_word[noun.lexical_key] - ref.c_by_noun[noun.lexical_key] for noun in nouns])
            analysis["dc"] = pm._mean(list(record.shifts.values()))
            analysis["p1_fraction"] = analysis["f_total"]  # ρ_f(Δr_c) equals the P1 projection (enforced above)
            per_token_frames[token["word"]][frame.frame_id] = analysis
        say(f"  {frame.frame_id}: valid; {sum(1 for token in confirmation.tokens if frame.frame_id in token['licensed_frames'])} tokens")
    results = score_confirmation(frames_out, per_token_frames, confirmation, lock)
    results["identities"] = identities
    return results


def score_confirmation(frames_out: Mapping[str, Any], per_token_frames: Mapping[str, Mapping[str, Mapping[str, Any]]], confirmation: Confirmation011, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Pure scoring: validity counts, token means over identical frame sets, Y1, Y2, descriptive items, outcome."""
    valid_frames = [frame_id for frame_id, entry in frames_out.items() if entry["valid"]]
    # Scoring: a token is scored iff it has at least MIN_VALID_FRAMES_PER_TOKEN valid licensed frames; F(w) = those frames for both means.
    tokens_out = {}
    for token in confirmation.tokens:
        word = token["word"]
        frames = per_token_frames[word]
        locked = lock["predictions"]["tokens"][word]
        scored = len(frames) >= MIN_VALID_FRAMES_PER_TOKEN
        g_values = [locked["by_template"][confirmation_frame_template(confirmation, frame_id)]["g_E"] for frame_id in frames]
        tokens_out[word] = {"category": token["category"], "scored": scored, "n_valid_frames": len(frames), "frames": sorted(frames),
                            "g_E_mean": pm._mean(g_values) if frames else None, "q_T_mean": pm._mean([a["q_T"] for a in frames.values()]) if frames else None,
                            "g_E_locked_mean": locked["means"].get("g_E"), "g_par_mean": pm._mean([locked["by_template"][confirmation_frame_template(confirmation, f)]["g_par"] for f in frames]) if frames else None,
                            "g_perp_mean": pm._mean([locked["by_template"][confirmation_frame_template(confirmation, f)]["g_perp"] for f in frames]) if frames else None,
                            "baseline_009_mean": pm._mean([locked["baseline_009"][confirmation_frame_template(confirmation, f)] for f in frames]) if frames and "baseline_009" in locked else None,
                            "f_E_mean": pm._mean([a["f_E"] for a in frames.values()]) if frames else None, "f_perp_mean": pm._mean([a["f_perp"] for a in frames.values()]) if frames else None,
                            "f_layers_mean": pm._mean([a["f_layers"] for a in frames.values()]) if frames else None, "G_mean": pm._mean([a["G"] for a in frames.values()]) if frames else None,
                            "p1_mean": pm._mean([a["p1_fraction"] for a in frames.values()]) if frames else None, "dc_mean": pm._mean([a["dc"] for a in frames.values()]) if frames else None,
                            "dc_beh_mean": pm._mean([a["dc_beh"] for a in frames.values()]) if frames else None}
    scored_tokens = [word for word, entry in tokens_out.items() if entry["scored"]]
    results: dict[str, Any] = {"frames": frames_out, "valid_frames": valid_frames, "tokens": tokens_out, "scored_tokens": scored_tokens, "per_frame": per_token_frames}
    precondition_ok = len(valid_frames) >= MIN_VALID_FRAMES and len(scored_tokens) >= MIN_SCORED_TOKENS
    results["precondition"] = {"passed": precondition_ok, "valid_frames": len(valid_frames), "scored_tokens": len(scored_tokens), "min_valid_frames": MIN_VALID_FRAMES, "min_scored_tokens": MIN_SCORED_TOKENS}
    if precondition_ok:
        g = [tokens_out[w]["g_E_mean"] for w in scored_tokens]
        q = [tokens_out[w]["q_T_mean"] for w in scored_tokens]
        spearman = pm.spearman(g, q)
        mae = pm._mean([abs(a - b) for a, b in zip(g, q)])
        results["Y1"] = {"spearman": spearman, "mae": mae, "tau_g": lock["tolerances"]["tau_g"], "n": len(scored_tokens), "passed": spearman >= Y1_SPEARMAN and mae <= lock["tolerances"]["tau_g"],
                         "failing": [name for name, ok in (("spearman", spearman >= Y1_SPEARMAN), ("mae", mae <= lock["tolerances"]["tau_g"])) if not ok]}
        if all(tokens_out[w]["baseline_009_mean"] is not None for w in scored_tokens):
            b = [tokens_out[w]["baseline_009_mean"] for w in scored_tokens]
            results["baseline_009"] = {"spearman": pm.spearman(b, q), "mae": pm._mean([abs(a - c) for a, c in zip(b, q)])}
        pairs = [(a["p1_fraction"], a["q_T"]) for w in scored_tokens for a in per_token_frames[w].values()]
        spearman_2 = pm.spearman([p for p, _ in pairs], [t for _, t in pairs])
        mae_2 = pm._mean([abs(p - t) for p, t in pairs])
        results["Y2"] = {"spearman": spearman_2, "mae": mae_2, "tau_M": lock["tolerances"]["tau_M"], "n_pairs": len(pairs), "passed": spearman_2 >= Y2_SPEARMAN and mae_2 <= lock["tolerances"]["tau_M"]}
        low = [w for w in scored_tokens if tokens_out[w]["g_E_locked_mean"] is not None and tokens_out[w]["g_E_locked_mean"] <= 0.35]
        results["descriptive"] = {"low_predicted_tokens": low, "f_perp_negative_frames": {w: sum(1 for a in per_token_frames[w].values() if a["f_perp"] < 0) for w in low},
                                  "net_layer_change_mean": pm._mean([tokens_out[w]["f_layers_mean"] for w in scored_tokens]), "gross_layer_change_mean": pm._mean([tokens_out[w]["G_mean"] for w in scored_tokens]),
                                  "exposed_gross_baseline_note": "Experiment 010: median gross layer change 0.44 over the 63 exposed tokens; no relay claim is made without this baseline"}
        exposed_range = lock.get("exposed_net_layer_range")
        if exposed_range:
            inside = sum(1 for w in scored_tokens if exposed_range["min"] <= tokens_out[w]["f_layers_mean"] <= exposed_range["max"])
            results["descriptive"]["net_layer_change_within_exposed_range"] = {"count": inside, "of": len(scored_tokens), "range": exposed_range}
    results["outcome"] = outcome(results)
    return results


def confirmation_frame_template(confirmation: Confirmation011, frame_id: str) -> str:
    return next(frame.template_id for frame in confirmation.frames if frame.frame_id == frame_id)


def outcome(results: Mapping[str, Any]) -> dict[str, Any]:
    if not results["precondition"]["passed"]:
        return {"label": "PRECONDITION_FAILED", "Y1": None, "Y2": None}
    y1 = "ENCODING_READ_PREDICTS_TRANSPORT" if results["Y1"]["passed"] else "ENCODING_READ_FAILS"
    y2 = "HEAD_P1_REPLICATED" if results["Y2"]["passed"] else "HEAD_P1_NOT_REPLICATED"
    return {"label": f"{y1} | {y2}", "Y1": y1, "Y2": y2}


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 011 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state['confirmation_011_sha256']}`", f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    incidents = list(exploration.get("incidents", [])) + ([state["confirmation"]["incident"]] if isinstance(state.get("confirmation"), dict) and "incident" in state["confirmation"] else [])
    if incidents:
        lines += ["## Incidents", ""] + [f"- `{e['phase']}` at commit `{e.get('commit', '?')}` ({e['at']}): {e['message']}" for e in incidents] + [""]
    if "summary" in exploration:
        s = exploration["summary"]
        lines += ["## Tier A — exposed pool (calibration only)", "", f"- Replication against Experiment 010: {exploration['replication']['experiment_010']['n']} pairs, max deviation {exploration['replication']['experiment_010']['max_abs_deviation']:.2e}",
                  f"- Template denominators r(E(pl) − E(ref)): {exploration['denominators']['denominators']}; σ_r {f(exploration['denominators']['sigma_r'], 3)}; defined {s['defined_templates']}",
                  f"- τ_g {f(s['tau_g'], 3)} (from {exploration['tolerances']['n_tokens']} token-mean residuals); τ_M {f(s['tau_M'], 3)} (from {exploration['tolerances']['n_pairs']} pairs)",
                  f"- Exposed (descriptive): ḡ_E vs q̄_T Spearman {f(exploration['exposed_check']['g_E_vs_q_T']['spearman'], 3)}, MAE {f(exploration['exposed_check']['g_E_vs_q_T']['mae'], 3)}; P1 pair MAE {f(exploration['exposed_check']['p1_vs_q_T']['mae'], 3)}", ""]
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}`; predictions sha256 `{state['lock']['predictions_sha256']}`", ""]
    confirmation = state.get("confirmation")
    if confirmation and "outcome" in confirmation:
        o = confirmation["outcome"]
        pre = confirmation["precondition"]
        lines += [f"## Confirmation — `{o['label']}`", "", f"- Valid frames {pre['valid_frames']}/6 (min {pre['min_valid_frames']}); scored tokens {pre['scored_tokens']} (min {pre['min_scored_tokens']})"]
        for frame_id, entry in confirmation["frames"].items():
            lines.append(f"  - {frame_id}: {'valid' if entry['valid'] else 'INVALID'} (plural head change {f(entry['plural_head_change'], 3)}, cue effect {entry['cue_effect_positive']}/{entry['cue_effect_required']})")
        if "Y1" in confirmation:
            y1, y2 = confirmation["Y1"], confirmation["Y2"]
            lines += [f"- Y1 (ḡ_E vs q̄_T over {y1['n']} scored tokens): Spearman {f(y1['spearman'], 3)}, MAE {f(y1['mae'], 3)} (τ_g {f(y1['tau_g'], 3)}) → {'pass' if y1['passed'] else 'FAIL'} {y1['failing'] or ''}",
                      f"- Y2 (P1 vs q_T over {y2['n_pairs']} pairs): Spearman {f(y2['spearman'], 3)}, MAE {f(y2['mae'], 3)} (τ_M {f(y2['tau_M'], 3)}) → {'pass' if y2['passed'] else 'FAIL'}"]
            if "baseline_009" in confirmation:
                lines.append(f"- Baseline (Experiment 009 rank-1 rule, reported only): Spearman {f(confirmation['baseline_009']['spearman'], 3)}, MAE {f(confirmation['baseline_009']['mae'], 3)}")
            d = confirmation["descriptive"]
            lines.append(f"- Descriptive: tokens with locked ḡ_E ≤ 0.35 {d['low_predicted_tokens']} with f_⊥ < 0 in frames {d['f_perp_negative_frames']}; mean net layer change {f(d['net_layer_change_mean'], 3)}, mean gross {f(d['gross_layer_change_mean'], 3)} ({d['exposed_gross_baseline_note']})")
        lines += ["", "| token | class | scored | valid frames | predicted ḡ_E | measured q̄_T | g_∥ | g_⊥ | f_E | f_⊥ | net L1–2 | P1 | 009 rule | contrast | behavior |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for word, row in sorted(confirmation["tokens"].items(), key=lambda kv: (kv[1]["g_E_mean"] if kv[1]["g_E_mean"] is not None else 9)):
            lines.append(f"| {word} | {row['category']} | {row['scored']} | {row['n_valid_frames']} | {f(row['g_E_mean'], 3)} | {f(row['q_T_mean'], 3)} | {f(row['g_par_mean'])} | {f(row['g_perp_mean'])} | {f(row['f_E_mean'])} | {f(row['f_perp_mean'])} | {f(row['f_layers_mean'])} | {f(row['p1_mean'])} | {f(row['baseline_009_mean'])} | {f(row['dc_mean'])} | {f(row['dc_beh_mean'])} |")
        lines.append("")
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
