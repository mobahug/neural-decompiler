"""Experiment 009: the transport rule of ``L03.H04`` — exact OV decomposition and a prospective test on frozen new cues.

Reuses the Experiment 008 pool, stage axes, fractions, and replication checks (``cue_suppression``),
the Experiment 007 estimators, gates, and exports (``supervised_subspace``), and the Experiment 006
confirmation and lock conventions (``cue_decompilation``). Adds: the head's weights and the exact
decomposition of its output change into the fixed-normalization linear OV read-out (P1), the exact
LayerNorm at the cue position (P2), the patched attention pattern (P3), and the remainder; the
component patches traced through the stages; the transport rule and the forty-token contrast rule
with leave-one-cue-out gates; a tokenizer-only confirmation set under the frozen
grammatical-compatibility policy; the lock with its human-readable prediction artifact; the single
confirmation scored only against the locked predictions. Constants are copied from design revision 2.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import cue_decompilation as cd
from . import cue_suppression as cs
from . import plural_mechanism as pm
from . import supervised_subspace as ss
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest, Split
from .interventions import ReplacementSource
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 2).

EXPERIMENT_DIR = "experiments/009-head-transport-rule"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
PROGRAM_SOURCE = "experiments/007-supervised-cue-subspace/linear_cue_program.py"  # the weight-only linear cue program (reused)
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260922
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
PHASES = ("explore", "lock", "confirm", "report")

HEAD_LAYER, HEAD_INDEX, HEAD_KEY = 3, 4, "L03.H04"
G_MAX = 0.25  # non-additivity stage: first stage with |g_s| > G_MAX
MECHANISM_R2_FLOOR = 0.90
MECHANISM_MAE_CEILING = 0.10
TAU_MIN = 0.10  # τ_M and τ₂ floors (q units)
TAU_MULTIPLIER = 3.0
Y1_SPEARMAN = 0.90
Y2_SPEARMAN = 0.80
SINGULAR_MAX_Q = 0.35
PLURAL_MIN_Q = 0.65
PLURAL_RATE = 0.8
AN_MIN_VOWEL_NOUNS = 10
VOWELS = "aeiou"
OV_IDENTITY_TOLERANCE = 1e-4  # relative to the plural cue's |ΔT| in the frame
RECONSTRUCTION_TOLERANCE = 1e-3  # clean head result versus Σ_k A_k v_k W_O (relative)
LOCK_PREDICTION_TOLERANCE = 1e-9
ADDITIVITY_STAGES = ("R1", "T", "R2", "R3", "c")
LEVELS = (1, 2, 3)
LEVEL_MEANING = {1: "fixed-normalization linear OV read-out of the incoming cue residual; LayerNorm-scale and attention modulation unnecessary",
                 2: "LayerNorm-dependent gating at the cue position", 3: "attention modulation matters"}

CANDIDATES: dict[str, tuple[str, ...]] = {
    "singular-selecting": ("either", "neither", "an"),
    "plural-numeral": ("eleven", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"),
    "plural-quantity": ("more", "most", "other", "enough", "certain", "additional", "extra", "assorted", "sufficient", "innumerable"),
    "number-neutral": ("my", "your", "his", "her", "our", "their", "its"),
    "bare-adjective": ("small", "blue", "new", "cold", "green", "cheap", "warm", "dark"),
}
QUOTAS = {"singular-selecting": 3, "plural-numeral": 6, "plural-quantity": 5, "number-neutral": 6, "bare-adjective": 4}
EXPECTATIONS = {"singular-selecting": "low", "plural-numeral": "high", "plural-quantity": "high", "number-neutral": "unconstrained", "bare-adjective": "high"}
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The shelf carries {cue}"),
    ("cardinal", "The ledger names {cue}"),
    ("quantifier", "The manual describes {cue}"),
    ("quantifier", "The bulletin mentions {cue}"),
    ("coordinated-adjective", "Ida and Tomas counted {cue} shiny"),
    ("coordinated-adjective", "Yusuf and Petra wrapped {cue} thin"),
)
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", "experiments/008-cue-suppression-localization/", "experiments/007-supervised-cue-subspace/",
                            "experiments/006-low-rank-cue-decompilation/", "experiments/005-regular-plural-mechanism/", "screening/behavior-candidates/manifest-v1.json")
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)


class PhaseError(pm.PhaseError):
    pass


# ---------------------------------------------------------------------------
# The head's weights and the exact OV decomposition.


@dataclass(frozen=True)
class HeadWeights:
    layer: int
    head: int
    ln_w: torch.Tensor  # γ of the attention LayerNorm (float64)
    ln_b: torch.Tensor
    eps: float
    W_V: torch.Tensor  # [d_model, d_head]
    b_V: torch.Tensor  # [d_head]
    W_O: torch.Tensor  # [d_head, d_model]

    @classmethod
    def from_model(cls, model: Any, layer: int = HEAD_LAYER, head: int = HEAD_INDEX) -> "HeadWeights":
        block = model.blocks[layer]
        attn = block.attn

        def grab(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().to("cpu", torch.float64).clone()

        W_V = grab(attn.W_V[head])
        b_V = grab(attn.b_V[head]) if getattr(attn, "b_V", None) is not None else torch.zeros(W_V.shape[1], dtype=torch.float64)
        return cls(layer, head, grab(block.ln1.w), grab(block.ln1.b), float(model.cfg.eps), W_V, b_V, grab(attn.W_O[head]))

    def normalize(self, residual: torch.Tensor) -> torch.Tensor:
        return pm.exact_layer_norm(residual.double(), self.ln_w, self.ln_b, self.eps)

    def value(self, residual: torch.Tensor) -> torch.Tensor:
        return self.normalize(residual) @ self.W_V + self.b_V

    def scale(self, residual: torch.Tensor) -> float:
        centered = residual.double() - residual.double().mean()
        return float(torch.sqrt((centered * centered).mean() + self.eps))

    def read_direction(self, axis_direction: torch.Tensor) -> torch.Tensor:
        """m = W_V W_O d̂_T in normalized coordinates (the direction of the LayerNorm output the head reads for the number axis)."""
        return (self.W_V @ (self.W_O @ axis_direction.double())).contiguous()


def reconstruct_head_result(weights: HeadWeights, attention_row: torch.Tensor, residuals: Sequence[torch.Tensor]) -> torch.Tensor:
    """Σ_k A_k v_k W_O over the key positions (float64)."""
    total = torch.zeros(weights.W_O.shape[1], dtype=torch.float64)
    for k, residual in enumerate(residuals):
        total = total + float(attention_row[k]) * (weights.value(residual) @ weights.W_O)
    return total


def ov_levels(weights: HeadWeights, *, p_c: int, attention_ref: torch.Tensor, attention_patch: torch.Tensor, residuals_ref: Sequence[torch.Tensor],
              residuals_patch: Sequence[torch.Tensor], delta_head: torch.Tensor) -> dict[str, Any]:
    """The frozen identity ΔT = Σ A^ref Δv W_O + Σ ΔA v^patch W_O and the nested cue-position predictors P1, P2, P3 with the remainder."""
    A_ref = attention_ref.double()
    A_patch = attention_patch.double()
    v_ref = [weights.value(r) for r in residuals_ref]
    v_patch = [weights.value(r) for r in residuals_patch]
    r_c_ref, r_c_patch = residuals_ref[p_c].double(), residuals_patch[p_c].double()
    delta_r = r_c_patch - r_c_ref
    sigma_ref = weights.scale(r_c_ref)
    # P1: fixed-normalization linear read-out (centering is linear; the denominator stays at the reference scale).
    delta_nu_linear = weights.ln_w * (delta_r - delta_r.mean()) / sigma_ref
    delta_T1 = float(A_ref[p_c]) * (delta_nu_linear @ weights.W_V @ weights.W_O)
    # P2: exact LayerNorm at the cue position, reference attention.
    delta_T2 = float(A_ref[p_c]) * ((weights.normalize(r_c_patch) - weights.normalize(r_c_ref)) @ weights.W_V @ weights.W_O)
    # P3: exact LayerNorm and the patched attention pattern, cue-position values only.
    delta_T3 = float(A_patch[p_c]) * (v_patch[p_c] @ weights.W_O) - float(A_ref[p_c]) * (v_ref[p_c] @ weights.W_O)
    for k in range(len(residuals_ref)):
        if k != p_c:
            delta_T3 = delta_T3 + float(A_patch[k] - A_ref[k]) * (v_ref[k] @ weights.W_O)
    # Remainder: value changes at the other positions, Σ_{k≠c} A^patch_k Δv_k W_O.
    remainder = torch.zeros_like(delta_T3)
    for k in range(len(residuals_ref)):
        if k != p_c:
            remainder = remainder + float(A_patch[k]) * ((v_patch[k] - v_ref[k]) @ weights.W_O)
    identity = torch.zeros_like(delta_T3)
    for k in range(len(residuals_ref)):
        identity = identity + float(A_ref[k]) * ((v_patch[k] - v_ref[k]) @ weights.W_O) + float(A_patch[k] - A_ref[k]) * (v_patch[k] @ weights.W_O)
    return {"delta_T1": delta_T1, "delta_T2": delta_T2, "delta_T3": delta_T3, "remainder": remainder, "identity": identity, "measured": delta_head.double(),
            "identity_error": float((identity - delta_head.double()).abs().max()), "decomposition_error": float((delta_T3 + remainder - delta_head.double()).abs().max()),
            "sigma_ref": sigma_ref, "sigma_patch": weights.scale(r_c_patch), "attention_ref": float(A_ref[p_c]), "attention_patch": float(A_patch[p_c])}


# ---------------------------------------------------------------------------
# Confirmation set: tokenizer-only construction under the frozen grammatical-compatibility policy.


def _vowel_initial(noun: pm.Noun) -> bool:
    return noun.lexical_key[:1].lower() in VOWELS


def licensed_frames(word: str, frames: Sequence[pm.Frame]) -> list[str]:
    """`an` is licensed only in cue-final frames; every other fresh cue in every fresh frame."""
    if word == "an":
        return [frame.frame_id for frame in frames if frame.template_id in pm.CUE_FINAL_TEMPLATES]
    return [frame.frame_id for frame in frames]


def licensed_noun_keys(word: str, nouns: Sequence[pm.Noun]) -> list[str]:
    single = [noun for noun in nouns if noun.single_token]
    if word == "an":
        return [noun.lexical_key for noun in single if _vowel_initial(noun)]
    return [noun.lexical_key for noun in single]


def fresh_tokens_009(tokenizer: Any, excluded_ids: set[int], nouns: Sequence[pm.Noun]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen = set(excluded_ids)
    vowel_nouns = sum(1 for noun in nouns if noun.single_token and _vowel_initial(noun))
    for category, words in CANDIDATES.items():
        count = 0
        for word in words:
            if word == "an" and vowel_nouns < AN_MIN_VOWEL_NOUNS:
                continue  # the frozen policy drops `an` when too few vowel-initial nouns exist
            ids = pm._encode(tokenizer, " " + word)
            if len(ids) != 1 or ids[0] in seen:
                continue
            seen.add(ids[0])
            chosen.append({"word": word, "token_id": ids[0], "category": category, "expectation": EXPECTATIONS[category]})
            count += 1
            if count == QUOTAS[category]:
                break
    return chosen


def build_confirmation_payload(tokenizer: Any, manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension, confirmation_006: cd.Confirmation) -> dict[str, Any]:
    pool = cs.build_pool(manifest, extension, confirmation_006)
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_009(tokenizer, excluded_ids, pool.nouns)
    exposed_texts = {frame.text_template for frame in pool.frames}
    cue_ids_by_template = {frame.template_id: dict(frame.cue_ids) for frame in pool.frames if pool.frame_origin[frame.frame_id] == "manifest"}
    frames: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for template_id, text_template in FRESH_FRAMES:
        if text_template in exposed_texts:
            raise ValueError(f"fresh frame text {text_template!r} is an exposed frame")
        counters[template_id] = counters.get(template_id, 0) + 1
        frame = pm._build_new_frame(tokenizer, template_id, text_template, cue_ids_by_template[template_id], f"{template_id}-009-{counters[template_id]}")
        frames.append({"frame_id": frame.frame_id, "template_id": template_id, "text_template": text_template, "prefix_ids": list(frame.prefix_ids), "suffix_ids": list(frame.suffix_ids),
                       "cue_ids": dict(frame.cue_ids), "p_c": frame.p_c, "p_t": frame.p_t,
                       "prompts": {label: {"text": pm._decode(tokenizer, frame.prompt_ids(frame.cue_ids[label])), "token_ids": list(frame.prompt_ids(frame.cue_ids[label]))} for label in ("sg", "pl")}})
    frame_objects = [pm.Frame(e["template_id"], e["frame_id"], tuple(e["prefix_ids"]), tuple(e["suffix_ids"]), e["cue_ids"], e["text_template"], origin="extension") for e in frames]
    for token in tokens:
        token["licensed_frames"] = licensed_frames(token["word"], frame_objects)
        token["licensed_noun_keys"] = licensed_noun_keys(token["word"], pool.nouns)
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
    payload = {
        "schema_version": CONFIRMATION_SCHEMA_VERSION, "manifest_sha256": manifest_sha256, "extension_sha256": extension.content_sha256, "confirmation_006_sha256": confirmation_006.content_sha256,
        "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "reference_cue_ids": dict(pool.reference_ids),
        "exposed_token_ids": sorted(token_id for _, token_id in pool.tokens), "noun_keys": [noun.lexical_key for noun in pool.nouns],
        "policy": {"an_min_vowel_nouns": AN_MIN_VOWEL_NOUNS, "an_frames": "cue-final only", "an_nouns": "vowel-initial singular", "quotas": dict(QUOTAS), "expectations": dict(EXPECTATIONS)},
        "tokens": tokens, "frames": frames, "token_prompts": token_prompts,
        "construction": "tokenizer-only under the frozen grammatical-compatibility policy; first eligible candidates per category; no model output",
    }
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


@dataclass(frozen=True)
class Confirmation009:
    manifest_sha256: str
    extension_sha256: str
    confirmation_006_sha256: str
    reference_ids: Mapping[str, int]
    frames: tuple[pm.Frame, ...]
    tokens: tuple[dict[str, Any], ...]  # word, token_id, category, expectation, licensed_frames, licensed_noun_keys
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

    def frame(self, frame_id: str) -> pm.Frame:
        return next(frame for frame in self.frames if frame.frame_id == frame_id)


def validate_confirmation(payload: Mapping[str, Any], manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension, confirmation_006: cd.Confirmation) -> Confirmation009:
    pm._require_exact_keys(payload, {"schema_version", "manifest_sha256", "extension_sha256", "confirmation_006_sha256", "model", "reference_cue_ids", "exposed_token_ids", "noun_keys", "policy",
                                     "tokens", "frames", "token_prompts", "construction", "content_sha256"}, "confirmation")
    if payload["schema_version"] != CONFIRMATION_SCHEMA_VERSION:
        raise ValueError("confirmation schema version is not frozen")
    if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("confirmation content_sha256 does not match canonical payload")
    if (payload["manifest_sha256"], payload["extension_sha256"], payload["confirmation_006_sha256"]) != (manifest_sha256, extension.content_sha256, confirmation_006.content_sha256):
        raise ValueError("confirmation was built against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}:
        raise ValueError("confirmation was built for a different pinned model")
    pool = cs.build_pool(manifest, extension, confirmation_006)
    if dict(payload["reference_cue_ids"]) != dict(pool.reference_ids) or list(payload["exposed_token_ids"]) != sorted(token_id for _, token_id in pool.tokens):
        raise ValueError("reference cues or exposed token IDs disagree with the exposed pool")
    if list(payload["noun_keys"]) != [noun.lexical_key for noun in pool.nouns]:
        raise ValueError("noun keys disagree with the exposed pool")
    exposed_texts = {frame.text_template for frame in pool.frames}
    frames = tuple(pm._parse_frame(entry, "extension") for entry in payload["frames"])
    expected, counters = [], {}
    for template_id, text_template in FRESH_FRAMES:
        counters[template_id] = counters.get(template_id, 0) + 1
        expected.append((f"{template_id}-009-{counters[template_id]}", template_id, text_template))
    if [(frame.frame_id, frame.template_id, frame.text_template) for frame in frames] != expected:
        raise ValueError("fresh frames are not the frozen literal frames")
    cue_ids_by_template = {frame.template_id: dict(frame.cue_ids) for frame in pool.frames if pool.frame_origin[frame.frame_id] == "manifest"}
    for frame in frames:
        if frame.text_template in exposed_texts or dict(frame.cue_ids) != cue_ids_by_template[frame.template_id]:
            raise ValueError(f"{frame.frame_id}: fresh frame text exposed or original cue tokens missing")
    exposed_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = []
    vowel_nouns = sum(1 for noun in pool.nouns if noun.single_token and _vowel_initial(noun))
    for entry in payload["tokens"]:
        pm._require_exact_keys(entry, {"word", "token_id", "category", "expectation", "licensed_frames", "licensed_noun_keys"}, "token")
        if entry["token_id"] in exposed_ids or entry["word"] not in CANDIDATES[entry["category"]] or entry["expectation"] != EXPECTATIONS[entry["category"]]:
            raise ValueError(f"fresh token {entry['word']} violates the frozen candidate lists")
        if entry["word"] == "an" and vowel_nouns < AN_MIN_VOWEL_NOUNS:
            raise ValueError("`an` is not licensed with fewer than ten vowel-initial nouns")
        if list(entry["licensed_frames"]) != licensed_frames(entry["word"], frames) or list(entry["licensed_noun_keys"]) != licensed_noun_keys(entry["word"], pool.nouns):
            raise ValueError(f"fresh token {entry['word']}: licensed frames or nouns disagree with the frozen policy")
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
    return Confirmation009(payload["manifest_sha256"], payload["extension_sha256"], payload["confirmation_006_sha256"], dict(payload["reference_cue_ids"]), frames, tuple(tokens), tuple(prompts), payload["content_sha256"])


def freeze_confirmation(path: Path, tokenizer: Any, manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension, confirmation_006: cd.Confirmation) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; the confirmation set is frozen and cannot be rebuilt")
    payload = build_confirmation_payload(tokenizer, manifest, manifest_sha256, extension, confirmation_006)
    validate_confirmation(payload, manifest, manifest_sha256, extension, confirmation_006)
    validate_json_safe(payload, path="confirmation")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    return payload["content_sha256"]


def load_confirmation(path: Path, manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension, confirmation_006: cd.Confirmation) -> Confirmation009:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, manifest, manifest_sha256, extension, confirmation_006)


# ---------------------------------------------------------------------------
# Results state (four phases) — the same conventions as Experiments 006–008.

_STATE_KEYS = {"schema_version", "run_id", "created_at", "manifest_sha256", "extension_sha256", "confirmation_sha256", "protocol_code_commit", "git_dirty",
               "model", "versions", "phases", "executed_prompt_keys", "executed_noun_keys", "exploration", "lock", "confirmation", "invalidated_runs", "state_sha256"}


def new_results_state(*, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str, protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {"schema_version": RESULTS_SCHEMA_VERSION, "run_id": pm.sha256_text(manifest_sha256 + extension_sha256 + confirmation_sha256 + protocol_code_commit + pm.utc_now())[:16],
            "created_at": pm.utc_now(), "manifest_sha256": manifest_sha256, "extension_sha256": extension_sha256, "confirmation_sha256": confirmation_sha256,
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
    status = {name: entry["status"] for name, entry in state["phases"].items()}
    if phase == "explore":
        if status["explore"] == "running" and not state["exploration"].get("summary"):
            return
        if status["explore"] != "not_started":
            raise PhaseError("explore already ran; exploration is never re-run in one protocol version")
    elif phase == "lock":
        if status["explore"] != "complete":
            raise PhaseError("lock requires the completed explore phase")
        if status["lock"] == "complete":
            raise PhaseError("lock already written; a new candidate lock requires a new protocol version")
    elif phase == "confirm":
        if status["lock"] != "complete":
            raise PhaseError("confirm requires the lock phase")
        if status["confirm"] != "not_started":
            raise PhaseError("confirm already ran; a second scientific attempt requires a new protocol version")
    elif phase == "report":
        if status["explore"] != "complete" and not state["exploration"].get("incidents"):
            raise PhaseError("report requires the completed explore phase (or a recorded incident)")
    else:
        raise PhaseError(f"unknown phase {phase}")


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation009) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


# ---------------------------------------------------------------------------
# Measurements: three patches per (token, frame) with the head's internals captured.


@dataclass
class FrameReference:
    frame: pm.Frame
    reference: pm.Prompt
    vectors: dict[str, torch.Tensor]  # stage vectors (float64)
    residuals: list[torch.Tensor]  # layer-3 input residual at every key position ≤ p_t (float64)
    attention: torch.Tensor  # head row from p_t over keys (float64)
    head: torch.Tensor  # captured head result at p_t (float64)
    c_by_noun: dict[str, float]
    reconstruction_error: float


@dataclass
class PatchRun:
    delta: dict[str, torch.Tensor]  # stage deltas (float64)
    residuals: list[torch.Tensor]
    attention: torch.Tensor
    head_delta: torch.Tensor
    shifts: dict[str, float]  # per single-token noun


@dataclass
class TokenFrame:
    token: str
    token_id: int
    frame_id: str
    template: str
    full: PatchRun
    par: PatchRun
    perp: PatchRun
    ov: dict[str, Any]
    dc_beh: float


def _capture_sites(model: Any, frame: pm.Frame) -> list[pm.Site]:
    sites = cs._sites(model, frame)
    return [sites[stage] for stage in cs.VECTOR_STAGES] + [sites["A"]] + [(f"RESID_PRE.L{HEAD_LAYER}", k) for k in range(frame.p_t + 1)]


def capture_reference(model: Any, head: HeadWeights, cache: pm.PromptCache, reference: pm.Prompt, nouns: Sequence[pm.Noun]) -> FrameReference:
    frame = reference.frame
    sites = cs._sites(model, frame)
    run = pm.capture_prompt(model, reference, _capture_sites(model, frame))
    vectors = {stage: run.vector(sites[stage]).double() for stage in cs.VECTOR_STAGES}
    residuals = [run.vector((f"RESID_PRE.L{HEAD_LAYER}", k)).double() for k in range(frame.p_t + 1)]
    attention = run.vector(sites["A"])[HEAD_INDEX].double()
    head_result = vectors["T"]
    reconstructed = reconstruct_head_result(head, attention, residuals)
    scale = max(float(head_result.norm()), 1e-12)
    error = float((reconstructed - head_result).norm()) / scale
    if error > RECONSTRUCTION_TOLERANCE:
        raise pm.IncidentError(f"{frame.frame_id}: Σ_k A_k v_k W_O does not reproduce the captured {HEAD_KEY} result (relative error {error:.2e})")
    c_by_noun = pm.contrasts(run.logits, nouns)
    return FrameReference(frame, reference, vectors, residuals, attention, head_result, c_by_noun, error)


def _patch_run(model: Any, ref: FrameReference, replacement: torch.Tensor, nouns: Sequence[pm.Noun]) -> PatchRun:
    frame = ref.frame
    sites = cs._sites(model, frame)
    e_site = ("L00.MLP", frame.p_c)
    run = pm.run_patched(model, ref.reference, {e_site: replacement.reshape(1, 1, -1).to(torch.float32)}, {e_site: ReplacementSource.RESAMPLE}, capture_sites=_capture_sites(model, frame))
    delta = {stage: run.vector(sites[stage]).double() - ref.vectors[stage] for stage in cs.VECTOR_STAGES}
    residuals = [run.vector((f"RESID_PRE.L{HEAD_LAYER}", k)).double() for k in range(frame.p_t + 1)]
    attention = run.vector(sites["A"])[HEAD_INDEX].double()
    c_patched = pm.contrasts(run.logits, nouns)
    shifts = {noun.lexical_key: c_patched[noun.lexical_key] - ref.c_by_noun[noun.lexical_key] for noun in nouns if noun.single_token}
    return PatchRun(delta, residuals, attention, delta["T"], shifts)


def _zero_run(ref: FrameReference, nouns: Sequence[pm.Noun]) -> PatchRun:
    return PatchRun({stage: torch.zeros_like(vector) for stage, vector in ref.vectors.items()}, list(ref.residuals), ref.attention.clone(), torch.zeros_like(ref.head),
                    {noun.lexical_key: 0.0 for noun in nouns if noun.single_token})


def measure_token(model: Any, weights: pm.Weights, head: HeadWeights, cache: pm.PromptCache, ref: FrameReference, name: str, token_id: int, e_axis: pm.SiteAxis,
                  nouns: Sequence[pm.Noun], *, token_prompt: pm.Prompt | None) -> TokenFrame:
    frame = ref.frame
    e_ref = pm.lexicon_vector(weights, ref.reference.cue_token_id).double()
    e_w = pm.lexicon_vector(weights, token_id).double()
    delta_e = e_w - e_ref
    direction = e_axis.direction.double()
    parallel = (delta_e @ direction) * direction
    if token_id == ref.reference.cue_token_id:
        full = par = perp = _zero_run(ref, nouns)
        ov = ov_levels(head, p_c=frame.p_c, attention_ref=ref.attention, attention_patch=ref.attention, residuals_ref=ref.residuals, residuals_patch=ref.residuals, delta_head=torch.zeros_like(ref.head))
        dc_beh = 0.0
    else:
        full = _patch_run(model, ref, e_w, nouns)
        identity_error = float((full.delta["R0"] - delta_e).abs().max())
        if identity_error > cs.IDENTITY_TOLERANCE:
            raise pm.IncidentError(f"{frame.frame_id}/{name}: the residual after layer 0 does not change by ΔE_T(w) (max {identity_error:.2e})")
        par = _patch_run(model, ref, e_ref + parallel, nouns)
        perp = _patch_run(model, ref, e_ref + (delta_e - parallel), nouns)
        ov = ov_levels(head, p_c=frame.p_c, attention_ref=ref.attention, attention_patch=full.attention, residuals_ref=ref.residuals, residuals_patch=full.residuals, delta_head=full.head_delta)
        prompt = token_prompt if token_prompt is not None else pm.Prompt(frame, token_id, name)
        c_word = cache.c(prompt)
        dc_beh = pm._mean([c_word[noun.lexical_key] - ref.c_by_noun[noun.lexical_key] for noun in nouns if noun.single_token])
    return TokenFrame(name, token_id, frame.frame_id, frame.template_id, full, par, perp, ov, dc_beh)


# ---------------------------------------------------------------------------
# Fractions, additivity stages, OV level fractions.


def _projection(delta: torch.Tensor, axis: pm.SiteAxis) -> float:
    return float(delta.double() @ axis.direction.double())


def head_fraction(delta_head: torch.Tensor, plural_delta_head: torch.Tensor, axis: pm.SiteAxis) -> float | None:
    return cs._ratio(_projection(delta_head, axis), _projection(plural_delta_head, axis), cs.STAGE_UNINFORMATIVE_FLOOR * axis.sigma)


def frame_analysis_009(record: TokenFrame, plural: TokenFrame, axes: Mapping[str, pm.SiteAxis], noun_keys: Sequence[str]) -> dict[str, Any]:
    """q_T, stage fractions for the three patches, additivity gaps and the non-additivity stage, the OV level fractions."""
    axis_T = axes["T"]
    denominator_T = _projection(plural.full.head_delta, axis_T)
    dc_pl = pm._mean([plural.full.shifts[key] for key in noun_keys])

    def fraction(stage: str, run: PatchRun) -> float | None:
        axis = axes[stage]
        return cs._ratio(_projection(run.delta[stage], axis), _projection(plural.full.delta[stage], axis), cs.STAGE_UNINFORMATIVE_FLOOR * axis.sigma)

    def contrast(run: PatchRun) -> float | None:
        return cs._ratio(pm._mean([run.shifts[key] for key in noun_keys]), dc_pl, cs.CONTRAST_UNINFORMATIVE_NATS)

    fractions = {name: {stage: fraction(stage, run) for stage in cs.VECTOR_STAGES} | {"c": contrast(run)} for name, run in (("full", record.full), ("par", record.par), ("perp", record.perp))}
    s_r0 = fractions["full"]["R0"]
    orientation = cs._sign(s_r0) if s_r0 is not None else 1.0
    gaps = {}
    for stage in ADDITIVITY_STAGES:
        values = [fractions[name][stage] for name in ("full", "par", "perp")]
        gaps[stage] = None if None in values else orientation * (values[0] - values[1] - values[2])
    non_additivity = next((stage for stage in ADDITIVITY_STAGES if gaps[stage] is not None and abs(gaps[stage]) > G_MAX), "NONE")
    q_T = fractions["full"]["T"]
    levels = {f"P{j}": cs._ratio(_projection(record.ov[f"delta_T{j}"], axis_T), denominator_T, cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma) for j in LEVELS}
    levels["remainder"] = cs._ratio(_projection(record.ov["remainder"], axis_T), denominator_T, cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma)
    return {"fractions": fractions, "q_T": q_T, "q_R1": fractions["full"]["R1"], "s_R0": s_r0, "orientation": orientation, "gaps": gaps, "non_additivity_stage": non_additivity,
            "levels": levels, "dc": pm._mean([record.full.shifts[key] for key in noun_keys]), "dc_pl": dc_pl, "dc_beh": record.dc_beh,
            "attention_ratio": cs._ratio(record.ov["attention_patch"], record.ov["attention_ref"], 1e-9), "sigma_ratio": record.ov["sigma_patch"] / record.ov["sigma_ref"],
            "identity_error": record.ov["identity_error"], "decomposition_error": record.ov["decomposition_error"]}


def check_ov_identities(analysis: Mapping[str, Any], plural_head_norm: float, where: str) -> None:
    scale = max(plural_head_norm, 1e-12)
    for key in ("identity_error", "decomposition_error"):
        if analysis[key] / scale > OV_IDENTITY_TOLERANCE:
            raise pm.IncidentError(f"{where}: the OV {key.replace('_', ' ')} is {analysis[key] / scale:.2e} relative to the plural cue's head change")


def modal_stage(stages: Sequence[str]) -> tuple[str, float]:
    """Mode over frames of the non-additivity stage; ties go to the earliest stage (NONE last); returns the agreeing fraction."""
    counts = Counter(stages)
    order = list(ADDITIVITY_STAGES) + ["NONE"]
    best = min(counts, key=lambda stage: (-counts[stage], order.index(stage) if stage in order else len(order)))
    return best, counts[best] / len(stages)


def level_table(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Exposed MAE and R² of each level's q̂ against the measured q_T over the (token, frame) pairs, and the locked level."""
    out: dict[str, Any] = {"levels": {}}
    measured_all = [pair["q_T"] for pair in pairs if pair["q_T"] is not None]
    mean_q = pm._mean(measured_all) if measured_all else 0.0
    total = sum((q - mean_q) ** 2 for q in measured_all)
    for j in LEVELS:
        rows = [(pair["q_T"], pair["levels"][f"P{j}"]) for pair in pairs if pair["q_T"] is not None and pair["levels"][f"P{j}"] is not None]
        errors = [abs(q - q_hat) for q, q_hat in rows]
        residual = sum((q - q_hat) ** 2 for q, q_hat in rows)
        out["levels"][f"P{j}"] = {"mae": pm._mean(errors) if errors else None, "rmse": math.sqrt(pm._mean([e * e for e in errors])) if errors else None,
                                  "r2": (1.0 - residual / total) if total > 0 and rows else None, "n": len(rows)}
    locked = next((j for j in LEVELS if out["levels"][f"P{j}"]["r2"] is not None and out["levels"][f"P{j}"]["r2"] >= MECHANISM_R2_FLOOR and out["levels"][f"P{j}"]["mae"] <= MECHANISM_MAE_CEILING), None)
    out["locked_level"] = locked
    out["tau_M"] = max(TAU_MIN, TAU_MULTIPLIER * out["levels"][f"P{locked}"]["rmse"]) if locked else None
    out["meaning"] = LEVEL_MEANING.get(locked) if locked else "no cue-position level explains the head's output change; the remainder's share is reported"
    remainder = [abs(pair["levels"]["remainder"]) for pair in pairs if pair["levels"]["remainder"] is not None]
    out["mean_abs_remainder"] = pm._mean(remainder) if remainder else None
    return out


# ---------------------------------------------------------------------------
# The transport rule (rank-1 cross-moment direction for a scalar response) and the ridge-scalar baseline.


@dataclass(frozen=True)
class TransportRule:
    v: torch.Tensor  # unit direction [d_model] float64
    beta: Mapping[str, float]  # template -> slope
    fit_tokens: tuple[str, ...]

    def predict(self, template: str, delta_e: torch.Tensor) -> float:
        return self.beta[template] * float(delta_e.double() @ self.v)

    def digest(self) -> str:
        return pm.sha256_text(pm.canonical_json({"v": [float(x) for x in self.v], "beta": {k: float(v) for k, v in self.beta.items()}}))


def fit_transport_rule(rows: Sequence[tuple[str, str, torch.Tensor, float]], tokens: Sequence[str]) -> TransportRule:
    """rows: (token, template, ΔE_T(w), q_T); v ∝ Σ x_i y_i over pooled rows; β_T by no-intercept least squares per template."""
    accumulator = None
    for token, _, x, y in rows:
        if token in tokens:
            accumulator = x.double() * y if accumulator is None else accumulator + x.double() * y
    if accumulator is None or float(accumulator.norm()) <= 0.0:
        raise pm.IncidentError("the transport rule's cross-moment direction is zero")
    v = accumulator / accumulator.norm()
    beta = {}
    for template in pm.TEMPLATE_ORDER:
        z = [float(x.double() @ v) for token, t, x, _ in rows if token in tokens and t == template]
        y = [q for token, t, _, q in rows if token in tokens and t == template]
        denominator = sum(value * value for value in z)
        beta[template] = (sum(a * b for a, b in zip(z, y)) / denominator) if denominator > 0 else 0.0
    return TransportRule(v.contiguous(), beta, tuple(tokens))


def transport_rows(pool: cs.Pool008, e_vectors: Mapping[str, torch.Tensor], reference_vectors: Mapping[str, torch.Tensor], q_values: Mapping[tuple[str, str], float | None], frames: Sequence[pm.Frame]) -> list[tuple[str, str, torch.Tensor, float]]:
    rows = []
    for name, _ in pool.tokens:
        for frame in frames:
            q = q_values.get((name, frame.frame_id))
            if q is not None:
                rows.append((name, frame.template_id, e_vectors[name].double() - reference_vectors[frame.template_id].double(), q))
    return rows


def transport_loco(rows: Sequence[tuple[str, str, torch.Tensor, float]], tokens: Sequence[str]) -> dict[str, dict[str, float]]:
    table = {}
    for held_out in tokens:
        rule = fit_transport_rule(rows, [token for token in tokens if token != held_out])
        predicted = [rule.predict(template, x) for token, template, x, _ in rows if token == held_out]
        measured = [q for token, _, _, q in rows if token == held_out]
        table[held_out] = {"predicted": pm._mean(predicted), "measured": pm._mean(measured), "mae": pm._mean([abs(p - m) for p, m in zip(predicted, measured)])}
    return table


def _ridge_scalar_maps(rows: Sequence[tuple[str, str, torch.Tensor, float]], tokens: Sequence[str], lam: float) -> dict[str, torch.Tensor]:
    maps = {}
    for template in pm.TEMPLATE_ORDER:
        X = torch.stack([x.double() for token, t, x, _ in rows if token in tokens and t == template])
        y = torch.tensor([q for token, t, _, q in rows if token in tokens and t == template], dtype=torch.float64)
        maps[template] = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], dtype=torch.float64), X.T @ y)
    return maps


def _trace_scale_rows(rows: Sequence[tuple[str, str, torch.Tensor, float]], tokens: Sequence[str]) -> float:
    X = torch.stack([x.double() for token, _, x, _ in rows if token in tokens])
    return float((X * X).sum() / X.shape[1])


def ridge_scalar_loco(rows: Sequence[tuple[str, str, torch.Tensor, float]], tokens: Sequence[str], *, multipliers: Sequence[float] = ss.RIDGE_MULTIPLIERS) -> dict[str, Any]:
    """Template-specific ridge from ΔE to q_T with Experiment 007's nested cue-group multiplier selection (grid scale from inner-training rows only)."""
    table, chosen = {}, {}
    for held_out in tokens:
        training = [token for token in tokens if token != held_out]
        scores = {multiplier: 0.0 for multiplier in multipliers}
        for inner in training:
            inner_training = [token for token in training if token != inner]
            scale = _trace_scale_rows(rows, inner_training)
            for multiplier in multipliers:
                maps = _ridge_scalar_maps(rows, inner_training, multiplier * scale)
                errors = [abs(float(x.double() @ maps[template]) - q) for token, template, x, q in rows if token == inner]
                scores[multiplier] += pm._mean(errors) / len(training)
        multiplier = ss.choose_multiplier(scores)
        maps = _ridge_scalar_maps(rows, training, multiplier * _trace_scale_rows(rows, training))
        predicted = [float(x.double() @ maps[template]) for token, template, x, _ in rows if token == held_out]
        measured = [q for token, _, _, q in rows if token == held_out]
        table[held_out] = {"predicted": pm._mean(predicted), "measured": pm._mean(measured), "mae": pm._mean([abs(p - m) for p, m in zip(predicted, measured)])}
        chosen[held_out] = multiplier
    return {"table": table, "multipliers": chosen}


def tau_q(per_token: Mapping[str, Mapping[str, float]]) -> float:
    errors = [entry["mae"] for entry in per_token.values()]
    return max(TAU_MIN, TAU_MULTIPLIER * math.sqrt(pm._mean([e * e for e in errors])))


def gate_scalar(per_token: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Experiment 007's quality gate for a single rule (no rank comparison)."""
    error = pm._mean([entry["mae"] for entry in per_token.values()])
    return cd.quality_gate(per_token, selected=1, summary={1: {"error": error, "se": 0.0}})


# ---------------------------------------------------------------------------
# Exploration (Tier A) on the exposed pool.


def _responses_from(records: Mapping[tuple[str, str], TokenFrame]) -> dict[tuple[str, str], cd.EPatchResponse]:
    return {(name, frame_id): cd.EPatchResponse(name, record.token_id, frame_id, record.template, record.full.delta["R3"].float(), dict(record.full.shifts), 0.0, ())
            for (name, frame_id), record in records.items()}


def contrast_rule_loco(ev: ss.Evaluation, tokens: Sequence[str], *, reference_ids: Mapping[str, int], ranks: Sequence[int] = ss.RANKS) -> dict[int, dict[str, dict[str, float]]]:
    table: dict[int, dict[str, dict[str, float]]] = {rank: {} for rank in ranks}
    for held_out in tokens:
        training = [token for token in tokens if token != held_out]
        rows = ss.design_rows(e_vectors=ev.e_vectors, reference_vectors=ev.reference_vectors, responses=ev.responses, frames=ev.frames, tokens=training)
        svd = ss.cross_moment_svd(rows.X, rows.Y, ranks=ranks, where=f"contrast fold {held_out}")
        for rank in ranks:
            fit = ss.fit_supervised(rows, rank, reference_ids=reference_ids, svd=svd, where=f"contrast fold {held_out}, rank {rank}")
            table[rank][held_out] = ss.cue_level_values(fit, ev, held_out)
    return table


def run_exploration(model: Any, pool: cs.Pool008, *, state: dict[str, Any], results_path: Path | None, parameters_dir: Path, program_source: Path, program_007: Any | None,
                    inherited_006: Mapping[str, Any], inherited_007: Mapping[str, Any], ranks: Sequence[int] | None = None, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    ranks = tuple(ranks) if ranks is not None else tuple(ss.RANKS)
    weights = pm.Weights.from_model(model)
    head = HeadWeights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    nouns = pool.single_nouns
    noun_keys = [noun.lexical_key for noun in nouns]
    exploration = state["exploration"]
    say("stage axes from the eighteen frames' clean cue pairs")
    axes = cs.stage_axes(cache, weights, pool)
    e_axis = axes["R0"]
    exploration["axes"] = {stage: {"sigma": axis.sigma, "cos_with_E_axis": pm.cosine(axis.direction, e_axis.direction)} for stage, axis in axes.items()}
    exploration["head"] = {"key": HEAD_KEY, "read_direction_norm": float(head.read_direction(axes["T"].direction).norm()), "d_head": int(head.W_V.shape[1]),
                           "read_direction": head.read_direction(axes["T"].direction).tolist()}
    exploration["axes_vectors"] = {"T": axes["T"].direction.double().tolist(), "R0": e_axis.direction.double().tolist()}
    say("measurements: 40 tokens × 18 frames × 3 patches with the head's internals")
    records: dict[tuple[str, str], TokenFrame] = {}
    references: dict[str, FrameReference] = {}
    for frame in pool.frames:
        reference = pool.reference_prompt(frame)
        ref = capture_reference(model, head, cache, reference, nouns)
        references[frame.frame_id] = ref
        for name, token_id in pool.tokens:
            records[(name, frame.frame_id)] = measure_token(model, weights, head, cache, ref, name, token_id, e_axis, nouns, token_prompt=pool.token_prompt(frame, name))
        say(f"  frame {frame.frame_id}: reconstruction error {ref.reconstruction_error:.2e}")
    exploration["reconstruction"] = {frame_id: ref.reconstruction_error for frame_id, ref in references.items()}
    # Replication on the noun subsets of Experiments 006 and 007.
    exposed_keys = [noun.lexical_key for noun in pool.single_nouns_from("exposed-60")]
    confirmation_keys = [noun.lexical_key for noun in pool.single_nouns_from("confirmation-20")]
    exploration["replication"] = {
        "experiment_006": cs.check_replication({f"{n}|{f}": pm._mean([r.full.shifts[k] for k in exposed_keys]) for (n, f), r in records.items()}, inherited_006["responses"], label="Experiment 006 exposed responses"),
        "experiment_007": cs.check_replication({f"{n}|{f}": pm._mean([r.full.shifts[k] for k in confirmation_keys]) for (n, f), r in records.items()}, inherited_007["responses"], label="Experiment 007 confirmation responses"),
    }
    say(f"replication: 006 {exploration['replication']['experiment_006']['max_abs_deviation']:.2e}; 007 {exploration['replication']['experiment_007']['max_abs_deviation']:.2e}")
    if results_path is not None:
        write_results_state(results_path, state)
    # Per-(token, frame) analysis and the OV identities.
    per_frame: dict[str, dict[str, dict[str, Any]]] = {name: {} for name, _ in pool.tokens}
    q_values: dict[tuple[str, str], float | None] = {}
    for frame in pool.frames:
        plural = records[(pool.plural_cue[frame.template_id], frame.frame_id)]
        plural_norm = float(plural.full.head_delta.norm())
        for name, _ in pool.tokens:
            analysis = frame_analysis_009(records[(name, frame.frame_id)], plural, axes, noun_keys)
            check_ov_identities(analysis, plural_norm, f"{frame.frame_id}/{name}")
            per_frame[name][frame.frame_id] = analysis
            q_values[(name, frame.frame_id)] = analysis["q_T"]
    pairs = [analysis for frames in per_frame.values() for analysis in frames.values()]
    exploration["mechanism"] = level_table(pairs)
    say(f"mechanism level: {exploration['mechanism']['locked_level']} ({exploration['mechanism']['levels']})")
    # Token-level summaries.
    tokens_out = {}
    for name, token_id in pool.tokens:
        frames = per_frame[name]
        modal, agreement = modal_stage([analysis["non_additivity_stage"] for analysis in frames.values()])
        tokens_out[name] = {"token": name, "token_id": token_id, "category": pool.token_category[name], "source": pool.token_source[name],
                            "q_T": cs._mean_or_none([a["q_T"] for a in frames.values()]), "q_R1": cs._mean_or_none([a["q_R1"] for a in frames.values()]), "s_R0": cs._mean_or_none([a["s_R0"] for a in frames.values()]),
                            "levels": {key: cs._mean_or_none([a["levels"][key] for a in frames.values()]) for key in ("P1", "P2", "P3", "remainder")},
                            "gaps": {stage: cs._mean_or_none([a["gaps"][stage] for a in frames.values()]) for stage in ADDITIVITY_STAGES},
                            "non_additivity_stage": modal, "non_additivity_agreement": agreement,
                            "attention_ratio": cs._mean_or_none([a["attention_ratio"] for a in frames.values()]), "sigma_ratio": pm._mean([a["sigma_ratio"] for a in frames.values()]),
                            "dc": pm._mean([a["dc"] for a in frames.values()]), "dc_beh": pm._mean([a["dc_beh"] for a in frames.values()])}
    exploration["tokens"] = tokens_out
    exploration["per_frame"] = per_frame
    # Rules fitted on the forty tokens.
    e_vectors, reference_vectors = cd._token_vectors(weights, pool.tokens, pool.reference_ids)
    token_names = [name for name, _ in pool.tokens]
    say("transport rule (rank-1 cross-moment direction) with leave-one-cue-out")
    rows = transport_rows(pool, e_vectors, reference_vectors, q_values, pool.frames)
    transport_table = transport_loco(rows, token_names)
    transport_gate = gate_scalar(transport_table)
    transport_rule = fit_transport_rule(rows, token_names)
    ridge = ridge_scalar_loco(rows, token_names)
    exploration["transport_rule"] = {"loco": transport_table, "gate": transport_gate, "tau_2": tau_q(transport_table), "beta": dict(transport_rule.beta), "digest": transport_rule.digest(),
                                     "v_cos_with_E_axis": pm.cosine(transport_rule.v, e_axis.direction), "ridge_scalar": {"loco": ridge["table"], "multipliers": ridge["multipliers"], "error": pm._mean([e["mae"] for e in ridge["table"].values()])},
                                     "error": pm._mean([e["mae"] for e in transport_table.values()])}
    say(f"transport rule gate {'passed' if transport_gate['passed'] else 'FAILED'} (Spearman {transport_gate['spearman']:.3f}, nRMSE {transport_gate['normalized_rmse']:.3f}); ridge error {exploration['transport_rule']['ridge_scalar']['error']:.3f}")
    say("contrast rule (Experiment 007 family on forty tokens) with leave-one-cue-out")
    contexts = ss.context_residuals(cache, pool.frames, pool.reference_ids)
    responses = _responses_from(records)
    ev = ss.make_evaluation(weights, contexts, e_vectors, reference_vectors, responses, pool.frames, pool.nouns)
    contrast_table = contrast_rule_loco(ev, token_names, reference_ids=pool.reference_ids, ranks=ranks)
    selection = cd.select_rank(contrast_table)
    selected = int(selection["selected"])
    contrast_gate = cd.quality_gate(contrast_table[selected], selected=selected, summary=selection["summary"])
    tau_3 = cd.tolerance_tau(contrast_table[selected])
    final_rows = ss.design_rows(e_vectors=e_vectors, reference_vectors=reference_vectors, responses=responses, frames=pool.frames, tokens=token_names)
    final_fit = ss.fit_supervised(final_rows, selected, reference_ids=pool.reference_ids, where="contrast final fit")
    parameters_dir = Path(parameters_dir)
    contrast_index = ss.export_program(parameters_dir / "contrast", weights, final_fit, contexts)
    torch.save(transport_rule.v.clone(), parameters_dir / "transport_v.pt")
    exploration["contrast_rule"] = {"loco": {str(rank): per_token for rank, per_token in contrast_table.items()}, "selection": selection, "gate": contrast_gate, "tau_3": tau_3,
                                    "parameters_sha256": pm.sha256_text(pm.canonical_json(contrast_index)), "final_fit": ss._json_safe(final_fit.diagnostics)}
    say(f"contrast rule: rank {selected}, gate {'passed' if contrast_gate['passed'] else 'FAILED'} (Spearman {contrast_gate['spearman']:.3f}, nRMSE {contrast_gate['normalized_rmse']:.3f}), tau {tau_3:.3f}")
    # Baseline: the Experiment 007 sixteen-token program on the exposed pool (descriptive).
    if program_007 is not None:
        errors = []
        for name, token_id in pool.tokens:
            for frame in pool.frames:
                predicted = pm._mean([program_007.predict_epatch_shift(frame.template_id, frame.frame_id, token_id, noun.sg_ids[0], noun.pl_ids[0]) for noun in nouns])
                errors.append(abs(predicted - per_frame[name][frame.frame_id]["dc"]))
        exploration["program_007_exposed_mae"] = pm._mean(errors)
    exploration["summary"] = {"locked_level": exploration["mechanism"]["locked_level"], "transport_gate": transport_gate["passed"], "contrast_gate": contrast_gate["passed"], "contrast_rank": selected,
                              "suppressed_non_additivity": {name: tokens_out[name]["non_additivity_stage"] for name in ("a", "this", "another") if name in tokens_out}}
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


# ---------------------------------------------------------------------------
# Lock, the prediction artifact, and its validation.

def load_transport_rule(parameters_dir: Path, exploration: Mapping[str, Any]) -> TransportRule:
    v = torch.load(Path(parameters_dir) / "transport_v.pt", map_location="cpu", weights_only=True).double()
    rule = TransportRule(v, dict(exploration["transport_rule"]["beta"]), tuple())
    if rule.digest() != exploration["transport_rule"]["digest"]:
        raise PhaseError("the transport rule on disk does not match the recorded digest")
    return rule


def lock_predictions(*, weights: pm.Weights, confirmation: Confirmation009, transport: TransportRule | None, contrast_program: Any | None, program_007: Any | None,
                     nouns_by_key: Mapping[str, pm.Noun]) -> dict[str, Any]:
    """Every fresh (token, licensed frame) prediction, computed from weights and exported rules without running any fresh prompt."""
    tokens = {}
    for token in confirmation.tokens:
        word, token_id = token["word"], token["token_id"]
        entry: dict[str, Any] = {"token_id": token_id, "category": token["category"], "expectation": token["expectation"], "frames": {}}
        for frame in confirmation.frames_for(word):
            template = frame.template_id
            delta_e = pm.lexicon_vector(weights, token_id).double() - pm.lexicon_vector(weights, confirmation.reference_ids[template]).double()
            frame_entry: dict[str, Any] = {"template_id": template}
            if transport is not None:
                frame_entry["q_T"] = transport.predict(template, delta_e)
            for name, program in (("contrast", contrast_program), ("program_007", program_007)):
                if program is None:
                    continue
                by_noun = {key: program.predict_epatch_shift(template, None, token_id, nouns_by_key[key].sg_ids[0], nouns_by_key[key].pl_ids[0]) for key in token["licensed_noun_keys"]}
                frame_entry[name] = {"mean": pm._mean(list(by_noun.values())), "by_noun": by_noun}
            entry["frames"][frame.frame_id] = frame_entry
        if transport is not None:
            entry["q_T_mean"] = pm._mean([f["q_T"] for f in entry["frames"].values()])
        if contrast_program is not None:
            entry["contrast_mean"] = pm._mean([f["contrast"]["mean"] for f in entry["frames"].values()])
        tokens[word] = entry
    return {"tokens": tokens}


def render_predictions(lock: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 009 — preregistered predictions", "", f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_sha256']}`",
             f"- Mechanism level locked: {lock['mechanism']['locked_level']} (τ_M {f(lock['mechanism']['tau_M'], 3)}); transport rule digest `{(lock['transport_rule'] or {}).get('digest', '—')}` (τ₂ {f((lock['transport_rule'] or {}).get('tau_2'), 3)}); "
             f"contrast rule parameters `{(lock['contrast_rule'] or {}).get('parameters_sha256', '—')}` (rank {(lock['contrast_rule'] or {}).get('rank', '—')}, τ₃ {f((lock['contrast_rule'] or {}).get('tau_3'), 3)})", "",
             "| token | category | expectation | frame | predicted q_T | predicted contrast shift | 007 program |", "|---|---|---|---|---|---|---|"]
    for word, entry in lock["predictions"]["tokens"].items():
        for frame_id, frame in entry["frames"].items():
            lines.append(f"| {word} | {entry['category']} | {entry['expectation']} | {frame_id} | {f(frame.get('q_T'), 3)} | {f((frame.get('contrast') or {}).get('mean'), 3)} | {f((frame.get('program_007') or {}).get('mean'), 3)} |")
        lines.append(f"| **{word}** | | | **mean** | **{f(entry.get('q_T_mean'), 3)}** | **{f(entry.get('contrast_mean'), 3)}** | |")
    lines.append("")
    return "\n".join(lines)


def build_candidate_lock(*, state: Mapping[str, Any], manifest_sha256: str, extension: pm.Extension, confirmation: Confirmation009, predictions: Mapping[str, Any], parameters_dir: Path,
                         program_source: Path, protocol_code_commit: str, axes_T_direction: Sequence[float], read_direction: Sequence[float], transport_v: Sequence[float] | None = None) -> dict[str, Any]:
    exploration = state["exploration"]
    mechanism = exploration["mechanism"]
    transport = exploration["transport_rule"] if exploration["transport_rule"]["gate"]["passed"] else None
    if transport is not None and transport_v is None:
        raise PhaseError("the lock must carry the transport rule's direction v")
    contrast = exploration["contrast_rule"] if exploration["contrast_rule"]["gate"]["passed"] else None
    lock = {
        "schema_version": 1, "experiment": "009", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit,
        "manifest_sha256": manifest_sha256, "extension_sha256": extension.content_sha256, "confirmation_sha256": confirmation.content_sha256,
        "confirmation_prompt_keys": sorted(prompt.key for prompt in confirmation.all_prompts),
        "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED},
        "head": HEAD_KEY, "axes": exploration["axes"], "d_T": list(axes_T_direction), "read_direction": list(read_direction),
        "mechanism": {"locked_level": mechanism["locked_level"], "levels": mechanism["levels"], "tau_M": mechanism["tau_M"], "meaning": mechanism["meaning"], "floors": {"r2": MECHANISM_R2_FLOOR, "mae": MECHANISM_MAE_CEILING, "y1_spearman": Y1_SPEARMAN}},
        "transport_rule": ({"digest": transport["digest"], "v": list(transport_v), "beta": transport["beta"], "tau_2": transport["tau_2"], "gate": transport["gate"], "loco_error": transport["error"]} if transport else None),
        "contrast_rule": ({"parameters_sha256": contrast["parameters_sha256"], "rank": contrast["selection"]["selected"], "tau_3": contrast["tau_3"], "gate": contrast["gate"], "selection": contrast["selection"]} if contrast else None),
        "program_source_sha256": pm.sha256_text(Path(program_source).read_text(encoding="utf-8")),
        "category_floors": {"singular_max_q": SINGULAR_MAX_Q, "plural_min_q": PLURAL_MIN_Q, "plural_rate": PLURAL_RATE, "y2_spearman": Y2_SPEARMAN},
        "contrast_floors": {"y_spearman": cd.Y1_SPEARMAN, "confident_nats": cd.Y1_CONFIDENT_NATS, "sign_frames": cd.Y1_SIGN_FRAMES, "cue_effect_fresh": cd.CUE_EFFECT_FRESH},
        "predictions": dict(predictions), "tier_a_results_sha256": state.get("state_sha256"),
    }
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], manifest_sha256: str, extension: pm.Extension, confirmation: Confirmation009, predictions_text: str,
                  git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "009" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked or git_state.get("dirty", True):
        raise PhaseError("confirm requires the committed lock and predictions on a clean Git tree")
    if (lock["manifest_sha256"], lock["extension_sha256"], lock["confirmation_sha256"]) != (manifest_sha256, extension.content_sha256, confirmation.content_sha256):
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
        for frame_id, frame in entry["frames"].items():
            other = fresh["frames"][frame_id]
            if "q_T" in frame:
                worst = max(worst, abs(frame["q_T"] - other["q_T"]))
            for name in ("contrast", "program_007"):
                if name in frame:
                    worst = max(worst, abs(frame[name]["mean"] - other[name]["mean"]), *[abs(frame[name]["by_noun"][k] - other[name]["by_noun"][k]) for k in frame[name]["by_noun"]])
    if worst > LOCK_PREDICTION_TOLERANCE:
        raise PhaseError(f"the on-disk rules do not reproduce the preregistered predictions (max difference {worst:.3e}); nothing was executed")
    return worst


# ---------------------------------------------------------------------------
# Confirmation (Tier C), floors, outcome, report.


def run_confirmation(model: Any, pool: cs.Pool008, confirmation: Confirmation009, lock: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = HeadWeights.from_model(model)
    nouns = pool.single_nouns
    nouns_by_key = {noun.lexical_key: noun for noun in nouns}
    cache = pm.PromptCache(model, tuple(pool.nouns))
    # The stage axes are re-estimated on the exposed frames (exposed prompts only); the T axis must equal the locked direction.
    say("stage axes (exposed frames)")
    axes = cs.stage_axes(cache, weights, pool)
    if pm.cosine(axes["T"].direction, torch.tensor(lock["d_T"], dtype=torch.float64)) < 1.0 - 1e-6:
        raise pm.IncidentError("the T axis differs from the locked direction")
    e_axis = axes["R0"]
    results: dict[str, Any] = {"frames": {}, "tokens": {}}
    positive = total = 0
    pairs: list[dict[str, Any]] = []
    say("fresh frames: reference, plural cue, original cue pairs")
    per_frame: dict[str, dict[str, dict[str, Any]]] = {token["word"]: {} for token in confirmation.tokens}
    for frame in confirmation.frames:
        reference = confirmation.reference_prompt(frame)
        ref = capture_reference(model, head, cache, reference, nouns)
        sg, pl = pm.frame_prompts(frame)
        c_a, c_b = cache.c(sg), cache.c(pl)
        for noun in nouns:
            total += 1
            positive += int(c_a[noun.lexical_key] - c_b[noun.lexical_key] > 0)
        plural_name = pool.plural_cue[frame.template_id]
        plural = measure_token(model, weights, head, cache, ref, plural_name, pl.cue_token_id, e_axis, nouns, token_prompt=pl)
        plural_analysis = frame_analysis_009(plural, plural, axes, [noun.lexical_key for noun in nouns])
        check_ov_identities(plural_analysis, float(plural.full.head_delta.norm()), f"{frame.frame_id}/{plural_name}")
        results["frames"][frame.frame_id] = {"reconstruction_error": ref.reconstruction_error, "plural_head_norm": float(plural.full.head_delta.norm()), "dc_pl": pm._mean(list(plural.full.shifts.values())),
                                             "plural_levels": plural_analysis["levels"]}
        for token in confirmation.tokens:
            if frame.frame_id not in token["licensed_frames"]:
                continue
            record = measure_token(model, weights, head, cache, ref, token["word"], token["token_id"], e_axis, nouns, token_prompt=pm.Prompt(frame, token["token_id"], token["word"]))
            analysis = frame_analysis_009(record, plural, axes, token["licensed_noun_keys"])
            check_ov_identities(analysis, float(plural.full.head_delta.norm()), f"{frame.frame_id}/{token['word']}")
            analysis["shifts"] = {key: record.full.shifts[key] for key in token["licensed_noun_keys"]}
            per_frame[token["word"]][frame.frame_id] = analysis
            pairs.append({"word": token["word"], "frame_id": frame.frame_id, **{k: analysis[k] for k in ("q_T", "levels")}})
        say(f"  {frame.frame_id}: {sum(1 for token in confirmation.tokens if frame.frame_id in token['licensed_frames'])} tokens")
    required = pm.exact_count_floor(cd.CUE_EFFECT_FRESH, total)
    results["cue_effect"] = {"positive_pairs": positive, "pairs": total, "required": required, "passed": positive >= required}
    # Y1 — mechanism at the locked level (no fitted parameter); every level's fresh statistics are reported, with the level that would have passed.
    level = lock["mechanism"]["locked_level"]
    fresh_levels = {}
    for j in LEVELS:
        rows = [(pair["q_T"], pair["levels"][f"P{j}"]) for pair in pairs if pair["q_T"] is not None and pair["levels"][f"P{j}"] is not None]
        if not rows:
            fresh_levels[f"P{j}"] = {"n_pairs": 0, "spearman": None, "mae": None, "tau": None, "would_pass": False}
            continue
        predicted, measured = [r[1] for r in rows], [r[0] for r in rows]
        mae = pm._mean([abs(p - m) for p, m in zip(predicted, measured)])
        spearman = pm.spearman(predicted, measured)
        exposed_rmse = lock["mechanism"]["levels"][f"P{j}"]["rmse"]
        tau_j = max(TAU_MIN, TAU_MULTIPLIER * exposed_rmse) if exposed_rmse is not None else None
        fresh_levels[f"P{j}"] = {"n_pairs": len(rows), "spearman": spearman, "mae": mae, "tau": tau_j, "would_pass": tau_j is not None and spearman >= Y1_SPEARMAN and mae <= tau_j}
    would_pass_at = next((j for j in LEVELS if fresh_levels[f"P{j}"]["would_pass"]), None)
    if level is not None:
        entry = fresh_levels[f"P{level}"]
        results["Y1"] = {"level": level, "n_pairs": entry["n_pairs"], "spearman": entry["spearman"], "mae": entry["mae"], "tau_M": lock["mechanism"]["tau_M"],
                         "passed": entry["spearman"] is not None and entry["spearman"] >= Y1_SPEARMAN and entry["mae"] <= lock["mechanism"]["tau_M"], "levels_fresh": fresh_levels, "would_pass_at": would_pass_at}
    else:
        results["Y1"] = {"level": None, "passed": None, "levels_fresh": fresh_levels, "would_pass_at": would_pass_at}
    # Token means.
    tokens_out = {}
    for token in confirmation.tokens:
        word = token["word"]
        frames = per_frame[word]
        locked = lock["predictions"]["tokens"][word]
        informative = [frame_id for frame_id, a in frames.items() if a["q_T"] is not None]
        stage, agreement = modal_stage([a["non_additivity_stage"] for a in frames.values()])
        entry = {"category": token["category"], "expectation": token["expectation"], "q_T": cs._mean_or_none([a["q_T"] for a in frames.values()]),
                 # the predicted mean is taken over the same informative frames as the measured mean
                 "q_T_predicted": (pm._mean([locked["frames"][frame_id]["q_T"] for frame_id in informative]) if informative and "q_T" in next(iter(locked["frames"].values())) else locked.get("q_T_mean")),
                 "q_T_predicted_all_frames": locked.get("q_T_mean"),
                 "dc": pm._mean([a["dc"] for a in frames.values()]), "dc_beh": pm._mean([a["dc_beh"] for a in frames.values()]),
                 "contrast_predicted": locked.get("contrast_mean"), "s_R0": cs._mean_or_none([a["s_R0"] for a in frames.values()]), "q_R1": cs._mean_or_none([a["q_R1"] for a in frames.values()]),
                 "attention_ratio": cs._mean_or_none([a["attention_ratio"] for a in frames.values()]),
                 "non_additivity_stage": stage, "non_additivity_agreement": agreement, "levels": {key: cs._mean_or_none([a["levels"][key] for a in frames.values()]) for key in ("P1", "P2", "P3", "remainder")}}
        if "program_007" in next(iter(locked["frames"].values())):
            entry["program_007_predicted"] = pm._mean([f["program_007"]["mean"] for f in locked["frames"].values()])
        tokens_out[word] = entry
    results["tokens"] = tokens_out
    results["per_frame"] = per_frame
    # Y2 — transport rule.
    if lock["transport_rule"] is not None:
        words = [w for w in tokens_out if tokens_out[w]["q_T"] is not None]
        predicted = [tokens_out[w]["q_T_predicted"] for w in words]
        measured = [tokens_out[w]["q_T"] for w in words]
        mae = pm._mean([abs(p - m) for p, m in zip(predicted, measured)])
        spearman = pm.spearman(predicted, measured)
        singular = [w for w in words if tokens_out[w]["expectation"] == "low"]
        plural = [w for w in words if tokens_out[w]["expectation"] == "high"]
        singular_measured_ok = sum(1 for w in singular if tokens_out[w]["q_T"] <= SINGULAR_MAX_Q) >= max(len(singular) - 1, 0)
        singular_predicted_ok = sum(1 for w in singular if tokens_out[w]["q_T_predicted"] <= SINGULAR_MAX_Q) >= max(len(singular) - 1, 0)
        plural_required = pm.exact_count_floor(PLURAL_RATE, len(plural))
        plural_measured_ok = sum(1 for w in plural if tokens_out[w]["q_T"] >= PLURAL_MIN_Q) >= plural_required
        plural_predicted_ok = sum(1 for w in plural if tokens_out[w]["q_T_predicted"] >= PLURAL_MIN_Q) >= plural_required
        categories_ok = singular_measured_ok and singular_predicted_ok and plural_measured_ok and plural_predicted_ok
        failing = [name for name, ok in (("spearman", spearman >= Y2_SPEARMAN), ("mae", mae <= lock["transport_rule"]["tau_2"]), ("singular_measured", singular_measured_ok),
                                        ("singular_predicted", singular_predicted_ok), ("plural_measured", plural_measured_ok), ("plural_predicted", plural_predicted_ok)) if not ok]
        results["Y2"] = {"spearman": spearman, "mae": mae, "tau_2": lock["transport_rule"]["tau_2"], "categories_ok": categories_ok, "failing": failing, "passed": not failing,
                         "singular": singular, "plural": plural, "plural_required": plural_required}
    else:
        results["Y2"] = {"passed": None}
    # Y3 — contrast rule (Experiment 007's Y1 floors) and the 007 program baseline.
    if lock["contrast_rule"] is not None:
        per_token = {}
        for token in confirmation.tokens:
            word = token["word"]
            locked = lock["predictions"]["tokens"][word]
            frames_entry = {}
            for frame_id, analysis in per_frame[word].items():
                predicted_by_noun = locked["frames"][frame_id]["contrast"]["by_noun"]
                frames_entry[frame_id] = {"template_id": confirmation.frame(frame_id).template_id, "epatch_measured": analysis["dc"], "behavior_measured": analysis["dc_beh"],
                                          "predicted:contrast": locked["frames"][frame_id]["contrast"]["mean"],
                                          "epatch_mae:contrast": pm._mean([abs(predicted_by_noun[k] - analysis["shifts"][k]) for k in predicted_by_noun]),
                                          "behavior_mae:contrast": abs(locked["frames"][frame_id]["contrast"]["mean"] - analysis["dc_beh"])}
                if "program_007" in locked["frames"][frame_id]:
                    frames_entry[frame_id]["predicted:program_007"] = locked["frames"][frame_id]["program_007"]["mean"]
                    frames_entry[frame_id]["epatch_mae:program_007"] = pm._mean([abs(locked["frames"][frame_id]["program_007"]["by_noun"][k] - analysis["shifts"][k]) for k in predicted_by_noun])
                    frames_entry[frame_id]["behavior_mae:program_007"] = abs(locked["frames"][frame_id]["program_007"]["mean"] - analysis["dc_beh"])
            per_token[word] = {"token_id": token["token_id"], "category": token["category"], "frames": frames_entry}
        y = cd.y_summary(per_token, "contrast", "epatch_measured", "epatch_mae:contrast")
        signs_ok = all(entry["agree"] >= pm.exact_count_floor(cd.Y1_SIGN_FRAMES / 6, entry["total"]) for entry in y["confident"].values())
        tau_3 = lock["contrast_rule"]["tau_3"]
        results["Y3"] = {"spearman": y["spearman"], "mae": y["mae"], "tau_3": tau_3, "signs_ok": signs_ok, "confident": y["confident"], "passed": y["spearman"] >= cd.Y1_SPEARMAN and y["mae"] <= tau_3 and signs_ok}
        if any("predicted:program_007" in f for entry in per_token.values() for f in entry["frames"].values()):
            baseline = cd.y_summary(per_token, "program_007", "epatch_measured", "epatch_mae:program_007")
            results["program_007_fresh"] = {"spearman": baseline["spearman"], "mae": baseline["mae"]}
        results["contrast_per_token"] = per_token
    else:
        results["Y3"] = {"passed": None}
    results["outcome"] = outcome(results)
    return results


def outcome(results: Mapping[str, Any]) -> dict[str, Any]:
    if not results["cue_effect"]["passed"]:
        return {"label": "CUE_EFFECT_NOT_REPLICATED", "mechanism": None, "transport": None, "contrast": None}
    y1, y2, y3 = results["Y1"], results["Y2"], results["Y3"]
    mechanism = "NOT_LOCKED" if y1["passed"] is None else (f"HEAD_MECHANISM_CONFIRMED_P{y1['level']}" if y1["passed"] else "HEAD_MECHANISM_NOT_SUPPORTED")
    transport = "NOT_LOCKED" if y2["passed"] is None else ("TRANSPORT_RULE_PREDICTED" if y2["passed"] else "TRANSPORT_RULE_FAILED")
    contrast = "NOT_LOCKED" if y3["passed"] is None else ("CONTRAST_RULE_PREDICTED" if y3["passed"] else "CONTRAST_RULE_FAILED")
    return {"label": f"{mechanism} | {transport} | {contrast}", "mechanism": mechanism, "transport": transport, "contrast": contrast}


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 009 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state['confirmation_sha256']}`", f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    incidents = list(exploration.get("incidents", [])) + ([state["confirmation"]["incident"]] if isinstance(state.get("confirmation"), dict) and "incident" in state["confirmation"] else [])
    if incidents:
        lines += ["## Incidents", ""] + [f"- `{e['phase']}` at commit `{e.get('commit', '?')}` ({e['at']}): {e['message']}" for e in incidents] + [""]
    if "mechanism" in exploration:
        m = exploration["mechanism"]
        rep = exploration["replication"]
        lines += ["## Tier A — exposed pool", "", f"- Replication: 006 max deviation {rep['experiment_006']['max_abs_deviation']:.2e}; 007 {rep['experiment_007']['max_abs_deviation']:.2e}; head reconstruction max error {max(exploration['reconstruction'].values()):.2e}",
                  f"- OV levels over the exposed pairs: " + "; ".join(f"{k}: MAE {f(v['mae'], 3)}, R² {f(v['r2'], 3)}" for k, v in m["levels"].items()) + f"; mean |remainder| {f(m['mean_abs_remainder'], 3)}",
                  f"- **Locked mechanism level: {m['locked_level']}** — {m['meaning']}; τ_M {f(m['tau_M'], 3)}",
                  f"- Transport rule: LOCO error {f(exploration['transport_rule']['error'], 3)}, gate {'passed' if exploration['transport_rule']['gate']['passed'] else 'FAILED'} (Spearman {f(exploration['transport_rule']['gate']['spearman'], 3)}, nRMSE {f(exploration['transport_rule']['gate']['normalized_rmse'], 3)}), τ₂ {f(exploration['transport_rule']['tau_2'], 3)}; ridge-scalar LOCO error {f(exploration['transport_rule']['ridge_scalar']['error'], 3)}; cos(v, d̂_E) {f(exploration['transport_rule']['v_cos_with_E_axis'], 3)}",
                  f"- Contrast rule: rank {exploration['contrast_rule']['selection']['selected']}, gate {'passed' if exploration['contrast_rule']['gate']['passed'] else 'FAILED'} (Spearman {f(exploration['contrast_rule']['gate']['spearman'], 3)}, nRMSE {f(exploration['contrast_rule']['gate']['normalized_rmse'], 3)}), τ₃ {f(exploration['contrast_rule']['tau_3'], 3)}; 007 program exposed MAE {f(exploration.get('program_007_exposed_mae'), 3)}", "",
                  "| token | category | s_R0 | q_R1 | q_T | P1 | P2 | P3 | rem | â | σ ratio | g_R1 | g_T | g_R2 | g_R3 | g_c | ν* |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in sorted(exploration["tokens"].values(), key=lambda r: (r["q_T"] if r["q_T"] is not None else 9)):
            g = row["gaps"]
            lines.append(f"| {row['token']} | {row['category']} | {f(row['s_R0'])} | {f(row['q_R1'])} | {f(row['q_T'])} | {f(row['levels']['P1'])} | {f(row['levels']['P2'])} | {f(row['levels']['P3'])} | {f(row['levels']['remainder'])} | "
                         f"{f(row['attention_ratio'])} | {f(row['sigma_ratio'])} | {f(g['R1'])} | {f(g['T'])} | {f(g['R2'])} | {f(g['R3'])} | {f(g['c'])} | {row['non_additivity_stage']} ({f(row['non_additivity_agreement'])}) |")
        lines.append("")
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}`; predictions sha256 `{state['lock']['predictions_sha256']}`", ""]
    confirmation = state.get("confirmation")
    if confirmation and "outcome" in confirmation:
        o = confirmation["outcome"]
        ce = confirmation["cue_effect"]
        lines += [f"## Confirmation — `{o['label']}`", "", f"- Fresh-frame cue effect {ce['positive_pairs']}/{ce['pairs']} (floor {ce['required']})"]
        y1, y2, y3 = confirmation["Y1"], confirmation["Y2"], confirmation["Y3"]
        if y1.get("level") is not None:
            lines.append(f"- Y1 (mechanism P{y1['level']}): Spearman {f(y1['spearman'], 3)}, MAE {f(y1['mae'], 3)} (τ_M {f(y1['tau_M'], 3)}) → {'pass' if y1['passed'] else 'FAIL'}; fresh levels " + "; ".join(f"{k}: Spearman {f(v['spearman'], 3)}, MAE {f(v['mae'], 3)} (τ {f(v['tau'], 3)})" for k, v in y1["levels_fresh"].items()) + f"; would pass at {y1['would_pass_at']}")
        if y2.get("passed") is not None:
            lines.append(f"- Y2 (transport rule): Spearman {f(y2['spearman'], 3)}, MAE {f(y2['mae'], 3)} (τ₂ {f(y2['tau_2'], 3)}), categories ok {y2['categories_ok']} → {'pass' if y2['passed'] else 'FAIL'} {y2['failing'] or ''}")
        if y3.get("passed") is not None:
            lines.append(f"- Y3 (contrast rule): Spearman {f(y3['spearman'], 3)}, MAE {f(y3['mae'], 3)} (τ₃ {f(y3['tau_3'], 3)}), signs ok {y3['signs_ok']} → {'pass' if y3['passed'] else 'FAIL'}" + (f"; 007 program fresh Spearman {f(confirmation['program_007_fresh']['spearman'], 3)}, MAE {f(confirmation['program_007_fresh']['mae'], 3)}" if "program_007_fresh" in confirmation else ""))
        lines += ["", "| token | category | expectation | q_T measured | q_T predicted | s_R0 | q_R1 | P1 | P2 | P3 | â | contrast measured | contrast predicted | 007 | behavior | ν* |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for word, row in confirmation["tokens"].items():
            lines.append(f"| {word} | {row['category']} | {row['expectation']} | {f(row['q_T'])} | {f(row['q_T_predicted'])} | {f(row['s_R0'])} | {f(row['q_R1'])} | {f(row['levels']['P1'])} | {f(row['levels']['P2'])} | {f(row['levels']['P3'])} | {f(row['attention_ratio'])} | "
                         f"{f(row['dc'])} | {f(row['contrast_predicted'])} | {f(row.get('program_007_predicted'))} | {f(row['dc_beh'])} | {row['non_additivity_stage']} |")
        lines.append("")
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
