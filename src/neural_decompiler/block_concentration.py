"""Experiment 018: is block 2's operating-point dependence — Experiment 017's channel D — carried by a fixed subset of its neurons?

Experiment 017 wrote the chain from a cue word's weight-only encoding change ``ΔE`` to the transport head's output as a
composed weight-only program; its least compact object is channel D, block 2's MLP at the frame's own operating point
(2048 reference pre-activations per changed position through the exact GELU with the predicted arriving change). This
module replaces channel D by its masked form ``D_S``: the neurons of a subset ``S`` at the frame's operating point,
every other neuron at the template base (Experiment 012's base at ``p_c``; a block-2 base at ``p_t`` for the coordinated
template locked at ``explore``). The subset is chosen once by a frozen read-unit rule on the exposed pool's licensed
pairs from Level 0-F's *predicted* arriving change — no measured quantity — and locked; the ladder ``S_0 … S_2048``,
three seeded random subsets and the 256 lowest-ranked neurons are the controls; ``S_2048`` is Experiment 017's Level 0
exactly (I8) and ``S_0`` at ``p_c`` its ``−D`` rung. The statistic is the closure fraction ``κ`` (unclipped; undefined
below a 0.05 gap). Every rung receives ``ΔE``, the weights, the locked axes, read, bases and subsets and the frame's
reference run only. Constants are copied from design revision 2.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import attention_paths as ap
from . import attention_patterns as atp
from . import cue_decompilation as cd
from . import cue_suppression as cs
from . import encoding_read as er
from . import frame_channels as fch
from . import head_pattern as hp
from . import head_transport as ht
from . import layer_correction as lc
from . import neuron_feature as nf
from . import plural_mechanism as pm
from . import read_assembly as ra
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 2).

EXPERIMENT_DIR = "experiments/018-block2-concentration"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
INHERITED_017_EXTRACT_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-017-pair-extract.json"
EXPERIMENT_017_LOCK_PATH = hp.LOCK_RELATIVE_PATH
EXPERIMENT_017_CONFIRMATION_PATH = hp.CONFIRMATION_RELATIVE_PATH
RUNTIME_SEED = hp.RUNTIME_SEED
CONTROL_SEED = hp.CONTROL_SEED
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
INHERITED_SCHEMA_VERSION = 1
PHASES = ("explore", "lock", "confirm", "report")

N_NEURONS = 2048  # block 2's MLP width (checked against the weights)
SUBSET_SIZES = (0, 1, 4, 16, 64, 256, 1024, 2048)  # the ladder S_n
HYPOTHESIS_SIZE = 256
N_RANDOM_CONTROLS = 3
BOTTOM_SIZE = 256
LADDER_RUNGS = tuple(f"S{n}" for n in SUBSET_SIZES)
RANDOM_RUNGS = tuple(f"R{k}" for k in range(1, N_RANDOM_CONTROLS + 1))
BOTTOM_RUNG = "B256"
RUNGS = (*LADDER_RUNGS, *RANDOM_RUNGS, BOTTOM_RUNG)
HYPOTHESIS, REFERENCE, TEMPLATE_BASE, SINGLE = f"S{HYPOTHESIS_SIZE}", f"S{N_NEURONS}", "S0", "S1"
ORACLE = "oracle"  # the frame's own top-256 (descriptive)
OBJECTS = ("c_L", "F", "Pi", "dT")  # the scalar objects per rung; the row is the fifth
FLOOR_OBJECTS = ("c_L", "Pi")

KAPPA_CL_FLOOR = 0.70  # Y1/Y2: κ_{c_L}(S_256) on the pooled scored pairs of the set
KAPPA_PI_FLOOR = 0.50  # Y1/Y2: κ_Π(S_256)
SPLIT_GUARD = {"cue_final": 0.60, "coordinated": 0.40}  # Y2: κ_{c_L}(S_256) over each template split of the valid fresh frames
NO_HARM_MARGIN = 0.05  # Y2: in every valid fresh frame R²_{c_L}(S_256) ≥ R²_{c_L}(S_0) − margin
GAP_MIN = 0.05  # κ is undefined (not evaluable) below this denominator
PRECONDITION_REFERENCE = {"c_L": 0.98, "rows": 0.95, "dT": 0.95}  # the reference rung S_2048 (Experiment 017's Level 0) on the set
Y3_KAPPA_MAX = 0.50  # the single neuron is rejected iff κ_{c_L}(S_1) on the pooled pairs of both sets is below this …
Y3_MARGIN = 0.25  # … and κ_{c_L}(S_256) exceeds it by at least this
Y4_FROZEN_MAX = 0.90  # per evaluable fresh cue-final frame: the frozen pattern's ΔT R² below this
Y4_REFERENCE_MIN = 0.95  # a frame is evaluable iff the reference rung's ΔT R² over its scored pairs is at least this
Y4_MIN_EVALUABLE = 6
ROW_DIFFUSE_MAX = 0.50  # predeclared expectation (i): κ_row(S_256) below this on both fresh sets
RANDOM_MARGIN = 0.30  # predeclared expectation (ii): κ_{c_L}(S_256) exceeds the best random control's by at least this
MIN_VALID_CUE_FINAL_FRAMES = 6
MIN_VALID_COORDINATED_FRAMES = 3
MIN_VALID_FRAMES_PER_TOKEN = 3
MIN_SCORED_TOKENS = 16
FRAME_CUE_EFFECT_RATE = hp.FRAME_CUE_EFFECT_RATE
I8_TOLERANCE = 1e-9  # S_2048 against Experiment 017's Level 0 and S_0 against its −D rung, in process
LOCK_PREDICTION_TOLERANCE = 1e-9
REPLICATION_TOLERANCE = 1e-6
LOCKED_STATE_TOLERANCE = 1e-9
EXPECTED_EXTRACT_SIZE_017 = 9636  # Experiment 017's 7764 exposed pairs + its 1584 Y1 + 288 Y2 pairs
HEAD_LAYER, HEAD_INDEX, HEAD_KEY = hp.HEAD_LAYER, hp.HEAD_INDEX, hp.HEAD_KEY
BLOCK = 2
PROGRAM_LAYERS = hp.PROGRAM_LAYERS
SCALAR_KEYS = tuple(f"{obj}_{rung}" for rung in (*RUNGS, ORACLE) for obj in OBJECTS) + ("dT_frozen", "c_L_level0F")
ROW_KEYS = tuple(f"row_{rung}" for rung in (*RUNGS, ORACLE))
PREDICTION_COLUMNS = ("token", "frame_id", "template", "p_c", "p_t", *ROW_KEYS, *SCALAR_KEYS)

CANDIDATES = hp.CANDIDATES  # Experiment 017's frozen lists; the first eligible entries not yet exposed are chosen
QUOTAS = hp.QUOTAS
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The harbor shelters {cue}"),
    ("cardinal", "The studio records {cue}"),
    ("cardinal", "The stable houses {cue}"),
    ("cardinal", "The depot receives {cue}"),
    ("quantifier", "The syllabus outlines {cue}"),
    ("quantifier", "The registry tracks {cue}"),
    ("quantifier", "The notice announces {cue}"),
    ("quantifier", "The atlas maps {cue}"),
    ("coordinated-adjective", "Ivan and Rosa carved {cue} smooth"),
    ("coordinated-adjective", "Jonah and Mei rinsed {cue} pale"),
    ("coordinated-adjective", "Ella and Kofi trimmed {cue} rough"),
    ("coordinated-adjective", "Ruth and Dario tasted {cue} sour"),
)
FRAME_ID_TAG = "018"
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", *hp.SCIENTIFIC_PATH_PREFIXES[1:])
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)
OUTCOME_Y1 = ("CHANNEL_D_CONCENTRATED_TOKENS", "CHANNEL_D_NOT_CONCENTRATED_TOKENS", "PRECONDITION_FAILED_TOKENS")
OUTCOME_Y2 = ("CHANNEL_D_CONCENTRATED_FRAMES_CONDITIONAL", "CHANNEL_D_NOT_CONCENTRATED_FRAMES_CONDITIONAL", "PRECONDITION_FAILED_FRAMES")
OUTCOME_Y3 = ("SINGLE_NEURON_REJECTED", "SINGLE_NEURON_NOT_REJECTED", "SINGLE_NEURON_NOT_EVALUABLE")
OUTCOME_Y4 = ("PATTERN_TERM_NEEDED_WITHIN_FRAMES", "PATTERN_TERM_NOT_ESTABLISHED_WITHIN_FRAMES", "PATTERN_TERM_NOT_EVALUABLE_WITHIN_FRAMES")


class PhaseError(pm.PhaseError):
    """Protocol violation in the Experiment 018 phase machinery."""


# ---------------------------------------------------------------------------
# Pool (231 tokens, 78 frames) and the inherited Experiment 017 extract.


def build_pool_018(manifest: ScreeningManifest, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009, confirmation_011: er.Confirmation011,
                   confirmation_012: lc.Confirmation012, confirmation_013: ap.Confirmation013, confirmation_014: nf.Confirmation014, confirmation_015: atp.Confirmation015, confirmation_016: fch.Confirmation016,
                   confirmation_017: hp.Confirmation017) -> cs.Pool008:
    base = hp.build_pool_017(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016)
    frames = base.frames + tuple(confirmation_017.frames)
    origin = dict(base.frame_origin) | {frame.frame_id: "confirmation-017" for frame in confirmation_017.frames}
    tokens = list(base.tokens)
    category, source = dict(base.token_category), dict(base.token_source)
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation_017.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"Experiment 017 token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-017"
    return cs.Pool008(frames, origin, tuple(tokens), category, source, base.nouns, base.noun_source, base.reference_ids, base.plural_cue)


EXTRACT_017_DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014", "confirmation_015", "confirmation_016", "confirmation_017", "lock_017")
EXTRACT_017_DIGEST_FIELDS = tuple(f"{key}_sha256" for key in EXTRACT_017_DIGEST_KEYS)
EXTRACT_SCALARS = ("F", "Pi", "dT", "c_L", "F_hat", "Pi_hat", "dT_hat", "c_L_level0D")
EXTRACT_ROWS = ("row", "row_level0")


def extract_entry_017(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """From an Experiment 017 recorded pair (or, with the same keys, an Experiment 018 analysis's ``analysis_017``): the measured objects and the Level 0 predictions."""
    prediction = analysis["prediction"]
    return {"F": float(analysis["F"]), "Pi": float(analysis["Pi"]), "dT": float(analysis["dT"]), "c_L": float(analysis["c_L"]), "row": [float(v) for v in analysis["row"]],
            "row_level0": [float(v) for v in prediction["row_level0"]], "F_hat": float(prediction["F_hat"]), "Pi_hat": float(prediction["Pi_hat"]), "dT_hat": float(prediction["dT_hat"]), "c_L_level0D": float(prediction["c_L_level0D"])}


def inherited_extract_payload(entries: Mapping[str, Mapping[str, Any]], *, source: Mapping[str, Any], digests: Mapping[str, str], stage1_state_digests: Mapping[str, str]) -> dict[str, Any]:
    payload = {"schema_version": INHERITED_SCHEMA_VERSION, "experiment": "017", "kind": "pair-extract",
               "description": "Derived extract of Experiment 017's recorded per-pair measured F, Π, ΔT, c_L and head-row change and its Level 0 predictions (row, F̂, Π̂, ΔT̂, decoded c_L with channel D) over its 7764 exposed pairs and its 1584 + 288 confirmed pairs, with the digests of its twelve fresh frames' digested stage-1 reference states. Experiment 018 recomputes the quantities and requires agreement within 1e-6, and re-captures the states to their digests.",
               "source": dict(source), **{field: digests[key] for field, key in zip(EXTRACT_017_DIGEST_FIELDS, EXTRACT_017_DIGEST_KEYS)},
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "stage1_state_digests": {key: str(value) for key, value in sorted(stage1_state_digests.items())},
               "entries": {key: dict(value) for key, value in sorted(entries.items())}}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_extract(path: Path, *, digests: Mapping[str, str], expected_size: int) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited 017 extract: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "experiment", "kind", "description", "source", *EXTRACT_017_DIGEST_FIELDS, "model", "stage1_state_digests", "entries", "content_sha256"}, "inherited 017 extract")
    if payload["schema_version"] != INHERITED_SCHEMA_VERSION or payload["experiment"] != "017" or payload["kind"] != "pair-extract" or payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("inherited 017 extract schema or digest is not frozen")
    if tuple(payload[field] for field in EXTRACT_017_DIGEST_FIELDS) != tuple(digests[key] for key in EXTRACT_017_DIGEST_KEYS):
        raise ValueError("inherited 017 extract was recorded against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision} or len(payload["entries"]) != expected_size or len(payload["stage1_state_digests"]) != len(hp.FRESH_FRAMES):
        raise ValueError("inherited 017 extract model, size or stage-1 digests are not frozen")
    return payload


def check_extract_replication(measured: Mapping[str, Mapping[str, Any]], recorded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = set(recorded) - set(measured)
    if missing:
        raise pm.IncidentError(f"{len(missing)} recorded Experiment 017 pairs lack a recomputed quantity (e.g. {sorted(missing)[:3]})")
    worst_key, worst = "", 0.0
    for key, value in recorded.items():
        other = measured[key]
        for name in EXTRACT_SCALARS:
            deviation = abs(value[name] - other[name])
            if deviation > worst:
                worst_key, worst = f"{key}:{name}", deviation
        for name in EXTRACT_ROWS:
            if len(value[name]) != len(other[name]):
                raise pm.IncidentError(f"{key}:{name}: the recomputed row has a different length from Experiment 017's record")
            deviation = max((abs(a - b) for a, b in zip(value[name], other[name])), default=0.0)
            if deviation > worst:
                worst_key, worst = f"{key}:{name}", deviation
    if worst > REPLICATION_TOLERANCE:
        raise pm.IncidentError(f"{worst_key}: recomputed quantity deviates from Experiment 017's record by {worst:.3e}")
    return {"passed": True, "n": len(recorded), "max_abs_deviation": worst, "worst_key": worst_key}


# ---------------------------------------------------------------------------
# The confirmation set (tokenizer rules only).


def fresh_tokens_018(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
    return hp.fresh_tokens_017(tokenizer, excluded_ids)  # the same rule on the same lists; the exposed ids now include Experiment 017's 24


def _expected_frames() -> list[tuple[str, str, str]]:
    expected, counters = [], {}
    for template_id, text_template in FRESH_FRAMES:
        counters[template_id] = counters.get(template_id, 0) + 1
        expected.append((f"{template_id}-{FRAME_ID_TAG}-{counters[template_id]}", template_id, text_template))
    return expected


CONFIRMATION_DIGEST_KEYS = (*hp.CONFIRMATION_DIGEST_KEYS, "confirmation_017")
CONFIRMATION_DIGEST_FIELDS = tuple(f"{key}_sha256" for key in CONFIRMATION_DIGEST_KEYS)
Confirmation018 = ap.Confirmation013


def build_confirmation_payload(tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> dict[str, Any]:
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_018(tokenizer, excluded_ids)
    exposed_texts = {frame.text_template for frame in pool.frames}
    cue_ids_by_template = {frame.template_id: dict(frame.cue_ids) for frame in pool.frames if pool.frame_origin[frame.frame_id] == "manifest"}
    frames: list[dict[str, Any]] = []
    for frame_id, template_id, text_template in _expected_frames():
        if text_template in exposed_texts:
            raise ValueError(f"fresh frame text {text_template!r} is an exposed frame")
        frame = pm._build_new_frame(tokenizer, template_id, text_template, cue_ids_by_template[template_id], frame_id)
        frames.append({"frame_id": frame.frame_id, "template_id": template_id, "text_template": text_template, "prefix_ids": list(frame.prefix_ids), "suffix_ids": list(frame.suffix_ids),
                       "cue_ids": dict(frame.cue_ids), "p_c": frame.p_c, "p_t": frame.p_t,
                       "prompts": {label: {"text": pm._decode(tokenizer, frame.prompt_ids(frame.cue_ids[label])), "token_ids": list(frame.prompt_ids(frame.cue_ids[label]))} for label in ("sg", "pl")}})
    frame_objects = [pm.Frame(e["template_id"], e["frame_id"], tuple(e["prefix_ids"]), tuple(e["suffix_ids"]), e["cue_ids"], e["text_template"], origin="extension") for e in frames]
    for token in tokens:
        token["licensed_frames"] = [frame.frame_id for frame in frame_objects]
    payload = {"schema_version": CONFIRMATION_SCHEMA_VERSION, **{field: digests[key] for field, key in zip(CONFIRMATION_DIGEST_FIELDS, CONFIRMATION_DIGEST_KEYS)},
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "reference_cue_ids": dict(pool.reference_ids),
               "exposed_token_ids": sorted(token_id for _, token_id in pool.tokens), "exposed_frame_ids": [frame.frame_id for frame in pool.frames],
               "policy": {"licensing": "every fresh cue in every fresh frame and in every exposed frame", "quotas": dict(QUOTAS),
                          "expectations": "none: the committed per-rung ĉ_L, F̂, Π̂, ΔT̂ and head-row predictions are the only predictions; no label is preregistered for any neuron, frame or cue"},
               "tokens": tokens, "frames": frames, "token_prompts": [nf._prompt_entry(tokenizer, frame, token) for frame in frame_objects for token in tokens],
               "exposed_frame_prompts": [nf._prompt_entry(tokenizer, frame, token) for frame in pool.frames for token in tokens],
               "construction": "tokenizer-only; first eligible candidates per lexical class of Experiment 017's lists; classes carry no expectation; no model output"}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def validate_confirmation(payload: Mapping[str, Any], pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation018:
    pm._require_exact_keys(payload, {"schema_version", *CONFIRMATION_DIGEST_FIELDS, "model", "reference_cue_ids", "exposed_token_ids", "exposed_frame_ids", "policy", "tokens", "frames", "token_prompts", "exposed_frame_prompts", "construction", "content_sha256"}, "confirmation")
    if payload["schema_version"] != CONFIRMATION_SCHEMA_VERSION:
        raise ValueError("confirmation schema version is not frozen")
    if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("confirmation content_sha256 does not match canonical payload")
    if tuple(payload[field] for field in CONFIRMATION_DIGEST_FIELDS) != tuple(digests[key] for key in CONFIRMATION_DIGEST_KEYS):
        raise ValueError("confirmation was built against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}:
        raise ValueError("confirmation was built for a different pinned model")
    if dict(payload["reference_cue_ids"]) != dict(pool.reference_ids) or list(payload["exposed_token_ids"]) != sorted(token_id for _, token_id in pool.tokens) or list(payload["exposed_frame_ids"]) != [frame.frame_id for frame in pool.frames]:
        raise ValueError("reference cues, exposed token IDs or exposed frames disagree with the exposed pool")
    exposed_texts = {frame.text_template for frame in pool.frames}
    frames = tuple(pm._parse_frame(entry, "extension") for entry in payload["frames"])
    if [(frame.frame_id, frame.template_id, frame.text_template) for frame in frames] != _expected_frames():
        raise ValueError("fresh frames are not the frozen literal frames")
    cue_ids_by_template = {frame.template_id: dict(frame.cue_ids) for frame in pool.frames if pool.frame_origin[frame.frame_id] == "manifest"}
    for frame in frames:
        if frame.text_template in exposed_texts or dict(frame.cue_ids) != cue_ids_by_template[frame.template_id]:
            raise ValueError(f"{frame.frame_id}: fresh frame text exposed or original cue tokens missing")
    exposed_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = []
    all_frame_ids = [frame.frame_id for frame in frames]
    for entry in payload["tokens"]:
        pm._require_exact_keys(entry, {"word", "token_id", "category", "licensed_frames"}, "token")
        if entry["category"] not in CANDIDATES:
            raise ValueError(f"fresh token {entry['word']} has an unknown class {entry['category']}")
        if entry["token_id"] in exposed_ids or entry["word"] not in CANDIDATES[entry["category"]] or list(entry["licensed_frames"]) != all_frame_ids:
            raise ValueError(f"fresh token {entry['word']} violates the frozen candidate lists or the licensing policy")
        tokens.append(dict(entry))
    for category, words in CANDIDATES.items():
        chosen = [entry["word"] for entry in tokens if entry["category"] == category]
        if len(chosen) > QUOTAS[category] or [words.index(word) for word in chosen] != sorted(words.index(word) for word in chosen):
            raise ValueError(f"{category}: quota or candidate order violated")
    if len({entry["token_id"] for entry in tokens}) != len(tokens) or not tokens:
        raise ValueError("fresh tokens must be distinct and non-empty")

    def parse_prompts(entries: Sequence[Mapping[str, Any]], frame_list: Sequence[pm.Frame], label: str) -> list[pm.Prompt]:
        frames_by_id = {frame.frame_id: frame for frame in frame_list}
        prompts = []
        for entry in entries:
            pm._require_exact_keys(entry, {"frame_id", "word", "token_id", "text", "token_ids", "p_c", "p_t"}, label)
            frame = frames_by_id[entry["frame_id"]]
            token = next(token for token in tokens if token["word"] == entry["word"])
            if token["token_id"] != entry["token_id"] or tuple(entry["token_ids"]) != frame.prompt_ids(entry["token_id"]) or (entry["p_c"], entry["p_t"]) != (frame.p_c, frame.p_t):
                raise ValueError(f"{entry['frame_id']}/{entry['word']}: stored prompt disagrees with the frame")
            prompts.append(pm.Prompt(frame, int(entry["token_id"]), entry["word"]))
        expected = [(frame.frame_id, token["word"]) for frame in frame_list for token in tokens]
        if [(prompt.frame.frame_id, prompt.cue_label) for prompt in prompts] != expected:
            raise ValueError(f"{label}: prompts must cover exactly the (frame, token) pairs in order")
        return prompts

    fresh_prompts = parse_prompts(payload["token_prompts"], frames, "token_prompt")
    exposed_prompts = parse_prompts(payload["exposed_frame_prompts"], pool.frames, "exposed_frame_prompt")
    return Confirmation018(dict(payload["reference_cue_ids"]), frames, tuple(payload["exposed_frame_ids"]), tuple(tokens), tuple(fresh_prompts), tuple(exposed_prompts), payload["content_sha256"])


def freeze_confirmation(path: Path, tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; the confirmation set is frozen and cannot be rebuilt")
    payload = build_confirmation_payload(tokenizer, pool, digests)
    validate_confirmation(payload, pool, digests)
    validate_json_safe(payload, path="confirmation")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    return payload["content_sha256"]


def load_confirmation(path: Path, pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation018:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, pool, digests)


# ---------------------------------------------------------------------------
# The masked chain: Experiment 017's chain with channel D replaced by D_S.


@dataclass(frozen=True)
class UpstreamParts:
    """Everything of a pair that does not depend on the subset: per changed position the arriving change plus the layer-2 attention output change (``base``) and block 2's per-neuron hidden deltas at the frame's operating point (``own``) and at the template base (``template``)."""

    parts: Mapping[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]  # position -> (base [512], own [2048], template [2048])
    delta_e: torch.Tensor
    denominator: float
    c_L_level0F: float  # Experiment 016's decoded c_L (block 2 at the template base, no channel D)

    def effects(self, position: int) -> torch.Tensor:
        """The operating-point effect e_j = own_j − tmpl_j of every neuron at a changed position."""
        _, own, template = self.parts[position]
        return own - template


@dataclass(frozen=True)
class MaskedChainModel:
    """Experiment 017's ``HeadChainModel`` with channel D masked: neurons in the mask at the frame's operating point, the others at the template base (the Experiment 012 base at ``p_c``, the locked ``base2_pt`` at ``p_t``)."""

    hcm: hp.HeadChainModel
    base2_pt: torch.Tensor | None  # x̄₂^T(p_t) for the coordinated template (None before explore locks it; an incident if a coordinated pair needs it)
    masks: Mapping[str, torch.Tensor]  # rung name -> 0/1 mask over block 2's neurons

    @property
    def lw(self) -> lc.LayerWeights:
        return self.hcm.fcm.lw

    @property
    def n_neurons(self) -> int:
        return int(self.lw.W_out[BLOCK].shape[0])

    def upstream_parts(self, weights: pm.Weights, rows16: Mapping[int, atp.ReferenceRow], x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], p_c: int, p_t: int, token_id: int, template: str) -> UpstreamParts:
        fcm, lw = self.hcm.fcm, self.lw
        x1, x2 = x1_all[p_c].double(), x2_all[p_c].double()
        up = fcm.predict_channels(weights, rows16, x1, x2, token_id, template, fch.LEVEL0F)
        delta_e, dx2_pc = up["delta_e"], up["dx2"]
        out2_pc = rows16[2].output_change(up["rows"][2], up["values"][2])
        xb2 = fcm.bases_012[template][1].double()
        parts = {p_c: (dx2_pc + out2_pc, lw.hidden(BLOCK, x2 + dx2_pc) - lw.hidden(BLOCK, x2), lw.hidden(BLOCK, xb2 + dx2_pc) - lw.hidden(BLOCK, xb2))}
        if p_t != p_c:
            # Experiment 017's propagation step, verbatim: layer 1 at query p_t with the key and value at p_c at x₁(p_c) + ΔE; layer 2 at query p_t with p_c at x₂(p_c) + Δ̂x₂(p_c) and p_t at x₂(p_t) + Δ̂x₂(p_t).
            program1, program2 = fcm.programs[1], fcm.programs[2]
            rr1 = atp.ReferenceRow(program1, [x.double() for x in x1_all[: p_t + 1]])
            rows1, values1 = hp._row_and_values(program1, rr1, None, {p_c: program1.normalize(x1 + delta_e)})
            dx2_pt = hp._attention_output(program1, rows1, values1) - rr1.output_ref
            rr2 = atp.ReferenceRow(program2, [x.double() for x in x2_all[: p_t + 1]])
            x2_pt = x2_all[p_t].double()
            normed2 = {p_c: program2.normalize(x2 + dx2_pc), p_t: program2.normalize(x2_pt + dx2_pt)}
            rows2, values2 = hp._row_and_values(program2, rr2, normed2[p_t], normed2)
            out2_pt = hp._attention_output(program2, rows2, values2) - rr2.output_ref
            if self.base2_pt is None:
                raise pm.IncidentError(f"no locked block-2 base at p_t for template {template}")
            xb2_pt = self.base2_pt.double()
            parts[p_t] = (dx2_pt + out2_pt, lw.hidden(BLOCK, x2_pt + dx2_pt) - lw.hidden(BLOCK, x2_pt), lw.hidden(BLOCK, xb2_pt + dx2_pt) - lw.hidden(BLOCK, xb2_pt))
        return UpstreamParts(parts, delta_e, float(up["denominator"]), float(up["c_L"]))

    def dx3(self, up: UpstreamParts, mask: torch.Tensor) -> dict[int, torch.Tensor]:
        W_out = self.lw.W_out[BLOCK]
        return {pos: base + ((mask * own + (1.0 - mask) * template) @ W_out) for pos, (base, own, template) in up.parts.items()}

    def rung(self, rr3: atp.ReferenceRow, x3_all: Sequence[torch.Tensor], up: UpstreamParts, mask: torch.Tensor, p_c: int, p_t: int, template: str) -> dict[str, Any]:
        dx3 = self.dx3(up, mask)
        head = self.hcm.head(rr3, x3_all, dx3, p_c, p_t, template, hp.LEVEL0)
        return {"row": head["row"], "F": head["F"], "Pi": head["Pi"], "dT": head["dT"], "c_L": self.hcm.fcm.read.inner(dx3[p_c] - up.delta_e) / up.denominator}

    def predict(self, weights: pm.Weights, x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], x3_all: Sequence[torch.Tensor], p_c: int, p_t: int, token_id: int, template: str,
                oracle_mask: torch.Tensor | None = None) -> dict[str, Any]:
        """The table entry: every rung's row, F̂, Π̂, ΔT̂ and ĉ_L, the frozen pattern from the reference rung, Experiment 016's decoded c_L, and the frame's own subset (the oracle) when given."""
        rows16 = atp.reference_rows(self.hcm.fcm.programs, x1_all[: p_c + 1], x2_all[: p_c + 1])
        rr3 = atp.ReferenceRow(self.hcm.program3, [x.double() for x in x3_all[: p_t + 1]])
        up = self.upstream_parts(weights, rows16, x1_all, x2_all, p_c, p_t, token_id, template)
        entry: dict[str, Any] = {"p_c": p_c, "p_t": p_t}
        for name in RUNGS:
            out = self.rung(rr3, x3_all, up, self.masks[name], p_c, p_t, template)
            entry[f"row_{name}"] = out["row"].tolist()
            for obj in OBJECTS:
                entry[f"{obj}_{name}"] = float(out[obj])
        if oracle_mask is not None:
            out = self.rung(rr3, x3_all, up, oracle_mask, p_c, p_t, template)
            entry[f"row_{ORACLE}"] = out["row"].tolist()
            for obj in OBJECTS:
                entry[f"{obj}_{ORACLE}"] = float(out[obj])
        else:
            entry[f"row_{ORACLE}"] = [0.0] * (p_t + 1)
            for obj in OBJECTS:
                entry[f"{obj}_{ORACLE}"] = 0.0
        entry["dT_frozen"] = entry[f"F_{REFERENCE}"]
        entry["c_L_level0F"] = up.c_L_level0F
        return entry

    def predict_from_locked(self, weights: pm.Weights, locked: Mapping[str, Any], token_id: int, template: str, oracle_mask: torch.Tensor | None = None) -> dict[str, Any]:
        return self.predict(weights, hp._tensors(locked["x1_all"]), hp._tensors(locked["x2_all"]), hp._tensors(locked["x3_all"]), int(locked["p_c"]), int(locked["p_t"]), token_id, template, oracle_mask)

    def predict_from_state(self, weights: pm.Weights, state: hp.FrameState017, token_id: int, template: str, oracle_mask: torch.Tensor | None = None) -> dict[str, Any]:
        return self.predict(weights, state.x1_all, state.x2_all, state.x3_all, state.p_c, state.p_t, token_id, template, oracle_mask)


def mask_of(indices: Sequence[int], n_neurons: int = N_NEURONS) -> torch.Tensor:
    mask = torch.zeros(n_neurons, dtype=torch.float64)
    if len(indices):
        mask[torch.tensor(sorted(int(i) for i in indices), dtype=torch.long)] = 1.0
    return mask


def masks_from_subsets(subsets: Mapping[str, Sequence[int]], n_neurons: int = N_NEURONS) -> dict[str, torch.Tensor]:
    return {name: mask_of(subsets[name], n_neurons) for name in RUNGS}


def block2_base_pt(frames: Sequence[pm.Frame], states: Mapping[str, hp.FrameState017]) -> tuple[torch.Tensor | None, int]:
    """The coordinated template's block-2 base at p_t: the mean of x₂(p_t) over the exposed coordinated frames (None if there are none)."""
    chosen = [states[frame.frame_id] for frame in frames if frame.p_t != frame.p_c and frame.frame_id in states]
    if not chosen:
        return None, 0
    return torch.stack([s.x2_all[s.p_t] for s in chosen]).mean(0), len(chosen)


# ---------------------------------------------------------------------------
# The ranking rule, the subsets and the controls.


def read_of_outputs(read: lc.CorrectionRead, lw: lc.LayerWeights) -> torch.Tensor:
    """r(W_out^(2)[j, :]) for every neuron: the weight-only read of each neuron's output direction."""
    W_out = lw.W_out[BLOCK]
    return torch.tensor([read.inner(W_out[j]) for j in range(W_out.shape[0])], dtype=torch.float64)


class RankingAccumulator:
    """score_j = mean over (pair, changed position) of |e_j| · |r(W_out[j])| / |D_T|; the records are counted, never weighted."""

    def __init__(self, read_out: torch.Tensor) -> None:
        self.read_out = read_out.abs()
        self.total = torch.zeros_like(read_out)
        self.count = 0

    def add(self, up: UpstreamParts) -> None:
        for position in up.parts:
            self.total += up.effects(position).abs() * self.read_out / abs(up.denominator)
            self.count += 1

    def scores(self) -> torch.Tensor:
        if self.count == 0:
            raise pm.IncidentError("the ranking pool is empty")
        return self.total / self.count


def order_of(scores: torch.Tensor) -> list[int]:
    """Neurons by descending score; ties broken by the lower index."""
    values = scores.tolist()
    return sorted(range(len(values)), key=lambda j: (-values[j], j))


def subsets_from_scores(scores: torch.Tensor, *, seed: int = CONTROL_SEED) -> dict[str, list[int]]:
    """The ladder S_n (top n of the order), the three random controls drawn in sequence from one seeded generator (overlaps expected, never redrawn) and the bottom 256."""
    order = order_of(scores)
    n = len(order)
    subsets = {f"S{size}": sorted(order[:size]) for size in SUBSET_SIZES}
    rng = random.Random(seed)
    for name in RANDOM_RUNGS:
        subsets[name] = sorted(rng.sample(range(n), HYPOTHESIS_SIZE))
    subsets[BOTTOM_RUNG] = sorted(order[-BOTTOM_SIZE:])
    return subsets


def subset_overlaps(subsets: Mapping[str, Sequence[int]]) -> dict[str, int]:
    names = (HYPOTHESIS, *RANDOM_RUNGS, BOTTOM_RUNG)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}&{b}"] = len(set(subsets[a]) & set(subsets[b]))
    return out


def check_subsets(subsets: Mapping[str, Sequence[int]], scores: torch.Tensor) -> None:
    """The lock's subsets are the frozen rule applied to the lock's scores: reproduced exactly, each control 256 unique indices."""
    expected = subsets_from_scores(scores)
    for name in RUNGS:
        mine = [int(i) for i in subsets[name]]
        if mine != expected[name] or len(set(mine)) != len(mine):
            raise PhaseError(f"the locked subset {name} is not the frozen rule applied to the locked scores")
    for name in (*RANDOM_RUNGS, BOTTOM_RUNG):
        if len(subsets[name]) != BOTTOM_SIZE:
            raise PhaseError(f"the control {name} does not hold {BOTTOM_SIZE} indices")


def frame_subset(scores: torch.Tensor, size: int = HYPOTHESIS_SIZE) -> list[int]:
    return sorted(order_of(scores)[:size])


def overlap_with(a: Sequence[int], b: Sequence[int]) -> int:
    return len(set(int(i) for i in a) & set(int(i) for i in b))


# ---------------------------------------------------------------------------
# Measurement per pair.


@dataclass(frozen=True)
class AnalysisContext:
    model: MaskedChainModel
    hp_context: hp.AnalysisContext
    weights: pm.Weights
    axis_T: pm.SiteAxis


def check_i8(entry: Mapping[str, Any], prediction_017: Mapping[str, Any], cue_final: bool, where: str) -> float:
    """I8: the reference rung equals Experiment 017's Level 0 (row, F̂, Π̂, ΔT̂, decoded c_L) and, on cue-final pairs, S_0 equals its −D rung."""
    worst = max(abs(entry[f"F_{REFERENCE}"] - prediction_017["F_hat"]), abs(entry[f"Pi_{REFERENCE}"] - prediction_017["Pi_hat"]), abs(entry[f"dT_{REFERENCE}"] - prediction_017["dT_hat"]),
                abs(entry[f"c_L_{REFERENCE}"] - prediction_017["c_L_level0D"]), max(abs(a - b) for a, b in zip(entry[f"row_{REFERENCE}"], prediction_017["row_level0"])))
    if cue_final:
        worst = max(worst, abs(entry[f"Pi_{TEMPLATE_BASE}"] - prediction_017["Pi_no_D"]), abs(entry[f"dT_{TEMPLATE_BASE}"] - prediction_017["dT_no_D"]), max(abs(a - b) for a, b in zip(entry[f"row_{TEMPLATE_BASE}"], prediction_017["row_no_D"])))
    if worst > I8_TOLERANCE:
        raise pm.IncidentError(f"{where}: the masked chain does not reproduce Experiment 017's rungs (I8 {worst:.2e})")
    return worst


def analyse_pair_018(record: ra.Attribution, plural: ra.Attribution, *, context: AnalysisContext, state: hp.FrameState017, oracle_mask: torch.Tensor | None = None) -> dict[str, Any] | None:
    """Experiment 017's analysis (I1–I7, its Level 0 and rungs, the measured objects) plus every rung of the masked chain and I8."""
    base = hp.analyse_pair_017(record, plural, context=context.hp_context, state=state)
    if base is None:
        return None
    entry = context.model.predict_from_state(context.weights, state, record.token_id, state.frame.template_id, oracle_mask)
    identities = dict(base["identities"])
    identities["I8_reference_rung"] = check_i8(entry, base["prediction"], base["cue_final"], f"{state.frame.frame_id}/{record.token}")
    measured_row = torch.tensor(base["row"], dtype=torch.float64)
    statistics = {name: atp.row_statistics(torch.tensor(entry[f"row_{name}"], dtype=torch.float64), measured_row) for name in (*RUNGS, ORACLE)}
    return {"token": base["token"], "token_id": base["token_id"], "frame_id": base["frame_id"], "template": base["template"], "p_c": base["p_c"], "p_t": base["p_t"], "cue_final": base["cue_final"],
            "row": base["row"], "F": base["F"], "Pi": base["Pi"], "dT": base["dT"], "c_L": base["c_L"], "prediction": entry, "statistics": statistics, "identities": identities,
            "x3_remainder": base["x3_remainder"], "analysis_017": {"F": base["F"], "Pi": base["Pi"], "dT": base["dT"], "c_L": base["c_L"], "row": base["row"], "prediction": base["prediction"]}}


def compact_analysis(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """What the results state stores per pair: everything but the Experiment 017 sub-analysis (the digest-checked extract keeps that record)."""
    return {key: value for key, value in analysis.items() if key != "analysis_017"}


def prediction_row(word: str, frame_id: str, template: str, prediction: Mapping[str, Any]) -> dict[str, Any]:
    row = {"token": word, "frame_id": frame_id, "template": template}
    for key in PREDICTION_COLUMNS[3:]:
        row[key] = prediction[key]
    return row


# ---------------------------------------------------------------------------
# Results state, phases.

DIGEST_KEYS = (*(key for key in hp.DIGEST_KEYS if key != "extract_016"), "lock_017", "extract_017", "confirmation_018")  # Experiment 016's extract was Experiment 017's input, bound by the 017 lock; it is not read here
STATE_DIGEST_FIELDS = (*(field for field in hp.STATE_DIGEST_FIELDS if field != "extract_016_sha256"), "lock_017_sha256", "extract_017_sha256", "confirmation_018_sha256")
_STATE_KEYS = {"schema_version", "run_id", "created_at", *STATE_DIGEST_FIELDS, "protocol_code_commit", "git_dirty", "model", "versions", "phases", "executed_prompt_keys", "executed_noun_keys", "exploration", "lock", "confirmation", "invalidated_runs", "state_sha256"}


def new_results_state(*, digests: Mapping[str, str], protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    state = {"schema_version": RESULTS_SCHEMA_VERSION, "run_id": pm.sha256_text("".join(digests[key] for key in sorted(digests)) + protocol_code_commit + pm.utc_now())[:16], "created_at": pm.utc_now(),
             "protocol_code_commit": protocol_code_commit, "git_dirty": False, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "versions": dict(versions),
             "phases": {phase: {"status": "not_started"} for phase in PHASES}, "executed_prompt_keys": [], "executed_noun_keys": [], "exploration": {}, "lock": None, "confirmation": None, "invalidated_runs": []}
    for field, key in zip(STATE_DIGEST_FIELDS, DIGEST_KEYS):
        state[field] = digests[key]
    return state


def state_digests(state: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(state[field] for field in STATE_DIGEST_FIELDS)


def digest_tuple(digests: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(digests[key] for key in DIGEST_KEYS)


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
    try:
        ht.assert_phase_allowed(phase, state)
    except pm.PhaseError as error:
        raise PhaseError(str(error)) from None


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation018) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


def stage_digest(rows: Sequence[Mapping[str, Any]], states: Mapping[str, Any], frame_subsets: Mapping[str, Sequence[int]]) -> str:
    return pm.sha256_text(pm.canonical_json({"rows": list(rows), "states": dict(states), "frame_subsets": {key: list(value) for key, value in frame_subsets.items()}}))


def assert_stage_one_digest(stage1: Mapping[str, Any]) -> None:
    if not stage1 or stage1.get("digest") != stage_digest(stage1["rows"], stage1["states"], stage1.get("frame_subsets", {})):
        raise PhaseError("stage 1's prediction table is missing or its digest does not verify; no fresh cue prompt may run")


comparison = lc.comparison


# ---------------------------------------------------------------------------
# Statistics: R² per rung and object, the closure fraction κ, per frame, splits.


def _mean(values: Sequence[float]) -> float:
    return pm._mean(list(values))


def _r2_scalar(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], obj: str, rung: str) -> dict[str, Any]:
    return comparison([p[f"{obj}_{rung}"] for p in predictions], [a[obj] for a in analyses])


def _rows_r2(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], rung: str) -> dict[str, Any]:
    stats = [atp.row_statistics(torch.tensor(p[f"row_{rung}"], dtype=torch.float64), torch.tensor(a["row"], dtype=torch.float64)) for a, p in zip(analyses, predictions)]
    return atp.pooled_summary(stats)


def kappa(value: float | None, base: float | None, full: float | None) -> float | None:
    """The closure fraction (R²(S) − R²(S_0)) / (R²(S_2048) − R²(S_0)): unclipped; None when any R² is undefined or the gap is below GAP_MIN."""
    if value is None or base is None or full is None:
        return None
    gap = full - base
    if gap < GAP_MIN:
        return None
    return (value - base) / gap


def rung_statistics(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], rungs: Sequence[str] = (*RUNGS, ORACLE)) -> dict[str, Any]:
    """Per rung: pooled row entries and R² of c_L, F, Π, ΔT on the pairs; the gaps and κ per object; the frozen pattern and Experiment 016's decoded c_L beside them."""
    out: dict[str, Any] = {"n_pairs": len(analyses), "rungs": {}}
    for rung in rungs:
        out["rungs"][rung] = {"rows": _rows_r2(analyses, predictions, rung), **{obj: _r2_scalar(analyses, predictions, obj, rung) for obj in OBJECTS}}
    out["frozen"] = comparison([p["dT_frozen"] for p in predictions], [a["dT"] for a in analyses])
    out["c_L_level0F"] = comparison([p["c_L_level0F"] for p in predictions], [a["c_L"] for a in analyses])
    r2 = lambda obj, rung: out["rungs"][rung]["rows"]["entry_r2"] if obj == "rows" else out["rungs"][rung][obj]["r2"]  # noqa: E731
    out["gaps"] = {}
    out["kappa"] = {}
    for obj in (*OBJECTS, "rows"):
        base, full = r2(obj, TEMPLATE_BASE), r2(obj, REFERENCE)
        out["gaps"][obj] = (full - base) if base is not None and full is not None else None
        out["kappa"][obj] = {rung: kappa(r2(obj, rung), base, full) for rung in rungs}
    out["evaluable"] = {obj: out["gaps"][obj] is not None and out["gaps"][obj] >= GAP_MIN for obj in (*OBJECTS, "rows")}
    return out


def token_mean_statistics(tokens_out: Mapping[str, Mapping[str, Any]], scored: Sequence[str]) -> dict[str, Any]:
    """The same R²/κ structure on token means over identical frame sets (descriptive)."""
    out: dict[str, Any] = {"n_tokens": len(scored), "rungs": {}}
    if len(scored) < 2:
        return out
    for rung in (*RUNGS, ORACLE):
        out["rungs"][rung] = {obj: comparison([tokens_out[w][f"{obj}_{rung}_mean"] for w in scored], [tokens_out[w][f"{obj}_mean"] for w in scored]) for obj in OBJECTS}
    out["frozen"] = comparison([tokens_out[w]["dT_frozen_mean"] for w in scored], [tokens_out[w]["dT_mean"] for w in scored])
    out["kappa"] = {}
    for obj in OBJECTS:
        base, full = out["rungs"][TEMPLATE_BASE][obj]["r2"], out["rungs"][REFERENCE][obj]["r2"]
        out["kappa"][obj] = {rung: kappa(out["rungs"][rung][obj]["r2"], base, full) for rung in (*RUNGS, ORACLE)}
    return out


def _token_means(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not analyses:
        return {}
    out = {"n_frames": len(analyses), "frames": sorted(a["frame_id"] for a in analyses), "n_cue_final": sum(1 for a in analyses if a["cue_final"]),
           **{f"{obj}_mean": _mean([a[obj] for a in analyses]) for obj in OBJECTS}}
    out.update({f"{key}_mean": _mean([p[key] for p in predictions]) for key in SCALAR_KEYS})
    return out


def per_frame_statistics(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per frame: c_L and rows R² of S_0, S_1, S_256, S_2048 and the oracle, the reference rung's and the frozen pattern's ΔT R², κ_{c_L}(S_256) and the no-harm difference."""
    out = {}
    for frame_id in sorted({a["frame_id"] for a in analyses}):
        mine = [(a, p) for a, p in zip(analyses, predictions) if a["frame_id"] == frame_id]
        a_list, p_list = [a for a, _ in mine], [p for _, p in mine]
        a0 = a_list[0]
        entry: dict[str, Any] = {"n_pairs": len(mine), "template": a0["template"], "cue_final": a0["cue_final"], "c_L": {}, "rows": {}, "Pi": {}, "dT": {}}
        for rung in (TEMPLATE_BASE, SINGLE, HYPOTHESIS, REFERENCE, ORACLE):
            entry["c_L"][rung] = _r2_scalar(a_list, p_list, "c_L", rung)["r2"]
            entry["rows"][rung] = _rows_r2(a_list, p_list, rung)["entry_r2"]
            entry["Pi"][rung] = _r2_scalar(a_list, p_list, "Pi", rung)["r2"]
            entry["dT"][rung] = _r2_scalar(a_list, p_list, "dT", rung)["r2"]
        entry["dT_frozen"] = comparison([p["dT_frozen"] for p in p_list], [a["dT"] for a in a_list])["r2"]
        entry["kappa_c_L"] = kappa(entry["c_L"][HYPOTHESIS], entry["c_L"][TEMPLATE_BASE], entry["c_L"][REFERENCE])
        base, hyp = entry["c_L"][TEMPLATE_BASE], entry["c_L"][HYPOTHESIS]
        entry["no_harm_difference"] = (hyp - base) if base is not None and hyp is not None else None
        out[frame_id] = entry
    return out


def split_statistics(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The cue-final / coordinated split (Y2's guard) and every template: the R² and κ of c_L and Π at the hypothesis, and the gaps."""
    def one(chosen: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
        if not chosen:
            return {"n_pairs": 0}
        stats = rung_statistics([a for a, _ in chosen], [p for _, p in chosen], rungs=(TEMPLATE_BASE, SINGLE, HYPOTHESIS, REFERENCE))
        return {"n_pairs": len(chosen), "n_tokens": len({a["token"] for a, _ in chosen}), "n_frames": len({a["frame_id"] for a, _ in chosen}),
                "c_L_r2": {rung: stats["rungs"][rung]["c_L"]["r2"] for rung in (TEMPLATE_BASE, SINGLE, HYPOTHESIS, REFERENCE)}, "Pi_r2": {rung: stats["rungs"][rung]["Pi"]["r2"] for rung in (TEMPLATE_BASE, SINGLE, HYPOTHESIS, REFERENCE)},
                "gap_c_L": stats["gaps"]["c_L"], "gap_Pi": stats["gaps"]["Pi"], "kappa_c_L": stats["kappa"]["c_L"][HYPOTHESIS], "kappa_Pi": stats["kappa"]["Pi"][HYPOTHESIS], "kappa_c_L_single": stats["kappa"]["c_L"][SINGLE],
                "evaluable_c_L": stats["evaluable"]["c_L"], "frozen_dT_r2": stats["frozen"]["r2"], "reference_dT_r2": stats["rungs"][REFERENCE]["dT"]["r2"]}

    pairs = list(zip(analyses, predictions))
    return {"cue_final": one([x for x in pairs if x[0]["cue_final"]]), "coordinated": one([x for x in pairs if not x[0]["cue_final"]]),
            "per_template": {template: one([x for x in pairs if x[0]["template"] == template]) for template in pm.TEMPLATE_ORDER}}


def predeclared_orderings(stats: Mapping[str, Any]) -> dict[str, Any]:
    """(i) the row is diffuse at S_256; (ii) S_256 beats the best random control by RANDOM_MARGIN and beats bottom_256; (iii) κ_{c_L}(S_256) ≥ κ_Π(S_256)."""
    k = stats["kappa"]
    row, cl, pi = k["rows"].get(HYPOTHESIS), k["c_L"].get(HYPOTHESIS), k["Pi"].get(HYPOTHESIS)
    randoms = [k["c_L"].get(name) for name in RANDOM_RUNGS]
    best_random = max((v for v in randoms if v is not None), default=None)
    bottom = k["c_L"].get(BOTTOM_RUNG)
    return {"row_diffuse": {"kappa_row": row, "max": ROW_DIFFUSE_MAX, "holds": row is not None and row < ROW_DIFFUSE_MAX},
            "random_margin": {"kappa_c_L": cl, "random": randoms, "best_random": best_random, "bottom": bottom, "margin": RANDOM_MARGIN,
                              "holds": cl is not None and best_random is not None and bottom is not None and cl - best_random >= RANDOM_MARGIN and cl > bottom},
            "c_L_over_Pi": {"kappa_c_L": cl, "kappa_Pi": pi, "holds": cl is not None and pi is not None and cl >= pi}}


def statistics_for(pairs: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, Any]:
    """The exposed record: pooled rung statistics, token means, per frame, splits and the orderings."""
    by_token = {name: [(a, p) for a, p in zip(pairs, predictions) if a["token"] == name] for name in tokens}
    means = {name: _token_means([a for a, _ in rows], [p for _, p in rows]) for name, rows in by_token.items() if rows}
    stats = rung_statistics(pairs, predictions)
    return {"n_tokens": len(means), "n_pairs": len(pairs), "pairs": stats, "token_means": token_mean_statistics(means, sorted(means)), "per_frame": per_frame_statistics(pairs, predictions),
            "split": split_statistics(pairs, predictions), "orderings": predeclared_orderings(stats), "token_means_table": means}


# ---------------------------------------------------------------------------
# Components and the ranking pool.


def _components(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], bases_3: Mapping[str, Mapping[str, torch.Tensor | None]],
                base2_pt: torch.Tensor | None, subsets: Mapping[str, Sequence[int]] | None) -> dict[str, Any]:
    parts = hp._components(model, pool, pool_010, lock_011, lock_012, bases_3)
    n = int(parts["lw"].W_out[BLOCK].shape[0])
    if n != N_NEURONS:
        raise pm.IncidentError(f"block 2 has {n} neurons; the design fixes {N_NEURONS}")
    masks = masks_from_subsets(subsets, n) if subsets is not None else {name: torch.ones(n, dtype=torch.float64) for name in RUNGS}
    out = dict(parts)
    out["masked"] = MaskedChainModel(parts["model"], base2_pt, masks)
    out["context_018"] = AnalysisContext(out["masked"], parts["context"], parts["weights"], parts["axis_T"])
    out["read_out"] = read_of_outputs(parts["model"].fcm.read, parts["lw"])
    return out


def ranking_pool(extract_entries: Mapping[str, Any], pool: cs.Pool008) -> list[tuple[str, str, str, int]]:
    """The licensed exposed pairs (token, frame_id, template, token_id) — exactly the pairs Experiment 017 recorded — in sorted order."""
    token_ids = dict(pool.tokens)
    templates = {frame.frame_id: frame.template_id for frame in pool.frames}
    out = []
    for key in sorted(extract_entries):
        word, frame_id = key.split("|", 1)
        if word not in token_ids or frame_id not in templates:
            raise pm.IncidentError(f"{key}: the recorded pair is not in the exposed pool")
        out.append((word, frame_id, templates[frame_id], token_ids[word]))
    return out


def tokens_by_template(ranking: Sequence[tuple[str, str, str, int]]) -> dict[str, list[tuple[str, int]]]:
    """The exposed tokens licensed in each template's frames (the reference cue of a template is licensed only outside it)."""
    out: dict[str, dict[str, int]] = {}
    for word, _, template, token_id in ranking:
        out.setdefault(template, {})[word] = token_id
    return {template: sorted(entries.items()) for template, entries in out.items()}


def frame_ranking(masked: MaskedChainModel, weights: pm.Weights, read_out: torch.Tensor, locked: Mapping[str, Any], template: str, tokens: Sequence[tuple[str, int]]) -> torch.Tensor:
    """One frame ranked on the given exposed tokens' predicted changes at its own reference state (reference-run quantities and ΔE only)."""
    x1_all, x2_all = hp._tensors(locked["x1_all"]), hp._tensors(locked["x2_all"])
    p_c, p_t = int(locked["p_c"]), int(locked["p_t"])
    rows16 = atp.reference_rows(masked.hcm.fcm.programs, x1_all[: p_c + 1], x2_all[: p_c + 1])
    accumulator = RankingAccumulator(read_out)
    for _, token_id in tokens:
        accumulator.add(masked.upstream_parts(weights, rows16, x1_all, x2_all, p_c, p_t, token_id, template))
    return accumulator.scores()


def pooled_ranking(masked: MaskedChainModel, weights: pm.Weights, read_out: torch.Tensor, locked_states: Mapping[str, Mapping[str, Any]], ranking: Sequence[tuple[str, str, str, int]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The pooled scores over the ranking pool and, from the same records, every exposed frame's own scores."""
    accumulator = RankingAccumulator(read_out)
    per_frame: dict[str, RankingAccumulator] = {}
    rows_cache: dict[str, tuple[Any, list[torch.Tensor], list[torch.Tensor]]] = {}
    for _, frame_id, template, token_id in ranking:
        if frame_id not in rows_cache:
            locked = locked_states[frame_id]
            x1_all, x2_all = hp._tensors(locked["x1_all"]), hp._tensors(locked["x2_all"])
            rows_cache[frame_id] = (atp.reference_rows(masked.hcm.fcm.programs, x1_all[: int(locked["p_c"]) + 1], x2_all[: int(locked["p_c"]) + 1]), x1_all, x2_all)
            per_frame[frame_id] = RankingAccumulator(read_out)
        rows16, x1_all, x2_all = rows_cache[frame_id]
        locked = locked_states[frame_id]
        up = masked.upstream_parts(weights, rows16, x1_all, x2_all, int(locked["p_c"]), int(locked["p_t"]), token_id, template)
        accumulator.add(up)
        per_frame[frame_id].add(up)
    return accumulator.scores(), {frame_id: acc.scores() for frame_id, acc in per_frame.items()}


def _check_previous_state(state_f: hp.FrameState017, lock_017: Mapping[str, Any], inherited_extract: Mapping[str, Any]) -> None:
    frame_id = state_f.frame.frame_id
    previous = lock_017.get("locked_states", {}).get(frame_id)
    if previous is not None:
        hp._check_locked_state_017(state_f, previous, frame_id)
    elif frame_id in inherited_extract.get("stage1_state_digests", {}):
        if ap.state_digest(hp.locked_state(state_f)) != inherited_extract["stage1_state_digests"][frame_id]:
            raise pm.IncidentError(f"{frame_id}: the reference state differs from Experiment 017's digested stage-1 state")
    else:
        raise pm.IncidentError(f"{frame_id}: no recorded Experiment 017 reference state to check against")


def masked_model_from_source(source: Mapping[str, Any], lock_012: Mapping[str, Any], lw: lc.LayerWeights, programs: Mapping[int, atp.LayerProgram], pool: cs.Pool008) -> MaskedChainModel:
    """``source`` is the exploration record at lock time and the lock itself at confirm time: the axes, read weight, layer-3 bases, the p_t base and the subsets."""
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    hcm = hp.model_from_locks(source, lock_012, lw, programs, pool.reference_ids, plural_ids, hp.bases_from_json(source["bases_3"]))
    base2_pt = torch.tensor(source["base2_pt"]["vector"], dtype=torch.float64) if source["base2_pt"]["vector"] is not None else None
    return MaskedChainModel(hcm, base2_pt, masks_from_subsets(source["subsets"], int(lw.W_out[BLOCK].shape[0])))


# ---------------------------------------------------------------------------
# Exploration (Tier A).


def run_exploration(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, *, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_017: Mapping[str, Any], inherited_extract: Mapping[str, Any],
                    state: dict[str, Any], results_path: Path | None, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    nouns = pool.single_nouns
    exploration = state["exploration"]
    bases_3 = hp.bases_from_json(lock_017["bases_3"])
    say("stage axes from the Experiment 010 pool, checked against the Experiment 011 lock; Experiment 017's layer-3 bases")
    parts = _components(model, pool, pool_010, lock_011, lock_012, bases_3, None, None)
    weights, head, axis_T, e_axis, programs, hcm = parts["weights"], parts["head"], parts["axis_T"], parts["e_axis"], parts["programs"], parts["model"]
    exploration["read_weight_check"] = lc.check_read_weight(parts["fpm"].read, head, axis_T)
    exploration["axes_vectors"] = {"T": axis_T.direction.tolist(), "R0": torch.tensor(lock_011["axes_vectors"]["R0"], dtype=torch.float64).tolist()}
    exploration["sigma_T"] = axis_T.sigma
    exploration["read_weight"] = parts["fpm"].read.weight.tolist()
    exploration["defined_templates"] = list(lock_012["defined_templates"])
    exploration["program"] = atp._program_record(programs)
    exploration["bases_3"] = dict(lock_017["bases_3"])
    say(f"reference states of the {len(pool.frames)} exposed frames, checked against Experiment 017")
    states: dict[str, hp.FrameState017] = {}
    identities: dict[str, float] = {}
    for frame in pool.frames:
        state_f = hp.capture_frame_017(model, head, pool.reference_prompt(frame), nouns, axis_T)
        rows16 = atp.reference_rows({layer: programs[layer] for layer in hp.UPSTREAM_LAYERS}, state_f.state.x1_all, state_f.state.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows16, state_f.state))
        identities["I5_reference_head_row"] = max(identities.get("I5_reference_head_row", 0.0), hp.check_reference_row(state_f, programs[HEAD_LAYER]))
        _check_previous_state(state_f, lock_017, inherited_extract)
        states[frame.frame_id] = state_f
    base2_pt, n_pt = block2_base_pt(pool.frames, states)
    exploration["base2_pt"] = {"vector": None if base2_pt is None else base2_pt.tolist(), "n_frames": n_pt, "template": "coordinated-adjective"}
    exploration["locked_states"] = {frame_id: hp.locked_state(s) for frame_id, s in states.items()}
    exploration["locked_state_digests"] = {frame_id: ap.state_digest(entry) for frame_id, entry in exploration["locked_states"].items()}
    # The ranking: from the locked states, the weights and the exposed cues' ΔE only, before any pair is measured.
    ranking = ranking_pool(inherited_extract["entries"], pool)
    say(f"the ranking over the {len(ranking)} licensed exposed pairs (predicted changes only)")
    unmasked = MaskedChainModel(hcm, base2_pt, {name: torch.ones(N_NEURONS, dtype=torch.float64) for name in RUNGS})
    read_out = parts["read_out"]
    scores, frame_scores = pooled_ranking(unmasked, weights, read_out, exploration["locked_states"], ranking)
    subsets = subsets_from_scores(scores)
    exploration["scores"] = scores.tolist()
    exploration["subsets"] = subsets
    exploration["overlaps"] = subset_overlaps(subsets)
    exploration["frame_subsets"] = {frame_id: frame_subset(frame_scores[frame_id]) for frame_id in sorted(frame_scores)}
    exploration["frame_overlaps"] = {frame_id: {HYPOTHESIS: overlap_with(exploration["frame_subsets"][frame_id], subsets[HYPOTHESIS]), "S64": overlap_with(frame_subset(frame_scores[frame_id], 64), subsets["S64"])} for frame_id in sorted(frame_scores)}
    exploration["ranking_pool"] = {"n_pairs": len(ranking), "n_records": len(ranking) + sum(1 for _, frame_id, _, _ in ranking if states[frame_id].p_t != states[frame_id].p_c), "tokens_by_template": {t: len(v) for t, v in tokens_by_template(ranking).items()}}
    order = order_of(scores)
    say(f"top neurons {order[:10]}; S_1 {subsets[SINGLE]}; overlaps {exploration['overlaps']}")
    masked = MaskedChainModel(hcm, base2_pt, masks_from_subsets(subsets, N_NEURONS))
    context = AnalysisContext(masked, parts["context"], weights, axis_T)
    recorded = inherited_extract["entries"]
    say(f"re-measurement of the {len(recorded)} recorded pairs with the layer-3 residuals and the head's row captured; every rung predicted")
    pairs: dict[str, dict[str, Any]] = {}
    measured_extract: dict[str, dict[str, Any]] = {}
    for frame in pool.frames:
        state_f = states[frame.frame_id]
        names = [name for name, _ in pool.tokens if f"{name}|{frame.frame_id}" in recorded]
        if not names:
            continue
        oracle = mask_of(exploration["frame_subsets"][frame.frame_id])
        plural_name = pool.plural_cue[frame.template_id]
        records = {name: hp.measure_pair(model, weights, head, state_f, name, pool.token_id(name), e_axis, axis_T, nouns) for name in names}
        plural = records[plural_name] if plural_name in records else hp.measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for name, record in records.items():
            analysis = analyse_pair_018(record, plural, context=context, state=state_f, oracle_mask=oracle)
            key = f"{name}|{frame.frame_id}"
            if analysis is None:
                raise pm.IncidentError(f"{key}: no analysis although Experiment 017 recorded the pair")
            er._max_errors(identities, analysis["identities"])
            pairs[key] = analysis
            measured_extract[key] = extract_entry_017(analysis["analysis_017"])
        say(f"  {frame.frame_id}: {len(names)} pairs")
    exploration["identities"] = identities
    exploration["replication"] = {"experiment_017": check_extract_replication(measured_extract, recorded)}
    say(f"replication against Experiment 017: max deviation {exploration['replication']['experiment_017']['max_abs_deviation']:.2e}")
    analyses = list(pairs.values())
    stats = statistics_for(analyses, [a["prediction"] for a in analyses], [name for name, _ in pool.tokens])
    exploration["pairs"] = {key: compact_analysis(value) for key, value in pairs.items()}
    exploration["exposed_check"] = stats
    exploration["neurons"] = {"hypothesis": list(subsets[HYPOTHESIS]), "single": list(subsets[SINGLE]), "overlaps": exploration["overlaps"]}
    k = stats["pairs"]["kappa"]
    exploration["summary"] = {"defined_templates": exploration["defined_templates"], "kappa_c_L": k["c_L"][HYPOTHESIS], "kappa_Pi": k["Pi"][HYPOTHESIS], "kappa_rows": k["rows"][HYPOTHESIS], "kappa_c_L_single": k["c_L"][SINGLE],
                              "reference_c_L_r2": stats["pairs"]["rungs"][REFERENCE]["c_L"]["r2"], "reference_rows_r2": stats["pairs"]["rungs"][REFERENCE]["rows"]["entry_r2"], "gaps": stats["pairs"]["gaps"],
                              "orderings": {name: entry["holds"] for name, entry in stats["orderings"].items()}, "prediction_undefined": len(exploration["defined_templates"]) < 2 or base2_pt is None}
    f = cd._f
    s = exploration["summary"]
    say(f"exposed: κ_c_L(S_256) {f(s['kappa_c_L'], 3)}, κ_Π {f(s['kappa_Pi'], 3)}, κ_rows {f(s['kappa_rows'], 3)}, κ_c_L(S_1) {f(s['kappa_c_L_single'], 3)}; reference rung c_L R² {f(s['reference_c_L_r2'], 3)}, rows {f(s['reference_rows_r2'], 3)}; gaps {', '.join(f'{k} {f(v, 3)}' for k, v in s['gaps'].items())}")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


# ---------------------------------------------------------------------------
# Lock and its artifacts.


def prediction_table(masked: MaskedChainModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], frame_subsets: Mapping[str, Sequence[int]], frames: Sequence[pm.Frame], tokens: Sequence[Mapping[str, Any]], defined: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        if frame.template_id not in defined:
            continue
        oracle = mask_of(frame_subsets[frame.frame_id], masked.n_neurons)
        for token in tokens:
            rows.append(prediction_row(token["word"], frame.frame_id, frame.template_id, masked.predict_from_locked(weights, locked_states[frame.frame_id], token["token_id"], frame.template_id, oracle)))
    return rows


def token_means_from_table(rows: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for word in tokens:
        mine = [row for row in rows if row["token"] == word]
        if mine:
            out[word] = {key: _mean([row[key] for row in mine]) for key in SCALAR_KEYS}
            out[word]["n_frames"] = len(mine)
            out[word]["n_cue_final"] = sum(1 for row in mine if row["p_t"] == row["p_c"])
    return out


def lock_predictions(masked: MaskedChainModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], frame_subsets: Mapping[str, Sequence[int]], pool: cs.Pool008, confirmation: Confirmation018, defined: Sequence[str]) -> dict[str, Any]:
    rows = prediction_table(masked, weights, locked_states, frame_subsets, pool.frames, confirmation.tokens, defined)
    return {"rows": rows, "token_means": token_means_from_table(rows, [token["word"] for token in confirmation.tokens])}


def render_predictions(lock: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 018 — preregistered predictions (channel D concentrated in a locked 256-neuron subset; every rung of the masked chain)", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_018_sha256']}`; Experiment 017 lock sha256 `{lock['lock_017_sha256']}`",
             f"- Floors: Y1/Y2 κ_c_L(S_256) ≥ {KAPPA_CL_FLOOR} and κ_Π(S_256) ≥ {KAPPA_PI_FLOOR} on pooled pairs (κ unclipped; undefined below a {GAP_MIN} gap); Y2 split guard cue-final ≥ {SPLIT_GUARD['cue_final']}, coordinated ≥ {SPLIT_GUARD['coordinated']}, no-harm margin {NO_HARM_MARGIN}; "
             f"precondition: reference rung c_L ≥ {PRECONDITION_REFERENCE['c_L']}, rows ≥ {PRECONDITION_REFERENCE['rows']}, ΔT ≥ {PRECONDITION_REFERENCE['dT']}; Y3 κ_c_L(S_1) < {Y3_KAPPA_MAX} and S_256 − S_1 ≥ {Y3_MARGIN} (pooled, pair-weighted); "
             f"Y4 per evaluable fresh cue-final frame (reference ΔT R² ≥ {Y4_REFERENCE_MIN}; ≥ {Y4_MIN_EVALUABLE} of 8): frozen ΔT R² < {Y4_FROZEN_MAX}",
             f"- Locked subsets: S_1 {lock['subsets'][SINGLE]}; S_256 (first 32) {lock['subsets'][HYPOTHESIS][:32]}…; overlaps {lock['overlaps']}",
             f"- Y1 table: {len(lock['predictions']['rows'])} rows (fresh tokens × exposed frames), each with every rung's ĉ_L, F̂, Π̂, ΔT̂ and head row; Y2 rows are computed at confirm stage 1 and digested before any fresh cue prompt", "",
             f"| token | class | frames | **ĉ_L S_256** | ĉ_L S_0 | ĉ_L S_1 | ĉ_L S_2048 | **ΔT̂ S_256** | ΔT̂ S_2048 | frozen ΔT̂ | **Π̂ S_256** | Π̂ S_2048 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    categories = {token["word"]: token["category"] for token in lock["tokens"]}
    means = lock["predictions"]["token_means"]
    for word, m in sorted(means.items(), key=lambda kv: -kv[1][f"dT_{REFERENCE}"]):
        lines.append(f"| {word} | {categories.get(word, '')} | {m['n_frames']} | **{f(m[f'c_L_{HYPOTHESIS}'], 4)}** | {f(m[f'c_L_{TEMPLATE_BASE}'], 4)} | {f(m[f'c_L_{SINGLE}'], 4)} | {f(m[f'c_L_{REFERENCE}'], 4)} | **{f(m[f'dT_{HYPOTHESIS}'], 4)}** | {f(m[f'dT_{REFERENCE}'], 4)} | {f(m['dT_frozen'], 4)} | **{f(m[f'Pi_{HYPOTHESIS}'], 4)}** | {f(m[f'Pi_{REFERENCE}'], 4)} |")
    lines.append("")
    return "\n".join(lines)


def frozen_floors() -> dict[str, Any]:
    return {"kappa_c_L_floor": KAPPA_CL_FLOOR, "kappa_Pi_floor": KAPPA_PI_FLOOR, "split_guard": dict(SPLIT_GUARD), "no_harm_margin": NO_HARM_MARGIN, "gap_min": GAP_MIN, "precondition_reference": dict(PRECONDITION_REFERENCE),
            "y3_kappa_max": Y3_KAPPA_MAX, "y3_margin": Y3_MARGIN, "y4_frozen_max": Y4_FROZEN_MAX, "y4_reference_min": Y4_REFERENCE_MIN, "y4_min_evaluable": Y4_MIN_EVALUABLE, "row_diffuse_max": ROW_DIFFUSE_MAX, "random_margin": RANDOM_MARGIN,
            "hypothesis_size": HYPOTHESIS_SIZE, "subset_sizes": list(SUBSET_SIZES), "n_random_controls": N_RANDOM_CONTROLS, "i8_tolerance": I8_TOLERANCE, "replication_tolerance": REPLICATION_TOLERANCE,
            "min_valid_cue_final_frames": MIN_VALID_CUE_FINAL_FRAMES, "min_valid_coordinated_frames": MIN_VALID_COORDINATED_FRAMES, "min_valid_frames_per_token": MIN_VALID_FRAMES_PER_TOKEN, "min_scored_tokens": MIN_SCORED_TOKENS,
            "frame_cue_effect_rate": FRAME_CUE_EFFECT_RATE, "head_stage_floor": cs.STAGE_UNINFORMATIVE_FLOOR}


def build_candidate_lock(*, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation018, predictions: Mapping[str, Any], protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    if exploration["summary"]["prediction_undefined"]:
        raise PhaseError("fewer than two templates are defined or no block-2 base at p_t: PREDICTION_UNDEFINED, no lock")
    lock = {"schema_version": 1, "experiment": "018", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit}
    for field, key in zip(STATE_DIGEST_FIELDS, DIGEST_KEYS):
        lock[field] = digests[key] if key != "confirmation_018" else confirmation.content_sha256
    lock.update({"confirmation_prompt_keys": sorted(prompt.key for prompt in confirmation.all_prompts), "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
                 "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED}, "head": HEAD_KEY, "block": BLOCK, "program": exploration["program"],
                 "axes_vectors": exploration["axes_vectors"], "sigma_T": exploration["sigma_T"], "read_weight": exploration["read_weight"], "defined_templates": exploration["defined_templates"],
                 "bases_3": exploration["bases_3"], "base2_pt": exploration["base2_pt"], "locked_states": exploration["locked_states"], "locked_state_digests": exploration["locked_state_digests"],
                 "scores": exploration["scores"], "subsets": exploration["subsets"], "overlaps": exploration["overlaps"], "frame_subsets": exploration["frame_subsets"], "frame_overlaps": exploration["frame_overlaps"],
                 "floors": frozen_floors(), "tokens": [dict(token) for token in confirmation.tokens], "exposed_check": exploration["exposed_check"],
                 "predictions": dict(predictions), "tier_a_results_sha256": state.get("state_sha256")})
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation018, predictions_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "018" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked or git_state.get("dirty", True):
        raise PhaseError("confirm requires the committed lock and predictions on a clean Git tree")
    expected = dict(zip(STATE_DIGEST_FIELDS, digest_tuple(digests)))
    expected["confirmation_018_sha256"] = confirmation.content_sha256
    if any(lock.get(field) != value for field, value in expected.items()):
        raise PhaseError("lock was built against different frozen inputs")
    if not state.get("lock") or lock["content_sha256"] != state["lock"]["content_sha256"] or lock["run_id"] != state["run_id"]:
        raise PhaseError("the installed lock is not the candidate lock written by the lock phase")
    if pm.sha256_text(predictions_text) != state["lock"]["predictions_sha256"]:
        raise PhaseError("the installed predictions.md is not the artifact written by the lock phase")
    if dict(lock["floors"]) != frozen_floors():
        raise PhaseError("the lock's floors are not the frozen constants")
    check_subsets(lock["subsets"], torch.tensor(lock["scores"], dtype=torch.float64))
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = [path for path in changed_paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and path not in NON_SCIENTIFIC_PATHS and not path.startswith(NON_SCIENTIFIC_PREFIXES)]
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock commit: {scientific}")
    assert_confirmation_untouched(state, confirmation)


def assert_lock_predictions_reproduced(lock: Mapping[str, Any], recomputed: Mapping[str, Any]) -> float:
    locked_rows, fresh_rows = lock["predictions"]["rows"], recomputed["rows"]
    if len(locked_rows) != len(fresh_rows):
        raise PhaseError("the recomputed prediction table has a different number of rows; nothing was executed")
    worst = 0.0
    for a, b in zip(locked_rows, fresh_rows):
        if (a["token"], a["frame_id"], a["template"], a["p_c"], a["p_t"]) != (b["token"], b["frame_id"], b["template"], b["p_c"], b["p_t"]):
            raise PhaseError("the recomputed prediction table is ordered differently; nothing was executed")
        for key in PREDICTION_COLUMNS[5:]:
            worst = max(worst, atp._max_numeric_difference(a[key], b[key], f"{a['token']}/{a['frame_id']}/{key}"))
    if worst > LOCK_PREDICTION_TOLERANCE:
        raise PhaseError(f"the weights, locked axes, bases, subsets and locked reference states do not reproduce the preregistered predictions (max difference {worst:.3e}); nothing was executed")
    return worst


def assert_ranking_reproduced(lock: Mapping[str, Any], scores: torch.Tensor, frame_subsets: Mapping[str, Sequence[int]]) -> float:
    """At confirm the ranking is recomputed from the locked states and must give the locked subsets exactly."""
    worst = float((scores - torch.tensor(lock["scores"], dtype=torch.float64)).abs().max())
    if subsets_from_scores(scores) != {name: [int(i) for i in lock["subsets"][name]] for name in RUNGS} or {k: [int(i) for i in v] for k, v in frame_subsets.items()} != {k: [int(i) for i in v] for k, v in lock["frame_subsets"].items()}:
        raise PhaseError("the ranking recomputed from the locked states does not give the locked subsets; nothing was executed")
    return worst


# ---------------------------------------------------------------------------
# Confirmation: stage 1, stage 2, scoring.


def stage_one(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation018, lock: Mapping[str, Any], lock_012: Mapping[str, Any], inherited_extract: Mapping[str, Any], *, protocol_code_commit: str, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    base2_pt = torch.tensor(lock["base2_pt"]["vector"], dtype=torch.float64) if lock["base2_pt"]["vector"] is not None else None
    parts = _components(model, pool, pool_010, lock, lock_012, hp.bases_from_json(lock["bases_3"]), base2_pt, lock["subsets"])
    weights, head, cache, axis_T, e_axis, masked = parts["weights"], parts["head"], parts["cache"], parts["axis_T"], parts["e_axis"], parts["masked"]
    if atp._program_record(masked.hcm.programs) != dict(lock["program"]):
        raise pm.IncidentError("the program's frozen configuration differs from the lock's record")
    nouns = pool.single_nouns
    by_template = tokens_by_template(ranking_pool(inherited_extract["entries"], pool))
    say("stage 1: fresh frames' reference states, validity, the frame's own ranking, the prediction table")
    frames_out: dict[str, Any] = {}
    states: dict[str, Any] = {}
    frame_subsets: dict[str, list[int]] = {}
    frame_overlaps: dict[str, Any] = {}
    rows_out: list[dict[str, Any]] = []
    identities: dict[str, float] = {}
    for frame in confirmation.frames:
        template = frame.template_id
        if template not in lock["defined_templates"]:
            frames_out[frame.frame_id] = {"template_id": template, "valid": False, "template_defined": False, "note": "template excluded before the lock; nothing run"}
            continue
        state_f = hp.capture_frame_017(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        rows16 = atp.reference_rows(masked.hcm.fcm.programs, state_f.state.x1_all, state_f.state.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows16, state_f.state))
        identities["I5_reference_head_row"] = max(identities.get("I5_reference_head_row", 0.0), hp.check_reference_row(state_f, masked.hcm.program3))
        sg, pl = pm.frame_prompts(frame)
        c_a, c_b = cache.c(sg), cache.c(pl)
        positive = sum(1 for noun in nouns if c_a[noun.lexical_key] - c_b[noun.lexical_key] > 0)
        required = pm.exact_count_floor(FRAME_CUE_EFFECT_RATE, len(nouns))
        plural = hp.measure_pair(model, weights, head, state_f, pool.plural_cue[template], pl.cue_token_id, e_axis, axis_T, nouns)
        er._max_errors(identities, er.enforce_identities(plural, plural, f"{frame.frame_id}/{pool.plural_cue[template]}"))
        head_informative = abs(plural.head_change) >= cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma
        valid = head_informative and positive >= required
        frames_out[frame.frame_id] = {"template_id": template, "valid": valid, "head_informative": head_informative, "plural_head_change": plural.head_change, "cue_effect_positive": positive, "cue_effect_required": required,
                                      "template_defined": True, "reconstruction_error": state_f.state.ref.reconstruction_error, "p_c": frame.p_c, "p_t": frame.p_t, "cue_final": frame.p_t == frame.p_c}
        locked = hp.locked_state(state_f)
        states[frame.frame_id] = locked
        scores = frame_ranking(masked, weights, parts["read_out"], locked, template, by_template[template])
        frame_subsets[frame.frame_id] = frame_subset(scores)
        frame_overlaps[frame.frame_id] = {HYPOTHESIS: overlap_with(frame_subsets[frame.frame_id], lock["subsets"][HYPOTHESIS]), "S64": overlap_with(frame_subset(scores, 64), lock["subsets"]["S64"])}
        oracle = mask_of(frame_subsets[frame.frame_id], masked.n_neurons)
        for token in confirmation.tokens:
            rows_out.append(prediction_row(token["word"], frame.frame_id, template, masked.predict_from_state(weights, state_f, token["token_id"], template, oracle)))
        say(f"  {frame.frame_id}: {'valid' if valid else 'INVALID'} (head informative {head_informative}, cue effect {positive}/{required}); p_c {frame.p_c}, p_t {frame.p_t}; own top-256 shares {frame_overlaps[frame.frame_id][HYPOTHESIS]} with S_256; {len(confirmation.tokens)} predictions")
    return {"frames": frames_out, "states": states, "state_digests": {frame_id: ap.state_digest(entry) for frame_id, entry in states.items()}, "frame_subsets": frame_subsets, "frame_overlaps": frame_overlaps, "rows": rows_out, "identities": identities,
            "token_means": token_means_from_table(rows_out, [token["word"] for token in confirmation.tokens]),
            "digest": stage_digest(rows_out, states, frame_subsets), "commit": protocol_code_commit, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}


def stage_two(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation018, lock: Mapping[str, Any], lock_012: Mapping[str, Any], stage1: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    assert_stage_one_digest(stage1)
    base2_pt = torch.tensor(lock["base2_pt"]["vector"], dtype=torch.float64) if lock["base2_pt"]["vector"] is not None else None
    parts = _components(model, pool, pool_010, lock, lock_012, hp.bases_from_json(lock["bases_3"]), base2_pt, lock["subsets"])
    weights, head, axis_T, e_axis, masked, context = parts["weights"], parts["head"], parts["axis_T"], parts["e_axis"], parts["masked"], parts["context_018"]
    nouns = pool.single_nouns
    identities: dict[str, float] = {}
    say("stage 2: fresh cues in the exposed frames (Y1) and in the valid fresh frames (Y2)")
    pairs_exposed: dict[str, dict[str, Any]] = {}
    for frame in pool.frames:
        if frame.template_id not in lock["defined_templates"]:
            continue
        state_f = hp.capture_frame_017(model, head, pool.reference_prompt(frame), nouns, axis_T)
        hp._check_locked_state_017(state_f, lock["locked_states"][frame.frame_id], frame.frame_id)
        rows16 = atp.reference_rows(masked.hcm.fcm.programs, state_f.state.x1_all, state_f.state.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows16, state_f.state))
        identities["I5_reference_head_row"] = max(identities.get("I5_reference_head_row", 0.0), hp.check_reference_row(state_f, masked.hcm.program3))
        oracle = mask_of(lock["frame_subsets"][frame.frame_id], masked.n_neurons)
        plural_name = pool.plural_cue[frame.template_id]
        plural = hp.measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = hp.measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_018(record, plural, context=context, state=state_f, oracle_mask=oracle)
            if analysis is None:
                raise pm.IncidentError(f"{frame.frame_id}/{token['word']}: no analysis although the exposed frame is informative")
            er._max_errors(identities, analysis["identities"])
            pairs_exposed[f"{token['word']}|{frame.frame_id}"] = analysis
        say(f"  {frame.frame_id} (exposed): {len(confirmation.tokens)} tokens")
    pairs_fresh: dict[str, dict[str, Any]] = {}
    for frame in confirmation.frames:
        entry = stage1["frames"][frame.frame_id]
        if not entry.get("template_defined") or not entry["valid"]:
            continue
        state_f = hp.capture_frame_017(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        if ap.state_digest(hp.locked_state(state_f)) != stage1["state_digests"][frame.frame_id]:
            raise pm.IncidentError(f"{frame.frame_id}: the re-captured reference state differs from stage 1's digested state")
        oracle = mask_of(stage1["frame_subsets"][frame.frame_id], masked.n_neurons)
        sg, pl = pm.frame_prompts(frame)
        plural = hp.measure_pair(model, weights, head, state_f, pool.plural_cue[frame.template_id], pl.cue_token_id, e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = hp.measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_018(record, plural, context=context, state=state_f, oracle_mask=oracle)
            if analysis is None:
                raise pm.IncidentError(f"{frame.frame_id}/{token['word']}: no analysis although the frame is valid")
            er._max_errors(identities, analysis["identities"])
            pairs_fresh[f"{token['word']}|{frame.frame_id}"] = analysis
        say(f"  {frame.frame_id} (fresh): {len(confirmation.tokens)} tokens")
    results = score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    results["per_frame_exposed"] = {key: compact_analysis(value) for key, value in pairs_exposed.items()}
    results["per_frame_fresh"] = {key: compact_analysis(value) for key, value in pairs_fresh.items()}
    er._max_errors(identities, stage1.get("identities", {}))
    results["identities"] = identities
    return results


def _rows_by_pair(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {f"{row['token']}|{row['frame_id']}": row for row in rows}


def _score_set(pairs: Mapping[str, Mapping[str, Any]], table: Mapping[str, Mapping[str, Any]], words: Sequence[str], *, fresh: bool, valid_frames: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Token means and pooled statistics over the scored pairs; the reference-rung precondition, the gaps, the two κ floors and, for Y2, the split guard and the no-harm guard."""
    tokens_out, scored_pairs = {}, []
    for word in words:
        keys = sorted(key for key in pairs if key.split("|")[0] == word and key in table)
        analyses = [pairs[key] for key in keys]
        predictions = [table[key] for key in keys]
        tokens_out[word] = {"scored": len(keys) >= MIN_VALID_FRAMES_PER_TOKEN, "n_valid_frames": len(keys), **_token_means(analyses, predictions)}
        if len(keys) >= MIN_VALID_FRAMES_PER_TOKEN:
            scored_pairs.extend(zip(analyses, predictions))
    scored = [word for word, entry in tokens_out.items() if entry["scored"]]
    every_pair = [(pairs[key], table[key]) for key in sorted(pairs) if key in table]
    result: dict[str, Any] = {"tokens": tokens_out, "scored_tokens": scored, "n_pairs": len(scored_pairs), "per_frame": per_frame_statistics([a for a, _ in every_pair], [p for _, p in every_pair])}
    if len(scored) < 2:
        return result
    analyses = [a for a, _ in scored_pairs]
    predictions = [p for _, p in scored_pairs]
    result["per_frame_scored"] = per_frame_statistics(analyses, predictions)
    stats = rung_statistics(analyses, predictions)
    means = token_mean_statistics(tokens_out, scored)
    split = split_statistics(analyses, predictions)
    reference = {"c_L": stats["rungs"][REFERENCE]["c_L"]["r2"], "rows": stats["rungs"][REFERENCE]["rows"]["entry_r2"], "dT": stats["rungs"][REFERENCE]["dT"]["r2"]}
    pre_checks = {f"reference_{obj}": reference[obj] is not None and reference[obj] >= PRECONDITION_REFERENCE[obj] for obj in PRECONDITION_REFERENCE}
    pre_checks.update({f"gap_{obj}": stats["evaluable"][obj] for obj in FLOOR_OBJECTS})
    if fresh:
        for name in ("cue_final", "coordinated"):
            pre_checks[f"split_gap_{name}"] = bool(split[name].get("n_pairs") and split[name]["evaluable_c_L"])
    precondition = {"reference": reference, "gaps": {obj: stats["gaps"][obj] for obj in (*OBJECTS, "rows")}, "checks": pre_checks, "failing": [name for name, ok in pre_checks.items() if not ok]}
    k = stats["kappa"]
    checks = [("kappa_c_L", k["c_L"][HYPOTHESIS] is not None and k["c_L"][HYPOTHESIS] >= KAPPA_CL_FLOOR), ("kappa_Pi", k["Pi"][HYPOTHESIS] is not None and k["Pi"][HYPOTHESIS] >= KAPPA_PI_FLOOR)]
    test: dict[str, Any] = {"kappa_c_L": k["c_L"][HYPOTHESIS], "kappa_Pi": k["Pi"][HYPOTHESIS], "floors": {"kappa_c_L": KAPPA_CL_FLOOR, "kappa_Pi": KAPPA_PI_FLOOR}}
    if fresh:
        guard = {name: {"kappa_c_L": split[name].get("kappa_c_L"), "floor": SPLIT_GUARD[name], "n_frames": split[name].get("n_frames", 0), "passed": split[name].get("kappa_c_L") is not None and split[name]["kappa_c_L"] >= SPLIT_GUARD[name]} for name in SPLIT_GUARD}
        test["split_guard"] = guard
        checks += [(f"split_guard_{name}", entry["passed"]) for name, entry in guard.items()]
        scored_frames = result["per_frame_scored"]
        below = {frame_id: entry["no_harm_difference"] for frame_id, entry in scored_frames.items() if entry["no_harm_difference"] is None or entry["no_harm_difference"] < -NO_HARM_MARGIN}
        test["no_harm_guard"] = {"margin": NO_HARM_MARGIN, "per_frame": {frame_id: entry["no_harm_difference"] for frame_id, entry in scored_frames.items()}, "frames_below": below, "passed": not below}
        checks.append(("no_harm_guard", not below))
    failing = [name for name, ok in checks if not ok]
    test.update({"passed": not failing, "failing": failing})
    result.update({"precondition": precondition, "test": test, "statistics": stats, "token_means_statistics": means, "split": split, "orderings": predeclared_orderings(stats),
                   "single": {"kappa_c_L": k["c_L"][SINGLE], "kappa_Pi": k["Pi"][SINGLE], "gap_to_hypothesis": (k["c_L"][HYPOTHESIS] - k["c_L"][SINGLE]) if k["c_L"][HYPOTHESIS] is not None and k["c_L"][SINGLE] is not None else None}})
    return result


def score_y4(per_frame: Mapping[str, Mapping[str, Any]], frames: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The secondary question from the reference rung: per valid fresh cue-final frame, the frozen pattern's ΔT R² below Y4_FROZEN_MAX wherever the reference rung reaches Y4_REFERENCE_MIN."""
    cue_final, coordinated = {}, {}
    for frame_id, entry in per_frame.items():
        meta = frames.get(frame_id, {})
        if not meta.get("valid"):
            continue
        record = {"reference_dT_r2": entry["dT"][REFERENCE], "frozen_dT_r2": entry["dT_frozen"], "n_pairs": entry["n_pairs"]}
        if entry["cue_final"]:
            record["evaluable"] = record["reference_dT_r2"] is not None and record["reference_dT_r2"] >= Y4_REFERENCE_MIN
            record["below"] = bool(record["evaluable"] and record["frozen_dT_r2"] is not None and record["frozen_dT_r2"] < Y4_FROZEN_MAX)
            cue_final[frame_id] = record
        else:
            coordinated[frame_id] = record
    evaluable = [frame_id for frame_id, record in cue_final.items() if record["evaluable"]]
    at_or_above = [frame_id for frame_id in evaluable if not cue_final[frame_id]["below"]]
    return {"cue_final": cue_final, "coordinated": coordinated, "evaluable_frames": evaluable, "n_evaluable": len(evaluable), "min_evaluable": Y4_MIN_EVALUABLE, "evaluable": len(evaluable) >= Y4_MIN_EVALUABLE,
            "frames_at_or_above": at_or_above, "passed": len(evaluable) >= Y4_MIN_EVALUABLE and not at_or_above, "floors": {"frozen_max": Y4_FROZEN_MAX, "reference_min": Y4_REFERENCE_MIN}}


def score_confirmation(stage1: Mapping[str, Any], pairs_exposed: Mapping[str, Mapping[str, Any]], pairs_fresh: Mapping[str, Mapping[str, Any]], confirmation: Confirmation018, lock: Mapping[str, Any]) -> dict[str, Any]:
    words = [token["word"] for token in confirmation.tokens]
    locked_table = _rows_by_pair(lock["predictions"]["rows"])
    stage1_table = _rows_by_pair(stage1["rows"])
    frames = stage1["frames"]
    valid_frames = [frame_id for frame_id, entry in frames.items() if entry.get("valid")]
    valid_cue_final = [frame_id for frame_id in valid_frames if frames[frame_id].get("cue_final")]
    valid_coordinated = [frame_id for frame_id in valid_frames if not frames[frame_id].get("cue_final")]
    y1 = _score_set(pairs_exposed, locked_table, words, fresh=False)
    y2 = _score_set(pairs_fresh, stage1_table, words, fresh=True)
    results: dict[str, Any] = {"frames": frames, "valid_frames": valid_frames, "Y1": y1, "Y2": y2}
    pre1 = {"scored_tokens": len(y1["scored_tokens"]), "min_scored_tokens": MIN_SCORED_TOKENS, "checks": {"scored_tokens": len(y1["scored_tokens"]) >= MIN_SCORED_TOKENS, **y1.get("precondition", {}).get("checks", {})}}
    pre1["failing"] = [name for name, ok in pre1["checks"].items() if not ok]
    pre1["passed"] = not pre1["failing"] and "test" in y1
    pre2 = {"scored_tokens": len(y2["scored_tokens"]), "min_scored_tokens": MIN_SCORED_TOKENS, "valid_cue_final": len(valid_cue_final), "valid_coordinated": len(valid_coordinated),
            "min_valid_cue_final": MIN_VALID_CUE_FINAL_FRAMES, "min_valid_coordinated": MIN_VALID_COORDINATED_FRAMES,
            "checks": {"scored_tokens": len(y2["scored_tokens"]) >= MIN_SCORED_TOKENS, "valid_cue_final": len(valid_cue_final) >= MIN_VALID_CUE_FINAL_FRAMES, "valid_coordinated": len(valid_coordinated) >= MIN_VALID_COORDINATED_FRAMES,
                       **y2.get("precondition", {}).get("checks", {})}}
    pre2["failing"] = [name for name, ok in pre2["checks"].items() if not ok]
    pre2["passed"] = not pre2["failing"] and "test" in y2
    results["precondition_Y1"], results["precondition_Y2"] = pre1, pre2
    # Y3: the single-neuron account on the pooled (pair-weighted) scored pairs of both sets; per-set values descriptive.
    pair_items = []
    for y, pairs, table in ((y1, pairs_exposed, locked_table), (y2, pairs_fresh, stage1_table)):
        for w in y["scored_tokens"]:
            pair_items.extend((pairs[k], table[k]) for k in sorted(key for key in pairs if key.split("|")[0] == w and key in table))
    pooled = rung_statistics([a for a, _ in pair_items], [p for _, p in pair_items], rungs=(*LADDER_RUNGS, *RANDOM_RUNGS, BOTTOM_RUNG)) if len(pair_items) >= 2 else None
    k_single = pooled["kappa"]["c_L"][SINGLE] if pooled else None
    k_hyp = pooled["kappa"]["c_L"][HYPOTHESIS] if pooled else None
    evaluable = bool(pre1["passed"] and pre2["passed"] and pooled is not None and pooled["evaluable"]["c_L"] and k_single is not None and k_hyp is not None)
    rejected = bool(evaluable and k_single < Y3_KAPPA_MAX and (k_hyp - k_single) >= Y3_MARGIN)
    results["Y3"] = {"evaluable": evaluable, "rejected": rejected, "n_pairs": len(pair_items), "n_pairs_Y1": y1["n_pairs"], "n_pairs_Y2": y2["n_pairs"], "kappa_c_L_single": k_single, "kappa_c_L_hypothesis": k_hyp,
                     "gap": (k_hyp - k_single) if k_hyp is not None and k_single is not None else None, "kappa_Pi_single": pooled["kappa"]["Pi"][SINGLE] if pooled else None,
                     "pooled_gap_c_L": pooled["gaps"]["c_L"] if pooled else None, "floors": {"kappa_max": Y3_KAPPA_MAX, "margin": Y3_MARGIN},
                     "reading": "rejection on the pair-weighted combined prospective distribution (dominated by the cue-new, exposed-frame pairs), not separately in each generalization dimension",
                     "per_set": {"Y1": y1.get("single"), "Y2": y2.get("single")}}
    results["ladder_both_sets"] = pooled
    results["Y4"] = score_y4(y2.get("per_frame_scored", {}), frames)
    results["outcome"] = outcome(results)
    return results


def outcome(results: Mapping[str, Any]) -> dict[str, Any]:
    def family(labels: Sequence[str], precondition: Mapping[str, Any], y: Mapping[str, Any]) -> str:
        if not precondition["passed"]:
            return labels[2]
        return labels[0] if y.get("test", {}).get("passed") else labels[1]

    y1 = family(OUTCOME_Y1, results["precondition_Y1"], results["Y1"])
    y2 = family(OUTCOME_Y2, results["precondition_Y2"], results["Y2"])
    y3_entry = results["Y3"]
    y3 = OUTCOME_Y3[2] if not y3_entry.get("evaluable") else (OUTCOME_Y3[0] if y3_entry["rejected"] else OUTCOME_Y3[1])
    y4_entry = results["Y4"]
    y4 = OUTCOME_Y4[2] if not y4_entry.get("evaluable") else (OUTCOME_Y4[0] if y4_entry["passed"] else OUTCOME_Y4[1])
    return {"label": f"{y1} | {y2} | {y3} | {y4}", "Y1": y1, "Y2": y2, "Y3": y3, "Y4": y4}


# ---------------------------------------------------------------------------
# Report.


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 018 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state['confirmation_018_sha256']}`", f"- Experiment 017 lock sha256: `{state['lock_017_sha256']}`",
             f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    incidents = list(exploration.get("incidents", [])) + ([state["confirmation"]["incident"]] if isinstance(state.get("confirmation"), dict) and "incident" in state["confirmation"] else [])
    if incidents:
        lines += ["## Incidents", ""] + [f"- `{e['phase']}` at commit `{e.get('commit', '?')}` ({e['at']}): {e['message']}" for e in incidents] + [""]

    def ladder_lines(stats: Mapping[str, Any], label: str) -> list[str]:
        out = [f"  - {label} ({stats['n_pairs']} pairs) — rung: c_L R² (κ) / F R² (κ) / Π R² (κ) / ΔT R² (κ) / rows R² (κ):"]
        for rung in (*RUNGS, ORACLE):
            if rung not in stats["rungs"]:
                continue
            r = stats["rungs"][rung]
            k = stats["kappa"]
            out.append(f"    - `{rung}`: {f(r['c_L']['r2'], 3)} ({f(k['c_L'][rung], 2)}) / {f(r['F']['r2'], 3)} ({f(k['F'][rung], 2)}) / {f(r['Pi']['r2'], 3)} ({f(k['Pi'][rung], 2)}) / {f(r['dT']['r2'], 3)} ({f(k['dT'][rung], 2)}) / {f(r['rows']['entry_r2'], 3)} ({f(k['rows'][rung], 2)})")
        out.append(f"    - gaps: {', '.join(f'{obj} {f(v, 3)}' for obj, v in stats['gaps'].items())}; frozen pattern ΔT R² {f(stats['frozen']['r2'], 3)}; Experiment 016's decoded c_L R² {f(stats['c_L_level0F']['r2'], 3)}")
        return out

    def split_line(s: Mapping[str, Any]) -> str:
        if not s.get("n_pairs"):
            return "no pairs"
        return f"{s['n_pairs']} pairs, {s['n_tokens']} tokens, {s['n_frames']} frames: c_L R² S_0 {f(s['c_L_r2'][TEMPLATE_BASE], 3)} / S_256 {f(s['c_L_r2'][HYPOTHESIS], 3)} / S_2048 {f(s['c_L_r2'][REFERENCE], 3)}, gap {f(s['gap_c_L'], 3)}, κ_c_L {f(s['kappa_c_L'], 2)} (S_1 {f(s['kappa_c_L_single'], 2)}); Π κ {f(s['kappa_Pi'], 2)} (gap {f(s['gap_Pi'], 3)}); frozen ΔT R² {f(s['frozen_dT_r2'], 3)} vs reference {f(s['reference_dT_r2'], 3)}"

    def orderings_lines(o: Mapping[str, Any]) -> list[str]:
        return [f"  - predeclared orderings: row diffuse κ_row(S_256) {f(o['row_diffuse']['kappa_row'], 2)} < {ROW_DIFFUSE_MAX} holds {o['row_diffuse']['holds']}; random margin κ_c_L {f(o['random_margin']['kappa_c_L'], 2)} vs best random {f(o['random_margin']['best_random'], 2)} / bottom {f(o['random_margin']['bottom'], 2)} holds {o['random_margin']['holds']}; "
                f"κ_c_L ≥ κ_Π holds {o['c_L_over_Pi']['holds']}"]

    def frame_lines(per_frame: Mapping[str, Any]) -> list[str]:
        return [f"  - frame {frame_id} ({entry['template']}): {entry['n_pairs']} pairs; c_L R² S_0 {f(entry['c_L'][TEMPLATE_BASE], 3)} / S_1 {f(entry['c_L'][SINGLE], 3)} / S_256 {f(entry['c_L'][HYPOTHESIS], 3)} / S_2048 {f(entry['c_L'][REFERENCE], 3)} / oracle {f(entry['c_L'][ORACLE], 3)} (κ_c_L {f(entry['kappa_c_L'], 2)}, no-harm {f(entry['no_harm_difference'], 3)}); "
                f"rows S_256 {f(entry['rows'][HYPOTHESIS], 3)} / S_2048 {f(entry['rows'][REFERENCE], 3)}; Π S_256 {f(entry['Pi'][HYPOTHESIS], 3)} / S_2048 {f(entry['Pi'][REFERENCE], 3)}; ΔT reference {f(entry['dT'][REFERENCE], 3)}, frozen {f(entry['dT_frozen'], 3)}" for frame_id, entry in per_frame.items()]

    if "summary" in exploration:
        x = exploration["exposed_check"]
        rep = exploration["replication"]["experiment_017"]
        lines += ["## Tier A — exposed pool (calibration record only; the floors are frozen constants)", "",
                  f"- Replication of Experiment 017: {rep['n']} pairs, max deviation {rep['max_abs_deviation']:.2e}",
                  f"- Identities (checks only): {', '.join(f'{k} {v:.1e}' for k, v in exploration['identities'].items())}; read weight vs Experiment 011 lock {exploration['read_weight_check']:.1e}",
                  f"- Ranking pool: {exploration['ranking_pool']['n_pairs']} pairs ({exploration['ranking_pool']['n_records']} records); top neurons {order_of(torch.tensor(exploration['scores'], dtype=torch.float64))[:10]}; S_1 {exploration['subsets'][SINGLE]}; overlaps {exploration['overlaps']}; "
                  f"per-frame top-256 overlap with S_256 min/median/max {min(v[HYPOTHESIS] for v in exploration['frame_overlaps'].values())}/{sorted(v[HYPOTHESIS] for v in exploration['frame_overlaps'].values())[len(exploration['frame_overlaps']) // 2]}/{max(v[HYPOTHESIS] for v in exploration['frame_overlaps'].values())}",
                  f"- Block-2 base at p_t over {exploration['base2_pt']['n_frames']} coordinated frames; program: {', '.join(f'layer {k}: d_head {v['d_head']}, rotary_dim {v['rotary_dim']}, base {v['rotary_base']:g}' for k, v in exploration['program'].items())}"]
        lines += ladder_lines(x["pairs"], "pooled") + orderings_lines(x["orderings"])
        lines += [f"- Split — cue-final: {split_line(x['split']['cue_final'])}", f"- Split — coordinated: {split_line(x['split']['coordinated'])}"]
        lines += [f"- Template {template}: {split_line(entry)}" for template, entry in x["split"]["per_template"].items()]
        lines += [""]
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}`; predictions sha256 `{state['lock']['predictions_sha256']}`", ""]
    confirmation = state.get("confirmation")
    if confirmation and "stage1" in confirmation:
        s1 = confirmation["stage1"]
        lines += ["## Confirmation — stage 1 (fresh frames' reference states; prediction table digested before any fresh cue prompt)", "", f"- Table rows {len(s1['rows'])}; digest `{s1['digest']}`; commit `{s1['commit']}`"]
        for frame_id, entry in s1["frames"].items():
            if entry.get("template_defined"):
                lines.append(f"  - {frame_id}: {'valid' if entry['valid'] else 'INVALID'} (plural head change {f(entry['plural_head_change'], 3)}, cue effect {entry['cue_effect_positive']}/{entry['cue_effect_required']}; p_c {entry['p_c']}, p_t {entry['p_t']}); own top-256 overlap with S_256 {s1['frame_overlaps'].get(frame_id, {}).get(HYPOTHESIS)}")
            else:
                lines.append(f"  - {frame_id}: {entry.get('note')}")
        lines.append("")
    if confirmation and "outcome" in confirmation:
        o = confirmation["outcome"]
        lines += [f"## Confirmation — stage 2 — `{o['label']}`", ""]
        for label, key in (("Y1 (strict prospective: fresh cues × exposed frames)", "Y1"), ("Y2 (frame-conditional prospective with the split and no-harm guards: fresh cues × new frames)", "Y2")):
            y = confirmation[key]
            pre = confirmation["precondition_Y1" if key == "Y1" else "precondition_Y2"]
            if "test" in y:
                t = y["test"]
                guard_text = ""
                if "split_guard" in t:
                    guard_text = "; split guard " + ", ".join(f"{name} κ_c_L {f(entry['kappa_c_L'], 2)} (≥ {entry['floor']}, {entry['n_frames']} frames) {'ok' if entry['passed'] else 'FAILED'}" for name, entry in t["split_guard"].items())
                    guard_text += f"; no-harm guard {'passed' if t['no_harm_guard']['passed'] else 'FAILED ' + str(t['no_harm_guard']['frames_below'])}"
                lines.append(f"- {label}: scored tokens {len(y['scored_tokens'])} ({'precondition ok' if pre['passed'] else 'PRECONDITION FAILED ' + str(pre['failing'])}); κ_c_L(S_256) {f(t['kappa_c_L'], 3)} (≥ {KAPPA_CL_FLOOR}), κ_Π(S_256) {f(t['kappa_Pi'], 3)} (≥ {KAPPA_PI_FLOOR}){guard_text} → {'pass' if t['passed'] else 'FAIL'} {t['failing'] or ''}")
                lines.append(f"  - reference rung: c_L R² {f(y['precondition']['reference']['c_L'], 3)}, rows {f(y['precondition']['reference']['rows'], 3)}, ΔT {f(y['precondition']['reference']['dT'], 3)}; gaps {', '.join(f'{obj} {f(v, 3)}' for obj, v in y['precondition']['gaps'].items())}; single neuron κ_c_L {f(y['single']['kappa_c_L'], 2)}, gap to S_256 {f(y['single']['gap_to_hypothesis'], 2)}")
                lines += ladder_lines(y["statistics"], "pooled pairs") + orderings_lines(y["orderings"])
                tm = y["token_means_statistics"]
                if tm.get("kappa"):
                    lines.append(f"  - token means: κ_c_L(S_256) {f(tm['kappa']['c_L'][HYPOTHESIS], 2)}, κ_Π {f(tm['kappa']['Pi'][HYPOTHESIS], 2)}, κ_F {f(tm['kappa']['F'][HYPOTHESIS], 2)}, κ_ΔT {f(tm['kappa']['dT'][HYPOTHESIS], 2)}; reference c_L R² {f(tm['rungs'][REFERENCE]['c_L']['r2'], 3)}")
                lines.append(f"  - split — cue-final: {split_line(y['split']['cue_final'])}; coordinated: {split_line(y['split']['coordinated'])}")
                lines += [f"  - template {template}: {split_line(entry)}" for template, entry in y["split"]["per_template"].items()]
            else:
                lines.append(f"- {label}: not evaluable ({len(y['scored_tokens'])} scored tokens; {'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'})")
            if key == "Y2":
                lines += frame_lines(y.get("per_frame", {}))
                for frame_id, entry in confirmation["frames"].items():
                    if entry.get("template_defined") and not entry.get("valid"):
                        lines.append(f"  - frame {frame_id}: invalid at stage 1 (no fresh cue prompt run)")
        y3 = confirmation["Y3"]
        lines.append(f"- Y3 (single neuron on the pooled pairs of both sets, {y3['n_pairs']} pairs = {y3['n_pairs_Y1']} + {y3['n_pairs_Y2']}): κ_c_L(S_1) {f(y3['kappa_c_L_single'], 3)} (rejected iff < {Y3_KAPPA_MAX}) against S_256 {f(y3['kappa_c_L_hypothesis'], 3)}, gap {f(y3['gap'], 3)} (rejected iff ≥ {Y3_MARGIN}); pooled c_L gap {f(y3['pooled_gap_c_L'], 3)}; κ_Π(S_1) {f(y3['kappa_Pi_single'], 2)} → "
                     f"{'REJECTED' if y3['rejected'] else ('NOT REJECTED' if y3['evaluable'] else 'NOT EVALUABLE')}; per set: Y1 {f((y3['per_set']['Y1'] or {}).get('kappa_c_L'), 2)} / gap {f((y3['per_set']['Y1'] or {}).get('gap_to_hypothesis'), 2)}, Y2 {f((y3['per_set']['Y2'] or {}).get('kappa_c_L'), 2)} / gap {f((y3['per_set']['Y2'] or {}).get('gap_to_hypothesis'), 2)}")
        y4 = confirmation["Y4"]
        lines.append(f"- Y4 (secondary, reference rung only; {y4['n_evaluable']} of the valid fresh cue-final frames evaluable, ≥ {Y4_MIN_EVALUABLE} required): frozen ΔT R² below {Y4_FROZEN_MAX} in every evaluable frame → {'PASS' if y4['passed'] else ('FAIL ' + str(y4['frames_at_or_above']) if y4['evaluable'] else 'NOT EVALUABLE')}")
        lines += [f"  - {frame_id}: reference ΔT R² {f(r['reference_dT_r2'], 3)}, frozen {f(r['frozen_dT_r2'], 3)} ({'evaluable' if r['evaluable'] else 'not evaluable'}{', below' if r.get('below') else ''})" for frame_id, r in y4["cue_final"].items()]
        lines += [f"  - {frame_id} (coordinated, descriptive): reference ΔT R² {f(r['reference_dT_r2'], 3)}, frozen {f(r['frozen_dT_r2'], 3)}" for frame_id, r in y4["coordinated"].items()]
        if confirmation.get("ladder_both_sets"):
            lines += ladder_lines(confirmation["ladder_both_sets"], "both sets pooled")
        lines += ["", f"| token | class | Y1 frames | Y1 ĉ_L S_256 | Y1 ĉ_L S_2048 | Y1 c_L | Y1 ΔT̂ S_256 | Y1 ΔT | Y2 frames | Y2 ĉ_L S_256 | Y2 ĉ_L S_2048 | Y2 c_L | Y2 ΔT̂ S_256 | Y2 ΔT |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        categories = {token["word"]: token["category"] for token in confirmation.get("tokens_meta", [])}
        for word in sorted(confirmation["Y1"]["tokens"], key=lambda w: -(confirmation["Y1"]["tokens"][w].get(f"dT_{REFERENCE}_mean") or -9)):
            a, b = confirmation["Y1"]["tokens"][word], confirmation["Y2"]["tokens"].get(word, {})
            lines.append(f"| {word} | {categories.get(word, '')} | {a.get('n_valid_frames')} | {f(a.get(f'c_L_{HYPOTHESIS}_mean'), 4)} | {f(a.get(f'c_L_{REFERENCE}_mean'), 4)} | {f(a.get('c_L_mean'), 4)} | {f(a.get(f'dT_{HYPOTHESIS}_mean'), 4)} | {f(a.get('dT_mean'), 4)} | "
                         f"{b.get('n_valid_frames')} | {f(b.get(f'c_L_{HYPOTHESIS}_mean'), 4)} | {f(b.get(f'c_L_{REFERENCE}_mean'), 4)} | {f(b.get('c_L_mean'), 4)} | {f(b.get(f'dT_{HYPOTHESIS}_mean'), 4)} | {f(b.get('dT_mean'), 4)} |")
        lines.append("")
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
