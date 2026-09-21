"""Experiment 019: is the frame-specific membership of block 2's channel-D recovery subset predicted from the frame's reference state?

Experiment 018 showed that channel D — block 2's MLP at the frame's own operating point, the least compact object of the
decoded cue-to-transport program — is carried for unseen cues by a locked subset of 256 neurons, and, descriptively,
that *which* neurons carry it depends on the frame. This module tests whether that redistribution is predictable: a
frame-specific subset of matched size, selected from the frame's reference state before any fresh cue is run in it,
against the matched global subset of the same size. Every selector is a frozen rule applied to locked reference
states, the weights and the exposed cues' weight-only encoding changes; every subset is an explicit index list. The
masked chain of Experiment 018 is used with one mask per changed position. Two post-confirmation objects enter no
selector: the measured-effect ranking oracle ``O_k`` (descriptive; the target of the membership question) and the
leave-one-cue-out greedy ``O*_k`` — a cross-validated empirical headroom witness, never a global optimum, ceiling or
upper bound — whose headroom ``H*`` classifies only a negative routing result. Constants are copied from design
revision 3 (commit b130b50; wording efe0623).
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import attention_paths as ap
from . import attention_patterns as atp
from . import block_concentration as bc
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
# Frozen protocol constants (design revision 3).

EXPERIMENT_DIR = "experiments/019-block2-routing"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
INHERITED_018_EXTRACT_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-018-pair-extract.json"
EXPERIMENT_018_LOCK_PATH = bc.LOCK_RELATIVE_PATH
EXPERIMENT_018_CONFIRMATION_PATH = bc.CONFIRMATION_RELATIVE_PATH
RUNTIME_SEED = bc.RUNTIME_SEED
CONTROL_SEED = bc.CONTROL_SEED
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
INHERITED_SCHEMA_VERSION = 1
EXTRACTION_SCHEMA_VERSION = 1
PHASES = ("explore", "lock", "confirm", "report")

N_NEURONS = bc.N_NEURONS
BLOCK = bc.BLOCK
SIZES = (16, 64, 256)  # the subset sizes; every selector is evaluated at each
DECISION_SIZE = 64  # the decision size of Y1–Y5
DRIVE_QUANTILES = (0.1, 0.5, 0.9)  # the frozen drive profile of G: three quantiles per neuron, template and position type
N_RANDOM_CONTROLS = 3
POSITION_TYPES = ("p_c", "p_t")

ROUTING_GAIN_FLOOR = 0.05  # Y1/Y2: Δκ_64(E) on the pooled scored pairs of the set
SPLIT_GAIN_FLOOR = 0.02  # Y1/Y2: Δκ_64(E) on the cue-final split and on the coordinated split
RHO_POOLED_FLOOR = 0.60  # Y3: ρ_64 on the scored pairs of both sets pooled …
RHO_SET_FLOOR = 0.50  # … and on each set
RHO_DENOMINATOR_MIN = 0.05  # Y3: Δκ_64(E) on each set and pooled, else not evaluable
TEMPLATE_MARGIN = 0.05  # Y4: the margins of A = κ(E) − κ(T) and B = Δκ(T), pooled and on Y2
MEMBERSHIP_FLOOR = 0.60  # Y5: mean |E_64 ∩ O_64| / 64 per set …
MEMBERSHIP_MARGIN = 0.20  # … and at least this above the same for S'_64
HEADROOM_MIN = 0.05  # the routing headroom H*_64 below which a negative routing result is NOT_EVALUABLE (with E's gain also below the floor)
GAP_MIN = bc.GAP_MIN
PRECONDITION_REFERENCE = bc.PRECONDITION_REFERENCE
OTHER_SIZE_GAIN_EXPECTATION = 0.03  # descriptive (i): Δκ_16(E) and Δκ_256(E) on each set
RANDOM_KAPPA_MAX = 0.50  # descriptive (iii): every random control below this at every k …
RANDOM_MARGIN = 0.30  # … and every prospective selector above the best random control by at least this at k = 64
ROW_DIFFUSE_MAX = 0.50  # descriptive (iv): κ_row below this at k ≤ 256
MIN_VALID_CUE_FINAL_FRAMES = 9
MIN_VALID_COORDINATED_FRAMES = 4
MIN_VALID_FRAMES_PER_TOKEN = 3
MIN_SCORED_TOKENS = 16
FRAME_CUE_EFFECT_RATE = hp.FRAME_CUE_EFFECT_RATE
I8_TOLERANCE = 1e-9  # the reference rung against Experiment 017's Level 0 / Experiment 018's S2048 record
I10_TOLERANCE = 1e-9  # the read identity ĉ_L(S) = ĉ_L(S_0) + Σ u_j
LOCK_PREDICTION_TOLERANCE = 1e-9
REPLICATION_TOLERANCE = 1e-6
LOCKED_STATE_TOLERANCE = 1e-9
EXPECTED_EXTRACT_SIZE_018 = 11796  # Experiment 018's 9636 exposed pairs + its 1872 Y1 + 288 Y2 pairs
EXTRACT_SET_SIZES_018 = (9636, 1872, 288)
N_018_STAGE1_FRAMES = 12
N_018_LOCKED_FRAMES = 78
HEAD_LAYER, HEAD_INDEX, HEAD_KEY = hp.HEAD_LAYER, hp.HEAD_INDEX, hp.HEAD_KEY
PROGRAM_LAYERS = hp.PROGRAM_LAYERS
OBJECTS = ("c_L", "F", "Pi", "dT")

# Rungs. Fixed lists: S0, S2048, Sp1, Sp{k}, L{k} (the inherited Experiment 018 S_k), R{i}_{k}; per frame and position: E1, E{k}, G{k}; per template and position type: T{k}.
TEMPLATE_BASE, REFERENCE, GLOBAL_SINGLE, FRAME_SINGLE = "S0", "S2048", "Sp1", "E1"
GLOBAL_RUNGS = tuple(f"Sp{k}" for k in SIZES)
TEMPLATE_RUNGS = tuple(f"T{k}" for k in SIZES)
FULL_RUNGS = tuple(f"E{k}" for k in SIZES)
RULE_RUNGS = tuple(f"G{k}" for k in SIZES)
INHERITED_RUNGS = tuple(f"L{k}" for k in SIZES)
RANDOM_RUNGS = tuple(f"R{i}_{k}" for k in SIZES for i in range(1, N_RANDOM_CONTROLS + 1))
PROSPECTIVE_RUNGS = (TEMPLATE_BASE, REFERENCE, GLOBAL_SINGLE, FRAME_SINGLE, *GLOBAL_RUNGS, *TEMPLATE_RUNGS, *FULL_RUNGS, *RULE_RUNGS, *INHERITED_RUNGS, *RANDOM_RUNGS)
RANKING_ORACLE_RUNGS = tuple(f"O{k}" for k in SIZES)
WITNESS_RUNGS = tuple(f"Os{k}" for k in SIZES)
ORACLE_RUNGS = (*RANKING_ORACLE_RUNGS, *WITNESS_RUNGS)
ALL_RUNGS = (*PROSPECTIVE_RUNGS, *ORACLE_RUNGS)
DECISION = {"global": f"Sp{DECISION_SIZE}", "template": f"T{DECISION_SIZE}", "full": f"E{DECISION_SIZE}", "rule": f"G{DECISION_SIZE}", "inherited": f"L{DECISION_SIZE}", "oracle": f"O{DECISION_SIZE}", "witness": f"Os{DECISION_SIZE}"}
SCALAR_KEYS = tuple(f"{obj}_{rung}" for rung in PROSPECTIVE_RUNGS for obj in OBJECTS) + ("dT_frozen", "c_L_level0F")
ROW_KEYS = tuple(f"row_{rung}" for rung in PROSPECTIVE_RUNGS)
PREDICTION_COLUMNS = ("token", "frame_id", "template", "p_c", "p_t", *ROW_KEYS, *SCALAR_KEYS)
ORACLE_SCALAR_KEYS = tuple(f"{obj}_{rung}" for rung in ORACLE_RUNGS for obj in OBJECTS)
ORACLE_ROW_KEYS = tuple(f"row_{rung}" for rung in ORACLE_RUNGS)

OUTCOME_Y1 = ("ROUTING_PREDICTED_TOKENS", "ROUTING_NOT_PREDICTED_TOKENS", "ROUTING_NOT_EVALUABLE_TOKENS", "PRECONDITION_FAILED_TOKENS")
OUTCOME_Y2 = ("ROUTING_PREDICTED_FRAMES_CONDITIONAL", "ROUTING_NOT_PREDICTED_FRAMES_CONDITIONAL", "ROUTING_NOT_EVALUABLE_FRAMES_CONDITIONAL", "PRECONDITION_FAILED_FRAMES")
OUTCOME_Y3 = ("OPERATING_POINT_PLUS_DRIVE_RULE_SUFFICIENT", "OPERATING_POINT_PLUS_DRIVE_RULE_INSUFFICIENT", "OPERATING_POINT_PLUS_DRIVE_RULE_NOT_EVALUABLE")
OUTCOME_Y4 = ("TEMPLATE_FAMILY_INSUFFICIENT", "TEMPLATE_FAMILY_SUFFICIENT", "TEMPLATE_FAMILY_NOT_DISTINGUISHED", "TEMPLATE_FAMILY_NOT_EVALUABLE")
OUTCOME_Y5 = ("MEMBERSHIP_PREDICTED", "MEMBERSHIP_NOT_PREDICTED", "MEMBERSHIP_NOT_EVALUABLE")
WITNESS_TERMINOLOGY = ("O* is a cross-validated empirical headroom witness (a leave-one-cue-out greedy fit of the scored objective), not a global oracle, ceiling, certificate or upper bound: "
                       "a high H* establishes transferable selectable headroom under the frozen witness; a low H* means the witness did not establish enough headroom for the "
                       "fixed-population-versus-unpredicted distinction and is never a proof that no frame-specific headroom exists.")

# The frozen candidate lists: Experiment 017's lists extended (the entries after the 017 entries), with the same quotas; first eligible entries, no class expectation.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "determiner-like": (*hp.CANDIDATES["determiner-like"], "similar", "different", "prior", "recent", "current", "original", "typical", "ordinary", "identical", "alternate", "random"),
    "ordinal-or-numeral": tuple(hp.CANDIDATES["ordinal-or-numeral"]),
    "quantity": (*hp.CANDIDATES["quantity"], "negligible", "maximum", "minimum", "average", "excessive", "exhaustive", "comprehensive", "thorough", "sweeping", "bulk", "spare", "dense", "lengthy", "lots", "loads", "tons", "scores"),
    "possessive-or-pronoun": (*hp.CANDIDATES["possessive-or-pronoun"], "who", "thee", "thou", "yourself", "themselves", "ones", "others"),
    "adjective": (*hp.CANDIDATES["adjective"], "hidden", "hard", "dirty", "rare", "square", "wild", "brave", "calm", "eager", "fierce", "humble", "proud", "shy", "lazy", "busy", "sturdy", "fragile", "polished", "muddy", "straight", "hollow", "solid", "tender", "tough", "stale", "sticky", "fuzzy"),
}
QUOTAS = hp.QUOTAS
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The garage stores {cue}"),
    ("cardinal", "The nursery raises {cue}"),
    ("cardinal", "The chef prepares {cue}"),
    ("cardinal", "The vault protects {cue}"),
    ("cardinal", "The lodge hosts {cue}"),
    ("cardinal", "The mill grinds {cue}"),
    ("quantifier", "The handbook explains {cue}"),
    ("quantifier", "The catalog features {cue}"),
    ("quantifier", "The bulletin reports {cue}"),
    ("quantifier", "The digest summarizes {cue}"),
    ("quantifier", "The charter defines {cue}"),
    ("quantifier", "The gazette prints {cue}"),
    ("coordinated-adjective", "Bruno and Greta baked {cue} warm"),
    ("coordinated-adjective", "Iris and Milo scrubbed {cue} clean"),
    ("coordinated-adjective", "Otto and Vera arranged {cue} flat"),
    ("coordinated-adjective", "Fiona and Jack dried {cue} stiff"),
    ("coordinated-adjective", "Hana and Piet sliced {cue} thin"),
    ("coordinated-adjective", "Kurt and Dana brushed {cue} bright"),
)
FRAME_ID_TAG = "019"
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", *bc.SCIENTIFIC_PATH_PREFIXES[1:])
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)


class PhaseError(pm.PhaseError):
    """Protocol violation in the Experiment 019 phase machinery."""


# ---------------------------------------------------------------------------
# Pool (255 tokens, 90 frames).


def build_pool_019(manifest: ScreeningManifest, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009, confirmation_011: er.Confirmation011,
                   confirmation_012: lc.Confirmation012, confirmation_013: ap.Confirmation013, confirmation_014: nf.Confirmation014, confirmation_015: atp.Confirmation015, confirmation_016: fch.Confirmation016,
                   confirmation_017: hp.Confirmation017, confirmation_018: bc.Confirmation018) -> cs.Pool008:
    base = bc.build_pool_018(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016, confirmation_017)
    frames = base.frames + tuple(confirmation_018.frames)
    origin = dict(base.frame_origin) | {frame.frame_id: "confirmation-018" for frame in confirmation_018.frames}
    tokens = list(base.tokens)
    category, source = dict(base.token_category), dict(base.token_source)
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation_018.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"Experiment 018 token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-018"
    return cs.Pool008(frames, origin, tuple(tokens), category, source, base.nouns, base.noun_source, base.reference_ids, base.plural_cue)


# ---------------------------------------------------------------------------
# The inherited Experiment 018 extract (a scientific input: built by a committed deterministic builder from the closed 018 results state and lock).

EXTRACT_018_DIGEST_KEYS = (*bc.EXTRACT_017_DIGEST_KEYS, "confirmation_018", "lock_018")
EXTRACT_018_DIGEST_FIELDS = tuple(f"{key}_sha256" for key in EXTRACT_018_DIGEST_KEYS)
EXTRACT_SCALARS = ("F", "Pi", "dT", "c_L", "c_L_S0", "c_L_S256", "c_L_S2048", "F_S0", "Pi_S0", "dT_S0", "F_S2048", "Pi_S2048", "dT_S2048")
EXTRACT_ROWS = ("row", "row_S0", "row_S2048")
EXTRACT_SOURCE_FIELDS = ("results_state_path", "results_state_file_sha256", "results_state_sha256", "run_id", "explore_commit", "confirm_commit", "lock_path", "lock_file_sha256", "lock_content_sha256", "extraction_schema_version")


def extract_entry_018(analysis: Mapping[str, Any], subset: str) -> dict[str, Any]:
    """From an Experiment 018 recorded pair: the measured objects and its S0, S256 and S2048 predictions."""
    p = analysis["prediction"]
    return {"set": subset, "F": float(analysis["F"]), "Pi": float(analysis["Pi"]), "dT": float(analysis["dT"]), "c_L": float(analysis["c_L"]), "row": [float(v) for v in analysis["row"]],
            "c_L_S0": float(p["c_L_S0"]), "c_L_S256": float(p["c_L_S256"]), "c_L_S2048": float(p["c_L_S2048"]), "F_S0": float(p["F_S0"]), "Pi_S0": float(p["Pi_S0"]), "dT_S0": float(p["dT_S0"]),
            "F_S2048": float(p["F_S2048"]), "Pi_S2048": float(p["Pi_S2048"]), "dT_S2048": float(p["dT_S2048"]), "row_S0": [float(v) for v in p["row_S0"]], "row_S2048": [float(v) for v in p["row_S2048"]]}


def build_inherited_extract_018(results_state: Mapping[str, Any], lock_018: Mapping[str, Any], *, digests: Mapping[str, str], results_state_path: str, results_state_file_sha256: str, lock_path: str, lock_file_sha256: str) -> dict[str, Any]:
    """The deterministic extraction: every recorded 018 pair (explore, Y1, Y2), the twelve stage-1 state digests and the 78 + 12 per-frame top-256 lists, with the source identifiers."""
    if results_state.get("phases", {}).get("confirm", {}).get("status") != "complete" or not results_state.get("confirmation"):
        raise ValueError("the Experiment 018 results state is not a completed confirmation")
    if results_state.get("lock", {}).get("content_sha256") != lock_018.get("content_sha256"):
        raise ValueError("the Experiment 018 lock is not the lock of the results state")
    entries: dict[str, dict[str, Any]] = {}
    for key, analysis in results_state["exploration"]["pairs"].items():
        entries[key] = extract_entry_018(analysis, "explore")
    for key, analysis in results_state["confirmation"]["per_frame_exposed"].items():
        if key in entries:
            raise ValueError(f"{key}: an Experiment 018 confirmation pair duplicates an exposed pair")
        entries[key] = extract_entry_018(analysis, "Y1")
    for key, analysis in results_state["confirmation"]["per_frame_fresh"].items():
        if key in entries:
            raise ValueError(f"{key}: an Experiment 018 fresh-frame pair duplicates a recorded pair")
        entries[key] = extract_entry_018(analysis, "Y2")
    stage1 = results_state["confirmation"]["stage1"]
    source = {"results_state_path": results_state_path, "results_state_file_sha256": results_state_file_sha256, "results_state_sha256": results_state["state_sha256"], "run_id": results_state["run_id"],
              "explore_commit": results_state["protocol_code_commit"], "confirm_commit": results_state["phases"]["confirm"]["confirm_commit"], "lock_path": lock_path, "lock_file_sha256": lock_file_sha256,
              "lock_content_sha256": lock_018["content_sha256"], "extraction_schema_version": EXTRACTION_SCHEMA_VERSION}
    payload = {"schema_version": INHERITED_SCHEMA_VERSION, "experiment": "018", "kind": "pair-extract",
               "description": "Derived extract of Experiment 018's recorded per-pair measured F, Π, ΔT, c_L and head-row change and its S0, S256 and S2048 predictions over its 9636 exposed pairs and its 1872 + 288 confirmed pairs, with the digests of its twelve fresh frames' digested stage-1 reference states and its per-frame top-256 lists (78 at explore, 12 at stage 1). Experiment 019 recomputes the quantities and requires agreement within 1e-6, re-captures the states to their digests, and reproduces the per-frame lists with Experiment 018's own inputs (I9).",
               "source": source, **{field: digests[key] for field, key in zip(EXTRACT_018_DIGEST_FIELDS, EXTRACT_018_DIGEST_KEYS)},
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "stage1_state_digests": {key: str(value) for key, value in sorted(stage1["state_digests"].items())},
               "frame_subsets_explore": {key: [int(i) for i in value] for key, value in sorted(lock_018["frame_subsets"].items())},
               "frame_subsets_stage1": {key: [int(i) for i in value] for key, value in sorted(stage1["frame_subsets"].items())},
               "entries": {key: dict(value) for key, value in sorted(entries.items())}}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_extract_018(path: Path, *, digests: Mapping[str, str], expected_size: int) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited 018 extract: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "experiment", "kind", "description", "source", *EXTRACT_018_DIGEST_FIELDS, "model", "stage1_state_digests", "frame_subsets_explore", "frame_subsets_stage1", "entries", "content_sha256"}, "inherited 018 extract")
    if payload["schema_version"] != INHERITED_SCHEMA_VERSION or payload["experiment"] != "018" or payload["kind"] != "pair-extract" or payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("inherited 018 extract schema or digest is not frozen")
    if tuple(payload[field] for field in EXTRACT_018_DIGEST_FIELDS) != tuple(digests[key] for key in EXTRACT_018_DIGEST_KEYS):
        raise ValueError("inherited 018 extract was recorded against different frozen inputs")
    source = payload["source"]
    if set(source) != set(EXTRACT_SOURCE_FIELDS) or source["extraction_schema_version"] != EXTRACTION_SCHEMA_VERSION or source["lock_content_sha256"] != digests["lock_018"]:
        raise ValueError("inherited 018 extract source identifiers are incomplete or name a different Experiment 018 lock")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision} or len(payload["entries"]) != expected_size or len(payload["stage1_state_digests"]) != N_018_STAGE1_FRAMES:
        raise ValueError("inherited 018 extract model, size or stage-1 digests are not frozen")
    if len(payload["frame_subsets_explore"]) != N_018_LOCKED_FRAMES or len(payload["frame_subsets_stage1"]) != N_018_STAGE1_FRAMES or any(len(v) != 256 for v in (*payload["frame_subsets_explore"].values(), *payload["frame_subsets_stage1"].values())):
        raise ValueError("inherited 018 extract per-frame lists are not frozen")
    counts = {"explore": 0, "Y1": 0, "Y2": 0}
    for entry in payload["entries"].values():
        counts[entry["set"]] = counts.get(entry["set"], 0) + 1
    if (counts["explore"], counts["Y1"], counts["Y2"]) != tuple(EXTRACT_SET_SIZES_018):
        raise ValueError("inherited 018 extract set sizes are not Experiment 018's")
    return payload


def check_extract_replication(measured: Mapping[str, Mapping[str, Any]], recorded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = set(recorded) - set(measured)
    if missing:
        raise pm.IncidentError(f"{len(missing)} recorded Experiment 018 pairs lack a recomputed quantity (e.g. {sorted(missing)[:3]})")
    worst_key, worst = "", 0.0
    for key, value in recorded.items():
        other = measured[key]
        for name in EXTRACT_SCALARS:
            deviation = abs(value[name] - other[name])
            if deviation > worst:
                worst_key, worst = f"{key}:{name}", deviation
        for name in EXTRACT_ROWS:
            if len(value[name]) != len(other[name]):
                raise pm.IncidentError(f"{key}:{name}: the recomputed row has a different length from Experiment 018's record")
            deviation = max((abs(a - b) for a, b in zip(value[name], other[name])), default=0.0)
            if deviation > worst:
                worst_key, worst = f"{key}:{name}", deviation
    if worst > REPLICATION_TOLERANCE:
        raise pm.IncidentError(f"{worst_key}: recomputed quantity deviates from Experiment 018's record by {worst:.3e}")
    return {"passed": True, "n": len(recorded), "max_abs_deviation": worst, "worst_key": worst_key}


# ---------------------------------------------------------------------------
# The confirmation set (tokenizer rules only).


def fresh_tokens_019(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
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


def _expected_frames() -> list[tuple[str, str, str]]:
    expected, counters = [], {}
    for template_id, text_template in FRESH_FRAMES:
        counters[template_id] = counters.get(template_id, 0) + 1
        expected.append((f"{template_id}-{FRAME_ID_TAG}-{counters[template_id]}", template_id, text_template))
    return expected


CONFIRMATION_DIGEST_KEYS = (*bc.CONFIRMATION_DIGEST_KEYS, "confirmation_018")
CONFIRMATION_DIGEST_FIELDS = tuple(f"{key}_sha256" for key in CONFIRMATION_DIGEST_KEYS)
Confirmation019 = ap.Confirmation013


def prompt_key_manifest(confirmation: Confirmation019) -> list[str]:
    """Every prompt confirm may execute: the fresh cues in the fresh and exposed frames, the fresh frames' reference prompts and their singular/plural cue prompts."""
    keys = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    return sorted(keys)


def build_confirmation_payload(tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> dict[str, Any]:
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_019(tokenizer, excluded_ids)
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
                          "validity": {"frame": f"plural-cue head change at least {cs.STAGE_UNINFORMATIVE_FLOOR} σ_T and the cue-pair check at rate {FRAME_CUE_EFFECT_RATE}", "token": f"scored in a set iff at least {MIN_VALID_FRAMES_PER_TOKEN} valid frames there",
                                       "sets": f"Y1 at least {MIN_SCORED_TOKENS} scored tokens; Y2 at least {MIN_VALID_CUE_FINAL_FRAMES} valid cue-final and {MIN_VALID_COORDINATED_FRAMES} valid coordinated frames and {MIN_SCORED_TOKENS} scored tokens"},
                          "expectations": "none: the committed per-rung ĉ_L, F̂, Π̂, ΔT̂ and head-row predictions of every selector are the only predictions; no label is preregistered for any neuron, frame or cue"},
               "tokens": tokens, "frames": frames, "token_prompts": [nf._prompt_entry(tokenizer, frame, token) for frame in frame_objects for token in tokens],
               "exposed_frame_prompts": [nf._prompt_entry(tokenizer, frame, token) for frame in pool.frames for token in tokens],
               "construction": "tokenizer-only; first eligible candidates per lexical class of Experiment 017's lists extended; classes carry no expectation; no model output"}
    provisional = validate_confirmation({**payload, "prompt_key_manifest": [], "content_sha256": ""}, pool, digests, check_manifest=False)
    payload["prompt_key_manifest"] = prompt_key_manifest(provisional)
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def validate_confirmation(payload: Mapping[str, Any], pool: cs.Pool008, digests: Mapping[str, str], *, check_manifest: bool = True) -> Confirmation019:
    pm._require_exact_keys(payload, {"schema_version", *CONFIRMATION_DIGEST_FIELDS, "model", "reference_cue_ids", "exposed_token_ids", "exposed_frame_ids", "policy", "tokens", "frames", "token_prompts", "exposed_frame_prompts", "construction", "prompt_key_manifest", "content_sha256"}, "confirmation")
    if payload["schema_version"] != CONFIRMATION_SCHEMA_VERSION:
        raise ValueError("confirmation schema version is not frozen")
    if check_manifest and payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
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
    confirmation = Confirmation019(dict(payload["reference_cue_ids"]), frames, tuple(payload["exposed_frame_ids"]), tuple(tokens), tuple(fresh_prompts), tuple(exposed_prompts), payload["content_sha256"])
    if check_manifest and list(payload["prompt_key_manifest"]) != prompt_key_manifest(confirmation):
        raise ValueError("the prompt-key manifest does not list exactly the prompts confirm may execute")
    return confirmation


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


def load_confirmation(path: Path, pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation019:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, pool, digests)


# ---------------------------------------------------------------------------
# The masked chain with one mask per changed position.


def position_type(position: int, p_c: int) -> str:
    return "p_c" if position == p_c else "p_t"


@dataclass(frozen=True)
class RoutingChain:
    """Experiment 018's masked chain (its ``HeadChainModel`` and the locked ``p_t`` base) with a mask table resolved per rung and changed position."""

    masked: bc.MaskedChainModel

    @property
    def hcm(self) -> hp.HeadChainModel:
        return self.masked.hcm

    @property
    def lw(self) -> lc.LayerWeights:
        return self.masked.lw

    def upstream_parts(self, weights: pm.Weights, x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], p_c: int, p_t: int, token_id: int, template: str) -> tuple[bc.UpstreamParts, Any]:
        rows16 = atp.reference_rows(self.hcm.fcm.programs, x1_all[: p_c + 1], x2_all[: p_c + 1])
        return self.masked.upstream_parts(weights, rows16, x1_all, x2_all, p_c, p_t, token_id, template), rows16

    def dx3(self, up: bc.UpstreamParts, masks: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        W_out = self.lw.W_out[BLOCK]
        return {pos: base + ((masks[pos] * own + (1.0 - masks[pos]) * template) @ W_out) for pos, (base, own, template) in up.parts.items()}

    def rung(self, rr3: atp.ReferenceRow, x3_all: Sequence[torch.Tensor], up: bc.UpstreamParts, masks: Mapping[int, torch.Tensor], p_c: int, p_t: int, template: str) -> dict[str, Any]:
        dx3 = self.dx3(up, masks)
        head = self.hcm.head(rr3, x3_all, dx3, p_c, p_t, template, hp.LEVEL0)
        return {"row": head["row"], "F": head["F"], "Pi": head["Pi"], "dT": head["dT"], "c_L": self.hcm.fcm.read.inner(dx3[p_c] - up.delta_e) / up.denominator}

    def predict(self, weights: pm.Weights, x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], x3_all: Sequence[torch.Tensor], p_c: int, p_t: int, token_id: int, template: str,
                rung_masks: Mapping[str, Mapping[int, torch.Tensor]]) -> tuple[dict[str, Any], bc.UpstreamParts, atp.ReferenceRow]:
        """Every rung of ``rung_masks``: its row, F̂, Π̂, ΔT̂ and ĉ_L; the frozen pattern and Experiment 016's decoded c_L when the reference rung is among them."""
        up, _ = self.upstream_parts(weights, x1_all, x2_all, p_c, p_t, token_id, template)
        rr3 = atp.ReferenceRow(self.hcm.program3, [x.double() for x in x3_all[: p_t + 1]])
        entry: dict[str, Any] = {"p_c": p_c, "p_t": p_t}
        for name, masks in rung_masks.items():
            out = self.rung(rr3, x3_all, up, masks, p_c, p_t, template)
            entry[f"row_{name}"] = out["row"].tolist()
            for obj in OBJECTS:
                entry[f"{obj}_{name}"] = float(out[obj])
        if REFERENCE in rung_masks:
            entry["dT_frozen"] = entry[f"F_{REFERENCE}"]
            entry["c_L_level0F"] = up.c_L_level0F
        return entry, up, rr3

    def predict_from_locked(self, weights: pm.Weights, locked: Mapping[str, Any], token_id: int, template: str, rung_masks: Mapping[str, Mapping[int, torch.Tensor]]) -> dict[str, Any]:
        return self.predict(weights, hp._tensors(locked["x1_all"]), hp._tensors(locked["x2_all"]), hp._tensors(locked["x3_all"]), int(locked["p_c"]), int(locked["p_t"]), token_id, template, rung_masks)[0]

    def predict_from_state(self, weights: pm.Weights, state: hp.FrameState017, token_id: int, template: str, rung_masks: Mapping[str, Mapping[int, torch.Tensor]]) -> tuple[dict[str, Any], bc.UpstreamParts, atp.ReferenceRow]:
        return self.predict(weights, state.x1_all, state.x2_all, state.x3_all, state.p_c, state.p_t, token_id, template, rung_masks)


def chain_from_source(source: Mapping[str, Any], lock_012: Mapping[str, Any], lw: lc.LayerWeights, programs: Mapping[int, atp.LayerProgram], pool: cs.Pool008) -> RoutingChain:
    """``source`` is the exploration record at lock time and the lock itself at confirm time: the axes, read weight, layer-3 bases and the p_t base."""
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    hcm = hp.model_from_locks(source, lock_012, lw, programs, pool.reference_ids, plural_ids, hp.bases_from_json(source["bases_3"]))
    base2_pt = torch.tensor(source["base2_pt"]["vector"], dtype=torch.float64) if source["base2_pt"]["vector"] is not None else None
    return RoutingChain(bc.MaskedChainModel(hcm, base2_pt, {}))


def read_unit_effects(up: bc.UpstreamParts, read_out_abs: torch.Tensor, position: int) -> torch.Tensor:
    """|e_j| · |r(W_out[j])| / |D_T| at a changed position: the read-unit operating-point effect (Experiment 018's score unit)."""
    return up.effects(position).abs() * read_out_abs / abs(up.denominator)


def signed_contributions(up: bc.UpstreamParts, read_out: torch.Tensor, position: int) -> torch.Tensor:
    """u_j = e_j · r(W_out[j]) / D_T: the neuron's signed contribution to the decoded read (the read identity's basis)."""
    return up.effects(position) * read_out / up.denominator


def base_drive(lw: lc.LayerWeights, base: torch.Tensor, arriving: torch.Tensor) -> torch.Tensor:
    """Δpre_j^T: the drive the predicted arriving change delivers to each neuron at the template base."""
    return lw.pre_activations(BLOCK, base.double() + arriving.double()) - lw.pre_activations(BLOCK, base.double())


def template_base(chain: RoutingChain, template: str, position: int, p_c: int) -> torch.Tensor:
    """x̄₂^T(p): the Experiment 012 base at p_c; the locked Experiment 018 block-2 base at p_t."""
    if position == p_c:
        return chain.hcm.fcm.bases_012[template][1].double()
    if chain.masked.base2_pt is None:
        raise pm.IncidentError(f"no locked block-2 base at p_t for template {template}")
    return chain.masked.base2_pt.double()


def top_k(scores: torch.Tensor, k: int) -> list[int]:
    """The k largest scores, ties by the lower index (Experiment 018's order), as a sorted index list."""
    return sorted(bc.order_of(scores)[:k])


def random_controls(seed: int = CONTROL_SEED) -> dict[str, list[list[int]]]:
    """Three k-subsets per size from one generator: for k in SIZES, sample(range(2048), k) three times; overlaps are recorded, never redrawn."""
    rng = random.Random(seed)
    return {str(k): [sorted(rng.sample(range(N_NEURONS), k)) for _ in range(N_RANDOM_CONTROLS)] for k in SIZES}


class _MeanAccumulator:
    def __init__(self) -> None:
        self.total = torch.zeros(N_NEURONS, dtype=torch.float64)
        self.count = 0

    def add(self, vector: torch.Tensor) -> None:
        self.total += vector
        self.count += 1

    def mean(self) -> torch.Tensor:
        if self.count == 0:
            raise pm.IncidentError("an empty selector population")
        return self.total / self.count


def g_scores(pre_frame: torch.Tensor, pre_base: torch.Tensor, quantiles: Sequence[torch.Tensor], read_out_abs: torch.Tensor, denominator: float) -> torch.Tensor:
    """The operating-point-plus-drive rule: the exact GELU response at the frame's operating point against the template base, averaged over the locked drive quantiles, in read units."""
    a, b = pre_frame.double(), pre_base.double()
    gelu = torch.nn.functional.gelu
    total = torch.zeros_like(a)
    for d in quantiles:
        total += ((gelu(a + d) - gelu(a)) - (gelu(b + d) - gelu(b))).abs()
    return total / len(quantiles) * read_out_abs / abs(denominator)


def frame_lists(scores_by_position: Mapping[str, torch.Tensor]) -> dict[str, dict[str, list[int]]]:
    return {pos: {str(k): top_k(scores, k) for k in SIZES} for pos, scores in scores_by_position.items()}


def selectors_from_states(chain: RoutingChain, weights: pm.Weights, read_out: torch.Tensor, locked_states: Mapping[str, Mapping[str, Any]], frames: Sequence[pm.Frame], tokens_by_template: Mapping[str, Sequence[tuple[str, int]]],
                          licensed_keys: set[str], inherited_subsets: Mapping[str, Sequence[int]], *, replication: Mapping[str, Any] | None = None, log: Any = None) -> dict[str, Any]:
    """Every prospective selector from the locked reference states, the weights and the exposed cues' ΔE alone.

    ``S'`` (global) and ``T`` (template) are means of the read-unit effect over the licensed records at each position type; the drive quantiles of ``G`` are taken over the same
    records' base drives; ``E`` evaluates every exposed cue of the template at every frame's state; ``G`` reads the frame's reference pre-activations against the template base.
    With ``replication`` (the Experiment 018 pool's token ids per template and its licensed keys, and the frame ids it locked), the E rule is also accumulated with Experiment 018's
    own inputs and pooled over positions, for I9. No measured quantity of any pair enters, and no fresh token: the function receives exposed token ids only.
    """
    say = log or (lambda message: None)
    read_out_abs = read_out.abs()
    lw = chain.lw
    global_acc: dict[str, _MeanAccumulator] = {pt: _MeanAccumulator() for pt in POSITION_TYPES}
    template_acc: dict[str, dict[str, _MeanAccumulator]] = {}
    drives: dict[str, dict[str, list[torch.Tensor]]] = {}
    e_scores: dict[str, dict[str, torch.Tensor]] = {}
    e_018: dict[str, torch.Tensor] = {}
    pre_frames: dict[str, dict[str, torch.Tensor]] = {}
    denominators: dict[str, float] = {}
    n_records = {"licensed": 0, "evaluated": 0}
    for frame in frames:
        locked = locked_states[frame.frame_id]
        template = frame.template_id
        x1_all, x2_all = hp._tensors(locked["x1_all"]), hp._tensors(locked["x2_all"])
        p_c, p_t = int(locked["p_c"]), int(locked["p_t"])
        positions = sorted({p_c, p_t})
        pre_frames[frame.frame_id] = {str(pos): lw.pre_activations(BLOCK, x2_all[pos].double()) for pos in positions}
        acc_e = {str(pos): _MeanAccumulator() for pos in positions}
        acc_018 = bc.RankingAccumulator(read_out) if replication is not None and frame.frame_id in replication["frame_ids"] else None
        tokens_018 = set(replication["token_ids_by_template"][template]) if replication is not None else set()
        for word, token_id in tokens_by_template[template]:
            up, _ = chain.upstream_parts(weights, x1_all, x2_all, p_c, p_t, token_id, template)
            denominators.setdefault(template, float(up.denominator))
            n_records["evaluated"] += 1
            key = f"{word}|{frame.frame_id}"
            licensed = key in licensed_keys
            for pos in positions:
                effect = read_unit_effects(up, read_out_abs, pos)
                acc_e[str(pos)].add(effect)
                if licensed:
                    pt = position_type(pos, p_c)
                    global_acc[pt].add(effect)
                    template_acc.setdefault(template, {}).setdefault(pt, _MeanAccumulator()).add(effect)
                    drives.setdefault(template, {}).setdefault(pt, []).append(base_drive(lw, template_base(chain, template, pos, p_c), up.arriving[pos]).to(torch.float32))
            if licensed:
                n_records["licensed"] += 1
            if acc_018 is not None and token_id in tokens_018 and (frame.frame_id in replication["stage1_frame_ids"] or key in replication["licensed_keys"]):
                acc_018.add(up)
        e_scores[frame.frame_id] = {pos: acc.mean() for pos, acc in acc_e.items()}
        if acc_018 is not None:
            e_018[frame.frame_id] = acc_018.scores()
        say(f"  {frame.frame_id}: {len(tokens_by_template[template])} exposed cues evaluated at the frame's state")
    quantiles: dict[str, dict[str, dict[str, list[float]]]] = {}
    for template, by_type in drives.items():
        quantiles[template] = {}
        for pt, rows in by_type.items():
            stacked = torch.stack(rows).double()
            quantiles[template][pt] = {str(q): torch.quantile(stacked, q, dim=0, interpolation="linear").tolist() for q in DRIVE_QUANTILES}
    sp_scores = {pt: acc.mean() for pt, acc in global_acc.items() if acc.count}
    t_scores = {template: {pt: acc.mean() for pt, acc in by_type.items()} for template, by_type in template_acc.items()}
    g_scores_by_frame: dict[str, dict[str, torch.Tensor]] = {}
    for frame in frames:
        template = frame.template_id
        locked = locked_states[frame.frame_id]
        p_c = int(locked["p_c"])
        g_scores_by_frame[frame.frame_id] = {}
        for pos_str, pre in pre_frames[frame.frame_id].items():
            pos = int(pos_str)
            pt = position_type(pos, p_c)
            q = [torch.tensor(quantiles[template][pt][str(qq)], dtype=torch.float64) for qq in DRIVE_QUANTILES]
            g_scores_by_frame[frame.frame_id][pos_str] = g_scores(pre, lw.pre_activations(BLOCK, template_base(chain, template, pos, p_c)), q, read_out_abs, denominators[template])
    lists = {"Sp": {pt: {str(k): top_k(s, k) for k in SIZES} for pt, s in sp_scores.items()},
             "T": {template: {pt: {str(k): top_k(s, k) for k in SIZES} for pt, s in by_type.items()} for template, by_type in t_scores.items()},
             "E": {frame_id: frame_lists(scores) for frame_id, scores in e_scores.items()},
             "G": {frame_id: frame_lists(scores) for frame_id, scores in g_scores_by_frame.items()},
             "Sp1": top_k(sp_scores["p_c"], 1), "E1": {frame_id: {pos: top_k(s, 1) for pos, s in scores.items()} for frame_id, scores in e_scores.items()},
             "L": {str(k): [int(i) for i in inherited_subsets[f"S{k}"]] for k in SIZES}, "R": random_controls()}
    scores = {"Sp": {pt: s.tolist() for pt, s in sp_scores.items()}, "T": {template: {pt: s.tolist() for pt, s in by_type.items()} for template, by_type in t_scores.items()},
              "E": {frame_id: {pos: s.tolist() for pos, s in by_frame.items()} for frame_id, by_frame in e_scores.items()}, "G": {frame_id: {pos: s.tolist() for pos, s in by_frame.items()} for frame_id, by_frame in g_scores_by_frame.items()}}
    out = {"lists": lists, "scores": scores, "drive_quantiles": quantiles, "denominators": denominators, "n_records": n_records, "sizes": list(SIZES), "quantile_levels": list(DRIVE_QUANTILES)}
    if replication is not None:
        out["e_018_lists"] = {frame_id: bc.frame_subset(s) for frame_id, s in e_018.items()}
    out["digest"] = selectors_digest(out)
    return out


def selectors_digest(selectors: Mapping[str, Any]) -> str:
    return pm.sha256_text(pm.canonical_json({key: selectors[key] for key in ("lists", "scores", "drive_quantiles", "denominators")}))


def lists_from_scores(selectors: Mapping[str, Any], inherited_subsets: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """The frozen top-k rule applied to the recorded score vectors (what validate_lock checks the locked lists against)."""
    scores = selectors["scores"]
    t = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    return {"Sp": {pt: {str(k): top_k(t(s), k) for k in SIZES} for pt, s in scores["Sp"].items()},
            "T": {template: {pt: {str(k): top_k(t(s), k) for k in SIZES} for pt, s in by_type.items()} for template, by_type in scores["T"].items()},
            "E": {frame_id: {pos: {str(k): top_k(t(s), k) for k in SIZES} for pos, s in by_frame.items()} for frame_id, by_frame in scores["E"].items()},
            "G": {frame_id: {pos: {str(k): top_k(t(s), k) for k in SIZES} for pos, s in by_frame.items()} for frame_id, by_frame in scores["G"].items()},
            "Sp1": top_k(t(scores["Sp"]["p_c"]), 1), "E1": {frame_id: {pos: top_k(t(s), 1) for pos, s in by_frame.items()} for frame_id, by_frame in scores["E"].items()},
            "L": {str(k): [int(i) for i in inherited_subsets[f"S{k}"]] for k in SIZES}, "R": random_controls()}


def check_selector_lists(selectors: Mapping[str, Any], inherited_subsets: Mapping[str, Sequence[int]]) -> None:
    expected = lists_from_scores(selectors, inherited_subsets)
    mine = json.loads(pm.canonical_json(selectors["lists"]))
    if json.loads(pm.canonical_json(expected)) != mine:
        raise PhaseError("the locked selector lists are not the frozen top-k rule applied to the locked scores (or the controls and the inherited lists are not the frozen ones)")


def frame_rung_masks(lists: Mapping[str, Any], frame_id: str, template: str, p_c: int, p_t: int, frame_lists_override: Mapping[str, Any] | None = None) -> dict[str, dict[int, torch.Tensor]]:
    """The mask per prospective rung and changed position for one frame: fixed lists for S0/S2048/Sp/L/R, the template's lists for T, the frame's lists for E/G/E1 (from ``frame_lists_override`` for a fresh frame)."""
    positions = sorted({p_c, p_t})
    own = frame_lists_override if frame_lists_override is not None else {"E": lists["E"][frame_id], "G": lists["G"][frame_id], "E1": lists["E1"][frame_id]}
    fixed = lambda idx: {pos: bc.mask_of(idx) for pos in positions}  # noqa: E731
    masks: dict[str, dict[int, torch.Tensor]] = {TEMPLATE_BASE: fixed([]), REFERENCE: fixed(list(range(N_NEURONS))), GLOBAL_SINGLE: fixed(lists["Sp1"]),
                                                  FRAME_SINGLE: {pos: bc.mask_of(own["E1"][str(pos)]) for pos in positions}}
    for k in SIZES:
        masks[f"Sp{k}"] = {pos: bc.mask_of(lists["Sp"][position_type(pos, p_c)][str(k)]) for pos in positions}
        masks[f"T{k}"] = {pos: bc.mask_of(lists["T"][template][position_type(pos, p_c)][str(k)]) for pos in positions}
        masks[f"E{k}"] = {pos: bc.mask_of(own["E"][str(pos)][str(k)]) for pos in positions}
        masks[f"G{k}"] = {pos: bc.mask_of(own["G"][str(pos)][str(k)]) for pos in positions}
        masks[f"L{k}"] = fixed(lists["L"][str(k)])
        for i in range(1, N_RANDOM_CONTROLS + 1):
            masks[f"R{i}_{k}"] = fixed(lists["R"][str(k)][i - 1])
    return masks


# ---------------------------------------------------------------------------
# Measured per-neuron effects, the ranking oracle and the greedy witness (stage 2 only).


def measured_effects(lw: lc.LayerWeights, x2_ref: torch.Tensor, x2_patched: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """e_j^meas = own_j^meas − tmpl_j^meas from the captured residual before block 2 (the neurons' observed responses in the patched run)."""
    x2_ref, x2_patched, base = x2_ref.double(), x2_patched.double(), base.double()
    arriving = x2_patched - x2_ref
    own = lw.hidden(BLOCK, x2_patched) - lw.hidden(BLOCK, x2_ref)
    tmpl = lw.hidden(BLOCK, base + arriving) - lw.hidden(BLOCK, base)
    return own - tmpl


def effect_fidelity(predicted: torch.Tensor, measured: torch.Tensor) -> float:
    """R² of the chain's per-neuron effects against the measured ones over the 2048 neurons (descriptive)."""
    ss_tot = float(((measured - measured.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return 1.0 if float(((predicted - measured) ** 2).sum()) == 0.0 else 0.0
    return 1.0 - float(((predicted - measured) ** 2).sum()) / ss_tot


def ranking_oracle(effects: Sequence[torch.Tensor], read_out_abs: torch.Tensor, denominator: float) -> torch.Tensor:
    """The measured-effect ranking score of a frame: the mean |e_j^meas| |r_j| / |D_T| over its scored pairs."""
    if not effects:
        raise pm.IncidentError("the ranking oracle has no scored pairs")
    return torch.stack([e.abs() for e in effects]).mean(0) * read_out_abs / abs(denominator)


def greedy_witness(train_u: torch.Tensor, train_y: torch.Tensor, k: int) -> list[int]:
    """The leave-one-cue-out greedy fit: from the OTHER cues' contribution vectors (n × 2048) and residual reads (n), the k neurons that successively most reduce the residual
    sum of squares ‖r − u_j‖²; ties by the lower index; no early stop. The held-out cue's own u_j and read are not arguments and cannot enter."""
    if train_u.dim() != 2 or train_u.shape[1] != N_NEURONS or train_y.shape != (train_u.shape[0],):
        raise ValueError("greedy_witness expects n × 2048 contribution vectors and n residual reads of the other cues")
    if train_u.shape[0] == 0:
        raise pm.IncidentError("the witness has no other scored cue to fit")
    u = train_u.double()
    r = train_y.double().clone()
    norms = (u * u).sum(0)
    available = torch.ones(N_NEURONS, dtype=torch.bool)
    chosen: list[int] = []
    for _ in range(k):
        delta = -2.0 * (u.T @ r) + norms  # the change of ‖r − u_j‖² − ‖r‖² when neuron j is added
        delta[~available] = float("inf")
        best = float(delta.min())
        j = int((delta == best).nonzero()[0])  # the lowest index among exact ties
        chosen.append(j)
        available[j] = False
        r = r - u[:, j]
    return sorted(chosen)


def apply_mask_read(c_L_base: float, u: torch.Tensor, mask_indices: Sequence[int]) -> float:
    """ĉ_L(S) = ĉ_L(S_0) + Σ_{j ∈ S} u_j — the read identity, used to check I10 and to state what the witness optimizes."""
    if not len(mask_indices):
        return float(c_L_base)
    return float(c_L_base + u[torch.tensor(sorted(int(i) for i in mask_indices), dtype=torch.long)].sum())


# ---------------------------------------------------------------------------
# Measurement per pair.


def patched_sites_019(frame: pm.Frame) -> list[pm.Site]:
    """Experiment 017's patched sites plus the residual before block 2 at p_t (coordinated frames), so that the measured per-neuron effect is defined at both changed positions."""
    sites = hp.patched_sites_017(frame)
    extra = (f"RESID_PRE.L{BLOCK}", frame.p_t)
    return sites if extra in sites else sites + [extra]


def measure_pair(model: Any, weights: pm.Weights, head: ht.HeadWeights, state: hp.FrameState017, name: str, token_id: int, e_axis: pm.SiteAxis, axis_T: pm.SiteAxis, nouns: Sequence[pm.Noun]) -> ra.Attribution:
    s = state.state
    return ra.measure_token_010(model, weights, head, s.ref, s.components, s.functional, name, token_id, e_axis, axis_T, nouns, extra_sites=patched_sites_019(s.ref.frame))


@dataclass(frozen=True)
class AnalysisContext:
    chain: RoutingChain
    hp_context: hp.AnalysisContext
    weights: pm.Weights
    axis_T: pm.SiteAxis
    read_out: torch.Tensor


def check_i8(entry: Mapping[str, Any], prediction_017: Mapping[str, Any], where: str) -> float:
    """I8: the reference rung equals Experiment 017's Level 0 (row, F̂, Π̂, ΔT̂, decoded c_L)."""
    worst = max(abs(entry[f"F_{REFERENCE}"] - prediction_017["F_hat"]), abs(entry[f"Pi_{REFERENCE}"] - prediction_017["Pi_hat"]), abs(entry[f"dT_{REFERENCE}"] - prediction_017["dT_hat"]),
                abs(entry[f"c_L_{REFERENCE}"] - prediction_017["c_L_level0D"]), max(abs(a - b) for a, b in zip(entry[f"row_{REFERENCE}"], prediction_017["row_level0"])))
    if worst > I8_TOLERANCE:
        raise pm.IncidentError(f"{where}: the masked chain does not reproduce Experiment 017's Level 0 (I8 {worst:.2e})")
    return worst


def check_i10(entry: Mapping[str, Any], up: bc.UpstreamParts, read_out: torch.Tensor, rung_masks: Mapping[str, Mapping[int, torch.Tensor]], p_c: int, where: str) -> float:
    """I10: for every rung, ĉ_L(S) = ĉ_L(S_0) + Σ_{j ∈ S(p_c)} u_j."""
    u = signed_contributions(up, read_out, p_c)
    base = entry[f"c_L_{TEMPLATE_BASE}"]
    worst = 0.0
    for name, masks in rung_masks.items():
        expected = float(base + (masks[p_c] * u).sum())
        worst = max(worst, abs(entry[f"c_L_{name}"] - expected))
    if worst > I10_TOLERANCE:
        raise pm.IncidentError(f"{where}: the read identity fails (I10 {worst:.2e})")
    return worst


def analyse_pair_019(record: ra.Attribution, plural: ra.Attribution, *, context: AnalysisContext, state: hp.FrameState017, rung_masks: Mapping[str, Mapping[int, torch.Tensor]]) -> dict[str, Any] | None:
    """Experiment 017's analysis (I1–I7, its Level 0, the measured objects) plus every prospective rung of the per-position masked chain, I8 and I10, the measured per-neuron effects and the chain's contributions."""
    base = hp.analyse_pair_017(record, plural, context=context.hp_context, state=state)
    if base is None:
        return None
    chain, weights = context.chain, context.weights
    template, p_c, p_t = state.frame.template_id, state.p_c, state.p_t
    where = f"{state.frame.frame_id}/{record.token}"
    entry, up, rr3 = chain.predict_from_state(weights, state, record.token_id, template, rung_masks)
    identities = dict(base["identities"])
    identities["I8_reference_rung"] = check_i8(entry, base["prediction"], where)
    identities["I10_read_identity"] = check_i10(entry, up, context.read_out, rung_masks, p_c, where)
    measured_row = torch.tensor(base["row"], dtype=torch.float64)
    statistics = {name: atp.row_statistics(torch.tensor(entry[f"row_{name}"], dtype=torch.float64), measured_row) for name in rung_masks}
    effects, fidelity = {}, {}
    for pos in sorted({p_c, p_t}):
        x2_patched = record.extra[pm.site_label((f"RESID_PRE.L{BLOCK}", pos))].double()
        e_meas = measured_effects(chain.lw, state.x2_all[pos], x2_patched, template_base(chain, template, pos, p_c))
        effects[pos] = e_meas
        fidelity[str(pos)] = effect_fidelity(up.effects(pos), e_meas)
    analysis = {"token": base["token"], "token_id": base["token_id"], "frame_id": base["frame_id"], "template": base["template"], "p_c": p_c, "p_t": p_t, "cue_final": base["cue_final"],
                "row": base["row"], "F": base["F"], "Pi": base["Pi"], "dT": base["dT"], "c_L": base["c_L"], "prediction": entry, "statistics": statistics, "identities": identities,
                "x3_remainder": base["x3_remainder"], "effect_fidelity": fidelity,
                "analysis_017": {"F": base["F"], "Pi": base["Pi"], "dT": base["dT"], "c_L": base["c_L"], "row": base["row"], "prediction": base["prediction"]}}
    return {"analysis": analysis, "effects": effects, "u": signed_contributions(up, context.read_out, p_c), "up": up, "rr3": rr3}


def compact_analysis(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in analysis.items() if key != "analysis_017"}


def extract_entry_from_analysis(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """The Experiment 018 extract's fields recomputed here: the measured objects and the S0 / L256 (= 018's S256) / S2048 predictions."""
    p = analysis["prediction"]
    return {"F": analysis["F"], "Pi": analysis["Pi"], "dT": analysis["dT"], "c_L": analysis["c_L"], "row": list(analysis["row"]),
            "c_L_S0": p[f"c_L_{TEMPLATE_BASE}"], "c_L_S256": p["c_L_L256"], "c_L_S2048": p[f"c_L_{REFERENCE}"], "F_S0": p[f"F_{TEMPLATE_BASE}"], "Pi_S0": p[f"Pi_{TEMPLATE_BASE}"], "dT_S0": p[f"dT_{TEMPLATE_BASE}"],
            "F_S2048": p[f"F_{REFERENCE}"], "Pi_S2048": p[f"Pi_{REFERENCE}"], "dT_S2048": p[f"dT_{REFERENCE}"], "row_S0": list(p[f"row_{TEMPLATE_BASE}"]), "row_S2048": list(p[f"row_{REFERENCE}"])}


def prediction_row(word: str, frame_id: str, template: str, prediction: Mapping[str, Any]) -> dict[str, Any]:
    row = {"token": word, "frame_id": frame_id, "template": template}
    for key in PREDICTION_COLUMNS[3:]:
        row[key] = prediction[key]
    return row


def oracle_rungs_for_frame(chain: RoutingChain, weights: pm.Weights, state: hp.FrameState017, token_id: int, template: str, oracle_lists: Mapping[str, Mapping[str, list[int]]], witness_lists: Mapping[str, list[int]]) -> dict[str, Any]:
    """The oracle rungs of one pair: O_k at both positions; O*_k (the pair's own leave-one-cue-out witness) at p_c with O_k at p_t."""
    positions = sorted({state.p_c, state.p_t})
    masks: dict[str, dict[int, torch.Tensor]] = {}
    for k in SIZES:
        masks[f"O{k}"] = {pos: bc.mask_of(oracle_lists[str(pos)][str(k)]) for pos in positions}
        masks[f"Os{k}"] = {pos: (bc.mask_of(witness_lists[str(k)]) if pos == state.p_c else bc.mask_of(oracle_lists[str(pos)][str(k)])) for pos in positions}
    entry, _, _ = chain.predict_from_state(weights, state, token_id, template, masks)
    return {key: value for key, value in entry.items() if key not in ("p_c", "p_t")}


def oracles_for_set(pairs: Mapping[str, Mapping[str, Any]], effects: Mapping[str, Mapping[int, torch.Tensor]], contributions: Mapping[str, torch.Tensor], scored_tokens: Sequence[str], read_out_abs: torch.Tensor,
                    denominators: Mapping[str, float]) -> tuple[dict[str, dict[str, dict[str, list[int]]]], dict[str, dict[str, list[int]]]]:
    """From the stage-2 measurements of one set: per frame the ranking-oracle lists at each position over its scored pairs; per pair the witness lists fitted to the frame's OTHER scored pairs."""
    scored = set(scored_tokens)
    by_frame: dict[str, list[str]] = {}
    for key in sorted(pairs):
        word, frame_id = key.split("|", 1)
        if word in scored:
            by_frame.setdefault(frame_id, []).append(key)
    oracle_lists: dict[str, dict[str, dict[str, list[int]]]] = {}
    witness_lists: dict[str, dict[str, list[int]]] = {}
    for frame_id, keys in by_frame.items():
        template = pairs[keys[0]]["template"]
        p_c = int(pairs[keys[0]]["p_c"])
        positions = sorted({p_c, int(pairs[keys[0]]["p_t"])})
        oracle_lists[frame_id] = {str(pos): {str(k): top_k(ranking_oracle([effects[key][pos] for key in keys], read_out_abs, denominators[template]), k) for k in SIZES} for pos in positions}
        for key in keys:
            others = [other for other in keys if other != key]
            if not others:
                raise pm.IncidentError(f"{key}: the witness needs at least one other scored cue in the frame")
            train_u = torch.stack([contributions[other] for other in others])
            train_y = torch.tensor([pairs[other]["c_L"] - pairs[other]["prediction"][f"c_L_{TEMPLATE_BASE}"] for other in others], dtype=torch.float64)
            witness_lists[key] = {str(k): greedy_witness(train_u, train_y, k) for k in SIZES}
    return oracle_lists, witness_lists


# ---------------------------------------------------------------------------
# Results state, phases.

DIGEST_KEYS = (*bc.DIGEST_KEYS, "lock_018", "extract_018", "confirmation_019")
DIGEST_KEYS = tuple(key for key in DIGEST_KEYS if key != "extract_017")  # Experiment 017's extract was Experiment 018's input, bound by the 018 lock; it is not read here
STATE_DIGEST_FIELDS = tuple(f"{key}_sha256" for key in DIGEST_KEYS)
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


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation019) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


def assert_no_target_pair_executed(state: Mapping[str, Any], confirmation: Confirmation019) -> None:
    """After stage 1: none of the fresh-cue prompts (the 432 target pairs and the exposed-frame pairs) has run."""
    executed = set(state["executed_prompt_keys"])
    targets = {prompt.key for prompt in confirmation.token_prompts} | {prompt.key for prompt in confirmation.exposed_frame_prompts}
    if executed & targets:
        raise PhaseError("a fresh cue prompt was executed before the stage barrier")


STAGE1_FORBIDDEN_FIELDS = ("effects", "oracle", "witness", "measured", "c_L", "F", "Pi", "dT")


def stage_digest(rows: Sequence[Mapping[str, Any]], states: Mapping[str, Any], frame_selectors: Mapping[str, Any]) -> str:
    return pm.sha256_text(pm.canonical_json({"rows": list(rows), "states": dict(states), "frame_selectors": dict(frame_selectors)}))


def assert_stage_one_digest(stage1: Mapping[str, Any]) -> None:
    if not stage1 or "frame_selectors" not in stage1 or stage1.get("digest") != stage_digest(stage1["rows"], stage1["states"], stage1["frame_selectors"]):
        raise PhaseError("stage 1's prediction table is missing or its digest does not verify; no fresh cue prompt may run")
    if any(field in stage1 for field in STAGE1_FORBIDDEN_FIELDS):
        raise PhaseError("stage 1's record carries a measured fresh-cue quantity")


comparison = lc.comparison


# ---------------------------------------------------------------------------
# Statistics: κ per rung, the routing gains Δκ, ρ, the headroom, per frame, splits, membership.


def _mean(values: Sequence[float]) -> float:
    return pm._mean(list(values))


def merged_prediction(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """The prospective columns and, when present, the oracle columns of one pair as one dictionary (what the statistics read)."""
    out = dict(analysis["prediction"])
    out.update(analysis.get("oracle") or {})
    return out


def rungs_present(prediction: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(rung for rung in ALL_RUNGS if f"c_L_{rung}" in prediction)


def rung_statistics(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], rungs: Sequence[str] | None = None) -> dict[str, Any]:
    rungs = tuple(rungs) if rungs is not None else rungs_present(predictions[0]) if predictions else PROSPECTIVE_RUNGS
    return bc.rung_statistics(analyses, predictions, rungs=rungs)


def gains(kappas: Mapping[str, float | None], k: int) -> dict[str, float | None]:
    """Δκ_k(X) = κ_{c_L}(X_k) − κ_{c_L}(S'_k) for every selector present, plus E1 − Sp1."""
    global_k = kappas.get(f"Sp{k}")
    out: dict[str, float | None] = {}
    for prefix in ("T", "E", "G", "L", "O", "Os"):
        value = kappas.get(f"{prefix}{k}")
        out[f"{prefix}{k}"] = (value - global_k) if value is not None and global_k is not None else None
    for i in range(1, N_RANDOM_CONTROLS + 1):
        value = kappas.get(f"R{i}_{k}")
        out[f"R{i}_{k}"] = (value - global_k) if value is not None and global_k is not None else None
    single, frame_single = kappas.get(GLOBAL_SINGLE), kappas.get(FRAME_SINGLE)
    out["E1"] = (frame_single - single) if single is not None and frame_single is not None else None
    return out


def rho(gain_g: float | None, gain_e: float | None) -> float | None:
    """ρ = Δκ(G) / Δκ(E), unclipped; None where the denominator is below RHO_DENOMINATOR_MIN."""
    if gain_g is None or gain_e is None or gain_e < RHO_DENOMINATOR_MIN:
        return None
    return gain_g / gain_e


def per_frame_c_L(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per frame: the c_L R² of every rung over the frame's scored pairs, the gap, κ per rung."""
    out = {}
    rungs = rungs_present(predictions[0]) if predictions else PROSPECTIVE_RUNGS
    for frame_id in sorted({a["frame_id"] for a in analyses}):
        mine = [(a, p) for a, p in zip(analyses, predictions) if a["frame_id"] == frame_id]
        a_list, p_list = [a for a, _ in mine], [p for _, p in mine]
        measured = [a["c_L"] for a in a_list]
        r2 = {rung: lc.explained_variance([p[f"c_L_{rung}"] for p in p_list], measured) for rung in rungs}
        base, full = r2[TEMPLATE_BASE], r2[REFERENCE]
        gap = (full - base) if base is not None and full is not None else None
        out[frame_id] = {"n_pairs": len(mine), "template": a_list[0]["template"], "cue_final": a_list[0]["cue_final"], "r2": r2, "gap": gap, "kappa": {rung: bc.kappa(r2[rung], base, full) for rung in rungs}}
    return out


def frame_wins(per_frame: Mapping[str, Mapping[str, Any]], challenger: str, incumbent: str) -> dict[str, Any]:
    """The frame-count guard: the challenger's c_L R² strictly above the incumbent's; a tie is not a win."""
    wins, losses, ties, undefined = [], [], [], []
    for frame_id, entry in per_frame.items():
        a, b = entry["r2"].get(challenger), entry["r2"].get(incumbent)
        if a is None or b is None:
            undefined.append(frame_id)
        elif a > b:
            wins.append(frame_id)
        elif a < b:
            losses.append(frame_id)
        else:
            ties.append(frame_id)
    n = len(per_frame)
    return {"n_frames": n, "wins": len(wins), "losses": len(losses), "ties": len(ties), "undefined": len(undefined), "winning_frames": wins, "holds": n > 0 and len(wins) * 2 > n}


def split_gains(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], k: int) -> dict[str, Any]:
    """The cue-final / coordinated split and every template: the c_L gap, κ of the decision selectors and Δκ_k over the split's pairs."""
    def one(chosen: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
        if not chosen:
            return {"n_pairs": 0, "gap_c_L": None, "evaluable": False, "kappa": {}, "gains": {}}
        stats = rung_statistics([a for a, _ in chosen], [p for _, p in chosen])
        kappas = stats["kappa"]["c_L"]
        return {"n_pairs": len(chosen), "n_tokens": len({a["token"] for a, _ in chosen}), "n_frames": len({a["frame_id"] for a, _ in chosen}), "gap_c_L": stats["gaps"]["c_L"], "evaluable": stats["evaluable"]["c_L"],
                "kappa": {rung: kappas.get(rung) for rung in kappas}, "gains": gains(kappas, k)}
    pairs = list(zip(analyses, predictions))
    return {"cue_final": one([x for x in pairs if x[0]["cue_final"]]), "coordinated": one([x for x in pairs if not x[0]["cue_final"]]),
            "per_template": {template: one([x for x in pairs if x[0]["template"] == template]) for template in pm.TEMPLATE_ORDER}}


def jackknife_gain(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], k: int, selector: str = "E") -> dict[str, Any]:
    """Δκ_k(selector) with each frame removed in turn (descriptive vii)."""
    frames = sorted({a["frame_id"] for a in analyses})
    values = {}
    for frame_id in frames:
        keep = [(a, p) for a, p in zip(analyses, predictions) if a["frame_id"] != frame_id]
        if not keep:
            continue
        stats = rung_statistics([a for a, _ in keep], [p for _, p in keep], rungs=(TEMPLATE_BASE, REFERENCE, f"Sp{k}", f"{selector}{k}"))
        values[frame_id] = gains(stats["kappa"]["c_L"], k).get(f"{selector}{k}")
    defined = [v for v in values.values() if v is not None]
    return {"removed_frame": values, "min": min(defined) if defined else None, "max": max(defined) if defined else None}


def overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return len(set(int(i) for i in a) & set(int(i) for i in b))


def membership(frame_lists_by_frame: Mapping[str, Mapping[str, Any]], global_lists: Mapping[str, Any], template_lists: Mapping[str, Any], oracle_lists: Mapping[str, Mapping[str, Any]], frame_templates: Mapping[str, str],
               frame_pc: Mapping[str, int], k: int = DECISION_SIZE) -> dict[str, Any]:
    """Y5's statistic: over the frames with an oracle list, the mean fraction of the oracle's top-k at p_c that E_k (and S'_k, G_k, T_k) names."""
    rows = {}
    for frame_id, oracle in oracle_lists.items():
        p_c = str(frame_pc[frame_id])
        if p_c not in oracle:
            continue
        target = oracle[p_c][str(k)]
        own = frame_lists_by_frame[frame_id]
        rows[frame_id] = {"E": overlap(own["E"][p_c][str(k)], target) / k, "G": overlap(own["G"][p_c][str(k)], target) / k, "Sp": overlap(global_lists["Sp"]["p_c"][str(k)], target) / k,
                          "T": overlap(template_lists[frame_templates[frame_id]]["p_c"][str(k)], target) / k}
    if not rows:
        return {"n_frames": 0, "mean": {}, "per_frame": {}}
    return {"n_frames": len(rows), "mean": {name: _mean([row[name] for row in rows.values()]) for name in ("E", "G", "Sp", "T")}, "per_frame": rows}


def token_means(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not analyses:
        return {}
    rungs = rungs_present(predictions[0])
    out = {"n_frames": len(analyses), "n_cue_final": sum(1 for a in analyses if a["cue_final"]), **{f"{obj}_mean": _mean([a[obj] for a in analyses]) for obj in OBJECTS}}
    out.update({f"{obj}_{rung}_mean": _mean([p[f"{obj}_{rung}"] for p in predictions]) for rung in rungs for obj in OBJECTS})
    out["dT_frozen_mean"] = _mean([p["dT_frozen"] for p in predictions])
    return out


def set_statistics(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything the labels read for one population of scored pairs: pooled κ per rung, the gains at every size, ρ, the headroom, per frame, the frame-count guard, the splits, the jackknife."""
    stats = rung_statistics(analyses, predictions)
    kappas = stats["kappa"]["c_L"]
    out: dict[str, Any] = {"n_pairs": len(analyses), "n_frames": len({a["frame_id"] for a in analyses}), "n_tokens": len({a["token"] for a in analyses}), "pairs": stats,
                           "gains": {str(k): gains(kappas, k) for k in SIZES}, "per_frame": per_frame_c_L(analyses, predictions)}
    out["rho"] = {str(k): rho(out["gains"][str(k)].get(f"G{k}"), out["gains"][str(k)].get(f"E{k}")) for k in SIZES}
    out["wins"] = {str(k): frame_wins(out["per_frame"], f"E{k}", f"Sp{k}") for k in SIZES}
    out["wins_G"] = {str(k): frame_wins(out["per_frame"], f"G{k}", f"Sp{k}") for k in SIZES}
    out["wins_T"] = {str(k): frame_wins(out["per_frame"], f"T{k}", f"Sp{k}") for k in SIZES}
    out["splits"] = split_gains(analyses, predictions, DECISION_SIZE)
    out["headroom"] = {"witness": out["gains"][str(DECISION_SIZE)].get(DECISION["witness"]), "ranking_proxy": out["gains"][str(DECISION_SIZE)].get(DECISION["oracle"])}
    e_gain = out["gains"][str(DECISION_SIZE)].get(DECISION["full"])
    out["witness_share"] = (e_gain / out["headroom"]["witness"]) if e_gain is not None and out["headroom"]["witness"] not in (None, 0.0) else None
    out["jackknife"] = {"E": jackknife_gain(analyses, predictions, DECISION_SIZE, "E"), "G": jackknife_gain(analyses, predictions, DECISION_SIZE, "G")}
    losses = {}
    for name in ("E", "G"):
        losses[name] = [fid for fid, e in out["per_frame"].items() if e["r2"].get(f"{name}{DECISION_SIZE}") is not None and e["r2"].get(f"Sp{DECISION_SIZE}") is not None and e["r2"][f"{name}{DECISION_SIZE}"] < e["r2"][f"Sp{DECISION_SIZE}"] - 0.05]
    out["frames_losing_over_0_05"] = losses
    return out


def precondition(stats: Mapping[str, Any], n_scored_tokens: int) -> dict[str, Any]:
    """The reference rung's fidelity, the pooled c_L gap and each family split's gap, and the token count."""
    pairs = stats["pairs"]
    reference = pairs["rungs"][REFERENCE]
    checks = {"scored_tokens": n_scored_tokens >= MIN_SCORED_TOKENS,
              "reference_c_L": reference["c_L"]["r2"] is not None and reference["c_L"]["r2"] >= PRECONDITION_REFERENCE["c_L"],
              "reference_rows": reference["rows"]["entry_r2"] is not None and reference["rows"]["entry_r2"] >= PRECONDITION_REFERENCE["rows"],
              "reference_dT": reference["dT"]["r2"] is not None and reference["dT"]["r2"] >= PRECONDITION_REFERENCE["dT"],
              "gap_c_L": bool(pairs["evaluable"]["c_L"]),
              "gap_cue_final": bool(stats["splits"]["cue_final"]["evaluable"]), "gap_coordinated": bool(stats["splits"]["coordinated"]["evaluable"])}
    failed = [name for name, ok in checks.items() if not ok]
    return {"ok": not failed, "failed": failed, "checks": checks, "n_scored_tokens": n_scored_tokens, "reference": {"c_L": reference["c_L"]["r2"], "rows": reference["rows"]["entry_r2"], "dT": reference["dT"]["r2"]},
            "gaps": {"pooled": pairs["gaps"]["c_L"], "cue_final": stats["splits"]["cue_final"]["gap_c_L"], "coordinated": stats["splits"]["coordinated"]["gap_c_L"]}}


def routing_test(stats: Mapping[str, Any], outcomes: Sequence[str]) -> dict[str, Any]:
    """Y1/Y2 at the decision size: the gain floor, the frame-count guard and the split guard; a negative result classified by the witness headroom (and E's own gain)."""
    k = str(DECISION_SIZE)
    gain = stats["gains"][k].get(DECISION["full"])
    wins = stats["wins"][k]
    split_cf, split_co = stats["splits"]["cue_final"]["gains"].get(DECISION["full"]), stats["splits"]["coordinated"]["gains"].get(DECISION["full"])
    conditions = {"gain": gain is not None and gain >= ROUTING_GAIN_FLOOR, "frame_count": bool(wins["holds"]),
                  "split_cue_final": split_cf is not None and split_cf >= SPLIT_GAIN_FLOOR, "split_coordinated": split_co is not None and split_co >= SPLIT_GAIN_FLOOR}
    failed = [name for name, ok in conditions.items() if not ok]
    headroom = stats["headroom"]["witness"]
    headroom_present = (headroom is not None and headroom >= HEADROOM_MIN) or (gain is not None and gain >= ROUTING_GAIN_FLOOR)
    if not failed:
        label = outcomes[0]
    elif headroom_present:
        label = outcomes[1]
    else:
        label = outcomes[2]
    return {"label": label, "passed": not failed, "failed": failed, "conditions": conditions, "gain": gain, "wins": {key: wins[key] for key in ("n_frames", "wins", "losses", "ties", "undefined", "holds")},
            "split_gains": {"cue_final": split_cf, "coordinated": split_co}, "headroom_witness": headroom, "headroom_ranking_proxy": stats["headroom"]["ranking_proxy"], "headroom_present": headroom_present,
            "witness_share": stats.get("witness_share"), "floors": {"gain": ROUTING_GAIN_FLOOR, "split": SPLIT_GAIN_FLOOR, "headroom": HEADROOM_MIN}}


def rule_test(sets: Mapping[str, Mapping[str, Any]], pooled: Mapping[str, Any]) -> dict[str, Any]:
    """Y3: ρ_64 pooled ≥ 0.60 and ≥ 0.50 on each set; evaluable iff Δκ_64(E) ≥ 0.05 on each set and pooled."""
    k = str(DECISION_SIZE)
    values = {name: s["gains"][k].get(DECISION["full"]) for name, s in sets.items()}
    values["pooled"] = pooled["gains"][k].get(DECISION["full"])
    rhos = {name: s["rho"][k] for name, s in sets.items()}
    rhos["pooled"] = pooled["rho"][k]
    not_evaluable = [name for name, v in values.items() if v is None or v < RHO_DENOMINATOR_MIN]
    if not_evaluable:
        return {"label": OUTCOME_Y3[2], "evaluable": False, "not_evaluable": not_evaluable, "denominators": values, "rho": rhos}
    holds = {"pooled": rhos["pooled"] is not None and rhos["pooled"] >= RHO_POOLED_FLOOR, **{name: rhos[name] is not None and rhos[name] >= RHO_SET_FLOOR for name in sets}}
    failed = [name for name, ok in holds.items() if not ok]
    return {"label": OUTCOME_Y3[0] if not failed else OUTCOME_Y3[1], "evaluable": True, "failed": failed, "denominators": values, "rho": rhos, "floors": {"pooled": RHO_POOLED_FLOOR, "set": RHO_SET_FLOOR, "denominator": RHO_DENOMINATOR_MIN}}


def template_test(sets: Mapping[str, Mapping[str, Any]], pooled: Mapping[str, Any], new_frames_set: str, evaluable: bool) -> dict[str, Any]:
    """Y4: A = κ(E_64) − κ(T_64), B = Δκ_64(T), on the new-frame set and pooled; three-way plus the set-dependent case."""
    k = str(DECISION_SIZE)

    def ab(s: Mapping[str, Any]) -> tuple[float | None, float | None]:
        kap = s["pairs"]["kappa"]["c_L"]
        e, t = kap.get(DECISION["full"]), kap.get(DECISION["template"])
        return ((e - t) if e is not None and t is not None else None, s["gains"][k].get(DECISION["template"]))

    values = {name: dict(zip(("A", "B"), ab(s))) for name, s in sets.items()}
    values["pooled"] = dict(zip(("A", "B"), ab(pooled)))
    if not evaluable:
        return {"label": OUTCOME_Y4[3], "values": values}
    ins = {name: values[name]["A"] is not None and values[name]["A"] >= TEMPLATE_MARGIN for name in (new_frames_set, "pooled")}
    suf = {name: values[name]["B"] is not None and values[name]["B"] >= TEMPLATE_MARGIN and values[name]["A"] is not None and values[name]["A"] < TEMPLATE_MARGIN for name in (new_frames_set, "pooled")}
    if all(ins.values()):
        label, reason = OUTCOME_Y4[0], "A ≥ margin pooled and on the new frames"
    elif all(suf.values()):
        label, reason = OUTCOME_Y4[1], "B ≥ margin and A < margin pooled and on the new frames"
    elif any(ins.values()) or any(suf.values()):
        label, reason = OUTCOME_Y4[2], "set-dependent: the pooled and the new-frame values sit on different sides of a margin"
    else:
        label, reason = OUTCOME_Y4[2], "neither margin reached on both"
    return {"label": label, "reason": reason, "values": values, "insufficient_holds": ins, "sufficient_holds": suf, "margin": TEMPLATE_MARGIN}


def membership_test(sets: Mapping[str, Mapping[str, Any]], evaluable: Mapping[str, bool]) -> dict[str, Any]:
    """Y5 per set: mean |E_64 ∩ O_64| / 64 ≥ 0.60 and ≥ S'_64's + 0.20; both sets → predicted."""
    per_set = {}
    for name, m in sets.items():
        if not evaluable.get(name) or not m or m.get("n_frames", 0) == 0:
            per_set[name] = {"evaluable": False}
            continue
        e, sp = m["mean"]["E"], m["mean"]["Sp"]
        per_set[name] = {"evaluable": True, "E": e, "Sp": sp, "G": m["mean"]["G"], "T": m["mean"]["T"], "n_frames": m["n_frames"], "floor": e >= MEMBERSHIP_FLOOR, "margin": e - sp >= MEMBERSHIP_MARGIN, "holds": e >= MEMBERSHIP_FLOOR and e - sp >= MEMBERSHIP_MARGIN}
    if any(not v["evaluable"] for v in per_set.values()):
        return {"label": OUTCOME_Y5[2], "per_set": per_set}
    failed = [name for name, v in per_set.items() if not v["holds"]]
    return {"label": OUTCOME_Y5[0] if not failed else OUTCOME_Y5[1], "failed": failed, "per_set": per_set, "floors": {"mean": MEMBERSHIP_FLOOR, "margin": MEMBERSHIP_MARGIN}}


def descriptive_expectations(sets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The predeclared descriptive expectations (i)–(v) of the design, reported against."""
    out = {}
    for name, s in sets.items():
        g = s["gains"]
        kap = s["pairs"]["kappa"]
        rows = kap.get("rows", {})
        randoms = {k: [kap["c_L"].get(f"R{i}_{k}") for i in range(1, N_RANDOM_CONTROLS + 1)] for k in SIZES}
        best_random = {k: max((v for v in randoms[k] if v is not None), default=None) for k in SIZES}
        prospective = {k: {prefix: kap["c_L"].get(f"{prefix}{k}") for prefix in ("Sp", "T", "E", "G")} for k in SIZES}
        out[name] = {"i_other_sizes": {str(k): {"gain_E": g[str(k)].get(f"E{k}"), "holds": g[str(k)].get(f"E{k}") is not None and g[str(k)].get(f"E{k}") >= OTHER_SIZE_GAIN_EXPECTATION} for k in (16, 256)},
                     "ii_inherited": {"gain_over_inherited": (kap["c_L"].get(DECISION["full"]) - kap["c_L"].get(DECISION["inherited"])) if kap["c_L"].get(DECISION["full"]) is not None and kap["c_L"].get(DECISION["inherited"]) is not None else None,
                                      "gain_over_global": g[str(DECISION_SIZE)].get(DECISION["full"]),
                                      "holds": None},
                     "iii_random": {str(k): {"random": randoms[k], "best_random": best_random[k], "below_max": all(v is not None and v < RANDOM_KAPPA_MAX for v in randoms[k]),
                                             "margin_holds": k != DECISION_SIZE or all(v is not None and best_random[k] is not None and v - best_random[k] >= RANDOM_MARGIN for v in prospective[k].values())} for k in SIZES},
                     "iv_orderings": {str(k): {"c_L_over_Pi": all(kap["c_L"].get(f"{p}{k}") is not None and kap["Pi"].get(f"{p}{k}") is not None and kap["c_L"][f"{p}{k}"] >= kap["Pi"][f"{p}{k}"] for p in ("Sp", "T", "E", "G")),
                                               "row_diffuse": all(rows.get(f"{p}{k}") is not None and rows[f"{p}{k}"] < ROW_DIFFUSE_MAX for p in ("Sp", "T", "E", "G"))} for k in SIZES},
                     "v_G_cardinal": {template: s["splits"]["per_template"][template]["gains"].get(DECISION["rule"]) for template in pm.TEMPLATE_ORDER},
                     "vi_single": {"Sp1": kap["c_L"].get(GLOBAL_SINGLE), "E1": kap["c_L"].get(FRAME_SINGLE)},
                     "vii_frames": {"losing_over_0_05": s["frames_losing_over_0_05"], "jackknife": {n: {"min": j["min"], "max": j["max"]} for n, j in s["jackknife"].items()}},
                     "viii_head_objects": {str(k): {obj: {p: kap[obj].get(f"{p}{k}") for p in ("Sp", "T", "E", "G", "O", "Os")} for obj in ("F", "Pi", "dT", "rows")} for k in SIZES},
                     "x_witness": {str(k): {"kappa_witness": kap["c_L"].get(f"Os{k}"), "kappa_ranking": kap["c_L"].get(f"O{k}"), "gain_witness": g[str(k)].get(f"Os{k}"), "gain_ranking": g[str(k)].get(f"O{k}")} for k in SIZES},
                     "witness_share": s.get("witness_share")}
        ii = out[name]["ii_inherited"]
        ii["holds"] = ii["gain_over_inherited"] is not None and ii["gain_over_global"] is not None and ii["gain_over_inherited"] >= ii["gain_over_global"]
    return out


# ---------------------------------------------------------------------------
# Scoring of a set and of the confirmation.


def scored_population(pairs: Mapping[str, Mapping[str, Any]], words: Sequence[str]) -> tuple[dict[str, Any], list[str], list[tuple[Mapping[str, Any], dict[str, Any]]]]:
    """Tokens with at least MIN_VALID_FRAMES_PER_TOKEN measured frames are scored; their pairs form the set's population (each pair with its merged prediction)."""
    tokens_out, scored_pairs = {}, []
    for word in words:
        keys = sorted(key for key in pairs if key.split("|")[0] == word)
        analyses = [pairs[key] for key in keys]
        predictions = [merged_prediction(a) for a in analyses]
        tokens_out[word] = {"scored": len(keys) >= MIN_VALID_FRAMES_PER_TOKEN, "n_valid_frames": len(keys), **token_means(analyses, predictions)}
        if len(keys) >= MIN_VALID_FRAMES_PER_TOKEN:
            scored_pairs.extend(zip(analyses, predictions))
    scored = [word for word, entry in tokens_out.items() if entry["scored"]]
    return tokens_out, scored, scored_pairs


def score_set(pairs: Mapping[str, Mapping[str, Any]], words: Sequence[str], *, outcomes: Sequence[str], lists: Mapping[str, Any], frame_lists_by_frame: Mapping[str, Mapping[str, Any]], oracle_lists: Mapping[str, Mapping[str, Any]],
              frame_templates: Mapping[str, str], frame_pc: Mapping[str, int]) -> dict[str, Any]:
    tokens_out, scored, scored_pairs = scored_population(pairs, words)
    result: dict[str, Any] = {"tokens": tokens_out, "scored_tokens": scored, "n_pairs": len(scored_pairs), "n_frames": len({a["frame_id"] for a, _ in scored_pairs})}
    if len(scored) < 2:
        result["precondition"] = {"ok": False, "failed": ["scored_tokens"], "n_scored_tokens": len(scored)}
        result["test"] = {"label": outcomes[3], "passed": False, "failed": ["precondition"]}
        return result
    analyses = [a for a, _ in scored_pairs]
    predictions = [p for _, p in scored_pairs]
    stats = set_statistics(analyses, predictions)
    result["statistics"] = stats
    result["precondition"] = precondition(stats, len(scored))
    result["membership"] = membership(frame_lists_by_frame, lists, lists["T"], {fid: oracle_lists[fid] for fid in oracle_lists if fid in stats["per_frame"]}, frame_templates, frame_pc)
    if result["precondition"]["ok"]:
        result["test"] = routing_test(stats, outcomes)
    else:
        result["test"] = {"label": outcomes[3], "passed": False, "failed": ["precondition"] + result["precondition"]["failed"]}
    return result


def score_confirmation(stage1: Mapping[str, Any], pairs_exposed: Mapping[str, Mapping[str, Any]], pairs_fresh: Mapping[str, Mapping[str, Any]], confirmation: Confirmation019, lock: Mapping[str, Any],
                       oracle_lists: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Y1 on the exposed frames, Y2 on the valid fresh frames, the pooled population for Y3/Y4, Y5 per set, the descriptive expectations and the outcome."""
    words = [token["word"] for token in confirmation.tokens]
    lists = lock["selectors"]["lists"]
    frame_templates = {**{fid: st["template"] for fid, st in lock["frame_meta"].items()}, **{fid: entry["template_id"] for fid, entry in stage1["frames"].items()}}
    frame_pc = {**{fid: int(st["p_c"]) for fid, st in lock["frame_meta"].items()}, **{fid: int(entry["p_c"]) for fid, entry in stage1["frames"].items() if "p_c" in entry}}
    exposed_frame_lists = {fid: {"E": lists["E"][fid], "G": lists["G"][fid], "E1": lists["E1"][fid]} for fid in lists["E"]}
    fresh_frame_lists = {fid: {"E": sel["E"], "G": sel["G"], "E1": sel["E1"]} for fid, sel in stage1["frame_selectors"].items()}
    y1 = score_set(pairs_exposed, words, outcomes=OUTCOME_Y1, lists=lists, frame_lists_by_frame=exposed_frame_lists, oracle_lists=oracle_lists.get("Y1", {}), frame_templates=frame_templates, frame_pc=frame_pc)
    y2 = score_set(pairs_fresh, words, outcomes=OUTCOME_Y2, lists=lists, frame_lists_by_frame=fresh_frame_lists, oracle_lists=oracle_lists.get("Y2", {}), frame_templates=frame_templates, frame_pc=frame_pc)
    valid_frames = {fid: entry for fid, entry in stage1["frames"].items() if entry.get("valid")}
    n_cue_final = sum(1 for e in valid_frames.values() if e.get("cue_final"))
    n_coordinated = sum(1 for e in valid_frames.values() if not e.get("cue_final"))
    if n_cue_final < MIN_VALID_CUE_FINAL_FRAMES or n_coordinated < MIN_VALID_COORDINATED_FRAMES:
        y2["precondition"] = {**y2.get("precondition", {}), "ok": False, "failed": sorted(set(y2.get("precondition", {}).get("failed", [])) | {"valid_fresh_frames"})}
        y2["test"] = {"label": OUTCOME_Y2[3], "passed": False, "failed": ["precondition", "valid_fresh_frames"]}
    both_ok = bool(y1["precondition"].get("ok")) and bool(y2["precondition"].get("ok"))
    pooled_stats = None
    if both_ok:
        _, _, p1 = scored_population(pairs_exposed, words)
        _, _, p2 = scored_population(pairs_fresh, words)
        pooled_pairs = p1 + p2
        pooled_stats = set_statistics([a for a, _ in pooled_pairs], [p for _, p in pooled_pairs])
    sets = {"Y1": y1.get("statistics"), "Y2": y2.get("statistics")}
    if both_ok and all(sets.values()):
        y3 = rule_test(sets, pooled_stats)
        y4 = template_test(sets, pooled_stats, "Y2", True)
    else:
        y3 = {"label": OUTCOME_Y3[2], "evaluable": False, "not_evaluable": [name for name, y in (("Y1", y1), ("Y2", y2)) if not y["precondition"].get("ok")]}
        y4 = template_test({name: s for name, s in sets.items() if s}, pooled_stats or next(s for s in sets.values() if s), "Y2", False) if any(sets.values()) else {"label": OUTCOME_Y4[3], "values": {}}
    y5 = membership_test({"Y1": y1.get("membership"), "Y2": y2.get("membership")}, {"Y1": bool(y1["precondition"].get("ok")), "Y2": bool(y2["precondition"].get("ok"))})
    described = {name: s for name, s in sets.items() if s}
    if pooled_stats is not None:
        described["pooled"] = pooled_stats
    results = {"Y1": y1, "Y2": y2, "Y3": y3, "Y4": y4, "Y5": y5, "pooled": {"n_pairs": pooled_stats["n_pairs"], "gains": pooled_stats["gains"], "rho": pooled_stats["rho"], "kappa_c_L": pooled_stats["pairs"]["kappa"]["c_L"], "headroom": pooled_stats["headroom"]} if pooled_stats else None,
               "descriptive": descriptive_expectations(described), "valid_frames": {fid: {"valid": True, "template_id": e["template_id"], "cue_final": e.get("cue_final")} for fid, e in valid_frames.items()},
               "precondition_Y1": y1["precondition"], "precondition_Y2": y2["precondition"], "terminology": WITNESS_TERMINOLOGY}
    results["outcome"] = outcome(results)
    return results


def outcome(results: Mapping[str, Any]) -> dict[str, Any]:
    labels = [results["Y1"]["test"]["label"], results["Y2"]["test"]["label"], results["Y3"]["label"], results["Y4"]["label"], results["Y5"]["label"]]
    return {"label": " | ".join(labels), "labels": labels}


# ---------------------------------------------------------------------------
# Components, the licensed pool and the frame selectors of one state.


def _components(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], bases_3: Mapping[str, Mapping[str, torch.Tensor | None]], base2_pt: torch.Tensor | None) -> dict[str, Any]:
    parts = bc._components(model, pool, pool_010, lock_011, lock_012, bases_3, base2_pt, None)
    chain = RoutingChain(bc.MaskedChainModel(parts["model"], base2_pt, {}))
    out = dict(parts)
    out["chain"] = chain
    out["context_019"] = AnalysisContext(chain, parts["context"], parts["weights"], parts["axis_T"], parts["read_out"])
    return out


def licensed_pool(extract_entries: Mapping[str, Any], pool: cs.Pool008) -> list[tuple[str, str, str, int]]:
    return bc.ranking_pool(extract_entries, pool)


def frame_selectors_from_state(chain: RoutingChain, weights: pm.Weights, read_out: torch.Tensor, locked: Mapping[str, Any], template: str, tokens: Sequence[tuple[str, int]], quantiles: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
                               denominator: float, forbidden_ids: set[int]) -> dict[str, Any]:
    """A fresh frame's E and G lists from its reference run alone: the exposed cues' predicted effects at its state and its pre-activations. A fresh token id is refused."""
    if any(token_id in forbidden_ids for _, token_id in tokens):
        raise PhaseError("a fresh token was offered to the full-evaluation selector; only exposed cues may enter")
    if not tokens:
        raise pm.IncidentError("the full-evaluation selector has no exposed cue")
    read_out_abs = read_out.abs()
    x1_all, x2_all = hp._tensors(locked["x1_all"]), hp._tensors(locked["x2_all"])
    p_c, p_t = int(locked["p_c"]), int(locked["p_t"])
    positions = sorted({p_c, p_t})
    acc = {str(pos): _MeanAccumulator() for pos in positions}
    for _, token_id in tokens:
        up, _ = chain.upstream_parts(weights, x1_all, x2_all, p_c, p_t, token_id, template)
        for pos in positions:
            acc[str(pos)].add(read_unit_effects(up, read_out_abs, pos))
    e_scores = {pos: a.mean() for pos, a in acc.items()}
    g = {}
    for pos in positions:
        pt = position_type(pos, p_c)
        q = [torch.tensor(quantiles[template][pt][str(qq)], dtype=torch.float64) for qq in DRIVE_QUANTILES]
        g[str(pos)] = g_scores(chain.lw.pre_activations(BLOCK, x2_all[pos].double()), chain.lw.pre_activations(BLOCK, template_base(chain, template, pos, p_c)), q, read_out_abs, denominator)
    return {"E": frame_lists(e_scores), "G": frame_lists(g), "E1": {pos: top_k(s, 1) for pos, s in e_scores.items()}, "scores": {"E": {pos: s.tolist() for pos, s in e_scores.items()}, "G": {pos: s.tolist() for pos, s in g.items()}}}


def selector_overlaps(lists: Mapping[str, Any], own: Mapping[str, Any], template: str, p_c: int, k: int = DECISION_SIZE) -> dict[str, Any]:
    out = {}
    for pos_str in own["E"]:
        pt = position_type(int(pos_str), p_c)
        e, g, sp, t = own["E"][pos_str][str(k)], own["G"][pos_str][str(k)], lists["Sp"][pt][str(k)], lists["T"][template][pt][str(k)]
        out[pos_str] = {"E&Sp": overlap(e, sp), "E&T": overlap(e, t), "E&G": overlap(e, g), "G&Sp": overlap(g, sp), "T&Sp": overlap(t, sp)}
    return out


def _check_previous_state(state_f: hp.FrameState017, lock_018: Mapping[str, Any], inherited_extract: Mapping[str, Any]) -> None:
    frame_id = state_f.frame.frame_id
    previous = lock_018.get("locked_states", {}).get(frame_id)
    if previous is not None:
        hp._check_locked_state_017(state_f, previous, frame_id)
    elif frame_id in inherited_extract.get("stage1_state_digests", {}):
        if ap.state_digest(hp.locked_state(state_f)) != inherited_extract["stage1_state_digests"][frame_id]:
            raise pm.IncidentError(f"{frame_id}: the reference state differs from Experiment 018's digested stage-1 state")
    else:
        raise pm.IncidentError(f"{frame_id}: an exposed frame without an Experiment 018 record")


def frame_meta(pool: cs.Pool008, states: Mapping[str, hp.FrameState017]) -> dict[str, Any]:
    return {frame.frame_id: {"template": frame.template_id, "p_c": states[frame.frame_id].p_c, "p_t": states[frame.frame_id].p_t, "cue_final": states[frame.frame_id].p_t == states[frame.frame_id].p_c} for frame in pool.frames if frame.frame_id in states}


# ---------------------------------------------------------------------------
# Exploration (Tier A).


def run_exploration(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, *, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_018: Mapping[str, Any], inherited_extract: Mapping[str, Any],
                    confirmation_token_ids: set[int], state: dict[str, Any], results_path: Path | None, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    nouns = pool.single_nouns
    exploration = state["exploration"]
    bases_3 = hp.bases_from_json(lock_018["bases_3"])
    base2_pt = torch.tensor(lock_018["base2_pt"]["vector"], dtype=torch.float64) if lock_018["base2_pt"]["vector"] is not None else None
    if base2_pt is None:
        raise pm.IncidentError("the Experiment 018 lock carries no block-2 base at p_t")
    say("stage axes from the Experiment 010 pool, checked against the Experiment 011 lock; Experiment 017's layer-3 bases and Experiment 018's block-2 base at p_t")
    parts = _components(model, pool, pool_010, lock_011, lock_012, bases_3, base2_pt)
    weights, head, axis_T, e_axis, programs, chain, read_out = parts["weights"], parts["head"], parts["axis_T"], parts["e_axis"], parts["programs"], parts["chain"], parts["read_out"]
    exploration["read_weight_check"] = lc.check_read_weight(parts["fpm"].read, head, axis_T)
    exploration["axes_vectors"] = {"T": axis_T.direction.tolist(), "R0": torch.tensor(lock_011["axes_vectors"]["R0"], dtype=torch.float64).tolist()}
    exploration["sigma_T"] = axis_T.sigma
    exploration["read_weight"] = parts["fpm"].read.weight.tolist()
    exploration["defined_templates"] = list(lock_012["defined_templates"])
    exploration["program"] = atp._program_record(programs)
    exploration["bases_3"] = dict(lock_018["bases_3"])
    exploration["base2_pt"] = dict(lock_018["base2_pt"])
    exploration["lock_018_sha256"] = lock_018["content_sha256"]
    say(f"reference states of the {len(pool.frames)} exposed frames, checked against Experiment 018")
    states: dict[str, hp.FrameState017] = {}
    identities: dict[str, float] = {}
    for frame in pool.frames:
        state_f = hp.capture_frame_017(model, head, pool.reference_prompt(frame), nouns, axis_T)
        rows16 = atp.reference_rows({layer: programs[layer] for layer in hp.UPSTREAM_LAYERS}, state_f.state.x1_all, state_f.state.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows16, state_f.state))
        identities["I5_reference_head_row"] = max(identities.get("I5_reference_head_row", 0.0), hp.check_reference_row(state_f, programs[HEAD_LAYER]))
        _check_previous_state(state_f, lock_018, inherited_extract)
        states[frame.frame_id] = state_f
    exploration["locked_states"] = {frame_id: hp.locked_state(s) for frame_id, s in states.items()}
    exploration["locked_state_digests"] = {frame_id: ap.state_digest(entry) for frame_id, entry in exploration["locked_states"].items()}
    exploration["frame_meta"] = frame_meta(pool, states)
    # The selectors: from the locked states, the weights and the exposed cues' ΔE only, before any pair is measured.
    licensed = licensed_pool(inherited_extract["entries"], pool)
    licensed_keys = {f"{word}|{frame_id}" for word, frame_id, _, _ in licensed}
    tokens_by_template = bc.tokens_by_template(licensed)
    if any(token_id in confirmation_token_ids for tokens in tokens_by_template.values() for _, token_id in tokens):
        raise PhaseError("a confirmation token is in the exposed pool")
    replication = {"frame_ids": {frame.frame_id for frame in pool.frames}, "stage1_frame_ids": set(inherited_extract["stage1_state_digests"]),
                   "token_ids_by_template": {template: {token_id for word, token_id in tokens if pool.token_source.get(word) != "confirmation-018"} for template, tokens in tokens_by_template.items()},
                   "licensed_keys": {key for key, entry in inherited_extract["entries"].items() if entry["set"] == "explore"}}
    say(f"the selectors over the {len(licensed)} licensed pairs and the templates' exposed cues at every frame (predicted changes only)")
    selectors = selectors_from_states(chain, weights, read_out, exploration["locked_states"], pool.frames, tokens_by_template, licensed_keys, lock_018["subsets"], replication=replication, log=say)
    e_018 = selectors.pop("e_018_lists")
    recorded = {**inherited_extract["frame_subsets_explore"], **inherited_extract["frame_subsets_stage1"]}
    mismatched = [frame_id for frame_id, mine in e_018.items() if [int(i) for i in mine] != [int(i) for i in recorded[frame_id]]]
    if mismatched:
        raise pm.IncidentError(f"I9: the E rule with Experiment 018's inputs does not reproduce its recorded per-frame lists in {len(mismatched)} frames (e.g. {mismatched[:3]})")
    identities["I9_frame_lists"] = 0.0
    exploration["selectors"] = selectors
    exploration["i9_replication"] = {"passed": True, "n_frames": len(e_018)}
    exploration["licensed_pool"] = {"n_pairs": len(licensed), "tokens_by_template": {t: len(v) for t, v in tokens_by_template.items()}}
    say(f"selectors locked (digest {selectors['digest'][:16]}…): Sp1 {selectors['lists']['Sp1']}, Sp64 first eight {selectors['lists']['Sp']['p_c']['64'][:8]}; I9 held on {len(e_018)} frames")
    context = parts["context_019"]
    recorded_entries = inherited_extract["entries"]
    say(f"re-measurement of the {len(recorded_entries)} recorded pairs with the layer-3 residuals, the head's row and the block-2 input captured; every prospective rung predicted")
    pairs: dict[str, dict[str, Any]] = {}
    measured_extract: dict[str, dict[str, Any]] = {}
    effects: dict[str, dict[int, torch.Tensor]] = {}
    contributions: dict[str, torch.Tensor] = {}
    denominators = selectors["denominators"]
    for frame in pool.frames:
        state_f = states[frame.frame_id]
        names = [name for name, _ in pool.tokens if f"{name}|{frame.frame_id}" in recorded_entries]
        if not names:
            continue
        masks = frame_rung_masks(selectors["lists"], frame.frame_id, frame.template_id, state_f.p_c, state_f.p_t)
        plural_name = pool.plural_cue[frame.template_id]
        records = {name: measure_pair(model, weights, head, state_f, name, pool.token_id(name), e_axis, axis_T, nouns) for name in names}
        plural = records[plural_name] if plural_name in records else measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for name, record in records.items():
            out = analyse_pair_019(record, plural, context=context, state=state_f, rung_masks=masks)
            key = f"{name}|{frame.frame_id}"
            if out is None:
                raise pm.IncidentError(f"{key}: no analysis although Experiment 018 recorded the pair")
            er._max_errors(identities, out["analysis"]["identities"])
            pairs[key] = out["analysis"]
            effects[key] = out["effects"]
            contributions[key] = out["u"]
            measured_extract[key] = extract_entry_from_analysis(out["analysis"])
        say(f"  {frame.frame_id}: {len(names)} pairs")
    exploration["replication"] = {"experiment_018": check_extract_replication(measured_extract, recorded_entries)}
    say(f"replication against Experiment 018: max deviation {exploration['replication']['experiment_018']['max_abs_deviation']:.2e}")
    say("the exposed oracles (cue-in-sample ranking; leave-one-cue-out witness) and their rungs")
    oracle_lists, witness_lists = oracles_for_set(pairs, effects, contributions, [name for name, _ in pool.tokens], read_out.abs(), denominators)
    for key, analysis in pairs.items():
        frame_id = analysis["frame_id"]
        analysis["oracle"] = oracle_rungs_for_frame(chain, weights, states[frame_id], analysis["token_id"], analysis["template"], oracle_lists[frame_id], witness_lists[key])
    exploration["identities"] = identities
    analyses = list(pairs.values())
    predictions = [merged_prediction(a) for a in analyses]
    stats = set_statistics(analyses, predictions)
    frame_lists_by_frame = {fid: {"E": selectors["lists"]["E"][fid], "G": selectors["lists"]["G"][fid], "E1": selectors["lists"]["E1"][fid]} for fid in selectors["lists"]["E"]}
    exploration["exposed_check"] = {"n_pairs": len(analyses), "statistics": stats, "membership": membership(frame_lists_by_frame, selectors["lists"], selectors["lists"]["T"], oracle_lists, {f: m["template"] for f, m in exploration["frame_meta"].items()}, {f: m["p_c"] for f, m in exploration["frame_meta"].items()}),
                                   "descriptive": descriptive_expectations({"exposed": stats}), "oracle_lists": oracle_lists}
    exploration["pairs"] = {key: compact_analysis(value) for key, value in pairs.items()}
    k = str(DECISION_SIZE)
    kap = stats["pairs"]["kappa"]["c_L"]
    exploration["summary"] = {"defined_templates": exploration["defined_templates"], "kappa_global": kap.get(DECISION["global"]), "kappa_template": kap.get(DECISION["template"]), "kappa_full": kap.get(DECISION["full"]), "kappa_rule": kap.get(DECISION["rule"]),
                              "kappa_inherited": kap.get(DECISION["inherited"]), "kappa_ranking_oracle": kap.get(DECISION["oracle"]), "kappa_witness": kap.get(DECISION["witness"]), "gains": stats["gains"][k], "rho": stats["rho"][k],
                              "wins": stats["wins"][k]["wins"], "n_frames": stats["wins"][k]["n_frames"], "headroom": stats["headroom"], "reference_c_L_r2": stats["pairs"]["rungs"][REFERENCE]["c_L"]["r2"],
                              "reference_rows_r2": stats["pairs"]["rungs"][REFERENCE]["rows"]["entry_r2"], "gaps": stats["pairs"]["gaps"], "prediction_undefined": len(exploration["defined_templates"]) < 2}
    f = cd._f
    s = exploration["summary"]
    say(f"exposed (cue-in-sample): κ_c_L at k=64 — S' {f(s['kappa_global'], 3)}, T {f(s['kappa_template'], 3)}, E {f(s['kappa_full'], 3)}, G {f(s['kappa_rule'], 3)}, L {f(s['kappa_inherited'], 3)}, O {f(s['kappa_ranking_oracle'], 3)}, O* {f(s['kappa_witness'], 3)}; ρ {f(s['rho'], 3)}; E wins {s['wins']}/{s['n_frames']}; reference rung c_L R² {f(s['reference_c_L_r2'], 3)}")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


# ---------------------------------------------------------------------------
# Lock and its artifacts.


def prediction_table(chain: RoutingChain, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], lists: Mapping[str, Any], frames: Sequence[pm.Frame], tokens: Sequence[Mapping[str, Any]], defined: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        if frame.template_id not in defined:
            continue
        locked = locked_states[frame.frame_id]
        masks = frame_rung_masks(lists, frame.frame_id, frame.template_id, int(locked["p_c"]), int(locked["p_t"]))
        for token in tokens:
            rows.append(prediction_row(token["word"], frame.frame_id, frame.template_id, chain.predict_from_locked(weights, locked, token["token_id"], frame.template_id, masks)))
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


def lock_predictions(chain: RoutingChain, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], lists: Mapping[str, Any], pool: cs.Pool008, confirmation: Confirmation019, defined: Sequence[str]) -> dict[str, Any]:
    rows = prediction_table(chain, weights, locked_states, lists, pool.frames, confirmation.tokens, defined)
    return {"rows": rows, "token_means": token_means_from_table(rows, [token["word"] for token in confirmation.tokens])}


def frozen_floors() -> dict[str, Any]:
    return {"sizes": list(SIZES), "decision_size": DECISION_SIZE, "drive_quantiles": list(DRIVE_QUANTILES), "n_random_controls": N_RANDOM_CONTROLS, "routing_gain_floor": ROUTING_GAIN_FLOOR, "split_gain_floor": SPLIT_GAIN_FLOOR,
            "frame_count_rule": "strictly more than half of the valid frames; a tie is not a win", "rho_pooled_floor": RHO_POOLED_FLOOR, "rho_set_floor": RHO_SET_FLOOR, "rho_denominator_min": RHO_DENOMINATOR_MIN,
            "template_margin": TEMPLATE_MARGIN, "membership_floor": MEMBERSHIP_FLOOR, "membership_margin": MEMBERSHIP_MARGIN, "headroom_min": HEADROOM_MIN, "gap_min": GAP_MIN, "precondition_reference": dict(PRECONDITION_REFERENCE),
            "other_size_gain_expectation": OTHER_SIZE_GAIN_EXPECTATION, "random_kappa_max": RANDOM_KAPPA_MAX, "random_margin": RANDOM_MARGIN, "row_diffuse_max": ROW_DIFFUSE_MAX,
            "min_valid_cue_final_frames": MIN_VALID_CUE_FINAL_FRAMES, "min_valid_coordinated_frames": MIN_VALID_COORDINATED_FRAMES, "min_valid_frames_per_token": MIN_VALID_FRAMES_PER_TOKEN, "min_scored_tokens": MIN_SCORED_TOKENS,
            "i8_tolerance": I8_TOLERANCE, "i10_tolerance": I10_TOLERANCE, "replication_tolerance": REPLICATION_TOLERANCE, "frame_cue_effect_rate": FRAME_CUE_EFFECT_RATE, "head_stage_floor": cs.STAGE_UNINFORMATIVE_FLOOR,
            "witness": "leave-one-cue-out forward greedy on the frame's other scored cues' contribution vectors and measured reads; lower-index ties; no early stop; p_c only; stage 2 only",
            "outcomes": {"Y1": list(OUTCOME_Y1), "Y2": list(OUTCOME_Y2), "Y3": list(OUTCOME_Y3), "Y4": list(OUTCOME_Y4), "Y5": list(OUTCOME_Y5)}}


def build_candidate_lock(*, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation019, predictions: Mapping[str, Any], protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    if exploration["summary"]["prediction_undefined"]:
        raise PhaseError("fewer than two templates are defined: PREDICTION_UNDEFINED, no lock")
    lock = {"schema_version": 1, "experiment": "019", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit}
    for field, key in zip(STATE_DIGEST_FIELDS, DIGEST_KEYS):
        lock[field] = digests[key] if key != "confirmation_019" else confirmation.content_sha256
    lock.update({"confirmation_prompt_keys": prompt_key_manifest(confirmation), "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
                 "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED}, "head": HEAD_KEY, "block": BLOCK, "program": exploration["program"],
                 "axes_vectors": exploration["axes_vectors"], "sigma_T": exploration["sigma_T"], "read_weight": exploration["read_weight"], "defined_templates": exploration["defined_templates"],
                 "bases_3": exploration["bases_3"], "base2_pt": exploration["base2_pt"], "lock_018_sha256": exploration["lock_018_sha256"], "locked_states": exploration["locked_states"], "locked_state_digests": exploration["locked_state_digests"],
                 "frame_meta": exploration["frame_meta"], "selectors": exploration["selectors"], "floors": frozen_floors(), "tokens": [dict(token) for token in confirmation.tokens],
                 "exposed_check": {"summary": exploration["summary"], "gains": exploration["exposed_check"]["statistics"]["gains"], "kappa_c_L": exploration["exposed_check"]["statistics"]["pairs"]["kappa"]["c_L"]},
                 "predictions": dict(predictions), "tier_a_results_sha256": state.get("state_sha256"), "terminology": WITNESS_TERMINOLOGY})
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation019, predictions_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None,
                  inherited_subsets: Mapping[str, Sequence[int]]) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "019" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked or git_state.get("dirty", True):
        raise PhaseError("confirm requires the committed lock and predictions on a clean Git tree")
    expected = dict(zip(STATE_DIGEST_FIELDS, digest_tuple(digests)))
    expected["confirmation_019_sha256"] = confirmation.content_sha256
    if any(lock.get(field) != value for field, value in expected.items()):
        raise PhaseError("lock was built against different frozen inputs")
    if lock.get("lock_018_sha256") != digests["lock_018"]:
        raise PhaseError("the lock names a different Experiment 018 lock")
    if not state.get("lock") or lock["content_sha256"] != state["lock"]["content_sha256"] or lock["run_id"] != state["run_id"]:
        raise PhaseError("the installed lock is not the candidate lock written by the lock phase")
    if pm.sha256_text(predictions_text) != state["lock"]["predictions_sha256"]:
        raise PhaseError("the installed predictions.md is not the artifact written by the lock phase")
    if dict(lock["floors"]) != frozen_floors():
        raise PhaseError("the lock's floors are not the frozen constants")
    if list(lock["confirmation_prompt_keys"]) != prompt_key_manifest(confirmation):
        raise PhaseError("the lock's prompt-key manifest is not the confirmation set's")
    check_selector_lists(lock["selectors"], inherited_subsets)
    if lock["selectors"]["digest"] != selectors_digest(lock["selectors"]):
        raise PhaseError("the lock's selector digest does not verify")
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
        raise PhaseError(f"the weights, locked axes, bases, lists and locked reference states do not reproduce the preregistered predictions (max difference {worst:.3e}); nothing was executed")
    return worst


def assert_selectors_reproduced(lock: Mapping[str, Any], recomputed: Mapping[str, Any]) -> float:
    """At confirm the selectors are recomputed from the locked states and must give the locked lists exactly (the scores within floating-point noise)."""
    if json.loads(pm.canonical_json(recomputed["lists"])) != json.loads(pm.canonical_json(lock["selectors"]["lists"])):
        raise PhaseError("the selectors recomputed from the locked states do not give the locked lists; nothing was executed")
    worst = 0.0
    for name in ("Sp", "T", "E", "G"):
        worst = max(worst, atp._max_numeric_difference(recomputed["scores"][name], lock["selectors"]["scores"][name], f"scores/{name}"))
    worst = max(worst, atp._max_numeric_difference(recomputed["drive_quantiles"], lock["selectors"]["drive_quantiles"], "drive_quantiles"))
    if worst > LOCK_PREDICTION_TOLERANCE:
        raise PhaseError(f"the selector scores recomputed from the locked states differ from the lock by {worst:.3e}; nothing was executed")
    return worst


def render_predictions(lock: Mapping[str, Any]) -> str:
    f = cd._f
    lists = lock["selectors"]["lists"]
    lines = ["# Experiment 019 — preregistered predictions (frame-conditioned routing of channel D; every prospective rung of the per-position masked chain)", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_019_sha256']}`; Experiment 018 lock sha256 `{lock['lock_018_sha256']}`",
             f"- Decision size k = {DECISION_SIZE}; Y1/Y2 Δκ_64(E) ≥ {ROUTING_GAIN_FLOOR} pooled, E_64 above S'_64 in strictly more than half of the valid frames, split gains ≥ {SPLIT_GAIN_FLOOR}; a negative result is NOT_EVALUABLE iff the witness headroom H*_64 < {HEADROOM_MIN} and E's gain < {ROUTING_GAIN_FLOOR}; "
             f"Y3 ρ_64 ≥ {RHO_POOLED_FLOOR} pooled and ≥ {RHO_SET_FLOOR} per set (denominators ≥ {RHO_DENOMINATOR_MIN}); Y4 margins {TEMPLATE_MARGIN} pooled and on Y2; Y5 per set mean overlap ≥ {MEMBERSHIP_FLOOR} and ≥ S' + {MEMBERSHIP_MARGIN}; "
             f"precondition: reference rung c_L ≥ {PRECONDITION_REFERENCE['c_L']}, rows ≥ {PRECONDITION_REFERENCE['rows']}, ΔT ≥ {PRECONDITION_REFERENCE['dT']}, c_L gaps ≥ {GAP_MIN} pooled and per family split",
             f"- Locked lists: S'_1 {lists['Sp1']}; S'_64 at p_c (first 16) {lists['Sp']['p_c']['64'][:16]}…; inherited S_64 (first 16) {lists['L']['64'][:16]}…; E_64 and G_64 per exposed frame and position in the lock",
             f"- {WITNESS_TERMINOLOGY}",
             f"- Y1 table: {len(lock['predictions']['rows'])} rows (fresh tokens × exposed frames), each with every prospective rung's ĉ_L, F̂, Π̂, ΔT̂ and head row; Y2 rows are computed at confirm stage 1 and digested before any fresh cue prompt", "",
             "| token | class | frames | **ĉ_L E_64** | ĉ_L S'_64 | ĉ_L G_64 | ĉ_L T_64 | ĉ_L S_0 | ĉ_L S_2048 | **ΔT̂ E_64** | ΔT̂ S'_64 | ΔT̂ S_2048 | frozen ΔT̂ |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    categories = {token["word"]: token["category"] for token in lock["tokens"]}
    means = lock["predictions"]["token_means"]
    for word, m in sorted(means.items(), key=lambda kv: -kv[1][f"dT_{REFERENCE}"]):
        lines.append(f"| {word} | {categories.get(word, '')} | {m['n_frames']} | **{f(m['c_L_E64'], 4)}** | {f(m['c_L_Sp64'], 4)} | {f(m['c_L_G64'], 4)} | {f(m['c_L_T64'], 4)} | {f(m[f'c_L_{TEMPLATE_BASE}'], 4)} | {f(m[f'c_L_{REFERENCE}'], 4)} | **{f(m['dT_E64'], 4)}** | {f(m['dT_Sp64'], 4)} | {f(m[f'dT_{REFERENCE}'], 4)} | {f(m['dT_frozen'], 4)} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confirmation: stage 1 (reference runs, the fresh frames' selectors, the table), the barrier, stage 2 (the fresh cues, the oracles, scoring).


def stage_one(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation019, lock: Mapping[str, Any], lock_012: Mapping[str, Any], inherited_extract: Mapping[str, Any], *, protocol_code_commit: str, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    base2_pt = torch.tensor(lock["base2_pt"]["vector"], dtype=torch.float64) if lock["base2_pt"]["vector"] is not None else None
    parts = _components(model, pool, pool_010, lock, lock_012, hp.bases_from_json(lock["bases_3"]), base2_pt)
    weights, head, cache, axis_T, e_axis, chain, read_out = parts["weights"], parts["head"], parts["cache"], parts["axis_T"], parts["e_axis"], parts["chain"], parts["read_out"]
    if atp._program_record(chain.hcm.programs) != dict(lock["program"]):
        raise pm.IncidentError("the program's frozen configuration differs from the lock's record")
    nouns = pool.single_nouns
    tokens_by_template = bc.tokens_by_template(licensed_pool(inherited_extract["entries"], pool))
    forbidden = {token["token_id"] for token in confirmation.tokens}
    lists = lock["selectors"]["lists"]
    say("stage 1: fresh frames' reference states, validity, the frames' own E and G lists, the prediction table — no fresh cue prompt")
    frames_out: dict[str, Any] = {}
    states: dict[str, Any] = {}
    frame_selectors: dict[str, Any] = {}
    overlaps: dict[str, Any] = {}
    rows_out: list[dict[str, Any]] = []
    identities: dict[str, float] = {}
    for frame in confirmation.frames:
        template = frame.template_id
        if template not in lock["defined_templates"]:
            frames_out[frame.frame_id] = {"template_id": template, "valid": False, "template_defined": False, "note": "template excluded before the lock; nothing run"}
            continue
        state_f = hp.capture_frame_017(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        rows16 = atp.reference_rows(chain.hcm.fcm.programs, state_f.state.x1_all, state_f.state.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows16, state_f.state))
        identities["I5_reference_head_row"] = max(identities.get("I5_reference_head_row", 0.0), hp.check_reference_row(state_f, chain.hcm.program3))
        sg, pl = pm.frame_prompts(frame)
        c_a, c_b = cache.c(sg), cache.c(pl)
        positive = sum(1 for noun in nouns if c_a[noun.lexical_key] - c_b[noun.lexical_key] > 0)
        required = pm.exact_count_floor(FRAME_CUE_EFFECT_RATE, len(nouns))
        plural = measure_pair(model, weights, head, state_f, pool.plural_cue[template], pl.cue_token_id, e_axis, axis_T, nouns)
        er._max_errors(identities, er.enforce_identities(plural, plural, f"{frame.frame_id}/{pool.plural_cue[template]}"))
        head_informative = abs(plural.head_change) >= cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma
        valid = head_informative and positive >= required
        frames_out[frame.frame_id] = {"template_id": template, "valid": valid, "head_informative": head_informative, "plural_head_change": plural.head_change, "cue_effect_positive": positive, "cue_effect_required": required,
                                      "template_defined": True, "reconstruction_error": state_f.state.ref.reconstruction_error, "p_c": frame.p_c, "p_t": frame.p_t, "cue_final": frame.p_t == frame.p_c}
        locked = hp.locked_state(state_f)
        states[frame.frame_id] = locked
        own = frame_selectors_from_state(chain, weights, read_out, locked, template, tokens_by_template[template], lock["selectors"]["drive_quantiles"], lock["selectors"]["denominators"][template], forbidden)
        frame_selectors[frame.frame_id] = own
        overlaps[frame.frame_id] = selector_overlaps(lists, own, template, frame.p_c)
        masks = frame_rung_masks(lists, frame.frame_id, template, frame.p_c, frame.p_t, frame_lists_override=own)
        for token in confirmation.tokens:
            entry, _, _ = chain.predict_from_state(weights, state_f, token["token_id"], template, masks)
            rows_out.append(prediction_row(token["word"], frame.frame_id, template, entry))
        say(f"  {frame.frame_id}: {'valid' if valid else 'INVALID'} (head informative {head_informative}, cue effect {positive}/{required}); p_c {frame.p_c}, p_t {frame.p_t}; E_64 shares {overlaps[frame.frame_id][str(frame.p_c)]['E&Sp']} with S'_64, {overlaps[frame.frame_id][str(frame.p_c)]['E&G']} with G_64; {len(confirmation.tokens)} predictions")
    return {"frames": frames_out, "states": states, "state_digests": {frame_id: ap.state_digest(entry) for frame_id, entry in states.items()}, "frame_selectors": frame_selectors, "overlaps": overlaps, "rows": rows_out, "identities": identities,
            "token_means": token_means_from_table(rows_out, [token["word"] for token in confirmation.tokens]),
            "digest": stage_digest(rows_out, states, frame_selectors), "commit": protocol_code_commit, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}


def _rows_by_pair(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {f"{row['token']}|{row['frame_id']}": row for row in rows}


def stage_two(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation019, lock: Mapping[str, Any], lock_012: Mapping[str, Any], stage1: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    assert_stage_one_digest(stage1)
    base2_pt = torch.tensor(lock["base2_pt"]["vector"], dtype=torch.float64) if lock["base2_pt"]["vector"] is not None else None
    parts = _components(model, pool, pool_010, lock, lock_012, hp.bases_from_json(lock["bases_3"]), base2_pt)
    weights, head, axis_T, e_axis, chain, context, read_out = parts["weights"], parts["head"], parts["axis_T"], parts["e_axis"], parts["chain"], parts["context_019"], parts["read_out"]
    nouns = pool.single_nouns
    lists = lock["selectors"]["lists"]
    denominators = lock["selectors"]["denominators"]
    identities: dict[str, float] = {}
    states: dict[str, hp.FrameState017] = {}
    say("stage 2: fresh cues in the exposed frames (Y1) and in the valid fresh frames (Y2); the oracles only after every measurement")
    pairs: dict[str, dict[str, dict[str, Any]]] = {"Y1": {}, "Y2": {}}
    effects: dict[str, dict[str, dict[int, torch.Tensor]]] = {"Y1": {}, "Y2": {}}
    contributions: dict[str, dict[str, torch.Tensor]] = {"Y1": {}, "Y2": {}}
    for frame in pool.frames:
        if frame.template_id not in lock["defined_templates"]:
            continue
        state_f = hp.capture_frame_017(model, head, pool.reference_prompt(frame), nouns, axis_T)
        hp._check_locked_state_017(state_f, lock["locked_states"][frame.frame_id], frame.frame_id)
        rows16 = atp.reference_rows(chain.hcm.fcm.programs, state_f.state.x1_all, state_f.state.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows16, state_f.state))
        identities["I5_reference_head_row"] = max(identities.get("I5_reference_head_row", 0.0), hp.check_reference_row(state_f, chain.hcm.program3))
        states[frame.frame_id] = state_f
        masks = frame_rung_masks(lists, frame.frame_id, frame.template_id, state_f.p_c, state_f.p_t)
        plural_name = pool.plural_cue[frame.template_id]
        plural = measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            out = analyse_pair_019(record, plural, context=context, state=state_f, rung_masks=masks)
            if out is None:
                raise pm.IncidentError(f"{frame.frame_id}/{token['word']}: no analysis although the exposed frame is informative")
            key = f"{token['word']}|{frame.frame_id}"
            er._max_errors(identities, out["analysis"]["identities"])
            pairs["Y1"][key], effects["Y1"][key], contributions["Y1"][key] = out["analysis"], out["effects"], out["u"]
        say(f"  {frame.frame_id} (exposed): {len(confirmation.tokens)} tokens")
    for frame in confirmation.frames:
        entry = stage1["frames"][frame.frame_id]
        if not entry.get("template_defined") or not entry["valid"]:
            continue
        state_f = hp.capture_frame_017(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        if ap.state_digest(hp.locked_state(state_f)) != stage1["state_digests"][frame.frame_id]:
            raise pm.IncidentError(f"{frame.frame_id}: the re-captured reference state differs from stage 1's digested state")
        states[frame.frame_id] = state_f
        masks = frame_rung_masks(lists, frame.frame_id, frame.template_id, state_f.p_c, state_f.p_t, frame_lists_override=stage1["frame_selectors"][frame.frame_id])
        sg, pl = pm.frame_prompts(frame)
        plural = measure_pair(model, weights, head, state_f, pool.plural_cue[frame.template_id], pl.cue_token_id, e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            out = analyse_pair_019(record, plural, context=context, state=state_f, rung_masks=masks)
            if out is None:
                raise pm.IncidentError(f"{frame.frame_id}/{token['word']}: no analysis although the frame is valid")
            key = f"{token['word']}|{frame.frame_id}"
            er._max_errors(identities, out["analysis"]["identities"])
            pairs["Y2"][key], effects["Y2"][key], contributions["Y2"][key] = out["analysis"], out["effects"], out["u"]
        say(f"  {frame.frame_id} (fresh): {len(confirmation.tokens)} tokens")
    for label, name, table in (("locked", "Y1", _rows_by_pair(lock["predictions"]["rows"])), ("stage 1", "Y2", _rows_by_pair(stage1["rows"]))):
        for key, analysis in pairs[name].items():
            if key not in table:
                raise pm.IncidentError(f"{key}: no {label} table row for a measured pair")
            worst = max(atp._max_numeric_difference(analysis["prediction"][column], table[key][column], f"{key}/{column}") for column in PREDICTION_COLUMNS[3:])
            if worst > LOCK_PREDICTION_TOLERANCE:
                raise pm.IncidentError(f"{key}: the prediction recomputed at stage 2 differs from the {label} table row by {worst:.2e}")
    # Only now, with every measurement of both sets in hand: the ranking oracle per frame and the leave-one-cue-out witness per pair, then their rungs.
    words = [token["word"] for token in confirmation.tokens]
    oracle_lists: dict[str, Any] = {}
    witness_lists: dict[str, Any] = {}
    for name in ("Y1", "Y2"):
        _, scored, _ = scored_population(pairs[name], words)
        if not pairs[name]:
            oracle_lists[name], witness_lists[name] = {}, {}
            continue
        oracle_lists[name], witness_lists[name] = oracles_for_set(pairs[name], effects[name], contributions[name], scored, read_out.abs(), denominators)
        for key, analysis in pairs[name].items():
            if key not in witness_lists[name]:
                continue  # an unscored token: no oracle rung (its pair enters no statistic)
            frame_id = analysis["frame_id"]
            analysis["oracle"] = oracle_rungs_for_frame(chain, weights, states[frame_id], analysis["token_id"], analysis["template"], oracle_lists[name][frame_id], witness_lists[name][key])
            analysis["witness_lists"] = witness_lists[name][key]
        say(f"  {name}: oracle lists for {len(oracle_lists[name])} frames; witness lists for {len(witness_lists[name])} pairs")
    results = score_confirmation(stage1, pairs["Y1"], pairs["Y2"], confirmation, lock, oracle_lists)
    results["oracle_lists"] = oracle_lists
    results["per_frame_exposed"] = {key: compact_analysis(value) for key, value in pairs["Y1"].items()}
    results["per_frame_fresh"] = {key: compact_analysis(value) for key, value in pairs["Y2"].items()}
    er._max_errors(identities, stage1.get("identities", {}))
    results["identities"] = identities
    return results


# ---------------------------------------------------------------------------
# Report.


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 019 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state['confirmation_019_sha256']}`", f"- Experiment 018 lock sha256: `{state['lock_018_sha256']}`", f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", "", "## Phases", ""]
    for phase, entry in state["phases"].items():
        lines.append(f"- `{phase}`: `{entry.get('status')}`")
    lines.append("")
    lines.append(f"> {WITNESS_TERMINOLOGY}")
    lines.append("")
    exploration = state.get("exploration") or {}
    k = str(DECISION_SIZE)

    def kappa_line(stats: Mapping[str, Any], rungs: Sequence[str]) -> str:
        kap = stats["pairs"]["kappa"]
        return ", ".join(f"{rung} {f(kap['c_L'].get(rung), 3)}" for rung in rungs)

    def ladder_lines(stats: Mapping[str, Any]) -> list[str]:
        out = []
        kap = stats["pairs"]["kappa"]
        for size in SIZES:
            names = [f"Sp{size}", f"T{size}", f"E{size}", f"G{size}", f"L{size}", f"O{size}", f"Os{size}", *[f"R{i}_{size}" for i in range(1, N_RANDOM_CONTROLS + 1)]]
            g = stats["gains"][str(size)]
            out.append(f"  - k={size}: κ_c_L " + ", ".join(f"{n} {f(kap['c_L'].get(n), 3)}" for n in names) + f"; gains over S'_{size}: " + ", ".join(f"{n} {f(g.get(n), 3)}" for n in names[1:]) +
                       f"; ρ {f(stats['rho'][str(size)], 3)}; E wins {stats['wins'][str(size)]['wins']}/{stats['wins'][str(size)]['n_frames']}; κ_Π E {f(kap['Pi'].get(f'E{size}'), 3)} / S' {f(kap['Pi'].get(f'Sp{size}'), 3)}; κ_row E {f(kap['rows'].get(f'E{size}'), 3)}")
        return out

    if exploration.get("summary"):
        s = exploration["summary"]
        sel = exploration["selectors"]
        lines += ["## Tier A — exposed pool (calibration record only; the floors are frozen constants)", "",
                  f"- Replication of Experiment 018: {exploration['replication']['experiment_018']['n']} pairs, max deviation {exploration['replication']['experiment_018']['max_abs_deviation']:.2e}; I9 (the E rule with Experiment 018's inputs reproduces its per-frame lists): {exploration['i9_replication']}",
                  "- Identities (checks only): " + ", ".join(f"{name} {value:.1e}" for name, value in sorted(exploration["identities"].items())),
                  f"- Licensed pool: {exploration['licensed_pool']['n_pairs']} pairs; selectors digest `{sel['digest']}`; S'_1 {sel['lists']['Sp1']}; S'_64 at p_c (first 10) {sel['lists']['Sp']['p_c']['64'][:10]}; inherited S_64 (first 10) {sel['lists']['L']['64'][:10]}",
                  f"- Exposed (cue-in-sample; {exploration['exposed_check']['n_pairs']} pairs): reference rung c_L R² {f(s['reference_c_L_r2'], 3)}, rows {f(s['reference_rows_r2'], 3)}; gaps " + ", ".join(f"{kk} {f(v, 3)}" for kk, v in s["gaps"].items())]
        lines += ladder_lines(exploration["exposed_check"]["statistics"])
        m = exploration["exposed_check"]["membership"]["mean"]
        lines.append(f"  - membership at k=64 (mean overlap with the ranking oracle's top-64 at p_c): E {f(m.get('E'), 3)}, G {f(m.get('G'), 3)}, T {f(m.get('T'), 3)}, S' {f(m.get('Sp'), 3)}; witness headroom H*_64 {f(s['headroom']['witness'], 3)} (ranking proxy {f(s['headroom']['ranking_proxy'], 3)})")
        splits = exploration["exposed_check"]["statistics"]["splits"]
        for name in ("cue_final", "coordinated"):
            sp = splits[name]
            lines.append(f"  - split {name}: {sp['n_pairs']} pairs, gap {f(sp['gap_c_L'], 3)}; gains at k=64 E {f(sp['gains'].get('E64'), 3)}, G {f(sp['gains'].get('G64'), 3)}, T {f(sp['gains'].get('T64'), 3)}, O* {f(sp['gains'].get('Os64'), 3)}")
        for template, sp in splits["per_template"].items():
            lines.append(f"  - template {template}: {sp['n_pairs']} pairs, gap {f(sp['gap_c_L'], 3)}; gains at k=64 E {f(sp['gains'].get('E64'), 3)}, G {f(sp['gains'].get('G64'), 3)}, T {f(sp['gains'].get('T64'), 3)}")
        lines.append("")
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}`; predictions sha256 `{state['lock']['predictions_sha256']}`", ""]
    confirmation = state.get("confirmation") or {}
    if confirmation.get("stage1"):
        s1 = confirmation["stage1"]
        lines += ["## Confirmation — stage 1 (fresh frames' reference states; the frames' E and G lists and the prediction table digested before any fresh cue prompt)", "", f"- Table rows {len(s1['rows'])}; digest `{s1['digest']}`; commit `{s1['commit']}`"]
        for frame_id, entry in s1["frames"].items():
            if entry.get("template_defined"):
                ov = s1["overlaps"][frame_id][str(entry["p_c"])]
                lines.append(f"  - {frame_id}: {'valid' if entry['valid'] else 'INVALID'} (plural head change {f(entry['plural_head_change'], 3)}, cue effect {entry['cue_effect_positive']}/{entry['cue_effect_required']}; p_c {entry['p_c']}, p_t {entry['p_t']}); E_64 ∩ S'_64 {ov['E&Sp']}, E_64 ∩ T_64 {ov['E&T']}, E_64 ∩ G_64 {ov['E&G']}, G_64 ∩ S'_64 {ov['G&Sp']}")
            else:
                lines.append(f"  - {frame_id}: {entry.get('note')}")
        lines.append("")
    if confirmation.get("outcome"):
        lines += [f"## Confirmation — stage 2 — `{confirmation['outcome']['label']}`", ""]
        for name, y in (("Y1", confirmation["Y1"]), ("Y2", confirmation["Y2"])):
            t = y["test"]
            lines.append(f"- {name}: scored tokens {len(y.get('scored_tokens', []))} ({y['n_pairs']} pairs, {y.get('n_frames')} frames); precondition {'ok' if y['precondition'].get('ok') else 'FAILED ' + str(y['precondition'].get('failed'))} → **{t['label']}**")
            if "conditions" in t:
                lines.append(f"  - Δκ_64(E) {f(t['gain'], 3)} (floor {ROUTING_GAIN_FLOOR}); E_64 wins {t['wins']['wins']}/{t['wins']['n_frames']} (ties {t['wins']['ties']}); split gains cue-final {f(t['split_gains']['cue_final'], 3)}, coordinated {f(t['split_gains']['coordinated'], 3)} (floor {SPLIT_GAIN_FLOOR}); "
                             f"witness headroom H*_64 {f(t['headroom_witness'], 3)} (ranking proxy H_64 {f(t['headroom_ranking_proxy'], 3)}; E's share of H* {f(t.get('witness_share'), 2)}); conditions {t['conditions']}")
            if y.get("statistics"):
                st = y["statistics"]
                pre = y["precondition"]
                lines.append(f"  - reference rung c_L R² {f(pre['reference']['c_L'], 3)}, rows {f(pre['reference']['rows'], 3)}, ΔT {f(pre['reference']['dT'], 3)}; gaps pooled {f(pre['gaps']['pooled'], 3)}, cue-final {f(pre['gaps']['cue_final'], 3)}, coordinated {f(pre['gaps']['coordinated'], 3)}")
                lines += ladder_lines(st)
                lines.append(f"  - jackknife of Δκ_64(E): min {f(st['jackknife']['E']['min'], 3)}, max {f(st['jackknife']['E']['max'], 3)}; frames where E_64 loses more than 0.05 R² to S'_64: {len(st['frames_losing_over_0_05']['E'])} (G: {len(st['frames_losing_over_0_05']['G'])})")
                for template, sp in st["splits"]["per_template"].items():
                    lines.append(f"  - template {template}: {sp['n_pairs']} pairs, gap {f(sp['gap_c_L'], 3)}; κ S' {f(sp['kappa'].get('Sp64'), 3)}, T {f(sp['kappa'].get('T64'), 3)}, E {f(sp['kappa'].get('E64'), 3)}, G {f(sp['kappa'].get('G64'), 3)}, O* {f(sp['kappa'].get('Os64'), 3)}")
                lines.append("  - per frame at k=64 (c_L R²: S_0 | S' | T | E | G | O | O* | S_2048; gap):")
                for frame_id, e in st["per_frame"].items():
                    r = e["r2"]
                    lines.append(f"    - {frame_id} ({e['template']}, {e['n_pairs']} pairs): {f(r.get('S0'), 3)} | {f(r.get('Sp64'), 3)} | {f(r.get('T64'), 3)} | {f(r.get('E64'), 3)} | {f(r.get('G64'), 3)} | {f(r.get('O64'), 3)} | {f(r.get('Os64'), 3)} | {f(r.get('S2048'), 3)}; gap {f(e['gap'], 3)}")
            if y.get("membership") and y["membership"].get("n_frames"):
                m = y["membership"]["mean"]
                lines.append(f"  - membership at k=64 over {y['membership']['n_frames']} frames: E {f(m['E'], 3)}, G {f(m['G'], 3)}, T {f(m['T'], 3)}, S' {f(m['Sp'], 3)}")
        y3, y4, y5 = confirmation["Y3"], confirmation["Y4"], confirmation["Y5"]
        lines.append(f"- Y3 (the operating-point-plus-drive rule against the full evaluation; ρ_64 per set and pooled): **{y3['label']}**; ρ {({n: f(v, 3) for n, v in y3.get('rho', {}).items()})}; denominators Δκ_64(E) {({n: f(v, 3) for n, v in y3.get('denominators', {}).items()})}" + (f"; not evaluable on {y3['not_evaluable']}" if y3.get("not_evaluable") else ""))
        lines.append(f"- Y4 (the template-family account; A = κ(E_64) − κ(T_64), B = Δκ_64(T), on Y2 and pooled): **{y4['label']}**" + (f" — {y4['reason']}" if y4.get("reason") else "") + f"; values {({n: {kk: f(vv, 3) for kk, vv in v.items()} for n, v in y4.get('values', {}).items()})}")
        lines.append(f"- Y5 (membership per set: mean |E_64 ∩ O_64| / 64 ≥ {MEMBERSHIP_FLOOR} and ≥ S' + {MEMBERSHIP_MARGIN}): **{y5['label']}**; " + "; ".join(f"{n}: E {f(v.get('E'), 3)}, S' {f(v.get('Sp'), 3)}, G {f(v.get('G'), 3)}, T {f(v.get('T'), 3)}" if v.get("evaluable") else f"{n}: not evaluable" for n, v in y5["per_set"].items()))
        if confirmation.get("pooled"):
            pooled = confirmation["pooled"]
            lines.append(f"- Pooled (both sets, {pooled['n_pairs']} pairs): κ_c_L at k=64 S' {f(pooled['kappa_c_L'].get('Sp64'), 3)}, T {f(pooled['kappa_c_L'].get('T64'), 3)}, E {f(pooled['kappa_c_L'].get('E64'), 3)}, G {f(pooled['kappa_c_L'].get('G64'), 3)}, O* {f(pooled['kappa_c_L'].get('Os64'), 3)}; ρ {f(pooled['rho'][k], 3)}; headroom H*_64 {f(pooled['headroom']['witness'], 3)}")
        d = confirmation.get("descriptive", {})
        for name, entry in d.items():
            lines.append(f"- Descriptive expectations on {name}: (i) other sizes {({kk: (f(v['gain_E'], 3), v['holds']) for kk, v in entry['i_other_sizes'].items()})}; (ii) gain over the inherited S_64 {f(entry['ii_inherited']['gain_over_inherited'], 3)} vs over S' {f(entry['ii_inherited']['gain_over_global'], 3)} holds {entry['ii_inherited']['holds']}; "
                         f"(iii) random at k=64 best {f(entry['iii_random'][k]['best_random'], 3)} below {RANDOM_KAPPA_MAX} {entry['iii_random'][k]['below_max']}, margin holds {entry['iii_random'][k]['margin_holds']}; (iv) κ_c_L ≥ κ_Π {entry['iv_orderings'][k]['c_L_over_Pi']}, row diffuse {entry['iv_orderings'][k]['row_diffuse']}; "
                         f"(v) G's gain per template {({t: f(v, 3) for t, v in entry['v_G_cardinal'].items()})}; (vi) S'_1 {f(entry['vi_single']['Sp1'], 3)}, E_1 {f(entry['vi_single']['E1'], 3)}; (x) witness κ per k {({kk: f(v['kappa_witness'], 3) for kk, v in entry['x_witness'].items()})}, E's share of H*_64 {f(entry.get('witness_share'), 2)}")
        lines.append("")
        header = "| token | class | Y1 frames | Y1 ĉ_L E_64 | Y1 ĉ_L S'_64 | Y1 ĉ_L S_2048 | Y1 c_L | Y2 frames | Y2 ĉ_L E_64 | Y2 ĉ_L S'_64 | Y2 ĉ_L S_2048 | Y2 c_L |"
        lines += [header, "|" + "---|" * 12]
        categories = {token["word"]: token["category"] for token in confirmation.get("tokens_meta", [])}
        for word in sorted(confirmation["Y1"]["tokens"], key=lambda w: -(confirmation["Y1"]["tokens"][w].get("dT_mean") or 0.0)):
            a, b = confirmation["Y1"]["tokens"][word], confirmation["Y2"]["tokens"].get(word, {})
            lines.append(f"| {word} | {categories.get(word, '')} | {a.get('n_frames', 0)} | {f(a.get('c_L_E64_mean'), 4)} | {f(a.get('c_L_Sp64_mean'), 4)} | {f(a.get('c_L_S2048_mean'), 4)} | {f(a.get('c_L_mean'), 4)} | {b.get('n_frames', 0)} | {f(b.get('c_L_E64_mean'), 4)} | {f(b.get('c_L_Sp64_mean'), 4)} | {f(b.get('c_L_S2048_mean'), 4)} | {f(b.get('c_L_mean'), 4)} |")
        lines.append("")
    if confirmation.get("incident"):
        lines += ["## Incident", "", f"- {confirmation['incident']}", ""]
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
