"""Experiment 013: does frozen-pattern value transport by the layer-1–2 heads predict the residual of the token-local model?

Experiment 012's frozen token-local MLP model is taken as given. The candidate for the residual it left is the part
of the computation it omitted: the frame's own base state and the sixteen heads of layers 1–2 at the cue position,
modelled with their attention patterns held at the reference. Under the E-patch only the cue position's residual
changes at the input of block 1, so each head's output change splits exactly into a value path with the reference
pattern, ``A_h^ref(p_c, p_c) · Δv_h(p_c) W_O^h``, and a pattern-change path, ``Σ_k ΔA_h(p_c, k) v_h^patch(k) W_O^h``;
the first is computable from the reference prompt alone and is propagated into block 2's LayerNorm and MLP and onto
the read direction, the second is left out and measured as the remainder. Predictions for new cues in the 36 exposed
frames are committed before any fresh prompt (Y1, strict); predictions for new cues in six new frames are computed
from each frame's reference run and digested before any fresh cue enters the network (Y2, frame-conditional); the
MLP/heads attribution is scored at the pair level (Y3). Tolerances are frozen constants from design revision 2.
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
from . import encoding_read as er
from . import head_transport as ht
from . import layer_correction as lc
from . import plural_mechanism as pm
from . import read_assembly as ra
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 2).

EXPERIMENT_DIR = "experiments/013-attention-paths-frozen-pattern"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
INHERITED_012_LEDGER_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-012-pair-ledger.json"
EXPERIMENT_011_LOCK_PATH = er.LOCK_RELATIVE_PATH
EXPERIMENT_012_LOCK_PATH = lc.LOCK_RELATIVE_PATH
EXPERIMENT_012_CONFIRMATION_PATH = lc.CONFIRMATION_RELATIVE_PATH
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260924
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
INHERITED_LEDGER_SCHEMA_VERSION = 1
PHASES = ("explore", "lock", "confirm", "report")

Y_SPEARMAN = 0.80  # Y1 and Y2
Y_R2 = 0.50  # Y1 and Y2
TAU_R = 0.070  # Y1/Y2 MAE ceiling on token means; frozen as max(0.05, 3 × exposed RMSE 0.0233)
TAU_A = 0.106  # Y3 MAE ceiling on pairs (ĉ_H vs c_H); frozen as max(0.05, 3 × exposed RMSE 0.0355)
TAU_MIN = 0.05
TAU_MULTIPLIER = 3.0
DESIGN_RMSE_R = 0.0233
DESIGN_RMSE_A = 0.0355
CALIBRATION_TOLERANCE = 0.001  # recomputed exposed RMSE against the design value (record; incident beyond)
Y3_SPEARMAN_H = 0.70
Y3_SPEARMAN_M = 0.90
C_H_DEGENERATE_SD = 0.017  # a quarter of the exposed pair-level spread 0.068: below it the c_H correlation is non-evaluable
COMPENSATION_MIN = 0.05
MIN_VALID_FRAMES = 4
MIN_VALID_FRAMES_PER_TOKEN = 3
MIN_SCORED_TOKENS = 16
FRAME_CUE_EFFECT_RATE = 108 / 120
LOCK_PREDICTION_TOLERANCE = 1e-9
REPLICATION_TOLERANCE = 1e-6
MODEL_012_TOLERANCE = 1e-9  # ĉ_012 recomputed against the Experiment 012 record
EXPECTED_LEDGER_SIZE = 2724  # Experiment 012's 87 × 30 exposed pairs less the 30 own-reference pairs, plus its 24 × 6 confirmed pairs
OV_IDENTITY_TOLERANCE = 1e-4  # relative, per head: frozen-pattern term + pattern-change term against the captured head output change
LOCKED_STATE_TOLERANCE = 1e-9  # a re-captured exposed reference state against the locked one
HEAD_LAYERS = (1, 2)
N_HEADS = 8
HEAD_KEYS = lc.HEAD_KEYS  # L01.H00 … L02.H07 in ra.COMPONENT_ORDER
PREDICTION_COLUMNS = ("token", "frame_id", "template", "c_012", "c_own", "base_point", "frozen_attention", "r_hat", "c_M_hat", "c_H_hat", "c_L_hat", "c_mlp1_hat", "c_mlp2_hat", "g_E", "head_terms")

CANDIDATES: dict[str, tuple[str, ...]] = {
    "determiner-like": ("same", "own", "last", "next", "first", "second"),
    "numeral": ("zero", "thousand", "million", "billion", "trillion", "dozens", "hundreds", "thousands", "millions"),
    "quantity": ("limited", "surplus", "endless", "plenty", "vast", "minimal", "infinite", "excess", "lesser", "considerable", "insufficient"),
    "possessive-or-pronoun": ("whom", "someone", "nobody", "everyone", "anybody", "somebody", "everybody", "anyone"),
    "adjective": ("tall", "thick", "quiet", "broken", "wide", "narrow", "sharp", "smooth", "rough", "golden"),
}
QUOTAS = {"determiner-like": 5, "numeral": 5, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 5}
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The pantry stocks {cue}"),
    ("cardinal", "The courier delivers {cue}"),
    ("quantifier", "The audit examines {cue}"),
    ("quantifier", "The podcast discusses {cue}"),
    ("coordinated-adjective", "Mateo and Ines washed {cue} clean"),
    ("coordinated-adjective", "Sofia and Anders bought {cue} loose"),
)
FRAME_ID_TAG = "013"
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", "experiments/012-layer-correction-token-local/", "experiments/011-encoding-read-prospective/", "experiments/010-read-direction-assembly/",
                            "experiments/009-head-transport-rule/", "experiments/008-cue-suppression-localization/", "experiments/007-supervised-cue-subspace/", "experiments/006-low-rank-cue-decompilation/",
                            "experiments/005-regular-plural-mechanism/", "screening/behavior-candidates/manifest-v1.json")
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)
OUTCOME_Y1 = ("RESIDUAL_PREDICTED_TOKENS", "RESIDUAL_NOT_PREDICTED_TOKENS", "PRECONDITION_FAILED_TOKENS")
OUTCOME_Y2 = ("RESIDUAL_PREDICTED_FRAMES_CONDITIONAL", "RESIDUAL_NOT_PREDICTED_FRAMES_CONDITIONAL", "PRECONDITION_FAILED_FRAMES")
OUTCOME_Y3 = ("ATTRIBUTION_PREDICTED", "ATTRIBUTION_NOT_PREDICTED", "ATTRIBUTION_NOT_EVALUABLE")


class PhaseError(pm.PhaseError):
    pass


# ---------------------------------------------------------------------------
# The sixteen heads, the frame's reference state, the frozen-pattern model.


def head_key(layer: int, head: int) -> str:
    return f"L0{layer}.H0{head}"


def layer_of(key: str) -> int:
    return int(key[2])


@dataclass(frozen=True)
class HeadSet:
    heads: Mapping[str, ht.HeadWeights]

    @classmethod
    def from_model(cls, model: Any) -> "HeadSet":
        return cls({head_key(layer, head): ht.HeadWeights.from_model(model, layer, head) for layer in HEAD_LAYERS for head in range(N_HEADS)})

    def keys_of(self, layer: int) -> list[str]:
        return [key for key in HEAD_KEYS if layer_of(key) == layer]


@dataclass
class FrameState013:
    """A frame's reference run: Experiment 012's state plus the residuals at every position ≤ p_c and the layer-1–2 pattern rows at p_c."""

    ref: ht.FrameReference
    components: dict[str, torch.Tensor]
    functional: ra.ReadFunctional
    x1: torch.Tensor
    x2: torch.Tensor
    x1_all: list[torch.Tensor]  # RESID_PRE.L1 at positions 0..p_c
    x2_all: list[torch.Tensor]
    A1: torch.Tensor  # [n_heads, keys] pattern rows at query p_c, layer 1
    A2: torch.Tensor

    @property
    def p_c(self) -> int:
        return self.ref.frame.p_c

    def pattern_row(self, key: str) -> torch.Tensor:
        row = self.A1 if layer_of(key) == 1 else self.A2
        return row[int(key[-1])]

    def pattern_weight(self, key: str) -> float:
        return float(self.pattern_row(key)[self.p_c])

    def locked_state(self) -> dict[str, Any]:
        """What the lock stores per exposed frame: the cue position's residuals and the sixteen self-attention weights; enough for every prediction."""
        return {"x1": self.x1.tolist(), "x2": self.x2.tolist(), "A_pc": {key: self.pattern_weight(key) for key in HEAD_KEYS}, "p_c": self.p_c}


def state_digest(locked: Mapping[str, Any]) -> str:
    return pm.sha256_text(pm.canonical_json(locked))


def reference_sites(frame: pm.Frame) -> list[pm.Site]:
    return [(f"RESID_PRE.L{layer}", k) for layer in HEAD_LAYERS for k in range(frame.p_c + 1)] + [(f"ATTN_PATTERN.L{layer}", frame.p_c) for layer in HEAD_LAYERS]


def patched_sites(frame: pm.Frame) -> list[pm.Site]:
    return lc.extra_sites(frame) + [(f"ATTN_PATTERN.L{layer}", frame.p_c) for layer in HEAD_LAYERS] + [(key, frame.p_c) for key in HEAD_KEYS]


def capture_frame_013(model: Any, head: ht.HeadWeights, reference: pm.Prompt, nouns: Sequence[pm.Noun], axis_T: pm.SiteAxis) -> FrameState013:
    frame = reference.frame
    ref, components = ra.capture_reference(model, head, reference, nouns, extra_sites=reference_sites(frame))
    x1_all = [components.pop(pm.site_label(("RESID_PRE.L1", k))) for k in range(frame.p_c + 1)]
    x2_all = [components.pop(pm.site_label(("RESID_PRE.L2", k))) for k in range(frame.p_c + 1)]
    A1 = components.pop(pm.site_label(("ATTN_PATTERN.L1", frame.p_c)))
    A2 = components.pop(pm.site_label(("ATTN_PATTERN.L2", frame.p_c)))
    if not torch.equal(x1_all[frame.p_c], ref.vectors["R0"].double()):
        raise pm.IncidentError(f"{frame.frame_id}: the captured residual before block 1 disagrees with the R0 stage vector of the same run")
    if A1.shape[0] != N_HEADS or A1.shape[1] <= frame.p_c or A2.shape[0] != N_HEADS:
        raise pm.IncidentError(f"{frame.frame_id}: unexpected attention pattern shape {tuple(A1.shape)} / {tuple(A2.shape)}")
    return FrameState013(ref, components, ra.read_functional(head, ref, axis_T), x1_all[frame.p_c], x2_all[frame.p_c], x1_all, x2_all, A1, A2)


def measure_pair_013(model: Any, weights: pm.Weights, head: ht.HeadWeights, state: FrameState013, name: str, token_id: int, e_axis: pm.SiteAxis, axis_T: pm.SiteAxis, nouns: Sequence[pm.Noun]) -> ra.Attribution:
    return ra.measure_token_010(model, weights, head, state.ref, state.components, state.functional, name, token_id, e_axis, axis_T, nouns, extra_sites=patched_sites(state.ref.frame))


@dataclass(frozen=True)
class FrozenPatternModel:
    """The frozen Experiment 012 model (Level 0 at its locked bases, Level 1 at the frame's own base) and Level 2: the sixteen heads' value paths with reference patterns."""

    read: lc.CorrectionRead
    lw: lc.LayerWeights
    heads: HeadSet
    bases_012: Mapping[str, tuple[torch.Tensor, torch.Tensor]]

    def value_path(self, key: str, x: torch.Tensor, delta: torch.Tensor, weight: float) -> torch.Tensor:
        hw = self.heads.heads[key]
        return weight * ((hw.value(x.double() + delta.double()) - hw.value(x.double())) @ hw.W_O)

    def predict(self, weights: pm.Weights, x1: torch.Tensor, x2: torch.Tensor, pattern_weights: Mapping[str, float], token_id: int, template: str) -> dict[str, Any]:
        read, lw = self.read, self.lw
        delta_e = read.encoding_delta(weights, token_id, template)
        denominator = read.denominator(weights, template)
        d1 = lw.delta_out(1, x1, delta_e)
        heads_1 = {key: self.value_path(key, x1, delta_e, pattern_weights[key]) for key in self.heads.keys_of(1)}
        dx2 = delta_e + d1 + sum(heads_1.values())
        d2 = lw.delta_out(2, x2, dx2)
        heads_2 = {key: self.value_path(key, x2, dx2, pattern_weights[key]) for key in self.heads.keys_of(2)}
        head_terms = {key: read.inner(vector) / denominator for key, vector in (heads_1 | heads_2).items()}
        c_mlp1, c_mlp2 = read.inner(d1) / denominator, read.inner(d2) / denominator
        c_M = c_mlp1 + c_mlp2
        c_H = sum(head_terms.values())
        c_L = c_M + c_H
        c_own = read.predict(weights, lw, (x1, x2), token_id, template)["c_hat"]
        c_012 = read.predict(weights, lw, self.bases_012[template], token_id, template)["c_hat"]
        return {"c_012": c_012, "c_own": c_own, "base_point": c_own - c_012, "frozen_attention": c_L - c_own, "r_hat": c_L - c_012, "c_M_hat": c_M, "c_H_hat": c_H, "c_L_hat": c_L,
                "c_mlp1_hat": c_mlp1, "c_mlp2_hat": c_mlp2, "g_E": read.inner(delta_e) / denominator, "head_terms": head_terms}

    def predict_from_state(self, weights: pm.Weights, state: FrameState013, token_id: int, template: str) -> dict[str, Any]:
        return self.predict(weights, state.x1, state.x2, {key: state.pattern_weight(key) for key in HEAD_KEYS}, token_id, template)

    def predict_from_locked(self, weights: pm.Weights, locked: Mapping[str, Any], token_id: int, template: str) -> dict[str, Any]:
        return self.predict(weights, torch.tensor(locked["x1"], dtype=torch.float64), torch.tensor(locked["x2"], dtype=torch.float64), dict(locked["A_pc"]), token_id, template)


def model_from_locks(lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lw: lc.LayerWeights, heads: HeadSet, reference_ids: Mapping[str, int], plural_ids: Mapping[str, int]) -> FrozenPatternModel:
    read = lc.read_from_lock_011(lock_011, reference_ids, plural_ids)
    return FrozenPatternModel(read, lw, heads, lc.bases_from_json(lock_012["base_states"]))


def is_compensation_case(c_M_hat: float, c_H_hat: float) -> bool:
    return c_M_hat * c_H_hat < 0 and abs(c_M_hat) > COMPENSATION_MIN and abs(c_H_hat) > COMPENSATION_MIN


def prediction_row(word: str, frame_id: str, template: str, prediction: Mapping[str, Any]) -> dict[str, Any]:
    row = {"token": word, "frame_id": frame_id, "template": template}
    row.update({key: prediction[key] for key in PREDICTION_COLUMNS if key not in ("token", "frame_id", "template")})
    row["compensation_case"] = is_compensation_case(prediction["c_M_hat"], prediction["c_H_hat"])
    return row


# ---------------------------------------------------------------------------
# The exact per-head split of a measured pair and the pair analysis.


def head_identity(heads: HeadSet, key: str, state: FrameState013, record: ra.Attribution) -> dict[str, Any]:
    """Δout_h = A_h^ref(p_c,p_c)·Δv_h(p_c) W_O + Σ_{k ≤ p_c} ΔA_h(p_c,k) v_h^patch(k) W_O, both terms from the captured runs; an incident above 1e-4 relative."""
    hw = heads.heads[key]
    layer = layer_of(key)
    p_c = state.p_c
    x_ref_all = state.x1_all if layer == 1 else state.x2_all
    x_patch = record.extra[pm.site_label((f"RESID_PRE.L{layer}", p_c))]
    row_ref = state.pattern_row(key)
    row_patch = record.extra[pm.site_label((f"ATTN_PATTERN.L{layer}", p_c))][int(key[-1])]
    measured = record.extra[pm.site_label((key, p_c))] - state.components[key]
    frozen = float(row_ref[p_c]) * ((hw.value(x_patch) - hw.value(x_ref_all[p_c])) @ hw.W_O)
    pattern_change = torch.zeros_like(frozen)
    for k in range(p_c + 1):
        value = hw.value(x_patch if k == p_c else x_ref_all[k])
        pattern_change = pattern_change + float(row_patch[k] - row_ref[k]) * (value @ hw.W_O)
    error = lc.relative_vector_error(frozen + pattern_change, measured, state.components[key])
    if error > OV_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{state.ref.frame.frame_id}/{record.token}/{key}: the frozen-pattern and pattern-change terms do not sum to the captured head output change (relative error {error:.2e})")
    return {"frozen": frozen, "pattern_change": pattern_change, "measured": measured, "error": error, "delta_A_pc": float(row_patch[p_c] - row_ref[p_c])}


def analyse_pair_013(record: ra.Attribution, plural: ra.Attribution, *, model: FrozenPatternModel, weights: pm.Weights, state: FrameState013, axis_T: pm.SiteAxis) -> dict[str, Any] | None:
    """Experiment 012's measured quantities, the frozen-pattern prediction at the frame's own state, the exact per-head split, and the ladder."""
    base = lc.analyse_pair(record, plural, read=model.read, lw=model.lw, weights=weights, state=state, axis_T=axis_T)
    if base is None:
        return None
    template = state.ref.frame.template_id
    denominator = model.read.denominator(weights, template)
    prediction = model.predict_from_state(weights, state, record.token_id, template)
    identities = {key: head_identity(model.heads, key, state, record) for key in HEAD_KEYS}
    direct_pattern_change = sum(model.read.inner(entry["pattern_change"]) for entry in identities.values()) / denominator
    remainder = base["c_L"] - prediction["c_L_hat"]
    return {"token": record.token, "token_id": record.token_id, "frame_id": state.ref.frame.frame_id, "template": template,
            "c_L": base["c_L"], "c_M": base["c_M"], "c_H": base["c_H"], "c_k": base["c_k"], "P1": base["P1"], "P1_prime": base["P1_prime"], "q_T": base["q_T"], "g_E": base["g_E"],
            "f_layers_010": base["fractions_010"]["f_layers"], "prediction": prediction, "r": base["c_L"] - prediction["c_012"],
            "ladder": {"base_point": prediction["base_point"], "frozen_attention": prediction["frozen_attention"], "remainder": remainder,
                       "remainder_direct_pattern_change": direct_pattern_change, "remainder_indirect": remainder - direct_pattern_change,
                       "attention_input_term": base["ladder"]["attention_input_term"], "mlp1_exact_error": base["ladder"]["mlp1_exact_error"], "mlp2_identity_error": base["ladder"]["mlp2_identity_error"]},
            "head_identity_max_error": max(entry["error"] for entry in identities.values()), "delta_A_pc": {key: entry["delta_A_pc"] for key, entry in identities.items()},
            "identities": base["identities"], "compensation_case": is_compensation_case(prediction["c_M_hat"], prediction["c_H_hat"])}


# ---------------------------------------------------------------------------
# Pool (111 tokens × 36 frames), the Experiment 012 pair ledger, the confirmation set.


def build_pool_013(manifest: ScreeningManifest, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009, confirmation_011: er.Confirmation011, confirmation_012: lc.Confirmation012) -> cs.Pool008:
    base = lc.build_pool_012(manifest, extension, confirmation_006, confirmation_009, confirmation_011)
    frames = base.frames + tuple(confirmation_012.frames)
    origin = dict(base.frame_origin) | {frame.frame_id: "confirmation-012" for frame in confirmation_012.frames}
    tokens = list(base.tokens)
    category, source = dict(base.token_category), dict(base.token_source)
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation_012.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"Experiment 012 token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-012"
    return cs.Pool008(frames, origin, tuple(tokens), category, source, base.nouns, base.noun_source, base.reference_ids, base.plural_cue)


LEDGER_FIELDS = ("c_L", "c_M", "c_H", "c_own", "P1", "q_T", "g_E")


def ledger_entry_012(analysis_012: Mapping[str, Any], c_012: float) -> dict[str, Any]:
    return {"c_L": float(analysis_012["c_L"]), "c_M": float(analysis_012["c_M"]), "c_H": float(analysis_012["c_H"]), "c_k": {key: float(analysis_012["c_k"][key]) for key in ra.COMPONENT_ORDER},
            "c_own": float(analysis_012["own"]["c_hat"]), "c_012": float(c_012), "P1": float(analysis_012["P1"]), "q_T": float(analysis_012["q_T"]), "g_E": float(analysis_012["g_E"])}


def ledger_entry_from_013(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {"c_L": analysis["c_L"], "c_M": analysis["c_M"], "c_H": analysis["c_H"], "c_k": dict(analysis["c_k"]), "c_own": analysis["prediction"]["c_own"], "c_012": analysis["prediction"]["c_012"],
            "P1": analysis["P1"], "q_T": analysis["q_T"], "g_E": analysis["g_E"]}


def inherited_ledger_payload(entries: Mapping[str, Mapping[str, Any]], *, source: Mapping[str, Any], digests: Mapping[str, str]) -> dict[str, Any]:
    payload = {"schema_version": INHERITED_LEDGER_SCHEMA_VERSION, "experiment": "012",
               "description": "Derived extract of Experiment 012's recorded per-pair ledger in its Y1 units (c_L, c_M, c_H, c_k, the own-base ĉ, the locked 012 model's ĉ_012, P1, q_T, g_E) over its 87 × 30 exposed pairs and 24 × 6 confirmed pairs. Experiment 013 recomputes these and requires agreement within 1e-6 (ĉ_012 within 1e-9).",
               "source": dict(source), "manifest_sha256": digests["manifest"], "extension_sha256": digests["extension"], "confirmation_006_sha256": digests["confirmation_006"], "confirmation_009_sha256": digests["confirmation_009"],
               "confirmation_011_sha256": digests["confirmation_011"], "confirmation_012_sha256": digests["confirmation_012"], "lock_012_sha256": digests["lock_012"],
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "entries": {key: dict(value) for key, value in sorted(entries.items())}}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_ledger(path: Path, *, digests: Mapping[str, str], expected_size: int) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited 012 ledger: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "experiment", "description", "source", "manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256",
                                     "confirmation_012_sha256", "lock_012_sha256", "model", "entries", "content_sha256"}, "inherited 012 ledger")
    if payload["schema_version"] != INHERITED_LEDGER_SCHEMA_VERSION or payload["experiment"] != "012" or payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("inherited 012 ledger schema or digest is not frozen")
    recorded = (payload["manifest_sha256"], payload["extension_sha256"], payload["confirmation_006_sha256"], payload["confirmation_009_sha256"], payload["confirmation_011_sha256"], payload["confirmation_012_sha256"], payload["lock_012_sha256"])
    if recorded != (digests["manifest"], digests["extension"], digests["confirmation_006"], digests["confirmation_009"], digests["confirmation_011"], digests["confirmation_012"], digests["lock_012"]):
        raise ValueError("inherited 012 ledger was recorded against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision} or len(payload["entries"]) != expected_size:
        raise ValueError("inherited 012 ledger model or size is not frozen")
    return payload


def check_ledger_replication(measured: Mapping[str, Mapping[str, Any]], recorded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = set(recorded) - set(measured)
    if missing:
        raise pm.IncidentError(f"recomputed ledger lacks {len(missing)} recorded pairs (e.g. {sorted(missing)[:3]})")
    worst_key, worst, worst_012 = None, 0.0, 0.0
    for key, value in recorded.items():
        other = measured[key]
        for name in LEDGER_FIELDS:
            deviation = abs(value[name] - other[name])
            if deviation > worst:
                worst_key, worst = f"{key}:{name}", deviation
        for component in ra.COMPONENT_ORDER:
            deviation = abs(value["c_k"][component] - other["c_k"][component])
            if deviation > worst:
                worst_key, worst = f"{key}:{component}", deviation
        worst_012 = max(worst_012, abs(value["c_012"] - other["c_012"]))
    if worst > REPLICATION_TOLERANCE:
        raise pm.IncidentError(f"{worst_key}: recomputed quantity deviates from Experiment 012's record by {worst:.3e}")
    if worst_012 > MODEL_012_TOLERANCE:
        raise pm.IncidentError(f"the frozen Experiment 012 model is not reproduced (max ĉ_012 difference {worst_012:.3e})")
    return {"passed": True, "n": len(recorded), "max_abs_deviation": worst, "worst_key": worst_key, "max_c_012_difference": worst_012}


def fresh_tokens_013(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
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


def _prompt_entry(tokenizer: Any, frame: pm.Frame, token: Mapping[str, Any]) -> dict[str, Any]:
    ids = frame.prompt_ids(token["token_id"])
    text = pm._decode(tokenizer, ids)
    if pm._encode(tokenizer, text) != ids:
        raise ValueError(f"{frame.frame_id}/{token['word']}: text does not round-trip to the constructed token IDs")
    return {"frame_id": frame.frame_id, "word": token["word"], "token_id": token["token_id"], "text": text, "token_ids": list(ids), "p_c": frame.p_c, "p_t": frame.p_t}


def build_confirmation_payload(tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> dict[str, Any]:
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_013(tokenizer, excluded_ids)
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
    fresh_prompts = [_prompt_entry(tokenizer, frame, token) for frame in frame_objects for token in tokens]
    exposed_prompts = [_prompt_entry(tokenizer, frame, token) for frame in pool.frames for token in tokens]
    payload = {"schema_version": CONFIRMATION_SCHEMA_VERSION, "manifest_sha256": digests["manifest"], "extension_sha256": digests["extension"],
               "confirmation_006_sha256": digests["confirmation_006"], "confirmation_009_sha256": digests["confirmation_009"], "confirmation_011_sha256": digests["confirmation_011"], "confirmation_012_sha256": digests["confirmation_012"],
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "reference_cue_ids": dict(pool.reference_ids),
               "exposed_token_ids": sorted(token_id for _, token_id in pool.tokens), "exposed_frame_ids": [frame.frame_id for frame in pool.frames],
               "policy": {"licensing": "every fresh cue in every fresh frame and in every exposed frame", "quotas": dict(QUOTAS), "expectations": "none: the committed residual predictions are the only predictions"},
               "tokens": tokens, "frames": frames, "token_prompts": fresh_prompts, "exposed_frame_prompts": exposed_prompts,
               "construction": "tokenizer-only; first eligible candidates per lexical class; classes carry no expectation; no model output"}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


@dataclass(frozen=True)
class Confirmation013:
    reference_ids: Mapping[str, int]
    frames: tuple[pm.Frame, ...]  # the six fresh frames
    exposed_frame_ids: tuple[str, ...]
    tokens: tuple[dict[str, Any], ...]
    token_prompts: tuple[pm.Prompt, ...]  # fresh tokens × fresh frames
    exposed_frame_prompts: tuple[pm.Prompt, ...]  # fresh tokens × exposed frames
    content_sha256: str

    def frame_prompts(self) -> tuple[pm.Prompt, ...]:
        return pm.manifest_prompts(self.frames)

    def reference_prompt(self, frame: pm.Frame) -> pm.Prompt:
        return pm.Prompt(frame, self.reference_ids[frame.template_id], "ref")

    @property
    def all_prompts(self) -> tuple[pm.Prompt, ...]:
        return self.frame_prompts() + self.token_prompts + self.exposed_frame_prompts

    def frames_for(self, word: str) -> tuple[pm.Frame, ...]:
        licensed = next(token["licensed_frames"] for token in self.tokens if token["word"] == word)
        return tuple(frame for frame in self.frames if frame.frame_id in licensed)


def validate_confirmation(payload: Mapping[str, Any], pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation013:
    pm._require_exact_keys(payload, {"schema_version", "manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "model",
                                     "reference_cue_ids", "exposed_token_ids", "exposed_frame_ids", "policy", "tokens", "frames", "token_prompts", "exposed_frame_prompts", "construction", "content_sha256"}, "confirmation")
    if payload["schema_version"] != CONFIRMATION_SCHEMA_VERSION:
        raise ValueError("confirmation schema version is not frozen")
    if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("confirmation content_sha256 does not match canonical payload")
    recorded = (payload["manifest_sha256"], payload["extension_sha256"], payload["confirmation_006_sha256"], payload["confirmation_009_sha256"], payload["confirmation_011_sha256"], payload["confirmation_012_sha256"])
    if recorded != (digests["manifest"], digests["extension"], digests["confirmation_006"], digests["confirmation_009"], digests["confirmation_011"], digests["confirmation_012"]):
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
    return Confirmation013(dict(payload["reference_cue_ids"]), frames, tuple(payload["exposed_frame_ids"]), tuple(tokens), tuple(fresh_prompts), tuple(exposed_prompts), payload["content_sha256"])


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


def load_confirmation(path: Path, pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation013:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, pool, digests)


# ---------------------------------------------------------------------------
# Results state, phases, statistics.

_STATE_KEYS = {"schema_version", "run_id", "created_at", "manifest_sha256", "extension_sha256", "confirmation_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256",
               "lock_011_sha256", "lock_012_sha256", "protocol_code_commit", "git_dirty", "model", "versions", "phases", "executed_prompt_keys", "executed_noun_keys", "exploration", "lock", "confirmation", "invalidated_runs", "state_sha256"}
DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "lock_011", "lock_012")
STATE_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "lock_011_sha256", "lock_012_sha256")


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


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation013) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


def tau_rule(rmse: float) -> float:
    return max(TAU_MIN, TAU_MULTIPLIER * rmse)


def rmse(predicted: Sequence[float], measured: Sequence[float]) -> float | None:
    return math.sqrt(pm._mean([(p - m) ** 2 for p, m in zip(predicted, measured)])) if predicted else None


def spread(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = pm._mean(list(values))
    return math.sqrt(pm._mean([(v - mean) ** 2 for v in values]))


comparison = lc.comparison
explained_variance = lc.explained_variance


# ---------------------------------------------------------------------------
# Exploration (Tier A): reference states, re-measurement, replication, ladder, calibration record.


def _token_means(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Means over one frame set: measured quantities from the analyses; predicted quantities from ``predictions`` (the lock/stage-1 rows) or from the analyses' live predictions."""
    if not analyses:
        return {}
    preds = list(predictions) if predictions is not None else [a["prediction"] for a in analyses]
    mean = lambda values: pm._mean(list(values))  # noqa: E731
    out = {"n_frames": len(analyses), "frames": sorted(a["frame_id"] for a in analyses),
           "c_L_mean": mean(a["c_L"] for a in analyses), "c_M_mean": mean(a["c_M"] for a in analyses), "c_H_mean": mean(a["c_H"] for a in analyses), "P1_mean": mean(a["P1"] for a in analyses),
           "P1_prime_mean": mean(a["P1_prime"] for a in analyses), "q_T_mean": mean(a["q_T"] for a in analyses), "g_E_mean": mean(a["g_E"] for a in analyses),
           "c_012_mean": mean(p["c_012"] for p in preds), "c_own_mean": mean(p["c_own"] for p in preds), "c_L_hat_mean": mean(p["c_L_hat"] for p in preds), "c_M_hat_mean": mean(p["c_M_hat"] for p in preds),
           "c_H_hat_mean": mean(p["c_H_hat"] for p in preds), "r_hat_mean": mean(p["r_hat"] for p in preds), "base_point_mean": mean(p["base_point"] for p in preds), "frozen_attention_mean": mean(p["frozen_attention"] for p in preds)}
    out["r_mean"] = out["c_L_mean"] - out["c_012_mean"]
    out["remainder_mean"] = out["c_L_mean"] - out["c_L_hat_mean"]
    out["remainder_direct_pattern_change_mean"] = mean(a["ladder"]["remainder_direct_pattern_change"] for a in analyses)
    out["q_hat_prime_mean"] = out["g_E_mean"] + out["c_L_hat_mean"]
    return out


def _head_agreement(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for key in HEAD_KEYS:
        out[key] = comparison([p["head_terms"][key] for p in predictions], [a["c_k"][key] for a in analyses])
    return out


def _compensation_summary(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cases = []
    for a, p in zip(analyses, predictions):
        if is_compensation_case(p["c_M_hat"], p["c_H_hat"]):
            cases.append({"token": a["token"], "frame_id": a["frame_id"], "c_M_hat": p["c_M_hat"], "c_H_hat": p["c_H_hat"], "c_M": a["c_M"], "c_H": a["c_H"],
                          "signs_as_predicted": (a["c_M"] > 0) == (p["c_M_hat"] > 0) and (a["c_H"] > 0) == (p["c_H_hat"] > 0)})
    return {"n": len(cases), "fraction_signs_as_predicted": (sum(1 for c in cases if c["signs_as_predicted"]) / len(cases)) if cases else None, "cases": cases}


def exposed_statistics(pairs: Mapping[str, Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, Any]:
    analyses = list(pairs.values())
    predictions = [a["prediction"] for a in analyses]
    by_token = {name: [a for a in analyses if a["token"] == name] for name in tokens}
    means = {name: _token_means(rows) for name, rows in by_token.items() if rows}
    names = sorted(means)
    m = lambda key: [means[n][key] for n in names]  # noqa: E731
    stats = {
        "n_tokens": len(names), "n_pairs": len(analyses),
        "token_means": {
            "residual_r_hat_vs_r": comparison(m("r_hat_mean"), m("r_mean")),
            "base_point_only_vs_r": comparison(m("base_point_mean"), m("r_mean")),
            "full_c_L_hat_vs_c_L": comparison(m("c_L_hat_mean"), m("c_L_mean")),
            "model_012_vs_c_L": comparison(m("c_012_mean"), m("c_L_mean")),
            "own_base_vs_c_L": comparison(m("c_own_mean"), m("c_L_mean")),
            "c_M_hat_vs_c_M": comparison(m("c_M_hat_mean"), m("c_M_mean")),
            "c_H_hat_vs_c_H": comparison(m("c_H_hat_mean"), m("c_H_mean")),
            "q_hat_prime_vs_P1_prime": comparison(m("q_hat_prime_mean"), m("P1_prime_mean")),
            "r_spread": spread(m("r_mean")), "r_mean_of_means": pm._mean(m("r_mean")) if names else None,
        },
        "pairs": {
            "residual_r_hat_vs_r": comparison([p["r_hat"] for p in predictions], [a["r"] for a in analyses]),
            "full_c_L_hat_vs_c_L": comparison([p["c_L_hat"] for p in predictions], [a["c_L"] for a in analyses]),
            "c_H_hat_vs_c_H": comparison([p["c_H_hat"] for p in predictions], [a["c_H"] for a in analyses]),
            "c_M_hat_vs_c_M": comparison([p["c_M_hat"] for p in predictions], [a["c_M"] for a in analyses]),
            "attention_input_hat_vs_measured": comparison([p["c_M_hat"] - p["c_own"] for p in predictions], [a["ladder"]["attention_input_term"] for a in analyses]),
            "c_H_spread": spread([a["c_H"] for a in analyses]),
        },
        "per_template_pairs_residual": {template: comparison([a["prediction"]["r_hat"] for a in analyses if a["template"] == template], [a["r"] for a in analyses if a["template"] == template]) for template in pm.TEMPLATE_ORDER},
        "per_head_pairs": _head_agreement(analyses, predictions),
        "ladder_means": {"base_point": pm._mean([a["ladder"]["base_point"] for a in analyses]), "frozen_attention": pm._mean([a["ladder"]["frozen_attention"] for a in analyses]),
                         "remainder": pm._mean([a["ladder"]["remainder"] for a in analyses]), "remainder_abs": pm._mean([abs(a["ladder"]["remainder"]) for a in analyses]),
                         "remainder_direct_pattern_change_abs": pm._mean([abs(a["ladder"]["remainder_direct_pattern_change"]) for a in analyses]),
                         "remainder_indirect_abs": pm._mean([abs(a["ladder"]["remainder_indirect"]) for a in analyses]),
                         "remainder_token_mean_abs": pm._mean([abs(means[n]["remainder_mean"]) for n in names]) if names else None},
        "compensation": _compensation_summary(analyses, predictions),
        "head_identity_max_error": max(a["head_identity_max_error"] for a in analyses) if analyses else None,
    }
    return stats


def run_exploration(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, *, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], inherited: Mapping[str, Any], state: dict[str, Any], results_path: Path | None, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = HeadSet.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    nouns = pool.single_nouns
    exploration = state["exploration"]
    say("stage axes from the Experiment 010 pool, checked against the Experiment 011 lock")
    axis_T, e_axis = lc.locked_axes(lock_011, cs.stage_axes(cache, weights, pool_010))
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    fpm = model_from_locks(lock_011, lock_012, lw, heads, pool.reference_ids, plural_ids)
    exploration["read_weight_check"] = lc.check_read_weight(fpm.read, head, axis_T)
    exploration["axes_vectors"] = {"T": axis_T.direction.tolist(), "R0": e_axis.direction.tolist()}
    exploration["sigma_T"] = axis_T.sigma
    exploration["read_weight"] = fpm.read.weight.tolist()
    exploration["denominators_E"] = {template: fpm.read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}
    exploration["defined_templates"] = list(lock_012["defined_templates"])
    recorded = inherited["entries"]
    say(f"reference states of the {len(pool.frames)} exposed frames; re-measurement of the {len(recorded)} recorded pairs with the pattern rows captured")
    states: dict[str, FrameState013] = {}
    locked_states: dict[str, Any] = {}
    pairs: dict[str, dict[str, Any]] = {}
    ledger: dict[str, dict[str, Any]] = {}
    identities: dict[str, float] = {}
    for frame in pool.frames:
        state_f = capture_frame_013(model, head, pool.reference_prompt(frame), nouns, axis_T)
        states[frame.frame_id] = state_f
        locked_states[frame.frame_id] = state_f.locked_state()
        names = [name for name, _ in pool.tokens if f"{name}|{frame.frame_id}" in recorded]
        if not names:
            continue
        records = {name: measure_pair_013(model, weights, head, state_f, name, pool.token_id(name), e_axis, axis_T, nouns) for name in names}
        plural_name = pool.plural_cue[frame.template_id]
        plural = records[plural_name] if plural_name in records else measure_pair_013(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for name, record in records.items():
            analysis = analyse_pair_013(record, plural, model=fpm, weights=weights, state=state_f, axis_T=axis_T)
            key = f"{name}|{frame.frame_id}"
            if analysis is None:
                raise pm.IncidentError(f"{key}: no analysis although Experiment 012 recorded the pair")
            er._max_errors(identities, analysis["identities"])
            pairs[key] = analysis
            ledger[key] = ledger_entry_from_013(analysis)
        say(f"  {frame.frame_id}: {len(names)} pairs")
    exploration["identities"] = identities
    exploration["locked_states"] = locked_states
    exploration["locked_state_digests"] = {frame_id: state_digest(entry) for frame_id, entry in locked_states.items()}
    exploration["replication"] = {"experiment_012": check_ledger_replication(ledger, recorded)}
    say(f"replication against Experiment 012: max deviation {exploration['replication']['experiment_012']['max_abs_deviation']:.2e}; ĉ_012 max difference {exploration['replication']['experiment_012']['max_c_012_difference']:.2e}")
    exploration["pairs"] = pairs
    tokens = [name for name, _ in pool.tokens]
    stats = exposed_statistics(pairs, tokens)
    exploration["exposed_check"] = stats
    rmse_r = stats["token_means"]["residual_r_hat_vs_r"]["rmse"]
    rmse_a = stats["pairs"]["c_H_hat_vs_c_H"]["rmse"]
    exploration["calibration"] = {"rmse_r": rmse_r, "rmse_a": rmse_a, "design_rmse_r": DESIGN_RMSE_R, "design_rmse_a": DESIGN_RMSE_A, "tau_r": TAU_R, "tau_a": TAU_A,
                                  "tau_r_from_rule": tau_rule(rmse_r), "tau_a_from_rule": tau_rule(rmse_a), "c_H_degenerate_sd": C_H_DEGENERATE_SD}
    if abs(rmse_r - DESIGN_RMSE_R) > CALIBRATION_TOLERANCE or abs(rmse_a - DESIGN_RMSE_A) > CALIBRATION_TOLERANCE:
        raise pm.IncidentError(f"the recomputed exposed RMSEs ({rmse_r:.4f}, {rmse_a:.4f}) differ from the design values ({DESIGN_RMSE_R}, {DESIGN_RMSE_A}) by more than {CALIBRATION_TOLERANCE}")
    thy = {frame_id: {"c_M": a["c_M"], "c_H": a["c_H"], "c_M_hat": a["prediction"]["c_M_hat"], "c_H_hat": a["prediction"]["c_H_hat"]} for key, a in pairs.items() if key.startswith("thy|") for frame_id in [a["frame_id"]]}
    exploration["thy"] = thy
    exploration["summary"] = {"defined_templates": exploration["defined_templates"], "rmse_r": rmse_r, "rmse_a": rmse_a, "tau_r": TAU_R, "tau_a": TAU_A,
                              "spearman_r": stats["token_means"]["residual_r_hat_vs_r"]["spearman"], "r2_r": stats["token_means"]["residual_r_hat_vs_r"]["r2"], "prediction_undefined": len(exploration["defined_templates"]) < 2}
    say(f"exposed residual prediction: Spearman {exploration['summary']['spearman_r']:.3f}, R² {exploration['summary']['r2_r']:.3f}; RMSE_r {rmse_r:.4f} (design {DESIGN_RMSE_R}), RMSE_A {rmse_a:.4f} (design {DESIGN_RMSE_A})")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


# ---------------------------------------------------------------------------
# Lock: the prediction table for fresh tokens × exposed frames from the locked states.


def prediction_table(fpm: FrozenPatternModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], frames: Sequence[pm.Frame], tokens: Sequence[Mapping[str, Any]], defined: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        if frame.template_id not in defined:
            continue
        for token in tokens:
            rows.append(prediction_row(token["word"], frame.frame_id, frame.template_id, fpm.predict_from_locked(weights, locked_states[frame.frame_id], token["token_id"], frame.template_id)))
    return rows


def table_digest(rows: Sequence[Mapping[str, Any]], states: Mapping[str, Any]) -> str:
    return pm.sha256_text(pm.canonical_json({"rows": list(rows), "states": dict(states)}))


def token_means_from_table(rows: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for word in tokens:
        mine = [row for row in rows if row["token"] == word]
        if mine:
            out[word] = {key: pm._mean([row[key] for row in mine]) for key in ("c_012", "c_own", "base_point", "frozen_attention", "r_hat", "c_M_hat", "c_H_hat", "c_L_hat", "g_E")} | {"n_frames": len(mine)}
    return out


def lock_predictions(fpm: FrozenPatternModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], pool: cs.Pool008, confirmation: Confirmation013, defined: Sequence[str]) -> dict[str, Any]:
    rows = prediction_table(fpm, weights, locked_states, pool.frames, confirmation.tokens, defined)
    c_012 = {token["word"]: {template: fpm.read.predict(weights, fpm.lw, fpm.bases_012[template], token["token_id"], template)["c_hat"] for template in defined} for token in confirmation.tokens}
    return {"rows": rows, "token_means": token_means_from_table(rows, [token["word"] for token in confirmation.tokens]), "c_012_by_template": c_012}


def render_predictions(lock: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 013 — preregistered predictions (frozen-pattern attention paths; residual of the Experiment 012 model)", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_013_sha256']}`; Experiment 012 lock sha256 `{lock['lock_012_sha256']}`",
             f"- Frozen tolerances τ_r {TAU_R} (Y1/Y2 MAE ceiling, token means), τ_A {TAU_A} (Y3, pairs); floors Spearman ≥ {Y_SPEARMAN} and R² ≥ {Y_R2} (Y1/Y2), Spearman ≥ {Y3_SPEARMAN_H} (c_H) and ≥ {Y3_SPEARMAN_M} (c_M) (Y3); c_H degenerate-spread threshold {C_H_DEGENERATE_SD}",
             f"- Y1 table: {len(lock['predictions']['rows'])} rows (fresh tokens × exposed frames); Y2 rows are computed at confirm stage 1 from each fresh frame's reference state and digested before any fresh cue prompt", "",
             "| token | class | frames | ĉ_012 | base-point | frozen-attention | **predicted residual r̂** | ĉ_M | ĉ_H | ĉ_L | g_E |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    categories = {token["word"]: token["category"] for token in lock["tokens"]}
    for word, m in lock["predictions"]["token_means"].items():
        lines.append(f"| {word} | {categories.get(word, '')} | {m['n_frames']} | {f(m['c_012'], 3)} | {f(m['base_point'], 3)} | {f(m['frozen_attention'], 3)} | **{f(m['r_hat'], 3)}** | {f(m['c_M_hat'], 3)} | {f(m['c_H_hat'], 3)} | {f(m['c_L_hat'], 3)} | {f(m['g_E'], 3)} |")
    lines += ["", "Per-template ĉ_012 (the frozen Experiment 012 model): " + "; ".join(f"{w}: " + ", ".join(f"{t} {f(v, 3)}" for t, v in entry.items()) for w, entry in lock["predictions"]["c_012_by_template"].items()), ""]
    return "\n".join(lines)


def build_candidate_lock(*, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation013, predictions: Mapping[str, Any], protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    if exploration["summary"]["prediction_undefined"]:
        raise PhaseError("fewer than two templates are defined: PREDICTION_UNDEFINED, no lock")
    lock = {"schema_version": 1, "experiment": "013", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit}
    for field, key in zip(STATE_DIGEST_FIELDS, DIGEST_KEYS):
        lock[field] = digests[key] if key != "confirmation_013" else confirmation.content_sha256
    lock.update({"inherited_ledger_sha256": digests["ledger_012"], "confirmation_prompt_keys": sorted(prompt.key for prompt in confirmation.all_prompts),
                 "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED}, "head": ht.HEAD_KEY,
                 "axes_vectors": exploration["axes_vectors"], "sigma_T": exploration["sigma_T"], "read_weight": exploration["read_weight"], "denominators_E": exploration["denominators_E"],
                 "defined_templates": exploration["defined_templates"], "locked_states": exploration["locked_states"], "locked_state_digests": exploration["locked_state_digests"],
                 "tolerances": {"tau_r": TAU_R, "tau_a": TAU_A, "design_rmse_r": DESIGN_RMSE_R, "design_rmse_a": DESIGN_RMSE_A, "recomputed_rmse_r": exploration["calibration"]["rmse_r"], "recomputed_rmse_a": exploration["calibration"]["rmse_a"]},
                 "floors": {"y_spearman": Y_SPEARMAN, "y_r2": Y_R2, "y3_spearman_h": Y3_SPEARMAN_H, "y3_spearman_m": Y3_SPEARMAN_M, "c_h_degenerate_sd": C_H_DEGENERATE_SD, "compensation_min": COMPENSATION_MIN,
                            "min_valid_frames": MIN_VALID_FRAMES, "min_valid_frames_per_token": MIN_VALID_FRAMES_PER_TOKEN, "min_scored_tokens": MIN_SCORED_TOKENS, "frame_cue_effect_rate": FRAME_CUE_EFFECT_RATE, "head_stage_floor": cs.STAGE_UNINFORMATIVE_FLOOR},
                 "tokens": [dict(token) for token in confirmation.tokens], "exposed_check": exploration["exposed_check"], "thy": exploration.get("thy"),
                 "predictions": dict(predictions), "tier_a_results_sha256": state.get("state_sha256")})
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation013, predictions_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "013" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked or git_state.get("dirty", True):
        raise PhaseError("confirm requires the committed lock and predictions on a clean Git tree")
    expected = dict(zip(STATE_DIGEST_FIELDS, digest_tuple(digests)))
    expected["confirmation_013_sha256"] = confirmation.content_sha256
    if any(lock.get(field) != value for field, value in expected.items()) or lock.get("inherited_ledger_sha256") != digests["ledger_012"]:
        raise PhaseError("lock was built against different frozen inputs")
    if not state.get("lock") or lock["content_sha256"] != state["lock"]["content_sha256"] or lock["run_id"] != state["run_id"]:
        raise PhaseError("the installed lock is not the candidate lock written by the lock phase")
    if pm.sha256_text(predictions_text) != state["lock"]["predictions_sha256"]:
        raise PhaseError("the installed predictions.md is not the artifact written by the lock phase")
    if lock["tolerances"]["tau_r"] != TAU_R or lock["tolerances"]["tau_a"] != TAU_A or lock["floors"]["y_r2"] != Y_R2:
        raise PhaseError("the lock's tolerances or floors are not the frozen constants")
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = [path for path in changed_paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and path not in NON_SCIENTIFIC_PATHS and not path.startswith(NON_SCIENTIFIC_PREFIXES)]
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock commit: {scientific}")
    assert_confirmation_untouched(state, confirmation)


def assert_lock_predictions_reproduced(lock: Mapping[str, Any], recomputed: Mapping[str, Any]) -> float:
    worst = 0.0
    locked_rows, fresh_rows = lock["predictions"]["rows"], recomputed["rows"]
    if len(locked_rows) != len(fresh_rows):
        raise PhaseError("the recomputed prediction table has a different number of rows; nothing was executed")
    for a, b in zip(locked_rows, fresh_rows):
        if (a["token"], a["frame_id"]) != (b["token"], b["frame_id"]):
            raise PhaseError("the recomputed prediction table is ordered differently; nothing was executed")
        for key in PREDICTION_COLUMNS[3:]:
            if key == "head_terms":
                worst = max(worst, max(abs(a["head_terms"][h] - b["head_terms"][h]) for h in HEAD_KEYS))
            else:
                worst = max(worst, abs(a[key] - b[key]))
    for word, entry in lock["predictions"]["c_012_by_template"].items():
        for template, value in entry.items():
            worst = max(worst, abs(value - recomputed["c_012_by_template"][word][template]))
    if worst > LOCK_PREDICTION_TOLERANCE:
        raise PhaseError(f"the weights, locked axes, the frozen 012 model and the locked reference states do not reproduce the preregistered predictions (max difference {worst:.3e}); nothing was executed")
    return worst


# ---------------------------------------------------------------------------
# Confirmation (Tier C): stage 1 (fresh frames' reference states, validity, digested prediction table), stage 2 (fresh cues), scoring.


def stage_one(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation013, lock: Mapping[str, Any], lock_012: Mapping[str, Any], *, protocol_code_commit: str, log: Any = None) -> dict[str, Any]:
    """The fresh frames' reference runs (exposed tokens only), their validity, and the frame-conditional prediction table; nothing fresh-cue runs here."""
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = HeadSet.from_model(model)
    nouns = pool.single_nouns
    cache = pm.PromptCache(model, tuple(pool.nouns))
    say("stage 1: axes against the lock; fresh frames' reference states, validity, prediction table")
    axis_T, e_axis = lc.locked_axes(lock, cs.stage_axes(cache, weights, pool_010))
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    fpm = model_from_locks(lock, lock_012, lw, heads, pool.reference_ids, plural_ids)
    lc.check_read_weight(fpm.read, head, axis_T)
    frames_out: dict[str, Any] = {}
    states: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for frame in confirmation.frames:
        template = frame.template_id
        if template not in lock["defined_templates"]:
            frames_out[frame.frame_id] = {"template_id": template, "valid": False, "template_defined": False, "note": "template excluded before the lock; nothing run"}
            continue
        state_f = capture_frame_013(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        sg, pl = pm.frame_prompts(frame)
        c_a, c_b = cache.c(sg), cache.c(pl)
        positive = sum(1 for noun in nouns if c_a[noun.lexical_key] - c_b[noun.lexical_key] > 0)
        required = pm.exact_count_floor(FRAME_CUE_EFFECT_RATE, len(nouns))
        plural = measure_pair_013(model, weights, head, state_f, pool.plural_cue[template], pl.cue_token_id, e_axis, axis_T, nouns)
        er.enforce_identities(plural, plural, f"{frame.frame_id}/{pool.plural_cue[template]}")
        head_informative = abs(plural.head_change) >= cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma
        valid = head_informative and positive >= required
        frames_out[frame.frame_id] = {"template_id": template, "valid": valid, "head_informative": head_informative, "plural_head_change": plural.head_change, "cue_effect_positive": positive, "cue_effect_required": required,
                                      "template_defined": True, "reconstruction_error": state_f.ref.reconstruction_error, "attention_ref": state_f.functional.attention_ref, "sigma_ref": state_f.functional.sigma_ref}
        states[frame.frame_id] = state_f.locked_state()
        for token in confirmation.tokens:
            rows.append(prediction_row(token["word"], frame.frame_id, template, fpm.predict_from_state(weights, state_f, token["token_id"], template)))
        say(f"  {frame.frame_id}: {'valid' if valid else 'INVALID'} (head informative {head_informative}, cue effect {positive}/{required}); {len(confirmation.tokens)} predictions")
    digest = table_digest(rows, states)
    return {"frames": frames_out, "states": states, "state_digests": {frame_id: state_digest(entry) for frame_id, entry in states.items()}, "rows": rows,
            "token_means": token_means_from_table(rows, [token["word"] for token in confirmation.tokens]), "digest": digest, "commit": protocol_code_commit,
            "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}


def assert_stage_one_digest(stage1: Mapping[str, Any]) -> None:
    if not stage1 or stage1.get("digest") != table_digest(stage1["rows"], stage1["states"]):
        raise PhaseError("stage 1's prediction table is missing or its digest does not verify; no fresh cue prompt may run")


def stage_two(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation013, lock: Mapping[str, Any], lock_012: Mapping[str, Any], stage1: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    """The fresh cues in both sets, the exact ledger and per-head identities, scored against the locked table (Y1) and the digested stage-1 table (Y2, Y3)."""
    say = log or (lambda message: None)
    assert_stage_one_digest(stage1)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = HeadSet.from_model(model)
    nouns = pool.single_nouns
    cache = pm.PromptCache(model, tuple(pool.nouns))
    axis_T, e_axis = lc.locked_axes(lock, cs.stage_axes(cache, weights, pool_010))
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    fpm = model_from_locks(lock, lock_012, lw, heads, pool.reference_ids, plural_ids)
    identities: dict[str, float] = {}
    tokens_by_word = {token["word"]: token for token in confirmation.tokens}
    say("stage 2: fresh cues in the exposed frames (Y1) and in the valid fresh frames (Y2)")
    pairs_exposed: dict[str, dict[str, Any]] = {}
    for frame in pool.frames:
        if frame.template_id not in lock["defined_templates"]:
            continue
        state_f = capture_frame_013(model, head, pool.reference_prompt(frame), nouns, axis_T)
        locked = lock["locked_states"][frame.frame_id]
        drift = max(float((state_f.x1 - torch.tensor(locked["x1"], dtype=torch.float64)).abs().max()), float((state_f.x2 - torch.tensor(locked["x2"], dtype=torch.float64)).abs().max()),
                    max(abs(state_f.pattern_weight(key) - locked["A_pc"][key]) for key in HEAD_KEYS))
        if drift > LOCKED_STATE_TOLERANCE:
            raise pm.IncidentError(f"{frame.frame_id}: the re-captured reference state differs from the locked one (max {drift:.2e})")
        plural_name = pool.plural_cue[frame.template_id]
        plural = measure_pair_013(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair_013(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_013(record, plural, model=fpm, weights=weights, state=state_f, axis_T=axis_T)
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
        state_f = capture_frame_013(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        if state_digest(state_f.locked_state()) != stage1["state_digests"][frame.frame_id]:
            raise pm.IncidentError(f"{frame.frame_id}: the re-captured reference state differs from stage 1's digested state")
        sg, pl = pm.frame_prompts(frame)
        plural = measure_pair_013(model, weights, head, state_f, pool.plural_cue[frame.template_id], pl.cue_token_id, e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair_013(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_013(record, plural, model=fpm, weights=weights, state=state_f, axis_T=axis_T)
            if analysis is None:
                raise pm.IncidentError(f"{frame.frame_id}/{token['word']}: no analysis although the frame is valid")
            er._max_errors(identities, analysis["identities"])
            c_word = cache.c(pm.Prompt(frame, token["token_id"], token["word"]))
            analysis["dc_beh"] = pm._mean([c_word[noun.lexical_key] - state_f.ref.c_by_noun[noun.lexical_key] for noun in nouns])
            analysis["dc"] = pm._mean(list(record.shifts.values()))
            pairs_fresh[f"{token['word']}|{frame.frame_id}"] = analysis
        say(f"  {frame.frame_id} (fresh): {len(confirmation.tokens)} tokens")
    results = score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock)
    results["identities"] = identities
    results["head_identity_max_error"] = max([a["head_identity_max_error"] for a in list(pairs_exposed.values()) + list(pairs_fresh.values())] or [0.0])
    return results


def _rows_by_pair(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {f"{row['token']}|{row['frame_id']}": row for row in rows}


def _score_residual(pairs: Mapping[str, Mapping[str, Any]], table: Mapping[str, Mapping[str, Any]], words: Sequence[str], *, min_frames: int) -> dict[str, Any]:
    """Token means over identical frame sets of the table's r̂ and the measured r = c_L − table ĉ_012; the three floors."""
    tokens_out = {}
    for word in words:
        keys = sorted(key for key in pairs if key.split("|")[0] == word and key in table)
        analyses = [pairs[key] for key in keys]
        predictions = [table[key] for key in keys]
        means = _token_means(analyses, predictions) if analyses else {}
        tokens_out[word] = {"scored": len(keys) >= min_frames, "n_valid_frames": len(keys), **means}
    scored = [word for word, entry in tokens_out.items() if entry["scored"]]
    result: dict[str, Any] = {"tokens": tokens_out, "scored_tokens": scored}
    if len(scored) >= 2:
        r_hat = [tokens_out[w]["r_hat_mean"] for w in scored]
        r = [tokens_out[w]["r_mean"] for w in scored]
        c = comparison(r_hat, r)
        failing = [name for name, ok in (("spearman", c["spearman"] is not None and c["spearman"] >= Y_SPEARMAN), ("mae", c["mae"] <= TAU_R), ("r2", c["r2"] is not None and c["r2"] >= Y_R2)) if not ok]
        result["test"] = {**c, "tau_r": TAU_R, "r2_floor": Y_R2, "spearman_floor": Y_SPEARMAN, "passed": not failing, "failing": failing}
        result["pairs"] = comparison([table[k]["r_hat"] for w in scored for k in sorted(key for key in pairs if key.split("|")[0] == w and key in table)], [pairs[k]["r"] for w in scored for k in sorted(key for key in pairs if key.split("|")[0] == w and key in table)])
        result["beside"] = {"model_012_vs_c_L": comparison([tokens_out[w]["c_012_mean"] for w in scored], [tokens_out[w]["c_L_mean"] for w in scored]),
                            "full_c_L_hat_vs_c_L": comparison([tokens_out[w]["c_L_hat_mean"] for w in scored], [tokens_out[w]["c_L_mean"] for w in scored]),
                            "own_base_vs_c_L": comparison([tokens_out[w]["c_own_mean"] for w in scored], [tokens_out[w]["c_L_mean"] for w in scored]),
                            "base_point_only_vs_r": comparison([tokens_out[w]["base_point_mean"] for w in scored], r),
                            "q_hat_prime_vs_P1_prime": comparison([tokens_out[w]["q_hat_prime_mean"] for w in scored], [tokens_out[w]["P1_prime_mean"] for w in scored]),
                            "r_spread": spread(r), "r_mean": pm._mean(r),
                            "ladder": {"base_point": pm._mean([tokens_out[w]["base_point_mean"] for w in scored]), "frozen_attention": pm._mean([tokens_out[w]["frozen_attention_mean"] for w in scored]),
                                       "remainder": pm._mean([tokens_out[w]["remainder_mean"] for w in scored]), "remainder_abs": pm._mean([abs(tokens_out[w]["remainder_mean"]) for w in scored]),
                                       "remainder_direct_pattern_change": pm._mean([tokens_out[w]["remainder_direct_pattern_change_mean"] for w in scored])}}
    return result


def score_confirmation(stage1: Mapping[str, Any], pairs_exposed: Mapping[str, Mapping[str, Any]], pairs_fresh: Mapping[str, Mapping[str, Any]], confirmation: Confirmation013, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Pure scoring: Y1 on the locked table (token means), Y2 on the stage-1 table (token means, with the frame precondition), Y3 on the pairs of both sets (with the degenerate-spread guard)."""
    words = [token["word"] for token in confirmation.tokens]
    locked_table = _rows_by_pair(lock["predictions"]["rows"])
    stage1_table = _rows_by_pair(stage1["rows"])
    valid_frames = [frame_id for frame_id, entry in stage1["frames"].items() if entry.get("valid")]
    y1 = _score_residual(pairs_exposed, locked_table, words, min_frames=MIN_VALID_FRAMES_PER_TOKEN)
    y2 = _score_residual(pairs_fresh, stage1_table, words, min_frames=MIN_VALID_FRAMES_PER_TOKEN)
    results: dict[str, Any] = {"frames": stage1["frames"], "valid_frames": valid_frames, "Y1": y1, "Y2": y2, "per_frame_exposed": pairs_exposed, "per_frame_fresh": pairs_fresh}
    results["precondition_Y1"] = {"passed": len(y1["scored_tokens"]) >= MIN_SCORED_TOKENS, "scored_tokens": len(y1["scored_tokens"]), "min_scored_tokens": MIN_SCORED_TOKENS}
    results["precondition_Y2"] = {"passed": len(valid_frames) >= MIN_VALID_FRAMES and len(y2["scored_tokens"]) >= MIN_SCORED_TOKENS, "valid_frames": len(valid_frames), "scored_tokens": len(y2["scored_tokens"]),
                                  "min_valid_frames": MIN_VALID_FRAMES, "min_scored_tokens": MIN_SCORED_TOKENS}
    # Y3: pairs of both sets whose token is scored in that set.
    pair_items = [(pairs_exposed[k], locked_table[k]) for w in y1["scored_tokens"] for k in sorted(key for key in pairs_exposed if key.split("|")[0] == w and key in locked_table)]
    pair_items += [(pairs_fresh[k], stage1_table[k]) for w in y2["scored_tokens"] for k in sorted(key for key in pairs_fresh if key.split("|")[0] == w and key in stage1_table)]
    if len(pair_items) >= 2:
        c_H = [a["c_H"] for a, _ in pair_items]
        c_H_hat = [p["c_H_hat"] for _, p in pair_items]
        c_M = [a["c_M"] for a, _ in pair_items]
        c_M_hat = [p["c_M_hat"] for _, p in pair_items]
        sd_H = spread(c_H)
        h = comparison(c_H_hat, c_H)
        m = comparison(c_M_hat, c_M)
        degenerate = sd_H is None or sd_H < C_H_DEGENERATE_SD
        failing = [name for name, ok in (("c_H_mae", h["mae"] <= TAU_A), ("c_M_spearman", m["spearman"] is not None and m["spearman"] >= Y3_SPEARMAN_M)) if not ok]
        if not degenerate and not (h["spearman"] is not None and h["spearman"] >= Y3_SPEARMAN_H):
            failing.append("c_H_spearman")
        results["Y3"] = {"n_pairs": len(pair_items), "c_H": h, "c_M": m, "c_H_spread": sd_H, "degenerate_spread": degenerate, "tau_a": TAU_A, "floors": {"c_H_spearman": Y3_SPEARMAN_H, "c_M_spearman": Y3_SPEARMAN_M},
                         "evaluable": not degenerate, "passed": (not degenerate) and not failing, "failing": failing,
                         "per_head": _head_agreement([a for a, _ in pair_items], [p for _, p in pair_items]),
                         "attention_input_hat_vs_measured": comparison([p["c_M_hat"] - p["c_own"] for _, p in pair_items], [a["ladder"]["attention_input_term"] for a, _ in pair_items]),
                         "compensation": _compensation_summary([a for a, _ in pair_items], [p for _, p in pair_items])}
    else:
        results["Y3"] = {"n_pairs": len(pair_items), "evaluable": False, "passed": False, "failing": ["no pairs"], "degenerate_spread": True}
    categories = {token["word"]: token["category"] for token in confirmation.tokens}
    results["class_means"] = {}
    for label, y in (("Y1", y1), ("Y2", y2)):
        by_class: dict[str, list[str]] = {}
        for w in y["scored_tokens"]:
            by_class.setdefault(categories[w], []).append(w)
        results["class_means"][label] = {c: {"n": len(ws), "r_mean": pm._mean([y["tokens"][w]["r_mean"] for w in ws]), "r_hat_mean": pm._mean([y["tokens"][w]["r_hat_mean"] for w in ws])} for c, ws in sorted(by_class.items())}
    results["outcome"] = outcome(results)
    return results


def outcome(results: Mapping[str, Any]) -> dict[str, Any]:
    if not results["precondition_Y1"]["passed"]:
        y1 = OUTCOME_Y1[2]
    else:
        y1 = OUTCOME_Y1[0] if results["Y1"].get("test", {}).get("passed") else OUTCOME_Y1[1]
    if not results["precondition_Y2"]["passed"]:
        y2 = OUTCOME_Y2[2]
    else:
        y2 = OUTCOME_Y2[0] if results["Y2"].get("test", {}).get("passed") else OUTCOME_Y2[1]
    y3_entry = results["Y3"]
    y3 = OUTCOME_Y3[2] if not y3_entry.get("evaluable") else (OUTCOME_Y3[0] if y3_entry["passed"] else OUTCOME_Y3[1])
    return {"label": f"{y1} | {y2} | {y3}", "Y1": y1, "Y2": y2, "Y3": y3}


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 013 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state['confirmation_013_sha256']}`", f"- Experiment 012 lock sha256: `{state['lock_012_sha256']}`",
             f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    incidents = list(exploration.get("incidents", [])) + ([state["confirmation"]["incident"]] if isinstance(state.get("confirmation"), dict) and "incident" in state["confirmation"] else [])
    if incidents:
        lines += ["## Incidents", ""] + [f"- `{e['phase']}` at commit `{e.get('commit', '?')}` ({e['at']}): {e['message']}" for e in incidents] + [""]

    def cmp_line(c: Mapping[str, Any]) -> str:
        return f"Spearman {f(c.get('spearman'), 3)}, MAE {f(c.get('mae'), 3)}, RMSE {f(c.get('rmse'), 4)}, R² {f(c.get('r2'), 3)}, bias {f(c.get('bias'), 3)} (n {c.get('n')})"

    if "summary" in exploration:
        s, x, cal = exploration["summary"], exploration["exposed_check"], exploration["calibration"]
        rep = exploration["replication"]["experiment_012"]
        lines += ["## Tier A — exposed pool (calibration record only; tolerances are frozen constants)", "",
                  f"- Replication of Experiment 012: {rep['n']} pairs, max deviation {rep['max_abs_deviation']:.2e}; ĉ_012 reproduced to {rep['max_c_012_difference']:.2e}",
                  f"- Identities (max relative errors): {', '.join(f'{k} {v:.1e}' for k, v in exploration['identities'].items())}; per-head OV identity max {x['head_identity_max_error']:.1e}; read weight vs Experiment 011 lock {exploration['read_weight_check']:.1e}",
                  f"- Frozen τ_r {cal['tau_r']} (design RMSE {cal['design_rmse_r']}, recomputed {f(cal['rmse_r'], 4)}); τ_A {cal['tau_a']} (design RMSE {cal['design_rmse_a']}, recomputed {f(cal['rmse_a'], 4)}); c_H degenerate threshold {cal['c_H_degenerate_sd']}",
                  f"- Token means ({x['n_tokens']} tokens): residual r̂ vs r — {cmp_line(x['token_means']['residual_r_hat_vs_r'])}; r spread sd {f(x['token_means']['r_spread'], 4)}; base-point only R² {f(x['token_means']['base_point_only_vs_r']['r2'], 3)}",
                  f"- Token means: full ĉ_L vs c_L — {cmp_line(x['token_means']['full_c_L_hat_vs_c_L'])}; the 012 model vs c_L — {cmp_line(x['token_means']['model_012_vs_c_L'])}; own base vs c_L — {cmp_line(x['token_means']['own_base_vs_c_L'])}",
                  f"- Pairs ({x['n_pairs']}): residual — {cmp_line(x['pairs']['residual_r_hat_vs_r'])}; ĉ_H vs c_H — {cmp_line(x['pairs']['c_H_hat_vs_c_H'])} (c_H spread sd {f(x['pairs']['c_H_spread'], 4)}); ĉ_M vs c_M — {cmp_line(x['pairs']['c_M_hat_vs_c_M'])}; attention-input — {cmp_line(x['pairs']['attention_input_hat_vs_measured'])}",
                  f"- Per template (pairs, residual): {{{', '.join(f'{t}: {f(v['spearman'], 3)}/{f(v['mae'], 3)}' for t, v in x['per_template_pairs_residual'].items())}}}",
                  f"- Ladder means: base-point {f(x['ladder_means']['base_point'], 3)}, frozen-attention {f(x['ladder_means']['frozen_attention'], 3)}, remainder {f(x['ladder_means']['remainder'], 3)} (|.| {f(x['ladder_means']['remainder_abs'], 3)}; direct pattern change |.| {f(x['ladder_means']['remainder_direct_pattern_change_abs'], 3)}, indirect |.| {f(x['ladder_means']['remainder_indirect_abs'], 3)}; token-mean |.| {f(x['ladder_means']['remainder_token_mean_abs'], 3)})",
                  f"- Per head (pairs Spearman): {{{', '.join(f'{k} {f(v['spearman'], 2)}' for k, v in x['per_head_pairs'].items())}}}",
                  f"- Compensation cases (frozen rule): {x['compensation']['n']} pairs, measured signs as predicted in {f(x['compensation']['fraction_signs_as_predicted'], 2)}",
                  f"- thy (measured c_M / c_H vs predicted): {{{', '.join(f'{fid}: ({f(v['c_M'], 3)}, {f(v['c_H'], 3)}) vs ({f(v['c_M_hat'], 3)}, {f(v['c_H_hat'], 3)})' for fid, v in exploration.get('thy', {}).items())}}}", ""]
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}`; predictions sha256 `{state['lock']['predictions_sha256']}`", ""]
    confirmation = state.get("confirmation")
    if confirmation and "stage1" in confirmation:
        s1 = confirmation["stage1"]
        lines += ["## Confirmation — stage 1 (fresh frames' reference states; prediction table digested before any fresh cue prompt)", "", f"- Table rows {len(s1['rows'])}; digest `{s1['digest']}`; commit `{s1['commit']}`"]
        for frame_id, entry in s1["frames"].items():
            if entry.get("template_defined"):
                lines.append(f"  - {frame_id}: {'valid' if entry['valid'] else 'INVALID'} (plural head change {f(entry['plural_head_change'], 3)}, cue effect {entry['cue_effect_positive']}/{entry['cue_effect_required']})")
            else:
                lines.append(f"  - {frame_id}: {entry.get('note')}")
        lines.append("")
    if confirmation and "outcome" in confirmation:
        o = confirmation["outcome"]
        lines += [f"## Confirmation — stage 2 — `{o['label']}`", ""]
        for label, key in (("Y1 (strict: new tokens × exposed frames; token means)", "Y1"), ("Y2 (frame-conditional: new tokens × fresh frames; token means)", "Y2")):
            y = confirmation[key]
            pre = confirmation["precondition_Y1" if key == "Y1" else "precondition_Y2"]
            if "test" in y:
                t = y["test"]
                lines.append(f"- {label}: scored tokens {len(y['scored_tokens'])} ({'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'}); r̂ vs r — {cmp_line(t)} → {'pass' if t['passed'] else 'FAIL'} {t['failing'] or ''}; pairs — {cmp_line(y['pairs'])}")
                b = y["beside"]
                lines.append(f"  - beside (never judged): the 012 model vs c_L {cmp_line(b['model_012_vs_c_L'])}; full ĉ_L vs c_L {cmp_line(b['full_c_L_hat_vs_c_L'])}; own base vs c_L {cmp_line(b['own_base_vs_c_L'])}; base-point only vs r R² {f(b['base_point_only_vs_r']['r2'], 3)}; q̂' vs P1' {cmp_line(b['q_hat_prime_vs_P1_prime'])}; r spread sd {f(b['r_spread'], 4)}, mean {f(b['r_mean'], 3)}")
                lines.append(f"  - ladder means: base-point {f(b['ladder']['base_point'], 3)}, frozen-attention {f(b['ladder']['frozen_attention'], 3)}, remainder {f(b['ladder']['remainder'], 3)} (|.| {f(b['ladder']['remainder_abs'], 3)}; direct pattern change {f(b['ladder']['remainder_direct_pattern_change'], 3)})")
            else:
                lines.append(f"- {label}: not evaluable ({len(y['scored_tokens'])} scored tokens; {'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'})")
        y3 = confirmation["Y3"]
        if y3.get("evaluable") or "c_H" in y3:
            lines.append(f"- Y3 (attribution, {y3['n_pairs']} pairs): ĉ_H vs c_H — {cmp_line(y3['c_H'])} (spread sd {f(y3['c_H_spread'], 4)}, degenerate {y3['degenerate_spread']}; floors Spearman ≥ {Y3_SPEARMAN_H}, MAE ≤ τ_A {TAU_A}); ĉ_M vs c_M — {cmp_line(y3['c_M'])} (floor ≥ {Y3_SPEARMAN_M}) → {'pass' if y3['passed'] else ('NOT EVALUABLE' if not y3['evaluable'] else 'FAIL')} {y3['failing'] or ''}")
            lines.append(f"  - per head (pairs Spearman): {{{', '.join(f'{k} {f(v['spearman'], 2)}' for k, v in y3['per_head'].items())}}}; attention-input term {cmp_line(y3['attention_input_hat_vs_measured'])}")
            comp = y3["compensation"]
            lines.append(f"  - compensation cases (frozen rule, all listed): {comp['n']} pairs, measured signs as predicted in {f(comp['fraction_signs_as_predicted'], 2)}: " + "; ".join(f"{c['token']}/{c['frame_id']} pred ({f(c['c_M_hat'], 2)}, {f(c['c_H_hat'], 2)}) meas ({f(c['c_M'], 2)}, {f(c['c_H'], 2)}){'' if c['signs_as_predicted'] else ' ✗'}" for c in comp["cases"]))
        else:
            lines.append(f"- Y3: not evaluable ({y3.get('failing')})")
        lines.append(f"- Class means (r, r̂): {{{', '.join(label + ': ' + ', '.join(f'{c} {f(v['r_mean'], 3)}/{f(v['r_hat_mean'], 3)}' for c, v in entry.items()) for label, entry in confirmation['class_means'].items())}}}")
        lines += ["", "| token | class | Y1 frames | Y1 r̂ | Y1 r | Y1 ĉ_012 | Y1 c_L | Y2 frames | Y2 r̂ | Y2 r | Y2 ĉ_012 | Y2 c_L | ĉ_M/c_M (Y2) | ĉ_H/c_H (Y2) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        categories = {token["word"]: token["category"] for token in state["confirmation"].get("tokens_meta", [])} if state["confirmation"].get("tokens_meta") else {}
        for word in sorted(confirmation["Y1"]["tokens"], key=lambda w: confirmation["Y1"]["tokens"][w].get("r_hat_mean") if confirmation["Y1"]["tokens"][w].get("r_hat_mean") is not None else 9):
            a, b = confirmation["Y1"]["tokens"][word], confirmation["Y2"]["tokens"].get(word, {})
            lines.append(f"| {word} | {categories.get(word, '')} | {a.get('n_valid_frames')} | {f(a.get('r_hat_mean'), 3)} | {f(a.get('r_mean'), 3)} | {f(a.get('c_012_mean'), 3)} | {f(a.get('c_L_mean'), 3)} | {b.get('n_valid_frames')} | {f(b.get('r_hat_mean'), 3)} | {f(b.get('r_mean'), 3)} | {f(b.get('c_012_mean'), 3)} | {f(b.get('c_L_mean'), 3)} | {f(b.get('c_M_hat_mean'), 2)}/{f(b.get('c_M_mean'), 2)} | {f(b.get('c_H_hat_mean'), 2)}/{f(b.get('c_H_mean'), 2)} |")
        lines.append("")
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
