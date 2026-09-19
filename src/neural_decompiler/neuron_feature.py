"""Experiment 014: what drives block-2 MLP neuron 1987 — a compact input-side predictor locked before interpretation.

The neuron's computation is reconstructed exactly: residual before block 2 → LayerNorm → one input weight vector →
preactivation → GELU → one output vector onto the head's read direction. The predictor evaluates the exact LayerNorm
and the neuron's own input weights at the state the previously decoded circuit predicts — the frame's reference residual
plus Experiment 013's frozen-pattern arriving change (``ΔE + Δ̂₁ + Σ heads₁`` from weights and the reference state, never a
measured patched residual) — and converts the preactivation change through GELU at the frame's reference operating
point. The first-order LayerNorm Jacobian form is frozen as the compact single-direction feature for the accounting,
and the number-axis alternative (the same computation on the number-axis component of the same predicted change) is
committed for rejection. Predictions for new cues in the 42 exposed frames are locked before any fresh prompt (Y1);
predictions for new cues in six new frames are computed at confirm stage 1 and digested before any fresh cue prompt
(Y2); the alternative is scored on both (Y3). Constants are copied from design revision 2.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import attention_paths as ap
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

EXPERIMENT_DIR = "experiments/014-neuron-1987-feature"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
INHERITED_013_LEDGER_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-013-pair-ledger.json"
EXPERIMENT_011_LOCK_PATH = er.LOCK_RELATIVE_PATH
EXPERIMENT_012_LOCK_PATH = lc.LOCK_RELATIVE_PATH
EXPERIMENT_013_LOCK_PATH = ap.LOCK_RELATIVE_PATH
EXPERIMENT_013_CONFIRMATION_PATH = ap.CONFIRMATION_RELATIVE_PATH
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260924
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
INHERITED_LEDGER_SCHEMA_VERSION = 1
PHASES = ("explore", "lock", "confirm", "report")

NEURON = 1987
NEURON_LAYER = 2
FIRING_THRESHOLD = 0.5  # Δa ≥ 0.5 on both sides
Y_SPEARMAN = 0.80
Y_R2 = 0.50
Y_BALANCED_ACCURACY = 0.90
Y3_R2_MAX = 0.30  # the axis-only alternative is rejected iff its R² is below this …
Y3_BALANCED_ACCURACY_MAX = 0.70  # … and its balanced accuracy below this
MIN_VALID_FRAMES = 4
MIN_VALID_FRAMES_PER_TOKEN = 3
MIN_SCORED_TOKENS = 16
FRAME_CUE_EFFECT_RATE = 108 / 120
LOCK_PREDICTION_TOLERANCE = 1e-9
REPLICATION_TOLERANCE = 1e-6
LOCKED_STATE_TOLERANCE = 1e-9
ACTIVATION_IDENTITY_TOLERANCE = 1e-6  # Δa from the captured residual against the neuron's term in the exact ledger, relative to the term's scale
EXPECTED_LEDGER_SIZE = 3732  # Experiment 013's 2724 exposed pairs + its 864 + 144 confirmed pairs
PREDICTION_COLUMNS = ("token", "frame_id", "template", "pre_ref", "dpre_hat", "da_hat", "fires_hat", "dpre_J", "dpre_J_E", "dpre_J_mlp1", "dpre_J_heads1", "dpre_J_axis", "dpre_J_offaxis",
                      "dpre_axis", "da_axis", "fires_axis", "term_hat")

CANDIDATES: dict[str, tuple[str, ...]] = {
    "determiner-like": ("second", "third", "former", "latter", "only", "main", "whole", "entire", "very"),
    "numeral": ("dozens", "hundreds", "thousands", "millions", "billions", "twice", "double", "triple", "couple", "pair"),
    "quantity": ("minimal", "infinite", "excess", "lesser", "considerable", "insufficient", "extensive"),
    "possessive-or-pronoun": ("anybody", "somebody", "everybody", "anyone", "whoever", "yourself", "themselves"),
    "adjective": ("narrow", "sharp", "smooth", "rough", "golden", "silver", "purple", "yellow", "tiny", "giant"),
}
QUOTAS = {"determiner-like": 5, "numeral": 5, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 5}
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The bakery bakes {cue}"),
    ("cardinal", "The library lends {cue}"),
    ("quantifier", "The journal publishes {cue}"),
    ("quantifier", "The committee approves {cue}"),
    ("coordinated-adjective", "Omar and Lucia piled {cue} neat"),
    ("coordinated-adjective", "Farah and Tobias lifted {cue} wet"),
)
FRAME_ID_TAG = "014"
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", "experiments/013-attention-paths-frozen-pattern/", "experiments/012-layer-correction-token-local/", "experiments/011-encoding-read-prospective/",
                            "experiments/010-read-direction-assembly/", "experiments/009-head-transport-rule/", "experiments/008-cue-suppression-localization/", "experiments/007-supervised-cue-subspace/",
                            "experiments/006-low-rank-cue-decompilation/", "experiments/005-regular-plural-mechanism/", "screening/behavior-candidates/manifest-v1.json")
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)
OUTCOME_Y1 = ("NEURON_FEATURE_PREDICTED_TOKENS", "NEURON_FEATURE_NOT_PREDICTED_TOKENS", "PRECONDITION_FAILED_TOKENS")
OUTCOME_Y2 = ("NEURON_FEATURE_PREDICTED_FRAMES_CONDITIONAL", "NEURON_FEATURE_NOT_PREDICTED_FRAMES_CONDITIONAL", "PRECONDITION_FAILED_FRAMES")
OUTCOME_Y3 = ("AXIS_ONLY_REJECTED", "AXIS_ONLY_NOT_REJECTED", "AXIS_ONLY_NOT_EVALUABLE")
CLASSIFICATION_NOT_EVALUABLE_SUFFIX = "_CLASSIFICATION_NOT_EVALUABLE"


class PhaseError(pm.PhaseError):
    pass


# ---------------------------------------------------------------------------
# The neuron and its exact computation.


@dataclass(frozen=True)
class NeuronWeights:
    """Block 2's LayerNorm and neuron 1987's own weights (float64), and the read of its output vector."""

    layer: int
    index: int
    ln_w: torch.Tensor
    ln_b: torch.Tensor
    eps: float
    w_in: torch.Tensor  # [d_model]
    b_in: float
    w_out: torch.Tensor  # [d_model]
    read_out: float  # r(W_out[j]) with Experiment 011's inner read

    @classmethod
    def from_layer_weights(cls, lw: lc.LayerWeights, read: lc.CorrectionRead, *, layer: int | None = None, index: int | None = None) -> "NeuronWeights":
        layer = NEURON_LAYER if layer is None else layer  # resolved at call time (the module constants are the frozen protocol; tests substitute a fake-sized neuron)
        index = NEURON if index is None else index
        w_out = lw.W_out[layer][index].clone()
        return cls(layer, index, lw.ln_w[layer], lw.ln_b[layer], lw.eps, lw.W_in[layer][:, index].clone(), float(lw.b_in[layer][index]), w_out, read.inner(w_out))

    @property
    def u(self) -> torch.Tensor:
        """γ₂ ⊙ W_in[:, j]: the effective residual-space input direction (up to the LayerNorm's centering and scale)."""
        return self.ln_w * self.w_in

    def pre(self, x: torch.Tensor) -> float:
        return float(pm.exact_layer_norm(x.double(), self.ln_w, self.ln_b, self.eps) @ self.w_in + self.b_in)

    def act(self, x: torch.Tensor) -> float:
        return gelu(self.pre(x))

    def jacobian_pre(self, x_ref: torch.Tensor, v: torch.Tensor) -> float:
        """⟨W_in, J_LN(x_ref) v⟩ = [⟨u, v_c⟩ − ⟨u, x̂⟩ · mean(x̂ ⊙ v_c)] / σ: centering, scale, and the variance derivative; population variance."""
        x = x_ref.double()
        mu = x.mean()
        sigma = torch.sqrt(((x - mu) ** 2).mean() + self.eps)
        x_hat = (x - mu) / sigma
        v_c = v.double() - v.double().mean()
        u = self.u
        return float((u @ v_c - (u @ x_hat) * (x_hat * v_c).mean()) / sigma)


def gelu(value: float) -> float:
    return float(torch.nn.functional.gelu(torch.tensor(float(value), dtype=torch.float64)))


def fires(delta_a: float) -> bool:
    return delta_a >= FIRING_THRESHOLD


# ---------------------------------------------------------------------------
# The predictor (weights + reference state + the frozen-pattern predicted arriving change; never a patched residual).


def arriving_change(fpm: ap.FrozenPatternModel, weights: pm.Weights, x1: torch.Tensor, pattern_weights: Mapping[str, float], token_id: int, template: str) -> dict[str, torch.Tensor]:
    """Experiment 013's Level-2 arriving change at block 2's input, by source: ΔE (weight-only), Δ̂₁ (block 1's MLP on ΔE at the reference), the layer-1 heads' frozen-pattern value paths."""
    delta_e = fpm.read.encoding_delta(weights, token_id, template)
    d1 = fpm.lw.delta_out(1, x1, delta_e)
    heads_1 = sum(fpm.value_path(key, x1, delta_e, pattern_weights[key]) for key in fpm.heads.keys_of(1))
    return {"E": delta_e, "mlp1": d1, "heads1": heads_1, "total": delta_e + d1 + heads_1}


def predict(neuron: NeuronWeights, fpm: ap.FrozenPatternModel, weights: pm.Weights, x1: torch.Tensor, x2: torch.Tensor, pattern_weights: Mapping[str, float], token_id: int, template: str, d_E: torch.Tensor) -> dict[str, Any]:
    """The exact-LayerNorm predictor, the Jacobian form with its source and axis splits, the axis-only alternative, and the predicted read term."""
    change = arriving_change(fpm, weights, x1, pattern_weights, token_id, template)
    v = change["total"]
    pre_ref = neuron.pre(x2)
    dpre_hat = neuron.pre(x2 + v) - pre_ref
    da_hat = gelu(pre_ref + dpre_hat) - gelu(pre_ref)
    axis = (v @ d_E) * d_E
    dpre_axis = neuron.pre(x2 + axis) - pre_ref
    da_axis = gelu(pre_ref + dpre_axis) - gelu(pre_ref)
    denominator = fpm.read.denominator(weights, template)
    return {"pre_ref": pre_ref, "dpre_hat": dpre_hat, "da_hat": da_hat, "fires_hat": fires(da_hat),
            "dpre_J": neuron.jacobian_pre(x2, v), "dpre_J_E": neuron.jacobian_pre(x2, change["E"]), "dpre_J_mlp1": neuron.jacobian_pre(x2, change["mlp1"]), "dpre_J_heads1": neuron.jacobian_pre(x2, change["heads1"]),
            "dpre_J_axis": neuron.jacobian_pre(x2, axis), "dpre_J_offaxis": neuron.jacobian_pre(x2, v - axis),
            "dpre_axis": dpre_axis, "da_axis": da_axis, "fires_axis": fires(da_axis), "term_hat": da_hat * neuron.read_out / denominator}


def predict_from_state(neuron: NeuronWeights, fpm: ap.FrozenPatternModel, weights: pm.Weights, state: ap.FrameState013, token_id: int, template: str, d_E: torch.Tensor) -> dict[str, Any]:
    return predict(neuron, fpm, weights, state.x1, state.x2, {key: state.pattern_weight(key) for key in ap.HEAD_KEYS}, token_id, template, d_E)


def predict_from_locked(neuron: NeuronWeights, fpm: ap.FrozenPatternModel, weights: pm.Weights, locked: Mapping[str, Any], token_id: int, template: str, d_E: torch.Tensor) -> dict[str, Any]:
    return predict(neuron, fpm, weights, torch.tensor(locked["x1"], dtype=torch.float64), torch.tensor(locked["x2"], dtype=torch.float64), dict(locked["A_pc"]), token_id, template, d_E)


def locked_state(state: ap.FrameState013, neuron: NeuronWeights) -> dict[str, Any]:
    return {**state.locked_state(), "pre_ref": neuron.pre(state.x2)}


def prediction_row(word: str, frame_id: str, template: str, prediction: Mapping[str, Any]) -> dict[str, Any]:
    row = {"token": word, "frame_id": frame_id, "template": template}
    row.update({key: prediction[key] for key in PREDICTION_COLUMNS[3:]})
    return row


# ---------------------------------------------------------------------------
# Measurement per pair (the patched run's residual before block 2; Experiment 013's fields for replication).


def analyse_pair_014(record: ra.Attribution, plural: ra.Attribution, *, neuron: NeuronWeights, fpm: ap.FrozenPatternModel, weights: pm.Weights, state: ap.FrameState013, axis_T: pm.SiteAxis, d_E: torch.Tensor) -> dict[str, Any] | None:
    base = lc.analyse_pair(record, plural, read=fpm.read, lw=fpm.lw, weights=weights, state=state, axis_T=axis_T)
    if base is None:
        return None
    frame = state.ref.frame
    template = frame.template_id
    x2_patch = record.extra[pm.site_label(("RESID_PRE.L2", frame.p_c))]
    pre_ref = neuron.pre(state.x2)
    dpre = neuron.pre(x2_patch) - pre_ref
    da = gelu(pre_ref + dpre) - gelu(pre_ref)
    denominator = fpm.read.denominator(weights, template)
    term = da * neuron.read_out / denominator
    # Identity: Δa computed through the neuron's own weights equals the neuron's entry of the exact block-2 ledger (the hidden-activation difference the neuron-sum identity is built on).
    ledger_da = float((fpm.lw.hidden(neuron.layer, x2_patch) - fpm.lw.hidden(neuron.layer, state.x2))[neuron.index])
    if abs(ledger_da - da) > ACTIVATION_IDENTITY_TOLERANCE * max(1.0, abs(da)):
        raise pm.IncidentError(f"{frame.frame_id}/{record.token}: the neuron's activation change disagrees with the block-2 ledger ({da:.6f} vs {ledger_da:.6f})")
    prediction = predict_from_state(neuron, fpm, weights, state, record.token_id, template, d_E)
    measured_change = x2_patch - state.x2
    return {"token": record.token, "token_id": record.token_id, "frame_id": frame.frame_id, "template": template,
            "pre_ref": pre_ref, "dpre": dpre, "da": da, "fires": fires(da), "term": term, "c_mlp2": base["c_k"]["L02.MLP"],
            "c_L": base["c_L"], "c_M": base["c_M"], "c_H": base["c_H"], "c_k": base["c_k"], "P1": base["P1"], "q_T": base["q_T"], "g_E": base["g_E"],
            "prediction": prediction,
            "remainders": {"pattern_change": dpre - prediction["dpre_hat"], "jacobian": prediction["dpre_hat"] - prediction["dpre_J"],
                           "measured_change_J": neuron.jacobian_pre(state.x2, measured_change), "ln_exact_at_measured": neuron.pre(x2_patch) - pre_ref},
            "identities": base["identities"], "ledger_da": ledger_da}


def measure_pair(model: Any, weights: pm.Weights, head: ht.HeadWeights, state: ap.FrameState013, name: str, token_id: int, e_axis: pm.SiteAxis, axis_T: pm.SiteAxis, nouns: Sequence[pm.Noun]) -> ra.Attribution:
    return ap.measure_pair_013(model, weights, head, state, name, token_id, e_axis, axis_T, nouns)


# ---------------------------------------------------------------------------
# Pool (135 tokens × 42 frames), the Experiment 013 pair ledger, the confirmation set.


def build_pool_014(manifest: ScreeningManifest, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009, confirmation_011: er.Confirmation011,
                   confirmation_012: lc.Confirmation012, confirmation_013: ap.Confirmation013) -> cs.Pool008:
    base = ap.build_pool_013(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012)
    frames = base.frames + tuple(confirmation_013.frames)
    origin = dict(base.frame_origin) | {frame.frame_id: "confirmation-013" for frame in confirmation_013.frames}
    tokens = list(base.tokens)
    category, source = dict(base.token_category), dict(base.token_source)
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation_013.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"Experiment 013 token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-013"
    return cs.Pool008(frames, origin, tuple(tokens), category, source, base.nouns, base.noun_source, base.reference_ids, base.plural_cue)


LEDGER_FIELDS = ("c_L", "c_M", "c_H")


def ledger_entry(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {"c_L": float(analysis["c_L"]), "c_M": float(analysis["c_M"]), "c_H": float(analysis["c_H"]), "c_k": {key: float(analysis["c_k"][key]) for key in ra.COMPONENT_ORDER}}


def inherited_ledger_payload(entries: Mapping[str, Mapping[str, Any]], *, source: Mapping[str, Any], digests: Mapping[str, str]) -> dict[str, Any]:
    payload = {"schema_version": INHERITED_LEDGER_SCHEMA_VERSION, "experiment": "013",
               "description": "Derived extract of Experiment 013's recorded per-pair c_L, c_M, c_H and c_k (Experiment 012's Y1 units) over its 2724 exposed pairs and its 864 + 144 confirmed pairs. Experiment 014 recomputes these and requires agreement within 1e-6.",
               "source": dict(source), "manifest_sha256": digests["manifest"], "extension_sha256": digests["extension"], "confirmation_006_sha256": digests["confirmation_006"], "confirmation_009_sha256": digests["confirmation_009"],
               "confirmation_011_sha256": digests["confirmation_011"], "confirmation_012_sha256": digests["confirmation_012"], "confirmation_013_sha256": digests["confirmation_013"], "lock_013_sha256": digests["lock_013"],
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "entries": {key: dict(value) for key, value in sorted(entries.items())}}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_ledger(path: Path, *, digests: Mapping[str, str], expected_size: int) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited 013 ledger: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "experiment", "description", "source", "manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256",
                                     "confirmation_012_sha256", "confirmation_013_sha256", "lock_013_sha256", "model", "entries", "content_sha256"}, "inherited 013 ledger")
    if payload["schema_version"] != INHERITED_LEDGER_SCHEMA_VERSION or payload["experiment"] != "013" or payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("inherited 013 ledger schema or digest is not frozen")
    recorded = tuple(payload[field] for field in ("manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "lock_013_sha256"))
    if recorded != tuple(digests[key] for key in ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "lock_013")):
        raise ValueError("inherited 013 ledger was recorded against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision} or len(payload["entries"]) != expected_size:
        raise ValueError("inherited 013 ledger model or size is not frozen")
    return payload


def check_ledger_replication(measured: Mapping[str, Mapping[str, Any]], recorded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = set(recorded) - set(measured)
    if missing:
        raise pm.IncidentError(f"recomputed ledger lacks {len(missing)} recorded pairs (e.g. {sorted(missing)[:3]})")
    worst_key, worst = None, 0.0
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
    if worst > REPLICATION_TOLERANCE:
        raise pm.IncidentError(f"{worst_key}: recomputed quantity deviates from Experiment 013's record by {worst:.3e}")
    return {"passed": True, "n": len(recorded), "max_abs_deviation": worst, "worst_key": worst_key}


def fresh_tokens_014(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
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


CONFIRMATION_DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013")
CONFIRMATION_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256")


def build_confirmation_payload(tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> dict[str, Any]:
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_014(tokenizer, excluded_ids)
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
               "policy": {"licensing": "every fresh cue in every fresh frame and in every exposed frame", "quotas": dict(QUOTAS), "expectations": "none: the committed neuron predictions are the only predictions; no semantic label is preregistered"},
               "tokens": tokens, "frames": frames, "token_prompts": [_prompt_entry(tokenizer, frame, token) for frame in frame_objects for token in tokens],
               "exposed_frame_prompts": [_prompt_entry(tokenizer, frame, token) for frame in pool.frames for token in tokens],
               "construction": "tokenizer-only; first eligible candidates per lexical class; classes carry no expectation; no model output"}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


Confirmation014 = ap.Confirmation013  # the same structure: fresh frames, exposed frame ids, tokens, two prompt lists


def validate_confirmation(payload: Mapping[str, Any], pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation014:
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
    return Confirmation014(dict(payload["reference_cue_ids"]), frames, tuple(payload["exposed_frame_ids"]), tuple(tokens), tuple(fresh_prompts), tuple(exposed_prompts), payload["content_sha256"])


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


def load_confirmation(path: Path, pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation014:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, pool, digests)


# ---------------------------------------------------------------------------
# Results state, phases, statistics.

DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014", "lock_011", "lock_012", "lock_013", "ledger_013")
STATE_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "confirmation_014_sha256",
                       "lock_011_sha256", "lock_012_sha256", "lock_013_sha256", "ledger_013_sha256")
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


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation014) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


comparison = lc.comparison
explained_variance = lc.explained_variance


def balanced_accuracy(predicted: Sequence[bool], measured: Sequence[bool]) -> dict[str, Any]:
    """½(sensitivity + specificity) of the predicted firing against the measured firing; None when a measured class is absent (the guard)."""
    tp = sum(1 for p, m in zip(predicted, measured) if p and m)
    fn = sum(1 for p, m in zip(predicted, measured) if not p and m)
    tn = sum(1 for p, m in zip(predicted, measured) if not p and not m)
    fp = sum(1 for p, m in zip(predicted, measured) if p and not m)
    n = tp + fn + tn + fp
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"n": n, "measured_firing": tp + fn, "measured_not_firing": tn + fp, "predicted_firing": tp + fp, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": sensitivity, "specificity": specificity, "balanced_accuracy": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
            "raw_accuracy": (tp + tn) / n if n else None, "majority_baseline": max(tp + fn, tn + fp) / n if n else None, "evaluable": sensitivity is not None and specificity is not None}


# ---------------------------------------------------------------------------
# Exploration (Tier A).


def _token_means(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not analyses:
        return {}
    mean = lambda values: pm._mean(list(values))  # noqa: E731
    return {"n_frames": len(analyses), "frames": sorted(a["frame_id"] for a in analyses),
            "da_mean": mean(a["da"] for a in analyses), "dpre_mean": mean(a["dpre"] for a in analyses), "term_mean": mean(a["term"] for a in analyses), "c_mlp2_mean": mean(a["c_mlp2"] for a in analyses),
            "pre_ref_mean": mean(a["pre_ref"] for a in analyses), "fires_fraction": mean(1.0 if a["fires"] else 0.0 for a in analyses),
            "da_hat_mean": mean(p["da_hat"] for p in predictions), "dpre_hat_mean": mean(p["dpre_hat"] for p in predictions), "dpre_J_mean": mean(p["dpre_J"] for p in predictions),
            "da_axis_mean": mean(p["da_axis"] for p in predictions), "fires_hat_fraction": mean(1.0 if p["fires_hat"] else 0.0 for p in predictions),
            "J_E_mean": mean(p["dpre_J_E"] for p in predictions), "J_mlp1_mean": mean(p["dpre_J_mlp1"] for p in predictions), "J_heads1_mean": mean(p["dpre_J_heads1"] for p in predictions),
            "J_axis_mean": mean(p["dpre_J_axis"] for p in predictions), "J_offaxis_mean": mean(p["dpre_J_offaxis"] for p in predictions), "term_hat_mean": mean(p["term_hat"] for p in predictions)}


def statistics_for(pairs: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, Any]:
    """Token-mean and pair-level statistics of the predictor, its Jacobian form, and the axis-only alternative, plus the accounting."""
    by_token = {name: [(a, p) for a, p in zip(pairs, predictions) if a["token"] == name] for name in tokens}
    means = {name: _token_means([a for a, _ in rows], [p for _, p in rows]) for name, rows in by_token.items() if rows}
    names = sorted(means)
    m = lambda key: [means[n][key] for n in names]  # noqa: E731
    firing_pairs = [(a, p) for a, p in zip(pairs, predictions) if a["fires"]]
    return {"n_tokens": len(names), "n_pairs": len(pairs),
            "token_means": {"da_hat_vs_da": comparison(m("da_hat_mean"), m("da_mean")), "da_axis_vs_da": comparison(m("da_axis_mean"), m("da_mean")),
                            "dpre_hat_vs_dpre": comparison(m("dpre_hat_mean"), m("dpre_mean")), "dpre_J_vs_dpre": comparison(m("dpre_J_mean"), m("dpre_mean")), "da_spread": ap.spread(m("da_mean"))},
            "pairs": {"da_hat_vs_da": comparison([p["da_hat"] for p in predictions], [a["da"] for a in pairs]), "da_axis_vs_da": comparison([p["da_axis"] for p in predictions], [a["da"] for a in pairs]),
                      "dpre_hat_vs_dpre": comparison([p["dpre_hat"] for p in predictions], [a["dpre"] for a in pairs]), "dpre_J_vs_dpre": comparison([p["dpre_J"] for p in predictions], [a["dpre"] for a in pairs]),
                      "dpre_J_vs_dpre_hat": comparison([p["dpre_J"] for p in predictions], [p["dpre_hat"] for p in predictions]),
                      "amplitude_error_firing": pm._mean([abs(p["da_hat"] - a["da"]) for a, p in firing_pairs]) if firing_pairs else None, "mean_da_firing": pm._mean([a["da"] for a, _ in firing_pairs]) if firing_pairs else None,
                      "classification": balanced_accuracy([p["fires_hat"] for p in predictions], [a["fires"] for a in pairs]), "classification_axis": balanced_accuracy([p["fires_axis"] for p in predictions], [a["fires"] for a in pairs])},
            "accounting": {"J_E_abs": pm._mean([abs(p["dpre_J_E"]) for p in predictions]), "J_mlp1_abs": pm._mean([abs(p["dpre_J_mlp1"]) for p in predictions]), "J_heads1_abs": pm._mean([abs(p["dpre_J_heads1"]) for p in predictions]),
                           "J_axis_abs": pm._mean([abs(p["dpre_J_axis"]) for p in predictions]), "J_offaxis_abs": pm._mean([abs(p["dpre_J_offaxis"]) for p in predictions]),
                           "J_heads1_negative_fraction": pm._mean([1.0 if p["dpre_J_heads1"] < 0 else 0.0 for p in predictions]),
                           "pattern_change_remainder_abs": pm._mean([abs(a["remainders"]["pattern_change"]) for a in pairs]), "jacobian_remainder_abs": pm._mean([abs(a["remainders"]["jacobian"]) for a in pairs]),
                           "pre_ref_min": min(a["pre_ref"] for a in pairs) if pairs else None, "pre_ref_max": max(a["pre_ref"] for a in pairs) if pairs else None,
                           "reference_activation_max": max(gelu(a["pre_ref"]) for a in pairs) if pairs else None,
                           "term_share_of_mlp2_firing": pm._mean([a["term"] / a["c_mlp2"] for a, _ in firing_pairs if abs(a["c_mlp2"]) > 0.05]) if firing_pairs else None,
                           "term_mean_firing": pm._mean([a["term"] for a, _ in firing_pairs]) if firing_pairs else None,
                           "term_mean_not_firing": pm._mean([a["term"] for a in pairs if not a["fires"]]) if any(not a["fires"] for a in pairs) else None},
            "token_means_table": means}


def firing_set(means: Mapping[str, Mapping[str, Any]], categories: Mapping[str, str], *, key: str = "fires_fraction") -> dict[str, list[str]]:
    """Tokens that fire in at least half of their frames, by lexical class (descriptive; the frozen threshold applied per pair)."""
    out: dict[str, list[str]] = {}
    for name, entry in sorted(means.items()):
        if entry[key] >= 0.5:
            out.setdefault(categories.get(name, "?"), []).append(name)
    return out


def run_exploration(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, *, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_013: Mapping[str, Any], inherited: Mapping[str, Any], state: dict[str, Any], results_path: Path | None, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    nouns = pool.single_nouns
    exploration = state["exploration"]
    say("stage axes from the Experiment 010 pool, checked against the Experiment 011 lock")
    axis_T, e_axis = lc.locked_axes(lock_011, cs.stage_axes(cache, weights, pool_010))
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    fpm = ap.model_from_locks(lock_011, lock_012, lw, heads, pool.reference_ids, plural_ids)
    exploration["read_weight_check"] = lc.check_read_weight(fpm.read, head, axis_T)
    neuron = NeuronWeights.from_layer_weights(lw, fpm.read)
    d_E = e_axis.direction.double()
    exploration["axes_vectors"] = {"T": axis_T.direction.tolist(), "R0": d_E.tolist()}
    exploration["sigma_T"] = axis_T.sigma
    exploration["read_weight"] = fpm.read.weight.tolist()
    exploration["defined_templates"] = list(lock_012["defined_templates"])
    exploration["neuron"] = {"layer": NEURON_LAYER, "index": NEURON, "b_in": neuron.b_in, "read_out": neuron.read_out, "cos_u_dE": pm.cosine(neuron.u, d_E), "u_norm": float(neuron.u.norm()), "w_out_norm": float(neuron.w_out.norm()),
                             "read_out_over_denominators": {template: neuron.read_out / fpm.read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}}
    recorded = inherited["entries"]
    say(f"reference states of the {len(pool.frames)} exposed frames; re-measurement of the {len(recorded)} recorded pairs with the residual before block 2 captured")
    locked_states: dict[str, Any] = {}
    pairs: dict[str, dict[str, Any]] = {}
    ledger: dict[str, dict[str, Any]] = {}
    identities: dict[str, float] = {}
    for frame in pool.frames:
        state_f = ap.capture_frame_013(model, head, pool.reference_prompt(frame), nouns, axis_T)
        locked = locked_state(state_f, neuron)
        if frame.frame_id in lock_013.get("locked_states", {}):
            previous = lock_013["locked_states"][frame.frame_id]
            drift = max(float((state_f.x1 - torch.tensor(previous["x1"], dtype=torch.float64)).abs().max()), float((state_f.x2 - torch.tensor(previous["x2"], dtype=torch.float64)).abs().max()))
            if drift > LOCKED_STATE_TOLERANCE:
                raise pm.IncidentError(f"{frame.frame_id}: the reference state differs from the Experiment 013 lock (max {drift:.2e})")
        locked_states[frame.frame_id] = locked
        names = [name for name, _ in pool.tokens if f"{name}|{frame.frame_id}" in recorded]
        if not names:
            continue
        plural_name = pool.plural_cue[frame.template_id]
        records = {name: measure_pair(model, weights, head, state_f, name, pool.token_id(name), e_axis, axis_T, nouns) for name in names}
        plural = records[plural_name] if plural_name in records else measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for name, record in records.items():
            analysis = analyse_pair_014(record, plural, neuron=neuron, fpm=fpm, weights=weights, state=state_f, axis_T=axis_T, d_E=d_E)
            key = f"{name}|{frame.frame_id}"
            if analysis is None:
                raise pm.IncidentError(f"{key}: no analysis although Experiment 013 recorded the pair")
            er._max_errors(identities, analysis["identities"])
            pairs[key] = analysis
            ledger[key] = ledger_entry(analysis)
        say(f"  {frame.frame_id}: {len(names)} pairs")
    exploration["identities"] = identities
    exploration["locked_states"] = locked_states
    exploration["locked_state_digests"] = {frame_id: ap.state_digest(entry) for frame_id, entry in locked_states.items()}
    exploration["replication"] = {"experiment_013": check_ledger_replication(ledger, recorded)}
    say(f"replication against Experiment 013: max deviation {exploration['replication']['experiment_013']['max_abs_deviation']:.2e}")
    exploration["pairs"] = pairs
    analyses = list(pairs.values())
    stats = statistics_for(analyses, [a["prediction"] for a in analyses], [name for name, _ in pool.tokens])
    exploration["exposed_check"] = stats
    exploration["firing_set"] = firing_set(stats["token_means_table"], pool.token_category)
    exploration["predicted_firing_set"] = firing_set(stats["token_means_table"], pool.token_category, key="fires_hat_fraction")
    tm, pr = stats["token_means"]["da_hat_vs_da"], stats["pairs"]
    exploration["summary"] = {"defined_templates": exploration["defined_templates"], "spearman": tm["spearman"], "r2": tm["r2"], "balanced_accuracy": pr["classification"]["balanced_accuracy"],
                              "axis_r2": stats["token_means"]["da_axis_vs_da"]["r2"], "axis_balanced_accuracy": pr["classification_axis"]["balanced_accuracy"], "prediction_undefined": len(exploration["defined_templates"]) < 2}
    f = cd._f
    say(f"exposed: Δâ vs Δa token means Spearman {f(tm['spearman'], 3)}, R² {f(tm['r2'], 3)}; balanced accuracy {f(pr['classification']['balanced_accuracy'], 3)}; axis-only R² {f(exploration['summary']['axis_r2'], 3)}, balanced accuracy {f(exploration['summary']['axis_balanced_accuracy'], 3)}")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


# ---------------------------------------------------------------------------
# Lock and its artifacts.


def prediction_table(neuron: NeuronWeights, fpm: ap.FrozenPatternModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], frames: Sequence[pm.Frame], tokens: Sequence[Mapping[str, Any]], defined: Sequence[str], d_E: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        if frame.template_id not in defined:
            continue
        for token in tokens:
            rows.append(prediction_row(token["word"], frame.frame_id, frame.template_id, predict_from_locked(neuron, fpm, weights, locked_states[frame.frame_id], token["token_id"], frame.template_id, d_E)))
    return rows


def token_means_from_table(rows: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for word in tokens:
        mine = [row for row in rows if row["token"] == word]
        if mine:
            out[word] = {key: pm._mean([row[key] for row in mine]) for key in ("pre_ref", "dpre_hat", "da_hat", "dpre_J", "dpre_axis", "da_axis", "term_hat", "dpre_J_E", "dpre_J_mlp1", "dpre_J_heads1", "dpre_J_axis", "dpre_J_offaxis")}
            out[word]["n_frames"] = len(mine)
            out[word]["fires_hat_fraction"] = pm._mean([1.0 if row["fires_hat"] else 0.0 for row in mine])
            out[word]["fires_axis_fraction"] = pm._mean([1.0 if row["fires_axis"] else 0.0 for row in mine])
    return out


def lock_predictions(neuron: NeuronWeights, fpm: ap.FrozenPatternModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], pool: cs.Pool008, confirmation: Confirmation014, defined: Sequence[str], d_E: torch.Tensor) -> dict[str, Any]:
    rows = prediction_table(neuron, fpm, weights, locked_states, pool.frames, confirmation.tokens, defined, d_E)
    words = [token["word"] for token in confirmation.tokens]
    return {"rows": rows, "token_means": token_means_from_table(rows, words),
            "predicted_firing_counts": {"pairs_firing": sum(1 for row in rows if row["fires_hat"]), "pairs_total": len(rows), "axis_pairs_firing": sum(1 for row in rows if row["fires_axis"])}}


def render_predictions(lock: Mapping[str, Any]) -> str:
    f = cd._f
    counts = lock["predictions"]["predicted_firing_counts"]
    lines = ["# Experiment 014 — preregistered predictions (block-2 neuron 1987; exact LayerNorm and input weights at the frozen-pattern predicted state)", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_014_sha256']}`; Experiment 013 lock sha256 `{lock['lock_013_sha256']}`",
             f"- Floors: Y1/Y2 Spearman ≥ {Y_SPEARMAN}, R² ≥ {Y_R2} (token means), balanced firing accuracy ≥ {Y_BALANCED_ACCURACY} (pairs, threshold Δa ≥ {FIRING_THRESHOLD}); Y3 axis-only rejected iff R² < {Y3_R2_MAX} and balanced accuracy < {Y3_BALANCED_ACCURACY_MAX}",
             f"- Y1 table: {counts['pairs_total']} rows (fresh tokens × exposed frames); predicted to fire in {counts['pairs_firing']} pairs (axis-only alternative: {counts['axis_pairs_firing']}); Y2 rows are computed at confirm stage 1 and digested before any fresh cue prompt",
             f"- Neuron: cos(γ₂ ⊙ W_in, d̂_E) {f(lock['neuron']['cos_u_dE'], 3)}; r(W_out) / r(E(pl) − E(ref)) per template {{{', '.join(f'{t}: {f(v, 3)}' for t, v in lock['neuron']['read_out_over_denominators'].items())}}}", "",
             "| token | class | frames | pre_ref | **predicted Δâ** | fires (fraction of frames) | Jacobian Δp̂re | axis part | off-axis part | E | MLP₁ | heads₁ | axis-only Δâ | predicted term |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    categories = {token["word"]: token["category"] for token in lock["tokens"]}
    for word, m in sorted(lock["predictions"]["token_means"].items(), key=lambda kv: -kv[1]["da_hat"]):
        lines.append(f"| {word} | {categories.get(word, '')} | {m['n_frames']} | {f(m['pre_ref'], 2)} | **{f(m['da_hat'], 3)}** | {f(m['fires_hat_fraction'], 2)} | {f(m['dpre_J'], 2)} | {f(m['dpre_J_axis'], 2)} | {f(m['dpre_J_offaxis'], 2)} | {f(m['dpre_J_E'], 2)} | {f(m['dpre_J_mlp1'], 2)} | {f(m['dpre_J_heads1'], 2)} | {f(m['da_axis'], 3)} | {f(m['term_hat'], 3)} |")
    lines.append("")
    return "\n".join(lines)


def frozen_floors() -> dict[str, Any]:
    return {"y_spearman": Y_SPEARMAN, "y_r2": Y_R2, "y_balanced_accuracy": Y_BALANCED_ACCURACY, "y3_r2_max": Y3_R2_MAX, "y3_balanced_accuracy_max": Y3_BALANCED_ACCURACY_MAX, "firing_threshold": FIRING_THRESHOLD,
            "min_valid_frames": MIN_VALID_FRAMES, "min_valid_frames_per_token": MIN_VALID_FRAMES_PER_TOKEN, "min_scored_tokens": MIN_SCORED_TOKENS, "frame_cue_effect_rate": FRAME_CUE_EFFECT_RATE, "head_stage_floor": cs.STAGE_UNINFORMATIVE_FLOOR}


def build_candidate_lock(*, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation014, predictions: Mapping[str, Any], protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    if exploration["summary"]["prediction_undefined"]:
        raise PhaseError("fewer than two templates are defined: PREDICTION_UNDEFINED, no lock")
    lock = {"schema_version": 1, "experiment": "014", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit}
    for field, key in zip(STATE_DIGEST_FIELDS, DIGEST_KEYS):
        lock[field] = digests[key] if key != "confirmation_014" else confirmation.content_sha256
    lock.update({"confirmation_prompt_keys": sorted(prompt.key for prompt in confirmation.all_prompts), "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
                 "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED}, "head": ht.HEAD_KEY, "neuron": exploration["neuron"],
                 "axes_vectors": exploration["axes_vectors"], "sigma_T": exploration["sigma_T"], "read_weight": exploration["read_weight"], "defined_templates": exploration["defined_templates"],
                 "locked_states": exploration["locked_states"], "locked_state_digests": exploration["locked_state_digests"], "floors": frozen_floors(),
                 "tokens": [dict(token) for token in confirmation.tokens], "exposed_check": exploration["exposed_check"], "exposed_firing_set": exploration["firing_set"],
                 "predictions": dict(predictions), "tier_a_results_sha256": state.get("state_sha256")})
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation014, predictions_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "014" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked or git_state.get("dirty", True):
        raise PhaseError("confirm requires the committed lock and predictions on a clean Git tree")
    expected = dict(zip(STATE_DIGEST_FIELDS, digest_tuple(digests)))
    expected["confirmation_014_sha256"] = confirmation.content_sha256
    if any(lock.get(field) != value for field, value in expected.items()):
        raise PhaseError("lock was built against different frozen inputs")
    if not state.get("lock") or lock["content_sha256"] != state["lock"]["content_sha256"] or lock["run_id"] != state["run_id"]:
        raise PhaseError("the installed lock is not the candidate lock written by the lock phase")
    if pm.sha256_text(predictions_text) != state["lock"]["predictions_sha256"]:
        raise PhaseError("the installed predictions.md is not the artifact written by the lock phase")
    if dict(lock["floors"]) != frozen_floors() or lock["neuron"]["index"] != NEURON or lock["neuron"]["layer"] != NEURON_LAYER:
        raise PhaseError("the lock's floors or neuron are not the frozen constants")
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
            if isinstance(a[key], bool):
                if a[key] != b[key]:
                    raise PhaseError(f"the recomputed firing prediction differs for {a['token']}/{a['frame_id']}; nothing was executed")
            else:
                worst = max(worst, abs(a[key] - b[key]))
    if worst > LOCK_PREDICTION_TOLERANCE:
        raise PhaseError(f"the weights, locked axes, frozen models and locked reference states do not reproduce the preregistered predictions (max difference {worst:.3e}); nothing was executed")
    return worst


# ---------------------------------------------------------------------------
# Confirmation: stage 1 (fresh frames' reference states, digested predictions), stage 2 (fresh cues), scoring.


def _neuron_and_model(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, lock: Mapping[str, Any], lock_012: Mapping[str, Any]):
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    axis_T, e_axis = lc.locked_axes(lock, cs.stage_axes(cache, weights, pool_010))
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    fpm = ap.model_from_locks(lock, lock_012, lw, heads, pool.reference_ids, plural_ids)
    lc.check_read_weight(fpm.read, head, axis_T)
    neuron = NeuronWeights.from_layer_weights(lw, fpm.read)
    return weights, head, cache, axis_T, e_axis, fpm, neuron


def stage_one(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation014, lock: Mapping[str, Any], lock_012: Mapping[str, Any], *, protocol_code_commit: str, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights, head, cache, axis_T, e_axis, fpm, neuron = _neuron_and_model(model, pool, pool_010, lock, lock_012)
    d_E = e_axis.direction.double()
    nouns = pool.single_nouns
    say("stage 1: fresh frames' reference states, validity, prediction table")
    frames_out: dict[str, Any] = {}
    states: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    identities: dict[str, float] = {}
    for frame in confirmation.frames:
        template = frame.template_id
        if template not in lock["defined_templates"]:
            frames_out[frame.frame_id] = {"template_id": template, "valid": False, "template_defined": False, "note": "template excluded before the lock; nothing run"}
            continue
        state_f = ap.capture_frame_013(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        sg, pl = pm.frame_prompts(frame)
        c_a, c_b = cache.c(sg), cache.c(pl)
        positive = sum(1 for noun in nouns if c_a[noun.lexical_key] - c_b[noun.lexical_key] > 0)
        required = pm.exact_count_floor(FRAME_CUE_EFFECT_RATE, len(nouns))
        plural = measure_pair(model, weights, head, state_f, pool.plural_cue[template], pl.cue_token_id, e_axis, axis_T, nouns)
        er._max_errors(identities, er.enforce_identities(plural, plural, f"{frame.frame_id}/{pool.plural_cue[template]}"))
        head_informative = abs(plural.head_change) >= cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma
        valid = head_informative and positive >= required
        frames_out[frame.frame_id] = {"template_id": template, "valid": valid, "head_informative": head_informative, "plural_head_change": plural.head_change, "cue_effect_positive": positive, "cue_effect_required": required,
                                      "template_defined": True, "reconstruction_error": state_f.ref.reconstruction_error, "pre_ref": neuron.pre(state_f.x2)}
        states[frame.frame_id] = locked_state(state_f, neuron)
        for token in confirmation.tokens:
            rows.append(prediction_row(token["word"], frame.frame_id, template, predict_from_state(neuron, fpm, weights, state_f, token["token_id"], template, d_E)))
        say(f"  {frame.frame_id}: {'valid' if valid else 'INVALID'} (head informative {head_informative}, cue effect {positive}/{required}); pre_ref {neuron.pre(state_f.x2):.2f}; {len(confirmation.tokens)} predictions")
    return {"frames": frames_out, "states": states, "state_digests": {frame_id: ap.state_digest(entry) for frame_id, entry in states.items()}, "rows": rows, "identities": identities,
            "token_means": token_means_from_table(rows, [token["word"] for token in confirmation.tokens]), "predicted_firing_counts": {"pairs_firing": sum(1 for row in rows if row["fires_hat"]), "pairs_total": len(rows)},
            "digest": ap.table_digest(rows, states), "commit": protocol_code_commit, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}


def stage_two(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation014, lock: Mapping[str, Any], lock_012: Mapping[str, Any], stage1: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    ap.assert_stage_one_digest(stage1)
    weights, head, cache, axis_T, e_axis, fpm, neuron = _neuron_and_model(model, pool, pool_010, lock, lock_012)
    d_E = e_axis.direction.double()
    nouns = pool.single_nouns
    identities: dict[str, float] = {}
    say("stage 2: fresh cues in the exposed frames (Y1) and in the valid fresh frames (Y2)")
    pairs_exposed: dict[str, dict[str, Any]] = {}
    for frame in pool.frames:
        if frame.template_id not in lock["defined_templates"]:
            continue
        state_f = ap.capture_frame_013(model, head, pool.reference_prompt(frame), nouns, axis_T)
        locked = lock["locked_states"][frame.frame_id]
        drift = max(float((state_f.x1 - torch.tensor(locked["x1"], dtype=torch.float64)).abs().max()), float((state_f.x2 - torch.tensor(locked["x2"], dtype=torch.float64)).abs().max()),
                    max(abs(state_f.pattern_weight(key) - locked["A_pc"][key]) for key in ap.HEAD_KEYS), abs(neuron.pre(state_f.x2) - locked["pre_ref"]))
        if drift > LOCKED_STATE_TOLERANCE:
            raise pm.IncidentError(f"{frame.frame_id}: the re-captured reference state differs from the locked one (max {drift:.2e})")
        plural_name = pool.plural_cue[frame.template_id]
        plural = measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_014(record, plural, neuron=neuron, fpm=fpm, weights=weights, state=state_f, axis_T=axis_T, d_E=d_E)
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
        state_f = ap.capture_frame_013(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        if ap.state_digest(locked_state(state_f, neuron)) != stage1["state_digests"][frame.frame_id]:
            raise pm.IncidentError(f"{frame.frame_id}: the re-captured reference state differs from stage 1's digested state")
        sg, pl = pm.frame_prompts(frame)
        plural = measure_pair(model, weights, head, state_f, pool.plural_cue[frame.template_id], pl.cue_token_id, e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_014(record, plural, neuron=neuron, fpm=fpm, weights=weights, state=state_f, axis_T=axis_T, d_E=d_E)
            if analysis is None:
                raise pm.IncidentError(f"{frame.frame_id}/{token['word']}: no analysis although the frame is valid")
            er._max_errors(identities, analysis["identities"])
            pairs_fresh[f"{token['word']}|{frame.frame_id}"] = analysis
        say(f"  {frame.frame_id} (fresh): {len(confirmation.tokens)} tokens")
    results = score_confirmation(stage1, pairs_exposed, pairs_fresh, confirmation, lock, pool.token_category | {token["word"]: token["category"] for token in confirmation.tokens})
    er._max_errors(identities, stage1.get("identities", {}))
    results["identities"] = identities
    return results


def _rows_by_pair(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {f"{row['token']}|{row['frame_id']}": row for row in rows}


def _score_set(pairs: Mapping[str, Mapping[str, Any]], table: Mapping[str, Mapping[str, Any]], words: Sequence[str], *, min_frames: int) -> dict[str, Any]:
    """Token means of Δâ (from the table) and Δa over identical frame sets, plus the pair-level classification; the three floors."""
    tokens_out, scored_pairs = {}, []
    for word in words:
        keys = sorted(key for key in pairs if key.split("|")[0] == word and key in table)
        analyses = [pairs[key] for key in keys]
        predictions = [table[key] for key in keys]
        tokens_out[word] = {"scored": len(keys) >= min_frames, "n_valid_frames": len(keys), **_token_means(analyses, predictions)}
        if len(keys) >= min_frames:
            scored_pairs.extend(zip(analyses, predictions))
    scored = [word for word, entry in tokens_out.items() if entry["scored"]]
    result: dict[str, Any] = {"tokens": tokens_out, "scored_tokens": scored, "n_pairs": len(scored_pairs)}
    if len(scored) >= 2:
        da_hat = [tokens_out[w]["da_hat_mean"] for w in scored]
        da = [tokens_out[w]["da_mean"] for w in scored]
        c = comparison(da_hat, da)
        classification = balanced_accuracy([p["fires_hat"] for _, p in scored_pairs], [a["fires"] for a, _ in scored_pairs])
        failing = [name for name, ok in (("spearman", c["spearman"] is not None and c["spearman"] >= Y_SPEARMAN), ("r2", c["r2"] is not None and c["r2"] >= Y_R2)) if not ok]
        if classification["evaluable"] and classification["balanced_accuracy"] < Y_BALANCED_ACCURACY:
            failing.append("balanced_accuracy")
        result["test"] = {**c, "classification": classification, "classification_evaluable": classification["evaluable"], "passed": not failing, "failing": failing,
                          "floors": {"spearman": Y_SPEARMAN, "r2": Y_R2, "balanced_accuracy": Y_BALANCED_ACCURACY}}
        axis = comparison([tokens_out[w]["da_axis_mean"] for w in scored], da)
        result["axis"] = {**axis, "classification": balanced_accuracy([p["fires_axis"] for _, p in scored_pairs], [a["fires"] for a, _ in scored_pairs])}
        firing_pairs = [(a, p) for a, p in scored_pairs if a["fires"]]
        result["descriptive"] = {"pairs_da_hat_vs_da": comparison([p["da_hat"] for _, p in scored_pairs], [a["da"] for a, _ in scored_pairs]),
                                 "pairs_dpre_hat_vs_dpre": comparison([p["dpre_hat"] for _, p in scored_pairs], [a["dpre"] for a, _ in scored_pairs]),
                                 "pairs_dpre_J_vs_dpre": comparison([p["dpre_J"] for _, p in scored_pairs], [a["dpre"] for a, _ in scored_pairs]),
                                 "token_means_dpre_J_vs_dpre": comparison([tokens_out[w]["dpre_J_mean"] for w in scored], [tokens_out[w]["dpre_mean"] for w in scored]),
                                 "mae_da": pm._mean([abs(p["da_hat"] - a["da"]) for a, p in scored_pairs]), "amplitude_error_firing": pm._mean([abs(p["da_hat"] - a["da"]) for a, p in firing_pairs]) if firing_pairs else None,
                                 "mean_da_firing": pm._mean([a["da"] for a, _ in firing_pairs]) if firing_pairs else None, "da_spread": ap.spread(da),
                                 "pattern_change_remainder_abs": pm._mean([abs(a["remainders"]["pattern_change"]) for a, _ in scored_pairs]), "jacobian_remainder_abs": pm._mean([abs(a["remainders"]["jacobian"]) for a, _ in scored_pairs]),
                                 "term_share_of_mlp2_firing": pm._mean([a["term"] / a["c_mlp2"] for a, _ in firing_pairs if abs(a["c_mlp2"]) > 0.05]) if firing_pairs else None,
                                 "accounting_abs": {"E": pm._mean([abs(p["dpre_J_E"]) for _, p in scored_pairs]), "mlp1": pm._mean([abs(p["dpre_J_mlp1"]) for _, p in scored_pairs]), "heads1": pm._mean([abs(p["dpre_J_heads1"]) for _, p in scored_pairs]),
                                                    "axis": pm._mean([abs(p["dpre_J_axis"]) for _, p in scored_pairs]), "offaxis": pm._mean([abs(p["dpre_J_offaxis"]) for _, p in scored_pairs])}}
    return result


def score_confirmation(stage1: Mapping[str, Any], pairs_exposed: Mapping[str, Mapping[str, Any]], pairs_fresh: Mapping[str, Mapping[str, Any]], confirmation: Confirmation014, lock: Mapping[str, Any], categories: Mapping[str, str]) -> dict[str, Any]:
    words = [token["word"] for token in confirmation.tokens]
    locked_table = _rows_by_pair(lock["predictions"]["rows"])
    stage1_table = _rows_by_pair(stage1["rows"])
    valid_frames = [frame_id for frame_id, entry in stage1["frames"].items() if entry.get("valid")]
    y1 = _score_set(pairs_exposed, locked_table, words, min_frames=MIN_VALID_FRAMES_PER_TOKEN)
    y2 = _score_set(pairs_fresh, stage1_table, words, min_frames=MIN_VALID_FRAMES_PER_TOKEN)
    results: dict[str, Any] = {"frames": stage1["frames"], "valid_frames": valid_frames, "Y1": y1, "Y2": y2, "per_frame_exposed": pairs_exposed, "per_frame_fresh": pairs_fresh}
    results["precondition_Y1"] = {"passed": len(y1["scored_tokens"]) >= MIN_SCORED_TOKENS, "scored_tokens": len(y1["scored_tokens"]), "min_scored_tokens": MIN_SCORED_TOKENS}
    results["precondition_Y2"] = {"passed": len(valid_frames) >= MIN_VALID_FRAMES and len(y2["scored_tokens"]) >= MIN_SCORED_TOKENS, "valid_frames": len(valid_frames), "scored_tokens": len(y2["scored_tokens"]),
                                  "min_valid_frames": MIN_VALID_FRAMES, "min_scored_tokens": MIN_SCORED_TOKENS}
    # Y3: the axis-only alternative over the token means and the pairs of both sets (scored tokens only).
    means_axis, means_da, pair_items = [], [], []
    for y, pairs, table in ((y1, pairs_exposed, locked_table), (y2, pairs_fresh, stage1_table)):
        for w in y["scored_tokens"]:
            means_axis.append(y["tokens"][w]["da_axis_mean"])
            means_da.append(y["tokens"][w]["da_mean"])
            pair_items.extend((pairs[k], table[k]) for k in sorted(key for key in pairs if key.split("|")[0] == w and key in table))
    if len(means_da) >= 2:
        axis = comparison(means_axis, means_da)
        classification = balanced_accuracy([p["fires_axis"] for _, p in pair_items], [a["fires"] for a, _ in pair_items])
        evaluable = classification["evaluable"] and axis["r2"] is not None
        rejected = evaluable and axis["r2"] < Y3_R2_MAX and classification["balanced_accuracy"] < Y3_BALANCED_ACCURACY_MAX
        results["Y3"] = {**axis, "classification": classification, "evaluable": evaluable, "rejected": rejected, "n_token_means": len(means_da), "n_pairs": len(pair_items),
                         "floors": {"r2_max": Y3_R2_MAX, "balanced_accuracy_max": Y3_BALANCED_ACCURACY_MAX}, "full_predictor_on_same_data": comparison([y["tokens"][w]["da_hat_mean"] for y in (y1, y2) for w in y["scored_tokens"]], means_da)}
    else:
        results["Y3"] = {"evaluable": False, "rejected": False, "n_token_means": len(means_da), "n_pairs": len(pair_items)}
    results["firing_sets"] = {label: {"measured": firing_set(y["tokens"], categories) if y["scored_tokens"] else {}, "predicted": firing_set(y["tokens"], categories, key="fires_hat_fraction") if y["scored_tokens"] else {}} for label, y in (("Y1", y1), ("Y2", y2))}
    results["outcome"] = outcome(results)
    return results


def _family_label(passed_label: str, failed_label: str, y: Mapping[str, Any]) -> str:
    test = y.get("test", {})
    label = passed_label if test.get("passed") else failed_label
    if test and not test.get("classification_evaluable", True):
        label += CLASSIFICATION_NOT_EVALUABLE_SUFFIX
    return label


def outcome(results: Mapping[str, Any]) -> dict[str, Any]:
    y1 = OUTCOME_Y1[2] if not results["precondition_Y1"]["passed"] else _family_label(OUTCOME_Y1[0], OUTCOME_Y1[1], results["Y1"])
    y2 = OUTCOME_Y2[2] if not results["precondition_Y2"]["passed"] else _family_label(OUTCOME_Y2[0], OUTCOME_Y2[1], results["Y2"])
    y3_entry = results["Y3"]
    y3 = OUTCOME_Y3[2] if not y3_entry.get("evaluable") else (OUTCOME_Y3[0] if y3_entry["rejected"] else OUTCOME_Y3[1])
    return {"label": f"{y1} | {y2} | {y3}", "Y1": y1, "Y2": y2, "Y3": y3}


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 014 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state['confirmation_014_sha256']}`", f"- Experiment 013 lock sha256: `{state['lock_013_sha256']}`",
             f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    incidents = list(exploration.get("incidents", [])) + ([state["confirmation"]["incident"]] if isinstance(state.get("confirmation"), dict) and "incident" in state["confirmation"] else [])
    if incidents:
        lines += ["## Incidents", ""] + [f"- `{e['phase']}` at commit `{e.get('commit', '?')}` ({e['at']}): {e['message']}" for e in incidents] + [""]

    def cmp_line(c: Mapping[str, Any]) -> str:
        return f"Spearman {f(c.get('spearman'), 3)}, R² {f(c.get('r2'), 3)}, MAE {f(c.get('mae'), 3)}, bias {f(c.get('bias'), 3)} (n {c.get('n')})"

    def cls_line(c: Mapping[str, Any]) -> str:
        return f"balanced accuracy {f(c.get('balanced_accuracy'), 3)} (sensitivity {f(c.get('sensitivity'), 3)}, specificity {f(c.get('specificity'), 3)}; raw {f(c.get('raw_accuracy'), 3)}, majority baseline {f(c.get('majority_baseline'), 3)}; measured firing {c.get('measured_firing')}/{c.get('n')})"

    def firing_line(sets: Mapping[str, Sequence[str]]) -> str:
        return "; ".join(f"{cls}: {', '.join(words)}" for cls, words in sets.items()) or "none"

    if "summary" in exploration:
        x, acc, n = exploration["exposed_check"], exploration["exposed_check"]["accounting"], exploration["neuron"]
        rep = exploration["replication"]["experiment_013"]
        lines += ["## Tier A — exposed pool (calibration record only; the floors are frozen constants)", "",
                  f"- Replication of Experiment 013: {rep['n']} pairs, max deviation {rep['max_abs_deviation']:.2e}; identities {', '.join(f'{k} {v:.1e}' for k, v in exploration['identities'].items())}; read weight vs Experiment 011 lock {exploration['read_weight_check']:.1e}",
                  f"- Neuron {n['layer']}/{n['index']}: cos(γ₂ ⊙ W_in, d̂_E) {f(n['cos_u_dE'], 3)}; r(W_out) {f(n['read_out'], 3)}; r(W_out)/denominator per template {{{', '.join(f'{t}: {f(v, 3)}' for t, v in n['read_out_over_denominators'].items())}}}; b_in {f(n['b_in'], 3)}",
                  f"- Operating points: pre_ref {f(acc['pre_ref_min'], 2)} to {f(acc['pre_ref_max'], 2)}; reference activation max {f(acc['reference_activation_max'], 3)} (below the firing threshold {FIRING_THRESHOLD})",
                  f"- Token means ({x['n_tokens']}): Δâ vs Δa — {cmp_line(x['token_means']['da_hat_vs_da'])}; Δa spread sd {f(x['token_means']['da_spread'], 3)}; axis-only Δâ vs Δa — {cmp_line(x['token_means']['da_axis_vs_da'])}",
                  f"- Pairs ({x['n_pairs']}): Δâ vs Δa — {cmp_line(x['pairs']['da_hat_vs_da'])}; Δp̂re vs Δpre — {cmp_line(x['pairs']['dpre_hat_vs_dpre'])}; Jacobian form vs Δpre — {cmp_line(x['pairs']['dpre_J_vs_dpre'])} (vs the exact form R² {f(x['pairs']['dpre_J_vs_dpre_hat']['r2'], 3)})",
                  f"- Classification (pairs): predictor {cls_line(x['pairs']['classification'])}; axis-only {cls_line(x['pairs']['classification_axis'])}",
                  f"- Amplitude among firing pairs: error {f(x['pairs']['amplitude_error_firing'], 3)} on a mean Δa of {f(x['pairs']['mean_da_firing'], 2)}; MAE all pairs {f(x['pairs']['da_hat_vs_da']['mae'], 3)}",
                  f"- Accounting (mean |Jacobian contribution|): E {f(acc['J_E_abs'], 2)}, MLP₁ {f(acc['J_mlp1_abs'], 2)}, heads₁ {f(acc['J_heads1_abs'], 2)} (negative in {f(acc['J_heads1_negative_fraction'], 2)} of pairs); axis {f(acc['J_axis_abs'], 2)}, off-axis {f(acc['J_offaxis_abs'], 2)}; remainders: pattern change |.| {f(acc['pattern_change_remainder_abs'], 3)}, LayerNorm linearization |.| {f(acc['jacobian_remainder_abs'], 3)}",
                  f"- Neuron's share of the block-2 read change among firing pairs {f(acc['term_share_of_mlp2_firing'], 2)}; mean term firing {f(acc['term_mean_firing'], 3)}, not firing {f(acc['term_mean_not_firing'], 3)}",
                  f"- Measured firing set (≥ half of frames), by class: {firing_line(exploration['firing_set'])}",
                  f"- Predicted firing set: {firing_line(exploration['predicted_firing_set'])}", ""]
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}`; predictions sha256 `{state['lock']['predictions_sha256']}`", ""]
    confirmation = state.get("confirmation")
    if confirmation and "stage1" in confirmation:
        s1 = confirmation["stage1"]
        lines += ["## Confirmation — stage 1 (fresh frames' reference states; prediction table digested before any fresh cue prompt)", "", f"- Table rows {len(s1['rows'])}; predicted to fire in {s1['predicted_firing_counts']['pairs_firing']} pairs; digest `{s1['digest']}`; commit `{s1['commit']}`"]
        for frame_id, entry in s1["frames"].items():
            if entry.get("template_defined"):
                lines.append(f"  - {frame_id}: {'valid' if entry['valid'] else 'INVALID'} (plural head change {f(entry['plural_head_change'], 3)}, cue effect {entry['cue_effect_positive']}/{entry['cue_effect_required']}; pre_ref {f(entry['pre_ref'], 2)})")
            else:
                lines.append(f"  - {frame_id}: {entry.get('note')}")
        lines.append("")
    if confirmation and "outcome" in confirmation:
        o = confirmation["outcome"]
        lines += [f"## Confirmation — stage 2 — `{o['label']}`", ""]
        for label, key in (("Y1 (strict: new tokens × exposed frames)", "Y1"), ("Y2 (frame-conditional: new tokens × fresh frames)", "Y2")):
            y = confirmation[key]
            pre = confirmation["precondition_Y1" if key == "Y1" else "precondition_Y2"]
            if "test" in y:
                t, d = y["test"], y["descriptive"]
                lines.append(f"- {label}: scored tokens {len(y['scored_tokens'])} ({'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'}); token means Δâ vs Δa — {cmp_line(t)}; {cls_line(t['classification'])} → {'pass' if t['passed'] else 'FAIL'} {t['failing'] or ''}")
                lines.append(f"  - pairs: Δâ vs Δa {cmp_line(d['pairs_da_hat_vs_da'])}; Δp̂re vs Δpre {cmp_line(d['pairs_dpre_hat_vs_dpre'])}; Jacobian form vs Δpre {cmp_line(d['pairs_dpre_J_vs_dpre'])}; amplitude error among firing pairs {f(d['amplitude_error_firing'], 3)} (mean Δa firing {f(d['mean_da_firing'], 2)}); MAE {f(d['mae_da'], 3)}; Δa spread sd {f(d['da_spread'], 3)}")
                lines.append(f"  - accounting |.|: E {f(d['accounting_abs']['E'], 2)}, MLP₁ {f(d['accounting_abs']['mlp1'], 2)}, heads₁ {f(d['accounting_abs']['heads1'], 2)}, axis {f(d['accounting_abs']['axis'], 2)}, off-axis {f(d['accounting_abs']['offaxis'], 2)}; remainders pattern change {f(d['pattern_change_remainder_abs'], 3)}, LayerNorm {f(d['jacobian_remainder_abs'], 3)}; neuron's share of the block-2 read change among firing pairs {f(d['term_share_of_mlp2_firing'], 2)}")
                lines.append(f"  - axis-only on this set: {cmp_line(y['axis'])}; {cls_line(y['axis']['classification'])}")
                sets = confirmation["firing_sets"][key]
                lines.append(f"  - measured firing set: {firing_line(sets['measured'])} | predicted: {firing_line(sets['predicted'])}")
            else:
                lines.append(f"- {label}: not evaluable ({len(y['scored_tokens'])} scored tokens; {'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'})")
        y3 = confirmation["Y3"]
        if y3.get("evaluable"):
            lines.append(f"- Y3 (axis-only alternative over both sets, {y3['n_token_means']} token means / {y3['n_pairs']} pairs): {cmp_line(y3)} (rejected iff R² < {Y3_R2_MAX}); {cls_line(y3['classification'])} (rejected iff < {Y3_BALANCED_ACCURACY_MAX}) → {'REJECTED' if y3['rejected'] else 'NOT REJECTED'}; the predictor on the same token means {cmp_line(y3['full_predictor_on_same_data'])}")
        else:
            lines.append(f"- Y3: not evaluable ({y3.get('n_token_means')} token means)")
        lines += ["", "| token | class | Y1 frames | Y1 Δâ | Y1 Δa | Y1 fires ĉ/meas | Y2 frames | Y2 Δâ | Y2 Δa | Y2 fires ĉ/meas | axis Δâ (Y1) | pre_ref (Y2) | term (Y1 meas) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        categories = {token["word"]: token["category"] for token in confirmation.get("tokens_meta", [])}
        for word in sorted(confirmation["Y1"]["tokens"], key=lambda w: -(confirmation["Y1"]["tokens"][w].get("da_hat_mean") or -9)):
            a, b = confirmation["Y1"]["tokens"][word], confirmation["Y2"]["tokens"].get(word, {})
            lines.append(f"| {word} | {categories.get(word, '')} | {a.get('n_valid_frames')} | {f(a.get('da_hat_mean'), 3)} | {f(a.get('da_mean'), 3)} | {f(a.get('fires_hat_fraction'), 2)}/{f(a.get('fires_fraction'), 2)} | {b.get('n_valid_frames')} | {f(b.get('da_hat_mean'), 3)} | {f(b.get('da_mean'), 3)} | {f(b.get('fires_hat_fraction'), 2)}/{f(b.get('fires_fraction'), 2)} | {f(a.get('da_axis_mean'), 3)} | {f(b.get('pre_ref_mean'), 2)} | {f(a.get('term_mean'), 3)} |")
        lines.append("")
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
