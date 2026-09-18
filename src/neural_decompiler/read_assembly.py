"""Experiment 010: exact attribution of ``L03.H04``'s head-readable number signal through layers 0–2.

Experiment 009 confirmed that the head's number-axis output change is a fixed-normalization linear
read-out of the residual arriving at the cue position (level P1). This module applies that exact
read functional — with the frame's frozen reference attention weight and LayerNorm scale — to every
additive piece of the residual change at the cue position under an E-patch: the token-local encoding
``ΔE`` (split along the encoding number axis and into the layer-0 MLP's neurons) and the outputs of the
attention heads and MLPs of layers 1 and 2. Nothing is fitted; every number is a measured or weight-only
quantity under one linear functional, normalized by the template's plural cue. Constants are copied
from design revision 2.
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
from . import head_transport as ht
from . import plural_mechanism as pm
from . import supervised_subspace as ss
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest
from .interventions import ReplacementSource
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 2).

EXPERIMENT_DIR = "experiments/010-read-direction-assembly"
INHERITED_009_EXTRACT_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-009-confirmation-epatch-means.json"
EXPERIMENT_009_LOCK_PATH = ht.LOCK_RELATIVE_PATH
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260923
RESULTS_SCHEMA_VERSION = 1
PHASES = ("explore", "report")
INHERITED_009_SCHEMA_VERSION = 1

S_MIN = 0.35
S_HIGH = 0.65
C_MIN = 0.10  # net layer contribution bound for relay
G_MAX = 0.25  # gross layer change and maximum cumulative deviation bound for relay
CONSENSUS = 0.75
CONSISTENCY_MIN = 0.10  # |f_k| for a consistent opposer / supporter
N80_CONCENTRATED = 40
TOP_NEURONS = 20
NEURON_MASS_FRACTION = 0.8
IDENTITY_TOLERANCE = 1e-4  # ρ identity, relative to the plural cue's measured head change
P1_CROSS_CHECK_TOLERANCE = 1e-6
NEURON_SUM_TOLERANCE = 1e-4  # relative; ΔE is a float32 quantity while the neuron terms accumulate in float64
COMPONENT_ORDER = tuple(f"L01.H0{i}" for i in range(8)) + ("L01.MLP",) + tuple(f"L02.H0{i}" for i in range(8)) + ("L02.MLP",)
LAYER_OF = {key: 1 if key.startswith("L01") else 2 for key in COMPONENT_ORDER}
TOKEN_CLASSES = ("E_BORNE", "LAYER_BORNE", "RELAYED", "AMPLIFIED", "MIXED")
SUMMARY_LABELS = ("E_BORNE", "LAYER_BORNE", "MIXED")


class PhaseError(pm.PhaseError):
    pass


# ---------------------------------------------------------------------------
# Pool: 63 tokens × 24 frames (Experiments 008 + 009, all exposed).


def build_pool_010(manifest: ScreeningManifest, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009) -> cs.Pool008:
    base = cs.build_pool(manifest, extension, confirmation_006)
    frames = base.frames + confirmation_009.frames
    origin = dict(base.frame_origin) | {frame.frame_id: "confirmation-009" for frame in confirmation_009.frames}
    tokens = list(base.tokens)
    category, source = dict(base.token_category), dict(base.token_source)
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation_009.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"Experiment 009 token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-23"
    if len(tokens) != 63 or len(frames) != 24:
        raise ValueError(f"expected 63 tokens and 24 frames, found {len(tokens)} and {len(frames)}")
    by_id = {token_id: name for name, token_id in tokens}
    for frame in confirmation_009.frames:
        if by_id[frame.cue_ids["pl"]] != base.plural_cue[frame.template_id] or by_id[frame.cue_ids["sg"]] != by_id[base.reference_ids[frame.template_id]]:
            raise ValueError(f"{frame.frame_id}: cue tokens differ from the template's original cues")
    return cs.Pool008(frames, origin, tuple(tokens), category, source, base.nouns, base.noun_source, dict(base.reference_ids), dict(base.plural_cue))


# ---------------------------------------------------------------------------
# Inherited Experiment 009 confirmation responses (derived extract).


def inherited_009_payload(responses: Mapping[str, Mapping[str, Any]], *, source: Mapping[str, Any], manifest_sha256: str, extension_sha256: str, confirmation_009_sha256: str,
                          lock_sha256: str, model: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"schema_version": INHERITED_009_SCHEMA_VERSION,
               "description": "Derived extract of Experiment 009's confirmation E-patch responses (mean contrast shift per (token, fresh frame) over the 79 single-token exposed nouns). Experiment 010 recomputes these and requires agreement within 1e-6.",
               "source": dict(source), "manifest_sha256": manifest_sha256, "extension_sha256": extension_sha256, "confirmation_009_sha256": confirmation_009_sha256, "lock_sha256": lock_sha256, "model": dict(model),
               "responses": {key: {"template_id": entry["template_id"], "mean_shift": float(entry["mean_shift"])} for key, entry in sorted(responses.items())}}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_009(path: Path, *, manifest_sha256: str, extension_sha256: str, confirmation_009_sha256: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited 009 extract: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "description", "source", "manifest_sha256", "extension_sha256", "confirmation_009_sha256", "lock_sha256", "model", "responses", "content_sha256"}, "inherited 009 extract")
    if payload["schema_version"] != INHERITED_009_SCHEMA_VERSION:
        raise ValueError("inherited 009 extract schema version is not frozen")
    if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("inherited 009 extract content_sha256 does not match its canonical payload")
    if (payload["manifest_sha256"], payload["extension_sha256"], payload["confirmation_009_sha256"]) != (manifest_sha256, extension_sha256, confirmation_009_sha256):
        raise ValueError("inherited 009 extract was recorded against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}:
        raise ValueError("inherited 009 extract was recorded for a different pinned model")
    if len(payload["responses"]) != 23 * 6:
        raise ValueError(f"inherited 009 extract must hold 138 responses, found {len(payload['responses'])}")
    return payload


# ---------------------------------------------------------------------------
# Results state (two phases).

_STATE_KEYS = {"schema_version", "run_id", "created_at", "manifest_sha256", "extension_sha256", "confirmation_sha256", "confirmation_009_sha256", "protocol_code_commit", "git_dirty",
               "model", "versions", "phases", "executed_prompt_keys", "executed_noun_keys", "exploration", "invalidated_runs", "state_sha256"}


def new_results_state(*, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str, confirmation_009_sha256: str, protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {"schema_version": RESULTS_SCHEMA_VERSION, "run_id": pm.sha256_text(manifest_sha256 + extension_sha256 + confirmation_sha256 + confirmation_009_sha256 + protocol_code_commit + pm.utc_now())[:16],
            "created_at": pm.utc_now(), "manifest_sha256": manifest_sha256, "extension_sha256": extension_sha256, "confirmation_sha256": confirmation_sha256, "confirmation_009_sha256": confirmation_009_sha256,
            "protocol_code_commit": protocol_code_commit, "git_dirty": False, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "versions": dict(versions),
            "phases": {phase: {"status": "not_started"} for phase in PHASES}, "executed_prompt_keys": [], "executed_noun_keys": [], "exploration": {}, "invalidated_runs": []}


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
    elif phase == "report":
        if status["explore"] != "complete" and not state["exploration"].get("incidents"):
            raise PhaseError("report requires the completed explore phase (or a recorded incident)")
    else:
        raise PhaseError(f"unknown phase {phase}")


# ---------------------------------------------------------------------------
# The exact P1 read functional and the encoding's neuron split.


@dataclass(frozen=True)
class ReadFunctional:
    """ρ_f(x) = (A_c^ref / σ_c^ref) · ⟨x − mean(x)·1, γ₃ ⊙ m⟩ with the frame's frozen reference statistics."""

    attention_ref: float
    sigma_ref: float
    weight: torch.Tensor  # γ₃ ⊙ m (float64)

    @property
    def scale(self) -> float:
        return self.attention_ref / self.sigma_ref

    def inner(self, x: torch.Tensor) -> float:
        x = x.double()
        return float((x - x.mean()) @ self.weight)

    def __call__(self, x: torch.Tensor) -> float:
        return self.scale * self.inner(x)

    def rows(self, matrix: torch.Tensor) -> torch.Tensor:
        """ρ_f applied to every row of ``matrix`` (float64 vector)."""
        matrix = matrix.double()
        return self.scale * ((matrix - matrix.mean(dim=1, keepdim=True)) @ self.weight)


def read_weight(head: ht.HeadWeights, axis_T: pm.SiteAxis) -> torch.Tensor:
    return (head.ln_w * head.read_direction(axis_T.direction)).contiguous()


def read_functional(head: ht.HeadWeights, ref: ht.FrameReference, axis_T: pm.SiteAxis) -> ReadFunctional:
    return ReadFunctional(float(ref.attention[ref.frame.p_c]), head.scale(ref.residuals[ref.frame.p_c]), read_weight(head, axis_T))


def neuron_activations(weights: pm.Weights, token_id: int) -> torch.Tensor:
    """a_j = GELU(pre_j) of the layer-0 MLP for one token (weight-only, float64), the pieces of E(w) before W_out."""
    hidden = pm.exact_layer_norm(weights.W_E[int(token_id)], weights.ln2_0_w, weights.ln2_0_b, weights.eps)
    pre = hidden @ weights.mlp0_W_in + weights.mlp0_b_in
    return torch.nn.functional.gelu(pre).double()


def neuron_terms(weights: pm.Weights, functional: ReadFunctional, token_id: int, reference_id: int) -> torch.Tensor:
    """c_j = Δa_j · ρ_f(W_out[j]); Σ_j c_j = ρ_f(ΔE) exactly (b_out cancels in ΔE)."""
    delta_a = neuron_activations(weights, token_id) - neuron_activations(weights, reference_id)
    return delta_a * functional.rows(weights.mlp0_W_out)


def neuron_concentration(terms: torch.Tensor) -> dict[str, Any]:
    """n_80 on absolute attribution mass (sorted |c_j| descending, ties by index ascending), positive and negative masses."""
    magnitude = terms.abs()
    total = float(magnitude.sum())
    if total <= 0.0:
        return {"n_80": None, "positive_mass": 0.0, "negative_mass": 0.0, "total_abs_mass": 0.0, "top": []}
    order = sorted(range(terms.numel()), key=lambda j: (-float(magnitude[j]), j))
    running, n_80 = 0.0, terms.numel()
    for count, j in enumerate(order, start=1):
        running += float(magnitude[j])
        if running >= NEURON_MASS_FRACTION * total - 1e-12:
            n_80 = count
            break
    return {"n_80": n_80, "positive_mass": float(terms[terms > 0].sum()), "negative_mass": float(terms[terms < 0].sum()), "total_abs_mass": total,
            "top": [{"neuron": j, "c": float(terms[j])} for j in order[:TOP_NEURONS]]}


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else 0.0


# ---------------------------------------------------------------------------
# Measurement: one E-patch per (token, frame) with the component captures.


@dataclass
class Attribution:
    token: str
    token_id: int
    frame_id: str
    template: str
    rho_E: float
    rho_par: float
    rho_perp: float
    rho_components: dict[str, float]  # k -> ρ_f(Δout_k)
    rho_total: float  # ρ_f(Δr_c)
    identity_error: float
    p1_cross_check: float  # |ρ_f(Δr_c) − ⟨ΔT₁, d̂_T⟩| from ht.ov_levels
    own_reference: bool  # the token is the frame's reference cue: the E-patch is the identity and the frame is uninformative for it
    head_change: float  # ⟨ΔT_measured, d̂_T⟩
    shifts: dict[str, float]  # per single-token noun E-patch contrast shift
    neuron: dict[str, Any]
    neuron_sum_error: float
    g_E_inner: float  # ⟨ΔE − mean, γ₃ ⊙ m⟩ (weight-only inner read)
    terms: torch.Tensor  # c_j per layer-0 MLP neuron (float64)


def _component_sites(frame: pm.Frame) -> list[pm.Site]:
    return [(key, frame.p_c) for key in COMPONENT_ORDER]


def _capture_sites(model: Any, frame: pm.Frame) -> list[pm.Site]:
    sites = cs._sites(model, frame)
    return [sites["R0"], sites["R1"], sites["T"], sites["A"], sites["R3"]] + _component_sites(frame) + [(f"RESID_PRE.L{ht.HEAD_LAYER}", k) for k in range(frame.p_t + 1)]


def capture_reference(model: Any, head: ht.HeadWeights, reference: pm.Prompt, nouns: Sequence[pm.Noun]) -> tuple[ht.FrameReference, dict[str, torch.Tensor]]:
    frame = reference.frame
    sites = cs._sites(model, frame)
    run = pm.capture_prompt(model, reference, _capture_sites(model, frame))
    vectors = {stage: run.vector(sites[stage]).double() for stage in ("R0", "R1", "T")}
    residuals = [run.vector((f"RESID_PRE.L{ht.HEAD_LAYER}", k)).double() for k in range(frame.p_t + 1)]
    attention = run.vector(sites["A"])[ht.HEAD_INDEX].double()
    reconstructed = ht.reconstruct_head_result(head, attention, residuals)
    error = float((reconstructed - vectors["T"]).norm()) / max(float(vectors["T"].norm()), 1e-12)
    if error > ht.RECONSTRUCTION_TOLERANCE:
        raise pm.IncidentError(f"{frame.frame_id}: Σ_k A_k v_k W_O does not reproduce the captured {ht.HEAD_KEY} result (relative error {error:.2e})")
    components = {key: run.vector((key, frame.p_c)).double() for key in COMPONENT_ORDER}
    return ht.FrameReference(frame, reference, vectors, residuals, attention, vectors["T"], pm.contrasts(run.logits, nouns), error), components


def measure_token_010(model: Any, weights: pm.Weights, head: ht.HeadWeights, ref: ht.FrameReference, ref_components: Mapping[str, torch.Tensor], functional: ReadFunctional,
                      name: str, token_id: int, e_axis: pm.SiteAxis, axis_T: pm.SiteAxis, nouns: Sequence[pm.Noun]) -> Attribution:
    frame = ref.frame
    sites = cs._sites(model, frame)
    e_ref = pm.lexicon_vector(weights, ref.reference.cue_token_id).double()
    e_w = pm.lexicon_vector(weights, token_id).double()
    delta_e = e_w - e_ref
    direction = e_axis.direction.double()
    parallel = (delta_e @ direction) * direction
    terms = neuron_terms(weights, functional, token_id, ref.reference.cue_token_id)
    rho_E = functional(delta_e)
    natural_scale = functional.scale * float(functional.weight.norm()) * max(float(e_w.norm()), float(e_ref.norm()), 1e-12)
    neuron_sum_error = abs(float(terms.sum()) - rho_E) / natural_scale
    if token_id == ref.reference.cue_token_id:
        components = {key: 0.0 for key in COMPONENT_ORDER}
        return Attribution(name, token_id, frame.frame_id, frame.template_id, 0.0, 0.0, 0.0, components, 0.0, 0.0, 0.0, True, 0.0, {noun.lexical_key: 0.0 for noun in nouns if noun.single_token},
                           neuron_concentration(torch.zeros_like(terms)), 0.0, 0.0, torch.zeros_like(terms))
    e_site = ("L00.MLP", frame.p_c)
    run = pm.run_patched(model, ref.reference, {e_site: e_w.reshape(1, 1, -1).to(torch.float32)}, {e_site: ReplacementSource.RESAMPLE}, capture_sites=_capture_sites(model, frame))
    delta_r0 = run.vector(sites["R0"]).double() - ref.vectors["R0"]
    identity_r0 = float((delta_r0 - delta_e).abs().max())
    if identity_r0 > cs.IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{frame.frame_id}/{name}: the residual after layer 0 does not change by ΔE_T(w) (max {identity_r0:.2e})")
    delta_r_c = run.vector(sites["R1"]).double() - ref.vectors["R1"]
    components = {key: functional(run.vector((key, frame.p_c)).double() - ref_components[key]) for key in COMPONENT_ORDER}
    rho_total = functional(delta_r_c)
    identity_error = abs(rho_total - rho_E - sum(components.values()))
    residuals_patch = [run.vector((f"RESID_PRE.L{ht.HEAD_LAYER}", k)).double() for k in range(frame.p_t + 1)]
    attention_patch = run.vector(sites["A"])[ht.HEAD_INDEX].double()
    delta_head = run.vector(sites["T"]).double() - ref.vectors["T"]
    ov = ht.ov_levels(head, p_c=frame.p_c, attention_ref=ref.attention, attention_patch=attention_patch, residuals_ref=ref.residuals, residuals_patch=residuals_patch, delta_head=delta_head)
    p1_cross = abs(rho_total - float(ov["delta_T1"] @ axis_T.direction.double()))
    c_patched = pm.contrasts(run.logits, nouns)
    shifts = {noun.lexical_key: c_patched[noun.lexical_key] - ref.c_by_noun[noun.lexical_key] for noun in nouns if noun.single_token}
    return Attribution(name, token_id, frame.frame_id, frame.template_id, rho_E, functional(parallel), functional(delta_e - parallel), components, rho_total, identity_error, p1_cross, False,
                       float(delta_head @ axis_T.direction.double()), shifts, neuron_concentration(terms), neuron_sum_error, functional.inner(delta_e), terms)


# ---------------------------------------------------------------------------
# Fractions, layer statistics, classification, strata, summary.


def fractions(record: Attribution, plural: Attribution, axis_T: pm.SiteAxis, plural_inner_E: float) -> dict[str, Any] | None:
    """Normalized by the plural cue's measured head-output change in the frame; None if the frame is uninformative for the token."""
    denominator = plural.head_change
    if record.own_reference or abs(denominator) < cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma:
        return None
    f_k = {key: record.rho_components[key] / denominator for key in COMPONENT_ORDER}
    cumulative, running = [], 0.0
    for key in COMPONENT_ORDER:
        running += f_k[key]
        cumulative.append(running)
    out = {"f_E": record.rho_E / denominator, "f_par": record.rho_par / denominator, "f_perp": record.rho_perp / denominator, "f_k": f_k,
           "f_total": record.rho_total / denominator, "q_T": record.head_change / denominator,
           "f_L1": sum(f_k[key] for key in COMPONENT_ORDER if LAYER_OF[key] == 1), "f_L2": sum(f_k[key] for key in COMPONENT_ORDER if LAYER_OF[key] == 2)}
    out["f_layers"] = out["f_L1"] + out["f_L2"]
    out["G"] = sum(abs(value) for value in f_k.values())
    out["D"] = max(abs(value) for value in cumulative)
    out["cumulative"] = cumulative
    out["g_E"] = record.g_E_inner / plural_inner_E if abs(plural_inner_E) > 1e-12 else None
    out["raw"] = {"rho_E": record.rho_E, "rho_par": record.rho_par, "rho_perp": record.rho_perp, "rho_components": dict(record.rho_components), "rho_total": record.rho_total,
                  "head_change": record.head_change, "denominator_measured": denominator, "denominator_rho_plural": plural.rho_total,
                  "identity_error": record.identity_error, "p1_cross_check": record.p1_cross_check, "neuron_sum_error": record.neuron_sum_error}
    return out


def classify_token(means: Mapping[str, float | None]) -> str:
    required = ("f_E", "f_par", "f_layers", "G", "D", "f_total")
    if any(means.get(key) is None for key in required):
        return "MIXED"
    f_E, f_par, f_layers, G, D, f_total = (means[key] for key in required)
    relay = abs(f_layers) <= C_MIN and G <= G_MAX and D <= G_MAX
    if f_E <= S_MIN and f_par >= S_MIN and relay:
        return "E_BORNE"
    if f_E >= S_MIN and f_layers <= -(f_E - S_MIN) and f_total <= S_MIN:
        return "LAYER_BORNE"
    if f_total >= S_HIGH and relay:
        return "RELAYED"
    if f_layers >= C_MIN and f_total >= S_HIGH:
        return "AMPLIFIED"
    return "MIXED"


def stratum(q_T: float | None) -> str:
    if q_T is None:
        return "uninformative"
    if q_T <= S_MIN:
        return "low"
    if q_T >= S_HIGH:
        return "high"
    return "mid"


def consistent_components(per_frame_by_token: Mapping[str, Mapping[str, Mapping[str, Any]]], tokens: Sequence[str]) -> dict[str, list[str]]:
    """Consistent opposers / supporters for a token set: |f_k| ≥ 0.10 with the right sign in ≥ 75% of informative frames of ≥ 75% of the tokens."""
    opposers, supporters = [], []
    if not tokens:
        return {"opposers": [], "supporters": []}
    for key in COMPONENT_ORDER:
        opposing_tokens = supporting_tokens = 0
        for token in tokens:
            frames = [analysis for analysis in per_frame_by_token[token].values() if analysis is not None]
            if not frames:
                continue
            opposing = sum(1 for analysis in frames if analysis["f_k"][key] <= -CONSISTENCY_MIN) / len(frames)
            supporting = sum(1 for analysis in frames if analysis["f_k"][key] >= CONSISTENCY_MIN) / len(frames)
            opposing_tokens += opposing >= CONSENSUS
            supporting_tokens += supporting >= CONSENSUS
        if opposing_tokens >= CONSENSUS * len(tokens):
            opposers.append(key)
        if supporting_tokens >= CONSENSUS * len(tokens):
            supporters.append(key)
    return {"opposers": opposers, "supporters": supporters}


def summarize(rows: Mapping[str, Mapping[str, Any]], low_components: Mapping[str, list[str]], concentrated: bool) -> dict[str, Any]:
    low = [row for row in rows.values() if row["stratum"] == "low"]
    out: dict[str, Any] = {"low_tokens": [row["token"] for row in low], "n_low": len(low), "class_counts_low": dict(Counter(row["class"] for row in low)), "concentrated_encoding": concentrated}
    if not low:
        out["label"] = "MIXED"
        out["note"] = "the low stratum is empty"
        return out
    e_borne = sum(1 for row in low if row["class"] == "E_BORNE")
    layer_borne = sum(1 for row in low if row["class"] == "LAYER_BORNE")
    if e_borne >= CONSENSUS * len(low):
        label = "E_BORNE"
    elif layer_borne >= CONSENSUS * len(low) and low_components["opposers"]:
        label = "LAYER_BORNE"
    else:
        label = "MIXED"
    out["label"] = label + ("+CONCENTRATED_ENCODING" if concentrated else "")
    out["low_opposers"] = list(low_components["opposers"])
    out["low_supporters"] = list(low_components["supporters"])
    return out


def predictor_check(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive only: Spearman and MAE of f_E, g_E, and f_total against the measured q_T over the tokens with informative means."""
    out = {}
    for key in ("f_E", "g_E", "f_total"):
        pairs = [(row["means"][key], row["means"]["q_T"]) for row in rows.values() if row["means"].get(key) is not None and row["means"].get("q_T") is not None]
        if len(pairs) >= 3:
            out[key] = {"spearman": pm.spearman([p for p, _ in pairs], [q for _, q in pairs]), "mae": pm._mean([abs(p - q) for p, q in pairs]), "n": len(pairs)}
        else:
            out[key] = {"spearman": None, "mae": None, "n": len(pairs)}
    return out


# ---------------------------------------------------------------------------
# Exploration and the report.


def run_exploration(model: Any, pool: cs.Pool008, *, state: dict[str, Any], results_path: Path | None, inherited_006: Mapping[str, Any], inherited_007: Mapping[str, Any],
                    inherited_009: Mapping[str, Any], log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    nouns = pool.single_nouns
    exploration = state["exploration"]
    say("stage axes from the 24 frames' clean cue pairs")
    axes = cs.stage_axes(cache, weights, pool)
    e_axis, axis_T = axes["R0"], axes["T"]
    exploration["axes"] = {stage: {"sigma": axis.sigma, "cos_with_E_axis": pm.cosine(axis.direction, e_axis.direction)} for stage, axis in axes.items()}
    exploration["read_weight_norm"] = float(read_weight(head, axis_T).norm())
    say("measurements: 63 tokens × 24 frames with the layer-1/2 component captures")
    records: dict[tuple[str, str], Attribution] = {}
    functionals: dict[str, ReadFunctional] = {}
    reconstruction: dict[str, float] = {}
    for frame in pool.frames:
        ref, ref_components = capture_reference(model, head, pool.reference_prompt(frame), nouns)
        functional = read_functional(head, ref, axis_T)
        functionals[frame.frame_id] = functional
        reconstruction[frame.frame_id] = ref.reconstruction_error
        for name, token_id in pool.tokens:
            records[(name, frame.frame_id)] = measure_token_010(model, weights, head, ref, ref_components, functional, name, token_id, e_axis, axis_T, nouns)
        say(f"  frame {frame.frame_id}: A_c^ref {functional.attention_ref:.3f}, σ_c^ref {functional.sigma_ref:.3f}")
    exploration["reference_scalars"] = {frame_id: {"attention_ref": f.attention_ref, "sigma_ref": f.sigma_ref, "reconstruction_error": reconstruction[frame_id]} for frame_id, f in functionals.items()}
    # Replication on the noun subsets of Experiments 006, 007, and 009.
    exposed_keys = [noun.lexical_key for noun in pool.single_nouns_from("exposed-60")]
    confirmation_keys = [noun.lexical_key for noun in pool.single_nouns_from("confirmation-20")]
    all_keys = [noun.lexical_key for noun in nouns]
    exploration["replication"] = {
        "experiment_006": cs.check_replication({f"{n}|{f}": pm._mean([r.shifts[k] for k in exposed_keys]) for (n, f), r in records.items()}, inherited_006["responses"], label="Experiment 006 exposed responses"),
        "experiment_007": cs.check_replication({f"{n}|{f}": pm._mean([r.shifts[k] for k in confirmation_keys]) for (n, f), r in records.items()}, inherited_007["responses"], label="Experiment 007 confirmation responses"),
        "experiment_009": cs.check_replication({f"{n}|{f}": pm._mean([r.shifts[k] for k in all_keys]) for (n, f), r in records.items()}, inherited_009["responses"], label="Experiment 009 confirmation responses"),
    }
    say("replication: " + "; ".join(f"{key} {value['max_abs_deviation']:.2e}" for key, value in exploration["replication"].items()))
    if results_path is not None:
        write_results_state(results_path, state)
    # Identities and fractions.
    per_frame: dict[str, dict[str, dict[str, Any] | None]] = {name: {} for name, _ in pool.tokens}
    worst_identity = worst_p1 = worst_neuron = 0.0
    for frame in pool.frames:
        plural = records[(pool.plural_cue[frame.template_id], frame.frame_id)]
        scale = max(abs(plural.rho_total), 1e-12)  # the plural cue's ρ_f(Δr_c) in the frame
        for name, _ in pool.tokens:
            record = records[(name, frame.frame_id)]
            worst_identity = max(worst_identity, record.identity_error / scale)
            worst_p1 = max(worst_p1, record.p1_cross_check / scale)
            worst_neuron = max(worst_neuron, record.neuron_sum_error)
            if record.identity_error / scale > IDENTITY_TOLERANCE:
                raise pm.IncidentError(f"{frame.frame_id}/{name}: ρ(Δr_c) ≠ ρ(ΔE) + Σ_k ρ(Δout_k) (relative error {record.identity_error / scale:.2e})")
            if record.p1_cross_check / scale > P1_CROSS_CHECK_TOLERANCE:
                raise pm.IncidentError(f"{frame.frame_id}/{name}: ρ(Δr_c) does not equal Experiment 009's P1 prediction (relative error {record.p1_cross_check / scale:.2e})")
            if record.neuron_sum_error > NEURON_SUM_TOLERANCE:
                raise pm.IncidentError(f"{frame.frame_id}/{name}: the neuron terms do not sum to ρ(ΔE) (error {record.neuron_sum_error:.2e} of the functional's natural scale)")
            per_frame[name][frame.frame_id] = fractions(record, plural, axis_T, plural.g_E_inner)
    exploration["identities"] = {"max_rho_identity_error": worst_identity, "max_p1_cross_check": worst_p1, "max_neuron_sum_error": worst_neuron}
    # Token rows.
    rows: dict[str, dict[str, Any]] = {}
    for name, token_id in pool.tokens:
        informative = {frame_id: analysis for frame_id, analysis in per_frame[name].items() if analysis is not None}
        keys = ("f_E", "f_par", "f_perp", "f_total", "q_T", "f_L1", "f_L2", "f_layers", "G", "D", "g_E")
        means = {key: cs._mean_or_none([analysis[key] for analysis in informative.values()]) for key in keys}
        means["f_k"] = {key: cs._mean_or_none([analysis["f_k"][key] for analysis in informative.values()]) for key in COMPONENT_ORDER}
        per_template = {}
        for template in pm.TEMPLATE_ORDER:
            template_frames = [analysis for frame in pool.frames_of(template) for analysis in [informative.get(frame.frame_id)] if analysis is not None]
            per_template[template] = {key: cs._mean_or_none([analysis[key] for analysis in template_frames]) for key in keys}
            per_template[template]["f_k"] = {key: cs._mean_or_none([analysis["f_k"][key] for analysis in template_frames]) for key in COMPONENT_ORDER}
        # Neurons: one deterministic concentration and top-20 per token and template, computed on the template-mean terms over the token's informative frames.
        neurons = {}
        for template in pool.plural_cue:
            frames = [frame.frame_id for frame in pool.frames_of(template) if frame.frame_id in informative]
            if not frames:
                neurons[template] = {"n_80": None, "positive_mass": None, "negative_mass": None, "total_abs_mass": 0.0, "top": [], "n_80_frame_mean": None}
                continue
            mean_terms = torch.stack([records[(name, frame_id)].terms for frame_id in frames]).mean(dim=0)
            concentration = neuron_concentration(mean_terms)
            concentration["n_80_frame_mean"] = cs._mean_or_none([records[(name, frame_id)].neuron["n_80"] for frame_id in frames])
            neurons[template] = concentration
        is_reference = token_id in set(pool.reference_ids.values())
        rows[name] = {"token": name, "token_id": token_id, "category": pool.token_category[name], "source": pool.token_source[name], "n_informative": len(informative), "reference_cue": is_reference,
                      "means": means, "per_template": per_template, "class": classify_token(means), "stratum": ("reference" if is_reference else stratum(means["q_T"])), "neurons": neurons}
    low_tokens = [name for name, row in rows.items() if row["stratum"] == "low"]
    high_tokens = [name for name, row in rows.items() if row["stratum"] == "high"]
    low_components = consistent_components(per_frame, low_tokens)
    high_components = consistent_components(per_frame, high_tokens)
    # Neuron overlap versus the template's plural cue.
    overlap = {}
    for template, plural_name in pool.plural_cue.items():
        plural_top = [entry["neuron"] for entry in rows[plural_name]["neurons"][template]["top"]]
        overlap[template] = {"low": cs._mean_or_none([jaccard(plural_top, [e["neuron"] for e in rows[name]["neurons"][template]["top"]]) for name in low_tokens if name != plural_name and rows[name]["neurons"][template]["top"]]),
                             "high": cs._mean_or_none([jaccard(plural_top, [e["neuron"] for e in rows[name]["neurons"][template]["top"]]) for name in high_tokens if name != plural_name and rows[name]["neurons"][template]["top"]])}

    def concentrated_value(name: str, template: str) -> bool:
        value = rows[name]["neurons"][template]["n_80"]
        return value is not None and value <= N80_CONCENTRATED

    # Each plural cue in its own template(s); every low-stratum token in every template.
    concentrated = all(concentrated_value(plural_name, template) for template, plural_name in pool.plural_cue.items()) and all(concentrated_value(name, template) for name in low_tokens for template in pool.plural_cue)
    exploration["tokens"] = rows
    exploration["per_frame"] = per_frame
    exploration["strata"] = {"low": low_tokens, "mid": [name for name, row in rows.items() if row["stratum"] == "mid"], "high": high_tokens,
                             "uninformative": [name for name, row in rows.items() if row["stratum"] == "uninformative"], "reference": [name for name, row in rows.items() if row["stratum"] == "reference"]}
    exploration["components"] = {"low": low_components, "high": high_components}
    exploration["neuron_overlap"] = overlap
    exploration["predictor_check"] = predictor_check(rows)
    exploration["summary"] = summarize(rows, low_components, concentrated)
    say(f"summary: {exploration['summary']['label']} (low stratum {low_tokens}; opposers {low_components['opposers']})")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 010 Report", "", f"- Run ID: `{state['run_id']}`", f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    if exploration.get("incidents"):
        lines += ["## Incidents", ""] + [f"- `{e['phase']}` at commit `{e.get('commit', '?')}` ({e['at']}): {e['message']}" for e in exploration["incidents"]] + [""]
    if "summary" in exploration:
        rep = exploration["replication"]
        ident = exploration["identities"]
        lines += ["## Replication and identities", "", "- " + "; ".join(f"{key}: {value['n']} pairs, max deviation {value['max_abs_deviation']:.2e}" for key, value in rep.items()),
                  f"- ρ identity max relative error {ident['max_rho_identity_error']:.2e}; P1 cross-check {ident['max_p1_cross_check']:.2e}; neuron-sum {ident['max_neuron_sum_error']:.2e}", ""]
        summary = exploration["summary"]
        lines += [f"## Summary — `{summary['label']}`", "", f"- Low stratum ({summary['n_low']}): {summary['low_tokens']}; classes {summary['class_counts_low']}",
                  f"- Consistent opposers for the low stratum: {exploration['components']['low']['opposers'] or 'none'}; supporters: {exploration['components']['low']['supporters'] or 'none'}",
                  f"- Consistent opposers for the high stratum: {exploration['components']['high']['opposers'] or 'none'}; supporters: {exploration['components']['high']['supporters'] or 'none'}",
                  f"- Strata: low {exploration['strata']['low']}; mid {exploration['strata']['mid']}; high {len(exploration['strata']['high'])} tokens",
                  "- Descriptive predictor check (no fitting): " + "; ".join(f"{key}: Spearman {f(v['spearman'], 3)}, MAE {f(v['mae'], 3)} (n {v['n']})" for key, v in exploration["predictor_check"].items()),
                  f"- Neuron top-20 Jaccard versus the template's plural cue: {exploration['neuron_overlap']}", ""]
        lines += ["## Ledger of the head-readable number signal (fractions of the plural cue's measured head change; means over informative frames)", "",
                  "| token | category | q_T | f_E | f_∥ | f_⊥ | f_L1 | f_L2 | net | G | D | f_total | class | stratum | n_80 (cardinal) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in sorted(exploration["tokens"].values(), key=lambda r: (r["means"]["q_T"] if r["means"]["q_T"] is not None else 9)):
            m = row["means"]
            lines.append(f"| {row['token']} | {row['category']} | {f(m['q_T'])} | {f(m['f_E'])} | {f(m['f_par'])} | {f(m['f_perp'])} | {f(m['f_L1'])} | {f(m['f_L2'])} | {f(m['f_layers'])} | {f(m['G'])} | {f(m['D'])} | {f(m['f_total'])} | {row['class']} | {row['stratum']} | {f(row['neurons']['cardinal']['n_80'], 0)} |")
        lines += ["", "## Component contributions (means over informative frames) for the low and high strata", "", "| component | " + " | ".join(exploration["strata"]["low"]) + " | high-stratum mean |", "|---|" + "---|" * (len(exploration["strata"]["low"]) + 1)]
        high = exploration["strata"]["high"]
        for key in COMPONENT_ORDER:
            values = [f(exploration["tokens"][name]["means"]["f_k"][key]) for name in exploration["strata"]["low"]]
            high_mean = cs._mean_or_none([exploration["tokens"][name]["means"]["f_k"][key] for name in high])
            lines.append(f"| {key} | " + " | ".join(values) + f" | {f(high_mean)} |")
        lines += ["", "## Neurons (layer-0 MLP; template-mean signed contributions to ρ(ΔE))", ""]
        for name in list(exploration["strata"]["low"]) + [exploration["tokens"][n]["token"] for n in exploration["tokens"] if exploration["tokens"][n]["category"] == "original-cue" and exploration["tokens"][n]["stratum"] == "high"]:
            row = exploration["tokens"][name]
            for template, entry in row["neurons"].items():
                top = ", ".join(f"{e['neuron']}:{e['c']:+.3f}" for e in entry["top"][:10])
                lines.append(f"- {name} / {template}: n_80 {f(entry['n_80'], 0)}, +mass {f(entry['positive_mass'])}, −mass {f(entry['negative_mass'])}; top: {top}")
        lines.append("")
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
