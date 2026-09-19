"""Experiment 012: is the layers-1–2 correction to the encoding read a token-local MLP computation?

Experiment 011 confirmed that the weight-defined encoding read ``g_E`` predicts ``L03.H04``'s transport of
new cues; its systematic residual is the net contribution of layers 1–2. This module tests a zero-parameter
candidate for that contribution: the two MLPs of blocks 1 and 2 evaluated at the cue position on the
encoding difference ``ΔE = E(w) − E(ref_T)`` with attention held at the reference and the base state taken
from exposed reference prompts (``Δ̂₁ = MLP₁(ln2₁(x̄₁ + ΔE)) − MLP₁(ln2₁(x̄₁))``,
``Δ̂₂ = MLP₂(ln2₂(x̄₂ + ΔE + Δ̂₁)) − MLP₂(ln2₂(x̄₂))``). Because the block-0 MLP is parallel to block-0
attention, the E-patch changes the cue position's residual before block 1 by exactly ``ΔE``, so block 1's
part is exact given the base state and block 2's part omits only block 1's attention change. Predictions
for a frozen set of new cue tokens and new frames are committed before any fresh forward pass (the lock
phase touches weights, the Experiment 011 lock, and locked base states only); the single confirmation
measures the exact net layer change of those cues (Experiment 010's identity, heads included) and scores it
against the committed numbers in ordering, absolute error, and explained variance (Y1), and the composite in
the head's units against the measured P1 fraction (Y2). Tolerances are calibrated leave-one-frame-out on the
exposed pool. Constants are copied from design revision 2.
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

from . import cue_decompilation as cd
from . import cue_suppression as cs
from . import encoding_read as er
from . import head_transport as ht
from . import plural_mechanism as pm
from . import read_assembly as ra
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 2).

EXPERIMENT_DIR = "experiments/012-layer-correction-token-local"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
INHERITED_010_LEDGER_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-010-component-fractions.json"
INHERITED_011_LEDGER_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-011-component-fractions.json"
EXPERIMENT_011_LOCK_PATH = er.LOCK_RELATIVE_PATH
EXPERIMENT_011_CONFIRMATION_PATH = er.CONFIRMATION_RELATIVE_PATH
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260924
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
INHERITED_LEDGER_SCHEMA_VERSION = 1
PHASES = ("explore", "lock", "confirm", "report")

Y1_SPEARMAN = 0.80
Y1_R2 = 0.50  # explained variance of the fresh token means; frozen from the exposed leave-one-frame-out value 0.76
Y2_SPEARMAN = 0.90
TAU_MIN = 0.10
TAU_MULTIPLIER = 3.0
DENOMINATOR_RELATIVE_FLOOR = er.DENOMINATOR_RELATIVE_FLOOR
MIN_VALID_FRAMES = 4
MIN_VALID_FRAMES_PER_TOKEN = 3
MIN_SCORED_TOKENS = 16
FRAME_CUE_EFFECT_RATE = 108 / 120
LOCK_PREDICTION_TOLERANCE = 1e-9
REPLICATION_TOLERANCE = 1e-6
AXIS_COSINE_TOLERANCE = 1e-6  # recomputed stage axes against the Experiment 011 lock
SIGMA_TOLERANCE = 1e-9
READ_WEIGHT_TOLERANCE = 1e-9
LADDER_IDENTITY_TOLERANCE = 1e-4  # relative: ‖Δ̂ − Δout‖ / max(‖Δout‖, ‖out_ref‖) for block 1 at the own base and for block 2 at the captured patched input
NEURON_IDENTITY_TOLERANCE = 1e-4  # relative: ‖Σ_j Δa_j W_out[j] − Δout‖ / max(‖Δout‖, ‖out_ref‖) for each MLP
SIGMA_R_TOLERANCE = 1e-9  # σ_r over the 30 exposed frames against the Experiment 011 lock (identical by equal template weights)
SUBSAMPLE_SIZE = 24
SUBSAMPLE_DRAWS = 5000
SUBSAMPLE_SEED = 20260916
SUBSAMPLE_PERCENTILES = (1, 5, 10, 50)
LAYERS = (1, 2)
MLP_KEYS = ("L01.MLP", "L02.MLP")
HEAD_KEYS = tuple(key for key in ra.COMPONENT_ORDER if key not in MLP_KEYS)
LAYER_OF_MLP = {"L01.MLP": 1, "L02.MLP": 2}
TOP_NEURONS = 20
TOP_NEURONS_PER_PAIR = 10

CANDIDATES: dict[str, tuple[str, ...]] = {
    "determiner-like": ("half", "whichever", "whatever", "which", "what", "same", "own", "last", "next", "first"),
    "numeral": ("fifty", "sixty", "seventy", "eighty", "ninety", "zero", "thousand", "million", "billion"),
    "quantity": ("scarce", "least", "myriad", "manifold", "sparse", "limited", "surplus", "endless"),
    "possessive-or-pronoun": ("hers", "theirs", "ours", "thy", "whom"),
    "adjective": ("black", "white", "young", "empty", "wooden", "tall", "thick", "quiet", "broken"),
}
QUOTAS = {"determiner-like": 5, "numeral": 5, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 5}
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The warehouse ships {cue}"),
    ("cardinal", "The florist arranges {cue}"),
    ("quantifier", "The lecture reviews {cue}"),
    ("quantifier", "The brochure advertises {cue}"),
    ("coordinated-adjective", "Hugo and Amara folded {cue} flat"),
    ("coordinated-adjective", "Nadia and Erik hauled {cue} dusty"),
)
FRAME_ID_TAG = "012"
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", "experiments/011-encoding-read-prospective/", "experiments/010-read-direction-assembly/", "experiments/009-head-transport-rule/",
                            "experiments/008-cue-suppression-localization/", "experiments/007-supervised-cue-subspace/", "experiments/006-low-rank-cue-decompilation/",
                            "experiments/005-regular-plural-mechanism/", "screening/behavior-candidates/manifest-v1.json")
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)
OUTCOME_Y1 = ("LAYER_CORRECTION_TOKEN_LOCAL_MLP", "LAYER_CORRECTION_NOT_TOKEN_LOCAL")
OUTCOME_Y2 = ("COMPOSITE_PREDICTS_P1", "COMPOSITE_FAILS_P1")


class PhaseError(pm.PhaseError):
    pass


# ---------------------------------------------------------------------------
# The MLP weights of blocks 1–2 and the token-local model.


@dataclass(frozen=True)
class LayerWeights:
    """Detached float64 copies of blocks 1–2's MLP LayerNorm and MLP weights."""

    eps: float
    layers: tuple[int, ...]
    ln_w: Mapping[int, torch.Tensor]
    ln_b: Mapping[int, torch.Tensor]
    W_in: Mapping[int, torch.Tensor]  # [d_model, d_mlp]
    b_in: Mapping[int, torch.Tensor]
    W_out: Mapping[int, torch.Tensor]  # [d_mlp, d_model]
    b_out: Mapping[int, torch.Tensor]

    @classmethod
    def from_model(cls, model: Any, layers: Sequence[int] = LAYERS) -> "LayerWeights":
        cfg = model.cfg
        if str(cfg.act_fn) != "gelu":
            raise pm.IncidentError(f"unsupported activation {cfg.act_fn}; the token-local model assumes exact GELU")
        if str(getattr(cfg, "normalization_type", "LN")) != "LN":
            raise pm.IncidentError("the token-local model requires LayerNorm normalization")

        def grab(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().to("cpu", torch.float64).clone()

        blocks = {layer: model.blocks[layer] for layer in layers}
        return cls(float(cfg.eps), tuple(layers), {l: grab(b.ln2.w) for l, b in blocks.items()}, {l: grab(b.ln2.b) for l, b in blocks.items()},
                   {l: grab(b.mlp.W_in) for l, b in blocks.items()}, {l: grab(b.mlp.b_in) for l, b in blocks.items()},
                   {l: grab(b.mlp.W_out) for l, b in blocks.items()}, {l: grab(b.mlp.b_out) for l, b in blocks.items()})

    def hidden(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        normed = pm.exact_layer_norm(x.double(), self.ln_w[layer], self.ln_b[layer], self.eps)
        return torch.nn.functional.gelu(normed @ self.W_in[layer] + self.b_in[layer])

    def out(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(layer, x) @ self.W_out[layer] + self.b_out[layer]

    def delta_out(self, layer: int, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.out(layer, base.double() + delta.double()) - self.out(layer, base.double())

    def input_direction(self, layer: int, neuron: int) -> torch.Tensor:
        """The residual-space direction a neuron reads (its LayerNorm gain times its input weight)."""
        return self.ln_w[layer] * self.W_in[layer][:, neuron]


def propagate(lw: LayerWeights, base: tuple[torch.Tensor, torch.Tensor], delta_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Δ̂₁, Δ̂₂ of the token-local model: MLP₁ on ΔE at the block-1 base, MLP₂ on ΔE + Δ̂₁ at the block-2 base; attention held fixed."""
    d1 = lw.delta_out(1, base[0], delta_e)
    d2 = lw.delta_out(2, base[1], delta_e.double() + d1)
    return d1, d2


@dataclass(frozen=True)
class CorrectionRead:
    """The inherited inner read r(x) = ⟨x − mean(x), γ₃ ⊙ m⟩ (Experiment 011's locked weight) applied to the token-local model."""

    weight: torch.Tensor  # γ₃ ⊙ m (float64)
    e_direction: torch.Tensor  # d̂_E (float64)
    reference_ids: Mapping[str, int]
    plural_ids: Mapping[str, int]

    def inner(self, x: torch.Tensor) -> float:
        x = x.double()
        return float((x - x.mean()) @ self.weight)

    def rows(self, matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix.double()
        return (matrix - matrix.mean(dim=1, keepdim=True)) @ self.weight

    def encoding_delta(self, weights: pm.Weights, token_id: int, template: str) -> torch.Tensor:
        return pm.lexicon_vector(weights, token_id).double() - pm.lexicon_vector(weights, self.reference_ids[template]).double()

    def denominator(self, weights: pm.Weights, template: str) -> float:
        """r(E(pl_T) − E(ref_T)): the weight-only denominator shared by every Y1 quantity."""
        return self.inner(self.encoding_delta(weights, self.plural_ids[template], template))

    def evaluate(self, lw: LayerWeights, base: tuple[torch.Tensor, torch.Tensor], delta: torch.Tensor, denominator: float) -> dict[str, float]:
        d1, d2 = propagate(lw, base, delta)
        r1, r2 = self.inner(d1), self.inner(d2)
        return {"c_hat": (r1 + r2) / denominator, "c_mlp1": r1 / denominator, "c_mlp2": r2 / denominator, "r_delta": r1 + r2}

    def predict(self, weights: pm.Weights, lw: LayerWeights, base: tuple[torch.Tensor, torch.Tensor], token_id: int, template: str) -> dict[str, float]:
        """ĉ and its companions for one token and template at one base state; g_E and q̂' = g_E + ĉ beside them."""
        delta = self.encoding_delta(weights, token_id, template)
        denominator = self.denominator(weights, template)
        whole = self.evaluate(lw, base, delta, denominator)
        parallel = (delta @ self.e_direction) * self.e_direction
        c_par = self.evaluate(lw, base, parallel, denominator)["c_hat"]
        c_perp = self.evaluate(lw, base, delta - parallel, denominator)["c_hat"]
        g_E = self.inner(delta) / denominator
        return {"g_E": g_E, "c_hat": whole["c_hat"], "c_mlp1": whole["c_mlp1"], "c_mlp2": whole["c_mlp2"], "c_par": c_par, "c_perp": c_perp,
                "interaction": whole["c_hat"] - c_par - c_perp, "q_hat_prime": g_E + whole["c_hat"], "r_E": self.inner(delta), "r_delta": whole["r_delta"]}

    def plural_total(self, weights: pm.Weights, lw: LayerWeights, base: tuple[torch.Tensor, torch.Tensor], template: str) -> dict[str, float]:
        """D̂_T = r(E(pl_T) − E(ref_T)) + r(Δ̂₁ + Δ̂₂)(pl_T): the plural cue's modelled total, Y2's denominator."""
        denominator = self.denominator(weights, template)
        plural = self.evaluate(lw, base, self.encoding_delta(weights, self.plural_ids[template], template), denominator)
        return {"denominator": denominator, "c_hat_plural": plural["c_hat"], "D_hat": denominator + plural["r_delta"]}


def composite(prediction: Mapping[str, float], d_hat: float) -> float:
    """q̂ = [r(ΔE) + r(Δ̂₁ + Δ̂₂)] / D̂_T, the composite in the head's units."""
    return (prediction["r_E"] + prediction["r_delta"]) / d_hat


def read_from_lock_011(lock_011: Mapping[str, Any], reference_ids: Mapping[str, int], plural_ids: Mapping[str, int]) -> CorrectionRead:
    return CorrectionRead(torch.tensor(lock_011["read_weight"], dtype=torch.float64), torch.tensor(lock_011["axes_vectors"]["R0"], dtype=torch.float64), dict(reference_ids), dict(plural_ids))


def locked_axes(lock_011: Mapping[str, Any], recomputed: Mapping[str, pm.SiteAxis]) -> tuple[pm.SiteAxis, pm.SiteAxis]:
    """The Experiment 011 lock's d̂_T, σ_T and d̂_E as SiteAxis objects, after checking the recomputed stage axes agree with them (incident otherwise)."""
    direction_T = torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64)
    direction_E = torch.tensor(lock_011["axes_vectors"]["R0"], dtype=torch.float64)
    if pm.cosine(recomputed["T"].direction, direction_T) < 1.0 - AXIS_COSINE_TOLERANCE or pm.cosine(recomputed["R0"].direction, direction_E) < 1.0 - AXIS_COSINE_TOLERANCE:
        raise pm.IncidentError("the recomputed stage axes differ from the Experiment 011 lock's directions")
    if abs(recomputed["T"].sigma - float(lock_011["sigma_T"])) > SIGMA_TOLERANCE:
        raise pm.IncidentError("the recomputed head-stage scale σ_T differs from the Experiment 011 lock")
    axis_T = pm.SiteAxis("T", recomputed["T"].mu, direction_T, float(lock_011["sigma_T"]))
    e_axis = pm.SiteAxis("R0", recomputed["R0"].mu, direction_E, recomputed["R0"].sigma)
    return axis_T, e_axis


def check_read_weight(read: CorrectionRead, head: ht.HeadWeights, axis_T: pm.SiteAxis) -> float:
    difference = float((read.weight - ra.read_weight(head, axis_T).double()).abs().max())
    if difference > READ_WEIGHT_TOLERANCE:
        raise pm.IncidentError(f"the head weights do not reproduce the Experiment 011 locked read weight (max difference {difference:.3e})")
    return difference


# ---------------------------------------------------------------------------
# Base states.


def mean_state(states: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    if not states:
        raise PhaseError("a base state needs at least one frame")
    return torch.stack([x1.double() for x1, _ in states]).mean(dim=0), torch.stack([x2.double() for _, x2 in states]).mean(dim=0)


def template_bases(frames: Sequence[pm.Frame], states: Mapping[str, tuple[torch.Tensor, torch.Tensor]], *, exclude: str | None = None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Template means of the cue position's reference residuals before blocks 1 and 2; ``exclude`` leaves one frame out (calibration)."""
    bases = {}
    for template in pm.TEMPLATE_ORDER:
        chosen = [states[frame.frame_id] for frame in frames if frame.template_id == template and frame.frame_id != exclude and frame.frame_id in states]
        if chosen:
            bases[template] = mean_state(chosen)
    return bases


def bases_to_json(bases: Mapping[str, tuple[torch.Tensor, torch.Tensor]], counts: Mapping[str, int]) -> dict[str, Any]:
    return {template: {"x1": x1.tolist(), "x2": x2.tolist(), "n_frames": int(counts[template])} for template, (x1, x2) in bases.items()}


def bases_from_json(entry: Mapping[str, Any]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {template: (torch.tensor(value["x1"], dtype=torch.float64), torch.tensor(value["x2"], dtype=torch.float64)) for template, value in entry.items()}


# ---------------------------------------------------------------------------
# Frame capture and pair measurement.


@dataclass
class FrameState:
    ref: ht.FrameReference
    components: dict[str, torch.Tensor]
    functional: ra.ReadFunctional
    x1: torch.Tensor  # RESID_PRE.L1 at p_c in the reference prompt (float64)
    x2: torch.Tensor  # RESID_PRE.L2 at p_c


def extra_sites(frame: pm.Frame) -> list[pm.Site]:
    """The cue position's residuals before blocks 1 and 2 and the two MLP outputs (the latter already among the component sites; captured in the same run)."""
    return [("RESID_PRE.L1", frame.p_c), ("RESID_PRE.L2", frame.p_c), ("L01.MLP", frame.p_c), ("L02.MLP", frame.p_c)]


def capture_frame(model: Any, head: ht.HeadWeights, reference: pm.Prompt, nouns: Sequence[pm.Noun], axis_T: pm.SiteAxis) -> FrameState:
    """One reference forward: the head's internals, the 18 components, and the residuals before blocks 1 and 2 at p_c, all from the same run."""
    p_c = reference.frame.p_c
    ref, components = ra.capture_reference(model, head, reference, nouns, extra_sites=[("RESID_PRE.L1", p_c), ("RESID_PRE.L2", p_c)])
    x1 = components.pop(pm.site_label(("RESID_PRE.L1", p_c)))
    x2 = components.pop(pm.site_label(("RESID_PRE.L2", p_c)))
    if not torch.equal(x1, ref.vectors["R0"].double()):
        raise pm.IncidentError(f"{reference.frame.frame_id}: the captured residual before block 1 disagrees with the R0 stage vector of the same run")
    return FrameState(ref, components, ra.read_functional(head, ref, axis_T), x1, x2)


def measure_pair(model: Any, weights: pm.Weights, head: ht.HeadWeights, state: FrameState, name: str, token_id: int, e_axis: pm.SiteAxis, axis_T: pm.SiteAxis, nouns: Sequence[pm.Noun]) -> ra.Attribution:
    return ra.measure_token_010(model, weights, head, state.ref, state.components, state.functional, name, token_id, e_axis, axis_T, nouns, extra_sites=extra_sites(state.ref.frame))


def relative_vector_error(candidate: torch.Tensor, measured_delta: torch.Tensor, reference_output: torch.Tensor) -> float:
    """‖candidate − Δout‖ / max(‖Δout‖, ‖out_ref‖): float32 capture noise is ~1e-6 of the output's size; a wrong LayerNorm, block or activation is of order one."""
    scale = max(float(measured_delta.double().norm()), float(reference_output.double().norm()), 1e-12)
    return float((candidate.double() - measured_delta.double()).norm()) / scale


def neuron_ledger(read: CorrectionRead, lw: LayerWeights, layer: int, x_ref: torch.Tensor, x_patch: torch.Tensor, denominator: float, *, delta_out: torch.Tensor, out_ref: torch.Tensor, where: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact per-neuron terms Δa_j · r(W_out[j]) / r(E(pl) − E(ref)) from the captured inputs; Σ_j Δa_j W_out[j] must equal the captured Δout (vector identity, relative)."""
    delta_a = lw.hidden(layer, x_patch) - lw.hidden(layer, x_ref)
    identity_error = relative_vector_error(delta_a @ lw.W_out[layer], delta_out, out_ref)
    if identity_error > NEURON_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{where}: block-{layer} neuron terms do not sum to the captured MLP output change (relative error {identity_error:.2e})")
    terms = delta_a * read.rows(lw.W_out[layer]) / denominator
    concentration = ra.neuron_concentration(terms)
    concentration["top"] = concentration["top"][:TOP_NEURONS_PER_PAIR]
    concentration["identity_error"] = identity_error
    concentration["projected_sum_error"] = abs(float(terms.sum()) - read.inner(delta_out) / denominator)
    return terms, concentration


def analyse_pair(record: ra.Attribution, plural: ra.Attribution, *, read: CorrectionRead, lw: LayerWeights, weights: pm.Weights, state: FrameState, axis_T: pm.SiteAxis) -> dict[str, Any] | None:
    """Every measured quantity of one (token, frame) pair in Y1's units, Experiment 010's fractions beside them, the own-base ladder, and the neuron ledgers."""
    fractions_010 = ra.fractions(record, plural, axis_T, plural.g_E_inner)
    if record.own_reference or fractions_010 is None:
        return None
    frame = state.ref.frame
    template = frame.template_id
    identities = er.enforce_identities(record, plural, f"{frame.frame_id}/{record.token}")
    denominator = read.denominator(weights, template)
    unit = state.functional.scale * denominator
    c_k = {key: record.rho_components[key] / unit for key in ra.COMPONENT_ORDER}
    c_M = sum(c_k[key] for key in MLP_KEYS)
    c_H = sum(c_k[key] for key in HEAD_KEYS)
    own_base = (state.x1, state.x2)
    own = read.predict(weights, lw, own_base, record.token_id, template)
    own_total = read.plural_total(weights, lw, own_base, template)
    x1_patch = record.extra[pm.site_label(("RESID_PRE.L1", frame.p_c))]
    x2_patch = record.extra[pm.site_label(("RESID_PRE.L2", frame.p_c))]
    delta_out = {key: record.extra[pm.site_label((key, frame.p_c))] - state.components[key] for key in MLP_KEYS}
    where = f"{frame.frame_id}/{record.token}"
    # Block 1 at the frame's own base is exact (Δx₁ = ΔE); block 2 is reproduced from its captured patched input. Both as relative vector identities.
    delta_e = read.encoding_delta(weights, record.token_id, template)
    own_delta_1, _ = propagate(lw, own_base, delta_e)
    mlp1_exact_error = relative_vector_error(own_delta_1, delta_out["L01.MLP"], state.components["L01.MLP"])
    mlp2_identity_error = relative_vector_error(lw.delta_out(2, state.x2, x2_patch - state.x2), delta_out["L02.MLP"], state.components["L02.MLP"])
    if mlp1_exact_error > LADDER_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{where}: block 1's MLP change is not the token-local evaluation at the frame's own base (relative error {mlp1_exact_error:.2e})")
    if mlp2_identity_error > LADDER_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{where}: block 2's MLP change is not reproduced from its captured input (relative error {mlp2_identity_error:.2e})")
    neurons = {}
    for key, layer, x_ref, x_patch in (("L01.MLP", 1, state.x1, x1_patch), ("L02.MLP", 2, state.x2, x2_patch)):
        _, neurons[key] = neuron_ledger(read, lw, layer, x_ref, x_patch, denominator, delta_out=delta_out[key], out_ref=state.components[key], where=where)
    return {"token": record.token, "token_id": record.token_id, "frame_id": frame.frame_id, "template": template,
            "c_L": c_M + c_H, "c_M": c_M, "c_H": c_H, "c_k": c_k, "g_E": record.g_E_inner / denominator, "P1_prime": record.rho_total / unit,
            "P1": record.rho_total / plural.rho_total, "q_T": fractions_010["q_T"],
            "fractions_010": {"f_E": fractions_010["f_E"], "f_total": fractions_010["f_total"], "f_layers": fractions_010["f_layers"], "G": fractions_010["G"], "f_k": dict(fractions_010["f_k"])},
            "own": {**own, "D_hat": own_total["D_hat"], "q_hat": composite(own, own_total["D_hat"])},
            "ladder": {"mlp1_exact_error": mlp1_exact_error, "mlp2_identity_error": mlp2_identity_error, "mlp1_projected_error": abs(own["c_mlp1"] - c_k["L01.MLP"]), "attention_input_term": c_M - own["c_hat"]},
            "neurons": neurons, "identities": identities}


def neuron_terms_for_mass(read: CorrectionRead, lw: LayerWeights, layer: int, x_ref: torch.Tensor, x_patch: torch.Tensor, denominator: float) -> torch.Tensor:
    delta_a = lw.hidden(layer, x_patch) - lw.hidden(layer, x_ref)
    return (delta_a * read.rows(lw.W_out[layer]) / denominator).abs()


class NeuronMass:
    """Running absolute attribution mass per neuron and template, for the top-20 summaries."""

    def __init__(self) -> None:
        self.mass: dict[str, dict[str, torch.Tensor]] = {key: {} for key in MLP_KEYS}
        self.count: dict[str, dict[str, int]] = {key: {} for key in MLP_KEYS}

    def add(self, key: str, template: str, terms_abs: torch.Tensor) -> None:
        if template not in self.mass[key]:
            self.mass[key][template] = torch.zeros_like(terms_abs)
            self.count[key][template] = 0
        self.mass[key][template] += terms_abs
        self.count[key][template] += 1

    def summary(self, read: CorrectionRead, lw: LayerWeights) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in MLP_KEYS:
            layer = LAYER_OF_MLP[key]
            per_template = {}
            for template, mass in self.mass[key].items():
                mean_mass = mass / max(self.count[key][template], 1)
                order = sorted(range(mean_mass.numel()), key=lambda j: (-float(mean_mass[j]), j))[:TOP_NEURONS]
                per_template[template] = {"n_pairs": self.count[key][template],
                                          "top": [{"neuron": j, "mean_abs_mass": float(mean_mass[j]), "cos_with_d_E": pm.cosine(lw.input_direction(layer, j), read.e_direction)} for j in order]}
            templates = list(per_template)
            overlap = {f"{a}|{b}": ra.jaccard([e["neuron"] for e in per_template[a]["top"]], [e["neuron"] for e in per_template[b]["top"]]) for i, a in enumerate(templates) for b in templates[i + 1:]}
            out[key] = {"per_template": per_template, "template_overlap_jaccard": overlap}
        return out


# ---------------------------------------------------------------------------
# Pool (87 tokens × 30 frames) and the confirmation set.


def build_pool_012(manifest: ScreeningManifest, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009, confirmation_011: er.Confirmation011) -> cs.Pool008:
    base = ra.build_pool_010(manifest, extension, confirmation_006, confirmation_009)
    frames = base.frames + tuple(confirmation_011.frames)
    origin = dict(base.frame_origin) | {frame.frame_id: "confirmation-011" for frame in confirmation_011.frames}
    tokens = list(base.tokens)
    category, source = dict(base.token_category), dict(base.token_source)
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation_011.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"Experiment 011 token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-011"
    return cs.Pool008(frames, origin, tuple(tokens), category, source, base.nouns, base.noun_source, base.reference_ids, base.plural_cue)


def fresh_tokens_012(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
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


def build_confirmation_payload(tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> dict[str, Any]:
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_012(tokenizer, excluded_ids)
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
    token_prompts = []
    for frame in frame_objects:
        for token in tokens:
            ids = frame.prompt_ids(token["token_id"])
            text = pm._decode(tokenizer, ids)
            if pm._encode(tokenizer, text) != ids:
                raise ValueError(f"{frame.frame_id}/{token['word']}: text does not round-trip to the constructed token IDs")
            token_prompts.append({"frame_id": frame.frame_id, "word": token["word"], "token_id": token["token_id"], "text": text, "token_ids": list(ids), "p_c": frame.p_c, "p_t": frame.p_t})
    payload = {"schema_version": CONFIRMATION_SCHEMA_VERSION, "manifest_sha256": digests["manifest"], "extension_sha256": digests["extension"],
               "confirmation_006_sha256": digests["confirmation_006"], "confirmation_009_sha256": digests["confirmation_009"], "confirmation_011_sha256": digests["confirmation_011"],
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "reference_cue_ids": dict(pool.reference_ids),
               "exposed_token_ids": sorted(token_id for _, token_id in pool.tokens), "policy": {"licensing": "every fresh cue in every fresh frame", "quotas": dict(QUOTAS), "expectations": "none: the committed ĉ and q̂ values are the only predictions"},
               "tokens": tokens, "frames": frames, "token_prompts": token_prompts, "construction": "tokenizer-only; first eligible candidates per lexical class; classes carry no expectation; no model output"}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


@dataclass(frozen=True)
class Confirmation012:
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


def validate_confirmation(payload: Mapping[str, Any], pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation012:
    pm._require_exact_keys(payload, {"schema_version", "manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "model", "reference_cue_ids",
                                     "exposed_token_ids", "policy", "tokens", "frames", "token_prompts", "construction", "content_sha256"}, "confirmation")
    if payload["schema_version"] != CONFIRMATION_SCHEMA_VERSION:
        raise ValueError("confirmation schema version is not frozen")
    if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("confirmation content_sha256 does not match canonical payload")
    recorded = (payload["manifest_sha256"], payload["extension_sha256"], payload["confirmation_006_sha256"], payload["confirmation_009_sha256"], payload["confirmation_011_sha256"])
    if recorded != (digests["manifest"], digests["extension"], digests["confirmation_006"], digests["confirmation_009"], digests["confirmation_011"]):
        raise ValueError("confirmation was built against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}:
        raise ValueError("confirmation was built for a different pinned model")
    if dict(payload["reference_cue_ids"]) != dict(pool.reference_ids) or list(payload["exposed_token_ids"]) != sorted(token_id for _, token_id in pool.tokens):
        raise ValueError("reference cues or exposed token IDs disagree with the exposed pool")
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
    frames_by_id = {frame.frame_id: frame for frame in frames}
    prompts = []
    for entry in payload["token_prompts"]:
        pm._require_exact_keys(entry, {"frame_id", "word", "token_id", "text", "token_ids", "p_c", "p_t"}, "token_prompt")
        frame = frames_by_id[entry["frame_id"]]
        token = next(token for token in tokens if token["word"] == entry["word"])
        if token["token_id"] != entry["token_id"] or tuple(entry["token_ids"]) != frame.prompt_ids(entry["token_id"]) or (entry["p_c"], entry["p_t"]) != (frame.p_c, frame.p_t):
            raise ValueError(f"{entry['frame_id']}/{entry['word']}: stored prompt disagrees with the frame or the policy")
        prompts.append(pm.Prompt(frame, int(entry["token_id"]), entry["word"]))
    expected_prompts = [(frame.frame_id, token["word"]) for frame in frames for token in tokens]
    if [(prompt.frame.frame_id, prompt.cue_label) for prompt in prompts] != expected_prompts:
        raise ValueError("token prompts must cover exactly the (frame, token) pairs in order")
    return Confirmation012(dict(payload["reference_cue_ids"]), frames, tuple(tokens), tuple(prompts), payload["content_sha256"])


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


def load_confirmation(path: Path, pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation012:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, pool, digests)


# ---------------------------------------------------------------------------
# Inherited per-component ledgers (derived extracts of Experiments 010 and 011).

LEDGER_FIELDS = ("f_E", "f_total", "f_layers", "q_T")


def ledger_entry(fractions_010: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if fractions_010 is None:
        return None
    return {"f_k": {key: float(fractions_010["f_k"][key]) for key in ra.COMPONENT_ORDER}, **{key: float(fractions_010[key]) for key in LEDGER_FIELDS}}


def inherited_ledger_payload(entries: Mapping[str, Mapping[str, Any] | None], *, experiment: str, source: Mapping[str, Any], digests: Mapping[str, str]) -> dict[str, Any]:
    payload = {"schema_version": INHERITED_LEDGER_SCHEMA_VERSION, "experiment": experiment,
               "description": f"Derived extract of Experiment {experiment}'s recorded per-component fractions (f_k for the 18 components of layers 1–2, f_E, f_total, f_layers, q_T) per (token, frame), normalized by the plural cue's measured head change (null where the frame was uninformative for the token). Experiment 012 recomputes these and requires agreement within 1e-6.",
               "source": dict(source), "manifest_sha256": digests["manifest"], "extension_sha256": digests["extension"], "confirmation_006_sha256": digests["confirmation_006"],
               "confirmation_009_sha256": digests["confirmation_009"], "confirmation_011_sha256": digests["confirmation_011"], "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
               "entries": {key: (None if value is None else dict(value)) for key, value in sorted(entries.items())}}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_ledger(path: Path, *, experiment: str, digests: Mapping[str, str], expected_size: int) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited {experiment} ledger: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "experiment", "description", "source", "manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "model", "entries", "content_sha256"}, f"inherited {experiment} ledger")
    if payload["schema_version"] != INHERITED_LEDGER_SCHEMA_VERSION or payload["experiment"] != experiment or payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError(f"inherited {experiment} ledger schema or digest is not frozen")
    recorded = (payload["manifest_sha256"], payload["extension_sha256"], payload["confirmation_006_sha256"], payload["confirmation_009_sha256"], payload["confirmation_011_sha256"])
    if recorded != (digests["manifest"], digests["extension"], digests["confirmation_006"], digests["confirmation_009"], digests["confirmation_011"]):
        raise ValueError(f"inherited {experiment} ledger was recorded against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision} or len(payload["entries"]) != expected_size:
        raise ValueError(f"inherited {experiment} ledger model or size is not frozen")
    return payload


def check_ledger_replication(measured: Mapping[str, Mapping[str, Any] | None], recorded: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
    """Every recorded (token, frame) pair must be recomputed with the same informativeness and every field within 1e-6."""
    missing = set(recorded) - set(measured)
    if missing:
        raise pm.IncidentError(f"recomputed ledger lacks {len(missing)} recorded pairs (e.g. {sorted(missing)[:3]})")
    worst_key, worst = None, 0.0
    for key, value in recorded.items():
        other = measured[key]
        if (value is None) != (other is None):
            raise pm.IncidentError(f"{key}: informativeness differs from the record")
        if value is None:
            continue
        for name in LEDGER_FIELDS:
            deviation = abs(value[name] - other[name])
            if deviation > worst:
                worst_key, worst = f"{key}:{name}", deviation
        for component in ra.COMPONENT_ORDER:
            deviation = abs(value["f_k"][component] - other["f_k"][component])
            if deviation > worst:
                worst_key, worst = f"{key}:{component}", deviation
    if worst > REPLICATION_TOLERANCE:
        raise pm.IncidentError(f"{worst_key}: recomputed fraction deviates from the record by {worst:.3e}")
    return {"passed": True, "n": len(recorded), "max_abs_deviation": worst, "worst_key": worst_key}


# ---------------------------------------------------------------------------
# Results state, phases, statistics.

_STATE_KEYS = {"schema_version", "run_id", "created_at", "manifest_sha256", "extension_sha256", "confirmation_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256",
               "lock_011_sha256", "protocol_code_commit", "git_dirty", "model", "versions", "phases", "executed_prompt_keys", "executed_noun_keys", "exploration", "lock", "confirmation", "invalidated_runs", "state_sha256"}


def new_results_state(*, digests: Mapping[str, str], protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {"schema_version": RESULTS_SCHEMA_VERSION, "run_id": pm.sha256_text("".join(digests[key] for key in sorted(digests)) + protocol_code_commit + pm.utc_now())[:16], "created_at": pm.utc_now(),
            "manifest_sha256": digests["manifest"], "extension_sha256": digests["extension"], "confirmation_sha256": digests["confirmation_006"], "confirmation_009_sha256": digests["confirmation_009"],
            "confirmation_011_sha256": digests["confirmation_011"], "confirmation_012_sha256": digests["confirmation_012"], "lock_011_sha256": digests["lock_011"],
            "protocol_code_commit": protocol_code_commit, "git_dirty": False, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "versions": dict(versions),
            "phases": {phase: {"status": "not_started"} for phase in PHASES}, "executed_prompt_keys": [], "executed_noun_keys": [], "exploration": {}, "lock": None, "confirmation": None, "invalidated_runs": []}


def state_digests(state: Mapping[str, Any]) -> tuple[str, ...]:
    return (state["manifest_sha256"], state["extension_sha256"], state["confirmation_sha256"], state["confirmation_009_sha256"], state["confirmation_011_sha256"], state["confirmation_012_sha256"], state["lock_011_sha256"])


def digest_tuple(digests: Mapping[str, str]) -> tuple[str, ...]:
    return (digests["manifest"], digests["extension"], digests["confirmation_006"], digests["confirmation_009"], digests["confirmation_011"], digests["confirmation_012"], digests["lock_011"])


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


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation012) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


def tau(residuals: Sequence[float]) -> float:
    return max(TAU_MIN, TAU_MULTIPLIER * math.sqrt(pm._mean([value * value for value in residuals]))) if residuals else TAU_MIN


def explained_variance(predicted: Sequence[float], measured: Sequence[float]) -> float | None:
    """R² = 1 − Σ(p − m)² / Σ(m − mean m)²; None when the measured values have no spread."""
    if len(measured) < 2:
        return None
    mean = pm._mean(list(measured))
    variance = sum((m - mean) ** 2 for m in measured)
    if variance <= 0.0:
        return None
    return 1.0 - sum((p - m) ** 2 for p, m in zip(predicted, measured)) / variance


def comparison(predicted: Sequence[float], measured: Sequence[float]) -> dict[str, Any]:
    pairs = [(p, m) for p, m in zip(predicted, measured)]
    if not pairs:
        return {"n": 0, "spearman": None, "mae": None, "rmse": None, "r2": None, "bias": None}
    residuals = [p - m for p, m in pairs]
    return {"n": len(pairs), "spearman": pm.spearman([p for p, _ in pairs], [m for _, m in pairs]) if len(pairs) >= 2 else None, "mae": pm._mean([abs(r) for r in residuals]),
            "rmse": math.sqrt(pm._mean([r * r for r in residuals])), "r2": explained_variance([p for p, _ in pairs], [m for _, m in pairs]), "bias": pm._mean(residuals)}


def subsample_distribution(predicted_by_token: Mapping[str, float], measured_by_token: Mapping[str, float], *, size: int = SUBSAMPLE_SIZE, draws: int = SUBSAMPLE_DRAWS, seed: int = SUBSAMPLE_SEED) -> dict[str, Any]:
    """Percentiles of the explained variance over seeded random ``size``-token subsamples (the record beside the frozen floor)."""
    names = sorted(predicted_by_token)
    if len(names) <= size:
        return {"n_tokens": len(names), "size": size, "draws": 0, "percentiles": {}, "note": "no subsampling: at most `size` tokens"}
    rng = random.Random(seed)
    values, undefined = [], 0
    for _ in range(draws):
        chosen = rng.sample(names, size)
        r2 = explained_variance([predicted_by_token[n] for n in chosen], [measured_by_token[n] for n in chosen])
        if r2 is None:
            undefined += 1
        else:
            values.append(r2)
    values.sort()
    percentiles = {str(q): values[min(len(values) - 1, int(q / 100 * len(values)))] for q in SUBSAMPLE_PERCENTILES} if values else {}
    return {"n_tokens": len(names), "size": size, "draws": draws, "undefined_draws": undefined, "percentiles": percentiles}


def token_mean(values_by_frame: Mapping[str, float | None]) -> float | None:
    return cs._mean_or_none(list(values_by_frame.values()))


# ---------------------------------------------------------------------------
# Exploration (Tier A): measurements, replication, base states, leave-one-frame-out calibration.


def _frame_denominators(read: CorrectionRead, weights: pm.Weights, lw: LayerWeights, bases: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, dict[str, float]]:
    return {template: read.plural_total(weights, lw, bases[template], template) for template in pm.TEMPLATE_ORDER if template in bases}


def run_exploration(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, *, lock_011: Mapping[str, Any], inherited: Mapping[str, Mapping[str, Any]], state: dict[str, Any], results_path: Path | None, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = LayerWeights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    nouns = pool.single_nouns
    exploration = state["exploration"]
    say("stage axes from the Experiment 010 pool, checked against the Experiment 011 lock")
    axis_T, e_axis = locked_axes(lock_011, cs.stage_axes(cache, weights, pool_010))
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = read_from_lock_011(lock_011, pool.reference_ids, plural_ids)
    exploration["read_weight_check"] = check_read_weight(read, head, axis_T)
    denominators_E = {template: read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}
    for template, value in denominators_E.items():
        if abs(value - float(lock_011["denominators"]["denominators"][template])) > READ_WEIGHT_TOLERANCE:
            raise pm.IncidentError(f"{template}: the weight-only denominator differs from the Experiment 011 lock")
    exploration["axes_vectors"] = {"T": axis_T.direction.tolist(), "R0": e_axis.direction.tolist()}
    exploration["sigma_T"] = axis_T.sigma
    exploration["read_weight"] = read.weight.tolist()
    exploration["denominators_E"] = denominators_E
    say(f"measurements: {len(pool.tokens)} tokens × {len(pool.frames)} frames with the residuals before blocks 1 and 2 captured")
    states: dict[str, FrameState] = {}
    residuals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    pairs: dict[str, dict[str, Any]] = {}
    ledger: dict[str, dict[str, Any] | None] = {}
    identities: dict[str, float] = {}
    mass = NeuronMass()
    for frame in pool.frames:
        state_f = capture_frame(model, head, pool.reference_prompt(frame), nouns, axis_T)
        states[frame.frame_id] = state_f
        residuals[frame.frame_id] = (state_f.x1, state_f.x2)
        records = {name: measure_pair(model, weights, head, state_f, name, token_id, e_axis, axis_T, nouns) for name, token_id in pool.tokens}
        plural = records[pool.plural_cue[frame.template_id]]
        for name, record in records.items():
            analysis = analyse_pair(record, plural, read=read, lw=lw, weights=weights, state=state_f, axis_T=axis_T)
            key = f"{name}|{frame.frame_id}"
            ledger[key] = ledger_entry(ra.fractions(record, plural, axis_T, plural.g_E_inner))
            if analysis is None:
                continue
            er._max_errors(identities, analysis["identities"])
            pairs[key] = analysis
            denominator = denominators_E[frame.template_id]
            for mlp_key, layer, x_ref in (("L01.MLP", 1, state_f.x1), ("L02.MLP", 2, state_f.x2)):
                mass.add(mlp_key, frame.template_id, neuron_terms_for_mass(read, lw, layer, x_ref, record.extra[pm.site_label((f"RESID_PRE.L{layer}", frame.p_c))], denominator))
        say(f"  {frame.frame_id}: {sum(1 for name, _ in pool.tokens if f'{name}|{frame.frame_id}' in pairs)} informative pairs")
    exploration["identities"] = identities
    exploration["replication"] = {"experiment_010": check_ledger_replication(ledger, inherited["010"]["entries"]), "experiment_011": check_ledger_replication(ledger, inherited["011"]["entries"])}
    say(f"replication: 010 max deviation {exploration['replication']['experiment_010']['max_abs_deviation']:.2e}; 011 {exploration['replication']['experiment_011']['max_abs_deviation']:.2e}")
    # Base states: all frames (locked) and leave-one-frame-out (calibration).
    full_bases = template_bases(pool.frames, residuals)
    counts = {template: len(pool.frames_of(template)) for template in pm.TEMPLATE_ORDER}
    exploration["base_states"] = bases_to_json(full_bases, counts)
    lofo_bases = {frame.frame_id: template_bases(pool.frames, residuals, exclude=frame.frame_id)[frame.template_id] for frame in pool.frames}
    full_totals = _frame_denominators(read, weights, lw, full_bases)
    sigma_r = er.sigma_r_from_pairs(read, weights, pool.frames)
    if abs(sigma_r - float(lock_011["denominators"]["sigma_r"])) > SIGMA_R_TOLERANCE:
        raise pm.IncidentError(f"σ_r over the 30 exposed frames ({sigma_r:.6f}) differs from the Experiment 011 lock ({float(lock_011['denominators']['sigma_r']):.6f})")
    validity_E = er.denominator_validity(denominators_E, sigma_r)
    validity_D = er.denominator_validity({template: full_totals[template]["D_hat"] for template in pm.TEMPLATE_ORDER}, sigma_r)
    defined = {template: bool(validity_E["defined"][template] and validity_D["defined"][template]) for template in pm.TEMPLATE_ORDER}
    exploration["denominators"] = {"weight_only": validity_E, "modelled_total": validity_D, "plural_totals": full_totals, "defined": defined, "sigma_r": sigma_r}
    say(f"denominators r(ΔE_pl) {denominators_E}; modelled totals {[round(v['D_hat'], 4) for v in full_totals.values()]}; defined {defined}")
    # Leave-one-frame-out predictions for every informative pair; all-frames predictions per (token, template).
    lofo_totals = {frame.frame_id: read.plural_total(weights, lw, lofo_bases[frame.frame_id], frame.template_id) for frame in pool.frames}
    per_token: dict[str, dict[str, Any]] = {}
    for name, token_id in pool.tokens:
        frames_of_token = {}
        for frame in pool.frames:
            analysis = pairs.get(f"{name}|{frame.frame_id}")
            if analysis is None or not defined[frame.template_id]:
                continue
            lofo = read.predict(weights, lw, lofo_bases[frame.frame_id], token_id, frame.template_id)
            lofo["q_hat"] = composite(lofo, lofo_totals[frame.frame_id]["D_hat"])
            analysis["lofo"] = lofo
            frames_of_token[frame.frame_id] = analysis
        full = {template: read.predict(weights, lw, full_bases[template], token_id, template) for template in pm.TEMPLATE_ORDER if defined[template]}
        for template, prediction in full.items():
            prediction["q_hat"] = composite(prediction, full_totals[template]["D_hat"])
        means = _token_means(frames_of_token, full)
        per_token[name] = {"token": name, "token_id": token_id, "category": pool.token_category[name], "source": pool.token_source[name], "n_frames": len(frames_of_token), "frames": sorted(frames_of_token),
                           "predictions_full": full, **means}
    exploration["pairs"] = pairs
    exploration["tokens"] = per_token
    scored = [row for row in per_token.values() if row["n_frames"] > 0]
    if len(scored) < 2:
        raise pm.IncidentError("fewer than two exposed tokens have an informative frame in a defined template; no calibration is possible")
    c_hat_lofo = {row["token"]: row["c_hat_lofo_mean"] for row in scored}
    c_L = {row["token"]: row["c_L_mean"] for row in scored}
    tau_c = tau([c_hat_lofo[n] - c_L[n] for n in c_hat_lofo])
    tau_P = tau([row["q_hat_lofo_mean"] - row["P1_mean"] for row in scored])
    exploration["tolerances"] = {"tau_c": tau_c, "tau_P": tau_P, "n_tokens": len(scored), "calibration": "leave-one-frame-out template bases"}
    names = sorted(c_hat_lofo)
    exploration["exposed_check"] = {
        "y1_lofo": comparison([c_hat_lofo[n] for n in names], [c_L[n] for n in names]),
        "y1_lofo_vs_c_M": comparison([c_hat_lofo[n] for n in names], [per_token[n]["c_M_mean"] for n in names]),
        "y1_full": comparison([per_token[n]["c_hat_full_mean"] for n in names], [c_L[n] for n in names]),
        "y1_own": comparison([per_token[n]["c_hat_own_mean"] for n in names], [c_L[n] for n in names]),
        "y2_lofo": comparison([per_token[n]["q_hat_lofo_mean"] for n in names], [per_token[n]["P1_mean"] for n in names]),
        "y2_full": comparison([per_token[n]["q_hat_full_mean"] for n in names], [per_token[n]["P1_mean"] for n in names]),
        "g_E_vs_P1": comparison([per_token[n]["g_E_mean"] for n in names], [per_token[n]["P1_mean"] for n in names]),
        "g_E_vs_q_T": comparison([per_token[n]["g_E_mean"] for n in names], [per_token[n]["q_T_mean"] for n in names]),
        "q_hat_lofo_vs_q_T": comparison([per_token[n]["q_hat_lofo_mean"] for n in names], [per_token[n]["q_T_mean"] for n in names]),
        "pairs_y1_lofo": comparison([a["lofo"]["c_hat"] for a in pairs.values() if "lofo" in a], [a["c_L"] for a in pairs.values() if "lofo" in a]),
        "pairs_y2_lofo": comparison([a["lofo"]["q_hat"] for a in pairs.values() if "lofo" in a], [a["P1"] for a in pairs.values() if "lofo" in a]),
        "per_template_pairs_y1_lofo": {template: comparison([a["lofo"]["c_hat"] for a in pairs.values() if "lofo" in a and a["template"] == template], [a["c_L"] for a in pairs.values() if "lofo" in a and a["template"] == template]) for template in pm.TEMPLATE_ORDER},
        "heads_share": _heads_share(pairs.values()),
        "ladder": {"base_point_lofo_mean": pm._mean([row["base_point_lofo_mean"] for row in scored]), "attention_input_mean": pm._mean([row["attention_input_mean"] for row in scored]),
                   "mean_abs_base_point_lofo": pm._mean([abs(row["base_point_lofo_mean"]) for row in scored]), "mean_abs_attention_input": pm._mean([abs(row["attention_input_mean"]) for row in scored])},
        "r2_lofo_subsamples": subsample_distribution(c_hat_lofo, c_L),
        "c_L_spread": {"mean": pm._mean(list(c_L.values())), "sd": math.sqrt(pm._mean([(v - pm._mean(list(c_L.values()))) ** 2 for v in c_L.values()]))} if c_L else None,
        "class_means": _class_means(per_token.values()),
    }
    exploration["neurons"] = mass.summary(read, lw)
    exploration["summary"] = {"defined_templates": [template for template, ok in defined.items() if ok], "tau_c": tau_c, "tau_P": tau_P, "r2_lofo": exploration["exposed_check"]["y1_lofo"]["r2"],
                              "spearman_lofo": exploration["exposed_check"]["y1_lofo"]["spearman"], "prediction_undefined": sum(defined.values()) < 2}
    say(f"tau_c {tau_c:.3f}, tau_P {tau_P:.3f}; leave-one-frame-out Spearman {exploration['summary']['spearman_lofo']:.3f}, R² {exploration['summary']['r2_lofo']:.3f}")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


def _heads_share(analyses: Any) -> dict[str, float | None]:
    c_H = [abs(a["c_H"]) for a in analyses]
    c_M = [abs(a["c_M"]) for a in analyses]
    if not c_M:
        return {"mean_abs_c_H": None, "mean_abs_c_M": None, "share": None}
    return {"mean_abs_c_H": pm._mean(c_H), "mean_abs_c_M": pm._mean(c_M), "share": pm._mean(c_H) / pm._mean(c_M) if pm._mean(c_M) > 0 else None}


def _class_means(rows: Any) -> dict[str, dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("n_frames", 0) > 0:
            by_class.setdefault(row["category"], []).append(row)
    out = {}
    for category, members in sorted(by_class.items()):
        entry = {"n": len(members), "c_L_mean": pm._mean([m["c_L_mean"] for m in members])}
        for key in ("c_hat_lofo_mean", "c_hat_locked_mean", "c_hat_full_mean"):
            if all(m.get(key) is not None for m in members):
                entry[key] = pm._mean([m[key] for m in members])
        out[category] = entry
    return out


def _token_means(frames_of_token: Mapping[str, Mapping[str, Any]], predictions_by_template: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Means over one frame set F(w) of every measured quantity and of the template-level predictions over the same frames."""
    if not frames_of_token:
        return {key: None for key in ("c_L_mean", "c_M_mean", "c_H_mean", "P1_mean", "P1_prime_mean", "q_T_mean", "g_E_mean", "f_layers_010_mean", "c_hat_own_mean", "q_hat_own_mean", "attention_input_mean",
                                      "c_hat_lofo_mean", "q_hat_lofo_mean", "base_point_lofo_mean", "c_hat_full_mean", "q_hat_full_mean", "c_par_full_mean", "c_perp_full_mean", "interaction_full_mean", "c_mlp1_full_mean", "c_mlp2_full_mean")}
    values = list(frames_of_token.values())
    mean = lambda picker: pm._mean([picker(a) for a in values])  # noqa: E731
    out = {"c_L_mean": mean(lambda a: a["c_L"]), "c_M_mean": mean(lambda a: a["c_M"]), "c_H_mean": mean(lambda a: a["c_H"]), "P1_mean": mean(lambda a: a["P1"]), "P1_prime_mean": mean(lambda a: a["P1_prime"]),
           "q_T_mean": mean(lambda a: a["q_T"]), "g_E_mean": mean(lambda a: a["g_E"]), "f_layers_010_mean": mean(lambda a: a["fractions_010"]["f_layers"]),
           "c_hat_own_mean": mean(lambda a: a["own"]["c_hat"]), "q_hat_own_mean": mean(lambda a: a["own"]["q_hat"]), "attention_input_mean": mean(lambda a: a["ladder"]["attention_input_term"])}
    if all("lofo" in a for a in values):
        out["c_hat_lofo_mean"] = mean(lambda a: a["lofo"]["c_hat"])
        out["q_hat_lofo_mean"] = mean(lambda a: a["lofo"]["q_hat"])
        out["base_point_lofo_mean"] = out["c_hat_own_mean"] - out["c_hat_lofo_mean"]
    else:
        out["c_hat_lofo_mean"] = out["q_hat_lofo_mean"] = out["base_point_lofo_mean"] = None
    templated = [predictions_by_template[a["template"]] for a in values]
    for key, name in (("c_hat", "c_hat_full_mean"), ("q_hat", "q_hat_full_mean"), ("c_par", "c_par_full_mean"), ("c_perp", "c_perp_full_mean"), ("interaction", "interaction_full_mean"), ("c_mlp1", "c_mlp1_full_mean"), ("c_mlp2", "c_mlp2_full_mean")):
        out[name] = pm._mean([p[key] for p in templated])
    return out


# ---------------------------------------------------------------------------
# Lock (weights + locked axes + locked base states), the prediction artifact, validation, reproduction.


def lock_predictions(weights: pm.Weights, lw: LayerWeights, read: CorrectionRead, bases: Mapping[str, tuple[torch.Tensor, torch.Tensor]], confirmation: Confirmation012, defined: Mapping[str, bool]) -> dict[str, Any]:
    """Every fresh token's ĉ (with parts, ∥/⊥ evaluations), g_E, q̂', and q̂ per defined template, and the licensed-frame means — from weights and locked states, no forward pass."""
    totals = {template: read.plural_total(weights, lw, bases[template], template) for template in pm.TEMPLATE_ORDER if defined.get(template)}
    tokens = {}
    for token in confirmation.tokens:
        word, token_id = token["word"], token["token_id"]
        by_template = {}
        for template in pm.TEMPLATE_ORDER:
            if not defined.get(template):
                continue
            prediction = read.predict(weights, lw, bases[template], token_id, template)
            prediction["q_hat"] = composite(prediction, totals[template]["D_hat"])
            by_template[template] = prediction
        licensed = [frame for frame in confirmation.frames_for(word) if frame.template_id in by_template]
        means = {key: pm._mean([by_template[frame.template_id][key] for frame in licensed]) for key in ("c_hat", "c_mlp1", "c_mlp2", "c_par", "c_perp", "interaction", "g_E", "q_hat_prime", "q_hat")} if licensed else {}
        tokens[word] = {"token_id": token_id, "category": token["category"], "licensed_frames": [frame.frame_id for frame in licensed], "by_template": by_template, "means": means}
    return {"tokens": tokens, "plural_totals": totals}


def render_predictions(lock: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 012 — preregistered predictions (token-local MLP model of the layers-1–2 correction)", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_012_sha256']}`; Experiment 011 lock sha256 `{lock['lock_011_sha256']}`",
             f"- τ_c {f(lock['tolerances']['tau_c'], 3)} (Y1 MAE ceiling, token means, leave-one-frame-out); τ_P {f(lock['tolerances']['tau_P'], 3)} (Y2); floors Spearman ≥ {Y1_SPEARMAN} and R² ≥ {Y1_R2} (Y1), Spearman ≥ {Y2_SPEARMAN} (Y2)",
             f"- Defined templates: {lock['defined_templates']}; weight-only denominators {lock['denominators']['weight_only']['denominators']}; modelled plural totals D̂_T {{{', '.join(f'{t}: {f(v['D_hat'], 4)}' for t, v in lock['predictions']['plural_totals'].items())}}}", "",
             "| token | class | template | predicted ĉ | MLP₁ part | MLP₂ part | ĉ(ΔE_∥) | ĉ(ΔE_⊥) | interaction | g_E | q̂' = g_E + ĉ | q̂ (head's units) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for word, entry in lock["predictions"]["tokens"].items():
        for template, p in entry["by_template"].items():
            lines.append(f"| {word} | {entry['category']} | {template} | {f(p['c_hat'], 3)} | {f(p['c_mlp1'], 3)} | {f(p['c_mlp2'], 3)} | {f(p['c_par'], 3)} | {f(p['c_perp'], 3)} | {f(p['interaction'], 3)} | {f(p['g_E'], 3)} | {f(p['q_hat_prime'], 3)} | {f(p['q_hat'], 3)} |")
        m = entry["means"]
        lines.append(f"| **{word}** | | **mean over {len(entry['licensed_frames'])} licensed frames** | **{f(m.get('c_hat'), 3)}** | {f(m.get('c_mlp1'), 3)} | {f(m.get('c_mlp2'), 3)} | {f(m.get('c_par'), 3)} | {f(m.get('c_perp'), 3)} | {f(m.get('interaction'), 3)} | {f(m.get('g_E'), 3)} | {f(m.get('q_hat_prime'), 3)} | **{f(m.get('q_hat'), 3)}** |")
    lines.append("")
    return "\n".join(lines)


def build_candidate_lock(*, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation012, predictions: Mapping[str, Any], protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    if exploration["summary"]["prediction_undefined"]:
        raise PhaseError("fewer than two templates have defined denominators: PREDICTION_UNDEFINED, no lock")
    lock = {"schema_version": 1, "experiment": "012", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit,
            "manifest_sha256": digests["manifest"], "extension_sha256": digests["extension"], "confirmation_sha256": digests["confirmation_006"], "confirmation_009_sha256": digests["confirmation_009"],
            "confirmation_011_sha256": digests["confirmation_011"], "confirmation_012_sha256": confirmation.content_sha256, "lock_011_sha256": digests["lock_011"],
            "inherited_ledgers_sha256": {"010": digests["ledger_010"], "011": digests["ledger_011"]},
            "confirmation_prompt_keys": sorted(prompt.key for prompt in confirmation.all_prompts), "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED},
            "head": ht.HEAD_KEY, "axes_vectors": exploration["axes_vectors"], "sigma_T": exploration["sigma_T"], "read_weight": exploration["read_weight"], "base_states": exploration["base_states"],
            "denominators": exploration["denominators"], "defined_templates": exploration["summary"]["defined_templates"], "tolerances": exploration["tolerances"], "exposed_check": exploration["exposed_check"],
            "exposed_neurons": exploration["neurons"],
            "floors": {"y1_spearman": Y1_SPEARMAN, "y1_r2": Y1_R2, "y2_spearman": Y2_SPEARMAN, "min_valid_frames": MIN_VALID_FRAMES, "min_valid_frames_per_token": MIN_VALID_FRAMES_PER_TOKEN, "min_scored_tokens": MIN_SCORED_TOKENS,
                       "frame_cue_effect_rate": FRAME_CUE_EFFECT_RATE, "head_stage_floor": cs.STAGE_UNINFORMATIVE_FLOOR},
            "predictions": dict(predictions), "tier_a_results_sha256": state.get("state_sha256")}
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation012, predictions_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "012" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked or git_state.get("dirty", True):
        raise PhaseError("confirm requires the committed lock and predictions on a clean Git tree")
    recorded = (lock["manifest_sha256"], lock["extension_sha256"], lock["confirmation_sha256"], lock["confirmation_009_sha256"], lock["confirmation_011_sha256"], lock["confirmation_012_sha256"], lock["lock_011_sha256"])
    if recorded != (digests["manifest"], digests["extension"], digests["confirmation_006"], digests["confirmation_009"], digests["confirmation_011"], confirmation.content_sha256, digests["lock_011"]):
        raise PhaseError("lock was built against different frozen inputs")
    if dict(lock.get("inherited_ledgers_sha256", {})) != {"010": digests["ledger_010"], "011": digests["ledger_011"]}:
        raise PhaseError("lock was built against different inherited ledgers")
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
        for template, prediction in entry["by_template"].items():
            for key, value in prediction.items():
                worst = max(worst, abs(value - fresh["by_template"][template][key]))
        for key, value in entry["means"].items():
            worst = max(worst, abs(value - fresh["means"][key]))
    for template, total in lock["predictions"]["plural_totals"].items():
        for key, value in total.items():
            worst = max(worst, abs(value - recomputed["plural_totals"][template][key]))
    if worst > LOCK_PREDICTION_TOLERANCE:
        raise PhaseError(f"the weights, locked axes and locked base states do not reproduce the preregistered predictions (max difference {worst:.3e}); nothing was executed")
    return worst


# ---------------------------------------------------------------------------
# Confirmation (Tier C), validity, floors, outcome, report.


def run_confirmation(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation012, lock: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = LayerWeights.from_model(model)
    nouns = pool.single_nouns
    cache = pm.PromptCache(model, tuple(pool.nouns))
    say("stage axes (Experiment 010 pool) against the lock")
    axis_T, e_axis = locked_axes(lock, cs.stage_axes(cache, weights, pool_010))
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    read = read_from_lock_011(lock, pool.reference_ids, plural_ids)
    check_read_weight(read, head, axis_T)
    denominators_E = {template: read.denominator(weights, template) for template in pm.TEMPLATE_ORDER}
    frames_out: dict[str, Any] = {}
    per_token_frames: dict[str, dict[str, dict[str, Any]]] = {token["word"]: {} for token in confirmation.tokens}
    identities: dict[str, float] = {}
    mass = NeuronMass()
    say("fresh frames: reference, plural cue (validity and normalization), cue pairs, then every fresh token")
    for frame in confirmation.frames:
        template = frame.template_id
        if template not in lock["defined_templates"]:
            frames_out[frame.frame_id] = {"template_id": template, "valid": False, "template_defined": False, "note": "template excluded before the lock; nothing run"}
            say(f"  {frame.frame_id}: template {template} undefined; skipped")
            continue
        state_f = capture_frame(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        sg, pl = pm.frame_prompts(frame)
        c_a, c_b = cache.c(sg), cache.c(pl)
        positive = sum(1 for noun in nouns if c_a[noun.lexical_key] - c_b[noun.lexical_key] > 0)
        required = pm.exact_count_floor(FRAME_CUE_EFFECT_RATE, len(nouns))
        plural = measure_pair(model, weights, head, state_f, pool.plural_cue[template], pl.cue_token_id, e_axis, axis_T, nouns)
        er._max_errors(identities, er.enforce_identities(plural, plural, f"{frame.frame_id}/{pool.plural_cue[template]}"))
        head_informative = abs(plural.head_change) >= cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma
        valid = head_informative and positive >= required
        frames_out[frame.frame_id] = {"template_id": template, "valid": valid, "head_informative": head_informative, "plural_head_change": plural.head_change, "cue_effect_positive": positive, "cue_effect_required": required,
                                      "template_defined": True, "reconstruction_error": state_f.ref.reconstruction_error, "attention_ref": state_f.functional.attention_ref, "sigma_ref": state_f.functional.sigma_ref,
                                      "own_base_plural_total": read.plural_total(weights, lw, (state_f.x1, state_f.x2), template)["D_hat"], "measured_plural_total_r": plural.rho_total / state_f.functional.scale}
        if not valid:
            say(f"  {frame.frame_id}: INVALID (head informative {head_informative}, cue effect {positive}/{required})")
            continue
        for token in confirmation.tokens:
            if frame.frame_id not in token["licensed_frames"]:
                continue
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair(record, plural, read=read, lw=lw, weights=weights, state=state_f, axis_T=axis_T)
            if analysis is None:
                raise pm.IncidentError(f"{frame.frame_id}/{token['word']}: no analysis although the frame is valid")
            er._max_errors(identities, analysis["identities"])
            c_word = cache.c(pm.Prompt(frame, token["token_id"], token["word"]))
            analysis["dc_beh"] = pm._mean([c_word[noun.lexical_key] - state_f.ref.c_by_noun[noun.lexical_key] for noun in nouns])
            analysis["dc"] = pm._mean(list(record.shifts.values()))
            for mlp_key, layer, x_ref in (("L01.MLP", 1, state_f.x1), ("L02.MLP", 2, state_f.x2)):
                mass.add(mlp_key, template, neuron_terms_for_mass(read, lw, layer, x_ref, record.extra[pm.site_label((f"RESID_PRE.L{layer}", frame.p_c))], denominators_E[template]))
            per_token_frames[token["word"]][frame.frame_id] = analysis
        say(f"  {frame.frame_id}: valid; {sum(1 for token in confirmation.tokens if frame.frame_id in token['licensed_frames'])} tokens")
    results = score_confirmation(frames_out, per_token_frames, confirmation, lock)
    results["identities"] = identities
    results["neurons"] = mass.summary(read, lw)
    results["neuron_overlap_exposed_vs_fresh"] = _neuron_overlap(lock.get("exposed_neurons", {}), results["neurons"])
    return results


def _neuron_overlap(exposed: Mapping[str, Any], fresh: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in MLP_KEYS:
        out[key] = {}
        for template, entry in fresh.get(key, {}).get("per_template", {}).items():
            exposed_top = [e["neuron"] for e in exposed.get(key, {}).get("per_template", {}).get(template, {}).get("top", [])]
            out[key][template] = ra.jaccard(exposed_top, [e["neuron"] for e in entry["top"]])
    return out


def confirmation_frame_template(confirmation: Confirmation012, frame_id: str) -> str:
    return next(frame.template_id for frame in confirmation.frames if frame.frame_id == frame_id)


def score_confirmation(frames_out: Mapping[str, Any], per_token_frames: Mapping[str, Mapping[str, Mapping[str, Any]]], confirmation: Confirmation012, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Pure scoring: validity counts, token means over identical frame sets, Y1 (three floors), Y2, descriptive items, outcome."""
    valid_frames = [frame_id for frame_id, entry in frames_out.items() if entry["valid"]]
    tokens_out = {}
    for token in confirmation.tokens:
        word = token["word"]
        frames = per_token_frames[word]
        locked = lock["predictions"]["tokens"][word]
        scored = len(frames) >= MIN_VALID_FRAMES_PER_TOKEN
        templates = {frame_id: confirmation_frame_template(confirmation, frame_id) for frame_id in frames}
        by_template = locked["by_template"]
        means = _token_means(frames, {t: by_template[t] for t in set(templates.values())}) if frames else _token_means({}, {})
        row = {"category": token["category"], "scored": scored, "n_valid_frames": len(frames), "frames": sorted(frames), **means}
        row["c_hat_locked_mean"] = pm._mean([by_template[templates[f]]["c_hat"] for f in frames]) if frames else None
        row["q_hat_locked_mean"] = pm._mean([by_template[templates[f]]["q_hat"] for f in frames]) if frames else None
        row["q_hat_prime_locked_mean"] = pm._mean([by_template[templates[f]]["q_hat_prime"] for f in frames]) if frames else None
        row["base_point_term"] = row["c_hat_own_mean"] - row["c_hat_locked_mean"] if frames else None
        row["dc_mean"] = pm._mean([a["dc"] for a in frames.values()]) if frames and all("dc" in a for a in frames.values()) else None
        row["dc_beh_mean"] = pm._mean([a["dc_beh"] for a in frames.values()]) if frames and all("dc_beh" in a for a in frames.values()) else None
        tokens_out[word] = row
    scored_tokens = [word for word, entry in tokens_out.items() if entry["scored"]]
    results: dict[str, Any] = {"frames": frames_out, "valid_frames": valid_frames, "tokens": tokens_out, "scored_tokens": scored_tokens, "per_frame": per_token_frames}
    precondition_ok = len(valid_frames) >= MIN_VALID_FRAMES and len(scored_tokens) >= MIN_SCORED_TOKENS
    results["precondition"] = {"passed": precondition_ok, "valid_frames": len(valid_frames), "scored_tokens": len(scored_tokens), "min_valid_frames": MIN_VALID_FRAMES, "min_scored_tokens": MIN_SCORED_TOKENS}
    if precondition_ok:
        c_hat = [tokens_out[w]["c_hat_locked_mean"] for w in scored_tokens]
        c_L = [tokens_out[w]["c_L_mean"] for w in scored_tokens]
        y1 = comparison(c_hat, c_L)
        tau_c = lock["tolerances"]["tau_c"]
        failing = [name for name, ok in (("spearman", y1["spearman"] >= Y1_SPEARMAN), ("mae", y1["mae"] <= tau_c), ("r2", y1["r2"] is not None and y1["r2"] >= Y1_R2)) if not ok]
        results["Y1"] = {**y1, "tau_c": tau_c, "r2_floor": Y1_R2, "passed": not failing, "failing": failing}
        q_hat = [tokens_out[w]["q_hat_locked_mean"] for w in scored_tokens]
        p1 = [tokens_out[w]["P1_mean"] for w in scored_tokens]
        y2 = comparison(q_hat, p1)
        tau_P = lock["tolerances"]["tau_P"]
        failing_2 = [name for name, ok in (("spearman", y2["spearman"] >= Y2_SPEARMAN), ("mae", y2["mae"] <= tau_P)) if not ok]
        results["Y2"] = {**y2, "tau_P": tau_P, "passed": not failing_2, "failing": failing_2}
        pairs = [a for w in scored_tokens for a in per_token_frames[w].values()]
        results["descriptive"] = {
            "y1_vs_c_M": comparison(c_hat, [tokens_out[w]["c_M_mean"] for w in scored_tokens]),
            "q_hat_prime_vs_P1_prime": comparison([tokens_out[w]["q_hat_prime_locked_mean"] for w in scored_tokens], [tokens_out[w]["P1_prime_mean"] for w in scored_tokens]),
            "own_base_vs_c_L": comparison([tokens_out[w]["c_hat_own_mean"] for w in scored_tokens], c_L),
            "g_E_vs_P1": comparison([tokens_out[w]["g_E_mean"] for w in scored_tokens], p1),
            "g_E_vs_q_T": comparison([tokens_out[w]["g_E_mean"] for w in scored_tokens], [tokens_out[w]["q_T_mean"] for w in scored_tokens]),
            "q_hat_vs_q_T": comparison(q_hat, [tokens_out[w]["q_T_mean"] for w in scored_tokens]),
            "heads_share": _heads_share(pairs), "exposed_heads_share": lock.get("exposed_check", {}).get("heads_share"),
            "ladder": {"base_point_mean": pm._mean([tokens_out[w]["base_point_term"] for w in scored_tokens]), "attention_input_mean": pm._mean([tokens_out[w]["attention_input_mean"] for w in scored_tokens]),
                       "mean_abs_base_point": pm._mean([abs(tokens_out[w]["base_point_term"]) for w in scored_tokens]), "mean_abs_attention_input": pm._mean([abs(tokens_out[w]["attention_input_mean"]) for w in scored_tokens]),
                       "heads_mean": pm._mean([tokens_out[w]["c_H_mean"] for w in scored_tokens])},
            "pairs_y1": comparison([a["lofo"]["c_hat"] if "lofo" in a else lock["predictions"]["tokens"][a["token"]]["by_template"][a["template"]]["c_hat"] for a in pairs], [a["c_L"] for a in pairs]),
            "class_means": _class_means(tokens_out[w] | {"n_frames": tokens_out[w]["n_valid_frames"]} for w in scored_tokens),
            "exposed_c_L_spread": lock.get("exposed_check", {}).get("c_L_spread"),
        }
    results["outcome"] = outcome(results)
    return results


def outcome(results: Mapping[str, Any]) -> dict[str, Any]:
    if not results["precondition"]["passed"]:
        return {"label": "PRECONDITION_FAILED", "Y1": None, "Y2": None}
    y1 = OUTCOME_Y1[0] if results["Y1"]["passed"] else OUTCOME_Y1[1]
    y2 = OUTCOME_Y2[0] if results["Y2"]["passed"] else OUTCOME_Y2[1]
    return {"label": f"{y1} | {y2}", "Y1": y1, "Y2": y2}


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 012 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state['confirmation_012_sha256']}`", f"- Experiment 011 lock sha256: `{state['lock_011_sha256']}`",
             f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    incidents = list(exploration.get("incidents", [])) + ([state["confirmation"]["incident"]] if isinstance(state.get("confirmation"), dict) and "incident" in state["confirmation"] else [])
    if incidents:
        lines += ["## Incidents", ""] + [f"- `{e['phase']}` at commit `{e.get('commit', '?')}` ({e['at']}): {e['message']}" for e in incidents] + [""]
    if "summary" in exploration:
        s, x = exploration["summary"], exploration["exposed_check"]
        rep = exploration["replication"]
        lines += ["## Tier A — exposed pool (calibration only)", "",
                  f"- Replication: Experiment 010 {rep['experiment_010']['n']} pairs, max deviation {rep['experiment_010']['max_abs_deviation']:.2e}; Experiment 011 {rep['experiment_011']['n']} pairs, max deviation {rep['experiment_011']['max_abs_deviation']:.2e}",
                  f"- Identities (max relative errors): {', '.join(f'{k} {v:.1e}' for k, v in exploration['identities'].items())}; read weight vs Experiment 011 lock {exploration['read_weight_check']:.1e}",
                  f"- Weight-only denominators {exploration['denominators_E']}; modelled plural totals D̂_T {{{', '.join(f'{t}: {f(v['D_hat'], 3)}' for t, v in exploration['denominators']['plural_totals'].items())}}}; σ_r {f(exploration['denominators']['sigma_r'], 3)}; defined {s['defined_templates']}",
                  f"- τ_c {f(s['tau_c'], 3)}, τ_P {f(s['tau_P'], 3)} (leave-one-frame-out token means, {exploration['tolerances']['n_tokens']} tokens)",
                  f"- Y1 calibration (ĉ̄ vs c̄_L, leave-one-frame-out): Spearman {f(x['y1_lofo']['spearman'], 3)}, MAE {f(x['y1_lofo']['mae'], 3)}, R² {f(x['y1_lofo']['r2'], 3)}, bias {f(x['y1_lofo']['bias'], 3)}; all-frames base {f(x['y1_full']['spearman'], 3)}/{f(x['y1_full']['mae'], 3)}/{f(x['y1_full']['r2'], 3)}; own base {f(x['y1_own']['spearman'], 3)}/{f(x['y1_own']['mae'], 3)}/{f(x['y1_own']['r2'], 3)}; c̄_L spread sd {f(x['c_L_spread']['sd'], 3) if x.get('c_L_spread') else '—'}",
                  f"- R²_LOFO over {x['r2_lofo_subsamples'].get('size')}-token subsamples: percentiles {{{', '.join(f'{k}: {f(v, 3)}' for k, v in x['r2_lofo_subsamples'].get('percentiles', {}).items())}}} (frozen floor {Y1_R2})",
                  f"- Y2 calibration (q̂̄ vs P̄1, leave-one-frame-out): Spearman {f(x['y2_lofo']['spearman'], 3)}, MAE {f(x['y2_lofo']['mae'], 3)}, R² {f(x['y2_lofo']['r2'], 3)}; g_E vs P̄1 {f(x['g_E_vs_P1']['spearman'], 3)}/{f(x['g_E_vs_P1']['mae'], 3)}; g_E vs q̄_T {f(x['g_E_vs_q_T']['spearman'], 3)}/{f(x['g_E_vs_q_T']['mae'], 3)}; q̂ vs q̄_T {f(x['q_hat_lofo_vs_q_T']['spearman'], 3)}/{f(x['q_hat_lofo_vs_q_T']['mae'], 3)}",
                  f"- Pairs (leave-one-frame-out): Y1 {f(x['pairs_y1_lofo']['spearman'], 3)}/{f(x['pairs_y1_lofo']['mae'], 3)}; per template {{{', '.join(f'{t}: {f(v['spearman'], 3)}/{f(v['mae'], 3)}' for t, v in x['per_template_pairs_y1_lofo'].items())}}}",
                  f"- Heads' share mean|c_H|/mean|c_M| {f(x['heads_share']['share'], 3)}; ladder means: base-point (own − LOFO) {f(x['ladder']['base_point_lofo_mean'], 3)} (|.| {f(x['ladder']['mean_abs_base_point_lofo'], 3)}), attention-input {f(x['ladder']['attention_input_mean'], 3)} (|.| {f(x['ladder']['mean_abs_attention_input'], 3)})",
                  f"- Class means (c̄_L, ĉ̄_LOFO): {{{', '.join(f'{k}: {f(v['c_L_mean'], 3)}/{f(v.get('c_hat_lofo_mean'), 3)}' for k, v in x['class_means'].items())}}}", ""]
        for key, entry in exploration.get("neurons", {}).items():
            lines.append(f"- `{key}` top-{TOP_NEURONS} neurons by mean |term|: " + "; ".join(f"{t}: {[e['neuron'] for e in v['top'][:10]]}…" for t, v in entry["per_template"].items()) + f"; template overlap Jaccard {{{', '.join(f'{k}: {f(v, 2)}' for k, v in entry['template_overlap_jaccard'].items())}}}")
        lines.append("")
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}`; predictions sha256 `{state['lock']['predictions_sha256']}`", ""]
    confirmation = state.get("confirmation")
    if confirmation and "outcome" in confirmation:
        o, pre = confirmation["outcome"], confirmation["precondition"]
        lines += [f"## Confirmation — `{o['label']}`", "", f"- Valid frames {pre['valid_frames']}/6 (min {pre['min_valid_frames']}); scored tokens {pre['scored_tokens']} (min {pre['min_scored_tokens']})"]
        for frame_id, entry in confirmation["frames"].items():
            if entry.get("template_defined"):
                lines.append(f"  - {frame_id}: {'valid' if entry['valid'] else 'INVALID'} (plural head change {f(entry['plural_head_change'], 3)}, cue effect {entry['cue_effect_positive']}/{entry['cue_effect_required']}; own-base plural total {f(entry['own_base_plural_total'], 3)} vs measured {f(entry['measured_plural_total_r'], 3)})")
            else:
                lines.append(f"  - {frame_id}: {entry.get('note')}")
        if "Y1" in confirmation:
            y1, y2, d = confirmation["Y1"], confirmation["Y2"], confirmation["descriptive"]
            lines += [f"- Y1 (ĉ̄ vs c̄_L over {y1['n']} scored tokens): Spearman {f(y1['spearman'], 3)} (≥ {Y1_SPEARMAN}), MAE {f(y1['mae'], 3)} (≤ τ_c {f(y1['tau_c'], 3)}), R² {f(y1['r2'], 3)} (≥ {Y1_R2}), bias {f(y1['bias'], 3)} → {'pass' if y1['passed'] else 'FAIL'} {y1['failing'] or ''}",
                      f"- Y2 (q̂̄ vs P̄1 over {y2['n']} scored tokens): Spearman {f(y2['spearman'], 3)} (≥ {Y2_SPEARMAN}), MAE {f(y2['mae'], 3)} (≤ τ_P {f(y2['tau_P'], 3)}), R² {f(y2['r2'], 3)} → {'pass' if y2['passed'] else 'FAIL'} {y2['failing'] or ''}",
                      f"- Descriptive: ĉ̄ vs c̄_M {f(d['y1_vs_c_M']['spearman'], 3)}/{f(d['y1_vs_c_M']['mae'], 3)}/{f(d['y1_vs_c_M']['r2'], 3)}; own-base ĉ̄ vs c̄_L {f(d['own_base_vs_c_L']['spearman'], 3)}/{f(d['own_base_vs_c_L']['mae'], 3)}/{f(d['own_base_vs_c_L']['r2'], 3)}; heads' share {f(d['heads_share']['share'], 3)} (exposed {f((d.get('exposed_heads_share') or {}).get('share'), 3)}); ladder means base-point {f(d['ladder']['base_point_mean'], 3)} (|.| {f(d['ladder']['mean_abs_base_point'], 3)}), attention-input {f(d['ladder']['attention_input_mean'], 3)} (|.| {f(d['ladder']['mean_abs_attention_input'], 3)}), heads {f(d['ladder']['heads_mean'], 3)}",
                      f"- Reported beside (never judged): g_E vs P̄1 {f(d['g_E_vs_P1']['spearman'], 3)}/{f(d['g_E_vs_P1']['mae'], 3)}; g_E vs q̄_T {f(d['g_E_vs_q_T']['spearman'], 3)}/{f(d['g_E_vs_q_T']['mae'], 3)}; q̂ vs q̄_T {f(d['q_hat_vs_q_T']['spearman'], 3)}/{f(d['q_hat_vs_q_T']['mae'], 3)}; q̂' vs P1' {f(d['q_hat_prime_vs_P1_prime']['spearman'], 3)}/{f(d['q_hat_prime_vs_P1_prime']['mae'], 3)}",
                      f"- Class means (c̄_L, locked ĉ̄): {{{', '.join(f'{k}: {f(v['c_L_mean'], 3)}/{f(v.get('c_hat_locked_mean'), 3)}' for k, v in d['class_means'].items())}}}",
                      f"- Block-2 top-{TOP_NEURONS} neuron overlap exposed vs fresh (Jaccard): {{{', '.join(f'{k}: {f(v, 2)}' for k, v in confirmation.get('neuron_overlap_exposed_vs_fresh', {}).get('L02.MLP', {}).items())}}}"]
        lines += ["", "| token | class | scored | valid frames | predicted ĉ | measured c_L | c_M | c_H | own-base ĉ | base-point | attention-input | ĉ(∥) | ĉ(⊥) | g_E | q̂ | P1 | q_T | contrast | behavior |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for word, row in sorted(confirmation["tokens"].items(), key=lambda kv: (kv[1]["c_hat_locked_mean"] if kv[1]["c_hat_locked_mean"] is not None else 9)):
            lines.append(f"| {word} | {row['category']} | {row['scored']} | {row['n_valid_frames']} | {f(row['c_hat_locked_mean'], 3)} | {f(row['c_L_mean'], 3)} | {f(row['c_M_mean'], 3)} | {f(row['c_H_mean'], 3)} | {f(row['c_hat_own_mean'], 3)} | {f(row['base_point_term'], 3)} | {f(row['attention_input_mean'], 3)} | {f(row['c_par_full_mean'], 3)} | {f(row['c_perp_full_mean'], 3)} | {f(row['g_E_mean'], 3)} | {f(row['q_hat_locked_mean'], 3)} | {f(row['P1_mean'], 3)} | {f(row['q_T_mean'], 3)} | {f(row['dc_mean'])} | {f(row['dc_beh_mean'])} |")
        lines.append("")
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
