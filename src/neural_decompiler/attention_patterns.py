"""Experiment 015: the cue-induced attention-pattern change from the decoded query/key computation.

Experiment 013 measured the pattern-change term ``Σ_k ΔA_h(p_c,k) v_h'(k) W_O^h`` of every layer-1–2 head from the
patched run. This module writes the pattern change as a weight-only program — LayerNorm → query/key/value projections
→ partial rotary rotation → scaled dot products → softmax — and evaluates it at two levels. **Level 1** recomputes the
cue rows from the frame's actual reference residuals and ``ΔE``; because positions before the cue are causally
invariant and the parallel residual makes ``x₁'(p_c) = x₁(p_c) + ΔE`` exactly, Level 1 is the model itself: it must
reproduce the captured rows and the captured residual before block 2 within a frozen tolerance (identities I1–I3;
violations are incidents; agreement counts toward no floor). **Level 0** — the hypothesis — is the token-local
cue-change model conditional on the reference frame state: the cue position's query, key and value *changes* are
computed at the Experiment 012 locked template-mean bases from ``ΔE`` alone, and the frame contributes only its
reference keys, values and logit rows at positions ``≤ p_c``. The same-position rotary rotation cancels in the self
logit, so the diagonal change is exactly token-local at the base. The rigid number-axis alternative (the same program
on the ``d̂_E`` component of the exact Level-1 change at the frame's own state) is committed for rejection; the oracle
and Level-0 diagonal-proportional comparators are named descriptive competitors with a predeclared interpretation
rule. Predictions for new cues in the 48 exposed frames are locked before any fresh prompt (Y1); predictions for new
cues in six new frames are computed at confirm stage 1 and digested before any fresh cue prompt (Y2); the alternative
is scored on both (Y3). Constants are copied from design revision 2.
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
from . import neuron_feature as nf
from . import plural_mechanism as pm
from . import read_assembly as ra
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 2).

EXPERIMENT_DIR = "experiments/015-attention-pattern-token-local"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
INHERITED_014_LEDGER_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-014-pair-ledger.json"
INHERITED_013_EXTRACT_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-013-pattern-extract.json"
EXPERIMENT_011_LOCK_PATH = er.LOCK_RELATIVE_PATH
EXPERIMENT_012_LOCK_PATH = lc.LOCK_RELATIVE_PATH
EXPERIMENT_013_LOCK_PATH = ap.LOCK_RELATIVE_PATH
EXPERIMENT_014_LOCK_PATH = nf.LOCK_RELATIVE_PATH
EXPERIMENT_013_CONFIRMATION_PATH = ap.CONFIRMATION_RELATIVE_PATH
EXPERIMENT_014_CONFIRMATION_PATH = nf.CONFIRMATION_RELATIVE_PATH
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260924
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
INHERITED_SCHEMA_VERSION = 1
PHASES = ("explore", "lock", "confirm", "report")

Y_SPEARMAN = 0.80  # token means of ĉ_ΔA against c_ΔA
Y_R2 = 0.50
Y_ENTRY_R2 = 0.50  # pooled entry R² of ΔÂ against ΔA, separately at layer 1 and at layer 2
Y3_R2_MAX = 0.30  # the axis-only alternative is rejected iff its token-mean R² is below this …
Y3_ENTRY_R2_MAX = 0.30  # … and its pooled entry R² over both layers and both sets below this
COMPARATOR_MARGIN = 0.10  # Level 0 decodes the interaction beyond the self logit iff its entry R² exceeds the Level-0 diagonal-proportional comparator's by this at both layers
MIN_VALID_FRAMES = 4
MIN_VALID_FRAMES_PER_TOKEN = 3
MIN_SCORED_TOKENS = 16
FRAME_CUE_EFFECT_RATE = 108 / 120
LOCK_PREDICTION_TOLERANCE = 1e-9
REPLICATION_TOLERANCE = 1e-6
LOCKED_STATE_TOLERANCE = 1e-9
ROW_IDENTITY_TOLERANCE = 1e-4  # I1: recomputed pattern rows against the captured rows (absolute)
CHAIN_IDENTITY_TOLERANCE = 1e-4  # I2: the exact chain against the captured residual before block 2 (relative)
EXPECTED_LEDGER_SIZE_014 = 4884  # Experiment 014's 3732 exposed pairs + its 1008 + 144 confirmed pairs
EXPECTED_EXTRACT_SIZE_013 = 3732  # Experiment 013's 2724 exposed pairs + its 864 + 144 confirmed pairs
HEAD_LAYERS = ap.HEAD_LAYERS
N_HEADS = ap.N_HEADS
HEAD_KEYS = ap.HEAD_KEYS
ROW_MODELS = ("level0", "axis", "diag_L0")  # the committed row predictions of a table entry
RUNGS = ("level1", "diag_oracle", "query_only", "key_only", "additive", "linearised", "frozen_arrival")  # descriptive, own state, exact arriving change
PREDICTION_COLUMNS = ("token", "frame_id", "template", "p_c", "rows_level0", "self_level0", "c_hat_1", "c_hat_2", "c_hat", "rows_axis", "c_axis", "rows_diag_L0", "c_diag_L0", "arrival_read",
                      "c_L_level0", "c_M_level0", "c_H_level0")  # the last three: the fully decoded layer contribution at Level 0 (the re-derived Experiment 013 ladder; descriptive)
LADDER_KEYS = ("c_012", "c_013_own", "c_L_level0", "c_L_level1")  # the ladder of decoded predictions of c_L: 012 token-local MLPs, 013 frozen-pattern at the own base, 015 Level 0, 015 Level 1 (exact)

CANDIDATES: dict[str, tuple[str, ...]] = {
    "determiner-like": ("fourth", "fifth", "particular", "previous", "final", "initial", "upper", "respective", "individual"),
    "ordinal-or-numeral": ("sixth", "seventh", "eighth", "ninth", "tenth", "twentieth", "quarter", "twin", "dual", "tens"),
    "quantity": ("scant", "adequate", "moderate", "substantial", "enormous", "unlimited", "countable", "spare", "remaining", "total"),
    "possessive-or-pronoun": ("himself", "herself", "itself", "ourselves", "oneself", "myself", "yourselves"),
    "adjective": ("plastic", "frozen", "ancient", "modern", "loud", "bright", "orange", "pink", "brown", "grey", "heavy", "gentle"),
}
QUOTAS = {"determiner-like": 5, "ordinal-or-numeral": 5, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 5}
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The farmer grows {cue}"),
    ("cardinal", "The printer produces {cue}"),
    ("quantifier", "The clinic treats {cue}"),
    ("quantifier", "The agency recruits {cue}"),
    ("coordinated-adjective", "Emma and Marco gathered {cue} dry"),
    ("coordinated-adjective", "Sam and Julia loaded {cue} warm"),
)
FRAME_ID_TAG = "015"
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", "experiments/014-neuron-1987-feature/", "experiments/013-attention-paths-frozen-pattern/", "experiments/012-layer-correction-token-local/",
                            "experiments/011-encoding-read-prospective/", "experiments/010-read-direction-assembly/", "experiments/009-head-transport-rule/", "experiments/008-cue-suppression-localization/",
                            "experiments/007-supervised-cue-subspace/", "experiments/006-low-rank-cue-decompilation/", "experiments/005-regular-plural-mechanism/", "screening/behavior-candidates/manifest-v1.json")
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)
OUTCOME_Y1 = ("PATTERN_CHANGE_PREDICTED_TOKENS", "PATTERN_CHANGE_NOT_PREDICTED_TOKENS", "PRECONDITION_FAILED_TOKENS")
OUTCOME_Y2 = ("PATTERN_CHANGE_PREDICTED_FRAMES_CONDITIONAL", "PATTERN_CHANGE_NOT_PREDICTED_FRAMES_CONDITIONAL", "PRECONDITION_FAILED_FRAMES")
OUTCOME_Y3 = ("AXIS_ONLY_REJECTED", "AXIS_ONLY_NOT_REJECTED", "AXIS_ONLY_NOT_EVALUABLE")


class PhaseError(pm.PhaseError):
    """Protocol violation in the Experiment 015 phase machinery."""


# ---------------------------------------------------------------------------
# The weight-only query/key/value program of one layer (frozen).


def _grab(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu", torch.float64).clone()


def _rotary_configuration(model: Any, d_head: int) -> tuple[int, float]:
    """(rotary_dim, base): ``cfg.rotary_dim`` when set; else the HuggingFace ``partial_rotary_factor × d_head`` and ``rope_theta``."""
    cfg = model.cfg
    if getattr(cfg, "rotary_adjacent_pairs", False):
        raise ValueError("the program implements the half-split rotary convention; adjacent-pair rotation is not supported")
    base = float(getattr(cfg, "rotary_base", 10000) or 10000)
    rotary_dim = getattr(cfg, "rotary_dim", None)
    if rotary_dim is None:
        config = getattr(getattr(model, "original_model", None), "config", None)
        parameters = getattr(config, "rope_parameters", None)
        if not isinstance(parameters, Mapping):
            raise ValueError("the rotary dimension is unknown: neither cfg.rotary_dim nor the HuggingFace rope parameters are available")
        rotary_dim = int(round(float(parameters.get("partial_rotary_factor", 1.0)) * d_head))
        base = float(parameters.get("rope_theta", base))
    rotary_dim = int(rotary_dim)
    if rotary_dim < 0 or rotary_dim > d_head or rotary_dim % 2:
        raise ValueError(f"invalid rotary dimension {rotary_dim} for d_head {d_head}")
    return rotary_dim, base


@dataclass(frozen=True)
class LayerProgram:
    """LayerNorm → q̃/k̃/v projections (all heads) → partial rotary rotation → W_O, for one layer; every constant the checkpoint's."""

    layer: int
    ln_w: torch.Tensor
    ln_b: torch.Tensor
    eps: float
    W_Q: torch.Tensor  # [heads, d_model, d_head]
    b_Q: torch.Tensor  # [heads, d_head]
    W_K: torch.Tensor
    b_K: torch.Tensor
    W_V: torch.Tensor
    b_V: torch.Tensor
    W_O: torch.Tensor  # [heads, d_head, d_model]
    rotary_dim: int
    rotary_base: float

    @classmethod
    def from_model(cls, model: Any, layer: int) -> "LayerProgram":
        block = model.blocks[layer]
        attn = block.attn
        W_Q, W_K, W_V, W_O = (_grab(getattr(attn, name)) for name in ("W_Q", "W_K", "W_V", "W_O"))
        n_heads, _, d_head = W_Q.shape

        def bias(name: str) -> torch.Tensor:
            value = getattr(attn, name, None)
            return _grab(value) if value is not None else torch.zeros(n_heads, d_head, dtype=torch.float64)

        rotary_dim, base = _rotary_configuration(model, d_head)
        original = getattr(attn, "_original_component", None)
        if original is not None and getattr(original, "rotary_ndims", rotary_dim) != rotary_dim:
            raise ValueError(f"the rotary dimension {rotary_dim} disagrees with the component's {original.rotary_ndims}")
        return cls(layer, _grab(block.ln1.w), _grab(block.ln1.b), float(model.cfg.eps), W_Q, bias("b_Q"), W_K, bias("b_K"), W_V, bias("b_V"), W_O, rotary_dim, base)

    @property
    def n_heads(self) -> int:
        return int(self.W_Q.shape[0])

    @property
    def d_head(self) -> int:
        return int(self.W_Q.shape[2])

    @property
    def scale(self) -> float:
        return math.sqrt(self.d_head)

    def record(self) -> dict[str, Any]:
        return {"layer": self.layer, "n_heads": self.n_heads, "d_head": self.d_head, "rotary_dim": self.rotary_dim, "rotary_base": self.rotary_base, "eps": self.eps}

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return pm.exact_layer_norm(x.double(), self.ln_w, self.ln_b, self.eps)

    def q_tilde(self, normed: torch.Tensor) -> torch.Tensor:
        return torch.einsum("d,hde->he", normed, self.W_Q) + self.b_Q

    def k_tilde(self, normed: torch.Tensor) -> torch.Tensor:
        return torch.einsum("d,hde->he", normed, self.W_K) + self.b_K

    def v(self, normed: torch.Tensor) -> torch.Tensor:
        return torch.einsum("d,hde->he", normed, self.W_V) + self.b_V

    def rotate(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """R_p on the first ``rotary_dim`` coordinates: the pair (i, i + rotary_dim/2) rotated by p · base^(−2i/rotary_dim); the rest pass through."""
        if self.rotary_dim == 0:
            return x
        half = self.rotary_dim // 2
        inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float64) / self.rotary_dim))
        angles = float(position) * inv_freq
        cos, sin = torch.cos(angles), torch.sin(angles)
        x1, x2, rest = x[..., :half], x[..., half:self.rotary_dim], x[..., self.rotary_dim:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], dim=-1)

    def output(self, values: torch.Tensor) -> torch.Tensor:
        """Per-head OV vectors: [heads, keys, d_head] → [heads, keys, d_model]."""
        return torch.einsum("hke,hed->hkd", values, self.W_O)


def programs_from_model(model: Any) -> dict[int, LayerProgram]:
    return {layer: LayerProgram.from_model(model, layer) for layer in HEAD_LAYERS}


class ReferenceRow:
    """One layer's reference quantities at the cue row, from the frame's reference residuals at positions 0..p_c."""

    def __init__(self, program: LayerProgram, residuals: Sequence[torch.Tensor]) -> None:
        self.program = program
        self.p_c = len(residuals) - 1
        normed = [program.normalize(x) for x in residuals]
        self.normed_pc = normed[self.p_c]
        self.q_ref = program.rotate(program.q_tilde(self.normed_pc), self.p_c)
        self.keys = torch.stack([program.rotate(program.k_tilde(normed[k]), k) for k in range(self.p_c + 1)], dim=1)  # [heads, keys, d_head]
        self.values = torch.stack([program.v(n) for n in normed], dim=1)
        self.scores_ref = torch.einsum("he,hke->hk", self.q_ref, self.keys) / program.scale
        self.A_ref = torch.softmax(self.scores_ref, dim=-1)
        self.output_ref = torch.einsum("hk,hkd->d", self.A_ref, program.output(self.values))

    def cue(self, normed_pc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rotated query, rotated self key and value of the cue position from a normalized cue residual."""
        return self.program.rotate(self.program.q_tilde(normed_pc), self.p_c), self.program.rotate(self.program.k_tilde(normed_pc), self.p_c), self.program.v(normed_pc)

    def row(self, q: torch.Tensor, k_pc: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("he,hke->hk", q, self.keys) / self.program.scale
        scores = scores.clone()
        scores[:, self.p_c] = (q * k_pc).sum(-1) / self.program.scale
        return torch.softmax(scores, dim=-1)

    def row_from_state(self, normed_pc: torch.Tensor) -> torch.Tensor:
        q, k_pc, _ = self.cue(normed_pc)
        return self.row(q, k_pc)

    def row_from_changes(self, dq_tilde: torch.Tensor, ds_pc: torch.Tensor) -> torch.Tensor:
        """Level 0: the reference logits moved by ⟨R_{p_c} Δq̂, k_ref(k)⟩/√d off the diagonal and by the token-local Δŝ on it."""
        scores = self.scores_ref + torch.einsum("he,hke->hk", self.program.rotate(dq_tilde, self.p_c), self.keys) / self.program.scale
        scores = scores.clone()
        scores[:, self.p_c] = self.scores_ref[:, self.p_c] + ds_pc
        return torch.softmax(scores, dim=-1)

    def row_additive(self, normed_pc: torch.Tensor) -> torch.Tensor:
        """Both sides moved, the diagonal interaction ⟨Δq, Δk⟩ dropped."""
        q, k_pc, _ = self.cue(normed_pc)
        dq, dk = q - self.q_ref, k_pc - self.keys[:, self.p_c]
        scores = torch.einsum("he,hke->hk", q, self.keys) / self.program.scale
        scores = scores.clone()
        scores[:, self.p_c] = self.scores_ref[:, self.p_c] + ((dq * self.keys[:, self.p_c]).sum(-1) + (self.q_ref * dk).sum(-1)) / self.program.scale
        return torch.softmax(scores, dim=-1)

    def proportional(self, a_pc: torch.Tensor) -> torch.Tensor:
        """The self weight set to ``a_pc``; the other keys rescaled in the proportions of the reference row."""
        row = self.A_ref.clone()
        row = row * ((1.0 - a_pc) / (1.0 - self.A_ref[:, self.p_c]).clamp_min(1e-12))[:, None]
        row[:, self.p_c] = a_pc
        return row

    def diagonal_terms(self, normed_pc: torch.Tensor) -> dict[str, torch.Tensor]:
        q, k_pc, _ = self.cue(normed_pc)
        dq, dk = q - self.q_ref, k_pc - self.keys[:, self.p_c]
        k_ref = self.keys[:, self.p_c]
        return {"dq_k": (dq * k_ref).sum(-1) / self.program.scale, "q_dk": (self.q_ref * dk).sum(-1) / self.program.scale, "dq_dk": (dq * dk).sum(-1) / self.program.scale}

    def values_with_cue(self, v_pc: torch.Tensor) -> torch.Tensor:
        values = self.values.clone()
        values[:, self.p_c] = v_pc
        return values

    def pattern_change_vectors(self, row: torch.Tensor, v_pc: torch.Tensor) -> torch.Tensor:
        """Per head Σ_k (row − A_ref)(p_c,k) v(k) W_O^h with the cue value replaced: [heads, d_model]."""
        return torch.einsum("hk,hkd->hd", row - self.A_ref, self.program.output(self.values_with_cue(v_pc)))

    def output_change(self, row: torch.Tensor, v_pc: torch.Tensor) -> torch.Tensor:
        """The layer's attention output change at the cue position (all heads): Σ_h [Σ_k row v' − Σ_k A_ref v] W_O^h."""
        return torch.einsum("hk,hkd->d", row, self.program.output(self.values_with_cue(v_pc))) - self.output_ref


def reference_rows(programs: Mapping[int, LayerProgram], x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor]) -> dict[int, ReferenceRow]:
    return {1: ReferenceRow(programs[1], [x.double() for x in x1_all]), 2: ReferenceRow(programs[2], [x.double() for x in x2_all])}


def check_reference_rows(rows: Mapping[int, ReferenceRow], state: ap.FrameState013) -> float:
    """I1 on the reference run: the recomputed rows against the captured pattern rows at p_c."""
    p_c = state.p_c
    worst = max(float((rows[1].A_ref - state.A1[:, : p_c + 1].double()).abs().max()), float((rows[2].A_ref - state.A2[:, : p_c + 1].double()).abs().max()))
    if worst > ROW_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{state.ref.frame.frame_id}: the program does not reproduce the captured reference pattern rows (max {worst:.2e})")
    return worst


# ---------------------------------------------------------------------------
# The models: Level 0 (the hypothesis), Level 1 (the identity), the axis-only alternative, the comparators and rungs.


def rows_to_json(row: torch.Tensor, layer: int) -> dict[str, list[float]]:
    return {ap.head_key(layer, h): [float(v) for v in row[h]] for h in range(row.shape[0])}


def rows_from_json(entry: Mapping[str, Sequence[float]], layer: int) -> torch.Tensor:
    return torch.tensor([list(entry[ap.head_key(layer, h)]) for h in range(N_HEADS)], dtype=torch.float64)


@dataclass(frozen=True)
class TokenLocalModel:
    """Level 0 and its companions; every prediction takes the token identity, the locked bases, axes and read, the weights and the frame's reference residuals."""

    read: lc.CorrectionRead
    lw: lc.LayerWeights
    programs: Mapping[int, LayerProgram]
    bases_012: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    d_E: torch.Tensor

    def changes(self, layer: int, base: torch.Tensor, dx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Token-local cue changes at a base: Δq̃, Δk̃, Δv and the rotation-free diagonal logit change."""
        program = self.programs[layer]
        nb, n0 = program.normalize(base), program.normalize(base.double() + dx.double())
        qb, kb, vb = program.q_tilde(nb), program.k_tilde(nb), program.v(nb)
        q0, k0, v0 = program.q_tilde(n0), program.k_tilde(n0), program.v(n0)
        return {"dq": q0 - qb, "dk": k0 - kb, "dv": v0 - vb, "ds_pc": ((q0 * k0).sum(-1) - (qb * kb).sum(-1)) / program.scale}

    def read_of(self, vectors: torch.Tensor, denominator: float) -> float:
        return sum(self.read.inner(vectors[h]) for h in range(vectors.shape[0])) / denominator

    def level_zero(self, weights: pm.Weights, rows: Mapping[int, ReferenceRow], token_id: int, template: str) -> dict[str, Any]:
        delta_e = self.read.encoding_delta(weights, token_id, template)
        denominator = self.read.denominator(weights, template)
        xb1, xb2 = (b.double() for b in self.bases_012[template])
        c1 = self.changes(1, xb1, delta_e)
        A1 = rows[1].row_from_changes(c1["dq"], c1["ds_pc"])
        v1 = rows[1].values[:, rows[1].p_c] + c1["dv"]
        dx2 = delta_e + self.lw.delta_out(1, xb1, delta_e) + rows[1].output_change(A1, v1)
        c2 = self.changes(2, xb2, dx2)
        A2 = rows[2].row_from_changes(c2["dq"], c2["ds_pc"])
        v2 = rows[2].values[:, rows[2].p_c] + c2["dv"]
        out = {"rows": {1: A1, 2: A2}, "values": {1: v1, 2: v2}, "dx2": dx2, "delta_e": delta_e, "denominator": denominator}
        out["c"] = {layer: self.read_of(rows[layer].pattern_change_vectors(out["rows"][layer], out["values"][layer]), denominator) for layer in HEAD_LAYERS}
        diag = {layer: rows[layer].proportional(out["rows"][layer][:, rows[layer].p_c]) for layer in HEAD_LAYERS}
        out["diag_rows"] = diag
        out["c_diag"] = {layer: self.read_of(rows[layer].pattern_change_vectors(diag[layer], out["values"][layer]), denominator) for layer in HEAD_LAYERS}
        out["arrival_read"] = self.read.inner(dx2) / denominator
        # The fully decoded layer contribution at Level 0 (the re-derived Experiment 013 ladder): both MLP terms at the bases, both attention output changes from the Level-0 rows and values.
        mlp = {1: self.lw.delta_out(1, xb1, delta_e), 2: self.lw.delta_out(2, xb2, dx2)}
        attention = {1: rows[1].output_change(A1, v1), 2: rows[2].output_change(A2, v2)}
        out["c_M"] = self.read.inner(mlp[1] + mlp[2]) / denominator
        out["c_H"] = self.read.inner(attention[1] + attention[2]) / denominator
        out["c_L"] = out["c_M"] + out["c_H"]
        return out

    def level_one(self, weights: pm.Weights, rows: Mapping[int, ReferenceRow], x1: torch.Tensor, x2: torch.Tensor, token_id: int, template: str) -> dict[str, Any]:
        """The exact recomputation at the frame's own state (the identity): rows, cue values and the arriving change of both layers."""
        delta_e = self.read.encoding_delta(weights, token_id, template)
        denominator = self.read.denominator(weights, template)
        n1 = self.programs[1].normalize(x1.double() + delta_e)
        q1, k1, v1 = rows[1].cue(n1)
        A1 = rows[1].row(q1, k1)
        dx2 = delta_e + self.lw.delta_out(1, x1, delta_e) + rows[1].output_change(A1, v1)
        n2 = self.programs[2].normalize(x2.double() + dx2)
        q2, k2, v2 = rows[2].cue(n2)
        A2 = rows[2].row(q2, k2)
        out = {"rows": {1: A1, 2: A2}, "values": {1: v1, 2: v2}, "normed": {1: n1, 2: n2}, "dx": {1: delta_e, 2: dx2}, "delta_e": delta_e, "denominator": denominator}
        out["c"] = {layer: self.read_of(rows[layer].pattern_change_vectors(out["rows"][layer], out["values"][layer]), denominator) for layer in HEAD_LAYERS}
        mlp = {1: self.lw.delta_out(1, x1, delta_e), 2: self.lw.delta_out(2, x2, dx2)}
        attention = {1: rows[1].output_change(A1, v1), 2: rows[2].output_change(A2, v2)}
        out["c_M"] = self.read.inner(mlp[1] + mlp[2]) / denominator
        out["c_H"] = self.read.inner(attention[1] + attention[2]) / denominator
        out["c_L"] = out["c_M"] + out["c_H"]
        return out

    def axis_only(self, rows: Mapping[int, ReferenceRow], x_ref: Mapping[int, torch.Tensor], dx_exact: Mapping[int, torch.Tensor], denominator: float) -> dict[str, Any]:
        """The same program on the d̂_E component of the exact change at the frame's own state; no scale or intercept."""
        out: dict[str, Any] = {"rows": {}, "values": {}, "c": {}}
        for layer in HEAD_LAYERS:
            axis = (dx_exact[layer].double() @ self.d_E) * self.d_E
            normed = self.programs[layer].normalize(x_ref[layer].double() + axis)
            q, k_pc, v = rows[layer].cue(normed)
            out["rows"][layer] = rows[layer].row(q, k_pc)
            out["values"][layer] = v
            out["c"][layer] = self.read_of(rows[layer].pattern_change_vectors(out["rows"][layer], v), denominator)
        return out

    def parts(self, weights: pm.Weights, rows: Mapping[int, ReferenceRow], x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], token_id: int, template: str) -> dict[str, Any]:
        """Level 0, Level 1 and the axis-only alternative on one frame's reference rows, plus the table entry built from them."""
        p_c = rows[1].p_c
        zero = self.level_zero(weights, rows, token_id, template)
        one = self.level_one(weights, rows, x1_all[p_c], x2_all[p_c], token_id, template)
        axis = self.axis_only(rows, {1: x1_all[p_c].double(), 2: x2_all[p_c].double()}, one["dx"], zero["denominator"])
        entry = {"p_c": p_c,
                 "rows_level0": {str(layer): rows_to_json(zero["rows"][layer] - rows[layer].A_ref, layer) for layer in HEAD_LAYERS},
                 "self_level0": {ap.head_key(layer, h): float(zero["rows"][layer][h, p_c] - rows[layer].A_ref[h, p_c]) for layer in HEAD_LAYERS for h in range(N_HEADS)},
                 "c_hat_1": zero["c"][1], "c_hat_2": zero["c"][2], "c_hat": zero["c"][1] + zero["c"][2],
                 "rows_axis": {str(layer): rows_to_json(axis["rows"][layer] - rows[layer].A_ref, layer) for layer in HEAD_LAYERS}, "c_axis": axis["c"][1] + axis["c"][2],
                 "rows_diag_L0": {str(layer): rows_to_json(zero["diag_rows"][layer] - rows[layer].A_ref, layer) for layer in HEAD_LAYERS}, "c_diag_L0": zero["c_diag"][1] + zero["c_diag"][2],
                 "arrival_read": zero["arrival_read"], "c_L_level0": zero["c_L"], "c_M_level0": zero["c_M"], "c_H_level0": zero["c_H"]}
        return {"rows": rows, "zero": zero, "one": one, "axis": axis, "entry": entry}

    def predict(self, weights: pm.Weights, x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], token_id: int, template: str) -> dict[str, Any]:
        """A table entry: the Level-0 rows and reads, the axis-only rows and read, the Level-0 diagonal-proportional rows and read, the decoded layer contribution."""
        return self.parts(weights, reference_rows(self.programs, x1_all, x2_all), x1_all, x2_all, token_id, template)["entry"]

    def predict_from_state(self, weights: pm.Weights, state: ap.FrameState013, token_id: int, template: str) -> dict[str, Any]:
        return self.predict(weights, state.x1_all, state.x2_all, token_id, template)

    def predict_from_locked(self, weights: pm.Weights, locked: Mapping[str, Any], token_id: int, template: str) -> dict[str, Any]:
        return self.predict(weights, [torch.tensor(x, dtype=torch.float64) for x in locked["x1_all"]], [torch.tensor(x, dtype=torch.float64) for x in locked["x2_all"]], token_id, template)


def model_from_locks(lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lw: lc.LayerWeights, programs: Mapping[int, LayerProgram], reference_ids: Mapping[str, int], plural_ids: Mapping[str, int]) -> TokenLocalModel:
    read = lc.read_from_lock_011(lock_011, reference_ids, plural_ids)
    return TokenLocalModel(read, lw, programs, lc.bases_from_json(lock_012["base_states"]), torch.tensor(lock_011["axes_vectors"]["R0"], dtype=torch.float64))


def locked_state(state: ap.FrameState013) -> dict[str, Any]:
    """What the lock stores per exposed frame: the reference residuals before blocks 1 and 2 at every position ≤ p_c."""
    return {"p_c": state.p_c, "x1_all": [x.double().tolist() for x in state.x1_all], "x2_all": [x.double().tolist() for x in state.x2_all]}


def prediction_row(word: str, frame_id: str, template: str, prediction: Mapping[str, Any]) -> dict[str, Any]:
    row = {"token": word, "frame_id": frame_id, "template": template}
    row.update({key: prediction[key] for key in PREDICTION_COLUMNS[3:]})
    return row


# ---------------------------------------------------------------------------
# Row statistics: sufficient statistics per pair, pooled R² and total-variation ratio.


def row_statistics(predicted: torch.Tensor, measured: torch.Tensor) -> dict[str, float]:
    """Sufficient statistics of predicted against measured pattern-change entries (all heads of one layer) plus the total variations."""
    p, m = predicted.double().flatten(), measured.double().flatten()
    return {"n": int(m.numel()), "sum_m": float(m.sum()), "sum_m2": float((m * m).sum()), "ss_res": float(((p - m) ** 2).sum()),
            "tv_err": 0.5 * float((predicted.double() - measured.double()).abs().sum()), "tv": 0.5 * float((measured.double()).abs().sum())}


def pooled_entry_r2(statistics: Sequence[Mapping[str, float]]) -> float | None:
    n = sum(int(s["n"]) for s in statistics)
    if n < 2:
        return None
    sum_m = sum(s["sum_m"] for s in statistics)
    ss_tot = sum(s["sum_m2"] for s in statistics) - sum_m * sum_m / n
    if ss_tot <= 0.0:
        return None
    return 1.0 - sum(s["ss_res"] for s in statistics) / ss_tot


def tv_ratio(statistics: Sequence[Mapping[str, float]]) -> float | None:
    total = sum(s["tv"] for s in statistics)
    return sum(s["tv_err"] for s in statistics) / total if total > 0 else None


def pooled_summary(statistics: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    return {"n_pairs": len(statistics), "n_entries": sum(int(s["n"]) for s in statistics), "entry_r2": pooled_entry_r2(statistics), "tv_ratio": tv_ratio(statistics)}


def _statistics_of(measured_rows: Mapping[str, Mapping[str, Sequence[float]]], predicted_rows: Mapping[str, Mapping[str, Sequence[float]]], layer: int) -> dict[str, float]:
    measured = rows_from_json(measured_rows[str(layer)], layer)
    predicted = rows_from_json(predicted_rows[str(layer)], layer)
    return row_statistics(predicted, measured)


# ---------------------------------------------------------------------------
# Measurement per pair: the captured rows and residuals, the identities, the measured pattern-change term, the models.


def measure_pair(model: Any, weights: pm.Weights, head: ht.HeadWeights, state: ap.FrameState013, name: str, token_id: int, e_axis: pm.SiteAxis, axis_T: pm.SiteAxis, nouns: Sequence[pm.Noun]) -> ra.Attribution:
    return ap.measure_pair_013(model, weights, head, state, name, token_id, e_axis, axis_T, nouns)


@dataclass(frozen=True)
class AnalysisContext:
    """Everything a pair analysis needs besides the captures: the models, the head weights and the frozen-pattern model for the arrival rung."""

    model: TokenLocalModel
    fpm: ap.FrozenPatternModel
    heads: ap.HeadSet
    weights: pm.Weights
    axis_T: pm.SiteAxis


def analyse_pair_015(record: ra.Attribution, plural: ra.Attribution, *, context: AnalysisContext, state: ap.FrameState013, rows: Mapping[int, ReferenceRow]) -> dict[str, Any] | None:
    """The measured pattern change and its read, I1–I3, the Level-0 prediction with its statistics, the alternative, the comparators and rungs."""
    tlm, weights = context.model, context.weights
    base = lc.analyse_pair(record, plural, read=tlm.read, lw=tlm.lw, weights=weights, state=state, axis_T=context.axis_T)
    if base is None:
        return None
    frame = state.ref.frame
    p_c, template = frame.p_c, frame.template_id
    denominator = tlm.read.denominator(weights, template)
    x_ref = {1: state.x1.double(), 2: state.x2.double()}
    x_patch = {layer: record.extra[pm.site_label((f"RESID_PRE.L{layer}", p_c))].double() for layer in HEAD_LAYERS}
    A_patch = {layer: record.extra[pm.site_label((f"ATTN_PATTERN.L{layer}", p_c))][:, : p_c + 1].double() for layer in HEAD_LAYERS}
    identities: dict[str, float] = {}
    # I1 on the patched run: the program on the captured patched residual reproduces the captured patched rows.
    identities["I1_patched_rows"] = max(float((rows[layer].row_from_state(tlm.programs[layer].normalize(x_patch[layer])) - A_patch[layer]).abs().max()) for layer in HEAD_LAYERS)
    if identities["I1_patched_rows"] > ROW_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{frame.frame_id}/{record.token}: the program does not reproduce the captured patched pattern rows (max {identities['I1_patched_rows']:.2e})")
    # I3 (Experiment 013's split) and the measured pattern-change term per head.
    split = {key: ap.head_identity(context.heads, key, state, record) for key in HEAD_KEYS}
    identities["I3_head_split"] = max(entry["error"] for entry in split.values())
    measured_c = {layer: sum(tlm.read.inner(split[key]["pattern_change"]) for key in HEAD_KEYS if ap.layer_of(key) == layer) / denominator for layer in HEAD_LAYERS}
    A_ref = {1: state.A1[:, : p_c + 1].double(), 2: state.A2[:, : p_c + 1].double()}
    measured_rows = {layer: A_patch[layer] - A_ref[layer] for layer in HEAD_LAYERS}  # the captured patched row minus the captured reference row: a measurement, nothing recomputed
    # Level 1 and I2: the exact chain against the captured residual before block 2; Level 0 and the alternative as the table would hold them.
    parts = tlm.parts(weights, rows, state.x1_all, state.x2_all, record.token_id, template)
    one, zero, prediction = parts["one"], parts["zero"], parts["entry"]
    identities["I2_chain"] = lc.relative_vector_error(one["dx"][2], x_patch[2] - x_ref[2], x_ref[2])
    if identities["I2_chain"] > CHAIN_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{frame.frame_id}/{record.token}: the exact chain does not reproduce the captured residual before block 2 (relative error {identities['I2_chain']:.2e})")
    identities["I1_level1_rows"] = max(float((one["rows"][layer] - A_patch[layer]).abs().max()) for layer in HEAD_LAYERS)
    measured_json = {str(layer): rows_to_json(measured_rows[layer], layer) for layer in HEAD_LAYERS}
    statistics: dict[str, dict[str, dict[str, float]]] = {}
    for name, key in (("level0", "rows_level0"), ("axis", "rows_axis"), ("diag_L0", "rows_diag_L0")):
        statistics[name] = {str(layer): _statistics_of(measured_json, prediction[key], layer) for layer in HEAD_LAYERS}
    # Rungs at the frame's own state with the exact arriving change (descriptive); the oracle comparator's read uses the Level-0 values (V/O part (b)), as the spec scores both comparators.
    rung_rows: dict[str, dict[int, torch.Tensor]] = {"level1": one["rows"], "diag_oracle": {layer: rows[layer].proportional(one["rows"][layer][:, p_c]) for layer in HEAD_LAYERS}}
    query_only, key_only, additive, linearised = {}, {}, {}, {}
    for layer in HEAD_LAYERS:
        q, k_pc, _ = rows[layer].cue(one["normed"][layer])
        query_only[layer] = rows[layer].row(q, rows[layer].keys[:, p_c])
        key_only[layer] = rows[layer].row(rows[layer].q_ref, k_pc)
        additive[layer] = rows[layer].row_additive(one["normed"][layer])
        linearised[layer] = rows[layer].row_from_state(_linearised_norm(tlm.programs[layer], x_ref[layer], one["dx"][layer]))
    rung_rows.update({"query_only": query_only, "key_only": key_only, "additive": additive, "linearised": linearised})
    arrival_013 = nf.arriving_change(context.fpm, weights, state.x1, {key: state.pattern_weight(key) for key in HEAD_KEYS}, record.token_id, template)["total"]
    rung_rows["frozen_arrival"] = {1: one["rows"][1], 2: rows[2].row_from_state(tlm.programs[2].normalize(x_ref[2] + arrival_013))}
    rung_statistics = {name: {str(layer): row_statistics(entry[layer] - rows[layer].A_ref, measured_rows[layer]) for layer in HEAD_LAYERS} for name, entry in rung_rows.items()}
    rung_values = {name: (zero["values"] if name == "diag_oracle" else one["values"]) for name in rung_rows}
    rung_reads = {name: {str(layer): tlm.read_of(rows[layer].pattern_change_vectors(entry[layer], rung_values[name][layer]), denominator) for layer in HEAD_LAYERS} for name, entry in rung_rows.items()}
    diagonal = {str(layer): {name: value.tolist() for name, value in rows[layer].diagonal_terms(one["normed"][layer]).items()} for layer in HEAD_LAYERS}
    norm_ratio = {str(layer): float(one["dx"][layer].norm() / (x_ref[layer] - x_ref[layer].mean()).norm()) for layer in HEAD_LAYERS}
    # The re-derived Experiment 013 ladder: the decoded predictions of c_L from Experiment 012 (token-local MLPs at the bases), Experiment 013 (frozen patterns at the own base), Level 0 and Level 1.
    thirteen = context.fpm.predict_from_state(weights, state, record.token_id, template)
    ladder = {"c_012": thirteen["c_012"], "c_013_own": thirteen["c_L_hat"], "c_L_level0": zero["c_L"], "c_M_level0": zero["c_M"], "c_H_level0": zero["c_H"], "c_L_level1": one["c_L"], "c_M_level1": one["c_M"], "c_H_level1": one["c_H"],
              "level1_error": abs(one["c_L"] - base["c_L"])}
    return {"token": record.token, "token_id": record.token_id, "frame_id": frame.frame_id, "template": template, "p_c": p_c,
            "rows": measured_json, "self": {key: split[key]["delta_A_pc"] for key in HEAD_KEYS},
            "c_1": measured_c[1], "c_2": measured_c[2], "c": measured_c[1] + measured_c[2], "c_level1": one["c"][1] + one["c"][2],
            "c_L": base["c_L"], "c_M": base["c_M"], "c_H": base["c_H"], "c_k": base["c_k"], "P1": base["P1"], "q_T": base["q_T"], "g_E": base["g_E"],
            "prediction": prediction, "statistics": statistics, "rung_statistics": rung_statistics, "rung_reads": rung_reads, "ladder": ladder,
            "diagonal_terms": diagonal, "norm_ratio": norm_ratio, "identities": {**base["identities"], **identities}}


def compact_analysis(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """The stored form of a pair analysis: the predicted rows are dropped (they are recomputable from the locked state and, at confirm, live in the tables); the statistics against them stay."""
    prediction = {key: value for key, value in analysis["prediction"].items() if key not in ("rows_level0", "rows_axis", "rows_diag_L0")}
    return {**analysis, "prediction": prediction}


def _linearised_norm(program: LayerProgram, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """LN(x) plus its first-order change for dx: (v_c − x̂·mean(x̂ ⊙ v_c))/σ ⊙ γ with v_c the centred change."""
    x = x.double()
    mu = x.mean()
    sigma = torch.sqrt(((x - mu) ** 2).mean() + program.eps)
    x_hat = (x - mu) / sigma
    v_c = dx.double() - dx.double().mean()
    return program.normalize(x) + (v_c - x_hat * (x_hat * v_c).mean()) / sigma * program.ln_w


# ---------------------------------------------------------------------------
# Pool (159 tokens × 48 frames), the inherited extracts, the confirmation set.


def build_pool_015(manifest: ScreeningManifest, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009, confirmation_011: er.Confirmation011,
                   confirmation_012: lc.Confirmation012, confirmation_013: ap.Confirmation013, confirmation_014: nf.Confirmation014) -> cs.Pool008:
    base = nf.build_pool_014(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013)
    frames = base.frames + tuple(confirmation_014.frames)
    origin = dict(base.frame_origin) | {frame.frame_id: "confirmation-014" for frame in confirmation_014.frames}
    tokens = list(base.tokens)
    category, source = dict(base.token_category), dict(base.token_source)
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation_014.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"Experiment 014 token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-014"
    return cs.Pool008(frames, origin, tuple(tokens), category, source, base.nouns, base.noun_source, base.reference_ids, base.plural_cue)


LEDGER_FIELDS = ("c_L", "c_M", "c_H")
LEDGER_014_DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014", "lock_014")
LEDGER_014_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "confirmation_014_sha256", "lock_014_sha256")
EXTRACT_013_DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "lock_013")
EXTRACT_013_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "lock_013_sha256")


def ledger_entry(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {"c_L": float(analysis["c_L"]), "c_M": float(analysis["c_M"]), "c_H": float(analysis["c_H"]), "c_k": {key: float(analysis["c_k"][key]) for key in ra.COMPONENT_ORDER}}


def pattern_entry(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {"delta_A_pc": {key: float(analysis["self"][key]) for key in HEAD_KEYS}, "direct_pattern_change": float(analysis["c"])}


def _inherited_payload(*, experiment: str, kind: str, description: str, entries: Mapping[str, Mapping[str, Any]], source: Mapping[str, Any], digests: Mapping[str, str], keys: Sequence[str], fields: Sequence[str]) -> dict[str, Any]:
    payload = {"schema_version": INHERITED_SCHEMA_VERSION, "experiment": experiment, "kind": kind, "description": description, "source": dict(source), **{field: digests[key] for field, key in zip(fields, keys)},
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "entries": {key: dict(value) for key, value in sorted(entries.items())}}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def inherited_ledger_payload(entries: Mapping[str, Mapping[str, Any]], *, source: Mapping[str, Any], digests: Mapping[str, str]) -> dict[str, Any]:
    return _inherited_payload(experiment="014", kind="pair-ledger", description="Derived extract of Experiment 014's recorded per-pair c_L, c_M, c_H and c_k (Experiment 012's Y1 units) over its 3732 exposed pairs and its 1008 + 144 confirmed pairs. Experiment 015 recomputes these and requires agreement within 1e-6.",
                              entries=entries, source=source, digests=digests, keys=LEDGER_014_DIGEST_KEYS, fields=LEDGER_014_DIGEST_FIELDS)


def inherited_extract_payload(entries: Mapping[str, Mapping[str, Any]], *, source: Mapping[str, Any], digests: Mapping[str, str]) -> dict[str, Any]:
    return _inherited_payload(experiment="013", kind="pattern-extract", description="Derived extract of Experiment 013's recorded per-pair self-attention weight changes ΔA_h(p_c,p_c) of the sixteen layer-1–2 heads and its direct pattern-change read (remainder_direct_pattern_change) over its 2724 exposed pairs and its 864 + 144 confirmed pairs. Experiment 015 recomputes these and requires agreement within 1e-6.",
                              entries=entries, source=source, digests=digests, keys=EXTRACT_013_DIGEST_KEYS, fields=EXTRACT_013_DIGEST_FIELDS)


def _load_inherited(path: Path, *, experiment: str, kind: str, digests: Mapping[str, str], keys: Sequence[str], fields: Sequence[str], expected_size: int) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited {experiment} {kind}: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "experiment", "kind", "description", "source", *fields, "model", "entries", "content_sha256"}, f"inherited {experiment} {kind}")
    if payload["schema_version"] != INHERITED_SCHEMA_VERSION or payload["experiment"] != experiment or payload["kind"] != kind or payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError(f"inherited {experiment} {kind} schema or digest is not frozen")
    if tuple(payload[field] for field in fields) != tuple(digests[key] for key in keys):
        raise ValueError(f"inherited {experiment} {kind} was recorded against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision} or len(payload["entries"]) != expected_size:
        raise ValueError(f"inherited {experiment} {kind} model or size is not frozen")
    return payload


def load_inherited_ledger(path: Path, *, digests: Mapping[str, str], expected_size: int) -> dict[str, Any]:
    return _load_inherited(path, experiment="014", kind="pair-ledger", digests=digests, keys=LEDGER_014_DIGEST_KEYS, fields=LEDGER_014_DIGEST_FIELDS, expected_size=expected_size)


def load_inherited_extract(path: Path, *, digests: Mapping[str, str], expected_size: int) -> dict[str, Any]:
    return _load_inherited(path, experiment="013", kind="pattern-extract", digests=digests, keys=EXTRACT_013_DIGEST_KEYS, fields=EXTRACT_013_DIGEST_FIELDS, expected_size=expected_size)


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
        raise pm.IncidentError(f"{worst_key}: recomputed quantity deviates from Experiment 014's record by {worst:.3e}")
    return {"passed": True, "n": len(recorded), "max_abs_deviation": worst, "worst_key": worst_key}


def check_extract_replication(measured: Mapping[str, Mapping[str, Any]], recorded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = set(recorded) - set(measured)
    if missing:
        raise pm.IncidentError(f"recomputed pattern quantities lack {len(missing)} recorded pairs (e.g. {sorted(missing)[:3]})")
    worst_key, worst = None, 0.0
    for key, value in recorded.items():
        other = measured[key]
        deviation = abs(value["direct_pattern_change"] - other["direct_pattern_change"])
        if deviation > worst:
            worst_key, worst = f"{key}:direct_pattern_change", deviation
        for head in HEAD_KEYS:
            deviation = abs(value["delta_A_pc"][head] - other["delta_A_pc"][head])
            if deviation > worst:
                worst_key, worst = f"{key}:{head}", deviation
    if worst > REPLICATION_TOLERANCE:
        raise pm.IncidentError(f"{worst_key}: recomputed pattern quantity deviates from Experiment 013's record by {worst:.3e}")
    return {"passed": True, "n": len(recorded), "max_abs_deviation": worst, "worst_key": worst_key}


def fresh_tokens_015(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
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


CONFIRMATION_DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014")
CONFIRMATION_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "confirmation_014_sha256")


def build_confirmation_payload(tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> dict[str, Any]:
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_015(tokenizer, excluded_ids)
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
               "policy": {"licensing": "every fresh cue in every fresh frame and in every exposed frame", "quotas": dict(QUOTAS), "expectations": "none: the committed pattern-change predictions are the only predictions; no label is preregistered for any head or cue"},
               "tokens": tokens, "frames": frames, "token_prompts": [nf._prompt_entry(tokenizer, frame, token) for frame in frame_objects for token in tokens],
               "exposed_frame_prompts": [nf._prompt_entry(tokenizer, frame, token) for frame in pool.frames for token in tokens],
               "construction": "tokenizer-only; first eligible candidates per lexical class; classes carry no expectation; no model output"}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


Confirmation015 = ap.Confirmation013  # the same structure: fresh frames, exposed frame ids, tokens, two prompt lists


def validate_confirmation(payload: Mapping[str, Any], pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation015:
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
    return Confirmation015(dict(payload["reference_cue_ids"]), frames, tuple(payload["exposed_frame_ids"]), tuple(tokens), tuple(fresh_prompts), tuple(exposed_prompts), payload["content_sha256"])


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


def load_confirmation(path: Path, pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation015:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, pool, digests)


# ---------------------------------------------------------------------------
# Results state, phases.

DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014", "confirmation_015",
               "lock_011", "lock_012", "lock_013", "lock_014", "ledger_014", "extract_013")
STATE_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "confirmation_014_sha256",
                       "confirmation_015_sha256", "lock_011_sha256", "lock_012_sha256", "lock_013_sha256", "lock_014_sha256", "ledger_014_sha256", "extract_013_sha256")
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


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation015) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


comparison = lc.comparison


# ---------------------------------------------------------------------------
# Statistics: exposed record and set scoring.


def _token_means(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not analyses:
        return {}
    mean = lambda values: pm._mean(list(values))  # noqa: E731
    return {"n_frames": len(analyses), "frames": sorted(a["frame_id"] for a in analyses),
            "c_mean": mean(a["c"] for a in analyses), "c_1_mean": mean(a["c_1"] for a in analyses), "c_2_mean": mean(a["c_2"] for a in analyses),
            "c_hat_mean": mean(p["c_hat"] for p in predictions), "c_hat_1_mean": mean(p["c_hat_1"] for p in predictions), "c_hat_2_mean": mean(p["c_hat_2"] for p in predictions),
            "c_axis_mean": mean(p["c_axis"] for p in predictions), "c_diag_L0_mean": mean(p["c_diag_L0"] for p in predictions), "arrival_read_mean": mean(p["arrival_read"] for p in predictions),
            "c_L_mean": mean(a["c_L"] for a in analyses), "c_M_mean": mean(a["c_M"] for a in analyses), "c_H_mean": mean(a["c_H"] for a in analyses),
            "ladder_mean": {key: mean(a["ladder"][key] for a in analyses) for key in (*LADDER_KEYS, "c_M_level0", "c_H_level0", "c_M_level1", "c_H_level1")},
            "rung_read_mean": {name: mean(a["rung_reads"][name]["1"] + a["rung_reads"][name]["2"] for a in analyses) for name in RUNGS},
            "self_mean": {key: mean(a["self"][key] for a in analyses) for key in HEAD_KEYS}, "self_hat_mean": {key: mean(p["self_level0"][key] for p in predictions) for key in HEAD_KEYS}}


def _pair_statistics(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled pattern-space statistics of the committed row models against the measured rows, per layer, plus self-weight and read comparisons."""
    out: dict[str, Any] = {"n_pairs": len(analyses)}
    for name, key in (("level0", "rows_level0"), ("axis", "rows_axis"), ("diag_L0", "rows_diag_L0")):
        per_layer = {}
        for layer in HEAD_LAYERS:
            stats = [_statistics_of(a["rows"], p[key], layer) for a, p in zip(analyses, predictions)]
            per_layer[str(layer)] = pooled_summary(stats)
        both = [_statistics_of(a["rows"], p[key], layer) for a, p in zip(analyses, predictions) for layer in HEAD_LAYERS]
        per_layer["both"] = pooled_summary(both)
        out[name] = per_layer
    self_hat = [p["self_level0"][key] for p in predictions for key in HEAD_KEYS]
    self_meas = [a["self"][key] for a in analyses for key in HEAD_KEYS]
    out["self_level0"] = {"pooled": comparison(self_hat, self_meas),
                          "per_head": {key: comparison([p["self_level0"][key] for p in predictions], [a["self"][key] for a in analyses]) for key in HEAD_KEYS}}
    out["c_hat_vs_c"] = comparison([p["c_hat"] for p in predictions], [a["c"] for a in analyses])
    out["c_hat_1_vs_c_1"] = comparison([p["c_hat_1"] for p in predictions], [a["c_1"] for a in analyses])
    out["c_hat_2_vs_c_2"] = comparison([p["c_hat_2"] for p in predictions], [a["c_2"] for a in analyses])
    out["c_axis_vs_c"] = comparison([p["c_axis"] for p in predictions], [a["c"] for a in analyses])
    out["c_diag_L0_vs_c"] = comparison([p["c_diag_L0"] for p in predictions], [a["c"] for a in analyses])
    out["c_level1_vs_c"] = comparison([a["c_level1"] for a in analyses], [a["c"] for a in analyses])
    out["comparator_margin"] = {str(layer): (out["level0"][str(layer)]["entry_r2"] - out["diag_L0"][str(layer)]["entry_r2"]) if out["level0"][str(layer)]["entry_r2"] is not None and out["diag_L0"][str(layer)]["entry_r2"] is not None else None for layer in HEAD_LAYERS}
    out["interaction_beyond_self_logit"] = all(margin is not None and margin >= COMPARATOR_MARGIN for margin in out["comparator_margin"].values())
    return out


def _rung_statistics(analyses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in RUNGS:
        out[name] = {str(layer): pooled_summary([a["rung_statistics"][name][str(layer)] for a in analyses]) for layer in HEAD_LAYERS}
        out[name]["read_vs_c"] = comparison([a["rung_reads"][name]["1"] + a["rung_reads"][name]["2"] for a in analyses], [a["c"] for a in analyses])
    rms = lambda values: math.sqrt(pm._mean([v * v for v in values])) if values else None  # noqa: E731
    out["diagonal_terms_rms"] = {str(layer): {name: rms([v for a in analyses for v in a["diagonal_terms"][str(layer)][name]]) for name in ("dq_k", "q_dk", "dq_dk")} for layer in HEAD_LAYERS}
    out["norm_ratio_mean"] = {str(layer): pm._mean([a["norm_ratio"][str(layer)] for a in analyses]) for layer in HEAD_LAYERS}
    out["measured_tv_mean"] = {str(layer): pm._mean([0.5 * float(rows_from_json(a["rows"][str(layer)], layer).abs().sum()) for a in analyses]) for layer in HEAD_LAYERS}
    out["self_abs_max"] = max(abs(a["self"][key]) for a in analyses for key in HEAD_KEYS) if analyses else None
    return out


def _ladder_statistics(analyses: Sequence[Mapping[str, Any]], means: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The re-derived Experiment 013 ladder: each decoded prediction of c_L against the measured c_L, per pair and (when token means are given) per token; Level 1's residual is a check."""
    out: dict[str, Any] = {"pairs": {key: comparison([a["ladder"][key] for a in analyses], [a["c_L"] for a in analyses]) for key in LADDER_KEYS},
                           "pairs_M_level0": comparison([a["ladder"]["c_M_level0"] for a in analyses], [a["c_M"] for a in analyses]), "pairs_H_level0": comparison([a["ladder"]["c_H_level0"] for a in analyses], [a["c_H"] for a in analyses]),
                           "level1_error_max": max((a["ladder"]["level1_error"] for a in analyses), default=None)}
    if means:
        names = sorted(means)
        out["token_means"] = {key: comparison([means[n]["ladder_mean"][key] for n in names], [means[n]["c_L_mean"] for n in names]) for key in LADDER_KEYS}
        out["token_means_rung_reads"] = {name: comparison([means[n]["rung_read_mean"][name] for n in names], [means[n]["c_mean"] for n in names]) for name in RUNGS}
    return out


def statistics_for(pairs: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, Any]:
    """Token-mean and pair-level statistics of Level 0, the alternative and the comparators, plus the rungs (the exposed record)."""
    by_token = {name: [(a, p) for a, p in zip(pairs, predictions) if a["token"] == name] for name in tokens}
    means = {name: _token_means([a for a, _ in rows], [p for _, p in rows]) for name, rows in by_token.items() if rows}
    names = sorted(means)
    m = lambda key: [means[n][key] for n in names]  # noqa: E731
    return {"n_tokens": len(names), "n_pairs": len(pairs),
            "token_means": {"c_hat_vs_c": comparison(m("c_hat_mean"), m("c_mean")), "c_axis_vs_c": comparison(m("c_axis_mean"), m("c_mean")), "c_diag_L0_vs_c": comparison(m("c_diag_L0_mean"), m("c_mean")),
                            "c_hat_1_vs_c_1": comparison(m("c_hat_1_mean"), m("c_1_mean")), "c_hat_2_vs_c_2": comparison(m("c_hat_2_mean"), m("c_2_mean")), "c_spread": ap.spread(m("c_mean"))},
            "pairs": _pair_statistics(pairs, predictions), "rungs": _rung_statistics(pairs), "ladder": _ladder_statistics(pairs, means), "token_means_table": means}


def _per_frame(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frames = sorted({a["frame_id"] for a in analyses})
    out = {}
    for frame_id in frames:
        mine = [(a, p) for a, p in zip(analyses, predictions) if a["frame_id"] == frame_id]
        entry = {"n_pairs": len(mine), "c_hat_vs_c": comparison([p["c_hat"] for _, p in mine], [a["c"] for a, _ in mine])}
        for layer in HEAD_LAYERS:
            entry[f"level0_layer{layer}"] = pooled_summary([_statistics_of(a["rows"], p["rows_level0"], layer) for a, p in mine])
            entry[f"axis_layer{layer}"] = pooled_summary([_statistics_of(a["rows"], p["rows_axis"], layer) for a, p in mine])
        out[frame_id] = entry
    return out


# ---------------------------------------------------------------------------
# Exploration (Tier A).


def _components(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any]) -> tuple[Any, ...]:
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    programs = programs_from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    axis_T, e_axis = lc.locked_axes(lock_011, cs.stage_axes(cache, weights, pool_010))
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    fpm = ap.model_from_locks(lock_011, lock_012, lw, heads, pool.reference_ids, plural_ids)
    lc.check_read_weight(fpm.read, head, axis_T)
    tlm = model_from_locks(lock_011, lock_012, lw, programs, pool.reference_ids, plural_ids)
    if float((tlm.d_E - e_axis.direction.double()).abs().max()) > 1e-7:
        raise pm.IncidentError("the Experiment 011 lock's R0 vector disagrees with the recomputed encoding axis")
    return weights, head, cache, axis_T, e_axis, fpm, tlm, AnalysisContext(tlm, fpm, heads, weights, axis_T)


def _program_record(programs: Mapping[int, LayerProgram]) -> dict[str, Any]:
    return {str(layer): program.record() for layer, program in programs.items()}


def run_exploration(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, *, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_014: Mapping[str, Any], inherited_ledger: Mapping[str, Any], inherited_extract: Mapping[str, Any],
                    state: dict[str, Any], results_path: Path | None, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    nouns = pool.single_nouns
    exploration = state["exploration"]
    say("stage axes from the Experiment 010 pool, checked against the Experiment 011 lock")
    weights, head, cache, axis_T, e_axis, fpm, tlm, context = _components(model, pool, pool_010, lock_011, lock_012)
    exploration["read_weight_check"] = lc.check_read_weight(fpm.read, head, axis_T)
    exploration["axes_vectors"] = {"T": axis_T.direction.tolist(), "R0": tlm.d_E.tolist()}
    exploration["sigma_T"] = axis_T.sigma
    exploration["read_weight"] = fpm.read.weight.tolist()
    exploration["defined_templates"] = list(lock_012["defined_templates"])
    exploration["program"] = _program_record(tlm.programs)
    recorded_ledger, recorded_extract = inherited_ledger["entries"], inherited_extract["entries"]
    say(f"reference states of the {len(pool.frames)} exposed frames; re-measurement of the {len(recorded_ledger)} recorded pairs with the residuals and pattern rows captured")
    locked_states: dict[str, Any] = {}
    pairs: dict[str, dict[str, Any]] = {}
    ledger: dict[str, dict[str, Any]] = {}
    patterns: dict[str, dict[str, Any]] = {}
    identities: dict[str, float] = {}
    for frame in pool.frames:
        state_f = ap.capture_frame_013(model, head, pool.reference_prompt(frame), nouns, axis_T)
        rows = reference_rows(tlm.programs, state_f.x1_all, state_f.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), check_reference_rows(rows, state_f))
        if frame.frame_id in lock_014.get("locked_states", {}):
            previous = lock_014["locked_states"][frame.frame_id]
            drift = max(float((state_f.x1 - torch.tensor(previous["x1"], dtype=torch.float64)).abs().max()), float((state_f.x2 - torch.tensor(previous["x2"], dtype=torch.float64)).abs().max()))
            if drift > LOCKED_STATE_TOLERANCE:
                raise pm.IncidentError(f"{frame.frame_id}: the reference state differs from the Experiment 014 lock (max {drift:.2e})")
        locked_states[frame.frame_id] = locked_state(state_f)
        names = [name for name, _ in pool.tokens if f"{name}|{frame.frame_id}" in recorded_ledger]
        if not names:
            continue
        plural_name = pool.plural_cue[frame.template_id]
        records = {name: measure_pair(model, weights, head, state_f, name, pool.token_id(name), e_axis, axis_T, nouns) for name in names}
        plural = records[plural_name] if plural_name in records else measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for name, record in records.items():
            analysis = analyse_pair_015(record, plural, context=context, state=state_f, rows=rows)
            key = f"{name}|{frame.frame_id}"
            if analysis is None:
                raise pm.IncidentError(f"{key}: no analysis although Experiment 014 recorded the pair")
            er._max_errors(identities, analysis["identities"])
            pairs[key] = analysis
            ledger[key] = ledger_entry(analysis)
            patterns[key] = pattern_entry(analysis)
        say(f"  {frame.frame_id}: {len(names)} pairs")
    exploration["identities"] = identities
    exploration["locked_states"] = locked_states
    exploration["locked_state_digests"] = {frame_id: ap.state_digest(entry) for frame_id, entry in locked_states.items()}
    exploration["replication"] = {"experiment_014": check_ledger_replication(ledger, recorded_ledger), "experiment_013_patterns": check_extract_replication(patterns, recorded_extract)}
    say(f"replication: Experiment 014 ledger max deviation {exploration['replication']['experiment_014']['max_abs_deviation']:.2e}; Experiment 013 patterns {exploration['replication']['experiment_013_patterns']['max_abs_deviation']:.2e}")
    analyses = list(pairs.values())
    stats = statistics_for(analyses, [a["prediction"] for a in analyses], [name for name, _ in pool.tokens])
    exploration["pairs"] = {key: compact_analysis(value) for key, value in pairs.items()}
    exploration["exposed_check"] = stats
    tm, pr = stats["token_means"]["c_hat_vs_c"], stats["pairs"]
    exploration["summary"] = {"defined_templates": exploration["defined_templates"], "spearman": tm["spearman"], "r2": tm["r2"], "entry_r2_1": pr["level0"]["1"]["entry_r2"], "entry_r2_2": pr["level0"]["2"]["entry_r2"],
                              "axis_r2": stats["token_means"]["c_axis_vs_c"]["r2"], "axis_entry_r2": pr["axis"]["both"]["entry_r2"], "comparator_margin": pr["comparator_margin"],
                              "prediction_undefined": len(exploration["defined_templates"]) < 2}
    f = cd._f
    say(f"exposed: ĉ_ΔA vs c_ΔA token means Spearman {f(tm['spearman'], 3)}, R² {f(tm['r2'], 3)}; entry R² layer 1 {f(pr['level0']['1']['entry_r2'], 3)}, layer 2 {f(pr['level0']['2']['entry_r2'], 3)}; axis-only R² {f(exploration['summary']['axis_r2'], 3)}, entries {f(exploration['summary']['axis_entry_r2'], 3)}")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


# ---------------------------------------------------------------------------
# Lock and its artifacts.


def prediction_table(tlm: TokenLocalModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], frames: Sequence[pm.Frame], tokens: Sequence[Mapping[str, Any]], defined: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        if frame.template_id not in defined:
            continue
        for token in tokens:
            rows.append(prediction_row(token["word"], frame.frame_id, frame.template_id, tlm.predict_from_locked(weights, locked_states[frame.frame_id], token["token_id"], frame.template_id)))
    return rows


def token_means_from_table(rows: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for word in tokens:
        mine = [row for row in rows if row["token"] == word]
        if mine:
            out[word] = {key: pm._mean([row[key] for row in mine]) for key in ("c_hat", "c_hat_1", "c_hat_2", "c_axis", "c_diag_L0", "arrival_read")}
            out[word]["n_frames"] = len(mine)
            out[word]["self_hat_mean"] = {key: pm._mean([row["self_level0"][key] for row in mine]) for key in HEAD_KEYS}
    return out


def lock_predictions(tlm: TokenLocalModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], pool: cs.Pool008, confirmation: Confirmation015, defined: Sequence[str]) -> dict[str, Any]:
    rows = prediction_table(tlm, weights, locked_states, pool.frames, confirmation.tokens, defined)
    return {"rows": rows, "token_means": token_means_from_table(rows, [token["word"] for token in confirmation.tokens])}


def render_predictions(lock: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 015 — preregistered predictions (the cue-induced attention-pattern change from the token-local cue-change Q/K model, conditional on the reference frame state)", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_015_sha256']}`; Experiment 014 lock sha256 `{lock['lock_014_sha256']}`",
             f"- Floors: Y1/Y2 Spearman ≥ {Y_SPEARMAN} and R² ≥ {Y_R2} on token means of ĉ_ΔA, pooled entry R² ≥ {Y_ENTRY_R2} at layer 1 and at layer 2; Y3 axis-only rejected iff token-mean R² < {Y3_R2_MAX} and pooled entry R² < {Y3_ENTRY_R2_MAX}; comparator rule margin {COMPARATOR_MARGIN}",
             f"- Y1 table: {len(lock['predictions']['rows'])} rows (fresh tokens × exposed frames), each with the sixteen predicted rows ΔÂ_h(p_c, ·), the axis-only rows and the Level-0 diagonal-proportional rows; Y2 rows are computed at confirm stage 1 and digested before any fresh cue prompt",
             f"- Program: {', '.join(f'layer {k}: d_head {v['d_head']}, rotary_dim {v['rotary_dim']}, base {v['rotary_base']:g}' for k, v in lock['program'].items())}", "",
             "| token | class | frames | **predicted ĉ_ΔA** | layer 1 | layer 2 | axis-only ĉ | diagonal-proportional ĉ | arrival read r(Δ̂x₂)/D |", "|---|---|---|---|---|---|---|---|---|"]
    categories = {token["word"]: token["category"] for token in lock["tokens"]}
    means = lock["predictions"]["token_means"]
    for word, m in sorted(means.items(), key=lambda kv: -kv[1]["c_hat"]):
        lines.append(f"| {word} | {categories.get(word, '')} | {m['n_frames']} | **{f(m['c_hat'], 4)}** | {f(m['c_hat_1'], 4)} | {f(m['c_hat_2'], 4)} | {f(m['c_axis'], 4)} | {f(m['c_diag_L0'], 4)} | {f(m['arrival_read'], 3)} |")
    lines += ["", "Predicted self-attention weight change ΔÂ_h(p_c, p_c) per head, averaged over the exposed frames:", "", "| token | " + " | ".join(HEAD_KEYS) + " |", "|---|" + "---|" * len(HEAD_KEYS)]
    for word, m in sorted(means.items(), key=lambda kv: -kv[1]["c_hat"]):
        lines.append(f"| {word} | " + " | ".join(f(m["self_hat_mean"][key], 3) for key in HEAD_KEYS) + " |")
    lines.append("")
    return "\n".join(lines)


def frozen_floors() -> dict[str, Any]:
    return {"y_spearman": Y_SPEARMAN, "y_r2": Y_R2, "y_entry_r2": Y_ENTRY_R2, "y3_r2_max": Y3_R2_MAX, "y3_entry_r2_max": Y3_ENTRY_R2_MAX, "comparator_margin": COMPARATOR_MARGIN,
            "min_valid_frames": MIN_VALID_FRAMES, "min_valid_frames_per_token": MIN_VALID_FRAMES_PER_TOKEN, "min_scored_tokens": MIN_SCORED_TOKENS, "frame_cue_effect_rate": FRAME_CUE_EFFECT_RATE, "head_stage_floor": cs.STAGE_UNINFORMATIVE_FLOOR,
            "row_identity_tolerance": ROW_IDENTITY_TOLERANCE, "chain_identity_tolerance": CHAIN_IDENTITY_TOLERANCE}


def build_candidate_lock(*, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation015, predictions: Mapping[str, Any], protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    if exploration["summary"]["prediction_undefined"]:
        raise PhaseError("fewer than two templates are defined: PREDICTION_UNDEFINED, no lock")
    lock = {"schema_version": 1, "experiment": "015", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit}
    for field, key in zip(STATE_DIGEST_FIELDS, DIGEST_KEYS):
        lock[field] = digests[key] if key != "confirmation_015" else confirmation.content_sha256
    lock.update({"confirmation_prompt_keys": sorted(prompt.key for prompt in confirmation.all_prompts), "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
                 "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED}, "head": ht.HEAD_KEY, "program": exploration["program"],
                 "axes_vectors": exploration["axes_vectors"], "sigma_T": exploration["sigma_T"], "read_weight": exploration["read_weight"], "defined_templates": exploration["defined_templates"],
                 "locked_states": exploration["locked_states"], "locked_state_digests": exploration["locked_state_digests"], "floors": frozen_floors(),
                 "tokens": [dict(token) for token in confirmation.tokens], "exposed_check": exploration["exposed_check"],
                 "predictions": dict(predictions), "tier_a_results_sha256": state.get("state_sha256")})
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation015, predictions_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "015" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked or git_state.get("dirty", True):
        raise PhaseError("confirm requires the committed lock and predictions on a clean Git tree")
    expected = dict(zip(STATE_DIGEST_FIELDS, digest_tuple(digests)))
    expected["confirmation_015_sha256"] = confirmation.content_sha256
    if any(lock.get(field) != value for field, value in expected.items()):
        raise PhaseError("lock was built against different frozen inputs")
    if not state.get("lock") or lock["content_sha256"] != state["lock"]["content_sha256"] or lock["run_id"] != state["run_id"]:
        raise PhaseError("the installed lock is not the candidate lock written by the lock phase")
    if pm.sha256_text(predictions_text) != state["lock"]["predictions_sha256"]:
        raise PhaseError("the installed predictions.md is not the artifact written by the lock phase")
    if dict(lock["floors"]) != frozen_floors():
        raise PhaseError("the lock's floors are not the frozen constants")
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = [path for path in changed_paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and path not in NON_SCIENTIFIC_PATHS and not path.startswith(NON_SCIENTIFIC_PREFIXES)]
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock commit: {scientific}")
    assert_confirmation_untouched(state, confirmation)


def _max_numeric_difference(a: Any, b: Any, path: str) -> float:
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            raise PhaseError(f"the recomputed prediction differs at {path}; nothing was executed")
        return 0.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            raise PhaseError(f"the recomputed prediction has different keys at {path}; nothing was executed")
        return max((_max_numeric_difference(a[key], b[key], f"{path}.{key}") for key in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise PhaseError(f"the recomputed prediction has a different length at {path}; nothing was executed")
        return max((_max_numeric_difference(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))), default=0.0)
    if a != b:
        raise PhaseError(f"the recomputed prediction differs at {path}; nothing was executed")
    return 0.0


def assert_lock_predictions_reproduced(lock: Mapping[str, Any], recomputed: Mapping[str, Any]) -> float:
    locked_rows, fresh_rows = lock["predictions"]["rows"], recomputed["rows"]
    if len(locked_rows) != len(fresh_rows):
        raise PhaseError("the recomputed prediction table has a different number of rows; nothing was executed")
    worst = 0.0
    for a, b in zip(locked_rows, fresh_rows):
        if (a["token"], a["frame_id"], a["template"], a["p_c"]) != (b["token"], b["frame_id"], b["template"], b["p_c"]):
            raise PhaseError("the recomputed prediction table is ordered differently; nothing was executed")
        for key in PREDICTION_COLUMNS[4:]:
            worst = max(worst, _max_numeric_difference(a[key], b[key], f"{a['token']}/{a['frame_id']}/{key}"))
    if worst > LOCK_PREDICTION_TOLERANCE:
        raise PhaseError(f"the weights, locked axes, bases and locked reference states do not reproduce the preregistered predictions (max difference {worst:.3e}); nothing was executed")
    return worst


# ---------------------------------------------------------------------------
# Confirmation: stage 1 (fresh frames' reference states, digested predictions), stage 2 (fresh cues), scoring.


def stage_one(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation015, lock: Mapping[str, Any], lock_012: Mapping[str, Any], *, protocol_code_commit: str, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights, head, cache, axis_T, e_axis, fpm, tlm, context = _components(model, pool, pool_010, lock, lock_012)
    if _program_record(tlm.programs) != dict(lock["program"]):
        raise pm.IncidentError("the program's frozen configuration differs from the lock's record")
    nouns = pool.single_nouns
    say("stage 1: fresh frames' reference states, validity, prediction table")
    frames_out: dict[str, Any] = {}
    states: dict[str, Any] = {}
    rows_out: list[dict[str, Any]] = []
    identities: dict[str, float] = {}
    for frame in confirmation.frames:
        template = frame.template_id
        if template not in lock["defined_templates"]:
            frames_out[frame.frame_id] = {"template_id": template, "valid": False, "template_defined": False, "note": "template excluded before the lock; nothing run"}
            continue
        state_f = ap.capture_frame_013(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        rows = reference_rows(tlm.programs, state_f.x1_all, state_f.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), check_reference_rows(rows, state_f))
        sg, pl = pm.frame_prompts(frame)
        c_a, c_b = cache.c(sg), cache.c(pl)
        positive = sum(1 for noun in nouns if c_a[noun.lexical_key] - c_b[noun.lexical_key] > 0)
        required = pm.exact_count_floor(FRAME_CUE_EFFECT_RATE, len(nouns))
        plural = measure_pair(model, weights, head, state_f, pool.plural_cue[template], pl.cue_token_id, e_axis, axis_T, nouns)
        er._max_errors(identities, er.enforce_identities(plural, plural, f"{frame.frame_id}/{pool.plural_cue[template]}"))
        head_informative = abs(plural.head_change) >= cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma
        valid = head_informative and positive >= required
        frames_out[frame.frame_id] = {"template_id": template, "valid": valid, "head_informative": head_informative, "plural_head_change": plural.head_change, "cue_effect_positive": positive, "cue_effect_required": required,
                                      "template_defined": True, "reconstruction_error": state_f.ref.reconstruction_error, "p_c": frame.p_c}
        states[frame.frame_id] = locked_state(state_f)
        for token in confirmation.tokens:
            rows_out.append(prediction_row(token["word"], frame.frame_id, template, tlm.predict_from_state(weights, state_f, token["token_id"], template)))
        say(f"  {frame.frame_id}: {'valid' if valid else 'INVALID'} (head informative {head_informative}, cue effect {positive}/{required}); p_c {frame.p_c}; {len(confirmation.tokens)} predictions")
    return {"frames": frames_out, "states": states, "state_digests": {frame_id: ap.state_digest(entry) for frame_id, entry in states.items()}, "rows": rows_out, "identities": identities,
            "token_means": token_means_from_table(rows_out, [token["word"] for token in confirmation.tokens]),
            "digest": ap.table_digest(rows_out, states), "commit": protocol_code_commit, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}


def stage_two(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation015, lock: Mapping[str, Any], lock_012: Mapping[str, Any], stage1: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    ap.assert_stage_one_digest(stage1)
    weights, head, cache, axis_T, e_axis, fpm, tlm, context = _components(model, pool, pool_010, lock, lock_012)
    nouns = pool.single_nouns
    identities: dict[str, float] = {}
    say("stage 2: fresh cues in the exposed frames (Y1) and in the valid fresh frames (Y2)")
    pairs_exposed: dict[str, dict[str, Any]] = {}
    for frame in pool.frames:
        if frame.template_id not in lock["defined_templates"]:
            continue
        state_f = ap.capture_frame_013(model, head, pool.reference_prompt(frame), nouns, axis_T)
        locked = lock["locked_states"][frame.frame_id]
        if locked["p_c"] != state_f.p_c:
            raise pm.IncidentError(f"{frame.frame_id}: the cue position differs from the locked one")
        drift = max(max(float((x.double() - torch.tensor(y, dtype=torch.float64)).abs().max()) for x, y in zip(state_f.x1_all, locked["x1_all"])), max(float((x.double() - torch.tensor(y, dtype=torch.float64)).abs().max()) for x, y in zip(state_f.x2_all, locked["x2_all"])))
        if drift > LOCKED_STATE_TOLERANCE:
            raise pm.IncidentError(f"{frame.frame_id}: the re-captured reference state differs from the locked one (max {drift:.2e})")
        rows = reference_rows(tlm.programs, state_f.x1_all, state_f.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), check_reference_rows(rows, state_f))
        plural_name = pool.plural_cue[frame.template_id]
        plural = measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_015(record, plural, context=context, state=state_f, rows=rows)
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
        if ap.state_digest(locked_state(state_f)) != stage1["state_digests"][frame.frame_id]:
            raise pm.IncidentError(f"{frame.frame_id}: the re-captured reference state differs from stage 1's digested state")
        rows = reference_rows(tlm.programs, state_f.x1_all, state_f.x2_all)
        sg, pl = pm.frame_prompts(frame)
        plural = measure_pair(model, weights, head, state_f, pool.plural_cue[frame.template_id], pl.cue_token_id, e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_015(record, plural, context=context, state=state_f, rows=rows)
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


def _score_set(pairs: Mapping[str, Mapping[str, Any]], table: Mapping[str, Mapping[str, Any]], words: Sequence[str], *, min_frames: int) -> dict[str, Any]:
    """Token means of ĉ_ΔA (from the table) and c_ΔA over identical frame sets, and the pooled entry R² per layer over the scored pairs; the four floors."""
    tokens_out, scored_pairs = {}, []
    for word in words:
        keys = sorted(key for key in pairs if key.split("|")[0] == word and key in table)
        analyses = [pairs[key] for key in keys]
        predictions = [table[key] for key in keys]
        tokens_out[word] = {"scored": len(keys) >= min_frames, "n_valid_frames": len(keys), **_token_means(analyses, predictions)}
        if len(keys) >= min_frames:
            scored_pairs.extend(zip(analyses, predictions))
    scored = [word for word, entry in tokens_out.items() if entry["scored"]]
    every_pair = [(pairs[key], table[key]) for key in sorted(pairs) if key in table]
    result: dict[str, Any] = {"tokens": tokens_out, "scored_tokens": scored, "n_pairs": len(scored_pairs),
                              "per_frame": _per_frame([a for a, _ in every_pair], [p for _, p in every_pair])}  # every measured pair of the set, reported regardless of the outcome
    if len(scored) >= 2:
        analyses = [a for a, _ in scored_pairs]
        predictions = [p for _, p in scored_pairs]
        c = comparison([tokens_out[w]["c_hat_mean"] for w in scored], [tokens_out[w]["c_mean"] for w in scored])
        pair_stats = _pair_statistics(analyses, predictions)
        entry_1, entry_2 = pair_stats["level0"]["1"]["entry_r2"], pair_stats["level0"]["2"]["entry_r2"]
        failing = [name for name, ok in (("spearman", c["spearman"] is not None and c["spearman"] >= Y_SPEARMAN), ("r2", c["r2"] is not None and c["r2"] >= Y_R2),
                                         ("entry_r2_layer1", entry_1 is not None and entry_1 >= Y_ENTRY_R2), ("entry_r2_layer2", entry_2 is not None and entry_2 >= Y_ENTRY_R2)) if not ok]
        result["test"] = {**c, "entry_r2_layer1": entry_1, "entry_r2_layer2": entry_2, "passed": not failing, "failing": failing, "floors": {"spearman": Y_SPEARMAN, "r2": Y_R2, "entry_r2": Y_ENTRY_R2}}
        result["axis"] = {"token_means": comparison([tokens_out[w]["c_axis_mean"] for w in scored], [tokens_out[w]["c_mean"] for w in scored]), "entries": pair_stats["axis"]}
        result["descriptive"] = {"pairs": pair_stats, "rungs": _rung_statistics(analyses), "ladder": _ladder_statistics(analyses, {w: tokens_out[w] for w in scored}),
                                 "token_means_c_1": comparison([tokens_out[w]["c_hat_1_mean"] for w in scored], [tokens_out[w]["c_1_mean"] for w in scored]),
                                 "token_means_c_2": comparison([tokens_out[w]["c_hat_2_mean"] for w in scored], [tokens_out[w]["c_2_mean"] for w in scored]),
                                 "token_means_diag_L0": comparison([tokens_out[w]["c_diag_L0_mean"] for w in scored], [tokens_out[w]["c_mean"] for w in scored]),
                                 "c_spread": ap.spread([tokens_out[w]["c_mean"] for w in scored])}
    return result


def score_confirmation(stage1: Mapping[str, Any], pairs_exposed: Mapping[str, Mapping[str, Any]], pairs_fresh: Mapping[str, Mapping[str, Any]], confirmation: Confirmation015, lock: Mapping[str, Any]) -> dict[str, Any]:
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
    # Y3 and the comparator rule: over the scored token means and pairs of both sets.
    means_axis, means_c, pair_items = [], [], []
    for y, pairs, table in ((y1, pairs_exposed, locked_table), (y2, pairs_fresh, stage1_table)):
        for w in y["scored_tokens"]:
            means_axis.append(y["tokens"][w]["c_axis_mean"])
            means_c.append(y["tokens"][w]["c_mean"])
            pair_items.extend((pairs[k], table[k]) for k in sorted(key for key in pairs if key.split("|")[0] == w and key in table))
    if len(means_c) >= 2:
        axis = comparison(means_axis, means_c)
        pooled = _pair_statistics([a for a, _ in pair_items], [p for _, p in pair_items])
        entry_r2 = pooled["axis"]["both"]["entry_r2"]
        evaluable = axis["r2"] is not None and entry_r2 is not None
        rejected = evaluable and axis["r2"] < Y3_R2_MAX and entry_r2 < Y3_ENTRY_R2_MAX
        results["Y3"] = {**axis, "entry_r2": entry_r2, "axis_entries": pooled["axis"], "evaluable": evaluable, "rejected": rejected, "n_token_means": len(means_c), "n_pairs": len(pair_items),
                         "floors": {"r2_max": Y3_R2_MAX, "entry_r2_max": Y3_ENTRY_R2_MAX}, "level0_on_same_data": {"token_means": comparison([y["tokens"][w]["c_hat_mean"] for y in (y1, y2) for w in y["scored_tokens"]], means_c), "entries": pooled["level0"]}}
        results["comparator"] = {"level0": pooled["level0"], "diag_L0": pooled["diag_L0"], "margin": pooled["comparator_margin"], "interaction_beyond_self_logit": pooled["interaction_beyond_self_logit"], "margin_required": COMPARATOR_MARGIN, "n_pairs": len(pair_items)}
    else:
        results["Y3"] = {"evaluable": False, "rejected": False, "n_token_means": len(means_c), "n_pairs": len(pair_items)}
        results["comparator"] = {"interaction_beyond_self_logit": None, "n_pairs": len(pair_items)}
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
    return {"label": f"{y1} | {y2} | {y3}", "Y1": y1, "Y2": y2, "Y3": y3}


# ---------------------------------------------------------------------------
# Report.


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 015 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state['confirmation_015_sha256']}`", f"- Experiment 014 lock sha256: `{state['lock_014_sha256']}`",
             f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    incidents = list(exploration.get("incidents", [])) + ([state["confirmation"]["incident"]] if isinstance(state.get("confirmation"), dict) and "incident" in state["confirmation"] else [])
    if incidents:
        lines += ["## Incidents", ""] + [f"- `{e['phase']}` at commit `{e.get('commit', '?')}` ({e['at']}): {e['message']}" for e in incidents] + [""]

    def cmp_line(c: Mapping[str, Any]) -> str:
        return f"Spearman {f(c.get('spearman'), 3)}, R² {f(c.get('r2'), 3)}, MAE {f(c.get('mae'), 4)}, bias {f(c.get('bias'), 4)} (n {c.get('n')})"

    def pooled_line(p: Mapping[str, Any]) -> str:
        return f"layer 1 entry R² {f(p['1']['entry_r2'], 3)} (TV ratio {f(p['1']['tv_ratio'], 3)}), layer 2 entry R² {f(p['2']['entry_r2'], 3)} (TV ratio {f(p['2']['tv_ratio'], 3)}), both {f(p['both']['entry_r2'], 3)}"

    def ladder_line(l: Mapping[str, Any]) -> str:
        tm = l.get("token_means", {})
        parts = [f"`{key}` pairs R² {f(l['pairs'][key]['r2'], 3)}" + (f" / token means R² {f(tm[key]['r2'], 3)}" if key in tm else "") for key in LADDER_KEYS]
        return "Ladder of decoded c_L: " + "; ".join(parts) + f"; Level-0 c_M R² {f(l['pairs_M_level0']['r2'], 3)}, c_H R² {f(l['pairs_H_level0']['r2'], 3)}; Level 1's residual max {f(l['level1_error_max'], 6)}"

    def rungs_lines(r: Mapping[str, Any]) -> list[str]:
        out = [f"  - rung `{name}`: layer 1 entry R² {f(r[name]['1']['entry_r2'], 3)} (TV {f(r[name]['1']['tv_ratio'], 3)}), layer 2 {f(r[name]['2']['entry_r2'], 3)} (TV {f(r[name]['2']['tv_ratio'], 3)}); read vs c_ΔA {cmp_line(r[name]['read_vs_c'])}" for name in RUNGS]
        d = r["diagonal_terms_rms"]
        out.append(f"  - diagonal logit change rms: layer 1 ⟨Δq,k⟩ {f(d['1']['dq_k'], 2)}, ⟨q,Δk⟩ {f(d['1']['q_dk'], 2)}, ⟨Δq,Δk⟩ {f(d['1']['dq_dk'], 2)}; layer 2 {f(d['2']['dq_k'], 2)} / {f(d['2']['q_dk'], 2)} / {f(d['2']['dq_dk'], 2)}; ‖Δx‖/‖x−μ‖ {f(r['norm_ratio_mean']['1'], 2)} / {f(r['norm_ratio_mean']['2'], 2)}; measured TV per pair {f(r['measured_tv_mean']['1'], 2)} / {f(r['measured_tv_mean']['2'], 2)}; max |ΔA_pc| {f(r['self_abs_max'], 3)}")
        return out

    if "summary" in exploration:
        x = exploration["exposed_check"]
        rep = exploration["replication"]
        lines += ["## Tier A — exposed pool (calibration record only; the floors are frozen constants)", "",
                  f"- Replication: Experiment 014 ledger {rep['experiment_014']['n']} pairs, max deviation {rep['experiment_014']['max_abs_deviation']:.2e}; Experiment 013 patterns {rep['experiment_013_patterns']['n']} pairs, max deviation {rep['experiment_013_patterns']['max_abs_deviation']:.2e}",
                  f"- Identities (Level 1, checks only): {', '.join(f'{k} {v:.1e}' for k, v in exploration['identities'].items())}; read weight vs Experiment 011 lock {exploration['read_weight_check']:.1e}",
                  f"- Program: {', '.join(f'layer {k}: d_head {v['d_head']}, rotary_dim {v['rotary_dim']}, base {v['rotary_base']:g}' for k, v in exploration['program'].items())}",
                  f"- Token means ({x['n_tokens']}): ĉ_ΔA vs c_ΔA — {cmp_line(x['token_means']['c_hat_vs_c'])}; c_ΔA spread sd {f(x['token_means']['c_spread'], 4)}; per layer: {cmp_line(x['token_means']['c_hat_1_vs_c_1'])} | {cmp_line(x['token_means']['c_hat_2_vs_c_2'])}",
                  f"- Level 0 entries ({x['n_pairs']} pairs): {pooled_line(x['pairs']['level0'])}; self weights pooled {cmp_line(x['pairs']['self_level0']['pooled'])}; pairs ĉ_ΔA vs c_ΔA {cmp_line(x['pairs']['c_hat_vs_c'])}",
                  f"- Axis-only: token means {cmp_line(x['token_means']['c_axis_vs_c'])}; entries {pooled_line(x['pairs']['axis'])}",
                  f"- Diagonal-proportional (Level-0 self weight): token means {cmp_line(x['token_means']['c_diag_L0_vs_c'])}; entries {pooled_line(x['pairs']['diag_L0'])}; margin of Level 0 over it {', '.join(f'layer {k} {f(v, 3)}' for k, v in x['pairs']['comparator_margin'].items())} → interaction beyond the self logit: {x['pairs']['interaction_beyond_self_logit']}",
                  f"- {ladder_line(x['ladder'])}",
                  "- Rungs (own state, exact arriving change; descriptive):"] + rungs_lines(x["rungs"]) + [""]
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}`; predictions sha256 `{state['lock']['predictions_sha256']}`", ""]
    confirmation = state.get("confirmation")
    if confirmation and "stage1" in confirmation:
        s1 = confirmation["stage1"]
        lines += ["## Confirmation — stage 1 (fresh frames' reference states; prediction table digested before any fresh cue prompt)", "", f"- Table rows {len(s1['rows'])}; digest `{s1['digest']}`; commit `{s1['commit']}`"]
        for frame_id, entry in s1["frames"].items():
            if entry.get("template_defined"):
                lines.append(f"  - {frame_id}: {'valid' if entry['valid'] else 'INVALID'} (plural head change {f(entry['plural_head_change'], 3)}, cue effect {entry['cue_effect_positive']}/{entry['cue_effect_required']}; p_c {entry['p_c']})")
            else:
                lines.append(f"  - {frame_id}: {entry.get('note')}")
        lines.append("")
    if confirmation and "outcome" in confirmation:
        o = confirmation["outcome"]
        lines += [f"## Confirmation — stage 2 — `{o['label']}`", ""]
        for label, key in (("Y1 (strict prospective: fresh cues × exposed frames)", "Y1"), ("Y2 (frame-conditional prospective, aggregate over the valid fresh frames: fresh cues × new frames)", "Y2")):
            y = confirmation[key]
            pre = confirmation["precondition_Y1" if key == "Y1" else "precondition_Y2"]
            if "test" in y:
                t, d = y["test"], y["descriptive"]
                lines.append(f"- {label}: scored tokens {len(y['scored_tokens'])} ({'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'}); token means ĉ_ΔA vs c_ΔA — {cmp_line(t)}; entry R² layer 1 {f(t['entry_r2_layer1'], 3)}, layer 2 {f(t['entry_r2_layer2'], 3)} → {'pass' if t['passed'] else 'FAIL'} {t['failing'] or ''}")
                lines.append(f"  - Level 0 entries: {pooled_line(d['pairs']['level0'])}; self weights pooled {cmp_line(d['pairs']['self_level0']['pooled'])}; pairs ĉ_ΔA vs c_ΔA {cmp_line(d['pairs']['c_hat_vs_c'])}; per layer token means {cmp_line(d['token_means_c_1'])} | {cmp_line(d['token_means_c_2'])}; c_ΔA spread sd {f(d['c_spread'], 4)}")
                lines.append(f"  - axis-only on this set: token means {cmp_line(y['axis']['token_means'])}; entries {pooled_line(y['axis']['entries'])}")
                lines.append(f"  - diagonal-proportional (Level-0 self weight) on this set: token means {cmp_line(d['token_means_diag_L0'])}; entries {pooled_line(d['pairs']['diag_L0'])}; margin {', '.join(f'layer {k} {f(v, 3)}' for k, v in d['pairs']['comparator_margin'].items())}")
                lines.append(f"  - {ladder_line(d['ladder'])}")
                lines += rungs_lines(d["rungs"])
            else:
                lines.append(f"- {label}: not evaluable ({len(y['scored_tokens'])} scored tokens; {'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'})")
            if key == "Y2":
                for frame_id, entry in y.get("per_frame", {}).items():
                    lines.append(f"  - frame {frame_id}: {entry['n_pairs']} pairs; Level 0 entry R² layer 1 {f(entry['level0_layer1']['entry_r2'], 3)} (TV {f(entry['level0_layer1']['tv_ratio'], 3)}), layer 2 {f(entry['level0_layer2']['entry_r2'], 3)} (TV {f(entry['level0_layer2']['tv_ratio'], 3)}); ĉ_ΔA vs c_ΔA {cmp_line(entry['c_hat_vs_c'])}; axis-only entry R² {f(entry['axis_layer1']['entry_r2'], 3)} / {f(entry['axis_layer2']['entry_r2'], 3)}")
                for frame_id, entry in confirmation["frames"].items():
                    if entry.get("template_defined") and not entry.get("valid"):
                        lines.append(f"  - frame {frame_id}: invalid at stage 1 (no fresh cue prompt run)")
        y3 = confirmation["Y3"]
        if y3.get("evaluable"):
            lines.append(f"- Y3 (axis-only alternative over both sets, {y3['n_token_means']} token means / {y3['n_pairs']} pairs): token means {cmp_line(y3)} (rejected iff R² < {Y3_R2_MAX}); pooled entry R² {f(y3['entry_r2'], 3)} (rejected iff < {Y3_ENTRY_R2_MAX}) → {'REJECTED' if y3['rejected'] else 'NOT REJECTED'}; Level 0 on the same data: token means {cmp_line(y3['level0_on_same_data']['token_means'])}, entries {pooled_line(y3['level0_on_same_data']['entries'])}")
        else:
            lines.append(f"- Y3: not evaluable ({y3.get('n_token_means')} token means)")
        comp = confirmation.get("comparator", {})
        if comp.get("interaction_beyond_self_logit") is not None:
            lines.append(f"- Comparator rule over both sets ({comp['n_pairs']} pairs): Level 0 {pooled_line(comp['level0'])}; Level-0 diagonal-proportional {pooled_line(comp['diag_L0'])}; margins {', '.join(f'layer {k} {f(v, 3)}' for k, v in comp['margin'].items())} (required ≥ {comp['margin_required']} at both layers) → "
                         + ("Level 0 decodes the query/key interaction beyond the self-logit effect" if comp["interaction_beyond_self_logit"] else "most of the pattern change is captured by the self-logit change with proportional redistribution (narrower wording)"))
        lines += ["", "| token | class | Y1 frames | Y1 ĉ_ΔA | Y1 c_ΔA | Y2 frames | Y2 ĉ_ΔA | Y2 c_ΔA | axis ĉ (Y1) | diag-prop ĉ (Y1) | arrival read (Y1) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        categories = {token["word"]: token["category"] for token in confirmation.get("tokens_meta", [])}
        for word in sorted(confirmation["Y1"]["tokens"], key=lambda w: -(confirmation["Y1"]["tokens"][w].get("c_hat_mean") or -9)):
            a, b = confirmation["Y1"]["tokens"][word], confirmation["Y2"]["tokens"].get(word, {})
            lines.append(f"| {word} | {categories.get(word, '')} | {a.get('n_valid_frames')} | {f(a.get('c_hat_mean'), 4)} | {f(a.get('c_mean'), 4)} | {b.get('n_valid_frames')} | {f(b.get('c_hat_mean'), 4)} | {f(b.get('c_mean'), 4)} | {f(a.get('c_axis_mean'), 4)} | {f(a.get('c_diag_L0_mean'), 4)} | {f(a.get('arrival_read_mean'), 3)} |")
        lines += ["", "Measured / predicted self-attention weight change ΔA_h(p_c, p_c), token means over the Y1 frames:", "", "| token | " + " | ".join(HEAD_KEYS) + " |", "|---|" + "---|" * len(HEAD_KEYS)]
        for word in sorted(confirmation["Y1"]["tokens"], key=lambda w: -(confirmation["Y1"]["tokens"][w].get("c_hat_mean") or -9)):
            a = confirmation["Y1"]["tokens"][word]
            if a.get("self_mean"):
                lines.append(f"| {word} | " + " | ".join(f"{f(a['self_mean'][key], 2)} / {f(a['self_hat_mean'][key], 2)}" for key in HEAD_KEYS) + " |")
        lines.append("")
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
