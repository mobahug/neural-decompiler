"""Experiment 016: what the frame contributes — the frame-conditioned token-local Q/K model.

Experiment 015's token-local cue-change model (its Level 0) computed the cue position's query, key and value changes at
the Experiment 012 template-mean bases and missed its frame-conditional floors at layer 2. This module adds three
literal reference-state channels and nothing else: (A) the frame's own reference query and key vectors at the cue
position as the operands of the self-logit's bilinear form; (B) the frame's LayerNorm scale before the change and,
derived algebraically from the *predicted* change, after it; (C) block 1's MLP evaluated at the frame's own cue
residual. Every channel is a function of the frame's reference run and the weights; nothing is read from a fresh cue
forward pass. The channels are switches on one code path, so the scale-only alternative, the three ablations,
Experiment 015's Level 0 (all switches off, recovered to 1e-12) and Level 1 (all switches on plus the discarded
renormalization term, recovered to 1e-12) are the same computation. Predictions for new cues in the 54 exposed frames
are locked before any fresh prompt (Y1); predictions for new cues in twelve new frames are computed at confirm stage 1
and digested before any fresh cue prompt (Y2, with a per-frame anti-collapse guard); the scale-only alternative is
scored on both (Y3). Constants are copied from design revision 2.
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
from . import attention_patterns as atp
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

EXPERIMENT_DIR = "experiments/016-frame-conditioned-qk"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
INHERITED_015_EXTRACT_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-015-pair-extract.json"
EXPERIMENT_011_LOCK_PATH = er.LOCK_RELATIVE_PATH
EXPERIMENT_012_LOCK_PATH = lc.LOCK_RELATIVE_PATH
EXPERIMENT_013_LOCK_PATH = ap.LOCK_RELATIVE_PATH
EXPERIMENT_014_LOCK_PATH = nf.LOCK_RELATIVE_PATH
EXPERIMENT_015_LOCK_PATH = atp.LOCK_RELATIVE_PATH
EXPERIMENT_013_CONFIRMATION_PATH = ap.CONFIRMATION_RELATIVE_PATH
EXPERIMENT_014_CONFIRMATION_PATH = nf.CONFIRMATION_RELATIVE_PATH
EXPERIMENT_015_CONFIRMATION_PATH = atp.CONFIRMATION_RELATIVE_PATH
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260924
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
INHERITED_SCHEMA_VERSION = 1
PHASES = ("explore", "lock", "confirm", "report")

Y_SPEARMAN = 0.90  # token means of ĉ_ΔA against c_ΔA
Y_R2 = 0.85
Y_ENTRY_R2 = 0.85  # pooled entry R², separately at layer 1 and at layer 2
FRAME_GUARD_R2 = 0.80  # Y2: every valid fresh frame's pooled layer-2 entry R² over its scored pairs
Y3_ENTRY_R2_MAX = 0.85  # the scale-only alternative is rejected iff its pooled layer-2 entry R² over both sets is below this …
Y3_MARGIN = 0.15  # … and at least this far below Level 0-F's
COMPARATOR_MARGIN = atp.COMPARATOR_MARGIN  # descriptive, as in Experiment 015
RECOVERY_TOLERANCE = 1e-9  # the switch-off recovery of Experiment 015's Level 0 and the all-channels-plus-remainder recovery of Level 1 (rows, absolute)
MIN_VALID_FRAMES = 8
MIN_VALID_FRAMES_PER_TOKEN = 3
MIN_SCORED_TOKENS = 16
FRAME_CUE_EFFECT_RATE = 108 / 120
LOCK_PREDICTION_TOLERANCE = 1e-9
REPLICATION_TOLERANCE = 1e-6
LOCKED_STATE_TOLERANCE = 1e-9
EXPECTED_EXTRACT_SIZE_015 = 6180  # Experiment 015's 4884 exposed pairs + its 1152 + 144 confirmed pairs
HEAD_LAYERS = atp.HEAD_LAYERS
N_HEADS = atp.N_HEADS
HEAD_KEYS = atp.HEAD_KEYS
CHANNEL_NAMES = ("operands", "scale", "operating_point")
ABLATIONS = ("ablate_operands", "ablate_scale", "ablate_operating_point")
SINGLES = ("only_operands", "only_scale", "only_operating_point")  # each channel alone over Experiment 015's Level 0 (only_scale is the scale-only alternative)
LADDER_RUNGS = (*ABLATIONS, *SINGLES, "level0_015", "level1", "diag_oracle")
LADDER_KEYS = ("c_012", "c_013_own", "c_015_level0", "c_L_level0F", "c_L_level1")
PREDICTION_COLUMNS = ("token", "frame_id", "template", "p_c", "rows_level0F", "self_level0F", "c_hat_1", "c_hat_2", "c_hat", "rows_scale_only", "c_scale_only", "rows_diag_L0F", "c_diag_L0F",
                      "c_ablate_operands", "c_ablate_scale", "c_ablate_operating_point", "c_level0_015", "sigma_ratio", "arrival_read", "c_L_level0F", "c_M_level0F", "c_H_level0F")

CANDIDATES: dict[str, tuple[str, ...]] = {
    "determiner-like": ("initial", "upper", "respective", "individual", "specific", "separate", "distinct", "remaining", "spare", "select"),
    "ordinal-or-numeral": ("twentieth", "quarter", "twin", "dual", "tens", "double", "triple", "couple", "pair", "single"),
    "quantity": ("unlimited", "countable", "total", "insufficient", "extensive", "overall", "average", "typical", "rare", "frequent"),
    "possessive-or-pronoun": ("oneself", "myself", "yourselves", "whoever", "yourself", "themselves", "who", "thee"),
    "adjective": ("bright", "orange", "pink", "brown", "grey", "heavy", "gentle", "tiny", "giant", "silver", "purple", "yellow"),
}
QUOTAS = {"determiner-like": 5, "ordinal-or-numeral": 4, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 6}
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The garden yields {cue}"),
    ("cardinal", "The factory assembles {cue}"),
    ("cardinal", "The kitchen serves {cue}"),
    ("cardinal", "The workshop builds {cue}"),
    ("quantifier", "The essay explores {cue}"),
    ("quantifier", "The teacher explains {cue}"),
    ("quantifier", "The guide recommends {cue}"),
    ("quantifier", "The seminar addresses {cue}"),
    ("coordinated-adjective", "Liam and Ava stored {cue} cold"),
    ("coordinated-adjective", "Zoe and Theo shipped {cue} fresh"),
    ("coordinated-adjective", "Nina and Arjun mixed {cue} thick"),
    ("coordinated-adjective", "Elif and Marco tied {cue} tight"),
)
FRAME_ID_TAG = "016"
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", *atp.SCIENTIFIC_PATH_PREFIXES[1:])
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)
OUTCOME_Y1 = ("FRAME_CHANNELS_PREDICTED_TOKENS", "FRAME_CHANNELS_NOT_PREDICTED_TOKENS", "PRECONDITION_FAILED_TOKENS")
OUTCOME_Y2 = ("FRAME_CHANNELS_PREDICTED_FRAMES_CONDITIONAL", "FRAME_CHANNELS_NOT_PREDICTED_FRAMES_CONDITIONAL", "PRECONDITION_FAILED_FRAMES")
OUTCOME_Y3 = ("SCALE_ONLY_REJECTED", "SCALE_ONLY_NOT_REJECTED", "SCALE_ONLY_NOT_EVALUABLE")


class PhaseError(pm.PhaseError):
    """Protocol violation in the Experiment 016 phase machinery."""


# ---------------------------------------------------------------------------
# The channels and the model.


@dataclass(frozen=True)
class Channels:
    """Which frame channels are on: A (self-logit operands), B (scale), C (block 1's operating point); ``remainder`` adds the discarded renormalization term (Level 1)."""

    operands: bool = True
    scale: bool = True
    operating_point: bool = True
    remainder: bool = False

    @property
    def name(self) -> str:
        if self.remainder:
            return "level1"
        on = [name for name in CHANNEL_NAMES if getattr(self, name)]
        if len(on) == 3:
            return "level0F"
        if not on:
            return "level0_015"
        if on == ["scale"]:
            return "scale_only"
        if len(on) == 2:
            return "ablate_" + next(name for name in CHANNEL_NAMES if not getattr(self, name))
        return "only_" + on[0]


LEVEL0F = Channels()
SCALE_ONLY = Channels(operands=False, scale=True, operating_point=False)
LEVEL0_015 = Channels(operands=False, scale=False, operating_point=False)
LEVEL1 = Channels(remainder=True)
ABLATION_CHANNELS = {"ablate_operands": Channels(operands=False), "ablate_scale": Channels(scale=False), "ablate_operating_point": Channels(operating_point=False)}
SINGLE_CHANNELS = {"only_operands": Channels(operands=True, scale=False, operating_point=False), "only_scale": SCALE_ONLY, "only_operating_point": Channels(operands=False, scale=False, operating_point=True)}


def _sigma(x: torch.Tensor, eps: float) -> torch.Tensor:
    centred = x - x.mean()
    return torch.sqrt((centred * centred).mean() + eps)


@dataclass(frozen=True)
class FrameChannelModel:
    """Level 0-F and its switch variants on one code path; every prediction takes the token identity, the locked bases, axes and read, the weights and the frame's reference residuals."""

    read: lc.CorrectionRead
    lw: lc.LayerWeights
    programs: Mapping[int, atp.LayerProgram]
    bases_012: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    d_E: torch.Tensor

    def normalized_change(self, layer: int, base: torch.Tensor, x_f: torch.Tensor, dx: torch.Tensor, channels: Channels) -> dict[str, Any]:
        """n_Δ = γ ⊙ [(x̄ + Δ̂x)_c/σ' − x̄_c/σ] with σ, σ' the frame's (channel B) or the template's; plus the remainder γ ⊙ (x_f − x̄)_c (1/σ' − 1/σ) for Level 1."""
        program = self.programs[layer]
        base, x_f, dx = base.double(), x_f.double(), dx.double()
        eps = program.eps
        sigma_ref, sigma_after = (_sigma(x_f, eps), _sigma(x_f + dx, eps)) if channels.scale or channels.remainder else (_sigma(base, eps), _sigma(base + dx, eps))
        shifted = base + dx
        n = program.ln_w * ((shifted - shifted.mean()) / sigma_after - (base - base.mean()) / sigma_ref)
        if channels.remainder:
            deviation = x_f - base
            n = n + program.ln_w * (deviation - deviation.mean()) * (1.0 / sigma_after - 1.0 / sigma_ref)
        return {"n": n, "sigma_ref": float(sigma_ref), "sigma_after": float(sigma_after)}

    def layer_rows(self, layer: int, rows: atp.ReferenceRow, base: torch.Tensor, x_f: torch.Tensor, dx: torch.Tensor, channels: Channels) -> dict[str, Any]:
        """The cue row of one layer: the changes from n_Δ, the self logit with the frame's (channel A) or the template's operands."""
        program = self.programs[layer]
        change = self.normalized_change(layer, base, x_f, dx, channels)
        n = change["n"]
        dq, dk, dv = program.q_tilde(n) - program.b_Q, program.k_tilde(n) - program.b_K, program.v(n) - program.b_V
        if channels.operands or channels.remainder:
            n_ref = rows.normed_pc
        else:
            n_ref = program.normalize(base)
        q_ref, k_ref = program.q_tilde(n_ref), program.k_tilde(n_ref)
        ds = ((dq * k_ref).sum(-1) + (q_ref * dk).sum(-1) + (dq * dk).sum(-1)) / program.scale
        row = rows.row_from_changes(dq, ds)
        v_pc = rows.values[:, rows.p_c] + dv
        return {"row": row, "v_pc": v_pc, "sigma_ref": change["sigma_ref"], "sigma_after": change["sigma_after"], "dq": dq, "dk": dk}

    def predict_channels(self, weights: pm.Weights, rows: Mapping[int, atp.ReferenceRow], x1: torch.Tensor, x2: torch.Tensor, token_id: int, template: str, channels: Channels) -> dict[str, Any]:
        delta_e = self.read.encoding_delta(weights, token_id, template)
        denominator = self.read.denominator(weights, template)
        xb1, xb2 = (b.double() for b in self.bases_012[template])
        one = self.layer_rows(1, rows[1], xb1, x1, delta_e, channels)
        mlp_base = x1.double() if (channels.operating_point or channels.remainder) else xb1
        d1 = self.lw.delta_out(1, mlp_base, delta_e)
        out1 = rows[1].output_change(one["row"], one["v_pc"])
        dx2 = delta_e + d1 + out1
        two = self.layer_rows(2, rows[2], xb2, x2, dx2, channels)
        out = {"rows": {1: one["row"], 2: two["row"]}, "values": {1: one["v_pc"], 2: two["v_pc"]}, "dx2": dx2, "delta_e": delta_e, "denominator": denominator,
               "sigma": {1: (one["sigma_ref"], one["sigma_after"]), 2: (two["sigma_ref"], two["sigma_after"])}}
        out["c"] = {layer: sum(self.read.inner(rows[layer].pattern_change_vectors(out["rows"][layer], out["values"][layer])[h]) for h in range(N_HEADS)) / denominator for layer in HEAD_LAYERS}
        mlp2_base = x2.double() if channels.remainder else xb2  # channel C is block 1 only; the decoded c_L keeps block 2's MLP at the template base except for Level 1
        mlp = {1: d1, 2: self.lw.delta_out(2, mlp2_base, dx2)}
        attention = {1: out1, 2: rows[2].output_change(two["row"], two["v_pc"])}
        out["c_M"] = self.read.inner(mlp[1] + mlp[2]) / denominator
        out["c_H"] = self.read.inner(attention[1] + attention[2]) / denominator
        out["c_L"] = out["c_M"] + out["c_H"]
        out["arrival_read"] = self.read.inner(dx2) / denominator
        return out

    def parts(self, weights: pm.Weights, rows: Mapping[int, atp.ReferenceRow], x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], token_id: int, template: str) -> dict[str, Any]:
        """Level 0-F, the scale-only alternative, the ablations, Experiment 015's Level 0 and the table entry built from them."""
        p_c = rows[1].p_c
        x1, x2 = x1_all[p_c].double(), x2_all[p_c].double()
        full = self.predict_channels(weights, rows, x1, x2, token_id, template, LEVEL0F)
        scale_only = self.predict_channels(weights, rows, x1, x2, token_id, template, SCALE_ONLY)
        ablations = {name: self.predict_channels(weights, rows, x1, x2, token_id, template, channels) for name, channels in ABLATION_CHANNELS.items()}
        singles = {name: self.predict_channels(weights, rows, x1, x2, token_id, template, channels) for name, channels in SINGLE_CHANNELS.items() if name != "only_scale"}
        level0_015 = self.predict_channels(weights, rows, x1, x2, token_id, template, LEVEL0_015)
        diag = {layer: rows[layer].proportional(full["rows"][layer][:, p_c]) for layer in HEAD_LAYERS}
        c_diag = {layer: sum(self.read.inner(rows[layer].pattern_change_vectors(diag[layer], full["values"][layer])[h]) for h in range(N_HEADS)) / full["denominator"] for layer in HEAD_LAYERS}
        entry = {"p_c": p_c,
                 "rows_level0F": {str(layer): atp.rows_to_json(full["rows"][layer] - rows[layer].A_ref, layer) for layer in HEAD_LAYERS},
                 "self_level0F": {ap.head_key(layer, h): float(full["rows"][layer][h, p_c] - rows[layer].A_ref[h, p_c]) for layer in HEAD_LAYERS for h in range(N_HEADS)},
                 "c_hat_1": full["c"][1], "c_hat_2": full["c"][2], "c_hat": full["c"][1] + full["c"][2],
                 "rows_scale_only": {str(layer): atp.rows_to_json(scale_only["rows"][layer] - rows[layer].A_ref, layer) for layer in HEAD_LAYERS}, "c_scale_only": scale_only["c"][1] + scale_only["c"][2],
                 "rows_diag_L0F": {str(layer): atp.rows_to_json(diag[layer] - rows[layer].A_ref, layer) for layer in HEAD_LAYERS}, "c_diag_L0F": c_diag[1] + c_diag[2],
                 **{f"c_{name}": ablations[name]["c"][1] + ablations[name]["c"][2] for name in ABLATIONS},
                 "c_level0_015": level0_015["c"][1] + level0_015["c"][2],
                 "sigma_ratio": {str(layer): full["sigma"][layer][1] / full["sigma"][layer][0] for layer in HEAD_LAYERS},
                 "arrival_read": full["arrival_read"], "c_L_level0F": full["c_L"], "c_M_level0F": full["c_M"], "c_H_level0F": full["c_H"]}
        singles["only_scale"] = scale_only
        return {"rows": rows, "full": full, "scale_only": scale_only, "ablations": ablations, "singles": singles, "level0_015": level0_015, "diag_rows": diag, "entry": entry}

    def predict(self, weights: pm.Weights, x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], token_id: int, template: str) -> dict[str, Any]:
        return self.parts(weights, atp.reference_rows(self.programs, x1_all, x2_all), x1_all, x2_all, token_id, template)["entry"]

    def predict_from_state(self, weights: pm.Weights, state: ap.FrameState013, token_id: int, template: str) -> dict[str, Any]:
        return self.predict(weights, state.x1_all, state.x2_all, token_id, template)

    def predict_from_locked(self, weights: pm.Weights, locked: Mapping[str, Any], token_id: int, template: str) -> dict[str, Any]:
        return self.predict(weights, [torch.tensor(x, dtype=torch.float64) for x in locked["x1_all"]], [torch.tensor(x, dtype=torch.float64) for x in locked["x2_all"]], token_id, template)

    def as_token_local(self) -> atp.TokenLocalModel:
        """Experiment 015's model on the same ingredients (for the recovery identities and the decoded-c_L ladder)."""
        return atp.TokenLocalModel(self.read, self.lw, self.programs, self.bases_012, self.d_E)


def model_from_locks(lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lw: lc.LayerWeights, programs: Mapping[int, atp.LayerProgram], reference_ids: Mapping[str, int], plural_ids: Mapping[str, int]) -> FrameChannelModel:
    read = lc.read_from_lock_011(lock_011, reference_ids, plural_ids)
    return FrameChannelModel(read, lw, programs, lc.bases_from_json(lock_012["base_states"]), torch.tensor(lock_011["axes_vectors"]["R0"], dtype=torch.float64))


def recovery_errors(model: FrameChannelModel, weights: pm.Weights, rows: Mapping[int, atp.ReferenceRow], x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], token_id: int, template: str) -> dict[str, float]:
    """The two algebraic recoveries: all switches off equals Experiment 015's Level 0; all channels plus the remainder equals Level 1 (both exact up to arithmetic)."""
    p_c = rows[1].p_c
    x1, x2 = x1_all[p_c].double(), x2_all[p_c].double()
    token_local = model.as_token_local()
    zero = token_local.level_zero(weights, rows, token_id, template)
    one = token_local.level_one(weights, rows, x1, x2, token_id, template)
    off = model.predict_channels(weights, rows, x1, x2, token_id, template, LEVEL0_015)
    on = model.predict_channels(weights, rows, x1, x2, token_id, template, LEVEL1)
    return {"level0_015_recovery": max(float((off["rows"][layer] - zero["rows"][layer]).abs().max()) for layer in HEAD_LAYERS),
            "level1_recovery": max(float((on["rows"][layer] - one["rows"][layer]).abs().max()) for layer in HEAD_LAYERS)}


locked_state = atp.locked_state


def prediction_row(word: str, frame_id: str, template: str, prediction: Mapping[str, Any]) -> dict[str, Any]:
    row = {"token": word, "frame_id": frame_id, "template": template}
    row.update({key: prediction[key] for key in PREDICTION_COLUMNS[3:]})
    return row


# ---------------------------------------------------------------------------
# Measurement per pair.

measure_pair = atp.measure_pair


@dataclass(frozen=True)
class AnalysisContext:
    model: FrameChannelModel
    fpm: ap.FrozenPatternModel
    heads: ap.HeadSet
    weights: pm.Weights
    axis_T: pm.SiteAxis


def analyse_pair_016(record: ra.Attribution, plural: ra.Attribution, *, context: AnalysisContext, state: ap.FrameState013, rows: Mapping[int, atp.ReferenceRow]) -> dict[str, Any] | None:
    """The measured pattern change and its read, I1–I3, the recovery identities, Level 0-F with its statistics, the alternative, the comparators, the ablations, the remainders and the ladder."""
    fcm, weights = context.model, context.weights
    base = lc.analyse_pair(record, plural, read=fcm.read, lw=fcm.lw, weights=weights, state=state, axis_T=context.axis_T)
    if base is None:
        return None
    frame = state.ref.frame
    p_c, template = frame.p_c, frame.template_id
    denominator = fcm.read.denominator(weights, template)
    x_ref = {1: state.x1.double(), 2: state.x2.double()}
    x_patch = {layer: record.extra[pm.site_label((f"RESID_PRE.L{layer}", p_c))].double() for layer in HEAD_LAYERS}
    A_patch = {layer: record.extra[pm.site_label((f"ATTN_PATTERN.L{layer}", p_c))][:, : p_c + 1].double() for layer in HEAD_LAYERS}
    identities: dict[str, float] = {}
    identities["I1_patched_rows"] = max(float((rows[layer].row_from_state(fcm.programs[layer].normalize(x_patch[layer])) - A_patch[layer]).abs().max()) for layer in HEAD_LAYERS)
    if identities["I1_patched_rows"] > atp.ROW_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{frame.frame_id}/{record.token}: the program does not reproduce the captured patched pattern rows (max {identities['I1_patched_rows']:.2e})")
    split = {key: ap.head_identity(context.heads, key, state, record) for key in HEAD_KEYS}
    identities["I3_head_split"] = max(entry["error"] for entry in split.values())
    measured_c = {layer: sum(fcm.read.inner(split[key]["pattern_change"]) for key in HEAD_KEYS if ap.layer_of(key) == layer) / denominator for layer in HEAD_LAYERS}
    A_ref = {1: state.A1[:, : p_c + 1].double(), 2: state.A2[:, : p_c + 1].double()}
    measured_rows = {layer: A_patch[layer] - A_ref[layer] for layer in HEAD_LAYERS}
    token_local = fcm.as_token_local()
    one = token_local.level_one(weights, rows, state.x1, state.x2, record.token_id, template)
    identities["I2_chain"] = lc.relative_vector_error(one["dx"][2], x_patch[2] - x_ref[2], x_ref[2])
    if identities["I2_chain"] > atp.CHAIN_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{frame.frame_id}/{record.token}: the exact chain does not reproduce the captured residual before block 2 (relative error {identities['I2_chain']:.2e})")
    recoveries = recovery_errors(fcm, weights, rows, state.x1_all, state.x2_all, record.token_id, template)
    identities["level0_015_recovery"] = recoveries["level0_015_recovery"]
    identities["level1_recovery"] = recoveries["level1_recovery"]
    if max(recoveries.values()) > RECOVERY_TOLERANCE:
        raise pm.IncidentError(f"{frame.frame_id}/{record.token}: the channel switches do not recover Experiment 015's Level 0 and Level 1 ({recoveries})")
    parts = fcm.parts(weights, rows, state.x1_all, state.x2_all, record.token_id, template)
    prediction, full = parts["entry"], parts["full"]
    measured_json = {str(layer): atp.rows_to_json(measured_rows[layer], layer) for layer in HEAD_LAYERS}
    statistics = {name: {str(layer): atp._statistics_of(measured_json, prediction[key], layer) for layer in HEAD_LAYERS} for name, key in (("level0F", "rows_level0F"), ("scale_only", "rows_scale_only"), ("diag_L0F", "rows_diag_L0F"))}
    ladder_rows = {name: parts["ablations"][name]["rows"] for name in ABLATIONS} | {name: parts["singles"][name]["rows"] for name in SINGLES} | {"level0_015": parts["level0_015"]["rows"], "level1": one["rows"],
                                                                                                                                                       "diag_oracle": {layer: rows[layer].proportional(one["rows"][layer][:, p_c]) for layer in HEAD_LAYERS}}
    ladder_statistics = {name: {str(layer): atp.row_statistics(entry[layer] - rows[layer].A_ref, measured_rows[layer]) for layer in HEAD_LAYERS} for name, entry in ladder_rows.items()}
    # Remainders: the scale of the captured patched residual against the predicted post-change scale (never fed back), and the renormalization term's effect (Level 1 − Level 0-F).
    scale_remainder = {str(layer): float(_sigma(x_patch[layer], fcm.programs[layer].eps)) - full["sigma"][layer][1] for layer in HEAD_LAYERS}
    thirteen = context.fpm.predict_from_state(weights, state, record.token_id, template)
    ladder = {"c_012": thirteen["c_012"], "c_013_own": thirteen["c_L_hat"], "c_015_level0": parts["level0_015"]["c_L"], "c_L_level0F": full["c_L"], "c_L_level1": one["c_L"], "level1_error": abs(one["c_L"] - base["c_L"])}
    return {"token": record.token, "token_id": record.token_id, "frame_id": frame.frame_id, "template": template, "p_c": p_c,
            "rows": measured_json, "self": {key: split[key]["delta_A_pc"] for key in HEAD_KEYS},
            "c_1": measured_c[1], "c_2": measured_c[2], "c": measured_c[1] + measured_c[2], "c_level1": one["c"][1] + one["c"][2],
            "c_L": base["c_L"], "c_M": base["c_M"], "c_H": base["c_H"], "c_k": base["c_k"], "P1": base["P1"], "q_T": base["q_T"], "g_E": base["g_E"],
            "prediction": prediction, "statistics": statistics, "ladder_statistics": ladder_statistics, "ladder": ladder,
            "sigma": {str(layer): {"ref": full["sigma"][layer][0], "after_predicted": full["sigma"][layer][1], "after_measured": float(_sigma(x_patch[layer], fcm.programs[layer].eps))} for layer in HEAD_LAYERS},
            "norm_ratio": {str(layer): float(one["dx"][layer].norm() / (x_ref[layer] - x_ref[layer].mean()).norm()) for layer in HEAD_LAYERS},
            "scale_remainder": scale_remainder, "identities": {**base["identities"], **identities}}


def compact_analysis(analysis: Mapping[str, Any]) -> dict[str, Any]:
    prediction = {key: value for key, value in analysis["prediction"].items() if key not in ("rows_level0F", "rows_scale_only", "rows_diag_L0F")}
    return {**analysis, "prediction": prediction}


# ---------------------------------------------------------------------------
# Pool (183 tokens × 54 frames), the inherited extract, the confirmation set.


def build_pool_016(manifest: ScreeningManifest, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009, confirmation_011: er.Confirmation011,
                   confirmation_012: lc.Confirmation012, confirmation_013: ap.Confirmation013, confirmation_014: nf.Confirmation014, confirmation_015: atp.Confirmation015) -> cs.Pool008:
    base = atp.build_pool_015(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014)
    frames = base.frames + tuple(confirmation_015.frames)
    origin = dict(base.frame_origin) | {frame.frame_id: "confirmation-015" for frame in confirmation_015.frames}
    tokens = list(base.tokens)
    category, source = dict(base.token_category), dict(base.token_source)
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation_015.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"Experiment 015 token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-015"
    return cs.Pool008(frames, origin, tuple(tokens), category, source, base.nouns, base.noun_source, base.reference_ids, base.plural_cue)


EXTRACT_015_DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014", "confirmation_015", "lock_015")
EXTRACT_015_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "confirmation_014_sha256",
                             "confirmation_015_sha256", "lock_015_sha256")


def extract_entry(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {"c_dA": float(analysis["c"]), "delta_A_pc": {key: float(analysis["self"][key]) for key in HEAD_KEYS}, "c_L": float(analysis["c_L"]), "c_M": float(analysis["c_M"]), "c_H": float(analysis["c_H"])}


def inherited_extract_payload(entries: Mapping[str, Mapping[str, Any]], *, source: Mapping[str, Any], digests: Mapping[str, str], stage1_state_digests: Mapping[str, str]) -> dict[str, Any]:
    """The committed extract: per-pair quantities of Experiment 015 and the digests of its six fresh frames' stage-1 reference states (the last six exposed frames)."""
    payload = {"schema_version": INHERITED_SCHEMA_VERSION, "experiment": "015", "kind": "pair-extract",
               "description": "Derived extract of Experiment 015's recorded per-pair pattern-change read c_ΔA, the sixteen self-attention weight changes, and c_L, c_M, c_H over its 4884 exposed pairs and its 1152 + 144 confirmed pairs, with the digests of its six fresh frames' digested stage-1 reference states. Experiment 016 recomputes the quantities and requires agreement within 1e-6, and re-captures the states to their digests.",
               "source": dict(source), **{field: digests[key] for field, key in zip(EXTRACT_015_DIGEST_FIELDS, EXTRACT_015_DIGEST_KEYS)},
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "stage1_state_digests": {key: str(value) for key, value in sorted(stage1_state_digests.items())},
               "entries": {key: dict(value) for key, value in sorted(entries.items())}}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_extract(path: Path, *, digests: Mapping[str, str], expected_size: int) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited 015 extract: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "experiment", "kind", "description", "source", *EXTRACT_015_DIGEST_FIELDS, "model", "stage1_state_digests", "entries", "content_sha256"}, "inherited 015 extract")
    if payload["schema_version"] != INHERITED_SCHEMA_VERSION or payload["experiment"] != "015" or payload["kind"] != "pair-extract" or payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("inherited 015 extract schema or digest is not frozen")
    if tuple(payload[field] for field in EXTRACT_015_DIGEST_FIELDS) != tuple(digests[key] for key in EXTRACT_015_DIGEST_KEYS):
        raise ValueError("inherited 015 extract was recorded against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision} or len(payload["entries"]) != expected_size or len(payload["stage1_state_digests"]) != len(atp.FRESH_FRAMES):
        raise ValueError("inherited 015 extract model, size or stage-1 digests are not frozen")
    return payload


def check_extract_replication(measured: Mapping[str, Mapping[str, Any]], recorded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = set(recorded) - set(measured)
    if missing:
        raise pm.IncidentError(f"recomputed quantities lack {len(missing)} recorded pairs (e.g. {sorted(missing)[:3]})")
    worst_key, worst = None, 0.0
    for key, value in recorded.items():
        other = measured[key]
        for name in ("c_dA", "c_L", "c_M", "c_H"):
            deviation = abs(value[name] - other[name])
            if deviation > worst:
                worst_key, worst = f"{key}:{name}", deviation
        for head in HEAD_KEYS:
            deviation = abs(value["delta_A_pc"][head] - other["delta_A_pc"][head])
            if deviation > worst:
                worst_key, worst = f"{key}:{head}", deviation
    if worst > REPLICATION_TOLERANCE:
        raise pm.IncidentError(f"{worst_key}: recomputed quantity deviates from Experiment 015's record by {worst:.3e}")
    return {"passed": True, "n": len(recorded), "max_abs_deviation": worst, "worst_key": worst_key}


def fresh_tokens_016(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
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


CONFIRMATION_DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014", "confirmation_015")
CONFIRMATION_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_006_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "confirmation_014_sha256", "confirmation_015_sha256")


def build_confirmation_payload(tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> dict[str, Any]:
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_016(tokenizer, excluded_ids)
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
               "policy": {"licensing": "every fresh cue in every fresh frame and in every exposed frame", "quotas": dict(QUOTAS), "expectations": "none: the committed pattern-change predictions are the only predictions; no label is preregistered for any head, frame or cue"},
               "tokens": tokens, "frames": frames, "token_prompts": [nf._prompt_entry(tokenizer, frame, token) for frame in frame_objects for token in tokens],
               "exposed_frame_prompts": [nf._prompt_entry(tokenizer, frame, token) for frame in pool.frames for token in tokens],
               "construction": "tokenizer-only; first eligible candidates per lexical class; classes carry no expectation; no model output"}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


Confirmation016 = ap.Confirmation013


def validate_confirmation(payload: Mapping[str, Any], pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation016:
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
    return Confirmation016(dict(payload["reference_cue_ids"]), frames, tuple(payload["exposed_frame_ids"]), tuple(tokens), tuple(fresh_prompts), tuple(exposed_prompts), payload["content_sha256"])


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


def load_confirmation(path: Path, pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation016:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, pool, digests)


# ---------------------------------------------------------------------------
# Results state, phases.

DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014", "confirmation_015", "confirmation_016",
               "lock_011", "lock_012", "lock_013", "lock_014", "lock_015", "extract_015")
STATE_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "confirmation_014_sha256",
                       "confirmation_015_sha256", "confirmation_016_sha256", "lock_011_sha256", "lock_012_sha256", "lock_013_sha256", "lock_014_sha256", "lock_015_sha256", "extract_015_sha256")
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


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation016) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


comparison = lc.comparison


# ---------------------------------------------------------------------------
# Statistics.


def _token_means(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not analyses:
        return {}
    mean = lambda values: pm._mean(list(values))  # noqa: E731
    return {"n_frames": len(analyses), "frames": sorted(a["frame_id"] for a in analyses),
            "c_mean": mean(a["c"] for a in analyses), "c_1_mean": mean(a["c_1"] for a in analyses), "c_2_mean": mean(a["c_2"] for a in analyses),
            "c_hat_mean": mean(p["c_hat"] for p in predictions), "c_hat_1_mean": mean(p["c_hat_1"] for p in predictions), "c_hat_2_mean": mean(p["c_hat_2"] for p in predictions),
            "c_scale_only_mean": mean(p["c_scale_only"] for p in predictions), "c_diag_L0F_mean": mean(p["c_diag_L0F"] for p in predictions), "c_level0_015_mean": mean(p["c_level0_015"] for p in predictions),
            **{f"c_{name}_mean": mean(p[f"c_{name}"] for p in predictions) for name in ABLATIONS},
            "arrival_read_mean": mean(p["arrival_read"] for p in predictions), "sigma_ratio_mean": {str(layer): mean(p["sigma_ratio"][str(layer)] for p in predictions) for layer in HEAD_LAYERS},
            "c_L_mean": mean(a["c_L"] for a in analyses), "c_M_mean": mean(a["c_M"] for a in analyses), "c_H_mean": mean(a["c_H"] for a in analyses),
            "ladder_mean": {key: mean(a["ladder"][key] for a in analyses) for key in LADDER_KEYS},
            "self_mean": {key: mean(a["self"][key] for a in analyses) for key in HEAD_KEYS}, "self_hat_mean": {key: mean(p["self_level0F"][key] for p in predictions) for key in HEAD_KEYS}}


def _pair_statistics(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_pairs": len(analyses)}
    for name, key in (("level0F", "rows_level0F"), ("scale_only", "rows_scale_only"), ("diag_L0F", "rows_diag_L0F")):
        per_layer = {str(layer): atp.pooled_summary([atp._statistics_of(a["rows"], p[key], layer) for a, p in zip(analyses, predictions)]) for layer in HEAD_LAYERS}
        per_layer["both"] = atp.pooled_summary([atp._statistics_of(a["rows"], p[key], layer) for a, p in zip(analyses, predictions) for layer in HEAD_LAYERS])
        out[name] = per_layer
    out["self_level0F"] = {"pooled": comparison([p["self_level0F"][key] for p in predictions for key in HEAD_KEYS], [a["self"][key] for a in analyses for key in HEAD_KEYS]),
                           "per_head": {key: comparison([p["self_level0F"][key] for p in predictions], [a["self"][key] for a in analyses]) for key in HEAD_KEYS}}
    for name in ("c_hat", "c_scale_only", "c_diag_L0F", "c_level0_015", *(f"c_{a}" for a in ABLATIONS)):
        out[f"{name}_vs_c"] = comparison([p[name] for p in predictions], [a["c"] for a in analyses])
    out["c_hat_1_vs_c_1"] = comparison([p["c_hat_1"] for p in predictions], [a["c_1"] for a in analyses])
    out["c_hat_2_vs_c_2"] = comparison([p["c_hat_2"] for p in predictions], [a["c_2"] for a in analyses])
    out["comparator_margin"] = {str(layer): (out["level0F"][str(layer)]["entry_r2"] - out["diag_L0F"][str(layer)]["entry_r2"]) if out["level0F"][str(layer)]["entry_r2"] is not None and out["diag_L0F"][str(layer)]["entry_r2"] is not None else None for layer in HEAD_LAYERS}
    out["interaction_beyond_self_logit"] = all(margin is not None and margin >= COMPARATOR_MARGIN for margin in out["comparator_margin"].values())
    out["scale_only_margin"] = {str(layer): (out["level0F"][str(layer)]["entry_r2"] - out["scale_only"][str(layer)]["entry_r2"]) if out["level0F"][str(layer)]["entry_r2"] is not None and out["scale_only"][str(layer)]["entry_r2"] is not None else None for layer in HEAD_LAYERS}
    return out


def _ladder_statistics(analyses: Sequence[Mapping[str, Any]], means: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The ablation ladder in pattern space (pooled per layer) with its predeclared ordering, the remainders, and the decoded-c_L ladder."""
    out: dict[str, Any] = {"rows": {}}
    for name in LADDER_RUNGS:
        out["rows"][name] = {str(layer): atp.pooled_summary([a["ladder_statistics"][name][str(layer)] for a in analyses]) for layer in HEAD_LAYERS}
    level0F = atp.pooled_entry_r2([a["statistics"]["level0F"]["2"] for a in analyses])
    costs = {name: (level0F - out["rows"][name]["2"]["entry_r2"]) if level0F is not None and out["rows"][name]["2"]["entry_r2"] is not None else None for name in ABLATIONS}
    out["ablation_cost_layer2"] = costs
    ordered = all(costs[n] is not None for n in ABLATIONS) and costs["ablate_operands"] >= costs["ablate_operating_point"] >= costs["ablate_scale"]
    out["predeclared_ordering_holds"] = ordered
    out["renormalization_remainder_layer2"] = (out["rows"]["level1"]["2"]["entry_r2"] - level0F) if level0F is not None and out["rows"]["level1"]["2"]["entry_r2"] is not None else None
    out["scale_remainder_abs"] = {str(layer): pm._mean([abs(a["scale_remainder"][str(layer)]) for a in analyses]) for layer in HEAD_LAYERS}
    out["sigma_ratio_mean"] = {str(layer): pm._mean([a["sigma"][str(layer)]["after_predicted"] / a["sigma"][str(layer)]["ref"] for a in analyses]) for layer in HEAD_LAYERS}
    out["norm_ratio_mean"] = {str(layer): pm._mean([a["norm_ratio"][str(layer)] for a in analyses]) for layer in HEAD_LAYERS}
    out["decoded_c_L"] = {"pairs": {key: comparison([a["ladder"][key] for a in analyses], [a["c_L"] for a in analyses]) for key in LADDER_KEYS}, "level1_error_max": max((a["ladder"]["level1_error"] for a in analyses), default=None)}
    if means:
        names = sorted(means)
        out["decoded_c_L"]["token_means"] = {key: comparison([means[n]["ladder_mean"][key] for n in names], [means[n]["c_L_mean"] for n in names]) for key in LADDER_KEYS}
    return out


def statistics_for(pairs: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, Any]:
    by_token = {name: [(a, p) for a, p in zip(pairs, predictions) if a["token"] == name] for name in tokens}
    means = {name: _token_means([a for a, _ in rows], [p for _, p in rows]) for name, rows in by_token.items() if rows}
    names = sorted(means)
    m = lambda key: [means[n][key] for n in names]  # noqa: E731
    token_means = {"c_hat_vs_c": comparison(m("c_hat_mean"), m("c_mean")), "c_scale_only_vs_c": comparison(m("c_scale_only_mean"), m("c_mean")), "c_diag_L0F_vs_c": comparison(m("c_diag_L0F_mean"), m("c_mean")),
                   "c_level0_015_vs_c": comparison(m("c_level0_015_mean"), m("c_mean")), "c_hat_1_vs_c_1": comparison(m("c_hat_1_mean"), m("c_1_mean")), "c_hat_2_vs_c_2": comparison(m("c_hat_2_mean"), m("c_2_mean")),
                   **{f"c_{name}_vs_c": comparison(m(f"c_{name}_mean"), m("c_mean")) for name in ABLATIONS}, "c_spread": ap.spread(m("c_mean"))}
    return {"n_tokens": len(names), "n_pairs": len(pairs), "token_means": token_means, "pairs": _pair_statistics(pairs, predictions), "ladder": _ladder_statistics(pairs, means),
            "per_frame": _per_frame(pairs, predictions), "token_means_table": means}


def _per_frame(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for frame_id in sorted({a["frame_id"] for a in analyses}):
        mine = [(a, p) for a, p in zip(analyses, predictions) if a["frame_id"] == frame_id]
        entry = {"n_pairs": len(mine), "c_hat_vs_c": comparison([p["c_hat"] for _, p in mine], [a["c"] for a, _ in mine])}
        for layer in HEAD_LAYERS:
            entry[f"level0F_layer{layer}"] = atp.pooled_summary([atp._statistics_of(a["rows"], p["rows_level0F"], layer) for a, p in mine])
            entry[f"scale_only_layer{layer}"] = atp.pooled_summary([atp._statistics_of(a["rows"], p["rows_scale_only"], layer) for a, p in mine])
            entry[f"level0_015_layer{layer}"] = atp.pooled_summary([a["ladder_statistics"]["level0_015"][str(layer)] for a, _ in mine])
        out[frame_id] = entry
    return out


def frame_minimum(per_frame: Mapping[str, Mapping[str, Any]], key: str = "level0F_layer2") -> float | None:
    values = [entry[key]["entry_r2"] for entry in per_frame.values() if entry[key]["entry_r2"] is not None]
    return min(values) if values else None


# ---------------------------------------------------------------------------
# Exploration (Tier A).


def _components(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any]) -> tuple[Any, ...]:
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    programs = atp.programs_from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    axis_T, e_axis = lc.locked_axes(lock_011, cs.stage_axes(cache, weights, pool_010))
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    fpm = ap.model_from_locks(lock_011, lock_012, lw, heads, pool.reference_ids, plural_ids)
    lc.check_read_weight(fpm.read, head, axis_T)
    fcm = model_from_locks(lock_011, lock_012, lw, programs, pool.reference_ids, plural_ids)
    return weights, head, cache, axis_T, e_axis, fpm, fcm, AnalysisContext(fcm, fpm, heads, weights, axis_T)


def _check_locked_state(state_f: ap.FrameState013, previous: Mapping[str, Any], label: str) -> None:
    if "x1_all" in previous:
        if previous.get("p_c", state_f.p_c) != state_f.p_c or len(previous["x1_all"]) != len(state_f.x1_all) or len(previous["x2_all"]) != len(state_f.x2_all):
            raise pm.IncidentError(f"{label}: the cue position or the number of positions differs from the recorded state")
        drift = max(max(float((x.double() - torch.tensor(y, dtype=torch.float64)).abs().max()) for x, y in zip(state_f.x1_all, previous["x1_all"])), max(float((x.double() - torch.tensor(y, dtype=torch.float64)).abs().max()) for x, y in zip(state_f.x2_all, previous["x2_all"])))
    else:
        drift = max(float((state_f.x1 - torch.tensor(previous["x1"], dtype=torch.float64)).abs().max()), float((state_f.x2 - torch.tensor(previous["x2"], dtype=torch.float64)).abs().max()))
    if drift > LOCKED_STATE_TOLERANCE:
        raise pm.IncidentError(f"{label}: the reference state differs from the recorded one (max {drift:.2e})")


def run_exploration(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, *, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_015: Mapping[str, Any], inherited_extract: Mapping[str, Any],
                    state: dict[str, Any], results_path: Path | None, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    nouns = pool.single_nouns
    exploration = state["exploration"]
    say("stage axes from the Experiment 010 pool, checked against the Experiment 011 lock")
    weights, head, cache, axis_T, e_axis, fpm, fcm, context = _components(model, pool, pool_010, lock_011, lock_012)
    exploration["read_weight_check"] = lc.check_read_weight(fpm.read, head, axis_T)
    exploration["axes_vectors"] = {"T": axis_T.direction.tolist(), "R0": fcm.d_E.tolist()}
    exploration["sigma_T"] = axis_T.sigma
    exploration["read_weight"] = fpm.read.weight.tolist()
    exploration["defined_templates"] = list(lock_012["defined_templates"])
    exploration["program"] = atp._program_record(fcm.programs)
    recorded = inherited_extract["entries"]
    say(f"reference states of the {len(pool.frames)} exposed frames; re-measurement of the {len(recorded)} recorded pairs with the residuals and pattern rows captured")
    locked_states: dict[str, Any] = {}
    pairs: dict[str, dict[str, Any]] = {}
    measured_extract: dict[str, dict[str, Any]] = {}
    identities: dict[str, float] = {}
    for frame in pool.frames:
        state_f = ap.capture_frame_013(model, head, pool.reference_prompt(frame), nouns, axis_T)
        rows = atp.reference_rows(fcm.programs, state_f.x1_all, state_f.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows, state_f))
        locked_states[frame.frame_id] = locked_state(state_f)
        previous = lock_015.get("locked_states", {}).get(frame.frame_id)
        if previous is not None:
            _check_locked_state(state_f, previous, frame.frame_id)
        elif frame.frame_id in inherited_extract.get("stage1_state_digests", {}):
            if ap.state_digest(locked_states[frame.frame_id]) != inherited_extract["stage1_state_digests"][frame.frame_id]:
                raise pm.IncidentError(f"{frame.frame_id}: the reference state differs from Experiment 015's digested stage-1 state")
        else:
            raise pm.IncidentError(f"{frame.frame_id}: no recorded Experiment 015 reference state to check against")
        names = [name for name, _ in pool.tokens if f"{name}|{frame.frame_id}" in recorded]
        if not names:
            continue
        plural_name = pool.plural_cue[frame.template_id]
        records = {name: measure_pair(model, weights, head, state_f, name, pool.token_id(name), e_axis, axis_T, nouns) for name in names}
        plural = records[plural_name] if plural_name in records else measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for name, record in records.items():
            analysis = analyse_pair_016(record, plural, context=context, state=state_f, rows=rows)
            key = f"{name}|{frame.frame_id}"
            if analysis is None:
                raise pm.IncidentError(f"{key}: no analysis although Experiment 015 recorded the pair")
            er._max_errors(identities, analysis["identities"])
            pairs[key] = analysis
            measured_extract[key] = extract_entry(analysis)
        say(f"  {frame.frame_id}: {len(names)} pairs")
    exploration["identities"] = identities
    exploration["locked_states"] = locked_states
    exploration["locked_state_digests"] = {frame_id: ap.state_digest(entry) for frame_id, entry in locked_states.items()}
    exploration["replication"] = {"experiment_015": check_extract_replication(measured_extract, recorded)}
    say(f"replication against Experiment 015: max deviation {exploration['replication']['experiment_015']['max_abs_deviation']:.2e}")
    analyses = list(pairs.values())
    stats = statistics_for(analyses, [a["prediction"] for a in analyses], [name for name, _ in pool.tokens])
    exploration["pairs"] = {key: compact_analysis(value) for key, value in pairs.items()}
    exploration["exposed_check"] = stats
    tm, pr = stats["token_means"]["c_hat_vs_c"], stats["pairs"]
    exploration["summary"] = {"defined_templates": exploration["defined_templates"], "spearman": tm["spearman"], "r2": tm["r2"], "entry_r2_1": pr["level0F"]["1"]["entry_r2"], "entry_r2_2": pr["level0F"]["2"]["entry_r2"],
                              "frame_minimum_layer2": frame_minimum(stats["per_frame"]), "scale_only_entry_r2_2": pr["scale_only"]["2"]["entry_r2"], "scale_only_margin": pr["scale_only_margin"],
                              "ablation_cost_layer2": stats["ladder"]["ablation_cost_layer2"], "prediction_undefined": len(exploration["defined_templates"]) < 2}
    f = cd._f
    say(f"exposed: ĉ_ΔA vs c_ΔA token means Spearman {f(tm['spearman'], 3)}, R² {f(tm['r2'], 3)}; entry R² layer 1 {f(pr['level0F']['1']['entry_r2'], 3)}, layer 2 {f(pr['level0F']['2']['entry_r2'], 3)} (per-frame min {f(exploration['summary']['frame_minimum_layer2'], 3)}); scale-only layer 2 {f(pr['scale_only']['2']['entry_r2'], 3)}")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


# ---------------------------------------------------------------------------
# Lock and its artifacts.


def prediction_table(fcm: FrameChannelModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], frames: Sequence[pm.Frame], tokens: Sequence[Mapping[str, Any]], defined: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        if frame.template_id not in defined:
            continue
        for token in tokens:
            rows.append(prediction_row(token["word"], frame.frame_id, frame.template_id, fcm.predict_from_locked(weights, locked_states[frame.frame_id], token["token_id"], frame.template_id)))
    return rows


def token_means_from_table(rows: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for word in tokens:
        mine = [row for row in rows if row["token"] == word]
        if mine:
            out[word] = {key: pm._mean([row[key] for row in mine]) for key in ("c_hat", "c_hat_1", "c_hat_2", "c_scale_only", "c_diag_L0F", "c_level0_015", *(f"c_{name}" for name in ABLATIONS), "arrival_read", "c_L_level0F")}
            out[word]["n_frames"] = len(mine)
            out[word]["sigma_ratio_mean"] = {str(layer): pm._mean([row["sigma_ratio"][str(layer)] for row in mine]) for layer in HEAD_LAYERS}
            out[word]["self_hat_mean"] = {key: pm._mean([row["self_level0F"][key] for row in mine]) for key in HEAD_KEYS}
    return out


def lock_predictions(fcm: FrameChannelModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], pool: cs.Pool008, confirmation: Confirmation016, defined: Sequence[str]) -> dict[str, Any]:
    rows = prediction_table(fcm, weights, locked_states, pool.frames, confirmation.tokens, defined)
    return {"rows": rows, "token_means": token_means_from_table(rows, [token["word"] for token in confirmation.tokens])}


def render_predictions(lock: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 016 — preregistered predictions (the frame-conditioned token-local Q/K model: reference operands, normalization scale, block 1's operating point)", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_016_sha256']}`; Experiment 015 lock sha256 `{lock['lock_015_sha256']}`",
             f"- Floors: Y1/Y2 Spearman ≥ {Y_SPEARMAN} and R² ≥ {Y_R2} on token means of ĉ_ΔA, pooled entry R² ≥ {Y_ENTRY_R2} at layer 1 and at layer 2; Y2 frame-collapse guard: every valid fresh frame's layer-2 entry R² ≥ {FRAME_GUARD_R2}; Y3 scale-only rejected iff its layer-2 entry R² < {Y3_ENTRY_R2_MAX} and ≥ {Y3_MARGIN} below Level 0-F's",
             f"- Y1 table: {len(lock['predictions']['rows'])} rows (fresh tokens × exposed frames), each with the sixteen predicted rows of Level 0-F, the scale-only rows and the diagonal-proportional rows; Y2 rows are computed at confirm stage 1 and digested before any fresh cue prompt",
             f"- Program: {', '.join(f'layer {k}: d_head {v['d_head']}, rotary_dim {v['rotary_dim']}, base {v['rotary_base']:g}' for k, v in lock['program'].items())}", "",
             "| token | class | frames | **predicted ĉ_ΔA** | layer 1 | layer 2 | scale-only ĉ | 015 Level-0 ĉ | −operands | −scale | −operating point | σ'/σ (L1, L2) | decoded c_L |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    categories = {token["word"]: token["category"] for token in lock["tokens"]}
    means = lock["predictions"]["token_means"]
    for word, m in sorted(means.items(), key=lambda kv: -kv[1]["c_hat"]):
        lines.append(f"| {word} | {categories.get(word, '')} | {m['n_frames']} | **{f(m['c_hat'], 4)}** | {f(m['c_hat_1'], 4)} | {f(m['c_hat_2'], 4)} | {f(m['c_scale_only'], 4)} | {f(m['c_level0_015'], 4)} | {f(m['c_ablate_operands'], 4)} | {f(m['c_ablate_scale'], 4)} | {f(m['c_ablate_operating_point'], 4)} | {f(m['sigma_ratio_mean']['1'], 3)}, {f(m['sigma_ratio_mean']['2'], 3)} | {f(m['c_L_level0F'], 3)} |")
    lines += ["", "Predicted self-attention weight change ΔÂ_h(p_c, p_c) per head (Level 0-F), averaged over the exposed frames:", "", "| token | " + " | ".join(HEAD_KEYS) + " |", "|---|" + "---|" * len(HEAD_KEYS)]
    for word, m in sorted(means.items(), key=lambda kv: -kv[1]["c_hat"]):
        lines.append(f"| {word} | " + " | ".join(f(m["self_hat_mean"][key], 3) for key in HEAD_KEYS) + " |")
    lines.append("")
    return "\n".join(lines)


def frozen_floors() -> dict[str, Any]:
    return {"y_spearman": Y_SPEARMAN, "y_r2": Y_R2, "y_entry_r2": Y_ENTRY_R2, "frame_guard_r2": FRAME_GUARD_R2, "y3_entry_r2_max": Y3_ENTRY_R2_MAX, "y3_margin": Y3_MARGIN, "comparator_margin": COMPARATOR_MARGIN,
            "recovery_tolerance": RECOVERY_TOLERANCE, "min_valid_frames": MIN_VALID_FRAMES, "min_valid_frames_per_token": MIN_VALID_FRAMES_PER_TOKEN, "min_scored_tokens": MIN_SCORED_TOKENS,
            "frame_cue_effect_rate": FRAME_CUE_EFFECT_RATE, "head_stage_floor": cs.STAGE_UNINFORMATIVE_FLOOR, "row_identity_tolerance": atp.ROW_IDENTITY_TOLERANCE, "chain_identity_tolerance": atp.CHAIN_IDENTITY_TOLERANCE}


def build_candidate_lock(*, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation016, predictions: Mapping[str, Any], protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    if exploration["summary"]["prediction_undefined"]:
        raise PhaseError("fewer than two templates are defined: PREDICTION_UNDEFINED, no lock")
    lock = {"schema_version": 1, "experiment": "016", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit}
    for field, key in zip(STATE_DIGEST_FIELDS, DIGEST_KEYS):
        lock[field] = digests[key] if key != "confirmation_016" else confirmation.content_sha256
    lock.update({"confirmation_prompt_keys": sorted(prompt.key for prompt in confirmation.all_prompts), "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
                 "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED}, "head": ht.HEAD_KEY, "program": exploration["program"],
                 "axes_vectors": exploration["axes_vectors"], "sigma_T": exploration["sigma_T"], "read_weight": exploration["read_weight"], "defined_templates": exploration["defined_templates"],
                 "locked_states": exploration["locked_states"], "locked_state_digests": exploration["locked_state_digests"], "floors": frozen_floors(),
                 "tokens": [dict(token) for token in confirmation.tokens], "exposed_check": exploration["exposed_check"],
                 "predictions": dict(predictions), "tier_a_results_sha256": state.get("state_sha256")})
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation016, predictions_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "016" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked or git_state.get("dirty", True):
        raise PhaseError("confirm requires the committed lock and predictions on a clean Git tree")
    expected = dict(zip(STATE_DIGEST_FIELDS, digest_tuple(digests)))
    expected["confirmation_016_sha256"] = confirmation.content_sha256
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


def assert_lock_predictions_reproduced(lock: Mapping[str, Any], recomputed: Mapping[str, Any]) -> float:
    locked_rows, fresh_rows = lock["predictions"]["rows"], recomputed["rows"]
    if len(locked_rows) != len(fresh_rows):
        raise PhaseError("the recomputed prediction table has a different number of rows; nothing was executed")
    worst = 0.0
    for a, b in zip(locked_rows, fresh_rows):
        if (a["token"], a["frame_id"], a["template"], a["p_c"]) != (b["token"], b["frame_id"], b["template"], b["p_c"]):
            raise PhaseError("the recomputed prediction table is ordered differently; nothing was executed")
        for key in PREDICTION_COLUMNS[4:]:
            worst = max(worst, atp._max_numeric_difference(a[key], b[key], f"{a['token']}/{a['frame_id']}/{key}"))
    if worst > LOCK_PREDICTION_TOLERANCE:
        raise PhaseError(f"the weights, locked axes, bases and locked reference states do not reproduce the preregistered predictions (max difference {worst:.3e}); nothing was executed")
    return worst


# ---------------------------------------------------------------------------
# Confirmation: stage 1, stage 2, scoring.


def stage_one(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation016, lock: Mapping[str, Any], lock_012: Mapping[str, Any], *, protocol_code_commit: str, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights, head, cache, axis_T, e_axis, fpm, fcm, context = _components(model, pool, pool_010, lock, lock_012)
    if atp._program_record(fcm.programs) != dict(lock["program"]):
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
        rows = atp.reference_rows(fcm.programs, state_f.x1_all, state_f.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows, state_f))
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
            rows_out.append(prediction_row(token["word"], frame.frame_id, template, fcm.predict_from_state(weights, state_f, token["token_id"], template)))
        say(f"  {frame.frame_id}: {'valid' if valid else 'INVALID'} (head informative {head_informative}, cue effect {positive}/{required}); p_c {frame.p_c}; {len(confirmation.tokens)} predictions")
    return {"frames": frames_out, "states": states, "state_digests": {frame_id: ap.state_digest(entry) for frame_id, entry in states.items()}, "rows": rows_out, "identities": identities,
            "token_means": token_means_from_table(rows_out, [token["word"] for token in confirmation.tokens]),
            "digest": ap.table_digest(rows_out, states), "commit": protocol_code_commit, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}


def stage_two(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation016, lock: Mapping[str, Any], lock_012: Mapping[str, Any], stage1: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    ap.assert_stage_one_digest(stage1)
    weights, head, cache, axis_T, e_axis, fpm, fcm, context = _components(model, pool, pool_010, lock, lock_012)
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
        _check_locked_state(state_f, locked, frame.frame_id)
        rows = atp.reference_rows(fcm.programs, state_f.x1_all, state_f.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows, state_f))
        plural_name = pool.plural_cue[frame.template_id]
        plural = measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_016(record, plural, context=context, state=state_f, rows=rows)
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
        rows = atp.reference_rows(fcm.programs, state_f.x1_all, state_f.x2_all)
        sg, pl = pm.frame_prompts(frame)
        plural = measure_pair(model, weights, head, state_f, pool.plural_cue[frame.template_id], pl.cue_token_id, e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_016(record, plural, context=context, state=state_f, rows=rows)
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


def _score_set(pairs: Mapping[str, Mapping[str, Any]], table: Mapping[str, Mapping[str, Any]], words: Sequence[str], *, min_frames: int, guard: bool) -> dict[str, Any]:
    """Token means and pooled entries over the scored pairs; the four floors, and for Y2 the frame-collapse guard over the valid frames' scored pairs."""
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
    result: dict[str, Any] = {"tokens": tokens_out, "scored_tokens": scored, "n_pairs": len(scored_pairs), "per_frame": _per_frame([a for a, _ in every_pair], [p for _, p in every_pair])}
    if len(scored) >= 2:
        analyses = [a for a, _ in scored_pairs]
        predictions = [p for _, p in scored_pairs]
        c = comparison([tokens_out[w]["c_hat_mean"] for w in scored], [tokens_out[w]["c_mean"] for w in scored])
        pair_stats = _pair_statistics(analyses, predictions)
        entry_1, entry_2 = pair_stats["level0F"]["1"]["entry_r2"], pair_stats["level0F"]["2"]["entry_r2"]
        failing = [name for name, ok in (("spearman", c["spearman"] is not None and c["spearman"] >= Y_SPEARMAN), ("r2", c["r2"] is not None and c["r2"] >= Y_R2),
                                         ("entry_r2_layer1", entry_1 is not None and entry_1 >= Y_ENTRY_R2), ("entry_r2_layer2", entry_2 is not None and entry_2 >= Y_ENTRY_R2)) if not ok]
        test: dict[str, Any] = {**c, "entry_r2_layer1": entry_1, "entry_r2_layer2": entry_2, "floors": {"spearman": Y_SPEARMAN, "r2": Y_R2, "entry_r2": Y_ENTRY_R2}}
        if guard:
            scored_per_frame = _per_frame(analyses, predictions)
            below = {frame_id: entry["level0F_layer2"]["entry_r2"] for frame_id, entry in scored_per_frame.items() if entry["level0F_layer2"]["entry_r2"] is None or entry["level0F_layer2"]["entry_r2"] < FRAME_GUARD_R2}
            test["frame_guard"] = {"floor": FRAME_GUARD_R2, "per_frame_layer2": {frame_id: entry["level0F_layer2"]["entry_r2"] for frame_id, entry in scored_per_frame.items()}, "frames_below": below, "passed": not below}
            if below:
                failing.append("frame_guard")
        test.update({"passed": not failing, "failing": failing})
        result["test"] = test
        result["scale_only"] = {"token_means": comparison([tokens_out[w]["c_scale_only_mean"] for w in scored], [tokens_out[w]["c_mean"] for w in scored]), "entries": pair_stats["scale_only"], "margin": pair_stats["scale_only_margin"]}
        result["descriptive"] = {"pairs": pair_stats, "ladder": _ladder_statistics(analyses, {w: tokens_out[w] for w in scored}),
                                 "token_means_c_1": comparison([tokens_out[w]["c_hat_1_mean"] for w in scored], [tokens_out[w]["c_1_mean"] for w in scored]),
                                 "token_means_c_2": comparison([tokens_out[w]["c_hat_2_mean"] for w in scored], [tokens_out[w]["c_2_mean"] for w in scored]),
                                 "token_means_diag_L0F": comparison([tokens_out[w]["c_diag_L0F_mean"] for w in scored], [tokens_out[w]["c_mean"] for w in scored]),
                                 "token_means_level0_015": comparison([tokens_out[w]["c_level0_015_mean"] for w in scored], [tokens_out[w]["c_mean"] for w in scored]),
                                 "c_spread": ap.spread([tokens_out[w]["c_mean"] for w in scored])}
    return result


def score_confirmation(stage1: Mapping[str, Any], pairs_exposed: Mapping[str, Mapping[str, Any]], pairs_fresh: Mapping[str, Mapping[str, Any]], confirmation: Confirmation016, lock: Mapping[str, Any]) -> dict[str, Any]:
    words = [token["word"] for token in confirmation.tokens]
    locked_table = _rows_by_pair(lock["predictions"]["rows"])
    stage1_table = _rows_by_pair(stage1["rows"])
    valid_frames = [frame_id for frame_id, entry in stage1["frames"].items() if entry.get("valid")]
    y1 = _score_set(pairs_exposed, locked_table, words, min_frames=MIN_VALID_FRAMES_PER_TOKEN, guard=False)
    y2 = _score_set(pairs_fresh, stage1_table, words, min_frames=MIN_VALID_FRAMES_PER_TOKEN, guard=True)
    results: dict[str, Any] = {"frames": stage1["frames"], "valid_frames": valid_frames, "Y1": y1, "Y2": y2, "per_frame_exposed": pairs_exposed, "per_frame_fresh": pairs_fresh}
    results["precondition_Y1"] = {"passed": len(y1["scored_tokens"]) >= MIN_SCORED_TOKENS, "scored_tokens": len(y1["scored_tokens"]), "min_scored_tokens": MIN_SCORED_TOKENS}
    results["precondition_Y2"] = {"passed": len(valid_frames) >= MIN_VALID_FRAMES and len(y2["scored_tokens"]) >= MIN_SCORED_TOKENS, "valid_frames": len(valid_frames), "scored_tokens": len(y2["scored_tokens"]),
                                  "min_valid_frames": MIN_VALID_FRAMES, "min_scored_tokens": MIN_SCORED_TOKENS}
    pair_items = []
    means_scale, means_c = [], []
    for y, pairs, table in ((y1, pairs_exposed, locked_table), (y2, pairs_fresh, stage1_table)):
        for w in y["scored_tokens"]:
            means_scale.append(y["tokens"][w]["c_scale_only_mean"])
            means_c.append(y["tokens"][w]["c_mean"])
            pair_items.extend((pairs[k], table[k]) for k in sorted(key for key in pairs if key.split("|")[0] == w and key in table))
    if len(means_c) >= 2:
        pooled = _pair_statistics([a for a, _ in pair_items], [p for _, p in pair_items])
        scale_2, full_2 = pooled["scale_only"]["2"]["entry_r2"], pooled["level0F"]["2"]["entry_r2"]
        evaluable = scale_2 is not None and full_2 is not None
        rejected = evaluable and scale_2 < Y3_ENTRY_R2_MAX and (full_2 - scale_2) >= Y3_MARGIN
        results["Y3"] = {"scale_only_entry_r2_layer2": scale_2, "level0F_entry_r2_layer2": full_2, "margin": (full_2 - scale_2) if evaluable else None, "token_means": comparison(means_scale, means_c),
                         "scale_only_entries": pooled["scale_only"], "level0F_entries": pooled["level0F"], "evaluable": evaluable, "rejected": rejected, "n_token_means": len(means_c), "n_pairs": len(pair_items),
                         "floors": {"entry_r2_max": Y3_ENTRY_R2_MAX, "margin": Y3_MARGIN}}
        results["comparator"] = {"level0F": pooled["level0F"], "diag_L0F": pooled["diag_L0F"], "margin": pooled["comparator_margin"], "interaction_beyond_self_logit": pooled["interaction_beyond_self_logit"], "margin_required": COMPARATOR_MARGIN, "n_pairs": len(pair_items)}
        results["ladder_both_sets"] = _ladder_statistics([a for a, _ in pair_items])
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
    lines = ["# Experiment 016 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state['confirmation_016_sha256']}`", f"- Experiment 015 lock sha256: `{state['lock_015_sha256']}`",
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

    def ladder_lines(l: Mapping[str, Any]) -> list[str]:
        out = [f"  - ablation ladder (layer-2 entry R²): " + "; ".join(f"`{name}` {f(l['rows'][name]['2']['entry_r2'], 3)} (cost {f(l['ablation_cost_layer2'][name], 3)})" for name in ABLATIONS)
               + f"; `level0_015` {f(l['rows']['level0_015']['2']['entry_r2'], 3)}; `level1` {f(l['rows']['level1']['2']['entry_r2'], 3)}; predeclared ordering operands ≥ operating point ≥ scale holds: {l['predeclared_ordering_holds']}",
               f"  - each channel alone over Level 0 (layer-2 entry R²): " + "; ".join(f"`{name}` {f(l['rows'][name]['2']['entry_r2'], 3)}" for name in SINGLES) + f"; oracle diagonal-proportional {f(l['rows']['diag_oracle']['2']['entry_r2'], 3)}; ‖Δ̂x‖/‖x−μ‖ {f(l['norm_ratio_mean']['1'], 2)} / {f(l['norm_ratio_mean']['2'], 2)}",
               f"  - renormalization remainder (Level 1 − Level 0-F, layer-2 entry R²) {f(l['renormalization_remainder_layer2'], 3)}; scale remainder |σ(x') − σ̂'| mean {f(l['scale_remainder_abs']['1'], 4)} / {f(l['scale_remainder_abs']['2'], 4)}; σ̂'/σ mean {f(l['sigma_ratio_mean']['1'], 3)} / {f(l['sigma_ratio_mean']['2'], 3)}",
               "  - decoded c_L: " + "; ".join(f"`{key}` pairs R² {f(l['decoded_c_L']['pairs'][key]['r2'], 3)}" + (f" / token means R² {f(l['decoded_c_L']['token_means'][key]['r2'], 3)}" if "token_means" in l["decoded_c_L"] else "") for key in LADDER_KEYS) + f"; Level 1's residual max {f(l['decoded_c_L']['level1_error_max'], 6)}"]
        return out

    def frame_lines(per_frame: Mapping[str, Any]) -> list[str]:
        return [f"  - frame {frame_id}: {entry['n_pairs']} pairs; Level 0-F entry R² layer 1 {f(entry['level0F_layer1']['entry_r2'], 3)} (TV {f(entry['level0F_layer1']['tv_ratio'], 3)}), layer 2 {f(entry['level0F_layer2']['entry_r2'], 3)} (TV {f(entry['level0F_layer2']['tv_ratio'], 3)}); "
                f"ĉ_ΔA vs c_ΔA {cmp_line(entry['c_hat_vs_c'])}; scale-only layer 2 {f(entry['scale_only_layer2']['entry_r2'], 3)}; 015 Level 0 layer 2 {f(entry['level0_015_layer2']['entry_r2'], 3)}" for frame_id, entry in per_frame.items()]

    if "summary" in exploration:
        x = exploration["exposed_check"]
        rep = exploration["replication"]["experiment_015"]
        lines += ["## Tier A — exposed pool (calibration record only; the floors are frozen constants)", "",
                  f"- Replication of Experiment 015: {rep['n']} pairs, max deviation {rep['max_abs_deviation']:.2e}",
                  f"- Identities (checks only): {', '.join(f'{k} {v:.1e}' for k, v in exploration['identities'].items())}; read weight vs Experiment 011 lock {exploration['read_weight_check']:.1e}",
                  f"- Program: {', '.join(f'layer {k}: d_head {v['d_head']}, rotary_dim {v['rotary_dim']}, base {v['rotary_base']:g}' for k, v in exploration['program'].items())}",
                  f"- Token means ({x['n_tokens']}): ĉ_ΔA vs c_ΔA — {cmp_line(x['token_means']['c_hat_vs_c'])}; c_ΔA spread sd {f(x['token_means']['c_spread'], 4)}; per layer: {cmp_line(x['token_means']['c_hat_1_vs_c_1'])} | {cmp_line(x['token_means']['c_hat_2_vs_c_2'])}",
                  f"- Level 0-F entries ({x['n_pairs']} pairs): {pooled_line(x['pairs']['level0F'])}; per-frame layer-2 minimum {f(exploration['summary']['frame_minimum_layer2'], 3)}; self weights pooled {cmp_line(x['pairs']['self_level0F']['pooled'])}; pairs ĉ_ΔA vs c_ΔA {cmp_line(x['pairs']['c_hat_vs_c'])}",
                  f"- Scale-only: token means {cmp_line(x['token_means']['c_scale_only_vs_c'])}; entries {pooled_line(x['pairs']['scale_only'])}; margin of Level 0-F over it at layer 2 {f(x['pairs']['scale_only_margin']['2'], 3)}",
                  f"- Experiment 015 Level 0 on the same pairs: token means {cmp_line(x['token_means']['c_level0_015_vs_c'])}; layer-2 entry R² {f(x['ladder']['rows']['level0_015']['2']['entry_r2'], 3)}",
                  f"- Diagonal-proportional (Level 0-F self weight): entries {pooled_line(x['pairs']['diag_L0F'])}; margin {', '.join(f'layer {k} {f(v, 3)}' for k, v in x['pairs']['comparator_margin'].items())} → interaction beyond the self logit: {x['pairs']['interaction_beyond_self_logit']}"]
        lines += ladder_lines(x["ladder"]) + [""]
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
        for label, key in (("Y1 (strict prospective: fresh cues × exposed frames)", "Y1"), ("Y2 (frame-conditional prospective, aggregate with the frame-collapse guard: fresh cues × new frames)", "Y2")):
            y = confirmation[key]
            pre = confirmation["precondition_Y1" if key == "Y1" else "precondition_Y2"]
            if "test" in y:
                t, d = y["test"], y["descriptive"]
                guard = t.get("frame_guard")
                guard_text = f"; frame guard (≥ {guard['floor']}): {'passed' if guard['passed'] else 'FAILED ' + str(guard['frames_below'])}" if guard else ""
                lines.append(f"- {label}: scored tokens {len(y['scored_tokens'])} ({'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'}); token means ĉ_ΔA vs c_ΔA — {cmp_line(t)}; entry R² layer 1 {f(t['entry_r2_layer1'], 3)}, layer 2 {f(t['entry_r2_layer2'], 3)}{guard_text} → {'pass' if t['passed'] else 'FAIL'} {t['failing'] or ''}")
                lines.append(f"  - Level 0-F entries: {pooled_line(d['pairs']['level0F'])}; self weights pooled {cmp_line(d['pairs']['self_level0F']['pooled'])}; pairs ĉ_ΔA vs c_ΔA {cmp_line(d['pairs']['c_hat_vs_c'])}; per layer token means {cmp_line(d['token_means_c_1'])} | {cmp_line(d['token_means_c_2'])}; c_ΔA spread sd {f(d['c_spread'], 4)}")
                lines.append(f"  - scale-only on this set: token means {cmp_line(y['scale_only']['token_means'])}; entries {pooled_line(y['scale_only']['entries'])}; margin at layer 2 {f(y['scale_only']['margin']['2'], 3)}")
                lines.append(f"  - Experiment 015 Level 0 on this set: token means {cmp_line(d['token_means_level0_015'])}; layer-2 entry R² {f(d['ladder']['rows']['level0_015']['2']['entry_r2'], 3)}")
                lines.append(f"  - diagonal-proportional (Level 0-F self weight): token means {cmp_line(d['token_means_diag_L0F'])}; entries {pooled_line(d['pairs']['diag_L0F'])}; margin {', '.join(f'layer {k} {f(v, 3)}' for k, v in d['pairs']['comparator_margin'].items())}")
                lines += ladder_lines(d["ladder"])
            else:
                lines.append(f"- {label}: not evaluable ({len(y['scored_tokens'])} scored tokens; {'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'})")
            if key == "Y2":
                lines += frame_lines(y.get("per_frame", {}))
                for frame_id, entry in confirmation["frames"].items():
                    if entry.get("template_defined") and not entry.get("valid"):
                        lines.append(f"  - frame {frame_id}: invalid at stage 1 (no fresh cue prompt run)")
            else:
                lines.append(f"  - per-frame layer-2 minimum over the exposed frames {f(frame_minimum(y.get('per_frame', {})), 3)}")
        y3 = confirmation["Y3"]
        if y3.get("evaluable"):
            lines.append(f"- Y3 (scale-only alternative over both sets, {y3['n_pairs']} pairs): layer-2 entry R² {f(y3['scale_only_entry_r2_layer2'], 3)} (rejected iff < {Y3_ENTRY_R2_MAX}) against Level 0-F {f(y3['level0F_entry_r2_layer2'], 3)}, margin {f(y3['margin'], 3)} (rejected iff ≥ {Y3_MARGIN}) → {'REJECTED' if y3['rejected'] else 'NOT REJECTED'}; token means {cmp_line(y3['token_means'])}")
        else:
            lines.append(f"- Y3: not evaluable ({y3.get('n_token_means')} token means)")
        comp = confirmation.get("comparator", {})
        if comp.get("interaction_beyond_self_logit") is not None:
            lines.append(f"- Comparator rule over both sets ({comp['n_pairs']} pairs): Level 0-F {pooled_line(comp['level0F'])}; diagonal-proportional {pooled_line(comp['diag_L0F'])}; margins {', '.join(f'layer {k} {f(v, 3)}' for k, v in comp['margin'].items())} (required ≥ {comp['margin_required']}) → "
                         + ("Level 0-F decodes the query/key interaction beyond the self-logit effect" if comp["interaction_beyond_self_logit"] else "most of the pattern change is captured by the self-logit change with proportional redistribution (narrower wording)"))
        if "ladder_both_sets" in confirmation:
            lines += ["- Ladder over both sets:"] + ladder_lines(confirmation["ladder_both_sets"])
        lines += ["", "| token | class | Y1 frames | Y1 ĉ_ΔA | Y1 c_ΔA | Y2 frames | Y2 ĉ_ΔA | Y2 c_ΔA | scale-only ĉ (Y1) | 015 L0 ĉ (Y1) | σ̂'/σ L2 (Y1) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        categories = {token["word"]: token["category"] for token in confirmation.get("tokens_meta", [])}
        for word in sorted(confirmation["Y1"]["tokens"], key=lambda w: -(confirmation["Y1"]["tokens"][w].get("c_hat_mean") or -9)):
            a, b = confirmation["Y1"]["tokens"][word], confirmation["Y2"]["tokens"].get(word, {})
            lines.append(f"| {word} | {categories.get(word, '')} | {a.get('n_valid_frames')} | {f(a.get('c_hat_mean'), 4)} | {f(a.get('c_mean'), 4)} | {b.get('n_valid_frames')} | {f(b.get('c_hat_mean'), 4)} | {f(b.get('c_mean'), 4)} | {f(a.get('c_scale_only_mean'), 4)} | {f(a.get('c_level0_015_mean'), 4)} | {f((a.get('sigma_ratio_mean') or {}).get('2'), 3)} |")
        for label, key in (("Y1", "Y1"), ("Y2 (the new frames)", "Y2")):
            lines += ["", f"Measured / predicted self-attention weight change ΔA_h(p_c, p_c), token means over the {label} frames:", "", "| token | " + " | ".join(HEAD_KEYS) + " |", "|---|" + "---|" * len(HEAD_KEYS)]
            for word in sorted(confirmation[key]["tokens"], key=lambda w: -(confirmation[key]["tokens"][w].get("c_hat_mean") or -9)):
                a = confirmation[key]["tokens"][word]
                if a.get("self_mean"):
                    lines.append(f"| {word} | " + " | ".join(f"{f(a['self_mean'][key_], 2)} / {f(a['self_hat_mean'][key_], 2)}" for key_ in HEAD_KEYS) + " |")
        lines.append("")
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
