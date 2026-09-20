"""Experiment 017: the transport head's own pattern change and its end-to-end number signal from the decoded upstream state.

Experiments 010–016 decoded how a cue word's weight-only encoding change ``ΔE`` becomes the transport head's *input*;
the head ``L03.H04`` itself was described with a frozen attention pattern since Experiments 009, 011 and 013. This
module predicts the head's own attention-row change at the transport position ``p_t`` and its output change ``ΔT``
along the locked axis ``d̂_T`` from the decoded chain alone: Experiment 016's Level 0-F at the cue position, block 2's MLP
at the frame's operating point (channel D), one propagation step to ``p_t`` in coordinated frames, and the head's own
query, key and value changes reduced as Experiment 016 reduced layers 1–2 (the template direction of the change at a
locked layer-3 template base, the frame's scale, the frame's own reference operands). Level 0 never receives a measured
fresh residual, LayerNorm statistic, attention row, MLP state or head state at either position: its only cue-specific
input is ``ΔE``. The three head objects are the row change ``ΔA_H``, the pattern term ``Π`` (the changed row carrying
the changed values) and the value term ``F`` (the reference row carrying the value change; the frozen-pattern account),
with ``ΔT = F + Π`` exactly. The variants — Level 0, the exact-head rung, the two ablations and Level 1 — are switches
on one code path; an independent exact chain carries the identities I4–I7. Constants are copied from design revision 2.
"""

from __future__ import annotations

import json
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
from . import frame_channels as fch
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

EXPERIMENT_DIR = "experiments/017-transport-head-pattern"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
INHERITED_016_EXTRACT_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-016-pair-extract.json"
EXPERIMENT_011_LOCK_PATH = er.LOCK_RELATIVE_PATH
EXPERIMENT_012_LOCK_PATH = lc.LOCK_RELATIVE_PATH
EXPERIMENT_013_LOCK_PATH = ap.LOCK_RELATIVE_PATH
EXPERIMENT_014_LOCK_PATH = nf.LOCK_RELATIVE_PATH
EXPERIMENT_015_LOCK_PATH = atp.LOCK_RELATIVE_PATH
EXPERIMENT_016_LOCK_PATH = fch.LOCK_RELATIVE_PATH
EXPERIMENT_013_CONFIRMATION_PATH = ap.CONFIRMATION_RELATIVE_PATH
EXPERIMENT_014_CONFIRMATION_PATH = nf.CONFIRMATION_RELATIVE_PATH
EXPERIMENT_015_CONFIRMATION_PATH = atp.CONFIRMATION_RELATIVE_PATH
EXPERIMENT_016_CONFIRMATION_PATH = fch.CONFIRMATION_RELATIVE_PATH
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260924
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
INHERITED_SCHEMA_VERSION = 1
PHASES = ("explore", "lock", "confirm", "report")

Y_ROW_R2 = 0.95  # the head's row change: pooled entry R² over the scored pairs of a set
Y_PI_SPEARMAN = 0.90  # the pattern term Π: token means
Y_PI_R2 = 0.90
Y_DT_R2 = 0.95  # ΔT: pooled pairs and token means
FRAME_GUARD_R2 = 0.90  # Y2: every valid fresh frame's pooled row-entry R² over its scored pairs
Y3_R2_MAX = 0.95  # the frozen pattern is rejected iff its ΔT R² over the pooled cue-final pairs of both sets is below this …
Y3_MARGIN = 0.05  # … and at least this far below Level 0's
X3_IDENTITY_TOLERANCE = 1e-4  # I4: the exact chain against the captured layer-3 residuals at p_c and p_t (relative)
ROW_IDENTITY_TOLERANCE = atp.ROW_IDENTITY_TOLERANCE  # I5: the recomputed head row against the captured one (absolute, per entry)
DT_IDENTITY_TOLERANCE = 1e-3  # I6: the recomputed ΔT against the measured head change (absolute, residual units along d̂_T)
SPLIT_IDENTITY_TOLERANCE = 1e-4  # I7: ΔT = F + Π from the captured quantities (absolute)
RECOVERY_TOLERANCE = 1e-9  # the Level 1 variant of the switch model against the independent exact chain (rows, absolute)
MIN_VALID_FRAMES = 8
MIN_VALID_FRAMES_PER_TOKEN = 3
MIN_SCORED_TOKENS = 16
FRAME_CUE_EFFECT_RATE = 108 / 120
LOCK_PREDICTION_TOLERANCE = 1e-9
REPLICATION_TOLERANCE = 1e-6
LOCKED_STATE_TOLERANCE = 1e-9
EXPECTED_EXTRACT_SIZE_016 = 7764  # Experiment 016's 6180 exposed pairs + its 1296 + 288 confirmed pairs
HEAD_LAYER, HEAD_INDEX, HEAD_KEY = ht.HEAD_LAYER, ht.HEAD_INDEX, ht.HEAD_KEY
UPSTREAM_LAYERS = atp.HEAD_LAYERS  # (1, 2)
PROGRAM_LAYERS = (1, 2, HEAD_LAYER)
RUNGS = ("exact_head", "no_D", "template_head")  # the descriptive rungs committed in the table beside Level 0; Level 1 is analysed, not tabled
ABLATIONS = ("no_D", "template_head")
ALL_RUNGS = ("level0", *RUNGS, "level1")
LADDER_KEYS = ("c_L_level0F", "c_L_level0D", "c_L_level1")  # the decoded-c_L ladder: Experiment 016's Level 0-F, with channel D, Level 1
PREDICTION_COLUMNS = ("token", "frame_id", "template", "p_c", "p_t", "row_level0", "self_level0", "F_hat", "Pi_hat", "dT_hat", "dT_frozen",
                      "row_exact_head", "Pi_exact_head", "dT_exact_head", "row_no_D", "Pi_no_D", "dT_no_D", "row_template_head", "Pi_template_head", "dT_template_head",
                      "sigma_ratio_3", "x3_change_norm", "c_L_level0D", "c_L_level0F")

CANDIDATES: dict[str, tuple[str, ...]] = {
    "determiner-like": ("further", "usual", "adjacent", "nearby", "leading", "primary", "secondary", "principal", "opposite", "distinct", "separate", "remaining"),
    "ordinal-or-numeral": ("twice", "once", "score", "trio", "duo", "solo", "triplet", "tens", "double", "triple", "couple", "pair"),
    "quantity": ("finite", "partial", "whole", "entire", "complete", "maximal", "greater", "massive", "immense", "modest", "generous", "overall"),
    "possessive-or-pronoun": ("none", "them", "us", "you", "ye", "him", "me", "it", "she", "he", "they", "we", "anything", "something", "everything", "nothing"),
    "adjective": ("violet", "metal", "paper", "leather", "blunt", "costly", "fancy", "sweet", "bitter", "spicy", "ripe", "raw", "rotten", "faded", "stolen", "gentle", "giant", "silver", "purple", "yellow"),
}
QUOTAS = {"determiner-like": 5, "ordinal-or-numeral": 4, "quantity": 5, "possessive-or-pronoun": 4, "adjective": 6}
FRESH_FRAMES: tuple[tuple[str, str], ...] = (
    ("cardinal", "The bakery sells {cue}"),
    ("cardinal", "The museum displays {cue}"),
    ("cardinal", "The fleet carries {cue}"),
    ("cardinal", "The orchard bears {cue}"),
    ("quantifier", "The report lists {cue}"),
    ("quantifier", "The lecture covers {cue}"),
    ("quantifier", "The survey counts {cue}"),
    ("quantifier", "The archive holds {cue}"),
    ("coordinated-adjective", "Clara and Dev packed {cue} hot"),
    ("coordinated-adjective", "Anna and Paul carried {cue} cool"),
    ("coordinated-adjective", "Eva and Tom hauled {cue} damp"),
    ("coordinated-adjective", "Alice and Ben served {cue} crisp"),
)
FRAME_ID_TAG = "017"
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", *fch.SCIENTIFIC_PATH_PREFIXES[1:])
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)
OUTCOME_Y1 = ("HEAD_PATTERN_PREDICTED_TOKENS", "HEAD_PATTERN_NOT_PREDICTED_TOKENS", "PRECONDITION_FAILED_TOKENS")
OUTCOME_Y2 = ("HEAD_PATTERN_PREDICTED_FRAMES_CONDITIONAL", "HEAD_PATTERN_NOT_PREDICTED_FRAMES_CONDITIONAL", "PRECONDITION_FAILED_FRAMES")
OUTCOME_Y3 = ("FROZEN_PATTERN_REJECTED", "FROZEN_PATTERN_NOT_REJECTED", "FROZEN_PATTERN_NOT_EVALUABLE")


class PhaseError(pm.PhaseError):
    """Protocol violation in the Experiment 017 phase machinery."""


# ---------------------------------------------------------------------------
# The frame state: the 013/016 state plus the residuals of layers 1–3 at every position ≤ p_t and the head's reference row.


@dataclass(frozen=True)
class FrameState017:
    """A frame's reference run for Experiment 017: everything the decoded chain and the head need, all from one forward pass."""

    state: ap.FrameState013  # positions ≤ p_c of layers 1–2 and their pattern rows at p_c (Experiments 013–016's machinery)
    x1_all: list[torch.Tensor]  # RESID_PRE.L1 at positions 0..p_t (float64)
    x2_all: list[torch.Tensor]
    x3_all: list[torch.Tensor]  # RESID_PRE.L3 at positions 0..p_t
    A3: torch.Tensor  # the head's reference row at query p_t over keys 0..p_t

    @property
    def frame(self) -> pm.Frame:
        return self.state.ref.frame

    @property
    def p_c(self) -> int:
        return self.frame.p_c

    @property
    def p_t(self) -> int:
        return self.frame.p_t

    @property
    def positions(self) -> tuple[int, ...]:
        """The changed positions P: the cue position and, in coordinated frames, the transport position."""
        return tuple(sorted({self.p_c, self.p_t}))


def reference_sites_017(frame: pm.Frame) -> list[pm.Site]:
    return [(f"RESID_PRE.L{layer}", k) for layer in UPSTREAM_LAYERS for k in range(frame.p_t + 1)] + [(f"ATTN_PATTERN.L{layer}", frame.p_c) for layer in UPSTREAM_LAYERS]


def patched_sites_017(frame: pm.Frame) -> list[pm.Site]:
    return ap.patched_sites(frame) + [(f"RESID_PRE.L{HEAD_LAYER}", k) for k in sorted({frame.p_c, frame.p_t})] + [(f"ATTN_PATTERN.L{HEAD_LAYER}", frame.p_t)]


def capture_frame_017(model: Any, head: ht.HeadWeights, reference: pm.Prompt, nouns: Sequence[pm.Noun], axis_T: pm.SiteAxis) -> FrameState017:
    frame = reference.frame
    ref, components = ra.capture_reference(model, head, reference, nouns, extra_sites=reference_sites_017(frame))
    x1_all = [components.pop(pm.site_label(("RESID_PRE.L1", k))).double() for k in range(frame.p_t + 1)]
    x2_all = [components.pop(pm.site_label(("RESID_PRE.L2", k))).double() for k in range(frame.p_t + 1)]
    A1 = components.pop(pm.site_label(("ATTN_PATTERN.L1", frame.p_c)))
    A2 = components.pop(pm.site_label(("ATTN_PATTERN.L2", frame.p_c)))
    if len(ref.residuals) != frame.p_t + 1 or ref.attention.shape[0] <= frame.p_t:
        raise pm.IncidentError(f"{frame.frame_id}: the reference capture does not cover every position up to p_t")
    if not torch.equal(x1_all[frame.p_c], ref.vectors["R0"].double()):
        raise pm.IncidentError(f"{frame.frame_id}: the captured residual before block 1 disagrees with the R0 stage vector of the same run")
    if A1.shape[0] != ap.N_HEADS or A1.shape[1] <= frame.p_c or A2.shape[0] != ap.N_HEADS:
        raise pm.IncidentError(f"{frame.frame_id}: unexpected attention pattern shape {tuple(A1.shape)} / {tuple(A2.shape)}")
    state = ap.FrameState013(ref, components, ra.read_functional(head, ref, axis_T), x1_all[frame.p_c], x2_all[frame.p_c], x1_all[: frame.p_c + 1], x2_all[: frame.p_c + 1], A1, A2)
    return FrameState017(state, x1_all, x2_all, [x.double() for x in ref.residuals], ref.attention.double()[: frame.p_t + 1])


def measure_pair(model: Any, weights: pm.Weights, head: ht.HeadWeights, state: FrameState017, name: str, token_id: int, e_axis: pm.SiteAxis, axis_T: pm.SiteAxis, nouns: Sequence[pm.Noun]) -> ra.Attribution:
    s = state.state
    return ra.measure_token_010(model, weights, head, s.ref, s.components, s.functional, name, token_id, e_axis, axis_T, nouns, extra_sites=patched_sites_017(s.ref.frame))


def locked_state(state: FrameState017) -> dict[str, Any]:
    """What the lock stores per exposed frame: the reference residuals before blocks 1, 2 and 3 at every position ≤ p_t."""
    return {"p_c": state.p_c, "p_t": state.p_t, "x1_all": [x.tolist() for x in state.x1_all], "x2_all": [x.tolist() for x in state.x2_all], "x3_all": [x.tolist() for x in state.x3_all]}


def _tensors(rows: Sequence[Sequence[float]]) -> list[torch.Tensor]:
    return [torch.tensor(list(row), dtype=torch.float64) for row in rows]


def head_row(program: atp.LayerProgram, x3_all: Sequence[torch.Tensor], p_t: int) -> torch.Tensor:
    """The head's reference row at query p_t from the reference residuals (the reference half of I5)."""
    return atp.ReferenceRow(program, [x.double() for x in x3_all[: p_t + 1]]).A_ref[HEAD_INDEX]


def check_reference_row(state: FrameState017, program: atp.LayerProgram) -> float:
    worst = float((head_row(program, state.x3_all, state.p_t) - state.A3).abs().max())
    if worst > ROW_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{state.frame.frame_id}: the program does not reproduce the head's captured reference row (max {worst:.2e})")
    return worst


# ---------------------------------------------------------------------------
# Rows with changed positions, the exact chain (Level 1) and the head's three objects.


def _row_and_values(program: atp.LayerProgram, rr: atp.ReferenceRow, query_normed: torch.Tensor | None, key_normed: Mapping[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """All heads' rows at query ``rr.p_c`` with the query recomputed from ``query_normed`` (None: the reference query) and the keys and values at ``key_normed``'s positions recomputed from their normalized residuals."""
    q_pos = rr.p_c
    q = rr.q_ref if query_normed is None else program.rotate(program.q_tilde(query_normed), q_pos)
    keys, values = rr.keys.clone(), rr.values.clone()
    for pos, n in key_normed.items():
        keys[:, pos] = program.rotate(program.k_tilde(n), pos)
        values[:, pos] = program.v(n)
    rows = torch.softmax(torch.einsum("he,hke->hk", q, keys) / program.scale, dim=-1)
    return rows, values


def _attention_output(program: atp.LayerProgram, rows: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("hk,hkd->d", rows, program.output(values))


def exact_chain(programs: Mapping[int, atp.LayerProgram], lw: lc.LayerWeights, x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], x3_all: Sequence[torch.Tensor], p_c: int, p_t: int, delta_e: torch.Tensor) -> dict[str, Any]:
    """Level 1: the weight-only program at the frame's own reference state on ΔE through layers 1–2 at the changed positions, then the head's rows and values at p_t (an identity against the captured patched run)."""
    positions = sorted({p_c, p_t})
    residuals = {1: [x.double() for x in x1_all], 2: [x.double() for x in x2_all]}
    dx: dict[int, dict[int, torch.Tensor]] = {1: {p_c: delta_e.double()}}
    for layer in UPSTREAM_LAYERS:
        program, xs = programs[layer], residuals[layer]
        normed = {pos: program.normalize(xs[pos] + d) for pos, d in dx[layer].items()}
        following: dict[int, torch.Tensor] = {}
        for pos in positions:
            rr = atp.ReferenceRow(program, xs[: pos + 1])
            rows, values = _row_and_values(program, rr, normed.get(pos), {k: n for k, n in normed.items() if k <= pos})
            d_in = dx[layer].get(pos)
            change = _attention_output(program, rows, values) - rr.output_ref
            if d_in is not None:
                change = change + d_in + lw.delta_out(layer, xs[pos], d_in)
            following[pos] = change
        dx[layer + 1] = following
    program = programs[HEAD_LAYER]
    rr3 = atp.ReferenceRow(program, [x.double() for x in x3_all[: p_t + 1]])
    normed3 = {pos: program.normalize(x3_all[pos].double() + d) for pos, d in dx[HEAD_LAYER].items()}
    rows, values = _row_and_values(program, rr3, normed3.get(p_t), normed3)
    return {"dx": dx, "dx3": dx[HEAD_LAYER], "rr3": rr3, "rows3": rows, "values3": values}


def head_terms(program: atp.LayerProgram, rr3: atp.ReferenceRow, rows_new: torch.Tensor, values_new: torch.Tensor, d_T: torch.Tensor, row_ref: torch.Tensor | None = None) -> dict[str, Any]:
    """The head's three objects from a row and values at p_t: the row change, F (the reference row carrying the value change), Π (the changed row carrying the changed values), ΔT = F + Π.

    ``row_ref`` is the head's reference row (the program's recomputation from the reference residuals unless the captured one is supplied)."""
    o_ref = program.output(rr3.values)[HEAD_INDEX]
    o_new = program.output(values_new)[HEAD_INDEX]
    A_ref = rr3.A_ref[HEAD_INDEX] if row_ref is None else row_ref.double()
    A_new = rows_new[HEAD_INDEX]
    F = float((A_ref @ (o_new - o_ref)) @ d_T)
    Pi = float(((A_new - A_ref) @ o_new) @ d_T)
    return {"row": A_new - A_ref, "F": F, "Pi": Pi, "dT": float(((A_new @ o_new) - (A_ref @ o_ref)) @ d_T)}


def measured_terms(program: atp.LayerProgram, rr3: atp.ReferenceRow, A3_ref: torch.Tensor, A3_patch: torch.Tensor, x3_patch: Mapping[int, torch.Tensor], d_T: torch.Tensor) -> dict[str, Any]:
    """F, Π and the row change from the captured reference and patched rows and the captured patched residuals (the values at the changed positions recomputed by the head's value program)."""
    values = rr3.values.clone()
    for pos, x in x3_patch.items():
        values[:, pos] = program.v(program.normalize(x.double()))
    rows = rr3.A_ref.clone()
    rows[HEAD_INDEX] = A3_patch.double()
    return head_terms(program, rr3, rows, values, d_T, row_ref=A3_ref)


# ---------------------------------------------------------------------------
# The switch model: Level 0, the exact-head rung, the two ablations, Level 1.


@dataclass(frozen=True)
class Variant:
    """Which switches are on: the layers-1–2 channels (Experiment 016's), channel D, the head's frame channels, the exact head program."""

    name: str
    upstream: fch.Channels = fch.LEVEL0F
    block2_own: bool = True  # channel D: block 2's MLP at the frame's operating point (else at the Experiment 012 template base)
    head_channels: bool = True  # A₃ and B₃: the frame's own operands and scale at layer 3 (else the template base's — Experiment 015's reduction at the head)
    exact_head: bool = False  # the head's exact program at the frame's own state on the predicted Δ̂x₃ (the exact-head rung; Level 1 with the exact upstream)


LEVEL0 = Variant("level0")
EXACT_HEAD = Variant("exact_head", exact_head=True)
NO_D = Variant("no_D", block2_own=False)
TEMPLATE_HEAD = Variant("template_head", head_channels=False)
LEVEL1 = Variant("level1", upstream=fch.LEVEL1, exact_head=True)
TABLE_VARIANTS = (LEVEL0, EXACT_HEAD, NO_D, TEMPLATE_HEAD)
VARIANTS = (*TABLE_VARIANTS, LEVEL1)


def bases_to_json(bases: Mapping[str, Mapping[str, torch.Tensor | None]], counts: Mapping[str, int]) -> dict[str, Any]:
    return {template: {"p_c": entry["p_c"].tolist(), "p_t": None if entry["p_t"] is None else entry["p_t"].tolist(), "n_frames": int(counts[template])} for template, entry in bases.items()}


def bases_from_json(entry: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor | None]]:
    return {template: {"p_c": torch.tensor(value["p_c"], dtype=torch.float64), "p_t": None if value["p_t"] is None else torch.tensor(value["p_t"], dtype=torch.float64)} for template, value in entry.items()}


def layer3_bases(frames: Sequence[pm.Frame], states: Mapping[str, FrameState017]) -> tuple[dict[str, dict[str, torch.Tensor | None]], dict[str, int]]:
    """The layer-3 template-mean bases: x₃ at p_c averaged over the exposed frames of the template, and x₃ at p_t for the coordinated template."""
    bases, counts = {}, {}
    for template in pm.TEMPLATE_ORDER:
        chosen = [states[frame.frame_id] for frame in frames if frame.template_id == template and frame.frame_id in states]
        if not chosen:
            continue
        p_c = torch.stack([s.x3_all[s.p_c] for s in chosen]).mean(0)
        p_t = torch.stack([s.x3_all[s.p_t] for s in chosen]).mean(0) if all(s.p_t != s.p_c for s in chosen) else None
        bases[template] = {"p_c": p_c, "p_t": p_t}
        counts[template] = len(chosen)
    return bases, counts


@dataclass(frozen=True)
class HeadChainModel:
    """The decoded chain to the head and the head itself; every prediction takes the token identity, the weights, the locked axes, read and bases and the frame's reference residuals at positions ≤ p_t."""

    fcm: fch.FrameChannelModel  # Experiment 016's model: read, block weights, layer-1–2 programs, Experiment 012 bases, d̂_E
    program3: atp.LayerProgram
    bases_3: Mapping[str, Mapping[str, torch.Tensor | None]]  # template -> {"p_c": x̄₃(p_c), "p_t": x̄₃(p_t) or None}
    d_T: torch.Tensor  # the locked unit axis of the head's output site

    @property
    def programs(self) -> dict[int, atp.LayerProgram]:
        return {**dict(self.fcm.programs), HEAD_LAYER: self.program3}

    def upstream(self, weights: pm.Weights, rows16: Mapping[int, atp.ReferenceRow], x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], p_c: int, p_t: int, token_id: int, template: str, variant: Variant) -> dict[str, Any]:
        """Δ̂x₃ at the changed positions: Experiment 016's channels at p_c, channel D, and the one-step propagation to p_t in coordinated frames."""
        x1, x2 = x1_all[p_c].double(), x2_all[p_c].double()
        up = self.fcm.predict_channels(weights, rows16, x1, x2, token_id, template, variant.upstream)
        delta_e, dx2_pc = up["delta_e"], up["dx2"]
        out2_pc = rows16[2].output_change(up["rows"][2], up["values"][2])
        base2 = x2 if variant.block2_own else self.fcm.bases_012[template][1].double()
        dx3 = {p_c: dx2_pc + out2_pc + self.fcm.lw.delta_out(2, base2, dx2_pc)}
        dx2 = {p_c: dx2_pc}
        if p_t != p_c:
            # Layer 1 at query p_t: x₁(p_t) is unchanged; the key and value at p_c are recomputed at x₁(p_c) + ΔE (exact at the frame's own state).
            program1, program2 = self.fcm.programs[1], self.fcm.programs[2]
            rr1 = atp.ReferenceRow(program1, [x.double() for x in x1_all[: p_t + 1]])
            rows1, values1 = _row_and_values(program1, rr1, None, {p_c: program1.normalize(x1 + delta_e)})
            dx2_pt = _attention_output(program1, rows1, values1) - rr1.output_ref
            # Layer 2 at query p_t: the key and value at p_c at x₂(p_c) + Δ̂x₂(p_c); the query, key and value at p_t at x₂(p_t) + Δ̂x₂(p_t); block 2's MLP at the frame's operating point x₂(p_t).
            rr2 = atp.ReferenceRow(program2, [x.double() for x in x2_all[: p_t + 1]])
            x2_pt = x2_all[p_t].double()
            normed2 = {p_c: program2.normalize(x2 + dx2_pc), p_t: program2.normalize(x2_pt + dx2_pt)}
            rows2, values2 = _row_and_values(program2, rr2, normed2[p_t], normed2)
            # Channel D at p_t is always the frame's operating point: there is no locked template base at p_t, so the −D ablation moves block 2's operating point at p_c only.
            dx3[p_t] = dx2_pt + (_attention_output(program2, rows2, values2) - rr2.output_ref) + self.fcm.lw.delta_out(2, x2_pt, dx2_pt)
            dx2[p_t] = dx2_pt
        return {"dx3": dx3, "dx2": dx2, "delta_e": delta_e, "denominator": up["denominator"], "c_L_016": up["c_L"],
                "c_L": self.fcm.read.inner(dx3[p_c] - delta_e) / up["denominator"]}

    def head(self, rr3: atp.ReferenceRow, x3_all: Sequence[torch.Tensor], dx3: Mapping[int, torch.Tensor], p_c: int, p_t: int, template: str, variant: Variant) -> dict[str, Any]:
        """The head's row and values at p_t from Δ̂x₃: the exact program at the frame's state, or the reduced changes at the layer-3 template base with the frame's (or the base's) scale and operands."""
        program, eps = self.program3, self.program3.eps
        x3 = {pos: x3_all[pos].double() for pos in dx3}
        if variant.exact_head:
            normed = {pos: program.normalize(x3[pos] + d) for pos, d in dx3.items()}
            rows, values = _row_and_values(program, rr3, normed.get(p_t), normed)
            sigma = {pos: (float(fch._sigma(x3[pos], eps)), float(fch._sigma(x3[pos] + d, eps))) for pos, d in dx3.items()}
        else:
            base_of = {p_c: self.bases_3[template]["p_c"]}
            if p_t != p_c:
                base_of[p_t] = self.bases_3[template]["p_t"]
            n_delta, sigma = {}, {}
            for pos, d in dx3.items():
                base = base_of[pos]
                if base is None:
                    raise pm.IncidentError(f"no layer-3 template base for position {pos} of template {template}")
                base = base.double()
                anchor = x3[pos] if variant.head_channels else base
                s_ref, s_after = fch._sigma(anchor, eps), fch._sigma(anchor + d, eps)
                shifted = base + d
                n_delta[pos] = program.ln_w * ((shifted - shifted.mean()) / s_after - (base - base.mean()) / s_ref)
                sigma[pos] = (float(s_ref), float(s_after))
            dq = program.q_tilde(n_delta[p_t]) - program.b_Q if p_t in n_delta else torch.zeros_like(program.b_Q)
            dk = {pos: program.k_tilde(n) - program.b_K for pos, n in n_delta.items()}
            dv = {pos: program.v(n) - program.b_V for pos, n in n_delta.items()}
            if variant.head_channels:
                # The frame's own operands: the general rotated form on the reference query and keys (for p_t = p_c the same-position rotation cancels and this is Experiment 016's self logit).
                q = program.rotate(program.q_tilde(rr3.normed_pc) + dq, p_t)
                keys = rr3.keys.clone()
                for pos in n_delta:
                    keys[:, pos] = keys[:, pos] + program.rotate(dk[pos], pos)
                scores = torch.einsum("he,hke->hk", q, keys) / program.scale
            else:
                # The base's operands (Experiment 015's convention): the changed logits' changes computed at the template base and added to the frame's reference logits; the other keys the frame's.
                q_base = program.q_tilde(program.normalize(base_of[p_t].double()))
                scores = rr3.scores_ref + torch.einsum("he,hke->hk", program.rotate(dq, p_t), rr3.keys) / program.scale
                scores = scores.clone()
                for pos in n_delta:
                    k_base = program.k_tilde(program.normalize(base_of[pos].double()))
                    before = (program.rotate(q_base, p_t) * program.rotate(k_base, pos)).sum(-1)
                    after = (program.rotate(q_base + dq, p_t) * program.rotate(k_base + dk[pos], pos)).sum(-1)
                    scores[:, pos] = rr3.scores_ref[:, pos] + (after - before) / program.scale
            rows = torch.softmax(scores, dim=-1)
            values = rr3.values.clone()
            for pos in n_delta:
                values[:, pos] = values[:, pos] + dv[pos]
        terms = head_terms(program, rr3, rows, values, self.d_T)
        return {**terms, "rows": rows, "values": values, "sigma": sigma}

    def variant_parts(self, weights: pm.Weights, rows16: Mapping[int, atp.ReferenceRow], rr3: atp.ReferenceRow, x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], x3_all: Sequence[torch.Tensor],
                      p_c: int, p_t: int, token_id: int, template: str, variant: Variant) -> dict[str, Any]:
        up = self.upstream(weights, rows16, x1_all, x2_all, p_c, p_t, token_id, template, variant)
        return {"upstream": up, "head": self.head(rr3, x3_all, up["dx3"], p_c, p_t, template, variant)}

    def parts(self, weights: pm.Weights, x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], x3_all: Sequence[torch.Tensor], p_c: int, p_t: int, token_id: int, template: str,
              variants: Sequence[Variant] = TABLE_VARIANTS) -> dict[str, Any]:
        rows16 = atp.reference_rows(self.fcm.programs, x1_all[: p_c + 1], x2_all[: p_c + 1])
        rr3 = atp.ReferenceRow(self.program3, [x.double() for x in x3_all[: p_t + 1]])
        results = {variant.name: self.variant_parts(weights, rows16, rr3, x1_all, x2_all, x3_all, p_c, p_t, token_id, template, variant) for variant in variants}
        full = results["level0"]
        head0, up0 = full["head"], full["upstream"]
        entry: dict[str, Any] = {"p_c": p_c, "p_t": p_t, "row_level0": head0["row"].tolist(), "self_level0": float(head0["row"][p_c]), "F_hat": head0["F"], "Pi_hat": head0["Pi"], "dT_hat": head0["dT"], "dT_frozen": head0["F"]}
        for name in RUNGS:
            if name in results:
                entry[f"row_{name}"] = results[name]["head"]["row"].tolist()
                entry[f"Pi_{name}"] = results[name]["head"]["Pi"]
                entry[f"dT_{name}"] = results[name]["head"]["dT"]
        entry["sigma_ratio_3"] = {str(pos): after / before for pos, (before, after) in head0["sigma"].items()}
        entry["x3_change_norm"] = {str(pos): float(d.norm()) for pos, d in up0["dx3"].items()}
        entry["c_L_level0D"] = up0["c_L"]
        entry["c_L_level0F"] = up0["c_L_016"]
        return {"rows16": rows16, "rr3": rr3, "variants": results, "entry": entry}

    def predict(self, weights: pm.Weights, x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor], x3_all: Sequence[torch.Tensor], p_c: int, p_t: int, token_id: int, template: str) -> dict[str, Any]:
        return self.parts(weights, x1_all, x2_all, x3_all, p_c, p_t, token_id, template)["entry"]

    def predict_from_state(self, weights: pm.Weights, state: FrameState017, token_id: int, template: str) -> dict[str, Any]:
        return self.predict(weights, state.x1_all, state.x2_all, state.x3_all, state.p_c, state.p_t, token_id, template)

    def predict_from_locked(self, weights: pm.Weights, locked: Mapping[str, Any], token_id: int, template: str) -> dict[str, Any]:
        return self.predict(weights, _tensors(locked["x1_all"]), _tensors(locked["x2_all"]), _tensors(locked["x3_all"]), int(locked["p_c"]), int(locked["p_t"]), token_id, template)


def model_from_locks(lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lw: lc.LayerWeights, programs: Mapping[int, atp.LayerProgram], reference_ids: Mapping[str, int], plural_ids: Mapping[str, int],
                     bases_3: Mapping[str, Mapping[str, torch.Tensor | None]]) -> HeadChainModel:
    fcm = fch.model_from_locks(lock_011, lock_012, lw, {layer: programs[layer] for layer in UPSTREAM_LAYERS}, reference_ids, plural_ids)
    return HeadChainModel(fcm, programs[HEAD_LAYER], bases_3, torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64))  # the locked axis as the measurement projects on it (no renormalization)


def prediction_row(word: str, frame_id: str, template: str, prediction: Mapping[str, Any]) -> dict[str, Any]:
    row = {"token": word, "frame_id": frame_id, "template": template}
    row.update({key: prediction[key] for key in PREDICTION_COLUMNS[3:]})
    return row


# ---------------------------------------------------------------------------
# Measurement per pair.


@dataclass(frozen=True)
class AnalysisContext:
    model: HeadChainModel
    fch_context: fch.AnalysisContext
    weights: pm.Weights
    axis_T: pm.SiteAxis


def analyse_pair_017(record: ra.Attribution, plural: ra.Attribution, *, context: AnalysisContext, state: FrameState017) -> dict[str, Any] | None:
    """Experiment 016's analysis (inherited quantities, I1–I3, recoveries), the identities I4–I7 of the exact chain, the head's measured objects, Level 0 with its rungs and the remainders."""
    model, weights = context.model, context.weights
    rows16 = atp.reference_rows(model.fcm.programs, state.state.x1_all, state.state.x2_all)
    base = fch.analyse_pair_016(record, plural, context=context.fch_context, state=state.state, rows=rows16)
    if base is None:
        return None
    frame, p_c, p_t, template = state.frame, state.p_c, state.p_t, state.frame.template_id
    positions = state.positions
    where = f"{frame.frame_id}/{record.token}"
    x3_patch = {pos: record.extra[pm.site_label((f"RESID_PRE.L{HEAD_LAYER}", pos))].double() for pos in positions}
    A3_patch = record.extra[pm.site_label((f"ATTN_PATTERN.L{HEAD_LAYER}", p_t))][HEAD_INDEX, : p_t + 1].double()
    delta_e = model.fcm.read.encoding_delta(weights, record.token_id, template)
    exact = exact_chain(model.programs, model.fcm.lw, state.x1_all, state.x2_all, state.x3_all, p_c, p_t, delta_e)
    rr3 = exact["rr3"]
    identities: dict[str, float] = {}
    identities["I4_x3"] = max(float((state.x3_all[pos] + exact["dx3"][pos] - x3_patch[pos]).norm() / max(float(x3_patch[pos].norm()), 1e-12)) for pos in positions)
    if identities["I4_x3"] > X3_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{where}: the exact chain does not reproduce the captured layer-3 residuals (relative error {identities['I4_x3']:.2e})")
    identities["I5_head_row"] = float((exact["rows3"][HEAD_INDEX] - A3_patch).abs().max())
    if identities["I5_head_row"] > ROW_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{where}: the exact chain does not reproduce the head's captured patched row (max {identities['I5_head_row']:.2e})")
    level1 = head_terms(model.program3, rr3, exact["rows3"], exact["values3"], model.d_T)
    identities["I6_dT"] = abs(level1["dT"] - record.head_change)
    if identities["I6_dT"] > DT_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{where}: the exact chain does not reproduce the measured head change ({level1['dT']:.5f} against {record.head_change:.5f})")
    measured = measured_terms(model.program3, rr3, state.A3, A3_patch, x3_patch, model.d_T)
    identities["I7_split"] = abs(measured["F"] + measured["Pi"] - record.head_change)
    if identities["I7_split"] > SPLIT_IDENTITY_TOLERANCE:
        raise pm.IncidentError(f"{where}: ΔT ≠ F + Π from the captured quantities (difference {identities['I7_split']:.2e})")
    parts = model.parts(weights, state.x1_all, state.x2_all, state.x3_all, p_c, p_t, record.token_id, template, variants=VARIANTS)
    one = parts["variants"]["level1"]
    identities["head_level1_recovery"] = max(float((one["head"]["rows"] - exact["rows3"]).abs().max()), max(float((one["upstream"]["dx3"][pos] - exact["dx3"][pos]).abs().max()) for pos in positions))
    if identities["head_level1_recovery"] > RECOVERY_TOLERANCE:
        raise pm.IncidentError(f"{where}: the switch model's Level 1 does not recover the exact chain ({identities['head_level1_recovery']:.2e})")
    prediction = parts["entry"]
    measured_row = measured["row"]
    rungs = {name: {"row": parts["variants"][name]["head"]["row"].tolist(), "F": parts["variants"][name]["head"]["F"], "Pi": parts["variants"][name]["head"]["Pi"], "dT": parts["variants"][name]["head"]["dT"]} for name in (*RUNGS, "level1")}
    statistics = {"level0": atp.row_statistics(torch.tensor(prediction["row_level0"], dtype=torch.float64), measured_row)}
    statistics.update({name: atp.row_statistics(torch.tensor(rungs[name]["row"], dtype=torch.float64), measured_row) for name in rungs})
    dx3_level0 = parts["variants"]["level0"]["upstream"]["dx3"]
    eps = model.program3.eps
    return {"token": record.token, "token_id": record.token_id, "frame_id": frame.frame_id, "template": template, "p_c": p_c, "p_t": p_t, "cue_final": p_t == p_c,
            "row": measured_row.tolist(), "self": float(measured_row[p_c]), "F": measured["F"], "Pi": measured["Pi"], "dT": record.head_change,
            "c_dA": base["c"], "c_hat_016": base["prediction"]["c_hat"], "delta_A_pc": dict(base["self"]), "c_L": base["c_L"], "c_M": base["c_M"], "c_H": base["c_H"],
            "ladder": {"c_L_level0F": base["ladder"]["c_L_level0F"], "c_L_level0D": prediction["c_L_level0D"], "c_L_level1": base["ladder"]["c_L_level1"], "level1_error": base["ladder"]["level1_error"]},
            "prediction": prediction, "rungs": rungs, "statistics": statistics,
            "x3_remainder": {str(pos): float((dx3_level0[pos] - exact["dx3"][pos]).norm() / max(float(exact["dx3"][pos].norm()), 1e-12)) for pos in positions},
            "scale_remainder": {str(pos): float(fch._sigma(x3_patch[pos], eps)) - parts["variants"]["level0"]["head"]["sigma"][pos][1] for pos in positions},
            "sigma_3": {str(pos): {"ref": before, "after_predicted": after, "after_measured": float(fch._sigma(x3_patch[pos], eps))} for pos, (before, after) in parts["variants"]["level0"]["head"]["sigma"].items()},
            "identities": {**base["identities"], **identities}}


def compact_analysis(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """What the results state stores per pair: everything except the sixteen-head detail inherited from Experiment 016 (its digest-checked extract keeps that)."""
    return {key: value for key, value in analysis.items() if key != "delta_A_pc"}


# ---------------------------------------------------------------------------
# Pool (207 tokens × 66 frames), the inherited extract, the confirmation set.


def build_pool_017(manifest: ScreeningManifest, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009, confirmation_011: er.Confirmation011,
                   confirmation_012: lc.Confirmation012, confirmation_013: ap.Confirmation013, confirmation_014: nf.Confirmation014, confirmation_015: atp.Confirmation015, confirmation_016: fch.Confirmation016) -> cs.Pool008:
    base = fch.build_pool_016(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015)
    frames = base.frames + tuple(confirmation_016.frames)
    origin = dict(base.frame_origin) | {frame.frame_id: "confirmation-016" for frame in confirmation_016.frames}
    tokens = list(base.tokens)
    category, source = dict(base.token_category), dict(base.token_source)
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation_016.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"Experiment 016 token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-016"
    return cs.Pool008(frames, origin, tuple(tokens), category, source, base.nouns, base.noun_source, base.reference_ids, base.plural_cue)


EXTRACT_016_DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014", "confirmation_015", "confirmation_016", "lock_016")
EXTRACT_016_DIGEST_FIELDS = tuple(f"{key}_sha256" for key in EXTRACT_016_DIGEST_KEYS)
EXTRACT_FIELDS = ("c_dA", "c_hat_016", "c_L", "c_M", "c_H", "c_L_level0F")


def extract_entry(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """From an Experiment 017 analysis (or, with the same keys, an Experiment 016 analysis mapped by ``extract_entry_016``)."""
    return {"c_dA": float(analysis["c_dA"]), "c_hat_016": float(analysis["c_hat_016"]), "delta_A_pc": {key: float(analysis["delta_A_pc"][key]) for key in fch.HEAD_KEYS},
            "c_L": float(analysis["c_L"]), "c_M": float(analysis["c_M"]), "c_H": float(analysis["c_H"]), "c_L_level0F": float(analysis["ladder"]["c_L_level0F"])}


def extract_entry_016(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """The same quantities read from an Experiment 016 recorded pair."""
    return extract_entry({"c_dA": analysis["c"], "c_hat_016": analysis["prediction"]["c_hat"], "delta_A_pc": analysis["self"], "c_L": analysis["c_L"], "c_M": analysis["c_M"], "c_H": analysis["c_H"], "ladder": analysis["ladder"]})


def inherited_extract_payload(entries: Mapping[str, Mapping[str, Any]], *, source: Mapping[str, Any], digests: Mapping[str, str], stage1_state_digests: Mapping[str, str]) -> dict[str, Any]:
    payload = {"schema_version": INHERITED_SCHEMA_VERSION, "experiment": "016", "kind": "pair-extract",
               "description": "Derived extract of Experiment 016's recorded per-pair pattern-change read c_ΔA (measured) and ĉ_ΔA (Level 0-F), the sixteen self-attention weight changes, c_L, c_M, c_H and the decoded c_L at Level 0-F over its 6180 exposed pairs and its 1296 + 288 confirmed pairs, with the digests of its twelve fresh frames' digested stage-1 reference states. Experiment 017 recomputes the quantities and requires agreement within 1e-6, and re-captures the states to their digests.",
               "source": dict(source), **{field: digests[key] for field, key in zip(EXTRACT_016_DIGEST_FIELDS, EXTRACT_016_DIGEST_KEYS)},
               "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "stage1_state_digests": {key: str(value) for key, value in sorted(stage1_state_digests.items())},
               "entries": {key: dict(value) for key, value in sorted(entries.items())}}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_extract(path: Path, *, digests: Mapping[str, str], expected_size: int) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited 016 extract: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "experiment", "kind", "description", "source", *EXTRACT_016_DIGEST_FIELDS, "model", "stage1_state_digests", "entries", "content_sha256"}, "inherited 016 extract")
    if payload["schema_version"] != INHERITED_SCHEMA_VERSION or payload["experiment"] != "016" or payload["kind"] != "pair-extract" or payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("inherited 016 extract schema or digest is not frozen")
    if tuple(payload[field] for field in EXTRACT_016_DIGEST_FIELDS) != tuple(digests[key] for key in EXTRACT_016_DIGEST_KEYS):
        raise ValueError("inherited 016 extract was recorded against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision} or len(payload["entries"]) != expected_size or len(payload["stage1_state_digests"]) != len(fch.FRESH_FRAMES):
        raise ValueError("inherited 016 extract model, size or stage-1 digests are not frozen")
    return payload


def check_extract_replication(measured: Mapping[str, Mapping[str, Any]], recorded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = set(recorded) - set(measured)
    if missing:
        raise pm.IncidentError(f"recomputed quantities lack {len(missing)} recorded pairs (e.g. {sorted(missing)[:3]})")
    worst_key, worst = None, 0.0
    for key, value in recorded.items():
        other = measured[key]
        for name in EXTRACT_FIELDS:
            deviation = abs(value[name] - other[name])
            if deviation > worst:
                worst_key, worst = f"{key}:{name}", deviation
        for head in fch.HEAD_KEYS:
            deviation = abs(value["delta_A_pc"][head] - other["delta_A_pc"][head])
            if deviation > worst:
                worst_key, worst = f"{key}:{head}", deviation
    if worst > REPLICATION_TOLERANCE:
        raise pm.IncidentError(f"{worst_key}: recomputed quantity deviates from Experiment 016's record by {worst:.3e}")
    return {"passed": True, "n": len(recorded), "max_abs_deviation": worst, "worst_key": worst_key}


def fresh_tokens_017(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
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


CONFIRMATION_DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014", "confirmation_015", "confirmation_016")
CONFIRMATION_DIGEST_FIELDS = tuple(f"{key}_sha256" for key in CONFIRMATION_DIGEST_KEYS)


def build_confirmation_payload(tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> dict[str, Any]:
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_017(tokenizer, excluded_ids)
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
               "policy": {"licensing": "every fresh cue in every fresh frame and in every exposed frame", "quotas": dict(QUOTAS), "expectations": "none: the committed head-row, Π and ΔT predictions are the only predictions; no label is preregistered for any head, position, frame or cue"},
               "tokens": tokens, "frames": frames, "token_prompts": [nf._prompt_entry(tokenizer, frame, token) for frame in frame_objects for token in tokens],
               "exposed_frame_prompts": [nf._prompt_entry(tokenizer, frame, token) for frame in pool.frames for token in tokens],
               "construction": "tokenizer-only; first eligible candidates per lexical class; classes carry no expectation; no model output"}
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


Confirmation017 = ap.Confirmation013


def validate_confirmation(payload: Mapping[str, Any], pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation017:
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
    return Confirmation017(dict(payload["reference_cue_ids"]), frames, tuple(payload["exposed_frame_ids"]), tuple(tokens), tuple(fresh_prompts), tuple(exposed_prompts), payload["content_sha256"])


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


def load_confirmation(path: Path, pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation017:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read confirmation set: {path}") from error
    return validate_confirmation(payload, pool, digests)


# ---------------------------------------------------------------------------
# Results state, phases.

DIGEST_KEYS = ("manifest", "extension", "confirmation_006", "confirmation_009", "confirmation_011", "confirmation_012", "confirmation_013", "confirmation_014", "confirmation_015", "confirmation_016", "confirmation_017",
               "lock_011", "lock_012", "lock_013", "lock_014", "lock_015", "lock_016", "extract_016")
STATE_DIGEST_FIELDS = ("manifest_sha256", "extension_sha256", "confirmation_sha256", "confirmation_009_sha256", "confirmation_011_sha256", "confirmation_012_sha256", "confirmation_013_sha256", "confirmation_014_sha256",
                       "confirmation_015_sha256", "confirmation_016_sha256", "confirmation_017_sha256", "lock_011_sha256", "lock_012_sha256", "lock_013_sha256", "lock_014_sha256", "lock_015_sha256", "lock_016_sha256", "extract_016_sha256")
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


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation017) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts} | {confirmation.reference_prompt(frame).key for frame in confirmation.frames}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


comparison = lc.comparison


# ---------------------------------------------------------------------------
# Statistics.

SCALAR_PREDICTIONS = ("dT_hat", "Pi_hat", "F_hat", "dT_frozen", *(f"dT_{name}" for name in RUNGS), *(f"Pi_{name}" for name in RUNGS))


def _mean(values: Sequence[float]) -> float:
    return pm._mean(list(values))


def _token_means(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not analyses:
        return {}
    out = {"n_frames": len(analyses), "frames": sorted(a["frame_id"] for a in analyses), "n_cue_final": sum(1 for a in analyses if a["cue_final"]),
           "dT_mean": _mean([a["dT"] for a in analyses]), "Pi_mean": _mean([a["Pi"] for a in analyses]), "F_mean": _mean([a["F"] for a in analyses]), "self_mean": _mean([a["self"] for a in analyses]),
           "self_hat_mean": _mean([p["self_level0"] for p in predictions]), "c_L_mean": _mean([a["c_L"] for a in analyses]), "ladder_mean": {key: _mean([a["ladder"][key] for a in analyses]) for key in LADDER_KEYS},
           "dT_level1_mean": _mean([a["rungs"]["level1"]["dT"] for a in analyses]), "Pi_level1_mean": _mean([a["rungs"]["level1"]["Pi"] for a in analyses])}
    out.update({f"{key}_mean": _mean([p[key] for p in predictions]) for key in SCALAR_PREDICTIONS})
    return out


def _row_stats(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], name: str) -> list[dict[str, float]]:
    if name == "level0":
        return [atp.row_statistics(torch.tensor(p["row_level0"], dtype=torch.float64), torch.tensor(a["row"], dtype=torch.float64)) for a, p in zip(analyses, predictions)]
    if name == "level1":
        return [a["statistics"]["level1"] for a in analyses]
    return [atp.row_statistics(torch.tensor(p[f"row_{name}"], dtype=torch.float64), torch.tensor(a["row"], dtype=torch.float64)) for a, p in zip(analyses, predictions)]


def _scalar(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], name: str, measured: str) -> dict[str, Any]:
    if name.endswith("_level1"):
        return comparison([a["rungs"]["level1"][name[: -len("_level1")]] for a in analyses], [a[measured] for a in analyses])
    return comparison([p[name] for p in predictions], [a[measured] for a in analyses])


def _pair_statistics(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled row entries per rung; ΔT, Π and F on the pairs; the frozen pattern's gap; the pattern term's share of the ΔT variance."""
    out: dict[str, Any] = {"n_pairs": len(analyses), "rows": {name: atp.pooled_summary(_row_stats(analyses, predictions, name)) for name in ALL_RUNGS}}
    out["dT_hat_vs_dT"] = _scalar(analyses, predictions, "dT_hat", "dT")
    out["Pi_hat_vs_Pi"] = _scalar(analyses, predictions, "Pi_hat", "Pi")
    out["F_hat_vs_F"] = _scalar(analyses, predictions, "F_hat", "F")
    out["dT_frozen_vs_dT"] = _scalar(analyses, predictions, "dT_frozen", "dT")
    out["self_hat_vs_self"] = comparison([p["self_level0"] for p in predictions], [a["self"] for a in analyses])
    for name in (*RUNGS, "level1"):
        out[f"dT_{name}_vs_dT"] = _scalar(analyses, predictions, f"dT_{name}", "dT")
        out[f"Pi_{name}_vs_Pi"] = _scalar(analyses, predictions, f"Pi_{name}", "Pi")
    level0, frozen = out["dT_hat_vs_dT"]["r2"], out["dT_frozen_vs_dT"]["r2"]
    out["frozen_gap"] = (level0 - frozen) if level0 is not None and frozen is not None else None
    dT_var, Pi_var = ap.spread([a["dT"] for a in analyses]), ap.spread([a["Pi"] for a in analyses])
    out["Pi_variance_share"] = (Pi_var * Pi_var) / (dT_var * dT_var) if dT_var else None
    out["dT_spread"], out["Pi_spread"] = dT_var, Pi_var
    return out


def _ladder_statistics(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], means: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The rungs against Level 0 (rows, Π, ΔT), the ablation costs with their predeclared orderings, the remainders and the decoded-c_L ladder."""
    pairs = _pair_statistics(analyses, predictions)
    out: dict[str, Any] = {"rows": {name: pairs["rows"][name]["entry_r2"] for name in ALL_RUNGS}, "dT": {"level0": pairs["dT_hat_vs_dT"]["r2"], **{name: pairs[f"dT_{name}_vs_dT"]["r2"] for name in (*RUNGS, "level1")}, "frozen": pairs["dT_frozen_vs_dT"]["r2"]},
                           "Pi": {"level0": pairs["Pi_hat_vs_Pi"]["r2"], **{name: pairs[f"Pi_{name}_vs_Pi"]["r2"] for name in (*RUNGS, "level1")}}}
    level0_rows, level0_Pi = out["rows"]["level0"], out["Pi"]["level0"]
    out["ablation_cost_rows"] = {name: (level0_rows - out["rows"][name]) if level0_rows is not None and out["rows"][name] is not None else None for name in ABLATIONS}
    out["ablation_cost_Pi"] = {name: (level0_Pi - out["Pi"][name]) if level0_Pi is not None and out["Pi"][name] is not None else None for name in ABLATIONS}
    ordered = lambda costs: all(costs[n] is not None for n in ABLATIONS) and costs["template_head"] >= costs["no_D"]  # noqa: E731
    out["predeclared_ordering_rows_holds"] = ordered(out["ablation_cost_rows"])
    out["predeclared_ordering_Pi_holds"] = ordered(out["ablation_cost_Pi"])
    out["frozen_gap"] = pairs["frozen_gap"]
    out["Pi_variance_share"] = pairs["Pi_variance_share"]
    out["x3_remainder"] = {"mean": _mean([v for a in analyses for v in a["x3_remainder"].values()]), "max": max(v for a in analyses for v in a["x3_remainder"].values())}
    out["scale_remainder_abs_mean"] = _mean([abs(v) for a in analyses for v in a["scale_remainder"].values()])
    out["sigma_ratio_3_mean"] = _mean([v for p in predictions for v in p["sigma_ratio_3"].values()])
    out["decoded_c_L"] = {"pairs": {key: comparison([a["ladder"][key] for a in analyses], [a["c_L"] for a in analyses]) for key in LADDER_KEYS}, "level1_error_max": max((a["ladder"]["level1_error"] for a in analyses), default=None)}
    if means:
        names = sorted(means)
        out["decoded_c_L"]["token_means"] = {key: comparison([means[n]["ladder_mean"][key] for n in names], [means[n]["c_L_mean"] for n in names]) for key in LADDER_KEYS}
    return out


def _split(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], cue_final: bool | None = None, template: str | None = None) -> dict[str, Any]:
    """The frozen pattern against Level 0 on the cue-final or the coordinated pairs (the predeclared split), or on one template (descriptive)."""
    chosen = [(a, p) for a, p in zip(analyses, predictions) if (cue_final is None or a["cue_final"] == cue_final) and (template is None or a["template"] == template)]
    if not chosen:
        return {"n_pairs": 0}
    stats = _pair_statistics([a for a, _ in chosen], [p for _, p in chosen])
    return {"n_pairs": len(chosen), "n_tokens": len({a["token"] for a, _ in chosen}), "level0_dT_r2": stats["dT_hat_vs_dT"]["r2"], "frozen_dT_r2": stats["dT_frozen_vs_dT"]["r2"], "gap": stats["frozen_gap"],
            "Pi_variance_share": stats["Pi_variance_share"], "row_entry_r2": stats["rows"]["level0"]["entry_r2"], "Pi_r2": stats["Pi_hat_vs_Pi"]["r2"]}


def statistics_for(pairs: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, Any]:
    by_token = {name: [(a, p) for a, p in zip(pairs, predictions) if a["token"] == name] for name in tokens}
    means = {name: _token_means([a for a, _ in rows], [p for _, p in rows]) for name, rows in by_token.items() if rows}
    names = sorted(means)
    m = lambda key: [means[n][key] for n in names]  # noqa: E731
    token_means = {"dT_hat_vs_dT": comparison(m("dT_hat_mean"), m("dT_mean")), "Pi_hat_vs_Pi": comparison(m("Pi_hat_mean"), m("Pi_mean")), "F_hat_vs_F": comparison(m("F_hat_mean"), m("F_mean")),
                   "dT_frozen_vs_dT": comparison(m("dT_frozen_mean"), m("dT_mean")), **{f"dT_{name}_vs_dT": comparison(m(f"dT_{name}_mean"), m("dT_mean")) for name in RUNGS},
                   **{f"Pi_{name}_vs_Pi": comparison(m(f"Pi_{name}_mean"), m("Pi_mean")) for name in RUNGS}, "dT_spread": ap.spread(m("dT_mean")), "Pi_spread": ap.spread(m("Pi_mean"))}
    return {"n_tokens": len(names), "n_pairs": len(pairs), "token_means": token_means, "pairs": _pair_statistics(pairs, predictions), "ladder": _ladder_statistics(pairs, predictions, means),
            "per_frame": _per_frame(pairs, predictions), "split": _splits(pairs, predictions), "token_means_table": means}


def _splits(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The predeclared cue-final / coordinated split and, descriptively, every template on its own."""
    return {"cue_final": _split(analyses, predictions, cue_final=True), "coordinated": _split(analyses, predictions, cue_final=False),
            "per_template": {template: _split(analyses, predictions, template=template) for template in pm.TEMPLATE_ORDER}}


def _per_frame(analyses: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for frame_id in sorted({a["frame_id"] for a in analyses}):
        mine = [(a, p) for a, p in zip(analyses, predictions) if a["frame_id"] == frame_id]
        a0 = mine[0][0]
        out[frame_id] = {"n_pairs": len(mine), "template": a0["template"], "cue_final": a0["cue_final"], "rows_level0": atp.pooled_summary(_row_stats([a for a, _ in mine], [p for _, p in mine], "level0")),
                         "rows_template_head": atp.pooled_summary(_row_stats([a for a, _ in mine], [p for _, p in mine], "template_head")),
                         "dT_hat_vs_dT": comparison([p["dT_hat"] for _, p in mine], [a["dT"] for a, _ in mine]), "Pi_hat_vs_Pi": comparison([p["Pi_hat"] for _, p in mine], [a["Pi"] for a, _ in mine]),
                         "dT_frozen_vs_dT": comparison([p["dT_frozen"] for _, p in mine], [a["dT"] for a, _ in mine]), "Pi_spread": ap.spread([a["Pi"] for a, _ in mine])}
    return out


def frame_minimum(per_frame: Mapping[str, Mapping[str, Any]]) -> float | None:
    values = [entry["rows_level0"]["entry_r2"] for entry in per_frame.values() if entry["rows_level0"]["entry_r2"] is not None]
    return min(values) if values else None


# ---------------------------------------------------------------------------
# Exploration (Tier A).


def _components(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], bases_3: Mapping[str, Mapping[str, torch.Tensor | None]] | None) -> dict[str, Any]:
    """Weights, programs, axes and the frozen-pattern model; with ``bases_3`` also the head-chain model and its analysis context (``_with_model`` adds them later otherwise)."""
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model)
    heads = ap.HeadSet.from_model(model)
    programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in PROGRAM_LAYERS}
    cache = pm.PromptCache(model, tuple(pool.nouns))
    axis_T, e_axis = lc.locked_axes(lock_011, cs.stage_axes(cache, weights, pool_010))
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    fpm = ap.model_from_locks(lock_011, lock_012, lw, heads, pool.reference_ids, plural_ids)
    lc.check_read_weight(fpm.read, head, axis_T)
    out = {"weights": weights, "head": head, "lw": lw, "heads": heads, "programs": programs, "cache": cache, "axis_T": axis_T, "e_axis": e_axis, "fpm": fpm, "plural_ids": plural_ids}
    return _with_model(out, lock_011, lock_012, pool, bases_3) if bases_3 is not None else out


def _with_model(parts: Mapping[str, Any], lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], pool: cs.Pool008, bases_3: Mapping[str, Mapping[str, torch.Tensor | None]]) -> dict[str, Any]:
    out = dict(parts)
    out["model"] = model_from_locks(lock_011, lock_012, out["lw"], out["programs"], pool.reference_ids, out["plural_ids"], bases_3)
    out["context"] = AnalysisContext(out["model"], fch.AnalysisContext(out["model"].fcm, out["fpm"], out["heads"], out["weights"], out["axis_T"]), out["weights"], out["axis_T"])
    return out


def _check_previous_state(state_f: FrameState017, lock_016: Mapping[str, Any], inherited_extract: Mapping[str, Any]) -> None:
    """Every exposed frame's reference state is verified against Experiment 016's lock or its digested stage-1 states."""
    frame_id = state_f.frame.frame_id
    previous = lock_016.get("locked_states", {}).get(frame_id)
    if previous is not None:
        fch._check_locked_state(state_f.state, previous, frame_id)
    elif frame_id in inherited_extract.get("stage1_state_digests", {}):
        if ap.state_digest(atp.locked_state(state_f.state)) != inherited_extract["stage1_state_digests"][frame_id]:
            raise pm.IncidentError(f"{frame_id}: the reference state differs from Experiment 016's digested stage-1 state")
    else:
        raise pm.IncidentError(f"{frame_id}: no recorded Experiment 016 reference state to check against")


def _check_locked_state_017(state_f: FrameState017, previous: Mapping[str, Any], label: str) -> None:
    if (previous["p_c"], previous["p_t"]) != (state_f.p_c, state_f.p_t) or any(len(previous[key]) != len(getattr(state_f, key)) for key in ("x1_all", "x2_all", "x3_all")):
        raise pm.IncidentError(f"{label}: the positions or the number of positions differ from the locked state")
    drift = max(float((x - torch.tensor(y, dtype=torch.float64)).abs().max()) for key in ("x1_all", "x2_all", "x3_all") for x, y in zip(getattr(state_f, key), previous[key]))
    if drift > LOCKED_STATE_TOLERANCE:
        raise pm.IncidentError(f"{label}: the reference state differs from the locked one (max {drift:.2e})")


def run_exploration(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, *, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_016: Mapping[str, Any], inherited_extract: Mapping[str, Any],
                    state: dict[str, Any], results_path: Path | None, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    nouns = pool.single_nouns
    exploration = state["exploration"]
    say("stage axes from the Experiment 010 pool, checked against the Experiment 011 lock")
    parts = _components(model, pool, pool_010, lock_011, lock_012, None)
    weights, head, axis_T, e_axis, programs = parts["weights"], parts["head"], parts["axis_T"], parts["e_axis"], parts["programs"]
    exploration["read_weight_check"] = lc.check_read_weight(parts["fpm"].read, head, axis_T)
    exploration["axes_vectors"] = {"T": axis_T.direction.tolist(), "R0": torch.tensor(lock_011["axes_vectors"]["R0"], dtype=torch.float64).tolist()}
    exploration["sigma_T"] = axis_T.sigma
    exploration["read_weight"] = parts["fpm"].read.weight.tolist()
    exploration["defined_templates"] = list(lock_012["defined_templates"])
    exploration["program"] = atp._program_record(programs)
    say(f"reference states of the {len(pool.frames)} exposed frames (layers 1–3 at every position ≤ p_t), checked against Experiment 016")
    states: dict[str, FrameState017] = {}
    identities: dict[str, float] = {}
    for frame in pool.frames:
        state_f = capture_frame_017(model, head, pool.reference_prompt(frame), nouns, axis_T)
        rows16 = atp.reference_rows({layer: programs[layer] for layer in UPSTREAM_LAYERS}, state_f.state.x1_all, state_f.state.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows16, state_f.state))
        identities["I5_reference_head_row"] = max(identities.get("I5_reference_head_row", 0.0), check_reference_row(state_f, programs[HEAD_LAYER]))
        _check_previous_state(state_f, lock_016, inherited_extract)
        states[frame.frame_id] = state_f
    bases_3, counts = layer3_bases(pool.frames, states)
    exploration["bases_3"] = bases_to_json(bases_3, counts)
    exploration["locked_states"] = {frame_id: locked_state(s) for frame_id, s in states.items()}
    exploration["locked_state_digests"] = {frame_id: ap.state_digest(entry) for frame_id, entry in exploration["locked_states"].items()}
    parts = _with_model(parts, lock_011, lock_012, pool, bases_3)
    context: AnalysisContext = parts["context"]
    recorded = inherited_extract["entries"]
    say(f"layer-3 template bases over {dict(counts)} frames; re-measurement of the {len(recorded)} recorded pairs with the layer-3 residuals and the head's row captured")
    pairs: dict[str, dict[str, Any]] = {}
    measured_extract: dict[str, dict[str, Any]] = {}
    for frame in pool.frames:
        state_f = states[frame.frame_id]
        names = [name for name, _ in pool.tokens if f"{name}|{frame.frame_id}" in recorded]
        if not names:
            continue
        plural_name = pool.plural_cue[frame.template_id]
        records = {name: measure_pair(model, weights, head, state_f, name, pool.token_id(name), e_axis, axis_T, nouns) for name in names}
        plural = records[plural_name] if plural_name in records else measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for name, record in records.items():
            analysis = analyse_pair_017(record, plural, context=context, state=state_f)
            key = f"{name}|{frame.frame_id}"
            if analysis is None:
                raise pm.IncidentError(f"{key}: no analysis although Experiment 016 recorded the pair")
            er._max_errors(identities, analysis["identities"])
            pairs[key] = analysis
            measured_extract[key] = extract_entry(analysis)
        say(f"  {frame.frame_id}: {len(names)} pairs")
    exploration["identities"] = identities
    exploration["replication"] = {"experiment_016": check_extract_replication(measured_extract, recorded)}
    say(f"replication against Experiment 016: max deviation {exploration['replication']['experiment_016']['max_abs_deviation']:.2e}")
    analyses = list(pairs.values())
    stats = statistics_for(analyses, [a["prediction"] for a in analyses], [name for name, _ in pool.tokens])
    exploration["pairs"] = {key: compact_analysis(value) for key, value in pairs.items()}
    exploration["exposed_check"] = stats
    tm, pr = stats["token_means"], stats["pairs"]
    exploration["summary"] = {"defined_templates": exploration["defined_templates"], "row_entry_r2": pr["rows"]["level0"]["entry_r2"], "Pi_spearman": tm["Pi_hat_vs_Pi"]["spearman"], "Pi_r2": tm["Pi_hat_vs_Pi"]["r2"],
                              "dT_pairs_r2": pr["dT_hat_vs_dT"]["r2"], "dT_token_means_r2": tm["dT_hat_vs_dT"]["r2"], "frame_minimum_rows": frame_minimum(stats["per_frame"]),
                              "frozen_cue_final": stats["split"]["cue_final"], "frozen_coordinated": stats["split"]["coordinated"], "ablation_cost_rows": stats["ladder"]["ablation_cost_rows"],
                              "prediction_undefined": len(exploration["defined_templates"]) < 2}
    f = cd._f
    s = exploration["summary"]
    say(f"exposed: row entry R² {f(s['row_entry_r2'], 3)} (per-frame min {f(s['frame_minimum_rows'], 3)}); Π token means Spearman {f(s['Pi_spearman'], 3)}, R² {f(s['Pi_r2'], 3)}; ΔT pairs R² {f(s['dT_pairs_r2'], 3)}, token means R² {f(s['dT_token_means_r2'], 3)}; "
        f"frozen pattern on cue-final pairs {f(s['frozen_cue_final'].get('frozen_dT_r2'), 3)} against Level 0 {f(s['frozen_cue_final'].get('level0_dT_r2'), 3)}")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


# ---------------------------------------------------------------------------
# Lock and its artifacts.


def prediction_table(model: HeadChainModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], frames: Sequence[pm.Frame], tokens: Sequence[Mapping[str, Any]], defined: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        if frame.template_id not in defined:
            continue
        for token in tokens:
            rows.append(prediction_row(token["word"], frame.frame_id, frame.template_id, model.predict_from_locked(weights, locked_states[frame.frame_id], token["token_id"], frame.template_id)))
    return rows


def token_means_from_table(rows: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for word in tokens:
        mine = [row for row in rows if row["token"] == word]
        if mine:
            out[word] = {key: _mean([row[key] for row in mine]) for key in (*SCALAR_PREDICTIONS, "self_level0", "c_L_level0D", "c_L_level0F")}
            out[word]["n_frames"] = len(mine)
            out[word]["n_cue_final"] = sum(1 for row in mine if row["p_t"] == row["p_c"])
            out[word]["sigma_ratio_3_mean"] = _mean([v for row in mine for v in row["sigma_ratio_3"].values()])
    return out


def lock_predictions(model: HeadChainModel, weights: pm.Weights, locked_states: Mapping[str, Mapping[str, Any]], pool: cs.Pool008, confirmation: Confirmation017, defined: Sequence[str]) -> dict[str, Any]:
    rows = prediction_table(model, weights, locked_states, pool.frames, confirmation.tokens, defined)
    return {"rows": rows, "token_means": token_means_from_table(rows, [token["word"] for token in confirmation.tokens])}


def render_predictions(lock: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 017 — preregistered predictions (the transport head's own pattern change and its end-to-end ΔT from the decoded upstream state)", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_017_sha256']}`; Experiment 016 lock sha256 `{lock['lock_016_sha256']}`",
             f"- Floors: Y1/Y2 head-row entry R² ≥ {Y_ROW_R2}; Π token means Spearman ≥ {Y_PI_SPEARMAN} and R² ≥ {Y_PI_R2}; ΔT pairs R² ≥ {Y_DT_R2} and token means R² ≥ {Y_DT_R2}; Y2 frame guard: every valid fresh frame's row entry R² ≥ {FRAME_GUARD_R2}; "
             f"Y3 (cue-final pairs of both sets): the frozen pattern rejected iff its ΔT R² < {Y3_R2_MAX} and ≥ {Y3_MARGIN} below Level 0's; the coordinated split descriptive",
             f"- Y1 table: {len(lock['predictions']['rows'])} rows (fresh tokens × exposed frames), each with the predicted row change of {HEAD_KEY} at p_t, F̂, Π̂, ΔT̂, the frozen-pattern ΔT̂ and the exact-head, −D and template-head rungs; Y2 rows are computed at confirm stage 1 and digested before any fresh cue prompt",
             f"- Program: {', '.join(f'layer {k}: d_head {v['d_head']}, rotary_dim {v['rotary_dim']}, base {v['rotary_base']:g}' for k, v in lock['program'].items())}; ΔT, F̂, Π̂ in residual units along d̂_T (σ_T {f(lock.get('sigma_T'), 3)})", "",
             "| token | class | frames | **predicted ΔT̂** | F̂ (frozen) | **Π̂** | self ΔÂ(p_t,p_c) | exact head ΔT̂ | −D ΔT̂ | template head ΔT̂ | σ̂'/σ (L3) | decoded c_L (D) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    categories = {token["word"]: token["category"] for token in lock["tokens"]}
    means = lock["predictions"]["token_means"]
    for word, m in sorted(means.items(), key=lambda kv: -kv[1]["dT_hat"]):
        lines.append(f"| {word} | {categories.get(word, '')} | {m['n_frames']} | **{f(m['dT_hat'], 4)}** | {f(m['F_hat'], 4)} | **{f(m['Pi_hat'], 4)}** | {f(m['self_level0'], 3)} | {f(m['dT_exact_head'], 4)} | {f(m['dT_no_D'], 4)} | {f(m['dT_template_head'], 4)} | {f(m['sigma_ratio_3_mean'], 3)} | {f(m['c_L_level0D'], 3)} |")
    lines.append("")
    return "\n".join(lines)


def frozen_floors() -> dict[str, Any]:
    return {"y_row_r2": Y_ROW_R2, "y_pi_spearman": Y_PI_SPEARMAN, "y_pi_r2": Y_PI_R2, "y_dt_r2": Y_DT_R2, "frame_guard_r2": FRAME_GUARD_R2, "y3_r2_max": Y3_R2_MAX, "y3_margin": Y3_MARGIN,
            "x3_identity_tolerance": X3_IDENTITY_TOLERANCE, "row_identity_tolerance": ROW_IDENTITY_TOLERANCE, "dt_identity_tolerance": DT_IDENTITY_TOLERANCE, "split_identity_tolerance": SPLIT_IDENTITY_TOLERANCE,
            "recovery_tolerance": RECOVERY_TOLERANCE, "min_valid_frames": MIN_VALID_FRAMES, "min_valid_frames_per_token": MIN_VALID_FRAMES_PER_TOKEN, "min_scored_tokens": MIN_SCORED_TOKENS,
            "frame_cue_effect_rate": FRAME_CUE_EFFECT_RATE, "head_stage_floor": cs.STAGE_UNINFORMATIVE_FLOOR}


def build_candidate_lock(*, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation017, predictions: Mapping[str, Any], protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    if exploration["summary"]["prediction_undefined"]:
        raise PhaseError("fewer than two templates are defined: PREDICTION_UNDEFINED, no lock")
    lock = {"schema_version": 1, "experiment": "017", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit}
    for field, key in zip(STATE_DIGEST_FIELDS, DIGEST_KEYS):
        lock[field] = digests[key] if key != "confirmation_017" else confirmation.content_sha256
    lock.update({"confirmation_prompt_keys": sorted(prompt.key for prompt in confirmation.all_prompts), "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
                 "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED}, "head": HEAD_KEY, "program": exploration["program"],
                 "axes_vectors": exploration["axes_vectors"], "sigma_T": exploration["sigma_T"], "read_weight": exploration["read_weight"], "defined_templates": exploration["defined_templates"],
                 "bases_3": exploration["bases_3"], "locked_states": exploration["locked_states"], "locked_state_digests": exploration["locked_state_digests"], "floors": frozen_floors(),
                 "tokens": [dict(token) for token in confirmation.tokens], "exposed_check": exploration["exposed_check"],
                 "predictions": dict(predictions), "tier_a_results_sha256": state.get("state_sha256")})
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation017, predictions_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "017" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked or git_state.get("dirty", True):
        raise PhaseError("confirm requires the committed lock and predictions on a clean Git tree")
    expected = dict(zip(STATE_DIGEST_FIELDS, digest_tuple(digests)))
    expected["confirmation_017_sha256"] = confirmation.content_sha256
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
        if (a["token"], a["frame_id"], a["template"], a["p_c"], a["p_t"]) != (b["token"], b["frame_id"], b["template"], b["p_c"], b["p_t"]):
            raise PhaseError("the recomputed prediction table is ordered differently; nothing was executed")
        for key in PREDICTION_COLUMNS[5:]:
            worst = max(worst, atp._max_numeric_difference(a[key], b[key], f"{a['token']}/{a['frame_id']}/{key}"))
    if worst > LOCK_PREDICTION_TOLERANCE:
        raise PhaseError(f"the weights, locked axes, bases and locked reference states do not reproduce the preregistered predictions (max difference {worst:.3e}); nothing was executed")
    return worst


# ---------------------------------------------------------------------------
# Confirmation: stage 1, stage 2, scoring.


def stage_one(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation017, lock: Mapping[str, Any], lock_012: Mapping[str, Any], *, protocol_code_commit: str, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    parts = _components(model, pool, pool_010, lock, lock_012, bases_from_json(lock["bases_3"]))
    weights, head, cache, axis_T, e_axis, hcm = parts["weights"], parts["head"], parts["cache"], parts["axis_T"], parts["e_axis"], parts["model"]
    if atp._program_record(hcm.programs) != dict(lock["program"]):
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
        state_f = capture_frame_017(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        rows16 = atp.reference_rows(hcm.fcm.programs, state_f.state.x1_all, state_f.state.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows16, state_f.state))
        identities["I5_reference_head_row"] = max(identities.get("I5_reference_head_row", 0.0), check_reference_row(state_f, hcm.program3))
        sg, pl = pm.frame_prompts(frame)
        c_a, c_b = cache.c(sg), cache.c(pl)
        positive = sum(1 for noun in nouns if c_a[noun.lexical_key] - c_b[noun.lexical_key] > 0)
        required = pm.exact_count_floor(FRAME_CUE_EFFECT_RATE, len(nouns))
        plural = measure_pair(model, weights, head, state_f, pool.plural_cue[template], pl.cue_token_id, e_axis, axis_T, nouns)
        er._max_errors(identities, er.enforce_identities(plural, plural, f"{frame.frame_id}/{pool.plural_cue[template]}"))
        head_informative = abs(plural.head_change) >= cs.STAGE_UNINFORMATIVE_FLOOR * axis_T.sigma
        valid = head_informative and positive >= required
        frames_out[frame.frame_id] = {"template_id": template, "valid": valid, "head_informative": head_informative, "plural_head_change": plural.head_change, "cue_effect_positive": positive, "cue_effect_required": required,
                                      "template_defined": True, "reconstruction_error": state_f.state.ref.reconstruction_error, "p_c": frame.p_c, "p_t": frame.p_t}
        states[frame.frame_id] = locked_state(state_f)
        for token in confirmation.tokens:
            rows_out.append(prediction_row(token["word"], frame.frame_id, template, hcm.predict_from_state(weights, state_f, token["token_id"], template)))
        say(f"  {frame.frame_id}: {'valid' if valid else 'INVALID'} (head informative {head_informative}, cue effect {positive}/{required}); p_c {frame.p_c}, p_t {frame.p_t}; {len(confirmation.tokens)} predictions")
    return {"frames": frames_out, "states": states, "state_digests": {frame_id: ap.state_digest(entry) for frame_id, entry in states.items()}, "rows": rows_out, "identities": identities,
            "token_means": token_means_from_table(rows_out, [token["word"] for token in confirmation.tokens]),
            "digest": ap.table_digest(rows_out, states), "commit": protocol_code_commit, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "lock_sha256": lock["content_sha256"], "completed_at": pm.utc_now()}


def stage_two(model: Any, pool: cs.Pool008, pool_010: cs.Pool008, confirmation: Confirmation017, lock: Mapping[str, Any], lock_012: Mapping[str, Any], stage1: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    ap.assert_stage_one_digest(stage1)
    parts = _components(model, pool, pool_010, lock, lock_012, bases_from_json(lock["bases_3"]))
    weights, head, axis_T, e_axis, hcm, context = parts["weights"], parts["head"], parts["axis_T"], parts["e_axis"], parts["model"], parts["context"]
    nouns = pool.single_nouns
    identities: dict[str, float] = {}
    say("stage 2: fresh cues in the exposed frames (Y1) and in the valid fresh frames (Y2)")
    pairs_exposed: dict[str, dict[str, Any]] = {}
    for frame in pool.frames:
        if frame.template_id not in lock["defined_templates"]:
            continue
        state_f = capture_frame_017(model, head, pool.reference_prompt(frame), nouns, axis_T)
        _check_locked_state_017(state_f, lock["locked_states"][frame.frame_id], frame.frame_id)
        rows16 = atp.reference_rows(hcm.fcm.programs, state_f.state.x1_all, state_f.state.x2_all)
        identities["I1_reference_rows"] = max(identities.get("I1_reference_rows", 0.0), atp.check_reference_rows(rows16, state_f.state))
        identities["I5_reference_head_row"] = max(identities.get("I5_reference_head_row", 0.0), check_reference_row(state_f, hcm.program3))
        plural_name = pool.plural_cue[frame.template_id]
        plural = measure_pair(model, weights, head, state_f, plural_name, pool.token_id(plural_name), e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_017(record, plural, context=context, state=state_f)
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
        state_f = capture_frame_017(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        if ap.state_digest(locked_state(state_f)) != stage1["state_digests"][frame.frame_id]:
            raise pm.IncidentError(f"{frame.frame_id}: the re-captured reference state differs from stage 1's digested state")
        sg, pl = pm.frame_prompts(frame)
        plural = measure_pair(model, weights, head, state_f, pool.plural_cue[frame.template_id], pl.cue_token_id, e_axis, axis_T, nouns)
        for token in confirmation.tokens:
            record = measure_pair(model, weights, head, state_f, token["word"], token["token_id"], e_axis, axis_T, nouns)
            analysis = analyse_pair_017(record, plural, context=context, state=state_f)
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
    """Token means and pooled entries over the scored pairs; the five floors, and for Y2 the frame-collapse guard over the valid frames' scored pairs."""
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
        pair_stats = _pair_statistics(analyses, predictions)
        pi_means = comparison([tokens_out[w]["Pi_hat_mean"] for w in scored], [tokens_out[w]["Pi_mean"] for w in scored])
        dt_means = comparison([tokens_out[w]["dT_hat_mean"] for w in scored], [tokens_out[w]["dT_mean"] for w in scored])
        row_r2, dt_pairs = pair_stats["rows"]["level0"]["entry_r2"], pair_stats["dT_hat_vs_dT"]["r2"]
        checks = (("row_entry_r2", row_r2 is not None and row_r2 >= Y_ROW_R2), ("Pi_spearman", pi_means["spearman"] is not None and pi_means["spearman"] >= Y_PI_SPEARMAN), ("Pi_r2", pi_means["r2"] is not None and pi_means["r2"] >= Y_PI_R2),
                  ("dT_pairs_r2", dt_pairs is not None and dt_pairs >= Y_DT_R2), ("dT_token_means_r2", dt_means["r2"] is not None and dt_means["r2"] >= Y_DT_R2))
        failing = [name for name, ok in checks if not ok]
        test: dict[str, Any] = {"row_entry_r2": row_r2, "Pi_token_means": pi_means, "dT_pairs": pair_stats["dT_hat_vs_dT"], "dT_token_means": dt_means,
                                "floors": {"row_entry_r2": Y_ROW_R2, "Pi_spearman": Y_PI_SPEARMAN, "Pi_r2": Y_PI_R2, "dT_r2": Y_DT_R2}}
        if guard:
            scored_per_frame = _per_frame(analyses, predictions)
            below = {frame_id: entry["rows_level0"]["entry_r2"] for frame_id, entry in scored_per_frame.items() if entry["rows_level0"]["entry_r2"] is None or entry["rows_level0"]["entry_r2"] < FRAME_GUARD_R2}
            test["frame_guard"] = {"floor": FRAME_GUARD_R2, "per_frame_rows": {frame_id: entry["rows_level0"]["entry_r2"] for frame_id, entry in scored_per_frame.items()}, "frames_below": below, "passed": not below}
            if below:
                failing.append("frame_guard")
        test.update({"passed": not failing, "failing": failing})
        result["test"] = test
        result["frozen"] = {"token_means": comparison([tokens_out[w]["dT_frozen_mean"] for w in scored], [tokens_out[w]["dT_mean"] for w in scored]), "pairs": pair_stats["dT_frozen_vs_dT"], "gap": pair_stats["frozen_gap"]}
        result["descriptive"] = {"pairs": pair_stats, "ladder": _ladder_statistics(analyses, predictions, {w: tokens_out[w] for w in scored}),
                                 "token_means_F": comparison([tokens_out[w]["F_hat_mean"] for w in scored], [tokens_out[w]["F_mean"] for w in scored]),
                                 "token_means_rungs": {name: {"dT": comparison([tokens_out[w][f"dT_{name}_mean"] for w in scored], [tokens_out[w]["dT_mean"] for w in scored]), "Pi": comparison([tokens_out[w][f"Pi_{name}_mean"] for w in scored], [tokens_out[w]["Pi_mean"] for w in scored])} for name in RUNGS},
                                 "split": _splits(analyses, predictions),
                                 "dT_spread": ap.spread([tokens_out[w]["dT_mean"] for w in scored]), "Pi_spread": ap.spread([tokens_out[w]["Pi_mean"] for w in scored])}
    return result


def score_confirmation(stage1: Mapping[str, Any], pairs_exposed: Mapping[str, Mapping[str, Any]], pairs_fresh: Mapping[str, Mapping[str, Any]], confirmation: Confirmation017, lock: Mapping[str, Any]) -> dict[str, Any]:
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
    for y, pairs, table in ((y1, pairs_exposed, locked_table), (y2, pairs_fresh, stage1_table)):
        for w in y["scored_tokens"]:
            pair_items.extend((pairs[k], table[k]) for k in sorted(key for key in pairs if key.split("|")[0] == w and key in table))
    analyses, predictions = [a for a, _ in pair_items], [p for _, p in pair_items]
    cue_final = _split(analyses, predictions, cue_final=True)
    coordinated = _split(analyses, predictions, cue_final=False)
    evaluable = cue_final.get("n_tokens", 0) >= 2 and cue_final.get("frozen_dT_r2") is not None and cue_final.get("level0_dT_r2") is not None
    rejected = bool(evaluable and cue_final["frozen_dT_r2"] < Y3_R2_MAX and cue_final["gap"] >= Y3_MARGIN)
    results["Y3"] = {"evaluable": evaluable, "rejected": rejected, "cue_final": cue_final, "floors": {"r2_max": Y3_R2_MAX, "margin": Y3_MARGIN},
                     "coordinated_split": {**coordinated, "expected_not_rejected": True, "would_be_rejected": bool(coordinated.get("frozen_dT_r2") is not None and coordinated.get("gap") is not None and coordinated["frozen_dT_r2"] < Y3_R2_MAX and coordinated["gap"] >= Y3_MARGIN)},
                     "per_template": {template: _split(analyses, predictions, template=template) for template in pm.TEMPLATE_ORDER}, "n_pairs": len(pair_items)}
    if len(pair_items) >= 2:
        results["ladder_both_sets"] = _ladder_statistics(analyses, predictions)
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
    lines = ["# Experiment 017 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state['confirmation_017_sha256']}`", f"- Experiment 016 lock sha256: `{state['lock_016_sha256']}`",
             f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    incidents = list(exploration.get("incidents", [])) + ([state["confirmation"]["incident"]] if isinstance(state.get("confirmation"), dict) and "incident" in state["confirmation"] else [])
    if incidents:
        lines += ["## Incidents", ""] + [f"- `{e['phase']}` at commit `{e.get('commit', '?')}` ({e['at']}): {e['message']}" for e in incidents] + [""]

    def cmp_line(c: Mapping[str, Any]) -> str:
        return f"Spearman {f(c.get('spearman'), 3)}, R² {f(c.get('r2'), 3)}, MAE {f(c.get('mae'), 4)}, bias {f(c.get('bias'), 4)} (n {c.get('n')})"

    def split_line(s: Mapping[str, Any]) -> str:
        if not s.get("n_pairs"):
            return "no pairs"
        return f"{s['n_pairs']} pairs, {s.get('n_tokens')} tokens: Level 0 ΔT R² {f(s['level0_dT_r2'], 3)}, frozen pattern {f(s['frozen_dT_r2'], 3)}, gap {f(s['gap'], 3)}; Π variance share {f(s['Pi_variance_share'], 3)}; row entry R² {f(s['row_entry_r2'], 3)}; Π R² {f(s['Pi_r2'], 3)}"

    def ladder_lines(l: Mapping[str, Any]) -> list[str]:
        return [f"  - ladder (rows / ΔT / Π R²): " + "; ".join(f"`{name}`{' (identity)' if name == 'level1' else ''} {f(l['rows'][name], 3)} / {f(l['dT'][name], 3)} / {f(l['Pi'][name], 3)}" for name in ALL_RUNGS) + f"; `frozen` — / {f(l['dT']['frozen'], 3)} / ≡ 0 (`no_D` moves block 2's operating point at p_c only)",
                f"  - ablation costs rows: " + ", ".join(f"`{n}` {f(l['ablation_cost_rows'][n], 3)}" for n in ABLATIONS) + f" (template head ≥ −D holds: {l['predeclared_ordering_rows_holds']}); Π: " + ", ".join(f"`{n}` {f(l['ablation_cost_Pi'][n], 3)}" for n in ABLATIONS)
                + f" (holds: {l['predeclared_ordering_Pi_holds']}); frozen gap {f(l['frozen_gap'], 3)}; Π variance share {f(l['Pi_variance_share'], 3)}",
                f"  - x₃ remainder (relative) mean {f(l['x3_remainder']['mean'], 4)} max {f(l['x3_remainder']['max'], 4)}; scale remainder |σ(x₃') − σ̂'| mean {f(l['scale_remainder_abs_mean'], 4)}; σ̂'/σ mean {f(l['sigma_ratio_3_mean'], 3)}",
                "  - decoded c_L: " + "; ".join(f"`{key}` pairs R² {f(l['decoded_c_L']['pairs'][key]['r2'], 3)}" + (f" / token means R² {f(l['decoded_c_L']['token_means'][key]['r2'], 3)}" if "token_means" in l["decoded_c_L"] else "") for key in LADDER_KEYS) + f"; Level 1's residual max {f(l['decoded_c_L']['level1_error_max'], 6)}"]

    def frame_lines(per_frame: Mapping[str, Any]) -> list[str]:
        return [f"  - frame {frame_id} ({entry['template']}): {entry['n_pairs']} pairs; row entry R² {f(entry['rows_level0']['entry_r2'], 3)} (TV {f(entry['rows_level0']['tv_ratio'], 3)}); ΔT̂ vs ΔT {cmp_line(entry['dT_hat_vs_dT'])}; "
                f"Π̂ vs Π {cmp_line(entry['Pi_hat_vs_Pi'])} (Π sd {f(entry['Pi_spread'], 3)}); frozen ΔT R² {f(entry['dT_frozen_vs_dT']['r2'], 3)}; template-head rows {f(entry['rows_template_head']['entry_r2'], 3)}" for frame_id, entry in per_frame.items()]

    if "summary" in exploration:
        x = exploration["exposed_check"]
        rep = exploration["replication"]["experiment_016"]
        lines += ["## Tier A — exposed pool (calibration record only; the floors are frozen constants)", "",
                  f"- Replication of Experiment 016: {rep['n']} pairs, max deviation {rep['max_abs_deviation']:.2e}",
                  f"- Identities (checks only): {', '.join(f'{k} {v:.1e}' for k, v in exploration['identities'].items())}; read weight vs Experiment 011 lock {exploration['read_weight_check']:.1e}",
                  f"- Program: {', '.join(f'layer {k}: d_head {v['d_head']}, rotary_dim {v['rotary_dim']}, base {v['rotary_base']:g}' for k, v in exploration['program'].items())}; layer-3 bases over {', '.join(f'{t}: {v['n_frames']}' for t, v in exploration['bases_3'].items())} frames",
                  f"- Head row ({x['n_pairs']} pairs): Level 0 entry R² {f(x['pairs']['rows']['level0']['entry_r2'], 3)} (TV ratio {f(x['pairs']['rows']['level0']['tv_ratio'], 3)}); per-frame minimum {f(exploration['summary']['frame_minimum_rows'], 3)}; self weight {cmp_line(x['pairs']['self_hat_vs_self'])}",
                  f"- Π: token means ({x['n_tokens']}) {cmp_line(x['token_means']['Pi_hat_vs_Pi'])}; pairs {cmp_line(x['pairs']['Pi_hat_vs_Pi'])}; Π spread sd {f(x['token_means']['Pi_spread'], 4)}",
                  f"- ΔT: token means {cmp_line(x['token_means']['dT_hat_vs_dT'])}; pairs {cmp_line(x['pairs']['dT_hat_vs_dT'])}; F̂ vs F pairs {cmp_line(x['pairs']['F_hat_vs_F'])}; ΔT spread sd {f(x['token_means']['dT_spread'], 4)}",
                  f"- Frozen pattern: pairs {cmp_line(x['pairs']['dT_frozen_vs_dT'])}; token means {cmp_line(x['token_means']['dT_frozen_vs_dT'])}",
                  f"- Split — cue-final: {split_line(x['split']['cue_final'])}", f"- Split — coordinated: {split_line(x['split']['coordinated'])}"]
        lines += [f"- Template {template}: {split_line(entry)}" for template, entry in x["split"]["per_template"].items()]
        lines += ladder_lines(x["ladder"]) + [""]
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}`; predictions sha256 `{state['lock']['predictions_sha256']}`", ""]
    confirmation = state.get("confirmation")
    if confirmation and "stage1" in confirmation:
        s1 = confirmation["stage1"]
        lines += ["## Confirmation — stage 1 (fresh frames' reference states; prediction table digested before any fresh cue prompt)", "", f"- Table rows {len(s1['rows'])}; digest `{s1['digest']}`; commit `{s1['commit']}`"]
        for frame_id, entry in s1["frames"].items():
            if entry.get("template_defined"):
                lines.append(f"  - {frame_id}: {'valid' if entry['valid'] else 'INVALID'} (plural head change {f(entry['plural_head_change'], 3)}, cue effect {entry['cue_effect_positive']}/{entry['cue_effect_required']}; p_c {entry['p_c']}, p_t {entry['p_t']})")
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
                lines.append(f"- {label}: scored tokens {len(y['scored_tokens'])} ({'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'}); head row entry R² {f(t['row_entry_r2'], 3)}; Π token means {cmp_line(t['Pi_token_means'])}; "
                             f"ΔT pairs {cmp_line(t['dT_pairs'])}; ΔT token means {cmp_line(t['dT_token_means'])}{guard_text} → {'pass' if t['passed'] else 'FAIL'} {t['failing'] or ''}")
                lines.append(f"  - frozen pattern on this set: pairs {cmp_line(y['frozen']['pairs'])}; token means {cmp_line(y['frozen']['token_means'])}; gap to Level 0 {f(y['frozen']['gap'], 3)}; F̂ vs F token means {cmp_line(d['token_means_F'])}")
                lines.append(f"  - split — cue-final: {split_line(d['split']['cue_final'])}; coordinated: {split_line(d['split']['coordinated'])}")
                lines += [f"  - template {template}: {split_line(entry)}" for template, entry in d["split"]["per_template"].items()]
                lines += ladder_lines(d["ladder"])
            else:
                lines.append(f"- {label}: not evaluable ({len(y['scored_tokens'])} scored tokens; {'precondition ok' if pre['passed'] else 'PRECONDITION FAILED'})")
            if key == "Y2":
                lines += frame_lines(y.get("per_frame", {}))
                for frame_id, entry in confirmation["frames"].items():
                    if entry.get("template_defined") and not entry.get("valid"):
                        lines.append(f"  - frame {frame_id}: invalid at stage 1 (no fresh cue prompt run)")
            else:
                lines.append(f"  - per-frame row-entry minimum over the exposed frames {f(frame_minimum(y.get('per_frame', {})), 3)}")
        y3 = confirmation["Y3"]
        if y3.get("evaluable"):
            c = y3["cue_final"]
            lines.append(f"- Y3 (frozen pattern on the cue-final pairs of both sets, {c['n_pairs']} pairs, {c['n_tokens']} tokens): ΔT R² {f(c['frozen_dT_r2'], 3)} (rejected iff < {Y3_R2_MAX}) against Level 0 {f(c['level0_dT_r2'], 3)}, gap {f(c['gap'], 3)} (rejected iff ≥ {Y3_MARGIN}) → {'REJECTED' if y3['rejected'] else 'NOT REJECTED'}; Π variance share {f(c['Pi_variance_share'], 3)}")
            k = y3["coordinated_split"]
            lines.append(f"- Coordinated split (descriptive; predeclared: not rejected there): {split_line(k)}; would the criterion reject: {k['would_be_rejected']}")
            lines += [f"- Template {template} over both sets (descriptive): {split_line(entry)}" for template, entry in y3["per_template"].items()]
        else:
            lines.append(f"- Y3: not evaluable ({y3.get('cue_final', {}).get('n_tokens', 0)} cue-final token means)")
        if "ladder_both_sets" in confirmation:
            lines += ["- Ladder over both sets:"] + ladder_lines(confirmation["ladder_both_sets"])
        lines += ["", "| token | class | Y1 frames | Y1 ΔT̂ | Y1 ΔT | Y1 Π̂ | Y1 Π | Y1 frozen ΔT̂ | Y2 frames | Y2 ΔT̂ | Y2 ΔT | Y2 Π̂ | Y2 Π | self ΔÂ/ΔA (Y1) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        categories = {token["word"]: token["category"] for token in confirmation.get("tokens_meta", [])}
        for word in sorted(confirmation["Y1"]["tokens"], key=lambda w: -(confirmation["Y1"]["tokens"][w].get("dT_hat_mean") or -9)):
            a, b = confirmation["Y1"]["tokens"][word], confirmation["Y2"]["tokens"].get(word, {})
            lines.append(f"| {word} | {categories.get(word, '')} | {a.get('n_valid_frames')} | {f(a.get('dT_hat_mean'), 4)} | {f(a.get('dT_mean'), 4)} | {f(a.get('Pi_hat_mean'), 4)} | {f(a.get('Pi_mean'), 4)} | {f(a.get('dT_frozen_mean'), 4)} | "
                         f"{b.get('n_valid_frames')} | {f(b.get('dT_hat_mean'), 4)} | {f(b.get('dT_mean'), 4)} | {f(b.get('Pi_hat_mean'), 4)} | {f(b.get('Pi_mean'), 4)} | {f(a.get('self_hat_mean'), 3)} / {f(a.get('self_mean'), 3)} |")
        lines.append("")
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
