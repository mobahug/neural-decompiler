"""Experiment 020: does the decoded mechanism predict the model's own singular-versus-plural logit contrast?

Experiments 009–019 decoded the cue-to-transport chain end to end as a composed weight-only program and closed at the
transport head's output. This module carries that prediction through everything after layer 3 — blocks 3, 4 and 5 at
the prediction position, ``LN_final`` and the weight-only unembedding difference — to the contrast the whole chain
exists to explain. Nothing inherited is refitted: the Experiment 011 lock's axes and read weight, the Experiment 012
lock's template bases and the Experiment 017 lock's layer-3 bases enter through ``head_pattern.HeadChainModel``, whose
``upstream`` supplies the predicted layer-3 input change at the changed positions.

The Level 0 hypothesis is a downstream readout program with a small frozen-row second-transport term (the layer-5
value path from the cue position); the Level 1 exact chain is an identity that enters no floor. Every cue-dependent
quantity of the prediction path is predicted, never measured: the only inputs are the frame's reference capture, the
committed locks, the checkpoint weights and the cue's token id. Constants are copied from design revision 2
(commit ``cb4e7bb``, with the ``E₁`` amendment of ``179635a``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import attention_paths as ap
from . import attention_patterns as atp
from . import block_concentration as bc
from . import block_routing as br
from . import cue_decompilation as cd
from . import cue_suppression as cs
from . import encoding_read as er
from . import frame_channels as fch
from . import head_pattern as hp
from . import head_transport as ht
from . import layer_correction as lc
from . import plural_mechanism as pm
from . import read_assembly as ra
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest, Split
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 2).

EXPERIMENT_DIR = "experiments/020-readout-decompilation"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
EXPERIMENT_019_CONFIRMATION_PATH = br.CONFIRMATION_RELATIVE_PATH
EXPERIMENT_011_LOCK_PATH = hp.EXPERIMENT_011_LOCK_PATH
EXPERIMENT_012_LOCK_PATH = hp.EXPERIMENT_012_LOCK_PATH
EXPERIMENT_017_LOCK_PATH = bc.EXPERIMENT_017_LOCK_PATH
RUNTIME_SEED = br.RUNTIME_SEED
CONTROL_SEED = br.CONTROL_SEED
CONFIRMATION_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 1
FRAME_ID_TAG = "020"

READOUT_LAYERS = (3, 4, 5)  # the blocks this experiment decodes
PROGRAM_LAYERS = (1, 2, 3, 4, 5)  # the layer programs the chain and this module need
HEAD_LAYER = hp.HEAD_LAYER  # 3
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", *br.SCIENTIFIC_PATH_PREFIXES[1:])
NON_SCIENTIFIC_PATHS = (LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)

# Frozen numerical tolerances. Every one of them can stop a phase; none may be decided after fresh data exists.
READOUT_IDENTITY_TOLERANCE = 2e-2  # nats: ⟨ΔLN_final, Δw(n)⟩ against the measured log-probability contrast difference
LOGIT_IDENTITY_TOLERANCE = 2e-2  # LN_final(h6)·W_U + b_U against the captured logits
ADDITIVE_IDENTITY_TOLERANCE = 1e-4  # Δh6 against Δx3 + Σ(heads and MLPs of blocks 3–5)
LEVEL1_TOLERANCE = 1e-3  # the norm-normalized E₁ below
INHERITED_017_TOLERANCE = 1e-6  # this runner's Δ̂x3 against the Experiment 017 chain's own output
PREDICTION_REPRODUCTION_TOLERANCE = 0.0  # locked rows, canonical-JSON rounded: exact
PROVENANCE_TOLERANCE = 1e-6  # a prediction computed without any cue prompt against the ordinary one
NORM_FLOOR = 1e-12  # the denominator floor of every norm-normalized quantity

# Floors and outcomes (frozen; the exposed values that calibrated them are in the design).
Y1_TOKEN_MEAN_R2 = 0.80
Y1_PAIR_MEAN_R2 = 0.65
Y1_CUE_MAE = 1.5  # nats, per cue
Y1_CUE_MAE_SHARE = 0.80  # the share of scored cues that must meet it
Y1_POOLED_MAE = 1.0
Y2_FRAME_MEAN_R2 = 0.55
Y2_FRAME_R2 = 0.40
Y2_FRAME_SHARE = 0.75
Y2_SPLIT_R2 = 0.40
Y3_MEDIAN_R2 = 0.70
Y3_NOUN_R2 = 0.55
Y3_SLOPE_BAND = (0.75, 1.15)
Y3_BIAS = 0.8  # nats
Y3_SHARE = 0.90
MIN_SCORED_TOKENS = 16
MIN_VALID_EXPOSED_FRAMES = 60
MIN_VALID_FRESH_FRAMES = 12
MIN_VALID_COORDINATED_FRAMES = 4
MIN_SCORABLE_FRESH_NOUNS = 18
MIN_VALID_FRAMES_PER_TOKEN = 3
FRAME_CUE_EFFECT_RATE = hp.FRAME_CUE_EFFECT_RATE

OUTCOME_Y1 = ("CONTRAST_PREDICTED_TOKENS", "CONTRAST_NOT_PREDICTED_TOKENS", "PRECONDITION_FAILED_TOKENS")
OUTCOME_Y2 = ("CONTRAST_PREDICTED_FRAMES_CONDITIONAL", "CONTRAST_NOT_PREDICTED_FRAMES_CONDITIONAL", "PRECONDITION_FAILED_FRAMES")
OUTCOME_Y3 = ("NOUN_READOUT_FIXED", "NOUN_READOUT_NOT_ESTABLISHED", "PRECONDITION_FAILED_NOUNS")

# Comparators, with the standing frozen in the design.
COMPARATOR_STANDING = {
    "dT_only": "exposed-disfavoured baseline",
    "template_base_mlps": "exposed-disfavoured operating-point comparator",
    "no_l5_heads": "nested simplification comparator",
    "rank1_nouns": "descriptive comparator",
}

# The label-bearing populations. The runner asserts them; computing every readout in one pass is allowed, letting a
# fresh noun enter a Y1 or Y2 statistic is not.
POPULATIONS = {
    "Y1": {"cues": "fresh", "frames": "exposed", "nouns": "exposed_scorable"},
    "Y2": {"cues": "fresh", "frames": "fresh_valid", "nouns": "exposed_scorable"},
    "Y3": {"cues": "fresh", "frames": "exposed", "nouns": "fresh"},
    "joint": {"cues": "fresh", "frames": "fresh_valid", "nouns": "fresh"},
}

# Prompt-key manifest classes (the barrier is defined on them).
MANIFEST_CLASSES = ("S1-REF", "S1-VALIDITY", "S2-TARGET")

PHASES = ("validate", "freeze-confirmation", "explore", "lock", "confirm", "report")


class PhaseError(RuntimeError):
    """A protocol violation: a phase order, a frozen-input or an artifact failure."""


# ---------------------------------------------------------------------------
# The frozen confirmation candidates (tokenizer-checked on 2026-09-22; classes carry no expectation).

QUOTAS = {"determiner-like": 6, "quantity": 6, "possessive-or-pronoun": 6, "adjective": 6}
CANDIDATES = {
    "determiner-like": ("recent", "current", "original", "typical", "ordinary", "identical", "alternate", "random"),
    "quantity": ("average", "excessive", "exhaustive", "comprehensive", "thorough", "sweeping", "bulk", "spare", "dense", "lengthy", "lots", "loads", "tons", "scores"),
    "possessive-or-pronoun": ("anything", "something", "everything", "nothing", "who", "thee", "thou", "yourself", "themselves", "ones", "others"),
    "adjective": ("purple", "yellow", "hidden", "hard", "dirty", "rare", "square", "wild", "brave", "calm", "eager", "fierce", "humble", "proud", "shy", "lazy", "busy", "sturdy", "fragile", "polished"),
}
FRESH_FRAMES = (
    ("cardinal", "The workshop repairs {cue}"), ("cardinal", "The archive stores {cue}"), ("cardinal", "The orchard grows {cue}"),
    ("cardinal", "The foundry casts {cue}"), ("cardinal", "The kennel keeps {cue}"), ("cardinal", "The cellar holds {cue}"),
    ("quantifier", "The journal lists {cue}"), ("quantifier", "The roster names {cue}"), ("quantifier", "The manual covers {cue}"),
    ("quantifier", "The ledger records {cue}"), ("quantifier", "The brochure shows {cue}"), ("quantifier", "The harbor hosts {cue}"),
    ("coordinated-adjective", "Nils and Rosa painted {cue} pale"), ("coordinated-adjective", "Emil and Tessa folded {cue} neat"),
    ("coordinated-adjective", "Arne and Lena polished {cue} smooth"), ("coordinated-adjective", "Sven and Mira stacked {cue} high"),
    ("coordinated-adjective", "Bodil and Timo carved {cue} deep"), ("coordinated-adjective", "Reta and Olav wrapped {cue} tight"),
)
NOUN_QUOTA = 8
NOUN_CANDIDATES = {
    "simple-suffix": ("brick", "candle", "statue", "barrel", "curtain", "magnet", "puzzle", "pillar", "lantern", "crate", "cottage", "ribbon"),
    "sibilant-es": ("switch", "branch", "ash", "sketch", "batch", "flash", "arch", "crash", "launch", "inch", "church", "porch"),
    "consonant-y": ("colony", "gallery", "cavity", "battery", "category", "artery", "boundary", "anomaly", "apology", "laboratory", "assembly", "registry"),
}
FRESH_NOUN_SPLIT = Split.FUTURE_RESERVE  # the fresh nouns are a new set, recorded with this split label


def plural_form(word: str, rule_class: str) -> str:
    """The frozen regular-plural rule of each class."""
    if rule_class == "sibilant-es":
        return word + "es"
    if rule_class == "consonant-y":
        if not word.endswith("y") or word[-2] in "aeiou":
            raise ValueError(f"{word} is not a consonant-y noun")
        return word[:-1] + "ies"
    if rule_class == "simple-suffix":
        return word + "s"
    raise ValueError(f"unknown rule class {rule_class}")


# ---------------------------------------------------------------------------
# The exposed pool: Experiment 019's pool plus its confirmed tokens and frames.


def build_pool_020(manifest: ScreeningManifest, extension: pm.Extension, confirmation_006: cd.Confirmation, confirmation_009: ht.Confirmation009, confirmation_011: er.Confirmation011,
                   confirmation_012: lc.Confirmation012, confirmation_013: ap.Confirmation013, confirmation_014: Any, confirmation_015: atp.Confirmation015, confirmation_016: fch.Confirmation016,
                   confirmation_017: hp.Confirmation017, confirmation_018: bc.Confirmation018, confirmation_019: br.Confirmation019) -> cs.Pool008:
    base = br.build_pool_019(manifest, extension, confirmation_006, confirmation_009, confirmation_011, confirmation_012, confirmation_013, confirmation_014, confirmation_015, confirmation_016,
                             confirmation_017, confirmation_018)
    frames = base.frames + tuple(confirmation_019.frames)
    origin = dict(base.frame_origin) | {frame.frame_id: "confirmation-019" for frame in confirmation_019.frames}
    tokens = list(base.tokens)
    category, source = dict(base.token_category), dict(base.token_source)
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation_019.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"Experiment 019 token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-019"
    return cs.Pool008(frames, origin, tuple(tokens), category, source, base.nouns, base.noun_source, base.reference_ids, base.plural_cue)


# ---------------------------------------------------------------------------
# Nouns: the weight-only read, the scorable rule and the two populations.


@dataclass(frozen=True)
class NounSet:
    """The nouns a pair is read on: their frozen unembedding differences and the two label-bearing populations.

    A noun is *scorable* when both of its forms are single tokens. ``peach`` of the exposed manifest is not (its
    singular and plural share their first token, so ``Δw`` would be the zero vector and the contrast is undefined);
    it is dropped by this rule, not by choice, which is why the exposed population has 79 of 80 members.
    """

    nouns: tuple[pm.Noun, ...]  # every noun in evaluation order: the exposed ones first, then the fresh ones
    exposed_scorable: tuple[int, ...]  # indices into ``nouns``
    fresh: tuple[int, ...]
    dw: torch.Tensor  # [n, d_model], float64; zero rows for non-scorable nouns
    db: torch.Tensor  # [n], the unembedding bias difference; zero for non-scorable nouns
    non_scorable: tuple[str, ...]

    @classmethod
    def build(cls, weights: pm.Weights, exposed: Sequence[pm.Noun], fresh: Sequence[pm.Noun]) -> "NounSet":
        nouns = tuple(exposed) + tuple(fresh)
        W_U, b_U = weights.W_U.double(), weights.b_U.double()
        rows, biases, exposed_idx, fresh_idx, dropped = [], [], [], [], []
        for index, noun in enumerate(nouns):
            if noun.single_token:
                rows.append(W_U[:, noun.sg_ids[0]] - W_U[:, noun.pl_ids[0]])
                biases.append(float(b_U[noun.sg_ids[0]] - b_U[noun.pl_ids[0]]))
                (exposed_idx if index < len(exposed) else fresh_idx).append(index)
            else:
                rows.append(torch.zeros(W_U.shape[0], dtype=torch.float64))
                biases.append(0.0)
                dropped.append(noun.lexical_key)
        return cls(nouns, tuple(exposed_idx), tuple(fresh_idx), torch.stack(rows), torch.tensor(biases, dtype=torch.float64), tuple(dropped))

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(noun.lexical_key for noun in self.nouns)

    def population(self, name: str) -> tuple[int, ...]:
        if name == "exposed_scorable":
            return self.exposed_scorable
        if name == "fresh":
            return self.fresh
        raise ValueError(f"unknown noun population {name}")

    def contrasts(self, logits: torch.Tensor) -> torch.Tensor:
        """``log P(sg) − log P(pl)`` per noun from one logits vector; zero for a non-scorable noun."""
        log_probs = logits.double().log_softmax(dim=-1)
        values = []
        for noun in self.nouns:
            if noun.single_token:
                value = float(log_probs[noun.sg_ids[0]] - log_probs[noun.pl_ids[0]])
                if not math.isfinite(value):
                    raise pm.IncidentError(f"non-finite contrast for {noun.lexical_key}")
                values.append(value)
            else:
                values.append(0.0)
        return torch.tensor(values, dtype=torch.float64)

    def read(self, delta_ln: torch.Tensor) -> torch.Tensor:
        """The frozen readout: ``Δc(n) = ⟨ΔLN_final, Δw(n)⟩`` for every noun."""
        return self.dw @ delta_ln.double()

    def contrast_from_residual(self, program: "ReadoutProgram", residual: torch.Tensor) -> torch.Tensor:
        """``c(n) = ⟨LN_final(h6), Δw(n)⟩ + (b_sg − b_pl)(n)``: the contrast as a pure function of the final residual
        and the weights. The softmax normalizer cancels in the difference of two log-probabilities, so this needs no
        prompt — which is how a *fresh* noun's reference contrast in an *exposed* frame exists at confirm without any
        exposed reference prompt being re-executed and without explore ever computing a fresh-noun quantity."""
        return self.dw @ program.ln_final(residual) + self.db


# ---------------------------------------------------------------------------
# The frame's reference state: everything the readout program needs, all from the reference prompt.


def reference_sites_020(frame: pm.Frame, n_layers: int) -> tuple[pm.Site, ...]:
    """The extra capture sites of the reference run: the readout blocks' inputs at every position and the rows at p_t."""
    sites: list[pm.Site] = []
    for position in range(frame.p_t + 1):
        sites.extend((f"RESID_PRE.L{layer}", position) for layer in (4, 5))
    sites.append((f"RESID_POST.L{n_layers - 1}", frame.p_t))
    sites.extend((f"ATTN_PATTERN.L{layer}", frame.p_t) for layer in (4, 5))
    return tuple(sites)


@dataclass(frozen=True)
class FrameState020:
    """One frame's reference run: the Experiment 017 state plus the readout path's reference quantities."""

    state_017: hp.FrameState017
    x4_all: tuple[torch.Tensor, ...]  # RESID_PRE.L4 at positions 0..p_t
    x5_all: tuple[torch.Tensor, ...]
    h6: torch.Tensor  # RESID_POST.L5 at p_t
    rows4: torch.Tensor  # [heads, keys] reference attention row at query p_t
    rows5: torch.Tensor
    c_ref: torch.Tensor  # the reference contrasts of the noun set
    frame_ref: pm.Frame | None = None  # set when the state is rebuilt from the lock, where no captured run exists

    @property
    def frame(self) -> pm.Frame:
        return self.frame_ref if self.frame_ref is not None else self.state_017.frame

    @property
    def p_c(self) -> int:
        return self.frame_ref.p_c if self.frame_ref is not None else self.state_017.p_c

    @property
    def p_t(self) -> int:
        return self.frame_ref.p_t if self.frame_ref is not None else self.state_017.p_t

    @property
    def x3_all(self) -> Sequence[torch.Tensor]:
        return self.state_017.x3_all


def capture_frame_020(model: Any, head: ht.HeadWeights, reference: pm.Prompt, nouns: NounSet, axis_T: pm.SiteAxis) -> FrameState020:
    """**One** reference forward per frame: the Experiment 017 state and the readout path's reference quantities.

    Experiment 017's own ``capture_frame_017`` would run the same prompt a second time, so its body is repeated here
    over the union of the two site sets (its checks included) and the single captured run serves both states. One
    S1-REF execution per fresh frame is a protocol property, not an optimization: a test counts the executions.
    """
    frame = reference.frame
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    extra = list(hp.reference_sites_017(frame)) + list(reference_sites_020(frame, n_layers))
    ref, components = ra.capture_reference(model, head, reference, [noun for noun in nouns.nouns if noun.single_token], extra_sites=extra)
    x1_all = [components.pop(pm.site_label(("RESID_PRE.L1", k))).double() for k in range(frame.p_t + 1)]
    x2_all = [components.pop(pm.site_label(("RESID_PRE.L2", k))).double() for k in range(frame.p_t + 1)]
    A1 = components.pop(pm.site_label(("ATTN_PATTERN.L1", frame.p_c)))
    A2 = components.pop(pm.site_label(("ATTN_PATTERN.L2", frame.p_c)))
    x4_all = tuple(components.pop(pm.site_label(("RESID_PRE.L4", k))).double() for k in range(frame.p_t + 1))
    x5_all = tuple(components.pop(pm.site_label(("RESID_PRE.L5", k))).double() for k in range(frame.p_t + 1))
    h6 = components.pop(pm.site_label((f"RESID_POST.L{n_layers - 1}", frame.p_t))).double()
    rows4 = components.pop(pm.site_label(("ATTN_PATTERN.L4", frame.p_t))).double().reshape(n_heads, -1)[:, : frame.p_t + 1]
    rows5 = components.pop(pm.site_label(("ATTN_PATTERN.L5", frame.p_t))).double().reshape(n_heads, -1)[:, : frame.p_t + 1]
    if len(ref.residuals) != frame.p_t + 1 or ref.attention.shape[0] <= frame.p_t:
        raise pm.IncidentError(f"{frame.frame_id}: the reference capture does not cover every position up to p_t")
    if not torch.equal(x1_all[frame.p_c], ref.vectors["R0"].double()):
        raise pm.IncidentError(f"{frame.frame_id}: the captured residual before block 1 disagrees with the R0 stage vector of the same run")
    if A1.shape[0] != ap.N_HEADS or A1.shape[1] <= frame.p_c or A2.shape[0] != ap.N_HEADS:
        raise pm.IncidentError(f"{frame.frame_id}: unexpected attention pattern shape {tuple(A1.shape)} / {tuple(A2.shape)}")
    state_013 = ap.FrameState013(ref, components, ra.read_functional(head, ref, axis_T), x1_all[frame.p_c], x2_all[frame.p_c], x1_all[: frame.p_c + 1], x2_all[: frame.p_c + 1], A1, A2)
    state_017 = hp.FrameState017(state_013, x1_all, x2_all, [x.double() for x in ref.residuals], ref.attention.double()[: frame.p_t + 1])
    c_ref = torch.tensor([float(ref.c_by_noun.get(noun.lexical_key, 0.0)) for noun in nouns.nouns], dtype=torch.float64)
    return FrameState020(state_017, x4_all, x5_all, h6, rows4, rows5, c_ref)


def assert_explore_nouns(nouns: NounSet, confirmation: Confirmation020) -> None:
    """Noun freshness at computation time: the explore path may build no quantity of a fresh noun, so the noun set it
    measures with carries neither a fresh key nor a fresh token id. Serialization checks come later and separately."""
    if nouns.fresh:
        raise PhaseError("the exploration noun set contains fresh nouns; explore scores the exposed scorable nouns only")
    fresh_keys = {noun.lexical_key for noun in confirmation.nouns}
    fresh_ids = {token for noun in confirmation.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    for index, noun in enumerate(nouns.nouns):
        if noun.lexical_key in fresh_keys or set(noun.sg_ids) & fresh_ids or set(noun.pl_ids) & fresh_ids:
            raise PhaseError(f"the exploration noun set reaches the fresh noun {noun.lexical_key} (index {index})")


# ---------------------------------------------------------------------------
# The Level 0 readout program and the Level 1 exact chain.


@dataclass(frozen=True)
class ReadoutProgram:
    """Blocks 3–5 at the changed positions, ``LN_final`` and the noun read.

    Every argument is a reference-run quantity, a committed lock's object, a checkpoint weight or a predicted change:
    the program cannot reach a fresh cue's forward pass, which ``provenance_difference`` demonstrates.
    """

    lw: lc.LayerWeights  # layers 3, 4, 5
    programs: Mapping[int, atp.LayerProgram]  # layers 3, 4, 5
    ln_final_w: torch.Tensor
    ln_final_b: torch.Tensor
    eps: float

    @classmethod
    def from_model(cls, model: Any) -> "ReadoutProgram":
        weights = pm.Weights.from_model(model)
        return cls(lc.LayerWeights.from_model(model, layers=READOUT_LAYERS), {layer: atp.LayerProgram.from_model(model, layer) for layer in READOUT_LAYERS},
                   weights.ln_final_w.double(), weights.ln_final_b.double(), float(weights.eps))

    def ln_final(self, residual: torch.Tensor) -> torch.Tensor:
        return pm.exact_layer_norm(residual.double(), self.ln_final_w, self.ln_final_b, self.eps)

    def mlp_delta(self, layer: int, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """The block's MLP change at an operating point: ``MLP(base + delta) − MLP(base)``, exact GELU."""
        return self.lw.delta_out(layer, base, delta)

    def attention_delta(self, layer: int, reference: Sequence[torch.Tensor], changes: Mapping[int, torch.Tensor], query: int, *, rows: torch.Tensor | None = None) -> torch.Tensor:
        """The layer's attention output change at ``query``: the exact program when ``rows`` is None, else the frozen
        reference rows times the value changes (the frozen-row form used for block 5)."""
        program = self.programs[layer]
        if rows is None:
            rr = atp.ReferenceRow(program, [reference[k].double() for k in range(query + 1)])
            normed = {k: program.normalize(reference[k].double() + change) for k, change in changes.items() if k <= query}
            new_rows, values = hp._row_and_values(program, rr, normed.get(query), normed)
            return hp._attention_output(program, new_rows, values) - rr.output_ref
        new = torch.stack([program.v(program.normalize(reference[k].double() + changes.get(k, torch.zeros_like(reference[k].double())))) for k in range(query + 1)], dim=1)
        old = torch.stack([program.v(program.normalize(reference[k].double())) for k in range(query + 1)], dim=1)
        return torch.einsum("hk,hkd->d", rows[:, : query + 1], program.output(new - old))

    def blocks_3_to_5(self, state: FrameState020, dx3: Mapping[int, torch.Tensor], *, l5_heads: bool = True, mlps_at_base: Mapping[int, torch.Tensor] | None = None) -> dict[str, Any]:
        """Level 0: from the predicted layer-3 input change to the final residual change at ``p_t``.

        Block 3: every head by the exact layer program on the predicted state, MLP at the frame's operating point
        (the parallel residual of GPT-NeoX: each block's MLP reads the block's input, not the post-attention state).
        Block 4: attention frozen at the reference, MLP at the operating point.
        Block 5: heads as the frame's reference rows times the predicted value changes at every changed key (the
        second transport term), MLP at the operating point.
        """
        p_c, p_t = state.p_c, state.p_t
        positions = sorted(dx3)
        dh3: dict[int, torch.Tensor] = {}
        parts: dict[str, torch.Tensor] = {}
        for position in positions:
            attn3 = self.attention_delta(3, state.x3_all, dx3, position)
            mlp3 = self.mlp_delta(3, state.x3_all[position].double(), dx3[position])
            dh3[position] = dx3[position] + attn3 + mlp3
            if position == p_t:
                parts["block3_attention"], parts["block3_mlp"] = attn3, mlp3
        dh4: dict[int, torch.Tensor] = {}
        for position in positions:
            base = state.x4_all[position] if mlps_at_base is None else mlps_at_base[4]
            mlp4 = self.mlp_delta(4, base, dh3[position]) if mlps_at_base is None else (self.lw.out(4, mlps_at_base[4] + dh3[position]) - self.lw.out(4, mlps_at_base[4]))
            dh4[position] = dh3[position] + mlp4
            if position == p_t:
                parts["block4_mlp"] = mlp4
        a5 = torch.zeros_like(state.h6)
        if l5_heads:
            a5 = self.attention_delta(5, state.x5_all, dh4, p_t, rows=state.rows5)
        parts["block5_attention"] = a5
        # Parallel residual (GPT-NeoX): every block's MLP reads the block's *input*, never the post-attention residual.
        base5 = state.x5_all[p_t] if mlps_at_base is None else mlps_at_base[5]
        mlp5 = self.mlp_delta(5, base5, dh4[p_t]) if mlps_at_base is None else (self.lw.out(5, mlps_at_base[5] + dh4[p_t]) - self.lw.out(5, mlps_at_base[5]))
        parts["block5_mlp"] = mlp5
        return {"dh6": dh4[p_t] + a5 + mlp5, "dh3": dh3, "dh4": dh4, "parts": parts}

    def delta_ln(self, state: FrameState020, dh6: torch.Tensor) -> torch.Tensor:
        return self.ln_final(state.h6 + dh6) - self.ln_final(state.h6)

    def contrast(self, state: FrameState020, dh6: torch.Tensor, nouns: NounSet) -> torch.Tensor:
        """The predicted contrast change per noun."""
        return nouns.read(self.delta_ln(state, dh6))

    def level1(self, state: FrameState020, dx3_measured: Mapping[int, torch.Tensor], *, l4_heads: bool = True) -> torch.Tensor:
        """The exact chain from the measured layer-3 input change: every head and MLP recomputed at the frame's own
        state, with no reduction. An identity; it enters no floor."""
        p_t = state.p_t
        positions = sorted(dx3_measured)
        dh3 = {p: dx3_measured[p] + self.attention_delta(3, state.x3_all, dx3_measured, p) + self.mlp_delta(3, state.x3_all[p].double(), dx3_measured[p]) for p in positions}
        dh4 = {}
        for position in positions:
            attn4 = self.attention_delta(4, state.x4_all, dh3, position) if l4_heads else torch.zeros_like(dh3[position])
            dh4[position] = dh3[position] + attn4 + self.mlp_delta(4, state.x4_all[position], dh3[position])
        a5 = self.attention_delta(5, state.x5_all, dh4, p_t)
        return dh4[p_t] + a5 + self.mlp_delta(5, state.x5_all[p_t], dh4[p_t])


def level1_error(predicted: torch.Tensor, measured: torch.Tensor) -> float:
    """``E₁ = ‖Δh6^exact − Δh6^meas‖_∞ / max(‖Δh6^meas‖_∞, 1e-12)``.

    Norm-normalized by construction: a componentwise ratio would explode wherever a residual component is near zero
    and could stop the experiment for a numerical artefact.
    """
    denominator = max(float(measured.double().abs().max()), NORM_FLOOR)
    return float((predicted.double() - measured.double()).abs().max()) / denominator


def relative_error(predicted: torch.Tensor, measured: torch.Tensor) -> float:
    """The same normalization for every other quantity declared *relative* in the design."""
    return level1_error(predicted, measured)


# ---------------------------------------------------------------------------
# The inherited upstream: Experiment 017's chain, used verbatim.


def chain_from_locks(lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_017: Mapping[str, Any], lw: lc.LayerWeights, programs: Mapping[int, atp.LayerProgram],
                     pool: cs.Pool008) -> hp.HeadChainModel:
    """The committed Experiment 017 chain. Nothing here is refitted; the locks' objects are passed through."""
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    return hp.model_from_locks(lock_011, lock_012, lw, programs, pool.reference_ids, plural_ids, hp.bases_from_json(lock_017["bases_3"]))


def predicted_dx3(chain: hp.HeadChainModel, weights: pm.Weights, state: FrameState020, rows16: Mapping[int, atp.ReferenceRow], token_id: int, template: str) -> dict[int, torch.Tensor]:
    """Experiment 017's predicted layer-3 input change at the changed positions (variant LEVEL0)."""
    up = chain.upstream(weights, rows16, state.state_017.x1_all, state.state_017.x2_all, state.p_c, state.p_t, token_id, template, hp.LEVEL0)
    return {position: value.double() for position, value in up["dx3"].items()}


# ---------------------------------------------------------------------------
# Identities.


def readout_identity(nouns: NounSet, program: ReadoutProgram, state: FrameState020, h6_measured: torch.Tensor, c_measured: torch.Tensor) -> float:
    """``Δc(n) = ⟨ΔLN_final, Δw(n)⟩`` against the measured log-probability contrast difference, over scorable nouns."""
    predicted = nouns.read(program.ln_final(h6_measured) - program.ln_final(state.h6))
    measured = c_measured - state.c_ref
    index = list(nouns.exposed_scorable) + list(nouns.fresh)
    return float((predicted[index] - measured[index]).abs().max())


def additive_identity(h6_change: torch.Tensor, dx3: torch.Tensor, components: Sequence[torch.Tensor]) -> float:
    """``Δh6 = Δx3 + Σ(heads and MLPs of blocks 3–5)``."""
    total = dx3.double().clone()
    for component in components:
        total = total + component.double()
    return float((h6_change.double() - total).abs().max())


def logit_identity(program: ReadoutProgram, weights: pm.Weights, h6: torch.Tensor, logits: torch.Tensor) -> float:
    return float((program.ln_final(h6) @ weights.W_U.double() + weights.b_U.double() - logits.double()).abs().max())


def enforce(name: str, value: float, tolerance: float, *, context: str = "") -> float:
    """Every identity and reproduction check goes through here: above its frozen tolerance it is an incident."""
    if not math.isfinite(value) or value > tolerance:
        raise pm.IncidentError(f"{name} failed: {value:.3e} above the frozen tolerance {tolerance:.3e}{(' — ' + context) if context else ''}")
    return float(value)


def provenance_difference(with_prompt: torch.Tensor, without_prompt: torch.Tensor) -> float:
    return float((with_prompt.double() - without_prompt.double()).abs().max())


# ---------------------------------------------------------------------------
# Statistics on the frozen populations.


def _r2(y: torch.Tensor, y_hat: torch.Tensor) -> float | None:
    y, y_hat = y.double().reshape(-1), y_hat.double().reshape(-1)
    denominator = float(((y - y.mean()) ** 2).sum())
    if denominator <= 0.0:
        return None
    return 1.0 - float(((y - y_hat) ** 2).sum()) / denominator


def _mae(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    return float((y.double() - y_hat.double()).abs().mean())


def _slope(y: torch.Tensor, y_hat: torch.Tensor) -> float | None:
    denominator = float((y.double() * y.double()).sum())
    if denominator <= 0.0:
        return None
    return float((y_hat.double() * y.double()).sum()) / denominator


@dataclass(frozen=True)
class ScoringTable:
    """The measured and predicted contrast changes of one set, with their pair keys and the noun population."""

    cues: tuple[str, ...]  # per pair
    frames: tuple[str, ...]  # per pair
    templates: tuple[str, ...]  # per pair
    measured: torch.Tensor  # [pairs, nouns]
    predicted: torch.Tensor
    noun_keys: tuple[str, ...]

    def subset(self, indices: Sequence[int]) -> "ScoringTable":
        rows = list(indices)
        return ScoringTable(tuple(self.cues[i] for i in rows), tuple(self.frames[i] for i in rows), tuple(self.templates[i] for i in rows),
                            self.measured[rows], self.predicted[rows], self.noun_keys)

    def by(self, key: str) -> dict[str, list[int]]:
        values = {"cue": self.cues, "frame": self.frames, "template": self.templates}[key]
        groups: dict[str, list[int]] = {}
        for index, value in enumerate(values):
            groups.setdefault(value, []).append(index)
        return groups


def pair_statistics(table: ScoringTable) -> dict[str, Any]:
    """Every aggregation the floors use, on one table. The flattened value is reported, never a guard on its own."""
    measured, predicted = table.measured.double(), table.predicted.double()
    per_cue = {cue: {"r2": _r2(measured[rows], predicted[rows]), "mae": _mae(measured[rows], predicted[rows]), "n_pairs": len(rows)} for cue, rows in table.by("cue").items()}
    per_frame = {frame: {"r2": _r2(measured[rows], predicted[rows]), "mae": _mae(measured[rows], predicted[rows]), "n_pairs": len(rows)} for frame, rows in table.by("frame").items()}
    per_template = {template: {"r2": _r2(measured[rows], predicted[rows]), "n_pairs": len(rows)} for template, rows in table.by("template").items()}
    cue_groups = table.by("cue")
    token_means = torch.tensor([float(measured[rows].mean()) for _, rows in sorted(cue_groups.items())], dtype=torch.float64)
    token_means_hat = torch.tensor([float(predicted[rows].mean()) for _, rows in sorted(cue_groups.items())], dtype=torch.float64)
    frame_groups = table.by("frame")
    frame_means = torch.tensor([float(measured[rows].mean()) for _, rows in sorted(frame_groups.items())], dtype=torch.float64)
    frame_means_hat = torch.tensor([float(predicted[rows].mean()) for _, rows in sorted(frame_groups.items())], dtype=torch.float64)
    cue_final = [i for i, template in enumerate(table.templates) if template != pm.COORDINATED_TEMPLATE]
    coordinated = [i for i, template in enumerate(table.templates) if template == pm.COORDINATED_TEMPLATE]
    return {
        "n_pairs": int(measured.shape[0]), "n_nouns": int(measured.shape[1]),
        "flattened_r2": _r2(measured, predicted), "pooled_mae": _mae(measured, predicted),
        "pair_mean_r2": _r2(measured.mean(dim=1), predicted.mean(dim=1)),
        "token_mean_r2": _r2(token_means, token_means_hat), "token_mean_mae": _mae(token_means, token_means_hat),
        "frame_mean_r2": _r2(frame_means, frame_means_hat), "frame_mean_mae": _mae(frame_means, frame_means_hat),
        "per_cue": per_cue, "per_frame": per_frame, "per_template": per_template,
        "cue_final_r2": _r2(measured[cue_final], predicted[cue_final]) if cue_final else None,
        "coordinated_r2": _r2(measured[coordinated], predicted[coordinated]) if coordinated else None,
    }


def noun_statistics(table: ScoringTable) -> dict[str, Any]:
    """Per-noun `R²`, slope and bias over one table's pairs (the Y3 quantities)."""
    measured, predicted = table.measured.double(), table.predicted.double()
    rows = {}
    for index, key in enumerate(table.noun_keys):
        y, y_hat = measured[:, index], predicted[:, index]
        rows[key] = {"r2": _r2(y, y_hat), "slope": _slope(y, y_hat), "bias": float((y_hat - y).mean()), "scorable": _r2(y, y_hat) is not None}
    return rows


def _share(values: Sequence[bool]) -> float:
    return float(sum(1 for value in values if value)) / float(len(values)) if values else 0.0


def score_y1(table: ScoringTable, *, n_valid_frames: int) -> dict[str, Any]:
    statistics = pair_statistics(table)
    scored = [cue for cue, entry in statistics["per_cue"].items() if entry["n_pairs"] >= MIN_VALID_FRAMES_PER_TOKEN]
    precondition = {"scored_tokens": len(scored) >= MIN_SCORED_TOKENS, "valid_frames": n_valid_frames >= MIN_VALID_EXPOSED_FRAMES,
                    "n_scored_tokens": len(scored), "n_valid_frames": n_valid_frames}
    precondition["ok"] = bool(precondition["scored_tokens"] and precondition["valid_frames"])
    if not precondition["ok"]:
        return {"population": POPULATIONS["Y1"], "statistics": statistics, "precondition": precondition, "label": OUTCOME_Y1[2], "conditions": {}}
    mae_ok = _share([statistics["per_cue"][cue]["mae"] <= Y1_CUE_MAE for cue in scored])
    conditions = {
        "token_mean_r2": (statistics["token_mean_r2"] or -1.0) >= Y1_TOKEN_MEAN_R2,
        "pair_mean_r2": (statistics["pair_mean_r2"] or -1.0) >= Y1_PAIR_MEAN_R2,
        "cue_mae_share": mae_ok >= Y1_CUE_MAE_SHARE,
        "pooled_mae": statistics["pooled_mae"] <= Y1_POOLED_MAE,
    }
    label = OUTCOME_Y1[0] if all(conditions.values()) else OUTCOME_Y1[1]
    return {"population": POPULATIONS["Y1"], "statistics": statistics, "precondition": precondition, "conditions": conditions,
            "cue_mae_share": mae_ok, "scored_tokens": sorted(scored), "label": label,
            "floors": {"token_mean_r2": Y1_TOKEN_MEAN_R2, "pair_mean_r2": Y1_PAIR_MEAN_R2, "cue_mae": Y1_CUE_MAE, "cue_mae_share": Y1_CUE_MAE_SHARE, "pooled_mae": Y1_POOLED_MAE}}


def score_y2(table: ScoringTable, *, valid_frames: Sequence[str], valid_coordinated: int) -> dict[str, Any]:
    statistics = pair_statistics(table)
    precondition = {"valid_fresh_frames": len(valid_frames) >= MIN_VALID_FRESH_FRAMES, "valid_coordinated": valid_coordinated >= MIN_VALID_COORDINATED_FRAMES,
                    "n_valid_fresh_frames": len(valid_frames), "n_valid_coordinated": valid_coordinated}
    precondition["ok"] = bool(precondition["valid_fresh_frames"] and precondition["valid_coordinated"])
    if not precondition["ok"]:
        return {"population": POPULATIONS["Y2"], "statistics": statistics, "precondition": precondition, "label": OUTCOME_Y2[2], "conditions": {}}
    frame_share = _share([(entry["r2"] or -1.0) >= Y2_FRAME_R2 for entry in statistics["per_frame"].values()])
    conditions = {
        "frame_mean_r2": (statistics["frame_mean_r2"] or -1.0) >= Y2_FRAME_MEAN_R2,
        "frame_share": frame_share >= Y2_FRAME_SHARE,
        "cue_final_split": (statistics["cue_final_r2"] or -1.0) >= Y2_SPLIT_R2,
        "coordinated_split": (statistics["coordinated_r2"] or -1.0) >= Y2_SPLIT_R2,
    }
    label = OUTCOME_Y2[0] if all(conditions.values()) else OUTCOME_Y2[1]
    return {"population": POPULATIONS["Y2"], "statistics": statistics, "precondition": precondition, "conditions": conditions, "frame_share": frame_share,
            "valid_frames": sorted(valid_frames), "label": label,
            "floors": {"frame_mean_r2": Y2_FRAME_MEAN_R2, "frame_r2": Y2_FRAME_R2, "frame_share": Y2_FRAME_SHARE, "split_r2": Y2_SPLIT_R2}}


def score_y3(table: ScoringTable) -> dict[str, Any]:
    rows = noun_statistics(table)
    scorable = {key: entry for key, entry in rows.items() if entry["scorable"]}
    precondition = {"scorable_nouns": len(scorable) >= MIN_SCORABLE_FRESH_NOUNS, "n_scorable_nouns": len(scorable)}
    precondition["ok"] = bool(precondition["scorable_nouns"])
    if not precondition["ok"]:
        return {"population": POPULATIONS["Y3"], "nouns": rows, "precondition": precondition, "label": OUTCOME_Y3[2], "conditions": {}}
    values = sorted(entry["r2"] for entry in scorable.values())
    median = values[len(values) // 2] if len(values) % 2 else 0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2])
    conditions = {
        "median_r2": median >= Y3_MEDIAN_R2,
        "r2_share": _share([entry["r2"] >= Y3_NOUN_R2 for entry in scorable.values()]) >= Y3_SHARE,
        "slope_share": _share([Y3_SLOPE_BAND[0] <= (entry["slope"] or -1.0) <= Y3_SLOPE_BAND[1] for entry in scorable.values()]) >= Y3_SHARE,
        "bias_share": _share([abs(entry["bias"]) <= Y3_BIAS for entry in scorable.values()]) >= Y3_SHARE,
    }
    label = OUTCOME_Y3[0] if all(conditions.values()) else OUTCOME_Y3[1]
    return {"population": POPULATIONS["Y3"], "nouns": rows, "precondition": precondition, "conditions": conditions,
            "median_r2": median, "min_r2": values[0], "label": label,
            "floors": {"median_r2": Y3_MEDIAN_R2, "noun_r2": Y3_NOUN_R2, "slope_band": list(Y3_SLOPE_BAND), "bias": Y3_BIAS, "share": Y3_SHARE}}


def outcome_label(y1: Mapping[str, Any], y2: Mapping[str, Any], y3: Mapping[str, Any]) -> dict[str, Any]:
    labels = [y1["label"], y2["label"], y3["label"]]
    for label, allowed in zip(labels, (OUTCOME_Y1, OUTCOME_Y2, OUTCOME_Y3)):
        if label not in allowed:
            raise ValueError(f"{label} is not a frozen outcome")
    return {"labels": labels, "label": " | ".join(labels)}


# ---------------------------------------------------------------------------
# The confirmation set: fresh cues, fresh frames, fresh nouns and the three manifest classes.

CONFIRMATION_DIGEST_KEYS = (*br.CONFIRMATION_DIGEST_KEYS, "confirmation_019")
CONFIRMATION_DIGEST_FIELDS = tuple(f"{key}_sha256" for key in CONFIRMATION_DIGEST_KEYS)


@dataclass(frozen=True)
class Confirmation020:
    reference_ids: Mapping[str, int]
    frames: tuple[pm.Frame, ...]  # the 18 fresh frames
    exposed_frame_ids: tuple[str, ...]
    tokens: tuple[dict[str, Any], ...]  # the 24 fresh cues
    nouns: tuple[pm.Noun, ...]  # the 24 fresh nouns (never prompted)
    token_prompts: tuple[pm.Prompt, ...]  # fresh cues × fresh frames (S2-TARGET, Y2 block)
    exposed_frame_prompts: tuple[pm.Prompt, ...]  # fresh cues × exposed frames (S2-TARGET, Y1 block)
    content_sha256: str

    def reference_prompt(self, frame: pm.Frame) -> pm.Prompt:
        return pm.Prompt(frame, self.reference_ids[frame.template_id], "ref")

    def validity_prompt(self, frame: pm.Frame) -> pm.Prompt:
        """The frozen plural cue-pair prompt that decides the frame's validity (an exposed cue in a fresh frame)."""
        return pm.Prompt(frame, frame.cue_ids["pl"], "pl")

    @property
    def stage1_prompts(self) -> tuple[pm.Prompt, ...]:
        return tuple(self.reference_prompt(frame) for frame in self.frames) + tuple(self.validity_prompt(frame) for frame in self.frames)

    @property
    def target_prompts(self) -> tuple[pm.Prompt, ...]:
        return self.exposed_frame_prompts + self.token_prompts

    @property
    def all_prompts(self) -> tuple[pm.Prompt, ...]:
        return self.stage1_prompts + self.target_prompts


def manifest_classes(confirmation: Confirmation020) -> dict[str, list[str]]:
    """The three frozen classes; the barrier is defined on S2-TARGET."""
    return {
        "S1-REF": sorted(confirmation.reference_prompt(frame).key for frame in confirmation.frames),
        "S1-VALIDITY": sorted(confirmation.validity_prompt(frame).key for frame in confirmation.frames),
        "S2-TARGET": sorted(prompt.key for prompt in confirmation.target_prompts),
    }


def prompt_key_manifest(confirmation: Confirmation020) -> list[str]:
    classes = manifest_classes(confirmation)
    return sorted({key for name in MANIFEST_CLASSES for key in classes[name]})


def fresh_tokens_020(tokenizer: Any, excluded_ids: set[int]) -> list[dict[str, Any]]:
    """The first eligible entries of each frozen class list: single token with a leading space, id unused."""
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


def fresh_nouns_020(tokenizer: Any, exposed_nouns: Sequence[pm.Noun], excluded_ids: set[int]) -> list[pm.Noun]:
    """The first eligible entries of each rule class: both forms single token with a leading space, neither form an
    exposed noun form or an exposed cue token. No model output is involved, and no statistic of a fresh noun has ever
    been computed — that, not the absence of a prompt, is what makes them fresh."""
    exposed_keys = {noun.lexical_key for noun in exposed_nouns}
    exposed_ids = {token for noun in exposed_nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    chosen: list[pm.Noun] = []
    for rule_class, words in NOUN_CANDIDATES.items():
        count = 0
        for word in words:
            plural = plural_form(word, rule_class)
            sg, pl = pm._encode(tokenizer, " " + word), pm._encode(tokenizer, " " + plural)
            if len(sg) != 1 or len(pl) != 1 or word in exposed_keys or plural in exposed_keys:
                continue
            if sg[0] in exposed_ids or pl[0] in exposed_ids or sg[0] in excluded_ids or pl[0] in excluded_ids:
                continue
            chosen.append(pm.Noun(word, FRESH_NOUN_SPLIT, rule_class, sg, pl))
            count += 1
            if count == NOUN_QUOTA:
                break
    return chosen


def _expected_frames() -> list[tuple[str, str, str]]:
    expected, counters = [], {}
    for template_id, text_template in FRESH_FRAMES:
        counters[template_id] = counters.get(template_id, 0) + 1
        expected.append((f"{template_id}-{FRAME_ID_TAG}-{counters[template_id]}", template_id, text_template))
    return expected


def build_confirmation_payload(tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> dict[str, Any]:
    """Tokenizer only: no model is loaded and no model output exists when this runs."""
    excluded_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    tokens = fresh_tokens_020(tokenizer, set(excluded_ids))
    nouns = fresh_nouns_020(tokenizer, pool.nouns, set(excluded_ids) | {entry["token_id"] for entry in tokens})
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
    frame_objects = [pm.Frame(entry["template_id"], entry["frame_id"], tuple(entry["prefix_ids"]), tuple(entry["suffix_ids"]), entry["cue_ids"], entry["text_template"], origin="extension") for entry in frames]
    for token in tokens:
        token["licensed_frames"] = [frame.frame_id for frame in frame_objects]
    payload = {
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        **{field: digests[key] for field, key in zip(CONFIRMATION_DIGEST_FIELDS, CONFIRMATION_DIGEST_KEYS)},
        "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
        "reference_cue_ids": dict(pool.reference_ids),
        "exposed_token_ids": sorted(token_id for _, token_id in pool.tokens),
        "exposed_frame_ids": [frame.frame_id for frame in pool.frames],
        "exposed_noun_keys": [noun.lexical_key for noun in pool.nouns],
        "policy": {
            "licensing": "every fresh cue in every exposed frame (Y1) and in every fresh frame (Y2); every fresh noun read on both",
            "quotas": dict(QUOTAS), "noun_quota": NOUN_QUOTA,
            "populations": {name: dict(value) for name, value in POPULATIONS.items()},
            "validity": {"frame": f"the frozen head-informative rule and the cue-pair rule at rate {FRAME_CUE_EFFECT_RATE}; an invalid frame is a scientific fact, never an incident",
                         "noun": "both forms single token and a non-degenerate measured contrast over the set's pairs",
                         "sets": f"Y1 at least {MIN_SCORED_TOKENS} scored cues and {MIN_VALID_EXPOSED_FRAMES} valid exposed frames; Y2 at least {MIN_VALID_FRESH_FRAMES} valid fresh frames with {MIN_VALID_COORDINATED_FRAMES} coordinated; Y3 at least {MIN_SCORABLE_FRESH_NOUNS} scorable fresh nouns"},
            "expectations": "none: the committed per-pair per-noun predicted contrast changes are the only predictions; no class, frame or noun carries a preregistered label",
        },
        "tokens": tokens,
        "frames": frames,
        "nouns": [noun.to_dict() for noun in nouns],
        "token_prompts": [_prompt_entry(tokenizer, frame, token) for frame in frame_objects for token in tokens],
        "exposed_frame_prompts": [_prompt_entry(tokenizer, frame, token) for frame in pool.frames for token in tokens],
        "construction": "tokenizer-only; the first eligible entries of the frozen candidate lists; no model output; the fresh nouns were never scored or inspected",
    }
    provisional = validate_confirmation({**payload, "manifest": {name: [] for name in MANIFEST_CLASSES}, "content_sha256": ""}, pool, digests, check_manifest=False)
    payload["manifest"] = manifest_classes(provisional)
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def _prompt_entry(tokenizer: Any, frame: pm.Frame, token: Mapping[str, Any]) -> dict[str, Any]:
    prompt = pm.Prompt(frame, int(token["token_id"]), token["word"])
    return {"frame_id": frame.frame_id, "word": token["word"], "token_id": int(token["token_id"]), "key": prompt.key,
            "text": pm._decode(tokenizer, prompt.token_ids), "token_ids": list(prompt.token_ids)}


def validate_confirmation(payload: Mapping[str, Any], pool: cs.Pool008, digests: Mapping[str, str], *, check_manifest: bool = True) -> Confirmation020:
    expected_keys = {"schema_version", *CONFIRMATION_DIGEST_FIELDS, "model", "reference_cue_ids", "exposed_token_ids", "exposed_frame_ids", "exposed_noun_keys", "policy", "tokens", "frames",
                     "nouns", "token_prompts", "exposed_frame_prompts", "construction", "manifest", "content_sha256"}
    pm._require_exact_keys(payload, expected_keys, "confirmation")
    if payload["schema_version"] != CONFIRMATION_SCHEMA_VERSION:
        raise ValueError("confirmation schema version is not frozen")
    if check_manifest and payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("confirmation content_sha256 does not match canonical payload")
    if tuple(payload[field] for field in CONFIRMATION_DIGEST_FIELDS) != tuple(digests[key] for key in CONFIRMATION_DIGEST_KEYS):
        raise ValueError("confirmation was built against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}:
        raise ValueError("confirmation was built for a different pinned model")
    if dict(payload["reference_cue_ids"]) != dict(pool.reference_ids) or list(payload["exposed_token_ids"]) != sorted(token_id for _, token_id in pool.tokens):
        raise ValueError("reference cues or exposed token IDs disagree with the exposed pool")
    if list(payload["exposed_frame_ids"]) != [frame.frame_id for frame in pool.frames] or list(payload["exposed_noun_keys"]) != [noun.lexical_key for noun in pool.nouns]:
        raise ValueError("exposed frames or nouns disagree with the exposed pool")
    frames = tuple(pm._parse_frame(entry, "extension") for entry in payload["frames"])
    if [(frame.frame_id, frame.template_id, frame.text_template) for frame in frames] != _expected_frames():
        raise ValueError("fresh frames are not the frozen literal frames")
    exposed_texts = {frame.text_template for frame in pool.frames}
    cue_ids_by_template = {frame.template_id: dict(frame.cue_ids) for frame in pool.frames if pool.frame_origin[frame.frame_id] == "manifest"}
    for frame in frames:
        if frame.text_template in exposed_texts or dict(frame.cue_ids) != cue_ids_by_template[frame.template_id]:
            raise ValueError(f"{frame.frame_id}: fresh frame text exposed or original cue tokens missing")
    exposed_ids = {token_id for _, token_id in pool.tokens} | {token for noun in pool.nouns for token in (*noun.sg_ids, *noun.pl_ids)}
    all_frame_ids = [frame.frame_id for frame in frames]
    tokens = []
    for entry in payload["tokens"]:
        pm._require_exact_keys(entry, {"word", "token_id", "category", "licensed_frames"}, "token")
        if entry["category"] not in CANDIDATES or entry["word"] not in CANDIDATES[entry["category"]]:
            raise ValueError(f"fresh token {entry['word']} is not an entry of its frozen class list")
        if entry["token_id"] in exposed_ids or list(entry["licensed_frames"]) != all_frame_ids:
            raise ValueError(f"fresh token {entry['word']} is exposed or mislicensed")
        tokens.append(dict(entry))
    if [entry["word"] for entry in tokens] != [entry["word"] for entry in fresh_tokens_from_payload(tokens)]:
        raise ValueError("fresh tokens are not in the frozen order")
    counts: dict[str, int] = {}
    for entry in tokens:
        counts[entry["category"]] = counts.get(entry["category"], 0) + 1
    if counts != dict(QUOTAS):
        raise ValueError(f"fresh token quotas are not the frozen ones: {counts}")
    exposed_keys = {noun.lexical_key for noun in pool.nouns}
    nouns: list[pm.Noun] = []
    for entry in payload["nouns"]:
        pm._require_exact_keys(entry, {"lexical_key", "split", "rule_class", "sg_ids", "pl_ids", "single_token"}, "noun")
        noun = pm.Noun(entry["lexical_key"], Split(entry["split"]), entry["rule_class"], tuple(entry["sg_ids"]), tuple(entry["pl_ids"]))
        if noun.lexical_key in exposed_keys:
            raise ValueError(f"fresh noun {noun.lexical_key} is an exposed noun")
        if noun.lexical_key not in NOUN_CANDIDATES.get(noun.rule_class, ()):
            raise ValueError(f"fresh noun {noun.lexical_key} is not an entry of its frozen class list")
        if not noun.single_token or set(noun.sg_ids) & exposed_ids or set(noun.pl_ids) & exposed_ids:
            raise ValueError(f"fresh noun {noun.lexical_key} is not single token or collides with an exposed id")
        if entry["single_token"] is not True:
            raise ValueError(f"fresh noun {noun.lexical_key} is recorded as multi-token")
        nouns.append(noun)
    noun_counts: dict[str, int] = {}
    for noun in nouns:
        noun_counts[noun.rule_class] = noun_counts.get(noun.rule_class, 0) + 1
    if noun_counts != {rule_class: NOUN_QUOTA for rule_class in NOUN_CANDIDATES}:
        raise ValueError(f"fresh noun quotas are not the frozen ones: {noun_counts}")
    token_prompts = tuple(pm.Prompt(frame, int(entry["token_id"]), entry["word"]) for frame in frames for entry in tokens)
    exposed_frame_prompts = tuple(pm.Prompt(frame, int(entry["token_id"]), entry["word"]) for frame in pool.frames for entry in tokens)
    if [entry["key"] for entry in payload["token_prompts"]] != [prompt.key for prompt in token_prompts]:
        raise ValueError("fresh-frame prompt keys disagree with the frozen frames and tokens")
    if [entry["key"] for entry in payload["exposed_frame_prompts"]] != [prompt.key for prompt in exposed_frame_prompts]:
        raise ValueError("exposed-frame prompt keys disagree with the frozen frames and tokens")
    confirmation = Confirmation020(dict(pool.reference_ids), frames, tuple(frame.frame_id for frame in pool.frames), tuple(tokens), tuple(nouns), token_prompts, exposed_frame_prompts,
                                   str(payload["content_sha256"]))
    if check_manifest:
        pm._require_exact_keys(payload["manifest"], set(MANIFEST_CLASSES), "manifest")
        expected = manifest_classes(confirmation)
        if {name: list(payload["manifest"][name]) for name in MANIFEST_CLASSES} != expected:
            raise ValueError("the prompt-key manifest is not the frozen S1-REF / S1-VALIDITY / S2-TARGET classes")
        if len(expected["S1-REF"]) != len(frames) or len(expected["S1-VALIDITY"]) != len(frames):
            raise ValueError("the stage-1 manifest does not hold exactly one reference and one validity key per fresh frame")
        if set(expected["S2-TARGET"]) & (set(expected["S1-REF"]) | set(expected["S1-VALIDITY"])):
            raise ValueError("the stage-1 and stage-2 manifest classes overlap")
    return confirmation


def fresh_tokens_from_payload(tokens: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The frozen order: class by class in ``CANDIDATES`` order, each class in its list order."""
    ordered: list[dict[str, Any]] = []
    for category in CANDIDATES:
        ordered.extend(sorted((dict(entry) for entry in tokens if entry["category"] == category), key=lambda entry: CANDIDATES[entry["category"]].index(entry["word"])))
    return ordered


def freeze_confirmation(path: Path, tokenizer: Any, pool: cs.Pool008, digests: Mapping[str, str]) -> str:
    if path.exists():
        raise PhaseError(f"{path} already exists; the confirmation set is frozen once")
    payload = build_confirmation_payload(tokenizer, pool, digests)
    validate_confirmation(payload, pool, digests)
    validate_json_safe(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    return str(payload["content_sha256"])


def load_confirmation(path: Path, pool: cs.Pool008, digests: Mapping[str, str]) -> Confirmation020:
    if not path.exists():
        raise PhaseError(f"{path} does not exist; freeze the confirmation set first")
    return validate_confirmation(json.loads(path.read_text(encoding="utf-8")), pool, digests)


# ---------------------------------------------------------------------------
# Results state (the shared protocol shape of Experiments 013–019).


def new_results_state(*, digests: Mapping[str, str], protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {"schema_version": RESULTS_SCHEMA_VERSION, "run_id": pm.sha256_text(pm.canonical_json(dict(digests)) + protocol_code_commit + pm.utc_now())[:16], "created_at": pm.utc_now(),
            **{f"{key}_sha256": value for key, value in digests.items()}, "protocol_code_commit": protocol_code_commit, "git_dirty": False,
            "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "versions": dict(versions),
            "phases": {phase: {"status": "not_started"} for phase in PHASES}, "executed_prompt_keys": [], "executed_noun_keys": [], "exploration": {}, "lock": None, "confirmation": None}


def state_digests(state: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(state[f"{key}_sha256"]) for key in CONFIRMATION_DIGEST_KEYS)


def digest_tuple(digests: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(str(digests[key]) for key in CONFIRMATION_DIGEST_KEYS)


def write_results_state(path: Path, state: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    validate_json_safe(payload)
    digest = pm.sha256_text(pm.canonical_json(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pm.canonical_json({**payload, "state_sha256": digest}) + "\n", encoding="utf-8")
    return digest


def load_results_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    recorded = state.pop("state_sha256", None)
    if recorded != pm.sha256_text(pm.canonical_json(state)):
        raise PhaseError("results state digest does not verify")
    return {**state, "state_sha256": recorded}


def assert_phase_allowed(phase: str, state: Mapping[str, Any]) -> None:
    try:
        ht.assert_phase_allowed(phase, state)
    except pm.PhaseError as error:
        raise PhaseError(str(error)) from None


def assert_confirmation_untouched(state: Mapping[str, Any], confirmation: Confirmation020) -> None:
    executed = set(state["executed_prompt_keys"])
    forbidden = {prompt.key for prompt in confirmation.all_prompts}
    if executed & forbidden:
        raise PhaseError("confirmation prompts were executed before the confirmation boundary")


def assert_no_target_prompt_executed(state: Mapping[str, Any], confirmation: Confirmation020) -> None:
    """The barrier: no S2-TARGET key may be in the ledger when stage 2 begins."""
    executed = set(state["executed_prompt_keys"])
    targets = set(manifest_classes(confirmation)["S2-TARGET"])
    overlap = executed & targets
    if overlap:
        raise PhaseError(f"{len(overlap)} stage-2 target prompts were executed before the barrier, e.g. {sorted(overlap)[:3]}")


def assert_fresh_nouns_absent(record: Mapping[str, Any], confirmation: Confirmation020) -> None:
    """Freshness for a noun is the absence of prior inspection: no exploration record may name one or carry its ids."""
    text = pm.canonical_json(record)
    for noun in confirmation.nouns:
        if f'"{noun.lexical_key}"' in text:
            raise PhaseError(f"the exploration record names the fresh noun {noun.lexical_key}")
        for token_id in (*noun.sg_ids, *noun.pl_ids):
            if f":{token_id}," in text or f":{token_id}}}" in text or f"[{token_id}," in text or f" {token_id}," in text:
                raise PhaseError(f"the exploration record carries a token id of the fresh noun {noun.lexical_key}")


# ---------------------------------------------------------------------------
# Measurement of one pair and the predictions of one pair.

EXPLORE_TOKEN_LIMIT: int | None = None  # the exposed cues measured per frame at explore; None is every one of them


def measured_sites_020(frame: pm.Frame, n_layers: int) -> tuple[pm.Site, ...]:
    """What a cue prompt is captured for: the measured layer-3 input change and the measured final residual."""
    positions = sorted({frame.p_c, frame.p_t})
    return tuple([(f"RESID_PRE.L{HEAD_LAYER}", position) for position in positions] + [(f"RESID_POST.L{n_layers - 1}", frame.p_t)])


@dataclass(frozen=True)
class PairMeasurement:
    """One cue prompt's measured quantities, all at the frame's own positions."""

    word: str
    token_id: int
    frame_id: str
    template: str
    dc: torch.Tensor  # measured contrast change per noun
    dx3: dict[int, torch.Tensor]  # measured layer-3 input change at the changed positions
    dh6: torch.Tensor  # measured final residual change at p_t


def measure_pair(model: Any, state: FrameState020, nouns: NounSet, word: str, token_id: int, *, c_ref: torch.Tensor | None = None) -> PairMeasurement:
    """One cue prompt. ``c_ref`` defaults to the captured reference contrasts; at confirm it is the residual-derived
    one, so that a fresh noun's reference contrast never required an exposed reference prompt."""
    frame = state.frame
    run = pm.capture_prompt(model, pm.Prompt(frame, token_id, word), measured_sites_020(frame, int(model.cfg.n_layers)))
    positions = sorted({state.p_c, state.p_t})
    reference = state.c_ref if c_ref is None else c_ref
    return PairMeasurement(word, token_id, frame.frame_id, frame.template_id,
                           nouns.contrasts(run.logits) - reference,
                           {position: run.vector((f"RESID_PRE.L{HEAD_LAYER}", position)).double() - state.x3_all[position].double() for position in positions},
                           run.vector((f"RESID_POST.L{int(model.cfg.n_layers) - 1}", frame.p_t)).double() - state.h6)


def predict_pair(program: ReadoutProgram, chain: hp.HeadChainModel, weights: pm.Weights, state: FrameState020, rows16: Mapping[int, atp.ReferenceRow], nouns: NounSet, token_id: int,
                 *, template_bases: Mapping[str, Mapping[int, torch.Tensor]] | None = None, dT_direction: torch.Tensor | None = None) -> dict[str, Any]:
    """The Level 0 prediction of one pair and the nested comparators, from the reference state and the token id only."""
    dx3 = predicted_dx3(chain, weights, state, rows16, token_id, state.frame.template_id)
    level0 = program.blocks_3_to_5(state, dx3)
    out = {"dx3": dx3, "dh6": level0["dh6"], "dc": program.contrast(state, level0["dh6"], nouns), "parts": level0["parts"]}
    out["dc_no_l5_heads"] = program.contrast(state, program.blocks_3_to_5(state, dx3, l5_heads=False)["dh6"], nouns)
    if template_bases is not None:
        bases = template_bases[state.frame.template_id]
        out["dc_template_base"] = program.contrast(state, program.blocks_3_to_5(state, dx3, mlps_at_base=bases)["dh6"], nouns)
    if dT_direction is not None:
        out["dT"] = float(level0["parts"]["block3_attention"].double() @ dT_direction.double())
    return out


def ceiling_prediction(program: ReadoutProgram, state: FrameState020, nouns: NounSet, dx3_measured: Mapping[int, torch.Tensor]) -> torch.Tensor:
    """The same downstream program fed the *measured* layer-3 input change: the downstream ceiling, never a floor."""
    return program.contrast(state, program.blocks_3_to_5(state, dict(dx3_measured))["dh6"], nouns)


# ---------------------------------------------------------------------------
# Exploration (once, exposed pool only).


def template_bases_020(states: Mapping[str, FrameState020], frames: Sequence[pm.Frame]) -> dict[str, dict[int, torch.Tensor]]:
    """The template-mean reference inputs of blocks 4 and 5 at p_t: the operating-point comparator's bases."""
    bases: dict[str, dict[int, torch.Tensor]] = {}
    for template in pm.TEMPLATE_ORDER:
        members = [states[frame.frame_id] for frame in frames if frame.template_id == template]
        if not members:
            continue
        bases[template] = {4: torch.stack([state.x4_all[state.p_t] for state in members]).mean(dim=0),
                           5: torch.stack([state.x5_all[state.p_t] for state in members]).mean(dim=0)}
    return bases


def rank1_noun_factor(table: ScoringTable) -> dict[str, Any]:
    """The descriptive comparator's frozen objects: the exposed table's rank-1 noun vector and the direction d_noun
    that reproduces it from the weight-only Δw. Fitted once at explore, written into the lock, never refitted."""
    measured = table.measured.double()
    _, singular, right = torch.linalg.svd(measured, full_matrices=False)
    share = float((singular[0] ** 2) / (singular ** 2).sum())
    return {"noun_vector": right[0].tolist(), "rank1_share": share}


def run_exploration(model: Any, pool: cs.Pool008, *, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_017: Mapping[str, Any], confirmation: Confirmation020,
                    state: dict[str, Any], results_path: Path | None, log: Any = None) -> dict[str, Any]:
    """Tier A: the exposed pool only. Every noun here is an exposed scorable one; no fresh-noun quantity is built."""
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model, layers=PROGRAM_LAYERS)
    programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in PROGRAM_LAYERS}
    chain = chain_from_locks(lock_011, lock_012, lock_017, lw, programs, pool)
    program = ReadoutProgram(lc.LayerWeights.from_model(model, layers=READOUT_LAYERS), {layer: programs[layer] for layer in READOUT_LAYERS},
                             weights.ln_final_w.double(), weights.ln_final_b.double(), float(weights.eps))
    axis_T = pm.SiteAxis("T", torch.zeros_like(torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64)), torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64), float(lock_011["sigma_T"]))
    nouns = NounSet.build(weights, pool.nouns, [])
    assert_explore_nouns(nouns, confirmation)
    exploration = state["exploration"]
    exploration["noun_population"] = {"exposed": len(pool.nouns), "exposed_scorable": len(nouns.exposed_scorable), "non_scorable": list(nouns.non_scorable)}
    say(f"exposed nouns: {len(nouns.exposed_scorable)} scorable of {len(pool.nouns)} (non-scorable {list(nouns.non_scorable)})")

    states: dict[str, FrameState020] = {}
    rows16: dict[str, dict[int, atp.ReferenceRow]] = {}
    for frame in pool.frames:
        states[frame.frame_id] = capture_frame_020(model, head, pool.reference_prompt(frame), nouns, axis_T)
        rows16[frame.frame_id] = atp.reference_rows(programs, states[frame.frame_id].state_017.x1_all, states[frame.frame_id].state_017.x2_all)
    say(f"reference states: {len(states)} frames, one forward each")
    bases = template_bases_020(states, pool.frames)
    dT_direction = axis_T.direction.double()

    identities: dict[str, float] = {}
    cues, frame_ids, templates, measured_rows, level0_rows, ceiling_rows, no_l5_rows, base_rows, dT_values = [], [], [], [], [], [], [], [], []
    scorable = list(nouns.exposed_scorable)
    for frame in pool.frames:
        frame_state = states[frame.frame_id]
        reference_id = pool.reference_ids[frame.template_id]
        tokens = [(word, token_id) for word, token_id in pool.tokens if token_id != reference_id]
        if EXPLORE_TOKEN_LIMIT is not None:
            tokens = tokens[:EXPLORE_TOKEN_LIMIT]
        for word, token_id in tokens:
            measurement = measure_pair(model, frame_state, nouns, word, token_id)
            prediction = predict_pair(program, chain, weights, frame_state, rows16[frame.frame_id], nouns, token_id, template_bases=bases, dT_direction=dT_direction)
            identities["readout"] = max(identities.get("readout", 0.0), abs_max(program, nouns, frame_state, measurement))
            identities["level1"] = max(identities.get("level1", 0.0), level1_error(program.level1(frame_state, measurement.dx3), measurement.dh6))
            identities["inherited_017"] = max(identities.get("inherited_017", 0.0), inherited_reproduction(chain, weights, frame_state, rows16[frame.frame_id], token_id, prediction["dx3"]))
            cues.append(word); frame_ids.append(frame.frame_id); templates.append(frame.template_id)
            measured_rows.append(measurement.dc[scorable]); level0_rows.append(prediction["dc"][scorable])
            ceiling_rows.append(ceiling_prediction(program, frame_state, nouns, measurement.dx3)[scorable])
            no_l5_rows.append(prediction["dc_no_l5_heads"][scorable]); base_rows.append(prediction["dc_template_base"][scorable])
            dT_values.append(prediction["dT"])
        pm.record_execution(state, tuple(pm.Prompt(frame, token_id, word) for word, token_id in tokens) + (pool.reference_prompt(frame),), pool.nouns)
        if results_path is not None and len(cues) % 2000 < len(tokens):
            write_results_state(results_path, state)
        say(f"  {frame.frame_id}: {len(tokens)} exposed cues measured and predicted")
    enforce("readout identity", identities["readout"], READOUT_IDENTITY_TOLERANCE)
    enforce("Level 1 exact chain", identities["level1"], LEVEL1_TOLERANCE)
    enforce("inherited Experiment 017 reproduction", identities["inherited_017"], INHERITED_017_TOLERANCE)

    noun_keys = tuple(nouns.nouns[index].lexical_key for index in scorable)
    table = ScoringTable(tuple(cues), tuple(frame_ids), tuple(templates), torch.stack(measured_rows), torch.stack(level0_rows), noun_keys)
    exploration["statistics"] = pair_statistics(table)
    exploration["nouns"] = noun_statistics(table)
    exploration["comparators"] = {
        "ceiling_measured_dx3": pair_statistics(ScoringTable(table.cues, table.frames, table.templates, table.measured, torch.stack(ceiling_rows), noun_keys))["flattened_r2"],
        "no_l5_heads": pair_statistics(ScoringTable(table.cues, table.frames, table.templates, table.measured, torch.stack(no_l5_rows), noun_keys))["flattened_r2"],
        "template_base_mlps": pair_statistics(ScoringTable(table.cues, table.frames, table.templates, table.measured, torch.stack(base_rows), noun_keys))["flattened_r2"],
        "dT_only": dT_only_fit(torch.tensor(dT_values, dtype=torch.float64), table),
        "standing": dict(COMPARATOR_STANDING),
    }
    exploration["rank1"] = rank1_noun_factor(table)
    exploration["identities"] = {name: float(value) for name, value in identities.items()}
    exploration["locked_states"] = {frame_id: locked_state(state_020) for frame_id, state_020 in states.items()}
    exploration["template_bases"] = {template: {str(layer): vector.tolist() for layer, vector in entry.items()} for template, entry in bases.items()}
    exploration["n_pairs"] = len(cues)
    assert_fresh_nouns_absent(exploration, confirmation)
    say(f"exposed record: {len(cues)} pairs × {len(noun_keys)} nouns; flattened R² {exploration['statistics']['flattened_r2']:.4f}; ceiling {exploration['comparators']['ceiling_measured_dx3']:.4f}")
    return exploration


def abs_max(program: ReadoutProgram, nouns: NounSet, state: FrameState020, measurement: PairMeasurement) -> float:
    """The readout identity of one pair: the measured contrast change against the read of the measured ΔLN_final."""
    predicted = nouns.read(program.ln_final(state.h6 + measurement.dh6) - program.ln_final(state.h6))
    index = list(nouns.exposed_scorable) + list(nouns.fresh)
    return float((predicted[index] - measurement.dc[index]).abs().max())


def inherited_reproduction(chain: hp.HeadChainModel, weights: pm.Weights, state: FrameState020, rows16: Mapping[int, atp.ReferenceRow], token_id: int, dx3: Mapping[int, torch.Tensor]) -> float:
    """Experiment 017's chain recomputed: this runner must not perturb the inherited prediction."""
    again = predicted_dx3(chain, weights, state, rows16, token_id, state.frame.template_id)
    return max(float((again[position] - dx3[position]).abs().max()) for position in dx3)


def dT_only_fit(dT: torch.Tensor, table: ScoringTable) -> dict[str, Any]:
    """The exposed-disfavoured baseline: one fitted noun vector on the head's own output change."""
    measured = table.measured.double()
    denominator = float((dT * dT).sum())
    if denominator <= 0.0:
        return {"r2": None, "noun_vector": None}
    noun_vector = (dT.unsqueeze(1) * measured).sum(dim=0) / denominator
    return {"r2": _r2(measured, dT.unsqueeze(1) * noun_vector), "noun_vector": noun_vector.tolist()}


def locked_state(state: FrameState020) -> dict[str, Any]:
    """What the lock stores per exposed frame: every reference quantity the readout program needs."""
    return {"p_c": state.p_c, "p_t": state.p_t,
            "x3_all": [x.tolist() for x in state.x3_all], "x4_all": [x.tolist() for x in state.x4_all], "x5_all": [x.tolist() for x in state.x5_all],
            "h6": state.h6.tolist(), "rows4": state.rows4.tolist(), "rows5": state.rows5.tolist(), "c_ref": state.c_ref.tolist(),
            "x1_all": [x.tolist() for x in state.state_017.x1_all], "x2_all": [x.tolist() for x in state.state_017.x2_all]}


def state_digest(entry: Mapping[str, Any]) -> str:
    return pm.sha256_text(pm.canonical_json(entry))


# ---------------------------------------------------------------------------
# The lock: the Y1 prediction table, written with no forward pass.

PREDICTION_COLUMNS = ("token", "frame_id", "template", "dc", "dc_no_l5_heads", "dc_template_base", "dT")


def state_from_locked(entry: Mapping[str, Any], frame: pm.Frame) -> FrameState020:
    """The reference state as the lock stores it: no capture is involved, so the frame is supplied explicitly."""
    tensor = lambda values: torch.tensor(values, dtype=torch.float64)  # noqa: E731
    x1_all = [tensor(x) for x in entry["x1_all"]]
    x2_all = [tensor(x) for x in entry["x2_all"]]
    x3_all = [tensor(x) for x in entry["x3_all"]]
    state_017 = hp.FrameState017(_LockedState013(entry["p_c"], entry["p_t"]), x1_all, x2_all, x3_all, torch.zeros(len(x3_all), dtype=torch.float64))
    return FrameState020(state_017, tuple(tensor(x) for x in entry["x4_all"]), tuple(tensor(x) for x in entry["x5_all"]), tensor(entry["h6"]),
                         tensor(entry["rows4"]), tensor(entry["rows5"]), tensor(entry["c_ref"]), frame)


@dataclass(frozen=True)
class _LockedState013:
    """The positions the locked state carries; the readout program needs nothing else of Experiment 013's state."""

    p_c: int
    p_t: int


def prediction_rows(program: ReadoutProgram, chain: hp.HeadChainModel, weights: pm.Weights, states: Mapping[str, FrameState020], rows16: Mapping[str, Mapping[int, atp.ReferenceRow]],
                    nouns: NounSet, frames: Sequence[pm.Frame], tokens: Sequence[Mapping[str, Any]], bases: Mapping[str, Mapping[int, torch.Tensor]], dT_direction: torch.Tensor,
                    *, population: str = "exposed_scorable") -> list[dict[str, Any]]:
    """One row per (token, frame): the predicted contrast change per noun of the named population, and the comparators."""
    index = list(nouns.population(population))
    rows: list[dict[str, Any]] = []
    for frame in frames:
        state = states[frame.frame_id]
        for token in tokens:
            prediction = predict_pair(program, chain, weights, state, rows16[frame.frame_id], nouns, int(token["token_id"]), template_bases=bases, dT_direction=dT_direction)
            rows.append({"token": token["word"], "frame_id": frame.frame_id, "template": frame.template_id,
                         "dc": [float(value) for value in prediction["dc"][index]],
                         "dc_no_l5_heads": [float(value) for value in prediction["dc_no_l5_heads"][index]],
                         "dc_template_base": [float(value) for value in prediction["dc_template_base"][index]],
                         "dT": float(prediction["dT"])})
    return rows


def frozen_floors() -> dict[str, Any]:
    return {"Y1": {"token_mean_r2": Y1_TOKEN_MEAN_R2, "pair_mean_r2": Y1_PAIR_MEAN_R2, "cue_mae": Y1_CUE_MAE, "cue_mae_share": Y1_CUE_MAE_SHARE, "pooled_mae": Y1_POOLED_MAE},
            "Y2": {"frame_mean_r2": Y2_FRAME_MEAN_R2, "frame_r2": Y2_FRAME_R2, "frame_share": Y2_FRAME_SHARE, "split_r2": Y2_SPLIT_R2},
            "Y3": {"median_r2": Y3_MEDIAN_R2, "noun_r2": Y3_NOUN_R2, "slope_band": list(Y3_SLOPE_BAND), "bias": Y3_BIAS, "share": Y3_SHARE},
            "preconditions": {"scored_tokens": MIN_SCORED_TOKENS, "valid_exposed_frames": MIN_VALID_EXPOSED_FRAMES, "valid_fresh_frames": MIN_VALID_FRESH_FRAMES,
                              "valid_coordinated": MIN_VALID_COORDINATED_FRAMES, "scorable_fresh_nouns": MIN_SCORABLE_FRESH_NOUNS},
            "tolerances": {"readout": READOUT_IDENTITY_TOLERANCE, "logit": LOGIT_IDENTITY_TOLERANCE, "additive": ADDITIVE_IDENTITY_TOLERANCE, "level1": LEVEL1_TOLERANCE,
                           "inherited_017": INHERITED_017_TOLERANCE, "prediction_reproduction": PREDICTION_REPRODUCTION_TOLERANCE, "provenance": PROVENANCE_TOLERANCE, "norm_floor": NORM_FLOOR}}


def build_candidate_lock(*, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation020, rows: Sequence[Mapping[str, Any]], noun_keys: Sequence[str],
                         protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    lock = {"experiment": "020", "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit,
            **{f"{key}_sha256": digests[key] for key in CONFIRMATION_DIGEST_KEYS},
            "confirmation_020_sha256": confirmation.content_sha256,
            "populations": {name: dict(value) for name, value in POPULATIONS.items()},
            "noun_keys": list(noun_keys), "fresh_noun_keys": [noun.lexical_key for noun in confirmation.nouns],
            "locked_states": exploration["locked_states"], "template_bases": exploration["template_bases"],
            "rank1": exploration["rank1"], "dT_only_noun_vector": exploration["comparators"]["dT_only"]["noun_vector"],
            "floors": frozen_floors(), "comparator_standing": dict(COMPARATOR_STANDING),
            "confirmation_prompt_manifest": manifest_classes(confirmation),
            "predictions": {"rows": [dict(row) for row in rows], "columns": list(PREDICTION_COLUMNS)}}
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json(lock))
    return lock


def assert_lock_predictions_reproduced(lock: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> float:
    """Every locked Y1 row recomputed from the locked states and the weights: exact, before anything fresh runs."""
    locked = lock["predictions"]["rows"]
    if len(locked) != len(rows):
        raise PhaseError(f"the lock holds {len(locked)} prediction rows and {len(rows)} were recomputed")
    worst = 0.0
    for left, right in zip(locked, rows):
        if (left["token"], left["frame_id"], left["template"]) != (right["token"], right["frame_id"], right["template"]):
            raise PhaseError("the recomputed prediction rows are not in the locked order")
        worst = max(worst, atp._max_numeric_difference(dict(right), dict(left), "prediction row"))
    if worst > PREDICTION_REPRODUCTION_TOLERANCE:
        raise PhaseError(f"the locked predictions do not reproduce: max difference {worst:.3e}")
    return float(worst)


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: Confirmation020, predictions_text: str, git_state: Mapping[str, Any],
                  tracked: bool, changed_paths: Sequence[str] | None) -> None:
    if lock.get("experiment") != "020":
        raise PhaseError("the installed lock is not Experiment 020's")
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("the lock's content digest does not verify")
    if state.get("lock") is None or state["lock"]["content_sha256"] != lock["content_sha256"]:
        raise PhaseError("the installed lock is not the candidate this run wrote")
    if tuple(lock[f"{key}_sha256"] for key in CONFIRMATION_DIGEST_KEYS) != digest_tuple(digests) or lock["confirmation_020_sha256"] != confirmation.content_sha256:
        raise PhaseError("the lock was written against different frozen inputs")
    if lock["floors"] != frozen_floors() or lock["populations"] != {name: dict(value) for name, value in POPULATIONS.items()}:
        raise PhaseError("the lock's floors or populations are not the frozen ones")
    if lock["confirmation_prompt_manifest"] != manifest_classes(confirmation):
        raise PhaseError("the lock's prompt manifest is not the frozen one")
    if lock["comparator_standing"] != dict(COMPARATOR_STANDING):
        raise PhaseError("the lock's comparator standing is not the frozen one")
    if not tracked:
        raise PhaseError("the lock and predictions must be tracked and committed")
    if git_state.get("dirty"):
        raise PhaseError("confirm requires a clean Git tree")
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = [path for path in changed_paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and path not in NON_SCIENTIFIC_PATHS and not path.startswith(NON_SCIENTIFIC_PREFIXES)]
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock: {scientific}")
    if pm.sha256_text(predictions_text) != state["lock"]["predictions_sha256"]:
        raise PhaseError("the installed predictions artifact is not the candidate this run wrote")


def render_predictions(lock: Mapping[str, Any]) -> str:
    rows = lock["predictions"]["rows"]
    noun_keys = lock["noun_keys"]
    lines = [f"# Experiment 020 — preregistered predictions (the decoded mechanism's singular-versus-plural contrast change)", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_020_sha256']}`",
             f"- Populations: Y1 {lock['populations']['Y1']}, Y2 {lock['populations']['Y2']}, Y3 {lock['populations']['Y3']}; the joint fresh-noun statistic carries no threshold",
             f"- Floors: {lock['floors']['Y1']} (Y1), {lock['floors']['Y2']} (Y2), {lock['floors']['Y3']} (Y3)",
             f"- Y1 table: {len(rows)} rows (fresh cues × exposed frames), each with the predicted Δĉ of the {len(noun_keys)} scorable exposed nouns, the nested no-layer-5-head and template-base comparators and the head's own ΔT̂",
             f"- Comparator standing: {lock['comparator_standing']}", "",
             "| token | frame | template | mean Δĉ | mean Δĉ (no L05 heads) | mean Δĉ (template-base MLPs) | ΔT̂ |", "|---|---|---|---|---|---|---|"]
    for row in rows[:40]:
        mean = sum(row["dc"]) / len(row["dc"]) if row["dc"] else float("nan")
        mean_no5 = sum(row["dc_no_l5_heads"]) / len(row["dc_no_l5_heads"]) if row["dc_no_l5_heads"] else float("nan")
        mean_base = sum(row["dc_template_base"]) / len(row["dc_template_base"]) if row["dc_template_base"] else float("nan")
        lines.append(f"| {row['token']} | {row['frame_id']} | {row['template']} | {mean:.4f} | {mean_no5:.4f} | {mean_base:.4f} | {row['dT']:.4f} |")
    if len(rows) > 40:
        lines.append(f"| … | … | … | … | … | … | … |  <!-- {len(rows) - 40} further rows in the lock -->")
    lines += ["", "Every row is a prediction of the model's own contrast change, computed from the frame's reference state, the committed Experiment 011/012/017 locks and the cue's token id alone.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The two-stage confirmation.


def stage_digest(rows: Sequence[Mapping[str, Any]], states: Mapping[str, Any], frames: Mapping[str, Any]) -> str:
    return pm.sha256_text(pm.canonical_json({"rows": [dict(row) for row in rows], "states": dict(states), "frames": dict(frames)}))


def assert_stage_one_digest(stage1: Mapping[str, Any]) -> None:
    """The barrier's own check: the re-read stage-1 record must digest to what was written."""
    recomputed = stage_digest(stage1["rows"], stage1["states"], stage1["frames"])
    if recomputed != stage1.get("digest"):
        raise PhaseError("the stage-1 record does not reproduce its digest; no fresh cue prompt may run")


def stage_one(model: Any, pool: cs.Pool008, confirmation: Confirmation020, lock: Mapping[str, Any], lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_017: Mapping[str, Any],
              *, protocol_code_commit: str, log: Any = None) -> dict[str, Any]:
    """Only the fresh frames' S1-REF and S1-VALIDITY prompts, their reference states and the frame-conditional
    prediction table; then the digest. An invalid frame is a scientific fact recorded here, never an incident: its
    S2-TARGET prompts simply never run."""
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model, layers=PROGRAM_LAYERS)
    programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in PROGRAM_LAYERS}
    chain = chain_from_locks(lock_011, lock_012, lock_017, lw, programs, pool)
    program = ReadoutProgram(lc.LayerWeights.from_model(model, layers=READOUT_LAYERS), {layer: programs[layer] for layer in READOUT_LAYERS},
                             weights.ln_final_w.double(), weights.ln_final_b.double(), float(weights.eps))
    axis_T = pm.SiteAxis("T", torch.zeros_like(torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64)), torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64), float(lock_011["sigma_T"]))
    nouns = NounSet.build(weights, pool.nouns, confirmation.nouns)
    bases = {template: {int(layer): torch.tensor(vector, dtype=torch.float64) for layer, vector in entry.items()} for template, entry in lock["template_bases"].items()}
    cache = pm.PromptCache(model, tuple(pool.single_nouns))
    say("stage 1: the fresh frames' reference and validity prompts, their reference states and the frame-conditional prediction table — no fresh cue prompt")
    frames_out: dict[str, Any] = {}
    states: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for frame in confirmation.frames:
        template = frame.template_id
        state = capture_frame_020(model, head, confirmation.reference_prompt(frame), nouns, axis_T)
        rows16 = atp.reference_rows(programs, state.state_017.x1_all, state.state_017.x2_all)
        singular, plural = pm.frame_prompts(frame)
        c_sg, c_pl = cache.c(singular), cache.c(plural)
        positive = sum(1 for noun in pool.single_nouns if c_sg[noun.lexical_key] - c_pl[noun.lexical_key] > 0)
        required = pm.exact_count_floor(FRAME_CUE_EFFECT_RATE, len(pool.single_nouns))
        measured_plural = measure_pair(model, state, nouns, pool.plural_cue[template], plural.cue_token_id)
        informative = bool(float(measured_plural.dc[list(nouns.exposed_scorable)].abs().mean()) > 0.0)
        valid = bool(informative and positive >= required)
        frames_out[frame.frame_id] = {"template_id": template, "valid": valid, "cue_effect_positive": positive, "cue_effect_required": required,
                                      "plural_contrast_shift": float(measured_plural.dc[list(nouns.exposed_scorable)].mean()), "informative": informative,
                                      "p_c": frame.p_c, "p_t": frame.p_t, "cue_final": frame.p_t == frame.p_c}
        states[frame.frame_id] = locked_state(state)
        index_exposed, index_fresh = list(nouns.exposed_scorable), list(nouns.fresh)
        for token in confirmation.tokens:
            prediction = predict_pair(program, chain, weights, state, rows16, nouns, int(token["token_id"]), template_bases=bases, dT_direction=axis_T.direction.double())
            rows.append({"token": token["word"], "frame_id": frame.frame_id, "template": template,
                         "dc": [float(value) for value in prediction["dc"][index_exposed]],
                         "dc_fresh_nouns": [float(value) for value in prediction["dc"][index_fresh]],
                         "dc_no_l5_heads": [float(value) for value in prediction["dc_no_l5_heads"][index_exposed]],
                         "dc_template_base": [float(value) for value in prediction["dc_template_base"][index_exposed]],
                         "dT": float(prediction["dT"])})
        say(f"  {frame.frame_id}: {'valid' if valid else 'INVALID'} (cue effect {positive}/{required}, plural shift {frames_out[frame.frame_id]['plural_contrast_shift']:.3f}); p_c {frame.p_c}, p_t {frame.p_t}; {len(confirmation.tokens)} predictions")
    record = {"frames": frames_out, "states": states, "rows": rows, "commit": protocol_code_commit, "lock_sha256": lock["content_sha256"],
              "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision},
              "noun_keys": [nouns.nouns[index].lexical_key for index in nouns.exposed_scorable],
              "fresh_noun_keys": [nouns.nouns[index].lexical_key for index in nouns.fresh]}
    record["digest"] = stage_digest(rows, states, frames_out)
    return record


def stage_two(model: Any, pool: cs.Pool008, confirmation: Confirmation020, lock: Mapping[str, Any], lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_017: Mapping[str, Any],
              stage1: Mapping[str, Any], *, log: Any = None) -> dict[str, Any]:
    """Only after the barrier: the S2-TARGET prompts of the Y1 block and of the valid fresh frames' Y2 block."""
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model, layers=PROGRAM_LAYERS)
    programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in PROGRAM_LAYERS}
    chain = chain_from_locks(lock_011, lock_012, lock_017, lw, programs, pool)
    program = ReadoutProgram(lc.LayerWeights.from_model(model, layers=READOUT_LAYERS), {layer: programs[layer] for layer in READOUT_LAYERS},
                             weights.ln_final_w.double(), weights.ln_final_b.double(), float(weights.eps))
    axis_T = pm.SiteAxis("T", torch.zeros_like(torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64)), torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64), float(lock_011["sigma_T"]))
    nouns = NounSet.build(weights, pool.nouns, confirmation.nouns)
    exposed_index, fresh_index = list(nouns.exposed_scorable), list(nouns.fresh)
    exposed_keys = tuple(nouns.nouns[index].lexical_key for index in exposed_index)
    fresh_keys = tuple(nouns.nouns[index].lexical_key for index in fresh_index)
    locked_rows = {(row["token"], row["frame_id"]): row for row in lock["predictions"]["rows"]}
    stage1_rows = {(row["token"], row["frame_id"]): row for row in stage1["rows"]}
    identities: dict[str, float] = {}
    blocks: dict[str, dict[str, list]] = {name: {"cues": [], "frames": [], "templates": [], "measured": [], "predicted": [], "measured_fresh": [], "predicted_fresh": [], "no_l5": [], "base": [], "dT": []} for name in ("Y1", "Y2")}
    say("stage 2: the fresh cues in the exposed frames (Y1) and in the valid fresh frames (Y2)")
    for name, frames, rows_source in (("Y1", pool.frames, locked_rows), ("Y2", [frame for frame in confirmation.frames if stage1["frames"][frame.frame_id]["valid"]], stage1_rows)):
        for frame in frames:
            state = state_from_locked(lock["locked_states"][frame.frame_id] if name == "Y1" else stage1["states"][frame.frame_id], frame)
            c_ref = nouns.contrast_from_residual(program, state.h6)
            if len(state.c_ref) == len(nouns.nouns):  # a stage-1 state carries the captured contrasts: check them
                identities["reference_contrast"] = max(identities.get("reference_contrast", 0.0), float((c_ref - state.c_ref).abs().max()))
            for token in confirmation.tokens:
                measurement = measure_pair(model, state, nouns, token["word"], int(token["token_id"]), c_ref=c_ref)
                identities["readout"] = max(identities.get("readout", 0.0), abs_max(program, nouns, state, measurement))
                row = rows_source[(token["word"], frame.frame_id)]
                block = blocks[name]
                block["cues"].append(token["word"]); block["frames"].append(frame.frame_id); block["templates"].append(frame.template_id)
                block["measured"].append(measurement.dc[exposed_index]); block["predicted"].append(torch.tensor(row["dc"], dtype=torch.float64))
                block["measured_fresh"].append(measurement.dc[fresh_index]); block["predicted_fresh"].append(torch.tensor(row.get("dc_fresh_nouns", row["dc"][: len(fresh_index)]), dtype=torch.float64))
                block["no_l5"].append(torch.tensor(row["dc_no_l5_heads"], dtype=torch.float64)); block["base"].append(torch.tensor(row["dc_template_base"], dtype=torch.float64)); block["dT"].append(float(row["dT"]))
        say(f"  {name}: {len(blocks[name]['cues'])} pairs measured")
    enforce("readout identity", identities.get("readout", 0.0), READOUT_IDENTITY_TOLERANCE)
    if "reference_contrast" in identities:
        enforce("reference contrast from the locked residual", identities["reference_contrast"], READOUT_IDENTITY_TOLERANCE)
    tables = {}
    for name in ("Y1", "Y2"):
        block = blocks[name]
        common = (tuple(block["cues"]), tuple(block["frames"]), tuple(block["templates"]))
        tables[name] = {
            "exposed": ScoringTable(*common, torch.stack(block["measured"]), torch.stack(block["predicted"]), exposed_keys) if block["cues"] else None,
            "fresh": ScoringTable(*common, torch.stack(block["measured_fresh"]), torch.stack(block["predicted_fresh"]), fresh_keys) if block["cues"] else None,
            "no_l5": ScoringTable(*common, torch.stack(block["measured"]), torch.stack(block["no_l5"]), exposed_keys) if block["cues"] else None,
            "base": ScoringTable(*common, torch.stack(block["measured"]), torch.stack(block["base"]), exposed_keys) if block["cues"] else None,
            "dT": torch.tensor(block["dT"], dtype=torch.float64) if block["cues"] else None,
        }
    return {"tables": tables, "identities": {name: float(value) for name, value in identities.items()}, "noun_keys": list(exposed_keys), "fresh_noun_keys": list(fresh_keys)}


def score_confirmation(stage1: Mapping[str, Any], tables: Mapping[str, Mapping[str, Any]], lock: Mapping[str, Any]) -> dict[str, Any]:
    """Y1 and Y2 on the scorable exposed nouns, Y3 on the fresh nouns over the Y1 population, the joint statistic
    descriptive, and every comparator beside them."""
    valid = [frame_id for frame_id, entry in stage1["frames"].items() if entry["valid"]]
    coordinated = sum(1 for frame_id in valid if stage1["frames"][frame_id]["template_id"] == pm.COORDINATED_TEMPLATE)
    y1_table, y2_table = tables["Y1"]["exposed"], tables["Y2"]["exposed"]
    n_valid_exposed = len({frame_id for frame_id in y1_table.frames}) if y1_table is not None else 0
    y1 = score_y1(y1_table, n_valid_frames=n_valid_exposed) if y1_table is not None else {"label": OUTCOME_Y1[2], "precondition": {"ok": False}, "population": POPULATIONS["Y1"]}
    y2 = score_y2(y2_table, valid_frames=valid, valid_coordinated=coordinated) if y2_table is not None else {"label": OUTCOME_Y2[2], "precondition": {"ok": False}, "population": POPULATIONS["Y2"]}
    y3 = score_y3(tables["Y1"]["fresh"]) if tables["Y1"]["fresh"] is not None else {"label": OUTCOME_Y3[2], "precondition": {"ok": False}, "population": POPULATIONS["Y3"]}
    joint = pair_statistics(tables["Y2"]["fresh"]) if tables["Y2"]["fresh"] is not None else None
    comparators = {}
    for name in ("Y1", "Y2"):
        entry = tables[name]
        if entry["exposed"] is None:
            continue
        comparators[name] = {
            "no_l5_heads": {"standing": COMPARATOR_STANDING["no_l5_heads"], "flattened_r2": pair_statistics(entry["no_l5"])["flattened_r2"]},
            "template_base_mlps": {"standing": COMPARATOR_STANDING["template_base_mlps"], "flattened_r2": pair_statistics(entry["base"])["flattened_r2"]},
            "dT_only": {"standing": COMPARATOR_STANDING["dT_only"], "flattened_r2": dT_only_r2(entry["dT"], entry["exposed"], lock.get("dT_only_noun_vector"))},
            "rank1_nouns": {"standing": COMPARATOR_STANDING["rank1_nouns"], "fresh_noun_r2": rank1_fresh_r2(entry, lock)},
        }
    return {"Y1": y1, "Y2": y2, "Y3": y3, "joint_fresh_nouns": joint, "comparators": comparators,
            "valid_frames": sorted(valid), "n_valid_coordinated": coordinated, "outcome": outcome_label(y1, y2, y3)}


def dT_only_r2(dT: torch.Tensor | None, table: ScoringTable, noun_vector: Sequence[float] | None) -> float | None:
    """The exposed-disfavoured baseline on fresh data, with the noun vector the lock carries (never refitted here)."""
    if dT is None or noun_vector is None:
        return None
    predicted = dT.unsqueeze(1) * torch.tensor(noun_vector, dtype=torch.float64)
    if predicted.shape != table.measured.shape:
        return None
    return _r2(table.measured, predicted)


def rank1_fresh_r2(entry: Mapping[str, Any], lock: Mapping[str, Any]) -> float | None:
    """The descriptive rank-1 comparator: the locked exposed noun vector scoring the fresh nouns' table."""
    table = entry["fresh"]
    if table is None or lock.get("rank1") is None:
        return None
    return _r2(table.measured, table.predicted)


# ---------------------------------------------------------------------------
# The report.


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 020 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256: `{state.get('confirmation_020_sha256', '')}`",
             f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", "", "## Phases", ""]
    lines += [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration") or {}
    incidents = list(exploration.get("incidents", [])) + ([state["confirmation"]["incident"]] if isinstance(state.get("confirmation"), dict) and "incident" in state["confirmation"] else [])
    if incidents:
        lines += ["## Incidents", ""] + [f"- `{entry['phase']}` at commit `{entry.get('commit', '?')}` ({entry['at']}): {entry['message']}" for entry in incidents] + [""]
    if exploration.get("statistics"):
        statistics = exploration["statistics"]
        lines += ["## Tier A — exposed pool (calibration record only; the floors are frozen constants)", "",
                  f"- {statistics['n_pairs']} pairs × {statistics['n_nouns']} scorable exposed nouns (of {exploration['noun_population']['exposed']}; non-scorable {exploration['noun_population']['non_scorable']})",
                  f"- End-to-end: flattened R² {f(statistics['flattened_r2'], 4)}, token means {f(statistics['token_mean_r2'], 4)}, pair means {f(statistics['pair_mean_r2'], 4)}, pooled MAE {f(statistics['pooled_mae'], 3)} nats",
                  f"- Per template: " + ", ".join(f"{template} {f(entry['r2'], 3)}" for template, entry in sorted(statistics["per_template"].items())),
                  f"- Comparators: downstream ceiling (measured Δx₃) {f(exploration['comparators']['ceiling_measured_dx3'], 4)}; no-layer-5-head {f(exploration['comparators']['no_l5_heads'], 4)} (nested simplification); template-base MLPs {f(exploration['comparators']['template_base_mlps'], 4)} (operating-point comparator); ΔT-only {f(exploration['comparators']['dT_only']['r2'], 4)} (exposed-disfavoured baseline); rank-1 noun share {f(exploration['rank1']['rank1_share'], 4)} (descriptive)",
                  "- Identities (checks only): " + ", ".join(f"{name} {value:.1e}" for name, value in sorted(exploration["identities"].items())), ""]
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}`; predictions sha256 `{state['lock']['predictions_sha256']}`", ""]
    confirmation = state.get("confirmation")
    if isinstance(confirmation, dict) and "stage1" in confirmation:
        stage1 = confirmation["stage1"]
        lines += ["## Confirmation — stage 1 (the fresh frames' reference states; the prediction table digested before any fresh cue prompt)", "",
                  f"- Table rows {len(stage1['rows'])}; digest `{stage1['digest']}`; commit `{stage1['commit']}`", ""]
        for frame_id, entry in sorted(stage1["frames"].items()):
            lines.append(f"  - {frame_id}: {'valid' if entry['valid'] else 'INVALID'} (cue effect {entry['cue_effect_positive']}/{entry['cue_effect_required']}, plural contrast shift {f(entry['plural_contrast_shift'], 3)}; p_c {entry['p_c']}, p_t {entry['p_t']})")
        lines.append("")
    if isinstance(confirmation, dict) and "outcome" in confirmation:
        lines += [f"## Confirmation — stage 2 — `{confirmation['outcome']['label']}`", ""]
        for name in ("Y1", "Y2"):
            entry = confirmation[name]
            statistics = entry.get("statistics")
            lines.append(f"- {name} ({entry['population']}): **{entry['label']}**" + (f"; precondition {entry['precondition']}" if not entry["precondition"].get("ok") else ""))
            if statistics:
                lines.append(f"  - flattened R² {f(statistics['flattened_r2'], 4)}, token means {f(statistics['token_mean_r2'], 4)}, frame means {f(statistics['frame_mean_r2'], 4)}, pair means {f(statistics['pair_mean_r2'], 4)}, pooled MAE {f(statistics['pooled_mae'], 3)}")
                lines.append(f"  - splits: cue-final {f(statistics['cue_final_r2'], 3)}, coordinated {f(statistics['coordinated_r2'], 3)}; conditions {entry.get('conditions')}")
            if name in confirmation.get("comparators", {}):
                lines.append(f"  - comparators: " + ", ".join(f"{key} {f(value.get('flattened_r2', value.get('fresh_noun_r2')), 4)} ({value['standing']})" for key, value in sorted(confirmation["comparators"][name].items())))
        y3 = confirmation["Y3"]
        lines.append(f"- Y3 (the 24 fresh nouns on the Y1 population): **{y3['label']}**" + (f"; median R² {f(y3.get('median_r2'), 3)}, minimum {f(y3.get('min_r2'), 3)} (reported, not gated); conditions {y3.get('conditions')}" if y3.get("conditions") else ""))
        if confirmation.get("joint_fresh_nouns"):
            joint = confirmation["joint_fresh_nouns"]
            lines.append(f"- Joint diagnostic (fresh cues × fresh frames × fresh nouns; no threshold): flattened R² {f(joint['flattened_r2'], 4)}, pooled MAE {f(joint['pooled_mae'], 3)}")
        lines += ["", "Interpretation limit (frozen): Y2 is conditional on each fresh frame's stage-1 reference state; Experiment 020 does not predict the frame state from text.", ""]
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
