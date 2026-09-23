"""Experiment 022: prospective localization of the upstream ``Δx3`` error.

Implements design revision 3 (``b0c7382``) through implementation plan revision 2 (``fb26a23``). The committed
program's upstream prediction of ``Δx3`` (the Experiment 017 chain at ``LEVEL0``, fed the Experiment 005/011 E-patch
``ΔE``) is compared with the exact weight-only layers-0–2 program through an exact five-factor Shapley attribution of
the Level-0-to-ceiling ``Δc`` gap. The factors are, as bits of a coalition mask:

    R   = 1   layers 1–2: the committed reduced program (off) or the exact program (on)
    emb = 2   the embedding's own change at ``p_c``
    Bv  = 4   block 0's attention value term at ``p_c`` (the reference row carrying the cue key's changed value)
    Bp  = 8   block 0's attention pattern term at ``p_c`` (the changed row carrying the new values)
    T   = 16  block 0's attention change at ``p_t`` (coordinated frames; a structural null player in cue-final frames)

This module *calls* the frozen modules (``readout_decompilation``, ``readout_calibration``, ``head_pattern``,
``frame_channels``, ``attention_patterns``, ``layer_correction``, ``plural_mechanism`` …) and edits none of them; their
git blobs are pinned below and checked by every phase.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy
import torch

from neural_decompiler import attention_patterns as atp
from neural_decompiler import encoding_read as er
from neural_decompiler import frame_channels as fch
from neural_decompiler import head_pattern as hp
from neural_decompiler import head_transport as ht
from neural_decompiler import layer_correction as lc
from neural_decompiler import models as models_module
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler.behavior import validate_json_safe

PhaseError = pm.PhaseError
IncidentError = pm.IncidentError

# ---------------------------------------------------------------------------
# Paths, design, frozen inputs.

EXPERIMENT = "022"
EXPERIMENT_DIR = "experiments/022-upstream-error-localization"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
CALIBRATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/calibration-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREREGISTRATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration.md"
Y1_TABLE_RELATIVE_PATH = f"{EXPERIMENT_DIR}/locked-y1-table.f64"
Y1_TABLE_INDEX_RELATIVE_PATH = f"{EXPERIMENT_DIR}/locked-y1-table.json"
DESIGN = {"path": "docs/superpowers/specs/2026-09-23-experiment-022-upstream-error-localization-design.md", "revision": 3, "commit": "b0c7382"}
PLAN = {"path": "docs/superpowers/plans/2026-09-23-experiment-022-upstream-error-localization-plan.md", "revision": 2, "commit": "fb26a23"}

# The frozen modules, by git blob (computed without git: sha1(b"blob <len>\0" + bytes)).
FROZEN_BLOBS = {
    "readout_decompilation.py": "caa73b40192f4c910dc63371bd19db75a3258339",
    "readout_calibration.py": "9107da975128d9b0383f3b346695675104e4cde9",
    "head_pattern.py": "386682fe0a47d093dfbee2507afc1df61811ab15",
    "frame_channels.py": "c95d6fb4c98b8069c26ec46e9d8f85eb927bb9e8",
    "attention_patterns.py": "3f5acd65397bd11723ba6fc2ec4253f1d5c81943",
    "layer_correction.py": "04df0cc9a20093cc48ee5ef62da7f206bf1a3186",
    "plural_mechanism.py": "d39da8a8d9931005d411258bcddbb7f9beed35e4",
    "encoding_read.py": "cab99c942de970332e726a5626b584242d4cf600",
    "head_transport.py": "936093a4e82c83e299a65a7c85529a7226f80c95",
    "models.py": "b1c6f03379af7e0918d0d1a6460a264651603fb2",
}
_FROZEN_MODULES = {"readout_decompilation.py": rd, "readout_calibration.py": rc, "head_pattern.py": hp, "frame_channels.py": fch, "attention_patterns.py": atp,
                   "layer_correction.py": lc, "plural_mechanism.py": pm, "encoding_read.py": er, "head_transport.py": ht, "models.py": models_module}

# Experiment 021's committed calibration record and local artifacts (read for the R1 gate and, after report only, for
# the exploratory replication); every digest below is frozen.
CALIBRATION_021_RELATIVE_PATH = rc.CALIBRATION_RELATIVE_PATH
EXPOSED_TABLE_021_RELATIVE_PATH = "outputs/experiment-021/exposed-table.pt"
RESULTS_021_RELATIVE_PATH = "outputs/experiment-021/results.json"
STAGE2_021_RELATIVE_PATH = "outputs/experiment-021/stage2-tables.pt"
EXTRACT_021_RELATIVE_PATH = "experiments/021-corrected-readout-confirmation/evidence/confirmation-record-2026-09-23.json"
INHERITED_021 = {
    "calibration_file_sha256": "240e0345d29b2d8fd1202cf810eff84da96a4995959729d7a0ab3d855e892fd1",
    "calibration_content_sha256": "f939a84df3ff3dcf55f95877c951059c5157857d72d6be8ef036e10f7923044d",
    "exposed_measured_sha256": "799ec908f2a651d0ba4af182f6bf5cf839cb600aba22a0546b37e9affa0ba3fe",
    "results_file_sha256": "8caf2a026cdb0b8f831a82abc7b90393ed33ec82399c56ec1eb70dbecfd3b73e",
    "results_state_sha256": "80b9b654e256106ccf19bef3cbd58f641567f6e8009916cb6cea0085880cbb85",
    "extract_content_sha256": "678714b0ee1257b0a87438ca15b0ae5ec46c1222ef6aae5f7d78335eb4960793",
}

# ---------------------------------------------------------------------------
# The game.

FACTORS = ("R", "emb", "Bv", "Bp", "T")
BIT = {"R": 1, "emb": 2, "Bv": 4, "Bp": 8, "T": 16}
N_FACTORS = len(FACTORS)
N_MASKS = 1 << N_FACTORS  # 32 coalitions
FULL_MASK = N_MASKS - 1  # 31
CUE_FINAL_MASKS = tuple(mask for mask in range(N_MASKS) if not mask & BIT["T"])  # 0…15: T is structurally absent at p_t = p_c
CUE_FINAL_FULL_MASK = FULL_MASK & ~BIT["T"]  # 15
# The exact Shapley weights w(k) = k!(n−k−1)!/n! for n = 5: 1/5, 1/20, 1/30, 1/20, 1/5.
SHAPLEY_WEIGHTS = tuple(Fraction(math.factorial(k) * math.factorial(N_FACTORS - k - 1), math.factorial(N_FACTORS)) for k in range(N_FACTORS))
GROUPS = ("cue_final", "coordinated")
POPULATIONS = ("Y1", "Y2")
CLAIMS = ("C1", "C2", "C3", "C4")
CLAIM_GROUP = {"C1": "cue_final", "C2": "cue_final", "C3": "coordinated", "C4": "coordinated"}
CLAIM_WORDING = {
    "C1": "block-0 attention dominates the cue-final gap",
    "C2": "the layer-1–2 reductions contribute little to the cue-final gap",
    "C3": "the coordinated gap is split between layer 0 and the reductions",
    "C4": "block-0 cue→target attention contributes positively to the coordinated gap",
}
CLAIM_STATISTIC = {"C1": "share(Bv) + share(Bp), cue-final", "C2": "share(R), cue-final", "C3": "share(R), coordinated", "C4": "share(T), coordinated"}
# Meaning guards (frozen by the design): they can only make a PASS harder.
GUARD_C1_MIN = 0.50
GUARD_C2_MAX = 0.10
GUARD_C3_RANGE = (0.10, 0.90)
GUARD_C4_EXCLUSIVE_MIN = 0.0
GAP_MIN = 0.02  # the interpretability rule, on the full aggregate: SST > 0 and G ≥ 0.02
# The four condition results, in precedence order.
RESULTS = ("NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS")

# ---------------------------------------------------------------------------
# Calibration constants and tolerances.

B = 10_000
CROSS_CHECK_DRAWS = 16
SLOTS = {"cue": 6, "frame": 6}
DRAW_TAG = "022|primary"
TOLERANCES = {
    "I1": 1e-4,  # block-0 decomposition against the measured Δx1, max abs
    "I2": 1e-12,  # V + P = ΔA0 (algebra)
    "I3": 1e-4,  # max over applicable positions of ‖pred − meas‖₂ / max(‖meas‖₂, 1e-12)
    "I4": 1e-3,  # full composition Δĉ against the ceiling Δĉ, max abs (nats)
    "I5": 0.0,  # empty composition against the committed Level 0, exactly
    "I6": 1e-12,  # Shapley efficiency, relative to max(1, |G|)
    "R1": 1e-9,  # re-measured exposed Δc against Experiment 021's table, max abs
    "kernel": 1e-10,  # kernel against the direct recomputation, |k − d| / max(1, |d|)
}
I3_FLOOR = 1e-12

# ---------------------------------------------------------------------------
# Fresh units: the ordered candidate lists and the structural rules (verbatim from the design).

CUE_QUOTA = 6
FRAME_QUOTA = 6
CUE_CANDIDATES = {
    "determiner-like": ("alternate", "random", "standard", "regular", "normal", "common", "general", "subsequent", "given", "chosen", "selected"),
    "quantity": ("bulk", "spare", "dense", "lengthy", "lots", "loads", "tons", "piles", "masses", "stacks", "gross", "net"),
    "possessive-or-pronoun": ("thou", "yourself", "themselves", "ones", "others", "naught", "whatsoever"),
    "adjective": ("square", "wild", "brave", "calm", "eager", "fierce", "humble", "proud", "shy", "lazy", "busy", "sturdy", "fragile", "polished", "shiny", "dusty",
                  "hollow", "noble"),
}
FRAME_CANDIDATES = {
    "cardinal": ("The pantry keeps {cue}", "The studio makes {cue}", "The market trades {cue}", "The factory builds {cue}", "The kitchen bakes {cue}",
                 "The quarry yields {cue}", "The dairy produces {cue}", "The shelter feeds {cue}", "The boutique sells {cue}", "The hangar shelters {cue}"),
    "quantifier": ("The census counts {cue}", "The survey names {cue}", "The bulletin lists {cue}", "The directory shows {cue}", "The agenda names {cue}",
                   "The appendix cites {cue}", "The timetable shows {cue}", "The glossary defines {cue}"),
    "coordinated-adjective": ("Hugo and Nina sanded {cue} flat", "Ines and Karl trimmed {cue} short", "Otto and Vera washed {cue} clean",
                              "Ivo and Greta sorted {cue} neat", "Mats and Alma cooled {cue} fast", "Nora and Paul rinsed {cue} dry",
                              "Ella and Tomas stacked {cue} low", "Anna and Erik glazed {cue} smooth", "Leo and Maja folded {cue} flat"),
}
P_C_RANGE = {"cardinal": (3, 5), "quantifier": (3, 5), "coordinated-adjective": (4, 7)}  # the exposed range of each template
FRAME_ID_TAG = "022"
CUE_CLASSES = tuple(CUE_CANDIDATES)
TEMPLATES = tuple(pm.TEMPLATE_ORDER)

TABLE_FORMAT = "raw IEEE-754 float64, little-endian, C order; blocks back to back in the order listed"
TABLE_DTYPE = "<f8"


# ---------------------------------------------------------------------------
# Frozen inputs.


def module_blobs() -> dict[str, str]:
    return {name: rc.program_blob_sha1(Path(module.__file__)) for name, module in _FROZEN_MODULES.items()}


def assert_frozen_blobs() -> dict[str, str]:
    """Every module the program depends on is byte for byte the frozen one."""
    actual = module_blobs()
    differing = sorted(name for name, blob in FROZEN_BLOBS.items() if actual.get(name) != blob)
    if differing:
        raise PhaseError(f"frozen modules changed: {differing}; Experiment 022 runs the committed program only")
    return actual


# ---------------------------------------------------------------------------
# Coalitions.


def popcount(mask: int) -> int:
    return bin(int(mask)).count("1")


def mask_of(names: Sequence[str]) -> int:
    mask = 0
    for name in names:
        mask |= BIT[name]
    return mask


def names_of(mask: int) -> tuple[str, ...]:
    return tuple(name for name in FACTORS if mask & BIT[name])


def canonical_mask(mask: int, cue_final: bool) -> int:
    """In a cue-final frame ``p_t = p_c``: block 0's change at the target position *is* its change at the cue position,
    already split into Bv and Bp, so T contributes no input and the mask is canonicalized to its T-free form. The
    stored value of ``m | T`` is then the same number as that of ``m``, which makes T an exact null player."""
    return int(mask) & ~BIT["T"] if cue_final else int(mask)


# ---------------------------------------------------------------------------
# The one Shapley kernel (calibration, confirm and the replication all call it).


def group_sums(sse_cells: torch.Tensor, count_cells: torch.Tensor, mean_cells: torch.Tensor, m2_cells: torch.Tensor,
               index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate a multiset of cells (pairs, or already-aggregated groups of pairs) per row of ``index``.

    ``sse_cells`` is ``[P, 32]``, the moments ``[P]``; ``index`` is ``[D, K]`` (a unit repeated ``k`` times counts ``k``
    times). SSE sums are additive; the moments combine by the pooled two-pass identity (021's ``_combine``), never
    the one-pass ``Σy² − (Σy)²/n``. Returns ``(sse [D, 32], count [D], mean [D], m2 [D])``."""
    sse = sse_cells[index].sum(dim=1)
    count, mean, m2 = rc._combine(count_cells[index], mean_cells[index], m2_cells[index])
    return sse, count, mean, m2


def group_statistics(sse: torch.Tensor, sst: torch.Tensor) -> dict[str, torch.Tensor]:
    """``v(m) = 1 − SSE(m)/SST``, the gap ``G = v(31) − v(0)``, the exact Shapley values by the subset formula in a
    fixed coalition order, the shares ``φ/G`` and the interpretability flag (``SST > 0`` and ``G ≥ 0.02``).

    ``sse`` is ``[D, 32]``, ``sst`` ``[D]``, both float64. A non-interpretable row's shares are NaN."""
    sse = sse.double()
    sst = sst.double()
    positive = sst > 0.0
    denominator = torch.where(positive, sst, torch.ones_like(sst))
    v = 1.0 - sse / denominator.unsqueeze(-1)
    gap = v[:, FULL_MASK] - v[:, 0]
    phi = torch.zeros(sse.shape[0], N_FACTORS, dtype=torch.float64)
    for index, name in enumerate(FACTORS):
        bit = BIT[name]
        for mask in range(N_MASKS):
            if mask & bit:
                continue
            phi[:, index] += float(SHAPLEY_WEIGHTS[popcount(mask)]) * (v[:, mask | bit] - v[:, mask])
    interpretable = positive & torch.isfinite(gap) & (gap >= GAP_MIN)
    safe_gap = torch.where(interpretable, gap, torch.ones_like(gap))
    shares = torch.where(interpretable.unsqueeze(-1), phi / safe_gap.unsqueeze(-1), torch.full_like(phi, float("nan")))
    return {"v": v, "gap": gap, "phi": phi, "shares": shares, "interpretable": interpretable, "sst_positive": positive,
            "efficiency": (phi.sum(dim=-1) - gap).abs() / torch.clamp(gap.abs(), min=1.0)}


def claim_statistics(cue_final: Mapping[str, torch.Tensor], coordinated: Mapping[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """``s1 = σ_Bv + σ_Bp`` and ``s2 = σ_R`` on the cue-final group; ``s3 = σ_R`` and ``s4 = σ_T`` on the coordinated
    group. Each carries its own group's interpretability flag."""
    at = {name: position for position, name in enumerate(FACTORS)}
    return {
        "C1": {"value": cue_final["shares"][:, at["Bv"]] + cue_final["shares"][:, at["Bp"]], "interpretable": cue_final["interpretable"], "gap": cue_final["gap"]},
        "C2": {"value": cue_final["shares"][:, at["R"]], "interpretable": cue_final["interpretable"], "gap": cue_final["gap"]},
        "C3": {"value": coordinated["shares"][:, at["R"]], "interpretable": coordinated["interpretable"], "gap": coordinated["gap"]},
        "C4": {"value": coordinated["shares"][:, at["T"]], "interpretable": coordinated["interpretable"], "gap": coordinated["gap"]},
    }


def direct_group_statistics(measured: torch.Tensor, predicted: torch.Tensor) -> dict[str, Any]:
    """The independent check (implementation-only): the flattened two-pass ``R²`` of every coalition from the
    materialized pairs, and the Shapley values by the **permutation** formula (the mean marginal contribution over all
    120 orders). ``measured`` is ``[K, N]`` (a repeated pair appears repeatedly), ``predicted`` ``[K, 32, N]``."""
    y = measured.double().reshape(-1)
    centred = y - y.mean()
    sst = float((centred * centred).sum())
    values = []
    for mask in range(N_MASKS):
        residual = y - predicted[:, mask, :].double().reshape(-1)
        values.append(None if sst <= 0.0 else 1.0 - float((residual * residual).sum()) / sst)
    if sst <= 0.0:
        return {"v": values, "gap": None, "phi": None, "shares": None, "interpretable": False, "sst": sst}
    phi = [0.0] * N_FACTORS
    orders = list(itertools.permutations(range(N_FACTORS)))
    for order in orders:
        mask = 0
        for position in order:
            bit = BIT[FACTORS[position]]
            phi[position] += values[mask | bit] - values[mask]
            mask |= bit
    phi = [value / len(orders) for value in phi]
    gap = values[FULL_MASK] - values[0]
    interpretable = math.isfinite(gap) and gap >= GAP_MIN
    return {"v": values, "gap": gap, "phi": phi, "shares": [value / gap for value in phi] if interpretable else None, "interpretable": interpretable, "sst": sst}


def agreement(kernel: float | None, direct: float | None) -> float:
    """``|k − d| / max(1, |d|)``; an undefined value on one side only is ∞, on both sides 0."""
    if kernel is None or direct is None or not math.isfinite(kernel) or not math.isfinite(direct):
        return 0.0 if (kernel is None or not math.isfinite(kernel)) and (direct is None or not math.isfinite(direct)) else math.inf
    return abs(kernel - direct) / max(1.0, abs(direct))


# ---------------------------------------------------------------------------
# Envelopes: exact 1-based order statistics of the ascending calibration values.


def lower_rank(draws: int) -> int:
    """⌈0.025·B⌉ in integer arithmetic: 250 at B = 10,000 (element [249])."""
    return (25 * int(draws) + 999) // 1000


def upper_rank(draws: int) -> int:
    """The ⌈0.025·B⌉-th largest: 9751 at B = 10,000 (element [9750]). Never the lower rank."""
    return int(draws) - lower_rank(draws) + 1


def c3_low_rank(draws: int) -> int:
    """⌈0.0125·B⌉: 125 at B = 10,000 (element [124])."""
    return (125 * int(draws) + 9999) // 10000


def c3_high_rank(draws: int) -> int:
    """9876 at B = 10,000 (element [9875])."""
    return int(draws) - c3_low_rank(draws) + 1


def order_statistic(values: torch.Tensor, defined: torch.Tensor, rank: int, *, undefined_at: float) -> float:
    """``v₍rank₎`` of the ascending array, with undefined values placed at ``undefined_at`` (−∞ for a lower bound,
    +∞ for an upper bound, so they always count against the envelope)."""
    view = torch.where(defined, values.double(), torch.full_like(values.double(), undefined_at))
    return float(torch.sort(view).values[int(rank) - 1])


def undefined_threshold(claim: str, draws: int) -> int:
    """The calibration stop (design revision 3): at this many undefined values the claim's order statistic is infinite."""
    return c3_low_rank(draws) if claim == "C3" else lower_rank(draws)


def claim_envelope(claim: str, values: torch.Tensor, defined: torch.Tensor) -> dict[str, Any]:
    draws = int(values.shape[0])
    if claim in ("C1", "C4"):
        rank = lower_rank(draws)
        return {"kind": "lower", "rank": rank, "element": rank - 1, "bound": order_statistic(values, defined, rank, undefined_at=-math.inf)}
    if claim == "C2":
        rank = upper_rank(draws)
        return {"kind": "upper", "rank": rank, "element": rank - 1, "bound": order_statistic(values, defined, rank, undefined_at=math.inf)}
    if claim == "C3":
        low, high = c3_low_rank(draws), c3_high_rank(draws)
        return {"kind": "two-sided", "ranks": [low, high], "elements": [low - 1, high - 1],
                "low": order_statistic(values, defined, low, undefined_at=-math.inf), "high": order_statistic(values, defined, high, undefined_at=math.inf)}
    raise ValueError(f"unknown claim {claim}")


def defined_median(values: torch.Tensor, defined: torch.Tensor) -> float | None:
    """The median of the defined values (the mean of the two middle ones for an even count)."""
    kept = torch.sort(values.double()[defined]).values
    n = int(kept.shape[0])
    if n == 0:
        return None
    return float(kept[n // 2]) if n % 2 else 0.5 * float(kept[n // 2 - 1] + kept[n // 2])


def direction_check(claim: str, envelope: Mapping[str, Any], values: torch.Tensor, defined: torch.Tensor) -> dict[str, Any]:
    """A lower bound lies at or below the median of the defined draws, an upper bound at or above it, and C3's band
    contains it. A violation means a tail was reversed: an implementation incident."""
    median = defined_median(values, defined)
    if median is None:
        ok = False
    elif envelope["kind"] == "lower":
        ok = envelope["bound"] <= median
    elif envelope["kind"] == "upper":
        ok = envelope["bound"] >= median
    else:
        ok = envelope["low"] <= median <= envelope["high"]
    return {"ok": bool(ok), "median": median}


# ---------------------------------------------------------------------------
# The four-way condition result (the only function that decides a result).


def meaning_guard(claim: str, value: float) -> bool:
    if claim == "C1":
        return value >= GUARD_C1_MIN
    if claim == "C2":
        return value <= GUARD_C2_MAX
    if claim == "C3":
        return GUARD_C3_RANGE[0] <= value <= GUARD_C3_RANGE[1]
    if claim == "C4":
        return value > GUARD_C4_EXCLUSIVE_MIN
    raise ValueError(f"unknown claim {claim}")


def within_envelope(claim: str, value: float, envelope: Mapping[str, Any]) -> bool:
    if claim in ("C1", "C4"):
        return value >= envelope["bound"]
    if claim == "C2":
        return value <= envelope["bound"]
    if claim == "C3":
        return envelope["low"] <= value <= envelope["high"]
    raise ValueError(f"unknown claim {claim}")


def classify(claim: str, value: float | None, interpretable: bool, envelope: Mapping[str, Any]) -> str:
    """``NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS``, in that precedence."""
    if not interpretable:
        return "NOT_INTERPRETABLE"
    if value is None or not math.isfinite(float(value)):
        raise IncidentError(f"{claim}: a non-finite share where the gap rule holds")
    value = float(value)
    if not meaning_guard(claim, value):
        return "GUARD_FAILURE"
    if not within_envelope(claim, value, envelope):
        return "ENVELOPE_ONLY_FAILURE"
    return "PASS"


def guard_bound(claim: str, envelope: Mapping[str, Any]) -> bool:
    """True when the meaning guard is stricter than the envelope somewhere (recorded; the guard is never relaxed)."""
    if claim == "C1":
        return envelope["bound"] < GUARD_C1_MIN
    if claim == "C2":
        return envelope["bound"] > GUARD_C2_MAX
    if claim == "C3":
        return envelope["low"] < GUARD_C3_RANGE[0] or envelope["high"] > GUARD_C3_RANGE[1]
    if claim == "C4":
        return envelope["bound"] <= GUARD_C4_EXCLUSIVE_MIN
    raise ValueError(f"unknown claim {claim}")


# ---------------------------------------------------------------------------
# I3 and the report's CDF percentile.


def i3_error(predicted: Mapping[int, torch.Tensor], measured: Mapping[int, torch.Tensor]) -> float:
    """``max_p ‖Δx̂3(p) − Δx3(p)‖₂ / max(‖Δx3(p)‖₂, 1e-12)`` over the pair's applicable changed positions."""
    if set(predicted) != set(measured):
        raise IncidentError(f"I3 compares different positions: {sorted(predicted)} against {sorted(measured)}")
    return max(float((predicted[p].double() - measured[p].double()).norm()) / max(float(measured[p].double().norm()), I3_FLOOR) for p in sorted(measured))


def cdf_percentile(values: torch.Tensor, defined: torch.Tensor, fresh: float | None) -> dict[str, Any]:
    """The fraction of *defined* calibration draws ``≤`` the fresh value, and the undefined count. Descriptive only;
    for the upper-bound C2 a high percentile lies toward the unfavorable upper tail."""
    n = int(defined.sum())
    undefined = int((~defined).sum())
    if fresh is None or not math.isfinite(float(fresh)) or n == 0:
        return {"cdf_percentile": None, "defined": n, "undefined": undefined}
    below = int(((values.double() <= float(fresh)) & defined).sum())
    return {"cdf_percentile": below / n, "defined": n, "undefined": undefined}


# ---------------------------------------------------------------------------
# The committed table byte format.


def table_bytes(blocks: Sequence[tuple[str, torch.Tensor]]) -> bytes:
    return b"".join(numpy.ascontiguousarray(block.detach().cpu().double().numpy(), dtype=TABLE_DTYPE).tobytes() for _, block in blocks)


def table_index(blocks: Sequence[tuple[str, torch.Tensor]], data: bytes, meta: Mapping[str, Any]) -> dict[str, Any]:
    offset, entries = 0, []
    for name, block in blocks:
        nbytes = int(block.numel()) * 8
        entries.append({"name": name, "shape": list(block.shape), "offset_bytes": offset, "nbytes": nbytes})
        offset += nbytes
    return {"format": TABLE_FORMAT, "dtype": TABLE_DTYPE, "blocks": entries, "total_bytes": offset, "file_sha256": hashlib.sha256(data).hexdigest(), **dict(meta)}


def write_table(data_path: Path, index_path: Path, blocks: Sequence[tuple[str, torch.Tensor]], meta: Mapping[str, Any]) -> dict[str, Any]:
    data = table_bytes(blocks)
    index = table_index(blocks, data, meta)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(data)
    index_path.write_text(pm.canonical_json(index) + "\n", encoding="utf-8")
    return index


def read_table(data_path: Path, index: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """The blocks, after the file's sha256 is verified against the index (a mismatch is an incident)."""
    data = data_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != index["file_sha256"] or len(data) != index["total_bytes"]:
        raise IncidentError(f"{data_path.name}: the table bytes do not match their index digest")
    out = {}
    for entry in index["blocks"]:
        count = int(numpy.prod(entry["shape"])) if entry["shape"] else 1
        array = numpy.frombuffer(data, dtype=TABLE_DTYPE, count=count, offset=int(entry["offset_bytes"])).reshape(entry["shape"])
        out[entry["name"]] = torch.from_numpy(array.astype(numpy.float64, copy=True))
    return out


# ---------------------------------------------------------------------------
# SHA-indexed draws (no random generator).


def slot_index(b: int, stratum: str, slot: int, n: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{DRAW_TAG}|{b}|{stratum}|{slot}".encode("utf-8")).digest()[:8], "big") % int(n)


def draw_indices(sizes: Mapping[str, int], slots: Mapping[str, int], draws: int) -> dict[str, torch.Tensor]:
    """Per stratum ``[draws, slots]`` indices into that stratum's frozen unit order (cues by token id, frames by
    ``frame_id``)."""
    return {stratum: torch.tensor([[slot_index(b, stratum, i, n) for i in range(slots[stratum])] for b in range(draws)], dtype=torch.int64)
            for stratum, n in sizes.items()}


# ---------------------------------------------------------------------------
# The frozen input chain (Experiment 020's, exactly as Experiment 021's runner loads it; no model).

LOCK_011_REQUIRED_KEYS = ("axes_vectors", "read_weight", "sigma_T", "denominators", "confirmation_011_sha256", "content_sha256")
LOCK_012_REQUIRED_KEYS = ("axes_vectors", "read_weight", "sigma_T", "base_states", "defined_templates", "confirmation_012_sha256", "lock_011_sha256", "content_sha256")
LOCK_017_REQUIRED_KEYS = ("locked_states", "bases_3", "confirmation_017_sha256", "lock_016_sha256", "content_sha256")


def load_lock(path: Path, experiment: str) -> dict[str, Any]:
    import json

    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("experiment") != experiment or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError(f"the Experiment {experiment} lock's content digest does not verify")
    return lock


@dataclass
class FrozenInputs:
    """Everything the program reads that no phase of 022 may change: the pool, the locks, Experiment 020's
    confirmation set (its manifest keys are forbidden before 022's confirm, never executed), its closure (the
    locked exposed states and the exposed ledger), the confirmation objects of 006–020 (for the freeze's exclusion)
    and every digest."""

    pool: Any
    lock_011: Mapping[str, Any]
    lock_012: Mapping[str, Any]
    lock_017: Mapping[str, Any]
    confirmation_020: Any
    closure: Mapping[str, Any]
    digests: dict[str, str]
    confirmations: dict[str, Any]  # "006", "009", "011", …, "020" -> the frozen loader's object
    manifest: Any
    extension: Any


def load_frozen_inputs(root: Path, *, lock_011_loader: Callable[[Path], Mapping[str, Any]] | None = None,
                       lock_012_loader: Callable[[Path], Mapping[str, Any]] | None = None, lock_017_loader: Callable[[Path], Mapping[str, Any]] | None = None,
                       tracked: Callable[[Path], bool] = lambda path: True) -> FrozenInputs:
    """Experiment 020's input chain, unchanged (the same loaders, checks and order as Experiment 021's runner)."""
    from neural_decompiler import attention_paths as ap
    from neural_decompiler import block_concentration as bc
    from neural_decompiler import block_routing as br
    from neural_decompiler import cue_decompilation as cd
    from neural_decompiler import neuron_feature as nf
    from neural_decompiler import supervised_subspace as ss

    lock_011_loader = lock_011_loader or (lambda path: load_lock(path, "011"))
    lock_012_loader = lock_012_loader or (lambda path: load_lock(path, "012"))
    lock_017_loader = lock_017_loader or (lambda path: load_lock(path, "017"))
    manifest, manifest_sha256, extension = pm.load_inputs(root)
    c006 = cd.load_confirmation(root / cd.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension)
    if c006.content_sha256 != ss.INHERITED_CONFIRMATION_SHA256:
        raise PhaseError("the Experiment 006 confirmation set digest is not the frozen constant")
    c009 = ht.load_confirmation(root / ht.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, c006)
    c011 = er.load_confirmation(root / er.CONFIRMATION_RELATIVE_PATH, manifest, manifest_sha256, extension, c006, c009)
    for relative in (cd.CONFIRMATION_RELATIVE_PATH, ht.CONFIRMATION_RELATIVE_PATH, er.CONFIRMATION_RELATIVE_PATH, rd.EXPERIMENT_011_LOCK_PATH, lc.CONFIRMATION_RELATIVE_PATH,
                     rd.EXPERIMENT_012_LOCK_PATH, hp.EXPERIMENT_013_CONFIRMATION_PATH, hp.EXPERIMENT_014_CONFIRMATION_PATH, hp.EXPERIMENT_015_CONFIRMATION_PATH,
                     hp.EXPERIMENT_016_CONFIRMATION_PATH, bc.EXPERIMENT_017_CONFIRMATION_PATH, rd.EXPERIMENT_017_LOCK_PATH, br.EXPERIMENT_018_CONFIRMATION_PATH,
                     rd.EXPERIMENT_019_CONFIRMATION_PATH, rd.CONFIRMATION_RELATIVE_PATH, rc.CLOSURE_020_RELATIVE_PATH, rc.EXTRACT_020_RELATIVE_PATH):
        if not (root / relative).exists() or not tracked(root / relative):
            raise PhaseError(f"{relative} must exist and be tracked and committed")
    digests = {"manifest": manifest_sha256, "extension": extension.content_sha256, "confirmation_006": c006.content_sha256, "confirmation_009": c009.content_sha256,
               "confirmation_011": c011.content_sha256}
    lock_011 = lock_011_loader(root / rd.EXPERIMENT_011_LOCK_PATH)
    if any(key not in lock_011 for key in LOCK_011_REQUIRED_KEYS) or lock_011["confirmation_011_sha256"] != c011.content_sha256:
        raise PhaseError("the Experiment 011 lock does not carry the locked axes and read weight for the frozen 011 confirmation set")
    digests["lock_011"] = lock_011["content_sha256"]
    pool_012 = lc.build_pool_012(manifest, extension, c006, c009, c011)
    c012 = lc.load_confirmation(root / lc.CONFIRMATION_RELATIVE_PATH, pool_012, digests)
    digests["confirmation_012"] = c012.content_sha256
    lock_012 = lock_012_loader(root / rd.EXPERIMENT_012_LOCK_PATH)
    if any(key not in lock_012 for key in LOCK_012_REQUIRED_KEYS) or lock_012["confirmation_012_sha256"] != c012.content_sha256 or lock_012["lock_011_sha256"] != lock_011["content_sha256"]:
        raise PhaseError("the Experiment 012 lock does not carry the locked bases for the frozen 012 confirmation set and the 011 lock")
    digests["lock_012"] = lock_012["content_sha256"]
    c013 = ap.load_confirmation(root / hp.EXPERIMENT_013_CONFIRMATION_PATH, ap.build_pool_013(manifest, extension, c006, c009, c011, c012), digests)
    digests["confirmation_013"] = c013.content_sha256
    c014 = nf.load_confirmation(root / hp.EXPERIMENT_014_CONFIRMATION_PATH, nf.build_pool_014(manifest, extension, c006, c009, c011, c012, c013), digests)
    digests["confirmation_014"] = c014.content_sha256
    c015 = atp.load_confirmation(root / hp.EXPERIMENT_015_CONFIRMATION_PATH, atp.build_pool_015(manifest, extension, c006, c009, c011, c012, c013, c014), digests)
    digests["confirmation_015"] = c015.content_sha256
    c016 = fch.load_confirmation(root / hp.EXPERIMENT_016_CONFIRMATION_PATH, fch.build_pool_016(manifest, extension, c006, c009, c011, c012, c013, c014, c015), digests)
    digests["confirmation_016"] = c016.content_sha256
    pool_017 = hp.build_pool_017(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016)
    c017 = hp.load_confirmation(root / bc.EXPERIMENT_017_CONFIRMATION_PATH, pool_017, digests)
    digests["confirmation_017"] = c017.content_sha256
    lock_017 = lock_017_loader(root / rd.EXPERIMENT_017_LOCK_PATH)
    if any(key not in lock_017 for key in LOCK_017_REQUIRED_KEYS) or lock_017["confirmation_017_sha256"] != c017.content_sha256:
        raise PhaseError("the Experiment 017 lock does not carry the locked layer-3 bases for the frozen 017 confirmation set")
    digests["lock_017"] = lock_017["content_sha256"]
    rc.assert_lock_digests(digests)
    pool_018 = bc.build_pool_018(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017)
    c018 = bc.load_confirmation(root / br.EXPERIMENT_018_CONFIRMATION_PATH, pool_018, digests)
    digests["confirmation_018"] = c018.content_sha256
    pool_019 = br.build_pool_019(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017, c018)
    c019 = br.load_confirmation(root / rd.EXPERIMENT_019_CONFIRMATION_PATH, pool_019, digests)
    digests["confirmation_019"] = c019.content_sha256
    pool = rd.build_pool_020(manifest, extension, c006, c009, c011, c012, c013, c014, c015, c016, c017, c018, c019)
    rc.assert_program_blob()
    confirmation_020 = rd.load_confirmation(root / rd.CONFIRMATION_RELATIVE_PATH, pool, digests)
    closure = rc.verify_020_closure(root, confirmation_020)
    digests = dict(digests) | dict(closure["digests"])
    confirmations = {"006": c006, "009": c009, "011": c011, "012": c012, "013": c013, "014": c014, "015": c015, "016": c016, "017": c017, "018": c018, "019": c019,
                     "020": confirmation_020}
    return FrozenInputs(pool, lock_011, lock_012, lock_017, confirmation_020, closure, digests, confirmations, manifest, extension)


# ---------------------------------------------------------------------------
# The model-side objects shared by every pair.


@dataclass
class ModelPrograms:
    """The weights and the programs derived from them once per phase: layers 1–3 (the upstream chain and its exact
    counterpart), layer 0 (block 0's attention), the frozen readout, the committed 017 chain and the noun read."""

    weights: pm.Weights
    lw: lc.LayerWeights
    programs: Mapping[int, atp.LayerProgram]
    program0: atp.LayerProgram
    readout: rd.ReadoutProgram
    chain: hp.HeadChainModel
    nouns: rd.NounSet
    scorable: list[int]
    head: Any = None
    axis_T: Any = None

    @classmethod
    def from_model(cls, model: Any, inputs: FrozenInputs) -> "ModelPrograms":
        weights = pm.Weights.from_model(model)
        lw = lc.LayerWeights.from_model(model, layers=rd.PROGRAM_LAYERS)
        programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in rd.PROGRAM_LAYERS}
        program0 = atp.LayerProgram.from_model(model, 0)
        readout = rd.ReadoutProgram.from_model(model)
        chain = rd.chain_from_locks(inputs.lock_011, inputs.lock_012, inputs.lock_017, lw, programs, inputs.pool)
        nouns = rd.NounSet.build(weights, inputs.pool.nouns, [])  # no fresh noun anywhere in 022
        axis_vector = torch.tensor(inputs.lock_011["axes_vectors"]["T"], dtype=torch.float64)
        axis_T = pm.SiteAxis("T", torch.zeros_like(axis_vector), axis_vector, float(inputs.lock_011["sigma_T"]))
        return cls(weights, lw, programs, program0, readout, chain, nouns, list(nouns.exposed_scorable), ht.HeadWeights.from_model(model), axis_T)


# ---------------------------------------------------------------------------
# Block 0 and the factors.


@dataclass(frozen=True)
class Block0Terms:
    total: torch.Tensor  # ΔA0(q): block 0's attention output change at query q
    value: torch.Tensor  # the reference row carrying the cue key's changed value (017's F, at block 0)
    pattern: torch.Tensor  # total − value: the changed row carrying the new values (017's Π, at block 0)
    weight_to_cue: torch.Tensor  # [heads] the reference attention from q to p_c
    per_head_value_norm: torch.Tensor  # [heads]


def reference_embeddings(weights: pm.Weights, frame: pm.Frame, reference_id: int) -> list[torch.Tensor]:
    """The reference prompt's embedding rows (Pythia adds no positional embedding; rotary position enters inside
    attention)."""
    return [weights.W_E[int(token)].double() for token in frame.prompt_ids(int(reference_id))]


def block0_terms(program0: atp.LayerProgram, x0_all: Sequence[torch.Tensor], p_c: int, q: int, d_emb: torch.Tensor) -> Block0Terms:
    """Block 0's exact attention program at the reference prefix with the embedding at ``p_c`` changed by ``d_emb``:
    the query is recomputed only when ``q = p_c``; the key and value at ``p_c`` always."""
    rr = atp.ReferenceRow(program0, [x.double() for x in x0_all[: q + 1]])
    normed_pc = program0.normalize(x0_all[p_c].double() + d_emb.double())
    rows, values = hp._row_and_values(program0, rr, normed_pc if q == p_c else None, {p_c: normed_pc})
    out_new, out_ref = program0.output(values), program0.output(rr.values)
    per_head = rr.A_ref[:, p_c].unsqueeze(-1) * (out_new[:, p_c] - out_ref[:, p_c])
    value = per_head.sum(dim=0)
    total = hp._attention_output(program0, rows, values) - rr.output_ref
    return Block0Terms(total, value, total - value, rr.A_ref[:, p_c].clone(), per_head.norm(dim=-1))


@dataclass(frozen=True)
class Factors:
    delta_e: torch.Tensor  # the committed cue input: MLP₀(LN₂(W_E[cue])) − MLP₀(LN₂(W_E[ref]))
    d_emb: torch.Tensor  # W_E[cue] − W_E[ref]
    value_pc: torch.Tensor
    pattern_pc: torch.Tensor
    attn_pt: torch.Tensor | None  # ΔA0(p_t), coordinated frames only
    block0_pc: Block0Terms
    block0_pt: Block0Terms | None


def pair_factors(progs: ModelPrograms, x0_all: Sequence[torch.Tensor], frame: pm.Frame, token_id: int, reference_id: int) -> Factors:
    delta_e = progs.chain.fcm.read.encoding_delta(progs.weights, int(token_id), frame.template_id).double()
    d_emb = progs.weights.W_E[int(token_id)].double() - progs.weights.W_E[int(reference_id)].double()
    b_pc = block0_terms(progs.program0, x0_all, frame.p_c, frame.p_c, d_emb)
    b_pt = block0_terms(progs.program0, x0_all, frame.p_c, frame.p_t, d_emb) if frame.p_t != frame.p_c else None
    return Factors(delta_e, d_emb, b_pc.value, b_pc.pattern, None if b_pt is None else b_pt.total, b_pc, b_pt)


# ---------------------------------------------------------------------------
# Layers 1–2: the exact program at several input positions, and the committed reduced program with its input replaced.


def exact_chain_multi(programs: Mapping[int, atp.LayerProgram], lw: lc.LayerWeights, x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor],
                      p_c: int, p_t: int, dx1: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Experiment 017's ``exact_chain`` loop over a layer-1 input change at any of the changed positions; with an input
    at ``p_c`` only it is ``hp.exact_chain(...)["dx3"]`` bit for bit (a test pins it)."""
    positions = sorted({p_c, p_t})
    residuals = {1: [x.double() for x in x1_all], 2: [x.double() for x in x2_all]}
    dx: dict[int, dict[int, torch.Tensor]] = {1: {pos: d.double() for pos, d in dx1.items()}}
    for layer in hp.UPSTREAM_LAYERS:
        program, xs = programs[layer], residuals[layer]
        normed = {pos: program.normalize(xs[pos] + d) for pos, d in dx[layer].items()}
        following: dict[int, torch.Tensor] = {}
        for pos in positions:
            rr = atp.ReferenceRow(program, xs[: pos + 1])
            rows, values = hp._row_and_values(program, rr, normed.get(pos), {k: n for k, n in normed.items() if k <= pos})
            d_in = dx[layer].get(pos)
            change = hp._attention_output(program, rows, values) - rr.output_ref
            if d_in is not None:
                change = change + d_in + lw.delta_out(layer, xs[pos], d_in)
            following[pos] = change
        dx[layer + 1] = following
    return dx[hp.HEAD_LAYER]


def reduced_chain(chain: hp.HeadChainModel, rows16: Mapping[int, atp.ReferenceRow], x1_all: Sequence[torch.Tensor], x2_all: Sequence[torch.Tensor],
                  p_c: int, p_t: int, template: str, u_pc: torch.Tensor, u_pt: torch.Tensor | None = None) -> dict[int, torch.Tensor]:
    """``HeadChainModel.upstream`` at ``LEVEL0`` (016's ``LEVEL0F`` channels at ``p_c``, channel D at the frame's
    operating point, the exact one-step propagation to ``p_t``) with its cue input replaced by ``u_pc``; a nonzero
    ``u_pt`` enters layer 1 at ``p_t`` exactly. With ``u_pc = ΔE`` and no ``u_pt`` it is ``rd.predicted_dx3`` bit for bit
    (identity I5)."""
    fcm, channels = chain.fcm, fch.LEVEL0F
    x1, x2 = x1_all[p_c].double(), x2_all[p_c].double()
    xb1, xb2 = (base.double() for base in fcm.bases_012[template])
    delta_e = u_pc.double()
    one = fcm.layer_rows(1, rows16[1], xb1, x1, delta_e, channels)
    d1 = fcm.lw.delta_out(1, x1, delta_e)
    out1 = rows16[1].output_change(one["row"], one["v_pc"])
    dx2_pc = delta_e + d1 + out1
    two = fcm.layer_rows(2, rows16[2], xb2, x2, dx2_pc, channels)
    out2_pc = rows16[2].output_change(two["row"], two["v_pc"])
    dx3 = {p_c: dx2_pc + out2_pc + fcm.lw.delta_out(2, x2, dx2_pc)}
    if p_t != p_c:
        program1, program2 = fcm.programs[1], fcm.programs[2]
        rr1 = atp.ReferenceRow(program1, [x.double() for x in x1_all[: p_t + 1]])
        keys = {p_c: program1.normalize(x1 + delta_e)}
        query = None
        if u_pt is not None:
            keys[p_t] = program1.normalize(x1_all[p_t].double() + u_pt.double())
            query = keys[p_t]
        rows1, values1 = hp._row_and_values(program1, rr1, query, keys)
        dx2_pt = hp._attention_output(program1, rows1, values1) - rr1.output_ref
        if u_pt is not None:
            dx2_pt = dx2_pt + u_pt.double() + fcm.lw.delta_out(1, x1_all[p_t].double(), u_pt.double())
        rr2 = atp.ReferenceRow(program2, [x.double() for x in x2_all[: p_t + 1]])
        x2_pt = x2_all[p_t].double()
        normed2 = {p_c: program2.normalize(x2 + dx2_pc), p_t: program2.normalize(x2_pt + dx2_pt)}
        rows2, values2 = hp._row_and_values(program2, rr2, normed2[p_t], normed2)
        dx3[p_t] = dx2_pt + (hp._attention_output(program2, rows2, values2) - rr2.output_ref) + fcm.lw.delta_out(2, x2_pt, dx2_pt)
    return dx3


# ---------------------------------------------------------------------------
# The canonical 32-coalition composition.


@dataclass
class PairContext:
    frame: pm.Frame
    state: rd.FrameState020
    rows16: Mapping[int, atp.ReferenceRow]
    factors: Factors
    token_id: int
    word: str

    @property
    def cue_final(self) -> bool:
        return self.frame.p_t == self.frame.p_c

    @property
    def positions(self) -> list[int]:
        return sorted({self.frame.p_c, self.frame.p_t})


def pair_context(progs: ModelPrograms, frame: pm.Frame, state: rd.FrameState020, reference_id: int, token_id: int, word: str,
                 rows16: Mapping[int, atp.ReferenceRow] | None = None) -> PairContext:
    s17 = state.state_017
    rows16 = rows16 if rows16 is not None else atp.reference_rows(progs.programs, s17.x1_all, s17.x2_all)
    x0_all = reference_embeddings(progs.weights, frame, reference_id)
    return PairContext(frame, state, rows16, pair_factors(progs, x0_all, frame, token_id, reference_id), int(token_id), word)


def composition_inputs(factors: Factors, mask: int, cue_final: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``u_pc = ΔE + [emb]·Δemb + [Bv]·V + [Bp]·P`` (added in that fixed order) and ``u_pt = [T]·ΔA0(p_t)``; the empty
    coalition's input is ``ΔE`` itself, bit for bit."""
    mask = canonical_mask(mask, cue_final)
    u_pc = factors.delta_e
    if mask & BIT["emb"]:
        u_pc = u_pc + factors.d_emb
    if mask & BIT["Bv"]:
        u_pc = u_pc + factors.value_pc
    if mask & BIT["Bp"]:
        u_pc = u_pc + factors.pattern_pc
    u_pt = factors.attn_pt if (mask & BIT["T"] and not cue_final) else None
    return u_pc, u_pt


def compose_dx3(progs: ModelPrograms, ctx: PairContext, mask: int) -> dict[int, torch.Tensor]:
    mask = canonical_mask(mask, ctx.cue_final)
    u_pc, u_pt = composition_inputs(ctx.factors, mask, ctx.cue_final)
    s17 = ctx.state.state_017
    p_c, p_t = ctx.frame.p_c, ctx.frame.p_t
    if mask & BIT["R"]:
        dx1 = {p_c: u_pc}
        if u_pt is not None:
            dx1[p_t] = u_pt
        return exact_chain_multi(progs.programs, progs.lw, s17.x1_all, s17.x2_all, p_c, p_t, dx1)
    return reduced_chain(progs.chain, ctx.rows16, s17.x1_all, s17.x2_all, p_c, p_t, ctx.frame.template_id, u_pc, u_pt)


def contrast_of(progs: ModelPrograms, state: rd.FrameState020, dx3: Mapping[int, torch.Tensor]) -> torch.Tensor:
    """The frozen downstream readout of a ``Δx3``: 020's blocks 3–5, ``LN_final`` and the noun read (79 nouns)."""
    return progs.readout.contrast(state, progs.readout.blocks_3_to_5(state, dict(dx3))["dh6"], progs.nouns)[progs.scorable]


def pair_compositions(progs: ModelPrograms, ctx: PairContext) -> tuple[torch.Tensor, dict[int, dict[int, torch.Tensor]]]:
    """``Δĉ`` of every coalition, ``[32, N]``: each canonical coalition is computed once, so in a cue-final frame the
    row of ``m | T`` is the very tensor of ``m``. Also returns the ``Δx̂3`` of the empty and the full coalition."""
    rows: dict[int, torch.Tensor] = {}
    kept: dict[int, dict[int, torch.Tensor]] = {}
    for mask in range(N_MASKS):
        canon = canonical_mask(mask, ctx.cue_final)
        if canon in rows:
            continue
        dx3 = compose_dx3(progs, ctx, canon)
        if canon in (0, canonical_mask(FULL_MASK, ctx.cue_final)):
            kept[canon] = dx3
        rows[canon] = contrast_of(progs, ctx.state, dx3)
    table = torch.stack([rows[canonical_mask(mask, ctx.cue_final)] for mask in range(N_MASKS)])
    return table, {"empty": kept[0], "full": kept[canonical_mask(FULL_MASK, ctx.cue_final)]}


def pair_cells(dc_measured: torch.Tensor, compositions: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
    """One pair's cell for the kernel: ``SSE[32]`` and the two-pass moments ``(count, mean, M2)`` of the measured
    ``Δc`` over the 79 nouns."""
    y = dc_measured.double()
    sse = ((y.unsqueeze(0) - compositions.double()) ** 2).sum(dim=-1)
    mean = float(y.mean())
    m2 = float(((y - mean) ** 2).sum())
    return sse, float(y.shape[0]), mean, m2


# ---------------------------------------------------------------------------
# One measured prompt, and the pair's gates.


def measured_sites(frame: pm.Frame) -> list[tuple[str, int]]:
    positions = sorted({frame.p_c, frame.p_t})
    return [("RESID_PRE.L1", position) for position in positions] + [(f"RESID_PRE.L{hp.HEAD_LAYER}", position) for position in positions]


def measure_prompt(model: Any, progs: ModelPrograms, frame: pm.Frame, state: rd.FrameState020, token_id: int, word: str) -> dict[str, Any]:
    """One forward of the real cue prompt: the measured ``Δc`` (79 nouns, against the state's reference contrasts, as
    020 measured it), ``Δx1`` and ``Δx3`` at the changed positions."""
    run = pm.capture_prompt(model, pm.Prompt(frame, int(token_id), word), measured_sites(frame))
    positions = sorted({frame.p_c, frame.p_t})
    s17 = state.state_017
    dx1 = {position: run.vector(("RESID_PRE.L1", position)).double() - s17.x1_all[position].double() for position in positions}
    dx3 = {position: run.vector((f"RESID_PRE.L{hp.HEAD_LAYER}", position)).double() - state.x3_all[position].double() for position in positions}
    dc = (progs.nouns.contrasts(run.logits) - state.c_ref)[progs.scorable]
    return {"dc": dc.double(), "dx1": dx1, "dx3": dx3}


def pair_gates(progs: ModelPrograms, ctx: PairContext, compositions: torch.Tensor, composed: Mapping[str, Mapping[int, torch.Tensor]],
               measurement: Mapping[str, Any]) -> tuple[dict[str, float], torch.Tensor]:
    """I1–I5 on one pair, and the ceiling ``Δĉ`` (the frozen readout fed the measured ``Δx3``)."""
    f, frame = ctx.factors, ctx.frame
    dx1 = measurement["dx1"]
    i1 = float((f.d_emb + f.delta_e + f.block0_pc.total - dx1[frame.p_c]).abs().max())
    if f.block0_pt is not None:
        i1 = max(i1, float((f.block0_pt.total - dx1[frame.p_t]).abs().max()))
    i2 = float((f.value_pc + f.pattern_pc - f.block0_pc.total).abs().max())
    i3 = i3_error(composed["full"], measurement["dx3"])
    ceiling = contrast_of(progs, ctx.state, measurement["dx3"])
    i4 = float((compositions[FULL_MASK] - ceiling).abs().max())
    level0 = rd.predicted_dx3(progs.chain, progs.weights, ctx.state, ctx.rows16, ctx.token_id, frame.template_id)
    i5 = max(float((composed["empty"][position] - level0[position]).abs().max()) for position in level0)
    if set(level0) != set(composed["empty"]):
        i5 = math.inf
    return {"I1": i1, "I2": i2, "I3": i3, "I4": i4, "I5": i5}, ceiling


# ---------------------------------------------------------------------------
# The freeze: tokenizer only, structural rules only; no model output (Task 4).

CONFIRMATION_SCHEMA_VERSION = 1
COORDINATED = pm.COORDINATED_TEMPLATE


class FreezeShortfall(RuntimeError):
    """A class or template yields fewer than its quota: the freeze writes nothing and stops for review (not an
    incident)."""


def extract_exclusion(inputs: FrozenInputs) -> dict[str, Any]:
    """Every cue token id and frame text any earlier experiment froze or executed, read from structured fields only
    (plan revision 2, Q5): the frozen loaders' ``tokens[*]["token_id"]`` and ``frames[*]`` of the 006–020
    confirmations, the 020 provenance pool (tokens, frames, reference and plural cues), and the screening manifest's
    extension (frames and cue words). Over-exclusion is allowed; under-exclusion is not."""
    pool = inputs.pool
    cue_ids: set[int] = {int(token_id) for _, token_id in pool.tokens}
    cue_ids |= {int(token_id) for token_id in pool.reference_ids.values()}
    cue_ids |= {int(pool.token_id(name)) for name in pool.plural_cue.values()}
    frames: list[pm.Frame] = list(pool.frames)
    sources = [{"source": "pool-020", "loader": "rd.build_pool_020", "tokens": len(pool.tokens), "frames": len(pool.frames)}]
    for key, confirmation in sorted(inputs.confirmations.items()):
        cue_ids |= {int(token["token_id"]) for token in confirmation.tokens}
        frames += list(confirmation.frames)
        sources.append({"source": f"confirmation-{key}", "content_sha256": confirmation.content_sha256, "tokens": len(confirmation.tokens), "frames": len(confirmation.frames)})
    extension = inputs.extension
    frames += list(extension.original_frames) + list(extension.new_frames)
    cue_ids |= {int(token_id) for _, token_id in extension.cue_words} | {int(token_id) for token_id in extension.reference_cue_ids.values()}
    sources.append({"source": "extension", "content_sha256": extension.content_sha256, "frames": len(extension.original_frames) + len(extension.new_frames),
                    "tokens": len(extension.cue_words)})
    for frame in frames:
        cue_ids |= {int(token_id) for token_id in frame.cue_ids.values()}
    texts = sorted({frame.text_template for frame in frames})
    ids = sorted(cue_ids)
    return {"cue_token_ids": ids, "cue_token_ids_sha256": pm.sha256_text(pm.canonical_json(ids)), "frame_texts": texts,
            "frame_texts_sha256": pm.sha256_text(pm.canonical_json(texts)), "sources": sources}


def freeze_payload(tokenizer: Any, inputs: FrozenInputs, *, model: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The first eligible entries of the frozen ordered lists (design revision 3). A cue is eligible when it is a
    single token with a leading space and its id is new; a frame when its text is new and the project's own frame
    builder accepts it with ``p_c`` in its template's exposed range (coordinated: one adjective token, so
    ``p_t = p_c + 1``). Frame ids are ``<template>-022-<k>``. A shortfall writes nothing (``FreezeShortfall``)."""
    exclusion = extract_exclusion(inputs)
    excluded_ids, excluded_texts = set(exclusion["cue_token_ids"]), set(exclusion["frame_texts"])
    cues: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for cls, words in CUE_CANDIDATES.items():
        taken = 0
        for rank, word in enumerate(words):
            if taken == CUE_QUOTA:
                break
            ids = pm._encode(tokenizer, " " + word)
            if len(ids) != 1:
                rejected.append({"kind": "cue", "class": cls, "candidate": word, "rank": rank, "reason": f"{len(ids)} tokens with a leading space"})
                continue
            if ids[0] in excluded_ids or ids[0] in {entry["token_id"] for entry in cues}:
                rejected.append({"kind": "cue", "class": cls, "candidate": word, "rank": rank, "reason": f"token id {ids[0]} already used"})
                continue
            cues.append({"word": word, "token_id": int(ids[0]), "class": cls, "candidate_rank": rank})
            taken += 1
        if taken < CUE_QUOTA:
            raise FreezeShortfall(f"cue class {cls}: {taken} eligible of {CUE_QUOTA}")
    pool = inputs.pool
    cue_ids_by_template = {frame.template_id: dict(frame.cue_ids) for frame in pool.frames if pool.frame_origin[frame.frame_id] == "manifest"}
    frames: list[dict[str, Any]] = []
    for template, texts in FRAME_CANDIDATES.items():
        taken = 0
        for rank, text in enumerate(texts):
            if taken == FRAME_QUOTA:
                break
            if text in excluded_texts or text in {entry["text_template"] for entry in frames}:
                rejected.append({"kind": "frame", "template": template, "candidate": text, "rank": rank, "reason": "text already used"})
                continue
            frame_id = f"{template}-{FRAME_ID_TAG}-{taken + 1}"
            try:
                frame = pm._build_new_frame(tokenizer, template, text, cue_ids_by_template[template], frame_id)
            except ValueError as error:
                rejected.append({"kind": "frame", "template": template, "candidate": text, "rank": rank, "reason": str(error)})
                continue
            low, high = P_C_RANGE[template]
            if not low <= frame.p_c <= high:
                rejected.append({"kind": "frame", "template": template, "candidate": text, "rank": rank, "reason": f"p_c {frame.p_c} outside {low}–{high}"})
                continue
            if (template == COORDINATED) != (frame.p_t == frame.p_c + 1):
                rejected.append({"kind": "frame", "template": template, "candidate": text, "rank": rank, "reason": f"p_t {frame.p_t} against p_c {frame.p_c}"})
                continue
            frames.append({**frame.to_dict(), "candidate_rank": rank})
            taken += 1
        if taken < FRAME_QUOTA:
            raise FreezeShortfall(f"template {template}: {taken} structurally eligible of {FRAME_QUOTA}")
    payload = {
        "experiment": EXPERIMENT, "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "kind": "the new cues and frames of Experiment 022, frozen from the tokenizer and structural rules alone; no model output",
        "design": dict(DESIGN), "plan": dict(PLAN),
        "model": dict(model or {"model_id": models_module.PYTHIA_70M.model_id, "revision": models_module.PYTHIA_70M.revision}),
        "candidates": {"cues": {cls: list(words) for cls, words in CUE_CANDIDATES.items()}, "frames": {template: list(texts) for template, texts in FRAME_CANDIDATES.items()}},
        "rules": {"cue_quota": CUE_QUOTA, "frame_quota": FRAME_QUOTA, "p_c_range": {template: list(bounds) for template, bounds in P_C_RANGE.items()},
                  "cue": "a single token with a leading space whose id no earlier experiment used", "frame": "a new text accepted by pm._build_new_frame; coordinated: one adjective token"},
        "exclusion": exclusion, "reference_cue_ids": {template: int(token_id) for template, token_id in pool.reference_ids.items()},
        "exposed_frame_ids": [frame.frame_id for frame in pool.frames],
        "cues": cues, "frames": frames, "rejected": rejected,
        "counts": {"classes": {cls: sum(1 for entry in cues if entry["class"] == cls) for cls in CUE_CANDIDATES},
                   "templates": {template: sum(1 for entry in frames if entry["template_id"] == template) for template in FRAME_CANDIDATES}},
    }
    payload["manifest"] = confirmation_from_payload(payload, pool, verify=False).manifest()
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


@dataclass(frozen=True)
class Confirmation022:
    reference_ids: Mapping[str, int]
    frames: tuple[pm.Frame, ...]  # the 18 new frames
    exposed_frames: tuple[pm.Frame, ...]  # the 108 exposed frames, in the pool's order
    tokens: tuple[dict[str, Any], ...]  # the 24 new cues: word, token_id, class
    content_sha256: str

    def reference_prompt(self, frame: pm.Frame) -> pm.Prompt:
        return pm.Prompt(frame, int(self.reference_ids[frame.template_id]), "ref")

    def validity_prompt(self, frame: pm.Frame) -> pm.Prompt:
        return pm.Prompt(frame, int(frame.cue_ids["pl"]), "pl")

    @property
    def stage1_prompts(self) -> tuple[pm.Prompt, ...]:
        return tuple(self.reference_prompt(frame) for frame in self.frames) + tuple(self.validity_prompt(frame) for frame in self.frames)

    @property
    def y1_prompts(self) -> tuple[pm.Prompt, ...]:
        return tuple(pm.Prompt(frame, int(token["token_id"]), token["word"]) for frame in self.exposed_frames for token in self.tokens)

    @property
    def y2_prompts(self) -> tuple[pm.Prompt, ...]:
        return tuple(pm.Prompt(frame, int(token["token_id"]), token["word"]) for frame in self.frames for token in self.tokens)

    @property
    def target_prompts(self) -> tuple[pm.Prompt, ...]:
        return self.y1_prompts + self.y2_prompts

    @property
    def all_prompts(self) -> tuple[pm.Prompt, ...]:
        return self.stage1_prompts + self.target_prompts

    def manifest(self) -> dict[str, Any]:
        return {"S1-REF": sorted(self.reference_prompt(frame).key for frame in self.frames), "S1-VALIDITY": sorted(self.validity_prompt(frame).key for frame in self.frames),
                "S2-TARGET": {"Y1": sorted(prompt.key for prompt in self.y1_prompts), "Y2": sorted(prompt.key for prompt in self.y2_prompts)}}

    def manifest_keys(self) -> frozenset[str]:
        return frozenset(prompt.key for prompt in self.all_prompts)

    def counts(self) -> dict[str, dict[str, int]]:
        return {"classes": {cls: sum(1 for token in self.tokens if token["class"] == cls) for cls in CUE_CANDIDATES},
                "templates": {template: sum(1 for frame in self.frames if frame.template_id == template) for template in FRAME_CANDIDATES}}


def confirmation_from_payload(payload: Mapping[str, Any], pool: Any, *, verify: bool = True) -> Confirmation022:
    frames = tuple(pm.Frame(entry["template_id"], entry["frame_id"], tuple(entry["prefix_ids"]), tuple(entry["suffix_ids"]), entry["cue_ids"], entry["text_template"],
                            origin="extension") for entry in payload["frames"])
    tokens = tuple({"word": entry["word"], "token_id": int(entry["token_id"]), "class": entry["class"]} for entry in payload["cues"])
    confirmation = Confirmation022(dict(pool.reference_ids), frames, tuple(pool.frames), tokens, str(payload.get("content_sha256", "")))
    if verify:
        if payload.get("experiment") != EXPERIMENT or payload.get("schema_version") != CONFIRMATION_SCHEMA_VERSION:
            raise PhaseError("not an Experiment 022 confirmation file")
        if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
            raise PhaseError("the confirmation file's content digest does not verify")
        if payload["manifest"] != confirmation.manifest():
            raise PhaseError("the confirmation file's manifest is not the one its cues and frames define")
        if confirmation.counts() != {"classes": {cls: CUE_QUOTA for cls in CUE_CANDIDATES}, "templates": {template: FRAME_QUOTA for template in FRAME_CANDIDATES}}:
            raise PhaseError(f"the confirmation file's composition is not 6/6/6/6 cues and 6/6/6 frames: {confirmation.counts()}")
        if payload.get("counts") != confirmation.counts():
            raise PhaseError("the confirmation file's recorded counts are not those of its cues and frames (the calibration reads these counts)")
        if payload["exposed_frame_ids"] != [frame.frame_id for frame in pool.frames] or dict(payload["reference_cue_ids"]) != {k: int(v) for k, v in pool.reference_ids.items()}:
            raise PhaseError("the confirmation file names a different exposed pool")
    return confirmation


def load_confirmation_022(path: Path, inputs: FrozenInputs) -> Confirmation022:
    """The committed freeze artifact, verified: digest, manifest, composition, and no overlap with anything used
    before (re-extracted from the frozen inputs now)."""
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    confirmation = confirmation_from_payload(payload, inputs.pool)
    exclusion = extract_exclusion(inputs)
    if payload["exclusion"]["cue_token_ids_sha256"] != exclusion["cue_token_ids_sha256"] or payload["exclusion"]["frame_texts_sha256"] != exclusion["frame_texts_sha256"]:
        raise PhaseError("the exclusion sets recorded at the freeze differ from those of the frozen inputs now")
    used_ids, used_texts = set(exclusion["cue_token_ids"]), set(exclusion["frame_texts"])
    if {int(token["token_id"]) for token in confirmation.tokens} & used_ids or {frame.text_template for frame in confirmation.frames} & used_texts:
        raise PhaseError("a new cue or frame was used by an earlier experiment")
    return confirmation



# ---------------------------------------------------------------------------
# The calibration (exposed only, once; Task 5).


class CrossCheckError(IncidentError):
    """The kernel disagreed with the direct recomputation beyond the implementation tolerance: an incident, with its
    location. No floor exists yet, so nothing is written."""

    def __init__(self, details: Mapping[str, Any]):
        self.details = dict(details)
        super().__init__(f"kernel/direct cross-check failed: {self.details.get('max_difference')} at {self.details.get('at')} above {TOLERANCES['kernel']:.0e} "
                         f"({self.details.get('n_exceeding')} of {self.details.get('n_checked')} quantities exceed)")


@dataclass(frozen=True)
class CalibrationUnits:
    """Experiment 021's provenance pools in the frozen unit order (plan revision 2, Q6).

    ``cues`` is flat: the classes in ``CUE_CLASSES`` order, token id within a class; ``class_offsets[cls]`` is its
    ``(start, count)``. The frame axis is the 108 exposed frames by ``frame_id``; ``y2_frames[template]`` are the
    frame-axis indices of the 14 frames first confirmed in 017–019, by ``frame_id``; ``slots`` are the fresh set's
    frozen counts per stratum."""

    cues: tuple[tuple[str, int, str], ...]
    class_offsets: Mapping[str, tuple[int, int]]
    frames: tuple[Any, ...]
    y2_frames: Mapping[str, tuple[int, ...]]
    groups: Mapping[str, tuple[int, ...]]
    slots: Mapping[str, int]

    def sizes(self) -> dict[str, int]:
        return {**{f"cue/{cls}": self.class_offsets[cls][1] for cls in CUE_CLASSES}, **{f"frame/{template}": len(self.y2_frames[template]) for template in TEMPLATES}}

    def to_json(self) -> dict[str, Any]:
        return {"cues": [[word, token_id, cls] for word, token_id, cls in self.cues], "class_offsets": {cls: list(entry) for cls, entry in self.class_offsets.items()},
                "frames": [frame.frame_id for frame in self.frames], "y2_frames": {template: [self.frames[i].frame_id for i in indices] for template, indices in self.y2_frames.items()},
                "groups": {group: len(indices) for group, indices in self.groups.items()}, "slots": dict(self.slots), "strata_sizes": self.sizes()}


def units_from(cues_by_class: Mapping[str, Sequence[tuple[str, int]]], frames: Sequence[Any], y2_frame_ids: Mapping[str, Sequence[str]],
               counts: Mapping[str, Mapping[str, int]]) -> CalibrationUnits:
    cues: list[tuple[str, int, str]] = []
    offsets: dict[str, tuple[int, int]] = {}
    for cls in CUE_CLASSES:
        members = sorted(cues_by_class[cls], key=lambda entry: int(entry[1]))
        offsets[cls] = (len(cues), len(members))
        cues += [(str(word), int(token_id), cls) for word, token_id in members]
    ordered = tuple(sorted(frames, key=lambda frame: frame.frame_id))
    position = {frame.frame_id: index for index, frame in enumerate(ordered)}
    y2 = {template: tuple(position[frame_id] for frame_id in sorted(y2_frame_ids[template])) for template in TEMPLATES}
    if any(ordered[i].template_id != template for template, indices in y2.items() for i in indices):
        raise PhaseError("a Y2-like frame is not of its stratum's template")
    groups = {"cue_final": tuple(i for i, frame in enumerate(ordered) if frame.template_id != COORDINATED),
              "coordinated": tuple(i for i, frame in enumerate(ordered) if frame.template_id == COORDINATED)}
    slots = {**{f"cue/{cls}": int(counts["classes"][cls]) for cls in CUE_CLASSES}, **{f"frame/{template}": int(counts["templates"][template]) for template in TEMPLATES}}
    return CalibrationUnits(tuple(cues), offsets, ordered, y2, groups, slots)


def calibration_units(pool: Any, counts: Mapping[str, Mapping[str, int]]) -> CalibrationUnits:
    """``counts`` are the committed confirmation file's class and template counts: the only thing the calibration
    reads from it (plan revision 2, Q1)."""
    pools = rc.production_pools(pool)
    return units_from(pools.cues, pool.frames, pools.frames_unscreened, counts)


@dataclass
class CalibrationTable:
    """The re-materialized exposed pairs, ``[C, F, …]`` in the units' order."""

    dc: torch.Tensor  # [C, F, N] measured Δc
    dc_hat: torch.Tensor  # [C, F, 32, N] every coalition; in a cue-final frame the row of m | T is the row of m
    sse: torch.Tensor  # [C, F, 32]
    count: torch.Tensor  # [C, F]
    mean: torch.Tensor  # [C, F]
    m2: torch.Tensor  # [C, F]
    ceiling: torch.Tensor  # [C, F, N]
    gates: dict[str, torch.Tensor]  # I1–I5, [C, F]

    TENSORS = ("dc", "dc_hat", "sse", "count", "mean", "m2", "ceiling")

    def cells(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c, f = self.count.shape
        return self.sse.reshape(c * f, N_MASKS), self.count.reshape(-1), self.mean.reshape(-1), self.m2.reshape(-1)

    def digests(self) -> dict[str, str]:
        return {**{name: rc.tensor_digest(getattr(self, name)) for name in self.TENSORS}, **{f"gate_{name}": rc.tensor_digest(values) for name, values in sorted(self.gates.items())}}

    def saved(self, units: CalibrationUnits, noun_keys: Sequence[str]) -> dict[str, Any]:
        return {"cues": [word for word, _, _ in units.cues], "token_ids": [token_id for _, token_id, _ in units.cues], "classes": [cls for _, _, cls in units.cues],
                "frames": [frame.frame_id for frame in units.frames], "templates": [frame.template_id for frame in units.frames], "noun_keys": list(noun_keys),
                **{name: getattr(self, name) for name in self.TENSORS}, "gates": dict(self.gates)}


def calibration_rematerialize(model: Any, progs: ModelPrograms, inputs: FrozenInputs, units: CalibrationUnits, *, forbidden_keys: frozenset[str],
                              log: Callable[[str], None] | None = None, executed: list[pm.Prompt] | None = None) -> CalibrationTable:
    """One forward per exposed pair (175 pool cues × 108 exposed frames) against Experiment 020's locked reference
    states. A frame's keys are checked before any of its prompts runs: every key in 020's ledger, none in
    ``forbidden_keys`` (the 022 manifest and 020's confirmation set). Then the factors, the canonical compositions,
    the pair cells, the ceiling and I1–I5. Every executed prompt is appended to ``executed`` the moment it runs.
    Nothing is enforced here: the caller writes the gate maxima first."""
    say = log or (lambda message: None)
    executed = executed if executed is not None else []
    ledger = inputs.closure["ledger"]
    locked = inputs.closure["exploration"]["locked_states"]
    n_cues, n_frames, n_nouns = len(units.cues), len(units.frames), len(progs.scorable)
    dc = torch.empty(n_cues, n_frames, n_nouns, dtype=torch.float64)
    dc_hat = torch.empty(n_cues, n_frames, N_MASKS, n_nouns, dtype=torch.float64)
    sse = torch.empty(n_cues, n_frames, N_MASKS, dtype=torch.float64)
    count, mean, m2 = (torch.empty(n_cues, n_frames, dtype=torch.float64) for _ in range(3))
    ceiling = torch.empty(n_cues, n_frames, n_nouns, dtype=torch.float64)
    gates = {name: torch.full((n_cues, n_frames), math.nan, dtype=torch.float64) for name in ("I1", "I2", "I3", "I4", "I5")}
    for fi, frame in enumerate(units.frames):
        prompts = [pm.Prompt(frame, token_id, word) for word, token_id, _ in units.cues]
        foreign = [prompt.key for prompt in prompts if prompt.key not in ledger]
        forbidden = [prompt.key for prompt in prompts if prompt.key in forbidden_keys]
        if forbidden or foreign:
            raise IncidentError(f"{frame.frame_id}: {len(forbidden)} manifest keys {forbidden[:2]} and {len(foreign)} keys outside Experiment 020's ledger {foreign[:2]}; none of this frame ran")
        state = rd.state_from_locked(locked[frame.frame_id], frame)
        s17 = state.state_017
        rows16 = atp.reference_rows(progs.programs, s17.x1_all, s17.x2_all)
        reference_id = int(inputs.pool.reference_ids[frame.template_id])
        for ci, (word, token_id, _) in enumerate(units.cues):
            measurement = measure_prompt(model, progs, frame, state, token_id, word)
            executed.append(prompts[ci])
            ctx = pair_context(progs, frame, state, reference_id, token_id, word, rows16)
            compositions, composed = pair_compositions(progs, ctx)
            values, ceiling_row = pair_gates(progs, ctx, compositions, composed, measurement)
            cell_sse, cell_count, cell_mean, cell_m2 = pair_cells(measurement["dc"], compositions)
            dc[ci, fi], dc_hat[ci, fi], sse[ci, fi], ceiling[ci, fi] = measurement["dc"], compositions, cell_sse, ceiling_row
            count[ci, fi], mean[ci, fi], m2[ci, fi] = cell_count, cell_mean, cell_m2
            for name, value in values.items():
                gates[name][ci, fi] = value
        say(f"  {frame.frame_id}: {n_cues} pool cues measured and composed ({fi + 1}/{n_frames})")
    return CalibrationTable(dc, dc_hat, sse, count, mean, m2, ceiling, gates)


def gate_summary(gates: Mapping[str, torch.Tensor], units: CalibrationUnits) -> dict[str, dict[str, Any]]:
    """Per gate: the maximum over every pair (``None`` when a value is missing or not finite) and where it is."""
    out = {}
    for name, values in sorted(gates.items()):
        flat = values.reshape(-1)
        bad = ~torch.isfinite(flat)
        position = int(bad.nonzero()[0]) if bool(bad.any()) else int(torch.argmax(flat))
        ci, fi = divmod(position, int(values.shape[1]))
        out[name] = {"max": None if bool(bad.any()) else float(flat[position]), "at": f"{units.cues[ci][0]}|{units.frames[fi].frame_id}", "tolerance": TOLERANCES[name]}
    return out


def enforce_gates(summary: Mapping[str, Mapping[str, Any]], names: Sequence[str] = ("I1", "I2", "I3", "I4", "I5")) -> None:
    for name in names:
        entry = summary.get(name) or {}
        value = entry.get("max")
        if value is None or not value <= TOLERANCES[name]:
            raise IncidentError(f"identity gate {name} failed: {value} at {entry.get('at')} against the frozen tolerance {TOLERANCES[name]:.0e}")


def assert_measurements_finite(table: CalibrationTable) -> None:
    for name in ("dc", "ceiling", "dc_hat", "sse", "m2"):
        if not bool(torch.isfinite(getattr(table, name)).all()):
            raise IncidentError(f"a missing or non-finite value in the calibration table's {name}")


def r1_gate(root: Path, table: CalibrationTable, units: CalibrationUnits, noun_keys: Sequence[str]) -> dict[str, Any]:
    """The re-measured exposed ``Δc`` against Experiment 021's digest-bound exposed table (plan revision 2, Q4):
    021's committed record (file and content digests) and the measured-table digest it records are verified before
    the local table is read, and the local table's digest before any value is compared."""
    import json

    record_path = root / CALIBRATION_021_RELATIVE_PATH
    if rc.file_sha256(record_path) != INHERITED_021["calibration_file_sha256"]:
        raise IncidentError("Experiment 021's committed calibration record is not the frozen file")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("content_sha256") != INHERITED_021["calibration_content_sha256"] or record.get("table_sha256", {}).get("measured") != INHERITED_021["exposed_measured_sha256"]:
        raise IncidentError("Experiment 021's calibration record does not bind the frozen exposed table")
    table_path = root / EXPOSED_TABLE_021_RELATIVE_PATH
    if not table_path.exists():
        raise IncidentError(f"{EXPOSED_TABLE_021_RELATIVE_PATH} is missing; R1 cannot be evaluated")
    exposed = torch.load(table_path)
    if rc.tensor_digest(exposed["measured"]) != INHERITED_021["exposed_measured_sha256"]:
        raise IncidentError("Experiment 021's local exposed table is not the one its record digests")
    if list(exposed["noun_keys"]) != list(noun_keys):
        raise IncidentError("Experiment 021's exposed table orders a different noun set")
    row = {(cue, frame): index for index, (cue, frame) in enumerate(zip(exposed["cues"], exposed["frames"]))}
    missing = [(word, frame.frame_id) for word, _, _ in units.cues for frame in units.frames if (word, frame.frame_id) not in row]
    if missing:
        raise IncidentError(f"Experiment 021's exposed table has no row for {missing[:2]}")
    index = torch.tensor([[row[(word, frame.frame_id)] for frame in units.frames] for word, _, _ in units.cues], dtype=torch.int64)
    difference = (table.dc - exposed["measured"].double()[index]).abs().amax(dim=-1)
    position = int(torch.argmax(difference.reshape(-1)))
    ci, fi = divmod(position, len(units.frames))
    worst = float(difference.reshape(-1)[position])
    return {"max_difference": worst, "at": f"{units.cues[ci][0]}|{units.frames[fi].frame_id}", "tolerance": TOLERANCES["R1"], "passed": bool(worst <= TOLERANCES["R1"]),
            "n_pairs": len(units.cues) * len(units.frames), "exposed_measured_sha256": INHERITED_021["exposed_measured_sha256"]}


def draw_units(units: CalibrationUnits, draws: int) -> dict[str, Any]:
    """The SHA-indexed draws: ``cue_index [B, 24]`` (global cue indices, the class blocks in ``CUE_CLASSES`` order) and
    per template ``frame_index [B, 6]`` (frame-axis indices of Y2-like frames); the index digests go into the record."""
    indices = draw_indices(units.sizes(), units.slots, draws)
    cue_index = torch.cat([indices[f"cue/{cls}"] + units.class_offsets[cls][0] for cls in CUE_CLASSES], dim=1)
    frame_index = {template: torch.tensor(units.y2_frames[template], dtype=torch.int64)[indices[f"frame/{template}"]] for template in TEMPLATES}
    return {"cue_index": cue_index, "frame_index": frame_index, "digests": {stratum: rc.tensor_digest(value) for stratum, value in sorted(indices.items())}}


def _y2_group_frames(frame_index: Mapping[str, torch.Tensor], group: str) -> torch.Tensor:
    if group == "cue_final":
        return torch.cat([frame_index[template] for template in TEMPLATES if template != COORDINATED], dim=1)
    return frame_index[COORDINATED]


def _chunks(total: int, size: int):
    for start in range(0, total, size):
        yield start, min(total, start + size)


def kernel_statistics(table: CalibrationTable, units: CalibrationUnits, draws: Mapping[str, Any], *, chunk: int = 500) -> dict[str, Any]:
    """Every draw's games through the one kernel. Y1-like: the drawn cues × all exposed frames of the group (pairs
    to per-cue cells, cells to the draw, by ``group_sums``); Y2-like: the drawn cues × the drawn frames of the group
    (pairs to the draw). Duplicates count with multiplicity."""
    sse_flat, count_flat, mean_flat, m2_flat = table.cells()
    n_frames = len(units.frames)
    cue_index = draws["cue_index"]
    n_draws = int(cue_index.shape[0])
    stats: dict[str, dict[str, dict[str, torch.Tensor]]] = {"Y1": {}, "Y2": {}}
    for group in GROUPS:
        frames = torch.tensor(units.groups[group], dtype=torch.int64)
        per_cue = torch.arange(len(units.cues)).unsqueeze(1) * n_frames + frames.unsqueeze(0)
        cells = group_sums(sse_flat, count_flat, mean_flat, m2_flat, per_cue)
        sse, _, _, sst = group_sums(*cells, cue_index)
        stats["Y1"][group] = {**group_statistics(sse, sst), "sst": sst}
        group_frames = _y2_group_frames(draws["frame_index"], group)
        parts = []
        for start, stop in _chunks(n_draws, chunk):
            pair_index = (cue_index[start:stop].unsqueeze(2) * n_frames + group_frames[start:stop].unsqueeze(1)).reshape(stop - start, -1)
            sse, _, _, sst = group_sums(sse_flat, count_flat, mean_flat, m2_flat, pair_index)
            parts.append({**group_statistics(sse, sst), "sst": sst})
        stats["Y2"][group] = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
    claims = {population: claim_statistics(stats[population]["cue_final"], stats[population]["coordinated"]) for population in POPULATIONS}
    return {"stats": stats, "claims": claims}


def draw_pairs(units: CalibrationUnits, draws: Mapping[str, Any], population: str, group: str, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The materialized pairs ``(cue index, frame index)`` of draw ``b``, every duplicate repeated."""
    cues = draws["cue_index"][b]
    frames = torch.tensor(units.groups[group], dtype=torch.int64) if population == "Y1" else _y2_group_frames(draws["frame_index"], group)[b]
    return cues.repeat_interleave(frames.shape[0]), frames.repeat(cues.shape[0])


def kernel_direct_check(table: CalibrationTable, units: CalibrationUnits, draws: Mapping[str, Any], kernel: Mapping[str, Any], *, n_draws: int | None = None) -> dict[str, Any]:
    """The kernel against the direct recomputation from the materialized pairs (flattened two-pass ``R²``, the
    permutation formula) on the first draws of each population and group: every coalition's value, the gap, the
    Shapley values, the shares and the interpretability flag. Raises ``CrossCheckError`` above the tolerance."""
    count = min(CROSS_CHECK_DRAWS if n_draws is None else int(n_draws), int(draws["cue_index"].shape[0]))
    worst: dict[str, Any] = {"max_difference": -1.0}
    first: dict[str, Any] | None = None
    checked = exceeding = 0
    for population in POPULATIONS:
        for group in GROUPS:
            stats = kernel["stats"][population][group]
            for b in range(count):
                cues, frames = draw_pairs(units, draws, population, group, b)
                direct = direct_group_statistics(table.dc[cues, frames], table.dc_hat[cues, frames])
                positive = bool(stats["sst_positive"][b])
                interpretable = bool(stats["interpretable"][b])
                quantities = [(f"v[{mask}]", float(stats["v"][b, mask]) if positive else None, direct["v"][mask]) for mask in range(N_MASKS)]
                quantities.append(("gap", float(stats["gap"][b]) if positive else None, direct["gap"]))
                quantities += [(f"phi[{name}]", float(stats["phi"][b, i]) if positive else None, None if direct["phi"] is None else direct["phi"][i]) for i, name in enumerate(FACTORS)]
                quantities += [(f"share[{name}]", float(stats["shares"][b, i]) if interpretable else None, None if direct["shares"] is None else direct["shares"][i])
                               for i, name in enumerate(FACTORS)]
                quantities.append(("interpretable", 0.0 if interpretable == bool(direct["interpretable"]) else math.inf, 0.0))
                for name, k, d in quantities:
                    difference = agreement(k, d)
                    checked += 1
                    location = {"population": population, "group": group, "draw": b, "quantity": name, "kernel": k, "direct": d}
                    if difference > worst["max_difference"]:
                        worst = {"max_difference": difference, **location}
                    if not difference <= TOLERANCES["kernel"]:
                        exceeding += 1
                        first = first or {"max_difference": difference, **location}
    result = rc.json_safe({**worst, "at": f"{worst.get('population')}/{worst.get('group')}/draw {worst.get('draw')}/{worst.get('quantity')}", "tolerance": TOLERANCES["kernel"],
                           "scale": rc.AGREEMENT_SCALE, "draws_per_population_and_group": count, "n_checked": checked, "n_exceeding": exceeding, "first_exceeding": first})
    if exceeding:
        raise CrossCheckError(result)
    return result


def efficiency_maxima(kernel: Mapping[str, Any]) -> dict[str, float]:
    """I6 over every draw's games: ``|Σφ − G| / max(1, |G|)``."""
    return {f"{population}/{group}": float(kernel["stats"][population][group]["efficiency"].max()) for population in POPULATIONS for group in GROUPS}


def enforce_efficiency(maxima: Mapping[str, float]) -> None:
    for key, value in maxima.items():
        if not value <= TOLERANCES["I6"]:
            raise IncidentError(f"Shapley efficiency I6 failed on {key}: {value} above {TOLERANCES['I6']:.0e}")


def assert_finite_where_interpretable(claims: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]]) -> None:
    """A non-finite share where the gap rule holds is an implementation incident, never an undefined draw."""
    for population in POPULATIONS:
        for claim in CLAIMS:
            entry = claims[population][claim]
            bad = entry["interpretable"] & ~torch.isfinite(entry["value"])
            if bool(bad.any()):
                raise IncidentError(f"{population}/{claim}: {int(bad.sum())} non-finite shares where the gap rule holds")


def undefined_counts(claims: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]]) -> dict[str, dict[str, int]]:
    return {population: {claim: int((~claims[population][claim]["interpretable"]).sum()) for claim in CLAIMS} for population in POPULATIONS}


def calibration_stop(counts: Mapping[str, Mapping[str, int]], draws: int) -> dict[str, Any]:
    """Design revision 3: 250 or more undefined values of C1, C2 or C4 (125 or more of C3) would make the order
    statistic infinite; the calibration then writes no floor table and stops for review (not an incident, never
    retried automatically)."""
    offending = {f"{population}/{claim}": count for population, entries in counts.items() for claim, count in entries.items() if count >= undefined_threshold(claim, draws)}
    return {"stop": bool(offending), "offending": offending, "thresholds": {claim: undefined_threshold(claim, draws) for claim in CLAIMS}, "counts": {k: dict(v) for k, v in counts.items()}}


def envelopes_and_rates(claims: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]]) -> dict[str, Any]:
    """Per population and claim: the envelope by the exact order statistics; the direction check against the median
    of the defined draws (a violation is an incident, before anything is written); the guard-bound flag; a summary;
    and the rate of each result over the draws, decided by ``classify``. Then the descriptive joint rates."""
    out: dict[str, Any] = {"envelopes": {}, "direction_checks": {}, "guard_bound": {}, "summaries": {}, "result_rates": {}}
    passes: dict[str, torch.Tensor] = {}
    for population in POPULATIONS:
        for key in out:
            out[key][population] = {}
        population_pass = torch.ones(int(claims[population][CLAIMS[0]]["value"].shape[0]), dtype=torch.bool)
        for claim in CLAIMS:
            entry = claims[population][claim]
            values = entry["value"].double()
            defined = entry["interpretable"].clone()
            envelope = claim_envelope(claim, values, defined)
            check = direction_check(claim, envelope, values, defined)
            if not check["ok"]:
                raise IncidentError(f"{population}/{claim}: the direction check failed ({envelope}, median of the defined draws {check['median']}); a tail was reversed")
            results = [classify(claim, float(values[b]) if bool(defined[b]) else None, bool(defined[b]), envelope) for b in range(int(values.shape[0]))]
            population_pass &= torch.tensor([result == "PASS" for result in results], dtype=torch.bool)
            kept = torch.sort(values[defined]).values
            n = int(kept.shape[0])
            out["envelopes"][population][claim] = rc.json_safe(envelope)
            out["direction_checks"][population][claim] = rc.json_safe(check)
            out["guard_bound"][population][claim] = guard_bound(claim, envelope)
            out["summaries"][population][claim] = rc.json_safe({"median": check["median"], "min": float(kept[0]) if n else None, "max": float(kept[-1]) if n else None,
                                                                "defined": n, "undefined": int(values.shape[0]) - n})
            out["result_rates"][population][claim] = {name: results.count(name) / len(results) for name in RESULTS}
        passes[population] = population_pass
    out["joint_rates"] = {"Y1_all_four_pass": float(passes["Y1"].double().mean()), "Y2_all_four_pass": float(passes["Y2"].double().mean()),
                          "all_eight_pass": float((passes["Y1"] & passes["Y2"]).double().mean()), "descriptive_only": True}
    return out


LADDER = (("Level 0", 0), ("R", BIT["R"]), ("R+emb", BIT["R"] | BIT["emb"]), ("R+emb+Bv+Bp", BIT["R"] | BIT["emb"] | BIT["Bv"] | BIT["Bp"]), ("all", FULL_MASK))
INPUTS_ONLY_MASK = FULL_MASK & ~BIT["R"]  # the complete layer-0 input with the reduced layers 1–2 (a number only, never a corrected program)


def game_summary(sse: torch.Tensor, sst: torch.Tensor) -> dict[str, Any]:
    """One aggregate's game, descriptively: the ladder in the spike's order, the ``R``-only and inputs-only values,
    every coalition's value, the gap, the Shapley values and the shares."""
    stats = group_statistics(sse.reshape(1, N_MASKS), sst.reshape(1))
    interpretable = bool(stats["interpretable"][0])
    return rc.json_safe({"ladder": {name: float(stats["v"][0, mask]) for name, mask in LADDER}, "r_only": float(stats["v"][0, BIT["R"]]),
                         "inputs_only": float(stats["v"][0, INPUTS_ONLY_MASK]), "v": [float(value) for value in stats["v"][0]], "gap": float(stats["gap"][0]),
                         "phi": {name: float(stats["phi"][0, i]) for i, name in enumerate(FACTORS)},
                         "shares": {name: float(stats["shares"][0, i]) for i, name in enumerate(FACTORS)} if interpretable else None,
                         "interpretable": interpretable, "efficiency": float(stats["efficiency"][0])})


def exposed_descriptives(table: CalibrationTable, units: CalibrationUnits) -> dict[str, Any]:
    """The whole calibration pool (175 cues × 108 frames): per group and per template. Descriptive only."""
    sse_flat, count_flat, mean_flat, m2_flat = table.cells()
    n_frames = len(units.frames)
    selections = {**{f"group/{group}": units.groups[group] for group in GROUPS},
                  **{f"template/{template}": tuple(i for i, frame in enumerate(units.frames) if frame.template_id == template) for template in TEMPLATES}}
    out = {}
    for key, frames in selections.items():
        index = (torch.arange(len(units.cues)).unsqueeze(1) * n_frames + torch.tensor(frames, dtype=torch.int64).unsqueeze(0)).reshape(1, -1)
        sse, _, _, sst = group_sums(sse_flat, count_flat, mean_flat, m2_flat, index)
        out[key] = game_summary(sse[0], sst[0])
    return out


def draw_arrays(kernel: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    arrays: dict[str, torch.Tensor] = {}
    for population in POPULATIONS:
        for group in GROUPS:
            stats = kernel["stats"][population][group]
            for name in ("phi", "gap", "sst"):
                arrays[f"{population}/{group}/{name}"] = stats[name]
            arrays[f"{population}/{group}/interpretable"] = stats["interpretable"].to(torch.int64)
        for claim in CLAIMS:
            arrays[f"{population}/{claim}"] = kernel["claims"][population][claim]["value"]
    return arrays


def record_constants(draws: int) -> dict[str, Any]:
    return {"B": int(draws), "ranks": {"lower": lower_rank(draws), "upper": upper_rank(draws), "c3_low": c3_low_rank(draws), "c3_high": c3_high_rank(draws)},
            "undefined_stop": {claim: undefined_threshold(claim, draws) for claim in CLAIMS},
            "guards": {"C1_min": GUARD_C1_MIN, "C2_max": GUARD_C2_MAX, "C3_range": list(GUARD_C3_RANGE), "C4_exclusive_min": GUARD_C4_EXCLUSIVE_MIN}, "gap_min": GAP_MIN,
            "tolerances": dict(TOLERANCES), "i3_floor": I3_FLOOR, "cross_check_draws": CROSS_CHECK_DRAWS, "draw_tag": DRAW_TAG, "factors": list(FACTORS), "bits": dict(BIT),
            "shapley_weights": [str(weight) for weight in SHAPLEY_WEIGHTS], "results": list(RESULTS), "claims": {claim: {"group": CLAIM_GROUP[claim], "statistic": CLAIM_STATISTIC[claim],
                                                                                                                        "wording": CLAIM_WORDING[claim]} for claim in CLAIMS}}


def calibration_record(*, run_id: str, protocol_code_commit: str, digests: Mapping[str, str], units: CalibrationUnits, confirmation_sha256: str,
                       gates: Mapping[str, Any], r1: Mapping[str, Any], table_digests: Mapping[str, str], draws: Mapping[str, Any], cross_check: Mapping[str, Any],
                       efficiency: Mapping[str, float], undefined: Mapping[str, Any], evaluated: Mapping[str, Any], descriptives: Mapping[str, Any],
                       array_digests: Mapping[str, str], n_draws: int = B) -> dict[str, Any]:
    record = {
        "experiment": EXPERIMENT, "schema_version": 1, "kind": "exposed-only calibration record (design revision 3)", "design": dict(DESIGN), "plan": dict(PLAN),
        "run_id": run_id, "protocol_code_commit": protocol_code_commit, "inputs": dict(digests), "module_blobs": dict(FROZEN_BLOBS),
        "confirmation_022_sha256": confirmation_sha256, "constants": record_constants(n_draws), "pools": units.to_json(),
        "rematerialization": {"n_pairs": len(units.cues) * len(units.frames), "gates": dict(gates), "r1": dict(r1), "table_sha256": dict(table_digests)},
        "draws": {"B": int(n_draws), "index_sha256": dict(draws["digests"]), "strata_sizes": units.sizes(), "slots": dict(units.slots)},
        "cross_check": dict(cross_check), "efficiency_I6": dict(efficiency), "undefined_counts": dict(undefined), **dict(evaluated),
        "descriptive_exposed": dict(descriptives), "draw_arrays_sha256": dict(array_digests),
    }
    validate_json_safe(record)
    record["content_sha256"] = rc.content_digest(record)
    return record


def verify_calibration_record(record: Mapping[str, Any], draws: int = B) -> None:
    """The installed record: its digest, the frozen constants, design, plan and module blobs, and exactly the eight
    envelopes with passing direction checks."""
    if record.get("experiment") != EXPERIMENT or record.get("content_sha256") != rc.content_digest(record):
        raise PhaseError("the calibration record's content digest does not verify")
    if record.get("constants") != record_constants(draws) or record.get("design") != dict(DESIGN) or record.get("plan") != dict(PLAN) or record.get("module_blobs") != dict(FROZEN_BLOBS):
        raise PhaseError("the calibration record was computed under different frozen constants, design, plan or modules")
    for key in ("envelopes", "direction_checks", "result_rates", "guard_bound"):
        if set(record.get(key, {})) != set(POPULATIONS) or any(set(record[key][population]) != set(CLAIMS) for population in POPULATIONS):
            raise PhaseError(f"the calibration record does not hold exactly the eight {key}")
    if not all(record["direction_checks"][population][claim]["ok"] for population in POPULATIONS for claim in CLAIMS):
        raise PhaseError("the calibration record carries a failed direction check")
    for population in POPULATIONS:
        for claim in CLAIMS:
            envelope = record["envelopes"][population][claim]
            bounds = [envelope["low"], envelope["high"]] if claim == "C3" else [envelope["bound"]]
            if any(bound is None for bound in bounds):
                raise PhaseError(f"the calibration record's {population}/{claim} envelope is not finite")
