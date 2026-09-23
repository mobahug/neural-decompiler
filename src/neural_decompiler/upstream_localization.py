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
