"""Experiment 023: prospective block-0 completion.

Implements design revision 2 (``5b38aba``) through implementation plan revision 1 (``3a795fb``). The completed program
``P1`` is Experiment 022's canonical "inputs only" coalition — the committed cue input ``ΔE``, the embedding change and
block 0's value and pattern terms at the cue position, block 0's single-logit update at the target position, the
committed reduced layers 1–2 (Experiment 017's chain, rows through ``p_c``) and the frozen Experiment 020 readout — and
``P0`` is 022's empty coalition (Level 0). Both are computed by 022's own functions, whose module is pinned by git blob
together with everything it calls; nothing here redefines them. The ceiling ``C`` (the frozen readout fed the measured
``Δx3``) is a comparator only and never enters ``P1``.

The primary statistic of each of the four conditions (Y1/Y2 × cue-final/coordinated) is

    g = (SSE0 − SSE1) / (SSE0 − SSEC)      (unclipped; g > 1 and g < 0 are valid values, never incidents)

pooled from per-pair sufficient statistics ``(n, S = Σy, Q = Σy², SSE0, SSE1, SSEC)``: for any selection of pairs
(a fresh group, each pair once, or a calibration draw, a repeated pair once per selection) ``SST = Q − S²/N`` — never a
sum of pair-local variances — cross-checked against the pooled two-pass identity (E6). One implementation
(``pool`` → ``statistics`` → ``classify``) serves the calibration and the confirmation alike.

Calibration reads only the committed exposed-cells artifact, extracted once, read-only, from Experiment 022's verified
calibration table (``extract``, identities E1–E6); nothing after ``extract`` opens that table.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
from neural_decompiler import upstream_localization as ul
from neural_decompiler.behavior import validate_json_safe

PhaseError = pm.PhaseError
IncidentError = pm.IncidentError

# ---------------------------------------------------------------------------
# Paths, design, plan.

EXPERIMENT = "023"
EXPERIMENT_DIR = "experiments/023-block0-completion"
CELLS_DATA_RELATIVE_PATH = f"{EXPERIMENT_DIR}/exposed-cells.f64"
CELLS_INDEX_RELATIVE_PATH = f"{EXPERIMENT_DIR}/exposed-cells.json"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
CALIBRATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/calibration-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREREGISTRATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration.md"
Y1_TABLE_RELATIVE_PATH = f"{EXPERIMENT_DIR}/locked-y1-table.f64"
Y1_TABLE_INDEX_RELATIVE_PATH = f"{EXPERIMENT_DIR}/locked-y1-table.json"
Y2_TABLE_OUTPUT = "outputs/experiment-023/y2-table.f64"
Y2_TABLE_INDEX_OUTPUT = "outputs/experiment-023/y2-table.json"
DESIGN = {"path": "docs/superpowers/specs/2026-09-24-experiment-023-block0-completion-design.md", "revision": 2, "commit": "5b38aba"}
PLAN = {"path": "docs/superpowers/plans/2026-09-24-experiment-023-block0-completion-plan.md", "revision": 1, "commit": "3a795fb"}

# ---------------------------------------------------------------------------
# The frozen dependencies, by git blob (sha1(b"blob <len>\0" + bytes), computed without git). Written out literally —
# never inherited from ``ul.FROZEN_BLOBS`` — so that a change anywhere in the reused Experiment 022 program, including
# 022's own module, makes every phase refuse rather than recompute a different "completed program".

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
    "upstream_localization.py": "465856962aa380747d1a4f1338d1d2762d03c9f9",
}
_FROZEN_MODULES = {"readout_decompilation.py": rd, "readout_calibration.py": rc, "head_pattern.py": hp, "frame_channels.py": fch, "attention_patterns.py": atp,
                   "layer_correction.py": lc, "plural_mechanism.py": pm, "encoding_read.py": er, "head_transport.py": ht, "models.py": models_module,
                   "upstream_localization.py": ul}

# Experiment 022's committed artifacts that 023 inherits (frozen digests) and the local table read by ``extract`` only.
TABLE_022_RELATIVE_PATH = "outputs/experiment-022/calibration-table.pt"
INHERITED_022 = {
    "calibration_file_sha256": "db745653cfa8f411514e80839bf986bd6cbbe3e816e3bb3095cf43c2d9d11a87",
    "calibration_content_sha256": "46985fd59d9d22dc08c9344a2fa96eacaf832dfe954e84526fc76c5ac0d12f6d",
    "confirmation_file_sha256": "1a9afef0791171129bcf3d716b41e2139857851088a30620481757409a2e04d2",
    "confirmation_content_sha256": "af848ae4bd48551cffec67f169a2a6d7b47f0eaaa5d2451010cc4c8bfd8fcda9",
    "table_file_sha256": "04659d5e8b20a70526e5862eb2b2de952b33496f25e3875de6c6254df2b8fd81",
}

# ---------------------------------------------------------------------------
# The statistic, the conditions and the results.

GROUPS = ("cue_final", "coordinated")
POPULATIONS = ("Y1", "Y2")
CONDITIONS = tuple(f"{population}/{group}" for population in POPULATIONS for group in GROUPS)
P0_MASK = 0
P1_MASK = {"cue_final": 14, "coordinated": 30}  # 022's canonical inputs-only coalition: emb + Bv + Bp (+ T), R off
FULL_MASK = {"cue_final": 15, "coordinated": 31}  # 022's full composition (layers 1–2 exact): the I3/I4 identity endpoint
GUARD_MIN = 0.90  # "recovers essentially all of the explainable upstream gap"
GAP_MIN = 0.02  # interpretable iff SST > 0 and SSE0 − SSEC ≥ 0.02 · SST
CEILING_LIMITED_R2 = 0.80  # descriptive flag only, no outcome force
RESULTS = ("NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS")
STATISTIC = "g = (SSE0 − SSE1) / (SSE0 − SSEC), pooled from per-pair sufficient statistics; unclipped"

# ---------------------------------------------------------------------------
# Calibration constants and tolerances.

B = 10_000
CROSS_CHECK_DRAWS = 16
DRAW_TAG = "023|primary"
TOLERANCES = {
    "E4": 1e-9,  # P1 recomputed from the weights against 022's stored P1, max abs
    "E5": 1e-10,  # cells against the per-noun table on the first draws, |k − d| / max(1, |d|)
    "E6": 1e-10,  # SST = Q − S²/N against the pooled two-pass identity, relative
    "kernel": 1e-10,  # the vectorized cell kernel against a direct loop / a direct per-noun recomputation
    "algebra": 1e-12,  # V + P = ΔA0(p_c); the closed-form T against the exact attention output at p_t
    "I1": 1e-4,  # block-0 decomposition against the measured Δx1, max abs
    "I3": 1e-4,  # the full composition's Δx̂3 against the measured Δx3, relative
    "I4": 1e-3,  # the full composition's Δĉ against the ceiling, max abs (nats)
    "I5": 0.0,  # P0 against the committed Level 0, exactly
}
CELL_COLUMNS = ("n", "S", "Q", "SSE0", "SSE1", "SSEC", "mean", "M2")
CELL_FORMULAS = {
    "n": "the number of scorable exposed nouns (79)", "S": "Σ y", "Q": "Σ y²", "SSE0": "Σ (y − P0)²", "SSE1": "Σ (y − P1)²", "SSEC": "Σ (y − C)²",
    "mean": "the two-pass mean of y (022's stored moment; the SST cross-check only)", "M2": "Σ (y − mean)² (022's stored moment; the SST cross-check only)",
}
CELLS_VERSION = "023-cells-v1"

# ---------------------------------------------------------------------------
# Fresh units (verbatim from design revision 2).

STRATA = ("determiner-like", "quantity", "adjective")
CUE_QUOTA = 8
FRAME_QUOTA = 6
CUE_CANDIDATES = {
    "determiner-like": ("general", "subsequent", "given", "chosen", "selected", "present", "ultimate", "preceding", "following", "latest", "earliest"),
    "quantity": ("tons", "piles", "masses", "stacks", "gross", "net", "scores", "batches", "bundles", "crowds", "herds", "multitude", "cumulative", "aggregate"),
    "adjective": ("humble", "proud", "shy", "lazy", "busy", "sturdy", "fragile", "shiny", "dusty", "hollow", "noble", "clever", "fuzzy", "crisp", "stiff", "tender",
                  "polished"),
}
FRAME_CANDIDATES = {
    "cardinal": ("The dairy produces {cue}", "The shelter feeds {cue}", "The boutique sells {cue}", "The hangar shelters {cue}", "The garage repairs {cue}",
                 "The nursery grows {cue}", "The warehouse stores {cue}", "The workshop carves {cue}", "The harbor ships {cue}", "The pharmacy stocks {cue}",
                 "The florist wraps {cue}", "The butcher cuts {cue}", "The tailor sews {cue}"),
    "quantifier": ("The timetable shows {cue}", "The glossary defines {cue}", "The inventory lists {cue}", "The chart shows {cue}", "The memo mentions {cue}",
                   "The journal notes {cue}", "The register records {cue}", "The index cites {cue}", "The brochure lists {cue}", "The logbook notes {cue}",
                   "The digest mentions {cue}", "The handbook names {cue}", "The spreadsheet counts {cue}", "The schedule lists {cue}"),
    "coordinated-adjective": ("Ella and Tomas stacked {cue} low", "Anna and Erik glazed {cue} smooth", "Leo and Maja folded {cue} flat",
                              "Lena and Jonas scrubbed {cue} clean", "Rosa and Felix polished {cue} bright", "Clara and Hans boiled {cue} soft",
                              "Mira and Sven chopped {cue} fine", "Ida and Lars packed {cue} tight", "Eva and Nils painted {cue} blue",
                              "Tina and Rolf baked {cue} brown", "Sofia and Emil pressed {cue} flat", "Maria and Jens brushed {cue} smooth"),
}
TEMPLATES = tuple(pm.TEMPLATE_ORDER)
FRAME_ID_TAG = "023"
COORDINATED = ul.COORDINATED
SCOPE = ("Experiment 023 prospectively tests completion across new determiner-like, quantity and adjective cues and new sentence frames only; it makes no "
         "claim about all grammatical-number cue classes and no new prospective claim for possessive or pronoun cues")
# The exposed spike's head ranking (cue-final ablation losses), frozen for the descriptive head-subset comparators.
HEAD_ORDER = (5, 0, 7, 6, 3, 2, 4, 1)
COMPARATORS = ("value_only", "relative_self_logit", "oracle_self_weight", "linear_response", "heads_4", "heads_6")


# ---------------------------------------------------------------------------
# Frozen dependencies and inherited inputs.


def module_blobs() -> dict[str, str]:
    return {name: rc.program_blob_sha1(Path(module.__file__)) for name, module in _FROZEN_MODULES.items()}


def assert_frozen_blobs() -> dict[str, str]:
    """Every module of the reused program — Experiment 022's module included — is byte for byte the pinned one."""
    actual = module_blobs()
    differing = sorted(name for name, blob in FROZEN_BLOBS.items() if actual.get(name) != blob)
    if differing:
        raise PhaseError(f"frozen modules changed: {differing}; Experiment 023 runs the pinned Experiment 022 program only")
    return actual


def own_blob() -> str:
    return rc.program_blob_sha1(Path(__file__))


def verify_022_inputs(root: Path) -> dict[str, str]:
    """Experiment 022's committed calibration record and confirmation file, each its frozen file and content digest."""
    out = {}
    for kind, relative in (("calibration", ul.CALIBRATION_RELATIVE_PATH), ("confirmation", ul.CONFIRMATION_RELATIVE_PATH)):
        path = root / relative
        if not path.exists() or rc.file_sha256(path) != INHERITED_022[f"{kind}_file_sha256"]:
            raise PhaseError(f"Experiment 022's committed {kind} file {relative} is missing or not the frozen file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("content_sha256") != rc.content_digest(payload) or payload["content_sha256"] != INHERITED_022[f"{kind}_content_sha256"]:
            raise PhaseError(f"Experiment 022's {kind} file does not verify against its frozen content digest")
        out[f"{kind}_022_file"] = INHERITED_022[f"{kind}_file_sha256"]
        out[f"{kind}_022_content"] = INHERITED_022[f"{kind}_content_sha256"]
    out["table_022_file"] = INHERITED_022["table_file_sha256"]
    return out


DIGEST_KEYS = (*rc.DIGEST_KEYS, "calibration_022_file", "calibration_022_content", "confirmation_022_file", "confirmation_022_content", "table_022_file")


def base_digests(inputs: ul.FrozenInputs, digests_022: Mapping[str, str]) -> dict[str, str]:
    return {**{key: inputs.digests[key] for key in rc.DIGEST_KEYS}, **dict(digests_022)}


# ---------------------------------------------------------------------------
# The exposed units in Experiment 022's canonical order (the order of its calibration table and of the artifact).


@dataclass(frozen=True)
class ExposedUnits:
    cues: tuple[tuple[str, int, str], ...]  # (word, token id, class): classes in 022's order, token id within a class
    frames: tuple[pm.Frame, ...]  # the 108 exposed frames by frame_id
    y2_like: Mapping[str, tuple[int, ...]]  # template -> frame indices of the 42 Y2-like frames (14 per template), by frame_id

    @property
    def n_pairs(self) -> int:
        return len(self.cues) * len(self.frames)

    def pair_index(self, cue: int, frame: int) -> int:
        """The flat, cue-major pair order: cue index × number of frames + frame index."""
        return cue * len(self.frames) + frame

    def group_frames(self, group: str) -> tuple[int, ...]:
        return tuple(i for i, frame in enumerate(self.frames) if (frame.template_id == COORDINATED) == (group == "coordinated"))

    def stratum_cues(self, stratum: str) -> tuple[int, ...]:
        return tuple(i for i, (_, _, cls) in enumerate(self.cues) if cls == stratum)

    def group_of(self, frame: int) -> str:
        return "coordinated" if self.frames[frame].template_id == COORDINATED else "cue_final"

    def to_json(self) -> dict[str, Any]:
        return {"cues": [[word, int(token_id), cls] for word, token_id, cls in self.cues],
                "frames": [[frame.frame_id, frame.template_id, self.group_of(i)] for i, frame in enumerate(self.frames)],
                "y2_like_frames": {template: [self.frames[i].frame_id for i in indices] for template, indices in self.y2_like.items()},
                "pair_order": "cue-major: flat index = cue index × number of frames + frame index; cues in 022's canonical order (classes in 022's order, token id "
                              "within a class), frames by frame_id"}


def exposed_units(inputs: ul.FrozenInputs) -> ExposedUnits:
    """Experiment 022's calibration units (021's provenance pools), rebuilt from the frozen inputs by 022's own
    ``units_from``; the counts only size 022's slots and do not affect the order."""
    pools = rc.production_pools(inputs.pool)
    counts = {"classes": {cls: 0 for cls in ul.CUE_CLASSES}, "templates": {template: 0 for template in ul.TEMPLATES}}
    units = ul.units_from(pools.cues, inputs.pool.frames, pools.frames_unscreened, counts)
    return ExposedUnits(tuple(units.cues), tuple(units.frames), {template: tuple(indices) for template, indices in units.y2_frames.items()})


# ---------------------------------------------------------------------------
# The canonical scoring path: per-pair sufficient statistics → pooling → statistics → classification. The calibration
# and the confirmation both use exactly these functions.


def pair_cells(y: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """One pair's eight cell values over its nouns, in ``CELL_COLUMNS`` order. The SSE and the two-pass moments are
    Experiment 022's own cell computation (``ul.pair_cells``); ``S`` and ``Q`` are plain float64 sums."""
    y = y.double()
    sse, count, mean, m2 = ul.pair_cells(y, torch.stack([p0.double(), p1.double(), c.double()]))
    return torch.tensor([count, float(y.sum()), float((y * y).sum()), float(sse[0]), float(sse[1]), float(sse[2]), mean, m2], dtype=torch.float64)


def pool(cells: torch.Tensor, index: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pool the cells selected by ``index`` (``[K]`` or ``[D, K]``; a repeated index counts once per selection) over its
    last axis: ``N``, ``S``, ``Q``, the three SSE, ``SST = Q − S²/N`` and, for the E6 cross-check only, the pooled
    two-pass identity ``Σ M2 + Σ n (mean − S/N)²``."""
    selected = cells[index]
    n, s, q = selected[..., 0].sum(-1), selected[..., 1].sum(-1), selected[..., 2].sum(-1)
    grand = s / n
    two_pass = selected[..., 7].sum(-1) + (selected[..., 0] * (selected[..., 6] - grand.unsqueeze(-1)) ** 2).sum(-1)
    return {"N": n, "S": s, "Q": q, "SSE0": selected[..., 3].sum(-1), "SSE1": selected[..., 4].sum(-1), "SSEC": selected[..., 5].sum(-1), "SST": q - s * s / n,
            "SST_two_pass": two_pass}


def e6_error(pooled: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """|SST − two-pass| relative to the two-pass value (absolute where that is zero)."""
    difference = (pooled["SST"] - pooled["SST_two_pass"]).abs()
    scale = pooled["SST_two_pass"].abs()
    return torch.where(scale > 0, difference / torch.where(scale > 0, scale, torch.ones_like(scale)), difference)


def enforce_e6(pooled: Mapping[str, torch.Tensor], where: str) -> float:
    worst = float(e6_error(pooled).max()) if pooled["SST"].numel() else 0.0
    if not worst <= TOLERANCES["E6"]:
        raise IncidentError(f"E6 failed at {where}: SST = Q − S²/N differs from the pooled two-pass identity by {worst:.3e} (relative) against {TOLERANCES['E6']:.0e}")
    return worst


def statistics(pooled: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``g`` (unclipped; NaN where undefined), the interpretability flag, the gap ``(SSE0 − SSEC)/SST``, the three
    ``R²`` and the descriptive ceiling-limited flag. Undefined iff ``SST ≤ 0`` or ``SSE0 − SSEC < 0.02 · SST``."""
    sst, sse0, sse1, ssec = (pooled[key].double() for key in ("SST", "SSE0", "SSE1", "SSEC"))
    positive = sst > 0.0
    safe = torch.where(positive, sst, torch.ones_like(sst))
    explainable = sse0 - ssec
    defined = positive & torch.isfinite(explainable) & (explainable >= GAP_MIN * sst)
    g = torch.where(defined, (sse0 - sse1) / torch.where(defined, explainable, torch.ones_like(explainable)), torch.full_like(sst, math.nan))
    nan = torch.full_like(sst, math.nan)
    r2 = {k: torch.where(positive, 1.0 - value / safe, nan) for k, value in (("R2_0", sse0), ("R2_1", sse1), ("R2_C", ssec))}
    return {"g": g, "defined": defined, "gap": torch.where(positive, explainable / safe, nan), **r2, "ceiling_limited": positive & (r2["R2_C"] < CEILING_LIMITED_R2),
            "SST": sst}


def lower_rank(draws: int) -> int:
    """⌈0.025·B⌉ in integer arithmetic: 250 at B = 10,000 (zero-based element [249])."""
    return (25 * int(draws) + 999) // 1000


def lower_bound(values: torch.Tensor, defined: torch.Tensor) -> dict[str, Any]:
    """The lower envelope: the ``lower_rank``-th ascending value with undefined values placed at −∞."""
    rank = lower_rank(int(values.shape[0]))
    return {"kind": "lower", "rank": rank, "element": rank - 1, "bound": ul.order_statistic(values, defined, rank, undefined_at=-math.inf)}


def direction_check(envelope: Mapping[str, Any], values: torch.Tensor, defined: torch.Tensor) -> dict[str, Any]:
    median = ul.defined_median(values, defined)
    return {"ok": bool(median is not None and envelope["bound"] <= median), "median": median}


def classify(g: float | None, defined: bool, bound: float) -> str:
    """``NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS``: the only function that decides a result.
    The effective requirement of a PASS is ``g ≥ max(bound, 0.90)``; ``g > 1`` is an ordinary value."""
    if not defined:
        return "NOT_INTERPRETABLE"
    if g is None or not math.isfinite(float(g)):
        raise IncidentError("a non-finite g where the gap rule holds")
    if float(g) < GUARD_MIN:
        return "GUARD_FAILURE"
    if float(g) < float(bound):
        return "ENVELOPE_ONLY_FAILURE"
    return "PASS"


def score_selection(cells: torch.Tensor, index: torch.Tensor, bound: float | None, where: str) -> dict[str, Any]:
    """One selection of pairs (a fresh condition): pooled, E6-checked, its statistics and, given a bound, its result."""
    pooled = pool(cells, index)
    e6 = enforce_e6(pooled, where)
    stats = statistics(pooled)
    defined = bool(stats["defined"])
    g = float(stats["g"]) if defined else None
    out = {"N": float(pooled["N"]), "SST": float(pooled["SST"]), "SSE0": float(pooled["SSE0"]), "SSE1": float(pooled["SSE1"]), "SSEC": float(pooled["SSEC"]),
           "g": g, "interpretable": defined, "gap": _float_or_none(stats["gap"]), "R2_0": _float_or_none(stats["R2_0"]), "R2_1": _float_or_none(stats["R2_1"]),
           "R2_C": _float_or_none(stats["R2_C"]), "ceiling_limited": bool(stats["ceiling_limited"]), "e6": e6}
    if bound is not None:
        out["result"] = classify(g, defined, bound)
    return out


def _float_or_none(value: torch.Tensor) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# SHA-indexed draws (022's formula with 023's tag).


def slot_index(b: int, stratum: str, slot: int, n: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{DRAW_TAG}|{b}|{stratum}|{slot}".encode("utf-8")).digest()[:8], "big") % int(n)


def draw_slots() -> dict[str, int]:
    return {**{f"cue/{stratum}": CUE_QUOTA for stratum in STRATA}, **{f"frame/{template}": FRAME_QUOTA for template in TEMPLATES}}


def draw_indices(units: ExposedUnits, draws: int) -> dict[str, torch.Tensor]:
    """Per stratum ``[draws, slots]`` indices into that stratum's frozen order (cues by token id, Y2-like frames by
    ``frame_id``)."""
    sizes = {**{f"cue/{stratum}": len(units.stratum_cues(stratum)) for stratum in STRATA}, **{f"frame/{template}": len(units.y2_like[template]) for template in TEMPLATES}}
    slots = draw_slots()
    if any(size == 0 for size in sizes.values()):
        raise PhaseError(f"an empty calibration stratum: {sizes}")
    return {stratum: torch.tensor([[slot_index(b, stratum, i, n) for i in range(slots[stratum])] for b in range(draws)], dtype=torch.int64) for stratum, n in sizes.items()}


def draw_index_digests(indices: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {stratum: rc.tensor_digest(values) for stratum, values in sorted(indices.items())}


def draw_pairs(units: ExposedUnits, indices: Mapping[str, torch.Tensor], population: str, group: str, rows: Sequence[int] | range) -> torch.Tensor:
    """``[D, K]`` flat pair indices of draws ``rows`` for a population and group, repeats included: Y1-like — the
    drawn cues × the group's exposed frames; Y2-like — the drawn cues × the group's drawn Y2-like frames."""
    rows = torch.as_tensor(list(rows), dtype=torch.int64)
    cues = torch.cat([torch.tensor(units.stratum_cues(stratum), dtype=torch.int64)[indices[f"cue/{stratum}"][rows]] for stratum in STRATA], dim=1)  # [D, 24]
    n_frames = len(units.frames)
    if population == "Y1":
        frames = torch.tensor(units.group_frames(group), dtype=torch.int64).unsqueeze(0).expand(len(rows), -1)
    elif population == "Y2":
        templates = [template for template in TEMPLATES if (template == COORDINATED) == (group == "coordinated")]
        frames = torch.cat([torch.tensor(units.y2_like[template], dtype=torch.int64)[indices[f"frame/{template}"][rows]] for template in templates], dim=1)
    else:
        raise ValueError(f"unknown population {population}")
    return (cues.unsqueeze(2) * n_frames + frames.unsqueeze(1)).reshape(len(rows), -1)


# ---------------------------------------------------------------------------
# The exposed-cells artifact: 022's table byte format (raw little-endian float64 with a canonical-JSON index).


def cells_source(record_022: Mapping[str, Any]) -> dict[str, Any]:
    """What the artifact binds of Experiment 022: the local table's file digest and tensor digests (from 022's committed
    record) and the record's file and content digests."""
    return {"table_path": TABLE_022_RELATIVE_PATH, "table_file_sha256": INHERITED_022["table_file_sha256"],
            "table_tensor_sha256": dict(record_022["rematerialization"]["table_sha256"]), "calibration_record_path": ul.CALIBRATION_RELATIVE_PATH,
            "calibration_record_file_sha256": INHERITED_022["calibration_file_sha256"], "calibration_record_content_sha256": INHERITED_022["calibration_content_sha256"],
            "confirmation_022_content_sha256": INHERITED_022["confirmation_content_sha256"]}


def cells_meta(units: ExposedUnits, noun_keys: Sequence[str], record_022: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the artifact's index binds besides its bytes, all of it recomputable from the frozen inputs and 022's
    committed record: the columns and their formulas, the canonical pair order with every cue's id and stratum and
    every frame's id, template and group, the Y2-like frames, the nouns, the program's masks and 022's digests."""
    return {"experiment": EXPERIMENT, "kind": "exposed-cells", "schema_version": 1, "version": CELLS_VERSION, "design": dict(DESIGN), "columns": list(CELL_COLUMNS),
            "column_formulas": dict(CELL_FORMULAS), **units.to_json(), "nouns": list(noun_keys),
            "program": {"P0_mask": P0_MASK, "P1_mask": dict(P1_MASK), "ceiling": "the frozen Experiment 020 readout fed the measured Δx3 (022's stored ceiling)"},
            "source_022": cells_source(record_022)}


def cells_layout(units: ExposedUnits) -> list[dict[str, Any]]:
    return [{"name": "cells", "shape": [units.n_pairs, len(CELL_COLUMNS)]}]


def write_cells(data_path: Path, index_path: Path, cells: torch.Tensor, meta: Mapping[str, Any], extraction: Mapping[str, Any]) -> dict[str, Any]:
    """Write the artifact once, in 022's byte format; the index carries ``meta`` and the extraction record."""
    return ul.write_table(Path(data_path), Path(index_path), [("cells", cells.double())], {**dict(meta), "extraction": dict(extraction)})


def read_cells(data_path: Path, index_path: Path, *, units: ExposedUnits, meta: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    """The artifact re-read and verified: the byte format, the layout, every bound meta field against the one recomputed
    now, and the bytes against the index's digest. A mismatch refuses (``PhaseError``)."""
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    ul.verify_table_index(index, layout=cells_layout(units), meta=meta, error=PhaseError)
    try:
        blocks = ul.read_table(Path(data_path), index)
    except IncidentError as error:
        raise PhaseError(f"the exposed-cells artifact does not match its index: {error}") from None
    return blocks["cells"], index


# ---------------------------------------------------------------------------
# Extraction (once; read-only on Experiment 022's table; weights only): identities E1–E6.


class ExtractionMismatch(PhaseError):
    """E1–E3: the local table is not the one 022's record binds, or the cells do not reproduce 022's stored cells.
    Nothing is written and the phase stops for review (not an incident)."""


def verify_table_022(table: Mapping[str, Any], record_022: Mapping[str, Any], units: ExposedUnits, noun_keys: Sequence[str]) -> dict[str, Any]:
    """E1: every tensor of the local table against the digest in 022's committed record, and its pair and noun orders
    against the canonical ones rebuilt from the frozen inputs."""
    bound = dict(record_022["rematerialization"]["table_sha256"])
    actual = {name: rc.tensor_digest(table[name]) for name in ul.CalibrationTable.TENSORS}
    actual.update({f"gate_{name}": rc.tensor_digest(values) for name, values in sorted(table["gates"].items())})
    if actual != bound:
        raise ExtractionMismatch(f"E1: the local table's tensors are not the ones 022's record binds: {sorted(key for key in set(actual) | set(bound) if actual.get(key) != bound.get(key))}")
    orders = {"cues": [word for word, _, _ in units.cues], "token_ids": [int(token) for _, token, _ in units.cues], "classes": [cls for _, _, cls in units.cues],
              "frames": [frame.frame_id for frame in units.frames], "templates": [frame.template_id for frame in units.frames], "noun_keys": list(noun_keys)}
    wrong = sorted(key for key, value in orders.items() if list(table[key]) != value)
    if wrong:
        raise ExtractionMismatch(f"E1: the local table's orders differ from the canonical ones: {wrong}")
    return {"tensors": sorted(bound), "orders": sorted(orders), "passed": True}


def cells_from_table(table: Mapping[str, Any], units: ExposedUnits) -> torch.Tensor:
    """Every exposed pair's cells in the canonical flat order: ``y`` = the stored ``Δc``, ``P0`` = the stored mask-0
    coalition, ``P1`` = the stored mask 14 / 30, ``C`` = the stored ceiling."""
    dc, dc_hat, ceiling = table["dc"], table["dc_hat"], table["ceiling"]
    cells = torch.empty(units.n_pairs, len(CELL_COLUMNS), dtype=torch.float64)
    for fi in range(len(units.frames)):
        mask = P1_MASK[units.group_of(fi)]
        for ci in range(len(units.cues)):
            cells[units.pair_index(ci, fi)] = pair_cells(dc[ci, fi], dc_hat[ci, fi, P0_MASK], dc_hat[ci, fi, mask], ceiling[ci, fi])
    return cells


def verify_cells_against_table(cells: torch.Tensor, table: Mapping[str, Any], units: ExposedUnits) -> dict[str, Any]:
    """E2: ``SSE0``, ``SSE1``, ``n``, ``mean`` and ``M2`` equal 022's stored per-pair cells bit for bit; E3: the cells,
    computed a second time, are bit-identical."""
    n_pairs = units.n_pairs
    sse = table["sse"].reshape(n_pairs, -1)
    p1 = torch.tensor([P1_MASK[units.group_of(i % len(units.frames))] for i in range(n_pairs)], dtype=torch.int64)
    expected = {"n": table["count"].reshape(-1), "SSE0": sse[:, P0_MASK], "SSE1": sse[torch.arange(n_pairs), p1], "mean": table["mean"].reshape(-1),
                "M2": table["m2"].reshape(-1)}
    wrong = sorted(name for name, values in expected.items() if not torch.equal(cells[:, CELL_COLUMNS.index(name)], values.double()))
    if wrong:
        raise ExtractionMismatch(f"E2: the cells' {wrong} are not 022's stored cells bit for bit")
    if not torch.equal(cells_from_table(table, units), cells):
        raise ExtractionMismatch("E3: the cells computed a second time are not bit-identical")
    return {"E2": {"columns": sorted(expected), "passed": True}, "E3": {"passed": True}}


def recompute_p1(progs: ul.ModelPrograms, inputs: ul.FrozenInputs, units: ExposedUnits, table: Mapping[str, Any], *, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """E4: ``P1`` recomputed for every exposed pair from the weights and 020's locked states by 022's own functions,
    against the table's stored ``P1``. No prompt, no forward pass."""
    say = log or (lambda message: None)
    locked = inputs.closure["exploration"]["locked_states"]
    worst = {"max": 0.0, "at": ""}
    for fi, frame in enumerate(units.frames):
        state = rd.state_from_locked(locked[frame.frame_id], frame)
        rows16 = ul.reference_rows_017(progs.programs, state)
        reference_id = int(inputs.pool.reference_ids[frame.template_id])
        mask = P1_MASK[units.group_of(fi)]
        for ci, (word, token_id, _) in enumerate(units.cues):
            ctx = ul.pair_context(progs, frame, state, reference_id, int(token_id), word, rows16)
            p1 = ul.contrast_of(progs, state, ul.compose_dx3(progs, ctx, mask))
            ul._worse(worst, float((p1 - table["dc_hat"][ci, fi, mask].double()).abs().max()), f"{word}|{frame.frame_id}")
        if (fi + 1) % 12 == 0:
            say(f"  E4: {fi + 1}/{len(units.frames)} frames recomputed; max |ΔP1| so far {worst['max']}")
    worst["passed"] = worst.get("max") is not None and worst["max"] <= TOLERANCES["E4"]
    if not worst["passed"]:
        raise IncidentError(f"E4 failed: P1 recomputed from the weights differs from 022's stored P1 by {worst['max']} at {worst['at']} against {TOLERANCES['E4']:.0e}")
    return rc.json_safe(worst)


def direct_selection(table: Mapping[str, Any], units: ExposedUnits, pairs: torch.Tensor) -> dict[str, float]:
    """A selection's statistics recomputed directly from the per-noun table (the independent side of E5)."""
    n_nouns = table["dc"].shape[-1]
    flat = {"y": table["dc"].reshape(-1, n_nouns), "c": table["ceiling"].reshape(-1, n_nouns)}
    dc_hat = table["dc_hat"].reshape(units.n_pairs, -1, n_nouns)
    p1_mask = torch.tensor([P1_MASK[units.group_of(int(i) % len(units.frames))] for i in pairs], dtype=torch.int64)
    y = flat["y"][pairs].double().reshape(-1)
    p0 = dc_hat[pairs, P0_MASK].double().reshape(-1)
    p1 = dc_hat[pairs, p1_mask].double().reshape(-1)
    c = flat["c"][pairs].double().reshape(-1)
    sst = float(((y - y.mean()) ** 2).sum())
    sse0, sse1, ssec = (float(((y - k) ** 2).sum()) for k in (p0, p1, c))
    defined = sst > 0 and sse0 - ssec >= GAP_MIN * sst
    return {"SST": sst, "g": (sse0 - sse1) / (sse0 - ssec) if defined else math.nan, "gap": (sse0 - ssec) / sst if sst > 0 else math.nan}


def agreement(kernel: float, direct: float) -> float:
    if math.isnan(kernel) and math.isnan(direct):
        return 0.0
    if math.isnan(kernel) != math.isnan(direct):
        return math.inf
    return abs(kernel - direct) / max(1.0, abs(direct))


def verify_draws_against_table(cells: torch.Tensor, table: Mapping[str, Any], units: ExposedUnits, draws: int = CROSS_CHECK_DRAWS) -> dict[str, Any]:
    """E5 and E6: on the first ``draws`` calibration draws of every population and group, ``SST``, ``g`` and the gap
    from the cells against a direct recomputation from the per-noun table; ``SST`` against the pooled two-pass identity
    on every pair and every one of those draws."""
    per_pair = enforce_e6(pool(cells, torch.arange(units.n_pairs).unsqueeze(1)), "a single exposed pair")
    indices = draw_indices(units, draws)
    worst = {"max_difference": 0.0, "at": "", "n_checked": 0}
    e6_draws = 0.0
    for population in POPULATIONS:
        for group in GROUPS:
            pairs = draw_pairs(units, indices, population, group, range(draws))
            pooled = pool(cells, pairs)
            e6_draws = max(e6_draws, enforce_e6(pooled, f"{population}/{group} draws"))
            stats = statistics(pooled)
            for b in range(draws):
                direct = direct_selection(table, units, pairs[b])
                for name, value in (("SST", float(pooled["SST"][b])), ("g", float(stats["g"][b])), ("gap", float(stats["gap"][b]))):
                    difference = agreement(value, direct[name])
                    worst["n_checked"] += 1
                    if difference > worst["max_difference"]:
                        worst.update({"max_difference": difference, "at": f"{population}/{group}/draw {b}/{name}"})
    worst["passed"] = worst["max_difference"] <= TOLERANCES["E5"]
    if not worst["passed"]:
        raise IncidentError(f"E5 failed: the cells disagree with the per-noun table by {worst['max_difference']} at {worst['at']} against {TOLERANCES['E5']:.0e}")
    return {"E5": rc.json_safe(worst), "E6": {"max_per_pair": per_pair, "max_per_draw": e6_draws, "tolerance": TOLERANCES["E6"], "passed": True}}


def extract_cells(table: Mapping[str, Any], record_022: Mapping[str, Any], units: ExposedUnits, noun_keys: Sequence[str], *,
                  recompute: Callable[[], Mapping[str, Any]], log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The extraction in its frozen order — E1, the cells, E2 and E3, E4 (``recompute``: P1 from the weights), E5 and
    E6 — returning the cells, the index meta and every check. Any failure raises before anything is written."""
    say = log or (lambda message: None)
    checks: dict[str, Any] = {"E1": verify_table_022(table, record_022, units, noun_keys)}
    say(f"E1: {len(checks['E1']['tensors'])} tensor digests and the orders verified")
    cells = cells_from_table(table, units)
    checks.update(verify_cells_against_table(cells, table, units))
    say("E2, E3: the cells reproduce 022's stored cells bit for bit, twice")
    checks["E4"] = dict(recompute())
    say(f"E4: P1 recomputed from the weights, max |Δ| {checks['E4']['max']}")
    checks.update(verify_draws_against_table(cells, table, units))
    say(f"E5: {checks['E5']['n_checked']} draw quantities, max {checks['E5']['max_difference']:.2e}; E6: pairs {checks['E6']['max_per_pair']:.2e}, "
        f"draws {checks['E6']['max_per_draw']:.2e}")
    return {"cells": cells, "meta": cells_meta(units, noun_keys, record_022), "checks": rc.json_safe(checks)}


# ---------------------------------------------------------------------------
# The freeze: tokenizer only, structural rules only; no model output (Task 4).

CONFIRMATION_SCHEMA_VERSION = 1


class FreezeShortfall(ul.FreezeShortfall):
    """A stratum or a template yields fewer than its quota: the freeze writes nothing and stops for review (not an
    incident; no stratum is refilled from another and there is no runtime fallback)."""


def exclusion(inputs: ul.FrozenInputs, confirmation_022: Mapping[str, Any], confirmation_022_file_sha256: str) -> dict[str, Any]:
    """Experiment 022's freeze exclusion (every cue token and frame text of 020's pool, the 006–020 confirmations and
    the extension, read from their structured fields by 022's own ``extract_exclusion``) plus 022's own 24 frozen cues,
    18 frame texts and their frames' cue ids."""
    base = ul.extract_exclusion(inputs)
    ids = set(base["cue_token_ids"]) | {int(cue["token_id"]) for cue in confirmation_022["cues"]}
    ids |= {int(token_id) for frame in confirmation_022["frames"] for token_id in frame["cue_ids"].values()}
    texts = set(base["frame_texts"]) | {frame["text_template"] for frame in confirmation_022["frames"]}
    source = {"source": "confirmation-022", "loader": "json: Experiment 022's committed confirmation file",
              "files": [{"path": ul.CONFIRMATION_RELATIVE_PATH, "file_sha256": confirmation_022_file_sha256}], "content_sha256": confirmation_022["content_sha256"],
              "tokens": len(confirmation_022["cues"]), "frames": len(confirmation_022["frames"])}
    ordered_ids, ordered_texts = sorted(ids), sorted(texts)
    return {"cue_token_ids": ordered_ids, "cue_token_ids_sha256": pm.sha256_text(pm.canonical_json(ordered_ids)), "frame_texts": ordered_texts,
            "frame_texts_sha256": pm.sha256_text(pm.canonical_json(ordered_texts)), "sources": list(base["sources"]) + [source]}


@dataclass(frozen=True)
class Confirmation023:
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
        return {"classes": {stratum: sum(1 for token in self.tokens if token["class"] == stratum) for stratum in STRATA},
                "templates": {template: sum(1 for frame in self.frames if frame.template_id == template) for template in TEMPLATES}}


def freeze_payload(tokenizer: Any, inputs: ul.FrozenInputs, confirmation_022: Mapping[str, Any], confirmation_022_file_sha256: str, *,
                   model: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The first eligible entries of design revision 2's ordered lists: a cue when it is a single token with a leading
    space and its id is new; a frame when its text is new and 022's structural rules accept it. A shortfall writes
    nothing (``FreezeShortfall``)."""
    excluded = exclusion(inputs, confirmation_022, confirmation_022_file_sha256)
    excluded_ids, excluded_texts = set(excluded["cue_token_ids"]), set(excluded["frame_texts"])
    cues: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for stratum, words in CUE_CANDIDATES.items():
        taken = 0
        for rank, word in enumerate(words):
            if taken == CUE_QUOTA:
                break
            ids = pm._encode(tokenizer, " " + word)
            if len(ids) != 1:
                rejected.append({"kind": "cue", "class": stratum, "candidate": word, "rank": rank, "reason": f"{len(ids)} tokens with a leading space"})
                continue
            if ids[0] in excluded_ids or ids[0] in {entry["token_id"] for entry in cues}:
                rejected.append({"kind": "cue", "class": stratum, "candidate": word, "rank": rank, "reason": f"token id {ids[0]} already used"})
                continue
            cues.append({"word": word, "token_id": int(ids[0]), "class": stratum, "candidate_rank": rank})
            taken += 1
        if taken < CUE_QUOTA:
            raise FreezeShortfall(f"cue stratum {stratum}: {taken} eligible of {CUE_QUOTA}")
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
            low, high = ul.P_C_RANGE[template]
            if not low <= frame.p_c <= high:
                rejected.append({"kind": "frame", "template": template, "candidate": text, "rank": rank, "reason": f"p_c {frame.p_c} outside {low}–{high}"})
                continue
            if frame.p_t != frame.p_c + (1 if template == COORDINATED else 0):
                rejected.append({"kind": "frame", "template": template, "candidate": text, "rank": rank, "reason": f"p_t {frame.p_t} against p_c {frame.p_c}"})
                continue
            frames.append({**frame.to_dict(), "candidate_rank": rank})
            taken += 1
        if taken < FRAME_QUOTA:
            raise FreezeShortfall(f"template {template}: {taken} structurally eligible of {FRAME_QUOTA}")
    payload = {
        "experiment": EXPERIMENT, "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "kind": "the new cues and frames of Experiment 023, frozen from the tokenizer and structural rules alone; no model output",
        "design": dict(DESIGN), "plan": dict(PLAN), "scope": SCOPE,
        "model": dict(model or {"model_id": models_module.PYTHIA_70M.model_id, "revision": models_module.PYTHIA_70M.revision}),
        "candidates": {"cues": {stratum: list(words) for stratum, words in CUE_CANDIDATES.items()}, "frames": {template: list(texts) for template, texts in FRAME_CANDIDATES.items()}},
        "rules": {"cue_quota": CUE_QUOTA, "frame_quota": FRAME_QUOTA, "p_c_range": {template: list(bounds) for template, bounds in ul.P_C_RANGE.items()},
                  "cue": "a single token with a leading space whose id no earlier experiment (022 included) used",
                  "frame": "a new text accepted by pm._build_new_frame; coordinated: one adjective token"},
        "exclusion": excluded, "reference_cue_ids": {template: int(token_id) for template, token_id in pool.reference_ids.items()},
        "exposed_frame_ids": [frame.frame_id for frame in pool.frames],
        "cues": cues, "frames": frames, "rejected": rejected,
        "counts": {"classes": {stratum: sum(1 for entry in cues if entry["class"] == stratum) for stratum in STRATA},
                   "templates": {template: sum(1 for entry in frames if entry["template_id"] == template) for template in TEMPLATES}},
    }
    payload["manifest"] = confirmation_from_payload(payload, pool, verify=False).manifest()
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def confirmation_from_payload(payload: Mapping[str, Any], pool: Any, *, verify: bool = True) -> Confirmation023:
    frames = tuple(pm.Frame(entry["template_id"], entry["frame_id"], tuple(entry["prefix_ids"]), tuple(entry["suffix_ids"]), entry["cue_ids"], entry["text_template"],
                            origin="extension") for entry in payload["frames"])
    tokens = tuple({"word": entry["word"], "token_id": int(entry["token_id"]), "class": entry["class"]} for entry in payload["cues"])
    confirmation = Confirmation023(dict(pool.reference_ids), frames, tuple(pool.frames), tokens, str(payload.get("content_sha256", "")))
    if verify:
        if payload.get("experiment") != EXPERIMENT or payload.get("schema_version") != CONFIRMATION_SCHEMA_VERSION:
            raise PhaseError("not an Experiment 023 confirmation file")
        if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
            raise PhaseError("the confirmation file's content digest does not verify")
        if payload["manifest"] != confirmation.manifest():
            raise PhaseError("the confirmation file's manifest is not the one its cues and frames define")
        if confirmation.counts() != {"classes": {stratum: CUE_QUOTA for stratum in STRATA}, "templates": {template: FRAME_QUOTA for template in TEMPLATES}}:
            raise PhaseError(f"the confirmation file's composition is not {CUE_QUOTA}/{CUE_QUOTA}/{CUE_QUOTA} cues and 6/6/6 frames: {confirmation.counts()}")
        if payload.get("counts") != confirmation.counts():
            raise PhaseError("the confirmation file's recorded counts are not those of its cues and frames (the calibration reads these counts)")
        if payload["exposed_frame_ids"] != [frame.frame_id for frame in pool.frames] or dict(payload["reference_cue_ids"]) != {k: int(v) for k, v in pool.reference_ids.items()}:
            raise PhaseError("the confirmation file names a different exposed pool")
    return confirmation


def load_confirmation_023(path: Path, inputs: ul.FrozenInputs, confirmation_022: Mapping[str, Any], confirmation_022_file_sha256: str) -> Confirmation023:
    """The committed freeze artifact, verified: digest, manifest, composition, and no overlap with anything used before
    (the exclusion re-extracted from the frozen inputs and 022's confirmation now)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    confirmation = confirmation_from_payload(payload, inputs.pool)
    excluded = exclusion(inputs, confirmation_022, confirmation_022_file_sha256)
    if payload["exclusion"]["cue_token_ids_sha256"] != excluded["cue_token_ids_sha256"] or payload["exclusion"]["frame_texts_sha256"] != excluded["frame_texts_sha256"]:
        raise PhaseError("the exclusion sets recorded at the freeze differ from those of the frozen inputs now")
    if payload["exclusion"]["sources"] != excluded["sources"]:
        raise PhaseError("the exclusion sources recorded at the freeze differ from those of the frozen inputs now")
    used_ids, used_texts = set(excluded["cue_token_ids"]), set(excluded["frame_texts"])
    if {int(token["token_id"]) for token in confirmation.tokens} & used_ids or {frame.text_template for frame in confirmation.frames} & used_texts:
        raise PhaseError("a new cue or frame was used by an earlier experiment")
    return confirmation
