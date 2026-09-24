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
(``pool`` → ``statistics`` → ``result_codes``) serves the calibration and the confirmation alike.

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


def result_codes(values: torch.Tensor, defined: torch.Tensor, bound: float) -> torch.Tensor:
    """The four-way classification, as indices into ``RESULTS``: the only code that decides a result — a fresh
    condition's (``classify``), every calibration draw's (the rates) and the joint rate alike. Precedence
    ``NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS``: an undefined value is NOT_INTERPRETABLE; else
    ``g < 0.90`` is a GUARD_FAILURE; else ``g < bound`` an ENVELOPE_ONLY_FAILURE; else a PASS. The effective requirement
    of a PASS is ``g ≥ max(bound, 0.90)``; ``g > 1`` is an ordinary value. A non-finite ``g`` where the gap rule holds,
    or a NaN bound, is an incident."""
    values = torch.as_tensor(values, dtype=torch.float64)
    defined = torch.as_tensor(defined, dtype=torch.bool)
    if math.isnan(float(bound)):
        raise IncidentError("a NaN envelope bound")
    if bool((defined & ~torch.isfinite(values)).any()):
        raise IncidentError("a non-finite g where the gap rule holds")
    codes = torch.full(values.shape, RESULTS.index("PASS"), dtype=torch.int64)
    codes[values < float(bound)] = RESULTS.index("ENVELOPE_ONLY_FAILURE")
    codes[values < GUARD_MIN] = RESULTS.index("GUARD_FAILURE")
    codes[~defined] = RESULTS.index("NOT_INTERPRETABLE")
    return codes


def classify(g: float | None, defined: bool, bound: float) -> str:
    """One condition's result: ``result_codes`` on a single value (``None`` where undefined)."""
    codes = result_codes(torch.tensor([math.nan if g is None else float(g)], dtype=torch.float64), torch.tensor([bool(defined)]), bound)
    return RESULTS[int(codes[0])]


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
    """Write the artifact once, in 022's byte format (``ul.table_bytes`` and ``ul.table_index``); the index carries
    ``meta``, the extraction record and its own ``content_sha256`` (canonical JSON without that key)."""
    blocks = [("cells", cells.double())]
    data = ul.table_bytes(blocks)
    index = ul.table_index(blocks, data, {**dict(meta), "extraction": dict(extraction)})
    index["content_sha256"] = rc.content_digest(index)
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    Path(data_path).write_bytes(data)
    Path(index_path).write_text(pm.canonical_json(index) + "\n", encoding="utf-8")
    return index


def read_cells(data_path: Path, index_path: Path, *, units: ExposedUnits, meta: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    """The artifact re-read and verified: the index against its content digest, the byte format, the layout, every
    bound meta field against the one recomputed now, and the bytes against the index's digest. A mismatch refuses
    (``PhaseError``)."""
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if index.get("content_sha256") != rc.content_digest(index):
        raise PhaseError("the exposed-cells index does not verify against its content digest")
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


def independent_sums(table: Mapping[str, Any], units: ExposedUnits) -> torch.Tensor:
    """E3's second computation of ``S``, ``Q`` and ``SSEC`` for every pair (columns in that order, canonical flat pair
    order): whole-table float64 reductions over the noun axis, independent of the per-pair cell path."""
    dc, ceiling = table["dc"].double(), table["ceiling"].double()
    return torch.stack([dc.sum(-1), (dc * dc).sum(-1), ((dc - ceiling) ** 2).sum(-1)], dim=-1).reshape(units.n_pairs, 3)


E3_COLUMNS = ("S", "Q", "SSEC")


def verify_cells_against_table(cells: torch.Tensor, table: Mapping[str, Any], units: ExposedUnits) -> dict[str, Any]:
    """E2: ``SSE0``, ``SSE1``, ``n``, ``mean`` and ``M2`` equal 022's stored per-pair cells bit for bit; E3: ``S``,
    ``Q`` and ``SSEC`` — which 022 did not store — equal an independent second computation from the stored ``Δc`` and
    ceiling (``independent_sums``) bit for bit."""
    n_pairs = units.n_pairs
    sse = table["sse"].reshape(n_pairs, -1)
    p1 = torch.tensor([P1_MASK[units.group_of(i % len(units.frames))] for i in range(n_pairs)], dtype=torch.int64)
    expected = {"n": table["count"].reshape(-1), "SSE0": sse[:, P0_MASK], "SSE1": sse[torch.arange(n_pairs), p1], "mean": table["mean"].reshape(-1),
                "M2": table["m2"].reshape(-1)}
    wrong = sorted(name for name, values in expected.items() if not torch.equal(cells[:, CELL_COLUMNS.index(name)], values.double()))
    if wrong:
        raise ExtractionMismatch(f"E2: the cells' {wrong} are not 022's stored cells bit for bit")
    independent = independent_sums(table, units)
    differing = [name for k, name in enumerate(E3_COLUMNS) if not torch.equal(cells[:, CELL_COLUMNS.index(name)], independent[:, k])]
    if differing:
        raise ExtractionMismatch(f"E3: the cells' {differing} are not bit-identical to their independent whole-table computation")
    return {"E2": {"columns": sorted(expected), "passed": True},
            "E3": {"columns": list(E3_COLUMNS), "route": "whole-table float64 reductions over the noun axis", "passed": True}}


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


def verify_draws_against_table(cells: torch.Tensor, table: Mapping[str, Any], units: ExposedUnits, draws: int | None = None) -> dict[str, Any]:
    """E5 and E6: on the first ``draws`` calibration draws of every population and group, ``SST``, ``g`` and the gap
    from the cells against a direct recomputation from the per-noun table; ``SST`` against the pooled two-pass identity
    on every pair and every one of those draws."""
    draws = CROSS_CHECK_DRAWS if draws is None else int(draws)  # resolved at call time, never bound at definition
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
    say("E2, E3: the cells reproduce 022's stored cells and an independent computation of S, Q and SSEC bit for bit")
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


# ---------------------------------------------------------------------------
# The calibration (exposed only, once; no model): from the committed exposed-cells artifact alone (Task 5).

KERNEL_KEYS = ("g", "defined", "gap", "SST", "R2_0", "R2_1", "R2_C", "ceiling_limited")


class KernelCheckError(IncidentError):
    """The vectorized kernel disagreed with the direct loop beyond the implementation tolerance: an incident, with its
    location. No envelope exists yet, so nothing is written."""

    def __init__(self, details: Mapping[str, Any]):
        self.details = dict(details)
        super().__init__(f"kernel/direct cross-check failed: {self.details.get('max_difference')} at {self.details.get('at')} above {TOLERANCES['kernel']:.0e}")


def calibration_kernel(cells: torch.Tensor, units: ExposedUnits, indices: Mapping[str, torch.Tensor], draws: int, *, chunk: int = 500,
                       log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Every draw of every condition through the canonical path: ``pool`` (E6 enforced) → ``statistics``."""
    say = log or (lambda message: None)
    out = {condition: {key: torch.empty(draws, dtype=torch.bool if key in ("defined", "ceiling_limited") else torch.float64) for key in KERNEL_KEYS}
           for condition in CONDITIONS}
    e6 = 0.0
    for population in POPULATIONS:
        for group in GROUPS:
            condition = f"{population}/{group}"
            for start in range(0, draws, chunk):
                rows = range(start, min(draws, start + chunk))
                pooled = pool(cells, draw_pairs(units, indices, population, group, rows))
                e6 = max(e6, enforce_e6(pooled, f"{condition} draws {rows.start}–{rows.stop - 1}"))
                stats = statistics(pooled)
                for key in KERNEL_KEYS:
                    out[condition][key][rows.start:rows.stop] = stats[key]
            say(f"  {condition}: {draws} draws pooled")
    return {"conditions": out, "e6_max": e6}


def kernel_loop_check(cells: torch.Tensor, units: ExposedUnits, indices: Mapping[str, torch.Tensor], kernel: Mapping[str, Any],
                      n_draws: int | None = None) -> dict[str, Any]:
    """Implementation-only: on the first draws, ``SST``, ``g`` and the gap recomputed by an explicit loop over the drawn
    pairs' cells (pure Python sums, the frozen rule written out again) against the vectorized kernel."""
    n_draws = CROSS_CHECK_DRAWS if n_draws is None else int(n_draws)  # resolved at call time
    worst = {"max_difference": 0.0, "at": "", "n_checked": 0}
    for population in POPULATIONS:
        for group in GROUPS:
            condition = f"{population}/{group}"
            pairs = draw_pairs(units, indices, population, group, range(min(n_draws, int(kernel["conditions"][condition]["g"].shape[0]))))
            for b in range(int(pairs.shape[0])):
                n = s = q = sse0 = sse1 = ssec = 0.0
                for pair in pairs[b].tolist():
                    row = cells[pair].tolist()
                    n, s, q, sse0, sse1, ssec = n + row[0], s + row[1], q + row[2], sse0 + row[3], sse1 + row[4], ssec + row[5]
                sst = q - s * s / n
                defined = sst > 0 and sse0 - ssec >= GAP_MIN * sst
                direct = {"SST": sst, "g": (sse0 - sse1) / (sse0 - ssec) if defined else math.nan, "gap": (sse0 - ssec) / sst if sst > 0 else math.nan}
                for name, value in direct.items():
                    difference = agreement(float(kernel["conditions"][condition][name][b]), value)
                    worst["n_checked"] += 1
                    if difference > worst["max_difference"]:
                        worst.update({"max_difference": difference, "at": f"{condition}/draw {b}/{name}"})
    worst.update({"tolerance": TOLERANCES["kernel"], "passed": worst["max_difference"] <= TOLERANCES["kernel"]})
    if not worst["passed"]:
        raise KernelCheckError(worst)
    return rc.json_safe(worst)


def undefined_counts(kernel: Mapping[str, Any]) -> dict[str, int]:
    return {condition: int((~entries["defined"]).sum()) for condition, entries in kernel["conditions"].items()}


def calibration_stop(counts: Mapping[str, int], draws: int) -> dict[str, Any]:
    """250 or more undefined draws in any condition make its bound −∞: no envelope is written; stop for review."""
    threshold = lower_rank(draws)
    offending = {condition: count for condition, count in counts.items() if count >= threshold}
    return {"stop": bool(offending), "offending": offending, "threshold": threshold, "counts": dict(counts)}


def result_rates(codes: torch.Tensor) -> dict[str, float]:
    """The share of draws in each result, counted from ``result_codes``."""
    counts = torch.bincount(codes.reshape(-1), minlength=len(RESULTS))
    return {name: int(counts[index]) / int(codes.numel()) for index, name in enumerate(RESULTS)}


def evaluate(kernel: Mapping[str, Any]) -> dict[str, Any]:
    """Per condition: the envelope (element [249], undefined at −∞), its direction check, the guard-bound flag, the result
    rates and a summary; then the descriptive joint rate. The rates and the joint rate are counted from
    ``result_codes`` — the classification a fresh condition gets. A reversed tail is an incident, before anything is
    written."""
    out: dict[str, Any] = {"conditions": {}}
    passes = []
    for condition in CONDITIONS:
        entries = kernel["conditions"][condition]
        values, defined = entries["g"], entries["defined"]
        envelope = lower_bound(values, defined)
        direction = direction_check(envelope, values, defined)
        if not direction["ok"]:
            raise IncidentError(f"{condition}: the lower envelope {envelope['bound']} lies above the median of the defined draws {direction['median']}; a reversed tail")
        kept = torch.sort(values[defined]).values
        summary = {"defined": int(defined.sum()), "median": direction["median"], "min": float(kept[0]) if kept.numel() else None,
                   "max": float(kept[-1]) if kept.numel() else None, "gap_median": ul.defined_median(entries["gap"], defined),
                   "R2_1_median": ul.defined_median(entries["R2_1"], defined), "R2_C_median": ul.defined_median(entries["R2_C"], defined),
                   "ceiling_limited_share": int(entries["ceiling_limited"].sum()) / int(values.shape[0])}
        codes = result_codes(values, defined, envelope["bound"])
        out["conditions"][condition] = {"envelope": envelope, "direction_check": direction, "guard_bound": bool(envelope["bound"] < GUARD_MIN), "rates": result_rates(codes),
                                        "summary": summary}
        passes.append(codes == RESULTS.index("PASS"))
    out["joint_rates"] = {"all_four_pass": float(torch.stack(passes).all(dim=0).double().mean()), "descriptive_only": True}
    return rc.json_safe(out)


def draw_arrays(kernel: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {f"{condition}/{key}": entries[key] for condition, entries in kernel["conditions"].items() for key in KERNEL_KEYS}


def record_constants(draws: int) -> dict[str, Any]:
    return {"B": int(draws), "draw_tag": DRAW_TAG, "cross_check_draws": CROSS_CHECK_DRAWS, "lower_rank": lower_rank(draws), "guard_min": GUARD_MIN, "gap_min": GAP_MIN,
            "ceiling_limited_r2": CEILING_LIMITED_R2, "tolerances": dict(TOLERANCES), "cell_columns": list(CELL_COLUMNS), "cells_version": CELLS_VERSION,
            "strata": list(STRATA), "slots": draw_slots(), "p0_mask": P0_MASK, "p1_mask": dict(P1_MASK), "results": list(RESULTS), "statistic": STATISTIC,
            "conditions": list(CONDITIONS), "scope": SCOPE}


def calibration_record(*, run_id: str, protocol_code_commit: str, digests: Mapping[str, str], units: ExposedUnits, cells_files: Mapping[str, str],
                       confirmation: Mapping[str, str], index_digests: Mapping[str, str], kernel_check: Mapping[str, Any], e6_max: float,
                       undefined: Mapping[str, int], evaluated: Mapping[str, Any], array_digests: Mapping[str, str], draws: int) -> dict[str, Any]:
    record = {
        "experiment": EXPERIMENT, "schema_version": 1, "kind": "the exposed-only calibration record of Experiment 023 (design revision 2), from the committed exposed cells",
        "design": dict(DESIGN), "plan": dict(PLAN), "run_id": run_id, "protocol_code_commit": protocol_code_commit, "inputs": dict(digests), "module_blobs": dict(FROZEN_BLOBS),
        "constants": record_constants(draws), "exposed_cells": dict(cells_files), "confirmation_023": dict(confirmation),
        "pools": {"cues": {stratum: len(units.stratum_cues(stratum)) for stratum in STRATA}, "y2_like_frames": {t: len(units.y2_like[t]) for t in TEMPLATES},
                  "y1_frames": {group: len(units.group_frames(group)) for group in GROUPS}},
        "draws": {"B": int(draws), "index_sha256": dict(index_digests)}, "kernel_check": dict(kernel_check), "e6_max": float(e6_max),
        "undefined_counts": dict(undefined), "conditions": dict(evaluated["conditions"]), "joint_rates": dict(evaluated["joint_rates"]),
        "draw_arrays_sha256": dict(array_digests),
    }
    validate_json_safe(record)
    record["content_sha256"] = rc.content_digest(record)
    return record


def verify_calibration_record(record: Mapping[str, Any], draws: int | None = None) -> None:
    """A calibration record fit for the lock: its content digest; the frozen constants at ``draws`` (the module's ``B``
    unless given), design, plan and modules; exactly the four conditions, each with a finite lower envelope at element
    ``[lower_rank − 1]`` whose direction check passed and whose guard-bound flag it implies; and no condition at the
    undefined-draw stop."""
    draws = B if draws is None else int(draws)  # resolved at call time, never bound at definition
    if record.get("experiment") != EXPERIMENT or record.get("content_sha256") != rc.content_digest(record):
        raise PhaseError("not a verified Experiment 023 calibration record")
    if record["constants"] != record_constants(draws) or record["design"] != DESIGN or record["plan"] != PLAN:
        raise PhaseError("the calibration record's constants, design or plan are not the frozen ones")
    if record["module_blobs"] != FROZEN_BLOBS:
        raise PhaseError("the calibration record was written against different frozen modules")
    if set(record["conditions"]) != set(CONDITIONS) or set(record.get("undefined_counts") or {}) != set(CONDITIONS):
        raise PhaseError("the calibration record does not carry exactly the four conditions")
    rank = lower_rank(draws)
    for condition in CONDITIONS:
        entry = record["conditions"][condition]
        envelope = entry.get("envelope") or {}
        bound = envelope.get("bound")
        if (envelope.get("kind"), envelope.get("rank"), envelope.get("element")) != ("lower", rank, rank - 1):
            raise PhaseError(f"{condition}: the calibration record's envelope is not the frozen lower order statistic v₍{rank}₎")
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound):
            raise PhaseError(f"{condition}: the calibration record's envelope is not a finite number ({bound!r})")
        if not (entry.get("direction_check") or {}).get("ok"):
            raise PhaseError(f"{condition}: the calibration record's direction check did not pass")
        if entry.get("guard_bound") is not (bound < GUARD_MIN):
            raise PhaseError(f"{condition}: the calibration record's guard-bound flag is not the one its envelope implies")
        if int(record["undefined_counts"][condition]) >= rank:
            raise PhaseError(f"{condition}: {record['undefined_counts'][condition]} undefined draws reach the calibration stop at {rank}; no envelope may be used")


def confirmation_binding(confirmation: Confirmation023, file_sha256: str) -> dict[str, str]:
    """What the calibration record, the results state and the lock bind of the committed confirmation file."""
    return {"path": CONFIRMATION_RELATIVE_PATH, "file_sha256": file_sha256, "content_sha256": confirmation.content_sha256}


def verify_confirmation_binding(record: Mapping[str, Any], state: Mapping[str, Any], binding: Mapping[str, str]) -> None:
    """The calibration record and the results state bind the committed confirmation file now in the tree: the file the
    calibration read its counts from is the one the lock and the confirmation use."""
    if dict(record.get("confirmation_023") or {}) != dict(binding):
        raise PhaseError("the calibration record binds a confirmation file other than the committed one; the frozen units cannot change after calibrate")
    if dict(state.get("confirmation_023") or {}) != dict(binding):
        raise PhaseError("the results state binds a confirmation file other than the committed one; the frozen units cannot change after calibrate")


# ---------------------------------------------------------------------------
# The prediction tables (weights and reference states only; no measured quantity): P0 and P1 per pair (Task 6).

TABLE_SCHEMA_VERSION = 1
TABLE_CONSTRUCTION = (
    "per pair (cue t, frame f, the template's reference cue r), from the frame's reference state and the weights by Experiment 022's own functions "
    "(ul.pair_context, ul.compose_dx3, ul.contrast_of): P0 = 022's empty coalition (mask 0: ΔE through Experiment 017's reduced chain, layer-1/2 reference "
    "rows through p_c) and P1 = 022's inputs-only coalition (mask 14 in cue-final frames, 30 in coordinated frames: ΔE + Δemb + V + P at p_c and T at p_t, "
    "through the same reduced chain), each through the frozen Experiment 020 readout; columns (P0, P1); Δĉ over the scorable exposed nouns in the listed order")


def table_meta(kind: str, units: ul.TableUnits, noun_keys: Sequence[str]) -> dict[str, Any]:
    return {"experiment": EXPERIMENT, "kind": kind, "schema_version": TABLE_SCHEMA_VERSION, "construction": TABLE_CONSTRUCTION, "design": dict(DESIGN),
            "pair_order": "cue token id, then frame_id", "pairs": units.pair_json(), "columns": ["P0", "P1"], "masks": {"P0": P0_MASK, "P1": dict(P1_MASK)},
            "nouns": list(noun_keys)}


def table_layout(units: ul.TableUnits, n_nouns: int) -> list[dict[str, Any]]:
    return [{"name": group, "shape": [len(units.pairs[group]), 2, int(n_nouns)]} for group in GROUPS]


def closed_form_t(progs: ul.ModelPrograms, frame: pm.Frame, reference_id: int, token_id: int) -> torch.Tensor:
    """The design's closed form of block 0's change at the target position (one logit update per head):
    ``Σ_h [a_h (o'_h − o_h) + (σ(logit a_h + ⟨q_{p_t,h}, Δk_h⟩/√d) − a_h)(o'_h − rest_h)]``, from the embeddings alone."""
    program0 = progs.program0
    x0_all = ul.reference_embeddings(progs.weights, frame, reference_id)
    rr = atp.ReferenceRow(program0, [x.double() for x in x0_all[: frame.p_t + 1]])
    normed = program0.normalize(progs.weights.W_E[int(token_id)].double())
    k_new = program0.rotate(program0.k_tilde(normed), frame.p_c)
    o_new = torch.einsum("he,hed->hd", program0.v(normed), program0.W_O)
    o_all = program0.output(rr.values)
    a = rr.A_ref[:, frame.p_c]
    rest = (torch.einsum("hk,hkd->hd", rr.A_ref, o_all) - a[:, None] * o_all[:, frame.p_c]) / (1.0 - a)[:, None]
    shift = (rr.q_ref * (k_new - rr.keys[:, frame.p_c])).sum(-1) / program0.scale
    a_new = torch.sigmoid(torch.log(a) - torch.log1p(-a) + shift)
    return (a[:, None] * (o_new - o_all[:, frame.p_c]) + (a_new - a)[:, None] * (o_new - rest)).sum(0)


def pair_predictions(progs: ul.ModelPrograms, ctx: ul.PairContext) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """``P0`` and ``P1`` of one pair (022's mask 0 and mask 14 / 30), and P0's ``Δx̂3`` for I5."""
    group = "cue_final" if ctx.cue_final else "coordinated"
    empty = ul.compose_dx3(progs, ctx, P0_MASK)
    p0 = ul.contrast_of(progs, ctx.state, empty)
    p1 = ul.contrast_of(progs, ctx.state, ul.compose_dx3(progs, ctx, P1_MASK[group]))
    return p0, p1, empty


def prediction_tables(progs: ul.ModelPrograms, units: ul.TableUnits, states: Mapping[str, rd.FrameState020], reference_ids: Mapping[str, int], *,
                      log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Every pair's ``(P0, P1)`` in the table's order, weights and reference states only, with the maxima of I5 (P0
    against the committed Level 0, exactly) and of the block-0 algebra, and digests of every pair's P0 ``Δx̂3`` and
    block-0 terms. Of the algebra, ``V + P = ΔA0(p_c)`` only restates 022's decomposition (022 defines ``P`` as the total
    minus ``V``; it guards the stored factors' consistency), while the closed-form ``T`` against the exact attention
    output at ``p_t`` is an independent derivation."""
    say = log or (lambda message: None)
    rows16: dict[str, Mapping[int, atp.ReferenceRow]] = {}
    level0, factor_bytes = hashlib.sha256(), hashlib.sha256()
    gates = {"I5": {"max": 0.0, "at": ""}, "algebra": {"max": 0.0, "at": ""}}
    blocks = []
    for group in GROUPS:
        out = torch.empty(len(units.pairs[group]), 2, len(progs.scorable), dtype=torch.float64)
        for row, (t, f) in enumerate(units.pairs[group]):
            frame, token = units.frames[f], units.tokens[t]
            state = states[frame.frame_id]
            if frame.frame_id not in rows16:
                rows16[frame.frame_id] = ul.reference_rows_017(progs.programs, state)
            reference_id = int(reference_ids[frame.template_id])
            ctx = ul.pair_context(progs, frame, state, reference_id, int(token["token_id"]), token["word"], rows16[frame.frame_id])
            p0, p1, empty = pair_predictions(progs, ctx)
            out[row, 0], out[row, 1] = p0, p1
            committed = rd.predicted_dx3(progs.chain, progs.weights, state, ctx.rows16, ctx.token_id, frame.template_id)
            i5 = math.inf if set(committed) != set(empty) else max(float((empty[p] - committed[p]).abs().max()) for p in committed)
            factors = ctx.factors
            algebra = float((factors.value_pc + factors.pattern_pc - factors.block0_pc.total).abs().max())
            if factors.attn_pt is not None:
                algebra = max(algebra, float((closed_form_t(progs, frame, reference_id, int(token["token_id"])) - factors.attn_pt).abs().max()))
            where = f"{token['word']}|{frame.frame_id}"
            ul._worse(gates["I5"], i5, where)
            ul._worse(gates["algebra"], algebra, where)
            for position in sorted(empty):
                level0.update(ul._f64(empty[position]))
            for vector in (factors.delta_e, factors.d_emb, factors.value_pc, factors.pattern_pc) + (() if factors.attn_pt is None else (factors.attn_pt,)):
                factor_bytes.update(ul._f64(vector))
        blocks.append((group, out))
        say(f"  {group}: {len(units.pairs[group])} pairs predicted (P0, P1)")
    return {"blocks": blocks, "gates": {name: rc.json_safe(entry) for name, entry in gates.items()}, "p0_dx3_sha256": level0.hexdigest(), "factors_sha256": factor_bytes.hexdigest()}


def enforce_table_gates(gates: Mapping[str, Mapping[str, Any]], error: type[Exception] = IncidentError) -> None:
    for name in ("I5", "algebra"):
        value = gates[name]["max"]
        if value is None or not value <= TOLERANCES[name]:
            raise error(f"identity gate {name} failed while predicting a table: {value} at {gates[name]['at']} against {TOLERANCES[name]:.0e}")


# ---------------------------------------------------------------------------
# The lock (no forward pass): the four conditions, the Y1 companion, the Y2 specification.

GUARD = {"min_inclusive": GUARD_MIN}
SEMANTICS = {
    "results": {
        "NOT_INTERPRETABLE": "too little Level-0 → ceiling gap remains (SST ≤ 0 or SSE0 − SSEC < 0.02·SST on the full aggregate); neither a pass nor a failure",
        "GUARD_FAILURE": "g < 0.90: the completed program does not recover essentially all of the explainable upstream gap on that population and group",
        "ENVELOPE_ONLY_FAILURE": "g ≥ 0.90, so essentially all of the explainable gap is recovered, but g lies below the calibrated exposed-like envelope: a "
                                 "quantitative shift, never a refutation",
        "PASS": "the completed program recovers essentially all of the explainable upstream gap within the exposed-like envelope: the weight-derived upstream "
                "program reaches the ceiling the decoded downstream readout allows",
    },
    "precedence": list(RESULTS),
    "ceiling": "C is the frozen downstream ceiling comparator, not a mathematical upper bound on a finite sample's R²; P1 slightly outperforming it (g > 1) is "
               "permitted and never an incident, and says nothing beyond 'essentially all'",
    "aggregate": "none: the four condition results are the result, each read on its own; Y1 and Y2 are never pooled; no all-pass requirement",
    "gap_rule": "evaluated on the condition's full aggregate; it removes, re-weights or selects no pair, cue, frame or noun",
    "scope": SCOPE,
    "cdf_percentile": "descriptive only; the frozen envelope decides",
    "not_shown": "a pass does not show that the downstream readout is complete, anything about possessive or pronoun cues, new nouns or behavior beyond Δc at "
                 "p_t, or that a cheaper block-0 rule would suffice (the comparators are descriptive)",
    "incidents": "incidents carry no result",
}


def lock_conditions(record: Mapping[str, Any]) -> dict[str, Any]:
    return {condition: {"population": condition.split("/")[0], "group": condition.split("/")[1], "statistic": STATISTIC,
                        "envelope": dict(record["conditions"][condition]["envelope"]), "guard": dict(GUARD),
                        "guard_bound": bool(record["conditions"][condition]["guard_bound"])} for condition in CONDITIONS}


def y2_table_spec(confirmation: Confirmation023, noun_keys: Sequence[str]) -> dict[str, Any]:
    """What the lock binds of the Y2 table, which cannot exist before stage 1: construction, byte format, layout and
    every order. Its numbers are frozen at the stage-1 barrier."""
    units = ul.table_units(confirmation.tokens, confirmation.frames)
    return {"data_path": Y2_TABLE_OUTPUT, "index_path": Y2_TABLE_INDEX_OUTPUT, "format": ul.TABLE_FORMAT, "dtype": ul.TABLE_DTYPE,
            "layout": table_layout(units, len(noun_keys)), "meta": table_meta("Y2", units, noun_keys),
            "states": "each new frame's digested S1-REF reference state (rd.locked_state), rebuilt by rd.state_from_locked",
            "freeze": "written once at stage 1 with its digests in the stage-1 record; frozen at the stage-1 barrier, where it is re-read from disk and verified "
                      "against the digests held in memory; stage 2 consumes the verified file; any later change is an incident; committed as closure evidence "
                      "after a successful confirmation"}


def build_lock(*, run_id: str, protocol_code_commit: str, digests: Mapping[str, str], record: Mapping[str, Any], record_file_sha256: str,
               confirmation: Confirmation023, confirmation_file_sha256: str, cells_files: Mapping[str, str], y1_index: Mapping[str, Any], y1_index_sha256: str,
               y1_tables: Mapping[str, Any], noun_keys: Sequence[str], exposed_states: Mapping[str, Any]) -> dict[str, Any]:
    lock = {
        "experiment": EXPERIMENT, "schema_version": 1, "kind": "the preregistration lock of Experiment 023 (design revision 2, plan revision 1)", "design": dict(DESIGN),
        "plan": dict(PLAN), "run_id": run_id, "protocol_code_commit": protocol_code_commit, "inputs": dict(digests), "module_blobs": dict(FROZEN_BLOBS),
        "constants": record_constants(B), "calibration": {"path": CALIBRATION_RELATIVE_PATH, "file_sha256": record_file_sha256, "content_sha256": record["content_sha256"]},
        "exposed_cells": dict(cells_files),
        "confirmation_023": {**confirmation_binding(confirmation, confirmation_file_sha256), "counts": confirmation.counts(),
                             "manifest_sizes": {"S1-REF": len(confirmation.frames), "S1-VALIDITY": len(confirmation.frames), "Y1": len(confirmation.y1_prompts),
                                                "Y2": len(confirmation.y2_prompts)}},
        "conditions": lock_conditions(record), "semantics": dict(SEMANTICS), "program": {"P0_mask": P0_MASK, "P1_mask": dict(P1_MASK), "construction": TABLE_CONSTRUCTION},
        "noun_keys": list(noun_keys), "exposed_states_sha256": ul.exposed_states_digest(exposed_states),
        "y1_table": {"data_path": Y1_TABLE_RELATIVE_PATH, "index_path": Y1_TABLE_INDEX_RELATIVE_PATH, "file_sha256": y1_index["file_sha256"], "index_sha256": y1_index_sha256,
                     "total_bytes": y1_index["total_bytes"], "layout": [{"name": entry["name"], "shape": list(entry["shape"])} for entry in y1_index["blocks"]],
                     "p0_dx3_sha256": y1_tables["p0_dx3_sha256"], "factors_sha256": y1_tables["factors_sha256"], "gates": dict(y1_tables["gates"])},
        "y2_table": y2_table_spec(confirmation, noun_keys),
    }
    validate_json_safe(lock)
    lock["content_sha256"] = rc.content_digest(lock)
    return lock


def render_preregistration(lock: Mapping[str, Any]) -> str:
    lines = ["# Experiment 023 — preregistered conditions", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; design revision {lock['design']['revision']} (`{lock['design']['commit']}`), "
             f"plan revision {lock['plan']['revision']} (`{lock['plan']['commit']}`)",
             f"- Calibration record content sha256 `{lock['calibration']['content_sha256']}` (file `{lock['calibration']['file_sha256']}`), B = {lock['constants']['B']}",
             f"- Exposed cells: data `{lock['exposed_cells']['data_sha256']}`, index `{lock['exposed_cells']['index_sha256']}`",
             f"- Confirmation file content sha256 `{lock['confirmation_023']['content_sha256']}`: {lock['confirmation_023']['manifest_sizes']}",
             f"- Y1 table `{lock['y1_table']['data_path']}`: sha256 `{lock['y1_table']['file_sha256']}`, index sha256 `{lock['y1_table']['index_sha256']}`, blocks "
             f"{[(entry['name'], entry['shape']) for entry in lock['y1_table']['layout']]}",
             f"- Y2 table: written at stage 1 as `{lock['y2_table']['data_path']}` in the same format, blocks {[(entry['name'], entry['shape']) for entry in lock['y2_table']['layout']]}; "
             "frozen at the stage-1 barrier", "",
             f"Statistic: {STATISTIC}. Meaning guard: g ≥ {GUARD_MIN}. Effective requirement of a PASS: g ≥ max(F, {GUARD_MIN}).", "",
             "| condition | envelope F (lower, exact order statistic) | meaning guard | guard-bound |", "|---|---|---|---|"]
    for condition in CONDITIONS:  # a fixed order: the text must not depend on a mapping's order
        entry = lock["conditions"][condition]
        lines.append(f"| {condition} | ≥ v₍{entry['envelope']['rank']}₎ = {entry['envelope']['bound']:.6f} | g ≥ {GUARD_MIN} | {'yes' if entry['guard_bound'] else 'no'} |")
    lines += ["", "Each condition has exactly one result, decided in the order NOT_INTERPRETABLE → GUARD_FAILURE → ENVELOPE_ONLY_FAILURE → PASS:", ""]
    lines += [f"- `{name}`: {lock['semantics']['results'][name]}" for name in RESULTS]
    lines += ["", f"- The ceiling: {lock['semantics']['ceiling']}.", f"- Aggregate: {lock['semantics']['aggregate']}.", f"- Gap rule: {lock['semantics']['gap_rule']}.",
              f"- Scope: {lock['semantics']['scope']}.", f"- What a pass does not show: {lock['semantics']['not_shown']}.", ""]
    return "\n".join(lines)


SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", *ul.SCIENTIFIC_PATH_PREFIXES[1:])
NON_SCIENTIFIC_PATHS = (CELLS_DATA_RELATIVE_PATH, CELLS_INDEX_RELATIVE_PATH, CONFIRMATION_RELATIVE_PATH, CALIBRATION_RELATIVE_PATH, LOCK_RELATIVE_PATH,
                        PREREGISTRATION_RELATIVE_PATH, Y1_TABLE_RELATIVE_PATH, Y1_TABLE_INDEX_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md", *ul.NON_SCIENTIFIC_PATHS)
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/", *ul.NON_SCIENTIFIC_PREFIXES)


def scientific_changes(paths: Sequence[str]) -> list[str]:
    return [path for path in paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and path not in NON_SCIENTIFIC_PATHS and not path.startswith(NON_SCIENTIFIC_PREFIXES)]


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], record: Mapping[str, Any], record_file_sha256: str,
                  confirmation: Confirmation023, confirmation_file_sha256: str, cells_files: Mapping[str, str], noun_keys: Sequence[str],
                  exposed_states: Mapping[str, Any], preregistration_text: str, y1_index_text: str, y1_file_sha256: str, git_state: Mapping[str, Any], tracked: bool,
                  changed_paths: Sequence[str] | None) -> dict[str, Any]:
    """The installed lock before any fresh prompt: its digest; that it is this run's candidate; every bound input,
    constant, condition and semantics; the exposed cells; the Y1 companion; the Y2 specification; the rendered
    preregistration; and a clean tree with no scientific change since the lock commit. Returns the Y1 index."""
    if lock.get("experiment") != EXPERIMENT or lock.get("content_sha256") != rc.content_digest(lock):
        raise PhaseError("the installed lock is not a verified Experiment 023 lock")
    if state.get("lock") is None or state["lock"].get("content_sha256") != lock["content_sha256"]:
        raise PhaseError("the installed lock is not the candidate this run wrote")
    if lock["inputs"] != dict(digests) or lock["module_blobs"] != dict(FROZEN_BLOBS) or lock["design"] != dict(DESIGN) or lock["plan"] != dict(PLAN):
        raise PhaseError("the lock was written against different frozen inputs, modules, design or plan")
    verify_calibration_record(record)
    if lock["calibration"] != {"path": CALIBRATION_RELATIVE_PATH, "file_sha256": record_file_sha256, "content_sha256": record["content_sha256"]}:
        raise PhaseError("the lock was written against a different calibration record")
    if lock["constants"] != record_constants(B) or lock["conditions"] != lock_conditions(record) or lock["semantics"] != dict(SEMANTICS):
        raise PhaseError("the lock's constants, conditions or semantics are not those of the committed record and the frozen design")
    if lock["exposed_cells"] != dict(cells_files) or record["exposed_cells"] != dict(cells_files):
        raise PhaseError("the lock or the record binds different exposed cells")
    binding = confirmation_binding(confirmation, confirmation_file_sha256)
    if {key: lock["confirmation_023"].get(key) for key in binding} != binding:
        raise PhaseError("the lock was written against a different confirmation file")
    verify_confirmation_binding(record, state, binding)
    if lock["noun_keys"] != list(noun_keys) or lock["exposed_states_sha256"] != ul.exposed_states_digest(exposed_states):
        raise PhaseError("the lock names a different noun order or different exposed reference states")
    if lock["program"] != {"P0_mask": P0_MASK, "P1_mask": dict(P1_MASK), "construction": TABLE_CONSTRUCTION}:
        raise PhaseError("the lock's program is not the frozen P0/P1")
    if lock["y2_table"] != y2_table_spec(confirmation, noun_keys):
        raise PhaseError("the lock's Y2 table specification is not the frozen construction, format and order")
    if pm.sha256_text(y1_index_text) != lock["y1_table"]["index_sha256"]:
        raise PhaseError("the committed Y1 table index is not the one the lock binds")
    y1_index = json.loads(y1_index_text)
    if y1_index.get("file_sha256") != lock["y1_table"]["file_sha256"] or y1_file_sha256 != lock["y1_table"]["file_sha256"]:
        raise PhaseError("the committed Y1 table is not the one the lock binds")
    ul.verify_table_index(y1_index, layout=lock["y1_table"]["layout"], meta=table_meta("Y1", ul.table_units(confirmation.tokens, confirmation.exposed_frames), noun_keys),
                          error=PhaseError)
    if preregistration_text != render_preregistration(lock) or pm.sha256_text(preregistration_text) != state["lock"].get("preregistration_sha256"):
        raise PhaseError("the installed preregistration is not the one this lock renders")
    if not tracked:
        raise PhaseError("the lock, the preregistration and the Y1 table must be tracked and committed")
    if git_state.get("dirty"):
        raise PhaseError("confirm requires a clean Git tree")
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = scientific_changes(changed_paths)
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock: {scientific}")
    return y1_index


# ---------------------------------------------------------------------------
# Confirm: stage 1 and the write-once Y2 table, the barrier, the target gates, the scoring (once; no resume).


def stage_one_digest(stage1: Mapping[str, Any]) -> str:
    return pm.sha256_text(pm.canonical_json({key: value for key, value in stage1.items() if key != "digest"}))


def stage_one(model: Any, progs: ul.ModelPrograms, confirmation: Confirmation023, lock: Mapping[str, Any], *, root: Path, protocol_code_commit: str,
              executed: list[pm.Prompt], log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Exactly the S1-REF and the S1-VALIDITY prompt of each new frame, one forward each, in ``frame_id`` order (the
    validity verdicts are descriptive and select nothing); then the Y2 prediction table from the recorded reference
    states — weights and states only, never a target measurement — written once in the bound byte format, and the
    stage-1 record carrying its digests."""
    say = log or (lambda message: None)
    frames_out: dict[str, Any] = {}
    states: dict[str, Any] = {}
    for frame in sorted(confirmation.frames, key=lambda frame: frame.frame_id):
        reference = confirmation.reference_prompt(frame)
        state = rd.capture_frame_020(model, progs.head, reference, progs.nouns, progs.axis_T)
        executed.append(reference)
        coordinated = frame.template_id == COORDINATED
        if (state.p_c, state.p_t) != (frame.p_c, frame.p_t) or frame.p_t != frame.p_c + (1 if coordinated else 0):
            raise IncidentError(f"{frame.frame_id}: the reference capture's positions ({state.p_c}, {state.p_t}) break the frame's frozen structure")
        validity = confirmation.validity_prompt(frame)
        plural = rd.measure_pair(model, state, progs.nouns, validity.cue_label, validity.cue_token_id)
        executed.append(validity)
        verdict = rd.frame_validity(progs.readout, state, progs.nouns, plural, progs.axis_T)
        frames_out[frame.frame_id] = {"template_id": frame.template_id, "p_c": frame.p_c, "p_t": frame.p_t, "cue_final": not coordinated,
                                      "validity": rc.json_safe(dict(verdict)), "selects": "nothing: descriptive only"}
        states[frame.frame_id] = rd.locked_state(state)
        say(f"  {frame.frame_id}: reference state captured; validity {verdict['valid']} (descriptive)")
    data_path, index_path = root / lock["y2_table"]["data_path"], root / lock["y2_table"]["index_path"]
    if data_path.exists() or index_path.exists():
        raise IncidentError("a Y2 table is already on disk; it is written exactly once, at stage 1")
    rebuilt = {frame.frame_id: rd.state_from_locked(states[frame.frame_id], frame) for frame in confirmation.frames}
    tables = prediction_tables(progs, ul.table_units(confirmation.tokens, confirmation.frames), rebuilt, confirmation.reference_ids, log=say)
    index = ul.write_table(data_path, index_path, tables["blocks"], lock["y2_table"]["meta"])
    record = {"frames": frames_out, "states": states, "state_digests": {frame_id: rd.state_digest(entry) for frame_id, entry in sorted(states.items())},
              "y2_table": {"data_path": lock["y2_table"]["data_path"], "index_path": lock["y2_table"]["index_path"], "file_sha256": index["file_sha256"],
                           "index_sha256": rc.file_sha256(index_path), "total_bytes": index["total_bytes"], "p0_dx3_sha256": tables["p0_dx3_sha256"],
                           "factors_sha256": tables["factors_sha256"], "gates": tables["gates"]},
              "commit": protocol_code_commit, "lock_sha256": lock["content_sha256"], "noun_keys": list(lock["noun_keys"])}
    record["digest"] = stage_one_digest(record)
    return record


def stage_one_expectations(stage1: Mapping[str, Any]) -> dict[str, str]:
    """What stage 1 wrote, kept in memory: the barrier and the post-stage-2 check compare the re-read artifacts with
    these, not only with themselves."""
    return {"stage1_digest": stage1["digest"], "y2_file_sha256": stage1["y2_table"]["file_sha256"], "y2_index_sha256": stage1["y2_table"]["index_sha256"]}


def verified_y2_blocks(root: Path, stage1: Mapping[str, Any], lock: Mapping[str, Any], expected: Mapping[str, str] | None = None) -> dict[str, torch.Tensor]:
    """The Y2 table re-read from disk: its index against the stage-1 record's digest (and the digests stage 1 wrote,
    when given), the bytes against the index's, and the layout, construction and orders against the lock."""
    entry = stage1["y2_table"]
    if (entry["data_path"], entry["index_path"]) != (lock["y2_table"]["data_path"], lock["y2_table"]["index_path"]):
        raise IncidentError("the stage-1 record names a Y2 table other than the bound one")
    if expected is not None and (entry["file_sha256"], entry["index_sha256"]) != (expected["y2_file_sha256"], expected["y2_index_sha256"]):
        raise IncidentError("the re-read stage-1 record names Y2 digests other than the ones stage 1 wrote")
    data_path, index_path = root / entry["data_path"], root / entry["index_path"]
    if not data_path.exists() or not index_path.exists():
        raise IncidentError("the Y2 table written at stage 1 is missing")
    if rc.file_sha256(index_path) != entry["index_sha256"]:
        raise IncidentError("the Y2 table index changed after its stage-1 digest")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("file_sha256") != entry["file_sha256"]:
        raise IncidentError("the Y2 table index binds bytes other than the stage-1 record's")
    ul.verify_table_index(index, layout=lock["y2_table"]["layout"], meta=lock["y2_table"]["meta"])
    return ul.read_table(data_path, index)


def barrier(root: Path, state: Mapping[str, Any], lock: Mapping[str, Any], confirmation: Confirmation023, expected: Mapping[str, str] | None = None) -> dict[str, torch.Tensor]:
    """The hard boundary, on the results state re-read from disk: the stage-1 record reproduces its digest, is the one
    stage 1 wrote and names this lock; the ledger holds no S2-TARGET key; the Y2 table is re-read and verified. Returns
    the verified blocks, which the scoring consumes."""
    stage1 = (state.get("confirmation") or {}).get("stage1")
    if stage1 is None or stage1.get("digest") != stage_one_digest(stage1):
        raise IncidentError("the stage-1 record does not reproduce its digest; no S2-TARGET prompt may run")
    if expected is not None and stage1["digest"] != expected["stage1_digest"]:
        raise IncidentError("the re-read stage-1 record is not the one stage 1 wrote; no S2-TARGET prompt may run")
    if stage1.get("lock_sha256") != lock["content_sha256"]:
        raise IncidentError("the stage-1 record was written under a different lock")
    overlap = {prompt.key for prompt in confirmation.target_prompts} & set(state["executed_prompt_keys"])
    if overlap:
        raise IncidentError(f"{len(overlap)} S2-TARGET keys are in the ledger before the barrier, e.g. {sorted(overlap)[:2]}")
    return verified_y2_blocks(root, stage1, lock, expected)


def _changes(block: Mapping[str, torch.Tensor], row: int, name: str, frame: pm.Frame) -> dict[int, torch.Tensor]:
    return {frame.p_c: block[name][row, 0]} if frame.p_t == frame.p_c else {frame.p_c: block[name][row, 0], frame.p_t: block[name][row, 1]}


def target_gates(progs: ul.ModelPrograms, confirmation: Confirmation023, measured: Mapping[str, Any], states: Mapping[str, Mapping[str, rd.FrameState020]]) -> dict[str, Any]:
    """I1, I3 and I4 on every target pair, from the saved measurements and a weights-only recomputation of the pair's
    factors and of 022's full composition (layers 1–2 exact); descriptively, ``P1``'s ``Δx3`` relative error at the
    changed positions and block 0's head profile. The measured ``Δx3`` enters only these gates and the ceiling."""
    gates = {name: {"max": 0.0, "at": ""} for name in ("I1", "I3", "I4")}
    errors: dict[str, Any] = {}
    profile: dict[str, Any] = {}
    for population, entry in measured.items():
        units = entry["units"]
        rows16: dict[str, Any] = {}
        for group, pairs in units.pairs.items():
            block = entry["blocks"][group]
            p1_errors = {slot: [] for slot in (("p_c",) if group == "cue_final" else ("p_c", "p_t"))}
            self_weight, target_weight = [], []
            for row, (t, f) in enumerate(pairs):
                frame, token = units.frames[f], units.tokens[t]
                state = states[population][frame.frame_id]
                if frame.frame_id not in rows16:
                    rows16[frame.frame_id] = ul.reference_rows_017(progs.programs, state)
                ctx = ul.pair_context(progs, frame, state, int(confirmation.reference_ids[frame.template_id]), int(token["token_id"]), token["word"], rows16[frame.frame_id])
                factors, where = ctx.factors, f"{population}|{token['word']}|{frame.frame_id}"
                dx1, dx3 = _changes(block, row, "dx1", frame), _changes(block, row, "dx3", frame)
                i1 = float((factors.d_emb + factors.delta_e + factors.block0_pc.total - dx1[frame.p_c]).abs().max())
                if factors.block0_pt is not None:
                    i1 = max(i1, float((factors.block0_pt.total - dx1[frame.p_t]).abs().max()))
                ul._worse(gates["I1"], i1, where)
                full = ul.compose_dx3(progs, ctx, FULL_MASK[group])
                ul._worse(gates["I3"], ul.i3_error(full, dx3), where)
                ul._worse(gates["I4"], float((ul.contrast_of(progs, state, full) - block["ceiling"][row]).abs().max()), where)
                p1_dx3 = ul.compose_dx3(progs, ctx, P1_MASK[group])
                for position in sorted(dx3):
                    p1_errors["p_c" if position == frame.p_c else "p_t"].append(ul.i3_error({position: p1_dx3[position]}, {position: dx3[position]}))
                self_weight.append(factors.block0_pc.weight_to_cue)
                if factors.block0_pt is not None:
                    target_weight.append(factors.block0_pt.weight_to_cue)
            median = lambda rows: [float(value) for value in torch.stack(rows).median(dim=0).values] if rows else None  # noqa: E731
            errors[f"{population}/{group}"] = {slot: {"median": float(torch.tensor(values).median()), "max": float(max(values))} for slot, values in p1_errors.items() if values}
            profile[f"{population}/{group}"] = {"reference_self_weight_median": median(self_weight), "attention_pt_to_pc_median": median(target_weight)}
    return {"gates": {name: rc.json_safe(entry) for name, entry in gates.items()}, "p1_dx3_relative_error": rc.json_safe(errors), "block0_profile": rc.json_safe(profile)}


def enforce_target_gates(gates: Mapping[str, Mapping[str, Any]]) -> None:
    for name in ("I1", "I3", "I4"):
        value = gates[name]["max"]
        if value is None or not value <= TOLERANCES[name]:
            raise IncidentError(f"identity gate {name} failed at stage 2: {value} at {gates[name]['at']} against {TOLERANCES[name]:.0e}")


def fresh_cells(measured: Mapping[str, Any], tables: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Every fresh pair's cells, per condition, in the table's pair order: ``y`` = the measured ``Δc``, ``P0``/``P1``
    from the verified table, ``C`` = the ceiling."""
    out = {}
    for population in POPULATIONS:
        for group in GROUPS:
            block, table = measured[population]["blocks"][group], tables[population][group]
            rows = [pair_cells(block["dc"][i], table[i, 0], table[i, 1], block["ceiling"][i]) for i in range(int(block["dc"].shape[0]))]
            out[f"{population}/{group}"] = torch.stack(rows) if rows else torch.empty(0, len(CELL_COLUMNS), dtype=torch.float64)
    return out


def score(measured: Mapping[str, Any], tables: Mapping[str, Mapping[str, torch.Tensor]], lock: Mapping[str, Any]) -> dict[str, Any]:
    """The four conditions through the canonical path (each fresh pair once) against the locked envelopes, and the cell
    kernel checked against a direct flattened recomputation from the per-noun measurements (1e-10). No aggregate
    label."""
    cells = fresh_cells(measured, tables)
    conditions: dict[str, Any] = {}
    worst = {"max_difference": 0.0, "at": "", "n_checked": 0}
    for condition in CONDITIONS:
        population, group = condition.split("/")
        entry = lock["conditions"][condition]
        scored = score_selection(cells[condition], torch.arange(int(cells[condition].shape[0])), entry["envelope"]["bound"], condition)
        block, table = measured[population]["blocks"][group], tables[population][group]
        y, p0, p1, c = (tensor.double().reshape(-1) for tensor in (block["dc"], table[:, 0], table[:, 1], block["ceiling"]))
        sst = float(((y - y.mean()) ** 2).sum())
        sse0, sse1, ssec = (float(((y - k) ** 2).sum()) for k in (p0, p1, c))
        defined = sst > 0 and sse0 - ssec >= GAP_MIN * sst
        direct = {"SST": sst, "g": (sse0 - sse1) / (sse0 - ssec) if defined else math.nan}
        for name, value in direct.items():
            difference = agreement(scored["SST"] if name == "SST" else (scored["g"] if scored["g"] is not None else math.nan), value)
            worst["n_checked"] += 1
            if difference > worst["max_difference"]:
                worst.update({"max_difference": difference, "at": f"{condition}/{name}"})
        conditions[condition] = rc.json_safe({**scored, "population": population, "group": group, "statistic": STATISTIC, "envelope": entry["envelope"],
                                              "guard": entry["guard"], "reading": SEMANTICS["results"][scored["result"]], "n_pairs": int(cells[condition].shape[0])})
    worst.update({"tolerance": TOLERANCES["kernel"], "passed": worst["max_difference"] <= TOLERANCES["kernel"]})
    if not worst["passed"]:
        raise KernelCheckError(worst)
    return {"conditions": conditions, "kernel_check": rc.json_safe(worst), "aggregate_label": None}


def fresh_descriptives(measured: Mapping[str, Any], tables: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    """Per template and per stratum, the canonical statistics on subsets of the fresh pairs (descriptive; the
    conditions use the full groups)."""
    cells = fresh_cells(measured, tables)
    out: dict[str, Any] = {}
    for population in POPULATIONS:
        units = measured[population]["units"]
        for group in GROUPS:
            pairs = units.pairs[group]
            condition_cells = cells[f"{population}/{group}"]
            subsets = {f"template/{template}": [row for row, (_, f) in enumerate(pairs) if units.frames[f].template_id == template]
                       for template in sorted({units.frames[f].template_id for _, f in pairs})}
            subsets.update({f"stratum/{stratum}": [row for row, (t, _) in enumerate(pairs) if units.tokens[t]["class"] == stratum] for stratum in STRATA})
            for name, rows in subsets.items():
                if rows:
                    out[f"{population}/{group}/{name}"] = rc.json_safe(score_selection(condition_cells, torch.tensor(rows, dtype=torch.int64), None, name))
    return out


def comparator_inputs(progs: ul.ModelPrograms, ctx: ul.PairContext, reference_id: int) -> dict[str, torch.Tensor]:
    """The exposed spike's cheaper block-0 rules at the cue position (descriptive only): the value term alone; the
    self logit shifted relative to the mean query shift; the oracle self weight with the other keys proportional; the
    first-order (linear-response) softmax; the exact terms of the 4 and 6 highest-ranked heads."""
    program0, frame = progs.program0, ctx.frame
    x0_all = ul.reference_embeddings(progs.weights, frame, int(reference_id))
    rr = atp.ReferenceRow(program0, [x.double() for x in x0_all[: frame.p_c + 1]])
    normed = program0.normalize(progs.weights.W_E[int(ctx.token_id)].double())
    q_new, k_new, v_new = rr.cue(normed)
    values_new = rr.values.clone()
    values_new[:, frame.p_c] = v_new
    o_new_all, o_ref_all = program0.output(values_new), program0.output(rr.values)
    a = rr.A_ref[:, frame.p_c]
    head_ref = torch.einsum("hk,hkd->hd", rr.A_ref, o_ref_all)
    rest = (head_ref - a[:, None] * o_ref_all[:, frame.p_c]) / (1.0 - a)[:, None]
    row_exact = rr.row(q_new, k_new)
    exact_h = torch.einsum("hk,hkd->hd", row_exact, o_new_all) - head_ref
    value_h = a[:, None] * (o_new_all[:, frame.p_c] - o_ref_all[:, frame.p_c])
    scores = torch.einsum("he,hke->hk", q_new, rr.keys) / program0.scale
    scores = scores.clone()
    scores[:, frame.p_c] = (q_new * k_new).sum(-1) / program0.scale
    shift = scores - rr.scores_ref
    mean_off = (rr.A_ref[:, : frame.p_c] * shift[:, : frame.p_c]).sum(-1) / (1.0 - a)
    a_relative = torch.sigmoid(torch.log(a) - torch.log1p(-a) + shift[:, frame.p_c] - mean_off)
    linear = rr.A_ref * (shift - (rr.A_ref * shift).sum(-1, keepdim=True))
    base = ctx.factors.delta_e + ctx.factors.d_emb
    n_heads = int(a.shape[0])
    order = [h for h in HEAD_ORDER if h < n_heads]
    return {"value_only": base + value_h.sum(0),
            "relative_self_logit": base + value_h.sum(0) + ((a_relative - a)[:, None] * (o_new_all[:, frame.p_c] - rest)).sum(0),
            "oracle_self_weight": base + value_h.sum(0) + ((row_exact[:, frame.p_c] - a)[:, None] * (o_new_all[:, frame.p_c] - rest)).sum(0),
            "linear_response": base + value_h.sum(0) + torch.einsum("hk,hkd->d", linear, o_new_all),
            "heads_4": base + exact_h[order[:4]].sum(0), "heads_6": base + exact_h[order[:6]].sum(0)}


def comparators(progs: ul.ModelPrograms, confirmation: Confirmation023, measured: Mapping[str, Any], states: Mapping[str, Mapping[str, rd.FrameState020]],
                tables: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    """Each cheaper rule composed exactly like ``P1`` (the reduced layers 1–2, 022's exact ``T`` at the target) and
    scored descriptively on each condition: ``g_rule = (SSE0 − SSE_rule) / (SSE0 − SSEC)``. No outcome force."""
    out: dict[str, Any] = {}
    for population in POPULATIONS:
        units = measured[population]["units"]
        rows16: dict[str, Any] = {}
        for group in GROUPS:
            block, table = measured[population]["blocks"][group], tables[population][group]
            sse = {name: 0.0 for name in COMPARATORS}
            sse0 = ssec = 0.0
            for row, (t, f) in enumerate(units.pairs[group]):
                frame, token = units.frames[f], units.tokens[t]
                state = states[population][frame.frame_id]
                if frame.frame_id not in rows16:
                    rows16[frame.frame_id] = ul.reference_rows_017(progs.programs, state)
                reference_id = int(confirmation.reference_ids[frame.template_id])
                ctx = ul.pair_context(progs, frame, state, reference_id, int(token["token_id"]), token["word"], rows16[frame.frame_id])
                y = block["dc"][row].double()
                sse0 += float(((y - table[row, 0]) ** 2).sum())
                ssec += float(((y - block["ceiling"][row]) ** 2).sum())
                s17 = state.state_017
                for name, u_pc in comparator_inputs(progs, ctx, reference_id).items():
                    dx3 = ul.reduced_chain(progs.chain, ctx.rows16, s17.x1_all, s17.x2_all, frame.p_c, frame.p_t, frame.template_id, u_pc, ctx.factors.attn_pt)
                    sse[name] += float(((y - ul.contrast_of(progs, state, dx3)) ** 2).sum())
            denominator = sse0 - ssec
            out[f"{population}/{group}"] = {name: (sse0 - value) / denominator if denominator > 0 else None for name, value in sse.items()}
    return rc.json_safe(out)


# ---------------------------------------------------------------------------
# The results state and the phase rules (022's discipline).

RESULTS_SCHEMA_VERSION = 1
PHASES = ("validate", "extract", "freeze", "calibrate", "lock", "confirm", "report")
STATE_PHASES = ("extract", "calibrate", "lock", "confirm", "report")


def new_results_state(*, digests: Mapping[str, str], protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {"schema_version": RESULTS_SCHEMA_VERSION, "experiment": EXPERIMENT, "run_id": pm.sha256_text(pm.canonical_json(dict(digests)) + protocol_code_commit + pm.utc_now())[:16],
            "created_at": pm.utc_now(), "inputs": {key: digests[key] for key in DIGEST_KEYS}, "module_blobs": dict(FROZEN_BLOBS), "design": dict(DESIGN), "plan": dict(PLAN),
            "protocol_code_commit": protocol_code_commit, "git_dirty": False, "model": {"model_id": models_module.PYTHIA_70M.model_id, "revision": models_module.PYTHIA_70M.revision},
            "versions": dict(versions), "phases": {phase: {"status": "not_started"} for phase in STATE_PHASES}, "executed_prompt_keys": [], "executed_noun_keys": [],
            "extract": {}, "calibration": {}, "confirmation_023": None, "lock": None, "confirmation": None, "report": None}


def assert_phase_allowed(phase: str, state: Mapping[str, Any]) -> None:
    status = {name: entry["status"] for name, entry in state["phases"].items()}
    if phase == "extract":
        entry = state["phases"]["extract"]
        if status["extract"] == "running" and entry.get("incidents") and not state["extract"].get("data_sha256"):
            return  # resumable only after a recorded incident, at a new commit (the runner refuses the incident's commit)
        if status["extract"] != "not_started":
            raise PhaseError(f"extract is {status['extract']}; the extraction runs once in this protocol version (a stop for review is never retried automatically)")
    elif phase == "calibrate":
        if status["extract"] != "complete":
            raise PhaseError(f"calibrate requires the completed extract phase (it is {status['extract']})")
        calibration = state.get("calibration") or {}
        if status["calibrate"] == "running" and calibration.get("incidents") and not calibration.get("record_sha256"):
            # A rerun after an incident happens only at a new commit (the runner refuses the incident's commit) that changes
            # no scientific path since extract (calibrate refuses any other), e.g. a documentation commit recording an
            # interruption. The calibration is a deterministic function of the committed artifact, so such a rerun
            # reproduces it; a code fix is a new protocol version.
            return
        if status["calibrate"] != "not_started":
            raise PhaseError(f"calibrate is {status['calibrate']}; the calibration runs once in this protocol version (a stop for review is never retried automatically)")
    elif phase == "lock":
        if status["calibrate"] != "complete":
            raise PhaseError(f"lock requires the completed calibrate phase (it is {status['calibrate']})")
        if status["lock"] == "complete":
            raise PhaseError("lock already written; a new candidate lock requires a new protocol version")
        if state["phases"]["lock"].get("incidents"):
            raise PhaseError("a lock identity incident is recorded; lock is refused until the reviewer decides (a new protocol version)")
    elif phase == "confirm":
        if status["lock"] != "complete":
            raise PhaseError("confirm requires the lock phase")
        if status["confirm"] != "not_started":
            raise PhaseError("confirm already started; it runs once, never resumes, and a second attempt requires a new protocol version")
        if state["phases"]["confirm"].get("incidents"):
            raise PhaseError("an I7 incident is recorded; confirm is refused until the reviewer decides (a new protocol version)")
    elif phase == "report":
        calibration = state.get("calibration") or {}
        if status["calibrate"] not in ("complete", "stopped_for_review") and not calibration.get("incidents"):
            raise PhaseError("report requires a calibrate phase that completed, stopped for review or recorded an incident")
    else:
        raise PhaseError(f"unknown phase {phase}")


def assert_ledger_isolated(ledger: Sequence[str], forbidden: frozenset[str], what: str) -> None:
    overlap = set(ledger) & forbidden
    if overlap:
        raise PhaseError(f"{what} holds {len(overlap)} forbidden keys, e.g. {sorted(overlap)[:2]}")


# ---------------------------------------------------------------------------
# The report.


def _fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def cdf_percentile(values: torch.Tensor, defined: torch.Tensor, fresh: float | None) -> dict[str, Any]:
    """The fraction of *defined* calibration draws at or below the fresh value, and the undefined count; descriptive."""
    n, undefined = int(defined.sum()), int((~defined).sum())
    if fresh is None or not math.isfinite(float(fresh)) or n == 0:
        return {"cdf_percentile": None, "defined": n, "undefined": undefined}
    return {"cdf_percentile": int(((values.double() <= float(fresh)) & defined).sum()) / n, "defined": n, "undefined": undefined}


def draw_values_verified(arrays: Mapping[str, torch.Tensor], record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if {key: rc.tensor_digest(value) for key, value in arrays.items()} != dict(record["draw_arrays_sha256"]):
        raise PhaseError("the calibration draw arrays are not the ones the record binds")
    return dict(arrays)


def render_report(state: Mapping[str, Any], record: Mapping[str, Any] | None, arrays: Mapping[str, torch.Tensor] | None) -> str:
    phases = ", ".join(f"{name} {entry['status']}" for name, entry in state["phases"].items())
    commits = {name: entry.get("commit") or entry.get("confirm_commit") for name, entry in state["phases"].items() if entry.get("commit") or entry.get("confirm_commit")}
    lines = ["# Experiment 023 — report", "", f"- Run `{state['run_id']}`; phase commits {commits}; phases: {phases}",
             f"- Design revision {state['design']['revision']} (`{state['design']['commit']}`), plan revision {state['plan']['revision']} (`{state['plan']['commit']}`)",
             f"- Scope: {SCOPE}.", ""]
    for name in STATE_PHASES:
        for entry in state["phases"][name].get("incidents", []):
            lines.append(f"- **{name} incident** at `{entry['commit']}`: {entry['message']}")
    for entry in (state.get("calibration") or {}).get("incidents", []):
        lines.append(f"- **Calibration incident** at `{entry['commit']}`: {entry['message']}")
    incident = (state.get("confirmation") or {}).get("incident")
    if incident:
        lines.append(f"- **Confirmation incident** at `{incident['commit']}`: {incident['message']} — the one-shot confirmation is consumed; nothing is retried")
    stop = (state.get("calibration") or {}).get("stop")
    if stop:
        lines += ["", "## Calibration stopped for review (no envelope was written)", "",
                  f"- Undefined draws per condition, against the stop at {stop['threshold']} or more: "
                  + ", ".join(f"{condition} {stop['counts'].get(condition)}" for condition in CONDITIONS),
                  f"- At or above the stop: {', '.join(sorted(stop['offending']))}; never retried automatically — the reviewer decides"]
    extract = state.get("extract") or {}
    if extract.get("checks"):
        checks = extract["checks"]
        lines += ["", "## Extraction (exposed cells from Experiment 022's verified table)", "",
                  f"- E1 digests and orders verified; E2 and E3 bit for bit; E4 max |ΔP1| {_fmt(checks['E4'].get('max'), 3)}; E5 max {checks['E5']['max_difference']:.1e} over "
                  f"{checks['E5']['n_checked']}; E6 pairs {checks['E6']['max_per_pair']:.1e}, draws {checks['E6']['max_per_draw']:.1e}",
                  f"- Artifact data sha256 `{extract.get('data_sha256')}`, index `{extract.get('index_sha256')}`"]
    if record is not None:
        lines += ["", f"## Calibration (exposed only, B = {record['constants']['B']})", "",
                  f"- Kernel/loop check {record['kernel_check']['max_difference']:.1e} over {record['kernel_check']['n_checked']}; E6 max {record['e6_max']:.1e}",
                  "", "| condition | envelope F | min of defined draws | median | max | undefined | guard-bound | PASS | ENVELOPE_ONLY | GUARD | NOT_INTERPRETABLE |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for condition in CONDITIONS:
            entry = record["conditions"][condition]
            rates, summary = entry["rates"], entry["summary"]
            lines.append(f"| {condition} | ≥ v₍{entry['envelope']['rank']}₎ = {entry['envelope']['bound']:.6f} | {_fmt(summary['min'], 6)} | {_fmt(summary['median'], 6)} | "
                         f"{_fmt(summary['max'], 6)} | {record['undefined_counts'][condition]} | {'yes' if entry['guard_bound'] else 'no'} | {rates['PASS']:.4f} | "
                         f"{rates['ENVELOPE_ONLY_FAILURE']:.4f} | {rates['GUARD_FAILURE']:.4f} | {rates['NOT_INTERPRETABLE']:.4f} |")
        lines += ["", f"- Joint rate, all four passing (descriptive only): {record['joint_rates']['all_four_pass']:.4f}"]
    confirmation = state.get("confirmation") or {}
    conditions = confirmation.get("conditions")
    if conditions:
        lines += ["", "## The four conditions (the result; no aggregate label)", "",
                  "| condition | g | envelope F | meaning guard | result | gap (SSE0 − SSEC)/SST | R²₀ | R²₁ | R²_C | ceiling-limited | CDF percentile (descriptive) | pairs |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for condition in CONDITIONS:
            entry = conditions[condition]
            percentile = cdf_percentile(arrays[f"{condition}/g"], arrays[f"{condition}/defined"].bool(), entry["g"]) if arrays is not None else {}
            shown = "—" if not percentile else f"{_fmt(percentile.get('cdf_percentile'), 4)} ({percentile['defined']} / {percentile['undefined']})"
            lines.append(f"| {condition} | {_fmt(entry['g'], 6)} | {entry['envelope']['bound']:.6f} | g ≥ {GUARD_MIN} | **{entry['result']}** | {_fmt(entry['gap'], 4)} | "
                         f"{_fmt(entry['R2_0'], 4)} | {_fmt(entry['R2_1'], 4)} | {_fmt(entry['R2_C'], 4)} | {'yes' if entry['ceiling_limited'] else 'no'} | {shown} | "
                         f"{entry['n_pairs']} |")
        lines += ["", "Readings (frozen, design revision 2):", ""]
        lines += [f"- {condition}: **{conditions[condition]['result']}** — {conditions[condition]['reading']}." for condition in CONDITIONS]
        lines += ["", f"- The ceiling: {SEMANTICS['ceiling']}.", f"- Aggregate: {SEMANTICS['aggregate']}.", f"- What a pass does not show: {SEMANTICS['not_shown']}.",
                  f"- Kernel against the direct recomputation: {confirmation['kernel_check']['max_difference']:.1e} over {confirmation['kernel_check']['n_checked']}"]
        descriptives = confirmation.get("descriptives") or {}
        lines += ["", "## Descriptive records (no outcome force)", ""]
        gates = confirmation.get("gates") or {}
        lines.append("- Stage-2 identities (maxima): " + ", ".join(f"{name} {gates[name].get('max')} (tolerance {TOLERANCES[name]:.0e})" for name in sorted(gates)))
        for key, entry in (descriptives.get("subsets") or {}).items():
            lines.append(f"- {key}: g {_fmt(entry.get('g'), 4)}; gap {_fmt(entry.get('gap'), 4)}; R²₁ {_fmt(entry.get('R2_1'), 4)}; R²_C {_fmt(entry.get('R2_C'), 4)}")
        for key, entry in (descriptives.get("comparators") or {}).items():
            lines.append(f"- Cheaper block-0 rules, {key} (descriptive g): " + ", ".join(f"{name} {_fmt(entry.get(name), 4)}" for name in COMPARATORS))
        for key, entry in (descriptives.get("p1_dx3_relative_error") or {}).items():
            lines.append(f"- P1 Δx3 relative error {key}: " + ", ".join(f"{slot} median {value['median']:.2e} max {value['max']:.2e}" for slot, value in entry.items()))
        for key, profile in (descriptives.get("block0_profile") or {}).items():
            lines.append(f"- Block-0 profile {key}: self-weight {profile['reference_self_weight_median']}; p_t→p_c {profile['attention_pt_to_pc_median']}")
        for name, failure in sorted((descriptives.get("failures") or {}).items()):
            lines.append(f"- The descriptive record {name} was not computed ({failure['type']}: {failure['message']}); it has no outcome force and the four results stand")
        for frame_id, frame in sorted(((confirmation.get("stage1") or {}).get("frames") or {}).items()):
            lines.append(f"- Validity (descriptive, selects nothing) {frame_id}: {frame['validity'].get('valid')}")
    lines.append("")
    return "\n".join(lines)
