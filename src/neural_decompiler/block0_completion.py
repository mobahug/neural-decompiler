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
