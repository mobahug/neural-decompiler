"""Experiment 021: the Experiment 020 question with floors from a preregistered exposed-only calibration.

Design revision 3 (``0ac46aa``) keeps everything of Experiment 020 — the pinned model and runtime, the readout
program (``readout_decompilation.py`` is never edited here, and its git blob is checked before every phase), the
inherited 011/012/017 objects, the untouched confirmation set, the populations, the two-stage procedure, the
tolerances and the labels — and replaces only the numeric floors.

The floors come from one exposed-only calibration run. Experiment 020's exposed table is re-materialized from exactly
the keys of its execution ledger and must reproduce 020's recorded exploration at ``1e-9`` before anything is drawn.
10,000 SHA-indexed pseudo-confirmation base draws then give every statistic's distribution for every admissible
evaluable population: one Y1 row, 64 Y2 valid-frame compositions and 84 Y3 scorable-noun compositions, each scored on
a prefix of the same base draws. Every floor is the 2.5 % directional tail, ``R²``-type floors are clamped at zero,
and one pass predicate decides every condition, both for the calibration's pass rates and at confirmation.

Nothing in this module chooses, rounds, averages or replaces a scientific value after it is computed. The one
statistics kernel serves the draws and the fresh tables alike, and it is checked against Experiment 020's own
``pair_statistics`` / ``noun_statistics`` before any floor exists; a disagreement stops the phase with its location.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from . import attention_patterns as atp
from . import cue_suppression as cs
from . import head_transport as ht
from . import layer_correction as lc
from . import plural_mechanism as pm
from . import readout_decompilation as rd
from .behavior import validate_json_safe
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen constants (design revision 3).

EXPERIMENT_DIR = "experiments/021-corrected-readout-confirmation"
CALIBRATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/calibration-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREDICTIONS_RELATIVE_PATH = f"{EXPERIMENT_DIR}/predictions.md"
DESIGN = {"path": "docs/superpowers/specs/2026-09-22-experiment-021-corrected-readout-confirmation-design.md", "revision": 3, "commit": "0ac46aa"}
SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", *rd.SCIENTIFIC_PATH_PREFIXES[1:])
NON_SCIENTIFIC_PATHS = (CALIBRATION_RELATIVE_PATH, LOCK_RELATIVE_PATH, PREDICTIONS_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md")
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/",)
RESULTS_SCHEMA_VERSION = 1
CALIBRATION_SCHEMA_VERSION = 1
PHASES = ("validate", "calibrate", "lock", "confirm", "report")

# The program, byte for byte: Experiment 020's module at ce3766b.
PROGRAM_BLOB_SHA1 = "caa73b40192f4c910dc63371bd19db75a3258339"

# Experiment 020's closure (7f4a4f8), its committed evidence extract and its gitignored results state.
CLOSURE_020_RELATIVE_PATH = "experiments/020-readout-decompilation/closure.json"
EXTRACT_020_RELATIVE_PATH = "experiments/020-readout-decompilation/evidence/exploration-record-2026-09-22.json"
RESULTS_020_RELATIVE_PATH = "outputs/experiment-020/results.json"
CLOSURE_020_SHA256 = "f2b1b5e793dad75f648fe1fb8347fde75c128a3ea00c0b14b4946814068fe9e0"
EXTRACT_020_SHA256 = "99f25ee6000d74cdc307fbaf645fd2b83a4c192e7d201c1c3249b4b8013e2dea"
RESULTS_020_FILE_SHA256 = "da63b8c29f9553a9da62bea7a442e11cef999ccddc71a7a332ac7c938abc9e00"
RESULTS_020_STATE_SHA256 = "2e5485dccb39c018eb02ee8d0a3086004399690a4dfd9c114008f057b23bf8f5"
CONFIRMATION_020_SHA256 = "e098e2b44a1702d2b35c20db2ce111358897ea867996024c3e51a79f379c309d"
LOCK_DIGESTS = {"lock_011": "769bfeacd7c49fc18bed2ff3c5cf8d5ea2be4231e9f8e9c493819b5d7ba69d0b",  # the inherited objects, read verbatim
                "lock_012": "830abc3b4a2d86a8c904cc467d7b08223f0b197623945f1edb8f5edd55c2a6eb",
                "lock_017": "b4fc9014ade7d21d2cd2e5be391ce6234e46fb46887880c0c4d03517d411ed72"}

# The draws and the floor rule.
B = 10_000  # base draws
ALPHA_PER_MILLE = 25  # the 2.5 % directional tail: the ⌈0.025·B⌉-th value, in integer arithmetic
SHARES = {"cue": 80, "frame": 75, "noun": 90}  # per cent; k = ⌈p·n/100⌉
SLOTS = {"cue": 6, "frame": 6, "noun": 8}  # the fresh set's per-stratum maxima: 6/6/6/6 cues, 6/6/6 frames, 8/8/8 nouns
VARIANTS = ("primary", "all-frames", "lineage")  # only "primary" sets floors
CUE_CLASSES = ("determiner-like", "quantity", "possessive-or-pronoun", "adjective")
CUE_COHORTS = tuple(f"confirmation-0{number}" for number in range(11, 20))
LINEAGE_COHORTS = ("confirmation-017", "confirmation-018", "confirmation-019")
FRAME_ORIGINS = ("confirmation-017", "confirmation-018", "confirmation-019")  # the frames that entered no inherited fit
Y2_AXES = tuple(pm.TEMPLATE_ORDER)  # ("cardinal", "quantifier", "coordinated-adjective")
RULE_CLASSES = tuple(rd.NOUN_CANDIDATES)  # ("simple-suffix", "sibilant-es", "consonant-y")
EXPECTED_POOL_COUNTS = {
    "cues": {"determiner-like": 45, "quantity": 45, "possessive-or-pronoun": 36, "adjective": 49},
    "lineage_cues": {"determiner-like": 15, "quantity": 15, "possessive-or-pronoun": 12, "adjective": 18},
    "frames_unscreened": {template: 14 for template in Y2_AXES},
    "frames_all_unscreened": {template: 36 for template in Y2_AXES},
    "nouns": {"simple-suffix": 40, "sibilant-es": 19, "consonant-y": 20},
}
EXPECTED_COHORT_SIZES = (19, 20)

# Tolerances. The first two are revision 3's; the last two are implementation-only correctness guards with no
# scientific force (approved in the review of the implementation plan).
RECONSTRUCTION_TOLERANCE = 1e-9  # the reproduction gate, every numeric leaf
LOCKED_STATE_TOLERANCE = 1e-9  # the environment check: a re-captured reference state against 020's (013–019 constant)
MIN_SCREENED_FRAMES_PER_TEMPLATE = 6  # the calibration precondition
STATISTIC_AGREEMENT_TOLERANCE = 1e-10  # the kernel against 020's direct statistics, on the scale below
AGREEMENT_SCALE = "|kernel - direct| / max(1, |direct|)"
CROSS_CHECK_DRAWS = 16  # base draws of every row checked against the direct statistics inside calibrate

Y1_STATISTICS = ("token_mean_r2", "pair_mean_r2", "cue_mae_k80", "pooled_mae")
Y2_STATISTICS = ("frame_mean_r2", "frame_r2_k75", "cue_final_r2", "coordinated_r2")
Y3_STATISTICS = ("noun_median_r2", "noun_r2_k90", "noun_slope_dev_k90", "noun_bias_k90")
STATISTICS = {"Y1": Y1_STATISTICS, "Y2": Y2_STATISTICS, "Y3": Y3_STATISTICS}
R2_STATISTICS = frozenset({"token_mean_r2", "pair_mean_r2", "frame_mean_r2", "frame_r2_k75", "cue_final_r2", "coordinated_r2", "noun_median_r2", "noun_r2_k90"})
ERROR_STATISTICS = frozenset({"cue_mae_k80", "pooled_mae", "noun_slope_dev_k90", "noun_bias_k90"})
KIND = {name: ("r2" if name in R2_STATISTICS else "error") for names in STATISTICS.values() for name in names}
FULL_ROWS = {"Y1": (), "Y2": (6, 6, 6), "Y3": (8, 8, 8)}

PhaseError = rd.PhaseError


class CrossCheckError(pm.IncidentError):
    """The kernel disagreed with Experiment 020's direct statistics: an implementation incident, with its location."""

    def __init__(self, details: Mapping[str, Any]):
        self.details = dict(details)
        difference = self.details["max_difference"]
        super().__init__(f"kernel/direct cross-check failed: {'undefined on one side' if difference is None else f'{difference:.3e}'} above {STATISTIC_AGREEMENT_TOLERANCE:.0e} "
                         f"at {self.details['outcome']} row {self.details['row']} draw {self.details['draw']} statistic {self.details['statistic']} "
                         f"({self.details['n_exceeding']} of {self.details['n_checked']} row-draws exceed)")


def tail_index(draws: int) -> int:
    """⌈0.025·draws⌉ in integer arithmetic (250 for 10,000 draws)."""
    return (ALPHA_PER_MILLE * draws + 999) // 1000


def k_of(percent: int, n: int) -> int:
    """⌈percent·n/100⌉ in integer arithmetic: "at least percent % of n units" is "at least k units"."""
    return (percent * n + 99) // 100


def row_key(composition: Sequence[int]) -> str:
    return "/".join(str(int(count)) for count in composition) if composition else "all"


def content_digest(payload: Mapping[str, Any]) -> str:
    return pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    """sha256 over the shape and the little-endian row-major bytes (int64 or float64)."""
    array = tensor.detach().cpu().contiguous()
    kind = "<i8" if array.dtype in (torch.int64, torch.int32) else "<f8"
    data = array.numpy().astype(kind, copy=False).tobytes()
    return hashlib.sha256(json.dumps(list(array.shape)).encode("ascii") + b"|" + kind.encode("ascii") + b"|" + data).hexdigest()


def json_safe(value: Any) -> Any:
    """A record fit for the results state: every non-finite float becomes ``None`` (an incident must always be writable)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# The program blob and Experiment 020's closure.


def program_blob_sha1(path: Path) -> str:
    """git's blob id of a file, computed without git."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def assert_program_blob(path: Path | None = None) -> str:
    """The readout program actually imported is Experiment 020's, byte for byte."""
    blob = program_blob_sha1(Path(path) if path is not None else Path(rd.__file__))
    if blob != PROGRAM_BLOB_SHA1:
        raise PhaseError(f"readout_decompilation.py has blob {blob}, not Experiment 020's {PROGRAM_BLOB_SHA1}; Experiment 021 runs the unchanged program only")
    return blob


def verify_020_closure(root: Path, confirmation: rd.Confirmation020) -> dict[str, Any]:
    """Every digest of Experiment 020's closure, and its confirmation isolation, before anything of 021 runs.

    Returns 020's results state (read only): its ledger, its locked reference states, its template bases and its
    recorded exploration are the calibration's only exposed source.
    """
    closure_path = root / CLOSURE_020_RELATIVE_PATH
    if not closure_path.exists():
        raise PhaseError(f"{CLOSURE_020_RELATIVE_PATH} is missing; Experiment 021 requires Experiment 020's closure")
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if closure.get("content_sha256") != content_digest(closure) or closure["content_sha256"] != CLOSURE_020_SHA256:
        raise PhaseError("Experiment 020's closure record does not verify against its frozen digest")
    if closure.get("experiment") != "020" or closure.get("status") != "closed" or closure.get("lock") is not None or closure.get("confirmation") is not None or closure.get("labels") is not None:
        raise PhaseError("Experiment 020's closure record does not state a closure without lock, confirmation or label")
    if confirmation.content_sha256 != CONFIRMATION_020_SHA256 or closure["confirmation_set"]["content_sha256"] != confirmation.content_sha256:
        raise PhaseError("the confirmation set is not Experiment 020's frozen, closed one")
    if int(closure["confirmation_set"].get("executed_keys", -1)) != 0:
        raise PhaseError("Experiment 020's closure record does not show an unexecuted confirmation set")
    extract_path = root / EXTRACT_020_RELATIVE_PATH
    if not extract_path.exists():
        raise PhaseError(f"{EXTRACT_020_RELATIVE_PATH} is missing")
    extract = json.loads(extract_path.read_text(encoding="utf-8"))
    if extract.get("content_sha256") != content_digest(extract) or extract["content_sha256"] != EXTRACT_020_SHA256 or extract["content_sha256"] != closure["evidence"]["exploration_record_content_sha256"]:
        raise PhaseError("Experiment 020's committed exploration extract does not verify")
    results_path = root / RESULTS_020_RELATIVE_PATH
    if not results_path.exists():
        raise PhaseError(f"{RESULTS_020_RELATIVE_PATH} (gitignored) is not on this machine; the exposed calibration needs Experiment 020's recorded state")
    results_sha = file_sha256(results_path)
    if results_sha != RESULTS_020_FILE_SHA256 or results_sha != closure["explore"]["results_file_sha256"]:
        raise PhaseError("Experiment 020's results file is not the closed one")
    state = rd.load_results_state(results_path)
    if state["state_sha256"] != RESULTS_020_STATE_SHA256 or state["state_sha256"] != closure["explore"]["results_state_sha256"]:
        raise PhaseError("Experiment 020's results state digest is not the closed one")
    if state["lock"] is not None or state["confirmation"] is not None or state["phases"]["explore"]["status"] != "complete":
        raise PhaseError("Experiment 020's results state is not a completed exploration without lock or confirmation")
    rd.assert_confirmation_untouched(state, confirmation)
    fresh_noun_keys = {noun.key for noun in confirmation.nouns}
    if fresh_noun_keys & set(state["executed_noun_keys"]):
        raise PhaseError("Experiment 020's ledger scored a fresh noun")
    rd.assert_fresh_nouns_absent(state["exploration"], confirmation)
    digests = {frame_id: rd.state_digest(entry) for frame_id, entry in sorted(state["exploration"]["locked_states"].items())}
    if digests != dict(extract["exploration"]["locked_state_digests"]):
        raise PhaseError("Experiment 020's locked reference states do not match the committed extract's digests")
    return {"closure": closure, "extract": extract, "state": state, "exploration": state["exploration"], "ledger": frozenset(state["executed_prompt_keys"]),
            "digests": {"closure_020": closure["content_sha256"], "extract_020": extract["content_sha256"], "results_020_file": results_sha,
                        "results_020_state": state["state_sha256"], "confirmation_020": confirmation.content_sha256}}


# ---------------------------------------------------------------------------
# Pools (frozen provenance only) and the admissible rows.


@dataclass(frozen=True)
class CalibrationPools:
    cues: Mapping[str, tuple[tuple[str, int], ...]]  # class -> ((word, token id), ...), token-id order
    lineage_cues: Mapping[str, tuple[tuple[str, int], ...]]
    cohorts: Mapping[str, tuple[tuple[str, int], ...]]  # cohort -> its cues of the four classes, token-id order
    frames_unscreened: Mapping[str, tuple[str, ...]]  # template -> frame ids first confirmed in 017/018/019
    frames_all_unscreened: Mapping[str, tuple[str, ...]]  # template -> every exposed frame id
    nouns: Mapping[str, tuple[str, ...]]  # rule class -> lexical keys of the scorable exposed nouns
    screen: Mapping[str, bool] | None = None  # frame id -> valid, set once the validity screen ran

    def frames(self, *, all_frames: bool = False) -> dict[str, tuple[str, ...]]:
        if self.screen is None:
            raise PhaseError("the frame pools exist only after the validity screen")
        source = self.frames_all_unscreened if all_frames else self.frames_unscreened
        return {template: tuple(frame_id for frame_id in ids if self.screen[frame_id]) for template, ids in source.items()}

    def screened(self, screen: Mapping[str, bool]) -> "CalibrationPools":
        return CalibrationPools(self.cues, self.lineage_cues, self.cohorts, self.frames_unscreened, self.frames_all_unscreened, self.nouns, dict(screen))

    def to_json(self) -> dict[str, Any]:
        entry = {"cues": {c: [[w, t] for w, t in v] for c, v in self.cues.items()}, "lineage_cues": {c: [w for w, _ in v] for c, v in self.lineage_cues.items()},
                 "cohorts": {s: [w for w, _ in v] for s, v in self.cohorts.items()}, "frames_unscreened": {t: list(v) for t, v in self.frames_unscreened.items()},
                 "frames_all_unscreened": {t: list(v) for t, v in self.frames_all_unscreened.items()}, "nouns": {r: list(v) for r, v in self.nouns.items()}}
        if self.screen is not None:
            entry["frames"] = {t: list(v) for t, v in self.frames().items()}
            entry["frames_all"] = {t: list(v) for t, v in self.frames(all_frames=True).items()}
        return entry


def build_pools(pool: Any) -> CalibrationPools:
    """The pseudo-fresh pools, by frozen provenance only (never by any output)."""
    excluded = set(pool.reference_ids.values()) | {pool.token_id(name) for name in pool.plural_cue.values()}
    cues = {cls: tuple(sorted(((word, token_id) for word, token_id in pool.tokens if pool.token_category[word] == cls and pool.token_source[word] in CUE_COHORTS), key=lambda entry: entry[1]))
            for cls in CUE_CLASSES}
    for entries in cues.values():
        if any(token_id in excluded for _, token_id in entries):
            raise PhaseError("a template reference or plural cue entered the pseudo-fresh cue pool")
    lineage = {cls: tuple(entry for entry in cues[cls] if pool.token_source[entry[0]] in LINEAGE_COHORTS) for cls in CUE_CLASSES}
    cohorts = {cohort: tuple(sorted((entry for cls in CUE_CLASSES for entry in cues[cls] if pool.token_source[entry[0]] == cohort), key=lambda entry: entry[1])) for cohort in CUE_COHORTS}
    frames = {template: tuple(sorted(frame.frame_id for frame in pool.frames if frame.template_id == template and pool.frame_origin[frame.frame_id] in FRAME_ORIGINS)) for template in Y2_AXES}
    frames_all = {template: tuple(sorted(frame.frame_id for frame in pool.frames if frame.template_id == template)) for template in Y2_AXES}
    nouns = {rule: tuple(sorted(noun.lexical_key for noun in pool.nouns if noun.single_token and noun.rule_class == rule)) for rule in RULE_CLASSES}
    return CalibrationPools(cues, lineage, cohorts, frames, frames_all, nouns)


def assert_pool_counts(pools: CalibrationPools) -> None:
    counts = {"cues": {c: len(v) for c, v in pools.cues.items()}, "lineage_cues": {c: len(v) for c, v in pools.lineage_cues.items()},
              "frames_unscreened": {t: len(v) for t, v in pools.frames_unscreened.items()}, "frames_all_unscreened": {t: len(v) for t, v in pools.frames_all_unscreened.items()},
              "nouns": {r: len(v) for r, v in pools.nouns.items()}}
    if counts != EXPECTED_POOL_COUNTS:
        raise PhaseError(f"the pseudo-fresh pools are not the frozen ones: {counts}")
    if any(len(members) not in EXPECTED_COHORT_SIZES for members in pools.cohorts.values()) or len(pools.cohorts) != len(CUE_COHORTS):
        raise PhaseError("the historical cue cohorts are not the frozen nine of 19–20 cues")


def assert_lock_digests(digests: Mapping[str, str]) -> None:
    """The inherited 011/012/017 objects are the frozen ones: a replaced lock, however self-consistent, is refused."""
    differing = [key for key in LOCK_DIGESTS if digests.get(key) != LOCK_DIGESTS[key]]
    if differing:
        raise PhaseError(f"the inherited 011/012/017 locks are not the frozen ones: {differing}")


def assert_record_inputs(record: Mapping[str, Any], digests: Mapping[str, str]) -> None:
    """The calibration record was computed from exactly these inputs: every 021 digest, the inherited locks included."""
    differing = [key for key in DIGEST_KEYS if record.get("inputs", {}).get(key) != digests.get(key)]
    if differing:
        raise PhaseError(f"the calibration record was computed from different inputs: {differing}")


def production_pools(pool: Any) -> CalibrationPools:
    pools = build_pools(pool)
    assert_pool_counts(pools)
    return pools


def y2_rows() -> tuple[tuple[int, int, int], ...]:
    """Every admissible valid-frame composition under the frozen Y2 precondition and at most 6 frames per template."""
    top = SLOTS["frame"]
    return tuple((a, b, c) for a in range(top + 1) for b in range(top + 1) for c in range(top + 1)
                 if c >= rd.MIN_VALID_COORDINATED_FRAMES and a + b + c >= rd.MIN_VALID_FRESH_FRAMES)


def y3_rows() -> tuple[tuple[int, int, int], ...]:
    """Every admissible scorable-noun composition under the frozen Y3 precondition and at most 8 nouns per rule class."""
    top = SLOTS["noun"]
    return tuple((a, b, c) for a in range(top + 1) for b in range(top + 1) for c in range(top + 1) if a + b + c >= rd.MIN_SCORABLE_FRESH_NOUNS)


def prefix_columns(composition: Sequence[int], slots: int) -> list[int]:
    """The first ``n_t`` slots of each stratum block: the row's columns in a base draw's slot layout."""
    return [block * slots + slot for block, count in enumerate(composition) for slot in range(int(count))]


# ---------------------------------------------------------------------------
# The draws: SHA-256 indexed, no random generator.


def slot_index(variant: str, draw: int, stratum: str, slot: int, n: int) -> int:
    digest = hashlib.sha256(f"021|{variant}|{draw}|{stratum}|{slot}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def draw_indices(variant: str, sizes: Mapping[str, int], slots: Mapping[str, int], draws: int) -> dict[str, torch.Tensor]:
    """``{stratum: LongTensor[draws, slots]}`` of indices into that stratum's ordered pool."""
    out = {}
    for stratum, n in sizes.items():
        if n < 1:
            raise PhaseError(f"stratum {stratum} is empty")
        width = slots[stratum]
        out[stratum] = torch.tensor([[slot_index(variant, b, stratum, i, n) for i in range(width)] for b in range(draws)], dtype=torch.int64)
    return out


# ---------------------------------------------------------------------------
# The re-materialized exposed table, the environment check and the reproduction gate.


@dataclass(frozen=True)
class ExposedTable:
    cues: tuple[str, ...]
    frames: tuple[str, ...]
    templates: tuple[str, ...]
    noun_keys: tuple[str, ...]
    measured: torch.Tensor  # [pairs, nouns] float64
    level0: torch.Tensor
    ceiling: torch.Tensor
    no_l5: torch.Tensor
    base: torch.Tensor
    dT: torch.Tensor  # [pairs]

    def scoring(self, column: str = "level0") -> rd.ScoringTable:
        return rd.ScoringTable(self.cues, self.frames, self.templates, self.measured, getattr(self, column), self.noun_keys)

    def digests(self) -> dict[str, str]:
        return {name: tensor_digest(getattr(self, name)) for name in ("measured", "level0", "ceiling", "no_l5", "base", "dT")}

    def tensors(self) -> dict[str, Any]:
        return {"cues": list(self.cues), "frames": list(self.frames), "templates": list(self.templates), "noun_keys": list(self.noun_keys),
                **{name: getattr(self, name) for name in ("measured", "level0", "ceiling", "no_l5", "base", "dT")}}


def _difference(a: Any, b: Any, path: str) -> tuple[float, str]:
    """The largest absolute difference between two JSON-like records and where it is; a structural mismatch is ∞."""
    if isinstance(a, bool) or isinstance(b, bool):
        return (0.0, path) if a == b else (math.inf, path)
    if a is None or b is None:
        return (0.0, path) if a is None and b is None else (math.inf, path)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        value = abs(float(a) - float(b))
        return (value if math.isfinite(value) else math.inf), path
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            return math.inf, f"{path} (keys differ: {sorted(set(a) ^ set(b))[:4]})"
        worst = (0.0, path)
        for key in sorted(a):
            candidate = _difference(a[key], b[key], f"{path}.{key}")
            if candidate[0] > worst[0]:
                worst = candidate
        return worst
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return math.inf, f"{path} (length {len(a)} against {len(b)})"
        worst = (0.0, path)
        for index, (x, y) in enumerate(zip(a, b)):
            candidate = _difference(x, y, f"{path}[{index}]")
            if candidate[0] > worst[0]:
                worst = candidate
        return worst
    return (0.0, path) if a == b else (math.inf, path)


def _bases_to_json(bases: Mapping[str, Mapping[int, torch.Tensor]]) -> dict[str, dict[str, list[float]]]:
    return {template: {str(layer): vector.tolist() for layer, vector in entry.items()} for template, entry in bases.items()}


def _assert_keys(keys: Sequence[str], ledger: frozenset[str], manifest: frozenset[str], what: str) -> None:
    foreign = [key for key in keys if key not in ledger]
    forbidden = [key for key in keys if key in manifest]
    if forbidden:
        raise pm.IncidentError(f"{what}: {len(forbidden)} confirmation-manifest keys would run, e.g. {forbidden[:2]}")
    if foreign:
        raise pm.IncidentError(f"{what}: {len(foreign)} keys outside Experiment 020's exposed ledger would run, e.g. {foreign[:2]}")


def rematerialize(model: Any, pool: Any, *, lock_011: Mapping[str, Any], lock_012: Mapping[str, Any], lock_017: Mapping[str, Any], exploration_020: Mapping[str, Any],
                  ledger_020: frozenset[str], manifest_keys: frozenset[str], confirmation: rd.Confirmation020, log: Callable[[str], None] | None = None,
                  executed: list[pm.Prompt] | None = None) -> tuple[ExposedTable, dict[str, Any]]:
    """Experiment 020's two exploration loops, with its own functions in its own order, on exactly its ledger keys.

    First the 108 reference captures (every key checked against 020's ledger and the manifest before the first
    forward), then the environment check (each re-captured state and the recomputed template bases against 020's
    records at ``LOCKED_STATE_TOLERANCE``), then frame by frame — that frame's keys checked first — the measurement,
    the Level 0 prediction with 020's recorded template bases, the identities, the conditional Level-1 breakdown, the
    inherited reproduction and the ceiling. Every executed prompt is appended to the caller's ``executed`` list the
    moment it ran, so the caller's ledger is complete even when a later step stops the phase. Nothing is enforced here
    but the key checks and the environment check: the caller records the identities, then enforces them.
    """
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)
    lw = lc.LayerWeights.from_model(model, layers=rd.PROGRAM_LAYERS)
    programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in rd.PROGRAM_LAYERS}
    chain = rd.chain_from_locks(lock_011, lock_012, lock_017, lw, programs, pool)
    program = rd.ReadoutProgram.from_model(model)
    axis_T = pm.SiteAxis("T", torch.zeros_like(torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64)), torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64), float(lock_011["sigma_T"]))
    nouns = rd.NounSet.build(weights, pool.nouns, [])
    rd.assert_explore_nouns(nouns, confirmation)  # 020's own computation-time freshness guard
    executed = executed if executed is not None else []
    reference_prompts = [pool.reference_prompt(frame) for frame in pool.frames]
    _assert_keys([prompt.key for prompt in reference_prompts], ledger_020, manifest_keys, "reference captures")
    states: dict[str, rd.FrameState020] = {}
    rows16: dict[str, Any] = {}
    for frame, prompt in zip(pool.frames, reference_prompts):
        states[frame.frame_id] = rd.capture_frame_020(model, head, prompt, nouns, axis_T)
        rows16[frame.frame_id] = atp.reference_rows(programs, states[frame.frame_id].state_017.x1_all, states[frame.frame_id].state_017.x2_all)
        executed.append(prompt)
    say(f"re-captured {len(states)} reference states, one forward each")
    drift = (0.0, "")
    for frame_id, state in states.items():
        candidate = _difference(rd.locked_state(state), exploration_020["locked_states"][frame_id], f"locked_states.{frame_id}")
        if candidate[0] > drift[0]:
            drift = candidate
    bases_now = _bases_to_json(rd.template_bases_020(states, pool.frames))
    base_drift = _difference(bases_now, exploration_020["template_bases"], "template_bases")
    environment = {"max_state_drift": drift[0] if math.isfinite(drift[0]) else None, "at": drift[1], "structural_state_mismatch": not math.isfinite(drift[0]),
                   "max_template_base_drift": base_drift[0] if math.isfinite(base_drift[0]) else None, "template_base_at": base_drift[1],
                   "structural_template_base_mismatch": not math.isfinite(base_drift[0]), "tolerance": LOCKED_STATE_TOLERANCE}
    if not (drift[0] <= LOCKED_STATE_TOLERANCE and base_drift[0] <= LOCKED_STATE_TOLERANCE):
        raise EnvironmentIncident(environment, executed)
    say(f"environment check: states {drift[0]:.1e}, template bases {base_drift[0]:.1e} (tolerance {LOCKED_STATE_TOLERANCE:.0e})")
    bases = {template: {int(layer): torch.tensor(vector, dtype=torch.float64) for layer, vector in entry.items()} for template, entry in exploration_020["template_bases"].items()}
    dT_direction = axis_T.direction.double()
    identities: dict[str, float] = {}
    worst_level1: dict[str, Any] | None = None
    scorable = list(nouns.exposed_scorable)
    cues, frame_ids, templates, measured_rows, level0_rows, ceiling_rows, no_l5_rows, base_rows, dT_values = [], [], [], [], [], [], [], [], []
    plural: dict[str, rd.PairMeasurement] = {}
    plural_ids = {template: pool.token_id(name) for template, name in pool.plural_cue.items()}
    for frame in pool.frames:
        frame_state = states[frame.frame_id]
        reference_id = pool.reference_ids[frame.template_id]
        tokens = [(word, token_id) for word, token_id in pool.tokens if token_id != reference_id]
        if rd.EXPLORE_TOKEN_LIMIT is not None:
            tokens = tokens[: rd.EXPLORE_TOKEN_LIMIT]
        prompts = [pm.Prompt(frame, token_id, word) for word, token_id in tokens]
        _assert_keys([prompt.key for prompt in prompts], ledger_020, manifest_keys, f"cue prompts of {frame.frame_id}")
        for (word, token_id), prompt in zip(tokens, prompts):
            measurement = rd.measure_pair(model, frame_state, nouns, word, token_id)
            executed.append(prompt)
            prediction = rd.predict_pair(program, chain, weights, frame_state, rows16[frame.frame_id], nouns, token_id, template_bases=bases, dT_direction=dT_direction)
            pair = rd.pair_identities(program, weights, nouns, frame_state, measurement)
            for name, value in pair.items():
                identities[name] = max(identities.get(name, 0.0), value)
            if worst_level1 is None or pair["level1"] > worst_level1["e1"]:
                worst_level1 = {**rd.level1_breakdown(program, frame_state, measurement), "pair_identities": {name: float(value) for name, value in pair.items()}}
            identities["inherited_017"] = max(identities.get("inherited_017", 0.0), rd.inherited_reproduction(chain, weights, frame_state, rows16[frame.frame_id], token_id, prediction["dx3"]))
            cues.append(word); frame_ids.append(frame.frame_id); templates.append(frame.template_id)
            measured_rows.append(measurement.dc[scorable]); level0_rows.append(prediction["dc"][scorable])
            ceiling_rows.append(rd.ceiling_prediction(program, frame_state, nouns, measurement.dx3)[scorable])
            no_l5_rows.append(prediction["dc_no_l5_heads"][scorable]); base_rows.append(prediction["dc_template_base"][scorable])
            dT_values.append(prediction["dT"])
            if token_id == plural_ids[frame.template_id]:
                plural[frame.frame_id] = measurement
        say(f"  {frame.frame_id}: {len(tokens)} exposed cues re-measured and predicted")
    noun_keys = tuple(nouns.nouns[index].lexical_key for index in scorable)
    table = ExposedTable(tuple(cues), tuple(frame_ids), tuple(templates), noun_keys, torch.stack(measured_rows), torch.stack(level0_rows), torch.stack(ceiling_rows),
                         torch.stack(no_l5_rows), torch.stack(base_rows), torch.tensor(dT_values, dtype=torch.float64))
    context = {"states": states, "plural": plural, "program": program, "nouns": nouns, "axis_T": axis_T, "executed": executed, "environment": environment,
               "identities": {name: float(value) for name, value in identities.items()}, "level1_worst": worst_level1}
    return table, context


class EnvironmentIncident(pm.IncidentError):
    """A re-captured reference state or template base differs from Experiment 020's beyond the frozen tolerance."""

    def __init__(self, environment: Mapping[str, Any], executed: Sequence[pm.Prompt]):
        self.environment = dict(environment)
        self.executed = list(executed)
        super().__init__(f"environment check failed: state drift {environment['max_state_drift']} at {environment['at']}, template-base drift "
                         f"{environment['max_template_base_drift']} at {environment['template_base_at']} (None: a structural mismatch), above the frozen tolerance {LOCKED_STATE_TOLERANCE:.0e}")


def reproduction_gate(table: ExposedTable, exploration_020: Mapping[str, Any]) -> dict[str, Any]:
    """020's own statistics recomputed from the re-materialized table, every numeric leaf against 020's record."""
    scoring = table.scoring("level0")
    recomputed = {
        "statistics": rd.pair_statistics(scoring),
        "nouns": rd.noun_statistics(scoring),
        "comparators": {"ceiling_measured_dx3": rd.pair_statistics(table.scoring("ceiling"))["flattened_r2"],
                        "no_l5_heads": rd.pair_statistics(table.scoring("no_l5"))["flattened_r2"],
                        "template_base_mlps": rd.pair_statistics(table.scoring("base"))["flattened_r2"],
                        "dT_only": rd.dT_only_fit(table.dT, scoring)},
    }
    recorded = {"statistics": exploration_020["statistics"], "nouns": exploration_020["nouns"],
                "comparators": {key: value for key, value in exploration_020["comparators"].items() if key != "standing"}}
    sections = {}
    worst = (0.0, "")
    for name in recomputed:
        difference = _difference(json.loads(pm.canonical_json(recomputed[name])), recorded[name], name)
        sections[name] = {"max_difference": difference[0] if math.isfinite(difference[0]) else None, "at": difference[1]}
        if difference[0] > worst[0]:
            worst = difference
    return {"tolerance": RECONSTRUCTION_TOLERANCE, "max_difference": worst[0] if math.isfinite(worst[0]) else None, "at": worst[1], "sections": sections,
            "n_pairs": int(table.measured.shape[0]), "n_nouns": int(table.measured.shape[1]), "passed": bool(math.isfinite(worst[0]) and worst[0] <= RECONSTRUCTION_TOLERANCE)}


def enforce_gate(gate: Mapping[str, Any]) -> None:
    if not gate["passed"]:
        raise pm.IncidentError(f"reproduction gate failed: {gate['max_difference']} at {gate['at']} above the frozen tolerance {RECONSTRUCTION_TOLERANCE:.0e}; nothing is drawn")


def validity_screen(program: rd.ReadoutProgram, states: Mapping[str, rd.FrameState020], nouns: rd.NounSet, plural: Mapping[str, rd.PairMeasurement], axis_T: pm.SiteAxis) -> dict[str, dict[str, Any]]:
    """The frozen stage-1 validity rule on every exposed frame, from its reference state and its plural-cue measurement."""
    missing = sorted(set(states) - set(plural))
    if missing:
        raise pm.IncidentError(f"no plural-cue measurement for {missing[:3]}; the validity screen has no input")
    return {frame_id: rd.frame_validity(program, states[frame_id], nouns, plural[frame_id], axis_T) for frame_id in sorted(states)}


def screen_precondition(pools: CalibrationPools) -> dict[str, Any]:
    counts = {template: len(ids) for template, ids in pools.frames().items()}
    return {"minimum": MIN_SCREENED_FRAMES_PER_TEMPLATE, "counts": counts, "ok": all(count >= MIN_SCREENED_FRAMES_PER_TEMPLATE for count in counts.values())}


# ---------------------------------------------------------------------------
# The statistics kernel: per-group moments, vectorized over draws. One implementation for the draws and for confirm.
# Every sum of squared deviations is a two-pass sum inside a group, combined across groups with the pooled-variance
# identity Σ M2ᵢ + Σ nᵢ (x̄ᵢ − x̄)² — never the one-pass Σy² − (Σy)²/n, whose cancellation is the only realistic way
# the kernel could drift from Experiment 020's own two-pass ``_r2``.


def _nan_where(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, torch.full_like(values, math.nan), values)


def _r2_rows(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``R²`` along the last dimension (020's ``_r2`` per row); NaN where the measured variance is zero."""
    sst = ((y - y.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
    sse = ((y - x) ** 2).sum(dim=-1)
    return _r2_moments(sse, sst)


def _r2_moments(sse: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    return _nan_where(1.0 - sse / torch.where(m2 > 0, m2, torch.ones_like(m2)), ~(m2 > 0))


def _combine(counts: torch.Tensor, means: torch.Tensor, m2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pooled count, mean and centered sum of squares of groups along the last dimension."""
    total = counts.sum(dim=-1)
    grand = (counts * means).sum(dim=-1) / total
    return total, grand, m2.sum(dim=-1) + (counts * (means - grand.unsqueeze(-1)) ** 2).sum(dim=-1)


def _kth(values: torch.Tensor, k: int, *, largest: bool) -> torch.Tensor:
    """The k-th largest (or smallest) along the last dimension, undefined values ranked worst."""
    filled = torch.where(torch.isnan(values), torch.full_like(values, -math.inf if largest else math.inf), values)
    return filled.sort(dim=-1, descending=largest).values[..., k - 1]


def _median(values: torch.Tensor) -> torch.Tensor:
    """020's median (the mean of the two middle values for an even count), undefined values ranked worst."""
    filled = torch.where(torch.isnan(values), torch.full_like(values, -math.inf), values).sort(dim=-1).values
    n = values.shape[-1]
    if n % 2:
        return filled[..., n // 2]
    return 0.5 * (filled[..., n // 2 - 1] + filled[..., n // 2])


@dataclass(frozen=True)
class CueSums:
    """Per cue, over its pairs (every frame of the table) and the table's noun columns."""

    cues: tuple[str, ...]
    n_nouns: int
    n_pairs: torch.Tensor
    token_y: torch.Tensor
    token_x: torch.Tensor
    mae: torch.Tensor
    ae_sum: torch.Tensor
    pm_mean: torch.Tensor  # the cue's pair noun-means: their mean,
    pm_m2: torch.Tensor  # their centered sum of squares,
    pm_sse: torch.Tensor  # and Σ(ȳ − x̄)² against the predicted pair means
    flat: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]  # column -> (mean y, centered Σ y over pairs × nouns, Σ(y − x)²)


def cue_sums(table: rd.ScoringTable, cues: Sequence[str], *, columns: Mapping[str, torch.Tensor] | None = None) -> CueSums:
    groups = table.by("cue")
    y_all, x_all = table.measured.double(), table.predicted.double()
    names = ("n_pairs", "token_y", "token_x", "mae", "ae_sum", "pm_mean", "pm_m2", "pm_sse")
    fields: dict[str, list[float]] = {name: [] for name in names}
    flat: dict[str, list[list[float]]] = {name: [[], [], []] for name in (columns or {})}
    for cue in cues:
        if cue not in groups:
            raise PhaseError(f"the table holds no pair of the cue {cue}")
        rows = groups[cue]
        y, x = y_all[rows], x_all[rows]
        error = y - x
        ybar, xbar = y.mean(dim=1), x.mean(dim=1)
        for name, value in (("n_pairs", len(rows)), ("token_y", float(y.mean())), ("token_x", float(x.mean())), ("mae", float(error.abs().mean())), ("ae_sum", float(error.abs().sum())),
                            ("pm_mean", float(ybar.mean())), ("pm_m2", float(((ybar - ybar.mean()) ** 2).sum())), ("pm_sse", float(((ybar - xbar) ** 2).sum()))):
            fields[name].append(float(value))
        for name, predicted in (columns or {}).items():
            z = predicted.double()[rows]
            flat[name][0].append(float(y.mean())); flat[name][1].append(float(((y - y.mean()) ** 2).sum())); flat[name][2].append(float(((y - z) ** 2).sum()))
    tensor = lambda values: torch.tensor(values, dtype=torch.float64)  # noqa: E731
    return CueSums(tuple(cues), int(y_all.shape[1]), *(tensor(fields[name]) for name in names), {name: tuple(tensor(part) for part in parts) for name, parts in flat.items()})


def y1_statistics(sums: CueSums, cue_index: torch.Tensor) -> dict[str, torch.Tensor]:
    """S1–S4 for every row of ``cue_index`` ([draws, cues] of positions in ``sums.cues``; duplicates are units)."""
    n = int(cue_index.shape[1])
    counts = sums.n_pairs[cue_index]
    _, _, pm_m2 = _combine(counts, sums.pm_mean[cue_index], sums.pm_m2[cue_index])
    out = {"token_mean_r2": _r2_rows(sums.token_y[cue_index], sums.token_x[cue_index]),
           "pair_mean_r2": _r2_moments(sums.pm_sse[cue_index].sum(dim=1), pm_m2),
           "cue_mae_k80": _kth(sums.mae[cue_index], k_of(SHARES["cue"], n), largest=False),
           "pooled_mae": sums.ae_sum[cue_index].sum(dim=1) / (counts.sum(dim=1) * sums.n_nouns)}
    for name, (mean_y, m2_y, se2) in sums.flat.items():
        _, _, m2 = _combine(counts * sums.n_nouns, mean_y[cue_index], m2_y[cue_index])
        out[f"flattened_r2:{name}"] = _r2_moments(se2[cue_index].sum(dim=1), m2)
    return out


@dataclass(frozen=True)
class GridSums:
    """Per (cue, frame) pair, over the table's noun columns."""

    cues: tuple[str, ...]
    frames: tuple[str, ...]
    templates: tuple[str, ...]  # per frame
    n_nouns: int
    mean_y: torch.Tensor  # [cues, frames]
    m2_y: torch.Tensor
    mean_x: torch.Tensor
    se2: torch.Tensor
    flat_se2: Mapping[str, torch.Tensor]


def grid_sums(table: rd.ScoringTable, cues: Sequence[str], frames: Sequence[str], *, columns: Mapping[str, torch.Tensor] | None = None) -> GridSums:
    index = {(cue, frame): row for row, (cue, frame) in enumerate(zip(table.cues, table.frames))}
    template_of = {frame: template for frame, template in zip(table.frames, table.templates)}
    try:
        rows = torch.tensor([[index[(cue, frame)] for frame in frames] for cue in cues], dtype=torch.int64)
    except KeyError as error:
        raise PhaseError(f"the table holds no pair {error.args[0]}") from None
    y, x = table.measured.double()[rows], table.predicted.double()[rows]  # [cues, frames, nouns]
    mean_y = y.mean(dim=-1)
    flat = {name: ((y - predicted.double()[rows]) ** 2).sum(dim=-1) for name, predicted in (columns or {}).items()}
    return GridSums(tuple(cues), tuple(frames), tuple(template_of[frame] for frame in frames), int(y.shape[-1]), mean_y, ((y - mean_y.unsqueeze(-1)) ** 2).sum(dim=-1),
                    x.mean(dim=-1), ((y - x) ** 2).sum(dim=-1), flat)


def y2_slot_sums(grid: GridSums, cue_index: torch.Tensor, frame_index: torch.Tensor) -> dict[str, Any]:
    """Per draw and frame slot, pooled over the draw's cues ([draws, frame slots])."""
    gather = lambda values: values[cue_index[:, :, None], frame_index[:, None, :]].transpose(1, 2)  # noqa: E731  [draws, frame slots, cues]
    mean_y = gather(grid.mean_y)
    count, pooled_mean, pooled_m2 = _combine(torch.full_like(mean_y, float(grid.n_nouns)), mean_y, gather(grid.m2_y))
    out: dict[str, Any] = {"count": count, "mean_y": pooled_mean, "m2": pooled_m2, "mean_x": gather(grid.mean_x).mean(dim=-1), "se2": gather(grid.se2).sum(dim=-1)}
    for name, values in grid.flat_se2.items():
        out[f"se2:{name}"] = gather(values).sum(dim=-1)
    return out


def y2_statistics(slots: Mapping[str, Any], columns: Sequence[int], column_templates: Sequence[str]) -> dict[str, torch.Tensor]:
    """S5–S8 on the selected frame slots; ``column_templates`` names each selected slot's template."""
    cols = torch.tensor(list(columns), dtype=torch.int64)
    count, mean_y, m2, mean_x, se2 = (slots[key][:, cols] for key in ("count", "mean_y", "m2", "mean_x", "se2"))
    n = len(columns)
    out = {"frame_mean_r2": _r2_rows(mean_y, mean_x), "frame_r2_k75": _kth(_r2_moments(se2, m2), k_of(SHARES["frame"], n), largest=True)}
    for name, keep in (("cue_final_r2", lambda template: template != pm.COORDINATED_TEMPLATE), ("coordinated_r2", lambda template: template == pm.COORDINATED_TEMPLATE)):
        subset = [position for position, template in enumerate(column_templates) if keep(template)]
        if subset:
            part = torch.tensor(subset, dtype=torch.int64)
            _, _, pooled = _combine(count[:, part], mean_y[:, part], m2[:, part])
            out[name] = _r2_moments(se2[:, part].sum(dim=1), pooled)
        else:
            out[name] = torch.full((mean_y.shape[0],), math.nan, dtype=torch.float64)
    _, _, pooled_all = _combine(count, mean_y, m2)
    for key in [key for key in slots if key.startswith("se2:")]:
        out[f"flattened_r2:{key[4:]}"] = _r2_moments(slots[key][:, cols].sum(dim=1), pooled_all)
    return out


@dataclass(frozen=True)
class NounCueSums:
    """Per (noun, cue), over the cue's pairs (every frame of the table)."""

    nouns: tuple[str, ...]
    cues: tuple[str, ...]
    count: torch.Tensor  # [nouns, cues]
    mean_y: torch.Tensor
    m2_y: torch.Tensor
    se2: torch.Tensor
    sxy: torch.Tensor
    syy: torch.Tensor
    sd: torch.Tensor  # Σ(x − y)


def noun_cue_sums(table: rd.ScoringTable, cues: Sequence[str], nouns: Sequence[str]) -> NounCueSums:
    groups = table.by("cue")
    position = {key: index for index, key in enumerate(table.noun_keys)}
    columns = torch.tensor([position[key] for key in nouns], dtype=torch.int64)
    names = ("count", "mean_y", "m2_y", "se2", "sxy", "syy", "sd")
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for cue in cues:
        if cue not in groups:
            raise PhaseError(f"the table holds no pair of the cue {cue}")
        rows = groups[cue]
        y, x = table.measured.double()[rows][:, columns], table.predicted.double()[rows][:, columns]
        mean_y = y.mean(dim=0)
        for name, value in (("count", torch.full((len(nouns),), float(len(rows)), dtype=torch.float64)), ("mean_y", mean_y), ("m2_y", ((y - mean_y) ** 2).sum(dim=0)),
                            ("se2", ((y - x) ** 2).sum(dim=0)), ("sxy", (x * y).sum(dim=0)), ("syy", (y * y).sum(dim=0)), ("sd", (x - y).sum(dim=0))):
            parts[name].append(value)
    return NounCueSums(tuple(nouns), tuple(cues), *(torch.stack(parts[name], dim=1) for name in names))


def y3_slot_statistics(sums: NounCueSums, noun_index: torch.Tensor, cue_index: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per draw and noun slot, pooled over the draw's cues: ``R²``, slope and bias ([draws, noun slots])."""
    gather = lambda values: values[noun_index[:, :, None], cue_index[:, None, :]]  # noqa: E731  [draws, noun slots, cues]
    count, _, m2 = _combine(gather(sums.count), gather(sums.mean_y), gather(sums.m2_y))
    syy = gather(sums.syy).sum(dim=-1)
    slope = _nan_where(gather(sums.sxy).sum(dim=-1) / torch.where(syy > 0, syy, torch.ones_like(syy)), ~(syy > 0))
    return {"r2": _r2_moments(gather(sums.se2).sum(dim=-1), m2), "slope": slope, "bias": gather(sums.sd).sum(dim=-1) / count}


def y3_statistics(slot_statistics: Mapping[str, torch.Tensor], columns: Sequence[int]) -> dict[str, torch.Tensor]:
    """S9–S12 on the selected noun slots."""
    cols = torch.tensor(list(columns), dtype=torch.int64)
    r2, slope, bias = slot_statistics["r2"][:, cols], slot_statistics["slope"][:, cols], slot_statistics["bias"][:, cols]
    k = k_of(SHARES["noun"], len(columns))
    return {"noun_median_r2": _median(r2), "noun_r2_k90": _kth(r2, k, largest=True), "noun_slope_dev_k90": _kth((slope - 1.0).abs(), k, largest=False),
            "noun_bias_k90": _kth(bias.abs(), k, largest=False)}


def as_values(tensor: torch.Tensor) -> list[float | None]:
    """Per-draw values for the record: an undefined (NaN or infinite) value is ``None``."""
    return [float(value) if math.isfinite(value) else None for value in tensor.tolist()]


# ---------------------------------------------------------------------------
# The direct cross-check: 020's own statistics on a materialized table, duplicates relabelled ``unit#slot``.


def _kth_list(values: Sequence[float | None], k: int, *, largest: bool) -> float | None:
    worst = -math.inf if largest else math.inf
    ordered = sorted((worst if value is None else float(value) for value in values), reverse=largest)
    value = ordered[k - 1]
    return value if math.isfinite(value) else None


def _median_list(values: Sequence[float | None]) -> float | None:
    ordered = sorted(-math.inf if value is None else float(value) for value in values)
    n = len(ordered)
    value = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    return value if math.isfinite(value) else None


def direct_y1(table: rd.ScoringTable, cue_units: Sequence[str]) -> dict[str, float | None]:
    groups = table.by("cue")
    rows, labels = [], []
    for slot, cue in enumerate(cue_units):
        for row in groups[cue]:
            rows.append(row); labels.append(f"{cue}#{slot}")
    materialized = rd.ScoringTable(tuple(labels), tuple(table.frames[r] for r in rows), tuple(table.templates[r] for r in rows), table.measured[rows], table.predicted[rows], table.noun_keys)
    statistics = rd.pair_statistics(materialized)
    maes = [entry["mae"] for entry in statistics["per_cue"].values()]
    return {"token_mean_r2": statistics["token_mean_r2"], "pair_mean_r2": statistics["pair_mean_r2"], "cue_mae_k80": _kth_list(maes, k_of(SHARES["cue"], len(maes)), largest=False),
            "pooled_mae": statistics["pooled_mae"]}


def direct_y2(table: rd.ScoringTable, cue_units: Sequence[str], frame_units: Sequence[str]) -> dict[str, float | None]:
    index = {(cue, frame): row for row, (cue, frame) in enumerate(zip(table.cues, table.frames))}
    rows, cue_labels, frame_labels = [], [], []
    for j, cue in enumerate(cue_units):
        for i, frame in enumerate(frame_units):
            rows.append(index[(cue, frame)]); cue_labels.append(f"{cue}#{j}"); frame_labels.append(f"{frame}#{i}")
    materialized = rd.ScoringTable(tuple(cue_labels), tuple(frame_labels), tuple(table.templates[r] for r in rows), table.measured[rows], table.predicted[rows], table.noun_keys)
    statistics = rd.pair_statistics(materialized)
    r2s = [entry["r2"] for entry in statistics["per_frame"].values()]
    return {"frame_mean_r2": statistics["frame_mean_r2"], "frame_r2_k75": _kth_list(r2s, k_of(SHARES["frame"], len(r2s)), largest=True),
            "cue_final_r2": statistics["cue_final_r2"], "coordinated_r2": statistics["coordinated_r2"]}


def direct_y3(table: rd.ScoringTable, cue_units: Sequence[str], noun_units: Sequence[str]) -> dict[str, float | None]:
    groups = table.by("cue")
    position = {key: index for index, key in enumerate(table.noun_keys)}
    rows, labels = [], []
    for slot, cue in enumerate(cue_units):
        for row in groups[cue]:
            rows.append(row); labels.append(f"{cue}#{slot}")
    columns = [position[key] for key in noun_units]
    materialized = rd.ScoringTable(tuple(labels), tuple(table.frames[r] for r in rows), tuple(table.templates[r] for r in rows), table.measured[rows][:, columns],
                                   table.predicted[rows][:, columns], tuple(f"{key}#{i}" for i, key in enumerate(noun_units)))
    entries = list(rd.noun_statistics(materialized).values())
    k = k_of(SHARES["noun"], len(entries))
    return {"noun_median_r2": _median_list([entry["r2"] for entry in entries]), "noun_r2_k90": _kth_list([entry["r2"] for entry in entries], k, largest=True),
            "noun_slope_dev_k90": _kth_list([None if entry["slope"] is None else abs(entry["slope"] - 1.0) for entry in entries], k, largest=False),
            "noun_bias_k90": _kth_list([abs(entry["bias"]) for entry in entries], k, largest=False)}


def agreement(kernel: Mapping[str, float | None], direct: Mapping[str, float | None]) -> tuple[float, str]:
    """The largest kernel/direct difference over the statistics and its name, as |kernel − direct| / max(1, |direct|):
    absolute for well-conditioned values, relative where a statistic is itself ill-conditioned (an ``R²`` far below
    zero amplifies float reassociation by its own magnitude). One side undefined is ∞."""
    worst_value, worst_name = -1.0, ""
    for name, value in direct.items():
        other = kernel.get(name)
        if value is None or other is None:
            difference = 0.0 if value is None and other is None else math.inf
        else:
            difference = abs(float(value) - float(other)) / max(1.0, abs(float(value)))
        if difference > worst_value:
            worst_value, worst_name = difference, name
    return max(worst_value, 0.0), worst_name


# ---------------------------------------------------------------------------
# The floor rule and the one pass predicate.


def passes(kind: str, value: float | None, floor: float | None) -> bool:
    """The only pass/fail function of Experiment 021 (revision 3).

    ``R²``-type: finite, strictly positive, and at least the floor. Error-type: finite and at most the floor (a
    ``None`` error ceiling is +∞, recorded as uninformative). The calibration's pass rates and ``confirm`` both call it.
    """
    if value is None or isinstance(value, bool):
        return False
    value = float(value)
    if not math.isfinite(value):
        return False
    if kind == "r2":
        if floor is None or not math.isfinite(float(floor)):
            raise ValueError("an R²-type floor is always a finite number")
        return value > 0.0 and value >= float(floor)
    if kind == "error":
        return floor is None or value <= float(floor)
    raise ValueError(f"unknown statistic kind {kind}")


def tail_floor(values: Sequence[float | None], kind: str) -> dict[str, Any]:
    """The 250th smallest (≥, ``R²``) or 250th largest (≤, error) of the draws, undefined values ranked worst; the
    zero-skill clamp for ``R²``; full float64 precision."""
    k = tail_index(len(values))
    if kind == "r2":
        raw = sorted(-math.inf if value is None else float(value) for value in values)[k - 1]
        return {"floor": max(raw, 0.0), "raw": raw if math.isfinite(raw) else None, "clamped": bool(raw < 0.0)}
    if kind == "error":
        raw = sorted((math.inf if value is None else float(value) for value in values), reverse=True)[k - 1]
        return {"floor": raw if math.isfinite(raw) else None, "raw": raw if math.isfinite(raw) else None, "clamped": False}
    raise ValueError(f"unknown statistic kind {kind}")


def pass_rates(values: Mapping[str, Sequence[float | None]], floors: Mapping[str, float | None]) -> dict[str, Any]:
    """Per condition and joint, over the draws, through ``passes``."""
    names = list(values)
    draws = len(values[names[0]])
    passed = {name: [passes(KIND[name], value, floors[name]) for value in values[name]] for name in names}
    joint = [all(passed[name][b] for name in names) for b in range(draws)]
    return {"conditions": {name: sum(flags) / draws for name, flags in passed.items()}, "joint": sum(joint) / draws, "per_draw": joint}


def summary(values: Sequence[float | None], kind: str) -> dict[str, float | None]:
    """Descriptive: the median and the 250th smallest and largest values (undefined ranked worst)."""
    worst = -math.inf if kind == "r2" else math.inf
    ordered = sorted(worst if value is None else float(value) for value in values)
    k = tail_index(len(values))
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    finite = lambda value: value if math.isfinite(value) else None  # noqa: E731
    return {"median": finite(median), "lower": finite(ordered[k - 1]), "upper": finite(ordered[n - k])}


# ---------------------------------------------------------------------------
# The calibration: draws, rows, the cross-check, floors and descriptives.


@dataclass(frozen=True)
class KernelInputs:
    """The kernel's sums over the pools' units, built once from the re-materialized exposed table."""

    cue_units: tuple[str, ...]
    frame_units: tuple[str, ...]
    noun_units: tuple[str, ...]
    cue: CueSums
    grid: GridSums
    noun: NounCueSums
    level0: rd.ScoringTable


def kernel_inputs(table: ExposedTable, pools: CalibrationPools) -> KernelInputs:
    cue_units = tuple(word for cls in CUE_CLASSES for word, _ in pools.cues[cls])
    frame_units = tuple(frame_id for template in Y2_AXES for frame_id in pools.frames_all_unscreened[template])
    noun_units = tuple(key for rule in RULE_CLASSES for key in pools.nouns[rule])
    level0 = table.scoring("level0")
    columns = {"level0": table.level0, "ceiling": table.ceiling}
    return KernelInputs(cue_units, frame_units, noun_units, cue_sums(level0, cue_units, columns=columns), grid_sums(level0, cue_units, frame_units, columns=columns),
                        noun_cue_sums(level0, cue_units, noun_units), level0)


def variant_strata(pools: CalibrationPools, inputs: KernelInputs, variant: str) -> dict[str, list[int]]:
    """``{stratum: [positions in the kernel's unit lists]}`` in the frozen stratum order."""
    cue_position = {cue: index for index, cue in enumerate(inputs.cue_units)}
    frame_position = {frame: index for index, frame in enumerate(inputs.frame_units)}
    noun_position = {key: index for index, key in enumerate(inputs.noun_units)}
    cues = pools.lineage_cues if variant == "lineage" else pools.cues
    frames = pools.frames(all_frames=(variant == "all-frames"))
    strata: dict[str, list[int]] = {}
    strata.update({f"cue/{cls}": [cue_position[word] for word, _ in cues[cls]] for cls in CUE_CLASSES})
    strata.update({f"frame/{template}": [frame_position[frame] for frame in frames[template]] for template in Y2_AXES})
    strata.update({f"noun/{rule}": [noun_position[key] for key in pools.nouns[rule]] for rule in RULE_CLASSES})
    return strata


def base_draws(strata: Mapping[str, Sequence[int]], variant: str, draws: int) -> dict[str, Any]:
    """The base draws of a variant: indices into each stratum, mapped to kernel positions in slot layout."""
    slots = {stratum: SLOTS[stratum.split("/")[0]] for stratum in strata}
    local = draw_indices(variant, {stratum: len(members) for stratum, members in strata.items()}, slots, draws)
    mapped = {stratum: torch.tensor(members, dtype=torch.int64)[local[stratum]] for stratum, members in strata.items()}
    return {"local": local, "digests": {stratum: tensor_digest(indices) for stratum, indices in local.items()},
            "cue": torch.cat([mapped[f"cue/{cls}"] for cls in CUE_CLASSES], dim=1),
            "frame": torch.cat([mapped[f"frame/{template}"] for template in Y2_AXES], dim=1),
            "noun": torch.cat([mapped[f"noun/{rule}"] for rule in RULE_CLASSES], dim=1)}


def _chunks(draws: int, size: int = 2000):
    for start in range(0, draws, size):
        yield slice(start, min(draws, start + size))


def row_statistics(inputs: KernelInputs, draws: Mapping[str, Any], rows: Mapping[str, Sequence[tuple[int, ...]]]) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Every requested row's statistics over every base draw: ``{outcome: {row key: {statistic: [draws]}}}``."""
    total = int(draws["cue"].shape[0])
    out: dict[str, dict[str, dict[str, torch.Tensor]]] = {"Y1": {}, "Y2": {}, "Y3": {}}
    parts: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = {"Y1": {}, "Y2": {}, "Y3": {}}
    frame_templates = [template for template in Y2_AXES for _ in range(SLOTS["frame"])]
    for part in _chunks(total):
        cue_index, frame_index, noun_index = draws["cue"][part], draws["frame"][part], draws["noun"][part]
        if rows.get("Y1") is not None:
            for key, values in y1_statistics(inputs.cue, cue_index).items():
                parts["Y1"].setdefault("all", {}).setdefault(key, []).append(values)
        if rows.get("Y2"):
            slots = y2_slot_sums(inputs.grid, cue_index, frame_index)
            for composition in rows["Y2"]:
                columns = prefix_columns(composition, SLOTS["frame"])
                for key, values in y2_statistics(slots, columns, [frame_templates[c] for c in columns]).items():
                    parts["Y2"].setdefault(row_key(composition), {}).setdefault(key, []).append(values)
        if rows.get("Y3"):
            slot_statistics = y3_slot_statistics(inputs.noun, noun_index, cue_index)
            for composition in rows["Y3"]:
                for key, values in y3_statistics(slot_statistics, prefix_columns(composition, SLOTS["noun"])).items():
                    parts["Y3"].setdefault(row_key(composition), {}).setdefault(key, []).append(values)
    for outcome, entries in parts.items():
        for key, statistics in entries.items():
            out[outcome][key] = {name: torch.cat(chunks) for name, chunks in statistics.items()}
    return out


def draw_units(inputs: KernelInputs, draws: Mapping[str, Any], b: int, composition: Sequence[int], outcome: str) -> dict[str, list[str]]:
    cues = [inputs.cue_units[int(i)] for i in draws["cue"][b]]
    if outcome == "Y2":
        frames = [inputs.frame_units[int(draws["frame"][b][c])] for c in prefix_columns(composition, SLOTS["frame"])]
        return {"cues": cues, "frames": frames}
    if outcome == "Y3":
        nouns = [inputs.noun_units[int(draws["noun"][b][c])] for c in prefix_columns(composition, SLOTS["noun"])]
        return {"cues": cues, "nouns": nouns}
    return {"cues": cues}


def cross_check(inputs: KernelInputs, draws: Mapping[str, Any], statistics: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]], rows: Mapping[str, Sequence[tuple[int, ...]]],
                *, n_draws: int | None = None) -> dict[str, Any]:
    """The kernel against 020's direct statistics on the first ``CROSS_CHECK_DRAWS`` base draws of every row.

    It compares and records; it never modifies, selects, rounds or replaces a value. Above the tolerance it raises
    ``CrossCheckError`` with the maximum difference and the row, draw and statistic that carry it.
    """
    count = min(CROSS_CHECK_DRAWS if n_draws is None else n_draws, int(draws["cue"].shape[0]))
    worst: dict[str, Any] = {"max_difference": -1.0}
    first: dict[str, Any] | None = None
    checked = exceeding = 0
    plan = [("Y1", ())] + [("Y2", composition) for composition in rows.get("Y2", ())] + [("Y3", composition) for composition in rows.get("Y3", ())]
    for outcome, composition in plan:
        key = row_key(composition)
        for b in range(count):
            units = draw_units(inputs, draws, b, composition, outcome)
            if outcome == "Y1":
                direct = direct_y1(inputs.level0, units["cues"])
            elif outcome == "Y2":
                direct = direct_y2(inputs.level0, units["cues"], units["frames"])
            else:
                direct = direct_y3(inputs.level0, units["cues"], units["nouns"])
            kernel = {name: as_values(statistics[outcome][key][name][b : b + 1])[0] for name in STATISTICS[outcome]}
            difference, name = agreement(kernel, direct)
            checked += 1
            location = {"outcome": outcome, "row": key, "draw": b, "statistic": name, "kernel": kernel.get(name), "direct": direct.get(name)}
            if difference > worst["max_difference"]:
                worst = {"max_difference": difference, **location}
            if not difference <= STATISTIC_AGREEMENT_TOLERANCE:
                exceeding += 1
                first = first or {"max_difference": difference, **location}
    result = json_safe({**worst, "tolerance": STATISTIC_AGREEMENT_TOLERANCE, "scale": AGREEMENT_SCALE, "draws_per_row": count, "n_checked": checked,
                        "n_exceeding": exceeding, "first_exceeding": first})
    if exceeding:
        raise CrossCheckError(result)  # the worst location, recorded; nothing after this point runs, no floor exists
    return result


def _row_entry(outcome: str, composition: Sequence[int], statistics: Mapping[str, torch.Tensor]) -> tuple[dict[str, Any], dict[str, list[float | None]], list[bool]]:
    names = STATISTICS[outcome]
    values = {name: as_values(statistics[name]) for name in names}
    floors, raw, clamped, described = {}, {}, {}, {}
    for name in names:
        rule = tail_floor(values[name], KIND[name])
        floors[name], raw[name], clamped[name] = rule["floor"], rule["raw"], rule["clamped"]
        described[name] = summary(values[name], KIND[name])
    rates = pass_rates(values, floors)
    n = len(CUE_CLASSES) * SLOTS["cue"] if outcome == "Y1" else sum(composition)
    k = k_of(SHARES[{"Y1": "cue", "Y2": "frame", "Y3": "noun"}[outcome]], n)
    entry = {"key": row_key(composition), "composition": list(composition), "n": n, "k": k, "floors": floors, "raw_tail": raw, "clamped": clamped, "summary": described,
             "pass_rates": rates["conditions"], "joint_pass_rate": rates["joint"],
             "draw_values_sha256": tensor_digest(torch.stack([statistics[name] for name in names], dim=1))}
    flats = {name[len("flattened_r2:"):]: summary(as_values(tensor), "r2") for name, tensor in statistics.items() if name.startswith("flattened_r2:")}
    if flats:
        entry["flattened_r2"] = flats
    return entry, values, rates["per_draw"]


def full_row_tables(inputs: KernelInputs, draws: Mapping[str, Any], b: int) -> dict[str, rd.ScoringTable]:
    """The full rows' materialized tables of base draw ``b`` (duplicates relabelled), for 020's own scoring."""
    units_y1 = draw_units(inputs, draws, b, (), "Y1")["cues"]
    units_y2 = draw_units(inputs, draws, b, FULL_ROWS["Y2"], "Y2")
    units_y3 = draw_units(inputs, draws, b, FULL_ROWS["Y3"], "Y3")
    table = inputs.level0
    groups = table.by("cue")
    index = {(cue, frame): row for row, (cue, frame) in enumerate(zip(table.cues, table.frames))}
    position = {key: i for i, key in enumerate(table.noun_keys)}

    def by_cues(cues: Sequence[str], columns: Sequence[int] | None, keys: Sequence[str]) -> rd.ScoringTable:
        rows, labels = [], []
        for slot, cue in enumerate(cues):
            for row in groups[cue]:
                rows.append(row); labels.append(f"{cue}#{slot}")
        measured, predicted = table.measured[rows], table.predicted[rows]
        if columns is not None:
            measured, predicted = measured[:, list(columns)], predicted[:, list(columns)]
        return rd.ScoringTable(tuple(labels), tuple(table.frames[r] for r in rows), tuple(table.templates[r] for r in rows), measured, predicted, tuple(keys))

    rows, cue_labels, frame_labels = [], [], []
    for j, cue in enumerate(units_y2["cues"]):
        for i, frame in enumerate(units_y2["frames"]):
            rows.append(index[(cue, frame)]); cue_labels.append(f"{cue}#{j}"); frame_labels.append(f"{frame}#{i}")
    y2 = rd.ScoringTable(tuple(cue_labels), tuple(frame_labels), tuple(table.templates[r] for r in rows), table.measured[rows], table.predicted[rows], table.noun_keys)
    template_of = dict(zip(inputs.grid.frames, inputs.grid.templates))
    return {"Y1": by_cues(units_y1, None, table.noun_keys), "Y2": y2,
            "Y3": by_cues(units_y3["cues"], [position[key] for key in units_y3["nouns"]], [f"{key}#{i}" for i, key in enumerate(units_y3["nouns"])]),
            "Y2_frames": [f"{frame}#{i}" for i, frame in enumerate(units_y2["frames"])],
            "Y2_coordinated": sum(1 for frame in units_y2["frames"] if template_of[frame] == pm.COORDINATED_TEMPLATE)}


def pass_rates_020(inputs: KernelInputs, draws: Mapping[str, Any]) -> dict[str, Any]:
    """Descriptive: how often Experiment 020's floors, in 020's own condition forms and code, pass on the full rows."""
    total = int(draws["cue"].shape[0])
    counts = {"Y1": {}, "Y2": {}, "Y3": {}}
    joint = {"Y1": 0, "Y2": 0, "Y3": 0}
    n_frames = len(set(inputs.level0.frames))
    for b in range(total):
        tables = full_row_tables(inputs, draws, b)
        scored = {"Y1": rd.score_y1(tables["Y1"], n_valid_frames=n_frames),
                  "Y2": rd.score_y2(tables["Y2"], valid_frames=tables["Y2_frames"], valid_coordinated=tables["Y2_coordinated"]),
                  "Y3": rd.score_y3(tables["Y3"])}
        for outcome, result in scored.items():
            for name, value in result.get("conditions", {}).items():
                counts[outcome][name] = counts[outcome].get(name, 0) + int(bool(value))
            joint[outcome] += int(result["label"] in (rd.OUTCOME_Y1[0], rd.OUTCOME_Y2[0], rd.OUTCOME_Y3[0]))
    return {outcome: {"conditions": {name: value / total for name, value in counts[outcome].items()}, "joint": joint[outcome] / total} for outcome in counts}


def run_calibration(table: ExposedTable, pools: CalibrationPools, *, log: Callable[[str], None] | None = None, draws: int | None = None,
                    with_020_pass_rates: bool = True) -> dict[str, Any]:
    """The whole calibration after the gate: draws → statistics of every row → cross-check → floors → descriptives.

    No floor is computed before the cross-check has passed; a failing cross-check raises before anything is written.
    Returns the record body and the per-row per-draw arrays (for the outputs directory).
    """
    say = log or (lambda message: None)
    total = B if draws is None else draws
    inputs = kernel_inputs(table, pools)
    rows = {"Y1": [()], "Y2": list(y2_rows()), "Y3": list(y3_rows())}
    strata = variant_strata(pools, inputs, "primary")
    primary = base_draws(strata, "primary", total)
    say(f"base draws: {total} × ({', '.join(f'{name} {len(members)}' for name, members in strata.items())})")
    statistics = row_statistics(inputs, primary, rows)
    say(f"kernel statistics: 1 Y1 row, {len(rows['Y2'])} Y2 rows, {len(rows['Y3'])} Y3 rows")
    checked = cross_check(inputs, primary, statistics, rows)  # raises before any floor exists
    say(f"kernel/direct cross-check: max difference {checked['max_difference']:.1e} over {checked['n_checked']} row-draws")
    record_rows: dict[str, list[dict[str, Any]]] = {"Y1": [], "Y2": [], "Y3": []}
    arrays: dict[str, dict[str, torch.Tensor]] = {"Y1": {}, "Y2": {}, "Y3": {}}  # plus "ceiling:Y1" / "ceiling:Y2": [draws, (level 0, ceiling)]
    full_values: dict[str, dict[str, list[float | None]]] = {}
    full_joint: dict[str, list[bool]] = {}
    for outcome in ("Y1", "Y2", "Y3"):
        for composition in rows[outcome]:
            key = row_key(composition)
            entry, values, per_draw = _row_entry(outcome, composition, statistics[outcome]["all" if outcome == "Y1" else key])
            record_rows[outcome].append(entry)
            row_values = statistics[outcome]["all" if outcome == "Y1" else key]
            arrays[outcome][key] = torch.stack([row_values[name] for name in STATISTICS[outcome]], dim=1)
            if "flattened_r2:ceiling" in row_values:  # the downstream ceiling's distribution, for placing the fresh ceiling
                arrays.setdefault(f"ceiling:{outcome}", {})[key] = torch.stack([row_values["flattened_r2:level0"], row_values["flattened_r2:ceiling"]], dim=1)
            if tuple(composition) == FULL_ROWS[outcome]:
                full_values[outcome], full_joint[outcome] = values, per_draw
    joint_all = sum(1 for b in range(total) if full_joint["Y1"][b] and full_joint["Y2"][b] and full_joint["Y3"][b]) / total
    clamps = {outcome: sorted({name for entry in entries for name, flag in entry["clamped"].items() if flag}) for outcome, entries in record_rows.items()}
    say(f"floors: {sum(len(entries) for entries in record_rows.values())} rows; clamps bound in {clamps}")
    sensitivity: dict[str, Any] = {}
    for variant, wanted in (("all-frames", {"Y2": [FULL_ROWS["Y2"]]}), ("lineage", {"Y1": [()], "Y2": [FULL_ROWS["Y2"]], "Y3": [FULL_ROWS["Y3"]]})):
        variant_draws = base_draws(variant_strata(pools, inputs, variant), variant, total)
        variant_statistics = row_statistics(inputs, variant_draws, {"Y1": wanted.get("Y1"), "Y2": wanted.get("Y2", []), "Y3": wanted.get("Y3", [])})
        entries = {}
        for outcome, compositions in wanted.items():
            for composition in compositions:
                entry, _, _ = _row_entry(outcome, composition, variant_statistics[outcome]["all" if outcome == "Y1" else row_key(composition)])
                entries[outcome] = entry
        sensitivity[variant] = {"draw_digests": variant_draws["digests"], "rows": entries, "pools": {name: len(members) for name, members in variant_strata(pools, inputs, variant).items()}}
    y1_floors = record_rows["Y1"][0]["floors"]
    cohorts = {}
    cue_position = {cue: index for index, cue in enumerate(inputs.cue_units)}
    for cohort, members in pools.cohorts.items():
        values = y1_statistics(inputs.cue, torch.tensor([[cue_position[word] for word, _ in members]], dtype=torch.int64))
        scored = {name: as_values(values[name])[0] for name in Y1_STATISTICS}
        cohorts[cohort] = {"n_cues": len(members), "statistics": scored, "passes": {name: passes(KIND[name], scored[name], y1_floors[name]) for name in Y1_STATISTICS}}
    sensitivity["cohorts"] = cohorts
    body = {
        "draws": {"B": total, "tail_index": tail_index(total), "variant": "primary", "strata": {name: len(members) for name, members in strata.items()}, "index_digests": primary["digests"]},
        "cross_check": checked,
        "rows": record_rows,
        "full_rows": {"compositions": {outcome: list(value) for outcome, value in FULL_ROWS.items()}, "draw_values": full_values, "joint_pass_rate_all_outcomes": joint_all,
                      "pass_rates_020": pass_rates_020(inputs, primary) if with_020_pass_rates else None},
        "sensitivity": sensitivity,
    }
    return {"body": body, "arrays": arrays}


def calibration_record(*, body: Mapping[str, Any], run_id: str, protocol_code_commit: str, digests: Mapping[str, str], pools: CalibrationPools, screen: Mapping[str, Any],
                       precondition: Mapping[str, Any], rematerialization: Mapping[str, Any], gate: Mapping[str, Any], table_digests: Mapping[str, str],
                       array_digests: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "experiment": "021", "schema_version": CALIBRATION_SCHEMA_VERSION, "kind": "exposed-only calibration record (design revision 3)", "design": dict(DESIGN),
        "run_id": run_id, "protocol_code_commit": protocol_code_commit, "inputs": dict(digests), "program_blob_sha1": PROGRAM_BLOB_SHA1,
        "constants": {"B": body["draws"]["B"], "tail_index": body["draws"]["tail_index"], "alpha_per_mille": ALPHA_PER_MILLE, "shares": dict(SHARES), "slots": dict(SLOTS),
                      "variants": list(VARIANTS), "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE, "locked_state_tolerance": LOCKED_STATE_TOLERANCE,
                      "min_screened_frames_per_template": MIN_SCREENED_FRAMES_PER_TEMPLATE, "statistic_agreement_tolerance": STATISTIC_AGREEMENT_TOLERANCE,
                      "statistic_agreement_scale": AGREEMENT_SCALE, "cross_check_draws": CROSS_CHECK_DRAWS, "kinds": dict(KIND),
                      "preconditions": {"scored_tokens": rd.MIN_SCORED_TOKENS, "valid_exposed_frames": rd.MIN_VALID_EXPOSED_FRAMES, "valid_fresh_frames": rd.MIN_VALID_FRESH_FRAMES,
                                        "valid_coordinated": rd.MIN_VALID_COORDINATED_FRAMES, "scorable_fresh_nouns": rd.MIN_SCORABLE_FRESH_NOUNS}},
        "rematerialization": dict(rematerialization), "reproduction_gate": dict(gate), "table_sha256": dict(table_digests),
        "pools": pools.to_json(), "validity_screen": dict(screen), "precondition": dict(precondition),
        **dict(body), "draw_arrays_sha256": dict(array_digests),
    }
    validate_json_safe(record)
    record["content_sha256"] = content_digest(record)
    return record


def floor_tables(record: Mapping[str, Any]) -> dict[str, Any]:
    """The floors the lock carries and ``confirm`` selects from, read off the committed calibration record."""
    rows = record["rows"]
    return {"Y1": dict(rows["Y1"][0]["floors"]), "Y2": {entry["key"]: dict(entry["floors"]) for entry in rows["Y2"]}, "Y3": {entry["key"]: dict(entry["floors"]) for entry in rows["Y3"]}}


def verify_calibration_record(record: Mapping[str, Any]) -> None:
    if record.get("experiment") != "021" or record.get("content_sha256") != content_digest(record):
        raise PhaseError("the calibration record's content digest does not verify")
    constants = record["constants"]
    expected = {"B": B, "tail_index": tail_index(B), "alpha_per_mille": ALPHA_PER_MILLE, "shares": dict(SHARES), "slots": dict(SLOTS), "kinds": dict(KIND), "variants": list(VARIANTS),
                "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE, "locked_state_tolerance": LOCKED_STATE_TOLERANCE, "min_screened_frames_per_template": MIN_SCREENED_FRAMES_PER_TEMPLATE,
                "statistic_agreement_tolerance": STATISTIC_AGREEMENT_TOLERANCE, "statistic_agreement_scale": AGREEMENT_SCALE, "cross_check_draws": CROSS_CHECK_DRAWS,
                "preconditions": {"scored_tokens": rd.MIN_SCORED_TOKENS, "valid_exposed_frames": rd.MIN_VALID_EXPOSED_FRAMES, "valid_fresh_frames": rd.MIN_VALID_FRESH_FRAMES,
                                  "valid_coordinated": rd.MIN_VALID_COORDINATED_FRAMES, "scorable_fresh_nouns": rd.MIN_SCORABLE_FRESH_NOUNS}}
    if {key: constants.get(key) for key in expected} != expected or record.get("program_blob_sha1") != PROGRAM_BLOB_SHA1 or record.get("design") != dict(DESIGN):
        raise PhaseError("the calibration record was computed under different frozen constants, program or design")
    if len(record["rows"]["Y1"]) != 1 or [entry["key"] for entry in record["rows"]["Y2"]] != [row_key(c) for c in y2_rows()] or [entry["key"] for entry in record["rows"]["Y3"]] != [row_key(c) for c in y3_rows()]:
        raise PhaseError("the calibration record does not hold exactly the admissible rows")


# ---------------------------------------------------------------------------
# Row selection at confirm.


def y2_composition(frames: Mapping[str, Mapping[str, Any]]) -> tuple[int, int, int]:
    verdicts = [(bool(entry["valid"]), str(entry["template_id"])) for entry in frames.values()]
    return tuple(sum(1 for valid, template in verdicts if valid and template == axis) for axis in Y2_AXES)  # type: ignore[return-value]


def select_y2_row(frames: Mapping[str, Mapping[str, Any]], tables: Mapping[str, Any]) -> dict[str, Any]:
    """From the stage-1 validity verdicts only (``valid`` and ``template_id`` of each frame) to the committed row."""
    verdicts = {frame_id: {"valid": bool(entry["valid"]), "template_id": str(entry["template_id"])} for frame_id, entry in frames.items()}
    composition = y2_composition(verdicts)
    n, coordinated = sum(composition), composition[Y2_AXES.index(pm.COORDINATED_TEMPLATE)]
    precondition = {"valid_fresh_frames": n >= rd.MIN_VALID_FRESH_FRAMES, "valid_coordinated": coordinated >= rd.MIN_VALID_COORDINATED_FRAMES,
                    "n_valid_fresh_frames": n, "n_valid_coordinated": coordinated}
    precondition["ok"] = bool(precondition["valid_fresh_frames"] and precondition["valid_coordinated"])
    key = row_key(composition) if precondition["ok"] else None
    if key is not None and key not in tables["Y2"]:
        raise pm.IncidentError(f"no committed Y2 floor row for the admissible composition {key}")
    return {"composition": list(composition), "precondition": precondition, "row": key, "floors": dict(tables["Y2"][key]) if key is not None else None,
            "valid_frames": sorted(frame_id for frame_id, entry in verdicts.items() if entry["valid"])}


def selection_digest(selection: Mapping[str, Any], stage1_digest: str) -> str:
    return pm.sha256_text(pm.canonical_json({"selection": dict(selection), "stage1_digest": stage1_digest}))


def assert_y2_selection(stage1: Mapping[str, Any], tables: Mapping[str, Any]) -> dict[str, Any]:
    """The barrier's own check of the Y2 row: its digest, and the same row from the re-read verdicts."""
    selection = stage1.get("y2_selection")
    if selection is None or stage1.get("y2_selection_sha256") != selection_digest(selection, stage1["digest"]):
        raise PhaseError("the digested Y2 floor-row selection does not verify; no fresh cue prompt may run")
    if select_y2_row(stage1["frames"], tables) != selection:
        raise PhaseError("the recorded Y2 floor row is not the one the re-read validity verdicts select")
    return dict(selection)


def select_y3_row(measured_fresh: torch.Tensor, nouns: Sequence[pm.Noun], tables: Mapping[str, Any]) -> dict[str, Any]:
    """From the measured fresh-noun contrasts only (the frozen scorability rule) to the committed row. No prediction."""
    scorable = []
    for index, noun in enumerate(nouns):
        y = measured_fresh[:, index].double()
        if noun.single_token and float(((y - y.mean()) ** 2).sum()) > 0.0:
            scorable.append(index)
    composition = tuple(sum(1 for index in scorable if nouns[index].rule_class == rule) for rule in RULE_CLASSES)
    n = len(scorable)
    precondition = {"scorable_nouns": n >= rd.MIN_SCORABLE_FRESH_NOUNS, "n_scorable_nouns": n, "ok": n >= rd.MIN_SCORABLE_FRESH_NOUNS}
    key = row_key(composition) if precondition["ok"] else None
    if key is not None and key not in tables["Y3"]:
        raise pm.IncidentError(f"no committed Y3 floor row for the admissible composition {key}")
    return {"composition": list(composition), "precondition": precondition, "row": key, "floors": dict(tables["Y3"][key]) if key is not None else None,
            "scorable": [nouns[index].lexical_key for index in scorable], "scorable_index": scorable}


# ---------------------------------------------------------------------------
# Stage 2 with the ceiling, and the scoring.


def stage_two_021(model: Any, pool: Any, confirmation: rd.Confirmation020, lock: Mapping[str, Any], lock_011: Mapping[str, Any], lock_012: Mapping[str, Any],
                  lock_017: Mapping[str, Any], stage1: Mapping[str, Any], *, log: Any = None, enforce: bool = True) -> dict[str, Any]:
    """Experiment 020's stage 2 (``rd.stage_two``), the same calls in the same order, additionally keeping the
    ceiling: the same program fed each target pair's *measured* ``Δx3``, from the same single measurement. With
    ``enforce=False`` the identity maxima are returned unenforced, so the caller can persist the measurements first."""
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    head = ht.HeadWeights.from_model(model)  # noqa: F841 — as in rd.stage_two
    lw = lc.LayerWeights.from_model(model, layers=rd.PROGRAM_LAYERS)
    programs = {layer: atp.LayerProgram.from_model(model, layer) for layer in rd.PROGRAM_LAYERS}
    chain = rd.chain_from_locks(lock_011, lock_012, lock_017, lw, programs, pool)  # noqa: F841 — as in rd.stage_two
    program = rd.ReadoutProgram.from_model(model)
    axis_T = pm.SiteAxis("T", torch.zeros_like(torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64)), torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64), float(lock_011["sigma_T"]))  # noqa: F841
    nouns = rd.NounSet.build(weights, pool.nouns, confirmation.nouns)
    exposed_index, fresh_index = list(nouns.exposed_scorable), list(nouns.fresh)
    exposed_keys = tuple(nouns.nouns[index].lexical_key for index in exposed_index)
    fresh_keys = tuple(nouns.nouns[index].lexical_key for index in fresh_index)
    locked_rows = {(row["token"], row["frame_id"]): row for row in lock["predictions"]["rows"]}
    stage1_rows = {(row["token"], row["frame_id"]): row for row in stage1["rows"]}
    identities: dict[str, float] = {}
    blocks: dict[str, dict[str, list]] = {name: {"cues": [], "frames": [], "templates": [], "measured": [], "predicted": [], "measured_fresh": [], "predicted_fresh": [], "no_l5": [], "base": [], "dT": [],
                                                 "ceiling": []} for name in ("Y1", "Y2")}
    say("stage 2: the fresh cues in the exposed frames (Y1) and in the valid fresh frames (Y2)")
    for name, frames, rows_source in (("Y1", pool.frames, locked_rows), ("Y2", [frame for frame in confirmation.frames if stage1["frames"].get(frame.frame_id, {}).get("valid")], stage1_rows)):
        for frame in frames:
            state = rd.state_from_locked(lock["locked_states"][frame.frame_id] if name == "Y1" else stage1["states"][frame.frame_id], frame)
            c_ref = nouns.contrast_from_residual(program, state.h6)
            if len(state.c_ref) == len(nouns.nouns):  # a stage-1 state carries the captured contrasts: check them
                identities["reference_contrast"] = max(identities.get("reference_contrast", 0.0), float((c_ref - state.c_ref).abs().max()))
            for token in confirmation.tokens:
                measurement = rd.measure_pair(model, state, nouns, token["word"], int(token["token_id"]), c_ref=c_ref)
                for identity, value in rd.pair_identities(program, weights, nouns, state, measurement).items():
                    identities[identity] = max(identities.get(identity, 0.0), value)
                row = rows_source[(token["word"], frame.frame_id)]
                block = blocks[name]
                block["cues"].append(token["word"]); block["frames"].append(frame.frame_id); block["templates"].append(frame.template_id)
                block["measured"].append(measurement.dc[exposed_index]); block["predicted"].append(torch.tensor(row["dc"], dtype=torch.float64))
                if "dc_fresh_nouns" not in row or len(row["dc_fresh_nouns"]) != len(fresh_index):
                    raise PhaseError(f"the committed row for {token['word']}|{frame.frame_id} carries no fresh-noun prediction; Y3 has no committed prediction to score")
                block["measured_fresh"].append(measurement.dc[fresh_index]); block["predicted_fresh"].append(torch.tensor(row["dc_fresh_nouns"], dtype=torch.float64))
                block["no_l5"].append(torch.tensor(row["dc_no_l5_heads"], dtype=torch.float64)); block["base"].append(torch.tensor(row["dc_template_base"], dtype=torch.float64)); block["dT"].append(float(row["dT"]))
                block["ceiling"].append(rd.ceiling_prediction(program, state, nouns, measurement.dx3)[exposed_index])
        say(f"  {name}: {len(blocks[name]['cues'])} pairs measured")
    if enforce:
        rd.enforce_all(identities)
    tables = {}
    for name in ("Y1", "Y2"):
        block = blocks[name]
        common = (tuple(block["cues"]), tuple(block["frames"]), tuple(block["templates"]))
        tables[name] = {
            "exposed": rd.ScoringTable(*common, torch.stack(block["measured"]), torch.stack(block["predicted"]), exposed_keys) if block["cues"] else None,
            "fresh": rd.ScoringTable(*common, torch.stack(block["measured_fresh"]), torch.stack(block["predicted_fresh"]), fresh_keys) if block["cues"] else None,
            "no_l5": rd.ScoringTable(*common, torch.stack(block["measured"]), torch.stack(block["no_l5"]), exposed_keys) if block["cues"] else None,
            "base": rd.ScoringTable(*common, torch.stack(block["measured"]), torch.stack(block["base"]), exposed_keys) if block["cues"] else None,
            "dT": torch.tensor(block["dT"], dtype=torch.float64) if block["cues"] else None,
            "ceiling": rd.ScoringTable(*common, torch.stack(block["measured"]), torch.stack(block["ceiling"]), exposed_keys) if block["cues"] else None,
        }
    return {"tables": tables, "identities": {name: float(value) for name, value in identities.items()}, "noun_keys": list(exposed_keys), "fresh_noun_keys": list(fresh_keys),
            "dw_fresh": nouns.dw[fresh_index], "fresh_nouns": [nouns.nouns[index] for index in fresh_index]}


def _unit_index(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.int64).unsqueeze(0)


def fresh_statistics(outcome: str, table: rd.ScoringTable, *, units: Sequence[str] | None = None) -> dict[str, float | None]:
    """The kernel on a fresh table (every unit once): the same functions that produced the draws' statistics."""
    if outcome == "Y1":
        cues = list(dict.fromkeys(table.cues))
        values = y1_statistics(cue_sums(table, cues), _unit_index(len(cues)))
    elif outcome == "Y2":
        cues, frames = list(dict.fromkeys(table.cues)), list(dict.fromkeys(table.frames))
        grid = grid_sums(table, cues, frames)
        slots = y2_slot_sums(grid, _unit_index(len(cues)), _unit_index(len(frames)))
        values = y2_statistics(slots, list(range(len(frames))), list(grid.templates))
    elif outcome == "Y3":
        cues = list(dict.fromkeys(table.cues))
        nouns = list(units if units is not None else table.noun_keys)
        sums = noun_cue_sums(table, cues, nouns)
        values = y3_statistics(y3_slot_statistics(sums, _unit_index(len(nouns)), _unit_index(len(cues))), list(range(len(nouns))))
    else:
        raise ValueError(outcome)
    return {name: as_values(values[name])[0] for name in STATISTICS[outcome]}


def fresh_direct(outcome: str, table: rd.ScoringTable, *, units: Sequence[str] | None = None) -> dict[str, float | None]:
    if outcome == "Y1":
        return direct_y1(table, list(dict.fromkeys(table.cues)))
    if outcome == "Y2":
        return direct_y2(table, list(dict.fromkeys(table.cues)), list(dict.fromkeys(table.frames)))
    return direct_y3(table, list(dict.fromkeys(table.cues)), list(units if units is not None else table.noun_keys))


def _conditions(outcome: str, values: Mapping[str, float | None], floors: Mapping[str, float | None]) -> dict[str, bool]:
    return {name: passes(KIND[name], values[name], floors[name]) for name in STATISTICS[outcome]}


def score_021(stage1: Mapping[str, Any], measured: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    """Y1, Y2 and Y3 on the rows the evaluability facts selected, through the one predicate; 020's comparators and
    descriptive statistics beside them; the ceiling and the error split; the kernel/direct agreement."""
    tables, floors = measured["tables"], lock["floor_tables"]
    selection = stage1["y2_selection"]
    agreement_worst = (0.0, "")

    def checked(outcome: str, table: rd.ScoringTable, units: Sequence[str] | None = None) -> dict[str, float | None]:
        nonlocal agreement_worst
        kernel = fresh_statistics(outcome, table, units=units)
        difference, name = agreement(kernel, fresh_direct(outcome, table, units=units))
        if difference > agreement_worst[0] or not agreement_worst[1]:
            agreement_worst = (difference, f"{outcome}.{name}")
        if not difference <= STATISTIC_AGREEMENT_TOLERANCE:
            raise pm.IncidentError(f"kernel/direct agreement failed on the fresh {outcome} table: {difference:.3e} at {name}")
        return kernel

    y1_table = tables["Y1"]["exposed"]
    y1: dict[str, Any] = {"population": rd.POPULATIONS["Y1"], "floors": dict(floors["Y1"])}
    if y1_table is None:
        y1.update({"label": rd.OUTCOME_Y1[2], "precondition": {"ok": False}})
    else:
        statistics_020 = rd.pair_statistics(y1_table)
        scored = [cue for cue, entry in statistics_020["per_cue"].items() if entry["n_pairs"] >= rd.MIN_VALID_FRAMES_PER_TOKEN]
        n_frames = len(set(y1_table.frames))
        precondition = {"scored_tokens": len(scored) >= rd.MIN_SCORED_TOKENS, "valid_frames": n_frames >= rd.MIN_VALID_EXPOSED_FRAMES, "n_scored_tokens": len(scored), "n_valid_frames": n_frames}
        precondition["ok"] = bool(precondition["scored_tokens"] and precondition["valid_frames"])
        y1.update({"statistics_020": statistics_020, "precondition": precondition})
        if precondition["ok"]:
            values = checked("Y1", y1_table.subset([i for i, cue in enumerate(y1_table.cues) if cue in set(scored)]))
            conditions = _conditions("Y1", values, floors["Y1"])
            y1.update({"values": values, "conditions": conditions, "label": rd.OUTCOME_Y1[0] if all(conditions.values()) else rd.OUTCOME_Y1[1]})
        else:
            y1["label"] = rd.OUTCOME_Y1[2]
    y2_table = tables["Y2"]["exposed"]
    y2: dict[str, Any] = {"population": rd.POPULATIONS["Y2"], "precondition": selection["precondition"], "row": selection["row"], "composition": selection["composition"],
                          "floors": selection["floors"], "valid_frames": selection["valid_frames"]}
    if not selection["precondition"]["ok"] or y2_table is None:
        y2["label"] = rd.OUTCOME_Y2[2]
    else:
        if sorted(set(y2_table.frames)) != sorted(selection["valid_frames"]):
            raise pm.IncidentError("the Y2 table's frames are not the valid frames the stage-1 selection recorded")
        values = checked("Y2", y2_table)
        conditions = _conditions("Y2", values, selection["floors"])
        y2.update({"statistics_020": rd.pair_statistics(y2_table), "values": values, "conditions": conditions,
                   "label": rd.OUTCOME_Y2[0] if all(conditions.values()) else rd.OUTCOME_Y2[1]})
    fresh_table = tables["Y1"]["fresh"]
    y3: dict[str, Any] = {"population": rd.POPULATIONS["Y3"]}
    if fresh_table is None:
        y3.update({"label": rd.OUTCOME_Y3[2], "precondition": {"ok": False}})
    else:
        selection_y3 = select_y3_row(fresh_table.measured, measured["fresh_nouns"], floors)
        y3.update({"precondition": selection_y3["precondition"], "row": selection_y3["row"], "composition": selection_y3["composition"], "floors": selection_y3["floors"],
                   "scorable": selection_y3["scorable"], "nouns_020": rd.noun_statistics(fresh_table)})
        if selection_y3["precondition"]["ok"]:
            values = checked("Y3", fresh_table, units=[fresh_table.noun_keys[index] for index in selection_y3["scorable_index"]])
            conditions = _conditions("Y3", values, selection_y3["floors"])
            y3.update({"values": values, "conditions": conditions, "label": rd.OUTCOME_Y3[0] if all(conditions.values()) else rd.OUTCOME_Y3[1]})
        else:
            y3["label"] = rd.OUTCOME_Y3[2]
    comparators, ceiling = {}, {}
    for name in ("Y1", "Y2"):
        entry = tables[name]
        if entry["exposed"] is None:
            continue
        comparators[name] = {
            "no_l5_heads": {"standing": rd.COMPARATOR_STANDING["no_l5_heads"], "flattened_r2": rd.pair_statistics(entry["no_l5"])["flattened_r2"]},
            "template_base_mlps": {"standing": rd.COMPARATOR_STANDING["template_base_mlps"], "flattened_r2": rd.pair_statistics(entry["base"])["flattened_r2"]},
            "dT_only": {"standing": rd.COMPARATOR_STANDING["dT_only"], "flattened_r2": rd.dT_only_r2(entry["dT"], entry["exposed"], lock.get("dT_only_noun_vector"))},
            "rank1_nouns": {"standing": rd.COMPARATOR_STANDING["rank1_nouns"], "fresh_noun_r2": rd.rank1_fresh_r2(entry, lock, measured.get("dw_fresh"))},
        }
        level0 = rd.pair_statistics(entry["exposed"])["flattened_r2"]
        top = rd.pair_statistics(entry["ceiling"])["flattened_r2"]
        ceiling[name] = {"flattened_r2": top, "level0_flattened_r2": level0,
                         "error_split": None if level0 is None or top is None else {"total_unexplained": 1.0 - level0, "downstream": 1.0 - top, "inherited_upstream": top - level0}}
    joint = rd.pair_statistics(tables["Y2"]["fresh"]) if tables["Y2"]["fresh"] is not None else None
    return {"Y1": y1, "Y2": y2, "Y3": y3, "joint_fresh_nouns": joint, "comparators": comparators, "ceiling": ceiling,
            "agreement": {"max_difference": agreement_worst[0], "at": agreement_worst[1], "tolerance": STATISTIC_AGREEMENT_TOLERANCE, "scale": AGREEMENT_SCALE},
            "outcome": rd.outcome_label(y1, y2, y3)}


# ---------------------------------------------------------------------------
# The lock (no forward pass) and its validation.


def build_candidate_lock(*, run_id: str, protocol_code_commit: str, digests: Mapping[str, str], confirmation: rd.Confirmation020, rows: Sequence[Mapping[str, Any]],
                         noun_keys: Sequence[str], exploration_020: Mapping[str, Any], record: Mapping[str, Any], record_file_sha256: str) -> dict[str, Any]:
    if digests["confirmation_020"] != confirmation.content_sha256:
        raise PhaseError("the digests name a different confirmation set")
    lock = {"experiment": "021", "run_id": run_id, "protocol_code_commit": protocol_code_commit,
            **{f"{key}_sha256": digests[key] for key in DIGEST_KEYS},
            "program_blob_sha1": PROGRAM_BLOB_SHA1, "calibration_content_sha256": record["content_sha256"], "calibration_file_sha256": record_file_sha256,
            "populations": {name: dict(value) for name, value in rd.POPULATIONS.items()},
            "noun_keys": list(noun_keys), "fresh_noun_keys": [noun.lexical_key for noun in confirmation.nouns],
            "n_predicted_nouns": {"exposed_scorable": len(noun_keys), "fresh": len(confirmation.nouns)},
            "locked_states": exploration_020["locked_states"], "template_bases": exploration_020["template_bases"],
            "rank1": exploration_020["rank1"], "dT_only_noun_vector": exploration_020["comparators"]["dT_only"]["noun_vector"],
            "floor_tables": floor_tables(record), "kinds": dict(KIND),
            "preconditions": {"scored_tokens": rd.MIN_SCORED_TOKENS, "valid_exposed_frames": rd.MIN_VALID_EXPOSED_FRAMES, "valid_fresh_frames": rd.MIN_VALID_FRESH_FRAMES,
                              "valid_coordinated": rd.MIN_VALID_COORDINATED_FRAMES, "scorable_fresh_nouns": rd.MIN_SCORABLE_FRESH_NOUNS},
            "tolerances": {**rd.identity_tolerances(), "statistic_agreement": STATISTIC_AGREEMENT_TOLERANCE, "statistic_agreement_scale": AGREEMENT_SCALE},
            "comparator_standing": dict(rd.COMPARATOR_STANDING), "confirmation_prompt_manifest": rd.manifest_classes(confirmation),
            "predictions": {"rows": [dict(row) for row in rows], "columns": list(rd.PREDICTION_COLUMNS)}}
    lock["content_sha256"] = content_digest(lock)
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], confirmation: rd.Confirmation020, record: Mapping[str, Any],
                  record_file_sha256: str, predictions_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    if lock.get("experiment") != "021" or lock.get("content_sha256") != content_digest(lock):
        raise PhaseError("the installed lock is not a verified Experiment 021 lock")
    if state.get("lock") is None or state["lock"]["content_sha256"] != lock["content_sha256"]:
        raise PhaseError("the installed lock is not the candidate this run wrote")
    if tuple(lock.get(f"{key}_sha256") for key in DIGEST_KEYS) != digest_tuple(digests) or lock["confirmation_020_sha256"] != confirmation.content_sha256:
        raise PhaseError("the lock was written against different frozen inputs (the inherited locks and Experiment 020's closure included)")
    assert_record_inputs(record, digests)
    if lock["program_blob_sha1"] != PROGRAM_BLOB_SHA1:
        raise PhaseError("the lock names a different readout program")
    verify_calibration_record(record)
    if lock["calibration_content_sha256"] != record["content_sha256"] or lock["calibration_file_sha256"] != record_file_sha256:
        raise PhaseError("the lock was written against a different calibration record")
    if pm.canonical_json(lock["floor_tables"]) != pm.canonical_json(floor_tables(record)):
        raise PhaseError("the lock's floor tables are not the committed calibration record's")
    if lock["populations"] != {name: dict(value) for name, value in rd.POPULATIONS.items()} or lock["confirmation_prompt_manifest"] != rd.manifest_classes(confirmation):
        raise PhaseError("the lock's populations or prompt manifest are not the frozen ones")
    if lock["comparator_standing"] != dict(rd.COMPARATOR_STANDING) or lock["kinds"] != dict(KIND):
        raise PhaseError("the lock's comparator standing or statistic kinds are not the frozen ones")
    if not tracked:
        raise PhaseError("the lock and predictions must be tracked and committed")
    if git_state.get("dirty"):
        raise PhaseError("confirm requires a clean Git tree")
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = scientific_changes(changed_paths)
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock: {scientific}")
    if pm.sha256_text(predictions_text) != state["lock"]["predictions_sha256"]:
        raise PhaseError("the installed predictions artifact is not the candidate this run wrote")


def scientific_changes(paths: Sequence[str]) -> list[str]:
    return [path for path in paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and path not in NON_SCIENTIFIC_PATHS and not path.startswith(NON_SCIENTIFIC_PREFIXES)]


def render_predictions(lock: Mapping[str, Any]) -> str:
    rows = lock["predictions"]["rows"]
    y2 = lock["floor_tables"]["Y2"].get("6/6/6", {})
    y3 = lock["floor_tables"]["Y3"].get("8/8/8", {})
    lines = ["# Experiment 021 — preregistered predictions and floors", "",
             f"- Lock run `{lock['run_id']}` at commit `{lock['protocol_code_commit']}`; confirmation set sha256 `{lock['confirmation_020_sha256']}`; program blob `{lock['program_blob_sha1']}`",
             f"- Calibration record content sha256 `{lock['calibration_content_sha256']}`: 1 Y1 row, {len(lock['floor_tables']['Y2'])} Y2 rows, {len(lock['floor_tables']['Y3'])} Y3 rows",
             f"- Y1 floors: {lock['floor_tables']['Y1']}",
             f"- Y2 floors at 6/6/6 (the row is selected at stage 1 from the valid-frame composition): {y2}",
             f"- Y3 floors at 8/8/8 (the row is selected from the measured scorability of the fresh nouns): {y3}",
             "- Pass predicates: R²-type finite and > 0 and ≥ floor; error-type finite and ≤ floor",
             f"- Y1 table: {len(rows)} rows (fresh cues × exposed frames), each with the predicted Δĉ of the {len(lock['noun_keys'])} scorable exposed nouns and the {len(lock['fresh_noun_keys'])} fresh nouns", "",
             "| token | frame | template | mean Δĉ | mean Δĉ (no L05 heads) | mean Δĉ (template-base MLPs) | ΔT̂ |", "|---|---|---|---|---|---|---|"]
    for row in rows[:40]:
        mean = lambda values: sum(values) / len(values) if values else float("nan")  # noqa: E731
        lines.append(f"| {row['token']} | {row['frame_id']} | {row['template']} | {mean(row['dc']):.4f} | {mean(row['dc_no_l5_heads']):.4f} | {mean(row['dc_template_base']):.4f} | {row['dT']:.4f} |")
    if len(rows) > 40:
        lines.append(f"| … | … | … | … | … | … | … |  <!-- {len(rows) - 40} further rows in the lock -->")
    lines += ["", "Every row is a prediction of the model's own contrast change from the frame's reference state, the committed Experiment 011/012/017 locks and the cue's token id alone.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The results state and the phase rules.

DIGEST_KEYS = (*rd.CONFIRMATION_DIGEST_KEYS, "lock_011", "lock_012", "lock_017", "confirmation_020", "closure_020", "extract_020", "results_020_file", "results_020_state")


def new_results_state(*, digests: Mapping[str, str], protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {"schema_version": RESULTS_SCHEMA_VERSION, "experiment": "021", "run_id": pm.sha256_text(pm.canonical_json(dict(digests)) + protocol_code_commit + pm.utc_now())[:16],
            "created_at": pm.utc_now(), **{f"{key}_sha256": digests[key] for key in DIGEST_KEYS}, "program_blob_sha1": PROGRAM_BLOB_SHA1,
            "protocol_code_commit": protocol_code_commit, "git_dirty": False, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "versions": dict(versions),
            "phases": {phase: {"status": "not_started"} for phase in PHASES}, "executed_prompt_keys": [], "executed_noun_keys": [], "calibration": {}, "lock": None, "confirmation": None}


def state_digests(state: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(state[f"{key}_sha256"]) for key in DIGEST_KEYS)


def digest_tuple(digests: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(str(digests[key]) for key in DIGEST_KEYS)


def assert_phase_allowed(phase: str, state: Mapping[str, Any]) -> None:
    status = {name: entry["status"] for name, entry in state["phases"].items()}
    calibration = state.get("calibration") or {}
    if phase == "calibrate":
        if status["calibrate"] == "running" and calibration.get("incidents") and not calibration.get("record_sha256"):
            return  # resumable only after a recorded incident; the runner refuses every commit that carries one
        if status["calibrate"] == "running":
            raise PhaseError("calibrate is running without a recorded incident, or its record is already written; it cannot resume")
        if status["calibrate"] != "not_started":
            raise PhaseError(f"calibrate is {status['calibrate']}; the calibration runs once in this protocol version")
    elif phase == "lock":
        if status["calibrate"] != "complete":
            raise PhaseError(f"lock requires the completed calibrate phase (it is {status['calibrate']})")
        if status["lock"] == "complete":
            raise PhaseError("lock already written; a new candidate lock requires a new protocol version")
    elif phase == "confirm":
        if status["lock"] != "complete":
            raise PhaseError("confirm requires the lock phase")
        if status["confirm"] != "not_started":
            raise PhaseError("confirm already ran; a second scientific attempt requires a new protocol version")
    elif phase == "report":
        if status["calibrate"] not in ("complete", "stopped_for_review") and not calibration.get("incidents"):
            raise PhaseError("report requires a calibrate phase that completed, stopped for review or recorded an incident")
    else:
        raise PhaseError(f"unknown phase {phase}")


def assert_no_manifest_key(ledger: Sequence[str], confirmation: rd.Confirmation020, what: str) -> None:
    forbidden = {prompt.key for prompt in confirmation.all_prompts} & set(ledger)
    if forbidden:
        raise PhaseError(f"{what} holds {len(forbidden)} confirmation-manifest keys, e.g. {sorted(forbidden)[:2]}")


# ---------------------------------------------------------------------------
# The report.


def percentile_of(value: float | None, values: Sequence[float | None], kind: str) -> float | None:
    """Descriptive: the share of exposed-like draws the fresh value meets or beats."""
    if value is None or not math.isfinite(value):
        return None
    if kind == "r2":
        return sum(1 for other in values if other is None or other <= value) / len(values)  # an undefined draw is the worst
    return sum(1 for other in values if other is None or other >= value) / len(values)


def render_report(state: Mapping[str, Any], draw_values: Mapping[str, Mapping[str, Sequence[Sequence[float | None]]]] | None = None,
                  record: Mapping[str, Any] | None = None) -> str:
    f = lambda value, digits=4: "n/a" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))  # noqa: E731
    lines = ["# Experiment 021 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation set: `{state['confirmation_020_sha256']}` (Experiment 020's, read in place)",
             f"- Program blob: `{state['program_blob_sha1']}`; protocol/code commit at calibrate: `{state['protocol_code_commit']}`", "", "## Phases", ""]
    lines += [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    calibration = state.get("calibration") or {}
    incidents = list(calibration.get("incidents", [])) + ([state["confirmation"]["incident"]] if isinstance(state.get("confirmation"), dict) and "incident" in state["confirmation"] else [])
    if incidents:
        lines += ["## Incidents", ""] + [f"- `{entry['phase']}` at commit `{entry.get('commit', '?')}` ({entry['at']}): {entry['message']}" for entry in incidents] + [""]
    if calibration.get("gate"):
        gate = calibration["gate"]
        lines += ["## Calibration (exposed only)", "",
                  f"- Re-materialized {gate['n_pairs']} pairs × {gate['n_nouns']} nouns from Experiment 020's ledger ({len(state['executed_prompt_keys'])} keys); environment drift "
                  f"{calibration.get('environment', {}).get('max_state_drift')}; reproduction gate max difference {gate['max_difference']} (tolerance {gate['tolerance']:.0e})",
                  f"- Identities: " + ", ".join(f"{name} {value:.1e}" for name, value in sorted(calibration.get("identities", {}).items()))]
        if calibration.get("precondition"):
            lines.append(f"- Validity screen and precondition: {calibration['precondition']}")
        if calibration.get("record_summary"):
            lines.append(f"- Calibration record `{calibration.get('record_content_sha256')}`: {calibration['record_summary']}")
        lines.append("")
    confirmation = state.get("confirmation")
    if isinstance(confirmation, dict) and "stage1" in confirmation:
        stage1 = confirmation["stage1"]
        selection = stage1.get("y2_selection", {})
        lines += ["## Confirmation — stage 1", "", f"- Table rows {len(stage1['rows'])}; digest `{stage1['digest']}`; Y2 selection {selection.get('composition')} → row {selection.get('row')} "
                  f"(digest `{stage1.get('y2_selection_sha256')}`)", ""]
    if isinstance(confirmation, dict) and "outcome" in confirmation:
        lines += [f"## Confirmation — stage 2 — `{confirmation['outcome']['label']}`", ""]
        for outcome in ("Y1", "Y2", "Y3"):
            entry = confirmation[outcome]
            lines.append(f"- {outcome}: **{entry['label']}**; row {entry.get('row', 'all')}; precondition {entry.get('precondition')}")
            if entry.get("values"):
                floors = entry.get("floors") or {}
                key = entry.get("row") or "all"
                row_entry = next((row for row in (record or {}).get("rows", {}).get(outcome, []) if row["key"] == key), None)
                for name in STATISTICS[outcome]:
                    percentile = None
                    if draw_values and key in draw_values.get(outcome, {}):
                        column = [row[STATISTICS[outcome].index(name)] for row in draw_values[outcome][key]]
                        percentile = percentile_of(entry["values"][name], column, KIND[name])
                    median = row_entry["summary"][name]["median"] if row_entry else None
                    lines.append(f"  - {name}: fresh {f(entry['values'][name])} against floor {f(floors.get(name))} ({KIND[name]}) → {'pass' if entry['conditions'][name] else 'fail'}; "
                                 f"exposed-like draw median {f(median)}" + (f", fresh percentile among the draws {f(percentile, 3)}" if percentile is not None else ""))
        for name, entry in confirmation.get("ceiling", {}).items():
            key = "all" if name == "Y1" else confirmation[name].get("row")
            placed = None
            if draw_values and key in draw_values.get(f"ceiling:{name}", {}):
                placed = percentile_of(entry["flattened_r2"], [row[1] for row in draw_values[f"ceiling:{name}"][key]], "r2")
            lines.append(f"- {name} ceiling (measured Δx₃): flattened R² {f(entry['flattened_r2'])} against Level 0 {f(entry['level0_flattened_r2'])}; split {entry['error_split']}"
                         + (f"; the fresh ceiling's percentile among the row's exposed-like draws {f(placed, 3)}" if placed is not None else ""))
        for name, entry in confirmation.get("comparators", {}).items():
            lines.append(f"- {name} comparators: " + ", ".join(f"{key} {f(value.get('flattened_r2', value.get('fresh_noun_r2')))} ({value['standing']})" for key, value in sorted(entry.items())))
        if confirmation.get("joint_fresh_nouns"):
            lines.append(f"- Joint diagnostic (fresh cues × fresh frames × fresh nouns; no threshold): flattened R² {f(confirmation['joint_fresh_nouns']['flattened_r2'])}")
        lines += ["", "Interpretation limit (frozen): the floors are relative to the program's own exposed performance; Y2 is conditional on each fresh frame's stage-1 reference state.", ""]
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
