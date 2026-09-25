"""Experiment 024: does an operational nounness score predict when the frozen downstream routing stops holding?

Implements design revision 2 (``9d03dee``) through implementation plan revision 1 (``608088c``) and the reviewer's six
implementation requirements (exact E–N arithmetic; immutable production guard constants; a spent-only tier-C
measurement; bound scientific data dependencies; a pre-write freeze deviation; the confirmation's write order).

**The primary test.** The Spearman correlation across 40 never-executed cue words between a frozen, weight-derived
operational nounness score and each cue's readout MSE over the 108 exposed frames × 79 nouns:

    nounness(w) = cos(E_w, μ_noun) − cos(E_w, μ_cue)
    MSE(cue)    = Σ SSE_C / Σ n over the cue's 108 pairs, SSE_C = Σ_nouns (Δc − C)²

where ``C`` is Experiment 020's frozen Level-0 readout fed the measured ``Δx3`` (``ul.contrast_of``). It passes iff
``ρ ≥ max(F_ρ, null₉₇.₅)``, with the four-way results; ``F_ρ`` comes from SHA-indexed draws of the 139 exposed
calibration cues (023's committed exposed cells) and ``null₉₇.₅`` from SHA-indexed permutations.

**The E–N disambiguation guard.** An exact one-sided permutation test of ``D_EN = mean(log MSE_E) − mean(log MSE_N)``
over all ``C(16, 8) = 12,870`` assignments, in exact integer arithmetic on the observed binary64 values: it passes iff
at most 321 assignments reach the observed contrast (ties included).

**The outcome** combines the two in a frozen hierarchy. Every outcome is predictive and associational.

The module *calls* Experiment 022's ``upstream_localization`` (the measurement, ``C``, the composition used by the
identity gates) and Experiment 023's ``block0_completion`` (the exposed-cells artifact reader and the cell function),
both pinned by git blob with the ten modules 022 pinned; it never edits them. One implementation each of the per-cue
MSE, the score, the Spearman correlation, the OLS line, the E–N guard and the outcome serves every phase.

The per-cue MSE sums its pairs with ``math.fsum`` (an exactly rounded, order-independent sum) rather than the plan's
torch sum: the calibration's artifact rows and the confirmation's stacked cells then give the same number whatever
their memory layout.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from neural_decompiler import attention_patterns as atp
from neural_decompiler import block0_completion as b0c
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

EXPERIMENT = "024"
EXPERIMENT_DIR = "experiments/024-readout-routing-nounness"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
CALIBRATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/calibration-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREREGISTRATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration.md"
DESIGN = {"path": "docs/superpowers/specs/2026-09-24-experiment-024-readout-routing-nounness-design.md", "revision": 2, "commit": "9d03dee"}
PLAN = {"path": "docs/superpowers/plans/2026-09-25-experiment-024-readout-routing-nounness-plan.md", "revision": 1, "commit": "608088c"}

# ---------------------------------------------------------------------------
# The frozen dependencies, by git blob (sha1(b"blob <len>\0" + bytes), computed without git): the ten modules
# Experiment 022 pinned, 022's own module and 023's. Written out literally — never inherited — so that a change in any
# of them makes every phase refuse rather than silently redefine ``C``, the measurement or the calibration source.

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
    "block0_completion.py": "16d310fc7fab5599ec83b8bd8162fb9613f8dad2",
}
_FROZEN_MODULES = {"readout_decompilation.py": rd, "readout_calibration.py": rc, "head_pattern.py": hp, "frame_channels.py": fch, "attention_patterns.py": atp,
                   "layer_correction.py": lc, "plural_mechanism.py": pm, "encoding_read.py": er, "head_transport.py": ht, "models.py": models_module,
                   "upstream_localization.py": ul, "block0_completion.py": b0c}

# Experiment 023's committed, independently reviewed files that 024 reads (the calibration source) or binds (023's
# closure anchor): each file's sha256 and, for JSON, its content digest.
INHERITED_023_PATHS = {"cells_data": b0c.CELLS_DATA_RELATIVE_PATH, "cells_index": b0c.CELLS_INDEX_RELATIVE_PATH, "confirmation": b0c.CONFIRMATION_RELATIVE_PATH,
                       "lock": b0c.LOCK_RELATIVE_PATH}
INHERITED_023 = {
    "cells_data_file_sha256": "d2ee71e58f4555360cc97dbfd883095f5673b636ddaaa6ca560fb7e7d899bba4",
    "cells_index_file_sha256": "628181475e557728f257b428cd3c9e45a525129b6f099766c5a64fb37339a7fe",
    "cells_index_content_sha256": "d38305cb86624a4a1b7e70d21f85ed1b57c26e1d2f59f3ec7dbf2a940f582d16",
    "confirmation_file_sha256": "5fadfa503f4fe35308cb4473220a1d9cd6f7a46e3852f638594b8081909825f4",
    "confirmation_content_sha256": "4e64d4c2c85ae8f9c710171372a4bf28f0964a8e8864a0326182edba44aa14ed",
    "lock_file_sha256": "4bd5a5b14627768d49398c273fa1ce387db2e4739d7d31b7258072ef48beac9c",
    "lock_content_sha256": "97ca520f342e212d5172b1476e9d5a80c6c2622f80a4f99f4e15e0a399007117",
}

# ---------------------------------------------------------------------------
# The fresh classes and the ordered candidate lists (verbatim from design revision 2; each lemma's two forms written
# out, never generated by a rule).

CLASSES = ("N", "B", "D", "C", "E")
CLASS_CONTENT = {
    "N": "non-noun controls: adjectives and determiner-like words with no common noun use (single words)",
    "B": "plural measure and quantity nouns (the plurals of the measure lemmas)",
    "D": "singular measure and quantity nouns (the singulars of the same measure lemmas)",
    "C": "ordinary plural nouns (the plurals of the ordinary lemmas)",
    "E": "ordinary singular nouns (the singulars of the same ordinary lemmas)",
}
N_CANDIDATES = ("eager", "fierce", "honest", "polite", "rude", "sleepy", "wise", "lucky", "merry", "nervous", "anxious", "cheerful", "clumsy", "curious", "grumpy",
                "jealous", "lonely", "nasty", "careful", "careless", "famous", "friendly", "gorgeous", "hungry", "thirsty", "weary", "wicked", "ugly", "vivid", "vague",
                "rapid", "rigid", "clever", "fuzzy", "latest", "earliest", "brave")
MEASURE_LEMMAS = (("gallon", "gallons"), ("ounce", "ounces"), ("acre", "acres"), ("pint", "pints"), ("quart", "quarts"), ("herd", "herds"), ("flock", "flocks"),
                  ("swarm", "swarms"), ("crowd", "crowds"), ("bundle", "bundles"), ("bunch", "bunches"), ("heap", "heaps"), ("mound", "mounds"), ("cluster", "clusters"),
                  ("handful", "handfuls"), ("litre", "litres"), ("liter", "liters"), ("dozen", "dozens"), ("barrel", "barrels"), ("bucket", "buckets"), ("sack", "sacks"),
                  ("crate", "crates"), ("basket", "baskets"), ("carton", "cartons"))
ORDINARY_LEMMAS = (("apple", "apples"), ("horse", "horses"), ("doctor", "doctors"), ("king", "kings"), ("teacher", "teachers"), ("rabbit", "rabbits"), ("poet", "poets"),
                   ("tiger", "tigers"), ("castle", "castles"), ("pencil", "pencils"), ("dragon", "dragons"), ("lion", "lions"), ("farmer", "farmers"),
                   ("soldier", "soldiers"), ("sailor", "sailors"), ("priest", "priests"), ("queen", "queens"), ("baker", "bakers"), ("knight", "knights"),
                   ("lemon", "lemons"), ("banana", "bananas"), ("onion", "onions"), ("carrot", "carrots"), ("violin", "violins"), ("wizard", "wizards"),
                   ("pirate", "pirates"), ("tourist", "tourists"), ("statue", "statues"))
EXPECTED_PICKS = {
    "N": ("honest", "polite", "rude", "sleepy", "wise", "lucky", "merry", "nervous"),
    "measure": ("gallon", "ounce", "acre", "herd", "crowd", "bundle", "cluster", "litre"),
    "ordinary": ("apple", "horse", "doctor", "king", "rabbit", "poet", "dragon", "lion"),
}
PRONOUN_STRATUM = "possessive-or-pronoun"

# ---------------------------------------------------------------------------
# The E–N guard's constants and the configuration. Production is a literal, immutable object; a test world may use a
# smaller configuration only by passing it explicitly (the runner's ``config``), never by patching these.


@dataclass(frozen=True)
class GuardSpec:
    """The exact one-sided permutation test's sizes: ``assignments = C(n_E + n_N, n_E)`` and the largest passing count
    ``max_upper = ⌊0.025 · assignments⌋`` (integer arithmetic), so the size is at most 2.5 %."""

    n_e: int
    n_n: int
    assignments: int
    max_upper: int

    @staticmethod
    def derive(n_e: int, n_n: int) -> "GuardSpec":
        assignments = math.comb(int(n_e) + int(n_n), int(n_e))
        return GuardSpec(int(n_e), int(n_n), assignments, (25 * assignments) // 1000)

    def to_json(self) -> dict[str, Any]:
        return {"n_E": self.n_e, "n_N": self.n_n, "assignments": self.assignments, "max_upper": self.max_upper, "alpha": "0.025, one-sided",
                "statistic": "D_EN = mean(log MSE_E) − mean(log MSE_N)", "log": "natural", "order": "E first, then N, each class in its frozen order",
                "enumeration": "itertools.combinations(range(n_E + n_N), n_E); the observed assignment is the first",
                "arithmetic": "exact: every binary64 log MSE as its exact dyadic rational (as_integer_ratio), subset sums in Python integers",
                "rule": "K = #{assignments S : D(S) ≥ D_EN}, the observed one included (ties count against the guard); PASS iff K ≤ max_upper",
                "p_value": "K / assignments (exact)", "undefined": "NOT_INTERPRETABLE iff an MSE is not finite and positive"}


PRODUCTION_GUARD = GuardSpec(n_e=8, n_n=8, assignments=12_870, max_upper=321)


@dataclass(frozen=True)
class Configuration:
    """Every size the protocol fixes. ``PRODUCTION`` is the design's; a test world passes its own explicitly, and every
    artifact records the configuration it was written under (a lock under one is refused under another)."""

    name: str
    class_quota: int
    draws: int  # B, the primary floor's SHA-indexed draws
    null_permutations: int  # P, the null's SHA-indexed permutations
    contrast_resamples: int  # the secondary bootstrap
    cross_check_draws: int
    calibration_counts: tuple[tuple[str, int], ...]  # stratum -> number of exposed calibration cues
    n_pronoun: int
    n_frames: int
    n_nouns: int
    expected_picks: tuple[tuple[str, tuple[str, ...]], ...]  # "N" / "measure" / "ordinary" -> the design's first picks
    guard: GuardSpec

    def __post_init__(self) -> None:
        if self.guard != GuardSpec.derive(self.guard.n_e, self.guard.n_n):
            raise ValueError(f"inconsistent guard sizes {self.guard}")
        if self.guard.n_e != self.class_quota or self.guard.n_n != self.class_quota:
            raise ValueError("the E–N guard compares the full E and N classes")
        if tuple(stratum for stratum, _ in self.calibration_counts) != b0c.STRATA:
            raise ValueError("the calibration population is the three exposed strata, in 023's order")
        if tuple(key for key, _ in self.expected_picks) != ("N", "measure", "ordinary") or any(len(words) != self.class_quota for _, words in self.expected_picks):
            raise ValueError("expected picks: N, measure and ordinary, one quota each")

    @property
    def n_fresh(self) -> int:
        return len(CLASSES) * self.class_quota

    @property
    def n_calibration(self) -> int:
        return sum(count for _, count in self.calibration_counts)

    def expected(self, key: str) -> tuple[str, ...]:
        return dict(self.expected_picks)[key]

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "class_quota": self.class_quota, "n_fresh": self.n_fresh, "draws": self.draws, "null_permutations": self.null_permutations,
                "contrast_resamples": self.contrast_resamples, "cross_check_draws": self.cross_check_draws,
                "calibration_counts": dict(self.calibration_counts), "n_calibration": self.n_calibration, "n_pronoun": self.n_pronoun, "n_frames": self.n_frames,
                "n_nouns": self.n_nouns, "expected_picks": {key: list(words) for key, words in self.expected_picks}, "guard": self.guard.to_json()}


PRODUCTION = Configuration(
    name="production", class_quota=8, draws=10_000, null_permutations=100_000, contrast_resamples=10_000, cross_check_draws=16,
    calibration_counts=(("determiner-like", 45), ("quantity", 45), ("adjective", 49)), n_pronoun=36, n_frames=108, n_nouns=79,
    expected_picks=tuple((key, tuple(words)) for key, words in EXPECTED_PICKS.items()), guard=PRODUCTION_GUARD)

# ---------------------------------------------------------------------------
# Tags, tolerances, results.

PRIMARY_TAG = "024|primary"
NULL_TAG = "024|null"
CONTRAST_TAG = "024|contrast"
TOLERANCES = {
    "I1": 1e-4,  # block-0 decomposition against the measured Δx1, max abs
    "I3": 1e-4,  # 022's full composition against the measured Δx3, relative
    "I4": 1e-3,  # the full composition's Δĉ against C, max abs (nats)
    "C_recompute": 0.0,  # C recomputed from the saved Δx3 against the confirm-time C: bit for bit
    "I7": 0.0,  # the lock's weight-derived quantities recomputed before any prompt: bit for bit
    "spearman": 1e-12,  # the canonical Spearman against the independent route
    "en_float": 1e-12,  # D_EN in exact arithmetic against a plain float64 mean difference
    "mse": 1e-12,  # the canonical per-cue MSE against a torch float64 sum, relative
    "level1": 2e-2,  # 020's Level-1 identity, descriptive only
}
GROUPS = ("cue_final", "coordinated")
PRIMARY_RESULTS = ("NOT_INTERPRETABLE", "GUARD_FAILURE", "ENVELOPE_ONLY_FAILURE", "PASS")
GUARD_RESULTS = ("NOT_INTERPRETABLE", "FAIL", "PASS")
OUTCOMES = ("NOT_INTERPRETABLE", "NOUNNESS_PREDICTION_NOT_ESTABLISHED", "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED",
            "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS")
STATISTIC = ("ρ = Spearman(nounness, MSE) across the fresh cues; MSE = Σ SSE_C / Σ n over the cue's exposed pairs, SSE_C = Σ_nouns (Δc − C)²; ranks ascending with "
             "ties at the average rank, then the Pearson correlation of the two rank vectors in float64")
C_DEFINITION = ("C: Experiment 020's frozen Level-0 readout fed the measured Δx3 (ul.contrast_of): block 3 recomputed exactly with the MLP at the frame's operating "
                "point; block-4 attention frozen at the reference; block-5 heads as the reference rows times the value changes; exact LN_final; the noun read over the "
                "79 scorable exposed nouns")

# ---------------------------------------------------------------------------
# Frozen dependencies and inherited inputs.


def module_blobs() -> dict[str, str]:
    return {name: rc.program_blob_sha1(Path(module.__file__)) for name, module in _FROZEN_MODULES.items()}


def assert_frozen_blobs() -> dict[str, str]:
    """Every pinned module — 022's and 023's included — is byte for byte the pinned one."""
    actual = module_blobs()
    differing = sorted(name for name, blob in FROZEN_BLOBS.items() if actual.get(name) != blob)
    if differing:
        raise PhaseError(f"frozen modules changed: {differing}; Experiment 024 runs the pinned programs only")
    return actual


def own_blob() -> str:
    return rc.program_blob_sha1(Path(__file__))


def verify_023_inputs(root: Path) -> dict[str, str]:
    """Experiment 023's committed exposed cells (the calibration source), confirmation and lock: each is the reviewed
    file (sha256) and, for JSON, verifies against its reviewed content digest."""
    out: dict[str, str] = {}
    for kind, content in (("cells_data", False), ("cells_index", True), ("confirmation", True), ("lock", True)):
        relative = INHERITED_023_PATHS[kind]
        path = Path(root) / relative
        if not path.exists() or rc.file_sha256(path) != INHERITED_023[f"{kind}_file_sha256"]:
            raise PhaseError(f"Experiment 023's committed {kind.replace('_', ' ')} file {relative} is missing or not the reviewed file")
        out[f"023_{kind}_file"] = INHERITED_023[f"{kind}_file_sha256"]
        if content:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("content_sha256") != rc.content_digest(payload) or payload["content_sha256"] != INHERITED_023[f"{kind}_content_sha256"]:
                raise PhaseError(f"Experiment 023's {kind.replace('_', ' ')} file does not verify against its reviewed content digest")
            out[f"023_{kind}_content"] = INHERITED_023[f"{kind}_content_sha256"]
    return out


DIGEST_KEYS = (*b0c.DIGEST_KEYS, "023_cells_data_file", "023_cells_index_file", "023_cells_index_content", "023_confirmation_file", "023_confirmation_content",
               "023_lock_file", "023_lock_content")


def base_digests(inputs: ul.FrozenInputs, digests_022: Mapping[str, str], digests_023: Mapping[str, str]) -> dict[str, str]:
    return {**{key: inputs.digests[key] for key in rc.DIGEST_KEYS}, **dict(digests_022), **dict(digests_023)}


def calibration_source() -> dict[str, Any]:
    """What every record binds of the calibration source: 023's reviewed exposed-cells files."""
    return {"data_path": b0c.CELLS_DATA_RELATIVE_PATH, "index_path": b0c.CELLS_INDEX_RELATIVE_PATH, "data_sha256": INHERITED_023["cells_data_file_sha256"],
            "index_sha256": INHERITED_023["cells_index_file_sha256"], "index_content_sha256": INHERITED_023["cells_index_content_sha256"],
            "cells_version": b0c.CELLS_VERSION, "columns": list(b0c.CELL_COLUMNS)}


# ---------------------------------------------------------------------------
# 1. Cells and the per-cue MSE (one implementation, shared by calibration and confirm).

COUNT = b0c.CELL_COLUMNS.index("n")
SUM = b0c.CELL_COLUMNS.index("S")
SQUARES = b0c.CELL_COLUMNS.index("Q")
SSEC = b0c.CELL_COLUMNS.index("SSEC")


def fresh_pair_cells(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """A fresh pair's cells by 023's own cell function (``b0c.pair_cells``): 024 has no ``P0`` or ``P1``, so ``C``
    stands in for both and the SSE0/SSE1 columns equal SSEC; only ``n``, ``SSEC`` and (descriptively) ``S``, ``Q`` are
    read."""
    return b0c.pair_cells(y, c, c, c)


def cue_mse(cells: torch.Tensor) -> float:
    """``Σ SSE_C / Σ n`` over one cue's pairs (``[pairs, 8]`` cells in ``frame_id`` order), each sum exactly rounded
    (``math.fsum``), so the value does not depend on the order or layout of the rows."""
    return math.fsum(float(v) for v in cells[:, SSEC].tolist()) / math.fsum(float(v) for v in cells[:, COUNT].tolist())


def cue_mse_torch(cells: torch.Tensor) -> float:
    """The cross-check route: plain torch float64 sums."""
    return float(cells[:, SSEC].double().sum() / cells[:, COUNT].double().sum())


def log_mse(value: float | None) -> float | None:
    """The natural logarithm; defined only for a finite, positive MSE."""
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        return None
    return math.log(float(value))


# ---------------------------------------------------------------------------
# 2. The operational nounness score (weights only).


def embedding_digest(W_E: torch.Tensor) -> str:
    """sha256 over the shape and the little-endian float32 bytes of the input embedding matrix."""
    array = W_E.detach().cpu().contiguous().to(torch.float32)
    return hashlib.sha256(json.dumps(list(array.shape)).encode("ascii") + b"|<f4|" + array.numpy().astype("<f4", copy=False).tobytes()).hexdigest()


def parameters_digest(model: Any) -> str:
    """sha256 over every named parameter of the loaded model (name, shape and float32 bytes, by name): the exact weights
    ``C``, the measurement and the score are computed from."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        array = parameter.detach().cpu().contiguous().to(torch.float32)
        digest.update(name.encode("utf-8") + b"|" + json.dumps(list(array.shape)).encode("ascii") + b"|" + array.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def noun_row_ids(pool: Any) -> list[int]:
    """The ``μ_noun`` rows: the scorable exposed nouns in pool order (single-token), each singular then plural."""
    return [int(token_id) for noun in pool.nouns if noun.single_token for token_id in (noun.sg_ids[0], noun.pl_ids[0])]


def calibration_cues(units: b0c.ExposedUnits) -> list[tuple[str, int, str]]:
    """The exposed cues of the three strata, in 023's canonical order: the calibration population and ``μ_cue``."""
    return [(word, int(token_id), cls) for word, token_id, cls in units.cues if cls in b0c.STRATA]


def centroid(W_E: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    return torch.stack([W_E[int(i)].double() for i in ids]).mean(0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


def nounness(e: torch.Tensor, mu_noun: torch.Tensor, mu_cue: torch.Tensor) -> float:
    """``cos(E_w, μ_noun) − cos(E_w, μ_cue)`` in float64."""
    e = e.double()
    return cosine(e, mu_noun) - cosine(e, mu_cue)


def score_bindings(W_E: torch.Tensor, pool: Any, units: b0c.ExposedUnits) -> dict[str, Any]:
    """Everything the score depends on, bound by digest: the embedding matrix, the two id lists, the two centroids."""
    noun_ids = noun_row_ids(pool)
    cue_ids = [token_id for _, token_id, _ in calibration_cues(units)]
    mu_noun, mu_cue = centroid(W_E, noun_ids), centroid(W_E, cue_ids)
    return {"embedding_sha256": embedding_digest(W_E), "noun_row_ids": noun_ids, "noun_row_ids_sha256": pm.sha256_text(pm.canonical_json(noun_ids)),
            "calibration_cue_ids": cue_ids, "calibration_cue_ids_sha256": pm.sha256_text(pm.canonical_json(cue_ids)), "mu_noun_sha256": rc.tensor_digest(mu_noun),
            "mu_cue_sha256": rc.tensor_digest(mu_cue),
            "definition": "nounness(w) = cos(E_w, μ_noun) − cos(E_w, μ_cue); E_w = W_E[id(' ' + w)] in float64; μ_noun = the mean of the 158 singular and plural rows "
                          "of the 79 scorable exposed nouns; μ_cue = the mean of the 139 exposed calibration-cue rows, leave-one-out (a direct mean of the other 138) "
                          "for a calibration cue; cos = ⟨a, b⟩ / (‖a‖ ‖b‖)"}


def centroids(W_E: torch.Tensor, bindings: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return centroid(W_E, bindings["noun_row_ids"]), centroid(W_E, bindings["calibration_cue_ids"])


def calibration_scores(W_E: torch.Tensor, bindings: Mapping[str, Any]) -> list[float]:
    """The leave-one-out score of every calibration cue: ``μ_cue`` a direct mean of the other rows, in order."""
    ids = list(bindings["calibration_cue_ids"])
    mu_noun = centroid(W_E, bindings["noun_row_ids"])
    return [nounness(W_E[token_id], mu_noun, centroid(W_E, ids[:k] + ids[k + 1:])) for k, token_id in enumerate(ids)]


def full_scores(W_E: torch.Tensor, bindings: Mapping[str, Any], token_ids: Sequence[int]) -> list[float]:
    """The score with the full ``μ_cue``: every fresh cue, and the pronoun cues (descriptive)."""
    mu_noun, mu_cue = centroids(W_E, bindings)
    return [nounness(W_E[int(token_id)], mu_noun, mu_cue) for token_id in token_ids]


# ---------------------------------------------------------------------------
# 3. Spearman (one implementation for every draw, permutation, fresh ρ and secondary Spearman).


def _finite(values: Sequence[float]) -> list[float]:
    out = [float(v) for v in values]
    if not all(math.isfinite(v) for v in out):
        raise ValueError("a Spearman input is not finite")
    return out


def average_ranks(values: Sequence[float]) -> list[float]:
    """One-based ascending ranks; exactly equal values share the mean of their positions."""
    values = _finite(values)
    n = len(values)
    order = sorted(range(n), key=lambda i: (values[i], i))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """The Pearson correlation of the two average-rank vectors, in float64; ``None`` when either rank vector is constant.
    The centred ranks are multiples of 1/2, so every sum below is exact and only the last division and square root
    round."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman needs two equally long vectors of at least two values")
    ra, rb = average_ranks(x), average_ranks(y)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    a, b = [r - ma for r in ra], [r - mb for r in rb]
    saa, sbb = sum(v * v for v in a), sum(v * v for v in b)
    if saa == 0.0 or sbb == 0.0:
        return None
    return sum(p * q for p, q in zip(a, b)) / math.sqrt(saa * sbb)


def spearman_direct(x: Sequence[float], y: Sequence[float]) -> float | None:
    """The independent route, for the cross-checks only: ranks by pairwise counting, sums by ``math.fsum``."""
    x, y = _finite(x), _finite(y)

    def ranks(values: list[float]) -> list[float]:
        return [1.0 + sum(1 for w in values if w < v) + (sum(1 for w in values if w == v) - 1) / 2 for v in values]

    ra, rb = ranks(x), ranks(y)
    ma, mb = math.fsum(ra) / len(ra), math.fsum(rb) / len(rb)
    saa = math.fsum((r - ma) ** 2 for r in ra)
    sbb = math.fsum((r - mb) ** 2 for r in rb)
    if saa == 0.0 or sbb == 0.0:
        return None
    return math.fsum((p - ma) * (q - mb) for p, q in zip(ra, rb)) / math.sqrt(saa * sbb)


def spearman_agreement(canonical: float | None, direct: float | None) -> float:
    if canonical is None and direct is None:
        return 0.0
    if canonical is None or direct is None:
        return math.inf
    return abs(canonical - direct)


# ---------------------------------------------------------------------------
# 4. The exposed-data OLS line (secondary; never in the primary statistic).


def ols(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """``log MSE`` on nounness: slope ``Sxy/Sxx``, intercept ``ȳ − slope·x̄``, residual sd ``sqrt(Σr²/(n − 2))``; every sum
    by ``math.fsum``. Full precision; never rounded."""
    x, y = _finite(x), _finite(y)
    n = len(x)
    if n != len(y) or n < 3:
        raise ValueError("the line needs at least three points")
    mx, my = math.fsum(x) / n, math.fsum(y) / n
    dx, dy = [v - mx for v in x], [v - my for v in y]
    sxx = math.fsum(d * d for d in dx)
    if sxx == 0.0:
        raise ValueError("the line needs varying nounness")
    slope = math.fsum(p * q for p, q in zip(dx, dy)) / sxx
    intercept = my - slope * mx
    residual_sd = math.sqrt(math.fsum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y)) / (n - 2))
    return {"slope": slope, "intercept": intercept, "residual_sd": residual_sd, "n": n,
            "formulas": "slope = Sxy/Sxx, intercept = ȳ − slope·x̄, residual sd = sqrt(Σ r² / (n − 2)); sums by math.fsum; y = log MSE, x = leave-one-out nounness"}


def predict(line: Mapping[str, Any], x: float) -> float:
    return float(line["intercept"]) + float(line["slope"]) * float(x)


# ---------------------------------------------------------------------------
# 5. SHA indices and 6. order statistics.


def sha_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def primary_draw_index(b: int, slot: int, n: int) -> int:
    """``int.from_bytes(sha256(f"024|primary|{b}|{slot}")[:8], "big") % n``: a calibration cue, with replacement."""
    return sha_int(f"{PRIMARY_TAG}|{b}|{slot}") % int(n)


def primary_draw_indices(draws: int, slots: int, n: int) -> list[list[int]]:
    return [[primary_draw_index(b, slot, n) for slot in range(slots)] for b in range(draws)]


def null_permutation(p: int, n: int) -> list[int]:
    """Fisher–Yates from the identity: for ``i`` from ``n − 1`` down to 1, ``j = sha(f"024|null|{p}|{i}") % (i + 1)``,
    swap positions ``i`` and ``j``."""
    perm = list(range(int(n)))
    for i in range(int(n) - 1, 0, -1):
        j = sha_int(f"{NULL_TAG}|{p}|{i}") % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def contrast_draw_index(b: int, cls: str, slot: int, n: int) -> int:
    return sha_int(f"{CONTRAST_TAG}|{b}|{cls}|{slot}") % int(n)


def lower_rank(count: int) -> int:
    """⌈0.025·count⌉ in integer arithmetic: 250 at 10,000 (element [249])."""
    return (25 * int(count) + 999) // 1000


def null_rank(count: int) -> int:
    """⌈0.975·count⌉ in integer arithmetic: 97,500 at 100,000 (element [97499])."""
    return (975 * int(count) + 999) // 1000


def order_statistic(values: Sequence[float | None], rank: int) -> float:
    """The ``rank``-th ascending value (1-based), an undefined value placed at −∞."""
    tensor = torch.tensor([math.nan if v is None else float(v) for v in values], dtype=torch.float64)
    return ul.order_statistic(tensor, ~torch.isnan(tensor), int(rank), undefined_at=-math.inf)


def defined_median(values: Sequence[float | None]) -> float | None:
    tensor = torch.tensor([math.nan if v is None else float(v) for v in values], dtype=torch.float64)
    return ul.defined_median(tensor, ~torch.isnan(tensor))


# ---------------------------------------------------------------------------
# 7. The primary classification.


def classify_primary(rho: float | None, floor: float, null: float) -> str:
    """``NOT_INTERPRETABLE`` (ρ undefined) → ``GUARD_FAILURE`` (ρ < null₉₇.₅) → ``ENVELOPE_ONLY_FAILURE`` (ρ < F_ρ) →
    ``PASS``: a PASS needs ``ρ ≥ max(F_ρ, null₉₇.₅)``."""
    if any(not math.isfinite(float(bound)) for bound in (floor, null)):
        raise IncidentError("a non-finite primary threshold")
    if rho is None:
        return "NOT_INTERPRETABLE"
    if not math.isfinite(float(rho)):
        raise IncidentError("a non-finite ρ")
    if rho < null:
        return "GUARD_FAILURE"
    if rho < floor:
        return "ENVELOPE_ONLY_FAILURE"
    return "PASS"


# ---------------------------------------------------------------------------
# 8. The E–N disambiguation guard: an exact one-sided permutation test.


class GuardCheckError(IncidentError):
    """An implementation check of the exact enumeration failed: an incident, never a result."""


def exact_integers(values: Sequence[float]) -> tuple[list[int], int]:
    """Every finite binary64 value as an integer on one common dyadic scale: ``v_i = z_i / q`` exactly."""
    ratios = [float(v).as_integer_ratio() for v in values]
    q = max(den for _, den in ratios)
    return [num * (q // den) for num, den in ratios], q


def exact_subset_sums(logs: Sequence[float], spec: GuardSpec) -> dict[str, Any]:
    """Every assignment's exact E-side sum: ``itertools.combinations(range(n_E + n_N), n_E)`` (the observed assignment,
    positions ``0 … n_E − 1``, first) over the values as exact integers on one dyadic scale. Raises ``GuardCheckError``
    unless the enumeration has ``spec.assignments`` distinct assignments and, for equal group sizes, every complement
    carries the total minus the subset's sum."""
    n_all = spec.n_e + spec.n_n
    z, q = exact_integers(logs)
    total = sum(z)
    subsets = list(itertools.combinations(range(n_all), spec.n_e))
    if len(subsets) != spec.assignments or len(set(subsets)) != spec.assignments or subsets[0] != tuple(range(spec.n_e)):
        raise GuardCheckError(f"the enumeration does not give {spec.assignments} distinct assignments with the observed one first")
    sums = [sum(z[i] for i in subset) for subset in subsets]
    if spec.n_e == spec.n_n:
        index = {subset: row for row, subset in enumerate(subsets)}
        everything = frozenset(range(n_all))
        if any(sums[index[tuple(sorted(everything - set(subset)))]] != total - sums[row] for row, subset in enumerate(subsets)):
            raise GuardCheckError("an assignment's complement does not carry the total minus its exact sum")
    return {"sums": sums, "total": total, "q": q, "n_subsets": len(subsets)}


def upper_count(sums: Sequence[int], observed: int) -> int:
    """``K``: the assignments whose exact sum is at or above the observed one (ties included, the observed counted)."""
    return sum(1 for value in sums if value >= observed)


def en_decision(k: int, spec: GuardSpec) -> str:
    """PASS iff ``K ≤ max_upper`` (321 of 12,870 in production: 321/12,870 ≤ 0.025 < 322/12,870)."""
    return "PASS" if int(k) <= spec.max_upper else "FAIL"


def en_guard_logs(logs: Sequence[float], spec: GuardSpec) -> dict[str, Any]:
    """The exact test on ``n_E + n_N`` finite ``log MSE`` values, E first."""
    logs = [float(v) for v in logs]
    if len(logs) != spec.n_e + spec.n_n or not all(math.isfinite(v) for v in logs):
        raise PhaseError("the exact E–N test needs n_E + n_N finite values")
    exact = exact_subset_sums(logs, spec)
    sums, total, q = exact["sums"], exact["total"], exact["q"]
    observed = sums[0]
    k = upper_count(sums, observed)

    def contrast(t: int) -> Fraction:
        return Fraction(spec.n_n * t - spec.n_e * (total - t), spec.n_e * spec.n_n * q)

    d_exact = contrast(observed)
    plain = sum(logs[:spec.n_e]) / spec.n_e - sum(logs[spec.n_e:]) / spec.n_n
    difference = abs(float(d_exact) - plain)
    if not difference <= TOLERANCES["en_float"]:
        raise GuardCheckError(f"D_EN in exact arithmetic differs from the plain float64 mean difference by {difference:.3e}")
    ordered = sorted(sums)
    threshold = next((value for value in sorted(set(sums)) if len(ordered) - bisect.bisect_left(ordered, value) <= spec.max_upper), None)
    return {"result": en_decision(k, spec), "K": k, "assignments": spec.assignments, "max_upper": spec.max_upper, "p_exact": f"{k}/{spec.assignments}",
            "p_value": k / spec.assignments, "D_EN": float(d_exact), "D_EN_exact": f"{d_exact.numerator}/{d_exact.denominator}",
            "threshold_D": None if threshold is None else float(contrast(threshold)), "threshold": "+inf" if threshold is None else "finite",
            "ratio_of_geometric_means": math.exp(float(d_exact)),
            "checks": {"assignments": exact["n_subsets"], "complement_identity": spec.n_e == spec.n_n, "float_difference": difference}}


def en_guard(e_mse: Sequence[float | None], n_mse: Sequence[float | None], spec: GuardSpec) -> dict[str, Any]:
    """The guard on the observed per-cue MSE of the E cues then the N cues, each class in its frozen order: the natural
    log of each (``log_mse``), then the exact test. ``NOT_INTERPRETABLE`` iff an MSE is not finite and positive."""
    e_mse, n_mse = list(e_mse), list(n_mse)
    if len(e_mse) != spec.n_e or len(n_mse) != spec.n_n:
        raise PhaseError(f"the E–N guard is bound to {spec.n_e} + {spec.n_n} cues, not {len(e_mse)} + {len(n_mse)}")
    logs = [log_mse(value) for value in e_mse + n_mse]
    out: dict[str, Any] = {"spec": spec.to_json(), "mse": [None if v is None else float(v) for v in e_mse + n_mse], "log_mse": logs}
    if any(value is None for value in logs):
        return {**out, "result": "NOT_INTERPRETABLE", "K": None, "p_value": None, "D_EN": None}
    groups = {name: {"mean_mse": math.fsum(values) / len(values), "mean_log_mse": math.fsum(log_values) / len(log_values)}
              for name, values, log_values in (("E", e_mse, logs[:spec.n_e]), ("N", n_mse, logs[spec.n_e:]))}
    return {**out, **en_guard_logs(logs, spec), "groups": groups}


# ---------------------------------------------------------------------------
# 9. The outcome and the frozen semantics.


def outcome(primary: str, guard: str) -> str:
    """The frozen hierarchy (an incident writes no result at all, so it never reaches this)."""
    if primary not in PRIMARY_RESULTS or guard not in GUARD_RESULTS:
        raise ValueError(f"unknown results {primary!r} / {guard!r}")
    if primary == "NOT_INTERPRETABLE":
        return "NOT_INTERPRETABLE"
    if primary != "PASS":
        return "NOUNNESS_PREDICTION_NOT_ESTABLISHED"
    return "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS" if guard == "PASS" else "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED"


EXTRAPOLATION_SENTENCE = ("{k} of the {n} E cues lie above the calibration population's maximum nounness ({maximum:+.6f}); the fresh test is a prospective "
                          "extrapolation test there. Calibration supplied no evidence for that range, and the post-hoc behaviour of 023's five spent cues in it is "
                          "design motivation only.")
SEMANTICS = {
    "primary": {
        "NOT_INTERPRETABLE": "ρ cannot be defined (a constant rank vector); no result",
        "GUARD_FAILURE": "ρ < null₉₇.₅: no evidence that the score predicts the readout error on fresh cues beyond chance",
        "ENVELOPE_ONLY_FAILURE": "null₉₇.₅ ≤ ρ < F_ρ: the score carries predictive information beyond chance, but less than the exposed relationship would lead "
                                 "one to expect",
        "PASS": "ρ ≥ max(F_ρ, null₉₇.₅): higher operational nounness prospectively predicted larger error of the frozen block-4/5 attention readout, at least as "
                "strongly as the exposed-like relationship and beyond chance",
    },
    "guard": {
        "NOT_INTERPRETABLE": "an E or N MSE is not finite and positive; the guard cannot be computed",
        "FAIL": "the ordinary singular nouns did not have larger error than the non-noun controls beyond chance (K > the bound); this is not evidence against "
                "nounness",
        "PASS": "the ordinary singular nouns had larger error than the non-noun controls beyond chance (an exact one-sided permutation test, K ≤ the bound)",
    },
    "outcomes": {
        "NOT_INTERPRETABLE": "no result",
        "NOUNNESS_PREDICTION_NOT_ESTABLISHED": "the primary requirement failed; its four-way result says how, and the E–N guard is descriptive only",
        "ASSOCIATION_PREDICTED_BUT_NOUNNESS_NOT_DISAMBIGUATED": "the score predicted the error, but the result does not separate nounness from plural morphology "
                                                                "or measure semantics; the secondary contrasts describe the shape of the effect, with no winner",
        "NOUNNESS_PREDICTS_READOUT_ERROR_BEYOND_SIMPLE_PLURALITY_OR_MEASURE_CLASS":
            "the frozen weight-derived score prospectively predicted the frozen readout's error on new lexical representations, including substantial "
            "extrapolation beyond the exposed score range, and the ordinary singular nouns had larger error than the non-noun controls beyond chance — which "
            "neither of the two preregistered simple alternatives (plural morphology alone, measure class alone) predicts",
    },
    "simple": "the guard eliminates the two preregistered simple alternatives, plurality-only and measure-class-only; it does not establish nounness as a unique "
              "causal factor or eliminate every correlated lexical property",
    "not_shown": ["that nounness causes the attention change", "that the score measures linguistic nounhood",
                  "which part of the score matters: similarity to the target nouns or dissimilarity to the exposed cues (both separate E from N)",
                  "that no other property that differs between ordinary nouns and these adjectives explains E > N",
                  "that the exposed OLS line has been validated over the extrapolated range (the line is secondary)",
                  "generality beyond the 108 exposed frames, the 79 nouns and this checkpoint"],
    "predictive": "every outcome is predictive and associational; none establishes that nounness causally changes attention",
    "secondary": "the secondary analyses (the per-group Spearman, normalized MSE, Var(Δc), R²_C, bias, slope, the line, the B/C/D/E factorial contrasts and the "
                 "block-4/5 ladder) are computed after the outcome is written and can never rescue, alter or redefine it",
    "incidents": "an incident carries no result; an identity incident never coexists with an outcome",
    "precedence": {"primary": list(PRIMARY_RESULTS), "guard": list(GUARD_RESULTS), "outcomes": list(OUTCOMES)},
    "extrapolation": EXTRAPOLATION_SENTENCE,
}


# ---------------------------------------------------------------------------
# The freeze: tokenizer, text rules and committed files only; no model output (Task 3).

CONFIRMATION_SCHEMA_VERSION = 1
FREEZE_RULES = {
    "word": "' ' + w is a single token of the pinned tokenizer",
    "exclusion": "its id was never used as a cue by Experiments 005–023: 023's exclusion (b0c.exclusion, reproduced from the frozen inputs) plus 023's own cues "
                 "and its frames' cue ids",
    "target_nouns": "its id is not a form (singular or plural) of any target noun of the pool",
    "frame_tokens": "its id occurs nowhere in the 108 exposed frames (their prefix and suffix tokens): the lesson of 023's ' shiny'",
    "lemmas": "a measure or ordinary lemma counts only when both its singular and its plural are eligible (and distinct)",
    "picks": "the first class-quota eligible entries of each ordered list, mechanically; B and C are the plurals, D and E the singulars of the picked lemmas",
    "deviation": "picks that differ from the design's expected picks write nothing and stop for review; no reserve is ever substituted",
    "score": "the nounness score is never computed at the freeze and never selects, rejects, reorders or classifies a candidate",
}


class FreezeShortfall(ul.FreezeShortfall):
    """A list yields fewer eligible entries than its quota: nothing is written; stop for review (not an incident)."""


class FreezeDeviation(RuntimeError):
    """The mechanical picks differ from the design's expected picks: nothing is written and no state is created; stop for
    review. The power analysis and the extrapolation statement were computed for the expected 40 words."""


def exclusion(base: Mapping[str, Any], confirmation_023: Mapping[str, Any], confirmation_023_file_sha256: str) -> dict[str, Any]:
    """Experiment 023's freeze exclusion — ``b0c.exclusion`` recomputed from the frozen inputs and 022's committed
    confirmation, which must reproduce the digest 023's committed confirmation recorded — plus 023's 24 cue ids and its
    frames' cue ids."""
    if base["cue_token_ids_sha256"] != confirmation_023["exclusion"]["cue_token_ids_sha256"]:
        raise PhaseError("Experiment 023's freeze exclusion does not reproduce from the frozen inputs now")
    ids = set(int(i) for i in base["cue_token_ids"]) | {int(cue["token_id"]) for cue in confirmation_023["cues"]}
    ids |= {int(token_id) for frame in confirmation_023["frames"] for token_id in frame["cue_ids"].values()}
    source = {"source": "confirmation-023", "loader": "json: Experiment 023's committed confirmation file",
              "files": [{"path": b0c.CONFIRMATION_RELATIVE_PATH, "file_sha256": confirmation_023_file_sha256}], "content_sha256": confirmation_023["content_sha256"],
              "tokens": len(confirmation_023["cues"]), "frames": len(confirmation_023["frames"])}
    ordered = sorted(ids)
    return {"cue_token_ids": ordered, "cue_token_ids_sha256": pm.sha256_text(pm.canonical_json(ordered)), "sources": list(base["sources"]) + [source]}


def target_noun_form_ids(pool: Any) -> list[int]:
    return sorted({int(token_id) for noun in pool.nouns for token_id in (*noun.sg_ids, *noun.pl_ids)})


def frame_token_ids(pool: Any) -> list[int]:
    return sorted({int(token_id) for frame in pool.frames for token_id in (*frame.prefix_ids, *frame.suffix_ids)})


def _digested(ids: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(int(i) for i in ids)
    return {"ids": ordered, "sha256": pm.sha256_text(pm.canonical_json(ordered))}


def _status(tokenizer: Any, word: str, blocked: Mapping[str, frozenset[int]], picked: set[int]) -> tuple[int | None, str]:
    ids = pm._encode(tokenizer, " " + word)
    if len(ids) != 1:
        return None, f"{len(ids)} tokens with a leading space"
    token_id = int(ids[0])
    for key, reason in (("exclusion", "already used as a cue by Experiments 005–023"), ("target_forms", "a form of a target noun"),
                        ("frame_tokens", "a token of an exposed frame")):
        if token_id in blocked[key]:
            return None, f"token id {token_id}: {reason}"
    if token_id in picked:
        return None, f"token id {token_id}: already picked"
    return token_id, "eligible"


def select_cues(tokenizer: Any, blocked: Mapping[str, frozenset[int]], config: Configuration) -> dict[str, Any]:
    """The mechanical selection over the whole ordered lists (every entry's status recorded): the first quota eligible
    entries of each, then a shortfall check and the expected-picks check — both before anything is written."""
    quota = config.class_quota
    picked: set[int] = set()
    rejected: list[dict[str, Any]] = []
    words: list[tuple[str, int, int]] = []
    reserves: dict[str, list[str]] = {"N": [], "measure": [], "ordinary": []}
    for rank, word in enumerate(N_CANDIDATES):
        token_id, reason = _status(tokenizer, word, blocked, picked)
        if token_id is None:
            rejected.append({"list": "N", "candidate": word, "rank": rank, "reason": reason})
        elif len(words) < quota:
            words.append((word, token_id, rank))
            picked.add(token_id)
        else:
            reserves["N"].append(word)
    lemmas: dict[str, list[tuple[str, int, str, int, int]]] = {"measure": [], "ordinary": []}
    for name, pairs in (("measure", MEASURE_LEMMAS), ("ordinary", ORDINARY_LEMMAS)):
        for rank, (singular, plural) in enumerate(pairs):
            token_s, reason_s = _status(tokenizer, singular, blocked, picked)
            token_p, reason_p = _status(tokenizer, plural, blocked, picked)
            if token_s is None or token_p is None or token_s == token_p:
                rejected.append({"list": name, "candidate": f"{singular}/{plural}", "rank": rank, "reason": f"{singular}: {reason_s}; {plural}: {reason_p}"})
            elif len(lemmas[name]) < quota:
                lemmas[name].append((singular, token_s, plural, token_p, rank))
                picked |= {token_s, token_p}
            else:
                reserves[name].append(f"{singular}/{plural}")
    for name, count in (("N", len(words)), ("measure", len(lemmas["measure"])), ("ordinary", len(lemmas["ordinary"]))):
        if count < quota:
            raise FreezeShortfall(f"list {name}: {count} eligible of {quota}")
    picks = {"N": tuple(word for word, _, _ in words), "measure": tuple(entry[0] for entry in lemmas["measure"]),
             "ordinary": tuple(entry[0] for entry in lemmas["ordinary"])}
    differing = {key: {"picked": list(picks[key]), "expected": list(config.expected(key))} for key in picks if picks[key] != config.expected(key)}
    if differing:
        raise FreezeDeviation(f"the mechanical picks differ from the design's expected picks: {differing}")
    cues = [{"word": word, "token_id": token_id, "class": "N", "lemma": word, "form": "word", "candidate_rank": rank} for word, token_id, rank in words]
    for cls, name, plural in (("B", "measure", True), ("D", "measure", False), ("C", "ordinary", True), ("E", "ordinary", False)):
        cues += [{"word": p if plural else s, "token_id": tp if plural else ts, "class": cls, "lemma": s, "form": "plural" if plural else "singular",
                  "candidate_rank": rank} for s, ts, p, tp, rank in lemmas[name]]
    return {"cues": cues, "reserves": reserves, "rejected": rejected, "picks": {key: list(value) for key, value in picks.items()}}


def freeze_payload(tokenizer: Any, *, pool: Any, exclusion_base: Mapping[str, Any], confirmation_023: Mapping[str, Any], confirmation_023_file_sha256: str,
                   config: Configuration, model: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The confirmation file's content: the 40 cues, every rule's inputs and the 4,320-key manifest. A shortfall or a
    deviation raises before anything exists to write."""
    excluded = exclusion(exclusion_base, confirmation_023, confirmation_023_file_sha256)
    forms, tokens = target_noun_form_ids(pool), frame_token_ids(pool)
    blocked = {"exclusion": frozenset(excluded["cue_token_ids"]), "target_forms": frozenset(forms), "frame_tokens": frozenset(tokens)}
    selection = select_cues(tokenizer, blocked, config)
    payload = {
        "experiment": EXPERIMENT, "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "kind": "the fresh cues of Experiment 024, frozen from the tokenizer and the text rules alone; no model output", "design": dict(DESIGN), "plan": dict(PLAN),
        "model": dict(model or {"model_id": models_module.PYTHIA_70M.model_id, "revision": models_module.PYTHIA_70M.revision}), "configuration": config.to_json(),
        "classes": dict(CLASS_CONTENT), "candidates": {"N": list(N_CANDIDATES), "measure": [list(pair) for pair in MEASURE_LEMMAS],
                                                        "ordinary": [list(pair) for pair in ORDINARY_LEMMAS]},
        "rules": dict(FREEZE_RULES), "exclusion": excluded, "target_noun_form_ids": _digested(forms), "frame_token_ids": _digested(tokens),
        "reference_cue_ids": {template: int(token_id) for template, token_id in pool.reference_ids.items()}, "exposed_frame_ids": [frame.frame_id for frame in pool.frames],
        "cues": selection["cues"], "reserves": selection["reserves"], "rejected": selection["rejected"], "picks": selection["picks"],
        "expected_picks": config.to_json()["expected_picks"], "picks_match_expected": True,
    }
    confirmation = confirmation_from_payload(payload, pool, config, verify=False)
    payload["counts"] = confirmation.counts()
    payload["manifest"] = confirmation.manifest()
    payload["content_sha256"] = rc.content_digest(payload)
    return payload


@dataclass(frozen=True)
class Confirmation024:
    reference_ids: Mapping[str, int]
    exposed_frames: tuple[pm.Frame, ...]  # the 108 exposed frames, in the pool's order
    tokens: tuple[dict[str, Any], ...]  # the fresh cues in frozen order (class order N, B, D, C, E; list order within a class)
    content_sha256: str
    frames: tuple[pm.Frame, ...] = ()  # no new frame: ``ul.stage_two_022``'s Y2 block is empty by construction

    def class_tokens(self, cls: str) -> tuple[dict[str, Any], ...]:
        return tuple(token for token in self.tokens if token["class"] == cls)

    @property
    def target_prompts(self) -> tuple[pm.Prompt, ...]:
        return tuple(pm.Prompt(frame, int(token["token_id"]), token["word"]) for frame in self.exposed_frames for token in self.tokens)

    @property
    def all_prompts(self) -> tuple[pm.Prompt, ...]:
        return self.target_prompts

    def manifest(self) -> dict[str, Any]:
        return {"S2-TARGET": sorted(prompt.key for prompt in self.target_prompts)}

    def manifest_keys(self) -> frozenset[str]:
        return frozenset(prompt.key for prompt in self.target_prompts)

    def counts(self) -> dict[str, dict[str, int]]:
        return {"classes": {cls: len(self.class_tokens(cls)) for cls in CLASSES}}


def confirmation_from_payload(payload: Mapping[str, Any], pool: Any, config: Configuration, *, verify: bool = True) -> Confirmation024:
    tokens = tuple({"word": entry["word"], "token_id": int(entry["token_id"]), "class": entry["class"], "lemma": entry["lemma"], "form": entry["form"]}
                   for entry in payload["cues"])
    confirmation = Confirmation024(dict(pool.reference_ids), tuple(pool.frames), tokens, str(payload.get("content_sha256", "")))
    if verify:
        if payload.get("experiment") != EXPERIMENT or payload.get("schema_version") != CONFIRMATION_SCHEMA_VERSION:
            raise PhaseError("not an Experiment 024 confirmation file")
        if payload["content_sha256"] != rc.content_digest(payload):
            raise PhaseError("the confirmation file's content digest does not verify")
        if payload.get("configuration") != config.to_json():
            raise PhaseError("the confirmation file was frozen under a different configuration")
        if payload["manifest"] != confirmation.manifest() or payload.get("counts") != confirmation.counts():
            raise PhaseError("the confirmation file's manifest or counts are not the ones its cues define")
        if confirmation.counts() != {"classes": {cls: config.class_quota for cls in CLASSES}} or len({token["token_id"] for token in tokens}) != config.n_fresh:
            raise PhaseError(f"the confirmation file does not hold {config.class_quota} distinct cues per class")
        expected = config.to_json()["expected_picks"]
        if payload.get("picks_match_expected") is not True or payload.get("picks") != expected or payload.get("expected_picks") != expected:
            raise PhaseError("the confirmation file's picks are not the design's expected picks")
        if [token["lemma"] for token in confirmation.class_tokens("N")] != expected["N"] \
                or [token["lemma"] for token in confirmation.class_tokens("D")] != expected["measure"] \
                or [token["lemma"] for token in confirmation.class_tokens("B")] != expected["measure"] \
                or [token["lemma"] for token in confirmation.class_tokens("E")] != expected["ordinary"] \
                or [token["lemma"] for token in confirmation.class_tokens("C")] != expected["ordinary"]:
            raise PhaseError("the confirmation file's classes are not the expected lemmas in order")
        if payload["exposed_frame_ids"] != [frame.frame_id for frame in pool.frames] or dict(payload["reference_cue_ids"]) != {k: int(v) for k, v in pool.reference_ids.items()}:
            raise PhaseError("the confirmation file names a different exposed pool")
    return confirmation


def load_confirmation_024(path: Path, pool: Any, excluded_now: Mapping[str, Any], config: Configuration) -> Confirmation024:
    """The committed freeze artifact, verified: digest, configuration, manifest, composition, the expected picks, and the
    rules' inputs recomputed now — no fresh id used before, a target-noun form or a frame token."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    confirmation = confirmation_from_payload(payload, pool, config)
    if payload["exclusion"]["cue_token_ids_sha256"] != excluded_now["cue_token_ids_sha256"] or payload["exclusion"]["sources"] != excluded_now["sources"]:
        raise PhaseError("the exclusion recorded at the freeze differs from the one the frozen inputs give now")
    forms, tokens = _digested(target_noun_form_ids(pool)), _digested(frame_token_ids(pool))
    if payload["target_noun_form_ids"] != forms or payload["frame_token_ids"] != tokens:
        raise PhaseError("the target-noun forms or the exposed frames' tokens differ from those at the freeze")
    blocked = set(excluded_now["cue_token_ids"]) | set(forms["ids"]) | set(tokens["ids"])
    if {int(token["token_id"]) for token in confirmation.tokens} & blocked:
        raise PhaseError("a fresh cue id is excluded, a target-noun form or an exposed frame's token")
    return confirmation


# ---------------------------------------------------------------------------
# The scientific dependency record: everything ``C``, the measurement, the score and the calibration source are
# computed from. Every record binds it; a later phase refuses any drift.

MEASUREMENT = ("ul.stage_two_022 unchanged: one forward per S2-TARGET prompt through ul.measure_prompt (the measured Δc over the 79 scorable exposed nouns "
               "against 020's locked reference contrasts, Δx1 and Δx3 at the changed positions); C = ul.contrast_of(progs, state, measured Δx3)")


def scientific_dependencies(inputs: ul.FrozenInputs, *, parameters_sha256: str, embedding_sha256: str) -> dict[str, Any]:
    locked = inputs.closure["exploration"]["locked_states"]
    return {"module_blobs": dict(FROZEN_BLOBS), "C": C_DEFINITION, "measurement": MEASUREMENT,
            "readout_020": {"module": "readout_decompilation.py", "blob": FROZEN_BLOBS["readout_decompilation.py"], "exposed_states_sha256": ul.exposed_states_digest(locked),
                            "frozen_input_digests": {key: inputs.digests[key] for key in rc.DIGEST_KEYS}},
            "model": {"model_id": models_module.PYTHIA_70M.model_id, "revision": models_module.PYTHIA_70M.revision, "parameters_sha256": parameters_sha256,
                      "embedding_sha256": embedding_sha256},
            "calibration_source": calibration_source()}


def verify_dependencies(recorded: Mapping[str, Any], now: Mapping[str, Any], what: str) -> None:
    """Any drift in a bound dependency refuses: nothing is silently redefined."""
    differing = sorted(key for key in set(recorded) | set(now) if recorded.get(key) != now.get(key))
    if differing:
        raise PhaseError(f"{what} was written against different scientific dependencies: {differing}")


# ---------------------------------------------------------------------------
# The calibration (exposed only, once; the committed exposed cells and the weights; no forward pass) (Task 4).


class CalibrationStop(RuntimeError):
    """250 or more undefined draws, or a reversed direction: no record is written; stop for review (not an incident)."""

    def __init__(self, details: Mapping[str, Any]):
        self.details = dict(details)
        super().__init__(str(details.get("reason")))


class CrossCheckError(IncidentError):
    """An implementation cross-check failed (the MSE or the Spearman route): an incident, with its location."""

    def __init__(self, details: Mapping[str, Any]):
        self.details = dict(details)
        super().__init__(f"{details.get('check')} cross-check failed: {details.get('max_difference')} at {details.get('at')}")


def read_exposed_cells(root: Path, inputs: ul.FrozenInputs, noun_keys: Sequence[str], record_022: Mapping[str, Any]) -> tuple[torch.Tensor, b0c.ExposedUnits]:
    """023's committed exposed cells: the reviewed files (digests) re-read through 023's own reader, the index verified
    against the meta recomputed from the frozen inputs and 022's committed record now."""
    verify_023_inputs(root)
    units = b0c.exposed_units(inputs)
    data, index = Path(root) / b0c.CELLS_DATA_RELATIVE_PATH, Path(root) / b0c.CELLS_INDEX_RELATIVE_PATH
    cells, loaded = b0c.read_cells(data, index, units=units, meta=b0c.cells_meta(units, noun_keys, record_022))
    if loaded.get("content_sha256") != INHERITED_023["cells_index_content_sha256"] or loaded.get("file_sha256") != INHERITED_023["cells_data_file_sha256"]:
        raise PhaseError("the exposed cells are not the reviewed 023 artifact")
    return cells, units


def exposed_cue_mse(cells: torch.Tensor, units: b0c.ExposedUnits) -> tuple[list[float], dict[str, Any]]:
    """Every exposed cue's MSE from its 108 cue-major rows (frames by ``frame_id``), with the torch cross-check."""
    n_frames = len(units.frames)
    values, worst = [], {"check": "mse", "max_difference": 0.0, "at": ""}
    for ci, (word, _, _) in enumerate(units.cues):
        rows = cells[ci * n_frames:(ci + 1) * n_frames]
        value = cue_mse(rows)
        difference = abs(value - cue_mse_torch(rows)) / abs(value) if value else math.inf
        if difference > worst["max_difference"]:
            worst.update({"max_difference": difference, "at": word})
        values.append(value)
    worst.update({"tolerance": TOLERANCES["mse"], "passed": worst["max_difference"] <= TOLERANCES["mse"]})
    if not worst["passed"]:
        raise CrossCheckError(worst)
    return values, worst


def calibration_population(cells: torch.Tensor, units: b0c.ExposedUnits, W_E: torch.Tensor, bindings: Mapping[str, Any], config: Configuration) -> dict[str, Any]:
    """The 139 calibration cues (leave-one-out score, MSE, log MSE) and, descriptively, the pronoun cues (full-centroid
    score): the sizes are the configuration's, the frames and nouns too."""
    if len(units.frames) != config.n_frames or cells.shape[0] != len(units.cues) * config.n_frames:
        raise PhaseError(f"the exposed cells do not cover {config.n_frames} frames per cue")
    if not bool((cells[:, COUNT] == float(config.n_nouns)).all()):
        raise PhaseError(f"an exposed cell does not count {config.n_nouns} nouns")
    mse, check = exposed_cue_mse(cells, units)
    counts = {stratum: sum(1 for _, _, cls in units.cues if cls == stratum) for stratum in b0c.STRATA}
    if counts != dict(config.calibration_counts):
        raise PhaseError(f"the calibration population is {counts}, not {dict(config.calibration_counts)}")
    index = {int(token_id): ci for ci, (_, token_id, _) in enumerate(units.cues)}
    loo = calibration_scores(W_E, bindings)
    entries = []
    for (word, token_id, stratum), score in zip(calibration_cues(units), loo):
        value = mse[index[token_id]]
        entries.append({"word": word, "token_id": token_id, "stratum": stratum, "nounness_loo": score, "mse": value, "log_mse": log_mse(value)})
    pronoun_units = [(word, int(token_id)) for word, token_id, cls in units.cues if cls == PRONOUN_STRATUM]
    if len(pronoun_units) != config.n_pronoun:
        raise PhaseError(f"{len(pronoun_units)} pronoun cues, not {config.n_pronoun}")
    pronoun_scores = full_scores(W_E, bindings, [token_id for _, token_id in pronoun_units])
    pronouns = [{"word": word, "token_id": token_id, "nounness": score, "mse": mse[index[token_id]], "log_mse": log_mse(mse[index[token_id]])}
                for (word, token_id), score in zip(pronoun_units, pronoun_scores)]
    if any(entry["log_mse"] is None for entry in entries):
        raise PhaseError("an exposed calibration cue has no finite positive MSE")
    return {"calibration": entries, "pronouns": pronouns, "mse_check": check}


def primary_draws(entries: Sequence[Mapping[str, Any]], config: Configuration) -> dict[str, Any]:
    """``B`` SHA-indexed draws of ``n_fresh`` calibration cues with replacement; each draw's Spearman between the
    leave-one-out score and the MSE (ties from repeated cues at the average rank)."""
    x, y = [entry["nounness_loo"] for entry in entries], [entry["mse"] for entry in entries]
    indices = primary_draw_indices(config.draws, config.n_fresh, len(entries))
    values = [spearman([x[i] for i in row], [y[i] for i in row]) for row in indices]
    return {"values": values, "indices": indices}


def null_distribution(config: Configuration) -> dict[str, Any]:
    """``P`` SHA-indexed Fisher–Yates permutations of ``n_fresh`` distinct ranks; each one's Spearman against the
    identity ranking."""
    n = config.n_fresh
    identity = list(range(n))
    digest = hashlib.sha256()
    values = []
    for p in range(config.null_permutations):
        perm = null_permutation(p, n)
        digest.update(bytes(perm) if n < 256 else json.dumps(perm).encode("ascii"))
        values.append(spearman(identity, perm))
    if any(value is None for value in values):
        raise IncidentError("a null permutation's Spearman is undefined")
    return {"values": values, "permutations_sha256": digest.hexdigest()}


def spearman_cross_check(entries: Sequence[Mapping[str, Any]], draws: Mapping[str, Any], config: Configuration) -> dict[str, Any]:
    """The canonical Spearman against the independent route on the first draws and on the whole calibration population."""
    x, y = [entry["nounness_loo"] for entry in entries], [entry["mse"] for entry in entries]
    worst = {"check": "spearman", "max_difference": 0.0, "at": "", "n_checked": 0}
    cases = [(f"draw {b}", [x[i] for i in row], [y[i] for i in row]) for b, row in enumerate(draws["indices"][:config.cross_check_draws])]
    cases.append(("the calibration population", x, y))
    for where, xs, ys in cases:
        difference = spearman_agreement(spearman(xs, ys), spearman_direct(xs, ys))
        worst["n_checked"] += 1
        if difference > worst["max_difference"]:
            worst.update({"max_difference": difference, "at": where})
    worst.update({"tolerance": TOLERANCES["spearman"], "passed": worst["max_difference"] <= TOLERANCES["spearman"]})
    if not worst["passed"]:
        raise CrossCheckError(worst)
    return worst


def evaluate_calibration(entries: Sequence[Mapping[str, Any]], pronouns: Sequence[Mapping[str, Any]], draws: Mapping[str, Any], null: Mapping[str, Any],
                         config: Configuration) -> dict[str, Any]:
    """``F_ρ`` (element ``[lower_rank − 1]``, undefined draws at −∞; the stop at ``lower_rank`` undefined; the direction
    check), ``null₉₇.₅`` (element ``[null_rank − 1]``), the effective threshold, the line and the descriptives."""
    values = draws["values"]
    rank = lower_rank(config.draws)
    undefined = sum(1 for value in values if value is None)
    if undefined >= rank:
        raise CalibrationStop({"reason": f"{undefined} undefined draws reach the stop at {rank}", "undefined": undefined, "stop_at": rank})
    floor = order_statistic(values, rank)
    median = defined_median(values)
    if median is None or not floor <= median:
        raise CalibrationStop({"reason": f"a reversed direction: F_ρ {floor} above the median of the defined draws {median}", "floor": floor, "median": median})
    defined = sorted(value for value in values if value is not None)
    upper = config.draws - rank  # the draw distribution's upper tail element, descriptive
    null_values = sorted(null["values"])
    n_rank = null_rank(config.null_permutations)
    null_bound = null_values[n_rank - 1]
    x, y = [entry["nounness_loo"] for entry in entries], [entry["log_mse"] for entry in entries]
    line = ols(x, y)
    codes = [classify_primary(value, floor, null_bound) for value in values]
    pronoun_errors = [entry["log_mse"] - predict(line, entry["nounness"]) for entry in pronouns if entry["log_mse"] is not None]
    within = {stratum: spearman([e["nounness_loo"] for e in entries if e["stratum"] == stratum], [e["mse"] for e in entries if e["stratum"] == stratum])
              for stratum in b0c.STRATA}
    return {
        "primary_floor": {"draws": config.draws, "tag": PRIMARY_TAG, "rank": rank, "element": rank - 1, "F_rho": floor, "undefined": undefined, "stop_at": rank,
                          "direction_check": {"ok": True, "median": median},
                          "tails": {"min": defined[0], "element_lower": floor, "median": median, "element_upper": sorted(values, key=lambda v: -math.inf if v is None else v)[upper],
                                    "max": defined[-1]}},
        "null": {"permutations": config.null_permutations, "n": config.n_fresh, "tag": NULL_TAG, "rank": n_rank, "element": n_rank - 1, "null_975": null_bound,
                 "median": _median(null_values), "permutations_sha256": null["permutations_sha256"]},
        "effective_threshold": {"value": max(floor, null_bound), "binds": "null_975" if null_bound >= floor else "F_rho"},
        "line": line,
        "descriptive": {"calibration_rho": spearman(x, [entry["mse"] for entry in entries]), "within_stratum_rho": within,
                        "pronoun_check": {"n": len(pronouns), "rho": spearman([e["nounness"] for e in pronouns], [e["mse"] for e in pronouns]),
                                          "median_abs_log_error": _median([abs(v) for v in pronoun_errors]), "mean_signed_log_error": math.fsum(pronoun_errors) / len(pronoun_errors)},
                        "draw_rates": {name: codes.count(name) / len(codes) for name in PRIMARY_RESULTS},
                        "maximum_calibration_score": max(x)},
    }


def _median(values: Sequence[float]) -> float | None:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def record_constants(config: Configuration) -> dict[str, Any]:
    return {"configuration": config.to_json(), "tags": {"primary": PRIMARY_TAG, "null": NULL_TAG, "contrast": CONTRAST_TAG},
            "ranks": {"primary": lower_rank(config.draws), "null": null_rank(config.null_permutations), "contrast_lower": lower_rank(config.contrast_resamples),
                      "contrast_upper_element": config.contrast_resamples - lower_rank(config.contrast_resamples)},
            "tolerances": dict(TOLERANCES), "statistic": STATISTIC, "classes": dict(CLASS_CONTENT), "results": {"primary": list(PRIMARY_RESULTS), "guard": list(GUARD_RESULTS),
                                                                                                             "outcomes": list(OUTCOMES)}}


def calibration_arrays(draws: Mapping[str, Any], null: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {"draw_rho": torch.tensor([math.nan if v is None else v for v in draws["values"]], dtype=torch.float64),
            "draw_defined": torch.tensor([v is not None for v in draws["values"]], dtype=torch.bool),
            "draw_indices": torch.tensor(draws["indices"], dtype=torch.int64), "null_rho": torch.tensor(null["values"], dtype=torch.float64)}


def arrays_digests(arrays: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {key: rc.tensor_digest(value.to(torch.int64) if value.dtype == torch.bool else value) for key, value in sorted(arrays.items())}


def calibration_record(*, run_id: str, protocol_code_commit: str, digests: Mapping[str, str], config: Configuration, confirmation: Mapping[str, str],
                       bindings: Mapping[str, Any], dependencies: Mapping[str, Any], population: Mapping[str, Any], evaluated: Mapping[str, Any],
                       checks: Mapping[str, Any], array_digests: Mapping[str, str]) -> dict[str, Any]:
    record = {
        "experiment": EXPERIMENT, "schema_version": 1, "kind": "the exposed-only calibration record of Experiment 024 (design revision 2), from 023's committed exposed "
                                                                "cells and the weights; no forward pass",
        "design": dict(DESIGN), "plan": dict(PLAN), "run_id": run_id, "protocol_code_commit": protocol_code_commit, "inputs": dict(digests),
        "module_blobs": dict(FROZEN_BLOBS), "constants": record_constants(config), "configuration": config.to_json(), "exposed_cells": calibration_source(),
        "confirmation_024": dict(confirmation), "dependencies": dict(dependencies),
        "score": {key: value for key, value in bindings.items()}, "calibration_cues": list(population["calibration"]), "pronoun_cues": list(population["pronouns"]),
        "line": dict(evaluated["line"]), "primary_floor": dict(evaluated["primary_floor"]), "null": dict(evaluated["null"]),
        "effective_threshold": dict(evaluated["effective_threshold"]), "checks": dict(checks), "descriptive": dict(evaluated["descriptive"]),
        "arrays_sha256": dict(array_digests),
    }
    record = rc.json_safe(record)
    validate_json_safe(record)
    record["content_sha256"] = rc.content_digest(record)
    return record


def verify_calibration_record(record: Mapping[str, Any], config: Configuration) -> None:
    """A record fit for the lock: its digest; the frozen constants, configuration, design, plan and modules; finite
    thresholds at the frozen elements with a passed direction check and no stop; the calibration population's size; a
    finite line recomputed exactly from its own 139 entries."""
    if record.get("experiment") != EXPERIMENT or record.get("content_sha256") != rc.content_digest(record):
        raise PhaseError("not a verified Experiment 024 calibration record")
    if record["constants"] != record_constants(config) or record["configuration"] != config.to_json() or record["design"] != DESIGN or record["plan"] != PLAN:
        raise PhaseError("the calibration record's constants, configuration, design or plan are not the frozen ones")
    if record["module_blobs"] != FROZEN_BLOBS or record["exposed_cells"] != calibration_source() or record["dependencies"]["calibration_source"] != calibration_source():
        raise PhaseError("the calibration record was written against different modules or a different calibration source")
    floor, null = record["primary_floor"], record["null"]
    if (floor.get("rank"), floor.get("element")) != (lower_rank(config.draws), lower_rank(config.draws) - 1) or not (floor.get("direction_check") or {}).get("ok"):
        raise PhaseError("the calibration record's F_ρ is not the frozen order statistic with a passed direction check")
    if int(floor.get("undefined", config.draws)) >= lower_rank(config.draws):
        raise PhaseError("the calibration record's undefined draws reach the stop")
    if (null.get("rank"), null.get("element")) != (null_rank(config.null_permutations), null_rank(config.null_permutations) - 1):
        raise PhaseError("the calibration record's null is not the frozen order statistic")
    for value in (floor.get("F_rho"), null.get("null_975"), record["line"].get("slope"), record["line"].get("intercept")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PhaseError("the calibration record carries a non-finite threshold or line")
    entries = record["calibration_cues"]
    if len(entries) != config.n_calibration or [entry["token_id"] for entry in entries] != record["score"]["calibration_cue_ids"]:
        raise PhaseError("the calibration record's population is not the bound calibration cues")
    again = ols([entry["nounness_loo"] for entry in entries], [entry["log_mse"] for entry in entries])
    if {key: again[key] for key in ("slope", "intercept", "residual_sd", "n")} != {key: record["line"][key] for key in ("slope", "intercept", "residual_sd", "n")}:
        raise PhaseError("the calibration record's line does not recompute exactly from its own entries")
    if record["effective_threshold"] != {"value": max(floor["F_rho"], null["null_975"]), "binds": "null_975" if null["null_975"] >= floor["F_rho"] else "F_rho"}:
        raise PhaseError("the calibration record's effective threshold is not max(F_ρ, null₉₇.₅)")
