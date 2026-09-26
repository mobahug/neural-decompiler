"""Experiment 025: does moving the cue embedding along the frozen 024 nounness direction causally change the frozen
readout error and the attention routing?

Implements design revision 1 (``c0885e5``, corrected in ``26c9925``) through implementation plan revision 1
(``7d90d28``), with the reviewer's sign-off on the confirm-time patch-path check.

**The intervention.** For each of 40 fresh cues, a norm-preserving rotation of the cue's input embedding at
``("EMBED", p_c)``, in the plane of ``Ê`` and the frozen 024 score's gradient ``t̂``::

    d = μ̂_noun − μ̂_cue (024's calibration record)      s₀ = d·Ê      t = d − s₀·Ê      τ = |t|
    R(±θ) = |E|·(cos θ·Ê ± sin θ·t̂)                      θ = asin(odd/τ)

The directional (odd) score component is exactly ``±odd``: 0.32 primary, 0.16 secondary. The complete change is
``s₀·(cos θ − 1) ± odd``. There are seven deterministic random tangent controls per cue: ``u ⟂ Ê, t̂``, at the same
angle, and nounness-neutral because ``d·u = 0``. A plurality control is secondary.

**The outcome-bearing statistics.** They are computed per cue from two quantities. The first is the frozen-readout
discrepancy ``ℓ = log(Σ SSE_C / Σ n)`` over the 108 exposed frames × 79 nouns. The second is the direct L4/L5 routing
summary ``D_attn``::

    A_i = ½[ℓ(+θ) − ℓ(−θ)]      B_i = A_i − mean_j |A_ij|      G_i = ½[D_attn(+θ) − D_attn(−θ)]

Each is a pre-registered sign-count criterion: at least 27 of 40 strictly positive, with ties and zeros counting
against. The reference tail is exact Binomial(40, 0.5) under H0: P(positive) ≤ 0.5 for independent cue units. It is
not a design-based randomization test; the causal reading comes from the within-cue ±θ runs.

The module *calls* Experiment 024's ``readout_routing``, and through it 022's and 023's modules and the ten modules 022
pinned. All are pinned by git blob, and the module never edits them.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
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
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul
from neural_decompiler.behavior import validate_json_safe

PhaseError = pm.PhaseError
IncidentError = pm.IncidentError

# ---------------------------------------------------------------------------
# Paths, design, plan.

EXPERIMENT = "025"
EXPERIMENT_DIR = "experiments/025-nounness-direction-intervention"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREREGISTRATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration.md"
DESIGN = {"path": "docs/superpowers/specs/2026-09-26-experiment-025-nounness-direction-intervention-design.md", "revision": 1, "commit": "c0885e5",
          "correction": "26c9925"}
PLAN = {"path": "docs/superpowers/plans/2026-09-26-experiment-025-nounness-direction-intervention-plan.md", "revision": 1, "commit": "7d90d28"}

# ---------------------------------------------------------------------------
# The frozen dependencies, by git blob (sha1(b"blob <len>\0" + bytes), computed without git). They are written out
# literally, never inherited: the ten modules 022 pinned, 022's own module, 023's, and 024's.

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
    "readout_routing.py": "e6cb37767d1d06c6ff40804a88eab569723afdb5",
}
_FROZEN_MODULES = {"readout_decompilation.py": rd, "readout_calibration.py": rc, "head_pattern.py": hp, "frame_channels.py": fch, "attention_patterns.py": atp,
                   "layer_correction.py": lc, "plural_mechanism.py": pm, "encoding_read.py": er, "head_transport.py": ht, "models.py": models_module,
                   "upstream_localization.py": ul, "block0_completion.py": b0c, "readout_routing.py": rr}

# Experiment 024's committed, independently reviewed artifacts that 025 reads: the calibration record (the direction's
# id lists and centroid digests), the lock (024's scores, bound for the 0.32's provenance) and the freeze (the earlier
# cues, the exposed frames, the reference cues and 024's manifest). Each file's sha256 and content digest.
INHERITED_024_PATHS = {"calibration": rr.CALIBRATION_RELATIVE_PATH, "lock": rr.LOCK_RELATIVE_PATH, "confirmation": rr.CONFIRMATION_RELATIVE_PATH}
INHERITED_024 = {
    "calibration_file_sha256": "81fb499edf022ae6430beb2be1dbb36584a205ad3483d727ce98393643f2738d",
    "calibration_content_sha256": "09bf093ed2b961a935fc1728e2f7a71ef1373336f60dbdf7c8357e558fcfbce8",
    "lock_file_sha256": "5c2a9b903803536d75a453f00b5da5bcf88c3298fd86193bda5525bc4c88798d",
    "lock_content_sha256": "a39668bfa75e693a7f886b41aeb4bc2c0787ba46f02cc8bd2fdf3d4dd4fc0739",
    "confirmation_file_sha256": "68510e1b902b5ee3442caf9c7bbbd337abe5bbbf04766d8bfa98c09e4b199b60",
    "confirmation_content_sha256": "87f8aff1dca467f92a905b115681a0f6f80328c0f872c426d231b67d129b1e87",
}
# The tightened prior-noun rule's source: Experiment 020's committed confirmation file. Its 24 nouns were read as fresh
# target nouns by Experiment 021; 021's noun ledger does not record them.
PRIOR_NOUNS_020 = {"path": rd.CONFIRMATION_RELATIVE_PATH, "file_sha256": "de4ebb7b25eb3022a72ad202a623bd62fafc80268a9178250cb94f7ea96a38b9", "nouns": 24}

# ---------------------------------------------------------------------------
# The population (design revision 1): ordered lists, verbatim. The reserves are 024's committed freeze reserves; a tier-A
# test checks the literal lists against that file.

ADJECTIVE_RESERVES = ("anxious", "cheerful", "curious", "jealous", "lonely", "nasty", "careful", "careless", "famous", "friendly", "gorgeous", "hungry", "weary",
                      "wicked", "ugly", "vivid", "vague", "rapid", "rigid", "clever", "fuzzy", "latest", "earliest")
ORDINARY_RESERVES = (("soldier", "soldiers"), ("sailor", "sailors"), ("priest", "priests"), ("knight", "knights"), ("onion", "onions"), ("carrot", "carrots"),
                     ("pirate", "pirates"), ("tourist", "tourists"), ("statue", "statues"))
NEW_NOUN_LIST = tuple("author baker bishop camel carpet clerk dancer dolphin donkey duck eagle frog goat guitar hammer hunter ladder lawyer mirror monk nurse owl "
                      "painter parrot pencil pillow planet prince queen robot rocket shark singer snake tiger tractor violin whale wizard".split())
EXPECTED_PICKS = {
    "adjective": ("anxious", "cheerful", "curious", "jealous", "lonely", "nasty", "careful", "careless", "famous", "friendly", "gorgeous", "hungry", "weary",
                  "wicked", "ugly", "vivid", "vague", "rapid", "rigid", "clever"),
    "noun": ("soldier", "sailor", "priest", "knight", "onion", "carrot", "pirate", "tourist", "author", "bishop", "dancer", "duck", "goat", "guitar", "hunter",
             "lawyer", "monk", "nurse", "painter", "prince"),
}
STRATA = ("adjective", "noun")


def regular_plural(word: str) -> str:
    """The regular plural the new-list rule checks: ``+es`` after s, x, sh or ch; ``+s`` otherwise."""
    return word + ("es" if word.endswith(("s", "x", "sh", "ch")) else "s")


# ---------------------------------------------------------------------------
# The configuration. Production is a literal, immutable object; a test world may use a smaller one only by passing it
# explicitly (the runner's ``config``), never by patching these.

ALPHA = Fraction(1, 40)  # 0.025, one-sided


def binomial_tail(n: int, k: int) -> Fraction:
    """``P(X ≥ k)`` for ``X ~ Binomial(n, 1/2)``, as an exact rational."""
    return Fraction(sum(math.comb(int(n), j) for j in range(int(k), int(n) + 1)), 2 ** int(n))


def derive_threshold(n: int, alpha: Fraction = ALPHA) -> int:
    """The smallest positive count whose Binomial(n, 1/2) upper tail is at most ``alpha``: 27 of 40."""
    for k in range(int(n) + 1):
        if binomial_tail(n, k) <= alpha:
            return k
    return int(n) + 1


def _dose(value: float) -> str:
    return f"{float(value):.2f}"


@dataclass(frozen=True)
class Configuration:
    """Every size the protocol fixes. Every artifact records the configuration it was written under; one written under
    one configuration is refused under another."""

    name: str
    n_adjectives: int
    n_nouns: int
    k_controls: int
    primary_odd: float
    half_odd: float
    count_threshold: int
    n_frames: int
    n_scored_nouns: int
    expected_picks: tuple[tuple[str, tuple[str, ...]], ...]  # "adjective" / "noun" -> the design's picks, in order

    def __post_init__(self) -> None:
        if tuple(key for key, _ in self.expected_picks) != STRATA:
            raise ValueError("expected picks: adjective then noun")
        if len(self.expected("adjective")) != self.n_adjectives or len(self.expected("noun")) != self.n_nouns:
            raise ValueError("the expected picks do not match the quotas")
        if not 1 <= self.count_threshold <= self.n_cues or self.k_controls < 1:
            raise ValueError("inconsistent count threshold or control count")
        if not 0.0 < self.half_odd < self.primary_odd < 1.0:
            raise ValueError("the doses must satisfy 0 < half < primary < 1")
        if self.name == "production" and self.count_threshold != derive_threshold(self.n_cues):
            raise ValueError("the production threshold must be the smallest count with a Binomial upper tail ≤ 0.025")

    def expected(self, key: str) -> tuple[str, ...]:
        return dict(self.expected_picks)[key]

    @property
    def n_cues(self) -> int:
        return self.n_adjectives + self.n_nouns

    @property
    def primary(self) -> str:
        return _dose(self.primary_odd)

    @property
    def half(self) -> str:
        return _dose(self.half_odd)

    @property
    def conditions(self) -> tuple[str, ...]:
        p, h = self.primary, self.half
        controls = tuple(f"rand{j}{sign}{p}" for j in range(1, self.k_controls + 1) for sign in "+-")
        return ("base", f"noun+{p}", f"noun-{p}", f"noun+{h}", f"noun-{h}", *controls, f"plur+{p}", f"plur-{p}")

    @property
    def outcome_bearing(self) -> tuple[str, ...]:
        p = self.primary
        return (f"noun+{p}", f"noun-{p}", *(f"rand{j}{sign}{p}" for j in range(1, self.k_controls + 1) for sign in "+-"))

    def spec(self, condition: str) -> dict[str, Any]:
        """``kind`` (base, noun, rand, plur), ``sign`` (+1, −1, 0), ``odd`` (the directional component) and ``j``."""
        if condition == "base":
            return {"kind": "base", "sign": 0, "odd": 0.0, "j": None}
        for kind in ("noun", "plur"):
            for dose in ((self.primary_odd, self.half_odd) if kind == "noun" else (self.primary_odd,)):
                for sign, symbol in ((1, "+"), (-1, "-")):
                    if condition == f"{kind}{symbol}{_dose(dose)}":
                        return {"kind": kind, "sign": sign, "odd": float(dose), "j": None}
        for j in range(1, self.k_controls + 1):
            for sign, symbol in ((1, "+"), (-1, "-")):
                if condition == f"rand{j}{symbol}{self.primary}":
                    return {"kind": "rand", "sign": sign, "odd": float(self.primary_odd), "j": j}
        raise ValueError(f"unknown condition {condition!r}")

    def reference_tail(self) -> Fraction:
        return binomial_tail(self.n_cues, self.count_threshold)

    def to_json(self) -> dict[str, Any]:
        tail = self.reference_tail()
        return {"name": self.name, "n_adjectives": self.n_adjectives, "n_nouns": self.n_nouns, "n_cues": self.n_cues, "k_controls": self.k_controls,
                "primary_odd": self.primary_odd, "half_odd": self.half_odd, "count_threshold": self.count_threshold,
                "reference_tail": {"exact": f"{tail.numerator}/{tail.denominator}", "value": float(tail)}, "n_frames": self.n_frames,
                "n_scored_nouns": self.n_scored_nouns, "conditions": list(self.conditions), "outcome_bearing": list(self.outcome_bearing),
                "expected_picks": {key: list(words) for key, words in self.expected_picks}}


PRODUCTION = Configuration(name="production", n_adjectives=20, n_nouns=20, k_controls=7, primary_odd=0.32, half_odd=0.16, count_threshold=27, n_frames=108,
                           n_scored_nouns=79, expected_picks=tuple((key, tuple(words)) for key, words in EXPECTED_PICKS.items()))

# ---------------------------------------------------------------------------
# Tags, tolerances, spent keys, outcomes and the frozen wording.

CONTROL_TAG = "025|control|{token_id}|{j}"
TOLERANCES = {
    "geometry": 1e-12,  # float64 unit norms, orthogonality, d·u, neutrality, the odd component, length
    "angle": 1e-12,  # float64, radians
    "patched": 1e-6,  # the float32-patched vectors: neutrality, the odd component, length
    "angle_patched": 1e-6,  # radians
    "level1": 2e-2,  # the Level-1 identity on every outcome-bearing primary-dose run (a gate)
    "I1": 1e-4,
    "I3": 1e-4,  # relative
    "I4": 1e-3,
    "C_recompute": 0.0,  # C recomputed from the saved Δx3: bit for bit
    "I7": 0.0,  # the lock's geometry recomputed before any prompt: bit for bit
}
# The confirm-time patch-path check (signed off by the reviewer): four already-executed keys, one per stratum and
# template (every key is in Experiment 020's ledger), compared plain against a patched θ = 0 run, bit for bit.
PATCH_PATH_SPENT_KEYS = ("cardinal-009-1|an|271", "quantifier-009-1|least|1878", "coordinated-adjective-009-1|black|2806", "quantifier-new-2|he|344")

OUTCOMES = ("NOT_INTERPRETABLE", "CAUSAL_EFFECT_NOT_ESTABLISHED", "DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC", "READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED",
            "NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR")
OUTCOME_TABLE = (
    ("any failed validity gate or incident (no result is outcome-bearing)", "NOT_INTERPRETABLE"),
    ("A fails", "CAUSAL_EFFECT_NOT_ESTABLISHED"),
    ("A passes, B fails", "DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC"),
    ("A and B pass, G fails", "READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED"),
    ("A, B and G pass", "NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR"),
)
SEMANTICS = {
    "criterion": "pre-registered sign-count criteria with an exact Binomial(40, 0.5) reference tail under H0: P(positive) <= 0.5 for independent cue units; "
                 "not unconditional, design-based randomization tests (the 40 lexical items are selected mechanically, not sampled at random); the causal "
                 "reading comes from the within-cue intervention (+θ and −θ are both run for every cue), and the sign count measures consistency across the "
                 "frozen test population; ties and zeros count against PASS",
    "A": "A_i > 0: the +nounness intervention produced greater frozen-readout approximation error than the matched −nounness intervention; A alone does not "
         "establish that +θ rises above the unperturbed baseline or that −θ falls below it (the baseline and the even component are descriptive only)",
    "B": "B_i = A_i − mean_j |A_ij|: the nounness-direction effect in the predicted sign exceeds the typical size of an equal-angle, nounness-neutral "
         "directional effect; absolute values because the signed random mean has expectation 0 by the u → −u symmetry; t̂ is not assumed exchangeable "
         "with the controls",
    "G": "G_i = ½[D_attn(+θ) − D_attn(−θ)], D_attn the total-variation distance of the captured L4/L5 rows at p_t from the locked 020 reference rows, "
         "equally averaged over the frozen 16 heads, then over the 108 frames; equal weighting is frozen before outcomes and avoids outcome-based head "
         "selection; irrelevant or oppositely responding heads can dilute or oppose the aggregate signal, so failure of G is conservative with respect to "
         "this particular routing summary",
    "outcomes": {
        "NOT_INTERPRETABLE": "a validity gate failed or an incident was recorded; no statistic is outcome-bearing",
        "CAUSAL_EFFECT_NOT_ESTABLISHED": "A failed: the +nounness intervention did not consistently produce greater frozen-readout approximation error than the "
                                         "−nounness intervention",
        "DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC": "A passed but B failed: the direction moves the error, but not more than matched nounness-neutral equal-angle "
                                                  "rotations typically do",
        "READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED": "A and B passed, G failed: controlled movement along the frozen nounness-related direction causally "
                                                        "changes the frozen block-4/5 readout approximation error (+ against −), more than matched generic "
                                                        "rotations; no direct rerouting is claimed",
        "NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR": "controlled movement of the cue embedding along the frozen weight-derived "
                                                                        "nounness-related direction causally changes the frozen block-4/5 readout "
                                                                        "approximation error in the predicted direction (+ against −), more strongly than "
                                                                        "matched nounness-neutral rotations, and produces a directional change in the "
                                                                        "pre-registered L4/L5 attention-routing summary",
    },
    "not_claimed": ["semantic nounhood as the model's variable", "a general routing controller", "algorithm selection in general",
                    "generalization beyond this checkpoint and this mechanism",
                    "that the effect excludes a plurality-direction contribution (the nounness direction is partly aligned with plurality, cos +0.235)"],
    "strata": "the positive counts of A, B and G are reported for the 20 adjectives and the 20 ordinary singular nouns separately; they cannot change the "
              "label; if a global PASS is strongly concentrated in one stratum the final claim must say so (reporting trigger: a stratum below 14 of 20)",
    "secondary": "the magnitudes, the even components and baselines, the half dose, the plurality control, nMSE, the ladder, G against the random "
                 "directions, the per-template counts and the causal fraction are computed after the result is written and can never rescue, alter or "
                 "redefine it",
    "incidents": "an incident carries no result; an identity incident never coexists with an outcome",
}

# ---------------------------------------------------------------------------
# Frozen dependencies and inherited inputs.


def module_blobs() -> dict[str, str]:
    return {name: rc.program_blob_sha1(Path(module.__file__)) for name, module in _FROZEN_MODULES.items()}


def assert_frozen_blobs() -> dict[str, str]:
    """Every pinned module (022's, 023's and 024's included) is byte for byte the pinned one."""
    actual = module_blobs()
    differing = sorted(name for name, blob in FROZEN_BLOBS.items() if actual.get(name) != blob)
    if differing:
        raise PhaseError(f"frozen modules changed: {differing}; Experiment 025 runs the pinned programs only")
    return actual


def own_blob() -> str:
    return rc.program_blob_sha1(Path(__file__))


def verify_024_inputs(root: Path) -> dict[str, str]:
    """024's committed calibration record, lock and freeze: each the reviewed file, verifying against its content digest."""
    out: dict[str, str] = {}
    for kind, relative in INHERITED_024_PATHS.items():
        path = Path(root) / relative
        if not path.exists() or rc.file_sha256(path) != INHERITED_024[f"{kind}_file_sha256"]:
            raise PhaseError(f"Experiment 024's committed {kind} file {relative} is missing or not the reviewed file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("content_sha256") != rc.content_digest(payload) or payload["content_sha256"] != INHERITED_024[f"{kind}_content_sha256"]:
            raise PhaseError(f"Experiment 024's {kind} file does not verify against its reviewed content digest")
        out[f"024_{kind}_file"] = INHERITED_024[f"{kind}_file_sha256"]
        out[f"024_{kind}_content"] = INHERITED_024[f"{kind}_content_sha256"]
    return out


def load_024(root: Path) -> dict[str, dict[str, Any]]:
    verify_024_inputs(root)
    return {kind: json.loads((Path(root) / relative).read_text(encoding="utf-8")) for kind, relative in INHERITED_024_PATHS.items()}


def prior_nouns(root: Path) -> dict[str, Any]:
    """The tightened prior-noun rule's ids: every singular and plural token id of the nouns in 020's committed
    confirmation file."""
    path = Path(root) / PRIOR_NOUNS_020["path"]
    if not path.exists() or rc.file_sha256(path) != PRIOR_NOUNS_020["file_sha256"]:
        raise PhaseError(f"Experiment 020's committed confirmation file {PRIOR_NOUNS_020['path']} is missing or not the pinned file")
    nouns = json.loads(path.read_text(encoding="utf-8"))["nouns"]
    if len(nouns) != PRIOR_NOUNS_020["nouns"]:
        raise PhaseError(f"Experiment 020's confirmation file lists {len(nouns)} nouns, not {PRIOR_NOUNS_020['nouns']}")
    ids = sorted({int(token_id) for noun in nouns for token_id in (*noun["sg_ids"], *noun["pl_ids"])})
    return {"path": PRIOR_NOUNS_020["path"], "file_sha256": PRIOR_NOUNS_020["file_sha256"], "nouns": sorted(noun["lexical_key"] for noun in nouns),
            "ids": ids, "sha256": pm.sha256_text(pm.canonical_json(ids))}


DIGEST_KEYS = (*rr.DIGEST_KEYS, "024_calibration_file", "024_calibration_content", "024_lock_file", "024_lock_content", "024_confirmation_file",
               "024_confirmation_content", "020_prior_nouns_file")


def base_digests(inputs: ul.FrozenInputs, digests_022: Mapping[str, str], digests_023: Mapping[str, str], digests_024: Mapping[str, str],
                 prior: Mapping[str, Any]) -> dict[str, str]:
    return {**rr.base_digests(inputs, digests_022, digests_023), **dict(digests_024), "020_prior_nouns_file": prior["file_sha256"]}


# ---------------------------------------------------------------------------
# 1. Geometry (float64; weights only).


def unit(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.double()
    return vector / torch.linalg.vector_norm(vector)


def centroids(W_E: torch.Tensor, record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """μ_noun and μ_cue exactly as 024 computed them (``rr.centroid``), from the record's id lists."""
    return rr.centroid(W_E, record["score"]["noun_row_ids"]), rr.centroid(W_E, record["score"]["calibration_cue_ids"])


def direction(W_E: torch.Tensor, record: Mapping[str, Any]) -> torch.Tensor:
    """``d = μ̂_noun − μ̂_cue``: the frozen 024 direction. The centroids must reproduce the record's digests."""
    mu_noun, mu_cue = centroids(W_E, record)
    if rc.tensor_digest(mu_noun) != record["score"]["mu_noun_sha256"] or rc.tensor_digest(mu_cue) != record["score"]["mu_cue_sha256"]:
        raise IncidentError("the centroids do not reproduce 024's calibration record")
    return unit(mu_noun) - unit(mu_cue)


def score(d: torch.Tensor, vector: torch.Tensor) -> float:
    """024's score with the full centroids: ``d·v̂``."""
    return float(torch.dot(d, unit(vector)))


def angle(a: torch.Tensor, b: torch.Tensor) -> float:
    """The angle between two vectors, ``2·atan2(|â − b̂|, |â + b̂|)``, well conditioned at every angle."""
    ua, ub = unit(a), unit(b)
    return 2.0 * math.atan2(float(torch.linalg.vector_norm(ua - ub)), float(torch.linalg.vector_norm(ua + ub)))


@dataclass(frozen=True)
class CueGeometry:
    token_id: int
    E: torch.Tensor  # the float64 embedding row (of the model's float32 row)
    norm: float
    E_hat: torch.Tensor
    s0: float
    t_hat: torch.Tensor
    tau: float


def cue_geometry(W_E: torch.Tensor, token_id: int, d: torch.Tensor) -> CueGeometry:
    E = W_E[int(token_id)].double()
    norm = float(torch.linalg.vector_norm(E))
    E_hat = E / norm
    s0 = float(torch.dot(d, E_hat))
    t = d - s0 * E_hat
    tau = float(torch.linalg.vector_norm(t))
    return CueGeometry(int(token_id), E, norm, E_hat, s0, t / tau, tau)


def theta_for(odd: float, tau: float) -> float:
    """The rotation angle whose directional score component is ``odd``: ``τ·sin θ = odd``."""
    if not 0.0 <= float(odd) < float(tau):
        raise IncidentError(f"the odd component {odd} is not reachable with τ = {tau}")
    return math.asin(float(odd) / float(tau))


def rotate(geometry: CueGeometry, direction_unit: torch.Tensor, theta: float) -> torch.Tensor:
    """``|E|·(cos θ·Ê + sin θ·u)``: the geodesic rotation of the embedding, its length preserved."""
    return geometry.norm * (math.cos(theta) * geometry.E_hat + math.sin(theta) * direction_unit)


def sha_uniforms(tag: str, count: int) -> list[float]:
    """``count`` uniforms in (0, 1] from a SHA-256 counter-mode stream: each 32-byte block gives four 64-bit big-endian
    integers, each cut to 53 bits; ``(v + 1) / 2⁵³``."""
    out: list[float] = []
    counter = 0
    while len(out) < count:
        block = hashlib.sha256(f"{tag}|{counter}".encode("utf-8")).digest()
        for k in range(4):
            value = int.from_bytes(block[8 * k: 8 * k + 8], "big") >> 11
            out.append((value + 1) / 2.0 ** 53)
        counter += 1
    return out[:count]


def sha_gaussians(tag: str, count: int) -> torch.Tensor:
    """``count`` standard Gaussians by Box–Muller in float64 from ``sha_uniforms``."""
    pairs = (count + 1) // 2
    u = sha_uniforms(tag, 2 * pairs)
    values: list[float] = []
    for i in range(pairs):
        radius = math.sqrt(-2.0 * math.log(u[2 * i]))
        phase = 2.0 * math.pi * u[2 * i + 1]
        values += [radius * math.cos(phase), radius * math.sin(phase)]
    return torch.tensor(values[:count], dtype=torch.float64)


def project_off(vector: torch.Tensor, basis: Sequence[torch.Tensor], passes: int = 2) -> torch.Tensor:
    """Remove the components along orthonormal ``basis`` vectors (Gram–Schmidt, repeated for numerical cleanliness)."""
    vector = vector.double()
    for _ in range(passes):
        for b in basis:
            vector = vector - torch.dot(vector, b) * b
    return vector


def control_directions(geometry: CueGeometry, k: int) -> list[torch.Tensor]:
    """The ``k`` deterministic random tangent controls: SHA-seeded Gaussians projected off ``Ê`` and ``t̂``, unit."""
    dim = int(geometry.E.shape[0])
    return [unit(project_off(sha_gaussians(CONTROL_TAG.format(token_id=geometry.token_id, j=j), dim), (geometry.E_hat, geometry.t_hat)))
            for j in range(1, int(k) + 1)]


def plurality_direction(W_E: torch.Tensor, pool: Any) -> torch.Tensor:
    """``p̂``: the unit mean of plural minus singular rows over the scorable pool nouns (each noun's own singular and
    plural: the pairing the design correction fixed)."""
    nouns = [noun for noun in pool.nouns if noun.single_token]
    singular = torch.stack([W_E[int(noun.sg_ids[0])].double() for noun in nouns]).mean(0)
    plural = torch.stack([W_E[int(noun.pl_ids[0])].double() for noun in nouns]).mean(0)
    return unit(plural - singular)


def plurality_tangent(p_hat: torch.Tensor, geometry: CueGeometry) -> torch.Tensor:
    """``p̂′``: ``p̂``'s tangent part at ``Ê``, orthogonalized to ``t̂`` and normalized (nounness-neutral)."""
    return unit(project_off(p_hat, (geometry.E_hat, geometry.t_hat)))


def condition_directions(geometry: CueGeometry, controls: Sequence[torch.Tensor], p_prime: torch.Tensor, config: Configuration) -> dict[str, tuple[torch.Tensor, float]]:
    """Every condition's (signed unit direction, angle)."""
    out: dict[str, tuple[torch.Tensor, float]] = {}
    for condition in config.conditions:
        spec = config.spec(condition)
        if spec["kind"] == "base":
            out[condition] = (geometry.t_hat, 0.0)
            continue
        base = {"noun": geometry.t_hat, "plur": p_prime}.get(spec["kind"]) if spec["kind"] != "rand" else controls[spec["j"] - 1]
        theta = theta_for(spec["odd"] if spec["kind"] == "noun" else config.primary_odd, geometry.tau)
        out[condition] = (spec["sign"] * base, theta)
    return out


def condition_vectors(geometry: CueGeometry, controls: Sequence[torch.Tensor], p_prime: torch.Tensor, config: Configuration) -> dict[str, torch.Tensor]:
    """The 21 float64 vectors of one cue, in the frozen condition order."""
    return {condition: rotate(geometry, direction_unit, theta) for condition, (direction_unit, theta) in
            condition_directions(geometry, controls, p_prime, config).items()}


def geometry_checks(d: torch.Tensor, geometry: CueGeometry, controls: Sequence[torch.Tensor], p_prime: torch.Tensor, vectors64: Mapping[str, torch.Tensor],
                    vectors32: Mapping[str, torch.Tensor], model_row32: torch.Tensor, config: Configuration) -> dict[str, Any]:
    """I7′'s geometry gates for one cue: the maximum deviation of each check (the design's table)."""
    g = geometry
    directions = condition_directions(g, controls, p_prime, config)
    neutral = [c for c in config.conditions if config.spec(c)["kind"] in ("rand", "plur")]
    checks = {
        "unit_t": abs(float(torch.linalg.vector_norm(g.t_hat)) - 1.0),
        "unit_controls": max(abs(float(torch.linalg.vector_norm(u)) - 1.0) for u in controls),
        "unit_plurality": abs(float(torch.linalg.vector_norm(p_prime)) - 1.0),
        "orth_E_t": abs(float(torch.dot(g.E_hat, g.t_hat))),
        "orth_E_controls": max(abs(float(torch.dot(g.E_hat, u))) for u in controls),
        "orth_t_controls": max(abs(float(torch.dot(g.t_hat, u))) for u in controls),
        "orth_E_plurality": abs(float(torch.dot(g.E_hat, p_prime))),
        "orth_t_plurality": abs(float(torch.dot(g.t_hat, p_prime))),
        "d_dot_controls": max(abs(float(torch.dot(d, u))) for u in controls),
        "d_dot_plurality": abs(float(torch.dot(d, p_prime))),
    }
    pairs = {}
    for condition in neutral:
        stem = condition[:-(len(config.primary) + 1)]
        pairs.setdefault(stem, {})[condition[len(stem)]] = condition
    checks["neutral64"] = max(abs(score(d, vectors64[pair["+"]]) - score(d, vectors64[pair["-"]])) for pair in pairs.values())
    checks["neutral32"] = max(abs(score(d, vectors32[pair["+"]].double()) - score(d, vectors32[pair["-"]].double())) for pair in pairs.values())
    odd64, odd32 = [], []
    for dose in (config.primary_odd, config.half_odd):
        plus, minus = f"noun+{_dose(dose)}", f"noun-{_dose(dose)}"
        odd64.append(abs(0.5 * (score(d, vectors64[plus]) - score(d, vectors64[minus])) - dose))
        odd32.append(abs(0.5 * (score(d, vectors32[plus].double()) - score(d, vectors32[minus].double())) - dose))
    checks["odd64"], checks["odd32"] = max(odd64), max(odd32)
    checks["length64"] = max(abs(float(torch.linalg.vector_norm(v)) / g.norm - 1.0) for v in vectors64.values())
    checks["length32"] = max(abs(float(torch.linalg.vector_norm(v.double())) / g.norm - 1.0) for v in vectors32.values())
    rotated = [c for c in config.conditions if c != "base"]
    checks["angle64"] = max(abs(angle(g.E, vectors64[c]) - directions[c][1]) for c in rotated)
    checks["angle32"] = max(abs(angle(g.E, vectors32[c].double()) - directions[c][1]) for c in rotated)
    checks["base_equals_model_row"] = bool(torch.equal(vectors32["base"], model_row32.to(torch.float32)))
    return checks


GEOMETRY_LIMITS = {
    **{name: "geometry" for name in ("unit_t", "unit_controls", "unit_plurality", "orth_E_t", "orth_E_controls", "orth_t_controls", "orth_E_plurality",
                                     "orth_t_plurality", "d_dot_controls", "d_dot_plurality", "neutral64", "odd64", "length64")},
    "angle64": "angle", "neutral32": "patched", "odd32": "patched", "length32": "patched", "angle32": "angle_patched",
}


def enforce_geometry(checks: Mapping[str, Any], where: str) -> None:
    if checks.get("base_equals_model_row") is not True:
        raise IncidentError(f"I7′: {where}: the θ = 0 vector cast to float32 is not the model's embedding row")
    for name, tolerance in GEOMETRY_LIMITS.items():
        value = checks[name]
        if value is None or not math.isfinite(float(value)) or float(value) > TOLERANCES[tolerance]:
            raise IncidentError(f"I7′: {where}: {name} = {value} exceeds {TOLERANCES[tolerance]:.0e}")


def cue_vectors(W_E: torch.Tensor, d: torch.Tensor, p_hat: torch.Tensor, token_id: int, config: Configuration) -> dict[str, Any]:
    """One cue's geometry, controls, plurality tangent, the 21 float64 vectors and their float32 casts."""
    geometry = cue_geometry(W_E, token_id, d)
    controls = control_directions(geometry, config.k_controls)
    p_prime = plurality_tangent(p_hat, geometry)
    vectors64 = condition_vectors(geometry, controls, p_prime, config)
    vectors32 = {condition: vector.to(torch.float32) for condition, vector in vectors64.items()}
    return {"geometry": geometry, "controls": controls, "p_prime": p_prime, "vectors64": vectors64, "vectors32": vectors32}


def _stack(vectors: Mapping[str, torch.Tensor], order: Sequence[str]) -> torch.Tensor:
    return torch.stack([vectors[name] for name in order])


def geometry_block(W_E: torch.Tensor, record: Mapping[str, Any], pool: Any, tokens: Sequence[Mapping[str, Any]], config: Configuration) -> dict[str, Any]:
    """The lock's weight-derived geometry (and I7′'s recomputation): the direction, each cue's scalars and digests, and
    the geometry gates' deviations. Raises an incident if any gate fails. Returns the JSON block and the float32 vectors."""
    d = direction(W_E, record)
    p_hat = plurality_direction(W_E, pool)
    cues: list[dict[str, Any]] = []
    vectors: dict[int, dict[str, torch.Tensor]] = {}
    for token in tokens:
        token_id = int(token["token_id"])
        entry = cue_vectors(W_E, d, p_hat, token_id, config)
        g = entry["geometry"]
        checks = geometry_checks(d, g, entry["controls"], entry["p_prime"], entry["vectors64"], entry["vectors32"], W_E[token_id], config)
        enforce_geometry(checks, f"{token['word']} ({token_id})")
        theta_primary, theta_half = theta_for(config.primary_odd, g.tau), theta_for(config.half_odd, g.tau)
        cues.append({"word": token["word"], "token_id": token_id, "stratum": token["stratum"], "norm": g.norm, "s0": g.s0, "tau": g.tau,
                     "theta_primary": theta_primary, "theta_half": theta_half, "even_primary": g.s0 * (math.cos(theta_primary) - 1.0),
                     "even_half": g.s0 * (math.cos(theta_half) - 1.0), "t_hat_sha256": rc.tensor_digest(g.t_hat),
                     "controls_sha256": rc.tensor_digest(torch.stack(list(entry["controls"]))), "plurality_sha256": rc.tensor_digest(entry["p_prime"]),
                     "vectors64_sha256": rc.tensor_digest(_stack(entry["vectors64"], config.conditions)),
                     "vectors32_sha256": rc.tensor_digest(_stack(entry["vectors32"], config.conditions)), "checks": rc.json_safe(checks)})
        vectors[token_id] = entry["vectors32"]
    maxima = {name: max(float(cue["checks"][name]) for cue in cues) for name in GEOMETRY_LIMITS}
    block = {"direction_sha256": rc.tensor_digest(d), "plurality_sha256": rc.tensor_digest(p_hat), "mu_noun_sha256": record["score"]["mu_noun_sha256"],
             "mu_cue_sha256": record["score"]["mu_cue_sha256"], "d_norm": float(torch.linalg.vector_norm(d)), "cos_d_plurality": float(torch.dot(unit(d), p_hat)),
             "conditions": list(config.conditions), "control_tag": CONTROL_TAG, "cues": cues, "check_maxima": maxima,
             "tolerances": {name: TOLERANCES[kind] for name, kind in GEOMETRY_LIMITS.items()}}
    return {"block": json.loads(pm.canonical_json(rc.json_safe(block))), "vectors32": vectors}


def nearest_tokens(W_E: torch.Tensor, vectors32: Mapping[int, Mapping[str, torch.Tensor]], conditions: Sequence[str]) -> dict[str, Any]:
    """Descriptive (the lock): for the named conditions, the nearest vocabulary token to each rotated embedding and the
    angle to the nearest other token."""
    normalized = W_E.double() / torch.linalg.vector_norm(W_E.double(), dim=1, keepdim=True)
    out: dict[str, Any] = {}
    for token_id, vectors in vectors32.items():
        entry = {}
        for condition in conditions:
            sims = normalized @ unit(vectors[condition])
            top = torch.topk(sims, 2)
            nearest = int(top.indices[0])
            other = int(top.indices[1]) if nearest == int(token_id) else nearest
            other_sim = float(sims[other])
            entry[condition] = {"nearest_is_own": nearest == int(token_id), "angle_to_nearest_other_deg": math.degrees(math.acos(max(-1.0, min(1.0, other_sim))))}
        out[str(token_id)] = entry
    return out


# ---------------------------------------------------------------------------
# 2. The freeze: tokenizer, text rules and committed files only; no model output.

CONFIRMATION_SCHEMA_VERSION = 1
FREEZE_RULES = {
    "word": "' ' + w is a single token of the pinned tokenizer",
    "earlier_cues": "its id was never used as a cue by Experiments 005–024: 024's committed freeze exclusion (005–023) plus 024's 40 cues",
    "target_nouns": "its id is none of the token ids of any pool target noun's singular or plural forms (161 ids)",
    "prior_nouns": "its id is none of the singular or plural ids of the 24 nouns of 020's committed confirmation file, which 021 read as fresh target nouns "
                   "(the tightened rule: excludes statue and barrel)",
    "frame_tokens": "its id occurs nowhere in the 108 exposed frames",
    "nouns": "a noun counts only when its singular and its plural are both eligible and distinct; for the new list the plural is the regular one",
    "picks": "the first 20 eligible adjectives of 024's adjective reserves in list order; the first 20 eligible nouns of 024's ordinary reserves then the "
             "frozen new list, in textual order; mechanically",
    "deviation": "picks that differ from the design's expected picks write nothing and stop for review",
    "score": "no nounness score is computed at the freeze, and none selects, rejects or reorders a candidate",
}


class FreezeShortfall(ul.FreezeShortfall):
    """A list yields fewer eligible entries than its quota: nothing is written; stop for review (not an incident)."""


class FreezeDeviation(RuntimeError):
    """The mechanical picks differ from the design's expected picks: nothing is written; stop for review."""


def _digested(ids: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(int(i) for i in ids)
    return {"ids": ordered, "sha256": pm.sha256_text(pm.canonical_json(ordered))}


def blocked_sets(pool: Any, freeze_024: Mapping[str, Any], prior: Mapping[str, Any]) -> dict[str, Any]:
    earlier = sorted(set(int(i) for i in freeze_024["exclusion"]["cue_token_ids"]) | {int(cue["token_id"]) for cue in freeze_024["cues"]})
    return {"earlier_cues": _digested(earlier), "target_forms": _digested(rr.target_noun_form_ids(pool)), "prior_nouns": _digested(prior["ids"]),
            "frame_tokens": _digested(rr.frame_token_ids(pool))}


_BLOCK_REASONS = (("earlier_cues", "already used as a cue by Experiments 005–024"), ("target_forms", "a form of a pool target noun"),
                  ("prior_nouns", "a form of a noun of 020's confirmation list (read by 021)"), ("frame_tokens", "a token of an exposed frame"))


def _status(tokenizer: Any, word: str, blocked: Mapping[str, frozenset[int]], picked: set[int]) -> tuple[int | None, str]:
    ids = pm._encode(tokenizer, " " + word)
    if len(ids) != 1:
        return None, f"{len(ids)} tokens with a leading space"
    token_id = int(ids[0])
    for key, reason in _BLOCK_REASONS:
        if token_id in blocked[key]:
            return None, f"token id {token_id}: {reason}"
    if token_id in picked:
        return None, f"token id {token_id}: already picked"
    return token_id, "eligible"


def select_cues(tokenizer: Any, blocked: Mapping[str, frozenset[int]], config: Configuration) -> dict[str, Any]:
    """The mechanical selection (every entry's status recorded), a shortfall check and the expected-picks check, all
    before anything is written."""
    picked: set[int] = set()
    rejected: list[dict[str, Any]] = []
    adjectives: list[dict[str, Any]] = []
    for rank, word in enumerate(ADJECTIVE_RESERVES):
        token_id, reason = _status(tokenizer, word, blocked, picked)
        if token_id is None:
            rejected.append({"list": "adjective-reserve", "candidate": word, "rank": rank, "reason": reason})
        elif len(adjectives) < config.n_adjectives:
            adjectives.append({"word": word, "token_id": token_id, "stratum": "adjective", "source": "adjective-reserve", "rank": rank})
            picked.add(token_id)
    nouns: list[dict[str, Any]] = []
    sources = [("ordinary-reserve", rank, singular, plural) for rank, (singular, plural) in enumerate(ORDINARY_RESERVES)]
    sources += [("new-list", rank, word, regular_plural(word)) for rank, word in enumerate(NEW_NOUN_LIST)]
    for source, rank, singular, plural in sources:
        token_s, reason_s = _status(tokenizer, singular, blocked, picked)
        token_p, reason_p = _status(tokenizer, plural, blocked, picked)
        if token_s is None or token_p is None or token_s == token_p:
            rejected.append({"list": source, "candidate": f"{singular}/{plural}", "rank": rank, "reason": f"{singular}: {reason_s}; {plural}: {reason_p}"})
        elif len(nouns) < config.n_nouns:
            nouns.append({"word": singular, "token_id": token_s, "stratum": "noun", "source": source, "rank": rank, "plural": plural, "plural_token_id": token_p})
            picked |= {token_s, token_p}
    if len(adjectives) < config.n_adjectives or len(nouns) < config.n_nouns:
        raise FreezeShortfall(f"eligible adjectives {len(adjectives)} of {config.n_adjectives}, nouns {len(nouns)} of {config.n_nouns}")
    picks = {"adjective": tuple(cue["word"] for cue in adjectives), "noun": tuple(cue["word"] for cue in nouns)}
    differing = {key: {"picked": list(picks[key]), "expected": list(config.expected(key))} for key in STRATA if picks[key] != config.expected(key)}
    if differing:
        raise FreezeDeviation(f"the mechanical picks differ from the design's expected picks: {differing}")
    return {"cues": adjectives + nouns, "rejected": rejected, "picks": {key: list(value) for key, value in picks.items()}}


@dataclass(frozen=True)
class Confirmation025:
    reference_ids: Mapping[str, int]
    exposed_frames: tuple[pm.Frame, ...]  # the 108 exposed frames, in the pool's order
    tokens: tuple[dict[str, Any], ...]  # the 40 cues in frozen order: the adjectives, then the nouns
    conditions: tuple[str, ...]
    content_sha256: str
    frames: tuple[pm.Frame, ...] = ()

    def stratum_tokens(self, stratum: str) -> tuple[dict[str, Any], ...]:
        return tuple(token for token in self.tokens if token["stratum"] == stratum)

    @property
    def prompts(self) -> tuple[pm.Prompt, ...]:
        return tuple(pm.Prompt(frame, int(token["token_id"]), token["word"]) for frame in self.exposed_frames for token in self.tokens)

    def manifest_keys(self) -> frozenset[str]:
        """The untagged prompt keys (what the spent-key isolation checks)."""
        return frozenset(prompt.key for prompt in self.prompts)

    def tagged_keys(self) -> frozenset[str]:
        return frozenset(f"{prompt.key}|{condition}" for prompt in self.prompts for condition in self.conditions)

    def manifest(self) -> dict[str, Any]:
        return {"S2-TARGET": sorted(self.tagged_keys())}

    def counts(self) -> dict[str, Any]:
        return {"strata": {stratum: len(self.stratum_tokens(stratum)) for stratum in STRATA}, "frames": len(self.exposed_frames),
                "conditions": len(self.conditions), "runs": len(self.tokens) * len(self.exposed_frames) * len(self.conditions)}


def tagged_key(prompt: pm.Prompt, condition: str) -> str:
    return f"{prompt.key}|{condition}"


def freeze_payload(tokenizer: Any, *, pool: Any, freeze_024: Mapping[str, Any], prior: Mapping[str, Any], config: Configuration,
                   model: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The confirmation file's content: the 40 cues, every rule's inputs and the 90,720-key condition-tagged manifest.
    A shortfall or a deviation raises before anything exists to write."""
    blocked_json = blocked_sets(pool, freeze_024, prior)
    blocked = {key: frozenset(value["ids"]) for key, value in blocked_json.items()}
    selection = select_cues(tokenizer, blocked, config)
    payload = {
        "experiment": EXPERIMENT, "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "kind": "the fresh cues of Experiment 025, frozen from the tokenizer and the text rules alone; no model output", "design": dict(DESIGN), "plan": dict(PLAN),
        "model": dict(model or {"model_id": models_module.PYTHIA_70M.model_id, "revision": models_module.PYTHIA_70M.revision}), "configuration": config.to_json(),
        "candidates": {"adjective-reserve": list(ADJECTIVE_RESERVES), "ordinary-reserve": [list(pair) for pair in ORDINARY_RESERVES], "new-list": list(NEW_NOUN_LIST)},
        "rules": dict(FREEZE_RULES), "blocked": blocked_json,
        "sources": {"freeze_024": {"path": rr.CONFIRMATION_RELATIVE_PATH, "content_sha256": freeze_024["content_sha256"]},
                    "prior_nouns_020": {"path": prior["path"], "file_sha256": prior["file_sha256"], "nouns": prior["nouns"]}},
        "reference_cue_ids": {template: int(token_id) for template, token_id in pool.reference_ids.items()}, "exposed_frame_ids": [frame.frame_id for frame in pool.frames],
        "cues": selection["cues"], "rejected": selection["rejected"], "picks": selection["picks"], "expected_picks": config.to_json()["expected_picks"],
        "picks_match_expected": True, "conditions": list(config.conditions),
    }
    confirmation = confirmation_from_payload(payload, pool, config, verify=False)
    payload["counts"] = confirmation.counts()
    payload["manifest"] = confirmation.manifest()
    payload["manifest_sha256"] = pm.sha256_text(pm.canonical_json(payload["manifest"]))
    payload["content_sha256"] = rc.content_digest(payload)
    return payload


def confirmation_from_payload(payload: Mapping[str, Any], pool: Any, config: Configuration, *, verify: bool = True) -> Confirmation025:
    tokens = tuple({"word": entry["word"], "token_id": int(entry["token_id"]), "stratum": entry["stratum"]} for entry in payload["cues"])
    confirmation = Confirmation025(dict(pool.reference_ids), tuple(pool.frames), tokens, tuple(payload["conditions"]), str(payload.get("content_sha256", "")))
    if verify:
        if payload.get("experiment") != EXPERIMENT or payload.get("schema_version") != CONFIRMATION_SCHEMA_VERSION:
            raise PhaseError("not an Experiment 025 confirmation file")
        if payload["content_sha256"] != rc.content_digest(payload):
            raise PhaseError("the confirmation file's content digest does not verify")
        if payload.get("configuration") != config.to_json() or tuple(payload["conditions"]) != config.conditions:
            raise PhaseError("the confirmation file was frozen under a different configuration")
        if payload["manifest"] != confirmation.manifest() or payload.get("counts") != confirmation.counts() \
                or payload.get("manifest_sha256") != pm.sha256_text(pm.canonical_json(payload["manifest"])):
            raise PhaseError("the confirmation file's manifest or counts are not the ones its cues define")
        expected = config.to_json()["expected_picks"]
        if payload.get("picks_match_expected") is not True or payload.get("picks") != expected or payload.get("expected_picks") != expected:
            raise PhaseError("the confirmation file's picks are not the design's expected picks")
        if [token["word"] for token in confirmation.stratum_tokens("adjective")] != expected["adjective"] \
                or [token["word"] for token in confirmation.stratum_tokens("noun")] != expected["noun"] or len({t["token_id"] for t in tokens}) != config.n_cues:
            raise PhaseError("the confirmation file's strata are not the expected picks in order")
        if payload["exposed_frame_ids"] != [frame.frame_id for frame in pool.frames] or dict(payload["reference_cue_ids"]) != {k: int(v) for k, v in pool.reference_ids.items()}:
            raise PhaseError("the confirmation file names a different exposed pool")
        if len(pool.frames) != config.n_frames:
            raise PhaseError(f"the exposed pool has {len(pool.frames)} frames, not {config.n_frames}")
    return confirmation


def load_confirmation_025(path: Path, pool: Any, freeze_024: Mapping[str, Any], prior: Mapping[str, Any], config: Configuration) -> Confirmation025:
    """The committed freeze artifact, verified, with the rules' inputs recomputed now."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    confirmation = confirmation_from_payload(payload, pool, config)
    now = blocked_sets(pool, freeze_024, prior)
    if payload["blocked"] != now:
        raise PhaseError("the excluded ids recorded at the freeze differ from the ones the committed inputs give now")
    ids = {int(cue["token_id"]) for cue in payload["cues"]} | {int(cue["plural_token_id"]) for cue in payload["cues"] if "plural_token_id" in cue}
    if any(ids & set(value["ids"]) for value in now.values()):
        raise PhaseError("a fresh cue id is excluded")
    return confirmation


# ---------------------------------------------------------------------------
# 3. The lock (weights only), the preregistration, the lock's validation, the scientific paths.

LOCK_SCHEMA_VERSION = 1
MEASUREMENT = ("one forward per condition-tagged key through pm.run_patched, REPLACE at ('EMBED', p_c) with the condition's float32 vector; the capture "
               "sites are ul.measured_sites(frame) plus ATTN_PATTERN.L4/L5 at p_t; Δx1, Δx3 and Δc are derived exactly as ul.measure_prompt derives them "
               "against 020's locked reference state; C = ul.contrast_of(progs, state, Δx3)")


def scientific_dependencies(inputs: ul.FrozenInputs, *, parameters_sha256: str, embedding_sha256: str) -> dict[str, Any]:
    locked = inputs.closure["exploration"]["locked_states"]
    return {"module_blobs": dict(FROZEN_BLOBS), "C": rr.C_DEFINITION, "measurement": MEASUREMENT,
            "readout_020": {"module": "readout_decompilation.py", "blob": FROZEN_BLOBS["readout_decompilation.py"], "exposed_states_sha256": ul.exposed_states_digest(locked),
                            "frozen_input_digests": {key: inputs.digests[key] for key in rc.DIGEST_KEYS}},
            "model": {"model_id": models_module.PYTHIA_70M.model_id, "revision": models_module.PYTHIA_70M.revision, "parameters_sha256": parameters_sha256,
                      "embedding_sha256": embedding_sha256}}


def verify_dependencies(recorded: Mapping[str, Any], now: Mapping[str, Any], what: str) -> None:
    if json.loads(pm.canonical_json(dict(recorded))) != json.loads(pm.canonical_json(dict(now))):
        differing = sorted(key for key in set(recorded) | set(now) if recorded.get(key) != now.get(key))
        raise PhaseError(f"{what}: the scientific dependencies differ ({differing})")


def confirmation_binding(confirmation: Confirmation025, file_sha256: str) -> dict[str, Any]:
    return {"path": CONFIRMATION_RELATIVE_PATH, "file_sha256": file_sha256, "content_sha256": confirmation.content_sha256,
            "manifest_sha256": pm.sha256_text(pm.canonical_json(confirmation.manifest())), "counts": confirmation.counts()}


def binding_024(root: Path) -> dict[str, Any]:
    return {kind: {"path": relative, "file_sha256": INHERITED_024[f"{kind}_file_sha256"], "content_sha256": INHERITED_024[f"{kind}_content_sha256"]}
            for kind, relative in INHERITED_024_PATHS.items()}


def build_lock(*, run_id: str, protocol_code_commit: str, digests: Mapping[str, str], config: Configuration, confirmation: Confirmation025,
               confirmation_file_sha256: str, geometry: Mapping[str, Any], nearest: Mapping[str, Any], dependencies: Mapping[str, Any], noun_keys: Sequence[str],
               root: Path) -> dict[str, Any]:
    lock = {
        "experiment": EXPERIMENT, "schema_version": LOCK_SCHEMA_VERSION, "kind": "the Experiment 025 preregistration lock", "run_id": run_id,
        "protocol_code_commit": protocol_code_commit, "design": dict(DESIGN), "plan": dict(PLAN), "configuration": config.to_json(), "inputs": dict(digests),
        "module_blobs": dict(FROZEN_BLOBS), "module": {"path": "src/neural_decompiler/cue_rotation.py", "blob": own_blob()},
        "inherited_024": binding_024(root), "confirmation_025": confirmation_binding(confirmation, confirmation_file_sha256), "dependencies": dict(dependencies),
        "noun_keys": list(noun_keys), "geometry": dict(geometry), "nearest_tokens": dict(nearest), "patch_path_spent_keys": list(PATCH_PATH_SPENT_KEYS),
        "statistics": {"A": "A_i = ½[ℓ_i(+θ_i) − ℓ_i(−θ_i)] at the primary dose", "B": "B_i = A_i − (1/k)·Σ_j |A_ij|, A_ij the random controls' A",
                       "G": "G_i = ½[D_attn,i(+θ_i) − D_attn,i(−θ_i)] at the primary dose",
                       "ell": "ℓ = log(Σ SSE_C / Σ n) over the cue's 108 exposed frames × 79 scorable nouns (rr.fresh_pair_cells, rr.cue_mse)",
                       "D_attn": "total-variation distance of the captured L4/L5 rows at p_t from the locked 020 rows, equal mean over the 16 heads, then over "
                                 "the 108 frames",
                       "count_rule": f"PASS iff at least {config.count_threshold} of {config.n_cues} values are strictly positive; ties and zeros count against"},
        "tolerances": dict(TOLERANCES), "outcome": {"table": [list(row) for row in OUTCOME_TABLE], "labels": list(OUTCOMES)}, "semantics": dict(SEMANTICS),
    }
    lock["content_sha256"] = rc.content_digest(lock)
    return lock


STATISTICS_ORDER = ("A", "B", "G", "ell", "D_attn", "count_rule")


def _full(value: Any) -> str:
    return "—" if value is None else repr(float(value)) if isinstance(value, float) else str(value)


def render_preregistration(lock: Mapping[str, Any]) -> str:
    config = lock["configuration"]
    geometry = lock["geometry"]
    lines = [f"# Experiment 025 — preregistration", "",
             f"- Lock `{lock['content_sha256']}`; run `{lock['run_id']}` at `{lock['protocol_code_commit']}`; design revision {lock['design']['revision']} "
             f"(`{lock['design']['commit']}`, corrected `{lock['design']['correction']}`), plan revision {lock['plan']['revision']} (`{lock['plan']['commit']}`)",
             f"- The analysis module `{lock['module']['path']}` at blob `{lock['module']['blob']}`; configuration `{config['name']}`",
             f"- The freeze `{lock['confirmation_025']['content_sha256']}`; manifest `{lock['confirmation_025']['manifest_sha256']}` "
             f"({lock['confirmation_025']['counts']['runs']} condition-tagged runs: {config['n_cues']} cues × {config['n_frames']} frames × "
             f"{len(config['conditions'])} conditions)", "",
             "## The intervention", "",
             f"- `(\"EMBED\", p_c)`, REPLACE; the direction `d = μ̂_noun − μ̂_cue` (digest `{geometry['direction_sha256']}`, |d| {_full(geometry['d_norm'])}); "
             f"cos(d̂, p̂) = {_full(geometry['cos_d_plurality'])} (descriptive)",
             f"- The directional (odd) score component is exactly ±{config['primary_odd']} (primary) and ±{config['half_odd']} (secondary); the complete "
             f"change is s₀(cos θ − 1) ± odd", f"- {config['k_controls']} nounness-neutral random tangent controls per cue (tag `{geometry['control_tag']}`) "
             f"at the same angle; a plurality control (secondary)", f"- The conditions: {', '.join(config['conditions'])}", "",
             "## The outcome-bearing statistics", ""]
    lines += [f"- **{name}**: {lock['statistics'][name]}" for name in STATISTICS_ORDER]  # a fixed order: the installed lock is canonical JSON
    tail = config["reference_tail"]
    lines += [f"- The criterion: {lock['semantics']['criterion']}; P(X ≥ {config['count_threshold']}) = {tail['exact']} = {_full(tail['value'])}",
              f"- **A**: {lock['semantics']['A']}", f"- **B**: {lock['semantics']['B']}", f"- **G**: {lock['semantics']['G']}", "",
              "## The outcome", "", "| condition | label |", "|---|---|"]
    lines += [f"| {condition} | `{label}` |" for condition, label in lock["outcome"]["table"]]
    lines += ["", "Not claimed by any label:"] + [f"- {item}" for item in lock["semantics"]["not_claimed"]]
    lines += ["", f"- Strata: {lock['semantics']['strata']}", f"- Secondary: {lock['semantics']['secondary']}", "",
              "## The validity gates", "",
              f"- I7′: the geometry recomputed from the weights before any prompt, bit for bit, and every geometry check within its tolerance "
              f"({', '.join(f'{name} {value:.0e}' for name, value in sorted(geometry['tolerances'].items()))})",
              f"- The patch path on {len(lock['patch_path_spent_keys'])} spent keys before the ledger: {', '.join(lock['patch_path_spent_keys'])}",
              f"- I1, I3, I4 on every run ({lock['tolerances']['I1']:.0e}, {lock['tolerances']['I3']:.0e} relative, {lock['tolerances']['I4']:.0e}); C recomputed "
              f"bit for bit; the Level-1 identity ≤ {lock['tolerances']['level1']} on every outcome-bearing primary-dose run", "",
              "## The 40 cues and their geometry", "", "| stratum | word | token id | s₀ | τ | θ primary (°) | θ half (°) | even (primary) |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {cue['stratum']} | {cue['word']} | {cue['token_id']} | {cue['s0']:+.6f} | {cue['tau']:.6f} | {math.degrees(cue['theta_primary']):.4f} | "
              f"{math.degrees(cue['theta_half']):.4f} | {cue['even_primary']:+.6f} |" for cue in geometry["cues"]]
    lines += ["", "- Geometry check maxima: " + ", ".join(f"{name} {value:.3e}" for name, value in sorted(geometry["check_maxima"].items())), ""]
    return "\n".join(lines)


SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", *rr.SCIENTIFIC_PATH_PREFIXES[1:])
NON_SCIENTIFIC_PATHS = (CONFIRMATION_RELATIVE_PATH, LOCK_RELATIVE_PATH, PREREGISTRATION_RELATIVE_PATH, f"{EXPERIMENT_DIR}/README.md", *rr.NON_SCIENTIFIC_PATHS[4:])
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/", *rr.NON_SCIENTIFIC_PREFIXES)


def scientific_changes(paths: Sequence[str]) -> list[str]:
    """Scientific paths: all code, and 025's, 024's, 023's … experiment directories (024's committed artifacts included:
    they are 025's inputs), except 025's installed artifacts, the READMEs and the evidence directories."""
    return [path for path in paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and path not in NON_SCIENTIFIC_PATHS and not path.startswith(NON_SCIENTIFIC_PREFIXES)]


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], config: Configuration, confirmation: Confirmation025,
                  confirmation_file_sha256: str, dependencies: Mapping[str, Any], noun_keys: Sequence[str], preregistration_text: str,
                  git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    """The installed lock before the model is loaded: its digest; that it is this run's candidate; every bound input,
    module (the analysis module's own blob directly), configuration, dependency and wording; the rendered
    preregistration; a clean tree with no scientific change since the lock commit."""
    if lock.get("experiment") != EXPERIMENT or lock.get("content_sha256") != rc.content_digest(lock):
        raise PhaseError("the installed lock is not a verified Experiment 025 lock")
    if state.get("lock") is None or state["lock"].get("content_sha256") != lock["content_sha256"]:
        raise PhaseError("the installed lock is not the candidate this run wrote")
    if lock["inputs"] != dict(digests) or lock["module_blobs"] != dict(FROZEN_BLOBS) or lock["design"] != dict(DESIGN) or lock["plan"] != dict(PLAN):
        raise PhaseError("the lock was written against different frozen inputs, modules, design or plan")
    if lock["module"] != {"path": "src/neural_decompiler/cue_rotation.py", "blob": own_blob()} or rc.program_blob_sha1(Path(rr.__file__)) != FROZEN_BLOBS["readout_routing.py"]:
        raise PhaseError("the running analysis module is not the one the lock binds (direct module-blob check)")
    if lock["configuration"] != config.to_json():
        raise PhaseError("the lock was written under a different configuration")
    if lock["confirmation_025"] != confirmation_binding(confirmation, confirmation_file_sha256):
        raise PhaseError("the lock was written against a different confirmation file")
    comparable = {key: value for key, value in dependencies.items() if key != "model"}
    verify_dependencies({key: value for key, value in lock["dependencies"].items() if key != "model"}, comparable, "the lock")
    if lock["noun_keys"] != list(noun_keys) or lock["semantics"] != dict(SEMANTICS) or lock["outcome"]["table"] != [list(row) for row in OUTCOME_TABLE] \
            or lock["tolerances"] != dict(TOLERANCES) or lock["patch_path_spent_keys"] != list(PATCH_PATH_SPENT_KEYS):
        raise PhaseError("the lock names a different noun order, wording, outcome table, tolerance or patch-path key")
    if preregistration_text != render_preregistration(lock) or pm.sha256_text(preregistration_text) != state["lock"].get("preregistration_sha256"):
        raise PhaseError("the installed preregistration is not the one this lock renders")
    if not tracked:
        raise PhaseError("the lock and the preregistration must be tracked and committed")
    if git_state.get("dirty"):
        raise PhaseError("confirm requires a clean Git tree")
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = scientific_changes(changed_paths)
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock: {scientific}")


def verify_geometry_against_lock(recomputed: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    """I7′'s bit-for-bit comparison of the geometry block recomputed from the weights against the lock's."""
    differing = sorted(key for key in set(recomputed) | set(lock["geometry"]) if recomputed.get(key) != lock["geometry"].get(key))
    return {"bitwise_equal": not differing, "differing": differing}


# ---------------------------------------------------------------------------
# 4. The measurement: one patched forward per condition-tagged key.


def attention_sites(frame: pm.Frame) -> list[tuple[str, int]]:
    return [("ATTN_PATTERN.L4", frame.p_t), ("ATTN_PATTERN.L5", frame.p_t)]


def measure_rotated(model: Any, progs: ul.ModelPrograms, frame: pm.Frame, state: rd.FrameState020, token_id: int, word: str, vector32: torch.Tensor) -> dict[str, Any]:
    """One forward of the cue prompt with ``EMBED@p_c`` replaced by ``vector32`` (checked by ``pm.run_patched``): the raw
    float32 captures at (p_c, p_t) of the block-1 and block-3 inputs, the measured ``Δc`` (``ul.measure_prompt``'s
    formula), and the block-4/5 attention rows at ``p_t``."""
    site = ("EMBED", frame.p_c)
    replacement = vector32.detach().reshape(1, 1, -1).to(torch.float32)
    run = pm.run_patched(model, pm.Prompt(frame, int(token_id), word), {site: replacement}, {site: pm.ReplacementSource.DIRECT},
                         capture_sites=ul.measured_sites(frame) + attention_sites(frame))
    slots = (frame.p_c, frame.p_t)
    x1 = torch.stack([run.vector(("RESID_PRE.L1", position)).detach().to(torch.float32) for position in slots])
    x3 = torch.stack([run.vector((f"RESID_PRE.L{hp.HEAD_LAYER}", position)).detach().to(torch.float32) for position in slots])
    dc = (progs.nouns.contrasts(run.logits) - state.c_ref)[progs.scorable].double()
    return {"x1": x1, "x3": x3, "dc": dc, "rows4": run.vector(("ATTN_PATTERN.L4", frame.p_t)).detach().to(torch.float32),
            "rows5": run.vector(("ATTN_PATTERN.L5", frame.p_t)).detach().to(torch.float32), "integrity": run.integrity}


def changes_from_raw(raw: torch.Tensor, reference_all: Sequence[torch.Tensor], frame: pm.Frame) -> dict[int, torch.Tensor]:
    """``Δ`` at the changed positions from the raw float32 captures, exactly as ``ul.measure_prompt`` computes it."""
    out = {frame.p_c: raw[0].double() - reference_all[frame.p_c].double()}
    if frame.p_t != frame.p_c:
        out[frame.p_t] = raw[1].double() - reference_all[frame.p_t].double()
    return out


def routing_distance(rows4: torch.Tensor, rows5: torch.Tensor, state: rd.FrameState020) -> float:
    """``D_attn`` for one run: the equal mean over the 16 heads of blocks 4 and 5 of the total-variation distance
    ``½·Σ_keys |a − a_ref|`` between the captured row at ``p_t`` and the locked 020 reference row."""
    distances: list[float] = []
    for captured, reference in ((rows4, state.rows4), (rows5, state.rows5)):
        keys = int(reference.shape[1])
        a, b = captured.double()[:, :keys], reference.double()
        if a.shape != b.shape:
            raise IncidentError(f"attention rows {tuple(a.shape)} against reference {tuple(b.shape)}")
        distances += [0.5 * math.fsum((a[head] - b[head]).abs().tolist()) for head in range(a.shape[0])]
    return math.fsum(distances) / len(distances)


def patch_path_check(model: Any, frames_by_id: Mapping[str, pm.Frame], W_E32: torch.Tensor, spent: frozenset[str],
                     keys: Sequence[str] | None = None) -> dict[str, Any]:
    """The confirm-time engineering identity on already-executed keys (signed off by the reviewer): a plain capture
    against a patched θ = 0 run (the model's own embedding row), bit for bit; and the embedding hook against the block-0
    residual input. No scientific quantity is computed; only equality results and digests are recorded. The keys default
    to ``PATCH_PATH_SPENT_KEYS`` read at call time (the lock binds that constant)."""
    keys = tuple(PATCH_PATH_SPENT_KEYS if keys is None else keys)
    out: dict[str, Any] = {"keys": list(keys), "checks": {}}
    for key in keys:
        if key not in spent:
            raise IncidentError(f"the patch-path key {key} is not an already-executed key")
        frame_id, word, token_id = key.split("|")
        frame = frames_by_id[frame_id]
        prompt = pm.Prompt(frame, int(token_id), word)
        sites = ul.measured_sites(frame) + attention_sites(frame) + [("EMBED", frame.p_c), ("RESID_PRE.L0", frame.p_c)]
        plain = pm.capture_prompt(model, prompt, sites)
        site = ("EMBED", frame.p_c)
        patched = pm.run_patched(model, prompt, {site: W_E32[int(token_id)].detach().reshape(1, 1, -1).to(torch.float32)}, {site: pm.ReplacementSource.DIRECT},
                                 capture_sites=sites)
        same = set(plain.slices) == set(patched.slices) and bool(torch.equal(plain.logits, patched.logits)) \
            and all(torch.equal(plain.slices[label], patched.slices[label]) for label in plain.slices)
        embed = all(bool(torch.equal(run.vector(("EMBED", frame.p_c)), run.vector(("RESID_PRE.L0", frame.p_c)))) for run in (plain, patched))
        out["checks"][key] = {"plain_equals_patched_theta0": same, "embed_equals_resid_pre0": embed,
                              "logits_sha256": rc.tensor_digest(plain.logits.double()),
                              "slices_sha256": {label: rc.tensor_digest(plain.slices[label].double()) for label in sorted(plain.slices)}}
    out["passed"] = all(entry["plain_equals_patched_theta0"] and entry["embed_equals_resid_pre0"] for entry in out["checks"].values())
    return out


# ---------------------------------------------------------------------------
# 5. Stage 2: every condition-tagged key once, the tensors, the accounting.


def target_units(confirmation: Confirmation025) -> ul.TableUnits:
    """The measurement's pair order (``ul.table_units``): cue token id, then ``frame_id``; one block per group."""
    return ul.table_units(confirmation.tokens, confirmation.exposed_frames)


def tensor_name(group: str, condition: str, name: str) -> str:
    return f"{group}/{condition}/{name}"


def stage_two(model: Any, progs: ul.ModelPrograms, confirmation: Confirmation025, states: Mapping[str, rd.FrameState020],
              vectors32: Mapping[int, Mapping[str, torch.Tensor]], *, executed: list[str], log: Callable[[str], None] | None = None,
              out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """Every condition-tagged key once: frame by frame (``frame_id`` order), cues by token id, conditions in frozen
    order. Returns the raw captures, ``Δc``, ``C`` and the attention rows per group and condition; nothing is
    enforced. The tensors are allocated into ``out`` when it is given, so that a failure part-way leaves the runs
    measured so far (and ``executed``) with the caller."""
    say = log or (lambda message: None)
    units = target_units(confirmation)
    d_model = int(progs.weights.W_E.shape[1])
    n_nouns = len(progs.scorable)
    n_heads = int(next(iter(states.values())).rows4.shape[0])
    where = {group: {pair: row for row, pair in enumerate(pairs)} for group, pairs in units.pairs.items()}
    tensors: dict[str, torch.Tensor] = out if out is not None else {}
    for group, pairs in units.pairs.items():
        size = len(pairs)
        keys = max((units.frames[f].p_t + 1 for _, f in pairs), default=1)
        tensors[f"{group}/positions"] = torch.zeros(size, 2, dtype=torch.int64)
        tensors[f"{group}/keys"] = torch.zeros(size, dtype=torch.int64)
        for condition in confirmation.conditions:
            tensors[tensor_name(group, condition, "x1")] = torch.full((size, 2, d_model), math.nan, dtype=torch.float32)
            tensors[tensor_name(group, condition, "x3")] = torch.full((size, 2, d_model), math.nan, dtype=torch.float32)
            tensors[tensor_name(group, condition, "dc")] = torch.full((size, n_nouns), math.nan, dtype=torch.float64)
            tensors[tensor_name(group, condition, "C")] = torch.full((size, n_nouns), math.nan, dtype=torch.float64)
            tensors[tensor_name(group, condition, "rows4")] = torch.zeros(size, n_heads, keys, dtype=torch.float32)
            tensors[tensor_name(group, condition, "rows5")] = torch.zeros(size, n_heads, keys, dtype=torch.float32)
    for f, frame in enumerate(units.frames):
        state = states[frame.frame_id]
        group = "coordinated" if frame.template_id == ul.COORDINATED else "cue_final"
        for t, token in enumerate(units.tokens):
            row = where[group][(t, f)]
            tensors[f"{group}/positions"][row] = torch.tensor([frame.p_c, frame.p_t], dtype=torch.int64)
            tensors[f"{group}/keys"][row] = frame.p_t + 1
            prompt = pm.Prompt(frame, int(token["token_id"]), token["word"])
            for condition in confirmation.conditions:
                measured = measure_rotated(model, progs, frame, state, int(token["token_id"]), token["word"], vectors32[int(token["token_id"])][condition])
                executed.append(tagged_key(prompt, condition))
                tensors[tensor_name(group, condition, "x1")][row] = measured["x1"]
                tensors[tensor_name(group, condition, "x3")][row] = measured["x3"]
                tensors[tensor_name(group, condition, "dc")][row] = measured["dc"]
                tensors[tensor_name(group, condition, "C")][row] = ul.contrast_of(progs, state, changes_from_raw(measured["x3"], state.x3_all, frame))
                keys = frame.p_t + 1
                tensors[tensor_name(group, condition, "rows4")][row, :, :keys] = measured["rows4"][:, :keys]
                tensors[tensor_name(group, condition, "rows5")][row, :, :keys] = measured["rows5"][:, :keys]
        say(f"  frame {f + 1}/{len(units.frames)} ({frame.frame_id}) measured")
    say(f"  {len(executed)} condition-tagged runs measured")
    return tensors


def assert_measurements(tensors: Mapping[str, torch.Tensor], units: ul.TableUnits, conditions: Sequence[str], n_nouns: int) -> None:
    """Every expected tensor, of its shape, and every measured value finite."""
    for group, pairs in units.pairs.items():
        size = len(pairs)
        if tuple(tensors[f"{group}/positions"].shape) != (size, 2):
            raise IncidentError(f"the saved positions of {group} have the wrong shape")
        for condition in conditions:
            for name in ("x1", "x3"):
                tensor = tensors[tensor_name(group, condition, name)]
                if tensor.shape[0] != size or not bool(torch.isfinite(tensor).all()):
                    raise IncidentError(f"the saved {name} of {group}/{condition} is incomplete or not finite")
            for name in ("dc", "C"):
                tensor = tensors[tensor_name(group, condition, name)]
                if tuple(tensor.shape) != (size, n_nouns) or not bool(torch.isfinite(tensor).all()):
                    raise IncidentError(f"the saved {name} of {group}/{condition} is incomplete or not finite")
            for name in ("rows4", "rows5"):
                if not bool(torch.isfinite(tensors[tensor_name(group, condition, name)]).all()):
                    raise IncidentError(f"the saved {name} of {group}/{condition} is not finite")


def recompute_c(progs: ul.ModelPrograms, units: ul.TableUnits, states: Mapping[str, rd.FrameState020], tensors: Mapping[str, torch.Tensor],
                conditions: Sequence[str]) -> dict[str, Any]:
    """``C`` recomputed from the saved raw ``x3`` of every run against the saved ``C``: bit for bit."""
    worst: dict[str, Any] = {"max": 0.0, "at": "", "runs": 0, "bitwise_equal": True}
    for group, pairs in units.pairs.items():
        for condition in conditions:
            raw, saved = tensors[tensor_name(group, condition, "x3")], tensors[tensor_name(group, condition, "C")]
            for row, (t, f) in enumerate(pairs):
                frame = units.frames[f]
                state = states[frame.frame_id]
                again = ul.contrast_of(progs, state, changes_from_raw(raw[row], state.x3_all, frame))
                worst["runs"] += 1
                if not torch.equal(again, saved[row]):
                    worst["bitwise_equal"] = False
                    ul._worse(worst, float((again - saved[row]).abs().max()), f"{units.tokens[t]['word']}|{frame.frame_id}|{condition}")
    return rc.json_safe(worst)


# ---------------------------------------------------------------------------
# 6. The gates: vector I1, I3, I4 on every run; the Level-1 identity on the outcome-bearing runs.


def lexicon_of(weights: pm.Weights, vector: torch.Tensor) -> torch.Tensor:
    """``pm.lexicon_vector``'s body applied to a vector: ``MLP₀(LN₂(v))``, in the vector's dtype (float32 here)."""
    if weights.act_fn != "gelu":
        raise IncidentError(f"unsupported activation {weights.act_fn}; the lexicon assumes exact GELU")
    hidden = pm.exact_layer_norm(vector, weights.ln2_0_w, weights.ln2_0_b, weights.eps)
    pre = hidden @ weights.mlp0_W_in + weights.mlp0_b_in
    return torch.nn.functional.gelu(pre) @ weights.mlp0_W_out + weights.mlp0_b_out


def vector_factors(progs: ul.ModelPrograms, x0_all: Sequence[torch.Tensor], frame: pm.Frame, vector32: torch.Tensor, reference_id: int) -> ul.Factors:
    """``ul.pair_factors`` with the cue's embedding replaced by ``vector32``: ``d_emb = v − W_E[ref]`` and
    ``delta_e = lexicon(v) − lexicon(ref_template)``, the chain's own template reference (as ``encoding_delta``)."""
    reference_template = int(progs.chain.fcm.read.reference_ids[frame.template_id])
    delta_e = lexicon_of(progs.weights, vector32.to(torch.float32)).double() - pm.lexicon_vector(progs.weights, reference_template).double()
    d_emb = vector32.to(torch.float32).double() - progs.weights.W_E[int(reference_id)].double()
    b_pc = ul.block0_terms(progs.program0, x0_all, frame.p_c, frame.p_c, d_emb)
    b_pt = ul.block0_terms(progs.program0, x0_all, frame.p_c, frame.p_t, d_emb) if frame.p_t != frame.p_c else None
    return ul.Factors(delta_e, d_emb, b_pc.value, b_pc.pattern, None if b_pt is None else b_pt.total, b_pc, b_pt)


def identity_gates(progs: ul.ModelPrograms, confirmation: Confirmation025, units: ul.TableUnits, states: Mapping[str, rd.FrameState020],
                   tensors: Mapping[str, torch.Tensor], vectors32: Mapping[int, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    """I1, I3 and I4 on every run: 023's formulas (``rr.target_gates``'s), with the cue's embedding the run's patched
    vector (``vector_factors``)."""
    gates = {name: {"max": 0.0, "at": ""} for name in ("I1", "I3", "I4")}
    cache: dict[str, tuple[Any, Any]] = {}
    for group, pairs in units.pairs.items():
        for row, (t, f) in enumerate(pairs):
            frame, token = units.frames[f], units.tokens[t]
            state = states[frame.frame_id]
            reference_id = int(confirmation.reference_ids[frame.template_id])
            if frame.frame_id not in cache:
                cache[frame.frame_id] = (ul.reference_embeddings(progs.weights, frame, reference_id), ul.reference_rows_017(progs.programs, state))
            x0_all, rows16 = cache[frame.frame_id]
            for condition in confirmation.conditions:
                where = f"{token['word']}|{frame.frame_id}|{condition}"
                factors = vector_factors(progs, x0_all, frame, vectors32[int(token["token_id"])][condition], reference_id)
                ctx = ul.PairContext(frame, state, rows16, factors, int(token["token_id"]), token["word"])
                dx1 = changes_from_raw(tensors[tensor_name(group, condition, "x1")][row], state.state_017.x1_all, frame)
                dx3 = changes_from_raw(tensors[tensor_name(group, condition, "x3")][row], state.x3_all, frame)
                i1 = float((factors.d_emb + factors.delta_e + factors.block0_pc.total - dx1[frame.p_c]).abs().max())
                if factors.block0_pt is not None:
                    i1 = max(i1, float((factors.block0_pt.total - dx1[frame.p_t]).abs().max()))
                ul._worse(gates["I1"], i1, where)
                full = ul.compose_dx3(progs, ctx, b0c.FULL_MASK[group])
                ul._worse(gates["I3"], ul.i3_error(full, dx3), where)
                ul._worse(gates["I4"], float((ul.contrast_of(progs, state, full) - tensors[tensor_name(group, condition, "C")][row]).abs().max()), where)
    return {name: rc.json_safe(entry) for name, entry in gates.items()}


def level1_gates(progs: ul.ModelPrograms, units: ul.TableUnits, states: Mapping[str, rd.FrameState020], tensors: Mapping[str, torch.Tensor],
                 conditions: Sequence[str], config: Configuration) -> dict[str, Any]:
    """The Level-1 identity on every run: the outcome-bearing primary-dose runs (gated) and the others (recorded)."""
    gates = {name: {"max": 0.0, "at": ""} for name in ("level1_outcome_bearing", "level1_secondary")}
    bearing = set(config.outcome_bearing)
    for group, pairs in units.pairs.items():
        for row, (t, f) in enumerate(pairs):
            frame, token = units.frames[f], units.tokens[t]
            state = states[frame.frame_id]
            for condition in conditions:
                dx3 = changes_from_raw(tensors[tensor_name(group, condition, "x3")][row], state.x3_all, frame)
                level1 = progs.readout.contrast(state, progs.readout.level1_detail(state, dict(dx3), l4_heads=True)["dh6"], progs.nouns)[progs.scorable].double()
                value = float((level1 - tensors[tensor_name(group, condition, "dc")][row]).abs().max())
                ul._worse(gates["level1_outcome_bearing" if condition in bearing else "level1_secondary"], value, f"{token['word']}|{frame.frame_id}|{condition}")
    return {name: rc.json_safe(entry) for name, entry in gates.items()}


def run_gates(progs: ul.ModelPrograms, confirmation: Confirmation025, units: ul.TableUnits, states: Mapping[str, rd.FrameState020],
              tensors: Mapping[str, torch.Tensor], vectors32: Mapping[int, Mapping[str, torch.Tensor]], config: Configuration) -> dict[str, Any]:
    """I1, I3 and I4 on every run, and the Level-1 identity on every run (the outcome-bearing primary-dose ones gated,
    the others recorded)."""
    out = {**identity_gates(progs, confirmation, units, states, tensors, vectors32), **level1_gates(progs, units, states, tensors, confirmation.conditions, config)}
    out["tolerances"] = {"I1": TOLERANCES["I1"], "I3": TOLERANCES["I3"], "I4": TOLERANCES["I4"], "level1_outcome_bearing": TOLERANCES["level1"]}
    return out


def enforce_gates(gates: Mapping[str, Mapping[str, Any]]) -> None:
    for name, tolerance in (("I1", "I1"), ("I3", "I3"), ("I4", "I4"), ("level1_outcome_bearing", "level1")):
        value = gates[name]["max"]
        if value is None or not value <= TOLERANCES[tolerance]:
            raise IncidentError(f"validity gate {name} failed: {value} at {gates[name]['at']} against {TOLERANCES[tolerance]:.0e}")


# ---------------------------------------------------------------------------
# 7. Per-cue quantities, the three statistics, the outcome.


def per_cue(units: ul.TableUnits, tensors: Mapping[str, torch.Tensor], states: Mapping[str, rd.FrameState020], conditions: Sequence[str]) -> dict[int, dict[str, Any]]:
    """For every cue and condition: ``ℓ`` (log of ``rr.cue_mse`` over its frames' cells), the MSE, the normalized MSE,
    ``D_attn`` (the fsum mean over its frames), and the per-template ``ℓ``."""
    out: dict[int, dict[str, Any]] = {}
    for t, token in enumerate(units.tokens):
        token_id = int(token["token_id"])
        entry: dict[str, Any] = {}
        for condition in conditions:
            cells: list[torch.Tensor] = []
            templates: dict[str, list[torch.Tensor]] = {}
            distances: list[float] = []
            for group, pairs in units.pairs.items():
                dc, c = tensors[tensor_name(group, condition, "dc")], tensors[tensor_name(group, condition, "C")]
                rows4, rows5 = tensors[tensor_name(group, condition, "rows4")], tensors[tensor_name(group, condition, "rows5")]
                for row, (tt, f) in enumerate(pairs):
                    if tt != t:
                        continue
                    frame = units.frames[f]
                    cell = rr.fresh_pair_cells(dc[row], c[row])
                    cells.append(cell)
                    templates.setdefault(frame.template_id, []).append(cell)
                    distances.append(routing_distance(rows4[row], rows5[row], states[frame.frame_id]))
            stacked = torch.stack(cells)
            mse = rr.cue_mse(stacked)
            if not math.isfinite(mse) or mse <= 0.0:
                raise IncidentError(f"cue {token['word']} under {condition}: the MSE {mse} is not finite and positive")
            nmse = math.fsum(stacked[:, rr.SSEC].tolist()) / math.fsum(stacked[:, rr.SQUARES].tolist())
            entry[condition] = {"mse": mse, "ell": math.log(mse), "nmse": nmse, "d_attn": math.fsum(distances) / len(distances), "frames": len(distances),
                                "ell_by_template": {template: math.log(rr.cue_mse(torch.stack(rows))) for template, rows in sorted(templates.items())}}
        out[token_id] = entry
    return out


def count_positive(values: Sequence[float]) -> int:
    """Strictly positive values; ties, zeros and NaN count against."""
    return sum(1 for value in values if value is not None and math.isfinite(float(value)) and float(value) > 0.0)


def outcome_label(a_pass: bool, b_pass: bool, g_pass: bool) -> str:
    if not a_pass:
        return "CAUSAL_EFFECT_NOT_ESTABLISHED"
    if not b_pass:
        return "DIRECTIONAL_BUT_NOT_DIRECTION_SPECIFIC"
    if not g_pass:
        return "READOUT_ERROR_CAUSAL_ROUTING_NOT_ESTABLISHED"
    return "NOUNNESS_DIRECTION_CAUSALLY_SHIFTS_ROUTING_AND_READOUT_ERROR"


def statistics(values: Mapping[int, Mapping[str, Any]], confirmation: Confirmation025, config: Configuration) -> dict[str, Any]:
    """``A``, ``B`` and ``G`` per cue, their positive counts against the threshold, and the outcome; everything computed
    from the saved measurements."""
    p = config.primary
    rows = []
    for token in confirmation.tokens:
        v = values[int(token["token_id"])]
        a = 0.5 * (v[f"noun+{p}"]["ell"] - v[f"noun-{p}"]["ell"])
        a_random = [0.5 * (v[f"rand{j}+{p}"]["ell"] - v[f"rand{j}-{p}"]["ell"]) for j in range(1, config.k_controls + 1)]
        b = a - math.fsum(abs(value) for value in a_random) / config.k_controls
        g = 0.5 * (v[f"noun+{p}"]["d_attn"] - v[f"noun-{p}"]["d_attn"])
        rows.append({"word": token["word"], "token_id": int(token["token_id"]), "stratum": token["stratum"], "A": a, "A_random": a_random, "B": b, "G": g})
    tests = {}
    for name in ("A", "B", "G"):
        count = count_positive([row[name] for row in rows])
        tests[name] = {"positive": count, "n": len(rows), "threshold": config.count_threshold, "result": "PASS" if count >= config.count_threshold else "FAIL",
                       "by_stratum": {stratum: count_positive([row[name] for row in rows if row["stratum"] == stratum]) for stratum in STRATA}}
    label = outcome_label(*(tests[name]["result"] == "PASS" for name in ("A", "B", "G")))
    tail = config.reference_tail()
    return rc.json_safe({"per_cue": rows, "A": tests["A"], "B": tests["B"], "G": tests["G"],
                         "outcome": {"label": label, "reading": SEMANTICS["outcomes"][label], "not_claimed": SEMANTICS["not_claimed"]},
                         "criterion": {"text": SEMANTICS["criterion"], "reference_tail": {"exact": f"{tail.numerator}/{tail.denominator}", "value": float(tail)}}})


# ---------------------------------------------------------------------------
# 8. The descriptive records (after the result is written; no outcome force).


def _median(values: Sequence[float]) -> float | None:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def descriptives(values: Mapping[int, Mapping[str, Any]], results: Mapping[str, Any], confirmation: Confirmation025, config: Configuration,
                 slope_024: float) -> dict[str, Any]:
    p, h = config.primary, config.half
    rows = results["per_cue"]
    out: dict[str, Any] = {"magnitudes": {name: {"mean": math.fsum(row[name] for row in rows) / len(rows), "median": _median([row[name] for row in rows])}
                                          for name in ("A", "B", "G")}}
    per_cue_rows = []
    for token in confirmation.tokens:
        v = values[int(token["token_id"])]
        even = 0.5 * (v[f"noun+{p}"]["ell"] + v[f"noun-{p}"]["ell"]) - v["base"]["ell"]
        per_cue_rows.append({
            "word": token["word"], "stratum": token["stratum"], "even_noun": even,
            "even_random": [0.5 * (v[f"rand{j}+{p}"]["ell"] + v[f"rand{j}-{p}"]["ell"]) - v["base"]["ell"] for j in range(1, config.k_controls + 1)],
            "plus_above_base": v[f"noun+{p}"]["ell"] > v["base"]["ell"], "minus_below_base": v[f"noun-{p}"]["ell"] < v["base"]["ell"],
            "A_half": 0.5 * (v[f"noun+{h}"]["ell"] - v[f"noun-{h}"]["ell"]), "A_plurality": 0.5 * (v[f"plur+{p}"]["ell"] - v[f"plur-{p}"]["ell"]),
            "A_nmse": 0.5 * (math.log(v[f"noun+{p}"]["nmse"]) - math.log(v[f"noun-{p}"]["nmse"])),
            "G_random": [0.5 * (v[f"rand{j}+{p}"]["d_attn"] - v[f"rand{j}-{p}"]["d_attn"]) for j in range(1, config.k_controls + 1)],
            "A_by_template": {template: 0.5 * (v[f"noun+{p}"]["ell_by_template"][template] - v[f"noun-{p}"]["ell_by_template"][template])
                              for template in v[f"noun+{p}"]["ell_by_template"]},
        })
    for row, result in zip(per_cue_rows, rows):
        row["G_minus_random"] = result["G"] - math.fsum(abs(g) for g in row["G_random"]) / config.k_controls
    out["per_cue"] = per_cue_rows
    out["counts"] = {"plus_above_base": sum(1 for row in per_cue_rows if row["plus_above_base"]), "minus_below_base": sum(1 for row in per_cue_rows if row["minus_below_base"]),
                     "A_half_positive": count_positive([row["A_half"] for row in per_cue_rows]),
                     "A_plurality_positive": count_positive([row["A_plurality"] for row in per_cue_rows]),
                     "A_nmse_positive": count_positive([row["A_nmse"] for row in per_cue_rows]),
                     "G_minus_random_positive": count_positive([row["G_minus_random"] for row in per_cue_rows])}
    templates = sorted({template for row in per_cue_rows for template in row["A_by_template"]})
    out["A_positive_by_template"] = {template: count_positive([row["A_by_template"][template] for row in per_cue_rows]) for template in templates}
    out["even_noun_mean"] = math.fsum(row["even_noun"] for row in per_cue_rows) / len(per_cue_rows)
    out["even_random_mean"] = math.fsum(value for row in per_cue_rows for value in row["even_random"]) / (len(per_cue_rows) * config.k_controls)
    stratum_size = {stratum: len(confirmation.stratum_tokens(stratum)) for stratum in STRATA}
    trigger = {stratum: math.ceil(config.count_threshold * size / config.n_cues) for stratum, size in stratum_size.items()}
    out["strata"] = {name: {stratum: {"positive": results[name]["by_stratum"][stratum], "of": stratum_size[stratum], "below_trigger":
                                      results[name]["result"] == "PASS" and results[name]["by_stratum"][stratum] < trigger[stratum]} for stratum in STRATA}
                     for name in ("A", "B", "G")}
    out["strata_trigger"] = trigger
    median_a = out["magnitudes"]["A"]["median"]
    out["causal_fraction"] = {"median_A": median_a, "slope_024_exposed": slope_024, "odd": config.primary_odd,
                              "fraction": None if median_a is None else median_a / (slope_024 * config.primary_odd)}
    return rc.json_safe(out)


def ladder(progs: ul.ModelPrograms, units: ul.TableUnits, states: Mapping[str, rd.FrameState020], tensors: Mapping[str, torch.Tensor],
           conditions: Sequence[str]) -> dict[str, Any]:
    """Per named condition, the share of ``Σ (Δc − C)²`` removed by making block-5 attention exact (C5), then block-4
    attention too (L1), pooled over every pair (descriptive; 024's ladder on the nounness and baseline runs)."""
    readout = progs.readout
    out: dict[str, Any] = {}
    for condition in conditions:
        sums = {"C": [], "C5": [], "L1": []}
        for group, pairs in units.pairs.items():
            for row, (t, f) in enumerate(pairs):
                frame = units.frames[f]
                state = states[frame.frame_id]
                dx3 = changes_from_raw(tensors[tensor_name(group, condition, "x3")][row], state.x3_all, frame)
                y, c = tensors[tensor_name(group, condition, "dc")][row].double(), tensors[tensor_name(group, condition, "C")][row].double()
                c5 = readout.contrast(state, readout.level1_detail(state, dict(dx3), l4_heads=False)["dh6"], progs.nouns)[progs.scorable].double()
                l1 = readout.contrast(state, readout.level1_detail(state, dict(dx3), l4_heads=True)["dh6"], progs.nouns)[progs.scorable].double()
                sums["C"].append(float(((y - c) ** 2).sum()))
                sums["C5"].append(float(((y - c5) ** 2).sum()))
                sums["L1"].append(float(((y - l1) ** 2).sum()))
        total = {name: math.fsum(values) for name, values in sums.items()}
        out[condition] = {"sse": total, "block5_attention": (total["C"] - total["C5"]) / total["C"] if total["C"] > 0 else None,
                          "block4_attention": (total["C5"] - total["L1"]) / total["C"] if total["C"] > 0 else None}
    return rc.json_safe(out)


# ---------------------------------------------------------------------------
# 9. The results state, the phase rules, the report.

RESULTS_SCHEMA_VERSION = 1
PHASES = ("validate", "freeze", "lock", "confirm", "report")
STATE_PHASES = ("lock", "confirm", "report")


def new_results_state(*, digests: Mapping[str, str], protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any], config: Configuration) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {"schema_version": RESULTS_SCHEMA_VERSION, "experiment": EXPERIMENT,
            "run_id": pm.sha256_text(pm.canonical_json(dict(digests)) + protocol_code_commit + pm.utc_now())[:16], "created_at": pm.utc_now(),
            "inputs": {key: digests[key] for key in DIGEST_KEYS}, "module_blobs": dict(FROZEN_BLOBS), "design": dict(DESIGN), "plan": dict(PLAN),
            "configuration": config.to_json(), "protocol_code_commit": protocol_code_commit, "git_dirty": False,
            "model": {"model_id": models_module.PYTHIA_70M.model_id, "revision": models_module.PYTHIA_70M.revision}, "versions": dict(versions),
            "phases": {phase: {"status": "not_started"} for phase in STATE_PHASES}, "executed_prompt_keys": [], "executed_noun_keys": [],
            "confirmation_025": None, "lock": None, "confirmation": None, "report": None}


def assert_phase_allowed(phase: str, state: Mapping[str, Any] | None) -> None:
    if phase == "lock":
        if state is None:
            return
        if state["phases"]["lock"]["status"] == "complete":
            raise PhaseError("lock already written; a new candidate lock requires a new protocol version")
        if state["phases"]["lock"].get("incidents"):
            raise PhaseError("a lock incident is recorded; lock is refused until the reviewer decides (a new protocol version)")
        return
    if state is None:
        raise PhaseError(f"{phase} requires the results state the lock creates")
    status = {name: entry["status"] for name, entry in state["phases"].items()}
    if phase == "confirm":
        if status["lock"] != "complete":
            raise PhaseError("confirm requires the lock phase")
        if status["confirm"] != "not_started":
            raise PhaseError("confirm already started; it runs once, never resumes, and a second attempt requires a new protocol version")
        if state["phases"]["confirm"].get("incidents"):
            raise PhaseError("an I7′ or patch-path incident is recorded; confirm is refused until the reviewer decides (a new protocol version)")
    elif phase == "report":
        confirmation = state.get("confirmation") or {}
        if status["confirm"] != "complete" and not confirmation.get("incident") and not state["phases"]["confirm"].get("incidents"):
            raise PhaseError("report requires a completed confirm or a recorded confirm incident")
    else:
        raise PhaseError(f"unknown phase {phase}")


def record_nouns(state: dict[str, Any], nouns: Sequence[Any]) -> None:
    keys = set(state["executed_noun_keys"])
    keys.update(noun.key for noun in nouns)
    state["executed_noun_keys"] = sorted(keys)


def render_report(state: Mapping[str, Any]) -> str:
    phases = ", ".join(f"{name} {state['phases'][name]['status']}" for name in STATE_PHASES)
    lines = ["# Experiment 025 — report", "", f"- Run `{state['run_id']}`; phases: {phases}",
             f"- Design revision {state['design']['revision']} (`{state['design']['commit']}`), plan revision {state['plan']['revision']} (`{state['plan']['commit']}`); "
             f"configuration `{state['configuration']['name']}`", ""]
    for name in STATE_PHASES:
        for entry in state["phases"][name].get("incidents", []):
            lines.append(f"- **{name} incident** at `{entry['commit']}`: {entry['message']}")
    confirmation = state.get("confirmation") or {}
    incident = confirmation.get("incident")
    if incident:
        lines += [f"- **Confirmation incident** at `{incident['commit']}`: {incident['message']} — the one-shot confirmation is consumed; nothing is retried", "",
                  "## The outcome", "", "**`NOT_INTERPRETABLE`** — an incident carries no result.", ""]
        return "\n".join(lines)
    results = confirmation.get("results")
    if not results:
        if state["phases"]["confirm"].get("incidents"):
            lines += ["", "## The outcome", "", "**`NOT_INTERPRETABLE`** — a confirm incident before the ledger; no fresh prompt ran and no result exists."]
        lines.append("")
        return "\n".join(lines)
    label = results["outcome"]
    lines += ["## The outcome", "", f"**`{label['label']}`** — {label['reading']}.", "", "Not claimed by any label:"]
    lines += [f"- {item}" for item in label["not_claimed"]]
    lines += ["", f"- The criterion: {results['criterion']['text']}; reference tail {results['criterion']['reference_tail']['exact']} = "
              f"{results['criterion']['reference_tail']['value']!r}", "", "| statistic | positive | of | threshold | result | adjectives | nouns |", "|---|---|---|---|---|---|---|"]
    for name in ("A", "B", "G"):
        test = results[name]
        lines.append(f"| {name} | {test['positive']} | {test['n']} | {test['threshold']} | **{test['result']}** | {test['by_stratum']['adjective']} | "
                     f"{test['by_stratum']['noun']} |")
    lines += ["", "| stratum | word | A | B | G |", "|---|---|---|---|---|"]
    lines += [f"| {row['stratum']} | {row['word']} | {row['A']:+.6f} | {row['B']:+.6f} | {row['G']:+.6f} |" for row in results["per_cue"]]
    gates = confirmation.get("gates") or {}
    lines += ["", "## Validity (all passed for a result to exist)", ""]
    lines.append("- " + ", ".join(f"{name} max {gates[name].get('max')} (at {gates[name].get('at')})" for name in ("I1", "I3", "I4", "level1_outcome_bearing") if name in gates))
    c_check = confirmation.get("c_recompute") or {}
    lines.append(f"- C recomputed from the saved Δx3: bit for bit {c_check.get('bitwise_equal')} over {c_check.get('runs')} runs")
    lines.append(f"- The prompt accounting: {confirmation.get('accounting')}")
    lines.append(f"- The patch-path check: passed {((state['phases']['confirm'].get('patch_path_check') or {}).get('passed'))}")
    descriptive = confirmation.get("descriptives") or {}
    if descriptive.get("records"):
        records = descriptive["records"]
        lines += ["", "## Descriptive records (no outcome force)", "", f"- Magnitudes: {records['magnitudes']}", f"- Counts: {records['counts']}",
                  f"- A positive by template: {records['A_positive_by_template']}", f"- Even components: noun {records['even_noun_mean']!r}, random {records['even_random_mean']!r}",
                  f"- Strata: {records['strata']}", f"- Causal fraction: {records['causal_fraction']}"]
    if descriptive.get("ladder"):
        for condition, entry in descriptive["ladder"].items():
            lines.append(f"- Ladder {condition}: block-5 attention {entry['block5_attention']}, block-4 attention {entry['block4_attention']}")
    for name, failure in sorted((descriptive.get("failures") or {}).items()):
        lines.append(f"- The descriptive record {name} was not computed ({failure['type']}: {failure['message']}); the result stands")
    lines.append("")
    return "\n".join(lines)
