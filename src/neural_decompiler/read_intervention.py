"""Experiment 101: is the read component of the residual entering layer 3 at the cue position causal for the
``one``/``two`` singular–plural choice that ``L03.H04`` transports?

Implements design revision 3 (``a87e284``) as the reviewer decided it (``516df20``), through implementation plan
revision 3 (``96f0cc8``) and the reviewer's decision on G4 (``1e48c1c``).

**The variable.** The covector of ``L03.H04``'s P1 read, from Experiment 011's lock (``read_weight = γ₃⊙m``)::

    w_c = γ₃⊙m − mean(γ₃⊙m)·𝟙      r(x) = ⟨x, w_c⟩      ŵ = w_c/‖w_c‖

at ``RESID_PRE.L3``, the cue position ``p_c``, in coordinated frames, where ``p_t = p_c + 1``.

**The arms.** For the reference prompt (``one``) and a plural prompt (cue ``w``) of the same frame::

    Δx = x_w − x_ref      Δr = ⟨Δx, w_c⟩      read = (Δr/‖w_c‖²)·w_c      s = |Δr|/‖w_c‖
    off_c = the centred off-read part of Δx      û_off = off_c/‖off_c‖

Removal ``x_w − read``, installation ``x_ref + read``, the primary controls ``x_w − s·û_off`` and
``x_ref + s·û_off``, the doubled and reversed changes, the half dose, the whole-position swaps, the isotropic controls
and the head-only arms. Each is one forward pass with all 79 contrasts read at ``p_t``.

**The prediction.** 023's completed program predicts every main arm's ``Δc`` from the stage-1 reference states and the
weights alone. The table is committed before any target prompt runs, and the barrier recomputes it bit for bit.

**The outcome.** R and I by Holm at 0.025 on the two count p-values, each with a median floor of 0.05; then D; then P.
A gate failure after the measurements is an integrity failure under the incident procedure: no frame is dropped,
replaced or ignored, and no outcome is scored.

The module *calls* the pinned modules and never edits them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from neural_decompiler import attention_paths as ap
from neural_decompiler import attention_patterns as atp
from neural_decompiler import block0_completion as b0c
from neural_decompiler import capture as capture_module
from neural_decompiler import components as components_module
from neural_decompiler import cue_rotation as cr
from neural_decompiler import cue_suppression as cs
from neural_decompiler import encoding_read as er
from neural_decompiler import frame_channels as fch
from neural_decompiler import head_pattern as hp
from neural_decompiler import head_transport as ht
from neural_decompiler import interventions as interventions_module
from neural_decompiler import layer_correction as lc
from neural_decompiler import models as models_module
from neural_decompiler import plural_mechanism as pm
from neural_decompiler import read_assembly as ra
from neural_decompiler import readout_calibration as rc
from neural_decompiler import readout_decompilation as rd
from neural_decompiler import readout_routing as rr
from neural_decompiler import upstream_localization as ul

PhaseError = pm.PhaseError
IncidentError = pm.IncidentError

# ---------------------------------------------------------------------------
# Paths, design, plan.

EXPERIMENT = "101"
EXPERIMENT_DIR = "experiments/101-read-component-intervention"
CONFIRMATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/confirmation-v1.json"
CALIBRATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/calibration-v1.json"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
PREREGISTRATION_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration.md"
STAGE1_DIR = f"{EXPERIMENT_DIR}/stage1"
STAGE1_STATES_DATA = f"{STAGE1_DIR}/states.f64"
STAGE1_STATES_INDEX = f"{STAGE1_DIR}/states.json"
STAGE1_TABLE = f"{STAGE1_DIR}/table.json"
STAGE1_PATHS = (STAGE1_STATES_DATA, STAGE1_STATES_INDEX, STAGE1_TABLE)
DESIGN = {"path": "docs/superpowers/specs/2026-10-01-experiment-101-read-component-intervention-design.md", "revision": 3, "commit": "a87e284",
          "decision": "516df20"}
PLAN = {"path": "docs/superpowers/plans/2026-10-05-experiment-101-read-component-intervention-plan.md", "revision": 3, "commit": "96f0cc8",
        "decision": "1e48c1c"}

# ---------------------------------------------------------------------------
# The frozen dependencies, by git blob at 516df20 (sha1(b"blob <len>\0" + bytes), computed without git). Written out
# literally; every other module under src/ is covered by the scientific-path rule after the lock.

FROZEN_BLOBS = {
    "upstream_localization.py": "465856962aa380747d1a4f1338d1d2762d03c9f9",
    "block0_completion.py": "16d310fc7fab5599ec83b8bd8162fb9613f8dad2",
    "readout_decompilation.py": "caa73b40192f4c910dc63371bd19db75a3258339",
    "readout_calibration.py": "9107da975128d9b0383f3b346695675104e4cde9",
    "plural_mechanism.py": "d39da8a8d9931005d411258bcddbb7f9beed35e4",
    "readout_routing.py": "e6cb37767d1d06c6ff40804a88eab569723afdb5",
    "cue_rotation.py": "622aa832ae29047c3060d989928236320374e227",
    "models.py": "b1c6f03379af7e0918d0d1a6460a264651603fb2",
    "layer_correction.py": "04df0cc9a20093cc48ee5ef62da7f206bf1a3186",
    "head_pattern.py": "386682fe0a47d093dfbee2507afc1df61811ab15",
    "frame_channels.py": "c95d6fb4c98b8069c26ec46e9d8f85eb927bb9e8",
    "attention_patterns.py": "3f5acd65397bd11723ba6fc2ec4253f1d5c81943",
    "encoding_read.py": "cab99c942de970332e726a5626b584242d4cf600",
    "head_transport.py": "936093a4e82c83e299a65a7c85529a7226f80c95",
    "read_assembly.py": "4820191e0e067b382b8f8807279405a7589f7059",
    "cue_suppression.py": "21253ee61c04796966dec4733909476a09404c0c",
    "attention_paths.py": "42f8d1588f1f960f60498854a9ccb911bc4e427a",
    "interventions.py": "3a1e4186aeb3c8ae9625af336e8da7d2818505c7",
    "components.py": "28af3f3d8ba5512b6850f485f85ff88983bed24e",
    "capture.py": "4a4a78880b1c2ee410486d5bcfc1993ba2e9d316",
}
_FROZEN_MODULES = {"upstream_localization.py": ul, "block0_completion.py": b0c, "readout_decompilation.py": rd, "readout_calibration.py": rc,
                   "plural_mechanism.py": pm, "readout_routing.py": rr, "cue_rotation.py": cr, "models.py": models_module, "layer_correction.py": lc,
                   "head_pattern.py": hp, "frame_channels.py": fch, "attention_patterns.py": atp, "encoding_read.py": er, "head_transport.py": ht,
                   "read_assembly.py": ra, "cue_suppression.py": cs, "attention_paths.py": ap, "interventions.py": interventions_module,
                   "components.py": components_module, "capture.py": capture_module}


def module_blobs() -> dict[str, str]:
    return {name: rc.program_blob_sha1(Path(module.__file__)) for name, module in _FROZEN_MODULES.items()}


def assert_frozen_blobs() -> dict[str, str]:
    """Every pinned module is byte for byte the one at the design commit."""
    actual = module_blobs()
    differing = sorted(name for name, blob in FROZEN_BLOBS.items() if actual.get(name) != blob)
    if differing:
        raise PhaseError(f"frozen modules changed: {differing}; Experiment 101 runs the pinned programs only")
    return actual


def own_blob() -> str:
    return rc.program_blob_sha1(Path(__file__))


# ---------------------------------------------------------------------------
# The inherited inputs beyond 025's chain: 025's committed confirmation file (its manifest joins the forbidden set; its
# cues, candidate lists and blocked sets feed the freeze's exclusions).

INHERITED_025_PATH = cr.CONFIRMATION_RELATIVE_PATH
INHERITED_025 = {
    "confirmation_file_sha256": "54c8947c1f6b71d23de7a1f20cad1891571a16bd87849910fcd036825694b64a",
    "confirmation_content_sha256": "6eca7024fe3fdcde000b6dc6fe84ef9c547f1f92bd456b51cd73dfdd3443fea4",
}


def verify_025_inputs(root: Path) -> dict[str, str]:
    path = Path(root) / INHERITED_025_PATH
    if not path.exists() or rc.file_sha256(path) != INHERITED_025["confirmation_file_sha256"]:
        raise PhaseError(f"Experiment 025's committed confirmation file {INHERITED_025_PATH} is missing or not the reviewed file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("content_sha256") != rc.content_digest(payload) or payload["content_sha256"] != INHERITED_025["confirmation_content_sha256"]:
        raise PhaseError("Experiment 025's confirmation file does not verify against its reviewed content digest")
    return {"025_confirmation_file": INHERITED_025["confirmation_file_sha256"], "025_confirmation_content": INHERITED_025["confirmation_content_sha256"]}


def load_025(root: Path) -> dict[str, Any]:
    verify_025_inputs(root)
    return json.loads((Path(root) / INHERITED_025_PATH).read_text(encoding="utf-8"))


DIGEST_KEYS = (*cr.DIGEST_KEYS, "025_confirmation_file", "025_confirmation_content")


def base_digests(inputs: ul.FrozenInputs, digests_022: Mapping[str, str], digests_023: Mapping[str, str], digests_024: Mapping[str, str],
                 prior: Mapping[str, Any], digests_025: Mapping[str, str]) -> dict[str, str]:
    return {**cr.base_digests(inputs, digests_022, digests_023, digests_024, prior), **dict(digests_025)}


# ---------------------------------------------------------------------------
# The frames (design §11), verbatim, in candidate order.

TEMPLATE = "coordinated-adjective"
FRAME_CANDIDATES = (
    "Aria and Ben carried {cue} large", "Cora and Dan folded {cue} soft", "Eli and Faye stacked {cue} gray", "Gus and Hope washed {cue} sticky",
    "Ian and June sorted {cue} tiny", "Kit and Lou counted {cue} clean", "Max and Nell wrapped {cue} fine", "Oscar and Paula dried {cue} damp",
    "Quinn and Rex cleaned {cue} messy", "Sara and Tom fixed {cue} bent", "Uma and Vera moved {cue} full", "Walt and Yara lifted {cue} wet",
    "Zack and Abby filled {cue} deep", "Beth and Carl peeled {cue} bad", "Dora and Erik sliced {cue} thin", "Gina and Hank cooked {cue} sour",
    "Ivy and Jack grilled {cue} lean", "Kate and Luke served {cue} good", "Mia and Ned planted {cue} pale", "Olga and Pete picked {cue} neat",
    "Rita and Seth hugged {cue} tired", "Tess and Uri labeled {cue} dry", "Val and Wes measured {cue} cool", "Ada and Bo watched {cue} happy",
    "Cleo and Dev mended {cue} long", "Gil and Jo sewed {cue} cozy", "Kim and Lia carved {cue} dull", "Mo and Nia built {cue} tidy",
    "Pia and Ray painted {cue} open", "Sol and Ty repaired {cue} dead", "Ann and Bill tied {cue} loose", "Chris and Di pulled {cue} broad",
    "Ed and Fran pushed {cue} slim", "Greg and Holly sent {cue} sleek", "Jim and Kara brought {cue} bold", "Liz and Mark stored {cue} sealed",
    "Nate and Olive fed {cue} sick", "Phil and Rosa chose {cue} bland", "Ron and Sue grew {cue} lush", "Tim and Vicky calmed {cue} angry",
    "Adam and Betty drew {cue} quick", "Clare and Doug wrote {cue} funny", "Dean and Ellen sang {cue} slow", "Fred and Gwen told {cue} odd",
    "Hugh and Iris heard {cue} strange", "Jake and Karen made {cue} simple", "Lisa and Matt bought {cue} decent", "Neil and Olga found {cue} real",
    "Paul and Rita adopted {cue} lost", "Ray and Sally picked {cue} early", "Sam and Tina ate {cue} stale", "Ted and Una hired {cue} smart",
    "Vince and Wendy met {cue} nice", "Alan and Bea fed {cue} cute", "Carl and Dina drew {cue} pretty", "Eric and Flora fixed {cue} torn",
    "Gary and Helen trained {cue} keen", "Ian and Jill chased {cue} fast", "Ken and Laura cleaned {cue} worn", "Leo and Mona scrubbed {cue} stained",
)
EXPECTED_PICKS = tuple(range(1, 37))  # the candidate numbers both reviews reproduced with the pinned tokenizer
P_C_RANGE = (4, 7)
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
                "hundred", "thousand", "million", "billion", "dozen", "half", "single", "double", "both", "several", "few", "many", "couple", "pair")
FRAME_ID_FORMAT = "coordinated-adjective-101-{number:02d}"

# ---------------------------------------------------------------------------
# The arms, in fixed run order within a frame. ``REF`` runs first; then each cue in the configuration's order, with
# the primary cue's 32 arms and each stratum cue's 11. ``prompt`` names the prompt the arm runs: the reference prompt
# (``ref``) or the cue's own (``cue``).


@dataclass(frozen=True)
class ArmSpec:
    name: str
    prompt: str  # "ref" or "cue"
    kind: str  # plain, noop, read, off, swap, iso, head_full, head_value
    coefficient: float = 0.0  # read: x' = x + c·read; off/iso: x' = x + c·s·u
    j: int | None = None  # the isotropic direction's index (1-based)

    @property
    def residual_edit(self) -> bool:
        return self.kind in ("noop", "read", "off", "swap", "iso")

    @property
    def head(self) -> bool:
        return self.kind in ("head_full", "head_value")


REF_ARM = ArmSpec("REF", "ref", "plain")
MAIN_ARMS = (
    ArmSpec("PL", "cue", "plain"),
    ArmSpec("NEC-READ", "cue", "read", -1.0), ArmSpec("SUF-READ", "ref", "read", 1.0),
    ArmSpec("NEC-OFF", "cue", "off", -1.0), ArmSpec("SUF-OFF", "ref", "off", 1.0),
    ArmSpec("NEC-AMP", "cue", "read", 1.0), ArmSpec("SUF-NEG", "ref", "read", -1.0),
    ArmSpec("NEC-HALF", "cue", "read", -0.5), ArmSpec("SUF-HALF", "ref", "read", 0.5),
    ArmSpec("PC-SWAP-PL", "cue", "swap"), ArmSpec("PC-SWAP-REF", "ref", "swap"),
)
NOOP_ARM = ArmSpec("NOOP", "cue", "noop")
HEAD_ARMS = (ArmSpec("HEAD-NEC", "cue", "head_full", -1.0), ArmSpec("HEAD-SUF", "ref", "head_full", 1.0),
             ArmSpec("HEAD-NEC-V", "cue", "head_value", -1.0), ArmSpec("HEAD-SUF-V", "ref", "head_value", 1.0))
PREDICTED_ARMS = tuple(arm.name for arm in MAIN_ARMS)  # the arms the program's table predicts (design §5)
ISO_TAG = "101|iso|{frame_id}|{j}"


def primary_arms(k_iso: int) -> tuple[ArmSpec, ...]:
    """The primary cue's arms in run order: PL, NOOP, the ten main edits, the isotropic controls, the head arms."""
    iso = tuple(ArmSpec(f"NEC-ISO-{j}", "cue", "iso", -1.0, j) for j in range(1, k_iso + 1)) + \
        tuple(ArmSpec(f"SUF-ISO-{j}", "ref", "iso", 1.0, j) for j in range(1, k_iso + 1))
    return (MAIN_ARMS[0], NOOP_ARM, *MAIN_ARMS[1:], *iso, *HEAD_ARMS)


def stratum_arms() -> tuple[ArmSpec, ...]:
    return MAIN_ARMS


def tag_of(arm: str, cue_label: str | None) -> str:
    """``REF`` and ``S1`` stand alone; every other tag carries its cue, because the stratum's ``SUF-*`` and
    ``PC-SWAP-REF`` arms run the same reference prompt as the primary cue's."""
    return arm if arm in ("REF", "S1") else f"{arm}@{cue_label}"


# ---------------------------------------------------------------------------
# The statistics' constants (design §7 and §14; frozen exactly).

WEAK_SHIFT = -1.0  # nats: a frame is weak for a cue when that cue's Δc_full > −1 (strict)
HOLM_ALPHA = Fraction(1, 40)  # 0.025, the family level for R and I
MEDIAN_FLOOR = 0.05
TAU_CAP = 0.20
TAU_FLOOR = 0.02
HEAD_SHARE_GUARD = 0.05
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20261001
WORST = {"N": -math.inf, "S": -math.inf, "A_amp": math.inf, "A_neg": -math.inf, "head_share": -math.inf, "iso_margin": -math.inf,
         "F_head_rem": math.inf, "F_head_ins": -math.inf}
DIRECTION_FREE = ("F_half_rem", "F_half_ins", "F_swap_pl", "F_swap_ref", "additivity", "P_error_rem", "P_error_ins")
SUMMARY_FREE = DIRECTION_FREE[:5]  # the direction-free fractions summarized per cue (P's errors are summarized with the program's)
HEAD_COUNTS = {"F_head_rem": ("F_head_rem", -1, 1.0), "F_head_rem_v": ("F_head_rem", -1, 1.0), "F_head_ins": ("F_head_ins", 1, 0.0),
               "F_head_ins_v": ("F_head_ins", 1, 0.0)}  # name -> (WORST key, the predicted sign against the reference value, the reference value)


def binomial_tail(n: int, k: int) -> Fraction:
    """``P(X ≥ k)`` for ``X ~ Binomial(n, 1/2)``, as an exact rational."""
    return Fraction(sum(math.comb(int(n), j) for j in range(int(k), int(n) + 1)), 2 ** int(n))


def holm_thresholds(n: int, alpha: Fraction = HOLM_ALPHA) -> tuple[int, int]:
    """The smallest counts whose exact tails are at most ``alpha/2`` and ``alpha``: 26 and 25 of 36."""
    def smallest(level: Fraction) -> int:
        for k in range(int(n) + 1):
            if binomial_tail(n, k) <= level:
                return k
        return int(n) + 1
    return smallest(alpha / 2), smallest(alpha)


def nearest_rank(n: int, percentile: int) -> int:
    """The nearest-rank position (1-based) of a percentile of ``n`` sorted values: 33 for the 90th of 36."""
    return max(1, math.ceil(Fraction(percentile, 100) * int(n)))


# ---------------------------------------------------------------------------
# The configuration. Production is a literal, immutable object; a test world passes a smaller one explicitly.


@dataclass(frozen=True)
class Configuration:
    """Every size the protocol fixes. Every artifact records the configuration it was written under, and one written
    under one configuration is refused under another."""

    name: str
    n_frames: int
    frame_candidates: tuple[str, ...]
    expected_picks: tuple[int, ...]  # 1-based candidate numbers, in order
    reference: tuple[str, int]  # the reference prompt's label and cue id
    cues: tuple[tuple[str, int], ...]  # (label, id) of each plural cue; the first is the primary one
    k_iso: int
    n_scored_nouns: int
    exposed_frames: int  # the exposed coordinated frames calibration runs
    d_threshold: int
    p_threshold: int
    precondition: int
    tau_rank: int
    max_overshoot_frames: int
    max_weak_frames: int

    def __post_init__(self) -> None:
        if len(self.expected_picks) != self.n_frames or len(set(self.expected_picks)) != self.n_frames:
            raise ValueError("the expected picks must name n_frames distinct candidates")
        if not all(1 <= number <= len(self.frame_candidates) for number in self.expected_picks):
            raise ValueError("an expected pick is not a candidate number")
        if len(self.cues) < 1 or len({label for label, _ in self.cues}) != len(self.cues) or len({i for _, i in self.cues}) != len(self.cues):
            raise ValueError("the cues need distinct labels and ids, the primary one first")
        if self.reference[1] in {i for _, i in self.cues} or self.reference[0] != "ref":
            raise ValueError("the reference prompt is labelled ref and its cue is none of the plural cues")
        if self.k_iso < 1:
            raise ValueError("at least one isotropic direction")
        if self.tau_rank != nearest_rank(self.n_frames, 90):
            raise ValueError("tau's rank is the nearest-rank 90th percentile of the frames")
        if self.name == "production":
            better, other = holm_thresholds(self.n_frames)
            if (self.n_frames, better, other, self.d_threshold, self.p_threshold, self.precondition, self.tau_rank) != (36, 26, 25, 25, 27, 30, 33):
                raise ValueError("the production numbers are design §7's, frozen exactly")
            if self.expected_picks != EXPECTED_PICKS or self.frame_candidates != FRAME_CANDIDATES:
                raise ValueError("the production frames are design §11's")

    @property
    def primary(self) -> tuple[str, int]:
        return self.cues[0]

    @property
    def stratum(self) -> tuple[tuple[str, int], ...]:
        return self.cues[1:]

    @property
    def cue_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.cues)

    def cue_id(self, label: str) -> int:
        return dict(self.cues)[label]

    def arms_for(self, label: str) -> tuple[ArmSpec, ...]:
        return primary_arms(self.k_iso) if label == self.primary[0] else stratum_arms()

    def run_order(self) -> tuple[tuple[ArmSpec, str | None], ...]:
        """``(arm, cue label)`` in run order within one frame: REF, then each cue's arms."""
        return ((REF_ARM, None),) + tuple((arm, label) for label in self.cue_labels for arm in self.arms_for(label))

    def tags(self) -> tuple[str, ...]:
        return tuple(tag_of(arm.name, label) for arm, label in self.run_order())

    @property
    def runs_per_frame(self) -> int:
        return len(self.run_order())

    @property
    def holm(self) -> tuple[int, int]:
        return holm_thresholds(self.n_frames)

    def to_json(self) -> dict[str, Any]:
        better, other = self.holm
        return {"name": self.name, "n_frames": self.n_frames, "frame_candidates": list(self.frame_candidates), "expected_picks": list(self.expected_picks),
                "reference": list(self.reference), "cues": [list(cue) for cue in self.cues], "k_iso": self.k_iso, "n_scored_nouns": self.n_scored_nouns,
                "exposed_frames": self.exposed_frames, "holm_thresholds": [better, other], "d_threshold": self.d_threshold, "p_threshold": self.p_threshold,
                "precondition": self.precondition, "tau_rank": self.tau_rank, "max_overshoot_frames": self.max_overshoot_frames,
                "max_weak_frames": self.max_weak_frames, "runs_per_frame": self.runs_per_frame, "tags": list(self.tags())}


PRODUCTION = Configuration(name="production", n_frames=36, frame_candidates=FRAME_CANDIDATES, expected_picks=EXPECTED_PICKS, reference=("ref", 581),
                           cues=(("two", 767), ("three", 1264), ("many", 1142)), k_iso=8, n_scored_nouns=79, exposed_frames=36, d_threshold=25,
                           p_threshold=27, precondition=30, tau_rank=33, max_overshoot_frames=6, max_weak_frames=6)

# ---------------------------------------------------------------------------
# Tolerances (design §8; G4 as the reviewer decided it on 2026-10-05). No code path changes them: calibration may test
# them but never widen them, and the lock binds them.

TOLERANCES = {
    "G1_read_weight": 1e-9,  # w_c from the weights against the centred 011 read_weight, max abs
    "G1_basis": 1e-12,  # v_j unit, mean-zero, orthogonal to w_c; the basis (𝟙/√d, ŵ) orthonormal (float64)
    "G2_read": 1e-4,  # the realized r(x') against its target
    "G2_length": 1e-4,  # the centred length of x' − x against its intended length, relative
    "G2_direction": 1e-4,  # parallel (read arms) or orthogonal (off and isotropic arms) to ŵ
    "G2_u_off32": 1e-6,  # the float32 û_off: mean, norm − 1, ⟨·, ŵ⟩
    "G3_pass_through": 1e-4,  # RESID_PRE.L4(p_c) = x' + Σ_h L03.H0h(p_c) + b_O + L03.MLP(p_c), max abs
    "G4_I3": 1e-4,  # 023's I3 on each plain pair, relative
    "G4_I4": 1e-3,  # 023's I4 on each G4 run, nats, max abs
    "G4_level1": 2e-2,  # 020's Level-1 identity in 025's contrast form, nats, max abs
    "G4_readout": rd.READOUT_IDENTITY_TOLERANCE,  # 2e-2
    "G4_logit": rd.LOGIT_IDENTITY_TOLERANCE,  # 2e-2
    "G5_row": 1e-4,  # the head row recomputed from the unedited inputs, max abs
    "G5_output": 1e-3,  # the head output along d̂_T, max abs
    "environment": rc.LOCKED_STATE_TOLERANCE,  # a re-captured exposed reference state against 020's (calibration)
}
NORM_FLOOR = 1e-12

# ---------------------------------------------------------------------------
# The outcome, its scope and the frozen wording (design §7, §10, §14).

OUTCOMES = ("NOT_INTERPRETABLE", "BEHAVIOR_PRECONDITION_NOT_MET", "READ_COMPONENT_EFFECTS_NOT_ESTABLISHED", "READ_REMOVAL_EFFECT_INSTALLATION_NOT_ESTABLISHED",
            "READ_INSTALLATION_EFFECT_REMOVAL_NOT_ESTABLISHED", "READ_COMPONENT_EFFECTS_DIRECTION_NOT_ESTABLISHED", "READ_COMPONENT_EFFECTS_PROGRAM_NOT_EVALUABLE",
            "READ_COMPONENT_EFFECTS_PROGRAM_NOT_CONFIRMED", "READ_COMPONENT_EFFECTS_AS_PREDICTED")
OUTCOME_TABLE = (
    ("any gate fails, or an incident is recorded", "NOT_INTERPRETABLE"),
    ("the precondition fails", "BEHAVIOR_PRECONDITION_NOT_MET"),
    ("neither R nor I passes", "READ_COMPONENT_EFFECTS_NOT_ESTABLISHED"),
    ("R passes, I does not", "READ_REMOVAL_EFFECT_INSTALLATION_NOT_ESTABLISHED"),
    ("I passes, R does not", "READ_INSTALLATION_EFFECT_REMOVAL_NOT_ESTABLISHED"),
    ("R and I pass, D fails", "READ_COMPONENT_EFFECTS_DIRECTION_NOT_ESTABLISHED"),
    ("R, I and D pass, P is uninformative", "READ_COMPONENT_EFFECTS_PROGRAM_NOT_EVALUABLE"),
    ("R, I and D pass, P fails", "READ_COMPONENT_EFFECTS_PROGRAM_NOT_CONFIRMED"),
    ("R, I, D and P pass", "READ_COMPONENT_EFFECTS_AS_PREDICTED"),
)
SEMANTICS = {
    "scope": "the read component of the residual entering layer 3 at the cue position, as every reader of that residual sees it, in coordinated "
             "frames, for the cue pair one/two from which the read was estimated",
    "non_claims": ["no label says that the component is unused, or that it has no causal role",
                   "no label assigns the effect to L03.H04 alone: the residual edit reaches every component that reads it",
                   "no label generalizes beyond the one/two pair, this site, this checkpoint and these frames"],
    "hierarchy": {
        "primary": "does manipulating the decoded read direction at the cue-position residual causally alter the one/two singular–plural choice in "
                   "new coordinated frames? Answered by R, I and D against the matched non-read control, with the labels above",
        "secondary": "is L03.H04 specifically the component that mediates the effect? The head-only arms answer it descriptively; they cannot change "
                     "the primary label",
        "generalization": "does the effect extend to three and many? Secondary only; the stratum cannot rescue a failed primary result",
        "head_not_supported": "if the primary passes but the head-only evidence does not support the head: the decoded residual direction is causal "
                              "at this site, within the scope, but its effect cannot be assigned uniquely to L03.H04",
    },
    "families": {
        "R": "N_f = F_rem_off − F_rem > 0: the read removal takes away more of the shift than an equal-length off-read removal",
        "I": "S_f = F_ins − F_ins_off > 0: installing the read part produces more of the shift than an equal-length off-read installation",
        "D": "A_amp = Δc(NEC-AMP) − Δc_full < 0 and A_neg = Δc(SUF-NEG) > 0: doubling the read change adds plural shift, reversing it adds singular "
             "preference, the opposite of disruption",
        "P": "|F̂_rem − F_rem| ≤ max(τ_rem, 0.02) and |F̂_ins − F_ins| ≤ max(τ_ins, 0.02), the program's table committed before any target prompt",
    },
    "order": "R and I by Holm at 0.025 on the two count p-values alone (exact binomial tails under a symmetric sign); a family passes when Holm "
             "rejects its count and its median is at least 0.05, and a failed floor does not change the other family's threshold; D only if R and "
             "I pass; P only if D passes; a family not reached is reported descriptively and cannot change the label",
    "weak": "a frame whose Δc_full exceeds −1 nat counts as a failure in every count, enters every median as the worst value, and is excluded from "
            "the direction-free summaries with its count; ties count as failures",
    "gates": "G4 is an integrity gate, not an outcome filter: a gate failure after the measurements stops outcome scoring and is reviewed under "
             "the incident procedure; no frame is dropped, replaced or selectively ignored; no tolerance is widened by calibration or changed after "
             "target results exist",
    "secondary": "the stratum, the isotropic controls, the head arms, the midpoint, PC-SWAP, the additivity, the manipulation check, the LayerNorm "
                 "records and every bootstrap interval are computed after the result is written and can never rescue, alter or redefine it",
    "incidents": "an incident carries no result; the report then renders NOT_INTERPRETABLE",
}

# ---------------------------------------------------------------------------
# Non-finite values never reach a results file as such: null, with a sibling flag.


def _nonfinite_tag(value: float) -> str:
    return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")


def encode_nonfinite(value: Any) -> Any:
    """A record fit for canonical JSON: every non-finite float becomes ``None``, with a sibling key ``<name>_nonfinite``
    in a mapping (``"-inf"``, ``"+inf"`` or ``"nan"``), or ``<name>_nonfinite_at`` (index → tag) for a list."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                out[key] = None
                out[f"{key}_nonfinite"] = _nonfinite_tag(item)
            elif isinstance(item, (list, tuple)):
                flagged = {str(index): _nonfinite_tag(element) for index, element in enumerate(item) if isinstance(element, float) and not math.isfinite(element)}
                out[key] = [None if isinstance(element, float) and not math.isfinite(element) else encode_nonfinite(element) for element in item]
                if flagged:
                    out[f"{key}_nonfinite_at"] = flagged
            else:
                out[key] = encode_nonfinite(item)
        return out
    if isinstance(value, (list, tuple)):
        return [None if isinstance(item, float) and not math.isfinite(item) else encode_nonfinite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def median(values: Sequence[float]) -> float | None:
    """The median of ``n`` values: the middle value for odd ``n``, the mean of the ``n/2``-th and ``(n/2 + 1)``-th sorted
    values for even ``n`` (``torch.median``, which returns the lower one, is not used). The mean of an infinite and a
    finite value is the infinite one."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    low, high = ordered[middle - 1], ordered[middle]
    if math.isinf(low) and math.isinf(high) and low != high:
        return math.nan
    if math.isinf(low):
        return low
    if math.isinf(high):
        return high
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# 1. The read geometry (float64; the 011 lock and the weights only).

U_OFF_RULE = ("per pair at run time, from the plain runs of the same frame: Δx = x_w − x_ref at RESID_PRE.L3, p_c; Δr = ⟨Δx, w_c⟩; "
              "read = (Δr/⟨w_c, w_c⟩)·w_c; off_c = (Δx − read) − mean(Δx − read)·𝟙; û_off = off_c/‖off_c‖; s = |Δr|/‖w_c‖ (design §2, §14 D2)")


@dataclass(frozen=True)
class ReadGeometry:
    w_c: torch.Tensor  # the centred read covector (float64)
    w_hat: torch.Tensor  # its unit direction
    norm: float  # ‖w_c‖
    norm2: float  # ⟨w_c, w_c⟩
    d_T: torch.Tensor  # 011's frozen axis at the head's output: the manipulation check's direction

    @property
    def d_model(self) -> int:
        return int(self.w_c.shape[0])


def read_geometry(lock_011: Mapping[str, Any]) -> ReadGeometry:
    """``w_c = read_weight − mean(read_weight)·𝟙`` from 011's lock, verbatim (computation 1)."""
    rw = torch.tensor(lock_011["read_weight"], dtype=torch.float64)
    w_c = rw - rw.mean()
    norm = float(torch.linalg.vector_norm(w_c))
    if not math.isfinite(norm) or norm <= NORM_FLOOR:
        raise IncidentError("the 011 read weight has no centred component")
    return ReadGeometry(w_c, w_c / norm, norm, float(w_c @ w_c), torch.tensor(lock_011["axes_vectors"]["T"], dtype=torch.float64))


def g1_read_weight(head: ht.HeadWeights, axis_T: pm.SiteAxis, geometry: ReadGeometry) -> float:
    """G1: ``γ₃ ⊙ (W_V @ (W_O @ d̂_T))`` from the model's own weights, centred, against ``w_c`` (max abs)."""
    weights = ra.read_weight(head, axis_T).double()
    return float(((weights - weights.mean()) - geometry.w_c).abs().max())


def _ones_unit(d: int) -> torch.Tensor:
    return torch.full((d,), 1.0 / math.sqrt(d), dtype=torch.float64)


def iso_direction(geometry: ReadGeometry, frame_id: str, j: int) -> torch.Tensor:
    """``v_j = unit(project_off(sha_gaussians(ISO_TAG, d), (𝟙/√d, ŵ)))``: mean-zero, unit and orthogonal to ``w_c``."""
    tag = ISO_TAG.format(frame_id=frame_id, j=int(j))
    return cr.unit(cr.project_off(cr.sha_gaussians(tag, geometry.d_model), (_ones_unit(geometry.d_model), geometry.w_hat)))


def iso_directions(geometry: ReadGeometry, frame_id: str, k_iso: int) -> list[torch.Tensor]:
    return [iso_direction(geometry, frame_id, j) for j in range(1, int(k_iso) + 1)]


def basis_checks(geometry: ReadGeometry, directions: Mapping[str, Sequence[torch.Tensor]]) -> dict[str, float]:
    """G1's float64 checks: the basis ``(𝟙/√d, ŵ)`` orthonormal; every ``v_j`` unit, mean-zero and orthogonal to ``ŵ``."""
    ones = _ones_unit(geometry.d_model)
    out = {"basis_norms": max(abs(float(torch.linalg.vector_norm(ones)) - 1.0), abs(float(torch.linalg.vector_norm(geometry.w_hat)) - 1.0)),
           "basis_orthogonal": abs(float(ones @ geometry.w_hat)), "v_norm": 0.0, "v_mean": 0.0, "v_dot_w": 0.0}
    for vectors in directions.values():
        for v in vectors:
            out["v_norm"] = max(out["v_norm"], abs(float(torch.linalg.vector_norm(v)) - 1.0))
            out["v_mean"] = max(out["v_mean"], abs(float(v.mean())))
            out["v_dot_w"] = max(out["v_dot_w"], abs(float(v @ geometry.w_hat)))
    return out


def geometry_block(geometry: ReadGeometry, head: ht.HeadWeights, axis_T: pm.SiteAxis, frame_ids: Sequence[str], k_iso: int) -> dict[str, Any]:
    """What the lock binds about the variable: the digests of ``w_c``, ``ŵ``, ``d̂_T`` and every frame's ``v_j``, G1's
    values, and the rule for ``û_off``. Recomputed bit for bit at ``stage1`` and ``confirm``."""
    directions = {frame_id: iso_directions(geometry, frame_id, k_iso) for frame_id in frame_ids}
    checks = basis_checks(geometry, directions)
    return {"w_c_sha256": rc.tensor_digest(geometry.w_c), "w_c_norm": geometry.norm, "w_hat_sha256": rc.tensor_digest(geometry.w_hat),
            "d_T_sha256": rc.tensor_digest(geometry.d_T), "iso_tag": ISO_TAG, "k_iso": int(k_iso),
            "iso_sha256": {frame_id: [rc.tensor_digest(v) for v in vectors] for frame_id, vectors in directions.items()},
            "g1": {"read_weight_max_abs": g1_read_weight(head, axis_T, geometry), **checks}, "u_off_rule": U_OFF_RULE}


def enforce_g1(block: Mapping[str, Any]) -> None:
    g1 = block["g1"]
    if not g1["read_weight_max_abs"] <= TOLERANCES["G1_read_weight"]:
        raise IncidentError(f"G1 failed: the weights do not reproduce the 011 read weight ({g1['read_weight_max_abs']:.3e})")
    for name in ("basis_norms", "basis_orthogonal", "v_norm", "v_mean", "v_dot_w"):
        if not g1[name] <= TOLERANCES["G1_basis"]:
            raise IncidentError(f"G1 failed: {name} {g1[name]:.3e} above {TOLERANCES['G1_basis']:.0e}")


# ---------------------------------------------------------------------------
# 2. The pair quantities and the edits (computations 2–3).


@dataclass(frozen=True)
class Pair:
    """One frame and cue: the change at ``p_c`` between the plain reference and plain cue runs, and its parts."""

    dx: torch.Tensor
    dr: float
    read: torch.Tensor
    off_c: torch.Tensor
    off_norm: float
    u_off: torch.Tensor
    s: float

    @property
    def overshoot(self) -> bool:
        return self.off_norm < self.s

    @property
    def ratio(self) -> float:
        return self.s / self.off_norm

    def record(self) -> dict[str, Any]:
        return {"dr": self.dr, "dx_norm": float(torch.linalg.vector_norm(self.dx)), "off_c_norm": self.off_norm, "s": self.s,
                "s_over_off_c": self.ratio, "overshoot": self.overshoot}


def pair_quantities(dx: torch.Tensor, geometry: ReadGeometry) -> Pair:
    """``Δr``, the read part, the centred off-read part, ``û_off`` and ``s`` of a change at ``p_c`` (computation 2); the
    program's own quantities come from its ``Δx̂3(p_c)`` by the same function."""
    dx = dx.double()
    dr = float(dx @ geometry.w_c)
    read = (dr / geometry.norm2) * geometry.w_c
    off = dx - read
    off_c = off - off.mean()
    off_norm = float(torch.linalg.vector_norm(off_c))
    if not math.isfinite(off_norm) or off_norm <= NORM_FLOOR:
        raise IncidentError("the off-read part of the change vanishes; û_off is undefined")
    return Pair(dx, dr, read, off_c, off_norm, off_c / off_norm, abs(dr) / geometry.norm)


def edit_vector(arm: ArmSpec, pair: Pair, v: torch.Tensor | None = None) -> torch.Tensor:
    """The float64 change a read, off-read or isotropic arm adds to its prompt's plain residual."""
    if arm.kind == "read" or arm.head:
        return arm.coefficient * pair.read
    if arm.kind == "off":
        return arm.coefficient * pair.s * pair.u_off
    if arm.kind == "iso":
        if v is None:
            raise ValueError("an isotropic arm needs its direction")
        return arm.coefficient * pair.s * v
    raise ValueError(f"{arm.name} adds no computed change")


def residual_replacement(arm: ArmSpec, x_ref: torch.Tensor, x_cue: torch.Tensor, pair: Pair, v: torch.Tensor | None = None) -> torch.Tensor:
    """The replacement at ``RESID_PRE.L3``, ``p_c``: the float32 cast of the float64 edit, shaped ``[1, 1, d]``. NOOP and
    PC-SWAP reuse a captured float32 residual exactly."""
    if arm.kind == "noop":
        return x_cue.detach().reshape(1, 1, -1).to(torch.float32)
    if arm.kind == "swap":
        return (x_ref if arm.prompt == "cue" else x_cue).detach().reshape(1, 1, -1).to(torch.float32)
    base = x_cue if arm.prompt == "cue" else x_ref
    return (base.double().reshape(-1) + edit_vector(arm, pair, v)).to(torch.float32).reshape(1, 1, -1)


def g2_values(arm: ArmSpec, x_ref: torch.Tensor, x_cue: torch.Tensor, x_prime: torch.Tensor, pair: Pair, geometry: ReadGeometry) -> dict[str, float]:
    """G2's realized geometry of one residual edit, in float64 from the captured float32 vectors: ``r(x′)`` against its
    target, the centred length of ``e = x′ − x``, and parallelism (read arms) or orthogonality (off and isotropic arms)
    to ``ŵ``. PC-SWAP checks ``r`` and the centred length only; NOOP has none."""
    if arm.kind == "noop":
        return {}
    x_ref, x_cue, x_prime = x_ref.double().reshape(-1), x_cue.double().reshape(-1), x_prime.double().reshape(-1)
    base = x_cue if arm.prompt == "cue" else x_ref
    e = x_prime - base
    e_c = e - e.mean()
    e_norm = float(torch.linalg.vector_norm(e_c))
    if arm.kind == "swap":
        other = x_ref if arm.prompt == "cue" else x_cue
        target_r, target_length, direction = float(other @ geometry.w_c), float(torch.linalg.vector_norm((other - base) - (other - base).mean())), None
    elif arm.kind == "read":
        target_r, target_length, direction = float(base @ geometry.w_c) + arm.coefficient * pair.dr, abs(arm.coefficient) * pair.s, "parallel"
    else:  # off, iso
        target_r, target_length, direction = float(base @ geometry.w_c), pair.s, "orthogonal"
    out = {"r_error": abs(float(x_prime @ geometry.w_c) - target_r), "length_error": abs(e_norm - target_length) / max(target_length, NORM_FLOOR)}
    if direction == "parallel":
        out["direction_error"] = float(torch.linalg.vector_norm(e_c - float(e_c @ geometry.w_hat) * geometry.w_hat)) / max(e_norm, NORM_FLOOR)
    elif direction == "orthogonal":
        out["direction_error"] = abs(float(e_c @ geometry.w_hat)) / max(e_norm, NORM_FLOOR)
    return out


def u_off32_values(pair: Pair, geometry: ReadGeometry) -> dict[str, float]:
    u32 = pair.u_off.to(torch.float32).double()
    return {"mean": abs(float(u32.mean())), "norm": abs(float(torch.linalg.vector_norm(u32)) - 1.0), "dot_w": abs(float(u32 @ geometry.w_hat))}


def layer_norm_records(head: ht.HeadWeights, gamma_ln2: torch.Tensor, x_prime: torch.Tensor, e: torch.Tensor) -> dict[str, float]:
    """``σ(x′)``, ``‖γ₃ ⊙ e‖/σ(x′)`` and ``‖γ_ln2 ⊙ e‖/σ(x′)`` (design §2). Pythia's parallel block feeds both LayerNorms
    the same ``x′``."""
    sigma = head.scale(x_prime.double().reshape(-1))
    e = e.double().reshape(-1)
    return {"sigma": sigma, "gamma3_e_over_sigma": float(torch.linalg.vector_norm(head.ln_w.double() * e)) / sigma,
            "gamma_ln2_e_over_sigma": float(torch.linalg.vector_norm(gamma_ln2.double() * e)) / sigma}


# ---------------------------------------------------------------------------
# 3. The frame freeze (tokenizer only; no forward pass; design §11 and §14's frame-freeze guard).

CONFIRMATION_SCHEMA_VERSION = 1
FREEZE_RULES = {
    "structure": "pm._build_new_frame accepts the text with the reference and the primary cue, and again with each stratum cue sharing the same "
                 "prefix and suffix; the cue is a single token at p_c within the range; the adjective is one token at p_t = p_c + 1",
    "text": "the text template is none of the frame texts of b0c.exclusion (020's pool, the 006–020 confirmations, the 005 extension and 022's "
            "frames), as 023's committed exclusion records them, and none of 023's own frame templates; 024 and 025 added no frames",
    "earlier_cues": "the adjective's id is not in 024's committed exclusion (the cue ids of 005–023), not one of 024's or 025's cues",
    "candidates_025": "the adjective is not a word of 025's ordered candidate lists (the adjective reserve, the new list and both forms of the "
                      "ordinary reserve), by word or by single-token id",
    "target_forms": "the adjective's id is not a form of a pool target noun (rr.target_noun_form_ids) nor of 020's confirmation nouns (cr.prior_nouns)",
    "number_words": "the adjective is not a number word",
    "distinct": "no two 101 frames share the adjective",
    "not_applied": "025's frame_tokens set (a token of an exposed frame) is not applied: novelty is at the level of the text (design §11)",
    "selection": "the first n_frames eligible candidates in the listed order, by the tokenizer alone; no other choice, and no frame is replaced after "
                 "the freeze",
    "deviation": "a selection that differs from the expected picks writes nothing and stops for review",
}


class FreezeShortfall(ul.FreezeShortfall):
    """Fewer eligible candidates than frames: nothing is written; stop for review (not an incident)."""


class FreezeDeviation(RuntimeError):
    """The eligible frames are not the design's expected picks: nothing is written; stop for review."""


@dataclass(frozen=True)
class Exclusions:
    frame_texts: frozenset[str]
    earlier_cue_ids: frozenset[int]
    candidate_words_025: frozenset[str]
    target_form_ids: frozenset[int]
    prior_noun_ids: frozenset[int]
    record: Mapping[str, Any]


def _digested(values: Sequence[Any]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"count": len(ordered), "sha256": pm.sha256_text(pm.canonical_json(ordered))}


def exclusions_from_committed(*, confirmation_023: Mapping[str, Any], confirmation_024: Mapping[str, Any], confirmation_025: Mapping[str, Any],
                              prior: Mapping[str, Any]) -> Exclusions:
    """The freeze's named sources, from the committed and digest-pinned files alone, with their cross-checks."""
    texts = set(confirmation_023["exclusion"]["frame_texts"]) | {frame["text_template"] for frame in confirmation_023["frames"]}
    earlier_024 = {int(i) for i in confirmation_024["exclusion"]["cue_token_ids"]} | {int(cue["token_id"]) for cue in confirmation_024["cues"]}
    blocked = confirmation_025["blocked"]
    if sorted(earlier_024) != [int(i) for i in blocked["earlier_cues"]["ids"]]:
        raise PhaseError("024's exclusion and cues do not reproduce 025's committed earlier-cue set")
    earlier = earlier_024 | {int(cue["token_id"]) for cue in confirmation_025["cues"]}
    candidates = confirmation_025["candidates"]
    words = {str(word).lower() for word in candidates["adjective-reserve"]} | {str(word).lower() for word in candidates["new-list"]}
    words |= {str(word).lower() for pair in candidates["ordinary-reserve"] for word in pair}
    targets = {int(i) for i in blocked["target_forms"]["ids"]}
    if sorted(targets) != sorted(int(i) for i in confirmation_024["target_noun_form_ids"]["ids"]):
        raise PhaseError("025's target-noun forms are not 024's committed ones")
    prior_ids = {int(i) for i in prior["ids"]}
    if sorted(prior_ids) != sorted(int(i) for i in blocked["prior_nouns"]["ids"]):
        raise PhaseError("020's confirmation nouns do not reproduce 025's committed prior-noun set")
    record = {"frame_texts": _digested(list(texts)), "earlier_cue_ids": _digested(list(earlier)), "earlier_cue_ids_024": len(earlier_024),
              "candidate_words_025": _digested(list(words)), "target_form_ids": _digested(list(targets)), "prior_noun_ids": _digested(list(prior_ids)),
              "number_words": list(NUMBER_WORDS),
              "sources": {"confirmation_023": {"path": b0c.CONFIRMATION_RELATIVE_PATH, "content_sha256": confirmation_023["content_sha256"]},
                          "confirmation_024": {"path": rr.CONFIRMATION_RELATIVE_PATH, "content_sha256": confirmation_024["content_sha256"]},
                          "confirmation_025": {"path": cr.CONFIRMATION_RELATIVE_PATH, "content_sha256": confirmation_025["content_sha256"]},
                          "prior_nouns_020": {"path": prior["path"], "file_sha256": prior["file_sha256"]}}}
    return Exclusions(frozenset(texts), frozenset(earlier), frozenset(words), frozenset(targets), frozenset(prior_ids), record)


def frame_id_for(number: int) -> str:
    return FRAME_ID_FORMAT.format(number=int(number))


def candidate_status(tokenizer: Any, number: int, text: str, exclusions: Exclusions, config: Configuration, candidate_ids_025: frozenset[int],
                     taken: Mapping[int, int]) -> tuple[dict[str, Any], pm.Frame | None]:
    """One candidate's eligibility under every rule, with each failed rule named."""
    frame_id = frame_id_for(number)
    entry: dict[str, Any] = {"number": int(number), "text": text, "frame_id": frame_id, "reasons": []}
    reasons: list[str] = entry["reasons"]
    if text in exclusions.frame_texts:
        reasons.append("the text equals an earlier frame text")
    ref_id, (primary_label, primary_id) = config.reference[1], config.primary
    try:
        frame = pm._build_new_frame(tokenizer, TEMPLATE, text, {"sg": ref_id, "pl": primary_id}, frame_id)
    except ValueError as error:
        reasons.append(f"structure ({primary_label}): {error}")
        entry["eligible"] = False
        return entry, None
    for label, cue_id in config.stratum:
        try:
            other = pm._build_new_frame(tokenizer, TEMPLATE, text, {"sg": ref_id, "pl": cue_id}, frame_id)
        except ValueError as error:
            reasons.append(f"structure ({label}): {error}")
            continue
        if other.prefix_ids != frame.prefix_ids or other.suffix_ids != frame.suffix_ids:
            reasons.append(f"{label} does not share the prefix and the suffix")
    adjective = int(frame.suffix_ids[0])
    word = pm._decode(tokenizer, [adjective]).strip()
    entry.update({"p_c": frame.p_c, "p_t": frame.p_t, "adjective": word, "adjective_id": adjective})
    if not P_C_RANGE[0] <= frame.p_c <= P_C_RANGE[1]:
        reasons.append(f"p_c {frame.p_c} outside {P_C_RANGE[0]}–{P_C_RANGE[1]}")
    if adjective in exclusions.earlier_cue_ids:
        reasons.append("the adjective was an earlier cue")
    if word.lower() in exclusions.candidate_words_025 or adjective in candidate_ids_025:
        reasons.append("the adjective is a word of 025's candidate lists")
    if adjective in exclusions.target_form_ids or adjective in exclusions.prior_noun_ids:
        reasons.append("the adjective is a target-noun form")
    if word.lower() in NUMBER_WORDS:
        reasons.append("the adjective is a number word")
    if adjective in taken:
        reasons.append(f"the adjective is candidate {taken[adjective]}'s")
    entry["eligible"] = not reasons
    return entry, frame


@dataclass(frozen=True)
class RunSpec:
    """One run: its frame, arm, cue label and prompt; its key is ``prompt.key|tag``."""

    frame: pm.Frame
    arm: ArmSpec
    label: str | None
    prompt: pm.Prompt

    @property
    def tag(self) -> str:
        return tag_of(self.arm.name, self.label)

    @property
    def key(self) -> str:
        return cr.tagged_key(self.prompt, self.tag)


def frame_schedule(frame: pm.Frame, config: Configuration, reference: pm.Prompt, cue_prompts: Mapping[str, pm.Prompt]) -> tuple[RunSpec, ...]:
    """A frame's runs in their fixed order: REF, then each cue's arms in ``CUE_ORDER`` (the configuration's)."""
    return tuple(RunSpec(frame, arm, label, reference if arm.prompt == "ref" else cue_prompts[label]) for arm, label in config.run_order())


@dataclass(frozen=True)
class Confirmation101:
    frames: tuple[pm.Frame, ...]  # the frozen frames, in frozen order
    config: Configuration
    content_sha256: str

    def reference_prompt(self, frame: pm.Frame) -> pm.Prompt:
        return pm.Prompt(frame, int(self.config.reference[1]), self.config.reference[0])

    def cue_prompts(self, frame: pm.Frame) -> dict[str, pm.Prompt]:
        return {label: pm.Prompt(frame, int(cue_id), label) for label, cue_id in self.config.cues}

    def schedule(self, frame: pm.Frame) -> tuple[RunSpec, ...]:
        return frame_schedule(frame, self.config, self.reference_prompt(frame), self.cue_prompts(frame))

    def s1_keys(self) -> list[str]:
        return [cr.tagged_key(self.reference_prompt(frame), "S1") for frame in self.frames]

    def s2_keys(self) -> list[str]:
        return [run.key for frame in self.frames for run in self.schedule(frame)]

    def untagged_keys(self) -> frozenset[str]:
        return frozenset(cr.untagged(key) for key in (*self.s1_keys(), *self.s2_keys()))

    def manifest(self) -> dict[str, Any]:
        return {"S1-REF": sorted(self.s1_keys()), "S2-TARGET": sorted(self.s2_keys())}

    def counts(self) -> dict[str, Any]:
        return {"frames": len(self.frames), "runs_per_frame": self.config.runs_per_frame, "S1-REF": len(self.s1_keys()), "S2-TARGET": len(self.s2_keys()),
                "untagged_prompts": len(self.untagged_keys())}


def freeze_payload(tokenizer: Any, *, exclusions: Exclusions, config: Configuration, model: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The confirmation file's content: every candidate's eligibility, the first ``n_frames`` eligible frames, the cue
    and arm orders, and the S1 and S2 manifests. A shortfall or a deviation raises before anything exists to write."""
    candidate_ids_025 = frozenset(ids[0] for word in exclusions.candidate_words_025 if len(ids := pm._encode(tokenizer, " " + word)) == 1)
    statuses: list[dict[str, Any]] = []
    frames: list[pm.Frame] = []
    taken: dict[int, int] = {}
    for number, text in enumerate(config.frame_candidates, start=1):
        entry, frame = candidate_status(tokenizer, number, text, exclusions, config, candidate_ids_025, taken)
        statuses.append(entry)
        if entry["eligible"] and frame is not None:
            taken[int(entry["adjective_id"])] = number
            if len(frames) < config.n_frames:
                frames.append(frame)
    if len(frames) < config.n_frames:
        raise FreezeShortfall(f"{len(frames)} eligible candidates of the {config.n_frames} frames needed")
    picks = tuple(int(frame.frame_id.rsplit("-", 1)[1]) for frame in frames)
    if picks != config.expected_picks:
        raise FreezeDeviation(f"the eligible frames are candidates {list(picks)}, not the expected {list(config.expected_picks)}")
    payload = {
        "experiment": EXPERIMENT, "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "kind": "the new coordinated frames of Experiment 101, frozen from the tokenizer and the text rules alone; no model output",
        "design": dict(DESIGN), "plan": dict(PLAN),
        "model": dict(model or {"model_id": models_module.PYTHIA_70M.model_id, "revision": models_module.PYTHIA_70M.revision}),
        "configuration": config.to_json(), "rules": dict(FREEZE_RULES), "exclusions": dict(exclusions.record),
        "candidate_ids_025": _digested(list(candidate_ids_025)), "candidates": statuses,
        "frames": [{**frame.to_dict(), "number": number} for frame, number in zip(frames, picks)], "picks": list(picks),
        "expected_picks": list(config.expected_picks), "picks_match_expected": True,
        "cue_order": list(config.cue_labels), "run_order": list(config.tags()),
    }
    confirmation = confirmation_from_payload(payload, config, verify=False)
    payload["counts"] = confirmation.counts()
    payload["manifest"] = confirmation.manifest()
    payload["manifest_sha256"] = pm.sha256_text(pm.canonical_json(payload["manifest"]))
    payload["content_sha256"] = rc.content_digest(payload)
    return payload


def _frame_from(entry: Mapping[str, Any]) -> pm.Frame:
    return pm.Frame(entry["template_id"], entry["frame_id"], tuple(entry["prefix_ids"]), tuple(entry["suffix_ids"]), dict(entry["cue_ids"]),
                    entry["text_template"], origin=entry["origin"])


def confirmation_from_payload(payload: Mapping[str, Any], config: Configuration, *, verify: bool = True) -> Confirmation101:
    frames = tuple(_frame_from(entry) for entry in payload["frames"])
    confirmation = Confirmation101(frames, config, str(payload.get("content_sha256", "")))
    if verify:
        if payload.get("experiment") != EXPERIMENT or payload.get("schema_version") != CONFIRMATION_SCHEMA_VERSION:
            raise PhaseError("not an Experiment 101 confirmation file")
        if payload["content_sha256"] != rc.content_digest(payload):
            raise PhaseError("the confirmation file's content digest does not verify")
        if payload.get("configuration") != config.to_json():
            raise PhaseError("the confirmation file was frozen under a different configuration")
        if payload.get("picks") != list(config.expected_picks) or payload.get("picks_match_expected") is not True \
                or payload.get("expected_picks") != list(config.expected_picks):
            raise PhaseError("the confirmation file's picks are not the design's expected picks")
        if [frame.frame_id for frame in frames] != [frame_id_for(number) for number in config.expected_picks]:
            raise PhaseError("the confirmation file's frames are not the picks in order")
        for frame in frames:
            if frame.template_id != TEMPLATE or frame.p_t != frame.p_c + 1 or not P_C_RANGE[0] <= frame.p_c <= P_C_RANGE[1] \
                    or dict(frame.cue_ids) != {"sg": int(config.reference[1]), "pl": int(config.primary[1])}:
                raise PhaseError(f"{frame.frame_id}: the frozen frame breaks the template's structure")
        if payload["manifest"] != confirmation.manifest() or payload.get("counts") != confirmation.counts() \
                or payload.get("manifest_sha256") != pm.sha256_text(pm.canonical_json(payload["manifest"])):
            raise PhaseError("the confirmation file's manifest or counts are not the ones its frames define")
        if payload.get("cue_order") != list(config.cue_labels) or payload.get("run_order") != list(config.tags()):
            raise PhaseError("the confirmation file's cue or run order is not the configuration's")
        if len(set(confirmation.s2_keys())) != len(confirmation.s2_keys()) or set(confirmation.s1_keys()) & set(confirmation.s2_keys()):
            raise PhaseError("the run keys are not distinct")
    return confirmation


def load_confirmation_101(path: Path, exclusions: Exclusions, config: Configuration) -> Confirmation101:
    """The committed freeze, verified, with the exclusion sources recomputed now from the committed files."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    confirmation = confirmation_from_payload(payload, config)
    if payload["exclusions"] != dict(exclusions.record):
        raise PhaseError("the exclusion sources recorded at the freeze differ from the ones the committed files give now")
    return confirmation


def confirmation_binding(confirmation: Confirmation101, file_sha256: str) -> dict[str, Any]:
    return {"path": CONFIRMATION_RELATIVE_PATH, "file_sha256": file_sha256, "content_sha256": confirmation.content_sha256,
            "manifest_sha256": pm.sha256_text(pm.canonical_json(confirmation.manifest())), "counts": confirmation.counts()}


# ---------------------------------------------------------------------------
# 4. The program's predictions (computation 6) and the stage-1 files.

TABLE_SCHEMA_VERSION = 1
TABLE_CONSTRUCTION = ("per frame and cue w, from the frame's reference state (rd.state_from_locked) and the weights alone: rows16 = ul.reference_rows_017; "
                      "ctx = ul.pair_context(progs, frame, state, reference id, id_w, w, rows16); Δx̂3 = ul.compose_dx3(progs, ctx, b0c.P1_MASK['coordinated']) "
                      "(023's completed program, mask 30); the program's own Δr̂, read part, û_off and ŝ from Δx̂3(p_c) by the measurement's formula; each "
                      "main arm's change at both p_c and p_t (explicit float64 zeros) as design §5 tabulates; Δĉ = ul.contrast_of (020's frozen readout), "
                      "the scorable nouns in the listed order; the fractions with fractions_for_weak off")


def predicted_change(arm: str, dx_hat: Mapping[int, torch.Tensor], pair_hat: Pair, frame: pm.Frame) -> dict[int, torch.Tensor]:
    """The change at both positions that the readout is fed for one main arm (design §5's table)."""
    p_c, p_t = frame.p_c, frame.p_t
    dpc, dpt = dx_hat[p_c].double(), dx_hat[p_t].double()
    zero = torch.zeros_like(dpc)
    read, off = pair_hat.read, pair_hat.s * pair_hat.u_off
    table = {"PL": (dpc, dpt), "NEC-READ": (dpc - read, dpt), "SUF-READ": (read, zero), "NEC-OFF": (dpc - off, dpt), "SUF-OFF": (off, zero),
             "NEC-AMP": (dpc + read, dpt), "SUF-NEG": (-read, zero), "NEC-HALF": (dpc - 0.5 * read, dpt), "SUF-HALF": (0.5 * read, zero),
             "PC-SWAP-PL": (zero, dpt), "PC-SWAP-REF": (dpc, zero)}
    at_pc, at_pt = table[arm]
    return {p_c: at_pc, p_t: at_pt}


def ratio(numerator: float, denominator: float) -> float:
    """``numerator/denominator`` with IEEE semantics instead of an exception at zero (a non-finite fraction is a
    calibration trigger, never a crash)."""
    if denominator == 0.0:
        return math.nan if numerator == 0.0 else math.copysign(math.inf, numerator) * math.copysign(1.0, denominator)
    return numerator / denominator


def main_fractions(dc: Mapping[str, float], full: float) -> dict[str, float]:
    """The fractions of design §4 and their family statistics, from the arms' mean ``Δc`` (measured or predicted)."""
    f = {"F_rem": ratio(dc["NEC-READ"], full), "F_rem_off": ratio(dc["NEC-OFF"], full), "F_ins": ratio(dc["SUF-READ"], full),
         "F_ins_off": ratio(dc["SUF-OFF"], full), "F_half_rem": ratio(dc["NEC-HALF"], full), "F_half_ins": ratio(dc["SUF-HALF"], full),
         "F_swap_pl": ratio(dc["PC-SWAP-PL"], full), "F_swap_ref": ratio(dc["PC-SWAP-REF"], full)}
    f["N"] = f["F_rem_off"] - f["F_rem"]
    f["S"] = f["F_ins"] - f["F_ins_off"]
    f["additivity"] = f["F_rem"] + f["F_ins"]
    return f


def predict_cue(progs: ul.ModelPrograms, frame: pm.Frame, state: rd.FrameState020, rows16: Mapping[int, Any], geometry: ReadGeometry, reference_id: int,
                label: str, cue_id: int, *, fractions_for_weak: bool) -> dict[str, Any]:
    ctx = ul.pair_context(progs, frame, state, int(reference_id), int(cue_id), label, rows16)
    dx_hat = ul.compose_dx3(progs, ctx, b0c.P1_MASK["coordinated"])
    pair_hat = pair_quantities(dx_hat[frame.p_c], geometry)
    arms: dict[str, Any] = {}
    for arm in PREDICTED_ARMS:
        dc = ul.contrast_of(progs, state, predicted_change(arm, dx_hat, pair_hat, frame)).double()
        arms[arm] = {"dc": dc.tolist(), "mean": float(dc.mean())}
    means = {arm: entry["mean"] for arm, entry in arms.items()}
    full = means["PL"]
    weak = full > WEAK_SHIFT
    fractions = None
    if fractions_for_weak or not weak:
        fractions = main_fractions(means, full)
        fractions.update({"A_amp": means["NEC-AMP"] - full, "A_neg": means["SUF-NEG"]})
    return {"pair": pair_hat.record(), "dc_full": full, "weak": weak, "fractions": fractions, "arms": arms}


def table_meta(config: Configuration, noun_keys: Sequence[str]) -> dict[str, Any]:
    """Everything about the table except its values: the lock binds it, and the barrier checks it."""
    return {"experiment": EXPERIMENT, "schema_version": TABLE_SCHEMA_VERSION, "kind": "the program's prediction table (computation 6)",
            "construction": TABLE_CONSTRUCTION, "mask": int(b0c.P1_MASK["coordinated"]), "fractions_for_weak": False, "weak_shift": WEAK_SHIFT,
            "noun_keys": list(noun_keys), "arms": list(PREDICTED_ARMS), "cue_order": list(config.cue_labels), "reference_id": int(config.reference[1])}


def prediction_table(progs: ul.ModelPrograms, frames: Sequence[pm.Frame], states: Mapping[str, rd.FrameState020], geometry: ReadGeometry, config: Configuration,
                     noun_keys: Sequence[str], *, fractions_for_weak: bool = False) -> dict[str, Any]:
    """The program's predictions for every frame and cue, from the reference states and the weights alone; no target
    measurement exists or is read. The committed table is always computed with ``fractions_for_weak`` off (the
    ``confirm`` setting); calibration computes its own with it on."""
    out: dict[str, Any] = {}
    for frame in frames:
        state = states[frame.frame_id]
        rows16 = ul.reference_rows_017(progs.programs, state)
        out[frame.frame_id] = {"p_c": frame.p_c, "p_t": frame.p_t,
                               "cues": {label: predict_cue(progs, frame, state, rows16, geometry, config.reference[1], label, cue_id, fractions_for_weak=fractions_for_weak)
                                        for label, cue_id in config.cues}}
    payload = {**table_meta(config, noun_keys), "fractions_for_weak": bool(fractions_for_weak), "frames": out}
    if fractions_for_weak:
        payload = encode_nonfinite(payload)
    else:
        try:
            pm.canonical_json(payload)
        except ValueError as error:
            raise IncidentError(f"the prediction table holds a non-finite value: {error}") from error
    payload["content_sha256"] = rc.content_digest(payload)
    return payload


STATE_FIELDS = ("x1_all", "x2_all", "x3_all", "x4_all", "x5_all", "h6", "rows4", "rows5", "c_ref")


def states_layout(frames: Sequence[pm.Frame], d_model: int, n_heads: int, n_nouns: int) -> list[dict[str, Any]]:
    """The stage-1 states table's blocks: per frame in frozen order, the fields of ``rd.locked_state``."""
    layout = []
    for frame in frames:
        keys = frame.p_t + 1
        shapes = {"x1_all": [keys, d_model], "x2_all": [keys, d_model], "x3_all": [keys, d_model], "x4_all": [keys, d_model], "x5_all": [keys, d_model],
                  "h6": [d_model], "rows4": [n_heads, keys], "rows5": [n_heads, keys], "c_ref": [n_nouns]}
        layout += [{"name": f"{frame.frame_id}/{name}", "shape": shapes[name]} for name in STATE_FIELDS]
    return layout


def states_meta(frames: Sequence[pm.Frame]) -> dict[str, Any]:
    return {"experiment": EXPERIMENT, "kind": "the stage-1 reference states", "fields": list(STATE_FIELDS),
            "construction": "rd.locked_state of each S1-REF capture (rd.capture_frame_020), float64; rebuilt with rd.state_from_locked",
            "frames": [{"frame_id": frame.frame_id, "p_c": frame.p_c, "p_t": frame.p_t} for frame in frames]}


def states_blocks(locked: Mapping[str, Mapping[str, Any]], frames: Sequence[pm.Frame]) -> list[tuple[str, torch.Tensor]]:
    return [(f"{frame.frame_id}/{name}", torch.tensor(locked[frame.frame_id][name], dtype=torch.float64)) for frame in frames for name in STATE_FIELDS]


def states_from_blocks(blocks: Mapping[str, torch.Tensor], frames: Sequence[pm.Frame]) -> dict[str, dict[str, Any]]:
    """The locked entries back from the table, exactly as ``rd.locked_state`` wrote them (their digests must agree)."""
    return {frame.frame_id: {"p_c": frame.p_c, "p_t": frame.p_t, **{name: blocks[f"{frame.frame_id}/{name}"].tolist() for name in STATE_FIELDS}} for frame in frames}


# ---------------------------------------------------------------------------
# 5. Measurement: the sanctioned entry points, the capture set and one frame's runs.

RESID_L3 = f"RESID_PRE.L{hp.HEAD_LAYER}"
HEAD_SITE_KEY = f"L{hp.HEAD_LAYER:02d}.H{hp.HEAD_INDEX:02d}"
PATTERN_L3 = f"ATTN_PATTERN.L{hp.HEAD_LAYER}"
CAPTURE_FIELDS = ("x3_pc", "x3_pt", "x4_pc", "heads_pc", "mlp_pc", "T_pt", "attn_pt", "h6")
_ROLE_BY_ARM = {arm.name: arm.prompt for arm in (*MAIN_ARMS, NOOP_ARM, *HEAD_ARMS)}


def parse_tag(tag: str) -> tuple[str, str | None]:
    if tag in ("REF", "S1"):
        return tag, None
    name, separator, label = tag.partition("@")
    if not separator or not label or not name:
        raise PhaseError(f"the run tag {tag!r} is not ARM@cue")
    return name, label


def arm_role(name: str) -> str:
    if name in ("REF", "S1"):
        return "ref"
    if name in _ROLE_BY_ARM:
        return _ROLE_BY_ARM[name]
    for prefix, role in (("NEC-ISO-", "cue"), ("SUF-ISO-", "ref")):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            return role
    raise PhaseError(f"unknown arm {name!r}")


def capture_sites(frame: pm.Frame, n_layers: int, n_heads: int, *, plain: bool) -> list[tuple[str, int]]:
    """The fixed per-run capture set; the plain runs also capture ``RESID_PRE.L3`` at every position ``0..p_t``."""
    p_c, p_t = frame.p_c, frame.p_t
    sites = [(RESID_L3, p_c), (RESID_L3, p_t), (f"RESID_PRE.L{hp.HEAD_LAYER + 1}", p_c)]
    sites += [(f"L{hp.HEAD_LAYER:02d}.H{h:02d}", p_c) for h in range(int(n_heads))]
    sites += [(f"L{hp.HEAD_LAYER:02d}.MLP", p_c), (HEAD_SITE_KEY, p_t), (PATTERN_L3, p_t), (f"RESID_POST.L{int(n_layers) - 1}", p_t)]
    if plain:
        sites += [(RESID_L3, k) for k in range(p_t + 1)]
    return list(dict.fromkeys(sites))


def run_arm(model: Any, prompt: pm.Prompt, tag: str, *, replacements: Mapping[tuple[str, int], torch.Tensor] | None = None,
            capture_sites: Sequence[tuple[str, int]] = ()) -> pm.PromptRun:
    """The only function that runs an Experiment 101 arm: a plain run through ``pm.capture_prompt``, an edited one
    through ``pm.run_patched`` (whose landing and zero-outside-change checks raise). The tag must fit the prompt: a
    reference-prompt arm runs the ``ref`` prompt, every other arm a cue prompt. The guarded launchers wrap this
    function by identity and check the key ``prompt.key|tag`` against the phase's manifest."""
    name, _ = parse_tag(tag)
    if (arm_role(name) == "ref") != (prompt.cue_label == "ref"):
        raise PhaseError(f"the tag {tag} does not fit the prompt {prompt.key}")
    if not replacements:
        return pm.capture_prompt(model, prompt, list(capture_sites))
    return pm.run_patched(model, prompt, dict(replacements), {site: pm.ReplacementSource.DIRECT for site in replacements}, capture_sites=list(capture_sites))


def capture_reference(model: Any, progs: ul.ModelPrograms, prompt: pm.Prompt, tag: str = "S1") -> rd.FrameState020:
    """The only function that captures an Experiment 101 reference state (stage 1, and calibration's re-captures)."""
    if tag != "S1" or prompt.cue_label != "ref":
        raise PhaseError(f"a reference capture runs only the ref prompt under the S1 tag, not {prompt.key}|{tag}")
    return rd.capture_frame_020(model, progs.head, prompt, progs.nouns, progs.axis_T)


@dataclass(frozen=True)
class HeadInput:
    """Exactly the two fields ``rd.head_output_change`` reads of a state."""

    x3_all: tuple[torch.Tensor, ...]
    p_t: int


def head_delta(arm: ArmSpec, progs: ul.ModelPrograms, x3_all: Sequence[torch.Tensor], attention_row: torch.Tensor, p_c: int, p_t: int,
               e: torch.Tensor) -> torch.Tensor:
    """The exact float64 change of ``L03.H04``'s output at ``p_t`` when ``e`` is added at ``p_c`` to the plain run's
    input. Full: the row and the values recomputed through the exact LayerNorm (``rd.head_output_change``). Value-only:
    009's P2 form, the measured row's weight on ``p_c`` times the value change."""
    residuals = tuple(x.double() for x in x3_all)
    if arm.kind == "head_full":
        return rd.head_output_change(progs.readout, HeadInput(residuals, int(p_t)), {int(p_c): e.double()})
    weight = float(attention_row[int(p_c)])
    head = progs.head
    return weight * ((head.normalize(residuals[p_c] + e.double()) - head.normalize(residuals[p_c])) @ head.W_V @ head.W_O)


def logits_digest(logits: torch.Tensor) -> str:
    return rc.tensor_digest(logits.double())


def arranged_pm_contrasts(progs: ul.ModelPrograms, logits: torch.Tensor) -> torch.Tensor:
    """``pm.contrasts`` arranged exactly as ``rd.capture_frame_020`` arranges ``c_ref`` (G6's comparison)."""
    values = pm.contrasts(logits, [noun for noun in progs.nouns.nouns if noun.single_token])
    return torch.tensor([float(values.get(noun.lexical_key, 0.0)) for noun in progs.nouns.nouns], dtype=torch.float64)


def measure_frame(model: Any, progs: ul.ModelPrograms, schedule: Sequence[RunSpec], geometry: ReadGeometry, *, k_iso: int, n_layers: int, n_heads: int,
                  executed: list[str], out: dict[str, torch.Tensor], logits: dict[str, list[str]]) -> None:
    """Every run of one frame, once, in order. The plain runs come first in each cue's block, so each edit is built from
    the same frame's plain captures. Each run's contrasts and logit identity are computed before its logits are reduced
    to a digest; nothing is enforced here but ``pm.run_patched``'s landing checks and the finiteness of the contrasts."""
    frame = schedule[0].frame
    p_c, p_t, prefix = frame.p_c, frame.p_t, frame.frame_id
    rows, d = len(schedule), geometry.d_model
    nan32 = lambda *shape: torch.full(shape, math.nan, dtype=torch.float32)  # noqa: E731
    out[f"{prefix}/contrasts"] = torch.full((rows, len(progs.scorable)), math.nan, dtype=torch.float64)
    out[f"{prefix}/logit_identity"] = torch.full((rows,), math.nan, dtype=torch.float64)
    for name, shape in (("x3_pc", (d,)), ("x3_pt", (d,)), ("x4_pc", (d,)), ("heads_pc", (n_heads, d)), ("mlp_pc", (d,)), ("T_pt", (d,)),
                        ("attn_pt", (n_heads, p_t + 1)), ("h6", (d,))):
        out[f"{prefix}/{name}"] = nan32(rows, *shape)
    out[f"{prefix}/positions"] = torch.tensor([p_c, p_t], dtype=torch.int64)
    digests = logits.setdefault(prefix, [])
    plain: dict[str, pm.PromptRun] = {}
    pairs: dict[str, Pair] = {}
    directions = iso_directions(geometry, prefix, k_iso)
    for row, run in enumerate(schedule):
        arm = run.arm
        replacements = None
        if arm.residual_edit:
            x_ref, x_cue = plain["ref"].vector((RESID_L3, p_c)), plain[run.label].vector((RESID_L3, p_c))
            replacement = residual_replacement(arm, x_ref, x_cue, pairs[run.label], directions[arm.j - 1] if arm.kind == "iso" else None)
            replacements = {(RESID_L3, p_c): replacement}
            out[f"{prefix}/replacement/{run.tag}"] = replacement.reshape(-1).clone()
        elif arm.head:
            source = plain["ref"] if arm.prompt == "ref" else plain[run.label]
            x3_all = [source.vector((RESID_L3, k)).double() for k in range(p_t + 1)]
            attention_row = source.vector((PATTERN_L3, p_t))[hp.HEAD_INDEX].double()
            delta = head_delta(arm, progs, x3_all, attention_row, p_c, p_t, edit_vector(arm, pairs[run.label]))
            captured = source.slice((HEAD_SITE_KEY, p_t))
            replacement = (captured.double().reshape(-1) + delta).to(torch.float32).reshape(captured.shape)
            replacements = {(HEAD_SITE_KEY, p_t): replacement}
            out[f"{prefix}/delta/{run.tag}"] = delta.clone()
            out[f"{prefix}/head_replacement/{run.tag}"] = replacement.reshape(-1).clone()
        result = run_arm(model, run.prompt, run.tag, replacements=replacements, capture_sites=capture_sites(frame, n_layers, n_heads, plain=arm.kind == "plain"))
        executed.append(run.key)
        h6 = result.vector((f"RESID_POST.L{int(n_layers) - 1}", p_t))
        out[f"{prefix}/contrasts"][row] = progs.nouns.contrasts(result.logits)[progs.scorable].double()
        out[f"{prefix}/logit_identity"][row] = rd.logit_identity(progs.readout, progs.weights, h6, result.logits)
        digests.append(logits_digest(result.logits))
        out[f"{prefix}/x3_pc"][row] = result.vector((RESID_L3, p_c)).float()
        out[f"{prefix}/x3_pt"][row] = result.vector((RESID_L3, p_t)).float()
        out[f"{prefix}/x4_pc"][row] = result.vector((f"RESID_PRE.L{hp.HEAD_LAYER + 1}", p_c)).float()
        out[f"{prefix}/heads_pc"][row] = torch.stack([result.vector((f"L{hp.HEAD_LAYER:02d}.H{h:02d}", p_c)) for h in range(int(n_heads))]).float()
        out[f"{prefix}/mlp_pc"][row] = result.vector((f"L{hp.HEAD_LAYER:02d}.MLP", p_c)).float()
        out[f"{prefix}/T_pt"][row] = result.vector((HEAD_SITE_KEY, p_t)).float()
        out[f"{prefix}/attn_pt"][row] = result.vector((PATTERN_L3, p_t))[:, : p_t + 1].float()
        out[f"{prefix}/h6"][row] = h6.float()
        if arm.kind == "plain":
            role = "ref" if arm.prompt == "ref" else run.label
            plain[role] = result
            out[f"{prefix}/x3_all/{run.tag}"] = torch.stack([result.vector((RESID_L3, k)) for k in range(p_t + 1)]).float()
            if role == "ref":
                out[f"{prefix}/c_ref_pm"] = arranged_pm_contrasts(progs, result.logits)
            else:
                pairs[role] = pair_quantities(result.vector((RESID_L3, p_c)).double() - plain["ref"].vector((RESID_L3, p_c)).double(), geometry)


def stage_runs(model: Any, progs: ul.ModelPrograms, schedules: Sequence[Sequence[RunSpec]], geometry: ReadGeometry, config: Configuration, *,
               executed: list[str], out: dict[str, torch.Tensor], logits: dict[str, list[str]], log: Callable[[str], None] | None = None) -> None:
    """Every frame's runs, frames in order. The tensors and digests are filled into the caller's ``out`` and ``logits``
    as they are measured, so that a failure part-way leaves the runs measured so far with the caller."""
    say = log or (lambda message: None)
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    for index, schedule in enumerate(schedules):
        measure_frame(model, progs, schedule, geometry, k_iso=config.k_iso, n_layers=n_layers, n_heads=n_heads, executed=executed, out=out, logits=logits)
        say(f"  frame {index + 1}/{len(schedules)} ({schedule[0].frame.frame_id}): {len(schedule)} runs")


def expected_tensor_names(schedules: Sequence[Sequence[RunSpec]]) -> list[str]:
    names = []
    for schedule in schedules:
        prefix = schedule[0].frame.frame_id
        names += [f"{prefix}/{name}" for name in ("contrasts", "logit_identity", *CAPTURE_FIELDS, "positions", "c_ref_pm")]
        for run in schedule:
            if run.arm.kind == "plain":
                names.append(f"{prefix}/x3_all/{run.tag}")
            elif run.arm.residual_edit:
                names.append(f"{prefix}/replacement/{run.tag}")
            elif run.arm.head:
                names += [f"{prefix}/delta/{run.tag}", f"{prefix}/head_replacement/{run.tag}"]
    return sorted(names)


def assert_measurements(tensors: Mapping[str, torch.Tensor], schedules: Sequence[Sequence[RunSpec]]) -> None:
    """Exactly the expected tensors, every value finite."""
    expected = expected_tensor_names(schedules)
    if sorted(tensors) != expected:
        missing, extra = sorted(set(expected) - set(tensors)), sorted(set(tensors) - set(expected))
        raise IncidentError(f"the saved measurements are not the expected tensors (missing {missing[:3]}, extra {extra[:3]})")
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise IncidentError(f"the saved measurement {name} holds a non-finite value")


# ---------------------------------------------------------------------------
# 6. The gates, from the saved measurements (G2–G6), and the per-run records.

TOLERANCE_GATES = {"G2_r": "G2_read", "G2_length": "G2_length", "G2_direction": "G2_direction", "G2_u_off32": "G2_u_off32", "G3_pass_through": "G3_pass_through",
                   "G4_I3": "G4_I3", "G4_I4": "G4_I4", "G4_level1": "G4_level1", "G4_readout": "G4_readout", "G4_logit": "G4_logit", "G5_row": "G5_row",
                   "G5_output": "G5_output"}
BITWISE_GATES = ("G3_noop", "G4_x3_pt", "G5_landing", "G6_c_ref", "G6_state")
RECORDED = ("E1", "logit_other", "pass_through_other", "ceiling_error")


class GateBook:
    """Every gate's worst value and where it occurred, overall and per arm tag; bit-for-bit gates count failures."""

    def __init__(self) -> None:
        self.values = {name: {"max": 0.0, "at": ""} for name in TOLERANCE_GATES}
        self.bitwise = {name: {"failures": 0, "checked": 0, "first": None} for name in BITWISE_GATES}
        self.recorded = {name: {"max": 0.0, "at": ""} for name in RECORDED}
        self.per_arm: dict[str, dict[str, float]] = {}

    @staticmethod
    def _worse(entry: dict[str, Any], value: float, where: str) -> None:
        """Keep the worst value: a non-finite value is worst of all and stays."""
        value, current = float(value), float(entry["max"])
        if not math.isfinite(current):
            return
        if not math.isfinite(value) or value > current:
            entry["max"], entry["at"] = value, where

    def tol(self, name: str, value: float, where: str, arm: str) -> None:
        self._worse(self.values[name], value, where)
        bucket = self.per_arm.setdefault(name, {})
        entry = {"max": bucket.get(arm, 0.0), "at": ""}
        self._worse(entry, value, where)
        bucket[arm] = entry["max"]

    def bit(self, name: str, ok: bool, where: str) -> None:
        entry = self.bitwise[name]
        entry["checked"] += 1
        if not ok:
            entry["failures"] += 1
            entry["first"] = entry["first"] or where

    def record(self, name: str, value: float, where: str) -> None:
        self._worse(self.recorded[name], value, where)

    def failures(self) -> list[str]:
        failing = [name for name, entry in self.values.items() if not (math.isfinite(entry["max"]) and entry["max"] <= TOLERANCES[TOLERANCE_GATES[name]])]
        return failing + [name for name, entry in self.bitwise.items() if entry["failures"]]

    def to_json(self) -> dict[str, Any]:
        return encode_nonfinite({"tolerance": {name: {**entry, "tolerance": TOLERANCES[TOLERANCE_GATES[name]]} for name, entry in self.values.items()},
                                 "bitwise": {name: dict(entry) for name, entry in self.bitwise.items()},
                                 "recorded": {name: dict(entry) for name, entry in self.recorded.items()},
                                 "per_arm": {name: dict(sorted(entries.items())) for name, entries in sorted(self.per_arm.items())},
                                 "failures": self.failures()})


def _index(schedule: Sequence[RunSpec]) -> dict[str, int]:
    return {run.tag: row for row, run in enumerate(schedule)}


def frame_gates(book: GateBook, progs: ul.ModelPrograms, schedule: Sequence[RunSpec], tensors: Mapping[str, torch.Tensor], digests: Sequence[str],
                program_state: rd.FrameState020, g6_state: rd.FrameState020, geometry: ReadGeometry, config: Configuration) -> dict[str, Any]:
    """G2–G6 for one frame's runs, from the saved measurements; returns the frame's ceilings (recorded per G4 run).

    - G2: each residual edit's realized geometry; each pair's float32 ``û_off``.
    - G3: NOOP against ``PL`` of the primary cue, bit for bit at NOOP's capture sites, its contrasts and its logits'
      digest; the pass-through identity of every run (gated on the residual edits).
    - G4 (the reviewer's decision of 2026-10-05), on the residual edits and the plain cue runs: ``x3(p_t)`` unchanged;
      023's I3 on each plain pair; 023's I4 (the composition proxy against the run's ceiling); 020's Level-1 identity in
      025's contrast form; 020's readout and logit identities; ``E₁`` recorded.
    - G5: each head arm's row and output along ``d̂_T``, recomputed from the plain run's unedited inputs, against that
      plain run's captures; the head arm's own capture against its replacement.
    - G6: the REF run against the reference state (stage 1's, or calibration's re-capture), bit for bit."""
    frame = schedule[0].frame
    p_c, p_t, prefix = frame.p_c, frame.p_t, frame.frame_id
    index = _index(schedule)
    field = lambda name: tensors[f"{prefix}/{name}"]  # noqa: E731
    x3pc, x3pt, contrasts, h6 = field("x3_pc"), field("x3_pt"), field("contrasts"), field("h6")
    ref_row = index["REF"]
    cue_rows = {label: index[tag_of("PL", label)] for label in config.cue_labels}
    x_ref = x3pc[ref_row].double()
    pairs = {label: pair_quantities(x3pc[row].double() - x_ref, geometry) for label, row in cue_rows.items()}
    for label, pair in pairs.items():
        book.tol("G2_u_off32", max(u_off32_values(pair, geometry).values()), f"{prefix}|{label}", f"u_off@{label}")
    reference_x3 = [x.double() for x in program_state.x3_all]
    rows16 = ul.reference_rows_017(progs.programs, program_state)
    full: dict[str, dict[int, torch.Tensor]] = {}
    for label, cue_id in config.cues:
        ctx = ul.pair_context(progs, frame, program_state, int(config.reference[1]), int(cue_id), label, rows16)
        full[label] = ul.compose_dx3(progs, ctx, b0c.FULL_MASK["coordinated"])
        row = cue_rows[label]
        measured = {p_c: x3pc[row].double() - reference_x3[p_c], p_t: x3pt[row].double() - reference_x3[p_t]}
        book.tol("G4_I3", ul.i3_error(full[label], measured), f"{prefix}|{label}", tag_of("PL", label))
    b_O = progs.weights.attn_b_O[hp.HEAD_LAYER].double()
    c_ref_run = contrasts[ref_row].double()
    zero = torch.zeros(geometry.d_model, dtype=torch.float64)
    ceilings: dict[str, float] = {}
    for row, run in enumerate(schedule):
        arm, where, tag = run.arm, run.key, run.tag
        pass_through = float((field("x4_pc")[row].double() - (x3pc[row].double() + field("heads_pc")[row].double().sum(dim=0) + b_O
                                                               + field("mlp_pc")[row].double())).abs().max())
        if arm.residual_edit:
            book.tol("G3_pass_through", pass_through, where, tag)
        else:
            book.record("pass_through_other", pass_through, where)
        g4_run = arm.residual_edit or (arm.kind == "plain" and tag != "REF")
        logit = float(field("logit_identity")[row])
        if g4_run:
            book.tol("G4_logit", logit, where, tag)
        else:
            book.record("logit_other", logit, where)
        if arm.residual_edit:
            for name, value in g2_values(arm, x_ref, x3pc[cue_rows[run.label]], x3pc[row], pairs[run.label], geometry).items():
                book.tol({"r_error": "G2_r", "length_error": "G2_length", "direction_error": "G2_direction"}[name], value, where, tag)
            plain_row = ref_row if arm.prompt == "ref" else cue_rows[run.label]
            book.bit("G4_x3_pt", bool(torch.equal(x3pt[row], x3pt[plain_row])), where)
        if g4_run:
            label = run.label
            dx3_run = {p_c: x3pc[row].double() - reference_x3[p_c], p_t: x3pt[row].double() - reference_x3[p_t]}
            if arm.prompt == "ref":
                proxy = {p_c: x3pc[row].double() - reference_x3[p_c], p_t: zero}
            else:
                proxy = {p_c: full[label][p_c].double() + (x3pc[row].double() - x3pc[cue_rows[label]].double()), p_t: full[label][p_t].double()}
            ceiling = ul.contrast_of(progs, program_state, dx3_run).double()
            book.tol("G4_I4", float((ul.contrast_of(progs, program_state, proxy).double() - ceiling).abs().max()), where, tag)
            dc_run = contrasts[row].double() - c_ref_run
            detail = progs.readout.level1_detail(program_state, dict(dx3_run), l4_heads=True)
            level1 = progs.readout.contrast(program_state, detail["dh6"], progs.nouns)[progs.scorable].double()
            book.tol("G4_level1", float((level1 - dc_run).abs().max()), where, tag)
            book.record("E1", rd.level1_error(detail["dh6"], h6[row].double() - program_state.h6.double()), where)
            readout = progs.nouns.read(progs.readout.ln_final(h6[row].double()) - progs.readout.ln_final(program_state.h6.double()))[progs.scorable].double()
            book.tol("G4_readout", float((readout - dc_run).abs().max()), where, tag)
            book.record("ceiling_error", float((ceiling - dc_run).abs().max()), where)
            ceilings[tag] = float(ceiling.mean())
        if arm.head:
            source_row = ref_row if arm.prompt == "ref" else cue_rows[run.label]
            x3_all = tensors[f"{prefix}/x3_all/{schedule[source_row].tag}"]
            reference = atp.ReferenceRow(progs.readout.programs[hp.HEAD_LAYER], [x.double() for x in x3_all])
            captured_row = field("attn_pt")[source_row][hp.HEAD_INDEX, : p_t + 1].double()
            book.tol("G5_row", float((reference.A_ref[hp.HEAD_INDEX] - captured_row).abs().max()), where, tag)
            output = torch.einsum("hk,hkd->hd", reference.A_ref, reference.program.output(reference.values))[hp.HEAD_INDEX].double()
            d_T = geometry.d_T.double()
            book.tol("G5_output", abs(float(output @ d_T) - float(field("T_pt")[source_row].double() @ d_T)), where, tag)
            book.bit("G5_landing", bool(torch.equal(field("T_pt")[row], tensors[f"{prefix}/head_replacement/{tag}"])), where)
    primary_label = config.primary[0]
    noop_row, pl_row = index[tag_of("NOOP", primary_label)], cue_rows[primary_label]
    same = all(bool(torch.equal(field(name)[noop_row], field(name)[pl_row])) for name in CAPTURE_FIELDS) \
        and bool(torch.equal(contrasts[noop_row], contrasts[pl_row])) and digests[noop_row] == digests[pl_row] \
        and bool(torch.equal(field("logit_identity")[noop_row], field("logit_identity")[pl_row]))
    book.bit("G3_noop", same, schedule[noop_row].key)
    book.bit("G6_c_ref", bool(torch.equal(tensors[f"{prefix}/c_ref_pm"].double(), g6_state.c_ref.double())), prefix)
    book.bit("G6_state", bool(torch.equal(x3pc[ref_row].double(), g6_state.x3_all[p_c].double()) and torch.equal(x3pt[ref_row].double(), g6_state.x3_all[p_t].double())
                              and torch.equal(h6[ref_row].double(), g6_state.h6.double())), prefix)
    return ceilings


def run_gates(progs: ul.ModelPrograms, schedules: Sequence[Sequence[RunSpec]], tensors: Mapping[str, torch.Tensor], digests: Mapping[str, Sequence[str]],
              program_states: Mapping[str, rd.FrameState020], g6_states: Mapping[str, rd.FrameState020], geometry: ReadGeometry,
              config: Configuration) -> tuple[GateBook, dict[str, dict[str, float]]]:
    book = GateBook()
    ceilings = {}
    for schedule in schedules:
        frame_id = schedule[0].frame.frame_id
        ceilings[frame_id] = frame_gates(book, progs, schedule, tensors, digests[frame_id], program_states[frame_id], g6_states[frame_id], geometry, config)
    return book, ceilings


def run_records(progs: ul.ModelPrograms, schedules: Sequence[Sequence[RunSpec]], tensors: Mapping[str, torch.Tensor], geometry: ReadGeometry,
                config: Configuration) -> dict[str, Any]:
    """The per-run records (descriptive): the LayerNorm records of every residual edit (design §2), and the manipulation
    check of every run, ``ΔT = ⟨T(run) − T(REF), d̂_T⟩`` with ``T`` the captured ``L03.H04`` at ``p_t``, and its ratio
    to the plain cue run's (design §7)."""
    gamma_ln2 = progs.lw.ln_w[hp.HEAD_LAYER].double()
    d_T = geometry.d_T.double()
    out: dict[str, Any] = {}
    for schedule in schedules:
        frame = schedule[0].frame
        prefix = frame.frame_id
        index = _index(schedule)
        x3pc, T = tensors[f"{prefix}/x3_pc"], tensors[f"{prefix}/T_pt"]
        ref_row = index["REF"]
        delta_T = {run.tag: float((T[row].double() - T[ref_row].double()) @ d_T) for row, run in enumerate(schedule)}
        frame_out: dict[str, Any] = {}
        for row, run in enumerate(schedule):
            entry: dict[str, Any] = {"delta_T": delta_T[run.tag]}
            if run.label is not None:
                entry["delta_T_over_PL"] = ratio(delta_T[run.tag], delta_T[tag_of("PL", run.label)])
            if run.arm.residual_edit:
                base_row = ref_row if run.arm.prompt == "ref" else index[tag_of("PL", run.label)]
                e = x3pc[row].double() - x3pc[base_row].double()
                entry.update(layer_norm_records(progs.head, gamma_ln2, x3pc[row].double(), e))
            frame_out[run.tag] = entry
        out[prefix] = frame_out
    return encode_nonfinite(out)


# ---------------------------------------------------------------------------
# 7. The measured quantities (computation 7).


def frame_values(schedule: Sequence[RunSpec], tensors: Mapping[str, torch.Tensor], config: Configuration, *, fractions_for_weak: bool) -> dict[str, Any]:
    """Per cue: ``Δc`` of every arm (the mean over the scorable nouns of the arm's contrasts minus REF's, both from the
    same stage), ``Δc_full``, the weak flag, ``A_amp``, ``A_neg`` and the fractions. A weak frame's fractions are not
    computed when ``fractions_for_weak`` is off (``confirm``); calibration computes and flags them."""
    prefix = schedule[0].frame.frame_id
    contrasts = tensors[f"{prefix}/contrasts"].double()
    index = _index(schedule)
    c_ref = contrasts[index["REF"]]
    dc_by_tag = {run.tag: float((contrasts[row] - c_ref).mean()) for row, run in enumerate(schedule)}
    out: dict[str, Any] = {}
    for label in config.cue_labels:
        dc = {arm.name: dc_by_tag[tag_of(arm.name, label)] for arm in config.arms_for(label)}
        full = dc["PL"]
        weak = full > WEAK_SHIFT
        entry: dict[str, Any] = {"dc_full": full, "weak": weak, "dc": dc, "A_amp": dc["NEC-AMP"] - full, "A_neg": dc["SUF-NEG"], "fractions": None}
        if fractions_for_weak or not weak:
            fractions = main_fractions(dc, full)
            if label == config.primary[0]:
                rem_iso = [ratio(dc[f"NEC-ISO-{j}"], full) for j in range(1, config.k_iso + 1)]
                ins_iso = [ratio(dc[f"SUF-ISO-{j}"], full) for j in range(1, config.k_iso + 1)]
                fractions.update({"F_rem_iso": rem_iso, "F_ins_iso": ins_iso, "iso_rem_margin": median(rem_iso) - fractions["F_rem"],
                                  "iso_ins_margin": fractions["F_ins"] - median(ins_iso)})
                for variant, suffix in (("", ""), ("_v", "-V")):
                    fractions[f"F_head_rem{variant}"] = ratio(dc[f"HEAD-NEC{suffix}"], full)
                    fractions[f"F_head_ins{variant}"] = ratio(dc[f"HEAD-SUF{suffix}"], full)
                    removal_ok = 1.0 - fractions["F_rem"] >= HEAD_SHARE_GUARD
                    install_ok = fractions["F_ins"] >= HEAD_SHARE_GUARD
                    fractions[f"head_removal_share{variant}"] = ratio(full - dc[f"HEAD-NEC{suffix}"], full - dc["NEC-READ"]) if removal_ok else None
                    fractions[f"head_install_share{variant}"] = ratio(dc[f"HEAD-SUF{suffix}"], dc["SUF-READ"]) if install_ok else None
            entry["fractions"] = fractions
        else:
            entry["reason"] = "weak"
        out[label] = entry
    return out


# ---------------------------------------------------------------------------
# 8. The statistics and the label (computation 8).


def holm_decisions(p_values: Mapping[str, Fraction], alpha: Fraction = HOLM_ALPHA) -> dict[str, bool]:
    """Holm's step-down on the count p-values alone: the smallest against ``alpha/m``, the next against
    ``alpha/(m − 1)``, stopping at the first non-rejection."""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    decisions = {name: False for name in p_values}
    for rank, (name, value) in enumerate(ordered):
        if value <= alpha / (len(ordered) - rank):
            decisions[name] = True
        else:
            break
    return decisions


def outcome_label(*, precondition_met: bool, r_pass: bool, i_pass: bool, d_pass: bool, p_evaluable: bool, p_pass: bool) -> str:
    """Design §7's table, in order. ``NOT_INTERPRETABLE`` comes only from a gate failure or an incident, never from
    here."""
    if not precondition_met:
        return "BEHAVIOR_PRECONDITION_NOT_MET"
    if not r_pass and not i_pass:
        return "READ_COMPONENT_EFFECTS_NOT_ESTABLISHED"
    if r_pass and not i_pass:
        return "READ_REMOVAL_EFFECT_INSTALLATION_NOT_ESTABLISHED"
    if i_pass and not r_pass:
        return "READ_INSTALLATION_EFFECT_REMOVAL_NOT_ESTABLISHED"
    if not d_pass:
        return "READ_COMPONENT_EFFECTS_DIRECTION_NOT_ESTABLISHED"
    if not p_evaluable:
        return "READ_COMPONENT_EFFECTS_PROGRAM_NOT_EVALUABLE"
    if not p_pass:
        return "READ_COMPONENT_EFFECTS_PROGRAM_NOT_CONFIRMED"
    return "READ_COMPONENT_EFFECTS_AS_PREDICTED"


def _family_values(values: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    """A family statistic per frame, a weak frame as its worst value (``WORST``)."""
    out = []
    for entry in values:
        if entry["weak"]:
            out.append(WORST[name])
        elif name in ("A_amp", "A_neg"):
            out.append(float(entry[name]))
        else:
            out.append(float(entry["fractions"][name]))
    return out


def _count(values: Sequence[Mapping[str, Any]], name: str, sign: int) -> int:
    """The frames where the statistic has the predicted sign, strictly; weak frames and ties fail."""
    count = 0
    for value, entry in zip(_family_values(values, name), values):
        if not entry["weak"] and math.isfinite(value) and (value > 0.0 if sign > 0 else value < 0.0):
            count += 1
    return count


def primary_statistics(frame_ids: Sequence[str], values: Sequence[Mapping[str, Any]], predicted: Sequence[Mapping[str, Any]], tau_rem: float, tau_ins: float,
                       config: Configuration) -> dict[str, Any]:
    """R and I by Holm, then D, then P, the precondition and the label, for the primary cue. The statistics take all
    frozen frames in order, or nothing: there is no exclusion parameter."""
    n = config.n_frames
    if not (len(frame_ids) == len(values) == len(predicted) == n):
        raise IncidentError(f"the statistics take all {n} frozen frames in order, or nothing")
    counts = {"R": _count(values, "N", 1), "I": _count(values, "S", 1)}
    p_values = {family: binomial_tail(n, count) for family, count in counts.items()}
    rejected = holm_decisions(p_values)
    medians = {"R": median(_family_values(values, "N")), "I": median(_family_values(values, "S"))}
    passes = {family: bool(rejected[family] and medians[family] is not None and medians[family] >= MEDIAN_FLOOR) for family in ("R", "I")}
    d_counts = {"A_amp": _count(values, "A_amp", -1), "A_neg": _count(values, "A_neg", 1)}
    d_reached = passes["R"] and passes["I"]
    d_pass = d_counts["A_amp"] >= config.d_threshold and d_counts["A_neg"] >= config.d_threshold
    evaluable = tau_rem <= TAU_CAP and tau_ins <= TAU_CAP
    tolerance = {"rem": max(tau_rem, TAU_FLOOR), "ins": max(tau_ins, TAU_FLOOR)}
    frames, agreements = [], 0
    for frame_id, entry, prediction in zip(frame_ids, values, predicted):
        record = {"frame_id": frame_id, "dc_full": entry["dc_full"], "weak": entry["weak"], "predicted_weak": bool(prediction["weak"]),
                  "A_amp": entry["A_amp"], "A_neg": entry["A_neg"]}
        agree = False
        if not entry["weak"] and not prediction["weak"]:
            measured, expected = entry["fractions"], prediction["fractions"]
            error_rem, error_ins = abs(expected["F_rem"] - measured["F_rem"]), abs(expected["F_ins"] - measured["F_ins"])
            agree = error_rem <= tolerance["rem"] and error_ins <= tolerance["ins"]
            record.update({name: measured[name] for name in ("F_rem", "F_rem_off", "F_ins", "F_ins_off", "N", "S")})
            record.update({"F_rem_hat": expected["F_rem"], "F_ins_hat": expected["F_ins"], "N_hat": expected["N"], "S_hat": expected["S"],
                           "P_error_rem": error_rem, "P_error_ins": error_ins})
        elif not entry["weak"]:
            record.update({name: entry["fractions"][name] for name in ("F_rem", "F_rem_off", "F_ins", "F_ins_off", "N", "S")})
        record["P_agree"] = agree
        agreements += int(agree)
        frames.append(record)
    p_reached = d_reached and d_pass
    p_pass = agreements >= config.p_threshold
    precondition = sum(1 for entry in values if entry["dc_full"] <= WEAK_SHIFT)
    precondition_met = precondition >= config.precondition
    label = outcome_label(precondition_met=precondition_met, r_pass=passes["R"], i_pass=passes["I"], d_pass=d_pass, p_evaluable=evaluable, p_pass=p_pass)
    better, other = config.holm
    results = {
        "n": n, "cue": config.primary[0],
        "precondition": {"frames_at_most_minus_one": precondition, "required": config.precondition, "met": precondition_met},
        "R": {"count": counts["R"], "p_value": f"{p_values['R'].numerator}/{p_values['R'].denominator}", "p_value_float": float(p_values["R"]),
              "holm_rejected": rejected["R"], "median": medians["R"], "median_floor": MEDIAN_FLOOR, "result": "PASS" if passes["R"] else "FAIL"},
        "I": {"count": counts["I"], "p_value": f"{p_values['I'].numerator}/{p_values['I'].denominator}", "p_value_float": float(p_values["I"]),
              "holm_rejected": rejected["I"], "median": medians["I"], "median_floor": MEDIAN_FLOOR, "result": "PASS" if passes["I"] else "FAIL"},
        "holm": {"alpha": f"{HOLM_ALPHA.numerator}/{HOLM_ALPHA.denominator}", "thresholds": [better, other]},
        "D": {"reached": d_reached, "A_amp_negative": d_counts["A_amp"], "A_neg_positive": d_counts["A_neg"], "threshold": config.d_threshold,
              "median_A_amp": median(_family_values(values, "A_amp")), "median_A_neg": median(_family_values(values, "A_neg")),
              "result": ("PASS" if d_pass else "FAIL") if d_reached else "NOT_REACHED", "descriptive_pass": d_pass},
        "P": {"reached": p_reached, "tau_rem": tau_rem, "tau_ins": tau_ins, "evaluable": evaluable, "tolerance": tolerance, "agreements": agreements,
              "threshold": config.p_threshold,
              "result": (("PASS" if p_pass else "FAIL") if evaluable else "NOT_EVALUABLE") if p_reached else "NOT_REACHED", "descriptive_pass": p_pass},
        "frames": frames,
        "outcome": {"label": label, "scope": SEMANTICS["scope"], "non_claims": list(SEMANTICS["non_claims"])},
    }
    return encode_nonfinite(results)


# ---------------------------------------------------------------------------
# 9. The secondary statistics (after the result is written; no label force) and the bootstrap.


def bootstrap_index(n: int, draws: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> torch.Tensor:
    """One ``[draws, n]`` index matrix, drawn once and shared by every median's interval."""
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, int(n), (int(draws), int(n)), generator=generator)


def interval_ranks(draws: int = BOOTSTRAP) -> tuple[int, int]:
    """The nearest-rank 2.5th and 97.5th percentiles of ``draws`` sorted medians: 250 and 9,750 of 10,000."""
    return math.ceil(Fraction(25, 1000) * draws), math.ceil(Fraction(975, 1000) * draws)


def row_medians(samples: torch.Tensor) -> torch.Tensor:
    """The median rule per row, ignoring NaN (an excluded frame); ±∞ follow ``median``."""
    ordered = samples.sort(dim=1).values  # NaN sorts last
    valid = (~torch.isnan(ordered)).sum(dim=1)
    low = ordered.gather(1, ((valid - 1).clamp(min=0) // 2).unsqueeze(1)).squeeze(1)
    high = ordered.gather(1, (valid // 2).clamp(max=ordered.shape[1] - 1).unsqueeze(1)).squeeze(1)
    out = 0.5 * (low + high)
    out[valid == 0] = math.nan
    return out


def median_with_interval(values: Sequence[float | None], index: torch.Tensor) -> dict[str, Any]:
    """The median of the defined values (``None`` excluded) and its frame-bootstrap 95 % interval on the shared index."""
    vector = torch.tensor([math.nan if value is None else float(value) for value in values], dtype=torch.float64)
    medians = row_medians(vector[index]).sort().values
    low, high = interval_ranks(int(index.shape[0]))
    defined = [float(value) for value in values if value is not None]
    return {"median": median(defined), "n": len(defined), "excluded": len(values) - len(defined), "interval": [float(medians[low - 1]), float(medians[high - 1])]}


def _guarded(values: Sequence[Mapping[str, Any]], name: str, worst: float) -> tuple[list[float | None], int]:
    """A guarded share per frame: a weak frame as ``worst``, a frame the guard excludes as ``None`` (counted)."""
    out, guarded = [], 0
    for entry in values:
        if entry["weak"]:
            out.append(worst)
        elif entry["fractions"][name] is None:
            out.append(None)
            guarded += 1
        else:
            out.append(float(entry["fractions"][name]))
    return out, guarded


def _direction_free(values: Sequence[Mapping[str, Any]], name: str) -> list[float | None]:
    return [None if entry["weak"] or entry["fractions"] is None else float(entry["fractions"][name]) for entry in values]


def secondary_statistics(frame_ids: Sequence[str], values_by_cue: Mapping[str, Sequence[Mapping[str, Any]]], primary: Mapping[str, Any],
                         records: Mapping[str, Any], config: Configuration) -> dict[str, Any]:
    """Everything with no label force: the primary medians' intervals, the stratum, the isotropic controls, the head
    arms, the direction-free summaries, P's errors and the program's ``N̂``/``Ŝ``, the manipulation check and the
    LayerNorm records."""
    index = bootstrap_index(len(frame_ids))
    primary_label = config.primary[0]
    values = values_by_cue[primary_label]
    out: dict[str, Any] = {"bootstrap": {"draws": int(index.shape[0]), "seed": BOOTSTRAP_SEED, "ranks": list(interval_ranks(int(index.shape[0])))}}
    out["primary_medians"] = {name: median_with_interval(_family_values(values, name), index) for name in ("N", "S", "A_amp", "A_neg")}
    stratum = {}
    for label in config.cue_labels[1:]:
        cue_values = values_by_cue[label]
        stratum[label] = {"R_count": _count(cue_values, "N", 1), "I_count": _count(cue_values, "S", 1), "A_amp_negative": _count(cue_values, "A_amp", -1),
                          "A_neg_positive": _count(cue_values, "A_neg", 1), "weak": sum(1 for entry in cue_values if entry["weak"]),
                          "medians": {name: median_with_interval(_family_values(cue_values, name), index) for name in ("N", "S", "A_amp", "A_neg")},
                          "direction_free": {name: median_with_interval(_direction_free(cue_values, name), index) for name in SUMMARY_FREE}}
    out["stratum"] = stratum
    iso = {}
    for name in ("iso_rem_margin", "iso_ins_margin"):
        per_frame = [WORST["iso_margin"] if entry["weak"] else float(entry["fractions"][name]) for entry in values]
        iso[name] = {"positive": sum(1 for value, entry in zip(per_frame, values) if not entry["weak"] and value > 0.0), **median_with_interval(per_frame, index)}
    out["isotropic"] = iso
    head: dict[str, Any] = {}
    for name, (worst, sign, reference) in HEAD_COUNTS.items():
        per_frame = [WORST[worst] if entry["weak"] else float(entry["fractions"][name]) for entry in values]
        counted = sum(1 for value, entry in zip(per_frame, values) if not entry["weak"] and (value - reference) * sign > 0.0)
        head[name] = {"count": counted, "criterion": f"{name} {'<' if sign < 0 else '>'} {reference}", **median_with_interval(per_frame, index)}
    for name in ("head_removal_share", "head_removal_share_v", "head_install_share", "head_install_share_v"):
        per_frame, guarded = _guarded(values, name, WORST["head_share"])
        head[name] = {**median_with_interval(per_frame, index), "guard_excluded": guarded}
    out["head"] = head
    out["direction_free"] = {name: median_with_interval(_direction_free(values, name), index) for name in SUMMARY_FREE}
    errors = {"P_error_rem": [], "P_error_ins": [], "N_hat_minus_N": [], "S_hat_minus_S": []}
    for frame in primary["frames"]:
        defined = frame.get("P_error_rem") is not None
        errors["P_error_rem"].append(frame["P_error_rem"] if defined else None)
        errors["P_error_ins"].append(frame["P_error_ins"] if defined else None)
        errors["N_hat_minus_N"].append(frame["N_hat"] - frame["N"] if defined else None)
        errors["S_hat_minus_S"].append(frame["S_hat"] - frame["S"] if defined else None)
    out["program"] = {name: median_with_interval(series, index) for name, series in errors.items()}
    manipulation: dict[str, list[float | None]] = {}
    layer_norm: dict[str, dict[str, list[float]]] = {}
    for frame_id in frame_ids:
        for tag, entry in records[frame_id].items():
            manipulation.setdefault(tag, []).append(entry.get("delta_T_over_PL"))
            if "sigma" in entry:
                bucket = layer_norm.setdefault(tag, {"sigma": [], "gamma3_e_over_sigma": [], "gamma_ln2_e_over_sigma": []})
                for key in bucket:
                    bucket[key].append(float(entry[key]))
    out["manipulation_check"] = {tag: median_with_interval(series, index) for tag, series in sorted(manipulation.items()) if any(v is not None for v in series)}
    out["layer_norm"] = {tag: {key: {"median": median(series), "max": max(series)} for key, series in bucket.items()} for tag, bucket in sorted(layer_norm.items())}
    return encode_nonfinite(out)


# ---------------------------------------------------------------------------
# 10. Calibration (exposed, exploratory; design §6): τ, the revision triggers, the reports and the power check.

CALIBRATION_SCHEMA_VERSION = 1


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _nonfinite_fractions(entry: Mapping[str, Any] | None) -> list[str]:
    if entry is None:
        return []
    bad = []
    for name, value in entry.items():
        if name.endswith(("_nonfinite", "_nonfinite_at")):
            bad.append(name)
        elif isinstance(value, list):
            bad += [f"{name}[{i}]" for i, item in enumerate(value) if item is None or not _finite(item)]
        elif value is not None and not _finite(value):
            bad.append(name)
    return bad


def binomial_power(n: int, share: float, threshold: int) -> float:
    """``P(X ≥ threshold)`` for ``X ~ Binomial(n, share)`` (the sign-test power at an observed share)."""
    share = min(max(float(share), 0.0), 1.0)
    return math.fsum(math.comb(n, k) * share ** k * (1.0 - share) ** (n - k) for k in range(int(threshold), int(n) + 1))


def calibration_results(frame_ids: Sequence[str], values_by_cue: Mapping[str, Sequence[Mapping[str, Any]]], table: Mapping[str, Any],
                        pairs: Mapping[str, Mapping[str, Any]], gates: GateBook, environment: Mapping[str, Any], config: Configuration,
                        extra_failures: Sequence[str] = ()) -> dict[str, Any]:
    """``τ_rem`` and ``τ_ins`` for the primary cue (the nearest-rank 90th percentile, all frames entering), the four
    revision triggers, the reports, the exposed distributions and the power check. The size of an exposed result is
    never a trigger."""
    primary_label = config.primary[0]
    values = values_by_cue[primary_label]
    predicted = [table["frames"][frame_id]["cues"][primary_label] for frame_id in frame_ids]
    nonfinite = []
    for frame_id in frame_ids:
        for label in config.cue_labels:
            entry = table["frames"][frame_id]["cues"][label]
            nonfinite += [f"{frame_id}|{label}|predicted {name}" for name in _nonfinite_fractions(entry["fractions"])]
    for label, cue_values in values_by_cue.items():
        for frame_id, entry in zip(frame_ids, cue_values):
            nonfinite += [f"{frame_id}|{label}|measured {name}" for name in _nonfinite_fractions(entry["fractions"])]
    errors_rem, errors_ins = [], []
    for entry, prediction in zip(values, predicted):
        measured, expected = entry["fractions"] or {}, prediction["fractions"] or {}
        if _finite(measured.get("F_rem")) and _finite(expected.get("F_rem")) and _finite(measured.get("F_ins")) and _finite(expected.get("F_ins")):
            errors_rem.append(abs(float(expected["F_rem"]) - float(measured["F_rem"])))
            errors_ins.append(abs(float(expected["F_ins"]) - float(measured["F_ins"])))
    tau: dict[str, Any] = {"rank": config.tau_rank, "cap": TAU_CAP, "floor": TAU_FLOOR}
    if len(errors_rem) == len(frame_ids):
        tau.update({"rem": sorted(errors_rem)[config.tau_rank - 1], "ins": sorted(errors_ins)[config.tau_rank - 1]})
        tau["evaluable"] = tau["rem"] <= TAU_CAP and tau["ins"] <= TAU_CAP
    else:
        tau.update({"rem": None, "ins": None, "evaluable": False})
    overshoot = [frame_id for frame_id in frame_ids if pairs[frame_id][primary_label]["overshoot"]]
    weak = [frame_id for frame_id, entry in zip(frame_ids, values) if entry["weak"]]
    gate_failures = gates.failures() + list(extra_failures) + (["environment"] if not environment.get("passed") else [])
    triggers = {"gate_failure": {"fired": bool(gate_failures), "gates": gate_failures},
                "overshoot": {"fired": len(overshoot) > config.max_overshoot_frames, "frames": overshoot, "limit": config.max_overshoot_frames},
                "weak": {"fired": len(weak) > config.max_weak_frames, "frames": weak, "limit": config.max_weak_frames},
                "nonfinite_fraction": {"fired": bool(nonfinite), "where": nonfinite}}
    n = len(frame_ids)
    shares = {"N_positive": _count(values, "N", 1) / n, "S_positive": _count(values, "S", 1) / n, "A_amp_negative": _count(values, "A_amp", -1) / n,
              "A_neg_positive": _count(values, "A_neg", 1) / n}
    agree = 0
    if tau.get("rem") is not None:
        tolerance_rem, tolerance_ins = max(tau["rem"], TAU_FLOOR), max(tau["ins"], TAU_FLOOR)
        agree = sum(1 for entry, prediction in zip(values, predicted) if not entry["weak"] and not prediction["weak"]
                    and abs(prediction["fractions"]["F_rem"] - entry["fractions"]["F_rem"]) <= tolerance_rem
                    and abs(prediction["fractions"]["F_ins"] - entry["fractions"]["F_ins"]) <= tolerance_ins)
    shares["P_agree"] = agree / n
    better, other = config.holm
    power = {"R_or_I_at_better": {name: binomial_power(n, shares[name], better) for name in ("N_positive", "S_positive")},
             "R_or_I_at_other": {name: binomial_power(n, shares[name], other) for name in ("N_positive", "S_positive")},
             "D": {name: binomial_power(n, shares[name], config.d_threshold) for name in ("A_amp_negative", "A_neg_positive")},
             "P": binomial_power(n, shares["P_agree"], config.p_threshold)}
    reports = {frame_id: {"pairs": dict(pairs[frame_id]), "dc_full": {label: values_by_cue[label][i]["dc_full"] for label in config.cue_labels}}
               for i, frame_id in enumerate(frame_ids)}
    # G2's practical floor: float32 rounding of x′ is about 1e-6·‖x‖, so a frame with s near 0.01 would strain G2's relative 1e-4 at
    # confirm; the calibration review reads the smallest s per cue
    min_s = {label: min(((pairs[frame_id][label]["s"], frame_id) for frame_id in frame_ids), key=lambda item: item[0]) for label in config.cue_labels}
    distributions = {name: sorted(value for value in _family_values(values, name) if math.isfinite(value)) for name in ("N", "S", "A_amp", "A_neg")}
    distributions.update({"P_error_rem": sorted(errors_rem), "P_error_ins": sorted(errors_ins)})
    return encode_nonfinite({"tau": tau, "triggers": triggers, "stopped": any(entry["fired"] for entry in triggers.values()), "shares": shares,
                             "power": power, "reports": reports, "distributions": distributions, "table_content_sha256": table["content_sha256"],
                             "min_s": {label: {"s": value, "at": frame_id} for label, (value, frame_id) in min_s.items()}})


def calibration_payload(*, config: Configuration, confirmation_binding_: Mapping[str, Any], results: Mapping[str, Any], gates: Mapping[str, Any],
                        environment: Mapping[str, Any], g1: Mapping[str, Any], keys_sha256: str, n_keys: int, tensors_sha256: Mapping[str, str],
                        protocol_code_commit: str) -> dict[str, Any]:
    payload = {"experiment": EXPERIMENT, "schema_version": CALIBRATION_SCHEMA_VERSION,
               "kind": "the exposed calibration of Experiment 101 (exploratory; spent data): τ_rem and τ_ins, and the G4 viability record",
               "design": dict(DESIGN), "plan": dict(PLAN), "configuration": config.to_json(), "confirmation_101": dict(confirmation_binding_),
               "protocol_code_commit": protocol_code_commit, "tau": dict(results["tau"]), "triggers": dict(results["triggers"]), "gates": dict(gates),
               "environment": dict(environment), "g1": dict(g1), "shares": dict(results["shares"]), "power": dict(results["power"]),
               "reports": dict(results["reports"]), "distributions": dict(results["distributions"]), "min_s": dict(results["min_s"]),
               "table_content_sha256": results["table_content_sha256"], "module_blob": own_blob(), "tolerances": dict(TOLERANCES),
               "keys": {"count": int(n_keys), "sha256": keys_sha256}, "measurements_sha256": dict(tensors_sha256)}
    payload = encode_nonfinite(payload)
    payload["content_sha256"] = rc.content_digest(payload)
    return payload


def load_calibration_101(path: Path, config: Configuration, confirmation: Confirmation101, confirmation_file_sha256: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("experiment") != EXPERIMENT or payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION or payload.get("content_sha256") != rc.content_digest(payload):
        raise PhaseError("the calibration file is not a verified Experiment 101 calibration record")
    if payload.get("configuration") != config.to_json() or payload.get("confirmation_101") != confirmation_binding(confirmation, confirmation_file_sha256):
        raise PhaseError("the calibration file was written under another configuration or against another confirmation file")
    if any(entry.get("fired") for entry in payload["triggers"].values()) or not _finite(payload["tau"].get("rem")) or not _finite(payload["tau"].get("ins")):
        raise PhaseError("the calibration file records a revision trigger or no τ; it cannot be installed")
    if payload.get("module_blob") != own_blob() or payload.get("tolerances") != json.loads(pm.canonical_json(TOLERANCES)):
        raise PhaseError("the calibration was computed with another analysis module or other tolerances (rule 1: no silent tolerance tuning)")
    return payload


def calibration_binding(calibration: Mapping[str, Any], file_sha256: str) -> dict[str, Any]:
    tau = calibration["tau"]
    return {"path": CALIBRATION_RELATIVE_PATH, "file_sha256": file_sha256, "content_sha256": calibration["content_sha256"], "tau_rem": tau["rem"],
            "tau_ins": tau["ins"], "evaluable": bool(tau["evaluable"])}


# ---------------------------------------------------------------------------
# 11. The lock (weights only), the preregistration, the lock's validation, the scientific paths.

LOCK_SCHEMA_VERSION = 1
MEASUREMENT = ("one forward per tagged key through ri.run_arm: a plain run through pm.capture_prompt, an edit through pm.run_patched (REPLACE at "
               "RESID_PRE.L3@p_c with the float32 cast of the float64 edit, or at L03.H04@p_t for a head arm); the capture set of ri.capture_sites; "
               "Δc(arm) = mean over the scorable nouns of NounSet.contrasts(arm) − NounSet.contrasts(REF), both from stage 2")
STATISTICS = {
    "fractions": "F_rem = Δc(NEC-READ)/Δc_full, F_rem_off = Δc(NEC-OFF)/Δc_full, F_ins = Δc(SUF-READ)/Δc_full, F_ins_off = Δc(SUF-OFF)/Δc_full, "
                 "A_amp = Δc(NEC-AMP) − Δc_full, A_neg = Δc(SUF-NEG); a weak frame's fractions are not computed at confirm",
    "median": "the middle value for odd n; the mean of the n/2-th and (n/2 + 1)-th sorted values for even n (18th and 19th of 36); the mean of an "
              "infinite and a finite value is the infinite one",
    "worst": "a weak frame enters a median as its worst value: −∞ for N, S, A_neg, F_head_ins, the head shares and the isotropic margins; +∞ for "
             "A_amp and F_head_rem",
    "head_counts": "the head arms' counts (descriptive): F_head_rem < 1 and F_head_rem_v < 1 (the head-only removal leaves less than the full "
                   "shift), F_head_ins > 0 and F_head_ins_v > 0 (the head-only installation produces some of it); weak frames fail; the shares are "
                   "reported only under their guards, a guard-excluded frame counted and left out",
    "direction_free": "the midpoint and PC-SWAP fractions, the additivity and P's errors exclude weak frames, with their count",
    "strict": "N_f > 0, S_f > 0, A_amp < 0, A_neg > 0 strictly; ties fail; the precondition counts Δc_full ≤ −1; P agrees when both errors are ≤ "
              "max(τ, 0.02); medians pass at ≥ 0.05; counts pass at ≥ the threshold; P is not evaluable when τ_rem or τ_ins > 0.20",
    "holm": "exact binomial tails as Fractions; Holm at 1/40 on the two count p-values alone (the smaller against 1/80, the larger against 1/40)",
    "bootstrap": "one [10,000 × 36] index matrix from torch.Generator().manual_seed(20261001) and torch.randint, shared by every median; the "
                 "interval is the 250th and the 9,750th of the sorted resampled medians",
}


def scientific_dependencies(inputs: ul.FrozenInputs, *, parameters_sha256: str, embedding_sha256: str) -> dict[str, Any]:
    locked = inputs.closure["exploration"]["locked_states"]
    return {"module_blobs": dict(FROZEN_BLOBS), "measurement": MEASUREMENT, "table": TABLE_CONSTRUCTION,
            "readout_020": {"module": "readout_decompilation.py", "blob": FROZEN_BLOBS["readout_decompilation.py"], "exposed_states_sha256": ul.exposed_states_digest(locked),
                            "frozen_input_digests": {key: inputs.digests[key] for key in rc.DIGEST_KEYS}},
            "model": {"model_id": models_module.PYTHIA_70M.model_id, "revision": models_module.PYTHIA_70M.revision, "parameters_sha256": parameters_sha256,
                      "embedding_sha256": embedding_sha256}}


def stage1_binding(frames: Sequence[pm.Frame], config: Configuration, *, d_model: int, n_heads: int, n_nouns: int, noun_keys: Sequence[str]) -> dict[str, Any]:
    return {"paths": {"states_data": STAGE1_STATES_DATA, "states_index": STAGE1_STATES_INDEX, "table": STAGE1_TABLE},
            "states": {"layout": states_layout(frames, d_model, n_heads, n_nouns), "meta": states_meta(frames)}, "table": table_meta(config, noun_keys),
            "freeze": "written once at stage 1 directly at these paths, with their digests in the stage-1 record; committed by the user before confirm; the "
                      "barrier re-reads them, checks their digests, the layout and the metadata, and recomputes the table bit for bit before any S2 key"}


def build_lock(*, run_id: str, protocol_code_commit: str, digests: Mapping[str, str], config: Configuration, confirmation: Confirmation101,
               confirmation_file_sha256: str, calibration: Mapping[str, Any], calibration_file_sha256: str, geometry: Mapping[str, Any],
               dependencies: Mapping[str, Any], noun_keys: Sequence[str], stage1: Mapping[str, Any]) -> dict[str, Any]:
    lock = {
        "experiment": EXPERIMENT, "schema_version": LOCK_SCHEMA_VERSION, "kind": "the Experiment 101 preregistration lock", "run_id": run_id,
        "protocol_code_commit": protocol_code_commit, "design": dict(DESIGN), "plan": dict(PLAN), "configuration": config.to_json(), "inputs": dict(digests),
        "module_blobs": dict(FROZEN_BLOBS), "module": {"path": "src/neural_decompiler/read_intervention.py", "blob": own_blob()},
        "inherited_025": {"path": INHERITED_025_PATH, **INHERITED_025}, "confirmation_101": confirmation_binding(confirmation, confirmation_file_sha256),
        "calibration_101": calibration_binding(calibration, calibration_file_sha256), "dependencies": dict(dependencies), "noun_keys": list(noun_keys),
        "geometry": dict(geometry), "run_order": list(config.tags()), "cue_order": list(config.cue_labels), "stage1": dict(stage1),
        "statistics": dict(STATISTICS), "constants": statistics_constants(), "tolerances": dict(TOLERANCES),
        "outcome": {"table": [list(row) for row in OUTCOME_TABLE], "labels": list(OUTCOMES)}, "semantics": dict(SEMANTICS),
    }
    lock = json.loads(pm.canonical_json(encode_nonfinite(lock)))
    lock["content_sha256"] = rc.content_digest(lock)
    return lock


def statistics_constants() -> dict[str, Any]:
    return encode_nonfinite({"weak_shift": WEAK_SHIFT, "holm_alpha": f"{HOLM_ALPHA.numerator}/{HOLM_ALPHA.denominator}", "median_floor": MEDIAN_FLOOR,
                             "tau_cap": TAU_CAP, "tau_floor": TAU_FLOOR, "head_share_guard": HEAD_SHARE_GUARD, "bootstrap": BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
                             "interval_ranks": list(interval_ranks()), "worst": dict(WORST), "direction_free": list(DIRECTION_FREE),
                             "head_counts": {name: list(spec) for name, spec in HEAD_COUNTS.items()}})


def render_preregistration(lock: Mapping[str, Any]) -> str:
    config, geometry, calibration = lock["configuration"], lock["geometry"], lock["calibration_101"]
    holm = config["holm_thresholds"]
    lines = ["# Experiment 101 — preregistration", "",
             f"- Lock `{lock['content_sha256']}`; run `{lock['run_id']}` at `{lock['protocol_code_commit']}`; design revision {lock['design']['revision']} "
             f"(`{lock['design']['commit']}`, decided `{lock['design']['decision']}`), plan revision {lock['plan']['revision']} (`{lock['plan']['commit']}`, "
             f"G4 decided `{lock['plan']['decision']}`)",
             f"- The analysis module `{lock['module']['path']}` at blob `{lock['module']['blob']}`; configuration `{config['name']}`",
             f"- The freeze `{lock['confirmation_101']['content_sha256']}`; manifest `{lock['confirmation_101']['manifest_sha256']}` "
             f"({lock['confirmation_101']['counts']['S1-REF']} S1-REF keys, {lock['confirmation_101']['counts']['S2-TARGET']} S2-TARGET keys, "
             f"{config['runs_per_frame']} runs per frame)",
             f"- The calibration `{calibration['content_sha256']}`: τ_rem {calibration['tau_rem']!r}, τ_ins {calibration['tau_ins']!r}; P evaluable "
             f"{calibration['evaluable']}", "",
             "## The variable", "",
             f"- `w_c` = the centred 011 read weight (digest `{geometry['w_c_sha256']}`, ‖w_c‖ {geometry['w_c_norm']!r}); the edit at RESID_PRE.L3, p_c",
             f"- The isotropic controls: {geometry['k_iso']} per frame, tag `{geometry['iso_tag']}`", f"- û_off: {geometry['u_off_rule']}",
             f"- The cue order {', '.join(lock['cue_order'])}; the run order per frame: {', '.join(lock['run_order'])}", "",
             "## The statistics", ""]
    lines += [f"- **{name}**: {lock['statistics'][name]}" for name in ("fractions", "median", "worst", "direction_free", "strict", "holm", "bootstrap",
                                                                        "head_counts")]
    lines += [f"- **R**: {lock['semantics']['families']['R']}; **I**: {lock['semantics']['families']['I']}",
              f"- Holm thresholds {holm[0]} and {holm[1]} of {config['n_frames']}; median floor {lock['constants']['median_floor']}",
              f"- **D**: {lock['semantics']['families']['D']}; each count at least {config['d_threshold']} of {config['n_frames']}",
              f"- **P**: {lock['semantics']['families']['P']}; at least {config['p_threshold']} of {config['n_frames']}",
              f"- The order: {lock['semantics']['order']}", f"- Weak frames: {lock['semantics']['weak']}",
              f"- The precondition: at least {config['precondition']} of {config['n_frames']} frames with Δc_full ≤ −1", "",
              "## The outcome", "", "| condition | label |", "|---|---|"]
    lines += [f"| {condition} | `{label}` |" for condition, label in lock["outcome"]["table"]]
    lines += ["", f"The scope of every label: {lock['semantics']['scope']}.", "", "Not claimed by any label:"]
    lines += [f"- {item}" for item in lock["semantics"]["non_claims"]]
    lines += ["", "The claim hierarchy:"] + [f"- **{name}**: {text}" for name, text in lock["semantics"]["hierarchy"].items()]
    lines += ["", f"- Secondary: {lock['semantics']['secondary']}", f"- Gates: {lock['semantics']['gates']}", "", "## The gates", ""]
    lines += [f"- {name}: {value!r}" for name, value in sorted(lock["tolerances"].items())]
    lines += ["", "## The stage-1 files", "", f"- {lock['stage1']['freeze']}"]
    lines += [f"- `{path}`" for path in lock["stage1"]["paths"].values()]
    lines.append("")
    return "\n".join(lines)


SCIENTIFIC_PATH_PREFIXES = ("src/", f"{EXPERIMENT_DIR}/", *cr.SCIENTIFIC_PATH_PREFIXES[1:])
NON_SCIENTIFIC_PATHS = (CONFIRMATION_RELATIVE_PATH, CALIBRATION_RELATIVE_PATH, LOCK_RELATIVE_PATH, PREREGISTRATION_RELATIVE_PATH, *STAGE1_PATHS,
                        f"{EXPERIMENT_DIR}/README.md", *cr.NON_SCIENTIFIC_PATHS[3:])
NON_SCIENTIFIC_PREFIXES = (f"{EXPERIMENT_DIR}/evidence/", *cr.NON_SCIENTIFIC_PREFIXES)


def scientific_changes(paths: Sequence[str]) -> list[str]:
    """Scientific paths: all code, and 101's, 025's, 024's … experiment directories, except 101's installed artifacts, the
    three stage-1 files, the READMEs and the evidence directories."""
    return [path for path in paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and path not in NON_SCIENTIFIC_PATHS and not path.startswith(NON_SCIENTIFIC_PREFIXES)]


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], digests: Mapping[str, str], config: Configuration, confirmation: Confirmation101,
                  confirmation_file_sha256: str, calibration: Mapping[str, Any], calibration_file_sha256: str, dependencies: Mapping[str, Any],
                  noun_keys: Sequence[str], preregistration_text: str, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    """The installed lock before any prompt: its digest; that it is this run's candidate; every bound input, module (this
    module's own blob directly), configuration, binding, dependency, constant and wording; the rendered preregistration;
    a clean tree with no scientific change since the lock commit. Every failure is a refusal."""
    if lock.get("experiment") != EXPERIMENT or lock.get("content_sha256") != rc.content_digest(lock):
        raise PhaseError("the installed lock is not a verified Experiment 101 lock")
    if state.get("lock") is None or state["lock"].get("content_sha256") != lock["content_sha256"]:
        raise PhaseError("the installed lock is not the candidate this run wrote")
    if lock["inputs"] != dict(digests) or lock["module_blobs"] != dict(FROZEN_BLOBS) or lock["design"] != dict(DESIGN) or lock["plan"] != dict(PLAN):
        raise PhaseError("the lock was written against different frozen inputs, modules, design or plan")
    if lock["module"] != {"path": "src/neural_decompiler/read_intervention.py", "blob": own_blob()} or rc.program_blob_sha1(Path(cr.__file__)) != FROZEN_BLOBS["cue_rotation.py"]:
        raise PhaseError("the running analysis module is not the one the lock binds (direct module-blob check)")
    if lock["configuration"] != config.to_json() or lock["run_order"] != list(config.tags()) or lock["cue_order"] != list(config.cue_labels):
        raise PhaseError("the lock was written under a different configuration or order")
    if lock["confirmation_101"] != confirmation_binding(confirmation, confirmation_file_sha256):
        raise PhaseError("the lock was written against a different confirmation file")
    if lock["calibration_101"] != calibration_binding(calibration, calibration_file_sha256):
        raise PhaseError("the lock was written against a different calibration file")
    comparable = {key: value for key, value in dependencies.items() if key != "model"}
    cr.verify_dependencies({key: value for key, value in lock["dependencies"].items() if key != "model"}, comparable, "the lock")
    if lock["noun_keys"] != list(noun_keys) or lock["semantics"] != json.loads(pm.canonical_json(SEMANTICS)) or lock["statistics"] != dict(STATISTICS) \
            or lock["constants"] != json.loads(pm.canonical_json(statistics_constants())) or lock["tolerances"] != json.loads(pm.canonical_json(TOLERANCES)) \
            or lock["outcome"]["table"] != [list(row) for row in OUTCOME_TABLE]:
        raise PhaseError("the lock names a different noun order, wording, statistic, constant, tolerance or outcome table")
    if preregistration_text != render_preregistration(lock) or pm.sha256_text(preregistration_text) != state["lock"].get("preregistration_sha256"):
        raise PhaseError("the installed preregistration is not the one this lock renders")
    if not tracked:
        raise PhaseError("the lock and the preregistration must be tracked and committed")
    if git_state.get("dirty"):
        raise PhaseError("a scientific phase requires a clean Git tree")
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = scientific_changes(changed_paths)
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock: {scientific}")


def verify_geometry_against_lock(recomputed: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    """The geometry block recomputed from the weights against the lock's, bit for bit (G1 and the digests, N13)."""
    recomputed = json.loads(pm.canonical_json(encode_nonfinite(dict(recomputed))))
    differing = sorted(key for key in set(recomputed) | set(lock["geometry"]) if recomputed.get(key) != lock["geometry"].get(key))
    return {"bitwise_equal": not differing, "differing": differing}


# ---------------------------------------------------------------------------
# 12. The results state, the phase rules and the report.

RESULTS_SCHEMA_VERSION = 1
PHASES = ("validate", "freeze", "calibrate", "lock", "stage1", "confirm", "report")
STATE_PHASES = ("calibrate", "lock", "stage1", "confirm", "report")


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
            "phases": {phase: {"status": "not_started"} for phase in STATE_PHASES}, "executed_prompt_keys": [], "calibration_executed": [],
            "confirmation_101": None, "calibration": None, "lock": None, "stage1": None, "confirmation": None, "report": None}


def assert_phase_allowed(phase: str, state: Mapping[str, Any] | None) -> None:
    """The one-shot rules. ``calibrate`` creates the state and reruns only after a recorded calibrate incident (024's
    rule; the runner checks the new commit and that no scientific path changed); ``lock``, ``stage1`` and ``confirm``
    run once, each after the previous one completed and while no incident is recorded; ``report`` may re-render."""
    if phase == "calibrate":
        if state is None:
            return
        status = state["phases"]["calibrate"]["status"]
        calibration = state.get("calibration") or {}
        if status == "running" and calibration.get("incidents") and not calibration.get("content_sha256"):
            return
        raise PhaseError(f"calibrate is {status}; it runs once in this protocol version (a stop for review is never retried automatically)")
    if state is None:
        raise PhaseError(f"{phase} requires the results state calibrate creates")
    status = {name: entry["status"] for name, entry in state["phases"].items()}
    if phase == "lock":
        if status["calibrate"] != "complete":
            raise PhaseError("lock requires a completed calibration")
        if status["lock"] != "not_started" or state["phases"]["lock"].get("incidents"):
            raise PhaseError("lock already ran or recorded an incident; a new candidate lock requires a new protocol version")
    elif phase == "stage1":
        if status["lock"] != "complete":
            raise PhaseError("stage1 requires the lock phase")
        if status["stage1"] != "not_started" or state["phases"]["stage1"].get("incidents"):
            raise PhaseError("stage1 already ran or recorded an incident; it runs once, and a new attempt requires a new protocol version with new frames")
    elif phase == "confirm":
        if status["stage1"] != "complete" or (state.get("stage1") or {}).get("incident"):
            raise PhaseError("confirm requires a completed stage 1")
        if status["confirm"] != "not_started" or state["phases"]["confirm"].get("incidents"):
            raise PhaseError("confirm already started or recorded an incident; it runs once, never resumes, and a new attempt requires a new protocol version")
    elif phase == "report":
        confirmation = state.get("confirmation") or {}
        incident = confirmation.get("incident") or (state.get("stage1") or {}).get("incident") or any(state["phases"][name].get("incidents") for name in ("lock", "stage1", "confirm"))
        if status["confirm"] != "complete" and not incident:
            raise PhaseError("report requires a completed confirm or a recorded incident")
    else:
        raise PhaseError(f"unknown phase {phase}")


def _fmt(value: Any) -> str:
    return "—" if value is None else (f"{value:+.6f}" if isinstance(value, float) else str(value))


def render_report(state: Mapping[str, Any]) -> str:
    phases = ", ".join(f"{name} {state['phases'][name]['status']}" for name in STATE_PHASES)
    lines = ["# Experiment 101 — report", "", f"- Run `{state['run_id']}`; phases: {phases}",
             f"- Design revision {state['design']['revision']} (`{state['design']['commit']}`, decided `{state['design']['decision']}`), plan revision "
             f"{state['plan']['revision']} (`{state['plan']['commit']}`); configuration `{state['configuration']['name']}`", ""]
    for name in STATE_PHASES:
        for entry in state["phases"][name].get("incidents", []):
            lines.append(f"- **{name} incident** at `{entry['commit']}`: {entry['message']}")
    stage1 = state.get("stage1") or {}
    confirmation = state.get("confirmation") or {}
    incident = confirmation.get("incident") or stage1.get("incident")
    report_block = state.get("report") or {}
    g6_failed = report_block.get("g6") is not None and not report_block["g6"].get("bitwise_equal", True)
    if incident or any(state["phases"][name].get("incidents") for name in ("stage1", "confirm")):
        if incident:
            lines.append(f"- **Incident** at `{incident['commit']}`: {incident['message']} — the one-shot confirmation is consumed; nothing is retried")
        lines += ["", "## The outcome", "", "**`NOT_INTERPRETABLE`** — an incident carries no result.", ""]
        gates = confirmation.get("gates")
        if gates:
            lines.append(f"- The gate values on disk: failures {gates.get('failures')}")
        return "\n".join(lines) + "\n"
    results = confirmation.get("results")
    if not results:
        lines += ["", "## The outcome", "", "None yet: no confirmation result exists.", ""]
        return "\n".join(lines)
    outcome = results["outcome"]
    if g6_failed:
        lines += ["## The outcome", "", f"**`NOT_INTERPRETABLE`** — G6 failed at the report: the committed table no longer reproduces from its committed "
                  f"inputs ({report_block['g6'].get('message')}). The stored label `{outcome['label']}` is void; the confirm result is not rewritten.", ""]
    else:
        lines += ["## The outcome", "", f"**`{outcome['label']}`**", "", f"The scope of every label: {outcome['scope']}.", "", "Not claimed by any label:"]
        lines += [f"- {item}" for item in outcome["non_claims"]]
    lines += ["", "## The families", "", "| family | count | threshold | median | result |", "|---|---|---|---|---|"]
    holm = results["holm"]["thresholds"]
    for family, threshold in (("R", f"Holm ({holm[0]}/{holm[1]})"), ("I", f"Holm ({holm[0]}/{holm[1]})")):
        entry = results[family]
        lines.append(f"| {family} | {entry['count']} of {results['n']} (p {entry['p_value']}) | {threshold} | {_fmt(entry['median'])} | **{entry['result']}** |")
    d, p = results["D"], results["P"]
    lines.append(f"| D | A_amp < 0: {d['A_amp_negative']}, A_neg > 0: {d['A_neg_positive']} | {d['threshold']} each | — | **{d['result']}** |")
    lines.append(f"| P | {p['agreements']} agree | {p['threshold']} | τ_rem {_fmt(p['tau_rem'])}, τ_ins {_fmt(p['tau_ins'])} | **{p['result']}** |")
    pre = results["precondition"]
    lines += ["", f"- The precondition: {pre['frames_at_most_minus_one']} of {results['n']} frames with Δc_full ≤ −1 (required {pre['required']}): "
              f"{'met' if pre['met'] else 'NOT met'}", f"- The order: {SEMANTICS['order']}", f"- Weak frames: {SEMANTICS['weak']}", "",
              "## The claim hierarchy", ""]
    lines += [f"- **{name}**: {text}" for name, text in SEMANTICS["hierarchy"].items()]
    gates = confirmation.get("gates") or {}
    lines += ["", "## Validity (every gate passed for a result to exist)", "", f"- Gate failures: {gates.get('failures')}",
              f"- The accounting: {confirmation.get('accounting')}", f"- The barrier: {(confirmation.get('barrier') or {}).get('passed')}",
              f"- G6 before the result write: {(confirmation.get('g6_before_result') or {}).get('bitwise_equal')}"]
    if report_block.get("g6") is not None:
        lines.append(f"- G6 at the report: bit for bit {report_block['g6'].get('bitwise_equal')}")
    secondary = confirmation.get("secondary") or {}
    if secondary.get("statistics"):
        stats = secondary["statistics"]
        lines += ["", "## Secondary records (no label force)", "", f"- {SEMANTICS['secondary']}",
                  f"- Primary medians and intervals: {stats.get('primary_medians')}", f"- The stratum: {stats.get('stratum')}",
                  f"- The isotropic controls: {stats.get('isotropic')}", f"- The head arms: {stats.get('head')}", f"- Direction-free: {stats.get('direction_free')}",
                  f"- The program: {stats.get('program')}"]
    for name, failure in sorted((secondary.get("failures") or {}).items()):
        lines.append(f"- The secondary record {name} was not computed ({failure['type']}: {failure['message']}); the result stands")
    lines.append("")
    return "\n".join(lines)


def state_drift(now: Any, recorded: Any) -> float:
    """The maximum absolute difference between two locked states (nested lists and scalars); a structural mismatch is
    infinite. Calibration's environment check: each re-captured exposed reference state against 020's locked one."""
    if isinstance(now, Mapping) and isinstance(recorded, Mapping):
        if set(now) != set(recorded):
            return math.inf
        return max((state_drift(now[key], recorded[key]) for key in now), default=0.0)
    if isinstance(now, (list, tuple)) and isinstance(recorded, (list, tuple)):
        if len(now) != len(recorded):
            return math.inf
        return max((state_drift(a, b) for a, b in zip(now, recorded)), default=0.0)
    if isinstance(now, (int, float)) and isinstance(recorded, (int, float)):
        return abs(float(now) - float(recorded))
    return 0.0 if now == recorded else math.inf
