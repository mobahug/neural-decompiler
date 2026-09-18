"""Experiment 008: where is the apparent number signal of ``this``-like cues suppressed?

A discovery-only causal localization on the fully exposed pool (40 cue tokens, 18 frames, 80
nouns). For every (token, frame) the E-patch intervention is traced through the residual
stream with exact direct-effect deltas (M1, M4), split into its encoding-axis and orthogonal
components (M2), and swapped across cue contexts (M3); the frozen rules of the approved design
(revision 3) classify every token and summarize the suppressed stratum. No lock, no confirmation,
no claim promotion. Constants are copied from the design.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import cue_decompilation as cd
from . import plural_mechanism as pm
from . import supervised_subspace as ss
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest
from .interventions import ReplacementSource
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 3).

EXPERIMENT_DIR = "experiments/008-cue-suppression-localization"
INHERITED_007_EXTRACT_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-007-confirmation-epatch-means.json"
EXPERIMENT_007_LOCK_PATH = "experiments/007-supervised-cue-subspace/preregistration-lock.json"
PROGRAM_007_RELATIVE_DIR = "outputs/experiment-007/parameters/selected"
PROGRAM_007_SOURCE = "experiments/007-supervised-cue-subspace/linear_cue_program.py"
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260921
RESULTS_SCHEMA_VERSION = 1
PHASES = ("explore", "report")
INHERITED_007_SCHEMA_VERSION = 1

S_MIN = 0.3  # a token carries the mechanism-axis signal if |ŝ_R0| ≥ S_MIN
KAPPA = 0.5  # collapse: q_s ≤ KAPPA · q_{s−1}
G_MAX = 0.25  # additivity gap beyond which the encoding response is non-additive
X_MIN = 0.5  # context gating: x_in < X_MIN
C_MIN = 2.0  # late cancellation: C(w) ≥ C_MIN · C(pl_T) and C(w) ≥ C_MIN
PROBE_FLOOR = 0.5  # r_∥(pl_T, f) ≥ PROBE_FLOOR
CONSENSUS = 0.75  # fraction of frames / tokens for validity and summary
STAGE_UNINFORMATIVE_FLOOR = 0.25  # × σ_s on the plural cue's own stage signal
CONTRAST_UNINFORMATIVE_NATS = 0.25  # |Δc(pl_T, f)| below this makes the frame's fractions None
CANCELLATION_DENOMINATOR_FLOOR = 0.25  # × |Δc(pl_T, f)|
STRATUM_THRESHOLD = 1.5  # suppressed a ≤ −1.5, amplified a ≥ +1.5 (descriptive)
IDENTITY_TOLERANCE = 1e-4  # Δ_R0 = ΔE_T(w)
REPLICATION_TOLERANCE = 1e-6

HEAD_KEY, HEAD_LAYER, HEAD_INDEX = "L03.H04", 3, 4
RUNNING_STAGES = ("R0", "R1", "R2", "R3", "c")
COMPONENT_STAGES = ("T", "M4", "M5")
VECTOR_STAGES = ("R0", "R1", "T", "R2", "M4", "M5", "R3")
STAGE_MEANING = {"R1": "transformation at the cue position (layers 1–2)", "R2": "transport into the prediction position (layer 3)",
                 "R3": "late components (layers 4–5)", "c": "readout"}
ENCODING_CLASSES = ("LINEAR_CANCELLATION", "NONLINEAR_GATING", "AXIS_NOT_SUFFICIENT", "ADDITIVE_ORDINARY")
SUMMARY_LABELS = ("AXIS_ARTIFACT", "LOCALIZED", "CONTEXT_LOCALIZED", "MIXED", "PROBE_INVALID", "NO_SUPPRESSED_TOKENS")
E005_AXIS_RELATIVE_PATH = "outputs/experiment-007/parameters/e005-scalar/e_axis_direction.pt"  # Experiment 005's frozen E axis (informational cosine)
DETERMINER_GROUP = ("this", "that", "these", "those", "a", "the", "another", "every")
EXPOSED_16_CATEGORY = {"cardinal:sg": "original-cue", "cardinal:pl": "original-cue", "quantifier:sg": "original-cue", "quantifier:pl": "original-cue"}


class PhaseError(pm.PhaseError):
    pass


def _sites(model: Any, frame: pm.Frame) -> dict[str, pm.Site]:
    return {"R0": ("RESID_PRE.L1", frame.p_c), "R1": (f"RESID_PRE.L{HEAD_LAYER}", frame.p_c), "T": (HEAD_KEY, frame.p_t),
            "R2": (f"RESID_PRE.L{HEAD_LAYER + 1}", frame.p_t), "M4": ("L04.MLP", frame.p_t), "M5": ("L05.MLP", frame.p_t),
            "R3": (f"RESID_POST.L{int(model.cfg.n_layers) - 1}", frame.p_t), "A": (f"ATTN_PATTERN.L{HEAD_LAYER}", frame.p_t)}


# ---------------------------------------------------------------------------
# The exposed pool of Experiment 008.


@dataclass(frozen=True)
class Pool008:
    frames: tuple[pm.Frame, ...]  # 18
    frame_origin: Mapping[str, str]  # frame_id -> manifest | extension | confirmation
    tokens: tuple[tuple[str, int], ...]  # 40 (name, id)
    token_category: Mapping[str, str]
    token_source: Mapping[str, str]  # name -> exposed-16 | confirmation-24
    nouns: tuple[pm.Noun, ...]  # 80
    noun_source: Mapping[str, str]  # lexical_key -> exposed-60 | confirmation-20
    reference_ids: Mapping[str, int]
    plural_cue: Mapping[str, str]  # template -> token name of pl_T

    def frames_of(self, template: str) -> tuple[pm.Frame, ...]:
        return tuple(frame for frame in self.frames if frame.template_id == template)

    def reference_prompt(self, frame: pm.Frame) -> pm.Prompt:
        return pm.Prompt(frame, self.reference_ids[frame.template_id], "ref")

    def token_prompt(self, frame: pm.Frame, name: str) -> pm.Prompt:
        return pm.Prompt(frame, self.token_id(name), name)

    def token_id(self, name: str) -> int:
        return dict(self.tokens)[name]

    @property
    def single_nouns(self) -> tuple[pm.Noun, ...]:
        return tuple(noun for noun in self.nouns if noun.single_token)

    def single_nouns_from(self, source: str) -> tuple[pm.Noun, ...]:
        return tuple(noun for noun in self.single_nouns if self.noun_source[noun.lexical_key] == source)


def build_pool(manifest: ScreeningManifest, extension: pm.Extension, confirmation: cd.Confirmation) -> Pool008:
    exposed = cd.exposed_pool(manifest, extension)
    frames = exposed.frames + confirmation.frames
    origin = {frame.frame_id: frame.origin for frame in exposed.frames} | {frame.frame_id: "confirmation" for frame in confirmation.frames}
    tokens = list(exposed.tokens)
    category = {name: EXPOSED_16_CATEGORY.get(name, "extension") for name, _ in exposed.tokens}
    source = {name: "exposed-16" for name, _ in exposed.tokens}
    seen = {token_id for _, token_id in tokens}
    for entry in confirmation.tokens:
        if entry["token_id"] in seen:
            raise ValueError(f"confirmation token {entry['word']} duplicates an exposed token")
        seen.add(entry["token_id"])
        tokens.append((entry["word"], entry["token_id"]))
        category[entry["word"]] = entry["category"]
        source[entry["word"]] = "confirmation-24"
    if len(tokens) != 40 or len(frames) != 18:
        raise ValueError(f"expected 40 tokens and 18 frames, found {len(tokens)} and {len(frames)}")
    nouns = exposed.nouns + confirmation.nouns
    if len({noun.lexical_key for noun in nouns}) != 80:
        raise ValueError("expected eighty distinct nouns")
    noun_source = {noun.lexical_key: "exposed-60" for noun in exposed.nouns} | {noun.lexical_key: "confirmation-20" for noun in confirmation.nouns}
    by_id = {token_id: name for name, token_id in tokens}
    plural_cue = {}
    for template in pm.TEMPLATE_ORDER:
        frame = next(frame for frame in frames if frame.template_id == template)
        plural_cue[template] = by_id[frame.cue_ids["pl"]]
        if by_id[frame.cue_ids["sg"]] != by_id[exposed.reference_ids[template]]:
            raise ValueError(f"{template}: the reference cue is not the template's singular cue")
    return Pool008(frames, origin, tuple(tokens), category, source, nouns, noun_source, dict(exposed.reference_ids), plural_cue)


# ---------------------------------------------------------------------------
# Inherited Experiment 007 confirmation responses (derived extract) and replication.


def inherited_007_payload(responses: Mapping[str, Mapping[str, Any]], *, source: Mapping[str, Any], manifest_sha256: str, extension_sha256: str, confirmation_sha256: str,
                          lock_sha256: str, model: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": INHERITED_007_SCHEMA_VERSION,
        "description": ("Derived extract of Experiment 007's recorded confirmation-token E-patch responses (mean contrast shift per (token, confirmation frame) "
                        "over the twenty fresh nouns). Experiment 008 recomputes these and requires agreement within 1e-6."),
        "source": dict(source), "manifest_sha256": manifest_sha256, "extension_sha256": extension_sha256, "confirmation_sha256": confirmation_sha256,
        "lock_sha256": lock_sha256, "model": dict(model),
        "responses": {key: {"template_id": entry["template_id"], "mean_shift": float(entry["mean_shift"])} for key, entry in sorted(responses.items())},
    }
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_007(path: Path, *, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited 007 extract: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "description", "source", "manifest_sha256", "extension_sha256", "confirmation_sha256", "lock_sha256", "model", "responses", "content_sha256"}, "inherited 007 extract")
    if payload["schema_version"] != INHERITED_007_SCHEMA_VERSION:
        raise ValueError("inherited 007 extract schema version is not frozen")
    if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("inherited 007 extract content_sha256 does not match its canonical payload")
    if (payload["manifest_sha256"], payload["extension_sha256"], payload["confirmation_sha256"]) != (manifest_sha256, extension_sha256, confirmation_sha256):
        raise ValueError("inherited 007 extract was recorded against different frozen inputs")
    if dict(payload["model"]) != {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}:
        raise ValueError("inherited 007 extract was recorded for a different pinned model")
    if len(payload["responses"]) != 24 * 6:
        raise ValueError(f"inherited 007 extract must hold 144 responses, found {len(payload['responses'])}")
    return payload


def check_replication(measured: Mapping[str, float], recorded: Mapping[str, Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    """Recomputed mean shifts must reproduce the recorded ones on exactly the recorded keys."""
    missing = set(recorded) - set(measured)
    if missing:
        raise pm.IncidentError(f"{label}: {len(missing)} recorded (token, frame) pairs were not recomputed")
    deviations = {key: abs(measured[key] - recorded[key]["mean_shift"]) for key in recorded}
    worst = max(deviations, key=deviations.get)
    if deviations[worst] > REPLICATION_TOLERANCE:
        raise pm.IncidentError(f"{label}: {worst} deviates from the recorded mean shift by {deviations[worst]:.3e} (> {REPLICATION_TOLERANCE:.0e})")
    return {"passed": True, "n": len(deviations), "max_abs_deviation": deviations[worst], "worst_key": worst, "tolerance": REPLICATION_TOLERANCE}


# ---------------------------------------------------------------------------
# Results state (two phases) — the same conventions as Experiments 006/007.

_STATE_KEYS = {"schema_version", "run_id", "created_at", "manifest_sha256", "extension_sha256", "confirmation_sha256", "protocol_code_commit", "git_dirty",
               "model", "versions", "phases", "executed_prompt_keys", "executed_noun_keys", "exploration", "invalidated_runs", "state_sha256"}


def new_results_state(*, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str, protocol_code_commit: str, git_dirty: bool, versions: Mapping[str, Any]) -> dict[str, Any]:
    if not pm._COMMIT_SHA.fullmatch(protocol_code_commit or ""):
        raise PhaseError("scientific execution requires a 40-character committed protocol/code SHA")
    if git_dirty:
        raise PhaseError("scientific execution requires a clean Git tree")
    return {"schema_version": RESULTS_SCHEMA_VERSION, "run_id": pm.sha256_text(manifest_sha256 + extension_sha256 + confirmation_sha256 + protocol_code_commit + pm.utc_now())[:16],
            "created_at": pm.utc_now(), "manifest_sha256": manifest_sha256, "extension_sha256": extension_sha256, "confirmation_sha256": confirmation_sha256,
            "protocol_code_commit": protocol_code_commit, "git_dirty": False, "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "versions": dict(versions),
            "phases": {phase: {"status": "not_started"} for phase in PHASES}, "executed_prompt_keys": [], "executed_noun_keys": [], "exploration": {}, "invalidated_runs": []}


def write_results_state(path: Path, state: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    validate_json_safe(payload, path="results_state")
    payload["state_sha256"] = pm.state_digest(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(pm.canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload["state_sha256"]


def load_results_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhaseError(f"could not read results state: {path}") from error
    if not isinstance(state, Mapping) or state.get("schema_version") != RESULTS_SCHEMA_VERSION or set(state) != _STATE_KEYS:
        raise PhaseError("results state schema is not recognized")
    if state["state_sha256"] != pm.state_digest(state):
        raise PhaseError("results state digest mismatch: the artifact was modified outside the runner")
    if set(state["phases"]) != set(PHASES) or any(entry.get("status") not in pm.PHASE_STATUSES for entry in state["phases"].values()):
        raise PhaseError("results state phase records are invalid")
    return dict(state)


def assert_phase_allowed(phase: str, state: Mapping[str, Any]) -> None:
    status = {name: entry["status"] for name, entry in state["phases"].items()}
    if phase == "explore":
        if status["explore"] == "running" and not state["exploration"].get("summary"):
            return  # crash or incident recovery before any summary was concluded
        if status["explore"] != "not_started":
            raise PhaseError("explore already ran; exploration is never re-run in one protocol version")
    elif phase == "report":
        if status["explore"] != "complete" and not state["exploration"].get("incidents"):
            raise PhaseError("report requires the completed explore phase (or a recorded incident)")
    else:
        raise PhaseError(f"unknown phase {phase}")


# ---------------------------------------------------------------------------
# Stage axes.


def stage_axes(cache: pm.PromptCache, weights: pm.Weights, pool: Pool008) -> dict[str, pm.SiteAxis]:
    """Number axis per captured site from the eighteen frames' clean original cue pairs (plural positive); R0 is the weight-only E axis."""
    model = cache.model
    sg_vectors: dict[str, list[torch.Tensor]] = {stage: [] for stage in VECTOR_STAGES}
    pl_vectors: dict[str, list[torch.Tensor]] = {stage: [] for stage in VECTOR_STAGES}
    for frame in pool.frames:
        sg, pl = pm.frame_prompts(frame)
        run_sg, run_pl = cache.run(sg), cache.run(pl)
        sites = _sites(model, frame)
        sg_vectors["R0"].append(pm.lexicon_vector(weights, frame.cue_ids["sg"]))
        pl_vectors["R0"].append(pm.lexicon_vector(weights, frame.cue_ids["pl"]))
        for stage in VECTOR_STAGES[1:]:
            sg_vectors[stage].append(run_sg.vector(sites[stage]))
            pl_vectors[stage].append(run_pl.vector(sites[stage]))
    return {stage: pm.site_axis(stage, sg_vectors[stage], pl_vectors[stage]) for stage in VECTOR_STAGES}


# ---------------------------------------------------------------------------
# Measurements per (token, frame): M1 trace, M2 component patches, M3 context patches, M4 direct-effect deltas.


@dataclass
class TraceRecord:
    token: str
    token_id: int
    frame_id: str
    template: str
    delta: dict[str, torch.Tensor]  # stage -> Δ vector (float64) for VECTOR_STAGES
    attention_ref: float
    attention_patched: float
    dc: float  # full E-patch mean shift
    dc_par: float  # axis-component patch
    dc_perp: float  # orthogonal-component patch
    dc_beh: float  # clean behavioral shift c[prompt_w] − c_ref
    dc_context_only: float  # c[prompt_w with E(ref)] − c_ref
    dc_in: float  # c[prompt_w with E(pl)] − c[prompt_w with E(ref)]
    dc_out: float  # c[prompt_pl with E(w)] − c[prompt_pl with E(ref)]
    dde: dict[str, float]  # ΔDE_k, mean over nouns
    identity_error: float
    shifts: dict[str, float]  # per-noun E-patch shift of the full patch (for the replication checks on noun subsets)
    additivity_check: float = 0.0  # Σ_k ΔDE_k − dc (should be ~0)


def _mean_contrast(run_or_c: Any, nouns: Sequence[pm.Noun]) -> float:
    c = run_or_c if isinstance(run_or_c, Mapping) else pm.contrasts(run_or_c.logits, nouns)
    return pm._mean([c[noun.lexical_key] for noun in nouns if noun.single_token])


def _slice(vector: torch.Tensor) -> torch.Tensor:
    return vector.reshape(1, 1, -1).to(torch.float32)


def _direct_effect_means(weights: pm.Weights, run: pm.PromptRun, nouns: Sequence[pm.Noun], *, p_t: int, n_layers: int, universe: Sequence[str], where: str) -> dict[str, float]:
    effects = pm.direct_effects(weights, run, nouns, p_t=p_t, n_layers=n_layers, universe=universe)
    pm.check_direct_effects(effects, where=where)
    per_noun = effects["per_noun"]
    terms = [name for name in next(iter(per_noun.values())) if name not in ("c_reconstructed", "c_model")]
    return {name: pm._mean([per_noun[noun][name] for noun in per_noun]) for name in terms}


def measure_pool(model: Any, weights: pm.Weights, cache: pm.PromptCache, pool: Pool008, e_axis: pm.SiteAxis, *, log: Any = None) -> dict[tuple[str, str], TraceRecord]:
    """Every measurement of the design for all 40 tokens × 18 frames; the reference token's own patches are zero by definition."""
    say = log or (lambda message: None)
    n_layers = int(model.cfg.n_layers)
    universe = pm.universe_keys(model)
    nouns = pool.single_nouns
    direction = e_axis.direction.double()
    records: dict[tuple[str, str], TraceRecord] = {}
    for frame in pool.frames:
        template = frame.template_id
        sites = _sites(model, frame)
        capture = [sites[stage] for stage in VECTOR_STAGES] + [sites["A"], ("EMBED", frame.p_t)] + [(key, frame.p_t) for key in universe]
        reference = pool.reference_prompt(frame)
        ref_run = cache.run(reference)
        c_ref_by_noun = cache.c(reference)
        c_ref = _mean_contrast(c_ref_by_noun, nouns)
        ref_vectors = {stage: ref_run.vector(sites[stage]).double() for stage in VECTOR_STAGES}
        ref_attention = float(ref_run.vector(sites["A"])[HEAD_INDEX, frame.p_c])
        ref_dde = _direct_effect_means(weights, ref_run, nouns, p_t=frame.p_t, n_layers=n_layers, universe=universe, where=f"{frame.frame_id}/ref")
        e_ref = pm.lexicon_vector(weights, reference.cue_token_id)
        e_site = ("L00.MLP", frame.p_c)
        pl_name = pool.plural_cue[template]
        pl_prompt = pool.token_prompt(frame, pl_name)
        e_pl = pm.lexicon_vector(weights, pl_prompt.cue_token_id)
        # The plural-cue prompt with the reference encoding is shared by every token of the frame.
        pl_with_ref = _mean_contrast(pm.run_patched(model, pl_prompt, {e_site: _slice(e_ref)}, {e_site: ReplacementSource.RESAMPLE}), nouns)
        c_pl_clean = _mean_contrast(cache.c(pl_prompt), nouns)
        for name, token_id in pool.tokens:
            is_reference = token_id == reference.cue_token_id
            is_plural = token_id == pl_prompt.cue_token_id
            e_w = pm.lexicon_vector(weights, token_id)
            delta_e = (e_w - e_ref).double()
            parallel = (delta_e @ direction) * direction
            perpendicular = delta_e - parallel
            w_prompt = reference if is_reference else pool.token_prompt(frame, name)
            if is_reference:
                # Replacing E(ref) by E(ref) is the identity: every own-patch quantity is zero by definition (no forward).
                delta = {stage: torch.zeros_like(ref_vectors[stage]) for stage in VECTOR_STAGES}
                attention_patched, dc, dc_par, dc_perp, identity_error = ref_attention, 0.0, 0.0, 0.0, 0.0
                dde = {key: 0.0 for key in ref_dde}
                shifts = {noun.lexical_key: 0.0 for noun in nouns}
                w_with_ref = c_ref
            else:
                patched = pm.run_patched(model, reference, {e_site: _slice(e_w)}, {e_site: ReplacementSource.RESAMPLE}, capture_sites=capture)
                delta = {stage: patched.vector(sites[stage]).double() - ref_vectors[stage] for stage in VECTOR_STAGES}
                identity_error = float((delta["R0"] - delta_e).abs().max())
                if identity_error > IDENTITY_TOLERANCE:
                    raise pm.IncidentError(f"{frame.frame_id}/{name}: the residual after layer 0 does not change by ΔE_T(w) (max {identity_error:.2e})")
                attention_patched = float(patched.vector(sites["A"])[HEAD_INDEX, frame.p_c])
                c_patched = pm.contrasts(patched.logits, nouns)
                shifts = {noun.lexical_key: c_patched[noun.lexical_key] - c_ref_by_noun[noun.lexical_key] for noun in nouns}
                dc = pm._mean(list(shifts.values()))
                patched_dde = _direct_effect_means(weights, patched, nouns, p_t=frame.p_t, n_layers=n_layers, universe=universe, where=f"{frame.frame_id}/{name}")
                dde = {key: patched_dde[key] - ref_dde[key] for key in ref_dde}
                dc_par = _mean_contrast(pm.run_patched(model, reference, {e_site: _slice(e_ref.double() + parallel)}, {e_site: ReplacementSource.RESAMPLE}), nouns) - c_ref
                dc_perp = _mean_contrast(pm.run_patched(model, reference, {e_site: _slice(e_ref.double() + perpendicular)}, {e_site: ReplacementSource.RESAMPLE}), nouns) - c_ref
                w_with_ref = _mean_contrast(pm.run_patched(model, w_prompt, {e_site: _slice(e_ref)}, {e_site: ReplacementSource.RESAMPLE}), nouns)
            # M3 endpoints (identity patches use the clean contrast by the same convention).
            w_with_pl = c_pl_clean if is_plural else _mean_contrast(pm.run_patched(model, w_prompt, {e_site: _slice(e_pl)}, {e_site: ReplacementSource.RESAMPLE}), nouns)
            if is_plural:
                pl_with_w = c_pl_clean
            elif is_reference:
                pl_with_w = pl_with_ref
            else:
                pl_with_w = _mean_contrast(pm.run_patched(model, pl_prompt, {e_site: _slice(e_w)}, {e_site: ReplacementSource.RESAMPLE}), nouns)
            dc_beh = _mean_contrast(cache.c(w_prompt), nouns) - c_ref
            records[(name, frame.frame_id)] = TraceRecord(name, token_id, frame.frame_id, template, delta, ref_attention, attention_patched, dc, dc_par, dc_perp, dc_beh,
                                                        w_with_ref - c_ref, w_with_pl - w_with_ref, pl_with_w - pl_with_ref, dde, identity_error, shifts, sum(dde.values()) - dc)
        say(f"  frame {frame.frame_id}: {len(pool.tokens)} tokens traced")
    return records


# ---------------------------------------------------------------------------
# Fractions, oriented traces, and the per-(token, frame) analysis.


def _sign(value: float) -> float:
    return -1.0 if value < 0 else 1.0


def _ratio(numerator: float, denominator: float, floor: float) -> float | None:
    return None if abs(denominator) < floor else numerator / denominator


def _oriented(value: float | None, orientation: float) -> float | None:
    return None if value is None else orientation * value


def frame_analysis(record: TraceRecord, plural: TraceRecord, axes: Mapping[str, pm.SiteAxis], u1: torch.Tensor, e_axis: pm.SiteAxis, delta_e: torch.Tensor, delta_e_pl: torch.Tensor) -> dict[str, Any]:
    """Signal fractions, oriented trace, component and context fractions, cancellation index, and the collapse stage for one (token, frame)."""
    out: dict[str, Any] = {}
    fractions: dict[str, float | None] = {}
    for stage in VECTOR_STAGES:
        axis = axes[stage]
        numerator = float(record.delta[stage] @ axis.direction.double())
        denominator = float(plural.delta[stage] @ axis.direction.double())
        fractions[stage] = _ratio(numerator, denominator, STAGE_UNINFORMATIVE_FLOOR * axis.sigma)
    dc_pl = plural.dc
    fractions["c"] = _ratio(record.dc, dc_pl, CONTRAST_UNINFORMATIVE_NATS)
    out["fractions"] = fractions
    out["s_u1"] = _ratio(float(delta_e @ u1.double()), float(delta_e_pl @ u1.double()), 1e-9)
    s_r0 = fractions["R0"]
    orientation = _sign(s_r0) if s_r0 is not None else 1.0
    oriented = {stage: (None if value is None else orientation * value) for stage, value in fractions.items()}
    out["oriented"] = oriented
    out["carries_signal"] = s_r0 is not None and abs(s_r0) >= S_MIN
    # Component fractions are oriented by the encoding signal's sign (like q), so "cancellation" means opposing the token's own axis signal.
    out["r_par"] = _oriented(_ratio(record.dc_par, dc_pl, CONTRAST_UNINFORMATIVE_NATS), orientation)
    out["r_perp"] = _oriented(_ratio(record.dc_perp, dc_pl, CONTRAST_UNINFORMATIVE_NATS), orientation)
    out["r_full"] = _oriented(fractions["c"], orientation)
    out["g"] = None if None in (out["r_par"], out["r_perp"], out["r_full"]) else out["r_full"] - out["r_par"] - out["r_perp"]
    out["x_in"] = _ratio(record.dc_in, dc_pl, CONTRAST_UNINFORMATIVE_NATS)
    out["x_out"] = _ratio(record.dc_out, dc_pl, CONTRAST_UNINFORMATIVE_NATS)
    out["context_only"] = _ratio(record.dc_context_only, dc_pl, CONTRAST_UNINFORMATIVE_NATS)
    out["behavior"] = _ratio(record.dc_beh, dc_pl, CONTRAST_UNINFORMATIVE_NATS)
    out["attention_fraction"] = _ratio(record.attention_patched, record.attention_ref, 1e-9)
    out["attention_delta"] = record.attention_patched - record.attention_ref
    total = sum(record.dde.values())
    denominator = max(abs(total), CANCELLATION_DENOMINATOR_FLOOR * abs(dc_pl))
    out["cancellation_index"] = sum(abs(value) for value in record.dde.values()) / denominator if denominator > 0 else None
    ordered = sorted(record.dde.items(), key=lambda kv: kv[1])
    out["opposing_terms"] = {"most_negative": list(ordered[0]) if ordered else None, "most_positive": list(ordered[-1]) if ordered else None}
    out["dde_circuit"] = {key: record.dde.get(key) for key in (HEAD_KEY, "L04.MLP", "L05.MLP")}
    out["collapse"] = collapse_stage(oriented)
    out["dc"] = record.dc
    out["dc_beh"] = record.dc_beh
    return out


def collapse_stage(oriented: Mapping[str, float | None]) -> dict[str, Any]:
    """First running stage whose oriented signal is at most KAPPA times the previous informative stage's (a sign flip counts), given that previous stage ≥ S_MIN.

    The rule is applied literally from R0 onward; a frame in which no informative stage before the end ever reaches S_MIN
    has no signal to trace and is labelled NO_SIGNAL (excluded from the mode over frames).
    """
    informative = [(stage, oriented.get(stage)) for stage in RUNNING_STAGES if oriented.get(stage) is not None]
    if not informative:
        return {"stage": "NO_SIGNAL", "previous": None, "q_previous": None, "q_at": None, "transport": None}
    reached = False
    previous_stage, previous = informative[0]
    for stage, value in informative[1:]:
        if previous >= S_MIN:
            reached = True
            if value <= KAPPA * previous:
                transport = None
                if stage == "R2":
                    q_t, q_r1 = oriented.get("T"), oriented.get("R1")
                    if q_t is not None and q_r1 is not None:
                        transport = "HEAD_DROPPED" if q_t <= KAPPA * q_r1 else "OTHER_LAYER3_CANCELLED"
                return {"stage": stage, "previous": previous_stage, "q_previous": previous, "q_at": value, "transport": transport}
        previous_stage, previous = stage, value
    if not reached:
        return {"stage": "NO_SIGNAL", "previous": None, "q_previous": informative[0][1], "q_at": None, "transport": None}
    return {"stage": "NO_COLLAPSE", "previous": previous_stage, "q_previous": previous, "q_at": None, "transport": None}


# ---------------------------------------------------------------------------
# Anomaly score, token-level aggregation, classification, summary.


def anomaly_score(predicted: float, measured: float) -> float:
    """a = sign(p) × (m − p), sign(0) = +1: negative = suppression, positive = amplification, for either orientation."""
    return _sign(predicted) * (measured - predicted)


def stratum(score: float) -> str:
    if score <= -STRATUM_THRESHOLD:
        return "suppressed"
    if score >= STRATUM_THRESHOLD:
        return "amplified"
    return "ordinary"


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return pm._mean(kept) if kept else None


def _mode_stage(stages: Sequence[str]) -> tuple[str | None, float | None]:
    kept = [stage for stage in stages if stage != "NO_SIGNAL"]
    if not kept:
        return None, None
    counts = Counter(kept)
    order = list(RUNNING_STAGES) + ["NO_COLLAPSE"]
    best = min(counts, key=lambda stage: (-counts[stage], order.index(stage) if stage in order else len(order)))
    return best, counts[best] / len(kept)


def encoding_class(r_par: float | None, r_perp: float | None, g: float | None, s_r0: float | None) -> str | None:
    if None in (r_par, r_perp, g, s_r0):
        return None
    if r_perp <= -0.5 * r_par and abs(g) <= G_MAX:
        return "LINEAR_CANCELLATION"
    if abs(g) > G_MAX:
        return "NONLINEAR_GATING"
    if r_par < S_MIN and abs(s_r0) >= S_MIN:
        return "AXIS_NOT_SUFFICIENT"
    return "ADDITIVE_ORDINARY"


def token_row(name: str, pool: Pool008, per_frame: Mapping[str, Mapping[str, Any]], scores: Mapping[str, Mapping[str, float]], plural_cancellation: Mapping[str, float | None],
              probe_valid: bool) -> dict[str, Any]:
    frames = list(per_frame)
    means = {key: _mean_or_none([per_frame[frame][key] for frame in frames]) for key in ("s_u1", "r_par", "r_perp", "r_full", "g", "x_in", "x_out", "context_only", "behavior", "attention_fraction", "cancellation_index", "dc", "dc_beh")}
    fractions = {stage: _mean_or_none([per_frame[frame]["fractions"][stage] for frame in frames]) for stage in VECTOR_STAGES + ("c",)}
    oriented = {stage: _mean_or_none([per_frame[frame]["oriented"][stage] for frame in frames]) for stage in VECTOR_STAGES + ("c",)}
    per_template = {}
    for template in pm.TEMPLATE_ORDER:
        template_frames = [frame.frame_id for frame in pool.frames_of(template) if frame.frame_id in per_frame]
        entry = {key: _mean_or_none([per_frame[frame][key] for frame in template_frames]) for key in ("r_par", "r_perp", "g", "x_in", "x_out", "context_only", "cancellation_index", "dc")}
        entry["s_R0"] = _mean_or_none([per_frame[frame]["fractions"]["R0"] for frame in template_frames])
        entry["s_c"] = _mean_or_none([per_frame[frame]["fractions"]["c"] for frame in template_frames])
        entry["oriented"] = {stage: _mean_or_none([per_frame[frame]["oriented"][stage] for frame in template_frames]) for stage in VECTOR_STAGES + ("c",)}
        entry["collapse_stage"], entry["collapse_agreement"] = _mode_stage([per_frame[frame]["collapse"]["stage"] for frame in template_frames])
        per_template[template] = entry
    stage, agreement = _mode_stage([per_frame[frame]["collapse"]["stage"] for frame in frames])
    transports = Counter(per_frame[frame]["collapse"]["transport"] for frame in frames if per_frame[frame]["collapse"]["stage"] == "R2" and per_frame[frame]["collapse"]["transport"])
    s_r0 = fractions["R0"]
    carries = s_r0 is not None and abs(s_r0) >= S_MIN
    score = pm._mean([scores[frame]["a"] for frame in frames])
    row: dict[str, Any] = {
        "token": name, "token_id": pool.token_id(name), "category": pool.token_category[name], "source": pool.token_source[name], "in_sample_for_007": pool.token_source[name] == "exposed-16",
        "anomaly_score": score, "mean_abs_prediction": pm._mean([abs(scores[frame]["p"]) for frame in frames]), "stratum": stratum(score),
        "measured_shift": means["dc"], "behavioral_shift": means["dc_beh"], "fractions": fractions, "oriented": oriented, "s_u1": means["s_u1"], "carries_signal": carries,
        "r_par": means["r_par"], "r_perp": means["r_perp"], "r_full": means["r_full"], "g": means["g"],
        "encoding_class": (encoding_class(means["r_par"], means["r_perp"], means["g"], s_r0) if carries else None) if probe_valid else None,
        "collapse_stage": stage, "collapse_agreement": agreement, "collapse_meaning": STAGE_MEANING.get(stage) if stage else None,
        "transport_detail": transports.most_common(1)[0][0] if transports else None,
        "attention_fraction": means["attention_fraction"], "x_in": means["x_in"], "x_out": means["x_out"], "context_only": means["context_only"], "behavior_fraction": means["behavior"],
        "context_class": None if means["x_in"] is None else ("CONTEXT_GATED" if means["x_in"] < X_MIN else "CONTEXT_NEUTRAL"),
        "cancellation_index": means["cancellation_index"], "per_template": per_template,
    }
    plural_index = _mean_or_none([plural_cancellation.get(frame) for frame in frames])
    row["plural_cancellation_index"] = plural_index
    row["late_cancellation"] = (means["cancellation_index"] is not None and plural_index is not None and means["cancellation_index"] >= C_MIN * plural_index and means["cancellation_index"] >= C_MIN)
    row["axis_artifact"] = row["stratum"] == "suppressed" and not carries
    return row


def probe_validity(per_frame_by_token: Mapping[str, Mapping[str, Mapping[str, Any]]], pool: Pool008) -> dict[str, Any]:
    """The axis-only patch of the template's plural cue must reproduce its own effect in at least 75% of frames."""
    checks = []
    for frame in pool.frames:
        value = per_frame_by_token[pool.plural_cue[frame.template_id]][frame.frame_id]["r_par"]
        checks.append({"frame_id": frame.frame_id, "r_par": value, "ok": value is not None and value >= PROBE_FLOOR})
    passed = sum(1 for check in checks if check["ok"])
    return {"valid": passed >= CONSENSUS * len(checks), "frames_ok": passed, "frames": len(checks), "detail": checks}


def summarize(rows: Sequence[Mapping[str, Any]], probe_valid: bool) -> dict[str, Any]:
    """The experiment-level summary over the suppressed stratum (mechanical; design revision 4 precedence)."""
    suppressed = [row for row in rows if row["stratum"] == "suppressed"]
    out: dict[str, Any] = {"suppressed_tokens": [row["token"] for row in suppressed], "n_suppressed": len(suppressed), "probe_valid": probe_valid}
    if not suppressed:
        out["label"] = "NO_SUPPRESSED_TOKENS"
        return out
    carrying = [row for row in suppressed if row["carries_signal"]]
    collapse_stages = Counter(row["collapse_stage"] for row in carrying if row["collapse_stage"] in RUNNING_STAGES[1:])  # NO_COLLAPSE is not a collapse stage
    modal_stage, stage_count = (collapse_stages.most_common(1)[0] if collapse_stages else (None, 0))
    stage_consensus = bool(carrying) and stage_count >= CONSENSUS * len(carrying)
    pairs = Counter((row["collapse_stage"], row["encoding_class"]) for row in carrying if row["collapse_stage"] in RUNNING_STAGES[1:] and row["encoding_class"])
    (modal_pair, pair_count) = (pairs.most_common(1)[0] if pairs else ((None, None), 0))
    joint_consensus = bool(carrying) and pair_count >= CONSENSUS * len(carrying)
    classes = Counter(row["encoding_class"] for row in carrying if row["encoding_class"])
    gated = sum(1 for row in suppressed if row["context_class"] == "CONTEXT_GATED")
    no_collapse = sum(1 for row in carrying if row["collapse_stage"] == "NO_COLLAPSE")
    out.update({"n_carrying": len(carrying), "modal_stage": modal_stage, "stage_consensus": stage_consensus, "modal_pair": list(modal_pair), "joint_consensus": joint_consensus,
                "class_counts": dict(classes), "context_gated": gated, "no_collapse": no_collapse})
    trace_summary = f"LOCALIZED_{modal_stage}" if stage_consensus else "MIXED"
    if not probe_valid:
        out["label"] = "PROBE_INVALID"
        out["trace_summary"] = trace_summary
    elif all(row["axis_artifact"] for row in suppressed):
        out["label"] = "AXIS_ARTIFACT"
    elif joint_consensus:
        out["label"] = f"LOCALIZED_{modal_pair[0]}_{modal_pair[1]}"
    elif gated >= CONSENSUS * len(suppressed) and not stage_consensus:
        out["label"] = "CONTEXT_LOCALIZED"
    else:
        out["label"] = "MIXED"
        out["trace_summary"] = trace_summary
    return out


# ---------------------------------------------------------------------------
# The 007 program for the anomaly score.


def load_program_007(root: Path, *, parameters_dir: Path | None = None) -> tuple[Any, dict[str, Any]]:
    lock_path = Path(root) / EXPERIMENT_007_LOCK_PATH
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    directory = Path(parameters_dir) if parameters_dir is not None else Path(root) / PROGRAM_007_RELATIVE_DIR
    index_path = directory / "parameters.json"
    if not index_path.exists():
        raise PhaseError(f"the Experiment 007 selected program is not available at {directory}")
    digest = pm.sha256_text(index_path.read_text(encoding="utf-8"))
    if digest != lock["parameters"]["selected"]:
        raise PhaseError("the Experiment 007 selected program on disk does not match the committed 007 lock")
    source_path = Path(root) / PROGRAM_007_SOURCE
    if pm.sha256_text(source_path.read_text(encoding="utf-8")) != lock["program_source_sha256"]:
        raise PhaseError("the Experiment 007 program source differs from the committed 007 lock")
    program = ss.load_linear_program(directory, source_path)
    if program.kind != ss.KIND_SUPERVISED or program.rank != 1 or program.basis is None:
        raise PhaseError("the Experiment 007 selected program is not the rank-1 supervised subspace program")
    return program, {"parameters_sha256": digest, "lock_sha256": lock["content_sha256"], "rank": program.rank, "kind": program.kind}


def program_prediction(program: Any, frame: pm.Frame, token_id: int, nouns: Sequence[pm.Noun]) -> float:
    return pm._mean([program.predict_epatch_shift(frame.template_id, frame.frame_id, token_id, noun.sg_ids[0], noun.pl_ids[0]) for noun in nouns if noun.single_token])


# ---------------------------------------------------------------------------
# Exploration orchestration and the report.


def cosine_with_005_axis(e_axis: pm.SiteAxis, path: Path | None) -> float | None:
    """Informational: the cosine of the eighteen-frame E axis with Experiment 005's frozen E axis, when its tensor is available locally."""
    if path is None or not Path(path).exists():
        return None
    tensor = torch.load(Path(path), map_location="cpu", weights_only=True)
    return pm.cosine(e_axis.direction, tensor.reshape(-1))


def run_exploration(model: Any, pool: Pool008, *, state: dict[str, Any], results_path: Path | None, program: Any, program_record: Mapping[str, Any],
                    inherited_006: Mapping[str, Any], inherited_007: Mapping[str, Any], e005_axis_path: Path | None = None, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    cache = pm.PromptCache(model, tuple(pool.nouns))
    say("stage axes from the eighteen frames' clean cue pairs")
    axes = stage_axes(cache, weights, pool)
    e_axis = axes["R0"]
    if getattr(program, "basis", None) is None or program.basis.shape[1] < 1:
        raise PhaseError("the anomaly-score program has no basis direction")
    u1 = program.basis[:, 0].double()
    exploration = state["exploration"]
    exploration["axes"] = {stage: {"sigma": axis.sigma, "cos_with_E_axis": pm.cosine(axis.direction, e_axis.direction)} for stage, axis in axes.items()}
    exploration["axes"]["R0"]["cos_with_u1"] = pm.cosine(e_axis.direction, u1)
    exploration["axes"]["R0"]["cos_with_experiment_005_axis"] = cosine_with_005_axis(e_axis, e005_axis_path)
    exploration["program_007"] = dict(program_record)
    say("measurements: 40 tokens × 18 frames (trace, component, context, direct effects)")
    records = measure_pool(model, weights, cache, pool, e_axis, log=say)
    # Replication of Experiments 006 and 007 on the noun sets those experiments averaged over.
    exposed_keys = [noun.lexical_key for noun in pool.single_nouns_from("exposed-60")]
    confirmation_keys = [noun.lexical_key for noun in pool.single_nouns_from("confirmation-20")]
    measured_006 = {f"{name}|{frame_id}": pm._mean([record.shifts[key] for key in exposed_keys]) for (name, frame_id), record in records.items()}
    measured_007 = {f"{name}|{frame_id}": pm._mean([record.shifts[key] for key in confirmation_keys]) for (name, frame_id), record in records.items()}
    exploration["replication"] = {"experiment_006": check_replication(measured_006, inherited_006["responses"], label="Experiment 006 exposed responses"),
                                  "experiment_007": check_replication(measured_007, inherited_007["responses"], label="Experiment 007 confirmation responses")}
    say(f"replication: 006 max deviation {exploration['replication']['experiment_006']['max_abs_deviation']:.2e}; 007 {exploration['replication']['experiment_007']['max_abs_deviation']:.2e}")
    if results_path is not None:
        write_results_state(results_path, state)
    # Per-(token, frame) analysis.
    e_vectors = {name: pm.lexicon_vector(weights, token_id).double() for name, token_id in pool.tokens}
    per_frame_by_token: dict[str, dict[str, dict[str, Any]]] = {name: {} for name, _ in pool.tokens}
    scores: dict[str, dict[str, dict[str, float]]] = {name: {} for name, _ in pool.tokens}
    plural_cancellation: dict[str, float | None] = {}
    nouns = pool.single_nouns
    for frame in pool.frames:
        template = frame.template_id
        plural = records[(pool.plural_cue[template], frame.frame_id)]
        e_ref = e_vectors[next(name for name, token_id in pool.tokens if token_id == pool.reference_ids[template])]
        delta_e_pl = e_vectors[pool.plural_cue[template]] - e_ref
        for name, token_id in pool.tokens:
            record = records[(name, frame.frame_id)]
            analysis = frame_analysis(record, plural, axes, u1, e_axis, e_vectors[name] - e_ref, delta_e_pl)
            per_frame_by_token[name][frame.frame_id] = analysis
            predicted = program_prediction(program, frame, token_id, nouns)
            scores[name][frame.frame_id] = {"p": predicted, "m": record.dc, "a": anomaly_score(predicted, record.dc)}
        plural_cancellation[frame.frame_id] = per_frame_by_token[pool.plural_cue[template]][frame.frame_id]["cancellation_index"]
    probe = probe_validity(per_frame_by_token, pool)
    rows = [token_row(name, pool, per_frame_by_token[name], scores[name], plural_cancellation, probe["valid"]) for name, _ in pool.tokens]
    summary = summarize(rows, probe["valid"])
    exploration["probe_validity"] = probe
    exploration["per_frame"] = {name: {frame_id: _strip(analysis) for frame_id, analysis in frames.items()} for name, frames in per_frame_by_token.items()}
    exploration["anomaly_scores"] = scores
    exploration["tokens"] = {row["token"]: row for row in rows}
    exploration["determiner_group"] = {name: {"anomaly_score": exploration["tokens"][name]["anomaly_score"], "measured_shift": exploration["tokens"][name]["measured_shift"],
                                              "collapse_stage": exploration["tokens"][name]["collapse_stage"], "encoding_class": exploration["tokens"][name]["encoding_class"]}
                                       for name in DETERMINER_GROUP if name in exploration["tokens"]}
    exploration["identity"] = {"max_error": max(record.identity_error for record in records.values()), "max_direct_effect_additivity_gap": max(abs(record.additivity_check) for record in records.values())}
    exploration["summary"] = summary
    say(f"summary: {summary['label']} (suppressed {summary['suppressed_tokens']})")
    if results_path is not None:
        write_results_state(results_path, state)
    return exploration


def _strip(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in analysis.items()}


def _f(value: Any, digits: int = 2) -> str:
    return cd._f(value, digits)


def render_report(state: Mapping[str, Any]) -> str:
    lines = ["# Experiment 008 Report", "", f"- Run ID: `{state['run_id']}`", f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    if exploration.get("incidents"):
        lines += ["## Incidents", ""] + [f"- `{entry['phase']}` at commit `{entry.get('commit', '?')}` ({entry['at']}): {entry['message']}" for entry in exploration["incidents"]] + [""]
    if "replication" in exploration:
        rep = exploration["replication"]
        lines += ["## Replication", "", f"- Experiment 006 exposed responses: {rep['experiment_006']['n']} pairs, max deviation {rep['experiment_006']['max_abs_deviation']:.2e}",
                  f"- Experiment 007 confirmation responses: {rep['experiment_007']['n']} pairs, max deviation {rep['experiment_007']['max_abs_deviation']:.2e}", ""]
    if "axes" in exploration:
        lines += ["## Axes", "", f"- cos(d̂_E, u₁ of the 007 program) = {_f(exploration['axes']['R0']['cos_with_u1'], 3)}",
                  "- " + "; ".join(f"{stage}: σ {_f(axis['sigma'])}, cos with d̂_E {_f(axis['cos_with_E_axis'], 3)}" for stage, axis in exploration["axes"].items()), ""]
    if "summary" in exploration:
        summary = exploration["summary"]
        probe = exploration["probe_validity"]
        lines += [f"## Summary — `{summary['label']}`", "", f"- Suppressed stratum: {summary['suppressed_tokens']}",
                  f"- Probe validity: {'valid' if probe['valid'] else 'INVALID'} ({probe['frames_ok']}/{probe['frames']} frames with r_∥(pl_T) ≥ {PROBE_FLOOR})",
                  f"- Modal collapse stage {summary.get('modal_stage')} (consensus {summary.get('stage_consensus')}); modal encoding class {summary.get('modal_class')} (consensus {summary.get('class_consensus')}); context-gated {summary.get('context_gated')}",
                  f"- Identity max error {exploration['identity']['max_error']:.2e}; direct-effect additivity gap {exploration['identity']['max_direct_effect_additivity_gap']:.2e}", ""]
        lines += ["## Tokens", "", "Component fractions r_∥, r_⊥, g are oriented by sign(ŝ_R0); q_T is the oriented head-output fraction; â is descriptive only.", "",
                  "| token | category | stratum | a | measured | ŝ_R0 | ŝ_u₁ | carries | r_∥ | r_⊥ | g | class | collapse | transport | q_T | â | x_in | x_out | ctx | C | late |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in sorted(exploration["tokens"].values(), key=lambda entry: entry["anomaly_score"]):
            lines.append(f"| {row['token']} | {row['category']} | {row['stratum']} | {_f(row['anomaly_score'])} | {_f(row['measured_shift'])} | {_f(row['fractions']['R0'])} | {_f(row['s_u1'])} | {row['carries_signal']} | "
                         f"{_f(row['r_par'])} | {_f(row['r_perp'])} | {_f(row['g'])} | {row['encoding_class'] or '—'} | {row['collapse_stage'] or '—'} ({_f(row['collapse_agreement'])}) | {row['transport_detail'] or '—'} | "
                         f"{_f(row['oriented']['T'])} | {_f(row['attention_fraction'])} | {_f(row['x_in'])} | {_f(row['x_out'])} | {_f(row['context_only'])} | {_f(row['cancellation_index'])} | {row['late_cancellation']} |")
        lines += ["", "## Oriented traces (q by stage, 18-frame means)", "", "| token | R0 | R1 | T | R2 | M4 | M5 | R3 | c |", "|---|---|---|---|---|---|---|---|---|"]
        for row in sorted(exploration["tokens"].values(), key=lambda entry: entry["anomaly_score"]):
            q = row["oriented"]
            lines.append(f"| {row['token']} | {_f(q['R0'])} | {_f(q['R1'])} | {_f(q['T'])} | {_f(q['R2'])} | {_f(q['M4'])} | {_f(q['M5'])} | {_f(q['R3'])} | {_f(q['c'])} |")
        lines += ["", "## Determiner group", ""] + [f"- {name}: a {_f(entry['anomaly_score'])}, measured {_f(entry['measured_shift'])}, collapse {entry['collapse_stage']}, class {entry['encoding_class']}" for name, entry in exploration["determiner_group"].items()] + [""]
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
