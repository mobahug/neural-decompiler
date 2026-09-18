"""Experiment 007: a response-supervised, shared low-rank cue subspace of the layer-0 MLP encoding.

The circuit, the exposed pool, the E-patch residual response, the rank rule, the quality
gate, τ, the confirmation measurements, and the Y floors are inherited from Experiment 006
(``cue_decompilation``). This module adds the estimators frozen by the approved design
(revision 3): the supervised cross-moment SVD subspace with the singular-gap rule, the
pseudoinverse readout with the identifiability rule, the Experiment 006 PCA family refit
through the same solver, the template-specific nested ridge baseline, the leave-one-cue-out
tables for every family, the inherited-response replication check, the lock, the
program-axis outcome, the confirmation orchestration, and the report. The confirmation set
is Experiment 006's ``confirmation-v1.json``, read in place and never rewritten.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import cue_decompilation as cd
from . import plural_mechanism as pm
from .behavior import validate_json_safe
from .candidate_screening import ScreeningManifest
from .models import PYTHIA_70M

# ---------------------------------------------------------------------------
# Frozen protocol constants (design revision 3).

EXPERIMENT_DIR = "experiments/007-supervised-cue-subspace"
LOCK_RELATIVE_PATH = f"{EXPERIMENT_DIR}/preregistration-lock.json"
INHERITED_EXTRACT_RELATIVE_PATH = f"{EXPERIMENT_DIR}/inherited/experiment-006-epatch-means.json"
CONFIRMATION_RELATIVE_PATH = cd.CONFIRMATION_RELATIVE_PATH  # Experiment 006's file, read in place
INHERITED_CONFIRMATION_SHA256 = "dbb8dbcef9ab201cb5555a3c1d988e3ca8ab70f742f10b12f80a856c9ebf5521"
INHERITED_EPATCH_TOLERANCE = 1e-6
INHERITED_EXTRACT_SCHEMA_VERSION = 1
RUNTIME_SEED = 20260916
CONTROL_SEED = 20260920
RANKS = (1, 2, 3, 4)
GAP_MIN = 1e-8
MIN_CROSS_MOMENT_RANK = 5
RIDGE_MULTIPLIERS = (1e-2, 1e-1, 1.0, 10.0, 100.0)
FLOAT64_EPS = torch.finfo(torch.float64).eps
KIND_SUPERVISED, KIND_PCA, KIND_RIDGE = "supervised-svd", "pca-006", "ridge-full"
PROGRAM_NAMES = ("selected", "pca-006", "e005-scalar", "ridge-full")
PCA_IMPROVEMENT_PREDICTION = 0.8  # stated prediction: supervised LOCO error ≤ 0.8 × PCA-006 at the same rank (not a gate)
SCIENTIFIC_PATH_PREFIXES = ("src/", "experiments/007-supervised-cue-subspace/", "experiments/006-low-rank-cue-decompilation/",
                            "experiments/005-regular-plural-mechanism/", "screening/behavior-candidates/manifest-v1.json")
OUTCOME_LABELS = ("QUALITY_GATE_FAILED", "CUE_EFFECT_NOT_REPLICATED", "DECOMPILED", "DECOMPILED_MISCALIBRATED", "PROGRAM_NOT_SUPPORTED")


class PhaseError(cd.PhaseError):
    pass


# ---------------------------------------------------------------------------
# The inherited Experiment 006 responses (derived extract) and the replication check.


def inherited_extract_payload(responses: Mapping[str, Mapping[str, Any]], *, source: Mapping[str, Any], manifest_sha256: str, extension_sha256: str,
                              confirmation_sha256: str, model: Mapping[str, Any]) -> dict[str, Any]:
    """The committed extract format: ``token|frame`` → template, mean shift, residual-change norm, circuit share."""
    payload = {
        "schema_version": INHERITED_EXTRACT_SCHEMA_VERSION,
        "description": ("Derived extract of Experiment 006's recorded exposed E-patch residual responses (mean contrast shift per (token, frame) over the "
                        "sixty nouns, residual-change norm, circuit share). Experiment 007 recomputes these on the same exposed pool and requires the mean "
                        "shifts to match within 1e-6."),
        "source": dict(source), "manifest_sha256": manifest_sha256, "extension_sha256": extension_sha256, "confirmation_sha256": confirmation_sha256,
        "model": dict(model),
        "responses": {key: {"template_id": entry["template_id"], "mean_shift": float(entry["mean_shift"]), "delta_norm": float(entry["delta_norm"]),
                            "circuit_share": float(entry["circuit_share"])} for key, entry in sorted(responses.items())},
    }
    payload["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"}))
    return payload


def load_inherited_extract(path: Path, *, manifest_sha256: str, extension_sha256: str, confirmation_sha256: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read the inherited extract: {path}") from error
    pm._require_exact_keys(payload, {"schema_version", "description", "source", "manifest_sha256", "extension_sha256", "confirmation_sha256", "model", "responses", "content_sha256"}, "inherited extract")
    if payload["schema_version"] != INHERITED_EXTRACT_SCHEMA_VERSION:
        raise ValueError("inherited extract schema version is not frozen")
    if payload["content_sha256"] != pm.sha256_text(pm.canonical_json({key: value for key, value in payload.items() if key != "content_sha256"})):
        raise ValueError("inherited extract content_sha256 does not match its canonical payload")
    if payload["confirmation_sha256"] != INHERITED_CONFIRMATION_SHA256 or confirmation_sha256 != INHERITED_CONFIRMATION_SHA256:
        raise ValueError("the inherited extract or the confirmation set does not carry the frozen Experiment 006 confirmation digest")
    if (payload["manifest_sha256"], payload["extension_sha256"]) != (manifest_sha256, extension_sha256):
        raise ValueError("inherited extract was recorded against different frozen inputs")
    if len(payload["responses"]) != 16 * 12:
        raise ValueError(f"inherited extract must hold 192 responses, found {len(payload['responses'])}")
    return payload


def check_inherited_responses(responses: Mapping[tuple[str, str], cd.EPatchResponse], extract: Mapping[str, Any]) -> dict[str, Any]:
    """Recomputed mean shifts must reproduce Experiment 006's recorded means within the tolerance; otherwise an incident."""
    recorded = extract["responses"]
    keys = {f"{token}|{frame_id}" for token, frame_id in responses}
    if keys != set(recorded):
        raise pm.IncidentError("recomputed E-patch responses do not cover exactly the inherited (token, frame) pairs")
    deviations = {}
    for (token, frame_id), response in responses.items():
        key = f"{token}|{frame_id}"
        if recorded[key]["template_id"] != response.template_id:
            raise pm.IncidentError(f"{key}: template disagrees with the inherited record")
        deviations[key] = abs(pm._mean(list(response.shifts.values())) - recorded[key]["mean_shift"])
    worst = max(deviations, key=deviations.get)
    if deviations[worst] > INHERITED_EPATCH_TOLERANCE:
        raise pm.IncidentError(f"{worst}: recomputed E-patch mean shift deviates from Experiment 006 by {deviations[worst]:.3e} (> {INHERITED_EPATCH_TOLERANCE:.0e})")
    return {"passed": True, "n": len(deviations), "max_abs_deviation": deviations[worst], "worst_key": worst, "tolerance": INHERITED_EPATCH_TOLERANCE,
            "extract_content_sha256": extract["content_sha256"], "source": dict(extract["source"])}


# ---------------------------------------------------------------------------
# Design rows, the cross-moment SVD, the pseudoinverse readout, and the fits.


@dataclass(frozen=True)
class DesignRows:
    """Uncentered predictor and response rows for a token set: one row per (token, exposed frame)."""

    tokens: tuple[str, ...]
    X: torch.Tensor  # [n, d] float64, x(w, f) = ΔE_T(w)
    Y: torch.Tensor  # [n, d] float64, y(w, f) = Δr_Epatch(w, f)
    template_of_row: tuple[str, ...]
    token_of_row: tuple[str, ...]

    def rows_of(self, template: str) -> tuple[torch.Tensor, torch.Tensor]:
        index = [i for i, value in enumerate(self.template_of_row) if value == template]
        return self.X[index], self.Y[index]

    @property
    def d_model(self) -> int:
        return int(self.X.shape[1])


def design_rows(*, e_vectors: Mapping[str, torch.Tensor], reference_vectors: Mapping[str, torch.Tensor], responses: Mapping[tuple[str, str], cd.EPatchResponse],
                frames: Sequence[pm.Frame], tokens: Sequence[str]) -> DesignRows:
    xs, ys, templates, owners = [], [], [], []
    for token in tokens:
        for frame in frames:
            template = frame.template_id
            xs.append(e_vectors[token].double() - reference_vectors[template].double())
            ys.append(responses[(token, frame.frame_id)].delta_residual.double())
            templates.append(template)
            owners.append(token)
    return DesignRows(tuple(tokens), torch.stack(xs), torch.stack(ys), tuple(templates), tuple(owners))


@dataclass(frozen=True)
class CrossMomentSVD:
    """Left singular vectors of the uncentered cross-moment C = XᵀY, sign-fixed, with the gap diagnostics."""

    P: torch.Tensor  # [d, k] float64
    singular_values: tuple[float, ...]
    rank_C: int
    gaps: Mapping[int, float]  # r -> (σ_r − σ_{r+1}) / σ_1 for every evaluated rank

    def basis(self, rank: int) -> torch.Tensor:
        return self.P[:, :rank].contiguous()

    def diagnostics(self) -> dict[str, Any]:
        return {"singular_values": list(self.singular_values[: max(RANKS) + 1]), "rank_C": self.rank_C, "gaps": {str(rank): gap for rank, gap in self.gaps.items()}}


def fix_column_signs(matrix: torch.Tensor) -> torch.Tensor:
    """Each column's largest-magnitude entry is made positive (predictions do not depend on this; exports do)."""
    index = matrix.abs().argmax(dim=0)
    signs = torch.sign(matrix[index, torch.arange(matrix.shape[1])])
    signs[signs == 0] = 1.0
    return matrix * signs


def cross_moment_svd(X: torch.Tensor, Y: torch.Tensor, *, ranks: Sequence[int] = RANKS, where: str = "") -> CrossMomentSVD:
    """C = XᵀY (uncentered), C = P Σ Qᵀ; requires rank(C) ≥ 5 and gap_r ≥ 1e-8 for every evaluated rank (else incident)."""
    C = X.double().T @ Y.double()
    P, S, _ = torch.linalg.svd(C, full_matrices=False)
    sigma_1 = float(S[0])
    label = f"{where}: " if where else ""
    if not math.isfinite(sigma_1) or sigma_1 <= 0.0:
        raise pm.IncidentError(f"{label}the cross-moment XᵀY is zero or non-finite")
    rank_C = int((S > max(C.shape) * FLOAT64_EPS * sigma_1).sum())
    if rank_C < MIN_CROSS_MOMENT_RANK:
        raise pm.IncidentError(f"{label}rank(XᵀY) = {rank_C} < {MIN_CROSS_MOMENT_RANK}; σ_{{r+1}} is not defined for every evaluated rank")
    gaps = {int(rank): float((S[rank - 1] - S[rank]) / sigma_1) for rank in ranks}
    for rank, gap in gaps.items():
        if gap < GAP_MIN:
            raise pm.IncidentError(f"{label}singular gap at rank {rank} is {gap:.3e} < {GAP_MIN:.0e}; the span of U_{rank} is not identifiable")
    return CrossMomentSVD(fix_column_signs(P), tuple(float(value) for value in S), rank_C, gaps)


def pinv_readout(Z: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, int]:
    """V_Tᵀ = pinv(Z_T) Y_T with rcond = max(Z_T.shape) × eps(float64); returns V_T [d, k] and rank(Z_T)."""
    Z, Y = Z.double(), Y.double()
    rcond = max(Z.shape) * FLOAT64_EPS
    singular = torch.linalg.svdvals(Z)
    rank_Z = int((singular > rcond * singular[0]).sum()) if singular.numel() and float(singular[0]) > 0.0 else 0
    solution = torch.linalg.pinv(Z, rtol=rcond) @ Y  # [k, d]
    return solution.T.contiguous(), rank_Z


@dataclass(frozen=True)
class LinearFit:
    """A fitted linear cue program: δ̂ = maps[T] · (basisᵀ ΔE_T) for the subspace kinds, or maps[T] · ΔE_T for the ridge."""

    kind: str
    rank: int
    basis: torch.Tensor | None  # [d, r] float64, None for the ridge
    maps: Mapping[str, torch.Tensor]  # template -> [d, r] (or B_T [d, d])
    reference_ids: Mapping[str, int]
    fit_tokens: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    mu: torch.Tensor | None = None  # PCA-006 centering mean (informational; predictions are reference-relative)


def predict_delta(fit: LinearFit, template: str, e_vector: torch.Tensor, e_reference: torch.Tensor) -> torch.Tensor:
    delta_e = e_vector.double() - e_reference.double()
    coordinates = delta_e if fit.basis is None else fit.basis.T @ delta_e
    return fit.maps[template] @ coordinates


def fit_supervised(rows: DesignRows, rank: int, *, reference_ids: Mapping[str, int], svd: CrossMomentSVD | None = None, where: str = "") -> LinearFit:
    """The primary family: shared U_r from the cross-moment SVD, V_T by the pseudoinverse; rank(Z_T) < r is an incident."""
    svd = svd if svd is not None else cross_moment_svd(rows.X, rows.Y, where=where)
    basis = svd.basis(rank)
    maps, ranks_z = {}, {}
    for template in pm.TEMPLATE_ORDER:
        X_T, Y_T = rows.rows_of(template)
        V, rank_Z = pinv_readout(X_T @ basis, Y_T)
        if rank_Z < rank:
            raise pm.IncidentError(f"{where + ': ' if where else ''}rank(Z_T) = {rank_Z} < r = {rank} for template {template}; not every shared coordinate is identifiable")
        maps[template], ranks_z[template] = V, rank_Z
    return LinearFit(KIND_SUPERVISED, rank, basis, maps, dict(reference_ids), rows.tokens, {**svd.diagnostics(), "rank_Z": ranks_z})


def fit_pca_006(rows: DesignRows, rank: int, *, e_vectors: Mapping[str, torch.Tensor], reference_ids: Mapping[str, int]) -> LinearFit:
    """The Experiment 006 family (centered PCA of E over the fit tokens, reference-relative Δz) through the same pseudoinverse solve."""
    mu, basis, singular = cd.pca_basis([e_vectors[token] for token in rows.tokens], rank)
    maps, ranks_z = {}, {}
    for template in pm.TEMPLATE_ORDER:
        X_T, Y_T = rows.rows_of(template)
        V, rank_Z = pinv_readout(X_T @ basis, Y_T)
        maps[template], ranks_z[template] = V, rank_Z
    deficient = [template for template, value in ranks_z.items() if value < rank]
    return LinearFit(KIND_PCA, rank, basis, maps, dict(reference_ids), rows.tokens, {"singular_values": list(singular), "rank_Z": ranks_z, "rank_deficient_templates": deficient}, mu=mu)


def trace_scale(X: torch.Tensor) -> float:
    """tr(XᵀX)/d_model over the given training rows."""
    return float((X.double() * X.double()).sum() / X.shape[1])


def ridge_maps(rows: DesignRows, lam: float) -> dict[str, torch.Tensor]:
    """One B_T per template: B_Tᵀ = (X_TᵀX_T + λI)⁻¹ X_TᵀY_T (float64 solve)."""
    maps = {}
    identity = torch.eye(rows.d_model, dtype=torch.float64)
    for template in pm.TEMPLATE_ORDER:
        X_T, Y_T = rows.rows_of(template)
        solution = torch.linalg.solve(X_T.T @ X_T + lam * identity, X_T.T @ Y_T)  # [d, d] = B_Tᵀ
        maps[template] = solution.T.contiguous()
    return maps


def ridge_fit_from_maps(maps: Mapping[str, torch.Tensor], rows: DesignRows, *, reference_ids: Mapping[str, int], diagnostics: Mapping[str, Any]) -> LinearFit:
    return LinearFit(KIND_RIDGE, rows.d_model, None, dict(maps), dict(reference_ids), rows.tokens, dict(diagnostics))


# ---------------------------------------------------------------------------
# Contexts, predictions through the exact final LayerNorm, and cue-level values.


@dataclass(frozen=True)
class Contexts:
    rho_frame: Mapping[str, torch.Tensor]  # exposed frame id -> clean final pre-LayerNorm residual of the reference prompt (float64)
    rho_template: Mapping[str, torch.Tensor]  # template -> mean over its exposed frames

    def rho(self, template: str, frame_id: str | None) -> torch.Tensor:
        if frame_id is not None and frame_id in self.rho_frame:
            return self.rho_frame[frame_id]
        return self.rho_template[template]


def context_residuals(cache: pm.PromptCache, frames: Sequence[pm.Frame], reference_ids: Mapping[str, int]) -> Contexts:
    rho_frame = {frame.frame_id: cache.final_residual(pm.Prompt(frame, reference_ids[frame.template_id], "ref")).double() for frame in frames}
    rho_template = {template: torch.stack([rho_frame[frame.frame_id] for frame in frames if frame.template_id == template]).mean(dim=0) for template in pm.TEMPLATE_ORDER}
    return Contexts(rho_frame, rho_template)


def noun_matrix(weights: pm.Weights, nouns: Sequence[pm.Noun]) -> torch.Tensor:
    """Rows u_N = W_U[:, plural] − W_U[:, singular] (float64) for the single-token nouns, in order."""
    return torch.stack([weights.u(noun) for noun in nouns if noun.single_token])


def predict_shifts(weights: pm.Weights, rho: torch.Tensor, delta: torch.Tensor, u_matrix: torch.Tensor) -> torch.Tensor:
    """Δĉ_N = −u_N · [LN(ρ + δ̂) − LN(ρ)] for every noun row of ``u_matrix`` (exact LayerNorm, float64)."""
    gamma, beta = weights.ln_final_w.double(), weights.ln_final_b.double()
    difference = pm.exact_layer_norm(rho.double() + delta.double(), gamma, beta, weights.eps) - pm.exact_layer_norm(rho.double(), gamma, beta, weights.eps)
    return -(u_matrix @ difference)


@dataclass(frozen=True)
class Evaluation:
    """Everything a cue-level evaluation needs besides the fit."""

    weights: pm.Weights
    contexts: Contexts
    e_vectors: Mapping[str, torch.Tensor]
    reference_vectors: Mapping[str, torch.Tensor]
    responses: Mapping[tuple[str, str], cd.EPatchResponse]
    frames: tuple[pm.Frame, ...]
    nouns: tuple[pm.Noun, ...]  # single-token nouns, in order
    u_matrix: torch.Tensor


def make_evaluation(weights: pm.Weights, contexts: Contexts, e_vectors: Mapping[str, torch.Tensor], reference_vectors: Mapping[str, torch.Tensor],
                    responses: Mapping[tuple[str, str], cd.EPatchResponse], frames: Sequence[pm.Frame], nouns: Sequence[pm.Noun]) -> Evaluation:
    single = tuple(noun for noun in nouns if noun.single_token)
    return Evaluation(weights, contexts, dict(e_vectors), dict(reference_vectors), dict(responses), tuple(frames), single, noun_matrix(weights, single))


def cue_level_values(fit: LinearFit, ev: Evaluation, token: str) -> dict[str, float]:
    """For one token: mean predicted and measured E-patch shift over frames and nouns, and the mean absolute error."""
    predicted, measured = [], []
    for frame in ev.frames:
        template = frame.template_id
        delta = predict_delta(fit, template, ev.e_vectors[token], ev.reference_vectors[template])
        predicted.append(predict_shifts(ev.weights, ev.contexts.rho_frame[frame.frame_id], delta, ev.u_matrix))
        response = ev.responses[(token, frame.frame_id)]
        measured.append(torch.tensor([response.shifts[noun.lexical_key] for noun in ev.nouns], dtype=torch.float64))
    p, m = torch.cat(predicted), torch.cat(measured)
    return {"predicted": float(p.mean()), "measured": float(m.mean()), "mae": float((p - m).abs().mean())}


# ---------------------------------------------------------------------------
# The nested ridge selection and the leave-one-cue-out tables.


def select_ridge_multiplier(ev: Evaluation, training_tokens: Sequence[str], *, multipliers: Sequence[float] = RIDGE_MULTIPLIERS) -> dict[str, Any]:
    """Inner leave-one-cue-out over the training tokens; the grid scale comes from each inner fold's training rows only."""
    scores = {multiplier: 0.0 for multiplier in multipliers}
    for held_out in training_tokens:
        inner_tokens = [token for token in training_tokens if token != held_out]
        rows = design_rows(e_vectors=ev.e_vectors, reference_vectors=ev.reference_vectors, responses=ev.responses, frames=ev.frames, tokens=inner_tokens)
        scale = trace_scale(rows.X)
        for multiplier in multipliers:
            fit = ridge_fit_from_maps(ridge_maps(rows, multiplier * scale), rows, reference_ids={}, diagnostics={})
            scores[multiplier] += cue_level_values(fit, ev, held_out)["mae"] / len(training_tokens)
    chosen = choose_multiplier(scores)
    return {"multiplier": chosen, "inner_scores": {f"{multiplier:g}": score for multiplier, score in scores.items()}}


def choose_multiplier(scores: Mapping[float, float]) -> float:
    """The smallest inner score wins; exact ties go to the larger multiplier (more regularization)."""
    return min(scores, key=lambda multiplier: (scores[multiplier], -multiplier))


def fit_ridge_full(ev: Evaluation, training_tokens: Sequence[str], *, reference_ids: Mapping[str, int], multipliers: Sequence[float] = RIDGE_MULTIPLIERS) -> LinearFit:
    """Choose the multiplier by inner LOCO over the training tokens, then refit the three B_T on all of them at multiplier × tr(X_trainᵀX_train)/d."""
    selection = select_ridge_multiplier(ev, training_tokens, multipliers=multipliers)
    rows = design_rows(e_vectors=ev.e_vectors, reference_vectors=ev.reference_vectors, responses=ev.responses, frames=ev.frames, tokens=training_tokens)
    scale = trace_scale(rows.X)
    lam = selection["multiplier"] * scale
    diagnostics = {"multiplier": selection["multiplier"], "trace_scale": scale, "lambda": lam, "inner_scores": selection["inner_scores"], "grid_multipliers": list(multipliers)}
    return ridge_fit_from_maps(ridge_maps(rows, lam), rows, reference_ids=reference_ids, diagnostics=diagnostics)


def loco_tables(ev: Evaluation, tokens: Sequence[str], *, reference_ids: Mapping[str, int], ranks: Sequence[int] = RANKS, log: Any = None) -> dict[str, Any]:
    """Outer leave-one-cue-out by whole token for the supervised family, PCA-006, and Ridge-full, with per-fold diagnostics."""
    say = log or (lambda message: None)
    supervised: dict[int, dict[str, dict[str, float]]] = {rank: {} for rank in ranks}
    pca: dict[int, dict[str, dict[str, float]]] = {rank: {} for rank in ranks}
    ridge: dict[str, dict[str, float]] = {}
    folds: dict[str, dict[str, Any]] = {}
    for held_out in tokens:
        training = [token for token in tokens if token != held_out]
        rows = design_rows(e_vectors=ev.e_vectors, reference_vectors=ev.reference_vectors, responses=ev.responses, frames=ev.frames, tokens=training)
        svd = cross_moment_svd(rows.X, rows.Y, ranks=ranks, where=f"fold {held_out}")
        fold: dict[str, Any] = {**svd.diagnostics(), "rank_Z": {}, "pca_rank_Z": {}}
        for rank in ranks:
            fit = fit_supervised(rows, rank, reference_ids=reference_ids, svd=svd, where=f"fold {held_out}, rank {rank}")
            supervised[rank][held_out] = cue_level_values(fit, ev, held_out)
            fold["rank_Z"][str(rank)] = dict(fit.diagnostics["rank_Z"])
            baseline = fit_pca_006(rows, rank, e_vectors=ev.e_vectors, reference_ids=reference_ids)
            pca[rank][held_out] = cue_level_values(baseline, ev, held_out)
            fold["pca_rank_Z"][str(rank)] = dict(baseline.diagnostics["rank_Z"])
        ridge_fit = fit_ridge_full(ev, training, reference_ids=reference_ids)
        ridge[held_out] = cue_level_values(ridge_fit, ev, held_out)
        fold["ridge"] = {key: ridge_fit.diagnostics[key] for key in ("multiplier", "trace_scale", "lambda", "inner_scores")}
        folds[held_out] = fold
        say(f"  fold {held_out}: supervised MAE " + ", ".join(f"r{rank} {supervised[rank][held_out]['mae']:.3f}" for rank in ranks) + f"; ridge {ridge[held_out]['mae']:.3f} (×{fold['ridge']['multiplier']:g})")
    return {"supervised": supervised, "pca-006": pca, "ridge-full": ridge, "folds": folds}


def _mean_mae(per_token: Mapping[str, Mapping[str, float]]) -> float:
    return pm._mean([entry["mae"] for entry in per_token.values()])


def comparison_record(tables: Mapping[str, Any], selection: Mapping[str, Any], *, e005_error: float | None) -> dict[str, Any]:
    """The stated (non-gating) predictions: supervised vs PCA-006 per rank; Ridge-full vs the selected rank; E005-scalar."""
    per_rank = {}
    for rank, per_token in tables["supervised"].items():
        supervised_error = _mean_mae(per_token)
        pca_error = _mean_mae(tables["pca-006"][rank])
        per_rank[str(rank)] = {"supervised_error": supervised_error, "pca_006_error": pca_error, "ratio": supervised_error / pca_error if pca_error > 0 else None,
                               "prediction_met": supervised_error <= PCA_IMPROVEMENT_PREDICTION * pca_error}
    selected = int(selection["selected"])
    selected_error = float(selection["summary"][str(selected)]["error"])
    ridge_error = _mean_mae(tables["ridge-full"])
    return {"per_rank": per_rank, "selected_rank": selected, "selected_error": selected_error, "ridge_full_error": ridge_error,
            "ridge_over_selected": ridge_error / selected_error if selected_error > 0 else None, "e005_scalar_error": e005_error,
            "ridge_multipliers_by_fold": {token: fold["ridge"]["multiplier"] for token, fold in tables["folds"].items()},
            "reading": "supervised ≈ ridge: compact dimensionality plausible; ridge ≪ supervised: linear but not ≤ 4-dimensional; both poor: linear generalization from E(w) questionable"}


# ---------------------------------------------------------------------------
# Export of weight-only programs, loading, and the E005-scalar adapter.


def export_program(directory: Path, weights: pm.Weights, fit: LinearFit, contexts: Contexts) -> dict[str, Any]:
    from .provenance import tensor_digest

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {
        "W_E": weights.W_E, "W_U": weights.W_U, "ln_final_w": weights.ln_final_w, "ln_final_b": weights.ln_final_b,
        "ln2_0_w": weights.ln2_0_w, "ln2_0_b": weights.ln2_0_b, "mlp0_W_in": weights.mlp0_W_in, "mlp0_b_in": weights.mlp0_b_in,
        "mlp0_W_out": weights.mlp0_W_out, "mlp0_b_out": weights.mlp0_b_out,
    }
    if fit.basis is not None:
        tensors["basis"] = fit.basis
    for template, matrix in fit.maps.items():
        tensors[f"map.{template}"] = matrix
    for frame_id, vector in contexts.rho_frame.items():
        tensors[f"rho_frame.{frame_id}"] = vector
    for template, vector in contexts.rho_template.items():
        tensors[f"rho_template.{template}"] = vector
    digests = {}
    for name, tensor in tensors.items():
        path = directory / f"{name}.pt"
        dense = tensor.detach().cpu().clone(memory_format=torch.contiguous_format)  # a [d, 1] column slice can be "contiguous" with stride (1, d)
        torch.save(dense, path)
        digests[name] = {"file": path.name, "shape": list(dense.shape), "dtype": str(dense.dtype), "sha256": tensor_digest(dense)}
    index = {"eps": weights.eps, "act_fn": weights.act_fn, "kind": fit.kind, "rank": fit.rank, "templates": list(pm.TEMPLATE_ORDER),
             "reference_ids": dict(fit.reference_ids), "fit_tokens": list(fit.fit_tokens), "diagnostics": _json_safe(fit.diagnostics),
             "frames": sorted(contexts.rho_frame), "tensors": digests}
    validate_json_safe(index, path="parameters")
    (directory / "parameters.json").write_text(pm.canonical_json(index) + "\n", encoding="utf-8")
    return index


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_linear_program(parameters_dir: Path, program_path: Path) -> Any:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("linear_cue_program", str(program_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LinearCueProgram.load(Path(parameters_dir))


def load_programs(parameters_dir: Path, program_path: Path, program_005_path: Path, reference_ids: Mapping[str, int]) -> dict[str, Any]:
    parameters_dir = Path(parameters_dir)
    return {"selected": load_linear_program(parameters_dir / "selected", program_path), "pca-006": load_linear_program(parameters_dir / "pca-006", program_path),
            "e005-scalar": cd._E005Adapter(pm.load_program(parameters_dir / "e005-scalar", program_005_path)).bind(reference_ids),
            "ridge-full": load_linear_program(parameters_dir / "ridge-full", program_path)}


def _parameters_digest(directory: Path) -> str:
    return pm.sha256_text((Path(directory) / "parameters.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tier A orchestration.


def e005_cue_level(program: Any, pool: cd.ExposedPool, responses: Mapping[tuple[str, str], cd.EPatchResponse]) -> dict[str, dict[str, float]]:
    """The frozen Experiment 005 program's cue-level values on the exposed responses (its parameters are not refit per token)."""
    values = {}
    for token, token_id in pool.tokens:
        predicted, measured = [], []
        for frame in pool.frames:
            response = responses[(token, frame.frame_id)]
            for noun in pool.nouns:
                if noun.single_token:
                    predicted.append(program.predict_epatch_shift(frame.template_id, frame.frame_id, token_id, pool.reference_ids[frame.template_id], noun.sg_ids[0], noun.pl_ids[0]))
                    measured.append(response.shifts[noun.lexical_key])
        values[token] = {"predicted": pm._mean(predicted), "measured": pm._mean(measured), "mae": pm._mean([abs(p - m) for p, m in zip(predicted, measured)])}
    return values


def run_exploration(model: Any, pool: cd.ExposedPool, *, state: dict[str, Any], results_path: Path | None, parameters_dir: Path, program_path: Path, program_005_path: Path,
                    e005_index_sha256: str | None, inherited: Mapping[str, Any], circuit: pm.MechanismSet = cd.FIXED_CIRCUIT, ranks: Sequence[int] | None = None,
                    development_ctx_factory: Any = None, log: Any = None) -> dict[str, Any]:
    """Tier A: E-patch responses (replicated against Experiment 006), LOCO tables for every family, rank selection, gate, τ, exports."""
    say = log or (lambda message: None)
    ranks = tuple(ranks) if ranks is not None else tuple(RANKS)
    weights = pm.Weights.from_model(model)
    parameters_dir = Path(parameters_dir)
    say("E005-scalar baseline refit")
    ctx_dev = development_ctx_factory() if development_ctx_factory else None
    if ctx_dev is None:
        raise ValueError("a development context factory is required for the E005-scalar baseline")
    e005_program, e005_record = cd.e005_scalar_baseline(ctx_dev, parameters_dir=parameters_dir / "e005-scalar", program_path=program_005_path, expected_index_sha256=e005_index_sha256, circuit=circuit)
    state["exploration"]["e005_scalar"] = e005_record
    cache = pm.PromptCache(model, tuple(pool.nouns))
    say("E-patch residual responses (12 frames × 16 tokens)")
    responses = cd.measure_epatch_responses(model, weights, cache, pool.frames, pool.tokens, pool.reference_ids, circuit=circuit)
    state["exploration"]["epatch"] = {f"{token}|{frame_id}": {"template_id": r.template_id, "mean_shift": pm._mean(list(r.shifts.values())), "circuit_share": r.circuit_share,
                                                               "delta_norm": float(r.delta_residual.norm())} for (token, frame_id), r in responses.items()}
    state["exploration"]["inherited_check"] = check_inherited_responses(responses, inherited)
    say(f"inherited responses replicated: max deviation {state['exploration']['inherited_check']['max_abs_deviation']:.2e}")
    if results_path is not None:
        cd.write_results_state(results_path, state)
    e_vectors, reference_vectors = cd._token_vectors(weights, pool.tokens, pool.reference_ids)
    contexts = context_residuals(cache, pool.frames, pool.reference_ids)
    ev = make_evaluation(weights, contexts, e_vectors, reference_vectors, responses, pool.frames, pool.nouns)
    token_names = [token for token, _ in pool.tokens]
    say("leave-one-cue-out tables (supervised, PCA-006, Ridge-full)")
    tables = loco_tables(ev, token_names, reference_ids=pool.reference_ids, ranks=ranks, log=say)
    selection = cd.select_rank(tables["supervised"])
    selected = int(selection["selected"])
    gate = cd.quality_gate(tables["supervised"][selected], selected=selected, summary=selection["summary"])
    tau = cd.tolerance_tau(tables["supervised"][selected])
    e005_values = e005_cue_level(e005_program, pool, responses)
    e005_error = _mean_mae(e005_values)
    comparison = comparison_record(tables, selection, e005_error=e005_error)
    say(f"selected rank {selected} (gate {'passed' if gate['passed'] else 'FAILED'}); tau {tau:.3f}; ridge-full LOCO error {comparison['ridge_full_error']:.3f}")
    say("final fits on all sixteen tokens")
    rows = design_rows(e_vectors=e_vectors, reference_vectors=reference_vectors, responses=responses, frames=pool.frames, tokens=token_names)
    final = fit_supervised(rows, selected, reference_ids=pool.reference_ids, where="final fit")
    pca_final = fit_pca_006(rows, selected, e_vectors=e_vectors, reference_ids=pool.reference_ids)
    ridge_final = fit_ridge_full(ev, token_names, reference_ids=pool.reference_ids)
    indices = {"selected": export_program(parameters_dir / "selected", weights, final, contexts), "pca-006": export_program(parameters_dir / "pca-006", weights, pca_final, contexts),
               "ridge-full": export_program(parameters_dir / "ridge-full", weights, ridge_final, contexts)}
    exploration = state["exploration"]
    exploration["loco"] = {"supervised": {str(rank): per_token for rank, per_token in tables["supervised"].items()}, "pca-006": {str(rank): per_token for rank, per_token in tables["pca-006"].items()},
                           "ridge-full": tables["ridge-full"]}
    exploration["folds"] = tables["folds"]
    exploration["selection"] = selection
    exploration["quality_gate"] = gate
    exploration["tau"] = tau
    exploration["e005_scalar"]["cue_level"] = e005_values
    exploration["e005_scalar"]["error"] = e005_error
    exploration["comparison"] = comparison
    exploration["final_fit"] = {"supervised": _json_safe(final.diagnostics), "pca-006": _json_safe(pca_final.diagnostics), "ridge-full": _json_safe(ridge_final.diagnostics)}
    exploration["parameters"] = {name: pm.sha256_text(pm.canonical_json(index)) for name, index in indices.items()} | {"e005-scalar": e005_record["parameters_index_sha256"]}
    exploration["outcome"] = {"label": "QUALITY_GATE_FAILED", "note": "pre-lock; nothing fresh executed"} if not gate["passed"] else {"label": None, "note": "quality gate passed; the outcome is assigned by confirm"}
    if results_path is not None:
        cd.write_results_state(results_path, state)
    return exploration


# ---------------------------------------------------------------------------
# The lock, its validation, the outcome rule, the confirmation, and the report.


def build_candidate_lock(*, state: Mapping[str, Any], manifest: ScreeningManifest, manifest_sha256: str, extension: pm.Extension, confirmation: cd.Confirmation, inherited: Mapping[str, Any],
                         programs: Mapping[str, Any], parameters_dir: Path, program_path: Path, program_005_path: Path, protocol_code_commit: str) -> dict[str, Any]:
    exploration = state["exploration"]
    if not exploration["quality_gate"]["passed"]:
        raise PhaseError("the pre-lock quality gate failed; no lock may be written")
    if confirmation.content_sha256 != INHERITED_CONFIRMATION_SHA256:
        raise PhaseError("the confirmation set is not the frozen Experiment 006 set")
    tau = float(exploration["tau"])
    lock = {
        "schema_version": 1, "experiment": "007", "created_at": pm.utc_now(), "run_id": state["run_id"], "protocol_code_commit": protocol_code_commit,
        "manifest_sha256": manifest_sha256, "extension_sha256": extension.content_sha256, "confirmation_sha256": confirmation.content_sha256,
        "inherited": {"confirmation_path": CONFIRMATION_RELATIVE_PATH, "confirmation_sha256": INHERITED_CONFIRMATION_SHA256, "extract_content_sha256": inherited["content_sha256"],
                      "extract_source": dict(inherited["source"]), "replication": exploration["inherited_check"]},
        "confirmation_prompt_keys": sorted(prompt.key for prompt in confirmation.all_prompts), "fresh_noun_keys": [noun.key for noun in confirmation.nouns],
        "model": {"model_id": PYTHIA_70M.model_id, "revision": PYTHIA_70M.revision}, "seeds": {"runtime": RUNTIME_SEED, "control": CONTROL_SEED},
        "circuit": cd.FIXED_CIRCUIT.to_dict(),
        "estimator": {"kind": KIND_SUPERVISED, "ranks": list(RANKS), "gap_min": GAP_MIN, "min_cross_moment_rank": MIN_CROSS_MOMENT_RANK, "rcond_rule": "max(Z_T.shape) × eps(float64)",
                      "ridge_multipliers": list(RIDGE_MULTIPLIERS), "improvement_factor": cd.RANK_IMPROVEMENT_FACTOR, "quality_spearman": cd.QUALITY_SPEARMAN_FLOOR,
                      "quality_normalized_rmse": cd.QUALITY_NORMALIZED_RMSE_CEILING, "tau_min": cd.TAU_MIN_NATS, "tau_multiplier": cd.TAU_RMSE_MULTIPLIER},
        "rank": exploration["selection"]["selected"], "selection": exploration["selection"], "loco": exploration["loco"], "folds": exploration["folds"],
        "quality_gate": exploration["quality_gate"], "comparison": exploration["comparison"], "e005_scalar": exploration["e005_scalar"], "final_fit": exploration["final_fit"], "tau": tau,
        "parameters": {name: _parameters_digest(Path(parameters_dir) / name) for name in PROGRAM_NAMES},
        "program_source_sha256": pm.sha256_text(Path(program_path).read_text(encoding="utf-8")),
        "program_005_source_sha256": pm.sha256_text(Path(program_005_path).read_text(encoding="utf-8")),
        "floors": {"cue_effect_manifest": cd.CUE_EFFECT_MANIFEST, "cue_effect_fresh": cd.CUE_EFFECT_FRESH, "p3_sign_rate": cd.P3_SIGN_RATE, "p3_correlation": cd.P3_CORRELATION,
                   "p3_f": [cd.P3_F_OVERALL, cd.P3_F_TEMPLATE], "y1_spearman": cd.Y1_SPEARMAN, "y2_spearman": cd.Y2_SPEARMAN, "y2_tau_factor": cd.Y2_TAU_FACTOR,
                   "y1_confident_nats": cd.Y1_CONFIDENT_NATS, "y1_sign_frames": cd.Y1_SIGN_FRAMES, "y3_positive_rate": cd.Y3_POSITIVE_RATE, "band_tokens": cd.Y_BAND_TOKENS, "band_frames": cd.Y_BAND_FRAMES,
                   "circuit": {key: pm.FLOORS[key] for key in ("P1_overall", "P1_stratum", "P4", "P5a_H1", "P5b_H1", "P7_same", "P7_opposite_factor", "P8_loss", "P9", "P9_frozen_factor")}},
        "outcome_labels": list(OUTCOME_LABELS),
        "predictions": cd.lock_predictions(programs, confirmation, tau=tau),
        "tier_a_results_sha256": state.get("state_sha256"),
    }
    lock["content_sha256"] = pm.sha256_text(pm.canonical_json({key: value for key, value in lock.items() if key != "content_sha256"}))
    return lock


def validate_lock(lock: Mapping[str, Any], *, state: Mapping[str, Any], manifest_sha256: str, extension: pm.Extension, confirmation: cd.Confirmation, inherited: Mapping[str, Any],
                  parameters_dir: Path, program_path: Path, program_005_path: Path, git_state: Mapping[str, Any], tracked: bool, changed_paths: Sequence[str] | None) -> None:
    unsigned = {key: value for key, value in lock.items() if key != "content_sha256"}
    if lock.get("schema_version") != 1 or lock.get("experiment") != "007" or lock.get("content_sha256") != pm.sha256_text(pm.canonical_json(unsigned)):
        raise PhaseError("lock schema or content digest mismatch")
    if not tracked:
        raise PhaseError("the lock must be tracked and committed")
    if git_state.get("dirty", True):
        raise PhaseError("confirm requires a clean Git tree")
    if (lock["manifest_sha256"], lock["extension_sha256"], lock["confirmation_sha256"]) != (manifest_sha256, extension.content_sha256, confirmation.content_sha256):
        raise PhaseError("lock was built against different frozen inputs")
    if confirmation.content_sha256 != INHERITED_CONFIRMATION_SHA256 or lock["inherited"]["confirmation_sha256"] != INHERITED_CONFIRMATION_SHA256:
        raise PhaseError("the confirmation set differs from the frozen Experiment 006 set")
    if lock["inherited"]["extract_content_sha256"] != inherited["content_sha256"]:
        raise PhaseError("the inherited extract differs from the locked digest")
    if lock["run_id"] != state["run_id"] or lock["rank"] != state["exploration"]["selection"]["selected"]:
        raise PhaseError("lock does not match the results state")
    if not lock["quality_gate"]["passed"]:
        raise PhaseError("the locked quality gate did not pass")
    for name in PROGRAM_NAMES:
        if _parameters_digest(Path(parameters_dir) / name) != lock["parameters"][name]:
            raise PhaseError(f"{name} parameters differ from the locked digest")
    if pm.sha256_text(Path(program_path).read_text(encoding="utf-8")) != lock["program_source_sha256"] or pm.sha256_text(Path(program_005_path).read_text(encoding="utf-8")) != lock["program_005_source_sha256"]:
        raise PhaseError("a program source differs from the locked source")
    if changed_paths is None:
        raise PhaseError("the lock commit is not an ancestor of the current commit")
    scientific = [path for path in changed_paths if path.startswith(SCIENTIFIC_PATH_PREFIXES) and not path.endswith("preregistration-lock.json")]
    if scientific:
        raise PhaseError(f"scientific paths changed since the lock commit: {scientific}")
    cd.assert_confirmation_untouched(state, confirmation)


def outcome(*, fresh_floors: Mapping[str, Any], program_floors: Mapping[str, Any], bands: Mapping[str, Any]) -> dict[str, Any]:
    """The program-axis outcome; the circuit families never enter the label."""
    if not fresh_floors["cue_effect"]["passed"]:
        label = "CUE_EFFECT_NOT_REPLICATED"
    elif not program_floors["passed"]:
        label = "PROGRAM_NOT_SUPPORTED"
    elif bands["hit"]:
        label = "DECOMPILED"
    else:
        label = "DECOMPILED_MISCALIBRATED"
    return {"label": label, "program": "PROGRAM_PASS" if program_floors["passed"] else "PROGRAM_FAIL", "program_failures": list(program_floors["failures"]), "bands_hit": bool(bands["hit"]),
            "fresh_cue_effect_passed": bool(fresh_floors["cue_effect"]["passed"])}


def circuit_report(manifest_floors: Mapping[str, Any], fresh_floors: Mapping[str, Any]) -> dict[str, Any]:
    both = bool(manifest_floors["passed"] and fresh_floors["passed"])
    return {"manifest_passed": bool(manifest_floors["passed"]), "fresh_passed": bool(fresh_floors["passed"]), "manifest_failures": list(manifest_floors["failures"]),
            "fresh_failures": list(fresh_floors["failures"]), "c002_review_eligible": both, "enters_outcome": False}


def run_confirmation(model: Any, pool: cd.ExposedPool, confirmation: cd.Confirmation, programs: Mapping[str, Any], lock: Mapping[str, Any], *, circuit: pm.MechanismSet = cd.FIXED_CIRCUIT,
                     axes_dir: Path, log: Any = None) -> dict[str, Any]:
    say = log or (lambda message: None)
    weights = pm.Weights.from_model(model)
    universe = pm.universe_keys(model)
    axes = pm.site_axes_from_parameters(axes_dir)
    fresh_nouns = tuple(confirmation.nouns)
    say("fresh nouns on the manifest frames (circuit, reported)")
    ev_manifest = pm.EvalContext(model, weights, circuit, None, tuple(pool.manifest_frames), fresh_nouns, pm.PromptCache(model, fresh_nouns), axes, "H1")
    manifest_results = cd.circuit_families(ev_manifest, universe)
    manifest_floors = cd.circuit_floors(manifest_results, fresh=False)
    say("fresh frames with the original cue pairs (circuit, reported; cue-effect precondition)")
    cache_fresh = pm.PromptCache(model, fresh_nouns)
    ev_fresh = pm.EvalContext(model, weights, circuit, None, tuple(confirmation.frames), fresh_nouns, cache_fresh, axes, "H1")
    fresh_results = cd.circuit_families(ev_fresh, universe)
    fresh_floors = cd.circuit_floors(fresh_results, fresh=True)
    say("fresh cue tokens: behavior and E-patch (Y families)")
    measurements = cd.measure_confirmation_families(model, weights, confirmation, programs, cache=cache_fresh)
    tau = float(lock["tau"])
    program_floors = cd.y_floors(measurements, tau=tau)
    bands = cd.y_band_hits(measurements["per_token"], tau=tau)
    baselines = {name: {"Y1": cd.y_summary(measurements["per_token"], name, "epatch_measured", f"epatch_mae:{name}"), "Y2": cd.y_summary(measurements["per_token"], name, "behavior_measured", f"behavior_mae:{name}")}
                 for name in programs if name != "selected"}
    verdict = outcome(fresh_floors=fresh_floors, program_floors=program_floors, bands=bands)
    residual = {"y1_minus_y2_by_token": {word: pm._mean([entry["behavior_measured"] - entry["epatch_measured"] for entry in data["frames"].values()]) for word, data in measurements["per_token"].items()}}

    def strip(results: Mapping[str, Any]) -> dict[str, Any]:
        return {key: ({k: v for k, v in value.items() if k != "rows"} if isinstance(value, dict) else value) for key, value in results.items()}

    return {"manifest": {"results": strip(manifest_results), "floors": manifest_floors}, "fresh_frames": {"results": strip(fresh_results), "floors": fresh_floors},
            "circuit": circuit_report(manifest_floors, fresh_floors), "tokens": measurements, "program_floors": program_floors, "bands": bands, "baselines": baselines,
            "residual": residual, "outcome": verdict}


def render_report(state: Mapping[str, Any]) -> str:
    f = cd._f
    lines = ["# Experiment 007 Report", "", f"- Run ID: `{state['run_id']}`", f"- Confirmation sha256 (inherited from Experiment 006): `{state['confirmation_sha256']}`",
             f"- Protocol/code commit at explore: `{state['protocol_code_commit']}`", ""]
    lines += ["## Phases", ""] + [f"- `{phase}`: `{entry['status']}`" for phase, entry in state["phases"].items()] + [""]
    exploration = state.get("exploration", {})
    if "inherited_check" in exploration:
        check = exploration["inherited_check"]
        lines += ["## Tier A — inherited responses", "", f"- Recomputed E-patch mean shifts match Experiment 006 on {check['n']} (token, frame) pairs; max deviation {check['max_abs_deviation']:.2e} (tolerance {check['tolerance']:.0e})", ""]
    if "selection" in exploration:
        selection = exploration["selection"]
        comparison = exploration["comparison"]
        lines += ["## Rank selection (leave-one-cue-out, supervised cross-moment SVD)", ""]
        for rank, entry in sorted(selection["summary"].items(), key=lambda kv: int(kv[0])):
            pair = comparison["per_rank"][rank]
            lines.append(f"- rank {rank}: error {entry['error']:.4f} ± {entry['se']:.4f}; PCA-006 {pair['pca_006_error']:.4f} (ratio {f(pair['ratio'])}, prediction {'met' if pair['prediction_met'] else 'not met'})")
        lines += [f"- eligible {selection['eligible']}, best {selection['r_best']}, threshold {selection['threshold']:.4f}, **selected r = {selection['selected']}**",
                  f"- Ridge-full LOCO error {comparison['ridge_full_error']:.4f} (ratio to selected {f(comparison['ridge_over_selected'])}); E005-scalar cue-level error {f(comparison['e005_scalar_error'])}",
                  f"- Ridge multipliers by fold: {comparison['ridge_multipliers_by_fold']}", ""]
        gate = exploration["quality_gate"]
        lines += [f"- Quality gate: {'passed' if gate['passed'] else 'FAILED'} (Spearman {gate['spearman']:.3f}, normalized RMSE {gate['normalized_rmse']:.3f}, improvement {gate['improvement_ok']}); τ = {exploration['tau']:.3f}",
                  f"- Outcome at explore: `{exploration['outcome']['label']}`" if exploration["outcome"]["label"] else "- Outcome at explore: gate passed (assigned by confirm)", ""]
        final = exploration.get("final_fit", {}).get("supervised", {})
        if final:
            lines += [f"- Final fit: rank(C) {final['rank_C']}, gaps {final['gaps']}, rank(Z_T) {final['rank_Z']}", ""]
        lines += ["| token | measured | supervised (LOCO) | PCA-006 (LOCO) | Ridge-full (LOCO) | E005-scalar |", "|---|---|---|---|---|---|"]
        selected = str(selection["selected"])
        for token, entry in gate["cue_level"].items():
            pca = exploration["loco"]["pca-006"][selected].get(token, {})
            ridge = exploration["loco"]["ridge-full"].get(token, {})
            e005 = exploration["e005_scalar"]["cue_level"].get(token, {})
            lines.append(f"| {token} | {f(entry['measured'])} | {f(entry['predicted'])} | {f(pca.get('predicted'))} | {f(ridge.get('predicted'))} | {f(e005.get('predicted'))} |")
        lines.append("")
    if state.get("lock"):
        lines += ["## Lock", "", f"- Candidate lock sha256 `{state['lock']['content_sha256']}` (rank {state['lock']['rank']}, τ {f(state['lock']['tau'])})", ""]
    confirmation = state.get("confirmation")
    if confirmation:
        verdict = confirmation["outcome"]
        lines += [f"## Confirmation — outcome `{verdict['label']}`", "", f"- Program `{verdict['program']}` (failures {verdict['program_failures'] or 'none'}); bands hit {verdict['bands_hit']}; fresh cue effect {verdict['fresh_cue_effect_passed']}", ""]
        program_floors = confirmation["program_floors"]
        lines += ["### Program (Y families)", "", f"- Y1 {'pass' if program_floors['Y1']['passed'] else 'FAIL'}: Spearman {program_floors['Y1']['spearman']:.3f}, MAE {program_floors['Y1']['mae']:.3f} (τ {program_floors['tau']:.3f}), confident-sign ok {program_floors['Y1']['signs_ok']}",
                  f"- Y2 {'pass' if program_floors['Y2']['passed'] else 'FAIL'}: Spearman {program_floors['Y2']['spearman']:.3f}, MAE {program_floors['Y2']['mae']:.3f}",
                  f"- Y3 {'pass' if program_floors['Y3']['passed'] else 'FAIL'}: within τ {program_floors['Y3']['within_tau']}, positive {program_floors['Y3']['positive_pairs']}/{program_floors['Y3']['required']}",
                  f"- Bands: {confirmation['bands']['tokens_hit']}/24 tokens inside ± τ in ≥ 5/6 frames", ""]
        for name, entry in confirmation["baselines"].items():
            lines.append(f"- Y4 {name}: Y1 Spearman {entry['Y1']['spearman']:.3f}, MAE {entry['Y1']['mae']:.3f}; Y2 Spearman {entry['Y2']['spearman']:.3f}, MAE {entry['Y2']['mae']:.3f}")
        lines += ["", "| token | category | E-patch measured | behavior measured | selected | PCA-006 | Ridge-full | E005-scalar |", "|---|---|---|---|---|---|---|---|"]
        for word, data in confirmation["tokens"]["per_token"].items():
            frames = list(data["frames"].values())
            lines.append(f"| {word} | {data['category']} | {pm._mean([x['epatch_measured'] for x in frames]):.3f} | {pm._mean([x['behavior_measured'] for x in frames]):.3f} | "
                         f"{pm._mean([x['predicted:selected'] for x in frames]):.3f} | {pm._mean([x['predicted:pca-006'] for x in frames]):.3f} | {pm._mean([x['predicted:ridge-full'] for x in frames]):.3f} | {pm._mean([x['predicted:e005-scalar'] for x in frames]):.3f} |")
        lines.append("")
        circuit = confirmation["circuit"]
        lines += ["### Circuit families (reported; not in the outcome)", "", f"- Fresh nouns on manifest frames: {'pass' if circuit['manifest_passed'] else 'FAIL'} (failures {circuit['manifest_failures'] or 'none'})",
                  f"- Fresh frames: {'pass' if circuit['fresh_passed'] else 'FAIL'} (failures {circuit['fresh_failures'] or 'none'})", f"- C002 review eligible: {circuit['c002_review_eligible']}", ""]
        for label, key in (("Fresh nouns on manifest frames", "manifest"), ("Fresh frames", "fresh_frames")):
            floors = confirmation[key]["floors"]
            lines += [f"#### {label}", "", f"- Cue effect {floors['cue_effect']['positive_pairs']}/{floors['cue_effect']['n']} (floor {floors['cue_effect']['required']})",
                      "- " + ", ".join(f"{family} {'pass' if floors[family]['passed'] else 'FAIL'}" for family in ("P1", "P3", "P4", "P5", "P7", "P8", "P9")), cd._floor_line(floors), ""]
    lines += ["## Execution ledger", "", f"- Executed prompt keys: {len(state['executed_prompt_keys'])}", f"- Executed noun keys: {len(state['executed_noun_keys'])}", ""]
    return "\n".join(lines)
