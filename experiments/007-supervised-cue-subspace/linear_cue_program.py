"""The decompiled linear cue computation for Experiment 007, evaluated without the network.

Reads only the tensors exported under ``outputs/experiment-007/parameters/<name>/``: the
token embedding, block-0 second LayerNorm and MLP (the token-local encoder E), the final
LayerNorm, the unembedding, the shared basis ``U_r`` (absent for the ridge baseline), one
readout map per template, the reference cue ids, and the frozen context residuals. It
never imports TransformerLens, Transformers, or the instrumentation modules, and it never
runs attention, a block, or a prompt.

Coordinates (design revision 3):
    ΔE_T(w)   = E(w) − E(ref_T)
    z(w, T)   = U_rᵀ ΔE_T(w)                  (kinds supervised-svd and pca-006)
    δ̂(w, T)   = V_T z(w, T)                   or  B_T ΔE_T(w) for kind ridge-full
    Δĉ_N(w,f) = −u_N · [ LN(ρ_f + δ̂) − LN(ρ_f) ]
with ``c = log P(singular) − log P(plural)`` and ``u_N = W_U[:, plural] − W_U[:, singular]``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

_FORBIDDEN_IMPORTS = ("transformer_lens", "transformers", "neural_decompiler.capture", "neural_decompiler.interventions")
_KINDS = ("supervised-svd", "pca-006", "ridge-full")
_NATIVE_DTYPE_TENSORS = ("W_E", "W_U", "ln2_0_w", "ln2_0_b", "mlp0_W_in", "mlp0_b_in", "mlp0_W_out", "mlp0_b_out")


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def exact_layer_norm(r: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    centered = r - r.mean(dim=-1, keepdim=True)
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    return weight * centered / torch.sqrt(variance + eps) + bias


@dataclass(frozen=True)
class LinearCueProgram:
    kind: str
    eps: float
    rank: int
    W_E: torch.Tensor
    W_U: torch.Tensor
    ln_final_w: torch.Tensor
    ln_final_b: torch.Tensor
    ln2_0_w: torch.Tensor
    ln2_0_b: torch.Tensor
    mlp0_W_in: torch.Tensor
    mlp0_b_in: torch.Tensor
    mlp0_W_out: torch.Tensor
    mlp0_b_out: torch.Tensor
    basis: torch.Tensor | None  # U_r [d_model, r]; None for the ridge baseline
    maps: Mapping[str, torch.Tensor]  # template -> V_T [d_model, r] or B_T [d_model, d_model]
    reference_ids: Mapping[str, int]
    rho_frame: Mapping[str, torch.Tensor]
    rho_template: Mapping[str, torch.Tensor]

    @classmethod
    def load(cls, directory: Path) -> "LinearCueProgram":
        directory = Path(directory)
        index = json.loads((directory / "parameters.json").read_text(encoding="utf-8"))
        if index["act_fn"] != "gelu":
            raise ValueError("the program assumes exact GELU in the layer-0 MLP")
        if index["kind"] not in _KINDS:
            raise ValueError(f"unknown program kind {index['kind']}")
        tensors: dict[str, torch.Tensor] = {}
        for name, entry in index["tensors"].items():
            tensor = torch.load(directory / entry["file"], map_location="cpu", weights_only=True)
            if _tensor_digest(tensor) != entry["sha256"] or list(tensor.shape) != entry["shape"]:
                raise ValueError(f"parameter {name} does not match its recorded digest or shape")
            # Encoder and unembedding tensors keep their exported (float32) dtype so that E(w) and u_N are computed
            # exactly as the runner computes them; the fitted maps, contexts, and LayerNorm parameters are float64.
            tensors[name] = tensor if name in _NATIVE_DTYPE_TENSORS else tensor.double()
        templates = list(index["templates"])
        basis = tensors.get("basis")
        if (basis is None) != (index["kind"] == "ridge-full"):
            raise ValueError("the basis tensor must be present exactly for the subspace kinds")
        return cls(
            kind=index["kind"], eps=float(index["eps"]), rank=int(index["rank"]),
            W_E=tensors["W_E"], W_U=tensors["W_U"], ln_final_w=tensors["ln_final_w"], ln_final_b=tensors["ln_final_b"],
            ln2_0_w=tensors["ln2_0_w"], ln2_0_b=tensors["ln2_0_b"], mlp0_W_in=tensors["mlp0_W_in"], mlp0_b_in=tensors["mlp0_b_in"],
            mlp0_W_out=tensors["mlp0_W_out"], mlp0_b_out=tensors["mlp0_b_out"],
            basis=basis, maps={template: tensors[f"map.{template}"] for template in templates},
            reference_ids={template: int(index["reference_ids"][template]) for template in templates},
            rho_frame={name[len("rho_frame."):]: tensor for name, tensor in tensors.items() if name.startswith("rho_frame.")},
            rho_template={template: tensors[f"rho_template.{template}"] for template in templates},
        )

    # -- encoding -----------------------------------------------------------

    def E(self, token_id: int) -> torch.Tensor:
        """Weight-only L00.MLP output for a token, computed in the weights' own dtype and returned in float64."""
        hidden = exact_layer_norm(self.W_E[int(token_id)], self.ln2_0_w, self.ln2_0_b, self.eps)
        pre = hidden @ self.mlp0_W_in + self.mlp0_b_in
        return (torch.nn.functional.gelu(pre) @ self.mlp0_W_out + self.mlp0_b_out).double()

    def dE(self, template: str, token_id: int) -> torch.Tensor:
        return self.E(token_id) - self.E(self.reference_ids[template])

    def z(self, template: str, token_id: int) -> torch.Tensor:
        """Shared coordinates U_rᵀ ΔE_T(w); for the ridge baseline the coordinates are ΔE_T(w) itself."""
        delta = self.dE(template, token_id)
        return delta if self.basis is None else self.basis.T @ delta

    def delta(self, template: str, token_id: int) -> torch.Tensor:
        return self.maps[template] @ self.z(template, token_id)

    # -- readout ------------------------------------------------------------

    def _context(self, template: str, frame_id: str | None) -> torch.Tensor:
        if frame_id is not None and frame_id in self.rho_frame:
            return self.rho_frame[frame_id]
        return self.rho_template[template]

    def _u(self, sg_id: int, pl_id: int) -> torch.Tensor:
        return (self.W_U[:, int(pl_id)] - self.W_U[:, int(sg_id)]).double()

    def predict_epatch_shift(self, template: str, frame_id: str | None, token_id: int, sg_id: int, pl_id: int) -> float:
        rho = self._context(template, frame_id)
        delta = self.delta(template, token_id)
        normalized_after = exact_layer_norm(rho + delta, self.ln_final_w, self.ln_final_b, self.eps)
        normalized_before = exact_layer_norm(rho, self.ln_final_w, self.ln_final_b, self.eps)
        return float(-self._u(sg_id, pl_id) @ (normalized_after - normalized_before))

    def predict_behavior_shift(self, template: str, frame_id: str | None, token_id: int, sg_id: int, pl_id: int) -> float:
        """The behavioral prediction uses the same expression; routes not through E are residual."""
        return self.predict_epatch_shift(template, frame_id, token_id, sg_id, pl_id)

    def predict_pair(self, template: str, frame_id: str | None, sg_cue_id: int, pl_cue_id: int, sg_id: int, pl_id: int) -> float:
        """d̂_full = Δĉ(singular cue) − Δĉ(plural cue); the reference cue's own term is zero by construction."""
        return self.predict_epatch_shift(template, frame_id, sg_cue_id, sg_id, pl_id) - self.predict_epatch_shift(template, frame_id, pl_cue_id, sg_id, pl_id)


def forbidden_modules_loaded() -> list[str]:
    import sys

    return [name for name in sys.modules if any(name == item or name.startswith(item + ".") for item in _FORBIDDEN_IMPORTS)]
