"""The decompiled computation for Experiment 005, evaluated without the network.

This module reads only the tensors exported under ``outputs/experiment-005/parameters/``
(token embedding, block-0 second LayerNorm and MLP, final LayerNorm, unembedding, the
E_program axis, the per-template increment vectors, and the frozen context residuals).
It never imports TransformerLens, Transformers, or the instrumentation modules, and it
never runs attention, any block beyond the declared E_program sub-modules, or a forward
pass over a prompt. Its inputs are token IDs; its outputs are predicted contrasts.

Sign conventions: ``c = log P(singular) − log P(plural)``; the singular cue maps to
``n ≈ −1`` and the plural cue to ``n ≈ +1``; ``u_N = W_U[:, plural] − W_U[:, singular]``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

_FORBIDDEN_IMPORTS = ("transformer_lens", "transformers", "neural_decompiler.capture", "neural_decompiler.interventions")


def _tensor_digest(tensor: torch.Tensor) -> str:
    """Identical to ``neural_decompiler.provenance.tensor_digest`` (dtype, shape, exact bytes)."""
    value = tensor.detach().contiguous().cpu()
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def exact_layer_norm(r: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm with population variance (correction=0) and the pinned eps."""
    centered = r - r.mean(dim=-1, keepdim=True)
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    return weight * centered / torch.sqrt(variance + eps) + bias


@dataclass(frozen=True)
class MechanismProgram:
    eps: float
    e_program_keys: tuple[str, ...]
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
    e_axis_mu: torch.Tensor
    e_axis_direction: torch.Tensor
    e_axis_sigma: float
    k_t: float
    v_t: Mapping[str, torch.Tensor]
    rho_frame: Mapping[str, torch.Tensor]
    rho_template: Mapping[str, torch.Tensor]
    coordinated_template: str
    lexicon_table: torch.Tensor  # [vocab] n_c for every token, float64

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, directory: Path) -> "MechanismProgram":
        directory = Path(directory)
        index = json.loads((directory / "parameters.json").read_text(encoding="utf-8"))
        tensors: dict[str, torch.Tensor] = {}
        for name, entry in index["tensors"].items():
            tensor = torch.load(directory / entry["file"], map_location="cpu", weights_only=True)
            if _tensor_digest(tensor) != entry["sha256"] or list(tensor.shape) != entry["shape"]:
                raise ValueError(f"parameter {name} does not match its recorded digest or shape")
            tensors[name] = tensor.double()
        if index["act_fn"] != "gelu":
            raise ValueError("the program assumes exact GELU in the layer-0 MLP")
        v_t = {template: tensors[f"v_t.{template}"] for template in index["templates"]}
        rho_frame = {name[len("rho_frame."):]: tensor for name, tensor in tensors.items() if name.startswith("rho_frame.")}
        rho_template = {template: tensors[f"rho_template.{template}"] for template in index["templates"]}
        program = cls(
            eps=float(index["eps"]), e_program_keys=tuple(index["e_program_keys"]),
            W_E=tensors["W_E"], W_U=tensors["W_U"], ln_final_w=tensors["ln_final_w"], ln_final_b=tensors["ln_final_b"],
            ln2_0_w=tensors["ln2_0_w"], ln2_0_b=tensors["ln2_0_b"], mlp0_W_in=tensors["mlp0_W_in"], mlp0_b_in=tensors["mlp0_b_in"],
            mlp0_W_out=tensors["mlp0_W_out"], mlp0_b_out=tensors["mlp0_b_out"],
            e_axis_mu=tensors["e_axis_mu"], e_axis_direction=tensors["e_axis_direction"], e_axis_sigma=float(index["e_axis_sigma"]),
            k_t=float(index["k_t"]), v_t=v_t, rho_frame=rho_frame, rho_template=rho_template,
            coordinated_template=str(index["coordinated_template"]), lexicon_table=torch.empty(0),
        )
        table = program._build_lexicon_table(int(index["vocab_size"]))
        return cls(**{**program.__dict__, "lexicon_table": table})

    # -- encoding: the token-local lexicon ---------------------------------

    def e_program_vector(self, token_id: int) -> torch.Tensor:
        """Output of the declared token-local paths for one token, in float64."""
        parts = []
        for key in self.e_program_keys:
            if key == "L00.MLP":
                hidden = exact_layer_norm(self.W_E[int(token_id)], self.ln2_0_w, self.ln2_0_b, self.eps)
                pre = hidden @ self.mlp0_W_in + self.mlp0_b_in
                parts.append(torch.nn.functional.gelu(pre) @ self.mlp0_W_out + self.mlp0_b_out)
            elif key == "EMBED":
                parts.append(self.W_E[int(token_id)])
            else:
                raise ValueError(f"{key} is not a token-local path")
        return torch.stack(parts).sum(dim=0)

    def _e_program_batch(self, ids: torch.Tensor) -> torch.Tensor:
        parts = []
        for key in self.e_program_keys:
            if key == "L00.MLP":
                hidden = exact_layer_norm(self.W_E[ids], self.ln2_0_w, self.ln2_0_b, self.eps)
                pre = hidden @ self.mlp0_W_in + self.mlp0_b_in
                parts.append(torch.nn.functional.gelu(pre) @ self.mlp0_W_out + self.mlp0_b_out)
            elif key == "EMBED":
                parts.append(self.W_E[ids])
            else:
                raise ValueError(f"{key} is not a token-local path")
        return torch.stack(parts).sum(dim=0)

    def _build_lexicon_table(self, vocab_size: int) -> torch.Tensor:
        values = torch.empty(vocab_size, dtype=torch.float64)
        step = 4096
        for start in range(0, vocab_size, step):
            ids = torch.arange(start, min(start + step, vocab_size))
            vectors = self._e_program_batch(ids)
            values[start:start + len(ids)] = ((vectors - self.e_axis_mu) @ self.e_axis_direction) / self.e_axis_sigma
        return values

    def n_c(self, token_id: int) -> float:
        return float(self.lexicon_table[int(token_id)])

    def lexicon_digest(self) -> str:
        return _tensor_digest(self.lexicon_table)

    # -- transport and readout ----------------------------------------------

    def n_t(self, template: str, token_id: int) -> float:
        value = self.n_c(token_id)
        return self.k_t * value if template == self.coordinated_template else value

    def _context(self, template: str, frame_id: str | None) -> torch.Tensor:
        if frame_id is not None and frame_id in self.rho_frame:
            return self.rho_frame[frame_id]
        return self.rho_template[template]

    def _u(self, sg_id: int, pl_id: int) -> torch.Tensor:
        return self.W_U[:, int(pl_id)] - self.W_U[:, int(sg_id)]

    def _readout(self, residual: torch.Tensor, sg_id: int, pl_id: int) -> float:
        normalized = exact_layer_norm(residual, self.ln_final_w, self.ln_final_b, self.eps)
        return float(-self._u(sg_id, pl_id) @ normalized)

    def predict_contrast(self, template: str, frame_id: str | None, cue_token_id: int, sg_id: int, pl_id: int) -> float:
        """ĉ(x) = −u_N · LN(ρ + n_t · v_T)."""
        residual = self._context(template, frame_id) + self.n_t(template, cue_token_id) * self.v_t[template]
        return self._readout(residual, sg_id, pl_id)

    def predict_pair(self, template: str, frame_id: str | None, sg_cue_id: int, pl_cue_id: int, sg_id: int, pl_id: int) -> float:
        """d̂_full = ĉ(singular-cue prompt) − ĉ(plural-cue prompt)."""
        return self.predict_contrast(template, frame_id, sg_cue_id, sg_id, pl_id) - self.predict_contrast(template, frame_id, pl_cue_id, sg_id, pl_id)

    def predict_shift(self, template: str, frame_id: str | None, word_token_id: int, reference_cue_id: int, sg_id: int, pl_id: int) -> float:
        """Δ̂c(w) = ĉ(frame, w) − ĉ(frame, reference cue)."""
        return self.predict_contrast(template, frame_id, word_token_id, sg_id, pl_id) - self.predict_contrast(template, frame_id, reference_cue_id, sg_id, pl_id)

    def predict_epatch_shift(self, template: str, frame_id: str | None, word_token_id: int, reference_cue_id: int, sg_id: int, pl_id: int) -> float:
        """Shift from replacing E_program alone with the word's encoding in the reference prompt."""
        context = self._context(template, frame_id)
        delta_ref = self.n_t(template, reference_cue_id) * self.v_t[template]
        gain = self.k_t if template == self.coordinated_template else 1.0
        delta_e = gain * (self.n_c(word_token_id) - self.n_c(reference_cue_id)) * self.v_t[template]
        return self._readout(context + delta_ref + delta_e, sg_id, pl_id) - self._readout(context + delta_ref, sg_id, pl_id)


def forbidden_modules_loaded() -> list[str]:
    """Names of forbidden modules present in ``sys.modules`` (for the independence test)."""
    import sys

    return [name for name in sys.modules if any(name == item or name.startswith(item + ".") for item in _FORBIDDEN_IMPORTS)]
