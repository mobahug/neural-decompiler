"""A tiny GPT-NeoX-shaped parallel-residual transformer with the canonical Bridge hooks.

It has real LayerNorm, exact GELU, causal attention with per-head results, an
unembedding, and TransformerLens-style parameter names, so Experiment 005's
exact-accounting code paths run end to end offline.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Callable

import itertools

import torch

from neural_decompiler import plural_mechanism as pm
from neural_decompiler.candidate_screening import Split


class _Hook:
    def __init__(self, name: str) -> None:
        self.name = name


class _LN:
    def __init__(self, d: int, generator: torch.Generator) -> None:
        self.w = 1.0 + 0.1 * torch.randn(d, generator=generator)
        self.b = 0.05 * torch.randn(d, generator=generator)


class _MLP:
    def __init__(self, d: int, d_mlp: int, generator: torch.Generator) -> None:
        self.W_in = torch.randn(d, d_mlp, generator=generator) / d**0.5
        self.b_in = 0.1 * torch.randn(d_mlp, generator=generator)
        self.W_out = torch.randn(d_mlp, d, generator=generator) / d_mlp**0.5
        self.b_out = 0.1 * torch.randn(d, generator=generator)


class _Attn:
    def __init__(self, d: int, n_heads: int, generator: torch.Generator) -> None:
        d_head = d // n_heads
        self.W_Q = torch.randn(n_heads, d, d_head, generator=generator) / d**0.5
        self.W_K = torch.randn(n_heads, d, d_head, generator=generator) / d**0.5
        self.W_V = torch.randn(n_heads, d, d_head, generator=generator) / d**0.5
        self.W_O = torch.randn(n_heads, d_head, d, generator=generator) / d_head**0.5
        self.b_O = 0.1 * torch.randn(d, generator=generator)


class _Block:
    def __init__(self, d: int, n_heads: int, d_mlp: int, generator: torch.Generator) -> None:
        self.ln1 = _LN(d, generator)
        self.ln2 = _LN(d, generator)
        self.attn = _Attn(d, n_heads, generator)
        self.mlp = _MLP(d, d_mlp, generator)


def _layer_norm(x: torch.Tensor, ln: _LN, eps: float) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), ln.w, ln.b, eps)


class TinyPlural:
    """Deterministic two-layer parallel-residual model; tokens are vocabulary IDs."""

    def __init__(self, *, seed: int = 0, d_model: int = 8, n_heads: int = 2, d_mlp: int = 16, n_layers: int = 2, d_vocab: int = 16) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.cfg = SimpleNamespace(n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_mlp=d_mlp, d_vocab=d_vocab,
                                   eps=1e-5, act_fn="gelu", normalization_type="LN", use_attn_result=False,
                                   positional_embedding_type="rotary", parallel_attn_mlp=True, device="cpu")
        self.embed = SimpleNamespace(W_E=torch.randn(d_vocab, d_model, generator=generator))
        self.blocks = [_Block(d_model, n_heads, d_mlp, generator) for _ in range(n_layers)]
        self.ln_final = _LN(d_model, generator)
        self.unembed = SimpleNamespace(W_U=torch.randn(d_model, d_vocab, generator=generator), b_U=torch.zeros(d_vocab))
        names = {"embed.hook_out", "ln_final.hook_out", "unembed.hook_out"}
        for layer in range(n_layers):
            names.update({f"blocks.{layer}.hook_in", f"blocks.{layer}.attn.hook_out", f"blocks.{layer}.attn.hook_result",
                          f"blocks.{layer}.attn.hook_pattern", f"blocks.{layer}.mlp.out.hook_in", f"blocks.{layer}.mlp.hook_out",
                          f"blocks.{layer}.hook_out"})
        self.hook_dict = {name: _Hook(name) for name in names}
        self.compatibility_mode = False
        self.training = True

    def eval(self) -> "TinyPlural":
        self.training = False
        return self

    def set_use_attn_result(self, value: bool) -> None:
        self.cfg.use_attn_result = value

    def __call__(self, tokens: torch.Tensor, *, prepend_bos: bool = False) -> torch.Tensor:
        return self._forward(tokens, {}, prepend_bos=prepend_bos)

    def run_with_hooks(self, tokens: torch.Tensor, *, fwd_hooks: list[tuple[str, Callable]], prepend_bos: bool = False, **_: object) -> torch.Tensor:
        callbacks: dict[str, list[Callable]] = defaultdict(list)
        for name, callback in fwd_hooks:
            callbacks[name].append(callback)
        return self._forward(tokens, callbacks, prepend_bos=prepend_bos)

    def _forward(self, tokens: torch.Tensor, callbacks: dict[str, list[Callable]], *, prepend_bos: bool) -> torch.Tensor:
        if prepend_bos:
            raise ValueError("TinyPlural expects explicit tokens without BOS")

        def emit(name: str, value: torch.Tensor) -> torch.Tensor:
            for callback in callbacks.get(name, ()):
                replacement = callback(value, hook=self.hook_dict[name])
                if replacement is not None:
                    value = replacement
            return value

        eps = self.cfg.eps
        resid = emit("embed.hook_out", self.embed.W_E[tokens])
        batch, length, d = resid.shape
        mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
        for layer, block in enumerate(self.blocks):
            resid_pre = emit(f"blocks.{layer}.hook_in", resid)
            normed = _layer_norm(resid_pre, block.ln1, eps)
            q = torch.einsum("bpd,hde->bhpe", normed, block.attn.W_Q)
            k = torch.einsum("bpd,hde->bhpe", normed, block.attn.W_K)
            v = torch.einsum("bpd,hde->bhpe", normed, block.attn.W_V)
            scores = torch.einsum("bhqe,bhke->bhqk", q, k) / q.shape[-1] ** 0.5
            scores = scores.masked_fill(~mask, float("-inf"))
            pattern = emit(f"blocks.{layer}.attn.hook_pattern", scores.softmax(dim=-1))
            z = torch.einsum("bhqk,bhke->bhqe", pattern, v)
            per_head = torch.einsum("bhqe,hed->bqhd", z, block.attn.W_O)
            if self.cfg.use_attn_result:
                per_head = emit(f"blocks.{layer}.attn.hook_result", per_head)
            attn_out = emit(f"blocks.{layer}.attn.hook_out", per_head.sum(dim=2) + block.attn.b_O)
            hidden = torch.nn.functional.gelu(_layer_norm(resid_pre, block.ln2, eps) @ block.mlp.W_in + block.mlp.b_in)
            hidden = emit(f"blocks.{layer}.mlp.out.hook_in", hidden)
            mlp_out = emit(f"blocks.{layer}.mlp.hook_out", hidden @ block.mlp.W_out + block.mlp.b_out)
            resid = emit(f"blocks.{layer}.hook_out", resid_pre + attn_out + mlp_out)
        normed = emit("ln_final.hook_out", _layer_norm(resid, self.ln_final, eps))
        return emit("unembed.hook_out", normed @ self.unembed.W_U + self.unembed.b_U)


_PREFIXES = {
    "cardinal": (((1, 2, 3), ()), ((6, 2, 3), ())),
    "quantifier": (((9, 10, 11), ()), ((9, 6, 11), ())),
    "coordinated-adjective": (((1, 2, 3, 6, 7), (8,)), ((9, 2, 3, 6, 7), (8,))),
}


def _frames(cues: dict[str, tuple[int, int]]) -> tuple[pm.Frame, ...]:
    frames = []
    for template, variants in _PREFIXES.items():
        for index, (prefix, suffix) in enumerate(variants, start=1):
            sg, pl = cues[template]
            frames.append(pm.Frame(template, f"{template}-{index}", prefix, suffix, {"sg": sg, "pl": pl}, f"{template} {index} {{cue}}"))
    return tuple(frames)


def _nouns() -> tuple[pm.Noun, ...]:
    return (pm.Noun("n0", Split.DEVELOPMENT, "simple-suffix", (0,), (1,)),
            pm.Noun("n1", Split.DEVELOPMENT, "simple-suffix", (14,), (15,)),
            pm.Noun("n2", Split.DEVELOPMENT, "sibilant-es", (2,), (3,)),
            pm.Noun("n3", Split.DEVELOPMENT, "consonant-y", (10,), (11,)))


def make_context(seed: int = 3) -> "pm.DiscoveryContext":
    """Pick, per template, the cue pair that moves the contrast most, so denominators clear the floors."""
    model = TinyPlural(seed=seed)
    nouns = _nouns()
    cache = pm.PromptCache(model, nouns)
    candidates = [token for token in range(16) if token not in {0, 1, 2, 3, 10, 11, 14, 15}]
    cues: dict[str, tuple[int, int]] = {}
    for template, variants in _PREFIXES.items():
        best, best_value = None, -1.0
        for sg, pl in itertools.permutations(candidates, 2):
            values = []
            for index, (prefix, suffix) in enumerate(variants, start=1):
                frame = pm.Frame(template, f"{template}-{index}", prefix, suffix, {"sg": sg, "pl": pl}, "{cue}")
                a, b = pm.frame_prompts(frame)
                values.extend(cache.c(a)[noun.lexical_key] - cache.c(b)[noun.lexical_key] for noun in nouns)
            mean = sum(values) / len(values)
            if mean > best_value:
                best, best_value = (sg, pl), mean
        if best is None or best_value < 0.6:
            raise RuntimeError(f"{template}: no cue pair moves the contrast enough ({best_value})")
        cues[template] = best
    return pm.DiscoveryContext(model, pm.Weights.from_model(model), None, _frames(cues), nouns, pm.PromptCache(model, nouns))


