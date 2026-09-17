from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Callable

import torch


class FakeHook:
    def __init__(self, name: str) -> None:
        self.name = name


class TinyBridge:
    """Deterministic Bridge-shaped model for instrumentation semantics."""

    def __init__(self, readout: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        # An optional nonlinear readout gives input-dependent logit contrasts for patching tests.
        self.readout = readout
        self.cfg = SimpleNamespace(
            n_layers=2,
            n_heads=2,
            d_model=3,
            d_mlp=4,
            use_attn_result=False,
            positional_embedding_type="rotary",
            parallel_attn_mlp=True,
        )
        hook_names = {"embed.hook_out", "ln_final.hook_out", "unembed.hook_out"}
        for layer in range(self.cfg.n_layers):
            hook_names.update(
                {
                    f"blocks.{layer}.hook_in",
                    f"blocks.{layer}.attn.hook_out",
                    f"blocks.{layer}.attn.hook_result",
                    f"blocks.{layer}.attn.hook_pattern",
                    f"blocks.{layer}.mlp.out.hook_in",
                    f"blocks.{layer}.mlp.hook_out",
                    f"blocks.{layer}.hook_out",
                }
            )
        self.hook_dict = {name: FakeHook(name) for name in hook_names}
        self.training = True
        self.compatibility_mode = False
        self.setter_calls: list[bool] = []
        self.fired_hooks: set[str] = set()
        self.last_grad_enabled: bool | None = None

    def eval(self) -> "TinyBridge":
        self.training = False
        return self

    def set_use_attn_result(self, value: bool) -> None:
        self.setter_calls.append(value)
        self.cfg.use_attn_result = value

    def __call__(
        self, tokens: torch.Tensor, *, prepend_bos: bool = False
    ) -> torch.Tensor:
        return self._forward(tokens, {}, prepend_bos=prepend_bos)

    def run_with_hooks(
        self,
        tokens: torch.Tensor,
        *,
        fwd_hooks: list[tuple[str, Callable]],
        prepend_bos: bool = False,
        **_kwargs: object,
    ) -> torch.Tensor:
        callbacks: dict[str, list[Callable]] = defaultdict(list)
        for name, callback in fwd_hooks:
            callbacks[name].append(callback)
        self.fired_hooks = set()
        return self._forward(tokens, callbacks, prepend_bos=prepend_bos)

    def _forward(
        self,
        tokens: torch.Tensor,
        callbacks: dict[str, list[Callable]],
        *,
        prepend_bos: bool,
    ) -> torch.Tensor:
        if prepend_bos:
            raise ValueError("TinyBridge fixture expects explicit tokens without BOS")
        self.last_grad_enabled = torch.is_grad_enabled()

        def emit(name: str, value: torch.Tensor) -> torch.Tensor:
            for callback in callbacks.get(name, ()):
                self.fired_hooks.add(name)
                replacement = callback(value, hook=self.hook_dict[name])
                if replacement is not None:
                    value = replacement
            return value

        offsets = torch.arange(self.cfg.d_model, dtype=torch.float32)
        resid = emit("embed.hook_out", tokens.to(torch.float32).unsqueeze(-1) + offsets)
        for layer in range(self.cfg.n_layers):
            resid_pre = emit(f"blocks.{layer}.hook_in", resid)
            if self.cfg.use_attn_result:
                per_head = torch.stack(
                    (resid_pre * 0.1, resid_pre * 0.2), dim=2
                )
                per_head = emit(f"blocks.{layer}.attn.hook_result", per_head)
                attention = per_head.sum(dim=2)
            else:
                attention = resid_pre * 0.3
            attention = emit(f"blocks.{layer}.attn.hook_out", attention)
            pattern = torch.full(
                (
                    tokens.shape[0],
                    self.cfg.n_heads,
                    tokens.shape[1],
                    tokens.shape[1],
                ),
                1.0 / tokens.shape[1],
            )
            emit(f"blocks.{layer}.attn.hook_pattern", pattern)
            mlp_post = torch.stack(
                (
                    resid_pre[..., 0],
                    resid_pre[..., 1],
                    resid_pre[..., 2],
                    resid_pre.sum(dim=-1),
                ),
                dim=-1,
            )
            mlp_post = emit(f"blocks.{layer}.mlp.out.hook_in", mlp_post)
            mlp_output = emit(
                f"blocks.{layer}.mlp.hook_out", mlp_post[..., :3] * 0.05
            )
            resid = emit(
                f"blocks.{layer}.hook_out", resid_pre + attention + mlp_output
            )
        normalized = emit("ln_final.hook_out", resid / 10.0)
        if self.readout is not None:
            logits = self.readout(normalized)
        else:
            logits = torch.cat((normalized, normalized[..., :2] + 1.0), dim=-1)
        return emit("unembed.hook_out", logits)


def tiny_tokens() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
