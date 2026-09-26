"""The reviewer's own control stream (identical to the functions in r2a_own_geometry.py), importable without side
effects: SHA-256 counter mode -> 53-bit uniforms in (0, 1] -> Box-Muller in float64 -> projection off E_hat and t_hat
twice -> unit. No library random-number generator is used anywhere."""
import hashlib
import math

import torch


def uniforms(tag: str, n: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < n:
        digest = hashlib.sha256(f"{tag}|{counter}".encode("utf-8")).digest()
        for k in range(4):
            word = int.from_bytes(digest[8 * k:8 * k + 8], byteorder="big", signed=False) >> 11
            values.append((word + 1) / 2 ** 53)
        counter += 1
    return values[:n]


def gaussians(tag: str, n: int) -> list[float]:
    u = uniforms(tag, n + (n % 2))
    z: list[float] = []
    for i in range(0, len(u), 2):
        r = math.sqrt(-2.0 * math.log(u[i]))
        phi = 2.0 * math.pi * u[i + 1]
        z.append(r * math.cos(phi))
        z.append(r * math.sin(phi))
    return z[:n]


def orthogonalize(v: torch.Tensor, basis) -> torch.Tensor:
    for _ in range(2):
        for b in basis:
            v = v - torch.dot(v, b) * b
    return v


def control(token_id: int, j: int, E_hat: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
    g = torch.tensor(gaussians(f"025|control|{token_id}|{j}", 512), dtype=torch.float64)
    u = orthogonalize(g, (E_hat, t_hat))
    return u / torch.linalg.vector_norm(u)
