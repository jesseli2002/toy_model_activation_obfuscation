"""Synthetic data for the saturation task.

x ~ U[-3, 3]^num_x,  c ~ U[1, 2] (scalar).  x_full = [x, c].
target y = sat(x, -c, c) = min(c, max(-c, x)), elementwise, shape (B, num_x).
"""

import torch
from jaxtyping import Float
from torch import Tensor

from config import X_LOW, X_HIGH, C_LOW, C_HIGH


def _uniform(
    shape: tuple[int, ...], low: float, high: float, generator, device
) -> Float[Tensor, "..."]:
    return torch.rand(*shape, generator=generator, device=device) * (high - low) + low


def sample_batch(
    batch: int, num_x: int, generator=None, device="cpu"
) -> tuple[Float[Tensor, "batch num_x_plus_1"], Float[Tensor, "batch num_x"]]:
    """Return (x_full, y): x_full is (B, num_x+1), y is (B, num_x)."""
    x = _uniform((batch, num_x), X_LOW, X_HIGH, generator, device)
    c = _uniform((batch, 1), C_LOW, C_HIGH, generator, device)
    x_full = torch.cat([x, c], dim=1)
    y = torch.minimum(torch.maximum(x, -c), c)
    return x_full, y


def sample_fixed_c(
    batch: int, num_x: int, c_value: float, generator=None, device="cpu"
) -> tuple[Float[Tensor, "batch num_x_plus_1"], Float[Tensor, "batch num_x"]]:
    """Same as sample_batch but with c pinned to c_value (for probe datasets)."""
    x = _uniform((batch, num_x), X_LOW, X_HIGH, generator, device)
    c = torch.full((batch, 1), float(c_value), device=device)
    x_full = torch.cat([x, c], dim=1)
    y = torch.minimum(torch.maximum(x, -c), c)
    return x_full, y
