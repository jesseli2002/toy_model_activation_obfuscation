"""Synthetic data for the saturation task.

x ~ U[-3, 3]^num_x,  c ~ U[1, 2] (scalar).  x_full = [x, c].
target y = sat(x, -c, c) = min(c, max(-c, x)), elementwise, shape (B, num_x).

`sample_batch`'s x sampling can optionally be biased toward |x| >= a
threshold instead of plain uniform, via `x_p_outer`/`x_threshold` -- the
kink in sat(x,-c,c) falls somewhere in [C_LOW, C_HIGH], so oversampling near
there gives the model more gradient signal from the nonlinear region. Off by
default (x_p_outer=None), which reproduces the original plain-uniform
behavior exactly.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from config import X_LOW, X_HIGH, C_LOW, C_HIGH
from model import ResidualMLP


def _uniform(
    shape: tuple[int, ...], low: float, high: float, generator, device
) -> Float[Tensor, "..."]:
    return torch.rand(*shape, generator=generator, device=device) * (high - low) + low


def _biased_x(
    shape: tuple[int, ...], threshold: float, p_outer: float, generator, device
) -> Float[Tensor, "..."]:
    """Mixture over [X_LOW, X_HIGH] (assumed symmetric): with probability
    p_outer, sample |x| in [threshold, X_HIGH] (sign chosen uniformly);
    otherwise sample uniformly in [-threshold, threshold]."""
    is_outer = torch.rand(*shape, generator=generator, device=device) < p_outer
    sign = (
        2.0 * (torch.rand(*shape, generator=generator, device=device) < 0.5).float()
        - 1.0
    )
    outer = sign * _uniform(shape, threshold, X_HIGH, generator, device)
    inner = _uniform(shape, -threshold, threshold, generator, device)
    return torch.where(is_outer, outer, inner)


def sample_batch(
    batch: int,
    num_x: int,
    generator=None,
    device="cpu",
    x_p_outer: float | None = None,
    x_threshold: float = 1.0,
) -> tuple[Float[Tensor, "batch num_x_plus_1"], Float[Tensor, "batch num_x"]]:
    """Return (x_full, y): x_full is (B, num_x+1), y is (B, num_x).

    x_p_outer: if given (0..1), oversample |x| >= x_threshold with this
    probability per coordinate instead of plain U[X_LOW, X_HIGH]. None
    (default) = unbiased, unchanged behavior.
    """
    if x_p_outer is None:
        x = _uniform((batch, num_x), X_LOW, X_HIGH, generator, device)
    else:
        x = _biased_x((batch, num_x), x_threshold, x_p_outer, generator, device)
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


@torch.no_grad()
def eval_max_err(
    model: ResidualMLP,
    generator: torch.Generator,
    device: str = "cpu",
    n: int = 100_000,
    batch: int = 20_000,
) -> float:
    worst = 0.0
    done = 0
    while done < n:
        b = min(batch, n - done)
        x_full, y = sample_batch(b, model.num_x, generator=generator, device=device)
        pred: Float[Tensor, "b num_x"] = model.task_output(x_full)
        worst = max(worst, (pred - y).abs().max().item())
        done += b
    return worst
