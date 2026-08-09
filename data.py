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
    # Accumulated on-device and read back once: a per-batch .item() would
    # sync the host against the GPU on every chunk, which at a short
    # --log-interval costs more than the eval itself. Chunking (and so the
    # generator's draw sequence) is unchanged.
    worst = torch.zeros((), device=device)
    done = 0
    while done < n:
        b = min(batch, n - done)
        x_full, y = sample_batch(b, model.num_x, generator=generator, device=device)
        pred: Float[Tensor, "b num_x"] = model.task_output(x_full)
        worst = torch.maximum(worst, (pred - y).abs().max())
        done += b
    return worst.item()


@torch.no_grad()
def eval_task_loss(
    model: ResidualMLP,
    generator: torch.Generator,
    device: str = "cpu",
    n: int = 100_000,
    batch: int = 20_000,
    x_p_outer: float | None = None,
    x_threshold: float = 1.0,
    noise: float = 0.0,
) -> float:
    """Mean-squared task error over `n` fresh examples, matching the training
    loop's own `l_task` formula (see train_adversarial_logreg.py's
    forward_loss) -- deliberately excludes any probe/adversarial penalty, so
    this is purely "how well does the model solve the task", independent of
    lam. Pass a checkpoint's own `x_p_outer`/`x_threshold`/`resid_noise_std`
    (see `probe_lib.resolve_adv_config`) to match its training distribution."""
    total = torch.zeros((), device=device)
    done = 0
    while done < n:
        b = min(batch, n - done)
        x_full, y = sample_batch(
            b,
            model.num_x,
            generator=generator,
            device=device,
            x_p_outer=x_p_outer,
            x_threshold=x_threshold,
        )
        pred: Float[Tensor, "b num_x"] = model.task_output(
            x_full, noise=noise, generator=generator
        )
        total += ((pred - y) ** 2).sum()
        done += b
    return (total / (n * model.num_x)).item()


@torch.no_grad()
def eval_n_hot_loss(
    model: ResidualMLP,
    generator: torch.Generator,
    device: str = "cpu",
    n: int = 100_000,
    batch: int = 20_000,
    n_hot: int = 1,
    noise: float = 0.0,
) -> float:
    """Mean-squared task error with only `n_hot` input coordinates nonzero per
    example (the rest held at 0) -- the same input construction used by
    adversarial_report.plot_learned_curves for n_hot=1, but scored as a loss
    instead of plotted. Every training example has all `num_x` coordinates
    simultaneously nonzero, so this checks generalization to an OOD corner of
    input space, with larger `n_hot` interpolating toward the training
    distribution's density of nonzero coordinates. Scored only on the active
    coordinates' outputs (the passive coordinates' target is trivially 0,
    which would just dilute the number), so this is directly comparable to
    eval_task_loss's per-coordinate MSE."""
    total = torch.zeros((), device=device)
    done = 0
    while done < n:
        b = min(batch, n - done)
        # n_hot distinct coordinates per example: argsort random keys, same
        # trick as sampling without replacement per-row.
        active_idx = torch.argsort(
            torch.rand((b, model.num_x), generator=generator, device=device), dim=1
        )[:, :n_hot]
        x = torch.zeros((b, model.num_x), device=device)
        active = _uniform((b, n_hot), X_LOW, X_HIGH, generator, device)
        x.scatter_(1, active_idx, active)
        c = _uniform((b, 1), C_LOW, C_HIGH, generator, device)
        x_full = torch.cat([x, c], dim=1)
        y = torch.minimum(torch.maximum(active, -c), c)
        pred: Float[Tensor, "b num_x"] = model.task_output(
            x_full, noise=noise, generator=generator
        )
        pred_active = torch.gather(pred, 1, active_idx)
        total += ((pred_active - y) ** 2).sum()
        done += b
    return (total / (n * n_hot)).item()
