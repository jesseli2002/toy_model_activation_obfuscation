"""StableAdamW with a tunable update-clipping threshold.

AdamW plus Adafactor's update clipping (Shazeer & Stern 2018, "Adafactor:
Adaptive Learning Rates with Sublinear Memory Cost", section 6), applied
per-tensor as a learning-rate scaling rather than to the update itself:

    RMS_t = sqrt(E[g_t^2 / max(v_t, eps^2)])
    lr_t  = lr / max(1, RMS_t / d)

Vendored (rather than used from the `optimi` package) solely to expose the
clipping threshold `d`, which the paper sweeps but every implementation
downstream of Wortsman et al. (2023) hard-codes to 1. `d=1` reproduces the
upstream behaviour exactly; smaller `d` clips more aggressively, damping
steps only on tensors whose RMS exceeds the threshold and leaving quieter
steps untouched.

Derived from optimi's StableAdamW (MIT, Copyright (c) 2023-present Benjamin
Warner), reduced to the plain fp32 path -- Kahan summation, gradient release,
the Triton backend, and optimizer accumulation are deliberately not carried
over.

The step is written multi-tensor (`torch._foreach_*`) throughout, and the
clipped per-tensor learning rate is kept on-device as a tensor rather than
read back with `.item()`. On a launch-latency-bound workload the read-back
cost dominates: it forces one GPU->CPU sync *per parameter tensor* per step.
See PR #146.
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor

__all__ = ["StableAdamW"]


def _debias_beta(beta: float, step: int) -> float:
    """Fold the Adam bias correction into beta.

    Equivalent to `beta * (1 - beta**(step-1)) / (1 - beta**step)`.
    """
    return (beta**step - beta) / (beta**step - 1)


class StableAdamW(torch.optim.Optimizer):
    """AdamW with per-tensor update clipping.

    Args:
        params: Iterable of parameters to optimize, or dicts defining parameter groups
        lr: Learning rate
        betas: Coefficients for the gradient and squared-gradient moving averages
        weight_decay: Decoupled weight decay coefficient
        eps: Added to the denominator to improve numerical stability
        d: Update-clipping threshold. A tensor's step is scaled down only once its
            RMS exceeds `d`, so lowering `d` tightens clipping.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 1e-2,
        eps: float = 1e-6,
        d: float = 1.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr=}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]=}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]=}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps=}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay=}")
        if not 0.0 < d:
            raise ValueError(f"Invalid update clipping threshold: {d=}")

        super().__init__(
            params,
            dict(
                lr=lr,
                beta1=betas[0],
                beta2=betas[1],
                eps=eps,
                weight_decay=weight_decay,
                d=d,
                step=0,
            ),
        )
        # Kept off the param_groups so it never reaches state_dict().
        self._numel_cache: dict[int, tuple[tuple[int, ...], Tensor]] = {}

    def _numels(self, group_index: int, params: list[Tensor]) -> Tensor:
        """Per-tensor element counts as one device tensor, for turning the
        summed RMS ratios into means. Cached on the (validated) shape tuple so
        the host->device copy happens once rather than every step."""
        key = tuple(p.numel() for p in params)
        cached = self._numel_cache.get(group_index)
        if cached is None or cached[0] != key:
            cached = (
                key,
                torch.tensor(key, device=params[0].device, dtype=params[0].dtype),
            )
            self._numel_cache[group_index] = cached
        return cached[1]

    def _init_state(self, state: dict[str, Any], param: Tensor):
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_index, group in enumerate(self.param_groups):
            group["step"] += 1
            beta1_hat = _debias_beta(group["beta1"], group["step"])
            beta2_hat = _debias_beta(group["beta2"], group["step"])
            eps, eps_sq = group["eps"], group["eps"] ** 2
            weight_decay = group["weight_decay"]

            params, grads, exp_avgs, exp_avg_sqs = [], [], [], []
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                self._init_state(state, param)
                params.append(param)
                grads.append(param.grad)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
            if not params:
                continue

            torch._foreach_lerp_(exp_avgs, grads, 1 - beta1_hat)
            torch._foreach_mul_(exp_avg_sqs, beta2_hat)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2_hat)

            # Per-tensor RMS of the un-momentumed update. The ratio is
            # non-negative elementwise, so its L1 norm is its sum and the mean
            # is that over numel -- one fused reduction for every tensor at
            # once, where a per-tensor .mean() would be a launch each.
            ratio = torch._foreach_div(
                torch._foreach_mul(grads, grads),
                torch._foreach_clamp_min(exp_avg_sqs, eps_sq),
            )
            rms = (
                torch.stack(torch._foreach_norm(ratio, 1))
                / self._numels(group_index, params)
            ).sqrt()
            # The clipped lr, one entry per tensor and never leaving the
            # device. Computed in float64 and rounded back at each use site,
            # matching the reference implementation's Python-float arithmetic.
            lr = group["lr"] / torch.clamp(rms.double() / group["d"], min=1.0)
            dtype = params[0].dtype

            if weight_decay != 0:
                torch._foreach_mul_(
                    params, list((1 - lr * weight_decay).to(dtype).unbind())
                )

            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_add_(denom, eps)
            update = torch._foreach_div(exp_avgs, denom)
            torch._foreach_mul_(update, list((-lr).to(dtype).unbind()))
            torch._foreach_add_(params, update)

        return loss
