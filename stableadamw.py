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
Warner), reduced to the plain fp32 single-tensor path -- the foreach/Triton
backends, Kahan summation, gradient release, and optimizer accumulation are
deliberately not carried over.
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

    def _init_state(self, state: dict[str, Any], param: Tensor):
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )

    def load_state_dict(self, state_dict: dict[str, Any]):
        # Torch rebuilds each param group from the saved one, so a state dict
        # written by an implementation without a `d` key silently drops the
        # threshold -- surface that here rather than as a KeyError mid-step.
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            if "d" not in group:
                raise ValueError(
                    "optimizer state predates the tunable update-clipping "
                    "threshold and cannot be resumed. Such a checkpoint was "
                    "trained at the hard-coded d=1.0; fork from it instead of "
                    "resuming, which rebuilds the optimizer from config."
                )

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            beta1_hat = _debias_beta(group["beta1"], group["step"])
            beta2_hat = _debias_beta(group["beta2"], group["step"])
            eps, eps_sq = group["eps"], group["eps"] ** 2

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                self._init_state(state, param)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                exp_avg.lerp_(grad, weight=1 - beta1_hat)
                exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=1 - beta2_hat)

                # Per-tensor RMS of the un-momentumed update, and the clipped lr
                rms = grad.pow(2).div_(exp_avg_sq.clamp(min=eps_sq)).mean().sqrt()
                lr = group["lr"] / max(1.0, rms.item() / group["d"])

                if group["weight_decay"] != 0:
                    param.mul_(1 - lr * group["weight_decay"])
                param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr)

        return loss
