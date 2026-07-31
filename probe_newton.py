"""Newton/IRLS probe pipeline -- drop-in replacement for TorchProbePipeline.

Perf-debug scaffolding (see PERF_HANDOFF.md). Same duck-typed interface as
probe_backend.TorchProbePipeline (set_max_iter / fit / get_affine), so
probe_backend.fit_probe works on it unchanged.

Why Newton rather than L-BFGS here: the probe problem is d~64 parameters, so
an explicit (d+1)x(d+1) Hessian factorization is trivial GPU work, and unlike
torch.optim.LBFGS it needs neither a curvature history (whose two-loop
recursion issues O(history) tiny kernels per iteration) nor a strong-Wolfe
line search (whose `float(closure())` forces a device->host sync per trial
step). The step size is instead chosen by evaluating every candidate in one
batched matmul and taking an argmin, so a fit is a fixed, sync-free sequence
of large kernels.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from torch_logreg import TorchStandardScaler

# Candidate step sizes for the vectorized line search. Width is nearly free --
# one extra column of an [n, K] matmul, not an extra device->host round trip.
ALPHAS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)


class NewtonLogisticRegression:
    """L2-regularized binary logistic regression by damped Newton / IRLS.

    Matches TorchLogisticRegression's objective exactly (see its docstring):
        J(w, b) = 0.5 * w.w + C * sum_i softplus(-y_i * (X_i.w + b))
    with the intercept unpenalized.

    `steps` is a fixed iteration count, not a tolerance: a fixed instruction
    count is the point, since any convergence test would reintroduce the
    host sync this solver exists to avoid. Warm-started across fits by
    default, which is what makes a small `steps` sufficient in steady state.
    """

    def __init__(
        self,
        C: float = 1.0,
        steps: int = 3,
        warm_start: bool = True,
        step_sizes=ALPHAS,
        ridge: float = 1e-6,
    ):
        self.C = C
        self.steps = steps
        self.warm_start = warm_start
        self.step_sizes = step_sizes
        self.ridge = ridge
        self.coef_: Float[Tensor, " d"] | None = None
        self.intercept_: Float[Tensor, ""] | None = None

    def fit(
        self, X: Float[Tensor, "n d"], y: Bool[Tensor, " n"]
    ) -> "NewtonLogisticRegression":
        n, d = X.shape
        if self.warm_start and self.coef_ is not None and self.coef_.shape[0] == d:
            w, b = self.coef_.clone(), self.intercept_.clone()
        else:
            w = torch.zeros(d, device=X.device, dtype=X.dtype)
            b = torch.zeros((), device=X.device, dtype=X.dtype)
        t = y.to(X.dtype)
        y_pm1 = torch.where(y, 1.0, -1.0).to(X.dtype)
        alphas = torch.tensor(self.step_sizes, device=X.device, dtype=X.dtype)
        eye = torch.eye(d + 1, device=X.device, dtype=X.dtype)

        for _ in range(self.steps):
            p = torch.sigmoid(X @ w + b)
            r = p - t
            s = p * (1 - p)
            g = torch.cat([w + self.C * (X.T @ r), (self.C * r.sum()).reshape(1)])
            Xs = X * s.unsqueeze(1)
            H = torch.empty((d + 1, d + 1), device=X.device, dtype=X.dtype)
            H[:d, :d] = self.C * (X.T @ Xs) + eye[:d, :d]
            H[:d, d] = self.C * Xs.sum(dim=0)
            H[d, :d] = H[:d, d]
            H[d, d] = self.C * s.sum()
            delta = torch.linalg.solve(H + self.ridge * eye, g)
            # Evaluate every candidate step size at once; argmin picks the
            # best without ever reading a value back to the host.
            Wc = w.unsqueeze(1) - alphas.unsqueeze(0) * delta[:d].unsqueeze(1)
            Bc = b - alphas * delta[d]
            Jc = 0.5 * (Wc * Wc).sum(dim=0) + self.C * F.softplus(
                -y_pm1.unsqueeze(1) * (X @ Wc + Bc.unsqueeze(0))
            ).sum(dim=0)
            k = torch.argmin(Jc)
            w, b = Wc[:, k], Bc[k]

        self.coef_, self.intercept_ = w, b
        return self

    def decision_function(self, X: Float[Tensor, "n d"]) -> Float[Tensor, " n"]:
        return X @ self.coef_ + self.intercept_

    def predict(self, X: Float[Tensor, "n d"]) -> Bool[Tensor, " n"]:
        return self.decision_function(X) >= 0


class NewtonProbePipeline:
    """TorchStandardScaler + NewtonLogisticRegression, interface-compatible
    with probe_backend.TorchProbePipeline."""

    def __init__(self, C: float, steps: int = 3):
        self._scaler = TorchStandardScaler()
        self._logreg = NewtonLogisticRegression(C=C, steps=steps)

    def set_max_iter(self, max_iter: int) -> None:
        # The training loop calls this with PROBE_STEP_MAX_ITER; Newton's step
        # budget is set at construction and deliberately not driven by it.
        pass

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._logreg.fit(self._scaler.fit_transform(X), y)

    def get_affine(
        self, device, dtype=torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self._scaler.mean_.to(device=device, dtype=dtype)
        sigma = self._scaler.scale_.to(device=device, dtype=dtype)
        w = self._logreg.coef_.to(device=device, dtype=dtype)
        b = self._logreg.intercept_.to(device=device, dtype=dtype)
        return (w / sigma).detach(), (b - (w * mu / sigma).sum()).detach()
