"""GPU/CPU-agnostic L2-regularized logistic regression, sklearn-shaped.

Standalone and reusable -- not specific to any one probe. Operates on whatever
device the input tensors already live on, so the same code path is exercised
whether X is on CPU (used for unit tests / non-CUDA machines) or CUDA.

Matches sklearn's `LogisticRegression(penalty="l2")` primal objective (see the
sklearn User Guide's binary logistic-regression formula), y in {-1, +1}:

    J(w, b) = 0.5 * w.w + C * sum_i softplus(-y_i * (X_i.w + b))

The intercept is not L2-penalized, matching sklearn. Solved with
`torch.optim.LBFGS` (strong-Wolfe line search) instead of scipy's
L-BFGS-B -- numerically close but not bit-identical to sklearn's solver.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor


class TorchStandardScaler:
    """Mirrors sklearn.preprocessing.StandardScaler: zero-mean, unit-variance
    per-column scaling."""

    def __init__(self):
        self.mean_: Float[Tensor, " d"] | None = None
        self.scale_: Float[Tensor, " d"] | None = None

    def fit(self, X: Float[Tensor, "n d"]) -> "TorchStandardScaler":
        self.mean_ = X.mean(dim=0)
        std = X.std(dim=0, unbiased=False)
        # sklearn substitutes 1.0 for zero-variance columns rather than
        # dividing by zero.
        self.scale_ = torch.where(std == 0, torch.ones_like(std), std)
        return self

    def transform(self, X: Float[Tensor, "n d"]) -> Float[Tensor, "n d"]:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: Float[Tensor, "n d"]) -> Float[Tensor, "n d"]:
        return self.fit(X).transform(X)


class TorchLogisticRegression:
    """L2-regularized binary logistic regression, solved with torch.optim.LBFGS.

    `C` is sklearn's inverse regularization strength (smaller = more
    regularization). `warm_start=True` persists both the coefficients *and*
    the LBFGS optimizer's curvature history across `.fit()` calls (a
    deliberate improvement over sklearn's warm_start, which -- since scipy's
    L-BFGS-B doesn't expose solver memory across separate calls -- only ever
    warm-starts the coefficient values).
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        warm_start: bool = False,
    ):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.coef_: Float[Tensor, " d"] | None = None
        self.intercept_: Float[Tensor, ""] | None = None
        self._optimizer: torch.optim.LBFGS | None = None

    def _init_params(self, d: int, device, dtype) -> tuple[Tensor, Tensor]:
        if self.warm_start and self.coef_ is not None and self.coef_.shape[0] == d:
            w = self.coef_.detach().clone().to(device=device, dtype=dtype)
            b = self.intercept_.detach().clone().to(device=device, dtype=dtype)
        else:
            w = torch.zeros(d, device=device, dtype=dtype)
            b = torch.zeros((), device=device, dtype=dtype)
        w.requires_grad_(True)
        b.requires_grad_(True)
        return w, b

    def fit(
        self, X: Float[Tensor, "n d"], y: Bool[Tensor, " n"]
    ) -> "TorchLogisticRegression":
        _, d = X.shape
        y_pm1 = torch.where(
            y,
            torch.ones((), dtype=X.dtype, device=X.device),
            -torch.ones((), dtype=X.dtype, device=X.device),
        )

        reuse_optimizer = (
            self.warm_start
            and self._optimizer is not None
            and self.coef_ is not None
            and self.coef_.shape[0] == d
        )
        if reuse_optimizer:
            w, b = self._optimizer.param_groups[0]["params"]
            optimizer = self._optimizer
            optimizer.param_groups[0]["max_iter"] = self.max_iter
            optimizer.param_groups[0]["tolerance_grad"] = self.tol
        else:
            w, b = self._init_params(d, X.device, X.dtype)
            optimizer = torch.optim.LBFGS(
                [w, b],
                max_iter=self.max_iter,
                tolerance_grad=self.tol,
                tolerance_change=self.tol * 1e-2,
                line_search_fn="strong_wolfe",
            )
            if self.warm_start:
                self._optimizer = optimizer

        def closure():
            optimizer.zero_grad()
            z = X @ w + b
            loss = 0.5 * (w @ w) + self.C * F.softplus(-y_pm1 * z).sum()
            loss.backward()
            return loss

        optimizer.step(closure)

        self.coef_ = w.detach()
        self.intercept_ = b.detach()
        return self

    def decision_function(self, X: Float[Tensor, "n d"]) -> Float[Tensor, " n"]:
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: Float[Tensor, "n d"]) -> Float[Tensor, " n"]:
        return torch.sigmoid(self.decision_function(X))

    def predict(self, X: Float[Tensor, "n d"]) -> Bool[Tensor, " n"]:
        return self.decision_function(X) >= 0
