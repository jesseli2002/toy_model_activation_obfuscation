"""CPU (sklearn) / GPU (torch) probe pipeline adapter for
train_adversarial_logreg.py.

Both pipelines expose the same duck-typed interface -- set_max_iter, fit,
get_affine -- so the training loop stays backend-agnostic: it always hands
over torch tensors (whatever device they're already on) and lets each
pipeline decide how to consume them. `build_probe_pipeline` picks the
backend; `"auto"` uses the GPU (torch) pipeline iff CUDA is available.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from torch_logreg import TorchLogisticRegression, TorchStandardScaler


class SklearnProbePipeline:
    """Wraps make_pipeline(StandardScaler(), LogisticRegression(warm_start=True,
    ...)). fit() accepts a torch Tensor on any device and converts to numpy
    internally -- this is train_adversarial_logreg.py's original behavior,
    moved here unchanged."""

    def __init__(self, C: float, max_iter: int):
        self._pipeline: Pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(warm_start=True, C=C, max_iter=max_iter, tol=1e-3),
        )

    def set_max_iter(self, max_iter: int) -> None:
        self._pipeline.named_steps["logisticregression"].max_iter = max_iter

    def fit(self, X, y) -> None:
        X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
        self._pipeline.fit(X_np, y_np)

    def get_affine(
        self, device, dtype=torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The probe's current decision score is affine in the raw (unscaled)
        activation r: s(r) = w_eff . r + b_eff, folding the StandardScaler
        into the LogisticRegression coefficients. mean_/scale_/coef_/intercept_
        are already numpy arrays, so the fold is done in numpy and only the
        two final results cross into torch -- no intermediate tensors, and no
        detach() needed since arrays built from numpy never carry autograd
        history."""
        scaler = self._pipeline.named_steps["standardscaler"]
        logreg = self._pipeline.named_steps["logisticregression"]
        mu, sigma = scaler.mean_, scaler.scale_
        w, b = logreg.coef_[0], logreg.intercept_[0]
        w_eff = w / sigma
        b_eff = b - (w * mu / sigma).sum()
        return (
            torch.as_tensor(w_eff, device=device, dtype=dtype),
            torch.as_tensor(b_eff, device=device, dtype=dtype),
        )


class TorchProbePipeline:
    """TorchStandardScaler + TorchLogisticRegression. fit() keeps X on
    whatever device it's already on -- no numpy round-trip, which is the
    whole point of the GPU backend."""

    def __init__(self, C: float, max_iter: int):
        self._scaler = TorchStandardScaler()
        self._logreg = TorchLogisticRegression(
            C=C, max_iter=max_iter, tol=1e-3, warm_start=True
        )

    def set_max_iter(self, max_iter: int) -> None:
        self._logreg.max_iter = max_iter

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X_scaled = self._scaler.fit_transform(X)
        self._logreg.fit(X_scaled, y)

    def get_affine(
        self, device, dtype=torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self._scaler.mean_.to(device=device, dtype=dtype)
        sigma = self._scaler.scale_.to(device=device, dtype=dtype)
        w = self._logreg.coef_.to(device=device, dtype=dtype)
        b = self._logreg.intercept_.to(device=device, dtype=dtype)
        w_eff = w / sigma
        b_eff = b - (w * mu / sigma).sum()
        # unlike the sklearn pipeline, mu/sigma here are computed from
        # whatever X was passed to fit() -- detach() is the thing enforcing
        # "returns detached tensors" if a caller ever fits on a grad-carrying
        # tensor, not a no-op like it is for the numpy-sourced sklearn arm.
        return w_eff.detach(), b_eff.detach()


def resolve_probe_backend(backend: str, device: str) -> str:
    if backend == "auto":
        return "torch" if device == "cuda" else "sklearn"
    return backend


def build_probe_pipeline(
    C: float, max_iter: int, backend: str
) -> "SklearnProbePipeline | TorchProbePipeline":
    if backend == "torch":
        return TorchProbePipeline(C, max_iter)
    elif backend == "sklearn":
        return SklearnProbePipeline(C, max_iter)
    else:
        raise ValueError(
            f"unknown probe backend {backend!r} -- pass a resolved backend "
            f"('sklearn' or 'torch'), not 'auto'; use resolve_probe_backend() first."
        )


def fit_probe(
    pipeline: "SklearnProbePipeline | TorchProbePipeline",
    X: torch.Tensor,
    y: torch.Tensor,
    max_iter: int,
) -> None:
    """Update `pipeline` in place: refit the scaler (so the model can't dodge
    the probe by uniformly shrinking its own activations) and advance the
    warm-started solver by `max_iter` more iterations."""
    pipeline.set_max_iter(max_iter)
    pipeline.fit(X, y)
