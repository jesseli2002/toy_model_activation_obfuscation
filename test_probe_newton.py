"""pytest unit tests for probe_newton.py: NewtonLogisticRegression solves the
same objective as TorchLogisticRegression (and sklearn), and
NewtonProbePipeline is interface-compatible with TorchProbePipeline. Runs
entirely on CPU tensors -- same code path GPU would use, just a different
device."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from probe_backend import TorchProbePipeline
from probe_newton import NewtonLogisticRegression, NewtonProbePipeline

torch.manual_seed(0)
DTYPE = torch.float64


def make_binary_data(n=400, d=8, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    z = X @ rng.normal(size=d) + noise * rng.normal(size=n)
    y = z >= 0
    assert y.any() and (~y).any()
    return X, y


def to_torch(X, y):
    return torch.as_tensor(X, dtype=DTYPE), torch.as_tensor(y, dtype=torch.bool)


@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_matches_sklearn(C):
    X, y = make_binary_data()
    sk = LogisticRegression(C=C, tol=1e-10, max_iter=5000).fit(X, y)
    Xt, yt = to_torch(X, y)
    m = NewtonLogisticRegression(C=C, steps=25, warm_start=False).fit(Xt, yt)
    np.testing.assert_allclose(m.coef_.numpy(), sk.coef_[0], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        m.intercept_.numpy(), sk.intercept_[0], rtol=1e-4, atol=1e-6
    )


def test_intercept_is_unpenalized():
    """A constant offset in y's base rate must be absorbed by the intercept,
    not shrunk toward zero the way a penalized one would be."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 4))
    y = rng.random(2000) < 0.9  # heavily imbalanced -> large true intercept
    Xt, yt = to_torch(X, y)
    m = NewtonLogisticRegression(C=1.0, steps=30, warm_start=False).fit(Xt, yt)
    assert m.intercept_.item() > 1.5


def test_converges_from_cold_start_without_diverging():
    """Undamped Newton overshoots badly on near-separable data; the vectorized
    line search is what makes a cold start safe."""
    X, y = make_binary_data(noise=0.01)  # near-separable
    Xt, yt = to_torch(X, y)
    m = NewtonLogisticRegression(C=1.0, steps=30, warm_start=False).fit(Xt, yt)
    assert torch.isfinite(m.coef_).all()
    assert torch.isfinite(m.intercept_)
    assert (m.predict(Xt) == yt).float().mean() > 0.95


def test_warm_start_reaches_same_optimum_in_fewer_steps():
    X, y = make_binary_data()
    Xt, yt = to_torch(X, y)
    cold = NewtonLogisticRegression(C=1.0, steps=30, warm_start=False).fit(Xt, yt)
    warm = NewtonLogisticRegression(C=1.0, steps=30)
    warm.fit(Xt, yt)
    warm.steps = 2  # a converged warm start needs almost no further work
    warm.fit(Xt, yt)
    torch.testing.assert_close(warm.coef_, cold.coef_, rtol=1e-5, atol=1e-7)


def test_pipeline_affine_matches_torch_pipeline():
    """The pipeline's whole output contract is the folded affine handed to the
    training loop, so that is what must agree with the existing backend."""
    X, y = make_binary_data(n=600, d=6)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.bool)

    ref = TorchProbePipeline(C=1.0, max_iter=500)
    ref.fit(Xt, yt)
    w_ref, b_ref = ref.get_affine("cpu")

    new = NewtonProbePipeline(C=1.0, steps=25)
    new.fit(Xt, yt)
    w_new, b_new = new.get_affine("cpu")

    torch.testing.assert_close(w_new, w_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(b_new, b_ref, rtol=2e-3, atol=2e-3)


def test_get_affine_returns_detached():
    X, y = make_binary_data(n=200, d=5)
    Xt = torch.as_tensor(X, dtype=torch.float32).requires_grad_(True)
    yt = torch.as_tensor(y, dtype=torch.bool)
    p = NewtonProbePipeline(C=1.0, steps=10)
    p.fit(Xt, yt)
    w, b = p.get_affine("cpu")
    assert not w.requires_grad and not b.requires_grad
