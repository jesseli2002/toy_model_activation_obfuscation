"""pytest unit tests for torch_logreg.py: TorchStandardScaler and
TorchLogisticRegression, checked against sklearn on synthetic binary
classification data. Runs entirely on CPU tensors -- same code path GPU
would use, just a different device."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from torch_logreg import TorchLogisticRegression, TorchStandardScaler

torch.manual_seed(0)
DTYPE = torch.float64  # match sklearn's double-precision internals


def make_binary_data(n=400, d=8, seed=0, noise=0.5, separable=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    z = X @ w_true
    if not separable:
        z = z + noise * rng.normal(size=n)
    y = z >= 0
    assert y.any() and (~y).any()
    return X, y


def to_torch(X, y):
    return (
        torch.as_tensor(X, dtype=DTYPE),
        torch.as_tensor(y, dtype=torch.bool),
    )


class TestTorchStandardScaler:
    def test_matches_sklearn(self):
        X, _ = make_binary_data()
        sk = StandardScaler().fit(X)
        Xt = torch.as_tensor(X, dtype=DTYPE)
        tv = TorchStandardScaler().fit(Xt)
        np.testing.assert_allclose(tv.mean_.numpy(), sk.mean_, atol=1e-10)
        np.testing.assert_allclose(tv.scale_.numpy(), sk.scale_, atol=1e-10)
        Xt_scaled = tv.transform(Xt).numpy()
        Xsk_scaled = sk.transform(X)
        np.testing.assert_allclose(Xt_scaled, Xsk_scaled, atol=1e-8)

    def test_zero_variance_column_no_div_by_zero(self):
        X = np.zeros((20, 3))
        X[:, 0] = 1.0  # constant column
        X[:, 1:] = np.random.default_rng(1).normal(size=(20, 2))
        Xt = torch.as_tensor(X, dtype=DTYPE)
        scaler = TorchStandardScaler().fit(Xt)
        assert torch.isfinite(scaler.transform(Xt)).all()
        assert scaler.scale_[0].item() == 1.0


@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_logreg_matches_sklearn(C):
    X, y = make_binary_data(n=500, d=10, seed=42)
    Xt, yt = to_torch(X, y)

    sk = LogisticRegression(C=C, max_iter=2000, tol=1e-10).fit(X, y)
    tv = TorchLogisticRegression(C=C, max_iter=2000, tol=1e-10).fit(Xt, yt)

    w_sk, w_tv = sk.coef_[0], tv.coef_.numpy()
    cos_sim = np.dot(w_sk, w_tv) / (np.linalg.norm(w_sk) * np.linalg.norm(w_tv))
    assert cos_sim > 0.999
    np.testing.assert_allclose(np.linalg.norm(w_tv), np.linalg.norm(w_sk), rtol=0.02)
    assert abs(tv.intercept_.item() - sk.intercept_[0]) < 0.05

    pred_sk = sk.predict(X)
    pred_tv = tv.predict(Xt).numpy()
    assert (pred_sk == pred_tv).mean() > 0.99

    proba_sk = sk.predict_proba(X)[:, 1]
    proba_tv = tv.predict_proba(Xt).numpy()
    np.testing.assert_allclose(proba_tv, proba_sk, atol=0.02)


def test_decision_function_predict_proba_consistency():
    X, y = make_binary_data(n=200, d=5, seed=7)
    Xt, yt = to_torch(X, y)
    tv = TorchLogisticRegression(C=1.0, max_iter=500).fit(Xt, yt)

    z = tv.decision_function(Xt)
    proba = tv.predict_proba(Xt)
    pred = tv.predict(Xt)

    np.testing.assert_allclose(proba.numpy(), torch.sigmoid(z).numpy(), atol=1e-12)
    np.testing.assert_array_equal(pred.numpy(), (z >= 0).numpy())


def test_warm_start_monotonic_loss_decrease():
    X, y = make_binary_data(n=500, d=12, seed=3)
    Xt, yt = to_torch(X, y)

    def full_loss(model):
        z = model.decision_function(Xt)
        y_pm1 = torch.where(yt, 1.0, -1.0).to(DTYPE)
        return (
            0.5 * (model.coef_ @ model.coef_)
            + model.C * torch.nn.functional.softplus(-y_pm1 * z).sum()
        ).item()

    model = TorchLogisticRegression(C=1.0, max_iter=3, tol=1e-12, warm_start=True)
    losses = []
    for _ in range(15):
        model.fit(Xt, yt)
        losses.append(full_loss(model))

    optimal = TorchLogisticRegression(C=1.0, max_iter=2000, tol=1e-12).fit(Xt, yt)
    optimal_loss = full_loss(optimal)

    # the persisted LBFGS curvature history should let a handful of 3-iter
    # warm-started steps reach (near) the same optimum as one big fit.
    assert losses[-1] < losses[0] * 0.55
    assert losses[-1] < optimal_loss * 1.01
    increases = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
    assert sum(d for d in increases if d > 0) < 0.1 * abs(losses[0] - losses[-1])


def test_warm_start_converges_close_to_full_fit():
    X, y = make_binary_data(n=500, d=12, seed=3)
    Xt, yt = to_torch(X, y)

    warm = TorchLogisticRegression(C=1.0, max_iter=10, tol=1e-12, warm_start=True)
    for _ in range(30):
        warm.fit(Xt, yt)

    full = TorchLogisticRegression(C=1.0, max_iter=2000, tol=1e-12).fit(Xt, yt)

    cos_sim = torch.nn.functional.cosine_similarity(
        warm.coef_.unsqueeze(0), full.coef_.unsqueeze(0)
    ).item()
    assert cos_sim > 0.99


def test_warm_start_reset_interval_creates_fresh_optimizer():
    X, y = make_binary_data(n=200, d=5, seed=11)
    Xt, yt = to_torch(X, y)
    model = TorchLogisticRegression(
        C=1.0, max_iter=3, warm_start=True, warm_start_reset_interval=4
    )
    opt_ids = []
    for _ in range(9):
        model.fit(Xt, yt)
        opt_ids.append(id(model._optimizer))

    # calls 1-3 reuse the same optimizer; call 4 is a periodic reset (new
    # object); calls 5-7 reuse that one; call 8 resets again; call 9 reuses.
    assert opt_ids[0] == opt_ids[1] == opt_ids[2]
    assert opt_ids[3] != opt_ids[2]
    assert opt_ids[3] == opt_ids[4] == opt_ids[5] == opt_ids[6]
    assert opt_ids[7] != opt_ids[6]
    assert opt_ids[7] == opt_ids[8]


def test_overflow_triggers_reset_and_retry(monkeypatch):
    """Regression test for the crash in crash_log/1.txt: a warm-started
    LBFGS instance's curvature history occasionally blows up its search
    direction until the line search overflows float32
    (`RuntimeError: value cannot be converted to type float without
    overflow`). fit() should catch that specific failure, drop the
    corrupted optimizer state, and retry once instead of propagating."""
    X, y = make_binary_data(n=200, d=5, seed=13)
    Xt, yt = to_torch(X, y)
    model = TorchLogisticRegression(C=1.0, max_iter=5, warm_start=True)
    model.fit(Xt, yt)  # establish a real, non-None optimizer + coef_
    stale_optimizer = model._optimizer

    orig_step = torch.optim.LBFGS.step
    calls = {"n": 0}

    def flaky_step(self, closure):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(
                "value cannot be converted to type float without overflow"
            )
        return orig_step(self, closure)

    monkeypatch.setattr(torch.optim.LBFGS, "step", flaky_step)
    model.fit(Xt, yt)

    assert calls["n"] == 2
    assert model._optimizer is not stale_optimizer
    assert torch.isfinite(model.coef_).all()
    assert torch.isfinite(model.intercept_).all()


def test_non_overflow_runtime_error_is_not_swallowed(monkeypatch):
    X, y = make_binary_data(n=200, d=5, seed=17)
    Xt, yt = to_torch(X, y)
    model = TorchLogisticRegression(C=1.0, max_iter=5, warm_start=True)

    def broken_step(self, closure):
        raise RuntimeError("some unrelated bug")

    monkeypatch.setattr(torch.optim.LBFGS, "step", broken_step)
    with pytest.raises(RuntimeError, match="some unrelated bug"):
        model.fit(Xt, yt)


def test_no_warm_start_resets_each_fit():
    X, y = make_binary_data(n=300, d=6, seed=5)
    Xt, yt = to_torch(X, y)
    model = TorchLogisticRegression(C=1.0, max_iter=50, warm_start=False)
    model.fit(Xt, yt)
    coef_first = model.coef_.clone()
    # refit on the same data from scratch (no warm start) -- should land in
    # the same place, not drift from where the previous fit left off.
    model.fit(Xt, yt)
    np.testing.assert_allclose(model.coef_.numpy(), coef_first.numpy(), atol=1e-6)
