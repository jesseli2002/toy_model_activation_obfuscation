import numpy as np
import pytest
import torch

from adversarial_report import _binary_probe_metrics_all_layers
from config import ResidualMLPConfig
from model import ResidualMLP
from probe_lib import LinearBoundary, _boundary_accuracy


def _make_model(num_x=2, d_model=3, d_mlp=8, num_blocks=2):
    # d_model == num_x + 1 (no padding columns): the embedding W_E = [I; 0]
    # would otherwise leave a constant-zero residual-stream coordinate at
    # layer 0, which makes sklearn's LDA covariance singular.
    cfg = ResidualMLPConfig(
        num_x=num_x, d_model=d_model, d_mlp=d_mlp, num_blocks=num_blocks
    )
    return ResidualMLP(cfg)


# ----------------------------------------------------------------------------
# _boundary_accuracy
# ----------------------------------------------------------------------------
def test_boundary_accuracy_1d_two_points():
    b = LinearBoundary(np.array([1.0]), 0.0)
    X = np.array([[-1.0], [1.0]])
    y = np.array([False, True])
    assert _boundary_accuracy(b, X, y) == 1.0


def test_boundary_accuracy_1d_intermediate():
    # Boundary at x = 0. Points: -2, -1, 1, 2, 3 -> labels False,False,True,True,True
    # but flip one label so accuracy is 4/5, not 0 or 1.
    b = LinearBoundary(np.array([1.0]), 0.0)
    X = np.array([[-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([False, False, False, True, True])  # third point mislabeled
    assert _boundary_accuracy(b, X, y) == pytest.approx(4 / 5)


def test_boundary_accuracy_nd_intermediate():
    # 2-D boundary: score = x0 + x1 - 1 > 0.
    b = LinearBoundary(np.array([1.0, 1.0]), -1.0)
    X = np.array(
        [
            [0.0, 0.0],  # score -1 -> False
            [1.0, 1.0],  # score  1 -> True
            [2.0, 0.0],  # score  1 -> True
            [-1.0, -1.0],  # score -3 -> False
            [0.6, 0.6],  # score  0.2 -> True
            [0.0, 0.4],  # score -0.6 -> False
        ]
    )
    y_all_correct = np.array([False, True, True, False, True, False])
    assert _boundary_accuracy(b, X, y_all_correct) == 1.0

    # Flip two labels -> 4/6 correct.
    y_some_wrong = np.array([True, True, False, False, True, False])
    assert _boundary_accuracy(b, X, y_some_wrong) == pytest.approx(4 / 6)


# ----------------------------------------------------------------------------
# _binary_probe_metrics_all_layers (end-to-end, synthetic model)
# ----------------------------------------------------------------------------
def test_binary_probe_metrics_all_layers_perfect_separation():
    """Layer 1's residual stream is `layer0 + block(layer0)`, where layer0 =
    `x_full @ W_E` copies (x, c) verbatim -- so even an untrained model's
    layer 1 gives a dataset where c is linearly readable, a cheap
    deterministic case to check the end-to-end wiring (forward pass -> DoM/
    logreg/LDA fits -> plot_inputs) without needing a trained checkpoint.
    (Layer 0 itself is unsuitable: c is pinned to one exact value per class
    there, so its pooled within-class covariance is singular and sklearn's
    LDA misbehaves -- the random block mixing at layer 1 avoids that.)
    `n_train` is large so the fixed-c signal dominates the random-x sampling
    noise in the DoM direction (x's per-coordinate mean noise shrinks as
    sqrt(n_train))."""
    model = _make_model()
    g = torch.Generator().manual_seed(0)
    n_train, n_test = 4096, 256
    layer = 1

    metrics, plot_inputs = _binary_probe_metrics_all_layers(
        model,
        c_lo=1.0,
        c_hi=2.0,
        layers=[layer],
        n_train=n_train,
        n_test=n_test,
        g=g,
        probe_backend_name="sklearn",
        desc="test",
    )

    assert set(metrics.keys()) == {layer}
    m = metrics[layer]
    assert set(m.keys()) == {"dom", "delta_norm", "logreg", "lda"}
    assert m["dom"] == pytest.approx(1.0)
    assert m["logreg"] == pytest.approx(1.0)
    assert m["lda"] == pytest.approx(1.0)
    assert m["delta_norm"] > 0

    pi = plot_inputs[layer]
    assert set(pi.keys()) == {
        "w_dom",
        "midpoint",
        "w_probe",
        "b_probe",
        "X_te",
        "y_te",
        "dist_lo",
        "dist_hi",
    }
    assert pi["X_te"].shape == (2 * n_test, model.d_model)
    assert pi["y_te"].shape == (2 * n_test,)

    # Sanity check plot_inputs' own DoM boundary (main()'s convention:
    # LinearBoundary(w_dom, -midpoint)) reproduces the same accuracy.
    dom_boundary = LinearBoundary(pi["w_dom"], -pi["midpoint"])
    assert _boundary_accuracy(dom_boundary, pi["X_te"], pi["y_te"]) == pytest.approx(
        m["dom"]
    )
