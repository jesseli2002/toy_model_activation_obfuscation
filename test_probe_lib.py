import numpy as np
import pytest
import torch

from config import ResidualMLPConfig
from model import ResidualMLP
from probe_lib import (
    LinearBoundary,
    binary_dataset_all_layers,
    binary_probe_metrics_all_layers,
    boundary_accuracy,
    boundary_auroc,
    raw_signed_distance,
)


def _make_model(num_x=2, d_model=3, d_mlp=8, num_blocks=2):
    # d_model == num_x + 1 (no padding columns): the embedding W_E = [I; 0]
    # would otherwise leave a constant-zero residual-stream coordinate at
    # layer 0, which makes sklearn's LDA covariance singular.
    cfg = ResidualMLPConfig(
        num_x=num_x, d_model=d_model, d_mlp=d_mlp, num_blocks=num_blocks
    )
    return ResidualMLP(cfg)


# ----------------------------------------------------------------------------
# boundary_accuracy
# ----------------------------------------------------------------------------
def test_boundary_accuracy_1d_two_points():
    b = LinearBoundary(np.array([1.0]), 0.0)
    X = np.array([[-1.0], [1.0]])
    y = np.array([False, True])
    assert boundary_accuracy(b, X, y) == 1.0


def test_boundary_accuracy_1d_intermediate():
    # Boundary at x = 0. Points: -2, -1, 1, 2, 3 -> labels False,False,True,True,True
    # but flip one label so accuracy is 4/5, not 0 or 1.
    b = LinearBoundary(np.array([1.0]), 0.0)
    X = np.array([[-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([False, False, False, True, True])  # third point mislabeled
    assert boundary_accuracy(b, X, y) == pytest.approx(4 / 5)


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
    assert boundary_accuracy(b, X, y_all_correct) == 1.0

    # Flip two labels -> 4/6 correct.
    y_some_wrong = np.array([True, True, False, False, True, False])
    assert boundary_accuracy(b, X, y_some_wrong) == pytest.approx(4 / 6)


# ----------------------------------------------------------------------------
# boundary_auroc / raw_signed_distance -- hand-derived, no fitting involved
# ----------------------------------------------------------------------------
def test_boundary_auroc_matches_hand_counted_value():
    # 1-D boundary at x = 0, so score == x. Positives {2, 0.5}, negatives
    # {1, -1, -2}: of the 2*3 = 6 (pos, neg) pairs, every one is concordant
    # (pos score > neg score) except (0.5, 1), so AUROC = 5/6 by the
    # Mann-Whitney definition -- countable by hand, no probe fit involved.
    b = LinearBoundary(np.array([1.0]), 0.0)
    X = np.array([[-2.0], [-1.0], [0.5], [1.0], [2.0]])
    y = np.array([False, False, True, False, True])
    assert boundary_auroc(b, X, y) == pytest.approx(5 / 6)


def test_boundary_auroc_ignores_bias_unlike_accuracy():
    # Shifting the boundary changes which side of 0 points fall on (accuracy
    # moves) but not their rank order (AUROC, being threshold-free, doesn't).
    X = np.array([[-2.0], [-1.0], [0.5], [1.0], [2.0]])
    y = np.array([False, False, True, False, True])
    auroc_b0 = boundary_auroc(LinearBoundary(np.array([1.0]), 0.0), X, y)
    auroc_shifted = boundary_auroc(LinearBoundary(np.array([1.0]), -10.0), X, y)
    assert auroc_shifted == pytest.approx(auroc_b0)


def test_raw_signed_distance_matches_hand_computation():
    # w = (3, 4) has norm 5; threshold = -b/||w|| = 2/5 = 0.4 in the raw-space
    # projection X @ w_hat. Pick points whose projection onto w_hat is exact
    # multiples of 1/5 so the arithmetic is checkable by hand.
    w = np.array([3.0, 4.0])
    b = -2.0
    X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    # X @ w_hat: [0, 5, 0.6]; minus threshold 0.4: [-0.4, 4.6, 0.2]
    expected = np.array([-0.4, 4.6, 0.2])
    assert raw_signed_distance(w, b, X) == pytest.approx(expected)


# ----------------------------------------------------------------------------
# binary_probe_metrics_all_layers (end-to-end, synthetic model)
# ----------------------------------------------------------------------------
def test_dom_direction_at_layer0_matches_known_c_axis():
    """At layer 0 the residual stream is `x_full @ W_E` with `W_E = [I; 0]`
    (see `_make_model`'s comment) -- i.e. `x_full` is copied verbatim, and its
    last coordinate is `c`, pinned to one exact value per class. So the true
    difference-of-means direction is *exactly* the unit vector on that
    coordinate; any deviation is finite-sample noise in the (c-independent) x
    coordinates, which shrinks as 1/sqrt(n_train) -- a large n keeps the
    cosine similarity to the known axis close to 1 without needing an exact
    pinned value.

    (`binary_probe_metrics_all_layers` can't be used directly at layer 0: c's
    fixed value makes the pooled within-class covariance singular and
    sklearn's LDA misbehaves, see `_make_model`'s comment. DoM doesn't need a
    covariance, so this checks it directly via `binary_dataset_all_layers`.)
    """
    model = _make_model()
    g = torch.Generator().manual_seed(0)
    n_train = 8192

    ds = binary_dataset_all_layers(
        model,
        model.num_x,
        n_train,
        c_lo=1.0,
        c_hi=2.0,
        layers=[0],
        generator=g,
        device="cpu",
    )
    r_lo, r_hi = ds[0]
    w_dom = (r_hi.mean(dim=0) - r_lo.mean(dim=0)).numpy()
    known_axis = np.zeros(model.d_model)
    known_axis[model.num_x] = 1.0  # the c coordinate

    cos_sim = w_dom @ known_axis / np.linalg.norm(w_dom)
    assert cos_sim > 0.99


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

    metrics, plot_inputs = binary_probe_metrics_all_layers(
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
    assert boundary_accuracy(dom_boundary, pi["X_te"], pi["y_te"]) == pytest.approx(
        m["dom"]
    )


# ----------------------------------------------------------------------------
# Analysis-only solver settings must not leak into training
# ----------------------------------------------------------------------------
def test_training_pipeline_keeps_stock_cold_steps():
    """train_adversarial_logreg builds its probe with three positional args.
    That call must keep NewtonLogisticRegression's own cold-fit budget -- the
    raised analysis budget is opt-in via newton_cold_steps."""
    from probe_backend import build_probe_pipeline
    from probe_newton import NewtonLogisticRegression

    stock = NewtonLogisticRegression().cold_steps
    training_pipeline = build_probe_pipeline(1.0, 1000, "newton")
    assert training_pipeline._logreg.cold_steps == stock


def test_analysis_refit_uses_raised_cold_steps_and_float64():
    """The refit path solves in float64 at the analysis step budget: both are
    what keep the fit converging as n_train grows (see PROBE_EVAL_FIT_DTYPE)."""
    from probe_backend import PROBE_EVAL_NEWTON_COLD_STEPS, build_probe_pipeline
    from probe_lib import PROBE_EVAL_FIT_DTYPE

    assert PROBE_EVAL_FIT_DTYPE is torch.float64
    pipeline = build_probe_pipeline(
        1.0, 1000, "newton", newton_cold_steps=PROBE_EVAL_NEWTON_COLD_STEPS
    )
    assert pipeline._logreg.cold_steps == PROBE_EVAL_NEWTON_COLD_STEPS

    captured = {}
    model = _make_model()
    g = torch.Generator().manual_seed(0)
    real_build = build_probe_pipeline

    import probe_backend

    def spy(*args, **kwargs):
        p = real_build(*args, **kwargs)
        real_fit = p.fit

        def fit(X, y):
            captured["dtype"] = X.dtype
            captured["cold_steps"] = getattr(
                getattr(p, "_logreg", None), "cold_steps", None
            )
            return real_fit(X, y)

        p.fit = fit
        return p

    probe_backend.build_probe_pipeline = spy
    try:
        binary_probe_metrics_all_layers(
            model,
            c_lo=1.0,
            c_hi=2.0,
            layers=[1],
            n_train=256,
            n_test=64,
            g=g,
            probe_backend_name="newton",
        )
    finally:
        probe_backend.build_probe_pipeline = real_build

    assert captured["dtype"] is torch.float64
    assert captured["cold_steps"] == PROBE_EVAL_NEWTON_COLD_STEPS
