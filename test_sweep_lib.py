import math

import numpy as np
import pytest

from sweep_lib.outcomes import FAILED_COLOR, SEQ_RAMP, BandSpec

# ----------------------------------------------------------------------------
# BandSpec.classify
# ----------------------------------------------------------------------------
THREE = BandSpec(
    loss_threshold=0.01,
    n_hot_loss_threshold=0.01,
    n_hot_values=(1, 2, 4, 8),
    auroc_thresholds=(0.6, 0.75, 0.9),
)
OK_N_HOT = {1: 0.0, 2: 0.0, 4: 0.0, 8: 0.0}


def test_n_bands():
    assert THREE.n_bands == 5  # failed + 4 AUROC intervals


@pytest.mark.parametrize(
    "auroc,expected",
    [(0.95, 1), (0.9, 1), (0.8, 2), (0.75, 2), (0.7, 3), (0.6, 3), (0.5, 4)],
)
def test_classify_auroc_bands(auroc, expected):
    assert THREE.classify(0.0, OK_N_HOT, auroc) == expected


def test_classify_task_loss_at_threshold_fails():
    # thresholds are exclusive upper bounds: >= fails
    assert THREE.classify(0.01, OK_N_HOT, 0.5) == 0
    assert THREE.classify(0.00999, OK_N_HOT, 0.5) == 4


def test_classify_any_n_hot_over_threshold_fails():
    for n in (1, 2, 4, 8):
        losses = {**OK_N_HOT, n: 0.02}
        assert THREE.classify(0.0, losses, 0.5) == 0, f"N={n} should fail the run"


def test_classify_ignores_n_hot_outside_spec():
    spec = BandSpec(loss_threshold=0.01, n_hot_loss_threshold=0.01, n_hot_values=(1,))
    # N=8 is over the bar but not in n_hot_values, so it must not count
    assert spec.classify(0.0, {1: 0.0, 8: 99.0}, 0.5) == 4


# ----------------------------------------------------------------------------
# BandSpec degenerate cases -- each threshold disabled independently
# ----------------------------------------------------------------------------
def test_empty_n_hot_values_ignores_n_hot():
    spec = BandSpec(loss_threshold=0.01, n_hot_loss_threshold=0.01, n_hot_values=())
    assert spec.classify(0.0, {}, 0.5) == 4  # no KeyError, no max(()) ValueError
    assert spec.classify(0.0, {1: 99.0}, 0.5) == 4


def test_inf_loss_threshold_ignores_task_loss():
    spec = BandSpec(n_hot_loss_threshold=0.01, n_hot_values=(1,))
    assert spec.loss_threshold == math.inf
    assert spec.classify(1e9, {1: 0.0}, 0.5) == 4
    assert spec.classify(1e9, {1: 0.02}, 0.5) == 0  # N-hot still applies


def test_all_thresholds_disabled_never_fails():
    spec = BandSpec()
    assert spec.classify(1e9, {}, 0.99) == 1


def test_empty_auroc_thresholds():
    spec = BandSpec(loss_threshold=0.01, auroc_thresholds=())
    assert spec.n_bands == 2
    assert spec.classify(0.0, {}, 0.99) == 1
    assert spec.classify(1.0, {}, 0.99) == 0
    assert spec.labels() == ["failed task", "hidden"]  # no NoneType format crash
    assert len(spec.colors()) == 2


def test_auroc_thresholds_must_ascend():
    with pytest.raises(ValueError, match="ascending"):
        BandSpec(auroc_thresholds=(0.9, 0.6))


# ----------------------------------------------------------------------------
# BandSpec.labels / colors
# ----------------------------------------------------------------------------
def test_labels_and_colors_align_with_bands():
    assert len(THREE.labels()) == THREE.n_bands
    assert len(THREE.colors()) == THREE.n_bands


def test_colors_run_failure_then_light_to_dark():
    colors = THREE.colors()
    assert colors[0] == FAILED_COLOR
    assert colors[1] == SEQ_RAMP[0]  # least hidden -> lightest
    assert colors[-1] == SEQ_RAMP[-1]  # most hidden -> darkest


def test_labels_describe_each_interval():
    assert THREE.labels() == [
        "failed task",
        "not hidden\n(AUROC $\\geq$ 0.9)",
        "partially hidden\n(0.75 $\\leq$ AUROC < 0.9)",
        "partially hidden\n(0.6 $\\leq$ AUROC < 0.75)",
        "hidden\n(AUROC < 0.6)",
    ]


# ----------------------------------------------------------------------------
# Equivalence with the pre-refactor per-script implementations
# ----------------------------------------------------------------------------
def _old_classify_sweep7(loss, worst_n_hot_loss, auroc, thresholds, lt, nlt):
    if loss >= lt or worst_n_hot_loss >= nlt:
        return 0
    for i, t in enumerate(reversed(thresholds)):
        if auroc >= t:
            return i + 1
    return len(thresholds) + 1


def _old_band_colors(thresholds):
    n_hiding_bins = len(thresholds) + 1
    idxs = np.linspace(0, len(SEQ_RAMP) - 1, n_hiding_bins).astype(int)
    return [FAILED_COLOR] + [SEQ_RAMP[i] for i in idxs]


def test_matches_old_sweep7_classify():
    rng = np.random.default_rng(0)
    thresholds = [0.6, 0.75, 0.9]
    for _ in range(2000):
        loss = float(rng.uniform(0, 0.03))
        n_hot = {n: float(rng.uniform(0, 0.03)) for n in (1, 2, 4, 8)}
        auroc = float(rng.uniform(0.5, 1.0))
        old = _old_classify_sweep7(
            loss, max(n_hot.values()), auroc, thresholds, 0.01, 0.01
        )
        assert THREE.classify(loss, n_hot, auroc) == old


def test_matches_old_sweep8_13_classify():
    """sweep8/sweep13 gated on the one-hot loss alone -- n_hot_values=(1,)."""
    spec = BandSpec(
        loss_threshold=0.01,
        n_hot_loss_threshold=0.01,
        n_hot_values=(1,),
        auroc_thresholds=(0.6, 0.75, 0.9),
    )
    rng = np.random.default_rng(1)
    thresholds = [0.6, 0.75, 0.9]
    for _ in range(2000):
        loss = float(rng.uniform(0, 0.03))
        one_hot = float(rng.uniform(0, 0.03))
        auroc = float(rng.uniform(0.5, 1.0))
        old = _old_classify_sweep7(loss, one_hot, auroc, thresholds, 0.01, 0.01)
        assert spec.classify(loss, {1: one_hot}, auroc) == old


@pytest.mark.parametrize("thresholds", [(0.6, 0.75, 0.9), (0.5,), (0.6, 0.8)])
def test_matches_old_band_colors(thresholds):
    spec = BandSpec(auroc_thresholds=thresholds)
    assert spec.colors() == _old_band_colors(list(thresholds))


# ----------------------------------------------------------------------------
# sweep_lib.plots
# ----------------------------------------------------------------------------
def test_categorical_series_keeps_the_callers_order():
    """Colors and legend order follow the order the caller declared, not
    sorted order -- sorting would put nx128 ahead of nx32."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sweep_lib.plots import Series, loss_vs_auroc

    groups = ["nx32", "nx64", "nx128"]
    labels = np.array(groups * 4)
    losses = np.linspace(0.001, 0.01, len(labels))
    aurocs = np.linspace(0.5, 1.0, len(labels))

    fig, ax = plt.subplots()
    loss_vs_auroc(ax, losses, aurocs, Series.categorical(labels), all_values=groups)
    legend_order = [t.split(" (")[0] for t in ax.get_legend_handles_labels()[1]]
    plt.close(fig)
    assert legend_order == groups
