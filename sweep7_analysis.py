"""Aggregate a lambda-sweep (see RUN_GLOB below) into a stacked-fraction-vs-lambda
chart: for each lambda, what fraction of its trials failed the task (loss), and
among those that succeeded, how well they hid from the layer-PROBE_LAYER
linear probe (AUROC), at one or more AUROC thresholds.

x-axis is lam / (1 - lam) -- the probe-to-task loss weight ratio (see
`(1 - lam_eff) * l_task + lam_eff * l_probe` in
train_adversarial_logreg._compute_losses) -- rather than lam itself, since
it's unbounded above (lam is capped at 1) and matches what the adversarial
penalty actually trades off against. Plotted symlog so lam=0 (ratio 0) has a
finite position while still spreading out the many-orders-of-magnitude tail.

LOSS_THRESHOLD / AUROC_THRESHOLDS are provisional -- a separate exercise is
picking the final values -- so they're constants here, not a CLI, alongside
everything else that may still change. Prints a summary table; plt.show()
only, nothing written to disk.

"Loss", "N-hot loss", and "AUROC" are all freshly recomputed at CKPT --
task loss via data.eval_task_loss (deliberately excluding the
adversarial/probe penalty, since this only cares about task performance),
N-hot loss via data.eval_n_hot_loss for each N in N_HOT_VALUES (only N
input coordinates nonzero per example, an OOD corner of input space a run
could fail to generalize into even after solving the training
distribution), and AUROC via a freshly refit probe at PROBE_LAYER (fit and
scored under the same residual-stream noise, see PROBE_*_NOISE_MULT) --
rather than read from history.jsonl or the checkpoint's own stored
training-time probe. A run only counts as having "succeeded" the task if
its N-hot loss is below threshold for every N in N_HOT_VALUES, not just the
easiest one -- see N_HOT_LOSS_THRESHOLD below. See
sweep_threshold_report.py, which uses the same recomputation (and the same
N_HOT_VALUES) for the same reasons (some runs' final checkpoint predates
their last history record).

Recomputing over a whole sweep is slow, so metrics are cached (see
sweep_lib.metrics). Pass --clear-cache to force a full recompute.
"""

import argparse
import re

PLOT_DIR = "plot/sweep7"
RUN_GLOB = "sweep7_lam*_tr*"
TAG_RE = re.compile(r"sweep7_lam([0-9\.]+)_tr(\d+)$")
# sweep7 has no lambda=0 arm of its own; sweep3's is reused instead (lambda=0
# disables the adversarial penalty entirely, so nothing about the
# sweep7-specific config bears on it).
LAM0_GLOB = "sweep3_lam0_tr*"
LAM0_TAG_RE = re.compile(r"sweep3_lam0_tr(\d+)$")
EXCLUDE_LAMBDAS: set[float] = set()
CKPT = "best"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
PROBE_LAYER = 2  # matches adversarial.penalty_layers in these runs' config.json
N_HOT_VALUES = (1, 2, 4, 8)  # 1 = one-hot; larger N approaches the training density

# task "succeeded" iff the in-distribution loss is below threshold AND every
# one of N_HOT_VALUES' losses is too -- see sweep_threshold_report.py for what
# distinguishes the two (in-distribution vs. N-hot-OOD input construction).
LOSS_THRESHOLD = 0.01
N_HOT_LOSS_THRESHOLD = 0.01

AUROC_THRESHOLDS = (0.6, 0.75, 0.9)  # ascending; splits survivors into len+1 bins


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="delete the shared metrics cache before running, forcing a full recompute",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep_lib.discovery import group_tags, matching_tags
from sweep_lib.metrics import CACHE_PATH, MetricSpec, MetricStore
from sweep_lib.plots import Series, loss_vs_auroc, stacked_fractions
from sweep_lib.outcomes import BandSpec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPEC = MetricSpec(ckpt=CKPT, probe_layer=PROBE_LAYER)

BANDS = BandSpec(
    loss_threshold=LOSS_THRESHOLD,
    n_hot_loss_threshold=N_HOT_LOSS_THRESHOLD,
    n_hot_values=N_HOT_VALUES,
    auroc_thresholds=AUROC_THRESHOLDS,
)


def _lambda_of(m: re.Match) -> float | None:
    lam = float(m.group(1))
    assert lam != 0.0, "sweep7 has no lambda=0 arm -- see LAM0_GLOB above"
    return None if lam in EXCLUDE_LAMBDAS else lam


def _discover_tags_by_lambda() -> dict[float, list[str]]:
    by_lambda = group_tags(RUN_GLOB, TAG_RE, _lambda_of)
    if 0.0 not in EXCLUDE_LAMBDAS:
        lam0_tags = [tag for tag, _ in matching_tags(LAM0_GLOB, LAM0_TAG_RE)]
        if lam0_tags:
            by_lambda[0.0] = lam0_tags
    return by_lambda


def _plot_loss_vs_auroc(points: list[tuple[float, float, float]]) -> None:
    """Scatter of every run's (task loss, probe AUROC), colored by lambda."""
    losses, aurocs, lams = (np.array(v) for v in zip(*points))

    fig, ax = plt.subplots(figsize=(7, 5))
    loss_vs_auroc(
        ax, losses, aurocs, Series.continuous(lams, r"$\lambda$"), LOSS_THRESHOLD
    )
    ax.set_xlabel("task loss")
    ax.set_title(r"Task loss vs. probe AUROC, colored by $\lambda$")
    ax.legend()
    fig.savefig(f"{PLOT_DIR}/loss_vs_auroc_scatter.png", bbox_inches="tight")


def _lambda_counts(
    tags: list[str], store: MetricStore
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Per-band run counts for one lambda's `tags`, plus each usable run's
    (loss, auroc) for the scatter. Unusable runs are skipped, so the counts
    sum to len(usable) <= len(tags)."""
    counts = np.zeros(BANDS.n_bands)
    points = []
    for tag in tags:
        if not store.has_checkpoint(tag):
            print(f"  {tag}: no checkpoint yet, skipped")
            continue
        auroc = store.auroc(tag)
        if auroc is None:
            print(f"  {tag}: no probe in checkpoint, skipped")
            continue
        loss = store.task_loss(tag)
        n_hot = store.n_hot_losses(tag, N_HOT_VALUES)
        counts[BANDS.classify(loss, n_hot, auroc)] += 1
        points.append((loss, auroc))
    return counts, points


def _gather_sweep() -> (
    tuple[list[float], list[np.ndarray], list[tuple[float, float, float]]]
):
    """Walk the sweep's runs, returning per-lambda loss-weight ratios, the
    fraction of runs in each outcome band, and (loss, auroc, lam) per run.
    Prints a per-lambda summary table as it goes."""
    by_lambda = _discover_tags_by_lambda()
    store = MetricStore(SPEC, CACHE_PATH, DEVICE)
    ratios = []
    fractions = []  # one length-BANDS.n_bands array per lambda
    scatter_points = []  # (loss, auroc, lam) per run, across all lambdas

    print(f"{'lambda':>10s} {'ratio':>10s} {'n_ok':>5s} {'n_total':>7s}")
    for lam in sorted(by_lambda):
        tags = by_lambda[lam]
        counts, points = _lambda_counts(tags, store)
        if not points:
            print(f"  {lam}: no usable runs, skipped")
            continue
        ratio = lam / (1 - lam)
        ratios.append(ratio)
        fractions.append(counts / len(points))
        scatter_points.extend((loss, auroc, lam) for loss, auroc in points)
        print(f"{lam:10g} {ratio:10.4g} {len(points):5d} {len(tags):7d}")
    return ratios, fractions, scatter_points


def _plot_stacked_fractions(ratios: list[float], fractions: list[np.ndarray]) -> None:
    """Stacked fraction-of-runs-per-outcome-band vs. the probe-to-task loss
    weight ratio, one stack column's worth of data per lambda."""
    x = np.sort(np.array(ratios))

    fig, ax = plt.subplots(figsize=(8, 5))
    stacked_fractions(ax, np.array(ratios), fractions, BANDS)

    # linear region around lam=0, so the ratio-0 arm has a finite position
    ax.set_xscale("symlog", linthresh=x[1], linscale=0.5)
    ax.set_xlabel(
        r"$\lambda / (1 - \lambda)$ (ratio of probe-loss to task-loss weights)"
    )
    ax.set_ylabel("fraction of runs")
    ax.set_title(r"Outcome vs. $\lambda$")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(f"{PLOT_DIR}/lam_sweep_task{LOSS_THRESHOLD}.png", bbox_inches="tight")
    plt.tight_layout()


def main(clear_cache: bool = False):
    if clear_cache and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    ratios, fractions, scatter_points = _gather_sweep()

    _plot_stacked_fractions(ratios, fractions)
    _plot_loss_vs_auroc(scatter_points)

    plt.show()


if __name__ == "__main__":
    main(clear_cache=args.clear_cache)
