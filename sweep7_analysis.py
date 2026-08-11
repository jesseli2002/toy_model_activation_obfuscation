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

import glob
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep_lib.metrics import CACHE_PATH, MetricSpec, MetricStore
from sweep_lib.outcomes import BandSpec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPEC = MetricSpec(ckpt=CKPT, probe_layer=PROBE_LAYER)

BANDS = BandSpec(
    loss_threshold=LOSS_THRESHOLD,
    n_hot_loss_threshold=N_HOT_LOSS_THRESHOLD,
    n_hot_values=N_HOT_VALUES,
    auroc_thresholds=AUROC_THRESHOLDS,
)


def _discover_tags_by_lambda() -> dict[float, list[str]]:
    by_lambda: dict[float, list[str]] = {}
    for path in sorted(glob.glob(os.path.join("runs", RUN_GLOB))):
        tag = os.path.basename(path)
        m = TAG_RE.match(tag)
        if not m:
            continue
        lam = float(m.group(1))
        assert lam != 0.0, f"{tag}: sweep7 has no lambda=0 arm -- see LAM0_GLOB above"
        if lam in EXCLUDE_LAMBDAS:
            continue
        by_lambda.setdefault(lam, []).append(tag)
    if 0.0 not in EXCLUDE_LAMBDAS:
        lam0_tags = [
            os.path.basename(path)
            for path in sorted(glob.glob(os.path.join("runs", LAM0_GLOB)))
            if LAM0_TAG_RE.match(os.path.basename(path))
        ]
        if lam0_tags:
            by_lambda[0.0] = lam0_tags
    return by_lambda


def _plot_loss_vs_auroc(points: list[tuple[float, float, float]]) -> None:
    """Scatter of every run's (task loss, probe AUROC), colored by lambda
    on a log scale. SymLogNorm rather than LogNorm so lam=0 -- which a log
    scale can't represent -- still gets a (bottom-of-range) color, the same
    trick the ratio x-axis above uses for the stackplot."""
    losses, aurocs, lams = (np.array(v) for v in zip(*points))
    linthresh = lams[lams > 0].min() if np.any(lams > 0) else 1.0
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=lams.min(), vmax=lams.max())

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        losses,
        aurocs,
        c=lams,
        cmap="viridis",
        norm=norm,
        edgecolor="black",
        linewidth=0.5,
    )
    fig.colorbar(sc, ax=ax, label=r"$\lambda$")
    ax.set_xlabel("task loss")
    ax.set_ylabel("probe AUROC")
    ax.set_title(r"Task loss vs. probe AUROC, colored by $\lambda$")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    # Log-scale minor tick labels (2x, 3x, ...) crowd together when the data
    # spans less than a decade; rotating keeps them legible at any range.
    plt.setp(ax.get_xticklabels(which="both"), rotation=45, ha="right")

    ax.axvline(
        LOSS_THRESHOLD, linestyle="--", color="black", label="loss threshold", alpha=0.5
    )
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
    order = np.argsort(ratios)
    x = np.array(ratios)[order]
    frac_matrix = np.array(fractions)[order]  # (n_lambda, n_bands)

    labels = BANDS.labels()
    colors = BANDS.colors()

    fig, ax = plt.subplots(figsize=(8, 5))
    # stackplot stacks bottom-to-top in argument order; we want "failed" on
    # top and "most hidden" at the bottom, so feed bands in reverse.
    stacked = frac_matrix.T[::-1]
    ax.stackplot(
        x,
        *stacked,
        labels=labels[::-1],
        colors=colors[::-1],
        alpha=0.9,
    )
    # stackplot doesn't support markers itself; overlay them on each band's
    # upper boundary (cumulative sum) to show the underlying data points.
    cum = np.cumsum(stacked, axis=0)
    for row, color in zip(cum[:-1], colors[::-1][:-1]):  # don't plot top row, always 1
        ax.plot(
            x,
            row,
            marker="o",
            ls="-",
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

    SYMLOG_LINTHRESH = x[1]  # linear-region half-width (in ratio units) around lam=0
    ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=0.5)
    # ax.axvline(x[1] / 2, ls="--", lw=1, color="#52514e")
    # ax.axvline(x[1] / 2, ls="--", lw=1, color="black")
    ax.set_xlabel(
        r"$\lambda / (1 - \lambda)$ (ratio of probe-loss to task-loss weights)"
    )
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, x[-1])
    ax.set_title(f"Outcome vs. $\\lambda$")
    # ax.set_title(f"Outcome vs. $\\lambda$ \n({RUN_GLOB})")
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
