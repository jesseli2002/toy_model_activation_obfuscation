"""Aggregate the model-width sweep spread across runs/sweep17_*, sweep11_* and
sweep14_* (num_x = 32, 64, 128 respectively, width ratios otherwise held
constant) into the same two views as the other sweep*_analysis.py scripts,
each figure holding one panel per lambda side by side with a single shared
legend:

- Outcome bars: per-width stacked fractions, x-axis = num_x. A run either
  failed the task (loss) or, having succeeded, is binned by how well it hid
  from the linear probe (AUROC). Bars, not sweep7_analysis's stackplot:
  num_x only takes three values here and there's no meaningful
  interpolation between them.
- Loss-vs-AUROC scatter: per-run (task loss on the training distribution,
  probe AUROC), colored categorically by num_x.

Each width's runs come from a different sweep (num_x=32/64/128 also vary
iters and, since they're separate sweeps, the runs aren't discoverable by a
single glob+regex the way an in-sweep parameter is), so `_groups` below lists
tags explicitly rather than deriving them from sweep_lib.discovery -- same
approach as sweep_group_report.py, which covers this exact sweep.

LOSS_THRESHOLD / AUROC_THRESHOLDS match the other sweep_*_analysis.py
scripts' provisional values. PROBE_LAYER=2 matches
adversarial.penalty_layers for every run in this sweep.

"Loss" and "AUROC" are both freshly recomputed at CKPT -- loss via
data.eval_task_loss (deliberately excluding the adversarial/probe penalty,
since this only cares about task performance), and AUROC via a freshly refit
probe (see sweep13_analysis.py for why) -- rather than read from
history.jsonl or the checkpoint's own stored training-time probe.

Recomputing over a whole sweep is slow, so results are cached to CACHE_PATH
keyed by the settings they depend on; changing any of those settings misses
the cache rather than reusing a stale value. Pass --clear-cache to force a
full recompute.
"""

import argparse

PLOT_DIR = "plot/sweep_width"

LAMBDAS = ["0.1", "0.01"]  # one bar plot and one scatter each

# fmt: off
def _groups(lam: str) -> dict[int, list[str]]:
    """num_x -> run tags, for one lambda."""
    return {
        32: [f"sweep17_lr0.0015_iter100k_lam{lam}_tr{i}" for i in range(10)],
        64: [f"sweep11_lr0.0015_iter200k_lam{lam}_tr{i}" for i in range(10)],
        128: [f"sweep14_lr0.0015_iter400k_lam{lam}_tr{i}" for i in range(10)],
    }
# fmt: on

NUM_X = [32, 64, 128]
CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
PROBE_LAYER = 2  # matches adversarial.penalty_layers in these runs' config.json

MIN_RUNS = 1  # drop (num_x, lambda) points with fewer usable runs than this

LOSS_THRESHOLD = 0.01
AUROC_THRESHOLDS = (0.6, 0.75, 0.9)  # ascending; splits survivors into len+1 bins

FIG_WIDTH = (
    9  # inches; both multi-panel figures share this so they display consistently
)


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
import re
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep_lib.metrics import CACHE_PATH, MetricSpec, MetricStore
from sweep_lib.outcomes import BandSpec
from sweep_lib.plots import Series, loss_vs_auroc, stacked_bars

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPEC = MetricSpec(ckpt=CKPT, probe_layer=PROBE_LAYER)

BANDS = BandSpec(
    loss_threshold=LOSS_THRESHOLD,
    # n_hot_values=(),  # training-distribution loss only
    n_hot_values=(1, 2, 4, 8),  # match lambda sweep
    n_hot_loss_threshold=LOSS_THRESHOLD,
    auroc_thresholds=AUROC_THRESHOLDS,
)


class RunStats(NamedTuple):
    """A num_x's outcome-band fractions, aggregated over its trials."""

    frac: np.ndarray  # length n_bands, sums to 1
    n_ok: int  # usable trials this is averaged over


def _collect_run_stats(
    lam: str, store: MetricStore, n_bands: int
) -> tuple[dict[int, RunStats], list[tuple[float, float, int]]]:
    """Per-num_x band fractions and usable-run count at `lam`, recomputing
    each run's loss/AUROC where the cache doesn't already have them, plus
    every usable run's (loss, auroc, num_x) for the scatter."""
    groups = _groups(lam)
    stats: dict[int, RunStats] = {}
    scatter_points: list[tuple[float, float, int]] = []
    print(f"lambda={lam}")
    print(f"{'num_x':>6s} {'n_ok':>5s} {'n_total':>7s}")
    for num_x in NUM_X:
        tags = groups[num_x]
        counts = np.zeros(n_bands)
        n_ok = 0
        for tag in tags:
            if not store.has_checkpoint(tag):
                print(f"  {tag}: no checkpoint yet, skipped")
                continue
            auroc = store.auroc(tag)
            if auroc is None:
                print(f"  {tag}: no probe in checkpoint, skipped")
                continue
            loss = store.task_loss(tag)
            band = BANDS.classify(
                loss, {n: store.n_hot_loss(tag, n) for n in BANDS.n_hot_values}, auroc
            )
            counts[band] += 1
            n_ok += 1
            scatter_points.append((loss, auroc, num_x))
        if n_ok < MIN_RUNS:
            print(f"  num_x={num_x}: {n_ok} usable runs, skipped")
            continue
        stats[num_x] = RunStats(counts / n_ok, n_ok)
        print(f"{num_x:6d} {n_ok:5d} {len(tags):7d}")
    return stats, scatter_points


def _plot_by_width(run_stats_by_lam: dict[str, dict[int, RunStats]]) -> plt.Figure:
    """One panel per lambda, bars per num_x, sharing a single legend to the
    right of both panels."""
    lams = [lam for lam in LAMBDAS if lam in run_stats_by_lam]
    fig, axes = plt.subplots(1, len(lams), figsize=(FIG_WIDTH, 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for i, (ax, lam) in enumerate(zip(axes, lams)):
        run_stats = run_stats_by_lam[lam]
        widths = [w for w in NUM_X if w in run_stats]
        stacked_bars(
            ax,
            range(len(widths)),
            [run_stats[w].frac for w in widths],
            BANDS,
            n_runs=[run_stats[w].n_ok for w in widths],
            legend=i == 0,
        )
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_xlim(-0.6, len(widths) - 0.4)
        ax.set_xlabel("num_x")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(f"$\\lambda$ = {lam}", fontsize=11)

    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("fraction of runs")
    fig.suptitle("Outcome vs. model width", fontsize=12)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        leg_labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def _plot_loss_vs_auroc(
    scatter_by_lam: dict[str, list[tuple[float, float, int]]],
) -> plt.Figure:
    """One panel per lambda: scatter of (training-distribution task loss,
    probe AUROC), colored by num_x, sharing a single legend below both
    panels."""
    lams = [lam for lam in LAMBDAS if lam in scatter_by_lam]
    fig, axes = plt.subplots(1, len(lams), figsize=(FIG_WIDTH, 5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, lam in zip(axes, lams):
        losses, aurocs, widths = (np.array(v) for v in zip(*scatter_by_lam[lam]))
        loss_vs_auroc(
            ax,
            losses,
            aurocs,
            Series.categorical([f"num_x={w}" for w in widths]),
            LOSS_THRESHOLD,
            all_values=[f"num_x={w}" for w in NUM_X],
            show_loss_refs=True,
        )
        ax.set_xlabel("task loss on training distribution")
        ax.set_title(f"$\\lambda$ = {lam}")

    axes[0].set_ylabel("probe AUROC")
    fig.suptitle("Task loss vs. probe AUROC", fontsize=12)

    # One shared legend: per-panel run counts in the scatter labels would be
    # misleading once merged, so strip them.
    handles, leg_labels = axes[0].get_legend_handles_labels()
    leg_labels = [re.sub(r" \(n=\d+\)$", "", label) for label in leg_labels]
    fig.legend(
        handles,
        leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main(clear_cache: bool = False) -> None:
    if clear_cache and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    store = MetricStore(SPEC, CACHE_PATH, DEVICE)
    n_bands = BANDS.n_bands

    os.makedirs(PLOT_DIR, exist_ok=True)
    run_stats_by_lam: dict[str, dict[int, RunStats]] = {}
    scatter_by_lam: dict[str, list[tuple[float, float, int]]] = {}
    for lam in LAMBDAS:
        run_stats, scatter_points = _collect_run_stats(lam, store, n_bands)
        if not run_stats:
            print(f"  lambda={lam}: no usable runs, skipping its plots")
            continue
        run_stats_by_lam[lam] = run_stats
        scatter_by_lam[lam] = scatter_points

    if not run_stats_by_lam:
        raise SystemExit("no usable runs found across GROUPS")

    fig = _plot_by_width(run_stats_by_lam)
    fig.savefig(f"{PLOT_DIR}/by_width.png", bbox_inches="tight")

    fig = _plot_loss_vs_auroc(scatter_by_lam)
    fig.savefig(f"{PLOT_DIR}/loss_vs_auroc_scatter.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main(clear_cache=args.clear_cache)
