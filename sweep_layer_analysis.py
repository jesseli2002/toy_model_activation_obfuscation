"""Aggregate the probed-layer sweep spread across runs/sweep18_layer{2,4,6,8,10}_*
into the same two views as the other sweep*_analysis.py scripts, one figure
pair per group in GROUPS:

- Outcome bars: per-layer stacked fractions, x-axis = probed layer. A run
  either failed the task (loss) or, having succeeded, is binned by how well
  it hid from the linear probe (AUROC).
- Loss-vs-AUROC scatter: per-run (task loss on the training distribution,
  probe AUROC), colored by layer with a discrete viridis colormap -- layer
  is ordered, so a ramp (rather than loss_vs_auroc's categorical color cycle)
  is the more legible choice.

Each group is a separate sweep18 config (lambda / rampup) swept over the
penalized layer, so its runs aren't discoverable by a single glob+regex the
way an in-sweep parameter is; `GROUPS` below lists tags explicitly, same
approach as sweep_group_report.py and sweep_width_analysis.py.

LOSS_THRESHOLD / AUROC_THRESHOLDS match the other sweep_*_analysis.py
scripts' provisional values.

"Loss" and "AUROC" are both freshly recomputed at CKPT -- loss via
data.eval_task_loss (deliberately excluding the adversarial/probe penalty,
since this only cares about task performance), and AUROC via a freshly refit
probe at each run's own penalized layer (see sweep13_analysis.py for why) --
rather than read from history.jsonl or the checkpoint's own stored
training-time probe.

Recomputing over a whole sweep is slow, so results are cached to CACHE_PATH
keyed by the settings they depend on; changing any of those settings misses
the cache rather than reusing a stale value. Pass --clear-cache to force a
full recompute.
"""

import argparse

PLOT_DIR = "plot/sweep_layer"

LAYERS = [2, 4, 6, 8, 10]

# fmt: off
GROUPS: dict[str, dict[int, list[str]]] = {
    "lam0.01": {
        layer: [f"sweep18_layer{layer}_lam0.01_noise0.01_tr{i}" for i in range(10)]
        for layer in LAYERS
    },
    "lam0.032_ramp200k": {
        layer: [f"sweep18_layer{layer}_lam0.032_ramp200k_noise0.01_tr{i}" for i in range(10)]
        for layer in LAYERS
    },
    "lam0.1_ramp200k": {
        layer: [f"sweep18_layer{layer}_lam0.1_ramp200k_noise0.01_tr{i}" for i in range(10)]
        for layer in LAYERS
    },
}
# fmt: on

CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt

MIN_RUNS = 1  # drop (group, layer) points with fewer usable runs than this

LOSS_THRESHOLD = 0.01
AUROC_THRESHOLDS = (0.6, 0.75, 0.9)  # ascending; splits survivors into len+1 bins

FIG_WIDTH = 6  # inches; both figures share this so they display consistently


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
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep_lib.metrics import CACHE_PATH, MetricSpec, MetricStore
from sweep_lib.outcomes import BandSpec
from sweep_lib.plots import Series, loss_vs_auroc, stacked_bars

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# probe_layer is left unset on the base spec: each layer's tags are queried
# through a MetricStore.with_probe_layer(layer) below, so every run is
# probed at the layer it was actually penalized at.
SPEC = MetricSpec(ckpt=CKPT)

BANDS = BandSpec(
    loss_threshold=LOSS_THRESHOLD,
    n_hot_values=(),  # training-distribution loss only
    n_hot_loss_threshold=LOSS_THRESHOLD,
    auroc_thresholds=AUROC_THRESHOLDS,
)


class RunStats(NamedTuple):
    """A layer's outcome-band fractions, aggregated over its trials."""

    frac: np.ndarray  # length n_bands, sums to 1
    n_ok: int  # usable trials this is averaged over


def _collect_run_stats(
    group_tags: dict[int, list[str]], store: MetricStore, n_bands: int
) -> tuple[dict[int, RunStats], list[tuple[float, float, int]]]:
    """Per-layer band fractions and usable-run count, recomputing each run's
    loss/AUROC where the cache doesn't already have them, plus every usable
    run's (loss, auroc, layer) for the scatter."""
    stats: dict[int, RunStats] = {}
    scatter_points: list[tuple[float, float, int]] = []
    print(f"{'layer':>6s} {'n_ok':>5s} {'n_total':>7s}")
    for layer in LAYERS:
        tags = group_tags[layer]
        layer_store = store.with_probe_layer(layer)
        counts = np.zeros(n_bands)
        n_ok = 0
        for tag in tags:
            if not layer_store.has_checkpoint(tag):
                print(f"  {tag}: no checkpoint yet, skipped")
                continue
            auroc = layer_store.auroc(tag)
            if auroc is None:
                print(f"  {tag}: no probe in checkpoint, skipped")
                continue
            loss = layer_store.task_loss(tag)
            band = BANDS.classify(
                loss,
                {n: layer_store.n_hot_loss(tag, n) for n in BANDS.n_hot_values},
                auroc,
            )
            counts[band] += 1
            n_ok += 1
            scatter_points.append((loss, auroc, layer))
        if n_ok < MIN_RUNS:
            print(f"  layer={layer}: {n_ok} usable runs, skipped")
            continue
        stats[layer] = RunStats(counts / n_ok, n_ok)
        print(f"{layer:6d} {n_ok:5d} {len(tags):7d}")
    return stats, scatter_points


def _plot_by_layer(group_name: str, run_stats: dict[int, RunStats]) -> plt.Figure:
    layers = [l for l in LAYERS if l in run_stats]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.2))
    stacked_bars(
        ax,
        range(len(layers)),
        [run_stats[l].frac for l in layers],
        BANDS,
        n_runs=[run_stats[l].n_ok for l in layers],
        legend=True,
    )
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlim(-0.6, len(layers) - 0.4)
    ax.set_xlabel("probed layer")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction of runs")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"Outcome vs. probed layer ({group_name})", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    return fig


def _plot_loss_vs_auroc(
    group_name: str, scatter_points: list[tuple[float, float, int]]
) -> plt.Figure:
    losses, aurocs, layers = (np.array(v) for v in zip(*scatter_points))
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 5))
    loss_vs_auroc(
        ax,
        losses,
        aurocs,
        Series.ordinal(layers, "probed layer"),
        LOSS_THRESHOLD,
        all_values=LAYERS,
        show_loss_refs=True,
    )
    ax.set_xlabel("task loss on training distribution")
    ax.set_title(f"Task loss vs. probe AUROC ({group_name})", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main(clear_cache: bool = False) -> None:
    if clear_cache and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    store = MetricStore(SPEC, CACHE_PATH, DEVICE)
    n_bands = BANDS.n_bands

    os.makedirs(PLOT_DIR, exist_ok=True)
    any_usable = False
    for group_name, group_tags in GROUPS.items():
        print(f"group={group_name}")
        run_stats, scatter_points = _collect_run_stats(group_tags, store, n_bands)
        if not run_stats:
            print(f"  group={group_name}: no usable runs, skipping its plots")
            continue
        any_usable = True

        fig = _plot_by_layer(group_name, run_stats)
        fig.savefig(f"{PLOT_DIR}/{group_name}_by_layer.png", bbox_inches="tight")
        fig.savefig(f"{PLOT_DIR}/{group_name}_by_layer.svg", bbox_inches="tight")

        fig = _plot_loss_vs_auroc(group_name, scatter_points)
        fig.savefig(f"{PLOT_DIR}/{group_name}_loss_vs_auroc.png", bbox_inches="tight")
        fig.savefig(f"{PLOT_DIR}/{group_name}_loss_vs_auroc.svg", bbox_inches="tight")

    if not any_usable:
        raise SystemExit("no usable runs found across GROUPS")

    plt.show()


if __name__ == "__main__":
    main(clear_cache=args.clear_cache)
