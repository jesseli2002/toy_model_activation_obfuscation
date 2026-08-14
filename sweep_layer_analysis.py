"""Aggregate the probed-layer sweep spread across runs/sweep18_layer{2,4,6,8,10}_*
and runs/sweep19_layers{L1}-{L2}_* into three views:

- Outcome bars: per-layer stacked fractions, x-axis = probed layer. A run
  either failed the task (loss) or, having succeeded, is binned by how well
  it hid from the linear probe (AUROC).
- Loss-vs-AUROC scatter: per-run (task loss on the training distribution,
  probe AUROC), colored by layer with a discrete viridis colormap -- layer
  is ordered, so a ramp (rather than loss_vs_auroc's categorical color cycle)
  is the more legible choice.
- Pareto frontier: per-run (task loss, probe AUROC) -- both lower is better
  -- for a chosen set of "layer sets" (a layer set is the tuple of layers a
  run's probe penalty, and the refit probe scoring it, is concatenated over,
  e.g. (2,) or (2, 4)), pooling all of a layer set's lambdas and trials onto
  one panel. One color per layer set, one marker shape per lambda; points
  dominated in both loss and AUROC by another point in the same layer set are
  drawn translucent, frontier points opaque. Extending to a new comparison:
  add an entry to PARETO_COMPARISONS naming the layer sets (as tuples) to
  plot together; --comparison restricts a run to one entry (its own figure,
  legend alongside it) instead of the default all-of-them 2x2 grid. Layer
  sets not already covered by sweep18/19's naming need a `_pareto_tags` case
  added.

Each view is one 2x2 figure, one panel per lambda (bars/scatter) or
PARETO_COMPARISONS entry (pareto, absent --comparison), plus a shared legend
-- in the fourth (otherwise empty) panel for bars/scatter, in one row below
all four panels for pareto -- meant for publication, so panels carry only
the lambda value or layer sets being compared, not run tags or other sweep
configuration (recorded in the surrounding text instead).

Each lambda's runs come from a different sweep18/19 config (only the
penalized layer(s) vary within one), so they aren't discoverable by a single
glob+regex the way an in-sweep parameter is; `GROUPS` and `_pareto_tags`
below list/build tags explicitly, same approach as sweep_group_report.py and
sweep_width_analysis.py.

LOSS_THRESHOLD / AUROC_THRESHOLDS match the other sweep_*_analysis.py
scripts' provisional values.

"Loss" and "AUROC" are both freshly recomputed at CKPT -- loss via
data.eval_task_loss (deliberately excluding the adversarial/probe penalty,
since this only cares about task performance), and AUROC via a freshly refit
probe over each run's own penalized layers (see sweep13_analysis.py for why)
-- rather than read from history.jsonl or the checkpoint's own stored
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

# Grid position (row-major, left-to-right/top-to-bottom) -> group, display
# label. Fourth slot is left as None: its panel is cleared and holds the
# shared legend instead.
GRID_ORDER: list[tuple[str, str] | None] = [
    ("lam0.01", "0.01"),
    ("lam0.032_ramp200k", "0.032"),
    ("lam0.1_ramp200k", "0.1"),
    None,
]

CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt

MIN_RUNS = 1  # drop (group, layer) points with fewer usable runs than this

LOSS_THRESHOLD = 0.01
AUROC_THRESHOLDS = (0.6, 0.75, 0.9)  # ascending; splits survivors into len+1 bins

FIG_WIDTH = 8  # inches; all three figures share this so they display consistently

# -- Pareto-frontier view. LAMBDAS/its suffixes duplicate the lambda values
# GROUPS already encodes (as "0.01"/"0.032"/"0.1"), kept separate since this
# view indexes tags by layer *set* rather than by single layer.
LAMBDAS = ["0.01", "0.032", "0.1"]
_LAMBDA_SUFFIX = {
    "0.01": "noise0.01",
    "0.032": "ramp200k_noise0.01",
    "0.1": "ramp200k_noise0.01",
}
_LAMBDA_MARKER = {"0.01": "o", "0.032": "s", "0.1": "^"}

# Each comparison: name -> layer sets (tuples of penalized layers) to plot
# together on one Pareto-frontier figure.
PARETO_COMPARISONS: dict[str, list[tuple[int, ...]]] = {
    "layer2_4_24": [(2,), (4,), (2, 4)],
    "layer2_6_26": [(2,), (6,), (2, 6)],
    "layer2_8_28": [(2,), (8,), (2, 8)],
    "layer2_10_210": [(2,), (10,), (2, 10)],
}

FRONTIER_ALPHA = 1.0
OFF_FRONTIER_ALPHA = 0.25
MARKER_SIZE = 55
PARETO_XLIM_MAX = 0.4  # ~ the do-nothing task loss

# --pareto-bootstrap: shades each frontier with a trial-to-trial uncertainty
# band (see _bootstrap_frontier_band), quantifying how much the staircase
# would move under a different draw of trial seeds.
N_BOOTSTRAP = 200
BOOTSTRAP_PERCENTILES = (10, 90)
BOOTSTRAP_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="delete the shared metrics cache before running, forcing a full recompute",
    )
    p.add_argument(
        "--plot",
        choices=["bars", "scatter", "pareto", "all"],
        default="all",
        help="which view(s) to draw (default: all)",
    )
    p.add_argument(
        "--comparison",
        choices=list(PARETO_COMPARISONS),
        default=None,
        help="pareto view only: restrict to one entry of PARETO_COMPARISONS (default: all of them)",
    )
    p.add_argument(
        "--pareto-bootstrap",
        action="store_true",
        help="pareto view only: shade each frontier with a trial-bootstrap uncertainty band",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import os
from typing import NamedTuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from sweep_lib.metrics import CACHE_PATH, MetricSpec, MetricStore
from sweep_lib.outcomes import BandSpec
from sweep_lib.plots import Series, loss_vs_auroc, stacked_bars

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# probe_layers is left unset on the base spec: each layer/layer-set's tags
# are queried through a MetricStore.with_probe_layers(...) below, so every
# run is probed at the layers it was actually penalized at.
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
        layer_store = store.with_probe_layers(layer)
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


def _grid_axes(fig, axes) -> tuple[dict[str, plt.Axes], plt.Axes]:
    """Flatten a 2x2 `axes` array against GRID_ORDER: {group -> its axes},
    plus the empty slot's axes for the shared legend."""
    flat = axes.flatten()
    group_axes = {}
    legend_ax = None
    for ax, slot in zip(flat, GRID_ORDER):
        if slot is None:
            legend_ax = ax
        else:
            group, _label = slot
            group_axes[group] = ax
    return group_axes, legend_ax


def _plot_by_layer(run_stats_by_group: dict[str, dict[int, RunStats]]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 0.85), sharey=True)
    group_axes, legend_ax = _grid_axes(fig, axes)

    handles, labels = None, None
    for group, label in (slot for slot in GRID_ORDER if slot is not None):
        ax = group_axes[group]
        run_stats = run_stats_by_group.get(group, {})
        layers = [l for l in LAYERS if l in run_stats]
        stacked_bars(
            ax,
            range(len(layers)),
            [run_stats[l].frac for l in layers],
            BANDS,
            n_runs=[run_stats[l].n_ok for l in layers],
            legend=handles is None,
        )
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers])
        ax.set_xlim(-0.6, len(layers) - 0.4)
        ax.set_xlabel("probed layer")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(f"$\\lambda$ = {label}", fontsize=11)

    for ax in group_axes.values():
        ax.set_ylim(0, 1.08)
    axes[0, 0].set_ylabel("fraction of runs")
    axes[1, 0].set_ylabel("fraction of runs")

    legend_ax.axis("off")
    if handles is not None:
        legend_ax.legend(
            handles[::-1], labels[::-1], loc="center", fontsize=10, frameon=False
        )

    fig.tight_layout()
    return fig


def _plot_loss_vs_auroc(
    scatter_by_group: dict[str, list[tuple[float, float, int]]],
) -> plt.Figure:
    # Not sharex/sharey=True: that auto-hides tick labels on interior panels,
    # but each panel should keep its own labels even though all three share
    # the same limits (set manually below).
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 0.85))
    group_axes, legend_ax = _grid_axes(fig, axes)
    cmap = plt.get_cmap("viridis", len(LAYERS))

    ref_handles, ref_labels = [], []
    plotted_axes = []
    for group, label in (slot for slot in GRID_ORDER if slot is not None):
        ax = group_axes[group]
        points = scatter_by_group.get(group, [])
        if not points:
            ax.axis("off")
            continue
        losses, aurocs, layers = (np.array(v) for v in zip(*points))
        loss_vs_auroc(
            ax,
            losses,
            aurocs,
            Series.ordinal(layers, "probed layer"),
            LOSS_THRESHOLD,
            all_values=LAYERS,
            show_loss_refs=True,
            show_colorbar=False,
        )
        if not ref_handles:
            ref_handles, ref_labels = ax.get_legend_handles_labels()
        ax.set_xlabel("task loss on training distribution")
        ax.set_title(f"$\\lambda$ = {label}", fontsize=11)
        plotted_axes.append(ax)

    xlim = (
        min(ax.get_xlim()[0] for ax in plotted_axes),
        max(ax.get_xlim()[1] for ax in plotted_axes),
    )
    ylim = (
        min(ax.get_ylim()[0] for ax in plotted_axes),
        max(ax.get_ylim()[1] for ax in plotted_axes),
    )
    for ax in plotted_axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    legend_ax.axis("off")
    layer_handles = [
        mpatches.Patch(facecolor=cmap(i), edgecolor="black", label=f"layer {l}")
        for i, l in enumerate(LAYERS)
    ]
    legend_ax.legend(
        handles=layer_handles + ref_handles,
        loc="center",
        fontsize=10,
        frameon=False,
    )

    fig.tight_layout()
    return fig


def _layer_set_label(layers: tuple[int, ...]) -> str:
    return ",".join(map(str, layers))


def _pareto_tags(layers: tuple[int, ...], lam: str) -> list[str]:
    """Run tags for one layer set at one lambda, matching sweep18 (single
    layer) or sweep19 (concatenated pair) naming."""
    if len(layers) == 1:
        prefix = f"sweep18_layer{layers[0]}"
    else:
        prefix = f"sweep19_layers{'-'.join(map(str, layers))}"
    suffix = _LAMBDA_SUFFIX[lam]
    return [f"{prefix}_lam{lam}_{suffix}_tr{i}" for i in range(10)]


def pareto_frontier_mask(losses: np.ndarray, aurocs: np.ndarray) -> np.ndarray:
    """True for each point not dominated by another: both loss and AUROC are
    lower-is-better, so i is dominated by j when j is no worse on either and
    strictly better on at least one."""
    n = len(losses)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        worse_or_equal = (losses <= losses[i]) & (aurocs <= aurocs[i])
        strictly_better = (losses < losses[i]) | (aurocs < aurocs[i])
        dominated = worse_or_equal & strictly_better
        dominated[i] = False
        if dominated.any():
            mask[i] = False
    return mask


def _bootstrap_frontier_band(
    pts: list[tuple[float, float, str]],
    xlim: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(grid, lo, hi): a BOOTSTRAP_PERCENTILES band over N_BOOTSTRAP
    frontiers, each built by resampling every lambda's trials with
    replacement independently (preserving each lambda's trial count) and
    recomputing the Pareto frontier from the resampled points. `grid` is
    log-spaced over xlim.

    A resample's own achieved-AUROC-by-loss curve only improves (or holds)
    as loss increases -- more budget can only unlock more points, never
    fewer -- so lo/hi, as percentiles of curves that are each individually
    monotonic, are themselves guaranteed non-increasing in loss. That
    guarantee depends on every resample contributing a value at every grid
    column: a resample with no point at all below a given loss is filled
    with AUROC 1.0 (no hiding whatsoever), the true worst case, rather than
    excluded as missing -- excluding it would let each column's percentile
    silently run over a different, inconsistently-sized subset of resamples,
    which is what broke monotonicity before this fix. Below this layer set's
    true (non-resampled) minimum loss, no resample could ever claim a point
    even in principle, so that stretch is left NaN (no shading) instead."""
    by_lam: dict[str, list[tuple[float, float]]] = {}
    for loss, auroc, lam in pts:
        by_lam.setdefault(lam, []).append((loss, auroc))
    real_min_loss = min(p[0] for p in pts)

    grid = np.geomspace(xlim[0], xlim[1], 200)
    ys = np.full((N_BOOTSTRAP, len(grid)), 1.0)
    for b in range(N_BOOTSTRAP):
        resampled = [
            lam_pts[i]
            for lam_pts in by_lam.values()
            for i in rng.integers(0, len(lam_pts), size=len(lam_pts))
        ]
        losses = np.array([p[0] for p in resampled])
        aurocs = np.array([p[1] for p in resampled])
        mask = pareto_frontier_mask(losses, aurocs)
        order = np.argsort(losses[mask])
        fx, fy = losses[mask][order], aurocs[mask][order]
        # Step lookup matching the "post" staircase: at each grid loss, the
        # best AUROC achieved by any resampled point with loss no higher.
        idx = np.searchsorted(fx, grid, side="right") - 1
        valid = idx >= 0
        ys[b, valid] = fy[idx[valid]]
        # idx < 0 (no resampled point that cheap) keeps the worst-case fill.

    lo, hi = np.percentile(ys, BOOTSTRAP_PERCENTILES, axis=0)
    in_range = grid >= real_min_loss
    lo = np.where(in_range, lo, np.nan)
    hi = np.where(in_range, hi, np.nan)
    return grid, lo, hi


def _collect_pareto_points(
    layer_sets: list[tuple[int, ...]], store: MetricStore
) -> dict[tuple[int, ...], list[tuple[float, float, str]]]:
    """Per layer set, every usable (loss, auroc, lambda) point across
    LAMBDAS' trials."""
    points: dict[tuple[int, ...], list[tuple[float, float, str]]] = {}
    for layers in layer_sets:
        layer_store = store.with_probe_layers(*layers)
        pts: list[tuple[float, float, str]] = []
        for lam in LAMBDAS:
            tags = _pareto_tags(layers, lam)
            n_ok = 0
            for tag in tags:
                if not layer_store.has_checkpoint(tag):
                    continue
                auroc = layer_store.auroc(tag)
                if auroc is None:
                    continue
                pts.append((layer_store.task_loss(tag), auroc, lam))
                n_ok += 1
            print(
                f"  layers={_layer_set_label(layers)} lam={lam}: {n_ok}/{len(tags)} usable"
            )
        points[layers] = pts
    return points


def _pareto_role_letters(layer_sets: list[tuple[int, ...]]) -> dict[int, str]:
    """Layer -> "A"/"B"/... in the order each first appears as a singleton in
    layer_sets (a multi-layer entry like (2, 4) contributes no new letter,
    just reuses its layers'). Lets the legend/title name roles ("layer A")
    rather than actual layer numbers, which repeat across PARETO_COMPARISONS
    entries (e.g. every entry's first layer set is (2,)) but don't mean the
    same thing panel to panel."""
    letters: dict[int, str] = {}
    for ls in layer_sets:
        if len(ls) == 1 and ls[0] not in letters:
            letters[ls[0]] = chr(ord("A") + len(letters))
    return letters


def _pareto_role_label(layers: tuple[int, ...], letters: dict[int, str]) -> str:
    return "layer " + ",".join(letters[l] for l in layers)


def _plot_pareto_panel(
    ax: plt.Axes,
    layer_sets: list[tuple[int, ...]],
    points: dict[tuple[int, ...], list[tuple[float, float, str]]],
    bootstrap: bool = False,
) -> tuple[list[Line2D], list[Line2D]]:
    """Draw one comparison's frontier onto `ax`. Returns (layer-set handles,
    lambda handles) rather than drawing a legend itself, so callers can place
    it per-axes (single-comparison figure) or pool it into one shared legend
    (grid figure)."""
    if not any(points.get(layers) for layers in layer_sets):
        ax.axis("off")
        return [], []

    ax.set_xscale(
        "log"
    )  # before any autoscaling, so the captured xlim below is in log space
    colors = plt.cm.tab10.colors

    # (color, sorted frontier losses, sorted frontier aurocs, its raw points)
    # per layer set, for the step lines (and, with --pareto-bootstrap, the
    # uncertainty band) drawn once every scatter is in and xlim is settled.
    frontiers: list[tuple[tuple, np.ndarray, np.ndarray, list]] = []
    for color, layers in zip(colors, layer_sets):
        pts = points.get(layers, [])
        if not pts:
            continue
        losses = np.array([p[0] for p in pts])
        aurocs = np.array([p[1] for p in pts])
        lams = np.array([p[2] for p in pts])
        on_frontier = pareto_frontier_mask(losses, aurocs)
        print(
            f"  layers={_layer_set_label(layers)}: "
            f"{on_frontier.sum()}/{len(pts)} points on frontier"
        )

        order = np.argsort(losses[on_frontier])
        frontiers.append(
            (color, losses[on_frontier][order], aurocs[on_frontier][order], pts)
        )

        for lam in LAMBDAS:
            lam_sel = lams == lam
            if not lam_sel.any():
                continue
            for frontier, alpha in (
                (True, FRONTIER_ALPHA),
                (False, OFF_FRONTIER_ALPHA),
            ):
                sel = lam_sel & (on_frontier == frontier)
                if not sel.any():
                    continue
                ax.scatter(
                    losses[sel],
                    aurocs[sel],
                    color=color,
                    marker=_LAMBDA_MARKER[lam],
                    alpha=alpha,
                    s=MARKER_SIZE,
                    edgecolor="black" if frontier else "none",
                    linewidth=0.7,
                )

    # The frontier itself: sorted by ascending loss, a non-dominated point's
    # AUROC is never worse than the previous one's (else it would be
    # dominated by it), so connecting them in that order with a step
    # function draws exactly the horizontal/vertical staircase that defines
    # the frontier. Drawn only after every scatter call so xlim/ylim reflect
    # the full data range. Extended to the plot edges on both ends -- up from
    # the lowest-loss point to the top (nothing beats it on loss, so nothing
    # to its left is excluded by it), and right from the highest-loss point
    # to the right edge (it's the best AUROC achieved from there on) --
    # rather than stopping at the extreme data points.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xlim = (
        xlim[0],
        PARETO_XLIM_MAX,
    )  # ~ the do-nothing task loss; nothing informative past it
    ylim = (ylim[0], 1.0)  # AUROC is bounded above by 1.0; don't pad past it

    if bootstrap:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for color, fx, _fy, pts in frontiers:
            if len(fx) == 0:
                continue
            grid, lo, hi = _bootstrap_frontier_band(pts, xlim, rng)
            ax.fill_between(
                grid, lo, hi, step="post", color=color, alpha=0.15, zorder=0
            )

    for color, fx, fy, _pts in frontiers:
        if len(fx) == 0:
            continue
        ax.plot(
            [fx[0], fx[0]], [ylim[1], fy[0]], color=color, alpha=0.8, lw=1.5, zorder=1
        )
        x_ext = np.concatenate([fx, [xlim[1]]])
        y_ext = np.concatenate([fy, [fy[-1]]])
        ax.step(x_ext, y_ext, where="post", color=color, alpha=0.8, lw=1.5, zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    letters = _pareto_role_letters(layer_sets)
    layer_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=8,
            label=_pareto_role_label(layers, letters),
        )
        for c, layers in zip(colors, layer_sets)
    ]
    lam_handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            color="black",
            linestyle="none",
            markersize=8,
            label=f"$\\lambda$={lam}",
        )
        for lam, m in _LAMBDA_MARKER.items()
    ]
    ax.grid(True, alpha=0.3)
    return layer_handles, lam_handles


def _pareto_comparison_title(layer_sets: list[tuple[int, ...]]) -> str:
    letters = _pareto_role_letters(layer_sets)
    return ", ".join(f"{letter}={layer}" for layer, letter in letters.items())


def _plot_pareto_frontier(
    layer_sets: list[tuple[int, ...]],
    points: dict[tuple[int, ...], list[tuple[float, float, str]]],
    bootstrap: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    layer_handles, lam_handles = _plot_pareto_panel(
        ax, layer_sets, points, bootstrap=bootstrap
    )
    # Both legends sit outside the axes, stacked to the right, so they never
    # overlap a data point -- unlike loc="lower left"/"lower right", nothing
    # about the frontier's shape is guaranteed to leave a corner free.
    layer_legend = ax.legend(
        handles=layer_handles,
        title="layer set",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
    )
    ax.add_artist(layer_legend)
    ax.legend(
        handles=lam_handles,
        title="$\\lambda$",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0 - 0.09 * (len(layer_handles) + 1.5)),
        fontsize=9,
    )

    ax.set_xlabel("task loss on training distribution")
    ax.set_ylabel("probe AUROC")
    title = "Pareto frontier: task loss vs. probe AUROC"
    if bootstrap:
        title += (
            f"\n(shaded: {BOOTSTRAP_PERCENTILES[0]}-{BOOTSTRAP_PERCENTILES[1]}th "
            "pctile over trial bootstrap)"
        )
    ax.set_title(title, fontsize=11 if bootstrap else 12)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    return fig


def _plot_pareto_grid(
    points_by_name: dict[str, dict[tuple[int, ...], list[tuple[float, float, str]]]],
    bootstrap: bool = False,
) -> plt.Figure:
    """All of PARETO_COMPARISONS as one 2x2 figure, one panel per comparison,
    with a single legend (layer-set and lambda handles pooled and
    deduplicated by label) shared across panels in one row at the bottom."""
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH * 1.3, FIG_WIDTH * 1.15))
    handles_by_label: dict[str, Line2D] = {}
    active_axes: list[plt.Axes] = []
    for ax, (name, layer_sets) in zip(axes.flat, PARETO_COMPARISONS.items()):
        layer_handles, lam_handles = _plot_pareto_panel(
            ax, layer_sets, points_by_name[name], bootstrap=bootstrap
        )
        if not layer_handles:
            continue
        active_axes.append(ax)
        ax.set_title(_pareto_comparison_title(layer_sets), fontsize=10)
        for h in layer_handles + lam_handles:
            handles_by_label.setdefault(h.get_label(), h)

    # Union each panel's (independently autoscaled) x/ylim across panels, so
    # frontiers are visually comparable by position rather than needing to
    # check each panel's axis range -- plt.subplots(sharex=True, sharey=True)
    # would do this too, but also hides interior tick labels, which panels
    # this wide apart still want each carrying its own.
    if active_axes:
        xlim = (
            min(ax.get_xlim()[0] for ax in active_axes),
            max(ax.get_xlim()[1] for ax in active_axes),
        )
        ylim = (
            min(ax.get_ylim()[0] for ax in active_axes),
            max(ax.get_ylim()[1] for ax in active_axes),
        )
        for ax in active_axes:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    for ax in axes[:, 0]:
        ax.set_ylabel("probe AUROC")
    for ax in axes[1, :]:
        ax.set_xlabel("task loss on training distribution")

    title = "Pareto frontier: task loss vs. probe AUROC"
    if bootstrap:
        title += (
            f"\n(shaded: {BOOTSTRAP_PERCENTILES[0]}-{BOOTSTRAP_PERCENTILES[1]}th "
            "pctile over trial bootstrap)"
        )
    fig.suptitle(title, fontsize=12)

    fig.legend(
        handles=list(handles_by_label.values()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(handles_by_label),
        fontsize=plt.rcParams["axes.labelsize"],  # match the panels' x/y labels
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


def plot_bars_and_scatter(store: MetricStore, plot: str) -> None:
    n_bands = BANDS.n_bands
    run_stats_by_group: dict[str, dict[int, RunStats]] = {}
    scatter_by_group: dict[str, list[tuple[float, float, int]]] = {}
    for group, group_tags in GROUPS.items():
        print(f"group={group}")
        run_stats, scatter_points = _collect_run_stats(group_tags, store, n_bands)
        if not run_stats:
            print(f"  group={group}: no usable runs, skipping")
            continue
        run_stats_by_group[group] = run_stats
        scatter_by_group[group] = scatter_points

    if not run_stats_by_group:
        raise SystemExit("no usable runs found across GROUPS")

    if plot in ("bars", "all"):
        fig = _plot_by_layer(run_stats_by_group)
        fig.savefig(f"{PLOT_DIR}/by_layer.png", bbox_inches="tight")
        fig.savefig(f"{PLOT_DIR}/by_layer.svg", bbox_inches="tight")

    if plot in ("scatter", "all"):
        fig = _plot_loss_vs_auroc(scatter_by_group)
        fig.savefig(f"{PLOT_DIR}/loss_vs_auroc.png", bbox_inches="tight")
        fig.savefig(f"{PLOT_DIR}/loss_vs_auroc.svg", bbox_inches="tight")


def plot_pareto(store: MetricStore, comparison: str | None, bootstrap: bool) -> None:
    suffix = "_bootstrap" if bootstrap else ""

    if comparison:
        layer_sets = PARETO_COMPARISONS[comparison]
        print(f"comparison={comparison}")
        points = _collect_pareto_points(layer_sets, store)
        if not any(points.values()):
            print(f"  no usable runs found, skipping {comparison}")
            return
        fig = _plot_pareto_frontier(layer_sets, points, bootstrap=bootstrap)
        fig.savefig(f"{PLOT_DIR}/{comparison}{suffix}.png", bbox_inches="tight")
        fig.savefig(f"{PLOT_DIR}/{comparison}{suffix}.svg", bbox_inches="tight")
        return

    # No --comparison: all of PARETO_COMPARISONS, as one 2x2 grid figure.
    points_by_name = {}
    for name, layer_sets in PARETO_COMPARISONS.items():
        print(f"comparison={name}")
        points_by_name[name] = _collect_pareto_points(layer_sets, store)

    if not any(any(pts) for pts in points_by_name.values()):
        print("  no usable runs found across PARETO_COMPARISONS, skipping")
        return

    fig = _plot_pareto_grid(points_by_name, bootstrap=bootstrap)
    fig.savefig(f"{PLOT_DIR}/pareto_grid{suffix}.png", bbox_inches="tight")
    fig.savefig(f"{PLOT_DIR}/pareto_grid{suffix}.svg", bbox_inches="tight")


def main(
    clear_cache: bool = False,
    plot: str = "all",
    comparison: str | None = None,
    pareto_bootstrap: bool = False,
) -> None:
    if clear_cache and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    store = MetricStore(SPEC, CACHE_PATH, DEVICE)
    os.makedirs(PLOT_DIR, exist_ok=True)

    if plot in ("bars", "scatter", "all"):
        plot_bars_and_scatter(store, plot)
    if plot in ("pareto", "all"):
        plot_pareto(store, comparison, pareto_bootstrap)

    plt.show()


if __name__ == "__main__":
    main(
        clear_cache=args.clear_cache,
        plot=args.plot,
        comparison=args.comparison,
        pareto_bootstrap=args.pareto_bootstrap,
    )
