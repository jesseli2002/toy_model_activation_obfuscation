"""The figures the sweep reports draw repeatedly: stacked outcome bands, and
task loss against probe AUROC.

Each function takes an `ax` and returns nothing, leaving titles, axis labels
and saving to the caller -- those are what differ between reports.
"""

from dataclasses import dataclass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

import config
from analytic import reference_task_losses
from sweep_lib.outcomes import BandSpec

# Bands are drawn most-hidden at the bottom up to failed on top.
_BOTTOM_UP = slice(None, None, -1)

# Reference solutions plotted as vertical lines on loss_vs_auroc when
# show_loss_refs=True, matching adversarial_report._plot_training_traces.
_LOSS_REF_STYLES = [
    ("linreg", "linear regression", "--"),
    ("clamp", "c-unaware optimum", "-."),
]


def stacked_bars(
    ax,
    positions: list[float],
    fracs: list[Float[np.ndarray, " n_bands"]],
    bands: BandSpec,
    n_runs: list[int] | None = None,
    width: float = 0.6,
    legend: bool = False,
) -> None:
    """One stacked outcome bar per position, for a discrete x axis.

    `n_runs` annotates each bar with the number of runs it averages over.
    """
    labels, colors = bands.labels(), bands.colors()
    for i, (x, frac) in enumerate(zip(positions, fracs)):
        bottom = 0.0
        for band in range(bands.n_bands - 1, -1, -1):
            ax.bar(
                x,
                frac[band],
                bottom=bottom,
                width=width,
                color=colors[band],
                edgecolor="black",
                linewidth=0.4,
                label=labels[band] if legend and i == 0 else None,
            )
            bottom += frac[band]
        if n_runs is not None:
            ax.text(x, 1.01, f"n={n_runs[i]}", ha="center", va="bottom", fontsize=7)


def stacked_fractions(
    ax,
    x: Float[np.ndarray, " n"],
    fracs: list[Float[np.ndarray, " n_bands"]],
    bands: BandSpec,
) -> None:
    """Stacked outcome bands as filled areas, for a continuous x axis.

    Markers are drawn on each band's upper boundary, which stackplot has no
    way to do itself, so the sampled x positions stay visible.
    """
    order = np.argsort(x)
    x = np.asarray(x)[order]
    stacked = np.array(fracs)[order].T[_BOTTOM_UP]  # (n_bands, n_x)
    labels = bands.labels()[_BOTTOM_UP]
    colors = bands.colors()[_BOTTOM_UP]

    ax.stackplot(x, *stacked, labels=labels, colors=colors, alpha=0.9)

    cum = np.cumsum(stacked, axis=0)
    for row, color in zip(cum[:-1], colors[:-1]):  # the top row is always 1
        ax.plot(
            x,
            row,
            marker="o",
            ls="-",
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, x[-1])


@dataclass
class Series:
    """How to color the points of a loss-vs-AUROC scatter.

    Three shapes, because the sweeps vary in what they hold the points apart
    by: a continuous quantity, a handful of ordered values, or unordered
    labels. Build one with `continuous`, `ordinal` or `categorical`.
    """

    values: np.ndarray
    label: str
    kind: str  # "continuous" | "ordinal" | "categorical"

    @classmethod
    def continuous(cls, values, label: str) -> "Series":
        """A colorbar over a continuous quantity, on a symlog scale so a zero
        value still gets a color."""
        return cls(np.asarray(values), label, "continuous")

    @classmethod
    def ordinal(cls, values, label: str) -> "Series":
        """A discrete colorbar over a few ordered values. Discrete rather than
        continuous so a swept setting's handful of values don't read as
        interpolatable."""
        return cls(np.asarray(values), label, "ordinal")

    @classmethod
    def categorical(cls, values, label: str = "") -> "Series":
        """A legend over unordered labels, in the default color cycle -- for
        values with no meaningful ordering, where a ramp would imply one."""
        return cls(np.asarray(values), label, "categorical")


def loss_vs_auroc(
    ax,
    losses: Float[np.ndarray, " n"],
    aurocs: Float[np.ndarray, " n"],
    series: Series,
    loss_threshold: float | None = None,
    all_values=None,
    show_loss_refs: bool = False,
) -> None:
    """Scatter of per-run (task loss, probe AUROC), colored by `series`.

    `all_values` fixes the color scale across several figures drawn from
    different subsets, so a given series value keeps its color between them.

    `show_loss_refs` adds vertical lines at the analytic reference losses
    (do-nothing, linear regression, c-unaware optimum) from
    analytic.reference_task_losses, for scale against the training-loss
    thresholds a run has to clear.
    """
    losses, aurocs = np.asarray(losses), np.asarray(aurocs)
    fig = ax.get_figure()
    scale_values = np.asarray(all_values) if all_values is not None else series.values

    if series.kind == "categorical":
        # First-seen order, not sorted: the caller's order is what fixes which
        # value gets which color, and sorting would reorder e.g. nx128 ahead of
        # nx32.
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, value in enumerate(_in_order(scale_values)):
            mask = series.values == value
            if not mask.any():
                continue
            ax.scatter(
                losses[mask],
                aurocs[mask],
                color=cycle[i % len(cycle)],
                edgecolor="black",
                linewidth=0.5,
                label=f"{value} (n={int(mask.sum())})",
            )
    elif series.kind == "ordinal":
        uniq = _unique(scale_values)
        cmap = plt.get_cmap("viridis", len(uniq))
        norm = mcolors.BoundaryNorm(np.arange(len(uniq) + 1) - 0.5, len(uniq))
        sc = ax.scatter(
            losses,
            aurocs,
            c=np.searchsorted(uniq, series.values),
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.5,
        )
        cbar = fig.colorbar(sc, ax=ax, ticks=range(len(uniq)), label=series.label)
        cbar.set_ticklabels([str(v) for v in uniq])
    else:
        positive = scale_values[scale_values > 0]
        norm = mcolors.SymLogNorm(
            linthresh=positive.min() if positive.size else 1.0,
            vmin=scale_values.min(),
            vmax=scale_values.max(),
        )
        sc = ax.scatter(
            losses,
            aurocs,
            c=series.values,
            cmap="viridis",
            norm=norm,
            edgecolor="black",
            linewidth=0.5,
        )
        fig.colorbar(sc, ax=ax, label=series.label)

    ax.set_ylabel("probe AUROC")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    # Log-scale minor tick labels (2x, 3x, ...) crowd together when the data
    # spans less than a decade; rotating keeps them legible at any range.
    plt.setp(ax.get_xticklabels(which="both"), rotation=45, ha="right")

    if loss_threshold is not None:
        ax.axvline(
            loss_threshold,
            linestyle="--",
            color="red",
            label="loss threshold",
            alpha=0.5,
        )

    if show_loss_refs:
        ref_losses = reference_task_losses(
            config.X_LOW, config.X_HIGH, config.C_LOW, config.C_HIGH
        )
        for key, label, style in _LOSS_REF_STYLES:
            ax.axvline(ref_losses[key], color="gray", ls=style, label=label, zorder=0)


def _unique(values) -> list:
    """Distinct values, ascending -- for scales where the order is the data's."""
    return sorted(set(np.asarray(values).tolist()))


def _in_order(values) -> list:
    """Distinct values in first-seen order -- for scales where the order is the
    caller's."""
    return list(dict.fromkeys(np.asarray(values).tolist()))
