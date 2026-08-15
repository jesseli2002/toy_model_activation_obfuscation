"""Ad-hoc comparison between groups of runs -- the worked example of how
sweeps got compared against each other while the research was happening,
for a comparison not (or not yet) covered by its own sweep*_analysis.py.

A group is a set of runs that differ only by trial/seed suffix, named in
GROUPS below. Edit GROUPS to point this at whatever you're comparing; the
three commented-out blocks are worked examples (model size, layer scan,
layer-pair comparison) left as templates. Two views, selected with --plot:

- curves: training-loss trajectories from each run's history.jsonl. Trials in
  a group share a color and are drawn transparent, with the per-iteration
  median opaque on top.
- scatter: per-run task loss against probe AUROC, recomputed from each run's
  checkpoint. One color per group.

Settings live in the constants below rather than a CLI -- which runs are
being compared changes far more often than the shape of the comparison.
Nothing is written to disk except the metrics cache; plt.show() only.
"""

import argparse

# Each group: label -> run tags. Vary only the trial suffix within a group;
# everything else about the group's runs should match. Each label keeps its
# own color across every plot.

# fmt: off

# Model size (sweep11/14/17): what happens with larger models?
# GROUPS = {
#     "nx32": [f"sweep17_lr0.0015_iter100k_lam0.1_tr{i}" for i in range(10)],
#     "nx64": [f"sweep11_lr0.0015_iter200k_lam0.1_tr{i}" for i in range(10)],
#     "nx128": [f"sweep14_lr0.0015_iter400k_lam0.1_tr{i}" for i in range(10)],
# }

# Layer scan (sweep18): what happens if you penalize different layers?
# Closest analogue to sweep_layer_analysis.py.
# GROUPS = {
#     "layer2": [f"sweep18_layer2_lam0.1_ramp200k_noise0.01_tr{i}" for i in range(10)],
#     "layer4": [f"sweep18_layer4_lam0.1_ramp200k_noise0.01_tr{i}" for i in range(10)],
#     "layer6": [f"sweep18_layer6_lam0.1_ramp200k_noise0.01_tr{i}" for i in range(10)],
#     "layer8": [f"sweep18_layer8_lam0.1_ramp200k_noise0.01_tr{i}" for i in range(10)],
#     "layer10": [f"sweep18_layer10_lam0.1_ramp200k_noise0.01_tr{i}" for i in range(10)],
# }

# Layer pairs (sweep18 vs. sweep19): what happens if you penalize multiple layers?
GROUPS = {
    "2": [f"sweep18_layer2_lam0.1_ramp200k_noise0.01_tr{i}" for i in range(10)],
    "8": [f"sweep18_layer8_lam0.1_ramp200k_noise0.01_tr{i}" for i in range(10)],
    "2,8": [f"sweep19_layers2-8_lam0.1_ramp200k_noise0.01_tr{i}" for i in range(10)],
}

# fmt: on

CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt

# Which layers to probe at, or None for each run's own penalized layers (concatenated).
# None means groups penalized at different widths get differently-sized probes -- their AUROCs aren't equal-capacity comparisons.
PROBE_LAYER = None

# -- curves
LOSS_LOWPASS_WINDOW = 1  # running mean of loss over the past this-many iters
LOSS_TYPE = "task"
TRIAL_ALPHA = 0.4
MEDIAN_LINEWIDTH = 2.5

# -- scatter: one plot per entry. Each entry lists task-loss metrics -- "task"
# (in-distribution) or an int N (N-hot OOD) -- and that plot's x axis is the
# worst (maximum) of a run's losses across the listed metrics.
LOSS_METRIC_PLOTS: list[list[str | int]] = [
    ["task"],
    # [1],
    # ["task", 1],
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plot",
        choices=["curves", "scatter", "both"],
        default="both",
        help="which view to draw (default: both). curves reads history.jsonl "
        "only; scatter recomputes metrics from checkpoints",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import matplotlib.pyplot as plt

from sweep_lib.history import causal_lowpass, load_history, median_trajectory
from sweep_lib.plots import Series, loss_vs_auroc

# Metrics actually needed, derived from LOSS_METRIC_PLOTS so nothing unused
# gets computed.
_NEEDED_METRICS = {m for plot in LOSS_METRIC_PLOTS for m in plot}
NEED_TASK_LOSS = "task" in _NEEDED_METRICS
NEED_N_HOT = sorted(m for m in _NEEDED_METRICS if m != "task")


def plot_curves() -> None:
    """Training-loss trajectories per group, with the group median on top."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10.colors

    for (label, tags), color in zip(GROUPS.items(), colors):
        trials = []
        for tag in tags:
            iters, losses = load_history(tag, LOSS_TYPE)
            filtered = causal_lowpass(iters, losses, LOSS_LOWPASS_WINDOW)
            trials.append((iters, filtered))
            ax.plot(iters, filtered, color=color, alpha=TRIAL_ALPHA, lw=1)

        median_iters, median_vals = median_trajectory(trials)
        ax.plot(
            median_iters,
            median_vals,
            color=color,
            alpha=1.0,
            lw=MEDIAN_LINEWIDTH,
            label=f"{label} (median, n={len(tags)})",
        )

    ax.set_xlabel("iter")
    ax.set_ylabel(f"{LOSS_TYPE} loss (smoothed over past {LOSS_LOWPASS_WINDOW} iters)")
    ax.set_yscale("log")
    ax.set_title("training loss by group")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def _collect_metrics() -> dict[str, dict]:
    """Every group's tags' metrics, computed (or loaded from cache) once up
    front -- shared across every LOSS_METRIC_PLOTS entry."""
    # Imported here rather than at module scope so --plot curves, which only
    # reads history.jsonl, doesn't pay for torch.
    import torch

    from sweep_lib.metrics import CACHE_PATH, MetricSpec, MetricStore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = MetricStore(
        MetricSpec(ckpt=CKPT, probe_layers=PROBE_LAYER), CACHE_PATH, device
    )

    metrics: dict[str, dict] = {}
    for tag in sorted({tag for tags in GROUPS.values() for tag in tags}):
        if not store.has_checkpoint(tag):
            print(f"  {tag}: no checkpoint yet, skipped")
            continue
        auroc = store.auroc(tag)
        if auroc is None:
            print(f"  {tag}: no probe in checkpoint, skipped")
            continue
        metrics[tag] = {
            "task_loss": store.task_loss(tag) if NEED_TASK_LOSS else None,
            "n_hot_losses": {n: store.n_hot_loss(tag, n) for n in NEED_N_HOT},
            "auroc": auroc,
        }
    return metrics


def _metric_value(entry: dict, metric: str | int) -> float:
    return entry["task_loss"] if metric == "task" else entry["n_hot_losses"][metric]


def _plot_group_scatter(metrics: dict[str, dict], loss_metrics: list[str | int]):
    """One scatter of (worst loss over loss_metrics, probe AUROC), one color
    per group. Returns `ax` so the caller can align xlim across plots."""
    losses, aurocs, labels = [], [], []
    for label, tags in GROUPS.items():
        for tag in tags:
            entry = metrics.get(tag)
            if entry is None:
                continue
            losses.append(max(_metric_value(entry, m) for m in loss_metrics))
            aurocs.append(entry["auroc"])
            labels.append(label)

    fig, ax = plt.subplots(figsize=(7, 5))
    loss_vs_auroc(
        ax, losses, aurocs, Series.categorical(labels), all_values=list(GROUPS)
    )

    metric_str = " / ".join(str(m) for m in loss_metrics)
    ax.set_xlabel(f"max task loss over [{metric_str}]")
    ax.set_title(f"Loss vs. AUROC ({CKPT} ckpt), loss = max({metric_str})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return ax


def plot_scatters() -> None:
    metrics = _collect_metrics()
    if not metrics:
        raise SystemExit("no usable runs found across GROUPS")

    axes = [
        _plot_group_scatter(metrics, loss_metrics) for loss_metrics in LOSS_METRIC_PLOTS
    ]

    # Same xlim on every plot, so loss is visually comparable across them --
    # taken from each axis's auto-scaled bounds rather than the raw data, so
    # matplotlib's usual log-scale padding is preserved.
    lo = min(ax.get_xlim()[0] for ax in axes)
    hi = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(lo, hi)


def main(plot: str = "both") -> None:
    if plot in ("curves", "both"):
        plot_curves()
    if plot in ("scatter", "both"):
        plot_scatters()
    plt.show()


if __name__ == "__main__":
    main(plot=args.plot)
