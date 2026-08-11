"""Throwaway: scatter task loss vs. probe AUROC across ad-hoc groups of
runs, for a preliminary look at a sweep not yet covered by a stabilized
sweep*_analysis.py. GROUPS pattern from sweep_compare_groups.tmp.py; the
recompute/cache/scatter machinery is adapted from sweep7/8_analysis.py's
loss-vs-AUROC scatter.

Settings live in constants below rather than a CLI -- this script's shape
is still changing. Nothing written to disk except the metrics cache;
plt.show() only.
"""

CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt

# label -> run tags. Each label gets its own tableau color across every plot.
# sweep15 (layer 10 behaviour)
GROUPS = {
    f"lam{lam}/warm{warm}": [
        f"sweep15_layer10_lam{lam}_warm{warm}{warm}_lr0.003_tr{i}" for i in range(5)
    ]
    for lam, warm in [("0.1", "10k"), ("0.01", "10k"), ("0.1", "20k"), ("0.01", "20k")]
} | {
    f"lam0.1/warm0": [f"sweep13_layer10_lam0.1_tr{i}" for i in range(10)],
    f"lam0.01/warm0": [f"sweep13_layer10_lam0.01_tr{i}" for i in range(10)],
}

# # sweep 7/11/14 -> model sizes
# GROUPS = {
#     "nx32": [f"sweep7_lam0.01_tr{i}" for i in range(15)],
#     "nx64": [f"sweep11_lr0.0015_iter200k_lam0.01_tr{i}" for i in range(10)],
#     "nx128": [f"sweep14_lr0.0015_iter400k_lam0.01_tr{i}" for i in range(10)],
# }


# One scatter plot per entry. Each entry is a list of task-loss metrics --
# "task" (in-distribution, data.eval_task_loss) or an int N (N-hot OOD,
# data.eval_n_hot_loss) -- and that plot's x-axis is the *maximum* (worst
# case) of a run's losses across the listed metrics.
LOSS_METRIC_PLOTS: list[list[str | int]] = [
    ["task"],
    # # [1],
    # ["task", 1],
]

PROBE_LAYER = 10  # matches adversarial.penalty_layers in these runs' config.json

import matplotlib.pyplot as plt
import torch

from sweep_lib.metrics import CACHE_PATH, MetricSpec, MetricStore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPEC = MetricSpec(ckpt=CKPT, probe_layer=PROBE_LAYER)

# Metrics actually needed, derived from LOSS_METRIC_PLOTS so nothing unused
# gets computed.
_NEEDED_METRICS = {m for plot in LOSS_METRIC_PLOTS for m in plot}
NEED_TASK_LOSS = "task" in _NEEDED_METRICS
NEED_N_HOT = sorted(m for m in _NEEDED_METRICS if m != "task")


def _collect_metrics() -> dict[str, dict]:
    """Every group's tags' metrics, computed (or loaded from cache) once up
    front -- shared across every LOSS_METRIC_PLOTS entry below."""
    store = MetricStore(SPEC, CACHE_PATH, DEVICE)

    tags = sorted({tag for tags in GROUPS.values() for tag in tags})
    metrics: dict[str, dict] = {}
    for tag in tags:
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
    """One scatter of (max loss over loss_metrics, probe AUROC), one color
    per GROUPS label. Returns `ax` so main() can read/align xlim across
    every plot after the fact."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10.colors

    for (label, tags), color in zip(GROUPS.items(), colors):
        xs, ys = [], []
        for tag in tags:
            entry = metrics.get(tag)
            if entry is None:
                continue
            xs.append(max(_metric_value(entry, m) for m in loss_metrics))
            ys.append(entry["auroc"])
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=f"{label} (n={len(xs)})",
        )

    metric_str = " / ".join(str(m) for m in loss_metrics)
    ax.set_xlabel(f"max task loss over [{metric_str}]")
    ax.set_ylabel("probe AUROC")
    ax.set_title(f"Loss vs. AUROC ({CKPT} ckpt), loss = max({metric_str})")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(which="both"), rotation=45, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return ax


def main() -> None:
    metrics = _collect_metrics()
    if not metrics:
        raise SystemExit("no usable runs found across GROUPS")

    axes = [
        _plot_group_scatter(metrics, loss_metrics) for loss_metrics in LOSS_METRIC_PLOTS
    ]

    # Same xlim on every plot, so loss is visually comparable across them --
    # pulled from each axis's own (auto-scaled) bounds rather than the raw
    # data, so it still respects matplotlib's usual log-scale padding.
    lo = min(ax.get_xlim()[0] for ax in axes)
    hi = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(lo, hi)

    plt.show()


if __name__ == "__main__":
    main()
