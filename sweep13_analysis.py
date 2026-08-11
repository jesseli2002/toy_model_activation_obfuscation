"""sweep7_analysis.py's outcome breakdown, for runs/sweep13_* -- a sweep over
*which* layer the probe penalty is applied to, at two lambdas.

Every other sweep penalized (and probed) layer 2; sweep13 holds the model
fixed and moves that single penalized layer through {2, 4, 6, 8, 10} of a
12-block model. So unlike the rest of the sweep_* family, PROBE_LAYER is not
a constant here -- each run is probed at its own penalized layer, read off
the tag and cross-checked against the checkpoint's
adversarial.penalty_layers. Comparing a layer-8 run
against a layer-2 probe would answer a different question entirely.

Two views, each one plot per lambda:

- Outcome bars: per-layer stacked fractions, x-axis = penalized layer. A run
  either failed the task (loss) or, having succeeded, is binned by how well
  it hid from the linear probe at its own penalized layer (AUROC). Bars, not
  sweep7_analysis's stackplot: "layer" is discrete and unordered-ish, with no
  meaningful interpolation between neighbouring values.
- Loss-vs-AUROC scatter: per-run (task loss on the training distribution,
  probe AUROC), for SCATTER_LAYERS only. All five layers at once is
  unreadable, and since the layers form no clear gradient the subset is
  hand-picked and colored with matplotlib's default categorical cycle rather
  than a sequential ramp.

sweep13 has no lam=0 controls -- runs/sweep7_lam0_* etc. cover that, and a
control has no penalized layer to sweep in the first place.

LOSS_THRESHOLD / AUROC_THRESHOLDS are provisional -- a separate exercise is
picking the final values -- so they're constants here, not a CLI, alongside
everything else that may still change.

"Loss" and "AUROC" are both freshly recomputed at CKPT -- loss via
data.eval_task_loss (deliberately excluding the adversarial/probe penalty,
since this only cares about task performance), and AUROC via a freshly refit
probe (fit and scored under the same residual-stream noise, see PROBE_*_NOISE_MULT)
-- rather than read from history.jsonl or the checkpoint's own stored
training-time probe. See sweep7_analysis.py and sweep8_analysis.py, which
recompute the same way for the same reasons (some runs' final checkpoint
predates their last history record).

Recomputing over a whole sweep is slow, so results are cached to CACHE_PATH
keyed by the settings they depend on; changing any of those settings misses
the cache rather than reusing a stale value. Pass --clear-cache to force a
full recompute.
"""

import argparse
import re

PLOT_DIR = "plot/sweep13"
RUN_GLOB = "sweep13_layer*_tr*"
TAG_RE = re.compile(r"sweep13_layer(\d+)_lam([0-9\.]+)_tr(\d+)$")

LAMBDAS = [0.01, 0.1]  # one bar plot and one scatter each
SCATTER_LAYERS = [2, 4, 6, 8, 10]  # hand-picked subset; see module docstring

EXCLUDE_LAYERS: set[int] = set()
MIN_RUNS = 1  # drop (layer, lambda) points with fewer usable runs than this
CKPT = "best"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
# Multipliers on the checkpoint's own resid_noise_std for the refit probe's fit
# and scoring passes. Both 1.0: these runs trained with probe_noise, so the
# adversary they actually hid from was itself fit under noise -- fitting clean
# measures a different, weaker probe.
PROBE_TRAIN_NOISE_MULT = 1.0
PROBE_EVAL_NOISE_MULT = 1.0

# task "succeeded" iff both are below their threshold -- see
# sweep_threshold_report.py for what distinguishes the two losses (in-
# distribution vs. one-hot-OOD input construction).
LOSS_THRESHOLD = 0.01
ONE_HOT_LOSS_THRESHOLD = 0.01
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
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep_lib.discovery import group_tags
from sweep_lib.metrics import CACHE_PATH, MetricSpec, MetricStore
from sweep_lib.outcomes import BandSpec
from sweep_lib.plots import Series, loss_vs_auroc, stacked_bars

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# probe_layer=None: each run is probed at its own penalized layer
SPEC = MetricSpec(ckpt=CKPT)

RunKey = tuple[int, float]  # (penalized layer, lam)

BANDS = BandSpec(
    loss_threshold=LOSS_THRESHOLD,
    n_hot_loss_threshold=ONE_HOT_LOSS_THRESHOLD,
    n_hot_values=(1,),
    auroc_thresholds=AUROC_THRESHOLDS,
)


class RunStats(NamedTuple):
    """A run config's outcome-band fractions, aggregated over its trials."""

    frac: np.ndarray  # length n_bands, sums to 1
    n_ok: int  # usable trials this is averaged over


def _layer_lam_of(m: re.Match) -> RunKey | None:
    layer, lam = int(m.group(1)), float(m.group(2))
    return None if layer in EXCLUDE_LAYERS else (layer, lam)


def _discover_tags() -> dict[RunKey, list[str]]:
    """Run tags grouped by the (penalized layer, lambda) they belong to."""
    return group_tags(RUN_GLOB, TAG_RE, _layer_lam_of)


def _collect_run_stats(
    n_bands: int,
) -> tuple[dict[RunKey, RunStats], list[tuple[float, float, int, float]]]:
    """Per-(layer, lambda) band fractions and usable-run count, recomputing
    each run's loss/AUROC where the cache doesn't already have them, plus
    every usable run's (loss, auroc, layer, lam) for the scatter."""
    by_key = _discover_tags()
    store = MetricStore(SPEC, CACHE_PATH, DEVICE)

    stats: dict[RunKey, RunStats] = {}
    scatter_points: list[tuple[float, float, int, float]] = []
    print(f"{'layer':>6s} {'lambda':>8s} {'n_ok':>5s} {'n_total':>7s}")
    for key in sorted(by_key):
        layer, lam = key
        tags = by_key[key]
        # The tag's layer sets this run's x position, so pin it: a checkpoint
        # trained at a different layer raises rather than landing in the
        # wrong column with an AUROC measured somewhere else.
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
            band = BANDS.classify(loss, {1: layer_store.n_hot_loss(tag, 1)}, auroc)
            counts[band] += 1
            n_ok += 1
            scatter_points.append((loss, auroc, layer, lam))
        if n_ok < MIN_RUNS:
            print(f"  {key}: {n_ok} usable runs, skipped")
            continue
        stats[key] = RunStats(counts / n_ok, n_ok)
        print(f"{layer:6d} {lam:8g} {n_ok:5d} {len(tags):7d}")
    return stats, scatter_points


def _plot_by_layer(
    run_stats: dict[RunKey, RunStats], lam: float, n_bands: int
) -> plt.Figure:
    """One panel: bars per penalized layer at `lam`."""
    layers = sorted({k[0] for k in run_stats})

    fig, ax = plt.subplots(figsize=(1.2 * len(layers) + 1.5, 4.2))
    lam_stats = [run_stats[(ly, lam)] for ly in layers if (ly, lam) in run_stats]
    lam_pos = [i for i, ly in enumerate(layers) if (ly, lam) in run_stats]
    stacked_bars(
        ax,
        lam_pos,
        [s.frac for s in lam_stats],
        BANDS,
        n_runs=[s.n_ok for s in lam_stats],
        legend=True,
    )

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(ly) for ly in layers])
    ax.set_xlim(-0.6, len(layers) - 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Probed layer")
    ax.set_ylabel("fraction of runs")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"Outcome vs probed layer, $\\lambda$ = {lam:g}", fontsize=11, pad=14)

    handles, leg_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        leg_labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def _plot_loss_vs_auroc_by_layer(
    points: list[tuple[float, float, int, float]],
) -> None:
    """One scatter per lambda of that lambda's runs' (training-distribution
    task loss, probe AUROC), colored by penalized layer -- the same
    recomputed metrics behind the outcome bars, viewed per-run instead of
    binned.

    Only SCATTER_LAYERS are drawn, in matplotlib's default categorical
    colors: the swept layers form no clear gradient (so a sequential ramp
    would imply an ordering the data doesn't support) and all of them at
    once overplots into mush."""
    losses, aurocs, layers, lams = (np.array(v) for v in zip(*points))
    plot_layers = [ly for ly in SCATTER_LAYERS if ly in set(layers.tolist())]

    for lam in LAMBDAS:
        if not (lams == lam).any():
            continue
        keep = (lams == lam) & np.isin(layers, plot_layers)
        fig, ax = plt.subplots(figsize=(6, 5))
        loss_vs_auroc(
            ax,
            losses[keep],
            aurocs[keep],
            Series.categorical([f"layer {ly}" for ly in layers[keep]]),
            LOSS_THRESHOLD,
            all_values=[f"layer {ly}" for ly in plot_layers],
        )
        ax.set_xlabel("task loss on training distribution")
        ax.set_ylabel("probe AUROC (at each run's penalized layer)")
        ax.set_title(f"$\\lambda$ = {lam:g}: task loss vs. probe AUROC")
        ax.legend()
        fig.savefig(
            f"{PLOT_DIR}/loss_vs_auroc_scatter_lam{lam:g}.png", bbox_inches="tight"
        )


def main(clear_cache: bool = False) -> None:
    if clear_cache and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    n_bands = BANDS.n_bands

    run_stats, scatter_points = _collect_run_stats(n_bands)
    if not run_stats:
        raise SystemExit(f"no usable runs matched runs/{RUN_GLOB}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    for lam in LAMBDAS:
        fig = _plot_by_layer(run_stats, lam, n_bands)
        fig.savefig(f"{PLOT_DIR}/by_layer_lam{lam:g}.png", bbox_inches="tight")

    _plot_loss_vs_auroc_by_layer(scatter_points)

    plt.show()


if __name__ == "__main__":
    main(clear_cache=args.clear_cache)
