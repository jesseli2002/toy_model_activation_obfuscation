"""sweep7_analysis.py's outcome breakdown, for runs/sweep8_* -- a sweep over
model size at three lambdas, plus a side ray over model size at one lambda.

Same per-run classification as sweep7_analysis: a run either failed the task
(loss) or, having succeeded, is binned by how well it hid from the
layer-PROBE_LAYER linear probe (AUROC). sweep8 has two distinct rays through
(num_x, d_model, d_mlp, lambda) space, each answering a different question,
so each gets its own plot(s) rather than one generic size x lambda grid:

- Main sweep: num_x:d_model:d_mlp held at 2:4:1, lambda in {0.001, 0.01,
  0.1} -- "does hiding change with model size, at a given lambda?" One
  plot per lambda.
- Side ray: d_mlp fixed at 16, num_x (and d_model = 2*num_x) varying, at
  lam=0.01 -- "does hiding change with size alone, off the main ratio?"
  One plot.
- Main sweep, per-run: the same main-sweep runs as unbinned (task loss,
  AUROC) points, colored by num_x -- see sweep7_analysis's analogous
  loss-vs-AUROC scatter. One plot per lambda.

Each ray also has lam=0 controls, which aren't plotted (a run that always
succeeds and never hides is the same bar at every size) but are checked by
_assert_lam0_clean, which raises if one didn't behave like a control.

The (32, 64, 16) point sits on both rays; runs/sweep7_* already covers it
(see SWEEP7_*), so sweep8 didn't retrain it and this script pulls that data
in instead. It shows up on both the relevant main-sweep plot and the side
ray.

Bars, not sweep7_analysis's stackplot: sweep8 samples only a handful of
lambdas, too few for a filled area between them to mean anything.

Lambda coverage is deliberately uneven across sizes (only some sizes get the
full lambda grid), so a missing bar means no runs, not zero runs surviving.

LOSS_THRESHOLD / AUROC_THRESHOLDS are provisional -- a separate exercise is
picking the final values -- so they're constants here, not a CLI, alongside
everything else that may still change.

"Loss" and "AUROC" are both freshly recomputed at CKPT -- loss via
data.eval_task_loss (deliberately excluding the adversarial/probe penalty,
since this only cares about task performance), and AUROC via a freshly refit
probe at PROBE_LAYER (fit and scored under the same residual-stream noise,
see PROBE_*_NOISE_MULT) -- rather than read from history.jsonl or the
checkpoint's own stored training-time probe. See sweep7_analysis.py and
sweep_threshold_report.py, which recompute the same way for the same reasons
(some runs' final checkpoint predates their last history record).

Recomputing both over a whole sweep is slow, so results are cached to
CACHE_PATH keyed by the settings they depend on; changing any of those
settings misses the cache rather than reusing a stale value. Pass
--clear-cache to force a full recompute.
"""

import argparse
import re

PLOT_DIR = "plot/sweep8"
RUN_GLOB = "sweep8_nx*_tr*"
TAG_RE = re.compile(r"sweep8_nx(\d+)_dm(\d+)_mlp(\d+)_lam([0-9\.]+)_tr(\d+)$")

# sweep7 ran (32, 64, 16) -- the point shared by the main sweep and the side
# ray -- across more lambdas than sweep8 repeated; pull the matching ones in.
SWEEP7_GLOB = "sweep7_lam*_tr*"
SWEEP7_TAG_RE = re.compile(r"sweep7_lam([0-9\.]+)_tr(\d+)$")
SWEEP7_SIZE = (32, 64, 16)
SWEEP7_LAMBDAS = {0.0, 0.001, 0.01, 0.1}  # main-sweep lambdas + control

MAIN_LAMBDAS = [0.001, 0.01, 0.1]  # one plot each
SIDE_LAMBDA = 0.01  # the side ray's one non-control plot

EXCLUDE_LAMBDAS: set[float] = set()
EXCLUDE_SIZES: set[tuple[int, int, int]] = set()  # (num_x, d_model, d_mlp)
MIN_RUNS = 1  # drop (size, lambda) points with fewer usable runs than this
CKPT = "best"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
PROBE_LAYER = 2  # matches adversarial.penalty_layers in these runs' config.json

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

SPEC = MetricSpec(ckpt=CKPT, probe_layer=PROBE_LAYER)

RunKey = tuple[int, int, int, float]  # (num_x, d_model, d_mlp, lam)

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


def _is_main_sweep(key: RunKey) -> bool:
    num_x, d_model, d_mlp, _ = key
    return d_model == 2 * num_x and d_mlp == num_x // 2


def _is_side_ray(key: RunKey) -> bool:
    num_x, d_model, d_mlp, _ = key
    return d_mlp == 16 and d_model == 2 * num_x


def _size_lam_of(m: re.Match) -> RunKey | None:
    size = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    lam = float(m.group(4))
    if lam in EXCLUDE_LAMBDAS or size in EXCLUDE_SIZES:
        return None
    return (*size, lam)


def _sweep7_key_of(m: re.Match) -> RunKey | None:
    lam = float(m.group(1))
    if lam not in SWEEP7_LAMBDAS or lam in EXCLUDE_LAMBDAS:
        return None
    if SWEEP7_SIZE in EXCLUDE_SIZES:
        return None
    return (*SWEEP7_SIZE, lam)


def _discover_tags() -> dict[RunKey, list[str]]:
    """Run tags grouped by the (num_x, d_model, d_mlp, lambda) they belong
    to, from both runs/sweep8_* and the matching runs/sweep7_* points."""
    by_key = group_tags(RUN_GLOB, TAG_RE, _size_lam_of)
    for key, tags in group_tags(SWEEP7_GLOB, SWEEP7_TAG_RE, _sweep7_key_of).items():
        by_key.setdefault(key, []).extend(tags)
    return by_key


def _collect_run_stats(
    n_bands: int,
) -> tuple[dict[RunKey, RunStats], list[tuple[float, float, int, float]]]:
    """Per-run-config band fractions and usable-run count, recomputing each
    run's loss/AUROC where the cache doesn't already have them, plus each
    usable main-sweep run's (loss, auroc, num_x, lam) for the scatter."""
    by_key = _discover_tags()
    store = MetricStore(SPEC, CACHE_PATH, DEVICE)

    stats: dict[RunKey, RunStats] = {}
    scatter_points: list[tuple[float, float, int, float]] = []
    print(f"{'size':>18s} {'lambda':>8s} {'n_ok':>5s} {'n_total':>7s}")
    for key in sorted(by_key):
        num_x, d_model, d_mlp, lam = key
        tags = by_key[key]
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
            band = BANDS.classify(loss, {1: store.n_hot_loss(tag, 1)}, auroc)
            counts[band] += 1
            n_ok += 1
            if _is_main_sweep(key) and lam in MAIN_LAMBDAS:
                scatter_points.append((loss, auroc, num_x, lam))
        if n_ok < MIN_RUNS:
            print(f"  {key}: {n_ok} usable runs, skipped")
            continue
        stats[key] = RunStats(counts / n_ok, n_ok)
        size_str = f"nx{num_x} dm{d_model} mlp{d_mlp}"
        print(f"{size_str:>18s} {lam:8g} {n_ok:5d} {len(tags):7d}")
    return stats, scatter_points


def _assert_lam0_clean(
    run_stats: dict[RunKey, RunStats], sizes: list[tuple[int, int, int]], context: str
) -> None:
    """lam=0 is a control expected to always succeed the task and never
    hide from the probe -- it carries no information of its own, so rather
    than plot it alongside the swept lambdas (which was tried and just
    showed the same bar every time), this checks it's actually behaving
    like a control instead of silently trusting that."""
    for size in sizes:
        base = run_stats.get((*size, 0.0))
        if base is None:
            continue
        # bucket 0 = failed, bucket 1 = succeeded and not hidden.
        if base.frac[0] > 0 or base.frac[1] < 1.0:
            raise AssertionError(
                f"{context} lam=0 control at size {size} isn't clean "
                f"(expected all succeeded and not hidden, got frac={base.frac}); "
                "needs investigating before trusting this plot."
            )


def _plot_main_sweep(
    run_stats: dict[RunKey, RunStats], lam: float, n_bands: int
) -> plt.Figure:
    """One panel: bars per main-sweep size at `lam`. lam=0 controls aren't
    plotted -- they're asserted clean instead (see _assert_lam0_clean)."""
    sizes = sorted({k[:3] for k in run_stats if _is_main_sweep(k)}, key=lambda s: s[0])
    _assert_lam0_clean(run_stats, sizes, context="main sweep")

    fig, ax = plt.subplots(figsize=(1.2 * len(sizes) + 1.5, 4.2))
    lam_stats = [run_stats[(*s, lam)] for s in sizes if (*s, lam) in run_stats]
    lam_pos = [i for i, s in enumerate(sizes) if (*s, lam) in run_stats]
    stacked_bars(
        ax,
        lam_pos,
        [s.frac for s in lam_stats],
        BANDS,
        n_runs=[s.n_ok for s in lam_stats],
        legend=True,
    )

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s[0]) for s in sizes])
    ax.set_xlim(-0.6, len(sizes) - 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("num_x (d_model=2*num_x, d_mlp=num_x/2)")
    ax.set_ylabel("fraction of runs")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"Main sweep, $\\lambda$ = {lam:g}", fontsize=11, pad=14)

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


def _plot_side_ray(run_stats: dict[RunKey, RunStats], n_bands: int) -> plt.Figure:
    """One panel: bars per side-ray size at SIDE_LAMBDA. lam=0 controls
    aren't plotted -- they're asserted clean instead (see
    _assert_lam0_clean)."""
    sizes = sorted({k[:3] for k in run_stats if _is_side_ray(k)}, key=lambda s: s[0])
    _assert_lam0_clean(run_stats, sizes, context="side ray")

    fig, ax = plt.subplots(figsize=(1.2 * len(sizes) + 1.5, 4.2))
    stats_at = [
        run_stats[(*s, SIDE_LAMBDA)] for s in sizes if (*s, SIDE_LAMBDA) in run_stats
    ]
    pos = [i for i, s in enumerate(sizes) if (*s, SIDE_LAMBDA) in run_stats]
    stacked_bars(
        ax,
        pos,
        [s.frac for s in stats_at],
        BANDS,
        n_runs=[s.n_ok for s in stats_at],
        legend=True,
    )

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s[0]) for s in sizes])
    ax.set_xlim(-0.6, len(sizes) - 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("num_x (d_mlp=16, d_model=2*num_x)")
    ax.set_ylabel("fraction of runs")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"Side ray, $\\lambda$ = {SIDE_LAMBDA:g}", fontsize=11, pad=14)

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


def _plot_loss_vs_auroc_by_size(points: list[tuple[float, float, int, float]]) -> None:
    """One scatter per lambda of that lambda's main-sweep runs' (task loss,
    probe AUROC), colored by num_x -- the same recomputed metrics behind the
    outcome bars, viewed per-run instead of binned. lam=0 controls are
    excluded, matching _plot_main_sweep.

    The color scale spans every lambda's sizes, not just the one being
    plotted, so a given num_x keeps its color across the three figures."""
    losses, aurocs, sizes, lams = (np.array(v) for v in zip(*points))

    for lam in MAIN_LAMBDAS:
        mask = lams == lam
        if not mask.any():
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        loss_vs_auroc(
            ax,
            losses[mask],
            aurocs[mask],
            Series.ordinal(sizes[mask], "num_x"),
            LOSS_THRESHOLD,
            all_values=sizes,
        )
        ax.set_xlabel("task loss")
        ax.set_title(f"Main sweep, $\\lambda$ = {lam:g}: task loss vs. probe AUROC")
        ax.legend()
        fig.savefig(
            f"{PLOT_DIR}/main_loss_vs_auroc_scatter_lam{lam:g}.png", bbox_inches="tight"
        )


def main(clear_cache: bool = False) -> None:
    if clear_cache and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    n_bands = BANDS.n_bands

    run_stats, scatter_points = _collect_run_stats(n_bands)
    if not run_stats:
        raise SystemExit(f"no usable runs matched runs/{RUN_GLOB}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    for lam in MAIN_LAMBDAS:
        fig = _plot_main_sweep(run_stats, lam, n_bands)
        fig.savefig(f"{PLOT_DIR}/main_lam{lam:g}.png", bbox_inches="tight")

    fig = _plot_side_ray(run_stats, n_bands)
    fig.savefig(f"{PLOT_DIR}/side_ray.png", bbox_inches="tight")

    _plot_loss_vs_auroc_by_size(scatter_points)

    plt.show()


if __name__ == "__main__":
    main(clear_cache=args.clear_cache)
