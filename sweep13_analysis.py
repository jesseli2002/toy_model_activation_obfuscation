"""sweep7_analysis.py's outcome breakdown, for runs/sweep13_* -- a sweep over
*which* layer the probe penalty is applied to, at two lambdas.

Every other sweep penalized (and probed) layer 2; sweep13 holds the model
fixed and moves that single penalized layer through {2, 4, 6, 8, 10} of a
12-block model. So unlike the rest of the sweep_* family, PROBE_LAYER is not
a constant here -- each run is probed at its own penalized layer, read off
the tag and cross-checked against the checkpoint's
adversarial.penalty_layers (see _probe_layer). Comparing a layer-8 run
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

SMOKE = False  # if True, skip the real analysis and plot synthetic data instead
# -- lets you preview the plot's layout without waiting on a full sweep's
# worth of checkpoint loading / task-loss / probe recomputation.

PLOT_DIR = "plot/sweep13"
CACHE_PATH = "plot/sweep13/metrics_cache.json"
RUN_GLOB = "sweep13_layer*_tr*"
TAG_RE = re.compile(r"sweep13_layer(\d+)_lam([0-9\.]+)_tr(\d+)$")

LAMBDAS = [0.01, 0.1]  # one bar plot and one scatter each
SCATTER_LAYERS = [2, 4, 6, 8, 10]  # hand-picked subset; see module docstring

EXCLUDE_LAYERS: set[int] = set()
MIN_RUNS = 1  # drop (layer, lambda) points with fewer usable runs than this
CKPT = "best"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
TASK_LOSS_N_EVAL = 50_000  # fresh examples per run for the recomputed task loss
TASK_LOSS_NOISE_MULT = 1.0  # multiplier on the checkpoint's own resid_noise_std
ONE_HOT_LOSS_N_EVAL = 50_000  # fresh examples per run for the one-hot OOD loss
PROBE_N_TRAIN = 5000  # per class; refit per run, across many runs in a sweep
PROBE_N_TEST = 10_000  # per class
PROBE_BACKEND = "newton"
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
AUROC_THRESHOLDS = [
    0.6,
    0.75,
    0.9,
]  # ascending; splits "succeeded" runs into len+1 hiding bins


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help=f"delete {CACHE_PATH} before running, forcing a full recompute",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import glob
import json
import os
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from checkpoint_lib import (
    load_model,
    recompute_n_hot_loss,
    recompute_task_loss,
    resolve_adv_config,
)
from probe_backend import resolve_probe_backend
from probe_lib import (
    LinearBoundary,
    binary_probe_metrics_all_layers,
    boundary_auroc,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

RunKey = tuple[int, float]  # (penalized layer, lam)


# Sequential blue ramp (references/palette.md), lightest -> darkest, 100..700.
SEQ_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
FAILED_COLOR = "#eb6834"  # categorical slot 2 (orange) -- distinct "problem" hue


class RunStats(NamedTuple):
    """A run config's outcome-band fractions, aggregated over its trials."""

    frac: np.ndarray  # length n_bands, sums to 1
    n_ok: int  # usable trials this is averaged over


def _discover_tags() -> dict[RunKey, list[str]]:
    """Run tags grouped by the (penalized layer, lambda) they belong to."""
    by_key: dict[RunKey, list[str]] = {}
    for path in sorted(glob.glob(os.path.join("runs", RUN_GLOB))):
        tag = os.path.basename(path)
        m = TAG_RE.match(tag)
        if not m:
            continue
        layer, lam = int(m.group(1)), float(m.group(2))
        if layer in EXCLUDE_LAYERS:
            continue
        by_key.setdefault((layer, lam), []).append(tag)
    return by_key


def _probe_layer(tag: str, ck) -> int:
    """The layer a run was penalized at, from its tag, cross-checked against
    the checkpoint's own adversarial config. The whole sweep is over this
    value, so a tag that disagrees with what was actually trained would
    silently misattribute a run to the wrong x-axis position."""
    layer = int(TAG_RE.match(tag).group(1))
    adv_cfg = resolve_adv_config(ck)
    if adv_cfg is not None and list(adv_cfg.penalty_layers) != [layer]:
        raise AssertionError(
            f"{tag}: tag says layer {layer} but the checkpoint was trained with "
            f"penalty_layers={list(adv_cfg.penalty_layers)}"
        )
    return layer


def _probe_auroc(tag: str, g: torch.Generator, probe_backend_name: str) -> float | None:
    """AUROC of a freshly refit probe at the run's *own* penalized layer for
    `tag`'s CKPT checkpoint -- not the checkpoint's stored training-time
    probe, so this stays comparable across a sweep whose runs may have
    trained with different probe settings. None for a checkpoint with no
    adversarial config (nothing to probe for)."""
    model, ck = load_model(tag, CKPT, DEVICE)
    adv_cfg = resolve_adv_config(ck)
    if adv_cfg is None:
        return None
    layer = _probe_layer(tag, ck)
    eval_noise_std = adv_cfg.resid_noise_std * PROBE_EVAL_NOISE_MULT
    train_noise_std = adv_cfg.resid_noise_std * PROBE_TRAIN_NOISE_MULT
    _metrics, plot_inputs = binary_probe_metrics_all_layers(
        model,
        1.0,
        2.0,
        [layer],
        PROBE_N_TRAIN,
        PROBE_N_TEST,
        g,
        probe_backend_name,
        desc=tag,
        eval_noise=eval_noise_std,
        train_noise=train_noise_std,
    )
    pi = plot_inputs[layer]
    probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
    return boundary_auroc(probe, pi["X_te"], pi["y_te"])


def _cache_key(tag: str) -> str:
    """Cache identity for a run's (loss, auroc): the tag plus every setting
    the recomputation depends on, so edits to those miss rather than reuse.
    The probe layer isn't listed -- it's a property of the run, already
    pinned by the tag."""
    settings = (
        CKPT,
        TASK_LOSS_N_EVAL,
        TASK_LOSS_NOISE_MULT,
        ONE_HOT_LOSS_N_EVAL,
        PROBE_N_TRAIN,
        PROBE_N_TEST,
        PROBE_BACKEND,
        PROBE_TRAIN_NOISE_MULT,
        PROBE_EVAL_NOISE_MULT,
    )
    return "|".join([tag] + [f"{s}" for s in settings])


def _load_cache() -> dict[str, dict]:
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH) as f:
        return json.load(f)


def _save_cache(cache: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    os.replace(tmp, CACHE_PATH)


def _classify(
    loss: float, one_hot_loss: float, auroc: float, thresholds: list[float]
) -> int:
    """Bucket index, 0 = failed task, 1 = succeeded but not hidden (auroc
    above the highest threshold), ..., len(thresholds) + 1 = succeeded and
    most hidden (auroc below the lowest threshold). "Succeeded" requires
    both loss and one_hot_loss below their (separate) thresholds."""
    if loss >= LOSS_THRESHOLD or one_hot_loss >= ONE_HOT_LOSS_THRESHOLD:
        return 0
    for i, t in enumerate(reversed(thresholds)):
        if auroc >= t:
            return i + 1
    return len(thresholds) + 1


def _band_labels(thresholds: list[float]) -> list[str]:
    labels = [f"failed task"]
    desc = reversed(thresholds)
    prev = None
    for t in desc:
        if prev is None:
            labels.append(f"not hidden\n(AUROC >= {t:g})")
        else:
            labels.append(f"partially hidden\n({t:g} <= AUROC < {prev:g})")
        prev = t
    labels.append(f"hidden\n(AUROC < {prev:g})")
    return labels


def _band_colors(thresholds: list[float]) -> list[str]:
    n_hiding_bins = len(thresholds) + 1
    idxs = np.linspace(0, len(SEQ_RAMP) - 1, n_hiding_bins).astype(int)
    return [FAILED_COLOR] + [SEQ_RAMP[i] for i in idxs]


SMOKE_LAYERS = [2, 4, 6, 8, 10]


def _smoke_run_stats(n_bands: int) -> dict[RunKey, RunStats]:
    """Synthetic per-(layer, lambda) stats standing in for a real sweep's
    results, just to preview the plots' layout/styling without an analysis
    run."""
    rng = np.random.default_rng(SEED)

    def band_fracs(p: float) -> np.ndarray:
        # p: 0..1 knob, later layer / bigger lambda -> more hiding.
        failed = 0.05 + 0.4 * p
        weights = np.linspace(1 - p, p, n_bands - 1) + 0.1
        weights = weights / weights.sum() * (1 - failed)
        counts = np.clip(np.concatenate([[failed], weights]), 0.001, None)
        counts = np.clip(counts + rng.normal(0, 0.02, size=n_bands), 0.001, None)
        return counts / counts.sum()

    stats: dict[RunKey, RunStats] = {}
    for i, layer in enumerate(SMOKE_LAYERS):
        for j, lam in enumerate(LAMBDAS):
            p = (i / (len(SMOKE_LAYERS) - 1)) * (0.5 + 0.5 * j)
            stats[(layer, lam)] = RunStats(band_fracs(p), 10)
    return stats


def _smoke_scatter_points() -> list[tuple[float, float, int, float]]:
    """Synthetic (loss, auroc, layer, lam) quadruples, several per (layer,
    lambda), standing in for a real sweep's per-run results."""
    rng = np.random.default_rng(SEED)
    n = 10  # synthetic runs per (layer, lambda)
    points = []
    for i, layer in enumerate(SMOKE_LAYERS):
        p = i / (len(SMOKE_LAYERS) - 1)
        for lam in LAMBDAS:
            losses = np.clip(rng.normal(0.002 + 0.03 * p, 0.005, n), 1e-4, None)
            aurocs = np.clip(rng.normal(0.95 - 0.4 * p, 0.05, n), 0.5, 1.0)
            points += [
                (float(loss), float(auroc), layer, lam)
                for loss, auroc in zip(losses, aurocs)
            ]
    return points


def _run_metrics(
    tag: str, cache: dict[str, dict], g: torch.Generator, probe_backend_name: str
) -> dict | None:
    """A run's recomputed {loss, one_hot_loss, auroc}, from the cache when
    it's there and recomputed (then cached) when it isn't. None for a run
    whose checkpoint doesn't exist yet."""
    cache_key = _cache_key(tag)
    if cache_key in cache:
        return cache[cache_key]
    try:
        entry = {
            "loss": recompute_task_loss(
                tag, CKPT, g, DEVICE, TASK_LOSS_N_EVAL, TASK_LOSS_NOISE_MULT
            ),
            "one_hot_loss": recompute_n_hot_loss(
                tag, CKPT, g, DEVICE, ONE_HOT_LOSS_N_EVAL, 1, TASK_LOSS_NOISE_MULT
            ),
            "auroc": _probe_auroc(tag, g, probe_backend_name),
        }
    except FileNotFoundError:
        print(f"  {tag}: no history/checkpoint yet, skipped")
        return None
    cache[cache_key] = entry
    _save_cache(cache)  # per run, so an interrupted pass keeps progress
    return entry


def _collect_run_stats(
    n_bands: int,
) -> tuple[dict[RunKey, RunStats], list[tuple[float, float, int, float]]]:
    """Per-(layer, lambda) band fractions and usable-run count, recomputing
    each run's loss/AUROC where the cache doesn't already have them, plus
    every usable run's (loss, auroc, layer, lam) for the scatter."""
    thresholds = sorted(AUROC_THRESHOLDS)
    by_key = _discover_tags()
    cache = _load_cache()
    # One RNG across every run config, so its draws (eval noise, probe
    # train/test sampling) are reproducible across a full run of the script.
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    probe_backend_name = resolve_probe_backend(PROBE_BACKEND, DEVICE)

    stats: dict[RunKey, RunStats] = {}
    scatter_points: list[tuple[float, float, int, float]] = []
    print(f"{'layer':>6s} {'lambda':>8s} {'n_ok':>5s} {'n_total':>7s}")
    for key in sorted(by_key):
        layer, lam = key
        tags = by_key[key]
        counts = np.zeros(n_bands)
        n_ok = 0
        for tag in tags:
            entry = _run_metrics(tag, cache, g, probe_backend_name)
            if entry is None:
                continue
            if entry["auroc"] is None:
                print(f"  {tag}: no probe in checkpoint, skipped")
                continue
            band = _classify(
                entry["loss"], entry["one_hot_loss"], entry["auroc"], thresholds
            )
            counts[band] += 1
            n_ok += 1
            scatter_points.append((entry["loss"], entry["auroc"], layer, lam))
        if n_ok < MIN_RUNS:
            print(f"  {key}: {n_ok} usable runs, skipped")
            continue
        stats[key] = RunStats(counts / n_ok, n_ok)
        print(f"{layer:6d} {lam:8g} {n_ok:5d} {len(tags):7d}")
    return stats, scatter_points


def _draw_bars(
    ax,
    positions: list[float],
    stats_at: list[RunStats],
    n_bands: int,
    labels: list[str],
    colors: list[str],
    width: float = 0.6,
    legend: bool = False,
) -> None:
    """Draw one set of stacked outcome bars at the given x positions."""
    for i, (x, s) in enumerate(zip(positions, stats_at)):
        bottom = 0.0
        # Draw "most hidden" at the bottom up to "failed" on top, matching
        # sweep7_analysis's stack order.
        for band in range(n_bands - 1, -1, -1):
            ax.bar(
                x,
                s.frac[band],
                bottom=bottom,
                width=width,
                color=colors[band],
                edgecolor="black",
                linewidth=0.4,
                label=labels[band] if legend and i == 0 else None,
            )
            bottom += s.frac[band]
        ax.text(x, 1.01, f"n={s.n_ok}", ha="center", va="bottom", fontsize=7)


def _plot_by_layer(
    run_stats: dict[RunKey, RunStats], lam: float, n_bands: int
) -> plt.Figure:
    """One panel: bars per penalized layer at `lam`."""
    layers = sorted({k[0] for k in run_stats})
    labels = _band_labels(sorted(AUROC_THRESHOLDS))
    colors = _band_colors(sorted(AUROC_THRESHOLDS))

    fig, ax = plt.subplots(figsize=(1.2 * len(layers) + 1.5, 4.2))
    lam_stats = [run_stats[(ly, lam)] for ly in layers if (ly, lam) in run_stats]
    lam_pos = [i for i, ly in enumerate(layers) if (ly, lam) in run_stats]
    _draw_bars(ax, lam_pos, lam_stats, n_bands, labels, colors, legend=True)

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
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for lam in LAMBDAS:
        if not (lams == lam).any():
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, layer in enumerate(plot_layers):
            mask = (lams == lam) & (layers == layer)
            ax.scatter(
                losses[mask],
                aurocs[mask],
                color=cycle[i % len(cycle)],
                edgecolor="black",
                linewidth=0.5,
                label=f"layer {layer}",
            )
        ax.set_xlabel("task loss on training distribution")
        ax.set_ylabel("probe AUROC (at each run's penalized layer)")
        ax.set_title(f"$\\lambda$ = {lam:g}: task loss vs. probe AUROC")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        # Log-scale minor tick labels (2x, 3x, ...) crowd together when the
        # data spans less than a decade; rotating keeps them legible at any
        # range.
        plt.setp(ax.get_xticklabels(which="both"), rotation=45, ha="right")

        ax.axvline(
            LOSS_THRESHOLD,
            linestyle="--",
            color="black",
            label="loss threshold",
            alpha=0.5,
        )
        ax.legend()
        fig.savefig(
            f"{PLOT_DIR}/loss_vs_auroc_scatter_lam{lam:g}.png", bbox_inches="tight"
        )


def main(clear_cache: bool = False) -> None:
    if clear_cache and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    thresholds = sorted(AUROC_THRESHOLDS)
    n_bands = len(thresholds) + 2

    if SMOKE:
        run_stats, scatter_points = _smoke_run_stats(n_bands), _smoke_scatter_points()
    else:
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
