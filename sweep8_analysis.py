"""sweep_analysis.py's outcome breakdown, for runs/sweep8_* -- a sweep over
model size at three lambdas, plus a side ray over model size at one lambda.

Same per-run classification as sweep_analysis: a run either failed the task
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

Each ray also has lam=0 controls, which aren't plotted (a run that always
succeeds and never hides is the same bar at every size) but are checked by
_assert_lam0_clean, which raises if one didn't behave like a control.

The (32, 64, 16) point sits on both rays; runs/sweep7_* already covers it
(see SWEEP7_*), so sweep8 didn't retrain it and this script pulls that data
in instead. It shows up on both the relevant main-sweep plot and the side
ray.

Bars, not sweep_analysis's stackplot: sweep8 samples only a handful of
lambdas, too few for a filled area between them to mean anything.

Lambda coverage is deliberately uneven across sizes (only some sizes get the
full lambda grid), so a missing bar means no runs, not zero runs surviving.

LOSS_THRESHOLD / AUROC_THRESHOLDS are provisional -- a separate exercise is
picking the final values -- so they're constants here, not a CLI, alongside
everything else that may still change.

"Loss" and "AUROC" are both freshly recomputed at CKPT -- loss via
data.eval_task_loss (deliberately excluding the adversarial/probe penalty,
since this only cares about task performance), and AUROC via a freshly refit
probe at PROBE_LAYER -- rather than read from history.jsonl or the
checkpoint's own stored training-time probe. See sweep_analysis.py and
sweep_threshold_report.py, which recompute the same way for the same reasons
(some runs' final checkpoint predates their last history record).

Recomputing both over a whole sweep is slow, so results are cached to
CACHE_PATH keyed by the settings they depend on; changing any of those
settings misses the cache rather than reusing a stale value. Pass
--clear-cache to force a full recompute.
"""

import argparse
import re

SMOKE = False  # if True, skip the real analysis and plot synthetic data instead
# -- lets you preview the plot's layout without waiting on a full sweep's
# worth of checkpoint loading / task-loss / probe recomputation.

PLOT_DIR = "plot/sweep8"
CACHE_PATH = "plot/sweep8/metrics_cache.json"
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
TASK_LOSS_N_EVAL = 50_000  # fresh examples per run for the recomputed task loss
TASK_LOSS_NOISE_MULT = 1.0  # multiplier on the checkpoint's own resid_noise_std
ONE_HOT_LOSS_N_EVAL = 50_000  # fresh examples per run for the one-hot OOD loss
PROBE_N_TRAIN = 5000  # per class; refit per run, across many runs in a sweep
PROBE_N_TEST = 10_000  # per class
PROBE_BACKEND = "newton"
EVAL_NOISE_MULT = 1.0  # multiplier on resid_noise_std when retraining probe

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

from data import eval_n_hot_loss, eval_task_loss
from probe_backend import resolve_probe_backend
from probe_lib import (
    LinearBoundary,
    binary_probe_metrics_all_layers,
    boundary_auroc,
    load_model,
    resolve_adv_config,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

RunKey = tuple[int, int, int, float]  # (num_x, d_model, d_mlp, lam)


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


def _is_main_sweep(key: RunKey) -> bool:
    num_x, d_model, d_mlp, _ = key
    return d_model == 2 * num_x and d_mlp == num_x // 2


def _is_side_ray(key: RunKey) -> bool:
    num_x, d_model, d_mlp, _ = key
    return d_mlp == 16 and d_model == 2 * num_x


def _discover_tags() -> dict[RunKey, list[str]]:
    """Run tags grouped by the (num_x, d_model, d_mlp, lambda) they belong
    to, from both runs/sweep8_* and the matching runs/sweep7_* points."""
    by_key: dict[RunKey, list[str]] = {}
    for path in sorted(glob.glob(os.path.join("runs", RUN_GLOB))):
        tag = os.path.basename(path)
        m = TAG_RE.match(tag)
        if not m:
            continue
        num_x, d_model, d_mlp = int(m.group(1)), int(m.group(2)), int(m.group(3))
        lam = float(m.group(4))
        if lam in EXCLUDE_LAMBDAS or (num_x, d_model, d_mlp) in EXCLUDE_SIZES:
            continue
        by_key.setdefault((num_x, d_model, d_mlp, lam), []).append(tag)

    for path in sorted(glob.glob(os.path.join("runs", SWEEP7_GLOB))):
        tag = os.path.basename(path)
        m = SWEEP7_TAG_RE.match(tag)
        if not m:
            continue
        lam = float(m.group(1))
        if lam not in SWEEP7_LAMBDAS or lam in EXCLUDE_LAMBDAS:
            continue
        if SWEEP7_SIZE in EXCLUDE_SIZES:
            continue
        by_key.setdefault((*SWEEP7_SIZE, lam), []).append(tag)

    return by_key


def _final_loss(tag: str, g: torch.Generator) -> float:
    """Task loss (data.eval_task_loss), freshly recomputed at CKPT -- not
    read from history.jsonl (some runs' final checkpoint predates their last
    history record) and deliberately excludes the adversarial/probe penalty,
    since this script only cares about task performance here."""
    model, ck = load_model(tag, CKPT, DEVICE)
    adv_cfg = resolve_adv_config(ck)
    x_p_outer = adv_cfg.x_p_outer if adv_cfg is not None else None
    x_threshold = adv_cfg.x_threshold if adv_cfg is not None else 1.0
    noise_std = (
        adv_cfg.resid_noise_std * TASK_LOSS_NOISE_MULT if adv_cfg is not None else 0.0
    )
    return eval_task_loss(
        model,
        g,
        DEVICE,
        n=TASK_LOSS_N_EVAL,
        x_p_outer=x_p_outer,
        x_threshold=x_threshold,
        noise=noise_std,
    )


def _one_hot_loss(tag: str, g: torch.Generator) -> float:
    """One-hot OOD loss (data.eval_n_hot_loss, n_hot=1), freshly recomputed
    at CKPT -- only one input coordinate nonzero per example, unlike
    _final_loss's plain training-distribution eval. Uses the checkpoint's own
    resid_noise_std (TASK_LOSS_NOISE_MULT) so the two loss columns differ
    only in the input distribution, not the noise."""
    model, ck = load_model(tag, CKPT, DEVICE)
    adv_cfg = resolve_adv_config(ck)
    noise_std = (
        adv_cfg.resid_noise_std * TASK_LOSS_NOISE_MULT if adv_cfg is not None else 0.0
    )
    return eval_n_hot_loss(
        model,
        g,
        DEVICE,
        n=ONE_HOT_LOSS_N_EVAL,
        n_hot=1,
        noise=noise_std,
    )


def _probe_auroc(tag: str, g: torch.Generator, probe_backend_name: str) -> float | None:
    """AUROC of a freshly refit probe at PROBE_LAYER for `tag`'s CKPT
    checkpoint -- not the checkpoint's own stored training-time probe, so
    this stays comparable across a sweep whose runs may have trained with
    different probe settings. None for a checkpoint with no adversarial
    config (nothing to probe for)."""
    model, ck = load_model(tag, CKPT, DEVICE)
    adv_cfg = resolve_adv_config(ck)
    if adv_cfg is None:
        return None
    eval_noise_std = adv_cfg.resid_noise_std * EVAL_NOISE_MULT
    _metrics, plot_inputs = binary_probe_metrics_all_layers(
        model,
        1.0,
        2.0,
        [PROBE_LAYER],
        PROBE_N_TRAIN,
        PROBE_N_TEST,
        g,
        probe_backend_name,
        desc=tag,
        eval_noise=eval_noise_std,
        train_noise=adv_cfg.resid_noise_std,
    )
    pi = plot_inputs[PROBE_LAYER]
    probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
    return boundary_auroc(probe, pi["X_te"], pi["y_te"])


def _cache_key(tag: str) -> str:
    """Cache identity for a run's (loss, auroc): the tag plus every setting
    the recomputation depends on, so edits to those miss rather than reuse."""
    settings = (
        CKPT,
        PROBE_LAYER,
        TASK_LOSS_N_EVAL,
        TASK_LOSS_NOISE_MULT,
        ONE_HOT_LOSS_N_EVAL,
        PROBE_N_TRAIN,
        PROBE_N_TEST,
        PROBE_BACKEND,
        EVAL_NOISE_MULT,
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
    labels = [
        f"failed task (loss >= {LOSS_THRESHOLD:g} or "
        f"one-hot loss >= {ONE_HOT_LOSS_THRESHOLD:g})"
    ]
    desc = reversed(thresholds)
    prev = None
    for t in desc:
        if prev is None:
            labels.append(f"not hidden (auroc >= {t:g})")
        else:
            labels.append(f"partially hidden ({t:g} <= auroc < {prev:g})")
        prev = t
    labels.append(f"hidden (auroc < {prev:g})")
    return labels


def _band_colors(thresholds: list[float]) -> list[str]:
    n_hiding_bins = len(thresholds) + 1
    idxs = np.linspace(0, len(SEQ_RAMP) - 1, n_hiding_bins).astype(int)
    return [FAILED_COLOR] + [SEQ_RAMP[i] for i in idxs]


def _smoke_run_stats(n_bands: int) -> dict[RunKey, RunStats]:
    """Synthetic per-run-config stats standing in for a real sweep's
    results, just to preview the plots' layout/styling without an analysis
    run. Covers both rays, including clean (all not-hidden) lam=0 controls
    so _assert_lam0_clean has something to check without tripping."""
    rng = np.random.default_rng(SEED)

    def band_fracs(p: float) -> np.ndarray:
        # p: 0..1 knob, bigger model / bigger lambda -> more hiding.
        failed = 0.05 + 0.4 * p
        weights = np.linspace(1 - p, p, n_bands - 1) + 0.1
        weights = weights / weights.sum() * (1 - failed)
        counts = np.clip(np.concatenate([[failed], weights]), 0.001, None)
        counts = np.clip(counts + rng.normal(0, 0.02, size=n_bands), 0.001, None)
        return counts / counts.sum()

    clean = np.zeros(n_bands)
    clean[1] = 1.0  # all succeeded, none hidden

    stats: dict[RunKey, RunStats] = {}

    main_sizes = [(8, 16, 4), (16, 32, 8), (32, 64, 16), (64, 128, 32)]
    for i, size in enumerate(main_sizes):
        stats[(*size, 0.0)] = RunStats(clean, 10)
        for lam in MAIN_LAMBDAS:
            stats[(*size, lam)] = RunStats(band_fracs(i / len(main_sizes)), 10)

    side_sizes = [(16, 32, 16), (32, 64, 16), (48, 96, 16), (64, 128, 16)]
    for i, size in enumerate(side_sizes):
        stats[(*size, 0.0)] = RunStats(clean, 10)
        stats[(*size, SIDE_LAMBDA)] = RunStats(band_fracs(i / len(side_sizes)), 10)

    return stats


def _collect_run_stats(n_bands: int) -> dict[RunKey, RunStats]:
    """Per-run-config band fractions and usable-run count, recomputing each
    run's loss/AUROC where the cache doesn't already have them."""
    thresholds = sorted(AUROC_THRESHOLDS)
    by_key = _discover_tags()
    cache = _load_cache()
    # One RNG across every run config, so its draws (eval noise, probe
    # train/test sampling) are reproducible across a full run of the script.
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    probe_backend_name = resolve_probe_backend(PROBE_BACKEND, DEVICE)

    stats: dict[RunKey, RunStats] = {}
    print(f"{'size':>18s} {'lambda':>8s} {'n_ok':>5s} {'n_total':>7s}")
    for key in sorted(by_key):
        num_x, d_model, d_mlp, lam = key
        tags = by_key[key]
        counts = np.zeros(n_bands)
        n_ok = 0
        for tag in tags:
            cache_key = _cache_key(tag)
            if cache_key not in cache:
                try:
                    cache[cache_key] = {
                        "loss": _final_loss(tag, g),
                        "one_hot_loss": _one_hot_loss(tag, g),
                        "auroc": _probe_auroc(tag, g, probe_backend_name),
                    }
                except FileNotFoundError:
                    print(f"  {tag}: no history/checkpoint yet, skipped")
                    continue
                _save_cache(cache)  # per run, so an interrupted pass keeps progress
            entry = cache[cache_key]
            if entry["auroc"] is None:
                print(f"  {tag}: no probe in checkpoint, skipped")
                continue
            counts[
                _classify(
                    entry["loss"], entry["one_hot_loss"], entry["auroc"], thresholds
                )
            ] += 1
            n_ok += 1
        if n_ok < MIN_RUNS:
            print(f"  {key}: {n_ok} usable runs, skipped")
            continue
        stats[key] = RunStats(counts / n_ok, n_ok)
        size_str = f"nx{num_x} dm{d_model} mlp{d_mlp}"
        print(f"{size_str:>18s} {lam:8g} {n_ok:5d} {len(tags):7d}")
    return stats


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
        # sweep_analysis's stack order.
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

    labels = _band_labels(sorted(AUROC_THRESHOLDS))
    colors = _band_colors(sorted(AUROC_THRESHOLDS))

    fig, ax = plt.subplots(figsize=(1.2 * len(sizes) + 1.5, 4.2))
    lam_stats = [run_stats[(*s, lam)] for s in sizes if (*s, lam) in run_stats]
    lam_pos = [i for i, s in enumerate(sizes) if (*s, lam) in run_stats]
    _draw_bars(ax, lam_pos, lam_stats, n_bands, labels, colors, legend=True)

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

    labels = _band_labels(sorted(AUROC_THRESHOLDS))
    colors = _band_colors(sorted(AUROC_THRESHOLDS))

    fig, ax = plt.subplots(figsize=(1.2 * len(sizes) + 1.5, 4.2))
    stats_at = [
        run_stats[(*s, SIDE_LAMBDA)] for s in sizes if (*s, SIDE_LAMBDA) in run_stats
    ]
    pos = [i for i, s in enumerate(sizes) if (*s, SIDE_LAMBDA) in run_stats]
    _draw_bars(ax, pos, stats_at, n_bands, labels, colors, legend=True)

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


def main(clear_cache: bool = False) -> None:
    if clear_cache and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    thresholds = sorted(AUROC_THRESHOLDS)
    n_bands = len(thresholds) + 2

    run_stats = _smoke_run_stats(n_bands) if SMOKE else _collect_run_stats(n_bands)
    if not run_stats:
        raise SystemExit(f"no usable runs matched runs/{RUN_GLOB}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    for lam in MAIN_LAMBDAS:
        fig = _plot_main_sweep(run_stats, lam, n_bands)
        fig.savefig(f"{PLOT_DIR}/main_lam{lam:g}.png", bbox_inches="tight")

    fig = _plot_side_ray(run_stats, n_bands)
    fig.savefig(f"{PLOT_DIR}/side_ray.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main(clear_cache=args.clear_cache)
