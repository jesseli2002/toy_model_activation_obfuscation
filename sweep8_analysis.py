"""sweep_analysis.py's outcome breakdown, for runs/sweep8_* -- a sweep over
model size as well as lambda.

Same per-run classification as sweep_analysis: a run either failed the task
(loss) or, having succeeded, is binned by how well it hid from the
layer-PROBE_LAYER linear probe (AUROC). The difference is the layout: sweep8
varies (num_x, d_model, d_mlp) alongside lambda, so the breakdown is faceted
-- one panel per value of the FACET axis, stacked bars along the other. Set
FACET to see the same data as "does hiding change with lambda, per size?" or
"does hiding change with size, per lambda?".

Bars, not sweep_analysis's stackplot: sweep8 samples only a handful of
lambdas, too few for a filled area between them to mean anything.

Lambda coverage is deliberately uneven across sizes (only some sizes get the
full lambda grid), so panels have differing numbers of bars; a missing bar
means no runs, not zero runs surviving.

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
settings misses the cache rather than reusing a stale value. Delete the file
to force a full recompute.
"""

import re

SMOKE = False  # if True, skip the real analysis and plot synthetic data instead
# -- lets you preview the plot's layout without waiting on a full sweep's
# worth of checkpoint loading / task-loss / probe recomputation.

PLOT_DIR = "plot/sweep8"
CACHE_PATH = "plot/sweep8/metrics_cache.json"
RUN_GLOB = "sweep8_nx*_tr*"
TAG_RE = re.compile(r"sweep8_nx(\d+)_dm(\d+)_mlp(\d+)_lam([0-9\.]+)_tr(\d+)$")
FACET = "size"  # "size" (panel per model size, x = lambda) or "lam" (the transpose)
EXCLUDE_LAMBDAS: set[float] = set()
EXCLUDE_SIZES: set[tuple[int, int, int]] = set()  # (num_x, d_model, d_mlp)
MIN_RUNS = 1  # drop (size, lambda) cells with fewer usable runs than this
CKPT = "best"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
PROBE_LAYER = 2  # matches adversarial.penalty_layers in these runs' config.json
TASK_LOSS_N_EVAL = 50_000  # fresh examples per run for the recomputed task loss
TASK_LOSS_NOISE_MULT = 1.0  # multiplier on the checkpoint's own resid_noise_std
PROBE_N_TRAIN = 5000  # per class; refit per run, across many runs in a sweep
PROBE_N_TEST = 10_000  # per class
PROBE_BACKEND = "newton"
EVAL_NOISE_MULT = 1.0  # multiplier on resid_noise_std when retraining probe

LOSS_THRESHOLD = 0.01  # task "succeeded" iff final loss below this
AUROC_THRESHOLDS = [
    0.6,
    0.75,
    0.9,
]  # ascending; splits "succeeded" runs into len+1 hiding bins


import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import eval_task_loss
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

Size = tuple[int, int, int]  # (num_x, d_model, d_mlp)


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


def _size_label(size: Size) -> str:
    num_x, d_model, d_mlp = size
    return f"nx{num_x} dm{d_model} mlp{d_mlp}"


def _lam_label(lam: float) -> str:
    return f"{lam:g}"


def _discover_tags() -> dict[tuple[Size, float], list[str]]:
    """Run tags grouped by the (model size, lambda) cell they belong to."""
    cells: dict[tuple[Size, float], list[str]] = {}
    for path in sorted(glob.glob(os.path.join("runs", RUN_GLOB))):
        tag = os.path.basename(path)
        m = TAG_RE.match(tag)
        if not m:
            continue
        size: Size = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        lam = float(m.group(4))
        if lam in EXCLUDE_LAMBDAS or size in EXCLUDE_SIZES:
            continue
        cells.setdefault((size, lam), []).append(tag)
    return cells


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
        train_noise=adv_cfg.resid_noise_std
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


def _classify(loss: float, auroc: float, thresholds: list[float]) -> int:
    """Bucket index, 0 = failed task, 1 = succeeded but not hidden (auroc
    above the highest threshold), ..., len(thresholds) + 1 = succeeded and
    most hidden (auroc below the lowest threshold)."""
    if loss >= LOSS_THRESHOLD:
        return 0
    for i, t in enumerate(reversed(thresholds)):
        if auroc >= t:
            return i + 1
    return len(thresholds) + 1


def _band_labels(thresholds: list[float]) -> list[str]:
    labels = [f"failed task (loss >= {LOSS_THRESHOLD:g})"]
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


def _smoke_cells(n_bands: int) -> dict[tuple[Size, float], tuple[np.ndarray, int]]:
    """Synthetic per-cell (fractions, n_ok) standing in for a real sweep's
    results, just to preview the plot's layout/styling without an analysis run."""
    rng = np.random.default_rng(SEED)
    sizes: list[Size] = [(8, 16, 4), (16, 32, 8), (16, 32, 16), (64, 128, 32)]
    lambdas = [0.0, 0.001, 0.01, 0.1]
    cells = {}
    for i, size in enumerate(sizes):
        for j, lam in enumerate(lambdas):
            # p: 0..1 knob, bigger model / bigger lambda -> more hiding
            p = np.clip((i / len(sizes) + j / len(lambdas)) / 2, 0, 1)
            failed = 0.05 + 0.4 * p
            weights = np.linspace(1 - p, p, n_bands - 1) + 0.1
            weights = weights / weights.sum() * (1 - failed)
            counts = np.clip(np.concatenate([[failed], weights]), 0.001, None)
            counts = np.clip(counts + rng.normal(0, 0.02, size=n_bands), 0.001, None)
            cells[(size, lam)] = (counts / counts.sum(), 10)
    return cells


def _collect_cells(n_bands: int) -> dict[tuple[Size, float], tuple[np.ndarray, int]]:
    """Per-(size, lambda) band fractions and usable-run count, recomputing
    each run's loss/AUROC where the cache doesn't already have them."""
    thresholds = sorted(AUROC_THRESHOLDS)
    by_cell = _discover_tags()
    cache = _load_cache()
    # One RNG across every cell, so its draws (eval noise, probe train/test
    # sampling) are reproducible across a full run of the script.
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    probe_backend_name = resolve_probe_backend(PROBE_BACKEND, DEVICE)

    cells = {}
    print(f"{'size':>18s} {'lambda':>8s} {'n_ok':>5s} {'n_total':>7s}")
    for size, lam in sorted(by_cell):
        tags = by_cell[(size, lam)]
        counts = np.zeros(n_bands)
        n_ok = 0
        for tag in tags:
            key = _cache_key(tag)
            if key not in cache:
                try:
                    cache[key] = {
                        "loss": _final_loss(tag, g),
                        "auroc": _probe_auroc(tag, g, probe_backend_name),
                    }
                except FileNotFoundError:
                    print(f"  {tag}: no history/checkpoint yet, skipped")
                    continue
                _save_cache(cache)  # per run, so an interrupted pass keeps progress
            entry = cache[key]
            if entry["auroc"] is None:
                print(f"  {tag}: no probe in checkpoint, skipped")
                continue
            counts[_classify(entry["loss"], entry["auroc"], thresholds)] += 1
            n_ok += 1
        if n_ok < MIN_RUNS:
            print(f"  {size} lam={lam:g}: {n_ok} usable runs, skipped")
            continue
        cells[(size, lam)] = (counts / n_ok, n_ok)
        print(f"{str(size):>18s} {lam:8g} {n_ok:5d} {len(tags):7d}")
    return cells


def _facet_axes(
    cells: dict[tuple[Size, float], tuple[np.ndarray, int]],
) -> tuple[list, list, str, str]:
    """Split the cell keys into (panel values, x values) per FACET, plus the
    axis titles. Both are the full sorted set across all cells, so panels
    share an x axis even where a size/lambda combination wasn't run."""
    sizes = sorted({size for size, _ in cells}, key=lambda s: (s[0], s[2]))
    lambdas = sorted({lam for _, lam in cells})
    if FACET == "size":
        return sizes, lambdas, "size", r"$\lambda$"
    if FACET == "lam":
        return lambdas, sizes, "lambda", "model size"
    raise ValueError(f"FACET must be 'size' or 'lam', got {FACET!r}")


def main():
    thresholds = sorted(AUROC_THRESHOLDS)
    n_bands = len(thresholds) + 2

    cells = _smoke_cells(n_bands) if SMOKE else _collect_cells(n_bands)
    if not cells:
        raise SystemExit(f"no usable runs matched runs/{RUN_GLOB}")

    panels, xs, panel_kind, x_label = _facet_axes(cells)
    labels = _band_labels(thresholds)
    colors = _band_colors(thresholds)

    n_col = min(3, len(panels))
    n_row = -(-len(panels) // n_col)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(4.2 * n_col, 3.4 * n_row),
        squeeze=False,
        sharey=True,
    )
    flat = axes.ravel()

    for ax, panel in zip(flat, panels):
        present = [
            (i, x)
            for i, x in enumerate(xs)
            if ((panel, x) if FACET == "size" else (x, panel)) in cells
        ]
        for i, x in present:
            frac, n_ok = cells[(panel, x) if FACET == "size" else (x, panel)]
            # Label one bar only -- every bar draws the same bands, and a
            # label per bar would repeat the whole legend once per bar.
            legend_bar = ax is flat[0] and (i, x) == present[0]
            bottom = 0.0
            # Draw "most hidden" at the bottom up to "failed" on top, matching
            # sweep_analysis's stack order.
            for band in range(n_bands - 1, -1, -1):
                ax.bar(
                    i,
                    frac[band],
                    bottom=bottom,
                    width=0.7,
                    color=colors[band],
                    edgecolor="black",
                    linewidth=0.4,
                    label=labels[band] if legend_bar else None,
                )
                bottom += frac[band]
            ax.text(i, 1.01, f"n={n_ok}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(
            [_lam_label(x) if FACET == "size" else _size_label(x) for x in xs],
            fontsize=8,
            # Size labels are long enough to collide when laid flat.
            rotation=0 if FACET == "size" else 30,
            ha="center" if FACET == "size" else "right",
        )
        ax.set_xlim(-0.6, len(xs) - 0.4)
        ax.set_ylim(0, 1)
        title = _size_label(panel) if FACET == "size" else f"$\\lambda$ = {panel:g}"
        ax.set_title(title, fontsize=10, pad=14)  # pad clears the n= annotations
        ax.set_xlabel(x_label)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in flat[len(panels) :]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of runs")

    # Bands were drawn top-down; reverse so the legend reads top-down too.
    handles, leg_labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        leg_labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
    )
    fig.suptitle(f"Outcome by {panel_kind} and {x_label}\n({RUN_GLOB})")
    fig.tight_layout()

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(
        f"{PLOT_DIR}/size_sweep_task{LOSS_THRESHOLD}_{FACET}.png", bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
