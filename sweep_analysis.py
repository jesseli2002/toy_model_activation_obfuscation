"""Aggregate runs/sweep3_lam*_tr* into a stacked-fraction-vs-lambda chart:
for each lambda, what fraction of its trials failed the task (loss), and
among those that succeeded, how well they hid from the layer-PROBE_LAYER
linear probe (AUROC), at one or more AUROC thresholds.

x-axis is lam / (1 - lam) -- the probe-to-task loss weight ratio (see
`(1 - lam_eff) * l_task + lam_eff * l_probe` in
train_adversarial_logreg._compute_losses) -- rather than lam itself, since
it's unbounded above (lam is capped at 1) and matches what the adversarial
penalty actually trades off against. Plotted symlog so lam=0 (ratio 0) has a
finite position while still spreading out the many-orders-of-magnitude tail.

LOSS_THRESHOLD / AUROC_THRESHOLDS are provisional -- a separate exercise is
picking the final values -- so they're constants here, not a CLI, alongside
everything else that may still change. Prints a summary table; plt.show()
only, nothing written to disk.

"Loss" and "AUROC" are both freshly recomputed at CKPT -- loss via
data.eval_task_loss (deliberately excluding the adversarial/probe penalty,
since this only cares about task performance), and AUROC via a freshly
refit probe at PROBE_LAYER -- rather than read from history.jsonl or the
checkpoint's own stored training-time probe. See sweep_threshold_report.py,
which uses the same recomputation for the same reasons (some runs' final
checkpoint predates their last history record).

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

PLOT_DIR = "plot/sweep7"
CACHE_PATH = "plot/sweep7/metrics_cache.json"
RUN_GLOB = "sweep7_lam*_tr*"
TAG_RE = re.compile(r"sweep7_lam([0-9\.]+)_tr(\d+)$")
EXCLUDE_LAMBDAS: set[float] = set()
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


def _discover_tags_by_lambda() -> dict[float, list[str]]:
    by_lambda: dict[float, list[str]] = {}
    for path in sorted(glob.glob(os.path.join("runs", RUN_GLOB))):
        tag = os.path.basename(path)
        m = TAG_RE.match(tag)
        if not m:
            continue
        lam = float(m.group(1))
        if lam in EXCLUDE_LAMBDAS:
            continue
        by_lambda.setdefault(lam, []).append(tag)
    return by_lambda


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
            labels.append(f"not hidden\n(auroc >= {t:g})")
        else:
            labels.append(f"partially hidden\n({t:g} <= auroc < {prev:g})")
        prev = t
    labels.append(f"hidden\n(auroc < {prev:g})")
    return labels


def _band_colors(thresholds: list[float]) -> list[str]:
    n_hiding_bins = len(thresholds) + 1
    idxs = np.linspace(0, len(SEQ_RAMP) - 1, n_hiding_bins).astype(int)
    return [FAILED_COLOR] + [SEQ_RAMP[i] for i in idxs]


def _smoke_data(n_bands: int) -> tuple[list[float], list[np.ndarray]]:
    """Synthetic (ratios, fractions) standing in for a real sweep's results,
    just to preview the plot's layout/styling without a full analysis run."""
    rng = np.random.default_rng(SEED)
    lambdas = np.concatenate([[0.0], np.geomspace(1e-4, 0.9, 12)])
    ratios = (lambdas / (1 - lambdas)).tolist()
    fractions = []
    for ratio in ratios:
        # p: 0..1 knob, low ratio -> low p, high ratio -> high p
        p = np.clip((np.log10(ratio + 1e-4) + 4) / 4, 0, 1)
        failed = 0.05 + 0.5 * p
        weights = np.linspace(1 - p, p, n_bands - 1) + 0.1
        weights = weights / weights.sum() * (1 - failed)
        counts = np.clip(np.concatenate([[failed], weights]), 0.001, None)
        counts += rng.normal(0, 0.02, size=n_bands)
        counts = np.clip(counts, 0.001, None)
        fractions.append(counts / counts.sum())
    return ratios, fractions


def main(clear_cache: bool = False):
    if clear_cache and os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    thresholds = sorted(AUROC_THRESHOLDS)
    n_bands = len(thresholds) + 2

    if SMOKE:
        ratios, fractions = _smoke_data(n_bands)
    else:
        by_lambda = _discover_tags_by_lambda()
        cache = _load_cache()
        ratios = []
        fractions = []  # one length-n_bands array per lambda
        # One RNG across every lambda, so its draws (eval noise, probe
        # train/test sampling) are reproducible across a full run of the script.
        g = torch.Generator(device=DEVICE).manual_seed(SEED)
        probe_backend_name = resolve_probe_backend(PROBE_BACKEND, DEVICE)

        print(f"{'lambda':>10s} {'ratio':>10s} {'n_ok':>5s} {'n_total':>7s}")
        for lam in sorted(by_lambda):
            tags = by_lambda[lam]
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
            if n_ok == 0:
                print(f"  {lam}: no usable runs, skipped")
                continue
            ratio = lam / (1 - lam)
            ratios.append(ratio)
            fractions.append(counts / n_ok)
            print(f"{lam:10g} {ratio:10.4g} {n_ok:5d} {len(tags):7d}")

    order = np.argsort(ratios)
    x = np.array(ratios)[order]
    frac_matrix = np.array(fractions)[order]  # (n_lambda, n_bands)

    labels = _band_labels(thresholds)
    colors = _band_colors(thresholds)

    fig, ax = plt.subplots(figsize=(8, 5))
    # stackplot stacks bottom-to-top in argument order; we want "failed" on
    # top and "most hidden" at the bottom, so feed bands in reverse.
    stacked = frac_matrix.T[::-1]
    ax.stackplot(
        x,
        *stacked,
        labels=labels[::-1],
        colors=colors[::-1],
        alpha=0.9,
    )
    # stackplot doesn't support markers itself; overlay them on each band's
    # upper boundary (cumulative sum) to show the underlying data points.
    cum = np.cumsum(stacked, axis=0)
    for row, color in zip(cum[:-1], colors[::-1][:-1]):  # don't plot top row, always 1
        ax.plot(
            x,
            row,
            marker="o",
            ls="-",
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

    SYMLOG_LINTHRESH = x[1]  # linear-region half-width (in ratio units) around lam=0
    ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=0.5)
    # ax.axvline(x[1] / 2, ls="--", lw=1, color="#52514e")
    ax.axvline(x[1] / 2, ls="--", lw=1, color="black")
    ax.set_xlabel(r"$\lambda / (1 - \lambda)$  (probe-loss weight / task-loss weight)")
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, x[-1])
    ax.set_title(f"Outcome vs. $\\lambda$")
    # ax.set_title(f"Outcome vs. $\\lambda$ \n({RUN_GLOB})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(f"{PLOT_DIR}/lam_sweep_task{LOSS_THRESHOLD}.png", bbox_inches="tight")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(clear_cache=args.clear_cache)
