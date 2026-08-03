"""Help pick training-loss / probe-AUROC thresholds ("did the model solve the
task", "is the model hidden from a linear probe") by rendering diagnostic
plots for a handful of representative runs at each percentile of each
metric's distribution, across runs/sweep3_lam*_tr* (lam0 excluded -- that
sweep is still running).

Loss percentiles get a learned-function-curves plot each (from
adversarial_report.plot_learned_curves); AUROC percentiles get a probe
histogram+ROC plot and a logreg/PCA residual plot (from probe_lib.plot_probe
/ plot_probe_pca), both scored at PROBE_LAYER -- the layer the adversarial
penalty is actually applied to during training.

Settings live in the constants below rather than a CLI -- this script's
shape is still changing, so argparse would just be churn for now. Plots are
written under plot/sweep3/; a summary table prints to console."""

RUN_GLOB = "sweep3_lam*_tr*"
EXCLUDE_LAMBDAS = {}  # lam0 sweep is still running
EXCLUDE_TRIAL_ABOVE = 9  # for partial trials greater than this index
CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
PROBE_LAYER = 2  # matches adversarial.penalty_layers in these runs' config.json
LOSS_LOWPASS_WINDOW = 2000  # matches sweep_report.py's smoothing window
PERCENTILES = [10, 25, 50, 75, 90]
N_PER_PERCENTILE = 3  # a few runs per bucket, not just the nearest one, so a single outlier run doesn't set the impression for its whole percentile
PROBE_N_TRAIN = 5000  # per class; smaller than adversarial_report's default (20_000) since a probe gets refit per selected run, across many runs
PROBE_N_TEST = 10_000  # per class
PROBE_BACKEND = "auto"
PROBE_BACKEND = "newton"
EVAL_NOISE_MULT = 0.5
SEED = 20260718
OUT_DIR = "plot/sweep3"

import glob
import json
import os
import re

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from adversarial_report import (
    _binary_probe_metrics_all_layers,
    _resolve_adv_config,
    plot_learned_curves,
)
from paths import log_dir
from probe_backend import resolve_probe_backend
from probe_lib import LinearBoundary, load_model
from probe_lib import plot_probe as plot_probe_separation
from probe_lib import plot_probe_pca as plot_probe_pca_separation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TAG_RE = re.compile(r"sweep3_lam([0-9.]+)_tr(\d+)$")


def _discover_tags() -> list[str]:
    tags = []
    for path in sorted(glob.glob(os.path.join("runs", RUN_GLOB))):
        tag = os.path.basename(path)
        m = TAG_RE.match(tag)
        if not m:
            continue
        if float(m.group(1)) in EXCLUDE_LAMBDAS:
            continue
        if int(m.group(2)) > EXCLUDE_TRIAL_ABOVE:
            continue
        tags.append(tag)
    return tags


def _load_history(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """(iters, loss) from a run's logs/history.jsonl. Skips the trailing
    'final' summary record, which repeats the last iter with loss fields
    nulled out."""
    iters, losses = [], []
    with open(os.path.join(log_dir(tag), "history.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("final"):
                continue
            iters.append(rec["iter"])
            losses.append(rec["loss"])
    return np.array(iters), np.array(losses)


# def _running_mean(iters: np.ndarray, values: np.ndarray, window: int) -> np.ndarray:
#     """Causal low-pass: mean value among logged points within the trailing
#     `window` iterations (not `window` logged points -- --log-interval
#     defaults to 100 and can vary run to run, so this has to key off `iters`,
#     not array index). Matches sweep_report.py's smoothing."""
#     out = np.empty_like(values, dtype=float)
#     lo = 0
#     for i in range(len(iters)):
#         cutoff = iters[i] - window + 1
#         while iters[lo] < cutoff:
#             lo += 1
#         out[i] = values[lo : i + 1].mean()
#     return out


def _final_loss(tag: str) -> float:
    iters, losses = _load_history(tag)
    return float(losses[-1])
    # return float(_running_mean(iters, losses, LOSS_LOWPASS_WINDOW)[-1])


def _fit_probe(tag: str, g: torch.Generator, probe_backend_name: str) -> dict | None:
    """Fit a fresh probe at PROBE_LAYER for `tag`'s CKPT checkpoint and return
    its `plot_inputs` (see _binary_probe_metrics_all_layers) -- the single
    source of truth for both AUROC sorting/filenames and the by_auroc
    diagnostic plots, run once per run. (Previously these used two different
    probes -- the checkpoint's own stored training-time probe for sorting vs.
    a freshly-refit one for the plots -- so the AUROC in a plot's filename
    never matched the AUROC in its own title.) None for a checkpoint with no
    adversarial config (nothing to probe for)."""
    model, ck = load_model(tag, CKPT, DEVICE)
    adv_cfg = _resolve_adv_config(ck)
    if adv_cfg is None:
        return None
    eval_noise_std = adv_cfg.resid_noise_std * EVAL_NOISE_MULT
    _metrics, plot_inputs = _binary_probe_metrics_all_layers(
        model,
        1.0,
        2.0,
        [PROBE_LAYER],
        PROBE_N_TRAIN,
        PROBE_N_TEST,
        g,
        probe_backend_name,
        desc=tag,
        eval_noise_std=eval_noise_std,
    )
    return plot_inputs[PROBE_LAYER]


def _probe_auroc(pi: dict) -> float:
    probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
    return roc_auc_score(pi["y_te"], probe.score(pi["X_te"]))


def _select_percentile_indices(
    n: int, percentiles: list[int], k: int
) -> dict[int, list[int]]:
    """For each target percentile, up to `k` rank indices into a length-`n`
    sorted array, spread around the percentile (clipped/deduped at the
    array's edges) rather than just the single nearest rank."""
    selected = {}
    for p in percentiles:
        center = round(p / 100 * (n - 1))
        offsets = range(-(k // 2), k - k // 2)
        idxs = sorted({min(max(center + o, 0), n - 1) for o in offsets})
        selected[p] = idxs
    return selected


def _make_curve_plots(tags_by_loss: list[str], losses: list[float]) -> None:
    n = len(tags_by_loss)
    out_dir = os.path.join(OUT_DIR, "by_loss")
    os.makedirs(out_dir, exist_ok=True)
    for p, idxs in _select_percentile_indices(n, PERCENTILES, N_PER_PERCENTILE).items():
        for idx in idxs:
            tag = tags_by_loss[idx]
            model, _ck = load_model(tag, CKPT, DEVICE)
            label = f"p{p:02d}_rank{idx:03d}_{tag}_loss{losses[idx]:.4g}"
            plot_learned_curves(model, label, out_dir)


def _make_probe_plots(
    tags_by_auroc: list[str], aurocs: list[float], probe_fits: dict[str, dict]
) -> None:
    n = len(tags_by_auroc)
    out_dir = os.path.join(OUT_DIR, "by_auroc")
    os.makedirs(out_dir, exist_ok=True)
    for p, idxs in _select_percentile_indices(n, PERCENTILES, N_PER_PERCENTILE).items():
        for idx in idxs:
            tag = tags_by_auroc[idx]
            pi = probe_fits[tag]
            label = f"p{p:02d}_rank{idx:03d}_{tag}_auroc{aurocs[idx]:.4f}"
            # LinearBoundary scores as `w . x + b`, not `w . x - threshold`, so
            # its bias is the midpoint threshold negated (matches
            # adversarial_report._make_plots).
            dom = LinearBoundary(pi["w_dom"], -pi["midpoint"])
            probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
            plot_probe_separation(
                label, [PROBE_LAYER], dom, probe, pi["X_te"], pi["y_te"], out_dir
            )
            plot_probe_pca_separation(
                label, [PROBE_LAYER], dom, probe, pi["X_te"], pi["y_te"], out_dir
            )


def main():
    tags = _discover_tags()
    print(
        f"found {len(tags)} runs matching {RUN_GLOB!r} (excluding lam in {EXCLUDE_LAMBDAS})"
    )

    losses = {t: _final_loss(t) for t in tags}

    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    probe_backend_name = resolve_probe_backend(PROBE_BACKEND, DEVICE)
    probe_fits = {t: _fit_probe(t, g, probe_backend_name) for t in tags}
    aurocs = {
        t: (_probe_auroc(pi) if pi is not None else None)
        for t, pi in probe_fits.items()
    }

    missing = [t for t in tags if aurocs[t] is None]
    if missing:
        print(f"no adversarial config, excluded from AUROC ranking: {missing}")

    tags_by_loss = sorted(tags, key=lambda t: losses[t])
    tags_by_auroc = sorted(
        (t for t in tags if aurocs[t] is not None), key=lambda t: aurocs[t]
    )

    print(f"\n{'tag':30s} {'loss':>12s} {'auroc':>8s}")
    for t in tags_by_loss:
        a = aurocs[t]
        a_str = f"{a:.4f}" if a is not None else "n/a"
        print(f"{t:30s} {losses[t]:12.6g} {a_str:>8s}")

    _make_curve_plots(tags_by_loss, [losses[t] for t in tags_by_loss])
    _make_probe_plots(tags_by_auroc, [aurocs[t] for t in tags_by_auroc], probe_fits)


if __name__ == "__main__":
    main()
