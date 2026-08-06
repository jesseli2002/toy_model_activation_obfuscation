"""Help pick training-loss / probe-AUROC thresholds ("did the model solve the
task", "is the model hidden from a linear probe") by rendering diagnostic
plots for every RANK_STEP-th run in each metric's sorted order, across
runs/sweep3_lam*_tr* (lam0 excluded -- that sweep is still running).

Loss and one-hot-loss ranks each get a learned-function-curves plot (from
adversarial_report.plot_learned_curves); AUROC ranks get a probe
histogram+ROC plot and a PCA/residual plot (local, logreg-only trims of
probe_lib.plot_probe / plot_probe_pca, which also plot a difference-of-means
comparison this report doesn't need), both scored at PROBE_LAYER -- the layer
the adversarial penalty is actually applied to during training.

"Loss" here is the task loss alone (data.eval_task_loss), freshly recomputed
from each checkpoint -- not the combined lam-weighted training-loss field
logged to history.jsonl, and not read from history.jsonl at all (some runs'
final checkpoint predates their last history record, a past checkpoint-saving
bug).

"One-hot loss" (data.eval_one_hot_loss) is the same task loss but evaluated
OOD: only one input coordinate is nonzero per example (the rest held at 0),
as in plot_learned_curves's curve construction, instead of all coordinates
simultaneously nonzero as in training.

Settings live in the constants below rather than a CLI -- this script's
shape is still changing, so argparse would just be churn for now. Plots are
written under plot/sweep3/; a summary table prints to console."""

import numpy as np
import re

RUN_GLOB = "sweep7_lam*_tr*"
OUT_DIR = "plot/sweep7"
TAG_RE = re.compile(r"sweep7_lam([0-9\.]+)_tr(\d+)$")
EXCLUDE_LAMBDAS = {}  #  for partial lambdas currently running
EXCLUDE_TRIAL_ABOVE = 10  # for partial trials greater than this index
CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
PROBE_LAYER = 2  # matches adversarial.penalty_layers in these runs' config.json
TASK_LOSS_N_EVAL = 50_000  # fresh examples per run for the recomputed task loss
TASK_LOSS_NOISE_MULT = 1.0  # multiplier on the checkpoint's own resid_noise_std, matching the noise the model trained under (see EVAL_NOISE_MULT for the probe-eval analog, tuned separately)
ONE_HOT_LOSS_N_EVAL = 50_000  # fresh examples per run for the one-hot OOD loss
PROBE_N_TRAIN = 5000  # per class; smaller than adversarial_report's default (20_000) since a probe gets refit per selected run, across many runs
LOSS_LOWPASS_WINDOW = 2000  # matches sweep_report.py's smoothing window

RANK_STEP = (
    7  # plot every Nth rank (0-indexed, ascending) in each metric's sorted order
)
PROBE_N_TRAIN = 10000  # per class; smaller than adversarial_report's default (20_000) since a probe gets refit per selected run, across many runs
PROBE_N_TEST = 10_000  # per class
PROBE_BACKEND = "newton"
PROBE_EVAL_NOISE_MULT = 0.5  # multiplier on resid_noise_std when retraining probe
SEED = 20260718

import glob
import os

import matplotlib.pyplot as plt
import torch

from adversarial_report import plot_learned_curves
from data import eval_one_hot_loss, eval_task_loss
from probe_backend import resolve_probe_backend
from probe_lib import (
    LinearBoundary,
    binary_probe_metrics_all_layers,
    boundary_auroc,
    load_model,
    resolve_adv_config,
    save_plot,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        noise_std=noise_std,
    )


def _one_hot_loss(tag: str, g: torch.Generator) -> float:
    """One-hot OOD loss (data.eval_one_hot_loss), freshly recomputed at CKPT
    -- only one input coordinate nonzero per example, unlike _final_loss's
    plain training-distribution eval. Uses the checkpoint's own
    resid_noise_std (TASK_LOSS_NOISE_MULT) so the two loss columns differ
    only in the input distribution, not the noise."""
    model, ck = load_model(tag, CKPT, DEVICE)
    adv_cfg = resolve_adv_config(ck)
    noise_std = (
        adv_cfg.resid_noise_std * TASK_LOSS_NOISE_MULT if adv_cfg is not None else 0.0
    )
    return eval_one_hot_loss(
        model,
        g,
        DEVICE,
        n=ONE_HOT_LOSS_N_EVAL,
        noise_std=noise_std,
    )


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
    adv_cfg = resolve_adv_config(ck)
    if adv_cfg is None:
        return None
    eval_noise_std = adv_cfg.resid_noise_std * PROBE_EVAL_NOISE_MULT
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
        eval_noise_std=eval_noise_std,
    )
    return plot_inputs[PROBE_LAYER]


def _probe_auroc(pi: dict) -> float:
    probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
    return boundary_auroc(probe, pi["X_te"], pi["y_te"])


def _select_rank_indices(n: int) -> list[int]:
    """Every RANK_STEP-th rank index into a length-`n` sorted array, plus the
    last rank if it wasn't already hit, so the single best run always gets a
    plot."""
    idxs = list(range(0, n, RANK_STEP))
    if idxs and idxs[-1] != n - 1:
        idxs.append(n - 1)
    return idxs


def _make_curve_plots(
    tags_by_loss: list[str],
    losses: list[float],
    out_subdir: str = "by_loss",
    metric_tag: str = "loss",
) -> None:
    n = len(tags_by_loss)
    out_dir = os.path.join(OUT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    for idx in _select_rank_indices(n):
        tag = tags_by_loss[idx]
        model, _ck = load_model(tag, CKPT, DEVICE)
        title = f"rank{idx:03d}_{tag}_{metric_tag}{losses[idx]:.4g}"
        file_tag = f"rank{idx:03d}_{metric_tag}{losses[idx]:.4g}"
        plot_learned_curves(model, title, out_dir, filename_tag=file_tag)


def _plot_probe_hist_auroc(
    tag, layers, probe: LinearBoundary, X_test, y_test, plot_dir, file_tag=None
):
    """Trimmed variant of probe_lib.plot_probe: this report only cares about
    the logreg probe (no difference-of-means comparison), so it's just a
    logreg decision-function histogram plus its ROC/AUROC. `tag` names the
    plot title; `file_tag` (defaults to `tag`) names the output file
    ({file_tag}_L{layers}_probe.png)."""
    from sklearn.metrics import roc_auc_score, roc_curve

    proj = probe.score(X_test)
    auroc = roc_auc_score(y_test, proj)
    fpr, tpr, _ = roc_curve(y_test, proj)

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0
    layer_str = "-".join(str(i) for i in layers)

    fig, (ax_hist, ax_roc) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax_hist.hist(proj[lo_mask], bins=60, alpha=0.6, label="c=1")
    ax_hist.hist(proj[hi_mask], bins=60, alpha=0.6, label="c=2")
    percentile_5, percentile_95 = np.percentile(proj, [5, 95])
    percentile_diff = percentile_95 - percentile_5
    ax_hist.set_xlim(
        [
            percentile_5 - percentile_diff * 0.5,
            percentile_95 + percentile_diff * 0.5,
        ]
    )
    ax_hist.axvline(0.0, color="k", ls="--", lw=1, label="threshold")
    ax_hist.set_title("logreg decision function")
    ax_hist.set_xlabel("projection (test set)")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    ax_roc.plot(fpr, tpr, label="logreg")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"AUROC: {auroc:.3f}")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_aspect("equal", adjustable="box")

    fig.suptitle(f"probe separation ({tag}, layers={layer_str})")
    fig.tight_layout()
    fname = file_tag if file_tag is not None else tag
    return save_plot(fig, plot_dir, f"{fname}_L{layer_str}_probe.png")


def _plot_probe_pca_resid(
    tag, layers, probe: LinearBoundary, X_test, y_test, plot_dir, file_tag=None
):
    """Trimmed variant of probe_lib.plot_probe_pca: logreg-only companion to
    `_plot_probe_hist_auroc`, dropping the difference-of-means residual panel
    -- a shared top-2-component PCA plus the logreg direction projected out
    and PC1 of what's left. `tag` names the plot title; `file_tag` (defaults
    to `tag`) names the output file ({file_tag}_L{layers}_probe_pca.png)."""
    from sklearn.decomposition import PCA

    pca_xy = PCA(n_components=2).fit_transform(X_test)

    w_hat = probe.w / np.linalg.norm(probe.w)
    X_resid = X_test - np.outer(X_test @ w_hat, w_hat)
    pc1_resid = PCA(n_components=1).fit_transform(X_resid)[:, 0]
    proj_raw = X_test @ w_hat
    raw_threshold = float(-probe.b / np.linalg.norm(probe.w))

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0
    layer_str = "-".join(str(i) for i in layers)

    fig, (ax_pca, ax_resid) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax_pca.scatter(pca_xy[lo_mask, 0], pca_xy[lo_mask, 1], s=4, alpha=0.4, label="c=1")
    ax_pca.scatter(pca_xy[hi_mask, 0], pca_xy[hi_mask, 1], s=4, alpha=0.4, label="c=2")
    ax_pca.set_title("PCA (top 2 components)")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend(fontsize=8)
    ax_pca.grid(True, alpha=0.3)
    ax_pca.set_aspect("equal", adjustable="datalim")

    ax_resid.scatter(proj_raw[lo_mask], pc1_resid[lo_mask], s=4, alpha=0.4, label="c=1")
    ax_resid.scatter(proj_raw[hi_mask], pc1_resid[hi_mask], s=4, alpha=0.4, label="c=2")
    ax_resid.axvline(raw_threshold, color="k", ls="--", lw=1, label="threshold")
    ax_resid.set_title("logreg vs residual PCA")
    ax_resid.set_xlabel("logreg projection (data coords)")
    ax_resid.set_ylabel("PC1 of orthogonal residual")
    ax_resid.legend(fontsize=8)
    ax_resid.grid(True, alpha=0.3)

    fig.suptitle(f"probe separation, PCA ({tag}, layers={layer_str})")
    fig.tight_layout()
    fname = file_tag if file_tag is not None else tag
    return save_plot(fig, plot_dir, f"{fname}_L{layer_str}_probe_pca.png")


def _make_probe_plots(
    tags_by_auroc: list[str], aurocs: list[float], probe_fits: dict[str, dict]
) -> None:
    n = len(tags_by_auroc)
    out_dir = os.path.join(OUT_DIR, "by_auroc")
    os.makedirs(out_dir, exist_ok=True)
    for idx in _select_rank_indices(n):
        tag = tags_by_auroc[idx]
        pi = probe_fits[tag]
        title = f"rank{idx:03d}_{tag}_auroc{aurocs[idx]:.4f}"
        file_tag = f"rank{idx:03d}_auroc{aurocs[idx]:.4f}"
        probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
        _plot_probe_hist_auroc(
            title, [PROBE_LAYER], probe, pi["X_te"], pi["y_te"], out_dir, file_tag
        )
        _plot_probe_pca_resid(
            title, [PROBE_LAYER], probe, pi["X_te"], pi["y_te"], out_dir, file_tag
        )


def main():
    tags = _discover_tags()
    print(
        f"found {len(tags)} runs matching {RUN_GLOB!r} (excluding lam in {EXCLUDE_LAMBDAS})"
    )

    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    losses = {t: _final_loss(t, g) for t in tags}

    probe_backend_name = resolve_probe_backend(PROBE_BACKEND, DEVICE)
    probe_fits = {t: _fit_probe(t, g, probe_backend_name) for t in tags}
    aurocs = {
        t: (_probe_auroc(pi) if pi is not None else None)
        for t, pi in probe_fits.items()
    }

    missing = [t for t in tags if aurocs[t] is None]
    if missing:
        print(f"no adversarial config, excluded from AUROC ranking: {missing}")

    # Own generator (not `g`) so this pass doesn't shift the probe-fit draws
    # above -- appending here keeps `aurocs` and its plot filenames unchanged
    # from before this metric was added.
    g_one_hot = torch.Generator(device=DEVICE).manual_seed(SEED)
    one_hot_losses = {t: _one_hot_loss(t, g_one_hot) for t in tags}

    tags_by_loss = sorted(tags, key=lambda t: losses[t])
    tags_by_one_hot_loss = sorted(tags, key=lambda t: one_hot_losses[t])
    tags_by_auroc = sorted(
        (t for t in tags if aurocs[t] is not None), key=lambda t: aurocs[t]
    )

    print(f"\n{'tag':30s} {'loss':>12s} {'one_hot_loss':>14s} {'auroc':>8s}")
    for t in tags_by_loss:
        a = aurocs[t]
        a_str = f"{a:.4f}" if a is not None else "n/a"
        print(f"{t:30s} {losses[t]:12.6g} {one_hot_losses[t]:14.6g} {a_str:>8s}")

    _make_curve_plots(tags_by_loss, [losses[t] for t in tags_by_loss])
    _make_curve_plots(
        tags_by_one_hot_loss,
        [one_hot_losses[t] for t in tags_by_one_hot_loss],
        out_subdir="by_one_hot_loss",
        metric_tag="ohloss",
    )
    _make_probe_plots(tags_by_auroc, [aurocs[t] for t in tags_by_auroc], probe_fits)


if __name__ == "__main__":
    main()
