"""Help pick training-loss / probe-AUROC thresholds ("did the model solve the
task", "is the model hidden from a linear probe") by rendering diagnostic
plots for every RANK_STEP-th run in each metric's sorted order, across a
lambda-sweep (see RUN_GLOB below, and EXCLUDE_LAMBDAS for any arm excluded).

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

"N-hot loss" (data.eval_n_hot_loss) is the same task loss but evaluated OOD:
only N input coordinates are nonzero per example (the rest held at 0),
instead of all coordinates simultaneously nonzero as in training. N=1 is the
one-hot case, matching plot_learned_curves's curve construction; N_HOT_VALUES
also sweeps N=2,4,8 to see how loss degrades as the eval distribution
approaches the training distribution's density of nonzero coordinates.

Settings live in the constants below rather than a CLI -- this script's
shape is still changing, so argparse would just be churn for now. Plots are
written under OUT_DIR; a summary table prints to console."""

import numpy as np
import re

RUN_GLOB = "sweep7_lam*_tr*"
OUT_DIR = "plot/sweep7"
TAG_RE = re.compile(r"sweep7_lam([0-9\.]+)_tr(\d+)$")
EXCLUDE_LAMBDAS = {}  #  for partial lambdas currently running
EXCLUDE_TRIAL_ABOVE = 10  # for partial trials greater than this index
CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
PROBE_LAYER = 2  # matches adversarial.penalty_layers in these runs' config.json
N_HOT_VALUES = (1, 2, 4, 8)  # 1 = one-hot; larger N approaches the training density

# plot every Nth rank (0-indexed, ascending) in each metric's sorted order
RANK_STEP = 1

import glob
import os

import matplotlib.pyplot as plt
import torch

from adversarial_report import plot_learned_curves
from checkpoint_lib import load_model
from probe_lib import LinearBoundary, save_plot
from sweep_lib.metrics import CACHE_PATH, MetricSpec, MetricStore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPEC = MetricSpec(ckpt=CKPT, probe_layer=PROBE_LAYER)


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
    tags_by_auroc: list[str], aurocs: list[float], store: MetricStore
) -> None:
    """Diagnostic plots for the selected ranks. The probe is refit here rather
    than kept from the AUROC pass, since plot_inputs holds a whole test set per
    run; store.probe_fit seeds it to match the cached AUROC, so a plot's
    filename and its title agree."""
    n = len(tags_by_auroc)
    out_dir = os.path.join(OUT_DIR, "by_auroc")
    os.makedirs(out_dir, exist_ok=True)
    for idx in _select_rank_indices(n):
        tag = tags_by_auroc[idx]
        layer, pi = store.probe_fit(tag)
        title = f"rank{idx:03d}_{tag}_auroc{aurocs[idx]:.4f}"
        file_tag = f"rank{idx:03d}_auroc{aurocs[idx]:.4f}"
        probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
        _plot_probe_hist_auroc(
            title, [layer], probe, pi["X_te"], pi["y_te"], out_dir, file_tag
        )
        _plot_probe_pca_resid(
            title, [layer], probe, pi["X_te"], pi["y_te"], out_dir, file_tag
        )


def main():
    tags = _discover_tags()
    print(
        f"found {len(tags)} runs matching {RUN_GLOB!r} (excluding lam in {EXCLUDE_LAMBDAS})"
    )

    store = MetricStore(SPEC, CACHE_PATH, DEVICE)
    losses = {t: store.task_loss(t) for t in tags}
    aurocs = {t: store.auroc(t) for t in tags}

    missing = [t for t in tags if aurocs[t] is None]
    if missing:
        print(f"no adversarial config, excluded from AUROC ranking: {missing}")

    n_hot_losses = {
        n_hot: {t: store.n_hot_loss(t, n_hot) for t in tags} for n_hot in N_HOT_VALUES
    }

    tags_by_loss = sorted(tags, key=lambda t: losses[t])
    tags_by_n_hot_loss = {
        n_hot: sorted(tags, key=lambda t: n_hot_losses[n_hot][t])
        for n_hot in N_HOT_VALUES
    }
    tags_by_auroc = sorted(
        (t for t in tags if aurocs[t] is not None), key=lambda t: aurocs[t]
    )

    n_hot_headers = "".join(f"{f'{n}hot_loss':>14s}" for n in N_HOT_VALUES)
    print(f"\n{'tag':30s} {'loss':>12s}{n_hot_headers} {'auroc':>8s}")
    for t in tags_by_loss:
        a = aurocs[t]
        a_str = f"{a:.4f}" if a is not None else "n/a"
        n_hot_cols = "".join(
            f"{n_hot_losses[n_hot][t]:14.6g}" for n_hot in N_HOT_VALUES
        )
        print(f"{t:30s} {losses[t]:12.6g}{n_hot_cols} {a_str:>8s}")

    _make_curve_plots(tags_by_loss, [losses[t] for t in tags_by_loss])
    for n_hot in N_HOT_VALUES:
        _make_curve_plots(
            tags_by_n_hot_loss[n_hot],
            [n_hot_losses[n_hot][t] for t in tags_by_n_hot_loss[n_hot]],
            out_subdir=f"by_{n_hot}hot_loss",
            metric_tag=f"{n_hot}hloss",
        )
    _make_probe_plots(tags_by_auroc, [aurocs[t] for t in tags_by_auroc], store)


if __name__ == "__main__":
    main()
