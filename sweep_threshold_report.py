"""Help pick training-loss / probe-AUROC thresholds ("did the model solve the
task", "is the model hidden from a linear probe") by rendering diagnostic
plots for a handful of representative runs at each percentile of each
metric's distribution, across runs/sweep3_lam*_tr* (lam0 excluded -- that
sweep is still running).

Loss percentiles get a learned-function-curves plot each (from
adversarial_report.plot_learned_curves); AUROC percentiles get a probe
histogram+ROC plot and a PCA/residual plot (local, logreg-only trims of
probe_lib.plot_probe / plot_probe_pca, which also plot a difference-of-means
comparison this report doesn't need), both scored at PROBE_LAYER -- the layer
the adversarial penalty is actually applied to during training.

"Loss" here is the task loss alone (data.eval_task_loss), freshly recomputed
from each checkpoint -- not the combined lam-weighted training-loss field
logged to history.jsonl, and not read from history.jsonl at all (some runs'
final checkpoint predates their last history record, a past checkpoint-saving
bug).

Settings live in the constants below rather than a CLI -- this script's
shape is still changing, so argparse would just be churn for now. Plots are
written under plot/sweep3/; a summary table prints to console."""

RUN_GLOB = "sweep3_lam*_tr*"
EXCLUDE_LAMBDAS = {}  # lam0 sweep is still running
EXCLUDE_TRIAL_ABOVE = 9  # for partial trials greater than this index
CKPT = "last"  # "last" or "best", matching runs/<tag>/checkpoints/<CKPT>.pt
PROBE_LAYER = 2  # matches adversarial.penalty_layers in these runs' config.json
PERCENTILES = [10, 25, 50, 75, 90]
N_PER_PERCENTILE = 3  # a few runs per bucket, not just the nearest one, so a single outlier run doesn't set the impression for its whole percentile
TASK_LOSS_N_EVAL = 50_000  # fresh examples per run for the recomputed task loss
TASK_LOSS_NOISE_MULT = 1.0  # multiplier on the checkpoint's own resid_noise_std, matching the noise the model trained under (see EVAL_NOISE_MULT for the probe-eval analog, tuned separately)
PROBE_N_TRAIN = 5000  # per class; smaller than adversarial_report's default (20_000) since a probe gets refit per selected run, across many runs
PROBE_N_TEST = 10_000  # per class
PROBE_BACKEND = "auto"
PROBE_BACKEND = "newton"
EVAL_NOISE_MULT = 0.5
SEED = 20260718
OUT_DIR = "plot/sweep3"

import glob
import os
import re

import matplotlib.pyplot as plt
import torch

from adversarial_report import plot_learned_curves
from data import eval_task_loss
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
        eval_noise_std=eval_noise_std,
    )
    return plot_inputs[PROBE_LAYER]


def _probe_auroc(pi: dict) -> float:
    probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
    return boundary_auroc(probe, pi["X_te"], pi["y_te"])


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


def _plot_probe_hist_auroc(
    tag, layers, probe: LinearBoundary, X_test, y_test, plot_dir
):
    """Trimmed variant of probe_lib.plot_probe: this report only cares about
    the logreg probe (no difference-of-means comparison), so it's just a
    logreg decision-function histogram plus its ROC/AUROC
    ({tag}_L{layers}_probe.png)."""
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
    return save_plot(fig, plot_dir, f"{tag}_L{layer_str}_probe.png")


def _plot_probe_pca_resid(tag, layers, probe: LinearBoundary, X_test, y_test, plot_dir):
    """Trimmed variant of probe_lib.plot_probe_pca: logreg-only companion to
    `_plot_probe_hist_auroc`, dropping the difference-of-means residual panel
    -- a shared top-2-component PCA plus the logreg direction projected out
    and PC1 of what's left ({tag}_L{layers}_probe_pca.png)."""
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
    return save_plot(fig, plot_dir, f"{tag}_L{layer_str}_probe_pca.png")


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
            probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
            _plot_probe_hist_auroc(
                label, [PROBE_LAYER], probe, pi["X_te"], pi["y_te"], out_dir
            )
            _plot_probe_pca_resid(
                label, [PROBE_LAYER], probe, pi["X_te"], pi["y_te"], out_dir
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
