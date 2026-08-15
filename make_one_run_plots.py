"""Publication plots for the hiding_experiment writeup's single-run result
(Results part 2): clean, presentation-ready figures with no debugging info
(tag names, checkpoint iters, ...) baked into their titles.

adversarial_report.py and the sweep_*.py scripts are diagnostic tools -- their
titles deliberately embed the tag/run so a reader flipping between many runs
knows which is which. This script is for the opposite situation: one run,
already chosen, going into a writeup. It reuses adversarial_report.py's and
probe_lib.py's data-computation helpers directly (so the numbers agree with
the diagnostic reports) but supplies its own plotting code with quieter
titles and different layouts (e.g. 2x2 instead of 1x4).

Structured the same way as adversarial_report.py: a data-computation phase
(`_run_analysis`) produces a `PublishData`, then a separate plotting phase
(`_make_plots`) turns it into figures -- so adding a plot never needs a new
forward pass through data already computed, and vice versa.

Run with no --tag to regenerate every figure the writeup uses, across all of
TAGS, in one invocation; pass --tag (repeatable) to instead run only against
specific tag(s), e.g. for exploring a run ad hoc. Not every plot produced ends
up embedded in the writeup -- see copy_plots.sh in the blog's hiding_experiment
directory for which ones do; the rest are cheap to produce alongside them and
worth keeping in case the writeup grows.
"""

import argparse

# Every tag the hiding_experiment writeup draws figures from.
TAGS = ["sweep7_lam0.1_tr0", "sweep3_lam0_tr0"]
STEER_LAYERS = [1, 2, 3, 4, 5]
# Eval-noise multipliers swept by the noise-isolation ROC plot, independent of
# --train-noise-mult/--eval-noise-mult (which set the noise regime for every
# other plot) -- see _run_noise_grid_analysis for the rationale.
NOISE_GRID_EVAL_MULTS = (0.0, 0.5, 1.0)
NOISE_GRID_LAYER = 2

# Multiplier on noise when plotting learned curves.
PLOT_LEARNED_CURVES_NOISE_MULT = 1
# Multiplier on noise when steering, so steer plots reflect the model's
# actual (noisy) forward pass instead of an unrealistically clean one.
PLOT_STEER_NOISE_MULT = 1

# Direction-magnitude multiples plotted by _plot_steer_direction_magnitude,
# keyed by tag. sweep3_lam0_tr0 (no probe optimization pressure) is the
# writeup's demonstration that 1x DoM steering alone already matches theory
# exactly; the default 2x line has nothing to add there and only clutters
# that specific figure.
DEFAULT_STEER_MAGS = [1.0, 2.0]
STEER_MAGS_BY_TAG = {
    "sweep3_lam0_tr0": [1.0],
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tag",
        type=str,
        action="append",
        help=f"run only against this tag; may be repeated. Default: every "
        f"writeup tag, {TAGS}.",
    )
    p.add_argument(
        "--ckpt",
        choices=["last", "best", "both"],
        default="both",
        help="which checkpoint(s) to plot; 'both' writes to separate "
        "publish/<tag>/<ckpt>/ subdirectories.",
    )
    p.add_argument(
        "--n-train",
        type=int,
        default=config.PROBE_EVAL_N_TRAIN,
        help="per class/set",
    )
    p.add_argument(
        "--n-test", type=int, default=config.PROBE_EVAL_N_TEST, help="per class/set"
    )
    p.add_argument("--n-err-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument(
        "--probe-backend",
        choices=config.PROBE_BACKEND_CHOICES,
        default="newton",
    )
    p.add_argument(
        "--eval-noise-mult",
        type=float,
        default=1.0,
        help="multiplier on the checkpoint's own adv_config.resid_noise_std, "
        "injected into the residual stream when EVALUATING probes -- "
        "replicates the noisy environment the model itself saw at train "
        "time. 0 disables.",
    )
    p.add_argument(
        "--train-noise-mult",
        type=float,
        default=1.0,
        help="multiplier on the checkpoint's own adv_config.resid_noise_std, "
        "injected into the residual stream when FITTING probes. Default 1, "
        "matching --eval-noise-mult, so plots reflect the model's actual "
        "training-time noise regime by default.",
    )
    p.add_argument("--show", action="store_true")
    return p.parse_args()


# parse_args early-exits on --help before the heavy imports below are reached.
if __name__ == "__main__":
    import config  # noqa: E402 (needs to be visible to parse_args' choices=)

    args = parse_args()

import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

import config
from adversarial_report import _steer_vectors
from data import eval_task_loss, sample_batch
from model import Noise
from probe_backend import resolve_probe_backend
from probe_lib import (
    LinearBoundary,
    binary_probe_metrics_all_layers,
    boundary_auroc,
    forward_steered,
    linear_y_reconstruction,
    load_model,
    resolve_adv_config,
    save_plot,
)

PUBLISH_DIR = "publish"

# Single knob for output resolution of every plot in this file. probe_lib.save_plot
# defers to rcParams["savefig.dpi"], so bumping this (e.g. to 300 for print-quality
# figures) is all that's needed.
plt.rcParams["savefig.dpi"] = 200


# ----------------------------------------------------------------------------
# Phase 1: data computation (no plotting)
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class PublishData:
    """Everything phase 2 (`_make_plots`) needs, all recomputed fresh against
    the checkpoint rather than pulled from training history."""

    task_loss: float
    err_samples: np.ndarray  # (n_err_samples,) per-input Euclidean error
    noise: Noise  # adv_cfg.resid_noise_std -- see _plot_learned_curves
    hidden_layers: list[int]
    gap_plot_inputs: (
        dict  # {layer: {"w_dom", "midpoint", "w_probe", "b_probe", "X_te", "y_te"}}
    )
    auroc: dict  # {layer: {"dom": auroc, "logreg": auroc}}
    linear_y_r2: dict  # {layer: R^2}
    noise_roc: dict  # {eval_mult: {"fpr", "tpr", "auroc"}}, NOISE_GRID_LAYER only


@torch.no_grad()
def _sample_errors(model, n, g, device, noise: Noise) -> np.ndarray:
    """Per-input RMS error between the model's task output and the true
    sat(x, c), over `n` fresh c ~ U[C_LOW, C_HIGH] inputs.

    RMS (mean over the num_x coordinates, not summed) rather than the raw
    Euclidean norm -- the latter grows like sqrt(num_x) for i.i.d.
    per-coordinate errors of fixed typical size, which would make the
    histogram's scale an artifact of num_x rather than of the model."""
    x_full, y = sample_batch(n, model.num_x, generator=g, device=device)
    pred = model.task_output(x_full, noise=noise, generator=g)
    return ((pred - y) ** 2).mean(dim=-1).sqrt().cpu().numpy()


def _isolated_layer_noise(model, base_std, isolate_layer, isolate_mult, batch_size, g):
    """Residual-stream noise tensor (see `ResidualMLP.generate_noise`) at
    `base_std` for every injection point, except the one landing on
    `isolate_layer`'s cache, which is scaled by `isolate_mult` instead."""
    noise = model.generate_noise(batch_size, base_std, g)
    noise[isolate_layer - 1] *= isolate_mult
    return noise


def _run_noise_grid_analysis(
    model, adv_cfg, args, g, device, probe_backend_name
) -> dict:
    """Logreg probe ROC curve at layer NOISE_GRID_LAYER, refit/reevaluated at
    every eval multiplier in NOISE_GRID_EVAL_MULTS -- separate from (and in
    addition to) the single train/eval noise regime the rest of the plots
    use. The fit pass always sees the model's own noise, uniform across
    layers, at multiplier 1. The eval pass also holds every layer at
    multiplier 1 except NOISE_GRID_LAYER, whose multiplier sweeps the grid --
    isolates the cost of the nonlinear encoding at that layer from our own
    injected noise there, rather than conflating it with an off-distribution
    model (which uniformly scaling eval noise everywhere would do)."""
    results = {}
    for eval_mult in tqdm(NOISE_GRID_EVAL_MULTS, desc="noise grid"):
        eval_noise = _isolated_layer_noise(
            model,
            adv_cfg.resid_noise_std,
            NOISE_GRID_LAYER,
            eval_mult,
            args.n_test,
            g,
        )
        _, plot_inputs = binary_probe_metrics_all_layers(
            model,
            1.0,
            2.0,
            [NOISE_GRID_LAYER],
            g,
            probe_backend_name,
            n_train=args.n_train,
            n_test=args.n_test,
            desc=f"eval={eval_mult:g}",
            train_noise=adv_cfg.resid_noise_std,
            eval_noise=eval_noise,
        )
        pi = plot_inputs[NOISE_GRID_LAYER]
        proj = LinearBoundary(pi["w_probe"], pi["b_probe"]).score(pi["X_te"])
        fpr, tpr, _ = roc_curve(pi["y_te"], proj)
        results[eval_mult] = {
            "fpr": fpr,
            "tpr": tpr,
            "auroc": roc_auc_score(pi["y_te"], proj),
        }
    return results


def _run_analysis(model, adv_cfg, args, g, device, probe_backend_name) -> PublishData:
    task_loss = eval_task_loss(
        model,
        g,
        device=device,
        x_p_outer=adv_cfg.x_p_outer,
        x_threshold=adv_cfg.x_threshold,
        noise=adv_cfg.resid_noise_std,
    )
    err_samples = _sample_errors(
        model, args.n_err_samples, g, device, noise=adv_cfg.resid_noise_std
    )

    hidden_layers = list(range(1, model.num_blocks))
    # See --train-noise-mult/--eval-noise-mult help.
    train_noise_std = adv_cfg.resid_noise_std * args.train_noise_mult
    eval_noise_std = adv_cfg.resid_noise_std * args.eval_noise_mult
    gap, gap_plot_inputs = binary_probe_metrics_all_layers(
        model,
        1.0,
        2.0,
        hidden_layers,
        g,
        probe_backend_name,
        n_train=args.n_train,
        n_test=args.n_test,
        desc="probe gap @ {1,2}",
        train_noise=train_noise_std,
        eval_noise=eval_noise_std,
    )
    del gap  # accuracy metrics; unused here (AUROC is recomputed below instead)

    auroc = {}
    for lyr in hidden_layers:
        pi = gap_plot_inputs[lyr]
        dom_boundary = LinearBoundary(pi["w_dom"], -pi["midpoint"])
        probe_boundary = LinearBoundary(pi["w_probe"], pi["b_probe"])
        auroc[lyr] = {
            "dom": boundary_auroc(dom_boundary, pi["X_te"], pi["y_te"]),
            "logreg": boundary_auroc(probe_boundary, pi["X_te"], pi["y_te"]),
        }

    linear_y_r2 = linear_y_reconstruction(
        model, model.num_x, model.num_blocks, args.n_train, args.n_test, g, device
    )

    noise_roc = _run_noise_grid_analysis(
        model, adv_cfg, args, g, device, probe_backend_name
    )

    return PublishData(
        task_loss=task_loss,
        err_samples=err_samples,
        noise=adv_cfg.resid_noise_std,
        hidden_layers=hidden_layers,
        gap_plot_inputs=gap_plot_inputs,
        auroc=auroc,
        linear_y_r2=linear_y_r2,
        noise_roc=noise_roc,
    )


# ----------------------------------------------------------------------------
# Phase 2: plots (titles deliberately omit the tag -- see module docstring)
# ----------------------------------------------------------------------------
@torch.no_grad()
def _plot_learned_curves(
    model, task_loss, plot_dir, tag, noise: Noise, filename: str, show=False
):
    """2x2 grid of learned y(x) per coordinate, one panel per fixed c."""
    c_values = (1.0, 1.333, 1.667, 2.0)
    num_x = model.num_x
    device = next(model.parameters()).device
    xs = torch.linspace(-3, 3, 400, device=device)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, c_values):
        for j in range(num_x):
            x = torch.zeros(len(xs), num_x, device=device)
            x[:, j] = xs
            x_full = torch.cat([x, torch.full((len(xs), 1), c, device=device)], dim=1)
            y = model.task_output(x_full, noise=noise)[:, j]
            ax.plot(xs.cpu().numpy(), y.cpu().numpy(), alpha=0.5, zorder=5)
        ax.plot(
            xs.cpu().numpy(),
            torch.clamp(xs, -c, c).cpu().numpy(),
            "k--",
            lw=1,
            label="target sat",
            zorder=2,
        )
        ax.set_title(f"c = {c:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.suptitle(f"Learned y(x, c)  (task loss = {task_loss:.3e})")
    fig.tight_layout()
    return save_plot(fig, plot_dir, filename=filename, close=not show)


def _plot_error_histogram(err_samples, plot_dir, tag, show=False):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(err_samples, bins=80, color="steelblue")
    ax.set_xlabel("RMS error (per coordinate)")
    ax.set_ylabel("count")
    ax.set_title(f"Prediction error distribution (n={len(err_samples):,})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_err_hist.png", close=not show)


def _plot_auroc_line(auroc, hidden_layers, plot_dir, tag, show=False):
    x = np.arange(len(hidden_layers))
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(hidden_layers)), 4.2))
    ax.plot(x, [auroc[l]["dom"] for l in hidden_layers], marker="o", label="DoM")
    ax.plot(x, [auroc[l]["logreg"] for l in hidden_layers], marker="o", label="LogReg")
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in hidden_layers])
    ax.set_ylabel("AUROC")
    ax.set_title("Probe AUROC by layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_auroc_bar.png", close=not show)


def _plot_roc_noise_grid(noise_roc, plot_dir, tag, show=False):
    """Logreg probe ROC curve at layer NOISE_GRID_LAYER, one curve per eval
    multiplier in NOISE_GRID_EVAL_MULTS overlaid on a single axes -- shows how
    much of the probe's ROC comes from noise we inject at that layer, versus
    the nonlinear encoding alone (multiplier 0), holding the rest of the
    model at its normal training-time noise regime."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for eval_mult in NOISE_GRID_EVAL_MULTS:
        r = noise_roc[eval_mult]
        ax.plot(
            r["fpr"],
            r["tpr"],
            label=f"eval mult={eval_mult:g} (AUROC {r['auroc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Probe ROC vs. injected noise at layer {NOISE_GRID_LAYER} ")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_roc_noise_grid.png", close=not show)


def _plot_probe_pca(lyr, pi, plot_dir, tag, show=False):
    """Same content as probe_lib.plot_probe_pca, minus the tag in the title."""
    from sklearn.decomposition import PCA

    X_test, y_test = pi["X_te"], pi["y_te"]
    dom = LinearBoundary(pi["w_dom"], -pi["midpoint"])
    probe = LinearBoundary(pi["w_probe"], pi["b_probe"])
    pca_xy = PCA(n_components=2).fit_transform(X_test)

    def _resid_pc1(w):
        w_hat = w / np.linalg.norm(w)
        X_resid = X_test - np.outer(X_test @ w_hat, w_hat)
        pc1 = PCA(n_components=1).fit_transform(X_resid)[:, 0]
        return w_hat, pc1

    w_dom_hat, pc1_resid_dom = _resid_pc1(dom.w)
    w_logreg_hat, pc1_resid_logreg = _resid_pc1(probe.w)
    proj_dom_raw = X_test @ w_dom_hat
    proj_logreg_raw = X_test @ w_logreg_hat
    dom_threshold = float(-dom.b / np.linalg.norm(dom.w))
    logreg_threshold = float(-probe.b / np.linalg.norm(probe.w))

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0

    fig, (ax_pca, ax_dom_resid, ax_logreg_resid) = plt.subplots(1, 3, figsize=(15, 4.5))

    ax_pca.scatter(pca_xy[lo_mask, 0], pca_xy[lo_mask, 1], s=4, alpha=0.4, label="c=1")
    ax_pca.scatter(pca_xy[hi_mask, 0], pca_xy[hi_mask, 1], s=4, alpha=0.4, label="c=2")
    ax_pca.set_title("PCA (top 2 components)")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend(fontsize=8)
    ax_pca.grid(True, alpha=0.3)
    ax_pca.set_aspect("equal", adjustable="datalim")

    for ax, proj_raw, pc1_resid, threshold, title, xlabel in (
        (
            ax_dom_resid,
            proj_dom_raw,
            pc1_resid_dom,
            dom_threshold,
            "DoM vs residual PCA",
            "DoM projection (data coords)",
        ),
        (
            ax_logreg_resid,
            proj_logreg_raw,
            pc1_resid_logreg,
            logreg_threshold,
            "logreg vs residual PCA",
            "logreg projection (data coords)",
        ),
    ):
        ax.scatter(proj_raw[lo_mask], pc1_resid[lo_mask], s=4, alpha=0.4, label="c=1")
        ax.scatter(proj_raw[hi_mask], pc1_resid[hi_mask], s=4, alpha=0.4, label="c=2")
        ax.axvline(threshold, color="k", ls="--", lw=1, label="threshold")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PC1 of orthogonal residual")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Probe separation, PCA (layer {lyr})")
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_L{lyr}_probe_pca.png", close=not show)


def _plot_pca_scatter(lyr, pi, plot_dir, tag, show=False):
    """Just the top-2-component PCA scatter from `_plot_probe_pca`, without
    the DoM/logreg residual subplots."""
    from sklearn.decomposition import PCA

    X_test, y_test = pi["X_te"], pi["y_te"]
    pca_xy = PCA(n_components=2).fit_transform(X_test)

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0

    fig, ax_pca = plt.subplots(figsize=(5, 4.5))

    ax_pca.scatter(pca_xy[lo_mask, 0], pca_xy[lo_mask, 1], s=4, alpha=0.4, label="c=1")
    ax_pca.scatter(pca_xy[hi_mask, 0], pca_xy[hi_mask, 1], s=4, alpha=0.4, label="c=2")
    ax_pca.set_title(f"PCA (top 2 components, layer {lyr})")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend(fontsize=8)
    ax_pca.grid(True, alpha=0.3)
    ax_pca.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_L{lyr}_pca.png", close=not show)


def _plot_probe_hist_roc(lyr, pi, plot_dir, tag, show=False):
    """Same content as probe_lib.plot_probe, minus the tag in the title."""
    X_test, y_test = pi["X_te"], pi["y_te"]
    dom = LinearBoundary(pi["w_dom"], -pi["midpoint"])
    probe = LinearBoundary(pi["w_probe"], pi["b_probe"])

    proj_dom = dom.score(X_test)
    proj_logreg = probe.score(X_test)
    auroc_dom = roc_auc_score(y_test, proj_dom)
    auroc_logreg = roc_auc_score(y_test, proj_logreg)
    fpr_dom, tpr_dom, _ = roc_curve(y_test, proj_dom)
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, proj_logreg)

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0

    fig, (ax_dom, ax_logreg, ax_roc) = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, proj, title in (
        (ax_dom, proj_dom, "DoM projection"),
        (ax_logreg, proj_logreg, "logreg decision function"),
    ):
        ax.hist(proj[lo_mask], bins=60, alpha=0.6, label="c=1")
        ax.hist(proj[hi_mask], bins=60, alpha=0.6, label="c=2")
        p5, p95 = np.percentile(proj, [5, 95])
        pad = (p95 - p5) * 0.5
        ax.set_xlim([p5 - pad, p95 + pad])
        ax.axvline(0.0, color="k", ls="--", lw=1, label="threshold")
        ax.set_title(title)
        ax.set_xlabel("projection (test set)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_roc.plot(fpr_dom, tpr_dom, label="DoM")
    ax_roc.plot(fpr_logreg, tpr_logreg, label="logreg")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"AUROC: DoM {auroc_dom:.3f}; LogReg {auroc_logreg:.3f}")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_aspect("equal", adjustable="box")

    fig.suptitle(f"Probe separation (layer {lyr})")
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_L{lyr}_probe.png", close=not show)


@torch.no_grad()
def _plot_steer_comparison(
    steer_layer,
    num_x,
    model,
    w_dom,
    w_probe,
    plot_dir,
    tag,
    device,
    noise: Noise,
    filename: str,
    show=False,
):
    """Same content as adversarial_report._plot_steer_comparison, minus the
    tag in the title."""
    w_dom_vec, w_logreg_vec = _steer_vectors(w_dom, w_probe, 1.0)
    dtype = next(model.parameters()).dtype
    vecs = {
        "DoM": torch.tensor(w_dom_vec, dtype=dtype, device=device),
        "logreg": torch.tensor(w_logreg_vec, dtype=dtype, device=device),
    }
    xs = torch.linspace(-3, 3, 400, device=device)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (label, vec) in zip(axes, vecs.items()):
        for j in range(num_x):
            x = torch.zeros(len(xs), num_x, device=device)
            x[:, j] = xs
            x_full = torch.cat([x, torch.full((len(xs), 1), 1.0, device=device)], dim=1)
            y = forward_steered(model, x_full, steer_layer, vec, noise=noise)[:, j]
            ax.plot(
                xs.cpu().numpy(), y.cpu().numpy(), color="blue", alpha=0.3, zorder=5
            )
        for t, (ls, lbl) in {
            1.0: ("k--", "target sat(x,1)"),
            2.0: ("r--", "target sat(x,2)"),
        }.items():
            ax.plot(
                xs.cpu().numpy(),
                torch.clamp(xs, -t, t).cpu().numpy(),
                ls,
                lw=1.5,
                label=lbl,
                zorder=2,
            )
        ax.set_title(f"{label} direction")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("y (c=1 input, steered toward c=2)")
    axes[0].set_ylim([-2.8, 2.8])
    fig.suptitle(
        f"Steering effectiveness, DoM vs logreg direction (layer {steer_layer})"
    )
    fig.tight_layout()
    return save_plot(fig, plot_dir, filename=filename, close=not show)


@torch.no_grad()
def _plot_steer_direction_magnitude(
    steer_layer,
    num_x,
    model,
    w_dom,
    plot_dir,
    tag,
    device,
    noise: Noise,
    filename: str,
    mags: list[float],
    show=False,
):
    """DoM-direction steering, forward (c=1 -> c=2) vs reverse (c=2 -> c=1),
    at each multiple of the DoM shift magnitude in `mags`. Complements
    _plot_steer_comparison (which fixes magnitude at 1x and instead compares
    DoM vs logreg direction): here direction is the fixed axis (one panel
    per start point) and magnitude is the swept variable."""
    dtype = next(model.parameters()).dtype
    w_dom_t = torch.tensor(w_dom, dtype=dtype, device=device)
    xs = torch.linspace(-3, 3, 400, device=device)
    panels = [
        ("forward", 1, 2),  # start c=1, steer toward c=2
        ("reverse", 2, 1),  # start c=2, steer toward c=1
    ]
    colors = ["blue", "green", "purple", "orange"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (label, c_start, c_end) in zip(axes, panels):
        sign = c_end - c_start
        for mag, color in zip(mags, colors):
            vec = sign * mag * w_dom_t
            for j in range(num_x):
                x = torch.zeros(len(xs), num_x, device=device)
                x[:, j] = xs
                x_full = torch.cat(
                    [x, torch.full((len(xs), 1), c_start, device=device)], dim=1
                )
                y = forward_steered(model, x_full, steer_layer, vec, noise=noise)[:, j]
                ax.plot(
                    xs.cpu().numpy(),
                    y.cpu().numpy(),
                    color=color,
                    alpha=0.3,
                    zorder=5,
                    label=f"{mag:g}x" if j == 0 else None,
                )

        for t, (tls, lbl) in {
            c_start: ("k--", f"start sat(x,{c_start})"),
            c_end: ("r--", f"target sat(x,{c_end})"),
        }.items():
            ax.plot(
                xs.cpu().numpy(),
                torch.clamp(xs, -t, t).cpu().numpy(),
                tls,
                lw=1.5,
                label=lbl,
                zorder=2,
            )
        ax.set_title(f"steer c={c_start:g} to c={c_end:g}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("y")
    axes[0].set_ylim([-2.8, 2.8])
    fig.suptitle(f"DoM steering, forward and reverse (layer {steer_layer})")
    fig.tight_layout()
    return save_plot(fig, plot_dir, filename=filename, close=not show)


def _plot_linear_y_reconstruction(r2, plot_dir, tag, show=False):
    layers = sorted(r2)
    baseline = r2[0]
    rel_r2 = {l: (r2[l] - baseline) / (1 - baseline + 1e-9) for l in layers}
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(max(3.5, 0.65 * len(layers)), 4.2))
    ax.bar(x, [rel_r2[l] for l in layers], color="steelblue")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="exact reconstruction of true y")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("R² relative to L0 (embedding) baseline")
    ax.set_title("Linear reconstruction of true y = sat(x, c), per layer")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_linear_y.png", close=not show)


def _make_plots(model, data: PublishData, plot_dir, tag, device, show=False):
    steer_mags = STEER_MAGS_BY_TAG.get(tag, DEFAULT_STEER_MAGS)
    _plot_learned_curves(
        model,
        data.task_loss,
        plot_dir,
        tag,
        noise=data.noise * PLOT_LEARNED_CURVES_NOISE_MULT,
        filename=f"{tag}_curves.png",
        show=show,
    )
    _plot_learned_curves(
        model,
        data.task_loss,
        plot_dir,
        tag,
        noise=0.0,
        filename=f"{tag}_curves_noisefree.png",
        show=show,
    )
    _plot_error_histogram(data.err_samples, plot_dir, tag, show=show)
    _plot_auroc_line(data.auroc, data.hidden_layers, plot_dir, tag, show=show)
    _plot_roc_noise_grid(data.noise_roc, plot_dir, tag, show=show)
    for lyr in data.hidden_layers:
        _plot_pca_scatter(lyr, data.gap_plot_inputs[lyr], plot_dir, tag, show=show)
        _plot_probe_hist_roc(lyr, data.gap_plot_inputs[lyr], plot_dir, tag, show=show)
    _plot_linear_y_reconstruction(data.linear_y_r2, plot_dir, tag, show=show)
    for lyr in STEER_LAYERS:
        assert lyr in data.hidden_layers, (
            f"steer layer {lyr} must be one of the hidden layers "
            f"probed above: {data.hidden_layers}"
        )
        pi = data.gap_plot_inputs[lyr]
        _plot_steer_comparison(
            lyr,
            model.num_x,
            model,
            pi["w_dom"],
            pi["w_probe"],
            plot_dir,
            tag,
            device,
            noise=data.noise * PLOT_STEER_NOISE_MULT,
            filename=f"{tag}_L{lyr}_steer_cmp.png",
            show=show,
        )
        _plot_steer_direction_magnitude(
            lyr,
            model.num_x,
            model,
            pi["w_dom"],
            plot_dir,
            tag,
            device,
            noise=data.noise * PLOT_STEER_NOISE_MULT,
            filename=f"{tag}_L{lyr}_steer_dir_mag.png",
            mags=steer_mags,
            show=show,
        )
        _plot_steer_comparison(
            lyr,
            model.num_x,
            model,
            pi["w_dom"],
            pi["w_probe"],
            plot_dir,
            tag,
            device,
            noise=0.0,
            filename=f"{tag}_L{lyr}_steer_cmp_noisefree.png",
            show=show,
        )
        _plot_steer_direction_magnitude(
            lyr,
            model.num_x,
            model,
            pi["w_dom"],
            plot_dir,
            tag,
            device,
            noise=0.0,
            filename=f"{tag}_L{lyr}_steer_dir_mag_noisefree.png",
            mags=steer_mags,
            show=show,
        )

    if show:
        plt.show()
        plt.close("all")


# ----------------------------------------------------------------------------
def _run_one(args, tag, ckpt, device):
    model, ck = load_model(tag, ckpt, device)
    adv_cfg = resolve_adv_config(ck)
    assert adv_cfg is not None, (
        f"{tag}/{ckpt} has no adv_config (c-blind demonstration run) -- "
        "nothing to probe or steer."
    )
    plot_dir = os.path.join(PUBLISH_DIR, tag, ckpt)
    os.makedirs(plot_dir, exist_ok=True)

    g = torch.Generator(device=device).manual_seed(args.seed)
    probe_backend_name = resolve_probe_backend(args.probe_backend, device)

    data = _run_analysis(model, adv_cfg, args, g, device, probe_backend_name)
    _make_plots(model, data, plot_dir, tag, device, show=args.show)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = ["last", "best"] if args.ckpt == "both" else [args.ckpt]
    for tag in args.tag or TAGS:
        for ckpt in ckpts:
            _run_one(args, tag, ckpt, device)


if __name__ == "__main__":
    main(args)
