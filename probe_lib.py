"""Shared probing primitives: capture residual-stream activations from a
checkpoint, fit/compare linear decision boundaries over them, and plot the
result.

Used as a library by adversarial_report.py (the probing harness) and
analytic.py (activation capture for hand-built exact models).
"""

import os
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float

from model import ResidualMLP
from paths import ckpt_dir


class LinearBoundary(NamedTuple):
    """An affine decision boundary over probed features: score(x) = w . x + b,
    boundary at score = 0. Used to compare probes (e.g. difference-of-means vs
    a fit logreg/LDA probe) that share this representation regardless of how
    each was derived."""

    w: Float[np.ndarray, "d"]
    b: float

    def score(self, X: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n"]:
        return X @ self.w + self.b


def _boundary_accuracy(
    b: LinearBoundary, X: Float[np.ndarray, "n d"], y: Bool[np.ndarray, "n"]
) -> float:
    """Fraction of (X, y) classified correctly by `b.score(X) > 0`."""
    return float(((b.score(X) > 0) == y).mean())


def load_model(tag: str, ckpt: str, device: str) -> tuple[ResidualMLP, dict]:
    """Returns (model, full checkpoint dict) -- the dict carries any non-
    architecture fields (opt state, iter, adversarial-run metadata, ...) that
    rode along in the checkpoint; architecture lives on model.config."""
    path = os.path.join(ckpt_dir(tag), f"{ckpt}.pt")
    model, ck = ResidualMLP.load(path, map_location=device)
    model = model.to(device)
    model.eval()
    return model, ck


@torch.no_grad()
def capture_layers_dict(
    model: ResidualMLP,
    x_full: torch.Tensor,
    layers: list[int],
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> dict[int, torch.Tensor]:
    """Returns each requested layer's residual-stream activations, keyed by
    layer index, from a single shared forward pass. `noise_std`/`generator`
    are passed straight through to `ResidualMLP.forward` (as its `noise`/
    `generator` params)."""
    _, caches = model.forward(
        x_full, return_cache=True, noise=noise_std, generator=generator
    )
    return {i: caches[i] for i in layers}


@torch.no_grad()
def capture_layers(
    model: ResidualMLP,
    x_full: torch.Tensor,
    layers: list[int],
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Like capture_layers_dict, but concatenates the requested layers into a
    single flat feature tensor."""
    d = capture_layers_dict(
        model, x_full, layers, noise_std=noise_std, generator=generator
    )
    return torch.cat([d[i] for i in layers], dim=-1)


@torch.no_grad()
def forward_steered(
    model: ResidualMLP,
    x_full: torch.Tensor,
    steer_layer: int,
    steer_vec: torch.Tensor | None,
) -> torch.Tensor:
    """Manual replay of ResidualMLP.forward, injecting steer_vec into the
    residual stream at index steer_layer (0=embedding, i=after block i-1)."""
    r = x_full @ model.W_E
    if steer_layer == 0 and steer_vec is not None:
        r = r + steer_vec
    for i, block in enumerate(model.blocks):
        r = r + block(r)
        if (i + 1) == steer_layer and steer_vec is not None:
            r = r + steer_vec
    y = r @ model.W_U
    return y[:, : model.num_x]


def save_plot(fig, plot_dir: str, filename: str, close: bool = True) -> str:
    """Save a finished figure and log its path. Closes it by default; pass
    close=False to leave it open for a later plt.show() (only the plots
    adversarial_report.main wants --show to actually pop should do this)."""
    path = os.path.join(plot_dir, filename)
    fig.savefig(path, dpi=120)
    if close:
        plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_probe(
    tag,
    layers,
    dom: LinearBoundary,
    probe: LinearBoundary,
    X_test: Float[np.ndarray, "n d"],
    y_test: Bool[np.ndarray, "n"],
    plot_dir,
):
    """n = test-set size (both classes concatenated), d = probed feature dim
    (sum of d_model over the probed layers). `dom` and `probe` are two affine
    decision boundaries over the same feature space -- a difference-of-means
    direction and a fit logreg/LDA probe -- compared side by side. `probe`'s
    weight/bias are raw (unstandardized): whatever scaler the probe was fit
    with already folded in, so this function is agnostic to which backend
    (sklearn/torch) produced them.

    Writes a histogram+ROC comparison ({tag}_L{layers}_probe.png). See also
    `plot_probe_pca` for a companion PCA-based comparison over the same
    inputs."""
    from sklearn.metrics import roc_auc_score, roc_curve

    proj_dom = dom.score(X_test)
    proj_logreg = probe.score(X_test)

    auroc_dom = roc_auc_score(y_test, proj_dom)
    auroc_logreg = roc_auc_score(y_test, proj_logreg)
    fpr_dom, tpr_dom, _ = roc_curve(y_test, proj_dom)
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, proj_logreg)

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0
    layer_str = "-".join(str(i) for i in layers)

    # --- figure 1: histograms + ROC ---
    fig, (ax_dom, ax_logreg, ax_roc) = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, proj, title in (
        (ax_dom, proj_dom, "DoM projection"),
        (ax_logreg, proj_logreg, "logreg decision function"),
    ):
        ax.hist(proj[lo_mask], bins=60, alpha=0.6, label="c=1")
        ax.hist(proj[hi_mask], bins=60, alpha=0.6, label="c=2")

        # Set plot limit to avoid outliers
        percentile_5, percentile_95 = np.percentile(proj, [5, 95])
        percentile_diff = percentile_95 - percentile_5
        ax.set_xlim(
            [
                percentile_5 - percentile_diff * 0.5,
                percentile_95 + percentile_diff * 0.5,
            ]
        )

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
    ax_roc.set_title(f"AUROC: DoM: {auroc_dom:.3f}; LogReg: {auroc_logreg:.3f}")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_aspect("equal", adjustable="box")

    fig.suptitle(f"probe separation ({tag}, layers={layer_str})")
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_L{layer_str}_probe.png")


def plot_probe_pca(
    tag,
    layers,
    dom: LinearBoundary,
    probe: LinearBoundary,
    X_test: Float[np.ndarray, "n d"],
    y_test: Bool[np.ndarray, "n"],
    plot_dir,
):
    """Companion to `plot_probe`, over the same inputs: writes a PCA
    comparison ({tag}_L{layers}_probe_pca.png) -- each probe's own direction
    projected out and PC1 of what's left plotted against a shared
    top-2-component PCA, showing whether any orthogonal variance still
    separates the classes. Split out since PCA is the expensive part and
    only wanted for a detailed report."""
    from sklearn.decomposition import PCA

    pca_xy = PCA(n_components=2).fit_transform(X_test)

    def _resid_pc1(w: Float[np.ndarray, "d"]):
        w_hat = w / np.linalg.norm(w)
        X_resid = X_test - np.outer(X_test @ w_hat, w_hat)
        pc1 = PCA(n_components=1).fit_transform(X_resid)[:, 0]
        return w_hat, pc1

    # each probe's own direction, projected out, then PC1 of what's left --
    # raw projections are in data coordinates (unlike a decision-score
    # projection, which is in each probe's own units).
    w_dom_hat, pc1_resid_dom = _resid_pc1(dom.w)
    w_logreg_hat, pc1_resid_logreg = _resid_pc1(probe.w)
    proj_dom_raw = X_test @ w_dom_hat
    proj_logreg_raw = X_test @ w_logreg_hat
    dom_raw_threshold = float(-dom.b / np.linalg.norm(dom.w))
    logreg_raw_threshold = float(-probe.b / np.linalg.norm(probe.w))

    lo_mask = y_test == 0.0
    hi_mask = y_test == 1.0
    layer_str = "-".join(str(i) for i in layers)

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
            dom_raw_threshold,
            "DoM vs residual PCA",
            "DoM projection (data coords)",
        ),
        (
            ax_logreg_resid,
            proj_logreg_raw,
            pc1_resid_logreg,
            logreg_raw_threshold,
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

    fig.suptitle(f"probe separation, PCA ({tag}, layers={layer_str})")
    fig.tight_layout()
    return save_plot(fig, plot_dir, f"{tag}_L{layer_str}_probe_pca.png")
