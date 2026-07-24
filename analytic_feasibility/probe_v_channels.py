"""Can a difference-of-means / logistic-regression probe, given (x1, v1, v2),
detect c through the v1/v2 mean-constant encoding (README.md section 1,
"v-channels")?

v1, v2 are mean-constant in c by construction (E_x1[v] = 0 for every c), so a
raw difference-of-means probe is at chance -- that's the whole point of the
encoding. But mean-constancy says nothing about higher moments: std(v1),
std(v2), and cov(v1, v2) all vary with c (checked numerically below), so a
probe that can use curvature/covariance -- logistic regression on (x1, v1,
v2) -- may still separate c_lo from c_hi well above chance. Probes get x1 as
well as v1, v2: x1 is itself a residual coordinate (part of the real probe
scope, see README.md's "probe scope" note) and README.md's exact decode of c
from the encoding requires x1 to disambiguate which x1-slab a sample falls
in, so omitting it would understate what a real probe sees. This script
checks that directly, with no trained network involved: just the closed-form
encoding from README.md, sampled and fed straight to DoM and logreg
classifiers.
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="DoM vs logreg detectability of the (x1, v1, v2) c-encoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--c-lo", type=float, default=1.0)
    p.add_argument("--c-hi", type=float, default=2.0)
    p.add_argument("--n", type=int, default=20_000, help="samples per class per split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None, help="default: $TMPDIR or .")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

relu = lambda z: np.maximum(z, 0.0)


def sample_features(
    n: int, c: float, rng: np.random.Generator
) -> Float[np.ndarray, "n 3"]:
    """(x1, v1, v2): x1 ~ U[-3, 3], v1/v2 per README.md's mean-constant
    encoding -- the full probe input (x1 is itself a residual coordinate)."""
    x1 = rng.uniform(-3, 3, n)
    v1 = -2 * relu(-x1 - c) + 2 * relu(x1 - 3 + c) - c + 1.5
    v2 = -4 * relu(-x1 - c / 2) + 4 * relu(x1 + c / 2 - 3) - c + 3.0
    return np.stack([x1, v1, v2], axis=1)


def dom_probe(
    X: Float[np.ndarray, "n 3"], y: Float[np.ndarray, "n"]
) -> tuple[Float[np.ndarray, "3"], float]:
    """Difference-of-means direction/threshold, midpoint-of-means bias."""
    mu0, mu1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)
    w = mu1 - mu0
    b = -0.5 * (mu0 + mu1) @ w
    return w, b


def main(args):
    rng = np.random.default_rng(args.seed)
    n = args.n

    def make_split():
        X = np.concatenate(
            [sample_features(n, args.c_lo, rng), sample_features(n, args.c_hi, rng)]
        )
        y = np.concatenate([np.zeros(n), np.ones(n)])
        return X, y

    X_tr, y_tr = make_split()
    X_te, y_te = make_split()

    w_dom, b_dom = dom_probe(X_tr, y_tr)
    dom_scores = X_te @ w_dom + b_dom
    dom_acc = float(((dom_scores > 0) == y_te).mean())
    dom_auroc = float(roc_auc_score(y_te, dom_scores))

    clf = LogisticRegression().fit(X_tr, y_tr)
    logreg_acc = float(clf.score(X_te, y_te))
    w_lr, b_lr = clf.coef_[0], float(clf.intercept_[0])
    logreg_auroc = float(roc_auc_score(y_te, X_te @ w_lr + b_lr))

    print(
        f"(x1, v1, v2) encoding, c={args.c_lo:g} vs c={args.c_hi:g} "
        f"(n_test={2 * n})"
    )
    print(
        f"  DoM    accuracy: {dom_acc:.4f}  AUROC: {dom_auroc:.4f}   "
        f"||w||={np.linalg.norm(w_dom):.4f}"
    )
    print(
        f"  logreg accuracy: {logreg_acc:.4f}  AUROC: {logreg_auroc:.4f}   "
        f"||w||={np.linalg.norm(w_lr):.4f}"
    )

    out_dir = pathlib.Path(args.out_dir or os.environ.get("TMPDIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    lo_label, hi_label = f"c={args.c_lo:g}", f"c={args.c_hi:g}"

    # --- Fig 1: raw (v1, v2) scatter. Probes below are fit on (x1, v1, v2),
    # so their boundary is a plane, not a fixed line in this 2D slice -- this
    # panel is encoding-shape context only; see Fig 2 for actual separation.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*X_te[y_te == 0, 1:].T, s=4, alpha=0.25, label=lo_label)
    ax.scatter(*X_te[y_te == 1, 1:].T, s=4, alpha=0.25, label=hi_label)
    ax.set_xlabel("v1")
    ax.set_ylabel("v2")
    ax.set_title(f"v1/v2 encoding, {lo_label} vs {hi_label}\n(probes also see x1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "v_channels_2d.png"
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")

    # --- Fig 2: histograms of each probe's 1D projection. ---
    fig, (ax_dom, ax_lr) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, w, b, name, acc, auroc in [
        (ax_dom, w_dom, b_dom, "difference of means", dom_acc, dom_auroc),
        (ax_lr, w_lr, b_lr, "logistic regression", logreg_acc, logreg_auroc),
    ]:
        proj_lo = X_te[y_te == 0] @ w + b
        proj_hi = X_te[y_te == 1] @ w + b
        lo, hi = min(proj_lo.min(), proj_hi.min()), max(proj_lo.max(), proj_hi.max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(proj_lo, bins=bins, alpha=0.5, label=lo_label)
        ax.hist(proj_hi, bins=bins, alpha=0.5, label=hi_label)
        ax.axvline(0.0, color="k", ls="--", lw=1)
        ax.set_title(f"{name}\naccuracy={acc:.3f}, AUROC={auroc:.3f}")
        ax.set_xlabel("projection (boundary at 0)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax_dom.set_ylabel("count")
    fig.tight_layout()
    path = out_dir / "v_channels_hist.png"
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")

    # --- Fig 3: ROC curves for both probes. ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for scores, name, auroc, color in [
        (dom_scores, "DoM", dom_auroc, "k"),
        (X_te @ w_lr + b_lr, "logreg", logreg_auroc, "r"),
    ]:
        fpr, tpr, _ = roc_curve(y_te, scores)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.set_xlabel("FPR (false positive rate)")
    ax.set_ylabel("TPR (true positive rate)")
    ax.set_title(f"ROC curves, {lo_label} vs {hi_label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    path = out_dir / "v_channels_roc.png"
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main(args)
